/*
 * MGLRenderer+Geometry.m — Metal side of the 'geom' domain.
 *
 * Runs one draw of a program with a geometry shader as
 *   1. a rasterization-disabled render pass over the *vertex* stage in its capture
 *      variant (see geometry.c), which stores the VS outputs per (vertex, instance)
 *      into a buffer,
 *   2. a compute pass running the geometry-shader kernel, which appends output vertices
 *      and a raster index list and fills in a DrawIndexedPrimitives indirect argument
 *      buffer from its atomic counters (no CPU readback), and
 *   3. the ordinary raster pass with the generated pass-through vertex function.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_geom.h"
#include "mgl_ext_buffers.h"
#include "mgl_ext_program.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom_xfb.h"
#include "MGLRenderer_Private.h"

/* rstate's vertex/fragment function-constant specialization is declared privately inside
 * MGLRenderer+RenderState.m; the geometry pass-through function needs the vertex half of it
 * (it is the last pre-rasterization stage), so the prototype is repeated here rather than
 * editing the rstate category's header. */
@interface MGLRenderer (RenderStateSpecialize)
- (id<MTLFunction>) rstateSpecialize:(Shader *)shader stage:(int)stage;
@end

/* the raster pass is a separate method so the pass-through vertex function can be swapped
 * in for it and restored afterwards */
@interface MGLRenderer (Geometry)
- (bool) geomDraw: (MGLGeomDraw *) d;
- (bool) geomRasterPass: (MGLGeomDraw *) d;
- (bool) geomCanAssembleGPU;
- (void) geomXfbWait;
@end

/* private renderer methods used here (declared for the compiler; implemented by the
 * buffers domain category) */
#define GEOM()   ((struct MGLExtState_geom *)ctx->state.ext_geom)
#define BUF()    (ctx->mtl_funcs.ext_buffers)

/* per-process scratch (the buffers are shared storage; they are only used by encoders of
 * the command buffer they were written for, so a single set can serve every draw) */
/* a per-draw scratch pool: the buffer plus the next free offset (geomSubAlloc below) */
typedef struct {
    id<MTLBuffer> buf;
    size_t next;
} MGLGeomPool;

static id<MTLBuffer> gPrim;         /* header + capture index list + input primitive slots */
static id<MTLBuffer> gVsout;        /* capture header + captured VS records */
static MGLGeomPool gPrimPool, gVsoutPool; static size_t gPrimOff, gVsoutOff;
static id<MTLBuffer> gOutv;         /* geometry output vertex records */
static id<MTLBuffer> gIdx;          /* raster index list */
static void *gEnqueuedCb;            /* the command buffer this category enqueued */
static id<MTLBuffer> gCtl;          /* counters + indirect draw arguments */
static id<MTLBuffer> gPrep;         /* prep kernel inputs, staging command and outputs */
static MGLGeomPool gPrepPool; static size_t gPrepOff;
static id<MTLBuffer> gCnt;          /* per-thread emitted vertex counts (capture order) */
static MGLGeomPool gOutvPool, gCntPool; static size_t gOutvOff, gCntOff;
static id<MTLBuffer> gXbase;        /* compacted transform feedback index per thread */
static id<MTLBuffer> gXfb;          /* pack counters and write offsets (GPU-owned while pending) */
static bool gXfbPending;            /* a pack has not been folded into the object yet */
static id<MTLCommandBuffer> gXfbCb; /* that command buffer, for a deterministic wait */
static id<MTLFence> gXfbFence;     /* orders the packs of one block across command buffers */
static id<MTLCommandBuffer> gXfbFenceCb;   /* the command buffer that last updated it */
static GLuint gXfbOwner;            /* the object the counter block belongs to */
static bool gXfbBegin;              /* the counter block still needs its init kernel */
static size_t gPrimBytes, gVsoutBytes, gOutvBytes, gIdxBytes;

static id<MTLComputePipelineState> gGsPso;
static id<MTLComputePipelineState> gInitPso;
/* tile-parallel prep (large draws): per-tile restart counts -> prefix sum -> per-tile run
 * records -> parallel emit; see the comment block in kGeomPrepMSL */
static id<MTLComputePipelineState> gPrepcPso, gTilesPso, gPrefixPso, gRunsPso, gReducePso, gEmitPso;
static id<MTLBuffer> gPrepScr;       /* its scratch: per-tile counts and per-run records */
static size_t gPrepScrWords;
/* below this index count the serial prep kernel (one dispatch) is cheaper than the chain;
 * overridable so the test suite can exercise the tile-parallel path on every draw */
static uint32_t gTiledMin = 4096u;
/* the scratch is capped so absurdly large indirect draws fall back to the serial kernel
 * instead of asking for hundreds of megabytes */
#define MGL_GEOM_TILED_MAX_WORDS (64u * 1024u * 1024u / 4u)
/* the trace / behaviour switches are read once: this file sits on the draw path and
 * getenv() per draw (several times per draw) is not free */
static struct {
    int  read;
    bool draw, debug, xfb, worst_case, dump, serial_prep, pool_grow;
} gEnv = { 0, false, false, false, false, false, false, false };

static void geomEnvRead(void)
{
    if (gEnv.read) return;
    gEnv.read = 1;
    gEnv.draw = getenv("MGL_GEOM_TRACE_DRAW") != NULL;
    gEnv.debug = getenv("MGL_GEOM_DEBUG") != NULL;
    gEnv.xfb = getenv("MGL_GEOM_XFB_TRACE") != NULL;
    gEnv.worst_case = getenv("MGL_GEOM_WORST_CASE_DISPATCH") != NULL;
    gEnv.dump = getenv("MGL_GEOM_DUMP") != NULL;
    gEnv.serial_prep = getenv("MGL_GEOM_SERIAL_PREP") != NULL;   /* A/B the serial kernel */
    gEnv.pool_grow = getenv("MGL_GEOM_POOL_GROW") != NULL;       /* A/B grow-only scratch pools */
    const char *tmin = getenv("MGL_GEOM_TILED_MIN");             /* test hook: force the chain */
    if (tmin) { int v = atoi(tmin); gTiledMin = v > 0 ? (uint32_t)v : 0u; }
}

static id<MTLComputePipelineState> gPrepPso, gTessImportPso;
static id<MTLComputePipelineState> gXfbPrePso, gXfbScatterPso, gXfbFinPso, gXfbBeginPso;
static id<MTLComputePipelineState> gXfbTessPso, gXfbTessFinPso;
static id<MTLComputePipelineState> gFinPso;
static void *gPsoKey;               /* the kernel function the PSOs were built from */

/* stringise a macro value for the embedded MSL */
#define MGL_GEOM_STR2(x) #x
#define MGL_GEOM_STR(x) MGL_GEOM_STR2(x)

/* the prep block of gPrep, in words: the kernel's inputs (a 16-word struct), the staging
 * words of a direct draw's command, then the kernel's outputs (prep[0..11]: the assembly
 * counts, the geometry kernel's indirect threadgroup count at [4] and the capture pass's
 * indirect draw arguments at [7..10]).  Every user of the block goes through these. */
#define MGL_GEOM_PREP_IN_WORDS  16    /* c.v[0..15] of the prep kernel */
#define MGL_GEOM_PREP_CMD_OFF   16    /* 4 staging words a direct draw's command */
#define MGL_GEOM_PREP_OUT_OFF   32    /* prep[0..11], the kernel's outputs */
#define MGL_GEOM_PREP_GROUPS_OFF (MGL_GEOM_PREP_OUT_OFF + 4)   /* prep[4] */
#define MGL_GEOM_WG             32
#define MGL_GEOM_PREP_BYTES     (64 * sizeof(uint32_t))

/* the prep kernel builds, in one workgroup, everything the rest of the emulation needs from
 * the draw command and the element buffer:
 *   - the capture index list (primitive restart removed)
 *   - the input primitive slot list for the geometry kernel
 *   - the vertex range of the capture pass (min/max)
 *   - the geometry kernel's dispatch size and the capture pass's indirect draw arguments
 *
 * The algorithm follows Mesa's poly_unroll_geometry (poly/cl/restart.h): the workgroup scans
 * the index buffer in tiles, splits at primitive restart and emits the primitives of every
 * restart-free run. */
static const char *kGeomPrepMSL =
    "#include <metal_stdlib>\n"
    "#include <simd/simd.h>\n"
    "using namespace metal;\n"
    "\n"
    "struct mgl_geom_prep_in { uint v[" MGL_GEOM_STR(MGL_GEOM_PREP_IN_WORDS) "]; };\n"
    "\n"
    "#define WG " MGL_GEOM_STR(MGL_GEOM_WG) "\n"
    "\n"
    "kernel void mgl_geom_prep(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                          const device uchar* cmd [[buffer(1)]],\n"
    "                          const device uchar* indices [[buffer(2)]],\n"
    "                          device uint* prim [[buffer(3)]],\n"
    "                          device uint* cap [[buffer(4)]],\n"
    "                          device uint* hdr [[buffer(5)]],\n"
    "                          device uint* ctl [[buffer(6)]],\n"
    "                          device uint* prep [[buffer(7)]],\n"
    "                          uint tid [[thread_index_in_threadgroup]],\n"
    "                          uint slid [[thread_index_in_simdgroup]],\n"
    "                          uint sgid [[simdgroup_index_in_threadgroup]])\n"
    "{\n"
    "    /* in: mode, count, first, indexed, basevertex, instances, invocations, restart,\n"
    "     *     k (input vertices), provoking_first, index size, cmd layout,\n"
    "     *     cap index words, cap prim words, reserved, reserved */\n"
    "    uint mode = c.v[0], count = c.v[1], first = c.v[2], indexed = c.v[3];\n"
    "    uint basevertex = c.v[4], instances = c.v[5], restart = c.v[7];\n"
    "    uint k = c.v[8], provoking_first = c.v[9], idx_size = c.v[10], layout = c.v[11];\n"
    "    uint cap_idx = c.v[12], cap_prim = c.v[13];\n"
    "    bool restart_en = c.v[14] != 0u;\n"
    "    threadgroup uint tgr[WG];\n"
    "\n"
    "    /* the tile-parallel chain can take over large indirect draws (mgl_geom_prepc sets\n"
    "     * the flag after parsing the command): the outputs are identical, so just leave */\n"
    "    if (layout != 0u && prep[18] != 0u) return;\n"
    "\n"
    "    /* the draw command: a staging buffer for direct draws, the user's indirect buffer\n"
    "     * otherwise */\n"
    "    if (layout != 0u) {\n"
    "        const device uint* u = (const device uint*)cmd;\n"
    "        if (layout == 1u) { count = u[0]; instances = u[1]; first = u[2]; }\n"
    "        else              { count = u[0]; instances = u[1]; first = u[2]; basevertex = u[3]; }\n"
    "        if (count == 0u || instances == 0u) { if (tid == 0u) { prep[0] = 0u; prep[4] = 0u; hdr[0] = 0u; } return; }\n"
    "    }\n"
    "\n"
    "    uint vmin = 0xffffffffu, vmax = 0;\n"
    "    uint out_prims = 0, out_idx = 0;\n"
    "\n"
    "    /* the vertex index at position i (arrays derive it, indexed draws load it) */\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[_i] : (idx_size == 2u ? (uint)((const device ushort*)indices)[_i] : ((const device uint*)indices)[_i])) : (uint)(first + (int)(_i)))\n"
    "\n"
    "    uint needle = 0, m;\n"
    "    while (needle < count) {\n"
    "        /* find the next restart (or the end) in tiles of WG */\n"
    "        uint next_restart = needle;\n"
    "        for (;;) {\n"
    "            uint i = next_restart + tid;\n"
    "            bool r = i >= count || (indexed && restart_en && MGL_IDX(i) == restart);\n"
    "            {   /* first restart in this tile */\n"
    "#if WG <= 32\n"
    "                /* one simdgroup covers the whole workgroup: the ballot is the answer, no\n"
    "                 * threadgroup round trip (primitive restart means this loop runs once per\n"
    "                 * restart-free run, so a barrier here would dominate the kernel) */\n"
    "                uint bb = (uint)(uint64_t)simd_ballot(r);\n"
    "                m = bb ? ctz(bb) : 0xffffffffu;\n"
    "#else\n"
    "                uint bb = (uint)(uint64_t)simd_ballot(r);\n"
    "                uint local = bb ? (ctz(bb) + sgid * 32u) : 0xffffffffu;\n"
    "                if (slid == 0u) tgr[sgid] = local;\n"
    "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "                m = 0xffffffffu;\n"
    "                for (uint j = 0; j < WG / 32u; j++) if (tgr[j] < m) m = tgr[j];\n"
    "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "#endif\n"
    "            }\n"
    "            if (m != 0xffffffffu) { next_restart += m; break; }\n"
    "            next_restart += WG;\n"
    "        }\n"
    "\n"
    "        uint subcount = next_restart - needle;\n"
    "        uint subprims = 0u;\n"
    "        switch (mode) {\n"
    "            case 0u: subprims = subcount; break;                                   /* points */\n"
    "            case 1u: subprims = subcount / 2u; break;                              /* lines */\n"
    "            case 2u: subprims = subcount >= 2u ? subcount - 1u : 0u; break;         /* line strip */\n"
    "            case 3u: subprims = subcount >= 2u ? subcount : 0u; break;              /* line loop */\n"
    "            case 4u: subprims = subcount / 4u; break;                               /* lines adjacency */\n"
    "            case 5u: subprims = subcount >= 4u ? subcount - 3u : 0u; break;         /* line strip adj */\n"
    "            case 6u: subprims = subcount / 3u; break;                               /* triangles */\n"
    "            case 7u: subprims = subcount >= 3u ? subcount - 2u : 0u; break;         /* triangle strip */\n"
    "            case 8u: subprims = subcount >= 3u ? subcount - 2u : 0u; break;         /* triangle fan */\n"
    "            case 9u: subprims = subcount / 6u; break;                               /* triangles adj */\n"
    "            case 10u: subprims = subcount >= 6u ? (subcount - 4u) / 2u : 0u; break; /* tri strip adj */\n"
    "            default: subprims = 0u; break;\n"
    "        }\n"
    "\n"
    "        for (uint p = tid; p < subprims; p += WG) {\n"
    "            for (uint v = 0u; v < k; v++) {\n"
    "                uint id;\n"
    "                switch (mode) {\n"
    "                    case 0u: id = p; break;                                  /* points */\n"
    "                    case 1u: case 2u: case 4u: case 5u: id = p + v; break;   /* line-ish */\n"
    "                    case 3u: id = (p + v < subcount) ? p + v : 0u; break;    /* line loop */\n"
    "                    case 6u: case 9u: id = p * k + v; break;                 /* tri list / tri adj */\n"
    "                    case 10u: id = p * 2u + v; break;                        /* tri strip adj */\n"
    "                    case 8u:                                              /* triangle fan */\n"
    "                        if (provoking_first) id = (v == 2u) ? 0u : p + v + 1u;\n"
    "                        else                  id = (v == 0u) ? 0u : p + v;\n"
    "                        break;\n"
    "                    default: {                                             /* triangle strip */\n"
    "                        if ((p & 1u) == 0u) id = p + v;\n"
    "                        else if (provoking_first) id = (v == 0u) ? p + 2u : p + v - 1u;\n"
    "                        else                     id = (v == 0u) ? p + 1u : (v == 1u ? p : p + 2u);\n"
    "                        break;\n"
    "                    }\n"
    "                }\n"
    "                uint vtx = MGL_IDX(needle + id);\n"
    "                if (out_prims + p < cap_prim)\n"
    "                    prim[(out_prims + p) * k + v] = vtx + basevertex;\n"
    "            }\n"
    "        }\n"
    "        /* every vertex of the run is captured, not only the ones the primitives use */\n"
    "        for (uint i = tid; i < subcount; i += WG) {\n"
    "            uint vtx = MGL_IDX(needle + i);\n"
    "            if (out_idx + i < cap_idx) cap[out_idx + i] = vtx;\n"
    "            if (vtx < vmin) vmin = vtx;\n"
    "            if (vtx > vmax) vmax = vtx;\n"
    "        }\n"
    "\n"
    "        out_prims += subprims;\n"
    "        out_idx += subcount;\n"
    "        needle = next_restart + 1u;\n"
    "    }\n"
    "#undef MGL_IDX\n"
    "\n"
    "    /* reduce the vertex range across the workgroup (only thread 0 writes the results) */\n"
    "    tgr[tid] = vmin;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid == 0u) { for (uint j = 1u; j < WG; j++) if (tgr[j] < vmin) vmin = tgr[j]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    tgr[tid] = vmax;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid == 0u) { for (uint j = 1u; j < WG; j++) if (tgr[j] > vmax) vmax = tgr[j]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid != 0u) return;\n"
    "\n"
    "    if (vmin == 0xffffffffu) { vmin = 0u; vmax = 0u; }\n"
    "    prep[0] = out_prims;\n"
    "    prep[1] = out_idx;\n"
    "    prep[2] = vmax;\n"
    "    prep[3] = vmin;\n"
    "    prep[4] = (uint)(((uint64_t)out_prims * instances * c.v[6]) / WG + 1u);   /* threadgroups */\n"
    "    prep[5] = 1u;\n"
    "    prep[6] = 1u;\n"
    "    prep[7] = out_idx;                     /* capture indexCount    */\n"
    "    prep[8] = instances;                   /* capture instanceCount */\n"
    "    prep[9] = 0u;                          /* capture startIndex    */\n"
    "    prep[10] = indexed ? basevertex : 0u;  /* capture baseVertex    */\n"
    "    prep[11] = 0u;                         /* capture baseInstance  */\n"
    "    hdr[0] = out_prims;\n"
    "    hdr[1] = instances;\n"
    "    hdr[6] = vmax - vmin + 1u;\n"
    "    hdr[12] = vmin;\n"
    "    ctl[0] = vmin + (indexed ? basevertex : 0u);   /* capture header: vstart */\n"
    "    ctl[1] = vmax - vmin + 1u;                     /* capture header: vcount */\n"
    "}\n"
    "\n"
    "/* ---- tile-parallel variant of the same job, for large draws ----------------------\n"
    " * The serial kernel walks restart-free runs one at a time; with a restart every few\n"
    " * indices that is a dependency chain per run and a single workgroup cannot keep the GPU\n"
    " * busy.  This chain splits the index buffer into WG-sized tiles:\n"
    " *   mgl_geom_tiles   one workgroup per tile: restart count, index min/max, last restart\n"
    " *   mgl_geom_prefix  exclusive prefix sum of the per-tile restart counts\n"
    " *   mgl_geom_runs    one workgroup per tile: (start, subcount) of every run, the prefix\n"
    " *                    locating a run's start in an earlier tile\n"
    " *   mgl_geom_reduce  per-run prefix sums (capbase, primbase), totals, the same prep,\n"
    " *                    hdr and ctl words the serial kernel writes\n"
    " *   mgl_geom_emit    a binary search per primitive / captured index fills the same slots\n"
    " * prim slot (primbase[r] + p) * k + v and cap slot capbase[r] + i are exactly the\n"
    " * addresses the serial kernel computes while walking runs in order, so the results are\n"
    " * bit-identical.  Requires WG == 32 (one simdgroup per tile).\n"
    " *\n"
    " * Context words in the prep block after the serial outputs (prep[n] = block word\n"
    " * OUT_OFF + n): 12 count, 13 first, 14 basevertex, 15 instances, 16 tiles,\n"
    " * 17 record capacity, 18 owner (0 = the serial kernel owns the draw, 1 = this chain),\n"
    " * 19 total restarts, 21..23 the tile grid, 24..26 the emit grid, 27 run count.\n"
    " * Scratch: nres[0,T) tmin[T,2T) tmax[2T,3T) rbase[3T,4T) lres[4T,5T)\n"
    " * runs[5T, 5T+4*(Rmax+1)) as (start, subcount, capbase, primbase). */\n"
    "static inline uint mgl_geom_subprims(uint mode, uint subcount)\n"
    "{\n"
    "    switch (mode) {\n"
    "        case 0u: return subcount;                                   /* points */\n"
    "        case 1u: return subcount / 2u;                              /* lines */\n"
    "        case 2u: return subcount >= 2u ? subcount - 1u : 0u;        /* line strip */\n"
    "        case 3u: return subcount >= 2u ? subcount : 0u;             /* line loop */\n"
    "        case 4u: return subcount / 4u;                              /* lines adjacency */\n"
    "        case 5u: return subcount >= 4u ? subcount - 3u : 0u;        /* line strip adj */\n"
    "        case 6u: return subcount / 3u;                              /* triangles */\n"
    "        case 7u: case 8u: return subcount >= 3u ? subcount - 2u : 0u;  /* strip / fan */\n"
    "        case 9u: return subcount / 6u;                              /* triangles adj */\n"
    "        case 10u: return subcount >= 6u ? (subcount - 4u) / 2u : 0u;   /* tri strip adj */\n"
    "        default: return 0u;\n"
    "    }\n"
    "}\n"
    "\n"
    "static inline uint mgl_geom_vert_id(uint mode, uint p, uint v, uint k, uint subcount, uint provoking_first)\n"
    "{\n"
    "    uint id;\n"
    "    switch (mode) {\n"
    "        case 0u: id = p; break;                                   /* points */\n"
    "        case 1u: case 2u: case 4u: case 5u: id = p + v; break;    /* line-ish */\n"
    "        case 3u: id = (p + v < subcount) ? p + v : 0u; break;     /* line loop */\n"
    "        case 6u: case 9u: id = p * k + v; break;                  /* tri list / tri adj */\n"
    "        case 10u: id = p * 2u + v; break;                         /* tri strip adj */\n"
    "        case 8u:                                                 /* triangle fan */\n"
    "            if (provoking_first) id = (v == 2u) ? 0u : p + v + 1u;\n"
    "            else                  id = (v == 0u) ? 0u : p + v;\n"
    "            break;\n"
    "        default:                                                 /* triangle strip */\n"
    "            if ((p & 1u) == 0u) id = p + v;\n"
    "            else if (provoking_first) id = (v == 0u) ? p + 2u : p + v - 1u;\n"
    "            else                     id = (v == 0u) ? p + 1u : (v == 1u ? p : p + 2u);\n"
    "            break;\n"
    "    }\n"
    "    return id;\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_prepc(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                           const device uchar* cmd [[buffer(1)]],\n"
    "                           device uint* prep [[buffer(7)]],\n"
    "                           uint tid [[thread_index_in_threadgroup]])\n"
    "{\n"
    "    /* indirect draws only: normalise the command into the context words the\n"
    "     * tile-parallel kernels share, pick the owner (this chain or the serial kernel)\n"
    "     * and publish the tile grid.  c.v[15] is the scratch capacity in words. */\n"
    "    if (tid != 0u) return;\n"
    "    uint count = c.v[1], first = c.v[2], basevertex = c.v[4], instances = c.v[5], layout = c.v[11];\n"
    "    const device uint* u = (const device uint*)cmd;\n"
    "    if (layout == 1u) { count = u[0]; instances = u[1]; first = u[2]; }\n"
    "    else if (layout == 2u) { count = u[0]; instances = u[1]; first = u[2]; basevertex = u[3]; }\n"
    "    uint tiles = (count + WG - 1u) / WG;\n"
    "    uint64_t rmax = (uint64_t)count + 1u;\n"
    "    uint64_t words = 5ull * tiles + 4ull * (rmax + 1ull);\n"
    "    if (count == 0u || instances == 0u || words > (uint64_t)c.v[15]) return;  /* serial keeps it */\n"
    "    prep[12] = count; prep[13] = first; prep[14] = basevertex; prep[15] = instances;\n"
    "    prep[16] = tiles; prep[17] = (uint)rmax; prep[18] = 1u;\n"
    "    prep[21] = tiles; prep[22] = 1u; prep[23] = 1u;\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_tiles(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                           const device uchar* indices [[buffer(2)]],\n"
    "                           device uint* prep [[buffer(7)]],\n"
    "                           device uint* sc [[buffer(8)]],\n"
    "                           uint slid [[thread_index_in_simdgroup]],\n"
    "                           uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (prep[18] == 0u) return;\n"
    "    uint T = prep[16];\n"
    "    uint t = gid;\n"
    "    if (t >= T) return;\n"
    "    uint count = prep[12];\n"
    "    uint indexed = c.v[3], restart = c.v[7], idx_size = c.v[10];\n"
    "    bool restart_en = c.v[14] != 0u;\n"
    "    uint i = t * WG + slid;\n"
    "    bool valid = i < count;\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[_i] : (idx_size == 2u ? (uint)((const device ushort*)indices)[_i] : ((const device uint*)indices)[_i])) : 0u)\n"
    "    uint vtx = valid ? MGL_IDX(i) : 0u;\n"
    "    bool res = valid && indexed && restart_en && vtx == restart;\n"
    "    uint rb = (uint)(uint64_t)simd_ballot(res);\n"
    "    uint nres = popcount(rb);\n"
    "    uint tmin = (!res && valid) ? vtx : 0xffffffffu;\n"
    "    uint tmax = (!res && valid) ? vtx : 0u;\n"
    "    if (slid == 0u) {\n"
    "        sc[t] = nres;\n"
    "        sc[T + t] = simd_min(tmin);\n"
    "        sc[2u*T + t] = simd_max(tmax);\n"
    "    }\n"
    "    uint rank = popcount(rb & ((1u << slid) - 1u));\n"
    "    if (res && rank == nres - 1u) sc[4u*T + t] = i;   /* the tile's last restart */\n"
    "#undef MGL_IDX\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_prefix(device uint* prep [[buffer(7)]],\n"
    "                            device uint* sc [[buffer(8)]],\n"
    "                            uint tid [[thread_index_in_threadgroup]])\n"
    "{\n"
    "    if (prep[18] == 0u) return;\n"
    "    uint T = prep[16];\n"
    "    threadgroup uint part[WG];\n"
    "    uint chunk = (T + WG - 1u) / WG;\n"
    "    uint b = min(tid * chunk, T), e = min(b + chunk, T);\n"
    "    uint sum = 0u;\n"
    "    for (uint t = b; t < e; ++t) { sc[3u*T + t] = sum; sum += sc[t]; }   /* rbase */\n"
    "    part[tid] = sum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    uint off = 0u, total = 0u;\n"
    "    for (uint j = 0u; j < WG; ++j) { total += part[j]; if (j < tid) off += part[j]; }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint t = b; t < e; ++t) sc[3u*T + t] += off;\n"
    "    if (tid == 0u) prep[19] = total;              /* total restarts */\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_runs(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                          const device uchar* indices [[buffer(2)]],\n"
    "                          device uint* prep [[buffer(7)]],\n"
    "                          device uint* sc [[buffer(8)]],\n"
    "                          uint slid [[thread_index_in_simdgroup]],\n"
    "                          uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (prep[18] == 0u) return;\n"
    "    uint T = prep[16];\n"
    "    uint t = gid;\n"
    "    if (t >= T) return;\n"
    "    uint count = prep[12], total = prep[19];\n"
    "    uint indexed = c.v[3], restart = c.v[7], idx_size = c.v[10];\n"
    "    bool restart_en = c.v[14] != 0u;\n"
    "    uint i = t * WG + slid;\n"
    "    bool valid = i < count;\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[_i] : (idx_size == 2u ? (uint)((const device ushort*)indices)[_i] : ((const device uint*)indices)[_i])) : 0u)\n"
    "    uint vtx = valid ? MGL_IDX(i) : 0u;\n"
    "    bool res = valid && indexed && restart_en && vtx == restart;\n"
    "    uint rb = (uint)(uint64_t)simd_ballot(res);\n"
    "    uint nres = popcount(rb);\n"
    "    uint rb_t = sc[3u*T + t];                     /* restarts before this tile */\n"
    "    uint ro = 5u * T;                             /* run records */\n"
    "    if (res) {\n"
    "        uint rank = popcount(rb & ((1u << slid) - 1u));\n"
    "        uint run = rb_t + rank;\n"
    "        uint start;\n"
    "        if (rank != 0u) {\n"
    "            uint prev = 31u - clz(rb & ((1u << slid) - 1u));   /* the previous restart, same tile */\n"
    "            start = t * WG + prev + 1u;\n"
    "        } else if (run == 0u) {\n"
    "            start = 0u;\n"
    "        } else {\n"
    "            /* the previous restart is the last one of an earlier tile: the greatest\n"
    "             * tile u with rbase[u] < rbase[t] (rbase is non-decreasing) */\n"
    "            uint lo = 0u, hi = t;\n"
    "            while (lo + 1u < hi) { uint mid = (lo + hi) >> 1; if (sc[3u*T + mid] < rb_t) lo = mid; else hi = mid; }\n"
    "            start = sc[4u*T + lo] + 1u;\n"
    "        }\n"
    "        sc[ro + 4u*run + 0u] = start;\n"
    "        sc[ro + 4u*run + 1u] = i - start;         /* subcount */\n"
    "    }\n"
    "    /* the trailing run after the last restart (the whole buffer when there is none)\n"
    "     * is recorded by the tile holding that restart */\n"
    "    bool mine = nres > 0u && rb_t + nres == total;\n"
    "    if (total == 0u) mine = t == 0u;\n"
    "    if (mine && slid == 0u) {\n"
    "        uint start = total > 0u ? sc[4u*T + t] + 1u : 0u;\n"
    "        sc[ro + 4u*total + 0u] = start;\n"
    "        sc[ro + 4u*total + 1u] = count - start;\n"
    "    }\n"
    "#undef MGL_IDX\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_reduce(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                            device uint* hdr [[buffer(5)]],\n"
    "                            device uint* ctl [[buffer(6)]],\n"
    "                            device uint* prep [[buffer(7)]],\n"
    "                            device uint* sc [[buffer(8)]],\n"
    "                            uint tid [[thread_index_in_threadgroup]])\n"
    "{\n"
    "    if (prep[18] == 0u) return;\n"
    "    uint T = prep[16], count = prep[12], instances = prep[15], total = prep[19];\n"
    "    uint R = total + 1u;\n"
    "    uint ro = 5u * T;\n"
    "    threadgroup uint pc[WG], pp[WG], rm[WG], rx[WG];\n"
    "\n"
    "    /* per-run exclusive prefix sums: capbase over subcount, primbase over subprims */\n"
    "    uint chunk = (R + WG - 1u) / WG;\n"
    "    uint b = min(tid * chunk, R), e = min(b + chunk, R);\n"
    "    uint subc = 0u, subp = 0u;\n"
    "    for (uint r = b; r < e; ++r) {\n"
    "        sc[ro + 4u*r + 2u] = subc;\n"
    "        sc[ro + 4u*r + 3u] = subp;\n"
    "        uint s = sc[ro + 4u*r + 1u];\n"
    "        subc += s;\n"
    "        subp += mgl_geom_subprims(c.v[0], s);\n"
    "    }\n"
    "    pc[tid] = subc; pp[tid] = subp;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    uint offc = 0u, offp = 0u, totc = 0u, totp = 0u;\n"
    "    for (uint j = 0u; j < WG; ++j) { totc += pc[j]; totp += pp[j]; if (j < tid) { offc += pc[j]; offp += pp[j]; } }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint r = b; r < e; ++r) { sc[ro + 4u*r + 2u] += offc; sc[ro + 4u*r + 3u] += offp; }\n"
    "\n"
    "    /* the tile min/max reduce (over every captured, i.e. non-restart, vertex) */\n"
    "    uint mn = 0xffffffffu, mx = 0u;\n"
    "    for (uint t = tid; t < T; t += WG) { mn = min(mn, sc[T + t]); mx = max(mx, sc[2u*T + t]); }\n"
    "    rm[tid] = mn; rx[tid] = mx;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    if (tid == 0u) {\n"
    "        for (uint j = 1u; j < WG; ++j) { mn = min(mn, rm[j]); mx = max(mx, rx[j]); }\n"
    "        uint out_prims = totp, out_idx = totc, vmin = mn, vmax = mx;\n"
    "        if (vmin == 0xffffffffu) { vmin = 0u; vmax = 0u; }\n"
    "        prep[0] = out_prims; prep[1] = out_idx; prep[2] = vmax; prep[3] = vmin;\n"
    "        prep[4] = (uint)(((uint64_t)out_prims * instances * c.v[6]) / WG + 1ull);\n"
    "        prep[5] = 1u; prep[6] = 1u;\n"
    "        prep[7] = out_idx; prep[8] = instances; prep[9] = 0u;\n"
    "        prep[10] = c.v[3] != 0u ? prep[14] : 0u;   /* basevertex of the capture draw */\n"
    "        prep[11] = 0u;\n"
    "        hdr[0] = out_prims; hdr[1] = instances; hdr[6] = vmax - vmin + 1u; hdr[12] = vmin;\n"
    "        ctl[0] = vmin + (c.v[3] != 0u ? prep[14] : 0u);\n"
    "        ctl[1] = vmax - vmin + 1u;                /* capture header: vcount */\n"
    "        prep[27] = R;\n"
    "        /* the sentinel record behind the last run stops the emit kernel's searches */\n"
    "        sc[ro + 4u*R + 0u] = count; sc[ro + 4u*R + 1u] = 0u;\n"
    "        sc[ro + 4u*R + 2u] = totc;  sc[ro + 4u*R + 3u] = totp;\n"
    "        uint items = out_prims + out_idx;\n"
    "        prep[24] = items ? (items + WG - 1u) / WG : 1u;   /* emit grid (never zero) */\n"
    "        prep[25] = 1u; prep[26] = 1u;\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_emit(constant mgl_geom_prep_in& c [[buffer(0)]],\n"
    "                          const device uchar* indices [[buffer(2)]],\n"
    "                          device uint* prim [[buffer(3)]],\n"
    "                          device uint* cap [[buffer(4)]],\n"
    "                          device uint* prep [[buffer(7)]],\n"
    "                          device uint* sc [[buffer(8)]],\n"
    "                          uint gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (prep[18] == 0u) return;\n"
    "    uint out_prims = prep[0], out_idx = prep[1];\n"
    "    if (gid >= out_prims + out_idx) return;\n"
    "    uint mode = c.v[0], first = prep[13], basevertex = prep[14];\n"
    "    uint indexed = c.v[3], k = c.v[8], provoking_first = c.v[9], idx_size = c.v[10];\n"
    "    uint cap_idx = c.v[12], cap_prim = c.v[13];\n"
    "    uint T = prep[16], R = prep[27];\n"
    "    uint ro = 5u * T;\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[_i] : (idx_size == 2u ? (uint)((const device ushort*)indices)[_i] : ((const device uint*)indices)[_i])) : (uint)(first + (int)(_i)))\n"
    "    if (gid < out_prims) {\n"
    "        if (gid >= cap_prim) return;              /* the serial kernel's guard */\n"
    "        uint lo = 0u, hi = R;                     /* last run with primbase <= gid */\n"
    "        while (lo + 1u < hi) { uint mid = (lo + hi) >> 1; if (sc[ro + 4u*mid + 3u] <= gid) lo = mid; else hi = mid; }\n"
    "        uint p = gid - sc[ro + 4u*lo + 3u];\n"
    "        uint start = sc[ro + 4u*lo + 0u], subcount = sc[ro + 4u*lo + 1u];\n"
    "        for (uint v = 0u; v < k; ++v) {\n"
    "            uint id = mgl_geom_vert_id(mode, p, v, k, subcount, provoking_first);\n"
    "            prim[gid * k + v] = MGL_IDX(start + id) + basevertex;\n"
    "        }\n"
    "    } else {\n"
    "        uint j = gid - out_prims;\n"
    "        if (j >= cap_idx) return;\n"
    "        uint lo = 0u, hi = R;                     /* last run with capbase <= j */\n"
    "        while (lo + 1u < hi) { uint mid = (lo + hi) >> 1; if (sc[ro + 4u*mid + 2u] <= j) lo = mid; else hi = mid; }\n"
    "        uint i = j - sc[ro + 4u*lo + 2u];\n"
    "        cap[j] = MGL_IDX(sc[ro + 4u*lo + 0u] + i);\n"
    "    }\n"
    "#undef MGL_IDX\n"
    "}\n";

/* the tess producer's import kernel: repacks the TES-output capture records into this
 * program's input record (the layouts differ: the tess record follows the tess capture's
 * member table, this program's record follows geom_layout), copies the tessellator's index
 * list into the primitive slots and publishes the counts the geometry kernel needs
 * (gPrim's header words and the prep block, exactly what the prep kernel writes for a
 * plain draw) */
static const char *kGeomTessImportMSL =
    "#include <metal_stdlib>\n"
    "#include <simd/simd.h>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void mgl_geom_tess_import(const device uint* src [[buffer(0)]],      /* tess records */\n"
    "                                 const device uint* idx [[buffer(1)]],      /* the tessellator's index list */\n"
    "                                 const device uint* cnt [[buffer(2)]],      /* cnt[0] records, cnt[1] indices */\n"
    "                                 device uint* recs [[buffer(3)]],           /* this program's input records */\n"
    "                                 constant uint* pairs [[buffer(4)]],        /* (src word, dst word) */\n"
    "                                 device uint* hdr [[buffer(5)]],            /* gPrim header */\n"
    "                                 device uint* pslot [[buffer(6)]],          /* primitive slots */\n"
    "                                 device uint* cap_hdr [[buffer(7)]],        /* the capture header */\n"
    "                                 device uint* prep [[buffer(8)]],           /* prep[0] / prep[4] */\n"
    "                                 constant uint* cfg [[buffer(9)]],          /* stride,k,npairs,reccap,idxcap,in_words,inst,invoc */\n"
    "                                 uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    uint nrec = cnt[0];\n"
    "    uint nidx = cnt[1];\n"
    "    if (nrec > cfg[3]) nrec = cfg[3];                 /* the ring's record capacity */\n"
    "    if (nidx > cfg[4]) nidx = cfg[4];                 /* and its index capacity */\n"
    "    if (t == 0u)\n"
    "    {\n"
    "        uint prims = nidx / cfg[1];                   /* k index entries per primitive */\n"
    "        hdr[0] = prims;\n"
    "        hdr[1] = cfg[6];                              /* instances (folded into the patches) */\n"
    "        hdr[6] = nrec;\n"
    "        hdr[8] = 0u;\n"
    "        hdr[12] = 0u;\n"
    "        cap_hdr[0] = 0u;\n"
    "        cap_hdr[1] = nrec;\n"
    "        cap_hdr[2] = 0u;\n"
    "        cap_hdr[3] = cfg[5];\n"
    "        prep[0] = prims;\n"
    "        prep[4] = (prims * cfg[6] * cfg[7]) / 32u + 1u;   /* the kernel's indirect dispatch: x, y, z */\n"
    "        prep[5] = 1u;\n"
    "        prep[6] = 1u;\n"
    "    }\n"
    "    if (t < nrec)\n"
    "    {\n"
    "        const device uint* s = src + (size_t)t * cfg[0];\n"
    "        device uint* dst = recs + (size_t)t * cfg[5];\n"
    "        for (uint p = 0u; p < cfg[2] * 2u; p += 2u) dst[pairs[p + 1u]] = s[pairs[p]];\n"
    "    }\n"
    "    if (t < nidx) pslot[t] = idx[t];\n"
    "}\n";

#include "mgl_ext_geom_xfb.h"

/* the transform feedback pack kernels: a prefix sum over the per-thread emitted counts
 * gives every output vertex its (compacted) transform feedback index, a scatter writes the
 * varyings, and a finish kernel advances the write offsets and folds the primitive count */
static const char *kGeomXfbMSL =
    "#include <metal_stdlib>\n"
    "#include <simd/simd.h>\n"
    "using namespace metal;\n"
    "\n"
    "#define WG " MGL_GEOM_STR(MGL_GEOM_XFB_WG) "\n"
    "\n"
    "struct mgl_geom_xfb_var { uint src_word; uint dst_off; uint words; uint buffer; };\n"
    "\n"
    "kernel void mgl_geom_xfb_prefix(constant uint* in [[buffer(0)]],\n"
    "                                const device uint* cnt [[buffer(1)]],\n"
    "                                device uint* xbase [[buffer(2)]],\n"
    "                                device uint* xfb [[buffer(3)]],\n"
    "                                uint tid [[thread_index_in_threadgroup]])\n"
    "{\n"
    "    threadgroup uint tgr[WG];\n"
    "    threadgroup uint carry[1];\n"
    "    uint threads = in[0];\n"
    "    uint v;\n"
    "\n"
    "    if (tid == 0u) { carry[0] = 0u; xfb[8] = 0u; }   /* per-draw vertex counter */\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    for (uint base = 0u; base < threads; base += WG) {\n"
    "        uint i = base + tid;\n"
    "        v = (i < threads) ? cnt[i] : 0u;\n"
    "        tgr[tid] = v;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint s = 1u; s < WG; s *= 2u) {          /* inclusive scan */\n"
    "            uint add = 0u;\n"
    "            if (tid >= s) add = tgr[tid - s];\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "            tgr[tid] += add;\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        }\n"
    "        if (i < threads) xbase[i] = tgr[tid] - v + carry[0];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        if (tid == WG - 1u) carry[0] += tgr[WG - 1u];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_xfb_scatter(constant uint* in [[buffer(0)]],\n"
    "                                 constant mgl_geom_xfb_var* vars [[buffer(1)]],\n"
    "                                 const device uint* recs [[buffer(2)]],\n"
    "                                 const device uint* cnt [[buffer(3)]],\n"
    "                                 const device uint* xbase [[buffer(4)]],\n"
    "                                 device uint* xfb [[buffer(5)]],\n"
    "                                 device uint* dst0 [[buffer(6)]],\n"
    "                                 device uint* dst1 [[buffer(7)]],\n"
    "                                 device uint* dst2 [[buffer(8)]],\n"
    "                                 device uint* dst3 [[buffer(9)]],\n"
    "                                 uint slot [[thread_position_in_grid]])\n"
    "{\n"
    "    uint maxv = in[1], out_words = in[2], nvars = in[3];\n"
    "    uint t = slot / maxv, s = slot - t * maxv;\n"
    "    device uint* dst[4];\n"
    "    dst[0] = dst0; dst[1] = dst1; dst[2] = dst2; dst[3] = dst3;\n"
    "\n"
    "    if (t >= in[0] || s >= cnt[t]) return;\n"
    "\n"
    "    const device uint* rec = recs + (size_t)slot * out_words;\n"
    "    if (((rec[1] >> 24) & 0xffu) != 0u) return;       /* the rasterization stream */\n"
    "\n"
    "    uint out_index = xbase[t] + s;\n"
    "    for (uint v = 0u; v < nvars; v++) {\n"
    "        constant mgl_geom_xfb_var& vr = vars[v];\n"
    "        device uint* d = dst[vr.buffer];\n"
    "        uint stride = in[4 + vr.buffer];\n"
    "        if (!stride || !vr.words) continue;           /* skip / next-buffer markers */\n"
    "        uint pos = xfb[3 + vr.buffer] + out_index * stride + vr.dst_off;\n"
    "        if (pos + vr.words * 4u > in[8 + vr.buffer]) continue;   /* full: drop, as GL does */\n"
    "        for (uint w = 0u; w < vr.words; w++) d[(pos + w * 4u) / 4u] = rec[vr.src_word + w];\n"
    "    }\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[1], 1u, memory_order_relaxed);\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[8], 1u, memory_order_relaxed);\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_xfb_finish(constant uint* in [[buffer(0)]],\n"
    "                                const device uint* ctl [[buffer(1)]],\n"
    "                                device uint* xfb [[buffer(2)]])\n"
    "{\n"
    "    for (uint b = 0u; b < 4u; b++)\n"
    "        if (in[4 + b]) xfb[3 + b] += xfb[8] * in[4 + b];\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[2], ctl[13], memory_order_relaxed);\n"
    "}\n"
    "\n"
    "/* initialise the counter block on the command stream: the CPU must never write it while\n"
    " * a pack may be in flight.  in[0] = object name, in[1..4] = the starting write offsets.\n"
    " * xfb[8] (the per-draw vertex counter) is cleared here as well: the geometry path's\n"
    " * prefix kernel does it, the tessellation path has no prefix kernel. */\n"
    "kernel void mgl_geom_xfb_begin(constant uint* in [[buffer(0)]], device uint* xfb [[buffer(1)]])\n"
    "{\n"
    "    xfb[0] = in[0];\n"
    "    xfb[1] = 0u;\n"
    "    xfb[2] = 0u;\n"
    "    xfb[3] = in[1];\n"
    "    xfb[4] = in[2];\n"
    "    xfb[5] = in[3];\n"
    "    xfb[6] = in[4];\n"
    "    xfb[7] = 0u;\n"
    "    xfb[8] = 0u;\n"
    "}\n"
    "\n"
    "/* transform feedback from a tessellation stage WITHOUT a geometry shader: the TES is\n"
    " * the last pre-rasterization stage, so the captured records are exactly the vertices GL\n"
    " * records, in capture order and all on stream 0.  One thread per record, no prefix sums.\n"
    " * in: [0]=record capacity, [1]=words per capture record, [2]=nvars, [3..6]=buffer\n"
    " * strides, [8..11]=buffer sizes, [12]=captured vertices per primitive */\n"
    "kernel void mgl_geom_xfb_tess(constant uint* in [[buffer(0)]],\n"
    "                              constant mgl_geom_xfb_var* vars [[buffer(1)]],\n"
    "                              const device uint* recs [[buffer(2)]],\n"
    "                              const device uint* counts [[buffer(3)]],\n"
    "                              device uint* xfb [[buffer(4)]],\n"
    "                              device uint* dst0 [[buffer(5)]],\n"
    "                              device uint* dst1 [[buffer(6)]],\n"
    "                              device uint* dst2 [[buffer(7)]],\n"
    "                              device uint* dst3 [[buffer(8)]],\n"
    "                              uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    device uint* dst[4];\n"
    "    dst[0] = dst0; dst[1] = dst1; dst[2] = dst2; dst[3] = dst3;\n"
    "    uint nvars = in[2];\n"
    "    if (t >= in[0] || t >= counts[0]) return;\n"
    "    const device uint* rec = recs + (size_t)t * in[1];\n"
    "    for (uint v = 0u; v < nvars; v++) {\n"
    "        constant mgl_geom_xfb_var& vr = vars[v];\n"
    "        device uint* d = dst[vr.buffer];\n"
    "        uint stride = in[3 + vr.buffer];\n"
    "        if (!stride || !vr.words) continue;\n"
    "        uint pos = xfb[3 + vr.buffer] + t * stride + vr.dst_off;\n"
    "        if (pos + vr.words * 4u > in[8 + vr.buffer]) continue;   /* full: drop, as GL does */\n"
    "        for (uint w = 0u; w < vr.words; w++) d[(pos + w * 4u) / 4u] = rec[vr.src_word + w];\n"
    "    }\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[1], 1u, memory_order_relaxed);\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[8], 1u, memory_order_relaxed);\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_xfb_tess_finish(constant uint* in [[buffer(0)]], device uint* xfb [[buffer(1)]])\n"
    "{\n"
    "    for (uint b = 0u; b < 4u; b++)\n"
    "        if (in[4 + b]) xfb[3 + b] += xfb[8] * in[4 + b];\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[2], in[12] ? xfb[8] / in[12] : xfb[8],\n"
    "                              memory_order_relaxed);\n"
    "}\n";


/* the counters + indirect arguments live in one buffer; two tiny program-independent
 * kernels manage them (zeroing before the geometry kernel, filling the draw arguments
 * from the atomic counters afterwards) */
static const char *kGeomCtlMSL =
    "#include <metal_stdlib>\n"
    "#include <simd/simd.h>\n"
    "using namespace metal;\n"
    "\n"
    "struct mgl_geom_ctl { uint mgl_ctl[" MGL_GEOM_STR(MGL_GEOM_CTL_WORDS) "]; };\n"
    "\n"
    "kernel void mgl_geom_init(device mgl_geom_ctl& c [[buffer(0)]])\n"
    "{\n"
    "    for (int i = 0; i < " MGL_GEOM_STR(MGL_GEOM_CTL_WORDS) "; i++) c.mgl_ctl[i] = 0u;\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_finalize(device mgl_geom_ctl& c [[buffer(0)]])\n"
    "{\n"
    "    c.mgl_ctl[8]  = c.mgl_ctl[1];      /* indexCount  */\n"
    "    c.mgl_ctl[9]  = 1u;                /* instanceCount */\n"
    "    c.mgl_ctl[10] = 0u;                /* startIndex  */\n"
    "    c.mgl_ctl[11] = 0u;                /* baseVertex  */\n"
    "    c.mgl_ctl[12] = 0u;                /* baseInstance */\n"
    "}\n";

static bool geomCtlPipelines(id<MTLDevice> device)
{
    if (gInitPso && gFinPso && gPrepPso) return true;

    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource: [NSString stringWithUTF8String: kGeomCtlMSL]
                                              options: nil error: &error];
    if (!lib)
    {
        NSLog(@"MGL geom: control kernel library failed: %@", error);
        return false;
    }
    id<MTLFunction> initfn = [lib newFunctionWithName: @"mgl_geom_init"];
    id<MTLFunction> finfn = [lib newFunctionWithName: @"mgl_geom_finalize"];
    if (!initfn || !finfn) return false;

    gInitPso = [device newComputePipelineStateWithFunction: initfn error: &error];
    gFinPso = [device newComputePipelineStateWithFunction: finfn error: &error];
    if (!gInitPso || !gFinPso) return false;

    /* the prep kernel lives in its own library (it declares its own input struct) */
    id<MTLLibrary> plib = [device newLibraryWithSource: [NSString stringWithUTF8String: kGeomPrepMSL]
                                               options: nil error: &error];
    if (!plib)
    {
        NSLog(@"MGL geom: prep kernel library failed: %@", error);
        return false;
    }
    id<MTLFunction> prepfn = [plib newFunctionWithName: @"mgl_geom_prep"];
    if (!prepfn) return false;
    gPrepPso = [device newComputePipelineStateWithFunction: prepfn error: &error];
    if (!gPrepPso)
    {
        NSLog(@"MGL geom: prep pipeline failed: %@", error);
        return false;
    }
    /* the tile-parallel prep chain lives in the same library */
    if (MGL_GEOM_WG == 32)
    {
        struct { const char *name; __strong id<MTLComputePipelineState> *pso; } k[] = {
            { "mgl_geom_prepc",  &gPrepcPso },
            { "mgl_geom_tiles",  &gTilesPso },
            { "mgl_geom_prefix", &gPrefixPso },
            { "mgl_geom_runs",   &gRunsPso },
            { "mgl_geom_reduce", &gReducePso },
            { "mgl_geom_emit",   &gEmitPso },
        };
        for (size_t i = 0; i < sizeof k / sizeof k[0]; i++)
        {
            id<MTLFunction> fn = [plib newFunctionWithName: [NSString stringWithUTF8String: k[i].name]];
            if (fn) *k[i].pso = [device newComputePipelineStateWithFunction: fn error: &error];
            if (!*k[i].pso)
            {
                NSLog(@"MGL geom: tiled prep pipeline '%s' failed: %@", k[i].name, error);
                return false;
            }
        }
    }

    /* the tess import kernel (a geometry stage behind a tessellation stage) */
    {
        id<MTLLibrary> tlib = [device newLibraryWithSource: [NSString stringWithUTF8String: kGeomTessImportMSL]
                                                   options: nil error: &error];
        if (!tlib)
        {
            NSLog(@"MGL geom: tess import library failed: %@", error);
            return false;
        }
        id<MTLFunction> tfn = [tlib newFunctionWithName: @"mgl_geom_tess_import"];
        if (!tfn) return false;
        gTessImportPso = [device newComputePipelineStateWithFunction: tfn error: &error];
        if (!gTessImportPso)
        {
            NSLog(@"MGL geom: tess import pipeline failed: %@", error);
            return false;
        }
    }

    return true;
}


static id<MTLBuffer> geomBuffer(id<MTLDevice> device, id<MTLBuffer> old, size_t bytes, size_t *cap)
{
    if (old && old.length >= bytes) return old;
    size_t sz = 4096;
    while (sz < bytes) sz *= 2;
    if (cap) *cap = sz;
    return [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
}

static void geomReleaseStatics(void)
{
    gPrim = nil; gVsout = nil; gOutv = nil; gIdx = nil; gCtl = nil; gPrep = nil; gCnt = nil;
    gOutvOff = gCntOff = gPrepOff = gPrimOff = gVsoutOff = 0;
    gPrimPool.next = gVsoutPool.next = gPrepPool.next = gOutvPool.next = gCntPool.next = 0;
    gXbase = nil; gXfb = nil; gXfbPending = false;
    gXfbOwner = 0; gXfbBegin = false;
    gGsPso = nil; gInitPso = nil; gFinPso = nil; gPrepPso = nil; gTessImportPso = nil;
    gXfbPrePso = nil; gXfbScatterPso = nil; gXfbFinPso = nil; gXfbBeginPso = nil;
    gXfbTessPso = nil; gXfbTessFinPso = nil;

    if (gPsoKey) { CFRelease(gPsoKey); gPsoKey = NULL; }
    gPrimBytes = gVsoutBytes = gOutvBytes = gIdxBytes = 0;
}

static bool geomDrawC(GLMContext glm_ctx, MGLGeomDraw *d)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [r geomDraw: d];
}

static bool geomCanAssembleGPUC(GLMContext glm_ctx)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [r geomCanAssembleGPU];
}

@implementation MGLRenderer (Geometry)

/* ---------------- the transform feedback pack, Metal side ---------------- */

/* wait for the pack's own command buffer (see geom_xfb_gpu_wait) */
- (void) geomXfbWait
{
    id<MTLCommandBuffer> cb = gXfbCb;

    if ((geomEnvRead(), gEnv.xfb) && cb)
        fprintf(stderr, "geom: wait cb=%p cur=%p status=%ld dvert=%u woff=%u\n",
                (void *)(__bridge void *)cb, (void *)(__bridge void *)_currentCommandBuffer,
                (long)cb.status, ((uint32_t *)gXfb.contents)[1], ((uint32_t *)gXfb.contents)[3]);
    if (!cb) return;
    if (cb == _currentCommandBuffer)
    {
        BUF()->flush(ctx);                      /* commits it */
        [cb waitUntilCompleted];
        return;
    }
    if (cb.status < MTLCommandBufferStatusCommitted)
    {
        [self endRenderEncoding];
        [self commitCommandBufferWithAGXRecovery: cb];
    }
    if (cb.status != MTLCommandBufferStatusError)
        [cb waitUntilCompleted];
}

bool geom_xfb_gpu_pending(void)
{
    return gXfbPending;
}

bool geom_xfb_gpu_valid(GLMContext ctx, GLuint name)
{
    (void)ctx;
    return !gXfbPending || gXfbOwner == name;
}

/* give a transform feedback buffer its Metal backing so the scatter can write it */
bool geom_xfb_gpu_ensure(GLMContext ctx, Buffer *buf)
{
    MGLRenderer *r = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    if (!buf->data.mtl_data) [r bindMTLBuffer: buf];
    return buf->data.mtl_data != NULL;
}

/* encode the tess-source transform feedback pack (no geometry shader: the TES records are
 * captured as they are, one thread per record - see mgl_ext_geom_xfb.h) */
- (bool) geomXfbTessEncode: (MGLXfbObject *) o program: (MGLXfbProgram *) xp
                        tfb: (BufferBaseTarget *) tfb src: (const MGLGeomTessXfbSource *) src
{
    uint32_t in[MGL_GEOM_XFB_IN_WORDS];
    uint32_t vars[MGL_GEOM_XFB_MAX_VARS * MGL_GEOM_XFB_VAR_WORDS];
    id<MTLComputeCommandEncoder> enc;
    id<MTLBuffer> ring = (__bridge id<MTLBuffer>)src->ring;
    NSUInteger wg = MGL_GEOM_XFB_WG;

    if (!ring) return false;
    if (!gXfbTessPso && ![self geomXfbPipelines]) return false;
    if (!gXfbTessPso) return false;

    gXfb = geomBuffer(_device, gXfb, MGL_GEOM_XFB_BLOCK_WORDS * sizeof(uint32_t), NULL);
    if (!gXfb) { geomReleaseStatics(); return false; }

    if (!gXfbPending || gXfbOwner != o->name)
    {
        if (gXfbPending) return false;       /* another object: the caller flushes first */
        gXfbOwner = o->name;
        gXfbBegin = true;
    }

    memset(in, 0, sizeof in);
    in[0] = src->capacity;                 /* the dispatch bound, clamped again in the kernel */
    in[1] = src->stride;                   /* words per capture record */
    in[2] = (uint32_t)xp->count;
    for (GLuint b = 0; b < 4; b++)
    {
        in[3 + b] = (uint32_t)xp->stride[b];
        in[8 + b] = (uint32_t)(tfb[b].buf ? tfb[b].buf->data.buffer_size : 0);
    }
    in[12] = src->per_prim;                /* captured vertices per primitive */

    memset(vars, 0, sizeof vars);
    for (GLuint v = 0; v < xp->count && v < MGL_GEOM_XFB_MAX_VARS; v++)
    {
        vars[v * 4 + 0] = (uint32_t)xp->vars[v].src_word;   /* the capture record's word offset */
        vars[v * 4 + 1] = (uint32_t)xp->vars[v].offset;
        vars[v * 4 + 2] = (uint32_t)xp->vars[v].words;
        vars[v * 4 + 3] = (uint32_t)xp->vars[v].buffer;
    }

    if (!gXfbFence) gXfbFence = [_device newFence];

    /* the tessellated draw leaves its render encoder open (the next GL command would reuse
     * it): end it before this command buffer gets a compute encoder */
    [self endRenderEncoding];
    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return false;
    [enc pushDebugGroup: @"MGL geometry xfb tess pack"];

    if (gXfbPending && gXfbFenceCb != _currentCommandBuffer)
        [enc waitForFence: gXfbFence];

    if (gXfbBegin)
    {
        uint32_t seed[MGL_GEOM_XFB_IN_WORDS];
        memset(seed, 0, sizeof seed);
        seed[0] = o->name;
        for (GLuint b = 0; b < 4; b++) seed[1 + b] = (uint32_t)o->write_offset[b];
        [enc setComputePipelineState: gXfbBeginPso];
        [enc setBytes: seed length: sizeof seed atIndex: 0];
        [enc setBuffer: gXfb offset: 0 atIndex: 1];
        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        if ([enc respondsToSelector: @selector(memoryBarrierWithScope:)])
            [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];
        gXfbBegin = false;
    }

    [enc setComputePipelineState: gXfbTessPso];
    [enc setBytes: in length: sizeof in atIndex: 0];
    [enc setBytes: vars length: (NSUInteger)(xp->count < MGL_GEOM_XFB_MAX_VARS ? xp->count : MGL_GEOM_XFB_MAX_VARS)
                    * MGL_GEOM_XFB_VAR_WORDS * sizeof(uint32_t) atIndex: 1];
    [enc setBuffer: ring offset: src->records atIndex: 2];
    [enc setBuffer: ring offset: src->counts atIndex: 3];
    [enc setBuffer: gXfb offset: 0 atIndex: 4];
    for (GLuint b = 0; b < 4; b++)
    {
        id<MTLBuffer> mtl = (tfb[b].buf && tfb[b].buf->data.mtl_data)
                            ? (__bridge id<MTLBuffer>)tfb[b].buf->data.mtl_data : gXfb;
        [enc setBuffer: mtl offset: 0 atIndex: 5 + b];
    }
    [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)(src->capacity + wg - 1) / wg, 1, 1)
          threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];

    [enc setComputePipelineState: gXfbTessFinPso];
    [enc setBytes: in length: sizeof in atIndex: 0];
    [enc setBuffer: gXfb offset: 0 atIndex: 1];
    [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

    BUF()->current_serial(ctx);
    for (GLuint b = 0; b < 4; b++)
        if (tfb[b].buf)
            mgl_buffers_mark_gpu_use(ctx, tfb[b].buf, true);

    [enc updateFence: gXfbFence];
    gXfbFenceCb = _currentCommandBuffer;

    [enc popDebugGroup];
    [enc endEncoding];

    gXfbPending = true;
    gXfbCb = _currentCommandBuffer;

    if ((geomEnvRead(), gEnv.xfb))
        fprintf(stderr, "geom: tess pack encoded (cap=%u stride=%u nvars=%u per_prim=%u)\n",
                src->capacity, src->stride, xp->count, src->per_prim);

    return true;
}

/* encode the three pack kernels for one draw (see mgl_ext_geom_xfb.h) */
- (bool) geomXfbEncode: (MGLGeomProgram *) gp object: (MGLXfbObject *) o program: (MGLXfbProgram *) xp
                   tfb: (BufferBaseTarget *) tfb threads: (uint32_t) threads
             maxVertices: (uint32_t) max_vertices outWords: (uint32_t) out_words
              vertexCap: (uint32_t) vertex_cap
{
    MGLGeomDraw dummy; (void)dummy;
    uint32_t in[MGL_GEOM_XFB_IN_WORDS];
    uint32_t vars[MGL_GEOM_XFB_MAX_VARS * MGL_GEOM_XFB_VAR_WORDS];
    id<MTLComputeCommandEncoder> enc;
    NSUInteger wg = MGL_GEOM_XFB_WG;

    if (!gXfbScatterPso && ![self geomXfbPipelines]) return false;
    (void)gp;

    gXfb = geomBuffer(_device, gXfb, MGL_GEOM_XFB_BLOCK_WORDS * sizeof(uint32_t), NULL);
    gXbase = geomBuffer(_device, gXbase, 4096 + (size_t)threads * sizeof(uint32_t), NULL);
    if (!gXfb || !gXbase) { geomReleaseStatics(); return false; }

    /* the counter block is GPU-owned for the whole block: the CPU initialises it through a
     * kernel on the command stream (at the first pack, and again after a resume) and only
     * reads it back at a resolve.  Writing it here would race the previous draw's finish
     * kernel and is what made draws after the first land at offset 0. */
    if (!gXfbPending || gXfbOwner != o->name)
    {
        if (gXfbPending) return false;       /* another object: the caller flushes first */
        gXfbOwner = o->name;
        gXfbBegin = true;
    }

    /* the per-draw inputs and the varying table go into the next ring slot: the GPU may
     * still be reading the previous one */
    memset(in, 0, sizeof in);
    in[0] = threads;
    in[1] = max_vertices ? max_vertices : 1;
    in[2] = out_words;
    in[3] = (uint32_t)xp->count;
    for (GLuint b = 0; b < 4; b++)
    {
        in[4 + b] = (uint32_t)xp->stride[b];
        in[8 + b] = (uint32_t)(tfb[b].buf ? tfb[b].buf->data.buffer_size : 0);
    }

    memset(vars, 0, sizeof vars);
    for (GLuint v = 0; v < xp->count; v++)
    {
        vars[v * 4 + 0] = (uint32_t)xp->vars[v].src_word;
        vars[v * 4 + 1] = (uint32_t)xp->vars[v].offset;
        vars[v * 4 + 2] = (uint32_t)xp->vars[v].words;
        vars[v * 4 + 3] = (uint32_t)xp->vars[v].buffer;
    }

    if (!gXfbFence) gXfbFence = [_device newFence];

    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return false;
    [enc pushDebugGroup: @"MGL geometry xfb pack"];

    /* the counter block is shared storage, which Metal does not hazard-track across command
     * buffers: wait for the previous draw's pack before reading the write offsets.  Metal
     * forbids waiting on a fence that was updated in the same command buffer, and inside one
     * buffer the encoders are already ordered, so the wait is only needed across them. */
    if (gXfbPending && gXfbFenceCb != _currentCommandBuffer)
        [enc waitForFence: gXfbFence];

    /* every per-draw input is passed by value: Metal copies it into the command buffer at
     * encode time, so no later CPU write can change what a pending draw reads */

    if (gXfbBegin)
    {
        /* the seed must live in its own buffer: the ring slot is restored (and reused by the
         * next draw) before this kernel executes */
        uint32_t seed[MGL_GEOM_XFB_IN_WORDS];
        memset(seed, 0, sizeof seed);
        seed[0] = o->name;
        for (GLuint b = 0; b < 4; b++) seed[1 + b] = (uint32_t)o->write_offset[b];
        [enc setComputePipelineState: gXfbBeginPso];
        [enc setBytes: seed length: sizeof seed atIndex: 0];
        [enc setBuffer: gXfb offset: 0 atIndex: 1];
        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        if ([enc respondsToSelector: @selector(memoryBarrierWithScope:)])
            [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];
        gXfbBegin = false;
    }

    [enc setComputePipelineState: gXfbPrePso];
    [enc setBytes: in length: sizeof in atIndex: 0];
    [enc setBuffer: gCnt offset: 0 atIndex: 1];
    [enc setBuffer: gXbase offset: 0 atIndex: 2];
    [enc setBuffer: gXfb offset: 0 atIndex: 3];
    [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
    if ([enc respondsToSelector: @selector(memoryBarrierWithScope:)])
        [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];

    [enc setComputePipelineState: gXfbScatterPso];
    [enc setBytes: in length: sizeof in atIndex: 0];
    [enc setBytes: vars length: (NSUInteger)xp->count * MGL_GEOM_XFB_VAR_WORDS * sizeof(uint32_t) atIndex: 1];
    if ((geomEnvRead(), gEnv.xfb))
        fprintf(stderr, "geom: scatter reads out buf=%p off=%zu\n",
                (void *)(__bridge void *)gOutv, gOutvOff);
    [enc setBuffer: gOutv offset: gOutvOff atIndex: 2];
    [enc setBuffer: gCnt offset: gCntOff atIndex: 3];
    [enc setBuffer: gXbase offset: 0 atIndex: 4];
    [enc setBuffer: gXfb offset: 0 atIndex: 5];
    for (GLuint b = 0; b < 4; b++)
    {
        id<MTLBuffer> mtl = (tfb[b].buf && tfb[b].buf->data.mtl_data)
                            ? (__bridge id<MTLBuffer>)tfb[b].buf->data.mtl_data : gXfb;
        [enc setBuffer: mtl offset: 0 atIndex: 6 + b];
    }
    {
        NSUInteger slots = (NSUInteger)threads * (max_vertices ? max_vertices : 1);
        if (slots > vertex_cap) slots = vertex_cap;
        [enc dispatchThreadgroups: MTLSizeMake((slots + wg - 1) / wg, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
    }

    [enc setComputePipelineState: gXfbFinPso];
    [enc setBytes: in length: sizeof in atIndex: 0];
    [enc setBuffer: gCtl offset: 0 atIndex: 1];
    [enc setBuffer: gXfb offset: 0 atIndex: 2];
    [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

    /* the transform feedback buffers were written by this encoder: record it *after* the
     * scatter is encoded, and make sure the serial belongs to this command buffer (the
     * tracking only notices a new buffer when the serial is queried) */
    BUF()->current_serial(ctx);
    for (GLuint b = 0; b < 4; b++)
        if (tfb[b].buf)
        {
            if ((geomEnvRead(), gEnv.xfb))
                fprintf(stderr, "geom: pack stamp buf=%p name=%u ctx=%p slot=%u\n",
                        (void *)tfb[b].buf, tfb[b].buf->name, (void *)ctx, b);
            mgl_buffers_mark_gpu_use(ctx, tfb[b].buf, true);
            {
                MGLBufferTrack *t = mgl_buffers_track(ctx, tfb[b].buf, false);
                fprintf(stderr, "geom: pack stamped buf=%u -> write_serial=%llu (entry %p)\n",
                        tfb[b].buf->name,
                        (unsigned long long)(t ? t->write_serial : 12345), (void *)t);
            }
            if ((geomEnvRead(), gEnv.xfb))
                fprintf(stderr, "geom:   scatter dst[%u] = %p (serial %llu cb %p)\n", b,
                        tfb[b].buf->data.mtl_data,
                        (unsigned long long)BUF()->current_serial(ctx),
                        (void *)(__bridge void *)_currentCommandBuffer);
        }

    /* publish the advanced offsets to the next draw's pack */
    [enc updateFence: gXfbFence];
    gXfbFenceCb = _currentCommandBuffer;

    [enc popDebugGroup];
    [enc endEncoding];

    gXfbPending = true;
    gXfbCb = _currentCommandBuffer;

    /* diagnostic: commit (but do not wait) after each draw's pack */
    if (getenv("MGL_GEOM_COMMIT"))
    {
        BUF()->flush(ctx);
        if (getenv("MGL_GEOM_WAIT")) BUF()->wait_serial(ctx, BUF()->current_serial(ctx));
    }

    if ((geomEnvRead(), gEnv.xfb))
    {
        /* commit + wait, then dump the transform feedback buffer, the counter block and this
         * draw's counts / prefix output: bisects "never written" vs "written then clobbered" */
        Buffer *tfb0 = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf;
        float *tf = (tfb0 && tfb0->data.buffer_data) ? (float *)tfb0->data.buffer_data : NULL;
        uint32_t *cnt = (uint32_t *)((uint8_t *)gCnt.contents + gCntOff);
        uint32_t *xb = (uint32_t *)gXbase.contents;
        uint32_t *blk = (uint32_t *)gXfb.contents;
        int i;

        /* the flush+wait is itself the workaround under test: keep it optional so the trace
         * can observe the unmodified behaviour */
        if (getenv("MGL_GEOM_XFB_FLUSH"))
        {
            BUF()->flush(ctx);
            BUF()->wait_serial(ctx, BUF()->current_serial(ctx));
        }

        uint32_t *hp = (uint32_t *)((uint8_t *)gPrim.contents + gPrimOff);
        fprintf(stderr, "geom: trace draw threads=%u dvert=%u dprim=%u woff=%u hdr: prims=%u inst=%u vcount=%u vmin=%u\n",
                threads, blk[1], blk[2], blk[3], hp[0], hp[1], hp[6], hp[12]);
        fprintf(stderr, "  cnt =");
        for (uint32_t t = 0; t < threads && t < 8; t++) fprintf(stderr, " %u", cnt[t]);
        fprintf(stderr, "  xbase =");
        for (uint32_t t = 0; t < threads && t < 8; t++) fprintf(stderr, " %u", xb[t]);
        fprintf(stderr, "\n  tf[0..11] =");
        for (i = 0; i < 12; i++) fprintf(stderr, " %5.2f", tf ? tf[i] : -9.0f);
        fprintf(stderr, "\n");
    }
    if ((geomEnvRead(), gEnv.debug))
        fprintf(stderr, "geom: pack encoded cb=%p\n", (void *)(__bridge void *)_currentCommandBuffer);
    return true;
}

bool geom_xfb_gpu_encode(GLMContext ctx, MGLGeomProgram *gp, MGLXfbObject *o, MGLXfbProgram *xp,
                         BufferBaseTarget *tfb, uint32_t threads, uint32_t max_vertices,
                         uint32_t out_words, uint32_t vertex_cap)
{
    MGLRenderer *r = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    return [r geomXfbEncode: gp object: o program: xp tfb: tfb threads: threads
                maxVertices: max_vertices outWords: out_words vertexCap: vertex_cap];
}

/* the tess-source variant: one thread per captured record (a tessellation stage without a
 * geometry shader records the TES output as it is) */
bool geom_xfb_tess_encode(GLMContext ctx, MGLXfbObject *o, MGLXfbProgram *xp,
                          BufferBaseTarget *tfb, const MGLGeomTessXfbSource *src)
{
    MGLRenderer *r = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    return [r geomXfbTessEncode: o program: xp tfb: tfb src: src];
}

/* the serial of the command buffer that last wrote the counter block */
/* wait for the command buffer the last pack was encoded into, so the counter block is
 * complete before the CPU reads it.  Metal does not hazard-track shared-storage buffers
 * across command buffers, so "the current serial" may belong to a later, empty buffer. */
void geom_xfb_gpu_wait(GLMContext ctx)
{
    MGLRenderer *r = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    [r geomXfbWait];
}

/* read the pending counters (the caller has synchronised) */
void geom_xfb_dump(GLMContext ctx)
{
    if (!(geomEnvRead(), gEnv.xfb) || !gOutv) return;
    fprintf(stderr, "geom: out records @%zu:", gOutvOff / 4);
    for (int i = 0; i < 6 * 2; i++)
    {
        uint32_t w = ((uint32_t *)gOutv.contents)[gOutvOff / 4 + i];
        fprintf(stderr, " %8.3f", *(float *)&w);
    }
    fprintf(stderr, "\n");
}

void geom_xfb_gpu_read(GLMContext ctx, MGLXfbObject *o, uint32_t *dvert, uint32_t *dprim)
{
    if ((geomEnvRead(), gEnv.debug) && gXfb)
    {
        uint32_t *d = (uint32_t *)gXfb.contents;
        uint32_t *c = (uint32_t *)(gCnt ? gCnt.contents : NULL);
        NSLog(@"MGL geom: block name=%u dvert=%u dprim=%u off=%u cnt0=%u cnt1=%u xb0=%u xb1=%u",
              d[0], d[1], d[2], d[3], c ? c[0] : 0, c ? c[1] : 0,
              (uint32_t *)(gXbase ? gXbase.contents : NULL) ? ((uint32_t *)gXbase.contents)[0] : 0,
              (uint32_t *)(gXbase ? gXbase.contents : NULL) ? ((uint32_t *)gXbase.contents)[1] : 0);
    }
    uint32_t *b = (uint32_t *)(gXfb ? gXfb.contents : NULL);

    (void)ctx;
    if (!b) { *dvert = *dprim = 0; return; }
    if (o)
    {
        for (GLuint i = 0; i < MGL_XFB_MAX_BUFFERS; i++)
            o->write_offset[i] = (GLsizeiptr)b[3 + i];
    }
    *dvert = b[1];
    *dprim = b[2];
}

void geom_xfb_gpu_reset(GLMContext ctx)
{
    uint32_t *b = (uint32_t *)(gXfb ? gXfb.contents : NULL);

    (void)ctx;
    gXfbPending = false;
    gXfbOwner = 0;
    if (!b) return;
    b[1] = 0;
    b[2] = 0;
}


- (void) bindExtGeomFuncsToGLMContext: (GLMContext) glm_ctx
{
    struct MGLExtMetal_geom *m = glm_ctx->mtl_funcs.ext_geom;
    if (!m)
    {
        m = calloc(1, sizeof(struct MGLExtMetal_geom));
        if (!m) return;
        glm_ctx->mtl_funcs.ext_geom = m;
    }
    m->draw = geomDrawC;
    m->can_assemble_gpu = geomCanAssembleGPUC;
}

/* ---- compute pipelines of the geometry kernel ---- */

/* the transform feedback pack kernels are program independent: built once, on demand (the
 * geometry path builds them with the geometry kernel, the tessellation path has no geometry
 * program, so it asks for them directly) */
- (bool) geomXfbPipelines
{
    NSError *error = nil;
    id<MTLLibrary> xlib = [_device newLibraryWithSource: [NSString stringWithUTF8String: kGeomXfbMSL]
                                               options: nil error: &error];
    if (!xlib)
    {
        NSLog(@"MGL geom: xfb pack library failed: %@", error);
        return false;
    }
    id<MTLFunction> pref = [xlib newFunctionWithName: @"mgl_geom_xfb_prefix"];
    id<MTLFunction> scf = [xlib newFunctionWithName: @"mgl_geom_xfb_scatter"];
    id<MTLFunction> finf = [xlib newFunctionWithName: @"mgl_geom_xfb_finish"];
    id<MTLFunction> begf = [xlib newFunctionWithName: @"mgl_geom_xfb_begin"];
    id<MTLFunction> tessf = [xlib newFunctionWithName: @"mgl_geom_xfb_tess"];
    id<MTLFunction> tessff = [xlib newFunctionWithName: @"mgl_geom_xfb_tess_finish"];
    if (!pref || !scf || !finf || !begf || !tessf || !tessff) return false;
    gXfbPrePso = [_device newComputePipelineStateWithFunction: pref error: &error];
    gXfbScatterPso = [_device newComputePipelineStateWithFunction: scf error: &error];
    gXfbFinPso = [_device newComputePipelineStateWithFunction: finf error: &error];
    gXfbBeginPso = [_device newComputePipelineStateWithFunction: begf error: &error];
    gXfbTessPso = [_device newComputePipelineStateWithFunction: tessf error: &error];
    gXfbTessFinPso = [_device newComputePipelineStateWithFunction: tessff error: &error];
    if (!gXfbPrePso || !gXfbScatterPso || !gXfbFinPso || !gXfbBeginPso || !gXfbTessPso || !gXfbTessFinPso)
    {
        NSLog(@"MGL geom: xfb pack pipeline failed: %@", error);
        return false;
    }
    return true;
}

- (bool) geomPipelines: (MGLGeomProgram *) gp program: (Program *) program
{
    NSError *error = nil;
    Shader *gs = program->shader_slots[_GEOMETRY_SHADER];
    const char *kernel_msl = gp->kernel_msl ? gp->kernel_msl : program->spirv[_GEOMETRY_SHADER].msl_str;

    if (!gs || !kernel_msl)
    {
        NSLog(@"MGL geom: the geometry kernel was not generated (program %u)", program->name);
        return false;
    }

    /* the program may not have been through bindMTLProgram yet: compile the kernel here */
    if (!gs->mtl_data.function || !gs->mtl_data.library)
    {
        if (!ctx->mtl_funcs.ext_program || !ctx->mtl_funcs.ext_program->compile_msl) return false;

        void *lib = NULL, *fn = NULL;
        char *err = NULL;
        if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, kernel_msl,
                                                    gp->kernel_entry ? gp->kernel_entry
                                                                     : program->spirv[_GEOMETRY_SHADER].entry_point,
                                                    &lib, &fn, &err))
        {
            NSLog(@"MGL geom: geometry kernel compile failed: %@", err ? [NSString stringWithUTF8String: err] : @"?");
            free(err);
            return false;
        }
        if (gs->mtl_data.function) ctx->mtl_funcs.ext_program->release_obj(ctx, gs->mtl_data.function);
        if (gs->mtl_data.library) ctx->mtl_funcs.ext_program->release_obj(ctx, gs->mtl_data.library);
        gs->mtl_data.function = fn;
        gs->mtl_data.library = lib;
    }

    if (gGsPso && gPsoKey == gs->mtl_data.function) return geomCtlPipelines(_device);

    id<MTLFunction> gsfn = (__bridge id<MTLFunction>)gs->mtl_data.function;

    id<MTLComputePipelineState> pso = [_device newComputePipelineStateWithFunction: gsfn error: &error];
    if (!pso)
    {
        NSLog(@"MGL geom: geometry kernel pipeline failed: %@", error);
        return false;
    }
    if (gPsoKey) CFRelease(gPsoKey);
    gPsoKey = (void *)CFBridgingRetain(gsfn);
    gGsPso = pso;
    return geomCtlPipelines(_device);
}

/* ---- resources of the geometry stage (uniforms, storage, textures) ---- */

- (bool) geomBindStageBuffers: (id<MTLComputeCommandEncoder>) enc
{
    BufferMapList list;
    GLuint count;

    memset(&list, 0, sizeof(list));

    /* the same mapping the renderer builds for a normal stage */
    if (![self mapGLBuffersToMTLBufferMap: &list stage: _GEOMETRY_SHADER]) return false;
    if (ctx->state.dirty_bits & DIRTY_BUFFER)
    {
        [self updateDirtyBaseBufferList: &list];
        ctx->state.dirty_bits &= ~DIRTY_BUFFER;
    }

    for (int i = 0; i < (int)list.count; i++)
    {
        Buffer *b = list.buffers[i].buf;
        if (!b || !b->data.mtl_data) return false;
        [enc setBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                offset: 0
               atIndex: (NSUInteger)i];
        mgl_buffers_mark_gpu_use(ctx, b, true);
    }

    /* textures and images of the geometry stage */
    {
        struct { int spvc_type; int gl_type; } types[] = {
            { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, 0 },
            { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, 1 },
            { 0, 0 }
        };

        for (int t = 0; types[t].spvc_type; t++)
        {
            count = [self getProgramBindingCount: _GEOMETRY_SHADER type: types[t].spvc_type];
            int left = (int)count;
            for (int i = 0; left > 0 && i < TEXTURE_UNITS; i++)
            {
                GLuint binding = [self getProgramBinding: _GEOMETRY_SHADER type: types[t].spvc_type index: i];
                Texture *tex = types[t].gl_type == 0 ? STATE(active_textures[binding])
                                                     : STATE(image_units[binding].tex);
                if (!tex) continue;
                left--;
                if (![self bindMTLTexture: tex]) continue;
                id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
                if (types[t].gl_type == 0)
                {
                    id<MTLSamplerState> sampler = [self texparamSamplerForUnit: binding texture: tex];
                    texture = [self texparamTextureForTex: tex base: texture];
                    [enc setTexture: texture atIndex: (NSUInteger)i];
                    if (sampler) [enc setSamplerState: sampler atIndex: (NSUInteger)i];
                }
                else
                {
                    [enc setTexture: texture atIndex: (NSUInteger)i];
                }
            }
        }
    }

    return true;
}

/* ---- the whole draw ---- */

/* the topology class the raster pass needs: Metal requires inputPrimitiveTopology when the
 * vertex function writes render_target_array_index / viewport_array_index, and it must match
 * the geometry stage's output primitive type */
static MTLPrimitiveTopologyClass gGeomTopology = MTLPrimitiveTopologyClassUnspecified;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
- (void) geomConfigureTopology: (MTLRenderPipelineDescriptor *) pd
{
    if (!pd) return;
    if (gGeomTopology != MTLPrimitiveTopologyClassUnspecified)
        pd.inputPrimitiveTopology = gGeomTopology;
}
#pragma clang diagnostic pop

/* The pass-through vertex function is the LAST pre-rasterization stage of the emulated
 * geometry pipeline, so rstate's vertex fixup (clip-space flip, point size, rasterizer
 * discard) belongs to it — never to the vertex capture variant or the geometry kernel.
 * generatePipelineDescriptor re-applies the override after rstateConfigurePipelineDescriptor
 * has specialized the program's own vertex shader, which would otherwise win: rstateSpecialize
 * is handed a Shader standing in for the pass-through function, so both the function constants
 * and the specialization cache stay rstate's own (a stack Shader is enough — rstateSpecialize
 * reads only mtl_data.library / mtl_data.function).  NULL when the override is inactive. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-protocol-method-implementation"
- (id<MTLFunction>) geomSpecializeVertexOverride
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    MGLGeomProgram *gp;

    if (!g || !g->vertex_override || !g->vertex_override_gp) return nil;
    gp = (MGLGeomProgram *)g->vertex_override_gp;
    if (!gp->draw_function || !gp->draw_library) return nil;

    {
        Shader standin;

        memset(&standin, 0, sizeof standin);
        standin.mtl_data.function = gp->draw_function;
        standin.mtl_data.library = gp->draw_library;
        return [self rstateSpecialize: &standin stage: _VERTEX_SHADER];
    }
}
#pragma clang diagnostic pop

/* can the primitive assembly run on the GPU? It needs the element buffer as a device
 * buffer (a client-side index pointer has to be assembled on the CPU) */
- (bool) geomCanAssembleGPU
{
    Buffer *eb;

    if (!gPrepPso) return false;
    if (!ctx->state.vao) return true;
    eb = ctx->state.vao->element_array.buffer;
    if (!eb) return false;
    if (!eb->data.mtl_data) [self bindMTLBuffer: eb];
    return eb->data.mtl_data != NULL;
}

/* reserve `need` bytes: *start receives this draw's range and the pool advances past it, so
 * the next draw can never land on it.  When the range would not fit, start a NEW (larger,
 * zero-filled) buffer instead of wrapping to offset 0 — a silent wrap could hand a draw a
 * range whose previous user is still pending on the GPU. */
static id<MTLBuffer> geomSubAlloc(id<MTLDevice> device, MGLGeomPool *pool, size_t need, size_t *start)
{
    size_t s;

    if (need == 0) need = 4;

    s = (pool->next + 255) & ~(size_t)255;
    if (pool->buf && s + need <= pool->buf.length)
    {
        pool->next = s + need;
        *start = s;
        return pool->buf;
    }

    /* the pool is full: wrap to the start.  Every consumer of the ranges being reused was
     * encoded before this draw's producers — a draw consumes the ranges it allocated inside
     * its own encoders, which are encoded before the next draw allocates — and Metal runs
     * the encoders of a command buffer in encoding order and the command buffers of one
     * queue in FIFO order, so the GPU can never run a producer ahead of those consumers.
     * The pool therefore settles at the high-water mark of a single draw instead of growing
     * on every draw (which reached multi-gigabyte buffers for restart-heavy draws).
     * MGL_GEOM_POOL_GROW=1 keeps the old grow-only behaviour for A/B runs. */
    if (pool->buf && need <= pool->buf.length && !(geomEnvRead(), gEnv.pool_grow))
    {
        if ((geomEnvRead(), gEnv.draw))
            fprintf(stderr, "geom: trace pool %p wrap (%zu in use)\n", (void *)pool, need);
        pool->next = need;
        *start = 0;
        return pool->buf;
    }

    /* start a fresh buffer so the range is zero-filled and has no previous user */
    size_t sz = 4096;
    size_t want = (pool->buf ? pool->buf.length * 2 : 0);
    if (want < need * 2) want = need * 2;
    while (sz < want) sz *= 2;
    if ((geomEnvRead(), gEnv.draw))
        fprintf(stderr, "geom: trace pool %p grow need=%zu old=%zu -> %zu\n",
                (void *)pool, need, (size_t)(pool->buf ? pool->buf.length : 0), sz);
    pool->buf = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
    pool->next = need;
    *start = 0;
    return pool->buf;
}

- (bool) geomDraw: (MGLGeomDraw *) d
{
    if ((geomEnvRead(), gEnv.draw))
        fprintf(stderr, "geom: trace geomDraw enter tess=%d gp=%p\n", d->tess_ring != NULL, (void *)d->gp);
    MGLGeomProgram *gp = d->gp;    Program *program = d->program;
    const bool gpu_assembly = (d->prims == NULL);   /* the CPU path supplies the lists */
    uint64_t threads = (uint64_t)(d->prims_count ? d->prims_count : 1)
                       * (uint64_t)(d->instances > 0 ? d->instances : 1) * gp->invocations;
    uint32_t bound = d->count_bound ? d->count_bound : (d->count > 0 ? (uint32_t)d->count : 1);
    size_t outv_cap, idx_cap, prim_bytes, vsout_bytes, cap_idx, cap_prim;
    uint32_t *hdr;
    bool from_tess = (d->tess_ring != NULL);
    id<MTLComputeCommandEncoder> kenc = nil;     /* the import kernel's encoder (see above) */

    if (from_tess)
    {
        /* a tessellation stage feeds the geometry kernel: the capture ring block's record
         * capacity is the record bound and its index list's capacity the primitive bound
         * (the tess domain sizes both for this draw's patches; the actual counts are written
         * by the GPU).  The tess producer tessellates patches * instances patches, so every
         * instance is already in the capture: this draw has a single instance. */
        bound = d->tess_rec_cap ? d->tess_rec_cap : 1;
        d->indexed = false;      /* index_words = 0: the primitive slots sit behind the header */
        d->type = GL_UNSIGNED_INT;
        d->count = (GLsizei)(d->tess_idx_cap ? (GLsizei)d->tess_idx_cap : 1);
        d->count_bound = bound;
        d->instances = 1;
    }

    if (!_device || !_commandQueue || !program || !gp || !gp->linked) return false;
    if (ctx->state.caps.rasterizer_discard == GL_TRUE && !mgl_xfb_capturing(ctx))
        return false;                       /* nothing to do */

    if ((geomEnvRead(), gEnv.draw))
        fprintf(stderr, "geom: trace geomDraw gp=%p linked=%d mode=%u first=%u count=%u inst=%u\n",
                (void *)gp, (int)gp->linked, d->mode, d->first, d->count, d->instances);

    /* ---- sizing ---- */
    {
        size_t max_out = MGL_GEOM_BUDGET_BYTES / (gp->out_words * sizeof(uint32_t));
        /* with GPU assembly the primitive count is only known on the GPU: dispatch (and
         * size) by the mode's worst case, the kernel returns early past the real count.
         * Behind a tessellation stage the bound is the capture's index-list capacity: k
         * entries make one primitive. */
        uint64_t bound_threads = gpu_assembly
                                 ? (from_tess
                                    ? (uint64_t)(d->tess_idx_cap ? d->tess_idx_cap : 1)
                                      / (gp->in_verts ? gp->in_verts : 1)
                                    : mgl_geom_worst_prims(d->mode, bound))
                                   * (uint64_t)(d->instances > 0 ? d->instances : 1) * gp->invocations
                                 : threads;
        outv_cap = (size_t)bound_threads * (gp->max_vertices ? gp->max_vertices : 1);
        if (outv_cap > max_out) outv_cap = max_out;
        if (!outv_cap) outv_cap = 1;
        idx_cap = outv_cap * 3;
        threads = bound_threads;
    }

    if (from_tess)
    {
        /* the import kernel fills the primitive slots from the tessellator's index list:
         * there is no capture index list of our own (cap_idx = 0), and the slots are sized
         * by that list's capacity */
        cap_idx = 0;
        cap_prim = (size_t)(d->tess_idx_cap ? d->tess_idx_cap : 1)
                   / (gp->in_verts ? gp->in_verts : 1);
        if (!cap_prim) cap_prim = 1;
    }
    else if (gpu_assembly)
    {
        /* the prep kernel writes the lists: size the regions by the mode's worst case */
        cap_idx = (size_t)bound;
        cap_prim = (size_t)mgl_geom_worst_prims(d->mode, bound);
        if (!cap_idx) cap_idx = 1;
        if (!cap_prim) cap_prim = 1;
    }
    else
    {
        cap_idx = (size_t)(d->indexed ? d->index_count : 0);
        cap_prim = (size_t)d->prims_count;
    }

    /* the capture pass writes one record per (vertex, instance); with GPU assembly the
     * vertex range is only known on the GPU, so size by the index count */
    prim_bytes = (MGL_GEOM_HDR_WORDS + cap_idx + cap_prim * gp->in_verts) * sizeof(uint32_t);
    vsout_bytes = (8 + (gpu_assembly
                        ? (size_t)bound * (size_t)(d->instances > 0 ? d->instances : 1)
                        : (size_t)d->vcount * (size_t)(d->instances > 0 ? d->instances : 1))
                       * gp->in_words) * sizeof(uint32_t);

    /* every draw gets its own range of the record buffer: a later draw's geometry kernel
     * must never overwrite the records an earlier draw's pack still has to read */
    if ((geomEnvRead(), gEnv.draw))
        fprintf(stderr, "geom: trace sizing bound=%u inst=%d vcount=%u outv_cap=%zu idx_cap=%zu cap_idx=%zu cap_prim=%zu "
                        "prim_bytes=%zu vsout_bytes=%zu threads=%llu in_words=%u out_words=%u maxverts=%u\n",
                (unsigned)bound, d->instances, d->vcount, outv_cap, idx_cap, cap_idx, cap_prim, prim_bytes, vsout_bytes,
                (unsigned long long)threads, gp->in_words, gp->out_words, gp->max_vertices);
    gOutv = geomSubAlloc(_device, &gOutvPool, (size_t)outv_cap * gp->out_words * sizeof(uint32_t), &gOutvOff);
    if (!gOutv) { geomReleaseStatics(); return false; }
    gIdx = geomBuffer(_device, gIdx, idx_cap * sizeof(uint32_t), &gIdxBytes);
    gCtl = geomBuffer(_device, gCtl, 4096, NULL);
    /* the prep kernel's outputs (the capture draw arguments and the kernel's dispatch count)
     * are read by this draw's capture pass and geometry dispatch: a later draw's prep kernel
     * must not overwrite them first */
    gPrep = geomSubAlloc(_device, &gPrepPool, MGL_GEOM_PREP_BYTES, &gPrepOff);
    if (!gPrep) { geomReleaseStatics(); return false; }
    gCnt = geomSubAlloc(_device, &gCntPool, 4096 + (size_t)threads * sizeof(uint32_t), &gCntOff);
    if (!gCnt) { geomReleaseStatics(); return false; }
    if (!gOutv || !gIdx || !gCtl || !gPrep || !gCnt) { geomReleaseStatics(); return false; }


    /* the primitive slots / capture index list and the captured VS records are shared by the
     * prep kernel, the capture pass, the geometry kernel and the pack: every draw gets its
     * own range so a later draw's kernels cannot overwrite them first */
    gPrim = geomSubAlloc(_device, &gPrimPool, prim_bytes, &gPrimOff);
    if (!gPrim) { geomReleaseStatics(); return false; }
    gVsout = geomSubAlloc(_device, &gVsoutPool, vsout_bytes, &gVsoutOff);
    if (!gVsout) { geomReleaseStatics(); return false; }

    if (![self geomPipelines: gp program: program]) return false;

    /* ---- CPU-side upload: header, prep inputs, and (for the CPU path) the lists ---- */
    {
        size_t index_words = (size_t)(!gpu_assembly && d->indexed ? d->index_count : 0);
        size_t prim_words = (size_t)(!gpu_assembly ? d->prims_count * gp->in_verts : 0);

        hdr = (uint32_t *)gPrim.contents + gPrimOff / 4;
        memset(hdr, 0, MGL_GEOM_HDR_WORDS * sizeof(uint32_t));
        hdr[1] = (uint32_t)(d->instances > 0 ? d->instances : 1);
        hdr[2] = (uint32_t)gp->invocations;
        hdr[3] = (uint32_t)outv_cap;
        hdr[4] = (uint32_t)idx_cap;
        hdr[5] = (uint32_t)gp->max_vertices;
        hdr[6] = (uint32_t)d->vcount;
        hdr[7] = (uint32_t)gp->in_verts;
        hdr[8] = (uint32_t)d->vstart;
        hdr[9] = (uint32_t)gp->in_words;
        hdr[10] = (uint32_t)gp->out_words;
        hdr[11] = (uint32_t)d->baseinstance;

        if (gpu_assembly)
        {
            /* the capture index list and the primitive slots come from the prep kernel:
             * only the inputs are uploaded, never the element buffer contents */
            uint32_t *in = (uint32_t *)gPrep.contents + gPrepOff / 4;
            memset(in, 0, MGL_GEOM_PREP_BYTES);
            in[0] = (uint32_t)mgl_geom_mode_code(d->mode);
            in[1] = (uint32_t)d->count;
            in[2] = (uint32_t)d->first;
            in[3] = d->indexed ? 1u : 0u;
            in[4] = (uint32_t)d->basevertex;
            in[5] = (uint32_t)(d->instances > 0 ? d->instances : 1);
            in[6] = (uint32_t)gp->invocations;
            in[7] = d->restart;
            in[8] = (uint32_t)gp->in_verts;
            in[9] = d->provoking_first ? 1u : 0u;
            in[10] = d->type == GL_UNSIGNED_BYTE ? 1u : d->type == GL_UNSIGNED_SHORT ? 2u : 4u;
            in[11] = (uint32_t)d->cmd_layout;
            in[12] = (uint32_t)cap_idx;
            in[13] = (uint32_t)cap_prim;
            in[14] = (ctx->state.caps.primitive_restart || ctx->state.caps.primitive_restart_fixed_index) ? 1u : 0u;

            /* the staging command the prep kernel reads when the draw is not indirect */
            if (!d->cmd_buffer)
            {
                uint32_t *cmd = in + MGL_GEOM_PREP_CMD_OFF;
                cmd[0] = (uint32_t)d->count;
                cmd[1] = (uint32_t)d->first;
                cmd[2] = (uint32_t)d->basevertex;
                cmd[3] = (uint32_t)(d->instances > 0 ? d->instances : 1);
            }
        }
        else
        {
            hdr[0] = (uint32_t)d->prims_count;
            hdr[12] = d->indexed ? (uint32_t)((uint32_t)d->vstart - (uint32_t)d->basevertex)
                                 : (uint32_t)d->vstart;
            if (index_words)
                memcpy(hdr + MGL_GEOM_HDR_WORDS, d->indices, index_words * sizeof(uint32_t));
            memcpy(hdr + MGL_GEOM_HDR_WORDS + index_words, d->prims, prim_words * sizeof(uint32_t));
        }
    }

    /* capture variant header lives in its own buffer; with GPU assembly the prep kernel
     * fills in the vertex range */
    {
        uint32_t *ch = (uint32_t *)gVsout.contents + gVsoutOff / 4;
        ch[0] = !gpu_assembly && gp->vid_absolute ? (uint32_t)d->vstart : 0;
        ch[1] = (uint32_t)(!gpu_assembly ? d->vcount : 0);
        ch[2] = gp->iid_absolute ? (uint32_t)d->baseinstance : 0;
        ch[3] = (uint32_t)gp->in_words;
    }

    /* the whole emulation is encoded into one command buffer: close whatever encoder is open
     * and make sure the buffer can still take encoders (the ring wait above may have
     * committed it) */
    [self endRenderEncoding];
    if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
    {
        [self newCommandBuffer];
        ctx->state.dirty_bits = DIRTY_ALL;
    }
    if (!_currentCommandBuffer)
    {
        NSLog(@"MGL geom: no command buffer");
        return false;
    }

    /* reserve this command buffer's place in the queue now: the emulation's scratch is
     * shared between draws, so a later draw's kernels must not be allowed to run before an
     * earlier draw's pack just because the earlier command buffer is committed later.
     * Metal asserts on enqueueing twice, so only the first enqueue is ours. */
    if (_currentCommandBuffer.status < MTLCommandBufferStatusEnqueued &&
        (void *)(__bridge void *)_currentCommandBuffer != gEnqueuedCb)
    {
        gEnqueuedCb = (void *)(__bridge void *)_currentCommandBuffer;
        [_currentCommandBuffer enqueue];
    }

    if ((geomEnvRead(), gEnv.debug))
        fprintf(stderr, "geom: draw start cb=%p\n", (void *)(__bridge void *)_currentCommandBuffer);
    /* ---- 0. prep kernel: assemble the primitives on the GPU ---- */
    if (from_tess)
    {
        /* the tess producer's records are imported instead: repack them into this program's
         * input record, copy its output index list into the primitive slots and publish the
         * counts (what the prep kernel writes for a plain draw).  The encoder is left OPEN:
         * the geometry kernel below dispatches into it too, because only dispatches within
         * one encoder are guaranteed to be ordered — the kernel's indirect threadgroup count
         * (prep[4], written here on the GPU) must be visible when that dispatch is issued. */
        id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
        id<MTLBuffer> ring = (__bridge id<MTLBuffer>)d->tess_ring;
        uint32_t cfg[8];
        NSUInteger wg = 64, nt = bound > (NSUInteger)d->count ? bound : (NSUInteger)d->count;

        if (!enc) return false;
        [enc pushDebugGroup: @"MGL geometry tess import"];
        [enc setComputePipelineState: gTessImportPso];
        [enc setBuffer: ring offset: d->tess_recs_off atIndex: 0];
        [enc setBuffer: ring offset: d->tess_idx_off atIndex: 1];
        [enc setBuffer: ring offset: d->tess_cnt_off atIndex: 2];
        [enc setBuffer: gVsout offset: gVsoutOff + 8 * sizeof(uint32_t) atIndex: 3];
        [enc setBytes: d->tess_pairs length: sizeof(uint32_t) * (NSUInteger)d->tess_npairs atIndex: 4];
        [enc setBuffer: gPrim offset: gPrimOff atIndex: 5];
        [enc setBuffer: gPrim offset: gPrimOff + (MGL_GEOM_HDR_WORDS + cap_idx) * sizeof(uint32_t) atIndex: 6];
        [enc setBuffer: gVsout offset: gVsoutOff atIndex: 7];
        [enc setBuffer: gPrep offset: gPrepOff + MGL_GEOM_PREP_OUT_OFF * sizeof(uint32_t) atIndex: 8];
        cfg[0] = d->tess_stride;                 /* words per tess record */
        cfg[1] = gp->in_verts;                   /* k: input vertices per primitive */
        cfg[2] = d->tess_npairs / 2;             /* (src, dst) word pairs */
        cfg[3] = bound;                          /* record capacity */
        cfg[4] = d->tess_idx_cap;                /* index list capacity */
        cfg[5] = gp->in_words;                   /* words per input record */
        cfg[6] = (uint32_t)(d->instances > 0 ? d->instances : 1);
        cfg[7] = gp->invocations;
        [enc setBytes: cfg length: sizeof cfg atIndex: 9];
        [enc dispatchThreadgroups: MTLSizeMake((nt + wg - 1) / wg, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
        [enc popDebugGroup];
        kenc = enc;                       /* still open: the geometry kernel joins it */
        if ((geomEnvRead(), gEnv.draw))
            fprintf(stderr, "geom: trace tess import encoded (reccap=%u idxcap=%u stride=%u)\n",
                    bound, d->tess_idx_cap, d->tess_stride);
    }
    else if (gpu_assembly)
    {
        id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
        Buffer *eb = d->indexed ? (ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL) : NULL;
        bool have_eb = false;

        if (!enc) return false;
        if (eb)
        {
            if (!eb->data.mtl_data) [self bindMTLBuffer: eb];
            have_eb = eb->data.mtl_data != NULL;
            if (have_eb) mgl_buffers_mark_gpu_use(ctx, eb, true);
        }

        [enc pushDebugGroup: @"MGL geometry prep"];
        [enc setComputePipelineState: gPrepPso];
        [enc setBuffer: gPrep offset: gPrepOff atIndex: 0];                           /* inputs */
        {
            /* the draw command: the user's indirect buffer, or the staging words in gPrep */
            id<MTLBuffer> cmd = gPrep;
            size_t cmd_off = MGL_GEOM_PREP_CMD_OFF * sizeof(uint32_t);
            if (d->cmd_buffer)
            {
                cmd = (__bridge id<MTLBuffer>)d->cmd_buffer;
                cmd_off = d->cmd_offset;
            }
            [enc setBuffer: cmd offset: cmd_off atIndex: 1];
        }
        [enc setBuffer: have_eb ? (__bridge id<MTLBuffer>)eb->data.mtl_data : gPrep
                            offset: have_eb ? d->offset : 0
                           atIndex: 2];
        [enc setBuffer: gPrim offset: gPrimOff + (MGL_GEOM_HDR_WORDS + cap_idx) * sizeof(uint32_t) atIndex: 3];
        [enc setBuffer: gPrim offset: gPrimOff + MGL_GEOM_HDR_WORDS * sizeof(uint32_t) atIndex: 4];
        [enc setBuffer: gPrim offset: gPrimOff atIndex: 5];
        [enc setBuffer: gVsout offset: gVsoutOff atIndex: 6];
        [enc setBuffer: gPrep offset: gPrepOff + MGL_GEOM_PREP_OUT_OFF * sizeof(uint32_t) atIndex: 7];

        /* Large draws go through the tile-parallel chain instead of the serial run walk in
         * mgl_geom_prep: per-tile restart counts, a prefix sum, per-tile run records and a
         * parallel emit write the same lists bit for bit (see kGeomPrepMSL).  Direct draws
         * know the count here, so the CPU picks the path and writes the context words;
         * indirect draws let mgl_geom_prepc parse the command and decide — it publishes the
         * context and the grids, and mgl_geom_prep (dispatched right after it) leaves when
         * the chain takes over. */
        bool tiled = MGL_GEOM_WG == 32 && !(geomEnvRead(), gEnv.serial_prep);
        if (tiled)
        {
            uint64_t cnt = d->count_bound ? (uint64_t)d->count_bound : (uint64_t)(d->count > 0 ? d->count : 0);
            uint64_t tiles = (cnt + MGL_GEOM_WG - 1) / MGL_GEOM_WG;
            uint64_t words = 5ull * tiles + 4ull * ((uint64_t)cnt + 2ull);
            if (getenv("MGL_GEOM_DEBUG")) NSLog(@"MGL geom tiled: cnt=%llu words=%llu scr=%zu", (unsigned long long)cnt, (unsigned long long)words, gPrepScrWords);
            uint32_t *in = (uint32_t *)gPrep.contents + gPrepOff / 4;

            if (!gPrepScr || gPrepScrWords < words)
            {
                size_t want = words > MGL_GEOM_TILED_MAX_WORDS ? MGL_GEOM_TILED_MAX_WORDS : (size_t)words;
                id<MTLBuffer> grown = geomBuffer(_device, gPrepScr, want * sizeof(uint32_t), &gPrepScrWords);
                if (!grown) tiled = false;
                else gPrepScr = grown;
            }
            if (tiled && !d->cmd_buffer)
            {
                if ((uint64_t)d->count <= (uint64_t)gTiledMin || words > gPrepScrWords)
                    tiled = false;                       /* small draw: one dispatch is cheaper */
                else
                {
                    /* publish the context the tiled kernels share (word 18 = the owner) */
                    in[MGL_GEOM_PREP_OUT_OFF + 12] = (uint32_t)d->count;
                    in[MGL_GEOM_PREP_OUT_OFF + 13] = (uint32_t)d->first;
                    in[MGL_GEOM_PREP_OUT_OFF + 14] = (uint32_t)d->basevertex;
                    in[MGL_GEOM_PREP_OUT_OFF + 15] = (uint32_t)(d->instances > 0 ? d->instances : 1);
                    in[MGL_GEOM_PREP_OUT_OFF + 16] = (uint32_t)(((uint64_t)d->count + MGL_GEOM_WG - 1) / MGL_GEOM_WG);
                    in[MGL_GEOM_PREP_OUT_OFF + 17] = (uint32_t)((uint64_t)d->count + 1ull);
                    in[MGL_GEOM_PREP_OUT_OFF + 18] = 1u;
                }
            }
            else if (tiled)
            {
                /* the count is only known on the GPU: mgl_geom_prepc gets the scratch capacity
                 * and decides (falling back to the serial kernel when it does not fit) */
                in[15] = (uint32_t)gPrepScrWords;
                [enc setComputePipelineState: gPrepcPso];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(MGL_GEOM_WG, 1, 1)];
                [enc setComputePipelineState: gPrepPso];
            }
        }

        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(MGL_GEOM_WG, 1, 1)];
        if (tiled)
        {
            uint64_t tiles = ((uint64_t)d->count + MGL_GEOM_WG - 1) / MGL_GEOM_WG;
            MTLSize wg = MTLSizeMake(MGL_GEOM_WG, 1, 1);
            [enc setBuffer: gPrepScr offset: 0 atIndex: 8];
            {
                [enc setComputePipelineState: gTilesPso];
                if (d->cmd_buffer)
                    [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                                  indirectBufferOffset: gPrepOff + (MGL_GEOM_PREP_OUT_OFF + 21) * sizeof(uint32_t)
                                                 threadsPerThreadgroup: wg];
                else
                    [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)tiles, 1, 1) threadsPerThreadgroup: wg];
            }
            {
                [enc setComputePipelineState: gPrefixPso];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: wg];
            }
            {
                [enc setComputePipelineState: gRunsPso];
                if (d->cmd_buffer)
                    [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                                  indirectBufferOffset: gPrepOff + (MGL_GEOM_PREP_OUT_OFF + 21) * sizeof(uint32_t)
                                                 threadsPerThreadgroup: wg];
                else
                    [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)tiles, 1, 1) threadsPerThreadgroup: wg];
            }
            {
                [enc setComputePipelineState: gReducePso];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: wg];
            }
            {
                [enc setComputePipelineState: gEmitPso];
                [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                              indirectBufferOffset: gPrepOff + (MGL_GEOM_PREP_OUT_OFF + 24) * sizeof(uint32_t)
                                             threadsPerThreadgroup: wg];
            }
        }
        [enc popDebugGroup];
        [enc endEncoding];
    }

    /* ---- 1. vertex capture pass ---- */
    if (!from_tess)      /* the tess producer already wrote the records */
    {
        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        MTLRenderPassDescriptor *rp = [[MTLRenderPassDescriptor alloc] init];
        id<MTLRenderCommandEncoder> saved = _currentRenderEncoder;
        id<MTLRenderCommandEncoder> enc;
        id<MTLFunction> fn = (__bridge id<MTLFunction>)gp->capture_function;
        MTLVertexDescriptor *vd;
        Program *saved_program = ctx->state.program;
        Shader *saved_vs = NULL;
        void *saved_vs_fn = NULL;

        if (!fn)
        {
            NSLog(@"MGL geom: the vertex capture variant is not compiled");
            return false;
        }

        /* the vertex descriptor and the buffer map describe the *real* vertex stage, so the
         * capture function stands in for it while they are built */
        _currentRenderEncoder = nil;
        ctx->state.dirty_bits |= DIRTY_VAO;     /* rebuild the fetch plan for this function */
        ctx->state.program = program;
        saved_vs = program->shader_slots[_VERTEX_SHADER];
        if (saved_vs)
        {
            saved_vs_fn = saved_vs->mtl_data.function;
            saved_vs->mtl_data.function = gp->capture_function;
        }
        vd = [self generateVertexDescriptor];
        ctx->state.program = saved_program;


        pd.vertexFunction = fn;
        pd.vertexDescriptor = vd;
        pd.rasterizationEnabled = NO;
        pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
        pd.rasterSampleCount = 1;

        NSError *error = nil;
        id<MTLRenderPipelineState> pso = [_device newRenderPipelineStateWithDescriptor: pd error: &error];
        if (!pso)
        {
            NSLog(@"MGL geom: capture pipeline failed: %@", error);
            if (saved_vs) saved_vs->mtl_data.function = saved_vs_fn;
            _currentRenderEncoder = saved;
            return false;
        }

        rp.renderTargetWidth = 1;
        rp.renderTargetHeight = 1;
        rp.defaultRasterSampleCount = 1;

        enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor: rp];
        if (!enc)
        {
            NSLog(@"MGL geom: capture encoder failed");
            if (saved_vs) saved_vs->mtl_data.function = saved_vs_fn;
            _currentRenderEncoder = saved;
            return false;
        }
        [enc pushDebugGroup: @"MGL geometry vertex capture"];
        [enc setRenderPipelineState: pso];

        /* vertex stage resources of the program + the capture buffer */
        ctx->state.program = program;
        if ([self mapGLBuffersToMTLBufferMap: &ctx->state.vertex_buffer_map_list stage: _VERTEX_SHADER])
        {
            if (ctx->state.dirty_bits & DIRTY_BUFFER)
            {
                [self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list];
                ctx->state.dirty_bits &= ~DIRTY_BUFFER;
            }
            for (int i = 0; i < (int)ctx->state.vertex_buffer_map_list.count; i++)
            {
                Buffer *b = ctx->state.vertex_buffer_map_list.buffers[i].buf;
                if (!b) continue;
                if (!b->data.mtl_data) [self bindMTLBuffer: b];
                if (!b->data.mtl_data) continue;
                [enc setVertexBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                              offset: 0
                             atIndex: (NSUInteger)i];
                mgl_buffers_mark_gpu_use(ctx, b, true);
            }
        }
        ctx->state.program = saved_program;
        [enc setVertexBuffer: gVsout offset: gVsoutOff atIndex: MGL_GEOM_MSL_VSOUT];

        if (gpu_assembly)
        {
            /* the prep kernel wrote the draw arguments (index count, instances, bases) */
            [enc drawIndexedPrimitives: MTLPrimitiveTypePoint
                             indexType: MTLIndexTypeUInt32
                           indexBuffer: gPrim
                     indexBufferOffset: gPrimOff + MGL_GEOM_HDR_WORDS * sizeof(uint32_t)
                        indirectBuffer: gPrep
                  indirectBufferOffset: gPrepOff + (32 + 7) * sizeof(uint32_t)];
        }
        else if (d->indexed)
        {
            [enc drawIndexedPrimitives: MTLPrimitiveTypePoint
                             indexCount: (NSUInteger)d->index_count
                               indexType: MTLIndexTypeUInt32
                             indexBuffer: gPrim
                       indexBufferOffset: gPrimOff + MGL_GEOM_HDR_WORDS * sizeof(uint32_t)
                          instanceCount: d->instances > 0 ? (NSUInteger)d->instances : 1
                             baseVertex: d->basevertex
                           baseInstance: d->baseinstance];
        }
        else
        {
            [enc drawPrimitives: MTLPrimitiveTypePoint
                     vertexStart: (NSUInteger)d->vstart
                     vertexCount: (NSUInteger)d->vcount
                   instanceCount: d->instances > 0 ? (NSUInteger)d->instances : 1
                    baseInstance: d->baseinstance];
        }

        [enc popDebugGroup];
        [enc endEncoding];

        if (saved_vs) saved_vs->mtl_data.function = saved_vs_fn;
        _currentRenderEncoder = saved;
        ctx->state.dirty_bits |= DIRTY_VAO | DIRTY_PROGRAM | DIRTY_STATE;
    }

    if ((geomEnvRead(), gEnv.xfb) && gp->in_words)
    {
        uint32_t *vs = (uint32_t *)((uint8_t *)gVsout.contents + gVsoutOff);
        int nw = gp->in_words > 6 ? 6 : gp->in_words;
        for (int v = 0; v < 3; v++)
        {
            fprintf(stderr, "geom:   vsout[%d] @%zu =", v, gVsoutOff / 4 + v * gp->in_words);
            for (int k = 0; k < nw; k++)
            {
                uint32_t w = vs[(size_t)v * gp->in_words + 8 + k];
                if (k < 4) fprintf(stderr, " %8.3f", *(float *)&w);
                else fprintf(stderr, " %08x", w);
            }
            fprintf(stderr, "\n");
        }
    }

    /* ---- 2. geometry kernel ---- */
    {
        /* the tess import kernel's encoder is still open: joining it keeps the import's
         * writes ordered before this kernel (and before its indirect dispatch) */
        id<MTLComputeCommandEncoder> enc = kenc ? kenc : [_currentCommandBuffer computeCommandEncoder];
        if (!enc) return false;

        Program *saved_program = ctx->state.program;
        ctx->state.program = program;

        [enc pushDebugGroup: @"MGL geometry kernel"];
        [enc setComputePipelineState: gGsPso];

        size_t index_words = (size_t)(d->indexed ? d->count : 0);
        [enc setBuffer: gPrim offset: gPrimOff + (MGL_GEOM_HDR_WORDS + index_words) * sizeof(uint32_t)
               atIndex: MGL_GEOM_MSL_PRIM];
        [enc setBuffer: gVsout offset: gVsoutOff + 8 * sizeof(uint32_t) atIndex: MGL_GEOM_MSL_VSOUT];
        [enc setBuffer: gPrim offset: gPrimOff atIndex: MGL_GEOM_MSL_HDR];
        if ((geomEnvRead(), gEnv.xfb))
            fprintf(stderr, "geom: GS writes out buf=%p off=%zu (cap %zu)\n",
                    (void *)(__bridge void *)gOutv, gOutvOff, (size_t)gOutv.length);
        [enc setBuffer: gOutv offset: gOutvOff atIndex: MGL_GEOM_MSL_OUTV];
        [enc setBuffer: gIdx offset: 0 atIndex: MGL_GEOM_MSL_IDX];
        [enc setBuffer: gCtl offset: 0 atIndex: MGL_GEOM_MSL_CTL];
        [enc setBuffer: gCnt offset: gCntOff atIndex: MGL_GEOM_MSL_CNT];

        if ([self geomBindStageBuffers: enc])
        {
            NSUInteger wg = 32;
            uint64_t groups = (threads + wg - 1) / wg;
            if (!groups) groups = 1;
            if (groups > 0xffffffffull) groups = 0xffffffffull;

            /* the counters and indirect arguments start at zero for this draw (a kernel,
             * not a CPU write: the buffer may still be read by an earlier command buffer) */
            [enc setComputePipelineState: gInitPso];
            [enc setBuffer: gCtl offset: 0 atIndex: 0];      /* the ctl kernels use buffer(0) */
            [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [enc setComputePipelineState: gGsPso];
            if (gpu_assembly && !(geomEnvRead(), gEnv.worst_case))
            {
                /* the prep kernel computed the exact threadgroup count */
                [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                               indirectBufferOffset: gPrepOff + MGL_GEOM_PREP_GROUPS_OFF * sizeof(uint32_t)
                                              threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
            }
            else
                [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)groups, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];

            [enc setComputePipelineState: gFinPso];
            [enc setBuffer: gCtl offset: 0 atIndex: 0];
            [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
        }

        [enc popDebugGroup];
        [enc endEncoding];
        ctx->state.program = saved_program;
    }

    /* ---- transform feedback of the geometry output ---- */
    if (mgl_xfb_capturing(ctx) && !mgl_geom_xfb_paused(ctx))
    {
        /* while paused nothing is captured at all: neither pack may run, because the CPU pack
         * would advance the object's write offsets on the CPU while the GPU counter block
         * holds its own.  The GPU pack keeps the write offsets and counters in a block the
         * kernels maintain, so the CPU only folds them in when the application can observe
         * them */
        if (!mgl_geom_xfb_pack(ctx, gp, program, (uint32_t)threads,
                               gp->max_vertices ? gp->max_vertices : 1, gp->out_words,
                               (uint32_t)outv_cap))
        {
            BUF()->flush(ctx);
            BUF()->wait_serial(ctx, BUF()->current_serial(ctx));
            mgl_geom_xfb_capture(ctx, gp, (const uint32_t *)gOutv.contents + gOutvOff / 4,
                                 (gOutv.length - gOutvOff) / sizeof(uint32_t), (const uint32_t *)gCtl.contents,
                                 (const uint32_t *)gCnt.contents + gCntOff / 4,
                                 (gCnt.length - gCntOff) / sizeof(uint32_t));
        }
    }

    if ((geomEnvRead(), gEnv.debug))
        fprintf(stderr, "geom: before raster cb=%p\n", (void *)(__bridge void *)_currentCommandBuffer);
    if ((geomEnvRead(), gEnv.draw))
    {
        /* diagnostic only: wait for this draw's kernels, then dump the assembly header, the
         * indirect draw arguments, the capture index list, the prep block and the first
         * non-zero words of the captured VS records and the GS records */
        BUF()->flush(ctx);
        BUF()->wait_serial(ctx, BUF()->current_serial(ctx));
        fprintf(stderr, "geom: trace gpu_assembly=%d prims_count=%llu vstart=%u vcount=%u "
                        "in_words=%u out_words=%u capture_fn=%p draw_fn=%p\n",
                (int)gpu_assembly, (unsigned long long)d->prims_count, d->vstart, d->vcount,
                gp->in_words, gp->out_words, gp->capture_function, gp->draw_function);
        fprintf(stderr, "geom: trace hdr @%zu:", gPrimOff / 4);
        for (int i = 0; i < 13; i++)
            fprintf(stderr, " %u", ((uint32_t *)gPrim.contents)[gPrimOff / 4 + i]);
        fprintf(stderr, "\ngeom: trace ctl:");
        for (int i = 0; i < MGL_GEOM_ARG_OFFSET + 5; i++)
            fprintf(stderr, " %u", ((uint32_t *)gCtl.contents)[i]);
        fprintf(stderr, "\ngeom: trace idx:");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %u", ((uint32_t *)gIdx.contents)[i]);
        fprintf(stderr, "\ngeom: trace prep @%zu:", gPrepOff / 4);
        for (int i = 0; i < 44; i++)
            fprintf(stderr, " %u", ((uint32_t *)gPrep.contents)[gPrepOff / 4 + i]);
        fprintf(stderr, "\ngeom: trace prim slots @%zu:", gPrimOff / 4 + MGL_GEOM_HDR_WORDS + (size_t)(d->indexed ? d->count : 0));
        {
            size_t pw = (size_t)(d->indexed ? d->count : 0);
            uint32_t *pp = (uint32_t *)gPrim.contents + gPrimOff / 4 + MGL_GEOM_HDR_WORDS + pw;
            for (int i = 0; i < 8; i++) fprintf(stderr, " %u", pp[i]);
        }
        if (from_tess)
        {
            id<MTLBuffer> rb = (__bridge id<MTLBuffer>)d->tess_ring;
            const uint32_t *cnt = (const uint32_t *)((const uint8_t *)rb.contents + d->tess_cnt_off);
            const uint32_t *tidx = (const uint32_t *)((const uint8_t *)rb.contents + d->tess_idx_off);
            const float *rec = (const float *)((const uint8_t *)rb.contents + d->tess_recs_off);
            fprintf(stderr, "\ngeom: trace tess counts=%u,%u stride=%u idx:", cnt[0], cnt[1], d->tess_stride);
            for (int i = 0; i < 8; i++) fprintf(stderr, " %u", tidx[i]);
            fprintf(stderr, "\ngeom: trace tess rec[0]:");
            for (int i = 0; i < 4; i++) fprintf(stderr, " %f", rec[i]);
        }
        fprintf(stderr, "\ngeom: trace vsout @%zu:", gVsoutOff / 4);
        {
            uint32_t *p = (uint32_t *)gVsout.contents;
            size_t n = gVsout.length / 4, first = 0xffffffff;
            for (size_t i = 0; i < n; i++) if (p[i]) { first = i; break; }
            fprintf(stderr, " [first nonzero word = %zu]", first);
            for (size_t i = (first == 0xffffffff ? 0 : first); i < n && i < (first == 0xffffffff ? 8 : first + 16); i++)
                fprintf(stderr, " %8.3f", *(float *)&p[i]);
        }
        fprintf(stderr, "\ngeom: trace cnt @%zu:", gCntOff / 4);
        for (int i = 0; i < 8; i++)
            fprintf(stderr, " %u", ((uint32_t *)gCnt.contents)[gCntOff / 4 + i]);
        fprintf(stderr, "\ngeom: trace outv @%zu:", gOutvOff / 4);
        for (int i = 0; i < 12; i++)
        {
            uint32_t w = ((uint32_t *)gOutv.contents)[gOutvOff / 4 + i];
            fprintf(stderr, " %8.3f", *(float *)&w);
        }
        fprintf(stderr, "\n");
    }
    /* ---- 3. raster pass ---- */
    if (ctx->state.caps.rasterizer_discard == GL_TRUE)
        return true;


    /* the geometry stage picks the output topology; Metal requires inputPrimitiveTopology
     * when the vertex function writes render_target_array_index / viewport_array_index */
    gGeomTopology = (gp->out_has_layer || gp->out_has_vp)
                    ? (gp->out_type == GL_POINTS ? MTLPrimitiveTopologyClassPoint
                       : gp->out_type == GL_LINE_STRIP ? MTLPrimitiveTopologyClassLine
                                                       : MTLPrimitiveTopologyClassTriangle)
                    : MTLPrimitiveTopologyClassUnspecified;

    if (!gp->draw_function)
    {
        NSLog(@"MGL geom: the pass-through vertex function is not compiled (program %u)",
              program ? program->name : 0);
        return false;
    }

    /* publish the pass-through vertex function through the renderer-level override instead of by swapping the program: with a bound program pipeline the composite is rebuilt from the stage programs inside begin_draw (mgl_program_scope_enter copies MSL by value), so a swap of the composite is discarded there and mutating the shared Shader corrupts the stage program. */
    {
        struct MGLExtState_geom *g = (struct MGLExtState_geom *)ctx->state.ext_geom;

        if (!g) return false;
        g->vertex_override = gp->draw_function;
        g->vertex_override_gp = gp;
        ctx->state.dirty_bits |= DIRTY_PROGRAM;

        bool ok = [self geomRasterPass: d];

        g->vertex_override = NULL;
        g->vertex_override_gp = NULL;
        return ok;
    }
}

- (bool) geomRasterPass: (MGLGeomDraw *) d
{
    MGLGeomProgram *gp = d->gp;
    if (!BUF()->begin_draw(ctx)) return false;

    {
        /* the kernel expands every emitted strip into an index *list*, so the raster pass
         * draws lists (points stay points) */
        MTLPrimitiveType prim = gp->out_type == GL_POINTS ? MTLPrimitiveTypePoint
                             : gp->out_type == GL_LINE_STRIP ? MTLPrimitiveTypeLine
                                                             : MTLPrimitiveTypeTriangle;
        if (!_currentRenderEncoder) return false;

        [_currentRenderEncoder setVertexBuffer: gOutv offset: gOutvOff atIndex: MGL_GEOM_MSL_DRAW];

        if ((geomEnvRead(), gEnv.draw))
        {
            uint32_t *cp = (uint32_t *)gCtl.contents;
            fprintf(stderr, "geom: trace raster prim=%d args=%u %u %u %u %u idx=%u %u %u\n",
                    (int)prim, cp[8], cp[9], cp[10], cp[11], cp[12],
                    ((uint32_t *)gIdx.contents)[0], ((uint32_t *)gIdx.contents)[1], ((uint32_t *)gIdx.contents)[2]);
        }
        [_currentRenderEncoder drawIndexedPrimitives: prim
                                               indexType: MTLIndexTypeUInt32
                                             indexBuffer: gIdx
                                       indexBufferOffset: 0
                                          indirectBuffer: gCtl
                                    indirectBufferOffset: MGL_GEOM_ARG_OFFSET * sizeof(uint32_t)];

        BUF()->end_draw(ctx);
    }

    gGeomTopology = MTLPrimitiveTopologyClassUnspecified;

    return true;
}

@end
