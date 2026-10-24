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
#include "mgl_ext_tess.h"   /* MGLTessCapture debug dump in geom_xfb_dump */
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_queries.h"   /* discard-skip: active pre-raster query counters */
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
- (void) geomPoolFlushAndWait: (id<MTLCommandBuffer>) cur;
@end

/* private renderer methods used here (declared for the compiler; implemented by the
 * buffers domain category) */
#define GEOM()   ((struct MGLExtState_geom *)ctx->state.ext_geom)
#define BUF()    (ctx->mtl_funcs.ext_buffers)

/* per-process scratch (the buffers are shared storage; they are only used by encoders of
 * the command buffer they were written for, so a single set can serve every draw) */
/* a per-draw scratch pool: the CPU writes straight into the ranges it hands out (the prep
 * inputs, the record headers) while the command buffers that consume them may not have
 * executed yet.  A byte range is therefore never reused until every consumer of it has
 * completed:
 *   - the pool rotates over MGL_GEOM_POOL_N buffers and hands out *disjoint* bump ranges
 *     inside the current one, so any number of allocations can be outstanding at once
 *     (a later draw writes different bytes than the ones a still-pending draw reads);
 *   - a slot is reclaimed (its bump pointer reset) only when the command buffer recorded
 *     for it has completed — recording the consumer that completes LAST, which Metal's
 *     in-order queue makes sound (see geomMarkPools);
 *   - when every slot is full of live ranges the GPU is many draws behind: the oldest
 *     pending consumer is waited for (the current, still-uncommitted buffer is committed
 *     first — allocation happens before any encoder of the draw exists).  That is the
 *     only wait, and it keeps the result independent of GPU timing.
 * A single allocation larger than the slot cap still gets the memory it needs; the cap
 * only bounds how far a batch of draws can run ahead before the wait. */
#define MGL_GEOM_POOL_N 4u
#define MGL_GEOM_POOL_ALIGN 256u
typedef struct {
    id<MTLBuffer> buf[MGL_GEOM_POOL_N];
    void *cb[MGL_GEOM_POOL_N];        /* the consumer of the live ranges that completes last */
    size_t used[MGL_GEOM_POOL_N];     /* live ranges are [0, used) of buf */
    bool pin[MGL_GEOM_POOL_N];        /* ranges handed out by the draw in progress */
    unsigned cur;
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
static size_t gPoolCap;              /* per-slot byte cap (MGL_GEOM_POOL_CAP) */
static unsigned gPoolN;              /* rotation depth (MGL_GEOM_POOL_SLOTS, <= MGL_GEOM_POOL_N) */
static id<MTLBuffer> gXbase;        /* compacted transform feedback index per thread */
static id<MTLBuffer> gXfb;          /* pack counters and write offsets (GPU-owned while pending) */
static bool gXfbPending;            /* a pack has not been folded into the object yet */
static id<MTLCommandBuffer> gXfbCb; /* that command buffer, for a deterministic wait */
static id<MTLFence> gXfbFence;     /* orders the packs of one block across command buffers */
static id<MTLCommandBuffer> gXfbFenceCb;   /* the command buffer that last updated it */

/* the geometry kernel's compute encoder, left OPEN when the transform-feedback pack will
 * join it: Metal only orders dispatches (and their memory effects) WITHIN one compute
 * encoder — separate encoders of one command buffer may overlap, and the pack reads the
 * kernel's per-thread emitted counts and records (see the pack block in geomDraw) */
static id<MTLComputeCommandEncoder> gPackJoinEnc;
static id<MTLCommandBuffer> gPackJoinCb;
static GLuint gXfbOwner;            /* the object the counter block belongs to */
static bool gXfbBegin;              /* the counter block still needs its init kernel */
static size_t gPrimBytes, gVsoutBytes, gOutvBytes, gIdxBytes;

static id<MTLComputePipelineState> gGsPso;
static id<MTLComputePipelineState> gInitPso;
static id<MTLComputePipelineState> gLscanPso, gLcopyPso, gLzeroPso;   /* raster list order passes */
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
    bool draw, debug, xfb, worst_case, dump, serial_prep;
} gEnv = { 0, false, false, false, false, false, false };

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
    /* the scratch pools' shape is read here once: the per-slot byte cap (how far a batch
     * of draws may run ahead of the GPU before a wait) and the rotation depth.  Test
     * hook: MGL_GEOM_POOL_CAP=<bytes> forces tiny pools so every draw wraps/reclaims. */
    gPoolCap = 64u << 20;
    gPoolN = MGL_GEOM_POOL_N;
    {
        const char *c = getenv("MGL_GEOM_POOL_CAP");
        const char *n = getenv("MGL_GEOM_POOL_SLOTS");
        if (c) { unsigned long long v = strtoull(c, NULL, 0); if (v >= 1024) gPoolCap = (size_t)v; }
        if (n) { int v = atoi(n); if (v >= 2 && v <= (int)MGL_GEOM_POOL_N) gPoolN = (unsigned)v; }
    }
    const char *tmin = getenv("MGL_GEOM_TILED_MIN");             /* test hook: force the chain */
    if (tmin) { int v = atoi(tmin); gTiledMin = v > 0 ? (uint32_t)v : 0u; }
}

static id<MTLComputePipelineState> gPrepPso, gTessImportPso;
static id<MTLComputePipelineState> gXfbPrePso, gXfbScatterPso, gXfbFinPso, gXfbBeginPso;
static id<MTLComputePipelineState> gXfbTessPso, gXfbTessFinPso;
static id<MTLComputePipelineState> gFinPso;
static id<MTLComputePipelineState> gQaccPso;      /* query accumulator (see mgl_geom_qacc) */
static id<MTLBuffer> gQacc;                       /* 8 persistent query words, shared storage */
static uint32_t gQaccReset;                       /* words that restart at the next draw */
static bool gQaccRan;                             /* an update was encoded since the last begin */
static id<MTLCommandBuffer> gQaccCb;              /* last command buffer holding an update */
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
    "    /* the tile-parallel chain owns this draw (armed by the CPU for direct draws or by\n"
    "     * mgl_geom_prepc for indirect ones): its outputs are identical, so just leave.\n"
    "     * A direct draw has layout == 0 and still arms prep[18], so the layout is NOT part\n"
    "     * of the condition — running both used to cost the serial walk (~1 us per restart)\n"
    "     * on top of the chain */\n"
    "    if (prep[18] != 0u) return;\n"
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
    "                    case 2u: case 5u: id = p + v; break;                     /* line strip(-adj): stride 1 */\n"
    "                    case 3u: id = (p + v < subcount) ? p + v : 0u; break;    /* line loop */\n"
    "                    case 1u: case 4u: case 6u: case 9u: id = p * k + v; break; /* disjoint groups: lines (2) / lines adj (4) / tri (3) / tri adj (6) */\n"
    "                    case 10u: {                                             /* tri strip adj: GL 4.6 table 10.4 (even slots triangle\n"
    "                                                                               verts, odd adjacency; odd triangles flip winding; first/last\n"
    "                                                                               triangle has no preceding/following adjacency) */\n"
    "                        uint b = 2u * p, last = subprims - 1u, e = (p & 1u) == 0u;\n"
    "                        uint v0 = e ? b : b + 2u, v1 = e ? b + 2u : b, v2 = b + 4u;\n"
    "                        uint a0 = (p == 0u) ? 1u : b - 2u;\n"
    "                        uint a1 = (p == last) ? (e ? b + 5u : b + 3u) : (e ? b + 6u : b + 3u);\n"
    "                        uint a2 = (p == last) ? (e ? b + 3u : b + 5u) : (e ? b + 3u : b + 6u);\n"
    "                        id = (v == 0u) ? v0 : (v == 1u) ? a0 : (v == 2u) ? v1 : (v == 3u) ? a1 : (v == 4u) ? v2 : a2;\n"
    "                        break;\n"
    "                    }\n"
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
    "        case 1u: case 2u: case 5u: id = p + v; break;             /* line-ish, stride 1 */\n"
    "        case 3u: id = (p + v < subcount) ? p + v : 0u; break;     /* line loop */\n"
    "        case 4u: case 6u: case 9u: id = p * k + v; break;         /* disjoint groups: lines adj (4) / tri (3) / tri adj (6) */\n"
    "        case 10u: {                                              /* tri strip adj: GL 4.6 table 10.4 (see mgl_geom_prep) */\n"
    "            uint subprims = (subcount >= 6u) ? (subcount - 4u) / 2u : 0u;\n"
    "            uint b = 2u * p, last = subprims - 1u, e = (p & 1u) == 0u;\n"
    "            uint v0 = e ? b : b + 2u, v1 = e ? b + 2u : b, v2 = b + 4u;\n"
    "            uint a0 = (p == 0u) ? 1u : b - 2u;\n"
    "            uint a1 = (p == last) ? (e ? b + 5u : b + 3u) : (e ? b + 6u : b + 3u);\n"
    "            uint a2 = (p == last) ? (e ? b + 3u : b + 5u) : (e ? b + 3u : b + 6u);\n"
    "            id = (v == 0u) ? v0 : (v == 1u) ? a0 : (v == 2u) ? v1 : (v == 3u) ? a1 : (v == 4u) ? v2 : a2;\n"
    "            break;\n"
    "        }\n"
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
    "    uint t = gid / WG;               /* one 32-thread group per tile: gid is the global thread */\n"
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
    "    uint t = gid / WG;               /* one 32-thread group per tile */\n"
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
    "struct mgl_geom_xfb_var { uint src_word; uint dst_off; uint words; uint buffer; uint stream; };\n"
    "\n"
    "kernel void mgl_geom_xfb_prefix(constant uint* in [[buffer(0)]],\n"
    "                                const device uint* recs [[buffer(4)]],\n"
    "                                const device uint* cnt [[buffer(1)]],\n"
    "                                device uint* xbase [[buffer(2)]],\n"
    "                                device uint* xfb [[buffer(3)]],\n"
    "                                uint tid [[thread_index_in_threadgroup]])\n"
    "{\n"
    "    threadgroup uint tgr[WG];\n"
    "    threadgroup uint carry[1];\n"
    "    uint threads = in[0];\n"
    "    uint maxv = in[1], out_words = in[2];\n"
    "    uint v;\n"
    "\n"
    "    if (tid == 0u) carry[0] = 0u;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    for (uint base = 0u; base < threads; base += WG) {\n"
    "        uint i = base + tid;\n"
    "        v = 0u;\n"
    "        if (i < threads) {\n"
    "            /* GL captures whole primitives, so the pack is indexed by captured\n"
    "             * primitive, not by emitted vertex: the scan input is the thread's count\n"
    "             * of primitives it closed - each flagged record stands for the n-per+1\n"
    "             * primitives its strip (n vertices, tag bits 0..7) decomposes into.\n"
    "             * Degenerate strips never get the flag, matching the scatter.\n"
    "             * One pack pass per captured stream (in[20]): only that stream's records\n"
    "             * are counted, and the ordinals the kernel stamps are per stream too. */\n"
    "            const device uint* r = recs + (size_t)i * maxv * out_words;\n"
    "            uint per = (in[13] == 2u) ? 3u : (in[13] == 1u) ? 2u : 1u;\n"
    "            uint st = in[20];\n"
    "            for (uint s = 0u, n = cnt[i]; s < n; s++) {\n"
    "                uint w = r[(size_t)s * out_words + 1u];\n"
    "                uint sv = w & 0xffu;\n"
    "                if (((w >> 8) & 1u) && ((w >> 24) == st) && sv >= per) v += sv - per + 1u;\n"
    "            }\n"
    "        }\n"
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
    "    /* the scan's total is this draw's EMITTED primitives of this stream (\"needed\":\n"
    "     * the overflow query compares it against the written count) */\n"
    "    if (tid == 0u && carry[0])\n"
    "        atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[17 + in[20]], carry[0],\n"
    "                                  memory_order_relaxed);\n"
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
    "    if (((rec[1] >> 24) & 0xffu) != in[20]) return;  /* this pass packs one stream */\n"
    "    if (((rec[1] >> 8) & 1u) == 0u) return;          /* only the vertex closing a primitive */\n"
    "\n"
    "    /* GL records WHOLE primitives: the closed strip decomposes into its line /\n"
    "     * triangle primitives and every vertex of each is captured, so shared strip\n"
    "     * vertices are written once per primitive; a primitive whose last vertex would\n"
    "     * not fit is not recorded at all (no partial writes).  The tag carries the\n"
    "     * strip's length (bits 0..7) and the primitives the thread closed before it ON\n"
    "     * THIS STREAM (bits 9..23); with the prefix scan (xbase) that gives each\n"
    "     * decomposed primitive its global captured index inside its stream.  The\n"
    "     * geometry stage flags only strips long enough to close one primitive, so the\n"
    "     * strip's records are in range. */\n"
    "    uint per = (in[13] == 2u) ? 3u : (in[13] == 1u) ? 2u : 1u;\n"
    "    uint n = rec[1] & 0xffu;\n"
    "    uint before = (rec[1] >> 9) & 0x7fffu;\n"
    "    for (uint p = 0u; p + per <= n; p++) {\n"
    "        uint prim = xbase[t] + before + p;\n"
    "\n"
    "        /* capacity test on the primitive's LAST vertex: a buffer must have room for\n"
    "         * its largest varying there (in[14+b]).  Only THIS stream's buffers are tested\n"
    "         * (in[22]): another stream's pass owns their offsets and numbers its captured\n"
    "         * primitives independently.  The test is monotonic in the write offsets, so\n"
    "         * once a primitive does not fit, no later one does - GL stops recording. */\n"
    "        bool fits = true;\n"
    "        for (uint b = 0u; b < 4u; b++) {\n"
    "            if (!((in[22] >> b) & 1u)) continue;\n"
    "            uint end = xfb[3 + b] + (prim * per + per - 1u) * in[4 + b] + in[14 + b];\n"
    "            if (end > in[8 + b]) fits = false;      /* full: the primitive is dropped, as GL does */\n"
    "        }\n"
    "        if (!fits) return;\n"
    "\n"
    "        for (uint j = 0u; j < per; j++) {\n"
    "            const device uint* rj = recs + (size_t)(slot - n + 1u + p + j) * out_words;\n"
    "            for (uint v = 0u; v < nvars; v++) {\n"
    "                constant mgl_geom_xfb_var& vr = vars[v];\n"
    "                device uint* d = dst[vr.buffer];\n"
    "                uint stride = in[4 + vr.buffer];\n"
    "                if (!stride || !vr.words) continue; /* skip / next-buffer markers */\n"
    "                if (vr.stream != in[20]) continue; /* another pass packs this stream */\n"
    "                uint pos = xfb[3 + vr.buffer] + (prim * per + j) * stride + vr.dst_off;\n"
    "                for (uint w = 0u; w < vr.words; w++) d[(pos + w * 4u) / 4u] = rj[vr.src_word + w];\n"
    "            }\n"
    "        }\n"
    "        atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[1], per, memory_order_relaxed);\n"
    "        atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[9 + in[20]], per, memory_order_relaxed);\n"
    "        atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[2], 1u, memory_order_relaxed);\n"
    "        atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[13 + in[20]], 1u, memory_order_relaxed);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void mgl_geom_xfb_finish(constant uint* in [[buffer(0)]],\n"
    "                                const device uint* ctl [[buffer(1)]],\n"
    "                                device uint* xfb [[buffer(2)]])\n"
    "{\n"
    "    uint dv[4];\n"
    "    for (uint s = 0u; s < 4u; s++) dv[s] = xfb[9 + s];   /* this draw's captured vertices */\n"
    "    for (uint b = 0u; b < 4u; b++)\n"
    "        if (in[4 + b]) {\n"
    "            /* each buffer captures one stream's varyings (in[21] packs them 2 bits per\n"
    "             * buffer); its write offset advances by that stream's captured vertices */\n"
    "            uint st = (in[21] >> (2u * b)) & 3u;\n"
    "            xfb[3 + b] += dv[st] * in[4 + b];\n"
    "        }\n"
    "    for (uint s = 0u; s < 4u; s++) {\n"
    "        xfb[21 + s] += dv[s];   /* the block totals the CPU resolve folds in */\n"
    "        xfb[9 + s] = 0u;        /* per-draw counters for the next draw */\n"
    "    }\n"
    "    /* the primitives-written counts are accumulated by the scatter kernels (they know\n"
    "     * the buffer capacities, so a primitive that does not fit is not counted) */\n"
    "}\n"
    "\n"
    "/* initialise the counter block on the command stream: the CPU must never write it while\n"
    " * a pack may be in flight.  in[0] = object name, in[1..4] = the starting write offsets.\n"
    " * xfb[8] (the tess path's per-draw vertex counter) is cleared here as well, and so are\n"
    " * the per-draw / per-stream slots: the geometry path's finish kernel re-clears its own\n"
    " * after every draw, the tessellation path has no finish-per-stream. */\n"
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
    "    xfb[9] = 0u; xfb[10] = 0u; xfb[11] = 0u; xfb[12] = 0u;   /* per-draw vertices per stream */\n"
    "    xfb[13] = 0u; xfb[14] = 0u; xfb[15] = 0u; xfb[16] = 0u;  /* primitives written per stream */\n"
    "    xfb[17] = 0u; xfb[18] = 0u; xfb[19] = 0u; xfb[20] = 0u;  /* primitives emitted per stream */\n"
    "    xfb[21] = 0u; xfb[22] = 0u; xfb[23] = 0u; xfb[24] = 0u;  /* block vertices per stream */\n"
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
    "    uint dv = xfb[8];                       /* this draw's captured vertices (stream 0) */\n"
    "    uint prims = in[12] ? dv / in[12] : dv; /* whole captured primitives */\n"
    "    for (uint b = 0u; b < 4u; b++)\n"
    "        if (in[4 + b]) xfb[3 + b] += dv * in[4 + b];\n"
    "    atomic_fetch_add_explicit((volatile device atomic_uint*)&xfb[2], prims, memory_order_relaxed);\n"
    "    /* the tessellation path captures ONE stream: mirror this draw's counts into the\n"
    "     * per-stream block-total slots the CPU resolve reads (every capture is written:\n"
    "     * the kernel drops only vertices past the buffer's end, and GL counts the whole\n"
    "     * primitives that still fit) */\n"
    "    xfb[21] += dv;\n"
    "    xfb[13] += prims;\n"
    "    xfb[17] += prims;\n"
    "    xfb[8] = 0u;                            /* per-draw counter for the next draw */\n"
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
    "}\n"
    "\n"
    "/* query accumulator: folds one draw's per-stream geometry-stage counters into the\n"
    " * persistent query words.  A word whose reset bit is set restarts at this draw's\n"
    " * value (the CPU raises the bit when a query begins, so the first draw after it\n"
    " * opens a new counting period); every other word accumulates.  Dispatched with a\n"
    " * single thread right after the geometry kernel of every geometry-stage draw, in\n"
    " * the same compute encoder - consecutive dispatches there are ordered.\n"
    " * Word w < 4: PRIMITIVES_GENERATED stream w (ctl[17 + w], the primitives the\n"
    " * rasterizer / transform feedback consume, strips decomposed).  Word 4 + w:\n"
    " * GEOMETRY_SHADER_PRIMITIVES_EMITTED stream w (ctl[13 + w], one per emitted\n"
    " * primitive as the shader closed it). */\n"
    "kernel void mgl_geom_qacc(device mgl_geom_ctl& c [[buffer(0)]],\n"
    "                          device uint* acc [[buffer(1)]],\n"
    "                          constant uint& reset_mask [[buffer(2)]])\n"
    "{\n"
    "    for (uint w = 0u; w < 8u; w++)\n"
    "    {\n"
    "        uint v = (w < 4u) ? c.mgl_ctl[17u + w] : c.mgl_ctl[9u + w];\n"
    "        if ((reset_mask >> w) & 1u) acc[w] = v;\n"
    "        else                        acc[w] += v;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* order pass 0: the per-thread list counts start at zero - threads beyond this\n"
    " * draw's dispatch grid (an indirect draw's exact group count is smaller than the\n"
    " * worst-case bound) never run the geometry kernel to publish a zero themselves */\n"
    "kernel void mgl_geom_lzero(constant uint* in [[buffer(0)]],\n"
    "                           device uint* lc  [[buffer(1)]],\n"
    "                           uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    if (t >= in[0]) return;\n"
    "    lc[t] = 0u;\n"
    "    lc[in[0] + t] = 0u;\n"
    "}\n"
    "\n"
    "/* order pass 1: exclusive prefix sum of the per-thread list entry counts (lc[t])\n"
    " * into lc[T + t], and the total into ctl[1].  A single 32-wide simdgroup walks the\n"
    " * whole thread range in chunks of 32, carrying the running total. */\n"
    "kernel void mgl_geom_lscan(constant uint* in [[buffer(0)]],\n"
    "                           device uint* c   [[buffer(1)]],\n"
    "                           device uint* lc  [[buffer(2)]],\n"
    "                           uint sgid [[thread_index_in_simdgroup]])\n"
    "{\n"
    "    uint T = in[0], carry = 0u;\n"
    "    for (uint base = 0u; base < T; base += 32u)\n"
    "    {\n"
    "        uint t = base + sgid;\n"
    "        uint orig = (t < T) ? lc[t] : 0u;\n"
    "        uint v = orig;\n"
    "        for (uint o = 1u; o < 32u; o <<= 1)\n"
    "        {\n"
    "            uint u = simd_shuffle_up(v, o);\n"
    "            if (sgid >= o) v += u;\n"
    "        }\n"
    "        if (t < T) lc[T + t] = carry + v - orig;    /* exclusive base */\n"
    "        carry += simd_broadcast(v, 31u);\n"
    "    }\n"
    "    if (sgid == 0u) c[1] = carry;                   /* total list entries */\n"
    "}\n"
    "\n"
    "/* order pass 2: copy each thread's scratch segment into the raster list at its\n"
    " * scanned offset - the segments then sit in primitive order, which an atomicAdd\n"
    " * append could not guarantee */\n"
    "kernel void mgl_geom_lcopy(constant uint* in [[buffer(0)]],\n"
    "                           device uint* lscr [[buffer(1)]],\n"
    "                           device uint* lc   [[buffer(2)]],\n"
    "                           device uint* idx  [[buffer(3)]],\n"
    "                           uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    if (t >= in[0]) return;\n"
    "    uint n = lc[t];\n"
    "    if (n == 0u) return;\n"
    "    uint dst = lc[in[0] + t], src = t * in[1];\n"
    "    for (uint i = 0u; i < n; i++) idx[dst + i] = lscr[src + i];\n"
    "}\n";

static bool geomCtlPipelines(id<MTLDevice> device)
{
    if (gInitPso && gFinPso && gPrepPso && gLscanPso && gLcopyPso && gLzeroPso && gQaccPso) return true;

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
    id<MTLFunction> lscanfn = [lib newFunctionWithName: @"mgl_geom_lscan"];
    id<MTLFunction> lcopyfn = [lib newFunctionWithName: @"mgl_geom_lcopy"];
    id<MTLFunction> lzerofn = [lib newFunctionWithName: @"mgl_geom_lzero"];
    id<MTLFunction> qaccfn = [lib newFunctionWithName: @"mgl_geom_qacc"];
    if (!initfn || !finfn || !lscanfn || !lcopyfn || !lzerofn || !qaccfn) return false;

    gInitPso = [device newComputePipelineStateWithFunction: initfn error: &error];
    gFinPso = [device newComputePipelineStateWithFunction: finfn error: &error];
    gLscanPso = [device newComputePipelineStateWithFunction: lscanfn error: &error];
    gLcopyPso = [device newComputePipelineStateWithFunction: lcopyfn error: &error];
    gLzeroPso = [device newComputePipelineStateWithFunction: lzerofn error: &error];
    gQaccPso = [device newComputePipelineStateWithFunction: qaccfn error: &error];
    if (!gInitPso || !gFinPso || !gLscanPso || !gLcopyPso || !gLzeroPso || !gQaccPso) return false;

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
    {
        MGLGeomPool *pools[5] = { &gPrimPool, &gVsoutPool, &gPrepPool, &gOutvPool, &gCntPool };
        for (int i = 0; i < 5; i++)
            for (unsigned k = 0; k < MGL_GEOM_POOL_N; k++)
                if (pools[i]->cb[k]) { CFRelease(pools[i]->cb[k]); pools[i]->cb[k] = NULL; }
    }
    memset(&gPrimPool, 0, sizeof gPrimPool);
    memset(&gVsoutPool, 0, sizeof gVsoutPool);
    memset(&gPrepPool, 0, sizeof gPrepPool);
    memset(&gOutvPool, 0, sizeof gOutvPool);
    memset(&gCntPool, 0, sizeof gCntPool);
    gXbase = nil; gXfb = nil; gXfbPending = false;
    gQacc = nil; gQaccCb = nil; gQaccReset = 0; gQaccRan = false;
    gXfbOwner = 0; gXfbBegin = false;
    gGsPso = nil; gInitPso = nil; gFinPso = nil; gPrepPso = nil; gLscanPso = gLcopyPso = gLzeroPso = nil; gTessImportPso = nil;
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

/* exact geometry-stage query counting: the accumulator words of gQacc are written by a
 * dispatch that follows every geometry kernel, so a read must first let the command
 * buffer holding the last one retire (queries.c calls these through ext_geom) */
- (void) geomQueryBegin: (unsigned) words
{
    gQaccReset |= words;
    gQaccRan = false;
}

- (void) geomQueryRead: (unsigned) word out: (uint64_t *) out
{
    if (!out) return;
    if (!gQacc || !gQaccRan || word >= 8) { *out = 0; return; }

    if (gQaccCb == _currentCommandBuffer)
    {
        BUF()->flush(ctx);                      /* commits it */
        [gQaccCb waitUntilCompleted];
    }
    else if (gQaccCb.status < MTLCommandBufferStatusCommitted)
    {
        [self endRenderEncoding];
        [self commitCommandBufferWithAGXRecovery: gQaccCb];
        if (gQaccCb.status != MTLCommandBufferStatusError)
            [gQaccCb waitUntilCompleted];
    }
    else if (gQaccCb.status != MTLCommandBufferStatusError &&
             gQaccCb.status != MTLCommandBufferStatusCompleted)
    {
        [gQaccCb waitUntilCompleted];
    }

    *out = ((volatile uint32_t *)gQacc.contents)[word];
    if (getenv("MGL_GEOM_QDEBUG"))
        fprintf(stderr, "geom: qacc read w%u -> %u  [%u %u %u %u | %u %u %u %u] ran=%d\n",
                word, (unsigned)*out,
                ((volatile uint32_t *)gQacc.contents)[0], ((volatile uint32_t *)gQacc.contents)[1],
                ((volatile uint32_t *)gQacc.contents)[2], ((volatile uint32_t *)gQacc.contents)[3],
                ((volatile uint32_t *)gQacc.contents)[4], ((volatile uint32_t *)gQacc.contents)[5],
                ((volatile uint32_t *)gQacc.contents)[6], ((volatile uint32_t *)gQacc.contents)[7],
                (int)gQaccRan);
}

static void geomQueryBeginC(GLMContext glm_ctx, unsigned words)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    [r geomQueryBegin: words];
}

static void geomQueryReadC(GLMContext glm_ctx, unsigned word, uint64_t *out)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    [r geomQueryRead: word out: out];
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
        GLsizeiptr live;
        in[3 + b] = (uint32_t)xp->stride[b];
        /* the usable range, not the Metal backing size: a pooled buffer is rounded up
         * (4096 for small allocations) and the pack's fill test must use the range the
         * GL object actually exposes.  A RANGE binding freezes its extent, but a BASE
         * binding captures into the buffer's LIVE storage (GL 4.6 6.1.1: BindBufferBase
         * binds the whole buffer, whose size later bufferData calls re-specify — a
         * bind-time snapshot truncates every capture after the buffer is re-specified;
         * tcte.data_pass_through lost 50 of 160 records exactly that way).  o->is_base
         * tells the two apart (mgl_ext_xfb.h). */
        live = tfb[b].buf ? tfb[b].buf->size : 0;
        in[8 + b] = (uint32_t)(tfb[b].buf ? ((o->is_base[b] || !tfb[b].size) ? live : tfb[b].size) : 0);
    }
    in[12] = src->per_prim;                /* captured vertices per primitive */

    memset(vars, 0, sizeof vars);
    /* stride MUST be MGL_GEOM_XFB_VAR_WORDS (5): the kernel's mgl_geom_xfb_var has a
     * trailing stream field the tess path leaves 0 (the TES captures one stream), and a
     * 4-word pack shifted every var past the first onto the wrong fields — the second
     * varying copied one word of gl_Position.x into the first buffer's y slot. */
    for (GLuint v = 0; v < xp->count && v < MGL_GEOM_XFB_MAX_VARS; v++)
    {
        vars[v * MGL_GEOM_XFB_VAR_WORDS + 0] = (uint32_t)xp->vars[v].src_word;   /* the capture record's word offset */
        vars[v * MGL_GEOM_XFB_VAR_WORDS + 1] = (uint32_t)xp->vars[v].offset;
        vars[v * MGL_GEOM_XFB_VAR_WORDS + 2] = (uint32_t)xp->vars[v].words;
        vars[v * MGL_GEOM_XFB_VAR_WORDS + 3] = (uint32_t)xp->vars[v].buffer;
        vars[v * MGL_GEOM_XFB_VAR_WORDS + 4] = 0u;                               /* stream 0 */
    }
    if (geomEnvRead(), gEnv.xfb)
        for (GLuint v = 0; v < xp->count && v < 8; v++)
            fprintf(stderr, "geom: tess pack vars[%u]: src_word=%u dst_off=%u words=%u buffer=%u stride=%u tfsize=%u\n",
                    v, vars[v * MGL_GEOM_XFB_VAR_WORDS + 0], vars[v * MGL_GEOM_XFB_VAR_WORDS + 1],
                    vars[v * MGL_GEOM_XFB_VAR_WORDS + 2], vars[v * MGL_GEOM_XFB_VAR_WORDS + 3],
                    in[3 + vars[v * MGL_GEOM_XFB_VAR_WORDS + 3]], in[8 + vars[v * MGL_GEOM_XFB_VAR_WORDS + 3]]);

    if (!gXfbFence) gXfbFence = [_device newFence];

    /* the tessellated draw leaves its render encoder open (the next GL command would reuse
     * it): end it before this command buffer gets a compute encoder.  The buffer may also
     * have been COMMITTED under us (CPU readback in the tessellation draw, query result):
     * Metal asserts when an encoder is requested on a committed buffer; the commit paths
     * leave _currentCommandBuffer == nil, and a nil buffer yields a nil encoder, which the
     * geometry capture path treats as "no work done this pass".  A non-nil but COMMITTED
     * buffer asserts the same way, so it is treated as no work too (the only caller that
     * can reach here that way consumed a stale capture — nothing was recorded). */
    [self endRenderEncoding];
    if (_currentCommandBuffer && _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        return false;
    enc = _currentCommandBuffer ? [_currentCommandBuffer computeCommandEncoder] : nil;
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
    geomMarkPools(gXfbCb);          /* the pack (this command buffer) still reads the rings */

    if ((geomEnvRead(), gEnv.xfb))
        fprintf(stderr, "geom: tess pack encoded (cap=%u stride=%u nvars=%u per_prim=%u)\n",
                src->capacity, src->stride, xp->count, src->per_prim);

    return true;
}

/* encode the pack kernels for one draw (see mgl_ext_geom_xfb.h).  GL captures every
 * output stream: one prefix + scatter pass per captured stream, sharing one compute
 * encoder (the passes of one draw are ordered inside it), then a single finish. */
- (bool) geomXfbEncode: (MGLGeomProgram *) gp object: (MGLXfbObject *) o program: (MGLXfbProgram *) xp
                   tfb: (BufferBaseTarget *) tfb threads: (uint32_t) threads
             maxVertices: (uint32_t) max_vertices outWords: (uint32_t) out_words
              vertexCap: (uint32_t) vertex_cap
{
    MGLGeomDraw dummy; (void)dummy;
    uint32_t in[MGL_GEOM_XFB_IN_WORDS];
    uint32_t vars[MGL_GEOM_XFB_MAX_VARS * MGL_GEOM_XFB_VAR_WORDS];
    id<MTLComputeCommandEncoder> enc = nil;
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
    /* vertices per primitive of the feedback mode: the pack counts written primitives
     * in these units (a strip output decomposes into its n-1 / n-2 line / triangle
     * primitives) */
    in[12] = (o->primitive_mode == GL_LINES) ? 2u : (o->primitive_mode == GL_TRIANGLES) ? 3u : 1u;
    /* the geometry stage's output type (0 points, 1 line_strip, 2 triangle_strip):
     * it decides how the emitted vertices form feedback primitives */
    in[13] = (gp->out_type == GL_LINE_STRIP) ? 1u : (gp->out_type == GL_TRIANGLE_STRIP) ? 2u : 0u;
    for (GLuint b = 0; b < 4; b++)
    {
        GLsizeiptr live;
        in[4 + b] = (uint32_t)xp->stride[b];
        /* the usable range, not the Metal backing size: a pooled buffer is rounded up
         * (4096 for small allocations) and the pack's fill test must use the range the
         * GL object actually exposes.  A RANGE binding freezes its extent, but a BASE
         * binding captures into the buffer's LIVE storage (GL 4.6 6.1.1: BindBufferBase
         * binds the whole buffer, whose size later bufferData calls re-specify — a
         * bind-time snapshot truncates every capture after the buffer is re-specified).
         * o->is_base tells the two apart (mgl_ext_xfb.h).*/
        live = tfb[b].buf ? tfb[b].buf->size : 0;
        in[8 + b] = (uint32_t)(tfb[b].buf ? ((o->is_base[b] || !tfb[b].size) ? live : tfb[b].size) : 0);
    }

    memset(vars, 0, sizeof vars);
    for (GLuint v = 0; v < xp->count; v++)
    {
        vars[v * 5 + 0] = (uint32_t)xp->vars[v].src_word;
        vars[v * 5 + 1] = (uint32_t)xp->vars[v].offset;
        vars[v * 5 + 2] = (uint32_t)xp->vars[v].words;
        vars[v * 5 + 3] = (uint32_t)xp->vars[v].buffer;
        vars[v * 5 + 4] = (uint32_t)xp->vars[v].stream;
        /* the per-buffer captured extent: the scatter's capacity test checks the last
         * vertex of a primitive against the buffer's largest varying (bytes) */
        if (xp->vars[v].words)
        {
            uint32_t e = (uint32_t)(xp->vars[v].offset + xp->vars[v].words * 4);
            GLuint b = xp->vars[v].buffer;
            if (b < 4 && e > in[14 + b]) in[14 + b] = e;
            /* the buffer's stream: its write offset advances by that stream's vertices */
            if (b < 4 && xp->vars[v].stream < 4)
                in[21] = (in[21] & ~(3u << (2 * b))) | ((uint32_t)xp->vars[v].stream << (2 * b));
        }
    }

    if (!gXfbFence) gXfbFence = [_device newFence];

    /* GL captures every stream, each into the buffers its varyings name: dispatch one
     * prefix + scatter pass per captured stream (in[20]).  Without a geometry stage, or
     * with varyings that carry no stream, this is the single stream-0 pass of the old
     * code.  The passes share one compute encoder: they are ordered inside it, and every
     * one of them reads the same write offsets (only the finish kernel advances them). */
    for (GLuint st = 0; st < MGL_XFB_MAX_STREAMS; st++)
    {
        bool has = false;
        uint32_t mask = 0;
        for (GLuint v = 0; v < xp->count; v++)
            if (xp->vars[v].words && xp->vars[v].buffer < 4 &&
                xp->vars[v].stream == st && in[4 + xp->vars[v].buffer])
            {
                has = true;
                mask |= 1u << xp->vars[v].buffer;
            }
        if (!has) continue;
        in[20] = (uint32_t)st;
        in[22] = mask;

        if (getenv("MGL_GEOM_DUMP_CNT"))
            fprintf(stderr, "geom: pack in[]: threads=%u maxv=%u outwords=%u nvars=%u"
                    " stride0=%u bufsz0=%u maxvext0=%u tfb0=[%p sz=%zu backing=%zu]\n",
                    in[0], in[1], in[2], in[3], in[4], in[8], in[14],
                    (void *)(tfb[0].buf ? tfb[0].buf->data.mtl_data : NULL),
                    (size_t)tfb[0].size,
                    (size_t)(tfb[0].buf ? tfb[0].buf->size : 0));

        if (!enc)
        {
            /* a committed buffer cannot take an encoder (Metal asserts): treat as no work */
            if (!_currentCommandBuffer ||
                _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
                return false;
            if (gPackJoinEnc && gPackJoinCb == _currentCommandBuffer)
            {
                /* join the geometry kernel's encoder (left open for us): dispatches within
                 * one encoder are ordered, which the pack's reads of the kernel's writes
                 * rely on.  The cross-buffer fence wait happened at that encoder's creation
                 * (before any dispatch), so it is not repeated here. */
                enc = gPackJoinEnc;
                gPackJoinEnc = nil;
                gPackJoinCb = nil;
                if (geomEnvRead(), gEnv.debug)
                    fprintf(stderr, "geom: pack JOINED the kernel encoder cb=%p\n",
                            (void *)(__bridge void *)_currentCommandBuffer);
            }
            else
                enc = [_currentCommandBuffer computeCommandEncoder];
            if (!enc) return false;
            [enc pushDebugGroup: @"MGL geometry xfb pack"];

            /* the counter block is shared storage, which Metal does not hazard-track across
             * command buffers: wait for the previous draw's pack before reading the write
             * offsets.  Metal forbids waiting on a fence that was updated in the same
             * command buffer, and inside one buffer the encoders are already ordered, so
             * the wait is only needed across them. */
            if (gXfbPending && gXfbFenceCb != _currentCommandBuffer)
                [enc waitForFence: gXfbFence];

            /* every per-draw input is passed by value: Metal copies it into the command
             * buffer at encode time, so no later CPU write can change what a pending
             * draw reads */

            if (gXfbBegin)
            {
                /* the seed must live in its own buffer: the ring slot is restored (and
                 * reused by the next draw) before this kernel executes */
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
        }

        [enc setComputePipelineState: gXfbPrePso];
        [enc setBytes: in length: sizeof in atIndex: 0];
        [enc setBuffer: gCnt offset: gCntOff atIndex: 1];   /* this draw's counters, not the pool's start */
        [enc setBuffer: gXbase offset: 0 atIndex: 2];
        [enc setBuffer: gXfb offset: 0 atIndex: 3];
        [enc setBuffer: gOutv offset: gOutvOff atIndex: 4]; /* the records: the scan counts closed primitives */
        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
        if ([enc respondsToSelector: @selector(memoryBarrierWithScope:)])
            [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];

        [enc setComputePipelineState: gXfbScatterPso];
        [enc setBytes: in length: sizeof in atIndex: 0];
        [enc setBytes: vars length: (NSUInteger)xp->count * MGL_GEOM_XFB_VAR_WORDS * sizeof(uint32_t) atIndex: 1];
        if ((geomEnvRead(), gEnv.xfb))
            fprintf(stderr, "geom: scatter reads out buf=%p off=%zu stream=%u mask=%x\n",
                    (void *)(__bridge void *)gOutv, gOutvOff, st, mask);
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
        if ([enc respondsToSelector: @selector(memoryBarrierWithScope:)])
            [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];
    }   /* next captured stream */

    if (!enc)
    {
        /* no stream had a captured varying: nothing to do (the CPU resolve still folds the
         * emitted counters) */
        return true;
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
            if ((geomEnvRead(), gEnv.xfb))
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
    geomMarkPools(gXfbCb);          /* the pack (this command buffer) still reads the rings */

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
    uint32_t *cnt, *xb, *hp;

    if (!(geomEnvRead(), gEnv.xfb) || !gOutv) return;
    fprintf(stderr, "geom: out records @%zu:", gOutvOff / 4);
    for (int i = 0; i < 6 * 2; i++)
    {
        uint32_t w = ((uint32_t *)gOutv.contents)[gOutvOff / 4 + i];
        fprintf(stderr, " %8.3f", *(float *)&w);
    }
    fprintf(stderr, "\n");
    if (!getenv("MGL_GEOM_DUMP_CNT")) return;
    /* post-completion internals: the geometry kernel's per-thread emitted counts, the
     * pack prefix' ordinals and the assembly header (prims the import computed) */
    cnt = gCnt ? (uint32_t *)((uint8_t *)gCnt.contents + gCntOff) : NULL;
    xb = gXbase ? (uint32_t *)gXbase.contents : NULL;
    hp = gPrim ? (uint32_t *)((uint8_t *)gPrim.contents + gPrimOff) : NULL;
    fprintf(stderr, "geom: cntcnt:");
    if (cnt) { for (int i = 0; i < 100; i++) fprintf(stderr, " %u", cnt[i]); }
    fprintf(stderr, "\ngeom: xbase :");
    if (xb) { for (int i = 0; i < 24; i++) fprintf(stderr, " %u", xb[i]); }
    fprintf(stderr, "\ngeom: hdr   :");
    if (hp) { for (int i = 0; i < 13; i++) fprintf(stderr, " %u", hp[i]); }
    fprintf(stderr, "\n");
    if (gXfb)
    {
        uint32_t *bp = (uint32_t *)gXfb.contents;
        fprintf(stderr, "geom: xfb blk:");
        for (int i = 0; i < 28; i++) fprintf(stderr, " %u", bp[i]);
        fprintf(stderr, "\n");
    }
    /* MGL_TESS_DUMP_CAP: the published tess capture records + count (post-wait) */
    if (getenv("MGL_TESS_DUMP_CAP"))
    {
        const MGLTessCapture *cap = mgl_tess_get_capture(ctx);
        if (cap && cap->buffer)
        {
            const uint32_t *cnt = (const uint32_t *)((const uint8_t *)cap->buffer +
                                                     cap->count_offset);
            const uint32_t *rec = (const uint32_t *)((const uint8_t *)cap->buffer +
                                                     cap->records_offset);
            fprintf(stderr, "tesscap: count=%u stride=%u rec0:", cnt[0], cap->stride);
            for (int i = 0; i < 20 && i < (int)cap->stride; i++)
                fprintf(stderr, " %u(%g)", rec[i], *(const float *)&rec[i]);
            fprintf(stderr, "\n");
        }
    }
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

/* the per-stream counts of a finished pack (the caller has synchronised): [21..24] the
 * block-total captured vertices per stream (what the buffers' write offsets advanced by),
 * [13..16] the block-total primitives WRITTEN per stream and [17..20] the primitives
 * EMITTED per stream ("needed") - the overflow query compares the two. */
void geom_xfb_gpu_read_streams(GLMContext ctx, uint32_t dvert[MGL_XFB_MAX_STREAMS],
                               uint32_t dprim[2 * MGL_XFB_MAX_STREAMS])
{
    uint32_t *b = (uint32_t *)(gXfb ? gXfb.contents : NULL);

    (void)ctx;
    for (GLuint s = 0; s < MGL_XFB_MAX_STREAMS; s++)
    {
        dvert[s] = b ? b[21 + s] : 0;
        dprim[s] = b ? b[13 + s] : 0;
        dprim[s + MGL_XFB_MAX_STREAMS] = b ? b[17 + s] : 0;
    }
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
    m->query_begin = geomQueryBeginC;
    m->query_read = geomQueryReadC;
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

/* glBindImageTexture binds exactly ONE mip level of a texture (and, when layered is
 * GL_FALSE on an arrayed target, ONE layer), so the MSL image argument must see a view
 * of that level / slice: handing it the base texture makes imageSize() report level-0
 * dimensions and imageStore land at level 0 (KHR-GL46.shader_image_size binds level 1/2
 * of mip chains).  A layered binding keeps every slice of the level, with the texture
 * type matching the arrayed image type the shader declared (texture2d_array,
 * texturecube_array, texture3d ...); a non-layered binding of an arrayed target becomes
 * a single-slice 2D/1D view for the shader's non-arrayed image type.  Returns `base`
 * unchanged when no slicing is needed (level-0, whole texture, buffer textures). */
- (id<MTLTexture>) geomImageUnitView: (const ImageUnit *) iu base: (id<MTLTexture>) base
{
    MTLTextureType type, view_type;
    NSUInteger level, layer, levels, array_len;
    NSRange level_range, slice_range;

    if (!base)
        return nil;
    type = base.textureType;
    if (type == MTLTextureTypeTextureBuffer)
        return base;                       /* buffer textures have no levels or layers */

    levels = base.mipmapLevelCount;
    array_len = base.arrayLength;
    level = (iu->level > 0) ? (NSUInteger)iu->level : 0;
    layer = (iu->layer > 0) ? (NSUInteger)iu->layer : 0;
    if (level >= levels) level = levels ? levels - 1 : 0;
    level_range = NSMakeRange(level, 1);

    BOOL layered = (iu->layered != GL_FALSE);

    switch (type)
    {
        case MTLTextureType2D:
            if (level == 0) return base;
            view_type = MTLTextureType2D;
            slice_range = NSMakeRange(0, 1);
            break;
        case MTLTextureType2DMultisample:
            /* MS textures have a single level; a layered binding is the whole texture */
            return base;
        case MTLTextureType1D:
            if (level == 0) return base;
            view_type = MTLTextureType1D;
            slice_range = NSMakeRange(0, 1);
            break;
        case MTLTextureType1DArray:
            if (layered)
            {
                if (level == 0) return base;
                view_type = MTLTextureType1DArray;
                slice_range = NSMakeRange(0, array_len);
            }
            else
            {
                view_type = MTLTextureType1D;
                slice_range = NSMakeRange(MIN(layer, array_len ? array_len - 1 : 0), 1);
            }
            break;
        case MTLTextureType2DArray:
        case MTLTextureType2DMultisampleArray:
            if (layered)
            {
                if (level == 0 && type == MTLTextureType2DArray) return base;
                view_type = type;
                slice_range = NSMakeRange(0, array_len);
            }
            else
            {
                view_type = (type == MTLTextureType2DArray) ? MTLTextureType2D : MTLTextureType2DMultisample;
                slice_range = NSMakeRange(MIN(layer, array_len ? array_len - 1 : 0), 1);
            }
            break;
        case MTLTextureType3D:
            if (layered)
            {
                /* a texture3d view spans the LEVEL's depth (imageSize().z): the depth
                 * halves per level, so the base texture's full depth would overflow
                 * Metal's view validation */
                NSUInteger depth = base.depth >> level;

                if (level == 0) return base;
                if (!depth) depth = 1;
                view_type = MTLTextureType3D;
                slice_range = NSMakeRange(0, depth);
            }
            else
            {
                /* GL wants an image2D view of one 3D slice, but a 3D texture can only
                 * alias as 3D (Metal texture-view rules) — hand the whole texture and
                 * let the MSL's texture2d argument read the slice-0 plane.  Documented
                 * emulation gap (non-layered 3D image bindings read the wrong plane). */
                return base;
            }
            break;
        case MTLTextureTypeCube:
            if (layered)
            {
                if (level == 0) return base;
                view_type = MTLTextureTypeCube;
                slice_range = NSMakeRange(0, 6);
            }
            else
            {
                view_type = MTLTextureType2D;
                slice_range = NSMakeRange(MIN(layer, 5), 1);
            }
            break;
        case MTLTextureTypeCubeArray:
            if (layered)
            {
                if (level == 0) return base;
                view_type = MTLTextureTypeCubeArray;
                slice_range = NSMakeRange(0, array_len * 6);
            }
            else
            {
                view_type = MTLTextureType2D;
                slice_range = NSMakeRange(MIN(layer, array_len * 6 ? array_len * 6 - 1 : 0), 1);
            }
            break;
        default:
            return base;
    }

    id<MTLTexture> view = nil;
    @try {
        view = [base newTextureViewWithPixelFormat: base.pixelFormat
                                        textureType: view_type
                                             levels: level_range
                                           slices: slice_range];
    } @catch (NSException *e) {
        NSLog(@"MGL WARNING: image-unit view (level %d layer %d layered %d) failed: %@",
              iu->level, iu->layer, iu->layered, e);
        return base;
    }
    return view ? view : base;
}

/* the generated variants number their texture arguments themselves (geom_build_pins):
 * the walks look up the Metal slot by resource name; msl_binding is only the fallback */
static int geomTexSlot(const MGLGeomTexSlot *slots, int n, const char *name)
{
    if (!name) return -1;
    for (int i = 0; i < n; i++)
        if (!strcmp(slots[i].name, name)) return slots[i].msl;
    return -1;
}

/* the capture pass runs the vertex stage as a render encoder of its own: realize the
 * stage's textures before that encoder exists (a deferred upload opens a blit encoder,
 * which Metal forbids while another encoder is encoding) */
- (void) geomPrepareStageTextures: (int) stage
{
    struct { int spvc_type; bool image; } types[] = {
        { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
        { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
        { 0, false }
    };

    if (!STATE(program)) return;
    for (int t = 0; types[t].spvc_type; t++)
    {
        GLuint n = [self getProgramBindingCount: stage type: types[t].spvc_type];

        for (GLuint i = 0; i < n; i++)
        {
            SpirvResource *sres = &STATE(program)->spirv_resources_list[stage][types[t].spvc_type].list[i];
            int b = [self getProgramBinding: stage type: types[t].spvc_type index: i];
            int cnt = sres->array_size ? (int)sres->array_size : 1;

            for (int e = 0; e < cnt && b >= 0; e++)
            {
                int unit = b + e;

                if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;
                Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                              : mgl_texture_for_sampler(ctx, unit, sres->target);
                if (tex) [self bindMTLTexture: tex];
            }
        }
    }
}

/* bind one stage's textures to the capture pass's render encoder (vertex slots only: the
 * pass has no fragment function).  Consecutive units feed array elements (GL sampler
 * convention: base..base+N-1 at consecutive Metal argument slots).  The capture variant's
 * argument indices come from its own pinning (gp->tex_slots_vs). */
- (void) geomBindRenderStageTextures: (id<MTLRenderCommandEncoder>) enc stage: (int) stage
                                geomProgram: (MGLGeomProgram *) gp
{
    struct { int spvc_type; bool image; } types[] = {
        { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
        { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
        { 0, false }
    };

    if (!STATE(program)) return;
    for (int t = 0; types[t].spvc_type; t++)
    {
        GLuint n = [self getProgramBindingCount: stage type: types[t].spvc_type];

        for (GLuint i = 0; i < n; i++)
        {
            SpirvResource *sres = &STATE(program)->spirv_resources_list[stage][types[t].spvc_type].list[i];
            int spirv_binding = [self getProgramBinding: stage type: types[t].spvc_type index: i];
            NSUInteger mtl_index = (NSUInteger)geomTexSlot(gp->tex_slots_vs, gp->n_tex_slots_vs,
                                                           sres->name ? sres->name : "");
            if (mtl_index == (NSUInteger)-1) mtl_index = (NSUInteger)sres->msl_binding;
            int cnt = sres->array_size ? (int)sres->array_size : 1;

            if (spirv_binding < 0) continue;
            for (int e = 0; e < cnt; e++)
            {
                int unit = spirv_binding + e;

                if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;
                Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                              : mgl_texture_for_sampler(ctx, unit, sres->target);
                if (!tex || !tex->mtl_data) continue;
                id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
                if (!types[t].image)
                {
                    id<MTLSamplerState> sampler = [self texparamSamplerForUnit: (GLuint)unit texture: tex];
                    texture = [self texparamTextureForTex: tex base: texture];
                    if (sampler) [enc setVertexSamplerState: sampler atIndex: mtl_index + e];
                }
                else
                {
                    /* an image unit binds one mip level / layer (glBindImageTexture) */
                    texture = [self geomImageUnitView: &STATE(image_units[unit]) base: texture];
                }
                [enc setVertexTexture: texture atIndex: mtl_index + e];
                if (gEnv.debug)
                    fprintf(stderr, "geom: debug vs capture %s[%u] '%s' unit %d at msl %lu x%d\n",
                            types[t].image ? "image" : "tex", (unsigned)i,
                            sres->name ? sres->name : "?", unit, (unsigned long)mtl_index, cnt);
            }
        }
    }
}

- (bool) geomBindStageBuffers: (id<MTLComputeCommandEncoder>) enc geomProgram: (MGLGeomProgram *) gp
{
    BufferMapList list;
    GLuint count;

    memset(&list, 0, sizeof(list));

    /* the same mapping the renderer builds for a normal stage; the list already expands
     * instance arrays (one entry per element at consecutive slots), matching the flat
     * layout geom_build_pins assigns to the generated kernels */
    if (![self mapGLBuffersToMTLBufferMap: &list stage: _GEOMETRY_SHADER]) return false;
    if (getenv("MGL_SSBO_DEBUG"))
    {
        NSLog(@"MGL geom: gs kernel resource list: count=%u", list.count);
        for (int i = 0; i < (int)list.count; i++)
            NSLog(@"MGL geom:   gs slot[%d]: base=%u buf=%p off=%ld size=%ld data=%p mtl=%p", i,
                  list.buffers[i].buffer_base_index, (void *)list.buffers[i].buf,
                  (long)list.buffers[i].offset,
                  (long)(list.buffers[i].buf ? list.buffers[i].buf->size : -1),
                  list.buffers[i].buf ? (void *)list.buffers[i].buf->data.buffer_data : NULL,
                  list.buffers[i].buf ? (void *)list.buffers[i].buf->data.mtl_data : NULL);
    }
    if (ctx->state.dirty_bits & DIRTY_BUFFER)
    {
        [self updateDirtyBaseBufferList: &list];
        ctx->state.dirty_bits &= ~DIRTY_BUFFER;
    }

    for (int i = 0; i < (int)list.count; i++)
    {
        Buffer *b = list.buffers[i].buf;
        GLintptr off = list.buffers[i].offset;
        if (!b) return false;
        /* glBindBufferRange binds a *slice* of the buffer: the entry's offset is where the
         * resource starts.  Binding offset 0 instead made every range binding of one buffer
         * alias the buffer's start (KHR-GL46.geometry_shader.api.max_shader_storage_blocks
         * binds one SSBO eight times at consecutive offsets — all eight writes landed on
         * element 0).  The render paths already honour the offset (MGLRenderer+Draw.m). */
        if (off < 0 || (GLsizeiptr)off > b->size) off = 0;
        /* the program's default-uniform block lives in CPU memory until a draw flushes
         * it; materialise it the way the render encoders do (updateDirtyBuffer, or the
         * setBytes fast path for small buffers) */
        if (!b->data.mtl_data && b->data.dirty_bits)
            if (![self updateDirtyBuffer: b]) return false;
        if (!b->data.mtl_data) [self bindMTLBuffer: b];
        /* a storage buffer that is still CPU-shadowed must become a real MTLBuffer:
         * the setBytes fallback below binds inline bytes, and the kernel's SSBO writes
         * would vanish (KHR-GL46.shader_storage_buffer_object basic-atomic-case1) */
        if (!b->data.mtl_data)
        {
            if (!b->data.buffer_data || b->size <= 0) return false;
            [enc setBytes: (const void *)((uint8_t *)b->data.buffer_data + off)
                 length: (NSUInteger)(b->size - (size_t)off)
                 atIndex: (NSUInteger)i];
            b->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
            continue;
        }
        [enc setBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                offset: (NSUInteger)off
               atIndex: (NSUInteger)i];
        if (getenv("MGL_SSBO_DEBUG"))
            NSLog(@"MGL geom: gs kernel bind i=%d off=%ld len=%lu base=%u", i, (long)off,
                  (unsigned long)((__bridge id<MTLBuffer>)b->data.mtl_data).length,
                  list.buffers[i].buffer_base_index);
        mgl_buffers_mark_gpu_use(ctx, b, true);
    }

    /* textures and images of the geometry stage: the SPIR-V binding selects the GL texture
     * unit; the kernel's Metal argument index comes from its own SPIRV-Cross pinning
     * (gp->tex_slots_gs, recorded by geom_build_pins — the stage list's msl_binding is a
     * different numbering for the geometry stage, where program.c never refines it with a
     * Metal GS compile).  A sampler ARRAY spans consecutive Metal slots fed by consecutive
     * units (base..base+N-1, MGL's sampler-uniform convention).  Sampler objects bound to
     * the unit override the texture's own sampler state. */
    {
        struct { int spvc_type; bool image; } types[] = {
            { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
            { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
            { 0, false }
        };

        for (int t = 0; types[t].spvc_type; t++)
        {
            count = [self getProgramBindingCount: _GEOMETRY_SHADER type: types[t].spvc_type];
            for (GLuint i = 0; i < count; i++)
            {
                SpirvResource *sres = &STATE(program)->spirv_resources_list[_GEOMETRY_SHADER][types[t].spvc_type].list[i];
                int spirv_binding = [self getProgramBinding: _GEOMETRY_SHADER type: types[t].spvc_type index: i];
                NSUInteger mtl_index = (NSUInteger)geomTexSlot(gp->tex_slots_gs, gp->n_tex_slots_gs,
                                                               sres->name ? sres->name : "");
                if (mtl_index == (NSUInteger)-1) mtl_index = (NSUInteger)sres->msl_binding;
                int n = 1;

                for (int a = 0; a < gp->n_tex_arrays; a++)
                    if (!strcmp(gp->tex_arrays[a].name, sres->name ? sres->name : ""))
                    { n = gp->tex_arrays[a].n; break; }

                if (spirv_binding < 0) continue;

                for (int e = 0; e < n; e++)
                {
                    int unit = spirv_binding + e;
                    if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;

                    Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                                  : mgl_texture_for_sampler(ctx, unit, sres->target);
                    if (!tex) continue;
                    if (![self bindMTLTexture: tex]) continue;
                    id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
                    if (!texture) continue;
                    if (!types[t].image)
                    {
                        id<MTLSamplerState> sampler = [self texparamSamplerForUnit: (GLuint)unit texture: tex];
                        texture = [self texparamTextureForTex: tex base: texture];
                        if (sampler) [enc setSamplerState: sampler atIndex: mtl_index + e];
                    }
                    else
                    {
                        /* an image unit binds one mip level / layer (glBindImageTexture) */
                        texture = [self geomImageUnitView: &STATE(image_units[unit]) base: texture];
                    }
                    [enc setTexture: texture atIndex: mtl_index + e];
                    if (gEnv.debug && e == 0)
                        fprintf(stderr, "geom: debug gs %s[%u] '%s' unit %d bound at msl %lu x%d type=%lu len=%lu\n",
                                types[t].image ? "image" : "tex", (unsigned)i,
                                sres->name ? sres->name : "?", spirv_binding,
                                (unsigned long)mtl_index, n,
                                texture ? (unsigned long)texture.textureType : 999,
                                texture ? (unsigned long)texture.width : 0);
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
    /* a draw without an element-array binding has no indices to read back: the prep
     * kernel derives them from `first`, so the GPU route is always available (returning
     * false here used to push every non-indexed draw onto the CPU assembly path) */
    if (!eb) return true;
    if (!eb->data.mtl_data) [self bindMTLBuffer: eb];
    return eb->data.mtl_data != NULL;
}

/* reserve `need` bytes: *start receives this draw's range and the pool advances past it, so
 * the next draw can never land on it.  When the range would not fit, start a NEW (larger,
 * zero-filled) buffer instead of wrapping to offset 0 — a silent wrap could hand a draw a
 * range whose previous user is still pending on the GPU. */
/* commit and complete the current command buffer (geomSubAlloc at its memory cap).  It is
 * only called before any encoder of the current draw exists, so nothing is lost. */
- (void) geomPoolFlushAndWait: (id<MTLCommandBuffer>) cur
{
    /* light flush: nothing of the current draw is encoded yet, so ending the previous
     * draw's encoder and committing is enough (the full buffers-domain flush measured
     * far slower) */
    [self endRenderEncoding];
    if (cur.status < MTLCommandBufferStatusCommitted)
        [self commitCommandBufferWithAGXRecovery: cur];
    if (cur.status != MTLCommandBufferStatusError)
        [cur waitUntilCompleted];
}

/* record a consumer of a slot's live ranges, keeping the one that completes LAST: cb is
 * the current command buffer, which Metal executes after every buffer already committed,
 * so it replaces any committed consumer; a not-yet-committed recorded consumer is the
 * current buffer of an earlier mark and stays (conservative: reclaim is delayed, never
 * made early).  The pointer is retained — it can outlive the renderer's _currentCommandBuffer. */
static void geomPoolConsumer(MGLGeomPool *pool, unsigned c, id<MTLCommandBuffer> cb)
{
    void *old = pool->cb[c];

    if (!cb) return;
    if (old)
    {
        id<MTLCommandBuffer> o = (__bridge id<MTLCommandBuffer>)old;

        if (o == cb) return;
        if (o.status < MTLCommandBufferStatusCommitted) return;
        CFRelease(old);
    }
    pool->cb[c] = (void *)CFBridgingRetain(cb);
}

/* record that command buffer cb consumes every slot's live ranges (the draw's kernels, the
 * raster pass, an xfb pack — consumers can be encoded into a later command buffer than the
 * one the range was allocated in, e.g. when begin_draw commits) */
static void geomMarkPools(id<MTLCommandBuffer> cb)
{
    if (!cb) return;
    for (unsigned k = 0; k < MGL_GEOM_POOL_N; k++)
    {
        geomPoolConsumer(&gPrimPool, k, cb);
        geomPoolConsumer(&gVsoutPool, k, cb);
        geomPoolConsumer(&gOutvPool, k, cb);
        geomPoolConsumer(&gPrepPool, k, cb);
        geomPoolConsumer(&gCntPool, k, cb);
        gPrimPool.pin[k] = false; gVsoutPool.pin[k] = false; gOutvPool.pin[k] = false;
        gPrepPool.pin[k] = false; gCntPool.pin[k] = false;
    }
}

/* the previous draw's pins are stale: either it marked its consumers at the end (its
 * encoders exist) or it bailed out before encoding anything (its ranges were never
 * consumed).  The next draw's allocations start with a clean slate. */
static void geomPoolResetPins(void)
{
    for (unsigned k = 0; k < MGL_GEOM_POOL_N; k++)
    {
        gPrimPool.pin[k] = false; gVsoutPool.pin[k] = false; gOutvPool.pin[k] = false;
        gPrepPool.pin[k] = false; gCntPool.pin[k] = false;
    }
}

/* is the consumer recorded for a slot done (nil counts: no consumer ever ran)? */
static bool geomPoolSlotDone(MGLGeomPool *pool, unsigned c)
{
    void *p = pool->cb[c];

    if (!p) return true;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p;
    return cb.status >= MTLCommandBufferStatusCompleted;
}

/* hand out a byte range of `need` bytes.  Returns the backing buffer and the range's
 * offset in *start; nil only when Metal refuses the allocation. */
static id<MTLBuffer> geomSubAlloc(id r, id<MTLDevice> device, MGLGeomPool *pool, size_t need, size_t *start,
                                  id<MTLCommandBuffer> cur)
{
    if (need == 0) need = 4;
    need = (need + (size_t)MGL_GEOM_POOL_ALIGN - 1) & ~((size_t)MGL_GEOM_POOL_ALIGN - 1);
    (void) geomEnvRead();

    /* 1. a slot with room: hand out the next disjoint range.  No wait, no reuse — always
     *    safe however far the GPU is behind. */
    for (unsigned k = 0; k < gPoolN; k++)
    {
        unsigned c = (pool->cur + k) % gPoolN;
        if (pool->buf[c] && pool->used[c] + need <= pool->buf[c].length)
        {
            pool->cur = c;
            *start = pool->used[c];
            pool->used[c] += need;
            geomPoolConsumer(pool, c, cur);
            pool->pin[c] = true;
            return pool->buf[c];
        }
    }

    /* 2. reclaim a slot whose consumer completed (its whole buffer is dead); grow it when
     *    even empty it is too small — growth replaces the buffer object, and the old one is
     *    retained by the encoders still reading it, so pending consumers stay correct. */
    for (unsigned k = 1; k <= gPoolN; k++)
    {
        unsigned c = (pool->cur + k) % gPoolN;
        if (pool->pin[c] || !geomPoolSlotDone(pool, c)) continue;

        size_t room = pool->buf[c] ? pool->buf[c].length : 0;
        if (room < need)
        {
            size_t sz = 4096, want = need * 2;

            if (want > gPoolCap && need <= gPoolCap) want = gPoolCap;   /* soft cap */
            while (sz < want) sz *= 2;
            if (sz > gPoolCap && need > gPoolCap) sz = need;            /* one draw still gets its memory */
            if (gEnv.draw)
                fprintf(stderr, "geom: trace pool %p slot %u grow need=%zu -> %zu\n",
                        (void *)pool, c, need, sz);
            id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
            if (!grown) continue;               /* try the next slot; the old buffer stays usable */
            pool->buf[c] = grown;
            room = sz;
        }
        pool->cur = c;
        pool->used[c] = 0;
        *start = 0;
        pool->used[c] = need;
        geomPoolConsumer(pool, c, cur);
        pool->pin[c] = true;
        return pool->buf[c];
    }

    /* 3. every slot is full of live ranges: the GPU is several draws behind.  Wait for the
     *    oldest pending consumer — with in-order completion that retires the rest.  If it is
     *    the current, still-uncommitted buffer, commit it first (nothing of this draw is
     *    encoded yet, so nothing is lost). */
    {
        id<MTLCommandBuffer> wait_cb = nil;

        for (unsigned k = 1; k <= gPoolN; k++)
        {
            unsigned c = (pool->cur + k) % gPoolN;
            id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)pool->cb[c];

            if (!cb || cb.status >= MTLCommandBufferStatusCompleted) continue;
            if (cb.status >= MTLCommandBufferStatusCommitted) { wait_cb = cb; break; }
            wait_cb = cb;       /* uncommitted: the current buffer, or an older empty one */
        }
        if (gEnv.draw)
            fprintf(stderr, "geom: trace pool %p wait (all %u slots full) cb=%p\n",
                    (void *)pool, gPoolN, (void *)(wait_cb ? (__bridge void *)wait_cb : NULL));
        if (wait_cb && wait_cb != cur)
        {
            if (wait_cb.status < MTLCommandBufferStatusCommitted)
                [(id) r geomPoolFlushAndWait: wait_cb];
            else
                [wait_cb waitUntilCompleted];
        }
        else if (cur && cur.status < MTLCommandBufferStatusCommitted)
        {
            [(id) r geomPoolFlushAndWait: cur];
        }

        /* at least one consumer completed now: reclaim (step 2 again, without the wait) */
        for (unsigned k = 1; k <= gPoolN; k++)
        {
            unsigned c = (pool->cur + k) % gPoolN;
            if (pool->pin[c] || !geomPoolSlotDone(pool, c)) continue;

            size_t room = pool->buf[c] ? pool->buf[c].length : 0;
            if (room < need)
            {
                size_t sz = 4096, want = need * 2;

                if (want > gPoolCap && need <= gPoolCap) want = gPoolCap;
                while (sz < want) sz *= 2;
                if (sz > gPoolCap && need > gPoolCap) sz = need;
                id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
                if (!grown) continue;
                pool->buf[c] = grown;
                room = sz;
            }
            pool->cur = c;
            pool->used[c] = 0;
            *start = 0;
            pool->used[c] = need;
            geomPoolConsumer(pool, c, cur);
            pool->pin[c] = true;
            return pool->buf[c];
        }

        /* unreachable unless every consumer is still pending after the wait: grow the
         * current slot (the old buffer keeps the pending ranges alive) */
        {
            unsigned c = pool->cur % gPoolN;
            size_t sz = 4096, want = need * 2;

            while (sz < want) sz *= 2;
            if (gEnv.draw)
                fprintf(stderr, "geom: trace pool %p slot %u forced grow need=%zu -> %zu\n",
                        (void *)pool, c, need, sz);
            id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
            if (!grown) return nil;
            pool->buf[c] = grown;
            pool->used[c] = 0;
            *start = 0;
            pool->used[c] = need;
            geomPoolConsumer(pool, c, cur);
            pool->pin[c] = true;
            return pool->buf[c];
        }
    }
}

/* RASTERIZER_DISCARD optimization (GL 4.6 10.4.4): with discard on and no transform
 * feedback, a geometry stage whose only outputs feed rasterization is unobservable —
 * unless it writes images / SSBOs / atomics, or a pre-raster query is counting it
 * (PRIMITIVES_GENERATED and the stage-invocation counters stay live under discard). */
static bool geom_discard_skippable(GLMContext ctx, Program *program)
{
    static const int side[3] = { SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                 SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
                                 SPVC_RESOURCE_TYPE_ATOMIC_COUNTER };

    if (ctx->state.caps.rasterizer_discard != GL_TRUE) return false;
    if (mgl_xfb_capturing(ctx)) return false;

    if (ctx->state.ext_queries)
    {
        struct MGLExtState_queries *S = ctx->state.ext_queries;

        for (int i = MGL_QB_PRIMITIVES_GENERATED; i <= MGL_QB_GEOMETRY_SHADER_PRIMITIVES_EMITTED; i++)
            if (S->active[i]) return false;
    }

    if (program)
    {
        for (int t = 0; t < 3; t++)
            if (program->spirv_resources_list[_GEOMETRY_SHADER][side[t]].count)
                return false;
    }
    return true;
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
    /* RASTERIZER_DISCARD still runs the whole vertex + geometry pipeline when anything is
     * observable from it (GL 4.6 10.4.4: the stage's side effects — image / SSBO / atomic
     * writes — are observable, transform feedback keeps recording, and the pre-raster
     * query counters stay live); geom_discard_skippable decides when the stage's only
     * consumer (the raster pass below) is off and nothing else watches. */
    if (geom_discard_skippable(ctx, program)) return false;

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
        /* the prep kernel writes the lists: the capture list holds one entry per vertex
         * of every restart-free run (<= the drawn vertex count), and the primitive slots
         * sit behind it at HDR + cap_idx - the geometry kernel binds that same offset */
        cap_idx = (size_t)(d->count_bound ? d->count_bound : (d->count > 0 ? d->count : 1));
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
     * vertex range is only known on the GPU, so size by the index count.  The CPU route's */
    prim_bytes = (MGL_GEOM_HDR_WORDS + cap_idx + cap_prim * gp->in_verts) * sizeof(uint32_t);
    /* CPU route's vcount comes from the index range it read (vmax - vmin + 1): a corrupted element
     * vcount comes from the index range it read (vmax - vmin + 1): a corrupted element
     * store (or an index far outside the attribute buffers) would size the capture ring at
     * gigabytes and the allocation failure then takes the whole program's pipelines down
     * with it, so the same budget cap the GPU route uses bounds it here too.  The draw's
     * own output is undefined in that case anyway; the point is to stay standing. */
    if (!gpu_assembly)
    {
        size_t max_vsout = MGL_GEOM_BUDGET_BYTES / (gp->in_words ? (size_t)gp->in_words * sizeof(uint32_t) : 4u);
        size_t want = (size_t)d->vcount * (size_t)(d->instances > 0 ? d->instances : 1);
        if (want > max_vsout)
        {
            if ((geomEnvRead(), gEnv.draw))
                fprintf(stderr, "geom: trace vcount %u capped (%zu > %zu)\n", d->vcount, want, max_vsout);
            want = max_vsout;
        }
        vsout_bytes = (8 + want * gp->in_words) * sizeof(uint32_t);
    }
    else
        vsout_bytes = (8 + (size_t)bound * (size_t)(d->instances > 0 ? d->instances : 1)
                           * gp->in_words) * sizeof(uint32_t);

    /* every draw gets its own range of the record buffer: a later draw's geometry kernel
     * must never overwrite the records an earlier draw's pack still has to read */
    if ((geomEnvRead(), gEnv.draw))
        fprintf(stderr, "geom: trace sizing bound=%u inst=%d vcount=%u outv_cap=%zu idx_cap=%zu cap_idx=%zu cap_prim=%zu "
                        "prim_bytes=%zu vsout_bytes=%zu threads=%llu in_words=%u out_words=%u maxverts=%u\n",
                (unsigned)bound, d->instances, d->vcount, outv_cap, idx_cap, cap_idx, cap_prim, prim_bytes, vsout_bytes,
                (unsigned long long)threads, gp->in_words, gp->out_words, gp->max_vertices);
    geomPoolResetPins();
    gOutv = geomSubAlloc(self, _device, &gOutvPool, (size_t)outv_cap * gp->out_words * sizeof(uint32_t), &gOutvOff, _currentCommandBuffer);
    if (!gOutv) { geomReleaseStatics(); return false; }
    /* [ raster list | per-thread strip scratch | per-thread counts and scanned bases ] */
    gIdx = geomBuffer(_device, gIdx, ((size_t)idx_cap * 2 + (size_t)threads * 2) * sizeof(uint32_t), &gIdxBytes);
    gCtl = geomBuffer(_device, gCtl, 4096, NULL);
    /* the prep kernel's outputs (the capture draw arguments and the kernel's dispatch count)
     * are read by this draw's capture pass and geometry dispatch: a later draw's prep kernel
     * must not overwrite them first */
    gPrep = geomSubAlloc(self, _device, &gPrepPool, MGL_GEOM_PREP_BYTES, &gPrepOff, _currentCommandBuffer);
    if (!gPrep) { geomReleaseStatics(); return false; }
    gCnt = geomSubAlloc(self, _device, &gCntPool, 4096 + (size_t)threads * sizeof(uint32_t), &gCntOff, _currentCommandBuffer);
    if (!gCnt) { geomReleaseStatics(); return false; }
    if (!gOutv || !gIdx || !gCtl || !gPrep || !gCnt) { geomReleaseStatics(); return false; }


    /* the primitive slots / capture index list and the captured VS records are shared by the
     * prep kernel, the capture pass, the geometry kernel and the pack: every draw gets its
     * own range so a later draw's kernels cannot overwrite them first */
    gPrim = geomSubAlloc(self, _device, &gPrimPool, prim_bytes, &gPrimOff, _currentCommandBuffer);
    if (!gPrim) { geomReleaseStatics(); return false; }
    gVsout = geomSubAlloc(self, _device, &gVsoutPool, vsout_bytes, &gVsoutOff, _currentCommandBuffer);
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
        /* the transform-feedback pack below joins this encoder: its counter-block fence
         * wait must precede the encoder's first dispatch (Metal applies encoder-level
         * fence waits at encoder granularity), so it goes here, not in the pack */
        if (gXfbPending && gXfbFenceCb != _currentCommandBuffer)
            [enc waitForFence: gXfbFence];
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
    }
    else if (gpu_assembly)
    {
        id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
        Buffer *eb = d->indexed ? (ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL) : NULL;
        bool have_eb = false;

        if (!enc) return false;
        /* same as the tess-import branch: the pack joins this encoder, so its counter-block
         * fence wait belongs here, before the first dispatch */
        if (gXfbPending && gXfbFenceCb != _currentCommandBuffer)
            [enc waitForFence: gXfbFence];
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
        /* the vertex-attrib plan (which Metal buffer index each attribute reads) is built
         * from the buffer map of the program's VERTEX stage — build it NOW, under the
         * program, so the descriptor below matches the buffers bound after the encoder
         * (the raster pass's map may be one draw behind, e.g. on the first draw or after
         * the program's uniform count changed the map's layout) */
        if ([self mapGLBuffersToMTLBufferMap: &ctx->state.vertex_buffer_map_list stage: _VERTEX_SHADER])
        {
            if (ctx->state.dirty_bits & DIRTY_BUFFER)
            {
                [self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list];
                ctx->state.dirty_bits &= ~DIRTY_BUFFER;
            }
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

        /* realize the vertex stage's textures now: their deferred uploads would otherwise
         * try to open a blit encoder once this pass's render encoder exists */
        ctx->state.program = program;
        [self geomPrepareStageTextures: _VERTEX_SHADER];

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
                GLintptr off = ctx->state.vertex_buffer_map_list.buffers[i].offset;
                if (getenv("MGL_GEOM_TRACE"))
                    NSLog(@"MGL geom trace: capture bind i=%d buf=%p mtl=%p size=%d data=%p dirty=%llx w0=%08x w1=%08x",
                          i, b, b ? b->data.mtl_data : NULL, b ? (int)b->size : -1,
                          b ? b->data.buffer_data : NULL, b ? (unsigned long long)b->data.dirty_bits : 0ull,
                          (b && b->data.buffer_data) ? *(unsigned *)b->data.buffer_data : 0,
                          (b && b->data.buffer_data && b->size >= 8) ? ((unsigned *)b->data.buffer_data)[1] : 0);
                if (!b) continue;
                if (off < 0 || (GLsizeiptr)off > b->size) off = 0;
                /* the program's default-uniform block lives in CPU memory until a draw
                 * flushes it; materialise it the way the render encoders do */
                if (!b->data.mtl_data && b->data.dirty_bits)
                    if (![self updateDirtyBuffer: b]) continue;
                if (!b->data.mtl_data) [self bindMTLBuffer: b];
                if (!b->data.mtl_data)
                {
                    if (!b->data.buffer_data || b->size <= 0) continue;
                    [enc setVertexBytes: (const void *)((uint8_t *)b->data.buffer_data + off)
                                 length: (NSUInteger)(b->size - (size_t)off)
                                atIndex: (NSUInteger)i];
                    b->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
                    continue;
                }
                [enc setVertexBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                              offset: (NSUInteger)off
                             atIndex: (NSUInteger)i];
                mgl_buffers_mark_gpu_use(ctx, b, true);
            }
        }

        /* the vertex stage's textures sample in the capture variant just like in the
         * raster pass (GL allows the VS to sample; the CTS limits tests use 16 units) */
        [self geomBindRenderStageTextures: enc stage: _VERTEX_SHADER geomProgram: gp];
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
        /* realize the geometry stage's textures before any encoder of this kernel exists:
         * a deferred upload (texdataEncodeUpload) opens a blit encoder, and Metal forbids
         * that while a compute encoder is encoding (array elements cover consecutive units) */
        {
            struct { int spvc_type; bool image; } types[] = {
                { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
                { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
                { 0, false }
            };
            /* the walk reads STATE(program)'s resource lists: a program-pipeline draw
             * enters geomDraw with ctx->state.program == NULL (the composite is only
             * published inside the kernel section below), and without the switch the
             * walk binds nothing — the first deferred upload would then fire from
             * geomBindStageBuffers, into the already-open compute encoder */
            Program *saved_tex_program = ctx->state.program;

            ctx->state.program = program;
            for (int t = 0; types[t].spvc_type; t++)
            {
                GLuint n = [self getProgramBindingCount: _GEOMETRY_SHADER type: types[t].spvc_type];

                for (GLuint i = 0; i < n; i++)
                {
                    SpirvResource *sres = &STATE(program)->spirv_resources_list[_GEOMETRY_SHADER][types[t].spvc_type].list[i];
                    int b = [self getProgramBinding: _GEOMETRY_SHADER type: types[t].spvc_type index: i];
                    int cnt = 1;

                    for (int a = 0; a < gp->n_tex_arrays; a++)
                        if (!strcmp(gp->tex_arrays[a].name, sres->name ? sres->name : ""))
                        { cnt = gp->tex_arrays[a].n; break; }

                    for (int e = 0; e < cnt && b >= 0; e++)
                    {
                        int unit = b + e;

                        if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;
                        Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                                      : mgl_texture_for_sampler(ctx, unit, sres->target);
                        if (tex) [self bindMTLTexture: tex];
                    }
                }
            }
            ctx->state.program = saved_tex_program;
        }

        /* the tess import kernel's encoder is still open: joining it keeps the import's
         * writes ordered before this kernel (and before its indirect dispatch) */
        id<MTLComputeCommandEncoder> enc = kenc ? kenc : [_currentCommandBuffer computeCommandEncoder];
        if (!enc) return false;
        if (!kenc && gXfbPending && gXfbFenceCb != _currentCommandBuffer)
        {
            /* fresh encoder that the pack may join later: take the counter-block fence wait
             * here, before the first dispatch */
            [enc waitForFence: gXfbFence];
        }

        Program *saved_program = ctx->state.program;
        ctx->state.program = program;

        [enc pushDebugGroup: @"MGL geometry kernel"];

        /* zero the list-order pass' per-thread counts up front: threads beyond this
         * draw's dispatch grid (an indirect draw's exact group count is smaller than the
         * worst-case bound) never run the kernel to publish a zero themselves */
        {
            uint32_t zero_in[1] = { (uint32_t)threads };
            NSUInteger zwg = 32;
            uint64_t zgroups = (threads * 2 + zwg - 1) / zwg;
            if (!zgroups) zgroups = 1;
            [enc setComputePipelineState: gLzeroPso];
            [enc setBytes: zero_in length: sizeof zero_in atIndex: 0];
            [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * 2 * sizeof(uint32_t) atIndex: 1];
            [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)zgroups, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(zwg, 1, 1)];
        }

        /* the counters and indirect arguments start at zero for this draw (a kernel,
         * not a CPU write: the buffer may still be read by an earlier command buffer).
         * This kernel uses buffer(0), so it MUST be dispatched before the GS kernel's own
         * bindings: a compute encoder's bindings persist across setComputePipelineState,
         * and binding gCtl at slot 0 after the stage buffers would shadow the kernel's
         * first resource (e.g. element 0 of a uniform-block instance array) */
        [enc setComputePipelineState: gInitPso];
        [enc setBuffer: gCtl offset: 0 atIndex: 0];
        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

        [enc setComputePipelineState: gGsPso];
        /* the primitive slots start behind the capture index list (the prep kernel, the
         * tess import and the CPU route all write them at HDR + cap_idx) */
        [enc setBuffer: gPrim offset: gPrimOff + (MGL_GEOM_HDR_WORDS + cap_idx) * sizeof(uint32_t)
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
        [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * sizeof(uint32_t) atIndex: MGL_GEOM_MSL_LSCR];

        if ([self geomBindStageBuffers: enc geomProgram: gp])
        {
            NSUInteger wg = 32;
            uint64_t groups = (threads + wg - 1) / wg;
            if (!groups) groups = 1;
            if (groups > 0xffffffffull) groups = 0xffffffffull;

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

            /* order the emitted strips: scan the per-thread list counts into bases, then
             * copy the per-thread scratch segments into the raster list in primitive
             * order (the GL-required emission order); finalize folds the total into the
             * indirect draw arguments */
            {
                uint32_t scan_in[2] = { (uint32_t)threads, (uint32_t)gp->max_vertices * 3u };
                [enc setComputePipelineState: gLscanPso];
                [enc setBytes: scan_in length: sizeof scan_in atIndex: 0];
                [enc setBuffer: gCtl offset: 0 atIndex: 1];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * 2 * sizeof(uint32_t) atIndex: 2];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(32, 1, 1)];
                [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];

                [enc setComputePipelineState: gLcopyPso];
                [enc setBytes: scan_in length: sizeof scan_in atIndex: 0];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * sizeof(uint32_t) atIndex: 1];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * 2 * sizeof(uint32_t) atIndex: 2];
                [enc setBuffer: gIdx offset: 0 atIndex: 3];
                NSUInteger lwg = 32;
                uint64_t cgroups = (threads + lwg - 1) / lwg;
                if (!cgroups) cgroups = 1;
                [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)cgroups, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(lwg, 1, 1)];
                [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];
            }

            /* fold the counters into the indirect draw arguments (also uses buffer(0),
             * so it runs after the GS dispatch, which is the last user of the stage
             * buffers — no slot is left shadowed for a later kernel) */
            [enc setComputePipelineState: gFinPso];
            [enc setBuffer: gCtl offset: 0 atIndex: 0];
            [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            /* fold this draw's per-stream counters into the persistent query words
             * (queries.c reads them through the ext_geom query hooks); the dispatch is
             * ordered after the geometry kernel within this encoder */
            if (!gQacc)
                gQacc = [_device newBufferWithLength: 8 * sizeof(uint32_t)
                                             options: MTLResourceStorageModeShared];
            if (gQacc && gQaccPso)
            {
                uint32_t mask = gQaccReset;
                if (getenv("MGL_GEOM_QDEBUG"))
                    fprintf(stderr, "geom: qacc dispatch mask=%x\n", mask);
                gQaccReset = 0;
                gQaccRan = true;
                gQaccCb = _currentCommandBuffer;
                [enc setComputePipelineState: gQaccPso];
                [enc setBuffer: gCtl offset: 0 atIndex: 0];
                [enc setBuffer: gQacc offset: 0 atIndex: 1];
                [enc setBytes: &mask length: sizeof mask atIndex: 2];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
            }
        }

        [enc popDebugGroup];
        if (mgl_xfb_capturing(ctx) && !mgl_geom_xfb_paused(ctx))
        {
            /* leave the kernel encoder OPEN for the pack that follows (see the transform
             * feedback block below): it joins this encoder so the pack's reads of the
             * kernel's writes are ordered */
            gPackJoinEnc = enc;
            gPackJoinCb = _currentCommandBuffer;
        }
        else
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
         * them.
         *
         * When the geometry kernel dispatched above, its encoder is left OPEN and handed
         * over (gPackJoinEnc): Metal orders dispatches — and their memory effects — only
         * WITHIN one compute encoder; separate encoders of the same command buffer may
         * overlap, and the pack's prefix/scatter read the kernel's per-thread emitted
         * counts and records.  CTS tcte.data_pass_through captured 110 of 160 records
         * deterministically through exactly that gap. */
        if (!mgl_geom_xfb_pack(ctx, gp, program, (uint32_t)threads,
                               gp->max_vertices ? gp->max_vertices : 1, gp->out_words,
                               (uint32_t)outv_cap))
        {
            /* the GPU pack never ran: close the kernel encoder it would have joined (it is
             * consumed only at its encoder-creation point) */
            if (gPackJoinEnc)
            {
                if (gPackJoinCb == _currentCommandBuffer) [gPackJoinEnc endEncoding];
                gPackJoinEnc = nil;
                gPackJoinCb = nil;
            }
            BUF()->flush(ctx);
            BUF()->wait_serial(ctx, BUF()->current_serial(ctx));
            mgl_geom_xfb_capture(ctx, gp, (const uint32_t *)gOutv.contents + gOutvOff / 4,
                                 (gOutv.length - gOutvOff) / sizeof(uint32_t), (const uint32_t *)gCtl.contents,
                                 (const uint32_t *)gCnt.contents + gCntOff / 4,
                                 (gCnt.length - gCntOff) / sizeof(uint32_t));
        }
        /* the pack either consumed the encoder (and closed it) or never took it: nothing
         * may hold a still-open encoder past this point */
        if (gPackJoinEnc)
        {
            if (gPackJoinCb == _currentCommandBuffer) [gPackJoinEnc endEncoding];
            gPackJoinEnc = nil;
            gPackJoinCb = nil;
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
        if (_currentCommandBuffer && _currentCommandBuffer.error)
            fprintf(stderr, "geom: trace cb error: %s\n", _currentCommandBuffer.error.localizedDescription.UTF8String);
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
        if (getenv("MGL_GEOM_DUMP_RUNS"))
        {
            /* tiled-chain state: T tiles, total restarts, the GS/emit grids, then the
             * first run records at sc[5T] (start, subcount, prim prefix, cap prefix) */
            uint32_t *pin = (uint32_t *)gPrep.contents + gPrepOff / 4;
            uint32_t T = pin[MGL_GEOM_PREP_OUT_OFF + 16];
            uint32_t tot = pin[MGL_GEOM_PREP_OUT_OFF + 19];
            uint32_t *sc = (uint32_t *)gPrepScr.contents;
            fprintf(stderr, "\ngeom: trace runs T=%u total=%u R=%u groups=%u emit=%u rec0..3:",
                    T, tot, tot + 1u, pin[MGL_GEOM_PREP_OUT_OFF + 4],
                    pin[MGL_GEOM_PREP_OUT_OFF + 24]);
            for (int r = 0; gPrepScr && T && r < 4 && r < (int)(tot + 1u); r++)
                fprintf(stderr, " [%u %u %u %u]", sc[5u * T + 4u * r + 0u], sc[5u * T + 4u * r + 1u],
                        sc[5u * T + 4u * r + 2u], sc[5u * T + 4u * r + 3u]);
        }
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
    /* skipped under RASTERIZER_DISCARD (GL 4.6 10.4.4) and when the active program has no
     * fragment stage: without an FS nothing observable rasterizes (no color / depth
     * outputs — Metal cannot even build the pipeline: rasterizationEnabled=NO demands a
     * void vertex function, which the pass-through vertex function is not). */
    if (ctx->state.caps.rasterizer_discard == GL_TRUE ||
        !program->shader_slots[_FRAGMENT_SHADER])
    {
        geomMarkPools(_currentCommandBuffer);   /* the pack above was this draw's last consumer */
        return true;
    }


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
    geomMarkPools(_currentCommandBuffer);   /* after begin_draw: covers any command buffer it made */

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

#pragma mark - SSBO runtime-array lengths (shared program-resource utility)

/* SPIRV-Cross implements .length() on an SSBO's runtime-sized array through a
 * `constant uint* spvBufferSizeConstants [[buffer(25)]]` argument: element [i] is the
 * byte size of the bound range of the buffer at Metal buffer index i. The list index IS
 * the Metal buffer index (extBufBindList binds list[i] at index i), so one fill covers
 * every stage resource; entries that are not SSBOs are simply never read.
 * GL 4.6 4.4.1.3: the element count is (bound range size - array offset) / stride, and
 * SPIRV-Cross bakes the offset/stride as constants, so the bound range size is exactly
 * what must be fed (KHR-GL46.shader_storage_buffer_object advanced-unsizedArrayLength,
 * incl. the bindRange{Offset,Size} variants). */
- (void) geomFillSSBOSizes: (uint32_t *) sizes list: (BufferMapList *) list count: (GLuint *) out_count
{
    GLuint n = 0;
    memset(sizes, 0, MAX_MAPPED_BUFFERS * sizeof(uint32_t));
    for (GLuint i = 0; i < list->count && i < MAX_MAPPED_BUFFERS; i++) {
        BufferMap *m = &list->buffers[i];
        if (!m->buf) continue;
        /* base binding records the whole buffer size, a range binding its size;
         * a zero (unset) record falls back to buffer end - offset */
        GLsizeiptr sz = m->size > 0 ? m->size : (m->buf->size - (m->offset > 0 ? m->offset : 0));
        sizes[i] = sz > 0 ? (uint32_t)sz : 0;
        n = i + 1;
    }
    *out_count = n;
}

/* render pass: bind the sizes for the vertex / fragment table. Skipped while the
 * geometry stage's pass-through vertex function owns the vertex table, whose slot
 * MGL_SSBO_SIZE_BUFFER_INDEX is MGL_GEOM_MSL_VSOUT — the pass-through never reads
 * sizes (the program's vertex stage runs as a capture kernel before the pass). */
/* an empty fragment function for vertex-only pipelines: Apple GPUs reject a render
 * pipeline with rasterizationEnabled=NO whose vertex function has a non-void return
 * (every GL vertex-only program still declares gl_Position), so the pipeline keeps
 * rasterization enabled with this no-output fragment function and no attachments —
 * nothing can be rasterized, and the vertex stage's side effects (SSBO / image /
 * atomic-counter writes, the whole point of a vertex-only draw) still execute.
 * Rasterizer discard collapses the vertices through the same fixup constant. */
- (id<MTLFunction>) geomEmptyFragmentFunction
{
    static id<MTLFunction> fn = nil;
    static bool tried = false;

    if (!fn && !tried)
    {
        tried = true;
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
                        @"fragment void mgl_empty_fragment() { }\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:src options:nil error:&err];
        if (lib)
            fn = [lib newFunctionWithName:@"mgl_empty_fragment"];
    }
    return fn;
}

- (void) geomBindSSBOSizes: (BufferMapList *) list stage: (int) stage vertex: (bool) vertex
{    Program *p = ctx->state.program;

    if (!p || stage < 0 || !p->spirv[stage].msl_str) return;
    if (!strstr(p->spirv[stage].msl_str, "spvBufferSizeConstants")) return;
    if (vertex && ctx->state.ext_geom && ((struct MGLExtState_geom *)ctx->state.ext_geom)->vertex_override)
        return;
    if (!_currentRenderEncoder) return;

    uint32_t sizes[MAX_MAPPED_BUFFERS];
    GLuint n = 0;
    [self geomFillSSBOSizes: sizes list: list count: &n];
    if (!n) return;
    if (vertex)
        [_currentRenderEncoder setVertexBytes: sizes length: (NSUInteger)n * sizeof(uint32_t)
                                     atIndex: MGL_SSBO_SIZE_BUFFER_INDEX];
    else
        [_currentRenderEncoder setFragmentBytes: sizes length: (NSUInteger)n * sizeof(uint32_t)
                                       atIndex: MGL_SSBO_SIZE_BUFFER_INDEX];
}

/* compute pass: glDispatchCompute's own encoder binding (not the geometry kernels) */
- (void) geomBindSSBOSizesCompute: (id <MTLComputeCommandEncoder>) enc
{
    Program *p = ctx->state.program;

    if (!enc || !p || !p->spirv[_COMPUTE_SHADER].msl_str) return;
    if (!strstr(p->spirv[_COMPUTE_SHADER].msl_str, "spvBufferSizeConstants")) return;

    uint32_t sizes[MAX_MAPPED_BUFFERS];
    GLuint n = 0;
    [self geomFillSSBOSizes: sizes list: &ctx->state.compute_buffer_map_list count: &n];
    if (getenv("MGL_SSBO_DEBUG"))
    {
        NSLog(@"MGL geom: ssbo sizes (compute): n=%u list->count=%u [0]=%u [1]=%u msl=%p",
              n, ctx->state.compute_buffer_map_list.count, sizes[0], sizes[1],
              p->spirv[_COMPUTE_SHADER].msl_str);
        for (GLuint i = 0; i < ctx->state.compute_buffer_map_list.count; i++)
        {
            BufferMap *m = &ctx->state.compute_buffer_map_list.buffers[i];
            NSLog(@"MGL geom:   cs slot[%u]: base=%u off=%ld size=%ld buf=%p",
                  i, m->buffer_base_index, (long)m->offset, (long)m->size, (void *)m->buf);
        }
    }
    if (!n) return;
    /* the compute stage is compiled with its buffer-size slot at 30 (out of the
     * program's argument range — see MGL_CS_SSBO_SIZE_BUFFER_INDEX), unlike the
     * render stages' shared slot 25 */
    [enc setBytes: sizes length: (NSUInteger)n * sizeof(uint32_t) atIndex: MGL_CS_SSBO_SIZE_BUFFER_INDEX];
}

#pragma mark - vertex-stage shadow-sampling emulation (mgl_fix_shadow_vs)

/* the compare ops the last specialization baked in, for the per-draw staleness
 * check (a GL_TEXTURE_COMPARE_FUNC change alone does not dirty the pipeline) */
static struct { void *program; bool valid; int ops[32]; } gShadowSpec;

- (void) geomShadowFCValues:(MTLFunctionConstantValues *)cv stage:(int)stage hash:(uint64_t *)hash
{
    Program *p = ctx->state.program;
    if (!p || stage != _VERTEX_SHADER) return;
    uint32_t mask = p->spirv[_VERTEX_SHADER].shadow_slots;
    if (!mask) return;

    SpirvResourceList *rl = &p->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    int ops[32];
    for (int i = 0; i < 32; i++) ops[i] = -1;

    for (GLuint i = 0; i < rl->count; i++)
    {
        GLuint slot = rl->list[i].msl_binding;
        if (slot >= 32 || !(mask & (1u << slot))) continue;
        GLuint unit = rl->list[i].binding;
        Texture *tex = mgl_texture_for_sampler(ctx, unit, rl->list[i].target);
        if (!tex) continue;
        Sampler *gs = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;
        TextureParameter *pp = gs ? &gs->params : &tex->params;
        int op = mgl_geom_shadow_op_of(pp->compare_func);
        ops[slot] = op;
        int32_t v = op;
        [cv setConstantValue:&v type:MTLDataTypeInt atIndex:MGL_GEOM_FC_SHADOW_OP_BASE + slot];
        if (hash) { *hash ^= (uint64_t)(op + 1) << (slot & 7); *hash *= 1099511628211ULL; }
    }

    gShadowSpec.program = (void *)p;
    gShadowSpec.valid = true;
    memcpy(gShadowSpec.ops, ops, sizeof ops);
}

/* called on the texture-binding path (before the PSO build of the same draw): if the
 * compare function of a shadow-sampled unit changed since the last specialization,
 * the vertex function must be re-specialized — dirty the program so processGLState
 * rebuilds the pipeline (the constants are baked into the Metal function) */
- (void) geomShadowCheckStale:(GLuint)unit texture:(Texture *)tex slot:(GLuint)slot
{
    if (!gShadowSpec.valid || gShadowSpec.program != (void *)ctx->state.program || slot >= 32) return;
    if (!(ctx->state.program->spirv[_VERTEX_SHADER].shadow_slots & (1u << slot))) return;
    Sampler *gs = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;
    TextureParameter *pp = gs ? &gs->params : &tex->params;
    if (mgl_geom_shadow_op_of(pp->compare_func) != gShadowSpec.ops[slot])
    {
        gShadowSpec.valid = false;   /* the rebuild that follows re-records it */
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }
}

@end
