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
#include "mgl_ext_texext.h"   /* MGLTexExtInfo: buffer-texture writeback (r63) */
#include "mgl_ext_buffers.h"
#include "mgl_ext_program.h"
#include "mgl_ext_introspect.h"   /* mgli_sampler_elem_unit: image-array element -> its uniform's unit */
#include "mgl_ext_tess.h"   /* MGLTessCapture debug dump in geom_xfb_dump */
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_queries.h"   /* discard-skip: active pre-raster query counters */
#include "mgl_ext_texdata.h"    /* mgl_texdata_level_geometry: image-unit level completeness */
#include "texdata_formats.h"   /* tdFormatInfo: image-unit format reinterpretation */
#include "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"   /* compat: bind shadow (perf session 3) */

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
static size_t gPoolBudget;          /* total bytes the doubling may grow the pools to
                                      * (MGL_GEOM_POOL_BUDGET; run-ahead runway) */
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
static id<MTLComputePipelineState> gPscanPso, gPbasePso, gLcopyPso, gLzeroPso;   /* raster list order passes */
/* tile-parallel prep (large draws): per-tile restart counts -> prefix sum -> per-tile run
 * records -> parallel emit; see the comment block in kGeomPrepMSL */
static id<MTLComputePipelineState> gPrepcPso, gTilesPso, gPrefixPso, gRunsPso, gReducePso, gEmitPso;
static id<MTLBuffer> gPrepScr;       /* its scratch: per-tile counts and per-run records */
static size_t gPrepScrWords;
/* below this index count the serial prep kernel (one dispatch) is cheaper than the chain;
 * overridable so the test suite can exercise the tile-parallel path on every draw */
static uint32_t gTiledMin = 4096u;/* the scratch is capped so absurdly large indirect draws fall back to the serial kernel
 * instead of asking for hundreds of megabytes */
#define MGL_GEOM_TILED_MAX_WORDS (64u * 1024u * 1024u / 4u)
/* the trace / behaviour switches are read once: this file sits on the draw path and
 * getenv() per draw (several times per draw) is not free */
static struct {
    int  read;
    bool draw, debug, xfb, worst_case, dump, serial_prep;
    /* stage-skip diagnostics (MGL_GEOM_SKIP_*): GPU-time attribution of the emulation's
     * stages — the skipped stage's consumers keep reading the same VOLUME of data (a
     * previous draw's records / counters), so the difference of two runs is the stage's
     * own cost.  Development diagnostics only; never set by the library itself. */
    bool skip_capture, skip_kernel, skip_order, skip_raster, gputime, skip_gs, no_barrier, skip_scan, skip_copy, no_implicit;
    int scan_reps;
} gEnv = { 0, false, false, false, false, false, false, false, false, false, false, false, false, false, false, 0 };

static void geomEnvRead(void)
{
    if (gEnv.read) return;    gEnv.read = 1;
    gEnv.draw = getenv("MGL_GEOM_TRACE_DRAW") != NULL;
    gEnv.debug = getenv("MGL_GEOM_DEBUG") != NULL;
    gEnv.xfb = getenv("MGL_GEOM_XFB_TRACE") != NULL;
    gEnv.worst_case = getenv("MGL_GEOM_WORST_CASE_DISPATCH") != NULL;
    gEnv.dump = getenv("MGL_GEOM_DUMP") != NULL;
    gEnv.serial_prep = getenv("MGL_GEOM_SERIAL_PREP") != NULL;   /* A/B the serial kernel */
    gEnv.skip_capture = getenv("MGL_GEOM_SKIP_CAPTURE") != NULL;
    gEnv.skip_kernel = getenv("MGL_GEOM_SKIP_KERNEL") != NULL;
    gEnv.skip_order = getenv("MGL_GEOM_SKIP_ORDER") != NULL;
    gEnv.skip_raster = getenv("MGL_GEOM_SKIP_RASTER") != NULL;
    gEnv.skip_gs = getenv("MGL_GEOM_SKIP_GS") != NULL;
    gEnv.no_barrier = getenv("MGL_GEOM_NO_BARRIER") != NULL;   /* PERF PROBE ONLY: skip the
                                   * order passes' memory barriers; results may be wrong */
    gEnv.skip_scan = getenv("MGL_GEOM_SKIP_SCAN") != NULL;
    gEnv.skip_copy = getenv("MGL_GEOM_SKIP_COPY") != NULL;
    gEnv.no_implicit = getenv("MGL_GEOM_NO_IMPLICIT") != NULL;   /* A/B: force assembly */
    {
        const char *r = getenv("MGL_GEOM_SCAN_REPS");
        if (r) { int v = atoi(r); if (v >= 1 && v <= 32) gEnv.scan_reps = v; }
    }
    /* MGL_GEOM_GPUTIME=1: every command buffer the geometry path encodes into gets a
     * completed handler that sums its GPU busy time (MTLCommandBuffer GPUStartTime/
     * GPUEndTime) and reports the average per command buffer — ground truth for where
     * a scene's frame time goes, independent of CPU starvation */
    gEnv.gputime = getenv("MGL_GEOM_GPUTIME") != NULL;
    /* the scratch pools' shape is read here once: the per-slot byte cap (how far a batch
     * of draws may run ahead of the GPU before a wait) and the rotation depth.  Test
     * hook: MGL_GEOM_POOL_CAP=<bytes> forces tiny pools so every draw wraps/reclaims. */
    gPoolCap = 64u << 20;
    gPoolBudget = 512u << 20;
    gPoolN = MGL_GEOM_POOL_N;
    {
        const char *c = getenv("MGL_GEOM_POOL_CAP");
        const char *n = getenv("MGL_GEOM_POOL_SLOTS");
        const char *b = getenv("MGL_GEOM_POOL_BUDGET");
        if (c) { unsigned long long v = strtoull(c, NULL, 0); if (v >= 1024) gPoolCap = (size_t)v; }
        if (n) { int v = atoi(n); if (v >= 2 && v <= (int)MGL_GEOM_POOL_N) gPoolN = (unsigned)v; }
        if (b) { unsigned long long v = strtoull(b, NULL, 0); if (v >= 4096) gPoolBudget = (size_t)v; }
    }
    const char *tmin = getenv("MGL_GEOM_TILED_MIN");             /* test hook: force the chain */
    if (tmin) { int v = atoi(tmin); gTiledMin = v > 0 ? (uint32_t)v : 0u; }
}

/* ---- opt-in GPU-time accounting (MGL_GEOM_GPUTIME=1) --------------------------
 * Every command buffer the geometry path encodes into gets, exactly once and before it
 * can be committed, a completed handler that sums its GPU busy time and counts it.  The
 * report line prints the running average — per command buffer, which in a scene with one
 * draw per frame is the frame's GPU cost.  Mirrors the tess domain's tDiag accounting. */
static id<MTLCommandBuffer> gTimedCb;
static _Atomic uint64_t gGpuTimeNs, gGpuTimeCbs, gGpuTimeDraws;

void mgl_geom_gpu_time_reset(void) { gGpuTimeNs = 0; gGpuTimeCbs = 0; gGpuTimeDraws = 0; }
unsigned long long mgl_geom_gpu_cbs(void) { return gGpuTimeCbs; }
double mgl_geom_gpu_time_us_per_draw(void)
{
    return gGpuTimeDraws ? (double)gGpuTimeNs / 1000.0 / (double)gGpuTimeDraws : 0.0;
}

static void geomTimeCommandBuffer(id<MTLCommandBuffer> cb)
{
    if (!cb || !gEnv.gputime || cb == gTimedCb) return;
    if (cb.status >= MTLCommandBufferStatusCommitted) return;   /* handler must precede commit */
    gTimedCb = cb;
    [cb addCompletedHandler: ^(id<MTLCommandBuffer> b) {
        if (b.GPUEndTime > b.GPUStartTime)
            gGpuTimeNs += (uint64_t)((b.GPUEndTime - b.GPUStartTime) * 1e9);
        uint64_t n = ++gGpuTimeCbs;
        if (n % 60 == 0)
            fprintf(stderr, "geom: GPU-TIME %.1f us/command buffer, %.1f us/draw (%llu cbs, %llu draws)\n",
                    (double)gGpuTimeNs / 1000.0 / (double)n,
                    gGpuTimeDraws ? (double)gGpuTimeNs / 1000.0 / (double)gGpuTimeDraws : 0.0,
                    (unsigned long long)n, (unsigned long long)gGpuTimeDraws);
    }];
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
    "    /* `first' is the firstIndex of an indexed INDIRECT command (the element buffer is\n"
    "     * bound at its base there — glDrawElementsIndirect's command carries the element\n"
    "     * offset), and the firstVertex of an arrays draw; direct indexed draws pass 0 and\n"
    "     * carry their element offset in the binding */\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[first + (_i)] : (idx_size == 2u ? (uint)((const device ushort*)indices)[first + (_i)] : ((const device uint*)indices)[first + (_i)])) : (uint)(first + (int)(_i)))\n"
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
    "    uint count = prep[12], first = prep[13];\n"
    "    uint indexed = c.v[3], restart = c.v[7], idx_size = c.v[10];\n"
    "    bool restart_en = c.v[14] != 0u;\n"
    "    uint i = t * WG + slid;\n"
    "    bool valid = i < count;\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[first + (_i)] : (idx_size == 2u ? (uint)((const device ushort*)indices)[first + (_i)] : ((const device uint*)indices)[first + (_i)])) : 0u)\n"
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
    "    uint count = prep[12], first = prep[13], total = prep[19];\n"
    "    uint indexed = c.v[3], restart = c.v[7], idx_size = c.v[10];\n"
    "    bool restart_en = c.v[14] != 0u;\n"
    "    uint i = t * WG + slid;\n"
    "    bool valid = i < count;\n"
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[first + (_i)] : (idx_size == 2u ? (uint)((const device ushort*)indices)[first + (_i)] : ((const device uint*)indices)[first + (_i)])) : 0u)\n"
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
    "#define MGL_IDX(_i) (indexed ? (idx_size == 1u ? (uint)indices[first + (_i)] : (idx_size == 2u ? (uint)((const device ushort*)indices)[first + (_i)] : ((const device uint*)indices)[first + (_i)])) : (uint)(first + (int)(_i)))\n"
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
 * member table, this program's record follows geom_layout), fills the primitive slots and
 * publishes the counts the geometry kernel needs (gPrim's header words and the prep block,
 * exactly what the prep kernel writes for a plain draw).  The primitive slots follow the
 * capture's record order: when the records are in emission order (cfg[8], the GPU path's
 * permute — one record per emitted vertex) slot t IS record t; when they are per unique
 * coordinate (CPU reference path) the tessellator's index list maps them. */
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
    "                                 constant uint* cfg [[buffer(9)]],          /* stride,k,npairs,reccap,idxcap,in_words,inst,invoc,emission */\n"
    "                                 uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    uint nrec = cnt[0];\n"
    "    uint nidx = cnt[1];\n"
    "    if (nrec > cfg[3]) nrec = cfg[3];                 /* the ring's record capacity */\n"
    "    if (nidx > cfg[4]) nidx = cfg[4];                 /* and its index capacity */\n"
    "    if (t < nrec)\n"
    "    {\n"
    "        /* the tess capture's members only cover what the TES writes: unwritten clip\n"
    "         * and cull words (and varyings the consumer does not read) must start from\n"
    "         * a defined value — zero is positive, so an undefined clip/cull plane can\n"
    "         neither clip nor cull (GL leaves those values undefined) */\n"
    "        device uint* z = recs + (size_t)t * cfg[5];\n"
    "        for (uint i = 0u; i < cfg[5]; i++) z[i] = 0u;\n"
    "    }\n"
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
    "    /* records in GENERATION order (the emulated path's permuted capture): the emitted\n"
    "     * sequence already IS the primitive order, so the slots are the identity; only\n"
    "     * coordinate-ordered records (CPU reference path, point mode) go through the\n"
    "     * tessellator's index list — applying it in both cases permutes twice and\n"
    "     * scrambles the geometry kernel's input (KHR-GL46.shader_ballot tess pipelines) */\n"
    "    if (t < nidx) pslot[t] = cfg[8] ? t : idx[t];\n"
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
    " * primitive as the shader closed it).  Word 8: GEOMETRY_SHADER_INVOCATIONS\n"
    " * (ctl[21], one bump per (input primitive, invocation) thread - the only exact\n"
    " * source for indirect draws whose vertex/instance counts are GPU-side only,\n"
    " * KHR-GL46.pipeline_statistics_query_tests_ARB.*geometry_shader_queries). */\n"
    "kernel void mgl_geom_qacc(device mgl_geom_ctl& c [[buffer(0)]],\n"
    "                          device uint* acc [[buffer(1)]],\n"
    "                          constant uint& reset_mask [[buffer(2)]])\n"
    "{\n"
    "    for (uint w = 0u; w < 16u; w++)\n"
    "    {\n"
    "        uint v = (w < 4u) ? c.mgl_ctl[17u + w]\n"
    "                : (w < 8u) ? c.mgl_ctl[9u + w]\n"
    "                : (w == 8u) ? c.mgl_ctl[21u] : 0u;   /* w8: GEOMETRY_SHADER_INVOCATIONS */\n"
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
    "/* order pass 1a: chunk-parallel exclusive prefix scan of the per-thread list counts\n"
    " * (lc[t]) into lc[T + t].  The old single-32-lane scan walked T/32 chunks in a serial\n"
    " * dependent chain (load -> scan -> store, ~T/32 round trips); this one scans CHUNK\n"
    " * entries per 1024-thread group - 32 simds of 32 lanes, each simd prefix-scanned, the\n"
    " * 32 simd totals scanned by one more simd pass - leaving the chunk-LOCAL exclusive\n"
    " * base in lc[T+t] and the group's total in sums[g].  Pass 1b then scans the (T/CHUNK)\n"
    " * group totals, and lcopy adds that chunk base, giving the same global exclusive\n"
    " * prefix as before at a fraction of the serial latency. */\n"
    "kernel void mgl_geom_pscan(constant uint* in [[buffer(0)]],\n"
    "                           device uint* lc   [[buffer(1)]],\n"
    "                           device uint* sums [[buffer(2)]],\n"
    "                           uint gid  [[thread_position_in_grid]],\n"
    "                           uint lid  [[thread_position_in_threadgroup]],\n"
    "                           uint sgid [[thread_index_in_simdgroup]])\n"
    "{\n"
    "    uint T = in[0];\n"
    "    uint t = gid;\n"
    "    uint simd = lid >> 5;                     /* simd lane group within the group */\n"
    "    threadgroup uint tg[32];\n"
    "    uint orig = (t < T) ? lc[t] : 0u;\n"
    "    uint v = orig;\n"
    "    for (uint o = 1u; o < 32u; o <<= 1)\n"
    "    {\n"
    "        uint u = simd_shuffle_up(v, o);\n"
    "        if (sgid >= o) v += u;\n"
    "    }\n"
    "    uint loc = v - orig;                      /* exclusive prefix inside the simd */\n"
    "    uint tot = simd_broadcast(v, 31u);        /* this simd's total */\n"
    "    if (sgid == 0u) tg[simd] = tot;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (lid < 32u)\n"
    "    {\n"
    "        uint o2 = tg[lid];\n"
    "        uint w = o2;\n"
    "        for (uint o = 1u; o < 32u; o <<= 1)\n"
    "        {\n"
    "            uint u = simd_shuffle_up(w, o);\n"
    "            if (lid >= o) w += u;\n"
    "        }\n"
    "        tg[lid] = w - o2;                     /* exclusive prefix over the simds */\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (t < T) lc[T + t] = loc + tg[simd];\n"
    "    if (lid == 1023u) sums[gid >> 10] = tg[31] + tot;   /* the group's (chunk's) total:\n"
    "       tg[31] is the EXCLUSIVE prefix over the simds (sum of simds 0..30), so it must\n"
    "       take this lane's own simd (31) total on top.  tg[31] alone dropped every\n"
    "       chunk's last 32 threads: the following chunk's list base came up short, its\n"
    "       segments overwrote earlier ones and ctl[1] undercounted - visible as the\n"
    "       glmark2 gl41-geometry streaks once a draw exceeds 992 geometry threads. */\n"
    "}\n"
    "\n"
    "/* order pass 1b: exclusive scan of the per-chunk totals (pass 1a's sums) into the\n"
    " * chunk bases, and the grand total into ctl[1] (the indirect draw's indexCount). */\n"
    "kernel void mgl_geom_pbase(constant uint* in [[buffer(0)]],\n"
    "                           device uint* c    [[buffer(1)]],\n"
    "                           device uint* sums [[buffer(2)]],\n"
    "                           uint sgid [[thread_index_in_simdgroup]])\n"
    "{\n"
    "    uint C = in[0], carry = 0u;\n"
    "    for (uint base = 0u; base < C; base += 32u)\n"
    "    {\n"
    "        uint g = base + sgid;\n"
    "        uint orig = (g < C) ? sums[g] : 0u;\n"
    "        uint v = orig;\n"
    "        for (uint o = 1u; o < 32u; o <<= 1)\n"
    "        {\n"
    "            uint u = simd_shuffle_up(v, o);\n"
    "            if (sgid >= o) v += u;\n"
    "        }\n"
    "        if (g < C) sums[g] = carry + v - orig;   /* exclusive chunk base */\n"
    "        carry += simd_broadcast(v, 31u);\n"
    "    }\n"
    "    if (sgid == 0u) c[1] = carry;               /* total list entries */\n"
    "}\n"
    "\n"
    "/* order pass 2: copy each thread's scratch segment into the raster list at its\n"
    " * scanned offset - the segments then sit in primitive order, which an atomicAdd\n"
    " * append could not guarantee.  The destination is the chunk-local base from pass 1a\n"
    " * plus the chunk base from pass 1b. */\n"
    "kernel void mgl_geom_lcopy(constant uint* in [[buffer(0)]],\n"
    "                           device uint* lscr [[buffer(1)]],\n"
    "                           device uint* lc   [[buffer(2)]],\n"
    "                           device uint* idx  [[buffer(3)]],\n"
    "                           device uint* cb   [[buffer(4)]],\n"
    "                           uint t [[thread_position_in_grid]])\n"
    "{\n"
    "    if (t >= in[0]) return;\n"
    "    uint n = lc[t];\n"
    "    if (n == 0u) return;\n"
    "    uint dst = lc[in[0] + t] + cb[t >> 10];\n"
    "    uint src = t * in[1];\n"
    "    for (uint i = 0u; i < n; i++) idx[dst + i] = lscr[src + i];\n"
    "}\n";

static bool geomCtlPipelines(id<MTLDevice> device)
{
    if (gInitPso && gFinPso && gPrepPso && gPscanPso && gPbasePso && gLcopyPso && gLzeroPso && gQaccPso) return true;

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
    id<MTLFunction> pscanfn = [lib newFunctionWithName: @"mgl_geom_pscan"];
    id<MTLFunction> pbasefn = [lib newFunctionWithName: @"mgl_geom_pbase"];
    id<MTLFunction> lcopyfn = [lib newFunctionWithName: @"mgl_geom_lcopy"];
    id<MTLFunction> lzerofn = [lib newFunctionWithName: @"mgl_geom_lzero"];
    id<MTLFunction> qaccfn = [lib newFunctionWithName: @"mgl_geom_qacc"];
    if (!initfn || !finfn || !pscanfn || !pbasefn || !lcopyfn || !lzerofn || !qaccfn) return false;

    gInitPso = [device newComputePipelineStateWithFunction: initfn error: &error];
    gFinPso = [device newComputePipelineStateWithFunction: finfn error: &error];
    gPscanPso = [device newComputePipelineStateWithFunction: pscanfn error: &error];
    gPbasePso = [device newComputePipelineStateWithFunction: pbasefn error: &error];
    gLcopyPso = [device newComputePipelineStateWithFunction: lcopyfn error: &error];
    gLzeroPso = [device newComputePipelineStateWithFunction: lzerofn error: &error];
    gQaccPso = [device newComputePipelineStateWithFunction: qaccfn error: &error];
    if (!gInitPso || !gFinPso || !gPscanPso || !gPbasePso || !gLcopyPso || !gLzeroPso || !gQaccPso) return false;

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
    gGsPso = nil; gInitPso = nil; gFinPso = nil; gPrepPso = nil; gPscanPso = gPbasePso = gLcopyPso = gLzeroPso = nil; gTessImportPso = nil;
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

/* (defined in MGLRenderer+Draw.m; declared here so this category can create a
 * command buffer for shadow copies at readback time) */
@interface MGLRenderer (Geom3DShadowPriv)
- (void) extBufEnsureCommandBuffer;
@end

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
    if (!gQacc || !gQaccRan || word >= 16) { *out = 0; return; }

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
    if (getenv("MGL_GEOM_DUMP_CTL") && gCtl && gPrep)
    {
        const volatile uint32_t *c = (volatile uint32_t *)gCtl.contents;
        const volatile uint32_t *pr = (volatile uint32_t *)((char *)gPrep.contents + gPrepOff);
        const volatile uint32_t *h = (volatile uint32_t *)((char *)gPrim.contents + gPrimOff);
        fprintf(stderr, "ctl:");
        for (int i = 0; i < 26; i++) fprintf(stderr, " %u", c[i]);
        fprintf(stderr, "\nprep:");
        for (int i = 0; i < 12; i++) fprintf(stderr, " %u", pr[MGL_GEOM_PREP_OUT_OFF + i]);
        fprintf(stderr, "\nhdr:");
        for (int i = 0; i < 13; i++) fprintf(stderr, " %u", h[i]);
        fprintf(stderr, "\ncap:");
        for (int i = 0; i < 6; i++) fprintf(stderr, " %u", h[MGL_GEOM_HDR_WORDS + i]);
        fprintf(stderr, "\nprim:");
        for (int i = 0; i < 6; i++) fprintf(stderr, " %u", h[MGL_GEOM_HDR_WORDS + 8 + i]);
        fprintf(stderr, "\n");
    }
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
    /* the cull-distance pass-through (tess domain) has no geometry Shader object to keep
     * the kernel in: the MGLGeomProgram owns it instead */
    void **fn_slot = &gp->kernel_function, **lib_slot = &gp->kernel_library;
    if (gs) { fn_slot = &gs->mtl_data.function; lib_slot = &gs->mtl_data.library; }

    if ((!gs && !gp->synthetic) || !kernel_msl)
    {
        NSLog(@"MGL geom: the geometry kernel was not generated (program %u)", program->name);
        return false;
    }

    /* the program may not have been through bindMTLProgram yet: compile the kernel here */
    if (!*fn_slot || !*lib_slot)
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
        if (*fn_slot) ctx->mtl_funcs.ext_program->release_obj(ctx, *fn_slot);
        if (*lib_slot) ctx->mtl_funcs.ext_program->release_obj(ctx, *lib_slot);
        *fn_slot = fn;
        *lib_slot = lib;
    }

    if (gGsPso && gPsoKey == *fn_slot) return geomCtlPipelines(_device);

    id<MTLFunction> gsfn = (__bridge id<MTLFunction>)*fn_slot;

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
#pragma mark - 3D image units: 2D shadow planes

/* A non-layered image2D binding of a GL_TEXTURE_3D must alias one z-slice as a
 * 2D image.  Metal offers no path for that (probes in zz_*): no 3D->2D texture
 * view (validation aborts), no 3D<->2D blit (silently no-ops), and a 3D texture
 * bound into a texture2d argument silently kills the whole dispatch.  The slice
 * therefore runs through a 2D SHADOW texture: geom3DImageShadowSync (start of
 * every draw) copies the bound slice 3D->2D with a typed compute kernel, the
 * image unit binds the shadow, and pending stores are written 2D->3D when the
 * unit rebinds elsewhere, at buffer commit, and before any read of the 3D
 * texture (downloads).  Copy kernels are generated per MSL texel type from the
 * texture's GL format; unorm/snorm/float/integer formats round-trip exactly,
 * sRGB storage is copied through its linear variant (GL image access bypasses
 * sRGB decode).  Limitation: sampling the 3D texture in the SAME draw that
 * writes its shadowed image unit cannot see the pending stores (GL itself
 * leaves that race undefined). */

static NSMutableDictionary *gPlanePsoCache = nil;
static id<MTLTexture> g3DShadowTex[TEXTURE_UNITS];
static id<MTLTexture> g3DShadowBase[TEXTURE_UNITS];   /* retained 3D base of the shadow */
static void *g3DShadowOwner[TEXTURE_UNITS];
static GLuint g3DShadowLevel[TEXTURE_UNITS], g3DShadowLayer[TEXTURE_UNITS];
static BOOL g3DShadowValid[TEXTURE_UNITS], g3DShadowPending[TEXTURE_UNITS];
static BOOL g3DShadowLive = NO;  /* any shadow exists (shadows are never freed before teardown) */

/* per-draw gating (glmark2 ideas/jellyfish): the 3D-shadow sync scan runs only
 * for programs that actually have storage-image resources, or while any shadow
 * is alive (write-back duty).  Cached per program pointer + MSL pointer (a
 * program's address is recycled constantly by CTS-style apps). */
- (bool) geomProgramUsesImages: (Program *)p
{
    static struct { const void *prog, *msl; bool has; } cache[4];
    if (!p)
        return false;
    const void *msl = (const void *)p->spirv[_FRAGMENT_SHADER].msl_str;
    if (!msl)
        msl = (const void *)p->spirv[_VERTEX_SHADER].msl_str;

    for (int i = 0; i < 4; i++)
        if (cache[i].prog == (const void *)p && cache[i].msl == msl)
            return cache[i].has;

    bool has = false;
    for (int s = 0; s < _MAX_SHADER_TYPES && !has; s++)
    {
        if (!p->shader_slots[s])
            continue;
        const SpirvResourceList *l = &p->spirv_resources_list[s][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
        if (l && l->count)
            has = true;
    }

    int ci = 0;
    for (int i = 1; i < 4; i++) if (!cache[i].prog) { ci = i; break; }
    cache[ci].prog = (const void *)p;
    cache[ci].msl = msl;
    cache[ci].has = has;
    return has;
}

- (const char *) geom3DPlaneMslType: (GLenum) internalformat
{
    const TDFormat *f = tdFormatInfo(internalformat);

    if (!f || (f->flags & TDF_COMPRESSED))
        return NULL;
    if (f->base != GL_RED && f->base != GL_RG && f->base != GL_RGB && f->base != GL_RGBA)
        return NULL;
    /* Metal texture templates take the SCALAR channel type; the vector width
     * follows the pixel format automatically (read/write use vec<T,4>) */
    if (f->comp_type == GL_INT)
        return "int";
    if (f->comp_type == GL_UNSIGNED_INT)
        return "uint";
    return "float";
}

- (id<MTLComputePipelineState>) geom3DPlanePso: (int) dir mslType: (const char *) T
{
    NSString *key = [NSString stringWithFormat:@"p%d_%s", dir, T];
    id<MTLComputePipelineState> p = gPlanePsoCache[key];

    if (p)
        return p;

    NSString *src = [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
        @"using namespace metal;\n"
        @"kernel void mgl_geom_p3to2(texture3d<%s> s [[texture(0)]],\n"
        @"                            texture2d<%s, access::write> d [[texture(1)]],\n"
        @"                            constant uint2 &zl [[buffer(0)]],\n"
        @"                            ushort2 gid [[thread_position_in_grid]])\n"
        @"{\n"
        @"    if (gid.x >= d.get_width() || gid.y >= d.get_height()) return;\n"
        @"    d.write(s.read(uint3(uint2(gid), zl.x), zl.y), uint2(gid));\n"
        @"}\n"
        @"kernel void mgl_geom_p2to3(texture2d<%s> s [[texture(0)]],\n"
        @"                            texture3d<%s, access::write> d [[texture(1)]],\n"
        @"                            constant uint2 &zl [[buffer(0)]],\n"
        @"                            ushort2 gid [[thread_position_in_grid]])\n"
        @"{\n"
        @"    if (gid.x >= s.get_width() || gid.y >= s.get_height()) return;\n"
        @"    d.write(s.read(uint2(gid)), uint3(uint2(gid), zl.x), zl.y);\n"
        @"}\n", T, T, T, T];
    NSError *err = nil;
    id<MTLLibrary> lib = [_device newLibraryWithSource:src options:nil error:&err];

    if (!lib)
    {
        NSLog(@"MGL WARNING: 3D image-plane library (%@): %@", key, err);
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:dir ? @"mgl_geom_p2to3" : @"mgl_geom_p3to2"];
    p = [_device newComputePipelineStateWithFunction:fn error:&err];
    if (!p)
    {
        NSLog(@"MGL WARNING: 3D image-plane pso (%@): %@", key, err);
        return nil;
    }
    static dispatch_once_t once;
    dispatch_once(&once, ^{ gPlanePsoCache = [NSMutableDictionary new]; });
    gPlanePsoCache[key] = p;
    return p;
}

/* encode one plane copy (dir 0: 3D->2D, 1: 2D->3D) */
- (BOOL) geom3DPlaneCopy: (int) dir enc: (id<MTLComputeCommandEncoder>) enc
                     pso: (id<MTLComputePipelineState>) pso
                   tex3d: (id<MTLTexture>) t3 tex2d: (id<MTLTexture>) t2
                  layer: (NSUInteger) z level: (NSUInteger) level
{
    uint32_t zl[2] = {(uint32_t)z, (uint32_t)level};
    NSUInteger w = dir ? [t2 width] : [t3 width] >> level, h = dir ? [t2 height] : [t3 height] >> level;

    if (!dir)
    {
        w = [t2 width];
        h = [t2 height];
    }
    w = MAX(w, 1);
    h = MAX(h, 1);

    [enc setComputePipelineState:pso];
    [enc setTexture:dir ? t2 : t3 atIndex:0];
    [enc setTexture:dir ? t3 : t2 atIndex:1];
    [enc setBytes:zl length:8 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((w + 15) / 16, (h + 15) / 16, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    return YES;
}

/* refresh shadows before a draw; write back stale ones.  Must run before the
 * draw's render encoder is created (it ends the current one itself). */
- (void) geom3DImageShadowSync
{
    GLMContext glm = ctx;
    struct
    {
        int dir, unit;
        void *tex;
        NSUInteger layer, level;
    } todo[TEXTURE_UNITS];
    int ncopy = 0;

    if (!ctx || !ctx->mtl_funcs.ext_texdata)
        return;

    /* glmark2 ideas/jellyfish gating: nothing to write back and no stage can
     * consume an image unit — the scan is pure overhead.  With a bound program
     * PIPELINE and no current program, processGLState substitutes the composite
     * only later (MGL_PIPELINE_PROGRAM_SCOPE inside it), so consult the
     * pipeline's stage programs directly here. */
    bool uses_images = [self geomProgramUsesImages: ctx->state.program];
    if (!uses_images && !ctx->state.program && STATE(program_pipeline))
    {
        MGLPipelineObj *po = MGL_PIPELINE_OBJ(STATE(program_pipeline));
        for (int s = 0; s < _MAX_SHADER_TYPES && !uses_images; s++)
            uses_images = [self geomProgramUsesImages: po->base.stage_programs[s]];
    }
    if (!g3DShadowLive && !uses_images)
        return;

    /* write back shadows that stopped matching their unit's binding (or whose
     * unit no longer wants one) */
    for (GLuint u = 0; u < TEXTURE_UNITS; u++)
    {
        ImageUnit *iu = &STATE(image_units[u]);
        BOOL wants = iu->tex && !iu->layered && iu->tex->target == GL_TEXTURE_3D;

        if (g3DShadowTex[u] && g3DShadowPending[u] &&
            (!wants || g3DShadowOwner[u] != iu->tex || g3DShadowLevel[u] != iu->level ||
             g3DShadowLayer[u] != (GLuint)MAX(iu->layer, 0)))
        {
            /* capture the OLD slice here: the refresh loop below updates the
             * per-unit state before these copies are encoded */
            todo[ncopy].dir = 1;
            todo[ncopy].unit = u;
            todo[ncopy].tex = g3DShadowOwner[u];
            todo[ncopy].layer = g3DShadowLayer[u];
            todo[ncopy++].level = g3DShadowLevel[u];
        }
        else if (g3DShadowTex[u] && !wants)
            g3DShadowValid[u] = NO;   /* parked */
    }

    /* copy in slices whose shadow is new or was invalidated */
    for (GLuint u = 0; u < TEXTURE_UNITS; u++)
    {
        ImageUnit *iu = &STATE(image_units[u]);

        if (!(iu->tex && !iu->layered && iu->tex->target == GL_TEXTURE_3D))
            continue;
        if (!ctx->mtl_funcs.ext_texdata->realize || !ctx->mtl_funcs.ext_texdata->realize(ctx, iu->tex) || !iu->tex->mtl_data)
            continue;

        id<MTLTexture> base = (__bridge id<MTLTexture>)iu->tex->mtl_data;
        NSUInteger level = iu->level < [base mipmapLevelCount] ? iu->level : 0;
        NSUInteger layer = (NSUInteger)MAX(iu->layer, 0);
        NSUInteger w = MAX([base width] >> level, 1), h = MAX([base height] >> level, 1);

        if (layer >= [base depth])
            continue;

        if (!g3DShadowTex[u] || [g3DShadowTex[u] pixelFormat] != [base pixelFormat] ||
            [g3DShadowTex[u] width] != w || [g3DShadowTex[u] height] != h)
        {
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[base pixelFormat]
                                                                                          width:w height:h mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            d.storageMode = MTLStorageModePrivate;
            g3DShadowTex[u] = [_device newTextureWithDescriptor:d];
            g3DShadowLive = YES;
            g3DShadowValid[u] = NO;
            g3DShadowPending[u] = NO;
        }
        if (!g3DShadowValid[u] || g3DShadowOwner[u] != iu->tex ||
            g3DShadowLevel[u] != level || g3DShadowLayer[u] != layer)
        {
            todo[ncopy].dir = 0;
            todo[ncopy].unit = u;
            todo[ncopy].tex = iu->tex;
            todo[ncopy].layer = layer;
            todo[ncopy++].level = level;
        }
        g3DShadowOwner[u] = iu->tex;
        g3DShadowBase[u] = base;
        g3DShadowLevel[u] = level;
        g3DShadowLayer[u] = layer;
    }

    if (!ncopy)
        return;

    /* encode the copies (write-backs first, so a slice that moved between two
     * units of the same texture lands in order) */
    [self endRenderEncoding];
    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];

    if (!enc)
        return;
    for (int i = 0; i < ncopy; i++)
    {
        int u = todo[i].unit;
        const char *T = todo[i].tex ? [self geom3DPlaneMslType: ((Texture *)todo[i].tex)->internalformat] : NULL;

        if (!T || !g3DShadowTex[u])
            continue;
        id<MTLComputePipelineState> pso = [self geom3DPlanePso:todo[i].dir mslType:T];

        if (!pso)
            continue;
        [self geom3DPlaneCopy:todo[i].dir enc:enc pso:pso
                       tex3d:g3DShadowBase[u] tex2d:g3DShadowTex[u]
                      layer:todo[i].layer level:todo[i].level];
        if (todo[i].dir)
            g3DShadowPending[u] = NO;
        else
            g3DShadowValid[u] = YES;
    }
    [enc endEncoding];
    STATE(dirty_bits) = DIRTY_ALL;
}

/* write back every pending shadow (buffer commit) */
- (void) geom3DImageShadowFlush
{
    GLMContext glm = ctx;
    int ncopy = 0;
    int units[TEXTURE_UNITS];

    if (!ctx)
        return;

    /* no shadow ever created (g3DShadowLive, see geom3DImageShadowSync): the
     * per-commit unit scan below is pure overhead — glmark2 pays it per frame */
    if (!g3DShadowLive)
        return;

    for (GLuint u = 0; u < TEXTURE_UNITS; u++)
        if (g3DShadowTex[u] && g3DShadowPending[u] && g3DShadowBase[u])
            units[ncopy++] = u;
    if (!ncopy)
        return;

    [self extBufEnsureCommandBuffer];
    [self endRenderEncoding];
    id<MTLComputeCommandEncoder> enc = _currentCommandBuffer ? [_currentCommandBuffer computeCommandEncoder] : nil;

    if (!enc)
        return;
    for (int i = 0; i < ncopy; i++)
    {
        int u = units[i];
        const char *T = [self geom3DPlaneMslType: ((Texture *)g3DShadowOwner[u])->internalformat];

        if (!T)
            continue;
        id<MTLComputePipelineState> pso = [self geom3DPlanePso:1 mslType:T];

        if (!pso)
            continue;
        [self geom3DPlaneCopy:1 enc:enc pso:pso
                       tex3d:g3DShadowBase[u] tex2d:g3DShadowTex[u]
                      layer:g3DShadowLayer[u] level:g3DShadowLevel[u]];
        g3DShadowPending[u] = NO;
    }
    [enc endEncoding];
}

/* read of a 3D texture outside the image path: flush its shadow first */
- (void) geom3DImageShadowFlushTexture: (Texture *) tex
{
    /* no shadow ever created (see geom3DImageShadowSync): the unit scan is pure
     * overhead on every texture download (glmark2 pays it per frame readback) */
    if (!g3DShadowLive)
        return;

    for (GLuint u = 0; u < TEXTURE_UNITS; u++)
        if (g3DShadowPending[u] && g3DShadowOwner[u] == (void *)tex)
        {
            [self geom3DImageShadowFlush];
            return;
        }
}

- (id<MTLTexture>) geomImageUnitView: (const ImageUnit *) iu base: (id<MTLTexture>) base
{
    MTLTextureType type, view_type;
    NSUInteger level, layer, levels, array_len;
    NSRange level_range, slice_range;

    if (!base)
        return nil;

    /* incomplete: the unit's bound level has no image.  Reads must be zero and stores
     * must drop (GL 4.6 §8.26) — hand the shared 1x1 zero texture in the unit's format
     * instead of the texture's (realized default) storage, whose contents the store
     * would change (KHR-GL46.shader_image_load_store incomplete_textures: a 64x64
     * texture bound at a level with no image).  Every image binding path funnels through
     * here (render, tessellation kernels, geometry kernel, compute dispatch, the blit
     * compute path). */
    if (iu && iu->tex)
    {
        GLuint lw = 0, lh = 0, ld = 0;
        bool no_image = !iu->tex->width;

        if (!no_image)
            mgl_texdata_level_geometry((Texture *)iu->tex, 0, iu->level, &lw, &lh, &ld);
        if (getenv("MGL_GEOM_DEBUG") && (iu->tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                                         iu->tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY))
            fprintf(stderr, "MGL geom: iuv MS unit %d texw=%u lvl=%u lw=%u lh=%u base=%p %ux%lu\n",
                    iu->unit, iu->tex->width, iu->level, lw, lh, (__bridge void *)base,
                    (unsigned)base.width, (unsigned long)base.sampleCount);
        if (no_image || !lw)
        {
            id<MTLTexture> empty = [self geomImageEmptyTexture: iu->internalformat];

            if (empty)
                return empty;
        }
    }

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
                /* GL wants an image2D view of one 3D slice (see the shadow-plane
                 * block above geomImageUnitView).  The shadow was refreshed by
                 * geom3DImageShadowSync at the start of this draw. */
                id<MTLTexture> shadow = g3DShadowTex[iu->unit % TEXTURE_UNITS];

                if (shadow && iu->access != GL_READ_ONLY)
                    g3DShadowPending[iu->unit % TEXTURE_UNITS] = YES;
                return shadow ? shadow : base;
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

/* An image unit's `format` argument (glBindImageTexture) is how the shader sees the
 * texels — it does not have to be the texture's internal format, only to occupy the
 * same number of bytes (GL 4.6 §8.26: "the data in the texture is interpreted as if it
 * were the format named by <format>").  CTS multiple-uniforms binds ONE R32I texture to
 * eight units with eight different formats, whose MSL declares texture2d<float>/int/…
 * arguments — binding the R32Sint texture there is a Metal pixel-format mismatch.
 * A newTextureViewWithPixelFormat: view performs the reinterpretation (allowed between
 * formats of one texel size); elem_class is the SPIR-V image's sampled-type kind
 * (0 float, 1 int, 2 uint — SpirvResource.elem_class).  When the unit format's kind
 * DISAGREES with the shader's image type (a shader format-layout mismatch — undefined
 * per GL, not an error) the SHADER's interpretation wins: a same-texel-size format of
 * the shader's element kind is cast instead (mglImageCastFormat), because binding the
 * plain view hands Apple GPUs a texture2d<T> write against a pixel format of the wrong
 * element kind, which hangs all GPU work (KHR-GL46.shader_ballot_tests: rgba32f image
 * over an RGBA32UI texture). */
static GLenum mglImageCastFormat(uint8_t bpp, GLuint kind)
{
    switch (bpp)
    {
        case 16: return kind == 1u ? GL_RGBA32I  : kind == 2u ? GL_RGBA32UI  : GL_RGBA32F;
        case  8: return kind == 1u ? GL_RG32I    : kind == 2u ? GL_RG32UI    : GL_RG32F;
        case  4: return kind == 1u ? GL_R32I     : kind == 2u ? GL_R32UI     : GL_R32F;
        case  2: return kind == 1u ? GL_R16I     : kind == 2u ? GL_R16UI     : GL_R16F;
        case  1: return kind == 1u ? GL_R8I      : kind == 2u ? GL_R8UI      : GL_R8;
        default: return 0;                 /* no plain format of that texel size */
    }
}
- (id<MTLTexture>) geomImageUnitViewRes: (const ImageUnit *) iu base: (id<MTLTexture>) base
                              elemClass: (GLuint) elem_class
{
    id<MTLTexture> view = [self geomImageUnitView: iu base: base];
    const TDFormat *fi;
    MTLPixelFormat want;

    geomEnvRead();                             /* MGL_GEOM_DEBUG gates the trace below */
    if (!view || !iu || iu->internalformat == 0 || view.sampleCount > 1)
        return view;
    fi = tdFormatInfo(iu->internalformat);
    if (!fi || fi->mtl == 0)
        return view;
    want = (MTLPixelFormat)fi->mtl;         /* the unit <format>'s pixel format */
    {
        GLuint kind = (fi->comp_type == GL_INT) ? 1u :
                      (fi->comp_type == GL_UNSIGNED_INT) ? 2u : 0u;
        if (elem_class != 0xffffffffu && kind != elem_class)
        {
            /* shader format-layout mismatch: reinterpret in the SHADER's element kind
             * (see the comment above the helper) */
            GLenum cf = mglImageCastFormat(fi->bpp, elem_class);
            const TDFormat *cfi = cf ? tdFormatInfo(cf) : NULL;

            if (!cfi || !cfi->mtl)
                return view;             /* no same-texel-size format of the shader's kind */
            want = (MTLPixelFormat)cfi->mtl;
        }
    }
    if (want == view.pixelFormat)
        return view;
    /* the SAME cast view object must come back every draw: the image-atomic sidecar
     * keys its "binding unchanged, contents persist" check on the view's address
     * (geomImageAtomicBuffer) and the array layout signs it into its signature
     * (geomImageAtomicArrayLayout), so a fresh view per call would discard the
     * sidecar (and its accumulated atomics) on every draw.  A view keeps its parent
     * texture alive, so keying on the parent's address cannot alias a new texture. */
    {
        static NSMutableDictionary<NSString *, id<MTLTexture>> *castCache = nil;
        static dispatch_once_t once;
        NSString *key;
        id<MTLTexture> cached;

        dispatch_once(&once, ^{ castCache = [NSMutableDictionary dictionary]; });
        key = [NSString stringWithFormat: @"%p:%u", (void *)view, (unsigned)want];
        @synchronized (castCache) { cached = castCache[key]; }
        if (cached && cached.pixelFormat == want)
            return cached;
        @try {
            id<MTLTexture> rv = [view newTextureViewWithPixelFormat: want];
            if (gEnv.debug)
                NSLog(@"MGL geom: image-unit %u format view %d -> %d (elem_class %u, kind %u%s)",
                      iu->unit, (int)view.pixelFormat, (int)want, elem_class,
                      (fi->comp_type == GL_INT) ? 1u : (fi->comp_type == GL_UNSIGNED_INT) ? 2u : 0u,
                      rv ? "" : " FAILED");
            if (rv) {
                @synchronized (castCache) { castCache[key] = rv; }
                return rv;
            }
        } @catch (NSException *e) {
            NSLog(@"MGL geom: image-unit format view (%d -> %d) failed: %@",
                  (int)view.pixelFormat, (int)want, e);
        }
    }
    return view;
}

/* ------------------------------------------------------------------ void vertex stages
 * A rasterizing vertex stage that writes NO outputs (GLSL whose main() only stores to
 * images/buffers — KHR-GL46.shader_image_load_store multiple-uniforms' VS and TES
 * variants never assign gl_Position) compiles to `vertex void ...`, and Apple GPUs
 * reject the pipeline outright: "RasterizationEnabled is true but the vertex shader's
 * return type is void".  GL leaves rasterizing such a program undefined, so the draw
 * stays legal and the stage's side effects (the image writes the test measures) still
 * run when rasterization is disabled for the pipeline — exactly what the vertex-only
 * path already does (see generatePipelineDescriptor).  Call AFTER
 * rstateConfigurePipelineDescriptor, which derives rasterizationEnabled from the
 * fragment function.  The geometry pass-through override always returns a position
 * struct, so it is never affected. */
- (void) geomRasterizationForVertexStage: (MTLRenderPipelineDescriptor *) desc
{
    Program *program = ctx->state.program;
    int vstage = [self extTessActive] ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    if (!desc || !desc.rasterizationEnabled || !program)
        return;
    if (ctx->state.ext_geom &&
        ((struct MGLExtState_geom *)ctx->state.ext_geom)->vertex_override)
        return;                      /* the pass-through vertex function returns a struct */
    if (mgl_msl_vertex_returns_void(program->spirv[vstage].msl_str,
                                    program->spirv[vstage].entry_point))
    {
        if (gEnv.debug)
            NSLog(@"MGL geom: stage %d has no rasterization outputs: disabling "
                  @"rasterization (void vertex function)", vstage);
        desc.rasterizationEnabled = NO;
    }
}

/* ------------------------------------------------------------------ incomplete images
 * GL 4.6 §8.26: an image unit bound to a texture with no image (a texture that never
 * got a glTexImage / glTexStorage level — "incomplete") does not fail the draw:
 * imageLoad reads zeros and imageStore is dropped.  Realizing the texture (its default
 * storage) made the store land and the readback change —
 * KHR-GL46.shader_image_load_store incomplete_textures.  geomImageEmptyTexture hands
 * out a shared 1x1 zero texture in the unit's bound format, so both the load and the
 * store behave and the texture's own contents stay untouched.  The 8-entry cache covers
 * the formats a program actually reinterprets one texture as. */
- (id<MTLTexture>) geomImageEmptyTexture: (GLenum) internalformat
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    const TDFormat *fi = tdFormatInfo(internalformat);

    if (!g || !fi || !fi->mtl)
        return nil;

    for (int i = 0; i < 8; i++)
        if (g->img_empty[i].fmt == (GLuint)internalformat && g->img_empty[i].tex)
            return (__bridge id<MTLTexture>)g->img_empty[i].tex;

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                  (MTLPixelFormat)fi->mtl width:1 height:1 mipmapped:NO];
    if (!d)
        return nil;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;

    id<MTLTexture> t = [_device newTextureWithDescriptor:d];
    if (!t)
        return nil;

    int slot = -1;
    for (int i = 0; i < 8; i++)
        if (!g->img_empty[i].tex) { slot = i; break; }
    if (slot < 0) slot = 0;                   /* overwrite: formats are few */
    g->img_empty[slot].tex = (void *)CFBridgingRetain(t);
    g->img_empty[slot].fmt = (GLuint)internalformat;
    return t;
}

/* ------------------------------------------------------------------ deferred-texture realize
 * bindMTLTexture may run a deferred CPU-side upload (texdataEncodeUpload), which opens a
 * BLIT encoder — Metal forbids a second encoder on a command buffer while one is still
 * encoding.  The plain render path binds its stage textures before the render encoder
 * exists, but a tessellated draw opens the compute encoders of its kernels FIRST and
 * binds stage textures inside (tessBindStageTextures), so the first deferred upload of
 * the draw aborted with "A command encoder is already encoding to this command buffer"
 * (KHR-GL46.shader_image_load_store multiple-uniforms: eight image units over one R32I
 * texture, first bound for the tess VS kernel).
 * geomRealizeStageTextures walks the same units tessBindStageTextures /
 * bindTexturesToComputeEncoder bind and realizes them while no encoder is open — call it
 * before the first encoder of the draw.  Mirrors the pre-passes in extBufEncodeDispatch
 * (MGLRenderer+Draw.m) and the geometry-kernel section of this file. */
- (void) geomRealizeStageTextures: (Program *) program stage: (int) stage
{
    struct { int spvc_type; int image; } types[] = {
        { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, 0 },
        { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, 1 },
        { 0, 0 }
    };

    if (!program || !ctx) return;
    bool uploaded = false;      /* a walked texture had pending CPU data (deferred upload) */
    for (int t = 0; types[t].spvc_type; t++)
    {
        SpirvResourceList *list = &program->spirv_resources_list[stage][types[t].spvc_type];

        for (GLuint r = 0; r < list->count; r++)
        {
            SpirvResource *sres = &list->list[r];
            GLuint elems = sres->array_size ? sres->array_size : 1;

            for (GLuint e = 0; e < elems; e++)
            {
                /* GL semantics: image/sampler ARRAY elements are independent uniforms
                 * (mgli_sampler_elem_unit; see MGLRenderer.m) — element e's unit need
                 * not be base+e */
                GLuint unit = (types[t].image) ? mgl_res_elem_unit(sres, e)
                                               : sres->binding + e;

                if (unit >= TEXTURE_UNITS) continue;
                Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                              : mgl_texture_for_sampler(ctx, unit, sres->target);
                if (!tex) continue;             /* unbound unit: legal GL, no realize */
                if (tex->dirty_bits & DIRTY_TEXTURE_DATA) uploaded = true;
                (void)[self bindMTLTexture: tex];
            }
        }
    }
    /* A deferred upload encodes a blit into _currentCommandBuffer, but the kernels that
     * consume (and WRITE) these textures may be encoded on a different command buffer
     * that is committed EARLIER (the tess path's fresh compute_cb): GPU execution then
     * puts the upload after the kernels' image stores and the initial texel data
     * clobbers them (shader_image_load_store *GeometryStages). Committing the upload
     * and waiting here orders it before every later encoder on any buffer. */
    if (uploaded)
    {
        /* same commit+wait sequence as texdataSubmitAndWait (that method is not in the
         * private header; keep it local so no shared header grows a texdata decl) */
        [self endRenderEncoding];
        id<MTLCommandBuffer> cb = _currentCommandBuffer;
        if (cb && cb.status < MTLCommandBufferStatusCommitted)
        {
            [self commitCommandBufferWithAGXRecovery: cb];
            [cb waitUntilCompleted];
        }
        [self newCommandBuffer];
        ctx->state.dirty_bits = DIRTY_ALL;
    }
}

/* The geometry KERNEL's own texture walk (sampled + storage images): unlike the per-stage
 * helper above it goes through the program-binding introspection (array elements cover
 * consecutive units) and tolerates a program-pipeline draw entering geomDraw with
 * ctx->state.program == NULL (the composite is only published inside the kernel section).
 * Called right after endRenderEncoding in geomDraw: the tess import encoder opened next
 * stays open across the whole geometry pass, so any deferred upload (texdataEncodeUpload
 * opens a blit encoder) must happen before it exists. */
- (void) geomRealizeKernelStageTextures: (Program *) program geomProgram: (MGLGeomProgram *) gp
{
    struct { int spvc_type; bool image; } types[] = {
        { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
        { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
        { 0, false }
    };
    Program *saved_tex_program = ctx->state.program;

    if (!program) return;
    ctx->state.program = program;
    for (int t = 0; types[t].spvc_type; t++)
    {
        GLuint n = [self getProgramBindingCount: _GEOMETRY_SHADER type: types[t].spvc_type];

        for (GLuint i = 0; i < n; i++)
        {
            SpirvResource *sres = &STATE(program)->spirv_resources_list[_GEOMETRY_SHADER][types[t].spvc_type].list[i];
            int b = [self getProgramBinding: _GEOMETRY_SHADER type: types[t].spvc_type index: i];
            int cnt = 1;

            for (int a = 0; gp && a < gp->n_tex_arrays; a++)
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

/* ------------------------------------------------------------------ image atomics
 * SPIRV-Cross emulates GL image atomics with a side `device atomic_<T>* <name>_atomic
 * [[buffer(N)]]` argument (mgl_msl_fix_image_atomics linearizes the non-2D targets'
 * coordinates into the row-major layout computed here).  The buffer is a separate
 * zero-initialized MTLBuffer per image unit — only atomic operations see it: the
 * texture argument carries imageLoad/imageStore, so a mix of imageStore and an atomic
 * on the same texel does not alias (an emulation gap; GL tests that only use
 * imageAtomic* sequences — every CTS `*atomic*` case — read the old value back from
 * the atomic itself and never touch the texture). */

/* texel count of the sidecar of one bound image-unit view: the layout the MSL helpers
 * (and SPIRV-Cross's spvImage2DAtomicCoord macro for plain image2D, whose row pitch is
 * 16-texel aligned) index into */
static NSUInteger geomImageAtomicTexels(id<MTLTexture> view)
{
    if (!view) return 0;
    NSUInteger w = view.width, h = view.height;
    switch (view.textureType)
    {
        case MTLTextureType2DMultisample:
            /* image2DMS — the layout of mgl_spirv_lower_ms_images /
             * mgl_msl_fix_image_atomics (spvMglAIMS): 4 words per (texel, layer,
             * sample).  A texture2d_ms cannot be written by a shader, so every MS
             * image access goes through this buffer and the barrier-time flush
             * (geomResolveImageAtomics) rasterizes it back into the texture. */
            return w * h * view.sampleCount * 4u;
        case MTLTextureType2DMultisampleArray:
            return w * h * view.arrayLength * view.sampleCount * 4u;
        case MTLTextureType2D:
        {
            /* image2D / image2DRect — and image1D, which the MSL realizes as a
             * single-row 2D texture (spvImage2DAtomicCoord aligns the row to 16) */
            NSUInteger pitch = (w + 15u) & ~15u;
            return pitch * h;
        }
        case MTLTextureType3D:
            return w * h * view.depth;
        case MTLTextureType2DArray:
            return w * h * view.arrayLength;      /* a 1D array is a 2D array of height 1 */
        case MTLTextureTypeCube:
            return 6u * w * w;                   /* faces are square */
        case MTLTextureTypeCubeArray:
            return 6u * view.arrayLength * w * w;
        case MTLTextureTypeTextureBuffer:
            return w;
        default:
            return w * h;
    }
}

/* the "too big to sidecar" sink: a 4-byte buffer bound at the sidecar's slot
 * when geomImageAtomicBuffer refuses an oversized image, so the lowered
 * stores/atomics write somewhere legal instead of an unbound (NULL) buffer
 * argument — the values are lost, but nothing faults or OOMs */
- (id<MTLBuffer>) geomImageAtomicDummy
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);

    if (!g) return nil;
    if (!g->atom_dummy)
    {
        id<MTLBuffer> d = [_device newBufferWithLength:4 options:MTLResourceStorageModeShared];

        if (!d) return nil;
        memset(d.contents, 0, d.length);
        g->atom_dummy = (void *)CFBridgingRetain(d);
    }
    return (__bridge id<MTLBuffer>)g->atom_dummy;
}

- (id<MTLBuffer>) geomImageAtomicBuffer: (GLuint) unit view: (id<MTLTexture>) view
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    ImageUnit *iu;
    id<MTLBuffer> buf;    NSUInteger texels;

    if (!g || !view || unit >= TEXTURE_UNITS) return nil;
    iu = &STATE(image_units[unit]);
    texels = geomImageAtomicTexels(view);
    if (!texels) return nil;

    /* the MS sidecar costs as many WORDS as the texture has BYTES × its sample
     * count (one word per component per sample) — a huge multisample image
     * would double GPU memory and OOM the device (KHR advanced-sso-perSample
     * allocates a 10.3 GB MS texture; its sidecar killed the command buffer
     * with kIOGPUCommandBufferCallbackErrorOutOfMemory).  Refuse MS sidecars
     * past 1 GiB: the caller binds the tiny dump-only buffer instead, so the
     * lowered stores write harmlessly into it (the values are wrong — no
     * silent OOM, and the draw cannot fault on an unbound buffer argument).
     * Single-sample sidecars stay uncapped: a 16384² non-MS sidecar is 1 GiB
     * and KHR advanced-cast legitimately uses one. */
    if (view.sampleCount > 1 && (uint64_t)texels * 4 + 4 > (1ull << 30))
    {
        if (geomEnvRead(), gEnv.debug)
            fprintf(stderr, "geom: MS sidecar refused: %llu texels exceed the 1 GiB cap\n",
                    (unsigned long long)texels);
        return nil;
    }

    buf = (g->img_atom[unit].buf) ? (__bridge id<MTLBuffer>)(g->img_atom[unit].buf) : nil;
    if (buf && g->img_atom[unit].view == (__bridge void *)view &&
        g->img_atom[unit].fmt == iu->internalformat &&
        g->img_atom[unit].bytes == (size_t)(texels * 4))
    {
        g->img_atom[unit].dirty = true;         /* bound to a dispatch: contents may change */
        return buf;                             /* same binding: the contents persist */
    }

    /* drop the OLD record's retain (+1 from CFBridgingRetain) — leaking it per
     * rebind showed up in the CTS all-targets sweep, and the borrowed VIEW
     * pointer crashed geomMsFlushUnit once the GL texture was deleted */
    if (g->img_atom[unit].buf)
    {
        CFRelease((CFTypeRef)g->img_atom[unit].buf);
        g->img_atom[unit].buf = NULL;
    }
    if (g->img_atom[unit].view)
    {
        CFRelease((CFTypeRef)g->img_atom[unit].view);
        g->img_atom[unit].view = NULL;
    }
    buf = nil;                                  /* ARC release before overwriting */
    /* +1 DUMP word: the MSL index helpers (mgl_msl_fix_image_atomics) point
     * out-of-range image accesses at it instead of past the buffer */
    buf = [_device newBufferWithLength:texels * 4 + 4 options:MTLResourceStorageModeShared];
    if (!buf) return nil;
    memset(buf.contents, 0, buf.length);        /* atomics start from a cleared image */

    g->img_atom[unit].buf = (void *)CFBridgingRetain(buf);
    g->img_atom[unit].view = (void *)CFBridgingRetain(view);   /* OWNED: survives its GL texture */
    g->img_atom[unit].fmt = iu->internalformat;
    g->img_atom[unit].bytes = (size_t)(texels * 4);
    g->img_atom[unit].dirty = true;
    g->img_atom[unit].needs_seed = true;        /* seed from the texture before the next dispatch */
    g->img_atom[unit].ms = (view.textureType == MTLTextureType2DMultisample ||
                            view.textureType == MTLTextureType2DMultisampleArray);
    g->img_atom[unit].samples = g->img_atom[unit].ms ? (GLuint)view.sampleCount : 0;
    return buf;
}

/* context teardown: release every unit's sidecar */
- (void) geomReleaseImageAtomics
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    if (!g) return;
    for (int u = 0; u < TEXTURE_UNITS; u++)
    {
        if (g->img_atom[u].buf) { CFBridgingRelease(g->img_atom[u].buf); g->img_atom[u].buf = NULL; }
        if (g->img_atom[u].view) { CFBridgingRelease(g->img_atom[u].view); g->img_atom[u].view = NULL; }
        if (g->atom_dummy) { CFBridgingRelease(g->atom_dummy); g->atom_dummy = NULL; }
        g->img_atom[u].elems = 0;
        g->img_atom[u].sig = 0;
        g->img_atom[u].bytes = 0;
        g->img_atom[u].dirty = false;
        g->img_atom[u].needs_seed = false;
    }
}

/* The layout of an image-ARRAY's single side buffer, mirroring the MSL helper
 * mgl_msl_fix_image_atomics emits (spvMglAIArr_*): element e's region starts at
 * offset_e (in 32-bit words) and spans raw_width*height words — the same raw row
 * pitch SPIRV-Cross's spvImage2DAtomicCoord uses with the default function
 * constant (spvLinearTextureAlignment = 4 -> one-texel alignment).  Returns the
 * total word count and a cheap signature of the bound views. */
- (NSUInteger) geomImageAtomicArrayLayout: (GLuint) unit elems: (GLuint) elems
                                     offs: (NSUInteger *) offs maxe: (NSUInteger) maxe
                                      sig: (uint64_t *) sig
{
    NSUInteger total = 0;
    uint64_t s = 0;

    if (!offs || elems > maxe || unit + elems > TEXTURE_UNITS)
        return 0;
    for (GLuint e = 0; e < elems; e++)
    {
        ImageUnit *iu = &STATE(image_units[unit + e]);
        Texture *tex = iu->tex;
        id<MTLTexture> view;

        if (!tex || !tex->mtl_data)
            return 0;
        view = [self geomImageUnitView: iu base: (__bridge id<MTLTexture>)(tex->mtl_data)];
        if (!view)
            return 0;
        offs[e] = total;
        /* a multisample element is addressed by the (sample-folded) spvImage2DAtomic
         * coordinate, not the padded 2D row pitch — give it the full MS layout
         * (4 words per texel-sample) so an unlowered array access stays in bounds */
        total += (view.textureType == MTLTextureType2DMultisample ||
                  view.textureType == MTLTextureType2DMultisampleArray)
                     ? geomImageAtomicTexels(view)
                     : view.width * view.height;
        s ^= (uint64_t)(uintptr_t)(__bridge void *)view * 0x9E3779B97F4A7C15ULL
             ^ ((uint64_t)view.width * 0x85EBCA6BULL) ^ ((uint64_t)view.height * 0xC2B2AE35ULL)
             ^ ((uint64_t)e * 0x27D4EB2FULL);
    }
    if (sig)
        *sig = s;
    return total;
}

/* The ONE side buffer of an image ARRAY uniform (SPIRV-Cross declares a single
 * `<name>_atomic [[buffer(N)]]` for the whole array; its size covers every
 * element's region back to back — see geomImageAtomicArrayLayout). */
- (id<MTLBuffer>) geomImageAtomicBufferArray: (GLuint) unit elems: (GLuint) elems
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    NSUInteger offs[TEXTURE_UNITS];
    uint64_t sig = 0;
    NSUInteger words = [self geomImageAtomicArrayLayout: unit elems: elems offs: offs
                                                   maxe: TEXTURE_UNITS sig: &sig];
    id<MTLBuffer> buf;

    if (!g || !words)
        return nil;
    buf = (__bridge id<MTLBuffer>)g->img_atom[unit].buf;
    if (buf && g->img_atom[unit].elems == elems && g->img_atom[unit].sig == sig &&
        g->img_atom[unit].bytes == words * 4)
    {
        g->img_atom[unit].dirty = true;      /* a dispatch/encoder saw it */
        return buf;
    }
    buf = [_device newBufferWithLength: words * 4 + 4 options: MTLResourceStorageModeShared];  /* +1 DUMP word (OOB image accesses) */
    if (!buf)
        return nil;
    if (g->img_atom[unit].buf)
        CFBridgingRelease(g->img_atom[unit].buf);
    memset(buf.contents, 0, words * 4);
    g->img_atom[unit].buf = CFBridgingRetain(buf);
    if (g->img_atom[unit].view)
        CFBridgingRelease(g->img_atom[unit].view);
    g->img_atom[unit].view = NULL;
    g->img_atom[unit].fmt = 0;
    g->img_atom[unit].bytes = words * 4;
    g->img_atom[unit].elems = elems;
    g->img_atom[unit].sig = sig;
    g->img_atom[unit].dirty = true;
    g->img_atom[unit].needs_seed = true;     /* seed every element region before use */
    return buf;
}

/* ---- atomic sidecar -> texture resolve -------------------------------------------
 * The sidecar is the ONLY storage imageAtomic operations see (SPIRV-Cross emulation),
 * so results read back through the texture (glGetTexImage, sampling) need a copy.  The
 * kernel maps a linear thread index back onto the sidecar layout (spvImage2DAtomicCoord
 * for 2D — the row pitch is 16-aligned — mgl_msl_fix_image_atomics' helpers for the
 * rest, all matching geomImageAtomicTexels' length).  Only 32-bit single-channel
 * formats (r32i/r32ui/r32f) take part: the sidecar is one 32-bit word per texel. */
- (id<MTLComputePipelineState>) geomAtomResolvePso: (MTLTextureType) tt format: (MTLPixelFormat) pf
{
    static NSMutableDictionary<NSString *, id> *cache;
    NSString *key = [NSString stringWithFormat:@"%lu:%u", (unsigned long)tt, (unsigned)pf];
    id<MTLComputePipelineState> pso;

    if (!cache) cache = [NSMutableDictionary new];
    pso = cache[key];
    if (pso) return pso;

    const char *tex, *body, *ch;
    bool floatfmt = false;

    switch (pf)
    {
        case MTLPixelFormatR32Uint:  ch = "uint";  break;
        case MTLPixelFormatR32Sint:  ch = "int";   break;
        case MTLPixelFormatR32Float: ch = "float"; floatfmt = true; break;
        default: return nil;                     /* multi-channel / non-32-bit: unresolved */
    }

    switch (tt)
    {
        case MTLTextureType2D:
            tex = "texture2d";
            body = "uint pitch = (dims.x + 15u) & ~15u; uint x = t % dims.x; uint y = t / dims.x;\n"
                   "    uint v = side[pitch * y + x]; tex.write(V, uint2(x, y));";
            break;
        case MTLTextureType2DArray:
            tex = "texture2d_array";
            body = "uint wh = dims.x * dims.y; uint z = t / wh; uint r = t % wh;\n"
                   "    uint v = side[t]; tex.write(V, uint2(r % dims.x, r / dims.x), z);";
            break;
        case MTLTextureType3D:
            tex = "texture3d";
            body = "uint wh = dims.x * dims.y; uint z = t / wh; uint r = t % wh;\n"
                   "    uint v = side[t]; tex.write(V, uint3(r % dims.x, r / dims.x, z));";
            break;
        case MTLTextureTypeCube:
        case MTLTextureTypeCubeArray:
            tex = "texturecube_array";
            body = "uint ww = dims.x * dims.x; uint z = t / ww; uint r = t % ww;\n"
                   "    uint v = side[t]; tex.write(V, uint2(r % dims.x, r / dims.x), z);";
            break;
        case MTLTextureType1D:
            tex = "texture1d";
            body = "uint v = side[t]; tex.write(V, t);";
            break;
        case MTLTextureTypeTextureBuffer:
            tex = "texture_buffer";
            body = "uint v = side[t]; tex.write(V, t);";
            break;
        default:
            return nil;                          /* MS targets: Metal cannot write them */
    }

    /* every path assigns v then writes V (bitcast for r32f) — 2D reads the padded row */
    char src[2048];
    snprintf(src, sizeof src,
             "#include <metal_stdlib>\nusing namespace metal;\n"
             "kernel void mgl_atom_res(device const uint* side [[buffer(0)]],\n"
             "                        %s<%s, access::write> tex [[texture(0)]],\n"
             "                        constant uint4& dims [[buffer(1)]],\n"
             "                        uint t [[thread_position_in_grid]])\n"
             "{\n"
             "    if (t >= dims.w) return;\n"
             "    %s\n"
             "    (void)v;\n"
             "}\n",
             tex, floatfmt ? "float" : ch, body);

    /* splice the write in after the value load (2D: after the padded read) */
    {
        char withval[2400];
        const char *val = floatfmt ? "#define V float4(as_type<float>(v))\n"
                          : (pf == MTLPixelFormatR32Sint ? "#define V int4(int(v))\n"
                                                         : "#define V uint4(v)\n");
        snprintf(withval, sizeof withval, "%s%s", val, src);
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:
                                 [NSString stringWithUTF8String:withval]
                                                     options:nil error:&err];
        if (!lib)
        {
            if (gEnv.debug)
                fprintf(stderr, "geom: image-atomic resolve kernel compile failed: %s\n",
                        err.localizedDescription.UTF8String);
            return nil;
        }
        id<MTLFunction> fn = [lib newFunctionWithName: @"mgl_atom_res"];
        if (!fn) return nil;
        pso = [_device newComputePipelineStateWithFunction: fn error:&err];
        if (!pso) return nil;
        cache[key] = pso;
        return pso;
    }
}

/* ---- texture -> sidecar seed ------------------------------------------------------
 * The sidecar is the only storage imageAtomic* sees, so it has to start from the
 * image's current contents: imageAtomicCompSwap compares the OLD value
 * (KHR-GL46.texture_buffer texture_buffer_atomic_functions swaps the buffer's own
 * contents through the sidecar).  The kernel is the resolve's mirror image: same
 * padded sidecar layout, read from the texture instead of writing it. */
- (id<MTLComputePipelineState>) geomAtomSeedPso: (MTLTextureType) tt format: (MTLPixelFormat) pf
{
    static NSMutableDictionary<NSString *, id> *cache;
    NSString *key = [NSString stringWithFormat:@"seed:%lu:%u", (unsigned long)tt, (unsigned)pf];
    id<MTLComputePipelineState> pso;

    static dispatch_once_t once;
    dispatch_once(&once, ^{ cache = [NSMutableDictionary dictionary]; });
    @synchronized (cache) { pso = cache[key]; }
    if (pso) return pso;

    const char *tex = NULL, *body = NULL;
    bool floatfmt = (pf == MTLPixelFormatR32Float);
    const char *ch = floatfmt ? "float" : (pf == MTLPixelFormatR32Sint ? "int" : "uint");

    switch (tt)
    {
        case MTLTextureType2D:
            tex = "texture2d";
            body = "uint pitch = (dims.x + 15u) & ~15u; uint x = t % dims.x; uint y = t / dims.x;\n"
                   "    T v = tex.read(uint2(x, y)).x; side[pitch * y + x] = W;";
            break;
        case MTLTextureType2DArray:
            tex = "texture2d_array";
            body = "uint wh = dims.x * dims.y; uint z = t / wh; uint r = t % wh;\n"
                   "    T v = tex.read(uint2(r % dims.x, r / dims.x), z).x; side[t] = W;";
            break;
        case MTLTextureType3D:
            tex = "texture3d";
            body = "uint wh = dims.x * dims.y; uint z = t / wh; uint r = t % wh;\n"
                   "    T v = tex.read(uint3(r % dims.x, r / dims.x, z)).x; side[t] = W;";
            break;
        case MTLTextureTypeCube:
        case MTLTextureTypeCubeArray:
            tex = "texturecube_array";
            body = "uint ww = dims.x * dims.x; uint z = t / ww; uint r = t % ww;\n"
                   "    T v = tex.read(uint2(r % dims.x, r / dims.x), z).x; side[t] = W;";
            break;
        case MTLTextureType1D:
            tex = "texture1d";
            body = "T v = tex.read(t).x; side[t] = W;";
            break;
        case MTLTextureTypeTextureBuffer:
            tex = "texture_buffer";
            body = "T v = tex.read(t).x; side[t] = W;";
            break;
        default:
            return nil;                          /* MS targets: the sidecar is unused */
    }

    char src[2400];
    {
        const char *val = floatfmt ? "#define T float\n#define W as_type<uint>(v)\n"
                        : (pf == MTLPixelFormatR32Sint ? "#define T int\n#define W as_type<uint>(v)\n"
                                                       : "#define T uint\n#define W v\n");
        snprintf(src, sizeof src,
                 "%s#include <metal_stdlib>\nusing namespace metal;\n"
                 "kernel void mgl_atom_seed(device uint* side [[buffer(0)]],\n"
                 "                        %s<%s, access::read> tex [[texture(0)]],\n"
                 "                        constant uint4& dims [[buffer(1)]],\n"
                 "                        uint t [[thread_position_in_grid]])\n"
                 "{\n"
                 "    if (t >= dims.w) return;\n"
                 "    %s\n"
                 "    (void)0;\n"
                 "}\n",
                 val, tex, ch, body);
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:
                                 [NSString stringWithUTF8String:src]
                                                     options:nil error:&err];
        if (!lib)
        {
            if (gEnv.debug)
                fprintf(stderr, "geom: image-atomic seed kernel compile failed: %s\n",
                        err ? [[err localizedDescription] UTF8String] : "?");
            return nil;
        }
        id<MTLFunction> fn = [lib newFunctionWithName: @"mgl_atom_seed"];
        if (!fn) return nil;
        pso = [_device newComputePipelineStateWithFunction: fn error:&err];
        if (!pso) return nil;
        @synchronized (cache) { cache[key] = pso; }
        return pso;
    }
}

/* copy the texture into a freshly created sidecar on the given compute encoder
 * (returns true when it encoded something) */
- (bool) geomSeedImageAtomic: (int) unit
                    encoder: (id <MTLComputeCommandEncoder>) encoder
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    id<MTLTexture> view;
    id<MTLBuffer> buf;
    MTLTextureType tt;
    MTLPixelFormat pf;
    NSUInteger total;
    uint32_t dims[4];

    if (!g || !encoder || unit < 0 || unit >= TEXTURE_UNITS) return false;
    if (!g->img_atom[unit].buf || !g->img_atom[unit].needs_seed) return false;

    buf = (__bridge id<MTLBuffer>)(g->img_atom[unit].buf);

    /* an image ARRAY's single side buffer: seed each element region from its own
     * texture, through a 1-slice 2D-array view (the raw layout the MSL helper
     * spvMglAIArr_* indexes — NOT the scalar 16-texel-padded one) */
    if (g->img_atom[unit].elems > 1)
    {
        NSUInteger offs[TEXTURE_UNITS];
        NSUInteger words = [self geomImageAtomicArrayLayout: (GLuint)unit
                                                       elems: g->img_atom[unit].elems
                                                        offs: offs
                                                        maxe: TEXTURE_UNITS
                                                         sig: NULL];
        bool any = false;

        if (!words || words * 4 != g->img_atom[unit].bytes)
        { g->img_atom[unit].needs_seed = false; return false; }
        for (GLuint e = 0; e < g->img_atom[unit].elems; e++)
        {
            ImageUnit *iu2 = &STATE(image_units[unit + e]);
            id<MTLTexture> ev = (iu2->tex && iu2->tex->mtl_data)
                ? [self geomImageUnitViewRes: iu2 base: (__bridge id<MTLTexture>)(iu2->tex->mtl_data)
                                   elemClass: 0xffffffffu]
                : nil;
            id<MTLTexture> av;
            NSUInteger total_e;
            uint32_t d2[4];
            id<MTLComputePipelineState> pso2;

            if (!ev) continue;
            av = [ev newTextureViewWithPixelFormat: ev.pixelFormat
                                    textureType: MTLTextureType2DArray
                                         levels: NSMakeRange(0, 1)
                                         slices: NSMakeRange(0, 1)];
            if (!av) continue;
            total_e = ev.width * ev.height;
            pso2 = [self geomAtomSeedPso: MTLTextureType2DArray format: av.pixelFormat];
            if (!pso2 || !total_e) continue;
            d2[0] = (uint32_t)ev.width;
            d2[1] = (uint32_t)ev.height;
            d2[2] = 1;
            d2[3] = (uint32_t)total_e;
            [encoder setComputePipelineState: pso2];
            [encoder setBuffer: buf offset: offs[e] * 4 atIndex: 0];
            [encoder setTexture: av atIndex: 0];
            [encoder setBytes: &d2 length: sizeof(d2) atIndex: 1];
            [encoder dispatchThreads: MTLSizeMake(total_e, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
            any = true;
        }
        g->img_atom[unit].needs_seed = false;
        return any;
    }

    view = (__bridge id<MTLTexture>)(g->img_atom[unit].view);
    if (!buf || !view) { g->img_atom[unit].needs_seed = false; return false; }

    tt = view.textureType;
    pf = view.pixelFormat;
    total = geomImageAtomicTexels(view);
    if (!total) { g->img_atom[unit].needs_seed = false; return false; }

    /* multisample images: per-sample reads into the 4-words-per-texel-sample layout */
    if (tt == MTLTextureType2DMultisample || tt == MTLTextureType2DMultisampleArray)
    {
        NSUInteger L = (tt == MTLTextureType2DMultisampleArray) ? view.arrayLength : 1;
        id<MTLComputePipelineState> mpso = [self geomMsSeedPso: tt format: pf];
        if (!mpso || !view.sampleCount) { g->img_atom[unit].needs_seed = false; return false; }
        uint32_t md[4] = { (uint32_t)view.width, (uint32_t)view.height, (uint32_t)L,
                           (uint32_t)(view.width * view.height * L) };
        [encoder setComputePipelineState: mpso];
        [encoder setBuffer: buf offset: 0 atIndex: 0];
        [encoder setTexture: view atIndex: 0];
        [encoder setBytes: &md length: sizeof(md) atIndex: 1];
        [encoder dispatchThreads: MTLSizeMake(md[3], 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
        g->img_atom[unit].needs_seed = false;
        return true;
    }

    id<MTLComputePipelineState> pso = [self geomAtomSeedPso: tt format: pf];
    if (!pso) { g->img_atom[unit].needs_seed = false; return false; }

    dims[0] = (uint32_t)view.width;
    dims[1] = (uint32_t)(tt == MTLTextureType3D ? view.depth : view.height);
    if (tt == MTLTextureType2DArray)          dims[2] = (uint32_t)view.arrayLength;
    else if (tt == MTLTextureTypeCubeArray)   dims[2] = (uint32_t)view.arrayLength;
    else if (tt == MTLTextureTypeCube)        dims[2] = 6;
    else if (tt == MTLTextureType3D)          dims[2] = (uint32_t)view.depth;
    else                                      dims[2] = 1;
    dims[3] = (uint32_t)total;

    [encoder setComputePipelineState: pso];
    [encoder setBuffer: buf offset: 0 atIndex: 0];
    [encoder setTexture: view atIndex: 0];
    [encoder setBytes: &dims length: sizeof(dims) atIndex: 1];
    [encoder dispatchThreads: MTLSizeMake(total, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];
    g->img_atom[unit].needs_seed = false;
    return true;
}

/* ---- multisample sidecar flush ----
 * A multisample image's sidecar cannot be copied back with a compute write (Metal's
 * texture2d_ms is read-only), so it is rasterized back instead: one 1-pixel POINT per
 * texel, a render pipeline whose rasterSampleCount matches the texture, and a fragment
 * that runs PER SAMPLE ([[sample_id]] forces per-sample shading), reads the sample's
 * 4 sidecar words and outputs them with [[sample_mask]] = 1<<sample so exactly that
 * sample of the texel is replaced.  Metal rasterizes primitives in order, so the
 * sidecar's order is preserved.  LoadAction Load keeps samples that were never
 * written (r32* images only use word 0 of each texel-sample quad). */
- (id<MTLRenderPipelineState>) geomMsFlushPso: (MTLPixelFormat) pf
                                  sampleCount: (NSUInteger) sc
                                      arrayed: (bool) arrayed
{
    static NSMutableDictionary<NSString *, id> *cache;
    NSString *key = [NSString stringWithFormat: @"msflush:%u:%lu:%d", (unsigned)pf,
                                                         (unsigned long)sc, arrayed ? 1 : 0];
    id<MTLRenderPipelineState> pso;

    if (cache && (pso = cache[key])) return pso;

    /* channel type / count of the 32-bit image formats the MS lowering supports */
    const char *T = "float", *T4 = "float4", *FT = "as_type<float>", *W = "as_type<uint>";
    const char *V0 = "0.0", *V1 = "0.0";
    int C = 4;
    switch (pf)
    {
        case MTLPixelFormatR32Float:    case MTLPixelFormatRG32Float:    case MTLPixelFormatRGBA32Float:
            T = "float"; T4 = "float4"; FT = "as_type<float>"; W = "as_type<uint>";
            V0 = "0.0"; V1 = "1.0"; break;
        case MTLPixelFormatR32Sint:     case MTLPixelFormatRG32Sint:     case MTLPixelFormatRGBA32Sint:
            T = "int"; T4 = "int4"; FT = "static_cast<int>"; W = "as_type<uint>";
            V0 = "0"; V1 = "1"; break;
        case MTLPixelFormatR32Uint:     case MTLPixelFormatRG32Uint:     case MTLPixelFormatRGBA32Uint:
            T = "uint"; T4 = "uint4"; FT = "static_cast<uint>"; W = "";
            V0 = "0u"; V1 = "1u"; break;
        default:
            return nil;
    }
    switch (pf)
    {
        case MTLPixelFormatR32Float:  case MTLPixelFormatR32Sint:  case MTLPixelFormatR32Uint:   C = 1; break;
        case MTLPixelFormatRG32Float: case MTLPixelFormatRG32Sint: case MTLPixelFormatRG32Uint:  C = 2; break;
        default:                                                                                   C = 4; break;
    }

    char vs[1400], fs[2200];
    snprintf(vs, sizeof vs,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct MGLMsFV { float4 pos [[position]]; float psize [[point_size]]; %s };\n"
        "vertex MGLMsFV mgl_ms_flush_v(uint vid [[vertex_id]], constant uint4& dims [[buffer(0)]])\n"
        "{\n"
        "    uint wh = dims.x * dims.y;\n"
        "    uint layer = %s ? vid / wh : 0u;\n"
        "    uint r = %s ? vid %% wh : vid;\n"
        "    MGLMsFV o;\n"
        "    o.psize = 1.0;\n"      /* AGX rasterizes NOTHING for points without an
                                      * explicit [[point_size]] (verified against a
                                      * standalone render: nan before, per-sample
                                      * writes after) */
        "    o.pos = float4((float(r %% dims.x) + 0.5) / float(dims.x) * 2.0 - 1.0,\n"
        "                   (float(r / dims.x) + 0.5) / float(dims.y) * 2.0 - 1.0, 0.0, 1.0);\n"
        "    %s\n"
        "    return o;\n"
        "}\n",
        arrayed ? "uint rti [[render_target_array_index]];" : "",
        arrayed ? "1u" : "0u", arrayed ? "1u" : "0u",
        arrayed ? "o.rti = layer;" : "(void)layer;");

    const char *col =
        C == 1 ? "(FT(side[b + 0]), T(V0), T(V0), T(V0))" :
        (C == 2 ? "(FT(side[b + 0]), FT(side[b + 1]), T(V0), T(V0))" :
                  "(FT(side[b + 0]), FT(side[b + 1]), FT(side[b + 2]), FT(side[b + 3]))");
    char colbuf[320];
    {
        /* the template spells FT / T / V0 / V1 as tokens — substitute this format's
         * spellings (FT first: it contains "T").  The unexpanded tokens made the
         * whole flush FS fail to compile, so the sidecar NEVER rasterized back. */
        char tmp[320];
        size_t n = 0;

        n += (size_t)snprintf(tmp + n, sizeof tmp - n, "%s", T4);
        for (const char *p = col; *p && n + 1 < sizeof tmp; )
        {
            if (p[0] == 'F' && p[1] == 'T' && p[2] == '(')
            { n += (size_t)snprintf(tmp + n, sizeof tmp - n, "%s(", FT); p += 3; }
            else if (p[0] == 'T' && p[1] == '(')
            { n += (size_t)snprintf(tmp + n, sizeof tmp - n, "%s(", T); p += 2; }
            else if (p[0] == 'V' && p[1] == '0')
            { n += (size_t)snprintf(tmp + n, sizeof tmp - n, "%s", V0); p += 2; }
            else if (p[0] == 'V' && p[1] == '1')
            { n += (size_t)snprintf(tmp + n, sizeof tmp - n, "%s", V1); p += 2; }
            else tmp[n++] = *p++;
        }
        tmp[n] = 0;
        snprintf(colbuf, sizeof colbuf, "%s", tmp);
    }

    snprintf(fs, sizeof fs,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct MGLMsFF { %s col [[color(0)]]; uint mask [[sample_mask]]; };\n"
        "fragment MGLMsFF mgl_ms_flush_f(float4 pos [[position]], uint sid [[sample_id]],\n"
        "                                %s device const uint* side [[buffer(1)]],\n"
        "                                constant uint4& dims [[buffer(0)]])\n"
        "{\n"
        "    uint2 c((uint)pos.x, (uint)pos.y);\n"
        "    uint S = dims.w;\n"
        "    uint layer = %s;\n"
        "    uint b = 4u * (sid + S * (layer + dims.z * (c.x + dims.x * c.y)));\n"
        "    MGLMsFF o;\n"
        "    o.col = %s;\n"
        "    o.mask = 1u << sid;\n"
        "    return o;\n"
        "}\n",
        T4, arrayed ? "uint layer_id [[render_target_array_index]]," : "", arrayed ? "layer_id" : "0u", colbuf);
    (void)T; (void)W; (void)V1;

    NSError *err = nil;
    /* options nil, like every other runtime kernel this renderer builds: the AGX
     * driver's newLibraryWithSource rejects a non-nil options object here */
    id<MTLLibrary> lib = [_device newLibraryWithSource: [NSString stringWithUTF8String: vs]
                                               options: nil error: &err];
    if (!lib)
    {
        if (gEnv.debug) fprintf(stderr, "geom: MS flush VS compile failed: %s\n",
                                err.localizedDescription.UTF8String);
        return nil;
    }
    id<MTLLibrary> flib = nil;
    for (int attempt = 0; attempt < 3 && !flib; attempt++)
    {
        err = nil;
        flib = [_device newLibraryWithSource: [NSString stringWithUTF8String: fs]
                                     options: nil error: &err];
        if (!flib && !err)
            usleep(50000);   /* macOS quirk: a concurrently-running compile can
                              * fail with no error — retry */
    }
    if (!flib)
    {
        if (geomEnvRead(), gEnv.debug) fprintf(stderr, "geom: MS flush FS compile failed: dom=%s code=%ld src=%s\n",
                                err.domain.UTF8String ?: "(null)", (long)err.code, fs);
        return nil;
    }

    MTLRenderPipelineDescriptor *d = [[MTLRenderPipelineDescriptor alloc] init];
    d.label = @"mgl-ms-image-flush";
    d.vertexFunction = [lib newFunctionWithName: @"mgl_ms_flush_v"];
    d.fragmentFunction = [flib newFunctionWithName: @"mgl_ms_flush_f"];
    d.rasterSampleCount = sc;
    if (arrayed)
    {
        d.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;   /* required when the
                                                                      * VS writes rti */
    }
    d.colorAttachments[0].pixelFormat = pf;
    pso = [_device newRenderPipelineStateWithDescriptor: d error: &err];
    if (!pso)
    {
        if (getenv("MGL_FLUSH_DUMP") || (geomEnvRead(), gEnv.debug))
            fprintf(stderr, "geom: MS flush PSO failed: %s\nfs=%s\nvs=%s\n",
                    err.localizedDescription.UTF8String, fs, vs);
        return nil;
    }
    if (getenv("MGL_FLUSH_DUMP"))
        fprintf(stderr, "geom: MS flush FS (pf %u C %d):\n%s\nVS:\n%s\n", (unsigned)pf, C, fs, vs);
    if (!cache) cache = [NSMutableDictionary new];
    cache[key] = pso;
    return pso;
}

/* rasterize one unit's multisample sidecar back into its texture */
- (bool) geomMsFlushUnit: (int) unit
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    id<MTLBuffer> buf;
    id<MTLTexture> view;
    bool arrayed;
    uint32_t dims[4];
    NSUInteger total, L;

    if (!g || unit < 0 || unit >= TEXTURE_UNITS || !g->img_atom[unit].buf) return false;
    buf = (__bridge id<MTLBuffer>)(g->img_atom[unit].buf);
    view = (__bridge id<MTLTexture>)(g->img_atom[unit].view);
    if (!buf || !view) return false;

    arrayed = (view.textureType == MTLTextureType2DMultisampleArray);
    L = arrayed ? view.arrayLength : 1;
    total = view.width * view.height * L;
    if (!total || !view.sampleCount) return false;

    if (getenv("MGL_GEOM_DEBUG"))
    {
        /* the store path must have left its words in the sidecar BEFORE the flush:
         * word 0..3 = texel(0,0) sample0, and texel(1,0) sample3 */
        volatile uint32_t *w32 = (volatile uint32_t *)buf.contents;
        fprintf(stderr, "MGL geom: ms flush PRE buf=%p len=%zu words=%lu NONZERO at:",
                (__bridge void *)buf, buf.length, (unsigned long)(buf.length / 4));
        {
            int shown = 0;
            for (NSUInteger i = 0; i < buf.length / 4 && shown < 24; i++)
                if (w32[i]) { fprintf(stderr, " [%lu]=%08x", (unsigned long)i, w32[i]); shown++; }
        }
        fprintf(stderr, "\n");
    }

    id<MTLRenderPipelineState> pso = [self geomMsFlushPso: view.pixelFormat
                                              sampleCount: view.sampleCount
                                                  arrayed: arrayed];
    if (!pso)
    {
        if (getenv("MGL_GEOM_DEBUG"))
            fprintf(stderr, "MGL geom: ms flush unit %d PSO FAILED (pf %u sc %lu arr %d)\n",
                    unit, (unsigned)view.pixelFormat, (unsigned long)view.sampleCount, arrayed);
        return false;
    }

    dims[0] = (uint32_t)view.width;
    dims[1] = (uint32_t)view.height;
    dims[2] = (uint32_t)L;
    dims[3] = (uint32_t)view.sampleCount;

    /* an OWN command buffer, committed and waited synchronously: the flush must
     * not interleave with the shared one (a second render encoder on a buffer
     * whose encoder the draw path still appends to proved not to execute) */
    id<MTLCommandBuffer> cb = [_commandQueue commandBuffer];
    if (!cb) return false;
    cb.label = @"mgl-ms-image-flush";

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = view;
    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (getenv("MGL_FLUSH_CLEAR"))       /* debug: prove the pass itself runs */
    {
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(7.75, 5.5, 3.25, 1.0);
    }
    if (arrayed) rp.renderTargetArrayLength = L;

    id<MTLRenderCommandEncoder> renc = [cb renderCommandEncoderWithDescriptor: rp];
    if (getenv("MGL_GEOM_DEBUG") && !renc)
        fprintf(stderr, "MGL geom: ms flush ENCODER NIL (pf %u sc %lu arr %d tex %p %lux%lu s=%lu)\n",
                (unsigned)view.pixelFormat, (unsigned long)view.sampleCount, arrayed,
                (__bridge void *)view, (unsigned long)view.width, (unsigned long)view.height,
                (unsigned long)view.sampleCount);
    renc.label = @"mgl-ms-image-flush";
    [renc setRenderPipelineState: pso];
    [renc setViewport: (MTLViewport){ 0, 0, (double)view.width, (double)view.height, 0.0, 1.0 }];
    [renc setVertexBytes: &dims length: sizeof(dims) atIndex: 0];
    [renc setFragmentBuffer: buf offset: 0 atIndex: 1];
    [renc setFragmentBytes: &dims length: sizeof(dims) atIndex: 0];
    /* chunked: a single multi-million-point draw of a huge multisample image
     * pushed the render pass over the GPU's transient allocation and died with
     * kIOGPUCommandBufferCallbackErrorOutOfMemory — 8M points per call is well
     * under any tile/vertex working set and costs nothing when total is small */
    for (NSUInteger base = 0; base < total; base += 8u * 1024u * 1024u)
    {
        NSUInteger n = total - base;

        if (n > 8u * 1024u * 1024u) n = 8u * 1024u * 1024u;
        [renc drawPrimitives: MTLPrimitiveTypePoint vertexStart: base vertexCount: n];
    }
    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL geom: ms flush DRAW pf=%u total=%lu dims=%u,%u,%u,%u sc=%lu enc=%s\n",
                (unsigned)view.pixelFormat, (unsigned long)total, dims[0], dims[1], dims[2], dims[3],
                (unsigned long)view.sampleCount, renc ? "ok" : "NIL");
    [renc endEncoding];
    id<MTLBuffer> vbVerify = nil;
    if (getenv("MGL_FLUSH_VERIFY"))
    {
        /* DEBUG: read the flushed texture's texel (0,0), samples 0..S-1, back into a
         * shared buffer through a compute kernel and print — proves whether the
         * render pass above actually rasterized the sidecar into the texture */
        vbVerify = [_device newBufferWithLength:128 options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = vbVerify;
        if (vb)
        {
            memset(vb.contents, 0, vb.length);
            NSString *vsrc = @"#include <metal_stdlib>\nusing namespace metal;\n\
kernel void vkernel(device uint* out [[buffer(0)]], texture2d_ms<uint, access::read> t [[texture(0)]], constant uint4& dims [[buffer(1)]], uint3 g [[thread_position_in_grid]]) {\n\
    if (g.x != 0u || g.y != 0u) return;\n\
    for (uint s = 0u; s < 4u; s++) { uint4 v = t.read(uint2(0u, 0u), s); \
        out[s * 4u + 0u] = v.x; out[s * 4u + 1u] = v.y; \
        out[s * 4u + 2u] = v.z; out[s * 4u + 3u] = v.w; } \
    uint2 pc[5] = { uint2(1u,0u), uint2(0u,1u), uint2(2u,2u), uint2(3u,3u), uint2(7u,7u) }; \
    for (uint k = 0u; k < 5u; k++) for (uint s = 0u; s < 4u; s++) { uint4 v = t.read(pc[k], s); \
        out[16u + k * 4u + 0u] = v.x; out[16u + k * 4u + 1u] = v.y; \
        out[16u + k * 4u + 2u] = v.z; out[16u + k * 4u + 3u] = v.w; }\n}\n";
            NSError *verr = nil;
            id<MTLLibrary> vlib = [_device newLibraryWithSource:vsrc options:nil error:&verr];
            if (vlib)
            {
                id<MTLComputePipelineState> vpso = [_device newComputePipelineStateWithFunction:
                                                    [vlib newFunctionWithName:@"vkernel"] error:&verr];
                if (vpso)
                {
                    id<MTLComputeCommandEncoder> venc = [cb computeCommandEncoder];
                    [venc setComputePipelineState:vpso];
                    [venc setBuffer:vb offset:0 atIndex:0];
                    [venc setTexture:view atIndex:0];
                    [venc setBytes:&dims length:sizeof(dims) atIndex:1];
                    [venc dispatchThreads:MTLSizeMake(1, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [venc endEncoding];
                }
            }
            else if (getenv("MGL_GEOM_DEBUG"))
                fprintf(stderr, "MGL geom: ms flush VERIFY compile failed: %s\n",
                        verr ? verr.localizedDescription.UTF8String : "?");
            /* print after commit+wait below */
            (void)vb;
        }
    }
    [cb commit];
    /* no waitUntilCompleted: the flush is ordered against everything before it
     * (own buffer, same queue) and everything after it (later buffers on the
     * queue); only the DEBUG paths below read GPU results from the CPU */
    if (getenv("MGL_GEOM_DEBUG")) [cb waitUntilCompleted];

    if (getenv("MGL_FLUSH_VERIFY") && getenv("MGL_GEOM_DEBUG") && vbVerify)
    {
        volatile uint32_t *o = (volatile uint32_t *)[vbVerify contents];
        fprintf(stderr, "MGL geom: ms flush VERIFY cbstatus=%ld tex(0,0) s0=%g %g %g %g "
                "s1=%g %g %g %g s2=%g %g %g %g s3=%g %g %g %g "
                "(3,3)s0-3=%u %u %u %u (err=%s)\n",
                (long)cb.status,
                *(volatile float *)&o[0], *(volatile float *)&o[1],
                *(volatile float *)&o[2], *(volatile float *)&o[3],
                *(volatile float *)&o[4], *(volatile float *)&o[5],
                *(volatile float *)&o[6], *(volatile float *)&o[7],
                *(volatile float *)&o[8], *(volatile float *)&o[9],
                *(volatile float *)&o[10], *(volatile float *)&o[11],
                *(volatile float *)&o[12], *(volatile float *)&o[13],
                *(volatile float *)&o[14], *(volatile float *)&o[15],
                *(volatile float *)&o[16], *(volatile float *)&o[17],
                *(volatile float *)&o[18], *(volatile float *)&o[19],
                o[28], o[29], o[30], o[31],                cb.error ? cb.error.localizedDescription.UTF8String : "nil");
    }
    return true;
}

/* the MS seed kernel: one thread per (texel, layer) copies every sample's words out of
 * the texture2d_ms (a READ — the one thing Metal lets us do with it) */
- (id<MTLComputePipelineState>) geomMsSeedPso: (MTLTextureType) tt format: (MTLPixelFormat) pf
{
    static NSMutableDictionary<NSString *, id> *cache;
    NSString *key = [NSString stringWithFormat: @"msseed:%lu:%u", (unsigned long)tt, (unsigned)pf];
    id<MTLComputePipelineState> pso;

    if (cache && (pso = cache[key])) return pso;

    const char *T = "float", *W = "as_type<uint>";
    int C = 4;
    switch (pf)
    {
        case MTLPixelFormatR32Float:   T = "float"; W = "as_type<uint>"; C = 1; break;
        case MTLPixelFormatRG32Float:  T = "float"; W = "as_type<uint>"; C = 2; break;
        case MTLPixelFormatRGBA32Float:T = "float"; W = "as_type<uint>"; C = 4; break;
        case MTLPixelFormatR32Sint:    T = "int";   W = "as_type<uint>"; C = 1; break;
        case MTLPixelFormatRG32Sint:   T = "int";   W = "as_type<uint>"; C = 2; break;
        case MTLPixelFormatRGBA32Sint: T = "int";   W = "as_type<uint>"; C = 4; break;
        case MTLPixelFormatR32Uint:    T = "uint";  W = "";              C = 1; break;
        case MTLPixelFormatRG32Uint:   T = "uint";  W = "";              C = 2; break;
        case MTLPixelFormatRGBA32Uint: T = "uint";  W = "";              C = 4; break;
        default: return nil;
    }
    bool arr = (tt == MTLTextureType2DMultisampleArray);
    char src[2600];
    snprintf(src, sizeof src,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mgl_ms_seed(device uint* side [[buffer(0)]],\n"
        "                        %s<%s, access::read> tex [[texture(0)]],\n"
        "                        constant uint4& dims [[buffer(1)]],\n"
        "                        uint t [[thread_position_in_grid]])\n"
        "{\n"
        "    if (t >= dims.w) return;\n"
        "    uint wh = dims.x * dims.y;\n"
        "    uint layer = %s ? t / wh : 0u;\n"
        "    uint r = %s ? t %% wh : t;\n"
        "    uint2 c(r %% dims.x, r / dims.x);\n"
        "    uint S = tex.get_num_samples();\n"
        "    for (uint s = 0; s < S; s++)\n"
        "    {\n"
        "        vec<%s, %d> v = %s;\n"
        "        uint b = 4u * (s + S * (%s));\n"
        "        %s\n"
        "    }\n"
        "}\n",
        arr ? "texture2d_ms_array" : "texture2d_ms", T,
        arr ? "1u" : "0u", arr ? "1u" : "0u",
        T, C,
        arr ? "tex.read(c, layer, s)" : "tex.read(c, s)",
        arr ? "layer + dims.z * (c.x + dims.x * c.y)" : "c.y * dims.x + c.x",
        C == 4 ? "side[b+0] = W1(v.x); side[b+1] = W1(v.y); side[b+2] = W1(v.z); side[b+3] = W1(v.w);"
               : (C == 2 ? "side[b+0] = W1(v.x); side[b+1] = W1(v.y);"
                         : "side[b+0] = W1(v.x);"));

    char wmac[80];
    snprintf(wmac, sizeof wmac, "#define W1(x) %s(x)\n", W ? W : "");
    NSString *full = [NSString stringWithFormat: @"%s%s", wmac, src];

    NSError *err = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.preserveInvariance = false;
    id<MTLLibrary> lib = [_device newLibraryWithSource: full options: opts error: &err];
    if (!lib)
    {
        if (gEnv.debug) fprintf(stderr, "geom: MS seed compile failed (fmt %u): %s\n",
                                (unsigned)pf, err.localizedDescription.UTF8String);
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName: @"mgl_ms_seed"];
    if (!fn) return nil;
    pso = [_device newComputePipelineStateWithFunction: fn error: &err];
    if (!pso) return nil;
    if (!cache) cache = [NSMutableDictionary new];
    cache[key] = pso;
    return pso;
}

/* copy every dirty sidecar back into its texture: called at glMemoryBarrier time (the
 * CTS's texture-readback tests barrier before glGetTexImage) — cheap when nothing is
 * dirty, and it ends the render encoder only when a resolve is actually needed */
- (void) geomResolveImageAtomics
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    bool any = false;

    if (!g) return;
    for (int u = 0; u < TEXTURE_UNITS && !any; u++)
        if (g->img_atom[u].buf && g->img_atom[u].dirty) any = true;
    if (!any) return;

    [self endRenderEncoding];

    /* the draws that filled a multisample sidecar may still sit UNCOMMITTED in the
     * shared command buffer (the store path is a fragment/compute write to the
     * sidecar, recorded like any other draw): commit it, or the flush below
     * rasterizes a stale, all-zero sidecar into the texture and then drops the
     * dirty flag — the words would never reach the texture.  Commit WITHOUT
     * waiting: the flush runs on its own command buffer from the same queue, and
     * Metal orders buffers on a queue, so the flush still sees the finished
     * sidecar — and the CPU never stalls (a wait here cost two synchronous
     * round-trips per barrier and made the CTS store case crawl).  Only paid when
     * an MS sidecar is actually dirty (geomImageAtomicBuffer sets it), never on
     * the plain-image path. */
    bool any_ms = false;
    for (int u = 0; u < TEXTURE_UNITS; u++)
        if (g->img_atom[u].buf && g->img_atom[u].dirty && g->img_atom[u].ms && !g->img_atom[u].elems)
            any_ms = true;
    if (any_ms)
    {
        [self flushCommandBuffer: false];
        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
            [self newCommandBuffer];
    }

    /* multisample sidecars rasterize back into their textures first (render passes,
     * one per unit); the compute resolve below then handles the rest */
    for (int u = 0; u < TEXTURE_UNITS; u++)
        if (g->img_atom[u].buf && g->img_atom[u].dirty && g->img_atom[u].ms && !g->img_atom[u].elems)
        {
            if (getenv("MGL_GEOM_DEBUG"))
            {
                Texture *tptr = STATE(image_units[u]).tex;
                id<MTLTexture> live = (tptr && tptr->mtl_data)
                                          ? (__bridge id<MTLTexture>)tptr->mtl_data : nil;
                id<MTLTexture> sv = (__bridge id<MTLTexture>)g->img_atom[u].view;
                NSLog(@"MGL geom: ms flush unit %d sideview=%p livetex=%p SAME=%d",
                      u, (__bridge void *)sv, (__bridge void *)live, (live == sv) ? 1 : 0);
            }
            if ([self geomMsFlushUnit: u]) g->img_atom[u].dirty = false;
            else if (gEnv.debug)
                fprintf(stderr, "geom: MS sidecar flush skipped (unit %d, no PSO?)\n", u);
        }

    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];

    for (int u = 0; u < TEXTURE_UNITS; u++)
    {
        id<MTLBuffer> buf;
        id<MTLTexture> view;
        id<MTLComputePipelineState> pso;
        uint32_t dims[4];
        NSUInteger total;
        MTLTextureType tt;
        MTLPixelFormat pf;

        if (!g->img_atom[u].buf || !g->img_atom[u].dirty) continue;
        buf = (__bridge id<MTLBuffer>)(g->img_atom[u].buf);
        view = (__bridge id<MTLTexture>)(g->img_atom[u].view);
        g->img_atom[u].dirty = false;
        if (!buf) continue;

        /* an image ARRAY's single side buffer: resolve every element region into
         * its own texture (the layout mirrors geomImageAtomicArrayLayout) */
        if (g->img_atom[u].elems > 1)
        {
            NSUInteger offs[TEXTURE_UNITS];
            NSUInteger words = [self geomImageAtomicArrayLayout: (GLuint)u
                                                           elems: g->img_atom[u].elems
                                                            offs: offs
                                                            maxe: TEXTURE_UNITS
                                                             sig: NULL];
            if (!words || words * 4 != g->img_atom[u].bytes)
                continue;                     /* the binding changed since the dispatch */
            for (GLuint e = 0; e < g->img_atom[u].elems; e++)
            {
                ImageUnit *iu2 = &STATE(image_units[u + e]);
                id<MTLTexture> ev = (iu2->tex && iu2->tex->mtl_data)
                    ? [self geomImageUnitViewRes: iu2 base: (__bridge id<MTLTexture>)(iu2->tex->mtl_data)
                                     elemClass: 0xffffffffu]
                    : nil;
                id<MTLTexture> av;              /* a 1-slice 2D-array view: the raw (unpadded)
                                                  * sidecar layout, exactly what the MSL
                                                  * spvMglAIArr_* helper indexes */
                NSUInteger total_e;

                if (!ev) continue;
                av = [ev newTextureViewWithPixelFormat: ev.pixelFormat
                                        textureType: MTLTextureType2DArray
                                             levels: NSMakeRange(0, 1)
                                             slices: NSMakeRange(0, 1)];
                if (!av) continue;
                total_e = ev.width * ev.height;
                pso = [self geomAtomResolvePso: MTLTextureType2DArray format: av.pixelFormat];
                if (!pso || !total_e) continue;
                dims[0] = (uint32_t)ev.width;
                dims[1] = (uint32_t)ev.height;
                dims[2] = 1;
                dims[3] = (uint32_t)total_e;
                [enc setComputePipelineState: pso];
                [enc setBuffer: buf offset: offs[e] * 4 atIndex: 0];
                [enc setTexture: av atIndex: 0];
                [enc setBytes: &dims length: sizeof(dims) atIndex: 1];
                [enc dispatchThreads: MTLSizeMake(total_e, 1, 1)
                          threadsPerThreadgroup: MTLSizeMake(64 > pso.maxTotalThreadsPerThreadgroup ?
                                                             pso.maxTotalThreadsPerThreadgroup : 64, 1, 1)];
            }
            continue;
        }
        if (!view) continue;

        tt = view.textureType;
        pf = view.pixelFormat;
        /* only the REAL texels resolve — the sidecar may carry a padded row (2D pitch
         * 16) and its tail must not be written back (nor read out of bounds) */
        dims[0] = (uint32_t)view.width;
        dims[1] = (uint32_t)(tt == MTLTextureType3D ? view.depth : view.height);
        if (tt == MTLTextureType2DArray)          dims[2] = (uint32_t)view.arrayLength;
        else if (tt == MTLTextureTypeCube)        dims[2] = 6;
        else if (tt == MTLTextureTypeCubeArray)   dims[2] = (uint32_t)(view.arrayLength * 6);
        else if (tt == MTLTextureType3D)          dims[2] = (uint32_t)view.depth;
        else                                     dims[2] = 1;
        if (!geomImageAtomicTexels(view)) continue;
        total = (NSUInteger)dims[0] * dims[1] * dims[2];
        pso = [self geomAtomResolvePso: tt format: pf];
        if (!pso)
        {
            if (gEnv.debug)
                fprintf(stderr, "geom: image-atomic resolve skipped: unit %d type %lu fmt %u\n",
                        u, (unsigned long)tt, (unsigned)pf);
            continue;
        }
        if (getenv("MGL_GEOM_DEBUG"))
        {
            const uint32_t *sv = (const uint32_t *)buf.contents;
            fprintf(stderr, "MGL geom: img-atom resolve dump unit=%d side[%llu]=%u %u %u %u\n",
                    u, (unsigned long long)total, sv ? sv[0] : 0, sv && total > 1 ? sv[1] : 0,
                    sv && total > 2 ? sv[2] : 0, sv && total > 3 ? sv[3] : 0);
        }
        dims[3] = (uint32_t)total;

        dims[0] = (uint32_t)view.width;
        dims[1] = (uint32_t)(tt == MTLTextureType3D ? view.depth : view.height);
        if (tt == MTLTextureType2DArray)          dims[2] = (uint32_t)view.arrayLength;
        else if (tt == MTLTextureTypeCube)        dims[2] = 6;
        else if (tt == MTLTextureTypeCubeArray)   dims[2] = (uint32_t)(view.arrayLength * 6);
        else if (tt == MTLTextureType3D)          dims[2] = (uint32_t)view.depth;
        else                                     dims[2] = 1;
        dims[3] = (uint32_t)total;

        [enc setComputePipelineState: pso];
        [enc setBuffer: buf offset: 0 atIndex: 0];
        [enc setTexture: view atIndex: 0];
        [enc setBytes: &dims length: sizeof(dims) atIndex: 1];
        if (getenv("MGL_GEOM_DEBUG"))
            fprintf(stderr, "MGL geom: img-atom resolve unit=%d type=%lu w=%u total=%u\n", u,
                    (unsigned long)tt, dims[0], dims[3]);
        [enc dispatchThreads: MTLSizeMake(total, 1, 1)
        threadsPerThreadgroup: MTLSizeMake(64 > pso.maxTotalThreadsPerThreadgroup ?
                                           pso.maxTotalThreadsPerThreadgroup : 64, 1, 1)];

        /* r63: a TEXTURE BUFFER's atomics resolve into its SHADOW texture (buffer
         * textures are converted copies, mgl_uses_shadow): the GL buffer itself would
         * still read the pre-draw bytes through glMapBufferRange / glGetBufferSubData.
         * Blit the resolved shadow back into the GL buffer's storage (the compute
         * encoder above and this blit encoder are ordered within the command buffer).
         * KHR-GL46 shader_image_load_store.advanced-allStages-oneImage,
         * texture_buffer.texture_buffer_atomic_functions ("Buffer value is 0"). */
        if (tt == MTLTextureTypeTextureBuffer)
        {
            ImageUnit *iu2 = &STATE(image_units[u]);
            MGLTexExtInfo *info = (iu2->tex) ? mgl_texext_info(ctx, iu2->tex) : NULL;
            Buffer *gb = (info && info->buffer_name)
                         ? (Buffer *)searchHashTable(&ctx->state.buffer_table, info->buffer_name)
                         : NULL;
            id<MTLBuffer> shadow = [(__bridge id<MTLTexture>)iu2->tex->mtl_data buffer];

            if (info && info->kind == MGL_TEXEXT_BUFFER && info->mtl_uses_shadow &&
                gb && gb->data.mtl_data && shadow && iu2->tex->mtl_data)
            {
                size_t bytes = (size_t)total * 4;
                GLintptr dst = info->buffer_ranged ? info->buffer_offset : 0;

                if ((size_t)dst + bytes > (size_t)gb->size) bytes = ((size_t)gb->size > (size_t)dst)
                                                                 ? (size_t)gb->size - (size_t)dst : 0;
                if (bytes)
                {
                    id<MTLBlitCommandEncoder> bl = [_currentCommandBuffer blitCommandEncoder];

                    if (bl)
                    {
                        [bl copyFromBuffer: shadow sourceOffset: (NSUInteger)view.bufferOffset
                                    toBuffer: (__bridge id<MTLBuffer>)gb->data.mtl_data
                           destinationOffset: (NSUInteger)dst size: bytes];
                        [bl endEncoding];
                        if (gEnv.debug)
                            fprintf(stderr, "MGL geom: img-atom buffer writeback unit=%d %zu bytes @%ld\n",
                                    u, bytes, (long)dst);
                    }
                }
            }
        }
    }
    [enc endEncoding];
    ctx->state.dirty_bits |= DIRTY_ALL;      /* the encoder state must be rebuilt */
}

/* bind the sidecars of a raster stage's storage images (vertex or fragment slots of
 * the current render encoder); called from bindTextures (MGLRenderer.m) right after
 * the image-unit textures are bound */
- (bool) geomBindImageAtomicBuffersRender: (int) stage
{
    if (!_currentRenderEncoder) return false;
    GLuint count = [self getProgramBindingCount: stage type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL geom: img-atom bind render stage %d: %u storage images\n", stage, count);
    for (GLuint i = 0; i < count; i++)
    {
        SpirvResource *sres = &STATE(program)->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        GLuint unit = sres->binding, mtl_index;
        GLuint elems = sres->array_size ? sres->array_size : 1;
        bool ms_slot;

        if (sres->msl_atomic == 0xffffffffu) continue;  /* no atomic argument in the MSL */
        mtl_index = (sres->msl_atomic & ~MGL_GEOM_MS_SLOT);
        ms_slot = (sres->msl_atomic & MGL_GEOM_MS_SLOT) != 0;
        for (GLuint e = 0; e < elems; e++)
        {
            if (getenv("MGL_GEOM_DEBUG"))
                fprintf(stderr, "MGL geom: img-atom bind stage %d unit %u slot %u (tex %s)\n",
                        stage, unit + e, mtl_index + e, sres->name ? sres->name : "?");
            ImageUnit *iu;
            Texture *ptr;
            id<MTLTexture> texture;
            id<MTLBuffer> buf;

            if (unit + e >= TEXTURE_UNITS) continue;
            iu = &STATE(image_units[unit + e]);
            ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            /* r63: the atomics resolve straight into a buffer texture's storage (the
             * texture aliases the GL buffer) or through the writeback blit — either way
             * the GL buffer carries pending GPU writes the buffer domain cannot see
             * (its tracking only knows buffer bindings).  Mark it, or glMapBufferRange /
             * glGetBufferSubData read the stale pre-draw bytes
             * (shader_image_load_store.advanced-allStages-oneImage,
             * texture_buffer.texture_buffer_atomic_functions: "Buffer value is 0"). */
            {
                MGLTexExtInfo *bi = mgl_texext_info(ctx, ptr);
                Buffer *gb = (bi && bi->kind == MGL_TEXEXT_BUFFER && bi->buffer_name)
                             ? (Buffer *)searchHashTable(&ctx->state.buffer_table, bi->buffer_name) : NULL;

                if (gb) mgl_buffers_mark_gpu_use(ctx, gb, true);
            }
            texture = [self geomImageUnitViewRes: iu base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                      elemClass: sres->elem_class];
            buf = [self geomImageAtomicBuffer: unit + e view: texture];
            if (!buf) buf = [self geomImageAtomicDummy];     /* refused (oversized): sink */
            if (!buf) continue;
            if (getenv("MGL_GEOM_DEBUG") && stage == _FRAGMENT_SHADER)
                fprintf(stderr, "MGL geom: img-atom BIND stage %d unit %u slot %u buf=%p len=%zu "
                        "(view %lux%lux%lu s=%lu t=%lu)\n", stage, unit + e, mtl_index + e,
                        (__bridge void *)buf, buf.length,
                        (unsigned long)texture.width, (unsigned long)texture.height,
                        (unsigned long)texture.arrayLength, (unsigned long)texture.sampleCount,
                        (unsigned long)texture.textureType);
            if (stage != _FRAGMENT_SHADER)
                [_currentRenderEncoder setVertexBuffer: buf offset: 0 atIndex: mtl_index + e];
            else
                /* through the compat shadow, NOT a raw setFragmentBuffer: the render
                 * encoder is recreated around draws (state re-application) and a raw
                 * bind is then LOST (the shadow replays only what it knows) — a
                 * program with a default uniform block recreated the encoder after
                 * this bind and the MS atomic silently wrote nowhere. */
                [self compatBindFragmentBuffer: buf offset: 0 atIndex: mtl_index + e];
        }
    }
    return true;
}

/* the compute stage's version (bindTexturesToComputeEncoder, MGLRenderer.m) */
- (bool) geomBindImageAtomicBuffersCompute: (id <MTLComputeCommandEncoder>) encoder
{
    if (!encoder) return false;
    GLuint count = [self getProgramBindingCount: _COMPUTE_SHADER type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    for (GLuint i = 0; i < count; i++)
    {
        SpirvResource *sres = &STATE(program)->spirv_resources_list[_COMPUTE_SHADER][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        GLuint unit = sres->binding, mtl_index;
        GLuint elems = sres->array_size ? sres->array_size : 1;

        if (sres->msl_atomic == 0xffffffffu) continue;
        mtl_index = (sres->msl_atomic & ~MGL_GEOM_MS_SLOT);

        /* an ARRAY image: ONE side buffer covers every element (SPIRV-Cross declares a
         * single `<name>_atomic [[buffer(N)]]` for the whole array) */
        if (elems > 1)
        {
            id<MTLBuffer> buf = [self geomImageAtomicBufferArray: unit elems: elems];

            if (buf)
                [encoder setBuffer: buf offset: 0 atIndex: mtl_index];
            continue;
        }

        for (GLuint e = 0; e < elems; e++)
        {
            ImageUnit *iu;
            Texture *ptr;
            id<MTLTexture> texture;
            id<MTLBuffer> buf;

            if (unit + e >= TEXTURE_UNITS) continue;
            iu = &STATE(image_units[unit + e]);
            ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            {   /* buffer-texture atomics: pending GPU writes to the GL buffer (see the
                 * render-stage note above) */
                MGLTexExtInfo *bi = mgl_texext_info(ctx, ptr);
                Buffer *gb = (bi && bi->kind == MGL_TEXEXT_BUFFER && bi->buffer_name)
                             ? (Buffer *)searchHashTable(&ctx->state.buffer_table, bi->buffer_name) : NULL;

                if (gb) mgl_buffers_mark_gpu_use(ctx, gb, true);
            }
            texture = [self geomImageUnitViewRes: iu base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                      elemClass: sres->elem_class];
            buf = [self geomImageAtomicBuffer: unit + e view: texture];
            if (!buf) buf = [self geomImageAtomicDummy];     /* refused (oversized): sink */
            if (!buf) continue;
            [encoder setBuffer: buf offset: 0 atIndex: mtl_index + e];
        }
    }
    return true;
}

/* create (and seed) this dispatch's image-atomic sidecars BEFORE the dispatch's own
 * compute encoder is opened: the seed runs as its own compute pass on the same
 * command buffer, so it precedes the program's kernel without disturbing the
 * pipeline state the caller is about to install */
- (void) geomPrepareImageAtomicsCompute
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    GLuint count = [self getProgramBindingCount: _COMPUTE_SHADER type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
    int pending[TEXTURE_UNITS];
    int n_pending = 0;
    id<MTLComputeCommandEncoder> enc = NULL;

    if (!g || !count) return;

    for (GLuint i = 0; i < count; i++)
    {
        SpirvResource *sres = &STATE(program)->spirv_resources_list[_COMPUTE_SHADER]
                                                    [SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        int b = [self getProgramBinding: _COMPUTE_SHADER type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE index: i];
        GLuint elems = sres->array_size ? sres->array_size : 1;
        GLuint mtl = sres->msl_atomic;

        if (mtl == 0xffffffffu) continue;

        /* an ARRAY image: one combined side buffer; the seed path walks the elements
         * (geomSeedImageAtomic's elems>1 branch) */
        if (elems > 1 && b >= 0 && (GLuint)b < TEXTURE_UNITS)
        {
            if ([self geomImageAtomicBufferArray: (GLuint)b elems: elems] &&
                g->img_atom[b].needs_seed && n_pending < TEXTURE_UNITS)
                pending[n_pending++] = b;
            continue;
        }

        for (GLuint e = 0; e < elems && b >= 0; e++)
        {
            GLuint unit = (GLuint)(b + e);
            ImageUnit *iu;
            Texture *ptr;
            id<MTLTexture> view;

            if (unit >= TEXTURE_UNITS) break;
            iu = &STATE(image_units[unit]);
            ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            view = [self geomImageUnitViewRes: iu base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                    elemClass: sres->elem_class];
            if (![self geomImageAtomicBuffer: unit view: view]) continue;   /* creates + flags */
            if (g->img_atom[unit].needs_seed && n_pending < TEXTURE_UNITS)
                pending[n_pending++] = (int)unit;
        }
    }

    if (!n_pending) return;

    /* one compute pass seeds every unit that got a fresh sidecar this dispatch */
    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return;
    for (int i = 0; i < n_pending; i++)
        [self geomSeedImageAtomic: pending[i] encoder: enc];
    [enc endEncoding];
}

/* geom: render stages (vertex / fragment) with MULTISAMPLE image writes or atomics
 * need their sidecar seeded before the draw's render encoder opens — the seed is a
 * compute dispatch, which cannot run mid-encoder.  Called next to
 * geom3DImageShadowSync at draw time; the program-level gate keeps it free for
 * every program without a multisample image (the common case). */
- (void) geomPrepareImageAtomicsRender
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    int pending[TEXTURE_UNITS];
    int n_pending = 0;
    id<MTLComputeCommandEncoder> enc;

    if (!g || !ctx || !ctx->mtl_funcs.ext_texdata) return;

    /* cheap gate: only programs with an MS-flagged atomic slot can have work */
    bool any_ms = false;
    for (int s = 0; s < _MAX_SHADER_TYPES && !any_ms; s++)
    {
        Program *p = STATE(program);
        if (!p && STATE(program_pipeline))
        {
            MGLPipelineObj *po = MGL_PIPELINE_OBJ(STATE(program_pipeline));
            p = po->base.stage_programs[s];
        }
        if (!p) continue;
        GLuint count = [self getProgramBindingCount: s type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
        for (GLuint i = 0; i < count && !any_ms; i++)
            any_ms = (p->spirv_resources_list[s][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i].msl_atomic &
                      MGL_GEOM_MS_SLOT) != 0;
    }
    if (!any_ms) return;

    int stages[2] = { _VERTEX_SHADER, _FRAGMENT_SHADER };
    for (int si = 0; si < 2; si++)
    {
        int stage = stages[si];
        Program *p = STATE(program);
        if (!p && STATE(program_pipeline))
            p = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[stage];
        if (!p) continue;
        GLuint count = [self getProgramBindingCount: stage type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
        for (GLuint i = 0; i < count; i++)
        {
            SpirvResource *sres = &p->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
            if (sres->msl_atomic == 0xffffffffu || !(sres->msl_atomic & MGL_GEOM_MS_SLOT)) continue;
            GLuint mtl_index = (sres->msl_atomic & ~MGL_GEOM_MS_SLOT);
            if (sres->array_size) continue;                        /* arrays: sidecar path */

            int b = [self getProgramBinding: stage type: SPVC_RESOURCE_TYPE_STORAGE_IMAGE index: i];
            if (b < 0 || (GLuint)b >= TEXTURE_UNITS) continue;
            ImageUnit *iu = &STATE(image_units[b]);
            Texture *ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            id<MTLTexture> view = [self geomImageUnitViewRes: iu
                                                        base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                                   elemClass: sres->elem_class];
            if (![self geomImageAtomicBuffer: (GLuint)b view: view]) continue;
            if (g->img_atom[b].needs_seed && n_pending < TEXTURE_UNITS)
                pending[n_pending++] = b;
        }
    }
    if (!n_pending) return;

    [self endRenderEncoding];
    if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        [self newCommandBuffer];
    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return;
    enc.label = @"mgl-ms-image-seed";
    for (int i = 0; i < n_pending; i++)
        [self geomSeedImageAtomic: pending[i] encoder: enc];
    [enc endEncoding];
}

/* geom: the tessellation emulation's VS / TCS (and TES-capture) kernels run as
 * COMPUTE carrying their OWN stage's resource list (tessBindStageTextures,
 * MGLRenderer+Tessellation.m) — the two helpers above only walk the current
 * program's _COMPUTE_SHADER stage.  These (program, stage)-parameterized twins give
 * those kernels the same sidecar binding and seeding; without them the kernels'
 * imageAtomic* write an unbound `<name>_atomic` buffer argument and never land
 * (shader_image_load_store.advanced-allStages-oneImage: every VS/TCS add lost). */
- (bool) geomBindStageImageAtomicsCompute: (Program *) program
                                   encoder: (id <MTLComputeCommandEncoder>) encoder
                                     stage: (int) stage
{
    if (!encoder || !program) return false;
    GLuint count = program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count;
    for (GLuint i = 0; i < count; i++)
    {
        SpirvResource *sres = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        GLuint unit = sres->binding, mtl_index = sres->msl_atomic;
        GLuint elems = sres->array_size ? sres->array_size : 1;

        if (mtl_index == 0xffffffffu) continue;    /* no atomic argument in the MSL */

        if (elems > 1)
        {
            id<MTLBuffer> buf = [self geomImageAtomicBufferArray: unit elems: elems];
            if (buf) [encoder setBuffer: buf offset: 0 atIndex: mtl_index];
            continue;
        }

        for (GLuint e = 0; e < elems; e++)
        {
            ImageUnit *iu;
            Texture *ptr;
            id<MTLTexture> texture;
            id<MTLBuffer> buf;

            if (unit + e >= TEXTURE_UNITS) continue;
            iu = &STATE(image_units[unit + e]);
            ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            {   /* buffer-texture atomics mark the GL buffer GPU-written (as above) */
                MGLTexExtInfo *bi = mgl_texext_info(ctx, ptr);
                Buffer *gb = (bi && bi->kind == MGL_TEXEXT_BUFFER && bi->buffer_name)
                             ? (Buffer *)searchHashTable(&ctx->state.buffer_table, bi->buffer_name) : NULL;

                if (gb) mgl_buffers_mark_gpu_use(ctx, gb, true);
            }
            texture = [self geomImageUnitViewRes: iu base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                      elemClass: sres->elem_class];
            buf = [self geomImageAtomicBuffer: unit + e view: texture];
            if (!buf) buf = [self geomImageAtomicDummy];     /* refused (oversized): sink */
            if (!buf) continue;
            [encoder setBuffer: buf offset: 0 atIndex: mtl_index + e];
        }
    }
    return true;
}

/* create (and seed) those sidecars before the tess kernels' compute encoder opens —
 * same shape as geomPrepareImageAtomicsCompute, but for an explicit (program, stage) */
- (void) geomPrepareStageImageAtomicsCompute: (Program *) program stage: (int) stage
{
    struct MGLExtState_geom *g = (struct MGLExtState_geom *)(ctx ? ctx->state.ext_geom : NULL);
    int pending[TEXTURE_UNITS];
    int n_pending = 0;
    id<MTLComputeCommandEncoder> enc = NULL;
    GLuint count;

    if (!g || !program) return;
    count = program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].count;

    for (GLuint i = 0; i < count; i++)
    {
        SpirvResource *sres = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE].list[i];
        int b = (int)sres->binding;
        GLuint elems = sres->array_size ? sres->array_size : 1;
        GLuint mtl = sres->msl_atomic;

        if (mtl == 0xffffffffu) continue;

        if (elems > 1 && b >= 0 && (GLuint)b < TEXTURE_UNITS)
        {
            if ([self geomImageAtomicBufferArray: (GLuint)b elems: elems] &&
                g->img_atom[b].needs_seed && n_pending < TEXTURE_UNITS)
                pending[n_pending++] = b;
            continue;
        }

        for (GLuint e = 0; e < elems && b >= 0; e++)
        {
            GLuint unit = (GLuint)(b + e);
            ImageUnit *iu;
            Texture *ptr;
            id<MTLTexture> view;

            if (unit >= TEXTURE_UNITS) break;
            iu = &STATE(image_units[unit]);
            ptr = iu->tex;
            if (!ptr || !ptr->mtl_data) continue;
            view = [self geomImageUnitViewRes: iu base: (__bridge id<MTLTexture>)(ptr->mtl_data)
                                    elemClass: sres->elem_class];
            if (![self geomImageAtomicBuffer: unit view: view]) continue;   /* creates + flags */
            if (g->img_atom[unit].needs_seed && n_pending < TEXTURE_UNITS)
                pending[n_pending++] = (int)unit;
        }
    }

    if (!n_pending) return;

    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return;
    for (int i = 0; i < n_pending; i++)
        [self geomSeedImageAtomic: pending[i] encoder: enc];
    [enc endEncoding];
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
/* geom r60: SPIRV-Cross gives a combined image sampler's SAMPLER its own Metal
 * argument index.  In a compute kernel that also has image textures the two
 * drift apart (e.g. `dst [[texture(0)]]` pushes `s` to [[texture(1)]] while its
 * `sSmplr` stays [[sampler(0)]]) — the compute bind path used the TEXTURE index
 * for the sampler, so every sample() hit an unbound sampler slot and read 0
 * (KHR-GL46.stencil_texturing.functional's compute pass, texture_buffer cases).
 * Parse the sampler's own [[sampler(N)]] from the kernel MSL (cached per
 * program); fall back to the texture index when it cannot be found. */
- (NSUInteger) geomComputeSamplerSlotForImage: (const char *)img_name textureSlot: (NSUInteger)tslot
{
    static struct { const void *program; const void *msl; char name[96]; NSUInteger slot; } cache[8];
    if (!ctx || !ctx->state.program) return tslot;
    const char *msl = ctx->state.program->spirv[_COMPUTE_SHADER].msl_str;
    if (!msl || !img_name || !*img_name) return tslot;

    /* the MSL pointer is part of the key: programs are freed and their addresses
     * recycled constantly (CTS links thousands), and a different program's slot
     * for the same resource name would silently mis-bind the sampler */
    int ci = -1;
    for (int i = 0; i < 8; i++)
        if (cache[i].program == (const void *)ctx->state.program && cache[i].msl == (const void *)msl &&
            strncmp(cache[i].name, img_name, sizeof cache[i].name) == 0)
            return cache[i].slot;
    for (int i = 0; i < 8; i++) if (!cache[i].program) { ci = i; break; }
    if (ci < 0) { ci = 0; memset(cache, 0, sizeof cache); }

    /* the pair is emitted adjacently: ... texture2d<float> s [[texture(1)]],
     * sampler sSmplr [[sampler(0)]] ... */
    char pat[128];
    snprintf(pat, sizeof pat, "%sSmplr [[sampler(", img_name);
    const char *hit = strstr(msl, pat);
    NSUInteger slot = tslot;
    if (hit)
    {
        int v = atoi(hit + strlen(pat));
        if (v >= 0) slot = (NSUInteger)v;
    }
    cache[ci].program = (const void *)ctx->state.program;
    cache[ci].msl = (const void *)msl;
    snprintf(cache[ci].name, sizeof cache[ci].name, "%s", img_name);
    cache[ci].slot = slot;
    return slot;
}

- (void) geomBindRenderStageTextures: (id<MTLRenderCommandEncoder>) enc stage: (int) stage
                                geomProgram: (MGLGeomProgram *) gp
{
    struct { int spvc_type; bool image; } types[] = {
        { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, false },
        { SPVC_RESOURCE_TYPE_STORAGE_IMAGE, true },
        { 0, false }
    };

    if (!STATE(program)) return;
    if (getenv("MGL_ZZVCAP")) fprintf(stderr, "ZZVCAP enter stage=%d\n", stage);
    for (int t = 0; types[t].spvc_type; t++)
    {
        GLuint n = [self getProgramBindingCount: stage type: types[t].spvc_type];
        if (getenv("MGL_ZZVCAP"))
            fprintf(stderr, "ZZVCAP bind-stage=%d type=%d n=%u\n", stage, types[t].spvc_type, n);

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
                /* GL semantics: image-ARRAY elements are independent uniforms
                 * (mgli_sampler_elem_unit; see MGLRenderer.m) */
                int unit = (types[t].image) ? (int)mgl_res_elem_unit(sres, (GLuint)e)
                                            : spirv_binding + e;

                if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;
                Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                              : mgl_texture_for_sampler(ctx, unit, sres->target);
                if (tex && !tex->mtl_data)
                {
                    /* a texture VIEW (texext-owned: view / buffer / MS) materializes
                     * its Metal texture lazily; the render-encoder bind path realizes
                     * it on first bind, this emulation-kernel path must do the same
                     * or the kernel samples an unbound slot
                     * (KHR-GL46.texture_view.view_sampling: geometry stage) */
                    [self bindMTLTexture: tex];
                }
                if (!tex || !tex->mtl_data) continue;
                if (getenv("MGL_ZZVCAP"))
                    fprintf(stderr, "ZZVCAP vs-capture bind unit=%d tex=%u tgt=%x incomplete=%d mtl=%p\n",
                            unit, tex->name, tex->target,
                            (int)mgl_texture_incomplete_for_sampling(tex), (void *)tex->mtl_data);
                id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
                if (!types[t].image)
                {
                    /* GL 8.17: an INCOMPLETE texture samples (0,0,0,1) — the emulation
                     * kernels read() the bound texture directly, so they must see the
                     * shared 1x1 fallback, not the real mip chain (bindTextures'
                     * normal path does this for the non-emulated stages;
                     * KHR-GL46.texture_gather.incomplete-texture[-last-comp]) */
                    if (mgl_texture_incomplete_for_sampling(tex))
                    {
                        id<MTLTexture> ftex = [self computeFallbackTexture];
                        id<MTLSamplerState> fsamp = [self computeFallbackSampler];
                        if (ftex && fsamp)
                        {
                            [enc setVertexSamplerState: fsamp atIndex: mtl_index + e];
                            [enc setVertexTexture: ftex atIndex: mtl_index + e];
                            continue;
                        }
                    }
                    id<MTLSamplerState> sampler = [self texparamSamplerForUnit: (GLuint)unit texture: tex];
                    texture = [self texparamTextureForTex: tex base: texture];
                    if (sampler) [enc setVertexSamplerState: sampler atIndex: mtl_index + e];
                }
                else
                {
                    /* an image unit binds one mip level / layer (glBindImageTexture),
                     * reinterpreted as the unit's format (geomImageUnitViewRes) */
                    texture = [self geomImageUnitViewRes: &STATE(image_units[unit]) base: texture
                                                 elemClass: sres->elem_class];
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
        /* geometry-stage atomic counters are bound below, at the kernel's injected
         * argument slots: the map's own atomic-counter entries expand a declaration
         * (atomic_uint acs[N]) into one entry per ELEMENT, while the kernel has ONE
         * pointer argument per declaration (mgl_msl_inject_atomic_counters), and the
         * expansion can run past the user slot range into the emulation's own pinned
         * buffers — so the group is never bound from the list */
        if (gp->n_ac_vars > 0 && i >= gp->ac_base) break;
        if (i >= MGL_GEOM_KERNEL_USER_SLOTS) break;
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

    /* geometry-stage atomic counters: one kernel argument per GLSL declaration
     * (mgl_msl_inject_atomic_counters, geometry.c), bound to the GL atomic-counter
     * buffer of its layout(binding=) point at the ordinal the map gives the group */
    for (int i = 0; i < gp->n_ac_vars; i++)
    {
        GLuint unit = gp->ac_binding[i];
        BufferBaseTarget *t = (unit < MAX_BINDABLE_BUFFERS) ?
            &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[unit] : NULL;
        Buffer *b = t ? t->buf : NULL;
        GLintptr off = t ? t->offset : 0;

        if (!b)
        {
            b = [self nullShaderBuffer];
            off = 0;
        }
        else
        {
            if (off < 0 || (GLsizeiptr)off > b->size) off = 0;
            if (!b->data.mtl_data && b->data.dirty_bits)
                if (![self updateDirtyBuffer: b]) return false;
            if (!b->data.mtl_data) [self bindMTLBuffer: b];
            /* the kernel writes the counters: an inline setBytes binding would
             * silently drop them, so the buffer must be a real MTLBuffer */
            if (!b->data.mtl_data) return false;
            mgl_buffers_mark_gpu_use(ctx, b, true);
        }
        if (!b->data.mtl_data) continue;
        [enc setBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                offset: (NSUInteger)off
               atIndex: (NSUInteger)gp->ac_msl[i]];
        if (getenv("MGL_SSBO_DEBUG"))
            NSLog(@"MGL geom: gs kernel atomic counter %s: unit=%u at msl %d (off=%ld)",
                  gp->ac_vars[i].name ? gp->ac_vars[i].name : "?", unit, gp->ac_msl[i],
                  (long)off);
    }

    /* the kernel variant's .length() sizes: geom_spirv_to_msl pins its
     * spvBufferSizeConstants at MGL_GEOM_KERNEL_SIZE_BUFFER, the one user-range slot the
     * pin span check keeps free when the variant reads .length() on an SSBO runtime
     * array (OpArrayLength).  The kernel's SSBO arguments sit at their pin flats, which
     * this walk itself binds at the map index — so the sizes array is indexed by the map
     * index too (geomFillSSBOSizes indexes by the RASTER msl_binding, which the pins
     * renumber) */
    if (gp->kernel_msl && strstr(gp->kernel_msl, "spvBufferSizeConstants"))
    {
        uint32_t sizes[MAX_MAPPED_BUFFERS];
        GLuint n = 0;
        /* the entries past the kernel's user arguments (the atomic-counter group and
         * whatever the map's expansion added beyond it) are no SSBOs: their sizes would
         * only grow the inline binding */
        GLuint cap = (gp->n_ac_vars > 0 && gp->ac_base < MGL_GEOM_KERNEL_USER_SLOTS) ?
                     (GLuint)gp->ac_base : MGL_GEOM_KERNEL_USER_SLOTS;

        memset(sizes, 0, sizeof sizes);
        for (GLuint i = 0; i < list.count && i < MAX_MAPPED_BUFFERS && i < cap; i++)
        {
            BufferMap *m = &list.buffers[i];
            GLsizeiptr sz;

            if (!m->buf) continue;
            sz = m->size > 0 ? m->size : (m->buf->size - (m->offset > 0 ? m->offset : 0));
            sizes[i] = sz > 0 ? (uint32_t)sz : 0;
            n = i + 1;
        }
        if (n)
            [enc setBytes: sizes length: (NSUInteger)n * sizeof *sizes
                 atIndex: MGL_GEOM_KERNEL_SIZE_BUFFER];
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
                if (getenv("MGL_ZZTRACE"))
                {
                    FILE *f = fopen("/tmp/geom_zztrace.txt", "a");
                    if (f)
                    {
                        fprintf(f, "gs bind prog=%u name='%s' unit=%d slot=%ld fallback_msl=%d table:",
                                ctx->state.program ? ctx->state.program->name : 0,
                                sres->name ? sres->name : "?", spirv_binding,
                                (long)mtl_index, (int)sres->msl_binding);
                        for (int q = 0; q < gp->n_tex_slots_gs; q++)
                            fprintf(f, " ['%s'->%d]", gp->tex_slots_gs[q].name, gp->tex_slots_gs[q].msl);
                        fprintf(f, "\n");
                        fclose(f);
                    }
                }
                if (mtl_index == (NSUInteger)-1) mtl_index = (NSUInteger)sres->msl_binding;
                int n = 1;

                for (int a = 0; a < gp->n_tex_arrays; a++)
                    if (!strcmp(gp->tex_arrays[a].name, sres->name ? sres->name : ""))
                    { n = gp->tex_arrays[a].n; break; }

                if (spirv_binding < 0) continue;

                for (int e = 0; e < n; e++)
                {
                    /* GL semantics: image-ARRAY elements are independent uniforms
                     * (mgli_sampler_elem_unit; see MGLRenderer.m) */
                    int unit = (types[t].image) ? (int)mgl_res_elem_unit(sres, (GLuint)e)
                                                : spirv_binding + e;
                    if (unit < 0 || (GLuint)unit >= TEXTURE_UNITS) continue;

                    Texture *tex = types[t].image ? STATE(image_units[unit].tex)
                                                  : mgl_texture_for_sampler(ctx, unit, sres->target);
                    if (!tex) continue;
                    if (![self bindMTLTexture: tex]) continue;
                    id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
                    if (!texture) continue;
                    if (!types[t].image)
                    {
                        /* GL 8.17: incomplete texture -> shared 1x1 fallback (see
                         * geomBindRenderStageTextures) */
                        if (mgl_texture_incomplete_for_sampling(tex))
                        {
                            id<MTLTexture> ftex = [self computeFallbackTexture];
                            id<MTLSamplerState> fsamp = [self computeFallbackSampler];
                            if (ftex && fsamp)
                            {
                                [enc setSamplerState: fsamp atIndex: mtl_index + e];
                                [enc setTexture: ftex atIndex: mtl_index + e];
                                continue;
                            }
                        }
                        id<MTLSamplerState> sampler = [self texparamSamplerForUnit: (GLuint)unit texture: tex];
                        texture = [self texparamTextureForTex: tex base: texture];
                        if (sampler) [enc setSamplerState: sampler atIndex: mtl_index + e];
                    }
                    else
                    {
                        /* an image unit binds one mip level / layer (glBindImageTexture),
                         * reinterpreted as the unit's format (geomImageUnitViewRes) */
                        texture = [self geomImageUnitViewRes: &STATE(image_units[unit]) base: texture
                                                     elemClass: sres->elem_class];
                    }
                    [enc setTexture: texture atIndex: mtl_index + e];
                    /* image atomics: the kernel's `<name>_atomic` sidecar argument (see
                     * mgl_msl_fix_image_atomics at the kernel build) */
                    if (types[t].image && gp->n_atom_slots_gs > 0)
                    {
                        int aslot = geomTexSlot(gp->atom_slots_gs, gp->n_atom_slots_gs,
                                                sres->name ? sres->name : "");
                        if (aslot >= 0)
                        {
                            id<MTLBuffer> abuf = [self geomImageAtomicBuffer: (GLuint)unit
                                                                        view: texture];
                            if (abuf)
                                [enc setBuffer: abuf offset: 0
                                     atIndex: (NSUInteger)(aslot + e)];
                        }
                    }
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

/* the size a (re)created slot should have: at least need*2, and — when the slot ran out
 * of room while the GPU was still behind (the reclaim found it full) — double its
 * previous length, so every wrap buys twice the CPU run-ahead before the next
 * reclaim-wait.  Bounded by the per-slot cap, the global growth budget (gPoolBudget,
 * summed over every pool's slots: growth beyond need*2 stops once the pools already
 * hold that much) and "one draw always gets its memory" for draws over the cap.
 * Returns 0 when no growth is required or affordable — the caller keeps the current
 * buffer (a declined doubling must NOT fall through to need*2, which would SHRINK
 * the slot and re-allocate it on every wrap; that allocation thrash cost glmark2
 * gl41-geometry ~30% on a clean machine).  Without the doubling the slots stayed at
 * ~2 draws each and every third draw of a geometry-shader scene blocked in
 * waitUntilCompleted. */
static size_t geomPoolTotalBytes;   /* live slot bytes over all pools (growth accounting) */

static size_t geomSlotSize(MGLGeomPool *pool, unsigned c, size_t need, bool ran_out)
{
    size_t prev = pool->buf[c] ? pool->buf[c].length : 0;
    size_t sz = 4096, want = need * 2;

    if (ran_out && prev)
    {
        size_t dbl = prev * 2;
        if (dbl <= gPoolCap && dbl > want && geomPoolTotalBytes + prev <= gPoolBudget)
            want = dbl;                       /* doubling buys run-ahead */
        else if (prev >= need)
            return 0;                         /* declined (cap/budget): the slot still fits a draw */
    }
    if (want > gPoolCap && need <= gPoolCap) want = gPoolCap;   /* soft cap */
    while (sz < want) sz *= 2;
    if (sz > gPoolCap && need > gPoolCap) sz = need;            /* one draw still gets its memory */
    return sz > prev ? sz : prev;                               /* never shrink a live slot */
}

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
        bool ran_out = room && pool->used[c] + need > room;
        if (room < need || ran_out)
        {
            size_t sz = geomSlotSize(pool, c, need, ran_out);

            if (sz)
            {
                if (gEnv.draw)
                    fprintf(stderr, "geom: trace pool %p slot %u grow need=%zu -> %zu\n",
                            (void *)pool, c, need, sz);
                id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
                if (!grown) continue;               /* try the next slot; the old buffer stays usable */
                geomPoolTotalBytes = geomPoolTotalBytes + sz - room;
                pool->buf[c] = grown;
                room = sz;
            }
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
            bool ran_out = room && pool->used[c] + need > room;
            if (room < need || ran_out)
            {
                size_t sz = geomSlotSize(pool, c, need, ran_out);
                if (sz)
                {
                    id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
                    if (!grown) continue;
                    geomPoolTotalBytes = geomPoolTotalBytes + sz - room;
                    pool->buf[c] = grown;
                    room = sz;
                }
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
         * current slot (the old buffer keeps the pending ranges alive); never smaller
         * than the buffer it replaces (see geomSlotSize) */
        {
            unsigned c = pool->cur % gPoolN;
            size_t prev = pool->buf[c] ? pool->buf[c].length : 0;
            size_t sz = 4096, want = need * 2;

            while (sz < want) sz *= 2;
            if (sz < prev) sz = prev;
            if (gEnv.draw)
                fprintf(stderr, "geom: trace pool %p slot %u forced grow need=%zu -> %zu\n",
                        (void *)pool, c, need, sz);
            id<MTLBuffer> grown = [device newBufferWithLength: sz options: MTLResourceStorageModeShared];
            if (!grown) return nil;
            geomPoolTotalBytes = geomPoolTotalBytes + sz - prev;
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

    /* compat CS: this pass drives the command buffer directly (prep/capture/
     * raster encoders of its own); the encoder worker must be idle for it */
    [self compatCSDrain];

    MGLGeomProgram *gp = d->gp;    Program *program = d->program;
    const bool gpu_assembly = (d->prims == NULL);   /* the CPU path supplies the lists */
    /* affine direct draw: the geometry kernel derives the primitive windows arithmetically;
     * no prep stage, no capture index list, exact dispatch counts - all CPU-known */
    const bool implicit = d->implicit_stride != 0 && !(geomEnvRead(), gEnv.no_implicit);
    /* primitive count of an affine direct draw - the same arithmetic the kernel applies to
     * the windows: strip codes 1..3 advance one vertex per window (odd triangle-strip
     * windows only permute their vertices), disjoint codes (size + 3) drop the tail */
    const uint64_t implicit_prims = (d->implicit_stride >= 1 && d->implicit_stride <= 3)
        ? (d->count >= (GLsizei)gp->in_verts ? (uint64_t)d->count - gp->in_verts + 1 : 0)
        : (uint64_t)d->count / (gp->in_verts ? gp->in_verts : 1);
    if (implicit)
    {
        d->prims_count = (uint32_t)implicit_prims;
        if (!d->prims_count) d->prims_count = 1;      /* keeps the dispatch grid nonzero */
    }
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
         * entries make one primitive.  An implicit (affine) draw knows the exact count on
         * the CPU, so the pools and the dispatch grid are sized tightly.  An INDIRECT draw
         * whose command the CPU has read (pipeline-statistics query active, or the buffer
         * still carries its CPU copy) also sizes exactly — including the REAL instance
         * count, which the worst-case path could only assume to be 1 and which truncated
         * the output budget of every primcount > 1 indirect draw. */
        uint64_t bound_threads = gpu_assembly
                                 ? ((from_tess
                                     ? (uint64_t)(d->tess_idx_cap ? d->tess_idx_cap : 1)
                                       / (gp->in_verts ? gp->in_verts : 1)
                                     : (implicit
                                        ? implicit_prims
                                        : (d->known_prims
                                           ? (uint64_t)d->known_prims
                                           : mgl_geom_worst_prims(d->mode, bound))))
                                    * (uint64_t)(d->known_instances
                                                     ? d->known_instances
                                                     : (d->instances > 0 ? d->instances : 1))
                                    * gp->invocations)
                                 : threads;
        outv_cap = (size_t)bound_threads * (gp->max_vertices ? gp->max_vertices : 1);
        if (outv_cap > max_out) outv_cap = max_out;
        if (!outv_cap) outv_cap = 1;
        idx_cap = outv_cap * 3;
        threads = bound_threads;
        if (!threads) threads = 1;    /* empty draw: keep every dispatch grid nonzero */
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
        if (implicit)
        {
            /* affine direct draw: no capture index list and no primitive slots - the
             * capture pass draws the vertex range directly and the geometry kernel derives
             * the windows from hdr[19..21] */
            cap_idx = 0;
            cap_prim = 0;
        }
        else
        {
        /* the prep kernel writes the lists: the capture list holds one entry per vertex
         * of every restart-free run (<= the drawn vertex count), and the primitive slots
         * sit behind it at HDR + cap_idx - the geometry kernel binds that same offset */
        cap_idx = (size_t)(d->count_bound ? d->count_bound : (d->count > 0 ? d->count : 1));
        cap_prim = (size_t)mgl_geom_worst_prims(d->mode, bound);
        if (!cap_idx) cap_idx = 1;
        if (!cap_prim) cap_prim = 1;
        }
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
    {
        /* GPU (prep-kernel) assembly.  The prep kernel derives vmin/vmax and the
         * capture header on the GPU; until r63 this sized the pool by the INDEX
         * count ("the capture pass draws out_idx indices per instance") — but the
         * capture pass draws the prep's capture list as an INDEX BUFFER, so the
         * capture variant's gl_VertexID is the fetched index VALUE and its record
         * lands at (index - vmin)*in_words: the slice must cover the drawn VERTEX
         * RANGE, not the index count.  A sparse indexed draw — test_geom_xfbperf's
         * 30k indices over a 100k-vertex buffer — wrote 3x past the slice: silently
         * into the rest of the pool most of the time, into unmapped pages whenever
         * the slice landed at the pool buffer's end (the intermittent GPU page
         * fault; it needed several draws in flight for the pool to reach that
         * state).  geom_route_gpu scans the element range for direct draws and
         * supplies d->vstart/d->vcount; indirect draws leave vcount 0 and fall back
         * to the index-count bound, with the capture variant's capacity clamp
         * (mgl_hdr[4], written below) keeping it inside the slice. */
        uint64_t vb = bound;
        uint64_t ib = d->instances > 0 ? (uint64_t)d->instances : 1;

        if (d->indexed && d->vcount && (uint64_t)d->vcount > vb) vb = d->vcount;
        if (d->known_vcount && (uint64_t)d->known_vcount < vb) vb = d->known_vcount;
        if (d->known_instances) ib = d->known_instances;
        vsout_bytes = (8 + (size_t)vb * (size_t)ib * gp->in_words) * sizeof(uint32_t);
        if (getenv("MGL_GEOM_ASSERT") && d->indexed && d->vcount)
        {
            /* CPU-side proof every capture write stays in the slice: the largest
             * record word offset is (vcount-1 + (instances-1)*vcount)*in_words plus
             * in_words words, all under the 8-word header. */
            uint64_t need = (8 + (size_t)vb * (size_t)ib * (uint64_t)gp->in_words) * 4;
            assert(need <= vsout_bytes);
            assert((size_t)d->vcount <= vb);   /* the range fits the per-instance bound */
        }
    }

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
    gIdx = geomBuffer(_device, gIdx, ((size_t)idx_cap * 2 + (size_t)threads * 2 +
                                      /* the order pass' per-chunk scan totals (pscan/pbase) */
                                      (size_t)((threads + 1023) / 1024) + 64) * sizeof(uint32_t), &gIdxBytes);
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
    /* r63: the capture variant's record-capacity clamp reads VSOUT header word 4
     * (mgl_hdr[4], in words, after the 8-word header).  Nothing else writes word 4
     * (the prep/tess kernels fill 0..3), so writing it here — after this draw's
     * sizing, before any dispatch — keeps it exact for every draw. */
    if (gVsout && gVsout.contents)
        ((uint32_t *)gVsout.contents)[gVsoutOff / 4 + 4] = (uint32_t)(vsout_bytes / 4 - 8);
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

        if (implicit)
        {
            /* affine direct draw: the CPU supplies everything the prep kernel would have
             * computed - the primitive count, the record base (the drawn range starts at
             * `first`) and the window arithmetic in hdr[19..21] the kernel uses to derive
             * each primitive's vertex ids without the primitive slots */
            hdr[0]  = (uint32_t)implicit_prims;
            hdr[6]  = (uint32_t)(d->count > 0 ? d->count : 1);   /* per-instance records */
            hdr[8]  = d->first;
            hdr[12] = d->first;                                  /* record base: (vid - first) */
            hdr[19] = d->first;                                  /* window base vertex id */
            hdr[20] = d->implicit_stride;
            hdr[21] = 1;
        }
        else if (gpu_assembly)
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
     * fills in the vertex range.  An implicit draw is a direct non-indexed draw: the range
     * is CPU-known (first..first+count), so the variant gets it here as well */
    {
        uint32_t *ch = (uint32_t *)gVsout.contents + gVsoutOff / 4;
        const bool cap_direct = !gpu_assembly || implicit;
        ch[0] = cap_direct && gp->vid_absolute ? (uint32_t)d->first : 0;
        ch[1] = (uint32_t)(cap_direct ? (d->count > 0 ? d->count : 0) : 0);
        ch[2] = gp->iid_absolute ? (uint32_t)d->baseinstance : 0;
        ch[3] = (uint32_t)gp->in_words;
    }

    /* the whole emulation is encoded into one command buffer: close whatever encoder is open
     * and make sure the buffer can still take encoders (the ring wait above may have
     * committed it) */
    [self endRenderEncoding];
    /* realize the geometry stage's textures NOW: no encoder is open yet, and the tess
     * import encoder opened below stays open across the whole geometry pass — a deferred
     * texture upload inside it would open a blit encoder Metal forbids (seen as
     * "A command encoder is already encoding" on shader_image_load_store *GeometryStages) */
    [self geomRealizeKernelStageTextures: program geomProgram: gp];
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
    geomTimeCommandBuffer(_currentCommandBuffer);
    if (gEnv.gputime) gGpuTimeDraws++;   /* per-draw GPU accounting denominator */

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
        uint32_t cfg[10];
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
        cfg[8] = d->tess_emission;               /* records are in emission order:
                                                 * primitive slot t IS record t */
        cfg[9] = 0;
        [enc setBytes: cfg length: sizeof cfg atIndex: 9];
        [enc dispatchThreadgroups: MTLSizeMake((nt + wg - 1) / wg, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
        [enc popDebugGroup];
        kenc = enc;                       /* still open: the geometry kernel joins it */
    }
    else if (gpu_assembly && !implicit)   /* affine direct draws need no assembly at all */
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
    if (!from_tess && !(geomEnvRead(), gEnv.skip_capture))      /* the tess producer already wrote the records */
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

        /* the capture PSO only depends on (capture function, vertex layout): cache it on
         * the geom program instead of rebuilding (newRenderPipelineStateWithDescriptor)
         * on every draw */
        uint64_t vdkey = (uint64_t)(uintptr_t)fn;
        {
            uint64_t h = 1469598103934665603ULL;
            for (NSUInteger i = 0; i < 31; i++)
            {
                MTLVertexAttributeDescriptor *a = vd.attributes[i];
                if (!a) continue;
                h = (h ^ (uint64_t)a.format) * 1099511628211ULL;
                h = (h ^ (uint64_t)a.bufferIndex) * 1099511628211ULL;
                h = (h ^ (uint64_t)a.offset) * 1099511628211ULL;
            }
            for (NSUInteger i = 0; i < 31; i++)
            {
                MTLVertexBufferLayoutDescriptor *l = vd.layouts[i];
                if (!l) continue;
                h = (h ^ (uint64_t)l.stride) * 1099511628211ULL;
                h = (h ^ (uint64_t)l.stepFunction) * 1099511628211ULL;
                h = (h ^ (uint64_t)l.stepRate) * 1099511628211ULL;
            }
            vdkey ^= h;
        }
        id<MTLRenderPipelineState> pso;
        if (gp->capture_pso && gp->capture_pso_key == vdkey)
        {
            pso = (__bridge id<MTLRenderPipelineState>)gp->capture_pso;
        }
        else
        {
            NSError *error = nil;
            pso = [_device newRenderPipelineStateWithDescriptor: pd error: &error];
            if (!pso)
            {
                NSLog(@"MGL geom: capture pipeline failed: %@", error);
                if (saved_vs) saved_vs->mtl_data.function = saved_vs_fn;
                _currentRenderEncoder = saved;
                return false;
            }
            if (gp->capture_pso) ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_pso);
            gp->capture_pso = (void *)CFBridgingRetain(pso);
            gp->capture_pso_key = vdkey;
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
        /* this pass owns a private render encoder: no bind-shadow entry may survive
         * into it (the driver recycles encoder addresses, and a recycled address
         * would make compatBindVertexBuffer skip the bind) */
        [self compatBindInvalidate];
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
                /* vertex-stage atomic counters are bound below, at the capture variant's
                 * injected argument slots: the map's own atomic-counter entries expand a
                 * declaration into one entry per ELEMENT, while the variant has ONE
                 * pointer argument per declaration, and the expansion can reach into the
                 * variant's own user argument slots (the map does not count sampler
                 * arrays, which the variant pins as texture arguments) */
                if (gp->n_vs_ac_vars > 0 && i >= gp->vs_ac_map_base) break;
                if (i >= MGL_GEOM_KERNEL_USER_SLOTS) break;
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

            /* vertex-stage atomic counters: one capture-variant argument per GLSL
             * declaration (mgl_msl_inject_atomic_counters, geometry.c), bound to the GL
             * atomic-counter buffer of its layout(binding=) point */
            for (int i = 0; i < gp->n_vs_ac_vars && gp->vs_ac_msl[i] >= 0; i++)
            {
                GLuint unit = gp->vs_ac_binding[i];
                BufferBaseTarget *t = (unit < MAX_BINDABLE_BUFFERS) ?
                    &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[unit] : NULL;
                Buffer *b = t ? t->buf : NULL;
                GLintptr off = t ? t->offset : 0;

                if (!b)
                {
                    b = [self nullShaderBuffer];
                    off = 0;
                }
                else
                {
                    if (off < 0 || (GLsizeiptr)off > b->size) off = 0;
                    if (!b->data.mtl_data && b->data.dirty_bits)
                        if (![self updateDirtyBuffer: b]) continue;
                    if (!b->data.mtl_data) [self bindMTLBuffer: b];
                    if (!b->data.mtl_data) continue;
                    mgl_buffers_mark_gpu_use(ctx, b, true);
                }
                if (!b->data.mtl_data) continue;
                [enc setVertexBuffer: (__bridge id<MTLBuffer>)b->data.mtl_data
                              offset: (NSUInteger)off
                             atIndex: (NSUInteger)gp->vs_ac_msl[i]];
            }
        }

        /* the vertex stage's textures sample in the capture variant just like in the
         * raster pass (GL allows the VS to sample; the CTS limits tests use 16 units) */
        [self geomBindRenderStageTextures: enc stage: _VERTEX_SHADER geomProgram: gp];
        ctx->state.program = saved_program;
        [enc setVertexBuffer: gVsout offset: gVsoutOff atIndex: MGL_GEOM_MSL_VSOUT];

        /* the capture variant's own .length() sizes: its spvBufferSizeConstants is
         * compiled at MGL_GEOM_KERNEL_SIZE_BUFFER (geom_spirv_to_msl moved it off the
         * VSOUT slot), and geomBindSSBOSizes skips the vertex table for a geometry
         * program (the RASTER pass's vertex function is the pass-through, which reads no
         * SSBO) — bind them for the capture pass itself */
        if (gp->capture_msl && strstr(gp->capture_msl, "spvBufferSizeConstants"))
        {
            uint32_t sizes[MAX_MAPPED_BUFFERS];
            GLuint n = 0;

            [self geomFillSSBOSizes: sizes list: &ctx->state.vertex_buffer_map_list count: &n];
            if (n)
                [enc setVertexBytes: sizes length: (NSUInteger)n * sizeof *sizes
                            atIndex: MGL_GEOM_KERNEL_SIZE_BUFFER];
        }

        if (gpu_assembly && !implicit)
        {
            /* the prep kernel wrote the draw arguments (index count, instances, bases) */
            [enc drawIndexedPrimitives: MTLPrimitiveTypePoint
                             indexType: MTLIndexTypeUInt32
                           indexBuffer: gPrim
                     indexBufferOffset: gPrimOff + MGL_GEOM_HDR_WORDS * sizeof(uint32_t)
                        indirectBuffer: gPrep
                  indirectBufferOffset: gPrepOff + (32 + 7) * sizeof(uint32_t)];
        }
        else if (d->indexed && !implicit)
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
        /* the GL encoder returns: it must not inherit the capture encoder's bind
         * shadows (encoder addresses are recycled by the driver) */
        [self compatBindInvalidate];
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
    if (!(geomEnvRead(), gEnv.skip_kernel))
    {
        bool gGsDispatched = true;

        /* the geometry stage's textures were realized before the FIRST encoder of this
         * draw opened (geomRealizeKernelStageTextures, right after endRenderEncoding):
         * the tess import encoder opened there stays open across this whole section, so
         * a deferred upload here would open a blit encoder Metal forbids */

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
        if (!(geomEnvRead(), gEnv.skip_order))
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
        gGsDispatched = !(geomEnvRead(), gEnv.skip_gs);
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

        if ([self geomBindStageBuffers: enc geomProgram: gp] && gGsDispatched)
        {
            NSUInteger wg = 32;
            uint64_t groups = (threads + wg - 1) / wg;
            if (!groups) groups = 1;
            if (groups > 0xffffffffull) groups = 0xffffffffull;

            if (gpu_assembly && !implicit && !(geomEnvRead(), gEnv.worst_case))
            {
                /* the prep kernel computed the exact threadgroup count */
                [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                               indirectBufferOffset: gPrepOff + MGL_GEOM_PREP_GROUPS_OFF * sizeof(uint32_t)
                                              threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];
            }
            else
                /* an implicit draw's count is exact on the CPU already (the affine window
                 * arithmetic bounds it); no prep kernel ever wrote the indirect arguments */
                [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)groups, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(wg, 1, 1)];

            /* order the emitted strips: scan the per-thread list counts into bases, then
             * copy the per-thread scratch segments into the raster list in primitive
             * order (the GL-required emission order); finalize folds the total into the
             * indirect draw arguments */
            if (!(geomEnvRead(), gEnv.skip_order))
            {
                uint32_t scan_in[2] = { (uint32_t)threads, (uint32_t)gp->max_vertices * 3u };
                uint32_t chunks = (uint32_t)((threads + 1023) / 1024);
                uint32_t pscan_in[1] = { (uint32_t)threads };
                uint32_t pbase_in[1] = { chunks };
                size_t sums_off = ((size_t)idx_cap * 2 + (size_t)threads * 2) * sizeof(uint32_t);
                /* 1a: chunk-parallel scan of the per-thread counts (chunk-local bases) */
                [enc setComputePipelineState: gPscanPso];
                [enc setBytes: pscan_in length: sizeof pscan_in atIndex: 0];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * 2 * sizeof(uint32_t) atIndex: 1];
                [enc setBuffer: gIdx offset: sums_off atIndex: 2];
                if (!(geomEnvRead(), gEnv.skip_scan))
                [enc dispatchThreadgroups: MTLSizeMake(chunks, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(1024, 1, 1)];
                if (!(geomEnvRead(), gEnv.no_barrier))
                    [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];

                /* 1b: scan the chunk totals into chunk bases (+ the total into ctl[1]) */
                [enc setComputePipelineState: gPbasePso];
                [enc setBytes: pbase_in length: sizeof pbase_in atIndex: 0];
                [enc setBuffer: gCtl offset: 0 atIndex: 1];
                [enc setBuffer: gIdx offset: sums_off atIndex: 2];
                [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(32, 1, 1)];
                if (!(geomEnvRead(), gEnv.no_barrier))
                    [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];

                [enc setComputePipelineState: gLcopyPso];
                [enc setBytes: scan_in length: sizeof scan_in atIndex: 0];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * sizeof(uint32_t) atIndex: 1];
                [enc setBuffer: gIdx offset: (NSUInteger)idx_cap * 2 * sizeof(uint32_t) atIndex: 2];
                [enc setBuffer: gIdx offset: 0 atIndex: 3];
                [enc setBuffer: gIdx offset: sums_off atIndex: 4];
                NSUInteger lwg = 32;
                uint64_t cgroups = (threads + lwg - 1) / lwg;
                if (!cgroups) cgroups = 1;
                if (!(geomEnvRead(), gEnv.skip_copy))
                [enc dispatchThreadgroups: MTLSizeMake((NSUInteger)cgroups, 1, 1)
                      threadsPerThreadgroup: MTLSizeMake(lwg, 1, 1)];
                if (!(geomEnvRead(), gEnv.no_barrier))
                    [enc memoryBarrierWithScope: MTLBarrierScopeBuffers];
            }

            /* fold the counters into the indirect draw arguments (also uses buffer(0),
             * so it runs after the GS dispatch, which is the last user of the stage
             * buffers — no slot is left shadowed for a later kernel) */
            if (!(geomEnvRead(), gEnv.skip_order))
            {
            [enc setComputePipelineState: gFinPso];
            [enc setBuffer: gCtl offset: 0 atIndex: 0];
            [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            /* fold this draw's per-stream counters into the persistent query words
             * (queries.c reads them through the ext_geom query hooks); the dispatch is
             * ordered after the geometry kernel within this encoder */
            if (!gQacc)
                gQacc = [_device newBufferWithLength: 16 * sizeof(uint32_t)
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
            }   /* !skip_order */
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
    if ((geomEnvRead(), gEnv.skip_raster))
    {
        geomMarkPools(_currentCommandBuffer);   /* diagnostic: skip the pass-through raster */
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
    /* the pass installs the vertex-override pipeline (the export vertex function).
     * The compat pipeline fast path caches PSOs keyed on the PROGRAM's own stage
     * functions + VAO/rstate/FBO content — it does not know ext_geom->vertex_override,
     * so two cull/geometry passes whose programs compile to the same MSL but whose
     * export functions have different record strides (KHR-GL46 cull_distance: the
     * gl_ClipDistance[1]/gl_CullDistance[7] shape recompiled with an unsized
     * gl_ClipDistance[]) share a cache slot and the second pass draws with the first
     * pass's PSO — a stride mismatch that renders the previous draw's records.
     * Rebuilding the PSO through the descriptor path on EVERY pass is not an option
     * (glmark2 gl41-geometry: 21174 -> 2863 FPS), so borrow DIRTY_SHADER — nothing
     * but that fast path's decline test reads it — only when the export function
     * changed since the last pass: the first pass of a program/family forces the
     * descriptor build (which fills the slot with the override PSO) and every
     * following pass with the same function hits the cache, whose slot content is
     * then correct by construction (a different export function always forces). */
    {
        /* the pass's pipeline identity: the owning program, the export function and
         * its MSL text (all three change on every gp rebuild; address reuse after a
         * free would have to match all three) */
        static GLuint last_prog;
        static void *last_export_fn, *last_export_msl;
        if (gp->name != last_prog || gp->draw_function != last_export_fn ||
            (void *)gp->draw_msl != last_export_msl)
        {
            ctx->state.dirty_bits |= DIRTY_SHADER;
            last_prog = gp->name;
            last_export_fn = gp->draw_function;
            last_export_msl = gp->draw_msl;
        }
    }
    ctx->state.dirty_bits |= DIRTY_PROGRAM | DIRTY_VAO | DIRTY_STATE;
    if (!BUF()->begin_draw(ctx)) return false;
    ctx->state.dirty_bits &= ~DIRTY_SHADER;   /* borrowed: no other consumer exists */
    geomMarkPools(_currentCommandBuffer);   /* after begin_draw: covers any command buffer it made */
    geomTimeCommandBuffer(_currentCommandBuffer);

    {
        /* the kernel expands every emitted strip into an index *list*, so the raster pass
         * draws lists (points stay points) */
        MTLPrimitiveType prim = gp->out_type == GL_POINTS ? MTLPrimitiveTypePoint
                             : gp->out_type == GL_LINE_STRIP ? MTLPrimitiveTypeLine
                                                             : MTLPrimitiveTypeTriangle;
        if (!_currentRenderEncoder) return false;

        /* compat perf: shadow-filtered bind */
        [self compatBindVertexBuffer:gOutv offset:gOutvOff atIndex:MGL_GEOM_MSL_DRAW];

        if ((geomEnvRead(), gEnv.draw))
        {
            uint32_t *cp = (uint32_t *)gCtl.contents;
            fprintf(stderr, "geom: trace raster prim=%d args=%u %u %u %u %u idx=%u %u %u\n",
                    (int)prim, cp[8], cp[9], cp[10], cp[11], cp[12],
                    ((uint32_t *)gIdx.contents)[0], ((uint32_t *)gIdx.contents)[1], ((uint32_t *)gIdx.contents)[2]);
            if (getenv("MGL_CULL_TRACE"))
            {
                uint32_t *ix = (uint32_t *)gIdx.contents;
                float *ov = (float *)gOutv.contents;
                fprintf(stderr, "cull: idx[0..7] = %u %u %u %u %u %u %u %u  out_words=%u\n",
                        ix[0], ix[1], ix[2], ix[3], ix[4], ix[5], ix[6], ix[7], gp->out_words);
                for (int r = 0; r < 9; r++)
                {
                    int o = r * (int)gp->out_words;
                    fprintf(stderr, "cull: out rec %d: pos %5.2f %5.2f\n", r, ov[o + 2], ov[o + 3]);
                }
            }
        }
        [_currentRenderEncoder drawIndexedPrimitives: prim
                                               indexType: MTLIndexTypeUInt32
                                             indexBuffer: gIdx
                                       indexBufferOffset: 0
                                          indirectBuffer: gCtl
                                    indirectBufferOffset: MGL_GEOM_ARG_OFFSET * sizeof(uint32_t)];

        /* cull diagnostics: dump the compacted index list and the emitted records once
         * the GPU has finished (a completed handler sees the final contents; an inline
         * read would race the compute that produces them) */
        if (getenv("MGL_CULL_DUMP") && _currentCommandBuffer &&
            _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
        {
            id<MTLBuffer> idx = gIdx, outv = gOutv, ctl = gCtl;
            size_t idx_off = 0, out_off = gOutvOff / 4;
            uint32_t ow = gp->out_words, nrec = (uint32_t)((outv.length - gOutvOff) / 4 / (ow ? ow : 1));
            [_currentCommandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> b) {
                uint32_t *ix = (uint32_t *)idx.contents;
                uint32_t *cp = (uint32_t *)ctl.contents;
                uint32_t n = cp ? cp[MGL_GEOM_ARG_OFFSET] : 0;
                if (n > 64) n = 64;
                fprintf(stderr, "culldump: %u indices:", n);
                for (uint32_t i = 0; i < n; i++) fprintf(stderr, " %u", ix[idx_off + i]);
                fprintf(stderr, "\n");
                float *ov = (float *)outv.contents;
                uint32_t r = nrec < 16 ? nrec : 16;
                for (uint32_t k = 0; k < r; k++)
                {
                    float *rec = ov + out_off + (size_t)k * ow;
                    fprintf(stderr, "culldump: rec %2u: pos %6.3f %6.3f %6.3f %6.3f tag %u %u\n",
                            k, rec[2], rec[3], rec[4], rec[5],
                            *(uint32_t *)&rec[0], *(uint32_t *)&rec[1]);
                }
            }];
        }

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
        /* spvBufferSizeConstants[] is indexed by the resource's Metal argument index
         * (SpirvResource.msl_binding), which shifts off the map's ordinal as soon as
         * an image-atomic emulation buffer interleaves in the MSL signature */
        GLuint si = (m->mtl_index && m->mtl_index < MAX_MAPPED_BUFFERS) ? m->mtl_index : i;
        sizes[si] = sz > 0 ? (uint32_t)sz : 0;
        n = (si >= n) ? si + 1 : n;
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

/* "does this stage's MSL read spvBufferSizeConstants" is immutable after link —
 * the per-draw strstr over the full MSL source was the single biggest per-draw
 * cost in glmark2 (ideas: ~4 ms/frame in two strstr scans per draw); memoize.
 * Shared with the vertex-attribute planner (round 44 added two scans of its
 * own, one per plan rebuild) via mgl_msl_has_size_constants(). */
static bool geomSSBOSizesNeeded(Program *p, int stage)
{
    /* the cache key is (program, msl pointer, stage) plus the first 32 bytes of the
     * text: validating with strlen() cost 2 x strlen(30 KB) PER DRAW (the hit path
     * recomputed it to compare) — a head memcmp catches the recycled-pointer case
     * for free and no scan of the string is needed to hit */
    static struct { const void *p, *msl; char head[32]; int stage; bool needed; } cache[32];
    static int next;
    const char *msl = p->spirv[stage].msl_str;
    for (int i = 0; i < 32; i++)
        if (cache[i].p == (void *)p && cache[i].msl == (void *)msl && cache[i].stage == stage &&
            memcmp(cache[i].head, msl, 32) == 0)
            return cache[i].needed;
    bool needed = strstr(msl, "spvBufferSizeConstants") != NULL;
    cache[next] = (typeof(cache[0])){ (void *)p, (void *)msl, {0}, stage, needed };
    memcpy(cache[next].head, msl, 32);
    next = (next + 1) & 31;
    return needed;
}

/* public form for code holding only the MSL text (per-draw hot paths must not
 * strstr the whole source): memoized on the string pointer, which is unique
 * and immutable per compiled stage */
bool mgl_msl_has_size_constants(const char *msl)
{
    /* content-keyed, not pointer-keyed: a freed program's MSL string address
     * can be recycled for different text (test_fp64 link loops), which must
     * not alias the memoized answer.  The fingerprint is the first 32 bytes —
     * validating with strlen() rescanned the whole (tens-of-KB) string on
     * every hit, and this sits on the per-draw path */
    static struct { const void *msl; char head[32]; bool needed; } cache[64];
    static int next;
    if (!msl) return false;
    for (int i = 0; i < 64; i++)
        if (cache[i].msl == (void *)msl && memcmp(cache[i].head, msl, 32) == 0) return cache[i].needed;
    bool needed = strstr(msl, "spvBufferSizeConstants") != NULL;
    cache[next] = (typeof(cache[0])){ (void *)msl, {0}, needed };
    memcpy(cache[next].head, msl, 32);
    next = (next + 1) & 63;
    return needed;
}

- (void) geomBindSSBOSizes: (BufferMapList *) list stage: (int) stage vertex: (bool) vertex
{    Program *p = ctx->state.program;

    if (!p || stage < 0 || !p->spirv[stage].msl_str) return;
    if (!geomSSBOSizesNeeded(p, stage)) return;
    if (vertex && ctx->state.ext_geom && ((struct MGLExtState_geom *)ctx->state.ext_geom)->vertex_override)
        return;
    if (!_currentRenderEncoder) return;

    uint32_t sizes[MAX_MAPPED_BUFFERS];
    GLuint n = 0;
    [self geomFillSSBOSizes: sizes list: list count: &n];
    if (!n) return;
    /* compat perf: stream through the argument ring (the memcmp inside skips
     * the encode call whenever the SSBO sizes are unchanged between draws) */
    if (vertex)
        [self compatRingVertexBytes:sizes length:(NSUInteger)n * sizeof(uint32_t) atIndex:MGL_SSBO_SIZE_BUFFER_INDEX];
    else
        [self compatRingFragmentBytes:sizes length:(NSUInteger)n * sizeof(uint32_t) atIndex:MGL_SSBO_SIZE_BUFFER_INDEX];
}

/* compute pass: glDispatchCompute's own encoder binding (not the geometry kernels) */
- (void) geomBindSSBOSizesCompute: (id <MTLComputeCommandEncoder>) enc
{
    Program *p = ctx->state.program;

    if (!enc || !p || !p->spirv[_COMPUTE_SHADER].msl_str) return;
    if (!geomSSBOSizesNeeded(p, _COMPUTE_SHADER)) return;

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
static struct { void *program; bool valid; int ops[32]; int wraps[32]; float borders[32][4]; int has_border[32]; } gShadowSpec;

/* mgl_wco encodings (program.c read()-based gather taps) */
static inline int mgl_geom_wrap_code(GLenum w)
{
    switch (w) {
        case GL_REPEAT: return 0;
        case GL_MIRRORED_REPEAT: return 1;
        case GL_CLAMP_TO_BORDER: return 3;  /* taps outside -> the unit's border colour (mgl_border_fc) */
        default: return 2;   /* CLAMP_TO_EDGE; CLAMP / MIRROR_CLAMP_TO_EDGE approximate to edge texels */
    }
}

- (void) geomShadowFCValues:(MTLFunctionConstantValues *)cv stage:(int)stage hash:(uint64_t *)hash
{
    Program *p = ctx->state.program;
    if (!p || stage != _VERTEX_SHADER) return;
    uint32_t mask = p->spirv[_VERTEX_SHADER].shadow_slots;

    /* defaults first: a slot whose sampler has no bound texture must still provide
     * every function constant its MSL declares, or the Metal function cannot be
     * specialized at all */
    for (int s = 0; s < 32; s++) {
        int32_t zero = 0;
        [cv setConstantValue:&zero type:MTLDataTypeInt atIndex:MGL_GEOM_FC_SHADOW_OP_BASE + s];
        [cv setConstantValue:&zero type:MTLDataTypeInt atIndex:MGL_GEOM_FC_WRAP_BASE + s];
    }

    SpirvResourceList *rl = &p->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    int ops[32], wraps[32];
    for (int i = 0; i < 32; i++) { ops[i] = -1; wraps[i] = -1; }

    for (GLuint i = 0; i < rl->count; i++)
    {
        GLuint slot = rl->list[i].msl_binding;
        if (slot >= 32) continue;
        GLuint unit = rl->list[i].binding;
        Texture *tex = mgl_texture_for_sampler(ctx, unit, rl->list[i].target);
        if (!tex) continue;
        Sampler *gs = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;
        TextureParameter *pp = gs ? &gs->params : &tex->params;
        if (mask & (1u << slot)) {
            int op = mgl_geom_shadow_op_of(pp->compare_func);
            ops[slot] = op;
            int32_t v = op;
            [cv setConstantValue:&v type:MTLDataTypeInt atIndex:MGL_GEOM_FC_SHADOW_OP_BASE + slot];
            if (hash) { *hash ^= (uint64_t)(op + 1) << (slot & 7); *hash *= 1099511628211ULL; }
        }
        /* address mode of every sampler slot: the read()-based gather taps apply
         * it themselves, so the wrap mode is baked into the vertex function too */
        int wrap = mgl_geom_wrap_code(pp->wrap_s) + 16 * mgl_geom_wrap_code(pp->wrap_t);
        wraps[slot] = wrap;
        int32_t wv = wrap;
        [cv setConstantValue:&wv type:MTLDataTypeInt atIndex:MGL_GEOM_FC_WRAP_BASE + slot];
        if (hash) { *hash ^= (uint64_t)(wrap + 1) << ((slot + 3) & 7); *hash *= 1099511628211ULL; }

        /* CLAMP_TO_BORDER gathers: the border colour rides in as four float
         * function constants (default 0,0,0,1 — Metal only has three presets) */
        {
            float bcol[4];
            bcol[0] = pp->border_color[0]; bcol[1] = pp->border_color[1];
            bcol[2] = pp->border_color[2]; bcol[3] = pp->border_color[3];
            for (int c = 0; c < 4; c++)
                [cv setConstantValue:&bcol[c] type:MTLDataTypeFloat atIndex:MGL_GEOM_FC_BORDER_BASE + 4 * (int)slot + c];
            memcpy(gShadowSpec.borders[slot], bcol, sizeof bcol);
            gShadowSpec.has_border[slot] = 1;
            if (hash) {
                uint32_t bh = 2166136261u;
                for (int c = 0; c < 4; c++) { bh ^= (uint32_t)(bcol[c] * 1024.0f) + 1u; bh *= 16777619u; }
                *hash ^= (uint64_t)bh << ((slot + 5) & 7);
                *hash *= 1099511628211ULL;
            }
        }
    }

    gShadowSpec.program = (void *)p;
    gShadowSpec.valid = true;
    memcpy(gShadowSpec.ops, ops, sizeof ops);
    memcpy(gShadowSpec.wraps, wraps, sizeof wraps);
}

/* called on the texture-binding path (before the PSO build of the same draw): if the
 * compare function of a shadow-sampled unit or the address mode of a gather-sampled
 * unit changed since the last specialization, the vertex function must be
 * re-specialized — dirty the program so processGLState rebuilds the pipeline (the
 * constants are baked into the Metal function) */
- (void) geomShadowCheckStale:(GLuint)unit texture:(Texture *)tex slot:(GLuint)slot
{
    if (!gShadowSpec.valid || gShadowSpec.program != (void *)ctx->state.program || slot >= 32) return;
    if (!(ctx->state.program->spirv[_VERTEX_SHADER].shadow_slots & (1u << slot)) &&
        gShadowSpec.wraps[slot] < 0) return;
    Sampler *gs = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;
    TextureParameter *pp = gs ? &gs->params : &tex->params;
    int wrap = mgl_geom_wrap_code(pp->wrap_s) + 16 * mgl_geom_wrap_code(pp->wrap_t);
    bool border_changed = gShadowSpec.has_border[slot] &&
        memcmp(pp->border_color, gShadowSpec.borders[slot], sizeof pp->border_color) != 0;
    if ((mgl_geom_shadow_op_of(pp->compare_func) != gShadowSpec.ops[slot] &&
         (ctx->state.program->spirv[_VERTEX_SHADER].shadow_slots & (1u << slot))) ||
        (gShadowSpec.wraps[slot] >= 0 && wrap != gShadowSpec.wraps[slot]) ||
        border_changed)
    {
        gShadowSpec.valid = false;   /* the rebuild that follows re-records it */
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }
}

@end
