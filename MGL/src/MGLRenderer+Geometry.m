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

/* the raster pass is a separate method so the pass-through vertex function can be swapped
 * in for it and restored afterwards */
@interface MGLRenderer (Geometry)
- (bool) geomDraw: (MGLGeomDraw *) d;
- (bool) geomRasterPass: (MGLGeomDraw *) d;
- (bool) geomCanAssembleGPU;
@end

/* private renderer methods used here (declared for the compiler; implemented by the
 * buffers domain category) */
#define GEOM()   ((struct MGLExtState_geom *)ctx->state.ext_geom)
#define BUF()    (ctx->mtl_funcs.ext_buffers)

/* per-process scratch (the buffers are shared storage; they are only used by encoders of
 * the command buffer they were written for, so a single set can serve every draw) */
static id<MTLBuffer> gPrim;         /* header + capture index list + input primitive slots */
static id<MTLBuffer> gVsout;        /* capture header + captured VS records */
static id<MTLBuffer> gOutv;         /* geometry output vertex records */
static id<MTLBuffer> gIdx;          /* raster index list */
static void *gEnqueuedCb;            /* the command buffer this category enqueued */
static id<MTLBuffer> gCtl;          /* counters + indirect draw arguments */
static id<MTLBuffer> gPrep;         /* prep kernel inputs, staging command and outputs */
static size_t gPrepOff;             /* this draw's range in it */
static id<MTLBuffer> gCnt;          /* per-thread emitted vertex counts (capture order) */
static size_t gOutvOff, gCntOff;    /* this draw's range inside them */
static id<MTLBuffer> gXbase;        /* compacted transform feedback index per thread */
static id<MTLBuffer> gXfb;          /* pack counters and write offsets (GPU-owned while pending) */
static bool gXfbPending;            /* a pack has not been folded into the object yet */
static GLuint gXfbOwner;            /* the object the counter block belongs to */
static bool gXfbBegin;              /* the counter block still needs its init kernel */
static size_t gPrimBytes, gVsoutBytes, gOutvBytes, gIdxBytes;

static id<MTLComputePipelineState> gGsPso;
static id<MTLComputePipelineState> gInitPso;
static id<MTLComputePipelineState> gPrepPso;
static id<MTLComputePipelineState> gXfbPrePso, gXfbScatterPso, gXfbFinPso, gXfbBeginPso;
static id<MTLComputePipelineState> gFinPso;
static void *gPsoKey;               /* the kernel function the PSOs were built from */

/* stringise a macro value for the embedded MSL */
#define MGL_GEOM_STR2(x) #x
#define MGL_GEOM_STR(x) MGL_GEOM_STR2(x)

/* the input block of the prep kernel: everything the assembly needs, 16 words */
#define MGL_GEOM_PREP_IN_WORDS 16
#define MGL_GEOM_WG            64
#define MGL_GEOM_PREP_BYTES    (64 * sizeof(uint32_t))

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
    "            {   /* first restart in this tile: one ballot per simd group */\n"
    "                uint bb = (uint)(uint64_t)simd_ballot(r);\n"
    "                uint local = bb ? (ctz(bb) + sgid * 32u) : 0xffffffffu;\n"
    "                if (slid == 0u) tgr[sgid] = local;\n"
    "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "                m = 0xffffffffu;\n"
    "                for (uint j = 0; j < WG / 32u; j++) if (tgr[j] < m) m = tgr[j];\n"
    "                threadgroup_barrier(mem_flags::mem_threadgroup);\n"
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
    " * a pack may be in flight.  in[0] = object name, in[1..4] = the starting write offsets */\n"
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

    /* the transform feedback pack kernels */
    id<MTLLibrary> xlib = [device newLibraryWithSource: [NSString stringWithUTF8String: kGeomXfbMSL]
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
    if (!pref || !scf || !finf || !begf) return false;
    gXfbPrePso = [device newComputePipelineStateWithFunction: pref error: &error];
    gXfbScatterPso = [device newComputePipelineStateWithFunction: scf error: &error];
    gXfbFinPso = [device newComputePipelineStateWithFunction: finf error: &error];
    gXfbBeginPso = [device newComputePipelineStateWithFunction: begf error: &error];
    if (!gXfbPrePso || !gXfbScatterPso || !gXfbFinPso || !gXfbBeginPso)
    {
        NSLog(@"MGL geom: xfb pack pipeline failed: %@", error);
        return false;
    }
    return true;
}


/* ring of writes into gPrim so consecutive draws do not race with an earlier command
 * buffer still reading it */
#define MGL_GEOM_PRIM_RING 8
static uint64_t gPrimSerial[MGL_GEOM_PRIM_RING];
static int gPrimNext;

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
    gOutvOff = gCntOff = gPrepOff = 0;
    gXbase = nil; gXfb = nil; gXfbPending = false;
    gXfbOwner = 0; gXfbBegin = false;
    gGsPso = nil; gInitPso = nil; gFinPso = nil; gPrepPso = nil;
    gXfbPrePso = nil; gXfbScatterPso = nil; gXfbFinPso = nil; gXfbBeginPso = nil;

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

bool geom_xfb_gpu_pending(void)
{
    return gXfbPending;
}

bool geom_xfb_gpu_valid(GLMContext ctx, GLuint name)
{
    (void)ctx;
    return !gXfbPending || gXfbOwner == name;
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

    if (!gXfbScatterPso) return false;
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

    enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc) return false;
    [enc pushDebugGroup: @"MGL geometry xfb pack"];

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
    [enc setBuffer: gOutv offset: gOutvOff atIndex: 2];
    [enc setBuffer: gCnt offset: gCntOff atIndex: 3];
    [enc setBuffer: gXbase offset: 0 atIndex: 4];
    [enc setBuffer: gXfb offset: 0 atIndex: 5];
    for (GLuint b = 0; b < 4; b++)
    {
        id<MTLBuffer> mtl = (tfb[b].buf && tfb[b].buf->data.mtl_data)
                            ? (__bridge id<MTLBuffer>)tfb[b].buf->data.mtl_data : gXfb;
        [enc setBuffer: mtl offset: 0 atIndex: 6 + b];
        if (tfb[b].buf) mgl_buffers_mark_gpu_use(ctx, tfb[b].buf, true);
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

    [enc popDebugGroup];
    [enc endEncoding];

    gXfbPending = true;
    if (getenv("MGL_GEOM_DEBUG"))
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

/* read the pending counters (the caller has synchronised) */
void geom_xfb_gpu_read(GLMContext ctx, MGLXfbObject *o, uint32_t *dvert, uint32_t *dprim)
{
    if (getenv("MGL_GEOM_DEBUG") && gXfb)
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

- (bool) geomDraw: (MGLGeomDraw *) d
{
    MGLGeomProgram *gp = d->gp;
    Program *program = d->program;
    const bool gpu_assembly = (d->prims == NULL);   /* the CPU path supplies the lists */
    uint64_t threads = (uint64_t)(d->prims_count ? d->prims_count : 1)
                       * (uint64_t)(d->instances > 0 ? d->instances : 1) * gp->invocations;
    uint32_t bound = d->count_bound ? d->count_bound : (d->count > 0 ? (uint32_t)d->count : 1);
    size_t outv_cap, idx_cap, prim_bytes, vsout_bytes, cap_idx, cap_prim;
    uint32_t *hdr;
    uint64_t serial;

    if (!_device || !_commandQueue || !program || !gp || !gp->linked) return false;
    if (ctx->state.caps.rasterizer_discard == GL_TRUE && !mgl_xfb_capturing(ctx))
        return false;                       /* nothing to do */

    /* ---- sizing ---- */
    {
        size_t max_out = MGL_GEOM_BUDGET_BYTES / (gp->out_words * sizeof(uint32_t));
        /* with GPU assembly the primitive count is only known on the GPU: dispatch (and
         * size) by the mode's worst case, the kernel returns early past the real count */
        uint64_t bound_threads = gpu_assembly
                                 ? mgl_geom_worst_prims(d->mode, bound)
                                   * (uint64_t)(d->instances > 0 ? d->instances : 1) * gp->invocations
                                 : threads;
        outv_cap = (size_t)bound_threads * (gp->max_vertices ? gp->max_vertices : 1);
        if (outv_cap > max_out) outv_cap = max_out;
        if (!outv_cap) outv_cap = 1;
        idx_cap = outv_cap * 3;
        threads = bound_threads;
    }

    if (gpu_assembly)
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
    {
        size_t need = (size_t)outv_cap * gp->out_words * sizeof(uint32_t);
        if (!gOutv || gOutvOff + need > gOutv.length)
        {
            gOutv = geomBuffer(_device, gOutv, need * 2, &gOutvBytes);
            gOutvOff = 0;
        }
        if (!gOutv) { geomReleaseStatics(); return false; }
        gOutvOff = (gOutvOff + 255) & ~(size_t)255;
        if (gOutvOff + need > gOutv.length) { gOutv = geomBuffer(_device, gOutv, need * 2, &gOutvBytes); gOutvOff = 0; }
    }
    gIdx = geomBuffer(_device, gIdx, idx_cap * sizeof(uint32_t), &gIdxBytes);
    gCtl = geomBuffer(_device, gCtl, 4096, NULL);
    /* the prep kernel's outputs (the capture draw arguments and the kernel's dispatch count)
     * are read by this draw's capture pass and geometry dispatch: a later draw's prep kernel
     * must not overwrite them first */
    {
        if (!gPrep || gPrepOff + MGL_GEOM_PREP_BYTES > gPrep.length)
        {
            gPrep = geomBuffer(_device, gPrep, MGL_GEOM_PREP_BYTES * 16, NULL);
            gPrepOff = 0;
        }
        if (!gPrep) { geomReleaseStatics(); return false; }
        gPrepOff = (gPrepOff + 255) & ~(size_t)255;
        if (gPrepOff + MGL_GEOM_PREP_BYTES > gPrep.length) gPrepOff = 0;
    }
    {
        size_t need = 4096 + (size_t)threads * sizeof(uint32_t);
        if (!gCnt || gCntOff + need > gCnt.length)
        {
            gCnt = geomBuffer(_device, gCnt, need * 4, NULL);
            gCntOff = 0;
        }
        if (!gCnt) { geomReleaseStatics(); return false; }
        gCntOff = (gCntOff + 255) & ~(size_t)255;
        if (gCntOff + need > gCnt.length) gCntOff = 0;
    }
    if (!gOutv || !gIdx || !gCtl || !gPrep || !gCnt) { geomReleaseStatics(); return false; }

    /* wait for an earlier command buffer that still reads the primitive buffer: before a new
     * one is started (waiting afterwards would commit the one being encoded) */
    serial = BUF()->current_serial(ctx);
    if (gPrim && gPrimSerial[gPrimNext])
        BUF()->wait_serial(ctx, gPrimSerial[gPrimNext]);

    gPrim = geomBuffer(_device, gPrim, prim_bytes, &gPrimBytes);
    gVsout = geomBuffer(_device, gVsout, vsout_bytes, &gVsoutBytes);
    if (!gPrim || !gVsout) { geomReleaseStatics(); return false; }

    if (![self geomPipelines: gp program: program]) return false;

    /* ---- CPU-side upload: header, prep inputs, and (for the CPU path) the lists ---- */
    {
        size_t index_words = (size_t)(!gpu_assembly && d->indexed ? d->index_count : 0);
        size_t prim_words = (size_t)(!gpu_assembly ? d->prims_count * gp->in_verts : 0);

        hdr = (uint32_t *)gPrim.contents;
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
                uint32_t *cmd = in + MGL_GEOM_PREP_IN_WORDS;
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
        gPrimSerial[gPrimNext] = serial;
        gPrimNext = (gPrimNext + 1) % MGL_GEOM_PRIM_RING;
    }

    /* capture variant header lives in its own buffer; with GPU assembly the prep kernel
     * fills in the vertex range */
    {
        uint32_t *ch = (uint32_t *)gVsout.contents;
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

    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "geom: draw start cb=%p\n", (void *)(__bridge void *)_currentCommandBuffer);
    /* ---- 0. prep kernel: assemble the primitives on the GPU ---- */
    if (gpu_assembly)
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
            size_t cmd_off = MGL_GEOM_PREP_IN_WORDS * sizeof(uint32_t);
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
        [enc setBuffer: gPrim offset: (MGL_GEOM_HDR_WORDS + cap_idx) * sizeof(uint32_t) atIndex: 3];
        [enc setBuffer: gPrim offset: MGL_GEOM_HDR_WORDS * sizeof(uint32_t) atIndex: 4];
        [enc setBuffer: gPrim offset: 0 atIndex: 5];
        [enc setBuffer: gVsout offset: 0 atIndex: 6];
        [enc setBuffer: gPrep offset: gPrepOff + 32 * sizeof(uint32_t) atIndex: 7];

        [enc dispatchThreadgroups: MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup: MTLSizeMake(MGL_GEOM_WG, 1, 1)];
        [enc popDebugGroup];
        [enc endEncoding];
    }

    /* ---- 1. vertex capture pass ---- */
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
        [enc setVertexBuffer: gVsout offset: 0 atIndex: MGL_GEOM_MSL_VSOUT];

        if (gpu_assembly)
        {
            /* the prep kernel wrote the draw arguments (index count, instances, bases) */
            [enc drawIndexedPrimitives: MTLPrimitiveTypePoint
                             indexType: MTLIndexTypeUInt32
                           indexBuffer: gPrim
                     indexBufferOffset: MGL_GEOM_HDR_WORDS * sizeof(uint32_t)
                        indirectBuffer: gPrep
                  indirectBufferOffset: gPrepOff + (32 + 7) * sizeof(uint32_t)];
        }
        else if (d->indexed)
        {
            [enc drawIndexedPrimitives: MTLPrimitiveTypePoint
                             indexCount: (NSUInteger)d->index_count
                               indexType: MTLIndexTypeUInt32
                             indexBuffer: gPrim
                       indexBufferOffset: MGL_GEOM_HDR_WORDS * sizeof(uint32_t)
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

    /* ---- 2. geometry kernel ---- */
    {
        id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
        if (!enc) return false;

        Program *saved_program = ctx->state.program;
        ctx->state.program = program;

        [enc pushDebugGroup: @"MGL geometry kernel"];
        [enc setComputePipelineState: gGsPso];

        size_t index_words = (size_t)(d->indexed ? d->count : 0);
        [enc setBuffer: gPrim offset: (MGL_GEOM_HDR_WORDS + index_words) * sizeof(uint32_t)
               atIndex: MGL_GEOM_MSL_PRIM];
        [enc setBuffer: gVsout offset: 8 * sizeof(uint32_t) atIndex: MGL_GEOM_MSL_VSOUT];
        [enc setBuffer: gPrim offset: 0 atIndex: MGL_GEOM_MSL_HDR];
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
            if (gpu_assembly && !getenv("MGL_GEOM_WORST_CASE_DISPATCH"))
            {
                /* the prep kernel computed the exact threadgroup count */
                [enc dispatchThreadgroupsWithIndirectBuffer: gPrep
                                               indirectBufferOffset: gPrepOff + (32 + 4) * sizeof(uint32_t)
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

    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "geom: before raster cb=%p\n", (void *)(__bridge void *)_currentCommandBuffer);
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
        NSLog(@"MGL geom: the pass-through vertex function is not compiled");
        return false;
    }

    /* stand the pass-through function in for the vertex stage while the raster pipeline is
     * built (restored afterwards: the vertex program can still be used on its own) */
    {
        Shader *vs = program->shader_slots[_VERTEX_SHADER];
        char *saved_msl = NULL, *saved_entry = NULL;
        void *saved_fn = NULL;
        const char *saved_slot_entry = NULL;
        bool swapped = false;

        if (vs)
        {
            saved_fn = vs->mtl_data.function;
            saved_slot_entry = vs->entry_point;
            vs->mtl_data.function = gp->draw_function;
            vs->entry_point = gp->draw_entry;
            swapped = true;
        }
        if (program->spirv[_VERTEX_SHADER].msl_str)
        {
            saved_msl = program->spirv[_VERTEX_SHADER].msl_str;
            saved_entry = program->spirv[_VERTEX_SHADER].entry_point;
            program->spirv[_VERTEX_SHADER].msl_str = gp->draw_msl;
            program->spirv[_VERTEX_SHADER].entry_point = gp->draw_entry;
        }

        ctx->state.dirty_bits |= DIRTY_PROGRAM;

        bool ok = [self geomRasterPass: d];

        if (vs)
        {
            vs->mtl_data.function = saved_fn;
            vs->entry_point = saved_slot_entry;
        }
        if (saved_msl)
        {
            program->spirv[_VERTEX_SHADER].msl_str = saved_msl;
            program->spirv[_VERTEX_SHADER].entry_point = saved_entry;
        }
        else if (swapped)
        {
            program->spirv[_VERTEX_SHADER].msl_str = NULL;
            program->spirv[_VERTEX_SHADER].entry_point = NULL;
        }

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
