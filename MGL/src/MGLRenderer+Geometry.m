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
#include "MGLRenderer_Private.h"

/* the raster pass is a separate method so the pass-through vertex function can be swapped
 * in for it and restored afterwards */
@interface MGLRenderer (Geometry)
- (bool) geomDraw: (MGLGeomDraw *) d;
- (bool) geomRasterPass: (MGLGeomDraw *) d;
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
static id<MTLBuffer> gCtl;          /* counters + indirect draw arguments */
static size_t gPrimBytes, gVsoutBytes, gOutvBytes, gIdxBytes;

static id<MTLComputePipelineState> gGsPso;
static id<MTLComputePipelineState> gInitPso;
static id<MTLComputePipelineState> gFinPso;
static void *gPsoKey;               /* the kernel function the PSOs were built from */

/* stringise a macro value for the embedded MSL */
#define MGL_GEOM_STR2(x) #x
#define MGL_GEOM_STR(x) MGL_GEOM_STR2(x)

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
    if (gInitPso && gFinPso) return true;

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
    return gInitPso && gFinPso;
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
    gPrim = nil; gVsout = nil; gOutv = nil; gIdx = nil; gCtl = nil;
    gGsPso = nil; gInitPso = nil; gFinPso = nil;

    if (gPsoKey) { CFRelease(gPsoKey); gPsoKey = NULL; }
    gPrimBytes = gVsoutBytes = gOutvBytes = gIdxBytes = 0;
}

static bool geomDrawC(GLMContext glm_ctx, MGLGeomDraw *d)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [r geomDraw: d];
}

@implementation MGLRenderer (Geometry)

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

- (bool) geomDraw: (MGLGeomDraw *) d
{
    MGLGeomProgram *gp = d->gp;
    Program *program = d->program;
    size_t vs_records = (size_t)d->vcount * (size_t)(d->instances > 0 ? d->instances : 1);
    uint64_t threads = (uint64_t)d->prims_count * (uint64_t)(d->instances > 0 ? d->instances : 1) * gp->invocations;
    size_t outv_cap, idx_cap, prim_bytes, vsout_bytes;
    uint32_t *hdr;
    uint64_t serial;

    if (!_device || !_commandQueue || !program || !gp || !gp->linked) return false;
    if (ctx->state.caps.rasterizer_discard == GL_TRUE && !mgl_xfb_capturing(ctx))
        return false;                       /* nothing to do */

    /* ---- sizing ---- */
    {
        size_t max_out = MGL_GEOM_BUDGET_BYTES / (gp->out_words * sizeof(uint32_t));
        outv_cap = (size_t)threads * (gp->max_vertices ? gp->max_vertices : 1);
        if (outv_cap > max_out) outv_cap = max_out;
        if (!outv_cap) outv_cap = 1;
        idx_cap = outv_cap * 3;
    }
    prim_bytes = (MGL_GEOM_HDR_WORDS + (size_t)(d->indexed ? d->count : 0) +
                  (size_t)d->prims_count * gp->in_verts) * sizeof(uint32_t);
    vsout_bytes = (8 + vs_records * gp->in_words) * sizeof(uint32_t);

    gOutv = geomBuffer(_device, gOutv, outv_cap * gp->out_words * sizeof(uint32_t), &gOutvBytes);
    gIdx = geomBuffer(_device, gIdx, idx_cap * sizeof(uint32_t), &gIdxBytes);
    gCtl = geomBuffer(_device, gCtl, 4096, NULL);
    if (!gOutv || !gIdx || !gCtl) { geomReleaseStatics(); return false; }

    /* wait for an earlier command buffer that still reads the primitive buffer */
    serial = BUF()->current_serial(ctx);
    if (gPrim && gPrimSerial[gPrimNext])
        BUF()->wait_serial(ctx, gPrimSerial[gPrimNext]);

    gPrim = geomBuffer(_device, gPrim, prim_bytes, &gPrimBytes);
    gVsout = geomBuffer(_device, gVsout, vsout_bytes, &gVsoutBytes);
    if (!gPrim || !gVsout) { geomReleaseStatics(); return false; }

    if (![self geomPipelines: gp program: program]) return false;

    /* ---- CPU-side upload: header, capture index list, input primitive slots ---- */
    {
        size_t index_words = (size_t)(d->indexed ? d->count : 0);
        size_t prim_words = (size_t)d->prims_count * gp->in_verts;

        hdr = (uint32_t *)gPrim.contents;
        memset(hdr, 0, MGL_GEOM_HDR_WORDS * sizeof(uint32_t));
        hdr[0] = (uint32_t)d->prims_count;
        hdr[1] = (uint32_t)d->instances;
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

        if (index_words)
            memcpy(hdr + MGL_GEOM_HDR_WORDS, d->indices, index_words * sizeof(uint32_t));
        memcpy(hdr + MGL_GEOM_HDR_WORDS + index_words, d->prims, prim_words * sizeof(uint32_t));
        gPrimSerial[gPrimNext] = serial;
        gPrimNext = (gPrimNext + 1) % MGL_GEOM_PRIM_RING;
    }

    /* capture variant header lives in its own buffer */
    {
        uint32_t *ch = (uint32_t *)gVsout.contents;
        ch[0] = gp->vid_absolute ? (uint32_t)d->vstart : 0;
        ch[1] = (uint32_t)d->vcount;
        ch[2] = gp->iid_absolute ? (uint32_t)d->baseinstance : 0;
        ch[3] = (uint32_t)gp->in_words;
    }

    /* ---- 1. vertex capture pass ---- */
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

        if (d->indexed)
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
        [enc setBuffer: gOutv offset: 0 atIndex: MGL_GEOM_MSL_OUTV];
        [enc setBuffer: gIdx offset: 0 atIndex: MGL_GEOM_MSL_IDX];
        [enc setBuffer: gCtl offset: 0 atIndex: MGL_GEOM_MSL_CTL];

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

        [_currentRenderEncoder setVertexBuffer: gOutv offset: 0 atIndex: MGL_GEOM_MSL_DRAW];

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
