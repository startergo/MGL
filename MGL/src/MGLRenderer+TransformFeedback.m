/*
 * MGLRenderer+TransformFeedback.m — Metal side of the 'xfb' domain.
 *
 * Runs the capture variant of a program's vertex shader (see mgl_ext_xfb.h) in its own
 * command buffer: an attachment-less render pass with rasterization disabled, the vertex
 * buffers / uniform buffers bound exactly as the renderer binds them for the real draw, and
 * the capture SSBO at buffer(MGL_XFB_MSL_BUFFER_INDEX).  The command buffer is waited on and
 * the records are handed back to transform_feedback.c, which assembles and packs primitives.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"   /* compat: bind shadow (perf session 3) */
#include "mgl_ext_xfb.h"
#include "mgl_ext_buffers.h"   /* gpuaudit: GPU-write serial tracking for the xfb scatter */

@interface MGLRenderer (TransformFeedback)
- (void) bindExtXfbFuncsToGLMContext: (GLMContext) glm_ctx;
- (const uint32_t *) xfbCapture: (MGLXfbProgram *) xp vstart: (GLint) vstart vcount: (GLsizei) vcount
                      instances: (GLsizei) instances baseInstance: (GLuint) baseinstance;
@end

/* scratch buffer reused across captures (per renderer) */
static id<MTLBuffer> xfbScratch;

static const uint32_t *xfbCaptureC(GLMContext glm_ctx, MGLXfbProgram *xp, GLint vstart, GLsizei vcount,
                                   GLsizei instances, GLuint baseinstance)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [r xfbCapture: xp vstart: vstart vcount: vcount instances: instances baseInstance: baseinstance];
}

/* gpuaudit (see the implementation below): un-waited capture for the GPU scatter */
@interface MGLRenderer (GpuauditXfbCapture)
- (const uint32_t *) xfbCaptureAsync: (MGLXfbProgram *) xp vstart: (GLint) vstart vcount: (GLsizei) vcount
                          instances: (GLsizei) instances baseInstance: (GLuint) baseinstance
                        outCommand: (void **) out_cb outScratch: (void **) out_scratch;
@end

static void xfbBufferGPUWrittenC(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size)
{
    (void)offset; (void)size;
    /* the scatter kernel wrote through the buffer's MTLBuffer: stamp the GPU-write
     * serial so a later CPU read waits for it (see mgl_buffers_prepare_cpu_read) */
    mgl_buffers_mark_gpu_use(glm_ctx, buf, true);
}

static const uint32_t *xfbCaptureAsyncC(GLMContext glm_ctx, MGLXfbProgram *xp, GLint vstart, GLsizei vcount,
                                        GLsizei instances, GLuint baseinstance, void **out_cb, void **out_scratch)
{
    MGLRenderer *r = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [r xfbCaptureAsync: xp vstart: vstart vcount: vcount
                    instances: instances baseInstance: baseinstance outCommand: out_cb outScratch: out_scratch];
}

static void xfbBufferWrittenC(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size)
{
    (void)glm_ctx;
    if (!buf->data.mtl_data || !size) return;

    id<MTLBuffer> mb = (__bridge id<MTLBuffer>)buf->data.mtl_data;
    if (offset + size > mb.length) {
        if (offset >= mb.length) return;
        size = mb.length - offset;
    }
    /* small buffers keep a separate CPU copy (buffer_data) next to the MTLBuffer */
    if ((void *)buf->data.buffer_data != mb.contents)
        memcpy((uint8_t *)mb.contents + offset, (const uint8_t *)buf->data.buffer_data + offset, size);
    if (mb.storageMode == MTLStorageModeManaged)
        [mb didModifyRange: NSMakeRange(offset, size)];
}

static void xfbReleaseProgramC(GLMContext glm_ctx, MGLXfbProgram *xp)
{
    (void)glm_ctx;
    if (xp->mtl_pipeline) CFBridgingRelease(xp->mtl_pipeline);
    if (xp->mtl_function) CFBridgingRelease(xp->mtl_function);
    if (xp->mtl_library) CFBridgingRelease(xp->mtl_library);
    xp->mtl_pipeline = NULL;
    xp->mtl_function = NULL;
    xp->mtl_library = NULL;
    xp->mtl_pipeline_key = NULL;
}

@implementation MGLRenderer (TransformFeedback)

- (void) bindExtXfbFuncsToGLMContext: (GLMContext) glm_ctx
{
    struct MGLExtMetal_xfb *m = glm_ctx->mtl_funcs.ext_xfb;
    if (!m) {
        m = calloc(1, sizeof(struct MGLExtMetal_xfb));
        if (!m) return;
        glm_ctx->mtl_funcs.ext_xfb = m;
    }
    m->capture = xfbCaptureC;
    m->buffer_written = xfbBufferWrittenC;
    m->capture_async = xfbCaptureAsyncC;            /* gpuaudit: un-waited capture for the GPU scatter */
    m->buffer_gpu_written = xfbBufferGPUWrittenC;   /* gpuaudit: kernel-written range tracking */
    m->release_program = xfbReleaseProgramC;
    mgl_xfb_install_draw_hooks(glm_ctx);
}

- (id<MTLRenderPipelineState>) xfbPipelineFor: (MGLXfbProgram *) xp vertexDescriptor: (MTLVertexDescriptor *) vd
{
    NSError *error = nil;

    if (!xp->mtl_function) {
        id<MTLLibrary> lib = [_device newLibraryWithSource: [NSString stringWithUTF8String: xp->msl]
                                                   options: nil error: &error];
        if (!lib) {
            NSLog(@"MGL xfb: capture variant failed to compile: %@", error);
            return nil;
        }
        id<MTLFunction> fn = [lib newFunctionWithName: [NSString stringWithUTF8String: xp->entry]];
        if (!fn) {
            NSLog(@"MGL xfb: capture entry point %s not found", xp->entry);
            return nil;
        }
        xp->mtl_library = (void *)CFBridgingRetain(lib);
        xp->mtl_function = (void *)CFBridgingRetain(fn);
    }

    /* the pipeline depends on the vertex layout; rebuild when it changes */
    NSString *key = [vd description];
    if (xp->mtl_pipeline && xp->mtl_pipeline_key) {
        NSString *old = (__bridge NSString *)xp->mtl_pipeline_key;
        if ([old isEqualToString: key])
            return (__bridge id<MTLRenderPipelineState>)xp->mtl_pipeline;
    }

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.label = @"MGL transform feedback capture";
    pd.vertexFunction = (__bridge id<MTLFunction>)xp->mtl_function;
    pd.fragmentFunction = nil;
    pd.vertexDescriptor = vd;
    pd.rasterizationEnabled = NO;
    pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;

    id<MTLRenderPipelineState> ps = [_device newRenderPipelineStateWithDescriptor: pd error: &error];
    if (!ps) {
        NSLog(@"MGL xfb: capture pipeline creation failed: %@", error);
        return nil;
    }
    if (xp->mtl_pipeline) CFBridgingRelease(xp->mtl_pipeline);
    if (xp->mtl_pipeline_key) CFBridgingRelease(xp->mtl_pipeline_key);
    xp->mtl_pipeline = (void *)CFBridgingRetain(ps);
    xp->mtl_pipeline_key = (void *)CFBridgingRetain(key);
    return ps;
}

/* gpuaudit: async capture - encodes the capture pass but does not commit/wait, so the
 * record can be consumed on the GPU (see MGLRenderer+Gpuaudit.m).  The returned
 * command buffer is retained; the caller commits it (or chains after it). */
- (const uint32_t *) xfbCaptureAsync: (MGLXfbProgram *) xp vstart: (GLint) vstart vcount: (GLsizei) vcount
                          instances: (GLsizei) instances baseInstance: (GLuint) baseinstance
                        outCommand: (void **) out_cb outScratch: (void **) out_scratch
{
    /* geom (reported): the active program may come from a bound pipeline (no useProgram):
     * the vertex stage of the pipeline is the program the capture resolves against, the
     * same way the C side's xfb_active_program does */
    Program *xfb_prog = ctx->state.program;
    if (!xfb_prog && ctx->state.program_pipeline)
        xfb_prog = ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER];
    if (!_device || !_commandQueue || !xp->msl || !xfb_prog || !VAO()) return NULL;

    /* map uniform / storage / vertex buffers exactly like the real draw does */
    if (![self mapBuffersToMTL]) return NULL;
    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list]) return NULL;
    /* realize the vertex stage's textures BEFORE the capture encoder exists: the draw
     * path's bindTexturesToCurrentRenderEncoder (called on the capture encoder below)
     * would otherwise realize them mid-encode, and a deferred upload commits+waits —
     * ending this capture encoder twice crashed endEncoding (texture_size_vertex_sh) */
    [self geomRealizeStageTextures: xfb_prog stage: _VERTEX_SHADER];

    MTLVertexDescriptor *vd = [self generateVertexDescriptor];
    if (!vd) return NULL;
    /* the next real draw must rebuild its own state from the same mapping */
    ctx->state.dirty_bits |= DIRTY_VAO;

    id<MTLRenderPipelineState> ps = [self xfbPipelineFor: xp vertexDescriptor: vd];
    if (!ps) return NULL;

    size_t records = (size_t)vcount * (size_t)instances;
    size_t bytes = (MGL_XFB_HEADER_WORDS + records * xp->record_words) * sizeof(uint32_t);
    if (!xfbScratch || xfbScratch.length < bytes) {
        size_t sz = 4096;
        while (sz < bytes) sz *= 2;
        xfbScratch = [_device newBufferWithLength: sz options: MTLResourceStorageModeShared];
        if (!xfbScratch) return NULL;
    }
    uint32_t *hdr = (uint32_t *)xfbScratch.contents;
    hdr[0] = xp->vid_absolute ? (uint32_t)vstart : 0;
    hdr[1] = (uint32_t)vcount;
    hdr[2] = xp->iid_absolute ? baseinstance : 0;
    hdr[3] = xp->record_words;

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.renderTargetWidth = 1;
    rp.renderTargetHeight = 1;
    rp.defaultRasterSampleCount = 1;

    /* ordering hazard: the pending GL command buffer can still hold texture /
     * buffer uploads (uncommitted blits).  Metal runs same-queue command buffers
     * in COMMIT order, so a capture buffer created here would execute BEFORE
     * that pending work — a vertex stage sampling in the capture read an
     * un-uploaded (all-zero) texture (KHR-GL46.gpu_shader5 texture_gather_offset).
     * Commit the GL buffer first (like glFlush would). */
    if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
        [self flushCommandBuffer: false];

    id<MTLCommandBuffer> cb = [_commandQueue commandBuffer];

    /* atomic counters: GL runs the vertex stage ONCE per draw — the capture pass is an
     * MGL re-execution, so its atomicCounterIncrement/... must not hit the real counter
     * buffer a second time (the render draw below performs the real increments).  Snapshot
     * each bound counter buffer into a scratch copy (blit runs before the capture encoder)
     * and bind the copy over the injected atomic slots afterwards.  All vars of one GL
     * binding point share a snapshot so cross-offset visibility is preserved. */
    struct { id<MTLBuffer> buf; NSUInteger offset; } acShadow[MGL_MAX_ATOMIC_VARS];
    for (int i = 0; i < MGL_MAX_ATOMIC_VARS; i++) { acShadow[i].buf = nil; acShadow[i].offset = 0; }
    if (xp->ac_count > 0 && xp->ac_shadow)
    {
        id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
        for (int i = 0; i < xp->ac_count; i++)
        {
            GLuint binding = xp->ac_binding[i];
            if (binding >= MAX_BINDABLE_BUFFERS) continue;
            BufferBaseTarget *bt = &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[binding];
            if (!bt->buf || !bt->buf->data.mtl_data) continue;   /* generic bind keeps the
                                                                    zero-filled placeholder */
            id<MTLBuffer> src = (__bridge id<MTLBuffer>)bt->buf->data.mtl_data;
            /* one snapshot per binding point (vars of the same point see each other) */
            id<MTLBuffer> snap = nil;
            for (int j = 0; j < i && !snap; j++)
                if (xp->ac_binding[j] == binding && acShadow[j].buf)
                    snap = acShadow[j].buf;
            if (!snap)
            {
                snap = [_device newBufferWithLength: src.length options: MTLResourceStorageModeShared];
                if (!snap) { [be endEncoding]; return NULL; }
                [be copyFromBuffer: src sourceOffset: 0 toBuffer: snap destinationOffset: 0 size: src.length];
            }
            acShadow[i].buf = snap;
            acShadow[i].offset = (NSUInteger)bt->offset;
        }
        [be endEncoding];
    }
    cb.label = @"MGL transform feedback capture";
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor: rp];
    if (!enc) return NULL;
    [enc setRenderPipelineState: ps];

    /* reuse the renderer's binding code on our encoder */
    id<MTLRenderCommandEncoder> saved = _currentRenderEncoder;
    _currentRenderEncoder = enc;
    /* compat perf: the bind shadow must not leak entries between this capture
     * encoder and the GL encoder (a recycled encoder address would otherwise
     * inherit stale skips) */
    [self compatBindInvalidate];
    bool ok = [self bindVertexBuffersToCurrentRenderEncoder];
    if (ok)
    {
        /* the capture vertex function samples too: textureSize/sampling in a
         * transform-feedback-captured vertex shader read the default 1x1 fallback
         * without this (KHR-GL46.texture_cube_map_array texture_size_vertex_sh /
         * texture_size_geometry_sh: textureSize returned 1,1,1) */
        [self bindTexturesToCurrentRenderEncoder];
        /* vattrib: the plan generateVertexDescriptor just built routes each
         * attribute's buffer to its MSL slot; bindVertexBuffersToCurrentRenderEncoder
         * skips attribute_mask entries (they belong to the vattrib plan).  On the
         * normal draw path the processGLState "PreDraw:" hook does this; the
         * capture has no processGLState, so call it here (KHR-GL46.direct_state_access
         * vertex_arrays_enable_disable_attributes). */
        [self extVattribPreDraw: enc];
    }
    _currentRenderEncoder = saved;
    [self compatBindInvalidate];
    if (!ok) { [enc endEncoding]; return NULL; }

    /* the injected atomic-counter slots read the snapshots, not the real counters
     * (only when the render draw re-runs the vertex stage — see the comment above) */
    for (int i = 0; i < xp->ac_count; i++)
        if (acShadow[i].buf)
            [enc setVertexBuffer: acShadow[i].buf offset: acShadow[i].offset atIndex: xp->ac_mtl_index[i]];

    [enc setVertexBuffer: xfbScratch offset: 0 atIndex: MGL_XFB_MSL_BUFFER_INDEX];
    [enc drawPrimitives: MTLPrimitiveTypePoint vertexStart: (NSUInteger)vstart vertexCount: (NSUInteger)vcount
          instanceCount: (NSUInteger)instances baseInstance: baseinstance];
    [enc endEncoding];

    if (out_cb) *out_cb = (void *)CFBridgingRetain(cb);
    if (out_scratch) *out_scratch = (void *)CFBridgingRetain(xfbScratch);

    return hdr + MGL_XFB_HEADER_WORDS;
}

/* the synchronous capture of the original design: encode, commit, wait, return the
 * CPU-readable record (transform_feedback.c keeps using this for its CPU paths) */
- (const uint32_t *) xfbCapture: (MGLXfbProgram *) xp vstart: (GLint) vstart vcount: (GLsizei) vcount
                      instances: (GLsizei) instances baseInstance: (GLuint) baseinstance
{
    id<MTLCommandBuffer> pending = nil;
    void *cb_ref = NULL, *scratch_ref = NULL;

    const uint32_t *rec = [self xfbCaptureAsync: xp vstart: vstart vcount: vcount
                                     instances: instances baseInstance: baseinstance
                                   outCommand: &cb_ref outScratch: &scratch_ref];
    if (!rec)
        return NULL;

    pending = (__bridge_transfer id<MTLCommandBuffer>)cb_ref;
    if (scratch_ref) CFBridgingRelease(scratch_ref);

    [pending commit];
    [pending waitUntilCompleted];

    if (pending.status != MTLCommandBufferStatusCompleted)
    {
        NSLog(@"MGL xfb: capture command buffer failed: %@", pending.error);
        return NULL;
    }

    return rec;
}

@end
