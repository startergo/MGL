/*
 * MGLRenderer+Draw.m — Metal side of the 'buffers' domain:
 *   buffer storage (MTLBuffers aliasing Buffer.data.buffer_data), command-buffer
 *   serial tracking for CPU/GPU synchronisation, draws, compute dispatch,
 *   blit copies, memory barriers and fences.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import "MGLRenderer_Private.h"
#import <objc/runtime.h>
#include <stdatomic.h>
#include <time.h>

#include "mgl_ext_buffers.h"
#include "mgl_ext_xfb.h"

typedef struct MGLDrawPriv_t {
    struct MGLExtMetal_buffers pub;     // must be first: ctx->mtl_funcs.ext_buffers points here

    void *tracked_cb;                   // retained id<MTLCommandBuffer> that owns tracked_serial
    uint64_t tracked_serial;
    void *last_cb;                      // retained, last command buffer committed (WillCommit hook)
    uint64_t last_serial;
    _Atomic uint64_t completed;

    void *event;                        // retained id<MTLSharedEvent> for fences
    uint64_t event_value;
} MGLDrawPriv;

typedef struct MGLFence_t {
    uint64_t value;                     // shared event value signalled by the fence
    uint64_t serial;                    // command buffer serial that carries the signal
} MGLFence;

static char kComputePSOKey;

#define PRIV(_ctx_) ((MGLDrawPriv *)((_ctx_)->mtl_funcs.ext_buffers))
#define RENDERER(_ctx_) ((__bridge MGLRenderer *)((_ctx_)->mtl_funcs.mtlObj))

static void atomic_max_u64(_Atomic uint64_t *a, uint64_t v)
{
    uint64_t cur = atomic_load(a);
    while (cur < v && !atomic_compare_exchange_weak(a, &cur, v)) { }
}

@implementation MGLRenderer (Draw)

#pragma mark - command buffer serials

/* serial of the command buffer currently being encoded */
- (uint64_t) extBufSerial
{
    MGLDrawPriv *p = PRIV(ctx);
    id<MTLCommandBuffer> cb = _currentCommandBuffer;

    if (cb && (__bridge void *)cb != p->tracked_cb)
    {
        if (p->tracked_cb)
            CFRelease(p->tracked_cb);

        p->tracked_cb = (void *)CFBridgingRetain(cb);
        p->tracked_serial++;

        uint64_t serial = p->tracked_serial;

        if (cb.status < MTLCommandBufferStatusCommitted)
        {
            _Atomic uint64_t *completed = &p->completed;
            [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
                atomic_max_u64(completed, serial);
            }];
        }
    }

    return p->tracked_serial;
}

- (uint64_t) extBufCompletedSerial
{
    MGLDrawPriv *p = PRIV(ctx);
    uint64_t c = atomic_load(&p->completed);

    if (p->tracked_cb && p->tracked_serial > c)
    {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)p->tracked_cb;
        if (cb.status >= MTLCommandBufferStatusCompleted)
        {
            atomic_max_u64(&p->completed, p->tracked_serial);
            c = p->tracked_serial;
        }
    }

    return c;
}

/* called (via mglCallExtHooks) for every commit made through commitCommandBufferWithAGXRecovery */
- (void) extBuffersWillCommit: (id<MTLCommandBuffer>) cmdbuf
{
    MGLDrawPriv *p = PRIV(ctx);

    if (!p)
        return;

    if ((__bridge void *)cmdbuf == (__bridge void *)_currentCommandBuffer)
        [self extBufSerial];

    if ((__bridge void *)cmdbuf == p->tracked_cb)
    {
        if (p->last_cb)
            CFRelease(p->last_cb);

        p->last_cb = (void *)CFBridgingRetain(cmdbuf);
        p->last_serial = p->tracked_serial;
    }
}

/* the current command buffer can take new encoders */
- (void) extBufEnsureCommandBuffer
{
    if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
    {
        [self newCommandBuffer];
        ctx->state.dirty_bits = DIRTY_ALL;
    }
}

- (void) extBufFlush
{
    // encode pending state-only work (e.g. a glClear without a following draw)
    if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
        [self processGLState: false];

    [self endRenderEncoding];

    if (_currentCommandBuffer && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
    {
        [self extBufSerial];
        [self commitCommandBufferWithAGXRecovery: _currentCommandBuffer];
    }

    // a new command buffer is created on demand (extBufEnsureCommandBuffer); leaving the
    // committed one current keeps the legacy readback / present paths working unchanged

    // the next draw needs a new render encoder with all state re-applied (attachments are loaded)
    ctx->state.dirty_bits = DIRTY_ALL;
}

- (void) extBufWaitSerial: (uint64_t) serial
{
    MGLDrawPriv *p = PRIV(ctx);

    if (serial <= [self extBufCompletedSerial])
        return;

    [self extBufSerial];

    id<MTLCommandBuffer> target = nil;
    uint64_t target_serial = serial;

    if (p->last_cb && p->last_serial >= serial)
    {
        target = (__bridge id<MTLCommandBuffer>)p->last_cb;
        target_serial = p->last_serial;
        /* an open command buffer never completes: commit it before waiting (the tracked_cb
         * path below does this; the last_cb path could reach an uncommitted buffer when a
         * query split created a fresh one, which hung glQueryCounter / GL_TIME_ELAPSED
         * resolves around batched draws) */
        if (target.status < MTLCommandBufferStatusCommitted)
            [self extBufFlush];
    }
    else
    {
        id<MTLCommandBuffer> tracked = (__bridge id<MTLCommandBuffer>)p->tracked_cb;
        target_serial = p->tracked_serial;

        if (tracked && tracked.status < MTLCommandBufferStatusCommitted)
            [self extBufFlush];

        target = tracked;
    }

    if (target)
    {
        /* last resort: a buffer the geometry category enqueued but that was replaced before
         * it was committed will never run, and waiting on it never returns — kick it */
        if (target.status == MTLCommandBufferStatusEnqueued &&
            target != _currentCommandBuffer)
        {
            @try { [target commit]; } @catch (NSException *) { return; }
        }
        if (target.status < MTLCommandBufferStatusCommitted &&
            target == _currentCommandBuffer)
            return;                     /* cannot commit our own encoding buffer: nothing to wait on */

        // one in-order queue: completion of this buffer implies earlier ones completed
        [target waitUntilCompleted];
        atomic_max_u64(&p->completed, target_serial);
    }
}

#pragma mark - buffer storage

- (bool) extBufEnsure: (Buffer *) buf
{
    if (!buf)
        return false;

    if (!buf->data.mtl_data)
        [self bindMTLBuffer: buf];

    return buf->data.mtl_data != NULL;
}

- (bool) extBufRename: (Buffer *) buf
{
    if (!buf->data.mtl_data || !buf->data.buffer_data)
        return false;

    id<MTLBuffer> old = (__bridge id<MTLBuffer>)buf->data.mtl_data;
    id<MTLBuffer> fresh = [_device newBufferWithBytes: old.contents length: old.length options: old.resourceOptions];

    if (!fresh)
        return false;

    // queued command buffers keep their reference to the old storage
    CFBridgingRelease(buf->data.mtl_data);

    buf->data.mtl_data = (void *)CFBridgingRetain(fresh);
    buf->data.buffer_data = (vm_address_t)fresh.contents;
    buf->data.buffer_size = fresh.length;

    return true;
}

- (void) extBufDidModify: (Buffer *) buf offset: (size_t) offset length: (size_t) length
{
    if (!buf->data.mtl_data || length == 0)
        return;

    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf->data.mtl_data;

    if (b.storageMode == MTLStorageModeManaged && offset < b.length)
    {
        if (length > b.length - offset)
            length = b.length - offset;

        [b didModifyRange: NSMakeRange(offset, length)];
    }
}

- (void) extBufRelease: (Buffer *) buf
{
    if (buf->data.mtl_data)
    {
        CFBridgingRelease(buf->data.mtl_data);
        buf->data.mtl_data = NULL;
    }
}

- (void) extBufMakeCPUVisible: (Buffer *) buf serial: (uint64_t) serial
{
    if (buf->data.mtl_data && !_device.hasUnifiedMemory)
    {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf->data.mtl_data;

        if (b.storageMode == MTLStorageModeManaged)
        {
            // discrete GPUs: copy GPU writes back into the CPU-side storage
            [self extBufEnsureCommandBuffer];
            [self endRenderEncoding];

            id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
            [blit synchronizeResource: b];
            [blit endEncoding];

            ctx->state.dirty_bits = DIRTY_ALL;
            serial = [self extBufSerial];
        }
    }

    [self extBufWaitSerial: serial];
}

#pragma mark - draws

- (bool) extBufStorageBinding: (Buffer *) buf
{
    for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++)
    {
        if (ctx->state.buffer_base[_SHADER_STORAGE_BUFFER].buffers[i].buf == buf ||
            ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[i].buf == buf)
            return true;
    }

    return false;
}

/* bind the stage's buffer map to the encoder: real MTLBuffers (aliasing the GL data)
 * for storage / large buffers, setBytes snapshots for small read-only buffers */
- (bool) extBufBindList: (BufferMapList *) list vertex: (bool) vertex
{
    for (GLuint i = 0; i < list->count; i++)
    {
        BufferMap *map = &list->buffers[i];
        Buffer *buf = map->buf;

        if (!buf || !buf->data.buffer_data)
            continue;

        size_t offset = map->offset > 0 ? (size_t)map->offset : 0;
        bool storage = map->attribute_mask == 0 && [self extBufStorageBinding: buf];

        if (storage || buf->data.mtl_data || buf->size >= 4096 ||
            (buf->mapped && (buf->access_flags & GL_MAP_PERSISTENT_BIT)))
        {
            if (![self extBufEnsure: buf])
                return false;

            id<MTLBuffer> mb = (__bridge id<MTLBuffer>)buf->data.mtl_data;

            if (offset >= mb.length)
                continue;

            // coherent persistent mappings: CPU writes become GPU-visible at every draw
            if (buf->mapped && (buf->access_flags & GL_MAP_COHERENT_BIT))
                [self extBufDidModify: buf offset: buf->mapped_offset length: buf->mapped_length];

            if (vertex)
                [_currentRenderEncoder setVertexBuffer: mb offset: offset atIndex: i];
            else
                [_currentRenderEncoder setFragmentBuffer: mb offset: offset atIndex: i];
        }
        else
        {
            if ((GLsizeiptr)offset >= buf->size)
                continue;

            size_t len = buf->size - offset;

            if (vertex)
                [_currentRenderEncoder setVertexBytes: (const char *)buf->data.buffer_data + offset length: len atIndex: i];
            else
                [_currentRenderEncoder setFragmentBytes: (const char *)buf->data.buffer_data + offset length: len atIndex: i];
        }
    }

    return true;
}

- (bool) extBufBeginDraw
{
    [self extBufEnsureCommandBuffer];

    if (![self processGLState: true])
        return false;

    if (!_currentRenderEncoder)
    {
        ctx->state.dirty_bits = DIRTY_ALL;

        if (![self processGLState: true])
            return false;
    }

    if (!_currentRenderEncoder || !_pipelineState)
        return false;

    if (![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true])
        return false;

    if (![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
        return false;

    /* geom: SSBO runtime-array .length() sizes (spvBufferSizeConstants, slot 25).
     * The vertex table's stage is the evaluation shader while tessellation is active
     * (mapBuffersToMTL mirrors this). */
    [self geomBindSSBOSizes: &ctx->state.vertex_buffer_map_list
                     stage: [self extTessActive] ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER
                    vertex: true];
    [self geomBindSSBOSizes: &ctx->state.fragment_buffer_map_list stage: _FRAGMENT_SHADER vertex: false];

    return true;
}

- (void) extBufStampList: (BufferMapList *) list serial: (uint64_t) serial
{
    for (GLuint i = 0; i < list->count; i++)
    {
        Buffer *buf = list->buffers[i].buf;

        if (!buf || !buf->data.mtl_data)
            continue;

        bool storage = list->buffers[i].attribute_mask == 0 && [self extBufStorageBinding: buf];

        MGLBufferTrack *t = mgl_buffers_track(ctx, buf, true);
        if (t)
        {
            t->read_serial = serial;
            if (storage) t->write_serial = serial;
        }
    }
}

- (void) extBufEndDraw
{
    uint64_t serial = [self extBufSerial];

    [self extBufStampList: &ctx->state.vertex_buffer_map_list serial: serial];
    [self extBufStampList: &ctx->state.fragment_buffer_map_list serial: serial];

    if (VAO() && VAO_STATE(element_array.buffer))
        mgl_buffers_mark_gpu_use(ctx, VAO_STATE(element_array.buffer), false);

    if (ctx->state.buffers[_DRAW_INDIRECT_BUFFER])
        mgl_buffers_mark_gpu_use(ctx, ctx->state.buffers[_DRAW_INDIRECT_BUFFER], false);

    // transform feedback capture writes its buffers
    TransformFeedback *tf = ctx->state.transform_feedback;
    if (tf && tf->active && !tf->paused)
    {
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++)
        {
            Buffer *b = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[i].buf;
            if (b) mgl_buffers_mark_gpu_use(ctx, b, true);
        }
    }
}

- (void) extBufDrawArrays: (int) prim first: (GLint) first count: (GLsizei) count instances: (GLsizei) instances baseInstance: (GLuint) base_instance
{
    if (base_instance)
        [_currentRenderEncoder drawPrimitives: (MTLPrimitiveType)prim vertexStart: first vertexCount: count
                                instanceCount: instances baseInstance: base_instance];
    else
        [_currentRenderEncoder drawPrimitives: (MTLPrimitiveType)prim vertexStart: first vertexCount: count
                                instanceCount: instances];
}

- (void) extBufDrawIndexed: (int) prim type: (GLenum) type buffer: (Buffer *) ibuf offset: (size_t) offset
                       cpu: (const void *) cpu count: (GLsizei) count instances: (GLsizei) instances
                baseVertex: (GLint) base_vertex baseInstance: (GLuint) base_instance
{
    MTLIndexType itype = (type == GL_UNSIGNED_INT) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
    size_t isize = (type == GL_UNSIGNED_INT) ? 4 : 2;
    id<MTLBuffer> index_buffer = nil;
    size_t index_offset = 0;

    if (ibuf)
    {
        if (offset + (size_t)count * isize > (size_t)ibuf->size)
            return;  // indices outside the element buffer

        if ((offset % 4) == 0 && [self extBufEnsure: ibuf])
        {
            index_buffer = (__bridge id<MTLBuffer>)ibuf->data.mtl_data;
            index_offset = offset;
        }
        else
        {
            // Metal needs 4-byte aligned index offsets: copy the indices
            mgl_buffers_prepare_cpu_read(ctx, ibuf);
            cpu = (const char *)ibuf->data.buffer_data + offset;
        }
    }

    if (!index_buffer)
    {
        if (!cpu)
            return;

        index_buffer = [_device newBufferWithBytes: cpu length: (size_t)count * isize options: MTLResourceStorageModeShared];
        if (!index_buffer)
            return;
    }

    [_currentRenderEncoder drawIndexedPrimitives: (MTLPrimitiveType)prim indexCount: count indexType: itype
                                     indexBuffer: index_buffer indexBufferOffset: index_offset
                                   instanceCount: instances baseVertex: base_vertex baseInstance: base_instance];
}

- (void) extBufDrawIndirect: (int) prim buffer: (Buffer *) indirect offset: (size_t) offset type: (GLenum) type index: (Buffer *) ibuf
{
    if (![self extBufEnsure: indirect])
        return;

    id<MTLBuffer> ib = (__bridge id<MTLBuffer>)indirect->data.mtl_data;

    if (type)
    {
        if (![self extBufEnsure: ibuf])
            return;

        id<MTLBuffer> eb = (__bridge id<MTLBuffer>)ibuf->data.mtl_data;

        [_currentRenderEncoder drawIndexedPrimitives: (MTLPrimitiveType)prim
                                           indexType: (type == GL_UNSIGNED_INT) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                                         indexBuffer: eb indexBufferOffset: 0
                                      indirectBuffer: ib indirectBufferOffset: offset];
    }
    else
    {
        [_currentRenderEncoder drawPrimitives: (MTLPrimitiveType)prim indirectBuffer: ib indirectBufferOffset: offset];
    }
}

#pragma mark - compute

- (Program *) extBufComputeProgram
{
    if (ctx->state.program && ctx->state.program->shader_slots[_COMPUTE_SHADER])
        return ctx->state.program;

    ProgramPipeline *pp = ctx->state.program_pipeline;
    if (pp && pp->stage_programs[_COMPUTE_SHADER])
        return pp->stage_programs[_COMPUTE_SHADER];

    return NULL;
}

- (bool) extBufDispatchX: (GLuint) x y: (GLuint) y z: (GLuint) z indirect: (Buffer *) indirect offset: (size_t) offset
{
    Program *program = [self extBufComputeProgram];

    if (!program)
        return false;

    if (![self bindMTLProgram: program])
        return false;

    Shader *cs = program->shader_slots[_COMPUTE_SHADER];
    if (!cs || !cs->mtl_data.function)
        return false;

    id<MTLFunction> func = (__bridge id<MTLFunction>)cs->mtl_data.function;

    id<MTLComputePipelineState> pso = objc_getAssociatedObject(func, &kComputePSOKey);
    if (!pso)
    {
        NSError *err = nil;
        pso = [_device newComputePipelineStateWithFunction: func error: &err];
        if (!pso)
        {
            NSLog(@"MGL ERROR: compute pipeline creation failed: %@", err);
            return false;
        }
        objc_setAssociatedObject(func, &kComputePSOKey, pso, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    // the resource reflection helpers read ctx->state.program: present a pipeline's compute
    // program there while this dispatch is encoded
    Program *saved_program = ctx->state.program;
    ctx->state.program = program;
    bool ok = [self extBufEncodeDispatch: program pso: pso x: x y: y z: z indirect: indirect offset: offset];
    ctx->state.program = saved_program;

    return ok;
}

- (bool) extBufEncodeDispatch: (Program *) program pso: (id<MTLComputePipelineState>) pso
                            x: (GLuint) x y: (GLuint) y z: (GLuint) z indirect: (Buffer *) indirect offset: (size_t) offset
{
    if (![self mapGLBuffersToMTLBufferMap: &ctx->state.compute_buffer_map_list stage: _COMPUTE_SHADER])
        return false;

    if (indirect && ![self extBufEnsure: indirect])
        return false;

    [self extBufEnsureCommandBuffer];
    [self endRenderEncoding];

    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc)
        return false;

    [enc setComputePipelineState: pso];

    BufferMapList *list = &ctx->state.compute_buffer_map_list;
    for (GLuint i = 0; i < list->count; i++)
    {
        Buffer *buf = list->buffers[i].buf;

        if (!buf || !buf->data.buffer_data)
            continue;

        size_t off = list->buffers[i].offset > 0 ? (size_t)list->buffers[i].offset : 0;

        if (![self extBufEnsure: buf])
        {
            [enc endEncoding];
            return false;
        }

        id<MTLBuffer> mb = (__bridge id<MTLBuffer>)buf->data.mtl_data;
        if (off < mb.length)
            [enc setBuffer: mb offset: off atIndex: i];
    }

    /* geom: SSBO runtime-array .length() sizes (spvBufferSizeConstants, slot 25) */
    [self geomBindSSBOSizesCompute: enc];

    if (![self bindTexturesToComputeEncoder: enc])
    {
        [enc endEncoding];
        ctx->state.dirty_bits = DIRTY_ALL;
        return false;
    }

    MTLSize local = MTLSizeMake(program->local_workgroup_size.x ? program->local_workgroup_size.x : 1,
                                program->local_workgroup_size.y ? program->local_workgroup_size.y : 1,
                                program->local_workgroup_size.z ? program->local_workgroup_size.z : 1);

    if (indirect)
    {
        [enc dispatchThreadgroupsWithIndirectBuffer: (__bridge id<MTLBuffer>)indirect->data.mtl_data
                               indirectBufferOffset: offset
                              threadsPerThreadgroup: local];
    }
    else
    {
        [enc dispatchThreadgroups: MTLSizeMake(x, y, z) threadsPerThreadgroup: local];
    }

    [enc endEncoding];

    uint64_t serial = [self extBufSerial];

    // every storage binding may have been written
    for (GLuint i = 0; i < list->count; i++)
    {
        Buffer *buf = list->buffers[i].buf;
        if (!buf) continue;

        MGLBufferTrack *t = mgl_buffers_track(ctx, buf, true);
        if (t)
        {
            t->read_serial = serial;
            if ([self extBufStorageBinding: buf]) t->write_serial = serial;
        }
    }

    if (indirect)
        mgl_buffers_mark_gpu_use(ctx, indirect, false);

    // a new render encoder (and full state) is needed for the next draw
    ctx->state.dirty_bits = DIRTY_ALL;

    return true;
}

#pragma mark - copies and barriers

- (void) extBufCopy: (Buffer *) src to: (Buffer *) dst srcOffset: (size_t) so dstOffset: (size_t) dof size: (size_t) size
{
    if (![self extBufEnsure: src] || ![self extBufEnsure: dst])
        return;

    [self extBufEnsureCommandBuffer];
    [self endRenderEncoding];

    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    [blit copyFromBuffer: (__bridge id<MTLBuffer>)src->data.mtl_data sourceOffset: so
                toBuffer: (__bridge id<MTLBuffer>)dst->data.mtl_data destinationOffset: dof size: size];
    [blit endEncoding];

    mgl_buffers_mark_gpu_use(ctx, src, false);
    mgl_buffers_mark_gpu_use(ctx, dst, true);

    ctx->state.dirty_bits = DIRTY_ALL;
}

- (void) extBufBarrier: (GLbitfield) barriers byRegion: (bool) by_region
{
    if (!_currentRenderEncoder)
        return; // encoder boundaries already order compute / blit / render work

    if (by_region && [_currentRenderEncoder respondsToSelector: @selector(memoryBarrierWithScope:afterStages:beforeStages:)])
    {
        MTLBarrierScope scope = MTLBarrierScopeBuffers;
        if (barriers & (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT))
            scope |= MTLBarrierScopeTextures;
        if (barriers & GL_FRAMEBUFFER_BARRIER_BIT)
            scope |= MTLBarrierScopeRenderTargets;

        [_currentRenderEncoder memoryBarrierWithScope: scope
                                          afterStages: MTLRenderStageVertex | MTLRenderStageFragment
                                         beforeStages: MTLRenderStageVertex | MTLRenderStageFragment];
        return;
    }

    // draws within one render encoder are not ordered against each other's shader
    // writes: end the encoder, the next draw starts a new one (attachments loaded)
    [self endRenderEncoding];
    ctx->state.dirty_bits = DIRTY_ALL;
}

#pragma mark - fences

- (id<MTLSharedEvent>) extBufEvent
{
    MGLDrawPriv *p = PRIV(ctx);

    if (!p->event)
        p->event = (void *)CFBridgingRetain([_device newSharedEvent]);

    return (__bridge id<MTLSharedEvent>)p->event;
}

- (void) extBufFence: (Sync *) sync
{
    MGLDrawPriv *p = PRIV(ctx);
    MGLFence *f = (MGLFence *)calloc(1, sizeof(MGLFence));

    id<MTLSharedEvent> ev = [self extBufEvent];

    [self extBufEnsureCommandBuffer];
    [self endRenderEncoding];

    f->value = ++p->event_value;
    [_currentCommandBuffer encodeSignalEvent: ev value: f->value];
    f->serial = [self extBufSerial];

    ctx->state.dirty_bits = DIRTY_ALL;

    sync->mtl_event = f;
}

- (bool) extBufSignaled: (Sync *) sync
{
    MGLFence *f = (MGLFence *)sync->mtl_event;

    if (!f)
        return true;

    return [self extBufEvent].signaledValue >= f->value;
}

- (GLenum) extBufClientWait: (Sync *) sync flush: (bool) flush timeout: (GLuint64) timeout
{
    MGLFence *f = (MGLFence *)sync->mtl_event;
    MGLDrawPriv *p = PRIV(ctx);

    if ([self extBufSignaled: sync])
        return GL_ALREADY_SIGNALED;

    // the fence's command buffer must be submitted for the wait to ever finish
    if (f && (flush || timeout > 0))
    {
        [self extBufSerial];
        if (f->serial == p->tracked_serial && _currentCommandBuffer.status < MTLCommandBufferStatusCommitted)
            [self extBufFlush];
    }

    if (timeout == 0)
        return GL_TIMEOUT_EXPIRED;

    uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);

    while (![self extBufSignaled: sync])
    {
        uint64_t elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;

        if (elapsed >= timeout)
            return GL_TIMEOUT_EXPIRED;

        uint64_t remaining = timeout - elapsed;
        struct timespec ts = { 0, (long)(remaining < 50000 ? remaining : 50000) };
        nanosleep(&ts, NULL);
    }

    return GL_CONDITION_SATISFIED;
}

- (void) extBufServerWait: (Sync *) sync
{
    MGLFence *f = (MGLFence *)sync->mtl_event;

    if (!f || [self extBufSignaled: sync])
        return;

    // all of this context's work runs on one in-order queue, so commands after the
    // fence already execute after it; outside a render pass also make the GPU wait
    // on the event explicitly
    if (!_currentRenderEncoder)
    {
        [self extBufEnsureCommandBuffer];
        [_currentCommandBuffer encodeWaitForEvent: [self extBufEvent] value: f->value];
    }
}

- (void) extBufDeleteFence: (Sync *) sync
{
    free(sync->mtl_event);
    sync->mtl_event = NULL;
}

#pragma mark - C trampolines

static uint64_t c_current_serial(GLMContext c) { return [RENDERER(c) extBufSerial]; }
static uint64_t c_completed_serial(GLMContext c) { return [RENDERER(c) extBufCompletedSerial]; }
static void c_flush(GLMContext c) { [RENDERER(c) extBufFlush]; }
static void c_wait_serial(GLMContext c, uint64_t s) { [RENDERER(c) extBufWaitSerial: s]; }
static void c_make_cpu_visible(GLMContext c, Buffer *b, uint64_t s) { [RENDERER(c) extBufMakeCPUVisible: b serial: s]; }
static void c_barrier(GLMContext c, GLbitfield b, bool r) { [RENDERER(c) extBufBarrier: b byRegion: r]; }
static bool c_ensure(GLMContext c, Buffer *b) { return [RENDERER(c) extBufEnsure: b]; }
static bool c_rename(GLMContext c, Buffer *b) { return [RENDERER(c) extBufRename: b]; }
static void c_did_modify(GLMContext c, Buffer *b, size_t o, size_t l) { [RENDERER(c) extBufDidModify: b offset: o length: l]; }
static void c_release(GLMContext c, Buffer *b) { [RENDERER(c) extBufRelease: b]; }
static bool c_begin_draw(GLMContext c) { return [RENDERER(c) extBufBeginDraw]; }
static void c_end_draw(GLMContext c) { [RENDERER(c) extBufEndDraw]; }

static void c_draw_arrays(GLMContext c, int prim, GLint first, GLsizei count, GLsizei inst, GLuint base_inst)
{
    [RENDERER(c) extBufDrawArrays: prim first: first count: count instances: inst baseInstance: base_inst];
}

static void c_draw_indexed(GLMContext c, int prim, GLenum type, Buffer *ibuf, size_t offset, const void *cpu,
                           GLsizei count, GLsizei inst, GLint base_vertex, GLuint base_inst)
{
    [RENDERER(c) extBufDrawIndexed: prim type: type buffer: ibuf offset: offset cpu: cpu count: count
                         instances: inst baseVertex: base_vertex baseInstance: base_inst];
}

static void c_draw_indirect(GLMContext c, int prim, Buffer *indirect, size_t offset, GLenum type, Buffer *ibuf)
{
    [RENDERER(c) extBufDrawIndirect: prim buffer: indirect offset: offset type: type index: ibuf];
}

static bool c_dispatch(GLMContext c, GLuint x, GLuint y, GLuint z, Buffer *indirect, size_t offset)
{
    return [RENDERER(c) extBufDispatchX: x y: y z: z indirect: indirect offset: offset];
}

static void c_copy(GLMContext c, Buffer *s, Buffer *d, size_t so, size_t dof, size_t size)
{
    [RENDERER(c) extBufCopy: s to: d srcOffset: so dstOffset: dof size: size];
}

static void c_fence(GLMContext c, Sync *s) { [RENDERER(c) extBufFence: s]; }
static GLenum c_client_wait(GLMContext c, Sync *s, bool f, GLuint64 t) { return [RENDERER(c) extBufClientWait: s flush: f timeout: t]; }
static void c_server_wait(GLMContext c, Sync *s) { [RENDERER(c) extBufServerWait: s]; }
static bool c_is_signaled(GLMContext c, Sync *s) { return [RENDERER(c) extBufSignaled: s]; }
static void c_delete_fence(GLMContext c, Sync *s) { [RENDERER(c) extBufDeleteFence: s]; }

/* legacy mtl_funcs entry points routed through the new paths */
static void c_mtl_flush(GLMContext c, bool finish)
{
    MGLRenderer *r = RENDERER(c);
    uint64_t serial = [r extBufSerial];

    [r extBufFlush];

    if (finish)
        [r extBufWaitSerial: serial];
}

static void c_mtl_delete_obj(GLMContext c, void *obj)
{
    // command buffers retain the resources they use; no flush required
    if (obj)
        CFBridgingRelease(obj);
}

static void c_mtl_buffer_sub_data(GLMContext c, Buffer *buf, size_t offset, size_t size, const void *ptr)
{
    if (!buf->data.buffer_data || !ptr)
        return;

    mgl_buffers_prepare_cpu_write(c, buf, offset == 0 && (GLsizeiptr)size == buf->size);
    memcpy((char *)buf->data.buffer_data + offset, ptr, size);
    [RENDERER(c) extBufDidModify: buf offset: offset length: size];
}

static void *c_mtl_map_unmap(GLMContext c, Buffer *buf, size_t offset, size_t size, GLenum access, bool map)
{
    if (map)
    {
        mgl_buffers_prepare_cpu_read(c, buf);
        mgl_buffers_prepare_cpu_write(c, buf, false);
        return buf->data.buffer_data ? (char *)buf->data.buffer_data + offset : NULL;
    }

    [RENDERER(c) extBufDidModify: buf offset: offset length: size];
    return NULL;
}

static void c_mtl_flush_range(GLMContext c, Buffer *buf, GLintptr offset, GLsizeiptr length)
{
    [RENDERER(c) extBufDidModify: buf offset: (size_t)offset length: (size_t)length];
}

static void c_mtl_get_sync(GLMContext c, Sync *s) { [RENDERER(c) extBufFence: s]; }

static void c_mtl_wait_for_sync(GLMContext c, Sync *s)
{
    [RENDERER(c) extBufClientWait: s flush: true timeout: GL_TIMEOUT_IGNORED];
}

static void c_mtl_dispatch(GLMContext c, GLuint x, GLuint y, GLuint z)
{
    [RENDERER(c) extBufDispatchX: x y: y z: z indirect: NULL offset: 0];
}

static void c_mtl_dispatch_indirect(GLMContext c, GLintptr indirect)
{
    Buffer *b = c->state.buffers[_DISPATCH_INDIRECT_BUFFER];
    if (b)
        [RENDERER(c) extBufDispatchX: 0 y: 0 z: 0 indirect: b offset: (size_t)indirect];
}

/* raw draw entry points behind mtl_funcs.mtlDraw*: the GL entry points (draw_buffers.c) have
 * already validated and run the pre-draw hook; the transform feedback capture wrappers call
 * these after capturing */
#define RAW(...) mgl_buffers_raw_draw(__VA_ARGS__)
static void c_mtl_draw_arrays(GLMContext c, GLenum mode, GLint first, GLsizei count)
{ RAW(c, mode, false, 0, 0, first, count, 1, 0, 0); }
static void c_mtl_draw_arrays_instanced(GLMContext c, GLenum mode, GLint first, GLsizei count, GLsizei inst)
{ RAW(c, mode, false, 0, 0, first, count, inst, 0, 0); }
static void c_mtl_draw_arrays_bi(GLMContext c, GLenum mode, GLint first, GLsizei count, GLsizei inst, GLuint bi)
{ RAW(c, mode, false, 0, 0, first, count, inst, 0, bi); }
static void c_mtl_draw_elements(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, 1, 0, 0); }
static void c_mtl_draw_range_elements(GLMContext c, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, 1, 0, 0); }
static void c_mtl_draw_elements_instanced(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei inst)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, inst, 0, 0); }
static void c_mtl_draw_elements_bv(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint bv)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, 1, bv, 0); }
static void c_mtl_draw_range_elements_bv(GLMContext c, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint bv)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, 1, bv, 0); }
static void c_mtl_draw_elements_inst_bv(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei inst, GLint bv)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, inst, bv, 0); }
static void c_mtl_draw_elements_inst_bi(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei inst, GLuint bi)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, inst, 0, bi); }
static void c_mtl_draw_elements_inst_bv_bi(GLMContext c, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei inst, GLint bv, GLuint bi)
{ RAW(c, mode, true, type, (size_t)(uintptr_t)indices, 0, count, inst, bv, bi); }
#undef RAW

- (void) bindExtBuffersFuncsToGLMContext: (GLMContext) glm_ctx
{
    MGLDrawPriv *p = (MGLDrawPriv *)calloc(1, sizeof(MGLDrawPriv));
    assert(p);

    p->pub.current_serial = c_current_serial;
    p->pub.completed_serial = c_completed_serial;
    p->pub.flush = c_flush;
    p->pub.wait_serial = c_wait_serial;
    p->pub.make_cpu_visible = c_make_cpu_visible;
    p->pub.barrier = c_barrier;
    p->pub.ensure_buffer = c_ensure;
    p->pub.rename_buffer = c_rename;
    p->pub.did_modify = c_did_modify;
    p->pub.release_buffer = c_release;
    p->pub.begin_draw = c_begin_draw;
    p->pub.draw_arrays = c_draw_arrays;
    p->pub.draw_indexed = c_draw_indexed;
    p->pub.draw_indirect = c_draw_indirect;
    p->pub.end_draw = c_end_draw;
    p->pub.dispatch = c_dispatch;
    p->pub.copy_buffer = c_copy;
    p->pub.fence = c_fence;
    p->pub.client_wait = c_client_wait;
    p->pub.server_wait = c_server_wait;
    p->pub.is_signaled = c_is_signaled;
    p->pub.delete_fence = c_delete_fence;

    glm_ctx->mtl_funcs.ext_buffers = &p->pub;

    // route the legacy buffer / sync / compute / draw hooks through this category
    glm_ctx->mtl_funcs.mtlFlush = c_mtl_flush;
    glm_ctx->mtl_funcs.mtlDeleteMTLObj = c_mtl_delete_obj;
    glm_ctx->mtl_funcs.mtlBufferSubData = c_mtl_buffer_sub_data;
    glm_ctx->mtl_funcs.mtlMapUnmapBuffer = c_mtl_map_unmap;
    glm_ctx->mtl_funcs.mtlFlushBufferRange = c_mtl_flush_range;
    glm_ctx->mtl_funcs.mtlGetSync = c_mtl_get_sync;
    glm_ctx->mtl_funcs.mtlWaitForSync = c_mtl_wait_for_sync;
    glm_ctx->mtl_funcs.mtlDispatchCompute = c_mtl_dispatch;
    glm_ctx->mtl_funcs.mtlDispatchComputeIndirect = c_mtl_dispatch_indirect;
    // raw draws: when the xfb domain already wrapped mtl_funcs, install below its wrappers
    struct MGLExtMetal_xfb *xfb = glm_ctx->mtl_funcs.ext_xfb;
    struct GLMMetalFuncs *draw_funcs = (xfb && xfb->orig.mtlDrawArrays) ? &xfb->orig : &glm_ctx->mtl_funcs;

    draw_funcs->mtlDrawArrays = c_mtl_draw_arrays;
    draw_funcs->mtlDrawArraysInstanced = c_mtl_draw_arrays_instanced;
    draw_funcs->mtlDrawArraysInstancedBaseInstance = c_mtl_draw_arrays_bi;
    draw_funcs->mtlDrawElements = c_mtl_draw_elements;
    draw_funcs->mtlDrawRangeElements = c_mtl_draw_range_elements;
    draw_funcs->mtlDrawElementsInstanced = c_mtl_draw_elements_instanced;
    draw_funcs->mtlDrawElementsBaseVertex = c_mtl_draw_elements_bv;
    draw_funcs->mtlDrawRangeElementsBaseVertex = c_mtl_draw_range_elements_bv;
    draw_funcs->mtlDrawElementsInstancedBaseVertex = c_mtl_draw_elements_inst_bv;
    draw_funcs->mtlDrawElementsInstancedBaseInstance = c_mtl_draw_elements_inst_bi;
    draw_funcs->mtlDrawElementsInstancedBaseVertexBaseInstance = c_mtl_draw_elements_inst_bv_bi;
}

@end
