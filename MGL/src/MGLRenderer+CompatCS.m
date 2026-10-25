/*
 * MGLRenderer+CompatCS.m — compat domain: single encoder (CS) worker thread.
 *
 * Design (DXVK CS-thread + Apple GL "ACC" accumulator, see agents/compat_report.md
 * session 10): the GL thread never calls Metal on the render encoder while the
 * worker is behind.  newRenderEncoder() installs MGLCSEncoderProxy as
 * _currentRenderEncoder; every encoder method the shared code calls is recorded
 * as a compact packet in a single-producer/single-consumer ring (objects
 * CFRetained into the packet, byte payloads copied into an arena).  One worker
 * thread replays the packets against the real encoder in order, so the Metal
 * command stream is exactly the GL command stream.
 *
 * Ordering is preserved because
 *   - all encoder mutation flows through the proxy (recorded, in order),
 *   - buffer writes already wait/rename through the GPU serial tracking that
 *     extBufEndDraw stamps at record time on the GL thread,
 *   - and every point where the GL thread itself must touch Metal directly
 *     (encoder creation/teardown, new command buffer, commit, compute/blit
 *     encoders, geometry/tess passes) drains first: compatCSDrain waits until
 *     the worker executed every published packet.
 *
 * Enabled with MGL_ENCODER_THREAD=1 (default off; MGL_CS_THREAD is an alias).
 * One worker thread per renderer, no pools.  Any worker-side exception flips
 * the state to `faulted`: the proxy then drains and calls through directly,
 * i.e. the failure mode is the previous (synchronous) behaviour.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"

#import <objc/runtime.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* packets */

enum {
    CSOP_NOP = 0,
    CSOP_SWITCH,            /* obj = real encoder (worker owns the ref)      */
    CSOP_END,               /* endEncoding + release worker's encoder        */
    CSOP_PSO,
    CSOP_VBUF, CSOP_VBUFOFF, CSOP_VBYTES,
    CSOP_FBUF, CSOP_FBUFOFF, CSOP_FBYTES,
    CSOP_VTEX, CSOP_FTEX,
    CSOP_VSAMP, CSOP_FSAMP,
    CSOP_VIEWPORT, CSOP_SCISSOR,
    CSOP_VIEWPORTS, CSOP_SCISSORS,
    CSOP_CULL, CSOP_WIND, CSOP_FILL, CSOP_CLIPMODE,
    CSOP_DEPTHBIAS, CSOP_DSSTATE, CSOP_STENCILREF, CSOP_STENCILREF2,
    CSOP_BLENDCOLOR, CSOP_VISRESULT,
    CSOP_DRAW, CSOP_DRAW_INST, CSOP_DRAW_BASE,
    CSOP_DRAWI, CSOP_DRAWI_INST, CSOP_DRAWI_BASE,
    CSOP_DRAW_IND, CSOP_DRAWI_IND,
    CSOP_BARRIER, CSOP_USERES, CSOP_USERES_STAGES,
    CSOP_TESSFACT, CSOP_TESSSCALE
};

typedef struct {
    uint8_t     op;
    uint8_t     stage;      /* 0 = vertex, 1 = fragment (buffer binds) */
    uint16_t    idx;
    uint32_t    a, b;
    uint64_t    c;
    void       *obj;        /* +1 CFRetained object, or arena pointer   */
    void       *obj2;       /* +1 object or 64-bit scalar               */
    float       f0, f1, f2, f3;
} CSPacket;

#define CS_RING_BITS    14                      /* 16384 packets */
#define CS_RING_CAP     (1u << CS_RING_BITS)
#define CS_RING_MASK    (CS_RING_CAP - 1)
#define CS_ARENA_BYTES  (512 * 1024)

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;                    /* worker sleeps here */
    pthread_cond_t  cv_done;                    /* draining producers sleep here */

    _Atomic uint64_t written;                   /* published by the GL thread */
    _Atomic uint64_t executed;                  /* executed by the worker     */
    _Atomic uint32_t idle;                      /* worker parked              */
    _Atomic uint32_t waiters;                   /* producers in drain         */
    _Atomic int32_t  exit;
    _Atomic int32_t  faulted;

    uint64_t    head;                           /* GL thread only */
    CSPacket   *ring;
    uint8_t    *arena;
    size_t      arena_used;

    void       *cur_enc;                        /* worker only: +1 real encoder */
    pthread_t   th;
    bool        th_valid;
    bool        started;

    /* stats (GL thread only) */
    uint64_t    drains, packets, waits;
    int         dbg;                                   /* MGL_CS_DEBUG: teardown stats */
} MGLCSState;

/* ------------------------------------------------------------------ */
/* state helpers (single producer = the GL thread, single consumer) */

static void cs_kick(MGLCSState *s)
{
    pthread_mutex_lock(&s->mu);
    pthread_cond_signal(&s->cv_work);
    pthread_mutex_unlock(&s->mu);
}

/* wait until the worker executed every published packet */
static void cs_drain(MGLCSState *s)
{
    uint64_t w = atomic_load_explicit(&s->written, memory_order_relaxed);

    if (atomic_load_explicit(&s->executed, memory_order_acquire) >= w)
    {
        /* worker idle and everything executed: the byte arena is free */
        if (s->head == w)
            s->arena_used = 0;
        return;                                 /* fast path: worker idle */
    }

    s->drains++;

    /* short spin: the worker is usually only a few packets behind */
    for (int i = 0; i < 64; i++)
    {
        if (atomic_load_explicit(&s->executed, memory_order_acquire) >= w)
            return;
        if ((i & 7) == 7)
            sched_yield();
        else
            __builtin_arm_yield();
    }

    s->waits++;
    pthread_mutex_lock(&s->mu);
    atomic_fetch_add(&s->waiters, 1);
    while (atomic_load_explicit(&s->executed, memory_order_acquire) < w)
    {
        if (atomic_load_explicit(&s->exit, memory_order_relaxed))
            break;                              /* teardown: worker is gone */
        /* signal under the same mutex the worker parks on: this closes the
         * lost-wakeup window (a kick before the worker parked does nothing;
         * under the mutex the worker is either parked — the signal wakes it —
         * or it will re-check `written` before parking) */
        pthread_cond_signal(&s->cv_work);
        pthread_cond_wait(&s->cv_done, &s->mu);
    }
    atomic_fetch_sub(&s->waiters, 1);
    pthread_mutex_unlock(&s->mu);

    /* fully drained: the arena can be reused (every byte payload was copied
     * into its encoder by now) */
    if (atomic_load_explicit(&s->executed, memory_order_acquire) >= atomic_load_explicit(&s->written, memory_order_relaxed))
        s->arena_used = 0;

    if (s->dbg && (s->drains % 600) == 0)
        fprintf(stderr, "MGL CS: packets=%llu drains=%llu sleeps=%llu spins-ok=%llu\n",
                (unsigned long long)s->packets, (unsigned long long)s->drains,
                (unsigned long long)s->waits,
                (unsigned long long)(s->drains - s->waits));
}

static void *cs_arena_alloc(MGLCSState *s, size_t len)
{
    if (s->arena_used + len > CS_ARENA_BYTES)
    {
        cs_drain(s);
        if (s->arena_used + len > CS_ARENA_BYTES)
            return NULL;
    }

    void *p = s->arena + s->arena_used;
    s->arena_used += len;
    return p;
}

/* release any +1 references a packet holds (call only when the packet was
 * NOT consumed by cs_push; scalar slots are not references) */
static void cs_release_pkt(CSPacket *p)
{
    switch (p->op)
    {
        case CSOP_PSO: case CSOP_VBUF: case CSOP_FBUF:
        case CSOP_VTEX: case CSOP_FTEX:
        case CSOP_VSAMP: case CSOP_FSAMP:
        case CSOP_DSSTATE:
        case CSOP_USERES: case CSOP_USERES_STAGES:
        case CSOP_DRAWI: case CSOP_DRAWI_INST: case CSOP_DRAWI_BASE:
        case CSOP_DRAW_IND: case CSOP_TESSFACT:
            if (p->obj) CFRelease(p->obj);
            break;
        case CSOP_DRAWI_IND:                        /* obj and obj2 are both refs */
            if (p->obj) CFRelease(p->obj);
            if (p->obj2) CFRelease(p->obj2);
            break;
        default:
            break;                              /* scalars / arena pointers */
    }
}

/* record a packet; false = ring full after drain or faulted (caller falls
 * back to direct emission, after draining) */
static bool cs_push(MGLCSState *s, CSPacket p)
{
    if (atomic_load_explicit(&s->faulted, memory_order_relaxed))
        return false;

    uint64_t w = s->head;

    if (w - atomic_load_explicit(&s->executed, memory_order_acquire) >= CS_RING_CAP)
    {
        cs_drain(s);
        if (w - atomic_load_explicit(&s->executed, memory_order_acquire) >= CS_RING_CAP)
            return false;
    }

    s->ring[w & CS_RING_MASK] = p;
    s->head = w + 1;
    s->packets++;
    atomic_store_explicit(&s->written, w + 1, memory_order_release);

    if (atomic_load_explicit(&s->idle, memory_order_relaxed))
        cs_kick(s);

    return true;
}

/* ------------------------------------------------------------------ */
/* worker */

static void cs_exec(MGLCSState *s, CSPacket *p)
{
    id enc = (__bridge id<MTLRenderCommandEncoder>)s->cur_enc;

    if (!enc && p->op != CSOP_SWITCH && p->op != CSOP_END)
        return;                                 /* encoder already ended */

    switch (p->op)
    {
        case CSOP_SWITCH:
            if (s->cur_enc) CFRelease(s->cur_enc);
            s->cur_enc = p->obj;                /* takes the packet's +1 */
            return;

        case CSOP_END:
            if (s->cur_enc)
            {
                [(__bridge id<MTLRenderCommandEncoder>)s->cur_enc endEncoding];
                CFRelease(s->cur_enc);
                s->cur_enc = NULL;
            }
            return;

        case CSOP_PSO:   { id o = (__bridge_transfer id)p->obj; [enc setRenderPipelineState:o]; return; }
        case CSOP_VBUF:  { id o = (__bridge_transfer id)p->obj; [enc setVertexBuffer:o offset:p->c atIndex:p->idx]; return; }
        case CSOP_FBUF:  { id o = (__bridge_transfer id)p->obj; [enc setFragmentBuffer:o offset:p->c atIndex:p->idx]; return; }
        case CSOP_VTEX:  { id o = (__bridge_transfer id)p->obj; [enc setVertexTexture:o atIndex:p->idx]; return; }
        case CSOP_FTEX:  { id o = (__bridge_transfer id)p->obj; [enc setFragmentTexture:o atIndex:p->idx]; return; }
        case CSOP_VSAMP: { id o = (__bridge_transfer id)p->obj; [enc setVertexSamplerState:o atIndex:p->idx]; return; }
        case CSOP_FSAMP: { id o = (__bridge_transfer id)p->obj; [enc setFragmentSamplerState:o atIndex:p->idx]; return; }
        case CSOP_DSSTATE:{ id o = (__bridge_transfer id)p->obj; [enc setDepthStencilState:o]; return; }

        case CSOP_VBUFOFF: [enc setVertexBufferOffset:p->c atIndex:p->idx]; return;
        case CSOP_FBUFOFF: [enc setFragmentBufferOffset:p->c atIndex:p->idx]; return;
        case CSOP_VBYTES:  [enc setVertexBytes:p->obj length:p->a atIndex:p->idx]; return;
        case CSOP_FBYTES:  [enc setFragmentBytes:p->obj length:p->a atIndex:p->idx]; return;

        case CSOP_VIEWPORT:
        {
            MTLViewport v;
            v.originX = p->f0; v.originY = p->f1; v.width = p->f2; v.height = p->f3;
            memcpy(&v.znear, &p->c, sizeof(double));
            memcpy(&v.zfar, &p->obj, sizeof(double));
            [enc setViewport:v];
            return;
        }
        case CSOP_VIEWPORTS:                     /* array snapshot in the arena */
            if (@available(macOS 10.13, *))
                [enc setViewports:(const MTLViewport *)p->obj count:(NSUInteger)p->c];
            return;
        case CSOP_SCISSOR:
        {
            MTLScissorRect r = { (NSUInteger)p->a, (NSUInteger)p->b,
                                 (NSUInteger)(uintptr_t)p->obj, (NSUInteger)(uintptr_t)p->obj2 };
            [enc setScissorRect:r];
            return;
        }
        case CSOP_SCISSORS:                      /* array snapshot in the arena */
            if (@available(macOS 10.13, *))
                [enc setScissorRects:(const MTLScissorRect *)p->obj count:(NSUInteger)p->c];
            return;
        case CSOP_CULL:     [enc setCullMode:(MTLCullMode)p->a]; return;
        case CSOP_WIND:     [enc setFrontFacingWinding:(MTLWinding)p->a]; return;
        case CSOP_FILL:     [enc setTriangleFillMode:(MTLTriangleFillMode)p->a]; return;
        case CSOP_CLIPMODE: [enc setDepthClipMode:(MTLDepthClipMode)p->a]; return;
        case CSOP_DEPTHBIAS:[enc setDepthBias:p->f0 slopeScale:p->f1 clamp:p->f2]; return;
        case CSOP_STENCILREF:  [enc setStencilReferenceValue:p->a]; return;
        case CSOP_STENCILREF2: [enc setStencilFrontReferenceValue:p->a backReferenceValue:p->b]; return;
        case CSOP_BLENDCOLOR:  [enc setBlendColorRed:p->f0 green:p->f1 blue:p->f2 alpha:p->f3]; return;
        case CSOP_VISRESULT:   [enc setVisibilityResultMode:(MTLVisibilityResultMode)p->a offset:p->c]; return;

        case CSOP_DRAW:  [enc drawPrimitives:(MTLPrimitiveType)p->a vertexStart:(NSUInteger)p->obj vertexCount:p->c]; return;
        case CSOP_DRAW_INST:
            [enc drawPrimitives:(MTLPrimitiveType)p->a vertexStart:(NSUInteger)p->obj vertexCount:p->c
                    instanceCount:p->b];
            return;
        case CSOP_DRAW_BASE:
            [enc drawPrimitives:(MTLPrimitiveType)p->a vertexStart:(NSUInteger)p->obj vertexCount:p->c
                    instanceCount:p->b baseInstance:(NSUInteger)p->obj2];
            return;
        case CSOP_DRAWI:
        {
            id o = (__bridge_transfer id)p->obj;
            [enc drawIndexedPrimitives:(MTLPrimitiveType)p->a indexCount:p->b
                             indexType:(MTLIndexType)p->stage indexBuffer:o
                       indexBufferOffset:p->c];
            return;
        }
        case CSOP_DRAWI_INST:
        {
            id o = (__bridge_transfer id)p->obj;
            [enc drawIndexedPrimitives:(MTLPrimitiveType)p->a indexCount:p->b
                             indexType:(MTLIndexType)p->stage indexBuffer:o
                       indexBufferOffset:p->c instanceCount:(NSUInteger)p->obj2];
            return;
        }
        case CSOP_DRAWI_BASE:
        {
            id o = (__bridge_transfer id)p->obj;
            NSInteger bv;  NSUInteger bi;
            memcpy(&bv, &p->f0, sizeof bv);
            memcpy(&bi, &p->f1, sizeof bi);
            [enc drawIndexedPrimitives:(MTLPrimitiveType)p->a indexCount:p->b
                             indexType:(MTLIndexType)p->stage indexBuffer:o
                       indexBufferOffset:p->c instanceCount:(NSUInteger)p->obj2
                          baseVertex:bv baseInstance:bi];
            return;
        }
        case CSOP_DRAW_IND:
        {
            id o = (__bridge_transfer id)p->obj;
            [enc drawPrimitives:(MTLPrimitiveType)p->a indirectBuffer:o indirectBufferOffset:p->c];
            return;
        }
        case CSOP_DRAWI_IND:
        {
            id ind = (__bridge_transfer id)p->obj;
            id ib  = (__bridge_transfer id)p->obj2;
            [enc drawIndexedPrimitives:(MTLPrimitiveType)p->a
                             indexType:(MTLIndexType)p->stage
                         indexBuffer:ib indexBufferOffset:p->b
                     indirectBuffer:ind indirectBufferOffset:p->c];
            return;
        }

        case CSOP_BARRIER:
            if (@available(macOS 10.14, *))
                [enc memoryBarrierWithScope:(MTLBarrierScope)p->a
                                 afterStages:(MTLRenderStages)p->b beforeStages:(MTLRenderStages)p->c];
            return;
        case CSOP_USERES: { id o = (__bridge_transfer id)p->obj; [enc useResource:o usage:(MTLResourceUsage)p->a]; return; }
        case CSOP_USERES_STAGES: { id o = (__bridge_transfer id)p->obj; [enc useResource:o usage:(MTLResourceUsage)p->a stages:(MTLRenderStages)p->b]; return; }
        case CSOP_TESSFACT:
        {
            if (@available(macOS 10.12, *))
            {
                id o = (__bridge_transfer id)p->obj;
                [enc setTessellationFactorBuffer:o offset:p->c instanceStride:p->idx];
            }
            return;
        }
        case CSOP_TESSSCALE:
            if (@available(macOS 10.12, *))
                [enc setTessellationFactorScale:p->f0];
            return;

        default:
            return;
    }
}

static void *cs_worker(void *arg)
{
    MGLCSState *s = (MGLCSState *)arg;
    uint64_t seen = 0;

    for (;;)
    {
        uint64_t w = atomic_load_explicit(&s->written, memory_order_acquire);

        if (w == seen)
        {
            pthread_mutex_lock(&s->mu);
            if (atomic_load_explicit(&s->written, memory_order_acquire) == seen &&
                !atomic_load_explicit(&s->exit, memory_order_relaxed))
            {
                atomic_store_explicit(&s->idle, 1, memory_order_relaxed);
                pthread_cond_wait(&s->cv_work, &s->mu);
                atomic_store_explicit(&s->idle, 0, memory_order_relaxed);
            }
            bool quit = atomic_load_explicit(&s->exit, memory_order_relaxed) != 0;
            pthread_mutex_unlock(&s->mu);

            if (quit && atomic_load_explicit(&s->written, memory_order_acquire) == seen)
                break;
            continue;
        }

        @try
        {
            while (seen < w)
            {
                cs_exec(s, &s->ring[seen & CS_RING_MASK]);
                seen++;
            }
        }
        @catch (NSException *ex)
        {
            /* never leave the GL thread blocked and never encode further on a
             * possibly-broken encoder: retire the batch and disable recording */
            NSLog(@"MGL CS: worker exception %@ — falling back to direct encoding", ex);
            atomic_store_explicit(&s->faulted, 1, memory_order_relaxed);
            seen = w;
        }

        atomic_store_explicit(&s->executed, seen, memory_order_release);

        /* always signal under the mutex: reading `waiters` lock-free can miss a
         * producer that just incremented it (stale 0) and park it forever */
        pthread_mutex_lock(&s->mu);
        pthread_cond_broadcast(&s->cv_done);
        pthread_mutex_unlock(&s->mu);
    }

    if (s->cur_enc) { CFRelease(s->cur_enc); s->cur_enc = NULL; }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* state object (associated on the renderer: categories cannot add ivars) */

@interface MGLCSBox : NSObject
{ @public MGLCSState st; }
@end

@implementation MGLCSBox
- (instancetype) init
{
    self = [super init];
    if (!self) return nil;

    memset(&st, 0, sizeof st);
    pthread_mutex_init(&st.mu, NULL);
    pthread_cond_init(&st.cv_work, NULL);
    pthread_cond_init(&st.cv_done, NULL);
    st.ring = (CSPacket *)calloc(CS_RING_CAP, sizeof(CSPacket));
    st.arena = (uint8_t *)malloc(CS_ARENA_BYTES);
    atomic_init(&st.written, 0);
    atomic_init(&st.executed, 0);
    atomic_init(&st.idle, 0);
    atomic_init(&st.waiters, 0);
    atomic_init(&st.exit, 0);
    atomic_init(&st.faulted, 0);
    st.dbg = getenv("MGL_PROFILE") != NULL;

    if (!st.ring || !st.arena)
    {
        atomic_store_explicit(&st.faulted, 1, memory_order_relaxed);
        return self;
    }

    if (pthread_create(&st.th, NULL, cs_worker, &st) == 0)
        st.th_valid = st.started = true;

    return self;
}

- (void) dealloc
{
    if (st.th_valid)
    {
        pthread_mutex_lock(&st.mu);
        atomic_store_explicit(&st.exit, 1, memory_order_relaxed);
        pthread_cond_broadcast(&st.cv_work);
        pthread_cond_broadcast(&st.cv_done);
        pthread_mutex_unlock(&st.mu);
        pthread_join(st.th, NULL);          /* the box outlives the worker here */
    }
    free(st.ring);
    free(st.arena);
    pthread_mutex_destroy(&st.mu);
    pthread_cond_destroy(&st.cv_work);
    pthread_cond_destroy(&st.cv_done);
}
@end

static const char kCompatCSKey;

static int cs_env(void)
{
    static int v = -1;
    if (v < 0)
    {
        const char *e = getenv("MGL_ENCODER_THREAD");
        if (!e || !*e) e = getenv("MGL_CS_THREAD");
        v = (e && e[0] != '0' && strcmp(e, "no") != 0) ? 1 : 0;
    }
    return v;
}

/* ------------------------------------------------------------------ */
/* recording proxy: _currentRenderEncoder while the encoder thread runs */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wprotocol"       /* rare methods go through forwarding */

@interface MGLCSEncoderProxy : NSObject <MTLRenderCommandEncoder>
@property (nonatomic, strong) id<MTLRenderCommandEncoder> real;
@property (nonatomic, strong) MGLCSBox *box;
/* NSObject's forwarding entry points are not visible in this translation unit
 * when only the Metal headers are imported */
- (NSMethodSignature *) methodSignatureForSelector: (SEL) sel;
- (void) forwardInvocation: (NSInvocation *) inv;
@end

@implementation MGLCSEncoderProxy

- (MGLCSState *) cs { return &_box->st; }

/* direct-emission fallback (also the entire behaviour once faulted) */
- (void) csFallback { cs_drain(self.cs); }

- (void) csRec: (CSPacket) p fallback: (void (^)()) direct
{
    if (!cs_push(self.cs, p))
    {
        cs_release_pkt(&p);
        cs_drain(self.cs);
        direct();
    }
}

#pragma mark MTLCommandEncoder

- (id<MTLDevice>) device          { return _real.device; }
- (NSString *) label             { return _real.label; }
- (void) setLabel: (NSString *) l { [self csFallback]; _real.label = l; }
- (NSUInteger) tileWidth         { return _real.tileWidth; }
- (NSUInteger) tileHeight        { return _real.tileHeight; }

- (void) endEncoding
{
    /* a pass boundary: record the END, then drain so the GL thread may create
     * the next encoder / a compute or blit encoder the moment this returns */
    CSPacket p = { .op = CSOP_END };
    [self csRec:p fallback:^{ [_real endEncoding]; }];
    cs_drain(self.cs);
}

- (void) pushDebugGroup: (NSString *) s    { [self csFallback]; [_real pushDebugGroup:s]; }
- (void) popDebugGroup                      { [self csFallback]; [_real popDebugGroup]; }
- (void) insertDebugSignpost: (NSString *) s { [self csFallback]; [_real insertDebugSignpost:s]; }

- (BOOL) respondsToSelector: (SEL) sel
{
    if ([super respondsToSelector:sel]) return YES;
    return [(id)_real respondsToSelector:sel];
}

- (NSMethodSignature *) methodSignatureForSelector: (SEL) sel
{
    NSMethodSignature *sig = [(id)_real methodSignatureForSelector:sel];
    return sig ? sig : [super methodSignatureForSelector:sel];
}

- (void) forwardInvocation: (NSInvocation *) inv
{
    [self csFallback];                        /* ordered before the direct call */
    [inv invokeWithTarget:_real];
}

#pragma mark pipeline / state

- (void) setRenderPipelineState: (id<MTLRenderPipelineState>) ps
{
    CSPacket p = { .op = CSOP_PSO, .obj = (__bridge_retained void *)ps };
    [self csRec:p fallback:^{ [_real setRenderPipelineState:ps]; }];
}

- (void) setViewport: (MTLViewport) v
{
    CSPacket p = { .op = CSOP_VIEWPORT, .f0 = (float)v.originX, .f1 = (float)v.originY,
                   .f2 = (float)v.width, .f3 = (float)v.height };
    memcpy(&p.c, &v.znear, sizeof(double));
    memcpy(&p.obj, &v.zfar, sizeof(double));
    [self csRec:p fallback:^{ [_real setViewport:v]; }];
}

/* rstateApplyEncoderState pushes all viewports in one call: snapshot the array
 * into the arena (Metal has no per-index setters) */
- (void) setViewports: (const MTLViewport *) vp count: (NSUInteger) n
{
    void *dst = cs_arena_alloc(self.cs, n * sizeof(MTLViewport));
    if (dst)
    {
        memcpy(dst, vp, n * sizeof(MTLViewport));
        CSPacket p = { .op = CSOP_VIEWPORTS, .obj = dst, .c = n };
        if (cs_push(self.cs, p)) return;
    }
    [self csFallback];
    [_real setViewports:vp count:n];
}

- (void) setScissorRect: (MTLScissorRect) r
{
    CSPacket p = { .op = CSOP_SCISSOR, .idx = 0,
                   .a = (uint32_t)r.x, .b = (uint32_t)r.y };
    p.obj  = (void *)(uintptr_t)r.width;
    p.obj2 = (void *)(uintptr_t)r.height;
    [self csRec:p fallback:^{ [_real setScissorRect:r]; }];
}

- (void) setScissorRects: (const MTLScissorRect *) rs count: (NSUInteger) n
{
    void *dst = cs_arena_alloc(self.cs, n * sizeof(MTLScissorRect));
    if (dst)
    {
        memcpy(dst, rs, n * sizeof(MTLScissorRect));
        CSPacket p = { .op = CSOP_SCISSORS, .obj = dst, .c = n };
        if (cs_push(self.cs, p)) return;
    }
    [self csFallback];
    [_real setScissorRects:rs count:n];
}

- (void) setCullMode: (MTLCullMode) m
{
    CSPacket p = { .op = CSOP_CULL, .a = (uint32_t)m };
    [self csRec:p fallback:^{ [_real setCullMode:m]; }];
}

- (void) setFrontFacingWinding: (MTLWinding) w
{
    CSPacket p = { .op = CSOP_WIND, .a = (uint32_t)w };
    [self csRec:p fallback:^{ [_real setFrontFacingWinding:w]; }];
}

- (void) setTriangleFillMode: (MTLTriangleFillMode) m
{
    CSPacket p = { .op = CSOP_FILL, .a = (uint32_t)m };
    [self csRec:p fallback:^{ [_real setTriangleFillMode:m]; }];
}

- (void) setDepthClipMode: (MTLDepthClipMode) m
{
    CSPacket p = { .op = CSOP_CLIPMODE, .a = (uint32_t)m };
    [self csRec:p fallback:^{ [_real setDepthClipMode:m]; }];
}

- (void) setDepthBias: (float) b slopeScale: (float) ss clamp: (float) c
{
    CSPacket p = { .op = CSOP_DEPTHBIAS, .f0 = b, .f1 = ss, .f2 = c };
    [self csRec:p fallback:^{ [_real setDepthBias:b slopeScale:ss clamp:c]; }];
}

- (void) setDepthStencilState: (id<MTLDepthStencilState>) ds
{
    CSPacket p = { .op = CSOP_DSSTATE, .obj = (__bridge_retained void *)ds };
    [self csRec:p fallback:^{ [_real setDepthStencilState:ds]; }];
}

- (void) setStencilReferenceValue: (uint32_t) r
{
    CSPacket p = { .op = CSOP_STENCILREF, .a = r };
    [self csRec:p fallback:^{ [_real setStencilReferenceValue:r]; }];
}

- (void) setStencilFrontReferenceValue: (uint32_t) f backReferenceValue: (uint32_t) b
{
    CSPacket p = { .op = CSOP_STENCILREF2, .a = f, .b = b };
    [self csRec:p fallback:^{ [_real setStencilFrontReferenceValue:f backReferenceValue:b]; }];
}

- (void) setBlendColorRed: (float) r green: (float) g blue: (float) b alpha: (float) a
{
    CSPacket p = { .op = CSOP_BLENDCOLOR, .f0 = r, .f1 = g, .f2 = b, .f3 = a };
    [self csRec:p fallback:^{ [_real setBlendColorRed:r green:g blue:b alpha:a]; }];
}

- (void) setVisibilityResultMode: (MTLVisibilityResultMode) m offset: (NSUInteger) o
{
    CSPacket p = { .op = CSOP_VISRESULT, .a = (uint32_t)m, .c = o };
    [self csRec:p fallback:^{ [_real setVisibilityResultMode:m offset:o]; }];
}

#pragma mark buffers

- (void) setVertexBuffer: (id<MTLBuffer>) buf offset: (NSUInteger) o atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_VBUF, .idx = (uint16_t)i, .c = o, .obj = (__bridge_retained void *)buf };
    [self csRec:p fallback:^{ [_real setVertexBuffer:buf offset:o atIndex:i]; }];
}

- (void) setVertexBufferOffset: (NSUInteger) o atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_VBUFOFF, .idx = (uint16_t)i, .c = o };
    [self csRec:p fallback:^{ [_real setVertexBufferOffset:o atIndex:i]; }];
}

- (void) setVertexBytes: (const void *) bytes length: (NSUInteger) len atIndex: (NSUInteger) i
{
    void *dst = cs_arena_alloc(self.cs, len);
    if (dst)
    {
        memcpy(dst, bytes, len);
        CSPacket p = { .op = CSOP_VBYTES, .idx = (uint16_t)i, .a = (uint32_t)len, .obj = dst };
        if (cs_push(self.cs, p)) return;
    }
    [self csFallback];
    [_real setVertexBytes:bytes length:len atIndex:i];
}

- (void) setFragmentBuffer: (id<MTLBuffer>) buf offset: (NSUInteger) o atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_FBUF, .idx = (uint16_t)i, .c = o, .obj = (__bridge_retained void *)buf };
    [self csRec:p fallback:^{ [_real setFragmentBuffer:buf offset:o atIndex:i]; }];
}

- (void) setFragmentBufferOffset: (NSUInteger) o atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_FBUFOFF, .idx = (uint16_t)i, .c = o };
    [self csRec:p fallback:^{ [_real setFragmentBufferOffset:o atIndex:i]; }];
}

- (void) setFragmentBytes: (const void *) bytes length: (NSUInteger) len atIndex: (NSUInteger) i
{
    void *dst = cs_arena_alloc(self.cs, len);
    if (dst)
    {
        memcpy(dst, bytes, len);
        CSPacket p = { .op = CSOP_FBYTES, .idx = (uint16_t)i, .a = (uint32_t)len, .obj = dst };
        if (cs_push(self.cs, p)) return;
    }
    [self csFallback];
    [_real setFragmentBytes:bytes length:len atIndex:i];
}

#pragma mark textures / samplers

- (void) setVertexTexture: (id<MTLTexture>) t atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_VTEX, .idx = (uint16_t)i, .obj = (__bridge_retained void *)t };
    [self csRec:p fallback:^{ [_real setVertexTexture:t atIndex:i]; }];
}

- (void) setFragmentTexture: (id<MTLTexture>) t atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_FTEX, .idx = (uint16_t)i, .obj = (__bridge_retained void *)t };
    [self csRec:p fallback:^{ [_real setFragmentTexture:t atIndex:i]; }];
}

- (void) setVertexSamplerState: (id<MTLSamplerState>) s atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_VSAMP, .idx = (uint16_t)i, .obj = (__bridge_retained void *)s };
    [self csRec:p fallback:^{ [_real setVertexSamplerState:s atIndex:i]; }];
}

- (void) setFragmentSamplerState: (id<MTLSamplerState>) s atIndex: (NSUInteger) i
{
    CSPacket p = { .op = CSOP_FSAMP, .idx = (uint16_t)i, .obj = (__bridge_retained void *)s };
    [self csRec:p fallback:^{ [_real setFragmentSamplerState:s atIndex:i]; }];
}

#pragma mark draws

- (void) drawPrimitives: (MTLPrimitiveType) t vertexStart: (NSUInteger) s vertexCount: (NSUInteger) c
{
    CSPacket p = { .op = CSOP_DRAW, .a = t, .c = c, .obj = (void *)s };
    [self csRec:p fallback:^{ [_real drawPrimitives:t vertexStart:s vertexCount:c]; }];
}

- (void) drawPrimitives: (MTLPrimitiveType) t vertexStart: (NSUInteger) s vertexCount: (NSUInteger) c
          instanceCount: (NSUInteger) n
{
    CSPacket p = { .op = CSOP_DRAW_INST, .a = t, .c = c, .obj = (void *)s, .b = (uint32_t)n };
    [self csRec:p fallback:^{ [_real drawPrimitives:t vertexStart:s vertexCount:c instanceCount:n]; }];
}

- (void) drawPrimitives: (MTLPrimitiveType) t vertexStart: (NSUInteger) s vertexCount: (NSUInteger) c
          instanceCount: (NSUInteger) n baseInstance: (NSUInteger) bi
{
    CSPacket p = { .op = CSOP_DRAW_BASE, .a = t, .c = c, .obj = (void *)s, .b = (uint32_t)n,
                   .obj2 = (void *)bi };
    [self csRec:p fallback:^{ [_real drawPrimitives:t vertexStart:s vertexCount:c instanceCount:n
                                      baseInstance:bi]; }];
}

- (void) drawPrimitives: (MTLPrimitiveType) t indirectBuffer: (id<MTLBuffer>) ib
       indirectBufferOffset: (NSUInteger) o
{
    CSPacket p = { .op = CSOP_DRAW_IND, .a = t, .c = o, .obj = (__bridge_retained void *)ib };
    [self csRec:p fallback:^{ [_real drawPrimitives:t indirectBuffer:ib indirectBufferOffset:o]; }];
}

- (void) drawIndexedPrimitives: (MTLPrimitiveType) t indexCount: (NSUInteger) c
                     indexType: (MTLIndexType) it indexBuffer: (id<MTLBuffer>) ib
               indexBufferOffset: (NSUInteger) o
{
    CSPacket p = { .op = CSOP_DRAWI, .a = t, .b = (uint32_t)c, .stage = (uint8_t)it,
                   .c = o, .obj = (__bridge_retained void *)ib };
    [self csRec:p fallback:^{ [_real drawIndexedPrimitives:t indexCount:c indexType:it
                                            indexBuffer:ib indexBufferOffset:o]; }];
}

- (void) drawIndexedPrimitives: (MTLPrimitiveType) t indexCount: (NSUInteger) c
                     indexType: (MTLIndexType) it indexBuffer: (id<MTLBuffer>) ib
               indexBufferOffset: (NSUInteger) o instanceCount: (NSUInteger) n
{
    CSPacket p = { .op = CSOP_DRAWI_INST, .a = t, .b = (uint32_t)c, .stage = (uint8_t)it,
                   .c = o, .obj2 = (void *)n, .obj = (__bridge_retained void *)ib };
    if (!cs_push(self.cs, p))
    {
        cs_release_pkt(&p);
        [self csFallback];
        [_real drawIndexedPrimitives:t indexCount:c indexType:it
                                indexBuffer:ib indexBufferOffset:o instanceCount:n];
    }
}

- (void) drawIndexedPrimitives: (MTLPrimitiveType) t indexCount: (NSUInteger) c
                     indexType: (MTLIndexType) it indexBuffer: (id<MTLBuffer>) ib
               indexBufferOffset: (NSUInteger) o instanceCount: (NSUInteger) n
                    baseVertex: (NSInteger) bv baseInstance: (NSUInteger) bi
{
    CSPacket p = { .op = CSOP_DRAWI_BASE, .a = t, .b = (uint32_t)c, .stage = (uint8_t)it,
                   .c = o, .obj2 = (void *)n, .obj = (__bridge_retained void *)ib };
    memcpy(&p.f0, &bv, sizeof bv);             /* bit-exact int32 */
    memcpy(&p.f1, &bi, sizeof bi);
    if (!cs_push(self.cs, p))
    {
        cs_release_pkt(&p);
        [self csFallback];
        [_real drawIndexedPrimitives:t indexCount:c indexType:it indexBuffer:ib
                      indexBufferOffset:o instanceCount:n baseVertex:bv baseInstance:bi];
    }
}

- (void) drawIndexedPrimitives: (MTLPrimitiveType) t indexType: (MTLIndexType) it
                    indexBuffer: (id<MTLBuffer>) ib indexBufferOffset: (NSUInteger) io
                  indirectBuffer: (id<MTLBuffer>) ind indirectBufferOffset: (NSUInteger) o
{
    CSPacket p = { .op = CSOP_DRAWI_IND, .a = t, .stage = (uint8_t)it, .b = (uint32_t)io,
                   .c = o, .obj = (__bridge_retained void *)ind,
                   .obj2 = (__bridge_retained void *)ib };
    if (!cs_push(self.cs, p))
    {
        cs_release_pkt(&p);
        [self csFallback];
        [_real drawIndexedPrimitives:t indexType:it indexBuffer:ib indexBufferOffset:io
                       indirectBuffer:ind indirectBufferOffset:o];
    }
}

#pragma mark barriers / resources / tessellation

- (void) memoryBarrierWithScope: (MTLBarrierScope) sc afterStages: (MTLRenderStages) after
                    beforeStages: (MTLRenderStages) before
{
    CSPacket p = { .op = CSOP_BARRIER, .a = (uint32_t)sc, .b = (uint32_t)after, .c = (uint32_t)before };
    [self csRec:p fallback:^{ [_real memoryBarrierWithScope:sc afterStages:after beforeStages:before]; }];
}

- (void) useResource: (id<MTLResource>) r usage: (MTLResourceUsage) u
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    CSPacket p = { .op = CSOP_USERES, .a = (uint32_t)u, .obj = (__bridge_retained void *)r };
    [self csRec:p fallback:^{ [_real useResource:r usage:u]; }];
#pragma clang diagnostic pop
}

- (void) useResource: (id<MTLResource>) r usage: (MTLResourceUsage) u stages: (MTLRenderStages) st
{
    CSPacket p = { .op = CSOP_USERES_STAGES, .a = (uint32_t)u, .b = (uint32_t)st,
                   .obj = (__bridge_retained void *)r };
    [self csRec:p fallback:^{ [_real useResource:r usage:u stages:st]; }];
}

- (void) setTessellationFactorBuffer: (id<MTLBuffer>) b offset: (NSUInteger) o
                       instanceStride: (NSUInteger) is
{
    CSPacket p = { .op = CSOP_TESSFACT, .idx = (uint16_t)is, .c = o,
                   .obj = (__bridge_retained void *)b };
    [self csRec:p fallback:^{ [_real setTessellationFactorBuffer:b offset:o instanceStride:is]; }];
}

- (void) setTessellationFactorScale: (float) s
{
    CSPacket p = { .op = CSOP_TESSSCALE, .f0 = s };
    [self csRec:p fallback:^{ [_real setTessellationFactorScale:s]; }];
}

@end

#pragma clang diagnostic pop

/* ------------------------------------------------------------------ */
/* renderer category */

@implementation MGLRenderer (CompatCS)

- (MGLCSBox *) compatCSBox
{
    MGLCSBox *box = objc_getAssociatedObject(self, &kCompatCSKey);
    return box;
}

- (bool) compatCSOn
{
    if (cs_env() != 1)
        return false;
    return true;
}

/* wrap the freshly created real encoder: the GL thread records, the worker
 * encodes.  Drains first so the previous encoder's END (if any) is done
 * before Metal sees a new encoder on the command buffer. */
- (id<MTLRenderCommandEncoder>) compatCSEncoderProxyFor: (id<MTLRenderCommandEncoder>) real
{
    if (!real)
        return nil;

    if (![self compatCSOn])
        return real;

    MGLCSBox *box = objc_getAssociatedObject(self, &kCompatCSKey);
    if (!box)
    {
        box = [[MGLCSBox alloc] init];
        objc_setAssociatedObject(self, &kCompatCSKey, box, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        if (!box->st.started)
            NSLog(@"MGL CS: encoder thread could not start — direct encoding");
    }

    if (!box->st.started || atomic_load_explicit(&box->st.faulted, memory_order_relaxed))
        return real;

    cs_drain(&box->st);

    MGLCSEncoderProxy *proxy = [[MGLCSEncoderProxy alloc] init];
    proxy.real = real;
    proxy.box = box;

    CSPacket p = { .op = CSOP_SWITCH, .obj = (__bridge_retained void *)real };
    if (!cs_push(&box->st, p))
    {
        CFRelease(p.obj);
        return real;
    }

    return (id<MTLRenderCommandEncoder>)proxy;
}

/* wait for the worker to finish every recorded packet; no-op when idle */
- (void) compatCSDrain
{
    MGLCSBox *box = objc_getAssociatedObject(self, &kCompatCSKey);
    if (box && box->st.started)
        cs_drain(&box->st);
}

/* renderer teardown: drain, stop the worker, drop the state */
- (void) compatCSStop
{
    MGLCSBox *box = objc_getAssociatedObject(self, &kCompatCSKey);
    if (!box)
        return;

    if (box->st.started && !atomic_load_explicit(&box->st.faulted, memory_order_relaxed))
    {
        static int prof = -1;
        if (prof < 0) prof = getenv("MGL_PROFILE") != NULL;
        if (prof && box->st.packets)
            fprintf(stderr, "MGL CS: packets=%llu drains=%llu sleeps=%llu\n",
                    (unsigned long long)box->st.packets,
                    (unsigned long long)box->st.drains,
                    (unsigned long long)box->st.waits);
    }

    cs_drain(&box->st);

    objc_setAssociatedObject(self, &kCompatCSKey, nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

/* C bridge so GL-side code can drain without touching ObjC headers */
void mgl_compat_cs_drain(GLMContext glm_ctx)
{
    if (!glm_ctx || !glm_ctx->mtl_funcs.mtlObj)
        return;
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj compatCSDrain];
}

@end
