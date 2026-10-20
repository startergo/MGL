/*
 * MGLRenderer+Queries.m — Metal side of the 'queries' domain (see mgl_ext_queries.h).
 *
 * Occlusion family + FRAGMENT_SHADER_INVOCATIONS: every render pass gets a pooled
 * visibility result buffer; each change of the set of active sample-counting queries
 * (or a new render encoder) opens a new 8-byte slot in Counting mode. A query owns
 * the list of slots it was active for and sums them once their command buffers complete.
 *
 * TIME_ELAPSED / TIMESTAMP: Begin/End/QueryCounter split the command stream at a
 * command buffer boundary; results come from GPUStartTime/GPUEndTime (host timebase).
 */
#import "MGLRenderer_Private.h"
#import <objc/runtime.h>
#import <QuartzCore/QuartzCore.h>

#include "mgl_ext_queries.h"

#define MGLQ_VIS_SLOTS      2048
#define MGLQ_POOL_MAX       64

@class MGLQState;

@interface MGLQPass : NSObject {
@public
    id<MTLBuffer> buf;
    id<MTLCommandBuffer> cb;
    NSUInteger next;
    __weak id<MTLRenderCommandEncoder> enc;
    __weak MGLQState *owner;
}
@end

@interface MGLQSlot : NSObject {
@public
    MGLQPass *pass;
    NSUInteger off;
}
@end

@interface MGLQRecord : NSObject {
@public
    NSMutableArray<MGLQSlot *> *slots;
    id<MTLCommandBuffer> firstCB;
    id<MTLCommandBuffer> lastCB;
    id<MTLCommandBuffer> prevCB;    /* last command buffer committed before a TIMESTAMP */
    MGLQRecord *prevTS;             /* previous unresolved TIMESTAMP (monotonic chain) */
    CFTimeInterval hostIssue;
    BOOL tsResolved;
    CFTimeInterval tsValue;
}
@end

@interface MGLQState : NSObject {
@public
    NSMutableArray *pool;           /* @[buffer, commandBuffer | NSNull] */
    MGLQPass *cur;
    __weak id<MTLCommandBuffer> lastCommitted;
    MGLQRecord *lastTS;
    MGLQuery *sampleActive[2];      /* [0] occlusion family, [1] FRAGMENT_SHADER_INVOCATIONS */
}
@end

@implementation MGLQSlot
@end

@implementation MGLQRecord
- (instancetype) init { if ((self = [super init])) slots = [NSMutableArray array]; return self; }
@end

@implementation MGLQState
- (instancetype) init { if ((self = [super init])) pool = [NSMutableArray array]; return self; }
@end

@implementation MGLQPass
- (void) dealloc
{
    MGLQState *o = owner;
    if (o && buf && o->pool.count < MGLQ_POOL_MAX)
        [o->pool addObject: @[buf, cb ? (id)cb : (id)[NSNull null]]];
}
@end

static char kMGLQStateKey;

static inline int mglq_sample_slot(GLenum target)
{
    switch (target) {
        case GL_SAMPLES_PASSED:
        case GL_ANY_SAMPLES_PASSED:
        case GL_ANY_SAMPLES_PASSED_CONSERVATIVE:
            return 0;
        case GL_FRAGMENT_SHADER_INVOCATIONS:
            return 1;
    }
    return -1;
}

@implementation MGLRenderer (Queries)

- (MGLQState *) mglqState
{
    MGLQState *st = objc_getAssociatedObject(self, &kMGLQStateKey);
    if (!st)
    {
        st = [[MGLQState alloc] init];
        objc_setAssociatedObject(self, &kMGLQStateKey, st, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    return st;
}

#pragma mark renderer hooks

- (void) extQueriesRenderPassDescriptor: (MTLRenderPassDescriptor *) desc
{
    MGLQState *st = [self mglqState];
    id<MTLBuffer> buf = nil;

    for (NSUInteger i = 0; i < st->pool.count; i++)
    {
        NSArray *e = st->pool[i];
        id cbo = e[1];
        BOOL free_ = NO;
        if (cbo == [NSNull null])
            free_ = YES;
        else
        {
            id<MTLCommandBuffer> cb = cbo;
            MTLCommandBufferStatus s = cb.status;
            /* free when the buffer ran, or when it never will: an abandoned uncommitted
             * buffer (NotEnqueued, or Enqueued by the geometry category and then replaced)
             * holds no results */
            free_ = (s >= MTLCommandBufferStatusCompleted) ||
                    (s < MTLCommandBufferStatusCommitted && cb != _currentCommandBuffer);
        }
        if (free_)
        {
            buf = e[0];
            [st->pool removeObjectAtIndex: i];
            break;
        }
    }

    if (!buf)
        buf = [_device newBufferWithLength: MGLQ_VIS_SLOTS * sizeof(uint64_t) options: MTLResourceStorageModeShared];

    if (!buf)
    {
        desc.visibilityResultBuffer = nil;
        st->cur = nil;
        return;
    }

    MGLQPass *p = [[MGLQPass alloc] init];
    p->buf = buf;
    p->cb = _currentCommandBuffer;
    p->next = 0;
    p->owner = st;
    st->cur = p;

    desc.visibilityResultBuffer = buf;
}

- (void) extQueriesRenderEncoderCreated: (id<MTLRenderCommandEncoder>) enc
{
    MGLQState *st = [self mglqState];
    if (st->cur)
        st->cur->enc = enc;
    [self mglqRestartSegment];
}

- (void) extQueriesWillCommit: (id<MTLCommandBuffer>) cmdbuf
{
    [self mglqState]->lastCommitted = cmdbuf;
}

/* open a new visibility slot for the currently active sample queries (or disable counting) */
- (void) mglqRestartSegment
{
    MGLQState *st = [self mglqState];
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    MGLQPass *p = st->cur;

    if (!enc || !p || p->enc != enc)
        return;

    if (!st->sampleActive[0] && !st->sampleActive[1])
    {
        [enc setVisibilityResultMode: MTLVisibilityResultModeDisabled offset: 0];
        return;
    }

    NSUInteger slot;
    BOOL reused = NO;
    if (p->next < MGLQ_VIS_SLOTS)
        slot = p->next++;
    else
    {
        slot = MGLQ_VIS_SLOTS - 1;  /* out of slots in this pass: keep accumulating into the last one */
        reused = YES;
    }

    NSUInteger off = slot * sizeof(uint64_t);
    if (!reused)
        memset((uint8_t *)p->buf.contents + off, 0, sizeof(uint64_t));

    [enc setVisibilityResultMode: MTLVisibilityResultModeCounting offset: off];

    for (int i = 0; i < 2; i++)
    {
        MGLQuery *q = st->sampleActive[i];
        if (!q || !q->gpu)
            continue;
        MGLQRecord *rec = (__bridge MGLQRecord *)q->gpu;
        MGLQSlot *last = rec->slots.lastObject;
        if (reused && last && last->pass == p && last->off == off)
            continue;
        MGLQSlot *s = [[MGLQSlot alloc] init];
        s->pass = p;
        s->off = off;
        [rec->slots addObject: s];
    }
}

/* commit the current command buffer and start a new one */
- (void) mglqSplit
{
    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    /* < Committed, not just == NotEnqueued: the geometry category enqueues its command
     * buffer while it is still open, and abandoning that buffer here (creating a new one
     * without committing it) would leave GPU work that never runs — any later wait on it
     * (query resolve, fence, readback) hangs forever */
    if (cb && cb.status < MTLCommandBufferStatusCommitted)
        [self flushCommandBuffer: false];
    if (!_currentCommandBuffer || _currentCommandBuffer.status != MTLCommandBufferStatusNotEnqueued)
        [self newCommandBuffer];
    /* the flush ended the render encoder: make the next draw open a new render pass */
    ctx->state.dirty_bits |= DIRTY_VAO;
}

/* YES when cb's results are final (completed, or dropped without ever being committed) */
- (BOOL) mglqCommandBufferDone: (id<MTLCommandBuffer>) cb wait: (BOOL) wait
{
    if (!cb)
        return YES;

    MTLCommandBufferStatus s = cb.status;
    if (s >= MTLCommandBufferStatusCompleted)
        return YES;

    if (s == MTLCommandBufferStatusEnqueued)
    {
        /* the geometry category reserves queue slots with -enqueue and the renderer commits
         * the buffer later; if it was replaced first (a query split, a readback, a texture
         * upload), nobody will commit it anymore.  Kick it so the GPU work this query
         * covers actually runs — waiting without committing hangs forever. */
        if (cb == _currentCommandBuffer)
        {
            [self mglqSplit];
        }
        else
        {
            @try { [cb commit]; }
            @catch (NSException *) { return YES; }   /* uncommittable: it will never run */
        }
        if (cb.status == MTLCommandBufferStatusEnqueued)
            return YES;                              /* commit refused: never runs */
    }
    else if (s == MTLCommandBufferStatusNotEnqueued)
    {
        if (cb != _currentCommandBuffer)
            return YES;                 /* abandoned by the renderer: never executed */
        [self mglqSplit];               /* kick it so the result eventually becomes available */
        if (cb.status == MTLCommandBufferStatusNotEnqueued)
            return YES;
    }

    if (!wait)
        return cb.status >= MTLCommandBufferStatusCompleted;

    [cb waitUntilCompleted];
    return YES;
}

#pragma mark query operations

- (void) mglqBegin: (MGLQuery *) q
{
    MGLQState *st = [self mglqState];
    MGLQRecord *rec = [[MGLQRecord alloc] init];
    q->gpu = (void *)CFBridgingRetain(rec);

    int si = mglq_sample_slot(q->target);
    if (si >= 0)
    {
        st->sampleActive[si] = q;
        [self mglqRestartSegment];
        return;
    }

    if (q->target == GL_TIME_ELAPSED)
    {
        [self mglqSplit];
        rec->firstCB = _currentCommandBuffer;
    }
}

- (void) mglqEnd: (MGLQuery *) q
{
    MGLQState *st = [self mglqState];
    int si = mglq_sample_slot(q->target);
    if (si >= 0)
    {
        if (st->sampleActive[si] == q)
            st->sampleActive[si] = NULL;
        [self mglqRestartSegment];
        return;
    }

    if (q->target == GL_TIME_ELAPSED && q->gpu)
    {
        MGLQRecord *rec = (__bridge MGLQRecord *)q->gpu;
        rec->lastCB = _currentCommandBuffer;
        [self mglqSplit];
    }
}

- (void) mglqCounter: (MGLQuery *) q
{
    MGLQRecord *rec = [[MGLQRecord alloc] init];
    q->gpu = (void *)CFBridgingRetain(rec);
    MGLQState *st = [self mglqState];
    rec->prevCB = st->lastCommitted;
    rec->prevTS = st->lastTS;
    st->lastTS = rec;
    rec->hostIssue = CACurrentMediaTime();
    rec->lastCB = _currentCommandBuffer;
    [self mglqSplit];
}

- (void) mglqRelease: (MGLQuery *) q
{
    MGLQState *st = [self mglqState];
    for (int i = 0; i < 2; i++)
        if (st->sampleActive[i] == q)
            st->sampleActive[i] = NULL;
    if (q->gpu)
    {
        CFBridgingRelease(q->gpu);
        q->gpu = NULL;
    }
}

/* time at which all commands issued before the QueryCounter completed.  Metal command
 * buffers on one queue may overlap on the GPU, so take the max over the counter's own
 * command buffer, the previously committed one and the previous timestamp (monotonic). */
- (BOOL) mglqTimestamp: (MGLQRecord *) rec wait: (BOOL) wait
{
    /* oldest unresolved first (iterative: the chain can be long if results are never read) */
    NSMutableArray<MGLQRecord *> *chain = [NSMutableArray array];
    for (MGLQRecord *r = rec; r && !r->tsResolved; r = r->prevTS)
        [chain addObject: r];

    for (NSInteger i = (NSInteger)chain.count - 1; i >= 0; i--)
    {
        MGLQRecord *r = chain[(NSUInteger)i];
        if (![self mglqCommandBufferDone: r->lastCB wait: wait])
            return NO;
        if (r->prevCB && ![self mglqCommandBufferDone: r->prevCB wait: wait])
            return NO;

        CFTimeInterval t = r->lastCB.GPUEndTime;
        if (r->prevCB && r->prevCB.GPUEndTime > t)
            t = r->prevCB.GPUEndTime;
        if (t <= 0)
            t = r->hostIssue;
        if (r->prevTS && r->prevTS->tsValue > t)
            t = r->prevTS->tsValue;

        r->tsValue = t;
        r->tsResolved = YES;
        r->prevTS = nil;
        r->prevCB = nil;
    }
    return YES;
}

- (BOOL) mglqResolve: (MGLQuery *) q wait: (BOOL) wait
{
    if (!q->gpu)
    {
        q->result = 0;
        return YES;
    }

    MGLQRecord *rec = (__bridge MGLQRecord *)q->gpu;
    uint64_t result = 0;

    if (mglq_sample_slot(q->target) >= 0)
    {
        id<MTLCommandBuffer> seen = nil;
        for (MGLQSlot *s in rec->slots)
        {
            id<MTLCommandBuffer> cb = s->pass->cb;
            if (cb == seen)
                continue;
            if (![self mglqCommandBufferDone: cb wait: wait])
                return NO;
            seen = cb;
        }
        for (MGLQSlot *s in rec->slots)
        {
            uint64_t v;
            memcpy(&v, (uint8_t *)s->pass->buf.contents + s->off, sizeof v);
            result += v;
        }
    }
    else if (q->target == GL_TIME_ELAPSED)
    {
        if (![self mglqCommandBufferDone: rec->firstCB wait: wait] ||
            ![self mglqCommandBufferDone: rec->lastCB wait: wait])
            return NO;
        CFTimeInterval t0 = rec->firstCB.GPUStartTime, t1 = rec->lastCB.GPUEndTime;
        if (t0 <= 0) t0 = rec->lastCB.GPUStartTime;
        if (t0 > 0 && t1 > t0)
            result = (uint64_t)((t1 - t0) * 1e9);
    }
    else if (q->target == GL_TIMESTAMP)
    {
        if (![self mglqTimestamp: rec wait: wait])
            return NO;
        result = (uint64_t)(rec->tsValue * 1e9);
    }

    q->result = result;
    CFBridgingRelease(q->gpu);
    q->gpu = NULL;
    return YES;
}

static void mglq_begin(GLMContext ctx, MGLQuery *q)
{
    [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj mglqBegin: q];
}

static void mglq_end(GLMContext ctx, MGLQuery *q)
{
    [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj mglqEnd: q];
}

static void mglq_counter(GLMContext ctx, MGLQuery *q)
{
    [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj mglqCounter: q];
}

static bool mglq_resolve(GLMContext ctx, MGLQuery *q, bool wait)
{
    return [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj mglqResolve: q wait: wait];
}

static void mglq_release(GLMContext ctx, MGLQuery *q)
{
    [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj mglqRelease: q];
}

static int64_t mglq_now_ns(GLMContext ctx)
{
    (void)ctx;
    return (int64_t)(CACurrentMediaTime() * 1e9);
}

- (void) bindExtQueriesFuncsToGLMContext: (GLMContext) glm_ctx
{
    struct MGLExtMetal_queries *m = glm_ctx->mtl_funcs.ext_queries;
    if (!m)
    {
        m = (struct MGLExtMetal_queries *)calloc(1, sizeof(*m));
        if (!m)
            return;
        glm_ctx->mtl_funcs.ext_queries = m;
    }
    m->begin = mglq_begin;
    m->end = mglq_end;
    m->counter = mglq_counter;
    m->resolve = mglq_resolve;
    m->release = mglq_release;
    m->now_ns = mglq_now_ns;
    (void)[self mglqState];
}

@end
