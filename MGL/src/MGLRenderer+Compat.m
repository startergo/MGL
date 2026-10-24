/*
 * MGLRenderer+Compat.m — compat-domain renderer support.
 *
 * IOSurface present path ("CGL-like async swap"):
 * CAMetalLayer -nextDrawable blocks until the window server retires a drawable
 * (83% of glmark2's frame time on this Mac — sampled; FramePacing retirement
 * follows the compositor, which is slow/asleep in headless bench sessions), so
 * a swap-rate benchmark can never approach native NSOpenGL speed through
 * drawables: CGL swaps a client-side IOSurface and never waits for the server.
 * This path mirrors that: MGL renders into its offscreen default framebuffer,
 * blits into one of N IOSurface-backed textures, commits, and shows the
 * surface with layer.contents — no drawable, no blocking, the compositor picks
 * up the latest surface whenever it runs.  Enabled per renderer with
 * MGL_SURFACE_PRESENT=1 (the NSOpenGL interposer sets it before binding a
 * windowed renderer; GLFW/GLX/headless keep the drawable path unchanged).
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"
#import "mgl_ext_rstate.h"
#import "MGLRenderer+RenderState.h"   /* rstatePipelineContentHash (perf session 3) */
#include "mgl_compat_perf.h"

/* MGL_PROFILE counters (MGLRenderer.m) */
extern uint64_t mgl_prof_ns[18];
extern uint64_t mgl_prof_cnt[18];

static inline int compatProfOn(void)
{
    static int on = -1;
    if (on < 0)
    {
        const char *e = getenv("MGL_PROFILE");
        on = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return on;
}
#import <QuartzCore/QuartzCore.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>

/* fb-category internals used here (implemented in MGLRenderer+Framebuffer.m);
 * declared locally because the fb header only exports the public surface */
/* FBTarget is private to the fb .m; mirror the layout for the blit call
 * (id<MTLTexture> tex; NSUInteger level, slice, plane, layers;) */
typedef struct {
    __unsafe_unretained id<MTLTexture> tex;
    NSUInteger level, slice, plane, layers;
} CompatFBTarget;

@interface MGLRenderer (Framebuffer_compat_present)
- (id<MTLTexture>) fbEnsureDefaultFramebuffer;
- (void) fbEnsureCommandBuffer;
- (void) fbShaderBlitFrom:(CompatFBTarget *)src to:(CompatFBTarget *)dst src:(const GLint *)s dst:(const GLint *)d filter:(GLenum)filter scissor:(bool)scissor;
@end

@interface MGLCompatSurfacePool : NSObject
@property (nonatomic, strong) CALayer *layer;            /* displays the surfaces */
@property (nonatomic, strong) NSMutableArray<id<MTLTexture>> *textures;
@property (nonatomic, strong) NSMutableArray<id> *surfaces;   /* IOSurfaceRef bridged */
@property (nonatomic, assign) NSUInteger w, h;
@property (nonatomic, assign) unsigned next;             /* round-robin index (blit path) */
@property (nonatomic, assign) unsigned write;            /* slot the default framebuffer renders into (direct path) */
@property (nonatomic, assign) BOOL rotate;               /* advance `write` at the next fbEnsureDefaultFramebuffer */
@end

@implementation MGLCompatSurfacePool
@end

#define COMPAT_SURFACES 3

/* ------------------------------------------------------------------ */
/* pipeline fast path
 *
 * processGLState rebuilds a full MTLRenderPipelineDescriptor (and the vertex
 * descriptor, function specialization, gpuaudit content hash) on every draw
 * whose dirty bits include any pipeline-relevant bit — glmark2-style apps
 * re-specify their vertex attribs every frame, so that is EVERY draw
 * (~0.05 ms, the single biggest per-draw cost).  When nothing pipeline-affecting
 * changed except the VAO/encoder-state dirty bits, a cheap 128-bit key over the
 * GL state (program, render-state generation, VAO layout hash, FBO identity)
 * maps directly to the pipeline created for that exact state. */
typedef struct {
    uint64_t a, b;
} CompatPipelineKey;

/* one cached key -> pipeline association */
@interface CompatPipeBox : NSObject
@property (nonatomic, assign) CompatPipelineKey key;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipe;
@end

@implementation CompatPipeBox
@end

@interface MGLCompatPipelineCache : NSObject
@property (nonatomic, strong) NSMutableArray<CompatPipeBox *> *entries;
@property (nonatomic, strong) CompatPipeBox *pending;   /* key awaiting its pipeline */
@property (nonatomic, assign) uint64_t encoder_vao_hash;  /* VAO layout the current encoder was built for */
@property (nonatomic, assign) BOOL encoder_vao_valid;
@property (nonatomic, assign) CompatPipelineKey lastKey;  /* perf: key-component attribution of misses */
@property (nonatomic, assign) BOOL lastKeyValid;
@property (nonatomic, assign) NSUInteger missCount;
@end

@implementation MGLCompatPipelineCache
@end

static const void *kCompatPipeKey = &kCompatPipeKey;
static const void *kCompatPoolKey = &kCompatPoolKey;

/* last pipeline bound to the current encoder (skip redundant ObjC rebinds) */
@interface MGLCompatPipeBindBox : NSObject
@property (nonatomic, strong) id<MTLRenderPipelineState> pipe;
@property (nonatomic, weak) id<MTLRenderCommandEncoder> encoder;
@end
@implementation MGLCompatPipeBindBox
@end

/* ------------------------------------------------------------------ */
/* perf session 3: encoder bind shadow + streaming upload ring
 * (state: struct MGLExtMetal_compat, mgl_compat_perf.h)
 *
 * extBufSerial / extBufCompletedSerial / extBufWaitSerial live in the Draw
 * category (MGLRenderer+Draw.m); redeclared here for the ring's GPU fencing. */
@interface MGLRenderer (Draw_compat_perf)
- (uint64_t) extBufSerial;
- (uint64_t) extBufCompletedSerial;
- (void) extBufWaitSerial: (uint64_t) serial;
@end

static inline struct MGLExtMetal_compat *compatMetal(GLMContext ctx)
{
    return (struct MGLExtMetal_compat *)(ctx ? ctx->mtl_funcs.ext_compat : NULL);
}

/* current encoder generation, bumped whenever the argument binds see a
 * different encoder than the last one they recorded (render encoders are
 * also created directly by the xfb / geom capture paths, which swap
 * _currentRenderEncoder around their own encoder; a recycled encoder address
 * must never inherit another encoder's shadow).  extCompatRenderEncoderCreated:
 * bumps the same counter for the GL path. */
static inline uint64_t compatEncGen(struct MGLExtMetal_compat *m, void *enc)
{
    if (m->seen_encoder != enc)
    {
        m->seen_encoder = enc;
        m->encoder_gen++;
    }
    return m->encoder_gen;
}

@implementation MGLRenderer (Compat)

/* fold the bound VAO's attribute layout into 64 bits (enough entropy to notice
 * any re-specification of the same VAO; a full hash collision only picks a
 * stale-but-identical-layout pipeline) */
static uint64_t compatVAOHash(GLMContext ctx)
{
    VertexArray *vao = ctx->state.vao;
    uint64_t h = 1469598103934665603ULL;
    #define MIX(v) do { h ^= (uint64_t)(uintptr_t)(v); h *= 1099511628211ULL; } while (0)
    MIX(vao);
    MIX(vao->enabled_attribs);
    for (GLuint i = 0; i < MAX_ATTRIBS && i < 32; i++)
    {
        VertexAttrib *a = &vao->attrib[i];
        MIX(a->buffer); MIX(a->size); MIX(a->type); MIX(a->normalized);
        MIX(a->stride); MIX(a->divisor); MIX(a->relativeoffset);
    }
    MIX(vao->element_array.buffer);
    return h;
}
#undef MIX

/* the hashed bytes of a VertexArray: everything from enabled_attribs through
 * element_array (mtl_data is not hashed and is excluded, so a memcmp of this
 * range is an EXACT validator of the hash inputs) */
#define COMPAT_VAO_SNAP_OFF  offsetof(VertexArray, enabled_attribs)
#define COMPAT_VAO_SNAP_LEN  (offsetof(VertexArray, mtl_data) - offsetof(VertexArray, enabled_attribs))

/* cached variant: one entry, validated by memcmp against the live struct, so
 * the per-draw cost is a pointer compare plus ~700 bytes of memcmp instead of
 * ~130 dependent multiplies */
static uint64_t compatVAOHashCached(struct MGLExtMetal_compat *m, GLMContext ctx)
{
    VertexArray *vao = ctx->state.vao;
    const uint8_t *live = (const uint8_t *)vao + COMPAT_VAO_SNAP_OFF;
    if (!m)
        return compatVAOHash(ctx);
    if (m->vaokey.vao == (void *)vao && m->vaokey.len == COMPAT_VAO_SNAP_LEN &&
        m->vaokey.snap && memcmp(m->vaokey.snap, live, COMPAT_VAO_SNAP_LEN) == 0)
        return m->vaokey.hash;
    uint64_t h = compatVAOHash(ctx);
    if (!m->vaokey.snap)
        m->vaokey.snap = (uint8_t *)malloc(COMPAT_VAO_SNAP_LEN);
    if (m->vaokey.snap)
    {
        memcpy(m->vaokey.snap, live, COMPAT_VAO_SNAP_LEN);
        m->vaokey.vao = (void *)vao;
        m->vaokey.len = COMPAT_VAO_SNAP_LEN;
    }
    m->vaokey.hash = h;
    return h;
}

/* content-identity, not object identity: programs are recreated at
 * recycled addresses during link-test loops (tests/headless/test_fp64.c
 * failed with a program-pointer key), but a linked stage's generated MSL
 * string is immutable and unique per compiled shader.  Hashing the whole
 * source per draw is too slow, so the key mixes the string POINTER with a
 * content checksum of its head and tail — an address reused for different
 * text can no longer alias a live entry */
static uint64_t compatMSLKey(const char *msl)
{
    if (!msl)
        return 0;
    size_t len = strlen(msl);
    uint64_t h = (uint64_t)len * 0x9E3779B97F4A7C15ULL ^ (uint64_t)(uintptr_t)msl;
    size_t n = len < 192 ? len : 192;
    for (size_t i = 0; i < n; i++)
        h = (h ^ (unsigned char)msl[i]) * 1099511628211ULL;
    if (len > 256)
        for (size_t i = len - 128; i < len; i++)
            h = (h ^ (unsigned char)msl[i]) * 1099511628211ULL;
    return h;
}

/* cached variant for the per-draw fast path: the full hash touches ~320 bytes
 * of shader text per stage; the cache revalidates with two 128-byte memcmps.
 * Strings short enough for head/tail to overlap are never cached (hashed
 * directly — real generated MSL is orders of magnitude longer). */
#define MGLC_MSL_VALIDATE 128
static inline uint64_t compatMSLKeyCached(struct MGLExtMetal_compat *m, const char *msl)
{
    if (!msl)
        return 0;
    if (!m)
        return compatMSLKey(msl);
    __typeof__(m->mslkey[0]) *e = &m->mslkey[((uintptr_t)msl * 0x9E3779B97F4A7C15ULL) >> 59];
    if (e->msl == msl && e->len >= 2 * MGLC_MSL_VALIDATE &&
        memcmp(e->head, msl, MGLC_MSL_VALIDATE) == 0 &&
        memcmp(e->tail, msl + e->len - MGLC_MSL_VALIDATE, MGLC_MSL_VALIDATE) == 0)
        return e->key;
    uint64_t k = compatMSLKey(msl);
    size_t len = strlen(msl);
    if (len >= 2 * MGLC_MSL_VALIDATE)
    {
        e->msl = msl;
        e->key = k;
        e->len = len;
        memcpy(e->head, msl, MGLC_MSL_VALIDATE);
        memcpy(e->tail, msl + len - MGLC_MSL_VALIDATE, MGLC_MSL_VALIDATE);
    }
    return k;
}

- (bool) compatPipelineFastPath
{
    /* pipeline-relevant dirty bits VAO / program / render state / FBO / alpha
     * all go through the cache: the key covers the program content identity
     * (MSL), the whole VAO layout, the rstate pipeline-baked CONTENT hash,
     * the framebuffer pointer and the current encoder's attachment formats.
     * Shader relink and drawable (window/display) changes still take the
     * descriptor path — both are rare, and the latter can change formats
     * behind the framebuffer pointer. */
    if (ctx->state.dirty_bits & (DIRTY_SHADER | DIRTY_DRAWABLE))
    {
        if (mgl_prof_cnt[17] % 2000 == 0)
            fprintf(stderr, "MGL PROFILE pipeMiss-decline dirty=0x%llx\n",
                    (unsigned long long)ctx->state.dirty_bits);
        mgl_prof_cnt[17]++;   /* declined: a hard bit is set */
        return false;
    }
    if (!ctx->state.program)
        return false;

    struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    CompatPipelineKey key;
    /* content-identity (see compatMSLKey): the three raster stages' generated
     * MSL identifies the pipeline functions exactly */
    key.a = compatMSLKeyCached(m, ctx->state.program->spirv[_VERTEX_SHADER].msl_str)
          ^ compatMSLKeyCached(m, ctx->state.program->spirv[_FRAGMENT_SHADER].msl_str) * 0x100000001B3ULL
          ^ compatMSLKeyCached(m, ctx->state.program->spirv[_GEOMETRY_SHADER].msl_str);
    /* pipeline-baked rstate by CONTENT hash, not by the monotonic generation
     * counter: alternating blend / mask states must produce the same two keys
     * forever (rstatePipelineContentHash is cached against the generation, so
     * the per-draw cost stays two integer compares) */
    if (!m || m->rstate_hash_gen == 0 ||
        (rs && m->rstate_hash_gen != (uint64_t)rs->generation_pipeline))
    {
        uint64_t h = [self rstatePipelineContentHash];
        if (m)
        {
            if (rs) m->rstate_hash_gen = (uint64_t)rs->generation_pipeline;
            m->rstate_hash = h;
        }
    }
    key.b = compatVAOHashCached(m, ctx) ^ ((m ? m->rstate_hash : 1) << 1) ^ ((uint64_t)(uintptr_t)ctx->state.framebuffer * 0x9E3779B97F4A7C15ULL);

    /* attachment formats + sample count of the CURRENT encoder: an attached
     * texture can be re-specified with a different format behind an unchanged
     * framebuffer pointer, and the PSO is validated against these formats.
     * Every attachment change ends the encoder (framebuffers.c targetsChanged
     * -> endRenderEncoding), so hashing once per encoder generation is exact;
     * the per-draw cost is two integer compares. */
    if (!m || m->fmt_hash_gen == 0 ||
        m->fmt_hash_gen != m->encoder_gen || m->fmt_fbo != (uint64_t)(uintptr_t)ctx->state.framebuffer)
    {
        uint64_t fh = 0;
        MTLRenderPassDescriptor *rp = _renderPassDescriptor;
        if (rp)
        {
            for (int i = 0; i < 4; i++)
            {
                id<MTLTexture> t = rp.colorAttachments[i].texture;
                fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)t.pixelFormat;
                fh = fh * 1099511628211ULL + (uint64_t)t.sampleCount;
            }
            fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)rp.depthAttachment.texture.pixelFormat;
            fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)rp.stencilAttachment.texture.pixelFormat;
        }
        if (m)
        {
            m->fmt_hash = fh ? fh : 1;
            m->fmt_hash_gen = m->encoder_gen ? m->encoder_gen : 1;
            m->fmt_fbo = (uint64_t)(uintptr_t)ctx->state.framebuffer;
        }
    }
    key.b ^= (m ? m->fmt_hash : 0) * 0xD6E8FEB86659FD93ULL;

    MGLCompatPipelineCache *cache = objc_getAssociatedObject(self, kCompatPipeKey);
    if (!cache)
    {
        cache = [[MGLCompatPipelineCache alloc] init];
        cache.entries = [NSMutableArray array];
        objc_setAssociatedObject(self, kCompatPipeKey, cache, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    /* linear scan with move-to-front: the working set is small per scene but
     * the full glmark2 list touches hundreds of (MSL, layout) keys, and a
     * short FIFO evicted live entries between sub-tests (conditionals: 5700
     * FPS solo vs 780 in-suite with a 64-entry cache) */
    for (NSUInteger i = 0; i < cache.entries.count; i++)
    {
        CompatPipeBox *box = cache.entries[i];
        if (box.pipe && box.key.a == key.a && box.key.b == key.b)
        {
            if (i)
            {
                [cache.entries removeObjectAtIndex:i];
                [cache.entries insertObject:box atIndex:0];
            }
            _pipelineState = box.pipe;
            mgl_prof_cnt[16]++;
            return true;
        }
        /* perf instrumentation: a matching key whose box never received a
         * pipeline (the descriptor path must have bailed before storing) */
        if (!box.pipe && box.key.a == key.a && box.key.b == key.b && compatProfOn())
        {
            static uint64_t dead = 0;
            if (++dead % 200 == 1)
                fprintf(stderr, "MGL PROFILE pipeMiss dead-box (key match, pipe nil) entries=%lu\n",
                        (unsigned long)cache.entries.count);
        }
    }
    /* miss: remember the key; the descriptor path stores the pipeline into it
     * (compatPipelineStore below, called by processGLState) */
    CompatPipeBox *box = [[CompatPipeBox alloc] init];
    box.key = key;
    [cache.entries insertObject:box atIndex:0];
    if (cache.entries.count > 1024)
        [cache.entries removeLastObject];
    cache.pending = box;

    /* perf instrumentation: WHICH key component moved (a = MSL content ids,
     * b = VAO layout / pipeline-baked rstate / FBO identity) */
    if (compatProfOn())
    {
        if (!cache.lastKeyValid)
        {
            cache.lastKey = key;
            cache.lastKeyValid = YES;
            cache.missCount = 0;
        }
        else if (cache.lastKey.a != key.a || cache.lastKey.b != key.b)
        {
            cache.missCount++;
            if (cache.missCount % 200 == 1)
                fprintf(stderr, "MGL PROFILE pipeMiss %s (a %llx->%llx b %llx->%llx; vao=%llx gen=%u fbo=%p)\n",
                        cache.lastKey.a != key.a ? "msl" : "vao/rstate/fbo",
                        (unsigned long long)cache.lastKey.a, (unsigned long long)key.a,
                        (unsigned long long)cache.lastKey.b, (unsigned long long)key.b,
                        (unsigned long long)compatVAOHash(ctx),
                        rs ? rs->generation_pipeline : 0,
                        (void *)ctx->state.framebuffer);
            cache.lastKey = key;
        }
    }
    return false;
}

/* called after the descriptor path produced a pipeline for the pending key */
- (void) compatPipelineStore
{
    MGLCompatPipelineCache *cache = objc_getAssociatedObject(self, kCompatPipeKey);
    if (cache && cache.pending)
    {
        cache.pending.pipe = _pipelineState;
        cache.pending = nil;
    }
    else if (compatProfOn())
    {
        /* descriptor path finished but no pending key: the pipeline was built
         * without a fast-path miss (or the miss's box was already consumed) */
        static uint64_t n = 0;
        if (++n % 200 == 1)
            fprintf(stderr, "MGL PROFILE pipeStore without pending\n");
    }
}

static const void *kCompatPipeBindKey = &kCompatPipeBindKey;

/* skip the per-draw setRenderPipelineState: ObjC call when the encoder already
 * has this pipeline (a new encoder starts empty, a pipeline change rebinds) */
- (bool) compatPipelineBindNeeded
{
    if (!_currentRenderEncoder || !_pipelineState)
        return true;
    MGLCompatPipeBindBox *b = objc_getAssociatedObject(self, kCompatPipeBindKey);
    if (!b)
    {
        b = [[MGLCompatPipeBindBox alloc] init];
        objc_setAssociatedObject(self, kCompatPipeBindKey, b, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    if (b.pipe == _pipelineState && b.encoder == _currentRenderEncoder)
        return false;
    b.pipe = _pipelineState;
    b.encoder = _currentRenderEncoder;
    return true;
}

/* per-draw counters for the profiler: skipped vs issued Metal argument binds */
uint64_t mgl_compat_bind_skip, mgl_compat_bind_set, mgl_compat_ring_bytes;

- (void) bindExtCompatFuncsToGLMContext: (GLMContext) glm_ctx
{
    /* allocated for every renderer (both profiles): the bind shadow and the
     * upload ring are profile-neutral draw-path optimisations */
    struct MGLExtMetal_compat *m = (struct MGLExtMetal_compat *)calloc(1, sizeof(*m));
    assert(m);

    static int shadow_env = -1;
    if (shadow_env < 0)
        shadow_env = !(getenv("MGL_BIND_SHADOW") && strcmp(getenv("MGL_BIND_SHADOW"), "0") == 0);

    m->enabled = shadow_env;
    m->ring_on = 0;
    m->encoder_gen = 1;

    glm_ctx->mtl_funcs.ext_compat = m;
}

/* auto-called by newRenderEncoder: a fresh encoder starts with an empty
 * argument table, so every shadow entry is stale */
- (void) extCompatRenderEncoderCreated: (id<MTLRenderCommandEncoder>) enc
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (m)
    {
        m->encoder_gen++;
        m->seen_encoder = (__bridge void *)enc;
    }
}

/* foreign paths (xfb / geometry capture) build their own render encoder and
 * swap _currentRenderEncoder around it while reusing this category's binding
 * code: call this when the swap happens so no shadow entry survives into the
 * next encoder, even when the driver recycles the encoder's address */
- (void) compatBindInvalidate
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (m)
    {
        m->encoder_gen++;
        m->seen_encoder = NULL;
    }
}

/* -------- real-buffer binds (identity + offset filtered) ------------ */

- (void) compatBindVertexBuffer: (id<MTLBuffer>)buf offset: (NSUInteger)offset atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || idx >= MGLC_BIND_SLOTS)
    {
        [enc setVertexBuffer: buf offset: offset atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->vbuf[idx];

    if (sh->obj == (__bridge void *)buf && sh->offset == (size_t)offset &&
        sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setVertexBuffer: buf offset: offset atIndex: idx];
    sh->obj = (__bridge void *)buf;
    sh->offset = (size_t)offset;
    sh->gen = gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

- (void) compatBindFragmentBuffer: (id<MTLBuffer>)buf offset: (NSUInteger)offset atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || idx >= MGLC_BIND_SLOTS)
    {
        [enc setFragmentBuffer: buf offset: offset atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->fbuf[idx];

    if (sh->obj == (__bridge void *)buf && sh->offset == (size_t)offset &&
        sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setFragmentBuffer: buf offset: offset atIndex: idx];
    sh->obj = (__bridge void *)buf;
    sh->offset = (size_t)offset;
    sh->gen = gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

/* -------- texture / sampler binds (identity filtered) --------------- */

- (void) compatSetVertexTexture: (id<MTLTexture>)tex atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || idx >= MGLC_BIND_SLOTS)
    {
        [enc setVertexTexture: tex atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->vtex[idx];

    if (sh->obj == (__bridge void *)tex && sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setVertexTexture: tex atIndex: idx];
    sh->obj = (__bridge void *)tex;
    sh->offset = 0;
    sh->gen = m->encoder_gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

- (void) compatSetFragmentTexture: (id<MTLTexture>)tex atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || idx >= MGLC_BIND_SLOTS)
    {
        [enc setFragmentTexture: tex atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->ftex[idx];

    if (sh->obj == (__bridge void *)tex && sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setFragmentTexture: tex atIndex: idx];
    sh->obj = (__bridge void *)tex;
    sh->offset = 0;
    sh->gen = m->encoder_gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

- (void) compatSetVertexSamplerState: (id<MTLSamplerState>)samp atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || !samp || idx >= MGLC_BIND_SLOTS)
    {
        if (enc) [enc setVertexSamplerState: samp atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->vsamp[idx];

    if (sh->obj == (__bridge void *)samp && sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setVertexSamplerState: samp atIndex: idx];
    sh->obj = (__bridge void *)samp;
    sh->offset = 0;
    sh->gen = m->encoder_gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

- (void) compatSetFragmentSamplerState: (id<MTLSamplerState>)samp atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !enc || !samp || idx >= MGLC_BIND_SLOTS)
    {
        if (enc) [enc setFragmentSamplerState: samp atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->fsamp[idx];

    if (sh->obj == (__bridge void *)samp && sh->gen == gen && sh->encoder == (__bridge void *)enc)
    {
        mgl_compat_bind_skip++;
        return;
    }

    [enc setFragmentSamplerState: samp atIndex: idx];
    sh->obj = (__bridge void *)samp;
    sh->offset = 0;
    sh->gen = m->encoder_gen;
    sh->encoder = (__bridge void *)enc;
    mgl_compat_bind_set++;
}

/* -------- streaming upload ring ------------------------------------- */

/* lazily create the current ring slot; returns its CPU base or NULL */
- (uint8_t *) compatRingEnsureSlot: (struct MGLExtMetal_compat *) m
{
    unsigned cur = m->ring.cur;

    if (m->ring.slot[cur])
        return m->ring.cpu[cur];

    if (!_device)
        return NULL;

    id<MTLBuffer> b = [_device newBufferWithLength: MGLC_RING_SLOT_BYTES
                                           options: MTLResourceStorageModeShared];
    if (!b)
        return NULL;

    b.label = [NSString stringWithFormat: @"MGL arg ring %u", cur];
    m->ring.slot[cur] = (void *)CFBridgingRetain(b);
    m->ring.cpu[cur] = (uint8_t *)b.contents;
    m->ring_on = 1;
    return m->ring.cpu[cur];
}

/* allocate len bytes (caller passes the payload size; the allocation is
 * rounded up to 256) in the ring.  Returns false when the payload must take
 * the fallback path (ring unusable, payload too large, or the only recyclable
 * slot is still being read by the command buffer currently encoding). */
- (bool) compatRingAlloc: (size_t) len
                  buffer: (id<MTLBuffer> __strong *) out_buf
                  offset: (NSUInteger *) out_off
                     cpu: (uint8_t **) out_cpu
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !m->enabled || len == 0 || len > MGLC_RING_MAX_BYTES)
        return false;

    size_t need = (len + 255) & ~(size_t)255;
    size_t cap = MGLC_RING_SLOT_BYTES - MGLC_RING_GUARD_BYTES;

    if (need > cap)
        return false;

    if (m->ring.used + need > cap)
    {
        /* current slot exhausted: recycle the next one, waiting for the
         * command buffer that last read it */
        unsigned next = (m->ring.cur + 1) % MGLC_RING_SLOTS;

        if (m->ring.slot[next])
        {
            uint64_t serial = m->ring.serial[next];

            if (serial >= [self extBufSerial])
                return false;       /* still being read by the encoder in flight: fall back */

            if (serial > [self extBufCompletedSerial])
                [self extBufWaitSerial: serial];
        }

        m->ring.cur = next;
        m->ring.used = 0;
    }

    uint8_t *cpu = [self compatRingEnsureSlot: m];

    if (!cpu)
        return false;

    *out_cpu = cpu + m->ring.used;
    *out_off = (NSUInteger)m->ring.used;
    *out_buf = (__bridge id<MTLBuffer>)m->ring.slot[m->ring.cur];

    m->ring.used += need;
    m->ring.serial[m->ring.cur] = [self extBufSerial];
    mgl_compat_ring_bytes += len;
    return true;
}

/* bytes-payload binds: default-uniform blocks, small client arrays, constant
 * vertex attributes.  Replaces per-draw setVertexBytes/setFragmentBytes with
 * a ring sub-allocation + offset-only encoder update, skipped entirely when
 * the bytes at the previously bound offset are unchanged. */
- (void) compatRingVertexBytes: (const void *)p length: (NSUInteger)len atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !m->enabled || !enc || idx >= MGLC_BIND_SLOTS ||
        len == 0 || len > MGLC_RING_MAX_BYTES)
    {
        [enc setVertexBytes: p length: len atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->vbuf[idx];
    bool same_enc = (sh->gen == gen && sh->encoder == (__bridge void *)enc);

    /* identical bytes already bound at this index: skip the encode call
     * (the ring region still holds them, see mgl_compat_perf.h) */
    if (same_enc && sh->obj == m->ring.slot[m->ring.cur] &&
        memcmp(m->ring.cpu[m->ring.cur] + sh->offset, p, len) == 0)
    {
        mgl_compat_bind_skip++;
        return;
    }

    id<MTLBuffer> rb = nil;
    NSUInteger roff = 0;
    uint8_t *rc = NULL;

    if (![self compatRingAlloc: (size_t)len buffer: &rb offset: &roff cpu: &rc])
    {
        [enc setVertexBytes: p length: len atIndex: idx];
        mgl_compat_bind_set++;
        return;
    }

    memcpy(rc, p, len);

    bool offset_only = (same_enc && sh->obj == (__bridge void *)rb);

    sh->obj = (__bridge void *)rb;
    sh->offset = (size_t)roff;
    sh->gen = gen;
    sh->encoder = (__bridge void *)enc;

    if (offset_only)
        [enc setVertexBufferOffset: roff atIndex: idx];
    else
        [enc setVertexBuffer: rb offset: roff atIndex: idx];
    mgl_compat_bind_set++;
}

- (void) compatRingFragmentBytes: (const void *)p length: (NSUInteger)len atIndex: (NSUInteger)idx
{
    id<MTLRenderCommandEncoder> enc = _currentRenderEncoder;
    struct MGLExtMetal_compat *m = compatMetal(ctx);

    if (!m || !m->enabled || !enc || idx >= MGLC_BIND_SLOTS ||
        len == 0 || len > MGLC_RING_MAX_BYTES)
    {
        [enc setFragmentBytes: p length: len atIndex: idx];
        return;
    }

    uint64_t gen = compatEncGen(m, (__bridge void *)enc);
    struct MGLCompatBindShadow *sh = &m->fbuf[idx];
    bool same_enc = (sh->gen == gen && sh->encoder == (__bridge void *)enc);

    if (same_enc && sh->obj == m->ring.slot[m->ring.cur] &&
        memcmp(m->ring.cpu[m->ring.cur] + sh->offset, p, len) == 0)
    {
        mgl_compat_bind_skip++;
        return;
    }

    id<MTLBuffer> rb = nil;
    NSUInteger roff = 0;
    uint8_t *rc = NULL;

    if (![self compatRingAlloc: (size_t)len buffer: &rb offset: &roff cpu: &rc])
    {
        [enc setFragmentBytes: p length: len atIndex: idx];
        mgl_compat_bind_set++;
        return;
    }

    memcpy(rc, p, len);

    bool offset_only = (same_enc && sh->obj == (__bridge void *)rb);

    sh->obj = (__bridge void *)rb;
    sh->offset = (size_t)roff;
    sh->gen = gen;
    sh->encoder = (__bridge void *)enc;

    if (offset_only)
        [enc setFragmentBufferOffset: roff atIndex: idx];
    else
        [enc setFragmentBuffer: rb offset: roff atIndex: idx];
    mgl_compat_bind_set++;
}

/* copied index lists (unaligned element-buffer offsets / CPU pointers):
 * stream the indices through the ring instead of allocating an MTLBuffer per
 * draw.  Returns nil when the caller must take its old path. */
- (id<MTLBuffer>) compatRingIndexBytes: (const void *)p length: (NSUInteger)len offset: (NSUInteger *)out_off
{
    id<MTLBuffer> rb = nil;
    uint8_t *rc = NULL;

    if (len == 0 || len > MGLC_RING_MAX_BYTES)
        return nil;

    if (![self compatRingAlloc: (size_t)len buffer: &rb offset: out_off cpu: &rc])
        return nil;

    memcpy(rc, p, len);
    return rb;
}

/* -------- profiler --------------------------------------------------- */

/* MGL_PROFILE=2 dumps these with the per-draw profile (mgl_profile_dump in
 * MGLRenderer.m) */
- (void) compatBindProfileDump
{
    fprintf(stderr, "MGL PROFILE binds: set=%llu skipped=%llu ring_bytes=%llu\n",
            (unsigned long long)mgl_compat_bind_set, (unsigned long long)mgl_compat_bind_skip,
            (unsigned long long)mgl_compat_ring_bytes);
}

- (void) compatBindProfileReset
{
    mgl_compat_bind_set = mgl_compat_bind_skip = mgl_compat_ring_bytes = 0;
}

- (bool) compatSurfacePresentEnabled
{
    /* windowed renderers only (a layer exists) and explicitly opted in */
    if (_layer == nil)
        return false;
    /* Only a POSITIVE result is cached.  A call that happens before the
     * embedding layer (NSOpenGL/GLX shim) sets MGL_SURFACE_PRESENT — e.g.
     * during library init, when a renderer already owns an offscreen layer —
     * must not latch "disabled" for the lifetime of the process: every swap
     * would then fall through to -[CAMetalLayer nextDrawable] and pin the
     * frame rate to the display refresh (glmark2 regression mglR7/mglR8).
     * Re-checking while unset is one getenv per call, only at swap/ensure. */
    static int enabled = -1;
    if (enabled != 1)
    {
        const char *e = getenv("MGL_SURFACE_PRESENT");
        enabled = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

/* (re)create the surface pool for w x h */
- (MGLCompatSurfacePool *) compatSurfacePoolW:(NSUInteger)w h:(NSUInteger)h
{
    MGLCompatSurfacePool *pool = objc_getAssociatedObject(self, kCompatPoolKey);
    if (pool && pool.w == w && pool.h == h && pool.textures.count == COMPAT_SURFACES)
        return pool;
    if (!w || !h || !_device)
        return nil;

    if (!pool)
    {
        pool = [[MGLCompatSurfacePool alloc] init];
        pool.textures = [NSMutableArray array];
        pool.surfaces = [NSMutableArray array];
        objc_setAssociatedObject(self, kCompatPoolKey, pool, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    /* pixel format: mirror the drawable path (ctx->pixel_format), mapped to an
     * IOSurface fourcc; anything else falls back to BGRA */
    MTLPixelFormat pf = ctx ? ctx->pixel_format.mtl_pixel_format : MTLPixelFormatBGRA8Unorm;
    uint32_t fourcc = 'BGRA';
    if (pf == MTLPixelFormatRGBA8Unorm || pf == MTLPixelFormatRGBA8Unorm_sRGB)
        fourcc = 'RGBA';

    [pool.textures removeAllObjects];
    [pool.surfaces removeAllObjects];
    for (int i = 0; i < COMPAT_SURFACES; i++)
    {
        NSDictionary *d = @{
            (id)kIOSurfaceWidth: @(w),
            (id)kIOSurfaceHeight: @(h),
            (id)kIOSurfaceBytesPerElement: @4,
            (id)kIOSurfaceBytesPerRow: @((w * 4 + 63) & ~63ULL),
            (id)kIOSurfacePixelFormat: @(fourcc),
        };
        IOSurfaceRef s = IOSurfaceCreate((CFDictionaryRef)d);
        if (!s)
            break;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                      width:w height:h mipmapped:NO];
        td.storageMode = MTLStorageModeShared;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        id<MTLTexture> t = [_device newTextureWithDescriptor:td iosurface:s plane:0];
        if (!t)
        {
            CFRelease(s);
            break;
        }
        t.label = [NSString stringWithFormat:@"MGL present surface %d", i];
        [pool.textures addObject:t];
        [pool.surfaces addObject:(__bridge id)s];
        CFRelease(s);       /* the array owns it now */
    }
    if (pool.textures.count != COMPAT_SURFACES)
    {
        [pool.textures removeAllObjects];
        [pool.surfaces removeAllObjects];
        return nil;
    }
    pool.w = w;
    pool.h = h;
    pool.next = 0;

    /* the presenting layer rides on top of the (never-presented, transparent)
     * Metal layer; keep it exactly over it and stretch the surface.  The GL
     * framebuffer is bottom-up and the blit below is a straight copy (no
     * shader pass just to flip rows), so the layer itself is flipped once. */
    if (!pool.layer)
    {
        pool.layer = [[CALayer alloc] init];
        pool.layer.contentsGravity = kCAGravityResize;
        pool.layer.contentsScale = _layer.contentsScale;
        pool.layer.masksToBounds = YES;
        pool.layer.transform = CATransform3DMakeScale(1, -1, 1);   /* GL bottom-up -> upright */
        [_layer addSublayer:pool.layer];
    }
    pool.layer.frame = _layer.bounds;
    return pool;
}

/* present the default framebuffer through an IOSurface: returns true when the
 * frame was handled (the caller must not run the drawable path afterwards) */
/* the present-pool surface the default framebuffer renders into (direct path):
 * renders straight into the IOSurface-backed texture that compatSurfacePresent
 * shows, so the swap is only a layer.contents change — no blit.  The write slot
 * advances lazily HERE (after a present set `rotate`), so a readback between the
 * present and the next draw still sees the surface that was just shown. */
- (id<MTLTexture>) compatSurfaceWrite:(NSUInteger)w h:(NSUInteger)h
{
    /* OFF by default: rendering directly into the CA-displayed IOSurface pins the
     * frame rate to the display refresh (every glmark2 scene 120-121 FPS in a
     * visible window — cross-process surface sync at commit; score 119 vs 10918).
     * The blit path (private render texture + one blit encoder per swap) is the
     * default; MGL_DIRECT_PRESENT=1 re-enables the experimental direct path. */
    static int env_direct = -1;
    if (env_direct < 0) env_direct = getenv("MGL_DIRECT_PRESENT") ? atoi(getenv("MGL_DIRECT_PRESENT")) : 0;
    if (!env_direct) return nil;

    MGLCompatSurfacePool *pool = [self compatSurfacePoolW:w h:h];
    if (!pool) return nil;
    if (pool.rotate)
    {
        pool.write = (pool.write + 1) % COMPAT_SURFACES;
        pool.rotate = NO;
    }
    return pool.textures.count == COMPAT_SURFACES ? pool.textures[pool.write] : nil;
}

- (bool) compatSurfacePresent:(id<MTLTexture>)color
{
    uint64_t p6 = mgl_prof_begin(6);
    MGLCompatSurfacePool *pool = [self compatSurfacePoolW:color.width h:color.height];
    if (!pool)
        return false;

    /* direct mode: the default framebuffer already rendered into this exact pool
     * texture (fbEnsureDefaultFramebuffer handed out the write slot) — present it
     * as-is: no blit encoder, just the contents swap.  `rotate` advances the write
     * slot at the NEXT fbEnsureDefaultFramebuffer, keeping readback-after-present
     * correct. */
    unsigned di = COMPAT_SURFACES;
    for (unsigned k = 0; k < COMPAT_SURFACES; k++)
        if (pool.textures[k] == color) { di = k; break; }
    if (di < COMPAT_SURFACES)
    {
        [self commitCommandBufferWithAGXRecovery:_currentCommandBuffer];

        if (!CGRectEqualToRect(pool.layer.frame, _layer.bounds))
            pool.layer.frame = _layer.bounds;
        pool.layer.contents = pool.surfaces[di];
        pool.rotate = YES;

        [self newCommandBuffer];
        mgl_prof_end(6, p6);
        mgl_prof_frame();
        return true;
    }

    unsigned i = pool.next;
    pool.next = (pool.next + 1) % COMPAT_SURFACES;
    id<MTLTexture> dst = pool.textures[i];

    /* straight copy with a blit encoder — no render pass, no pipeline, no
     * viewport, no shader bind (the shader-blit present path costs a render
     * encoder per swap; the y-flip lives in the layer transform instead) */
    if (color.width == dst.width && color.height == dst.height &&
        color.pixelFormat == dst.pixelFormat)
    {
        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        [blit copyFromTexture:color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(color.width, color.height, 1)
                    toTexture:dst destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
    }
    else
    {
        /* format or size change (window resize race): shader-blit WITHOUT the
         * classic y-flip — the presenting layer's transform already flips, a
         * flipped copy would land upside down on screen */
        CompatFBTarget src = { color, 0, 0, 0, 1 }, dt = { dst, 0, 0, 0, 1 };
        GLint s[4] = { 0, 0, (GLint)color.width, (GLint)color.height };
        GLint d[4] = { 0, 0, (GLint)dst.width, (GLint)dst.height };
        [self fbShaderBlitFrom:&src to:&dt src:s dst:d
                        filter:GL_LINEAR
                       scissor:false];
    }

    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    [self commitCommandBufferWithAGXRecovery:cb];

    /* contents swap is the "present": cheap, no server round trip; the
     * compositor shows this surface at its next frame */
    pool.layer.frame = _layer.bounds;
    pool.layer.contents = pool.surfaces[i];

    [self newCommandBuffer];
    mgl_prof_end(6, p6);
    mgl_prof_frame();
    return true;
}

@end
