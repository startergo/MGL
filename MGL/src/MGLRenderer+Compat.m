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
#import "mgl_ext_vattrib.h"           /* draw plan / VAO layout snapshots (session 12) */
#import "mgl_ext_texext.h"            /* mgl_texext_info (session 12 texture gate) */
#include "mgl_compat_perf.h"

/* MGL_PROFILE counters (MGLRenderer.m) */
extern uint64_t mgl_prof_ns[18];
extern uint64_t mgl_prof_cnt[18];
/* MGLRenderer.m (perf session 12): installed PreDraw: hook count */
extern unsigned mglExtHookCacheCount(id self, const char *suffix);

/* draw-walk pieces (MGLRenderer+Draw.m / MGLRenderer.m) reused by the partial
 * fast path; declared here, not in the shared private header */
@interface MGLRenderer (CompatDrawWalk)
- (bool) extBufBindList: (BufferMapList *)list vertex: (bool)vertex;
- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list;
- (void) extRstatePreDraw: (id<MTLRenderCommandEncoder>) enc;   /* MGLRenderer+RenderState.m */
@end

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
/* compat perf session 12: per-draw state-walk fast-path hit/miss counters */
uint64_t mgl_compat_fast_hit, mgl_compat_fast_miss;

/* ===================== per-draw state-walk fast path =====================
 * (compat perf session 12)
 *
 * Apple's GL driver accumulates each GL draw and re-validates only the state
 * groups that changed since the previous draw.  MGL's per-draw walk —
 * processGLState, the three PreDraw: hooks (vattrib / rstate / texext),
 * extBufBindList, geomBindSSBOSizes and the end-of-draw GPU-serial stamps —
 * re-reads every group on EVERY draw even when nothing changed: glmark2
 * [ideas] draws 353x per frame, ~265 of those with zero dirty bits, and that
 * walk was ~2 us/draw, the largest remaining per-draw CPU cost.
 *
 * The fast path serves such draws from exact-content snapshots taken right
 * after a full walk: every state group the walk reads is recorded (identity
 * AND content), and the next draw re-verifies those groups with integer
 * compares + memcmps.  If everything matches, the entire walk is a proven
 * no-op and is skipped — no ObjC dispatch, no Metal calls.  Because the check
 * is content-based (not generation-based), a state change can never be missed
 * by a missing invalidation; anything the check cannot prove stable (mapped
 * buffers, client-array streams, buffer textures, extra PreDraw hooks)
 * declines to the full walk.
 */

struct MGLCompatBufId {
    const void *buf;
    void       *mtl;        /* buf->data.mtl_data identity */
    GLsizeiptr  size;
};

struct MGLCompatDrawSnap {
    uint64_t    valid;
    /* identity of the objects the walk resolved state for */
    uint64_t    encoder_gen;            /* m->encoder_gen (bind-shadow generation) */
    const void *encoder, *cmdbuf, *pipe, *program, *vao, *fbo;
    GLuint      rstate_gen;             /* ext_rstate->generation (encoder-state group) */
    GLuint      rstate_pipe_gen;        /* ext_rstate->generation_pipeline (PSO content) */
    ImageUnit   image_units[TEXTURE_UNITS];     /* glBindImageTexture group */

    /* content snapshots */
    unsigned    tex_mask[4];            /* ctx->state.active_texture_mask */
    BufferMapList vlist, flist;         /* vertex / fragment buffer maps (identity) */
    struct MGLCompatBufId vb[MAX_MAPPED_BUFFERS], fb[MAX_MAPPED_BUFFERS];
    MGLVattribDrawPlan plan;            /* vattrib draw plan */
    MGLVertexAttribFormat fmt[MAX_ATTRIBS];
    MGLVertexBinding binding[MGL_VA_MAX_BINDINGS];
    struct MGLCompatBufId vabuf[MGL_VA_MAX_BINDINGS];
    MGLCurrentAttrib current[MAX_ATTRIBS];      /* generic vertex attribs (const streams) */
    const void *tex[TEXTURE_UNITS], *samp[TEXTURE_UNITS];
    void *tex_mtl[TEXTURE_UNITS], *tex_par[TEXTURE_UNITS], *samp_par[TEXTURE_UNITS];
};

struct MGLCompatDrawFast {
    struct MGLCompatDrawSnap snap;

    /* end-of-draw gate: the serial stamps are a no-op while the command-buffer
     * serial, the stamped buffer set and the write-marker inputs are unchanged */
    uint64_t    end_valid;
    uint64_t    end_serial;
    const void *end_ebuf, *end_indirect, *end_tf;
    int         end_tf_active, end_tf_paused;
    const void *end_tf_base[MAX_BINDABLE_BUFFERS];
    struct MGLCompatBufId end_v[MAX_MAPPED_BUFFERS], end_f[MAX_MAPPED_BUFFERS];
    GLuint      end_vn, end_fn;

    /* MGL_DRAWFAST_DEBUG: miss-reason histogram (index = miss label below) */
    uint64_t    misswhy[24];
    uint64_t    dirtymask_seen[8];   /* MGL_DRAWFAST_DEBUG: distinct dirty_bits values */
    uint64_t    dirtymask_val[8];
};

static struct MGLCompatDrawFast *compatDrawFastState(GLMContext ctx)
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m)
        return NULL;
    if (!m->draw_fast)
        m->draw_fast = calloc(1, sizeof(struct MGLCompatDrawFast));
    return (struct MGLCompatDrawFast *)m->draw_fast;
}

/* MGL_DRAWFAST_DEBUG: most recent fast-path state (single-renderer processes) */
static struct MGLCompatDrawFast *s_compat_df_debug;
static const char *const s_compat_df_why[] = {
    "no-snap", "dirty-bits", "hook-count", "enc/cb/pipe-id", "cb-committed",
    "prog/vao/fbo-id", "rstate-gen", "vlist", "vlist-dirty", "flist",
    "flist-dirty", "no-vattrib", "plan", "vao-fmt/binding", "current-attribs",
    "va-mapped", "va-dirty", "va-mtl-id", "tex-mask",
    "tex/samp-id", "tex-dirty", "tex-mtl/params", "samp-params"
};

void mgl_compat_drawfast_debug(void)
{
    struct MGLCompatDrawFast *df = s_compat_df_debug;
    if (!df)
        return;
    fprintf(stderr, "MGL PROFILE drawfast-miss:");
    for (unsigned i = 0; i < sizeof(s_compat_df_why) / sizeof(s_compat_df_why[0]); i++)
        if (df->misswhy[i])
            fprintf(stderr, " %s=%llu", s_compat_df_why[i], (unsigned long long)df->misswhy[i]);
    fprintf(stderr, "\n");
    for (unsigned i = 0; i < 8; i++)
        if (df->dirtymask_seen[i])
            fprintf(stderr, "MGL PROFILE drawfast-dirty: 0x%llx x%llu\n",
                    (unsigned long long)df->dirtymask_val[i], (unsigned long long)df->dirtymask_seen[i]);
    memset(df->misswhy, 0, sizeof(df->misswhy));
    memset(df->dirtymask_seen, 0, sizeof(df->dirtymask_seen));
}

/* per-listed-buffer stability: not mapped, no pending content change, same
 * Metal storage.  Content changes (glBufferSubData / glUniform into the
 * default uniform block) set data.dirty_bits and force the full walk, which
 * re-streams them through the argument ring. */
static bool compatListStable(const BufferMapList *live, const BufferMapList *snap,
                             const struct MGLCompatBufId *snapbuf, int *why, bool allow_dirty)
{
    if (live->count != snap->count)
    { *why = 0; return false; }
    if (live->count && memcmp(live->buffers, snap->buffers, live->count * sizeof(BufferMap)) != 0)
    { *why = 1; return false; }

    for (GLuint i = 0; i < live->count; i++)
    {
        const Buffer *b = live->buffers[i].buf;

        if (!b)
            continue;
        if (b->mapped)                    /* coherent / persistent maps need per-draw work */
        { *why = 2; return false; }
        if (!allow_dirty && b->data.dirty_bits)   /* content change: full walk re-streams it */
        { *why = 3; return false; }
        if (b->data.mtl_data != snapbuf[i].mtl || b->size != snapbuf[i].size)
        { *why = 4; return false; }
    }

    return true;
}

/* after the partial (buffer-only) walk re-streamed the by-value buffers, clear
 * their content-dirty bits the way bindVertexBuffersToCurrentRenderEncoder does */
static void compatListClearDataDirty(BufferMapList *live)
{
    for (GLuint i = 0; i < live->count; i++)
    {
        Buffer *b = live->buffers[i].buf;
        if (b && b->data.mtl_data == NULL)
            b->data.dirty_bits &= ~DIRTY_BUFFER_DATA;
    }
}

static void compatListLatch(const BufferMapList *live, BufferMapList *snap,
                            struct MGLCompatBufId *snapbuf)
{
    snap->count = live->count;
    if (live->count)
        memcpy(snap->buffers, live->buffers, live->count * sizeof(BufferMap));

    for (GLuint i = 0; i < live->count; i++)
    {
        const Buffer *b = live->buffers[i].buf;

        snapbuf[i].buf  = b;
        snapbuf[i].mtl  = b ? b->data.mtl_data : NULL;
        snapbuf[i].size = b ? b->size : 0;
    }
}

/* true when the whole per-draw walk (processGLState + PreDraw hooks +
 * extBufBindList + geomBindSSBOSizes) is provably a no-op for this draw */
/* ctx dirty bits whose processing the content gates below prove to be a no-op:
 * VAO re-specified with identical layout / texture-unit rebinds that changed
 * nothing / buffer-base rebinds that remap to the same list / image-unit
 * writes with identical units (the sampler uniform that drove them lives in a
 * listed buffer, so a real change shows up as a listed-buffer dirty bit) */
#define MGL_DF_REDUNDANT (DIRTY_VAO | DIRTY_TEX_BINDING | DIRTY_BUFFER_BASE_STATE | DIRTY_IMAGE_UNIT_STATE)

- (int) compatDrawGate: (bool) allowDirtyListBuffers
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;
    const struct MGLCompatDrawSnap *s;

    if (!df)
    {
        df = compatDrawFastState(ctx);
        if (df)
            s_compat_df_debug = df;
    }

#define MGL_DF_MISS(_why_) do { if (df) df->misswhy[_why_]++; mgl_compat_fast_miss++; return MGL_DF_FULL; } while (0)

    if (!df || !(s = &df->snap)->valid || !m->enabled)
        MGL_DF_MISS(0);

    /* only dirty bits whose processing is proven redundant by the content
     * gates below may stand (they are cleared on a SKIP / BUFFERS verdict) */
    if (ctx->state.dirty_bits & ~MGL_DF_REDUNDANT)
    {
        if (df)
        {
            for (int i = 0; i < 8; i++)
            {
                if (df->dirtymask_val[i] == ctx->state.dirty_bits || !df->dirtymask_val[i])
                {
                    df->dirtymask_val[i] = ctx->state.dirty_bits;
                    df->dirtymask_seen[i]++;
                    break;
                }
            }
        }
        MGL_DF_MISS(1);
    }
    /* ...and only for the three PreDraw hooks this gate reasons about */
    if (mglExtHookCacheCount(self, "PreDraw:") != 3)
        MGL_DF_MISS(2);
    /* identity: encoder (incl. the bind-shadow generation), command buffer
     * (still uncommitted — commit paths set dirty state), pipeline (already
     * bound to this encoder by the latched walk), program / VAO / FBO */
    if (m->encoder_gen != s->encoder_gen ||
        (__bridge const void *)_currentRenderEncoder != s->encoder ||
        (__bridge const void *)_currentCommandBuffer != s->cmdbuf ||
        (__bridge const void *)_pipelineState != s->pipe)
        MGL_DF_MISS(3);
    if (_currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        MGL_DF_MISS(4);
    if ((const void *)ctx->state.program != s->program ||
        (const void *)ctx->state.vao != s->vao ||
        (const void *)ctx->state.framebuffer != s->fbo)
        MGL_DF_MISS(5);

    /* encoder-state group (rstate): applied for this generation.  A generation
     * bump without a pipeline-content change needs only the encoder-state
     * re-apply (extRstatePreDraw) — flagged here, returned after the texture
     * epoch is stamped below. */
    bool rstate_only = false;
    {
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        if (!rs)
            MGL_DF_MISS(6);
        if (rs->generation != s->rstate_gen)
        {
            if (rs->generation_pipeline != s->rstate_pipe_gen)
                MGL_DF_MISS(6);
            rstate_only = true;
        }
    }

    /* texture group: unit bindings, the realized Metal texture / view / sampler
     * ids behind each bound unit, and no pending texture work.  Buffer textures
     * and texture views follow state this gate cannot see: decline.
     *
     * Every path that can change any of this (glBindTexture, glSamplerParameter,
     * glTexParameter, texture upload / realloc / delete, sampler-uniform writes)
     * arms one of these bits, so a clean draw needs no texture check at all.
     * The verdict also feeds the texext PreDraw skip epoch (mgl_compat_perf.h):
     * a VAO-changed draw with clean texture state skips the sampler-binding
     * walk in extTexextPreDraw. */
    {
        bool tex_clean = !(ctx->state.dirty_bits & (DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING |
                                                    DIRTY_SAMPLER | DIRTY_IMAGE_UNIT_STATE | DIRTY_PROGRAM));
        if (!tex_clean)
        {
            if (memcmp(ctx->state.image_units, s->image_units, sizeof(s->image_units)) != 0)
                MGL_DF_MISS(24);
            if (memcmp(ctx->state.active_texture_mask, s->tex_mask, sizeof(s->tex_mask)) != 0)
                MGL_DF_MISS(19);
            for (int w = 0; w < 4; w++)
            {
                unsigned bits = ctx->state.active_texture_mask[w];
                while (bits)
                {
                    int u = w * 32 + __builtin_ctz(bits);
                    Texture *tex = ctx->state.active_textures[u];
                    Sampler *sam = ctx->state.texture_samplers[u];

                    bits &= bits - 1;
                    if ((const void *)tex != s->tex[u] || (const void *)sam != s->samp[u])
                        MGL_DF_MISS(20);
                    if (tex)
                    {
                        if (tex->dirty_bits)
                            MGL_DF_MISS(21);
                        if (tex->mtl_data != s->tex_mtl[u] || tex->params.mtl_data != s->tex_par[u] ||
                            mgl_texext_info(ctx, tex))
                            MGL_DF_MISS(22);
                    }
                    if (sam && sam->params.mtl_data != s->samp_par[u])
                        MGL_DF_MISS(23);
                }
            }
        }
        m->texext_epoch_seq++;                                /* serial of this gate run */
        /* consecutive clean draws share one epoch: the hook skips until a
         * texture-dirty draw (gate or walk) zeroes it, then runs once */
        if (!tex_clean)
            m->texext_skip_epoch = 0;
        else if (m->texext_skip_epoch == 0)
            m->texext_skip_epoch = ++m->texext_epoch_seq;
    }

    /* buffer-map groups: identity of the maps and of every mapped buffer */
    { int wv = 0, wf = 0;
      bool vok = compatListStable(&ctx->state.vertex_buffer_map_list, &s->vlist, s->vb, &wv, allowDirtyListBuffers);
      bool fok = vok && compatListStable(&ctx->state.fragment_buffer_map_list, &s->flist, s->fb, &wf, allowDirtyListBuffers);
      if (!vok) MGL_DF_MISS(7 + (wv == 3));           /* vlist / vlist-dirty */
      else if (!fok) MGL_DF_MISS(9 + (wf == 3));      /* flist / flist-dirty */
    }

    /* vertex-array group: the vattrib draw plan, the VAO layout, the generic
     * attribs (constant-attribute streams), and the Metal storage behind every
     * binding the plan uses (client-array-style by-value streams decline).
     * plan / fmt / binding can only be written behind DIRTY_VAO (core) or —
     * in the compat layout-hash fast path — not at all, so a clean draw only
     * pays the generic-attrib compare (glVertexAttrib* sets no dirty bit) */
    {
        struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
        MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

        if (!vs || !va)
            MGL_DF_MISS(11);
        if (ctx->state.dirty_bits & DIRTY_VAO)
        {
            if (memcmp(&vs->plan, &s->plan, sizeof(s->plan)) != 0)
                MGL_DF_MISS(12);
            if (memcmp(va->fmt, s->fmt, sizeof(s->fmt)) != 0 ||
                memcmp(va->binding, s->binding, sizeof(s->binding)) != 0)
                MGL_DF_MISS(13);
        }
        if (memcmp(vs->current, s->current, sizeof(s->current)) != 0)
            MGL_DF_MISS(14);

        for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
        {
            const Buffer *buf = va->binding[b].buffer;

            if (s->plan.binding_index[b] < 0 || !buf)
                continue;
            /* by-value (small, no MTLBuffer) streams re-stream through the ring,
             * which memcmp-skips unchanged bytes: content changes always set
             * data.dirty_bits (BufferSubData / Map/Unmap) or leave buf->mapped
             * set (persistent), so a clean, unmapped, unchanged binding needs
             * no re-stream — the encoder keeps its ring binding, and ring
             * slots are only recycled after their reading serial completed */
            if (buf->mapped)
                MGL_DF_MISS(16);
            if (buf->data.dirty_bits)
                MGL_DF_MISS(17);
            if ((buf->data.mtl_data != NULL) != (s->vabuf[b].mtl != NULL) ||
                buf->size != s->vabuf[b].size ||
                (buf->data.mtl_data && buf->data.mtl_data != s->vabuf[b].mtl))
                MGL_DF_MISS(18);
        }
    }


    /* tess routing installs its own pipeline and buffer maps: never skip its walks */
    if ([self extTessActive])
        MGL_DF_MISS(25);

    /* everything matched: the walk is a no-op except for listed-buffer content
     * (a listed buffer with dirty data needs the ring re-stream) */
    if (rstate_only)
        return MGL_DF_RSTATE;

    mgl_compat_fast_hit++;
    if (ctx->state.dirty_bits & MGL_DF_REDUNDANT)
        ctx->state.dirty_bits &= ~MGL_DF_REDUNDANT;

    if (allowDirtyListBuffers)
    {
        bool dirty = false;
        for (GLuint i = 0; i < ctx->state.vertex_buffer_map_list.count && !dirty; i++)
        {
            const Buffer *b = ctx->state.vertex_buffer_map_list.buffers[i].buf;
            dirty = b && b->data.dirty_bits != 0;
        }
        for (GLuint i = 0; i < ctx->state.fragment_buffer_map_list.count && !dirty; i++)
        {
            const Buffer *b = ctx->state.fragment_buffer_map_list.buffers[i].buf;
            dirty = b && b->data.dirty_bits != 0;
        }
        if (dirty)
            return MGL_DF_BUFFERS;
    }

    return MGL_DF_SKIP;
}

/* record the state the just-completed full walk resolved everything for */
- (void) compatDrawLatch
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;
    struct MGLCompatDrawSnap *s;

    if (!df)
        return;
    s = &df->snap;

    /* the walk must have produced a bound pipeline on a live encoder */
    if (!_currentRenderEncoder || !_pipelineState || !_currentCommandBuffer)
    {
        s->valid = 0;
        return;
    }

    s->valid = 1;
    s->encoder_gen = m->encoder_gen;
    s->encoder = (__bridge const void *)_currentRenderEncoder;
    s->cmdbuf = (__bridge const void *)_currentCommandBuffer;
    s->pipe = (__bridge const void *)_pipelineState;
    s->program = (const void *)ctx->state.program;
    s->vao = (const void *)ctx->state.vao;
    s->fbo = (const void *)ctx->state.framebuffer;

    {
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        s->rstate_gen = rs ? rs->generation : 0;
        s->rstate_pipe_gen = rs ? rs->generation_pipeline : 0;
    }

    memcpy(s->image_units, ctx->state.image_units, sizeof(s->image_units));

    compatListLatch(&ctx->state.vertex_buffer_map_list, &s->vlist, s->vb);
    compatListLatch(&ctx->state.fragment_buffer_map_list, &s->flist, s->fb);

    {
        struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
        MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

        if (vs && va)
        {
            memcpy(&s->plan, &vs->plan, sizeof(s->plan));
            memcpy(s->fmt, va->fmt, sizeof(s->fmt));
            memcpy(s->binding, va->binding, sizeof(s->binding));
            memcpy(s->current, vs->current, sizeof(s->current));
            for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
            {
                const Buffer *buf = va->binding[b].buffer;
                s->vabuf[b].buf = buf;
                s->vabuf[b].mtl = buf ? buf->data.mtl_data : NULL;
                s->vabuf[b].size = buf ? buf->size : 0;
            }
        }
        else
        {
            memset(&s->plan, 0, sizeof(s->plan));
            memset(s->fmt, 0, sizeof(s->fmt));
            memset(s->binding, 0, sizeof(s->binding));
            memset(s->current, 0, sizeof(s->current));
            memset(s->vabuf, 0, sizeof(s->vabuf));
        }
    }

    memcpy(s->tex_mask, ctx->state.active_texture_mask, sizeof(s->tex_mask));
    for (int w = 0; w < 4; w++)
    {
        unsigned bits = ctx->state.active_texture_mask[w];
        while (bits)
        {
            int u = w * 32 + __builtin_ctz(bits);
            Texture *tex = ctx->state.active_textures[u];
            Sampler *sam = ctx->state.texture_samplers[u];

            bits &= bits - 1;
            s->tex[u] = tex;
            s->samp[u] = sam;
            s->tex_mtl[u] = tex ? tex->mtl_data : NULL;
            s->tex_par[u] = tex ? tex->params.mtl_data : NULL;
            s->samp_par[u] = sam ? sam->params.mtl_data : NULL;
        }
    }

    /* the end-of-draw snapshot stays valid: it records "what was last stamped";
     * if this walk changed the stamped set, the end gate's own compares fail */
}

/* partial fast path (perf session 12): every group except listed-buffer
 * CONTENT matches the snapshot — e.g. only a uniform upload dirtied the
 * default uniform block.  Re-stream just the buffers (the ring re-encodes
 * changed bytes, memcmp-skips unchanged ones) and skip the whole walk: no
 * PSO check, no hook fan-out, no stamps (their gates still match). */
- (bool) compatDrawFastBuffers
{
    uint64_t t13 = mgl_prof_begin(13);

    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list] ||
        ![self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list])
    {
        mgl_prof_end(13, t13);
        return false;
    }

    if (![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true] ||
        ![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
    {
        mgl_prof_end(13, t13);
        return false;
    }

    compatListClearDataDirty(&ctx->state.vertex_buffer_map_list);
    compatListClearDataDirty(&ctx->state.fragment_buffer_map_list);

    /* identities did not change, the content dirty bits are now clear: the
     * snapshot is valid again for the next draw */
    [self compatDrawLatch];

    mgl_prof_end(13, t13);
    return true;
}

/* partial fast path (perf session 12): only the rstate encoder-state group
 * changed (generation bump, no pipeline-content change — e.g. a blend-colour /
 * stencil-ref / depth-mask write between draws).  Re-apply the encoder state
 * and skip the rest of the walk. */
- (bool) compatDrawFastRState
{
    [self extRstatePreDraw: _currentRenderEncoder];
    [self compatDrawLatch];
    return true;
}

/* ---- end-of-draw: the GPU-serial stamps are a no-op while the command-buffer
 * serial and every stamped buffer are unchanged ---- */
- (bool) compatEndDrawFast
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;

    if (!df || !df->end_valid || !m->enabled)
        return false;

    if ([self extBufSerial] != df->end_serial)
        return false;

    const BufferMapList *lists[2] = { &ctx->state.vertex_buffer_map_list, &ctx->state.fragment_buffer_map_list };
    const struct MGLCompatBufId *snap[2] = { df->end_v, df->end_f };
    const GLuint counts[2] = { df->end_vn, df->end_fn };

    for (int l = 0; l < 2; l++)
    {
        if (lists[l]->count != counts[l])
            return false;
        for (GLuint i = 0; i < lists[l]->count; i++)
        {
            const Buffer *b = lists[l]->buffers[i].buf;

            if (b != snap[l][i].buf || (b && b->data.mtl_data != snap[l][i].mtl))
                return false;
        }
    }

    if (VAO() && (const void *)VAO_STATE(element_array.buffer) != df->end_ebuf)
        return false;
    if ((const void *)ctx->state.buffers[_DRAW_INDIRECT_BUFFER] != df->end_indirect)
        return false;

    {
        TransformFeedback *tf = ctx->state.transform_feedback;
        if ((const void *)tf != df->end_tf)
            return false;
        if (tf && tf->active && !tf->paused)
        {
            if (!df->end_tf_active || df->end_tf_paused)
                return false;
            for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++)
                if ((const void *)ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[i].buf != df->end_tf_base[i])
                    return false;
        }
        else if (df->end_tf_active && !df->end_tf_paused)
            return false;
    }

    mgl_compat_fast_hit++;
    return true;
}

- (void) compatEndDrawLatch
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;

    if (!df)
        return;

    df->end_valid = 1;
    df->end_serial = [self extBufSerial];

    const BufferMapList *lists[2] = { &ctx->state.vertex_buffer_map_list, &ctx->state.fragment_buffer_map_list };
    struct MGLCompatBufId *snap[2] = { df->end_v, df->end_f };
    GLuint *counts[2] = { &df->end_vn, &df->end_fn };

    for (int l = 0; l < 2; l++)
    {
        *counts[l] = lists[l]->count;
        for (GLuint i = 0; i < lists[l]->count; i++)
        {
            const Buffer *b = lists[l]->buffers[i].buf;
            snap[l][i].buf = b;
            snap[l][i].mtl = b ? b->data.mtl_data : NULL;
            snap[l][i].size = b ? b->size : 0;
        }
    }

    df->end_ebuf = VAO() ? (const void *)VAO_STATE(element_array.buffer) : NULL;
    df->end_indirect = (const void *)ctx->state.buffers[_DRAW_INDIRECT_BUFFER];

    {
        TransformFeedback *tf = ctx->state.transform_feedback;
        df->end_tf = tf;
        df->end_tf_active = tf ? tf->active : 0;
        df->end_tf_paused = tf ? tf->paused : 0;
        for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++)
            df->end_tf_base[i] = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[i].buf;
    }
}

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

/* per-draw texture RESOLUTION cache (compat perf session 8; struct
 * MGLCompatTexResolve in mgl_compat_perf.h).  The chain
 * texextValidateTexture + texdataBindMTLTexture + texparamSamplerForUnit +
 * texparamTextureForTex is 5 ObjC calls per texture per stage per draw, and
 * glmark2 [ideas] spends ~40% of its per-draw CPU there while the unit keeps
 * feeding the same realized texture.  Keyed on every identity the chain
 * reads; any glTexParameter / sampler-parameter change releases the
 * MGLTexParamCache (mglTexParamInvalidate) and misses.  The (view, sampler)
 * pair is replayed through the normal shadow-filtered setters, so the
 * encoder still gets a real bind whenever the resolved object changed. */
- (bool) compatTexResolveHit: (bool)frag unit: (GLuint)unit slot: (GLuint)slot tex: (Texture *)tex
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m || !m->enabled || slot >= MGLC_BIND_SLOTS || !tex || !tex->mtl_data)
        return false;
    /* pending data / level work: bindMTLTexture's chain performs the upload,
     * so a dirty texture must always take the full path (dirty_bits is also
     * the re-realization signal: a storage re-specification sets it) */
    if (tex->dirty_bits)
        return false;
    /* a vertex-stage shadow-sampling slot must run geomShadowCheckStale every
     * draw (the depth texture may be re-bound behind our back) */
    if (!frag && mgl_geom_vs_shadow_slot(STATE(program), slot))
        return false;
    void *so = (unit < TEXTURE_UNITS) ? (void *)STATE(texture_samplers[unit]) : NULL;
    struct MGLCompatTexResolve *e = &m->vtexres[frag ? 1 : 0][slot];
    /* the CURRENT generation: a freshly created encoder must not inherit
     * entries recorded for its predecessor (compatEncGen bumps lazily — the
     * shadow inside the setters would still catch it, but a miss here keeps
     * the replayed ids from being trusted at all) */
    uint64_t gen = compatEncGen(m, (__bridge void *)_currentRenderEncoder);
    if (e->gen == gen && e->tex == (void *)tex && e->tex_mtl == tex->mtl_data &&
        e->tex_params == tex->params.mtl_data && e->sampler_obj == so &&
        (!so || e->sampler_params == ((Sampler *)so)->params.mtl_data) && e->view && e->samp)
    {
        id<MTLTexture> view = (__bridge id<MTLTexture>)e->view;
        id<MTLSamplerState> samp = (__bridge id<MTLSamplerState>)e->samp;
        if (frag)
        {
            [self compatSetFragmentTexture: view atIndex: slot];
            [self compatSetFragmentSamplerState: samp atIndex: slot];
        }
        else
        {
            [self compatSetVertexTexture: view atIndex: slot];
            [self compatSetVertexSamplerState: samp atIndex: slot];
        }
        return true;
    }
    return false;
}

/* record the resolution the caller computed, for the next draw's hit check */
- (void) compatTexResolveRecord: (bool)frag unit: (GLuint)unit slot: (GLuint)slot tex: (Texture *)tex
                            view: (id<MTLTexture>)view sampler: (id<MTLSamplerState>)samp
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m || !m->enabled || slot >= MGLC_BIND_SLOTS || !tex || !view || !samp)
        return;
    void *so = (unit < TEXTURE_UNITS) ? (void *)STATE(texture_samplers[unit]) : NULL;
    struct MGLCompatTexResolve *e = &m->vtexres[frag ? 1 : 0][slot];
    e->gen = compatEncGen(m, (__bridge void *)_currentRenderEncoder);
    e->tex = (void *)tex;
    e->tex_mtl = tex->mtl_data;
    e->tex_params = tex->params.mtl_data;
    e->sampler_obj = so;
    e->sampler_params = so ? ((Sampler *)so)->params.mtl_data : NULL;
    e->view = (__bridge void *)view;
    e->samp = (__bridge void *)samp;
}

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
        [self compatCSFrameStats];   /* compat CS: per-frame stats hook */
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
        /* compat CS: this blit runs on the GL thread — the encoder worker
         * must be done before a second encoder opens on this command buffer */
        [self compatCSDrain];
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
    [self compatCSFrameStats];       /* compat CS: per-frame stats hook */
    mgl_prof_frame();
    return true;
}

@end
