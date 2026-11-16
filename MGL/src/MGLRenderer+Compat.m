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
#import "mgl_ext_buffers.h"           /* mgl_buffers_mark_gpu_use (bindmap cache, session 13) */
#import "mgl_ext_introspect.h"        /* default-uniform buffer revalidation (session 13) */
#import "mgl_ext_program.h"           /* compile_msl hook (aliased-SSBO variants, r100c) */
#import "MGLRenderer+Framebuffer.h"    /* fbApplyPipelineFormats (FBO-dirty format key) */
#import "MGLRenderer+TextureExt.h"     /* texextAdjustPipelineDescriptor (same) */
#import "mgl_ext_tess.h"              /* r152: emulation vertex-stage choice in the key */
#import "mgl_ext_geom.h"
#include "mgl_ext_fb.h"                /* MGLRenderbufferEx (attachment GL formats in the key) */
#include "mgl_compat_perf.h"
#include "mgl_ext_compat.h"       /* compat domain state + advblend (GL_KHR_blend_equation_advanced) */

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
- (void) extTexextPreDraw: (id<MTLRenderCommandEncoder>) enc;   /* MGLRenderer+TextureExt.m */
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
- (void) compatForceOpaque:(id<MTLTexture>)tex;   /* rewrite the presented surface's alpha to 1 */
@end

@interface MGLCompatSurfacePool : NSObject
@property (nonatomic, strong) CALayer *layer;            /* displays the surfaces */
@property (nonatomic, strong) NSMutableArray<id<MTLTexture>> *textures;
@property (nonatomic, strong) NSMutableArray<id> *surfaces;   /* IOSurfaceRef bridged */
@property (nonatomic, assign) NSUInteger w, h;
@property (nonatomic, assign) unsigned next;             /* round-robin index (blit path) */
@property (nonatomic, assign) unsigned write;            /* slot the default framebuffer renders into (direct path) */
@property (nonatomic, assign) BOOL rotate;               /* advance `write` at the next fbEnsureDefaultFramebuffer */
@property (nonatomic, strong) id<MTLComputePipelineState> opaquePso;  /* force-alpha-255 kernel */
@property (nonatomic, assign) BOOL opaquePsoFailed;
@property (nonatomic, strong) id<MTLRenderPipelineState> presentPso;  /* copy+force-alpha present pass */
@property (nonatomic, assign) MTLPixelFormat presentPsoFormat;        /* dst format it was built for */
@property (nonatomic, assign) BOOL presentPsoFailed;
@end

/* GL window semantics: the default framebuffer's alpha is DISCARDED at
 * present — the window is opaque no matter what blending left in dst.a
 * (glBlendFunc leaves a*a + (1-a)).  CoreAnimation composites layer.contents
 * with its alpha, so a translucent region washes out against the window's
 * white backdrop (glmark2 jellyfish: the bell lost all caustic contrast —
 * every blended pixel was ~60% alpha over white).  -compatForceOpaque:
 * rewrites the presented IOSurface's alpha to 1.0 before the commit. */
static const char *s_mgl_force_opaque_msl =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void mgl_force_opaque(texture2d<float, access::read_write> tex [[texture(0)]],\n"
    "                             uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    "    if (gid.x < tex.get_width() && gid.y < tex.get_height())\n"
    "    {\n"
    "        float4 c = tex.read(gid);\n"
    "        tex.write(float4(c.rgb, 1.0f), gid);\n"
    "    }\n"
    "}\n";

/* present copy: ONE render pass that copies the frame to the presented
 * IOSurface AND writes alpha=1, so the surface never holds a half-alpha state
 * the compositor could catch (replaces the s16d pair "force-opaque kernel on
 * the source + blit copy" — same single-write transition to CA, half the GPU
 * work, one encoder instead of two, and the source's GL alpha channel is no
 * longer clobbered, so readbacks keep GL semantics). */
static const char *s_mgl_present_copy_msl =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "vertex float4 mgl_present_vert(uint vid [[vertex_id]])\n"
    "{\n"
    "    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));\n"
    "    return float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);   /* full-screen triangle */\n"
    "}\n"
    "fragment float4 mgl_present_frag(float4 pos [[position]],\n"
    "                                 texture2d<float> src [[texture(0)]])\n"
    "{\n"
    "    uint2 gid = uint2(pos.xy);\n"
    "    if (gid.x >= src.get_width() || gid.y >= src.get_height())\n"
    "        return float4(0.0f, 0.0f, 0.0f, 1.0f);\n"
    "    return float4(src.read(gid).rgb, 1.0f);\n"
    "}\n";

@implementation MGLCompatSurfacePool
@synthesize layer, textures, surfaces, w, h, next, write, rotate, opaquePso, opaquePsoFailed;
@synthesize presentPso, presentPsoFormat, presentPsoFailed;
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

/* one cached key -> pipeline association: plain C slots (compat perf session 13).
 * The hot path used to be an NSMutableArray of CompatPipeBox objects — every
 * hit paid objectAtIndex: + property msgSends and every move-to-front paid a
 * remove/insert pair (~0.5 us per VAO-changed walk in glmark2 [ideas], where
 * the pipeline block runs for each of the 89 re-specification walks per frame).
 * The pipeline object is retained via CFBridgingRetain so no ObjC object is
 * touched on a hit. */
#define MGLC_PIPE_CACHE_SLOTS 512

@interface MGLCompatPipelineCache : NSObject
{
@public
    struct MGLCompatPipeSlot {
        uint64_t a, b;
        void    *pipe_cf;      /* retained id<MTLRenderPipelineState>, NULL until stored */
        /* the vattrib draw-plan this pipeline was built against (binding_index
         * excluded: vattribMapVertexBuffers rebuilds it every walk).  A cache
         * hit proves the (MSL, VAO-layout) matches, but the plan currently in
         * s->plan can belong to the OTHER cached key when the scene alternates
         * two layouts (glmark2 [ideas]: shadow/main, 2 components vs 3); the
         * hit path restores the slot's copy so extVattribPreDraw feeds each
         * draw its own fetch/shadow/const analysis. */
        MGLVattribDrawPlan plan;
    } slots[MGLC_PIPE_CACHE_SLOTS];
    unsigned count;
    /* compat perf (session 16): which (program content, VAO layout) the plan
     * currently live in ext_vattrib's s->plan belongs to.  The plan's fetch /
     * shadow / const analysis is a pure function of those two, and only a hit
     * for a different key or the descriptor path (vattribGenerateVertexDe-
     * scriptor) rewrites it — vattribMapVertexBuffers touches binding_index
     * only.  When the same (msl, layout) hits again the memcmp/restore is
     * redundant (glmark2 [ideas]: 89 identical-layout VAOs hit one key all
     * frame, ~500 B of compare per walk). */
    int      plan_live_slot;      /* -1 unknown */
    uint64_t plan_live_msl, plan_live_vao;
    int      pending;          /* slot index awaiting its pipeline (-1 = none) */
    /* perf: key-component attribution of misses */
    uint64_t lastKey_a, lastKey_b;
    BOOL     lastKeyValid;
    uint64_t missCount;
    /* compat perf (session 17): direct-mapped key->slot memo.  Hits no longer
     * move-to-front (that memmove shifted up to 512 slot structs — several KB
     * per first-of-group walk, ~89 per glmark2 [ideas] frame), so slot indices
     * are STABLE while no miss inserts, and the memo answers a repeated key
     * with one compare.  The whole table is invalidated by any miss (the only
     * other place slots shift is the 512-entry eviction, same block). */
#define MGLC_PIPE_MEMO 16
    struct MGLCompatPipeMemo {
        uint64_t a, b;
        int      slot;
    } memo[MGLC_PIPE_MEMO];
}
@end

@implementation MGLCompatPipelineCache
- (void) dealloc
{
    for (unsigned i = 0; i < count; i++)
        if (slots[i].pipe_cf)
            CFBridgingRelease(slots[i].pipe_cf);
}
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

/* fold the bound VAO's attribute LAYOUT into 64 bits: the render pipeline's
 * vertex descriptor is a function of the attribute formats, strides,
 * relative offsets, divisors and whether each attrib has a buffer at all —
 * NOT of which buffer objects feed them (buffer identity only reaches the
 * encoder's setVertexBuffer, never the PSO).  glmark2 [ideas] draws ~89 VAOs
 * per frame whose layouts are identical but whose VBOs differ: hashing buffer
 * identity split the PSO cache into 89 entries and turned every cache hit
 * into a move-to-front memmove of tens of KB.  A full hash collision only
 * picks a stale pipeline with an identical vertex layout. */
static uint64_t compatVAOHash(GLMContext ctx)
{
    VertexArray *vao = ctx->state.vao;
    uint64_t h = 1469598103934665603ULL;
    #define MIX(v) do { h ^= (uint64_t)(uintptr_t)(v); h *= 1099511628211ULL; } while (0)
    MIX(vao->enabled_attribs);
    for (GLuint i = 0; i < MAX_ATTRIBS && i < 32; i++)
    {
        VertexAttrib *a = &vao->attrib[i];
        MIX(a->size); MIX(a->type); MIX(a->normalized);
        MIX(a->stride); MIX(a->divisor); MIX(a->relativeoffset);
        MIX(a->buffer_bindingindex);
        MIX(a->buffer != NULL);          /* bufferless attrib -> constant stream */
    }
    return h;
}
#undef MIX

/* the hashed bytes of a VertexArray: the LAYOUT words below — the exact
 * fields vattribGenerateVertexDescriptor reads (fmt: size/type/normalized/
 * integer/is_long/relativeoffset/binding; binding: buffer-bound/offset/
 * stride/divisor).  A memcmp of them is an EXACT validator of the hash
 * inputs, so two VAOs with identical layouts validate each other's entry. */
#define MGLC_VAO_LAYOUT_WORDS  (1 + MAX_ATTRIBS * 3 + MGL_VA_MAX_BINDINGS * 3)

/* VAO layout hash for the pipeline key.  Session 19 compaction: only the
 * layout-relevant words are packed — the ENABLED attribs' formats plus the
 * bindings those attribs reference (exactly the set vattribMapVertexBuffers
 * maps, so the PSO descriptor and the shared plan's binding_index[] depend on
 * nothing outside it: a disabled attrib feeds a fixed-format const stream
 * (double value bits, current-value content is not pipeline state), and a
 * binding no enabled attrib references never reaches the encoder).  A vertex
 * buffer's base byte offset reaches the encoder per draw (setVertexBuffer, read
 * live from va->binding[]), so only its 4-byte alignment class — the
 * aligned-vs-shadow fetch decision — is hashed.  glmark2-style per-mesh offsets
 * used to fan 89 same-layout meshes out into 89 distinct PSO keys; they now
 * share one.  Word order is deterministic (attrib index order, binding index
 * order), so equal byte sequences mean equal relevant layouts. */
static uint64_t compatVAOHashCached(struct MGLExtMetal_compat *m, GLMContext ctx)
{
    VertexArray *vao = ctx->state.vao;
    MGLVertexArrayFull *va = mgl_va_full(vao);
    uint64_t w[MGLC_VAO_LAYOUT_WORDS];
    unsigned nw = 0;
    GLuint used = 0;
    w[nw++] = (uint64_t)vao->enabled_attribs;
    for (GLuint i = 0; i < MAX_ATTRIBS; i++)
    {
        if (!(vao->enabled_attribs & (1u << i)))
            continue;
        MGLVertexAttribFormat *f = &va->fmt[i];
        if (f->binding < MGL_VA_MAX_BINDINGS)
            used |= 1u << f->binding;
        uint64_t *a = &w[nw];
        a[0] = (uint64_t)f->size | ((uint64_t)f->type << 32);
        a[1] = (uint64_t)f->normalized | ((uint64_t)f->integer << 8) |
               ((uint64_t)f->is_long << 16) | ((uint64_t)f->binding << 32);
        a[2] = (uint64_t)f->relativeoffset;
        nw += 3;
    }
    for (GLuint i = 0; i < MGL_VA_MAX_BINDINGS; i++)
    {
        if (!(used & (1u << i)))
            continue;
        MGLVertexBinding *b = &va->binding[i];
        uint64_t *a = &w[nw];
        a[0] = (uint64_t)(b->buffer != NULL);         /* bufferless -> constant stream */
        a[1] = (uint64_t)(b->offset & 3);             /* alignment class: aligned-fetch decision */
        a[2] = (uint64_t)b->stride | ((uint64_t)b->divisor << 32);
        nw += 3;
    }
    const size_t bytes = nw * sizeof w[0];
    uint64_t h = 1469598103934665603ULL;
    for (unsigned i = 0; i < nw; i++)
        h = (h ^ w[i]) * 1099511628211ULL;
    if (!m)
        return h;
    if (m->vaokey.len == bytes && m->vaokey.snap &&
        memcmp(m->vaokey.snap, w, bytes) == 0)
        return m->vaokey.hash;
    /* fixed-capacity snapshot: the packed word count varies with the enabled
     * set, so a shrink followed by a grow must not outgrow the allocation */
    if (!m->vaokey.snap)
        m->vaokey.snap = (uint8_t *)malloc(MGLC_VAO_LAYOUT_WORDS * sizeof w[0]);
    if (m->vaokey.snap)
    {
        memcpy(m->vaokey.snap, w, bytes);
        m->vaokey.vao = (void *)vao;
        m->vaokey.len = bytes;
        m->vaokey.hash = h;
    }
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

/* the descriptor path records the draw-plan it built into the slot it just
 * filled (see MGLCompatPipelineCache above) */
static void compatPlanStore(GLMContext ctx, struct MGLCompatPipeSlot *slot)
{
    struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
    if (vs)
        slot->plan = vs->plan;
    else
        memset(&slot->plan, 0, sizeof(slot->plan));
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
            compatProfOn() && fprintf(stderr, "MGL PROFILE pipeMiss-decline dirty=0x%llx\n",
                    (unsigned long long)ctx->state.dirty_bits);
        mgl_prof_cnt[17]++;   /* declined: a hard bit is set */
        return false;
    }
    /* No DIRTY_FBO / "VAO or program must be dirty" decline here (784c480 added
     * one for a KHR-GL46.texture_swizzle.smoke stale-render-target bug that was
     * later fixed at its source; texture_swizzle is 505 Pass / 0 Fail without
     * it).  The key below covers the framebuffer pointer and the current
     * encoder's attachment formats + sample count, which is everything the PSO
     * depends on; re-attaching a texture ends the encoder (targetsChanged), so
     * the new attachment is bound by the next render pass, not by the PSO.
     * The decline sent every draw of FBO-switching scenes through the full
     * descriptor rebuild: glmark2 desktop blur 4.7k -> 1.4k FPS, desktop shadow
     * 4.5k -> 0.8k, shadow 12.3k -> 6.9k (~1 ms/frame of pipeline-descriptor). */
    if (!ctx->state.program)
        return false;

    struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    CompatPipelineKey key;
    /* content identity of the three raster stages: the fnid registry value of
     * each stage's compiled MTLFunction (registered in extProgramCompileMSL at
     * link time as a hash of the FULL MSL text + entry point).  Keying on the
     * registry value instead of sampling the MSL string (compatMSLKey: pointer
     * + first 192 + last 128 bytes) closes an alias hole: two programs whose
     * generated MSL differs only in the MIDDLE — e.g. the fp64-lowered
     * operators, one helper call inside a ~200 KB common prelude — plus a
     * recycled program/string address (create -> draw -> delete -> create in a
     * loop) produced identical sampled keys, and the second program drew with
     * the first program's PSO ("abs" rendered "-ua", roundeven rendered ceil:
     * tests/headless/test_fp64.c fp64_scalar_ops, 3-4 of 10 runs).
     * A stage with no compiled function yet (first draw, bindMTLProgram has
     * not run) or a function the registry does not know (registry degraded)
     * declines the fast path; the descriptor path then compiles and caches. */
    {
        Shader *ss[5];
        ss[0] = ctx->state.program->shader_slots[_VERTEX_SHADER];
        ss[1] = ctx->state.program->shader_slots[_FRAGMENT_SHADER];
        ss[2] = ctx->state.program->shader_slots[_GEOMETRY_SHADER];
        /* tess: the evaluation shader IS the pipeline's vertex function for a
         * tessellated draw (MGLRenderer.m: the kernel-typed vertex function is
         * swapped for shader_slots[_TESS_EVALUATION_SHADER]->mtl_data.function,
         * or the TES-derived plain_function for emulated domains), and the
         * control shader feeds the tess kernel — two programs that share VS+FS
         * MSL but differ in TCS/TES therefore get DIFFERENT pipelines from the
         * same source stages. KHR-GL46.tessellation_shader.single.primitive_
         * coverage draws exactly that (a quad-tess and a triangle-tess program
         * with identical VS/FS): without the tess stages in the key the second
         * program reused the first program's PSO and its coverage differed
         * from the reference draw by one pixel (a stencil-flagged hole). */
        ss[3] = ctx->state.program->shader_slots[_TESS_CONTROL_SHADER];
        ss[4] = ctx->state.program->shader_slots[_TESS_EVALUATION_SHADER];
        uint64_t id[5] = {1, 1, 1, 1, 1};
        for (int i = 0; i < 5; i++)
        {
            if (!ss[i])
                continue;                 /* stage not attached: contributes the neutral 1 */
            if (!ss[i]->mtl_data.function)
            {
                mgl_prof_cnt[17]++;       /* declined: stage not compiled yet */
                return false;
            }
            uint64_t fid = mglProgramFunctionId(ss[i]->mtl_data.function);
            if (!fid)
            {
                mgl_prof_cnt[17]++;       /* declined: no registered content identity */
                return false;
            }
            id[i] = fid;
        }
        key.a = id[0]
              ^ id[1] * 0x100000001B3ULL
              ^ id[2] * 0xC2B2AE3D27D4EB4FULL
              ^ id[3] * 0xD6E8FEB86659FD93ULL
              ^ id[4] * 0x9E3779B97F4A7C15ULL;
    }
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
    uint64_t vao_h = compatVAOHashCached(m, ctx);
    key.b = vao_h ^ ((m ? m->rstate_hash : 1) << 1) ^ ((uint64_t)(uintptr_t)ctx->state.framebuffer * 0x9E3779B97F4A7C15ULL);

    /* attachment formats + sample count of the CURRENT encoder: an attached
     * texture can be re-specified with a different format behind an unchanged
     * framebuffer pointer, and the PSO is validated against these formats.
     * Every attachment change ends the encoder (framebuffers.c targetsChanged
     * -> endRenderEncoding), so hashing once per encoder generation is exact;
     * the per-draw cost is two integer compares. */
    /* Attachment formats + sample count, hashed from the inputs the
     * descriptor path bakes (fbApplyPipelineFormats + the texext multisample
     * adjust, on a scratch descriptor).  Cached per encoder generation and
     * framebuffer pointer; recomputed on EVERY draw that carries DIRTY_FBO,
     * because then the current encoder's render pass may still describe the
     * PREVIOUS attachments — hashing _renderPassDescriptor there let a
     * re-attached texture of another format reuse the old format's PSO
     * (KHR-GL46.texture_size_promotion.functional).  A scratch descriptor is
     * ~2 orders of magnitude cheaper than the full descriptor path. */
    bool fbo_dirty = (ctx->state.dirty_bits & DIRTY_FBO) != 0;
    if (!m || fbo_dirty || m->fmt_hash_gen == 0 ||
        m->fmt_hash_gen != m->encoder_gen || m->fmt_fbo != (uint64_t)(uintptr_t)ctx->state.framebuffer)
    {
        MTLRenderPipelineDescriptor *fd = [[MTLRenderPipelineDescriptor alloc] init];
        [self fbApplyPipelineFormats: fd];
        [self texextAdjustPipelineDescriptor: fd];
        uint64_t fh = 1469598103934665603ULL;
        for (int i = 0; i < 8; i++)
            fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)fd.colorAttachments[i].pixelFormat;
        fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)fd.depthAttachmentPixelFormat;
        fh = fh * 1099511628211ULL + (uint64_t)(NSUInteger)fd.stencilAttachmentPixelFormat;
        fh = fh * 1099511628211ULL + (uint64_t)fd.rasterSampleCount;
        fh = fh * 1099511628211ULL + (uint64_t)fd.isRasterizationEnabled;
        /* the attachments' GL identity too: emulated formats share a Metal
         * pixel format with their host (GL_RGBA2/RGBA4/RGB* live in RGBA8) but
         * the rstate variant differs (mgl_fb_slot_lacks_alpha, quantization),
         * so the Metal formats alone alias them — KHR-GL46.texture_size_
         * promotion: GL_RGBA2 on a 2D_MULTISAMPLE target read alpha 0 */
        Framebuffer *kfb = ctx->state.framebuffer;
        if (kfb)
        {
            FBOAttachment *atts[MAX_COLOR_ATTACHMENTS + 2];
            int na = 0;
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) atts[na++] = &kfb->color_attachments[i];
            atts[na++] = &kfb->depth;
            atts[na++] = &kfb->stencil;
            for (int i = 0; i < na; i++)
            {
                FBOAttachment *at = atts[i];
                GLenum ifmt = 0, tgt = at->textarget;
                if (at->textarget == GL_RENDERBUFFER)
                {
                    if (at->buf.rbo) ifmt = ((MGLRenderbufferEx *)at->buf.rbo)->internalformat;
                }
                else if (at->buf.tex)
                {
                    ifmt = at->buf.tex->internalformat;
                    tgt = tgt * 31u + at->buf.tex->target;
                }
                fh = fh * 1099511628211ULL + (((uint64_t)tgt << 32) | (uint64_t)ifmt);
            }
        }
        if (m)
        {
            m->fmt_hash = fh ? fh : 1;
            /* an FBO-dirty hash is not tied to an encoder: force a re-hash at
             * the next draw that has no FBO dirt */
            m->fmt_hash_gen = fbo_dirty ? 0 : (m->encoder_gen ? m->encoder_gen : 1);
            m->fmt_fbo = (uint64_t)(uintptr_t)ctx->state.framebuffer;
        }
    }
    key.b ^= (m ? m->fmt_hash : 0) * 0xD6E8FEB86659FD93ULL;

    /* compat: a CLAMP_TO_BORDER border colour Metal cannot express is emulated by
     * a per-table fragment MSL variant baked into the pipeline — the signature
     * must move this key as well, or a border-colour change would reuse the
     * pipeline built with the previous variant */
    if (rs && rs->border_sig)
        key.b ^= rs->border_sig * 0xC2B2AE3D27D4EB4FULL;

    /* r152: the emulation domains substitute the pipeline's VERTEX STAGE for the same
     * program: the tessellation wide-line replay function (wl_render_now), the plain
     * TES render function (isolines / point_mode / xfb capture / CPU reference —
     * mgl_tess_plain_render, which also drops the patch vertex descriptor), and the
     * geometry pass-through (vertex_override).  None of that is in the stage
     * function ids above, so two passes of one program could share a slot.  Every
     * encoder break used to force the descriptor path (DIRTY_ALL carried
     * DIRTY_SHADER), which hid the gap; plain breaks now keep the fast path
     * (DIRTY_REAPPLY), so the choice is keyed explicitly
     * (TessWideLineVerticalHorizontal). */
    {
        struct MGLExtState_tess *ts = (struct MGLExtState_tess *)ctx->state.ext_tess;
        struct MGLExtState_geom *gs = (struct MGLExtState_geom *)ctx->state.ext_geom;
        uint64_t emu = 0;

        if (gs && gs->vertex_override)
            emu ^= (uint64_t)(uintptr_t)gs->vertex_override * 0x9E3779B97F4A7C15ULL;
        if (ts && ts->wl_render_now)
            emu ^= ((uint64_t)(uintptr_t)ts->wl_replay_function | 1) * 0xC2B2AE3D27D4EB4FULL;
        if (ts && [self extTessActive] && mgl_tess_plain_render(ctx))
        {
            MGLTessProgram *tp = mgl_tess_active_program(ctx);
            emu ^= ((uint64_t)(uintptr_t)(tp ? tp->plain_function : NULL) | 2) * 0xD6E8FEB86659FD93ULL;
        }
        key.b ^= emu * 0x100000001B3ULL;
    }

    MGLCompatPipelineCache *cache = objc_getAssociatedObject(self, kCompatPipeKey);
    if (!cache)
    {
        cache = [[MGLCompatPipelineCache alloc] init];
        cache->pending = -1;
        for (unsigned mi = 0; mi < MGLC_PIPE_MEMO; mi++)
            cache->memo[mi].slot = -1;
        objc_setAssociatedObject(self, kCompatPipeKey, cache, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    /* session 17: slot lookup = memo first, then a linear scan.  Hits keep
     * their slot position (no move-to-front: see the memo comment above — the
     * scan only runs for a key the memo does not know, i.e. the first use of
     * a (program content, layout) pair), so memo slot indices stay valid. */
    int hit = -1;
    {
        unsigned mi = (unsigned)((key.a ^ (key.b * 0x9E3779B97F4A7C15ULL)) & (MGLC_PIPE_MEMO - 1));
        struct MGLCompatPipeMemo *mo = &cache->memo[mi];
        if (mo->slot >= 0 && (unsigned)mo->slot < cache->count &&
            cache->slots[mo->slot].pipe_cf &&
            cache->slots[mo->slot].a == key.a && cache->slots[mo->slot].b == key.b)
            hit = mo->slot;
    }
    if (hit < 0)
    for (unsigned i = 0; i < cache->count; i++)
    {
        if (cache->slots[i].pipe_cf && cache->slots[i].a == key.a && cache->slots[i].b == key.b)
        {
            hit = (int)i;
            unsigned mi = (unsigned)((key.a ^ (key.b * 0x9E3779B97F4A7C15ULL)) & (MGLC_PIPE_MEMO - 1));
            cache->memo[mi].a = key.a;
            cache->memo[mi].b = key.b;
            cache->memo[mi].slot = hit;
            break;
        }
        /* perf instrumentation: a matching key whose slot never received a
         * pipeline (the descriptor path must have bailed before storing) */
        if (!cache->slots[i].pipe_cf && cache->slots[i].a == key.a && cache->slots[i].b == key.b && compatProfOn())
        {
            static uint64_t dead = 0;
            if (++dead % 200 == 1)
                compatProfOn() && fprintf(stderr, "MGL PROFILE pipeMiss dead-box (key match, pipe nil) entries=%u\n",
                        cache->count);
        }
    }
    if (hit >= 0)
    {
            unsigned i = (unsigned)hit;
            /* the plan in s->plan must be the one this pipeline was built
             * against: when the scene alternates two cached layouts the plan
             * holds the other draw's fetch/shadow/const analysis — restore
             * the slot's copy (keep this walk's binding_index).  Skipped when
             * the live plan is already this (program content, layout)'s. */
            if (cache->plan_live_slot != (int)i ||
                cache->plan_live_msl != key.a || cache->plan_live_vao != vao_h)
            {
                struct MGLExtState_vattrib *vs =
                    (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
                if (vs && memcmp(&vs->plan.fetch, &cache->slots[i].plan.fetch,
                                 sizeof(MGLVattribDrawPlan) - offsetof(MGLVattribDrawPlan, fetch)) != 0)
                {
                    int bi[MGL_VA_MAX_BINDINGS];
                    memcpy(bi, vs->plan.binding_index, sizeof(bi));
                    vs->plan = cache->slots[i].plan;
                    memcpy(vs->plan.binding_index, bi, sizeof(bi));
                }
                cache->plan_live_slot = (int)i;
                cache->plan_live_msl = key.a;
                cache->plan_live_vao = vao_h;
            }
            _pipelineState = (__bridge id<MTLRenderPipelineState>)cache->slots[i].pipe_cf;
            mgl_prof_cnt[16]++;
            return true;
    }
    /* miss: remember the key; the descriptor path stores the pipeline into it
     * (compatPipelineStore below, called by processGLState) */
    if (cache->count < MGLC_PIPE_CACHE_SLOTS)
        cache->count++;
    else if (cache->slots[MGLC_PIPE_CACHE_SLOTS - 1].pipe_cf)
        CFBridgingRelease(cache->slots[MGLC_PIPE_CACHE_SLOTS - 1].pipe_cf);
    memmove(&cache->slots[1], &cache->slots[0],
            (cache->count - 1) * sizeof(cache->slots[0]));
    cache->slots[0].a = key.a;
    cache->slots[0].b = key.b;
    cache->slots[0].pipe_cf = NULL;
    cache->pending = 0;
    cache->plan_live_slot = -1;   /* slots shifted; the descriptor path rebuilds the plan */
    /* session 17: every slot index moved down by one — the memo table must be
     * re-learned from the scans (also covers the 512-entry eviction) */
    for (unsigned mi = 0; mi < MGLC_PIPE_MEMO; mi++)
        cache->memo[mi].slot = -1;

    /* perf instrumentation: WHICH key component moved (a = MSL content ids,
     * b = VAO layout / pipeline-baked rstate / FBO identity) */
    if (compatProfOn())
    {
        if (!cache->lastKeyValid)
        {
            cache->lastKey_a = key.a;
            cache->lastKey_b = key.b;
            cache->lastKeyValid = YES;
            cache->missCount = 0;
        }
        else if (cache->lastKey_a != key.a || cache->lastKey_b != key.b)
        {
            cache->missCount++;
            if (cache->missCount % 200 == 1)
                compatProfOn() && fprintf(stderr, "MGL PROFILE pipeMiss %s (a %llx->%llx b %llx->%llx; vao=%llx gen=%u fbo=%p)\n",
                        cache->lastKey_a != key.a ? "msl" : "vao/rstate/fbo",
                        (unsigned long long)cache->lastKey_a, (unsigned long long)key.a,
                        (unsigned long long)cache->lastKey_b, (unsigned long long)key.b,
                        (unsigned long long)compatVAOHash(ctx),
                        rs ? rs->generation_pipeline : 0,
                        (void *)ctx->state.framebuffer);
            cache->lastKey_a = key.a;
            cache->lastKey_b = key.b;
        }
    }
    return false;
}

/* called after the descriptor path produced a pipeline for the pending key */
- (void) compatPipelineStore
{
    MGLCompatPipelineCache *cache = objc_getAssociatedObject(self, kCompatPipeKey);
    if (cache && cache->pending >= 0 && cache->pending < (int)cache->count)
    {
        if (cache->slots[cache->pending].pipe_cf)
            CFBridgingRelease(cache->slots[cache->pending].pipe_cf);
        cache->slots[cache->pending].pipe_cf = (void *)CFBridgingRetain(_pipelineState);
        compatPlanStore(ctx, &cache->slots[cache->pending]);
        cache->plan_live_slot = -1;   /* descriptor path rewrote the plan: let the next hit re-arm */
        cache->pending = -1;
    }
    else if (compatProfOn())
    {
        /* descriptor path finished but no pending key: the pipeline was built
         * without a fast-path miss (or the miss's slot was already consumed) */
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
uint64_t mgl_compat_fast_hit, mgl_compat_fast_miss, mgl_compat_fast_rebind;
/* r154: partial-walk verdict counts (MGL_PROFILE drawfast-partial line) */
uint64_t mgl_compat_fast_tex, mgl_compat_fast_bufs, mgl_compat_fast_rs, mgl_compat_fast_moved;
uint64_t mgl_compat_fast_texunits;   /* r158: texture partial walks that re-bound only the moved units */

/* ============ per-walk buffer-map PREFIX cache (compat perf session 13) ====
 * mapBuffersToMTL (MGLRenderer.m) re-enumerates the program's SPIR-V buffer
 * resources for BOTH stage lists on every dirty walk — four resource types
 * through ObjC getProgramBindingCount/getProgramBinding dispatches — yet the
 * UBO / SSBO / atomic-counter / default-uniform PREFIX only depends on the
 * program (identity + link, both covered by DIRTY_PROGRAM) and the
 * glBindBufferBase/Range targets (buffer_base).  The VAO re-specification
 * that triggers most walks (glmark2 [ideas]: 82 of 89) cannot change it.
 * compatBindmapFast serves the prefix from an exact cache and re-runs only
 * vattribMapVertexBuffers (the attribute suffix); mapBuffersToMTL falls back
 * to the full enumeration and calls compatBindmapStore to re-arm it. */
struct MGLCompatBindmapCache {
    int        valid;
    const void *prog;               /* ctx->state.program at build time */
    int        vstage;              /* vertex stage the prefix was built for */
    /* input shadows: the four buffer_base targets the enumeration reads */
    BufferBase ub, uc, ssbo, atom;
    GLuint     ubo_high_gen;        /* r159: uniform-buffer points >= MAX_BINDABLE_BUFFERS */
    void      *nullbuf;             /* [nullShaderBuffer] seen at build (NULL if unused) */
    /* outputs: the prefix entries of the two stage lists */
    BufferMap  v[MAX_MAPPED_BUFFERS];
    GLuint     vn;
    BufferMap  f[MAX_MAPPED_BUFFERS];
    GLuint     fn;
    uint8_t    v_gpu_write[MAX_MAPPED_BUFFERS];  /* SSBO / ACB: per-walk serial stamp */
    uint8_t    f_gpu_write[MAX_MAPPED_BUFFERS];
    /* default-uniform-buffer entries: identity revalidated through the
     * introspection domain (its dub pointer can change on relink, which also
     * arms DIRTY_PROGRAM, but be exact anyway) */
    struct { int stage, index; void *buf; } dub[2 * 8];
    int        ndub;
    uint64_t   hits, misses;
    int        why;                /* last validation failure (profiling) */
};

static struct MGLCompatBindmapCache *compatBindmapState(GLMContext ctx)
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m)
        return NULL;
    if (!m->bindmap_cache)
        m->bindmap_cache = calloc(1, sizeof(struct MGLCompatBindmapCache));
    return (struct MGLCompatBindmapCache *)m->bindmap_cache;
}

/* exact validation of every input the prefix depends on */
- (bool) compatBindmapValid: (struct MGLCompatBindmapCache *) c vstage: (int) vstage
{
    if (!c->valid || c->prog != (const void *)ctx->state.program || c->vstage != vstage)
    {
        if (compatProfOn()) c->why = 1;
        return false;
    }
    /* a relink / program switch since the last build: rebuild.  glLinkProgram,
     * glUseProgram (and the pipeline-scope composite switch) all arm this bit,
     * and processGLState consumes it AFTER mapBuffersToMTL. */
    if (ctx->state.dirty_bits & DIRTY_PROGRAM)
    {
        if (compatProfOn()) c->why = 2;
        return false;
    }
    if (memcmp(&ctx->state.buffer_base[_UNIFORM_BUFFER], &c->ub, sizeof(BufferBase)) != 0 ||
        memcmp(&ctx->state.buffer_base[_UNIFORM_CONSTANT], &c->uc, sizeof(BufferBase)) != 0 ||
        memcmp(&ctx->state.buffer_base[_SHADER_STORAGE_BUFFER], &c->ssbo, sizeof(BufferBase)) != 0 ||
        memcmp(&ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER], &c->atom, sizeof(BufferBase)) != 0 ||
        ctx->state.ubo_high_gen != c->ubo_high_gen)
    {
        if (compatProfOn()) c->why = 3;
        return false;
    }
    if (c->nullbuf && c->nullbuf != (void *)[self nullShaderBuffer])
    {
        if (compatProfOn()) c->why = 4;
        return false;
    }
    for (int i = 0; i < c->ndub; i++)
        if ((void *)mgl_introspect_default_uniform_buffer(ctx, ctx->state.program,
                                                           c->dub[i].stage, c->dub[i].index) != c->dub[i].buf)
        {
            if (compatProfOn()) c->why = 5;
            return false;
        }
    return true;
}

- (bool) compatBindmapFast: (int) vertex_stage
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatBindmapCache *c = m ? (struct MGLCompatBindmapCache *)m->bindmap_cache : NULL;

    if (compatProfOn())
        s_compat_bindmap_dbg_m = m;

    /* kill switch (perf experiments): the prefix cache can be disabled without
     * touching the store path */
    {
        static int off = -1;
        if (off < 0) off = getenv("MGL_BINDMAP") != NULL && getenv("MGL_BINDMAP")[0] == '0';
        if (off)
        {
            if (c) c->misses++;
            return false;
        }
    }

    if (!m || !m->enabled || !c || ![self compatBindmapValid:c vstage:vertex_stage])
    {
        if (c) c->misses++;
        return false;   /* caller runs the full enumeration */
    }

    BufferMapList *vl = &ctx->state.vertex_buffer_map_list;
    BufferMapList *fl = &ctx->state.fragment_buffer_map_list;

    memcpy(vl->buffers, c->v, c->vn * sizeof(BufferMap));
    vl->count = c->vn;
    memcpy(fl->buffers, c->f, c->fn * sizeof(BufferMap));
    fl->count = c->fn;

    /* the enumeration stamps SSBO / ACB reads + writes every walk (CPU-side
     * read sync): keep that contract for the cached entries */
    for (GLuint i = 0; i < c->vn; i++)
        if (c->v_gpu_write[i] && c->v[i].buf)
            mgl_buffers_mark_gpu_use(ctx, c->v[i].buf, true);
    for (GLuint i = 0; i < c->fn; i++)
        if (c->f_gpu_write[i] && c->f[i].buf)
            mgl_buffers_mark_gpu_use(ctx, c->f[i].buf, true);

    /* vertex stage: the attribute suffix still follows the live VAO */
    if (vertex_stage == _VERTEX_SHADER &&
        ![self vattribMapVertexBuffers: vl])
        return false;

    c->hits++;
    return true;
}

/* mapBuffersToMTL calls this after the full enumeration rebuilt both lists */
- (void) compatBindmapStore: (int) vertex_stage
{
    struct MGLCompatBindmapCache *c = compatBindmapState(ctx);
    if (!c)
        return;

    /* re-arm (keep the profiling counters across the reset) */
    c->misses++;
    c->valid = 1;
    c->prog = (const void *)ctx->state.program;
    c->vstage = vertex_stage;
    memcpy(&c->ub, &ctx->state.buffer_base[_UNIFORM_BUFFER], sizeof(BufferBase));
    memcpy(&c->uc, &ctx->state.buffer_base[_UNIFORM_CONSTANT], sizeof(BufferBase));
    memcpy(&c->ssbo, &ctx->state.buffer_base[_SHADER_STORAGE_BUFFER], sizeof(BufferBase));
    memcpy(&c->atom, &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER], sizeof(BufferBase));
    c->ubo_high_gen = ctx->state.ubo_high_gen;

    const Buffer *nullbuf = [self nullShaderBuffer];
    const BufferMapList *vl = &ctx->state.vertex_buffer_map_list;
    const BufferMapList *fl = &ctx->state.fragment_buffer_map_list;
    c->ndub = 0;
    c->nullbuf = NULL;

    /* default-uniform buffers: at most ONE per stage (introspection's
     * dub_res_index), recovered exactly by asking the introspection domain
     * for the stage's UC indices — done once per rebuild, not per walk */
    struct { int index; void *buf; } dubv = { -1, NULL }, dubf = { -1, NULL };
    if (ctx->state.program)
    {
        const int stages[2] = { vertex_stage, _FRAGMENT_SHADER };
        for (int s = 0; s < 2; s++)
        {
            GLuint n = 0;
            if (stages[s] >= 0 && stages[s] < _MAX_SHADER_TYPES)
                n = ctx->state.program->spirv_resources_list[stages[s]][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT].count;
            for (GLuint i = 0; i < n; i++)
            {
                Buffer *d = mgl_introspect_default_uniform_buffer(ctx, ctx->state.program, stages[s], (int)i);
                if (d)
                {
                    if (s == 0) { dubv.index = (int)i; dubv.buf = d; }
                    else        { dubf.index = (int)i; dubf.buf = d; }
                    break;
                }
            }
        }
    }
    if (dubv.index >= 0) { c->dub[c->ndub].stage = vertex_stage; c->dub[c->ndub].index = dubv.index; c->dub[c->ndub].buf = dubv.buf; c->ndub++; }
    if (dubf.index >= 0) { c->dub[c->ndub].stage = _FRAGMENT_SHADER; c->dub[c->ndub].index = dubf.index; c->dub[c->ndub].buf = dubf.buf; c->ndub++; }

    if (compatProfOn())
    {
        static uint64_t sn = 0;
        if (++sn % 2000 == 1)
            fprintf(stderr, "MGL PROFILE bindmap store: vn=%u fn=%u ndub=%d listv=%u listf=%u nullbuf=%d\n",
                    c->vn, c->fn, c->ndub, vl->count, fl->count, c->nullbuf ? 1 : 0);
    }

    c->vn = 0;
    for (GLuint i = 0; i < vl->count && i < MAX_MAPPED_BUFFERS; i++)
    {
        const BufferMap *e = &vl->buffers[i];
        if (e->attribute_mask != 0 || e->buf == NULL)
            break;          /* attribute entry or the SSBO-size pad: suffix */
        if (e->buf == nullbuf) { c->nullbuf = (void *)nullbuf; }
        c->v[c->vn] = *e;
        /* SSBO / ACB entry: its Buffer lives in the corresponding buffer_base
         * slot (buffer_base_index is the array element the enumeration read) */
        GLuint bbi = e->buffer_base_index < MAX_BINDABLE_BUFFERS ? e->buffer_base_index : 0;
        c->v_gpu_write[c->vn] =
            ((e->buf == ctx->state.buffer_base[_SHADER_STORAGE_BUFFER].buffers[bbi].buf ||
              e->buf == ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[bbi].buf) &&
             e->buf != nullbuf) ? 1 : 0;
        c->vn++;
    }
    c->fn = 0;
    for (GLuint i = 0; i < fl->count && i < MAX_MAPPED_BUFFERS; i++)
    {
        const BufferMap *e = &fl->buffers[i];
        if (e->attribute_mask != 0 || e->buf == NULL)
            break;
        if (e->buf == nullbuf) { c->nullbuf = (void *)nullbuf; }
        c->f[c->fn] = *e;
        GLuint bbi = e->buffer_base_index < MAX_BINDABLE_BUFFERS ? e->buffer_base_index : 0;
        c->f_gpu_write[c->fn] =
            ((e->buf == ctx->state.buffer_base[_SHADER_STORAGE_BUFFER].buffers[bbi].buf ||
              e->buf == ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[bbi].buf) &&
             e->buf != nullbuf) ? 1 : 0;
        c->fn++;
    }
}

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
    /* compat perf (session 17): verdicts that cannot change while the identity
     * checks above hold — recorded at latch time so a clean draw pays a load
     * instead of the call: tess routing needs a tess program record for THIS
     * program (records appear/disappear at link, which arms DIRTY_PROGRAM and
     * fails the dirty-mask check first) */
    bool        tess_active;
    ImageUnit   image_units[TEXTURE_UNITS];     /* glBindImageTexture group */

    /* content snapshots */
    unsigned    tex_mask[4];            /* ctx->state.active_texture_mask */
    BufferMapList vlist, flist;         /* vertex / fragment buffer maps (identity) */
    struct MGLCompatBufId vb[MAX_MAPPED_BUFFERS], fb[MAX_MAPPED_BUFFERS];
    MGLVattribDrawPlan plan;            /* vattrib draw plan */
    uint64_t    current_serial;         /* vs->current_serial at latch (gate 14) */
    MGLVertexAttribFormat fmt[MAX_ATTRIBS];
    MGLVertexBinding binding[MGL_VA_MAX_BINDINGS];
    struct MGLCompatBufId vabuf[MGL_VA_MAX_BINDINGS];
    MGLCurrentAttrib current[MAX_ATTRIBS];      /* generic vertex attribs (const streams) */
    const void *tex[TEXTURE_UNITS], *samp[TEXTURE_UNITS];
    void *tex_mtl[TEXTURE_UNITS], *tex_par[TEXTURE_UNITS], *samp_par[TEXTURE_UNITS];
    uint32_t tex_pser[TEXTURE_UNITS], samp_pser[TEXTURE_UNITS];   /* r154: TextureParameter.mtl_serial */
    GLuint      sampler_map_gen;        /* r158: ctx->state.sampler_map_gen at latch */
    GLuint      ubo_high_gen;           /* r159: ctx->state.ubo_high_gen at latch */
    /* indexed binding tables (gate 26): DIRTY_BUFFER_BASE_STATE stands between
     * draws that re-point an indexed binding (glBindBufferBase/Range on UBO /
     * SSBO / ACB), and the skip path never re-runs mapGLBuffersToMTLBufferMap,
     * so the map lists — and with them the encoder's buffer bindings — would
     * keep the buffer object from the latched draw (KHR-GL46.shader_storage_
     * buffer_object.advanced-switchBuffers: 4 draws of one program through 4
     * SSBOs all read the first buffer).  The gate compares the tables' content
     * and only takes the full walk when it actually moved. */
    BufferBase   bb_ub, bb_uc, bb_ssbo, bb_atom;
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
    uint64_t    misswhy[32];
    uint64_t    listwhy[2][6];   /* r154: vlist / flist miss sub-reasons (compatListStable why) */
    /* r154: does the bound program's pipeline depend on texture state the
     * texture-only partial walk does not re-evaluate (vertex-stage shadow /
     * wrap function constants, geometry / tessellation stages)?  Cached per
     * (program, vertex MSL) — the MSL pointer moves on every relink. */
    const void *tdep_prog, *tdep_msl;
    bool        tdep;
    /* r158: the gate's texture verdict for compatDrawFastTextures — the units
     * whose binding moved, or tex_full when the whole bind walk must run */
    unsigned    tex_units[4];
    bool        tex_full;
    /* r158: does the program bind separate images / samplers? (per program) */
    const void *sep_prog;
    bool        sep;
    uint64_t    dirtymask_seen[8];   /* MGL_DRAWFAST_DEBUG: distinct dirty_bits values */
    uint64_t    dirtymask_val[8];

    /* session 17: PreDraw hook count memo (see compatDrawGate check 2) */
    uint64_t    hook_gen;
    unsigned    hook_count;

    /* session 17: current_serial whose content the gate verified unchanged
     * (see compatDrawGate check 14; s->current_serial is the latch's copy) */
    uint64_t    gate_current_serial;
};

static struct MGLCompatDrawFast *compatDrawFastState(GLMContext ctx)
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m)
        return NULL;
    if (!m->draw_fast)
    {
        struct MGLCompatDrawFast *df = calloc(1, sizeof(struct MGLCompatDrawFast));
        df->hook_gen = ~(uint64_t)0;   /* force the hook count on the first gate call */
        m->draw_fast = df;
    }
    return (struct MGLCompatDrawFast *)m->draw_fast;
}

/* MGL_DRAWFAST_DEBUG: most recent fast-path state (single-renderer processes) */
static struct MGLCompatDrawFast *s_compat_df_debug;      /* last gate's state (MGL_PROFILE dump) */
static struct MGLExtMetal_compat *s_compat_bindmap_dbg_m; /* last bindmap cache owner (same dump) */
static const char *const s_compat_df_why[] = {
    "no-snap", "dirty-bits", "hook-count", "enc/cb/pipe-id", "cb-committed",
    "prog/vao/fbo-id", "rstate-gen", "vlist", "vlist-dirty", "flist",
    "flist-dirty", "no-vattrib", "plan", "vao-fmt/binding", "current-attribs",
    "(15)", "va-mapped", "va-dirty", "va-mtl-id", "tex-mask",
    "tex/samp-id", "tex-dirty", "tex-mtl/params", "samp-params", "tex-image-units",
    "tess-route", "bufbase", "tex-vsdep", "rebind+tex", "sampler-map"
};

void mgl_compat_drawfast_debug(void)
{
    struct MGLCompatDrawFast *df = s_compat_df_debug;
    if (df)
    {
    fprintf(stderr, "MGL PROFILE drawfast-miss:");
    for (unsigned i = 0; i < sizeof(df->misswhy) / sizeof(df->misswhy[0]); i++)
        if (df->misswhy[i])
            fprintf(stderr, " %s=%llu",
                    i < sizeof(s_compat_df_why) / sizeof(s_compat_df_why[0]) ? s_compat_df_why[i] : "?",
                    (unsigned long long)df->misswhy[i]);
    fprintf(stderr, "\n");
    /* r154: list-miss sub-reasons (count / map entries / mapped / dirty / storage)
     * and the partial verdicts taken instead of a full walk */
    if (df->listwhy[0][0] + df->listwhy[0][1] + df->listwhy[0][2] + df->listwhy[0][4] + df->listwhy[0][5] +
        df->listwhy[1][0] + df->listwhy[1][1] + df->listwhy[1][2] + df->listwhy[1][4] + df->listwhy[1][5])
        fprintf(stderr, "MGL PROFILE drawfast-listwhy: vlist count=%llu map=%llu mapped=%llu storage=%llu size=%llu"
                        " | flist count=%llu map=%llu mapped=%llu storage=%llu size=%llu\n",
                (unsigned long long)df->listwhy[0][0], (unsigned long long)df->listwhy[0][1],
                (unsigned long long)df->listwhy[0][2], (unsigned long long)df->listwhy[0][4],
                (unsigned long long)df->listwhy[0][5],
                (unsigned long long)df->listwhy[1][0], (unsigned long long)df->listwhy[1][1],
                (unsigned long long)df->listwhy[1][2], (unsigned long long)df->listwhy[1][4],
                (unsigned long long)df->listwhy[1][5]);
    if (mgl_compat_fast_tex + mgl_compat_fast_bufs + mgl_compat_fast_rs + mgl_compat_fast_rebind)
        fprintf(stderr, "MGL PROFILE drawfast-partial/frame: textures=%.1f (moved units only %.1f) buffers=%.1f (storage-moved %.1f) rstate=%.1f rebind=%.1f\n",
                (double)mgl_compat_fast_tex / 120.0, (double)mgl_compat_fast_texunits / 120.0,
                (double)mgl_compat_fast_bufs / 120.0,
                (double)mgl_compat_fast_moved / 120.0, (double)mgl_compat_fast_rs / 120.0,
                (double)mgl_compat_fast_rebind / 120.0);
    mgl_compat_fast_tex = mgl_compat_fast_bufs = mgl_compat_fast_rs = mgl_compat_fast_moved = 0;
    mgl_compat_fast_texunits = 0;
    mgl_compat_fast_rebind = 0;
    memset(df->listwhy, 0, sizeof(df->listwhy));
    for (unsigned i = 0; i < 8; i++)
        if (df->dirtymask_seen[i])
            fprintf(stderr, "MGL PROFILE drawfast-dirty: 0x%llx x%llu\n",
                    (unsigned long long)df->dirtymask_val[i], (unsigned long long)df->dirtymask_seen[i]);
    memset(df->misswhy, 0, sizeof(df->misswhy));
    memset(df->dirtymask_seen, 0, sizeof(df->dirtymask_seen));
    }
    /* compat perf session 13: buffer-map prefix cache coverage */
    {
        struct MGLExtMetal_compat *mm = s_compat_bindmap_dbg_m;
        struct MGLCompatBindmapCache *cc =
            mm ? (struct MGLCompatBindmapCache *)mm->bindmap_cache : NULL;
        if (cc && (cc->hits || cc->misses))
        {
            fprintf(stderr, "MGL PROFILE bindmap: hit=%.1f miss=%.1f (%.0f%% hit) why=%d\n",
                    (double)cc->hits / 120.0, (double)cc->misses / 120.0,
                    100.0 * (double)cc->hits / (double)(cc->hits + cc->misses), cc->why);
            cc->hits = cc->misses = 0;
        }
    }
}

/* per-listed-buffer stability: not mapped, no pending content change, same
 * Metal storage.  Content changes (glBufferSubData / glUniform into the
 * default uniform block) set data.dirty_bits and force the full walk, which
 * re-streams them through the argument ring. */
static bool compatListStable(const BufferMapList *live, const BufferMapList *snap,
                             const struct MGLCompatBufId *snapbuf, int *why, bool allow_dirty,
                             bool allow_moved, bool *moved)
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
        {
            /* r154: the same buffer object on new Metal storage of the same size
             * (r153 pool rename after a glBufferSubData / orphaning glBufferData
             * while the old storage was in flight).  The map entry, its offset and
             * the SSBO size constant are unchanged; extBufBindList reads
             * data.mtl_data afresh, so the buffers-only partial walk re-binds it.
             * By-value (ring) <-> MTLBuffer transitions still take the full walk. */
            /* r157: also MTLBuffer <-> none (glBufferData re-specification
             * released the storage: updateDirtyBaseBufferList allocates again
             * for >= 4 KB, smaller ones stream through the ring), for entries
             * that are not vertex attributes — an attribute's fetch mode
             * follows whether its buffer has Metal storage */
            if (allow_moved && allow_dirty && b->size == snapbuf[i].size &&
                ((b->data.mtl_data && snapbuf[i].mtl) || live->buffers[i].attribute_mask == 0))
            {
                *moved = true;
                continue;
            }
            *why = b->size != snapbuf[i].size ? 5 : 4;   /* 5: size change */
            return false;
        }
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
    /* compat perf (session 17): compare-first — when the list already matches
     * the snapshot (count, map entries and backing-buffer identities), the
     * copies are skipped.  The snapshot is only ever compared back, so equal
     * content needs no re-copy; on a program switch only the entries that
     * actually moved are written. */
    if (snap->count == live->count &&
        (live->count == 0 ||
         memcmp(snap->buffers, live->buffers, live->count * sizeof(BufferMap)) == 0))
    {
        bool same = true;
        for (GLuint i = 0; i < live->count && same; i++)
        {
            const Buffer *b = live->buffers[i].buf;
            same = snapbuf[i].buf == b &&
                   snapbuf[i].mtl == (b ? b->data.mtl_data : NULL) &&
                   snapbuf[i].size == (b ? b->size : 0);
        }
        if (same)
            return;
    }

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

/* session 19: does a failed fmt/binding content check differ from the snapshot
 * ONLY by which buffer object each binding owns (and its base offset)?  Such a
 * respec (glmark2 Mesh::draw: every mesh re-specifies identical layouts against
 * its own VBO) needs no pipeline work: the PSO key (compacted: enabled formats
 * + referenced bindings + alignment class), the plan, the map enumeration and
 * every other gate group are unchanged — only the encoder's vertex-buffer
 * bindings and the map-suffix entries move. */
static bool compatRebindClassify(GLMContext ctx, MGLVertexArrayFull *va, struct MGLCompatDrawSnap *s)
{
    const struct MGLExtState_vattrib *vs =
        (const struct MGLExtState_vattrib *)ctx->state.ext_vattrib;

    /* shadow (converted) streams are re-encoded from the bound buffer's bytes
     * by the vattrib PreDraw hook on every walk: a respec that moves the
     * buffer needs that rebuild, so only direct/const fetches may rebind */
    if (vs)
        for (int a = 0; a < MAX_ATTRIBS; a++)
            if (vs->plan.fetch[a] == MGL_VA_FETCH_SHADOW)
                return false;

    for (GLuint a = 0; a < MAX_ATTRIBS; a++)
    {
        const MGLVertexAttribFormat *f = &va->fmt[a], *g = &s->fmt[a];
        if (f->binding != g->binding || f->size != g->size || f->type != g->type ||
            f->normalized != g->normalized || f->integer != g->integer || f->is_long != g->is_long ||
            f->relativeoffset != g->relativeoffset || f->pointer_stride != g->pointer_stride)
            return false;
    }
    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
    {
        const MGLVertexBinding *B = &va->binding[b], *G = &s->binding[b];
        if ((B->buffer != NULL) != (G->buffer != NULL))
            return false;                   /* bufferless <-> buffered: fetch mode change */
        if (B->stride != G->stride || B->divisor != G->divisor)
            return false;                   /* layout word change: new PSO key */
        if ((B->offset & 3) != (G->offset & 3))
            return false;                   /* alignment class change: new PSO key */
        if (B->buffer && B->buffer != G->buffer && !B->buffer->data.mtl_data)
            return false;                   /* ring-backed stream: the full walk re-streams */
    }
    return true;
}

/* r154: true when the texture-only partial walk must NOT be used for this program:
 * its pipeline (not just its encoder bindings) can depend on texture state —
 * the vertex stage's shadow-compare / wrap emulation bakes the bound textures'
 * compare function and address modes into function constants
 * (geomShadowFCValues), and geometry / tessellation routing binds textures for
 * their own stages.  Program pipelines (ctx->state.program NULL) are declined
 * too: the texture bind walk resolves stages through MGL_PIPELINE_PROGRAM_SCOPE
 * only inside processGLState. */
static bool compatTexPipeDependent(GLMContext ctx, struct MGLCompatDrawFast *df)
{
    Program *p = ctx->state.program;
    if (!p)
        return true;
    if (p->shader_slots[_GEOMETRY_SHADER] || p->shader_slots[_TESS_CONTROL_SHADER] ||
        p->shader_slots[_TESS_EVALUATION_SHADER])
        return true;
    if (ctx->state.ext_geom && ((struct MGLExtState_geom *)ctx->state.ext_geom)->vertex_override)
        return true;
    const char *msl = p->spirv[_VERTEX_SHADER].msl_str;
    if (df->tdep_prog != (const void *)p || df->tdep_msl != (const void *)msl)
    {
        df->tdep_prog = p;
        df->tdep_msl = msl;
        df->tdep = p->spirv[_VERTEX_SHADER].shadow_slots != 0 ||
                   (msl && (strstr(msl, "mgl_shadow_op_") || strstr(msl, "mgl_wrap_op_")));
    }
    return df->tdep;
}

/* r158 kill switch: MGL_DRAWFAST_UNITS=0 makes the texture partial walk run the
 * whole texture bind walk again (r154 behaviour) */
static bool compatUnitsOn(void)
{
    static int on = -1;
    if (on < 0)
    {
        const char *e = getenv("MGL_DRAWFAST_UNITS");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on;
}

/* r154 kill switch: MGL_DRAWFAST_R154=0 restores the r153 gate verdicts */
static bool compatR154On(void)
{
    static int on = -1;
    if (on < 0)
    {
        const char *e = getenv("MGL_DRAWFAST_R154");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on;
}

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

    bool rebind = false;                    /* session 19: buffer-identity-only respec */
    bool r154 = compatR154On();
    bool moved = false;                     /* r154: a listed buffer moved to new storage */
    bool tex_moved = false;                 /* r154: a sampled texture binding moved */
    if (!df || !(s = &df->snap)->valid || !m->enabled)
        MGL_DF_MISS(0);

    /* only dirty bits whose processing is proven redundant by the content
     * gates below may stand (they are cleared on a SKIP / BUFFERS verdict) */
    /* r154: texture-binding dirt is no longer an automatic miss — the texture
     * group below either proves it redundant or routes the draw to the
     * texture-only partial walk */
    if (ctx->state.dirty_bits & ~(MGL_DF_REDUNDANT |
                                  (r154 ? (DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_SAMPLER) : 0)))
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
    /* gate 26: an indexed respec is NOT redundant unless the tables' content is
     * unchanged — a re-pointed UBO/SSBO/ACB binding must reach a fresh
     * mapBuffersToMTL enumeration, not the latched map lists (see the snap
     * struct's bb_* comment).  Same-buffer rebinds compare equal and skip on. */
    if (ctx->state.dirty_bits & DIRTY_BUFFER_BASE_STATE)
    {
        if (memcmp(&ctx->state.buffer_base[_UNIFORM_BUFFER], &s->bb_ub, sizeof(BufferBase)) != 0 ||
            memcmp(&ctx->state.buffer_base[_UNIFORM_CONSTANT], &s->bb_uc, sizeof(BufferBase)) != 0 ||
            memcmp(&ctx->state.buffer_base[_SHADER_STORAGE_BUFFER], &s->bb_ssbo, sizeof(BufferBase)) != 0 ||
            memcmp(&ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER], &s->bb_atom, sizeof(BufferBase)) != 0 ||
            ctx->state.ubo_high_gen != s->ubo_high_gen)    /* r159: points >= MAX_BINDABLE_BUFFERS */
            MGL_DF_MISS(26);
    }
    /* ...and only for the three PreDraw hooks this gate reasons about */
    /* session 17: the hook table is filled during context setup; re-count at
     * most once per encoder generation (a strcmp scan of the cache list,
     * paid on the first draw of every encoder instead of every draw) */
    if (df->hook_gen != m->encoder_gen)
    {
        df->hook_gen = m->encoder_gen;
        df->hook_count = mglExtHookCacheCount(self, "PreDraw:");
    }
    if (df->hook_count != 3)
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
        df->tex_units[0] = df->tex_units[1] = df->tex_units[2] = df->tex_units[3] = 0;
        df->tex_full = false;
        /* r158: a sampler / image uniform re-pointed a slot at another unit: the
         * per-unit checks below cannot see that, the whole bind walk must run */
        if (ctx->state.sampler_map_gen != s->sampler_map_gen)
        {
            if (!r154)
                MGL_DF_MISS(29);
            tex_moved = true;
            df->tex_full = true;
            tex_clean = false;
        }
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
                    /* r154: a moved binding / sampler / parameter set is re-bound by
                     * the texture-only partial walk (compatDrawFastTextures) as long
                     * as the new texture is realized, clean and plain (no texext
                     * buffer / view / emulation record) */
                    bool unit_moved = false;
                    if ((const void *)tex != s->tex[u] || (const void *)sam != s->samp[u])
                    {
                        if (!r154)
                            MGL_DF_MISS(20);
                        unit_moved = true;
                    }
                    if (tex)
                    {
                        if (tex->dirty_bits)
                            MGL_DF_MISS(21);
                        if (mgl_texext_info(ctx, tex))
                            MGL_DF_MISS(22);
                        if (tex->mtl_data != s->tex_mtl[u] || tex->params.mtl_data != s->tex_par[u] ||
                            tex->params.mtl_serial != s->tex_pser[u])
                        {
                            if (!r154)
                                MGL_DF_MISS(22);
                            unit_moved = true;
                        }
                        /* an unrealized texture with levels would be created by the
                         * bind walk: leave that to the full walk */
                        if (unit_moved && !tex->mtl_data && tex->num_levels)
                            MGL_DF_MISS(22);
                    }
                    if (sam && (sam->params.mtl_data != s->samp_par[u] ||
                                sam->params.mtl_serial != s->samp_pser[u]))
                    {
                        if (!r154)
                            MGL_DF_MISS(23);
                        unit_moved = true;
                    }
                    tex_moved |= unit_moved;
                    if (unit_moved)
                        df->tex_units[w] |= 1u << (u - w * 32);
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
    /* r154: DIRTY_TEX / DIRTY_SAMPLER (content-side texture changes that keep
     * the bound identities) also take the texture rebind — only the
     * unit-binding bit is provable redundant by identity alone */
    bool tex_partial = r154 && (tex_moved || (ctx->state.dirty_bits & (DIRTY_TEX | DIRTY_SAMPLER)));
    if (tex_partial && compatTexPipeDependent(ctx, df))
        MGL_DF_MISS(27);
    /* r158: texture-side dirt with no unit seen moving (a content-side change
     * the identities do not show) keeps the full bind walk */
    if (tex_partial && !(df->tex_units[0] | df->tex_units[1] | df->tex_units[2] | df->tex_units[3]))
        df->tex_full = true;

    /* buffer-map groups: identity of the maps and of every mapped buffer */
    { int wv = 0, wf = 0;
      bool vok = compatListStable(&ctx->state.vertex_buffer_map_list, &s->vlist, s->vb, &wv,
                                  allowDirtyListBuffers, r154, &moved);
      bool fok = vok && compatListStable(&ctx->state.fragment_buffer_map_list, &s->flist, s->fb, &wf,
                                         allowDirtyListBuffers, r154, &moved);
      if (!vok) { df->listwhy[0][wv % 6]++; MGL_DF_MISS(7 + (wv == 3)); }      /* vlist / vlist-dirty */
      else if (!fok) { df->listwhy[1][wf % 6]++; MGL_DF_MISS(9 + (wf == 3)); }  /* flist / flist-dirty */
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
            {
                /* session 19: a respec that only moves a used binding to a
                 * different buffer object (same layout words, same alignment
                 * class, buffer-backed) needs nothing but the encoder
                 * rebind + a map-suffix rebuild: classify it and let the rest
                 * of the gate run — if nothing else fails we return
                 * MGL_DF_REBIND instead of a full walk */
                if (!vs->plan.valid || !compatRebindClassify(ctx, va, s))
                    MGL_DF_MISS(13);
                rebind = true;
            }
        }
        /* session 17: glVertexAttrib* arms no dirty bit, so the gate must
         * detect generic-attrib writes; every writer bumps current_serial
         * (mgl_ext_vattrib.h), replacing a ~900 B memcmp per draw.  The memo
         * in df records the newest serial whose content this gate (or the
         * latched walk) has already verified unchanged. */
        if (vs->current_serial != df->gate_current_serial)
        {
            if (memcmp(vs->current, s->current, sizeof(s->current)) != 0)
                MGL_DF_MISS(14);
            df->gate_current_serial = vs->current_serial;
        }

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
            /* r154: a vertex buffer the vertex map list carries at its plan slot
             * (the slot the vattrib hook binds it at) is flushed and re-bound by
             * the buffers-only partial walk exactly like a listed uniform buffer:
             * updateDirtyBaseBufferList uploads it, extBufBindList binds the
             * (possibly renamed) storage.  Shadow (converted) streams are rebuilt
             * from the buffer bytes by the vattrib hook only: full walk. */
            bool listed = false;
            if (r154 && allowDirtyListBuffers && !rebind)
            {
                int slot = s->plan.binding_index[b];
                const BufferMapList *vl = &ctx->state.vertex_buffer_map_list;
                listed = slot >= 0 && (GLuint)slot < vl->count && vl->buffers[slot].buf == buf;
                for (int a = 0; listed && a < MAX_ATTRIBS; a++)
                    if (vs->plan.fetch[a] == MGL_VA_FETCH_SHADOW)
                        listed = false;
            }
            if (buf->data.dirty_bits)
            {
                if (!listed)
                    MGL_DF_MISS(17);
                moved = true;               /* re-stream through the buffers path */
            }
            if ((buf->data.mtl_data != NULL) != (s->vabuf[b].mtl != NULL) ||
                (!rebind && (buf->size != s->vabuf[b].size ||
                 (buf->data.mtl_data && buf->data.mtl_data != s->vabuf[b].mtl))))
            {
                /* r154: same size, MTLBuffer-backed on both sides, listed: the
                 * storage moved (pool rename) — re-bind via the buffers path */
                if (!(listed && buf->size == s->vabuf[b].size && buf->data.mtl_data && s->vabuf[b].mtl))
                    MGL_DF_MISS(18);
                moved = true;
            }
        }
    }


    /* tess routing installs its own pipeline and buffer maps: never skip its walks.
     * session 17: the verdict is latched — it can only flip via a link or a
     * program/pipeline rebind, and both fail an earlier check (dirty mask /
     * identity), so a clean same-identity draw reads the cached boolean. */
    if (s->tess_active || [self extTessActive])
        MGL_DF_MISS(25);

    /* session 19: a respec that only re-points a used binding at a different
     * buffer object: rebuild the map suffix + rebind, skip the rest of the
     * walk (everything else about this draw was gate-proven unchanged) */
    if (rebind)
    {
        if (tex_partial)
            MGL_DF_MISS(28);
        mgl_compat_fast_rebind++;
        return MGL_DF_REBIND;
    }

    /* r154: only sampled-texture bindings (plus, possibly, listed-buffer content /
     * storage and the rstate encoder group) moved: re-run the texture branch of
     * the walk and the texext hook, skip mapBuffersToMTL, the VAO branch and the
     * pipeline block (the identity / content checks above prove their inputs
     * unchanged; border-colour emulation re-arms DIRTY_PROGRAM before this gate) */
    if (tex_partial)
    {
        mgl_compat_fast_tex++;
        return MGL_DF_TEXTURES;
    }

    /* everything matched: the walk is a no-op except for listed-buffer content
     * (a listed buffer with dirty data needs the ring re-stream) */
    if (rstate_only)
    {
        mgl_compat_fast_rs++;
        return MGL_DF_RSTATE;
    }

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
        if (dirty || moved)
        {
            mgl_compat_fast_bufs++;
            if (moved)
                mgl_compat_fast_moved++;
            return MGL_DF_BUFFERS;
        }
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
    s->tess_active = [self extTessActive] ? 1 : 0;
    s->sampler_map_gen = ctx->state.sampler_map_gen;
    s->ubo_high_gen = ctx->state.ubo_high_gen;

    {
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        s->rstate_gen = rs ? rs->generation : 0;
        s->rstate_pipe_gen = rs ? rs->generation_pipeline : 0;
    }

    /* session 17: compare-first latching — groups whose live content already
     * equals the snapshot (ideas' 89 same-layout walks per frame) turn into a
     * memcmp instead of a copy; only changed groups are written back. */
    if (memcmp(s->image_units, ctx->state.image_units, sizeof(s->image_units)) != 0)
        memcpy(s->image_units, ctx->state.image_units, sizeof(s->image_units));

    /* gate 26 inputs: the four indexed binding tables (compare-first like the
     * groups above; only a table that actually moved is written back) */
    if (memcmp(&s->bb_ub, &ctx->state.buffer_base[_UNIFORM_BUFFER], sizeof(BufferBase)) != 0)
        memcpy(&s->bb_ub, &ctx->state.buffer_base[_UNIFORM_BUFFER], sizeof(BufferBase));
    if (memcmp(&s->bb_uc, &ctx->state.buffer_base[_UNIFORM_CONSTANT], sizeof(BufferBase)) != 0)
        memcpy(&s->bb_uc, &ctx->state.buffer_base[_UNIFORM_CONSTANT], sizeof(BufferBase));
    if (memcmp(&s->bb_ssbo, &ctx->state.buffer_base[_SHADER_STORAGE_BUFFER], sizeof(BufferBase)) != 0)
        memcpy(&s->bb_ssbo, &ctx->state.buffer_base[_SHADER_STORAGE_BUFFER], sizeof(BufferBase));
    if (memcmp(&s->bb_atom, &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER], sizeof(BufferBase)) != 0)
        memcpy(&s->bb_atom, &ctx->state.buffer_base[_ATOMIC_COUNTER_BUFFER], sizeof(BufferBase));

    compatListLatch(&ctx->state.vertex_buffer_map_list, &s->vlist, s->vb);
    compatListLatch(&ctx->state.fragment_buffer_map_list, &s->flist, s->fb);

    {
        struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
        MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

        if (vs && va)
        {
            if (memcmp(&s->plan, &vs->plan, sizeof(s->plan)) != 0)
                memcpy(&s->plan, &vs->plan, sizeof(s->plan));
            if (memcmp(s->fmt, va->fmt, sizeof(s->fmt)) != 0)
                memcpy(s->fmt, va->fmt, sizeof(s->fmt));
            if (memcmp(s->binding, va->binding, sizeof(s->binding)) != 0)
                memcpy(s->binding, va->binding, sizeof(s->binding));
            if (memcmp(s->current, vs->current, sizeof(s->current)) != 0)
                memcpy(s->current, vs->current, sizeof(s->current));
            s->current_serial = vs->current_serial;   /* gate 14 compares this, not the array */
            df->gate_current_serial = vs->current_serial;
            for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
            {
                const Buffer *buf = va->binding[b].buffer;
                if (s->vabuf[b].buf == buf &&
                    s->vabuf[b].mtl == (buf ? buf->data.mtl_data : NULL) &&
                    s->vabuf[b].size == (buf ? buf->size : 0))
                    continue;
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
            s->tex_pser[u] = tex ? tex->params.mtl_serial : 0;
            s->samp_pser[u] = sam ? sam->params.mtl_serial : 0;
        }
    }

    /* the end-of-draw snapshot stays valid: it records "what was last stamped";
     * if this walk changed the stamped set, the end gate's own compares fail */
}

/* partial fast path (perf session 19): the vertex-array respec only re-pointed
 * used bindings at different buffer objects (gate-classified by
 * compatRebindClassify).  Everything a full walk would do is provably a no-op
 * except: rebuilding the vertex map list's attribute suffix (the entries must
 * carry the new buffer objects) and re-binding the changed entries to the
 * encoder.  The PSO key ignores buffer identity (compacted layout key), the
 * rstate/texext/vattrib-hook groups were gate-proven unchanged, and the
 * bindList rebinds below are exactly the vattrib PreDraw hook's own binds. */
- (bool) compatDrawFastRebind
{
    uint64_t t13 = mgl_prof_begin(30);

    struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;
    struct MGLCompatDrawSnap *s = df ? &df->snap : NULL;
    BufferMapList *list = &ctx->state.vertex_buffer_map_list;

    if (!vs || !va || !s || !ctx->state.vao)
    {
        mgl_prof_end(30, t13);
        return false;
    }

    /* 1. re-point the vertex map list's attribute suffix at the new buffers.
     * The entry set and order (hence every mtl index and the SSBO pad
     * position) are unchanged — the gate proved the used-binding set equal —
     * so only the buffer-object-derived fields of the mapped entries move. */
    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
    {
        int slot = vs->plan.binding_index[b];
        if (slot < 0 || (GLuint)slot >= list->count)
            continue;
        Buffer *nb = va->binding[b].buffer;
        if (!nb)
            continue;
        BufferMap *e = &list->buffers[slot];
        e->buf = nb;
        e->offset = va->binding[b].offset;
        e->size = nb->size;
    }

    /* 2. flush listed-buffer content and re-bind the changed entries: the
     * attribute entries carry the new buffers, the bindList shadow setters
     * issue one setVertexBuffer per changed slot (the exact binds the vattrib
     * PreDraw hook would make after a full walk) and the ring entries
     * memcmp-skip unchanged bytes */
    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list] ||
        ![self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list] ||
        ![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true] ||
        ![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
    {
        mgl_prof_end(30, t13);
        return false;
    }

    compatListClearDataDirty(&ctx->state.vertex_buffer_map_list);
    compatListClearDataDirty(&ctx->state.fragment_buffer_map_list);

    /* 3. consume the redundant dirty bits the full walk would clear: the gate
     * proved every one of their content checks passes for this draw */
    ctx->state.dirty_bits &= ~MGL_DF_REDUNDANT;

    /* 4. partial re-latch: only the binding records, the map entries and the
     * per-binding buffer identities changed (fmt, plan, current values,
     * textures, image units and the rstate generations were all gate-proven
     * equal, so re-copying them would just memcmp-copy unchanged bytes) */
    memcpy(s->binding, va->binding, sizeof s->binding);
    memcpy(s->fmt, va->fmt, sizeof s->fmt);     /* the client pointers moved with the respec */
    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
    {
        Buffer *buf = va->binding[b].buffer;
        if (vs->plan.binding_index[b] < 0 || !buf)
            continue;
        s->vabuf[b].buf = buf;
        s->vabuf[b].mtl = buf->data.mtl_data;
        s->vabuf[b].size = buf->size;
    }
    compatListLatch(&ctx->state.vertex_buffer_map_list, &s->vlist, s->vb);
    compatListLatch(&ctx->state.fragment_buffer_map_list, &s->flist, s->fb);

    mgl_prof_end(30, t13);
    return true;
}

/* partial fast path (perf session 12): every group except listed-buffer
 * CONTENT matches the snapshot — e.g. only a uniform upload dirtied the
 * default uniform block.  Re-stream just the buffers (the ring re-encodes
 * changed bytes, memcmp-skips unchanged ones) and skip the whole walk: no
 * PSO check, no hook fan-out, no stamps (their gates still match). */
- (bool) compatDrawFastBuffers
{
    uint64_t t13 = mgl_prof_begin(30);

    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list] ||
        ![self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list])
    {
        mgl_prof_end(30, t13);
        return false;
    }

    if (![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true] ||
        ![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
    {
        mgl_prof_end(30, t13);
        return false;
    }

    compatListClearDataDirty(&ctx->state.vertex_buffer_map_list);
    compatListClearDataDirty(&ctx->state.fragment_buffer_map_list);

    /* identities did not change, the content dirty bits are now clear: the
     * snapshot is valid again for the next draw */
    [self compatDrawLatch];

    mgl_prof_end(30, t13);
    return true;
}

/* partial fast path (perf session 12): only the rstate encoder-state group
 * changed (generation bump, no pipeline-content change — e.g. a blend-colour /
 * stencil-ref / depth-mask write between draws).  Re-apply the encoder state
 * and skip the rest of the walk. */
- (bool) compatDrawFastRState
{
    uint64_t t30 = mgl_prof_begin(30);
    [self extRstatePreDraw: _currentRenderEncoder];
    /* r154: the gate returns this verdict ahead of the listed-buffer scan, so a
     * uniform upload (or a pool-renamed storage) between the two draws must be
     * re-streamed here as the buffers path would (shadow-filtered: unchanged
     * slots cost a compare) */
    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list] ||
        ![self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list] ||
        ![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true] ||
        ![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
    {
        mgl_prof_end(30, t30);
        return false;
    }
    compatListClearDataDirty(&ctx->state.vertex_buffer_map_list);
    compatListClearDataDirty(&ctx->state.fragment_buffer_map_list);
    [self compatDrawLatch];
    mgl_prof_end(30, t30);
    return true;
}

/* r154: partial walk for a draw whose only moved state is the sampled-texture
 * bindings (compatDrawGate: same program / VAO layout / FBO / pipeline /
 * encoder, texture group changed, program pipeline not texture-dependent).
 * Runs the texture branch of processGLState and the texext PreDraw hook, then
 * the listed-buffer flush and the rstate re-apply when the gate also saw
 * those; mapBuffersToMTL, the VAO branch, the pipeline block and the
 * vattrib hook are skipped.  Returns false (dirty bits still armed) when the
 * texture walk changed the encoder: the caller then runs the full walk. */
- (bool) compatDrawFastTextures
{
    uint64_t t13 = mgl_prof_begin(30);
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    struct MGLCompatDrawFast *df = m ? (struct MGLCompatDrawFast *)m->draw_fast : NULL;
    struct MGLCompatDrawSnap *s = df ? &df->snap : NULL;
    const void *enc0 = (__bridge const void *)_currentRenderEncoder;
    uint64_t gen0 = m ? m->encoder_gen : 0;

    if (!s || !enc0)
    {
        mgl_prof_end(30, t13);
        return false;
    }

    m->texext_skip_epoch = 0;               /* the texext hook below must re-run */

    /* r158: re-bind only the units the gate saw move.  Every other unit's
     * texture, sampler, parameter caches and slot mapping are gate-proven
     * unchanged on this same encoder, and the moved textures are realized
     * (gate), so neither the unit realization walk nor the other slots run. */
    bool masked = !df->tex_full && compatUnitsOn();
    if (masked)
    {
        Program *p = ctx->state.program;
        if (df->sep_prog != (const void *)p)
        {
            df->sep_prog = p;
            df->sep = false;
            for (int st = 0; st < _MAX_SHADER_TYPES && p; st++)
                if (p->spirv_resources_list[st][SPVC_RESOURCE_TYPE_SEPARATE_IMAGE].count ||
                    p->spirv_resources_list[st][SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS].count)
                    df->sep = true;
        }
        masked = !df->sep;
    }
    if (masked)
        mgl_compat_fast_texunits++;

    uint64_t t15 = mgl_prof_begin(15);
    bool ok = masked || [self bindActiveTexturesToMTL];
    if (ok)
    {
        uint64_t p5 = mgl_prof_begin(5);
        ok = [self bindTexturesToCurrentRenderEncoderUnits: masked ? df->tex_units : NULL];
        mgl_prof_end(5, p5);
    }
    mgl_prof_end(15, t15);
    if (!ok || (__bridge const void *)_currentRenderEncoder != enc0 || m->encoder_gen != gen0)
    {
        mgl_prof_end(30, t13);
        return false;
    }
    ctx->state.dirty_bits &= ~(DIRTY_TEX | DIRTY_TEX_BINDING | DIRTY_SAMPLER | MGL_DF_REDUNDANT);

    [self extTexextPreDraw: _currentRenderEncoder];

    struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
    if (rs && rs->generation != s->rstate_gen)
        [self extRstatePreDraw: _currentRenderEncoder];

    if (![self updateDirtyBaseBufferList: &ctx->state.vertex_buffer_map_list] ||
        ![self updateDirtyBaseBufferList: &ctx->state.fragment_buffer_map_list] ||
        ![self extBufBindList: &ctx->state.vertex_buffer_map_list vertex: true] ||
        ![self extBufBindList: &ctx->state.fragment_buffer_map_list vertex: false])
    {
        mgl_prof_end(30, t13);
        return false;
    }
    compatListClearDataDirty(&ctx->state.vertex_buffer_map_list);
    compatListClearDataDirty(&ctx->state.fragment_buffer_map_list);

    [self compatDrawLatch];
    mgl_prof_end(30, t13);
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

    /* texext (r48): texture-buffer read stamps are outside every list this
     * snapshot compares — never skip the stamp walk while one can be bound */
    if (ctx->state.ext_texext && ctx->state.ext_texext->buffer_texture_count)
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

/* session 19: C-level pre-check for extBufBindList — skips the whole
 * storage/ring dispatch + ObjC bind call when the compat shadow already has
 * this (buffer, offset) bound at this index on this encoder.  Mirrors
 * compatBindVertexBuffer's own shadow test exactly. */
bool compatVbufAlreadyBound(GLMContext ctx, unsigned idx, const void *mtl, size_t offset, void *enc_ptr)
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)enc_ptr;
    if (!m || !enc || idx >= MGLC_BIND_SLOTS || !mtl || !enc_ptr)
        return false;
    struct MGLCompatBindShadow *sh = &m->vbuf[idx];
    uint64_t gen = compatEncGen(m, enc_ptr);
    return sh->obj == mtl && sh->offset == offset &&
           sh->gen == gen && sh->encoder == enc_ptr;
}

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
        e->tex_params == tex->params.mtl_data && e->tex_pser == tex->params.mtl_serial &&
        e->sampler_obj == so &&
        (!so || (e->sampler_params == ((Sampler *)so)->params.mtl_data &&
                 e->samp_pser == ((Sampler *)so)->params.mtl_serial)) && e->view && e->samp)
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
    e->tex_pser = tex->params.mtl_serial;
    e->samp_pser = so ? ((Sampler *)so)->params.mtl_serial : 0;
    e->view = (__bridge void *)view;
    e->samp = (__bridge void *)samp;
}

/* -------- GL_EXT_texture_sRGB_decode (compat r102) ------------------- */

/* the plain (non-decoding) counterpart of an sRGB Metal pixel format, or
 * MTLPixelFormatInvalid when the format does not carry an sRGB transfer */
static MTLPixelFormat compatSRGBPlainFormat(MTLPixelFormat f)
{
    switch (f)
    {
        case MTLPixelFormatRGBA8Unorm_sRGB:  return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm_sRGB:  return MTLPixelFormatBGRA8Unorm;
        case MTLPixelFormatR8Unorm_sRGB:     return MTLPixelFormatR8Unorm;
        case MTLPixelFormatRG8Unorm_sRGB:    return MTLPixelFormatRG8Unorm;
        case MTLPixelFormatBC1_RGBA_sRGB:    return MTLPixelFormatBC1_RGBA;
        case MTLPixelFormatBC2_RGBA_sRGB:    return MTLPixelFormatBC2_RGBA;
        case MTLPixelFormatBC3_RGBA_sRGB:    return MTLPixelFormatBC3_RGBA;
        case MTLPixelFormatBC7_RGBAUnorm_sRGB: return MTLPixelFormatBC7_RGBAUnorm;
        case MTLPixelFormatETC2_RGB8_sRGB:   return MTLPixelFormatETC2_RGB8;
        case MTLPixelFormatETC2_RGB8A1_sRGB: return MTLPixelFormatETC2_RGB8A1;
        case MTLPixelFormatEAC_RGBA8_sRGB:   return MTLPixelFormatEAC_RGBA8;
        case MTLPixelFormatASTC_4x4_sRGB:    return MTLPixelFormatASTC_4x4_LDR;
        case MTLPixelFormatASTC_5x4_sRGB:    return MTLPixelFormatASTC_5x4_LDR;
        case MTLPixelFormatASTC_5x5_sRGB:    return MTLPixelFormatASTC_5x5_LDR;
        case MTLPixelFormatASTC_6x5_sRGB:    return MTLPixelFormatASTC_6x5_LDR;
        case MTLPixelFormatASTC_6x6_sRGB:    return MTLPixelFormatASTC_6x6_LDR;
        case MTLPixelFormatASTC_8x5_sRGB:    return MTLPixelFormatASTC_8x5_LDR;
        case MTLPixelFormatASTC_8x6_sRGB:    return MTLPixelFormatASTC_8x6_LDR;
        case MTLPixelFormatASTC_8x8_sRGB:    return MTLPixelFormatASTC_8x8_LDR;
        case MTLPixelFormatASTC_10x5_sRGB:   return MTLPixelFormatASTC_10x5_LDR;
        case MTLPixelFormatASTC_10x6_sRGB:   return MTLPixelFormatASTC_10x6_LDR;
        case MTLPixelFormatASTC_10x8_sRGB:   return MTLPixelFormatASTC_10x8_LDR;
        case MTLPixelFormatASTC_10x10_sRGB:  return MTLPixelFormatASTC_10x10_LDR;
        case MTLPixelFormatASTC_12x10_sRGB:  return MTLPixelFormatASTC_12x10_LDR;
        case MTLPixelFormatASTC_12x12_sRGB:  return MTLPixelFormatASTC_12x12_LDR;
        default: break;
    }
    return MTLPixelFormatInvalid;
}

/* SKIP_DECODE sampling: hand back a plain (non-sRGB) VIEW of <base> so Metal's
 * automatic sRGB->linear conversion does not run (the EXT extension's raw
 * sample values).  The effective decode state follows GL sampler-object
 * semantics (and Mesa's st_sampler_view.c): the sampler object bound to the
 * unit replaces the texture's own parameter.  Callers pass the GL unit so the
 * effective state resolves per unit; the view is cached per unit, keyed on
 * every identity the decision reads.  Returns <base> unchanged when decoding
 * stays enabled or the format has no plain counterpart. */
- (id<MTLTexture>) compatSRGBDecodeViewForTex: (Texture *)tex base: (id<MTLTexture>)base unit: (GLuint)unit
{
    const GLenum decode = GL_DECODE_EXT;    const TextureParameter *sp = NULL;

    if (!tex || !base || unit >= 32)
        return base;

    /* effective sampler parameters for this unit */
    Sampler *so = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;
    sp = so ? &so->params : &tex->params;
    if (sp->srgb_decode != GL_SKIP_DECODE_EXT)
        return base;

    MTLPixelFormat plain = compatSRGBPlainFormat(base.pixelFormat);
    if (plain == MTLPixelFormatInvalid || plain == base.pixelFormat)
        return base;            /* nothing to skip */

    struct MGLExtMetal_compat *m = compatMetal(ctx);
    if (!m)
        return base;
    struct MGLCompatSrgbView *e = &m->srgbview[unit];
    void *so_id = (void *)so;
    void *so_params = so ? so->params.mtl_data : NULL;
    if (e->view && e->tex == (void *)tex && e->tex_mtl == tex->mtl_data &&
        e->tex_params == tex->params.mtl_data && e->sampler_obj == so_id &&
        e->sampler_params == so_params)
        return (__bridge id<MTLTexture>)e->view;

    id<MTLTexture> view = [base newTextureViewWithPixelFormat: plain];
    if (!view)
        return base;            /* PixelFormatView usage missing: decode, never wrong values */

    if (e->view)
        CFRelease(e->view);
    e->tex = (void *)tex;
    e->tex_mtl = tex->mtl_data;
    e->tex_params = tex->params.mtl_data;
    e->sampler_obj = so_id;
    e->sampler_params = so_params;
    e->view = (void *)CFBridgingRetain(view);
    return view;
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

    /* r48: no pointer-skip for samplers — Metal recycles MTLSamplerState
     * addresses, so a fresh state at a recycled address would be dropped as
     * "already bound" (stale GL_TEXTURE_LOD_BIAS / filters;
     * KHR-GL46.texture_lod_bias_all).  The texture-resolve gate already
     * suppresses redundant walk calls, so this costs an idempotent Metal set. */
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

    /* r48: no pointer-skip for samplers — Metal recycles MTLSamplerState
     * addresses, so a fresh state at a recycled address would be dropped as
     * "already bound" (stale GL_TEXTURE_LOD_BIAS / filters). */
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

/* ------------------------------------------------------------------ */
/* GL_KHR_blend_equation_advanced (GL 4.6 core §17.3.8) — the fragment  */
/* variant that composites through a [[color(0)]] framebuffer fetch.   */
/* See compat_advblend.c; zero work unless an advanced equation is set. */

- (bool) compatAdvancedBlendVariant: (Shader *)shader
                                msl: (const char **)out_msl
                            library: (id<MTLLibrary> *)out_lib
                          function: (id<MTLFunction> *)out_fn
{
    struct MGLExtMetal_compat *m = compatMetal(ctx);
    int mode = mgl_advblend_active(ctx);
    Program *program;
    const char *msl, *entry;

    *out_msl = NULL;
    *out_lib = nil;
    *out_fn = nil;

    if (mode < 0 || !m || !shader)
        return false;

    program = ctx->state.program;
    if (!program)
        return false;                       /* pipelines: composite program */
    msl = program->spirv[_FRAGMENT_SHADER].msl_str;
    entry = program->spirv[_FRAGMENT_SHADER].entry_point;
    if (!msl || !entry)
        return false;


    /* the [[color(0)]] fetch needs a color attachment 0 whose format can be
     * read as float4 (non-integer — GL ignores blending for integer targets
     * anyway).  Integer formats would make pipeline validation reject the
     * float4 fetch argument. */
    {
        id<MTLTexture> att = _renderPassDescriptor.colorAttachments[0].texture;
        if (!att || att.pixelFormat == MTLPixelFormatInvalid)
            return false;
        switch (att.pixelFormat)
        {
            case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint: case MTLPixelFormatR16Uint:
            case MTLPixelFormatR16Sint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
            case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatRG16Uint:
            case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint:
            case MTLPixelFormatRGB10A2Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint:
            case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Uint:
            case MTLPixelFormatRGBA32Sint:
                return false;
            default:
                break;
        }
        /* MSAA: the read is only defined per sample — rstateSpecialize forces
         * per-sample shading while the advanced variant is in use */
    }

    /* cache hit on (MSL identity, equation)? */
    if (m->adv_msl_src == (const void *)msl && m->adv_mode == mode && m->adv_failed)
        return false;
    if (m->adv_msl_src == (const void *)msl && m->adv_mode == mode && m->adv_lib)
    {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)m->adv_lib;
        id<MTLFunction> fn = [lib newFunctionWithName: [NSString stringWithUTF8String: entry]];
        if (fn)
        {
            *out_msl = m->adv_msl;
            *out_lib = lib;
            *out_fn = fn;
            return true;
        }
    }

    char *v = mgl_advblend_rewrite(msl, entry, mode);
    if (!v)
    {
        if (getenv("MGL_ADVBLEND_DEBUG"))
            fprintf(stderr, "MGL advblend: cannot rewrite fragment MSL %s for %s\n",
                    entry, mgl_advblend_mode_name(mode));
        free(m->adv_msl);        m->adv_msl = NULL;
        m->adv_msl_src = (const void *)msl;
        m->adv_mode = mode;
        m->adv_failed = 1;
        if (m->adv_lib) { CFRelease(m->adv_lib); m->adv_lib = NULL; }
        return false;
    }

    char err[256] = "";
    id<MTLLibrary> lib = [self extProgramLibraryForMSL: v key: nil error: (char **)&err];
    id<MTLFunction> fn = nil;
    if (lib)
        fn = [lib newFunctionWithName: [NSString stringWithUTF8String: entry]];
    if (!fn)
    {
        fprintf(stderr, "MGL advblend: %s library/function failed: %s\n",
                mgl_advblend_mode_name(mode), err[0] ? err : "no entry");
        free(v);
        m->adv_msl_src = (const void *)msl;
        m->adv_mode = mode;
        m->adv_failed = 1;
        if (m->adv_lib) { CFRelease(m->adv_lib); m->adv_lib = NULL; }
        return false;
    }

    /* remember the rewrite so repeated pipeline rebuilds (blend toggles,
     * encoder churn) do not re-walk the MSL */
    if (m->adv_msl && m->adv_msl != v)
        free(m->adv_msl);
    if (m->adv_lib && m->adv_lib != (__bridge void *)lib)
        CFRelease(m->adv_lib);
    m->adv_msl_src = (const void *)msl;
    m->adv_mode = mode;
    m->adv_failed = 0;
    m->adv_msl = v;
    m->adv_lib = (void *)CFBridgingRetain(lib);

    {
        const char *dbg = getenv("MGL_ADVBLEND_DEBUG");
        if (dbg && dbg[0] >= '2')
            fprintf(stderr, "===== MGL advblend MSL (%s) =====\n%s\n=====\n",
                    mgl_advblend_mode_name(mode), v);
    }

    *out_msl = v;
    *out_lib = lib;
    *out_fn = fn;
    return true;
}

/* dispatch the force-opaque kernel on tex (same command buffer, before the
 * commit that presents it) */
- (void) compatForceOpaque: (id<MTLTexture>)tex
{
    MGLCompatSurfacePool *pool = objc_getAssociatedObject(self, kCompatPoolKey);
    if (!pool || !tex || tex.width == 0)
        return;
    if (!pool.opaquePso)
    {
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:s_mgl_force_opaque_msl]
                                                   options:nil
                                                     error:&err];
        id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"mgl_force_opaque"] : nil;
        pool.opaquePso = fn ? [_device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!pool.opaquePso)
        {
            NSLog(@"MGL INFO: force-opaque kernel failed: %@", err.localizedDescription);
            pool.opaquePsoFailed = YES;                    /* do not retry */
            return;
        }
    }
    if (pool.opaquePsoFailed)
        return;                                             /* kernel unavailable */
    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
    if (!enc)
        return;
    enc.label = @"MGL force opaque";
    [enc setComputePipelineState:pool.opaquePso];
    [enc setTexture:tex atIndex:0];
    MTLSize tg = MTLSizeMake(16, 16, 1);
    MTLSize g = MTLSizeMake((tex.width + 15) / 16, (tex.height + 15) / 16, 1);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:tg];
    [enc endEncoding];
}

/* present-copy fast path: one render pass copies src -> dst row-for-row (same
 * orientation as the blit it replaces; the presenting layer's transform does
 * the y-flip) and writes alpha=1.  dst then goes old-frame -> opaque-new-frame
 * in a SINGLE write, which is exactly the e4142bb flash-fix property — without
 * the per-present compute kernel (kernel over the source + blit = two encoders,
 * two compatCSDrain calls and two full-surface GPU passes; this is one encoder,
 * one drain, one pass) and without touching the source's alpha (GL readback of
 * the default framebuffer keeps the blended alpha values).
 * Returns false when the caller must fall back to blit/fbShaderBlitFrom. */
- (bool) compatPresentCopyOpaque: (id<MTLTexture>)src to:(id<MTLTexture>)dst
{
    MGLCompatSurfacePool *pool = objc_getAssociatedObject(self, kCompatPoolKey);
    if (!pool || !src || !dst || src.width == 0 || src.height == 0)
        return false;
    /* exact-copy cases only (the blit path's own condition); integer/unusual
     * source formats would need a typed texture2d<> binding — stay out */
    if (src.width != dst.width || src.height != dst.height ||
        src.pixelFormat != dst.pixelFormat)
        return false;
    switch (src.pixelFormat)
    {
        case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatRGBA16Float:
            break;
        default:
            return false;
    }

    if (!pool.presentPso || pool.presentPsoFormat != dst.pixelFormat)
    {
        if (pool.presentPsoFailed && pool.presentPsoFormat == dst.pixelFormat)
            return false;                                   /* do not retry */
        NSError *err = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:s_mgl_present_copy_msl]
                                                   options:nil
                                                     error:&err];
        id<MTLFunction> vs = lib ? [lib newFunctionWithName:@"mgl_present_vert"] : nil;
        id<MTLFunction> fs = lib ? [lib newFunctionWithName:@"mgl_present_frag"] : nil;
        MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = vs;
        pd.fragmentFunction = fs;
        pd.colorAttachments[0].pixelFormat = dst.pixelFormat;
        id<MTLRenderPipelineState> ps = [_device newRenderPipelineStateWithDescriptor:pd
                                                                                error:&err];
        if (!ps)
        {
            NSLog(@"MGL INFO: present-copy pipeline failed: %@", err.localizedDescription);
            pool.presentPsoFailed = YES;                    /* do not retry */
            pool.presentPsoFormat = dst.pixelFormat;
            return false;
        }
        pool.presentPso = ps;
        pool.presentPsoFormat = dst.pixelFormat;
    }

    /* compat CS: this pass runs on the GL thread — the encoder worker must be
     * done before a second encoder opens on this command buffer */
    [self compatCSDrain];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;   /* every pixel is written */
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    if (!enc)
        return false;
    enc.label = @"MGL present copy";
    [enc setRenderPipelineState:pool.presentPso];
    [enc setViewport:(MTLViewport){ 0, 0, (double)dst.width, (double)dst.height, 0.0, 1.0 }];
    [enc setFragmentTexture:src atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    return true;
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
        /* GL window semantics: the framebuffer alpha is discarded at present
         * (the window is opaque).  CA composites layer.contents with its
         * alpha unless the layer is marked opaque — without this the
         * translucent jellyfish region (dst alpha = a*a + (1-a) after
         * blending, e.g. 191) is composited over the white window backdrop
         * and the whole bell washes out (glmark2 jellyfish caustics). */
        pool.layer.opaque = YES;
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
        [self compatCSDrain];   /* see the blit path: no concurrent encoders */
        [self compatForceOpaque:pool.textures[di]];
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

    /* FLASH (user: "camera flash" frames in [jellyfish]) — the presented
     * IOSurface (layer.contents) is sampled by the compositor asynchronously:
     * the swap command buffer is not synchronized with it, so the surface must
     * never hold a half-alpha frame the compositor could catch.  e4142bb fixed
     * that by forcing the copy SOURCE opaque first (one extra compute pass per
     * present: kernel over the source, then blit); compatPresentCopyOpaque:to:
     * keeps the property with a SINGLE render pass that copies AND writes
     * alpha=1 — dst's only write this frame is already opaque, the source's
     * GL alpha is untouched, and the per-present cost is one encoder + one
     * full-surface pass instead of two of each. */
    if (![self compatPresentCopyOpaque:color to:dst])
    {
        /* unsupported (size/format race, exotic format): the old path —
         * shader-blit WITHOUT the classic y-flip (the presenting layer's
         * transform already flips, a flipped copy would land upside down on
         * screen), then force the POOL surface's alpha (rare path; its small
         * compositor window is unchanged) */
        CompatFBTarget src = { color, 0, 0, 0, 1 }, dt = { dst, 0, 0, 0, 1 };
        GLint s[4] = { 0, 0, (GLint)color.width, (GLint)color.height };
        GLint d[4] = { 0, 0, (GLint)dst.width, (GLint)dst.height };
        [self compatCSDrain];
        [self fbShaderBlitFrom:&src to:&dt src:s dst:d
                        filter:GL_LINEAR
                       scissor:false];
        [self compatForceOpaque:dst];
    }
    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    [self commitCommandBufferWithAGXRecovery:cb];

    /* contents swap is the "present": cheap, no server round trip; the
     * compositor shows this surface at its next frame */
    pool.layer.frame = _layer.bounds;
    pool.layer.contents = pool.surfaces[i];

    [self newCommandBuffer];
    mgl_prof_end(6, p6);

    /* MGL_PRESENT_DUMP=<n>: dump the just-presented surface (post-commit, no
     * encoder is open) to /tmp/mgl_present_<n>.ppm — what the user actually
     * sees, composited by CA from this IOSurface */
    {
        static int pdump_env = -1;
        if (pdump_env < 0)
        {
            const char *e = getenv("MGL_PRESENT_DUMP");
            pdump_env = e ? atoi(e) : 0;
        }
        if (pdump_env > 0)
        {
            static int pdump_cnt = 0;
            if (++pdump_cnt % pdump_env == 0)
            {
                MTLRegion r = MTLRegionMake2D(0, 0, dst.width, dst.height);
                NSUInteger rb = 4 * dst.width;
                uint8_t *buf = (uint8_t *)malloc(rb * dst.height);
                if (buf)
                {
                    [dst getBytes:buf bytesPerRow:rb fromRegion:r mipmapLevel:0];
                    char path[128];
                    snprintf(path, sizeof path, "/tmp/mgl_present_%d.ppm", pdump_cnt / pdump_env);
                    FILE *f = fopen(path, "wb");
                    if (f)
                    {
                        FILE *fa;
                        char apath[128];
                        fprintf(f, "P6\n%u %u\n255\n", (unsigned)dst.width, (unsigned)dst.height);
                        for (NSUInteger y = 0; y < dst.height; y++)
                            for (NSUInteger x = 0; x < dst.width; x++)
                            {
                                uint8_t *p = &buf[y * rb + x * 4];
                                fputc(p[0], f); fputc(p[1], f); fputc(p[2], f);
                            }
                        fclose(f);
                        snprintf(apath, sizeof apath, "/tmp/mgl_present_alpha_%d.pgm", pdump_cnt / pdump_env);
                        fa = fopen(apath, "wb");
                        if (fa)
                        {
                            fprintf(fa, "P5\n%u %u\n255\n", (unsigned)dst.width, (unsigned)dst.height);
                            for (NSUInteger y = 0; y < dst.height; y++)
                                for (NSUInteger x = 0; x < dst.width; x++)
                                    fputc(buf[y * rb + x * 4 + 3], fa);
                            fclose(fa);
                        }
                        fprintf(stderr, "[presentdump] %s %ux%u\n", path,
                                (unsigned)dst.width, (unsigned)dst.height);
                    }
                    free(buf);
                }
            }
        }
    }

    [self compatCSFrameStats];       /* compat CS: per-frame stats hook */
    mgl_prof_frame();
    return true;
}

#pragma mark - aliased SSBO dispatch variants (compat_ssbo.c)

static char kSSBOAliasVarKey;

/* Compute-function variant whose ALIASED shader storage blocks share one Metal
 * pointer (see compat_ssbo.c): Metal's compiler assumes distinct [[buffer(n)]]
 * argument pointers never alias, so GL's intra-invocation ordering through two
 * blocks bound to the same buffer range (GL 4.6 §7.12) breaks — the first
 * block's store is forwarded to the later load
 * (dEQP ES31 synchronization.in_invocation.ssbo_alias_overwrite).
 *
 * Returns nil (and costs only the small C walk of the stage's storage-block
 * bindings) unless two storage blocks are bound to the identical buffer range
 * right now; then a per-signature variant of the entry is compiled once and
 * cached on the base function.  MGL_NO_SSBO_ALIAS=1 switches it off (A/B). */
- (id<MTLFunction>) compatSSBOAliasComputeFunction: (Shader *) cs program: (Program *) program
                                                                   map: (BufferMapList *) map
{
    static int sOff = -1;
    unsigned from[16], to[16];
    int npairs;
    id<MTLFunction> base;
    NSMutableDictionary *cache;
    char sig[16 * 16 + 1];
    NSString *key;
    id cached;

    if (sOff < 0)
        sOff = getenv("MGL_NO_SSBO_ALIAS") != NULL;
    if (sOff)
        return nil;
    if (!cs || !cs->mtl_data.function || !program)
        return nil;

    npairs = mgl_compat_ssbo_alias_pairs(ctx, map, from, to, 16);
    if (npairs <= 0)
        return nil;

    /* signature: the variant is a pure function of which block folds onto which */
    {
        size_t o = 0;
        int n = snprintf(sig + o, sizeof sig - o, "%u>%u", from[0], to[0]);
        if (n < 0) return nil;
        o += (size_t)n;
        for (int i = 1; i < npairs && o + 1 < sizeof sig; i++)
        {
            n = snprintf(sig + o, sizeof sig - o, ",%u>%u", from[i], to[i]);
            if (n < 0) return nil;
            o += (size_t)n;
        }
    }

    base = (__bridge id<MTLFunction>)cs->mtl_data.function;
    cache = objc_getAssociatedObject(base, &kSSBOAliasVarKey);
    if (!cache)
    {
        cache = [NSMutableDictionary new];
        objc_setAssociatedObject(base, &kSSBOAliasVarKey, cache, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    key = [NSString stringWithUTF8String: sig];
    cached = cache[key];
    if (cached)
        return cached[1];

    {
        const char *msl = program->spirv[_COMPUTE_SHADER].msl_str;
        const char *entry = program->spirv[_COMPUTE_SHADER].entry_point;
        char *rewritten, *err = NULL;
        void *lib = NULL, *fn = NULL;
        bool ok;

        if (!msl || !entry || !ctx->mtl_funcs.ext_program || !ctx->mtl_funcs.ext_program->compile_msl)
            return nil;

        rewritten = mgl_compat_msl_alias_ssbo(msl, entry, from, to, npairs);
        if (!rewritten)
        {
            if (getenv("MGL_SSBO_DEBUG"))
                NSLog(@"MGL compat: aliased-SSBO rewrite declined for entry %s (%d pairs)", entry, npairs);
            return nil;
        }

        ok = ctx->mtl_funcs.ext_program->compile_msl(ctx, rewritten, entry, &lib, &fn, &err);
        if (!ok)
        {
            if (getenv("MGL_SSBO_DEBUG"))
                NSLog(@"MGL compat: aliased-SSBO variant compile failed: %s", err ? err : "?");
            free(err);
            free(rewritten);
            return nil;
        }
        free(err);
        if (getenv("MGL_DUMP_MSL") || getenv("MGL_SSBO_DEBUG"))
            fprintf(stderr, "MGL: compat aliased-SSBO variant of entry %s (sig %s):\n%s\n", entry, sig, rewritten);
        free(rewritten);

        /* keep library + function alive with the cache entry (the compile hook
         * returned both retained) */
        cache[key] = @[ (__bridge_transfer id)lib, (__bridge_transfer id<MTLFunction>)fn ];
        return (__bridge id<MTLFunction>)fn;
    }
}

@end
