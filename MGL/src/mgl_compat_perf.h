/* mgl_compat_perf.h — per-renderer Metal argument-bind shadow + streaming
 * upload ring (compat perf session 3; see agents/compat_report.md).
 *
 * Plain C, no GLMContext dependency, so both mgl_ext_compat.h (which embeds it
 * in struct MGLExtMetal_compat) and the renderer category can include it.
 *
 * Two mechanisms, both keyed on the identity of the current render encoder:
 *
 *  - bind shadow: one entry per Metal argument index (buffers, textures,
 *    samplers; 0..30) recording which object+offset the *encoder* last
 *    received.  Re-issuing an identical setter is skipped, so a draw that
 *    changes nothing between binds costs zero Metal encode calls.  The shadow
 *    is invalidated wholesale by the encoder-generation counter, which is
 *    bumped every time a render encoder is created (extCompatRenderEncoderCreated:).
 *
 *  - upload ring: per-draw byte payloads that used to go through
 *    setVertexBytes/setFragmentBytes (default uniform blocks, client vertex
 *    arrays, constant vertex attributes, copied index lists) are memcpy'd
 *    into one of MGLC_RING_SLOTS shared MTLBuffers instead, 256-byte aligned,
 *    and handed to the encoder with offset-only updates
 *    (setVertexBufferOffset/setFragmentBufferOffset), which are much cheaper
 *    to encode than a by-value bytes argument.  The previous allocation of an
 *    argument index is memcmp'd first: unchanged bytes skip the bind entirely.
 *    A ring slot is recycled only after the command buffer serial that read
 *    it has completed (extBufWaitSerial), so the GPU never observes
 *    overwritten bytes.
 */
#ifndef mgl_compat_perf_h
#define mgl_compat_perf_h

#include <stdint.h>
#include <stddef.h>

/* Metal argument table indices (buffers/textures/samplers) run 0..30 */
#define MGLC_BIND_SLOTS 32

/* upload ring geometry: 4 x 512 KB = 2 MB of streaming argument data before a
 * slot is recycled; the trailing 4 KB of each slot stay unused as an
 * over-read guard for shaders that pad their argument structs */
#define MGLC_RING_SLOTS 4
#define MGLC_RING_SLOT_BYTES (512u * 1024u)
#define MGLC_RING_GUARD_BYTES 4096u

/* payloads above this stay on the by-value setBytes path (matches the
 * historical 4 KB cap of the small-buffer binds the ring replaces) */
#define MGLC_RING_MAX_BYTES 4096u

struct MGLCompatBindShadow {
    void    *obj;                   /* id<MTLBuffer/MTLTexture/MTLSamplerState> (not retained) */
    size_t   offset;                /* buffer offset (ring sub-allocation or GL bind offset) */
    uint64_t gen;                   /* encoder generation the entry was recorded in */
    void    *encoder;               /* encoder the entry was recorded for */
};

struct MGLCompatRing {
    void    *slot[MGLC_RING_SLOTS]; /* retained id<MTLBuffer> (CFBridgingRetain) */
    uint8_t *cpu[MGLC_RING_SLOTS];  /* slot contents pointer */
    uint64_t serial[MGLC_RING_SLOTS]; /* command-buffer serial that last read the slot */
    unsigned cur;                   /* slot receiving allocations */
    size_t   used;                  /* bytes allocated in the current slot */
};

/* Metal hooks of the compat domain.  Allocated in
 * -bindExtCompatFuncsToGLMContext: (MGLRenderer+Compat.m) for every renderer,
 * both profiles; NULL until then (helpers fall back to unfiltered binds). */
struct MGLExtMetal_compat {
    int      enabled;               /* kill switch: 0 = plain binds (MGL_BIND_SHADOW=0) */
    int      ring_on;               /* ring slots created and usable */
    uint64_t encoder_gen;           /* bumped on every render encoder creation */

    struct MGLCompatBindShadow vbuf[MGLC_BIND_SLOTS];
    struct MGLCompatBindShadow fbuf[MGLC_BIND_SLOTS];
    struct MGLCompatBindShadow vtex[MGLC_BIND_SLOTS];
    struct MGLCompatBindShadow ftex[MGLC_BIND_SLOTS];
    struct MGLCompatBindShadow vsamp[MGLC_BIND_SLOTS];
    struct MGLCompatBindShadow fsamp[MGLC_BIND_SLOTS];

    void    *seen_encoder;           /* last encoder the shadow recorded binds for */

    /* pipeline fast path: content hash of the pipeline-baked rstate, cached
     * against generation_pipeline (see compatPipelineFastPath) */
    uint64_t rstate_hash;
    uint64_t rstate_hash_gen;

    /* pipeline fast path: attachment formats + sample count of the current
     * encoder, hashed once per encoder generation (attachment changes always
     * end the encoder) */
    uint64_t fmt_hash;
    uint64_t fmt_hash_gen;
    uint64_t fmt_fbo;

    /* pipeline fast path: the two expensive key components are cached against
     * their inputs and revalidated with a pointer + memcmp (an order of
     * magnitude cheaper than hashing 3x320 bytes of MSL text and walking every
     * VAO attribute on EVERY draw):
     *  - MSL content key (compatMSLKey): direct-mapped on the msl_str pointer,
     *    validated by the first/last 128 bytes (a recycled pointer with equal
     *    head AND tail but different middle aliases to a stale key — same
     *    residual risk the uncached pointer+head192+tail128 hash accepted);
     *  - VAO layout hash (compatVAOHash): single entry, validated by a full
     *    memcmp of the hashed VertexArray bytes, so it is exact. */
    struct MGLCompatMSLKeyEnt {
        const char *msl;
        uint64_t    key;
        size_t      len;
        uint8_t     head[128], tail[128];
    } mslkey[32];
    struct {
        void    *vao;
        uint64_t hash;
        size_t   len;
        uint8_t *snap;              /* malloc'd copy of the hashed bytes */
    } vaokey;

    /* per-walk uniform/storage buffer-map PREFIX cache (compat perf session
     * 13): mapBuffersToMTL re-enumerates the program's SPIR-V buffer
     * resources on every dirty walk, but the UBO / SSBO / atomic-counter /
     * default-uniform prefix of the two stage lists only changes with the
     * program (relink arms DIRTY_PROGRAM) or a buffer_base rebind — never
     * with the VAO re-specification that triggers most walks.  The layout is
     * private to MGLRenderer+Compat.m (it needs GL types); this is the
     * malloc'd struct MGLCompatBindmapCache, NULL until first use. */
    void    *bindmap_cache;

    /* per-draw texture RESOLUTION cache (compat perf session 8): the bind
     * shadow above filters the encoder CALLS; this filters the CPU-side
     * resolution chain (texextValidate + texdata realize + sampler lookup +
     * texture-view lookup — five ObjC calls per texture per stage per draw in
     * glmark2 [ideas]) when the unit still feeds the same realized texture.
     * Keyed on the identity of everything that chain reads:
     *   - the GL texture object (the loop's mgl_texture_for_sampler result),
     *   - tex->mtl_data (re-realization / storage re-specification),
     *   - tex->params.mtl_data — ANY glTexParameter releases it
     *     (mglTexParamInvalidate), so a param change always misses,
     *   - the unit's sampler object and ITS params.mtl_data (same rule),
     *   - the encoder generation (a fresh encoder has an empty table).
     * The resolved ids stay unretained: the GL objects own them. */
    struct MGLCompatTexResolve {
        uint64_t gen;
        void    *tex;
        void    *tex_mtl;
        void    *tex_params;
        void    *sampler_obj;
        void    *sampler_params;
        void    *view;      /* resolved id<MTLTexture> */
        void    *samp;      /* resolved id<MTLSamplerState> */
    } vtexres[2][MGLC_BIND_SLOTS];

    struct MGLCompatRing ring;

    /* compat perf session 12: per-draw state-walk fast path.  Opaque here (it
     * snapshots GL-side structs this header must not depend on); allocated and
     * owned by MGLRenderer+Compat.m (struct MGLCompatDrawFast). */
    void    *draw_fast;

    /* compat perf session 12: texext PreDraw skip epoch.  compatDrawGate proves
     * the texture state clean for a draw (either no texture dirty bits armed, or
     * the unit/identity checks pass) and stamps texext_skip_epoch; the texext
     * PreDraw hook early-outs while the epoch, program and encoder generation
     * are unchanged (glmark2: 89 VAO-changed draws per frame no longer re-walk
     * every sampler binding) */
    uint64_t texext_epoch_seq, texext_skip_epoch, texext_epoch_applied;
    void    *texext_prog;
    uint64_t texext_encoder_gen;

    /* compat_advblend (GL_KHR_blend_equation_advanced): the framebuffer-fetch
     * fragment variant of the program's MSL for the current equation — one
     * entry, keyed on (source MSL pointer, equation mode).  adv_msl is the
     * malloc'd rewrite (NULL = this combination cannot be rewritten, remember
     * that), adv_lib the (id<MTLLibrary>) it compiled to (extProgramLibraryForMSL
     * dedupes by text, this entry just avoids re-walking the MSL per pipeline
     * rebuild).  Only touched while an advanced equation is active. */
    const void *adv_msl_src;
    int         adv_mode;
    char       *adv_msl;
    void       *adv_lib;
    int         adv_failed;
};

#endif /* mgl_compat_perf_h */
