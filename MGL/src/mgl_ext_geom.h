/* mgl_ext_geom.h — private state and Metal hooks of the 'geom' domain. Owned by agent 'geom'.
 *
 * Mechanism (Metal has no geometry stage):
 *   1. link time (mgl_geom_compile_stage / mgl_geom_post_link):
 *      - the geometry stage is reflected from its SPIR-V and turned into a *compute kernel*
 *        (a GLSL source transform: gl_in[] / user input arrays become reads from a buffer,
 *        EmitVertex/EndPrimitive become appends to an output vertex buffer; see geometry.c)
 *      - the vertex stage is duplicated into a "capture variant" (the transform_feedback.c
 *        technique: the original GLSL main() is wrapped and its outputs stored per
 *        (vertex, instance) into an SSBO) so that the kernel can consume the VS outputs
 *      - the raster pass uses a generated pass-through vertex function that reads the GS
 *        output buffer and feeds the program's fragment shader
 *   2. draw time (mgl_geom_draw):
 *      - CPU assembles the GL input primitives (points/lines/triangles + adjacency) into a
 *        slot list; a rasterization-disabled render pass runs the VS capture variant; a
 *        compute pass runs the GS kernel which appends output vertices + a raster index list
 *        and writes a DrawIndexedPrimitives indirect argument buffer (no CPU readback);
 *      - gl_Layer / gl_ViewportIndex are forwarded as [[render_target_array_index]] /
 *        [[viewport_array_index]] of the pass-through vertex function.
 *
 * Choice of the vertex capture: the source-level capture variant (xfb technique) rather than
 * SPIRV-Cross's MSL "capture output to buffer" option, because Metal's fixed-function vertex
 * fetch then handles every GL vertex-attribute format / stride / divisor / base offset and the
 * captured coordinates stay in GL clip space (the rstate Metal clip-space fixup must not be
 * applied to a stage that is not the last one before rasterization).
 */
#ifndef MGL_EXT_GEOM_H
#define MGL_EXT_GEOM_H

#include "glm_context.h"
#include "mgl_ext.h"

/* the linked program's private half (MGLProgramObj), opaque here */
struct MGLProgramObj_t;
#define MGL_GEOM_MAX_VARYINGS       64
#define MGL_GEOM_MAX_CLIP           8
#define MGL_GEOM_MAX_PROGRAMS       64
#define MGL_GEOM_CTL_WORDS          32
#define MGL_GEOM_HDR_WORDS          32
#define MGL_GEOM_ARG_OFFSET         8     /* word offset of the DrawIndexedPrimitives arguments */
#define MGL_GEOM_BUDGET_BYTES       (256u << 20)     /* per-buffer allocation cap */
#define MGL_GEOM_TESS_MAX_WORDS     128   /* (src word, dst word) pairs of a tess record import */

/* GLSL (set 0) bindings of the emulation SSBOs; above the per-stage SSBO limit so an
 * application's own layout(std430, binding=N) declarations cannot collide */
enum {
    MGL_GEOM_SSBO_DUB      = 18,    /* u32: the program's default-uniform block (see
                                     * geometry.c: the capture variant and the kernel
                                     * re-declare plain uniforms as this block so the
                                     * buffer built for the raster path binds as-is) */
    MGL_GEOM_SSBO_PRIM     = 20,    /* u32: k slots per input primitive (relative to vstart) */
    MGL_GEOM_SSBO_VSOUT    = 21,    /* u32: captured VS output records */
    MGL_GEOM_SSBO_HDR      = 22,    /* u32: kernel header (MGL_GEOM_HDR_WORDS) */
    MGL_GEOM_SSBO_OUTV     = 23,    /* u32: GS output vertex records */
    MGL_GEOM_SSBO_IDX      = 24,    /* u32: raster index list (stream 0) */
    MGL_GEOM_SSBO_CTL      = 25,    /* u32: counters + indirect draw arguments */
    MGL_GEOM_SSBO_CNT      = 27,    /* u32: per-thread emitted vertex count (capture order) */
    MGL_GEOM_SSBO_DRAW     = 26,    /* pass-through vertex function: GS output records */
    MGL_GEOM_SSBO_LSCR     = 28     /* u32: per-thread output index scratch + (at word
                                     * hdr[4]) the per-thread entry counts and scanned
                                     * bases; a scan + copy pass compacts the segments
                                     * into the raster list, because writing them there
                                     * straight from an atomicAdd reservation would leave
                                     * the rasterization order a thread race */
};

/* Metal buffer indices of the same buffers (pinned above the range SPIRV-Cross auto-assigns
 * to the program's own UBO/SSBO resources); Apple GPUs expose 31 buffer slots (0..30) */
enum {
    MGL_GEOM_MSL_PRIM      = 24,
    MGL_GEOM_MSL_VSOUT     = 25,
    MGL_GEOM_MSL_HDR       = 26,
    MGL_GEOM_MSL_OUTV      = 27,
    MGL_GEOM_MSL_IDX       = 28,
    MGL_GEOM_MSL_CTL       = 29,
    MGL_GEOM_MSL_CNT       = 22,
    MGL_GEOM_MSL_DRAW      = 23,
    MGL_GEOM_MSL_LSCR      = 30,

    /* Metal argument slots a stage's PROGRAM resources (UBOs incl. instance arrays, the
     * default-uniform block, SSBOs) may span in the compute kernels: the kernel's own
     * buffers are pinned at the fixed slots from MGL_GEOM_MSL_CNT (22) up, so a stage
     * whose resources exceed this fails the link with a clear message instead of
     * overlapping the emulation's own buffers */
    MGL_GEOM_KERNEL_USER_SLOTS = 22
};

/* SPIRV-Cross's buffer-size constants argument (`constant uint* spvBufferSizeConstants
 * [[buffer(25)]]`), how it implements .length() on SSBO runtime arrays: element [i] is
 * the byte size of the bound range of the SSBO at Metal buffer index i (SPIRV-Cross
 * default, spirv_cross_msl.hpp buffer_size_buffer_index). The geometry kernel variants
 * move it inside the user slot range (MGL_GEOM_KERNEL_SIZE_BUFFER) because the kernel's
 * own buffers are pinned from MGL_GEOM_MSL_CNT (22) up. */
#define MGL_SSBO_SIZE_BUFFER_INDEX 25
#define MGL_GEOM_KERNEL_SIZE_BUFFER 21

/* Vertex-stage shadow-sampling emulation (mgl_fix_shadow_vs, program.c): the GL
 * texture-compare function of the unit feeding Metal texture slot N is passed to
 * the vertex function as function constant MGL_GEOM_FC_SHADOW_OP_BASE+N.  Values:
 * 0 LEQUAL (the GL default), 1 GEQUAL, 2 LESS, 3 GREATER, 4 EQUAL, 5 NOTEQUAL,
 * 6 ALWAYS, 7 NEVER (see mgl_cmp_do in program.c).  The rstate fixups own
 * 32000..32009. */
#define MGL_GEOM_FC_SHADOW_OP_BASE 32010

/* read()-based gather taps must honour the sampler's address mode themselves
 * (integer-texture and shadow reads in the emulated VERTEX gather path); the
 * packed wrap mode (ws | wt<<4, encodings in mgl_wco: 0 REPEAT / 1 MIRRORED_
 * REPEAT / 2 CLAMP_TO_EDGE / ...; default 0 = REPEAT) rides in as function
 * constant MGL_GEOM_FC_WRAP_BASE+N per gather slot. */
#define MGL_GEOM_FC_WRAP_BASE 32200

/* geom r60: per-slot CLAMP_TO_BORDER gather tap colour (float4, 32300+4*slot+c;
 * Metal samplers only offer three preset border colours, GL allows any float4). */
#define MGL_GEOM_FC_BORDER_BASE 32300
#define MGL_GEOM_SHADOW_OP_LEQUAL   0
#define MGL_GEOM_SHADOW_OP_GEQUAL   1
#define MGL_GEOM_SHADOW_OP_LESS     2
#define MGL_GEOM_SHADOW_OP_GREATER  3
#define MGL_GEOM_SHADOW_OP_EQUAL    4
#define MGL_GEOM_SHADOW_OP_NOTEQUAL 5
#define MGL_GEOM_SHADOW_OP_ALWAYS   6
#define MGL_GEOM_SHADOW_OP_NEVER    7

/* the GL_TEXTURE_COMPARE_FUNC of a texture unit's effective sampler state, as the
 * function-constant encoding above (a bound sampler object overrides the texture's
 * own state, like texparamSamplerForUnit) */
static inline int mgl_geom_shadow_op_of(GLenum compare_func)
{
    switch (compare_func)
    {
        case GL_GEQUAL:   return MGL_GEOM_SHADOW_OP_GEQUAL;
        case GL_LESS:     return MGL_GEOM_SHADOW_OP_LESS;
        case GL_GREATER:  return MGL_GEOM_SHADOW_OP_GREATER;
        case GL_EQUAL:    return MGL_GEOM_SHADOW_OP_EQUAL;
        case GL_NOTEQUAL: return MGL_GEOM_SHADOW_OP_NOTEQUAL;
        case GL_ALWAYS:   return MGL_GEOM_SHADOW_OP_ALWAYS;
        case GL_NEVER:    return MGL_GEOM_SHADOW_OP_NEVER;
        default:          return MGL_GEOM_SHADOW_OP_LEQUAL;
    }
}

/* one interface varying of the geometry stage (canonical word layout shared by the
 * VS capture variant, the GS kernel and the pass-through vertex function) */
typedef struct MGLGeomVar {
    char    name[128];      /* declared name, as written in the source */
    char    member[64];     /* interface block member ("a" of "v[i].a"); empty for a plain varying */
    char    block[64];      /* interface block NAME ("" for a plain varying or anonymous block):
                             * the API names block members "<block>.<member>" */
    GLuint  location;
    GLenum  base;           /* GL_FLOAT / GL_INT / GL_UNSIGNED_INT / GL_DOUBLE / GL_BOOL */
    GLuint  rows;           /* vector components (1..4) */
    GLuint  cols;           /* matrix columns (1 when not a matrix) */
    GLuint  elems;          /* array elements (1 when not an array) */
    GLuint  words;          /* 32-bit words per vertex */
    GLuint  offset;         /* word offset inside the record */
    bool    flat, noperspective, centroid, sample;
    GLuint  stream;         /* outputs: stream the varying is declared for */
    char    expr[128];      /* writer side: expression of the producing stage output */
    int     xfb_buffer;     /* shader-declared capture buffer (xfb_buffer=), -1 when not captured */
    GLuint  xfb_offset;     /* shader-declared capture byte offset (xfb_offset=) */
} MGLGeomVar;

typedef struct { char name[64]; int msl; } MGLGeomTexSlot;

/* GL atomic-counter (atomic_uint) declaration of a stage, for the MSL argument
 * injection below — program.c collects them from the stage's SPIR-V; geometry.c
 * from the geometry stage's module for the emulation kernel */
#define MGL_MAX_ATOMIC_VARS 32
typedef struct MGLAtomicCounterVar
{
    const char *name;      /* points at name_buf once collected in program.c */
    char name_buf[256];
    unsigned offset;       /* SpvDecorationOffset, byte offset into the counter buffer */
} MGLAtomicCounterVar;

typedef struct MGLGeomProgram {
    Program  *program;      /* owner (link-time program; NULL when the entry is free) */
    GLuint    name;
    int       next_free;    /* free-list link while the entry is dead (-1 = none) */
    bool      linked;       /* the emulation is built and usable */
    bool      deferred;     /* separable program: built from the pipeline at first draw */
    bool      from_tess;   /* a tessellation stage feeds the geometry kernel (TES-output capture) */
    bool      fs_uses_prim; /* the linked fragment stage reads gl_PrimitiveID: the record
                             * then carries the primitive id and the FS attribute is
                             * redirected onto the pass-through VS's varying */
    void    *gs_exec;      /* the geometry stage executable Shader (identity across pipelines) */
    char     *fail;

    /* sampler / image ARRAYS of the geometry stage, from the kernel's SPIR-V: an
     * array<texture2d, N> Metal argument covers N consecutive argument slots, fed by the
     * N texture units base..base+N-1 (MGL's sampler-uniform convention, program.c). */
    struct { char name[64]; int n; } tex_arrays[8];
    int n_tex_arrays;

    /* the generated kernels number their texture (sampler & image) arguments themselves
     * (SPIRV-Cross pins from geom_build_pins), which is NOT the stage resource list's
     * msl_binding — for the geometry stage that list still carries the GL image-unit
     * number (program.c has no Metal GS compile to refine it), and declaration order
     * need not match layout(binding) order (KHR-GL46.shader_image_size advanced-nonMS-gs
     * binds its result imageBuffer at unit 7 but declares it first).  The bind walks look
     * the slot up here by resource name and fall back to msl_binding. */
    MGLGeomTexSlot tex_slots_gs[64];   int n_tex_slots_gs;   /* the geometry kernel */
    MGLGeomTexSlot tex_slots_vs[64];   int n_tex_slots_vs;   /* the vertex capture variant */
    /* the geometry kernel's image-atomic sidecar argument slots: mgl_msl_fix_image_atomics
     * rebases its `<name>_atomic [[buffer(N)]]` arguments into the span above the user
     * pins; geomBindStageBuffers binds the per-unit sidecar through this map */
    MGLGeomTexSlot atom_slots_gs[32];  int n_atom_slots_gs;

    /* atomic counters of the geometry stage (see geometry.c): the kernel's MSL gets the
     * atomic_uint arguments mgl_msl_inject_atomic_counters appends, ONE per GLSL
     * declaration, at the flat ordinals the renderer's buffer map gives the
     * atomic-counter group (the uniform-block + uniform-constant + storage-buffer totals).
     * geomBindStageBuffers binds them itself — the map's own atomic-counter entries expand
     * a declaration (atomic_uint acs[N]) into one entry per ELEMENT, which does not match
     * the one-pointer-per-declaration arguments and can run past the user slot range into
     * the emulation's pinned buffers. */
    MGLAtomicCounterVar ac_vars[MGL_MAX_ATOMIC_VARS];
    GLuint ac_binding[MGL_MAX_ATOMIC_VARS];  /* GL atomic-counter buffer binding point */
    int    ac_msl[MGL_MAX_ATOMIC_VARS];      /* Metal buffer index of the kernel argument */
    int    n_ac_vars;
    int    ac_base;                          /* the buffer map's ordinal of the group (the
                                                  renderer skips the map's own entries there,
                                                  which expand arrays one slot per element) */
    bool   array_len;                        /* the kernel uses SSBO .length() (slot 21 is
                                                  then reserved for the size constants) */

    /* atomic counters of the VERTEX stage: the capture variant is the only vertex
     * function of a geometry program (the normal vertex MSL is not), so ITS MSL gets the
     * injected atomic_uint arguments (geom_build) and geomCapturePass binds the GL atomic
     * counter buffers at the recorded indices */
    MGLAtomicCounterVar vs_ac_vars[MGL_MAX_ATOMIC_VARS];
    GLuint vs_ac_binding[MGL_MAX_ATOMIC_VARS];
    int    vs_ac_msl[MGL_MAX_ATOMIC_VARS];
    int    n_vs_ac_vars;
    int    vs_ac_map_base;                   /* the vertex buffer map's ordinal of the group */

    GLenum    in_type;      /* GL_POINTS / GL_LINES / GL_LINES_ADJACENCY / GL_TRIANGLES / GL_TRIANGLES_ADJACENCY */
    GLenum    out_type;     /* GL_POINTS / GL_LINE_STRIP / GL_TRIANGLE_STRIP */
    GLuint    in_verts;     /* vertices presented to gl_in[] */
    GLuint    max_vertices; /* layout(max_vertices = N) */
    GLuint    invocations;  /* layout(invocations = N) */
    GLuint    streams;      /* 1 + highest stream index written */

    MGLGeomVar in_v[MGL_GEOM_MAX_VARYINGS];   int n_in;
    MGLGeomVar out_v[MGL_GEOM_MAX_VARYINGS];  int n_out;
    GLuint    in_words, out_words;

    /* input record builtins */
    bool    in_has_pos, in_has_psize, in_has_clip, in_has_cull;
    GLuint  in_pos_off, in_psize_off, in_clip_off, in_cull_off, in_clip_size;
    /* output record builtins */
    bool    out_has_pos, out_has_psize, out_has_layer, out_has_vp, out_has_prim, out_has_clip, out_has_cull;
    GLuint  out_pos_off, out_psize_off, out_layer_off, out_vp_off, out_prim_off, out_clip_off, out_cull_off;
    GLuint  out_clip_size;
    GLuint  layer_loc, vp_loc;      /* varying locations the pass-through VS uses */
    GLuint  prim_loc;               /* ditto for gl_PrimitiveID forwarded to the FS */
    bool    vid_absolute, iid_absolute;

    char   *kernel_msl, *kernel_entry, *final_entry;
    char   *capture_msl, *capture_entry;
    char   *draw_msl, *draw_entry;
    void   *capture_function, *capture_library;     /* retained MTLFunction / MTLLibrary */
    void   *draw_function, *draw_library;           /* pass-through raster vertex function */
    /* ---- cull-distance emulation (tess domain) --------------------------------
     * A program whose last pre-rasterization stage writes gl_CullDistance culls whole
     * primitives, which no Metal raster stage can express: the draw then runs through
     * the geometry-emulation path with a SYNTHETIC pass-through "geometry" stage whose
     * kernel evaluates the cull rule over each input primitive and emits nothing for a
     * culled one (geometry.c: mgl_geom_cull_* below). */
    bool     synthetic;         /* this gp is the cull pass-through, not a user GS */
    int      cull_family;       /* the in_type the kernel was generated for (rebuild on change) */
    uint32_t cull_planes;       /* bitmask of gl_CullDistance entries that participate */
    uint32_t clip_planes;       /* bitmask of gl_ClipDistance entries the stage writes:
                                 * unwritten entries are undefined and must not clip the
                                 * primitive (the raster pass-through neutralizes them) */
    bool     writes_cull;       /* the stage stores gl_CullDistance at all */
    bool     provoking_first_emit;  /* emit order puts the GL provoking vertex last */
    void    *kernel_function, *kernel_library;   /* the synthetic kernel (no gs Shader
                                                  * exists to cache it in) */
    /* capture-pass pipeline cache: the capture PSO depends only on (capture function,
     * vertex layout), but geomDraw rebuilt it on EVERY draw (newRenderPipelineState ~
     * 50-100 us even on the driver's internal cache — 4-6% of a gl41-geometry frame).
     * Keyed on the vertex-descriptor layout hash + the capture function pointer. */
    void   *capture_pso;
    uint64_t capture_pso_key;
} MGLGeomProgram;

/* per-draw input of the routing helper: either the assembled lists (CPU assembly, the
 * fallback) or the sources the prep kernel assembles them from on the GPU */
/* transform feedback of a tessellation stage without a geometry shader: the same capture
 * is the recorded vertex stream (see mgl_geom_tess_xfb_route).  The varying table is
 * MGLXfbProgram's, resolved against the TES's outputs with src_word refreshed from the
 * published member table on every pack. */
typedef struct MGLGeomTessXfbSource {
    void    *ring;          /* the tess ring (id<MTLBuffer>) */
    size_t   records;       /* byte offset of the first capture record */
    size_t   counts;        /* byte offset of the GPU-written counts (counts[0] = vertices) */
    uint32_t stride;        /* words per capture record */
    uint32_t capacity;      /* record capacity: the dispatch bound */
    uint32_t per_prim;      /* captured vertices per primitive (primitive counter) */
} MGLGeomTessXfbSource;

typedef struct MGLGeomDraw {
    MGLGeomProgram *gp;
    Program        *program;

    GLenum   mode;              /* GL draw mode */
    bool     indexed;
    GLenum   type;              /* index type when indexed */
    size_t   offset;            /* element buffer offset */
    GLint    first;
    GLsizei  count;
    GLsizei  instances;
    uint32_t known_prims;       /* exact input prims per instance (0: unknown -> size worst-case) */
    uint32_t known_instances;   /* exact instance count (0: unknown -> use `instances') */
    uint32_t known_vcount;      /* exact vertex/index count (0: unknown -> use the bound) */
    GLint    basevertex;
    GLuint   baseinstance;

    /* assembled by the C side (CPU assembly: the fallback path) */
    const uint32_t *indices;    /* capture index list (indexed draws; restarts removed) */
    GLsizei index_count;
    const uint32_t *prims;      /* gp->in_verts * prims_count slots */
    uint64_t prims_count;       /* input primitives per instance */
    uint32_t vstart;            /* first captured VS record */
    uint32_t vcount;            /* captured VS records per instance */

    /* GPU assembly: the draw command comes from this buffer (the user's indirect buffer,
     * or a staging buffer the CPU fills for direct draws) */
    void    *cmd_buffer;        /* MTLBuffer holding the draw command, or NULL for CPU assembly */
    size_t   cmd_offset;
    int      cmd_layout;        /* 0 direct staging, 1 arrays-indirect, 2 elements-indirect */
    uint32_t restart;           /* restart index (or 0xffffffffu) */
    bool     provoking_first;
    uint32_t count_bound;       /* upper bound of the vertex count, for sizing only */

    /* affine fast path: a direct (non-indexed) draw whose primitive windows follow a fixed
     * stride needs no assembly at all - the geometry kernel derives each window's vertex ids
     * arithmetically (hdr[19] + prim * hdr[20] + k) and the capture pass draws the vertex
     * range directly.  0 = assemble normally; 1 = strip-like window (prim p covers
     * [first+p, first+p+in_verts-1]); gp->in_verts = list-like (prim p covers
     * [first+p*in_verts, first+(p+1)*in_verts-1]).  One kernel launch block less (the whole
     * prep stage) per draw. */
    uint32_t implicit_stride;

    /* tess producer: instead of a vertex capture pass, the tess domain's TES-output capture
     * feeds the geometry kernel.  tess_recs_off / tess_idx_off / tess_cnt_off are byte
     * offsets into tess_ring (an MTLBuffer the tess domain owns and keeps alive); tess_npairs
     * (src word, dst word) pairs repack a tess record into this program's input record.
     * tess_rec_cap / tess_idx_cap are the ring block's record and index-list capacities (the
     * sizing bounds: the actual counts are GPU-written). */
    void    *tess_ring;
    size_t   tess_recs_off, tess_idx_off, tess_cnt_off;
    uint32_t tess_stride;                       /* words per tess record */
    uint32_t tess_prim;                         /* its output primitive type */
    uint32_t tess_rec_cap, tess_idx_cap;        /* record / index-list capacities */
    uint32_t tess_emission;                     /* MGLTessCapture.emission_order: records
                                                 * are one per emitted vertex in emission
                                                 * order — primitive slot t IS record t,
                                                 * and tess_idx_off's list is the
                                                 * tessellator's coordinate-index list,
                                                 * not a record map */
    uint32_t tess_pairs[MGL_GEOM_TESS_MAX_WORDS];
    uint32_t tess_npairs;
} MGLGeomDraw;

struct MGLExtState_geom {
    /* grown on demand: a long-lived context can link far more geometry programs than a
     * fixed table holds (the CTS keeps hundreds alive at once) */
    MGLGeomProgram  *progs;
    int              n_progs, progs_cap;

    /* name -> slot map (open addressing, linear probing, power-of-two size) and a free
     * list of slots whose program died.  The table used to be walked linearly on every
     * link, and the reuse scan called the *creating* getProgram() on dead names — which
     * resurrected a zombie Program per dead slot and made both the walk and the table
     * grow forever (the texture_swizzle smoke decays quadratically: it links a VS+FS
     * program per sub-case).  mgl_geom_post_link now also skips programs with no
     * geometry stage, so a non-geometry program costs nothing at all. */
    GLuint          *map_names;     /* 0 = empty bucket */
    int             *map_slots;     /* slot index + 1, 0 with name 0 */
    int              map_cap, map_used;
    int              free_head;     /* progs[] index, -1 = none */

    /* image-atomic emulation sidecar buffers (SPIRV-Cross's `<name>_atomic` device
     * pointers): one zero-initialized MTLBuffer per image unit (void* are retained
     * id<MTLBuffer> / id<MTLTexture>), reallocated whenever the unit's bound view or
     * unit format changes.  Only atomic operations see this storage — an imageStore
     * before an imageAtomic on the same texel is an emulation gap (the texture and the
     * sidecar do not alias; see MGLRenderer+Geometry.m geomImageAtomicBuffer).
     * dirty: the sidecar was handed to a dispatch, so it may hold atomic results —
     * geomResolveImageAtomics copies it back into the texture at barrier time. */
    struct { void *buf, *view; GLuint fmt; size_t bytes; bool dirty, needs_seed;
            GLuint elems; uint64_t sig; } img_atom[TEXTURE_UNITS];

    /* the pass-through vertex function that reads the geometry kernel's output.  The PSO builder consults this so the pass-through wins even when a bound program pipeline's composite is rebuilt from the stage programs inside processGLState (mgl_program_scope_enter copies the stage programs' MSL by value, discarding a substitution on the composite). */
    void *vertex_override;
    MGLGeomProgram *vertex_override_gp;   /* its owner, for the rstate specialization */

    /* shared 1x1 zero textures per image-unit format: an image unit bound to a texture
     * with no image reads zeros and drops stores (GL 4.6 §8.26) — geomImageEmptyTexture
     * hands these out instead of realizing the texture (void* are retained
     * id<MTLTexture>) */
    struct { void *tex; GLuint fmt; } img_empty[8];
};

struct MGLExtMetal_geom {
    /* run one draw through the geometry emulation (capture pass + kernel + raster pass) */
    bool (*draw)(GLMContext ctx, MGLGeomDraw *d);
    /* the element buffer of the current VAO can be bound as a device buffer (GPU assembly) */
    bool (*can_assemble_gpu)(GLMContext ctx);
    /* exact geometry-stage query counting: the emulation keeps per-stream counters on the
     * GPU and folds them into persistent accumulator words after every draw.
     * query_begin marks the words that restart at the next draw (a counting period begins);
     * query_read flushes the pending GPU work and reports the accumulated value, 0 when no
     * geometry-stage draw ran since the begin.  Word w<4 is PRIMITIVES_GENERATED stream w,
     * w+4 is GEOMETRY_SHADER_PRIMITIVES_EMITTED stream w (queries.c owns the mapping). */
    void (*query_begin)(GLMContext ctx, unsigned words);            /* bitmask of words */
    void (*query_read)(GLMContext ctx, unsigned word, uint64_t *out);
};

#ifdef __cplusplus
extern "C" {
#endif

/* state (geometry.c) */
void mgl_ext_init_geom(GLMContext ctx);
/* the GL program object died: free its geometry slot eagerly (O(1); without it the
 * slot table only reclaimed lazily, via a full scan) */
void mgl_ext_delete_program_geom(GLMContext ctx, Program *p);
bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data);

/* geometry stage compilation hook called by program.c's compile_stage() */
bool mgl_geom_compile_stage(GLMContext ctx, Program *program, void *obj, int stage,
                            const uint32_t *words, size_t nwords, const char *entry, char **log);
/* program.c: the rect→2D SPIR-V patch (GL image2DRect / sampler2DRect have no MSL type) */
size_t mgl_program_spirv_rect_to_2d(const uint32_t *w, size_t n, uint32_t **out_words, size_t *out_n);

/* geometry.c: repair interface-block member offsets after glslang's implicit
 * array growth (see mgl_spirv_fix_block_offsets); returns NULL when unchanged */
uint32_t *mgl_spirv_fix_block_offsets(const uint32_t *in, size_t n, size_t *out_n);

/* arrays-of-matrices stage inputs (e.g. `in mat2x3 d[2]`): lowers the variable to
 * N element-typed inputs + a private shadow so SPIRV-Cross can emit MSL; returns
 * NULL when the module has nothing to lower (or cannot be lowered) */
uint32_t *mgl_spirv_lower_aom_input(const uint32_t *in, size_t n, size_t *out_n);
/* MSL pass (program.c): drop the `volatile` address-space qualifier SPIRV-Cross emits
 * for GLSL coherent/volatile SSBO blocks (not parseable MSL, not implementable) */
char *mgl_geom_strip_msl_volatile(const char *msl);
/* MSL pass (program.c): declare GL atomic-counter (atomic_uint) arguments SPIRV-Cross
 * omits — see geometry.c for the buffer-index scheme (MGLAtomicCounterVar above) */
int  mgl_geom_stage_ac_arg_base(Program *program, int stage);   /* geometry.c: the Metal
    buffer index program.c's injected atomic-counter arguments of `stage` start at, -1 if
    none — a recompiled variant of the stage (the XFB capture variant) shares the normal
    MSL's argument order and takes the arguments at the same indices */
char *mgl_msl_inject_atomic_counters(const char *msl, const char *entry,
                                     const MGLAtomicCounterVar *vars, int nvars,
                                     int max_metal_index, int forced_base, int *indices_out);
/* Metal buffer index of the plain-compute path's buffer-size slot: the compute
 * stage is compiled with SPIRV-Cross's buffer_size_buffer_index set to this, so
 * the program's own buffers (UBOs / DUB / SSBOs / atomic counters — up to 14+1+8+8
 * = 31 GL-declared blocks fit) span 0..29 while the render stages keep the shared
 * slot 25 (KHR-GL46.compute_shader.resources-max uses 12+8+8+1 = 29 arguments) */
#define MGL_CS_SSBO_SIZE_BUFFER_INDEX 30
/* Apple GPUs need tex.fence() between a write and a read of the same
 * access::read_write (GL image2D) texel inside one invocation — see
 * mgl_msl_insert_image_fences() in geometry.c */
char *mgl_msl_insert_image_fences(const char *msl);
/* SPIRV-Cross's image-atomic emulation (`device atomic_<T>* <name>_atomic` side
 * buffers): linearize the coordinate of every non-2D image target (3D / 1DArray /
 * 2DArray / Cube / CubeArray; SPIRV-Cross writes an int2/int3 array subscript there,
 * which does not compile) and optionally re-base the argument slots.  The renderer
 * binds one zero-initialized sidecar MTLBuffer per image unit at the recorded index
 * (geomImageAtomicBuffer) — see geometry.c */
char *mgl_msl_fix_image_atomics(const char *msl, int max_index, int forced_base,
                                MGLGeomTexSlot *slots_out, int *nslots_out, int max_slots,
                                int *dub_out);
/* SPIRV-Cross's array-copy helpers don't accept spvUnsafeArray<T,A> struct
 * members — see mgl_msl_add_unsafearray_copies() in geometry.c */
char *mgl_msl_add_unsafearray_copies(const char *msl);
bool mgl_msl_vertex_returns_void(const char *msl, const char *entry);
/* post-link hook: replaces the vertex stage with the pass-through function */
void mgl_geom_post_link(GLMContext ctx, struct Program_t *program);
bool mgl_geom_link_failed(GLMContext ctx, struct Program_t *program, const char **reason);
bool mgl_program_check_geom_link(GLMContext ctx, struct Program_t *program);
bool mgl_geom_get_programiv(GLMContext ctx, struct Program_t *program, GLenum pname, GLint *params);

/* draw routing: true when the current program (or pipeline) has a geometry stage */
bool mgl_geom_present(GLMContext ctx);
void mgl_geom_route(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                    GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance);

/* ---- cull-distance emulation (tess domain): Metal has no whole-primitive cull, so a
 * program whose last pre-rasterization stage writes gl_CullDistance is routed through the
 * geometry-emulation path with a synthetic pass-through stage that drops culled
 * primitives (see the MGLGeomProgram.synthetic fields above) */
bool mgl_geom_cull_active(GLMContext ctx);
bool mgl_geom_cull_ready(GLMContext ctx, GLenum mode);   /* true: route through the
                                                          * synthetic kernel for this mode */
/* an indirect draw: the prep kernel reads the command from `ibuf`; false when the CPU has
 * to read it after all (no bound on the counts) */
bool mgl_geom_route_indirect(GLMContext ctx, GLenum mode, GLenum type, struct Buffer_t *ibuf, size_t offset,
                             const uint32_t *cmd /* [5] elements / [4] arrays command, NULL when GPU-only */);
/* the emulation record of the current program (NULL when absent or unusable) */
MGLGeomProgram *mgl_geom_current(GLMContext ctx);

/* helpers shared with the renderer category (the prep kernel's mode codes) */
int  mgl_geom_mode_code(GLenum mode);
uint64_t mgl_geom_worst_prims(GLenum mode, uint64_t n);

/* transform feedback of the geometry output (called from the xfb domain) */
void mgl_geom_xfb_link(GLMContext ctx, struct Program_t *program);
void mgl_geom_xfb_capture(GLMContext ctx, MGLGeomProgram *gp, const uint32_t *records, size_t words,
                          const uint32_t *ctl, const uint32_t *cnt, size_t cnt_words);

/* transform feedback of a tessellation stage WITHOUT a geometry shader: the TES is the last
 * pre-rasterization stage, so GL records its outputs.  mgl_geom_tess_xfb_link resolves the
 * varyings against the TES's SPIR-V (called from the xfb domain's link hook); the draw-time
 * route runs the tessellated draw with the output capture on and packs the captured records
 * into the transform feedback buffers (one thread per record, all on stream 0). */
void mgl_geom_tess_xfb_link(GLMContext ctx, struct Program_t *program);
/* a separable TCS-only program with transform feedback varyings: the names resolve against
 * the control stage's outputs (arrays); no capture is built — GL_PATCHES needs a TES, so
 * the program can never feed a draw by itself, but the link must succeed (GL 4.6 13.3.2) */
void mgl_geom_tcs_xfb_link(GLMContext ctx, struct Program_t *program);
bool mgl_geom_tess_xfb_route(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                             GLint first, GLsizei count, GLsizei instances,
                             GLint basevertex, GLuint baseinstance);

#ifdef __cplusplus
}
#endif

/* GL 8.17 sampling completeness (geometry.c; used by bindTextures) */
bool mgl_texture_incomplete_for_sampling(const Texture *tex);


#endif
