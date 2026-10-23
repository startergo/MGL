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

typedef struct MGLGeomProgram {
    Program  *program;      /* owner (link-time program; NULL when the entry is free) */
    GLuint    name;
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
    uint32_t tess_pairs[MGL_GEOM_TESS_MAX_WORDS];
    uint32_t tess_npairs;
} MGLGeomDraw;

struct MGLExtState_geom {
    /* grown on demand: a long-lived context can link far more geometry programs than a
     * fixed table holds (the CTS keeps hundreds alive at once) */
    MGLGeomProgram  *progs;
    int              n_progs, progs_cap;

    /* the pass-through vertex function that reads the geometry kernel's output.  The PSO builder consults this so the pass-through wins even when a bound program pipeline's composite is rebuilt from the stage programs inside processGLState (mgl_program_scope_enter copies the stage programs' MSL by value, discarding a substitution on the composite). */
    void *vertex_override;
    MGLGeomProgram *vertex_override_gp;   /* its owner, for the rstate specialization */
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
bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data);

/* geometry stage compilation hook called by program.c's compile_stage() */
bool mgl_geom_compile_stage(GLMContext ctx, Program *program, void *obj, int stage,
                            const uint32_t *words, size_t nwords, const char *entry, char **log);
/* program.c: the rect→2D SPIR-V patch (GL image2DRect / sampler2DRect have no MSL type) */
size_t mgl_program_spirv_rect_to_2d(const uint32_t *w, size_t n, uint32_t **out_words, size_t *out_n);
/* MSL pass (program.c): drop the `volatile` address-space qualifier SPIRV-Cross emits
 * for GLSL coherent/volatile SSBO blocks (not parseable MSL, not implementable) */
char *mgl_geom_strip_msl_volatile(const char *msl);
/* MSL pass (program.c): declare GL atomic-counter (atomic_uint) arguments SPIRV-Cross
 * omits — see geometry.c for the buffer-index scheme */
#define MGL_MAX_ATOMIC_VARS 32
typedef struct MGLAtomicCounterVar
{
    const char *name;      /* points at name_buf once collected in program.c */
    char name_buf[256];
    unsigned offset;       /* SpvDecorationOffset, byte offset into the counter buffer */
} MGLAtomicCounterVar;
char *mgl_msl_inject_atomic_counters(const char *msl, const char *entry,
                                     const MGLAtomicCounterVar *vars, int nvars,
                                     int *indices_out);
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
/* an indirect draw: the prep kernel reads the command from `ibuf`; false when the CPU has
 * to read it after all (no bound on the counts) */
bool mgl_geom_route_indirect(GLMContext ctx, GLenum mode, GLenum type, struct Buffer_t *ibuf, size_t offset);
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

#endif
