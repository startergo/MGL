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
    MGL_GEOM_SSBO_DRAW     = 26     /* pass-through vertex function: GS output records */
};

/* Metal buffer indices of the same buffers (pinned above the range SPIRV-Cross auto-assigns
 * to the program's own UBO/SSBO resources); 30 is used by the xfb capture variant */
enum {
    MGL_GEOM_MSL_PRIM      = 24,
    MGL_GEOM_MSL_VSOUT     = 25,
    MGL_GEOM_MSL_HDR       = 26,
    MGL_GEOM_MSL_OUTV      = 27,
    MGL_GEOM_MSL_IDX       = 28,
    MGL_GEOM_MSL_CTL       = 29,
    MGL_GEOM_MSL_CNT       = 22,
    MGL_GEOM_MSL_DRAW      = 23,

    /* Metal argument slots a stage's PROGRAM resources (UBOs incl. instance arrays, the
     * default-uniform block, SSBOs) may span in the compute kernels: the kernel's own
     * buffers are pinned at the fixed slots from MGL_GEOM_MSL_CNT (22) up, so a stage
     * whose resources exceed this fails the link with a clear message instead of
     * overlapping the emulation's own buffers */
    MGL_GEOM_KERNEL_USER_SLOTS = 22
};

/* one interface varying of the geometry stage (canonical word layout shared by the
 * VS capture variant, the GS kernel and the pass-through vertex function) */
typedef struct MGLGeomVar {
    char    name[128];      /* declared name, as written in the source */
    char    member[64];     /* interface block member ("a" of "v[i].a"); empty for a plain varying */
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

typedef struct MGLGeomProgram {
    Program  *program;      /* owner (link-time program; NULL when the entry is free) */
    GLuint    name;
    bool      linked;       /* the emulation is built and usable */
    bool      deferred;     /* separable program: built from the pipeline at first draw */
    bool      from_tess;   /* a tessellation stage feeds the geometry kernel (TES-output capture) */
    void    *gs_exec;      /* the geometry stage executable Shader (identity across pipelines) */
    char     *fail;

    /* sampler / image ARRAYS of the geometry stage, from the kernel's SPIR-V: an
     * array<texture2d, N> Metal argument covers N consecutive argument slots, fed by the
     * N texture units base..base+N-1 (MGL's sampler-uniform convention, program.c). */
    struct { char name[64]; int n; } tex_arrays[8];
    int n_tex_arrays;

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
bool mgl_geom_tess_xfb_route(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                             GLint first, GLsizei count, GLsizei instances,
                             GLint basevertex, GLuint baseinstance);

#ifdef __cplusplus
}
#endif

#endif
