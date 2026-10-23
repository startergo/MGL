/* mgl_ext_tess.h — private state and Metal hooks of the 'tess' domain. Owned by agent 'tess'.
 *
 * GL 4.6 tessellation (TCS/TES + GL_PATCHES) on Metal.
 *
 * Metal mechanism (Apple silicon has no hardware tessellation — drawPatches is silently
 * dropped — so the tessellator is emulated, MoltenVK-style for the stages that can stay
 * on the GPU):
 *   - vertex stage: compiled by SPIRV-Cross as a COMPUTE kernel
 *     (SPVC_COMPILER_OPTION_MSL_VERTEX_FOR_TESSELLATION + CAPTURE_OUTPUT_TO_BUFFER); vertex
 *     attributes are fed with a compute-pipeline stageInputDescriptor (step functions
 *     MTLStepFunctionThreadPositionInGridX/GridY). The kernel writes one output record per
 *     (vertex, instance) to a scratch buffer.
 *   - tess control stage: compute kernel with MULTI_PATCH_WORKGROUP + CAPTURE_OUTPUT_TO_BUFFER
 *     writing control points, per-patch outputs and MTL*Half tessellation factors.
 *   - tess evaluation stage: SPIRV-Cross's post-tessellation vertex function is rewritten
 *     (mgl_tess_transform_tes_msl) into a plain vertex function that reads (tess coord, patch
 *     id) from a buffer indexed by [[vertex_id]] — so it plugs into an ordinary Metal render
 *     pipeline as the vertex function, with the fragment shader linked as usual.
 *   - the primitive generator (tessellation coordinates + index lists) runs on the CPU
 *     (tessellate.c logic in tessellation.c) from the (read back) tessellation factors.
 *   - no TCS: default levels (glPatchParameterfv) are used, control points pass through from
 *     the vertex kernel output — no GPU sync at all in that case.
 */
#ifndef MGL_EXT_TESS_H
#define MGL_EXT_TESS_H

#include "glm_context.h"
#include "mgl_ext.h"

/* ------------------------------------------------------------------ buffer indices
 * Compute (vertex kernel): uniforms 0..k, attribute buffers per the vattrib plan, then
 *   MGL_TESS_VS_OUT_BUF / MGL_TESS_VS_PARAMS at the top.
 * Compute (TCS kernel): uniforms 0..k, then the five SPIRV-Cross generated buffers.
 * Render (TES = vertex function): TES uniforms 0..n, tessellation buffers at 26..29.
 */
#define MGL_TESS_VS_OUT_BUF     30      /* SPVC MSL_SHADER_OUTPUT_BUFFER_INDEX (max index) */
#define MGL_TESS_VS_PARAMS      29      /* base vertex / base instance / counts */
#define MGL_TESS_TCS_IN         24      /* SPVC MSL_SHADER_INPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_OUT        25      /* SPVC MSL_SHADER_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_PATCH_OUT  26      /* SPVC MSL_SHADER_PATCH_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_FACTORS    27      /* SPVC MSL_SHADER_TESS_FACTOR_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_PARAMS     28      /* SPVC MSL_INDIRECT_PARAMS_BUFFER_INDEX */
#define MGL_TESS_TES_PARAMS     26      /* render stage: cp-per-patch, patch count */
#define MGL_TESS_TES_CAP_COUNT  30      /* capture kernel: GPU-written vertex count */
#define MGL_TESS_TES_CAP_RECS   24      /* capture kernel: packed output records (24 is a
                                         * TCS-only reservation, free in the TES kernel;
                                         * Metal compute allows buffer indices 0..30) */
/* hardware path vertex descriptor buffers (above any TES uniform buffers) */
#define MGL_TESS_VD_CP           8      /* control point records (PerPatchControlPoint) */
#define MGL_TESS_VD_PATCH        9      /* per-patch data (PerPatch) */
#define MGL_TESS_VD_FACTORS     10      /* half tessellation factors (PerPatch, stride 12) */
#define MGL_TESS_TES_COORDS     27      /* render stage: (u,v,w,patch id) per vertex */
#define MGL_TESS_TES_CP         28      /* render stage: control point records */
#define MGL_TESS_TES_PATCH      29      /* render stage: per-patch data blob */

#define MGL_TESS_MAX_PATCH_VERTICES  32 /* GL_MAX_PATCH_VERTICES */
#define MGL_TESS_MAX_LEVEL           64 /* GL_MAX_TESS_GEN_LEVEL */

enum {
    MGL_TESS_DOMAIN_TRIANGLES = 0,
    MGL_TESS_DOMAIN_QUADS,
    MGL_TESS_DOMAIN_ISOLINES
};

enum {
    MGL_TESS_SPACING_EQUAL = 0,
    MGL_TESS_SPACING_FRACTIONAL_EVEN,
    MGL_TESS_SPACING_FRACTIONAL_ODD
};

/* one per-patch output member of the TCS / patch input member of the TES (MSL struct layout) */
#define MGL_TESS_MAX_PATCH_MEMBERS 64

typedef struct MGLTessPatchMember_t {
    char    type[64];       /* MSL type ("float", "float4", "float4x4", "spvUnsafeArray<float3, 4>") */
    char    name[128];
    size_t  offset;         /* byte offset inside the MSL struct */
    size_t  size;           /* byte size inside the MSL struct */
    int     attr;           /* MSL [[attribute(N)]] of stage-in interface members, -1 if none */
    int     is_tess_level;  /* TES patch-in: 1 = gl_TessLevel/Outer, 2 = gl_TessLevelInner */
    uint32_t array_idx;     /* flattened array element (SPIRV-Cross "name_<i>" members of an
                             * arrayed varying "name"): 0 for plain members */
    uint32_t key;           /* canonical interface key: GL varying location, a fixed slot for
                             * builtins, or a name hash for members no stage reflection
                             * resolves (blocks) — identical across stages (see
                             * tess_annotate_keys) */
} MGLTessPatchMember;

/* per-program tessellation information, filled at link time */
/* ---- TES-output capture: the producer side of tess -> geometry / transform feedback.
 *
 * mgl_tess_set_capture(ctx, program, true) makes every GL_PATCHES draw of that
 * program write the TES outputs to a GPU buffer, one record per tessellated vertex:
 *
 *   word 0..3   gl_Position (float4, unmodified — no rasterizer y/depth fixup)
 *   then, per user output varying in declaration order, packed with no padding:
 *               words = elems * cols * rows * base_words   (base_words = 2 for
 *               double, else 1) — the same rule geometry.c uses for its captured
 *               VS-output records, so the geometry kernel can consume the records
 *   stride      tp->cap.stride words per record
 *
 * The consumer matches varyings by NAME from capture.members (geometry.c matches
 * its inputs by name the same way the tess blob map does). The draw is routed
 * through the emulated tessellation path so the vertex stream is GPU-visible,
 * and the results are published in the context's MGLTessCapture (below) — valid
 * until the next GL_PATCHES draw. The vertex count is GPU-written (at
 * count_offset) for indirect consumers; the tessellated primitive indices are
 * uint32 at index_offset. */
#define MGL_TESS_CAPTURE_MAX 16
typedef struct MGLTessCaptureMember {
    char     name[64];       /* record field name, as the MSL kernel writes it */
    char     api[64];        /* the API-facing name ("Block.member"), when it differs */
    uint32_t words;       /* packed word count of this member */
    uint32_t offset;      /* word offset within the record */
} MGLTessCaptureMember;

typedef struct MGLTessCapture {
    void    *buffer;         /* CPU address of the ring block's storage (ring.contents):
                              * stable for the lifetime of buffer_object */
    void    *buffer_object;  /* retained id<MTLBuffer>, keeps `buffer` mapped */
    size_t   records_offset; /* byte offset of the first record */
    size_t   index_offset;   /* byte offset of the uint32 vertex index list */
    size_t   count_offset;   /* byte offset of the uint32 counts: [0] records (GPU-written),
                              * [1] index entries (GPU-written) */
    uint32_t stride;         /* record stride in 32-bit words */
    uint32_t records;        /* record capacity of the ring block (sizing bound) */
    uint32_t indices;        /* index list capacity in uint32 words (sizing bound) */
    uint32_t prim;           /* GL_TRIANGLES / GL_LINES / GL_POINTS of the output */
    uint32_t members;        /* number of used entries in member[], [0] is gl_Position */
    MGLTessCaptureMember member[MGL_TESS_CAPTURE_MAX];
    uint64_t serial;         /* bumped on every producing draw */
    bool     valid;
} MGLTessCapture;

void mgl_tess_set_capture(GLMContext ctx, Program *program, bool enable);
/* GLSL compat: gl_PerVertex (gl_in/gl_out) may be redeclared as a SUBSET of the default
 * members (GLSL 4.60 §7.1.2), but glslang's cross-stage link rejects the interface unless
 * both sides declare the same member set — which the CTS tessellation shaders exercise
 * across vs/tcs/te/gs.  The rewrite expands any gl_PerVertex redeclaration to the full
 * built-in set (added outputs are never written, added inputs never read).
 * Returns a malloc'd rewritten source, or NULL when no rewrite is needed. */
char *mgl_tess_glsl_compat(GLenum type, const char *src);
const MGLTessCapture *mgl_tess_get_capture(GLMContext ctx);   /* last capture */

typedef struct MGLTessProgram_t {
    struct MGLTessProgram_t *next;
    Program *program;
    GLuint  program_name;

    int     has_tcs;
    int     domain;             /* MGL_TESS_DOMAIN_* */
    int     spacing;
    int     vertex_order_ccw;
    int     point_mode;

    int     tcs_out_vertices;   /* GL_TESS_CONTROL_OUTPUT_VERTICES */
    int     tcs_levels_float;   /* TCS kernel also writes a float level copy (buffer 30) */

    /* hardware TES post-tessellation function: Metal validates that the
     * [[patch(domain, N)]] control point count equals drawPatches' N, and
     * gl_PatchVerticesIn reads the same N — so without a TCS the function is
     * recompiled when GL_PATCH_VERTICES changes (cached per N here) */
    void   *tes_hw_function;    /* bridged id<MTLFunction> or NULL */
    void   *tes_hw_library;
    int     tes_hw_n;           /* N baked into tes_hw_function */
    int     tessellation_active;/* TES (or TCS) present and usable */

    /* control point record: TCS output struct == TES input struct (location-matched) */
    size_t  cp_record_size;     /* bytes per control point (TES 'main0_in') */
    MGLTessPatchMember tes_cp[MGL_TESS_MAX_PATCH_MEMBERS];
    int     num_tes_cp;
    MGLTessPatchMember cp_record[MGL_TESS_MAX_PATCH_MEMBERS];  /* full record layout (VS/TCS output) */
    int     num_cp_record;
    size_t  cp_record_full_size;   /* stride of one control point record */

    /* per-patch data: written by the TCS kernel (or defaults) and read by the TES */
    MGLTessPatchMember tcs_patch_out[MGL_TESS_MAX_PATCH_MEMBERS];
    int     num_tcs_patch_out;
    size_t  tcs_patch_out_size;
    MGLTessPatchMember tes_patch_in[MGL_TESS_MAX_PATCH_MEMBERS];
    int     num_tes_patch_in;
    size_t  tes_patch_in_size;

    /* GPU patch-blob map (mgl_tess_blob kernel, built once at link time by
     * mgl_tess_build_blob_map): for each TES patch-in member, where it comes
     * from — a byte range of the TCS patch output record, or the tessellation
     * levels of the half factor record. Mirrors the CPU tess_build_patch_blob. */
#define MGL_TESS_BLOB_COPY  0u
#define MGL_TESS_BLOB_OUTER 1u
#define MGL_TESS_BLOB_INNER 2u
#define MGL_TESS_BLOB_MAX   MGL_TESS_MAX_PATCH_MEMBERS
    uint32_t blob_kind[MGL_TESS_BLOB_MAX];
    uint32_t blob_dst[MGL_TESS_BLOB_MAX];
    uint32_t blob_src[MGL_TESS_BLOB_MAX];
    uint32_t blob_size[MGL_TESS_BLOB_MAX];
    int      blob_map_n;

    /* TES-output capture (see the MGLTessCapture contract above) */
    bool     capture;
    uint32_t cap_stride;                 /* record stride in words */
    MGLTessCaptureMember cap_members[MGL_TESS_CAPTURE_MAX];
    int      cap_members_n;
    void    *capture_function;           /* compiled kernel (owned by the renderer) */
    void    *capture_library;
    void    *capture_pso;                /* id<MTLComputePipelineState>, bridged */
    bool     capture_fc_fixup;           /* kernel MSL carries the rstate fixup (specialize) */

    /* hardware path: [[patch(domain, N)]] control point count baked into the TES MSL */
    int     hw_cp_count;

    /* plain-vertex variant of the TES (isolines / point_mode emulation), compiled on
     * first use; the string is owned by this struct */
    char   *plain_msl;
    bool    plain_msl_dumped;
    void   *plain_function;        /* id<MTLFunction>, bridged */
    void   *tcs_function;          /* canonical-record / triangle-domain TCS kernel */
    void   *tcs_library;
    void   *plain_library;

    /* canonical-record vertex kernel: the VS output struct was rewritten to the
     * canonical control point record (see "control point record" above); NULL when
     * the SPIRV-Cross layout already matches and the stage function is used as-is */
    void   *vs_function;
    void   *vs_library;

    /* ---- separable pipeline pairing (built at draw time) ----
     * A pipeline can pair this TES with a tc-only program (the TCS lives in the
     * pipeline's TESS_CONTROL stage while this program supplies the TES).  The TES
     * program's link saw no TCS, so this record says has_tcs=0 — such a draw runs
     * through a DERIVED record instead: a copy of this one with has_tcs=1, the
     * pipeline TCS's output-vertex count and the PAIR's canonical interface layouts
     * (the union of the tc program's structs and this program's TES structs), plus
     * its own compiled kernel variants.  mgl_tess_active_program() returns it while
     * that pipeline draws; it is never linked into the program hash and is rebuilt
     * when either side relinks (or is deleted). */
    struct MGLTessProgram_t *pair;     /* derived record of the active pipeline pair */
    Program *pair_program;             /* the tc-only program it was built from */
    const void *pair_key;              /* that program's TCS MSL pointer at build time */
    Program *pair_vs;                  /* (pair only) program its vs_function was built from */
} MGLTessProgram;

#define MGL_TESS_PROGRAM_HASH 257
#define MGL_TESS_VS_KERNEL_CACHE 32   /* separable pipeline vertex-kernel variants */

struct MGLExtState_tess {
    MGLTessProgram *programs[MGL_TESS_PROGRAM_HASH];

    /* published TES-output capture of the last GL_PATCHES draw (valid until the
     * next one) — see the MGLTessCapture contract above */
    MGLTessCapture capture;

    /* draw-time CPU tessellation scratch (grown on demand) */
    float  *coord_buf;      /* 4 floats per tessellated vertex: u, v, w, patch id */
    size_t  coord_cap;      /* in floats */
    uint32_t *index_buf;
    size_t  index_cap;      /* in indices */
    unsigned char *patch_data;  /* per-patch blob (TES patch-in layout) */
    size_t  patch_cap;      /* in bytes */

    /* GPU tessellated output of the last emulated-path draw (isolines / point_mode):
     * coordinate records, index list and indirect draw arguments inside one shared
     * buffer. Exposed for the geometry / transform-feedback stages to consume. */
    void   *gpu_scratch;         /* id<MTLBuffer> */
    size_t  gpu_coords_offset;   /* float4 (u, v, w, patch id) per tessellated vertex */
    size_t  gpu_indices_offset;  /* uint32 indices */
    size_t  gpu_args_offset;     /* MTLDrawPrimitivesIndirectArguments / indexed */

    /* per-draw results handed to the Metal side */
    GLsizei tess_vertex_count;
    GLsizei tess_index_count;
    GLsizei patch_count;    /* patches in the current draw (all instances) */
    GLuint  mtl_flags;      /* (unused) */

    /* draw-time vertex-kernel variants for SEPARABLE pipeline vertex programs: a
     * vs-only program linked alone keeps the plain vertex function (rstate fixup
     * baked in), and the tess draw path needs the compute-kernel rewrite of it.
     * Compiled on first use, released when the program's tess record goes away.
     * The kernel MSL is kept too: a pipeline pair (MGLTessProgram.pair) re-compiles
     * canonical-record variants of it against the pair's interface layouts. */
    struct {
        GLuint name;           /* program name the kernel belongs to */
        void  *library;        /* id<MTLLibrary>  (bridged-retain) */
        void  *function;       /* id<MTLFunction> (bridged-retain) */
        char  *msl;            /* kernel MSL (owned by the cache) */
        char   entry[128];     /* its entry point name */
    } vs_kernel[MGL_TESS_VS_KERNEL_CACHE];
    unsigned vs_kernel_n;
};

/* Metal hooks of the tess domain */
struct MGLExtMetal_tess {
    /* draw GL_PATCHES with an active tessellation program (returns false on internal error) */
    bool (*draw_patches)(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                         GLint first, GLsizei count, GLsizei instances,
                         GLint basevertex, GLint baseinstance);
};

/* tessellation.c (GL side) */
void mgl_ext_init_tess(GLMContext ctx);
bool mgl_ext_get_tess(GLMContext ctx, GLenum pname, GLenum type, void *data);
void mgl_ext_delete_program_tess(GLMContext ctx, Program *program);
MGLTessProgram *mgl_tess_program(GLMContext ctx, Program *program);
MGLTessProgram *mgl_tess_active_program(GLMContext ctx);
bool mgl_tess_plain_render(GLMContext ctx);
bool mgl_tess_cache_vs_kernel(GLMContext ctx, GLuint name, const char *msl, const char *entry);
/* GL_TESS_CONTROL_OUTPUT_VERTICES of a linked program: the tess link record when it has
 * one (a program with a TES), otherwise the TCS's own declaration — a separable tc-only
 * program links without a tess record (the TES comes from the pipeline at draw time) */
GLint mgl_tess_program_out_vertices(GLMContext ctx, Program *program);

/* called from program.c during link: MSL variants for the tessellation stages.
 * returns a malloc'd replacement for msl (caller frees the original), or NULL to keep it */
char *mgl_tess_rewrite_stage_msl(GLMContext ctx, int stage, bool tess_pipeline, const char *msl);

/* SPIR-V preparation: isolines are emulated on the quad domain (rewrites the execution mode) */
uint32_t *mgl_tess_prepare_spirv(const uint32_t *words, size_t n, size_t *out_n);

/* program.c / queries.c interaction */
void mgl_tess_note_draw(GLMContext ctx, GLuint patches, GLuint tes_invocations, GLuint tessellated_primitives);

/* draw_buffers.c routes GL_PATCHES here; forwards to the Metal hook */
bool mgl_tess_draw_patches(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                           GLint first, GLsizei count, GLsizei instances,
                           GLint basevertex, GLint baseinstance);

/* used by MGLRenderer+Tessellation.m at draw time */
bool mgl_tess_generate(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches);
void mgl_tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                               const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                               const float *factors, unsigned patches);
void mgl_tess_build_blob_map(GLMContext ctx, MGLTessProgram *tp);
GLsizei mgl_tess_patch_vertices_in(GLMContext ctx, MGLTessProgram *tp);

/* re-substitute the [[patch(domain, N)]] control point count in the stored
 * hardware TES MSL (heap string, free() it) */
char *mgl_tess_rewrite_tes_hw(const char *msl, int n);

/* hardware TES post-tessellation function matching the current draw's control
 * point count (cached per N in MGLTessProgram; returns a bridged id<MTLFunction>) */
void *mgl_tess_hw_tes_function(GLMContext ctx, MGLTessProgram *tp);
size_t mgl_tess_factor_stride(GLMContext ctx, MGLTessProgram *tp);   /* 8 tri / 12 quad */
char *mgl_tess_plain_msl(GLMContext ctx, MGLTessProgram *tp);   /* caller frees */
char *mgl_tess_capture_msl(GLMContext ctx, MGLTessProgram *tp); /* caller frees */
bool mgl_tess_cp_record_size(const char *msl, char *patchin_name, size_t patchin_len,
                             char *cp_name, size_t cp_len, size_t *record_size);

/* MGL_TESS_CPU (reference-path override), evaluated once per process: the renderer's
 * pipeline selection picks the plain TES instead of the post-tessellation variant */
bool mgl_tess_cpu_reference(void);

/* the cached separable vertex-kernel MSL of a vs-only program (NULL when absent);
 * a pipeline pair re-compiles canonical-record variants from it */
const char *mgl_tess_vs_kernel_msl(GLMContext ctx, GLuint name, const char **entry);



/* CPU-work diagnostics: incremented by the CPU-side paths above (mgl_tess_note_*);
 * the test-visible getters live in tests/headless/mgl_tess_diag.h */
void mgl_tess_note_scan_wait(void);
void mgl_tess_note_cpu_index_read(void);
void mgl_tess_note_restart_check(void);

/* query bookkeeping without CPU tessellation: the tessellated output counts from
 * the levels alone (used by the GPU paths when the levels are CPU-known) */
void mgl_tess_note_counts(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches);
bool mgl_program_check_tess_link(GLMContext ctx, struct Program_t *program);

#endif /* MGL_EXT_TESS_H */
