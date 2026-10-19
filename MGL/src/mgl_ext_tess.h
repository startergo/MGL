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
} MGLTessPatchMember;

/* per-program tessellation information, filled at link time */
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

    /* hardware path: [[patch(domain, N)]] control point count baked into the TES MSL */
    int     hw_cp_count;

    /* plain-vertex variant of the TES (isolines / point_mode emulation), compiled on
     * first use; the string is owned by this struct */
    char   *plain_msl;
    bool    plain_msl_dumped;
    void   *plain_function;        /* id<MTLFunction>, bridged */
    void   *tcs_function;          /* triangle-domain TCS kernel (8-byte factor records) */
    void   *tcs_library;
    void   *plain_library;
} MGLTessProgram;

#define MGL_TESS_PROGRAM_HASH 257

struct MGLExtState_tess {
    MGLTessProgram *programs[MGL_TESS_PROGRAM_HASH];

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
size_t mgl_tess_factor_stride(GLMContext ctx, MGLTessProgram *tp);   /* 8 tri / 12 quad */
char *mgl_tess_plain_msl(GLMContext ctx, MGLTessProgram *tp);   /* caller frees */
bool mgl_tess_cp_record_size(const char *msl, char *patchin_name, size_t patchin_len,
                             char *cp_name, size_t cp_len, size_t *record_size);

/* diagnostics for the GPU-path tests: CPU-side patch-blob builds / CPU
 * tessellator runs since library load (the GPU paths must not grow these) */
unsigned mgl_tess_cpu_blob_builds(void);
unsigned mgl_tess_cpu_generate_runs(void);

#endif /* MGL_EXT_TESS_H */
