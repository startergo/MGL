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
#define MGL_TESS_VS_OUT_BUF     30      /* SPVC MSL_SHADER_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_VS_PARAMS      31      /* base vertex / base instance / counts */
#define MGL_TESS_TCS_IN         24      /* SPVC MSL_SHADER_INPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_OUT        25      /* SPVC MSL_SHADER_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_PATCH_OUT  26      /* SPVC MSL_SHADER_PATCH_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_FACTORS    27      /* SPVC MSL_SHADER_TESS_FACTOR_OUTPUT_BUFFER_INDEX */
#define MGL_TESS_TCS_PARAMS     28      /* SPVC MSL_INDIRECT_PARAMS_BUFFER_INDEX */
#define MGL_TESS_TES_PARAMS     26      /* render stage: cp-per-patch, patch count */
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

    /* per-patch data: written by the TCS kernel (or defaults) and read by the TES */
    MGLTessPatchMember tcs_patch_out[MGL_TESS_MAX_PATCH_MEMBERS];
    int     num_tcs_patch_out;
    size_t  tcs_patch_out_size;
    MGLTessPatchMember tes_patch_in[MGL_TESS_MAX_PATCH_MEMBERS];
    int     num_tes_patch_in;
    size_t  tes_patch_in_size;
    int     tes_tess_level_members;   /* how many of tes_patch_in are the tess-level builtins */
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
GLsizei mgl_tess_patch_vertices_in(GLMContext ctx, MGLTessProgram *tp);

#endif /* MGL_EXT_TESS_H */
