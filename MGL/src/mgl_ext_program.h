/* mgl_ext_program.h — private state and Metal hooks of the 'program' domain. Owned by agent 'program'.
 *
 * Object layout: Shader / Program / ProgramPipeline objects are allocated by this domain
 * (shaders.c / program.c) as the larger MGLShaderObj / MGLProgramObj / MGLPipelineObj
 * structs below whose first member is the public struct, so the extra state needs no
 * edits to glm_context.h. Program.shader_slots[] holds the *linked executable* stages
 * (internal Shader objects owned by the program); the attachment list is separate.
 */
#ifndef MGL_EXT_PROGRAM_H
#define MGL_EXT_PROGRAM_H

#include "glm_context.h"
#include "mgl_ext.h"

/* private binary format returned by GL_PROGRAM_BINARY_FORMATS */
#define MGL_PROGRAM_BINARY_FORMAT   0x4D474C42u     /* 'MGLB' */
#define MGL_PROGRAM_BINARY_VERSION  2u

typedef struct MGLShaderObj {
    Shader      base;
    GLboolean   compile_status;
    GLboolean   in_table;           /* name still visible (not yet freed) */
    GLboolean   internal;           /* executable stage owned by a program */
    GLboolean   spirv_binary;       /* SPIR_V_BINARY */
    GLboolean   specialized;
    uint32_t   *spirv;              /* module from ShaderBinary (patched by SpecializeShader) */
    size_t      spirv_words;
    char       *spirv_entry;        /* entry point chosen by SpecializeShader */
} MGLShaderObj;

typedef struct MGLSamplerUse {
    GLuint  unit;                   /* binding == texture image unit */
    GLuint  type_key;               /* dim | arrayed | depth | ms | sampled base type */
    GLuint  stage;
    char   *name;
} MGLSamplerUse;

typedef struct MGLProgramObj {
    Program     base;
    GLboolean   link_status;
    GLboolean   validate_status;
    GLboolean   separable;
    GLboolean   binary_retrievable_hint;
    GLboolean   in_table;
    char       *info_log;

    Shader    **attached;           /* attachment list (refcounted) */
    int         num_attached, cap_attached;

    /* ---- linked executable (swapped in on a successful link) ---- */
    GLuint      linked_mask;        /* SHADER_MASK_BIT(stage) */
    char       *stage_entry[_MAX_SHADER_TYPES];   /* SPIR-V entry point name per stage */
    MGLSamplerUse *samplers;
    int         num_samplers;
    GLint       active_attributes, active_attribute_max_length;
    GLint       active_uniforms, active_uniform_max_length;
    GLint       active_uniform_blocks, active_uniform_block_max_name_length;
    GLint       active_atomic_counter_buffers;
} MGLProgramObj;

typedef struct MGLPipelineObj {
    ProgramPipeline base;
    GLboolean   created;            /* Gen'd names become objects on first bind */
    GLboolean   validate_status;
    Program    *active_program;
    char       *info_log;
    Program     composite;          /* stage programs merged for the renderer */
} MGLPipelineObj;

#define MGL_SHADER_OBJ(_s_)   ((MGLShaderObj *)(_s_))
#define MGL_PROGRAM_OBJ(_p_)  ((MGLProgramObj *)(_p_))
#define MGL_PIPELINE_OBJ(_p_) ((MGLPipelineObj *)(_p_))

/* GetProgramiv extension point: other domains (xfb: TRANSFORM_FEEDBACK_*, introspect:
 * ACTIVE_*) may answer pnames for a program; return true if handled. */
typedef bool (*mgl_programiv_hook)(GLMContext ctx, Program *program, GLenum pname, GLint *params);
void mgl_program_register_programiv_hook(mgl_programiv_hook fn);

/* GL-side state, reachable as ctx->state.ext_program; allocated in mgl_ext_init_program(). */
struct MGLExtState_program {
    int         live_glslang_shaders;   /* compiled glslang shader objects alive */
    GLboolean   compiler_released;      /* glslang process state finalized by ReleaseShaderCompiler */
    GLboolean   release_pending;        /* finalize once the last compiled glslang shader is gone */
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_program; filled by MGLRenderer+Program.m. */
struct MGLExtMetal_program {
    /* compile MSL; on success returns retained (CFBridgingRetain) library + function */
    bool (*compile_msl)(GLMContext ctx, const char *msl, const char *entry,
                        void **library, void **function, char **error_log);
    void (*release_obj)(GLMContext ctx, void *obj);
};

#ifdef __cplusplus
extern "C" {
#endif
bool     mgl_program_is_linked(Program *program);
/* program receiving glUniform*: current program, else the bound pipeline's active program */
Program *mgl_program_uniform_target(GLMContext ctx);
/* program used for rendering when no program is current: the bound pipeline's stages */
Program *mgl_program_pipeline_composite(GLMContext ctx);
const GLubyte *mgl_program_spirv_extension(GLMContext ctx, GLuint index);
void     mgl_program_invalidate_pipelines(GLMContext ctx);

/* internal helpers shared by shaders.c / program.c / program_ext.c */
Shader  *mgl_shader_new_internal(GLMContext ctx, GLuint glm_type);
void     mgl_shader_unref(GLMContext ctx, Shader *shader);
GLenum   mgl_shader_gl_type(GLuint glm_type);
void     mgl_shader_release_compiler_resources(GLMContext ctx);
void     mgl_set_info_log(char **dst, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void     mgl_append_info_log(char **dst, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void     mgl_copy_info_log(const char *log, GLsizei bufSize, GLsizei *length, GLchar *out);
Program *findProgram(GLMContext ctx, GLuint program);
bool     mgl_program_link_spirv_stages(GLMContext ctx, Program *program, const uint32_t *const *words,
                                       const size_t *word_counts, const char *const *entries,
                                       GLuint stage_mask);
bool     mgl_program_check_xfb_link(GLMContext ctx, Program *program);
bool     mgl_program_validate_samplers(GLMContext ctx, Program *const *programs, int count, char **log);

/* renderer scope: while resolving GL state, substitute the pipeline composite for
 * ctx->state.program when no program is current; restored on scope exit. */
typedef struct MGLProgramScope { GLMContext ctx; bool active; } MGLProgramScope;
static inline MGLProgramScope mgl_program_scope_enter(GLMContext ctx)
{
    MGLProgramScope s = { ctx, false };
    if (ctx && ctx->state.program == NULL && ctx->state.program_pipeline) {
        Program *p = mgl_program_pipeline_composite(ctx);
        if (p) { ctx->state.program = p; s.active = true; }
    }
    return s;
}
static inline void mgl_program_scope_exit(MGLProgramScope *s)
{
    if (s->active) s->ctx->state.program = NULL;
}
#define MGL_PIPELINE_PROGRAM_SCOPE(_ctx_) \
    __attribute__((cleanup(mgl_program_scope_exit), unused)) MGLProgramScope _mgl_program_scope = mgl_program_scope_enter(_ctx_)
#ifdef __cplusplus
}
#endif

#endif
