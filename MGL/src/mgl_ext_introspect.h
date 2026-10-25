/* mgl_ext_introspect.h — private state and Metal hooks of the 'introspect' domain. Owned by agent 'introspect'.
 *
 * The introspect domain owns the GL program interface:
 *   - default-block uniform storage (glUniform* / glProgramUniform* / glGetUniform*)
 *   - program resource reflection (GL 4.3 program interface queries, legacy queries)
 *   - attribute / fragment-output location binding, uniform/storage block bindings
 *   - subroutine emulation (source rewrite + uniform selector)
 *
 * Mechanism (see program_introspection.c for details):
 *   post-link hook: the linked SPIR-V of every stage is patched (loose non-opaque uniforms are
 *   packed into one Uniform block "mgl_DefaultUniforms", vertex input / fragment output / varying
 *   locations are assigned, buffer variables are reordered so SPIRV-Cross' automatic Metal buffer
 *   indices match the renderer's buffer map order) and MSL is regenerated with program.c's
 *   parseSPIRVShaderToMetal().  Reflection tables are built from the patched SPIR-V.
 */
#ifndef MGL_EXT_INTROSPECT_H
#define MGL_EXT_INTROSPECT_H

#include <stdint.h>
#include "glm_context.h"
#include "mgl_ext.h"

#define MGLI_STAGES _MAX_SHADER_TYPES

/* scalar kinds of a default-block uniform */
enum { MGLI_K_FLOAT = 0, MGLI_K_INT, MGLI_K_UINT, MGLI_K_BOOL, MGLI_K_DOUBLE, MGLI_K_OPAQUE };

typedef struct MGLIUniform {
    char   *name;              /* GL name ("u", "arr[0]", "s.a", "Block.member") */
    GLenum  type;              /* GL type enum */
    GLint   size;              /* array size (1 if not an array) */
    GLint   block_index;       /* -1 for the default block */
    GLint   offset;            /* -1 for the default block */
    GLint   array_stride;      /* -1 for the default block, 0 for non arrays */
    GLint   matrix_stride;     /* -1 for the default block, 0 for non matrices */
    GLint   is_row_major;
    GLint   atomic_buffer_index; /* -1 unless atomic counter */
    GLint   location;          /* base location, -1 if none */
    GLbitfield stages;         /* referenced-by mask (1 << _xxx_SHADER) */
    int     hidden;            /* internal (subroutine selector): not reported */

    /* value storage (default block / opaque) */
    int     kind;              /* MGLI_K_* */
    int     cols, rows;        /* matrix columns (1 for vectors), components per column */
    int     opaque_class;      /* 0 none, 1 sampler, 2 image, 3 atomic */
    uint8_t *shadow;           /* size * cols * rows * (kind==DOUBLE ? 8 : 4) bytes, column major */
    struct {
        int      present;
        uint32_t offset;       /* byte offset of element 0 in the stage's default uniform block */
        uint32_t array_stride;
        uint32_t matrix_stride;
        uint32_t scalar_size;  /* Metal scalar size: 1 (bool), 4, 8 */
        uint32_t spv_id;       /* opaque: SPIR-V variable id (to update resource bindings) */
        int      res_type;     /* opaque: SPVC_RESOURCE_TYPE_* list holding it */
        int      elem_base;    /* opaque arrays: element index of this uniform's [0] in the variable */
    } st[MGLI_STAGES];
} MGLIUniform;

typedef struct MGLIBlock {
    char   *name;              /* "Block" or "Block[i]" */
    GLint   binding;
    GLint   data_size;
    int    *members;           /* uniform indices (UBO) / buffer variable indices (SSBO) */
    int     member_count;
    GLbitfield stages;
    uint32_t spv_id[MGLI_STAGES]; /* SPIR-V variable id per stage (0 if absent) */
    int     array_element;     /* element of a block array, -1 if not an array */
} MGLIBlock;

typedef struct MGLIBufferVar {
    char   *name;
    GLenum  type;
    GLint   size, block_index, offset, array_stride, matrix_stride, is_row_major, top_size, top_stride;
    GLbitfield stages;
} MGLIBufferVar;

typedef struct MGLIVar {          /* program inputs / outputs */
    char   *name;
    GLenum  type;
    GLint   size;
    GLint   location;             /* -1 for built-ins; vertex inputs: the GL location
                                     (one per dvec3/dvec4 element and per matrix column,
                                     ARB_vertex_attrib_64bit "one generic attribute") */
    GLint   index;                /* fragment output index (dual source), else 0 / -1 */
    GLint   component;
    GLint   is_per_patch;
    GLbitfield stages;
    GLint   slot;                 /* vertex inputs: internal Metal/SPIR-V slot base the
                                     fp64 word-pair interface fetches from (== location
                                     except for dvec3/dvec4-based inputs, which take two
                                     slots per element); -1 when not a vertex input */
} MGLIVar;

typedef struct MGLIAtomicBuffer {
    GLint   binding, data_size;
    int    *members; int member_count;
    GLbitfield stages;
} MGLIAtomicBuffer;

typedef struct MGLISubFunc { char *name; GLint index; char **compat; int ncompat; } MGLISubFunc;
typedef struct MGLISubUniform {
    char   *name;
    char   *type_name;
    GLint   location;             /* subroutine uniform location */
    GLint   size;                 /* array size (flattened) */
    int     ndims;                /* arrays of arrays: number of dimensions */
    int     dims[8];              /* per-dimension size */
    int    *compat; int ncompat;  /* compatible subroutine indices */
    char   *selector;             /* name of the hidden int uniform holding the selection */
} MGLISubUniform;
typedef struct MGLIStageSub {
    MGLISubFunc    *funcs; int nfuncs;
    MGLISubUniform *unis;  int nunis;
    int             nlocs;        /* ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS */
    GLuint         *selection;    /* per location: subroutine index */
} MGLIStageSub;

/* per-shader compile-time record of the subroutine rewrite */
typedef struct MGLIShaderRec {
    GLuint  shader;
    Shader *ptr;
    char   *src;                  /* rewritten source (owned) */
    MGLIStageSub sub;             /* funcs / unis (indices & locations assigned at compile) */
    struct MGLIShaderRec *next;
} MGLIShaderRec;

typedef struct MGLIProgram {
    GLuint   name;
    Program *prog;
    struct MGLIProgram *next;

    /* bindings requested by the application (take effect at next link) */
    struct { char *name; GLuint index; } *attrib_binds; int nattrib_binds;
    struct { char *name; GLuint color, index; } *frag_binds; int nfrag_binds;

    int      linked;
    GLbitfield stage_mask;
    int      first_stage, last_stage;

    MGLIUniform *uniforms; int nuniforms;
    struct { int uniform; int element; } *locs; int nlocs;
    MGLIBlock   *ubos; int nubos;
    MGLIBlock   *ssbos; int nssbos;
    MGLIBufferVar *bvars; int nbvars;
    MGLIVar     *inputs; int ninputs;
    MGLIVar     *outputs; int noutputs;
    MGLIVar     *vtx_outputs; int nvtx_outputs; /* outputs of the last vertex-processing stage (xfb) */
    MGLIAtomicBuffer *abufs; int nabufs;
    MGLIStageSub sub[MGLI_STAGES];

    int      attrib_conflict;   /* two vertex inputs overlap locations (link error) */
    int      uniform_loc_conflict; /* same uniform, two different explicit locations (link error) */
    int      uniform_loc_max;   /* GL_MAX_UNIFORM_LOCATIONS as reported (link check) */
    int      uniform_loc_overflow; /* an explicit location reaches past uniform_loc_max (link error) */
    int      attrib_overflow;  /* vertex inputs exceed MAX_VERTEX_ATTRIBS vectors (link error) */
    /* conformant GL location per vertex-input variable (var id -> GL location);
       the SPIR-V Location decoration is the internal slot (see MGLIVar.slot) */
    struct { uint32_t var; GLint gl; } *attrib_gl; int nattrib_gl;

    int      nxfb;  /* transform feedback varyings (from mgl_xfb_program_info, refreshed per query) */

    /* per-stage default uniform block (one Buffer per stage, see renderer buffer map) */
    Buffer  *dub[MGLI_STAGES];
    uint32_t dub_size[MGLI_STAGES];
    int      dub_res_index[MGLI_STAGES]; /* index in spirv_resources_list[stage][UNIFORM_CONSTANT], -1 */

    /* GL-level leaf kinds (bool/double) of the default-block members, recorded at patch time:
     * the GPU stores int/float, introspection still reports GL_BOOL/GL_DOUBLE */
    uint8_t *dub_leaf[MGLI_STAGES];
    int      dub_leaf_n[MGLI_STAGES];

    /* constant initializers of default-block uniforms, captured when the stage is patched */
    struct { char *name; uint32_t *vals; int nvals; int wide; } *inits[MGLI_STAGES];
    int      ninits[MGLI_STAGES];
} MGLIProgram;

/* GL-side state, reachable as ctx->state.ext_introspect; allocate in mgl_ext_init_introspect(). */
struct MGLExtState_introspect {
    MGLIProgram   *programs;
    MGLIShaderRec *shaders;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_introspect; filled by the renderer category. */
struct MGLExtMetal_introspect {
    void *reserved;
};

#ifdef __cplusplus
extern "C" {
#endif

/* ---- exported to other domains ------------------------------------------------------------ */

/* glGetProgramiv pnames owned by introspection: GL_ACTIVE_UNIFORMS, GL_ACTIVE_UNIFORM_MAX_LENGTH,
 * GL_ACTIVE_ATTRIBUTES, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, GL_ACTIVE_UNIFORM_BLOCKS,
 * GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, GL_ACTIVE_ATOMIC_COUNTER_BUFFERS,
 * Returns true if pname was handled (program.c's mglGetProgramiv should call this). */
bool mgl_introspect_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params);

/* Buffer holding the default uniform block of `program` for `stage` (resource `index` in
 * spirv_resources_list[stage][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT]); NULL if none.
 * Used by MGLRenderer mapGLBuffersToMTLBufferMap. */
Buffer *mgl_introspect_default_uniform_buffer(GLMContext ctx, Program *program, int stage, int index);

/* SPIR-V patch applied by program.c compile_stage() before MSL translation (introspect_spirv.c):
 * default uniform block, attribute / fragment output locations, buffer ordering.
 * Returns malloc'd words (caller frees) or NULL to keep the module unchanged. */
uint32_t *mgl_introspect_patch_stage(GLMContext ctx, Program *program, int stage, const uint32_t *words,
                                     size_t nwords, size_t *out_words);

/* Shader source to hand to glslang (subroutine-rewritten copy or shader->src).  Called by
 * mglCompileShader (shaders.c). */
void mgl_introspect_reset_subroutines(GLMContext ctx, Program *p);
const char *mgl_introspect_shader_source(GLMContext ctx, Shader *shader);

/* Attribute location lookup (mglGetAttribLocation in program.c forwards here). */
GLint mgl_introspect_get_attrib_location(GLMContext ctx, GLuint program, const GLchar *name);

/* Resolve which GL attribute owns Metal attribute slot `loc` and its GL type:
 * a dvec3/dvec4 vertex input is split by the fp64 lowering into word slots at
 * locations k and k+1, so slot k+1 belongs to the attribute at k (cbase 2).
 * Returns the GL type of the owning input (0 if the slot is not a program
 * input); *owner gets the attribute location, *cbase its first component. */
GLenum mgl_introspect_slot_owner(GLMContext ctx, GLuint program, GLint loc, GLint *owner, GLint *cbase);

/* ---- internal (introspect domain files) ---------------------------------------------------- */
MGLIProgram *mgli_find(GLMContext ctx, GLuint program);
MGLIProgram *mgli_get(GLMContext ctx, GLuint program);
/* program lookup with GL error semantics; returns NULL (error set) on failure */
MGLIProgram *mgli_program_checked(GLMContext ctx, GLuint program, bool need_linked);
void mgli_set_error(GLMContext ctx, const char *func, GLenum err);

/* uniform value API shared by glUniform* and glProgramUniform* (uniforms.c) */
enum { MGLI_SET_F = 0, MGLI_SET_I, MGLI_SET_UI, MGLI_SET_D };
void mgli_set_uniform(GLMContext ctx, const char *func, MGLIProgram *ip, GLint location, GLsizei count,
                      int set_kind, int cols, int rows, GLboolean transpose, const void *data);
void mgli_get_uniform(GLMContext ctx, const char *func, GLuint program, GLint location, GLsizei bufSize,
                      int out_kind, void *params);
void mgli_upload_uniform(GLMContext ctx, MGLIProgram *ip, MGLIUniform *u);
GLint mgli_uniform_location(MGLIProgram *ip, const char *name);
int  mgli_uniform_index(MGLIProgram *ip, const char *name);

/* SPIR-V helpers (introspect_spirv.c) */
struct MGLSpvModule;
bool mgli_process_link(GLMContext ctx, Program *ptr, MGLIProgram *ip);
void mgli_free_inits(MGLIProgram *ip);

/* subroutine source rewrite (introspect_subroutine.c) */
char *mgli_rewrite_subroutines(const char *src, MGLIStageSub *out);
/* strip "#extension GL_ARB_shader_subroutine" directives (glslang does not know it) */
char *mgli_strip_subroutine_extension(const char *src);
void  mgli_free_stage_sub(MGLIStageSub *s);

#ifdef __cplusplus
}
#endif

#endif
