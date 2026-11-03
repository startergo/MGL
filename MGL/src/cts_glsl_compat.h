/*
 * cts_glsl_compat.h — GLSL source rewriting / analysis for GL-legal constructs that need
 * help under the glslang -> SPIR-V -> MSL pipeline.  See cts_glsl_compat.c.
 */
#ifndef MGL_CTS_GLSL_COMPAT_H
#define MGL_CTS_GLSL_COMPAT_H

#include "glm_context.h"
#include <stdbool.h>

/* Returns a malloc'd rewritten copy of `src`, or NULL when it can be compiled as is.
 * The caller frees the result. */
char *mgl_cts_glsl_compat(const char *src);

/* Reported binding-point limits for the layout(binding=N) compile-time range check
 * (GLSL 4.60 4.4.5): N at or beyond the implementation limit is a compile error.
 * Zero entries disable the check for that resource class. */
typedef struct MGLCompatBindingLimits {
    unsigned ubo_bindings;         /* GL_MAX_UNIFORM_BUFFER_BINDINGS */
    unsigned ssbo_bindings;        /* GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS */
    unsigned atomic_bindings;      /* GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS */
    unsigned image_units;          /* GL_MAX_IMAGE_UNITS */
    unsigned combined_texture_units; /* GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS */
} MGLCompatBindingLimits;

/* the binding-point limits as reported by glGet* (implemented in get.c, single
 * source of truth with its GL_MAX_* cases) */
GLuint mgl_reported_binding_limit(GLMContext ctx, GLenum pname);

/* as mgl_cts_glsl_compat(), additionally rejecting layout(binding=N) declarations
 * whose N is out of range for their resource class (filled from
 * mgl_reported_binding_limit() in get.c).  NULL limits skip the range check. */
char *mgl_cts_glsl_compat_limited(const char *src, const MGLCompatBindingLimits *limits);
/* the capture-variant path: the binding-range / std430 rejections are skipped
 * (MGL-authored GLSL uses its own renumbered binding points) */
char *mgl_cts_glsl_compat_generated(const char *src);

/*
 * The GLSL-to-SPIR-V convention (glslang: "'shared' : not allowed..."; ARB_gl_spirv)
 * stores bool/bvecN members of uniform and buffer blocks as uint/uvecN: "The GLSL
 * convention is a 32-bit int where non-0 means true".  GL introspection must report the
 * GLSL type (GL_BOOL, GL_BOOL_VEC*) though, so the declared types are recovered from the
 * shader source: mgl_cts_bool_block_members() collects every block-member path
 * ("Block.member", "Block.s.x", "Block.arr[0]", ...) whose GLSL type is bool/bvecN.
 *
 * mgl_cts_bool_paths_has() answers for one such dotted path; block type name and member
 * path are matched separately (the member path may contain dots/brackets).
 */
typedef struct MGLBoolPaths MGLBoolPaths;

MGLBoolPaths *mgl_cts_bool_block_members(const char *src);
int  mgl_cts_bool_paths_has(const MGLBoolPaths *p, const char *path);  /* "Block.a.b[0]" */
void mgl_cts_bool_paths_free(MGLBoolPaths *p);

/* hoist sampler/image members of default-uniform-block structs into plain
 * uniforms (SPIRV-Cross cannot lower them to MSL); returns a malloc'd copy or
 * NULL when the source has none.  mgl_hoist_lookup_name() maps the GL dotted
 * spelling ("s.c", "s[0].c") back to the hoisted name ("s_mglhoist_c",
 * "s_mglhoist_c[0]").  mgl_hoist_gl_path() is the inverse for introspection
 * (the resource list must carry the dotted GL name), and
 * mgl_hoist_outer_dims() tells how many leading dimensions of the hoisted
 * array belong to the uniform declaration (arrays of structs enumerate per
 * element, GL 4.6 §7.3.1). */
char *mgl_hoist_struct_samplers(const char *src);
char *mgl_reject_reserved_declarators(const char *src);   /* "#error ..." source or NULL */
int mgl_hoist_lookup_name(const char *name, char *out, size_t outn);
bool mgl_hoist_gl_path(const char *hoisted, char *out, size_t outn);
bool mgl_hoist_gl_path2(const char *hoisted, const char *dims, char *out, size_t outn, int *n_outer);
/* mgl_hoist_gl_path2 plus the per-segment array dims of the hoisted path
 * (seg[0] = the uniform declaration's dims, the last = the opaque member's own
 * bottom-level array): every segment but the last enumerates one GL resource
 * per element with the subscript right behind that segment ("u.i[1].a[0]"). */
bool mgl_hoist_gl_path3(const char *hoisted, const char *dims, char *out, size_t outn,
                        char seg[8][48], int *nseg);
int  mgl_hoist_outer_dims(const char *hoisted);

#endif
