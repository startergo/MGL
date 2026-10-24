/*
 * cts_glsl_compat.h — GLSL source rewriting / analysis for GL-legal constructs that need
 * help under the glslang -> SPIR-V -> MSL pipeline.  See cts_glsl_compat.c.
 */
#ifndef MGL_CTS_GLSL_COMPAT_H
#define MGL_CTS_GLSL_COMPAT_H

/* Returns a malloc'd rewritten copy of `src`, or NULL when it can be compiled as is.
 * The caller frees the result. */
char *mgl_cts_glsl_compat(const char *src);
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

#endif
