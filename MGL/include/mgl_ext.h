/*
 * mgl_ext.h — per-domain extension slots (see agents/README.md, "Ownership").
 *
 * Each domain <d> owns:
 *   MGL/src/mgl_ext_<d>.h            struct MGLExtState_<d> / struct MGLExtMetal_<d>
 *   ctx->state.ext_<d>               its private GL state (allocated in mgl_ext_init_<d>)
 *   ctx->mtl_funcs.ext_<d>           its Metal hook table (filled by the
 *                                    -bindExt<D>FuncsToGLMContext: category method)
 *   mgl_ext_get_<d>()                extra glGet* pnames; return true if handled
 */
#ifndef MGL_EXT_H
#define MGL_EXT_H

#include <stdbool.h>

#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

struct MGLExtState_fb;
struct MGLExtMetal_fb;
struct MGLExtState_queries;
struct MGLExtMetal_queries;
struct MGLExtState_debug;
struct MGLExtMetal_debug;
struct MGLExtState_introspect;
struct MGLExtMetal_introspect;
struct MGLExtState_program;
struct MGLExtMetal_program;
struct MGLExtState_xfb;
struct MGLExtMetal_xfb;
struct MGLExtState_vattrib;
struct MGLExtMetal_vattrib;
struct MGLExtState_rstate;
struct MGLExtMetal_rstate;
struct MGLExtState_texparam;
struct MGLExtMetal_texparam;
struct MGLExtState_texdata;
struct MGLExtMetal_texdata;
struct MGLExtState_texext;
struct MGLExtMetal_texext;
struct MGLExtState_buffers;
struct MGLExtMetal_buffers;
struct MGLExtState_geom;
struct MGLExtMetal_geom;
struct MGLExtState_tess;
struct MGLExtMetal_tess;
struct MGLExtState_gpuaudit;
struct MGLExtMetal_gpuaudit;
struct MGLExtState_compat;
struct MGLExtMetal_compat;

/* glGet value types (match get.c) */
enum { MGL_GET_BOOL = 0, MGL_GET_INT = 1, MGL_GET_FLOAT = 2, MGL_GET_DOUBLE = 3, MGL_GET_INT64 = 4 };

/* store element [index] of a glGet result in the caller's requested type */
static inline void mgl_get_store(GLuint type, void *data, int index, double v)
{
    switch (type) {
        case MGL_GET_BOOL:   ((GLboolean *)data)[index] = v != 0.0 ? GL_TRUE : GL_FALSE; break;
        case MGL_GET_INT:    ((GLint *)data)[index] = (GLint)v; break;
        case MGL_GET_FLOAT:  ((GLfloat *)data)[index] = (GLfloat)v; break;
        case MGL_GET_DOUBLE: ((GLdouble *)data)[index] = v; break;
        case MGL_GET_INT64:  ((GLint64 *)data)[index] = (GLint64)v; break;
    }
}

#ifdef __cplusplus
extern "C" {
#endif
void mgl_ext_init_fb(GLMContext ctx);
bool mgl_ext_get_fb(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_queries(GLMContext ctx);
bool mgl_ext_get_queries(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_debug(GLMContext ctx);
bool mgl_ext_get_debug(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_introspect(GLMContext ctx);
bool mgl_ext_get_introspect(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_program(GLMContext ctx);
bool mgl_ext_get_program(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_xfb(GLMContext ctx);
bool mgl_ext_get_xfb(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_vattrib(GLMContext ctx);
bool mgl_ext_get_vattrib(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_rstate(GLMContext ctx);
bool mgl_ext_get_rstate(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_texparam(GLMContext ctx);
bool mgl_ext_get_texparam(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_texdata(GLMContext ctx);
bool mgl_ext_get_texdata(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_texext(GLMContext ctx);
bool mgl_ext_get_texext(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_buffers(GLMContext ctx);
bool mgl_ext_get_buffers(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_geom(GLMContext ctx);
bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_tess(GLMContext ctx);
bool mgl_ext_get_tess(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_gpuaudit(GLMContext ctx);
bool mgl_ext_get_gpuaudit(GLMContext ctx, GLenum pname, GLuint type, void *data);
void mgl_ext_init_compat(GLMContext ctx);
bool mgl_ext_get_compat(GLMContext ctx, GLenum pname, GLuint type, void *data);


/* ---- cross-domain hook registries -------------------------------------------
 * Register from your mgl_ext_init_<d>(); call sites are owned as noted. */
struct Program_t;
typedef bool (*mgl_get_indexed_hook)(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data);
typedef void (*mgl_program_hook)(GLMContext ctx, struct Program_t *program);
/* return false to skip the draw (e.g. conditional rendering); also used to count primitives */
typedef bool (*mgl_pre_draw_hook)(GLMContext ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count);

void mgl_ext_register_get_indexed(mgl_get_indexed_hook fn);   /* called by get.c Get*i_v (owner: debug) */
void mgl_ext_register_pre_link(mgl_program_hook fn);          /* called by mglLinkProgram before linking (owner: program) */
void mgl_ext_register_post_link(mgl_program_hook fn);         /* called by mglLinkProgram after a successful link (owner: program) */
void mgl_ext_register_pre_draw(mgl_pre_draw_hook fn);         /* called by every draw entry point (owner: buffers) */

bool mgl_ext_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data);
void mgl_ext_pre_link(GLMContext ctx, struct Program_t *program);
void mgl_ext_post_link(GLMContext ctx, struct Program_t *program);
bool mgl_ext_pre_draw(GLMContext ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count);

void mgl_ext_init_all(GLMContext ctx);
bool mgl_ext_get(GLMContext ctx, GLenum pname, GLuint type, void *data);
#ifdef __cplusplus
}
#endif

#endif
