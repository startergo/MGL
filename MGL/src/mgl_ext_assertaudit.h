/*
 * mgl_ext_assertaudit.h — extension slot for the assertaudit domain.
 *
 * Holds the default vertex array object (GL name 0). GL 4.6 core starts every
 * context with VAO 0 bound and drawing / gl*VertexAttribPointer through it is
 * legal (client-side pointers are what core forbids, not the object). MGL used
 * to keep ctx->state.vao NULL for name 0, which turned those legal calls into
 * GL_INVALID_OPERATION; the object is created here and bound at context init.
 */
#ifndef MGL_EXT_ASSERTAUDIT_H
#define MGL_EXT_ASSERTAUDIT_H

#include "glm_context.h"

/* GL-side state, reachable as ctx->state.ext_assertaudit; allocated in
 * mgl_ext_init_assertaudit(). */
typedef struct MGLExtState_assertaudit {
    VertexArray *default_vao;   /* the name-0 vertex array object */
} MGLExtState_assertaudit;

#ifdef __cplusplus
extern "C" {
#endif

void mgl_ext_init_assertaudit(GLMContext ctx);

/* the default (name 0) vertex array object, or NULL if it could not be created */
VertexArray *mgl_assertaudit_default_vao(GLMContext ctx);

/* true when the default (name 0) vertex array object is bound. The
 * ARB_vertex_attrib_binding entry points (glVertexAttrib*Format, glVertexAttribBinding,
 * glBindVertexBuffer*, glVertexBindingDivisor, glVertexAttribDivisor) treat that state
 * as "no vertex array object bound" and must generate GL_INVALID_OPERATION; the classic
 * entry points (glVertexAttribPointer, glEnable/DisableVertexAttribArray, draws) work
 * through the default object (client-side pointers are what the core profile forbids). */
static inline bool mgl_assertaudit_default_vao_bound(GLMContext ctx)
{
    struct MGLExtState_assertaudit *s = ctx->state.ext_assertaudit;
    return s && ctx->state.vao == s->default_vao;
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_EXT_ASSERTAUDIT_H */
