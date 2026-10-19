/*
 * assertaudit.c — fixes found by auditing the removed assert(0) sites
 * (see agents/assertaudit_report.md).
 *
 * This file owns the GL-side state of the assertaudit domain:
 *   - the default vertex array object (GL name 0)
 */
#include <stdlib.h>
#include "glm_context.h"
#include "mgl_ext_assertaudit.h"

VertexArray *newVAO(GLMContext ctx, GLuint vao);   /* vertex_arrays.c */

void mgl_ext_init_assertaudit(GLMContext ctx)
{
    if (ctx->state.ext_assertaudit)
        return;

    ctx->state.ext_assertaudit = (struct MGLExtState_assertaudit *)
        calloc(1, sizeof(struct MGLExtState_assertaudit));
    if (!ctx->state.ext_assertaudit)
        return;

    /* the default VAO is created outside the object table (name 0 is reserved
     * for it) and lives for the context's lifetime */
    VertexArray *vao = newVAO(ctx, 0);
    if (vao)
    {
        ctx->state.ext_assertaudit->default_vao = vao;

        /* GL 4.6: the initial vertex array binding is the default object (0) */
        ctx->state.vao = vao;
    }
}

VertexArray *mgl_assertaudit_default_vao(GLMContext ctx)
{
    return ctx->state.ext_assertaudit ? ctx->state.ext_assertaudit->default_vao : NULL;
}
