/* mgl_ext.c — fan-out of per-domain init and glGet hooks. Do not add logic here. */
#include "glm_context.h"
#include "mgl_ext.h"

void mgl_ext_init_all(GLMContext ctx)
{
    mgl_ext_init_fb(ctx);
    mgl_ext_init_queries(ctx);
    mgl_ext_init_debug(ctx);
    mgl_ext_init_introspect(ctx);
    mgl_ext_init_program(ctx);
    mgl_ext_init_xfb(ctx);
    mgl_ext_init_vattrib(ctx);
    mgl_ext_init_rstate(ctx);
    mgl_ext_init_texparam(ctx);
    mgl_ext_init_texdata(ctx);
    mgl_ext_init_texext(ctx);
    mgl_ext_init_buffers(ctx);
}

bool mgl_ext_get(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    return mgl_ext_get_fb(ctx, pname, type, data)
        || mgl_ext_get_queries(ctx, pname, type, data)
        || mgl_ext_get_debug(ctx, pname, type, data)
        || mgl_ext_get_introspect(ctx, pname, type, data)
        || mgl_ext_get_program(ctx, pname, type, data)
        || mgl_ext_get_xfb(ctx, pname, type, data)
        || mgl_ext_get_vattrib(ctx, pname, type, data)
        || mgl_ext_get_rstate(ctx, pname, type, data)
        || mgl_ext_get_texparam(ctx, pname, type, data)
        || mgl_ext_get_texdata(ctx, pname, type, data)
        || mgl_ext_get_texext(ctx, pname, type, data)
        || mgl_ext_get_buffers(ctx, pname, type, data);
}
