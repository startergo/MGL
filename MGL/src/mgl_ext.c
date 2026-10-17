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

#define MGL_EXT_MAX_HOOKS 16
#define DEFINE_REGISTRY(name, type) \
    static type name##_hooks[MGL_EXT_MAX_HOOKS]; static int name##_count; \
    void mgl_ext_register_##name(type fn) { \
        for (int i = 0; i < name##_count; i++) if (name##_hooks[i] == fn) return; \
        if (name##_count < MGL_EXT_MAX_HOOKS) name##_hooks[name##_count++] = fn; }

DEFINE_REGISTRY(get_indexed, mgl_get_indexed_hook)
DEFINE_REGISTRY(pre_link, mgl_program_hook)
DEFINE_REGISTRY(post_link, mgl_program_hook)
DEFINE_REGISTRY(pre_draw, mgl_pre_draw_hook)

bool mgl_ext_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data)
{
    for (int i = 0; i < get_indexed_count; i++)
        if (get_indexed_hooks[i](ctx, pname, index, type, data)) return true;
    return false;
}

void mgl_ext_pre_link(GLMContext ctx, struct Program_t *program)
{
    for (int i = 0; i < pre_link_count; i++) pre_link_hooks[i](ctx, program);
}

void mgl_ext_post_link(GLMContext ctx, struct Program_t *program)
{
    for (int i = 0; i < post_link_count; i++) post_link_hooks[i](ctx, program);
}

bool mgl_ext_pre_draw(GLMContext ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count)
{
    bool draw = true;
    for (int i = 0; i < pre_draw_count; i++)
        if (!pre_draw_hooks[i](ctx, mode, vertex_count, instance_count)) draw = false;
    return draw;
}
