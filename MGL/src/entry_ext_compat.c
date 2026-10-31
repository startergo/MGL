/*
 * entry_ext_compat.c — extension entry points that must be dlsym-resolvable
 * (compat domain).  Every extension MGL advertises must export ALL of the
 * entry points gl.xml lists for it, or a GL loader that resolves by name
 * (glcts / glew / glfw) stores NULL and the process SEGFAULTs the first
 * time the test calls the function — see the glBlendBarrierKHR crash in
 * dEQP-GL45-ES3.functional.draw_buffers_indexed.overwrite_* (2026-10-31).
 *
 * Most of these are promoted-to-core extensions whose unsuffixed core
 * entry point already exists in gl_core.c -> ctx->dispatch.*: the alias
 * just forwards.  glClearColorIiEXT / glClearColorIuiEXT carry real state
 * (integer clear colour, EXT_texture_integer §3.6.2) implemented in
 * compat_state.c and consumed by the clear path in rendering.c.
 * glBlendBarrier* explains why it is a no-op in compat_advblend.c.
 *
 * Audit script: ab/ext_entry_audit.py (regenerates the list below from
 * MGL/src/get.c + external/OpenGL-Registry/xml/gl.xml).
 *
 * NOT exported (documented gaps, see agents/compat_report.md):
 *   GL_ARB_gpu_shader_int64 glUniform*{i,ui}64*ARB / glGetUniform*64vARB —
 *     64-bit int uniforms need width-64 introspection (introspect_spirv.c
 *     maps them to 4-byte MGLI_K_INT today); glcts does not load the ARB
 *     names (it only loads GL_NV_gpu_shader5's glUniform1i64NV, and
 *     GL_NV_gpu_shader5 is not advertised).
 */

#define GL_GLEXT_PROTOTYPES 1
#include "glcorearb.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"

extern GLMContext _ctx;
extern void mgl_lazy_init(void);

#define GET_CONTEXT()   (mgl_lazy_init(), _ctx)

/* glClearColorIiEXT / glClearColorIuiEXT (compat_state.c) */
extern void mglClearColorIi(GLMContext ctx, GLint r, GLint g, GLint b, GLint a);
extern void mglClearColorIui(GLMContext ctx, GLuint r, GLuint g, GLuint b, GLuint a);
/* glBlendBarrier* (compat_advblend.c) */
extern void mglBlendBarrier(GLMContext ctx);

#pragma mark - GL_KHR_blend_equation_advanced (+ NV / ES core aliases)

/*
 * GL_KHR_blend_equation_advanced: "barriers guarantee that the advanced blend
 * equation reads the results of previously drawn geometry".  MGL implements
 * the equations with framebuffer fetch inside the fragment shader, which
 * Metal executes in raster order group 0: the destination read is both
 * ordered (earlier draws in the same pass are visible) and coherent (GL_KHR_
 * blend_equation_advanced_coherent is advertised for the same reason).
 * Render passes themselves complete in submission order, so there is no GPU
 * work to do — the call is recorded (debug counter) and returns.
 */
void glBlendBarrierKHR(void)
{
    GLMContext ctx = GET_CONTEXT();
    mglBlendBarrier(ctx);
}

void glBlendBarrierNV(void)   { GLMContext ctx = GET_CONTEXT(); mglBlendBarrier(ctx); }
void glBlendBarrier(void)     { GLMContext ctx = GET_CONTEXT(); mglBlendBarrier(ctx); }

#pragma mark - GL_ARB_draw_buffers_blend (core 4.0 promoted)

void glBlendEquationiARB(GLuint buf, GLenum mode)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.blend_equationi(ctx, buf, mode);
}

void glBlendEquationSeparateiARB(GLuint buf, GLenum modeRGB, GLenum modeAlpha)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.blend_equation_separatei(ctx, buf, modeRGB, modeAlpha);
}

void glBlendFunciARB(GLuint buf, GLenum src, GLenum dst)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.blend_funci(ctx, buf, src, dst);
}

void glBlendFuncSeparateiARB(GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.blend_func_separatei(ctx, buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
}

#pragma mark - GL_ARB_occlusion_query (core 1.5/2.0 promoted)

void glGenQueriesARB(GLsizei n, GLuint *ids)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.gen_queries(ctx, n, ids);
}

void glDeleteQueriesARB(GLsizei n, const GLuint *ids)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.delete_queries(ctx, n, ids);
}

GLboolean glIsQueryARB(GLuint id)
{
    GLMContext ctx = GET_CONTEXT();
    return ctx->dispatch.is_query(ctx, id);
}

void glBeginQueryARB(GLenum target, GLuint id)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.begin_query(ctx, target, id);
}

void glEndQueryARB(GLenum target)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.end_query(ctx, target);
}

void glGetQueryivARB(GLenum target, GLenum pname, GLint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_queryiv(ctx, target, pname, params);
}

void glGetQueryObjectivARB(GLuint id, GLenum pname, GLint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_query_objectiv(ctx, id, pname, params);
}

void glGetQueryObjectuivARB(GLuint id, GLenum pname, GLuint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_query_objectuiv(ctx, id, pname, params);
}

#pragma mark - GL_ARB_texture_buffer_object (core 3.1 promoted)

void glTexBufferARB(GLenum target, GLenum internalformat, GLuint buffer)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.tex_buffer(ctx, target, internalformat, buffer);
}

#pragma mark - GL_ARB_viewport_array / NV_depth_buffer_float interaction

void glDepthRangeArraydvNV(GLuint first, GLsizei count, const GLdouble *v)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.depth_range_arrayv(ctx, first, count, v);
}

void glDepthRangeIndexeddNV(GLuint index, GLdouble n, GLdouble f)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.depth_range_indexed(ctx, index, n, f);
}

#pragma mark - GL_EXT_texture_integer

void glTexParameterIivEXT(GLenum target, GLenum pname, const GLint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.tex_parameter_iiv(ctx, target, pname, params);
}

void glTexParameterIuivEXT(GLenum target, GLenum pname, const GLuint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.tex_parameter_iuiv(ctx, target, pname, params);
}

void glGetTexParameterIivEXT(GLenum target, GLenum pname, GLint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_tex_parameter_iiv(ctx, target, pname, params);
}

void glGetTexParameterIuivEXT(GLenum target, GLenum pname, GLuint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_tex_parameter_iuiv(ctx, target, pname, params);
}

void glClearColorIiEXT(GLint red, GLint green, GLint blue, GLint alpha)
{
    GLMContext ctx = GET_CONTEXT();
    mglClearColorIi(ctx, red, green, blue, alpha);
}

void glClearColorIuiEXT(GLuint red, GLuint green, GLuint blue, GLuint alpha)
{
    GLMContext ctx = GET_CONTEXT();
    mglClearColorIui(ctx, red, green, blue, alpha);
}

#pragma mark - GL_KHR_debug (KHR suffixes are mandated for ES loaders)

void glDebugMessageControlKHR(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.debug_message_control(ctx, source, type, severity, count, ids, enabled);
}

void glDebugMessageInsertKHR(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.debug_message_insert(ctx, source, type, id, severity, length, buf);
}

void glDebugMessageCallbackKHR(GLDEBUGPROC callback, const void *userParam)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.debug_message_callback(ctx, callback, userParam);
}

GLuint glGetDebugMessageLogKHR(GLuint count, GLsizei bufSize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog)
{
    GLMContext ctx = GET_CONTEXT();
    return ctx->dispatch.get_debug_message_log(ctx, count, bufSize, sources, types, ids, severities, lengths, messageLog);
}

void glPushDebugGroupKHR(GLenum source, GLuint id, GLsizei length, const GLchar *message)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.push_debug_group(ctx, source, id, length, message);
}

void glPopDebugGroupKHR(void)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.pop_debug_group(ctx);
}

void glObjectLabelKHR(GLenum identifier, GLuint name, GLsizei length, const GLchar *label)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.object_label(ctx, identifier, name, length, label);
}

void glGetObjectLabelKHR(GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_object_label(ctx, identifier, name, bufSize, length, label);
}

void glObjectPtrLabelKHR(const void *ptr, GLsizei length, const GLchar *label)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.object_ptr_label(ctx, ptr, length, label);
}

void glGetObjectPtrLabelKHR(const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_object_ptr_label(ctx, ptr, bufSize, length, label);
}

void glGetPointervKHR(GLenum pname, void **params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.get_pointerv(ctx, pname, params);
}

#pragma mark - GL_ARB_robustness (core 4.5 promoted + compat-only getn)

GLenum glGetGraphicsResetStatusARB(void)
{
    GLMContext ctx = GET_CONTEXT();
    return ctx->dispatch.get_graphics_reset_status(ctx);
}

void glReadnPixelsARB(GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.readn_pixels(ctx, x, y, width, height, format, type, bufSize, data);
}

void glGetnTexImageARB(GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_tex_image(ctx, target, level, format, type, bufSize, pixels);
}

void glGetnCompressedTexImageARB(GLenum target, GLint lod, GLsizei bufSize, void *pixels)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_compressed_tex_image(ctx, target, lod, bufSize, pixels);
}

void glGetnUniformfvARB(GLuint program, GLint location, GLsizei bufSize, GLfloat *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_uniformfv(ctx, program, location, bufSize, params);
}

void glGetnUniformivARB(GLuint program, GLint location, GLsizei bufSize, GLint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_uniformiv(ctx, program, location, bufSize, params);
}

void glGetnUniformuivARB(GLuint program, GLint location, GLsizei bufSize, GLuint *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_uniformuiv(ctx, program, location, bufSize, params);
}

void glGetnUniformdvARB(GLuint program, GLint location, GLsizei bufSize, GLdouble *params)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_uniformdv(ctx, program, location, bufSize, params);
}

void glGetnMapdvARB(GLenum target, GLenum query, GLsizei bufSize, GLdouble *v)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_mapdv(ctx, target, query, bufSize, v);
}

void glGetnMapfvARB(GLenum target, GLenum query, GLsizei bufSize, GLfloat *v)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_mapfv(ctx, target, query, bufSize, v);
}

void glGetnMapivARB(GLenum target, GLenum query, GLsizei bufSize, GLint *v)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_mapiv(ctx, target, query, bufSize, v);
}

void glGetnPixelMapfvARB(GLenum map, GLsizei bufSize, GLfloat *values)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_pixel_mapfv(ctx, map, bufSize, values);
}

void glGetnPixelMapuivARB(GLenum map, GLsizei bufSize, GLuint *values)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_pixel_mapuiv(ctx, map, bufSize, values);
}

void glGetnPixelMapusvARB(GLenum map, GLsizei bufSize, GLushort *values)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_pixel_mapusv(ctx, map, bufSize, values);
}

void glGetnPolygonStippleARB(GLsizei bufSize, GLubyte *pattern)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_polygon_stipple(ctx, bufSize, pattern);
}

/* the ARB_imaging getn commands: ARB_robustness lists them, but MGL does not
 * implement the imaging subset (no GL_ARB_imaging in the extension list), so
 * the dispatch slots hold the same GL_INVALID_OPERATION error stubs as the
 * unsuffixed glGetnColorTable etc. exported from gl_core.c — forwarding keeps
 * the aliases loadable without inventing imaging behaviour. */
void glGetnColorTableARB(GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *table)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_color_table(ctx, target, format, type, bufSize, table);
}

void glGetnConvolutionFilterARB(GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *image)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_convolution_filter(ctx, target, format, type, bufSize, image);
}

void glGetnSeparableFilterARB(GLenum target, GLenum format, GLenum type, GLsizei rowBufSize, void *row, GLsizei columnBufSize, void *column, void *span)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_separable_filter(ctx, target, format, type, rowBufSize, row, columnBufSize, column, span);
}

void glGetnHistogramARB(GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_histogram(ctx, target, reset, format, type, bufSize, values);
}

void glGetnMinmaxARB(GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    GLMContext ctx = GET_CONTEXT();
    ctx->dispatch.getn_minmax(ctx, target, reset, format, type, bufSize, values);
}
