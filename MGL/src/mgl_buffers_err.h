/* mgl_buffers_err.h — returning variants of the glm_context.h error macros.
 * The shared ERROR_RETURN / ERROR_CHECK_RETURN macros record the error but do not
 * leave the function; GL requires commands that generate an error to have no other
 * effect, so the 'buffers' domain files include this after glm_context.h. */
#ifndef MGL_BUFFERS_ERR_H
#define MGL_BUFFERS_ERR_H

#undef ERROR_RETURN
#define ERROR_RETURN(_type_) do { ctx->error_func(ctx, __FUNCTION__, _type_); return; } while (0)

#undef ERROR_CHECK_RETURN
#define ERROR_CHECK_RETURN(_expr_, _type_) do { if ((_expr_) == false) { ctx->error_func(ctx, __FUNCTION__, _type_); return; } } while (0)

#undef ERROR_RETURN_VALUE
#define ERROR_RETURN_VALUE(_type_, _val_) do { ctx->error_func(ctx, __FUNCTION__, _type_); return _val_; } while (0)

#undef ERROR_CHECK_RETURN_VALUE
#define ERROR_CHECK_RETURN_VALUE(_expr_, _type_, _val_) do { if ((_expr_) == false) { ctx->error_func(ctx, __FUNCTION__, _type_); return _val_; } } while (0)

#endif
