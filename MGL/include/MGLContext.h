/*
  MGLContext.h
  mgl

  Created by Michael Larson on 12/17/21.
*/

#ifndef MGLContext_h
#define MGLContext_h

// probably a reason for this... can't remember
#ifndef __GLM_CONTEXT_
#define __GLM_CONTEXT_
typedef struct GLMContextRec_t *GLMContext;
#endif

enum {
    MGL_PIXEL_FORMAT,
    MGL_PIXEL_TYPE,
    MGL_DEPTH_FORMAT,
    MGL_DEPTH_TYPE,
    MGL_STENCIL_FORMAT,
    MGL_STENCIL_TYPE,
    MGL_CONTEXT_FLAGS
};

#ifdef __cplusplus
extern "C" {
#endif

GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);
GLMContext createGLMContextProfile(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type,
                            GLenum profile);
void mglSetNextContextProfile(GLenum profile);
GLenum mglNextContextProfile(void);

GLuint sizeForFormatType(GLenum format, GLenum type);

GLMContext MGLgetCurrentContext(void);
void MGLsetCurrentContext(GLMContext ctx);

// MGLswapBuffers can take NULL for the ctx, in this case it will use the current ctx
// (the default argument is C++ only: glfw compiles this header as C)
#ifdef __cplusplus
void MGLswapBuffers(GLMContext ctx=NULL);
#else
void MGLswapBuffers(GLMContext ctx);
#endif

// MGLget can take NULL for the ctx, in this case it will use the current ctx
void MGLget(GLMContext ctx, GLenum param, GLuint *data);

// r182: fence completion callback. Call with the context current, as for any GL
// call. fn(arg) runs exactly once when the GPU passes the fence, on an MGL-owned
// dispatch queue: never on the calling thread (not even for a fence that has
// already signalled) and never under the GL context, so fn must not call GL.
// The fence's commands are submitted, as GL_SYNC_FLUSH_COMMANDS_BIT would. The
// sync may be deleted before it signals: fn still runs once. Returns GL_FALSE
// with GL_INVALID_VALUE for an invalid sync or a NULL fn (fn is then not called).
// If the GPU work is abandoned (a device fault), fn may never run: keep a timeout.
GLboolean mglSyncNotify(GLsync sync, void (*fn)(void *arg), void *arg);

#ifdef __cplusplus
};
#endif

#endif
