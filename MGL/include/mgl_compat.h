/*
 * mgl_compat.h — public test/application entry points of the compatibility
 * profile (compat domain).  A compatibility-profile context is a normal MGL
 * context whose profile is GL_CONTEXT_COMPATIBILITY_PROFILE_BIT: it exposes
 * the fixed-function entry points; a core context keeps rejecting them.
 *
 * Handles are void* so this header can be included without the internal
 * MGL headers (which need the external include paths).
 */
#ifndef MGL_COMPAT_H
#define MGL_COMPAT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Create a compatibility-profile context (headless Metal renderer) and make
 * it current.  Returns NULL on failure. */
void *mglCompatCreateContext(void);

/* The library's default core-profile context (created at load time). */
void *mglCoreContext(void);

/* Make a context current (either profile). */
void mglMakeContextCurrent(void *ctx);

/* 1 when the *current* context is a compatibility-profile context. */
int mglCurrentContextIsCompat(void);

#ifdef __cplusplus
};
#endif

#endif /* MGL_COMPAT_H */
