/* mgl_ext_texext.h — private state and Metal hooks of the 'texext' domain. Owned by agent 'texext'. */
#ifndef MGL_EXT_TEXEXT_H
#define MGL_EXT_TEXEXT_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_texext; allocate in mgl_ext_init_texext(). */
struct MGLExtState_texext {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texext; filled by the renderer category. */
struct MGLExtMetal_texext {
    void *reserved;
};

#endif
