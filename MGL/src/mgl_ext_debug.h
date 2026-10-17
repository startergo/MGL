/* mgl_ext_debug.h — private state and Metal hooks of the 'debug' domain. Owned by agent 'debug'. */
#ifndef MGL_EXT_DEBUG_H
#define MGL_EXT_DEBUG_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_debug; allocate in mgl_ext_init_debug(). */
struct MGLExtState_debug {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_debug; filled by the renderer category. */
struct MGLExtMetal_debug {
    void *reserved;
};

#endif
