/* mgl_ext_introspect.h — private state and Metal hooks of the 'introspect' domain. Owned by agent 'introspect'. */
#ifndef MGL_EXT_INTROSPECT_H
#define MGL_EXT_INTROSPECT_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_introspect; allocate in mgl_ext_init_introspect(). */
struct MGLExtState_introspect {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_introspect; filled by the renderer category. */
struct MGLExtMetal_introspect {
    void *reserved;
};

#endif
