/* mgl_ext_rstate.h — private state and Metal hooks of the 'rstate' domain. Owned by agent 'rstate'. */
#ifndef MGL_EXT_RSTATE_H
#define MGL_EXT_RSTATE_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_rstate; allocate in mgl_ext_init_rstate(). */
struct MGLExtState_rstate {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_rstate; filled by the renderer category. */
struct MGLExtMetal_rstate {
    void *reserved;
};

#endif
