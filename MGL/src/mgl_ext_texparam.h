/* mgl_ext_texparam.h — private state and Metal hooks of the 'texparam' domain. Owned by agent 'texparam'. */
#ifndef MGL_EXT_TEXPARAM_H
#define MGL_EXT_TEXPARAM_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_texparam; allocate in mgl_ext_init_texparam(). */
struct MGLExtState_texparam {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texparam; filled by the renderer category. */
struct MGLExtMetal_texparam {
    void *reserved;
};

#endif
