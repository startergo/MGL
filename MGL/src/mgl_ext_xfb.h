/* mgl_ext_xfb.h — private state and Metal hooks of the 'xfb' domain. Owned by agent 'xfb'. */
#ifndef MGL_EXT_XFB_H
#define MGL_EXT_XFB_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_xfb; allocate in mgl_ext_init_xfb(). */
struct MGLExtState_xfb {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_xfb; filled by the renderer category. */
struct MGLExtMetal_xfb {
    void *reserved;
};

#endif
