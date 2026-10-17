/* mgl_ext_fb.h — private state and Metal hooks of the 'fb' domain. Owned by agent 'fb'. */
#ifndef MGL_EXT_FB_H
#define MGL_EXT_FB_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_fb; allocate in mgl_ext_init_fb(). */
struct MGLExtState_fb {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_fb; filled by the renderer category. */
struct MGLExtMetal_fb {
    void *reserved;
};

#endif
