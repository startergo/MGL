/* mgl_ext_vattrib.h — private state and Metal hooks of the 'vattrib' domain. Owned by agent 'vattrib'. */
#ifndef MGL_EXT_VATTRIB_H
#define MGL_EXT_VATTRIB_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_vattrib; allocate in mgl_ext_init_vattrib(). */
struct MGLExtState_vattrib {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_vattrib; filled by the renderer category. */
struct MGLExtMetal_vattrib {
    void *reserved;
};

#endif
