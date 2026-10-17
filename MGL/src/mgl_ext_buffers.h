/* mgl_ext_buffers.h — private state and Metal hooks of the 'buffers' domain. Owned by agent 'buffers'. */
#ifndef MGL_EXT_BUFFERS_H
#define MGL_EXT_BUFFERS_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_buffers; allocate in mgl_ext_init_buffers(). */
struct MGLExtState_buffers {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_buffers; filled by the renderer category. */
struct MGLExtMetal_buffers {
    void *reserved;
};

#endif
