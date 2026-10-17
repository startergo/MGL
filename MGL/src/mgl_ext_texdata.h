/* mgl_ext_texdata.h — private state and Metal hooks of the 'texdata' domain. Owned by agent 'texdata'. */
#ifndef MGL_EXT_TEXDATA_H
#define MGL_EXT_TEXDATA_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_texdata; allocate in mgl_ext_init_texdata(). */
struct MGLExtState_texdata {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texdata; filled by the renderer category. */
struct MGLExtMetal_texdata {
    void *reserved;
};

#endif
