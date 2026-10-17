/* mgl_ext_queries.h — private state and Metal hooks of the 'queries' domain. Owned by agent 'queries'. */
#ifndef MGL_EXT_QUERIES_H
#define MGL_EXT_QUERIES_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_queries; allocate in mgl_ext_init_queries(). */
struct MGLExtState_queries {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_queries; filled by the renderer category. */
struct MGLExtMetal_queries {
    void *reserved;
};

#endif
