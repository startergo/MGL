/* mgl_ext_program.h — private state and Metal hooks of the 'program' domain. Owned by agent 'program'. */
#ifndef MGL_EXT_PROGRAM_H
#define MGL_EXT_PROGRAM_H

#include "glm_context.h"
#include "mgl_ext.h"

/* GL-side state, reachable as ctx->state.ext_program; allocate in mgl_ext_init_program(). */
struct MGLExtState_program {
    int reserved;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_program; filled by the renderer category. */
struct MGLExtMetal_program {
    void *reserved;
};

#endif
