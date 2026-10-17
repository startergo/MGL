/* mgl_ext_queries.h — private state and Metal hooks of the 'queries' domain. Owned by agent 'queries'.
 *
 * Query objects (glGenQueries/glCreateQueries), timer queries, pipeline statistics and
 * conditional rendering.  See queries.c for the GL side and MGLRenderer+Queries.m for Metal.
 *
 * Mechanisms
 *   SAMPLES_PASSED / ANY_SAMPLES_PASSED(_CONSERVATIVE) / FRAGMENT_SHADER_INVOCATIONS
 *       Metal visibility results (MTLVisibilityResultModeCounting), one 8-byte slot per
 *       "segment" (render pass x set of active queries), summed at resolve time.
 *   TIME_ELAPSED / TIMESTAMP
 *       command-buffer boundaries: Begin/End/QueryCounter commit the current command buffer,
 *       results come from MTLCommandBuffer GPUStartTime/GPUEndTime (host mach timebase, ns).
 *       glGetInteger64v(GL_TIMESTAMP) uses the same clock (CACurrentMediaTime).
 *   everything else is counted on the CPU from draw parameters (pre-draw hook) or from
 *   mgl_queries_note_dispatch / mgl_queries_note_xfb_primitives.
 */
#ifndef MGL_EXT_QUERIES_H
#define MGL_EXT_QUERIES_H

#include <stdint.h>
#include <stdbool.h>
#include "glm_context.h"
#include "mgl_ext.h"
#include "hash_table.h"

#define MGL_QUERY_MAX_VERTEX_STREAMS 4

/* binding points ("active query" slots) */
enum {
    MGL_QB_OCCLUSION = 0,                 /* SAMPLES_PASSED, ANY_SAMPLES_PASSED(_CONSERVATIVE) share one */
    MGL_QB_TIME_ELAPSED,
    MGL_QB_PRIMITIVES_GENERATED,          /* + stream (4) */
    MGL_QB_XFB_PRIMITIVES_WRITTEN = MGL_QB_PRIMITIVES_GENERATED + MGL_QUERY_MAX_VERTEX_STREAMS,
    MGL_QB_XFB_STREAM_OVERFLOW = MGL_QB_XFB_PRIMITIVES_WRITTEN + MGL_QUERY_MAX_VERTEX_STREAMS,
    MGL_QB_XFB_OVERFLOW = MGL_QB_XFB_STREAM_OVERFLOW + MGL_QUERY_MAX_VERTEX_STREAMS,
    MGL_QB_VERTICES_SUBMITTED,
    MGL_QB_PRIMITIVES_SUBMITTED,
    MGL_QB_VERTEX_SHADER_INVOCATIONS,
    MGL_QB_TESS_CONTROL_SHADER_PATCHES,
    MGL_QB_TESS_EVALUATION_SHADER_INVOCATIONS,
    MGL_QB_GEOMETRY_SHADER_INVOCATIONS,
    MGL_QB_GEOMETRY_SHADER_PRIMITIVES_EMITTED,
    MGL_QB_FRAGMENT_SHADER_INVOCATIONS,
    MGL_QB_COMPUTE_SHADER_INVOCATIONS,
    MGL_QB_CLIPPING_INPUT_PRIMITIVES,
    MGL_QB_CLIPPING_OUTPUT_PRIMITIVES,
    MGL_QB_COUNT
};

typedef struct MGLQuery_t {
    GLuint   name;
    GLenum   target;        /* 0 = name reserved by glGenQueries, no object yet */
    GLuint   index;         /* vertex stream for indexed targets */
    bool     active;
    bool     deleted;       /* deleted while active: freed at End */
    bool     result_ready;
    uint64_t result;
    uint64_t cpu_count;     /* CPU-side counters / overflow flags */
    void    *gpu;           /* Metal record (CFBridgingRetain'ed), owned by MGLRenderer+Queries.m */
} MGLQuery;

/* GL-side state, reachable as ctx->state.ext_queries; allocated in mgl_ext_init_queries(). */
struct MGLExtState_queries {
    HashTable  query_table;
    MGLQuery  *active[MGL_QB_COUNT];

    /* conditional rendering */
    bool       cond_active;
    GLenum     cond_mode;
    MGLQuery  *cond_query;
    bool       cond_fixed;          /* query deleted during conditional render: result frozen */
    uint64_t   cond_fixed_result;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_queries; filled by the renderer category. */
struct MGLExtMetal_queries {
    void     (*begin)(GLMContext ctx, MGLQuery *q);     /* GPU-backed targets only */
    void     (*end)(GLMContext ctx, MGLQuery *q);
    void     (*counter)(GLMContext ctx, MGLQuery *q);   /* QueryCounter(GL_TIMESTAMP) */
    bool     (*resolve)(GLMContext ctx, MGLQuery *q, bool wait); /* true + q->result when available */
    void     (*release)(GLMContext ctx, MGLQuery *q);
    int64_t  (*now_ns)(GLMContext ctx);                 /* GL_TIMESTAMP clock */
};

#ifdef __cplusplus
extern "C" {
#endif
/* compute dispatch counter (COMPUTE_SHADER_INVOCATIONS); called by mglDispatchCompute*. */
void mgl_queries_note_dispatch(GLMContext ctx, uint64_t invocations);
void mgl_queries_note_tess(GLMContext ctx, uint64_t tes_invocations, uint64_t tessellated_primitives);
/* transform feedback: primitives actually written to stream and primitives that were needed
 * (needed > written ⇒ overflow).  For the 'xfb' domain. */
void mgl_queries_note_xfb_primitives(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
/* false while conditional rendering discards rendering (for Clear*, BlitFramebuffer, ...). */
bool mgl_queries_render_allowed(GLMContext ctx);
bool mgl_queries_counting_active(GLMContext ctx);   /* gpuaudit: no CPU reads of indirect commands */
#ifdef __cplusplus
}
#endif

#endif
