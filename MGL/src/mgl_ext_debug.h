/* mgl_ext_debug.h — private state and Metal hooks of the 'debug' domain. Owned by agent 'debug'.
 *
 * KHR_debug (GL 4.3 core): message callback / log / control, debug groups,
 * object labels, plus robustness reset status. Other domains may emit debug
 * messages with mgl_debug_message() / mgl_debug_messagef().
 */
#ifndef MGL_EXT_DEBUG_H
#define MGL_EXT_DEBUG_H

#include <stdint.h>
#include "glm_context.h"
#include "mgl_ext.h"

/* implementation limits (GL 4.6 minimums: 1 / 1 / 64 / 256) */
#define MGL_MAX_DEBUG_MESSAGE_LENGTH    1024
#define MGL_MAX_DEBUG_LOGGED_MESSAGES   64
#define MGL_MAX_DEBUG_GROUP_STACK_DEPTH 64
#define MGL_MAX_LABEL_LENGTH            256

typedef struct MGLDebugMessage_t {
    GLenum  source, type, severity;
    GLuint  id;
    GLsizei length;             /* strlen(msg) */
    char    *msg;
} MGLDebugMessage;

/* one DebugMessageControl call; evaluated in order, the last matching rule wins */
typedef struct MGLDebugRule_t {
    GLenum    source, type, severity;   /* GL_DONT_CARE = wildcard */
    GLuint    id;
    GLboolean has_id;                   /* rule from an id list (severity ignored) */
    GLboolean enabled;
} MGLDebugRule;

typedef struct MGLDebugGroup_t {
    GLenum  source;
    GLuint  id;
    char    *message;
    GLsizei length;
    MGLDebugRule *rules;
    int     rule_count, rule_cap;
} MGLDebugGroup;

typedef struct MGLDebugLabel_t {
    GLenum    identifier;
    uintptr_t name;             /* GL name, or GLsync pointer for GL_SYNC_FENCE */
    void      *object;          /* GL object pointer when known (staleness check) */
    char      *label;
    GLsizei   length;
    void      *applied_mtl;     /* Metal object last given this label */
    struct MGLDebugLabel_t *next;
} MGLDebugLabel;

#define MGL_DEBUG_LABEL_BUCKETS 256

/* GL-side state, reachable as ctx->state.ext_debug; allocated in mgl_ext_init_debug(). */
struct MGLExtState_debug {
    GLDEBUGPROC callback;
    const void  *user_param;

    MGLDebugMessage log[MGL_MAX_DEBUG_LOGGED_MESSAGES];   /* ring buffer */
    int log_head, log_count;

    MGLDebugGroup groups[MGL_MAX_DEBUG_GROUP_STACK_DEPTH]; /* [0] = default group */
    int group_depth;            /* GL_DEBUG_GROUP_STACK_DEPTH (>= 1) */

    MGLDebugLabel *labels[MGL_DEBUG_LABEL_BUCKETS];
    int label_count;

    int in_callback;            /* recursion guard for errors raised inside the callback */

    /* GL_RENDERER / GL_VENDOR / GL_VERSION strings (built lazily) */
    char vendor[64];
    char renderer[192];
    char version[64];

    /* reset-status latch */
    GLenum pending_reset;
    unsigned long reported_gpu_errors;

    /* Metal mirror of debug groups on the current render encoder */
    void *mtl_group_encoder;    /* identity only, not retained */
    int   mtl_group_depth;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_debug; filled by MGLRenderer+Debug.m. */
struct MGLExtMetal_debug {
    void (*mtlPushDebugGroup)(GLMContext ctx, const char *message);
    void (*mtlPopDebugGroup)(GLMContext ctx);
    void (*mtlInsertDebugSignpost)(GLMContext ctx, const char *message);
    void (*mtlSetObjectLabel)(GLMContext ctx, MGLDebugLabel *entry);
    const char *(*mtlDeviceName)(GLMContext ctx);
    GLenum (*mtlGetResetStatus)(GLMContext ctx);
};

#ifdef __cplusplus
extern "C" {
#endif
/* emit a KHR_debug message (honours DEBUG_OUTPUT and DebugMessageControl); length < 0 = NUL-terminated */
void mgl_debug_message(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char *msg);
void mgl_debug_messagef(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, const char *fmt, ...)
    __attribute__((format(printf, 6, 7)));
/* report a GL error as a KHR_debug message (called from error_func) */
void mgl_debug_report_error(GLMContext ctx, const char *func, GLenum error);
/* the live GL object for (identifier, name), or NULL if none (used for label staleness checks) */
void *mgl_debug_lookup_object(GLMContext ctx, GLenum identifier, GLuint name);
/* reapply labels whose Metal objects were (re)created since labelled (called from the renderer) */
void mgl_debug_refresh_metal_labels(GLMContext ctx);
#ifdef __cplusplus
}
#endif

#endif
