/*
 * debug.c
 * MGL — KHR_debug output, object labels, robustness status
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * Message routing: every GL error raised through ctx->error_func (ERROR_RETURN /
 * ERROR_CHECK_RETURN macros) is reported as a GL_DEBUG_SOURCE_API /
 * GL_DEBUG_TYPE_ERROR / GL_DEBUG_SEVERITY_HIGH message (id = the error enum).
 * Messages go to the application callback when one is installed, otherwise
 * into a bounded log read with glGetDebugMessageLog. Delivery is always
 * synchronous (MGL is single-threaded per context), which satisfies both
 * settings of GL_DEBUG_OUTPUT_SYNCHRONOUS.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_debug.h"

#define DBG(ctx) ((ctx)->state.ext_debug)
#define MTL_DBG(ctx) ((ctx)->mtl_funcs.ext_debug)

#pragma mark - enum validation

static bool valid_source(GLenum s, bool allow_dont_care)
{
    switch (s) {
        case GL_DEBUG_SOURCE_API:
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        case GL_DEBUG_SOURCE_SHADER_COMPILER:
        case GL_DEBUG_SOURCE_THIRD_PARTY:
        case GL_DEBUG_SOURCE_APPLICATION:
        case GL_DEBUG_SOURCE_OTHER:
            return true;
        case GL_DONT_CARE:
            return allow_dont_care;
    }
    return false;
}

static bool valid_type(GLenum t, bool allow_dont_care)
{
    switch (t) {
        case GL_DEBUG_TYPE_ERROR:
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        case GL_DEBUG_TYPE_PORTABILITY:
        case GL_DEBUG_TYPE_PERFORMANCE:
        case GL_DEBUG_TYPE_OTHER:
        case GL_DEBUG_TYPE_MARKER:
        case GL_DEBUG_TYPE_PUSH_GROUP:
        case GL_DEBUG_TYPE_POP_GROUP:
            return true;
        case GL_DONT_CARE:
            return allow_dont_care;
    }
    return false;
}

static bool valid_severity(GLenum s, bool allow_dont_care)
{
    switch (s) {
        case GL_DEBUG_SEVERITY_HIGH:
        case GL_DEBUG_SEVERITY_MEDIUM:
        case GL_DEBUG_SEVERITY_LOW:
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            return true;
        case GL_DONT_CARE:
            return allow_dont_care;
    }
    return false;
}

#pragma mark - message control

static bool rule_matches(const MGLDebugRule *r, GLenum source, GLenum type, GLuint id, GLenum severity)
{
    if (r->source != GL_DONT_CARE && r->source != source) return false;
    if (r->type != GL_DONT_CARE && r->type != type) return false;
    if (r->has_id) return r->id == id;
    if (r->severity != GL_DONT_CARE && r->severity != severity) return false;
    return true;
}

static bool message_enabled(struct MGLExtState_debug *st, GLenum source, GLenum type, GLuint id, GLenum severity)
{
    const MGLDebugGroup *g = &st->groups[st->group_depth - 1];
    /* KHR_debug: everything is enabled initially except DEBUG_SEVERITY_LOW */
    bool enabled = severity != GL_DEBUG_SEVERITY_LOW;

    for (int i = 0; i < g->rule_count; i++)
        if (rule_matches(&g->rules[i], source, type, id, severity))
            enabled = g->rules[i].enabled;

    return enabled;
}

/* does new rule n make the older rule p irrelevant? */
static bool rule_supersedes(const MGLDebugRule *n, const MGLDebugRule *p)
{
    if (n->source != GL_DONT_CARE && n->source != p->source) return false;
    if (n->type != GL_DONT_CARE && n->type != p->type) return false;
    if (n->has_id)
        return p->has_id && p->id == n->id;
    if (p->has_id)
        return n->severity == GL_DONT_CARE;
    return n->severity == GL_DONT_CARE || n->severity == p->severity;
}

static void group_add_rule(MGLDebugGroup *g, const MGLDebugRule *r)
{
    int w = 0;
    for (int i = 0; i < g->rule_count; i++)
        if (!rule_supersedes(r, &g->rules[i]))
            g->rules[w++] = g->rules[i];
    g->rule_count = w;

    if (g->rule_count == g->rule_cap) {
        int cap = g->rule_cap ? g->rule_cap * 2 : 16;
        MGLDebugRule *n = realloc(g->rules, cap * sizeof(MGLDebugRule));
        if (!n) return;
        g->rules = n;
        g->rule_cap = cap;
    }
    g->rules[g->rule_count++] = *r;
}

#pragma mark - message emission

void mgl_debug_message(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char *msg)
{
    struct MGLExtState_debug *st;

    if (!ctx || !(st = DBG(ctx)))
        return;

    if (!ctx->state.caps.debug_output)
        return;

    if (!msg)
        msg = "";

    if (length < 0)
        length = (GLsizei)strlen(msg);

    if (length >= MGL_MAX_DEBUG_MESSAGE_LENGTH)
        length = MGL_MAX_DEBUG_MESSAGE_LENGTH - 1;

    if (!message_enabled(st, source, type, id, severity))
        return;

    if (st->callback)
    {
        char buf[MGL_MAX_DEBUG_MESSAGE_LENGTH];

        /* an error raised by a GL call made from inside the callback is still
           reported, but bound the recursion */
        if (st->in_callback > 4)
            return;

        memcpy(buf, msg, length);
        buf[length] = '\0';

        st->in_callback++;
        st->callback(source, type, id, severity, length, buf, st->user_param);
        st->in_callback--;
        return;
    }

    /* no callback: append to the log; when full, new messages are discarded */
    if (st->log_count >= MGL_MAX_DEBUG_LOGGED_MESSAGES)
        return;

    char *copy = malloc(length + 1);
    if (!copy)
        return;
    memcpy(copy, msg, length);
    copy[length] = '\0';

    MGLDebugMessage *m = &st->log[(st->log_head + st->log_count) % MGL_MAX_DEBUG_LOGGED_MESSAGES];
    m->source = source;
    m->type = type;
    m->id = id;
    m->severity = severity;
    m->length = length;
    m->msg = copy;
    st->log_count++;
}

void mgl_debug_messagef(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, const char *fmt, ...)
{
    char buf[MGL_MAX_DEBUG_MESSAGE_LENGTH];
    va_list ap;

    if (!ctx || !DBG(ctx) || !ctx->state.caps.debug_output)
        return;

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    mgl_debug_message(ctx, source, type, id, severity, -1, buf);
}

static const char *error_name(GLenum error)
{
    switch (error) {
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST";
    }
    return "GL error";
}

/* called by error_func() for every GL error */
void mgl_debug_report_error(GLMContext ctx, const char *func, GLenum error)
{
    if (getenv("MGL_ERR_TRACE"))
        fprintf(stderr, "MGL-ERR %s: 0x%04x (%s)\n", func ? func : "?", error, error_name(error));
    if (!ctx || !DBG(ctx) || !ctx->state.caps.debug_output)
        return;

    /* entry points are named mglFoo; report them as glFoo */
    if (func && strncmp(func, "mgl", 3) == 0)
        func += 1;

    mgl_debug_messagef(ctx, GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, error, GL_DEBUG_SEVERITY_HIGH,
                       "%s error generated in %s", error_name(error), func ? func : "(unknown)");
}

#pragma mark - KHR_debug entry points

void mglDebugMessageCallback(GLMContext ctx, GLDEBUGPROC callback, const void *userParam)
{
    struct MGLExtState_debug *st = DBG(ctx);

    st->callback = callback;
    st->user_param = userParam;
}

void mglDebugMessageControl(GLMContext ctx, GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled)
{
    struct MGLExtState_debug *st = DBG(ctx);
    MGLDebugGroup *g;

    if (!valid_source(source, true) || !valid_type(type, true) || !valid_severity(severity, true))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (count < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (count > 0 && (source == GL_DONT_CARE || type == GL_DONT_CARE || severity != GL_DONT_CARE))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    g = &st->groups[st->group_depth - 1];

    if (count > 0)
    {
        if (!ids)
        {
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }

        for (GLsizei i = 0; i < count; i++)
        {
            MGLDebugRule r = { source, type, GL_DONT_CARE, ids[i], GL_TRUE, enabled ? GL_TRUE : GL_FALSE };
            group_add_rule(g, &r);
        }
    }
    else
    {
        MGLDebugRule r = { source, type, severity, 0, GL_FALSE, enabled ? GL_TRUE : GL_FALSE };
        group_add_rule(g, &r);
    }
}

void mglDebugMessageInsert(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf)
{
    if (source != GL_DEBUG_SOURCE_APPLICATION && source != GL_DEBUG_SOURCE_THIRD_PARTY)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!valid_type(type, false) || !valid_severity(severity, false))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!buf)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (length < 0)
        length = (GLsizei)strlen(buf);

    if (length >= MGL_MAX_DEBUG_MESSAGE_LENGTH)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (type == GL_DEBUG_TYPE_MARKER && MTL_DBG(ctx) && MTL_DBG(ctx)->mtlInsertDebugSignpost)
    {
        char tmp[MGL_MAX_DEBUG_MESSAGE_LENGTH];
        memcpy(tmp, buf, length);
        tmp[length] = '\0';
        MTL_DBG(ctx)->mtlInsertDebugSignpost(ctx, tmp);
    }

    mgl_debug_message(ctx, source, type, id, severity, length, buf);
}

GLuint mglGetDebugMessageLog(GLMContext ctx, GLuint count, GLsizei bufSize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog)
{
    struct MGLExtState_debug *st = DBG(ctx);
    GLuint n = 0;
    GLsizei used = 0;

    if (messageLog && bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return 0;
    }

    while (n < count && st->log_count > 0)
    {
        MGLDebugMessage *m = &st->log[st->log_head];
        GLsizei need = m->length + 1;

        if (messageLog)
        {
            if (used + need > bufSize)
                break;

            memcpy(messageLog + used, m->msg, m->length);
            messageLog[used + m->length] = '\0';
            used += need;
        }

        if (sources) sources[n] = m->source;
        if (types) types[n] = m->type;
        if (ids) ids[n] = m->id;
        if (severities) severities[n] = m->severity;
        if (lengths) lengths[n] = need;

        free(m->msg);
        m->msg = NULL;
        st->log_head = (st->log_head + 1) % MGL_MAX_DEBUG_LOGGED_MESSAGES;
        st->log_count--;
        n++;
    }

    return n;
}

void mglPushDebugGroup(GLMContext ctx, GLenum source, GLuint id, GLsizei length, const GLchar *message)
{
    struct MGLExtState_debug *st = DBG(ctx);
    MGLDebugGroup *parent, *g;

    if (source != GL_DEBUG_SOURCE_APPLICATION && source != GL_DEBUG_SOURCE_THIRD_PARTY)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!message)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (length < 0)
        length = (GLsizei)strlen(message);

    if (length >= MGL_MAX_DEBUG_MESSAGE_LENGTH)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (st->group_depth >= MGL_MAX_DEBUG_GROUP_STACK_DEPTH)
    {
        ERROR_RETURN(GL_STACK_OVERFLOW);
        return;
    }

    parent = &st->groups[st->group_depth - 1];
    g = &st->groups[st->group_depth];
    memset(g, 0, sizeof(*g));

    g->source = source;
    g->id = id;
    g->length = length;
    g->message = malloc(length + 1);
    if (!g->message)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }
    memcpy(g->message, message, length);
    g->message[length] = '\0';

    /* the new group starts with a copy of the parent's message control state */
    if (parent->rule_count)
    {
        g->rules = malloc(parent->rule_count * sizeof(MGLDebugRule));
        if (g->rules)
        {
            memcpy(g->rules, parent->rules, parent->rule_count * sizeof(MGLDebugRule));
            g->rule_count = g->rule_cap = parent->rule_count;
        }
    }

    st->group_depth++;

    mgl_debug_message(ctx, source, GL_DEBUG_TYPE_PUSH_GROUP, id, GL_DEBUG_SEVERITY_NOTIFICATION, length, g->message);

    if (MTL_DBG(ctx) && MTL_DBG(ctx)->mtlPushDebugGroup)
        MTL_DBG(ctx)->mtlPushDebugGroup(ctx, g->message);
}

void mglPopDebugGroup(GLMContext ctx)
{
    struct MGLExtState_debug *st = DBG(ctx);
    MGLDebugGroup g;

    if (st->group_depth <= 1)
    {
        ERROR_RETURN(GL_STACK_UNDERFLOW);
        return;
    }

    g = st->groups[st->group_depth - 1];
    memset(&st->groups[st->group_depth - 1], 0, sizeof(MGLDebugGroup));
    st->group_depth--;
    free(g.rules);

    if (MTL_DBG(ctx) && MTL_DBG(ctx)->mtlPopDebugGroup)
        MTL_DBG(ctx)->mtlPopDebugGroup(ctx);

    /* the pop message is filtered by the (restored) outer group's state */
    mgl_debug_message(ctx, g.source, GL_DEBUG_TYPE_POP_GROUP, g.id, GL_DEBUG_SEVERITY_NOTIFICATION, g.length, g.message);

    free(g.message);
}

#pragma mark - object labels

void *mgl_debug_lookup_object(GLMContext ctx, GLenum identifier, GLuint name)
{
    HashTable *table = NULL;

    switch (identifier)
    {
        case GL_BUFFER:             table = &STATE(buffer_table); break;
        case GL_SHADER:             table = &STATE(shader_table); break;
        case GL_PROGRAM:            table = &STATE(program_table); break;
        case GL_VERTEX_ARRAY:       table = &STATE(vao_table); break;
        case GL_PROGRAM_PIPELINE:   table = &STATE(program_pipeline_table); break;
        case GL_TRANSFORM_FEEDBACK: table = &STATE(transform_feedback_table); break;
        case GL_SAMPLER:            table = &STATE(sampler_table); break;
        case GL_TEXTURE:            table = &STATE(texture_table); break;
        case GL_RENDERBUFFER:       table = &STATE(renderbuffer_table); break;
        case GL_FRAMEBUFFER:        table = &STATE(framebuffer_table); break;
        default:
            return NULL;
    }

    if (name == 0 || table->keys == NULL || name >= table->size)
        return NULL;

    return searchHashTable(table, name);
}

static bool valid_label_identifier(GLenum identifier)
{
    switch (identifier)
    {
        case GL_BUFFER: case GL_SHADER: case GL_PROGRAM: case GL_VERTEX_ARRAY:
        case GL_QUERY: case GL_PROGRAM_PIPELINE: case GL_TRANSFORM_FEEDBACK:
        case GL_SAMPLER: case GL_TEXTURE: case GL_RENDERBUFFER: case GL_FRAMEBUFFER:
            return true;
    }
    return false;
}

/* resolve (identifier, name) to "exists" + object pointer; false if no such object */
static bool resolve_object(GLMContext ctx, GLenum identifier, GLuint name, void **object)
{
    *object = NULL;

    if (identifier == GL_QUERY)
        return mglIsQuery(ctx, name) == GL_TRUE;

    if (identifier == GL_TRANSFORM_FEEDBACK && name == 0)
        return true;    /* the default transform feedback object always exists */

    *object = mgl_debug_lookup_object(ctx, identifier, name);
    return *object != NULL;
}

static unsigned label_bucket(GLenum identifier, uintptr_t name)
{
    uintptr_t h = (uintptr_t)identifier * 2654435761u ^ (name * 40503u) ^ (name >> 16);
    return (unsigned)(h % MGL_DEBUG_LABEL_BUCKETS);
}

static MGLDebugLabel **find_label_slot(struct MGLExtState_debug *st, GLenum identifier, uintptr_t name)
{
    MGLDebugLabel **pp = &st->labels[label_bucket(identifier, name)];

    while (*pp && !((*pp)->identifier == identifier && (*pp)->name == name))
        pp = &(*pp)->next;

    return pp;
}

static void remove_label(struct MGLExtState_debug *st, MGLDebugLabel **pp)
{
    MGLDebugLabel *e = *pp;
    *pp = e->next;
    free(e->label);
    free(e);
    st->label_count--;
}

/* current label entry for an existing object; drops it if the name now refers to a different object */
static MGLDebugLabel *get_label(struct MGLExtState_debug *st, GLenum identifier, uintptr_t name, void *object)
{
    MGLDebugLabel **pp = find_label_slot(st, identifier, name);

    if (!*pp)
        return NULL;

    if ((*pp)->object != object)
    {
        remove_label(st, pp);
        return NULL;
    }

    return *pp;
}

static void set_label(GLMContext ctx, const char *fn, GLenum identifier, uintptr_t name, void *object, GLsizei length, const GLchar *label)
{
    struct MGLExtState_debug *st = DBG(ctx);
    MGLDebugLabel **pp, *e;

    if (label)
    {
        if (length < 0)
            length = (GLsizei)strlen(label);

        if (length >= MGL_MAX_LABEL_LENGTH)
        {
            ctx->error_func(ctx, fn, GL_INVALID_VALUE);
            return;
        }
    }

    pp = find_label_slot(st, identifier, name);

    if (!label || length == 0)
    {
        /* NULL (or empty) label removes any existing label */
        if (*pp)
        {
            (*pp)->length = 0;
            if (MTL_DBG(ctx) && MTL_DBG(ctx)->mtlSetObjectLabel && (*pp)->object &&
                (*pp)->object == mgl_debug_lookup_object(ctx, identifier, (GLuint)name))
            {
                (*pp)->label[0] = '\0';
                (*pp)->applied_mtl = NULL;
                MTL_DBG(ctx)->mtlSetObjectLabel(ctx, *pp);
            }
            remove_label(st, pp);
        }
        return;
    }

    e = *pp;
    if (!e)
    {
        e = calloc(1, sizeof(MGLDebugLabel));
        if (!e)
        {
            ctx->error_func(ctx, fn, GL_OUT_OF_MEMORY);
            return;
        }
        e->identifier = identifier;
        e->name = name;
        *pp = e;
        st->label_count++;
    }

    char *copy = malloc(length + 1);
    if (!copy)
    {
        ctx->error_func(ctx, fn, GL_OUT_OF_MEMORY);
        return;
    }
    memcpy(copy, label, length);
    copy[length] = '\0';

    free(e->label);
    e->label = copy;
    e->length = length;
    e->object = object;
    e->applied_mtl = NULL;

    if (object && MTL_DBG(ctx) && MTL_DBG(ctx)->mtlSetObjectLabel)
        MTL_DBG(ctx)->mtlSetObjectLabel(ctx, e);
}

static void get_label_string(GLMContext ctx, MGLDebugLabel *e, GLsizei bufSize, GLsizei *length, GLchar *label)
{
    GLsizei len = e ? e->length : 0;

    if (!label)
    {
        if (length) *length = len;
        return;
    }

    if (bufSize == 0)
    {
        if (length) *length = 0;
        return;
    }

    if (len > bufSize - 1)
        len = bufSize - 1;

    if (len)
        memcpy(label, e->label, len);
    label[len] = '\0';

    if (length) *length = len;
}

void mglObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei length, const GLchar *label)
{
    void *object;

    if (!valid_label_identifier(identifier))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!resolve_object(ctx, identifier, name, &object))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    set_label(ctx, "mglObjectLabel", identifier, name, object, length, label);
}

void mglGetObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label)
{
    void *object;

    if (!valid_label_identifier(identifier))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    if (!resolve_object(ctx, identifier, name, &object))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    get_label_string(ctx, get_label(DBG(ctx), identifier, name, object), bufSize, length, label);
}

void mglObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei length, const GLchar *label)
{
    if (!ptr || mglIsSync(ctx, (GLsync)ptr) != GL_TRUE)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    set_label(ctx, "mglObjectPtrLabel", GL_SYNC_FENCE, (uintptr_t)ptr, (void *)ptr, length, label);
}

void mglGetObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label)
{
    if (!ptr || mglIsSync(ctx, (GLsync)ptr) != GL_TRUE)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    get_label_string(ctx, get_label(DBG(ctx), GL_SYNC_FENCE, (uintptr_t)ptr, (void *)ptr), bufSize, length, label);
}

void mgl_debug_refresh_metal_labels(GLMContext ctx)
{
    struct MGLExtState_debug *st = DBG(ctx);

    if (!st || st->label_count == 0 || !MTL_DBG(ctx) || !MTL_DBG(ctx)->mtlSetObjectLabel)
        return;

    for (int b = 0; b < MGL_DEBUG_LABEL_BUCKETS; b++)
    {
        for (MGLDebugLabel *e = st->labels[b]; e; e = e->next)
        {
            if (e->identifier == GL_SYNC_FENCE || e->identifier == GL_QUERY || !e->object)
                continue;

            /* only touch objects that are still alive under this name */
            if (mgl_debug_lookup_object(ctx, e->identifier, (GLuint)e->name) != e->object)
                continue;

            MTL_DBG(ctx)->mtlSetObjectLabel(ctx, e);
        }
    }
}

#pragma mark - robustness / pointers

GLenum mglGetGraphicsResetStatus(GLMContext ctx)
{
    struct MGLExtState_debug *st = DBG(ctx);
    GLenum status = __atomic_exchange_n(&st->pending_reset, GL_NO_ERROR, __ATOMIC_SEQ_CST);

    if (status != GL_NO_ERROR)
        return status;

    if (MTL_DBG(ctx) && MTL_DBG(ctx)->mtlGetResetStatus)
        return MTL_DBG(ctx)->mtlGetResetStatus(ctx);

    return GL_NO_ERROR;
}

void mglGetPointerv(GLMContext ctx, GLenum pname, void **params)
{
    struct MGLExtState_debug *st = DBG(ctx);

    switch (pname)
    {
        case GL_DEBUG_CALLBACK_FUNCTION:
            if (params) *params = (void *)st->callback;
            break;

        case GL_DEBUG_CALLBACK_USER_PARAM:
            if (params) *params = (void *)st->user_param;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

#pragma mark - domain 'debug' hooks

void mgl_ext_init_debug(GLMContext ctx)
{
    struct MGLExtState_debug *st = calloc(1, sizeof(struct MGLExtState_debug));
    if (!st) {
        fprintf(stderr, "MGL: out of memory allocating debug state\n");
        abort();
    }

    st->group_depth = 1;    /* the default debug group */
    st->groups[0].source = GL_DEBUG_SOURCE_APPLICATION;

    ctx->state.ext_debug = st;
}

bool mgl_ext_get_debug(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    struct MGLExtState_debug *st = DBG(ctx);
    double v;

    switch (pname)
    {
        case GL_DEBUG_LOGGED_MESSAGES: v = st->log_count; break;
        case GL_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH:
            v = st->log_count ? st->log[st->log_head].length + 1 : 0;
            break;
        case GL_MAX_DEBUG_MESSAGE_LENGTH: v = MGL_MAX_DEBUG_MESSAGE_LENGTH; break;
        case GL_MAX_DEBUG_LOGGED_MESSAGES: v = MGL_MAX_DEBUG_LOGGED_MESSAGES; break;
        case GL_MAX_DEBUG_GROUP_STACK_DEPTH: v = MGL_MAX_DEBUG_GROUP_STACK_DEPTH; break;
        case GL_DEBUG_GROUP_STACK_DEPTH: v = st->group_depth; break;
        case GL_MAX_LABEL_LENGTH: v = MGL_MAX_LABEL_LENGTH; break;
        case GL_DEBUG_OUTPUT: v = ctx->state.caps.debug_output; break;
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: v = ctx->state.caps.debug_output_synchronous; break;
        default:
            return false;
    }

    mgl_get_store(type, data, 0, v);
    return true;
}
