/*
 * transform_feedback.c
 * MGL — transform feedback objects, varyings, capture and DrawTransformFeedback.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * See mgl_ext_xfb.h for the overall mechanism (Metal has no transform feedback; the vertex
 * shader is recompiled into a capture variant and primitives are assembled on the CPU).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <dlfcn.h>

#include <glslang_c_interface.h>
#include "fp64.h"
#include "mgl_ext_introspect.h"
#include <glslang_c_shader_types.h>
#include "spirv_cross_c.h"
#include "spirv.h"

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_gpuaudit.h"   /* gpuaudit: xfb scatter without a CPU round trip */
#include "msl_reserved.h"
#include "hash_table.h"

TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name);

/* other domains' helpers (no shared header exports them) */
Program *findProgram(GLMContext ctx, GLuint program);
int isShader(GLMContext ctx, GLuint shader);
Buffer *findBuffer(GLMContext ctx, GLuint buffer);
void initGLSLInput(GLMContext ctx, GLuint type, const char *src, glslang_input_t *input);
const glslang_resource_t *glslang_default_resource(void);

/* the 'queries' domain counts TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN / overflow queries through
 * mgl_queries_note_xfb_primitives() (mgl_ext_queries.h).  Looked up at run time so this domain
 * builds with or without that branch merged. */
typedef void (*xfb_note_queries_fn)(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
static xfb_note_queries_fn xfb_queries_note(void)
{
    static int looked;
    static xfb_note_queries_fn fn;
    if (!looked) { fn = (xfb_note_queries_fn)dlsym(RTLD_DEFAULT, "mgl_queries_note_xfb_primitives"); looked = 1; }
    return fn;
}

#define XFB()               (ctx->state.ext_xfb)
#define XFB_ERR(e)          do { ctx->error_func(ctx, __FUNCTION__, (e)); return; } while (0)
#define XFB_ERR_V(e, v)     do { ctx->error_func(ctx, __FUNCTION__, (e)); return (v); } while (0)
#define TFB_BASE()          (ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers)

#pragma mark - objects

static MGLXfbObject *xfb_obj(GLMContext ctx, GLuint name)
{
    if (!XFB()) return NULL;
    if (name == 0) return &XFB()->default_obj;
    if (name >= XFB()->obj_cap) return NULL;
    return XFB()->objs[name];
}

static MGLXfbObject *xfb_current(GLMContext ctx)
{
    GLuint name = STATE(transform_feedback) ? STATE(transform_feedback)->name : 0;
    MGLXfbObject *o = xfb_obj(ctx, name);
    return o ? o : &XFB()->default_obj;
}

static MGLXfbObject *xfb_new_obj(GLMContext ctx, GLuint name)
{
    if (name >= XFB()->obj_cap) {
        GLuint cap = XFB()->obj_cap ? XFB()->obj_cap : 16;
        while (cap <= name) cap *= 2;
        MGLXfbObject **n = realloc(XFB()->objs, cap * sizeof(*n));
        if (!n) return NULL;
        memset(n + XFB()->obj_cap, 0, (cap - XFB()->obj_cap) * sizeof(*n));
        XFB()->objs = n;
        XFB()->obj_cap = cap;
    }
    if (!XFB()->objs[name]) {
        MGLXfbObject *o = calloc(1, sizeof(MGLXfbObject));
        if (!o) return NULL;
        o->name = name;
        o->primitive_mode = GL_POINTS;
        XFB()->objs[name] = o;
    }
    return XFB()->objs[name];
}

/* resolve a binding point to a live buffer (deleted buffers count as unbound) */
static Buffer *xfb_binding_buffer(GLMContext ctx, BufferBaseTarget *t)
{
    if (!t->buffer) return NULL;
    Buffer *b = findBuffer(ctx, t->buffer);
    if (!b || b != t->buf) return NULL;
    return b;
}

static BufferBaseTarget *xfb_bindings(GLMContext ctx, MGLXfbObject *o)
{
    return (o == xfb_current(ctx)) ? TFB_BASE() : o->saved;
}

/* usable byte range of binding index b of object o */
static GLsizeiptr xfb_binding_size(GLMContext ctx, MGLXfbObject *o, GLuint b, Buffer **out_buf, GLintptr *out_off)
{
    BufferBaseTarget *t = &xfb_bindings(ctx, o)[b];
    Buffer *buf = xfb_binding_buffer(ctx, t);
    *out_buf = buf;
    *out_off = t->offset;
    if (!buf) return 0;
    GLsizeiptr total = (GLsizeiptr)buf->size;
    if (buf->data.buffer_size && (GLsizeiptr)buf->data.buffer_size < total) total = (GLsizeiptr)buf->data.buffer_size;
    if (t->offset >= total) return 0;
    GLsizeiptr avail = total - t->offset;
    if (!o->is_base[b] && t->size < avail) avail = t->size;
    return avail;
}

#pragma mark - programs

static MGLXfbProgram *xfb_prog(GLMContext ctx, Program *p, bool create)
{
    if (!XFB() || !p) return NULL;
    for (GLuint i = 0; i < XFB()->prog_count; i++) {
        MGLXfbProgram *xp = XFB()->progs[i];
        if (xp->name == p->name) {
            if (xp->program == p) return xp;
            /* name reused by a new program object: drop stale data */
            xp->program = p;
            return xp;
        }
    }
    if (!create) return NULL;
    if (XFB()->prog_count == XFB()->prog_cap) {
        GLuint cap = XFB()->prog_cap ? XFB()->prog_cap * 2 : 16;
        MGLXfbProgram **n = realloc(XFB()->progs, cap * sizeof(*n));
        if (!n) return NULL;
        XFB()->progs = n;
        XFB()->prog_cap = cap;
    }
    MGLXfbProgram *xp = calloc(1, sizeof(MGLXfbProgram));
    if (!xp) return NULL;
    xp->name = p->name;
    xp->program = p;
    xp->pending_mode = GL_INTERLEAVED_ATTRIBS;
    xp->mode = GL_INTERLEAVED_ATTRIBS;
    XFB()->progs[XFB()->prog_count++] = xp;
    return xp;
}

static void xfb_clear_linked(GLMContext ctx, MGLXfbProgram *xp)
{
    for (GLuint i = 0; i < xp->count; i++) {
        free(xp->vars[i].name);
        free(xp->vars[i].expr);
    }
    free(xp->vars);
    xp->vars = NULL;
    xp->count = 0;
    free(xp->link_log); xp->link_log = NULL;
    free(xp->msl); xp->msl = NULL;
    free(xp->entry); xp->entry = NULL;
    if (ctx->mtl_funcs.ext_xfb && ctx->mtl_funcs.ext_xfb->release_program)
        ctx->mtl_funcs.ext_xfb->release_program(ctx, xp);
    xp->link_failed = false;
    xp->from_shader = false;
    xp->buffer_mask = 0;
    xp->record_words = 0;
    xp->max_name_length = 0;
    memset(xp->stride, 0, sizeof(xp->stride));
}

static Program *xfb_active_program(GLMContext ctx)
{
    if (STATE(program)) return STATE(program);
    if (STATE(program_pipeline)) return STATE(program_pipeline)->stage_programs[_VERTEX_SHADER];
    return NULL;
}

#pragma mark - GL entry points: varyings

void mglTransformFeedbackVaryings(GLMContext ctx, GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode)
{
    Program *p = findProgram(ctx, program);

    if (!p) {
        if (isShader(ctx, program)) XFB_ERR(GL_INVALID_OPERATION);
        XFB_ERR(GL_INVALID_VALUE);
    }
    if (count < 0) XFB_ERR(GL_INVALID_VALUE);
    if (bufferMode != GL_INTERLEAVED_ATTRIBS && bufferMode != GL_SEPARATE_ATTRIBS) XFB_ERR(GL_INVALID_ENUM);
    if (bufferMode == GL_SEPARATE_ATTRIBS && count > MGL_XFB_MAX_SEPARATE_ATTRIBS) XFB_ERR(GL_INVALID_VALUE);
    if (xfb_current(ctx)->active) XFB_ERR(GL_INVALID_OPERATION);
    if (count && !varyings) XFB_ERR(GL_INVALID_VALUE);

    MGLXfbProgram *xp = xfb_prog(ctx, p, true);
    if (!xp) XFB_ERR(GL_OUT_OF_MEMORY);

    for (GLsizei i = 0; i < xp->pending_count; i++) free(xp->pending[i]);
    free(xp->pending);
    xp->pending = NULL;
    xp->pending_count = 0;

    if (count) {
        xp->pending = calloc((size_t)count, sizeof(char *));
        if (!xp->pending) XFB_ERR(GL_OUT_OF_MEMORY);
        for (GLsizei i = 0; i < count; i++)
            xp->pending[i] = strdup(varyings[i] ? varyings[i] : "");
    }
    xp->pending_count = count;
    xp->pending_mode = bufferMode;
}

void mglGetTransformFeedbackVarying(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name)
{
    Program *p = findProgram(ctx, program);

    if (!p) {
        if (isShader(ctx, program)) XFB_ERR(GL_INVALID_OPERATION);
        XFB_ERR(GL_INVALID_VALUE);
    }
    if (bufSize < 0) XFB_ERR(GL_INVALID_VALUE);

    MGLXfbProgram *xp = xfb_prog(ctx, p, false);
    if (!xp || !xp->linked || xp->link_failed || index >= xp->count) XFB_ERR(GL_INVALID_VALUE);

    MGLXfbVarying *v = &xp->vars[index];
    GLsizei n = 0;
    if (name && bufSize > 0) {
        n = (GLsizei)strlen(v->name);
        if (n > bufSize - 1) n = bufSize - 1;
        memcpy(name, v->name, (size_t)n);
        name[n] = 0;
    }
    if (length) *length = n;
    if (size) *size = v->size;
    if (type) *type = v->type;
}

bool mgl_xfb_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params)
{
    MGLXfbProgram *xp = xfb_prog(ctx, program, false);
    bool ok = xp && xp->linked && !xp->link_failed;

    switch (pname) {
        case GL_TRANSFORM_FEEDBACK_VARYINGS:
            *params = ok ? (GLint)xp->count : 0;
            return true;
        case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
            *params = ok ? xp->max_name_length : 0;
            return true;
        case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:
            *params = (GLint)(xp && xp->linked ? xp->mode : GL_INTERLEAVED_ATTRIBS);
            return true;
    }
    return false;
}

bool mgl_xfb_link_failed(GLMContext ctx, Program *program, const char **log)
{
    MGLXfbProgram *xp = xfb_prog(ctx, program, false);
    if (!xp || !xp->link_failed) return false;
    if (log) *log = xp->link_log ? xp->link_log : "";
    return true;
}

const MGLXfbProgram *mgl_xfb_program_info(GLMContext ctx, Program *program)
{
    MGLXfbProgram *xp = xfb_prog(ctx, program, false);
    return (xp && xp->linked && !xp->link_failed) ? xp : NULL;
}

#pragma mark - GL entry points: objects

void mglGenTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
    if (n < 0) XFB_ERR(GL_INVALID_VALUE);
    if (n && !ids) XFB_ERR(GL_INVALID_VALUE);

    for (GLsizei i = 0; i < n; i++) {
        GLuint name = getNewName(&STATE(transform_feedback_table));
        getTransformFeedback(ctx, name);
        if (!xfb_new_obj(ctx, name)) XFB_ERR(GL_OUT_OF_MEMORY);
        ids[i] = name;
    }
}

void mglCreateTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
    if (n < 0) XFB_ERR(GL_INVALID_VALUE);
    if (n && !ids) XFB_ERR(GL_INVALID_VALUE);

    mglGenTransformFeedbacks(ctx, n, ids);
    for (GLsizei i = 0; i < n; i++) {
        MGLXfbObject *o = xfb_obj(ctx, ids[i]);
        if (o) o->created = true;
    }
}

GLboolean mglIsTransformFeedback(GLMContext ctx, GLuint id)
{
    if (id == 0) return GL_FALSE;
    MGLXfbObject *o = xfb_obj(ctx, id);
    if (!o || !findTransformFeedback(ctx, id)) return GL_FALSE;
    return o->created ? GL_TRUE : GL_FALSE;
}

/* swap the TF binding points between the context and the object being (un)bound */
static void xfb_switch_bindings(GLMContext ctx, MGLXfbObject *from, MGLXfbObject *to)
{
    if (from == to) return;
    memcpy(from->saved, TFB_BASE(), sizeof(from->saved));
    memcpy(TFB_BASE(), to->saved, sizeof(to->saved));
    STATE(dirty_bits) |= DIRTY_BUFFER_BASE_STATE;
}

void mglBindTransformFeedback(GLMContext ctx, GLenum target, GLuint id)
{
    if (target != GL_TRANSFORM_FEEDBACK) XFB_ERR(GL_INVALID_ENUM);

    MGLXfbObject *cur = xfb_current(ctx);
    if (cur->active && !cur->paused) XFB_ERR(GL_INVALID_OPERATION);

    MGLXfbObject *o = xfb_obj(ctx, id);
    TransformFeedback *t = id ? findTransformFeedback(ctx, id) : NULL;
    if (id && (!o || !t)) XFB_ERR(GL_INVALID_OPERATION);

    xfb_switch_bindings(ctx, cur, o);

    if (id) {
        t->target = target;
        o->created = true;
        STATE(transform_feedback) = t;
    } else {
        STATE(transform_feedback) = NULL;
    }
}

void mglDeleteTransformFeedbacks(GLMContext ctx, GLsizei n, const GLuint *ids)
{
    if (n < 0) XFB_ERR(GL_INVALID_VALUE);
    if (n && !ids) XFB_ERR(GL_INVALID_VALUE);

    /* GL: INVALID_OPERATION if any of the objects is active; nothing is deleted then */
    for (GLsizei i = 0; i < n; i++) {
        MGLXfbObject *o = ids[i] ? xfb_obj(ctx, ids[i]) : NULL;
        if (o && o->active) XFB_ERR(GL_INVALID_OPERATION);
    }

    for (GLsizei i = 0; i < n; i++) {
        if (ids[i] == 0) continue;
        TransformFeedback *ptr = findTransformFeedback(ctx, ids[i]);
        MGLXfbObject *o = xfb_obj(ctx, ids[i]);
        if (!ptr || !o) continue;

        if (STATE(transform_feedback) == ptr) {
            xfb_switch_bindings(ctx, o, &XFB()->default_obj);
            STATE(transform_feedback) = NULL;
        }

        deleteHashElement(&STATE(transform_feedback_table), ids[i]);
        free(ptr);
        XFB()->objs[ids[i]] = NULL;
        free(o);
    }
}

#pragma mark - GL entry points: begin / end

static bool xfb_mode_compatible(GLenum tf_mode, GLenum draw_mode)
{
    switch (tf_mode) {
        case GL_POINTS:
            return draw_mode == GL_POINTS;
        case GL_LINES:
            return draw_mode == GL_LINES || draw_mode == GL_LINE_LOOP || draw_mode == GL_LINE_STRIP ||
                   draw_mode == GL_LINES_ADJACENCY || draw_mode == GL_LINE_STRIP_ADJACENCY;
        case GL_TRIANGLES:
            return draw_mode == GL_TRIANGLES || draw_mode == GL_TRIANGLE_STRIP || draw_mode == GL_TRIANGLE_FAN ||
                   draw_mode == GL_TRIANGLES_ADJACENCY || draw_mode == GL_TRIANGLE_STRIP_ADJACENCY;
    }
    return false;
}

void mglBeginTransformFeedback(GLMContext ctx, GLenum primitiveMode)
{
    if (primitiveMode != GL_POINTS && primitiveMode != GL_LINES && primitiveMode != GL_TRIANGLES)
        XFB_ERR(GL_INVALID_ENUM);

    MGLXfbObject *o = xfb_current(ctx);
    if (o->active) XFB_ERR(GL_INVALID_OPERATION);

    Program *p = xfb_active_program(ctx);
    if (!p) XFB_ERR(GL_INVALID_OPERATION);

    MGLXfbProgram *xp = xfb_prog(ctx, p, false);
    if (!xp || !xp->linked || xp->link_failed || xp->count == 0 || xp->buffer_mask == 0)
        XFB_ERR(GL_INVALID_OPERATION);

    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++) {
        if (!(xp->buffer_mask & (1u << b))) continue;
        Buffer *buf;
        GLintptr off;
        xfb_binding_size(ctx, o, b, &buf, &off);
        if (!buf) XFB_ERR(GL_INVALID_OPERATION);
        if (buf->mapped && !(buf->access & GL_MAP_PERSISTENT_BIT)) XFB_ERR(GL_INVALID_OPERATION);
    }

    o->active = true;
    o->paused = false;
    o->primitive_mode = primitiveMode;
    o->program = p;
    o->program_name = p->name;
    memset(o->write_offset, 0, sizeof(o->write_offset));
    memset(o->vertices, 0, sizeof(o->vertices));
    memset(o->prims_written, 0, sizeof(o->prims_written));
    memset(o->prims_needed, 0, sizeof(o->prims_needed));

    if (STATE(transform_feedback)) STATE(transform_feedback)->active = GL_TRUE,
                                   STATE(transform_feedback)->paused = GL_FALSE,
                                   STATE(transform_feedback)->primitive_mode = primitiveMode;
}

void mglPauseTransformFeedback(GLMContext ctx)
{
    /* the geometry domain's GPU counter block is left alone: pausing only sets a CPU flag,
     * the block already holds the right write offsets on the GPU */
    MGLXfbObject *o = xfb_current(ctx);
    if (!o->active || o->paused) XFB_ERR(GL_INVALID_OPERATION);
    o->paused = true;
    if (STATE(transform_feedback)) STATE(transform_feedback)->paused = GL_TRUE;
}

void mglResumeTransformFeedback(GLMContext ctx)
{
    /* no fold and no re-seed: the block keeps accumulating on the GPU across the pause */
    MGLXfbObject *o = xfb_current(ctx);
    if (!o->active || !o->paused) XFB_ERR(GL_INVALID_OPERATION);
    /* the program used at Begin must be the active one */
    if (xfb_active_program(ctx) != o->program) XFB_ERR(GL_INVALID_OPERATION);
    o->paused = false;
    if (STATE(transform_feedback)) STATE(transform_feedback)->paused = GL_FALSE;
}

void mglEndTransformFeedback(GLMContext ctx)   /* the geom domain resolves its GPU pack here */
{
    mgl_geom_xfb_flush(ctx);
    MGLXfbObject *o = xfb_current(ctx);
    if (!o->active) XFB_ERR(GL_INVALID_OPERATION);

    o->active = false;
    o->paused = false;
    o->ended_once = true;
    memcpy(o->draw_vertices, o->vertices, sizeof(o->draw_vertices));
    o->program = NULL;
    if (STATE(transform_feedback)) STATE(transform_feedback)->active = GL_FALSE,
                                   STATE(transform_feedback)->paused = GL_FALSE;
}

bool mgl_xfb_capturing(GLMContext ctx)
{
    if (!XFB()) return false;
    MGLXfbObject *o = xfb_current(ctx);
    return o->active && !o->paused;
}

#pragma mark - GL entry points: buffer bindings

bool mgl_xfb_validate_bind_buffer(GLMContext ctx, GLenum target, GLuint index, GLuint buffer,
                                  GLintptr offset, GLsizeiptr size, bool range)
{
    if (target != GL_TRANSFORM_FEEDBACK_BUFFER || !XFB()) return true;

    MGLXfbObject *o = xfb_current(ctx);
    if (o->active) XFB_ERR_V(GL_INVALID_OPERATION, false);
    if (index >= MGL_XFB_MAX_BUFFERS) XFB_ERR_V(GL_INVALID_VALUE, false);
    if (range && buffer && ((offset & 3) || (size & 3))) XFB_ERR_V(GL_INVALID_VALUE, false);

    o->is_base[index] = !range;
    /* Bind{Base,Range} also bind the generic TRANSFORM_FEEDBACK_BUFFER binding point */
    STATE(buffers[_TRANSFORM_FEEDBACK_BUFFER]) = buffer ? findBuffer(ctx, buffer) : NULL;
    return true;
}

static void xfb_buffer_range(GLMContext ctx, const char *func, GLuint xfb, GLuint index, GLuint buffer,
                             GLintptr offset, GLsizeiptr size, bool range)
{
    MGLXfbObject *o = xfb_obj(ctx, xfb);
    if (!o || (xfb && !findTransformFeedback(ctx, xfb))) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }
    if (index >= MGL_XFB_MAX_BUFFERS) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }
    if (o->active) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }

    Buffer *buf = NULL;
    if (buffer) {
        buf = findBuffer(ctx, buffer);
        if (!buf) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }
    }
    if (range) {
        if (offset < 0 || (buffer && size <= 0) || (offset & 3) || (size & 3)) {
            ctx->error_func(ctx, func, GL_INVALID_VALUE);
            return;
        }
    }

    BufferBaseTarget *t = &xfb_bindings(ctx, o)[index];
    if (buf) {
        t->buffer = buffer;
        t->buf = buf;
        t->offset = range ? offset : 0;
        t->size = range ? size : buf->size;
    } else {
        memset(t, 0, sizeof(*t));
    }
    o->is_base[index] = !range;
    if (o == xfb_current(ctx))
        STATE(dirty_bits) |= DIRTY_BUFFER_BASE_STATE;
    /* DSA forms bind to the object only, not to the generic binding point */
}

void mglTransformFeedbackBufferBase(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer)
{
    xfb_buffer_range(ctx, __FUNCTION__, xfb, index, buffer, 0, 0, false);
}

void mglTransformFeedbackBufferRange(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    xfb_buffer_range(ctx, __FUNCTION__, xfb, index, buffer, offset, size, true);
}

#pragma mark - GL entry points: object queries

void mglGetTransformFeedbackiv(GLMContext ctx, GLuint xfb, GLenum pname, GLint *param)
{
    MGLXfbObject *o = xfb_obj(ctx, xfb);
    if (!o || (xfb && !findTransformFeedback(ctx, xfb))) XFB_ERR(GL_INVALID_OPERATION);
    if (!param) XFB_ERR(GL_INVALID_VALUE);

    switch (pname) {
        case GL_TRANSFORM_FEEDBACK_PAUSED: *param = o->active && o->paused ? GL_TRUE : GL_FALSE; break;
        case GL_TRANSFORM_FEEDBACK_ACTIVE: *param = o->active ? GL_TRUE : GL_FALSE; break;
        default: XFB_ERR(GL_INVALID_ENUM);
    }
}

static bool xfb_indexed_value(GLMContext ctx, MGLXfbObject *o, GLenum pname, GLuint index, int64_t *out)
{
    BufferBaseTarget *t = &xfb_bindings(ctx, o)[index];
    Buffer *buf = xfb_binding_buffer(ctx, t);
    switch (pname) {
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: *out = buf ? (int64_t)t->buffer : 0; return true;
        case GL_TRANSFORM_FEEDBACK_BUFFER_START:   *out = (buf && !o->is_base[index]) ? (int64_t)t->offset : 0; return true;
        case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:    *out = (buf && !o->is_base[index]) ? (int64_t)t->size : 0; return true;
    }
    return false;
}

void mglGetTransformFeedbacki_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint *param)
{
    MGLXfbObject *o = xfb_obj(ctx, xfb);
    if (!o || (xfb && !findTransformFeedback(ctx, xfb))) XFB_ERR(GL_INVALID_OPERATION);
    if (pname != GL_TRANSFORM_FEEDBACK_BUFFER_BINDING) XFB_ERR(GL_INVALID_ENUM);
    if (index >= MGL_XFB_MAX_BUFFERS) XFB_ERR(GL_INVALID_VALUE);
    int64_t v = 0;
    xfb_indexed_value(ctx, o, pname, index, &v);
    if (param) *param = (GLint)v;
}

void mglGetTransformFeedbacki64_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint64 *param)
{
    MGLXfbObject *o = xfb_obj(ctx, xfb);
    if (!o || (xfb && !findTransformFeedback(ctx, xfb))) XFB_ERR(GL_INVALID_OPERATION);
    if (pname != GL_TRANSFORM_FEEDBACK_BUFFER_START && pname != GL_TRANSFORM_FEEDBACK_BUFFER_SIZE) XFB_ERR(GL_INVALID_ENUM);
    if (index >= MGL_XFB_MAX_BUFFERS) XFB_ERR(GL_INVALID_VALUE);
    int64_t v = 0;
    xfb_indexed_value(ctx, o, pname, index, &v);
    if (param) *param = (GLint64)v;
}

#pragma mark - DrawTransformFeedback*

static void xfb_draw_from(GLMContext ctx, const char *func, GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
    if (stream >= MGL_XFB_MAX_STREAMS) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }
    if (instancecount < 0) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }
    MGLXfbObject *o = xfb_obj(ctx, id);
    if (!o || (id && !findTransformFeedback(ctx, id))) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }
    if (!o->ended_once) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }

    uint64_t count = o->draw_vertices[stream];
    if (count > 0x7fffffff) count = 0x7fffffff;

    /* behaves like DrawArraysInstanced(mode, 0, count, instancecount); goes through the normal
     * draw path, so it validates mode and is itself captured if another TF object is active */
    mglDrawArraysInstanced(ctx, mode, 0, (GLsizei)count, instancecount);
}

void mglDrawTransformFeedback(GLMContext ctx, GLenum mode, GLuint id)
{
    xfb_draw_from(ctx, __FUNCTION__, mode, id, 0, 1);
}

void mglDrawTransformFeedbackInstanced(GLMContext ctx, GLenum mode, GLuint id, GLsizei instancecount)
{
    xfb_draw_from(ctx, __FUNCTION__, mode, id, 0, instancecount);
}

void mglDrawTransformFeedbackStream(GLMContext ctx, GLenum mode, GLuint id, GLuint stream)
{
    xfb_draw_from(ctx, __FUNCTION__, mode, id, stream, 1);
}

void mglDrawTransformFeedbackStreamInstanced(GLMContext ctx, GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
    xfb_draw_from(ctx, __FUNCTION__, mode, id, stream, instancecount);
}


#pragma mark - link: compile helpers

static bool xfb_debug(void)
{
    static int d = -1;
    if (d < 0) d = getenv("MGL_XFB_DEBUG") ? 1 : 0;
    return d == 1;
}

/* GLSL vertex source -> SPIR-V with the same glslang setup as mglCompileShader/mglLinkProgram.
 * force_version: 0 = as the source says, else compile as this (core) version. */
/* program.c: default-uniform-block packing for the capture variant's lone vertex stage */
extern uint32_t *mgl_program_pack_stage_for_msl(GLMContext ctx, Program *program,
                                                const uint32_t *in, size_t n, size_t *out_n);

static bool xfb_glsl_to_spirv(GLMContext ctx, const char *src, int force_version, uint32_t **ir, size_t *words, char **log)
{
    glslang_input_t input;
    memset(&input, 0, sizeof(input));
    initGLSLInput(ctx, GL_VERTEX_SHADER, src, &input);
    if (force_version && input.default_version < force_version) {
        input.default_version = force_version;
        input.client_version = GLSLANG_TARGET_OPENGL_450;
    }

    int version = 330;
    const char *vstr = strstr(src, "#version");
    if (vstr) sscanf(vstr, "#version %d", &version);
    int options = GLSLANG_SHADER_VULKAN_RULES_RELAXED;
    if (version < 330) options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS;
    /* match the program's own compile (shaders.c): GL assigns interface locations and
     * bindings at link time, so a shader that relies on that must compile here too — and the
     * invented locations/bindings are the same glslang would invent for the program, keeping
     * the reflection and the capture variant consistent with it */
    options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS | GLSLANG_SHADER_AUTO_MAP_BINDINGS;

    glslang_shader_t *sh = glslang_shader_create(&input);
    if (!sh) { *log = strdup("glslang_shader_create failed"); return false; }
    glslang_shader_set_options(sh, options);

    if (!glslang_shader_preprocess(sh, &input) || !glslang_shader_parse(sh, &input)) {
        const char *l = glslang_shader_get_info_log(sh);
        *log = strdup(l ? l : "compile failed");
        glslang_shader_delete(sh);
        return false;
    }

    glslang_program_t *prog = glslang_program_create();
    glslang_program_add_shader(prog, sh);
    if (!glslang_program_link(prog, GLSLANG_MSG_DEFAULT_BIT)) {
        const char *l = glslang_program_get_info_log(prog);
        *log = strdup(l ? l : "link failed");
        glslang_program_delete(prog);
        glslang_shader_delete(sh);
        return false;
    }
    /* run the I/O mapper exactly as the program's own link does (program.c): it is what
     * assigns the locations/bindings AUTO_MAP_* promise.  Without it the capture variant
     * compiles but SPIRV-Cross emits a stage_in struct whose members have no
     * [[attribute(n)]] (Metal: "invalid type of input declaration with attribute
     * 'stage_in'"), and invented bindings would not match the program's reflection */
    if (!glslang_program_map_io(prog)) {
        const char *l = glslang_program_get_info_log(prog);
        *log = strdup(l ? l : "I/O mapping failed");
        glslang_program_delete(prog);
        glslang_shader_delete(sh);
        return false;
    }
    glslang_program_SPIRV_generate(prog, GLSLANG_STAGE_VERTEX);
    *words = glslang_program_SPIRV_get_size(prog);
    *ir = malloc(*words * sizeof(uint32_t));
    if (*ir) glslang_program_SPIRV_get(prog, *ir);
    glslang_program_delete(prog);
    glslang_shader_delete(sh);
    if (!*ir) { *log = strdup("out of memory"); return false; }
    return true;
}

static void xfb_spvc_error(void *userdata, const char *error)
{
    (void)userdata;
    fprintf(stderr, "MGL xfb: SPIRV-Cross: %s\n", error);
}

#pragma mark - link: reflection

typedef struct {
    char         *name;      /* root name a varying string starts with */
    char         *expr;      /* GLSL expression for it */
    spvc_type_id  tid;       /* type (may be an array) */
    spvc_type_id  base_tid;  /* base (non array) type: member names live here */
} XfbRoot;

typedef struct {
    XfbRoot *roots;
    size_t   count, cap;
} XfbRoots;

static void xfb_add_root(XfbRoots *r, const char *name, const char *expr, spvc_type_id tid, spvc_type_id base)
{
    if (!name || !name[0]) return;
    if (r->count == r->cap) {
        r->cap = r->cap ? r->cap * 2 : 16;
        r->roots = realloc(r->roots, r->cap * sizeof(XfbRoot));
    }
    r->roots[r->count].name = strdup(name);
    r->roots[r->count].expr = strdup(expr);
    r->roots[r->count].tid = tid;
    r->roots[r->count].base_tid = base;
    r->count++;
}

static void xfb_free_roots(XfbRoots *r)
{
    for (size_t i = 0; i < r->count; i++) { free(r->roots[i].name); free(r->roots[i].expr); }
    free(r->roots);
}

/* collect output variables: plain outputs, named-instance blocks (by block type name) and the
 * members of anonymous blocks (gl_PerVertex: gl_Position, gl_PointSize, gl_ClipDistance...) */
static void xfb_collect_roots(spvc_compiler c, spvc_resources res, XfbRoots *roots)
{
    const spvc_reflected_resource *list = NULL;
    size_t n = 0;
    SpvId seen[256];
    size_t nseen = 0;

    for (int pass = 0; pass < 2; pass++) {
        const spvc_reflected_resource *items[256];
        size_t count = 0;

        if (pass == 0) {
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, &list, &n);
            for (size_t i = 0; i < n && count < 256; i++) items[count++] = &list[i];
        } else {
            const spvc_reflected_builtin_resource *bl = NULL;
            size_t bn = 0;
            spvc_resources_get_builtin_resource_list_for_type(res, SPVC_BUILTIN_RESOURCE_TYPE_STAGE_OUTPUT, &bl, &bn);
            for (size_t i = 0; i < bn && count < 256; i++) items[count++] = &bl[i].resource;
        }

        for (size_t i = 0; i < count; i++) {
            const spvc_reflected_resource *r = items[i];
            bool dup = false;
            for (size_t k = 0; k < nseen; k++) if (seen[k] == r->id) dup = true;
            if (dup || nseen >= 256) continue;
            seen[nseen++] = r->id;

            const char *var_name = spvc_compiler_get_name(c, r->id);
            spvc_type base = spvc_compiler_get_type_handle(c, r->base_type_id);
            bool is_block = spvc_type_get_basetype(base) == SPVC_BASETYPE_STRUCT &&
                            spvc_compiler_has_decoration(c, r->base_type_id, SpvDecorationBlock);

            if (is_block) {
                const char *block_name = spvc_compiler_get_name(c, r->base_type_id);
                if (var_name && var_name[0]) {
                    xfb_add_root(roots, block_name, var_name, r->type_id, r->base_type_id);
                } else {
                    unsigned nm = spvc_type_get_num_member_types(base);
                    for (unsigned m = 0; m < nm; m++) {
                        const char *mn = spvc_compiler_get_member_name(c, r->base_type_id, m);
                        spvc_type_id mt = spvc_type_get_member_type(base, m);
                        spvc_type mth = spvc_compiler_get_type_handle(c, mt);
                        xfb_add_root(roots, mn, mn, mt, spvc_type_get_base_type_id(mth));
                    }
                }
            } else {
                xfb_add_root(roots, var_name, var_name, r->type_id, r->base_type_id);
            }
        }
    }
}

static GLenum xfb_gl_type(spvc_basetype bt, unsigned rows, unsigned cols)
{
    static const GLenum f[4] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
    static const GLenum i[4] = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
    static const GLenum u[4] = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
    static const GLenum d[4] = { GL_DOUBLE, GL_DOUBLE_VEC2, GL_DOUBLE_VEC3, GL_DOUBLE_VEC4 };
    static const GLenum m[3][3] = {
        { GL_FLOAT_MAT2,   GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4 },
        { GL_FLOAT_MAT3x2, GL_FLOAT_MAT3,   GL_FLOAT_MAT3x4 },
        { GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, GL_FLOAT_MAT4 } };
    static const GLenum dm[3][3] = {
        { GL_DOUBLE_MAT2,   GL_DOUBLE_MAT2x3, GL_DOUBLE_MAT2x4 },
        { GL_DOUBLE_MAT3x2, GL_DOUBLE_MAT3,   GL_DOUBLE_MAT3x4 },
        { GL_DOUBLE_MAT4x2, GL_DOUBLE_MAT4x3, GL_DOUBLE_MAT4 } };

    if (rows < 1 || rows > 4 || cols < 1 || cols > 4) return GL_NONE;
    switch (bt) {
        case SPVC_BASETYPE_FP32:
            if (cols == 1) return f[rows - 1];
            if (rows < 2) return GL_NONE;
            return m[cols - 2][rows - 2];
        case SPVC_BASETYPE_FP64:
            if (cols == 1) return d[rows - 1];
            if (rows < 2) return GL_NONE;
            return dm[cols - 2][rows - 2];
        case SPVC_BASETYPE_INT32:  return cols == 1 ? i[rows - 1] : GL_NONE;
        case SPVC_BASETYPE_UINT32: return cols == 1 ? u[rows - 1] : GL_NONE;
        default: return GL_NONE;
    }
}

/* fill the type part of v for type `tid` with `consumed` outer array dimensions already indexed */
static const char *xfb_leaf(spvc_compiler c, spvc_type_id tid, unsigned consumed, MGLXfbVarying *v, bool *whole_array)
{
    spvc_type t = spvc_compiler_get_type_handle(c, tid);
    unsigned dims = spvc_type_get_num_array_dimensions(t);
    unsigned left = dims - consumed;

    if (left > 1) return "arrays of arrays cannot be captured";
    spvc_basetype bt = spvc_type_get_basetype(t);
    if (bt == SPVC_BASETYPE_STRUCT) return "a structure cannot be captured as a whole";

    unsigned rows = spvc_type_get_vector_size(t), cols = spvc_type_get_columns(t);
    GLenum type = xfb_gl_type(bt, rows, cols);
    if (type == GL_NONE) return "unsupported varying type";

    v->type = type;
    v->rows = rows;
    v->columns = cols;
    v->comp_kind = bt == SPVC_BASETYPE_FP32 ? GL_FLOAT : bt == SPVC_BASETYPE_INT32 ? GL_INT :
                   bt == SPVC_BASETYPE_FP64 ? GL_DOUBLE : GL_UNSIGNED_INT;
    v->comp_per_elem = rows * cols;
    *whole_array = left == 1;
    v->size = left == 1 ? (GLint)spvc_type_get_array_dimension(t, 0) : 1;
    if (v->size < 1) return "unsized array";
    /* a double is captured as its two u32 words: bit-exact GL xfb of double
     * varyings (the capture GLSL writes unpackDouble2x32 per component) */
    v->words = v->comp_per_elem * (GLuint)v->size * (bt == SPVC_BASETYPE_FP64 ? 2 : 1);
    return NULL;
}

/* resolve an API varying name against the reflected outputs */
static const char *xfb_resolve(spvc_compiler c, XfbRoots *roots, const char *name, MGLXfbVarying *v, bool *whole_array)
{
    size_t rl = strcspn(name, ".[");
    XfbRoot *root = NULL;
    for (size_t i = 0; i < roots->count; i++)
        if (strlen(roots->roots[i].name) == rl && !strncmp(roots->roots[i].name, name, rl)) { root = &roots->roots[i]; break; }
    if (!root) return "is not an output of the vertex shader";

    spvc_type_id tid = root->tid, base = root->base_tid;
    unsigned consumed = 0;
    const char *p = name + rl;

    while (*p) {
        spvc_type t = spvc_compiler_get_type_handle(c, tid);
        unsigned dims = spvc_type_get_num_array_dimensions(t);
        if (*p == '[') {
            char *end;
            long idx = strtol(p + 1, &end, 10);
            if (end == p + 1 || *end != ']') return "has a malformed array subscript";
            if (consumed >= dims) return "subscripts a non array";
            unsigned size = spvc_type_get_array_dimension(t, dims - 1 - consumed);
            if (idx < 0 || (unsigned long)idx >= size) return "has an out of range array subscript";
            consumed++;
            p = end + 1;
        } else if (*p == '.') {
            if (consumed != dims) return "selects a member of an array";
            const char *mname = p + 1;
            size_t ml = strcspn(mname, ".[");
            spvc_type bt = spvc_compiler_get_type_handle(c, base);
            if (spvc_type_get_basetype(bt) != SPVC_BASETYPE_STRUCT) return "selects a member of a non structure";
            unsigned nm = spvc_type_get_num_member_types(bt), m;
            for (m = 0; m < nm; m++) {
                const char *mn = spvc_compiler_get_member_name(c, base, m);
                if (mn && strlen(mn) == ml && !strncmp(mn, mname, ml)) break;
            }
            if (m == nm) return "selects an unknown member";
            tid = spvc_type_get_member_type(bt, m);
            base = spvc_type_get_base_type_id(spvc_compiler_get_type_handle(c, tid));
            consumed = 0;
            p = mname + ml;
        } else {
            return "is malformed";
        }
    }

    const char *err = xfb_leaf(c, tid, consumed, v, whole_array);
    if (err) return err;

    size_t el = strlen(root->expr) + strlen(name + rl) + 1;
    v->expr = malloc(el);
    snprintf(v->expr, el, "%s%s", root->expr, name + rl);
    return NULL;
}

#pragma mark - link: layout

static void xfb_fail(MGLXfbProgram *xp, const char *fmt, const char *a, const char *b)
{
    char buf[512];
    snprintf(buf, sizeof(buf), fmt, a ? a : "", b ? b : "");
    free(xp->link_log);
    xp->link_log = strdup(buf);
    xp->link_failed = true;
    fprintf(stderr, "MGL xfb: link error: %s\n", buf);
}

static MGLXfbVarying *xfb_push_var(MGLXfbProgram *xp)
{
    MGLXfbVarying *n = realloc(xp->vars, (xp->count + 1) * sizeof(MGLXfbVarying));
    if (!n) return NULL;
    xp->vars = n;
    memset(&xp->vars[xp->count], 0, sizeof(MGLXfbVarying));
    return &xp->vars[xp->count++];
}

/* API path: glTransformFeedbackVaryings list */
static bool xfb_layout_api(spvc_compiler c, XfbRoots *roots, MGLXfbProgram *xp, bool *whole_arrays)
{
    GLuint buffer = 0, offset = 0, total_components = 0, src_word = 0;
    bool separate = xp->mode == GL_SEPARATE_ATTRIBS;

    for (GLsizei i = 0; i < xp->pending_count; i++) {
        const char *name = xp->pending[i];

        for (GLsizei k = 0; k < i; k++) {
            if (!strcmp(xp->pending[k], name) && strncmp(name, "gl_SkipComponents", 17) && strcmp(name, "gl_NextBuffer")) {
                xfb_fail(xp, "transform feedback varying %s specified more than once", name, NULL);
                return false;
            }
        }

        MGLXfbVarying *v = xfb_push_var(xp);
        if (!v) return false;
        v->name = strdup(name);

        if (!strcmp(name, "gl_NextBuffer")) {
            if (separate) { xfb_fail(xp, "gl_NextBuffer is not allowed with GL_SEPARATE_ATTRIBS", NULL, NULL); return false; }
            v->type = GL_NONE; v->size = 0;
            if (offset > xp->stride[buffer]) xp->stride[buffer] = offset;
            buffer++;
            offset = 0;
            if (buffer >= MGL_XFB_MAX_BUFFERS) { xfb_fail(xp, "too many gl_NextBuffer entries", NULL, NULL); return false; }
            v->buffer = buffer;
            continue;
        }
        if (!strncmp(name, "gl_SkipComponents", 17)) {
            int nskip = atoi(name + 17);
            if (separate) { xfb_fail(xp, "%s is not allowed with GL_SEPARATE_ATTRIBS", name, NULL); return false; }
            if (nskip < 1 || nskip > 4 || name[18] != 0) { xfb_fail(xp, "%s is not a valid varying", name, NULL); return false; }
            v->type = GL_NONE; v->size = nskip;
            v->buffer = buffer;
            v->offset = offset;
            offset += 4 * (GLuint)nskip;
            total_components += (GLuint)nskip;
            xp->buffer_mask |= 1u << buffer;
            continue;
        }

        bool whole = false;
        const char *err = xfb_resolve(c, roots, name, v, &whole);
        if (err) { xfb_fail(xp, "transform feedback varying %s %s", name, err); return false; }
        whole_arrays[xp->count - 1] = whole;

        if (separate) {
            if (v->words > MGL_XFB_MAX_SEPARATE_COMPONENTS) {
                xfb_fail(xp, "transform feedback varying %s exceeds GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS", name, NULL);
                return false;
            }
            v->buffer = (GLuint)i;
            v->offset = 0;
            xp->stride[i] = v->words * 4;
        } else {
            /* GL 4.6 / ARB_gpu_shader_fp64: "each double-precision variable
             * captured must be aligned to a multiple of eight bytes relative
             * to the beginning of a vertex" (Mesa: align(xfb_offset, 2) in
             * float units).  Everything else is packed at 4-byte granularity. */
            if (v->comp_kind == GL_DOUBLE)
                offset = (offset + 7u) & ~7u;
            v->buffer = buffer;
            v->offset = offset;
            offset += v->words * 4;
            total_components += v->words;
        }
        v->src_word = src_word;
        src_word += v->words;
        xp->buffer_mask |= 1u << v->buffer;
    }

    if (!separate) {
        if (offset > xp->stride[buffer]) xp->stride[buffer] = offset;
        if (total_components > MGL_XFB_MAX_INTERLEAVED_COMPONENTS) {
            xfb_fail(xp, "too many components for GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS", NULL, NULL);
            return false;
        }
    }
    xp->record_words = src_word;
    return true;
}

/* shader path: outputs decorated with xfb_buffer / xfb_offset (layout qualifiers) */
static bool xfb_layout_shader(spvc_compiler c, spvc_resources res, MGLXfbProgram *xp, bool *whole_arrays, size_t whole_cap)
{
    const spvc_reflected_resource *list = NULL;
    size_t n = 0;
    GLuint src_word = 0;
    GLuint explicit_stride[MGL_XFB_MAX_BUFFERS] = {0};
    SpvId seen[256];
    size_t nseen = 0;

    for (int pass = 0; pass < 2; pass++) {
        const spvc_reflected_resource *items[256];
        size_t count = 0;
        if (pass == 0) {
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_STAGE_OUTPUT, &list, &n);
            for (size_t i = 0; i < n && count < 256; i++) items[count++] = &list[i];
        } else {
            const spvc_reflected_builtin_resource *bl = NULL;
            size_t bn = 0;
            spvc_resources_get_builtin_resource_list_for_type(res, SPVC_BUILTIN_RESOURCE_TYPE_STAGE_OUTPUT, &bl, &bn);
            for (size_t i = 0; i < bn && count < 256; i++) items[count++] = &bl[i].resource;
        }

        for (size_t i = 0; i < count; i++) {
            const spvc_reflected_resource *r = items[i];
            bool dup = false;
            for (size_t k = 0; k < nseen; k++) if (seen[k] == r->id) dup = true;
            if (dup || nseen >= 256) continue;
            seen[nseen++] = r->id;

            const char *var_name = spvc_compiler_get_name(c, r->id);
            spvc_type base = spvc_compiler_get_type_handle(c, r->base_type_id);
            bool is_block = spvc_type_get_basetype(base) == SPVC_BASETYPE_STRUCT &&
                            spvc_compiler_has_decoration(c, r->base_type_id, SpvDecorationBlock);
            GLuint var_buffer = spvc_compiler_has_decoration(c, r->id, SpvDecorationXfbBuffer) ?
                                spvc_compiler_get_decoration(c, r->id, SpvDecorationXfbBuffer) : 0;
            if (spvc_compiler_has_decoration(c, r->id, SpvDecorationXfbStride) && var_buffer < MGL_XFB_MAX_BUFFERS)
                explicit_stride[var_buffer] = spvc_compiler_get_decoration(c, r->id, SpvDecorationXfbStride);

            unsigned members = is_block ? spvc_type_get_num_member_types(base) : 1;
            /* an array of blocks is unrolled: SPIR-V gives the member offsets inside one
             * element but no ArrayStride for an interface block array, so the element size
             * is the extent of the members (std430 rules, which is what glslang used for the
             * offsets), rounded up to the block alignment.  Element e of `blk` then lands at
             * member_offset + e * elem_stride. */
            GLuint elems = 1, elem_stride = 0;
            if (is_block)
            {
                spvc_type vt = spvc_compiler_get_type_handle(c, r->type_id);
                unsigned dims = spvc_type_get_num_array_dimensions(vt);

                if (dims)
                {
                    if (dims != 1)
                    {
                        xfb_fail(xp, "arrays of blocks with xfb_offset may have one dimension (%s)",
                                 var_name ? var_name : "?", NULL);
                        return false;
                    }
                    elems = spvc_type_get_array_dimension(vt, 0);
                    if (spvc_compiler_has_decoration(c, r->type_id, SpvDecorationArrayStride))
                        elem_stride = spvc_compiler_get_decoration(c, r->type_id, SpvDecorationArrayStride);
                    if (!elem_stride)
                    {
                        for (unsigned mm = 0; mm < members; mm++)
                        {
                            spvc_type_id mt = spvc_type_get_member_type(base, mm);
                            MGLXfbVarying tmp;
                            bool tmp_whole = false;
                            GLuint o;

                            memset(&tmp, 0, sizeof tmp);
                            if (!spvc_compiler_has_member_decoration(c, r->base_type_id, mm, SpvDecorationOffset))
                                continue;
                            o = spvc_compiler_get_member_decoration(c, r->base_type_id, mm, SpvDecorationOffset);
                            if (xfb_leaf(c, mt, 0, &tmp, &tmp_whole) || !tmp.words) continue;   /* not capturable */
                            if (o + tmp.words * 4 > elem_stride) elem_stride = o + tmp.words * 4;
                        }
                        elem_stride = (elem_stride + 15) & ~15u;
                    }
                }
            }

            for (unsigned m = 0; m < members; m++) {
                for (GLuint e = 0; e < elems; e++) {
                bool has_off;
                GLuint off, buf = var_buffer;
                spvc_type_id tid;
                char name[256], expr[256];

                if (is_block) {
                    has_off = spvc_compiler_has_member_decoration(c, r->base_type_id, m, SpvDecorationOffset);
                    if (!has_off) continue;
                    off = spvc_compiler_get_member_decoration(c, r->base_type_id, m, SpvDecorationOffset)
                          + e * elem_stride;
                    if (spvc_compiler_has_member_decoration(c, r->base_type_id, m, SpvDecorationXfbBuffer))
                        buf = spvc_compiler_get_member_decoration(c, r->base_type_id, m, SpvDecorationXfbBuffer);
                    if (spvc_compiler_has_member_decoration(c, r->base_type_id, m, SpvDecorationXfbStride) && buf < MGL_XFB_MAX_BUFFERS)
                        explicit_stride[buf] = spvc_compiler_get_member_decoration(c, r->base_type_id, m, SpvDecorationXfbStride);
                    const char *mn = spvc_compiler_get_member_name(c, r->base_type_id, m);
                    const char *bn = spvc_compiler_get_name(c, r->base_type_id);
                    if (var_name && var_name[0]) {
                        if (elems > 1) snprintf(name, sizeof(name), "%s[%u].%s", bn, e, mn);
                        else snprintf(name, sizeof(name), "%s.%s", bn, mn);
                        if (elems > 1) snprintf(expr, sizeof(expr), "%s[%u].%s", var_name, e, mn);
                        else snprintf(expr, sizeof(expr), "%s.%s", var_name, mn);
                    } else {
                        snprintf(name, sizeof(name), "%s", mn);
                        snprintf(expr, sizeof(expr), "%s", mn);
                    }
                    tid = spvc_type_get_member_type(base, m);
                } else {
                    has_off = spvc_compiler_has_decoration(c, r->id, SpvDecorationOffset);
                    if (!has_off) continue;
                    off = spvc_compiler_get_decoration(c, r->id, SpvDecorationOffset);
                    snprintf(name, sizeof(name), "%s", var_name ? var_name : "");
                    snprintf(expr, sizeof(expr), "%s", var_name ? var_name : "");
                    tid = r->type_id;
                }

                if (buf >= MGL_XFB_MAX_BUFFERS) { xfb_fail(xp, "xfb_buffer of %s exceeds GL_MAX_TRANSFORM_FEEDBACK_BUFFERS", name, NULL); return false; }

                MGLXfbVarying *v = xfb_push_var(xp);
                if (!v) return false;
                bool whole = false;
                const char *err = xfb_leaf(c, tid, 0, v, &whole);
                v->name = strdup(name);
                if (err) { xfb_fail(xp, "xfb output %s: %s", name, err); return false; }
                if (xp->count - 1 < whole_cap) whole_arrays[xp->count - 1] = whole;
                v->expr = strdup(expr);
                v->buffer = buf;
                v->offset = off;
                if (off & 3) { xfb_fail(xp, "xfb_offset of %s is not a multiple of 4", name, NULL); return false; }
                /* GL 4.6: an offset applied to a double-precision type must
                 * be a multiple of 8 (defined behavior only then) */
                if (v->comp_kind == GL_DOUBLE && (off & 7))
                { xfb_fail(xp, "xfb_offset of %s is not a multiple of 8 as it is applied to a type that is or contains a double", name, NULL); return false; }
                v->src_word = src_word;
                src_word += v->words;
                xp->buffer_mask |= 1u << buf;
                if (off + v->words * 4 > xp->stride[buf]) xp->stride[buf] = off + v->words * 4;
                }
            }
        }
    }

    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++) {
        if (explicit_stride[b]) {
            if (explicit_stride[b] < xp->stride[b]) { xfb_fail(xp, "xfb_stride is too small for the captured outputs", NULL, NULL); return false; }
            xp->stride[b] = explicit_stride[b];
        }
    }
    xp->record_words = src_word;
    return true;
}

#pragma mark - link: capture variant

/* rename the shader's main() so a new main() can wrap it */
static char *xfb_rename_main(const char *src)
{
    size_t len = strlen(src);
    char *out = malloc(len * 2 + 64);
    if (!out) return NULL;
    size_t o = 0;
    char last_ident[64] = "";
    bool renamed = false;

    for (size_t i = 0; i < len;) {
        if (src[i] == '/' && src[i + 1] == '/') {
            while (i < len && src[i] != '\n') out[o++] = src[i++];
            continue;
        }
        if (src[i] == '/' && src[i + 1] == '*') {
            out[o++] = src[i++]; out[o++] = src[i++];
            while (i < len && !(src[i] == '*' && src[i + 1] == '/')) out[o++] = src[i++];
            if (i < len) { out[o++] = src[i++]; out[o++] = src[i++]; }
            continue;
        }
        if (isalpha((unsigned char)src[i]) || src[i] == '_') {
            size_t s = i;
            while (i < len && (isalnum((unsigned char)src[i]) || src[i] == '_')) i++;
            size_t l = i - s;
            bool is_main = l == 4 && !strncmp(src + s, "main", 4) && !strcmp(last_ident, "void");
            if (is_main) {
                size_t j = i;
                while (j < len && isspace((unsigned char)src[j])) j++;
                is_main = j < len && src[j] == '(';
            }
            if (is_main && !renamed) {
                memcpy(out + o, "mgl_xfb_original_main", 21);
                o += 21;
                renamed = true;
            } else {
                memcpy(out + o, src + s, l);
                o += l;
            }
            size_t cl = l < sizeof(last_ident) - 1 ? l : sizeof(last_ident) - 1;
            memcpy(last_ident, src + s, cl);
            last_ident[cl] = 0;
            continue;
        }
        if (!isspace((unsigned char)src[i])) last_ident[0] = 0;
        out[o++] = src[i++];
    }
    out[o] = 0;
    if (!renamed) { free(out); return NULL; }
    return out;
}

typedef struct { char *s; size_t len, cap; } XfbStr;

static void xfb_cat(XfbStr *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void xfb_cat(XfbStr *b, const char *fmt, ...)
{
    va_list ap;
    for (;;) {
        size_t room = b->cap - b->len;
        va_start(ap, fmt);
        int n = vsnprintf(b->s ? b->s + b->len : NULL, b->s ? room : 0, fmt, ap);
        va_end(ap);
        if (n < 0) return;
        if (b->s && (size_t)n < room) { b->len += (size_t)n; return; }
        size_t ncap = (b->cap ? b->cap : 256);
        while (ncap < b->len + (size_t)n + 1) ncap *= 2;
        char *ns = realloc(b->s, ncap);
        if (!ns) return;
        b->s = ns;
        b->cap = ncap;
    }
}

/* GLSL type name of a captured varying (for the fp64 lowering's local copy) */
static void xfb_glsl_type_name(MGLXfbVarying *v, char *buf, size_t n)
{
    const char *pfx;   /* scalar base + vector prefix */
    switch (v->comp_kind) {
    case GL_DOUBLE:       pfx = "d"; break;
    case GL_FLOAT:        pfx = "";  break;
    case GL_INT:          pfx = "i"; break;
    case GL_UNSIGNED_INT: pfx = "u"; break;
    case GL_BOOL:         pfx = "b"; break;
    default:              snprintf(buf, n, "vec4"); return;
    }
    char mid[48];
    if (v->columns > 1) {
        /* matrices: matN / dmatN, matCxR / dmatCxR when non-square */
        if (v->rows == v->columns) snprintf(mid, sizeof mid, "%smat%u", pfx, v->columns);
        else snprintf(mid, sizeof mid, "%smat%ux%u", pfx, v->columns, v->rows);
    } else if (v->rows > 1) {
        snprintf(mid, sizeof mid, "%svec%u", pfx, v->rows);
    } else {
        static const char *scalar[5] = { "double", "float", "int", "uint", "bool" };
        int k = !strcmp(pfx, "d") ? 0 : !strcmp(pfx, "") ? 1 : !strcmp(pfx, "i") ? 2
              : !strcmp(pfx, "u") ? 3 : 4;
        snprintf(mid, sizeof mid, "%s", scalar[k]);
    }
    if (v->size > 1) snprintf(buf, n, "%s[%d]", mid, v->size);
    else snprintf(buf, n, "%s", mid);
}

static char *xfb_capture_glsl(const char *src, MGLXfbProgram *xp, const bool *whole_arrays)
{
    char *body = strdup(src);
    if (!body) return NULL;
    /* The injected capture block uses a std430 SSBO (GLSL 430+); a shader whose
     * own "#version" directive is older (CTS's fp64 tests use 400) must have it
     * raised, because the directive wins over the compile-time default version.
     * Only modern-core syntax (>= 140, no legacy attribute/varying keywords) is
     * safe to re-version; legacy shaders keep their version and fail the
     * capture compile with a clear log instead. */
    {
        char *nl = strstr(body, "\n");
        size_t first = nl ? (size_t)(nl - body) : strlen(body);
        if (first < 64 && !strncmp(body, "#version", 8)) {
            int ver = 0; char prof[24] = "";
            sscanf(body, "#version %d %23s", &ver, prof);
            if (ver >= 140 && ver < 430) {
                const char *profile = (!strcmp(prof, "compatibility")) ? " compatibility" : " core";
                char *bumped = malloc(strlen(body) + 8);
                if (bumped) {
                    snprintf(bumped, strlen(body) + 8, "#version 460%s\n%s", profile, body + first + 1);
                    free(body);
                    body = bumped;
                }
            }
        }
    }

    /* The capture statements are appended INSIDE main's body (before its closing
     * brace) instead of wrapping it with a renamed function: a non-entry function
     * that reads uniforms gets its globals parameterized by SPIRV-Cross, and the
     * synthesized call then passes constant-buffer members into `thread&`
     * parameters - invalid MSL ("cannot bind reference in address space").
     * Inside the entry point the outputs are ordinary locals-in-scope and the
     * uniforms render as direct constant-buffer reads, like any shader. */
    XfbStr s = {0};
    xfb_cat(&s, "    uint mgl_xfb_o = ((uint(gl_VertexID) - mgl_xfb_hdr[0]) + (uint(gl_InstanceID) - mgl_xfb_hdr[2]) * mgl_xfb_hdr[1]) * mgl_xfb_hdr[3];\n");

    for (GLuint i = 0; i < xp->count; i++) {
        MGLXfbVarying *v = &xp->vars[i];
        if (!v->words) continue;
        GLuint w = v->src_word;
        /* The fp64 lowering only supports whole-variable (or single-index)
         * access to double stage outputs; reading element-wise requires a
         * Function-storage copy first ("copy it to a local first"). */
        char var[32], tyname[64];
        const char *expr;
        if (v->comp_kind == GL_DOUBLE) {
            xfb_glsl_type_name(v, tyname, sizeof tyname);
            snprintf(var, sizeof var, "mgl_xfb_v%u", i);
            xfb_cat(&s, "    %s %s = %s;\n", tyname, var, v->expr);
            expr = var;
        } else {
            expr = v->expr;
        }
        for (GLint e = 0; e < v->size; e++) {
            char elem[320];
            if (whole_arrays[i]) snprintf(elem, sizeof(elem), "(%s)[%d]", expr, e);
            else snprintf(elem, sizeof(elem), "(%s)", expr);
            for (GLuint col = 0; col < v->columns; col++) {
                for (GLuint row = 0; row < v->rows; row++) {
                    char comp[400];
                    if (v->columns > 1) snprintf(comp, sizeof(comp), "%s[%u][%u]", elem, col, row);
                    else if (v->rows > 1) snprintf(comp, sizeof(comp), "%s[%u]", elem, row);
                    else snprintf(comp, sizeof(comp), "%s", elem);
                    const char *conv = v->comp_kind == GL_FLOAT ? "floatBitsToUint" : v->comp_kind == GL_DOUBLE ? "unpackDouble2x32" : "uint";
                    if (v->comp_kind == GL_DOUBLE) {
                        /* one double component -> two u32 words, bit-exact */
                        xfb_cat(&s, "    mgl_xfb_data[mgl_xfb_o + %uu] = %s(%s).x;\n", w++, conv, comp);
                        xfb_cat(&s, "    mgl_xfb_data[mgl_xfb_o + %uu] = %s(%s).y;\n", w++, conv, comp);
                    } else {
                        xfb_cat(&s, "    mgl_xfb_data[mgl_xfb_o + %uu] = %s(%s);\n", w++, conv, comp);
                    }
                }
            }
        }
    }

    /* locate main()'s closing brace and splice the statements in front of it */
    char *m = strstr(body, "main");
    char *close = NULL;
    if (m) {
        while (*m && *m != '{') m++;
        int depth = 0;
        while (*m) {
            if (*m == '{') depth++;
            else if (*m == '}') { depth--; if (!depth) { close = m; break; } }
            m++;
        }
    }
    char *out = NULL;
    const char *body_base = body;
    if (close) {
        XfbStr b = {0};
        /* the block must be declared before main (the statements read mgl_xfb_hdr);
         * and after the shader's own #version line, which must stay first */
        size_t vlen = 0;
        if (!strncmp(body, "#version", 8)) {
            const char *vn = strchr(body, '\n');
            vlen = vn ? (size_t)(vn - body) + 1 : strlen(body);
        }
        xfb_cat(&b, "%.*slayout(std430, binding = %d) buffer MGLXfbCaptureBlock { uint mgl_xfb_hdr[%d]; uint mgl_xfb_data[]; };\n",
                (int)vlen, body, MGL_XFB_SSBO_BINDING, MGL_XFB_HEADER_WORDS);
        body += vlen;   /* the shader's own #version line was just emitted; close is
                           absolute, so close - body stays correct without shifting it */
        (void)body_base;
        if (getenv("MGL_XFB_DEBUG"))
            fprintf(stderr, "[xfb] splice: vlen=%zu bodylen=%zu closeoff=%ld stmts=%zu\n",
                    vlen, strlen(body_base), (long)(close - body_base), s.len);
        xfb_cat(&b, "%.*s\n%s}\n", (int)(close - body), body, s.s ? s.s : "");
        out = b.s;
    } else {
        /* no main found: fall back to the legacy wrapper shape */
        XfbStr b = {0};
        xfb_cat(&b, "%s\n", body);
        xfb_cat(&b, "layout(std430, binding = %d) buffer MGLXfbCaptureBlock { uint mgl_xfb_hdr[%d]; uint mgl_xfb_data[]; };\n",
                MGL_XFB_SSBO_BINDING, MGL_XFB_HEADER_WORDS);
        xfb_cat(&b, "void main()\n{\n    mgl_xfb_original_main();\n%s}\n", s.s ? s.s : "");
        out = b.s;
    }
    free(body_base);
    free(s.s);
    return out;
}

/* capture SPIR-V -> MSL with the options program.c uses, the capture SSBO pinned to
 * buffer(MGL_XFB_MSL_BUFFER_INDEX) so every other resource keeps its automatic index */
/* The capture VS is compiled as a LONE shader: GL only assigns interface
 * locations at link time (the program's GL view, assign_interface_locations),
 * and SPIRV-Cross needs them to emit [[attribute(n)]] members.  Copy the
 * linked program's locations onto the capture module's inputs by name. */
static uint32_t *xfb_patch_input_locations(GLMContext ctx, const uint32_t *ir, size_t words, size_t *out_words)
{
    size_t insert = 0;
    uint32_t bound = ir[3];

    /* find where the type definitions start: insert decorations there */
    {
        size_t k = 5;
        while (k + 1 < words) {
            uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;
            if (!wc) break;
            if (op == 19 /* SpvOpTypeVoid */ || op == 20 || op == 21 || op == 22) {
                insert = k;
                break;
            }
            k += wc;
        }
        if (!insert) return NULL;
    }

    /* collect OpName / Input variables / existing locations */
    uint32_t *nameoff = calloc(bound + 1, 4);   /* id -> offset into pool (+1, 0 = none) */
    char *pool = malloc(words * 4 + 64);
    size_t pool_len = 0;
    uint32_t *loc = calloc(bound + 1, 4);
    uint32_t *invars = malloc((bound + 1) * 4);
    size_t nin = 0;
    {
        size_t k = 5;
        while (k + 1 < words) {
            uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;
            if (!wc) break;
            if (op == 5 /* OpName */ && wc >= 3 && ir[k + 1] <= bound) {
                size_t nb = (wc - 2) * 4;
                char *dst = pool + pool_len;
                memcpy(dst, &ir[k + 2], nb);
                dst[nb - 1] = 0;
                nameoff[ir[k + 1]] = (uint32_t)(pool_len + 1);
                pool_len += nb;
            } else if (op == 71 /* OpDecorate */ && wc >= 4 && ir[k + 2] == 30 /* Location */ &&
                       ir[k + 1] <= bound) {
                loc[ir[k + 1]] = ir[k + 3] + 1;   /* biased */
            }
            k += wc;
        }
        /* second pass: global Input variables (before the first OpFunction) */
        k = insert;
        while (k + 1 < words) {
            uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;
            if (!wc) break;
            if (op == 54 /* SpvOpFunction */) break;
            if (op == 59 /* OpVariable */ && wc >= 4 && ir[k + 3] == 1 /* Input */ &&
                ir[k + 2] <= bound && nin < bound) {
                invars[nin++] = ir[k + 2];
            }
            k += wc;
        }
    }

    MGLIProgram *ip = ctx->state.program ? mgli_find(ctx, ctx->state.program->name) : NULL;
    uint32_t decs[64];
    uint32_t ndec = 0;

    for (size_t i = 0; i < nin; i++) {
        uint32_t v = invars[i];
        if (loc[v]) continue;   /* explicit location in the source */
        const char *nm = nameoff[v] ? pool + (size_t)(nameoff[v] - 1) : NULL;
        GLint l = -1;
        if (nm && ip) {
            for (int q = 0; q < ip->ninputs; q++)
                if (ip->inputs[q].location >= 0 && !strcmp(ip->inputs[q].name, nm)) {
                    l = ip->inputs[q].location;
                    break;
                }
        }
        if (l < 0) l = (GLint)i;   /* declaration order: what the GL view assigns */
        if (ndec + 4 <= 64) {
            decs[ndec++] = (4u << 16) | 71u;   /* OpDecorate, 4 words */
            decs[ndec++] = v;
            decs[ndec++] = 30;   /* Location */
            decs[ndec++] = (uint32_t)l;
        }
    }

    uint32_t *out = NULL;
    if (ndec) {
        out = malloc((words + ndec) * 4);
        if (out) {
            memcpy(out, ir, insert * 4);
            memcpy(out + insert, decs, ndec * 4);
            memcpy(out + insert + ndec, ir + insert, (words - insert) * 4);
            *out_words = words + ndec;
        }
    }
    free(nameoff); free(pool); free(loc); free(invars);
    return out;
}

static char *xfb_spirv_to_msl(GLMContext ctx, const uint32_t *ir_in, size_t words_in, const char *entry)
{
    spvc_context sctx = NULL;
    spvc_parsed_ir pir = NULL;
    spvc_compiler comp = NULL;
    spvc_compiler_options opts = NULL;
    const char *result = NULL;
    char *ret = NULL;
    uint32_t *lowered = NULL;
    uint32_t *patched = NULL;
    size_t lower_words = 0;
    const uint32_t *ir = ir_in;
    size_t words = words_in;

    /* the capture VS is a lone shader: GL locations only exist after link-time
     * assignment; copy the linked program's locations onto it by name */
    {
        size_t pw = 0;
        patched = xfb_patch_input_locations(ctx, ir, words, &pw);
        if (patched) {
            ir = patched;
            words = pw;
        }
    }

    /* the capture VS carries the same doubles as the original vertex stage:
     * lower them exactly like compile_stage does before translating to MSL
     * (MSL has no doubles; raw u64 words + soft-float calls) */
    {
        bool is64 = false;
        size_t lw = 0;
        char log[256] = {0};
        lowered = mgl_fp64_lower_module(ir, words, true, &is64, &lw, log, sizeof log);
        if (getenv("MGL_XFB_DEBUG"))
            fprintf(stderr, "MGL xfb: capture fp64 lower: is64=%d words=%zu log='%s'\n", (int)is64, lw, log);
        if (is64 && lowered) {
            lower_words = lw;
            if (getenv("MGL_XFB_DEBUG")) {
                FILE *df = fopen("/tmp/xfb_low.spv", "wb");
                if (df) { fwrite(lowered, 4, lw, df); fclose(df); }
                df = fopen("/tmp/xfb_pre.spv", "wb");
                if (df) { fwrite(ir, 4, words, df); fclose(df); }
            }
        } else {
            free(lowered);
            lowered = NULL;
        }
    }

    if (spvc_context_create(&sctx) != SPVC_SUCCESS) { free(lowered); free(patched); return NULL; }
    spvc_context_set_error_callback(sctx, xfb_spvc_error, NULL);
    {
        const uint32_t *parse_src = lowered ? lowered : ir;
        size_t parse_n = lowered ? lower_words : words;
        if (spvc_context_parse_spirv(sctx, parse_src, parse_n, &pir) != SPVC_SUCCESS) goto done;
    }
    if (spvc_context_create_compiler(sctx, SPVC_BACKEND_MSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS) goto done;
    spvc_compiler_msl_add_discrete_descriptor_set(comp, 3);
    if (spvc_compiler_create_compiler_options(comp, &opts) != SPVC_SUCCESS) goto done;
    spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE);
    spvc_compiler_options_set_uint(opts, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3, 1, 0));
    /* the capture pass runs with rasterization disabled: the vertex function must return void */
    spvc_compiler_options_set_bool(opts, SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION, SPVC_TRUE);
    if (spvc_compiler_install_compiler_options(comp, opts) != SPVC_SUCCESS) goto done;

    spvc_msl_resource_binding rb;
    spvc_msl_resource_binding_init(&rb);
    rb.stage = SpvExecutionModelVertex;
    rb.desc_set = 0;
    rb.binding = MGL_XFB_SSBO_BINDING;
    rb.msl_buffer = MGL_XFB_MSL_BUFFER_INDEX;
    rb.msl_texture = 0;
    rb.msl_sampler = 0;
    spvc_compiler_msl_add_resource_binding(comp, &rb);

    const char *cleansed = spvc_compiler_get_cleansed_entry_point_name(comp, "main", SpvExecutionModelVertex);
    spvc_compiler_rename_entry_point(comp, cleansed, entry, SpvExecutionModelVertex);

    /* GLSL identifiers that collide with MSL keywords/types must be renamed; see
     * msl_reserved.h. */
    mgl_msl_mangle_reserved(comp, ir, words);

    if (spvc_compiler_compile(comp, &result) == SPVC_SUCCESS && result) {
        /* fp64: the capture kernel carries the soft-float library; a
         * `thread const T&` parameter cannot bind the `constant`-space
         * uniform members at the call sites (Metal: "no matching function"),
         * so take them by value like every other lowered stage. */
        if (lowered)
        {
            char *fixed = mgl_fp64_fix_msl_scalar_params(result);
            if (fixed) ret = fixed;
        }
        if (!ret) ret = strdup(result);
        if (getenv("MGL_XFB_DEBUG")) {
            char path[256];
            static unsigned seq = 0;
            snprintf(path, sizeof path, "/tmp/xfb_capture_%u.metal", seq++);
            FILE *f = fopen(path, "w");
            if (f) { fputs(ret, f); fclose(f); }
        }
    }
done:
    spvc_context_destroy(sctx);
    free(lowered);
    free(patched);
    return ret;
}

static void xfb_link(GLMContext ctx, MGLXfbProgram *xp, const char *vs_src, bool api)
{
    uint32_t *ir = NULL;
    size_t words = 0;
    char *log = NULL;
    spvc_context sctx = NULL;
    spvc_parsed_ir pir = NULL;
    spvc_compiler c = NULL;
    spvc_resources res = NULL;
    XfbRoots roots = {0};
    bool *whole = NULL;
    char *glsl = NULL;

    /* reflect the ORIGINAL vertex shader (independent of the program's SPIR-V, which other
     * domains may rewrite after linking) */
    if (!xfb_glsl_to_spirv(ctx, vs_src, 0, &ir, &words, &log)) {
        xfb_fail(xp, "vertex shader recompilation failed: %s", log, NULL);
        goto done;
    }
    if (spvc_context_create(&sctx) != SPVC_SUCCESS) goto done;
    spvc_context_set_error_callback(sctx, xfb_spvc_error, NULL);
    if (spvc_context_parse_spirv(sctx, ir, words, &pir) != SPVC_SUCCESS ||
        spvc_context_create_compiler(sctx, SPVC_BACKEND_NONE, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &c) != SPVC_SUCCESS ||
        spvc_compiler_create_shader_resources(c, &res) != SPVC_SUCCESS) {
        xfb_fail(xp, "vertex shader reflection failed", NULL, NULL);
        goto done;
    }

    whole = calloc(256 + (size_t)xp->pending_count, sizeof(bool));
    if (api) {
        xfb_collect_roots(c, res, &roots);
        xp->mode = xp->pending_mode;
        if (!xfb_layout_api(c, &roots, xp, whole)) goto done;
    } else {
        xp->mode = GL_INTERLEAVED_ATTRIBS;
        if (!xfb_layout_shader(c, res, xp, whole, 256)) goto done;
        if (xp->count == 0) goto done;   /* "xfb_" appeared but nothing is captured */
        xp->from_shader = true;
    }

    for (GLuint i = 0; i < xp->count; i++) {
        GLint l = (GLint)strlen(xp->vars[i].name) + 1;
        if (l > xp->max_name_length) xp->max_name_length = l;
    }
    if (xp->record_words == 0) goto done;   /* only skips: nothing to run on the GPU */

    /* capture variant: original source + a wrapping main() that stores the outputs */
    glsl = xfb_capture_glsl(vs_src, xp, whole);
    if (!glsl) { xfb_fail(xp, "vertex shader has no main()", NULL, NULL); goto done; }
    if (xfb_debug()) fprintf(stderr, "MGL xfb: capture GLSL for program %u:\n%s\n", xp->name, glsl);

    free(ir); ir = NULL;
    free(log); log = NULL;
    if (!xfb_glsl_to_spirv(ctx, glsl, 430, &ir, &words, &log)) {
        if (getenv("MGL_SHADER_DUMP")) {   /* diagnosis: keep the capture source */
            char path[256];
            static unsigned seq = 0;
            snprintf(path, sizeof path, "%s/xfb_capture_%u.glsl", getenv("MGL_SHADER_DUMP"), seq++);
            FILE *f = fopen(path, "w");
            if (f) { fputs(glsl, f); fclose(f); }
        }
        if (getenv("MGL_SHADER_DUMP")) {
            char path[256]; static unsigned seq = 0;
            snprintf(path, sizeof path, "%s/xfb_glcapture_%u.glsl", getenv("MGL_SHADER_DUMP"), seq++);
            FILE *f = fopen(path, "w");
            if (f) { fputs(glsl, f); fclose(f); }
        }
        xfb_fail(xp, "capture variant compilation failed: %s", log, NULL);
        goto done;
    }

    /* pack loose uniforms into the default uniform block (and run the draw-params
     * patching) exactly like the normal vertex stage, see mgl_program_pack_stage_for_msl */
    {
        size_t pn = 0;
        uint32_t *prepped = mgl_program_pack_stage_for_msl(ctx, xp->program, ir, words, &pn);
        if (prepped) { free(ir); ir = prepped; words = pn; }
    }

    char entry[64];
    snprintf(entry, sizeof(entry), "mgl_xfb_vertex_%u", xp->name);
    xp->msl = xfb_spirv_to_msl(ctx, ir, words, entry);
    if (!xp->msl) { xfb_fail(xp, "capture variant MSL translation failed", NULL, NULL); goto done; }
    xp->entry = strdup(entry);
    /* SPIRV-Cross derives gl_VertexID / gl_InstanceID from [[vertex_id]] / [[instance_id]],
     * which in Metal include vertexStart / baseInstance unless it subtracts the bases itself */
    xp->vid_absolute = strstr(xp->msl, "[[base_vertex]]") == NULL;
    xp->iid_absolute = strstr(xp->msl, "[[base_instance]]") == NULL;
    if (xfb_debug()) fprintf(stderr, "MGL xfb: capture MSL:\n%s\n", xp->msl);

done:
    if (sctx) spvc_context_destroy(sctx);
    xfb_free_roots(&roots);
    free(whole);
    free(glsl);
    free(ir);
    free(log);
}

static void xfb_post_link(GLMContext ctx, Program *p)
{
    if (!XFB() || !p) return;
    Shader *vs = p->shader_slots[_VERTEX_SHADER];
    MGLXfbProgram *xp = xfb_prog(ctx, p, false);
    bool shader_xfb = vs && vs->src && strstr(vs->src, "xfb_");
    /* the in-shader xfb qualifiers sit on the OUTPUTS of the last pre-rasterization stage:
     * a geometry or tessellation-evaluation shader declaring them (possibly an interface
     * block member with xfb_buffer / xfb_offset) is captured through that stage's resolver */
    Shader *last = p->shader_slots[_GEOMETRY_SHADER] ? p->shader_slots[_GEOMETRY_SHADER] :
                   p->shader_slots[_TESS_EVALUATION_SHADER];
    bool stage_xfb = last && last->src && strstr(last->src, "xfb_");

    if (!xp && !shader_xfb && !stage_xfb) return;
    if (!xp) xp = xfb_prog(ctx, p, true);
    if (!xp) return;

    xfb_clear_linked(ctx, xp);
    xp->linked = true;
    xp->mode = xp->pending_mode;

    /* in-shader xfb qualifiers override glTransformFeedbackVaryings */
    if (shader_xfb || stage_xfb) {
        if (p->shader_slots[_GEOMETRY_SHADER]) {
            /* geometry stage outputs (incl. named interface blocks): the geometry domain
             * resolves the declared xfb_buffer / xfb_offset layout against its own table */
            mgl_geom_xfb_link(ctx, p);
            return;
        }
        if (p->shader_slots[_TESS_EVALUATION_SHADER]) {
            mgl_geom_tess_xfb_link(ctx, p);
            return;
        }
        xfb_link(ctx, xp, vs->src, false);
        if (xp->from_shader || xp->link_failed) return;
        xfb_clear_linked(ctx, xp);
        xp->linked = true;
        xp->mode = xp->pending_mode;
    }

    if (xp->pending_count == 0) return;

    /* a separable program may have no vertex stage at all: a GS-only (or TES-only)
     * program resolves its transform feedback varyings against ITS outputs, so the
     * geometry-domain linkers run before the "needs a vertex shader" failure */
    if (p->shader_slots[_GEOMETRY_SHADER]) {
        /* the geometry domain captures the geometry stage's output itself (see
         * mgl_geom_xfb_link / mgl_geom_xfb_capture); the varyings resolve against the
         * geometry outputs */
        mgl_geom_xfb_link(ctx, p);
        return;
    }
    if (p->shader_slots[_TESS_EVALUATION_SHADER]) {
        /* geom: the TES is the last pre-rasterization stage here, and the tess domain's
         * output capture is the recorded stream - the geometry domain resolves the varyings
         * against the TES outputs and packs the capture after the draw */
        mgl_geom_tess_xfb_link(ctx, p);
        return;
    }

    if (!vs || !vs->src) {
        xfb_fail(xp, "transform feedback varyings need a vertex shader", NULL, NULL);
        return;
    }
    xfb_link(ctx, xp, vs->src, true);
}

#pragma mark - capture: primitive assembly

#define XFB_RESTART 0xffffffffu

/* GL vertex order of the primitives assembled from seq[0..n) (restart markers split strips).
 * Writes k slots per primitive into out (capacity >= 3*n + 6), returns the primitive count. */
static size_t xfb_assemble(GLenum mode, bool provoking_first, const uint32_t *seq, size_t n, uint32_t *out, unsigned *k_out)
{
    unsigned k = 1;
    switch (mode) {
        case GL_POINTS: k = 1; break;
        case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
        case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: k = 2; break;
        default: k = 3; break;
    }
    *k_out = k;

    size_t np = 0, s = 0;
    while (s < n) {
        size_t e = s;
        while (e < n && seq[e] != XFB_RESTART) e++;
        const uint32_t *v = seq + s;
        size_t len = e - s;

#define EMIT1(a)        do { out[np * k] = v[a]; np++; } while (0)
#define EMIT2(a, b)     do { out[np * k] = v[a]; out[np * k + 1] = v[b]; np++; } while (0)
#define EMIT3(a, b, c)  do { out[np * k] = v[a]; out[np * k + 1] = v[b]; out[np * k + 2] = v[c]; np++; } while (0)
        switch (mode) {
            case GL_POINTS:
                for (size_t i = 0; i < len; i++) EMIT1(i);
                break;
            case GL_LINES:
                for (size_t i = 0; i + 1 < len; i += 2) EMIT2(i, i + 1);
                break;
            case GL_LINE_STRIP:
                for (size_t i = 0; i + 1 < len; i++) EMIT2(i, i + 1);
                break;
            case GL_LINE_LOOP:
                if (len >= 2) {
                    for (size_t i = 0; i + 1 < len; i++) EMIT2(i, i + 1);
                    EMIT2(len - 1, 0);
                }
                break;
            case GL_LINES_ADJACENCY:
                for (size_t i = 0; i + 3 < len; i += 4) EMIT2(i + 1, i + 2);
                break;
            case GL_LINE_STRIP_ADJACENCY:
                for (size_t i = 1; i + 2 < len; i++) EMIT2(i, i + 1);
                break;
            case GL_TRIANGLES:
                for (size_t i = 0; i + 2 < len; i += 3) EMIT3(i, i + 1, i + 2);
                break;
            case GL_TRIANGLE_STRIP:
                /* odd triangles swap two vertices to keep the winding, keeping the provoking vertex */
                for (size_t i = 0; i + 2 < len; i++) {
                    if (!(i & 1)) EMIT3(i, i + 1, i + 2);
                    else if (provoking_first) EMIT3(i, i + 2, i + 1);
                    else EMIT3(i + 1, i, i + 2);
                }
                break;
            case GL_TRIANGLE_FAN:
                for (size_t i = 1; i + 1 < len; i++) {
                    if (provoking_first) EMIT3(i, i + 1, 0);
                    else EMIT3(0, i, i + 1);
                }
                break;
            case GL_TRIANGLES_ADJACENCY:
                for (size_t i = 0; i + 5 < len; i += 6) EMIT3(i, i + 2, i + 4);
                break;
            case GL_TRIANGLE_STRIP_ADJACENCY:
                if (len >= 6) {
                    size_t ntri = (len - 4) / 2;
                    for (size_t i = 0; i < ntri; i++) {
                        if (!(i & 1)) EMIT3(2 * i, 2 * i + 2, 2 * i + 4);
                        else EMIT3(2 * i + 2, 2 * i, 2 * i + 4);
                    }
                }
                break;
        }
#undef EMIT1
#undef EMIT2
#undef EMIT3
        s = e + 1;
    }
    return np;
}

#pragma mark - capture: per draw

typedef struct {
    GLenum      mode;
    bool        indexed;
    GLint       first;
    GLsizei     count;
    GLenum      type;
    const void *indices;
    GLsizei     instances;
    GLint       basevertex;
    GLuint      baseinstance;
} XfbDraw;

static void xfb_note(GLMContext ctx, MGLXfbObject *o, GLuint stream, uint64_t written, uint64_t needed, unsigned k)
{
    o->prims_written[stream] += written;
    o->prims_needed[stream] += needed;
    o->vertices[stream] += written * k;
    XFB()->total_written[stream] += written;
    XFB()->total_needed[stream] += needed;
    xfb_note_queries_fn note = xfb_queries_note();
    if (note) note(ctx, stream, written, needed);
}

uint64_t mgl_xfb_primitives_written(GLMContext ctx, GLuint stream)
{
    return (XFB() && stream < MGL_XFB_MAX_STREAMS) ? XFB()->total_written[stream] : 0;
}

uint64_t mgl_xfb_primitives_needed(GLMContext ctx, GLuint stream)
{
    return (XFB() && stream < MGL_XFB_MAX_STREAMS) ? XFB()->total_needed[stream] : 0;
}

/* Runs transform feedback for one draw. Returns true if the regular draw must still run. */
static bool xfb_process_draw(GLMContext ctx, const XfbDraw *d)
{
    MGLXfbObject *o = xfb_current(ctx);
    struct MGLExtMetal_xfb *M = ctx->mtl_funcs.ext_xfb;
    Program *prog = o->program;
    bool keep_draw = !ctx->state.caps.rasterizer_discard &&
                     prog && prog->shader_slots[_FRAGMENT_SHADER] != NULL;

    if (!xfb_mode_compatible(o->primitive_mode, d->mode)) {
        ctx->error_func(ctx, "mglDraw (transform feedback active)", GL_INVALID_OPERATION);
        return false;
    }
    if (d->count <= 0 || d->instances <= 0) return keep_draw;

    MGLXfbProgram *xp = xfb_prog(ctx, prog, false);
    if (!xp || !xp->linked || xp->link_failed) return keep_draw;

    /* vertex sequence of one instance (absolute vertex ids, restart markers) */
    size_t n = (size_t)d->count;
    uint32_t *seq = malloc(n * sizeof(uint32_t));
    uint32_t *prims = malloc((3 * n + 6) * sizeof(uint32_t));
    if (!seq || !prims) { free(seq); free(prims); ctx->error_func(ctx, __FUNCTION__, GL_OUT_OF_MEMORY); return false; }

    int64_t vmin = INT64_MAX, vmax = INT64_MIN;
    if (d->indexed) {
        Buffer *eb = VAO() ? VAO()->element_array.buffer : NULL;
        size_t isz = d->type == GL_UNSIGNED_BYTE ? 1 : d->type == GL_UNSIGNED_SHORT ? 2 : 4;
        uintptr_t off = (uintptr_t)d->indices;
        if (!eb || !eb->data.buffer_data || off + n * isz > (size_t)eb->size) {
            free(seq); free(prims);
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return false;
        }
        const uint8_t *src = (const uint8_t *)eb->data.buffer_data + off;
        uint32_t maxval = isz == 1 ? 0xffu : isz == 2 ? 0xffffu : 0xffffffffu;
        bool restart = ctx->state.caps.primitive_restart_fixed_index || ctx->state.caps.primitive_restart;
        uint32_t restart_index = ctx->state.caps.primitive_restart_fixed_index ? maxval : STATE_VAR(primitive_restart_index);
        for (size_t i = 0; i < n; i++) {
            uint32_t idx = isz == 1 ? src[i] : isz == 2 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i];
            if (restart && idx == restart_index) { seq[i] = XFB_RESTART; continue; }
            int64_t vid = (int64_t)idx + d->basevertex;
            if (vid < 0 || vid >= (int64_t)XFB_RESTART) { seq[i] = XFB_RESTART; continue; }
            seq[i] = (uint32_t)vid;
            if (vid < vmin) vmin = vid;
            if (vid > vmax) vmax = vid;
        }
    } else {
        for (size_t i = 0; i < n; i++) seq[i] = (uint32_t)(d->first + (GLint)i);
        vmin = d->first;
        vmax = (int64_t)d->first + (int64_t)n - 1;
    }
    if (vmin > vmax) { free(seq); free(prims); return keep_draw; }

    GLint vstart = (GLint)vmin;
    GLsizei vcount = (GLsizei)(vmax - vmin + 1);
    for (size_t i = 0; i < n; i++) if (seq[i] != XFB_RESTART) seq[i] -= (uint32_t)vstart;

    unsigned k = 1;
    bool provoking_first = STATE_VAR(provoking_vertex) == GL_FIRST_VERTEX_CONVENTION;
    size_t np = xfb_assemble(d->mode, provoking_first, seq, n, prims, &k);
    uint64_t total = (uint64_t)np * (uint64_t)d->instances;

    /* how many whole primitives fit in every buffer written */
    uint64_t fit = total;
    Buffer *bufs[MGL_XFB_MAX_BUFFERS] = {0};
    GLintptr boff[MGL_XFB_MAX_BUFFERS] = {0};
    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++) {
        if (!(xp->buffer_mask & (1u << b)) || !xp->stride[b]) continue;
        GLsizeiptr size = xfb_binding_size(ctx, o, b, &bufs[b], &boff[b]);
        GLsizeiptr avail = size - o->write_offset[b];
        if (!bufs[b] || avail <= 0) { fit = 0; continue; }
        uint64_t f = (uint64_t)avail / ((uint64_t)k * xp->stride[b]);
        if (f < fit) fit = f;
    }

    uint64_t written = 0;
    bool scattered_on_gpu = false;

    if (fit && xp->record_words && xp->msl && M && M->capture) {
        /* gpuaudit: the capture pass is encoded but not waited on, and a compute kernel
         * copies the records straight into the transform feedback buffers on the same
         * queue - no CPU round trip and no per-vertex memcpy.  Everything the kernel
         * needs (primitive table, strides, varying layout) is known on the CPU.  When
         * the kernel or the layout is unsupported this falls back to the original
         * synchronous capture + CPU scatter below. */
        struct MGLExtMetal_gpuaudit *gpm = ctx->mtl_funcs.ext_gpuaudit;

        if (gpm && gpm->xfb_scatter && M->capture_async && !getenv("MGL_XFB_NO_SCATTER")) {
            void *cb = NULL, *scratch = NULL;
            const uint32_t *rec = M->capture_async(ctx, xp, vstart, vcount, d->instances, d->baseinstance, &cb, &scratch);

            if (rec && gpm->xfb_scatter(ctx, cb, scratch, xp, bufs, boff, o->write_offset,
                                        prims, np, k, (uint64_t)vcount, fit))
            {
                written = fit;
                scattered_on_gpu = true;
                if (ctx->state.ext_gpuaudit) ctx->state.ext_gpuaudit->xfb_scatters++;
            }
            else if (!rec)
            {
                fprintf(stderr, "MGL xfb: capture pass failed, nothing written\n");
            }
        }

        if (!scattered_on_gpu && !written) {
            const uint32_t *rec = M->capture(ctx, xp, vstart, vcount, d->instances, d->baseinstance);
            if (rec) {
                written = fit;
                for (uint64_t p = 0; p < written; p++) {
                    uint64_t inst = p / np, pp = p % np;
                    for (unsigned v = 0; v < k; v++) {
                        uint64_t slot = (uint64_t)prims[pp * k + v] + inst * (uint64_t)vcount;
                        const uint32_t *r = rec + slot * xp->record_words;
                        uint64_t vi = p * k + v;
                        for (GLuint i = 0; i < xp->count; i++) {
                            MGLXfbVarying *var = &xp->vars[i];
                            if (!var->words) continue;
                            Buffer *buf = bufs[var->buffer];
                            uint8_t *dst = (uint8_t *)buf->data.buffer_data + boff[var->buffer] +
                                           o->write_offset[var->buffer] + vi * xp->stride[var->buffer] + var->offset;
                            memcpy(dst, r + var->src_word, var->words * 4);
                        }
                    }
                }
            } else {
                fprintf(stderr, "MGL xfb: capture pass failed, nothing written\n");
            }
        }
    } else if (fit && xp->record_words == 0) {
        written = fit;   /* only gl_SkipComponents: nothing to store, space is consumed */
    }

    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++) {
        if (!bufs[b] || !xp->stride[b] || !written) continue;
        size_t bytes = (size_t)(written * k * xp->stride[b]);
        size_t start = (size_t)(boff[b] + o->write_offset[b]);
        if (scattered_on_gpu) {
            /* the range was written by the scatter kernel, not by the CPU: track it as a
             * GPU write so a later CPU read syncs instead of marking dirty CPU data */
            if (M && M->buffer_gpu_written) M->buffer_gpu_written(ctx, bufs[b], start, bytes);
        } else {
            bufs[b]->data.dirty_bits |= DIRTY_BUFFER_DATA;
            if (M && M->buffer_written) M->buffer_written(ctx, bufs[b], start, bytes);
        }
        o->write_offset[b] += (GLsizeiptr)bytes;
    }

    xfb_note(ctx, o, 0, written, total, k);
    free(seq);
    free(prims);
    return keep_draw;
}

#pragma mark - capture: draw entry point wrappers

#define XM (ctx->mtl_funcs.ext_xfb->orig)

static void xfb_mtlDrawArrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, false, first, count, 0, NULL, 1, 0, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawArrays(ctx, mode, first, count);
}

static void xfb_mtlDrawArraysInstanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, false, first, count, 0, NULL, instancecount, 0, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawArraysInstanced(ctx, mode, first, count, instancecount);
}

static void xfb_mtlDrawArraysInstancedBaseInstance(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, false, first, count, 0, NULL, instancecount, 0, baseinstance };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawArraysInstancedBaseInstance(ctx, mode, first, count, instancecount, baseinstance);
}

static void xfb_mtlDrawElements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, 1, 0, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElements(ctx, mode, count, type, indices);
}

static void xfb_mtlDrawRangeElements(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, 1, 0, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawRangeElements(ctx, mode, start, end, count, type, indices);
}

static void xfb_mtlDrawElementsInstanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, instancecount, 0, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElementsInstanced(ctx, mode, count, type, indices, instancecount);
}

static void xfb_mtlDrawElementsBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, 1, basevertex, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElementsBaseVertex(ctx, mode, count, type, indices, basevertex);
}

static void xfb_mtlDrawRangeElementsBaseVertex(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, 1, basevertex, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawRangeElementsBaseVertex(ctx, mode, start, end, count, type, indices, basevertex);
}

static void xfb_mtlDrawElementsInstancedBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, instancecount, basevertex, 0 };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElementsInstancedBaseVertex(ctx, mode, count, type, indices, instancecount, basevertex);
}

static void xfb_mtlDrawElementsInstancedBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, instancecount, 0, baseinstance };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElementsInstancedBaseInstance(ctx, mode, count, type, indices, instancecount, baseinstance);
}

static void xfb_mtlDrawElementsInstancedBaseVertexBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    if (mgl_xfb_capturing(ctx)) {
        XfbDraw d = { mode, true, 0, count, type, indices, instancecount, basevertex, baseinstance };
        if (!xfb_process_draw(ctx, &d)) return;
    }
    XM.mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, mode, count, type, indices, instancecount, basevertex, baseinstance);
}

/* indirect commands are read back from the DRAW_INDIRECT_BUFFER CPU copy while capturing */
static const void *xfb_indirect_ptr(GLMContext ctx, const void *indirect, size_t size)
{
    Buffer *ib = STATE(buffers[_DRAW_INDIRECT_BUFFER]);
    uintptr_t off = (uintptr_t)indirect;
    if (!ib || !ib->data.buffer_data || off + size > (size_t)ib->size) return NULL;
    return (const uint8_t *)ib->data.buffer_data + off;
}

static void xfb_mtlDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect)
{
    if (mgl_xfb_capturing(ctx)) {
        const DrawArraysIndirectCommand *c = xfb_indirect_ptr(ctx, indirect, sizeof(*c));
        if (!c) { ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
        xfb_mtlDrawArraysInstancedBaseInstance(ctx, mode, (GLint)c->first, (GLsizei)c->count, (GLsizei)c->instanceCount, c->baseInstance);
        return;
    }
    XM.mtlDrawArraysIndirect(ctx, mode, indirect);
}

static void xfb_mtlDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    if (mgl_xfb_capturing(ctx)) {
        const DrawElementsIndirectCommand *c = xfb_indirect_ptr(ctx, indirect, sizeof(*c));
        if (!c) { ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
        size_t isz = type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4;
        xfb_mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, mode, (GLsizei)c->count, type,
                (const void *)(uintptr_t)(c->first * isz), (GLsizei)c->instanceCount, c->baseVertex, c->baseInstance);
        return;
    }
    XM.mtlDrawElementsIndirect(ctx, mode, type, indirect);
}

static void xfb_mtlMultiDrawArrays(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
{
    if (mgl_xfb_capturing(ctx)) {
        for (GLsizei i = 0; i < drawcount; i++) xfb_mtlDrawArrays(ctx, mode, first[i], count[i]);
        return;
    }
    XM.mtlMultiDrawArrays(ctx, mode, first, count, drawcount);
}

static void xfb_mtlMultiDrawElements(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)
{
    if (mgl_xfb_capturing(ctx)) {
        for (GLsizei i = 0; i < drawcount; i++) xfb_mtlDrawElements(ctx, mode, count[i], type, indices[i]);
        return;
    }
    XM.mtlMultiDrawElements(ctx, mode, count, type, indices, drawcount);
}

static void xfb_mtlMultiDrawElementsBaseVertex(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)
{
    if (mgl_xfb_capturing(ctx)) {
        for (GLsizei i = 0; i < drawcount; i++) xfb_mtlDrawElementsBaseVertex(ctx, mode, count[i], type, indices[i], basevertex[i]);
        return;
    }
    XM.mtlMultiDrawElementsBaseVertex(ctx, mode, count, type, indices, drawcount, basevertex);
}

static void xfb_mtlMultiDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    if (mgl_xfb_capturing(ctx)) {
        size_t s = stride ? (size_t)stride : sizeof(DrawArraysIndirectCommand);
        for (GLsizei i = 0; i < drawcount; i++)
            xfb_mtlDrawArraysIndirect(ctx, mode, (const void *)((uintptr_t)indirect + i * s));
        return;
    }
    XM.mtlMultiDrawArraysIndirect(ctx, mode, indirect, drawcount, stride);
}

static void xfb_mtlMultiDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    if (mgl_xfb_capturing(ctx)) {
        size_t s = stride ? (size_t)stride : sizeof(DrawElementsIndirectCommand);
        for (GLsizei i = 0; i < drawcount; i++)
            xfb_mtlDrawElementsIndirect(ctx, mode, type, (const void *)((uintptr_t)indirect + i * s));
        return;
    }
    XM.mtlMultiDrawElementsIndirect(ctx, mode, type, indirect, drawcount, stride);
}

void mgl_xfb_install_draw_hooks(GLMContext ctx)
{
    struct MGLExtMetal_xfb *m = ctx->mtl_funcs.ext_xfb;
    if (!m) return;
    m->orig = ctx->mtl_funcs;

#define WRAP(f) do { if (ctx->mtl_funcs.f) ctx->mtl_funcs.f = xfb_##f; } while (0)
    WRAP(mtlDrawArrays);
    WRAP(mtlDrawElements);
    WRAP(mtlDrawRangeElements);
    WRAP(mtlDrawArraysInstanced);
    WRAP(mtlDrawElementsInstanced);
    WRAP(mtlDrawElementsBaseVertex);
    WRAP(mtlDrawRangeElementsBaseVertex);
    WRAP(mtlDrawElementsInstancedBaseVertex);
    WRAP(mtlDrawArraysIndirect);
    WRAP(mtlDrawElementsIndirect);
    WRAP(mtlDrawArraysInstancedBaseInstance);
    WRAP(mtlDrawElementsInstancedBaseInstance);
    WRAP(mtlDrawElementsInstancedBaseVertexBaseInstance);
    WRAP(mtlMultiDrawArrays);
    WRAP(mtlMultiDrawElements);
    WRAP(mtlMultiDrawElementsBaseVertex);
    WRAP(mtlMultiDrawArraysIndirect);
    WRAP(mtlMultiDrawElementsIndirect);
#undef WRAP
}

#pragma mark - glGet

static bool xfb_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data)
{
    if (pname != GL_TRANSFORM_FEEDBACK_BUFFER_BINDING && pname != GL_TRANSFORM_FEEDBACK_BUFFER_START &&
        pname != GL_TRANSFORM_FEEDBACK_BUFFER_SIZE)
        return false;
    if (!XFB()) return false;
    if (index >= MGL_XFB_MAX_BUFFERS) {
        ctx->error_func(ctx, "mglGet*i_v", GL_INVALID_VALUE);
        return true;
    }
    int64_t v = 0;
    xfb_indexed_value(ctx, xfb_current(ctx), pname, index, &v);
    mgl_get_store(type, data, 0, (double)v);
    return true;
}

bool mgl_ext_get_xfb(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    if (!XFB()) return false;
    MGLXfbObject *o = xfb_current(ctx);
    double v;

    switch (pname) {
        case GL_TRANSFORM_FEEDBACK_BINDING:
            v = STATE(transform_feedback) ? STATE(transform_feedback)->name : 0; break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: {
            Buffer *b = STATE(buffers[_TRANSFORM_FEEDBACK_BUFFER]);
            v = (b && findBuffer(ctx, b->name) == b) ? b->name : 0;
            break;
        }
        case GL_TRANSFORM_FEEDBACK_ACTIVE:  v = o->active; break;
        case GL_TRANSFORM_FEEDBACK_PAUSED:  v = o->active && o->paused; break;
        case GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: v = MGL_XFB_MAX_INTERLEAVED_COMPONENTS; break;
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS:       v = MGL_XFB_MAX_SEPARATE_ATTRIBS; break;
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS:    v = MGL_XFB_MAX_SEPARATE_COMPONENTS; break;
        case GL_MAX_TRANSFORM_FEEDBACK_BUFFERS:                v = MGL_XFB_MAX_BUFFERS; break;
        case GL_MAX_VERTEX_STREAMS:                            v = MGL_XFB_MAX_STREAMS; break;
        default:
            return false;
    }
    mgl_get_store(type, data, 0, v);
    return true;
}

#pragma mark - domain 'xfb' hooks

void mgl_ext_init_xfb(GLMContext ctx)
{
    if (!ctx->state.ext_xfb) {
        ctx->state.ext_xfb = calloc(1, sizeof(struct MGLExtState_xfb));
        if (!ctx->state.ext_xfb) return;
    }
    /* glm_context.c does not create the transform feedback name table; an empty table makes
     * insertHashElement loop forever and getNewName hand out 0 */
    if (STATE(transform_feedback_table).size == 0)
        initHashTable(&STATE(transform_feedback_table), 32);

    XFB()->default_obj.name = 0;
    XFB()->default_obj.created = true;
    XFB()->default_obj.primitive_mode = GL_POINTS;

    mgl_ext_register_post_link(xfb_post_link);
    mgl_ext_register_get_indexed(xfb_get_indexed);
}
