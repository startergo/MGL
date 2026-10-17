/*
 * draw_ext.c
 * MGL — GL 4.6 indirect-count draws, 'buffers' domain state, glGet hooks and
 *       CPU/GPU buffer synchronisation helpers.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_buffers.h"
#include "mgl_buffers_err.h"

#define EXT() (ctx->state.ext_buffers)
#define MTLX() (ctx->mtl_funcs.ext_buffers)

#pragma mark - GL 4.6 indirect count draws

void mglMultiDrawArraysIndirectCount(GLMContext ctx, GLenum mode, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
    mgl_buffers_multi_draw_indirect(ctx, mode, 0, indirect, maxdrawcount, stride, true, drawcount);
}

void mglMultiDrawElementsIndirectCount(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
    mgl_buffers_multi_draw_indirect(ctx, mode, type, indirect, maxdrawcount, stride, true, drawcount);
}

#pragma mark - GPU usage tracking

static inline uint32_t hash_ptr(const void *p)
{
    uintptr_t v = (uintptr_t)p;
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL; v ^= v >> 33;
    return (uint32_t)v;
}

#define TRACK_TOMBSTONE ((Buffer *)1)

static void track_grow(GLMContext ctx)
{
    struct MGLExtState_buffers *st = EXT();
    uint32_t old_cap = st->track_cap;
    MGLBufferTrack *old = st->track;

    st->track_cap = old_cap ? old_cap * 2 : 256;
    st->track = (MGLBufferTrack *)calloc(st->track_cap, sizeof(MGLBufferTrack));
    st->track_used = 0;

    for (uint32_t i = 0; i < old_cap; i++)
    {
        if (old[i].buf && old[i].buf != TRACK_TOMBSTONE)
        {
            MGLBufferTrack *t = mgl_buffers_track(ctx, old[i].buf, true);
            t->read_serial = old[i].read_serial;
            t->write_serial = old[i].write_serial;
        }
    }

    free(old);
}

MGLBufferTrack *mgl_buffers_track(GLMContext ctx, Buffer *buf, bool create)
{
    struct MGLExtState_buffers *st = EXT();

    if (!st || !buf)
        return NULL;

    if (create && (st->track_cap == 0 || (st->track_used + 1) * 4 >= st->track_cap * 3))
        track_grow(ctx);

    if (st->track_cap == 0)
        return NULL;

    uint32_t mask = st->track_cap - 1;
    uint32_t i = hash_ptr(buf) & mask;
    MGLBufferTrack *first_free = NULL;

    for (uint32_t n = 0; n < st->track_cap; n++, i = (i + 1) & mask)
    {
        MGLBufferTrack *t = &st->track[i];

        if (t->buf == buf)
            return t;

        if (t->buf == TRACK_TOMBSTONE)
        {
            if (!first_free) first_free = t;
            continue;
        }

        if (t->buf == NULL)
        {
            if (!create)
                return NULL;

            if (!first_free)
            {
                first_free = t;
                st->track_used++;
            }

            first_free->buf = buf;
            first_free->read_serial = first_free->write_serial = 0;
            return first_free;
        }
    }

    if (create && first_free)
    {
        first_free->buf = buf;
        first_free->read_serial = first_free->write_serial = 0;
        return first_free;
    }

    return NULL;
}

void mgl_buffers_forget(GLMContext ctx, Buffer *buf)
{
    MGLBufferTrack *t = mgl_buffers_track(ctx, buf, false);

    if (t)
    {
        t->buf = TRACK_TOMBSTONE;
        t->read_serial = t->write_serial = 0;
    }
}

void mgl_buffers_mark_gpu_use(GLMContext ctx, Buffer *buf, bool gpu_writes)
{
    if (!buf || !MTLX() || !MTLX()->current_serial)
        return;

    MGLBufferTrack *t = mgl_buffers_track(ctx, buf, true);
    if (!t)
        return;

    uint64_t serial = MTLX()->current_serial(ctx);

    t->read_serial = serial;
    if (gpu_writes)
        t->write_serial = serial;
}

/* buffers the GPU may write through bindings owned by other domains */
static bool bound_for_foreign_gpu_write(GLMContext ctx, Buffer *buf)
{
    if (STATE(buffers[_QUERY_BUFFER]) == buf || STATE(buffers[_PIXEL_PACK_BUFFER]) == buf)
        return true;

    for (int i = 0; i < MAX_BINDABLE_BUFFERS; i++)
        if (STATE(buffer_base[_TRANSFORM_FEEDBACK_BUFFER]).buffers[i].buf == buf)
            return true;

    return false;
}

void mgl_buffers_prepare_cpu_read(GLMContext ctx, Buffer *buf)
{
    if (!buf || !MTLX() || !MTLX()->completed_serial)
        return;

    uint64_t completed = MTLX()->completed_serial(ctx);
    MGLBufferTrack *t = mgl_buffers_track(ctx, buf, false);
    uint64_t need = t ? t->write_serial : 0;

    if (need == 0 && buf->data.mtl_data && bound_for_foreign_gpu_write(ctx, buf))
        need = MTLX()->current_serial(ctx);   // conservatively: everything queued so far

    if (need > completed)
        MTLX()->make_cpu_visible(ctx, buf, need);
}

void mgl_buffers_prepare_cpu_write(GLMContext ctx, Buffer *buf, bool whole_buffer)
{
    if (!buf || !MTLX() || !MTLX()->completed_serial)
        return;

    MGLBufferTrack *t = mgl_buffers_track(ctx, buf, false);
    if (!t)
        return;

    uint64_t completed = MTLX()->completed_serial(ctx);
    bool pending_read = t->read_serial > completed;
    bool pending_write = t->write_serial > completed;

    if (!pending_read && !pending_write)
        return;

    // queued GPU work keeps the old storage alive (Metal retains it); give the
    // buffer fresh storage instead of stalling, unless its address must stay put
    bool can_rename = buf->data.mtl_data &&
                      !(buf->storage_flags & GL_CLIENT_STORAGE_BIT) &&
                      !buf->mapped &&
                      MTLX()->rename_buffer &&
                      (whole_buffer || !pending_write);

    if (can_rename && MTLX()->rename_buffer(ctx, buf))
    {
        t->read_serial = t->write_serial = 0;
        return;
    }

    MTLX()->wait_serial(ctx, t->read_serial > t->write_serial ? t->read_serial : t->write_serial);
}

uint32_t *mgl_buffers_range_mask(GLMContext ctx, int slot)
{
    return &EXT()->range_mask[slot & 3];
}

#pragma mark - sync object registry

bool mgl_buffers_sync_valid(GLMContext ctx, Sync *sync)
{
    struct MGLExtState_buffers *st = EXT();

    if (!sync || !st)
        return false;

    for (uint32_t i = 0; i < st->sync_count; i++)
        if (st->syncs[i] == sync)
            return true;

    return false;
}

void mgl_buffers_sync_add(GLMContext ctx, Sync *sync)
{
    struct MGLExtState_buffers *st = EXT();

    if (st->sync_count == st->sync_cap)
    {
        uint32_t cap = st->sync_cap ? st->sync_cap * 2 : 16;
        Sync **n = (Sync **)realloc(st->syncs, cap * sizeof(Sync *));
        if (!n) return;
        st->syncs = n;
        st->sync_cap = cap;
    }

    st->syncs[st->sync_count++] = sync;
}

void mgl_buffers_sync_remove(GLMContext ctx, Sync *sync)
{
    struct MGLExtState_buffers *st = EXT();

    for (uint32_t i = 0; i < st->sync_count; i++)
    {
        if (st->syncs[i] == sync)
        {
            st->syncs[i] = st->syncs[--st->sync_count];
            return;
        }
    }
}

#pragma mark - domain 'buffers' hooks

static bool buffers_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data);

void mgl_ext_init_buffers(GLMContext ctx)
{
    if (ctx->state.ext_buffers == NULL)
    {
        ctx->state.ext_buffers = (struct MGLExtState_buffers *)calloc(1, sizeof(struct MGLExtState_buffers));
        assert(ctx->state.ext_buffers);
    }

    mgl_ext_register_get_indexed(buffers_get_indexed);
}

static GLuint bufname(Buffer *b)
{
    return b ? b->name : 0;
}

bool mgl_ext_get_buffers(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    Buffer *b;

    switch(pname)
    {
        case GL_COPY_READ_BUFFER_BINDING:        b = STATE(buffers[_COPY_READ_BUFFER]); break;
        case GL_COPY_WRITE_BUFFER_BINDING:       b = STATE(buffers[_COPY_WRITE_BUFFER]); break;
        case GL_DRAW_INDIRECT_BUFFER_BINDING:    b = STATE(buffers[_DRAW_INDIRECT_BUFFER]); break;
        case GL_QUERY_BUFFER_BINDING:            b = STATE(buffers[_QUERY_BUFFER]); break;
        case GL_ATOMIC_COUNTER_BUFFER_BINDING:   b = STATE(buffers[_ATOMIC_COUNTER_BUFFER]); break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: b = STATE(buffers[_TRANSFORM_FEEDBACK_BUFFER]); break;
        case GL_TEXTURE_BUFFER_BINDING:          b = STATE(buffers[_TEXTURE_BUFFER]); break;
        case GL_PARAMETER_BUFFER_BINDING:        b = EXT() ? EXT()->parameter_buffer : NULL; break;
        case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: mgl_get_store(type, data, 0, 8); return true;
        case GL_MAX_TRANSFORM_FEEDBACK_BUFFERS:  mgl_get_store(type, data, 0, 4); return true;
        default:
            return false;
    }

    mgl_get_store(type, data, 0, bufname(b));
    return true;
}

static bool buffers_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data)
{
    int base, slot, what;
    GLuint max;

    switch(pname)
    {
        case GL_UNIFORM_BUFFER_BINDING:  base = _UNIFORM_BUFFER; slot = 0; what = 0; break;
        case GL_UNIFORM_BUFFER_START:    base = _UNIFORM_BUFFER; slot = 0; what = 1; break;
        case GL_UNIFORM_BUFFER_SIZE:     base = _UNIFORM_BUFFER; slot = 0; what = 2; break;
        case GL_SHADER_STORAGE_BUFFER_BINDING: base = _SHADER_STORAGE_BUFFER; slot = 1; what = 0; break;
        case GL_SHADER_STORAGE_BUFFER_START:   base = _SHADER_STORAGE_BUFFER; slot = 1; what = 1; break;
        case GL_SHADER_STORAGE_BUFFER_SIZE:    base = _SHADER_STORAGE_BUFFER; slot = 1; what = 2; break;
        case GL_ATOMIC_COUNTER_BUFFER_BINDING: base = _ATOMIC_COUNTER_BUFFER; slot = 2; what = 0; break;
        case GL_ATOMIC_COUNTER_BUFFER_START:   base = _ATOMIC_COUNTER_BUFFER; slot = 2; what = 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_SIZE:    base = _ATOMIC_COUNTER_BUFFER; slot = 2; what = 2; break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: base = _TRANSFORM_FEEDBACK_BUFFER; slot = 3; what = 0; break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_START:   base = _TRANSFORM_FEEDBACK_BUFFER; slot = 3; what = 1; break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE:    base = _TRANSFORM_FEEDBACK_BUFFER; slot = 3; what = 2; break;
        default:
            return false;
    }

    max = (slot == 2) ? 8 : (slot == 3) ? 4 : MAX_BINDABLE_BUFFERS;
    if (index >= max)
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return true;
    }

    BufferBaseTarget *bt = &STATE(buffer_base[base]).buffers[index];
    bool is_range = EXT() && (EXT()->range_mask[slot] & (1u << index));
    double v;

    switch(what)
    {
        case 0:  v = bt->buf ? bt->buf->name : 0; break;
        case 1:  v = (bt->buf && is_range) ? (double)bt->offset : 0; break;
        default: v = (bt->buf && is_range) ? (double)bt->size : 0; break;
    }

    mgl_get_store(type, data, 0, v);
    return true;
}
