/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * vertex_buffers.c
 * MGL
 *
 * Vertex buffer bindings of vertex array objects (GL 4.6 §10.3.1 BindVertexBuffer(s),
 * VertexArrayVertexBuffer(s)).
 */

#include <strings.h>

#include "glm_context.h"
#include "mgl_ext_vattrib.h"

#define VA_ERROR(_e_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return; } while (0)

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);
extern Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer);
extern VertexArray *lookupVAO(GLMContext ctx, GLuint vao);

/* resolve a buffer name for a vertex buffer binding; false if not a valid name */
static bool resolveBuffer(GLMContext ctx, GLuint buffer, Buffer **out)
{
    if (buffer == 0)
    {
        *out = NULL;
        return true;
    }

    Buffer *buf = findBuffer(ctx, buffer);

    // a name returned by GenBuffers that has not been bound yet: create its object now
    if (buf == NULL && buffer < STATE(buffer_table).current_name)
        buf = getBuffer(ctx, GL_ARRAY_BUFFER, buffer);

    *out = buf;

    return buf != NULL;
}

static void setBinding(GLMContext ctx, VertexArray *vao, GLuint bindingindex, Buffer *buf, GLintptr offset, GLsizei stride)
{
    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVertexBinding *b = &va->binding[bindingindex];

    b->buffer = buf;
    b->buffer_name = buf ? buf->name : 0;
    b->offset = offset;
    b->stride = stride;

    for (GLuint i = 0; i < MAX_ATTRIBS; i++)
        if (va->fmt[i].binding == bindingindex)
            mgl_va_sync_attrib(vao, i);

    vao->dirty_bits |= DIRTY_VAO_BUFFER_BASE | DIRTY_VAO_ATTRIB;

    if (vao == ctx->state.vao)
        ctx->state.dirty_bits |= DIRTY_VAO;
}

static void bindVertexBufferTo(GLMContext ctx, VertexArray *vao, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    if (bindingindex >= MGL_VA_MAX_BINDINGS) VA_ERROR(GL_INVALID_VALUE);
    if (offset < 0 || stride < 0 || stride > MGL_VA_MAX_STRIDE) VA_ERROR(GL_INVALID_VALUE);

    Buffer *buf;
    if (!resolveBuffer(ctx, buffer, &buf)) VA_ERROR(GL_INVALID_OPERATION);

    setBinding(ctx, vao, bindingindex, buf, offset, stride);
}

static void bindVertexBuffersTo(GLMContext ctx, VertexArray *vao, GLuint first, GLsizei count,
                                const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides)
{
    if (count < 0) VA_ERROR(GL_INVALID_VALUE);
    if ((GLuint64)first + (GLuint64)count > MGL_VA_MAX_BINDINGS) VA_ERROR(GL_INVALID_OPERATION);

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint bindingindex = first + i;

        if (buffers == NULL)
        {
            // unbind every binding in the range, offsets/strides are ignored
            setBinding(ctx, vao, bindingindex, NULL, 0, 16);
            continue;
        }

        // an error for one binding leaves it unchanged; the others are still updated
        if (offsets[i] < 0 || strides[i] < 0 || strides[i] > MGL_VA_MAX_STRIDE)
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_VALUE);
            continue;
        }

        Buffer *buf;
        if (!resolveBuffer(ctx, buffers[i], &buf))
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            continue;
        }

        setBinding(ctx, vao, bindingindex, buf, offsets[i], strides[i]);
    }
}

/* kept for callers of the old helper: vaobj 0 means the bound VAO */
bool bindVertexBuffer(GLMContext ctx, GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    VertexArray *vao = vaobj ? lookupVAO(ctx, vaobj) : ctx->state.vao;

    if (vao == NULL)
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    GLenum before = ctx->state.error;
    bindVertexBufferTo(ctx, vao, bindingindex, buffer, offset, stride);

    return ctx->state.error == before;
}

void mglBindVertexBuffer(GLMContext ctx, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindVertexBufferTo(ctx, VAO(), bindingindex, buffer, offset, stride);
}

void mglBindVertexBuffers(GLMContext ctx, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindVertexBuffersTo(ctx, VAO(), first, count, buffers, offsets, strides);
}

void mglVertexArrayVertexBuffer(GLMContext ctx, GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride)
{
    VertexArray *vao = lookupVAO(ctx, vaobj);
    if (vao == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindVertexBufferTo(ctx, vao, bindingindex, buffer, offset, stride);
}

void mglVertexArrayVertexBuffers(GLMContext ctx, GLuint vaobj, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizei *strides)
{
    VertexArray *vao = lookupVAO(ctx, vaobj);
    if (vao == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindVertexBuffersTo(ctx, vao, first, count, buffers, offsets, strides);
}
