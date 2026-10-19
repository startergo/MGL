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
 * vertex_arrays.c
 * MGL
 *
 * Vertex array objects and generic attribute formats (GL 4.6 §10.3).
 * Vertex buffer bindings are in vertex_buffers.c.
 */

#include <stdlib.h>
#include <strings.h>

#include "glm_context.h"
#include "vertex_arrays.h"
#include "mgl_ext_vattrib.h"

#define VA_ERROR(_e_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return; } while (0)

Buffer *findBuffer(GLMContext ctx, GLuint buffer);

static inline GLuint maxAttribs(GLMContext ctx)
{
    GLuint m = ctx->state.max_vertex_attribs;
    if (m == 0 || m > MAX_ATTRIBS) m = MAX_ATTRIBS;
    return m;
}

#pragma mark - type sizes

GLsizei typeSize(GLenum type)
{
    switch(type)
    {
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return 1;

        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
            return 2;

        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_FLOAT:
        case GL_FIXED:
            return 4;

        case GL_DOUBLE:
            return 8;

        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;

        default:
            return 0;
    }
}

/* size in bytes of one attribute element */
GLsizei mgl_va_element_size(GLenum type, GLint size)
{
    switch (type)
    {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            return 4;
    }

    return typeSize(type) * (size == GL_BGRA ? 4 : size);
}

GLsizei genStrideFromTypeSize(GLenum type, GLint size)
{
    return mgl_va_element_size(type, size);
}

#pragma mark - legacy VertexArray.attrib[] mirror

void mgl_va_sync_attrib(VertexArray *vao, GLuint attrib)
{
    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVertexAttribFormat *f = &va->fmt[attrib];
    MGLVertexBinding *b = &va->binding[f->binding];

    vao->attrib[attrib].buffer = b->buffer;
    vao->attrib[attrib].size = (f->size == GL_BGRA) ? 4 : f->size;
    vao->attrib[attrib].type = f->type;
    vao->attrib[attrib].normalized = f->normalized;
    vao->attrib[attrib].stride = b->stride;
    vao->attrib[attrib].divisor = b->divisor;
    vao->attrib[attrib].relativeoffset = f->relativeoffset;
    vao->attrib[attrib].buffer_bindingindex = f->binding;
}

void mgl_va_sync_all(VertexArray *vao)
{
    for (GLuint i = 0; i < MAX_ATTRIBS; i++)
        mgl_va_sync_attrib(vao, i);
}

/* a VAO changed: re-derive the Metal vertex descriptor if it is the bound one */
static void vaoChanged(GLMContext ctx, VertexArray *vao)
{
    vao->dirty_bits |= DIRTY_VAO_ATTRIB | DIRTY_VAO_BUFFER_BASE;

    if (vao == ctx->state.vao)
        ctx->state.dirty_bits |= DIRTY_VAO;
}

#pragma mark - VAO objects

VertexArray *newVAO(GLMContext ctx, GLuint vao)
{
    MGLVertexArrayFull *ptr;

    ptr = (MGLVertexArrayFull *)calloc(1, sizeof(MGLVertexArrayFull));
    if (!ptr)
        return NULL;

    ptr->base.name = vao;

    for (int i = 0; i < MAX_ATTRIBS; i++)
    {
        ptr->fmt[i].size = 4;
        ptr->fmt[i].type = GL_FLOAT;
        ptr->fmt[i].binding = i;
    }

    for (int i = 0; i < MGL_VA_MAX_BINDINGS; i++)
    {
        ptr->binding[i].stride = 16;
    }

    mgl_va_sync_all(&ptr->base);

    return &ptr->base;
}

static void markGenerated(GLMContext ctx, GLuint name, bool gen)
{
    struct MGLExtState_vattrib *s = ctx->state.ext_vattrib;

    if (!s) return;

    if (name >= s->gen_names_cap)
    {
        if (!gen) return;

        size_t cap = s->gen_names_cap ? s->gen_names_cap : 64;
        while (cap <= name) cap *= 2;

        uint8_t *p = (uint8_t *)realloc(s->gen_names, cap);
        if (!p) return;

        bzero(p + s->gen_names_cap, cap - s->gen_names_cap);
        s->gen_names = p;
        s->gen_names_cap = cap;
    }

    s->gen_names[name] = gen ? 1 : 0;
}

static bool isGenerated(GLMContext ctx, GLuint name)
{
    struct MGLExtState_vattrib *s = ctx->state.ext_vattrib;

    if (!s || name >= s->gen_names_cap)
        return false;

    return s->gen_names[name] != 0;
}

/* an existing VAO object (created by CreateVertexArrays or a BindVertexArray), or NULL */
VertexArray *lookupVAO(GLMContext ctx, GLuint vao)
{
    if (vao == 0)
        return NULL;

    return (VertexArray *)searchHashTable(&STATE(vao_table), vao);
}

/* return the VAO object, creating the object of a generated name */
VertexArray *getVAO(GLMContext ctx, GLuint vao)
{
    VertexArray *ptr;

    ptr = lookupVAO(ctx, vao);

    if (!ptr)
    {
        ptr = newVAO(ctx, vao);

        if (ptr)
            insertHashElement(&STATE(vao_table), vao, ptr);
    }

    return ptr;
}

int isVAO(GLMContext ctx, GLuint vao)
{
    return lookupVAO(ctx, vao) != NULL;
}

void mglGenVertexArrays(GLMContext ctx, GLsizei n, GLuint *arrays)
{
    if (n < 0) VA_ERROR(GL_INVALID_VALUE);

    while(n--)
    {
        GLuint name = getNewName(&STATE(vao_table));

        markGenerated(ctx, name, true);

        *arrays++ = name;
    }
}

void mglCreateVertexArrays(GLMContext ctx, GLsizei n, GLuint *arrays)
{
    if (n < 0) VA_ERROR(GL_INVALID_VALUE);

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name = getNewName(&STATE(vao_table));

        markGenerated(ctx, name, true);

        if (getVAO(ctx, name) == NULL) VA_ERROR(GL_OUT_OF_MEMORY);

        arrays[i] = name;
    }
}

void mglBindVertexArray(GLMContext ctx, GLuint array)
{
    VertexArray *ptr;

    if (array == 0)
    {
        ptr = NULL;
    }
    else
    {
        ptr = lookupVAO(ctx, array);

        if (ptr == NULL)
        {
            // GL_INVALID_OPERATION if array is not zero or a name returned by glGenVertexArrays
            if (!isGenerated(ctx, array)) VA_ERROR(GL_INVALID_OPERATION);

            ptr = getVAO(ctx, array);
            if (ptr == NULL) VA_ERROR(GL_OUT_OF_MEMORY);
        }
    }

    STATE(var.vertex_array_binding) = array;
    STATE(var.element_array_buffer_binding) = (ptr && ptr->element_array.buffer) ? ptr->element_array.buffer->name : 0;
    STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = ptr ? ptr->element_array.buffer : NULL;

    if (STATE(vao) != ptr)
    {
        STATE(vao) = ptr;
        STATE(dirty_bits) |= DIRTY_VAO;
    }
}

void mglDeleteVertexArrays(GLMContext ctx, GLsizei n, const GLuint *arrays)
{
    if (n < 0) VA_ERROR(GL_INVALID_VALUE);

    while(n--)
    {
        GLuint vao = *arrays++;

        if (vao == 0)
            continue;   // silently ignored

        VertexArray *ptr = lookupVAO(ctx, vao);

        if (ptr)
        {
            // the bound VAO reverts to zero
            if (ptr == STATE(vao))
                mglBindVertexArray(ctx, 0);

            deleteHashElement(&STATE(vao_table), vao);

            free(ptr);
        }

        markGenerated(ctx, vao, false);
    }
}

GLboolean mglIsVertexArray(GLMContext ctx, GLuint array)
{
    return isVAO(ctx, array) ? GL_TRUE : GL_FALSE;
}

/* glBindBuffer(GL_ELEMENT_ARRAY_BUFFER) — the element array binding is VAO state */
void mgl_va_element_buffer_bound(GLMContext ctx, Buffer *buf)
{
    if (ctx->state.vao)
        ctx->state.vao->element_array.buffer = buf;

    STATE(var.element_array_buffer_binding) = buf ? buf->name : 0;
}

#pragma mark - attribute formats

enum { FMT_FLOAT, FMT_INTEGER, FMT_LONG };

/* validate size / type / normalized of a *Pointer or *Format call; returns 0 or the GL error */
static GLenum checkFormat(GLint size, GLenum type, GLboolean normalized, int kind)
{
    switch (kind)
    {
        case FMT_INTEGER:
            switch (type)
            {
                case GL_BYTE: case GL_UNSIGNED_BYTE: case GL_SHORT: case GL_UNSIGNED_SHORT:
                case GL_INT: case GL_UNSIGNED_INT:
                    break;
                default:
                    return GL_INVALID_ENUM;
            }
            if (size < 1 || size > 4) return GL_INVALID_VALUE;
            return 0;

        case FMT_LONG:
            if (type != GL_DOUBLE) return GL_INVALID_ENUM;
            if (size < 1 || size > 4) return GL_INVALID_VALUE;
            return 0;
    }

    switch (type)
    {
        case GL_BYTE: case GL_UNSIGNED_BYTE: case GL_SHORT: case GL_UNSIGNED_SHORT:
        case GL_INT: case GL_UNSIGNED_INT: case GL_HALF_FLOAT: case GL_FLOAT:
        case GL_DOUBLE: case GL_FIXED:
        case GL_INT_2_10_10_10_REV: case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            break;
        default:
            return GL_INVALID_ENUM;
    }

    if (!((size >= 1 && size <= 4) || size == GL_BGRA))
        return GL_INVALID_VALUE;

    if (size == GL_BGRA)
    {
        if (type != GL_UNSIGNED_BYTE && type != GL_INT_2_10_10_10_REV && type != GL_UNSIGNED_INT_2_10_10_10_REV)
            return GL_INVALID_OPERATION;

        if (normalized == GL_FALSE)
            return GL_INVALID_OPERATION;
    }

    if ((type == GL_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_2_10_10_10_REV) &&
        size != 4 && size != GL_BGRA)
        return GL_INVALID_OPERATION;

    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV && size != 3)
        return GL_INVALID_OPERATION;

    return 0;
}

static void storeFormat(GLMContext ctx, VertexArray *vao, GLuint attribindex, GLint size, GLenum type,
                        GLboolean normalized, int kind, GLuint relativeoffset)
{
    MGLVertexAttribFormat *f = &mgl_va_full(vao)->fmt[attribindex];

    f->size = size;
    f->type = type;
    f->normalized = (kind == FMT_FLOAT && normalized) ? GL_TRUE : GL_FALSE;
    f->integer = (kind == FMT_INTEGER) ? GL_TRUE : GL_FALSE;
    f->is_long = (kind == FMT_LONG) ? GL_TRUE : GL_FALSE;
    f->relativeoffset = relativeoffset;

    mgl_va_sync_attrib(vao, attribindex);
    vaoChanged(ctx, vao);
}

/* VertexAttribFormat / VertexArrayAttribFormat and the I / L variants */
static void attribFormat(GLMContext ctx, VertexArray *vao, GLuint attribindex, GLint size, GLenum type,
                         GLboolean normalized, int kind, GLuint relativeoffset)
{
    if (attribindex >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);
    if (relativeoffset > MGL_VA_MAX_RELATIVE_OFFSET) VA_ERROR(GL_INVALID_VALUE);

    GLenum err = checkFormat(size, type, normalized, kind);
    if (err) VA_ERROR(err);

    storeFormat(ctx, vao, attribindex, size, type, normalized, kind, relativeoffset);
}

/* shared by mglVertexAttrib{,I,L}Pointer (§10.3.2) */
static void attribPointer(GLMContext ctx, GLuint index, GLint size, GLenum type, GLboolean normalized,
                          GLsizei stride, const void *pointer, int kind)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);
    if (stride < 0 || stride > MGL_VA_MAX_STRIDE) VA_ERROR(GL_INVALID_VALUE);

    GLenum err = checkFormat(size, type, normalized, kind);
    if (err) VA_ERROR(err);

    // no default vertex array object in the core profile
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    Buffer *array_buffer = STATE(buffers[_ARRAY_BUFFER]);

    // client side arrays are not supported in the core profile
    if (array_buffer == NULL && pointer != NULL) VA_ERROR(GL_INVALID_OPERATION);

    VertexArray *vao = VAO();
    MGLVertexArrayFull *va = mgl_va_full(vao);

    storeFormat(ctx, vao, index, size, type, normalized, kind, 0);

    va->fmt[index].pointer_stride = stride;
    va->fmt[index].pointer = pointer;
    va->fmt[index].binding = index;

    MGLVertexBinding *b = &va->binding[index];
    b->buffer = array_buffer;
    b->buffer_name = array_buffer ? array_buffer->name : 0;
    b->offset = (GLintptr)pointer;
    b->stride = stride ? stride : mgl_va_element_size(type, size);

    // other attributes using this binding see the new buffer/stride as well
    for (GLuint i = 0; i < MAX_ATTRIBS; i++)
        if (va->fmt[i].binding == index)
            mgl_va_sync_attrib(vao, i);

    vaoChanged(ctx, vao);
}

void mglVertexAttribPointer(GLMContext ctx, GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer)
{
    attribPointer(ctx, index, size, type, normalized, stride, pointer, FMT_FLOAT);
}

void mglVertexAttribIPointer(GLMContext ctx, GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    attribPointer(ctx, index, size, type, GL_FALSE, stride, pointer, FMT_INTEGER);
}

void mglVertexAttribLPointer(GLMContext ctx, GLuint index, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    attribPointer(ctx, index, size, type, GL_FALSE, stride, pointer, FMT_LONG);
}

void mglVertexAttribFormat(GLMContext ctx, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, VAO(), attribindex, size, type, normalized, FMT_FLOAT, relativeoffset);
}

void mglVertexAttribIFormat(GLMContext ctx, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, VAO(), attribindex, size, type, GL_FALSE, FMT_INTEGER, relativeoffset);
}

void mglVertexAttribLFormat(GLMContext ctx, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, VAO(), attribindex, size, type, GL_FALSE, FMT_LONG, relativeoffset);
}

void mglVertexArrayAttribFormat(GLMContext ctx, GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, ptr, attribindex, size, type, normalized, FMT_FLOAT, relativeoffset);
}

void mglVertexArrayAttribIFormat(GLMContext ctx, GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, ptr, attribindex, size, type, GL_FALSE, FMT_INTEGER, relativeoffset);
}

void mglVertexArrayAttribLFormat(GLMContext ctx, GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribFormat(ctx, ptr, attribindex, size, type, GL_FALSE, FMT_LONG, relativeoffset);
}

#pragma mark - attribute -> binding association, divisors

static void attribBinding(GLMContext ctx, VertexArray *vao, GLuint attribindex, GLuint bindingindex)
{
    if (attribindex >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);
    if (bindingindex >= MGL_VA_MAX_BINDINGS) VA_ERROR(GL_INVALID_VALUE);

    mgl_va_full(vao)->fmt[attribindex].binding = bindingindex;

    mgl_va_sync_attrib(vao, attribindex);
    vaoChanged(ctx, vao);
}

void mglVertexAttribBinding(GLMContext ctx, GLuint attribindex, GLuint bindingindex)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribBinding(ctx, VAO(), attribindex, bindingindex);
}

void mglVertexArrayAttribBinding(GLMContext ctx, GLuint vaobj, GLuint attribindex, GLuint bindingindex)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribBinding(ctx, ptr, attribindex, bindingindex);
}

static void bindingDivisor(GLMContext ctx, VertexArray *vao, GLuint bindingindex, GLuint divisor)
{
    if (bindingindex >= MGL_VA_MAX_BINDINGS) VA_ERROR(GL_INVALID_VALUE);

    MGLVertexArrayFull *va = mgl_va_full(vao);
    va->binding[bindingindex].divisor = divisor;

    for (GLuint i = 0; i < MAX_ATTRIBS; i++)
        if (va->fmt[i].binding == bindingindex)
            mgl_va_sync_attrib(vao, i);

    vaoChanged(ctx, vao);
}

void mglVertexBindingDivisor(GLMContext ctx, GLuint bindingindex, GLuint divisor)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindingDivisor(ctx, VAO(), bindingindex, divisor);
}

void mglVertexArrayBindingDivisor(GLMContext ctx, GLuint vaobj, GLuint bindingindex, GLuint divisor)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    bindingDivisor(ctx, ptr, bindingindex, divisor);
}

/* VertexAttribDivisor == VertexAttribBinding(index, index) + VertexBindingDivisor(index, divisor) */
void mglVertexAttribDivisor(GLMContext ctx, GLuint index, GLuint divisor)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    attribBinding(ctx, VAO(), index, index);
    bindingDivisor(ctx, VAO(), index, divisor);
}

#pragma mark - enable / disable

static void setEnabled(GLMContext ctx, VertexArray *vao, GLuint index, bool enable)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    unsigned old = vao->enabled_attribs;

    if (enable)
        vao->enabled_attribs |= (0x1u << index);
    else
        vao->enabled_attribs &= ~(0x1u << index);

    if (old != vao->enabled_attribs)
        vaoChanged(ctx, vao);
}

void mglEnableVertexAttribArray(GLMContext ctx, GLuint index)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    setEnabled(ctx, VAO(), index, true);
}

void mglDisableVertexAttribArray(GLMContext ctx, GLuint index)
{
    if (VAO() == NULL) VA_ERROR(GL_INVALID_OPERATION);

    setEnabled(ctx, VAO(), index, false);
}

void mglEnableVertexArrayAttrib(GLMContext ctx, GLuint vaobj, GLuint index)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    setEnabled(ctx, ptr, index, true);
}

void mglDisableVertexArrayAttrib(GLMContext ctx, GLuint vaobj, GLuint index)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    setEnabled(ctx, ptr, index, false);
}

#pragma mark - element array buffer (DSA)

void mglVertexArrayElementBuffer(GLMContext ctx, GLuint vaobj, GLuint buffer)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    Buffer *buf_ptr = NULL;

    if (buffer)
    {
        buf_ptr = findBuffer(ctx, buffer);
        if (buf_ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);
    }

    ptr->element_array.buffer = buf_ptr;

    if (ptr == STATE(vao))
    {
        STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = buf_ptr;
        STATE(var.element_array_buffer_binding) = buffer;
    }
}

#pragma mark - vertex array queries (§10.5)

void mglGetVertexArrayiv(GLMContext ctx, GLuint vaobj, GLenum pname, GLint *param)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    if (pname != GL_ELEMENT_ARRAY_BUFFER_BINDING) VA_ERROR(GL_INVALID_ENUM);

    *param = ptr->element_array.buffer ? (GLint)ptr->element_array.buffer->name : 0;
}

void mglGetVertexArrayIndexediv(GLMContext ctx, GLuint vaobj, GLuint index, GLenum pname, GLint *param)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    switch (pname)
    {
        case GL_VERTEX_ATTRIB_ARRAY_ENABLED:
        case GL_VERTEX_ATTRIB_ARRAY_SIZE:
        case GL_VERTEX_ATTRIB_ARRAY_STRIDE:
        case GL_VERTEX_ATTRIB_ARRAY_TYPE:
        case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED:
        case GL_VERTEX_ATTRIB_ARRAY_INTEGER:
        case GL_VERTEX_ATTRIB_ARRAY_LONG:
        case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:
        case GL_VERTEX_ATTRIB_RELATIVE_OFFSET:
        {
            GLint64 v = 0;
            mgl_va_attrib_param(ptr, index, pname, &v);
            *param = (GLint)v;
            return;
        }
    }

    VA_ERROR(GL_INVALID_ENUM);
}

void mglGetVertexArrayIndexed64iv(GLMContext ctx, GLuint vaobj, GLuint index, GLenum pname, GLint64 *param)
{
    VertexArray *ptr = lookupVAO(ctx, vaobj);
    if (ptr == NULL) VA_ERROR(GL_INVALID_OPERATION);

    if (pname != GL_VERTEX_BINDING_OFFSET) VA_ERROR(GL_INVALID_ENUM);

    if (index >= MGL_VA_MAX_BINDINGS) VA_ERROR(GL_INVALID_VALUE);

    *param = (GLint64)mgl_va_full(ptr)->binding[index].offset;
}
