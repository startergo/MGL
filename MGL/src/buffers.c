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
 * buffers.c
 * MGL
 *
 * GL 4.6 buffer objects.
 *
 * Storage model: Buffer.data.buffer_data is the single copy of a buffer's
 * contents. Once the GPU needs it, the renderer wraps / replaces it with an
 * MTLBuffer whose contents alias buffer_data (see MGLRenderer+Draw.m), so CPU
 * and GPU see the same bytes. CPU access is ordered against queued GPU work with
 * mgl_buffers_prepare_cpu_read/_write (draw_ext.c).
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "glm_context.h"
/* MGL_PROFILE=1 per-draw profiler (MGLRenderer.m) */
uint64_t mgl_prof_begin(int id);
void mgl_prof_end(int id, uint64_t t0);
#include "buffers.h"
#include "mgl_ext_buffers.h"
#include "mgl_ext_gpuaudit.h"   /* gpuaudit: content-generation invalidation */
#include "mgl_ext_vattrib.h"
#include "mgl_ext_xfb.h"
#include "mgl_buffers_err.h"

#pragma mark Utility Functions

#define MGL_BUFFER_NO_INDEX 0xFFFFFFFF

GLuint bufferIndexFromTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_ARRAY_BUFFER: return _ARRAY_BUFFER;
        case GL_ELEMENT_ARRAY_BUFFER: return _ELEMENT_ARRAY_BUFFER;
        case GL_UNIFORM_BUFFER: return _UNIFORM_BUFFER;
        case GL_TEXTURE_BUFFER: return _TEXTURE_BUFFER;
        case GL_TRANSFORM_FEEDBACK_BUFFER: return _TRANSFORM_FEEDBACK_BUFFER;
        case GL_QUERY_BUFFER: return _QUERY_BUFFER;
        case GL_PIXEL_PACK_BUFFER: return _PIXEL_PACK_BUFFER;
        case GL_PIXEL_UNPACK_BUFFER: return _PIXEL_UNPACK_BUFFER;
        case GL_ATOMIC_COUNTER_BUFFER: return _ATOMIC_COUNTER_BUFFER;
        case GL_COPY_READ_BUFFER: return _COPY_READ_BUFFER;
        case GL_COPY_WRITE_BUFFER: return _COPY_WRITE_BUFFER;
        case GL_DISPATCH_INDIRECT_BUFFER: return _DISPATCH_INDIRECT_BUFFER;
        case GL_DRAW_INDIRECT_BUFFER: return _DRAW_INDIRECT_BUFFER;
        case GL_SHADER_STORAGE_BUFFER: return _SHADER_STORAGE_BUFFER;
        default: break;   /* GL_PARAMETER_BUFFER has no STATE(buffers) slot */
    }

    return MGL_BUFFER_NO_INDEX;
}

bool checkTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_ARRAY_BUFFER:
        case GL_ELEMENT_ARRAY_BUFFER:
        case GL_UNIFORM_BUFFER:
        case GL_TEXTURE_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_QUERY_BUFFER:
        case GL_PIXEL_PACK_BUFFER:
        case GL_PIXEL_UNPACK_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
        case GL_COPY_READ_BUFFER:
        case GL_COPY_WRITE_BUFFER:
        case GL_DISPATCH_INDIRECT_BUFFER:
        case GL_DRAW_INDIRECT_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
        case GL_PARAMETER_BUFFER:
            return true;
    }

    return false;
}

bool checkUsage(GLMContext ctx, GLenum usage)
{
    switch(usage)
    {
        case GL_STREAM_DRAW:
        case GL_STREAM_READ:
        case GL_STREAM_COPY:
        case GL_STATIC_DRAW:
        case GL_STATIC_READ:
        case GL_STATIC_COPY:
        case GL_DYNAMIC_DRAW:
        case GL_DYNAMIC_READ:
        case GL_DYNAMIC_COPY:
            return true;
    }

    return false;
}

/* the binding point a target refers to (element array binding is VAO state) */
static Buffer **bufferSlot(GLMContext ctx, GLenum target)
{
    if (target == GL_PARAMETER_BUFFER)
        return ctx->state.ext_buffers ? &ctx->state.ext_buffers->parameter_buffer : NULL;

    if (target == GL_ELEMENT_ARRAY_BUFFER && VAO())
        return &VAO()->element_array.buffer;

    GLuint index = bufferIndexFromTarget(ctx, target);
    if (index == MGL_BUFFER_NO_INDEX)
        return NULL;

    return &STATE(buffers[index]);
}

static Buffer *boundBuffer(GLMContext ctx, GLenum target)
{
    Buffer **slot = bufferSlot(ctx, target);
    return slot ? *slot : NULL;
}

Buffer *newBuffer(GLMContext ctx, GLenum target, GLuint name)
{
    Buffer *ptr;

    ptr = (Buffer *)calloc(1, sizeof(Buffer));
    assert(ptr);

    ptr->name = name;
    ptr->target = target;
    ptr->usage = GL_STATIC_DRAW;      // initial GL_BUFFER_USAGE
    ptr->access = GL_READ_WRITE;      // initial GL_BUFFER_ACCESS

    // create buffers doesn't provide a target
    if (target)
    {
        GLuint index = bufferIndexFromTarget(ctx, target);
        ptr->index = (index == MGL_BUFFER_NO_INDEX) ? 0 : index;
    }

    return ptr;
}

Buffer *getBuffer(GLMContext ctx, GLenum target, GLuint buffer)
{
    Buffer *ptr;

    ptr = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);

    if (!ptr)
    {
        ptr = newBuffer(ctx, target, buffer);

        insertHashElement(&STATE(buffer_table), buffer, ptr);
    }

    return ptr;
}

bool isBuffer(GLMContext ctx, GLuint buffer)
{
    if (buffer == 0)
        return false;

    return searchHashTable(&STATE(buffer_table), buffer) != NULL;
}

Buffer *findBuffer(GLMContext ctx, GLuint buffer)
{
    if (buffer == 0)
        return NULL;

    return (Buffer *)searchHashTable(&STATE(buffer_table), buffer);
}

/* a name returned by GenBuffers/CreateBuffers (object may not exist yet) */
static bool isBufferName(GLMContext ctx, GLuint buffer)
{
    return buffer != 0 && buffer < STATE(buffer_table).current_name;
}

static inline bool mgl_range_ok(GLintptr offset, GLsizeiptr size, GLsizeiptr total)
{
    if (offset < 0 || size < 0 || total < 0)
        return false;

    if ((uint64_t)offset > (uint64_t)total)
        return false;

    if ((uint64_t)size > (uint64_t)total - (uint64_t)offset)
        return false;

    return true;
}

size_t page_size_align(size_t size)
{
    if (size == 0)
        size = 1;

    return (size + 4095) & ~(size_t)4095;
}

static inline struct MGLExtMetal_buffers *mtlext(GLMContext ctx)
{
    return ctx->mtl_funcs.ext_buffers;
}

/* GL_BUFFER_ACCESS value derived from map access flags */
static GLenum accessFromFlags(GLbitfield flags)
{
    if ((flags & GL_MAP_READ_BIT) && (flags & GL_MAP_WRITE_BIT)) return GL_READ_WRITE;
    if (flags & GL_MAP_WRITE_BIT) return GL_WRITE_ONLY;
    return GL_READ_ONLY;
}

/* is any part of [offset, offset+size) of a non-persistently mapped buffer mapped */
static bool mappedNonPersistent(Buffer *ptr)
{
    return ptr->mapped && !(ptr->access_flags & GL_MAP_PERSISTENT_BIT);
}

/* GL_ARB_buffer_storage sub-data rule (Mesa buffer_object_subdata_range_good):
 * a persistent mapping never blocks client writes — the address is stable and
 * GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT orders it; a non-persistent mapping blocks
 * only the ranges it covers.  A whole-store range degenerates to the pre-4.4
 * "buffer is mapped" error. */
static bool mappedRangeBlocked(Buffer *ptr, GLintptr offset, GLsizeiptr size)
{
    if (!mappedNonPersistent(ptr))
        return false;
    return offset < ptr->mapped_offset + ptr->mapped_length &&
           ptr->mapped_offset < offset + size;
}

/* CPU wrote [offset, offset+size) of the buffer contents */
static void bufferDidModify(GLMContext ctx, Buffer *ptr, size_t offset, size_t size)
{
    ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;

    /* gpuaudit: the CPU-visible contents changed; cached index rewrites/scans must go */
    mgl_gpuaudit_invalidate(ctx, ptr);

    if (ptr->data.mtl_data && mtlext(ctx) && mtlext(ctx)->did_modify)
        mtlext(ctx)->did_modify(ctx, ptr, offset, size);
}

/* drop the backing store of a buffer (respecification / deletion) */
static void releaseBufferStorage(GLMContext ctx, Buffer *ptr)
{
    mgl_buffers_forget(ctx, ptr);

    if (ptr->data.mtl_data)
    {
        // the MTLBuffer owns the memory (or deallocates the client vm region);
        // queued command buffers keep their own reference until they complete
        if (mtlext(ctx) && mtlext(ctx)->release_buffer)
            mtlext(ctx)->release_buffer(ctx, ptr);
        else
            ctx->mtl_funcs.mtlDeleteMTLObj(ctx, ptr->data.mtl_data);

        ptr->data.mtl_data = NULL;
    }
    else if (ptr->data.buffer_data)
    {
        vm_deallocate(mach_task_self(), ptr->data.buffer_data, ptr->data.buffer_size);
    }

    ptr->data.buffer_data = 0;
    ptr->data.buffer_size = 0;
    ptr->data.dirty_bits = 0;
}

void *getBufferData(GLMContext ctx, Buffer *ptr)
{
    // used by texture uploads: reads the bound PIXEL_UNPACK_BUFFER
    ptr = STATE(buffers[_PIXEL_UNPACK_BUFFER]);

    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, NULL);
    ERROR_CHECK_RETURN_VALUE(mappedNonPersistent(ptr) == false, GL_INVALID_OPERATION, NULL);
    ERROR_CHECK_RETURN_VALUE(ptr->data.buffer_data, GL_INVALID_OPERATION, NULL);

    mgl_buffers_prepare_cpu_read(ctx, ptr);

    return (void *)ptr->data.buffer_data;
}

static void unmapBufferInternal(GLMContext ctx, Buffer *ptr)
{
    if (!ptr->mapped)
        return;

    if ((ptr->access_flags & GL_MAP_WRITE_BIT) && !(ptr->access_flags & GL_MAP_FLUSH_EXPLICIT_BIT))
    {
        bufferDidModify(ctx, ptr, ptr->mapped_offset, ptr->mapped_length);
    }

    ptr->mapped = GL_FALSE;
    ptr->access = GL_READ_WRITE;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = 0;
}

#pragma mark storage allocation

/* allocate (or reuse) the data store; returns 0 on success */
kern_return_t initBufferData(GLMContext ctx, Buffer *ptr, GLsizeiptr size, const void *data, bool isUniformConstant)
{
    mgl_gpuaudit_invalidate(ctx, ptr);   /* gpuaudit: new contents */

    kern_return_t err;
    vm_address_t buffer_data;
    size_t buffer_size;

    if (ptr->data.buffer_data && isUniformConstant && (size_t)size <= ptr->data.buffer_size)
    {
        // default-block uniform storage: reuse the allocation
        if (data)
        {
            mgl_buffers_prepare_cpu_write(ctx, ptr, true);

            memcpy((void *)ptr->data.buffer_data, data, size);
            ptr->size = size;

            bufferDidModify(ctx, ptr, 0, size);
        }

        return 0;
    }

    unmapBufferInternal(ctx, ptr);
    releaseBufferStorage(ctx, ptr);

    buffer_size = page_size_align(size);

    // Allocate directly from VM (page aligned, zero filled)
    err = vm_allocate((vm_map_t) mach_task_self(),
                      (vm_address_t*) &buffer_data,
                      buffer_size,
                      VM_FLAGS_ANYWHERE);
    if (err)
    {
        ptr->size = 0;
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, err);
    }

    ptr->size = size;
    ptr->data.buffer_data = buffer_data;
    ptr->data.buffer_size = buffer_size;
    ptr->data.dirty_bits = DIRTY_BUFFER_ADDR;

    if (data)
    {
        memcpy((void *)ptr->data.buffer_data, data, size);

        ptr->data.dirty_bits |= DIRTY_BUFFER_DATA;
    }

    return err;
}

void bufferStorage(GLMContext ctx, Buffer *ptr, GLenum target, GLuint index, GLsizeiptr size, const void *data, GLbitfield storage_flags, GLenum usage)
{
    if (initBufferData(ctx, ptr, size, data, false))
        return;

    if (target)
    {
        ptr->target = target;
        ptr->index = index;
    }

    ptr->usage = usage ? usage : GL_DYNAMIC_DRAW;
    ptr->immutable_storage = usage ? GL_FALSE : GL_TRUE;
    ptr->storage_flags = storage_flags;

    ptr->mapped = GL_FALSE;
    ptr->access = GL_READ_WRITE;
    ptr->access_flags = 0;
    ptr->mapped_offset = 0;
    ptr->mapped_length = 0;
}

static bool allowImmutableRealloc(void)
{
    static int cached = -1;
    if (cached < 0)
    {
        const char *e = getenv("MGL_RELAX_IMMUTABLE");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached == 1;
}

static void bufferData(GLMContext ctx, Buffer *ptr, GLsizeiptr size, const void *data, GLenum usage)
{
    ERROR_CHECK_RETURN(checkUsage(ctx, usage), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(size >= 0, GL_INVALID_VALUE);

    // GL_INVALID_OPERATION if the buffer's storage is immutable
    // (MGL_RELAX_IMMUTABLE=1 restores the old permissive behaviour for broken guests)
    if (ptr->immutable_storage && !allowImmutableRealloc())
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (initBufferData(ctx, ptr, size, data, false))
        return;

    ptr->usage = usage;
    ptr->immutable_storage = GL_FALSE;
    // storage flags of a mutable buffer
    ptr->storage_flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT;
}

static bool checkStorageFlags(GLMContext ctx, GLsizeiptr size, GLbitfield flags)
{
    ERROR_CHECK_RETURN_VALUE(size > 0, GL_INVALID_VALUE, false);

    ERROR_CHECK_RETURN_VALUE((flags & ~(GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                                        GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT | GL_CLIENT_STORAGE_BIT)) == 0,
                             GL_INVALID_VALUE, false);

    // persistent needs read or write, coherent needs persistent
    if ((flags & GL_MAP_PERSISTENT_BIT) && !(flags & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    if ((flags & GL_MAP_COHERENT_BIT) && !(flags & GL_MAP_PERSISTENT_BIT))
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    return true;
}

static void bufferSubData(GLMContext ctx, Buffer *ptr, GLintptr offset, GLsizeiptr size, const void *data)
{
    ERROR_CHECK_RETURN(offset >= 0 && size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mgl_range_ok(offset, size, ptr->size), GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(!mappedRangeBlocked(ptr, offset, size), GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!(ptr->immutable_storage && !(ptr->storage_flags & GL_DYNAMIC_STORAGE_BIT)), GL_INVALID_OPERATION);

    if (size == 0 || data == NULL)
        return;

    ERROR_CHECK_RETURN(ptr->data.buffer_data, GL_INVALID_OPERATION);

    // GL ordering: commands issued before this call must see the old contents
    mgl_buffers_prepare_cpu_write(ctx, ptr, offset == 0 && size == ptr->size);

    memcpy((char *)ptr->data.buffer_data + offset, data, size);

    bufferDidModify(ctx, ptr, offset, size);
}

#pragma mark GL Buffer Functions
void mglGenBuffers(GLMContext ctx, GLsizei n, GLuint *buffers)
{
    ERROR_CHECK_RETURN(n >= 0, GL_INVALID_VALUE);

    while(n--)
    {
        *buffers++ = getNewName(&STATE(buffer_table));
    }
}

void mglCreateBuffers(GLMContext ctx, GLsizei n, GLuint *buffers)
{
    GLuint name;

    ERROR_CHECK_RETURN(n >= 0, GL_INVALID_VALUE);

    while(n--)
    {
        name = getNewName(&STATE(buffer_table));

        // create an unbound buffer object
        getBuffer(ctx, 0, name);

        *buffers++ = name;
    }
}

static GLuint *bindingVarForIndex(GLMContext ctx, GLuint index)
{
    switch(index)
    {
        case _ARRAY_BUFFER: return &STATE_VAR(array_buffer_binding);
        case _ELEMENT_ARRAY_BUFFER: return &STATE_VAR(element_array_buffer_binding);
        case _PIXEL_PACK_BUFFER: return &STATE_VAR(pixel_pack_buffer_binding);
        case _PIXEL_UNPACK_BUFFER: return &STATE_VAR(pixel_unpack_buffer_binding);
        case _UNIFORM_BUFFER: return &STATE_VAR(uniform_buffer_binding);
        case _DISPATCH_INDIRECT_BUFFER: return &STATE_VAR(dispatch_indirect_buffer_binding);
        case _SHADER_STORAGE_BUFFER: return &STATE_VAR(shader_storage_buffer_binding);
    }

    return NULL;
}

static void setGenericBinding(GLMContext ctx, GLenum target, Buffer *ptr)
{
    Buffer **slot = bufferSlot(ctx, target);

    if (!slot)
        return;

    *slot = ptr;

    GLuint index = bufferIndexFromTarget(ctx, target);
    if (index != MGL_BUFFER_NO_INDEX)
    {
        // keep the non-VAO slot in sync too
        STATE(buffers[index]) = ptr;

        GLuint *var = bindingVarForIndex(ctx, index);
        if (var)
            *var = ptr ? ptr->name : 0;
    }

    // vattrib: the element array buffer binding is vertex array object state
    if (target == GL_ELEMENT_ARRAY_BUFFER)
    {
        mgl_va_element_buffer_bound(ctx, ptr);
        if (VAO())
            STATE(dirty_bits) |= DIRTY_VAO;
    }
}

/* ---- uniform buffer binding points above MAX_BINDABLE_BUFFERS ----------------
 *
 * STATE(buffer_base) rows hold MAX_BINDABLE_BUFFERS points; the uniform-buffer
 * points above them (up to MGL_MAX_UNIFORM_BUFFER_BINDINGS, compat r111) live in
 * STATE(ubo_high) (r159: per context, one entry per point).  The draw path reads UBO
 * slots through mgl_buffers_indexed_binding(), and get.c's indexed queries too. */

/* the storage of GL uniform-buffer binding point `index` (row slot below
 * MAX_BINDABLE_BUFFERS, ubo_high entry above it), or NULL beyond the advertised
 * range — get.c's indexed queries report 0 for those. */
BufferBaseTarget *mgl_buffers_indexed_binding(GLMContext ctx, GLenum target, GLuint index)
{
    if (target != GL_UNIFORM_BUFFER || index < MAX_BINDABLE_BUFFERS)
        return &ctx->state.buffer_base[_UNIFORM_BUFFER].buffers[index];
    if (index < MGL_MAX_UNIFORM_BUFFER_BINDINGS)
        return &ctx->state.ubo_high[index - MAX_BINDABLE_BUFFERS];
    return NULL;
}

static void unbindEverywhere(GLMContext ctx, Buffer *ptr)
{
    for (int i = 0; i < _MAX_BUFFER_TYPES; i++)
    {
        if (STATE(buffers[i]) == ptr)
        {
            STATE(buffers[i]) = NULL;

            GLuint *var = bindingVarForIndex(ctx, i);
            if (var) *var = 0;
        }

        for (int j = 0; j < MAX_BINDABLE_BUFFERS; j++)
        {
            if (STATE(buffer_base[i]).buffers[j].buf == ptr)
            {
                bzero(&STATE(buffer_base[i]).buffers[j], sizeof(BufferBaseTarget));
                STATE(dirty_bits) |= DIRTY_BUFFER_BASE_STATE;
            }
        }
    }

    if (ctx->state.ext_buffers && ctx->state.ext_buffers->parameter_buffer == ptr)
        ctx->state.ext_buffers->parameter_buffer = NULL;

    /* uniform-buffer points above the rows (>= MAX_BINDABLE_BUFFERS) hold Buffer pointers too */
    for (int i = 0; i < MGL_MAX_UNIFORM_BUFFER_BINDINGS - MAX_BINDABLE_BUFFERS; i++)
        if (STATE(ubo_high[i]).buf == ptr)
        {
            bzero(&STATE(ubo_high[i]), sizeof(BufferBaseTarget));
            STATE(ubo_high_gen)++;
            STATE(dirty_bits) |= DIRTY_BUFFER_BASE_STATE;
        }

    // vertex array objects referencing the buffer (all VAOs: the storage is freed)
    HashTable *vt = &STATE(vao_table);
    for (GLuint n = 0; vt->keys && n < vt->size; n++)
    {
        VertexArray *vao = (VertexArray *)vt->keys[n].data;
        if (!vao) continue;

        bool touched = false;
        for (int a = 0; a < MAX_ATTRIBS; a++)
        {
            if (vao->attrib[a].buffer == ptr)
            {
                vao->attrib[a].buffer = NULL;
                touched = true;
            }
        }

        if (vao->element_array.buffer == ptr)
        {
            vao->element_array.buffer = NULL;
            touched = true;
        }

        // vattrib: vertex buffer bindings of the full VAO record
        MGLVertexArrayFull *va = mgl_va_full(vao);
        for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
        {
            if (va->binding[b].buffer == ptr)
            {
                va->binding[b].buffer = NULL;
                va->binding[b].buffer_name = 0;
                touched = true;
            }
        }

        if (touched)
        {
            vao->dirty_bits |= DIRTY_VAO_BUFFER_BASE;
            if (vao == VAO())
                STATE(dirty_bits) |= DIRTY_VAO;
        }
    }

    if (VAO() && VAO()->element_array.buffer == NULL)
        STATE_VAR(element_array_buffer_binding) = 0;

    // drop stale entries of the Metal buffer maps
    BufferMapList *lists[3] = { &STATE(vertex_buffer_map_list), &STATE(fragment_buffer_map_list), &STATE(compute_buffer_map_list) };
    for (int l = 0; l < 3; l++)
    {
        for (GLuint i = 0; i < lists[l]->count; i++)
        {
            if (lists[l]->buffers[i].buf == ptr)
            {
                lists[l]->buffers[i].buf = NULL;
                STATE(dirty_bits) |= DIRTY_BUFFER_BASE_STATE;
            }
        }
    }
}

void mglDeleteBuffers(GLMContext ctx, GLsizei n, const GLuint *buffers)
{
    ERROR_CHECK_RETURN(n >= 0, GL_INVALID_VALUE);

    while(n--)
    {
        GLuint buffer = *buffers++;
        Buffer *ptr = findBuffer(ctx, buffer);

        if (!ptr)
            continue; // unused names and zero are silently ignored

        unmapBufferInternal(ctx, ptr);
        unbindEverywhere(ctx, ptr);

        if (ptr->data.mtl_data || ptr->data.buffer_data)
            releaseBufferStorage(ctx, ptr);

        mgl_buffers_forget(ctx, ptr);

        deleteHashElement(&STATE(buffer_table), buffer);

        mgl_buffer_pool_free(&ptr->data);   /* r153: retired storages */
        free(ptr);
    }
}

GLboolean mglIsBuffer(GLMContext ctx, GLuint buffer)
{
    return isBuffer(ctx, buffer) ? GL_TRUE : GL_FALSE;
}

void mglBindBuffer(GLMContext ctx, GLenum target, GLuint buffer)
{
    Buffer *ptr = NULL;

    // GL_INVALID_ENUM is generated if target is not supported.
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    if (buffer)
    {
        // GL_INVALID_VALUE (core: INVALID_OPERATION) for names not from GenBuffers
        ERROR_CHECK_RETURN(isBufferName(ctx, buffer), GL_INVALID_OPERATION);

        ptr = getBuffer(ctx, target, buffer);
        ERROR_CHECK_RETURN(ptr, GL_OUT_OF_MEMORY);

        if (ptr->target == 0)
        {
            ptr->target = target;
            GLuint index = bufferIndexFromTarget(ctx, target);
            ptr->index = (index == MGL_BUFFER_NO_INDEX) ? 0 : index;
        }
    }

    setGenericBinding(ctx, target, ptr);
}

#pragma mark indexed bindings

static GLuint maxBindingsForTarget(GLMContext ctx, GLenum target)
{
    GLuint max = MAX_BINDABLE_BUFFERS;

    switch(target)
    {
        case GL_UNIFORM_BUFFER:
            /* GL_MAX_UNIFORM_BUFFER_BINDINGS is reported as at least the GL 4.6 minimum
             * (84 — see get.c); binding points at or above MAX_BINDABLE_BUFFERS have no
             * row in STATE(buffer_base) and live in the overflow table below, because the
             * renderer consumes UBO bindings through a 16-entry table.  get.c's
             * reported_uniform_buffer_bindings() is the same expression.
             * compat r111: the reported limit is MGL_MAX_UNIFORM_BUFFER_BINDINGS
             * (512) — rsvga's D3D9 fragment constant block sits at binding 256. */
            max = STATE_VAR(max_uniform_buffer_bindings) > MGL_MAX_UNIFORM_BUFFER_BINDINGS
                      ? (GLuint)STATE_VAR(max_uniform_buffer_bindings)
                      : MGL_MAX_UNIFORM_BUFFER_BINDINGS;
            break;

        case GL_SHADER_STORAGE_BUFFER:
        {
            /* getMacOSDefaults() asks Apple's GL 4.1 for GL 4.3+ limits and gets
             * nothing back, so the cached value can be unset (0): fall back to the
             * limit get.c reports (MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS), keeping
             * the error paths of glBindBuffer{Base,Range} spec-correct
             * (KHR-GL46.shader_storage_buffer_object negative-api-bind).
             * compat r108/r109: that limit is 16 (per-stage SSBO bindings,
             * raised so KHR-GL46.gl_spirv.spirv_validation_capabilities'
             * 10-instance block array links) — the hardcoded 8 here made
             * glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8..15, ...) a silent
             * no-op and es_31_compatibility.shader_storage_buffer_object.
             * basic-binding fail (binding 8 read back as 0). */
            GLuint lim = STATE_VAR(max_shader_storage_buffer_bindings);
            if (!lim || lim > MAX_BINDABLE_BUFFERS) lim = MAX_BINDABLE_BUFFERS < 16 ? MAX_BINDABLE_BUFFERS : 16;
            if (lim < max) max = lim;
        }
            break;

        case GL_TRANSFORM_FEEDBACK_BUFFER:
            if (max > 4) max = 4;    // GL_MAX_TRANSFORM_FEEDBACK_BUFFERS
            break;

        case GL_ATOMIC_COUNTER_BUFFER:
            if (max > 8) max = 8;    // GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS
            break;

        default:
            return 0;
    }

    return max;
}

static GLuint offsetAlignmentForTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_UNIFORM_BUFFER:
            return STATE_VAR(uniform_buffer_offset_alignment) ? STATE_VAR(uniform_buffer_offset_alignment) : 256;

        case GL_SHADER_STORAGE_BUFFER:
            return STATE_VAR(shader_storage_buffer_offset_alignment) ? STATE_VAR(shader_storage_buffer_offset_alignment) : 16;

        default:
            return 4; // atomic counter / transform feedback
    }
}

static void setIndexedBinding(GLMContext ctx, GLenum target, GLuint index, Buffer *ptr, GLintptr offset, GLsizeiptr size, bool range)
{
    GLuint buffer_index = bufferIndexFromTarget(ctx, target);
    BufferBaseTarget *bt;

    /* a uniform-buffer binding point above the rows: its STATE(ubo_high) entry */
    if (index >= MAX_BINDABLE_BUFFERS)
    {
        BufferBaseTarget *hb = (target == GL_UNIFORM_BUFFER)
                                   ? mgl_buffers_indexed_binding(ctx, target, index) : NULL;

        if (!hb)
            return;                 /* beyond the advertised range: validated by the callers */
        if (!ptr)
        {
            bzero(hb, sizeof(*hb));
        }
        else
        {
            hb->buffer = ptr->name;
            hb->offset = offset;
            hb->size = range ? size : ptr->size;
            hb->buf = ptr;

            if (ptr->data.buffer_data && mtlext(ctx) && mtlext(ctx)->ensure_buffer)
                mtlext(ctx)->ensure_buffer(ctx, ptr);
        }
        {
            GLuint h = index - MAX_BINDABLE_BUFFERS;
            if (range && ptr) ctx->state.ubo_high_range[h >> 5] |= 1u << (h & 31);
            else              ctx->state.ubo_high_range[h >> 5] &= ~(1u << (h & 31));
        }

        ctx->state.ubo_high_gen++;
        setGenericBinding(ctx, target, ptr);
        ctx->state.dirty_bits |= DIRTY_BUFFER_BASE_STATE;
        return;
    }

    bt = &ctx->state.buffer_base[buffer_index].buffers[index];

    if (ptr)
    {
        bt->buffer = ptr->name;
        bt->offset = offset;
        bt->size = range ? size : ptr->size;
        bt->buf = ptr;

        // storage written by shaders must live in a real MTLBuffer (no setBytes copies)
        if ((target == GL_SHADER_STORAGE_BUFFER || target == GL_ATOMIC_COUNTER_BUFFER ||
             target == GL_TRANSFORM_FEEDBACK_BUFFER) && ptr->data.buffer_data &&
            mtlext(ctx) && mtlext(ctx)->ensure_buffer)
        {
            mtlext(ctx)->ensure_buffer(ctx, ptr);
        }
    }
    else
    {
        bzero(bt, sizeof(BufferBaseTarget));
    }

    if (ctx->state.ext_buffers)
    {
        uint32_t bit = 1u << (index & 31);
        int slot = (target == GL_UNIFORM_BUFFER) ? 0 : (target == GL_SHADER_STORAGE_BUFFER) ? 1 :
                   (target == GL_ATOMIC_COUNTER_BUFFER) ? 2 : 3;
        uint32_t *mask = mgl_buffers_range_mask(ctx, slot) + (index >> 5);
        if (range && ptr) *mask |= bit; else *mask &= ~bit;
    }

    // the generic binding point is updated as well
    setGenericBinding(ctx, target, ptr);

    ctx->state.dirty_bits |= DIRTY_BUFFER_BASE_STATE;

}

static bool checkIndexedTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_UNIFORM_BUFFER:
        case GL_TRANSFORM_FEEDBACK_BUFFER:
        case GL_SHADER_STORAGE_BUFFER:
        case GL_ATOMIC_COUNTER_BUFFER:
            return true;
    }

    return false;
}

void mglBindBufferBase(GLMContext ctx, GLenum target, GLuint index, GLuint buffer)
{
    Buffer *ptr = NULL;

    ERROR_CHECK_RETURN(checkIndexedTarget(ctx, target), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(index < maxBindingsForTarget(ctx, target), GL_INVALID_VALUE);

    if (buffer)
        ERROR_CHECK_RETURN(isBufferName(ctx, buffer), GL_INVALID_OPERATION);

    if (!mgl_xfb_validate_bind_buffer(ctx, target, index, buffer, 0, 0, false)) return; // xfb: TF binding rules

    if (buffer)
    {
        ptr = getBuffer(ctx, target, buffer);
        ERROR_CHECK_RETURN(ptr, GL_OUT_OF_MEMORY);

        if (ptr->target == 0)
            ptr->target = target;
    }

    setIndexedBinding(ctx, target, index, ptr, 0, 0, false);
}

void mglBindBufferRange(GLMContext ctx, GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    Buffer *ptr = NULL;

    ERROR_CHECK_RETURN(checkIndexedTarget(ctx, target), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(index < maxBindingsForTarget(ctx, target), GL_INVALID_VALUE);

    /* GL 4.6 6.1.1: BindBufferRange validates offset / alignment regardless of the
     * bound name — binding zero with an unaligned offset is still INVALID_VALUE
     * (KHR-GL46.shader_storage_buffer_object negative-api-bind); the size>0 check
     * applies to a real buffer only (basic-binding binds name 0 with size 0 and
     * expects the binding to simply break) */
    ERROR_CHECK_RETURN(offset >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN((offset % offsetAlignmentForTarget(ctx, target)) == 0, GL_INVALID_VALUE);

    if (buffer)
    {
        ERROR_CHECK_RETURN(isBufferName(ctx, buffer), GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN(size > 0, GL_INVALID_VALUE);

        if (target == GL_TRANSFORM_FEEDBACK_BUFFER || target == GL_ATOMIC_COUNTER_BUFFER)
        {
            if (target == GL_TRANSFORM_FEEDBACK_BUFFER)
                ERROR_CHECK_RETURN((size % 4) == 0, GL_INVALID_VALUE);
        }

    }

    if (!mgl_xfb_validate_bind_buffer(ctx, target, index, buffer, offset, size, true)) return; // xfb: TF binding rules

    if (buffer)
    {
        ptr = getBuffer(ctx, target, buffer);
        ERROR_CHECK_RETURN(ptr, GL_OUT_OF_MEMORY);

        if (ptr->target == 0)
            ptr->target = target;
    }

    setIndexedBinding(ctx, target, index, ptr, offset, size, true);
}

static void bindBuffersRangeInternal(GLMContext ctx, GLenum target, GLuint first, GLsizei count,
                                     const GLuint *buffers, const GLintptr *offsets, const GLsizeiptr *sizes, bool range)
{
    ERROR_CHECK_RETURN(checkIndexedTarget(ctx, target), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN((uint64_t)first + (uint64_t)count <= maxBindingsForTarget(ctx, target), GL_INVALID_OPERATION);

    bool error_op = false, error_val = false;

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint name = buffers ? buffers[i] : 0;
        Buffer *ptr = NULL;

        if (name)
        {
            /* multi-bind takes only EXISTING buffer objects (GL 4.6 16.2.1: a name
             * generated but never bound is not an existing object) — the same
             * definition glIsBuffer uses; KHR-GL46.multi_bind errors_bind_buffers
             * expects GL_INVALID_OPERATION for such names */
            if (!isBuffer(ctx, name))
            {
                error_op = true;   // this binding is skipped, the others proceed
                continue;
            }

            if (range)
            {
                if (offsets[i] < 0 || sizes[i] <= 0 ||
                    (offsets[i] % offsetAlignmentForTarget(ctx, target)) != 0 ||
                    (target == GL_TRANSFORM_FEEDBACK_BUFFER && (sizes[i] % 4) != 0))
                {
                    error_val = true;
                    continue;
                }
            }

            ptr = getBuffer(ctx, target, name);
            if (ptr->target == 0)
                ptr->target = target;
        }

        if (target == GL_TRANSFORM_FEEDBACK_BUFFER &&
            !mgl_xfb_validate_bind_buffer(ctx, target, first + i, name, (range && ptr) ? offsets[i] : 0,
                                          (range && ptr) ? sizes[i] : 0, range && ptr))
            continue; // xfb: TF binding rules (error already raised)

        setIndexedBinding(ctx, target, first + i, ptr, (range && ptr) ? offsets[i] : 0,
                          (range && ptr) ? sizes[i] : 0, range && ptr);
    }

    if (error_op)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
    }

    if (error_val)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }
}

void mglBindBuffersBase(GLMContext ctx, GLenum target, GLuint first, GLsizei count, const GLuint *buffers)
{
    bindBuffersRangeInternal(ctx, target, first, count, buffers, NULL, NULL, false);
}

void mglBindBuffersRange(GLMContext ctx, GLenum target, GLuint first, GLsizei count, const GLuint *buffers, const GLintptr *offsets, const GLsizeiptr *sizes)
{
    if (buffers)
    {
        ERROR_CHECK_RETURN(offsets != NULL && sizes != NULL, GL_INVALID_VALUE);
    }

    bindBuffersRangeInternal(ctx, target, first, count, buffers, offsets, sizes, true);
}

#pragma mark GL Buffer Data Functions

void mglBufferData(GLMContext ctx, GLenum target, GLsizeiptr size, const void *data, GLenum usage)
{
    Buffer *ptr;

    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferData(ctx, ptr, size, data, usage);
}

void mglNamedBufferData(GLMContext ctx, GLuint buffer, GLsizeiptr size, const void *data, GLenum usage)
{
    Buffer *ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferData(ctx, ptr, size, data, usage);
}

void mglBufferSubData(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr size, const void *data)
{
    Buffer *ptr;
    uint64_t t15 = mgl_prof_begin(15);

    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferSubData(ctx, ptr, offset, size, data);
    mgl_prof_end(15, t15);
}

void mglNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, const void *data)
{
    Buffer *ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferSubData(ctx, ptr, offset, size, data);
}

void copyBufferSubData(GLMContext ctx, Buffer *src_buf, Buffer *dst_buf, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    ERROR_CHECK_RETURN(readOffset >= 0 && writeOffset >= 0 && size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mgl_range_ok(readOffset, size, src_buf->size), GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mgl_range_ok(writeOffset, size, dst_buf->size), GL_INVALID_VALUE);

    if (src_buf == dst_buf)
    {
        // the source and destination ranges may not overlap
        uint64_t r0 = readOffset, w0 = writeOffset, n = size;
        if (n && r0 < w0 + n && w0 < r0 + n)
        {
            ERROR_RETURN(GL_INVALID_VALUE);
        }
    }

    ERROR_CHECK_RETURN(mappedNonPersistent(src_buf) == false, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(mappedNonPersistent(dst_buf) == false, GL_INVALID_OPERATION);

    if (size == 0)
        return;

    ERROR_CHECK_RETURN(src_buf->data.buffer_data && dst_buf->data.buffer_data, GL_INVALID_OPERATION);

    if (mtlext(ctx) && mtlext(ctx)->copy_buffer &&
        src_buf->data.mtl_data && dst_buf->data.mtl_data)
    {
        /* GPU blit path — only when both stores are already GPU-resident.  Promoting a
         * CPU-only store here (bindMTLBuffer's newBufferWithBytes + vm_deallocate) moves
         * the buffer's CPU address, which breaks anything still holding a pointer into
         * the old allocation (an unmap'ed mapping read back by the app, CTS
         * indirect_parameters_tests.ParameterBufferOperations) and can crash on the
         * deallocated pages.  The CPU path below keeps the address stable. */
        mtlext(ctx)->copy_buffer(ctx, src_buf, dst_buf, readOffset, writeOffset, size);
        return;
    }

    mgl_buffers_prepare_cpu_read(ctx, src_buf);
    mgl_buffers_prepare_cpu_write(ctx, dst_buf, writeOffset == 0 && size == dst_buf->size);

    memmove((char *)dst_buf->data.buffer_data + writeOffset, (const char *)src_buf->data.buffer_data + readOffset, size);

    bufferDidModify(ctx, dst_buf, writeOffset, size);
}

void mglCopyBufferSubData(GLMContext ctx, GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    Buffer *src_buf, *dst_buf;

    ERROR_CHECK_RETURN(checkTarget(ctx, readTarget), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(checkTarget(ctx, writeTarget), GL_INVALID_ENUM);

    src_buf = boundBuffer(ctx, readTarget);
    dst_buf = boundBuffer(ctx, writeTarget);

    ERROR_CHECK_RETURN(src_buf != NULL && dst_buf != NULL, GL_INVALID_OPERATION);

    copyBufferSubData(ctx, src_buf, dst_buf, readOffset, writeOffset, size);
}

void mglCopyNamedBufferSubData(GLMContext ctx, GLuint readBuffer, GLuint writeBuffer, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size)
{
    Buffer *src_buf = findBuffer(ctx, readBuffer);
    Buffer *dst_buf = findBuffer(ctx, writeBuffer);

    ERROR_CHECK_RETURN(src_buf != NULL && dst_buf != NULL, GL_INVALID_OPERATION);

    copyBufferSubData(ctx, src_buf, dst_buf, readOffset, writeOffset, size);
}

#pragma mark clear buffer (format conversion)

typedef enum { FMT_UNORM, FMT_FLOAT, FMT_SINT, FMT_UINT } FmtKind;

/* table 8.13 sized internal formats valid for buffer clears / texture buffers */
static bool clearInternalFormatInfo(GLenum internalformat, int *components, int *bytes_per_comp, FmtKind *kind)
{
    struct { GLenum f; int n; int b; FmtKind k; } table[] = {
        {GL_R8, 1, 1, FMT_UNORM},     {GL_R16, 1, 2, FMT_UNORM},     {GL_R16F, 1, 2, FMT_FLOAT},   {GL_R32F, 1, 4, FMT_FLOAT},
        {GL_R8I, 1, 1, FMT_SINT},     {GL_R16I, 1, 2, FMT_SINT},     {GL_R32I, 1, 4, FMT_SINT},
        {GL_R8UI, 1, 1, FMT_UINT},    {GL_R16UI, 1, 2, FMT_UINT},    {GL_R32UI, 1, 4, FMT_UINT},
        {GL_RG8, 2, 1, FMT_UNORM},    {GL_RG16, 2, 2, FMT_UNORM},    {GL_RG16F, 2, 2, FMT_FLOAT},  {GL_RG32F, 2, 4, FMT_FLOAT},
        {GL_RG8I, 2, 1, FMT_SINT},    {GL_RG16I, 2, 2, FMT_SINT},    {GL_RG32I, 2, 4, FMT_SINT},
        {GL_RG8UI, 2, 1, FMT_UINT},   {GL_RG16UI, 2, 2, FMT_UINT},   {GL_RG32UI, 2, 4, FMT_UINT},
        {GL_RGB32F, 3, 4, FMT_FLOAT}, {GL_RGB32I, 3, 4, FMT_SINT},   {GL_RGB32UI, 3, 4, FMT_UINT},
        {GL_RGBA8, 4, 1, FMT_UNORM},  {GL_RGBA16, 4, 2, FMT_UNORM},  {GL_RGBA16F, 4, 2, FMT_FLOAT}, {GL_RGBA32F, 4, 4, FMT_FLOAT},
        {GL_RGBA8I, 4, 1, FMT_SINT},  {GL_RGBA16I, 4, 2, FMT_SINT},  {GL_RGBA32I, 4, 4, FMT_SINT},
        {GL_RGBA8UI, 4, 1, FMT_UINT}, {GL_RGBA16UI, 4, 2, FMT_UINT}, {GL_RGBA32UI, 4, 4, FMT_UINT},
    };

    for (size_t i = 0; i < sizeof(table)/sizeof(table[0]); i++)
    {
        if (table[i].f == internalformat)
        {
            *components = table[i].n;
            *bytes_per_comp = table[i].b;
            *kind = table[i].k;
            return true;
        }
    }

    return false;
}

/* source pixel layout: component count, order (RGBA index per source component), integer-ness */
static bool clearSourceFormatInfo(GLenum format, int *n, int order[4], bool *integer)
{
    *integer = false;

    switch(format)
    {
        case GL_RED_INTEGER: *integer = true; /* fallthrough */
        case GL_RED: *n = 1; order[0] = 0; return true;
        case GL_GREEN_INTEGER: *integer = true; /* fallthrough */
        case GL_GREEN: *n = 1; order[0] = 1; return true;
        case GL_BLUE_INTEGER: *integer = true; /* fallthrough */
        case GL_BLUE: *n = 1; order[0] = 2; return true;
        case GL_RG_INTEGER: *integer = true; /* fallthrough */
        case GL_RG: *n = 2; order[0] = 0; order[1] = 1; return true;
        case GL_RGB_INTEGER: *integer = true; /* fallthrough */
        case GL_RGB: *n = 3; order[0] = 0; order[1] = 1; order[2] = 2; return true;
        case GL_BGR_INTEGER: *integer = true; /* fallthrough */
        case GL_BGR: *n = 3; order[0] = 2; order[1] = 1; order[2] = 0; return true;
        case GL_RGBA_INTEGER: *integer = true; /* fallthrough */
        case GL_RGBA: *n = 4; order[0] = 0; order[1] = 1; order[2] = 2; order[3] = 3; return true;
        case GL_BGRA_INTEGER: *integer = true; /* fallthrough */
        case GL_BGRA: *n = 4; order[0] = 2; order[1] = 1; order[2] = 0; order[3] = 3; return true;
    }

    return false;
}

static int sourceTypeSize(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2;
        case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4;
    }

    return 0;
}

static float halfToFloat(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

static uint16_t floatToHalf(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t  exp = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u & 0x7fffff;

    if (((v.u >> 23) & 0xff) == 0xff)   // inf / nan
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00);
    if (exp <= 0)
    {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1) half++;   // round
        return (uint16_t)(sign | half);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000) half++;   // round to nearest
    return (uint16_t)half;
}

/* read component i of the source pixel as a double (normalised for non-integer formats) */
static double readSourceComponent(const void *data, GLenum type, int i, bool normalize)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:  { double v = ((const uint8_t *)data)[i];  return normalize ? v / 255.0 : v; }
        case GL_BYTE:           { double v = ((const int8_t *)data)[i];   return normalize ? fmax(v / 127.0, -1.0) : v; }
        case GL_UNSIGNED_SHORT: { double v = ((const uint16_t *)data)[i]; return normalize ? v / 65535.0 : v; }
        case GL_SHORT:          { double v = ((const int16_t *)data)[i];  return normalize ? fmax(v / 32767.0, -1.0) : v; }
        case GL_UNSIGNED_INT:   { double v = ((const uint32_t *)data)[i]; return normalize ? v / 4294967295.0 : v; }
        case GL_INT:            { double v = ((const int32_t *)data)[i];  return normalize ? fmax(v / 2147483647.0, -1.0) : v; }
        case GL_HALF_FLOAT:     return halfToFloat(((const uint16_t *)data)[i]);
        case GL_FLOAT:          return ((const float *)data)[i];
    }

    return 0.0;
}

static void writeDestComponent(uint8_t *dst, FmtKind kind, int bytes, double v)
{
    switch(kind)
    {
        case FMT_UNORM:
        {
            double c = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
            if (isnan(v)) c = 0.0;
            if (bytes == 1) *dst = (uint8_t)lrint(c * 255.0);
            else { uint16_t s = (uint16_t)lrint(c * 65535.0); memcpy(dst, &s, 2); }
            break;
        }

        case FMT_FLOAT:
            if (bytes == 2) { uint16_t h = floatToHalf((float)v); memcpy(dst, &h, 2); }
            else { float f = (float)v; memcpy(dst, &f, 4); }
            break;

        case FMT_SINT:
        {
            double lo = bytes == 1 ? -128.0 : bytes == 2 ? -32768.0 : -2147483648.0;
            double hi = bytes == 1 ? 127.0 : bytes == 2 ? 32767.0 : 2147483647.0;
            double c = v < lo ? lo : (v > hi ? hi : v);
            int32_t i = (int32_t)c;
            if (bytes == 1) { int8_t b = (int8_t)i; memcpy(dst, &b, 1); }
            else if (bytes == 2) { int16_t s = (int16_t)i; memcpy(dst, &s, 2); }
            else memcpy(dst, &i, 4);
            break;
        }

        case FMT_UINT:
        {
            double hi = bytes == 1 ? 255.0 : bytes == 2 ? 65535.0 : 4294967295.0;
            double c = v < 0.0 ? 0.0 : (v > hi ? hi : v);
            uint32_t u = (uint32_t)c;
            if (bytes == 1) { uint8_t b = (uint8_t)u; memcpy(dst, &b, 1); }
            else if (bytes == 2) { uint16_t s = (uint16_t)u; memcpy(dst, &s, 2); }
            else memcpy(dst, &u, 4);
            break;
        }
    }
}

bool clearBufferData(GLMContext ctx, Buffer *ptr, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    int dst_n, dst_b;
    FmtKind kind;

    ERROR_CHECK_RETURN_VALUE(clearInternalFormatInfo(internalformat, &dst_n, &dst_b, &kind), GL_INVALID_ENUM, false);

    ERROR_CHECK_RETURN_VALUE(offset >= 0 && size >= 0, GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(mgl_range_ok(offset, size, ptr->size), GL_INVALID_VALUE, false);

    size_t texel = (size_t)dst_n * dst_b;
    ERROR_CHECK_RETURN_VALUE((offset % texel) == 0 && (size % texel) == 0, GL_INVALID_VALUE, false);

    int src_n, order[4];
    bool src_integer;
    ERROR_CHECK_RETURN_VALUE(clearSourceFormatInfo(format, &src_n, order, &src_integer), GL_INVALID_VALUE, false);
    ERROR_CHECK_RETURN_VALUE(sourceTypeSize(type) != 0, GL_INVALID_VALUE, false);

    bool dst_integer = (kind == FMT_SINT || kind == FMT_UINT);
    ERROR_CHECK_RETURN_VALUE(src_integer == dst_integer, GL_INVALID_OPERATION, false);
    ERROR_CHECK_RETURN_VALUE(!(src_integer && (type == GL_FLOAT || type == GL_HALF_FLOAT)), GL_INVALID_OPERATION, false);

    /* ClearBufferSubData uses the sub-data rule (range overlap); the whole-store
     * ClearBufferData callers pass [0, size) so they keep the blanket error */
    ERROR_CHECK_RETURN_VALUE(!mappedRangeBlocked(ptr, offset, size), GL_INVALID_OPERATION, false);

    if (size == 0)
        return true;

    ERROR_CHECK_RETURN_VALUE(ptr->data.buffer_data, GL_INVALID_OPERATION, false);

    // convert the clear value into one texel of internalformat
    uint8_t texel_bytes[16];
    bzero(texel_bytes, sizeof(texel_bytes));

    if (data)
    {
        double rgba[4] = { 0.0, 0.0, 0.0, dst_integer ? 1.0 : 1.0 };
        for (int i = 0; i < src_n; i++)
            rgba[order[i]] = readSourceComponent(data, type, i, !src_integer);

        for (int c = 0; c < dst_n; c++)
            writeDestComponent(&texel_bytes[c * dst_b], kind, dst_b, rgba[c]);
    }

    /* gpuaudit: the clear itself runs on the GPU (blit fill / staged copies) — the CPU
     * never walks the range, so it must not be made to wait for (or renamed because of)
     * queued GPU work either: the fill is encoded on the same queue and ordered after
     * it.  Only buffers that are already GPU-resident take this path: promoting a
     * CPU-only buffer would mean a full-copy allocation that costs more than the clear
     * it saves. */
    struct MGLExtMetal_gpuaudit *gpm = ctx->mtl_funcs.ext_gpuaudit;
    if (gpm && gpm->fill_buffer && ptr->data.mtl_data &&
        gpm->fill_buffer(ctx, ptr, offset, size, texel_bytes, (int)texel))
    {
        mgl_gpuaudit_invalidate(ctx, ptr);   /* contents changed */
        return true;
    }

    mgl_buffers_prepare_cpu_write(ctx, ptr, offset == 0 && size == ptr->size);

    uint8_t *dst = (uint8_t *)ptr->data.buffer_data + offset;
    for (size_t i = 0; i < (size_t)size; i += texel)
        memcpy(dst + i, texel_bytes, texel);

    bufferDidModify(ctx, ptr, offset, size);

    return true;
}

void mglClearBufferData(GLMContext ctx, GLenum target, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    clearBufferData(ctx, ptr, internalformat, 0, ptr->size, format, type, data);
}

void mglClearBufferSubData(GLMContext ctx, GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    clearBufferData(ctx, ptr, internalformat, offset, size, format, type, data);
}

void mglClearNamedBufferData(GLMContext ctx, GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void *data)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    clearBufferData(ctx, ptr, internalformat, 0, ptr->size, format, type, data);
}

void mglClearNamedBufferSubData(GLMContext ctx, GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type, const void *data)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    clearBufferData(ctx, ptr, internalformat, offset, size, format, type, data);
}

#pragma mark GL Buffer Map Functions

static void *mapBufferRange(GLMContext ctx, Buffer *ptr, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    ERROR_CHECK_RETURN_VALUE(offset >= 0 && length >= 0, GL_INVALID_VALUE, NULL);
    ERROR_CHECK_RETURN_VALUE(mgl_range_ok(offset, length, ptr->size), GL_INVALID_VALUE, NULL);
    ERROR_CHECK_RETURN_VALUE((access & ~(GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT |
                                         GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
                                         GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_UNSYNCHRONIZED_BIT)) == 0,
                             GL_INVALID_VALUE, NULL);

    /* GL 4.5+ (p.94) and ES 3.x: zero-length MapBufferRange is INVALID_OPERATION */
    ERROR_CHECK_RETURN_VALUE(length != 0, GL_INVALID_OPERATION, NULL);

    /* a buffer may have only one mapping at a time (INVALID_OPERATION) */
    ERROR_CHECK_RETURN_VALUE(ptr->mapped == GL_FALSE, GL_INVALID_OPERATION, NULL);

    ERROR_CHECK_RETURN_VALUE(access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT), GL_INVALID_OPERATION, NULL);

    if ((access & GL_MAP_READ_BIT) &&
        (access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT)))
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    if ((access & GL_MAP_FLUSH_EXPLICIT_BIT) && !(access & GL_MAP_WRITE_BIT))
    {
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, NULL);
    }

    // every requested capability must be part of the storage flags
    GLbitfield caps = access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    ERROR_CHECK_RETURN_VALUE((ptr->storage_flags & caps) == caps, GL_INVALID_OPERATION, NULL);

    ERROR_CHECK_RETURN_VALUE(ptr->data.buffer_data, GL_OUT_OF_MEMORY, NULL);

    if (!(access & GL_MAP_UNSYNCHRONIZED_BIT))
    {
        // reads must observe completed GPU writes, writes must not race queued GPU reads
        if (access & GL_MAP_READ_BIT)
            mgl_buffers_prepare_cpu_read(ctx, ptr);

        if (access & GL_MAP_WRITE_BIT)
            mgl_buffers_prepare_cpu_write(ctx, ptr,
                                          (access & GL_MAP_INVALIDATE_BUFFER_BIT) ||
                                          ((access & GL_MAP_INVALIDATE_RANGE_BIT) && offset == 0 && length == ptr->size));
    }

    // a persistent mapping keeps its address; the GPU may access it while mapped
    if ((access & GL_MAP_PERSISTENT_BIT) && mtlext(ctx) && mtlext(ctx)->ensure_buffer)
        mtlext(ctx)->ensure_buffer(ctx, ptr);

    ptr->mapped = GL_TRUE;
    ptr->access_flags = access;
    ptr->access = accessFromFlags(access);
    ptr->mapped_offset = offset;
    ptr->mapped_length = length;

    if ((access & GL_MAP_READ_BIT) && getenv("MGL_TESS_DUMP_CAP"))
    {
        /* one-off diagnostics: the first interleaved xfb records (chasing the CTS
         * point-mode pointsize mismatch) */
        const float *f = (const float *)(void *)(vm_address_t)ptr->data.buffer_data;

        for (int r = 0; r < 4 && (size_t)(r * 11 + 10) * 4 < (size_t)ptr->size; r++)
            fprintf(stderr, "tfmap: rec %d: %.4f %.4f %.4f %.4f | %.4f %.4f | %d %d %d %d | ps=%.4f\n",
                    r, f[r * 11 + 0], f[r * 11 + 1], f[r * 11 + 2], f[r * 11 + 3],
                    f[r * 11 + 4], f[r * 11 + 5],
                    (int)f[r * 11 + 6], (int)f[r * 11 + 7], (int)f[r * 11 + 8], (int)f[r * 11 + 9],
                    f[r * 11 + 10]);
    }

    return (char *)ptr->data.buffer_data + offset;
}

static bool legacyAccessToFlags(GLenum access, GLbitfield *flags)
{
    switch(access)
    {
        case GL_READ_ONLY: *flags = GL_MAP_READ_BIT; return true;
        case GL_WRITE_ONLY: *flags = GL_MAP_WRITE_BIT; return true;
        case GL_READ_WRITE: *flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT; return true;
    }

    return false;
}

void *mglMapBuffer(GLMContext ctx, GLenum target, GLenum access)
{
    GLbitfield flags;

    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, NULL);
    ERROR_CHECK_RETURN_VALUE(legacyAccessToFlags(access, &flags), GL_INVALID_ENUM, NULL);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, NULL);
    ERROR_CHECK_RETURN_VALUE(ptr->size > 0, GL_INVALID_OPERATION, NULL);

    void *p = mapBufferRange(ctx, ptr, 0, ptr->size, flags);
    if (p)
        ptr->access = access;

    return p;
}

void *mglMapNamedBuffer(GLMContext ctx, GLuint buffer, GLenum access)
{
    GLbitfield flags;

    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, NULL);
    ERROR_CHECK_RETURN_VALUE(legacyAccessToFlags(access, &flags), GL_INVALID_ENUM, NULL);
    ERROR_CHECK_RETURN_VALUE(ptr->size > 0, GL_INVALID_OPERATION, NULL);

    void *p = mapBufferRange(ctx, ptr, 0, ptr->size, flags);
    if (p)
        ptr->access = access;

    return p;
}

static void *mglMapBufferRangeInner(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access_flags);
static GLboolean mglUnmapBufferInner(GLMContext ctx, GLenum target);
void *mglMapBufferRange(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access_flags)
{
    uint64_t t20 = mgl_prof_begin(20);
    void *r = mglMapBufferRangeInner(ctx, target, offset, length, access_flags);
    mgl_prof_end(20, t20);
    return r;
}
static void *mglMapBufferRangeInner(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access_flags)
{
    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, NULL);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, NULL);

    return mapBufferRange(ctx, ptr, offset, length, access_flags);
}

void *mglMapNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, NULL);

    return mapBufferRange(ctx, ptr, offset, length, access);
}

static GLboolean unmapBuffer(GLMContext ctx, Buffer *ptr)
{
    ERROR_CHECK_RETURN_VALUE(ptr->mapped, GL_INVALID_OPERATION, GL_FALSE);

    unmapBufferInternal(ctx, ptr);

    // the data store can not become corrupt on this platform (no video memory loss)
    return GL_TRUE;
}

GLboolean mglUnmapBuffer(GLMContext ctx, GLenum target)
{
    uint64_t t21 = mgl_prof_begin(21);
    GLboolean r = mglUnmapBufferInner(ctx, target);
    mgl_prof_end(21, t21);
    return r;
}
static GLboolean mglUnmapBufferInner(GLMContext ctx, GLenum target)
{
    ERROR_CHECK_RETURN_VALUE(checkTarget(ctx, target), GL_INVALID_ENUM, GL_FALSE);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, GL_FALSE);

    return unmapBuffer(ctx, ptr);
}

GLboolean mglUnmapNamedBuffer(GLMContext ctx, GLuint buffer)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_INVALID_OPERATION, GL_FALSE);

    return unmapBuffer(ctx, ptr);
}

static void flushMappedRange(GLMContext ctx, Buffer *ptr, GLintptr offset, GLsizeiptr length)
{
    ERROR_CHECK_RETURN(offset >= 0 && length >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(ptr->mapped, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(ptr->access_flags & GL_MAP_FLUSH_EXPLICIT_BIT, GL_INVALID_OPERATION);
    // offset / length are relative to the mapped range
    ERROR_CHECK_RETURN(mgl_range_ok(offset, length, ptr->mapped_length), GL_INVALID_VALUE);

    if (length)
        bufferDidModify(ctx, ptr, ptr->mapped_offset + offset, length);
}

void mglFlushMappedBufferRange(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr length)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    flushMappedRange(ctx, ptr, offset, length);
}

void mglFlushMappedNamedBufferRange(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    flushMappedRange(ctx, ptr, offset, length);
}

#pragma mark GL Buffer Storage Functions

static void bufferStorageChecked(GLMContext ctx, Buffer *ptr, GLenum target, GLsizeiptr size, const void *data, GLbitfield flags)
{
    if (!checkStorageFlags(ctx, size, flags))
        return;

    ERROR_CHECK_RETURN(ptr->immutable_storage == GL_FALSE, GL_INVALID_OPERATION);

    GLuint index = target ? bufferIndexFromTarget(ctx, target) : ptr->index;
    if (index == MGL_BUFFER_NO_INDEX)
        index = 0;

    bufferStorage(ctx, ptr, target, index, size, data, flags, 0);

    // client storage: keep the data in the vm allocation, wrapped without copy
    if ((flags & GL_CLIENT_STORAGE_BIT) == 0 && (flags & GL_MAP_PERSISTENT_BIT) &&
        mtlext(ctx) && mtlext(ctx)->ensure_buffer)
    {
        // persistent buffers get their final (stable) address up front
        mtlext(ctx)->ensure_buffer(ctx, ptr);
    }
}

void mglBufferStorage(GLMContext ctx, GLenum target, GLsizeiptr size, const void *data, GLbitfield storage_flags)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferStorageChecked(ctx, ptr, target, size, data, storage_flags);
}

void mglNamedBufferStorage(GLMContext ctx, GLuint buffer, GLsizeiptr size, const void *data, GLbitfield storage_flags)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferStorageChecked(ctx, ptr, 0, size, data, storage_flags);
}

void mglInvalidateBufferData(GLMContext ctx, GLuint buffer)
{
    Buffer *ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mappedNonPersistent(ptr) == false, GL_INVALID_OPERATION);

    // contents become undefined: detach from queued GPU work instead of waiting for it
    mgl_buffers_prepare_cpu_write(ctx, ptr, true);
}

void mglInvalidateBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr length)
{
    Buffer *ptr = findBuffer(ctx, buffer);

    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(offset >= 0 && length >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mgl_range_ok(offset, length, ptr->size), GL_INVALID_VALUE);

    if (mappedNonPersistent(ptr))
    {
        // only an overlap with the mapped range is an error
        bool overlap = offset < ptr->mapped_offset + ptr->mapped_length && ptr->mapped_offset < offset + length;
        ERROR_CHECK_RETURN(!overlap, GL_INVALID_OPERATION);
    }

    // invalidating the whole store lets queued GPU work keep the old copy
    if (offset == 0 && length == ptr->size && !ptr->mapped)
        mgl_buffers_prepare_cpu_write(ctx, ptr, true);
}

#pragma mark GL Buffer Get Functions

static bool bufferParameter(GLMContext ctx, Buffer *ptr, GLenum pname, GLint64 *value)
{
    switch(pname)
    {
        case GL_BUFFER_ACCESS:            *value = ptr->access; return true;
        case GL_BUFFER_ACCESS_FLAGS:      *value = ptr->access_flags; return true;
        case GL_BUFFER_IMMUTABLE_STORAGE: *value = ptr->immutable_storage ? GL_TRUE : GL_FALSE; return true;
        case GL_BUFFER_MAPPED:            *value = ptr->mapped ? GL_TRUE : GL_FALSE; return true;
        case GL_BUFFER_MAP_LENGTH:        *value = ptr->mapped_length; return true;
        case GL_BUFFER_MAP_OFFSET:        *value = ptr->mapped_offset; return true;
        case GL_BUFFER_SIZE:              *value = ptr->size; return true;
        case GL_BUFFER_STORAGE_FLAGS:     *value = ptr->storage_flags; return true;
        case GL_BUFFER_USAGE:             *value = ptr->usage; return true;
    }

    ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
}

static GLint clampToInt(GLint64 v)
{
    if (v > INT_MAX) return INT_MAX;
    if (v < INT_MIN) return INT_MIN;
    return (GLint)v;
}

void mglGetBufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    GLint64 v;

    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    if (bufferParameter(ctx, ptr, pname, &v) && params)
        *params = clampToInt(v);
}

void mglGetBufferParameteri64v(GLMContext ctx, GLenum target, GLenum pname, GLint64 *params)
{
    GLint64 v;

    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    if (bufferParameter(ctx, ptr, pname, &v) && params)
        *params = v;
}

void mglGetNamedBufferParameteriv(GLMContext ctx, GLuint buffer, GLenum pname, GLint *params)
{
    GLint64 v;

    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    if (bufferParameter(ctx, ptr, pname, &v) && params)
        *params = clampToInt(v);
}

void mglGetNamedBufferParameteri64v(GLMContext ctx, GLuint buffer, GLenum pname, GLint64 *params)
{
    GLint64 v;

    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    if (bufferParameter(ctx, ptr, pname, &v) && params)
        *params = v;
}

static void bufferPointer(GLMContext ctx, Buffer *ptr, GLenum pname, void **params)
{
    ERROR_CHECK_RETURN(pname == GL_BUFFER_MAP_POINTER, GL_INVALID_ENUM);

    if (!params)
        return;

    if (ptr->mapped && ptr->data.buffer_data)
        *params = (char *)ptr->data.buffer_data + ptr->mapped_offset;
    else
        *params = NULL;
}

void mglGetBufferPointerv(GLMContext ctx, GLenum target, GLenum pname, void **params)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(pname == GL_BUFFER_MAP_POINTER, GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferPointer(ctx, ptr, pname, params);
}

void mglGetNamedBufferPointerv(GLMContext ctx, GLuint buffer, GLenum pname, void **params)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    bufferPointer(ctx, ptr, pname, params);
}

static void getBufferSubData(GLMContext ctx, Buffer *ptr, GLintptr offset, GLsizeiptr size, void *data)
{
    ERROR_CHECK_RETURN(offset >= 0 && size >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mgl_range_ok(offset, size, ptr->size), GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(mappedNonPersistent(ptr) == false, GL_INVALID_OPERATION);

    if (size == 0 || data == NULL)
        return;

    ERROR_CHECK_RETURN(ptr->data.buffer_data, GL_INVALID_OPERATION);

    // waits for queued GPU writes (compute / SSBO / xfb / copies) to land
    mgl_buffers_prepare_cpu_read(ctx, ptr);

    memcpy(data, (const uint8_t *)ptr->data.buffer_data + offset, (size_t)size);
}

void mglGetBufferSubData(GLMContext ctx, GLenum target, GLintptr offset, GLsizeiptr size, void *data)
{
    ERROR_CHECK_RETURN(checkTarget(ctx, target), GL_INVALID_ENUM);

    Buffer *ptr = boundBuffer(ctx, target);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    getBufferSubData(ctx, ptr, offset, size, data);
}

void mglGetNamedBufferSubData(GLMContext ctx, GLuint buffer, GLintptr offset, GLsizeiptr size, void *data)
{
    Buffer *ptr = findBuffer(ctx, buffer);
    ERROR_CHECK_RETURN(ptr != NULL, GL_INVALID_OPERATION);

    getBufferSubData(ctx, ptr, offset, size, data);
}
