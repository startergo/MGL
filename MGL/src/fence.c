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
 * fence.c
 * MGL
 *
 * Sync objects and memory barriers. A fence is an MTLSharedEvent signal encoded
 * into the command stream (MGLRenderer+Draw.m); Sync.mtl_event points at the
 * renderer's fence record.
 */

#include <strings.h>

#include "glm_context.h"
#include "mgl_ext_buffers.h"
#include "mgl_buffers_err.h"

#define MTLX() (ctx->mtl_funcs.ext_buffers)

Sync *newSync(GLMContext ctx)
{
    Sync *ptr = (Sync *)calloc(1, sizeof(Sync));

    if (!ptr)
        return NULL;

    ptr->name = ++STATE(sync_name);

    return ptr;
}

int isSync(GLMContext ctx, GLsync sync)
{
    return mgl_buffers_sync_valid(ctx, (Sync *)sync) ? 1 : 0;
}

GLsync mglFenceSync(GLMContext ctx, GLenum condition, GLbitfield flags)
{
    ERROR_CHECK_RETURN_VALUE(condition == GL_SYNC_GPU_COMMANDS_COMPLETE, GL_INVALID_ENUM, NULL);
    ERROR_CHECK_RETURN_VALUE(flags == 0, GL_INVALID_VALUE, NULL);

    Sync *ptr = newSync(ctx);
    ERROR_CHECK_RETURN_VALUE(ptr != NULL, GL_OUT_OF_MEMORY, NULL);

    MTLX()->fence(ctx, ptr);
    mgl_buffers_sync_add(ctx, ptr);

    return ptr;
}

GLboolean mglIsSync(GLMContext ctx, GLsync sync)
{
    return isSync(ctx, sync) ? GL_TRUE : GL_FALSE;
}

void mglDeleteSync(GLMContext ctx, GLsync sync)
{
    // zero is silently ignored
    if (sync == NULL)
        return;

    ERROR_CHECK_RETURN(isSync(ctx, sync), GL_INVALID_VALUE);

    mgl_buffers_sync_remove(ctx, sync);
    MTLX()->delete_fence(ctx, sync);

    free(sync);
}

GLenum mglClientWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    ERROR_CHECK_RETURN_VALUE(isSync(ctx, sync), GL_INVALID_VALUE, GL_WAIT_FAILED);
    ERROR_CHECK_RETURN_VALUE((flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) == 0, GL_INVALID_VALUE, GL_WAIT_FAILED);

    return MTLX()->client_wait(ctx, sync, (flags & GL_SYNC_FLUSH_COMMANDS_BIT) != 0, timeout);
}

/* r182: see mglSyncNotify (MGLContext.h) */
GLboolean mglSyncNotifyCtx(GLMContext ctx, GLsync sync, void (*fn)(void *), void *arg)
{
    ERROR_CHECK_RETURN_VALUE(isSync(ctx, sync), GL_INVALID_VALUE, GL_FALSE);
    ERROR_CHECK_RETURN_VALUE(fn != NULL, GL_INVALID_VALUE, GL_FALSE);

    MTLX()->notify(ctx, (Sync *)sync, fn, arg);

    return GL_TRUE;
}

void mglWaitSync(GLMContext ctx, GLsync sync, GLbitfield flags, GLuint64 timeout)
{
    ERROR_CHECK_RETURN(isSync(ctx, sync), GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(flags == 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(timeout == GL_TIMEOUT_IGNORED, GL_INVALID_VALUE);

    // the GPU waits for the fence before executing later commands
    MTLX()->server_wait(ctx, sync);
}

void mglGetSynciv(GLMContext ctx, GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values)
{
    GLint value;

    ERROR_CHECK_RETURN(isSync(ctx, sync), GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    switch(pname)
    {
        case GL_OBJECT_TYPE:
            value = GL_SYNC_FENCE;
            break;

        case GL_SYNC_STATUS:
            value = MTLX()->is_signaled(ctx, sync) ? GL_SIGNALED : GL_UNSIGNALED;
            break;

        case GL_SYNC_CONDITION:
            value = GL_SYNC_GPU_COMMANDS_COMPLETE;
            break;

        case GL_SYNC_FLAGS:
            value = 0;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    GLsizei written = 0;

    if (count > 0 && values)
    {
        values[0] = value;
        written = 1;
    }

    if (length)
        *length = written;
}

#define MGL_ALL_BARRIERS (GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT | \
                          GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_COMMAND_BARRIER_BIT | \
                          GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT | \
                          GL_FRAMEBUFFER_BARRIER_BIT | GL_TRANSFORM_FEEDBACK_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT | \
                          GL_SHADER_STORAGE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT | GL_QUERY_BUFFER_BARRIER_BIT)

#define MGL_REGION_BARRIERS (GL_ATOMIC_COUNTER_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | \
                             GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT)

void mglMemoryBarrier(GLMContext ctx, GLbitfield barriers)
{
    ERROR_CHECK_RETURN(barriers == GL_ALL_BARRIER_BITS || (barriers & ~MGL_ALL_BARRIERS) == 0, GL_INVALID_VALUE);

    if (barriers)
        MTLX()->barrier(ctx, barriers, false);
}

void mglMemoryBarrierByRegion(GLMContext ctx, GLbitfield barriers)
{
    ERROR_CHECK_RETURN(barriers == GL_ALL_BARRIER_BITS || (barriers & ~MGL_REGION_BARRIERS) == 0, GL_INVALID_VALUE);

    if (barriers)
        MTLX()->barrier(ctx, barriers, true);
}
