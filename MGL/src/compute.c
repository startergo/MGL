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
 * compute.c
 * MGL
 *
 * glDispatchCompute / glDispatchComputeIndirect. The Metal side
 * (MGLRenderer+Draw.m) encodes a compute pass with dispatchThreadgroups /
 * dispatchThreadgroupsWithIndirectBuffer using the shader's local size.
 */

#include "glm_context.h"
#include "mgl_ext_buffers.h"
#include "mgl_buffers_err.h"
#include "mgl_ext_queries.h"

static bool computeProgramActive(GLMContext ctx)
{
    if (ctx->state.program)
        return ctx->state.program->shader_slots[_COMPUTE_SHADER] != NULL;

    ProgramPipeline *pp = ctx->state.program_pipeline;
    if (pp && pp->stage_programs[_COMPUTE_SHADER])
        return pp->stage_programs[_COMPUTE_SHADER]->shader_slots[_COMPUTE_SHADER] != NULL;

    return false;
}

void mglDispatchCompute(GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z)
{
    ERROR_CHECK_RETURN(num_groups_x <= (GLuint)ctx->state.var.max_compute_work_group_count[0], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_y <= (GLuint)ctx->state.var.max_compute_work_group_count[1], GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(num_groups_z <= (GLuint)ctx->state.var.max_compute_work_group_count[2], GL_INVALID_VALUE);

    // GL_INVALID_OPERATION if there is no active program for the compute shader stage
    ERROR_CHECK_RETURN(computeProgramActive(ctx), GL_INVALID_OPERATION);

    // an empty grid is legal and does nothing
    if (num_groups_x == 0 || num_groups_y == 0 || num_groups_z == 0)
        return;

    ERROR_CHECK_RETURN(ctx->mtl_funcs.ext_buffers && ctx->mtl_funcs.ext_buffers->dispatch, GL_INVALID_OPERATION);

    if (!ctx->mtl_funcs.ext_buffers->dispatch(ctx, num_groups_x, num_groups_y, num_groups_z, NULL, 0))
        return;

    // GL_COMPUTE_SHADER_INVOCATIONS (queries domain)
    if (ctx->state.program)
        mgl_queries_note_dispatch(ctx, (uint64_t)num_groups_x * num_groups_y * num_groups_z *
                                  ctx->state.program->local_workgroup_size.x *
                                  ctx->state.program->local_workgroup_size.y *
                                  ctx->state.program->local_workgroup_size.z);
}

void mglDispatchComputeIndirect(GLMContext ctx, GLintptr indirect)
{
    // GL_INVALID_VALUE if indirect is negative or not a multiple of four
    ERROR_CHECK_RETURN(indirect >= 0 && (indirect % 4) == 0, GL_INVALID_VALUE);

    Buffer *buf = ctx->state.buffers[_DISPATCH_INDIRECT_BUFFER];

    // GL_INVALID_OPERATION if no buffer is bound, the command is out of range or the buffer is mapped
    ERROR_CHECK_RETURN(buf != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN((uint64_t)indirect + 3 * sizeof(GLuint) <= (uint64_t)buf->size, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!(buf->mapped && !(buf->access_flags & GL_MAP_PERSISTENT_BIT)), GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(buf->data.buffer_data, GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(computeProgramActive(ctx), GL_INVALID_OPERATION);

    ERROR_CHECK_RETURN(ctx->mtl_funcs.ext_buffers && ctx->mtl_funcs.ext_buffers->dispatch, GL_INVALID_OPERATION);

    // group counts are read by the GPU (values above the limits are undefined behaviour in GL)
    ctx->mtl_funcs.ext_buffers->dispatch(ctx, 0, 0, 0, buf, (size_t)indirect);
}
