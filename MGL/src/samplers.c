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
 * samplers.c
 * MGL
 *
 * Sampler objects. A sampler bound to unit N replaces the sampler state of the
 * texture bound to unit N at draw time (MGLRenderer+TextureParams.m,
 * -texparamSamplerForUnit:texture:). Like Mesa, glGenSamplers creates the
 * objects immediately so glSamplerParameter works on a generated-but-unbound name.
 */

#include <strings.h>
#include "glm_context.h"
#include "mgl_ext_texparam.h"
#include "tex_param_internal.h"

#define SP_ERROR(_e_) do { ERROR_RETURN(_e_); return; } while (0)

Sampler *newSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    ptr = (Sampler *)calloc(1, sizeof(Sampler));
    if (!ptr)
        return NULL;

    ptr->name = sampler;
    mglTexParamInitDefaults(&ptr->params, 0);

    return ptr;
}

Sampler *findSampler(GLMContext ctx, GLuint sampler)
{
    if (sampler == 0)
        return NULL;

    return (Sampler *)searchHashTable(&STATE(sampler_table), sampler);
}

Sampler *getSampler(GLMContext ctx, GLuint sampler)
{
    Sampler *ptr;

    ptr = findSampler(ctx, sampler);

    if (!ptr && sampler)
    {
        ptr = newSampler(ctx, sampler);

        if (ptr)
            insertHashElement(&STATE(sampler_table), sampler, ptr);
    }

    return ptr;
}

bool isSampler(GLMContext ctx, GLuint sampler)
{
    return findSampler(ctx, sampler) != NULL;
}

static GLuint sp_max_units(GLMContext ctx)
{
    GLuint max = STATE_VAR(max_combined_texture_image_units);

    if (max == 0 || max > TEXTURE_UNITS)
        max = TEXTURE_UNITS;

    return max;
}

static void sp_release_mtl(GLMContext ctx, Sampler *ptr)
{
    if (ptr->mtl_data && ctx->mtl_funcs.ext_texparam && ctx->mtl_funcs.ext_texparam->releaseObj)
    {
        ctx->mtl_funcs.ext_texparam->releaseObj(ctx, ptr->mtl_data);
        ptr->mtl_data = NULL;
    }

    mglTexParamInvalidate(ctx, &ptr->params);
}

GLboolean mglIsSampler(GLMContext ctx, GLuint sampler)
{
    return isSampler(ctx, sampler) ? GL_TRUE : GL_FALSE;
}

void mglGenSamplers(GLMContext ctx, GLsizei count, GLuint *samplers)
{
    if (count < 0)
        SP_ERROR(GL_INVALID_VALUE);

    if (count == 0 || !samplers)
        return;

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint name;

        // skip names already in use (e.g. after a name was bound explicitly)
        do {
            name = getNewName(&ctx->state.sampler_table);
        } while (findSampler(ctx, name));

        if (!getSampler(ctx, name))
            SP_ERROR(GL_OUT_OF_MEMORY);

        samplers[i] = name;
    }
}

void mglCreateSamplers(GLMContext ctx, GLsizei n, GLuint *samplers)
{
    mglGenSamplers(ctx, n, samplers);
}

static bool sp_bind(GLMContext ctx, GLuint unit, GLuint sampler)
{
    Sampler *ptr = NULL;

    if (sampler)
    {
        ptr = findSampler(ctx, sampler);
        if (!ptr)
            return false;
    }

    if (ctx->state.texture_samplers[unit] != ptr)
    {
        ctx->state.texture_samplers[unit] = ptr;
        ctx->state.dirty_bits |= DIRTY_SAMPLER;
    }

    return true;
}

void mglBindSampler(GLMContext ctx, GLuint unit, GLuint sampler)
{
    if (unit >= sp_max_units(ctx))
        SP_ERROR(GL_INVALID_VALUE);

    if (!sp_bind(ctx, unit, sampler))
        SP_ERROR(GL_INVALID_OPERATION);
}

void mglBindSamplers(GLMContext ctx, GLuint first, GLsizei count, const GLuint *samplers)
{
    bool failed = false;

    if (count < 0)
        SP_ERROR(GL_INVALID_VALUE);

    if ((GLuint64)first + (GLuint64)count > sp_max_units(ctx))
        SP_ERROR(GL_INVALID_OPERATION);

    // each failing entry is skipped, the rest are still bound
    for (GLsizei i = 0; i < count; i++)
    {
        GLuint name = samplers ? samplers[i] : 0;

        if (!sp_bind(ctx, first + i, name))
            failed = true;
    }

    if (failed)
        ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglDeleteSamplers(GLMContext ctx, GLsizei count, const GLuint *samplers)
{
    if (count < 0)
        SP_ERROR(GL_INVALID_VALUE);

    if (!samplers)
        return;

    for (GLsizei n = 0; n < count; n++)
    {
        GLuint sampler = samplers[n];
        Sampler *ptr = findSampler(ctx, sampler);

        // unused names and 0 are silently ignored
        if (!ptr)
            continue;

        for (int i = 0; i < TEXTURE_UNITS; i++)
        {
            if (ctx->state.texture_samplers[i] == ptr)
            {
                ctx->state.texture_samplers[i] = NULL;
                ctx->state.dirty_bits |= DIRTY_SAMPLER;
            }
        }

        deleteHashElement(&ctx->state.sampler_table, sampler);

        sp_release_mtl(ctx, ptr);

        free(ptr);
    }
}

#pragma mark sampler parameters

static void sp_apply(GLMContext ctx, GLuint sampler, GLenum pname, const void *v, int vt, bool vector)
{
    Sampler *ptr;
    bool changed;
    GLenum err;

    ptr = findSampler(ctx, sampler);
    if (!ptr)
        SP_ERROR(GL_INVALID_OPERATION);

    if (!v)
        SP_ERROR(GL_INVALID_VALUE);

    err = mglTexParamSet(&ptr->params, 0, true, pname, v, vt, vector, &changed, false);
    if (err != GL_NO_ERROR)
        SP_ERROR(err);

    if (changed)
    {
        sp_release_mtl(ctx, ptr);
        ptr->dirty_bits |= DIRTY_SAMPLER_PARAM;
        ctx->state.dirty_bits |= DIRTY_SAMPLER;
    }
}

static void sp_query(GLMContext ctx, GLuint sampler, GLenum pname, void *out, int vt)
{
    Sampler *ptr;
    GLenum err;

    ptr = findSampler(ctx, sampler);
    if (!ptr)
        SP_ERROR(GL_INVALID_OPERATION);

    if (!out)
        SP_ERROR(GL_INVALID_VALUE);

    err = mglTexParamGet(ctx, NULL, &ptr->params, true, pname, out, vt);
    if (err != GL_NO_ERROR)
        ERROR_RETURN(err);
}

void mglSamplerParameterf(GLMContext ctx, GLuint sampler, GLenum pname, GLfloat param)
{
    sp_apply(ctx, sampler, pname, &param, TP_FLOAT, false);
}

void mglSamplerParameterfv(GLMContext ctx, GLuint sampler, GLenum pname, const GLfloat *param)
{
    sp_apply(ctx, sampler, pname, param, TP_FLOAT, true);
}

void mglSamplerParameteri(GLMContext ctx, GLuint sampler, GLenum pname, GLint param)
{
    sp_apply(ctx, sampler, pname, &param, TP_INT, false);
}

void mglSamplerParameteriv(GLMContext ctx, GLuint sampler, GLenum pname, const GLint *param)
{
    sp_apply(ctx, sampler, pname, param, TP_INT, true);
}

void mglSamplerParameterIiv(GLMContext ctx, GLuint sampler, GLenum pname, const GLint *param)
{
    sp_apply(ctx, sampler, pname, param, TP_PURE_INT, true);
}

void mglSamplerParameterIuiv(GLMContext ctx, GLuint sampler, GLenum pname, const GLuint *param)
{
    sp_apply(ctx, sampler, pname, param, TP_UINT, true);
}

void mglGetSamplerParameterIiv(GLMContext ctx, GLuint sampler, GLenum pname, GLint *params)
{
    sp_query(ctx, sampler, pname, params, TP_PURE_INT);
}

void mglGetSamplerParameterIuiv(GLMContext ctx, GLuint sampler, GLenum pname, GLuint *params)
{
    sp_query(ctx, sampler, pname, params, TP_UINT);
}

void mglGetSamplerParameterfv(GLMContext ctx, GLuint sampler, GLenum pname, GLfloat *params)
{
    sp_query(ctx, sampler, pname, params, TP_FLOAT);
}

void mglGetSamplerParameteriv(GLMContext ctx, GLuint sampler, GLenum pname, GLint *params)
{
    sp_query(ctx, sampler, pname, params, TP_INT);
}
