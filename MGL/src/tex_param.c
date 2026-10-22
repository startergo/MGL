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
 * tex_param.c
 * MGL
 *
 * Texture parameters (glTexParameter* / glTextureParameter* / glGetTex*Parameter*)
 * and the shared parameter core used by sampler objects (samplers.c).
 *
 * Storage: TextureParameter (glm_context.h). The border color is kept in all
 * three representations (float / int / uint) so every query type returns a
 * defined value. Any change drops the cached Metal sampler/view state
 * (params->mtl_data) and marks DIRTY_TEX so the renderer rebinds.
 */

#include <math.h>
#include <string.h>
#include <strings.h>
#include <limits.h>
#include "glm_context.h"
#include "mgl_ext_texparam.h"
#include "tex_param_internal.h"

extern GLuint textureIndexFromTarget(GLMContext ctx, GLenum target);
extern Texture *currentTexture(GLMContext ctx, GLuint index);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
Texture *getTex(GLMContext ctx, GLuint name, GLenum target);

#define TP_ERROR(_e_) do { ERROR_RETURN(_e_); return; } while (0)

#pragma mark defaults

void mglTexParamInitDefaults(TextureParameter *p, GLenum target)
{
    void *mtl = p->mtl_data;

    bzero(p, sizeof(TextureParameter));
    p->mtl_data = mtl;

    p->depth_stencil_mode = GL_DEPTH_COMPONENT;
    p->base_level = 0;
    p->max_level = 1000;
    p->compare_func = GL_LEQUAL;
    p->compare_mode = GL_NONE;
    p->lod_bias = 0.0f;
    p->min_lod = -1000.0f;
    p->max_lod = 1000.0f;
    p->max_anisotropy = 1.0f;
    p->mag_filter = GL_LINEAR;
    p->swizzled = GL_FALSE;
    p->swizzle_r = GL_RED;
    p->swizzle_g = GL_GREEN;
    p->swizzle_b = GL_BLUE;
    p->swizzle_a = GL_ALPHA;

    if (target == GL_TEXTURE_RECTANGLE)
    {
        p->min_filter = GL_LINEAR;
        p->wrap_s = p->wrap_t = p->wrap_r = GL_CLAMP_TO_EDGE;
    }
    else
    {
        p->min_filter = GL_NEAREST_MIPMAP_LINEAR;
        p->wrap_s = p->wrap_t = p->wrap_r = GL_REPEAT;
    }
}

void mglTexParamInvalidate(GLMContext ctx, TextureParameter *p)
{
    if (p->mtl_data)
    {
        if (ctx->mtl_funcs.ext_texparam && ctx->mtl_funcs.ext_texparam->releaseObj)
        {
            ctx->mtl_funcs.ext_texparam->releaseObj(ctx, p->mtl_data);
            p->mtl_data = NULL;
        }
    }
}

#pragma mark value conversion helpers

static GLint tp_value_i(const void *v, int vt, int i)
{
    switch (vt)
    {
        case TP_FLOAT: return (GLint)lroundf(((const GLfloat *)v)[i]);
        case TP_UINT:  return (GLint)((const GLuint *)v)[i];
        default:       return ((const GLint *)v)[i];
    }
}

static GLfloat tp_value_f(const void *v, int vt, int i)
{
    switch (vt)
    {
        case TP_FLOAT: return ((const GLfloat *)v)[i];
        case TP_UINT:  return (GLfloat)((const GLuint *)v)[i];
        default:       return (GLfloat)((const GLint *)v)[i];
    }
}

/* store a scalar in the requested output type */
static void tp_store(void *out, int vt, int i, double v)
{
    switch (vt)
    {
        case TP_FLOAT: ((GLfloat *)out)[i] = (GLfloat)v; break;
        case TP_UINT:  ((GLuint *)out)[i] = (GLuint)(v < 0 ? 0 : llround(v)); break;
        default:
            if (v >= (double)INT_MAX) ((GLint *)out)[i] = INT_MAX;
            else if (v <= (double)INT_MIN) ((GLint *)out)[i] = INT_MIN;
            else ((GLint *)out)[i] = (GLint)llround(v);
            break;
    }
}

static bool tp_valid_swizzle(GLint v)
{
    switch (v)
    {
        case GL_RED: case GL_GREEN: case GL_BLUE: case GL_ALPHA: case GL_ZERO: case GL_ONE:
            return true;
    }
    return false;
}

static void tp_update_swizzled(TextureParameter *p)
{
    p->swizzled = (p->swizzle_r != GL_RED) || (p->swizzle_g != GL_GREEN) ||
                  (p->swizzle_b != GL_BLUE) || (p->swizzle_a != GL_ALPHA);
}

static bool tp_is_multisample_target(GLenum target)
{
    return target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

bool mglTexParamIsSamplerState(GLenum pname)
{
    switch (pname)
    {
        case GL_TEXTURE_WRAP_S: case GL_TEXTURE_WRAP_T: case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_MIN_FILTER: case GL_TEXTURE_MAG_FILTER:
        case GL_TEXTURE_BORDER_COLOR:
        case GL_TEXTURE_MIN_LOD: case GL_TEXTURE_MAX_LOD: case GL_TEXTURE_LOD_BIAS:
        case GL_TEXTURE_COMPARE_MODE: case GL_TEXTURE_COMPARE_FUNC:
        case GL_TEXTURE_MAX_ANISOTROPY:
            return true;
    }
    return false;
}

#pragma mark set core

/*
 * Apply one parameter. target is the texture target (0 for sampler objects).
 * vector: called through a *v entry point (vector pnames allowed).
 * Returns GL_NO_ERROR or the GL error to raise; *changed reports a state change.
 */
GLenum mglTexParamSet(TextureParameter *p, GLenum target, bool is_sampler, GLenum pname,
                      const void *v, int vt, bool vector, bool *changed)
{
    TextureParameter old = *p;
    bool rect = (target == GL_TEXTURE_RECTANGLE);

    *changed = false;

    if (!is_sampler && tp_is_multisample_target(target) && mglTexParamIsSamplerState(pname))
        return GL_INVALID_OPERATION;   /* GL 4.6 8.10: sampler states on multisample targets */

    switch (pname)
    {
        case GL_DEPTH_STENCIL_TEXTURE_MODE:
        {
            if (is_sampler) return GL_INVALID_ENUM;
            GLint m = tp_value_i(v, vt, 0);
            if (m != GL_DEPTH_COMPONENT && m != GL_STENCIL_INDEX) return GL_INVALID_ENUM;
            p->depth_stencil_mode = m;
            break;
        }

        case GL_TEXTURE_BASE_LEVEL:
        {
            if (is_sampler) return GL_INVALID_ENUM;
            GLint l = tp_value_i(v, vt, 0);
            if (l < 0) return GL_INVALID_VALUE;
            if ((rect || tp_is_multisample_target(target)) && l != 0) return GL_INVALID_OPERATION;
            p->base_level = l;
            break;
        }

        case GL_TEXTURE_MAX_LEVEL:
        {
            if (is_sampler) return GL_INVALID_ENUM;
            GLint l = tp_value_i(v, vt, 0);
            if (l < 0) return GL_INVALID_VALUE;
            p->max_level = l;
            break;
        }

        case GL_TEXTURE_SWIZZLE_R:
        case GL_TEXTURE_SWIZZLE_G:
        case GL_TEXTURE_SWIZZLE_B:
        case GL_TEXTURE_SWIZZLE_A:
        {
            if (is_sampler) return GL_INVALID_ENUM;
            GLint s = tp_value_i(v, vt, 0);
            if (!tp_valid_swizzle(s)) return GL_INVALID_ENUM;
            switch (pname)
            {
                case GL_TEXTURE_SWIZZLE_R: p->swizzle_r = s; break;
                case GL_TEXTURE_SWIZZLE_G: p->swizzle_g = s; break;
                case GL_TEXTURE_SWIZZLE_B: p->swizzle_b = s; break;
                default:                   p->swizzle_a = s; break;
            }
            tp_update_swizzled(p);
            break;
        }

        case GL_TEXTURE_SWIZZLE_RGBA:
        {
            if (is_sampler || !vector) return GL_INVALID_ENUM;
            GLint s[4];
            for (int i = 0; i < 4; i++)
            {
                s[i] = tp_value_i(v, vt, i);
                if (!tp_valid_swizzle(s[i])) return GL_INVALID_ENUM;
            }
            p->swizzle_r = s[0]; p->swizzle_g = s[1]; p->swizzle_b = s[2]; p->swizzle_a = s[3];
            tp_update_swizzled(p);
            break;
        }

        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T:
        case GL_TEXTURE_WRAP_R:
        {
            GLint w = tp_value_i(v, vt, 0);
            switch (w)
            {
                case GL_CLAMP_TO_EDGE: case GL_CLAMP_TO_BORDER:
                    break;
                case GL_REPEAT: case GL_MIRRORED_REPEAT: case GL_MIRROR_CLAMP_TO_EDGE:
                    if (rect) return GL_INVALID_ENUM;
                    break;
                default:
                    return GL_INVALID_ENUM;
            }
            if (pname == GL_TEXTURE_WRAP_S) p->wrap_s = w;
            else if (pname == GL_TEXTURE_WRAP_T) p->wrap_t = w;
            else p->wrap_r = w;
            break;
        }

        case GL_TEXTURE_MIN_FILTER:
        {
            GLint f = tp_value_i(v, vt, 0);
            switch (f)
            {
                case GL_NEAREST: case GL_LINEAR:
                    break;
                case GL_NEAREST_MIPMAP_NEAREST: case GL_LINEAR_MIPMAP_NEAREST:
                case GL_NEAREST_MIPMAP_LINEAR: case GL_LINEAR_MIPMAP_LINEAR:
                    if (rect) return GL_INVALID_ENUM;
                    break;
                default:
                    return GL_INVALID_ENUM;
            }
            p->min_filter = f;
            break;
        }

        case GL_TEXTURE_MAG_FILTER:
        {
            GLint f = tp_value_i(v, vt, 0);
            if (f != GL_NEAREST && f != GL_LINEAR) return GL_INVALID_ENUM;
            p->mag_filter = f;
            break;
        }

        case GL_TEXTURE_MIN_LOD:
            p->min_lod = tp_value_f(v, vt, 0);
            break;

        case GL_TEXTURE_MAX_LOD:
            p->max_lod = tp_value_f(v, vt, 0);
            break;

        case GL_TEXTURE_LOD_BIAS:
            p->lod_bias = tp_value_f(v, vt, 0);
            break;

        case GL_TEXTURE_COMPARE_MODE:
        {
            GLint m = tp_value_i(v, vt, 0);
            if (m != GL_NONE && m != GL_COMPARE_REF_TO_TEXTURE) return GL_INVALID_ENUM;
            p->compare_mode = m;
            break;
        }

        case GL_TEXTURE_COMPARE_FUNC:
        {
            GLint f = tp_value_i(v, vt, 0);
            switch (f)
            {
                case GL_LEQUAL: case GL_GEQUAL: case GL_LESS: case GL_GREATER:
                case GL_EQUAL: case GL_NOTEQUAL: case GL_ALWAYS: case GL_NEVER:
                    break;
                default:
                    return GL_INVALID_ENUM;
            }
            p->compare_func = f;
            break;
        }

        case GL_TEXTURE_MAX_ANISOTROPY:
        {
            GLfloat a = tp_value_f(v, vt, 0);
            if (!(a >= 1.0f)) return GL_INVALID_VALUE;
            p->max_anisotropy = a;
            break;
        }

        case GL_TEXTURE_BORDER_COLOR:
        {
            if (!vector) return GL_INVALID_ENUM;
            for (int i = 0; i < 4; i++)
            {
                switch (vt)
                {
                    case TP_FLOAT:
                    {
                        GLfloat f = ((const GLfloat *)v)[i];
                        p->border_color[i] = f;
                        p->border_color_i[i] = (GLint)f;
                        p->border_color_ui[i] = f < 0 ? 0 : (GLuint)f;
                        break;
                    }
                    case TP_INT:
                    {
                        /* glTexParameteriv: signed normalized conversion (eq. 2.2) */
                        GLint c = ((const GLint *)v)[i];
                        GLfloat f = (GLfloat)((double)c / 2147483647.0);
                        if (f < -1.0f) f = -1.0f;
                        p->border_color[i] = f;
                        p->border_color_i[i] = c;
                        p->border_color_ui[i] = (GLuint)c;
                        break;
                    }
                    case TP_PURE_INT:
                    {
                        GLint c = ((const GLint *)v)[i];
                        p->border_color_i[i] = c;
                        p->border_color_ui[i] = (GLuint)c;
                        p->border_color[i] = (GLfloat)c;
                        break;
                    }
                    case TP_UINT:
                    {
                        GLuint c = ((const GLuint *)v)[i];
                        p->border_color_ui[i] = c;
                        p->border_color_i[i] = (GLint)c;
                        p->border_color[i] = (GLfloat)c;
                        break;
                    }
                }
            }
            break;
        }

        default:
            return GL_INVALID_ENUM;
    }

    /* mtl_data is a cache, compare everything else */
    old.mtl_data = p->mtl_data;
    *changed = memcmp(&old, p, sizeof(TextureParameter)) != 0;

    return GL_NO_ERROR;
}

#pragma mark get core

static GLuint tp_view_num_layers(Texture *tex)
{
    switch (tex->target)
    {
        case GL_TEXTURE_1D_ARRAY: return tex->height;
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return tex->depth;
        case GL_TEXTURE_CUBE_MAP_ARRAY: return tex->depth;  /* layer-faces */
        case GL_TEXTURE_CUBE_MAP: return 6;
        default: return 1;
    }
}

/*
 * Read one parameter. tex is NULL for sampler objects.
 * Returns GL_NO_ERROR or the GL error to raise.
 */
GLenum mglTexParamGet(GLMContext ctx, Texture *tex, TextureParameter *p, bool is_sampler, GLenum pname, void *out, int vt)
{
    if (!is_sampler && tex)
    {
        /* GL 4.6 8.11: sampler-state queries on multisample targets, and any
         * GL_TEXTURE_TARGET query on a buffer texture, are INVALID_OPERATION */
        if (tp_is_multisample_target(tex->target) && mglTexParamIsSamplerState(pname))
            return GL_INVALID_OPERATION;
        if (tex->target == GL_TEXTURE_BUFFER && pname == GL_TEXTURE_TARGET)
            return GL_INVALID_OPERATION;
    }

    switch (pname)
    {
        case GL_TEXTURE_WRAP_S:      tp_store(out, vt, 0, p->wrap_s); return GL_NO_ERROR;
        case GL_TEXTURE_WRAP_T:      tp_store(out, vt, 0, p->wrap_t); return GL_NO_ERROR;
        case GL_TEXTURE_WRAP_R:      tp_store(out, vt, 0, p->wrap_r); return GL_NO_ERROR;
        case GL_TEXTURE_MIN_FILTER:  tp_store(out, vt, 0, p->min_filter); return GL_NO_ERROR;
        case GL_TEXTURE_MAG_FILTER:  tp_store(out, vt, 0, p->mag_filter); return GL_NO_ERROR;
        case GL_TEXTURE_MIN_LOD:     tp_store(out, vt, 0, p->min_lod); return GL_NO_ERROR;
        case GL_TEXTURE_MAX_LOD:     tp_store(out, vt, 0, p->max_lod); return GL_NO_ERROR;
        case GL_TEXTURE_LOD_BIAS:    tp_store(out, vt, 0, p->lod_bias); return GL_NO_ERROR;
        case GL_TEXTURE_COMPARE_MODE: tp_store(out, vt, 0, p->compare_mode); return GL_NO_ERROR;
        case GL_TEXTURE_COMPARE_FUNC: tp_store(out, vt, 0, p->compare_func); return GL_NO_ERROR;
        case GL_TEXTURE_MAX_ANISOTROPY: tp_store(out, vt, 0, p->max_anisotropy); return GL_NO_ERROR;

        case GL_TEXTURE_BORDER_COLOR:
            for (int i = 0; i < 4; i++)
            {
                switch (vt)
                {
                    case TP_FLOAT: ((GLfloat *)out)[i] = p->border_color[i]; break;
                    case TP_PURE_INT: ((GLint *)out)[i] = p->border_color_i[i]; break;
                    case TP_UINT: ((GLuint *)out)[i] = p->border_color_ui[i]; break;
                    case TP_INT:
                    {
                        /* float -> signed normalized integer (eq. 2.4) */
                        double f = p->border_color[i];
                        if (f > 1.0) f = 1.0;
                        if (f < -1.0) f = -1.0;
                        ((GLint *)out)[i] = (GLint)llround(f * 2147483647.0);
                        break;
                    }
                }
            }
            return GL_NO_ERROR;
    }

    if (is_sampler)
        return GL_INVALID_ENUM;

    switch (pname)
    {
        case GL_DEPTH_STENCIL_TEXTURE_MODE: tp_store(out, vt, 0, p->depth_stencil_mode); break;
        case GL_TEXTURE_BASE_LEVEL: tp_store(out, vt, 0, p->base_level); break;
        case GL_TEXTURE_MAX_LEVEL:  tp_store(out, vt, 0, p->max_level); break;
        case GL_TEXTURE_SWIZZLE_R:  tp_store(out, vt, 0, p->swizzle_r); break;
        case GL_TEXTURE_SWIZZLE_G:  tp_store(out, vt, 0, p->swizzle_g); break;
        case GL_TEXTURE_SWIZZLE_B:  tp_store(out, vt, 0, p->swizzle_b); break;
        case GL_TEXTURE_SWIZZLE_A:  tp_store(out, vt, 0, p->swizzle_a); break;
        case GL_TEXTURE_SWIZZLE_RGBA:
            tp_store(out, vt, 0, p->swizzle_r);
            tp_store(out, vt, 1, p->swizzle_g);
            tp_store(out, vt, 2, p->swizzle_b);
            tp_store(out, vt, 3, p->swizzle_a);
            break;

        case GL_TEXTURE_IMMUTABLE_FORMAT:
            tp_store(out, vt, 0, tex->immutable_storage ? GL_TRUE : GL_FALSE);
            break;

        case GL_TEXTURE_IMMUTABLE_LEVELS:
        {
            MGLTexParamRecord *rec = mgl_texparam_record(ctx, tex->name, false);
            if (rec && rec->is_view)
                tp_store(out, vt, 0, rec->view_num_levels);
            else
                tp_store(out, vt, 0, tex->immutable_storage ? tex->num_levels : 0);
            break;
        }

        case GL_TEXTURE_VIEW_MIN_LEVEL:
        case GL_TEXTURE_VIEW_NUM_LEVELS:
        case GL_TEXTURE_VIEW_MIN_LAYER:
        case GL_TEXTURE_VIEW_NUM_LAYERS:
        {
            MGLTexParamRecord *rec = mgl_texparam_record(ctx, tex->name, false);
            GLuint val = 0;

            if (rec && rec->is_view)
            {
                switch (pname)
                {
                    case GL_TEXTURE_VIEW_MIN_LEVEL: val = rec->view_min_level; break;
                    case GL_TEXTURE_VIEW_NUM_LEVELS: val = rec->view_num_levels; break;
                    case GL_TEXTURE_VIEW_MIN_LAYER: val = rec->view_min_layer; break;
                    default: val = rec->view_num_layers; break;
                }
            }
            else
            {
                MGLTexParamMtlInfo info;
                bool have = (tex->mtl_data && ctx->mtl_funcs.ext_texparam &&
                             ctx->mtl_funcs.ext_texparam->textureInfo &&
                             ctx->mtl_funcs.ext_texparam->textureInfo(ctx, tex, &info));

                if (have && info.is_view)
                {
                    switch (pname)
                    {
                        case GL_TEXTURE_VIEW_MIN_LEVEL: val = info.parent_level; break;
                        case GL_TEXTURE_VIEW_NUM_LEVELS: val = info.mipmap_level_count; break;
                        case GL_TEXTURE_VIEW_MIN_LAYER: val = info.parent_slice; break;
                        default: val = info.array_length; break;
                    }
                }
                else if (tex->immutable_storage)
                {
                    /* TexStorage*: NUM_LEVELS = levels, NUM_LAYERS per target, mins are 0 */
                    switch (pname)
                    {
                        case GL_TEXTURE_VIEW_NUM_LEVELS: val = tex->num_levels; break;
                        case GL_TEXTURE_VIEW_NUM_LAYERS: val = tp_view_num_layers(tex); break;
                        default: val = 0; break;
                    }
                }
            }
            tp_store(out, vt, 0, val);
            break;
        }

        case GL_TEXTURE_TARGET:
            tp_store(out, vt, 0, tex->target);
            break;

        case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE:
        {
            const MGLFormatDesc *d = mglTexParamFormatDesc(tex->internalformat);
            tp_store(out, vt, 0, (d && d->img_class != GL_NONE) ? GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE : GL_NONE);
            break;
        }

        default:
            return GL_INVALID_ENUM;
    }

    return GL_NO_ERROR;
}

#pragma mark target / object lookup

static bool tp_valid_param_target(GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
    }
    return false;
}

/* bound texture for a glTexParameter target, or NULL with the error raised */
static Texture *tp_bound_tex(GLMContext ctx, GLenum target)
{
    if (!tp_valid_param_target(target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return NULL;
    }

    Texture *tex = getTex(ctx, 0, target);
    if (!tex)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }
    return tex;
}

/* named texture for glTextureParameter, or NULL with the error raised */
static Texture *tp_named_tex(GLMContext ctx, GLuint texture, bool for_set)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }

    if (for_set && tex->target == GL_TEXTURE_BUFFER)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }
    return tex;
}

static void tp_apply(GLMContext ctx, Texture *tex, GLenum pname, const void *v, int vt, bool vector)
{
    bool changed;
    GLenum err;

    if (!tex)
        return;

    if (!v)
        TP_ERROR(GL_INVALID_VALUE);

    err = mglTexParamSet(&tex->params, tex->target, false, pname, v, vt, vector, &changed);
    if (err != GL_NO_ERROR)
        TP_ERROR(err);

    if (changed)
    {
        mglTexParamInvalidate(ctx, &tex->params);
        ctx->state.dirty_bits |= DIRTY_TEX;
    }
}

static void tp_query(GLMContext ctx, Texture *tex, GLenum pname, void *out, int vt)
{
    GLenum err;

    if (!tex)
        return;

    if (!out)
        TP_ERROR(GL_INVALID_VALUE);

    err = mglTexParamGet(ctx, tex, &tex->params, false, pname, out, vt);
    if (err != GL_NO_ERROR)
        ERROR_RETURN(err);
}

#pragma mark tex param gl calls
void mglTexParameterf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, &param, TP_FLOAT, false);
}

void mglTexParameterfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, params, TP_FLOAT, true);
}

void mglTexParameteri(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, &param, TP_INT, false);
}

void mglTexParameteriv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, params, TP_INT, true);
}

void mglTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, params, TP_PURE_INT, true);
}

void mglTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, const GLuint *params)
{
    tp_apply(ctx, tp_bound_tex(ctx, target), pname, params, TP_UINT, true);
}

void mglTextureParameterf(GLMContext ctx, GLuint texture, GLenum pname, GLfloat param)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, &param, TP_FLOAT, false);
}

void mglTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, const GLfloat *param)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, param, TP_FLOAT, true);
}

void mglTextureParameteri(GLMContext ctx, GLuint texture, GLenum pname, GLint param)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, &param, TP_INT, false);
}

void mglTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, const GLint *param)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, param, TP_INT, true);
}

void mglTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, const GLint *params)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, params, TP_PURE_INT, true);
}

void mglTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, const GLuint *params)
{
    tp_apply(ctx, tp_named_tex(ctx, texture, true), pname, params, TP_UINT, true);
}

#pragma mark get tex param gl calls
void mglGetTexParameterfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{
    tp_query(ctx, tp_bound_tex(ctx, target), pname, params, TP_FLOAT);
}

void mglGetTexParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    tp_query(ctx, tp_bound_tex(ctx, target), pname, params, TP_INT);
}

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    tp_query(ctx, tp_bound_tex(ctx, target), pname, params, TP_PURE_INT);
}

void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params)
{
    tp_query(ctx, tp_bound_tex(ctx, target), pname, params, TP_UINT);
}

void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params)
{
    tp_query(ctx, tp_named_tex(ctx, texture, false), pname, params, TP_FLOAT);
}

void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    tp_query(ctx, tp_named_tex(ctx, texture, false), pname, params, TP_INT);
}

void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    tp_query(ctx, tp_named_tex(ctx, texture, false), pname, params, TP_PURE_INT);
}

void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params)
{
    tp_query(ctx, tp_named_tex(ctx, texture, false), pname, params, TP_UINT);
}

#pragma mark level parameters

/* map a GetTexLevelParameter target to (texture, cube face); NULL with error raised */
static Texture *tp_level_target(GLMContext ctx, GLenum target, GLuint *face)
{
    *face = 0;

    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        case GL_TEXTURE_BUFFER:
            return getTex(ctx, 0, target);

        case GL_TEXTURE_CUBE_MAP_POSITIVE_X: case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y: case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z: case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            *face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            return getTex(ctx, 0, GL_TEXTURE_CUBE_MAP);

        case GL_PROXY_TEXTURE_1D: case GL_PROXY_TEXTURE_2D: case GL_PROXY_TEXTURE_3D:
        case GL_PROXY_TEXTURE_1D_ARRAY: case GL_PROXY_TEXTURE_2D_ARRAY:
        case GL_PROXY_TEXTURE_RECTANGLE: case GL_PROXY_TEXTURE_CUBE_MAP:
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE: case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return mgl_texparam_proxy_texture(ctx, target);
    }

    ERROR_RETURN(GL_INVALID_ENUM);
    return NULL;
}

void mglGetTexLevelParameterfv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLfloat *params)
{
    GLuint face;
    Texture *tex = tp_level_target(ctx, target, &face);

    if (!tex)
        return;

    mglTexParamGetLevel(ctx, tex, face ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + face : target, level, pname, NULL, params, false);
}

void mglGetTexLevelParameteriv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLint *params)
{
    GLuint face;
    Texture *tex = tp_level_target(ctx, target, &face);

    if (!tex)
        return;

    mglTexParamGetLevel(ctx, tex, face ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + face : target, level, pname, params, NULL, false);
}
