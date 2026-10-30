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
 * textures.c
 * MGL
 *
 * GL 4.6 texture objects, image specification, pixel transfer, compressed
 * images, copies, readback and clears ('texdata' domain).  GPU work is done by
 * MGLRenderer+TextureData.m through ctx->mtl_funcs.ext_texdata; the storage
 * model is described in mgl_ext_texdata.h.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pixel_utils.h"
#include "utils.h"
#include "glm_context.h"
#include "mgl_ext_texdata.h"
#include "mgl_ext_texext.h"

/* forward: defined with the pixel read/write helpers below */
static Texture *view_redirect(GLMContext ctx, Texture *tex, GLuint *face, GLint *level, GLint *z);
#include "mgl_ext_texparam.h"
#include "mgl_ext_fb.h"

#ifndef GL_COMPRESSED_RGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#endif

/* error helpers that really return (ERROR_RETURN in glm_context.h does not) */
#define TD_ERR(e)        do { ctx->error_func(ctx, __FUNCTION__, (e)); return; } while (0)
#define TD_ERRV(e, v)    do { ctx->error_func(ctx, __FUNCTION__, (e)); return (v); } while (0)
#define TD_CHECK(c, e)   do { if (!(c)) TD_ERR(e); } while (0)

#define TDS(ctx)  ((ctx)->state.ext_texdata)
#define TDM(ctx)  ((ctx)->mtl_funcs.ext_texdata)

enum { TD_PROXY_1D, TD_PROXY_2D, TD_PROXY_3D, TD_PROXY_RECT, TD_PROXY_1D_ARRAY, TD_PROXY_2D_ARRAY,
       TD_PROXY_CUBE, TD_PROXY_CUBE_ARRAY, TD_NUM_PROXIES };

/* ------------------------------------------------------------------------ */
/* limits                                                                   */
/* ------------------------------------------------------------------------ */

static GLuint limitOr(GLuint v, GLuint dflt) { return (v == 0 || v > (1u << 15)) ? dflt : v; }
static GLuint maxTexSize(GLMContext ctx)   { return limitOr(STATE_VAR(max_texture_size), 16384); }
static GLuint max3DSize(GLMContext ctx)    { return limitOr(STATE_VAR(max_3d_texture_size), 2048); }
static GLuint maxCubeSize(GLMContext ctx)  { return limitOr(STATE_VAR(max_cube_map_texture_size), 16384); }
static GLuint maxLayers(GLMContext ctx)    { return limitOr(STATE_VAR(max_array_texture_layers), 2048); }
static GLuint maxRectSize(GLMContext ctx)  { return limitOr(STATE_VAR(max_rectangle_texture_size), 16384); }
static GLuint maxUnits(GLMContext ctx)
{
    GLuint n = STATE_VAR(max_combined_texture_image_units);
    return (n == 0 || n > TEXTURE_UNITS) ? TEXTURE_UNITS : n;
}

static GLuint log2u(GLuint v) { GLuint r = 0; while (v > 1) { v >>= 1; r++; } return r; }

/* ------------------------------------------------------------------------ */
/* targets                                                                  */
/* ------------------------------------------------------------------------ */

GLuint textureIndexFromTarget(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_BUFFER: return _TEXTURE_BUFFER_TARGET;
        case GL_TEXTURE_1D: return _TEXTURE_1D;
        case GL_TEXTURE_2D: return _TEXTURE_2D;
        case GL_TEXTURE_3D: return _TEXTURE_3D;
        case GL_TEXTURE_RECTANGLE: return _TEXTURE_RECTANGLE;
        case GL_TEXTURE_1D_ARRAY: return _TEXTURE_1D_ARRAY;
        case GL_TEXTURE_2D_ARRAY: return _TEXTURE_2D_ARRAY;
        case GL_TEXTURE_CUBE_MAP: return _TEXTURE_CUBE_MAP;
        case GL_TEXTURE_CUBE_MAP_ARRAY: return _TEXTURE_CUBE_MAP_ARRAY;
        case GL_TEXTURE_2D_MULTISAMPLE: return _TEXTURE_2D_MULTISAMPLE;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case GL_RENDERBUFFER: return _RENDERBUFFER;
        default: break;
    }

    (void)ctx;
    return _MAX_TEXTURE_TYPES;
}

static GLenum targetFromIndex(GLuint index)
{
    static const struct { GLuint i; GLenum t; } map[] = {
        { _TEXTURE_BUFFER_TARGET, GL_TEXTURE_BUFFER }, { _TEXTURE_1D, GL_TEXTURE_1D }, { _TEXTURE_2D, GL_TEXTURE_2D },
        { _TEXTURE_3D, GL_TEXTURE_3D }, { _TEXTURE_RECTANGLE, GL_TEXTURE_RECTANGLE },
        { _TEXTURE_1D_ARRAY, GL_TEXTURE_1D_ARRAY }, { _TEXTURE_2D_ARRAY, GL_TEXTURE_2D_ARRAY },
        { _TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP }, { _TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_CUBE_MAP_ARRAY },
        { _TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_2D_MULTISAMPLE },
        { _TEXTURE_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_2D_MULTISAMPLE_ARRAY }, { _RENDERBUFFER, GL_RENDERBUFFER },
    };
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (map[i].i == index) return map[i].t;
    return 0;
}

static bool isCubeFace(GLenum target)
{
    return target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
}

static bool isArrayTarget(GLenum target)
{
    return target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY ||
           target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

GLuint mgl_texdata_num_faces(Texture *tex)
{
    return tex->target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
}

static int proxyIndex(GLenum target)
{
    switch (target)
    {
        case GL_PROXY_TEXTURE_1D: return TD_PROXY_1D;
        case GL_PROXY_TEXTURE_2D: return TD_PROXY_2D;
        case GL_PROXY_TEXTURE_3D: return TD_PROXY_3D;
        case GL_PROXY_TEXTURE_RECTANGLE: return TD_PROXY_RECT;
        case GL_PROXY_TEXTURE_1D_ARRAY: return TD_PROXY_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY: return TD_PROXY_2D_ARRAY;
        case GL_PROXY_TEXTURE_CUBE_MAP: return TD_PROXY_CUBE;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY: return TD_PROXY_CUBE_ARRAY;
        default: return -1;
    }
}

static GLenum proxyTexTarget(GLenum proxy)
{
    switch (proxy)
    {
        case GL_PROXY_TEXTURE_1D: return GL_TEXTURE_1D;
        case GL_PROXY_TEXTURE_2D: return GL_TEXTURE_2D;
        case GL_PROXY_TEXTURE_3D: return GL_TEXTURE_3D;
        case GL_PROXY_TEXTURE_RECTANGLE: return GL_TEXTURE_RECTANGLE;
        case GL_PROXY_TEXTURE_1D_ARRAY: return GL_TEXTURE_1D_ARRAY;
        case GL_PROXY_TEXTURE_2D_ARRAY: return GL_TEXTURE_2D_ARRAY;
        case GL_PROXY_TEXTURE_CUBE_MAP: return GL_TEXTURE_CUBE_MAP;
        case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY: return GL_TEXTURE_CUBE_MAP_ARRAY;
        default: return 0;
    }
}

/* ------------------------------------------------------------------------ */
/* objects                                                                  */
/* ------------------------------------------------------------------------ */

Texture *currentTexture(GLMContext ctx, GLuint index)
{
    if (index >= _MAX_TEXTURE_TYPES)
        return NULL;

    return STATE(texture_units[STATE(active_texture)].textures[index]);
}

Texture *newTexObj(GLMContext ctx, GLenum target)
{
    Texture *ptr;
    GLuint index;

    index = textureIndexFromTarget(ctx, target);
    if (index == _MAX_TEXTURE_TYPES && proxyIndex(target) < 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, NULL);
    }

    ptr = (Texture *)calloc(1, sizeof(Texture));
    if (!ptr)
    {
        ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, NULL);
    }

    ptr->name = TEX_OBJ_RES_NAME;
    ptr->target = target;
    ptr->index = index;
    ptr->access = GL_READ_ONLY;

    // GL 4.6 default texture/sampler state (owned by texparam, tex_param.c)
    mglTexParamInitDefaults(&ptr->params, target);

    return ptr;
}

Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;

    ptr = newTexObj(ctx, target);
    if (ptr)
        ptr->name = texture;

    return ptr;
}

/* create the object for a generated name on first bind */
static Texture *getTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    Texture *ptr;

    ptr = (Texture *)searchHashTable(&STATE(texture_table), texture);

    if (!ptr)
    {
        ptr = newTexture(ctx, target, texture);
        if (!ptr)
            return NULL;

        insertHashElement(&STATE(texture_table), texture, ptr);
    }

    return ptr;
}

Texture *findTexture(GLMContext ctx, GLuint texture)
{
    if (texture == 0)
        return NULL;

    return (Texture *)searchHashTable(&STATE(texture_table), texture);
}

static Texture *defaultTexture(GLMContext ctx, GLuint index)
{
    struct MGLExtState_texdata *td = TDS(ctx);
    GLenum target;

    if (index >= _MAX_TEXTURE_TYPES || !td)
        return NULL;

    if (td->defaults[index] == NULL)
    {
        target = targetFromIndex(index);
        if (!target)
            return NULL;
        td->defaults[index] = newTexObj(ctx, target);
    }

    return td->defaults[index];
}

/* bound texture for a target on the active unit (name 0 -> default object),
 * or the named texture (DSA, target ignored) */
Texture *getTex(GLMContext ctx, GLuint name, GLenum target)
{
    GLuint index;
    Texture *ptr;

    if (name == 0)
    {
        if (isCubeFace(target))
            target = GL_TEXTURE_CUBE_MAP;

        if (proxyIndex(target) >= 0)
            return mgl_texparam_proxy_texture(ctx, target);

        index = textureIndexFromTarget(ctx, target);
        if (index == _MAX_TEXTURE_TYPES)
            return NULL;

        ptr = currentTexture(ctx, index);
        if (!ptr)
            ptr = defaultTexture(ctx, index);

        return ptr;
    }

    return findTexture(ctx, name);
}

bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat)
{
    (void)ctx;
    return mtlFormatForGLInternalFormat(internalformat) != MTLPixelFormatInvalid;
}

/* ------------------------------------------------------------------------ */
/* side info                                                                */
/* ------------------------------------------------------------------------ */

TDTexInfo *mgl_texdata_find_info(GLMContext ctx, Texture *tex)
{
    struct MGLExtState_texdata *td = TDS(ctx);

    if (!td || !tex || tex->dirty_on_gpu == 0 || tex->dirty_on_gpu > td->num_infos)
        return NULL;

    TDTexInfo *info = td->infos[tex->dirty_on_gpu - 1];
    return (info && info->used) ? info : NULL;
}

TDTexInfo *mgl_texdata_info(GLMContext ctx, Texture *tex)
{
    struct MGLExtState_texdata *td = TDS(ctx);
    TDTexInfo *info;

    info = mgl_texdata_find_info(ctx, tex);
    if (info)
        return info;

    if (!td || !tex)
        return NULL;

    GLuint slot = td->num_infos;
    for (GLuint i = td->free_hint; i < td->num_infos; i++)
    {
        if (td->infos[i] == NULL || !td->infos[i]->used) { slot = i; break; }
    }

    if (slot == td->num_infos)
    {
        if (td->num_infos == td->cap_infos)
        {
            GLuint ncap = td->cap_infos ? td->cap_infos * 2 : 64;
            TDTexInfo **n = (TDTexInfo **)realloc(td->infos, ncap * sizeof(TDTexInfo *));
            if (!n)
                return NULL;
            memset(n + td->cap_infos, 0, (ncap - td->cap_infos) * sizeof(TDTexInfo *));
            td->infos = n;
            td->cap_infos = ncap;
        }
        td->num_infos++;
    }

    if (!td->infos[slot])
    {
        td->infos[slot] = (TDTexInfo *)calloc(1, sizeof(TDTexInfo));
        if (!td->infos[slot])
            return NULL;
    }

    info = td->infos[slot];
    memset(info, 0, sizeof(TDTexInfo));
    info->used = GL_TRUE;
    tex->dirty_on_gpu = slot + 1;
    td->free_hint = slot + 1;

    return info;
}

static void freeInfo(GLMContext ctx, Texture *tex)
{
    struct MGLExtState_texdata *td = TDS(ctx);
    TDTexInfo *info = mgl_texdata_find_info(ctx, tex);

    if (info)
    {
        for (int f = 0; f < 6; f++)
            for (int l = 0; l < TD_MAX_LEVELS; l++)
                free(info->orig[f][l]);
        memset(info, 0, sizeof(TDTexInfo));
        if (tex->dirty_on_gpu - 1 < td->free_hint)
            td->free_hint = tex->dirty_on_gpu - 1;
    }
    tex->dirty_on_gpu = 0;
}

/* ------------------------------------------------------------------------ */
/* binding                                                                  */
/* ------------------------------------------------------------------------ */

static GLuint texBindingName(Texture *tex)
{
    return (tex && tex->name != TEX_OBJ_RES_NAME) ? tex->name : 0;
}

/* keep the legacy GL_TEXTURE_BINDING_* state variables in sync with the active unit */
static void updateBindingVars(GLMContext ctx)
{
    TextureUnit *u = &STATE(texture_units[STATE(active_texture)]);

    STATE_VAR(texture_binding_1d) = texBindingName(u->textures[_TEXTURE_1D]);
    STATE_VAR(texture_binding_2d) = texBindingName(u->textures[_TEXTURE_2D]);
    STATE_VAR(texture_binding_3d) = texBindingName(u->textures[_TEXTURE_3D]);
    STATE_VAR(texture_binding_cube_map) = texBindingName(u->textures[_TEXTURE_CUBE_MAP]);
    STATE_VAR(texture_binding_1d_array) = texBindingName(u->textures[_TEXTURE_1D_ARRAY]);
    STATE_VAR(texture_binding_2d_array) = texBindingName(u->textures[_TEXTURE_2D_ARRAY]);
    STATE_VAR(texture_binding_rectangle) = texBindingName(u->textures[_TEXTURE_RECTANGLE]);
    STATE_VAR(texture_binding_buffer) = texBindingName(u->textures[_TEXTURE_BUFFER_TARGET]);
    STATE_VAR(texture_binding_2d_multisample) = texBindingName(u->textures[_TEXTURE_2D_MULTISAMPLE]);
    STATE_VAR(texture_binding_2d_multisample_array) = texBindingName(u->textures[_TEXTURE_2D_MULTISAMPLE_ARRAY]);
}

/* the sampler path uses one texture per unit (active_textures[]): the most
 * recently bound non-null target binding of that unit */
static void bindTextureToUnit(GLMContext ctx, GLuint unit, GLuint index, Texture *ptr)
{
    TextureUnit *u = &STATE(texture_units[unit]);
    Texture *old = u->textures[index];

    u->textures[index] = ptr;

    if (ptr)
    {
        STATE(active_textures[unit]) = ptr;
    }
    else if (STATE(active_textures[unit]) == old)
    {
        Texture *other = NULL;
        for (int i = 0; i < _MAX_TEXTURE_TYPES; i++)
            if (u->textures[i] && i != _RENDERBUFFER) { other = u->textures[i]; break; }
        STATE(active_textures[unit]) = other;
    }

    GLuint mask_index = unit / 32;
    GLuint mask = 0x1u << (unit % 32);

    if (STATE(active_textures[unit]))
        STATE(active_texture_mask[mask_index]) |= mask;
    else
        STATE(active_texture_mask[mask_index]) &= ~mask;

    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
}

static bool isGeneratedTextureName(GLMContext ctx, GLuint name)
{
    return name != TEX_OBJ_RES_NAME && name < STATE(texture_table.current_name);
}

#pragma mark basic tex calls bind / delete / gen...
void mglGenTextures(GLMContext ctx, GLsizei n, GLuint *textures)
{
    TD_CHECK(n >= 0, GL_INVALID_VALUE);

    if (!textures)
        return;

    while(n--)
    {
        GLuint name = getNewName(&STATE(texture_table));

        // TEX_OBJ_RES_NAME has special name.. skip it
        if (name == TEX_OBJ_RES_NAME)
            name = getNewName(&STATE(texture_table));

        *textures++ = name;
    }
}

static bool validObjectTarget(GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_BUFFER: case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
        default:
            return false;
    }
}

void mglCreateTextures(GLMContext ctx, GLenum target, GLsizei n, GLuint *textures)
{
    TD_CHECK(validObjectTarget(target), GL_INVALID_ENUM);
    TD_CHECK(n >= 0, GL_INVALID_VALUE);

    if (!textures)
        return;

    mglGenTextures(ctx, n, textures);

    for (GLsizei i = 0; i < n; i++)
    {
        if (!getTexture(ctx, target, textures[i]))
            TD_ERR(GL_OUT_OF_MEMORY);
    }
}

void mglBindTexture(GLMContext ctx, GLenum target, GLuint texture)
{
    GLuint index;
    Texture *ptr;

    TD_CHECK(validObjectTarget(target), GL_INVALID_ENUM);

    index = textureIndexFromTarget(ctx, target);

    if (texture)
    {
        ptr = findTexture(ctx, texture);

        if (ptr)
        {
            // a texture object can only ever be bound to the target it was created with
            TD_CHECK(ptr->target == target, GL_INVALID_OPERATION);
        }
        else
        {
            TD_CHECK(isGeneratedTextureName(ctx, texture), GL_INVALID_OPERATION);

            ptr = getTexture(ctx, target, texture);
            TD_CHECK(ptr, GL_OUT_OF_MEMORY);
        }
    }
    else
    {
        ptr = NULL;
    }

    bindTextureToUnit(ctx, STATE(active_texture), index, ptr);
    updateBindingVars(ctx);
}

void mglBindImageTexture(GLMContext ctx, GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum internalformat)
{
    Texture *ptr = NULL;
    GLint max_image_units = 0;

    /* GL 4.6 §8.26: <unit> is checked against GL_MAX_IMAGE_UNITS (not the larger
     * sampler-unit count) — KHR-GL46.shader_image_load_store negative-bind binds
     * unit 100 and expects GL_INVALID_VALUE */
    ctx->dispatch.get_integerv(ctx, GL_MAX_IMAGE_UNITS, &max_image_units);
    TD_CHECK(unit < (GLuint)(max_image_units > 0 ? max_image_units : 1), GL_INVALID_VALUE);
    TD_CHECK(level >= 0, GL_INVALID_VALUE);
    TD_CHECK(layer >= 0, GL_INVALID_VALUE);

    /* <internalformat> must be one of the image-unit formats (GL 4.6 §8.25 Table 8.26);
     * any other value — a random enum or a legal-but-unlisted texture internal format —
     * is GL_INVALID_VALUE, and the check applies even when <texture> is 0
     * (KHR-GL46.shader_image_load_store negative-bind binds unit 1, texture 0 with a
     * garbage format) */
    {
        static const GLint kImageFormats[] = {
            GL_RGBA32F, GL_RGBA16F, GL_RG32F, GL_RG16F, GL_R11F_G11F_B10F, GL_R32F,
            GL_R16F, GL_RGBA32UI, GL_RGBA16UI, GL_RGB10_A2UI, GL_RGBA8UI, GL_RG32UI,
            GL_RG16UI, GL_RG8UI, GL_R32UI, GL_R16UI, GL_R8UI, GL_RGBA32I, GL_RGBA16I,
            GL_RGBA8I, GL_RG32I, GL_RG16I, GL_RG8I, GL_R32I, GL_R16I, GL_R8I, GL_RGBA16,
            GL_RGB10_A2, GL_RGBA8, GL_RG16, GL_RG8, GL_R16, GL_R8,
            GL_RGBA16_SNORM, GL_RGBA8_SNORM, GL_RG16_SNORM, GL_RG8_SNORM, GL_R16_SNORM,
            GL_R8_SNORM
        };
        size_t f;

        for (f = 0; f < sizeof kImageFormats / sizeof kImageFormats[0]; f++)
            if ((GLint)internalformat == kImageFormats[f])
                break;
        TD_CHECK(f < sizeof kImageFormats / sizeof kImageFormats[0], GL_INVALID_VALUE);
    }

    switch(access)
    {
        case GL_READ_ONLY:
        case GL_WRITE_ONLY:
        case GL_READ_WRITE:
            break;

        default:
            TD_ERR(GL_INVALID_ENUM);
    }

    if (texture == 0)
    {
        bzero(&ctx->state.image_units[unit], sizeof(ImageUnit));
        ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
        return;
    }

    ptr = findTexture(ctx, texture);
    TD_CHECK(ptr, GL_INVALID_VALUE);

    TD_CHECK(checkInternalFormatForMetal(ctx, internalformat), GL_INVALID_VALUE);

    if (ptr->access != access)
    {
        ptr->dirty_bits |= DIRTY_TEXTURE_ACCESS;
        ptr->access = access;
    }

    ImageUnit unit_params;

    unit_params.unit = unit;
    unit_params.texture = texture;
    unit_params.level = level;
    unit_params.layered = layered;
    unit_params.layer = layer;
    unit_params.access = access;
    unit_params.internalformat = internalformat;
    unit_params.tex = ptr;

    ctx->state.image_units[unit] = unit_params;

    ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
}

static void detachFromFramebuffer(GLMContext ctx, Framebuffer *fbo, Texture *tex)
{
    if (!fbo)
        return;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        FBOAttachment *a = &fbo->color_attachments[i];
        if (a->textarget != GL_RENDERBUFFER && a->buf.tex == tex && a->texture)
        {
            memset(a, 0, sizeof(FBOAttachment));
            fbo->color_attachment_bitfield &= ~(0x1u << i);
            fbo->dirty_bits |= DIRTY_FBO_TEX;
        }
    }

    if (fbo->depth.textarget != GL_RENDERBUFFER && fbo->depth.buf.tex == tex && fbo->depth.texture)
    {
        memset(&fbo->depth, 0, sizeof(FBOAttachment));
        fbo->dirty_bits |= DIRTY_FBO_TEX;
    }

    if (fbo->stencil.textarget != GL_RENDERBUFFER && fbo->stencil.buf.tex == tex && fbo->stencil.texture)
    {
        memset(&fbo->stencil, 0, sizeof(FBOAttachment));
        fbo->dirty_bits |= DIRTY_FBO_TEX;
    }
}

static bool attachedToAnyFramebuffer(GLMContext ctx, Texture *tex)
{
    HashTable *t = &STATE(framebuffer_table);

    for (GLuint n = 0; n < t->size; n++)
    {
        Framebuffer *fbo = (Framebuffer *)t->keys[n].data;
        if (!fbo)
            continue;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
            if (fbo->color_attachments[i].texture && fbo->color_attachments[i].textarget != GL_RENDERBUFFER &&
                fbo->color_attachments[i].buf.tex == tex)
                return true;
        if (fbo->depth.texture && fbo->depth.textarget != GL_RENDERBUFFER && fbo->depth.buf.tex == tex)
            return true;
        if (fbo->stencil.texture && fbo->stencil.textarget != GL_RENDERBUFFER && fbo->stencil.buf.tex == tex)
            return true;
    }

    return false;
}

static void freeTextureStorage(GLMContext ctx, Texture *tex)
{
    if (TDM(ctx) && TDM(ctx)->release)
        TDM(ctx)->release(ctx, tex);
    else if (tex->mtl_data)
        ctx->mtl_funcs.mtlDeleteMTLObj(ctx, tex->mtl_data);
    tex->mtl_data = NULL;

    for (int face = 0; face < 6; face++)
    {
        if (!tex->faces[face].levels)
            continue;
        for (int l = 0; l < TD_MAX_LEVELS; l++)
            mgl_texdata_free_shadow(ctx, tex, face, l);
        free(tex->faces[face].levels);
        tex->faces[face].levels = NULL;
    }

    freeInfo(ctx, tex);
}

void mglDeleteTextures(GLMContext ctx, GLsizei n, const GLuint *textures)
{
    TD_CHECK(n >= 0, GL_INVALID_VALUE);

    if (!textures)
        return;

    while(n--)
    {
        GLuint name = *textures++;
        Texture *tex = findTexture(ctx, name);

        if (!tex)
            continue;

        // unbind from every texture unit (all targets)
        for (GLuint u = 0; u < TEXTURE_UNITS; u++)
        {
            for (GLuint i = 0; i < _MAX_TEXTURE_TYPES; i++)
            {
                if (STATE(texture_units[u].textures[i]) == tex)
                    bindTextureToUnit(ctx, u, i, NULL);
            }
            if (STATE(active_textures[u]) == tex)
            {
                STATE(active_textures[u]) = NULL;
                STATE(active_texture_mask[u / 32]) &= ~(0x1u << (u % 32));
            }
        }

        for (int i = 0; i < TEXTURE_UNITS; i++)
        {
            if (ctx->state.image_units[i].texture == name)
            {
                bzero(&ctx->state.image_units[i], sizeof(ImageUnit));
                ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
            }
        }

        // detach from the bound framebuffers
        detachFromFramebuffer(ctx, STATE(framebuffer), tex);
        detachFromFramebuffer(ctx, STATE(readbuffer), tex);

        /* GL 4.6 9.2: a view shares the parent's images, so deleting the parent (root) is
         * DEFERRED until no views remain — the view keeps sampling the shared storage
         * (KHR-GL46.texture_view.reference_counting).  Unbind a dying view's reference
         * first; the deferred root then dies with its last view. */
        {
            MGLTexExtInfo *info = mgl_texext_info(ctx, tex);
            if (info && info->kind == MGL_TEXEXT_VIEW && info->view_root)
            {
                Texture *root = info->view_root;
                if (mgl_texext_view_unref_root(ctx, root))
                {
                    /* the root's hash entry/name were dropped at its own delete; free it now
                     * (same storage+side-table+struct teardown as below), unless an FBO
                     * still holds it (that path owns the deferred free then) */
                    if (!attachedToAnyFramebuffer(ctx, root))
                    {
                        freeTextureStorage(ctx, root);
                        mgl_texext_info_free(ctx, root);
                        free(root);
                    }
                }
            }
            /* not a view: a root with live views stays alive for them */
            if (info && info->view_refs > 0)
            {
                info->pending_delete = GL_TRUE;
                tex->name = TEX_OBJ_RES_NAME;
                /* deleteHashElement CFReleases the slot object's Metal storage; the
                 * deferred root's storage must survive for its views, so take it out of
                 * the slot for the delete and restore it — freeTextureStorage releases
                 * it when the last view unrefs */
                {
                    void *keep_mtl = tex->mtl_data, *keep_pmtl = tex->params.mtl_data;
                    tex->mtl_data = NULL;
                    tex->params.mtl_data = NULL;
                    deleteHashElement(&STATE(texture_table), name);
                    tex->mtl_data = keep_mtl;
                    tex->params.mtl_data = keep_pmtl;
                }
                continue;
            }
        }

        deleteHashElement(&STATE(texture_table), name);

        // an object still attached to an unbound framebuffer stays alive (GL 4.6 5.1.3)
        if (!attachedToAnyFramebuffer(ctx, tex))
        {
            freeTextureStorage(ctx, tex);
            /* drop the texext side-table entry first: it is keyed by the Texture pointer and
             * would otherwise alias a Texture allocated later at this address (a stale "this is
             * a texture view / buffer / multisample" marker made texdata refuse the new object) */
            mgl_texext_info_free(ctx, tex);
            free(tex);
        }
        else
        {
            tex->name = TEX_OBJ_RES_NAME;
        }
    }

    updateBindingVars(ctx);
    STATE(dirty_bits) |= DIRTY_TEX_BINDING;
}

GLboolean mglIsTexture(GLMContext ctx, GLuint texture)
{
    return findTexture(ctx, texture) != NULL ? GL_TRUE : GL_FALSE;
}

void mglBindImageTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    TD_CHECK(count >= 0, GL_INVALID_VALUE);
    /* the image-unit count is GL_MAX_IMAGE_UNITS — get.c reports
     * MGL_MAX_IMAGE_UNITS (16), so bind against exactly that (an 8 here went
     * stale when MGL_MAX_IMAGE_UNITS rose and KHR-GL46.multi_bind
     * dispatch/functional_bind_image_textures began seeing a bogus
     * GL_INVALID_OPERATION binding all 16 units); state storage is
     * image_units[TEXTURE_UNITS=128], so 16 is safely addressable */
    TD_CHECK((GLuint)first + (GLuint)count <= MGL_IMAGE_UNITS_MAX, GL_INVALID_OPERATION);

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint tex_name = textures ? textures[i] : 0;

        if (tex_name == 0)
        {
            bzero(&ctx->state.image_units[first + i], sizeof(ImageUnit));
            ctx->state.dirty_bits |= DIRTY_IMAGE_UNIT_STATE;
            continue;
        }

        Texture *tex = findTexture(ctx, tex_name);
        if (!tex)
        {
            /* multi-bind: the invalid entry is skipped with an error, the other
             * entries still take effect */
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            continue;
        }

        /* an entry whose texture has no valid level 0 (invalid internal format /
         * zero dimensions) is skipped with INVALID_OPERATION as well
         * (KHR-GL46.multi_bind errors_bind_image_textures) */
        if (!tex->internalformat || !tex->width || !tex->height)
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            continue;
        }

        // GL 4.6 8.26: level 0, layered, READ_WRITE, the texture's internal format
        mglBindImageTexture(ctx, first + i, tex_name, 0, GL_TRUE, 0, GL_READ_WRITE, tex->internalformat);
    }
}

void mglActiveTexture(GLMContext ctx, GLenum texture)
{
    TD_CHECK(texture >= GL_TEXTURE0 && texture - GL_TEXTURE0 < maxUnits(ctx), GL_INVALID_ENUM);

    STATE(active_texture) = texture - GL_TEXTURE0;
    updateBindingVars(ctx);
    ctx->state.dirty_bits |= DIRTY_TEX_BINDING;
}

void mglBindTextures(GLMContext ctx, GLuint first, GLsizei count, const GLuint *textures)
{
    TD_CHECK(count >= 0, GL_INVALID_VALUE);
    TD_CHECK((GLuint)first + (GLuint)count <= maxUnits(ctx), GL_INVALID_OPERATION);

    for (GLsizei i = 0; i < count; i++)
    {
        GLuint unit = first + i;
        GLuint texture = textures ? textures[i] : 0;

        if (texture == 0)
        {
            for (GLuint idx = 0; idx < _MAX_TEXTURE_TYPES; idx++)
                if (STATE(texture_units[unit].textures[idx]))
                    bindTextureToUnit(ctx, unit, idx, NULL);
            continue;
        }

        Texture *ptr = findTexture(ctx, texture);
        if (!ptr || ptr->index >= _MAX_TEXTURE_TYPES)
        {
            // error for this entry only, the others are still bound
            ERROR_RETURN(GL_INVALID_OPERATION);
            continue;
        }

        bindTextureToUnit(ctx, unit, ptr->index, ptr);
    }

    updateBindingVars(ctx);
}

void mglBindTextureUnit(GLMContext ctx, GLuint unit, GLuint texture)
{
    TD_CHECK(unit < maxUnits(ctx), GL_INVALID_VALUE);

    if (texture == 0)
    {
        for (GLuint idx = 0; idx < _MAX_TEXTURE_TYPES; idx++)
            if (STATE(texture_units[unit].textures[idx]))
                bindTextureToUnit(ctx, unit, idx, NULL);
        updateBindingVars(ctx);
        return;
    }

    Texture *ptr = findTexture(ctx, texture);
    TD_CHECK(ptr && ptr->index < _MAX_TEXTURE_TYPES, GL_INVALID_OPERATION);

    bindTextureToUnit(ctx, unit, ptr->index, ptr);
    updateBindingVars(ctx);
}

/* ------------------------------------------------------------------------ */
/* levels                                                                   */
/* ------------------------------------------------------------------------ */

static bool ensureLevels(GLMContext ctx, Texture *tex)
{
    for (int face = 0; face < 6; face++)
    {
        if (tex->faces[face].levels)
            continue;
        tex->faces[face].levels = (TextureLevel *)calloc(TD_MAX_LEVELS, sizeof(TextureLevel));
        if (!tex->faces[face].levels)
        {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
    }
    return true;
}

static inline TextureLevel *texLevel(Texture *tex, GLuint face, GLuint level)
{
    if (face >= 6 || !tex->faces[face].levels || level >= TD_MAX_LEVELS)
        return NULL;
    /* renderbuffer textures (framebuffers.c) carry a single level */
    if (tex->target == GL_RENDERBUFFER && (face || level))
        return NULL;
    return &tex->faces[face].levels[level];
}

/* GL level internal format (stored in the otherwise unused TextureLevel.mtl_format) */
static inline GLenum levelFormat(Texture *tex, GLuint face, GLuint level)
{
    TextureLevel *l = texLevel(tex, face, level);
    if (l && tex->target == GL_RENDERBUFFER)
        return l->complete ? (l->mtl_format ? l->mtl_format : tex->internalformat) : 0;
    return (l && l->complete) ? l->mtl_format : 0;
}

/* Metal geometry of a GL level: 1D-array layers become slices */
void mgl_texdata_level_geometry(Texture *tex, GLuint face, GLuint level, GLuint *w, GLuint *h, GLuint *d)
{
    TextureLevel *l = texLevel(tex, face, level);

    if (!l || !l->complete)
    {
        *w = *h = *d = 0;
        return;
    }

    *w = l->width;
    if (tex->target == GL_TEXTURE_1D_ARRAY)
    {
        *h = 1;
        *d = l->height;
    }
    else
    {
        *h = l->height;
        *d = l->depth;
    }
}

const TDFormat *mgl_texdata_storage_format(GLenum internalformat)
{
    return tdStorageFormat(tdFormatInfo(internalformat));
}

static size_t shadowBytes(const TDFormat *sf, GLuint w, GLuint h, GLuint d, size_t *row, size_t *img, size_t *stencil_off)
{
    *row = tdStorageRowBytes(sf, (int)w);
    *img = tdStorageImageBytes(sf, (int)w, (int)h);
    size_t depth_plane = *img * d;
    *stencil_off = depth_plane;
    if (sf->kind == TDK_D32FS8)
        return depth_plane + (size_t)w * h * d;
    return depth_plane;
}

static void storageFromBlock(const TDFormat *sf, void *p, GLuint w, GLuint h, GLuint d, TDStorage *st)
{
    size_t row, img, soff;
    shadowBytes(sf, w, h, d, &row, &img, &soff);
    st->data = p;
    st->row_pitch = row;
    st->image_pitch = img;
    if (sf->kind == TDK_D32FS8)
    {
        st->stencil = (uint8_t *)p + soff;
        st->s_row_pitch = w;
        st->s_image_pitch = (size_t)w * h;
    }
    else
    {
        st->stencil = NULL;
        st->s_row_pitch = st->s_image_pitch = 0;
    }
}

bool mgl_texdata_shadow(GLMContext ctx, Texture *tex, GLuint face, GLuint level, bool alloc, TDStorage *st)
{
    TextureLevel *l = texLevel(tex, face, level);
    GLuint w, h, d;
    size_t row, img, soff, size;

    if (!l || !l->complete)
        return false;

    const TDFormat *sf = mgl_texdata_storage_format(l->mtl_format);
    if (!sf)
        return false;

    mgl_texdata_level_geometry(tex, face, level, &w, &h, &d);
    size = shadowBytes(sf, w, h, d, &row, &img, &soff);

    if (!l->data)
    {
        if (!alloc)
            return false;
        void *p = calloc(1, size ? size : 1);
        if (!p)
        {
            ERROR_RETURN_VALUE(GL_OUT_OF_MEMORY, false);
        }
        l->data = (vm_address_t)p;
        l->data_size = size;
    }

    l->pitch = row;
    storageFromBlock(sf, (void *)l->data, w, h, d, st);
    return true;
}

void mgl_texdata_free_shadow(GLMContext ctx, Texture *tex, GLuint face, GLuint level)
{
    TextureLevel *l = texLevel(tex, face, level);
    (void)ctx;

    if (l && l->data)
    {
        free((void *)l->data);
        l->data = 0;
        l->data_size = 0;
    }
}

static GLuint fullChainLevels(Texture *tex, GLuint w, GLuint h, GLuint d)
{
    GLuint m = w;
    if (tex->target == GL_TEXTURE_RECTANGLE || tex->target == GL_RENDERBUFFER)
        return 1;
    if (tex->target != GL_TEXTURE_1D && tex->target != GL_TEXTURE_1D_ARRAY && h > m) m = h;
    if (tex->target == GL_TEXTURE_3D && d > m) m = d;
    return log2u(m) + 1;
}

/* dimensions of level `level` given level 0 dims (GL geometry) */
static void mipDims(Texture *tex, GLuint level, GLuint w0, GLuint h0, GLuint d0, GLuint *w, GLuint *h, GLuint *d)
{
    *w = MAX(1, w0 >> level);
    *h = (tex->target == GL_TEXTURE_1D || tex->target == GL_TEXTURE_1D_ARRAY) ? h0 : MAX(1, h0 >> level);
    *d = (tex->target == GL_TEXTURE_3D) ? MAX(1, d0 >> level) : d0;
}

static void updateTexSummary(Texture *tex)
{
    GLuint highest = 0;
    bool any = false;
    GLuint faces = mgl_texdata_num_faces(tex);

    if (tex->immutable_storage)
        return;

    for (GLuint f = 0; f < faces; f++)
        for (GLuint l = 0; l < TD_MAX_LEVELS; l++)
            if (tex->faces[f].levels && tex->faces[f].levels[l].complete)
            {
                any = true;
                if (l + 1 > highest) highest = l + 1;
            }

    tex->num_levels = any ? highest : 0;

    // base dims: from level 0 if defined, else derived from the lowest level
    for (GLuint l = 0; l < TD_MAX_LEVELS; l++)
    {
        for (GLuint f = 0; f < faces; f++)
        {
            if (!tex->faces[f].levels || !tex->faces[f].levels[l].complete)
                continue;
            TextureLevel *lv = &tex->faces[f].levels[l];
            bool lin = (tex->target == GL_TEXTURE_1D || tex->target == GL_TEXTURE_1D_ARRAY);
            tex->width = lv->width << l;
            tex->height = lin ? lv->height : lv->height << l;
            tex->depth = tex->target == GL_TEXTURE_3D ? lv->depth << l : lv->depth;
            /* keep a SPECIFIC compressed request: CopyImageSubData validation uses it
             * even when the level stores the decoded (online-compressed) format */
            {
                const TDFormat *creq = tdFormatInfo(tex->internalformat);
                if (!(creq && (creq->flags & TDF_COMPRESSED)))
                    tex->internalformat = lv->mtl_format;
            }
            tex->mipmap_levels = MAX(fullChainLevels(tex, tex->width, tex->height, tex->depth), highest);
            tex->is_array = isArrayTarget(tex->target);
            return;
        }
    }

    tex->width = tex->height = tex->depth = 0;
    tex->mipmap_levels = 0;
}

/* (re)define a level; content becomes undefined unless written afterwards */
static bool defineLevel(GLMContext ctx, Texture *tex, GLuint face, GLuint level, GLenum ifmt, GLuint w, GLuint h, GLuint d)
{
    if (!ensureLevels(ctx, tex))
        return false;

    TextureLevel *l = texLevel(tex, face, level);
    TDTexInfo *info = mgl_texdata_info(ctx, tex);

    if (!l)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, false);
    }

    mgl_texdata_free_shadow(ctx, tex, face, level);

    if (info)
    {
        info->on_gpu[face] &= ~(1u << level);
        if (info->orig[face][level])
        {
            free(info->orig[face][level]);
            info->orig[face][level] = NULL;
            info->orig_size[face][level] = 0;
        }
    }

    l->complete = (w && h && d) ? GL_TRUE : GL_FALSE;
    l->width = l->complete ? w : 0;
    l->height = l->complete ? h : 0;
    l->depth = l->complete ? d : 0;
    l->mtl_format = l->complete ? ifmt : 0;
    l->pitch = 0;

    updateTexSummary(tex);

    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    STATE(dirty_bits) |= DIRTY_TEX;

    return true;
}

static void undefineAllLevels(GLMContext ctx, Texture *tex)
{
    for (int f = 0; f < 6; f++)
        for (int l = 0; l < TD_MAX_LEVELS; l++)
            if (tex->faces[f].levels && tex->faces[f].levels[l].complete)
                defineLevel(ctx, tex, f, l, 0, 0, 0, 0);
}

/* Shape of the MTLTexture to realize for a texture (Metal geometry). */
bool mgl_texdata_shape(GLMContext ctx, Texture *tex, TDShape *shape)
{
    GLuint faces = mgl_texdata_num_faces(tex);
    GLuint base = TD_MAX_LEVELS, bf = 0;

    (void)ctx;
    memset(shape, 0, sizeof(*shape));

    if (!tex->faces[0].levels)
        return false;

    for (GLuint l = 0; l < TD_MAX_LEVELS && base == TD_MAX_LEVELS; l++)
        for (GLuint f = 0; f < faces; f++)
            if (tex->faces[f].levels[l].complete) { base = l; bf = f; break; }

    if (base == TD_MAX_LEVELS)
        return false;

    TextureLevel *bl = &tex->faces[bf].levels[base];
    bool lin = (tex->target == GL_TEXTURE_1D || tex->target == GL_TEXTURE_1D_ARRAY);
    GLuint w0 = bl->width << base;
    GLuint h0 = lin ? bl->height : bl->height << base;
    GLuint d0 = (tex->target == GL_TEXTURE_3D) ? bl->depth << base : bl->depth;
    GLenum ifmt = bl->mtl_format;
    GLuint levels;

    if (tex->immutable_storage)
    {
        levels = tex->num_levels;
    }
    else
    {
        GLuint maxl = MIN(fullChainLevels(tex, w0, h0, d0), TD_MAX_LEVELS);
        levels = base + 1;
        for (GLuint l = base + 1; l < maxl; l++)
        {
            bool ok = false;
            for (GLuint f = 0; f < faces; f++)
            {
                TextureLevel *lv = &tex->faces[f].levels[l];
                if (!lv->complete)
                {
                    ok = false;
                    break;
                }
                GLuint ew, eh, ed;
                mipDims(tex, l, w0, h0, d0, &ew, &eh, &ed);
                ok = (lv->width == ew && lv->height == eh && lv->depth == ed && lv->mtl_format == ifmt);
                if (!ok)
                    break;
            }
            if (!ok)
                break;
            levels = l + 1;
        }
    }

    shape->target = tex->target;
    shape->internalformat = ifmt;
    shape->width = w0;
    if (tex->target == GL_TEXTURE_1D_ARRAY)
    {
        shape->height = 1;
        shape->depth = h0;
    }
    else
    {
        shape->height = h0;
        shape->depth = d0;
    }
    shape->levels = levels;

    return true;
}

/* ------------------------------------------------------------------------ */
/* pixel store and pixel buffer objects                                     */
/* ------------------------------------------------------------------------ */

static TDPixelStore pixelStore(GLMContext ctx, bool unpack)
{
    PixelStore *p = unpack ? &STATE(unpack) : &STATE(pack);
    struct MGLExtState_texdata *td = TDS(ctx);
    TDPixelStore ps;
    int *cb = td ? (unpack ? td->unpack_cb : td->pack_cb) : NULL;

    ps.swap_bytes = p->swap_bytes;
    ps.lsb_first = p->lsb_first;
    ps.row_length = p->row_length;
    ps.image_height = p->image_height;
    ps.skip_rows = p->skip_rows;
    ps.skip_pixels = p->skip_pixels;
    ps.skip_images = p->skip_images;
    ps.alignment = p->alignment ? p->alignment : 4;
    ps.cb_width = cb ? cb[0] : 0;
    ps.cb_height = cb ? cb[1] : 0;
    ps.cb_depth = cb ? cb[2] : 0;
    ps.cb_size = cb ? cb[3] : 0;
    return ps;
}

/* CPU storage of a buffer object (the Metal buffer contents once it has one) */
static uint8_t *bufferBase(GLMContext ctx, Buffer *buf)
{
    if (TDM(ctx) && TDM(ctx)->buffer_contents)
        return (uint8_t *)TDM(ctx)->buffer_contents(ctx, buf);
    return (uint8_t *)buf->data.buffer_data;
}

/* resolve a client pointer for reading (unpack) or writing (pack); with a PBO
 * bound `ptr` is an offset. Returns GL_NO_ERROR or the error to raise. */
static GLenum resolveClientPtr(GLMContext ctx, bool unpack, const void *ptr, size_t extent, size_t elem_align,
                               uint8_t **out, Buffer **pbo_out)
{
    Buffer *pbo = STATE(buffers[unpack ? _PIXEL_UNPACK_BUFFER : _PIXEL_PACK_BUFFER]);

    if (pbo_out) *pbo_out = pbo;

    if (!pbo)
    {
        *out = (uint8_t *)ptr;
        return GL_NO_ERROR;
    }

    size_t offset = (size_t)ptr;

    if (pbo->mapped)
        return GL_INVALID_OPERATION;
    if (pbo->size < 0 || offset > (size_t)pbo->size || extent > (size_t)pbo->size - offset)
        return GL_INVALID_OPERATION;
    if (elem_align > 1 && (offset % elem_align) != 0)
        return GL_INVALID_OPERATION;
    uint8_t *base = bufferBase(ctx, pbo);
    if (!base)
    {
        *out = NULL;
        return extent ? GL_INVALID_OPERATION : GL_NO_ERROR;
    }

    *out = base + offset;
    return GL_NO_ERROR;
}

/* a pack operation wrote [dst, dst+size) of the CPU copy of a pixel pack buffer:
 * propagate it to the Metal buffer (fb domain hook) */
static void pboWrittenRange(GLMContext ctx, Buffer *pbo, const uint8_t *dst, size_t size)
{
    if (!pbo)
        return;
    size_t offset = (size_t)(dst - bufferBase(ctx, pbo));
    if (pbo->data.mtl_data && ctx->mtl_funcs.ext_fb && ctx->mtl_funcs.ext_fb->write_buffer)
        ctx->mtl_funcs.ext_fb->write_buffer(ctx, pbo, offset, size, dst);
    else
        pbo->data.dirty_bits |= DIRTY_BUFFER_DATA;
}

static size_t typeAlign(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT:
        case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV: case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV: case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return 2;
        case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV: case GL_UNSIGNED_INT_10_10_10_2: case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_24_8: case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 4;
        default:
            return 1;
    }
}

/* ------------------------------------------------------------------------ */
/* data path: storage-layout regions to/from a level                        */
/* ------------------------------------------------------------------------ */

static bool levelOnGPU(GLMContext ctx, Texture *tex, GLuint face, GLuint level)
{
    TDTexInfo *info = mgl_texdata_find_info(ctx, tex);
    return info && info->realized && tex->mtl_data && (info->on_gpu[face] & (1u << level)) && level < info->shape.levels;
}

/* copy a region between a tight buffer and a full-level buffer at (x,y,z) */
static void copyStorageRegion(const TDFormat *sf, const TDStorage *tight, const TDStorage *full, const TDRegion *r,
                              bool to_full, bool depth_plane, bool stencil_plane)
{
    size_t bx, by, rowbytes, rows;

    if (sf->flags & TDF_COMPRESSED)
    {
        bx = (size_t)r->x / sf->block_w * sf->block_bytes;
        by = (size_t)r->y / sf->block_h;
        rowbytes = tdStorageRowBytes(sf, r->w);
        rows = ((size_t)r->h + sf->block_h - 1) / sf->block_h;
    }
    else
    {
        bx = (size_t)r->x * sf->bpp;
        by = (size_t)r->y;
        rowbytes = (size_t)r->w * sf->bpp;
        rows = (size_t)r->h;
    }

    if (depth_plane)
    {
        for (GLsizei z = 0; z < r->d; z++)
            for (size_t y = 0; y < rows; y++)
            {
                uint8_t *f = (uint8_t *)full->data + (size_t)(r->z + z) * full->image_pitch + (by + y) * full->row_pitch + bx;
                uint8_t *t = (uint8_t *)tight->data + (size_t)z * tight->image_pitch + y * tight->row_pitch;
                if (to_full) memcpy(f, t, rowbytes); else memcpy(t, f, rowbytes);
            }
    }

    if (stencil_plane && sf->kind == TDK_D32FS8 && tight->stencil && full->stencil)
    {
        for (GLsizei z = 0; z < r->d; z++)
            for (GLsizei y = 0; y < r->h; y++)
            {
                uint8_t *f = (uint8_t *)full->stencil + (size_t)(r->z + z) * full->s_image_pitch + (size_t)(r->y + y) * full->s_row_pitch + r->x;
                uint8_t *t = (uint8_t *)tight->stencil + (size_t)z * tight->s_image_pitch + (size_t)y * tight->s_row_pitch;
                if (to_full) memcpy(f, t, (size_t)r->w); else memcpy(t, f, (size_t)r->w);
            }
    }
}

/* write a tightly packed storage-layout region (Metal geometry) into a level */
static bool writeRegion(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *src, bool depth, bool stencil)
{
    const TDFormat *sf = mgl_texdata_storage_format(levelFormat(tex, r->face, r->level));

    if (!sf)
        return false;
    if (r->w <= 0 || r->h <= 0 || r->d <= 0)
        return true;

    if (sf->kind != TDK_D32FS8)
        depth = stencil = true;

    if (levelOnGPU(ctx, tex, r->face, r->level) && TDM(ctx) && TDM(ctx)->upload)
    {
        TDM(ctx)->upload(ctx, tex, r, src, depth, stencil);
        return true;
    }

    TDStorage full;
    if (!mgl_texdata_shadow(ctx, tex, r->face, r->level, true, &full))
        return false;

    copyStorageRegion(sf, src, &full, r, true, depth, stencil);

    tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    STATE(dirty_bits) |= DIRTY_TEX;
    return true;
}

/* read a region (Metal geometry) of a level into a tight storage-layout buffer */
static bool readRegion(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *dst)
{
    const TDFormat *sf = mgl_texdata_storage_format(levelFormat(tex, r->face, r->level));
    TDStorage full;

    if (!sf)
        return false;

    if (mgl_texdata_shadow(ctx, tex, r->face, r->level, false, &full))
    {
        copyStorageRegion(sf, dst, &full, r, false, true, true);
        return true;
    }

    if (!TDM(ctx) || !TDM(ctx)->download)
        return false;

    if (!levelOnGPU(ctx, tex, r->face, r->level))
    {
        // the level has neither CPU data nor GPU data yet: realize it (undefined content)
        if (TDM(ctx)->realize && !TDM(ctx)->realize(ctx, tex))
            return false;
        if (mgl_texdata_shadow(ctx, tex, r->face, r->level, false, &full))
        {
            copyStorageRegion(sf, dst, &full, r, false, true, true);
            return true;
        }
        if (!levelOnGPU(ctx, tex, r->face, r->level))
        {
            // level outside the realizable shape (e.g. inconsistent mip): no data was ever written
            size_t row, img, soff;
            size_t n = shadowBytes(sf, r->w, r->h, r->d, &row, &img, &soff);
            memset(dst->data, 0, n);
            return true;
        }
    }

    return TDM(ctx)->download(ctx, tex, r, dst);
}

/* tight storage buffer for a region (Metal geometry) */
static void *allocStorage(const TDFormat *sf, GLsizei w, GLsizei h, GLsizei d, TDStorage *st)
{
    size_t row, img, soff;
    size_t size = shadowBytes(sf, (GLuint)w, (GLuint)h, (GLuint)d, &row, &img, &soff);
    void *p = calloc(1, size ? size : 1);

    if (!p)
        return NULL;

    storageFromBlock(sf, p, (GLuint)w, (GLuint)h, (GLuint)d, st);
    return p;
}

/* GL region -> Metal region (1D arrays: y/height are layers) */
static TDRegion glRegion(Texture *tex, GLuint face, GLuint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d)
{
    TDRegion r;
    r.face = face;
    r.level = level;
    if (tex->target == GL_TEXTURE_1D_ARRAY)
    {
        r.x = x; r.y = 0; r.z = y;
        r.w = w; r.h = 1; r.d = h;
    }
    else
    {
        r.x = x; r.y = y; r.z = z;
        r.w = w; r.h = h; r.d = d;
    }
    return r;
}

/* storage of a GL-geometry region: a 1D array region (w x h layers) is laid out
 * like a 2D image, which is byte-identical to h slices of one row each */
static void glToMetalStorage(Texture *tex, TDStorage *st)
{
    if (tex->target == GL_TEXTURE_1D_ARRAY)
    {
        st->image_pitch = st->row_pitch;
        st->s_image_pitch = st->s_row_pitch;
    }
}

/* upload client pixels into a GL region of a level */
static void uploadClient(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                         GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                         GLenum format, GLenum type, const uint8_t *src, bool is3d)
{
    const TDFormat *sf = mgl_texdata_storage_format(levelFormat(tex, face, level));
    TDPixelStore ps = pixelStore(ctx, true);
    size_t off, row, img;
    TDStorage st;

    if (!sf || !src || w <= 0 || h <= 0 || d <= 0 || (sf->flags & TDF_COMPRESSED))
        return;

    tdClientLayout(&ps, format, type, w, h, d, is3d, &off, &row, &img);

    if (!allocStorage(sf, w, h, d, &st))
        TD_ERR(GL_OUT_OF_MEMORY);

    bool wd = true, ws = true;
    tdUnpack(sf, src + off, row, img, format, type, ps.swap_bytes, w, h, d, &st, &wd, &ws);
    glToMetalStorage(tex, &st);

    TDRegion r = glRegion(tex, face, level, x, y, z, w, h, d);
    bool ok = writeRegion(ctx, tex, &r, &st, wd, ws);
    free(st.data);
    if (!ok)
        TD_ERR(GL_OUT_OF_MEMORY);
}

/* ------------------------------------------------------------------------ */
/* TexImage                                                                 */
/* ------------------------------------------------------------------------ */

typedef struct {
    GLenum   tex_target;    /* object target (cube faces -> GL_TEXTURE_CUBE_MAP) */
    GLuint   face;
    bool     proxy;
} TDTarget;

static bool texImageTarget(GLenum target, int dims, TDTarget *t)
{
    t->face = 0;
    t->proxy = false;

    if (dims == 1)
    {
        switch (target)
        {
            case GL_TEXTURE_1D: t->tex_target = target; return true;
            case GL_PROXY_TEXTURE_1D: t->tex_target = GL_TEXTURE_1D; t->proxy = true; return true;
            default: return false;
        }
    }

    if (dims == 2)
    {
        if (isCubeFace(target))
        {
            t->tex_target = GL_TEXTURE_CUBE_MAP;
            t->face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            return true;
        }
        switch (target)
        {
            case GL_TEXTURE_2D: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_RECTANGLE:
                t->tex_target = target; return true;
            case GL_PROXY_TEXTURE_2D: case GL_PROXY_TEXTURE_1D_ARRAY: case GL_PROXY_TEXTURE_RECTANGLE:
            case GL_PROXY_TEXTURE_CUBE_MAP:
                t->tex_target = proxyTexTarget(target); t->proxy = true; return true;
            default: return false;
        }
    }

    switch (target)
    {
        case GL_TEXTURE_3D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
            t->tex_target = target; return true;
        case GL_PROXY_TEXTURE_3D: case GL_PROXY_TEXTURE_2D_ARRAY: case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
            t->tex_target = proxyTexTarget(target); t->proxy = true; return true;
        default: return false;
    }
}

static GLuint maxLevelForTarget(GLMContext ctx, GLenum tt)
{
    switch (tt)
    {
        case GL_TEXTURE_3D: return log2u(max3DSize(ctx));
        case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY: return log2u(maxCubeSize(ctx));
        case GL_TEXTURE_RECTANGLE: return 0;
        default: return MIN(log2u(maxTexSize(ctx)), TD_MAX_LEVELS - 1);
    }
}

/* size limit check (GL geometry); returns GL_NO_ERROR or GL_INVALID_VALUE */
static GLenum checkSize(GLMContext ctx, GLenum tt, GLint level, GLsizei w, GLsizei h, GLsizei d)
{
    GLuint m = maxTexSize(ctx);
    unsigned long long lw = (unsigned long long)w << level, lh = (unsigned long long)h << level, ld = (unsigned long long)d << level;

    if (w < 0 || h < 0 || d < 0)
        return GL_INVALID_VALUE;

    switch (tt)
    {
        case GL_TEXTURE_1D:
            if (lw > m) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_1D_ARRAY:
            if (lw > m || (GLuint)h > maxLayers(ctx)) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_2D:
            if (lw > m || lh > m) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_RECTANGLE:
            if ((GLuint)w > maxRectSize(ctx) || (GLuint)h > maxRectSize(ctx)) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_CUBE_MAP:
            if (lw > maxCubeSize(ctx) || lh > maxCubeSize(ctx) || w != h) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            if (lw > maxCubeSize(ctx) || lh > maxCubeSize(ctx) || w != h || (d % 6) != 0 || (GLuint)d > maxLayers(ctx) * 6)
                return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_2D_ARRAY:
            if (lw > m || lh > m || (GLuint)d > maxLayers(ctx)) return GL_INVALID_VALUE;
            break;
        case GL_TEXTURE_3D:
            if (lw > max3DSize(ctx) || lh > max3DSize(ctx) || ld > max3DSize(ctx)) return GL_INVALID_VALUE;
            break;
    }
    return GL_NO_ERROR;
}

static bool depthFormatAllowedOnTarget(GLenum tt)
{
    return tt != GL_TEXTURE_3D;
}

static bool compressedAllowedOnTarget(const TDFormat *f, GLenum tt)
{
    switch (tt)
    {
        case GL_TEXTURE_2D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY:
            return true;
        case GL_TEXTURE_3D:
            /* BPTC is the only core family that allows 3D (Table 8.17) */
            return f->internalformat == GL_COMPRESSED_RGBA_BPTC_UNORM || f->internalformat == GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM ||
                   f->internalformat == GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT || f->internalformat == GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
        default:
            return false;
    }
}

/* proxy state lives in texparam's proxy objects (queried by GetTexLevelParameter) */
static bool storeCompressed(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                            GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, const uint8_t *blocks);
static bool uploadRGTC(GLMContext ctx, Texture *tex, GLuint face, GLuint level, GLint ifmt,
                       GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels);

static void setProxy(GLMContext ctx, Texture *proxy, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLsizei d, bool ok)
{
    if (!proxy || !ensureLevels(ctx, proxy))
        return;

    GLuint faces = (proxy->target == GL_TEXTURE_CUBE_MAP) ? 6 : 1;
    for (GLuint f = 0; f < faces; f++)
    {
        TextureLevel *l = texLevel(proxy, f, level);
        if (!l) continue;
        l->complete = ok;
        l->width = ok ? w : 0;
        l->height = ok ? h : 0;
        l->depth = ok ? d : 0;
        l->mtl_format = ok ? ifmt : 0;
    }
    if (level == 0 || ok)
        proxy->internalformat = ok ? ifmt : 0;
    if (level == 0)
    {
        proxy->width = ok ? w : 0;
        proxy->height = ok ? h : 0;
        proxy->depth = ok ? d : 0;
    }
    proxy->mipmap_levels = TD_MAX_LEVELS;
    proxy->num_levels = 0;
    for (GLuint l = 0; l < TD_MAX_LEVELS; l++)
        if (proxy->faces[0].levels[l].complete)
            proxy->num_levels = l + 1;
}

/* the RGTC family (BC4/BC5): the only compressed formats with a CPU encoder
 * (tdEncodeRGTC), so the only ones MGL can online-compress round-trip */
static bool isRgtcFormat(const TDFormat *f)
{
    return f && (f->flags & TDF_BC) && (f->base == GL_RED || f->base == GL_RG);
}

/* BPTC (BC6H/BC7): the one BC family whose 3D support is specified
 * (table 8.19 "3D Tex." column; Mesa gates on ARB_texture_compression_bptc,
 * which MGL exposes) */
static bool isBptcFormat(const TDFormat *f)
{
    if (!f || !(f->flags & TDF_BC))
        return false;
    switch (f->internalformat)
    {
        case GL_COMPRESSED_RGBA_BPTC_UNORM:
        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT:
        case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            return true;
        default:
            return false;
    }
}

static void texImage(GLMContext ctx, GLenum target, int dims, GLint level, GLint internalformat,
                     GLsizei width, GLsizei height, GLsizei depth, GLint border,
                     GLenum format, GLenum type, const void *pixels)
{
    TDTarget t;
    GLenum err;
    bool rgtc_online = false;

    TD_CHECK(texImageTarget(target, dims, &t), GL_INVALID_ENUM);
    TD_CHECK(level >= 0 && (GLuint)level <= maxLevelForTarget(ctx, t.tex_target), GL_INVALID_VALUE);

    err = tdValidateFormatType(format, type);
    TD_CHECK(err == GL_NO_ERROR, err);

    GLenum ifmt = tdResolveInternalFormat(internalformat, format, type);
    TD_CHECK(ifmt != 0, GL_INVALID_VALUE);
    const TDFormat *f = tdFormatInfo(ifmt);
    const TDFormat *req = tdFormatInfo(internalformat);   /* as requested, before any
                                                           * resolution to an emulated format */
    if (!req)
        req = f;   /* unknown to the table: behave as before the resolution */

    TD_CHECK(border == 0, GL_INVALID_VALUE);

    err = checkSize(ctx, t.tex_target, level, width, height, depth);
    if (err != GL_NO_ERROR)
    {
        if (t.proxy && width >= 0 && height >= 0 && depth >= 0)
        {
            setProxy(ctx, mgl_texparam_proxy_texture(ctx, target), level, ifmt, width, height, depth, false);
            return;
        }
        TD_ERR(err);
    }

    err = tdCheckFormatCompat(f, format, type, TD_PIX_TEXIMAGE);
    TD_CHECK(err == GL_NO_ERROR, err);

    if (f->flags & (TDF_DEPTH | TDF_STENCIL))
        TD_CHECK(depthFormatAllowedOnTarget(t.tex_target), GL_INVALID_OPERATION);

    /* GL 4.6 spec, TexImage3D errors: "an INVALID_OPERATION error is generated
     * ... if internalformat is one of the EAC, ETC2, or RGTC compressed formats
     * and either border is non-zero, or target is not TEXTURE_2D_ARRAY"
     * (Mesa _mesa_target_can_be_compressed rejects the same).  S3TC has no 3D
     * support either (EXT_texture_compression_s3tc defines none; Mesa rejects
     * it with an error too), while BPTC supports 3D (spec table 8.19, "3D
     * Tex." column).  TEXTURE_2D_ARRAY is a legal target for every family —
     * its blocks are per-layer, so the blanket 2D_ARRAY rejection that used to
     * live here was over-broad and broke the uncompressed (online-compressed)
     * RGTC2 uploads of KHR-GL46.copy_image.incompatible_formats_compression;
     * KHR-GL46.packed_pixels.varied_rectangle.compressed_*_rgtc* exercises
     * the rejected TEXTURE_3D path (it uploads with UNPACK_IMAGE_HEIGHT). */
    if (t.tex_target == GL_TEXTURE_3D &&
        (((req->flags & TDF_ETC)) || ((req->flags & TDF_BC) && !isBptcFormat(req))))
        TD_CHECK(0, GL_INVALID_OPERATION);

    if (req->flags & TDF_COMPRESSED)
    {
        /* uncompressed client data given to a compressed internal format has to be
         * compressed by the GL ("online compression", GL 4.6 8.5.3).  MGL has a CPU
         * encoder only for the RGTC family (BC4/BC5), whose downloads re-encode from
         * the decoded storage; the other families report the "no online compression"
         * error instead of storing a decoded level that would report
         * GL_TEXTURE_COMPRESSED=TRUE while glGetCompressedTexImage fails.  Levels
         * created this way store the DECODED equivalent (R8/RG8), keeping uploads
         * and downloads byte-exact; tdEncodeRGTC remains for explicit
         * glCompressedTex* block uploads. */
        if (pixels && !isRgtcFormat(req))
            TD_CHECK(0, GL_INVALID_OPERATION);
        if (f->flags & TDF_COMPRESSED)
        {
            /* no decoded equivalent in the table (should not happen): reject */
            TD_CHECK(0, GL_INVALID_OPERATION);
        }
    }

    if (t.proxy)
    {
        setProxy(ctx, mgl_texparam_proxy_texture(ctx, target), level, ifmt, width, height, depth, true);
        return;
    }

    Texture *tex = getTex(ctx, 0, t.tex_target);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(!tex->immutable_storage, GL_INVALID_OPERATION);

    /* remember the compressed request on the texture object: CopyImageSubData
     * compatibility and block-alignment validation must use it even though the
     * level stores the decoded format (KHR-GL46 copy_image invalid_alignment /
     * incompatible_formats_compression) */
    if ((req->flags & TDF_COMPRESSED) && ifmt != internalformat)
        tex->internalformat = internalformat;

    /* validate the source before changing anything */
    uint8_t *src = NULL;
    if (width && height && depth && (pixels || STATE(buffers[_PIXEL_UNPACK_BUFFER])))
    {
        TDPixelStore ps = pixelStore(ctx, true);
        size_t extent = tdClientLayout(&ps, format, type, width, height, depth, dims == 3, NULL, NULL, NULL);
        err = resolveClientPtr(ctx, true, pixels, extent, typeAlign(type), &src, NULL);
        TD_CHECK(err == GL_NO_ERROR, err);
    }

    if (!defineLevel(ctx, tex, t.face, level, ifmt, width, height, depth))
        return;

    if (!tex->access)
        tex->access = GL_READ_ONLY;

    if (src)
    {
        if (rgtc_online)
            uploadRGTC(ctx, tex, t.face, level, ifmt, width, height, depth, format, type, src);
        else
            uploadClient(ctx, tex, t.face, level, 0, 0, 0, width, height, depth, format, type, src, dims == 3);
    }
}

/* used by framebuffers.c for renderbuffer storage */
bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy)
{
    GLenum ifmt = tdResolveInternalFormat(internalformat, format, type);

    (void)is_array; (void)proxy;

    if (!tex || ifmt == 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, false);
    }

    if (!defineLevel(ctx, tex, face, level, ifmt, width, height, depth))
        return false;

    if (pixels)
        uploadClient(ctx, tex, face, level, 0, 0, 0, width, height, depth, format, type, (const uint8_t *)pixels, depth > 1);

    return true;
}

static bool uploadRGTC(GLMContext ctx, Texture *tex, GLuint face, GLuint level, GLint ifmt,
                       GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    const TDFormat *f = tdFormatInfo(ifmt);
    TDStorage st;
    const uint8_t *src = (const uint8_t *)pixels;
    TDPixelStore ps;
    uint8_t *planes[2] = { NULL, NULL };
    size_t need = (size_t)width * height * (depth > 0 ? (size_t)depth : 1);
    size_t row = (size_t)width * 4;
    uint8_t *blocks = NULL;
    uint8_t *heap = NULL;
    int nch = (f->base == GL_RG) ? 2 : 1;
    size_t pr = (size_t)width;
    bool snorm = (f->flags & TDF_SIGNED) != 0;

    if (!need || need > SIZE_MAX / 4)
        return false;

    /* client (format, type) -> RGBA8, then split the channel(s) into planes for the
     * encoder (it takes one or two 8-bit sample planes with row / image pitch) */
    ps = pixelStore(ctx, true);
    st.row_pitch = row;
    st.image_pitch = row * height;
    st.s_row_pitch = st.s_image_pitch = 0;
    st.data = st.stencil = NULL;

    if (need * 4 > 64 * 1024 * 1024)
        return false;                       /* guard: absurd "uncompressed" uploads */

    heap = (uint8_t *)malloc(need * 4);
    planes[0] = (uint8_t *)malloc(need);
    if (nch == 2) planes[1] = (uint8_t *)malloc(need);
    if (!heap || !planes[0] || (nch == 2 && !planes[1]))
    {
        free(heap); free(planes[0]); free(planes[1]);
        return false;
    }
    st.data = heap;
    {
        size_t c_off, c_row, c_img;
        bool wd = true, ws = true;
        tdClientLayout(&ps, format, type, width, height, depth, depth > 1, &c_off, &c_row, &c_img);
        tdUnpack(tdFormatInfo(GL_RGBA8), src + c_off, c_row, c_img, format, type, ps.swap_bytes,
                 width, height, depth, &st, &wd, &ws);
    }

    for (GLsizei z = 0; z < (depth > 0 ? depth : 1); z++)
        for (GLsizei y = 0; y < height; y++)
        {
            const uint8_t *r = heap + (size_t)z * st.image_pitch + (size_t)y * st.row_pitch;
            uint8_t *d0 = planes[0] + (size_t)z * pr * height + (size_t)y * pr;
            uint8_t *d1 = planes[1] ? planes[1] + (size_t)z * pr * height + (size_t)y * pr : NULL;

            for (GLsizei x = 0; x < width; x++)
            {
                d0[x] = r[x * 4];
                if (d1) d1[x] = r[x * 4 + 1];
            }
        }

    blocks = tdEncodeRGTC(planes[0], planes[1], width, height, depth, pr, pr * height, nch == 2, snorm);
    free(heap);
    free(planes[0]);
    free(planes[1]);
    if (!blocks)
        return false;

    bool ok = storeCompressed(ctx, tex, face, level, 0, 0, 0, width, height, depth, blocks);
    free(blocks);
    return ok;
}

/*
 * Online compression for sub-images: uncompressed client data written into a
 * REAL compressed level (texStorage'd RGTC / BPTC).  The client data is
 * converted to the encoder's input layout and encoded to blocks, which are
 * then stored block-aligned exactly like an explicit glCompressedTexSubImage —
 * so the level keeps its hardware blocks and glGetCompressedTexImage reads
 * them back byte-exact (GL 4.6 8.5.3: "the GL will compress").
 * The caller has already validated the region and its block alignment.
 */
static bool uploadOnlineCompressed(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                                   GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                                   GLenum format, GLenum type, const void *pixels)
{
    const TDFormat *f = tdFormatInfo(levelFormat(tex, face, level));
    TDStorage st;
    const uint8_t *src = (const uint8_t *)pixels;
    TDPixelStore ps;
    uint8_t *heap = NULL, *blocks = NULL;
    size_t need = (size_t)w * h * (size_t)(d > 0 ? d : 1);
    bool ok = false;

    if (!f || !need || need > SIZE_MAX / 16)
        return false;

    ps = pixelStore(ctx, true);
    if (isRgtcFormat(f) || (isBptcFormat(f) && !(f->flags & TDF_BPTC_FLOAT)))
    {
        /* byte path: RGTC wants one or two 8-bit planes, BC7 wants RGBA8 */
        size_t row = (size_t)w * 4;
        st.row_pitch = row;
        st.image_pitch = row * h;
        st.s_row_pitch = st.s_image_pitch = 0;
        st.data = st.stencil = NULL;
        heap = (uint8_t *)malloc(need * 4);
        if (!heap)
            return false;
        st.data = heap;
    }
    else
    {
        /* float path: BC6H takes RGBA32F */
        size_t row = (size_t)w * 16;
        st.row_pitch = row;
        st.image_pitch = row * h;
        st.s_row_pitch = st.s_image_pitch = 0;
        st.data = st.stencil = NULL;
        heap = (uint8_t *)malloc(need * 16);
        if (!heap)
            return false;
        st.data = heap;
    }

    {
        size_t c_off, c_row, c_img;
        bool wd = true, ws = true;
        GLenum dst_ifmt = (isBptcFormat(f) && (f->flags & TDF_BPTC_FLOAT)) ? GL_RGBA32F : GL_RGBA8;
        tdClientLayout(&ps, format, type, w, h, d, d > 1, &c_off, &c_row, &c_img);
        tdUnpack(tdFormatInfo(dst_ifmt), src + c_off, c_row, c_img, format, type, ps.swap_bytes,
                 w, h, d, &st, &wd, &ws);
    }

    if (isRgtcFormat(f))
    {
        int nch = (f->base == GL_RG) ? 2 : 1;
        bool snorm = (f->comp_type == GL_SIGNED_NORMALIZED);
        uint8_t *planes[2] = { NULL, NULL };
        planes[0] = (uint8_t *)malloc(need);
        if (nch == 2) planes[1] = (uint8_t *)malloc(need);
        if (planes[0] && (nch == 1 || planes[1]))
        {
            size_t pr = (size_t)w;
            for (GLsizei zz = 0; zz < (d > 0 ? d : 1); zz++)
                for (GLsizei yy = 0; yy < h; yy++)
                {
                    const uint8_t *r = heap + (size_t)zz * st.image_pitch + (size_t)yy * st.row_pitch;
                    uint8_t *q0 = planes[0] + (size_t)zz * pr * h + (size_t)yy * pr;
                    uint8_t *q1 = planes[1] ? planes[1] + (size_t)zz * pr * h + (size_t)yy * pr : NULL;
                    for (GLsizei xx = 0; xx < w; xx++)
                    {
                        q0[xx] = r[xx * 4];
                        if (q1) q1[xx] = r[xx * 4 + 1];
                    }
                }
            blocks = tdEncodeRGTC(planes[0], planes[1], w, h, d, (size_t)w, (size_t)w * h, nch == 2, snorm);
        }
        free(planes[0]);
        free(planes[1]);
    }
    else if (f->flags & TDF_BPTC_FLOAT)
    {
        blocks = tdEncodeBPTCFloat((const float *)heap, w, h, d, (size_t)w * 16, (size_t)w * h * 16,
                                   (f->flags & TDF_SIGNED) ? 1 : 0);
    }
    else if (isBptcFormat(f))
    {
        blocks = tdEncodeBPTCUnorm(heap, w, h, d, (size_t)w * 4, (size_t)w * h * 4);
    }

    if (blocks)
        ok = storeCompressed(ctx, tex, face, level, x, y, z, w, h, d, blocks);
    free(blocks);
    free(heap);
    return ok;
}


void mglTexImage1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    texImage(ctx, target, 1, level, internalformat, width, 1, 1, border, format, type, pixels);
}

void mglTexImage2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels)
{
    texImage(ctx, target, 2, level, internalformat, width, height, 1, border, format, type, pixels);
}

void mglTexImage3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels)
{
    texImage(ctx, target, 3, level, internalformat, width, height, depth, border, format, type, pixels);
}

/* ------------------------------------------------------------------------ */
/* TexSubImage                                                              */
/* ------------------------------------------------------------------------ */

/* checks a GL-geometry sub region against a level; returns error */
static GLenum checkSubRegion(Texture *tex, GLuint face, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d)
{
    TextureLevel *l = texLevel(tex, face, level);

    if (w < 0 || h < 0 || d < 0)
        return GL_INVALID_VALUE;
    if (!l || !l->complete)
        return GL_INVALID_OPERATION;
    if (x < 0 || y < 0 || z < 0)
        return GL_INVALID_VALUE;
    if ((GLuint)x + (GLuint)w > l->width || (GLuint)y + (GLuint)h > l->height || (GLuint)z + (GLuint)d > l->depth)
        return GL_INVALID_VALUE;
    return GL_NO_ERROR;
}

static void texSubImage(GLMContext ctx, Texture *tex, GLuint face, int dims, GLint level,
                        GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                        GLenum format, GLenum type, const void *pixels)
{
    tex = view_redirect(ctx, tex, &face, &level, &z);
    GLenum err;

    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);

    err = tdValidateFormatType(format, type);
    TD_CHECK(err == GL_NO_ERROR, err);

    err = checkSubRegion(tex, face, level, x, y, z, w, h, d);
    TD_CHECK(err == GL_NO_ERROR, err);

    const TDFormat *f = tdFormatInfo(levelFormat(tex, face, level));
    TD_CHECK(f, GL_INVALID_OPERATION);

    err = tdCheckFormatCompat(f, format, type, TD_PIX_SUBIMAGE);
    TD_CHECK(err == GL_NO_ERROR, err);

    /* uncompressed data into a compressed texture: online compression.  The GL
     * compresses the data on the fly (GL 4.6 8.5.3); MGL has CPU block encoders
     * for the RGTC and BPTC families (every core view class with a compressed
     * member), so a texStorage'd level keeps real hardware blocks.  Families
     * without an encoder (ETC2/EAC) stay rejected. */
    if (f->flags & TDF_COMPRESSED)
    {
        bool online = isRgtcFormat(f) || isBptcFormat(f);
        TextureLevel *lvl = texLevel(tex, face, level);

        TD_CHECK(online, GL_INVALID_OPERATION);
        TD_CHECK(lvl, GL_INVALID_OPERATION);
        /* same block-alignment rules as CompressedTexSubImage */
        TD_CHECK((x % f->block_w) == 0 && (y % f->block_h) == 0, GL_INVALID_OPERATION);
        TD_CHECK((w % f->block_w) == 0 || (GLuint)(x + w) == lvl->width, GL_INVALID_OPERATION);
        TD_CHECK((h % f->block_h) == 0 || (GLuint)(y + h) == lvl->height, GL_INVALID_OPERATION);

        if (w == 0 || h == 0 || d == 0)
            return;

        TDPixelStore ps = pixelStore(ctx, true);
        size_t extent = tdClientLayout(&ps, format, type, w, h, d, dims == 3, NULL, NULL, NULL);
        uint8_t *src = NULL;

        err = resolveClientPtr(ctx, true, pixels, extent, typeAlign(type), &src, NULL);
        TD_CHECK(err == GL_NO_ERROR, err);
        if (!src)
            return;

        if (!uploadOnlineCompressed(ctx, tex, face, level, x, y, z, w, h, d, format, type, src))
            TD_ERR(GL_OUT_OF_MEMORY);
        return;
    }

    if (w == 0 || h == 0 || d == 0)
        return;

    TDPixelStore ps = pixelStore(ctx, true);
    size_t extent = tdClientLayout(&ps, format, type, w, h, d, dims == 3, NULL, NULL, NULL);
    uint8_t *src = NULL;

    err = resolveClientPtr(ctx, true, pixels, extent, typeAlign(type), &src, NULL);
    TD_CHECK(err == GL_NO_ERROR, err);

    if (!src)
        return;

    uploadClient(ctx, tex, face, level, x, y, z, w, h, d, format, type, src, dims == 3);
}

static bool subImageTarget(GLenum target, int dims, GLenum *tt, GLuint *face)
{
    *face = 0;
    *tt = target;
    if (dims == 1)
        return target == GL_TEXTURE_1D;
    if (dims == 2)
    {
        if (isCubeFace(target)) { *tt = GL_TEXTURE_CUBE_MAP; *face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X; return true; }
        return target == GL_TEXTURE_2D || target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_RECTANGLE;
    }
    return target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY;
}

/* DSA target check for Texture*SubImage* */
static bool dsaTargetOk(Texture *tex, int dims)
{
    switch (dims)
    {
        case 1: return tex->target == GL_TEXTURE_1D;
        case 2: return tex->target == GL_TEXTURE_2D || tex->target == GL_TEXTURE_1D_ARRAY || tex->target == GL_TEXTURE_RECTANGLE;
        default: return tex->target == GL_TEXTURE_3D || tex->target == GL_TEXTURE_2D_ARRAY ||
                        tex->target == GL_TEXTURE_CUBE_MAP_ARRAY || tex->target == GL_TEXTURE_CUBE_MAP;
    }
}

void mglTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 1, &tt, &face), GL_INVALID_ENUM);
    texSubImage(ctx, getTex(ctx, 0, tt), face, 1, level, xoffset, 0, 0, width, 1, 1, format, type, pixels);
}

void mglTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 1), GL_INVALID_ENUM);
    texSubImage(ctx, tex, 0, 1, level, xoffset, 0, 0, width, 1, 1, format, type, pixels);
}

void mglTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 2, &tt, &face), GL_INVALID_ENUM);
    texSubImage(ctx, getTex(ctx, 0, tt), face, 2, level, xoffset, yoffset, 0, width, height, 1, format, type, pixels);
}

void mglTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 2), GL_INVALID_ENUM);
    texSubImage(ctx, tex, 0, 2, level, xoffset, yoffset, 0, width, height, 1, format, type, pixels);
}

void mglTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 3, &tt, &face), GL_INVALID_ENUM);
    texSubImage(ctx, getTex(ctx, 0, tt), face, 3, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

void mglTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 3), GL_INVALID_ENUM);

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        /* the faces are consecutive images of the client data (GL 4.6 8.6) */
        GLenum err = tdValidateFormatType(format, type);
        TD_CHECK(err == GL_NO_ERROR, err);
        TD_CHECK(zoffset >= 0 && depth >= 0 && zoffset + depth <= 6, GL_INVALID_VALUE);
        TDPixelStore ps = pixelStore(ctx, true);
        size_t img;
        tdClientLayout(&ps, format, type, width, height, depth, true, NULL, NULL, &img);
        PixelStore saved = STATE(unpack);
        for (GLsizei i = 0; i < depth; i++)
        {
            GLenum before = STATE(error);
            /* the image offset is applied here */
            STATE(unpack).skip_images = 0;
            STATE(unpack).image_height = 0;
            const uint8_t *p = (const uint8_t *)pixels + (size_t)(i + saved.skip_images) * img;
            texSubImage(ctx, tex, zoffset + i, 2, level, xoffset, yoffset, 0, width, height, 1, format, type, p);
            STATE(unpack) = saved;
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    texSubImage(ctx, tex, 0, 3, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

/* ------------------------------------------------------------------------ */
/* TexStorage                                                               */
/* ------------------------------------------------------------------------ */

static bool storageTarget(GLenum target, int dims, GLenum *tt, bool *proxy)
{
    *proxy = false;
    switch (dims)
    {
        case 1:
            if (target == GL_TEXTURE_1D) { *tt = target; return true; }
            if (target == GL_PROXY_TEXTURE_1D) { *tt = GL_TEXTURE_1D; *proxy = true; return true; }
            return false;
        case 2:
            switch (target)
            {
                case GL_TEXTURE_2D: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP:
                    *tt = target; return true;
                case GL_PROXY_TEXTURE_2D: case GL_PROXY_TEXTURE_1D_ARRAY: case GL_PROXY_TEXTURE_RECTANGLE: case GL_PROXY_TEXTURE_CUBE_MAP:
                    *tt = proxyTexTarget(target); *proxy = true; return true;
                default: return false;
            }
        default:
            switch (target)
            {
                case GL_TEXTURE_3D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
                    *tt = target; return true;
                case GL_PROXY_TEXTURE_3D: case GL_PROXY_TEXTURE_2D_ARRAY: case GL_PROXY_TEXTURE_CUBE_MAP_ARRAY:
                    *tt = proxyTexTarget(target); *proxy = true; return true;
                default: return false;
            }
    }
}

static void texStorage(GLMContext ctx, Texture *tex, GLenum tt, GLenum proxy_target, GLsizei levels, GLenum internalformat,
                       GLsizei width, GLsizei height, GLsizei depth)
{
    TD_CHECK(tdFormatInfo(internalformat) != NULL, GL_INVALID_ENUM);
    TD_CHECK(width >= 1 && height >= 1 && depth >= 1 && levels >= 1, GL_INVALID_VALUE);

    const TDFormat *f = tdFormatInfo(internalformat);

    if (tt == GL_TEXTURE_CUBE_MAP)
        TD_CHECK(width == height, GL_INVALID_VALUE);
    if (tt == GL_TEXTURE_CUBE_MAP_ARRAY)
        TD_CHECK(width == height && (depth % 6) == 0, GL_INVALID_VALUE);

    GLenum err = checkSize(ctx, tt, 0, width, height, depth);
    if (err != GL_NO_ERROR)
    {
        if (proxy_target)
        {
            setProxy(ctx, mgl_texparam_proxy_texture(ctx, proxy_target), 0, internalformat, width, height, depth, false);
            return;
        }
        TD_ERR(err);
    }

    GLuint maxl;
    switch (tt)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_1D_ARRAY: maxl = log2u(width) + 1; break;
        case GL_TEXTURE_3D: maxl = log2u(MAX(width, MAX(height, depth))) + 1; break;
        case GL_TEXTURE_RECTANGLE: maxl = 1; break;
        default: maxl = log2u(MAX(width, height)) + 1; break;
    }
    TD_CHECK((GLuint)levels <= maxl && levels <= TD_MAX_LEVELS, GL_INVALID_OPERATION);

    if (f->flags & TDF_COMPRESSED)
        TD_CHECK(compressedAllowedOnTarget(f, tt), GL_INVALID_OPERATION);
    if (f->flags & (TDF_DEPTH | TDF_STENCIL))
        TD_CHECK(depthFormatAllowedOnTarget(tt), GL_INVALID_OPERATION);

    if (proxy_target)
    {
        Texture *p = mgl_texparam_proxy_texture(ctx, proxy_target);
        for (GLsizei l = 0; l < levels && p; l++)
        {
            GLuint w, h, d;
            mipDims(p, l, width, height, depth, &w, &h, &d);
            setProxy(ctx, p, l, internalformat, w, h, d, true);
        }
        return;
    }

    TD_CHECK(tex && tex->name != TEX_OBJ_RES_NAME && tex->name != 0, GL_INVALID_OPERATION);
    TD_CHECK(!tex->immutable_storage, GL_INVALID_OPERATION);

    undefineAllLevels(ctx, tex);

    GLuint faces = mgl_texdata_num_faces(tex);
    for (GLuint face = 0; face < faces; face++)
    {
        for (GLsizei l = 0; l < levels; l++)
        {
            GLuint w, h, d;
            mipDims(tex, l, width, height, depth, &w, &h, &d);
            if (!defineLevel(ctx, tex, face, l, internalformat, w, h, d))
                return;
        }
    }

    tex->immutable_storage = GL_TRUE;
    tex->num_levels = levels;
    tex->mipmap_levels = levels;
    tex->width = width;
    tex->height = height;
    tex->depth = depth;
    tex->internalformat = internalformat;
    tex->is_array = isArrayTarget(tt);
    if (!tex->access)
        tex->access = GL_READ_ONLY;

    /* create the Metal texture now (the content stays undefined) */
    if (TDM(ctx) && TDM(ctx)->realize)
        TDM(ctx)->realize(ctx, tex);
}

void mglTexStorage1D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width)
{
    GLenum tt; bool proxy;
    TD_CHECK(storageTarget(target, 1, &tt, &proxy), GL_INVALID_ENUM);
    texStorage(ctx, proxy ? NULL : getTex(ctx, 0, tt), tt, proxy ? target : 0, levels, internalformat, width, 1, 1);
}

void mglTexStorage2D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    GLenum tt; bool proxy;
    TD_CHECK(storageTarget(target, 2, &tt, &proxy), GL_INVALID_ENUM);
    texStorage(ctx, proxy ? NULL : getTex(ctx, 0, tt), tt, proxy ? target : 0, levels, internalformat, width, height, 1);
}

void mglTexStorage3D(GLMContext ctx, GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    GLenum tt; bool proxy;
    TD_CHECK(storageTarget(target, 3, &tt, &proxy), GL_INVALID_ENUM);
    texStorage(ctx, proxy ? NULL : getTex(ctx, 0, tt), tt, proxy ? target : 0, levels, internalformat, width, height, depth);
}

void mglTextureStorage1D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(tex->target == GL_TEXTURE_1D, GL_INVALID_OPERATION);
    texStorage(ctx, tex, tex->target, 0, levels, internalformat, width, 1, 1);
}

void mglTextureStorage2D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(tex->target == GL_TEXTURE_2D || tex->target == GL_TEXTURE_1D_ARRAY || tex->target == GL_TEXTURE_RECTANGLE ||
             tex->target == GL_TEXTURE_CUBE_MAP, GL_INVALID_OPERATION);
    texStorage(ctx, tex, tex->target, 0, levels, internalformat, width, height, 1);
}

void mglTextureStorage3D(GLMContext ctx, GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(tex->target == GL_TEXTURE_3D || tex->target == GL_TEXTURE_2D_ARRAY || tex->target == GL_TEXTURE_CUBE_MAP_ARRAY,
             GL_INVALID_OPERATION);
    texStorage(ctx, tex, tex->target, 0, levels, internalformat, width, height, depth);
}

/* ------------------------------------------------------------------------ */
/* compressed images                                                        */
/* ------------------------------------------------------------------------ */

static bool etcFallback(const TDFormat *f)
{
    return f && (f->flags & TDF_ETC) && tdStorageFormat(f) != f;
}

/* store tightly packed compressed blocks of a GL region (x,y block aligned) */
static bool storeCompressed(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                            GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d, const uint8_t *blocks)
{
    const TDFormat *f = tdFormatInfo(levelFormat(tex, face, level));
    const TDFormat *sf = tdStorageFormat(f);

    if (!f || !sf)
        return false;

    size_t row = tdStorageRowBytes(f, w), img = tdStorageImageBytes(f, w, h);
    TDRegion r = glRegion(tex, face, level, x, y, z, w, h, d);

    if (etcFallback(f))
    {
        /* keep the original blocks for GetCompressedTexImage, decode on the CPU */
        TDTexInfo *info = mgl_texdata_info(ctx, tex);
        TextureLevel *l = texLevel(tex, face, level);
        size_t lrow = tdStorageRowBytes(f, l->width), limg = tdStorageImageBytes(f, l->width, l->height);
        size_t lsize = limg * l->depth;

        if (!info)
            return false;
        if (!info->orig[face][level])
        {
            info->orig[face][level] = calloc(1, lsize ? lsize : 1);
            if (!info->orig[face][level])
                return false;
            info->orig_size[face][level] = lsize;
        }

        uint8_t *o = (uint8_t *)info->orig[face][level];
        size_t bx = (size_t)x / f->block_w * f->block_bytes, by = (size_t)y / f->block_h;
        size_t brows = ((size_t)h + f->block_h - 1) / f->block_h;
        for (GLsizei zz = 0; zz < d; zz++)
            for (size_t yy = 0; yy < brows; yy++)
                memcpy(o + (size_t)(z + zz) * limg + (by + yy) * lrow + bx, blocks + (size_t)zz * img + yy * row, row);

        TDStorage st;
        if (!allocStorage(sf, w, h, d, &st))
            return false;
        for (GLsizei zz = 0; zz < d; zz++)
            tdDecodeETC(f, blocks + (size_t)zz * img, w, h, (uint8_t *)st.data + (size_t)zz * st.image_pitch, st.row_pitch);
        bool ok = writeRegion(ctx, tex, &r, &st, true, true);
        free(st.data);
        return ok;
    }

    TDStorage st = { (void *)blocks, row, img, NULL, 0, 0 };
    return writeRegion(ctx, tex, &r, &st, true, true);
}

/* gather client compressed data into tight blocks */
static uint8_t *gatherCompressed(const TDFormat *f, const uint8_t *src, size_t off, size_t row_stride, size_t img_stride,
                                 GLsizei w, GLsizei h, GLsizei d)
{
    size_t row = tdStorageRowBytes(f, w), img = tdStorageImageBytes(f, w, h);
    size_t brows = ((size_t)h + f->block_h - 1) / f->block_h;
    uint8_t *t = (uint8_t *)malloc(img * (size_t)d + 1);

    if (!t)
        return NULL;
    for (GLsizei z = 0; z < d; z++)
        for (size_t y = 0; y < brows; y++)
            memcpy(t + (size_t)z * img + y * row, src + off + (size_t)z * img_stride + y * row_stride, row);
    return t;
}

static void compressedTexImage(GLMContext ctx, GLenum target, int dims, GLint level, GLenum internalformat,
                               GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data)
{
    TDTarget t;
    GLenum err;

    TD_CHECK(texImageTarget(target, dims, &t), GL_INVALID_ENUM);

    const TDFormat *f = tdFormatInfo(internalformat);
    TD_CHECK(f && (f->flags & TDF_COMPRESSED), GL_INVALID_ENUM);
    TD_CHECK(compressedAllowedOnTarget(f, t.tex_target), dims == 1 ? GL_INVALID_ENUM : GL_INVALID_OPERATION);
    TD_CHECK(level >= 0 && (GLuint)level <= maxLevelForTarget(ctx, t.tex_target), GL_INVALID_VALUE);
    TD_CHECK(border == 0, GL_INVALID_VALUE);
    TD_CHECK(imageSize >= 0, GL_INVALID_VALUE);

    err = checkSize(ctx, t.tex_target, level, width, height, depth);
    if (err != GL_NO_ERROR)
    {
        if (t.proxy && width >= 0 && height >= 0 && depth >= 0)
        {
            setProxy(ctx, mgl_texparam_proxy_texture(ctx, target), level, internalformat, width, height, depth, false);
            return;
        }
        TD_ERR(err);
    }

    TD_CHECK((size_t)imageSize == tdCompressedSize(f, width, height, depth), GL_INVALID_VALUE);

    if (t.proxy)
    {
        setProxy(ctx, mgl_texparam_proxy_texture(ctx, target), level, internalformat, width, height, depth, true);
        return;
    }

    Texture *tex = getTex(ctx, 0, t.tex_target);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(!tex->immutable_storage, GL_INVALID_OPERATION);

    TDPixelStore ps = pixelStore(ctx, true);
    size_t off, rs, is;
    size_t extent = tdCompressedClientLayout(&ps, f, width, height, depth, dims == 3, NULL, &off, &rs, &is);
    uint8_t *src = NULL;

    if (data || STATE(buffers[_PIXEL_UNPACK_BUFFER]))
    {
        err = resolveClientPtr(ctx, true, data, extent, 1, &src, NULL);
        TD_CHECK(err == GL_NO_ERROR, err);
    }

    if (!defineLevel(ctx, tex, t.face, level, internalformat, width, height, depth))
        return;

    if (!src || !width || !height || !depth)
        return;

    uint8_t *blocks = gatherCompressed(f, src, off, rs, is, width, height, depth);
    TD_CHECK(blocks, GL_OUT_OF_MEMORY);
    bool ok = storeCompressed(ctx, tex, t.face, level, 0, 0, 0, width, height, depth, blocks);
    free(blocks);
    TD_CHECK(ok, GL_OUT_OF_MEMORY);
}

void mglCompressedTexImage3D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data)
{
    compressedTexImage(ctx, target, 3, level, internalformat, width, height, depth, border, imageSize, data);
}

void mglCompressedTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border, GLsizei imageSize, const void *data)
{
    compressedTexImage(ctx, target, 2, level, internalformat, width, height, 1, border, imageSize, data);
}

void mglCompressedTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize, const void *data)
{
    /* no GL 4.6 compressed format supports 1D textures (Table 8.17) */
    compressedTexImage(ctx, target, 1, level, internalformat, width, 1, 1, border, imageSize, data);
}

static bool storeOnlineRGTC(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                            GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                            const TDFormat *f, const uint8_t *blocks);

static void compressedTexSubImage(GLMContext ctx, Texture *tex, GLuint face, int dims, GLint level,
                                  GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                                  GLenum format, GLsizei imageSize, const void *data)
{
    GLenum err;

    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);

    const TDFormat *f = tdFormatInfo(format);
    TD_CHECK(f && (f->flags & TDF_COMPRESSED), GL_INVALID_ENUM);

    TextureLevel *l = texLevel(tex, face, level);
    TD_CHECK(l && l->complete, GL_INVALID_OPERATION);
    /* exception: an online-compressed RGTC level stores the decoded format, so
     * the sub-image's compressed format matches the *texture's* internalformat
     * (storeOnlineRGTC decodes into the level and keeps the original blocks) */
    bool online_rgtc = l->mtl_format != format && isRgtcFormat(f) && f->decoded &&
                       tex->internalformat == format && l->mtl_format == f->decoded;
    TD_CHECK(l->mtl_format == format || online_rgtc, GL_INVALID_OPERATION);
    TD_CHECK(dims != 1, GL_INVALID_OPERATION);

    err = checkSubRegion(tex, face, level, x, y, z, w, h, d);
    TD_CHECK(err == GL_NO_ERROR, err);

    TD_CHECK((x % f->block_w) == 0 && (y % f->block_h) == 0, GL_INVALID_OPERATION);
    TD_CHECK((w % f->block_w) == 0 || (GLuint)(x + w) == l->width, GL_INVALID_OPERATION);
    TD_CHECK((h % f->block_h) == 0 || (GLuint)(y + h) == l->height, GL_INVALID_OPERATION);
    TD_CHECK(imageSize >= 0 && (size_t)imageSize == tdCompressedSize(f, w, h, d), GL_INVALID_VALUE);

    if (!w || !h || !d)
        return;

    TDPixelStore ps = pixelStore(ctx, true);
    size_t off, rs, is;
    size_t extent = tdCompressedClientLayout(&ps, f, w, h, d, dims == 3, NULL, &off, &rs, &is);
    uint8_t *src = NULL;

    err = resolveClientPtr(ctx, true, data, extent, 1, &src, NULL);
    TD_CHECK(err == GL_NO_ERROR, err);
    if (!src)
        return;

    uint8_t *blocks = gatherCompressed(f, src, off, rs, is, w, h, d);
    TD_CHECK(blocks, GL_OUT_OF_MEMORY);
    bool ok = online_rgtc ? storeOnlineRGTC(ctx, tex, face, level, x, y, z, w, h, d, f, blocks)
                          : storeCompressed(ctx, tex, face, level, x, y, z, w, h, d, blocks);
    free(blocks);
    TD_CHECK(ok, GL_OUT_OF_MEMORY);
}

/*
 * Compressed sub-image into an ONLINE-compressed RGTC level (its storage is the
 * decoded R8/RG8 format, see texImage): decode the blocks into the level and keep
 * the original blocks in the texdata info, so a later glGetCompressedTexImage
 * returns exactly the bytes the client uploaded (the level behaves, from the
 * outside, like the hardware-compressed texture its queries report).
 */
static bool storeOnlineRGTC(GLMContext ctx, Texture *tex, GLuint face, GLuint level,
                            GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                            const TDFormat *f, const uint8_t *blocks)
{
    const TDFormat *sf = tdFormatInfo(levelFormat(tex, face, level));
    TDTexInfo *info = mgl_texdata_info(ctx, tex);
    TextureLevel *l = texLevel(tex, face, level);
    size_t lrow, limg, row, img;

    if (!info || !l || !sf || sf->internalformat != f->decoded)
        return false;

    lrow = tdStorageRowBytes(f, l->width);
    limg = tdStorageImageBytes(f, l->width, l->height);
    row = tdStorageRowBytes(f, w);
    img = tdStorageImageBytes(f, w, h);

    /* keep the whole-level original blocks for byte-exact downloads */
    if (!info->orig[face][level])
    {
        size_t lsize = limg * l->depth;
        info->orig[face][level] = calloc(1, lsize ? lsize : 1);
        if (!info->orig[face][level])
            return false;
        info->orig_size[face][level] = lsize;
    }
    uint8_t *o = (uint8_t *)info->orig[face][level];
    size_t bx = (size_t)x / f->block_w * f->block_bytes, by = (size_t)y / f->block_h;
    size_t brows = ((size_t)h + f->block_h - 1) / f->block_h;
    for (GLsizei zz = 0; zz < d; zz++)
        for (size_t yy = 0; yy < brows; yy++)
            memcpy(o + (size_t)(z + zz) * limg + (by + yy) * lrow + bx, blocks + (size_t)zz * img + yy * row, row);

    TDRegion r = glRegion(tex, face, level, x, y, z, w, h, d);
    TDStorage st;
    if (!allocStorage(sf, w, h, d, &st))
        return false;
    for (GLsizei zz = 0; zz < d; zz++)
        tdDecodeRGTC(f, blocks + (size_t)zz * img, w, h, (uint8_t *)st.data + (size_t)zz * st.image_pitch, st.row_pitch);
    bool ok = writeRegion(ctx, tex, &r, &st, true, true);
    free(st.data);
    return ok;
}

void mglCompressedTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 3, &tt, &face), GL_INVALID_ENUM);
    compressedTexSubImage(ctx, getTex(ctx, 0, tt), face, 3, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
}

void mglCompressedTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 2, &tt, &face), GL_INVALID_ENUM);
    compressedTexSubImage(ctx, getTex(ctx, 0, tt), face, 2, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data);
}

void mglCompressedTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 1, &tt, &face), GL_INVALID_ENUM);
    compressedTexSubImage(ctx, getTex(ctx, 0, tt), face, 1, level, xoffset, 0, 0, width, 1, 1, format, imageSize, data);
}

void mglCompressedTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 1), GL_INVALID_ENUM);
    compressedTexSubImage(ctx, tex, 0, 1, level, xoffset, 0, 0, width, 1, 1, format, imageSize, data);
}

void mglCompressedTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 2), GL_INVALID_ENUM);
    compressedTexSubImage(ctx, tex, 0, 2, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data);
}

void mglCompressedTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void *data)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 3), GL_INVALID_ENUM);

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        const TDFormat *f = tdFormatInfo(format);
        TD_CHECK(f && (f->flags & TDF_COMPRESSED), GL_INVALID_ENUM);
        TD_CHECK(zoffset >= 0 && depth >= 0 && zoffset + depth <= 6, GL_INVALID_VALUE);
        TD_CHECK(imageSize >= 0 && (size_t)imageSize == tdCompressedSize(f, width, height, depth), GL_INVALID_VALUE);
        size_t face_size = tdCompressedSize(f, width, height, 1);
        for (GLsizei i = 0; i < depth; i++)
        {
            GLenum before = STATE(error);
            const uint8_t *p = (const uint8_t *)data + (size_t)i * face_size;
            compressedTexSubImage(ctx, tex, zoffset + i, 2, level, xoffset, yoffset, 0, width, height, 1, format, (GLsizei)face_size, p);
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    compressedTexSubImage(ctx, tex, 0, 3, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
}

/* read tight compressed blocks of a GL region */
/*
 * Download of an online-compressed RGTC level: read the decoded storage back and
 * re-encode it with the same deterministic CPU encoder the upload path uses.
 * Only whole-level reads are handled (sub-region reads fall back to the error).
 */
static bool fetchOnlineRGTC(GLMContext ctx, Texture *tex, GLuint face, GLint level,
                            GLsizei w, GLsizei h, GLsizei d, uint8_t *out)
{
    const TDFormat *req = tdFormatInfo(tex->internalformat);
    if (!req || !(req->flags & TDF_COMPRESSED) || !req->decoded || !isRgtcFormat(req))
        return false;

    const TDFormat *sf = tdFormatInfo(levelFormat(tex, face, level));
    if (!sf || sf->internalformat != req->decoded)
        return false;   /* not the decoded equivalent of the requested format */

    TextureLevel *l = texLevel(tex, face, level);
    if (!l || !l->complete)
        return false;

    /* whole level only */
    if (w != (GLsizei)l->width || h != (GLsizei)l->height || d != (GLsizei)l->depth)
        return false;

    /* an explicit glCompressedTexSubImage upload kept its exact bytes */
    TDTexInfo *info = mgl_texdata_find_info(ctx, tex);
    const uint8_t *o = info ? (const uint8_t *)info->orig[face][level] : NULL;
    if (o)
    {
        size_t brows = ((size_t)w + req->block_w - 1) / req->block_w;
        size_t bsize = brows * (((size_t)h + req->block_h - 1) / req->block_h) * (size_t)d *
                       req->block_bytes;
        memcpy(out, o, bsize);
        return true;
    }

    size_t row = (size_t)l->width * sf->bpp, img = row * (size_t)l->height;
    uint8_t *buf = (uint8_t *)malloc(img * (size_t)l->depth);
    if (!buf)
        return false;
    TDRegion r = glRegion(tex, face, level, 0, 0, 0, l->width, l->height, l->depth);
    TDStorage st = { buf, row, img, NULL, 0, 0 };
    if (!readRegion(ctx, tex, &r, &st))
    {
        free(buf);
        return false;
    }

    bool two = (req->base == GL_RG);
    bool snorm = (req->comp_type == GL_SIGNED_NORMALIZED);
    uint8_t *p0 = buf, *p1 = NULL, *plane1 = NULL;
    if (two)
    {
        /* RG8 storage is interleaved; the encoder wants separate planes.
         * one byte per texel per plane => img/2 (= w*h) bytes per image */
        plane1 = (uint8_t *)malloc((img / 2) * (size_t)l->depth);
        if (!plane1) { free(buf); return false; }
        for (GLsizei zz = 0; zz < (GLsizei)l->depth; zz++)
            for (size_t y = 0; y < (size_t)l->height; y++)
                for (size_t x = 0; x < (size_t)l->width; x++)
                {
                    size_t i = zz * img + y * row + 2 * x;
                    plane1[zz * img / 2 + y * (size_t)l->width + x] = buf[i + 1];
                    p0[zz * img / 2 + y * (size_t)l->width + x] = buf[i];
                }
        p1 = plane1;
    }

    uint8_t *blocks = tdEncodeRGTC(p0, p1, l->width, l->height, l->depth,
                                    l->width, (int)(l->width * l->height), two, snorm);
    bool ok = blocks != NULL;
    if (ok)
    {
        size_t brows = ((size_t)w + req->block_w - 1) / req->block_w;
        size_t bsize = brows * (((size_t)h + req->block_h - 1) / req->block_h) * (size_t)d *
                       req->block_bytes;
        memcpy(out, blocks, bsize);
        free(blocks);
    }
    free(plane1);
    free(buf);
    return ok;
}

static bool fetchCompressed(GLMContext ctx, Texture *tex, GLuint face, GLuint level, GLint x, GLint y, GLint z,
                            GLsizei w, GLsizei h, GLsizei d, uint8_t *out)
{
    const TDFormat *f = tdFormatInfo(levelFormat(tex, face, level));
    if (!f)
        return false;

    /* an online-compressed level (uncompressed data uploaded to a compressed
     * internalformat) stores the DECODED format; re-encode it here so the client
     * sees the blocks a hardware-compressed texture would return.  Deterministic
     * encoder => a download->upload->download round trip is byte-exact. */
    if (!(f->flags & TDF_COMPRESSED) && fetchOnlineRGTC(ctx, tex, face, level, w, h, d, out))
        return true;
    size_t row = tdStorageRowBytes(f, w), img = tdStorageImageBytes(f, w, h);

    if (etcFallback(f))
    {
        TDTexInfo *info = mgl_texdata_find_info(ctx, tex);
        TextureLevel *l = texLevel(tex, face, level);
        size_t lrow = tdStorageRowBytes(f, l->width), limg = tdStorageImageBytes(f, l->width, l->height);
        size_t bx = (size_t)x / f->block_w * f->block_bytes, by = (size_t)y / f->block_h;
        size_t brows = ((size_t)h + f->block_h - 1) / f->block_h;
        const uint8_t *o = info ? (const uint8_t *)info->orig[face][level] : NULL;
        for (GLsizei zz = 0; zz < d; zz++)
            for (size_t yy = 0; yy < brows; yy++)
            {
                if (o) memcpy(out + (size_t)zz * img + yy * row, o + (size_t)(z + zz) * limg + (by + yy) * lrow + bx, row);
                else memset(out + (size_t)zz * img + yy * row, 0, row);
            }
        return true;
    }

    TDRegion r = glRegion(tex, face, level, x, y, z, w, h, d);
    TDStorage st = { out, row, img, NULL, 0, 0 };
    return readRegion(ctx, tex, &r, &st);
}

static void getCompressed(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint x, GLint y, GLint z,
                          GLsizei w, GLsizei h, GLsizei d, GLsizei bufSize, void *pixels, bool whole, bool is3d)
{
    GLenum err;

    TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);
    TextureLevel *l = texLevel(tex, face, level);
    TD_CHECK(l && l->complete, GL_INVALID_OPERATION);

    const TDFormat *f = tdFormatInfo(l->mtl_format);
    if (!f || !(f->flags & TDF_COMPRESSED))
    {
        /* exception: an online-compressed RGTC level (decoded storage, see
         * fetchOnlineRGTC) reads back re-encoded blocks */
        const TDFormat *oreq = tdFormatInfo(tex->internalformat);
        bool online_rgtc = oreq && isRgtcFormat(oreq) && oreq->decoded &&
                           f && f->internalformat == oreq->decoded;
        if (!online_rgtc)
            TD_CHECK(f && (f->flags & TDF_COMPRESSED), GL_INVALID_OPERATION);
        f = online_rgtc ? oreq : f;   /* block layout of the requested format */
    }

    if (whole)
    {
        x = y = z = 0;
        w = l->width; h = l->height; d = l->depth;
    }
    else
    {
        err = checkSubRegion(tex, face, level, x, y, z, w, h, d);
        TD_CHECK(err == GL_NO_ERROR, err);
        TD_CHECK((x % f->block_w) == 0 && (y % f->block_h) == 0, GL_INVALID_OPERATION);
        TD_CHECK((w % f->block_w) == 0 || (GLuint)(x + w) == l->width, GL_INVALID_OPERATION);
        TD_CHECK((h % f->block_h) == 0 || (GLuint)(y + h) == l->height, GL_INVALID_OPERATION);
    }

    if (!w || !h || !d)
        return;

    TDPixelStore ps = pixelStore(ctx, false);
    size_t off, rs, is;
    size_t extent = tdCompressedClientLayout(&ps, f, w, h, d, is3d, NULL, &off, &rs, &is);

    if (bufSize >= 0 && !STATE(buffers[_PIXEL_PACK_BUFFER]))
        TD_CHECK(extent <= (size_t)bufSize, GL_INVALID_OPERATION);

    uint8_t *dst = NULL;
    Buffer *pbo = NULL;
    err = resolveClientPtr(ctx, false, pixels, extent, 1, &dst, &pbo);
    TD_CHECK(err == GL_NO_ERROR, err);
    if (!dst)
        return;

    size_t row = tdStorageRowBytes(f, w), img = tdStorageImageBytes(f, w, h);
    size_t brows = ((size_t)h + f->block_h - 1) / f->block_h;
    uint8_t *blocks = (uint8_t *)malloc(img * (size_t)d + 1);
    TD_CHECK(blocks, GL_OUT_OF_MEMORY);

    if (!fetchCompressed(ctx, tex, face, level, x, y, z, w, h, d, blocks))
    {
        free(blocks);
        TD_ERR(GL_INVALID_OPERATION);
    }

    for (GLsizei zz = 0; zz < d; zz++)
        for (size_t yy = 0; yy < brows; yy++)
            memcpy(dst + off + (size_t)zz * is + yy * rs, blocks + (size_t)zz * img + yy * row, row);

    free(blocks);
    pboWrittenRange(ctx, pbo, dst, extent);
}

static bool getImageTarget(GLenum target, GLenum *tt, GLuint *face)
{
    *face = 0;
    *tt = target;
    if (isCubeFace(target)) { *tt = GL_TEXTURE_CUBE_MAP; *face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X; return true; }
    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_RECTANGLE:
            return true;
        default:
            return false;
    }
}

static bool getTargetIs3D(GLenum tt)
{
    return tt == GL_TEXTURE_3D || tt == GL_TEXTURE_2D_ARRAY || tt == GL_TEXTURE_CUBE_MAP_ARRAY || tt == GL_TEXTURE_CUBE_MAP;
}

void mglGetCompressedTexImage(GLMContext ctx, GLenum target, GLint level, void *img)
{
    GLenum tt; GLuint face;
    TD_CHECK(getImageTarget(target, &tt, &face), GL_INVALID_ENUM);
    Texture *tex = getTex(ctx, 0, tt);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    getCompressed(ctx, tex, face, level, 0, 0, 0, 0, 0, 0, -1, img, true, getTargetIs3D(tt));
}

void mglGetnCompressedTexImage(GLMContext ctx, GLenum target, GLint lod, GLsizei bufSize, void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(getImageTarget(target, &tt, &face), GL_INVALID_ENUM);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);
    Texture *tex = getTex(ctx, 0, tt);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    getCompressed(ctx, tex, face, lod, 0, 0, 0, 0, 0, 0, bufSize, pixels, true, getTargetIs3D(tt));
}

void mglGetCompressedTextureImage(GLMContext ctx, GLuint texture, GLint level, GLsizei bufSize, void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        /* all six faces, as consecutive images */
        TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);
        TextureLevel *l = texLevel(tex, 0, level);
        TD_CHECK(l && l->complete, GL_INVALID_OPERATION);
        for (GLuint f = 0; f < 6; f++)
        {
            TextureLevel *lf = texLevel(tex, f, level);
            TD_CHECK(lf && lf->complete && lf->width == l->width && lf->mtl_format == l->mtl_format, GL_INVALID_OPERATION);
        }
        const TDFormat *fmt = tdFormatInfo(l->mtl_format);
        TD_CHECK(fmt && (fmt->flags & TDF_COMPRESSED), GL_INVALID_OPERATION);
        TDPixelStore ps = pixelStore(ctx, false);
        size_t off, rs, is;
        size_t extent = tdCompressedClientLayout(&ps, fmt, l->width, l->height, 6, true, NULL, &off, &rs, &is);
        if (!STATE(buffers[_PIXEL_PACK_BUFFER]))
            TD_CHECK(extent <= (size_t)bufSize, GL_INVALID_OPERATION);
        for (GLuint f = 0; f < 6; f++)
        {
            GLenum before = STATE(error);
            getCompressed(ctx, tex, f, level, 0, 0, 0, 0, 0, 0, -1, (uint8_t *)pixels + (size_t)f * is, true, true);
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    TD_CHECK(tex->target != GL_TEXTURE_BUFFER && tex->target != GL_TEXTURE_2D_MULTISAMPLE &&
             tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_INVALID_OPERATION);
    getCompressed(ctx, tex, 0, level, 0, 0, 0, 0, 0, 0, bufSize, pixels, true, getTargetIs3D(tex->target));
}

void mglGetCompressedTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLsizei bufSize, void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);
    TD_CHECK(tex->target != GL_TEXTURE_BUFFER && tex->target != GL_TEXTURE_2D_MULTISAMPLE &&
             tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_INVALID_OPERATION);

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        TD_CHECK(zoffset >= 0 && depth >= 0 && zoffset + depth <= 6, GL_INVALID_VALUE);
        TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);
        TextureLevel *l = texLevel(tex, zoffset < 6 ? zoffset : 0, level);
        TD_CHECK(l && l->complete, GL_INVALID_OPERATION);
        const TDFormat *fmt = tdFormatInfo(l->mtl_format);
        TD_CHECK(fmt && (fmt->flags & TDF_COMPRESSED), GL_INVALID_OPERATION);
        TDPixelStore ps = pixelStore(ctx, false);
        size_t off, rs, is;
        size_t extent = tdCompressedClientLayout(&ps, fmt, width, height, depth, true, NULL, &off, &rs, &is);
        if (!STATE(buffers[_PIXEL_PACK_BUFFER]))
            TD_CHECK(extent <= (size_t)bufSize, GL_INVALID_OPERATION);
        for (GLsizei i = 0; i < depth; i++)
        {
            GLenum before = STATE(error);
            getCompressed(ctx, tex, zoffset + i, level, xoffset, yoffset, 0, width, height, 1, -1,
                          (uint8_t *)pixels + (size_t)i * is, false, true);
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    getCompressed(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, bufSize, pixels, false,
                  getTargetIs3D(tex->target));
}

/* ------------------------------------------------------------------------ */
/* GetTexImage                                                              */
/* ------------------------------------------------------------------------ */

/* A texture VIEW's images live in the ROOT texture's storage (textures_ext.c
 * keeps the mapping).  GL-level pixel paths (GetTexImage / SubImage / CopyTex*)
 * must follow the view to the root's level/face/layer, or they look at the
 * view's own empty level bookkeeping and fail with INVALID_OPERATION
 * (KHR-GL46.texture_view.coherency: glGetTexImage of a view's LOD). */
static Texture *view_redirect(GLMContext ctx, Texture *tex, GLuint *face, GLint *level, GLint *z)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (!info || info->kind != MGL_TEXEXT_VIEW || !info->view_root || !info->view_root->faces)
        return tex;

    if (level)
        *level += (GLint)info->view_min_level;

    if (info->view_min_layer)
    {
        if (face && info->view_root->target == GL_TEXTURE_CUBE_MAP)
            *face += info->view_min_layer;              /* a cube face view: face index in the root */
        else if (z)
            *z += (GLint)info->view_min_layer;          /* array / 3D layer view */
    }
    return info->view_root;
}

static void getTexImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint x, GLint y, GLint z,
                        GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, GLsizei bufSize, void *pixels,
                        bool whole, bool is3d, size_t dst_skip)
{
    GLenum err;
    Texture *qview = tex;

    tex = view_redirect(ctx, tex, &face, &level, &z);

    TD_CHECK(level >= 0 && (GLuint)level <= maxLevelForTarget(ctx, tex->target), GL_INVALID_VALUE);

    err = tdValidateFormatType(format, type);
    TD_CHECK(err == GL_NO_ERROR, err);

    TextureLevel *l = texLevel(tex, face, level);
    if (!l || !l->complete)
    {
        /* an undefined image reads as nothing; region reads still validate the region */
        TD_CHECK(whole, GL_INVALID_VALUE);
        return;
    }

    const TDFormat *f = tdFormatInfo(levelFormat(tex, face, level));
    TD_CHECK(f, GL_INVALID_OPERATION);

    err = tdCheckGetFormatCompat(f, format, type,
                                 tex && tex->params.depth_stencil_mode ? tex->params.depth_stencil_mode
                                                                       : GL_DEPTH_COMPONENT);
    TD_CHECK(err == GL_NO_ERROR, err);

    if (whole)
    {
        x = y = z = 0;
        w = l->width; h = l->height; d = l->depth;
    }
    else
    {
        err = checkSubRegion(tex, face, level, x, y, z, w, h, d);
        TD_CHECK(err == GL_NO_ERROR, err);
    }

    if (!w || !h || !d)
        return;

    TDPixelStore ps = pixelStore(ctx, false);
    size_t off, rs, is;
    size_t extent = tdClientLayout(&ps, format, type, w, h, d, is3d, &off, &rs, &is);

    if (bufSize >= 0 && !STATE(buffers[_PIXEL_PACK_BUFFER]))
        TD_CHECK(dst_skip + extent <= (size_t)bufSize, GL_INVALID_OPERATION);

    uint8_t *dst = NULL;
    Buffer *pbo = NULL;
    err = resolveClientPtr(ctx, false, (const uint8_t *)pixels + dst_skip, extent, typeAlign(type), &dst, &pbo);
    TD_CHECK(err == GL_NO_ERROR, err);
    if (!dst)
        return;

    const TDFormat *sf = tdStorageFormat(f);
    TDRegion r = glRegion(tex, face, level, x, y, z, w, h, d);

    if (sf->flags & TDF_COMPRESSED)
    {
        /* decompress on the GPU into RGBA32F, then pack.  A view whose own
         * internalformat differs from the root's decodes with the VIEW's
         * decoder (BC6H float <-> ufloat, sRGB <-> linear): pass the view
         * record through; the download builds its format-cast view over the
         * root's storage with root-relative region coordinates. */
        Texture *dtex = tex;
        if (qview != tex)
        {
            const TDFormat *vf = tdFormatInfo(qview->internalformat);
            if (vf && (vf->flags & TDF_COMPRESSED))
                dtex = qview;
        }
        size_t texels = (size_t)w * h * d;
        float *rgba = (float *)malloc(texels * 16 + 16);
        TD_CHECK(rgba, GL_OUT_OF_MEMORY);
        if (!TDM(ctx) || !TDM(ctx)->download_decoded || !TDM(ctx)->download_decoded(ctx, dtex, &r, rgba))
        {
            free(rgba);
            TD_ERR(GL_INVALID_OPERATION);
        }
        TDFormat dec = *tdFormatInfo(GL_RGBA32F);
        dec.base = f->base;
        TDStorage st = { rgba, (size_t)w * 16, (size_t)w * h * 16, NULL, 0, 0 };
        tdPack(&dec, &st, w, h, d, format, type, ps.swap_bytes, dst + off, rs, is);
        free(rgba);
        pboWrittenRange(ctx, pbo, dst, extent);
        return;
    }

    TDStorage st;
    if (!allocStorage(sf, r.w, r.h, r.d, &st))
        TD_ERR(GL_OUT_OF_MEMORY);

    if (!readRegion(ctx, tex, &r, &st))
    {
        free(st.data);
        TD_ERR(GL_INVALID_OPERATION);
    }

    /* the tight storage of a 1D-array region (h layers of one row) is a w x h image */
    tdPack(sf, &st, w, h, d, format, type, ps.swap_bytes, dst + off, rs, is);
    free(st.data);
    pboWrittenRange(ctx, pbo, dst, extent);
}

void mglGetTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(getImageTarget(target, &tt, &face), GL_INVALID_ENUM);
    Texture *tex = getTex(ctx, 0, tt);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    getTexImage(ctx, tex, face, level, 0, 0, 0, 0, 0, 0, format, type, -1, pixels, true, getTargetIs3D(tt), 0);
}

void mglGetnTexImage(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    GLenum tt; GLuint face;
    TD_CHECK(getImageTarget(target, &tt, &face), GL_INVALID_ENUM);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);
    Texture *tex = getTex(ctx, 0, tt);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    getTexImage(ctx, tex, face, level, 0, 0, 0, 0, 0, 0, format, type, bufSize, pixels, true, getTargetIs3D(tt), 0);
}

/* GL cube completeness (spec 9.14): a cube map is cube complete at `level` when all six
 * face images at that level exist with identical internal formats and dimensions.
 * glGetTextureImage / glGetTextureSubImage must raise GL_INVALID_OPERATION for an
 * incomplete cube before any other validation (the per-face read would surface
 * GL_INVALID_VALUE instead — KHR-GL46.direct_state_access.textures_image_query_errors). */
static bool cubeCompleteAtLevel(Texture *tex, GLint level)
{
    TextureLevel *first;

    if (level < 0 || level >= TD_MAX_LEVELS)
        return false;
    first = texLevel(tex, 0, level);
    if (!first || !first->complete)
        return false;
    for (GLuint f = 1; f < 6; f++)
    {
        TextureLevel *l = texLevel(tex, f, level);
        if (!l || !l->complete || l->width != first->width || l->height != first->height ||
            l->mtl_format != first->mtl_format)
            return false;
    }
    return true;
}

void mglGetTextureImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);
    TD_CHECK(tex->target != GL_TEXTURE_BUFFER && tex->target != GL_TEXTURE_2D_MULTISAMPLE &&
             tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_INVALID_OPERATION);
    if (tex->target == GL_TEXTURE_CUBE_MAP)
        TD_CHECK(cubeCompleteAtLevel(tex, level), GL_INVALID_OPERATION);
    if (tex->target == GL_TEXTURE_CUBE_MAP_ARRAY)
    {
        TextureLevel *l = texLevel(tex, 0, level);
        TD_CHECK(l && l->complete, GL_INVALID_OPERATION);   /* cube array complete */
    }

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);
        TextureLevel *l = texLevel(tex, 0, level);
        if (!l || !l->complete)
            return;
        GLenum err = tdValidateFormatType(format, type);
        TD_CHECK(err == GL_NO_ERROR, err);
        TDPixelStore ps = pixelStore(ctx, false);
        size_t off, rs, is;
        size_t extent = tdClientLayout(&ps, format, type, l->width, l->height, 6, true, &off, &rs, &is);
        if (!STATE(buffers[_PIXEL_PACK_BUFFER]))
            TD_CHECK(extent <= (size_t)bufSize, GL_INVALID_OPERATION);
        for (GLuint f = 0; f < 6; f++)
        {
            GLenum before = STATE(error);
            getTexImage(ctx, tex, f, level, 0, 0, 0, l->width, l->height, 1, format, type, -1, pixels, false, true, (size_t)f * is);
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    getTexImage(ctx, tex, 0, level, 0, 0, 0, 0, 0, 0, format, type, bufSize, pixels, true, getTargetIs3D(tex->target), 0);
}

void mglGetTextureSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(bufSize >= 0, GL_INVALID_VALUE);
    TD_CHECK(tex->target != GL_TEXTURE_BUFFER && tex->target != GL_TEXTURE_2D_MULTISAMPLE &&
             tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_INVALID_OPERATION);
    if (tex->target == GL_TEXTURE_CUBE_MAP)
        TD_CHECK(cubeCompleteAtLevel(tex, level), GL_INVALID_OPERATION);
    if (tex->target == GL_TEXTURE_CUBE_MAP_ARRAY)
    {
        TextureLevel *l = texLevel(tex, 0, level);
        TD_CHECK(l && l->complete, GL_INVALID_OPERATION);   /* cube array complete */
    }

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        TD_CHECK(zoffset >= 0 && depth >= 0 && zoffset + depth <= 6, GL_INVALID_VALUE);
        GLenum err = tdValidateFormatType(format, type);
        TD_CHECK(err == GL_NO_ERROR, err);
        TDPixelStore ps = pixelStore(ctx, false);
        size_t off, rs, is;
        size_t extent = tdClientLayout(&ps, format, type, width, height, depth, true, &off, &rs, &is);
        if (!STATE(buffers[_PIXEL_PACK_BUFFER]))
            TD_CHECK(extent <= (size_t)bufSize, GL_INVALID_OPERATION);
        for (GLsizei i = 0; i < depth; i++)
        {
            GLenum before = STATE(error);
            getTexImage(ctx, tex, zoffset + i, level, xoffset, yoffset, 0, width, height, 1, format, type, -1, pixels,
                        false, true, (size_t)i * is);
            if (before == GL_NO_ERROR && STATE(error) != GL_NO_ERROR)
                return;
        }
        return;
    }

    getTexImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, width, height, depth, format, type, bufSize, pixels,
                false, getTargetIs3D(tex->target), 0);
}

/* ------------------------------------------------------------------------ */
/* CopyTexImage / CopyTexSubImage                                           */
/* ------------------------------------------------------------------------ */

static GLenum readFramebufferCheck(GLMContext ctx)
{
    if (STATE(readbuffer) && mgl_fb_status(ctx, STATE(readbuffer)) != GL_FRAMEBUFFER_COMPLETE)
        return GL_INVALID_FRAMEBUFFER_OPERATION;
    return GL_NO_ERROR;
}

/* class compatibility between the read buffer and a destination format */
static GLenum copyCompat(GLMContext ctx, const TDFormat *df, bool *depth, bool *stencil)
{
    GLenum src_ifmt = 0;
    GLuint sw = 0, sh = 0;

    *depth = (df->flags & TDF_DEPTH) != 0;
    *stencil = (df->flags & TDF_STENCIL) != 0;

    if (df->flags & TDF_COMPRESSED)
        return GL_INVALID_OPERATION;

    if (!TDM(ctx) || !TDM(ctx)->read_buffer_info ||
        !TDM(ctx)->read_buffer_info(ctx, *depth, *stencil, &src_ifmt, &sw, &sh))
        return GL_INVALID_OPERATION;

    /* the requested aspect must really be part of the READ framebuffer: a
     * depth/stencil texture attached to GL_DEPTH_ATTACHMENT alone has no stencil
     * plane as far as the FBO is concerned (GL 4.6 19.3.2: copying DEPTH_STENCIL
     * needs depth AND stencil; KHR-GL46 packed_depth_stencil validate_errors) */
    Framebuffer *rb_fbo = ctx->state.readbuffer;
    if (*stencil && !rb_fbo->stencil.buf.tex)
        return GL_INVALID_OPERATION;
    if (*depth && !rb_fbo->depth.buf.tex)
        return GL_INVALID_OPERATION;

    const TDFormat *sf = tdFormatInfo(src_ifmt);
    if (!sf)
        return GL_NO_ERROR;   /* default framebuffer formats we do not describe: normalized color */

    if (((sf->flags & TDF_INTEGER) != 0) != ((df->flags & TDF_INTEGER) != 0))
        return GL_INVALID_OPERATION;
    if ((df->flags & TDF_INTEGER) && sf->comp_type != df->comp_type)
        return GL_INVALID_OPERATION;
    return GL_NO_ERROR;
}

static void copyTexSubImage(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLint xoffset, GLint yoffset, GLint zoffset,
                            GLint x, GLint y, GLsizei width, GLsizei height)
{
    tex = view_redirect(ctx, tex, &face, &level, &zoffset);
    GLenum err;

    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);

    err = checkSubRegion(tex, face, level, xoffset, yoffset, zoffset, width, height, 1);
    TD_CHECK(err == GL_NO_ERROR, err);

    err = readFramebufferCheck(ctx);
    TD_CHECK(err == GL_NO_ERROR, err);

    const TDFormat *df = tdFormatInfo(levelFormat(tex, face, level));
    TD_CHECK(df, GL_INVALID_OPERATION);

    bool depth, stencil;
    err = copyCompat(ctx, df, &depth, &stencil);
    TD_CHECK(err == GL_NO_ERROR, err);

    if (!width || !height)
        return;

    TDRegion r = glRegion(tex, face, level, xoffset, yoffset, zoffset, width, height, 1);
    TD_CHECK(TDM(ctx) && TDM(ctx)->copy_from_read_fb, GL_INVALID_OPERATION);
    err = TDM(ctx)->copy_from_read_fb(ctx, tex, &r, x, y, width, height, depth, stencil);
    TD_CHECK(err == GL_NO_ERROR, err);
}

static void copyTexImage(GLMContext ctx, GLenum target, int dims, GLint level, GLenum internalformat,
                         GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    TDTarget t;
    GLenum err;

    TD_CHECK(texImageTarget(target, dims, &t) && !t.proxy && t.tex_target != GL_TEXTURE_RECTANGLE + 0x10000, GL_INVALID_ENUM);
    TD_CHECK(level >= 0 && (GLuint)level <= maxLevelForTarget(ctx, t.tex_target), GL_INVALID_VALUE);
    TD_CHECK(border == 0, GL_INVALID_VALUE);

    GLenum ifmt = tdResolveInternalFormat(internalformat, 0, 0);
    TD_CHECK(ifmt != 0, GL_INVALID_VALUE);
    const TDFormat *f = tdFormatInfo(ifmt);
    TD_CHECK(!(f->flags & TDF_COMPRESSED), GL_INVALID_OPERATION);

    err = checkSize(ctx, t.tex_target, level, width, height, 1);
    TD_CHECK(err == GL_NO_ERROR, err);

    err = readFramebufferCheck(ctx);
    TD_CHECK(err == GL_NO_ERROR, err);

    bool depth, stencil;
    err = copyCompat(ctx, f, &depth, &stencil);
    TD_CHECK(err == GL_NO_ERROR, err);

    Texture *tex = getTex(ctx, 0, t.tex_target);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(!tex->immutable_storage, GL_INVALID_OPERATION);

    if (!defineLevel(ctx, tex, t.face, level, ifmt, width, height, 1))
        return;

    if (!width || !height)
        return;

    copyTexSubImage(ctx, tex, t.face, level, 0, 0, 0, x, y, width, height);
}

void mglCopyTexImage1D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border)
{
    copyTexImage(ctx, target, 1, level, internalformat, x, y, width, 1, border);
}

void mglCopyTexImage2D(GLMContext ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height, GLint border)
{
    copyTexImage(ctx, target, 2, level, internalformat, x, y, width, height, border);
}

void mglCopyTexSubImage1D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 1, &tt, &face), GL_INVALID_ENUM);
    copyTexSubImage(ctx, getTex(ctx, 0, tt), face, level, xoffset, 0, 0, x, y, width, 1);
}

void mglCopyTexSubImage2D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 2, &tt, &face), GL_INVALID_ENUM);
    copyTexSubImage(ctx, getTex(ctx, 0, tt), face, level, xoffset, yoffset, 0, x, y, width, height);
}

void mglCopyTexSubImage3D(GLMContext ctx, GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    GLenum tt; GLuint face;
    TD_CHECK(subImageTarget(target, 3, &tt, &face), GL_INVALID_ENUM);
    copyTexSubImage(ctx, getTex(ctx, 0, tt), face, level, xoffset, yoffset, zoffset, x, y, width, height);
}

void mglCopyTextureSubImage1D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 1), GL_INVALID_OPERATION);
    copyTexSubImage(ctx, tex, 0, level, xoffset, 0, 0, x, y, width, 1);
}

void mglCopyTextureSubImage2D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 2), GL_INVALID_OPERATION);
    copyTexSubImage(ctx, tex, 0, level, xoffset, yoffset, 0, x, y, width, height);
}

void mglCopyTextureSubImage3D(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y, GLsizei width, GLsizei height)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(dsaTargetOk(tex, 3), GL_INVALID_OPERATION);

    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        TD_CHECK(zoffset >= 0 && zoffset < 6, GL_INVALID_VALUE);
        copyTexSubImage(ctx, tex, zoffset, level, xoffset, yoffset, 0, x, y, width, height);
        return;
    }

    copyTexSubImage(ctx, tex, 0, level, xoffset, yoffset, zoffset, x, y, width, height);
}

/* ------------------------------------------------------------------------ */
/* ClearTexImage / ClearTexSubImage                                         */
/* ------------------------------------------------------------------------ */

static void clearTex(GLMContext ctx, GLuint texture, GLint level, GLint x, GLint y, GLint z,
                     GLsizei w, GLsizei h, GLsizei d, GLenum format, GLenum type, const void *data, bool whole)
{
    GLenum err;
    Texture *tex = findTexture(ctx, texture);

    TD_CHECK(tex, GL_INVALID_OPERATION);
    TD_CHECK(tex->target != GL_TEXTURE_BUFFER, GL_INVALID_OPERATION);
    TD_CHECK(level >= 0 && level < TD_MAX_LEVELS, GL_INVALID_VALUE);

    err = tdValidateFormatType(format, type);
    TD_CHECK(err == GL_NO_ERROR, err);

    GLuint face0 = 0, nfaces = 1;
    if (tex->target == GL_TEXTURE_CUBE_MAP)
    {
        if (whole) { nfaces = 6; }
        else
        {
            TD_CHECK(z >= 0 && d >= 0 && z + d <= 6, GL_INVALID_OPERATION);
            face0 = z; nfaces = d; z = 0; d = 1;
        }
    }

    TextureLevel *l = texLevel(tex, face0 < 6 ? face0 : 0, level);
    TD_CHECK(l && l->complete, GL_INVALID_OPERATION);

    const TDFormat *f = tdFormatInfo(levelFormat(tex, face0 < 6 ? face0 : 0, level));
    TD_CHECK(f && !(f->flags & TDF_COMPRESSED), GL_INVALID_OPERATION);

    err = tdCheckFormatCompat(f, format, type, TD_PIX_SUBIMAGE);
    TD_CHECK(err == GL_NO_ERROR, err);

    for (GLuint fi = face0; fi < face0 + nfaces; fi++)
    {
        TextureLevel *lf = texLevel(tex, fi, level);
        TD_CHECK(lf && lf->complete, GL_INVALID_OPERATION);
        GLint cx = x, cy = y, cz = z;
        GLsizei cw = w, ch = h, cd = d;
        if (whole)
        {
            cx = cy = cz = 0;
            cw = lf->width; ch = lf->height; cd = lf->depth;
        }
        TD_CHECK(cw >= 0 && ch >= 0 && cd >= 0, GL_INVALID_OPERATION);
        TD_CHECK(cx >= 0 && cy >= 0 && cz >= 0 && (GLuint)(cx + cw) <= lf->width && (GLuint)(cy + ch) <= lf->height &&
                 (GLuint)(cz + cd) <= lf->depth, GL_INVALID_OPERATION);
        if (!cw || !ch || !cd)
            continue;

        const TDFormat *sf = tdStorageFormat(f);
        uint8_t texel[32] = { 0 };
        uint8_t stexel[4] = { 0 };
        TDStorage one = { texel, sizeof(texel), sizeof(texel), stexel, 1, 1 };
        bool wd = true, ws = true;

        if (data)
            tdUnpack(sf, data, 0, 0, format, type, false, 1, 1, 1, &one, &wd, &ws);

        TDRegion r = glRegion(tex, fi, level, cx, cy, cz, cw, ch, cd);
        TDStorage st;
        if (!allocStorage(sf, r.w, r.h, r.d, &st))
            TD_ERR(GL_OUT_OF_MEMORY);

        size_t texels = (size_t)r.w * r.h * r.d;
        uint8_t *p = (uint8_t *)st.data;
        for (size_t i = 0; i < texels; i++)
            memcpy(p + i * sf->bpp, texel, sf->bpp);
        if (st.stencil)
            memset(st.stencil, stexel[0], texels);

        bool ok = writeRegion(ctx, tex, &r, &st, wd, ws);
        free(st.data);
        TD_CHECK(ok, GL_OUT_OF_MEMORY);
    }
}

void mglClearTexImage(GLMContext ctx, GLuint texture, GLint level, GLenum format, GLenum type, const void *data)
{
    clearTex(ctx, texture, level, 0, 0, 0, 0, 0, 0, format, type, data, true);
}

void mglClearTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const void *data)
{
    clearTex(ctx, texture, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data, false);
}

/* ------------------------------------------------------------------------ */
/* CopyImageSubData                                                         */
/* ------------------------------------------------------------------------ */

static GLenum imageObject(GLMContext ctx, GLuint name, GLenum target, Texture **out)
{
    *out = NULL;

    if (target == GL_RENDERBUFFER)
    {
        Renderbuffer *rbo = name ? (Renderbuffer *)searchHashTable(&STATE(renderbuffer_table), name) : NULL;
        if (!rbo || !rbo->tex)
            return GL_INVALID_VALUE;
        *out = rbo->tex;
        return GL_NO_ERROR;
    }

    if (!validObjectTarget(target) || target == GL_TEXTURE_BUFFER)
        return GL_INVALID_ENUM;

    /* negative cube faces and other non-first-class targets are not valid
     * srcTarget/dstTarget values (KHR-GL46 copy_image invalid_target) */
    if (target >= GL_TEXTURE_CUBE_MAP_NEGATIVE_X && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
        return GL_INVALID_ENUM;

    Texture *tex = findTexture(ctx, name);
    if (!tex)
        return GL_INVALID_VALUE;
    if (tex->target != target)
        return GL_INVALID_ENUM;

    *out = tex;
    return GL_NO_ERROR;
}

static GLuint texelBits(const TDFormat *f)
{
    if (f->flags & TDF_COMPRESSED)
        return (GLuint)f->block_bytes * 8;
    if (f->internalformat == GL_RGB9_E5)
        return 32;
    return (GLuint)f->bits[0] + f->bits[1] + f->bits[2] + f->bits[3] + f->bits[4] + f->bits[5];
}

static bool isMultisampleTarget(GLenum target)
{
    return target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

/* sampling-completeness (GL 8.17) as far as CopyImageSubData cares: base..max levels
 * exist when the min filter needs mipmaps (KHR-GL46 copy_image incomplete_tex) */
static bool copyTexComplete(Texture *tex, GLuint face)
{
    if (tex->immutable_storage || tex->target == GL_RENDERBUFFER ||
        tex->target == GL_TEXTURE_2D_MULTISAMPLE || tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        return true;

    GLint base = tex->params.base_level;
    GLint max = tex->params.max_level;
    if (max < base || base < 0 || (GLuint)base >= tex->mipmap_levels)
        return false;

    bool mips = !(tex->params.min_filter == GL_NEAREST || tex->params.min_filter == GL_LINEAR);
    GLint top = mips ? max : base;
    if ((GLuint)top >= tex->mipmap_levels)
        return false;

    for (GLint l = base; l <= top; l++)
    {
        TextureLevel *lv = texLevel(tex, face, l);
        if (!lv || !lv->complete)
            return false;
    }
    return true;
}

void mglCopyImageSubData(GLMContext ctx, GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei srcWidth, GLsizei srcHeight, GLsizei srcDepth)
{
    Texture *st, *dt;
    GLenum err;

    err = imageObject(ctx, srcName, srcTarget, &st);
    TD_CHECK(err == GL_NO_ERROR, err);
    err = imageObject(ctx, dstName, dstTarget, &dt);
    TD_CHECK(err == GL_NO_ERROR, err);

    TD_CHECK(srcWidth >= 0 && srcHeight >= 0 && srcDepth >= 0, GL_INVALID_VALUE);
    TD_CHECK(srcLevel >= 0 && srcLevel < TD_MAX_LEVELS && dstLevel >= 0 && dstLevel < TD_MAX_LEVELS, GL_INVALID_VALUE);
    TD_CHECK(srcX >= 0 && srcY >= 0 && srcZ >= 0 && dstX >= 0 && dstY >= 0 && dstZ >= 0, GL_INVALID_VALUE);

    /* cube maps: z selects the face */
    GLuint sface0 = 0, dface0 = 0;
    bool scube = st->target == GL_TEXTURE_CUBE_MAP, dcube = dt->target == GL_TEXTURE_CUBE_MAP;
    if (scube) { TD_CHECK(srcZ >= 0 && srcZ + srcDepth <= 6, GL_INVALID_VALUE); sface0 = srcZ; }
    if (dcube) { TD_CHECK(dstZ >= 0 && dstZ + srcDepth <= 6, GL_INVALID_VALUE); dface0 = dstZ; }

    /* GL 4.6 18.3.3: two TEXTURE_BUFFER textures may be copied into each other
     * (srcWidth bytes); mixing a buffer target with anything else is INVALID_ENUM */
    if (srcTarget == GL_TEXTURE_BUFFER || dstTarget == GL_TEXTURE_BUFFER)
    {
        if (srcTarget != GL_TEXTURE_BUFFER || dstTarget != GL_TEXTURE_BUFFER)
            TD_CHECK(0, GL_INVALID_ENUM);

        MGLTexExtInfo *si = mgl_texext_info(ctx, st), *di = mgl_texext_info(ctx, dt);
        TD_CHECK(si && di && si->kind == MGL_TEXEXT_BUFFER && di->kind == MGL_TEXEXT_BUFFER,
                 GL_INVALID_OPERATION);

        Buffer *sb = si->buffer_name ? (Buffer *)searchHashTable(&STATE(buffer_table), si->buffer_name) : NULL;
        Buffer *db = di->buffer_name ? (Buffer *)searchHashTable(&STATE(buffer_table), di->buffer_name) : NULL;
        TD_CHECK(sb && db, GL_INVALID_OPERATION);

        /* byte counts: the requested range, or the whole buffer */
        GLsizeiptr sbytes = si->buffer_ranged ? si->buffer_size : (GLsizeiptr)sb->data.buffer_size;
        GLsizeiptr dbytes = di->buffer_ranged ? di->buffer_size : (GLsizeiptr)db->data.buffer_size;
        GLsizeiptr n = sbytes < dbytes ? sbytes : dbytes;
        if (srcWidth >= 0 && (GLsizeiptr)srcWidth < n) n = srcWidth;
        if (TDM(ctx) && TDM(ctx)->buffer_contents && n > 0)
        {
            void *sd = TDM(ctx)->buffer_contents(ctx, sb);
            void *dd = TDM(ctx)->buffer_contents(ctx, db);
            if (sd && dd)
                memcpy((uint8_t *)dd + di->buffer_offset, (uint8_t *)sd + si->buffer_offset, n);
        }
        return;
    }

    TextureLevel *sl = texLevel(st, sface0 < 6 ? sface0 : 0, srcLevel);
    TextureLevel *dl = texLevel(dt, dface0 < 6 ? dface0 : 0, dstLevel);
    /* an undefined level is INVALID_VALUE; texture incompleteness (8.17 base/max
     * level + filter) is INVALID_OPERATION */
    TD_CHECK(sl && sl->complete && dl && dl->complete, GL_INVALID_VALUE);
    TD_CHECK(copyTexComplete(st, sface0 < 6 ? sface0 : 0) && copyTexComplete(dt, dface0 < 6 ? dface0 : 0),
             GL_INVALID_OPERATION);

    /* multisample: both sides must be multisample, with equal sample counts */
    if (isMultisampleTarget(st->target) != isMultisampleTarget(dt->target))
        TD_CHECK(0, GL_INVALID_OPERATION);
    if (isMultisampleTarget(st->target) && isMultisampleTarget(dt->target))
    {
        GLuint s_samples = (GLuint)mgl_texext_samples(ctx, st);
        GLuint d_samples = (GLuint)mgl_texext_samples(ctx, dt);
        if (s_samples && d_samples)
            TD_CHECK(s_samples == d_samples, GL_INVALID_OPERATION);
    }

    /* GL offset semantics for arrays: for TEXTURE_1D_ARRAY srcY is the array layer
     * (MGL keeps the layer count in the level height — Metal's 1DArray layout — so
     * the bounds check below covers the layer axis directly) */

    const TDFormat *sf = tdFormatInfo(levelFormat(st, sface0 < 6 ? sface0 : 0, srcLevel));
    const TDFormat *df = tdFormatInfo(levelFormat(dt, dface0 < 6 ? dface0 : 0, dstLevel));
    TD_CHECK(sf && df, GL_INVALID_OPERATION);

    /* a level created through online compression keeps the compressed request in
     * tex->internalformat while levelFormat() reports the decoded storage —
     * compatibility, storage-size and alignment checks validate against the request
     * (KHR-GL46 copy_image invalid_alignment / incompatible_formats_compression);
     * the copy machinery below keeps using the real (decoded) level formats */
    const TDFormat *vsf = sf, *vdf = df;
    {
        const TDFormat *sreq = tdFormatInfo(st->internalformat);
        const TDFormat *dreq = tdFormatInfo(dt->internalformat);
        if (!(sf->flags & TDF_COMPRESSED) && sreq && (sreq->flags & TDF_COMPRESSED)) vsf = sreq;
        if (!(df->flags & TDF_COMPRESSED) && dreq && (dreq->flags & TDF_COMPRESSED)) vdf = dreq;
    }

    bool vsc = (vsf->flags & TDF_COMPRESSED) != 0, vdc = (vdf->flags & TDF_COMPRESSED) != 0;

    /* the copy machinery operates on the REAL level formats (decoded storage for
     * online-compressed textures), not on the request formats above */
    bool sc = (sf->flags & TDF_COMPRESSED) != 0, dc = (df->flags & TDF_COMPRESSED) != 0;
    bool decoded_mixed = (vsc != vdc) && !sc && !dc;

    /* compatibility (GL 4.6 18.3.2): identical format, two compressed formats with
     * equal block geometry, or a compressed block of N bytes against an
     * uncompressed texel of N bytes */
    if ((vsf->flags | vdf->flags) & (TDF_DEPTH | TDF_STENCIL))
        TD_CHECK(vsf->internalformat == vdf->internalformat, GL_INVALID_OPERATION);
    else if (vsc && vdc)
        TD_CHECK(vsf->block_bytes == vdf->block_bytes && vsf->block_w == vdf->block_w && vsf->block_h == vdf->block_h, GL_INVALID_OPERATION);
    else if (vsc != vdc)
    {
        const TDFormat *cf = vsc ? vsf : vdf;
        const TDFormat *uf = vsc ? vdf : vsf;
        TD_CHECK(cf->block_bytes == uf->bpp, GL_INVALID_OPERATION);
    }
    else
        TD_CHECK(texelBits(vsf) == texelBits(vdf), GL_INVALID_OPERATION);

    /* raw texel bytes must also match in the Metal storage */
    const TDFormat *ss = tdStorageFormat(vsf), *ds = tdStorageFormat(vdf);
    if (!vsc && !vdc)
        TD_CHECK(ss->bpp == ds->bpp, GL_INVALID_OPERATION);
    else if (vsc && !vdc && !etcFallback(vsf))
        TD_CHECK(ds->bpp == vsf->block_bytes, GL_INVALID_OPERATION);
    else if (!vsc && vdc && !etcFallback(vdf))
        TD_CHECK(ss->bpp == vdf->block_bytes, GL_INVALID_OPERATION);

    /* destination extent in destination texels */
    GLsizei dw = srcWidth, dh = srcHeight;
    if (vsc && !vdc) { dw = (srcWidth + vsf->block_w - 1) / vsf->block_w; dh = (srcHeight + vsf->block_h - 1) / vsf->block_h; }
    if (!vsc && vdc) { dw = srcWidth * vdf->block_w; dh = srcHeight * vdf->block_h; }

    /* regions in range and block aligned.  GL_TEXTURE_1D_ARRAY: Y addresses the
     * single row of a 1-high image and Z selects the layer (GL 4.6 CopyImageSubData,
     * same as Mesa prepare_target_err); MGL keeps the layer count in the level
     * height, so for 1D arrays the y/z and height/depth axes swap for this check */
    if (srcTarget == GL_TEXTURE_1D_ARRAY)
    {
        TD_CHECK(srcY + srcHeight <= 1, GL_INVALID_VALUE);   /* single-row image */
        err = checkSubRegion(st, sface0, srcLevel, srcX, srcZ, 0, srcWidth, srcDepth, srcHeight);
        TD_CHECK(err == GL_NO_ERROR, GL_INVALID_VALUE);
    }
    else
    {
        err = checkSubRegion(st, sface0, srcLevel, srcX, srcY, scube ? 0 : srcZ, srcWidth, srcHeight, scube ? 1 : srcDepth);
        TD_CHECK(err == GL_NO_ERROR, GL_INVALID_VALUE);
    }
    if (dstTarget == GL_TEXTURE_1D_ARRAY)
    {
        TD_CHECK(dstY + dh <= 1, GL_INVALID_VALUE);          /* single-row image */
        err = checkSubRegion(dt, dface0, dstLevel, dstX, dstZ, 0, dw, srcDepth, dh);
        TD_CHECK(err == GL_NO_ERROR, GL_INVALID_VALUE);
    }
    else
    {
        err = checkSubRegion(dt, dface0, dstLevel, dstX, dstY, dcube ? 0 : dstZ, dw, dh, dcube ? 1 : srcDepth);
        TD_CHECK(err == GL_NO_ERROR, GL_INVALID_VALUE);
    }

    if (vsc)
    {
        TD_CHECK(srcX % vsf->block_w == 0 && srcY % vsf->block_h == 0, GL_INVALID_VALUE);
        TD_CHECK(srcWidth % vsf->block_w == 0 || (GLuint)(srcX + srcWidth) == sl->width, GL_INVALID_VALUE);
        TD_CHECK(srcHeight % vsf->block_h == 0 || (GLuint)(srcY + srcHeight) == sl->height, GL_INVALID_VALUE);
    }
    if (vdc)
    {
        TD_CHECK(dstX % vdf->block_w == 0 && dstY % vdf->block_h == 0, GL_INVALID_VALUE);
        TD_CHECK(dw % vdf->block_w == 0 || (GLuint)(dstX + dw) == dl->width, GL_INVALID_VALUE);
        TD_CHECK(dh % vdf->block_h == 0 || (GLuint)(dstY + dh) == dl->height, GL_INVALID_VALUE);
    }

    if (!srcWidth || !srcHeight || !srcDepth)
        return;

    GLsizei layers = srcDepth;
    bool per_face = scube || dcube;
    bool cpu = etcFallback(sf) || etcFallback(df) || !TDM(ctx) || !TDM(ctx)->copy_image;

    for (GLsizei i = 0; i < (per_face ? layers : 1); i++)
    {
        GLuint sface = scube ? sface0 + i : 0, dface = dcube ? dface0 + i : 0;
        GLint szz = scube ? 0 : (per_face ? srcZ + i : srcZ);
        GLint dzz = dcube ? 0 : (per_face ? dstZ + i : dstZ);
        GLsizei n = per_face ? 1 : layers;

        /* GL_TEXTURE_1D_ARRAY: the GL API puts the layer in Z and keeps the
         * 1-texel-high image in Y; MGL's level stores the layers in height, so
         * the copy region is (x, z, 0, w, layers, 1) — see the bounds checks */
        GLint sry = srcY, srz = szz, dr_y = dstY, drz = dzz;
        GLsizei srh = srcHeight, srd = n, drh = dh, drd = n;
        if (srcTarget == GL_TEXTURE_1D_ARRAY) { sry = srcZ; srz = 0; srh = n; srd = srcHeight; }
        if (dstTarget == GL_TEXTURE_1D_ARRAY) { dr_y = dstZ; drz = 0; drh = n; drd = dh; }
        TDRegion sr = glRegion(st, sface, srcLevel, srcX, sry, srz, srcWidth, srh, srd);
        TDRegion dr = glRegion(dt, dface, dstLevel, dstX, dr_y, drz, dw, drh, drd);

        if (!cpu && TDM(ctx)->copy_image(ctx, st, &sr, dt, &dr))
            continue;

        /* CPU path (ETC decoded on the CPU, or a level outside the realized shape): raw bytes */
        uint8_t *raw;
        if (sc)
        {
            size_t bytes = tdCompressedSize(sf, srcWidth, srcHeight, n);
            raw = (uint8_t *)malloc(bytes + 1);
            TD_CHECK(raw, GL_OUT_OF_MEMORY);
            fetchCompressed(ctx, st, sface, srcLevel, srcX, sry, srz, srcWidth, srh, srd, raw);
        }
        else
        {
            TDStorage tmp;
            raw = (uint8_t *)allocStorage(ss, sr.w, sr.h, sr.d, &tmp);
            TD_CHECK(raw, GL_OUT_OF_MEMORY);
            readRegion(ctx, st, &sr, &tmp);
            /* online-compressed (decoded) source copied into an uncompressed
             * request: the destination texel count can exceed the source byte
             * volume — grow and zero the buffer so writeRegion never over-reads
             * (values are approximate; there is no encoder for the block data) */
            if (decoded_mixed)
            {
                size_t have = (size_t)sr.w * sr.h * sr.d * ss->bpp;
                size_t need = (size_t)dr.w * dr.h * dr.d * ds->bpp;
                if (need > have)
                {
                    uint8_t *big = (uint8_t *)calloc(1, need);
                    if (big) { memcpy(big, raw, have); free(raw); raw = big; }
                }
            }
        }

        if (dc)
        {
            storeCompressed(ctx, dt, dface, dstLevel, dstX, dr_y, drz, dw, drh, drd, raw);
        }
        else
        {
            TDStorage tmp;
            storageFromBlock(ds, raw, dr.w, dr.h, dr.d, &tmp);
            writeRegion(ctx, dt, &dr, &tmp, true, true);
        }
        free(raw);
    }
}

/* exported for MGLRenderer+TextureData.m (CPU conversion paths) */
bool mgl_texdata_write_region(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *src, bool depth, bool stencil)
{
    return writeRegion(ctx, tex, r, src, depth, stencil);
}

/* ------------------------------------------------------------------------ */
/* mipmaps                                                                  */
/* ------------------------------------------------------------------------ */

static void generateMipmap(GLMContext ctx, Texture *tex, GLenum bad_target_error)
{
    switch (tex->target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;
        default:
            TD_ERR(bad_target_error);
    }

    GLuint base = tex->params.base_level;
    GLuint faces = mgl_texdata_num_faces(tex);
    TextureLevel *bl = texLevel(tex, 0, base);

    TD_CHECK(base < TD_MAX_LEVELS && bl && bl->complete, GL_INVALID_OPERATION);

    for (GLuint f = 1; f < faces; f++)
    {
        TextureLevel *l = texLevel(tex, f, base);
        TD_CHECK(l && l->complete && l->width == bl->width && l->height == bl->height && l->mtl_format == bl->mtl_format,
                 GL_INVALID_OPERATION);
    }

    const TDFormat *f = tdFormatInfo(bl->mtl_format);
    TD_CHECK(f && (f->flags & TDF_RENDERABLE) && (f->flags & TDF_FILTERABLE) &&
             !(f->flags & (TDF_INTEGER | TDF_DEPTH | TDF_STENCIL | TDF_COMPRESSED)), GL_INVALID_OPERATION);

    bool lin = (tex->target == GL_TEXTURE_1D || tex->target == GL_TEXTURE_1D_ARRAY);
    GLuint m = bl->width;
    if (!lin) m = MAX(m, bl->height);
    if (tex->target == GL_TEXTURE_3D) m = MAX(m, bl->depth);

    GLuint q = base + log2u(m);
    if (q > tex->params.max_level) q = tex->params.max_level;
    if (q >= TD_MAX_LEVELS) q = TD_MAX_LEVELS - 1;
    if (tex->immutable_storage && tex->num_levels && q > tex->num_levels - 1) q = tex->num_levels - 1;

    if (q <= base)
        return;

    if (!tex->immutable_storage)
    {
        GLuint w0 = bl->width << base, h0 = lin ? bl->height : bl->height << base;
        GLuint d0 = tex->target == GL_TEXTURE_3D ? bl->depth << base : bl->depth;
        for (GLuint face = 0; face < faces; face++)
            for (GLuint l = base + 1; l <= q; l++)
            {
                GLuint w, h, d;
                mipDims(tex, l, w0, h0, d0, &w, &h, &d);
                TextureLevel *lv = texLevel(tex, face, l);
                if (lv->complete && lv->width == w && lv->height == h && lv->depth == d && lv->mtl_format == bl->mtl_format)
                    continue;
                if (!defineLevel(ctx, tex, face, l, bl->mtl_format, w, h, d))
                    return;
            }
    }

    tex->mipmapped = true;
    tex->genmipmaps = true;

    TD_CHECK(TDM(ctx) && TDM(ctx)->generate_mipmaps && TDM(ctx)->generate_mipmaps(ctx, tex, base, q), GL_OUT_OF_MEMORY);
}

void mglGenerateMipmap(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            break;

        default:
            TD_ERR(GL_INVALID_ENUM);
    }

    Texture *tex = getTex(ctx, 0, target);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    generateMipmap(ctx, tex, GL_INVALID_ENUM);
}

void mglGenerateTextureMipmap(GLMContext ctx, GLuint texture)
{
    Texture *tex = findTexture(ctx, texture);
    TD_CHECK(tex, GL_INVALID_OPERATION);
    generateMipmap(ctx, tex, GL_INVALID_OPERATION);
}

/* ------------------------------------------------------------------------ */
/* level queries are implemented by the texparam domain (texture_query.c)   */
/* ------------------------------------------------------------------------ */

extern void mglTexParamGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params);
extern void mglTexParamGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params);

void mglGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params)
{
    mglTexParamGetTextureLevelParameteriv(ctx, texture, level, pname, params);
}

void mglGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params)
{
    mglTexParamGetTextureLevelParameterfv(ctx, texture, level, pname, params);
}

/* ------------------------------------------------------------------------ */
/* domain 'texdata' hooks                                                   */
/* ------------------------------------------------------------------------ */

void mgl_ext_init_texdata(GLMContext ctx)
{
    if (!ctx->state.ext_texdata)
        ctx->state.ext_texdata = (struct MGLExtState_texdata *)calloc(1, sizeof(struct MGLExtState_texdata));
}

bool mgl_texdata_pixel_store(GLMContext ctx, GLenum pname, GLint param)
{
    struct MGLExtState_texdata *td = TDS(ctx);

    if (!td)
        return false;

    switch (pname)
    {
        case GL_UNPACK_COMPRESSED_BLOCK_WIDTH:  td->unpack_cb[0] = param; return true;
        case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT: td->unpack_cb[1] = param; return true;
        case GL_UNPACK_COMPRESSED_BLOCK_DEPTH:  td->unpack_cb[2] = param; return true;
        case GL_UNPACK_COMPRESSED_BLOCK_SIZE:   td->unpack_cb[3] = param; return true;
        case GL_PACK_COMPRESSED_BLOCK_WIDTH:    td->pack_cb[0] = param; return true;
        case GL_PACK_COMPRESSED_BLOCK_HEIGHT:   td->pack_cb[1] = param; return true;
        case GL_PACK_COMPRESSED_BLOCK_DEPTH:    td->pack_cb[2] = param; return true;
        case GL_PACK_COMPRESSED_BLOCK_SIZE:     td->pack_cb[3] = param; return true;
        default: return false;
    }
}

bool mgl_ext_get_texdata(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    struct MGLExtState_texdata *td = TDS(ctx);
    double v;

    switch (pname)
    {
        case GL_UNPACK_COMPRESSED_BLOCK_WIDTH:  v = td ? td->unpack_cb[0] : 0; break;
        case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT: v = td ? td->unpack_cb[1] : 0; break;
        case GL_UNPACK_COMPRESSED_BLOCK_DEPTH:  v = td ? td->unpack_cb[2] : 0; break;
        case GL_UNPACK_COMPRESSED_BLOCK_SIZE:   v = td ? td->unpack_cb[3] : 0; break;
        case GL_PACK_COMPRESSED_BLOCK_WIDTH:    v = td ? td->pack_cb[0] : 0; break;
        case GL_PACK_COMPRESSED_BLOCK_HEIGHT:   v = td ? td->pack_cb[1] : 0; break;
        case GL_PACK_COMPRESSED_BLOCK_DEPTH:    v = td ? td->pack_cb[2] : 0; break;
        case GL_PACK_COMPRESSED_BLOCK_SIZE:     v = td ? td->pack_cb[3] : 0; break;
        case GL_UNPACK_SWAP_BYTES:   v = STATE(unpack.swap_bytes); break;
        case GL_UNPACK_LSB_FIRST:    v = STATE(unpack.lsb_first); break;
        case GL_UNPACK_ROW_LENGTH:   v = STATE(unpack.row_length); break;
        case GL_UNPACK_IMAGE_HEIGHT: v = STATE(unpack.image_height); break;
        case GL_UNPACK_SKIP_ROWS:    v = STATE(unpack.skip_rows); break;
        case GL_UNPACK_SKIP_PIXELS:  v = STATE(unpack.skip_pixels); break;
        case GL_UNPACK_SKIP_IMAGES:  v = STATE(unpack.skip_images); break;
        case GL_UNPACK_ALIGNMENT:    v = STATE(unpack.alignment); break;
        case GL_PACK_SWAP_BYTES:     v = STATE(pack.swap_bytes); break;
        case GL_PACK_LSB_FIRST:      v = STATE(pack.lsb_first); break;
        case GL_PACK_ROW_LENGTH:     v = STATE(pack.row_length); break;
        case GL_PACK_IMAGE_HEIGHT:   v = STATE(pack.image_height); break;
        case GL_PACK_SKIP_ROWS:      v = STATE(pack.skip_rows); break;
        case GL_PACK_SKIP_PIXELS:    v = STATE(pack.skip_pixels); break;
        case GL_PACK_SKIP_IMAGES:    v = STATE(pack.skip_images); break;
        case GL_PACK_ALIGNMENT:      v = STATE(pack.alignment); break;
        default:
            return false;
    }

    mgl_get_store(type, data, 0, v);
    return true;
}
