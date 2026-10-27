/*
 * textures_ext.c
 * MGL — texture buffers, views, multisample storage, invalidation, barriers
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * GL-side validation and state for the 'texext' domain.  Metal objects are created lazily by
 * MGLRenderer+TextureExt.m (-texextCreateMTLTexture:) when the texture is first bound.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "utils.h"
#include "hash_table.h"
#include "mgl_ext_texext.h"
#include "mgl_ext_texparam.h"
#include "mgl_ext_texdata.h"     /* TD_MAX_LEVELS: textures.c walks every level slot */   /* mgl_texparam_record_*: texparam level / view queries */

extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *currentTexture(GLMContext ctx, GLuint index);
extern Texture *newTexture(GLMContext ctx, GLenum target, GLuint texture);
extern Texture *getTex(GLMContext ctx, GLuint name, GLenum target);
extern bool checkInternalFormatForMetal(GLMContext ctx, GLuint internalformat);

#define TEXEXT_ERROR(_e_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return; } while (0)

#pragma mark - side table

static unsigned texext_hash(const Texture *tex)
{
    uintptr_t p = (uintptr_t)tex;
    return (unsigned)((p >> 4) ^ (p >> 12)) & (MGL_TEXEXT_BUCKETS - 1);
}

MGLTexExtInfo *mgl_texext_info(GLMContext ctx, const Texture *tex)
{
    if (!ctx || !tex || !ctx->state.ext_texext)
        return NULL;

    for (MGLTexExtInfo *i = ctx->state.ext_texext->buckets[texext_hash(tex)]; i; i = i->next)
        if (i->tex == tex)
            return i;

    return NULL;
}

MGLTexExtInfo *mgl_texext_info_create(GLMContext ctx, Texture *tex)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (info || !tex || !ctx->state.ext_texext)
        return info;

    info = (MGLTexExtInfo *)calloc(1, sizeof(MGLTexExtInfo));
    if (!info)
        return NULL;

    unsigned h = texext_hash(tex);
    info->tex = tex;
    info->next = ctx->state.ext_texext->buckets[h];
    ctx->state.ext_texext->buckets[h] = info;

    return info;
}

/* drop the side-table entry of a texture that is being destroyed (see the header: entries are
 * keyed by the Texture pointer, so a stale entry would alias a Texture reusing the address and
 * e.g. make texdata refuse to handle it — glGenerateMipmap then failed with GL_OUT_OF_MEMORY
 * only when a previous test case in the same context had created and deleted a texture view) */
void mgl_texext_info_free(GLMContext ctx, const Texture *tex)
{
    MGLTexExtInfo **p;

    if (!ctx || !tex || !ctx->state.ext_texext)
        return;

    for (p = &ctx->state.ext_texext->buckets[texext_hash(tex)]; *p; p = &(*p)->next)
    {
        if ((*p)->tex == tex)
        {
            MGLTexExtInfo *dead = *p;
            *p = dead->next;
            free(dead);
            return;
        }
    }
}

bool mgl_texext_is_view(GLMContext ctx, const Texture *tex)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);
    return info && info->kind == MGL_TEXEXT_VIEW;
}

GLsizei mgl_texext_samples(GLMContext ctx, const Texture *tex)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (!info)
        return 0;

    if (info->kind == MGL_TEXEXT_VIEW && info->view_root)
        return mgl_texext_samples(ctx, info->view_root);

    return info->kind == MGL_TEXEXT_MULTISAMPLE ? info->samples : 0;
}

#pragma mark - helpers

static GLuint texext_name(const Texture *tex)
{
    if (!tex || tex->name == TEX_OBJ_RES_NAME)
        return 0;
    return tex->name;
}

static void texext_mark_dirty(GLMContext ctx, Texture *tex)
{
    tex->dirty_bits |= DIRTY_TEXTURE_LEVEL;
    STATE(dirty_bits) |= DIRTY_TEX | DIRTY_TEX_BINDING;
}

/* allocate per-face level arrays so code that inspects tex->faces[f].levels[l] sees storage */
static bool texext_alloc_levels(GLMContext ctx, Texture *tex, GLuint levels, GLuint width, GLuint height, GLuint depth, GLboolean is_3d, GLenum fmt)
{
    for (int face = 0; face < _CUBE_MAP_MAX_FACE; face++)
    {
        if (tex->faces[face].levels)
            free(tex->faces[face].levels);

        /* always TD_MAX_LEVELS slots: textures.c (texLevel, freeTextureStorage, shadow
         * frees) indexes the array up to TD_MAX_LEVELS, so a shorter array is read past
         * its end and garbage "data" pointers get freed (test_texparam abort) */
        tex->faces[face].levels = (TextureLevel *)calloc(levels > TD_MAX_LEVELS ? levels : TD_MAX_LEVELS,
                                                         sizeof(TextureLevel));
        if (!tex->faces[face].levels)
        {
            ctx->error_func(ctx, __FUNCTION__, GL_OUT_OF_MEMORY);
            return false;
        }

        GLuint w = width, h = height, d = depth;
        for (GLuint l = 0; l < levels; l++)
        {
            tex->faces[face].levels[l].width = w;
            tex->faces[face].levels[l].height = h;
            tex->faces[face].levels[l].depth = d;
            tex->faces[face].levels[l].complete = true;
            /* levelFormat() reads mtl_format: without it, TexSubImage* / copy /
             * GetTexImage into the object report GL_INVALID_OPERATION
             * (KHR-GL46.texture_view.view_classes uploads into a view) */
            tex->faces[face].levels[l].mtl_format = fmt;
            w = MAX(1, w >> 1);
            if (tex->target != GL_TEXTURE_1D_ARRAY)
                h = MAX(1, h >> 1);
            if (is_3d)
                d = MAX(1, d >> 1);
        }
    }

    tex->num_levels = levels;
    tex->mipmap_levels = levels;
    return true;
}

static int texext_is_depth_stencil_format(GLenum f)
{
    switch (f)
    {
        case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F: case GL_DEPTH24_STENCIL8: case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX8: case GL_DEPTH_COMPONENT: case GL_DEPTH_STENCIL:
            return 1;
    }
    return 0;
}

static int texext_is_integer_format(GLenum f)
{
    switch (f)
    {
        case GL_R8I: case GL_R8UI: case GL_R16I: case GL_R16UI: case GL_R32I: case GL_R32UI:
        case GL_RG8I: case GL_RG8UI: case GL_RG16I: case GL_RG16UI: case GL_RG32I: case GL_RG32UI:
        case GL_RGB8I: case GL_RGB8UI: case GL_RGB16I: case GL_RGB16UI: case GL_RGB32I: case GL_RGB32UI:
        case GL_RGBA8I: case GL_RGBA8UI: case GL_RGBA16I: case GL_RGBA16UI: case GL_RGBA32I: case GL_RGBA32UI:
        case GL_RGB10_A2UI:
            return 1;
    }
    return 0;
}

/* color-, depth- or stencil-renderable sized formats (GL 4.6 Table 8.12/8.13 "CR" + depth/stencil) */
static int texext_is_renderable_format(GLenum f)
{
    if (texext_is_depth_stencil_format(f))
        return f != GL_DEPTH_COMPONENT && f != GL_DEPTH_STENCIL;

    switch (f)
    {
        case GL_R8: case GL_R8_SNORM: case GL_R16: case GL_R16_SNORM: case GL_RG8: case GL_RG8_SNORM:
        case GL_RG16: case GL_RG16_SNORM: case GL_RGB8: case GL_RGB565: case GL_RGBA4: case GL_RGB5_A1:
        case GL_RGBA8: case GL_RGBA8_SNORM: case GL_RGB10_A2: case GL_RGB10_A2UI: case GL_RGBA16: case GL_RGBA16_SNORM:
        case GL_SRGB8_ALPHA8: case GL_R16F: case GL_RG16F: case GL_RGBA16F: case GL_R32F: case GL_RG32F:
        case GL_RGBA32F: case GL_R11F_G11F_B10F:
        case GL_R8I: case GL_R8UI: case GL_R16I: case GL_R16UI: case GL_R32I: case GL_R32UI:
        case GL_RG8I: case GL_RG8UI: case GL_RG16I: case GL_RG16UI: case GL_RG32I: case GL_RG32UI:
        case GL_RGBA8I: case GL_RGBA8UI: case GL_RGBA16I: case GL_RGBA16UI: case GL_RGBA32I: case GL_RGBA32UI:
        case GL_RGB16F: case GL_RGB32F: case GL_RGB16: case GL_RGB8I: case GL_RGB8UI: case GL_RGB16I: case GL_RGB16UI:
        case GL_RGB32I: case GL_RGB32UI: case GL_SRGB8:
            return 1;
    }
    return 0;
}

/* number of layers (GL sense) of a texture; views report their own layer range */
static GLuint texext_layers(GLMContext ctx, Texture *tex)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (info && info->kind == MGL_TEXEXT_VIEW)
        return info->view_num_layers;

    switch (tex->target)
    {
        case GL_TEXTURE_1D_ARRAY: return tex->height ? tex->height : 1;
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return tex->depth ? tex->depth : 1;
        case GL_TEXTURE_CUBE_MAP: return 6;
        default: return 1;
    }
}

static GLuint texext_levels(GLMContext ctx, Texture *tex)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (info && info->kind == MGL_TEXEXT_VIEW)
        return info->view_num_levels;
    if (info && info->kind == MGL_TEXEXT_MULTISAMPLE)
        return 1;

    return tex->num_levels;
}

#pragma mark - buffer textures

bool mgl_texext_buffer_format_info(GLenum f, GLuint *texel_size, GLuint *components, bool *expand3)
{
    GLuint ts = 0, nc = 0;
    bool e3 = false;

    switch (f)
    {
        case GL_R8: case GL_R8I: case GL_R8UI: ts = 1; nc = 1; break;
        case GL_R16: case GL_R16F: case GL_R16I: case GL_R16UI: ts = 2; nc = 1; break;
        case GL_R32F: case GL_R32I: case GL_R32UI: ts = 4; nc = 1; break;
        case GL_RG8: case GL_RG8I: case GL_RG8UI: ts = 2; nc = 2; break;
        case GL_RG16: case GL_RG16F: case GL_RG16I: case GL_RG16UI: ts = 4; nc = 2; break;
        case GL_RG32F: case GL_RG32I: case GL_RG32UI: ts = 8; nc = 2; break;
        case GL_RGB32F: case GL_RGB32I: case GL_RGB32UI: ts = 12; nc = 3; e3 = true; break;
        case GL_RGBA8: case GL_RGBA8I: case GL_RGBA8UI: ts = 4; nc = 4; break;
        case GL_RGBA16: case GL_RGBA16F: case GL_RGBA16I: case GL_RGBA16UI: ts = 8; nc = 4; break;
        case GL_RGBA32F: case GL_RGBA32I: case GL_RGBA32UI: ts = 16; nc = 4; break;
        default:
            return false;
    }

    if (texel_size) *texel_size = ts;
    if (components) *components = nc;
    if (expand3) *expand3 = e3;
    return true;
}

static void texBufferCommon(GLMContext ctx, const char *func, Texture *tex, GLenum internalformat, GLuint buffer,
                            GLintptr offset, GLsizeiptr size, GLboolean ranged)
{
    GLuint texel_size, components;
    bool expand3;
    Buffer *buf = NULL;

    if (!mgl_texext_buffer_format_info(internalformat, &texel_size, &components, &expand3))
    {
        ctx->error_func(ctx, func, GL_INVALID_ENUM);
        return;
    }

    if (buffer)
    {
        buf = (Buffer *)searchHashTable(&STATE(buffer_table), buffer);
        if (!buf)
        {
            ctx->error_func(ctx, func, GL_INVALID_OPERATION);
            return;
        }

        if (ranged)
        {
            GLuint align = STATE_VAR(texture_buffer_offset_alignment);
            if (align == 0)
                align = 1;

            if (offset < 0 || size <= 0 || (offset + size) > buf->size)
            {
                ctx->error_func(ctx, func, GL_INVALID_VALUE);
                return;
            }

            if (offset % align)
            {
                ctx->error_func(ctx, func, GL_INVALID_VALUE);
                return;
            }
        }
    }

    MGLTexExtInfo *info = mgl_texext_info_create(ctx, tex);
    if (!info)
    {
        ctx->error_func(ctx, func, GL_OUT_OF_MEMORY);
        return;
    }

    info->kind = MGL_TEXEXT_BUFFER;
    info->buffer_name = buffer;
    info->buffer_format = internalformat;
    info->buffer_ranged = buffer ? ranged : GL_FALSE;
    info->buffer_offset = (buffer && ranged) ? offset : 0;
    info->buffer_size = (buffer && ranged) ? size : 0;
    info->buffer_texel_size = texel_size;
    info->buffer_components = components;
    info->buffer_expand3 = expand3;
    info->mtl_built_from = NULL;
    info->mtl_built_size = 0;

    /* GL-visible image of the buffer texture: a 1D array of texels */
    GLsizeiptr bytes = 0;
    if (buf)
        bytes = ranged ? size : buf->size;

    GLsizeiptr texels = bytes / texel_size;
    if (texels > (GLsizeiptr)STATE_VAR(max_texture_buffer_size) && STATE_VAR(max_texture_buffer_size))
        texels = STATE_VAR(max_texture_buffer_size);

    tex->internalformat = internalformat;
    tex->width = (GLuint)texels;
    tex->height = 1;
    tex->depth = 1;
    tex->is_array = false;
    tex->mipmapped = 0;
    tex->access = GL_READ_WRITE;
    if (!texext_alloc_levels(ctx, tex, 1, tex->width, 1, 1, false, internalformat))   /* texel buffer */
        return;
    tex->complete = buf != NULL;

    /* a whole-buffer binding tracks the live store size (the record's -1 sentinel
     * makes the query recompute it after a glBufferData re-definition), an explicit
     * range keeps its requested size (KHR-GL46.texture_buffer texture_buffer_parameters) */
    mgl_texparam_record_buffer(ctx, tex, buffer, info->buffer_offset,
                               buf ? (ranged ? size : (GLsizeiptr)-1) : 0);

    texext_mark_dirty(ctx, tex);
}

void mglTexBuffer(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer)
{
    if (target != GL_TEXTURE_BUFFER)
        TEXEXT_ERROR(GL_INVALID_ENUM);

    Texture *tex = getTex(ctx, 0, GL_TEXTURE_BUFFER);
    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texBufferCommon(ctx, __FUNCTION__, tex, internalformat, buffer, 0, 0, GL_FALSE);
}

void mglTexBufferRange(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    if (target != GL_TEXTURE_BUFFER)
        TEXEXT_ERROR(GL_INVALID_ENUM);

    Texture *tex = getTex(ctx, 0, GL_TEXTURE_BUFFER);
    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texBufferCommon(ctx, __FUNCTION__, tex, internalformat, buffer, offset, size, buffer != 0);
}

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);
    if (tex->target != GL_TEXTURE_BUFFER)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texBufferCommon(ctx, __FUNCTION__, tex, internalformat, buffer, 0, 0, GL_FALSE);
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);
    if (tex->target != GL_TEXTURE_BUFFER)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texBufferCommon(ctx, __FUNCTION__, tex, internalformat, buffer, offset, size, buffer != 0);
}

#pragma mark - multisample textures

static GLuint texext_max_samples_for(GLMContext ctx, GLenum internalformat)
{
    if (texext_is_depth_stencil_format(internalformat))
        return STATE_VAR(max_depth_texture_samples);
    if (texext_is_integer_format(internalformat))
        return STATE_VAR(max_integer_samples);
    return STATE_VAR(max_color_texture_samples);
}

static void texMultisampleCommon(GLMContext ctx, const char *func, Texture *tex, GLenum target,
                                 GLsizei samples, GLenum internalformat,
                                 GLsizei width, GLsizei height, GLsizei depth,
                                 GLboolean fixedsamplelocations, GLboolean storage, GLboolean proxy)
{
    GLboolean is_array = (target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY || target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY);

    if (samples <= 0)
    {
        ctx->error_func(ctx, func, GL_INVALID_VALUE);
        return;
    }

    /* GL 4.6 8.8.1: internalformat for multisample textures must be a SIZED internal
     * format (INVALID_ENUM otherwise); there is no renderability requirement at
     * creation time — KHR-GL46.texture_swizzle.functional allocates R3_G3_B2 and
     * RGB9_E5 multisample textures.  checkInternalFormatForMetal covers the sized
     * set with a Metal storage mapping.  Compressed formats are the exception:
     * they are never legal for multisample textures (no block-compressed MSAA),
     * and accepting one would hand the texture immutable storage the caller can
     * never respecify (test_texext multisample_errors). */
    if (!checkInternalFormatForMetal(ctx, internalformat))
    {
        ctx->error_func(ctx, func, GL_INVALID_ENUM);
        return;
    }
    {
        const TDFormat *tdf = tdFormatInfo(internalformat);
        if (tdf && (tdf->flags & TDF_COMPRESSED))
        {
            ctx->error_func(ctx, func, GL_INVALID_ENUM);
            return;
        }
    }

    if ((GLuint)samples > texext_max_samples_for(ctx, internalformat))
    {
        ctx->error_func(ctx, func, GL_INVALID_OPERATION);
        return;
    }

    GLsizei min_dim = storage ? 1 : 0;
    if (width < min_dim || height < min_dim || depth < min_dim ||
        (GLuint)width > STATE_VAR(max_texture_size) || (GLuint)height > STATE_VAR(max_texture_size) ||
        (is_array && STATE_VAR(max_array_texture_layers) && (GLuint)depth > STATE_VAR(max_array_texture_layers)))
    {
        ctx->error_func(ctx, func, GL_INVALID_VALUE);
        return;
    }

    GLsizei actual = samples;
    if (ctx->mtl_funcs.ext_texext && ctx->mtl_funcs.ext_texext->sampleCountFor)
    {
        actual = ctx->mtl_funcs.ext_texext->sampleCountFor(ctx, internalformat, samples);
        if (actual == 0)
        {
            ctx->error_func(ctx, func, GL_INVALID_OPERATION);
            return;
        }
    }

    if (proxy)
        return; /* proxy targets only validate; MGL keeps no proxy image state */

    if (tex == NULL)
    {
        ctx->error_func(ctx, func, GL_INVALID_OPERATION);
        return;
    }

    if (tex->immutable_storage)
    {
        ctx->error_func(ctx, func, GL_INVALID_OPERATION);
        return;
    }

    MGLTexExtInfo *info = mgl_texext_info_create(ctx, tex);
    if (!info)
    {
        ctx->error_func(ctx, func, GL_OUT_OF_MEMORY);
        return;
    }

    /* the previous Metal object (if any) is released and recreated on next bind (dirty bits) */
    info->kind = MGL_TEXEXT_MULTISAMPLE;
    info->samples_requested = samples;
    info->samples = actual;
    info->fixed_sample_locations = fixedsamplelocations ? GL_TRUE : GL_FALSE;
    info->immutable_levels = storage ? 1 : 0;
    info->invalid_levels = 0;

    tex->internalformat = internalformat;
    tex->width = width;
    tex->height = height;
    tex->depth = is_array ? depth : 1;
    tex->is_array = is_array;
    tex->mipmapped = 0;
    tex->access = GL_READ_ONLY;
    tex->mtl_requires_private_storage = true;
    tex->is_render_target = true;
    if (!texext_alloc_levels(ctx, tex, 1, width, height, tex->depth, false, internalformat))   /* multisample */
        return;
    tex->complete = (width > 0 && height > 0);

    if (storage)
        tex->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;

    mgl_texparam_record_multisample(ctx, tex, actual, info->fixed_sample_locations);

    texext_mark_dirty(ctx, tex);
}

static Texture *texext_bound_ms(GLMContext ctx, GLenum target)
{
    Texture *tex;
    GLuint index = (target == GL_TEXTURE_2D_MULTISAMPLE) ? _TEXTURE_2D_MULTISAMPLE : _TEXTURE_2D_MULTISAMPLE_ARRAY;

    tex = currentTexture(ctx, index);
    if (!tex)
        tex = getTex(ctx, 0, target); /* default texture object */

    return tex;
}

void mglTexImage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    GLboolean proxy = GL_FALSE;

    switch (target)
    {
        case GL_TEXTURE_2D_MULTISAMPLE: break;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE: proxy = GL_TRUE; break;
        default: TEXEXT_ERROR(GL_INVALID_ENUM);
    }

    texMultisampleCommon(ctx, __FUNCTION__, proxy ? NULL : texext_bound_ms(ctx, target), target, samples,
                         internalformat, width, height, 1, fixedsamplelocations, GL_FALSE, proxy);
}

void mglTexImage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    GLboolean proxy = GL_FALSE;

    switch (target)
    {
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: break;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY: proxy = GL_TRUE; break;
        default: TEXEXT_ERROR(GL_INVALID_ENUM);
    }

    texMultisampleCommon(ctx, __FUNCTION__, proxy ? NULL : texext_bound_ms(ctx, target), target, samples,
                         internalformat, width, height, depth, fixedsamplelocations, GL_FALSE, proxy);
}

void mglTexStorage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    GLboolean proxy = GL_FALSE;
    Texture *tex = NULL;

    switch (target)
    {
        case GL_TEXTURE_2D_MULTISAMPLE: break;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE: proxy = GL_TRUE; break;
        default: TEXEXT_ERROR(GL_INVALID_ENUM);
    }

    if (!proxy)
    {
        /* TexStorage on the default (zero) texture object is an error */
        tex = currentTexture(ctx, _TEXTURE_2D_MULTISAMPLE);
        if (!tex || tex->name == TEX_OBJ_RES_NAME)
            TEXEXT_ERROR(GL_INVALID_OPERATION);
    }

    texMultisampleCommon(ctx, __FUNCTION__, tex, target, samples, internalformat, width, height, 1,
                         fixedsamplelocations, GL_TRUE, proxy);
}

void mglTexStorage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    GLboolean proxy = GL_FALSE;
    Texture *tex = NULL;

    switch (target)
    {
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: break;
        case GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY: proxy = GL_TRUE; break;
        default: TEXEXT_ERROR(GL_INVALID_ENUM);
    }

    if (!proxy)
    {
        tex = currentTexture(ctx, _TEXTURE_2D_MULTISAMPLE_ARRAY);
        if (!tex || tex->name == TEX_OBJ_RES_NAME)
            TEXEXT_ERROR(GL_INVALID_OPERATION);
    }

    texMultisampleCommon(ctx, __FUNCTION__, tex, target, samples, internalformat, width, height, depth,
                         fixedsamplelocations, GL_TRUE, proxy);
}

void mglTextureStorage2DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);
    if (tex->target != GL_TEXTURE_2D_MULTISAMPLE)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texMultisampleCommon(ctx, __FUNCTION__, tex, GL_TEXTURE_2D_MULTISAMPLE, samples, internalformat, width, height, 1,
                         fixedsamplelocations, GL_TRUE, GL_FALSE);
}

void mglTextureStorage3DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
        TEXEXT_ERROR(GL_INVALID_OPERATION);
    if (tex->target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    texMultisampleCommon(ctx, __FUNCTION__, tex, GL_TEXTURE_2D_MULTISAMPLE_ARRAY, samples, internalformat, width, height, depth,
                         fixedsamplelocations, GL_TRUE, GL_FALSE);
}

#pragma mark - texture views

/* GL 4.6 Table 8.22 view classes; 0 = not in the table (exact match required) */
static int texext_view_class(GLenum f)
{
    switch (f)
    {
        case GL_RGBA32F: case GL_RGBA32UI: case GL_RGBA32I:
            return 128;
        case GL_RGB32F: case GL_RGB32UI: case GL_RGB32I:
            return 96;
        case GL_RGBA16F: case GL_RG32F: case GL_RGBA16UI: case GL_RG32UI: case GL_RGBA16I: case GL_RG32I:
        case GL_RGBA16: case GL_RGBA16_SNORM:
            return 64;
        case GL_RGB16: case GL_RGB16_SNORM: case GL_RGB16F: case GL_RGB16UI: case GL_RGB16I:
            return 48;
        case GL_RG16F: case GL_R11F_G11F_B10F: case GL_R32F: case GL_RGB10_A2UI: case GL_RGBA8UI: case GL_RG16UI:
        case GL_R32UI: case GL_RGBA8I: case GL_RG16I: case GL_R32I: case GL_RGB10_A2: case GL_RGBA8: case GL_RG16:
        case GL_RGBA8_SNORM: case GL_RG16_SNORM: case GL_SRGB8_ALPHA8: case GL_RGB9_E5:
            return 32;
        case GL_RGB8: case GL_RGB8_SNORM: case GL_SRGB8: case GL_RGB8UI: case GL_RGB8I:
            return 24;
        case GL_R16F: case GL_RG8UI: case GL_R16UI: case GL_RG8I: case GL_R16I: case GL_RG8: case GL_R16:
        case GL_RG8_SNORM: case GL_R16_SNORM:
            return 16;
        case GL_R8UI: case GL_R8I: case GL_R8: case GL_R8_SNORM:
            return 8;
        case GL_COMPRESSED_RED_RGTC1: case GL_COMPRESSED_SIGNED_RED_RGTC1:
            return 1001;
        case GL_COMPRESSED_RG_RGTC2: case GL_COMPRESSED_SIGNED_RG_RGTC2:
            return 1002;
        case GL_COMPRESSED_RGBA_BPTC_UNORM: case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
            return 1003;
        case GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT: case GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT:
            return 1004;
    }
    return 0;
}

static bool texext_view_formats_compatible(GLenum orig, GLenum view)
{
    if (orig == view)
        return true;

    int a = texext_view_class(orig), b = texext_view_class(view);
    return a != 0 && a == b;
}

/* GL 4.6 Table 8.21 */
static bool texext_view_targets_compatible(GLenum orig, GLenum view)
{
    switch (orig)
    {
        case GL_TEXTURE_1D:
        case GL_TEXTURE_1D_ARRAY:
            return view == GL_TEXTURE_1D || view == GL_TEXTURE_1D_ARRAY;
        case GL_TEXTURE_2D:
            return view == GL_TEXTURE_2D || view == GL_TEXTURE_2D_ARRAY;
        case GL_TEXTURE_3D:
            return view == GL_TEXTURE_3D;
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            return view == GL_TEXTURE_CUBE_MAP || view == GL_TEXTURE_2D || view == GL_TEXTURE_2D_ARRAY ||
                   view == GL_TEXTURE_CUBE_MAP_ARRAY;
        case GL_TEXTURE_RECTANGLE:
            return view == GL_TEXTURE_RECTANGLE;
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return view == GL_TEXTURE_2D_MULTISAMPLE || view == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
        default:
            return false; /* GL_TEXTURE_BUFFER cannot be viewed */
    }
}

static bool texext_is_texture_target(GLenum t)
{
    switch (t)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_BUFFER: case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
    }
    return false;
}

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    Texture *orig, *tex;

    if (getenv("MGL_VIEW_DEBUG")) fprintf(stderr, "MGL view: enter texture=%u orig=%u target=%x fmt=%x %u/%u %u/%u\n",
                                          texture, origtexture, target, internalformat, minlevel, numlevels, minlayer, numlayers);
    /* texture must be a name returned by GenTextures that has not yet been bound */
    if (texture == 0)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    if (texture >= STATE(texture_table).current_name)
        TEXEXT_ERROR(GL_INVALID_OPERATION);   /* never handed out by GenTextures */

    if (findTexture(ctx, texture))
    { if (getenv("MGL_VIEW_DEBUG")) fprintf(stderr, "MGL view: name %u already exists\n", texture);
        TEXEXT_ERROR(GL_INVALID_OPERATION); }

    orig = origtexture ? findTexture(ctx, origtexture) : NULL;
    if (!orig)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    if (!texext_is_texture_target(target))
        TEXEXT_ERROR(GL_INVALID_ENUM);

    if (!orig->immutable_storage)
    { if (getenv("MGL_VIEW_DEBUG")) fprintf(stderr, "MGL view: parent not immutable\n");
        TEXEXT_ERROR(GL_INVALID_OPERATION); }

    if (!texext_view_targets_compatible(orig->target, target))
    { if (getenv("MGL_VIEW_DEBUG")) fprintf(stderr, "MGL view: target %x -> %x\n", orig->target, target);
        TEXEXT_ERROR(GL_INVALID_OPERATION); }

    if (!texext_view_formats_compatible(orig->internalformat, internalformat))
    { if (getenv("MGL_VIEW_DEBUG")) fprintf(stderr, "MGL view: format %x -> %x\n", orig->internalformat, internalformat);
        TEXEXT_ERROR(GL_INVALID_OPERATION); }

    GLuint orig_levels = texext_levels(ctx, orig);
    GLuint orig_layers = texext_layers(ctx, orig);

    if (minlevel >= orig_levels || minlayer >= orig_layers)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    /* the spec's <numlayers> validation uses the caller's value, before the view is
     * clamped to the parent's layer count (a 1D/2D/3D/RECTANGLE/2D_MULTISAMPLE view
     * must pass exactly 1) */
    if (numlevels == 0 || numlayers == 0)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D:
        case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_2D_MULTISAMPLE:
            if (numlayers != 1)
                TEXEXT_ERROR(GL_INVALID_VALUE);
            break;
        case GL_TEXTURE_CUBE_MAP:
            if (numlayers != 6)
                TEXEXT_ERROR(GL_INVALID_VALUE);
            break;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            if (numlayers == 0 || (numlayers % 6) != 0)
                TEXEXT_ERROR(GL_INVALID_VALUE);
            break;
        default:
            break;
    }

    numlevels = MIN(numlevels, orig_levels - minlevel);
    numlayers = MIN(numlayers, orig_layers - minlayer);

    if ((target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY) && orig->width != orig->height)
        TEXEXT_ERROR(GL_INVALID_OPERATION);

    MGLTexExtInfo *oinfo = mgl_texext_info(ctx, orig);
    Texture *root = (oinfo && oinfo->kind == MGL_TEXEXT_VIEW && oinfo->view_root) ? oinfo->view_root : orig;
    GLuint base_level = (oinfo && oinfo->kind == MGL_TEXEXT_VIEW) ? oinfo->view_min_level : 0;
    GLuint base_layer = (oinfo && oinfo->kind == MGL_TEXEXT_VIEW) ? oinfo->view_min_layer : 0;

    tex = newTexture(ctx, target, texture);
    if (!tex)
        TEXEXT_ERROR(GL_OUT_OF_MEMORY);
    insertHashElement(&STATE(texture_table), texture, tex);

    MGLTexExtInfo *info = mgl_texext_info_create(ctx, tex);
    MGLTexExtInfo *rinfo = mgl_texext_info_create(ctx, root);
    if (!info || !rinfo)
        TEXEXT_ERROR(GL_OUT_OF_MEMORY);

    info->kind = MGL_TEXEXT_VIEW;
    info->view_root = root;
    info->view_min_level = base_level + minlevel;
    info->view_num_levels = numlevels;
    info->view_min_layer = base_layer + minlayer;
    info->view_num_layers = numlayers;
    info->immutable_levels = (oinfo && oinfo->immutable_levels) ?
                                 oinfo->immutable_levels :   /* ARB_texture_view: a view reports
                                                               * the ORIGINAL's TEXTURE_IMMUTABLE_LEVELS */
                                 orig_levels;
    {
        bool upgrade = !rinfo->has_views;

        rinfo->has_views = GL_TRUE;
        if (internalformat != root->internalformat && !rinfo->has_format_views)
        {
            rinfo->has_format_views = GL_TRUE;
            upgrade = true;
        }

        if (upgrade && root->mtl_data && ctx->mtl_funcs.ext_texext && ctx->mtl_funcs.ext_texext->prepareViewSource)
            ctx->mtl_funcs.ext_texext->prepareViewSource(ctx, root);
    }

    MGLTexExtInfo *root_ms = mgl_texext_info(ctx, root);
    if (root_ms && root_ms->kind == MGL_TEXEXT_MULTISAMPLE)
    {
        info->samples = root_ms->samples;
        info->samples_requested = root_ms->samples_requested;
        info->fixed_sample_locations = root_ms->fixed_sample_locations;
    }

    /* texture parameters are inherited from origtexture */
    tex->params = orig->params;
    tex->params.mtl_data = NULL;

    GLuint w = MAX(1, orig->width >> minlevel);
    GLuint h = (orig->target == GL_TEXTURE_1D_ARRAY || orig->target == GL_TEXTURE_1D) ? 1 : MAX(1, orig->height >> minlevel);
    GLuint d = 1;
    if (target == GL_TEXTURE_3D)
        d = MAX(1, orig->depth >> minlevel);
    else if (target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        d = numlayers;
    if (target == GL_TEXTURE_1D_ARRAY)
        h = numlayers;

    tex->internalformat = internalformat;
    tex->width = w;
    tex->height = h;
    tex->depth = d;
    tex->is_array = (target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_2D_ARRAY ||
                     target == GL_TEXTURE_CUBE_MAP_ARRAY || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
    tex->access = orig->access ? orig->access : GL_READ_ONLY;
    tex->mtl_requires_private_storage = orig->mtl_requires_private_storage;
    if (!texext_alloc_levels(ctx, tex, numlevels, w, h, d, target == GL_TEXTURE_3D, internalformat))
        return;
    tex->mipmapped = numlevels > 1;
    tex->complete = true;
    tex->immutable_storage = BUFFER_IMMUTABLE_STORAGE_FLAG;

    GLuint orig_immutable_levels = (oinfo && oinfo->immutable_levels) ? oinfo->immutable_levels : orig_levels;
    mgl_texparam_record_view(ctx, tex, info->view_min_level, info->view_num_levels,
                             info->view_min_layer, info->view_num_layers, orig_immutable_levels);
    if (info->samples)
        mgl_texparam_record_multisample(ctx, tex, info->samples, info->fixed_sample_locations);

    texext_mark_dirty(ctx, tex);
    if (getenv("MGL_VIEW_DEBUG")) { fprintf(stderr, "MGL view: done name=%u levels=%u\n", tex->name, info->view_num_levels); }
}

#pragma mark - invalidation

static GLuint texext_max_levels_for_target(GLMContext ctx, GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_BUFFER: case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return 1;
        case GL_TEXTURE_3D:
            return ilog2(MAX(STATE_VAR(max_3d_texture_size), 1)) + 1;
        case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY:
            return ilog2(MAX(STATE_VAR(max_cube_map_texture_size), 1)) + 1;
        default:
            return ilog2(MAX(STATE_VAR(max_texture_size), 1)) + 1;
    }
}

static bool texext_validate_invalidate(GLMContext ctx, const char *func, GLuint texture, GLint level, Texture **out)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
    {
        ctx->error_func(ctx, func, GL_INVALID_VALUE);
        return false;
    }

    if (level < 0 || (GLuint)level >= texext_max_levels_for_target(ctx, tex->target))
    {
        ctx->error_func(ctx, func, GL_INVALID_VALUE);
        return false;
    }

    *out = tex;
    return true;
}

/* GL dimensions of a level: width, height, depth (layers for array/cube targets) */
static void texext_level_dims(GLMContext ctx, Texture *tex, GLint level, GLuint *w, GLuint *h, GLuint *d)
{
    *w = *h = *d = 0;

    if ((GLuint)level >= texext_levels(ctx, tex) && !(tex->target == GL_TEXTURE_BUFFER && level == 0))
        return;

    if (tex->target == GL_TEXTURE_BUFFER)
    {
        *w = tex->width; *h = 1; *d = 1;
        return;
    }

    if (tex->faces[0].levels && (GLuint)level < tex->num_levels && tex->faces[0].levels[level].width)
    {
        *w = tex->faces[0].levels[level].width;
        *h = tex->faces[0].levels[level].height;
        *d = tex->faces[0].levels[level].depth;
    }
    else
    {
        *w = MAX(1, tex->width >> level);
        *h = MAX(1, tex->height >> level);
        *d = MAX(1, tex->depth >> level);
    }

    switch (tex->target)
    {
        case GL_TEXTURE_1D: *h = 1; *d = 1; break;
        case GL_TEXTURE_1D_ARRAY: *h = texext_layers(ctx, tex); *d = 1; break;
        case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_2D_MULTISAMPLE: *d = 1; break;
        case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            *d = texext_layers(ctx, tex); break;
        default: break;
    }
}

static void texext_invalidate_level(GLMContext ctx, Texture *tex, GLint level)
{
    MGLTexExtInfo *info = mgl_texext_info_create(ctx, tex);

    if (info && level < 32)
        info->invalid_levels |= (1u << level);

    if (ctx->mtl_funcs.ext_texext && ctx->mtl_funcs.ext_texext->invalidateTexImage)
        ctx->mtl_funcs.ext_texext->invalidateTexImage(ctx, tex, level);
}

void mglInvalidateTexImage(GLMContext ctx, GLuint texture, GLint level)
{
    Texture *tex;

    if (!texext_validate_invalidate(ctx, __FUNCTION__, texture, level, &tex))
        return;

    texext_invalidate_level(ctx, tex, level);
}

void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
{
    Texture *tex;
    GLuint lw, lh, ld;

    if (!texext_validate_invalidate(ctx, __FUNCTION__, texture, level, &tex))
        return;

    if (width < 0 || height < 0 || depth < 0)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    texext_level_dims(ctx, tex, level, &lw, &lh, &ld);

    /* border is always 0 in core profile */
    if (xoffset < 0 || yoffset < 0 || zoffset < 0 ||
        (GLint64)xoffset + width > (GLint64)lw ||
        (GLint64)yoffset + height > (GLint64)lh ||
        (GLint64)zoffset + depth > (GLint64)ld)
        TEXEXT_ERROR(GL_INVALID_VALUE);

    if (width == 0 || height == 0 || depth == 0)
        return; /* empty region: nothing becomes undefined */

    /* a region covering the whole level is a whole-image invalidate (lets the next render pass
       skip loading it); a partial region leaves the other texels intact, so the old contents of
       the region are kept — a valid choice for "undefined" contents. */
    if (xoffset == 0 && yoffset == 0 && zoffset == 0 &&
        (GLuint)width == lw && (GLuint)height == lh && (GLuint)depth == ld)
        texext_invalidate_level(ctx, tex, level);
}

#pragma mark - barriers

void mglTextureBarrier(GLMContext ctx)
{
    if (ctx->mtl_funcs.ext_texext && ctx->mtl_funcs.ext_texext->textureBarrier)
        ctx->mtl_funcs.ext_texext->textureBarrier(ctx);
    else
        ctx->mtl_funcs.mtlFlush(ctx, false); /* no Metal hook table: submit work so it is ordered */
}

#pragma mark - queries for other domains

bool mgl_texext_get_tex_parameter(GLMContext ctx, Texture *tex, GLenum pname, GLint *out)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    switch (pname)
    {
        case GL_TEXTURE_VIEW_MIN_LEVEL:
            *out = (info && info->kind == MGL_TEXEXT_VIEW) ? (GLint)info->view_min_level : 0;
            return true;
        case GL_TEXTURE_VIEW_NUM_LEVELS:
            *out = (info && info->kind == MGL_TEXEXT_VIEW) ? (GLint)info->view_num_levels : 0;
            return true;
        case GL_TEXTURE_VIEW_MIN_LAYER:
            *out = (info && info->kind == MGL_TEXEXT_VIEW) ? (GLint)info->view_min_layer : 0;
            return true;
        case GL_TEXTURE_VIEW_NUM_LAYERS:
            *out = (info && info->kind == MGL_TEXEXT_VIEW) ? (GLint)info->view_num_layers : 0;
            return true;
        case GL_TEXTURE_IMMUTABLE_LEVELS:
            if (info && (info->kind == MGL_TEXEXT_VIEW || info->kind == MGL_TEXEXT_MULTISAMPLE))
            {
                *out = (GLint)info->immutable_levels;
                return true;
            }
            return false;
        case GL_TEXTURE_IMMUTABLE_FORMAT:
            if (info && (info->kind == MGL_TEXEXT_VIEW || info->kind == MGL_TEXEXT_MULTISAMPLE))
            {
                *out = tex->immutable_storage ? GL_TRUE : GL_FALSE;
                return true;
            }
            return false;
    }

    return false;
}

bool mgl_texext_get_tex_level_parameter(GLMContext ctx, Texture *tex, GLint level, GLenum pname, GLint *out)
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    switch (pname)
    {
        case GL_TEXTURE_SAMPLES:
            *out = (level == 0) ? mgl_texext_samples(ctx, tex) : 0;
            return true;
        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            if (info && info->kind == MGL_TEXEXT_VIEW && info->view_root)
                info = mgl_texext_info(ctx, info->view_root);
            *out = (info && info->kind == MGL_TEXEXT_MULTISAMPLE) ? info->fixed_sample_locations : GL_TRUE;
            return true;
        case GL_TEXTURE_BUFFER_DATA_STORE_BINDING:
            *out = (info && info->kind == MGL_TEXEXT_BUFFER) ? (GLint)info->buffer_name : 0;
            return true;
        case GL_TEXTURE_BUFFER_OFFSET:
            *out = (info && info->kind == MGL_TEXEXT_BUFFER) ? (GLint)info->buffer_offset : 0;
            return true;
        case GL_TEXTURE_BUFFER_SIZE:
            if (info && info->kind == MGL_TEXEXT_BUFFER && info->buffer_name)
            {
                /* the spec derives the size from the data store of the bound buffer:
                 * a resize after the binding shows up here (KHR-GL46.texture_buffer
                 * texture_buffer_parameters re-uploads and re-queries without rebinding).
                 * A ranged binding keeps its range while the store still matches it. */
                Buffer *buf = (Buffer *)searchHashTable(&STATE(buffer_table), info->buffer_name);
                if (!buf)
                    *out = 0;
                else if (info->buffer_ranged &&
                         (GLsizeiptr)buf->size == info->buffer_offset + (GLsizeiptr)info->buffer_size)
                    *out = (GLint)info->buffer_size;
                else
                    *out = (buf->size > info->buffer_offset) ? (GLint)(buf->size - info->buffer_offset) : 0;
            }
            else
                *out = 0;
            return true;
        case GL_TEXTURE_WIDTH:
        case GL_TEXTURE_HEIGHT:
        case GL_TEXTURE_DEPTH:
        case GL_TEXTURE_INTERNAL_FORMAT:
            if (info && info->kind == MGL_TEXEXT_BUFFER)
            {
                if (level != 0) { *out = 0; return true; }
                if (pname == GL_TEXTURE_INTERNAL_FORMAT) *out = (GLint)info->buffer_format;
                else if (pname == GL_TEXTURE_WIDTH) *out = info->buffer_name ? (GLint)tex->width : 0;
                else *out = info->buffer_name ? 1 : 0;
                return true;
            }
            return false;
    }

    return false;
}

#pragma mark - domain 'texext' hooks

void mgl_ext_init_texext(GLMContext ctx)
{
    if (ctx->state.ext_texext)
        return;

    ctx->state.ext_texext = (struct MGLExtState_texext *)calloc(1, sizeof(struct MGLExtState_texext));
}

static GLuint texext_unit_binding(GLMContext ctx, GLuint index)
{
    Texture *tex = STATE(texture_units[STATE(active_texture)].textures[index]);
    return texext_name(tex);
}

bool mgl_ext_get_texext(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname)
    {
        case GL_TEXTURE_BUFFER_BINDING: /* == GL_TEXTURE_BUFFER: buffer bound to the TEXTURE_BUFFER buffer target */
        {
            Buffer *buf = STATE(buffers[_TEXTURE_BUFFER]);
            mgl_get_store(type, data, 0, buf ? buf->name : 0);
            return true;
        }

        case GL_TEXTURE_BINDING_BUFFER:
            mgl_get_store(type, data, 0, texext_unit_binding(ctx, _TEXTURE_BUFFER_TARGET));
            return true;

        case GL_TEXTURE_BINDING_2D_MULTISAMPLE:
            mgl_get_store(type, data, 0, texext_unit_binding(ctx, _TEXTURE_2D_MULTISAMPLE));
            return true;

        case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY:
            mgl_get_store(type, data, 0, texext_unit_binding(ctx, _TEXTURE_2D_MULTISAMPLE_ARRAY));
            return true;
    }

    return false;
}
