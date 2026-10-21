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
 * framebuffers.c
 * MGL
 *
 * Framebuffer and renderbuffer objects: creation, attachment, completeness,
 * queries, invalidation and the DSA entry points. Metal work (render pass
 * construction, clears, blits, readback) lives in MGLRenderer+Framebuffer.m
 * and is reached through ctx->mtl_funcs.ext_fb.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "glm_context.h"
#include "pixel_utils.h"
#include "utils.h"
#include "mgl_ext_fb.h"

#define FB_ERROR(_e_)           do { ERROR_RETURN(_e_); return; } while (0)
#define FB_ERROR_V(_e_, _v_)    do { ERROR_RETURN(_e_); return (_v_); } while (0)

extern Texture *newTexObj(GLMContext ctx, GLenum target);
extern Texture *findTexture(GLMContext ctx, GLuint texture);

/* implemented in rendering.c / framebuffer_ext.c */
extern void mgl_fb_set_draw_buffers(GLMContext ctx, Framebuffer *fbo, GLsizei n, const GLenum *bufs, bool single);
extern void mgl_fb_set_read_buffer(GLMContext ctx, Framebuffer *fbo, GLenum src);
extern void mgl_fb_clear_buffer(GLMContext ctx, Framebuffer *fbo, GLenum buffer, GLint drawbuffer, int kind, const void *value, GLfloat depth, GLint stencil, bool fi);
extern void mgl_fb_sync_bindings(GLMContext ctx);

#define FBEX(_fbo_) ((MGLFramebufferEx *)(_fbo_))
#define RBEX(_rb_)  ((MGLRenderbufferEx *)(_rb_))

static struct MGLExtMetal_fb *fbm(GLMContext ctx) { return ctx->mtl_funcs.ext_fb; }

#pragma mark internal format table

#define CR  GL_TRUE, GL_FALSE, GL_FALSE
#define NCR GL_FALSE, GL_FALSE, GL_FALSE
#define DR  GL_FALSE, GL_TRUE, GL_FALSE
#define SR  GL_FALSE, GL_FALSE, GL_TRUE
#define DSR GL_FALSE, GL_TRUE, GL_TRUE
#define UN  GL_UNSIGNED_NORMALIZED
#define SN  GL_SIGNED_NORMALIZED
#define FL  GL_FLOAT
#define SI  GL_INT
#define UI  GL_UNSIGNED_INT

static const MGLFBFormatInfo fb_formats[] = {
    /* unsized */
    { GL_RED,               GL_RED,  8,0,0,0, 0,0, UN, 0, CR },
    { GL_RG,                GL_RG,   8,8,0,0, 0,0, UN, 0, CR },
    { GL_RGB,               GL_RGB,  8,8,8,0, 0,0, UN, 0, CR },
    { GL_RGBA,              GL_RGBA, 8,8,8,8, 0,0, UN, 0, CR },
    { GL_DEPTH_COMPONENT,   GL_DEPTH_COMPONENT, 0,0,0,0, 32,0, FL, 0, DR },
    { GL_DEPTH_STENCIL,     GL_DEPTH_STENCIL,   0,0,0,0, 32,8, FL, 0, DSR },
    { GL_STENCIL_INDEX,     GL_STENCIL_INDEX,   0,0,0,0, 0,8, UI, 0, SR },
    /* normalized */
    { GL_R8,                GL_RED,  8,0,0,0, 0,0, UN, 0, CR },
    { GL_R8_SNORM,          GL_RED,  8,0,0,0, 0,0, SN, 0, CR },
    { GL_R16,               GL_RED,  16,0,0,0, 0,0, UN, 0, CR },
    { GL_R16_SNORM,         GL_RED,  16,0,0,0, 0,0, SN, 0, CR },
    { GL_RG8,               GL_RG,   8,8,0,0, 0,0, UN, 0, CR },
    { GL_RG8_SNORM,         GL_RG,   8,8,0,0, 0,0, SN, 0, CR },
    { GL_RG16,              GL_RG,   16,16,0,0, 0,0, UN, 0, CR },
    { GL_RG16_SNORM,        GL_RG,   16,16,0,0, 0,0, SN, 0, CR },
    { GL_R3_G3_B2,          GL_RGB,  3,3,2,0, 0,0, UN, 0, CR },
    { GL_RGB4,              GL_RGB,  4,4,4,0, 0,0, UN, 0, CR },
    { GL_RGB5,              GL_RGB,  5,5,5,0, 0,0, UN, 0, CR },
    { GL_RGB565,            GL_RGB,  5,6,5,0, 0,0, UN, 0, CR },
    { GL_RGB8,              GL_RGB,  8,8,8,0, 0,0, UN, 0, CR },
    { GL_RGB8_SNORM,        GL_RGB,  8,8,8,0, 0,0, SN, 0, CR },
    { GL_RGB10,             GL_RGB,  10,10,10,0, 0,0, UN, 0, CR },
    { GL_RGB12,             GL_RGB,  12,12,12,0, 0,0, UN, 0, CR },
    { GL_RGB16,             GL_RGB,  16,16,16,0, 0,0, UN, 0, CR },
    { GL_RGB16_SNORM,       GL_RGB,  16,16,16,0, 0,0, SN, 0, CR },
    { GL_RGBA2,             GL_RGBA, 2,2,2,2, 0,0, UN, 0, CR },
    { GL_RGBA4,             GL_RGBA, 4,4,4,4, 0,0, UN, 0, CR },
    { GL_RGB5_A1,           GL_RGBA, 5,5,5,1, 0,0, UN, 0, CR },
    { GL_RGBA8,             GL_RGBA, 8,8,8,8, 0,0, UN, 0, CR },
    { GL_RGBA8_SNORM,       GL_RGBA, 8,8,8,8, 0,0, SN, 0, CR },
    { GL_RGB10_A2,          GL_RGBA, 10,10,10,2, 0,0, UN, 0, CR },
    { GL_RGBA12,            GL_RGBA, 12,12,12,12, 0,0, UN, 0, CR },
    { GL_RGBA16,            GL_RGBA, 16,16,16,16, 0,0, UN, 0, CR },
    { GL_RGBA16_SNORM,      GL_RGBA, 16,16,16,16, 0,0, SN, 0, CR },
    { GL_SRGB8,             GL_RGB,  8,8,8,0, 0,0, UN, 1, CR },
    { GL_SRGB8_ALPHA8,      GL_RGBA, 8,8,8,8, 0,0, UN, 1, CR },
    /* float */
    { GL_R16F,              GL_RED,  16,0,0,0, 0,0, FL, 0, CR },
    { GL_RG16F,             GL_RG,   16,16,0,0, 0,0, FL, 0, CR },
    { GL_RGB16F,            GL_RGB,  16,16,16,0, 0,0, FL, 0, CR },
    { GL_RGBA16F,           GL_RGBA, 16,16,16,16, 0,0, FL, 0, CR },
    { GL_R32F,              GL_RED,  32,0,0,0, 0,0, FL, 0, CR },
    { GL_RG32F,             GL_RG,   32,32,0,0, 0,0, FL, 0, CR },
    { GL_RGB32F,            GL_RGB,  32,32,32,0, 0,0, FL, 0, CR },
    { GL_RGBA32F,           GL_RGBA, 32,32,32,32, 0,0, FL, 0, CR },
    { GL_R11F_G11F_B10F,    GL_RGB,  11,11,10,0, 0,0, FL, 0, CR },
    { GL_RGB9_E5,           GL_RGB,  9,9,9,0, 0,0, FL, 0, NCR },
    /* integer */
    { GL_R8I,               GL_RED,  8,0,0,0, 0,0, SI, 0, CR },
    { GL_R8UI,              GL_RED,  8,0,0,0, 0,0, UI, 0, CR },
    { GL_R16I,              GL_RED,  16,0,0,0, 0,0, SI, 0, CR },
    { GL_R16UI,             GL_RED,  16,0,0,0, 0,0, UI, 0, CR },
    { GL_R32I,              GL_RED,  32,0,0,0, 0,0, SI, 0, CR },
    { GL_R32UI,             GL_RED,  32,0,0,0, 0,0, UI, 0, CR },
    { GL_RG8I,              GL_RG,   8,8,0,0, 0,0, SI, 0, CR },
    { GL_RG8UI,             GL_RG,   8,8,0,0, 0,0, UI, 0, CR },
    { GL_RG16I,             GL_RG,   16,16,0,0, 0,0, SI, 0, CR },
    { GL_RG16UI,            GL_RG,   16,16,0,0, 0,0, UI, 0, CR },
    { GL_RG32I,             GL_RG,   32,32,0,0, 0,0, SI, 0, CR },
    { GL_RG32UI,            GL_RG,   32,32,0,0, 0,0, UI, 0, CR },
    { GL_RGB8I,             GL_RGB,  8,8,8,0, 0,0, SI, 0, CR },
    { GL_RGB8UI,            GL_RGB,  8,8,8,0, 0,0, UI, 0, CR },
    { GL_RGB16I,            GL_RGB,  16,16,16,0, 0,0, SI, 0, CR },
    { GL_RGB16UI,           GL_RGB,  16,16,16,0, 0,0, UI, 0, CR },
    { GL_RGB32I,            GL_RGB,  32,32,32,0, 0,0, SI, 0, CR },
    { GL_RGB32UI,           GL_RGB,  32,32,32,0, 0,0, UI, 0, CR },
    { GL_RGBA8I,            GL_RGBA, 8,8,8,8, 0,0, SI, 0, CR },
    { GL_RGBA8UI,           GL_RGBA, 8,8,8,8, 0,0, UI, 0, CR },
    { GL_RGBA16I,           GL_RGBA, 16,16,16,16, 0,0, SI, 0, CR },
    { GL_RGBA16UI,          GL_RGBA, 16,16,16,16, 0,0, UI, 0, CR },
    { GL_RGBA32I,           GL_RGBA, 32,32,32,32, 0,0, SI, 0, CR },
    { GL_RGBA32UI,          GL_RGBA, 32,32,32,32, 0,0, UI, 0, CR },
    { GL_RGB10_A2UI,        GL_RGBA, 10,10,10,2, 0,0, UI, 0, CR },
    /* depth / stencil */
    { GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, 0,0,0,0, 16,0, UN, 0, DR },
    { GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, 0,0,0,0, 32,0, FL, 0, DR },
    { GL_DEPTH_COMPONENT32, GL_DEPTH_COMPONENT, 0,0,0,0, 32,0, FL, 0, DR },
    { GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT, 0,0,0,0, 32,0, FL, 0, DR },
    { GL_DEPTH24_STENCIL8,  GL_DEPTH_STENCIL,   0,0,0,0, 32,8, FL, 0, DSR },
    { GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL,   0,0,0,0, 32,8, FL, 0, DSR },
    { GL_STENCIL_INDEX1,    GL_STENCIL_INDEX,   0,0,0,0, 0,8, UI, 0, SR },
    { GL_STENCIL_INDEX4,    GL_STENCIL_INDEX,   0,0,0,0, 0,8, UI, 0, SR },
    { GL_STENCIL_INDEX8,    GL_STENCIL_INDEX,   0,0,0,0, 0,8, UI, 0, SR },
    { GL_STENCIL_INDEX16,   GL_STENCIL_INDEX,   0,0,0,0, 0,8, UI, 0, SR },
};

const MGLFBFormatInfo *mgl_fb_format_info(GLenum internalformat)
{
    for (size_t i = 0; i < sizeof(fb_formats) / sizeof(fb_formats[0]); i++)
        if (fb_formats[i].internalformat == internalformat)
            return &fb_formats[i];
    return NULL;
}

#pragma mark object management

static MGLRenderbufferEx *newRenderbufferEx(GLuint name)
{
    MGLRenderbufferEx *rb = (MGLRenderbufferEx *)calloc(1, sizeof(MGLRenderbufferEx));
    if (!rb) return NULL;
    rb->base.name = name;
    rb->internalformat = GL_RGBA;
    return rb;
}

Renderbuffer *findRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    if (renderbuffer == 0) return NULL;
    return (Renderbuffer *)searchHashTable(&STATE(renderbuffer_table), renderbuffer);
}

static void initDrawState(MGLFBDrawState *ds, GLenum def)
{
    memset(ds, 0, sizeof(*ds));
    ds->draw_buffers[0] = def;
    for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) ds->draw_buffers[i] = GL_NONE;
    ds->num_draw_buffers = 1;
    ds->read_buffer = def;
}

static MGLFramebufferEx *newFramebufferEx(GLuint name)
{
    MGLFramebufferEx *fb = (MGLFramebufferEx *)calloc(1, sizeof(MGLFramebufferEx));
    if (!fb) return NULL;
    fb->base.name = name;
    initDrawState(&fb->ds, GL_COLOR_ATTACHMENT0);
    return fb;
}

Framebuffer *findFrameBuffer(GLMContext ctx, GLuint framebuffer)
{
    if (framebuffer == 0) return NULL;
    return (Framebuffer *)searchHashTable(&STATE(framebuffer_table), framebuffer);
}

Framebuffer *currentFBOForType(GLMContext ctx, GLenum target)
{
    switch(target)
    {
        case GL_FRAMEBUFFER:
        case GL_DRAW_FRAMEBUFFER:
            return ctx->state.framebuffer;
        case GL_READ_FRAMEBUFFER:
            return ctx->state.readbuffer;
    }
    return NULL;
}

static bool validFBTarget(GLenum target)
{
    return target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER;
}

MGLFBDrawState *mgl_fb_draw_state(GLMContext ctx, Framebuffer *fbo)
{
    if (fbo) return &FBEX(fbo)->ds;
    return &ctx->state.ext_fb->default_ds;
}

static void targetsChanged(GLMContext ctx)
{
    STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE;
    if (fbm(ctx) && fbm(ctx)->targets_changed)
        fbm(ctx)->targets_changed(ctx);
}

/* an attachment change on fbo affects rendering if it is the draw framebuffer */
/* (DIRTY_FBO_BINDING is deliberately not used: attachments are resolved by
 * MGLRenderer+Framebuffer.m when the render pass is built) */
static void attachmentsChanged(GLMContext ctx, Framebuffer *fbo)
{
    if (fbo == ctx->state.framebuffer)
        targetsChanged(ctx);
}

#pragma mark attachments

FBOAttachment *mgl_fb_attachment(Framebuffer *fbo, GLenum attachment)
{
    switch(attachment)
    {
        case GL_DEPTH_ATTACHMENT:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            return &fbo->depth;
        case GL_STENCIL_ATTACHMENT:
            return &fbo->stencil;
    }
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)
        return &fbo->color_attachments[attachment - GL_COLOR_ATTACHMENT0];
    return NULL;
}

/* legacy names used elsewhere in the tree */
FBOAttachment *getFBOAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment)
{
    (void)ctx;
    return mgl_fb_attachment(fbo, attachment);
}

bool isColorAttachment(GLMContext ctx, GLuint attachment)
{
    return attachment >= GL_COLOR_ATTACHMENT0 &&
           attachment < GL_COLOR_ATTACHMENT0 + STATE(max_color_attachments);
}

bool isCubeMapTarget(GLMContext ctx, GLuint textarget)
{
    (void)ctx;
    return textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z;
}

static int attIndex(Framebuffer *fbo, FBOAttachment *att)
{
    if (att == &fbo->depth) return MGL_FB_ATT_DEPTH;
    if (att == &fbo->stencil) return MGL_FB_ATT_STENCIL;
    return (int)(att - fbo->color_attachments);
}

/* the attached texture (for renderbuffers: the renderbuffer's storage texture).
 * Objects deleted after attachment read back as nothing attached. */
Texture *mgl_fb_attachment_texture(GLMContext ctx, FBOAttachment *att)
{
    if (!att || att->texture == 0)
        return NULL;

    if (att->textarget == GL_RENDERBUFFER)
    {
        Renderbuffer *rb = findRenderbuffer(ctx, att->texture);
        if (!rb || rb != att->buf.rbo)
            return NULL;
        return rb->tex;
    }

    Texture *tex = findTexture(ctx, att->texture);
    if (!tex || tex != att->buf.tex)
        return NULL;
    return tex;
}

static bool attachmentIsLive(GLMContext ctx, FBOAttachment *att)
{
    if (!att || att->texture == 0) return false;
    if (att->textarget == GL_RENDERBUFFER)
    {
        Renderbuffer *rb = findRenderbuffer(ctx, att->texture);
        return rb && rb == att->buf.rbo;
    }
    Texture *tex = findTexture(ctx, att->texture);
    return tex && tex == att->buf.tex;
}

static GLenum attachmentInternalFormat(GLMContext ctx, FBOAttachment *att)
{
    if (att->textarget == GL_RENDERBUFFER)
        return RBEX(att->buf.rbo)->internalformat;
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    return tex ? tex->internalformat : GL_NONE;
}

static bool isLayeredTarget(GLenum target)
{
    switch(target)
    {
        case GL_TEXTURE_3D:
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
    }
    return false;
}

static GLuint textureLayerCount(Texture *tex, GLuint level)
{
    switch(tex->target)
    {
        case GL_TEXTURE_3D:
        {
            GLuint d = tex->depth >> level;
            return d ? d : 1;
        }
        case GL_TEXTURE_1D_ARRAY:
            return tex->height ? tex->height : 1;
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return tex->depth ? tex->depth : 1;
        case GL_TEXTURE_CUBE_MAP:
            return 6;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            return tex->depth ? tex->depth : 6;
    }
    return 1;
}

static GLuint faceForAttachment(FBOAttachment *att, Texture *tex)
{
    if (tex->target == GL_TEXTURE_CUBE_MAP && att->textarget >= GL_TEXTURE_CUBE_MAP_POSITIVE_X &&
        att->textarget <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
        return att->textarget - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    return 0;
}

static TextureLevel *attachmentLevel(Texture *tex, FBOAttachment *att)
{
    GLuint face = faceForAttachment(att, tex);
    if (tex->target == GL_RENDERBUFFER)
        return tex->faces[0].levels;
    if (!tex->faces[face].levels || att->level >= tex->mipmap_levels)
        return NULL;
    TextureLevel *lvl = &tex->faces[face].levels[att->level];
    if (lvl->width == 0 && tex->width && att->level == 0)
        return NULL;
    return lvl;
}

bool mgl_fb_attachment_size(GLMContext ctx, FBOAttachment *att, GLuint *w, GLuint *h, GLuint *layers)
{
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    if (!tex) return false;

    if (att->textarget == GL_RENDERBUFFER)
    {
        MGLRenderbufferEx *rb = RBEX(att->buf.rbo);
        if (w) *w = rb->width;
        if (h) *h = rb->height;
        if (layers) *layers = 1;
        return rb->width > 0 && rb->height > 0;
    }

    TextureLevel *lvl = attachmentLevel(tex, att);
    if (!lvl || lvl->width == 0)
        return false;
    GLuint lw = lvl->width, lh = lvl->height ? lvl->height : 1;
    if (tex->target == GL_TEXTURE_1D_ARRAY) lh = 1;
    if (w) *w = lw;
    if (h) *h = lh;
    if (layers) *layers = textureLayerCount(tex, att->level);
    return true;
}

void mgl_fb_attachment_slice(GLMContext ctx, Framebuffer *fbo, FBOAttachment *att, GLuint *level, GLuint *slice, GLuint *depth_plane, GLuint *layer_count)
{
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    *level = 0; *slice = 0; *depth_plane = 0; *layer_count = 1;
    if (!tex || att->textarget == GL_RENDERBUFFER)
        return;

    bool layered = fbo ? FBEX(fbo)->layered[attIndex(fbo, att)] : false;
    *level = att->level;

    if (layered)
    {
        *layer_count = textureLayerCount(tex, att->level);
        return;
    }

    switch(tex->target)
    {
        case GL_TEXTURE_3D:
            *depth_plane = att->layer;
            break;
        case GL_TEXTURE_CUBE_MAP:
            if (isCubeMapTarget(ctx, att->textarget))
                *slice = att->textarget - GL_TEXTURE_CUBE_MAP_POSITIVE_X;
            else
                *slice = att->layer;
            break;
        case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            *slice = att->layer;
            break;
    }
}

GLsizei mgl_fb_attachment_samples(GLMContext ctx, FBOAttachment *att)
{
    if (!attachmentIsLive(ctx, att)) return 0;
    if (att->textarget == GL_RENDERBUFFER)
        return RBEX(att->buf.rbo)->samples;
    Texture *tex = att->buf.tex;
    if (tex->target == GL_TEXTURE_2D_MULTISAMPLE || tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
    {
        extern GLuint mgl_fb_texture_sample_count(GLMContext ctx, Texture *tex);
        GLuint s = mgl_fb_texture_sample_count(ctx, tex);
        return s > 1 ? (GLsizei)s : 0;
    }
    return 0;
}

static void detachAttachment(FBOAttachment *att)
{
    memset(att, 0, sizeof(*att));
}

#pragma mark completeness

static GLenum attachmentCompleteness(GLMContext ctx, Framebuffer *fbo, FBOAttachment *att, int kind)
{
    /* kind: 0 color, 1 depth, 2 stencil */
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    if (!tex)
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;

    GLuint w = 0, h = 0, layers = 1;
    if (!mgl_fb_attachment_size(ctx, att, &w, &h, &layers) || w == 0 || h == 0)
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;

    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentInternalFormat(ctx, att));
    if (!fi)
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;

    if ((kind == 0 && !fi->color_renderable) ||
        (kind == 1 && !fi->depth_renderable) ||
        (kind == 2 && !fi->stencil_renderable))
        return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;

    if (att->textarget != GL_RENDERBUFFER)
    {
        if (tex->target == GL_TEXTURE_BUFFER)
            return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        if (!FBEX(fbo)->layered[attIndex(fbo, att)] && isLayeredTarget(tex->target) && tex->target != GL_TEXTURE_CUBE_MAP)
        {
            if (att->layer >= layers)
                return GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT;
        }
    }

    return GL_FRAMEBUFFER_COMPLETE;
}

GLenum mgl_fb_status(GLMContext ctx, Framebuffer *fbo)
{
    if (fbo == NULL)
        return GL_FRAMEBUFFER_COMPLETE;

    MGLFramebufferEx *fx = FBEX(fbo);
    int count = 0;
    GLint samples = -1;
    int layered = -1;
    GLenum layer_target = GL_NONE;
    FBOAttachment *all[MGL_FB_NUM_ATT];
    int kinds[MGL_FB_NUM_ATT];

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) { all[i] = &fbo->color_attachments[i]; kinds[i] = 0; }
    all[MGL_FB_ATT_DEPTH] = &fbo->depth;     kinds[MGL_FB_ATT_DEPTH] = 1;
    all[MGL_FB_ATT_STENCIL] = &fbo->stencil; kinds[MGL_FB_ATT_STENCIL] = 2;

    for (int i = 0; i < MGL_FB_NUM_ATT; i++)
    {
        FBOAttachment *att = all[i];
        if (att->texture == 0)
            continue;

        GLenum st = attachmentCompleteness(ctx, fbo, att, kinds[i]);
        if (st != GL_FRAMEBUFFER_COMPLETE)
            return st;
        count++;

        GLint s = mgl_fb_attachment_samples(ctx, att);
        if (samples < 0) samples = s;
        else if (samples != s) return GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE;

        int l = fx->layered[i] ? 1 : 0;
        if (layered < 0) layered = l;
        else if (layered != l) return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        if (l)
        {
            GLenum t = att->buf.tex->target;
            if (layer_target == GL_NONE) layer_target = t;
            else if (layer_target != t) return GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS;
        }
    }

    if (count == 0)
    {
        if (fbo->default_width == 0 || fbo->default_height == 0)
            return GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT;
        return GL_FRAMEBUFFER_COMPLETE;
    }

    /* Metal: a combined depth/stencil image must serve both attachment points */
    if (fbo->depth.texture && fbo->stencil.texture)
    {
        bool same = fbo->depth.textarget == fbo->stencil.textarget && fbo->depth.texture == fbo->stencil.texture;
        if (!same)
        {
            const MGLFBFormatInfo *d = mgl_fb_format_info(attachmentInternalFormat(ctx, &fbo->depth));
            const MGLFBFormatInfo *s = mgl_fb_format_info(attachmentInternalFormat(ctx, &fbo->stencil));
            if ((d && d->base == GL_DEPTH_STENCIL) || (s && s->base == GL_DEPTH_STENCIL))
                return GL_FRAMEBUFFER_UNSUPPORTED;
        }
    }

    return GL_FRAMEBUFFER_COMPLETE;
}

#pragma mark Framebuffer object calls

GLboolean mglIsFramebuffer(GLMContext ctx, GLuint framebuffer)
{
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);
    return (fbo && FBEX(fbo)->created) ? GL_TRUE : GL_FALSE;
}

static void genFramebuffers(GLMContext ctx, GLsizei n, GLuint *framebuffers, bool create)
{
    if (n < 0) FB_ERROR(GL_INVALID_VALUE);
    if (n && !framebuffers) return;

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name;
        do { name = getNewName(&STATE(framebuffer_table)); } while (name == 0 || findFrameBuffer(ctx, name));

        MGLFramebufferEx *fb = newFramebufferEx(name);
        if (!fb) FB_ERROR(GL_OUT_OF_MEMORY);
        fb->created = create;
        insertHashElement(&STATE(framebuffer_table), name, fb);
        framebuffers[i] = name;
    }
}

void mglGenFramebuffers(GLMContext ctx, GLsizei n, GLuint *framebuffers)
{
    genFramebuffers(ctx, n, framebuffers, false);
}

void mglCreateFramebuffers(GLMContext ctx, GLsizei n, GLuint *framebuffers)
{
    genFramebuffers(ctx, n, framebuffers, true);
}

void mglBindFramebuffer(GLMContext ctx, GLenum target, GLuint framebuffer)
{
    Framebuffer *ptr = NULL;

    if (!validFBTarget(target)) FB_ERROR(GL_INVALID_ENUM);

    if (framebuffer)
    {
        ptr = findFrameBuffer(ctx, framebuffer);
        if (!ptr) FB_ERROR(GL_INVALID_OPERATION);   /* not a name returned by Gen/Create */
        FBEX(ptr)->created = GL_TRUE;
    }

    if (target == GL_DRAW_FRAMEBUFFER || target == GL_FRAMEBUFFER)
    {
        if (ctx->state.framebuffer != ptr)
        {
            ctx->state.framebuffer = ptr;
            targetsChanged(ctx);
        }
    }
    if (target == GL_READ_FRAMEBUFFER || target == GL_FRAMEBUFFER)
    {
        ctx->state.readbuffer = ptr;
    }

    mgl_fb_sync_bindings(ctx);
    STATE(dirty_bits) |= DIRTY_FBO;
}

void mglDeleteFramebuffers(GLMContext ctx, GLsizei n, const GLuint *framebuffers)
{
    if (n < 0) FB_ERROR(GL_INVALID_VALUE);
    if (!framebuffers) return;

    for (GLsizei i = 0; i < n; i++)
    {
        Framebuffer *fbo = findFrameBuffer(ctx, framebuffers[i]);
        if (!fbo)
            continue;

        if (ctx->state.framebuffer == fbo)
        {
            ctx->state.framebuffer = NULL;
            targetsChanged(ctx);
        }
        if (ctx->state.readbuffer == fbo)
            ctx->state.readbuffer = NULL;

        deleteHashElement(&STATE(framebuffer_table), framebuffers[i]);
        free(fbo);
    }

    mgl_fb_sync_bindings(ctx);
    STATE(dirty_bits) |= DIRTY_FBO;
}

GLenum mglCheckFramebufferStatus(GLMContext ctx, GLenum target)
{
    if (!validFBTarget(target)) FB_ERROR_V(GL_INVALID_ENUM, 0);
    return mgl_fb_status(ctx, currentFBOForType(ctx, target));
}

GLenum mglCheckNamedFramebufferStatus(GLMContext ctx, GLuint framebuffer, GLenum target)
{
    if (!validFBTarget(target)) FB_ERROR_V(GL_INVALID_ENUM, 0);
    if (framebuffer == 0)
        return GL_FRAMEBUFFER_COMPLETE;
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);
    if (!fbo) FB_ERROR_V(GL_INVALID_OPERATION, 0);
    return mgl_fb_status(ctx, fbo);
}

#pragma mark Renderbuffer calls

GLboolean mglIsRenderbuffer(GLMContext ctx, GLuint renderbuffer)
{
    Renderbuffer *rb = findRenderbuffer(ctx, renderbuffer);
    return (rb && RBEX(rb)->created) ? GL_TRUE : GL_FALSE;
}

static void genRenderbuffers(GLMContext ctx, GLsizei n, GLuint *renderbuffers, bool create)
{
    if (n < 0) FB_ERROR(GL_INVALID_VALUE);
    if (n && !renderbuffers) return;

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name;
        do { name = getNewName(&STATE(renderbuffer_table)); } while (name == 0 || findRenderbuffer(ctx, name));

        MGLRenderbufferEx *rb = newRenderbufferEx(name);
        if (!rb) FB_ERROR(GL_OUT_OF_MEMORY);
        rb->created = create;
        insertHashElement(&STATE(renderbuffer_table), name, rb);
        renderbuffers[i] = name;
    }
}

void mglGenRenderbuffers(GLMContext ctx, GLsizei n, GLuint *renderbuffers)
{
    genRenderbuffers(ctx, n, renderbuffers, false);
}

void mglCreateRenderbuffers(GLMContext ctx, GLsizei n, GLuint *renderbuffers)
{
    genRenderbuffers(ctx, n, renderbuffers, true);
}

void mglBindRenderbuffer(GLMContext ctx, GLenum target, GLuint renderbuffer)
{
    Renderbuffer *ptr = NULL;

    if (target != GL_RENDERBUFFER) FB_ERROR(GL_INVALID_ENUM);

    if (renderbuffer)
    {
        ptr = findRenderbuffer(ctx, renderbuffer);
        if (!ptr) FB_ERROR(GL_INVALID_OPERATION);
        RBEX(ptr)->created = GL_TRUE;
    }

    ctx->state.renderbuffer = ptr;
    STATE_VAR(renderbuffer_binding) = renderbuffer;
}

static void releaseRenderbufferStorage(GLMContext ctx, MGLRenderbufferEx *rb)
{
    Texture *tex = rb->base.tex;
    if (!tex) return;

    if (fbm(ctx) && fbm(ctx)->release_texture)
        fbm(ctx)->release_texture(ctx, tex);

    for (int i = 0; i < 6; i++)
        if (tex->faces[i].levels) free(tex->faces[i].levels);
    free(tex);
    rb->base.tex = NULL;
}

static void detachRenderbufferFrom(GLMContext ctx, Framebuffer *fbo, Renderbuffer *rb)
{
    if (!fbo) return;
    bool changed = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        FBOAttachment *att = &fbo->color_attachments[i];
        if (att->textarget == GL_RENDERBUFFER && att->buf.rbo == rb && att->texture)
        {
            detachAttachment(att);
            fbo->color_attachment_bitfield &= ~(1u << i);
            changed = true;
        }
    }
    if (fbo->depth.textarget == GL_RENDERBUFFER && fbo->depth.buf.rbo == rb && fbo->depth.texture) { detachAttachment(&fbo->depth); changed = true; }
    if (fbo->stencil.textarget == GL_RENDERBUFFER && fbo->stencil.buf.rbo == rb && fbo->stencil.texture) { detachAttachment(&fbo->stencil); changed = true; }
    if (changed) attachmentsChanged(ctx, fbo);
}

void mglDeleteRenderbuffers(GLMContext ctx, GLsizei n, const GLuint *renderbuffers)
{
    if (n < 0) FB_ERROR(GL_INVALID_VALUE);
    if (!renderbuffers) return;

    for (GLsizei i = 0; i < n; i++)
    {
        Renderbuffer *rb = findRenderbuffer(ctx, renderbuffers[i]);
        if (!rb)
            continue;

        if (ctx->state.renderbuffer == rb)
        {
            ctx->state.renderbuffer = NULL;
            STATE_VAR(renderbuffer_binding) = 0;
        }

        /* detached from the currently bound framebuffers (GL 4.6 §9.2.7) */
        detachRenderbufferFrom(ctx, ctx->state.framebuffer, rb);
        if (ctx->state.readbuffer != ctx->state.framebuffer)
            detachRenderbufferFrom(ctx, ctx->state.readbuffer, rb);

        /* other framebuffers drop it via name revalidation; make sure the
         * current pass no longer references the storage */
        targetsChanged(ctx);

        releaseRenderbufferStorage(ctx, RBEX(rb));
        deleteHashElement(&STATE(renderbuffer_table), renderbuffers[i]);
        free(rb);
    }
}

static GLint maxSamples(GLMContext ctx)
{
    if (fbm(ctx) && fbm(ctx)->max_samples > 0)
        return fbm(ctx)->max_samples;
    return 4;
}

static void renderbufferStorage(GLMContext ctx, Renderbuffer *rbo, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    MGLRenderbufferEx *rb = RBEX(rbo);
    const MGLFBFormatInfo *fi = mgl_fb_format_info(internalformat);

    if (!fi || !(fi->color_renderable || fi->depth_renderable || fi->stencil_renderable))
        FB_ERROR(GL_INVALID_ENUM);

    if (width < 0 || height < 0 || samples < 0)
        FB_ERROR(GL_INVALID_VALUE);

    GLuint max_rb = STATE_VAR(max_renderbuffer_size) ? STATE_VAR(max_renderbuffer_size) : 16384;
    if ((GLuint)width > max_rb || (GLuint)height > max_rb)
        FB_ERROR(GL_INVALID_VALUE);

    if (samples > maxSamples(ctx))
        FB_ERROR(GL_INVALID_OPERATION);

    releaseRenderbufferStorage(ctx, rb);

    GLuint actual = 0;
    if (samples > 0)
    {
        actual = (fbm(ctx) && fbm(ctx)->round_samples) ? fbm(ctx)->round_samples(ctx, samples) : (GLuint)samples;
        if (actual < 2) actual = 0;
    }

    rb->internalformat = internalformat;
    rb->width = width;
    rb->height = height;
    rb->samples = actual;

    if (width > 0 && height > 0)
    {
        Texture *tex = newTexObj(ctx, GL_RENDERBUFFER);
        if (!tex) FB_ERROR(GL_OUT_OF_MEMORY);

        tex->faces[0].levels = (TextureLevel *)calloc(1, sizeof(TextureLevel));
        if (!tex->faces[0].levels) { free(tex); FB_ERROR(GL_OUT_OF_MEMORY); }
        tex->faces[0].levels[0].width = width;
        tex->faces[0].levels[0].height = height;
        tex->faces[0].levels[0].depth = 1;
        tex->faces[0].levels[0].complete = GL_TRUE;
        tex->internalformat = internalformat;
        tex->width = width;
        tex->height = height;
        tex->depth = 1;
        tex->num_levels = 1;
        tex->mipmap_levels = 1;
        tex->complete = GL_TRUE;
        tex->access = GL_READ_WRITE;
        tex->is_render_target = GL_TRUE;
        tex->mtl_requires_private_storage = GL_TRUE;
        tex->dirty_bits = 0;
        rb->base.tex = tex;

        if (!fbm(ctx) || !fbm(ctx)->renderbuffer_storage || !fbm(ctx)->renderbuffer_storage(ctx, rb))
        {
            releaseRenderbufferStorage(ctx, rb);
            rb->width = rb->height = 0;
            FB_ERROR(GL_OUT_OF_MEMORY);
        }
    }

    /* every framebuffer referencing this renderbuffer must rebuild its pass */
    targetsChanged(ctx);
}

void mglRenderbufferStorage(GLMContext ctx, GLenum target, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (target != GL_RENDERBUFFER) FB_ERROR(GL_INVALID_ENUM);
    if (!ctx->state.renderbuffer) FB_ERROR(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, ctx->state.renderbuffer, 0, internalformat, width, height);
}

void mglRenderbufferStorageMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    if (target != GL_RENDERBUFFER) FB_ERROR(GL_INVALID_ENUM);
    if (!ctx->state.renderbuffer) FB_ERROR(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, ctx->state.renderbuffer, samples, internalformat, width, height);
}

void mglNamedRenderbufferStorage(GLMContext ctx, GLuint renderbuffer, GLenum internalformat, GLsizei width, GLsizei height)
{
    Renderbuffer *rb = findRenderbuffer(ctx, renderbuffer);
    if (!rb) FB_ERROR(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, rb, 0, internalformat, width, height);
}

void mglNamedRenderbufferStorageMultisample(GLMContext ctx, GLuint renderbuffer, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height)
{
    Renderbuffer *rb = findRenderbuffer(ctx, renderbuffer);
    if (!rb) FB_ERROR(GL_INVALID_OPERATION);
    renderbufferStorage(ctx, rb, samples, internalformat, width, height);
}

static void getRenderbufferParameter(GLMContext ctx, Renderbuffer *rbo, GLenum pname, GLint *params)
{
    MGLRenderbufferEx *rb = RBEX(rbo);
    const MGLFBFormatInfo *fi = (rb->width > 0 || rb->base.tex) ? mgl_fb_format_info(rb->internalformat) : NULL;
    bool has = rb->base.tex != NULL;
    GLint v;

    switch(pname)
    {
        case GL_RENDERBUFFER_WIDTH:           v = rb->width; break;
        case GL_RENDERBUFFER_HEIGHT:          v = rb->height; break;
        case GL_RENDERBUFFER_INTERNAL_FORMAT: v = rb->internalformat; break;
        case GL_RENDERBUFFER_SAMPLES:         v = rb->samples; break;
        case GL_RENDERBUFFER_RED_SIZE:        v = (has && fi) ? fi->r : 0; break;
        case GL_RENDERBUFFER_GREEN_SIZE:      v = (has && fi) ? fi->g : 0; break;
        case GL_RENDERBUFFER_BLUE_SIZE:       v = (has && fi) ? fi->b : 0; break;
        case GL_RENDERBUFFER_ALPHA_SIZE:      v = (has && fi) ? fi->a : 0; break;
        case GL_RENDERBUFFER_DEPTH_SIZE:      v = (has && fi) ? fi->d : 0; break;
        case GL_RENDERBUFFER_STENCIL_SIZE:    v = (has && fi) ? fi->s : 0; break;
        default:
            FB_ERROR(GL_INVALID_ENUM);
    }
    if (params) *params = v;
}

void mglGetRenderbufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (target != GL_RENDERBUFFER) FB_ERROR(GL_INVALID_ENUM);
    if (!ctx->state.renderbuffer) FB_ERROR(GL_INVALID_OPERATION);
    getRenderbufferParameter(ctx, ctx->state.renderbuffer, pname, params);
}

void mglGetNamedRenderbufferParameteriv(GLMContext ctx, GLuint renderbuffer, GLenum pname, GLint *params)
{
    Renderbuffer *rb = findRenderbuffer(ctx, renderbuffer);
    if (!rb) FB_ERROR(GL_INVALID_OPERATION);
    getRenderbufferParameter(ctx, rb, pname, params);
}

#pragma mark Framebuffer attachment calls

/* returns false (and records the error) for an invalid attachment enum */
static bool checkAttachmentEnum(GLMContext ctx, GLenum attachment)
{
    switch(attachment)
    {
        case GL_DEPTH_ATTACHMENT:
        case GL_STENCIL_ATTACHMENT:
        case GL_DEPTH_STENCIL_ATTACHMENT:
            return true;
    }
    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment <= GL_COLOR_ATTACHMENT31)
    {
        if (attachment - GL_COLOR_ATTACHMENT0 >= STATE(max_color_attachments) ||
            attachment - GL_COLOR_ATTACHMENT0 >= MAX_COLOR_ATTACHMENTS)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return false;
        }
        return true;
    }
    ERROR_RETURN(GL_INVALID_ENUM);
    return false;
}

static void setAttachment(GLMContext ctx, Framebuffer *fbo, GLenum attachment, GLenum textarget, GLuint name, void *obj, GLint level, GLint layer, bool layered)
{
    FBOAttachment *att = mgl_fb_attachment(fbo, attachment);
    MGLFramebufferEx *fx = FBEX(fbo);

    if (name == 0)
    {
        detachAttachment(att);
    }
    else
    {
        memset(att, 0, sizeof(*att));
        att->texture = name;
        att->textarget = textarget;
        att->level = level;
        att->layer = layer;
        if (textarget == GL_RENDERBUFFER) att->buf.rbo = (Renderbuffer *)obj;
        else att->buf.tex = (Texture *)obj;
    }
    fx->layered[attIndex(fbo, att)] = (name && layered) ? GL_TRUE : GL_FALSE;

    if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)
    {
        GLuint idx = attachment - GL_COLOR_ATTACHMENT0;
        if (name) fbo->color_attachment_bitfield |= (1u << idx);
        else fbo->color_attachment_bitfield &= ~(1u << idx);
    }

    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT)
    {
        fbo->stencil = fbo->depth;
        fx->layered[MGL_FB_ATT_STENCIL] = fx->layered[MGL_FB_ATT_DEPTH];
    }

    fx->ds.invalid_mask &= ~(attachment == GL_DEPTH_STENCIL_ATTACHMENT ? (MGL_FB_DEPTH_BIT | MGL_FB_STENCIL_BIT) :
                             attachment == GL_DEPTH_ATTACHMENT ? MGL_FB_DEPTH_BIT :
                             attachment == GL_STENCIL_ATTACHMENT ? MGL_FB_STENCIL_BIT :
                             (1u << (attachment - GL_COLOR_ATTACHMENT0)));

    attachmentsChanged(ctx, fbo);
}

/* kind: 0 FramebufferTexture (layered), 1 = 1D, 2 = 2D, 3 = 3D, 4 = TextureLayer */
static void framebufferTextureCommon(GLMContext ctx, Framebuffer *fbo, GLenum attachment, int kind, GLenum textarget, GLuint texture, GLint level, GLint layer)
{
    if (!checkAttachmentEnum(ctx, attachment))
        return;

    /* textarget is validated even when detaching */
    if ((kind == 1 && textarget != GL_TEXTURE_1D) || (kind == 3 && textarget != GL_TEXTURE_3D) ||
        (kind == 2 && !(textarget == GL_TEXTURE_2D || textarget == GL_TEXTURE_RECTANGLE ||
                        textarget == GL_TEXTURE_2D_MULTISAMPLE || isCubeMapTarget(ctx, textarget))))
        FB_ERROR(GL_INVALID_ENUM);

    if (texture == 0)
    {
        setAttachment(ctx, fbo, attachment, GL_NONE, 0, NULL, 0, 0, false);
        return;
    }

    Texture *tex = findTexture(ctx, texture);
    if (!tex)
        FB_ERROR(kind == 0 || kind == 4 ? GL_INVALID_VALUE : GL_INVALID_OPERATION);

    GLuint max2d = STATE_VAR(max_texture_size) ? STATE_VAR(max_texture_size) : 16384;
    GLuint max3d = STATE_VAR(max_3d_texture_size) ? STATE_VAR(max_3d_texture_size) : 2048;
    GLuint maxcube = STATE_VAR(max_cube_map_texture_size) ? STATE_VAR(max_cube_map_texture_size) : 16384;
    GLuint maxlayers = STATE_VAR(max_array_texture_layers) ? STATE_VAR(max_array_texture_layers) : 2048;
    bool layered = false;
    GLenum stored_target = tex->target;

    switch(kind)
    {
        case 0: /* FramebufferTexture */
            if (tex->target == GL_TEXTURE_BUFFER) FB_ERROR(GL_INVALID_OPERATION);
            layered = isLayeredTarget(tex->target);
            break;

        case 1:
            if (textarget != GL_TEXTURE_1D) FB_ERROR(GL_INVALID_ENUM);
            if (tex->target != GL_TEXTURE_1D) FB_ERROR(GL_INVALID_OPERATION);
            break;

        case 2:
            if (!(textarget == GL_TEXTURE_2D || textarget == GL_TEXTURE_RECTANGLE ||
                  textarget == GL_TEXTURE_2D_MULTISAMPLE || isCubeMapTarget(ctx, textarget)))
                FB_ERROR(GL_INVALID_ENUM);
            if (isCubeMapTarget(ctx, textarget) ? (tex->target != GL_TEXTURE_CUBE_MAP) : (tex->target != textarget))
                FB_ERROR(GL_INVALID_OPERATION);
            stored_target = textarget;
            break;

        case 3:
            if (textarget != GL_TEXTURE_3D) FB_ERROR(GL_INVALID_ENUM);
            if (tex->target != GL_TEXTURE_3D) FB_ERROR(GL_INVALID_OPERATION);
            if (layer < 0 || (GLuint)layer >= max3d) FB_ERROR(GL_INVALID_VALUE);
            break;

        case 4: /* TextureLayer */
            switch(tex->target)
            {
                case GL_TEXTURE_3D:
                    if (layer < 0 || (GLuint)layer >= max3d) FB_ERROR(GL_INVALID_VALUE);
                    break;
                case GL_TEXTURE_1D_ARRAY:
                case GL_TEXTURE_2D_ARRAY:
                case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
                    if (layer < 0 || (GLuint)layer >= maxlayers) FB_ERROR(GL_INVALID_VALUE);
                    break;
                case GL_TEXTURE_CUBE_MAP:
                    if (layer < 0 || layer >= 6) FB_ERROR(GL_INVALID_VALUE);
                    break;
                case GL_TEXTURE_CUBE_MAP_ARRAY:
                    if (layer < 0 || (GLuint)layer >= maxlayers * 6) FB_ERROR(GL_INVALID_VALUE);
                    break;
                default:
                    FB_ERROR(GL_INVALID_OPERATION);
            }
            break;
    }

    if (level < 0) FB_ERROR(GL_INVALID_VALUE);
    /* GL 4.6 §9.2 (glFramebufferTexture): INVALID_VALUE when <level> is greater than the
     * texture's maximum mip level (tex->num_levels = highest allocated level + 1, exact for
     * texStorage textures; KHR-GL46.geometry_shader.layered_fbo.fb_texture_invalid_level_number
     * attaches level 2 of a 2-level texture).  Skipped when no level is allocated, where the
     * attachment is only framebuffer-incomplete. */
    if (tex->num_levels && (GLuint)level >= tex->num_levels) FB_ERROR(GL_INVALID_VALUE);

    GLenum t = (kind == 2) ? textarget : tex->target;
    if ((t == GL_TEXTURE_RECTANGLE || t == GL_TEXTURE_2D_MULTISAMPLE || t == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) && level != 0)
        FB_ERROR(GL_INVALID_VALUE);

    GLuint maxdim = (t == GL_TEXTURE_3D) ? max3d : (isCubeMapTarget(ctx, t) || t == GL_TEXTURE_CUBE_MAP || t == GL_TEXTURE_CUBE_MAP_ARRAY) ? maxcube : max2d;
    if ((GLuint)level > ilog2(maxdim))
        FB_ERROR(GL_INVALID_VALUE);

    setAttachment(ctx, fbo, attachment, stored_target, texture, tex, level, (kind == 3 || kind == 4) ? layer : 0, layered);
}

static Framebuffer *boundFBOForTarget(GLMContext ctx, GLenum target, bool *ok)
{
    *ok = false;
    if (!validFBTarget(target)) { ERROR_RETURN(GL_INVALID_ENUM); return NULL; }
    Framebuffer *fbo = currentFBOForType(ctx, target);
    if (!fbo) { ERROR_RETURN(GL_INVALID_OPERATION); return NULL; }
    *ok = true;
    return fbo;
}

static Framebuffer *namedFBO(GLMContext ctx, GLuint framebuffer, bool *ok)
{
    *ok = false;
    Framebuffer *fbo = findFrameBuffer(ctx, framebuffer);
    if (!fbo) { ERROR_RETURN(GL_INVALID_OPERATION); return NULL; }
    FBEX(fbo)->created = GL_TRUE;
    *ok = true;
    return fbo;
}

void mglFramebufferTexture(GLMContext ctx, GLenum target, GLenum attachment, GLuint texture, GLint level)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 0, GL_NONE, texture, level, 0);
}

void mglFramebufferTexture1D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 1, textarget, texture, level, 0);
}

void mglFramebufferTexture2D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 2, textarget, texture, level, 0);
}

void mglFramebufferTexture3D(GLMContext ctx, GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level, GLint zoffset)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 3, textarget, texture, level, zoffset);
}

void mglFramebufferTextureLayer(GLMContext ctx, GLenum target, GLenum attachment, GLuint texture, GLint level, GLint layer)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 4, GL_NONE, texture, level, layer);
}

void mglNamedFramebufferTexture(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level)
{
    bool ok; Framebuffer *fbo = namedFBO(ctx, framebuffer, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 0, GL_NONE, texture, level, 0);
}

void mglNamedFramebufferTextureLayer(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLuint texture, GLint level, GLint layer)
{
    bool ok; Framebuffer *fbo = namedFBO(ctx, framebuffer, &ok);
    if (ok) framebufferTextureCommon(ctx, fbo, attachment, 4, GL_NONE, texture, level, layer);
}

static void framebufferRenderbufferCommon(GLMContext ctx, Framebuffer *fbo, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    if (!checkAttachmentEnum(ctx, attachment))
        return;
    if (renderbuffertarget != GL_RENDERBUFFER)
        FB_ERROR(GL_INVALID_ENUM);

    Renderbuffer *rb = NULL;
    if (renderbuffer)
    {
        rb = findRenderbuffer(ctx, renderbuffer);
        if (!rb) FB_ERROR(GL_INVALID_OPERATION);
    }

    setAttachment(ctx, fbo, attachment, renderbuffer ? GL_RENDERBUFFER : GL_NONE, renderbuffer, rb, 0, 0, false);
}

void mglFramebufferRenderbuffer(GLMContext ctx, GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferRenderbufferCommon(ctx, fbo, attachment, renderbuffertarget, renderbuffer);
}

void mglNamedFramebufferRenderbuffer(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer)
{
    bool ok; Framebuffer *fbo = namedFBO(ctx, framebuffer, &ok);
    if (ok) framebufferRenderbufferCommon(ctx, fbo, attachment, renderbuffertarget, renderbuffer);
}

#pragma mark attachment queries

static void defaultFramebufferAttachmentParameter(GLMContext ctx, GLenum attachment, GLenum pname, GLint *params)
{
    bool exists;
    int kind; /* 0 color 1 depth 2 stencil */

    switch(attachment)
    {
        case GL_FRONT_LEFT:
        case GL_BACK_LEFT:
            exists = true; kind = 0; break;
        case GL_FRONT_RIGHT:
        case GL_BACK_RIGHT:
            exists = false; kind = 0; break;
        case GL_DEPTH:
            exists = ctx->depth_format.format != 0; kind = 1; break;
        case GL_STENCIL:
            exists = ctx->stencil_format.format != 0; kind = 2; break;
        default:
            FB_ERROR(GL_INVALID_ENUM);
    }

    if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)
    {
        *params = exists ? GL_FRAMEBUFFER_DEFAULT : GL_NONE;
        return;
    }
    if (!exists)
    {
        if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) { *params = 0; return; }
        FB_ERROR(GL_INVALID_OPERATION);
    }

    bool combined = ctx->depth_format.format && ctx->stencil_format.format;
    switch(pname)
    {
        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:
        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:
            *params = (kind == 0) ? 8 : 0; break;
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:
            *params = (kind == 1 || (kind == 2 && combined)) ? 32 : 0; break;
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE:
            *params = (kind == 2 || (kind == 1 && combined)) ? 8 : 0; break;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            *params = (kind == 0) ? GL_UNSIGNED_NORMALIZED : (kind == 1) ? GL_FLOAT : GL_UNSIGNED_INT; break;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            *params = GL_LINEAR; break;
        default:
            FB_ERROR(GL_INVALID_ENUM);
    }
}

static void getFramebufferAttachmentParameter(GLMContext ctx, Framebuffer *fbo, GLenum attachment, GLenum pname, GLint *params)
{
    GLint dummy;
    if (!params) params = &dummy;

    if (fbo == NULL)
    {
        defaultFramebufferAttachmentParameter(ctx, attachment, pname, params);
        return;
    }

    if (!checkAttachmentEnum(ctx, attachment))
        return;

    if (attachment == GL_DEPTH_STENCIL_ATTACHMENT)
    {
        if (fbo->depth.texture != fbo->stencil.texture || fbo->depth.textarget != fbo->stencil.textarget)
            FB_ERROR(GL_INVALID_OPERATION);
        if (pname == GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE)
            FB_ERROR(GL_INVALID_OPERATION);
    }

    FBOAttachment *att = mgl_fb_attachment(fbo, attachment);
    bool live = attachmentIsLive(ctx, att);
    GLenum type = !live ? GL_NONE : (att->textarget == GL_RENDERBUFFER ? GL_RENDERBUFFER : GL_TEXTURE);

    if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE) { *params = type; return; }
    if (pname == GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME) { *params = live ? (GLint)att->texture : 0; return; }

    if (type == GL_NONE)
        FB_ERROR(GL_INVALID_OPERATION);

    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentInternalFormat(ctx, att));
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    bool defined = tex && mgl_fb_attachment_size(ctx, att, NULL, NULL, NULL);

    switch(pname)
    {
        case GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE:     *params = (fi && defined) ? fi->r : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE:   *params = (fi && defined) ? fi->g : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE:    *params = (fi && defined) ? fi->b : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE:   *params = (fi && defined) ? fi->a : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE:   *params = (fi && defined) ? fi->d : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE: *params = (fi && defined) ? fi->s : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE:
            if (!fi || !defined) { *params = GL_NONE; return; }
            if (attachment == GL_STENCIL_ATTACHMENT) { *params = GL_UNSIGNED_INT; return; }
            if (fi->base == GL_STENCIL_INDEX) { *params = GL_UNSIGNED_INT; return; }
            *params = fi->component_type; return;
        case GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING:
            *params = (fi && fi->srgb) ? GL_SRGB : GL_LINEAR; return;
    }

    if (type == GL_RENDERBUFFER)
        FB_ERROR(GL_INVALID_ENUM);

    switch(pname)
    {
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL:
            *params = att->level; return;
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE:
            *params = isCubeMapTarget(ctx, att->textarget) ? (GLint)att->textarget : 0; return;
        case GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER:
            *params = att->layer; return;
        case GL_FRAMEBUFFER_ATTACHMENT_LAYERED:
            *params = FBEX(fbo)->layered[attIndex(fbo, att)] ? GL_TRUE : GL_FALSE; return;
    }

    FB_ERROR(GL_INVALID_ENUM);
}

void mglGetFramebufferAttachmentParameteriv(GLMContext ctx, GLenum target, GLenum attachment, GLenum pname, GLint *params)
{
    if (!validFBTarget(target)) FB_ERROR(GL_INVALID_ENUM);
    getFramebufferAttachmentParameter(ctx, currentFBOForType(ctx, target), attachment, pname, params);
}

void mglGetNamedFramebufferAttachmentParameteriv(GLMContext ctx, GLuint framebuffer, GLenum attachment, GLenum pname, GLint *params)
{
    Framebuffer *fbo = NULL;
    if (framebuffer)
    {
        bool ok; fbo = namedFBO(ctx, framebuffer, &ok);
        if (!ok) return;
    }
    getFramebufferAttachmentParameter(ctx, fbo, attachment, pname, params);
}

#pragma mark framebuffer parameters

static void framebufferParameter(GLMContext ctx, Framebuffer *fbo, GLenum pname, GLint param)
{
    GLint maxw = STATE_VAR(max_framebuffer_width) ? (GLint)STATE_VAR(max_framebuffer_width) : 16384;
    GLint maxh = STATE_VAR(max_framebuffer_height) ? (GLint)STATE_VAR(max_framebuffer_height) : 16384;
    GLint maxl = STATE_VAR(max_framebuffer_layers) ? (GLint)STATE_VAR(max_framebuffer_layers) : 2048;
    GLint maxs = STATE_VAR(max_framebuffer_samples) ? (GLint)STATE_VAR(max_framebuffer_samples) : maxSamples(ctx);

    switch(pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
            if (param < 0 || param > maxw) FB_ERROR(GL_INVALID_VALUE);
            fbo->default_width = param; break;
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
            if (param < 0 || param > maxh) FB_ERROR(GL_INVALID_VALUE);
            fbo->default_height = param; break;
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
            if (param < 0 || param > maxl) FB_ERROR(GL_INVALID_VALUE);
            fbo->default_layers = param; break;
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
            if (param < 0 || param > maxs) FB_ERROR(GL_INVALID_VALUE);
            fbo->default_samples = param; break;
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            fbo->default_fixed_sample_locations = param ? GL_TRUE : GL_FALSE; break;
        default:
            FB_ERROR(GL_INVALID_ENUM);
    }
    attachmentsChanged(ctx, fbo);
}

void mglFramebufferParameteri(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    bool ok; Framebuffer *fbo = boundFBOForTarget(ctx, target, &ok);
    if (ok) framebufferParameter(ctx, fbo, pname, param);
}

void mglNamedFramebufferParameteri(GLMContext ctx, GLuint framebuffer, GLenum pname, GLint param)
{
    bool ok; Framebuffer *fbo = namedFBO(ctx, framebuffer, &ok);
    if (ok) framebufferParameter(ctx, fbo, pname, param);
}

/* samples of a framebuffer (0 = single sampled) */
GLint mgl_fb_samples(GLMContext ctx, Framebuffer *fbo)
{
    if (!fbo) return 0;
    for (int i = 0; i < MGL_FB_NUM_ATT; i++)
    {
        FBOAttachment *att = (i < MAX_COLOR_ATTACHMENTS) ? &fbo->color_attachments[i] : (i == MGL_FB_ATT_DEPTH ? &fbo->depth : &fbo->stencil);
        if (attachmentIsLive(ctx, att))
            return mgl_fb_attachment_samples(ctx, att);
    }
    return fbo->default_samples > 1 ? (GLint)(fbm(ctx) && fbm(ctx)->round_samples ? fbm(ctx)->round_samples(ctx, fbo->default_samples) : fbo->default_samples) : 0;
}

extern void mgl_fb_color_read_format(GLMContext ctx, Framebuffer *fbo, GLenum *format, GLenum *type);

static void getFramebufferParameter(GLMContext ctx, Framebuffer *fbo, GLenum pname, GLint *param)
{
    GLint v;
    switch(pname)
    {
        case GL_FRAMEBUFFER_DEFAULT_WIDTH:
        case GL_FRAMEBUFFER_DEFAULT_HEIGHT:
        case GL_FRAMEBUFFER_DEFAULT_LAYERS:
        case GL_FRAMEBUFFER_DEFAULT_SAMPLES:
        case GL_FRAMEBUFFER_DEFAULT_FIXED_SAMPLE_LOCATIONS:
            if (!fbo) FB_ERROR(GL_INVALID_OPERATION);
            v = pname == GL_FRAMEBUFFER_DEFAULT_WIDTH ? fbo->default_width :
                pname == GL_FRAMEBUFFER_DEFAULT_HEIGHT ? fbo->default_height :
                pname == GL_FRAMEBUFFER_DEFAULT_LAYERS ? fbo->default_layers :
                pname == GL_FRAMEBUFFER_DEFAULT_SAMPLES ? fbo->default_samples :
                fbo->default_fixed_sample_locations;
            break;
        case GL_DOUBLEBUFFER:
            v = fbo ? GL_FALSE : GL_TRUE; break;
        case GL_STEREO:
            v = GL_FALSE; break;
        case GL_SAMPLES:
            v = mgl_fb_samples(ctx, fbo); break;
        case GL_SAMPLE_BUFFERS:
            v = mgl_fb_samples(ctx, fbo) > 0 ? 1 : 0; break;
        case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
        case GL_IMPLEMENTATION_COLOR_READ_TYPE:
        {
            GLenum f, t;
            mgl_fb_color_read_format(ctx, fbo, &f, &t);
            v = (pname == GL_IMPLEMENTATION_COLOR_READ_FORMAT) ? (GLint)f : (GLint)t;
            break;
        }
        default:
            FB_ERROR(GL_INVALID_ENUM);
    }
    if (param) *param = v;
}

void mglGetFramebufferParameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    if (!validFBTarget(target)) FB_ERROR(GL_INVALID_ENUM);
    getFramebufferParameter(ctx, currentFBOForType(ctx, target), pname, params);
}

void mglGetNamedFramebufferParameteriv(GLMContext ctx, GLuint framebuffer, GLenum pname, GLint *param)
{
    Framebuffer *fbo = NULL;
    if (framebuffer)
    {
        bool ok; fbo = namedFBO(ctx, framebuffer, &ok);
        if (!ok) return;
    }
    getFramebufferParameter(ctx, fbo, pname, param);
}

#pragma mark invalidation

/* framebuffer size used for "whole framebuffer" sub-invalidation checks */
static void framebufferSize(GLMContext ctx, Framebuffer *fbo, GLuint *w, GLuint *h)
{
    *w = *h = 0;
    if (!fbo)
    {
        if (fbm(ctx) && fbm(ctx)->default_size) fbm(ctx)->default_size(ctx, w, h);
        return;
    }
    GLuint mw = UINT32_MAX, mh = UINT32_MAX;
    bool any = false;
    for (int i = 0; i < MGL_FB_NUM_ATT; i++)
    {
        FBOAttachment *att = (i < MAX_COLOR_ATTACHMENTS) ? &fbo->color_attachments[i] : (i == MGL_FB_ATT_DEPTH ? &fbo->depth : &fbo->stencil);
        GLuint aw, ah;
        if (mgl_fb_attachment_size(ctx, att, &aw, &ah, NULL))
        {
            any = true;
            if (aw < mw) mw = aw;
            if (ah < mh) mh = ah;
        }
    }
    if (any) { *w = mw; *h = mh; }
    else { *w = fbo->default_width; *h = fbo->default_height; }
}

static void invalidateCommon(GLMContext ctx, Framebuffer *fbo, GLsizei numAttachments, const GLenum *attachments, bool sub, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (numAttachments < 0) FB_ERROR(GL_INVALID_VALUE);
    if (sub && (width < 0 || height < 0)) FB_ERROR(GL_INVALID_VALUE);
    if (numAttachments && !attachments) return;

    GLuint mask = 0;
    for (GLsizei i = 0; i < numAttachments; i++)
    {
        GLenum a = attachments[i];
        if (fbo)
        {
            if (a == GL_DEPTH_ATTACHMENT) mask |= MGL_FB_DEPTH_BIT;
            else if (a == GL_STENCIL_ATTACHMENT) mask |= MGL_FB_STENCIL_BIT;
            else if (a == GL_DEPTH_STENCIL_ATTACHMENT) mask |= MGL_FB_DEPTH_BIT | MGL_FB_STENCIL_BIT;
            else if (a >= GL_COLOR_ATTACHMENT0 && a <= GL_COLOR_ATTACHMENT31)
            {
                if (a - GL_COLOR_ATTACHMENT0 >= STATE(max_color_attachments)) FB_ERROR(GL_INVALID_OPERATION);
                mask |= 1u << (a - GL_COLOR_ATTACHMENT0);
            }
            else FB_ERROR(GL_INVALID_ENUM);
        }
        else
        {
            if (a == GL_COLOR) mask |= 1u;
            else if (a == GL_DEPTH) mask |= MGL_FB_DEPTH_BIT;
            else if (a == GL_STENCIL) mask |= MGL_FB_STENCIL_BIT;
            else FB_ERROR(GL_INVALID_ENUM);
        }
    }

    if (sub)
    {
        /* invalidation is a hint: only discard when the region covers the whole framebuffer */
        GLuint fw, fh;
        framebufferSize(ctx, fbo, &fw, &fh);
        if (!(x <= 0 && y <= 0 && (GLint64)x + width >= (GLint64)fw && (GLint64)y + height >= (GLint64)fh))
            return;
    }

    if (!mask) return;
    mgl_fb_draw_state(ctx, fbo)->invalid_mask |= mask;
    if (fbo == ctx->state.framebuffer)
        targetsChanged(ctx);
}

void mglInvalidateFramebuffer(GLMContext ctx, GLenum target, GLsizei numAttachments, const GLenum *attachments)
{
    if (!validFBTarget(target)) FB_ERROR(GL_INVALID_ENUM);
    invalidateCommon(ctx, currentFBOForType(ctx, target), numAttachments, attachments, false, 0, 0, 0, 0);
}

void mglInvalidateSubFramebuffer(GLMContext ctx, GLenum target, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height)
{
    if (!validFBTarget(target)) FB_ERROR(GL_INVALID_ENUM);
    invalidateCommon(ctx, currentFBOForType(ctx, target), numAttachments, attachments, true, x, y, width, height);
}

void mglInvalidateNamedFramebufferData(GLMContext ctx, GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments)
{
    Framebuffer *fbo = NULL;
    if (framebuffer) { bool ok; fbo = namedFBO(ctx, framebuffer, &ok); if (!ok) return; }
    invalidateCommon(ctx, fbo, numAttachments, attachments, false, 0, 0, 0, 0);
}

void mglInvalidateNamedFramebufferSubData(GLMContext ctx, GLuint framebuffer, GLsizei numAttachments, const GLenum *attachments, GLint x, GLint y, GLsizei width, GLsizei height)
{
    Framebuffer *fbo = NULL;
    if (framebuffer) { bool ok; fbo = namedFBO(ctx, framebuffer, &ok); if (!ok) return; }
    invalidateCommon(ctx, fbo, numAttachments, attachments, true, x, y, width, height);
}

#pragma mark DSA draw/read buffers and clears

void mglNamedFramebufferDrawBuffer(GLMContext ctx, GLuint framebuffer, GLenum buf)
{
    Framebuffer *fbo = NULL;
    if (framebuffer) { bool ok; fbo = namedFBO(ctx, framebuffer, &ok); if (!ok) return; }
    mgl_fb_set_draw_buffers(ctx, fbo, 1, &buf, true);
}

void mglNamedFramebufferDrawBuffers(GLMContext ctx, GLuint framebuffer, GLsizei n, const GLenum *bufs)
{
    Framebuffer *fbo = NULL;
    if (framebuffer) { bool ok; fbo = namedFBO(ctx, framebuffer, &ok); if (!ok) return; }
    mgl_fb_set_draw_buffers(ctx, fbo, n, bufs, false);
}

void mglNamedFramebufferReadBuffer(GLMContext ctx, GLuint framebuffer, GLenum src)
{
    Framebuffer *fbo = NULL;
    if (framebuffer) { bool ok; fbo = namedFBO(ctx, framebuffer, &ok); if (!ok) return; }
    mgl_fb_set_read_buffer(ctx, fbo, src);
}

static bool namedOrDefault(GLMContext ctx, GLuint framebuffer, Framebuffer **out)
{
    *out = NULL;
    if (!framebuffer) return true;
    bool ok; *out = namedFBO(ctx, framebuffer, &ok);
    return ok;
}

void mglClearNamedFramebufferiv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLint *value)
{
    Framebuffer *fbo; if (!namedOrDefault(ctx, framebuffer, &fbo)) return;
    mgl_fb_clear_buffer(ctx, fbo, buffer, drawbuffer, MGL_FB_KIND_INT, value, 0, 0, false);
}

void mglClearNamedFramebufferuiv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLuint *value)
{
    Framebuffer *fbo; if (!namedOrDefault(ctx, framebuffer, &fbo)) return;
    mgl_fb_clear_buffer(ctx, fbo, buffer, drawbuffer, MGL_FB_KIND_UINT, value, 0, 0, false);
}

void mglClearNamedFramebufferfv(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, const GLfloat *value)
{
    Framebuffer *fbo; if (!namedOrDefault(ctx, framebuffer, &fbo)) return;
    mgl_fb_clear_buffer(ctx, fbo, buffer, drawbuffer, MGL_FB_KIND_FLOAT, value, 0, 0, false);
}

void mglClearNamedFramebufferfi(GLMContext ctx, GLuint framebuffer, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    Framebuffer *fbo; if (!namedOrDefault(ctx, framebuffer, &fbo)) return;
    mgl_fb_clear_buffer(ctx, fbo, buffer, drawbuffer, MGL_FB_KIND_FLOAT, NULL, depth, stencil, true);
}

#pragma mark blits

extern void mgl_fb_blit(GLMContext ctx, Framebuffer *readfbo, Framebuffer *drawfbo, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);

void mglBlitFramebuffer(GLMContext ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    mgl_fb_blit(ctx, ctx->state.readbuffer, ctx->state.framebuffer, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}

void mglBlitNamedFramebuffer(GLMContext ctx, GLuint readFramebuffer, GLuint drawFramebuffer, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    Framebuffer *rf, *df;
    if (!namedOrDefault(ctx, readFramebuffer, &rf)) return;
    if (!namedOrDefault(ctx, drawFramebuffer, &df)) return;
    mgl_fb_blit(ctx, rf, df, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter);
}
