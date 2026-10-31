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
 * rendering.c
 * MGL
 *
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <limits.h>
#include <stdio.h>
#include <stdint.h>

#include "mgl.h"

#include "pixel_utils.h"
#include "glm_context.h"
#include "mgl_ext_texdata.h"
#include "mgl_ext_queries.h"

#include "mgl_ext_fb.h"

/* compat domain: EXT_texture_integer integer clear colour (glClearColorIiEXT) */
#include "mgl_ext_compat.h"

#define FB_ERROR(_e_)   do { ERROR_RETURN(_e_); return; } while (0)

extern GLenum mgl_fb_status(GLMContext ctx, Framebuffer *fbo);
extern int mgl_fb_slot_kind(GLMContext ctx, Framebuffer *fbo, GLuint slot);
extern bool mgl_fb_has_depth(GLMContext ctx, Framebuffer *fbo);
extern bool mgl_fb_has_stencil(GLMContext ctx, Framebuffer *fbo);
extern void mgl_fb_clear_buffer(GLMContext ctx, Framebuffer *fbo, GLenum buffer, GLint drawbuffer, int kind, const void *value, GLfloat depth, GLint stencil, bool fi);
extern void mgl_fb_read_pixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *pixels);

void mglClear(GLMContext ctx, GLbitfield mask)
{
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
        FB_ERROR(GL_INVALID_VALUE);

    Framebuffer *fbo = ctx->state.framebuffer;

    if (mgl_fb_status(ctx, fbo) != GL_FRAMEBUFFER_COMPLETE)
        FB_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    /* Clear is discarded together with primitives (GL 4.6 14.1) */
    if (ctx->state.caps.rasterizer_discard)
        return;

    /* conditional rendering (queries domain) discards clears too */
    if (!mgl_queries_render_allowed(ctx))
        return;

    MGLFBClearRequest req;
    memset(&req, 0, sizeof(req));
    req.fbo = fbo;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);

        for (GLsizei i = 0; i < ds->num_draw_buffers && i < MAX_COLOR_ATTACHMENTS; i++)
        {
            int kind = mgl_fb_slot_kind(ctx, fbo, i);
            if (kind < 0)
                continue;

            req.color_slots |= 1u << i;
            req.color_kind[i] = kind;
            /* compat domain (EXT_texture_integer): once glClearColorIiEXT/
             * glClearColorIuiEXT has been called, integer attachments clear
             * with that exact bit pattern instead of the converted float
             * value; float attachments are unaffected. */
            struct MGLExtState_compat *cm = ctx->state.ext_compat;
            GLint icc[4] = { 0, 0, 0, 0 };
            if (cm && cm->clear_color_i_set && kind != MGL_FB_KIND_FLOAT)
                memcpy(icc, cm->clear_color_i, sizeof icc);
            for (int c = 0; c < 4; c++)
            {
                GLfloat v = STATE(color_clear_value[c]);
                if (kind == MGL_FB_KIND_FLOAT) req.color[i].f[c] = v;
                else if (kind == MGL_FB_KIND_INT)
                    req.color[i].i[c] = cm && cm->clear_color_i_set ? icc[c] : (GLint)v;
                else
                    req.color[i].u[c] = cm && cm->clear_color_i_set ? (GLuint)icc[c] :
                                        (v < 0 ? 0 : (GLuint)v);
            }
        }
    }

    if ((mask & GL_DEPTH_BUFFER_BIT) && mgl_fb_has_depth(ctx, fbo))
    {
        GLdouble d = STATE_VAR(depth_clear_value);
        req.depth = GL_TRUE;
        req.depth_value = (GLfloat)(d < 0.0 ? 0.0 : (d > 1.0 ? 1.0 : d));
    }

    if ((mask & GL_STENCIL_BUFFER_BIT) && mgl_fb_has_stencil(ctx, fbo))
    {
        req.stencil = GL_TRUE;
        req.stencil_value = (GLint)STATE_VAR(stencil_clear_value);
    }

    if (!req.color_slots && !req.depth && !req.stencil)
        return;

    if (ctx->mtl_funcs.ext_fb && ctx->mtl_funcs.ext_fb->clear)
        ctx->mtl_funcs.ext_fb->clear(ctx, &req);
}

void mglClearColor(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    ctx->state.color_clear_value[0] = red;
    ctx->state.color_clear_value[1] = green;
    ctx->state.color_clear_value[2] = blue;
    ctx->state.color_clear_value[3] = alpha;

    ctx->state.dirty_bits |= DIRTY_STATE;
}

void mglClearStencil(GLMContext ctx, GLint s)
{
    ctx->state.var.stencil_clear_value = s;

    ctx->state.dirty_bits |= DIRTY_STATE;
}

void mglClearDepth(GLMContext ctx, GLdouble depth)
{
    ctx->state.var.depth_clear_value = depth;

    ctx->state.dirty_bits |= DIRTY_STATE;
}

void mglClearBufferfv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLfloat *value)
{
    mgl_fb_clear_buffer(ctx, ctx->state.framebuffer, buffer, drawbuffer, MGL_FB_KIND_FLOAT, value, 0, 0, false);
}

void mglClearBufferfi(GLMContext ctx, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil)
{
    mgl_fb_clear_buffer(ctx, ctx->state.framebuffer, buffer, drawbuffer, MGL_FB_KIND_FLOAT, NULL, depth, stencil, true);
}

void mglFinish(GLMContext ctx)
{
    ctx->mtl_funcs.mtlFlush(ctx, true);
}

void mglFlush(GLMContext ctx)
{
    ctx->mtl_funcs.mtlFlush(ctx, false);
}

#pragma mark draw / read buffer selection

/* keep the legacy single-value state (GL_DRAW_BUFFER / GL_READ_BUFFER queries) and
 * binding queries in sync with the bound framebuffers */
void mgl_fb_sync_bindings(GLMContext ctx)
{
    MGLFBDrawState *dds = mgl_fb_draw_state(ctx, ctx->state.framebuffer);
    MGLFBDrawState *rds = mgl_fb_draw_state(ctx, ctx->state.readbuffer);

    STATE(draw_buffer) = dds->draw_buffers[0];
    STATE(read_buffer) = rds->read_buffer;
    STATE_VAR(draw_buffer) = dds->draw_buffers[0];
    STATE_VAR(read_buffer) = rds->read_buffer;
    STATE_VAR(draw_framebuffer_binding) = ctx->state.framebuffer ? ctx->state.framebuffer->name : 0;
    STATE_VAR(read_framebuffer_binding) = ctx->state.readbuffer ? ctx->state.readbuffer->name : 0;
}

static bool isColorAttachmentEnum(GLenum buf)
{
    return buf >= GL_COLOR_ATTACHMENT0 && buf <= GL_COLOR_ATTACHMENT31;
}

static void drawTargetsChanged(GLMContext ctx, Framebuffer *fbo)
{
    if (fbo == ctx->state.framebuffer)
    {
        STATE(dirty_bits) |= DIRTY_FBO | DIRTY_STATE;
        if (ctx->mtl_funcs.ext_fb && ctx->mtl_funcs.ext_fb->targets_changed)
            ctx->mtl_funcs.ext_fb->targets_changed(ctx);
    }
    mgl_fb_sync_bindings(ctx);
}

/* single = glDrawBuffer semantics (accepts FRONT, BACK, LEFT, FRONT_AND_BACK) */
void mgl_fb_set_draw_buffers(GLMContext ctx, Framebuffer *fbo, GLsizei n, const GLenum *bufs, bool single)
{
    GLenum out[MAX_COLOR_ATTACHMENTS];
    GLuint maxdb = STATE_VAR(max_draw_buffers) ? STATE_VAR(max_draw_buffers) : MAX_COLOR_ATTACHMENTS;
    if (maxdb > MAX_COLOR_ATTACHMENTS) maxdb = MAX_COLOR_ATTACHMENTS;

    if (n < 0 || (GLuint)n > maxdb)
        FB_ERROR(GL_INVALID_VALUE);
    if (n && !bufs)
        return;

    for (GLsizei i = 0; i < n; i++)
    {
        GLenum b = bufs[i];

        if (b == GL_NONE)
        {
            out[i] = b;
            continue;
        }

        if (fbo)
        {
            if (isColorAttachmentEnum(b))
            {
                if (b - GL_COLOR_ATTACHMENT0 >= STATE(max_color_attachments))
                    FB_ERROR(GL_INVALID_OPERATION);
            }
            else
            {
                switch(b)
                {
                    case GL_FRONT_LEFT: case GL_FRONT_RIGHT: case GL_BACK_LEFT: case GL_BACK_RIGHT:
                        FB_ERROR(GL_INVALID_OPERATION);
                    case GL_FRONT: case GL_BACK: case GL_LEFT: case GL_RIGHT: case GL_FRONT_AND_BACK:
                        FB_ERROR(single ? GL_INVALID_OPERATION : GL_INVALID_ENUM);
                    default:
                        FB_ERROR(GL_INVALID_ENUM);
                }
            }
        }
        else
        {
            /* default framebuffer: one color image serving front-left and back-left */
            switch(b)
            {
                case GL_FRONT_LEFT:
                case GL_BACK_LEFT:
                    break;
                case GL_FRONT: case GL_BACK: case GL_LEFT: case GL_FRONT_AND_BACK:
                    if (!single) FB_ERROR(GL_INVALID_ENUM);
                    break;
                case GL_FRONT_RIGHT: case GL_BACK_RIGHT:
                    FB_ERROR(GL_INVALID_OPERATION);
                case GL_RIGHT:
                    if (!single) FB_ERROR(GL_INVALID_ENUM);
                    FB_ERROR(GL_INVALID_OPERATION);
                default:
                    if (isColorAttachmentEnum(b)) FB_ERROR(GL_INVALID_OPERATION);
                    FB_ERROR(GL_INVALID_ENUM);
            }
            if (!single && n != 1)
                FB_ERROR(GL_INVALID_OPERATION);
        }

        for (GLsizei j = 0; j < i; j++)
            if (out[j] == b)
                FB_ERROR(GL_INVALID_OPERATION);

        out[i] = b;
    }

    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        ds->draw_buffers[i] = (i < n) ? out[i] : GL_NONE;
    ds->num_draw_buffers = n;

    drawTargetsChanged(ctx, fbo);
}

void mgl_fb_set_read_buffer(GLMContext ctx, Framebuffer *fbo, GLenum src)
{
    if (fbo)
    {
        if (src != GL_NONE)
        {
            if (isColorAttachmentEnum(src))
            {
                if (src - GL_COLOR_ATTACHMENT0 >= STATE(max_color_attachments))
                    FB_ERROR(GL_INVALID_OPERATION);
            }
            else switch(src)
            {
                case GL_FRONT: case GL_BACK: case GL_LEFT: case GL_RIGHT:
                case GL_FRONT_LEFT: case GL_FRONT_RIGHT: case GL_BACK_LEFT: case GL_BACK_RIGHT:
                case GL_FRONT_AND_BACK:  /* table 17.4 constants on an FBO: INVALID_OPERATION (CTS) */
                    FB_ERROR(GL_INVALID_OPERATION);
                default:
                    FB_ERROR(GL_INVALID_ENUM);
            }
        }
    }
    else
    {
        switch(src)
        {
            case GL_NONE: case GL_FRONT: case GL_BACK: case GL_LEFT:
            case GL_FRONT_LEFT: case GL_BACK_LEFT:
                break;
            case GL_RIGHT: case GL_FRONT_RIGHT: case GL_BACK_RIGHT:
                FB_ERROR(GL_INVALID_OPERATION);
            default:
                if (isColorAttachmentEnum(src)) FB_ERROR(GL_INVALID_OPERATION);
                FB_ERROR(GL_INVALID_ENUM);
        }
    }

    mgl_fb_draw_state(ctx, fbo)->read_buffer = src;
    mgl_fb_sync_bindings(ctx);
}

void mglDrawBuffers(GLMContext ctx, GLsizei n, const GLenum *bufs)
{
    mgl_fb_set_draw_buffers(ctx, ctx->state.framebuffer, n, bufs, false);
}

void mglDrawBuffer(GLMContext ctx, GLenum buf)
{
    mgl_fb_set_draw_buffers(ctx, ctx->state.framebuffer, 1, &buf, true);
}

void mglReadBuffer(GLMContext ctx, GLenum buf)
{
    mgl_fb_set_read_buffer(ctx, ctx->state.readbuffer, buf);
}

void mglPixelStorei(GLMContext ctx, GLenum pname, GLint param)
{
    // ERROR_CHECK_RETURN(param >= 0, GL_INVALID_VALUE);
    if (param < 0 && pname != GL_PACK_SWAP_BYTES && pname != GL_PACK_LSB_FIRST &&
        pname != GL_UNPACK_SWAP_BYTES && pname != GL_UNPACK_LSB_FIRST) {
        fprintf(stderr, "MGL Error: mglPixelStorei: param < 0 (%d) for pname 0x%x\n", param, pname);
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    switch(pname)
    {
        case GL_PACK_SWAP_BYTES:
            ctx->state.pack.swap_bytes = (param != 0 ? true : false);
            break;

        case GL_PACK_LSB_FIRST:
            ctx->state.pack.lsb_first = (param != 0 ? true : false);
            break;

        case GL_PACK_ROW_LENGTH:
            ctx->state.pack.row_length = param;
            break;

        case GL_PACK_IMAGE_HEIGHT:
            ctx->state.pack.image_height = param;
            break;

        case GL_PACK_SKIP_ROWS:
            ctx->state.pack.skip_rows = param;
            break;

        case GL_PACK_SKIP_PIXELS:
            ctx->state.pack.skip_pixels = param;
            break;

        case GL_PACK_SKIP_IMAGES:
            ctx->state.pack.skip_images = param;
            break;

        case GL_PACK_ALIGNMENT:
            switch(param)
            {
                case 1:
                case 2:
                case 4:
                case 8:
                    ctx->state.pack.alignment = param;
                    break;

                default:
                    fprintf(stderr, "MGL Error: mglPixelStorei: invalid PACK_ALIGNMENT %d\n", param);
                    ERROR_RETURN(GL_INVALID_VALUE);
                    break;
            }
            break;

        case GL_UNPACK_SWAP_BYTES:
            ctx->state.unpack.swap_bytes = (param != 0 ? true : false);
            break;

        case GL_UNPACK_LSB_FIRST:
            ctx->state.unpack.lsb_first = (param != 0 ? true : false);
            break;

        case GL_UNPACK_ROW_LENGTH:
            ctx->state.unpack.row_length = param;
            break;
        case GL_UNPACK_IMAGE_HEIGHT:
            ctx->state.unpack.image_height = param;
            break;

        case GL_UNPACK_SKIP_ROWS:
            ctx->state.unpack.skip_rows = param;
            break;

        case GL_UNPACK_SKIP_PIXELS:
            ctx->state.unpack.skip_pixels = param;
            break;

        case GL_UNPACK_SKIP_IMAGES:
            ctx->state.unpack.skip_images = param;
            break;

        case GL_UNPACK_ALIGNMENT:
            switch(param)
            {
                case 1:
                case 2:
                case 4:
                case 8:
                    ctx->state.unpack.alignment = param;
                    break;

                default:
                    fprintf(stderr, "MGL Error: mglPixelStorei: invalid UNPACK_ALIGNMENT %d\n", param);
                    ERROR_RETURN(GL_INVALID_VALUE);
                    break;
            }
            break;

        default:
            /* texdata: *_COMPRESSED_BLOCK_WIDTH/HEIGHT/DEPTH/SIZE */
            if (mgl_texdata_pixel_store(ctx, pname, param))
                break;
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglPixelStoref(GLMContext ctx, GLenum pname, GLfloat param)
{
    mglPixelStorei(ctx, pname, (GLint)param);
}

void mglReadPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, void *pixels)
{
    mgl_fb_read_pixels(ctx, x, y, width, height, format, type, -1, pixels);
}
