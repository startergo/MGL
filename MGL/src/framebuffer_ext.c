/*
 * framebuffer_ext.c
 * MGL — ClearBuffer*, ReadPixels / ReadnPixels (format conversion + pack state),
 * BlitFramebuffer validation, multisample queries, fb-domain glGet pnames.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_fb.h"
#include "mgl_ext_gpuaudit.h"   /* gpuaudit: PBO readback on the GPU */
#include "mgl_ext_queries.h"

#define FB_ERROR(_e_)   do { ERROR_RETURN(_e_); return; } while (0)

extern Renderbuffer *findRenderbuffer(GLMContext ctx, GLuint renderbuffer);
extern GLint mgl_fb_samples(GLMContext ctx, Framebuffer *fbo);
/* compat r99: implicit image-atomic sidecar sync before texture reads */
extern void mgl_compat_image_atom_sync(GLMContext ctx);

static struct MGLExtMetal_fb *fbm(GLMContext ctx) { return ctx->mtl_funcs.ext_fb; }

#pragma mark attachment classification

static bool attachmentDefined(GLMContext ctx, FBOAttachment *att)
{
    return att && att->texture && mgl_fb_attachment_size(ctx, att, NULL, NULL, NULL);
}

static GLenum attachmentFormat(GLMContext ctx, FBOAttachment *att)
{
    if (att->textarget == GL_RENDERBUFFER)
        return ((MGLRenderbufferEx *)att->buf.rbo)->internalformat;
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    return tex ? tex->internalformat : GL_NONE;
}

static int kindForFormatInfo(const MGLFBFormatInfo *fi)
{
    if (!fi) return MGL_FB_KIND_FLOAT;
    if (fi->component_type == GL_INT) return MGL_FB_KIND_INT;
    if (fi->component_type == GL_UNSIGNED_INT) return MGL_FB_KIND_UINT;
    return MGL_FB_KIND_FLOAT;
}

/* color attachment selected by a draw-buffer slot, NULL for default fb / none */
static FBOAttachment *slotAttachment(GLMContext ctx, Framebuffer *fbo, GLuint slot)
{
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);
    if (!fbo || slot >= (GLuint)ds->num_draw_buffers || slot >= MAX_COLOR_ATTACHMENTS) return NULL;
    GLenum b = ds->draw_buffers[slot];
    if (b < GL_COLOR_ATTACHMENT0 || b >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) return NULL;
    return &fbo->color_attachments[b - GL_COLOR_ATTACHMENT0];
}

int mgl_fb_slot_kind(GLMContext ctx, Framebuffer *fbo, GLuint slot)
{
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);
    if (slot >= (GLuint)ds->num_draw_buffers || slot >= MAX_COLOR_ATTACHMENTS || ds->draw_buffers[slot] == GL_NONE)
        return -1;
    if (!fbo)
        return MGL_FB_KIND_FLOAT;
    FBOAttachment *att = slotAttachment(ctx, fbo, slot);
    if (!attachmentDefined(ctx, att))
        return -1;
    return kindForFormatInfo(mgl_fb_format_info(attachmentFormat(ctx, att)));
}

bool mgl_fb_has_depth(GLMContext ctx, Framebuffer *fbo)
{
    if (!fbo) return ctx->depth_format.format != 0;
    if (!attachmentDefined(ctx, &fbo->depth)) return false;
    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentFormat(ctx, &fbo->depth));
    return fi && fi->d > 0;
}

/* geom r58 (texture_swizzle.functional Crash->Pass): true when the draw
 * framebuffer's color slot `slot` writes a format with no alpha channel
 * (R3_G3_B2, RGB*, RG*, R*, ...). GL discards writes to a missing channel and
 * reads return 1 for alpha, but Metal has no alpha-less color attachment
 * format, so those formats are emulated on an RGBA backing — the pipeline must
 * mask the alpha write or a fill leaks its alpha into the backing and a
 * GL_TEXTURE_SWIZZLE_* that sources A reads it back instead of 1.0. */
bool mgl_fb_slot_lacks_alpha(GLMContext ctx, Framebuffer *fbo, GLuint slot)
{
    if (!fbo) return false; /* default framebuffer backing is opaque */
    FBOAttachment *att = slotAttachment(ctx, fbo, slot);
    if (!attachmentDefined(ctx, att))
        return false;
    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentFormat(ctx, att));
    return fi && fi->base != GL_DEPTH_COMPONENT && fi->base != GL_DEPTH_STENCIL && fi->a == 0;
}

bool mgl_fb_has_stencil(GLMContext ctx, Framebuffer *fbo)
{
    if (!fbo) return ctx->stencil_format.format != 0;
    if (!attachmentDefined(ctx, &fbo->stencil)) return false;
    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentFormat(ctx, &fbo->stencil));
    return fi && fi->s > 0;
}

/* color image read by ReadPixels / used as blit source; returns false if none.
 * *info receives the format info (default fb: RGBA8) */
static bool readColorSource(GLMContext ctx, Framebuffer *fbo, GLenum *attachment, const MGLFBFormatInfo **info)
{
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);
    GLenum rb = ds->read_buffer;

    if (rb == GL_NONE)
        return false;

    if (!fbo)
    {
        *attachment = rb;
        *info = mgl_fb_format_info(GL_RGBA8);
        return true;
    }

    if (rb < GL_COLOR_ATTACHMENT0 || rb >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS)
        return false;
    FBOAttachment *att = &fbo->color_attachments[rb - GL_COLOR_ATTACHMENT0];
    if (!attachmentDefined(ctx, att))
        return false;
    *attachment = rb;
    *info = mgl_fb_format_info(attachmentFormat(ctx, att));
    return *info != NULL;
}

/* depth / stencil format class for blit compatibility (Metal-equivalent formats) */
static void dsClass(GLMContext ctx, Framebuffer *fbo, bool stencil, GLuint *cls)
{
    if (!fbo)
    {
        bool both = ctx->depth_format.format && ctx->stencil_format.format;
        /* the default framebuffer's packed depth-stencil is normalized the same
         * way as FBO attachments (see below) */
        *cls = both ? 0x1808 : (stencil ? 0x0008 : 0x2000);
        return;
    }
    FBOAttachment *att = stencil ? &fbo->stencil : &fbo->depth;
    const MGLFBFormatInfo *fi = mgl_fb_format_info(attachmentFormat(ctx, att));
    if (!fi)
    {
        *cls = 0;
        return;
    }
    /* MGL stores every packed depth-stencil format (GL_DEPTH24_STENCIL8 and
     * GL_DEPTH32F_STENCIL8) in the same Metal pixel format, so they are one
     * storage class: normalize to the D24S8 class.  Without this the KHR-GL46
     * framebuffer_blit tests fail, because the CTS hardcodes the assumption
     * that its default framebuffer is DEPTH24_STENCIL8 while deqp actually
     * creates it as DEPTH32F_STENCIL8 (DONT_CARE depth/stencil bits select the
     * first entry of glu's format list) and blits between the two. */
    if (fi->d > 0 && fi->s > 0)
    {
        *cls = 0x1808;
        return;
    }
    *cls = ((GLuint)fi->d << 8) | fi->s | (fi->component_type == GL_FLOAT ? 0x10000 : 0);
}

void mgl_fb_color_read_format(GLMContext ctx, Framebuffer *fbo, GLenum *format, GLenum *type)
{
    GLenum att;
    const MGLFBFormatInfo *fi = NULL;
    *format = GL_RGBA;
    *type = GL_UNSIGNED_BYTE;
    if (!readColorSource(ctx, fbo, &att, &fi) || !fi)
        return;
    switch(fi->component_type)
    {
        case GL_INT:          *format = GL_RGBA_INTEGER; *type = GL_INT; break;
        case GL_UNSIGNED_INT: *format = GL_RGBA_INTEGER; *type = GL_UNSIGNED_INT; break;
        case GL_FLOAT:        *format = GL_RGBA; *type = GL_FLOAT; break;
        case GL_SIGNED_NORMALIZED: *format = GL_RGBA; *type = GL_BYTE; break;
        default:
            *format = GL_RGBA;
            *type = (fi->r > 8) ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
            break;
    }
}

#pragma mark ClearBuffer*

void mgl_fb_clear_buffer(GLMContext ctx, Framebuffer *fbo, GLenum buffer, GLint drawbuffer, int kind, const void *value, GLfloat depth, GLint stencil, bool fi)
{
    GLuint maxdb = STATE_VAR(max_draw_buffers) ? STATE_VAR(max_draw_buffers) : MAX_COLOR_ATTACHMENTS;
    if (maxdb > MAX_COLOR_ATTACHMENTS) maxdb = MAX_COLOR_ATTACHMENTS;

    if (fi)
    {
        if (buffer != GL_DEPTH_STENCIL) FB_ERROR(GL_INVALID_ENUM);
    }
    else switch(kind)
    {
        case MGL_FB_KIND_INT:
            if (buffer != GL_COLOR && buffer != GL_STENCIL) FB_ERROR(GL_INVALID_ENUM);
            break;
        case MGL_FB_KIND_UINT:
            if (buffer != GL_COLOR) FB_ERROR(GL_INVALID_ENUM);
            break;
        default:
            if (buffer != GL_COLOR && buffer != GL_DEPTH) FB_ERROR(GL_INVALID_ENUM);
            break;
    }

    if (buffer == GL_COLOR)
    {
        if (drawbuffer < 0 || (GLuint)drawbuffer >= maxdb) FB_ERROR(GL_INVALID_VALUE);
    }
    else if (drawbuffer != 0)
        FB_ERROR(GL_INVALID_VALUE);

    if (mgl_fb_status(ctx, fbo) != GL_FRAMEBUFFER_COMPLETE)
        FB_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    if (!fi && !value)
        return;

    if (ctx->state.caps.rasterizer_discard)
        return;

    /* conditional rendering (queries domain) */
    if (!mgl_queries_render_allowed(ctx))
        return;

    MGLFBClearRequest req;
    memset(&req, 0, sizeof(req));
    req.fbo = fbo;

    switch(buffer)
    {
        case GL_COLOR:
        {
            int akind = mgl_fb_slot_kind(ctx, fbo, drawbuffer);
            if (akind < 0)
                return;   /* NONE or nothing attached: no-op */
            req.color_slots = 1u << drawbuffer;
            req.color_kind[drawbuffer] = kind;
            memcpy(req.color[drawbuffer].f, value, 4 * sizeof(GLfloat));
            break;
        }
        case GL_DEPTH:
            if (!mgl_fb_has_depth(ctx, fbo)) return;
            req.depth = GL_TRUE;
            req.depth_value = *(const GLfloat *)value;
            break;
        case GL_STENCIL:
            if (!mgl_fb_has_stencil(ctx, fbo)) return;
            req.stencil = GL_TRUE;
            req.stencil_value = *(const GLint *)value;
            break;
        case GL_DEPTH_STENCIL:
            req.depth = mgl_fb_has_depth(ctx, fbo);
            req.stencil = mgl_fb_has_stencil(ctx, fbo);
            req.depth_value = depth;
            req.stencil_value = stencil;
            break;
    }

    if (req.depth)
        req.depth_value = req.depth_value < 0.f ? 0.f : (req.depth_value > 1.f ? 1.f : req.depth_value);

    if (!req.color_slots && !req.depth && !req.stencil)
        return;

    if (fbm(ctx) && fbm(ctx)->clear)
        fbm(ctx)->clear(ctx, &req);
}

void mglClearBufferiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLint *value)
{
    mgl_fb_clear_buffer(ctx, ctx->state.framebuffer, buffer, drawbuffer, MGL_FB_KIND_INT, value, 0, 0, false);
}

void mglClearBufferuiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLuint *value)
{
    mgl_fb_clear_buffer(ctx, ctx->state.framebuffer, buffer, drawbuffer, MGL_FB_KIND_UINT, value, 0, 0, false);
}

#pragma mark pixel packing

static GLuint formatComponents(GLenum format, int map[4], bool *integer)
{
    *integer = false;
    switch(format)
    {
        case GL_RED_INTEGER:   *integer = true; /* fall through */
        case GL_RED:           map[0] = 0; return 1;
        case GL_GREEN_INTEGER: *integer = true; /* fall through */
        case GL_GREEN:         map[0] = 1; return 1;
        case GL_BLUE_INTEGER:  *integer = true; /* fall through */
        case GL_BLUE:          map[0] = 2; return 1;
        case GL_ALPHA:         map[0] = 3; return 1;
        case GL_RG_INTEGER:    *integer = true; /* fall through */
        case GL_RG:            map[0] = 0; map[1] = 1; return 2;
        case GL_RGB_INTEGER:   *integer = true; /* fall through */
        case GL_RGB:           map[0] = 0; map[1] = 1; map[2] = 2; return 3;
        case GL_BGR_INTEGER:   *integer = true; /* fall through */
        case GL_BGR:           map[0] = 2; map[1] = 1; map[2] = 0; return 3;
        case GL_RGBA_INTEGER:  *integer = true; /* fall through */
        case GL_RGBA:          map[0] = 0; map[1] = 1; map[2] = 2; map[3] = 3; return 4;
        case GL_BGRA_INTEGER:  *integer = true; /* fall through */
        case GL_BGRA:          map[0] = 2; map[1] = 1; map[2] = 0; map[3] = 3; return 4;
        case GL_DEPTH_COMPONENT:
        case GL_STENCIL_INDEX:
            map[0] = 0; return 1;
        case GL_DEPTH_STENCIL:
            map[0] = 0; map[1] = 1; return 2;
    }
    return 0;
}

typedef struct {
    GLuint bytes;       /* size of the packed unit */
    GLuint n;           /* number of fields */
    GLubyte bits[4];    /* field widths, component order */
    bool rev;           /* first component in the least significant bits */
} PackedType;

static bool packedType(GLenum type, PackedType *p)
{
    memset(p, 0, sizeof(*p));
#define PT(_b,_n,_r,a,b,c,d) do { p->bytes=_b; p->n=_n; p->rev=_r; p->bits[0]=a; p->bits[1]=b; p->bits[2]=c; p->bits[3]=d; } while(0)
    switch(type)
    {
        case GL_UNSIGNED_BYTE_3_3_2:            PT(1,3,false,3,3,2,0); return true;
        case GL_UNSIGNED_BYTE_2_3_3_REV:        PT(1,3,true, 3,3,2,0); return true;
        case GL_UNSIGNED_SHORT_5_6_5:           PT(2,3,false,5,6,5,0); return true;
        case GL_UNSIGNED_SHORT_5_6_5_REV:       PT(2,3,true, 5,6,5,0); return true;
        case GL_UNSIGNED_SHORT_4_4_4_4:         PT(2,4,false,4,4,4,4); return true;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:     PT(2,4,true, 4,4,4,4); return true;
        case GL_UNSIGNED_SHORT_5_5_5_1:         PT(2,4,false,5,5,5,1); return true;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:     PT(2,4,true, 5,5,5,1); return true;
        case GL_UNSIGNED_INT_8_8_8_8:           PT(4,4,false,8,8,8,8); return true;
        case GL_UNSIGNED_INT_8_8_8_8_REV:       PT(4,4,true, 8,8,8,8); return true;
        case GL_UNSIGNED_INT_10_10_10_2:        PT(4,4,false,10,10,10,2); return true;
        case GL_UNSIGNED_INT_2_10_10_10_REV:    PT(4,4,true, 10,10,10,2); return true;
        case GL_UNSIGNED_INT_10F_11F_11F_REV:   PT(4,3,true, 11,11,10,0); return true;
        case GL_UNSIGNED_INT_5_9_9_9_REV:       PT(4,3,true, 9,9,9,0); return true;
        case GL_UNSIGNED_INT_24_8:              PT(4,2,false,24,8,0,0); return true;
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: PT(8,2,true, 32,8,0,0); return true;
    }
#undef PT
    return false;
}

static GLuint typeSize(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2;
        case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4;
    }
    PackedType p;
    if (packedType(type, &p)) return p.bytes;
    return 0;
}

static uint16_t floatToHalf(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u & 0x7fffff;

    if (((v.u >> 23) & 0xff) == 0xff)
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00);
    if (exp <= 0)
    {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000) h++;
    return (uint16_t)h;
}

/* unsigned small float (11 or 10 bits: 5-bit exponent, 6/5-bit mantissa) */
static uint32_t floatToUF(float f, int mbits)
{
    if (!(f > 0.f)) return 0;  /* negatives and NaN-free zero */
    if (isinf(f)) return (0x1fu << mbits);
    union { float f; uint32_t u; } v = { f };
    int32_t exp = (int32_t)((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u & 0x7fffff;
    if (exp >= 31) return (0x1eu << mbits) | ((1u << mbits) - 1);
    if (exp <= 0)
    {
        if (exp < -mbits) return 0;
        mant |= 0x800000;
        return mant >> (24 - mbits - exp);
    }
    return ((uint32_t)exp << mbits) | (mant >> (23 - mbits));
}

static uint32_t floatToRGB9E5(const float c[3])
{
    const float maxv = 65408.0f;
    float r = fminf(fmaxf(c[0], 0.f), maxv), g = fminf(fmaxf(c[1], 0.f), maxv), b = fminf(fmaxf(c[2], 0.f), maxv);
    float m = fmaxf(r, fmaxf(g, b));
    int e = (m > 0.f) ? (int)floorf(log2f(m)) + 1 + 15 : 0;
    if (e < 0) e = 0;
    float denom = exp2f((float)(e - 15 - 9));
    int mm = (int)floorf(m / denom + 0.5f);
    if (mm == 512) { e++; denom *= 2.f; }
    uint32_t rr = (uint32_t)floorf(r / denom + 0.5f), gg = (uint32_t)floorf(g / denom + 0.5f), bb = (uint32_t)floorf(b / denom + 0.5f);
    if (rr > 511) rr = 511; if (gg > 511) gg = 511; if (bb > 511) bb = 511;
    return rr | (gg << 9) | (bb << 18) | ((uint32_t)e << 27);
}

static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

static uint32_t normU(double c, GLuint bits)
{
    double m = (bits >= 32) ? 4294967295.0 : (double)((1ull << bits) - 1);
    return (uint32_t)llround(clampd(c, 0.0, 1.0) * m);
}

static int32_t normS(double c, GLuint bits)
{
    double m = (double)((1ull << (bits - 1)) - 1);
    return (int32_t)llround(clampd(c, -1.0, 1.0) * m);
}

static void swapBytes(uint8_t *p, GLuint n)
{
    for (GLuint i = 0; i < n / 2; i++) { uint8_t t = p[i]; p[i] = p[n - 1 - i]; p[n - 1 - i] = t; }
}

/* value of one component as double (float data) or int64 (integer data) */
typedef struct { bool is_int; double d; int64_t i; } CompVal;

static void storeComponent(uint8_t *dst, GLenum type, CompVal v, bool integer_format, bool swap)
{
    GLuint sz = typeSize(type);
    union { uint8_t b[4]; int8_t sb; uint16_t us; int16_t ss; uint32_t ui; int32_t si; float f; } u;
    memset(&u, 0, sizeof(u));

    if (integer_format || v.is_int)
    {
        int64_t i = v.is_int ? v.i : (int64_t)llround(v.d);
        switch(type)
        {
            case GL_UNSIGNED_BYTE:  u.b[0] = (uint8_t)(i < 0 ? 0 : (i > 255 ? 255 : i)); break;
            case GL_BYTE:           u.sb = (int8_t)(i < -128 ? -128 : (i > 127 ? 127 : i)); break;
            case GL_UNSIGNED_SHORT: u.us = (uint16_t)(i < 0 ? 0 : (i > 65535 ? 65535 : i)); break;
            case GL_SHORT:          u.ss = (int16_t)(i < -32768 ? -32768 : (i > 32767 ? 32767 : i)); break;
            case GL_UNSIGNED_INT:   u.ui = (uint32_t)(i < 0 ? 0 : (i > 0xffffffffll ? 0xffffffffll : i)); break;
            case GL_INT:            u.si = (int32_t)(i < INT32_MIN ? INT32_MIN : (i > INT32_MAX ? INT32_MAX : i)); break;
            case GL_FLOAT:          u.f = (float)i; break;          /* stencil index as float */
            case GL_HALF_FLOAT:     u.us = floatToHalf((float)i); break;
        }
    }
    else
    {
        switch(type)
        {
            case GL_UNSIGNED_BYTE:  u.b[0] = (uint8_t)normU(v.d, 8); break;
            case GL_BYTE:           u.sb = (int8_t)normS(v.d, 8); break;
            case GL_UNSIGNED_SHORT: u.us = (uint16_t)normU(v.d, 16); break;
            case GL_SHORT:          u.ss = (int16_t)normS(v.d, 16); break;
            case GL_UNSIGNED_INT:   u.ui = normU(v.d, 32); break;
            case GL_INT:            u.si = normS(v.d, 32); break;
            case GL_FLOAT:          u.f = (float)v.d; break;
            case GL_HALF_FLOAT:     u.us = floatToHalf((float)v.d); break;
        }
    }

    if (swap) swapBytes(u.b, sz);
    memcpy(dst, u.b, sz);
}

static void storePacked(uint8_t *dst, GLenum type, const PackedType *p, const CompVal *c, bool integer_format, bool swap)
{
    uint8_t out[8];
    memset(out, 0, sizeof(out));

    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV)
    {
        uint32_t v = floatToUF((float)c[0].d, 6) | (floatToUF((float)c[1].d, 6) << 11) | (floatToUF((float)c[2].d, 5) << 22);
        memcpy(out, &v, 4);
    }
    else if (type == GL_UNSIGNED_INT_5_9_9_9_REV)
    {
        float f[3] = { (float)c[0].d, (float)c[1].d, (float)c[2].d };
        uint32_t v = floatToRGB9E5(f);
        memcpy(out, &v, 4);
    }
    else if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
    {
        float d = (float)c[0].d;
        uint32_t s = (uint32_t)(c[1].i & 0xff);
        memcpy(out, &d, 4);
        memcpy(out + 4, &s, 4);
        if (swap) { swapBytes(out, 4); swapBytes(out + 4, 4); }
        memcpy(dst, out, 8);
        return;
    }
    else if (type == GL_UNSIGNED_INT_24_8)
    {
        uint32_t v = (normU(c[0].d, 24) << 8) | (uint32_t)(c[1].i & 0xff);
        memcpy(out, &v, 4);
    }
    else
    {
        uint32_t v = 0;
        GLuint total = p->bytes * 8, shift = 0;
        for (GLuint k = 0; k < p->n; k++)
        {
            GLuint b = p->bits[k];
            uint32_t f;
            if (integer_format || c[k].is_int)
            {
                int64_t i = c[k].is_int ? c[k].i : llround(c[k].d);
                int64_t mx = (1ll << b) - 1;
                f = (uint32_t)(i < 0 ? 0 : (i > mx ? mx : i));
            }
            else
                f = normU(c[k].d, b);

            if (p->rev) { v |= f << shift; shift += b; }
            else { shift += b; v |= f << (total - shift); }
        }
        if (p->bytes == 1) out[0] = (uint8_t)v;
        else if (p->bytes == 2) { uint16_t s = (uint16_t)v; memcpy(out, &s, 2); }
        else memcpy(out, &v, 4);
    }

    if (swap) swapBytes(out, p->bytes);
    memcpy(dst, out, p->bytes);
}

static bool validReadFormat(GLenum format)
{
    int map[4]; bool integer;
    return formatComponents(format, map, &integer) != 0;
}

static bool validReadType(GLenum type)
{
    return typeSize(type) != 0;
}

/* GL 4.6 table 8.5 / 8.8 combination rules; returns GL_NO_ERROR or the error */
static GLenum checkFormatType(GLenum format, GLenum type)
{
    int map[4]; bool integer;
    GLuint n = formatComponents(format, map, &integer);
    PackedType p;
    bool packed = packedType(type, &p);

    if (format == GL_DEPTH_STENCIL)
        return (type == GL_UNSIGNED_INT_24_8 || type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) ? GL_NO_ERROR : GL_INVALID_ENUM;

    if (type == GL_UNSIGNED_INT_24_8 || type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
        return GL_INVALID_OPERATION;

    if (integer && (type == GL_FLOAT || type == GL_HALF_FLOAT ||
                    type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_5_9_9_9_REV))
        return GL_INVALID_OPERATION;

    if (packed)
    {
        if ((format == GL_DEPTH_COMPONENT || format == GL_STENCIL_INDEX))
            return GL_INVALID_OPERATION;
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV || type == GL_UNSIGNED_INT_5_9_9_9_REV)
            return (format == GL_RGB) ? GL_NO_ERROR : GL_INVALID_OPERATION;
        if (p.n != n)
            return GL_INVALID_OPERATION;
        /* GL 4.6 table 8.3 (the same pairs tdValidateFormatType enforces for uploads):
         * the 3-component packed types carry RGB only, the 4-component ones RGBA / BGRA
         * and their integer twins (a BGR / 5_6_5 read is invalid even though the counts
         * match, KHR-GL46.packed_pixels checks it) */
        if (n == 3 && !(format == GL_RGB || format == GL_RGB_INTEGER))
            return GL_INVALID_OPERATION;
        if (n == 4 && !(format == GL_RGBA || format == GL_BGRA || format == GL_RGBA_INTEGER || format == GL_BGRA_INTEGER))
            return GL_INVALID_OPERATION;
    }
    return GL_NO_ERROR;
}

typedef struct {
    size_t group;       /* bytes per pixel */
    size_t stride;      /* bytes per row */
    size_t skip;        /* byte offset of the first pixel */
    size_t span;        /* bytes from first byte touched to last byte touched + 1, measured from 0 */
} PackLayout;

static void packLayout(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, PackLayout *l)
{
    int map[4]; bool integer;
    GLuint n = formatComponents(format, map, &integer);
    PackedType p;
    size_t s, group;

    if (packedType(type, &p)) { s = p.bytes; group = p.bytes; }
    else { s = typeSize(type); group = s * n; }

    size_t a = STATE(pack.alignment) ? (size_t)STATE(pack.alignment) : 4;
    size_t len = STATE(pack.row_length) > 0 ? (size_t)STATE(pack.row_length) : (size_t)width;
    size_t row = group * len;
    size_t stride = (s >= a) ? row : ((row + a - 1) / a) * a;

    l->group = group;
    l->stride = stride;
    l->skip = (size_t)STATE(pack.skip_rows) * stride + (size_t)STATE(pack.skip_pixels) * group;
    l->span = (width && height) ? l->skip + (size_t)(height - 1) * stride + (size_t)width * group : 0;
}

/* sRGB-encoded color formats: reads with float-kind clients must decode them
 * (GL 4.6 18.2.2); UNORM clients get the stored bytes back unchanged */
static bool fbInternalFormatIsSRGB(GLenum ifmt)
{
    switch (ifmt)
    {
        case GL_SRGB8:
        case GL_SRGB8_ALPHA8:
        case GL_SR8_EXT:
        case GL_SRG8_EXT:
        case GL_COMPRESSED_SRGB:
        case GL_COMPRESSED_SRGB_ALPHA:
        case GL_COMPRESSED_SRGB8_ETC2:
        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
        case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:
            return true;
        default:
            return false;
    }
}

void mgl_fb_read_pixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *pixels)
{
    if (!validReadFormat(format) || !validReadType(type))
        FB_ERROR(GL_INVALID_ENUM);

    GLenum err = checkFormatType(format, type);
    if (err != GL_NO_ERROR)
        FB_ERROR(err);

    if (width < 0 || height < 0)
        FB_ERROR(GL_INVALID_VALUE);

    Framebuffer *fbo = ctx->state.readbuffer;

    if (mgl_fb_status(ctx, fbo) != GL_FRAMEBUFFER_COMPLETE)
        FB_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    if (mgl_fb_samples(ctx, fbo) > 0)
        FB_ERROR(GL_INVALID_OPERATION);

    int map[4]; bool integer;
    GLuint ncomp = formatComponents(format, map, &integer);
    int aspect;
    GLenum attachment = GL_NONE;
    const MGLFBFormatInfo *fi = NULL;

    switch(format)
    {
        case GL_DEPTH_COMPONENT:
            if (!mgl_fb_has_depth(ctx, fbo)) FB_ERROR(GL_INVALID_OPERATION);
            aspect = MGL_FB_ASPECT_DEPTH;
            break;
        case GL_STENCIL_INDEX:
            if (!mgl_fb_has_stencil(ctx, fbo)) FB_ERROR(GL_INVALID_OPERATION);
            aspect = MGL_FB_ASPECT_STENCIL;
            break;
        case GL_DEPTH_STENCIL:
            if (!mgl_fb_has_depth(ctx, fbo) || !mgl_fb_has_stencil(ctx, fbo)) FB_ERROR(GL_INVALID_OPERATION);
            aspect = MGL_FB_ASPECT_DEPTH;
            break;
        default:
        {
            if (!readColorSource(ctx, fbo, &attachment, &fi))
                FB_ERROR(GL_INVALID_OPERATION);
            int k = kindForFormatInfo(fi);

            /* the client format must be integer exactly when the attachment is (GL 4.6
             * 8.2); beyond that any component count is accepted (GL_RED / GL_GREEN /
             * GL_BGR ... reads of an RGBA buffer drop or reorder components), and the
             * (format,type) pair must pass the table in checkFormatType above.  KHR-GL46
             * packed_pixels relies on reads like GL_RED / GL_FLOAT of an R32F attachment
             * working, which the strict RGBA-only reading of the spec would reject */
            if (integer != (k != MGL_FB_KIND_FLOAT))
                FB_ERROR(GL_INVALID_OPERATION);

            /* compat r104: a FLOAT-component attachment (R16F/R32F/RGBA32F/... or the
             * 10F_11F_11F packed float) is incompatible with a plain integer-typed
             * readback (GL_BYTE/_SHORT/_INT/_UNSIGNED_INT/... with a non-integer
             * format) — GLES 3.0 table 3.2 (the read type for a float buffer is
             * GL_FLOAT / GL_HALF_FLOAT), which the dEQP ES31 negative-coverage
             * read_pixels_fbo_format_mismatch case enforces on desktop contexts now
             * that float attachments are conformance-relevant
             * (GL_EXT_color_buffer_float advertised).  GL_UNSIGNED_BYTE and the
             * float-typed reads stay legal conversions (packed_pixels reads
             * GL_RED/GL_FLOAT of R32F, fbo.read tests read UBYTE). */
            if (fi && fi->component_type == GL_FLOAT && !integer)
            {
                switch (type)
                {
                    case GL_BYTE: case GL_SHORT: case GL_UNSIGNED_SHORT:
                    case GL_INT: case GL_UNSIGNED_INT:
                        FB_ERROR(GL_INVALID_OPERATION);
                    default:
                        break;
                }
            }
            aspect = MGL_FB_ASPECT_COLOR;
            break;
        }
    }

    PackLayout lay;
    packLayout(ctx, width, height, format, type, &lay);

    Buffer *pbo = STATE(buffers[_PIXEL_PACK_BUFFER]);
    size_t pbo_offset = 0;

    if (pbo)
    {
        /* a persistent mapping keeps a stable address: GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT
         * orders it; only a non-persistent mapping blocks the read (Mesa readpix.c) */
        if (pbo->mapped && !(pbo->access_flags & GL_MAP_PERSISTENT_BIT))
            FB_ERROR(GL_INVALID_OPERATION);
        pbo_offset = (size_t)(uintptr_t)pixels;
        GLuint ts = typeSize(type);
        if (ts && (pbo_offset % ts) != 0) FB_ERROR(GL_INVALID_OPERATION);
        if (pbo->size < 0 || pbo_offset + lay.span > (size_t)pbo->size) FB_ERROR(GL_INVALID_OPERATION);
    }
    else if (bufSize >= 0 && lay.span > (size_t)bufSize)
    {
        FB_ERROR(GL_INVALID_OPERATION);
    }

    if (width == 0 || height == 0)
        return;

    if (!pbo && !pixels)
        FB_ERROR(GL_INVALID_OPERATION);

    if (!fbm(ctx) || !fbm(ctx)->read)
        FB_ERROR(GL_INVALID_OPERATION);

    /* compat r99: image-atomic sidecars must be flushed into the texture before it
     * is read (GL 4.6 §8.25 implicit sync; no-op unless one is dirty) */
    mgl_compat_image_atom_sync(ctx);

    /* gpuaudit: a read into a PIXEL_PACK_BUFFER whose layout the Metal side can serve
     * with a single texture->buffer blit never touches the CPU (no decode loop, no
     * pack loop, no waitUntilCompleted); everything else keeps the path below.
     * Mismatched layouts, partial rectangles, sRGB and packed formats all decline. */
    if (pbo)
    {
        struct MGLExtMetal_gpuaudit *gpm = ctx->mtl_funcs.ext_gpuaudit;
        if (gpm && gpm->read_pixels_to_buffer &&
            gpm->read_pixels_to_buffer(ctx, fbo, aspect, attachment, x, y, width, height,
                                       format, type, pbo, pbo_offset, lay.stride, lay.skip))
            return;
    }

    size_t npix = (size_t)width * (size_t)height;
    void *color = NULL;
    float *depthv = NULL;
    GLuint *stencilv = NULL;
    int kind = MGL_FB_KIND_FLOAT;

    MGLFBReadRequest req;
    memset(&req, 0, sizeof(req));
    req.fbo = fbo; req.attachment = attachment;
    req.x = x; req.y = y; req.width = width; req.height = height;

    if (aspect == MGL_FB_ASPECT_COLOR)
    {
        color = calloc(npix, 16);
        if (!color) FB_ERROR(GL_OUT_OF_MEMORY);
        req.aspect = MGL_FB_ASPECT_COLOR; req.out = color;
        kind = fbm(ctx)->read(ctx, &req);
        if (kind < 0) { free(color); FB_ERROR(GL_INVALID_OPERATION); }
    }
    else
    {
        if (format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL)
        {
            depthv = (float *)calloc(npix, sizeof(float));
            if (!depthv) FB_ERROR(GL_OUT_OF_MEMORY);
            req.aspect = MGL_FB_ASPECT_DEPTH; req.out = depthv;
            if (fbm(ctx)->read(ctx, &req) < 0) { free(depthv); FB_ERROR(GL_INVALID_OPERATION); }
        }
        if (format == GL_STENCIL_INDEX || format == GL_DEPTH_STENCIL)
        {
            stencilv = (GLuint *)calloc(npix, sizeof(GLuint));
            if (!stencilv) { free(depthv); FB_ERROR(GL_OUT_OF_MEMORY); }
            req.aspect = MGL_FB_ASPECT_STENCIL; req.out = stencilv;
            if (fbm(ctx)->read(ctx, &req) < 0) { free(depthv); free(stencilv); FB_ERROR(GL_INVALID_OPERATION); }
        }
    }

    /* channels missing from the source image read as 0 (rgb) / 1 (alpha) */
    bool has[4] = { true, true, true, true };
    if (fi) { has[0] = fi->r > 0; has[1] = fi->g > 0; has[2] = fi->b > 0; has[3] = fi->a > 0; }

    PackedType pt;
    bool packed = packedType(type, &pt);
    bool swap = STATE(pack.swap_bytes) != 0;
    GLuint csz = typeSize(type);
    uint8_t *rowbuf = (uint8_t *)malloc(lay.group * (size_t)width);
    if (!rowbuf) { free(color); free(depthv); free(stencilv); FB_ERROR(GL_OUT_OF_MEMORY); }

    for (GLsizei r = 0; r < height; r++)
    {
        uint8_t *o = rowbuf;
        for (GLsizei c = 0; c < width; c++)
        {
            size_t idx = (size_t)r * (size_t)width + (size_t)c;
            CompVal cv[4];
            memset(cv, 0, sizeof(cv));

            if (aspect == MGL_FB_ASPECT_COLOR)
            {
                /* r54: NO sRGB transfer on readback.  GL 4.6 applies the sRGB decoder
                 * only when SAMPLING a texture from a shader (and the encoder only
                 * when writing with FRAMEBUFFER_SRGB on); glReadPixels /
                 * glGetTexImage of an sRGB-encoded attachment return the raw
                 * encoded bytes.  (The CTS packed-pixels model has no sRGB transfer
                 * at all: KHR-GL46.packed_pixels.rectangle.srgb8 /
                 * pbo_rectangle.srgb8 expect the verbatim byte values for
                 * GL_FLOAT clients; KHR-GL46.texture_size_promotion.functional
                 * passes without the decode too — its sRGB sub-cases were fixed by
                 * the r52 texelFetch-through-sample() and FRAMEBUFFER_SRGB view
                 * pieces, not by decoding reads.) */
                const bool srgb_read = false;
                CompVal src[4];
                for (int k = 0; k < 4; k++)
                {
                    if (kind == MGL_FB_KIND_FLOAT)
                    {
                        float f = ((float *)color)[idx * 4 + k];
                        /* GL 4.6 18.2: reading a signed-normalized color buffer
                         * clamps the values to [0, 1] (unnormalized reads are
                         * unaffected; getTexImage is not clamped either) */
                        if (fi && fi->component_type == GL_SIGNED_NORMALIZED)
                            f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
                        /* GL 4.6 18.2.2 / ES3 4.3.2: reading an sRGB-encoded color
                         * buffer converts to linear, unless the client wants the
                         * bytes themselves (UNORM-normalized client types pass
                         * through unchanged, e.g. GL_RGBA/GL_UNSIGNED_BYTE) */
                        else if (srgb_read)
                            f = (f <= 0.04045f) ? f / 12.92f : powf((f + 0.055f) / 1.055f, 2.4f);
                        src[k].is_int = false;
                        src[k].d = has[k] ? f : (k == 3 ? 1.0 : 0.0);
                    }
                    else
                    {
                        src[k].is_int = true;
                        if (kind == MGL_FB_KIND_INT) src[k].i = ((int32_t *)color)[idx * 4 + k];
                        else src[k].i = ((uint32_t *)color)[idx * 4 + k];
                        if (!has[k]) src[k].i = (k == 3) ? 1 : 0;
                    }
                }
                for (GLuint k = 0; k < ncomp; k++) cv[k] = src[map[k]];
            }
            else if (format == GL_DEPTH_COMPONENT)
            {
                cv[0].d = depthv[idx];
            }
            else if (format == GL_STENCIL_INDEX)
            {
                cv[0].is_int = true;
                cv[0].i = stencilv[idx] & 0xff;
            }
            else
            {
                cv[0].d = depthv[idx];
                cv[1].is_int = true;
                cv[1].i = stencilv[idx] & 0xff;
            }

            if (packed)
                storePacked(o, type, &pt, cv, integer, swap);
            else
                for (GLuint k = 0; k < ncomp; k++)
                    storeComponent(o + k * csz, type, cv[k], integer, swap);
            o += lay.group;
        }

        size_t off = lay.skip + (size_t)r * lay.stride;
        size_t len = lay.group * (size_t)width;
        if (pbo)
            fbm(ctx)->write_buffer(ctx, pbo, pbo_offset + off, len, rowbuf);
        else
            memcpy((uint8_t *)pixels + off, rowbuf, len);
    }

    free(rowbuf);
    free(color);
    free(depthv);
    free(stencilv);
}

void mglReadnPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data)
{
    if (bufSize < 0) FB_ERROR(GL_INVALID_VALUE);
    mgl_fb_read_pixels(ctx, x, y, width, height, format, type, bufSize, data);
}

#pragma mark blit validation

void mgl_fb_blit(GLMContext ctx, Framebuffer *readfbo, Framebuffer *drawfbo, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter)
{
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
        FB_ERROR(GL_INVALID_VALUE);
    if (filter != GL_NEAREST && filter != GL_LINEAR)
        FB_ERROR(GL_INVALID_ENUM);
    if (filter == GL_LINEAR && (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)))
        FB_ERROR(GL_INVALID_OPERATION);

    /* GL 4.6 §18.3.1: "the formats of the source and destination depth and stencil
     * buffers must match" — an INVALID_OPERATION is generated when DEPTH_BUFFER_BIT
     * (resp. STENCIL_BUFFER_BIT) is set and the two framebuffers' depth (stencil)
     * attachments have different internal formats.  A missing attachment on either
     * side is a silent no-op of the aspect (KHR-GL46 color-config blits COLOR|DEPTH|
     * STENCIL from a color-only read FB), and a multisample source is a RESOLVE —
     * depth formats may convert there (KHR-GL46 depth-config blits into the CTS's
     * D32FS8 window).  compat r99: dEQP ES31 debug.negative_coverage
     * get_error.buffer.blit_framebuffer blits DEPTH32F_STENCIL8 -> DEPTH24_STENCIL8
     * and expects the error. */
    if (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
    {
        for (int is_depth = 1; is_depth >= 0; is_depth--)
        {
            GLbitfield bit = is_depth ? GL_DEPTH_BUFFER_BIT : GL_STENCIL_BUFFER_BIT;
            if (!(mask & bit)) continue;
            const FBOAttachment *ra = is_depth ? &readfbo->depth : &readfbo->stencil;
            const FBOAttachment *da = is_depth ? &drawfbo->depth : &drawfbo->stencil;
            const Texture *rt = ra->textarget == GL_RENDERBUFFER ? ra->buf.rbo->tex : ra->buf.tex;
            const Texture *dt = da->textarget == GL_RENDERBUFFER ? da->buf.rbo->tex : da->buf.tex;
            if (!rt || !dt) continue;                 /* aspect missing: no-op */
            {
                extern GLsizei mgl_fb_attachment_samples(GLMContext ctx, FBOAttachment *att);
                if (mgl_fb_attachment_samples(ctx, (FBOAttachment *)ra) > 1 ||
                    mgl_fb_attachment_samples(ctx, (FBOAttachment *)da) > 1)
                    continue;                         /* multisample: a resolve */
            }
            if (rt->internalformat != dt->internalformat)
            {
                if (getenv("MGL_BLIT_DEBUG"))
                    fprintf(stderr, "MGL blit ds-mismatch: %s read=0x%x draw=0x%x\n",
                            is_depth ? "depth" : "stencil", rt->internalformat, dt->internalformat);
                FB_ERROR(GL_INVALID_OPERATION);
            }
        }
    }

    if (mgl_fb_status(ctx, readfbo) != GL_FRAMEBUFFER_COMPLETE ||
        mgl_fb_status(ctx, drawfbo) != GL_FRAMEBUFFER_COMPLETE)
        FB_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    /* conditional rendering (queries domain) */
    if (!mgl_queries_render_allowed(ctx))
        return;

    GLint rs = mgl_fb_samples(ctx, readfbo), ds = mgl_fb_samples(ctx, drawfbo);
    long sw = labs((long)srcX1 - srcX0), sh = labs((long)srcY1 - srcY0);
    long dw = labs((long)dstX1 - dstX0), dh = labs((long)dstY1 - dstY0);

    if (rs > 0 && ds > 0 && rs != ds)
        FB_ERROR(GL_INVALID_OPERATION);
    /* GL 4.6 18.3.1: identical rectangles are required when the read framebuffer is
     * multisampled (or both are); a single-sampled read framebuffer may be blitted
     * into a multisampled draw framebuffer with a scaled/partial region */
    if (rs > 0 && (sw != dw || sh != dh))
        FB_ERROR(GL_INVALID_OPERATION);

    GLbitfield eff = 0;

    if (mask & GL_COLOR_BUFFER_BIT)
    {
        GLenum att;
        const MGLFBFormatInfo *sfi = NULL;
        if (readColorSource(ctx, readfbo, &att, &sfi))
        {
            int sk = kindForFormatInfo(sfi);
            MGLFBDrawState *dst = mgl_fb_draw_state(ctx, drawfbo);
            bool any = false;
            for (GLsizei i = 0; i < dst->num_draw_buffers && i < MAX_COLOR_ATTACHMENTS; i++)
            {
                int dk = mgl_fb_slot_kind(ctx, drawfbo, i);
                if (dk < 0) continue;
                any = true;
                /* GL 4.6 18.3.1: format conversion is unsupported only across
                 * the int/uint/fixed-or-float class boundary (the check below).
                 * Desktop GL has NO exact-component match rule for multisample
                 * resolve blits (that is an ES 3.0 restriction); an R8 MS read
                 * buffer resolves into an RGBA8 default framebuffer
                 * (KHR-GL46.framebuffer_blit.multisampled_to_singlesampled_
                 * blit_color_config_test). */
                if ((sk == MGL_FB_KIND_FLOAT) != (dk == MGL_FB_KIND_FLOAT))
                    FB_ERROR(GL_INVALID_OPERATION);
            }
            if (sk != MGL_FB_KIND_FLOAT && filter == GL_LINEAR)
                FB_ERROR(GL_INVALID_OPERATION);
            if (any) eff |= GL_COLOR_BUFFER_BIT;
        }
    }

    if (mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
    {
        /* GL 4.6 18.3.1 depth/stencil bits:
         *  - the READ side having the aspect while the draw side does not is an
         *    INVALID_OPERATION ("formats do not match" — a missing buffer has no
         *    format); the reverse (read side missing the aspect) is a silent
         *    no-op for that aspect, which the KHR-GL46 color-config test relies
         *    on when it blits COLOR|DEPTH|STENCIL from a color-only multisampled
         *    texture.
         *  - for single-sample blits the source and destination depth AND
         *    stencil formats must match as a pair (the ES3 negative_api test
         *    blits DEPTH and STENCIL separately between D32FS8 and D24S8 and
         *    expects an error both times — NOT implemented: it contradicts
         *    KHR-GL46 framebuffer_blit.scissor_blit, which blits D32FS8
         *    renderbuffers into D24S8 renderbuffers and requires success;
         *    the classes below deliberately treat the two formats as equal,
         *    which also matches desktop GL on macOS.  dEQP
         *    debug.negative_coverage.get_error.buffer.blit_framebuffer is a
         *    known 1-case gap).  Multisample resolves may convert: MGL stores
         *    GL_DEPTH24_STENCIL8 and GL_DEPTH32F_STENCIL8 in the same Metal
         *    format, differing formats go through the GPU staging path in
         *    MGLRenderer+Framebuffer.m, and the KHR-GL46 tests blit D24S8
         *    multisampled sources into the CTS's D32FS8 default framebuffer
         *    and require success. */
        bool rd = mgl_fb_has_depth(ctx, readfbo), dd = mgl_fb_has_depth(ctx, drawfbo);
        bool rss = mgl_fb_has_stencil(ctx, readfbo), dss = mgl_fb_has_stencil(ctx, drawfbo);
        bool cmpD = (mask & GL_DEPTH_BUFFER_BIT) && rd && dd;
        bool cmpS = (mask & GL_STENCIL_BUFFER_BIT) && rss && dss;
        if ((mask & GL_DEPTH_BUFFER_BIT) && rd && !dd) FB_ERROR(GL_INVALID_OPERATION);
        if ((mask & GL_STENCIL_BUFFER_BIT) && rss && !dss) FB_ERROR(GL_INVALID_OPERATION);
        if ((cmpD || cmpS) && rs == 0)
        {
            GLuint a, b, as, bs;
            dsClass(ctx, readfbo, false, &a); dsClass(ctx, readfbo, true, &as);
            dsClass(ctx, drawfbo, false, &b); dsClass(ctx, drawfbo, true, &bs);
            if ((a | as) != (b | bs)) FB_ERROR(GL_INVALID_OPERATION);
        }
        if (cmpD) eff |= GL_DEPTH_BUFFER_BIT;
        if (cmpS) eff |= GL_STENCIL_BUFFER_BIT;
    }

    if (!eff || sw == 0 || sh == 0 || dw == 0 || dh == 0)
        return;

    if (!fbm(ctx) || !fbm(ctx)->blit)
        return;

    MGLFBBlitRequest req;
    req.read_fbo = readfbo; req.draw_fbo = drawfbo;
    req.src[0] = srcX0; req.src[1] = srcY0; req.src[2] = srcX1; req.src[3] = srcY1;
    req.dst[0] = dstX0; req.dst[1] = dstY0; req.dst[2] = dstX1; req.dst[3] = dstY1;
    req.mask = eff;
    req.filter = filter;
    fbm(ctx)->blit(ctx, &req);
}

#pragma mark multisample queries

void mglGetMultisamplefv(GLMContext ctx, GLenum pname, GLuint index, GLfloat *val)
{
    if (pname != GL_SAMPLE_POSITION)
        FB_ERROR(GL_INVALID_ENUM);

    GLint samples = mgl_fb_samples(ctx, ctx->state.framebuffer);
    if ((GLint)index >= samples || samples <= 0)
        FB_ERROR(GL_INVALID_VALUE);

    if (!val)
        return;

    if (!fbm(ctx) || !fbm(ctx)->sample_position || !fbm(ctx)->sample_position(ctx, samples, index, val))
    {
        val[0] = 0.5f;
        val[1] = 0.5f;
    }
}

#pragma mark - domain 'fb' hooks

void mgl_ext_init_fb(GLMContext ctx)
{
    if (ctx->state.ext_fb)
        return;
    ctx->state.ext_fb = (struct MGLExtState_fb *)calloc(1, sizeof(struct MGLExtState_fb));
    MGLFBDrawState *ds = &ctx->state.ext_fb->default_ds;
    memset(ds, 0, sizeof(*ds));
    ds->draw_buffers[0] = STATE(draw_buffer) ? STATE(draw_buffer) : GL_FRONT;
    ds->num_draw_buffers = 1;
    ds->read_buffer = STATE(read_buffer) ? STATE(read_buffer) : GL_FRONT;
    if (ds->read_buffer < GL_FRONT_LEFT || ds->read_buffer > GL_FRONT_AND_BACK)
        ds->read_buffer = GL_FRONT;
    if (ds->draw_buffers[0] < GL_FRONT_LEFT || ds->draw_buffers[0] > GL_FRONT_AND_BACK)
        ds->draw_buffers[0] = GL_FRONT;
}

bool mgl_ext_get_fb(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    GLuint maxdb = STATE_VAR(max_draw_buffers) ? STATE_VAR(max_draw_buffers) : MAX_COLOR_ATTACHMENTS;

    if (pname >= GL_DRAW_BUFFER0 && pname <= GL_DRAW_BUFFER15)
    {
        GLuint i = pname - GL_DRAW_BUFFER0;
        if (i >= maxdb) return false;
        MGLFBDrawState *ds = mgl_fb_draw_state(ctx, ctx->state.framebuffer);
        mgl_get_store(type, data, 0, (i < MAX_COLOR_ATTACHMENTS && (GLsizei)i < ds->num_draw_buffers) ? ds->draw_buffers[i] : GL_NONE);
        return true;
    }

    switch(pname)
    {
        case GL_SAMPLES:
            mgl_get_store(type, data, 0, mgl_fb_samples(ctx, ctx->state.framebuffer));
            return true;
        case GL_SAMPLE_BUFFERS:
            mgl_get_store(type, data, 0, mgl_fb_samples(ctx, ctx->state.framebuffer) > 0 ? 1 : 0);
            return true;
        case GL_MAX_SAMPLES:
            mgl_get_store(type, data, 0, (fbm(ctx) && fbm(ctx)->max_samples) ? fbm(ctx)->max_samples : 4);
            return true;
        case GL_DOUBLEBUFFER:
            mgl_get_store(type, data, 0, 1);
            return true;
        case GL_STEREO:
            mgl_get_store(type, data, 0, 0);
            return true;
        case GL_IMPLEMENTATION_COLOR_READ_FORMAT:
        case GL_IMPLEMENTATION_COLOR_READ_TYPE:
        {
            GLenum f, t;
            mgl_fb_color_read_format(ctx, ctx->state.readbuffer, &f, &t);
            mgl_get_store(type, data, 0, pname == GL_IMPLEMENTATION_COLOR_READ_FORMAT ? f : t);
            return true;
        }
    }
    return false;
}
