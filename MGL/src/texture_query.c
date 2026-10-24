/*
 * texture_query.c
 * MGL — texture level / internalformat queries, texparam domain hooks
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * GL-visible format properties (component sizes/types, compression blocks,
 * image/view classes, natural pixel transfer format) come from the format
 * table below; whether a format is supported and what it can do on this GPU
 * (render, blend, filter, MSAA sample counts, shader write/atomics) comes
 * from the Metal pixel format mapping (mtlFormatForGLInternalFormat) and
 * MTLDevice capabilities via ctx->mtl_funcs.ext_texparam->formatCaps.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "mgl.h"
#include "glm_context.h"
#include "pixel_utils.h"
#include "mgl_ext_texparam.h"
#include "tex_param_internal.h"

extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *newTexObj(GLMContext ctx, GLenum target);

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define TQ_ERROR(_e_) do { ERROR_RETURN(_e_); return; } while (0)

#pragma mark format table

#define UN GL_UNSIGNED_NORMALIZED
#define SN GL_SIGNED_NORMALIZED
#define FL GL_FLOAT
#define SI GL_INT
#define UI GL_UNSIGNED_INT
#define NO GL_NONE

/*      fmt                      base                r  g  b  a  d  s  sh type srgb cmp bw bh bb  img_class                 view_class               pix_format          pix_type                    preferred */
#define C(f, base, r, g, b, a, type, img, view, pf, pt) \
    { f, base, r, g, b, a, 0, 0, 0, type, GL_FALSE, GL_FALSE, 0, 0, 0, img, view, pf, pt, f }
#define CS(f, base, r, g, b, a, view, pf, pt) \
    { f, base, r, g, b, a, 0, 0, 0, UN, GL_TRUE, GL_FALSE, 0, 0, 0, NO, view, pf, pt, f }
#define DS(f, base, d, s, type, pf, pt) \
    { f, base, 0, 0, 0, 0, d, s, 0, type, GL_FALSE, GL_FALSE, 0, 0, 0, NO, NO, pf, pt, f }
#define CMP(f, base, r, g, b, a, type, srgb, bb, view, pf) \
    { f, base, r, g, b, a, 0, 0, 0, type, srgb, GL_TRUE, 4, 4, bb, NO, view, pf, GL_UNSIGNED_BYTE, f }
#define GEN(f, pref) \
    { f, 0, 0, 0, 0, 0, 0, 0, 0, NO, GL_FALSE, GL_FALSE, 0, 0, 0, NO, NO, 0, 0, pref }

static const MGLFormatDesc mgl_formats[] = {
    /* unsigned / signed normalized */
    C(GL_R8,            GL_RED,  8, 0, 0, 0, UN, GL_IMAGE_CLASS_1_X_8,  GL_VIEW_CLASS_8_BITS,  GL_RED,  GL_UNSIGNED_BYTE),
    C(GL_R8_SNORM,      GL_RED,  8, 0, 0, 0, SN, GL_IMAGE_CLASS_1_X_8,  GL_VIEW_CLASS_8_BITS,  GL_RED,  GL_BYTE),
    C(GL_R16,           GL_RED, 16, 0, 0, 0, UN, GL_IMAGE_CLASS_1_X_16, GL_VIEW_CLASS_16_BITS, GL_RED,  GL_UNSIGNED_SHORT),
    C(GL_R16_SNORM,     GL_RED, 16, 0, 0, 0, SN, GL_IMAGE_CLASS_1_X_16, GL_VIEW_CLASS_16_BITS, GL_RED,  GL_SHORT),
    C(GL_RG8,           GL_RG,   8, 8, 0, 0, UN, GL_IMAGE_CLASS_2_X_8,  GL_VIEW_CLASS_16_BITS, GL_RG,   GL_UNSIGNED_BYTE),
    C(GL_RG8_SNORM,     GL_RG,   8, 8, 0, 0, SN, GL_IMAGE_CLASS_2_X_8,  GL_VIEW_CLASS_16_BITS, GL_RG,   GL_BYTE),
    C(GL_RG16,          GL_RG,  16,16, 0, 0, UN, GL_IMAGE_CLASS_2_X_16, GL_VIEW_CLASS_32_BITS, GL_RG,   GL_UNSIGNED_SHORT),
    C(GL_RG16_SNORM,    GL_RG,  16,16, 0, 0, SN, GL_IMAGE_CLASS_2_X_16, GL_VIEW_CLASS_32_BITS, GL_RG,   GL_SHORT),
    C(GL_R3_G3_B2,      GL_RGB,  3, 3, 2, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_BYTE_3_3_2),
    C(GL_RGB4,          GL_RGB,  4, 4, 4, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_BYTE),
    C(GL_RGB5,          GL_RGB,  5, 5, 5, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_BYTE),
    C(GL_RGB565,        GL_RGB,  5, 6, 5, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_SHORT_5_6_5),
    C(GL_RGB8,          GL_RGB,  8, 8, 8, 0, UN, NO, GL_VIEW_CLASS_24_BITS, GL_RGB,  GL_UNSIGNED_BYTE),
    C(GL_RGB8_SNORM,    GL_RGB,  8, 8, 8, 0, SN, NO, GL_VIEW_CLASS_24_BITS, GL_RGB,  GL_BYTE),
    C(GL_RGB10,         GL_RGB, 10,10,10, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_SHORT),
    C(GL_RGB12,         GL_RGB, 12,12,12, 0, UN, NO, NO,                    GL_RGB,  GL_UNSIGNED_SHORT),
    C(GL_RGB16,         GL_RGB, 16,16,16, 0, UN, NO, GL_VIEW_CLASS_48_BITS, GL_RGB,  GL_UNSIGNED_SHORT),
    C(GL_RGB16_SNORM,   GL_RGB, 16,16,16, 0, SN, NO, GL_VIEW_CLASS_48_BITS, GL_RGB,  GL_SHORT),
    C(GL_RGBA2,         GL_RGBA, 2, 2, 2, 2, UN, NO, NO,                    GL_RGBA, GL_UNSIGNED_BYTE),
    C(GL_RGBA4,         GL_RGBA, 4, 4, 4, 4, UN, NO, NO,                    GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4),
    C(GL_RGB5_A1,       GL_RGBA, 5, 5, 5, 1, UN, NO, NO,                    GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1),
    C(GL_RGBA8,         GL_RGBA, 8, 8, 8, 8, UN, GL_IMAGE_CLASS_4_X_8,  GL_VIEW_CLASS_32_BITS, GL_RGBA, GL_UNSIGNED_BYTE),
    C(GL_RGBA8_SNORM,   GL_RGBA, 8, 8, 8, 8, SN, GL_IMAGE_CLASS_4_X_8,  GL_VIEW_CLASS_32_BITS, GL_RGBA, GL_BYTE),
    C(GL_RGB10_A2,      GL_RGBA,10,10,10, 2, UN, GL_IMAGE_CLASS_10_10_10_2, GL_VIEW_CLASS_32_BITS, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV),
    C(GL_RGB10_A2UI,    GL_RGBA,10,10,10, 2, UI, GL_IMAGE_CLASS_10_10_10_2, GL_VIEW_CLASS_32_BITS, GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV),
    C(GL_RGBA12,        GL_RGBA,12,12,12,12, UN, NO, NO,                    GL_RGBA, GL_UNSIGNED_SHORT),
    C(GL_RGBA16,        GL_RGBA,16,16,16,16, UN, GL_IMAGE_CLASS_4_X_16, GL_VIEW_CLASS_64_BITS, GL_RGBA, GL_UNSIGNED_SHORT),
    C(GL_RGBA16_SNORM,  GL_RGBA,16,16,16,16, SN, GL_IMAGE_CLASS_4_X_16, GL_VIEW_CLASS_64_BITS, GL_RGBA, GL_SHORT),
    CS(GL_SRGB8,        GL_RGB,  8, 8, 8, 0, GL_VIEW_CLASS_24_BITS, GL_RGB,  GL_UNSIGNED_BYTE),
    CS(GL_SRGB8_ALPHA8, GL_RGBA, 8, 8, 8, 8, GL_VIEW_CLASS_32_BITS, GL_RGBA, GL_UNSIGNED_BYTE),

    /* float */
    C(GL_R16F,          GL_RED, 16, 0, 0, 0, FL, GL_IMAGE_CLASS_1_X_16, GL_VIEW_CLASS_16_BITS,  GL_RED,  GL_HALF_FLOAT),
    C(GL_RG16F,         GL_RG,  16,16, 0, 0, FL, GL_IMAGE_CLASS_2_X_16, GL_VIEW_CLASS_32_BITS,  GL_RG,   GL_HALF_FLOAT),
    C(GL_RGB16F,        GL_RGB, 16,16,16, 0, FL, NO,                    GL_VIEW_CLASS_48_BITS,  GL_RGB,  GL_HALF_FLOAT),
    C(GL_RGBA16F,       GL_RGBA,16,16,16,16, FL, GL_IMAGE_CLASS_4_X_16, GL_VIEW_CLASS_64_BITS,  GL_RGBA, GL_HALF_FLOAT),
    C(GL_R32F,          GL_RED, 32, 0, 0, 0, FL, GL_IMAGE_CLASS_1_X_32, GL_VIEW_CLASS_32_BITS,  GL_RED,  GL_FLOAT),
    C(GL_RG32F,         GL_RG,  32,32, 0, 0, FL, GL_IMAGE_CLASS_2_X_32, GL_VIEW_CLASS_64_BITS,  GL_RG,   GL_FLOAT),
    C(GL_RGB32F,        GL_RGB, 32,32,32, 0, FL, NO,                    GL_VIEW_CLASS_96_BITS,  GL_RGB,  GL_FLOAT),
    C(GL_RGBA32F,       GL_RGBA,32,32,32,32, FL, GL_IMAGE_CLASS_4_X_32, GL_VIEW_CLASS_128_BITS, GL_RGBA, GL_FLOAT),
    C(GL_R11F_G11F_B10F,GL_RGB, 11,11,10, 0, FL, GL_IMAGE_CLASS_11_11_10, GL_VIEW_CLASS_32_BITS, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV),
    { GL_RGB9_E5, GL_RGB, 9, 9, 9, 0, 0, 0, 5, FL, GL_FALSE, GL_FALSE, 0, 0, 0, NO, GL_VIEW_CLASS_32_BITS, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, GL_RGB9_E5 },

    /* integer */
    C(GL_R8I,     GL_RED,  8, 0, 0, 0, SI, GL_IMAGE_CLASS_1_X_8,  GL_VIEW_CLASS_8_BITS,   GL_RED_INTEGER,  GL_BYTE),
    C(GL_R8UI,    GL_RED,  8, 0, 0, 0, UI, GL_IMAGE_CLASS_1_X_8,  GL_VIEW_CLASS_8_BITS,   GL_RED_INTEGER,  GL_UNSIGNED_BYTE),
    C(GL_R16I,    GL_RED, 16, 0, 0, 0, SI, GL_IMAGE_CLASS_1_X_16, GL_VIEW_CLASS_16_BITS,  GL_RED_INTEGER,  GL_SHORT),
    C(GL_R16UI,   GL_RED, 16, 0, 0, 0, UI, GL_IMAGE_CLASS_1_X_16, GL_VIEW_CLASS_16_BITS,  GL_RED_INTEGER,  GL_UNSIGNED_SHORT),
    C(GL_R32I,    GL_RED, 32, 0, 0, 0, SI, GL_IMAGE_CLASS_1_X_32, GL_VIEW_CLASS_32_BITS,  GL_RED_INTEGER,  GL_INT),
    C(GL_R32UI,   GL_RED, 32, 0, 0, 0, UI, GL_IMAGE_CLASS_1_X_32, GL_VIEW_CLASS_32_BITS,  GL_RED_INTEGER,  GL_UNSIGNED_INT),
    C(GL_RG8I,    GL_RG,   8, 8, 0, 0, SI, GL_IMAGE_CLASS_2_X_8,  GL_VIEW_CLASS_16_BITS,  GL_RG_INTEGER,   GL_BYTE),
    C(GL_RG8UI,   GL_RG,   8, 8, 0, 0, UI, GL_IMAGE_CLASS_2_X_8,  GL_VIEW_CLASS_16_BITS,  GL_RG_INTEGER,   GL_UNSIGNED_BYTE),
    C(GL_RG16I,   GL_RG,  16,16, 0, 0, SI, GL_IMAGE_CLASS_2_X_16, GL_VIEW_CLASS_32_BITS,  GL_RG_INTEGER,   GL_SHORT),
    C(GL_RG16UI,  GL_RG,  16,16, 0, 0, UI, GL_IMAGE_CLASS_2_X_16, GL_VIEW_CLASS_32_BITS,  GL_RG_INTEGER,   GL_UNSIGNED_SHORT),
    C(GL_RG32I,   GL_RG,  32,32, 0, 0, SI, GL_IMAGE_CLASS_2_X_32, GL_VIEW_CLASS_64_BITS,  GL_RG_INTEGER,   GL_INT),
    C(GL_RG32UI,  GL_RG,  32,32, 0, 0, UI, GL_IMAGE_CLASS_2_X_32, GL_VIEW_CLASS_64_BITS,  GL_RG_INTEGER,   GL_UNSIGNED_INT),
    C(GL_RGB8I,   GL_RGB,  8, 8, 8, 0, SI, NO,                    GL_VIEW_CLASS_24_BITS,  GL_RGB_INTEGER,  GL_BYTE),
    C(GL_RGB8UI,  GL_RGB,  8, 8, 8, 0, UI, NO,                    GL_VIEW_CLASS_24_BITS,  GL_RGB_INTEGER,  GL_UNSIGNED_BYTE),
    C(GL_RGB16I,  GL_RGB, 16,16,16, 0, SI, NO,                    GL_VIEW_CLASS_48_BITS,  GL_RGB_INTEGER,  GL_SHORT),
    C(GL_RGB16UI, GL_RGB, 16,16,16, 0, UI, NO,                    GL_VIEW_CLASS_48_BITS,  GL_RGB_INTEGER,  GL_UNSIGNED_SHORT),
    C(GL_RGB32I,  GL_RGB, 32,32,32, 0, SI, NO,                    GL_VIEW_CLASS_96_BITS,  GL_RGB_INTEGER,  GL_INT),
    C(GL_RGB32UI, GL_RGB, 32,32,32, 0, UI, NO,                    GL_VIEW_CLASS_96_BITS,  GL_RGB_INTEGER,  GL_UNSIGNED_INT),
    C(GL_RGBA8I,  GL_RGBA, 8, 8, 8, 8, SI, GL_IMAGE_CLASS_4_X_8,  GL_VIEW_CLASS_32_BITS,  GL_RGBA_INTEGER, GL_BYTE),
    C(GL_RGBA8UI, GL_RGBA, 8, 8, 8, 8, UI, GL_IMAGE_CLASS_4_X_8,  GL_VIEW_CLASS_32_BITS,  GL_RGBA_INTEGER, GL_UNSIGNED_BYTE),
    C(GL_RGBA16I, GL_RGBA,16,16,16,16, SI, GL_IMAGE_CLASS_4_X_16, GL_VIEW_CLASS_64_BITS,  GL_RGBA_INTEGER, GL_SHORT),
    C(GL_RGBA16UI,GL_RGBA,16,16,16,16, UI, GL_IMAGE_CLASS_4_X_16, GL_VIEW_CLASS_64_BITS,  GL_RGBA_INTEGER, GL_UNSIGNED_SHORT),
    C(GL_RGBA32I, GL_RGBA,32,32,32,32, SI, GL_IMAGE_CLASS_4_X_32, GL_VIEW_CLASS_128_BITS, GL_RGBA_INTEGER, GL_INT),
    C(GL_RGBA32UI,GL_RGBA,32,32,32,32, UI, GL_IMAGE_CLASS_4_X_32, GL_VIEW_CLASS_128_BITS, GL_RGBA_INTEGER, GL_UNSIGNED_INT),

    /* depth / stencil */
    DS(GL_DEPTH_COMPONENT16,  GL_DEPTH_COMPONENT, 16, 0, UN, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT),
    DS(GL_DEPTH_COMPONENT24,  GL_DEPTH_COMPONENT, 24, 0, UN, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT),
    DS(GL_DEPTH_COMPONENT32,  GL_DEPTH_COMPONENT, 32, 0, UN, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT),
    DS(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, 32, 0, FL, GL_DEPTH_COMPONENT, GL_FLOAT),
    DS(GL_DEPTH24_STENCIL8,   GL_DEPTH_STENCIL,   24, 8, UN, GL_DEPTH_STENCIL,   GL_UNSIGNED_INT_24_8),
    DS(GL_DEPTH32F_STENCIL8,  GL_DEPTH_STENCIL,   32, 8, FL, GL_DEPTH_STENCIL,   GL_FLOAT_32_UNSIGNED_INT_24_8_REV),
    DS(GL_STENCIL_INDEX1,     GL_STENCIL_INDEX,    0, 1, NO, GL_STENCIL_INDEX,   GL_UNSIGNED_BYTE),
    DS(GL_STENCIL_INDEX4,     GL_STENCIL_INDEX,    0, 4, NO, GL_STENCIL_INDEX,   GL_UNSIGNED_BYTE),
    DS(GL_STENCIL_INDEX8,     GL_STENCIL_INDEX,    0, 8, NO, GL_STENCIL_INDEX,   GL_UNSIGNED_BYTE),
    DS(GL_STENCIL_INDEX16,    GL_STENCIL_INDEX,    0,16, NO, GL_STENCIL_INDEX,   GL_UNSIGNED_SHORT),

    /* compressed */
    CMP(GL_COMPRESSED_RED_RGTC1,              GL_RED,  8, 0, 0, 0, UN, GL_FALSE,  8, GL_VIEW_CLASS_RGTC1_RED,   GL_RED),
    CMP(GL_COMPRESSED_SIGNED_RED_RGTC1,       GL_RED,  8, 0, 0, 0, SN, GL_FALSE,  8, GL_VIEW_CLASS_RGTC1_RED,   GL_RED),
    CMP(GL_COMPRESSED_RG_RGTC2,               GL_RG,   8, 8, 0, 0, UN, GL_FALSE, 16, GL_VIEW_CLASS_RGTC2_RG,    GL_RG),
    CMP(GL_COMPRESSED_SIGNED_RG_RGTC2,        GL_RG,   8, 8, 0, 0, SN, GL_FALSE, 16, GL_VIEW_CLASS_RGTC2_RG,    GL_RG),
    CMP(GL_COMPRESSED_RGBA_BPTC_UNORM,        GL_RGBA, 8, 8, 8, 8, UN, GL_FALSE, 16, GL_VIEW_CLASS_BPTC_UNORM,  GL_RGBA),
    CMP(GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,  GL_RGBA, 8, 8, 8, 8, UN, GL_TRUE,  16, GL_VIEW_CLASS_BPTC_UNORM,  GL_RGBA),
    CMP(GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,  GL_RGB, 16,16,16, 0, FL, GL_FALSE, 16, GL_VIEW_CLASS_BPTC_FLOAT,  GL_RGB),
    CMP(GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,GL_RGB, 16,16,16, 0, FL, GL_FALSE, 16, GL_VIEW_CLASS_BPTC_FLOAT,  GL_RGB),
    CMP(GL_COMPRESSED_RGB8_ETC2,              GL_RGB,  8, 8, 8, 0, UN, GL_FALSE,  8, GL_VIEW_CLASS_ETC2_RGB,    GL_RGB),
    CMP(GL_COMPRESSED_SRGB8_ETC2,             GL_RGB,  8, 8, 8, 0, UN, GL_TRUE,   8, GL_VIEW_CLASS_ETC2_RGB,    GL_RGB),
    CMP(GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,  GL_RGBA, 8, 8, 8, 1, UN, GL_FALSE, 8, GL_VIEW_CLASS_ETC2_RGBA, GL_RGBA),
    CMP(GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, GL_RGBA, 8, 8, 8, 1, UN, GL_TRUE,  8, GL_VIEW_CLASS_ETC2_RGBA, GL_RGBA),
    CMP(GL_COMPRESSED_RGBA8_ETC2_EAC,         GL_RGBA, 8, 8, 8, 8, UN, GL_FALSE, 16, GL_VIEW_CLASS_ETC2_EAC_RGBA, GL_RGBA),
    CMP(GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,  GL_RGBA, 8, 8, 8, 8, UN, GL_TRUE,  16, GL_VIEW_CLASS_ETC2_EAC_RGBA, GL_RGBA),
    CMP(GL_COMPRESSED_R11_EAC,                GL_RED, 11, 0, 0, 0, UN, GL_FALSE,  8, GL_VIEW_CLASS_EAC_R11,     GL_RED),
    CMP(GL_COMPRESSED_SIGNED_R11_EAC,         GL_RED, 11, 0, 0, 0, SN, GL_FALSE,  8, GL_VIEW_CLASS_EAC_R11,     GL_RED),
    CMP(GL_COMPRESSED_RG11_EAC,               GL_RG,  11,11, 0, 0, UN, GL_FALSE, 16, GL_VIEW_CLASS_EAC_RG11,    GL_RG),
    CMP(GL_COMPRESSED_SIGNED_RG11_EAC,        GL_RG,  11,11, 0, 0, SN, GL_FALSE, 16, GL_VIEW_CLASS_EAC_RG11,    GL_RG),

    /* unsized / generic: resolved to the preferred sized format */
    GEN(GL_RED,             GL_R8),
    GEN(GL_RG,              GL_RG8),
    GEN(GL_RGB,             GL_RGB8),
    GEN(GL_RGBA,            GL_RGBA8),
    GEN(GL_SRGB,            GL_SRGB8),
    GEN(GL_SRGB_ALPHA,      GL_SRGB8_ALPHA8),
    GEN(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32F),
    GEN(GL_DEPTH_STENCIL,   GL_DEPTH32F_STENCIL8),
    GEN(GL_STENCIL_INDEX,   GL_STENCIL_INDEX8),
    GEN(GL_COMPRESSED_RED,  GL_COMPRESSED_R11_EAC),
    GEN(GL_COMPRESSED_RG,   GL_COMPRESSED_RG11_EAC),
    GEN(GL_COMPRESSED_RGB,  GL_COMPRESSED_RGB8_ETC2),
    GEN(GL_COMPRESSED_RGBA, GL_COMPRESSED_RGBA8_ETC2_EAC),
    GEN(GL_COMPRESSED_SRGB, GL_COMPRESSED_SRGB8_ETC2),
    GEN(GL_COMPRESSED_SRGB_ALPHA, GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC),
};

#undef C
#undef CS
#undef DS
#undef CMP
#undef GEN

static const MGLFormatDesc *tq_raw_desc(GLenum internalformat)
{
    for (size_t i = 0; i < sizeof(mgl_formats) / sizeof(mgl_formats[0]); i++)
    {
        if (mgl_formats[i].fmt == internalformat)
            return &mgl_formats[i];
    }
    return NULL;
}

/* resolved (sized) description; generic formats resolve to their preferred sized format */
const MGLFormatDesc *mglTexParamFormatDesc(GLenum internalformat)
{
    const MGLFormatDesc *d = tq_raw_desc(internalformat);

    if (d && d->base == 0)
        d = tq_raw_desc(d->preferred);

    return d;
}

static bool tq_is_integer(const MGLFormatDesc *d)
{
    return d->type == GL_INT || d->type == GL_UNSIGNED_INT;
}

static bool tq_has_color(const MGLFormatDesc *d)
{
    return d->base == GL_RED || d->base == GL_RG || d->base == GL_RGB || d->base == GL_RGBA;
}

#pragma mark extension state / records

void mgl_ext_init_texparam(GLMContext ctx)
{
    if (ctx->state.ext_texparam)
        return;

    ctx->state.ext_texparam = (struct MGLExtState_texparam *)calloc(1, sizeof(struct MGLExtState_texparam));
}

MGLTexParamRecord *mgl_texparam_record(GLMContext ctx, GLuint name, bool create)
{
    struct MGLExtState_texparam *st = ctx->state.ext_texparam;

    if (!st || name == 0)
        return NULL;

    /* Guard the array growth below:
     *  - TEX_OBJ_RES_NAME (0xcafebeef) and friends are internal sentinel names
     *    (default texture / renderbuffer objects), never real GL names;
     *  - real names are small sequential integers, so cap the record array.
     * Without this, "while (n <= name) n *= 2" wraps a GLuint for names
     * >= 2^31 and spins forever (seen as a hang on glTexBuffer() against the
     * default GL_TEXTURE_BUFFER object, which the CTS state reset does). */
    if (name >= TEX_OBJ_RES_NAME || name >= MGL_TEXPARAM_MAX_RECORDS)
        return NULL;

    if (name >= st->num_records)
    {
        if (!create)
            return NULL;

        /* uint64: n *= 2 on a GLuint wraps to 0 for names with the top bit set
         * and looped forever; and CTS-style sessions allocate millions of
         * object names - the name-indexed array must not follow them into
         * gigabytes (2.6 GB RSS + minutes per glTexBuffer during
         * glu::resetStateGLCore). Above the cap the query paths fall back to
         * their spec defaults. */
        uint64_t n = st->num_records ? st->num_records : 64;
        while (n <= name)
            n *= 2;
        if (n > MGL_TEXPARAM_MAX_RECORDS)
            n = MGL_TEXPARAM_MAX_RECORDS;
        GLuint cap = (GLuint)n;

        MGLTexParamRecord *r = (MGLTexParamRecord *)realloc(st->records, cap * sizeof(MGLTexParamRecord));
        if (!r)
            return NULL;

        bzero(r + st->num_records, (cap - st->num_records) * sizeof(MGLTexParamRecord));
        st->records = r;
        st->num_records = cap;
    }

    MGLTexParamRecord *rec = &st->records[name];

    if (!rec->valid)
    {
        if (!create)
            return NULL;

        bzero(rec, sizeof(*rec));
        rec->valid = GL_TRUE;
        rec->fixed_sample_locations = GL_TRUE;
        rec->buffer_size = -1;
    }

    return rec;
}

void mgl_texparam_record_view(GLMContext ctx, Texture *tex, GLuint min_level, GLuint num_levels, GLuint min_layer, GLuint num_layers, GLuint immutable_levels)
{
    MGLTexParamRecord *rec = tex ? mgl_texparam_record(ctx, tex->name, true) : NULL;

    if (!rec)
        return;

    rec->is_view = GL_TRUE;
    rec->view_min_level = min_level;
    rec->view_num_levels = num_levels;
    rec->view_min_layer = min_layer;
    rec->view_num_layers = num_layers;
    rec->view_immutable_levels = immutable_levels;
}

void mgl_texparam_record_buffer(GLMContext ctx, Texture *tex, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    MGLTexParamRecord *rec = tex ? mgl_texparam_record(ctx, tex->name, true) : NULL;

    if (!rec)
        return;

    rec->is_buffer = buffer != 0;
    rec->buffer = buffer;
    rec->buffer_offset = buffer ? offset : 0;
    rec->buffer_size = buffer ? size : 0;
}

void mgl_texparam_record_multisample(GLMContext ctx, Texture *tex, GLsizei samples, GLboolean fixed_sample_locations)
{
    MGLTexParamRecord *rec = tex ? mgl_texparam_record(ctx, tex->name, true) : NULL;

    if (!rec)
        return;

    rec->samples = samples;
    rec->fixed_sample_locations = fixed_sample_locations ? GL_TRUE : GL_FALSE;
}

static int tq_proxy_index(GLenum proxy_target, GLenum *target)
{
    static const GLenum proxies[][2] = {
        { GL_PROXY_TEXTURE_1D, GL_TEXTURE_1D },
        { GL_PROXY_TEXTURE_2D, GL_TEXTURE_2D },
        { GL_PROXY_TEXTURE_3D, GL_TEXTURE_3D },
        { GL_PROXY_TEXTURE_1D_ARRAY, GL_TEXTURE_1D_ARRAY },
        { GL_PROXY_TEXTURE_2D_ARRAY, GL_TEXTURE_2D_ARRAY },
        { GL_PROXY_TEXTURE_RECTANGLE, GL_TEXTURE_RECTANGLE },
        { GL_PROXY_TEXTURE_CUBE_MAP, GL_TEXTURE_CUBE_MAP },
        { GL_PROXY_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_CUBE_MAP_ARRAY },
        { GL_PROXY_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_2D_MULTISAMPLE },
        { GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY, GL_TEXTURE_2D_MULTISAMPLE_ARRAY },
    };

    for (int i = 0; i < (int)(sizeof(proxies) / sizeof(proxies[0])); i++)
    {
        if (proxies[i][0] == proxy_target)
        {
            *target = proxies[i][1];
            return i;
        }
    }
    return -1;
}

Texture *mgl_texparam_proxy_texture(GLMContext ctx, GLenum proxy_target)
{
    struct MGLExtState_texparam *st = ctx->state.ext_texparam;
    GLenum target;
    int idx = tq_proxy_index(proxy_target, &target);

    if (!st || idx < 0)
        return NULL;

    if (!st->proxy[idx])
    {
        /* proxy state lives in an object nobody can bind; its levels start out empty */
        st->proxy[idx] = newTexObj(ctx, target);
        if (st->proxy[idx])
            st->proxy[idx]->name = 0;
    }

    return st->proxy[idx];
}

bool mgl_ext_get_texparam(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname)
    {
        case GL_MAX_TEXTURE_MAX_ANISOTROPY:
            /* MTLSamplerDescriptor.maxAnisotropy range is 1..16 */
            mgl_get_store(type, data, 0, 16.0);
            return true;

        case GL_SAMPLER_BINDING:
        {
            GLuint unit = STATE(active_texture);
            Sampler *s = unit < TEXTURE_UNITS ? STATE(texture_samplers[unit]) : NULL;
            mgl_get_store(type, data, 0, s ? s->name : 0);
            return true;
        }
    }

    return false;
}

#pragma mark Metal helpers

static bool tq_mtl_info(GLMContext ctx, Texture *tex, MGLTexParamMtlInfo *info)
{
    bzero(info, sizeof(*info));

    if (!tex || !tex->mtl_data || !ctx->mtl_funcs.ext_texparam || !ctx->mtl_funcs.ext_texparam->textureInfo)
        return false;

    return ctx->mtl_funcs.ext_texparam->textureInfo(ctx, tex, info);
}

/* capabilities of the Metal pixel format backing internalformat; false if unsupported */
static bool tq_caps(GLMContext ctx, GLenum internalformat, MGLTexParamFormatCaps *caps)
{
    bzero(caps, sizeof(*caps));

    if (!tq_raw_desc(internalformat))
        return false;

    GLuint mtl = (GLuint)mtlFormatForGLInternalFormat(internalformat);
    if (mtl == 0)
        return false;

    if (ctx->mtl_funcs.ext_texparam && ctx->mtl_funcs.ext_texparam->formatCaps)
        return ctx->mtl_funcs.ext_texparam->formatCaps(ctx, mtl, caps);

    caps->supported = GL_TRUE;
    return true;
}

static GLuint tq_buffer_name_for_mtl(GLMContext ctx, void *mtl_buffer)
{
    HashTable *t = &STATE(buffer_table);

    if (!mtl_buffer)
        return 0;

    for (GLuint i = 1; i < t->size; i++)
    {
        Buffer *b = (Buffer *)t->keys[i].data;
        if (b && b->data.mtl_data == mtl_buffer)
            return i;
    }
    return 0;
}

#pragma mark level parameters

static bool tq_target_is_array(GLenum target)
{
    return target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_2D_ARRAY ||
           target == GL_TEXTURE_CUBE_MAP_ARRAY || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

static GLuint tq_ilog2(GLuint v)
{
    GLuint r = 0;
    while (v >>= 1) r++;
    return r;
}

void mglTexParamGetLevel(GLMContext ctx, Texture *tex, GLenum target, GLint level, GLenum pname, GLint *iparams, GLfloat *fparams, bool dsa)
{
    GLuint face = 0;
    GLint val = 0;
    GLuint max_size;

    if (!iparams && !fparams)
        TQ_ERROR(GL_INVALID_VALUE);

    if (target >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && target <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)
        face = target - GL_TEXTURE_CUBE_MAP_POSITIVE_X;

    GLenum tex_target = tex->target;

    max_size = STATE_VAR(max_texture_size);
    if (max_size == 0) max_size = 16384;
    if (tex_target == GL_TEXTURE_3D && STATE_VAR(max_3d_texture_size))
        max_size = STATE_VAR(max_3d_texture_size);

    if (level < 0 || (GLuint)level > tq_ilog2(max_size))
        TQ_ERROR(GL_INVALID_VALUE);

    /* levels > 0 are invalid for targets without mipmaps */
    if (level > 0 && (tex_target == GL_TEXTURE_RECTANGLE || tex_target == GL_TEXTURE_BUFFER ||
                      tex_target == GL_TEXTURE_2D_MULTISAMPLE || tex_target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY))
        TQ_ERROR(GL_INVALID_VALUE);

    /* the level image */
    TextureLevel *lvl = NULL;
    if (tex->faces[face].levels && (GLuint)level < tex->mipmap_levels && (GLuint)level < MAX(tex->num_levels, 1))
    {
        lvl = &tex->faces[face].levels[level];
        if (lvl->width == 0)
            lvl = NULL;
    }

    MGLTexParamRecord *rec = tex->name ? mgl_texparam_record(ctx, tex->name, false) : NULL;
    MGLTexParamMtlInfo info;
    bool have_info = tq_mtl_info(ctx, tex, &info);

    bool is_buffer = (tex_target == GL_TEXTURE_BUFFER);
    bool buffer_attached = is_buffer && ((rec && rec->is_buffer) || (have_info && info.has_buffer));
    bool defined = lvl != NULL || buffer_attached;

    GLenum ifmt = defined ? tex->internalformat : GL_RGBA;
    const MGLFormatDesc *d = defined ? mglTexParamFormatDesc(ifmt) : NULL;

    switch (pname)
    {
        case GL_TEXTURE_WIDTH:
            if (lvl) val = lvl->width;
            else if (buffer_attached)
            {
                GLsizeiptr size = 0;
                if (rec && rec->is_buffer)
                {
                    size = rec->buffer_size;
                    if (size < 0)
                    {
                        Buffer *b = (Buffer *)searchHashTable(&STATE(buffer_table), rec->buffer);
                        size = b ? b->size - rec->buffer_offset : 0;
                    }
                }
                else size = info.buffer_length;
                GLuint texel = d ? (d->r + d->g + d->b + d->a) / 8 : 0;
                val = texel ? (GLint)(size / texel) : 0;
            }
            break;

        case GL_TEXTURE_HEIGHT:
            if (lvl) val = (tex_target == GL_TEXTURE_1D) ? 1 : lvl->height;
            else if (buffer_attached) val = 1;
            break;

        case GL_TEXTURE_DEPTH:
            if (lvl)
            {
                if (tex_target == GL_TEXTURE_3D || (tq_target_is_array(tex_target) && tex_target != GL_TEXTURE_1D_ARRAY))
                    val = lvl->depth ? lvl->depth : 1;
                else
                    val = 1;
            }
            else if (buffer_attached) val = 1;
            break;

        case GL_TEXTURE_INTERNAL_FORMAT:
            val = ifmt;
            break;

        case GL_TEXTURE_RED_SIZE:   val = d ? d->r : 0; break;
        case GL_TEXTURE_GREEN_SIZE: val = d ? d->g : 0; break;
        case GL_TEXTURE_BLUE_SIZE:  val = d ? d->b : 0; break;
        case GL_TEXTURE_ALPHA_SIZE: val = d ? d->a : 0; break;
        case GL_TEXTURE_DEPTH_SIZE: val = d ? d->d : 0; break;
        case GL_TEXTURE_STENCIL_SIZE: val = d ? d->s : 0; break;
        case GL_TEXTURE_SHARED_SIZE: val = d ? d->shared : 0; break;

        case GL_TEXTURE_RED_TYPE:   val = (d && d->r) ? d->type : GL_NONE; break;
        case GL_TEXTURE_GREEN_TYPE: val = (d && d->g) ? d->type : GL_NONE; break;
        case GL_TEXTURE_BLUE_TYPE:  val = (d && d->b) ? d->type : GL_NONE; break;
        case GL_TEXTURE_ALPHA_TYPE: val = (d && d->a) ? d->type : GL_NONE; break;
        case GL_TEXTURE_DEPTH_TYPE: val = (d && d->d) ? d->type : GL_NONE; break;

        case GL_TEXTURE_COMPRESSED:
            val = (d && d->compressed) ? GL_TRUE : GL_FALSE;
            break;

        case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
        {
            if (!d || !d->compressed)
                TQ_ERROR(GL_INVALID_OPERATION);

            GLuint bx = (lvl->width + d->bw - 1) / d->bw;
            GLuint by = (lvl->height + d->bh - 1) / d->bh;
            GLuint bz = (tex_target == GL_TEXTURE_3D || tq_target_is_array(tex_target)) ? MAX(lvl->depth, 1) : 1;
            val = (GLint)(bx * by * bz * d->bbytes);
            break;
        }

        case GL_TEXTURE_SAMPLES:
            if (!defined) val = 0;
            else if (rec && rec->samples) val = rec->samples;
            else if (have_info && info.sample_count > 1) val = info.sample_count;
            else val = 0;
            break;

        case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS:
            val = rec ? rec->fixed_sample_locations : GL_TRUE;
            break;

        case GL_TEXTURE_BUFFER_OFFSET:
            if (rec && rec->is_buffer) val = (GLint)rec->buffer_offset;
            else if (have_info && info.has_buffer) val = (GLint)info.buffer_offset;
            break;

        case GL_TEXTURE_BUFFER_SIZE:
            if (rec && rec->is_buffer)
            {
                GLsizeiptr size = rec->buffer_size;
                Buffer *b = (Buffer *)searchHashTable(&STATE(buffer_table), rec->buffer);
                if (b)
                {
                    /* a resize after the binding is visible in the query (the CTS
                     * re-uploads the store and re-queries without rebinding) */
                    if (size < 0 || (GLsizeiptr)b->size != rec->buffer_offset + size)
                        size = b->size > rec->buffer_offset ? b->size - rec->buffer_offset : 0;
                }
                else if (size < 0)
                    size = 0;
                val = (GLint)size;
            }
            else if (have_info && info.has_buffer) val = (GLint)info.buffer_length;
            break;

        case GL_TEXTURE_BUFFER_DATA_STORE_BINDING:
            if (rec && rec->is_buffer) val = rec->buffer;
            else if (have_info && info.has_buffer) val = tq_buffer_name_for_mtl(ctx, info.mtl_buffer);
            break;

        default:
            TQ_ERROR(GL_INVALID_ENUM);
    }

    if (iparams) *iparams = val;
    if (fparams) *fparams = (GLfloat)val;
}

/* glGetTextureLevelParameter*: target comes from the object; cube maps report face +X */
static void tq_texture_level(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *ip, GLfloat *fp)
{
    Texture *tex = texture ? findTexture(ctx, texture) : NULL;

    if (!tex)
        TQ_ERROR(GL_INVALID_OPERATION);

    GLenum target = tex->target;
    if (target == GL_TEXTURE_CUBE_MAP)
        target = GL_TEXTURE_CUBE_MAP_POSITIVE_X;

    mglTexParamGetLevel(ctx, tex, target, level, pname, ip, fp, true);
}

void mglTexParamGetTextureLevelParameteriv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLint *params)
{
    tq_texture_level(ctx, texture, level, pname, params, NULL);
}

void mglTexParamGetTextureLevelParameterfv(GLMContext ctx, GLuint texture, GLint level, GLenum pname, GLfloat *params)
{
    tq_texture_level(ctx, texture, level, pname, NULL, params);
}

#pragma mark internalformat queries

static bool tq_valid_ifq_target(GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D: case GL_TEXTURE_2D_ARRAY:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY:
        case GL_TEXTURE_RECTANGLE:
        case GL_TEXTURE_BUFFER:
        case GL_RENDERBUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return true;
    }
    return false;
}

static bool tq_is_ms_target(GLenum target)
{
    return target == GL_RENDERBUFFER || target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

/* formats usable with TexBuffer (GL 4.6 table 8.18) */
static bool tq_is_buffer_format(const MGLFormatDesc *d)
{
    if (!d || d->compressed || d->srgb || !tq_has_color(d) || d->type == GL_SIGNED_NORMALIZED)
        return false;

    if (d->shared || d->fmt == GL_R11F_G11F_B10F || d->fmt == GL_RGB10_A2 || d->fmt == GL_RGB10_A2UI)
        return false;

    GLuint bits = d->r;
    if (bits != 8 && bits != 16 && bits != 32)
        return false;
    if ((d->g && d->g != bits) || (d->b && d->b != bits) || (d->a && d->a != bits))
        return false;

    if (d->base == GL_RGB)
        return bits == 32;            /* RGB32F/I/UI only */

    if (bits == 8 && d->type == GL_FLOAT)
        return false;

    if (d->type == GL_UNSIGNED_NORMALIZED && bits == 32)
        return false;

    return true;
}

/* is this (sized) format usable as a resource of this target */
static bool tq_supported_for_target(GLMContext ctx, GLenum target, const MGLFormatDesc *d, const MGLTexParamFormatCaps *caps)
{
    bool depth_stencil = (d->d || d->s);

    if (!caps->supported)
        return false;

    switch (target)
    {
        case GL_TEXTURE_BUFFER:
            return tq_is_buffer_format(d);

        case GL_RENDERBUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            if (d->compressed) return false;
            if (target == GL_RENDERBUFFER)
                return caps->color_renderable || caps->depth_renderable || caps->stencil_renderable;
            return caps->msaa && (caps->color_renderable || caps->depth_renderable || caps->stencil_renderable);

        case GL_TEXTURE_3D:
            return !d->compressed && !depth_stencil;

        case GL_TEXTURE_RECTANGLE:
            return !d->compressed;

        case GL_TEXTURE_1D:
        case GL_TEXTURE_1D_ARRAY:
            return !d->compressed;

        default:
            return true;
    }
}

static void tq_max_dims(GLMContext ctx, GLenum target, GLint64 *w, GLint64 *h, GLint64 *dp, GLint64 *layers)
{
    GLint64 tex2d = STATE_VAR(max_texture_size) ? STATE_VAR(max_texture_size) : 16384;
    GLint64 tex3d = STATE_VAR(max_3d_texture_size) ? STATE_VAR(max_3d_texture_size) : 2048;
    GLint64 cube  = STATE_VAR(max_cube_map_texture_size) ? STATE_VAR(max_cube_map_texture_size) : 16384;
    GLint64 arr   = STATE_VAR(max_array_texture_layers) ? STATE_VAR(max_array_texture_layers) : 2048;
    GLint64 rect  = STATE_VAR(max_rectangle_texture_size) ? STATE_VAR(max_rectangle_texture_size) : tex2d;
    GLint64 tbuf  = STATE_VAR(max_texture_buffer_size) ? STATE_VAR(max_texture_buffer_size) : (64 * 1024 * 1024);

    *w = *h = *dp = *layers = 0;

    switch (target)
    {
        case GL_TEXTURE_1D: *w = tex2d; break;
        case GL_TEXTURE_1D_ARRAY: *w = tex2d; *layers = arr; break;
        case GL_TEXTURE_2D: case GL_TEXTURE_2D_MULTISAMPLE: case GL_RENDERBUFFER: *w = *h = tex2d; break;
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: *w = *h = tex2d; *layers = arr; break;
        case GL_TEXTURE_3D: *w = *h = *dp = tex3d; break;
        case GL_TEXTURE_CUBE_MAP: *w = *h = cube; break;
        case GL_TEXTURE_CUBE_MAP_ARRAY: *w = *h = cube; *layers = arr; break;
        case GL_TEXTURE_RECTANGLE: *w = *h = rect; break;
        case GL_TEXTURE_BUFFER: *w = tbuf; break;
    }
}

/* fill up to 'count' values of pname; returns number of values written, -1 on INVALID_ENUM */
static int tq_internalformat(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLint64 *out, int max_out)
{
    const MGLFormatDesc *raw = tq_raw_desc(internalformat);
    const MGLFormatDesc *d = mglTexParamFormatDesc(internalformat);
    MGLTexParamFormatCaps caps;
    bool supported = false;

    bzero(&caps, sizeof(caps));

    if (d)
    {
        /* Metal backing of the format the app asked for (generic formats: their preferred format) */
        if (tq_caps(ctx, raw->base == 0 ? d->fmt : internalformat, &caps))
            supported = tq_supported_for_target(ctx, target, d, &caps);
    }

    bool color = supported && tq_has_color(d);
    bool depth = supported && d->d;
    bool stencil = supported && d->s;
    bool color_r = color && caps.color_renderable && !d->compressed && d->fmt != GL_RGB9_E5;
    bool depth_r = depth && caps.depth_renderable;
    bool stencil_r = stencil && caps.stencil_renderable;
    bool renderable = (color_r || depth_r || stencil_r) && target != GL_TEXTURE_BUFFER;
    bool mip_target = !(target == GL_TEXTURE_RECTANGLE || target == GL_TEXTURE_BUFFER || tq_is_ms_target(target));
    bool image_fmt = supported && d->img_class != GL_NONE && target != GL_RENDERBUFFER;
    bool sampleable = supported && target != GL_RENDERBUFFER;
    bool gather_target = (target == GL_TEXTURE_2D || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP ||
                          target == GL_TEXTURE_CUBE_MAP_ARRAY || target == GL_TEXTURE_RECTANGLE);
    bool shadow_target = (target == GL_TEXTURE_1D || target == GL_TEXTURE_1D_ARRAY || gather_target);
    GLint64 v = 0;

#define FULL(c) ((c) ? GL_FULL_SUPPORT : GL_NONE)

    switch (pname)
    {
        case GL_NUM_SAMPLE_COUNTS:
            v = (supported && tq_is_ms_target(target) && renderable && caps.msaa) ? caps.num_sample_counts : 0;
            break;

        case GL_SAMPLES:
        {
            if (!(supported && tq_is_ms_target(target) && renderable && caps.msaa))
                return 0;   /* params not modified */
            int n = 0;
            for (int i = 0; i < caps.num_sample_counts && n < max_out; i++)
                out[n++] = caps.sample_counts[i];
            return n;
        }

        case GL_INTERNALFORMAT_SUPPORTED: v = supported ? GL_TRUE : GL_FALSE; break;
        case GL_INTERNALFORMAT_PREFERRED: v = supported ? d->fmt : GL_NONE; break;

        case GL_INTERNALFORMAT_RED_SIZE:     v = supported ? d->r : 0; break;
        case GL_INTERNALFORMAT_GREEN_SIZE:   v = supported ? d->g : 0; break;
        case GL_INTERNALFORMAT_BLUE_SIZE:    v = supported ? d->b : 0; break;
        case GL_INTERNALFORMAT_ALPHA_SIZE:   v = supported ? d->a : 0; break;
        case GL_INTERNALFORMAT_DEPTH_SIZE:   v = supported ? d->d : 0; break;
        case GL_INTERNALFORMAT_STENCIL_SIZE: v = supported ? d->s : 0; break;
        case GL_INTERNALFORMAT_SHARED_SIZE:  v = supported ? d->shared : 0; break;

        case GL_INTERNALFORMAT_RED_TYPE:     v = (supported && d->r) ? d->type : GL_NONE; break;
        case GL_INTERNALFORMAT_GREEN_TYPE:   v = (supported && d->g) ? d->type : GL_NONE; break;
        case GL_INTERNALFORMAT_BLUE_TYPE:    v = (supported && d->b) ? d->type : GL_NONE; break;
        case GL_INTERNALFORMAT_ALPHA_TYPE:   v = (supported && d->a) ? d->type : GL_NONE; break;
        case GL_INTERNALFORMAT_DEPTH_TYPE:   v = (supported && d->d) ? d->type : GL_NONE; break;
        case GL_INTERNALFORMAT_STENCIL_TYPE: v = (supported && d->s) ? GL_UNSIGNED_INT : GL_NONE; break;

        case GL_MAX_WIDTH: case GL_MAX_HEIGHT: case GL_MAX_DEPTH: case GL_MAX_LAYERS:
        case GL_MAX_COMBINED_DIMENSIONS:
        {
            GLint64 w, h, dp, l;
            tq_max_dims(ctx, target, &w, &h, &dp, &l);
            if (!supported) { v = 0; break; }
            switch (pname)
            {
                case GL_MAX_WIDTH: v = w; break;
                case GL_MAX_HEIGHT: v = h; break;
                case GL_MAX_DEPTH: v = dp; break;
                case GL_MAX_LAYERS: v = l; break;
                default:
                    v = w * (h ? h : 1) * (dp ? dp : 1) * (l ? l : 1);
                    if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY) v *= 6;
                    break;
            }
            break;
        }

        case GL_COLOR_COMPONENTS:   v = color ? GL_TRUE : GL_FALSE; break;
        case GL_DEPTH_COMPONENTS:   v = depth ? GL_TRUE : GL_FALSE; break;
        case GL_STENCIL_COMPONENTS: v = stencil ? GL_TRUE : GL_FALSE; break;
        case GL_COLOR_RENDERABLE:   v = (color_r && target != GL_TEXTURE_BUFFER) ? GL_TRUE : GL_FALSE; break;
        case GL_DEPTH_RENDERABLE:   v = (depth_r && target != GL_TEXTURE_BUFFER) ? GL_TRUE : GL_FALSE; break;
        case GL_STENCIL_RENDERABLE: v = (stencil_r && target != GL_TEXTURE_BUFFER) ? GL_TRUE : GL_FALSE; break;

        case GL_FRAMEBUFFER_RENDERABLE: v = FULL(renderable); break;
        case GL_FRAMEBUFFER_RENDERABLE_LAYERED:
            v = FULL(renderable && (tq_target_is_array(target) || target == GL_TEXTURE_3D ||
                                    target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY));
            break;
        case GL_FRAMEBUFFER_BLEND:
            v = FULL(renderable && color_r && caps.blendable && !tq_is_integer(d));
            break;

        case GL_READ_PIXELS: v = FULL(renderable); break;
        case GL_READ_PIXELS_FORMAT:
        case GL_TEXTURE_IMAGE_FORMAT:
        case GL_GET_TEXTURE_IMAGE_FORMAT:
            if (!supported || (pname == GL_READ_PIXELS_FORMAT && !renderable) ||
                (pname != GL_READ_PIXELS_FORMAT && target == GL_RENDERBUFFER))
                v = GL_NONE;
            else
                v = d->pix_format;
            break;
        case GL_READ_PIXELS_TYPE:
        case GL_TEXTURE_IMAGE_TYPE:
        case GL_GET_TEXTURE_IMAGE_TYPE:
            if (!supported || (pname == GL_READ_PIXELS_TYPE && !renderable) ||
                (pname != GL_READ_PIXELS_TYPE && target == GL_RENDERBUFFER))
                v = GL_NONE;
            else
                v = d->pix_type;
            break;

        case GL_MIPMAP: v = (supported && mip_target) ? GL_TRUE : GL_FALSE; break;
        case GL_MANUAL_GENERATE_MIPMAP:
            /* blit generateMipmapsForTexture needs a color-renderable, filterable format */
            v = FULL(supported && mip_target && color_r && caps.filterable && !d->compressed);
            break;
        case GL_AUTO_GENERATE_MIPMAP:
            v = GL_NONE;   /* core profile has no automatic mipmap generation */
            break;

        case GL_COLOR_ENCODING: v = color ? (d->srgb ? GL_SRGB : GL_LINEAR) : GL_NONE; break;
        case GL_SRGB_READ:  v = FULL(color && d->srgb); break;
        case GL_SRGB_WRITE: v = FULL(color_r && d->srgb); break;

        case GL_FILTER: v = FULL(sampleable && caps.filterable && !tq_is_integer(d) && !tq_is_ms_target(target)); break;

        case GL_VERTEX_TEXTURE:
        case GL_TESS_CONTROL_TEXTURE:
        case GL_TESS_EVALUATION_TEXTURE:
        case GL_GEOMETRY_TEXTURE:
        case GL_FRAGMENT_TEXTURE:
        case GL_COMPUTE_TEXTURE:
            v = FULL(sampleable);
            break;

        case GL_TEXTURE_SHADOW:        v = FULL(sampleable && depth && shadow_target); break;
        case GL_TEXTURE_GATHER:        v = FULL(sampleable && gather_target && (color || depth)); break;
        case GL_TEXTURE_GATHER_SHADOW: v = FULL(sampleable && depth && gather_target); break;

        case GL_SHADER_IMAGE_LOAD:   v = FULL(image_fmt); break;
        case GL_SHADER_IMAGE_STORE:  v = FULL(image_fmt && caps.writable); break;
        case GL_SHADER_IMAGE_ATOMIC: v = FULL(image_fmt && caps.atomic); break;

        case GL_IMAGE_TEXEL_SIZE: v = image_fmt ? (d->r + d->g + d->b + d->a) : 0; break;
        case GL_IMAGE_COMPATIBILITY_CLASS: v = image_fmt ? d->img_class : GL_NONE; break;
        case GL_IMAGE_PIXEL_FORMAT: v = image_fmt ? d->pix_format : GL_NONE; break;
        case GL_IMAGE_PIXEL_TYPE: v = image_fmt ? d->pix_type : GL_NONE; break;
        case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE: v = image_fmt ? GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE : GL_NONE; break;

        /* Metal cannot sample a texture while it is bound as the depth/stencil
           attachment of the same pass */
        case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_TEST:
        case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_TEST:
        case GL_SIMULTANEOUS_TEXTURE_AND_DEPTH_WRITE:
        case GL_SIMULTANEOUS_TEXTURE_AND_STENCIL_WRITE:
            v = GL_NONE;
            break;

        case GL_TEXTURE_COMPRESSED: v = (supported && d->compressed) ? GL_TRUE : GL_FALSE; break;
        case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH:  v = (supported && d->compressed) ? d->bw : 0; break;
        case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT: v = (supported && d->compressed) ? d->bh : 0; break;
        case GL_TEXTURE_COMPRESSED_BLOCK_SIZE:   v = (supported && d->compressed) ? d->bbytes : 0; break;

        case GL_CLEAR_BUFFER:  v = FULL(d && tq_is_buffer_format(d) && caps.supported); break;
        case GL_CLEAR_TEXTURE: v = FULL(supported && !d->compressed && target != GL_RENDERBUFFER); break;

        case GL_TEXTURE_VIEW:
            v = FULL(supported && d->view_class != GL_NONE && target != GL_RENDERBUFFER && target != GL_TEXTURE_BUFFER);
            break;
        case GL_VIEW_COMPATIBILITY_CLASS:
            v = (supported && target != GL_RENDERBUFFER && target != GL_TEXTURE_BUFFER) ? d->view_class : GL_NONE;
            break;

        default:
            return -1;
    }

#undef FULL

    if (max_out > 0)
    {
        out[0] = v;
        return 1;
    }
    return 0;
}

static bool tq_internalformat_common(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64 *tmp, int *n)
{
    if (!tq_valid_ifq_target(target))
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return false;
    }

    if (count < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return false;
    }

    *n = tq_internalformat(ctx, target, internalformat, pname, tmp, count < 8 ? count : 8);
    if (*n < 0)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return false;
    }

    return true;
}

void mglGetInternalformati64v(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64 *params)
{
    GLint64 tmp[8];
    int n;

    if (!tq_internalformat_common(ctx, target, internalformat, pname, count, tmp, &n))
        return;

    for (int i = 0; i < n && params; i++)
        params[i] = tmp[i];
}

void mglGetInternalformativ(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint *params)
{
    GLint64 tmp[8];
    int n;

    if (!tq_internalformat_common(ctx, target, internalformat, pname, count, tmp, &n))
        return;

    for (int i = 0; i < n && params; i++)
        params[i] = tmp[i] > 0x7fffffff ? 0x7fffffff : (GLint)tmp[i];
}
