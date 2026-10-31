/*
 * MGLRenderer+Framebuffer.m — Metal side of the 'fb' domain.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 *  - default framebuffer: offscreen color + depth/stencil textures; swap copies
 *    (vertically flipped) into a CAMetalLayer drawable and presents it
 *  - render pass construction for the bound draw framebuffer (MRT mapping via
 *    the draw-buffer list, layered/levelled attachments, invalidation -> DontCare)
 *    and the matching pipeline attachment formats / raster sample count
 *  - clears: loadAction clear when the clear covers every pixel with full write
 *    masks, otherwise a scissored/masked full-screen draw
 *  - readback: blit into a shared buffer, commit+wait, decode to canonical texels
 *  - blits: shader blit (scaling, flipping, filtering, MS resolve, format
 *    conversion), texture copies for exact depth/stencil copies, CPU path for
 *    scaled/scissored depth/stencil
 *  - renderbuffer storage (incl. MTLTextureType2DMultisample)
 *
 * Row convention: Metal texture row r == GL window row y=r. Nothing here flips
 * rows except presentation.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Compat.h"
#import "MGLRenderer+Framebuffer.h"
#include "mgl_ext_fb.h"
#include "mgl_ext_rstate.h"    /* RS(): indexed viewports (default-fb sizing) */

/* framebuffers.c keeps this cast private; mirror it for the renderbuffer
 * attachment lookups below (struct is public in mgl_ext_fb.h) */
#define FB_RBEX(_rb_)   ((MGLRenderbufferEx *)(_rb_))

// render-state per-index setters (state_ext.c)
void mgl_rstate_set_all_viewports(GLMContext ctx, GLfloat x, GLfloat y, GLfloat w, GLfloat h);
void mgl_rstate_set_all_scissors(GLMContext ctx, GLint x, GLint y, GLint w, GLint h);

static inline MGLRenderer *FBR(GLMContext c) { return (__bridge MGLRenderer *)c->mtl_funcs.mtlObj; }
static inline struct MGLExtMetal_fb *FBM(GLMContext c) { return c->mtl_funcs.ext_fb; }

typedef struct FBTarget {
    __unsafe_unretained id<MTLTexture> tex;
    NSUInteger level, slice, plane, layers;
    /* the attachment's GL NAME (texture or renderbuffer): the render-pass-descriptor
     * cache compares FBTargets with memcmp, and a Metal texture address alone is NOT
     * identity — the CTS deletes and re-creates an FBO attachment texture per sub-case,
     * and a recycled MTLTexture address made the memcmp hit, so the draw encoded into
     * the OLD texture (texture_swizzle smoke).  Appended last: gpuaudit casts its own
     * 40-byte MGLGAFBTarget over FBTargets, so tex/level/slice/plane/layers must keep
     * their offsets (the assert in MGLRenderer+Gpuaudit.m pins the total size). */
    GLuint name;
} FBTarget;

typedef struct FBTargets {
    FBTarget color[MAX_COLOR_ATTACHMENTS];
    GLuint color_att_bit[MAX_COLOR_ATTACHMENTS];   /* invalid_mask bit per slot */
    FBTarget depth, stencil;
    NSUInteger width, height, samples, layers;
    bool any;
} FBTargets;

#pragma mark pixel format helpers

static int fbKindForPixelFormat(MTLPixelFormat f)
{
    switch(f)
    {
        case MTLPixelFormatR8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG8Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA8Sint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Sint:
            return MGL_FB_KIND_INT;
        case MTLPixelFormatR8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatR32Uint:
        case MTLPixelFormatRG8Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA32Uint:
        case MTLPixelFormatRGB10A2Uint:
            return MGL_FB_KIND_UINT;
        default:
            return MGL_FB_KIND_FLOAT;
    }
}

static bool fbFormatHasDepth(MTLPixelFormat f)
{
    return f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float || f == MTLPixelFormatDepth32Float_Stencil8 ||
           f == MTLPixelFormatDepth24Unorm_Stencil8;
}

static bool fbFormatHasStencil(MTLPixelFormat f)
{
    return f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth32Float_Stencil8 || f == MTLPixelFormatDepth24Unorm_Stencil8 ||
           f == MTLPixelFormatX32_Stencil8;
}

static MTLPixelFormat fbMTLFormatForInternalFormat(GLenum ifmt)
{
    switch(ifmt)
    {
        case GL_RED: case GL_R8:            return MTLPixelFormatR8Unorm;
        case GL_R8_SNORM:                   return MTLPixelFormatR8Snorm;
        case GL_R16:                        return MTLPixelFormatR16Unorm;
        case GL_R16_SNORM:                  return MTLPixelFormatR16Snorm;
        case GL_RG: case GL_RG8:            return MTLPixelFormatRG8Unorm;
        case GL_RG8_SNORM:                  return MTLPixelFormatRG8Snorm;
        case GL_RG16:                       return MTLPixelFormatRG16Unorm;
        case GL_RG16_SNORM:                 return MTLPixelFormatRG16Snorm;
        case GL_R3_G3_B2: case GL_RGB4: case GL_RGB5: case GL_RGB565: case GL_RGB8: case GL_RGB:
        case GL_RGBA2: case GL_RGBA4: case GL_RGB5_A1: case GL_RGBA8: case GL_RGBA:
                                            return MTLPixelFormatRGBA8Unorm;
        case GL_RGB8_SNORM: case GL_RGBA8_SNORM: return MTLPixelFormatRGBA8Snorm;
        case GL_RGB10: case GL_RGB10_A2:    return MTLPixelFormatRGB10A2Unorm;
        case GL_RGB12: case GL_RGB16: case GL_RGBA12: case GL_RGBA16: return MTLPixelFormatRGBA16Unorm;
        case GL_RGB16_SNORM: case GL_RGBA16_SNORM: return MTLPixelFormatRGBA16Snorm;
        case GL_SRGB8: case GL_SRGB8_ALPHA8: return MTLPixelFormatRGBA8Unorm_sRGB;
        case GL_R16F:                       return MTLPixelFormatR16Float;
        case GL_RG16F:                      return MTLPixelFormatRG16Float;
        case GL_RGB16F: case GL_RGBA16F:    return MTLPixelFormatRGBA16Float;
        case GL_R32F:                       return MTLPixelFormatR32Float;
        case GL_RG32F:                      return MTLPixelFormatRG32Float;
        case GL_RGB32F: case GL_RGBA32F:    return MTLPixelFormatRGBA32Float;
        case GL_R11F_G11F_B10F:             return MTLPixelFormatRG11B10Float;
        /* compat s36: RGB9_E5 renderbuffers are emulated on an RGBA16Float
         * backing (Apple Metal has no renderable RGB9E5Float target).  The
         * GL_RGB9_E5 (5-bit shared exponent / 9-bit mantissa, bias 15) lattice
         * is re-applied to the backing by fbQuantizeRGB9E5 before ANY read /
         * copy-out consumes the attachment, so write/clear-then-read and
         * readPixels(GL_UNSIGNED_INT_5_9_9_9_REV) are bit-exact (every lattice
         * value is exactly representable in half float, so the quantised
         * backing IS the 5-9-9-9 image).  See fbQuantizeRGB9E5 for the two
         * documented caveats (blending, multisample). */
        case GL_RGB9_E5:                    return MTLPixelFormatRGBA16Float;
        case GL_R8I:                        return MTLPixelFormatR8Sint;
        case GL_R8UI:                       return MTLPixelFormatR8Uint;
        case GL_R16I:                       return MTLPixelFormatR16Sint;
        case GL_R16UI:                      return MTLPixelFormatR16Uint;
        case GL_R32I:                       return MTLPixelFormatR32Sint;
        case GL_R32UI:                      return MTLPixelFormatR32Uint;
        case GL_RG8I:                       return MTLPixelFormatRG8Sint;
        case GL_RG8UI:                      return MTLPixelFormatRG8Uint;
        case GL_RG16I:                      return MTLPixelFormatRG16Sint;
        case GL_RG16UI:                     return MTLPixelFormatRG16Uint;
        case GL_RG32I:                      return MTLPixelFormatRG32Sint;
        case GL_RG32UI:                     return MTLPixelFormatRG32Uint;
        case GL_RGB8I: case GL_RGBA8I:      return MTLPixelFormatRGBA8Sint;
        case GL_RGB8UI: case GL_RGBA8UI:    return MTLPixelFormatRGBA8Uint;
        case GL_RGB16I: case GL_RGBA16I:    return MTLPixelFormatRGBA16Sint;
        case GL_RGB16UI: case GL_RGBA16UI:  return MTLPixelFormatRGBA16Uint;
        case GL_RGB32I: case GL_RGBA32I:    return MTLPixelFormatRGBA32Sint;
        case GL_RGB32UI: case GL_RGBA32UI:  return MTLPixelFormatRGBA32Uint;
        case GL_RGB10_A2UI:                 return MTLPixelFormatRGB10A2Uint;
        case GL_DEPTH_COMPONENT16:          return MTLPixelFormatDepth16Unorm;
        case GL_DEPTH_COMPONENT: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32: case GL_DEPTH_COMPONENT32F:
                                            return MTLPixelFormatDepth32Float;
        case GL_DEPTH_STENCIL: case GL_DEPTH24_STENCIL8: case GL_DEPTH32F_STENCIL8:
                                            return MTLPixelFormatDepth32Float_Stencil8;
        case GL_STENCIL_INDEX: case GL_STENCIL_INDEX1: case GL_STENCIL_INDEX4: case GL_STENCIL_INDEX8: case GL_STENCIL_INDEX16:
                                            return MTLPixelFormatStencil8;
    }
    return MTLPixelFormatInvalid;
}

/* bytes per texel for buffer copies (aspect-specific for depth/stencil formats) */
static NSUInteger fbBytesPerPixel(MTLPixelFormat f, int aspect)
{
    if (aspect == MGL_FB_ASPECT_STENCIL) return 1;
    if (aspect == MGL_FB_ASPECT_DEPTH) return f == MTLPixelFormatDepth16Unorm ? 2 : 4;

    switch(f)
    {
        case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm: case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint:
            return 1;
        case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Uint: case MTLPixelFormatR16Sint:
        case MTLPixelFormatR16Float: case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRG8Sint: case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatA1BGR5Unorm:
        case MTLPixelFormatABGR4Unorm: case MTLPixelFormatBGR5A1Unorm:
            return 2;
        case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatR32Float:
        case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Snorm: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB: case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatRGB10A2Uint: case MTLPixelFormatRG11B10Float: case MTLPixelFormatRGB9E5Float:
        case MTLPixelFormatBGR10A2Unorm:
            return 4;
        case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRG32Float:
        case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Uint:
        case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA16Float:
            return 8;
        case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint: case MTLPixelFormatRGBA32Float:
            return 16;
        default:
            return 0;
    }
}

static float fbHalfToFloat(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

static float fbUFToFloat(uint32_t v, int mbits)
{
    uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return ldexpf((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf((float)(m | (1u << mbits)), (int)e - 15 - mbits);
}

/* decode one texel to canonical rgba; returns kind */
static int fbDecodeColor(MTLPixelFormat f, const uint8_t *p, void *out4)
{
    float *o = (float *)out4;
    int32_t *oi = (int32_t *)out4;
    uint32_t *ou = (uint32_t *)out4;
    o[0] = o[1] = o[2] = 0; o[3] = 1;

#define U8(i)  (p[i] / 255.0f)
#define S8(i)  fmaxf(((int8_t)p[i]) / 127.0f, -1.f)
#define U16(i) (((const uint16_t *)p)[i] / 65535.0f)
#define S16(i) fmaxf(((const int16_t *)p)[i] / 32767.0f, -1.f)
#define H16(i) fbHalfToFloat(((const uint16_t *)p)[i])
#define F32(i) (((const float *)p)[i])
    switch(f)
    {
        case MTLPixelFormatR8Unorm:   o[0] = U8(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatR8Snorm:   o[0] = S8(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatR16Unorm:  o[0] = U16(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatR16Snorm:  o[0] = S16(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatR16Float:  o[0] = H16(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatR32Float:  o[0] = F32(0); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG8Unorm:  o[0] = U8(0); o[1] = U8(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG8Snorm:  o[0] = S8(0); o[1] = S8(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG16Unorm: o[0] = U16(0); o[1] = U16(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG16Snorm: o[0] = S16(0); o[1] = S16(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG16Float: o[0] = H16(0); o[1] = H16(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRG32Float: o[0] = F32(0); o[1] = F32(1); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA8Unorm:
            for (int i = 0; i < 4; i++) o[i] = U8(i);
            return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA8Unorm_sRGB:
            /* GL: the sRGB transfer function applies at SAMPLING only — pixel
             * reads (glReadPixels / getTexImage) return the stored bytes raw,
             * exactly like the non-sRGB format (the CTS packed_pixels and
             * texture_size_promotion tests assert this). */
            for (int i = 0; i < 4; i++) o[i] = U8(i);
            return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatBGRA8Unorm:
            o[0] = U8(2); o[1] = U8(1); o[2] = U8(0); o[3] = U8(3); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatBGRA8Unorm_sRGB:
            o[0] = U8(2); o[1] = U8(1); o[2] = U8(0); o[3] = U8(3);
            return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA8Snorm: for (int i = 0; i < 4; i++) o[i] = S8(i); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA16Unorm: for (int i = 0; i < 4; i++) o[i] = U16(i); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA16Snorm: for (int i = 0; i < 4; i++) o[i] = S16(i); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA16Float: for (int i = 0; i < 4; i++) o[i] = H16(i); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGBA32Float: for (int i = 0; i < 4; i++) o[i] = F32(i); return MGL_FB_KIND_FLOAT;
        case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatBGR10A2Unorm:
        {
            uint32_t v = *(const uint32_t *)p;
            float a = (v & 0x3ff) / 1023.f, b = ((v >> 10) & 0x3ff) / 1023.f, c = ((v >> 20) & 0x3ff) / 1023.f;
            if (f == MTLPixelFormatRGB10A2Unorm) { o[0] = a; o[1] = b; o[2] = c; } else { o[0] = c; o[1] = b; o[2] = a; }
            o[3] = (v >> 30) / 3.f;
            return MGL_FB_KIND_FLOAT;
        }
        case MTLPixelFormatRG11B10Float:
        {
            uint32_t v = *(const uint32_t *)p;
            o[0] = fbUFToFloat(v & 0x7ff, 6); o[1] = fbUFToFloat((v >> 11) & 0x7ff, 6); o[2] = fbUFToFloat(v >> 22, 5);
            return MGL_FB_KIND_FLOAT;
        }
        case MTLPixelFormatRGB9E5Float:
        {
            uint32_t v = *(const uint32_t *)p;
            float sc = ldexpf(1.f, (int)(v >> 27) - 15 - 9);
            o[0] = (v & 0x1ff) * sc; o[1] = ((v >> 9) & 0x1ff) * sc; o[2] = ((v >> 18) & 0x1ff) * sc;
            return MGL_FB_KIND_FLOAT;
        }
        case MTLPixelFormatB5G6R5Unorm:
        {
            uint16_t v = *(const uint16_t *)p;
            o[2] = (v & 0x1f) / 31.f; o[1] = ((v >> 5) & 0x3f) / 63.f; o[0] = (v >> 11) / 31.f;
            return MGL_FB_KIND_FLOAT;
        }
        default: break;
    }

    /* integer formats */
    oi[0] = oi[1] = oi[2] = 0; oi[3] = 1;
    switch(f)
    {
        case MTLPixelFormatR8Sint:  oi[0] = (int8_t)p[0]; return MGL_FB_KIND_INT;
        case MTLPixelFormatR16Sint: oi[0] = ((const int16_t *)p)[0]; return MGL_FB_KIND_INT;
        case MTLPixelFormatR32Sint: oi[0] = ((const int32_t *)p)[0]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRG8Sint: oi[0] = (int8_t)p[0]; oi[1] = (int8_t)p[1]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRG16Sint: for (int i = 0; i < 2; i++) oi[i] = ((const int16_t *)p)[i]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRG32Sint: for (int i = 0; i < 2; i++) oi[i] = ((const int32_t *)p)[i]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRGBA8Sint: for (int i = 0; i < 4; i++) oi[i] = (int8_t)p[i]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRGBA16Sint: for (int i = 0; i < 4; i++) oi[i] = ((const int16_t *)p)[i]; return MGL_FB_KIND_INT;
        case MTLPixelFormatRGBA32Sint: for (int i = 0; i < 4; i++) oi[i] = ((const int32_t *)p)[i]; return MGL_FB_KIND_INT;
        default: break;
    }
    ou[0] = ou[1] = ou[2] = 0; ou[3] = 1;
    switch(f)
    {
        case MTLPixelFormatR8Uint:  ou[0] = p[0]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatR16Uint: ou[0] = ((const uint16_t *)p)[0]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatR32Uint: ou[0] = ((const uint32_t *)p)[0]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRG8Uint: ou[0] = p[0]; ou[1] = p[1]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRG16Uint: for (int i = 0; i < 2; i++) ou[i] = ((const uint16_t *)p)[i]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRG32Uint: for (int i = 0; i < 2; i++) ou[i] = ((const uint32_t *)p)[i]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRGBA8Uint: for (int i = 0; i < 4; i++) ou[i] = p[i]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRGBA16Uint: for (int i = 0; i < 4; i++) ou[i] = ((const uint16_t *)p)[i]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRGBA32Uint: for (int i = 0; i < 4; i++) ou[i] = ((const uint32_t *)p)[i]; return MGL_FB_KIND_UINT;
        case MTLPixelFormatRGB10A2Uint:
        {
            uint32_t v = *(const uint32_t *)p;
            ou[0] = v & 0x3ff; ou[1] = (v >> 10) & 0x3ff; ou[2] = (v >> 20) & 0x3ff; ou[3] = v >> 30;
            return MGL_FB_KIND_UINT;
        }
        default: break;
    }
#undef U8
#undef S8
#undef U16
#undef S16
#undef H16
#undef F32
    return -1;
}

static void fbValueAsDoubles(int kind, const void *v, double out[4])
{
    for (int c = 0; c < 4; c++)
    {
        if (kind == MGL_FB_KIND_INT) out[c] = ((const GLint *)v)[c];
        else if (kind == MGL_FB_KIND_UINT) out[c] = ((const GLuint *)v)[c];
        else out[c] = ((const GLfloat *)v)[c];
    }
}

#pragma mark - C trampolines

static void fbDefaultSizeC(GLMContext c, GLuint *w, GLuint *h) { [FBR(c) fbDefaultSizeW:w h:h]; }
/* perf instrumentation: how often the pass is torn down for a target change
 * (MGL_PROFILE dumps it; a plain local counter — all profiler slots are in use) */
static void fbTargetsChangedC(GLMContext c)
{
    [FBR(c) endRenderEncoding];
    static int prof_on = -1;
    static uint64_t n = 0;
    if (prof_on < 0) prof_on = getenv("MGL_PROFILE") != NULL;
    if (prof_on && (++n % 500) == 1)
        fprintf(stderr, "MGL PROFILE targetsChanged #%llu\n", (unsigned long long)n);
}
static void fbClearC(GLMContext c, const MGLFBClearRequest *r) { @autoreleasepool { [FBR(c) fbClear:r]; } }
static int  fbReadC(GLMContext c, const MGLFBReadRequest *r) { @autoreleasepool { return [FBR(c) fbRead:r]; } }
static void fbBlitC(GLMContext c, const MGLFBBlitRequest *r) { @autoreleasepool { [FBR(c) fbBlit:r]; } }
static bool fbRenderbufferStorageC(GLMContext c, MGLRenderbufferEx *rb) { return [FBR(c) fbRenderbufferStorage:rb]; }
static void fbReleaseTextureC(GLMContext c, Texture *t)
{
    (void)c;
    if (t->mtl_data) { CFBridgingRelease(t->mtl_data); t->mtl_data = NULL; }
    if (t->params.mtl_data) { CFBridgingRelease(t->params.mtl_data); t->params.mtl_data = NULL; }
}
static bool fbSamplePositionC(GLMContext c, GLuint samples, GLuint index, GLfloat *xy) { return [FBR(c) fbSamplePosition:samples index:index out:xy]; }
static GLuint fbRoundSamplesC(GLMContext c, GLuint samples) { return [FBR(c) fbRoundSamples:samples]; }
static void fbWriteBufferC(GLMContext c, Buffer *buf, size_t offset, size_t size, const void *src)
{
    if (buf->data.mtl_data)
    {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf->data.mtl_data;
        if (offset + size <= b.length)
        {
            memcpy((uint8_t *)b.contents + offset, src, size);
            if (b.storageMode == MTLStorageModeManaged)
                [b didModifyRange:NSMakeRange(offset, size)];
        }
        if (buf->data.buffer_data && (void *)buf->data.buffer_data != b.contents && offset + size <= buf->data.buffer_size)
            memcpy((uint8_t *)buf->data.buffer_data + offset, src, size);
    }
    else if (buf->data.buffer_data && offset + size <= buf->data.buffer_size)
    {
        memcpy((uint8_t *)buf->data.buffer_data + offset, src, size);
        buf->data.dirty_bits |= DIRTY_BUFFER_DATA;
        c->state.dirty_bits |= DIRTY_BUFFER;
    }
}
static void fbSwapC(GLMContext c)
{
    if (!c || !c->mtl_funcs.mtlObj) return;
    @autoreleasepool { [FBR(c) fbSwapBuffers]; }
}
static void fbFlushC(GLMContext c, bool finish) { @autoreleasepool { [FBR(c) fbFlush:finish]; } }
static void fbReadDrawableC(GLMContext c, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei w, GLsizei h)
{
    /* legacy entry: RGBA8 rows of the current read buffer, GL row order */
    MGLFBReadRequest req = { c->state.readbuffer, mgl_fb_draw_state(c, c->state.readbuffer)->read_buffer,
                             MGL_FB_ASPECT_COLOR, x, y, w, h, NULL };
    float *tmp = calloc((size_t)w * h, 16);
    if (!tmp) return;
    req.out = tmp;
    int kind = fbReadC(c, &req);
    for (GLsizei r = 0; r < h && kind >= 0; r++)
        for (GLsizei i = 0; i < w; i++)
        {
            uint8_t *o = (uint8_t *)pixelBytes + (size_t)r * bytesPerRow + (size_t)i * 4;
            for (int k = 0; k < 4; k++)
            {
                double v = (kind == MGL_FB_KIND_FLOAT) ? tmp[((size_t)r * w + i) * 4 + k] : ((int32_t *)tmp)[((size_t)r * w + i) * 4 + k] / 255.0;
                o[k] = (uint8_t)lround(fmin(fmax(v, 0.0), 1.0) * 255.0);
            }
        }
    free(tmp);
    (void)bytesPerImage;
}
static void fbClearBufferLegacyC(GLMContext c, GLuint type, GLbitfield mask) { (void)type; (void)mask; [FBR(c) endRenderEncoding]; }
static void fbBlitLegacyC(GLMContext c, GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter)
{
    MGLFBBlitRequest r = { c->state.readbuffer, c->state.framebuffer, { sx0, sy0, sx1, sy1 }, { dx0, dy0, dx1, dy1 }, mask, filter };
    fbBlitC(c, &r);
}

GLuint mgl_fb_texture_sample_count(GLMContext c, Texture *tex)
{
    MGLRenderer *r = FBR(c);
    if (!r || !tex) return 1;
    if (!tex->mtl_data) [r bindMTLTexture:tex];
    if (!tex->mtl_data) return 1;
    return (GLuint)((__bridge id<MTLTexture>)tex->mtl_data).sampleCount;
}

@implementation MGLRenderer (Framebuffer)

- (void) bindExtFramebufferFuncsToGLMContext: (GLMContext) glm_ctx
{
    struct MGLExtMetal_fb *m = glm_ctx->mtl_funcs.ext_fb;
    if (!m)
    {
        m = (struct MGLExtMetal_fb *)calloc(1, sizeof(struct MGLExtMetal_fb));
        glm_ctx->mtl_funcs.ext_fb = m;
    }
    m->default_size = fbDefaultSizeC;
    m->targets_changed = fbTargetsChangedC;
    m->clear = fbClearC;
    m->read = fbReadC;
    m->blit = fbBlitC;
    m->renderbuffer_storage = fbRenderbufferStorageC;
    m->release_texture = fbReleaseTextureC;
    m->sample_position = fbSamplePositionC;
    m->round_samples = fbRoundSamplesC;
    m->write_buffer = fbWriteBufferC;
    m->max_samples = 0;     /* computed lazily once the device exists */

    /* the fb domain owns presentation, flushing and the legacy fb entry points */
    glm_ctx->mtl_funcs.mtlSwapBuffers = fbSwapC;
    glm_ctx->mtl_funcs.mtlFlush = fbFlushC;
    glm_ctx->mtl_funcs.mtlReadDrawable = fbReadDrawableC;
    glm_ctx->mtl_funcs.mtlClearBuffer = fbClearBufferLegacyC;
    glm_ctx->mtl_funcs.mtlBlitFramebuffer = fbBlitLegacyC;
}

#pragma mark command buffer helpers

- (void) fbEnsureCommandBuffer
{
    if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        [self newCommandBuffer];
}

- (void) fbSubmitAndWait
{
    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    if (cb && cb.status < MTLCommandBufferStatusCommitted)
    {
        [self commitCommandBufferWithAGXRecovery:cb];
        [cb waitUntilCompleted];
    }
    [self newCommandBuffer];
}

- (void) fbFlush:(bool)finish
{
    [self flushCommandBuffer:finish];
    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    if (cb && cb.status >= MTLCommandBufferStatusCommitted)
    {
        if (finish) [cb waitUntilCompleted];
        [self newCommandBuffer];
    }
    else if (!cb)
    {
        [self newCommandBuffer];
    }
}

- (void) fbUpdateLimits
{
    struct MGLExtMetal_fb *m = FBM(ctx);
    if (m->max_samples || !_device) return;
    GLint best = 1;
    for (GLint s = 2; s <= 8; s *= 2)
        if ([_device supportsTextureSampleCount:s]) best = s;
    m->max_samples = best;

    /* sample limits reported by glGet follow what Metal can actually allocate */
    STATE_VAR(max_framebuffer_samples) = best;
    STATE_VAR(max_color_texture_samples) = best;
    STATE_VAR(max_depth_texture_samples) = best;
    STATE_VAR(max_integer_samples) = best;
}

- (GLuint) fbRoundSamples:(GLuint)samples
{
    [self fbUpdateLimits];
    if (samples <= 1) return 1;
    for (GLuint s = 2; s <= 8; s *= 2)
        if (s >= samples && [_device supportsTextureSampleCount:s]) return s;
    return (GLuint)FBM(ctx)->max_samples;
}

- (bool) fbSamplePosition:(GLuint)samples index:(GLuint)index out:(GLfloat *)xy
{
    if (!_device || samples < 1 || index >= samples || samples > 16) return false;
    MTLSamplePosition pos[16];
    [_device getDefaultSamplePositions:pos count:samples];
    xy[0] = pos[index].x;
    xy[1] = pos[index].y;   /* Metal row r == GL row r: no flip */
    return true;
}

#pragma mark default framebuffer

- (MTLPixelFormat) fbDefaultColorFormat
{
    MTLPixelFormat f = (MTLPixelFormat)ctx->pixel_format.mtl_pixel_format;
    return f == MTLPixelFormatInvalid ? MTLPixelFormatRGBA8Unorm : f;
}

- (MTLPixelFormat) fbDefaultDepthStencilFormat
{
    bool d = ctx->depth_format.format != 0, s = ctx->stencil_format.format != 0;
    if (d && s) return MTLPixelFormatDepth32Float_Stencil8;
    if (d) return MTLPixelFormatDepth32Float;
    if (s) return MTLPixelFormatStencil8;
    return MTLPixelFormatInvalid;
}

- (CGSize) fbDrawableSize
{
    CGSize s = CGSizeZero;
    /* Headless contexts are bound to a dummy NSView that is not in any window:
     * its layer bounds are arbitrary (a fresh AppKit layer reports whatever
     * frame the dummy view was created with), so there is no real drawable.
     * The "window" for the default framebuffer is then the union of the GL
     * viewports the app has used, and it never shrinks: the GL viewport is a
     * transform, not a window size, so a later smaller viewport must not clip
     * away pixels that are still part of the framebuffer.  Only a view that
     * lives in a window has a real drawable size. */
    BOOL windowless = (_view && _view.window == nil);
    if (_layer && !windowless)
    {
        s = _layer.drawableSize;
        if (s.width < 1 || s.height < 1)
        {
            CGFloat sc = _layer.contentsScale > 0 ? _layer.contentsScale : 1;
            s = CGSizeMake(_layer.bounds.size.width * sc, _layer.bounds.size.height * sc);
        }
    }
    if (s.width < 1 || s.height < 1)
    {
        struct MGLExtMetal_fb *m = FBM(ctx);
        NSUInteger x1 = 1, y1 = 1;
        if (ctx->state.ext_rstate)
            for (int i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++)
            {
                NSUInteger rx = (NSUInteger)(ctx->state.ext_rstate->viewport[i][0] + ctx->state.ext_rstate->viewport[i][2]);
                NSUInteger ry = (NSUInteger)(ctx->state.ext_rstate->viewport[i][1] + ctx->state.ext_rstate->viewport[i][3]);
                if (rx > x1) x1 = rx;
                if (ry > y1) y1 = ry;
            }
        if (m)
        {
            if ((NSUInteger)m->default_width > x1) x1 = m->default_width;
            if ((NSUInteger)m->default_height > y1) y1 = m->default_height;
        }
        s = CGSizeMake((CGFloat)x1, (CGFloat)y1);
    }
    return s;
}

/* make the GL viewport / scissor defaults match the default framebuffer (once) */
- (void) fbInitDefaultFramebuffer
{
    struct MGLExtMetal_fb *m = FBM(ctx);
    if (!m || m->scissor_initialized || !_device) return;
    [self fbUpdateLimits];
    [self fbEnsureDefaultFramebuffer];
    m->scissor_initialized = GL_TRUE;
    GLuint w = m->default_width, h = m->default_height;
    STATE(viewport[0]) = 0; STATE(viewport[1]) = 0;
    STATE(viewport[2]) = w; STATE(viewport[3]) = h;
    STATE_VAR(scissor_box[0]) = 0; STATE_VAR(scissor_box[1]) = 0;
    STATE_VAR(scissor_box[2]) = w; STATE_VAR(scissor_box[3]) = h;
    // rstate keeps per-index viewport/scissor arrays; seed all 16 with the default size
    mgl_rstate_set_all_viewports(ctx, 0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
    mgl_rstate_set_all_scissors(ctx, 0, 0, (GLint)w, (GLint)h);
    STATE(dirty_bits) |= DIRTY_RENDER_STATE;
}

- (void) fbDefaultSizeW:(GLuint *)w h:(GLuint *)h
{
    struct MGLExtMetal_fb *m = FBM(ctx);
    if (!m->default_color && !_currentRenderEncoder)
        [self fbEnsureDefaultFramebuffer];
    *w = m->default_width;
    *h = m->default_height;
}

/* (re)create the default framebuffer images when the drawable size changes.
 * Ends an open render pass (it encodes an init clear). */
- (id<MTLTexture>) fbEnsureDefaultFramebuffer
{
    struct MGLExtMetal_fb *m = FBM(ctx);
    CGSize sz = [self fbDrawableSize];
    NSUInteger w = (NSUInteger)sz.width, h = (NSUInteger)sz.height;
    id<MTLTexture> color = (__bridge id<MTLTexture>)m->default_color;

    /* compat domain: when the IOSurface present path is enabled the default
     * framebuffer renders DIRECTLY into the present surface pool — the swap then
     * only changes layer.contents (no blit; see compatSurfacePresent).  The write
     * slot advances lazily here (after the previous present), so readbacks between
     * a present and the next draw still see the surface that was just shown. */
    if ([self compatSurfacePresentEnabled])
    {
        id<MTLTexture> surf = [self compatSurfaceWrite:w h:h];
        if (surf && surf == color)
            return color;                       /* same size, same slot: fast path */
        if (surf && color && color.width == w && color.height == h)
        {
            /* slot rotation: keep the (still correctly sized) depth/stencil, just
             * rebind the color surface; content of a fresh back buffer is undefined
             * (GL) — mark it invalid so the first pass loads nothing */
            if (getenv("MGL_DEBUG"))
                fprintf(stderr, "MGL fb: default-fb slot rotation (direct present)\n");
            [self endRenderEncoding];
            if (m->default_color) CFBridgingRelease(m->default_color);
            m->default_color = (void *)CFBridgingRetain(surf);
            m->default_fb_gen++;    /* new backing texture: descriptor-cache key changes */
            ctx->state.dirty_bits |= DIRTY_FBO;
            mgl_fb_draw_state(ctx, NULL)->invalid_mask |= 1u;
            return surf;
        }
        /* else: fall through to the (re)create path, which prefers the surface
         * pool over a private texture */
    }

    if (color && color.width == w && color.height == h)
        return color;
    if (!_device)
        return nil;

    if (_currentRenderEncoder)
        [self endRenderEncoding];

    if (m->default_color) { CFBridgingRelease(m->default_color); m->default_color = NULL; }
    if (m->default_depth_stencil) { CFBridgingRelease(m->default_depth_stencil); m->default_depth_stencil = NULL; }

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:[self fbDefaultColorFormat] width:w height:h mipmapped:NO];
    /* compat domain: ShaderRead lets the present pass sample the frame directly
     * (compatPresentCopyOpaque — one copy+alpha=1 render pass per swap);
     * ShaderWrite kept for the env-gated direct-present force-opaque kernel */
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;
    /* compat domain: the color attachment prefers the present surface (render
     * directly into what compatSurfacePresent shows — the swap is then only a
     * layer.contents change, no blit); a private texture is the fallback */
    {
        id<MTLTexture> surf = [self compatSurfacePresentEnabled] ? [self compatSurfaceWrite:w h:h] : nil;
        color = surf ? surf : [_device newTextureWithDescriptor:d];
    }
    color.label = @"MGL default framebuffer color";
    m->default_color = (void *)CFBridgingRetain(color);
    m->default_fb_gen++;            /* new backing textures: descriptor-cache key changes */

    id<MTLTexture> ds = nil;
    MTLPixelFormat dsf = [self fbDefaultDepthStencilFormat];
    if (dsf != MTLPixelFormatInvalid)
    {
        MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:dsf width:w height:h mipmapped:NO];
        dd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        dd.storageMode = MTLStorageModePrivate;
        ds = [_device newTextureWithDescriptor:dd];
        ds.label = @"MGL default framebuffer depth/stencil";
        m->default_depth_stencil = (void *)CFBridgingRetain(ds);
    }

    m->default_width = (GLuint)w;
    m->default_height = (GLuint)h;

    /* defined initial contents: color 0, depth 1, stencil 0 */
    [self fbEnsureCommandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = color;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (ds && fbFormatHasDepth(dsf))
    {
        rp.depthAttachment.texture = ds;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.clearDepth = 1.0;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (ds && fbFormatHasStencil(dsf))
    {
        rp.stencilAttachment.texture = ds;
        rp.stencilAttachment.loadAction = MTLLoadActionClear;
        rp.stencilAttachment.clearStencil = 0;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    [enc endEncoding];

    STATE(dirty_bits) |= DIRTY_FBO;
    return color;
}

#pragma mark attachment resolution

/* new texture with the same contents and RenderTarget usage (GL textures are
 * created before anyone knows they will be rendered to) */
- (id<MTLTexture>) fbPromoteToRenderTarget:(Texture *)tex
{
    id<MTLTexture> old = (__bridge id<MTLTexture>)tex->mtl_data;
    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = old.textureType;
    d.pixelFormat = old.pixelFormat;
    d.width = old.width; d.height = old.height; d.depth = old.depth;
    d.mipmapLevelCount = old.mipmapLevelCount;
    d.arrayLength = old.arrayLength;
    d.sampleCount = old.sampleCount;
    d.storageMode = old.storageMode == MTLStorageModeMemoryless ? MTLStorageModePrivate : old.storageMode;
    d.usage = old.usage | MTLTextureUsageRenderTarget;
    d.swizzle = old.swizzle;
    id<MTLTexture> nt = [_device newTextureWithDescriptor:d];
    if (!nt) return old;

    if (_currentRenderEncoder) [self endRenderEncoding];
    [self fbEnsureCommandBuffer];
    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    [blit copyFromTexture:old toTexture:nt];
    [blit endEncoding];

    CFBridgingRelease(tex->mtl_data);
    tex->mtl_data = (void *)CFBridgingRetain(nt);
    tex->is_render_target = GL_TRUE;
    return nt;
}

/* plain-UNORM view of an _sRGB texture: rendering and clearing through it skips
 * Metal's automatic sRGB encode.  GL 4.6: fragment output converts linear->sRGB
 * only while GL_FRAMEBUFFER_SRGB is enabled (off by default), and glClear applies
 * NO transfer at all (the clear color is stored raw — Mesa parity, and what the
 * CTS readback model assumes: reads return the stored bytes raw). */
static bool fbPixelIsSRGB(MTLPixelFormat f)
{
    switch (f)
    {
        case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatR8Unorm_sRGB:    case MTLPixelFormatRG8Unorm_sRGB:
            return true;
        default:
            return false;
    }
}

- (id<MTLTexture>) fbPlainUnormView:(id<MTLTexture>)t
{
    MTLPixelFormat plain = MTLPixelFormatInvalid;
    switch (t.pixelFormat)
    {
        case MTLPixelFormatRGBA8Unorm_sRGB: plain = MTLPixelFormatRGBA8Unorm; break;
        case MTLPixelFormatBGRA8Unorm_sRGB: plain = MTLPixelFormatBGRA8Unorm; break;
        case MTLPixelFormatR8Unorm_sRGB:    plain = MTLPixelFormatR8Unorm; break;
        case MTLPixelFormatRG8Unorm_sRGB:   plain = MTLPixelFormatRG8Unorm; break;
        default: break;
    }
    if (plain == MTLPixelFormatInvalid) return t;
    NSNumber *key = [NSNumber numberWithUnsignedLongLong:(uintptr_t)(__bridge void *)t];
    NSMutableDictionary *cache = [self fbCache];
    NSString *vkey = [NSString stringWithFormat:@"srgbview:%@", key];
    id<MTLTexture> v = cache[vkey];
    if (!v)
    {
        v = [t newTextureViewWithPixelFormat:plain];
        if (v) cache[vkey] = v;
    }
    return v ? v : t;
}

- (id<MTLTexture>) fbTextureForAttachment:(FBOAttachment *)att renderTarget:(bool)rt
{
    Texture *tex = mgl_fb_attachment_texture(ctx, att);
    if (!tex) return nil;

    if (att->textarget == GL_RENDERBUFFER)
    {
        /* compat s34: renderbuffers used to be returned in their _sRGB Metal
         * format, so clears (always) and fragment writes (while
         * GL_FRAMEBUFFER_SRGB is off) picked up Metal's automatic sRGB encode —
         * unlike texture attachments, which the r52 code below routed through
         * the plain-UNORM view.  Same swap for renderbuffers now
         * (KHR-GL46.direct_state_access.renderbuffers_storage{,_multisample}
         * cleared 0.25 into SRGB8_ALPHA8 and read back 0.537255 = sRGB(0.25)). */
        if (rt && !ctx->state.caps.framebuffer_srgb)
            return [self fbPlainUnormView:(__bridge id<MTLTexture>)tex->mtl_data];
        return (__bridge id<MTLTexture>)tex->mtl_data;
    }

    if (rt && !tex->mtl_data)
        tex->is_render_target = GL_TRUE;

    if (![self bindMTLTexture:tex] || !tex->mtl_data)
        return nil;

    id<MTLTexture> t = (__bridge id<MTLTexture>)tex->mtl_data;
    if (rt && !(t.usage & MTLTextureUsageRenderTarget))
        t = [self fbPromoteToRenderTarget:tex];

    /* GL 4.6: writes into an sRGB-encoded color attachment convert linear to
     * sRGB only while GL_FRAMEBUFFER_SRGB is enabled (it is disabled by
     * default).  Metal would encode automatically for an _sRGB render target,
     * so render through a plain-UNORM view until the cap is turned on. */
    if (rt && !ctx->state.caps.framebuffer_srgb)
        t = [self fbPlainUnormView:t];
    return t;
}

- (void) fbFillTarget:(FBTarget *)t fbo:(Framebuffer *)fbo att:(FBOAttachment *)att tex:(id<MTLTexture>)tex
{
    GLuint level, slice, plane, layers;
    mgl_fb_attachment_slice(ctx, fbo, att, &level, &slice, &plane, &layers);
    t->tex = tex;
    t->level = level;
    t->slice = slice;
    t->plane = plane;
    t->layers = layers;
    /* GL object names are unique per context (never recycled), so the descriptor
     * cache cannot alias two different attachments that happen to reuse a Metal
     * texture address (see FBTarget) */
    t->name = att->texture;
}

/* render targets of a draw framebuffer */
- (bool) fbCollectTargets:(Framebuffer *)fbo into:(FBTargets *)T renderTarget:(bool)rt
{
    memset(T, 0, sizeof(*T));
    NSUInteger w = NSUIntegerMax, h = NSUIntegerMax, layers = NSUIntegerMax;
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);

#define ACCUM(_t_) do { NSUInteger lw = MAX((NSUInteger)1, (_t_).tex.width >> (_t_).level), lh = MAX((NSUInteger)1, (_t_).tex.height >> (_t_).level); \
        if (lw < w) w = lw; if (lh < h) h = lh; if ((_t_).layers < layers) layers = (_t_).layers; \
        T->samples = (_t_).tex.sampleCount; T->any = true; } while (0)

    if (!fbo)
    {
        id<MTLTexture> color = rt ? [self fbEnsureDefaultFramebuffer] : (__bridge id<MTLTexture>)FBM(ctx)->default_color;
        id<MTLTexture> dst = (__bridge id<MTLTexture>)FBM(ctx)->default_depth_stencil;
        /* the default framebuffer has no GL name; the generation of its backing
         * textures is the cache identity (bumped on every re-creation) */
        GLuint def_name = (GLuint)FBM(ctx)->default_fb_gen;
        if (ds->num_draw_buffers > 0 && ds->draw_buffers[0] != GL_NONE && color)
        {
            T->color[0].tex = color; T->color[0].layers = 1; T->color[0].name = def_name;
            T->color_att_bit[0] = 1u;
            ACCUM(T->color[0]);
        }
        if (dst && fbFormatHasDepth(dst.pixelFormat)) { T->depth.tex = dst; T->depth.layers = 1; T->depth.name = def_name; ACCUM(T->depth); }
        if (dst && fbFormatHasStencil(dst.pixelFormat)) { T->stencil.tex = dst; T->stencil.layers = 1; T->stencil.name = def_name; ACCUM(T->stencil); }
        if (!T->any && color) { w = color.width; h = color.height; layers = 1; T->samples = 1; }
    }
    else
    {
        for (GLsizei i = 0; i < ds->num_draw_buffers && i < MAX_COLOR_ATTACHMENTS; i++)
        {
            GLenum b = ds->draw_buffers[i];
            if (b < GL_COLOR_ATTACHMENT0 || b >= GL_COLOR_ATTACHMENT0 + MAX_COLOR_ATTACHMENTS) continue;
            FBOAttachment *att = &fbo->color_attachments[b - GL_COLOR_ATTACHMENT0];
            if (!mgl_fb_attachment_size(ctx, att, NULL, NULL, NULL)) continue;
            id<MTLTexture> tex = [self fbTextureForAttachment:att renderTarget:rt];
            if (!tex) continue;
            [self fbFillTarget:&T->color[i] fbo:fbo att:att tex:tex];
            T->color_att_bit[i] = 1u << (b - GL_COLOR_ATTACHMENT0);
            ACCUM(T->color[i]);
        }
        if (mgl_fb_attachment_size(ctx, &fbo->depth, NULL, NULL, NULL))
        {
            id<MTLTexture> tex = [self fbTextureForAttachment:&fbo->depth renderTarget:rt];
            if (tex && fbFormatHasDepth(tex.pixelFormat)) { [self fbFillTarget:&T->depth fbo:fbo att:&fbo->depth tex:tex]; ACCUM(T->depth); }
        }
        if (mgl_fb_attachment_size(ctx, &fbo->stencil, NULL, NULL, NULL))
        {
            id<MTLTexture> tex = [self fbTextureForAttachment:&fbo->stencil renderTarget:rt];
            if (tex && fbFormatHasStencil(tex.pixelFormat)) { [self fbFillTarget:&T->stencil fbo:fbo att:&fbo->stencil tex:tex]; ACCUM(T->stencil); }
        }
        if (!T->any)
        {
            w = fbo->default_width; h = fbo->default_height;
            layers = fbo->default_layers ? fbo->default_layers : 1;
            T->samples = fbo->default_samples > 1 ? [self fbRoundSamples:fbo->default_samples] : 1;
        }
    }
#undef ACCUM

    if (w == NSUIntegerMax || w == 0 || h == 0) return false;
    T->width = w; T->height = h;
    T->layers = (layers == NSUIntegerMax || layers == 0) ? 1 : layers;
    if (T->samples == 0) T->samples = 1;
    return true;
}

static void fbSetAttachment(MTLRenderPassAttachmentDescriptor *a, const FBTarget *t, MTLLoadAction load)
{
    a.texture = t->tex;
    a.level = t->level;
    a.slice = t->slice;
    a.depthPlane = t->plane;
    a.loadAction = load;
    a.storeAction = MTLStoreActionStore;
}

/* pass for the draw framebuffer fbo; req: fast-path clear folded into the load actions */
- (MTLRenderPassDescriptor *) fbRenderPassFor:(Framebuffer *)fbo targets:(FBTargets *)T clear:(const MGLFBClearRequest *)req
{
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, fbo);
    GLuint inv = ds->invalid_mask;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        if (!T->color[i].tex) continue;
        MTLLoadAction load = (inv & T->color_att_bit[i]) ? MTLLoadActionDontCare : MTLLoadActionLoad;
        if (req && (req->color_slots & (1u << i)))
        {
            double v[4];
            fbValueAsDoubles(req->color_kind[i], req->color[i].f, v);
            int k = fbKindForPixelFormat(T->color[i].tex.pixelFormat);
            if (k == MGL_FB_KIND_UINT) for (int c = 0; c < 4; c++) v[c] = v[c] < 0 ? 0 : floor(v[c]);
            if (k == MGL_FB_KIND_INT) for (int c = 0; c < 4; c++) v[c] = trunc(v[c]);
            /* compat s34: with GL_FRAMEBUFFER_SRGB on the clear still targets the
             * _sRGB texture, and Metal ENCODES its clear color.  GL applies no
             * transfer on clear (the color is stored raw, like Mesa and like the
             * plain-UNORM path above), so pre-decode to cancel the encode. */
            switch (T->color[i].tex.pixelFormat)
            {
                case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatBGRA8Unorm_sRGB:
                case MTLPixelFormatR8Unorm_sRGB:    case MTLPixelFormatRG8Unorm_sRGB:
                    for (int c = 0; c < 3; c++)
                    {
                        double x = v[c] < 0.0 ? 0.0 : (v[c] > 1.0 ? 1.0 : v[c]);
                        v[c] = x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
                    }
                    break;
                default: break;
            }
            /* geom r58: an alpha-less GL format clears A to 1.0 (GL 4.6: reads of a
             * missing alpha channel return one); the RGBA Metal backing would
             * otherwise store the clear color's alpha (texture_swizzle.functional). */
            if (v[3] != 1.0 && mgl_fb_slot_lacks_alpha(ctx, fbo, (GLuint)i)) v[3] = 1.0;
            rp.colorAttachments[i].clearColor = MTLClearColorMake(v[0], v[1], v[2], v[3]);
            load = MTLLoadActionClear;
        }
        fbSetAttachment(rp.colorAttachments[i], &T->color[i], load);
    }

    if (T->depth.tex)
    {
        MTLLoadAction load = (inv & MGL_FB_DEPTH_BIT) ? MTLLoadActionDontCare : MTLLoadActionLoad;
        if (req && req->depth) { load = MTLLoadActionClear; rp.depthAttachment.clearDepth = req->depth_value; }
        fbSetAttachment(rp.depthAttachment, &T->depth, load);
    }
    if (T->stencil.tex)
    {
        MTLLoadAction load = (inv & MGL_FB_STENCIL_BIT) ? MTLLoadActionDontCare : MTLLoadActionLoad;
        if (req && req->stencil) { load = MTLLoadActionClear; rp.stencilAttachment.clearStencil = (uint32_t)req->stencil_value & 0xff; }
        fbSetAttachment(rp.stencilAttachment, &T->stencil, load);
    }

    /* contents are defined again once this pass stores them */
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) if (T->color[i].tex) inv &= ~T->color_att_bit[i];
    if (T->depth.tex) inv &= ~MGL_FB_DEPTH_BIT;
    if (T->stencil.tex) inv &= ~MGL_FB_STENCIL_BIT;
    ds->invalid_mask = inv;

    rp.renderTargetWidth = T->width;
    rp.renderTargetHeight = T->height;
    if (T->layers > 1) rp.renderTargetArrayLength = T->layers;
    if (!T->any) rp.defaultRasterSampleCount = T->samples;
    return rp;
}

/* called by -newRenderEncoder: sets _renderPassDescriptor for the bound draw framebuffer.
 *
 * perf: a new render encoder needs a fresh MTLRenderPassDescriptor, and the allocation +
 * per-attachment descriptor writes measured ~4 us per encoder — a shadow ping-pong scene
 * builds ten per frame.  The last descriptor per (context, framebuffer, targets,
 * invalid-mask) is cached and reused when nothing changed: the encoder copies the
 * descriptor it is created from, so reuse is legal, and every mutation path of the FBO's
 * attachments goes through fbCollectTargets, whose result is part of the cache key (the
 * invalid-mask is read BEFORE fbRenderPassFor clears its bits, and the exact
 * (targets, invalid-mask) pair must repeat for a hit). */
- (bool) fbBuildRenderPassDescriptor
{
    FBTargets T = {0};
    [self fbInitDefaultFramebuffer];
    if (![self fbCollectTargets:ctx->state.framebuffer into:&T renderTarget:true])
    {
        _renderPassDescriptor = nil;
        return false;
    }

    static struct {
        GLMContext ctx;
        Framebuffer *fbo;
        MTLRenderPassDescriptor *rp;    /* retained (the cache owns it under ARC) */
        FBTargets T;
        GLuint inv;
        uint64_t used;
        bool valid;
    } cache[4];
    static uint64_t tick;
    /* test hook: MGL_FB_PASSDESC_CACHE=0 disables the reuse (A/B) */
    static int env_cache = -1;
    if (env_cache < 0) env_cache = getenv("MGL_FB_PASSDESC_CACHE") ? atoi(getenv("MGL_FB_PASSDESC_CACHE")) : 1;
    MGLFBDrawState *ds = mgl_fb_draw_state(ctx, ctx->state.framebuffer);
    GLuint inv = ds->invalid_mask;      /* read BEFORE fbRenderPassFor clears bits */
    int hit = -1, lru = 0;

    tick++;
    for (int i = 0; i < 4 && env_cache; i++)
    {
        if (cache[i].valid && cache[i].ctx == ctx && cache[i].fbo == ctx->state.framebuffer &&
            cache[i].inv == inv &&
            memcmp(&cache[i].T, &T, sizeof(FBTargets)) == 0)
        {
            hit = i;
            break;
        }
        if (cache[i].used < cache[lru].used)
            lru = i;
    }
    if (hit >= 0)
    {
        cache[hit].used = tick;
        _renderPassDescriptor = cache[hit].rp;
        ctx->state.clear_bitmask = 0;
        if (getenv("MGL_DEBUG"))
            fprintf(stderr, "MGL fb: pass-descriptor cache HIT (fbo=%p)\n", (void *)ctx->state.framebuffer);
        return _renderPassDescriptor != nil;
    }

    _renderPassDescriptor = [self fbRenderPassFor:ctx->state.framebuffer targets:&T clear:NULL];
    ctx->state.clear_bitmask = 0;

    if (_renderPassDescriptor && env_cache)
    {
        cache[lru].ctx = ctx;
        cache[lru].fbo = ctx->state.framebuffer;
        cache[lru].rp = _renderPassDescriptor;
        cache[lru].T = T;
        cache[lru].inv = inv;
        cache[lru].used = tick;
        cache[lru].valid = true;
    }
    return _renderPassDescriptor != nil;
}

/* called by -generatePipelineDescriptor */
- (void) fbApplyPipelineFormats:(MTLRenderPipelineDescriptor *)desc
{
    FBTargets T;
    Framebuffer *fbo = ctx->state.framebuffer;

    if (!fbo)
    {
        MGLFBDrawState *ds = mgl_fb_draw_state(ctx, NULL);
        if (ds->num_draw_buffers > 0 && ds->draw_buffers[0] != GL_NONE)
            desc.colorAttachments[0].pixelFormat = [self fbDefaultColorFormat];
        MTLPixelFormat dsf = [self fbDefaultDepthStencilFormat];
        if (fbFormatHasDepth(dsf)) desc.depthAttachmentPixelFormat = dsf;
        if (fbFormatHasStencil(dsf)) desc.stencilAttachmentPixelFormat = dsf;
        desc.rasterSampleCount = 1;
        return;
    }

    if (![self fbCollectTargets:fbo into:&T renderTarget:true])
        return;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        if (T.color[i].tex) desc.colorAttachments[i].pixelFormat = T.color[i].tex.pixelFormat;
    if (T.depth.tex) desc.depthAttachmentPixelFormat = T.depth.tex.pixelFormat;
    if (T.stencil.tex) desc.stencilAttachmentPixelFormat = T.stencil.tex.pixelFormat;
    desc.rasterSampleCount = T.samples;
}

#pragma mark shader library / pipeline cache

- (NSMutableDictionary *) fbCache
{
    struct MGLExtMetal_fb *m = FBM(ctx);
    if (!m->pipeline_cache)
        m->pipeline_cache = (void *)CFBridgingRetain([NSMutableDictionary dictionary]);
    return (__bridge NSMutableDictionary *)m->pipeline_cache;
}

- (id<MTLLibrary>) fbLibrary:(NSString *)source key:(NSString *)key
{
    NSMutableDictionary *cache = [self fbCache];
    NSString *k = [@"lib:" stringByAppendingString:key];
    id<MTLLibrary> lib = cache[k];
    if (lib) return lib;
    NSError *err = nil;
    lib = [_device newLibraryWithSource:source options:nil error:&err];
    if (!lib)
    {
        NSLog(@"MGL fb: shader compile failed (%@): %@", key, err);
        return nil;
    }
    cache[k] = lib;
    return lib;
}

static const char *fbKindType(int k) { return k == MGL_FB_KIND_INT ? "int4" : (k == MGL_FB_KIND_UINT ? "uint4" : "float4"); }

#pragma mark clears

typedef struct FBClearValues {
    float f[MAX_COLOR_ATTACHMENTS][4];
    int32_t i[MAX_COLOR_ATTACHMENTS][4];
    uint32_t u[MAX_COLOR_ATTACHMENTS][4];
} FBClearValues;

static MTLColorWriteMask fbGLWriteMask(GLMContext ctx, int i)
{
    /* state.c keeps the mask in color_writemask only while use_color_mask is set */
    if (!ctx->state.caps.use_color_mask[i])
        return MTLColorWriteMaskAll;
    GLboolean *m = ctx->state.var.color_writemask[i];
    return (m[0] ? MTLColorWriteMaskRed : 0) | (m[1] ? MTLColorWriteMaskGreen : 0) |
           (m[2] ? MTLColorWriteMaskBlue : 0) | (m[3] ? MTLColorWriteMaskAlpha : 0);
}

- (id<MTLRenderPipelineState>) fbClearPipelineFor:(FBTargets *)T req:(const MGLFBClearRequest *)req
{
    NSMutableString *key = [NSMutableString stringWithString:@"clear:"];
    NSMutableString *outs = [NSMutableString string];
    NSMutableString *body = [NSMutableString string];
    NSMutableString *shaderKey = [NSMutableString string];

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        MTLColorWriteMask wm = MTLColorWriteMaskNone;
        if (T->color[i].tex && (req->color_slots & (1u << i)))
        {
            int k = fbKindForPixelFormat(T->color[i].tex.pixelFormat);
            const char *v = k == MGL_FB_KIND_INT ? "i" : (k == MGL_FB_KIND_UINT ? "u" : "f");
            const char *sc = k == MGL_FB_KIND_INT ? "int" : (k == MGL_FB_KIND_UINT ? "uint" : "float");
            [outs appendFormat:@"    %s c%d [[color(%d)]];\n", fbKindType(k), i, i];
            [body appendFormat:@"    o.c%d = %s4(v.%s[%d][0], v.%s[%d][1], v.%s[%d][2], v.%s[%d][3]);\n", i, sc, v, i, v, i, v, i, v, i];
            [shaderKey appendFormat:@"%d%s", i, v];
            wm = fbGLWriteMask(ctx, i);
        }
        [key appendFormat:@"%lu/%lu,", T->color[i].tex ? (unsigned long)T->color[i].tex.pixelFormat : 0ul, (unsigned long)wm];
    }
    [key appendFormat:@"d%lu,s%lu,n%lu", T->depth.tex ? (unsigned long)T->depth.tex.pixelFormat : 0ul,
         T->stencil.tex ? (unsigned long)T->stencil.tex.pixelFormat : 0ul, (unsigned long)T->samples];
    [key appendString:shaderKey];

    /* geom: a layered attachment (whole 2D-array / 3D bound with
     * glFramebufferTexture) must clear on EVERY layer.  The load-action path
     * already does (Metal clears all array layers); the scissored/masked
     * full-screen draw needs a vertex function that writes
     * render_target_array_index, drawn once per layer (instance count). */
    const bool layered = T->layers > 1;
    if (layered) [key appendString:@",L"];

    NSMutableDictionary *cache = [self fbCache];
    id<MTLRenderPipelineState> ps = cache[key];
    if (ps) return ps;

    NSString *vs = @"#include <metal_stdlib>\nusing namespace metal;\n"
                    "struct ClearVals { float f[8][4]; int i[8][4]; uint u[8][4]; };\n"
                    "struct VOut { float4 pos [[position]]; };\n"
                    "vertex VOut fb_clear_vs(uint vid [[vertex_id]], constant float &z [[buffer(0)]]) {\n"
                    "    float2 p = float2((vid << 1) & 2, vid & 2); VOut o; o.pos = float4(p * 2.0 - 1.0, z, 1.0); return o; }\n";
    if (layered)
        vs = @"#include <metal_stdlib>\nusing namespace metal;\n"
              "struct ClearVals { float f[8][4]; int i[8][4]; uint u[8][4]; };\n"
              "struct VOut { float4 pos [[position]]; uint rtl [[render_target_array_index]]; };\n"
              "vertex VOut fb_clear_vs(uint vid [[vertex_id]], uint iid [[instance_id]], constant float &z [[buffer(0)]]) {\n"
              "    float2 p = float2((vid << 1) & 2, vid & 2); VOut o; o.pos = float4(p * 2.0 - 1.0, z, 1.0); o.rtl = iid; return o; }\n";
    NSString *src;
    if (outs.length)
        src = [NSString stringWithFormat:@"%@struct FOut {\n%@};\n"
               "fragment FOut fb_clear_fs(VOut in [[stage_in]], constant ClearVals &v [[buffer(0)]]) {\n    FOut o;\n%@    return o; }\n",
               vs, outs, body];
    else
        src = [vs stringByAppendingString:@"fragment void fb_clear_fs(VOut in [[stage_in]]) { }\n"];

    id<MTLLibrary> lib = [self fbLibrary:src key:(layered ? [@"clearL" stringByAppendingString:shaderKey]
                                                         : [@"clear" stringByAppendingString:shaderKey])];
    if (!lib) return nil;

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.label = @"MGL clear";
    pd.vertexFunction = [lib newFunctionWithName:@"fb_clear_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fb_clear_fs"];
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        if (!T->color[i].tex) continue;
        pd.colorAttachments[i].pixelFormat = T->color[i].tex.pixelFormat;
        pd.colorAttachments[i].writeMask = (req->color_slots & (1u << i)) ? fbGLWriteMask(ctx, i) : MTLColorWriteMaskNone;
    }
    if (T->depth.tex) pd.depthAttachmentPixelFormat = T->depth.tex.pixelFormat;
    if (T->stencil.tex) pd.stencilAttachmentPixelFormat = T->stencil.tex.pixelFormat;
    pd.rasterSampleCount = T->samples;
    if (layered) pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;

    NSError *err = nil;
    ps = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) { NSLog(@"MGL fb: clear pipeline failed: %@", err); return nil; }
    cache[key] = ps;
    return ps;
}

- (MTLScissorRect) fbScissorFor:(NSUInteger)w h:(NSUInteger)h
{
    MTLScissorRect r = { 0, 0, w, h };
    if (ctx->state.caps.scissor_test)
    {
        NSInteger x0 = (GLint)STATE_VAR(scissor_box[0]), y0 = (GLint)STATE_VAR(scissor_box[1]);
        NSInteger x1 = x0 + (NSInteger)STATE_VAR(scissor_box[2]), y1 = y0 + (NSInteger)STATE_VAR(scissor_box[3]);
        x0 = MAX(x0, 0); y0 = MAX(y0, 0);
        x1 = MIN(x1, (NSInteger)w); y1 = MIN(y1, (NSInteger)h);
        if (x1 <= x0 || y1 <= y0) { r.width = 0; r.height = 0; return r; }
        r.x = x0; r.y = y0; r.width = x1 - x0; r.height = y1 - y0;
    }
    return r;
}

- (void) fbClear:(const MGLFBClearRequest *)req
{
    FBTargets T;

    [self endRenderEncoding];
    [self fbEnsureCommandBuffer];
    [self fbInitDefaultFramebuffer];

    if (![self fbCollectTargets:req->fbo into:&T renderTarget:true])
        return;

    MTLScissorRect sc = [self fbScissorFor:T.width h:T.height];
    if (sc.width == 0 || sc.height == 0)
        return;

    bool full = sc.x == 0 && sc.y == 0 && sc.width == T.width && sc.height == T.height;
    bool masks_full = true;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        if ((req->color_slots & (1u << i)) && T.color[i].tex && fbGLWriteMask(ctx, i) != MTLColorWriteMaskAll)
            masks_full = false;
    if (req->depth && T.depth.tex && !STATE_VAR(depth_writemask)) masks_full = false;
    if (req->stencil && T.stencil.tex && (STATE_VAR(stencil_writemask) & 0xff) != 0xff) masks_full = false;

    if (full && masks_full)
    {
        MGLFBClearRequest r = *req;
        if (!T.depth.tex) r.depth = GL_FALSE;
        if (!T.stencil.tex) r.stencil = GL_FALSE;
        MTLRenderPassDescriptor *rp = [self fbRenderPassFor:req->fbo targets:&T clear:&r];
        id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
        enc.label = @"MGL clear (load action)";
        [enc endEncoding];
        return;
    }

    /* scissored / masked: draw a full-screen triangle */
    id<MTLRenderPipelineState> ps = [self fbClearPipelineFor:&T req:req];
    if (!ps) return;

    MTLRenderPassDescriptor *rp = [self fbRenderPassFor:req->fbo targets:&T clear:NULL];
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    enc.label = @"MGL clear (draw)";
    [enc setRenderPipelineState:ps];
    [enc setViewport:(MTLViewport){ 0, 0, (double)T.width, (double)T.height, 0.0, 1.0 }];
    [enc setScissorRect:sc];
    [enc setCullMode:MTLCullModeNone];

    if (T.depth.tex || T.stencil.tex)
    {
        MTLDepthStencilDescriptor *dsd = [[MTLDepthStencilDescriptor alloc] init];
        dsd.depthCompareFunction = MTLCompareFunctionAlways;
        dsd.depthWriteEnabled = (req->depth && T.depth.tex && STATE_VAR(depth_writemask));
        if (T.stencil.tex)
        {
            MTLStencilDescriptor *sd = [[MTLStencilDescriptor alloc] init];
            sd.stencilCompareFunction = MTLCompareFunctionAlways;
            sd.stencilFailureOperation = MTLStencilOperationKeep;
            sd.depthFailureOperation = MTLStencilOperationKeep;
            sd.depthStencilPassOperation = req->stencil ? MTLStencilOperationReplace : MTLStencilOperationKeep;
            sd.readMask = 0xff;
            sd.writeMask = req->stencil ? (STATE_VAR(stencil_writemask) & 0xff) : 0;
            dsd.frontFaceStencil = sd;
            dsd.backFaceStencil = sd;
        }
        [enc setDepthStencilState:[_device newDepthStencilStateWithDescriptor:dsd]];
        [enc setStencilReferenceValue:(uint32_t)req->stencil_value & 0xff];
    }

    FBClearValues vals;
    memset(&vals, 0, sizeof(vals));
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        if (!(req->color_slots & (1u << i))) continue;
        double v[4];
        fbValueAsDoubles(req->color_kind[i], req->color[i].f, v);
        for (int c = 0; c < 4; c++)
        {
            vals.f[i][c] = (float)v[c];
            vals.i[i][c] = (int32_t)fmax(fmin(v[c], 2147483647.0), -2147483648.0);
            vals.u[i][c] = (uint32_t)fmax(fmin(v[c], 4294967295.0), 0.0);
        }
    }
    float z = req->depth ? req->depth_value : 0.0f;
    [enc setVertexBytes:&z length:sizeof(z) atIndex:0];
    [enc setFragmentBytes:&vals length:sizeof(vals) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3
           instanceCount:(T.layers > 1 ? T.layers : 1)];
    [enc endEncoding];
}

#pragma mark readback

#pragma mark RGB9_E5 renderbuffer emulation (compat s36)

/* Re-apply the GL_RGB9_E5 (5-bit shared exponent, bias 15 / 9-bit mantissa)
 * lattice to an RGBA16F renderbuffer backing IN PLACE, right before the
 * attachment is consumed as a read / copy-out source (glReadPixels,
 * glBlitFramebuffer, glCopyTex*: both acquisition points fbSourceFor and
 * texdataReadSource call this).  GL 4.6 2.3.5.2 encode + decode in one pass.
 * Every 5-9-9-9 value is exactly representable as half float (min 2^-24,
 * max sharedExpMax 65504 == half max), and the pass is idempotent, so after it
 * the backing IS the exact 5-9-9-9 image and every consumer — including
 * readPixels(GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV) — returns spec-exact values.
 * Documented caveats of the emulation:
 *  - blending runs in half precision BEFORE the lattice is applied; GL does
 *    not require blending for RGB9_E5 (not a mandated renderable format), and
 *    blended results stay within one lattice step (2^-9 relative) of a
 *    hypothetical true 5-9-9-9 surface.
 *  - multisample RGB9_E5 renderbuffers are stored per-sample in half and are
 *    NOT per-sample quantised (the MSAA resolve filter is implementation
 *    defined for float formats anyway); single-sample attachments are exact.
 *  - the fragment store rounds to half, then this pass rounds to the lattice:
 *    a fragment value that lands within a quarter lattice step BELOW a tie can
 *    end up one mantissa step above what a true 5-9-9-9 target would store. */
- (void) fbQuantizeRGB9E5:(id<MTLTexture>)tex
{
    if (!tex || tex.sampleCount > 1 || tex.textureType != MTLTextureType2D ||
        tex.width == 0 || tex.height == 0)
        return;

    [self endRenderEncoding];
    [self fbEnsureCommandBuffer];
    if (!_currentCommandBuffer) return;

    id<MTLComputePipelineState> ps = [self fbCache][@"ps:rgb9e5"];
    if (!ps)
    {
        NSString *src = @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "/* GL 4.6 2.3.5.2: float -> shared exponent (N=9, B=15, E_max=31) and back */\n"
        "kernel void mgl_rgb9e5_quantize(texture2d<float, access::read_write> t [[texture(0)]],\n"
        "                                uint2 gid [[thread_position_in_grid]])\n"
        "{\n"
        "    if (gid.x >= t.get_width() || gid.y >= t.get_height()) return;\n"
        "    float4 c = t.read(gid);\n"
        "    float3 v = clamp(c.rgb, 0.0f, 65504.0f);   /* sharedExpMax = 511/512 * 2^16 */\n"
        "    float mx = max(v.x, max(v.y, v.z));\n"
        "    float e = max(-16.0f, floor(log2(mx))) + 16.0f;  /* max(-B-1, floor(log2(max))) + 1 + B */\n"
        "    float scale = exp2(e - 24.0f);             /* 2^(E - B - N) */\n"
        "    float max_s = floor(mx / scale + 0.5f);\n"
        "    if (!(max_s >= 0.0f && max_s < 512.0f)) scale *= 2.0f;   /* E = E_p + 1 */\n"
        "    float3 q = float3(floor(v / scale + 0.5f)) * scale;      /* F * 2^(E-B-N) */\n"
        "    t.write(float4(q, c.a), gid);\n"
        "}\n";
        id<MTLLibrary> lib = [self fbLibrary:src key:@"rgb9e5"];
        id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"mgl_rgb9e5_quantize"] : nil;
        NSError *err = nil;
        ps = fn ? [_device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps)
        {
            NSLog(@"MGL fb: rgb9e5 quantize pipeline failed (%@)", err);
            return;
        }
        [self fbCache][@"ps:rgb9e5"] = ps;
    }

    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
    enc.label = @"MGL rgb9e5 quantize";
    [enc setComputePipelineState:ps];
    [enc setTexture:tex atIndex:0];
    MTLSize groups = MTLSizeMake((tex.width + 15) / 16, (tex.height + 15) / 16, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
}

- (bool) fbSourceFor:(Framebuffer *)fbo aspect:(int)aspect attachment:(GLenum)attachment out:(FBTarget *)t
{
    memset(t, 0, sizeof(*t));
    if (!fbo)
    {
        /* readbacks must see the surface that was just presented: do NOT go
         * through fbEnsureDefaultFramebuffer here — the compat direct-present
         * path rotates the write slot in it (only writes rotate) */
        struct MGLExtMetal_fb *fm = FBM(ctx);
        id<MTLTexture> color = (__bridge id<MTLTexture>)fm->default_color;
        if (!color) color = [self fbEnsureDefaultFramebuffer];
        t->tex = (aspect == MGL_FB_ASPECT_COLOR) ? color : (__bridge id<MTLTexture>)fm->default_depth_stencil;
        t->layers = 1;
        return t->tex != nil;
    }

    FBOAttachment *att = aspect == MGL_FB_ASPECT_COLOR ? mgl_fb_attachment(fbo, attachment) :
                         aspect == MGL_FB_ASPECT_DEPTH ? &fbo->depth : &fbo->stencil;
    if (!att || !mgl_fb_attachment_size(ctx, att, NULL, NULL, NULL)) return false;
    id<MTLTexture> tex = [self fbTextureForAttachment:att renderTarget:false];
    if (!tex) return false;
    /* RGB9_E5 renderbuffer: snap the RGBA16F backing onto the 5-9-9-9 lattice
     * before it is used as a read/copy-out source (see fbQuantizeRGB9E5) */
    if (aspect == MGL_FB_ASPECT_COLOR && att->textarget == GL_RENDERBUFFER &&
        FB_RBEX(att->buf.rbo)->internalformat == GL_RGB9_E5)
        [self fbQuantizeRGB9E5:tex];
    [self fbFillTarget:t fbo:fbo att:att tex:tex];
    if (t->layers > 1) t->layers = 1;   /* layered attachments read layer 0 */
    return true;
}

- (int) fbRead:(const MGLFBReadRequest *)req
{
    FBTarget src;

    [self endRenderEncoding];
    [self fbEnsureCommandBuffer];

    if (![self fbSourceFor:req->fbo aspect:req->aspect attachment:req->attachment out:&src])
        return -1;
    if (src.tex.sampleCount > 1)
        return -1;

    MTLPixelFormat pf = src.tex.pixelFormat;
    int kind = req->aspect == MGL_FB_ASPECT_COLOR ? fbKindForPixelFormat(pf) : MGL_FB_KIND_FLOAT;
    NSUInteger bpp = fbBytesPerPixel(pf, req->aspect);
    if (bpp == 0) return -1;

    NSInteger W = MAX((NSUInteger)1, src.tex.width >> src.level), H = MAX((NSUInteger)1, src.tex.height >> src.level);
    NSInteger x0 = MAX((NSInteger)req->x, 0), y0 = MAX((NSInteger)req->y, 0);
    NSInteger x1 = MIN((NSInteger)req->x + req->width, W), y1 = MIN((NSInteger)req->y + req->height, H);

    if (x1 <= x0 || y1 <= y0)
    {
        [self fbSubmitAndWait];   /* reads still synchronize */
        return kind;
    }

    NSUInteger cw = (NSUInteger)(x1 - x0), ch = (NSUInteger)(y1 - y0);
    NSUInteger bpr = cw * bpp;
    id<MTLBuffer> buf = [_device newBufferWithLength:bpr * ch options:MTLResourceStorageModeShared];
    if (!buf) return -1;

    MTLBlitOption opt = MTLBlitOptionNone;
    if (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatDepth24Unorm_Stencil8)
        opt = req->aspect == MGL_FB_ASPECT_STENCIL ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;

    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    blit.label = @"MGL readback";
    [blit copyFromTexture:src.tex sourceSlice:src.slice sourceLevel:src.level
             sourceOrigin:MTLOriginMake(x0, y0, src.plane) sourceSize:MTLSizeMake(cw, ch, 1)
                 toBuffer:buf destinationOffset:0 destinationBytesPerRow:bpr destinationBytesPerImage:bpr * ch options:opt];
    [blit endEncoding];
    [self fbSubmitAndWait];

    const uint8_t *data = (const uint8_t *)buf.contents;
    for (NSUInteger r = 0; r < ch; r++)
    {
        NSUInteger orow = (NSUInteger)(y0 - req->y) + r;   /* Metal row y == GL row y */
        for (NSUInteger c = 0; c < cw; c++)
        {
            NSUInteger o = orow * (NSUInteger)req->width + (NSUInteger)(x0 - req->x) + c;
            const uint8_t *p = data + r * bpr + c * bpp;
            if (req->aspect == MGL_FB_ASPECT_COLOR)
                fbDecodeColor(pf, p, (uint8_t *)req->out + o * 16);
            else if (req->aspect == MGL_FB_ASPECT_DEPTH)
                ((float *)req->out)[o] = (pf == MTLPixelFormatDepth16Unorm) ? *(const uint16_t *)p / 65535.0f : *(const float *)p;
            else
                ((GLuint *)req->out)[o] = p[0];
        }
    }
    return kind;
}

#pragma mark blits

typedef struct FBBlitParams {
    float dst0[2];
    float scale[2];
    float src0[2];
    float srcSize[2];
    int32_t samples;
    int32_t linear;
} FBBlitParams;

- (id<MTLRenderPipelineState>) fbBlitPipelineSrcKind:(int)sk ms:(bool)ms dst:(id<MTLTexture>)dst
{
    int dk = fbKindForPixelFormat(dst.pixelFormat);
    NSString *key = [NSString stringWithFormat:@"blit:%d%d:%lu:%lu", sk, ms ? 1 : 0, (unsigned long)dst.pixelFormat, (unsigned long)dst.sampleCount];
    NSMutableDictionary *cache = [self fbCache];
    id<MTLRenderPipelineState> ps = cache[key];
    if (ps) return ps;

    const char *scalar = sk == MGL_FB_KIND_INT ? "int" : (sk == MGL_FB_KIND_UINT ? "uint" : "float");
    const char *outT = fbKindType(dk);
    NSString *texT = [NSString stringWithFormat:@"%s<%s>", ms ? "texture2d_ms" : "texture2d", scalar];
    NSString *fetch;
    if (!ms)
        fetch = @"    return OUT(t.read(uint2(ic)));\n";
    else if (sk == MGL_FB_KIND_FLOAT)
        fetch = @"    float4 acc = 0.0; for (int k = 0; k < P.samples; k++) acc += t.read(uint2(ic), k);\n    return OUT(acc / float(P.samples));\n";
    else
        fetch = @"    return OUT(t.read(uint2(ic), 0));\n";
    NSString *lin = (!ms && sk == MGL_FB_KIND_FLOAT) ? @"    if (P.linear != 0) return OUT(t.sample(smp, s / P.srcSize));\n" : @"";

    NSString *src = [NSString stringWithFormat:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "#define OUT(x) %s(x)\n"
         "struct BP { float2 dst0; float2 scale; float2 src0; float2 srcSize; int samples; int linear; };\n"
         "vertex float4 fb_blit_vs(uint vid [[vertex_id]]) { float2 p = float2((vid << 1) & 2, vid & 2); return float4(p * 2.0 - 1.0, 0.0, 1.0); }\n"
         "fragment %s fb_blit_fs(float4 pos [[position]], constant BP &P [[buffer(0)]], %@ t [[texture(0)]], sampler smp [[sampler(0)]]) {\n"
         "    float2 s = P.src0 + (pos.xy - P.dst0) * P.scale;\n"
         "%@"
         "    int2 ic = int2(floor(s));\n"
         "    if (ic.x < 0 || ic.y < 0 || ic.x >= int(P.srcSize.x) || ic.y >= int(P.srcSize.y)) discard_fragment();\n"
         "%@"
         "}\n", outT, outT, texT, lin, fetch];

    id<MTLLibrary> lib = [self fbLibrary:src key:[NSString stringWithFormat:@"blit%d%d%d", sk, ms ? 1 : 0, dk]];
    if (!lib) return nil;

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.label = @"MGL blit";
    pd.vertexFunction = [lib newFunctionWithName:@"fb_blit_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fb_blit_fs"];
    pd.colorAttachments[0].pixelFormat = dst.pixelFormat;
    pd.rasterSampleCount = dst.sampleCount;
    NSError *err = nil;
    ps = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps) { NSLog(@"MGL fb: blit pipeline failed: %@", err); return nil; }
    cache[key] = ps;
    return ps;
}

/* 2D (or 2DMS) view / copy of one level+slice+plane of t, for sampling */
- (id<MTLTexture>) fbSampleable:(const FBTarget *)t forceCopy:(bool)copy
{
    id<MTLTexture> tex = t->tex;
    bool plain = (tex.textureType == MTLTextureType2D || tex.textureType == MTLTextureType2DMultisample) && t->level == 0 && t->slice == 0;
    if (plain && !copy) return tex;

    NSUInteger w = MAX((NSUInteger)1, tex.width >> t->level), h = MAX((NSUInteger)1, tex.height >> t->level);
    if (!copy && tex.textureType != MTLTextureType3D && tex.sampleCount == 1)
    {
        id<MTLTexture> v = [tex newTextureViewWithPixelFormat:tex.pixelFormat textureType:MTLTextureType2D
                                                       levels:NSMakeRange(t->level, 1) slices:NSMakeRange(t->slice, 1)];
        if (v) return v;
    }

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:tex.pixelFormat width:w height:h mipmapped:NO];
    d.textureType = tex.sampleCount > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    d.sampleCount = tex.sampleCount;
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> tmp = [_device newTextureWithDescriptor:d];
    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    [blit copyFromTexture:tex sourceSlice:t->slice sourceLevel:t->level sourceOrigin:MTLOriginMake(0, 0, t->plane)
               sourceSize:MTLSizeMake(w, h, 1) toTexture:tmp destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return tmp;
}

/* shader blit of src rect -> dst rect (GL coordinates, Metal rows == GL rows) */
- (void) fbShaderBlitFrom:(const FBTarget *)src to:(const FBTarget *)dst src:(const GLint *)s dst:(const GLint *)d
                   filter:(GLenum)filter scissor:(bool)useScissor
{
    bool same = src->tex == dst->tex;
    id<MTLTexture> stex = [self fbSampleable:src forceCopy:same];
    if (!stex) return;

    /* compat s34: sRGB-encoding neutrality.  MSL read() on an _sRGB source
     * DECODES, and while GL_FRAMEBUFFER_SRGB is off the dst renders through its
     * plain-UNORM view, so the decoded value would be stored raw
     * (KHR-GL46.direct_state_access.renderbuffers_storage_multisample: clear
     * 0.25 -> resolve blit -> read back decode(0.25)=0.051).  A GL blit between
     * color buffers of the same encoding passes the bytes through unchanged, so
     * blit through plain-UNORM views of BOTH sides whenever the source is _sRGB
     * (a blit whose sides differ in encoding is INVALID_OPERATION, so the dst
     * is then _sRGB too and fbPlainUnormView is a no-op for the already-plain
     * framebuffer_srgb-off dst).  Raw bytes in, raw bytes out — the linear-
     * filter path therefore filters the encoded values, which is exact for the
     * uniform sources the CTS blits and lossless for nearest. */
    FBTarget dstv;
    if (fbPixelIsSRGB(stex.pixelFormat))
    {
        stex = [self fbPlainUnormView:stex];
        dstv = *dst;
        dstv.tex = [self fbPlainUnormView:dst->tex];
        dst = &dstv;
    }

    int sk = fbKindForPixelFormat(stex.pixelFormat);
    bool ms = stex.sampleCount > 1;
    id<MTLRenderPipelineState> ps = [self fbBlitPipelineSrcKind:sk ms:ms dst:dst->tex];
    if (!ps) return;

    NSUInteger DW = MAX((NSUInteger)1, dst->tex.width >> dst->level), DH = MAX((NSUInteger)1, dst->tex.height >> dst->level);

    /* destination rect clipped to the target and the scissor */
    NSInteger rx0 = MIN(d[0], d[2]), rx1 = MAX(d[0], d[2]), ry0 = MIN(d[1], d[3]), ry1 = MAX(d[1], d[3]);
    rx0 = MAX(rx0, 0); ry0 = MAX(ry0, 0); rx1 = MIN(rx1, (NSInteger)DW); ry1 = MIN(ry1, (NSInteger)DH);
    if (useScissor && ctx->state.caps.scissor_test)
    {
        MTLScissorRect sc = [self fbScissorFor:DW h:DH];
        rx0 = MAX(rx0, (NSInteger)sc.x); ry0 = MAX(ry0, (NSInteger)sc.y);
        rx1 = MIN(rx1, (NSInteger)(sc.x + sc.width)); ry1 = MIN(ry1, (NSInteger)(sc.y + sc.height));
    }
    if (rx1 <= rx0 || ry1 <= ry0) return;

    FBBlitParams P;
    P.dst0[0] = d[0]; P.dst0[1] = d[1];
    P.scale[0] = (float)((double)(s[2] - s[0]) / (double)(d[2] - d[0]));
    P.scale[1] = (float)((double)(s[3] - s[1]) / (double)(d[3] - d[1]));
    P.src0[0] = s[0]; P.src0[1] = s[1];
    P.srcSize[0] = stex.width; P.srcSize[1] = stex.height;
    P.samples = (int32_t)stex.sampleCount;
    P.linear = (filter == GL_LINEAR && sk == MGL_FB_KIND_FLOAT && !ms) ? 1 : 0;

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    fbSetAttachment(rp.colorAttachments[0], dst, MTLLoadActionLoad);
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    enc.label = @"MGL blit";
    [enc setRenderPipelineState:ps];
    [enc setViewport:(MTLViewport){ 0, 0, (double)DW, (double)DH, 0.0, 1.0 }];
    [enc setScissorRect:(MTLScissorRect){ (NSUInteger)rx0, (NSUInteger)ry0, (NSUInteger)(rx1 - rx0), (NSUInteger)(ry1 - ry0) }];
    [enc setCullMode:MTLCullModeNone];
    [enc setFragmentBytes:&P length:sizeof(P) atIndex:0];
    [enc setFragmentTexture:stex atIndex:0];
    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = sd.magFilter = P.linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    [enc setFragmentSamplerState:[_device newSamplerStateWithDescriptor:sd] atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}

/* single-sample copy of a multisampled depth/stencil image */
- (id<MTLTexture>) fbResolveDepthStencil:(id<MTLTexture>)ms slice:(NSUInteger)slice
{
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:ms.pixelFormat width:ms.width height:ms.height mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> res = [_device newTextureWithDescriptor:d];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    if (fbFormatHasDepth(ms.pixelFormat))
    {
        rp.depthAttachment.texture = ms;
        rp.depthAttachment.slice = slice;
        rp.depthAttachment.loadAction = MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionMultisampleResolve;
        rp.depthAttachment.resolveTexture = res;
        rp.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
    }
    if (fbFormatHasStencil(ms.pixelFormat))
    {
        rp.stencilAttachment.texture = ms;
        rp.stencilAttachment.slice = slice;
        rp.stencilAttachment.loadAction = MTLLoadActionLoad;
        rp.stencilAttachment.storeAction = MTLStoreActionMultisampleResolve;
        rp.stencilAttachment.resolveTexture = res;
        rp.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
    }
    rp.renderTargetWidth = ms.width;
    rp.renderTargetHeight = ms.height;
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    [enc endEncoding];
    return res;
}

#pragma mark depth/stencil blits into a multisampled image

/* GL 4.6 18.3.1: blitting INTO a multisampled draw framebuffer is legal from a
 * single-sampled read framebuffer with any (scaled/partial) rectangle, nearest
 * filtering only (LINEAR + DEPTH/STENCIL is rejected in framebuffer_ext.c). Every
 * sample of a destination pixel takes the value of the mapped source pixel.
 *
 * Implementation: the source aspect is staged into a shader-readable color texture
 * through a buffer round-trip (packed depth expands to one 32-bit float per pixel,
 * float depth copies verbatim, unorm16 depth keeps its 2 bytes in R16Unorm, stencil
 * bytes land in R8Uint), then a full-screen-triangle render pass into the
 * destination writes [[depth]] and/or [[stencil]]. One invocation per pixel, whose
 * outputs the rasterizer replicates over every sample — exactly the semantics above.
 * Where the device cannot export stencil from a fragment shader, stencil is written
 * with one stencil-REPLACE pass per distinct source value instead. */

enum { DSBLIT_DEPTH = 1, DSBLIT_STENCIL = 2 };

typedef struct DSBlitParams {
    float dst0[2];
    float scale[2];
    float src0[2];
    float srcSize[2];
    int32_t ref;
    int32_t pad;   /* match the float2 alignment of the MSL struct (40 bytes) */
} DSBlitParams;

/* stage one aspect of a single-sampled source as a shader-readable color texture */
- (id<MTLTexture>) fbStageAspect:(const FBTarget *)src aspect:(int)aspect
{
    MTLPixelFormat pf = src->tex.pixelFormat;
    if (aspect == MGL_FB_ASPECT_DEPTH && !fbFormatHasDepth(pf)) return nil;
    if (aspect == MGL_FB_ASPECT_STENCIL && !fbFormatHasStencil(pf)) return nil;
    NSUInteger W = MAX((NSUInteger)1, src->tex.width >> src->level);
    NSUInteger H = MAX((NSUInteger)1, src->tex.height >> src->level);

    NSUInteger bpp = fbBytesPerPixel(pf, aspect);
    if (!bpp) return nil;
    MTLBlitOption opt = MTLBlitOptionNone;
    if (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatDepth24Unorm_Stencil8)
        opt = aspect == MGL_FB_ASPECT_STENCIL ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;

    /* the staged color format matches the buffer bytes pixel for pixel */
    MTLPixelFormat stage = aspect == MGL_FB_ASPECT_DEPTH
        ? (pf == MTLPixelFormatDepth16Unorm ? MTLPixelFormatR16Unorm : MTLPixelFormatR32Float)
        : MTLPixelFormatR8Uint;

    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:stage width:W height:H mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> out = [_device newTextureWithDescriptor:d];
    id<MTLBuffer> buf = [_device newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
    if (!out || !buf) return nil;

    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    blit.label = @"MGL ds blit stage";
    [blit copyFromTexture:src->tex sourceSlice:src->slice sourceLevel:src->level
             sourceOrigin:MTLOriginMake(0, 0, src->plane) sourceSize:MTLSizeMake(W, H, 1)
                   toBuffer:buf destinationOffset:0 destinationBytesPerRow:W * bpp destinationBytesPerImage:W * H * bpp options:opt];
    [blit copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:W * bpp sourceBytesPerImage:W * H * bpp
              sourceSize:MTLSizeMake(W, H, 1) toTexture:out destinationSlice:0 destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return out;
}

/* mode: DSBLIT_DEPTH and/or DSBLIT_STENCIL (fragment exports), 0 = per-reference stencil pass */
- (id<MTLRenderPipelineState>) fbDSBlitPipeline:(int)mode dst:(id<MTLTexture>)dst
{
    NSMutableDictionary *cache = [self fbCache];
    NSString *key = [NSString stringWithFormat:@"dsblit:%d:%lu:%lu", mode, (unsigned long)dst.pixelFormat, (unsigned long)dst.sampleCount];
    id ps = cache[key];
    if (ps) return [ps isEqual:[NSNull null]] ? nil : ps;

    bool wd = (mode & DSBLIT_DEPTH) != 0, exportS = (mode & DSBLIT_STENCIL) != 0, refmode = mode == 0;
    bool needS = exportS || refmode;

    NSMutableString *params = [NSMutableString string];
    NSMutableString *members = [NSMutableString string];
    NSMutableString *body = [NSMutableString string];
    if (wd)
    {
        [params appendString:@", texture2d<float> sd [[texture(0)]]"];
        [members appendString:@"    float z [[depth(any)]];\n"];
        [body appendString:@"    o.z = sd.read(uint2(ic)).r;\n"];
    }
    if (needS)
    {
        [params appendFormat:@", texture2d<uint> ss [[texture(%d)]]", wd ? 1 : 0];
        if (exportS)
        {
            [members appendString:@"    uint st [[stencil]];\n"];
            [body appendString:@"    o.st = ss.read(uint2(ic)).r;\n"];
        }
    }
    if (refmode)
        [body appendString:@"    if (int(ss.read(uint2(ic)).r) != P.ref) discard_fragment();\n"];

    NSString *src = [NSString stringWithFormat:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "struct DP { float2 dst0; float2 scale; float2 src0; float2 srcSize; int ref; };\n"
         "vertex float4 fb_dsblit_vs(uint vid [[vertex_id]]) { float2 p = float2((vid << 1) & 2, vid & 2); return float4(p * 2.0 - 1.0, 0.0, 1.0); }\n"
         "%@fragment %@ fb_dsblit_fs(float4 pos [[position]], constant DP &P [[buffer(0)]]%@) {\n"
         "    float2 s = P.src0 + (pos.xy - P.dst0) * P.scale;\n"
         "    int2 ic = int2(floor(s));\n"
         "    if (ic.x < 0 || ic.y < 0 || ic.x >= int(P.srcSize.x) || ic.y >= int(P.srcSize.y)) discard_fragment();\n"
         "%@%@    %@}\n",
        (wd || exportS) ? [NSString stringWithFormat:@"struct DSO {\n%@};\n", members] : @"",
        (wd || exportS) ? @"DSO" : @"void",
        params,
        (wd || exportS) ? @"    DSO o;\n" : @"",
        body,
        (wd || exportS) ? @"return o;\n" : @""];

    id<MTLLibrary> lib = [self fbLibrary:src key:[NSString stringWithFormat:@"dsblit%d%d%d", wd ? 1 : 0, exportS ? 1 : 0, refmode ? 1 : 0]];
    if (!lib) { cache[key] = [NSNull null]; return nil; }

    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.label = @"MGL ds blit";
    pd.vertexFunction = [lib newFunctionWithName:@"fb_dsblit_vs"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fb_dsblit_fs"];
    pd.rasterSampleCount = dst.sampleCount;
    if (wd && fbFormatHasDepth(dst.pixelFormat)) pd.depthAttachmentPixelFormat = dst.pixelFormat;
    if (needS && fbFormatHasStencil(dst.pixelFormat)) pd.stencilAttachmentPixelFormat = dst.pixelFormat;

    NSError *err = nil;
    ps = [_device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!ps)
    {
        if (exportS) cache[@"dsblit:no-stencil-export"] = @YES;   /* fall back to per-reference passes */
        NSLog(@"MGL fb: depth/stencil blit pipeline (mode %d) failed: %@", mode, err);
        cache[key] = [NSNull null];
        return nil;
    }
    cache[key] = ps;
    return ps;
}

/* 0 = depth only, 1 = depth + stencil, 2 = stencil only (always-pass tests, REPLACE writes) */
- (id<MTLDepthStencilState>) fbDSBlitState:(int)mode
{
    NSMutableDictionary *cache = [self fbCache];
    NSString *key = [NSString stringWithFormat:@"dsblit-state:%d", mode];
    id<MTLDepthStencilState> st = cache[key];
    if (st) return st;

    MTLDepthStencilDescriptor *d = [[MTLDepthStencilDescriptor alloc] init];
    d.depthCompareFunction = MTLCompareFunctionAlways;
    d.depthWriteEnabled = mode != 2;   /* stencil-only passes leave depth untouched */
    if (mode != 0)
    {
        MTLStencilDescriptor *s = [[MTLStencilDescriptor alloc] init];
        s.stencilCompareFunction = MTLCompareFunctionAlways;
        s.stencilFailureOperation = MTLStencilOperationReplace;
        s.depthFailureOperation = MTLStencilOperationReplace;
        s.depthStencilPassOperation = MTLStencilOperationReplace;
        s.writeMask = 0xffffffff;
        d.frontFaceStencil = s;
        d.backFaceStencil = s;
    }
    st = [_device newDepthStencilStateWithDescriptor:d];
    if (st) cache[key] = st;
    return st;
}

/* distinct stencil byte values present in a staged R8Uint texture */
- (bool) fbDSBlitStencilValues:(id<MTLTexture>)staged vals:(uint8_t *)vals count:(int *)count
{
    NSUInteger W = staged.width, H = staged.height;
    id<MTLBuffer> buf = [_device newBufferWithLength:W * H options:MTLResourceStorageModeShared];
    if (!buf) return false;
    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    [blit copyFromTexture:staged sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(W, H, 1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:W
       destinationBytesPerImage:W * H options:MTLBlitOptionNone];
    [blit endEncoding];
    [self fbSubmitAndWait];

    bool seen[256] = { false };
    int n = 0;
    const uint8_t *p = (const uint8_t *)buf.contents;
    for (NSUInteger i = 0; i < W * H; i++)
        if (!seen[p[i]]) { seen[p[i]] = true; vals[n++] = p[i]; }
    *count = n;
    return true;
}

- (void) fbDSBlitEncode:(const FBTarget *)dst pipeline:(id<MTLRenderPipelineState>)ps state:(id<MTLDepthStencilState>)st
                   rect:(MTLScissorRect)rect params:(const DSBlitParams *)P ref:(uint32_t)ref
               depthTex:(id<MTLTexture>)dt stencilTex:(id<MTLTexture>)stex
{
    NSUInteger DW = MAX((NSUInteger)1, dst->tex.width >> dst->level), DH = MAX((NSUInteger)1, dst->tex.height >> dst->level);

    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    /* attach both aspects of a packed depth/stencil texture so the untouched one is preserved */
    if (fbFormatHasDepth(dst->tex.pixelFormat)) fbSetAttachment(rp.depthAttachment, dst, MTLLoadActionLoad);
    if (fbFormatHasStencil(dst->tex.pixelFormat)) fbSetAttachment(rp.stencilAttachment, dst, MTLLoadActionLoad);
    rp.renderTargetWidth = DW;
    rp.renderTargetHeight = DH;

    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    enc.label = @"MGL depth/stencil blit";
    [enc setRenderPipelineState:ps];
    [enc setDepthStencilState:st];
    if (ref != UINT32_MAX) [enc setStencilReferenceValue:ref];
    [enc setViewport:(MTLViewport){ 0, 0, (double)DW, (double)DH, 0.0, 1.0 }];
    [enc setScissorRect:rect];
    [enc setCullMode:MTLCullModeNone];
    [enc setFragmentBytes:P length:sizeof(*P) atIndex:0];
    if (dt) [enc setFragmentTexture:dt atIndex:0];
    if (stex) [enc setFragmentTexture:stex atIndex:dt ? 1 : 0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}

- (void) fbBlitDepthStencilToMS:(const FBTarget *)src to:(const FBTarget *)dst req:(const MGLFBBlitRequest *)req depth:(bool)wantD stencil:(bool)wantS
{
    const GLint *s = req->src, *d = req->dst;
    NSUInteger DW = MAX((NSUInteger)1, dst->tex.width >> dst->level), DH = MAX((NSUInteger)1, dst->tex.height >> dst->level);

    bool dstD = wantD && fbFormatHasDepth(dst->tex.pixelFormat);
    bool dstS = wantS && fbFormatHasStencil(dst->tex.pixelFormat);
    if (!dstD && !dstS) return;

    /* a multisampled read framebuffer (scissored same-size MS->MS blit) is resolved first */
    FBTarget S = *src;
    if (S.tex.sampleCount > 1)
    {
        id<MTLTexture> res = [self fbResolveDepthStencil:S.tex slice:S.slice];
        if (!res) return;
        S.tex = res;
        S.level = S.slice = S.plane = 0;
    }

    /* destination rectangle clipped to the image and the scissor */
    NSInteger rx0 = MIN(d[0], d[2]), rx1 = MAX(d[0], d[2]), ry0 = MIN(d[1], d[3]), ry1 = MAX(d[1], d[3]);
    rx0 = MAX(rx0, 0); ry0 = MAX(ry0, 0); rx1 = MIN(rx1, (NSInteger)DW); ry1 = MIN(ry1, (NSInteger)DH);
    if (ctx->state.caps.scissor_test)
    {
        MTLScissorRect sc = [self fbScissorFor:DW h:DH];
        rx0 = MAX(rx0, (NSInteger)sc.x); ry0 = MAX(ry0, (NSInteger)sc.y);
        rx1 = MIN(rx1, (NSInteger)(sc.x + sc.width)); ry1 = MIN(ry1, (NSInteger)(sc.y + sc.height));
    }
    if (rx1 <= rx0 || ry1 <= ry0) return;

    id<MTLTexture> dt = dstD ? [self fbStageAspect:&S aspect:MGL_FB_ASPECT_DEPTH] : nil;
    id<MTLTexture> st = dstS ? [self fbStageAspect:&S aspect:MGL_FB_ASPECT_STENCIL] : nil;
    if (dstD && !dt) return;
    if (dstS && !st) return;

    DSBlitParams P;
    P.dst0[0] = d[0]; P.dst0[1] = d[1];
    P.scale[0] = (float)((double)(s[2] - s[0]) / (double)(d[2] - d[0]));
    P.scale[1] = (float)((double)(s[3] - s[1]) / (double)(d[3] - d[1]));
    P.src0[0] = s[0]; P.src0[1] = s[1];
    P.srcSize[0] = (float)(dt ? dt.width : st.width);
    P.srcSize[1] = (float)(dt ? dt.height : st.height);
    P.ref = 0;
    P.pad = 0;

    MTLScissorRect rect = { (NSUInteger)rx0, (NSUInteger)ry0, (NSUInteger)(rx1 - rx0), (NSUInteger)(ry1 - ry0) };
    bool exportStencil = ![self fbCache][@"dsblit:no-stencil-export"];

    /* depth — together with stencil in the same pass when export is available */
    if (dstD)
    {
        bool both = dstS && exportStencil;
        id<MTLRenderPipelineState> ps = both ? [self fbDSBlitPipeline:DSBLIT_DEPTH | DSBLIT_STENCIL dst:dst->tex] : nil;
        if (ps)
        {
            [self fbDSBlitEncode:dst pipeline:ps state:[self fbDSBlitState:1] rect:rect params:&P ref:UINT32_MAX
                        depthTex:dt stencilTex:st];
            dstS = false;
        }
        else
        {
            ps = [self fbDSBlitPipeline:DSBLIT_DEPTH dst:dst->tex];
            if (ps)
                [self fbDSBlitEncode:dst pipeline:ps state:[self fbDSBlitState:0] rect:rect params:&P ref:UINT32_MAX
                            depthTex:dt stencilTex:nil];
        }
    }

    /* stencil, on its own */
    if (dstS)
    {
        id<MTLRenderPipelineState> ps = exportStencil ? [self fbDSBlitPipeline:DSBLIT_STENCIL dst:dst->tex] : nil;
        if (ps)
        {
            [self fbDSBlitEncode:dst pipeline:ps state:[self fbDSBlitState:2] rect:rect params:&P ref:UINT32_MAX
                        depthTex:nil stencilTex:st];
        }
        else
        {
            /* no stencil export: one stencil-REPLACE pass per distinct source value */
            id<MTLRenderPipelineState> rp = [self fbDSBlitPipeline:0 dst:dst->tex];
            uint8_t vals[256];
            int n = 0;
            if (rp && [self fbDSBlitStencilValues:st vals:vals count:&n])
                for (int i = 0; i < n; i++)
                {
                    P.ref = vals[i];   /* the shader keeps only matching pixels; REPLACE writes the same value */
                    [self fbDSBlitEncode:dst pipeline:rp state:[self fbDSBlitState:2] rect:rect params:&P ref:vals[i]
                                depthTex:nil stencilTex:st];
                }
        }
    }
}


- (void) fbDepthStencilBlitFrom:(FBTarget)src to:(FBTarget)dst req:(const MGLFBBlitRequest *)req depth:(bool)wantD stencil:(bool)wantS
{
    MTLPixelFormat pf = src.tex.pixelFormat;
    const GLint *s = req->src, *d = req->dst;
    NSInteger SW = MAX((NSUInteger)1, src.tex.width >> src.level), SH = MAX((NSUInteger)1, src.tex.height >> src.level);
    NSInteger DW = MAX((NSUInteger)1, dst.tex.width >> dst.level), DH = MAX((NSUInteger)1, dst.tex.height >> dst.level);

    bool exact = (s[2] - s[0]) == (d[2] - d[0]) && (s[3] - s[1]) == (d[3] - d[1]) && s[2] > s[0] && s[3] > s[1];
    MTLScissorRect sc = [self fbScissorFor:DW h:DH];
    bool scissorClips = ctx->state.caps.scissor_test &&
        !((NSInteger)sc.x <= MAX(d[0], 0) && (NSInteger)sc.y <= MAX(d[1], 0) &&
          (NSInteger)(sc.x + sc.width) >= MIN((NSInteger)d[2], DW) && (NSInteger)(sc.y + sc.height) >= MIN((NSInteger)d[3], DH));
    bool allAspects = (!fbFormatHasDepth(pf) || wantD) && (!fbFormatHasStencil(pf) || wantS);

    if (exact && !scissorClips && allAspects && src.tex.pixelFormat == dst.tex.pixelFormat &&
        src.tex.sampleCount == dst.tex.sampleCount)
    {
        /* clip src/dst consistently against both images */
        NSInteger ox = d[0] - s[0], oy = d[1] - s[1];
        NSInteger x0 = MAX(MAX((NSInteger)s[0], 0), -ox), y0 = MAX(MAX((NSInteger)s[1], 0), -oy);
        NSInteger x1 = MIN(MIN((NSInteger)s[2], SW), DW - ox), y1 = MIN(MIN((NSInteger)s[3], SH), DH - oy);
        if (x1 <= x0 || y1 <= y0) return;
        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        [blit copyFromTexture:src.tex sourceSlice:src.slice sourceLevel:src.level sourceOrigin:MTLOriginMake(x0, y0, src.plane)
                   sourceSize:MTLSizeMake(x1 - x0, y1 - y0, 1) toTexture:dst.tex destinationSlice:dst.slice destinationLevel:dst.level
            destinationOrigin:MTLOriginMake(x0 + ox, y0 + oy, dst.plane)];
        [blit endEncoding];
        return;
    }

    if (dst.tex.sampleCount > 1)
    {
        [self fbBlitDepthStencilToMS:&src to:&dst req:req depth:wantD stencil:wantS];
        return;
    }

    id<MTLTexture> stex = src.tex;
    NSUInteger slevel = src.level, sslice = src.slice, splane = src.plane;
    if (stex.sampleCount > 1) { stex = [self fbResolveDepthStencil:stex slice:sslice]; slevel = 0; sslice = 0; splane = 0; }

    /* CPU nearest resample: fetch source & destination images, combine, upload */
    for (int pass = 0; pass < 2; pass++)
    {
        bool isS = pass == 1;
        if ((isS && !wantS) || (!isS && !wantD)) continue;
        int aspect = isS ? MGL_FB_ASPECT_STENCIL : MGL_FB_ASPECT_DEPTH;
        NSUInteger sb = fbBytesPerPixel(pf, aspect), db = fbBytesPerPixel(dst.tex.pixelFormat, aspect);
        if (sb != db) continue;
        MTLBlitOption so = (fbFormatHasDepth(pf) && fbFormatHasStencil(pf)) ? (isS ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil) : MTLBlitOptionNone;
        MTLBlitOption dopt = (fbFormatHasDepth(dst.tex.pixelFormat) && fbFormatHasStencil(dst.tex.pixelFormat)) ? (isS ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil) : MTLBlitOptionNone;

        id<MTLBuffer> sbuf = [_device newBufferWithLength:SW * SH * sb options:MTLResourceStorageModeShared];
        id<MTLBuffer> dbuf = [_device newBufferWithLength:DW * DH * db options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        [blit copyFromTexture:stex sourceSlice:sslice sourceLevel:slevel sourceOrigin:MTLOriginMake(0, 0, splane) sourceSize:MTLSizeMake(SW, SH, 1)
                     toBuffer:sbuf destinationOffset:0 destinationBytesPerRow:SW * sb destinationBytesPerImage:SW * SH * sb options:so];
        [blit copyFromTexture:dst.tex sourceSlice:dst.slice sourceLevel:dst.level sourceOrigin:MTLOriginMake(0, 0, dst.plane) sourceSize:MTLSizeMake(DW, DH, 1)
                     toBuffer:dbuf destinationOffset:0 destinationBytesPerRow:DW * db destinationBytesPerImage:DW * DH * db options:dopt];
        [blit endEncoding];
        [self fbSubmitAndWait];

        NSInteger rx0 = MAX(MIN(d[0], d[2]), 0), rx1 = MIN(MAX(d[0], d[2]), (GLint)DW);
        NSInteger ry0 = MAX(MIN(d[1], d[3]), 0), ry1 = MIN(MAX(d[1], d[3]), (GLint)DH);
        if (ctx->state.caps.scissor_test)
        {
            rx0 = MAX(rx0, (NSInteger)sc.x); ry0 = MAX(ry0, (NSInteger)sc.y);
            rx1 = MIN(rx1, (NSInteger)(sc.x + sc.width)); ry1 = MIN(ry1, (NSInteger)(sc.y + sc.height));
        }
        double scx = (double)(s[2] - s[0]) / (double)(d[2] - d[0]), scy = (double)(s[3] - s[1]) / (double)(d[3] - d[1]);
        uint8_t *sp = sbuf.contents, *dp = dbuf.contents;
        for (NSInteger y = ry0; y < ry1; y++)
            for (NSInteger x = rx0; x < rx1; x++)
            {
                double fx = s[0] + (x + 0.5 - d[0]) * scx, fy = s[1] + (y + 0.5 - d[1]) * scy;
                NSInteger ix = (NSInteger)floor(fx), iy = (NSInteger)floor(fy);
                if (ix < 0 || iy < 0 || ix >= SW || iy >= SH) continue;
                memcpy(dp + (y * DW + x) * db, sp + (iy * SW + ix) * sb, db);
            }

        blit = [_currentCommandBuffer blitCommandEncoder];
        [blit copyFromBuffer:dbuf sourceOffset:0 sourceBytesPerRow:DW * db sourceBytesPerImage:DW * DH * db sourceSize:MTLSizeMake(DW, DH, 1)
                   toTexture:dst.tex destinationSlice:dst.slice destinationLevel:dst.level destinationOrigin:MTLOriginMake(0, 0, dst.plane) options:dopt];
        [blit endEncoding];
    }
}

- (void) fbBlit:(const MGLFBBlitRequest *)req
{
    [self endRenderEncoding];
    [self fbEnsureCommandBuffer];
    [self fbInitDefaultFramebuffer];

    if (req->mask & GL_COLOR_BUFFER_BIT)
    {
        FBTarget src;
        MGLFBDrawState *rds = mgl_fb_draw_state(ctx, req->read_fbo);
        if ([self fbSourceFor:req->read_fbo aspect:MGL_FB_ASPECT_COLOR attachment:rds->read_buffer out:&src])
        {
            FBTargets T;
            if ([self fbCollectTargets:req->draw_fbo into:&T renderTarget:true])
            {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
                {
                    if (!T.color[i].tex) continue;
                    FBTarget dst = T.color[i];
                    dst.layers = 1;
                    [self fbShaderBlitFrom:&src to:&dst src:req->src dst:req->dst filter:req->filter scissor:true];
                }
            }
        }
    }

    if (req->mask & (GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT))
    {
        bool wd = (req->mask & GL_DEPTH_BUFFER_BIT) != 0, ws = (req->mask & GL_STENCIL_BUFFER_BIT) != 0;
        FBTarget sd;
        int aspect = wd ? MGL_FB_ASPECT_DEPTH : MGL_FB_ASPECT_STENCIL;
        FBTargets T;
        if ([self fbSourceFor:req->read_fbo aspect:aspect attachment:GL_NONE out:&sd] &&
            [self fbCollectTargets:req->draw_fbo into:&T renderTarget:true])
        {
            if (wd && ws && T.depth.tex != T.stencil.tex)
            {
                FBTarget ss;
                if (T.depth.tex) [self fbDepthStencilBlitFrom:sd to:T.depth req:req depth:true stencil:false];
                if ([self fbSourceFor:req->read_fbo aspect:MGL_FB_ASPECT_STENCIL attachment:GL_NONE out:&ss] && T.stencil.tex)
                    [self fbDepthStencilBlitFrom:ss to:T.stencil req:req depth:false stencil:true];
            }
            else
            {
                FBTarget dd = wd ? T.depth : T.stencil;
                if (dd.tex) [self fbDepthStencilBlitFrom:sd to:dd req:req depth:wd stencil:ws];
            }
        }
    }
}

#pragma mark swap

- (void) fbSwapBuffers
{
    [self endRenderEncoding];
    [self fbEnsureCommandBuffer];

    /* compat domain: IOSurface present (CGL-like async swap, no drawable
     * blocking) when the renderer opted in — MGLRenderer+Compat.m */
    if ([self compatSurfacePresentEnabled])
    {
        id<MTLTexture> color = [self fbEnsureDefaultFramebuffer];
        if (color && [self compatSurfacePresent:color])
            return;
    }

    id<MTLTexture> color = [self fbEnsureDefaultFramebuffer];
    id<CAMetalDrawable> drawable = _layer ? [_layer nextDrawable] : nil;

    if (color && drawable && drawable.texture)
    {
        FBTarget src = { color, 0, 0, 0, 1 };
        FBTarget dst = { drawable.texture, 0, 0, 0, 1 };
        GLint W = (GLint)color.width, H = (GLint)color.height;
        GLint DW = (GLint)drawable.texture.width, DH = (GLint)drawable.texture.height;
        GLint s[4] = { 0, 0, W, H };
        GLint d[4] = { 0, DH, DW, 0 };    /* GL bottom-up rows -> upright on screen */
        [self fbShaderBlitFrom:&src to:&dst src:s dst:d filter:(W == DW && H == DH) ? GL_NEAREST : GL_LINEAR scissor:false];
        [_currentCommandBuffer presentDrawable:drawable];
    }

    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    [self commitCommandBufferWithAGXRecovery:cb];
    _drawable = nil;
    [self newCommandBuffer];
}

#pragma mark renderbuffers

- (bool) fbRenderbufferStorage:(MGLRenderbufferEx *)rb
{
    MTLPixelFormat pf = fbMTLFormatForInternalFormat(rb->internalformat);
    if (pf == MTLPixelFormatInvalid || !_device) return false;

    NSUInteger samples = rb->samples > 1 ? (NSUInteger)rb->samples : 1;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf width:rb->width height:rb->height mipmapped:NO];
    if (samples > 1)
    {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = samples;
    }
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    /* compat s36: the RGB9_E5 emulation quantises the backing IN PLACE with a
     * read_write compute kernel (fbQuantizeRGB9E5), so it needs ShaderWrite too */
    if (rb->internalformat == GL_RGB9_E5)
        d.usage |= MTLTextureUsageShaderWrite;

    /* non-renderable sized formats would be legal renderbuffer storage but could
     * never be attached to a live render pass: keep them sample-only so Metal does
     * not reject the descriptor; framebuffer completeness reports the GL-level failure.
     * (compat s36: GL_RGB9_E5 no longer takes this branch — it is RGBA16F-backed
     * and complete as a colour attachment — so no sized format is left that is
     * accepted by storage yet not renderable; the branch stays as a guard.) */
    const MGLFBFormatInfo *rfi = mgl_fb_format_info(rb->internalformat);
    bool renderable = rfi && (rfi->color_renderable || rfi->depth_renderable || rfi->stencil_renderable);
    if (rb->internalformat == GL_RGB9_E5)
        renderable = true;   /* emulated: RGBA16Float backing, quantised on read-out */
    if (!renderable)
        d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [_device newTextureWithDescriptor:d];
    if (!t) return false;
    t.label = [NSString stringWithFormat:@"MGL renderbuffer %u", rb->base.name];

    if (!renderable)
    {
        /* no render pass may clear a non-renderable texture; such a renderbuffer can
         * never be read back through a complete framebuffer, so initial contents stay
         * undefined rather than zeroed */
        Texture *tex_nr = rb->base.tex;
        if (tex_nr->mtl_data) CFBridgingRelease(tex_nr->mtl_data);
        tex_nr->mtl_data = (void *)CFBridgingRetain(t);
        tex_nr->dirty_bits = 0;
        return true;
    }

    /* defined (zero) initial contents */
    if (_currentRenderEncoder) [self endRenderEncoding];
    [self fbEnsureCommandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    if (fbFormatHasDepth(pf) || fbFormatHasStencil(pf))
    {
        if (fbFormatHasDepth(pf)) { rp.depthAttachment.texture = t; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.clearDepth = 0.0; rp.depthAttachment.storeAction = MTLStoreActionStore; }
        if (fbFormatHasStencil(pf)) { rp.stencilAttachment.texture = t; rp.stencilAttachment.loadAction = MTLLoadActionClear; rp.stencilAttachment.clearStencil = 0; rp.stencilAttachment.storeAction = MTLStoreActionStore; }
    }
    else
    {
        rp.colorAttachments[0].texture = t;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> enc = [_currentCommandBuffer renderCommandEncoderWithDescriptor:rp];
    [enc endEncoding];

    Texture *tex = rb->base.tex;
    if (tex->mtl_data) CFBridgingRelease(tex->mtl_data);
    tex->mtl_data = (void *)CFBridgingRetain(t);
    tex->dirty_bits = 0;
    return true;
}

@end
