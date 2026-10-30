/*
 * texdata_formats.c — texture format table, pixel transfer engine (GL 4.6
 * tables 8.2/8.3/8.8, PixelStore state) and CPU ETC2/EAC decoder.
 * Owned by agent 'texdata'.
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "pixel_utils.h"
#include "texdata_formats.h"

#ifndef GL_ALPHA
#define GL_ALPHA 0x1906
#endif
#ifndef GL_LUMINANCE
#define GL_LUMINANCE 0x1909
#endif
#ifndef GL_LUMINANCE_ALPHA
#define GL_LUMINANCE_ALPHA 0x190A
#endif
#ifndef GL_ALPHA_INTEGER
#define GL_ALPHA_INTEGER 0x8D97
#endif
#ifndef GL_ALPHA8
#define GL_ALPHA8 0x803C
#endif
#ifndef GL_ALPHA16
#define GL_ALPHA16 0x803E
#endif
#ifndef GL_LUMINANCE8
#define GL_LUMINANCE8 0x8040
#endif
#ifndef GL_LUMINANCE16
#define GL_LUMINANCE16 0x8042
#endif
#ifndef GL_LUMINANCE8_ALPHA8
#define GL_LUMINANCE8_ALPHA8 0x8045
#endif
#ifndef GL_LUMINANCE16_ALPHA16
#define GL_LUMINANCE16_ALPHA16 0x8048
#endif
#ifndef GL_COMPRESSED_RGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#endif
#ifndef GL_COMPRESSED_SRGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_SRGB_S3TC_DXT1_EXT 0x8C4C
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT 0x8C4D
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT 0x8C4E
#define GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT 0x8C4F
#endif

/* ------------------------------------------------------------------------ */
/* format table                                                             */
/* ------------------------------------------------------------------------ */

#define UN GL_UNSIGNED_NORMALIZED
#define SN GL_SIGNED_NORMALIZED
#define FL GL_FLOAT
#define SI GL_INT
#define UI GL_UNSIGNED_INT
#define RF (TDF_RENDERABLE | TDF_FILTERABLE)
#define RI (TDF_RENDERABLE | TDF_INTEGER)

/* uncompressed: ifmt, base, ctype, mtl, kind, ncomp, bpp, flags, bits r g b a d s */
#define F(ifmt, base, ct, mtl, kind, n, bpp, fl, r, g, b, a, d, s) \
    { ifmt, base, ct, mtl, kind, n, bpp, fl, 0, 0, 0, { r, g, b, a, d, s }, 0 }
/* compressed: ifmt, base, ctype, mtl, flags, bw, bh, bytes, decoded */
#define C(ifmt, base, ct, mtl, fl, bw, bh, bb, dec) \
    { ifmt, base, ct, mtl, TDK_COMPRESSED, 0, 0, (fl) | TDF_COMPRESSED | TDF_FILTERABLE, bw, bh, bb, { 0, 0, 0, 0, 0, 0 }, dec }

static const TDFormat td_formats[] = {
    /* 8 bit normalized */
    F(GL_R8,            GL_RED,  UN, MTLPixelFormatR8Unorm,       TDK_UN8, 1, 1, RF,  8, 0, 0, 0, 0, 0),
    F(GL_R8_SNORM,      GL_RED,  SN, MTLPixelFormatR8Snorm,       TDK_SN8, 1, 1, RF|TDF_SIGNED,  8, 0, 0, 0, 0, 0),
    F(GL_RG8,           GL_RG,   UN, MTLPixelFormatRG8Unorm,      TDK_UN8, 2, 2, RF,  8, 8, 0, 0, 0, 0),
    F(GL_RG8_SNORM,     GL_RG,   SN, MTLPixelFormatRG8Snorm,      TDK_SN8, 2, 2, RF|TDF_SIGNED,  8, 8, 0, 0, 0, 0),
    F(GL_R3_G3_B2,      GL_RGB,  UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  3, 3, 2, 0, 0, 0),
    F(GL_RGB4,          GL_RGB,  UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  4, 4, 4, 0, 0, 0),
    F(GL_RGB5,          GL_RGB,  UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  5, 5, 5, 0, 0, 0),
    F(GL_RGB565,        GL_RGB,  UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  5, 6, 5, 0, 0, 0),
    F(GL_RGB8,          GL_RGB,  UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  8, 8, 8, 0, 0, 0),
    F(GL_RGB8_SNORM,    GL_RGB,  SN, MTLPixelFormatRGBA8Snorm,    TDK_SN8, 4, 4, RF|TDF_SIGNED,  8, 8, 8, 0, 0, 0),
    F(GL_RGBA2,         GL_RGBA, UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  2, 2, 2, 2, 0, 0),
    F(GL_RGBA4,         GL_RGBA, UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  4, 4, 4, 4, 0, 0),
    F(GL_RGB5_A1,       GL_RGBA, UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  5, 5, 5, 1, 0, 0),
    F(GL_RGBA8,         GL_RGBA, UN, MTLPixelFormatRGBA8Unorm,    TDK_UN8, 4, 4, RF,  8, 8, 8, 8, 0, 0),
    F(GL_RGBA8_SNORM,   GL_RGBA, SN, MTLPixelFormatRGBA8Snorm,    TDK_SN8, 4, 4, RF|TDF_SIGNED,  8, 8, 8, 8, 0, 0),
    F(GL_SRGB8,         GL_RGB,  UN, MTLPixelFormatRGBA8Unorm_sRGB, TDK_UN8, 4, 4, RF|TDF_SRGB, 8, 8, 8, 0, 0, 0),
    F(GL_SRGB8_ALPHA8,  GL_RGBA, UN, MTLPixelFormatRGBA8Unorm_sRGB, TDK_UN8, 4, 4, RF|TDF_SRGB, 8, 8, 8, 8, 0, 0),
    /* 10/12/16 bit normalized */
    F(GL_RGB10,         GL_RGB,  UN, MTLPixelFormatRGB10A2Unorm,  TDK_RGB10A2UN, 4, 4, RF, 10, 10, 10, 0, 0, 0),
    F(GL_RGB10_A2,      GL_RGBA, UN, MTLPixelFormatRGB10A2Unorm,  TDK_RGB10A2UN, 4, 4, RF, 10, 10, 10, 2, 0, 0),
    F(GL_RGB10_A2UI,    GL_RGBA, UI, MTLPixelFormatRGB10A2Uint,   TDK_RGB10A2UI, 4, 4, RI, 10, 10, 10, 2, 0, 0),
    F(GL_R16,           GL_RED,  UN, MTLPixelFormatR16Unorm,      TDK_UN16, 1, 2, RF, 16, 0, 0, 0, 0, 0),
    F(GL_R16_SNORM,     GL_RED,  SN, MTLPixelFormatR16Snorm,      TDK_SN16, 1, 2, RF|TDF_SIGNED, 16, 0, 0, 0, 0, 0),
    F(GL_RG16,          GL_RG,   UN, MTLPixelFormatRG16Unorm,     TDK_UN16, 2, 4, RF, 16, 16, 0, 0, 0, 0),
    F(GL_RG16_SNORM,    GL_RG,   SN, MTLPixelFormatRG16Snorm,     TDK_SN16, 2, 4, RF|TDF_SIGNED, 16, 16, 0, 0, 0, 0),
    F(GL_RGB12,         GL_RGB,  UN, MTLPixelFormatRGBA16Unorm,   TDK_UN16, 4, 8, RF, 12, 12, 12, 0, 0, 0),
    F(GL_RGB16,         GL_RGB,  UN, MTLPixelFormatRGBA16Unorm,   TDK_UN16, 4, 8, RF, 16, 16, 16, 0, 0, 0),
    F(GL_RGB16_SNORM,   GL_RGB,  SN, MTLPixelFormatRGBA16Snorm,   TDK_SN16, 4, 8, RF|TDF_SIGNED, 16, 16, 16, 0, 0, 0),
    F(GL_RGBA12,        GL_RGBA, UN, MTLPixelFormatRGBA16Unorm,   TDK_UN16, 4, 8, RF, 12, 12, 12, 12, 0, 0),
    F(GL_RGBA16,        GL_RGBA, UN, MTLPixelFormatRGBA16Unorm,   TDK_UN16, 4, 8, RF, 16, 16, 16, 16, 0, 0),
    F(GL_RGBA16_SNORM,  GL_RGBA, SN, MTLPixelFormatRGBA16Snorm,   TDK_SN16, 4, 8, RF|TDF_SIGNED, 16, 16, 16, 16, 0, 0),
    /* float */
    F(GL_R16F,          GL_RED,  FL, MTLPixelFormatR16Float,      TDK_F16, 1, 2, RF, 16, 0, 0, 0, 0, 0),
    F(GL_RG16F,         GL_RG,   FL, MTLPixelFormatRG16Float,     TDK_F16, 2, 4, RF, 16, 16, 0, 0, 0, 0),
    F(GL_RGB16F,        GL_RGB,  FL, MTLPixelFormatRGBA16Float,   TDK_F16, 4, 8, RF, 16, 16, 16, 0, 0, 0),
    F(GL_RGBA16F,       GL_RGBA, FL, MTLPixelFormatRGBA16Float,   TDK_F16, 4, 8, RF, 16, 16, 16, 16, 0, 0),
    F(GL_R32F,          GL_RED,  FL, MTLPixelFormatR32Float,      TDK_F32, 1, 4, RF, 32, 0, 0, 0, 0, 0),
    F(GL_RG32F,         GL_RG,   FL, MTLPixelFormatRG32Float,     TDK_F32, 2, 8, RF, 32, 32, 0, 0, 0, 0),
    F(GL_RGB32F,        GL_RGB,  FL, MTLPixelFormatRGBA32Float,   TDK_F32, 4, 16, RF, 32, 32, 32, 0, 0, 0),
    F(GL_RGBA32F,       GL_RGBA, FL, MTLPixelFormatRGBA32Float,   TDK_F32, 4, 16, RF, 32, 32, 32, 32, 0, 0),
    F(GL_R11F_G11F_B10F, GL_RGB, FL, MTLPixelFormatRG11B10Float,  TDK_RG11B10F, 3, 4, RF, 11, 11, 10, 0, 0, 0),
    F(GL_RGB9_E5,       GL_RGB,  FL, MTLPixelFormatRGB9E5Float,   TDK_RGB9E5, 3, 4, TDF_FILTERABLE, 9, 9, 9, 0, 0, 0),
    /* integer */
    F(GL_R8I,           GL_RED,  SI, MTLPixelFormatR8Sint,        TDK_SI8, 1, 1, RI, 8, 0, 0, 0, 0, 0),
    F(GL_R8UI,          GL_RED,  UI, MTLPixelFormatR8Uint,        TDK_UI8, 1, 1, RI, 8, 0, 0, 0, 0, 0),
    F(GL_R16I,          GL_RED,  SI, MTLPixelFormatR16Sint,       TDK_SI16, 1, 2, RI, 16, 0, 0, 0, 0, 0),
    F(GL_R16UI,         GL_RED,  UI, MTLPixelFormatR16Uint,       TDK_UI16, 1, 2, RI, 16, 0, 0, 0, 0, 0),
    F(GL_R32I,          GL_RED,  SI, MTLPixelFormatR32Sint,       TDK_SI32, 1, 4, RI, 32, 0, 0, 0, 0, 0),
    F(GL_R32UI,         GL_RED,  UI, MTLPixelFormatR32Uint,       TDK_UI32, 1, 4, RI, 32, 0, 0, 0, 0, 0),
    F(GL_RG8I,          GL_RG,   SI, MTLPixelFormatRG8Sint,       TDK_SI8, 2, 2, RI, 8, 8, 0, 0, 0, 0),
    F(GL_RG8UI,         GL_RG,   UI, MTLPixelFormatRG8Uint,       TDK_UI8, 2, 2, RI, 8, 8, 0, 0, 0, 0),
    F(GL_RG16I,         GL_RG,   SI, MTLPixelFormatRG16Sint,      TDK_SI16, 2, 4, RI, 16, 16, 0, 0, 0, 0),
    F(GL_RG16UI,        GL_RG,   UI, MTLPixelFormatRG16Uint,      TDK_UI16, 2, 4, RI, 16, 16, 0, 0, 0, 0),
    F(GL_RG32I,         GL_RG,   SI, MTLPixelFormatRG32Sint,      TDK_SI32, 2, 8, RI, 32, 32, 0, 0, 0, 0),
    F(GL_RG32UI,        GL_RG,   UI, MTLPixelFormatRG32Uint,      TDK_UI32, 2, 8, RI, 32, 32, 0, 0, 0, 0),
    F(GL_RGB8I,         GL_RGB,  SI, MTLPixelFormatRGBA8Sint,     TDK_SI8, 4, 4, RI, 8, 8, 8, 0, 0, 0),
    F(GL_RGB8UI,        GL_RGB,  UI, MTLPixelFormatRGBA8Uint,     TDK_UI8, 4, 4, RI, 8, 8, 8, 0, 0, 0),
    F(GL_RGB16I,        GL_RGB,  SI, MTLPixelFormatRGBA16Sint,    TDK_SI16, 4, 8, RI, 16, 16, 16, 0, 0, 0),
    F(GL_RGB16UI,       GL_RGB,  UI, MTLPixelFormatRGBA16Uint,    TDK_UI16, 4, 8, RI, 16, 16, 16, 0, 0, 0),
    F(GL_RGB32I,        GL_RGB,  SI, MTLPixelFormatRGBA32Sint,    TDK_SI32, 4, 16, RI, 32, 32, 32, 0, 0, 0),
    F(GL_RGB32UI,       GL_RGB,  UI, MTLPixelFormatRGBA32Uint,    TDK_UI32, 4, 16, RI, 32, 32, 32, 0, 0, 0),
    F(GL_RGBA8I,        GL_RGBA, SI, MTLPixelFormatRGBA8Sint,     TDK_SI8, 4, 4, RI, 8, 8, 8, 8, 0, 0),
    F(GL_RGBA8UI,       GL_RGBA, UI, MTLPixelFormatRGBA8Uint,     TDK_UI8, 4, 4, RI, 8, 8, 8, 8, 0, 0),
    F(GL_RGBA16I,       GL_RGBA, SI, MTLPixelFormatRGBA16Sint,    TDK_SI16, 4, 8, RI, 16, 16, 16, 16, 0, 0),
    F(GL_RGBA16UI,      GL_RGBA, UI, MTLPixelFormatRGBA16Uint,    TDK_UI16, 4, 8, RI, 16, 16, 16, 16, 0, 0),
    F(GL_RGBA32I,       GL_RGBA, SI, MTLPixelFormatRGBA32Sint,    TDK_SI32, 4, 16, RI, 32, 32, 32, 32, 0, 0),
    F(GL_RGBA32UI,      GL_RGBA, UI, MTLPixelFormatRGBA32Uint,    TDK_UI32, 4, 16, RI, 32, 32, 32, 32, 0, 0),
    /* depth / stencil (Apple GPUs have no D24S8: DEPTH24_STENCIL8 lives in D32F_S8) */
    /* DEPTH_COMPONENT16 is also kept in D32F storage, quantized to the 16-bit unorm
     * lattice on upload: Metal's hardware sample_compare against a Depth16Unorm
     * texel compares the reference in the 16-bit fixed-point domain (or scales the
     * texel by 1/65536), so ref == texel_value loses ("less" is false) where GL
     * compares in float against texel*65535/65536 (KHR-GL46.ext_texture_shadow_lod
     * sampler2darrayshadow_vertex hits exactly this boundary: ref 0.5 vs texel
     * 0x8000).  Depth32Float compares in float, so D16-in-D32F matches GL.  Same
     * trick DEPTH_COMPONENT24 already uses below. */
    F(GL_DEPTH_COMPONENT16,  GL_DEPTH_COMPONENT, UN, MTLPixelFormatDepth32Float, TDK_D16F, 1, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH, 0, 0, 0, 0, 16, 0),
    F(GL_DEPTH_COMPONENT24,  GL_DEPTH_COMPONENT, UN, MTLPixelFormatDepth32Float, TDK_D32F, 1, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH, 0, 0, 0, 0, 24, 0),
    F(GL_DEPTH_COMPONENT32,  GL_DEPTH_COMPONENT, UN, MTLPixelFormatDepth32Float, TDK_D32F, 1, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH, 0, 0, 0, 0, 32, 0),
    F(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, FL, MTLPixelFormatDepth32Float, TDK_D32F, 1, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH, 0, 0, 0, 0, 32, 0),
    F(GL_DEPTH24_STENCIL8,   GL_DEPTH_STENCIL,   UN, MTLPixelFormatDepth32Float_Stencil8, TDK_D32FS8, 2, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH|TDF_STENCIL, 0, 0, 0, 0, 24, 8),
    F(GL_DEPTH32F_STENCIL8,  GL_DEPTH_STENCIL,   FL, MTLPixelFormatDepth32Float_Stencil8, TDK_D32FS8, 2, 4, TDF_RENDERABLE|TDF_FILTERABLE|TDF_DEPTH|TDF_STENCIL, 0, 0, 0, 0, 32, 8),
    F(GL_STENCIL_INDEX1,     GL_STENCIL_INDEX,   UI, MTLPixelFormatStencil8, TDK_S8, 1, 1, TDF_RENDERABLE|TDF_STENCIL, 0, 0, 0, 0, 0, 1),
    F(GL_STENCIL_INDEX4,     GL_STENCIL_INDEX,   UI, MTLPixelFormatStencil8, TDK_S8, 1, 1, TDF_RENDERABLE|TDF_STENCIL, 0, 0, 0, 0, 0, 4),
    F(GL_STENCIL_INDEX8,     GL_STENCIL_INDEX,   UI, MTLPixelFormatStencil8, TDK_S8, 1, 1, TDF_RENDERABLE|TDF_STENCIL, 0, 0, 0, 0, 0, 8),
    F(GL_STENCIL_INDEX16,    GL_STENCIL_INDEX,   UI, MTLPixelFormatStencil8, TDK_S8, 1, 1, TDF_RENDERABLE|TDF_STENCIL, 0, 0, 0, 0, 0, 8),
    /* legacy alpha / luminance (compatibility, used by virglrenderer guests) */
    F(GL_ALPHA8,             GL_ALPHA,           UN, MTLPixelFormatR8Unorm,  TDK_UN8, 1, 1, RF|TDF_LEGACY, 0, 0, 0, 8, 0, 0),
    F(GL_ALPHA16,            GL_ALPHA,           UN, MTLPixelFormatR16Unorm, TDK_UN16, 1, 2, RF|TDF_LEGACY, 0, 0, 0, 16, 0, 0),
    F(GL_LUMINANCE8,         GL_LUMINANCE,       UN, MTLPixelFormatR8Unorm,  TDK_UN8, 1, 1, RF|TDF_LEGACY, 8, 0, 0, 0, 0, 0),
    F(GL_LUMINANCE16,        GL_LUMINANCE,       UN, MTLPixelFormatR16Unorm, TDK_UN16, 1, 2, RF|TDF_LEGACY, 16, 0, 0, 0, 0, 0),
    F(GL_LUMINANCE8_ALPHA8,  GL_LUMINANCE_ALPHA, UN, MTLPixelFormatRG8Unorm, TDK_UN8, 2, 2, RF|TDF_LEGACY, 8, 0, 0, 8, 0, 0),
    F(GL_LUMINANCE16_ALPHA16, GL_LUMINANCE_ALPHA, UN, MTLPixelFormatRG16Unorm, TDK_UN16, 2, 4, RF|TDF_LEGACY, 16, 0, 0, 16, 0, 0),

    /* RGTC */
    C(GL_COMPRESSED_RED_RGTC1,        GL_RED,  UN, MTLPixelFormatBC4_RUnorm,  TDF_BC, 4, 4, 8, GL_R8),
    C(GL_COMPRESSED_SIGNED_RED_RGTC1, GL_RED,  SN, MTLPixelFormatBC4_RSnorm,  TDF_BC|TDF_SIGNED, 4, 4, 8, GL_R8_SNORM),
    C(GL_COMPRESSED_RG_RGTC2,         GL_RG,   UN, MTLPixelFormatBC5_RGUnorm, TDF_BC, 4, 4, 16, GL_RG8),
    C(GL_COMPRESSED_SIGNED_RG_RGTC2,  GL_RG,   SN, MTLPixelFormatBC5_RGSnorm, TDF_BC|TDF_SIGNED, 4, 4, 16, GL_RG8_SNORM),
    /* BPTC */
    C(GL_COMPRESSED_RGBA_BPTC_UNORM,         GL_RGBA, UN, MTLPixelFormatBC7_RGBAUnorm,      TDF_BC, 4, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,   GL_RGBA, UN, MTLPixelFormatBC7_RGBAUnorm_sRGB, TDF_BC|TDF_SRGB, 4, 4, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT,   GL_RGB,  FL, MTLPixelFormatBC6H_RGBFloat,      TDF_BC|TDF_SIGNED|TDF_BPTC_FLOAT, 4, 4, 16, GL_RGB16F),
    C(GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, GL_RGB,  FL, MTLPixelFormatBC6H_RGBUfloat,     TDF_BC|TDF_BPTC_FLOAT, 4, 4, 16, GL_RGB16F),
    /* S3TC (EXT_texture_compression_s3tc / EXT_texture_sRGB) */
    C(GL_COMPRESSED_RGB_S3TC_DXT1_EXT,        GL_RGB,  UN, MTLPixelFormatBC1_RGBA,      TDF_BC, 4, 4, 8, GL_RGB8),
    C(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,       GL_RGBA, UN, MTLPixelFormatBC1_RGBA,      TDF_BC, 4, 4, 8, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,       GL_RGBA, UN, MTLPixelFormatBC2_RGBA,      TDF_BC, 4, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,       GL_RGBA, UN, MTLPixelFormatBC3_RGBA,      TDF_BC, 4, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,       GL_RGB,  UN, MTLPixelFormatBC1_RGBA_sRGB, TDF_BC|TDF_SRGB, 4, 4, 8, GL_SRGB8),
    C(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT, GL_RGBA, UN, MTLPixelFormatBC1_RGBA_sRGB, TDF_BC|TDF_SRGB, 4, 4, 8, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT, GL_RGBA, UN, MTLPixelFormatBC2_RGBA_sRGB, TDF_BC|TDF_SRGB, 4, 4, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT, GL_RGBA, UN, MTLPixelFormatBC3_RGBA_sRGB, TDF_BC|TDF_SRGB, 4, 4, 16, GL_SRGB8_ALPHA8),
    /* ETC2 / EAC (GL 4.3 core) */
    C(GL_COMPRESSED_RGB8_ETC2,                      GL_RGB,  UN, MTLPixelFormatETC2_RGB8,        TDF_ETC, 4, 4, 8,  GL_RGBA8),
    C(GL_COMPRESSED_SRGB8_ETC2,                     GL_RGB,  UN, MTLPixelFormatETC2_RGB8_sRGB,   TDF_ETC|TDF_SRGB, 4, 4, 8, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,  GL_RGBA, UN, MTLPixelFormatETC2_RGB8A1,      TDF_ETC, 4, 4, 8,  GL_RGBA8),
    C(GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2, GL_RGBA, UN, MTLPixelFormatETC2_RGB8A1_sRGB, TDF_ETC|TDF_SRGB, 4, 4, 8, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_RGBA8_ETC2_EAC,                 GL_RGBA, UN, MTLPixelFormatEAC_RGBA8,        TDF_ETC, 4, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,          GL_RGBA, UN, MTLPixelFormatEAC_RGBA8_sRGB,   TDF_ETC|TDF_SRGB, 4, 4, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_R11_EAC,                        GL_RED,  UN, MTLPixelFormatEAC_R11Unorm,     TDF_ETC, 4, 4, 8,  GL_R16),
    C(GL_COMPRESSED_SIGNED_R11_EAC,                 GL_RED,  SN, MTLPixelFormatEAC_R11Snorm,     TDF_ETC|TDF_SIGNED, 4, 4, 8, GL_R16_SNORM),
    C(GL_COMPRESSED_RG11_EAC,                       GL_RG,   UN, MTLPixelFormatEAC_RG11Unorm,    TDF_ETC, 4, 4, 16, GL_RG16),
    C(GL_COMPRESSED_SIGNED_RG11_EAC,                GL_RG,   SN, MTLPixelFormatEAC_RG11Snorm,    TDF_ETC|TDF_SIGNED, 4, 4, 16, GL_RG16_SNORM),
    /* ASTC LDR (KHR_texture_compression_astc_ldr): every block is 16 bytes.
     * No CPU decoder — only usable when the device has ASTC (tdDeviceHasASTC). */
    C(GL_COMPRESSED_RGBA_ASTC_4x4_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_4x4_LDR,   TDF_ASTC, 4, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_5x4_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_5x4_LDR,   TDF_ASTC, 5, 4, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_5x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_5x5_LDR,   TDF_ASTC, 5, 5, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_6x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_6x5_LDR,   TDF_ASTC, 6, 5, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_6x6_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_6x6_LDR,   TDF_ASTC, 6, 6, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_8x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x5_LDR,   TDF_ASTC, 8, 5, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_8x6_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x6_LDR,   TDF_ASTC, 8, 6, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_8x8_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x8_LDR,   TDF_ASTC, 8, 8, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_10x5_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x5_LDR,  TDF_ASTC, 10, 5, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_10x6_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x6_LDR,  TDF_ASTC, 10, 6, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_10x8_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x8_LDR,  TDF_ASTC, 10, 8, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_10x10_KHR, GL_RGBA, UN, MTLPixelFormatASTC_10x10_LDR, TDF_ASTC, 10, 10, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_12x10_KHR, GL_RGBA, UN, MTLPixelFormatASTC_12x10_LDR, TDF_ASTC, 12, 10, 16, GL_RGBA8),
    C(GL_COMPRESSED_RGBA_ASTC_12x12_KHR, GL_RGBA, UN, MTLPixelFormatASTC_12x12_LDR, TDF_ASTC, 12, 12, 16, GL_RGBA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_4x4_sRGB,   TDF_ASTC|TDF_SRGB, 4, 4, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_5x4_sRGB,   TDF_ASTC|TDF_SRGB, 5, 4, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_5x5_sRGB,   TDF_ASTC|TDF_SRGB, 5, 5, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_6x5_sRGB,   TDF_ASTC|TDF_SRGB, 6, 5, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_6x6_sRGB,   TDF_ASTC|TDF_SRGB, 6, 6, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x5_sRGB,   TDF_ASTC|TDF_SRGB, 8, 5, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x6_sRGB,   TDF_ASTC|TDF_SRGB, 8, 6, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,   GL_RGBA, UN, MTLPixelFormatASTC_8x8_sRGB,   TDF_ASTC|TDF_SRGB, 8, 8, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x5_sRGB,  TDF_ASTC|TDF_SRGB, 10, 5, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x6_sRGB,  TDF_ASTC|TDF_SRGB, 10, 6, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,  GL_RGBA, UN, MTLPixelFormatASTC_10x8_sRGB,  TDF_ASTC|TDF_SRGB, 10, 8, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR, GL_RGBA, UN, MTLPixelFormatASTC_10x10_sRGB, TDF_ASTC|TDF_SRGB, 10, 10, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR, GL_RGBA, UN, MTLPixelFormatASTC_12x10_sRGB, TDF_ASTC|TDF_SRGB, 12, 10, 16, GL_SRGB8_ALPHA8),
    C(GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR, GL_RGBA, UN, MTLPixelFormatASTC_12x12_sRGB, TDF_ASTC|TDF_SRGB, 12, 12, 16, GL_SRGB8_ALPHA8),
};

#define TD_NUM_FORMATS (sizeof(td_formats) / sizeof(td_formats[0]))

const TDFormat *tdFormatInfo(GLenum internalformat)
{
    for (size_t i = 0; i < TD_NUM_FORMATS; i++)
    {
        if (td_formats[i].internalformat == internalformat)
        {
            /* ASTC is device-only: without hardware ASTC the format is not a legal
             * internal format at all (as if the extension were absent) */
            if ((td_formats[i].flags & TDF_ASTC) && !tdDeviceHasASTC())
                return NULL;
            return &td_formats[i];
        }
    }
    return NULL;
}

static bool td_cap_etc = true, td_cap_bc = true, td_cap_astc = false;

void tdSetDeviceCaps(bool etc, bool bc) { td_cap_etc = etc; td_cap_bc = bc; }
void tdSetASTCCap(bool astc) { td_cap_astc = astc; }

bool tdDeviceHasETC(void)
{
    const char *e = getenv("MGL_FORCE_ETC_DECODE");
    if (e && e[0] == '1') return false;
    return td_cap_etc;
}

bool tdDeviceHasBC(void) { return td_cap_bc; }

bool tdDeviceHasASTC(void)
{
    const char *e = getenv("MGL_FORCE_ETC_DECODE");
    if (e && e[0] == '1') return false;
    return td_cap_astc;
}

const TDFormat *tdStorageFormat(const TDFormat *f)
{
    if (f && (f->flags & TDF_ETC) && !tdDeviceHasETC())
        return tdFormatInfo(f->decoded);
    return f;
}

GLenum tdResolveInternalFormat(GLenum internalformat, GLenum format, GLenum type)
{
    bool fl = (type == GL_FLOAT);
    bool hf = (type == GL_HALF_FLOAT);

    switch (internalformat)
    {
        case GL_RED:  return fl ? GL_R32F : hf ? GL_R16F : GL_R8;
        case GL_RG:   return fl ? GL_RG32F : hf ? GL_RG16F : GL_RG8;
        case GL_RGB:  return fl ? GL_RGB32F : hf ? GL_RGB16F : GL_RGB8;
        case GL_RGBA: return fl ? GL_RGBA32F : hf ? GL_RGBA16F : GL_RGBA8;
        case GL_SRGB: return GL_SRGB8;
        case GL_SRGB_ALPHA: return GL_SRGB8_ALPHA8;
        case GL_DEPTH_COMPONENT:
            return (fl || hf) ? GL_DEPTH_COMPONENT32F : (type == GL_UNSIGNED_SHORT ? GL_DEPTH_COMPONENT16 : GL_DEPTH_COMPONENT24);
        case GL_DEPTH_STENCIL:
            return type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV ? GL_DEPTH32F_STENCIL8 : GL_DEPTH24_STENCIL8;
        case GL_STENCIL_INDEX: return GL_STENCIL_INDEX8;
        /* generic compressed formats: the GL may choose any representation,
         * we store them uncompressed (GL 4.6 8.5.3 allows this) */
        case GL_COMPRESSED_RED:        return GL_R8;
        case GL_COMPRESSED_RG:         return GL_RG8;
        case GL_COMPRESSED_RGB:        return GL_RGB8;
        case GL_COMPRESSED_RGBA:       return GL_RGBA8;
        case GL_COMPRESSED_SRGB:       return GL_SRGB8;
        case GL_COMPRESSED_SRGB_ALPHA: return GL_SRGB8_ALPHA8;
        /* legacy unsized */
        case GL_ALPHA:           return GL_ALPHA8;
        case GL_LUMINANCE:       return GL_LUMINANCE8;
        case GL_LUMINANCE_ALPHA: return GL_LUMINANCE8_ALPHA8;
        default: break;
    }

    /* a SPECIFIC compressed format given ordinary client data means "the GL will
     * compress this" (GL 4.6 8.5.4); storing the decoded equivalent is a legal
     * representation (hardware compression is optional) — this is what the CTS
     * packed_pixels rectangle.compressed_* cases drive.  Compressed upload paths
     * pass format == 0 and keep the real compressed format. */
    if (format && format != GL_NONE)
    {
        const TDFormat *cf = tdFormatInfo(internalformat);
        if (cf && (cf->flags & TDF_COMPRESSED) && cf->decoded)
            return cf->decoded;
    }

    (void)format;
    return tdFormatInfo(internalformat) ? internalformat : 0;
}

size_t tdCompressedSize(const TDFormat *f, int w, int h, int d)
{
    if (!f || !(f->flags & TDF_COMPRESSED)) return 0;
    size_t bx = (size_t)(w + f->block_w - 1) / f->block_w;
    size_t by = (size_t)(h + f->block_h - 1) / f->block_h;
    return bx * by * (size_t)(d > 0 ? d : 1) * f->block_bytes;
}

size_t tdStorageRowBytes(const TDFormat *f, int w)
{
    if (f->flags & TDF_COMPRESSED)
        return (size_t)((w + f->block_w - 1) / f->block_w) * f->block_bytes;
    return (size_t)w * f->bpp;
}

size_t tdStorageImageBytes(const TDFormat *f, int w, int h)
{
    if (f->flags & TDF_COMPRESSED)
        return tdStorageRowBytes(f, w) * (size_t)((h + f->block_h - 1) / f->block_h);
    return tdStorageRowBytes(f, w) * (size_t)h;
}

/* ------------------------------------------------------------------------ */
/* half / small float helpers                                               */
/* ------------------------------------------------------------------------ */

float tdHalfToFloat(uint16_t h)
{
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}

uint16_t tdFloatToHalf(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int e = (int)((x >> 23) & 0xff);
    uint32_t m = x & 0x7fffff;

    if (e == 0xff) return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
    int ne = e - 127 + 15;
    if (ne >= 31) return (uint16_t)(s | 0x7c00);
    if (ne <= 0)
    {
        if (ne < -10) return (uint16_t)s;
        m |= 0x800000;
        int shift = 14 - ne;
        uint32_t hm = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return (uint16_t)(s | hm);
    }
    uint32_t hm = m >> 13, rem = m & 0x1fff;
    uint32_t r = (uint32_t)(s | ((uint32_t)ne << 10) | hm);
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) r++;
    return (uint16_t)r;
}

static uint32_t td_float_to_uf(float f, int mbits)
{
    /* unsigned float with 5 bit exponent and mbits mantissa (6 -> uf11, 5 -> uf10) */
    if (isnan(f)) return (31u << mbits) | 1u;
    if (f <= 0.0f) return 0;
    if (isinf(f)) return 31u << mbits;
    uint16_t h = tdFloatToHalf(f);
    uint32_t e = (h >> 10) & 0x1f;
    if (e == 31) return (30u << mbits) | ((1u << mbits) - 1);   /* clamp to max finite */
    int drop = 10 - mbits;
    uint32_t v = (uint32_t)(h & 0x7fff);
    uint32_t r = (v + (1u << (drop - 1))) >> drop;
    if ((r >> mbits) >= 31) r = (30u << mbits) | ((1u << mbits) - 1);
    return r;
}

static float td_uf_to_float(uint32_t v, int mbits)
{
    uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1);
    if (e == 0) return ldexpf((float)m, -14 - mbits);
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf((float)(m | (1u << mbits)), (int)e - 15 - mbits);
}

static uint32_t td_encode_rgb9e5(double r, double g, double b)
{
    const int N = 9, B = 15;
    const double maxv = (double)((1 << N) - 1) / (1 << N) * ldexp(1.0, 31 - B);
    double rc = isnan(r) ? 0 : fmin(fmax(r, 0.0), maxv);
    double gc = isnan(g) ? 0 : fmin(fmax(g, 0.0), maxv);
    double bc = isnan(b) ? 0 : fmin(fmax(b, 0.0), maxv);
    double maxc = fmax(rc, fmax(gc, bc));
    if (maxc <= 0.0) return 0;
    int expp = (int)fmax(-B - 1, floor(log2(maxc))) + 1 + B;
    double maxs = floor(maxc / ldexp(1.0, expp - B - N) + 0.5);
    int exps = (maxs == (1 << N)) ? expp + 1 : expp;
    double sc = ldexp(1.0, exps - B - N);
    uint32_t rs = (uint32_t)floor(rc / sc + 0.5), gs = (uint32_t)floor(gc / sc + 0.5), bs = (uint32_t)floor(bc / sc + 0.5);
    if (rs > 511) rs = 511;
    if (gs > 511) gs = 511;
    if (bs > 511) bs = 511;
    return rs | (gs << 9) | (bs << 18) | ((uint32_t)exps << 27);
}

static void td_decode_rgb9e5(uint32_t v, double *out)
{
    int e = (int)(v >> 27);
    double sc = ldexp(1.0, e - 15 - 9);
    out[0] = (double)(v & 511) * sc;
    out[1] = (double)((v >> 9) & 511) * sc;
    out[2] = (double)((v >> 18) & 511) * sc;
}

/* ------------------------------------------------------------------------ */
/* client format / type description                                         */
/* ------------------------------------------------------------------------ */

enum { CR = 0, CG = 1, CB = 2, CA = 3, CD = 4, CS = 5 };

typedef struct {
    int n;
    int comp[4];
    bool integer;
} TDClientFormat;

static bool td_client_format(GLenum format, TDClientFormat *cf)
{
    cf->integer = false;
    switch (format)
    {
        case GL_RED_INTEGER:   cf->integer = true; /* fallthrough */
        case GL_RED:           cf->n = 1; cf->comp[0] = CR; return true;
        case GL_GREEN_INTEGER: cf->integer = true; /* fallthrough */
        case GL_GREEN:         cf->n = 1; cf->comp[0] = CG; return true;
        case GL_BLUE_INTEGER:  cf->integer = true; /* fallthrough */
        case GL_BLUE:          cf->n = 1; cf->comp[0] = CB; return true;
        case GL_ALPHA_INTEGER: cf->integer = true; /* fallthrough */
        case GL_ALPHA:         cf->n = 1; cf->comp[0] = CA; return true;
        case GL_LUMINANCE:     cf->n = 1; cf->comp[0] = CR; return true;
        case GL_LUMINANCE_ALPHA: cf->n = 2; cf->comp[0] = CR; cf->comp[1] = CA; return true;
        case GL_RG_INTEGER:    cf->integer = true; /* fallthrough */
        case GL_RG:            cf->n = 2; cf->comp[0] = CR; cf->comp[1] = CG; return true;
        case GL_RGB_INTEGER:   cf->integer = true; /* fallthrough */
        case GL_RGB:           cf->n = 3; cf->comp[0] = CR; cf->comp[1] = CG; cf->comp[2] = CB; return true;
        case GL_BGR_INTEGER:   cf->integer = true; /* fallthrough */
        case GL_BGR:           cf->n = 3; cf->comp[0] = CB; cf->comp[1] = CG; cf->comp[2] = CR; return true;
        case GL_RGBA_INTEGER:  cf->integer = true; /* fallthrough */
        case GL_RGBA:          cf->n = 4; cf->comp[0] = CR; cf->comp[1] = CG; cf->comp[2] = CB; cf->comp[3] = CA; return true;
        case GL_BGRA_INTEGER:  cf->integer = true; /* fallthrough */
        case GL_BGRA:          cf->n = 4; cf->comp[0] = CB; cf->comp[1] = CG; cf->comp[2] = CR; cf->comp[3] = CA; return true;
        case GL_DEPTH_COMPONENT: cf->n = 1; cf->comp[0] = CD; return true;
        case GL_STENCIL_INDEX:   cf->n = 1; cf->comp[0] = CS; cf->integer = true; return true;
        case GL_DEPTH_STENCIL:   cf->n = 2; cf->comp[0] = CD; cf->comp[1] = CS; return true;
        default: return false;
    }
}

/* packed type description: total bits, number of fields, widths, lsb-first */
typedef struct { int bytes; int n; int w[4]; bool rev; } TDPacked;

static bool td_packed(GLenum type, TDPacked *p)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE_3_3_2:         *p = (TDPacked){1, 3, {3, 3, 2, 0}, false}; return true;
        case GL_UNSIGNED_BYTE_2_3_3_REV:     *p = (TDPacked){1, 3, {3, 3, 2, 0}, true}; return true;
        case GL_UNSIGNED_SHORT_5_6_5:        *p = (TDPacked){2, 3, {5, 6, 5, 0}, false}; return true;
        case GL_UNSIGNED_SHORT_5_6_5_REV:    *p = (TDPacked){2, 3, {5, 6, 5, 0}, true}; return true;
        case GL_UNSIGNED_SHORT_4_4_4_4:      *p = (TDPacked){2, 4, {4, 4, 4, 4}, false}; return true;
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:  *p = (TDPacked){2, 4, {4, 4, 4, 4}, true}; return true;
        case GL_UNSIGNED_SHORT_5_5_5_1:      *p = (TDPacked){2, 4, {5, 5, 5, 1}, false}; return true;
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:  *p = (TDPacked){2, 4, {5, 5, 5, 1}, true}; return true;
        case GL_UNSIGNED_INT_8_8_8_8:        *p = (TDPacked){4, 4, {8, 8, 8, 8}, false}; return true;
        case GL_UNSIGNED_INT_8_8_8_8_REV:    *p = (TDPacked){4, 4, {8, 8, 8, 8}, true}; return true;
        case GL_UNSIGNED_INT_10_10_10_2:     *p = (TDPacked){4, 4, {10, 10, 10, 2}, false}; return true;
        case GL_UNSIGNED_INT_2_10_10_10_REV: *p = (TDPacked){4, 4, {10, 10, 10, 2}, true}; return true;
        default: return false;
    }
}

static int td_type_size(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2;
        case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4;
        default: return 0;
    }
}

GLenum tdValidateFormatType(GLenum format, GLenum type)
{
    TDClientFormat cf;
    TDPacked pk;

    if (!td_client_format(format, &cf)) return GL_INVALID_ENUM;

    switch (type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE: case GL_UNSIGNED_SHORT: case GL_SHORT:
        case GL_UNSIGNED_INT: case GL_INT:
            if (format == GL_DEPTH_STENCIL) return GL_INVALID_OPERATION;
            return GL_NO_ERROR;
        case GL_HALF_FLOAT: case GL_FLOAT:
            if (format == GL_DEPTH_STENCIL) return GL_INVALID_OPERATION;
            if (cf.integer && format != GL_STENCIL_INDEX) return GL_INVALID_OPERATION;
            return GL_NO_ERROR;
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            /* table 8.3: the float-packed types carry RGB only */
            return format == GL_RGB ? GL_NO_ERROR : GL_INVALID_OPERATION;
        case GL_UNSIGNED_INT_24_8:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return format == GL_DEPTH_STENCIL ? GL_NO_ERROR : GL_INVALID_OPERATION;
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            /* table 8.3: RGBA / BGRA and their integer twins */
            return (format == GL_RGBA || format == GL_BGRA || format == GL_RGBA_INTEGER ||
                    format == GL_BGRA_INTEGER) ? GL_NO_ERROR : GL_INVALID_OPERATION;
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            /* table 8.3: RGBA / BGRA and their integer twins */
            return (format == GL_RGBA || format == GL_BGRA || format == GL_RGBA_INTEGER ||
                    format == GL_BGRA_INTEGER) ? GL_NO_ERROR : GL_INVALID_OPERATION;
        default:
            if (!td_packed(type, &pk)) return GL_INVALID_ENUM;
            if (pk.n == 3)
                return (format == GL_RGB || format == GL_RGB_INTEGER) ? GL_NO_ERROR : GL_INVALID_OPERATION;
            /* table 8.3: 4_4_4_4(_REV), 5_5_5_1, 1_5_5_5_REV, 10_10_10_2 carry the
             * 4-component formats only */
            return (format == GL_RGBA || format == GL_BGRA || format == GL_RGBA_INTEGER ||
                    format == GL_BGRA_INTEGER) ? GL_NO_ERROR : GL_INVALID_OPERATION;
    }
}

/* Readback-time compatibility (GetTexImage / GetTextureSubImage): honours the
 * depth-stencil texture mode.  GL 4.6 8.11.4 — with DEPTH_STENCIL_TEXTURE_MODE =
 * STENCIL_INDEX a combined depth-stencil texture behaves as STENCIL_INDEX8 for
 * all reads (GL_RED_INTEGER / GL_STENCIL_INTEGER answers); with DEPTH_COMPONENT
 * (the default) the depth channel additionally answers GL_RED/GL_FLOAT reads
 * (KHR-GL46.stencil_texturing.functional reads the D channel that way). */
GLenum tdCheckGetFormatCompat(const TDFormat *f, GLenum format, GLenum type, GLenum ds_mode)
{
    if (f->base == GL_DEPTH_STENCIL)
    {
        if (ds_mode == GL_STENCIL_INDEX)
        {
            if (format == GL_STENCIL_INDEX || format == GL_RED_INTEGER)
                return tdCheckFormatCompat(tdFormatInfo(GL_STENCIL_INDEX8), format, type, TD_PIX_GETTEXIMAGE);
            return GL_INVALID_OPERATION;
        }
        /* GL_RED / GL_FLOAT readback of a combined depth-stencil texture is NOT
         * legal (GL 4.6 table 8.2: only DEPTH_COMPONENT / DEPTH_STENCIL client
         * formats) — KHR-GL46.packed_pixels.varied_rectangle.depth24_stencil8
         * expects an error for every non-depth/stencil format, GL_RED included.
         * (stencil_texturing reads its R32F RESULT texture with RED/FLOAT, not the
         * depth-stencil texture.) */
    }
    return tdCheckFormatCompat(f, format, type, TD_PIX_GETTEXIMAGE);
}

GLenum tdCheckFormatCompat(const TDFormat *f, GLenum format, GLenum type, int op)
{
    TDClientFormat cf;
    if (!td_client_format(format, &cf)) return GL_INVALID_ENUM;
    bool tex_depth = (f->base == GL_DEPTH_COMPONENT || f->base == GL_DEPTH_STENCIL);
    bool tex_stencil = (f->base == GL_STENCIL_INDEX || f->base == GL_DEPTH_STENCIL);
    bool tex_ds = tex_depth || tex_stencil;

    switch (format)
    {
        case GL_DEPTH_COMPONENT:
            if (!tex_depth) return GL_INVALID_OPERATION;
            break;
        case GL_DEPTH_STENCIL:
            /* DEPTH_STENCIL data also feeds a DEPTH_COMPONENT texture (the stencil
             * component is dropped); reading it back needs the two-channel internal
             * format (GL 4.6 8.4: getTexImage of a depth-only texture answers
             * DEPTH_COMPONENT / FLOAT) */
            if (!tex_depth) return GL_INVALID_OPERATION;
            if (f->base != GL_DEPTH_STENCIL && op != TD_PIX_TEXIMAGE && op != TD_PIX_SUBIMAGE)
                return GL_INVALID_OPERATION;
            break;
        case GL_STENCIL_INDEX:
            /* glTexImage cannot upload stencil data at all (GL 4.6 table 8.11: the
             * stencil internal formats are TexStorage-only), SubImage and the clears
             * can (table 8.14) */
            if (!tex_stencil) return GL_INVALID_OPERATION;
            if (op == TD_PIX_TEXIMAGE) return GL_INVALID_OPERATION;
            break;
        default:
            /* stencil-indexed textures also answer GL_RED_INTEGER READS (the
             * stencil-texturing readback form, KHR-GL46.stencil_texturing).  The
             * exception is readback-only and STENCIL-ONLY textures: GL_RED_INTEGER
             * stays INVALID for uploads and for combined depth-stencil formats
             * (KHR-GL46.packed_pixels.varied_rectangle.depth24_stencil8 /
             * depth32f_stencil8 reject RED_INTEGER TexImage: table 8.11), whose
             * stencil read goes through tdCheckGetFormatCompat's ds_mode branch */
            {
                bool stencil_read = (op == TD_PIX_GETTEXIMAGE && f->base == GL_STENCIL_INDEX &&
                                     format == GL_RED_INTEGER);
                if (tex_ds && !stencil_read) return GL_INVALID_OPERATION;
                /* integer formats need *_INTEGER client data and vice versa; the
                 * stencil-index internal formats predate that flag, so accept the
                 * stencil-texturing read regardless */
                if (((f->flags & TDF_INTEGER) != 0) != cf.integer && !stencil_read)
                    return GL_INVALID_OPERATION;
            }
            break;
    }
    if (op == TD_PIX_TEXIMAGE && f->base == GL_STENCIL_INDEX)
        return GL_INVALID_OPERATION;      /* no stencil internal format for TexImage */
    (void)type;
    return GL_NO_ERROR;
}

size_t tdClientPixelSize(GLenum format, GLenum type)
{
    TDClientFormat cf;
    TDPacked pk;
    if (!td_client_format(format, &cf)) return 0;
    switch (type)
    {
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
            return 4;
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 8;
        default:
            if (td_packed(type, &pk)) return (size_t)pk.bytes;
            return (size_t)(td_type_size(type) * cf.n);
    }
}

static int td_element_size(GLenum format, GLenum type)
{
    TDPacked pk;
    if (td_packed(type, &pk)) return pk.bytes;
    switch (type)
    {
        case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8: case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 4;
        default:
            (void)format;
            return td_type_size(type);
    }
}

size_t tdClientLayout(const TDPixelStore *ps, GLenum format, GLenum type, int w, int h, int d,
                      bool is3d, size_t *offset, size_t *row_stride, size_t *image_stride)
{
    size_t group = tdClientPixelSize(format, type);
    size_t s = (size_t)td_element_size(format, type);
    size_t a = ps->alignment > 0 ? (size_t)ps->alignment : 1;
    size_t l = ps->row_length > 0 ? (size_t)ps->row_length : (size_t)w;
    size_t row = l * group;

    if (s < a && row % a)
        row = (row + a - 1) / a * a;

    size_t rows = (is3d && ps->image_height > 0) ? (size_t)ps->image_height : (size_t)h;
    size_t img = row * rows;
    size_t off = (size_t)ps->skip_pixels * group + (size_t)ps->skip_rows * row + (is3d ? (size_t)ps->skip_images * img : 0);

    if (offset) *offset = off;
    if (row_stride) *row_stride = row;
    if (image_stride) *image_stride = img;

    if (w <= 0 || h <= 0 || d <= 0) return 0;
    return off + (size_t)(d - 1) * img + (size_t)(h - 1) * row + (size_t)w * group;
}

size_t tdCompressedClientLayout(const TDPixelStore *ps, const TDFormat *f, int w, int h, int d,
                                bool is3d, bool *uses_block_state,
                                size_t *offset, size_t *row_stride, size_t *image_stride)
{
    size_t bw = f->block_w, bh = f->block_h, bb = f->block_bytes;
    size_t bx = ((size_t)w + bw - 1) / bw, by = ((size_t)h + bh - 1) / bh;
    bool blk = ps->cb_width && ps->cb_height && ps->cb_size && (!is3d || ps->cb_depth);
    size_t row, img, off;

    if (blk)
    {
        size_t l = ps->row_length > 0 ? ((size_t)ps->row_length + bw - 1) / bw : bx;
        size_t ih = (is3d && ps->image_height > 0) ? ((size_t)ps->image_height + bh - 1) / bh : by;
        row = l * bb;
        img = row * ih;
        off = ((size_t)ps->skip_pixels / bw) * bb + ((size_t)ps->skip_rows / bh) * row +
              (is3d ? (size_t)ps->skip_images * img : 0);
    }
    else
    {
        row = bx * bb;
        img = row * by;
        off = 0;
    }

    if (uses_block_state) *uses_block_state = blk;
    if (offset) *offset = off;
    if (row_stride) *row_stride = row;
    if (image_stride) *image_stride = img;
    if (w <= 0 || h <= 0 || d <= 0) return 0;
    return off + (size_t)(d - 1) * img + (by - 1) * row + bx * bb;
}

/* ------------------------------------------------------------------------ */
/* intermediate pixel                                                       */
/* ------------------------------------------------------------------------ */

typedef struct { double c[4]; double d; uint32_t s; } TDPix;

static inline uint16_t bswap16(uint16_t v) { return (uint16_t)((v >> 8) | (v << 8)); }
static inline uint32_t bswap32(uint32_t v) { return __builtin_bswap32(v); }

static inline uint32_t rd16(const uint8_t *p, bool sw) { uint16_t v; memcpy(&v, p, 2); return sw ? bswap16(v) : v; }
static inline uint32_t rd32(const uint8_t *p, bool sw) { uint32_t v; memcpy(&v, p, 4); return sw ? bswap32(v) : v; }
static inline void wr16(uint8_t *p, uint32_t v, bool sw) { uint16_t x = (uint16_t)v; if (sw) x = bswap16(x); memcpy(p, &x, 2); }
static inline void wr32(uint8_t *p, uint32_t v, bool sw) { if (sw) v = bswap32(v); memcpy(p, &v, 4); }

static inline double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline double un_to_f(double v, int bits) { return v / (double)((1ull << bits) - 1); }
static inline double sn_to_f(double v, int bits) { double r = v / (double)((1ull << (bits - 1)) - 1); return r < -1.0 ? -1.0 : r; }
static inline uint64_t f_to_un(double f, int bits){
    if (isnan(f)) f = 0;
    return (uint64_t)llround(clampd(f, 0.0, 1.0) * (double)((1ull << bits) - 1));
}
static inline int64_t f_to_sn(double f, int bits)
{
    if (isnan(f)) f = 0;
    /* GL 4.6 2.3.5.2: -1.0 maps to the MOST NEGATIVE value, not to -(2^(b-1)-1), so the
     * snorm round-trip of the minimum is exact (RGB8_SNORM uploads of -128 came back -127) */
    if (f <= -1.0) return -(int64_t)(1ull << (bits - 1));
    return (int64_t)llround(clampd(f, -1.0, 1.0) * (double)((1ull << (bits - 1)) - 1));
}

static void pix_default(TDPix *p) { p->c[0] = p->c[1] = p->c[2] = 0.0; p->c[3] = 1.0; p->d = 0.0; p->s = 0; }

/* client -> intermediate, one row */
static void client_read_row(const uint8_t *src, GLenum format, GLenum type, bool sw, int w, TDPix *out)
{
    TDClientFormat cf;
    TDPacked pk;
    td_client_format(format, &cf);
    size_t gs = tdClientPixelSize(format, type);

    for (int x = 0; x < w; x++, src += gs)
    {
        TDPix *p = &out[x];
        pix_default(p);

        if (type == GL_UNSIGNED_INT_24_8)
        {
            uint32_t v = rd32(src, sw);
            p->d = un_to_f((double)(v >> 8), 24);
            p->s = v & 0xff;
            continue;
        }
        if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
        {
            uint32_t v = rd32(src, sw); float fd; memcpy(&fd, &v, 4);
            p->d = fd;
            p->s = rd32(src + 4, sw) & 0xff;
            continue;
        }
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV)
        {
            uint32_t v = rd32(src, sw);
            p->c[0] = td_uf_to_float(v & 0x7ff, 6);
            p->c[1] = td_uf_to_float((v >> 11) & 0x7ff, 6);
            p->c[2] = td_uf_to_float((v >> 22) & 0x3ff, 5);
            continue;
        }
        if (type == GL_UNSIGNED_INT_5_9_9_9_REV)
        {
            td_decode_rgb9e5(rd32(src, sw), p->c);
            continue;
        }
        if (td_packed(type, &pk))
        {
            uint32_t v = pk.bytes == 1 ? src[0] : pk.bytes == 2 ? rd16(src, sw) : rd32(src, sw);
            int total = pk.bytes * 8, shift = 0;
            for (int i = 0; i < pk.n && i < cf.n; i++)
            {
                int wbits = pk.w[i];
                int sh = pk.rev ? shift : total - shift - wbits;
                uint32_t f = (v >> sh) & ((1u << wbits) - 1);
                shift += wbits;
                p->c[cf.comp[i]] = cf.integer ? (double)f : un_to_f((double)f, wbits);
            }
            continue;
        }

        int es = td_type_size(type);
        for (int i = 0; i < cf.n; i++)
        {
            const uint8_t *e = src + i * es;
            double v = 0.0;
            bool raw = cf.integer || cf.comp[i] == CS;
            switch (type)
            {
                case GL_UNSIGNED_BYTE:  v = raw ? e[0] : un_to_f(e[0], 8); break;
                case GL_BYTE:           v = raw ? (int8_t)e[0] : sn_to_f((int8_t)e[0], 8); break;
                case GL_UNSIGNED_SHORT: v = raw ? rd16(e, sw) : un_to_f(rd16(e, sw), 16); break;
                case GL_SHORT:          v = raw ? (int16_t)rd16(e, sw) : sn_to_f((int16_t)rd16(e, sw), 16); break;
                case GL_UNSIGNED_INT:   v = raw ? rd32(e, sw) : un_to_f(rd32(e, sw), 32); break;
                case GL_INT:            v = raw ? (int32_t)rd32(e, sw) : sn_to_f((int32_t)rd32(e, sw), 32); break;
                case GL_HALF_FLOAT:     v = tdHalfToFloat((uint16_t)rd16(e, sw)); break;
                case GL_FLOAT:          { uint32_t u = rd32(e, sw); float fv; memcpy(&fv, &u, 4); v = fv; } break;
            }
            int c = cf.comp[i];
            if (c == CD) p->d = v;
            else if (c == CS) p->s = (uint32_t)(int64_t)v;
            else p->c[c] = v;
        }
    }
}

/* intermediate -> client, one row */
static void client_write_row(uint8_t *dst, GLenum format, GLenum type, bool sw, int w, const TDPix *in)
{
    TDClientFormat cf;
    TDPacked pk;
    td_client_format(format, &cf);
    size_t gs = tdClientPixelSize(format, type);

    for (int x = 0; x < w; x++, dst += gs)
    {
        const TDPix *p = &in[x];

        if (type == GL_UNSIGNED_INT_24_8)
        {
            wr32(dst, (uint32_t)(f_to_un(p->d, 24) << 8) | (p->s & 0xff), sw);
            continue;
        }
        if (type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV)
        {
            float fd = (float)p->d; uint32_t u; memcpy(&u, &fd, 4);
            wr32(dst, u, sw);
            wr32(dst + 4, p->s & 0xff, sw);
            continue;
        }
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV)
        {
            uint32_t v = td_float_to_uf((float)p->c[0], 6) | (td_float_to_uf((float)p->c[1], 6) << 11) |
                         (td_float_to_uf((float)p->c[2], 5) << 22);
            wr32(dst, v, sw);
            continue;
        }
        if (type == GL_UNSIGNED_INT_5_9_9_9_REV)
        {
            wr32(dst, td_encode_rgb9e5(p->c[0], p->c[1], p->c[2]), sw);
            continue;
        }
        if (td_packed(type, &pk))
        {
            uint32_t v = 0;
            int total = pk.bytes * 8, shift = 0;
            for (int i = 0; i < pk.n && i < cf.n; i++)
            {
                int wbits = pk.w[i];
                int sh = pk.rev ? shift : total - shift - wbits;
                double c = p->c[cf.comp[i]];
                uint32_t mx = (1u << wbits) - 1, f;
                if (cf.integer) f = (uint32_t)clampd(c, 0.0, (double)mx);
                else f = (uint32_t)f_to_un(c, wbits);
                v |= (f & mx) << sh;
                shift += wbits;
            }
            if (pk.bytes == 1) dst[0] = (uint8_t)v;
            else if (pk.bytes == 2) wr16(dst, v, sw);
            else wr32(dst, v, sw);
            continue;
        }

        int es = td_type_size(type);
        for (int i = 0; i < cf.n; i++)
        {
            uint8_t *e = dst + i * es;
            int c = cf.comp[i];
            double v = (c == CD) ? p->d : (c == CS) ? (double)p->s : p->c[c];
            bool raw = cf.integer || c == CS;
            switch (type)
            {
                case GL_UNSIGNED_BYTE:  e[0] = (uint8_t)(raw ? clampd(v, 0, 255) : f_to_un(v, 8)); break;
                case GL_BYTE:           e[0] = (uint8_t)(int8_t)(raw ? clampd(v, -128, 127) : f_to_sn(v, 8)); break;
                case GL_UNSIGNED_SHORT: wr16(e, (uint32_t)(raw ? clampd(v, 0, 65535) : f_to_un(v, 16)), sw); break;
                case GL_SHORT:          wr16(e, (uint32_t)(uint16_t)(int16_t)(raw ? clampd(v, -32768, 32767) : f_to_sn(v, 16)), sw); break;
                case GL_UNSIGNED_INT:   wr32(e, (uint32_t)(raw ? clampd(v, 0, 4294967295.0) : f_to_un(v, 32)), sw); break;
                case GL_INT:            wr32(e, (uint32_t)(int32_t)(raw ? clampd(v, -2147483648.0, 2147483647.0) : f_to_sn(v, 32)), sw); break;
                case GL_HALF_FLOAT:     wr16(e, tdFloatToHalf((float)v), sw); break;
                case GL_FLOAT:          { float fv = (float)v; uint32_t u; memcpy(&u, &fv, 4); wr32(e, u, sw); } break;
            }
        }
    }
}

/* storage component i <-> intermediate component index */
static int storage_comp(const TDFormat *f, int i)
{
    switch (f->base)
    {
        case GL_ALPHA: return CA;
        case GL_LUMINANCE: return CR;
        case GL_LUMINANCE_ALPHA: return i == 0 ? CR : CA;
        default: return i;
    }
}

/* force components that the base format does not have to their defaults */
static void apply_base(const TDFormat *f, TDPix *p, int w)
{
    bool hr = true, hg = true, hb = true, ha = true;
    switch (f->base)
    {
        case GL_RED: hg = hb = ha = false; break;
        case GL_RG: hb = ha = false; break;
        case GL_RGB: ha = false; break;
        case GL_ALPHA: hr = hg = hb = false; break;
        case GL_LUMINANCE: hg = hb = ha = false; break;
        case GL_LUMINANCE_ALPHA: hg = hb = false; break;
        default: break;
    }
    for (int x = 0; x < w; x++)
    {
        if (!hr) p[x].c[0] = 0.0;
        if (!hg) p[x].c[1] = 0.0;
        if (!hb) p[x].c[2] = 0.0;
        if (!ha) p[x].c[3] = 1.0;
    }
}

static void storage_read_row(const TDFormat *f, const uint8_t *src, const uint8_t *st, int w, TDPix *out)
{
    for (int x = 0; x < w; x++)
    {
        TDPix *p = &out[x];
        pix_default(p);
        const uint8_t *s = src + (size_t)x * f->bpp;
        switch (f->kind)
        {
            case TDK_RGB10A2UN: case TDK_RGB10A2UI:
            {
                uint32_t v = rd32(s, false);
                uint32_t c[4] = { v & 1023, (v >> 10) & 1023, (v >> 20) & 1023, v >> 30 };
                bool in = f->kind == TDK_RGB10A2UI;
                for (int i = 0; i < 4; i++) p->c[i] = in ? (double)c[i] : un_to_f(c[i], i == 3 ? 2 : 10);
                continue;
            }
            case TDK_RG11B10F:
            {
                uint32_t v = rd32(s, false);
                p->c[0] = td_uf_to_float(v & 0x7ff, 6);
                p->c[1] = td_uf_to_float((v >> 11) & 0x7ff, 6);
                p->c[2] = td_uf_to_float(v >> 22, 5);
                continue;
            }
            case TDK_RGB9E5:
                td_decode_rgb9e5(rd32(s, false), p->c);
                continue;
            case TDK_D16: p->d = un_to_f(rd16(s, false), 16); continue;
            case TDK_D16F: { float fd; memcpy(&fd, s, 4); p->d = fd; } continue;   /* already on the 16-bit lattice */
            case TDK_D32F: { float fd; memcpy(&fd, s, 4); p->d = fd; } continue;
            case TDK_S8: p->s = s[0]; continue;
            case TDK_D32FS8:
            {
                float fd; memcpy(&fd, s, 4); p->d = fd;
                p->s = st ? st[x] : 0;
                continue;
            }
            default: break;
        }

        for (int i = 0; i < f->ncomp; i++)
        {
            double v = 0.0;
            switch (f->kind)
            {
                case TDK_UN8:  v = un_to_f(s[i], 8); break;
                case TDK_SN8:  v = sn_to_f((int8_t)s[i], 8); break;
                case TDK_UI8:  v = s[i]; break;
                case TDK_SI8:  v = (int8_t)s[i]; break;
                case TDK_UN16: v = un_to_f(rd16(s + 2 * i, false), 16); break;
                case TDK_SN16: v = sn_to_f((int16_t)rd16(s + 2 * i, false), 16); break;
                case TDK_UI16: v = rd16(s + 2 * i, false); break;
                case TDK_SI16: v = (int16_t)rd16(s + 2 * i, false); break;
                case TDK_F16:  v = tdHalfToFloat((uint16_t)rd16(s + 2 * i, false)); break;
                case TDK_UI32: v = rd32(s + 4 * i, false); break;
                case TDK_SI32: v = (int32_t)rd32(s + 4 * i, false); break;
                case TDK_F32:  { float fv; memcpy(&fv, s + 4 * i, 4); v = fv; } break;
            }
            p->c[storage_comp(f, i)] = v;
        }
    }
    apply_base(f, out, w);
}

static void storage_write_row(const TDFormat *f, uint8_t *dst, uint8_t *st, int w, TDPix *in, bool wd, bool ws)
{
    bool clamp_depth = (f->comp_type == GL_UNSIGNED_NORMALIZED);
    apply_base(f, in, w);

    for (int x = 0; x < w; x++)
    {
        TDPix *p = &in[x];
        uint8_t *s = dst + (size_t)x * f->bpp;

        switch (f->kind)
        {
            case TDK_RGB10A2UN:
                wr32(s, (uint32_t)(f_to_un(p->c[0], 10) | (f_to_un(p->c[1], 10) << 10) |
                                   (f_to_un(p->c[2], 10) << 20) | (f_to_un(p->c[3], 2) << 30)), false);
                continue;
            case TDK_RGB10A2UI:
                wr32(s, (uint32_t)clampd(p->c[0], 0, 1023) | ((uint32_t)clampd(p->c[1], 0, 1023) << 10) |
                        ((uint32_t)clampd(p->c[2], 0, 1023) << 20) | ((uint32_t)clampd(p->c[3], 0, 3) << 30), false);
                continue;
            case TDK_RG11B10F:
                wr32(s, td_float_to_uf((float)p->c[0], 6) | (td_float_to_uf((float)p->c[1], 6) << 11) |
                        (td_float_to_uf((float)p->c[2], 5) << 22), false);
                continue;
            case TDK_RGB9E5:
                wr32(s, td_encode_rgb9e5(p->c[0], p->c[1], p->c[2]), false);
                continue;
            case TDK_D16:
                if (wd) wr16(s, (uint32_t)f_to_un(p->d, 16), false);
                continue;
            case TDK_D16F:
                /* D16 semantics in D32F storage: quantize to the 16-bit lattice so
                 * sampling and the shadow compare see exactly the GL D16 value */
                if (wd) { float fd = (float)((double)f_to_un(p->d, 16) / 65535.0); memcpy(s, &fd, 4); }
                continue;
            case TDK_D32F:
                if (wd) { float fd = (float)(clamp_depth ? clampd(p->d, 0.0, 1.0) : p->d); memcpy(s, &fd, 4); }
                continue;
            case TDK_S8:
                if (ws) s[0] = (uint8_t)p->s;
                continue;
            case TDK_D32FS8:
                if (wd) { float fd = (float)(clamp_depth ? clampd(p->d, 0.0, 1.0) : p->d); memcpy(s, &fd, 4); }
                if (ws && st) st[x] = (uint8_t)p->s;
                continue;
            default: break;
        }

        for (int i = 0; i < f->ncomp; i++)
        {
            double v = p->c[storage_comp(f, i)];
            switch (f->kind)
            {
                case TDK_UN8:  s[i] = (uint8_t)f_to_un(v, 8); break;
                case TDK_SN8:  s[i] = (uint8_t)(int8_t)f_to_sn(v, 8); break;
                case TDK_UI8:  s[i] = (uint8_t)clampd(v, 0, 255); break;
                case TDK_SI8:  s[i] = (uint8_t)(int8_t)clampd(v, -128, 127); break;
                case TDK_UN16: wr16(s + 2 * i, (uint32_t)f_to_un(v, 16), false); break;
                case TDK_SN16: wr16(s + 2 * i, (uint32_t)(uint16_t)(int16_t)f_to_sn(v, 16), false); break;
                case TDK_UI16: wr16(s + 2 * i, (uint32_t)clampd(v, 0, 65535), false); break;
                case TDK_SI16: wr16(s + 2 * i, (uint32_t)(uint16_t)(int16_t)clampd(v, -32768, 32767), false); break;
                case TDK_F16:  wr16(s + 2 * i, tdFloatToHalf((float)v), false); break;
                case TDK_UI32: wr32(s + 4 * i, (uint32_t)clampd(v, 0, 4294967295.0), false); break;
                case TDK_SI32: wr32(s + 4 * i, (uint32_t)(int32_t)clampd(v, -2147483648.0, 2147483647.0), false); break;
                case TDK_F32:  { float fv = (float)v; memcpy(s + 4 * i, &fv, 4); } break;
            }
        }
    }
}

/* Is the client layout byte-identical to the storage layout? */
static bool td_direct(const TDFormat *f, GLenum format, GLenum type)
{
    if (f->flags & TDF_LEGACY) return false;
    TDClientFormat cf;
    if (!td_client_format(format, &cf)) return false;

    switch (f->kind)
    {
        case TDK_RGB10A2UN: return format == GL_RGBA && type == GL_UNSIGNED_INT_2_10_10_10_REV && f->base == GL_RGBA;
        case TDK_RGB10A2UI: return format == GL_RGBA_INTEGER && type == GL_UNSIGNED_INT_2_10_10_10_REV;
        case TDK_RG11B10F:  return format == GL_RGB && type == GL_UNSIGNED_INT_10F_11F_11F_REV;
        case TDK_RGB9E5:    return format == GL_RGB && type == GL_UNSIGNED_INT_5_9_9_9_REV;
        case TDK_D16:       return format == GL_DEPTH_COMPONENT && type == GL_UNSIGNED_SHORT;
        case TDK_D16F:      return false;   /* must quantize to the 16-bit lattice, never copy raw */
        case TDK_D32F:      return format == GL_DEPTH_COMPONENT && type == GL_FLOAT && f->comp_type == GL_FLOAT;
        case TDK_S8:        return format == GL_STENCIL_INDEX && type == GL_UNSIGNED_BYTE;
        case TDK_D32FS8: case TDK_COMPRESSED: case TDK_NONE: return false;
        default: break;
    }

    int basen = f->base == GL_RED ? 1 : f->base == GL_RG ? 2 : f->base == GL_RGBA ? 4 : 3;
    if (basen != f->ncomp || cf.n != f->ncomp) return false;
    for (int i = 0; i < cf.n; i++) if (cf.comp[i] != i) return false;

    bool in = (f->flags & TDF_INTEGER) != 0;
    if (in != cf.integer) return false;

    switch (f->kind)
    {
        case TDK_UN8: return type == GL_UNSIGNED_BYTE && !in;
        case TDK_SN8: return type == GL_BYTE && !in;
        case TDK_UI8: return type == GL_UNSIGNED_BYTE && in;
        case TDK_SI8: return type == GL_BYTE && in;
        case TDK_UN16: return type == GL_UNSIGNED_SHORT && !in;
        case TDK_SN16: return type == GL_SHORT && !in;
        case TDK_UI16: return type == GL_UNSIGNED_SHORT && in;
        case TDK_SI16: return type == GL_SHORT && in;
        case TDK_F16: return type == GL_HALF_FLOAT;
        case TDK_UI32: return type == GL_UNSIGNED_INT && in;
        case TDK_SI32: return type == GL_INT && in;
        case TDK_F32: return type == GL_FLOAT;
        default: return false;
    }
}



void tdUnpack(const TDFormat *f, const void *src, size_t src_row, size_t src_img,
              GLenum format, GLenum type, bool swap_bytes,
              int w, int h, int d, const TDStorage *dst,
              bool *wrote_depth, bool *wrote_stencil)
{
    TDClientFormat cf;
    td_client_format(format, &cf);
    bool wd = false, ws = false;
    for (int i = 0; i < cf.n; i++) { if (cf.comp[i] == CD) wd = true; if (cf.comp[i] == CS) ws = true; }
    if (wrote_depth) *wrote_depth = wd;
    if (wrote_stencil) *wrote_stencil = ws;

    if (w <= 0 || h <= 0 || d <= 0) return;

    bool direct = !swap_bytes && td_direct(f, format, type);
    size_t rowbytes = (size_t)w * f->bpp;
    TDPix *tmp = direct ? NULL : (TDPix *)malloc(sizeof(TDPix) * (size_t)w);
    if (!direct && !tmp) return;

    for (int z = 0; z < d; z++)
    {
        for (int y = 0; y < h; y++)
        {
            const uint8_t *s = (const uint8_t *)src + (size_t)z * src_img + (size_t)y * src_row;
            uint8_t *o = (uint8_t *)dst->data + (size_t)z * dst->image_pitch + (size_t)y * dst->row_pitch;
            uint8_t *so = dst->stencil ? (uint8_t *)dst->stencil + (size_t)z * dst->s_image_pitch + (size_t)y * dst->s_row_pitch : NULL;

            if (direct) { memcpy(o, s, rowbytes); continue; }

            client_read_row(s, format, type, swap_bytes, w, tmp);
            /* No sRGB gamma conversion here: GL applies the sRGB transfer function
             * ONLY while sampling/filtering an sRGB internal format (GL 4.6 8.5.1);
             * pixel transfer (TexImage/TexSubImage/GetTexImage) moves raw storage
             * values through the canonical float (GL 4.6 8.5.4) — a FLOAT upload of
             * 0.25 stores byte 64 and a FLOAT readback returns 64/255.  (Encoding
             * on upload returned decode(encode(v)) from every gather/sample and
             * failed KHR-GL46.texture_gather plain-gather-float-2d-srgb and the
             * packed_pixels sRGB mixed-type round trips.) */
            storage_write_row(f, o, so, w, tmp, wd, ws);
        }
    }
    free(tmp);
}

void tdPack(const TDFormat *f, const TDStorage *src, int w, int h, int d,
            GLenum format, GLenum type, bool swap_bytes,
            void *dst, size_t dst_row, size_t dst_img)
{
    if (w <= 0 || h <= 0 || d <= 0) return;

    bool direct = !swap_bytes && td_direct(f, format, type);
    size_t rowbytes = (size_t)w * f->bpp;
    TDPix *tmp = direct ? NULL : (TDPix *)malloc(sizeof(TDPix) * (size_t)w);
    if (!direct && !tmp) return;

    /* the client's integer-ness / signedness (for the snorm->unorm readback map) */
    bool cf_integer = false, cf_signed_norm = false, cf_float_client = false;
    {
        TDClientFormat cf;

        if (td_client_format(format, &cf))
        {
            cf_integer = cf.integer;
            cf_signed_norm = !cf.integer && (type == GL_BYTE || type == GL_SHORT ||
                                             type == GL_INT);
            cf_float_client = (type == GL_FLOAT || type == GL_HALF_FLOAT ||
                               type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                               type == GL_UNSIGNED_INT_5_9_9_9_REV ||
                               type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV);
        }
    }

    for (int z = 0; z < d; z++)
    {
        for (int y = 0; y < h; y++)
        {
            const uint8_t *s = (const uint8_t *)src->data + (size_t)z * src->image_pitch + (size_t)y * src->row_pitch;
            const uint8_t *ss = src->stencil ? (const uint8_t *)src->stencil + (size_t)z * src->s_image_pitch + (size_t)y * src->s_row_pitch : NULL;
            uint8_t *o = (uint8_t *)dst + (size_t)z * dst_img + (size_t)y * dst_row;

            if (direct) { memcpy(o, s, rowbytes); continue; }

            storage_read_row(f, s, ss, w, tmp);
            /* a depth-stencil storage read through GL_RED carries the DEPTH channel
             * and, in stencil-texturing mode, GL_RED_INTEGER the STENCIL plane
             * (GL_RED/GL_RED_INTEGER otherwise address the (empty) colour channel) */
            if (format == GL_RED &&
                (f->kind == TDK_D16 || f->kind == TDK_D16F || f->kind == TDK_D32F || f->kind == TDK_D32FS8))
            {
                for (int x = 0; x < w; x++) tmp[x].c[0] = tmp[x].d;
            }
            else if (format == GL_RED_INTEGER &&
                     (f->kind == TDK_S8 || f->kind == TDK_D32FS8))
            {
                for (int x = 0; x < w; x++) tmp[x].c[0] = (double)tmp[x].s;
            }
            /* No sRGB decode on readback either (GL 4.6 8.5.1/8.5.4: the gamma
             * conversion is a sampling-time operation only; GetTexImage moves raw
             * storage values — FLOAT clients get storage/255, not decode(storage)) */
            /* a signed-normalized storage read back through any client keeps its value:
             * GL 4.6 8.5.4 converts through floats and clamps to what the CLIENT type
             * represents (an UNSIGNED_BYTE read of -0.5 clamps to 0, it is not biased to
             * 0.25 - KHR-GL46.packed_pixels compares these) */
            client_write_row(o, format, type, swap_bytes, w, tmp);
        }
    }
    free(tmp);
}

void tdConvertStorage(const TDFormat *sf, const TDStorage *src, const TDFormat *df, const TDStorage *dst,
                      int w, int h, int d)
{
    if (w <= 0 || h <= 0 || d <= 0) return;
    TDPix *tmp = (TDPix *)malloc(sizeof(TDPix) * (size_t)w);
    if (!tmp) return;
    bool same_class = ((sf->flags & TDF_INTEGER) != 0) == ((df->flags & TDF_INTEGER) != 0);

    for (int z = 0; z < d; z++)
        for (int y = 0; y < h; y++)
        {
            const uint8_t *s = (const uint8_t *)src->data + (size_t)z * src->image_pitch + (size_t)y * src->row_pitch;
            const uint8_t *ss = src->stencil ? (const uint8_t *)src->stencil + (size_t)z * src->s_image_pitch + (size_t)y * src->s_row_pitch : NULL;
            uint8_t *o = (uint8_t *)dst->data + (size_t)z * dst->image_pitch + (size_t)y * dst->row_pitch;
            uint8_t *so = dst->stencil ? (uint8_t *)dst->stencil + (size_t)z * dst->s_image_pitch + (size_t)y * dst->s_row_pitch : NULL;
            storage_read_row(sf, s, ss, w, tmp);
            if (!same_class)
                for (int x = 0; x < w; x++) for (int c = 0; c < 4; c++) tmp[x].c[c] = 0;
            storage_write_row(df, o, so, w, tmp, true, true);
        }
    free(tmp);
}

/* ------------------------------------------------------------------------ */
/* ETC2 / EAC decoder (Khronos Data Format Spec, ETC2 / EAC chapters)       */
/* ------------------------------------------------------------------------ */

static const int etc1_mod[8][2] = {
    {2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183}
};

static const int etc2_dist[8] = { 3, 6, 11, 16, 23, 32, 41, 64 };

static const int eac_mod[16][8] = {
    {-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12}, {-2, -5, -8, -13, 1, 4, 7, 12},
    {-2, -4, -6, -13, 1, 3, 5, 12}, {-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10},
    {-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10}, {-2, -6, -8, -10, 1, 5, 7, 9},
    {-2, -5, -8, -10, 1, 4, 7, 9}, {-2, -4, -8, -10, 1, 3, 7, 9}, {-2, -5, -7, -10, 1, 4, 6, 9},
    {-3, -4, -7, -10, 2, 3, 6, 9}, {-1, -2, -3, -10, 0, 1, 2, 9}, {-4, -6, -8, -9, 3, 5, 7, 8},
    {-3, -5, -7, -9, 2, 4, 6, 8}
};

static inline uint64_t be64(const uint8_t *b)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
    return v;
}

static inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }
static inline uint32_t bits(uint64_t v, int hi, int lo) { return (uint32_t)((v >> lo) & ((1ull << (hi - lo + 1)) - 1)); }
static inline int ext4(int v) { return (v << 4) | v; }
static inline int ext5(int v) { return (v << 3) | (v >> 2); }
static inline int ext6(int v) { return (v << 2) | (v >> 4); }
static inline int ext7(int v) { return (v << 1) | (v >> 6); }

/* decode one ETC2 RGB block into out[y][x][4]; punch = punchthrough alpha variant */
static void etc2_rgb_block(const uint8_t *blk, uint8_t out[4][4][4], bool punch)
{
    uint64_t v = be64(blk);
    bool diff = bits(v, 33, 33);
    bool flip = bits(v, 32, 32);
    bool opaque = true;
    uint32_t idx = (uint32_t)v;

    if (punch) { opaque = diff; diff = true; }

    int base[2][3];
    int mode = 0;   /* 0 = individual/differential, 1 = T, 2 = H, 3 = planar */

    if (!diff)
    {
        base[0][0] = ext4(bits(v, 63, 60)); base[1][0] = ext4(bits(v, 59, 56));
        base[0][1] = ext4(bits(v, 55, 52)); base[1][1] = ext4(bits(v, 51, 48));
        base[0][2] = ext4(bits(v, 47, 44)); base[1][2] = ext4(bits(v, 43, 40));
    }
    else
    {
        int r = (int)bits(v, 63, 59), dr = (int)bits(v, 58, 56);
        int g = (int)bits(v, 55, 51), dg = (int)bits(v, 50, 48);
        int b = (int)bits(v, 47, 43), db = (int)bits(v, 42, 40);
        if (dr & 4) dr -= 8;
        if (dg & 4) dg -= 8;
        if (db & 4) db -= 8;
        if (r + dr < 0 || r + dr > 31) mode = 1;
        else if (g + dg < 0 || g + dg > 31) mode = 2;
        else if (b + db < 0 || b + db > 31) mode = 3;
        else
        {
            base[0][0] = ext5(r); base[1][0] = ext5(r + dr);
            base[0][1] = ext5(g); base[1][1] = ext5(g + dg);
            base[0][2] = ext5(b); base[1][2] = ext5(b + db);
        }
    }

    if (mode == 0)
    {
        int cw[2] = { (int)bits(v, 39, 37), (int)bits(v, 36, 34) };
        for (int x = 0; x < 4; x++)
            for (int y = 0; y < 4; y++)
            {
                int sub = flip ? (y >= 2) : (x >= 2);
                int bit = x * 4 + y;
                int pi = (int)(((idx >> (bit + 16)) & 1) << 1 | ((idx >> bit) & 1));
                uint8_t *o = out[y][x];
                if (punch && !opaque && pi == 2) { o[0] = o[1] = o[2] = o[3] = 0; continue; }
                int m;
                if (punch && !opaque && pi == 0) m = 0;
                else
                {
                    int a = etc1_mod[cw[sub]][0], bb = etc1_mod[cw[sub]][1];
                    m = pi == 0 ? a : pi == 1 ? bb : pi == 2 ? -a : -bb;
                }
                o[0] = (uint8_t)clamp255(base[sub][0] + m);
                o[1] = (uint8_t)clamp255(base[sub][1] + m);
                o[2] = (uint8_t)clamp255(base[sub][2] + m);
                o[3] = 255;
            }
        return;
    }

    if (mode == 3)
    {
        int ro = ext6((int)bits(v, 62, 57));
        int go = ext7((int)((bits(v, 56, 56) << 6) | bits(v, 54, 49)));
        int bo = ext6((int)((bits(v, 48, 48) << 5) | (bits(v, 44, 43) << 3) | bits(v, 41, 39)));
        int rh = ext6((int)((bits(v, 38, 34) << 1) | bits(v, 32, 32)));
        int gh = ext7((int)bits(v, 31, 25));
        int bh = ext6((int)bits(v, 24, 19));
        int rv = ext6((int)bits(v, 18, 13));
        int gv = ext7((int)bits(v, 12, 6));
        int bv = ext6((int)bits(v, 5, 0));
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
            {
                uint8_t *o = out[y][x];
                o[0] = (uint8_t)clamp255((x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2);
                o[1] = (uint8_t)clamp255((x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2);
                o[2] = (uint8_t)clamp255((x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2);
                o[3] = 255;
            }
        return;
    }

    int c1[3], c2[3], paint[4][3], d;
    if (mode == 1)
    {
        c1[0] = ext4((int)((bits(v, 60, 59) << 2) | bits(v, 57, 56)));
        c1[1] = ext4((int)bits(v, 55, 52));
        c1[2] = ext4((int)bits(v, 51, 48));
        c2[0] = ext4((int)bits(v, 47, 44));
        c2[1] = ext4((int)bits(v, 43, 40));
        c2[2] = ext4((int)bits(v, 39, 36));
        d = etc2_dist[(bits(v, 35, 34) << 1) | bits(v, 32, 32)];
        for (int i = 0; i < 3; i++)
        {
            paint[0][i] = c1[i];
            paint[1][i] = clamp255(c2[i] + d);
            paint[2][i] = c2[i];
            paint[3][i] = clamp255(c2[i] - d);
        }
    }
    else
    {
        c1[0] = ext4((int)bits(v, 62, 59));
        c1[1] = ext4((int)((bits(v, 58, 56) << 1) | bits(v, 52, 52)));
        c1[2] = ext4((int)((bits(v, 51, 51) << 3) | bits(v, 49, 47)));
        c2[0] = ext4((int)bits(v, 46, 43));
        c2[1] = ext4((int)bits(v, 42, 39));
        c2[2] = ext4((int)bits(v, 38, 35));
        int v1 = (c1[0] << 16) | (c1[1] << 8) | c1[2];
        int v2 = (c2[0] << 16) | (c2[1] << 8) | c2[2];
        int di = (int)((bits(v, 34, 34) << 2) | (bits(v, 32, 32) << 1) | (v1 >= v2 ? 1 : 0));
        d = etc2_dist[di];
        for (int i = 0; i < 3; i++)
        {
            paint[0][i] = clamp255(c1[i] + d);
            paint[1][i] = clamp255(c1[i] - d);
            paint[2][i] = clamp255(c2[i] + d);
            paint[3][i] = clamp255(c2[i] - d);
        }
    }

    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
        {
            int bit = x * 4 + y;
            int pi = (int)(((idx >> (bit + 16)) & 1) << 1 | ((idx >> bit) & 1));
            uint8_t *o = out[y][x];
            if (punch && !opaque && pi == 2) { o[0] = o[1] = o[2] = o[3] = 0; continue; }
            o[0] = (uint8_t)paint[pi][0];
            o[1] = (uint8_t)paint[pi][1];
            o[2] = (uint8_t)paint[pi][2];
            o[3] = 255;
        }
}

static void eac_alpha_block(const uint8_t *blk, uint8_t out[4][4][4])
{
    uint64_t v = be64(blk);
    int base = (int)bits(v, 63, 56), mul = (int)bits(v, 55, 52), tab = (int)bits(v, 51, 48);
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
        {
            int i = x * 4 + y;
            int pi = (int)bits(v, 47 - 3 * i, 45 - 3 * i);
            out[y][x][3] = (uint8_t)clamp255(base + eac_mod[tab][pi] * mul);
        }
}

/* EAC 11 bit channel -> 16 bit values (unsigned or signed) */
static void eac11_block(const uint8_t *blk, bool sgn, int32_t out[4][4])
{
    uint64_t v = be64(blk);
    int base = (int)bits(v, 63, 56), mul = (int)bits(v, 55, 52), tab = (int)bits(v, 51, 48);
    if (sgn) { base = (int8_t)base; if (base == -128) base = -127; }
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
        {
            int i = x * 4 + y;
            int pi = (int)bits(v, 47 - 3 * i, 45 - 3 * i);
            int m = eac_mod[tab][pi];
            int val;
            if (!sgn)
            {
                val = base * 8 + 4 + (mul ? m * mul * 8 : m);
                if (val < 0) val = 0;
                if (val > 2047) val = 2047;
                out[y][x] = (val << 5) | (val >> 6);
            }
            else
            {
                val = base * 8 + (mul ? m * mul * 8 : m);
                if (val < -1023) val = -1023;
                if (val > 1023) val = 1023;
                int a = val < 0 ? -val : val;
                int e = (a << 5) | (a >> 5);
                out[y][x] = val < 0 ? -e : e;
            }
        }
}

void tdDecodeRGTC(const TDFormat *f, const void *blocks, int w, int h, void *dst, size_t dst_row)
{
    const uint8_t *b = (const uint8_t *)blocks;
    uint8_t *out = (uint8_t *)dst;
    bool two = (f->base == GL_RG);
    bool snorm = (f->comp_type == GL_SIGNED_NORMALIZED);
    int ncomp = two ? 2 : 1;
    int bw = (w + 3) / 4, bh = (h + 3) / 4;

    for (int by = 0; by < bh; by++)
        for (int bx = 0; bx < bw; bx++)
        {
            const uint8_t *blk = b + ((size_t)by * bw + bx) * 8 * ncomp;
            for (int c = 0; c < ncomp; c++)
            {
                const uint8_t *p = blk + 8 * c;
                int e0 = snorm ? (int)(int8_t)p[0] : (int)p[0];
                int e1 = snorm ? (int)(int8_t)p[1] : (int)p[1];
                /* 48 bits of 3-bit codes, little endian */
                uint64_t bits = 0;
                for (int k = 0; k < 6; k++) bits |= (uint64_t)p[2 + k] << (8 * k);

                for (int yy = 0; yy < 4; yy++)
                {
                    int py = by * 4 + yy;
                    if (py >= h) continue;
                    for (int xx = 0; xx < 4; xx++)
                    {
                        int px = bx * 4 + xx;
                        if (px >= w) continue;
                        int code = (int)((bits >> (3 * (yy * 4 + xx))) & 7ull);
                        int v;
                        if (e0 > e1)
                        {
                            static const int w8[8][2] = { {6,1},{5,2},{4,3},{3,4},{2,5},{1,6} };
                            if (code == 0) v = e0;
                            else if (code == 1) v = e1;
                            else v = (w8[code - 2][0] * e0 + w8[code - 2][1] * e1) / 7;
                        }
                        else
                        {
                            static const int w6[4][2] = { {4,1},{3,2},{2,3},{1,4} };
                            if (code == 0) v = e0;
                            else if (code == 1) v = e1;
                            else if (code < 6) v = (w6[code - 2][0] * e0 + w6[code - 2][1] * e1) / 5;
                            else v = (code == 6) ? (snorm ? -128 : 0) : (snorm ? 127 : 255);
                        }
                        out[(size_t)py * dst_row + (size_t)px * ncomp + c] = (uint8_t)v;
                    }
                }
            }
        }
}

void tdDecodeETC(const TDFormat *f, const void *blocks, int w, int h, void *dst, size_t dst_row)
{
    const uint8_t *b = (const uint8_t *)blocks;
    int bx = (w + 3) / 4, by = (h + 3) / 4;
    GLenum fmt = f->internalformat;
    bool eac11 = (fmt == GL_COMPRESSED_R11_EAC || fmt == GL_COMPRESSED_SIGNED_R11_EAC ||
                  fmt == GL_COMPRESSED_RG11_EAC || fmt == GL_COMPRESSED_SIGNED_RG11_EAC);

    for (int j = 0; j < by; j++)
        for (int i = 0; i < bx; i++, b += f->block_bytes)
        {
            if (eac11)
            {
                bool sgn = (fmt == GL_COMPRESSED_SIGNED_R11_EAC || fmt == GL_COMPRESSED_SIGNED_RG11_EAC);
                int nch = (fmt == GL_COMPRESSED_RG11_EAC || fmt == GL_COMPRESSED_SIGNED_RG11_EAC) ? 2 : 1;
                int32_t ch[2][4][4];
                for (int c = 0; c < nch; c++) eac11_block(b + 8 * c, sgn, ch[c]);
                for (int y = 0; y < 4; y++)
                    for (int x = 0; x < 4; x++)
                    {
                        int px = i * 4 + x, py = j * 4 + y;
                        if (px >= w || py >= h) continue;
                        uint8_t *o = (uint8_t *)dst + (size_t)py * dst_row + (size_t)px * 2 * nch;
                        for (int c = 0; c < nch; c++)
                        {
                            uint16_t u = (uint16_t)(int16_t)ch[c][y][x];
                            memcpy(o + 2 * c, &u, 2);
                        }
                    }
                continue;
            }

            uint8_t px4[4][4][4];
            bool punch = (fmt == GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 ||
                          fmt == GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2);
            bool alpha = (fmt == GL_COMPRESSED_RGBA8_ETC2_EAC || fmt == GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC);
            etc2_rgb_block(alpha ? b + 8 : b, px4, punch);
            if (alpha) eac_alpha_block(b, px4);
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                {
                    int px = i * 4 + x, py = j * 4 + y;
                    if (px >= w || py >= h) continue;
                    memcpy((uint8_t *)dst + (size_t)py * dst_row + (size_t)px * 4, px4[y][x], 4);
                }
        }
}

/* ------------------------------------------------------------------ RGTC (BC4 / BC5) encoder
 *
 * "Online compression" (GL 4.6 spec 8.5.3): TexImage with an RGTC internal format and
 * UNCOMPRESSED client data must compress.  The encoder is deliberately simple — the
 * block endpoints are the min / max of its samples and every sample takes the nearest
 * of the six interpolation shades (plus the two extreme codes) — because nothing ever
 * compares the blocks against the source pixels: glGetCompressedTexImage returns
 * exactly these bytes and a re-upload of them stores them verbatim, which is what
 * KHR-GL46.direct_state_access.textures_compressed_subimage verifies byte-exactly.
 * min / max samples always pick endpoint 0 / 1 by construction, so encoding a decoded
 * block again yields the same bytes (the encoder is idempotent). */

static void rgtc_bc4_block(const uint8_t *p, size_t pitch, bool snorm, uint8_t out[8])
{
    int v[16], mn, mx, shade[8];
    uint64_t bits = 0;

    for (int i = 0; i < 16; i++)
    {
        int s = p[(i >> 2) * pitch + (i & 3)];
        if (snorm)
        {
            s -= 127;                       /* unsigned samples -> signed range */
            if (s < -127) s = -127;
            if (s > 127) s = 127;
        }
        v[i] = s;
    }
    mn = mx = v[0];
    for (int i = 1; i < 16; i++)
    {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }

    out[0] = (uint8_t)mn;                   /* endpoints: raw (sign-extended) bytes */
    out[1] = (uint8_t)mx;
    /* BC4/5 palette (code0 <= code1 branch): idx0 = code0, idx1 = code1,
     * idx2..5 interpolate in 1/5 steps, idx6/7 are the clamp values */
    shade[0] = mn;
    shade[1] = mx;
    for (int k = 0; k < 4; k++)
        shade[2 + k] = (mn * (4 - k) + mx * (1 + k)) / 5;   /* exact decoder math */
    shade[6] = snorm ? -127 : 0;
    shade[7] = snorm ? 127 : 255;

    for (int i = 0; i < 16; i++)
    {
        int best = 0, bestd = 1 << 30;
        for (int k = 0; k < 8; k++)
        {
            int dd = v[i] - shade[k];
            if (dd < 0) dd = -dd;
            if (dd < bestd) { bestd = dd; best = k; }
        }
        bits |= (uint64_t)best << (3 * i);
    }
    for (int b = 0; b < 6; b++)
        out[2 + b] = (uint8_t)((bits >> (8 * b)) & 0xff);
}

uint8_t *tdEncodeRGTC(const uint8_t *chan0, const uint8_t *chan1, int w, int h, int d,
                      size_t rw, size_t iw, bool two_channel, bool snorm)
{
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    size_t block_bytes = two_channel ? 16 : 8;
    size_t bytes = (size_t)bw * bh * (size_t)(d > 0 ? d : 1) * block_bytes;
    uint8_t *dst = (uint8_t *)malloc(bytes ? bytes : 1), *out;

    if (!dst)
        return NULL;

    out = dst;
    for (int z = 0; z < (d > 0 ? d : 1); z++)
    {
        const uint8_t *p0 = chan0 + (size_t)z * iw;
        const uint8_t *p1 = chan1 ? chan1 + (size_t)z * iw : NULL;

        for (int by = 0; by < bh; by++)
            for (int bx = 0; bx < bw; bx++)
            {
                uint8_t tile[2][16];
                int x0 = bx * 4, y0 = by * 4;

                memset(tile, 0, sizeof(tile));
                for (int y = 0; y < 4 && y0 + y < h; y++)
                {
                    const uint8_t *r0 = p0 + (size_t)(y0 + y) * rw + x0;
                    const uint8_t *r1 = p1 ? p1 + (size_t)(y0 + y) * rw + x0 : NULL;

                    for (int x = 0; x < 4 && x0 + x < w; x++)
                    {
                        tile[0][y * 4 + x] = r0[x];
                        if (r1) tile[1][y * 4 + x] = r1[x];
                    }
                }
                rgtc_bc4_block(tile[0], 4, snorm, out);
                if (two_channel)
                    rgtc_bc4_block(tile[1], 4, snorm, out + 8);
                out += block_bytes;
            }
    }
    return dst;
}

/* ------------------------------------------------------------------------ */
/* BPTC (BC6H / BC7) CPU encoders                                           */
/* ------------------------------------------------------------------------ */
/* Used for online compression: uncompressed client data uploaded to a
 * compressed BPTC internalformat (GL 4.6 8.5.3 lets the GL compress), and for
 * re-encoding decoded storage on glGetCompressedTexImage of such a level.
 *
 * Bit layouts follow the D3D / Khronos data-format specs, cross-checked field
 * by field against VK-GL-CTS's reference decoders decompressBc6H() /
 * decompressBc7() (framework/common/tcuCompressedTexture.cpp), which are the
 * conformance ground truth.  The decoders' exact arithmetic is replicated here
 * (unquantize / interpolate / finishUnquantize for BC6H; endpoint expansion +
 * interpolate for BC7), so encode -> decode round-trips reproduce the chosen
 * palette values on any conformant decoder, including the Metal hardware one.
 *
 * Only the simplest adequate single-subset modes are emitted:
 *   BC6H mode 10  - absolute (non-delta) 10-bit endpoints, 4-bit indices
 *   BC7  mode 6   - 7-bit + P RGB endpoints, 8-bit A endpoints, 4-bit indices
 * Both use per-channel range-fit endpoint selection and exact palette
 * reconstruction (nearest of the 16 interpolation weights). */

/* BC6H / BC7 4-bit interpolation weights (2.6 fixed point, /64) */
static const uint8_t bptc_w4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

static void bptc_put(uint8_t *dst, int bit, int nbits, uint32_t v)
{
    for (int i = 0; i < nbits; i++)
        if ((v >> i) & 1)
            dst[(bit + i) >> 3] |= (uint8_t)(1u << ((bit + i) & 7));
}

/* BC6H mode-10 unquantize of a 10-bit endpoint code -> the 16/17-bit value the
 * interpolator works on (decoder math; epBits[10] = 10) */
static int32_t bc6h_unquant(int32_t code, int isign)
{
    int32_t s = isign ? (code < 0 ? -1 : 1) : 1;
    int32_t m = isign ? (code < 0 ? -code : code) : code;

    if (m == 0)
        return 0;
    if (isign)
    {
        if (m >= 511)
            return s * 0x7fff;
    }
    else if (m >= 1023)
        return 0xffff;
    /* ((m << 15) + 0x4000) >> 9 == 64*m + 32 for 0 < m < the clamp above */
    return s * (64 * m + 32);
}

/* BC6H mode-10 finishUnquantize: interpolated value -> half bit pattern */
static uint32_t bc6h_finish(int32_t x, int isign)
{
    if (isign)
    {
        int32_t s = x < 0;
        int32_t m = s ? -x : x;
        m = (m * 31) >> 5;
        return (uint32_t)((s ? 0x8000u : 0u) | (uint32_t)m);
    }
    return (uint32_t)((x * 31) / 64);
}

/* decoded (interpolated, finished) half bits of palette entry idx */
static uint32_t bc6h_palette(int32_t u0, int32_t u1, int idx, int isign)
{
    int32_t w = bptc_w4[idx];
    int32_t v = ((64 - w) * u0 + w * u1 + 32) >> 6;
    return bc6h_finish(v, isign);
}

/* endpoint code whose unquantized value is nearest the desired unquantized
 * value (unquant(c) = 64*c + 32 on the interior, 0 for c = 0, and the clamp
 * code maps to the max) */
static int32_t bc6h_code_from_u(int32_t u, int isign)
{
    int32_t s = u < 0 ? -1 : 1;
    int32_t m = u < 0 ? -u : u;
    int32_t c, limit = isign ? 511 : 1023;

    if (m == 0)
        return 0;
    if (isign)
    {
        if (m >= 0x7fff)
            return s * limit;
    }
    else if (m >= 0xffff)
        return limit;
    /* interior lattice: u = 64*c + 32 => c = (u - 32) / 64 (rounded) */
    c = (m - 32 + 32) / 64;
    if (c < 1)
        c = 1;
    if (c > limit - 1)
        c = limit - 1;
    return s * c;
}

uint8_t *tdEncodeBPTCFloat(const float *rgba, int w, int h, int d, size_t row, size_t img, int isign)
{
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    size_t bytes = (size_t)bw * bh * (size_t)(d > 0 ? d : 1) * 16;
    uint8_t *dst = (uint8_t *)calloc(bytes ? bytes : 1, 1);

    if (!dst)
        return NULL;

    uint8_t *out = dst;
    for (int z = 0; z < (d > 0 ? d : 1); z++)
    {
        for (int by = 0; by < bh; by++)
            for (int bx = 0; bx < bw; bx++)
            {
                float tile[3][16];
                uint16_t src_bits[3][16];
                int32_t udes[3][16];
                int32_t ucm[3][2];
                int32_t ep[2][3];
                int32_t ep_ls[2][3];
                int idx[16];
                int x0 = bx * 4, y0 = by * 4;

                memset(tile, 0, sizeof(tile));
                for (int y = 0; y < 4 && y0 + y < h; y++)
                {
                    /* row/img are BYTE pitches (tdEncodeRGTC convention) */
                    const float *r = rgba + (size_t)z * (img / 4) + (size_t)(y0 + y) * (row / 4) +
                                     (size_t)x0 * 4;
                    for (int x = 0; x < 4 && x0 + x < w; x++)
                    {
                        int t = y * 4 + x;
                        tile[0][t] = r[x * 4];
                        tile[1][t] = r[x * 4 + 1];
                        tile[2][t] = r[x * 4 + 2];
                    }
                }

                /* BC6H interpolates in the unquantized domain, whose
                 * finishUnquantize output is the half BIT pattern - so range-fit
                 * there: convert each texel to half bits, invert the finish map
                 * back to the unquantized scale, and pick endpoint codes for the
                 * extremes (NaN / Inf are not representable: clamp to 0). */
                for (int ch = 0; ch < 3; ch++)
                {
                    int32_t umin = 0x7fffffff, umax = -0x7fffffff;
                    for (int t = 0; t < 16; t++)
                    {
                        float v = tile[ch][t];
                        uint16_t bits;
                        int32_t u;

                        if (!(v == v) || v > 65504.0f || v < -65504.0f)
                            v = 0.0f;
                        bits = tdFloatToHalf(v);

                        if (isign)
                        {
                            int s = (bits >> 15) & 1;
                            int32_t m = bits & 0x7fff;
                            /* finish: m' = (|u| * 31) >> 5 => |u| ~ m*32/31 */
                            u = (int32_t)(((int64_t)m * 32 + 15) / 31);
                            if (s)
                                u = -u;
                        }
                        else
                        {
                            /* finish: bits = (u * 31) / 64 => u ~ bits*64/31 */
                            u = (int32_t)(((int64_t)bits * 64 + 15) / 31);
                        }
                        udes[ch][t] = u;
                        if (u < umin)
                            umin = u;
                        if (u > umax)
                            umax = u;
                        src_bits[ch][t] = bits;
                    }
                    if (umax < umin)
                        umin = umax = 0;
                    ucm[ch][0] = umin;
                    ucm[ch][1] = umax;
                }

                /* Shared-index alignment: BC6H interpolates linearly in the
                 * unquantized domain with ONE index for all channels, so
                 * channels with different [umin,umax] spans want different
                 * weights for the same texel.  Express each texel as a fraction
                 * of its channel span, average over channels, then least-squares
                 * refit each channel (u = a + b*f) and snap a / a+b to endpoint
                 * codes.  First-order fix: the single shared index still caps
                 * quality (fine for online compression, which GL does not
                 * accuracy-guarantee). */
                {
                    double ff[16];
                    for (int t = 0; t < 16; t++)
                    {
                        double f = 0;
                        for (int ch = 0; ch < 3; ch++)
                        {
                            double span = (double)(ucm[ch][1] - ucm[ch][0]);
                            double fc = span > 0 ? (udes[ch][t] - ucm[ch][0]) / span : 0.0;
                            if (fc < 0) fc = 0;
                            if (fc > 1) fc = 1;
                            f += fc;
                        }
                        ff[t] = f / 3.0;
                    }
                    for (int ch = 0; ch < 3; ch++)
                    {
                        double sf = 0, sf2 = 0, su = 0, sfu = 0;
                        for (int t = 0; t < 16; t++)
                        {
                            sf += ff[t];
                            sf2 += ff[t] * ff[t];
                            su += udes[ch][t];
                            sfu += ff[t] * udes[ch][t];
                        }
                        double var = sf2 - sf * sf / 16.0;
                        double a, b;
                        if (var > 1e-9)
                        {
                            b = (sfu - sf * su / 16.0) / var;
                            a = (su - b * sf) / 16.0;
                        }
                        else
                        {
                            a = ucm[ch][0];
                            b = (double)(ucm[ch][1] - ucm[ch][0]);
                        }
                        if (b < 0)
                        {
                            a = ucm[ch][0];
                            b = ucm[ch][1] - ucm[ch][0];
                        }
                        ep_ls[0][ch] = bc6h_code_from_u((int32_t)(a + 0.5), isign);
                        ep_ls[1][ch] = bc6h_code_from_u((int32_t)(a + b + 0.5), isign);
                    }
                }

                /* choose per channel between range-fit (exact extremes) and the
                 * LS-aligned pair: a 2x2x2 search, scoring each combination by
                 * its optimal shared index per texel in bit space */
                for (int ch = 0; ch < 3; ch++)
                {
                    ep[0][ch] = bc6h_code_from_u(ucm[ch][0], isign);
                    ep[1][ch] = bc6h_code_from_u(ucm[ch][1], isign);
                }
                {
                    uint64_t best_err = ~0ull;
                    for (int mask = 0; mask < 8; mask++)
                    {
                        int32_t cand[2][3], uc[2][3];
                        uint32_t pal[3][16];
                        uint64_t terr = 0;

                        for (int ch = 0; ch < 3; ch++)
                            for (int e = 0; e < 2; e++)
                            {
                                int use_ls = (mask >> (ch + (e ? 3 : 0))) & 1;
                                cand[e][ch] = use_ls ? ep_ls[e][ch] : (e ? bc6h_code_from_u(ucm[ch][1], isign)
                                                                          : bc6h_code_from_u(ucm[ch][0], isign));
                                uc[e][ch] = bc6h_unquant(cand[e][ch], isign);
                            }
                        for (int ch = 0; ch < 3; ch++)
                            for (int t = 0; t < 16; t++)
                                pal[ch][t] = bc6h_palette(uc[0][ch], uc[1][ch], t, isign);
                        for (int t = 0; t < 16; t++)
                        {
                            uint32_t be = 0xffffffffu;
                            int lim = (t == 0) ? 8 : 16;
                            for (int i = 0; i < lim; i++)
                            {
                                uint32_t err = 0;
                                for (int ch = 0; ch < 3; ch++)
                                {
                                    uint32_t a = pal[ch][i], b = src_bits[ch][t];
                                    err += a > b ? a - b : b - a;
                                }
                                if (err < be)
                                    be = err;
                            }
                            terr += be;
                        }
                        if (terr < best_err)
                        {
                            best_err = terr;
                            for (int ch = 0; ch < 3; ch++)
                                for (int e = 0; e < 2; e++)
                                    ep[e][ch] = cand[e][ch];
                        }
                    }
                }

                /* exact palette (decoder arithmetic) + nearest shared index in
                 * bit space; texel 0 is the anchor: only 3 index bits stored */
                {
                    int32_t u[2][3];
                    uint32_t pal[3][16];
                    for (int e = 0; e < 2; e++)
                        for (int ch = 0; ch < 3; ch++)
                            u[e][ch] = bc6h_unquant(ep[e][ch], isign);
                    for (int ch = 0; ch < 3; ch++)
                        for (int t = 0; t < 16; t++)
                            pal[ch][t] = bc6h_palette(u[0][ch], u[1][ch], t, isign);
                    for (int t = 0; t < 16; t++)
                    {
                        uint32_t be = 0xffffffffu;
                        int bi = 0;
                        int lim = (t == 0) ? 8 : 16;
                        for (int i = 0; i < lim; i++)
                        {
                            uint32_t err = 0;
                            for (int ch = 0; ch < 3; ch++)
                            {
                                uint32_t a = pal[ch][i], b = src_bits[ch][t];
                                err += a > b ? a - b : b - a;
                            }
                            if (err < be)
                            {
                                be = err;
                                bi = i;
                            }
                        }
                        idx[t] = bi;
                    }
                }

                /* mode 10 (mode field 0x03), absolute endpoints at bits 5..64 */
                bptc_put(out, 0, 5, 0x03);
                for (int ch = 0; ch < 3; ch++)
                {
                    bptc_put(out, 5 + 10 * ch, 10, (uint32_t)(ep[0][ch] & 0x3ff));
                    bptc_put(out, 35 + 10 * ch, 10, (uint32_t)(ep[1][ch] & 0x3ff));
                }
                {
                    int bit = 65;
                    bptc_put(out, bit, 3, (uint32_t)idx[0]);
                    bit += 3;
                    for (int t = 1; t < 16; t++)
                    {
                        bptc_put(out, bit, 4, (uint32_t)idx[t]);
                        bit += 4;
                    }
                }
                out += 16;
            }
    }
    return dst;
}

uint8_t *tdEncodeBPTCUnorm(const uint8_t *rgba, int w, int h, int d, size_t row, size_t img)
{
    int bw = (w + 3) / 4, bh = (h + 3) / 4;
    size_t bytes = (size_t)bw * bh * (size_t)(d > 0 ? d : 1) * 16;
    uint8_t *dst = (uint8_t *)calloc(bytes ? bytes : 1, 1);

    if (!dst)
        return NULL;

    uint8_t *out = dst;
    for (int z = 0; z < (d > 0 ? d : 1); z++)
    {
        for (int by = 0; by < bh; by++)
            for (int bx = 0; bx < bw; bx++)
            {
                uint8_t tile[4][16];
                uint8_t ep[2][4];
                int pbit[2];
                int idx[16];
                int x0 = bx * 4, y0 = by * 4;

                memset(tile, 0, sizeof(tile));
                for (int y = 0; y < 4 && y0 + y < h; y++)
                {
                    const uint8_t *r = rgba + (size_t)z * img + (size_t)(y0 + y) * row + (size_t)x0 * 4;
                    for (int x = 0; x < 4 && x0 + x < w; x++)
                        for (int ch = 0; ch < 4; ch++)
                            tile[ch][y * 4 + x] = r[x * 4 + ch];
                }

                for (int ch = 0; ch < 4; ch++)
                {
                    int mn = tile[ch][0], mx = tile[ch][0];
                    for (int t = 1; t < 16; t++)
                    {
                        if (tile[ch][t] < mn) mn = tile[ch][t];
                        if (tile[ch][t] > mx) mx = tile[ch][t];
                    }
                    ep[0][ch] = (uint8_t)mn;
                    ep[1][ch] = (uint8_t)mx;
                }

                /* mode-6 endpoints are 7-bit + one P bit per ENDPOINT shared by
                 * all four channels: decoded = (v7 << 1) | p, so quantize each
                 * endpoint to the majority parity (worst case -1 on a channel) */
                for (int e = 0; e < 2; e++)
                {
                    int odd = 0;
                    for (int ch = 0; ch < 4; ch++)
                        if (ep[e][ch] & 1)
                            odd++;
                    pbit[e] = odd >= 2 ? 1 : 0;
                    for (int ch = 0; ch < 4; ch++)
                        if ((ep[e][ch] & 1) != pbit[e])
                            ep[e][ch] = (uint8_t)(ep[e][ch] ? ep[e][ch] - 1 : 1);
                }

                /* exact palette (decoder interpolation on the 8-bit endpoints)
                 * + nearest shared index; texel 0 is the anchor (3 bits) */
                for (int t = 0; t < 16; t++)
                {
                    int be = 1 << 30;
                    int bi = 0;
                    int lim = (t == 0) ? 8 : 16;
                    for (int i = 0; i < lim; i++)
                    {
                        int err = 0;
                        for (int ch = 0; ch < 4; ch++)
                        {
                            int p = ((64 - bptc_w4[i]) * ep[0][ch] + bptc_w4[i] * ep[1][ch] + 32) >> 6;
                            int e = p - tile[ch][t];
                            err += e * e;
                        }
                        if (err < be)
                        {
                            be = err;
                            bi = i;
                        }
                    }
                    idx[t] = bi;
                }

                /* mode 6: bit 6 set, bits 0..5 clear; endpoints 7-bit + P */
                bptc_put(out, 6, 1, 1);
                for (int ch = 0; ch < 4; ch++)
                {
                    bptc_put(out, 7 + 14 * ch, 7, (uint32_t)(ep[0][ch] >> 1));
                    bptc_put(out, 14 + 14 * ch, 7, (uint32_t)(ep[1][ch] >> 1));
                }
                bptc_put(out, 63, 1, (uint32_t)pbit[0]);
                bptc_put(out, 64, 1, (uint32_t)pbit[1]);
                {
                    int bit = 65;
                    bptc_put(out, bit, 3, (uint32_t)idx[0]);
                    bit += 3;
                    for (int t = 1; t < 16; t++)
                    {
                        bptc_put(out, bit, 4, (uint32_t)idx[t]);
                        bit += 4;
                    }
                }
                out += 16;
            }
    }
    return dst;
}

