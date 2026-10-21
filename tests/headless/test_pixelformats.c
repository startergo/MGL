/* test_pixelformats.c — D4 table-driven format sweep.
 *
 * Every GL 4.6 core required sized internal format (tables 8.12/8.13 + the
 * depth/stencil sized formats) is exercised with every (format,type) upload
 * that table 8.3 declares legal for its base format: the texel is uploaded
 * with reference component values, read back with glGetTexImage using the
 * SAME (format,type), and compared against the value the storage can
 * represent (double quantisation: through the client type, then through the
 * storage, then back through the client type).
 *
 * The compressed formats are checked for resolution (storage allocation +
 * INTERNAL_FORMAT readback) since generating valid compressed blocks needs an
 * encoder.
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "glcorearb.h"

/* ---- component kinds of a storage ------------------------------------- */
enum { K_UNORM8, K_UNORM16, K_SNORM8, K_SNORM16, K_F16, K_F32,
       K_RGB10A2U, K_RGB10A2I, K_RG11B10F, K_RGB9E5,
       K_I8, K_I16, K_I32, K_U8, K_U16, K_U32, K_D16, K_D24, K_D32F, K_S8 };

typedef struct {
    GLenum   internal;
    const char *name;
    int      kind;      /* K_* of the storage */
    int      ncomp;     /* components of the storage (0 = depth/stencil special) */
    GLenum   base;      /* GL_RED / GL_RG / GL_RGB / GL_RGBA / GL_DEPTH_COMPONENT /
                           GL_DEPTH_STENCIL / GL_STENCIL_INDEX */
} Case;

static const Case cases[] = {
    /* ---- normalized 8 ---- */
    { GL_R8,             "R8",             K_UNORM8,  1, GL_RED },
    { GL_RG8,            "RG8",            K_UNORM8,  2, GL_RG },
    { GL_RGB8,           "RGB8",           K_UNORM8,  3, GL_RGB },
    { GL_RGBA8,          "RGBA8",          K_UNORM8,  4, GL_RGBA },
    { GL_SRGB8,          "SRGB8",          K_UNORM8,  3, GL_RGB },
    { GL_SRGB8_ALPHA8,   "SRGB8_ALPHA8",   K_UNORM8,  4, GL_RGBA },
    /* ---- normalized 16 ---- */
    { GL_R16,            "R16",            K_UNORM16, 1, GL_RED },
    { GL_RG16,           "RG16",           K_UNORM16, 2, GL_RG },
    { GL_RGBA16,         "RGBA16",         K_UNORM16, 4, GL_RGBA },
    /* ---- snorm ---- */
    { GL_R8_SNORM,       "R8_SNORM",       K_SNORM8,  1, GL_RED },
    { GL_RG8_SNORM,      "RG8_SNORM",      K_SNORM8,  2, GL_RG },
    { GL_RGB8_SNORM,     "RGB8_SNORM",     K_SNORM8,  3, GL_RGB },
    { GL_RGBA8_SNORM,    "RGBA8_SNORM",    K_SNORM8,  4, GL_RGBA },
    { GL_R16_SNORM,      "R16_SNORM",      K_SNORM16, 1, GL_RED },
    { GL_RG16_SNORM,     "RG16_SNORM",     K_SNORM16, 2, GL_RG },
    { GL_RGB16_SNORM,    "RGB16_SNORM",    K_SNORM16, 3, GL_RGB },
    { GL_RGBA16_SNORM,   "RGBA16_SNORM",   K_SNORM16, 4, GL_RGBA },
    /* ---- packed fixed ---- */
    { GL_R3_G3_B2,       "R3_G3_B2",       K_UNORM8,  3, GL_RGB },
    { GL_RGB4,           "RGB4",           K_UNORM16, 3, GL_RGB },
    { GL_RGB5,           "RGB5",           K_UNORM16, 3, GL_RGB },
    { GL_RGB565,         "RGB565",         K_UNORM16, 3, GL_RGB },
    { GL_RGB10,          "RGB10",          K_UNORM16, 3, GL_RGB },
    { GL_RGB12,          "RGB12",          K_UNORM16, 3, GL_RGB },
    { GL_RGBA2,          "RGBA2",          K_UNORM16, 4, GL_RGBA },
    { GL_RGBA4,          "RGBA4",          K_UNORM16, 4, GL_RGBA },
    { GL_RGB5_A1,        "RGB5_A1",        K_UNORM16, 4, GL_RGBA },
    { GL_RGB10_A2,       "RGB10_A2",       K_RGB10A2U,4, GL_RGBA },
    { GL_RGBA12,         "RGBA12",         K_UNORM16, 4, GL_RGBA },
    /* ---- float ---- */
    { GL_R16F,           "R16F",           K_F16,     1, GL_RED },
    { GL_RG16F,          "RG16F",          K_F16,     2, GL_RG },
    { GL_RGB16F,         "RGB16F",         K_F16,     3, GL_RGB },
    { GL_RGBA16F,        "RGBA16F",        K_F16,     4, GL_RGBA },
    { GL_R32F,           "R32F",           K_F32,     1, GL_RED },
    { GL_RG32F,          "RG32F",          K_F32,     2, GL_RG },
    { GL_RGB32F,         "RGB32F",         K_F32,     3, GL_RGB },
    { GL_RGBA32F,        "RGBA32F",        K_F32,     4, GL_RGBA },
    { GL_R11F_G11F_B10F, "R11F_G11F_B10F", K_RG11B10F,3, GL_RGB },
    { GL_RGB9_E5,        "RGB9_E5",        K_RGB9E5,  3, GL_RGB },
    /* ---- integer ---- */
    { GL_R8I,            "R8I",            K_I8,      1, GL_RED_INTEGER },
    { GL_R16I,           "R16I",           K_I16,     1, GL_RED_INTEGER },
    { GL_R32I,           "R32I",           K_I32,     1, GL_RED_INTEGER },
    { GL_RG8I,           "RG8I",           K_I8,      2, GL_RG_INTEGER },
    { GL_RG16I,          "RG16I",          K_I16,     2, GL_RG_INTEGER },
    { GL_RG32I,          "RG32I",          K_I32,     2, GL_RG_INTEGER },
    { GL_RGB8I,          "RGB8I",          K_I8,      3, GL_RGB_INTEGER },
    { GL_RGB16I,         "RGB16I",         K_I16,     3, GL_RGB_INTEGER },
    { GL_RGB32I,         "RGB32I",         K_I32,     3, GL_RGB_INTEGER },
    { GL_RGBA8I,         "RGBA8I",         K_I8,      4, GL_RGBA_INTEGER },
    { GL_RGBA16I,        "RGBA16I",        K_I16,     4, GL_RGBA_INTEGER },
    { GL_RGBA32I,        "RGBA32I",        K_I32,     4, GL_RGBA_INTEGER },
    { GL_R8UI,           "R8UI",           K_U8,      1, GL_RED_INTEGER },
    { GL_R16UI,          "R16UI",          K_U16,     1, GL_RED_INTEGER },
    { GL_R32UI,          "R32UI",          K_U32,     1, GL_RED_INTEGER },
    { GL_RG8UI,          "RG8UI",          K_U8,      2, GL_RG_INTEGER },
    { GL_RG16UI,         "RG16UI",         K_U16,     2, GL_RG_INTEGER },
    { GL_RG32UI,         "RG32UI",         K_U32,     2, GL_RG_INTEGER },
    { GL_RGB8UI,         "RGB8UI",         K_U8,      3, GL_RGB_INTEGER },
    { GL_RGB16UI,        "RGB16UI",        K_U16,     3, GL_RGB_INTEGER },
    { GL_RGB32UI,        "RGB32UI",        K_U32,     3, GL_RGB_INTEGER },
    { GL_RGBA8UI,        "RGBA8UI",        K_U8,      4, GL_RGBA_INTEGER },
    { GL_RGBA16UI,       "RGBA16UI",       K_U16,     4, GL_RGBA_INTEGER },
    { GL_RGB10_A2UI,     "RGB10_A2UI",     K_RGB10A2I,4, GL_RGBA_INTEGER },
    { GL_RGBA32UI,       "RGBA32UI",       K_U32,     4, GL_RGBA_INTEGER },
    /* ---- depth / stencil ---- */
    { GL_DEPTH_COMPONENT16,  "DEPTH_COMPONENT16",  K_D16,   1, GL_DEPTH_COMPONENT },
    { GL_DEPTH_COMPONENT24,  "DEPTH_COMPONENT24",  K_D24,   1, GL_DEPTH_COMPONENT },
    { GL_DEPTH_COMPONENT32,  "DEPTH_COMPONENT32",  K_D24,   1, GL_DEPTH_COMPONENT },
    { GL_DEPTH_COMPONENT32F, "DEPTH_COMPONENT32F", K_D32F,  1, GL_DEPTH_COMPONENT },
    { GL_DEPTH24_STENCIL8,   "DEPTH24_STENCIL8",   K_D24,   2, GL_DEPTH_STENCIL },
    { GL_DEPTH32F_STENCIL8,  "DEPTH32F_STENCIL8",  K_D32F,  2, GL_DEPTH_STENCIL },
};

/* reference component values (float for normalised/float, int for integers) */
static const float ref_unorm[5] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
static const float ref_snorm[5] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
static const float ref_float[5] = { -1.5f, -0.25f, 0.5f, 2.0f, 0.0625f };

/* ---- the (format,type) matrix of table 8.3 ---------------------------- */
typedef struct { GLenum fmt; GLenum type; const char *tag; int ncomp; int is_int; int bits[4]; } FT;

/* the packed (table 8.3) client types */
static int ft_packed(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE_3_3_2: case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8: case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2: case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV: case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8: case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 1;
        default:
            return 0;
    }
}

/* ncomp: components of the FORMAT (not storage). bits[]: bits per component
 * (for the reference quantisation); -1 = float precision. */
static const FT fts[] = {
    /* non-packed: legal with every colour format of matching integer-ness */
    { GL_RED,  GL_UNSIGNED_BYTE,  "RED/UBYTE",      1, 0, {8,0,0,0} },
    { GL_RG,   GL_UNSIGNED_BYTE,  "RG/UBYTE",       2, 0, {8,8,0,0} },
    { GL_RGB,  GL_UNSIGNED_BYTE,  "RGB/UBYTE",      3, 0, {8,8,8,0} },
    { GL_RGBA, GL_UNSIGNED_BYTE,  "RGBA/UBYTE",     4, 0, {8,8,8,8} },
    { GL_BGRA, GL_UNSIGNED_BYTE,  "BGRA/UBYTE",     4, 0, {8,8,8,8} },
    { GL_BGR,  GL_UNSIGNED_BYTE,  "BGR/UBYTE",      3, 0, {8,8,8,0} },
    { GL_RED,  GL_BYTE,           "RED/BYTE",       1, 0, {8,0,0,0} },
    { GL_RGBA, GL_BYTE,           "RGBA/BYTE",      4, 0, {8,8,8,8} },
    { GL_RED,  GL_UNSIGNED_SHORT, "RED/USHORT",     1, 0, {16,0,0,0} },
    { GL_RGBA, GL_UNSIGNED_SHORT, "RGBA/USHORT",    4, 0, {16,16,16,16} },
    { GL_RGB,  GL_SHORT,          "RGB/SHORT",      3, 0, {16,16,16,0} },
    { GL_RGBA, GL_SHORT,          "RGBA/SHORT",     4, 0, {16,16,16,16} },
    { GL_RED,  GL_UNSIGNED_INT,   "RED/UINT",       1, 0, {32,0,0,0} },
    { GL_RGBA, GL_UNSIGNED_INT,   "RGBA/UINT",      4, 0, {32,32,32,32} },
    { GL_RGB,  GL_INT,            "RGB/INT",        3, 0, {32,32,32,0} },
    { GL_RED,  GL_INT,            "RED/INT",        1, 0, {32,0,0,0} },
    { GL_RED,  GL_HALF_FLOAT,     "RED/HALF",       1, 0, {-1,0,0,0} },
    { GL_RGB,  GL_HALF_FLOAT,     "RGB/HALF",       3, 0, {-1,-1,-1,0} },
    { GL_RGBA, GL_FLOAT,          "RGBA/FLOAT",     4, 0, {-1,-1,-1,-1} },
    { GL_RED,  GL_FLOAT,          "RED/FLOAT",      1, 0, {-1,0,0,0} },
    /* packed normalised */
    { GL_RGB,  GL_UNSIGNED_BYTE_3_3_2,        "RGB/332",      3, 0, {3,3,2,0} },
    { GL_RGB,  GL_UNSIGNED_BYTE_2_3_3_REV,    "RGB/332R",     3, 0, {3,3,2,0} },
    { GL_RGB,  GL_UNSIGNED_SHORT_5_6_5,       "RGB/565",      3, 0, {5,6,5,0} },
    { GL_RGB,  GL_UNSIGNED_SHORT_5_6_5_REV,   "RGB/565R",     3, 0, {5,6,5,0} },
    { GL_RGB,  GL_UNSIGNED_SHORT_4_4_4_4,     "RGB/4444",     3, 0, {4,4,4,0} },
    { GL_RGB,  GL_UNSIGNED_SHORT_4_4_4_4_REV, "RGB/4444R",    3, 0, {4,4,4,0} },
    { GL_RGB,  GL_UNSIGNED_SHORT_5_5_5_1,     "RGB/5551",     3, 0, {5,5,5,1} },
    { GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1,     "RGBA/5551",    4, 0, {5,5,5,1} },
    { GL_RGB,  GL_UNSIGNED_SHORT_1_5_5_5_REV, "RGB/1555R",    3, 0, {5,5,5,1} },
    { GL_RGBA, GL_UNSIGNED_SHORT_1_5_5_5_REV, "RGBA/1555R",   4, 0, {5,5,5,1} },
    { GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4,     "RGBA/4444",    4, 0, {4,4,4,4} },
    { GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4_REV, "RGBA/4444R",   4, 0, {4,4,4,4} },
    { GL_RGB,  GL_UNSIGNED_INT_8_8_8_8,       "RGB/8888",     3, 0, {8,8,8,0} },
    { GL_RGBA, GL_UNSIGNED_INT_8_8_8_8,       "RGBA/8888",    4, 0, {8,8,8,8} },
    { GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,   "BGRA/8888R",   4, 0, {8,8,8,8} },
    { GL_RGB,  GL_UNSIGNED_INT_8_8_8_8_REV,   "RGB/8888R",    3, 0, {8,8,8,0} },
    { GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV,   "RGBA/8888R",   4, 0, {8,8,8,8} },
    { GL_RGB,  GL_UNSIGNED_INT_10_10_10_2,    "RGB/1010102",  3, 0, {10,10,10,2} },
    { GL_RGBA, GL_UNSIGNED_INT_10_10_10_2,    "RGBA/1010102", 4, 0, {10,10,10,2} },
    { GL_RGB,  GL_UNSIGNED_INT_2_10_10_10_REV,"RGB/2101010R", 3, 0, {10,10,10,2} },
    { GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV,"RGBA/2101010R",4, 0, {10,10,10,2} },
    { GL_BGRA, GL_UNSIGNED_INT_2_10_10_10_REV,"BGRA/2101010R",4, 0, {10,10,10,2} },
    { GL_RGB,  GL_UNSIGNED_INT_10F_11F_11F_REV,"RGB/10f11f11f",3,0, {-2,-2,-3,0} },
    { GL_RGB,  GL_UNSIGNED_INT_5_9_9_9_REV,   "RGB/5999",     3, 0, {-2,-2,-2,0} },
    /* integer formats */
    { GL_RED_INTEGER,  GL_UNSIGNED_BYTE,  "REDi/UBYTE",   1, 1, {8,0,0,0} },
    { GL_RG_INTEGER,   GL_UNSIGNED_BYTE,  "RGi/UBYTE",    2, 1, {8,8,0,0} },
    { GL_RGB_INTEGER,  GL_UNSIGNED_BYTE,  "RGBi/UBYTE",   3, 1, {8,8,8,0} },
    { GL_RGBA_INTEGER, GL_UNSIGNED_BYTE,  "RGBAi/UBYTE",  4, 1, {8,8,8,8} },
    { GL_BGRA_INTEGER, GL_UNSIGNED_BYTE,  "BGRAi/UBYTE",  4, 1, {8,8,8,8} },
    { GL_RED_INTEGER,  GL_BYTE,           "REDi/BYTE",    1, 1, {8,0,0,0} },
    { GL_RGBA_INTEGER, GL_BYTE,           "RGBAi/BYTE",   4, 1, {8,8,8,8} },
    { GL_RED_INTEGER,  GL_UNSIGNED_SHORT, "REDi/USHORT",  1, 1, {16,0,0,0} },
    { GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, "RGBAi/USHORT", 4, 1, {16,16,16,16} },
    { GL_RGBA_INTEGER, GL_SHORT,          "RGBAi/SHORT",  4, 1, {16,16,16,16} },
    { GL_RED_INTEGER,  GL_UNSIGNED_INT,   "REDi/UINT",    1, 1, {32,0,0,0} },
    { GL_RGBA_INTEGER, GL_UNSIGNED_INT,   "RGBAi/UINT",   4, 1, {32,32,32,32} },
    { GL_RED_INTEGER,  GL_INT,            "REDi/INT",     1, 1, {32,0,0,0} },
    { GL_RGBA_INTEGER, GL_INT,            "RGBAi/INT",    4, 1, {32,32,32,32} },
    { GL_RGBA_INTEGER, GL_UNSIGNED_INT_8_8_8_8_REV,     "RGBAi/8888R",   4, 1, {8,8,8,8} },
    { GL_BGRA_INTEGER, GL_UNSIGNED_INT_8_8_8_8_REV,     "BGRAi/8888R",   4, 1, {8,8,8,8} },
    { GL_RGBA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV,  "RGBAi/2101010R",4, 1, {10,10,10,2} },
    { GL_BGRA_INTEGER, GL_UNSIGNED_INT_2_10_10_10_REV,  "BGRAi/2101010R",4, 1, {10,10,10,2} },
    /* depth / stencil */
    { GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, "D/USHORT", 1, 0, {16,0,0,0} },
    { GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,   "D/UINT",   1, 0, {24,0,0,0} },
    { GL_DEPTH_COMPONENT, GL_FLOAT,          "D/FLOAT",  1, 0, {-1,0,0,0} },
    { GL_DEPTH_COMPONENT, GL_HALF_FLOAT,     "D/HALF",   1, 0, {-1,0,0,0} },
    { GL_DEPTH_STENCIL,   GL_UNSIGNED_INT_24_8, "DS/248", 2, 0, {24,8,0,0} },
    { GL_DEPTH_STENCIL,   GL_FLOAT_32_UNSIGNED_INT_24_8_REV, "DS/32f8", 2, 0, {-1,8,0,0} },
    { GL_STENCIL_INDEX,   GL_UNSIGNED_BYTE,  "S/UBYTE",  1, 1, {8,0,0,0} },
};

static int n_fts = (int)(sizeof fts / sizeof fts[0]);

/* ---- encoding / decoding of the reference values into (format,type) --- */

/* the value the STORAGE can represent, as float (norm/float kinds) */
static float storage_quant(int kind, int comp, float v, int *exact_int)
{
    *exact_int = 0;
    switch (kind)
    {
        case K_UNORM8:  *exact_int = 1; return floorf(fminf(1.0f, fmaxf(0.0f, v)) * 255.0f + 0.5f) / 255.0f;
        case K_UNORM16: *exact_int = 1; return floorf(fminf(1.0f, fmaxf(0.0f, v)) * 65535.0f + 0.5f) / 65535.0f;
        case K_SNORM8:  *exact_int = 1; { float c = fmaxf(-1.0f, fminf(1.0f, v));
                                          float q = roundf(c * 127.0f) / 127.0f;
                                          if (c == 1.0f) q = 1.0f;
                                          return q; }
        case K_SNORM16: *exact_int = 1; { float c = fmaxf(-1.0f, fminf(1.0f, v));
                                          float q = roundf(c * 32767.0f) / 32767.0f;
                                          if (c == 1.0f) q = 1.0f;
                                          return q; }
        case K_F16:     return v;      /* half round-trip handled by tolerance */
        case K_F32:     return v;
        case K_RGB10A2U: *exact_int = 1; { int bits = (comp == 3) ? 2 : 10;
                                           int maxv = (1 << bits) - 1;
                                           return floorf(fminf(1.0f, fmaxf(0.0f, v)) * maxv + 0.5f) / maxv; }
        case K_RG11B10F: return v;     /* float with reduced mantissa: tolerance */
        case K_RGB9E5:   return v;
        default:        return v;      /* integer kinds use the int path */
    }
}

/* expected channel value after upload+readback, for normalised/float kinds */
static float expected_channel(const Case *c, const FT *ft, int comp, int chan)
{
    /* chan is the channel INDEX in the FORMAT (0..ncomp-1); the storage
     * component it lands in depends on BGR/REV orders being consistent —
     * both upload and readback use the same (format,type), so the channel
     * order cancels out: chan of the format maps to storage component chan
     * for non-REV, and the readback maps it back identically. */
    float v;
    int exact = 0;

    if (chan >= c->ncomp && c->ncomp > 0 && c->base != GL_DEPTH_STENCIL)
    {
        /* a channel the storage does not have: 0, alpha reads as 1 */
        if (chan == 3)
            return 1.0f;
        return 0.0f;
    }
    if (ft->bits[chan] > 0 && c->kind >= K_UNORM8 && c->kind <= K_RGB9E5)
        v = ref_unorm[(chan + comp) % 5];
    else if (ft->bits[chan] > 0 && c->kind >= K_I8 && c->kind <= K_U32)
        v = ref_unorm[(chan + comp) % 5];   /* int path fills its own values */
    else if (c->kind == K_SNORM8 || c->kind == K_SNORM16)
        v = ref_snorm[(chan + comp) % 5];
    else
        v = ref_float[(chan + comp) % 5];
    return storage_quant(c->kind, comp, v, &exact);
}

static int tol_of(int kind)
{
    switch (kind)
    {
        case K_UNORM8:
        case K_SNORM8:   return 3;               /* a couple of ULPs of a byte */
        case K_UNORM16:
        case K_SNORM16: return 700;              /* ULPs of 1/65535 */
        case K_F16:      return 0.02f * 1000;    /* half mantissa */
        case K_RG11B10F: return 0.05f * 1000;    /* 10f has a 5-bit mantissa */
        case K_RGB9E5:   return 0.01f * 1000;
        default:         return 1.0f * 1000;
    }
}

/* encode the reference texels into client memory of (ft); returns bytes/texel */
static int encode(const Case *c, const FT *ft, uint32_t out[4], int comp)
{
    memset(out, 0, 16);

    for (int chan = 0; chan < ft->ncomp; chan++)
    {
        /* reference value of this channel (respecting the format's integer-ness) */
        float fv;
        int iv = 0;

        if (ft->is_int || c->kind >= K_I8)
        {
            int bits = (ft->bits[chan] > 0) ? ft->bits[chan] : 32;
            int maxv = (ft->is_int && (ft->type == GL_BYTE || ft->type == GL_SHORT ||
                                       ft->type == GL_INT)) ? (1 << (bits - 1)) - 1
                                                           : (1 << bits) - 1;
            int ival[] = { 0, 1, (int)(maxv / 2), maxv - 1, maxv };
            int neg[]  = { 0, -1, -(maxv / 2), -(maxv / 2) + 1, -maxv };

            if (ft->is_int && (ft->type == GL_BYTE || ft->type == GL_SHORT || ft->type == GL_INT))
                iv = neg[(chan + comp) % 5];
            else
                iv = ival[(chan + comp) % 5];
            fv = 0.0f;
        }
        else if (ft->type == GL_BYTE || ft->type == GL_SHORT || ft->type == GL_INT)
            fv = ref_snorm[(chan + comp) % 5];
        else if (ft->bits[chan] == -1 || ft->bits[chan] < 0)
            fv = ref_float[(chan + comp) % 5];
        else
            fv = ref_unorm[(chan + comp) % 5];

        switch (ft->type)
        {
            case GL_UNSIGNED_BYTE: case GL_BYTE:
                if (ft->is_int || c->kind >= K_I8)
                    ((int8_t *)out)[chan] = (int8_t)iv;
                else if (ft->type == GL_BYTE)
                    ((int8_t *)out)[chan] = (int8_t)lroundf(fmaxf(-1.0f, fv) * 127.0f);
                else
                    ((uint8_t *)out)[chan] = (uint8_t)lroundf(fminf(1.0f, fmaxf(0.0f, fv)) * 255.0f);
                break;
            case GL_UNSIGNED_SHORT: case GL_SHORT:
                if (ft->is_int || c->kind >= K_I8)
                    ((int16_t *)out)[chan] = (int16_t)iv;
                else if (ft->type == GL_SHORT)
                    ((int16_t *)out)[chan] = (int16_t)lroundf(fmaxf(-1.0f, fv) * 32767.0f);
                else
                    ((uint16_t *)out)[chan] = (uint16_t)lroundf(fminf(1.0f, fmaxf(0.0f, fv)) * 65535.0f);
                break;
            case GL_UNSIGNED_INT: case GL_INT:
                if (ft->is_int || c->kind >= K_I8)
                    out[chan] = (uint32_t)iv;
                else if (ft->type == GL_INT)
                    out[chan] = (uint32_t)(int64_t)llround((double)fmaxf(-1.0f, fv) * 2147483647.0);
                else
                    out[chan] = (uint32_t)llround((double)fminf(1.0f, fmaxf(0.0f, fv)) * 4294967295.0);
                break;
            case GL_HALF_FLOAT: {
                /* encode float->half */
                union { float f; uint32_t u; } v32;
                v32.f = fv;
                {
                    int sign = (v32.u >> 16) & 0x8000;
                    int exp = (int)((v32.u >> 23) & 0xff) - 127 + 15;
                    int man = (v32.u >> 13) & 0x3ff;
                    uint16_t h;

                    if (exp <= 0) { h = (uint16_t)(sign | 0); }
                    else if (exp >= 31) { h = (uint16_t)(sign | 0x7c00); }
                    else h = (uint16_t)(sign | (exp << 10) | man);
                    ((uint16_t *)out)[chan] = h;
                }
                break;
            }
            case GL_FLOAT:
                ((float *)out)[chan] = fv;
                break;

            /* packed: encode from the (already storage-quantisable) value */
            case GL_UNSIGNED_BYTE_3_3_2: {
                int r = (int)lroundf(fminf(1, fmaxf(0, fv)) * 7);
                ((uint8_t *)out)[0] = (uint8_t)((r << 5) | (r << 2) | (r >> 1));
                break; }
            case GL_UNSIGNED_BYTE_2_3_3_REV: {
                int r = (int)lroundf(fminf(1, fmaxf(0, fv)) * 7);
                ((uint8_t *)out)[0] = (uint8_t)((r) | (r << 3) | (r << 6));
                break; }
            case GL_UNSIGNED_SHORT_5_6_5: {
                int bits[] = { 5, 6, 5, 0 }, shift[] = { 11, 5, 0, 0 };
                uint16_t w = 0;

                for (int k = 0; k < 3; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int maxv = (1 << bits[k]) - 1;
                    w |= (uint16_t)(lroundf(fminf(1, fmaxf(0, v2)) * maxv) << shift[k]);
                }
                *(uint16_t *)out = w;
                return 2; }
            case GL_UNSIGNED_SHORT_5_6_5_REV: {
                int bits[] = { 5, 6, 5, 0 }, shift[] = { 0, 5, 11, 0 };
                uint16_t w = 0;

                for (int k = 0; k < 3; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int maxv = (1 << bits[k]) - 1;
                    w |= (uint16_t)(lroundf(fminf(1, fmaxf(0, v2)) * maxv) << shift[k]);
                }
                *(uint16_t *)out = w;
                return 2; }
            case GL_UNSIGNED_SHORT_4_4_4_4:
            case GL_UNSIGNED_SHORT_4_4_4_4_REV: {
                uint16_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int sh = (ft->type == GL_UNSIGNED_SHORT_4_4_4_4) ? (12 - 4 * k) : (4 * k);

                    w |= (uint16_t)(lroundf(fminf(1, fmaxf(0, v2)) * 15) << sh);
                }
                *(uint16_t *)out = w;
                return 2; }
            case GL_UNSIGNED_SHORT_5_5_5_1: {
                uint16_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int sh = (k == 3) ? 0 : (11 - 5 * k);
                    int maxv = (k == 3) ? 1 : 31;

                    w |= (uint16_t)(lroundf(fminf(1, fmaxf(0, v2)) * maxv) << sh);
                }
                *(uint16_t *)out = w;
                return 2; }
            case GL_UNSIGNED_SHORT_1_5_5_5_REV: {
                uint16_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    /* bit 15 = A, then B, G, R down to bit 0 */
                    int sh = (k == 3) ? 15 : (5 * k);

                    w |= (uint16_t)(lroundf(fminf(1, fmaxf(0, v2)) * 31) << sh);
                }
                *(uint16_t *)out = w;
                return 2; }
            case GL_UNSIGNED_INT_8_8_8_8:
            case GL_UNSIGNED_INT_8_8_8_8_REV: {
                uint32_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    /* non-REV lists components from the most significant byte; a
                     * 3-component format leaves the LOW byte unused (REV: low first) */
                    int sh = (ft->type == GL_UNSIGNED_INT_8_8_8_8) ? (8 * (4 - 1 - k))
                                                                   : (8 * k);

                    w |= (uint32_t)(lroundf(fminf(1, fmaxf(0, v2)) * 255) << sh);
                }
                *out = w;
                return 4; }
            case GL_UNSIGNED_INT_10_10_10_2: {
                uint32_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int sh = (k == 3) ? 0 : (22 - 10 * k);
                    int maxv = (k == 3) ? 3 : 1023;

                    w |= (uint32_t)(lroundf(fminf(1, fmaxf(0, v2)) * maxv) << sh);
                }
                *out = w;
                return 4; }
            case GL_UNSIGNED_INT_2_10_10_10_REV: {
                uint32_t w = 0;

                for (int k = 0; k < ft->ncomp; k++)
                {
                    float v2 = ref_unorm[(k + comp) % 5];
                    int sh = (k == 3) ? 30 : (10 * k);
                    int maxv = (k == 3) ? 3 : 1023;

                    w |= (uint32_t)(lroundf(fminf(1, fmaxf(0, v2)) * maxv) << sh);
                }
                *out = w;
                return 4; }
            case GL_UNSIGNED_INT_10F_11F_11F_REV: {
                /* encode ref per channel into 11f/11f/10f (clamped >= 0):
                 * 11f = 5-bit exponent (bias 15) + 6-bit mantissa, implicit 1 */
                float vr = fmaxf(0.0f, ref_float[(0 + comp) % 5]);
                float vg = fmaxf(0.0f, ref_float[(1 + comp) % 5]);
                float vb = fmaxf(0.0f, ref_float[(2 + comp) % 5]);
                int er, eg, eb;
                uint32_t r, g, b;

                frexpf(vr, &er); frexpf(vg, &eg); frexpf(vb, &eb);
                r = (vr == 0) ? 0
                    : (((uint32_t)(er - 1 + 15) << 6) |
                       ((uint32_t)llround((double)vr / ldexp(1.0, er - 1) * 64.0 - 64.0) & 63u));
                g = (vg == 0) ? 0
                    : (((uint32_t)(eg - 1 + 15) << 6) |
                       ((uint32_t)llround((double)vg / ldexp(1.0, eg - 1) * 64.0 - 64.0) & 63u));
                b = (vb == 0) ? 0
                    : (((uint32_t)(eb - 1 + 15) << 5) |
                       ((uint32_t)llround((double)vb / ldexp(1.0, eb - 1) * 32.0 - 32.0) & 31u));
                *out = r | (g << 11) | (b << 22);
                (void)fv;
                return 4; }
            case GL_UNSIGNED_INT_5_9_9_9_REV: {
                /* shared exponent: value = mantissa * 2^(e-15-9), no implicit bit */
                float vs[3];
                float maxc = 0.0f;
                int e = 0;
                uint32_t m[3] = { 0, 0, 0 };

                for (int k = 0; k < 3; k++)
                {
                    vs[k] = fmaxf(0.0f, ref_float[(k + comp) % 5]);
                    if (vs[k] > maxc) maxc = vs[k];
                }
                if (maxc > 0.0f)
                {
                    float sc;

                    frexpf(maxc, &e);          /* maxc = m * 2^e */
                    e = e - 1 + 15;            /* biased */
                    sc = ldexpf(1.0f, e - 15 - 9);
                    for (int k = 0; k < 3; k++)
                        m[k] = (uint32_t)lroundf(vs[k] / sc);
                    /* rounding may push a mantissa to 512: bump the shared exponent */
                    if (m[0] >= 512 || m[1] >= 512 || m[2] >= 512)
                    {
                        e++;
                        sc = ldexpf(1.0f, e - 15 - 9);
                        for (int k = 0; k < 3; k++)
                            m[k] = (uint32_t)lroundf(vs[k] / sc);
                    }
                }
                *out = m[0] | (m[1] << 9) | (m[2] << 18) | ((uint32_t)e << 27);
                (void)fv;
                return 4; }
            case GL_UNSIGNED_INT_24_8: {
                float d = (comp == 0) ? 0.0f : (comp == 1) ? 0.25f : 0.5f;
                uint32_t dv = (uint32_t)lroundf(d * 16777215.0f);

                *out = (dv << 8) | 0x5a;
                return 4; }
            case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: {
                float d = (comp == 0) ? 0.0f : (comp == 1) ? 0.25f : 0.5f;

                ((float *)out)[0] = d;
                ((uint32_t *)out)[1] = 0x5a;
                return 8; }
            default:
                return 0;
        }
    }

    switch (ft->type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE:      return ft->ncomp;
        case GL_UNSIGNED_SHORT: case GL_SHORT:
        case GL_HALF_FLOAT:                       return ft->ncomp * 2;
        case GL_UNSIGNED_INT: case GL_INT:
        case GL_FLOAT:                            return ft->ncomp * 4;
        default:                                  return 0;
    }
}

/* decode one texel of (ft) back to per-channel floats / ints */
static void decode(const FT *ft, const void *texel, float out_f[4], int out_i[4])
{
    const uint8_t *b = (const uint8_t *)texel;
    const uint16_t *h = (const uint16_t *)texel;
    const uint32_t *w = (const uint32_t *)texel;

    for (int i = 0; i < 4; i++) { out_f[i] = 0.0f; out_i[i] = 0; }
    switch (ft->type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE:
            for (int i = 0; i < ft->ncomp; i++)
                out_i[i] = (ft->type == GL_BYTE) ? (int8_t)b[i] : b[i];
            return;
        case GL_UNSIGNED_SHORT: case GL_SHORT:
            for (int i = 0; i < ft->ncomp; i++)
                out_i[i] = (ft->type == GL_SHORT) ? (int16_t)h[i] : h[i];
            return;
        case GL_UNSIGNED_INT: case GL_INT:
            for (int i = 0; i < ft->ncomp; i++)
                out_i[i] = (ft->type == GL_INT) ? (int32_t)w[i] : (int)w[i];
            return;
        case GL_FLOAT:
            for (int i = 0; i < ft->ncomp; i++) out_f[i] = ((const float *)texel)[i];
            return;
        case GL_HALF_FLOAT: {
            for (int i = 0; i < ft->ncomp; i++)
            {
                uint32_t v = h[i];
                int sign = (v >> 15) & 1, exp = (v >> 10) & 0x1f, man = v & 0x3ff;
                union { float f; uint32_t u; } c;

                if (!exp && !man)      c.u = (uint32_t)sign << 31;
                else if (!exp)         { c.u = ((uint32_t)sign << 31) | ((127 - 14 - __builtin_clz(man) + 1) << 23) | ((man << (9 + __builtin_clz(man) + 1)) & 0x7fffff); }
                else if (exp == 31)    c.u = ((uint32_t)sign << 31) | 0x7f800000u | (man << 13);
                else                   c.u = ((uint32_t)sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
                out_f[i] = c.f;
            }
            return; }
        case GL_UNSIGNED_BYTE_3_3_2: {
            out_i[0] = (b[0] >> 5) & 7; out_i[1] = (b[0] >> 2) & 7; out_i[2] = b[0] & 3;
            return; }
        case GL_UNSIGNED_BYTE_2_3_3_REV: {
            out_i[0] = b[0] & 3; out_i[1] = (b[0] >> 2) & 7; out_i[2] = (b[0] >> 5) & 7;
            return; }
        case GL_UNSIGNED_SHORT_5_6_5: {
            out_i[0] = (h[0] >> 11) & 31; out_i[1] = (h[0] >> 5) & 63; out_i[2] = h[0] & 31;
            return; }
        case GL_UNSIGNED_SHORT_5_6_5_REV: {
            out_i[0] = h[0] & 31; out_i[1] = (h[0] >> 5) & 63; out_i[2] = (h[0] >> 11) & 31;
            return; }
        case GL_UNSIGNED_SHORT_4_4_4_4: {
            for (int i = 0; i < 4; i++) out_i[i] = (h[0] >> (12 - 4 * i)) & 15;
            return; }
        case GL_UNSIGNED_SHORT_4_4_4_4_REV: {
            for (int i = 0; i < 4; i++) out_i[i] = (h[0] >> (4 * i)) & 15;
            return; }
        case GL_UNSIGNED_SHORT_5_5_5_1: {
            out_i[0] = (h[0] >> 11) & 31; out_i[1] = (h[0] >> 6) & 31;
            out_i[2] = (h[0] >> 1) & 31; out_i[3] = h[0] & 1;
            return; }
        case GL_UNSIGNED_SHORT_1_5_5_5_REV: {
            out_i[0] = h[0] & 31; out_i[1] = (h[0] >> 5) & 31;
            out_i[2] = (h[0] >> 10) & 31; out_i[3] = (h[0] >> 15) & 1;
            return; }
        case GL_UNSIGNED_INT_8_8_8_8: {
            for (int i = 0; i < 4; i++) out_i[i] = (int)((w[0] >> (8 * (3 - i))) & 255);
            return; }
        case GL_UNSIGNED_INT_8_8_8_8_REV: {
            for (int i = 0; i < 4; i++) out_i[i] = (int)((w[0] >> (8 * i)) & 255);
            return; }
        case GL_UNSIGNED_INT_10_10_10_2: {
            out_i[0] = (w[0] >> 22) & 1023; out_i[1] = (w[0] >> 12) & 1023;
            out_i[2] = (w[0] >> 2) & 1023; out_i[3] = w[0] & 3;
            return; }
        case GL_UNSIGNED_INT_2_10_10_10_REV: {
            out_i[0] = w[0] & 1023; out_i[1] = (w[0] >> 10) & 1023;
            out_i[2] = (w[0] >> 20) & 1023; out_i[3] = (w[0] >> 30) & 3;
            return; }
        case GL_UNSIGNED_INT_10F_11F_11F_REV: {
            uint32_t r11 = w[0] & 0x7ff, g11 = (w[0] >> 11) & 0x7ff, b10 = (w[0] >> 22) & 0x3ff;
            int ex; uint32_t m;

            /* 11f: 5-bit exponent (bias 15) + 6-bit mantissa with an implicit
             * leading 1; 10f: 5-bit exponent + 5-bit mantissa */
            ex = (int)(r11 >> 6); m = r11 & 0x3f;
            out_f[0] = ex ? ldexpf((float)(64u | m), ex - 15 - 6) : ldexpf((float)m, -24);
            ex = (int)(g11 >> 6); m = g11 & 0x3f;
            out_f[1] = ex ? ldexpf((float)(64u | m), ex - 15 - 6) : ldexpf((float)m, -24);
            ex = (int)(b10 >> 5); m = b10 & 0x1f;
            out_f[2] = ex ? ldexpf((float)(32u | m), ex - 15 - 5) : ldexpf((float)m, -24);
            return; }
        case GL_UNSIGNED_INT_5_9_9_9_REV: {
            /* shared exponent: value = mantissa * 2^(e-15-9), no implicit bit */
            uint32_t exp5 = w[0] >> 27;
            int m[3] = { (int)(w[0] & 511), (int)((w[0] >> 9) & 511), (int)((w[0] >> 18) & 511) };

            for (int i = 0; i < 3; i++)
                out_f[i] = ldexpf((float)m[i], (int)exp5 - 15 - 9);
            return; }
        case GL_UNSIGNED_INT_24_8: {
            out_i[0] = (int)(w[0] >> 8); out_i[1] = (int)(w[0] & 255);
            return; }
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: {
            out_f[0] = ((const float *)texel)[0];
            out_i[1] = (int)(w[1] & 255);
            return; }
        default:
            return;
    }
}

/* how many texels to use */
#define NTEX 3

static int fail_count, total_count;
static char failures[256][96];
static int n_failures;

static void note_fail(const char *internal, const char *ft, const char *what)
{
    if (n_failures < 240)
        snprintf(failures[n_failures++], sizeof failures[0], "%s / %s: %s", internal, ft, what);
    fail_count++;
}

TEST(RequiredInternalFormatsSweep)
{
    GLuint tex = 0;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++)
    {
        const Case *c = &cases[ci];

        for (int fi = 0; fi < n_fts; fi++)
        {
            const FT *ft = &fts[fi];
            uint8_t up[NTEX * 16], down[NTEX * 16];
            int bpt;

            /* legality: base format class and integer-ness must match */
            {
                int base_color = (c->base == GL_RED || c->base == GL_RG ||
                                  c->base == GL_RGB || c->base == GL_RGBA);
                int ft_color = (ft->fmt == GL_RED || ft->fmt == GL_RG || ft->fmt == GL_RGB ||
                                ft->fmt == GL_RGBA || ft->fmt == GL_BGRA || ft->fmt == GL_BGR);
                int base_int = (c->base == GL_RED_INTEGER || c->base == GL_RG_INTEGER ||
                                c->base == GL_RGB_INTEGER || c->base == GL_RGBA_INTEGER);
                int ft_int = (ft->fmt == GL_RED_INTEGER || ft->fmt == GL_RG_INTEGER ||
                              ft->fmt == GL_RGB_INTEGER || ft->fmt == GL_RGBA_INTEGER ||
                              ft->fmt == GL_BGRA_INTEGER);

                if (c->base == GL_DEPTH_COMPONENT &&
                    !(ft->fmt == GL_DEPTH_COMPONENT && !ft->is_int))
                    continue;
                if (c->base == GL_DEPTH_STENCIL &&
                    !(ft->fmt == GL_DEPTH_STENCIL && !ft->is_int))
                    continue;
                if (c->base == GL_STENCIL_INDEX)
                {
                    if (!(ft->fmt == GL_STENCIL_INDEX && ft->is_int)) continue;
                }
                else if (base_color)
                {
                    if (!ft_color && !ft_int) continue;
                    if (base_int != ft_int) continue;
                }
                else continue;

                /* GL 4.6 table 8.3: a packed client type accepts only its formats -
                 * RGB / RGB_INTEGER for the 3-component ones, RGBA / BGRA and their
                 * integer twins for the 4-component ones, RGB for the float-packed,
                 * DEPTH_STENCIL for the depth-stencil ones (matches the CTS rule table
                 * KHR-GL46.packed_pixels enforces) */
                if (ft_packed(ft->type))
                {
                    int rgb = (ft->fmt == GL_RGB || ft->fmt == GL_RGB_INTEGER);
                    int rgba4 = (ft->fmt == GL_RGBA || ft->fmt == GL_BGRA ||
                                 ft->fmt == GL_RGBA_INTEGER || ft->fmt == GL_BGRA_INTEGER);
                    switch (ft->type)
                    {
                        case GL_UNSIGNED_BYTE_3_3_2:
                        case GL_UNSIGNED_BYTE_2_3_3_REV:
                        case GL_UNSIGNED_SHORT_5_6_5:
                        case GL_UNSIGNED_SHORT_5_6_5_REV:
                            if (!rgb) continue;
                            break;
                        case GL_UNSIGNED_INT_10F_11F_11F_REV:
                        case GL_UNSIGNED_INT_5_9_9_9_REV:
                            if (ft->fmt != GL_RGB) continue;
                            break;
                        case GL_UNSIGNED_INT_24_8:
                        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
                            if (ft->fmt != GL_DEPTH_STENCIL) continue;
                            break;
                        default:
                            if (!rgba4) continue;
                            break;
                    }
                }
            }

            memset(up, 0, sizeof up);
            memset(down, 0, sizeof down);

            /* encode the reference texels */
            for (int t = 0; t < NTEX; t++)
            {
                uint32_t word[4] = { 0, 0, 0, 0 };
                int used = encode(c, ft, word, t);

                if (!used) { bpt = 0; break; }
                memcpy(up + (size_t)t * used, word, (size_t)used);
                bpt = used;
            }
            if (!bpt)
                continue;

            while (glGetError() != GL_NO_ERROR)
                ;
            glTexImage2D(GL_TEXTURE_2D, 0, (GLint)c->internal, NTEX, 1, 0,
                         ft->fmt, ft->type, up);
            {
                GLenum e = glGetError();

                if (e != GL_NO_ERROR)
                {
                    total_count++;
                    note_fail(c->name, ft->tag, "TexImage error");
                    continue;
                }
            }
            glGetTexImage(GL_TEXTURE_2D, 0, ft->fmt, ft->type, down);
            if (glGetError() != GL_NO_ERROR)
            {
                total_count++;
                note_fail(c->name, ft->tag, "GetTexImage error");
                continue;
            }

            total_count++;
            /* verify every texel/channel against the reference, through the
             * storage's quantisation and back through the client type's grid */
            for (int t = 0; t < NTEX; t++)
            {
                float got_f[4];
                int got_i[4];

                decode(ft, down + (size_t)t * bpt, got_f, got_i);

                for (int chan = 0; chan < ft->ncomp; chan++)
                {
                    const char *kindname = c->name;
                    char what[128];
                    /* the format's channel -> storage component (BGRA/BGR swap) */
                    int smap[4] = { 0, 1, 2, 3 };

                    if (ft->fmt == GL_BGRA || ft->fmt == GL_BGR ||
                        ft->fmt == GL_BGRA_INTEGER)
                        smap[0] = 2, smap[1] = 1, smap[2] = 0;
                    if (c->base == GL_DEPTH_COMPONENT || c->base == GL_DEPTH_STENCIL ||
                        c->base == GL_STENCIL_INDEX)
                        smap[0] = 0, smap[1] = 1, smap[2] = 2, smap[3] = 3;
                    {
                    int chan_stored = (smap[chan] < c->ncomp || c->base == GL_DEPTH_STENCIL ||
                                       c->base == GL_STENCIL_INDEX);
                    int bits = ft->bits[chan];
                    int float_client = (bits < 0);
                    int signed_client = (ft->type == GL_BYTE || ft->type == GL_SHORT ||
                                         ft->type == GL_INT);

                    if (!chan_stored)
                    {
                        /* a channel the storage does not keep: alpha of a 3-component
                         * storage reads back as its maximum, everything else as 0 */
                        int is_alpha = (chan == 3);
                        int want;

                        if (float_client)
                        {
                            if (fabsf(got_f[chan] - (is_alpha ? 1.0f : 0.0f)) > 1e-6f)
                            {
                                snprintf(what, sizeof what,
                                         "chan %d unused: got %g want %g", chan,
                                         got_f[chan], is_alpha ? 1.0f : 0.0f);
                                note_fail(kindname, ft->tag, what);
                                goto next_ft;
                            }
                        }
                        else
                        {
                            uint32_t mx = signed_client
                                ? (bits >= 32 ? 2147483647u : ((1u << (bits - 1)) - 1u))
                                : (bits >= 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u));

                            want = is_alpha ? (int)mx : 0;
                            if (got_i[chan] != want)
                            {
                                snprintf(what, sizeof what,
                                         "chan %d unused: got %d want %d", chan,
                                         got_i[chan], want);
                                note_fail(kindname, ft->tag, what);
                                goto next_ft;
                            }
                        }
                        continue;
                    }

                    /* stencil is its own tiny case: the encoded 0x5a must survive */
                    if (c->kind == K_S8)
                    {
                        if (got_i[chan] != 0x5a)
                        {
                            snprintf(what, sizeof what, "stencil: got %d want 90",
                                     got_i[chan]);
                            note_fail(kindname, ft->tag, what);
                            goto next_ft;
                        }
                        continue;
                    }

                    /* ---- expected value: client -> storage -> client ----
                     * the encoder (above) writes: unsigned codes = round(ref_unorm*max),
                     * signed codes = round(ref_snorm*(2^(b-1)-1)), floats = ref_float.
                     * The storage clamps (unorm/depth: [0,1], snorm: [-1,1]) and
                     * quantises; the readback maps the stored value back into the
                     * client's domain. */
                    {
                        uint32_t maxc = float_client ? 0 :
                                        (bits >= 32 ? 0xFFFFFFFFu : ((1u << bits) - 1u));
                        uint32_t smaxc = float_client ? 0 :
                                         (bits >= 32 ? 2147483647u : ((1u << (bits - 1)) - 1u));
                        float v;      /* semantic value entering the storage */
                        float stored; /* the storage's value of it */
                        float out;    /* value presented to the readback */

                        if (ft->type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                            ft->type == GL_UNSIGNED_INT_5_9_9_9_REV)
                            v = fmaxf(0.0f, ref_float[(chan + t) % 5]);
                        else if (float_client)
                            v = ref_float[(chan + t) % 5];
                        else if (signed_client)
                            v = ref_snorm[(chan + t) % 5];
                        else
                            v = ref_unorm[(chan + t) % 5];

                        /* clamp into the storage's range: unorm/depth clamp to
                         * [0,1], snorm to [-1,1]; the packed float formats
                         * (11f/10f, shared-exponent) have no sign bit, so they
                         * clamp to [0,inf); only 16f/32f keep negatives */
                        if (c->kind == K_SNORM8 || c->kind == K_SNORM16)
                            v = fmaxf(-1.0f, fminf(1.0f, v));
                        else if (c->kind == K_RG11B10F || c->kind == K_RGB9E5)
                            v = fmaxf(0.0f, v);         /* unsigned float: no upper clamp */
                        else if (c->kind != K_F16 && c->kind != K_F32)
                            v = fmaxf(0.0f, fminf(1.0f, v));

                        /* quantise into the storage */
                        switch (c->kind)
                        {
                            case K_UNORM8:
                            case K_SNORM8: {
                                int q = (c->kind == K_SNORM8) ? 127 : 255;

                                stored = roundf(v * q) / q;
                                break; }
                            case K_UNORM16:
                            case K_SNORM16: {
                                int q = (c->kind == K_SNORM16) ? 32767 : 65535;

                                stored = roundf(v * q) / q;
                                break; }
                            case K_RGB10A2U: {
                                int q = (chan == 3) ? 3 : 1023;

                                stored = roundf(v * q) / q;
                                break; }
                            case K_D16:
                                stored = roundf(v * 65535.0f) / 65535.0f;
                                break;
                            case K_D24:
                                stored = roundf(v * 16777215.0f) / 16777215.0f;
                                break;
                            case K_D32F:
                            case K_F32:
                            case K_F16:
                            case K_RG11B10F:
                            case K_RGB9E5:
                                stored = v;      /* float storage: tolerance covers */
                                break;
                            default:
                                stored = roundf(v * 255.0f) / 255.0f;
                                break;
                        }

                        /* present the stored value to the readback: a snorm storage read
                         * back through an unsigned normalised client CLAMPS to [0,1]
                         * (GL 4.6 8.5.4 converts through floats; a negative value
                         * clamps to 0, it is not biased - KHR-GL46.packed_pixels) */
                        if ((c->kind == K_SNORM8 || c->kind == K_SNORM16) &&
                            !float_client && !signed_client)
                            out = stored < 0.0f ? 0.0f : stored;
                        else
                            out = stored;

                        if (float_client)
                        {
                            /* tolerance covers the storage's precision plus the
                             * float-packed client's own re-quantisation */
                            float tol = (float)tol_of(c->kind) / 1000.0f;

                            if (ft->type == GL_UNSIGNED_INT_10F_11F_11F_REV ||
                                ft->type == GL_UNSIGNED_INT_5_9_9_9_REV)
                                tol += 0.02f;

                            if (fabsf(got_f[chan] - out) > tol)
                            {
                                char what[128];

                                snprintf(what, sizeof what,
                                         "chan %d f: got %.5f want %.5f tol %g",
                                         chan, got_f[chan], out, tol);
                                note_fail(c->name, ft->tag, what);
                                goto next_ft;
                            }
                        }
                        else
                        {
                            double want_units = (signed_client)
                                ? floor((double)out * smaxc + 0.5)
                                : floor((double)out * maxc + 0.5);
                            double ulp = (double)maxc / 255.0;
                            double tol_units = ulp + 3.0;
                            (void)smaxc;

                            if (c->internal == GL_SRGB8 || c->internal == GL_SRGB8_ALPHA8)
                                tol_units += 3.0;
                            if (c->kind == K_F16 || c->kind == K_RG11B10F ||
                                c->kind == K_RGB9E5 || c->kind == K_F32)
                                tol_units = (double)maxc * 0.03 + 3.0;  /* float storage */
                            if (c->kind == K_D24)
                                tol_units = (double)maxc / 16777215.0 + 3.0;

                            {
                                double gv = signed_client
                                    ? (double)got_i[chan]
                                    : (double)(float)(uint32_t)got_i[chan];

                                if (fabs(gv - want_units) > tol_units)
                                {
                                    char what[128];

                                    snprintf(what, sizeof what,
                                             "chan %d: got %.1f want %.1f tol %.1f",
                                             chan, gv, want_units, tol_units);
                                    note_fail(c->name, ft->tag, what);
                                    goto next_ft;
                                }
                            }
                        }
                    }
                    }
                }
                continue;
            next_ft:
                break;
            }
        }
    }

    printf("D4 sweep: %d combos, %d failures\n", total_count, fail_count);
    for (int i = 0; i < n_failures; i++)
        printf("  FAIL %s\n", failures[i]);
    EXPECT_TRUE(fail_count == 0);
    glDeleteTextures(1, &tex);
}

/* the advertised compressed formats must allocate storage (resolve to a Metal
 * format); block upload/download is exercised in test_texdata */
TEST(CompressedFormatsResolve)
{
    GLint n = 0, formats[64];

    glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &n);
    if (n > 64) n = 64;
    glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, formats);

    glGetIntegerv(GL_NUM_COMPRESSED_TEXTURE_FORMATS, &n);
    if (n > 64) n = 64;
    glGetIntegerv(GL_COMPRESSED_TEXTURE_FORMATS, formats);

    printf("compressed formats advertised: %d\n", n);
    for (int i = 0; i < n; i++)
    {
        GLint res = 0;

        glGetInternalformativ(GL_TEXTURE_2D, formats[i], GL_INTERNALFORMAT_SUPPORTED, 1, &res);
        printf("  0x%04x supported=%d\n", (unsigned)formats[i], res);
        if (!res)
        {
            char what[64];

            snprintf(what, sizeof what, "0x%04x not supported", (unsigned)formats[i]);
            note_fail("compressed", "", what);
        }
    }
    EXPECT_TRUE(fail_count == 0);
}

int main(void)
{
    RUN(RequiredInternalFormatsSweep);
    RUN(CompressedFormatsResolve);
    return mgl_test_summary();
}
