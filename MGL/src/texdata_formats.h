/*
 * texdata_formats.h — texture format table and pixel transfer engine of the
 * 'texdata' domain (owned by agent 'texdata').
 *
 * Plain C, usable from C and Objective-C (Metal pixel formats are carried as
 * uint32_t so this header does not depend on Metal or pixel_utils.h).
 *
 *   tdFormatInfo()          GL sized internal format -> storage description
 *   tdResolveInternalFormat unsized / generic-compressed -> sized
 *   tdClientLayout()        UNPACK_* / PACK_* geometry of client memory
 *   tdUnpack() / tdPack()   GL format/type <-> Metal storage layout
 *   tdConvertStorage()      storage layout -> storage layout (copy w/ conversion)
 *   tdDecodeETC()           CPU ETC2/EAC decoder (fallback when the GPU lacks ETC)
 *   tdEncodeRGTC()          CPU BC4/BC5 (RGTC) encoder (glTexImage with a compressed
 *                           internal format and uncompressed client data)
 */
#ifndef TEXDATA_FORMATS_H
#define TEXDATA_FORMATS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "glcorearb.h"

#ifdef __cplusplus
extern "C" {
#endif

/* storage kinds: how one texel is laid out in Metal memory */
enum {
    TDK_NONE = 0,
    TDK_UN8, TDK_SN8, TDK_UI8, TDK_SI8,
    TDK_UN16, TDK_SN16, TDK_UI16, TDK_SI16, TDK_F16,
    TDK_UI32, TDK_SI32, TDK_F32,
    TDK_RGB10A2UN, TDK_RGB10A2UI, TDK_RG11B10F, TDK_RGB9E5,
    TDK_D16, TDK_D32F, TDK_S8, TDK_D32FS8,
    TDK_COMPRESSED,
    /* appended (values above are stable): DEPTH_COMPONENT16 stored as Depth32Float
     * with values quantized to the 16-bit unorm lattice (see texdata_formats.c) */
    TDK_D16F
};

/* flags */
#define TDF_SRGB        0x0001
#define TDF_COMPRESSED  0x0002
#define TDF_INTEGER     0x0004
#define TDF_ETC         0x0008   /* ETC2/EAC: CPU decode when the device lacks ETC */
#define TDF_BC          0x0010
#define TDF_RENDERABLE  0x0020   /* color/depth/stencil renderable */
#define TDF_FILTERABLE  0x0040
#define TDF_DEPTH       0x0080
#define TDF_STENCIL     0x0100
#define TDF_LEGACY      0x0200   /* ALPHA / LUMINANCE / LUMINANCE_ALPHA (compat) */
#define TDF_SIGNED      0x0400
#define TDF_BPTC_FLOAT  0x0800
#define TDF_ASTC        0x1000   /* ASTC LDR: device-only (no CPU decoder) */

typedef struct TDFormat {
    GLenum   internalformat;
    GLenum   base;          /* GL_RED..GL_RGBA, GL_DEPTH_COMPONENT, GL_DEPTH_STENCIL, GL_STENCIL_INDEX,
                               GL_ALPHA, GL_LUMINANCE, GL_LUMINANCE_ALPHA (legacy) */
    GLenum   comp_type;     /* GL_UNSIGNED_NORMALIZED, GL_SIGNED_NORMALIZED, GL_FLOAT, GL_INT, GL_UNSIGNED_INT */
    uint32_t mtl;           /* MTLPixelFormat of the storage */
    uint8_t  kind;          /* TDK_* */
    uint8_t  ncomp;         /* storage components (per-component kinds) */
    uint8_t  bpp;           /* bytes per texel of the (first plane of the) storage; 0 for compressed */
    uint16_t flags;
    uint8_t  block_w, block_h, block_bytes;
    uint8_t  bits[6];       /* nominal GL bits r,g,b,a,depth,stencil */
    GLenum   decoded;       /* ETC/EAC: internal format of the CPU-decoded storage */
} TDFormat;

const TDFormat *tdFormatInfo(GLenum internalformat);

/* geom r77: GL_RGB-based format stored on a 4-component Metal backing (no
 * alpha channel) — sampling must read alpha 1 (texparam builds an A->One view) */
bool tdFormatLacksAlphaBacking(GLenum internalformat);

/* Resolve unsized base formats (GL_RGBA, ...) and generic compressed formats
 * (GL_COMPRESSED_RGBA, ...) to a concrete sized format, using format/type as a
 * hint. Returns 0 if internalformat is not a valid texture internal format. */
GLenum tdResolveInternalFormat(GLenum internalformat, GLenum format, GLenum type);

/* device capabilities (set by the renderer; ETC may be forced off with
 * the environment variable MGL_FORCE_ETC_DECODE=1) */
void tdSetDeviceCaps(bool etc, bool bc);
void tdSetASTCCap(bool astc);
bool tdDeviceHasETC(void);
bool tdDeviceHasBC(void);
bool tdDeviceHasASTC(void);

/* storage actually used for a format on this device (ETC fallback -> decoded) */
const TDFormat *tdStorageFormat(const TDFormat *f);

/* size in bytes of a compressed image of w x h x d texels */
size_t tdCompressedSize(const TDFormat *f, int w, int h, int d);

/* bytes of one storage image row / layer (plane 0) */
size_t tdStorageRowBytes(const TDFormat *f, int w);
size_t tdStorageImageBytes(const TDFormat *f, int w, int h);

/* ---- client side pixel transfer ---------------------------------------- */

/* GL_NO_ERROR, GL_INVALID_ENUM or GL_INVALID_OPERATION for a format/type pair */
GLenum tdValidateFormatType(GLenum format, GLenum type);
/* GL_NO_ERROR or GL_INVALID_OPERATION: can client format be used with a texture of this format? */
GLenum tdCheckFormatCompat(const TDFormat *f, GLenum format, GLenum type, int op);
/* geom r60: readback variant honouring DEPTH_STENCIL_TEXTURE_MODE (textures.c) */
GLenum tdCheckGetFormatCompat(const TDFormat *f, GLenum format, GLenum type, GLenum ds_mode);
/* pixel-operation selector for tdCheckFormatCompat (see its comment) */
enum { TD_PIX_TEXIMAGE = 0, TD_PIX_SUBIMAGE = 1, TD_PIX_GETTEXIMAGE = 2 };

/* bytes per pixel group (0 if invalid) */
size_t tdClientPixelSize(GLenum format, GLenum type);

typedef struct TDPixelStore {
    bool swap_bytes, lsb_first;
    int row_length, image_height, skip_rows, skip_pixels, skip_images, alignment;
    int cb_width, cb_height, cb_depth, cb_size;   /* *_COMPRESSED_BLOCK_* */
} TDPixelStore;

/* Geometry of a client image. Returns the number of bytes spanned from the base
 * pointer (offset of first pixel + extent), offset = skip offset. */
size_t tdClientLayout(const TDPixelStore *ps, GLenum format, GLenum type, int w, int h, int d,
                      bool is3d, size_t *offset, size_t *row_stride, size_t *image_stride);

/* compressed client layout (honours the compressed block pixel store state) */
size_t tdCompressedClientLayout(const TDPixelStore *ps, const TDFormat *f, int w, int h, int d,
                                bool is3d, bool *uses_block_state,
                                size_t *offset, size_t *row_stride, size_t *image_stride);

/* A region of storage-layout memory. For TDK_D32FS8 data holds the float depth
 * plane and stencil the 1-byte stencil plane. */
typedef struct TDStorage {
    void  *data;
    size_t row_pitch, image_pitch;
    void  *stencil;
    size_t s_row_pitch, s_image_pitch;
} TDStorage;

/* client (format,type) -> storage of format f. Writes depth and/or stencil planes
 * as present in the client format; wrote_depth / wrote_stencil report which. */
void tdUnpack(const TDFormat *f, const void *src, size_t src_row, size_t src_img,
              GLenum format, GLenum type, bool swap_bytes,
              int w, int h, int d, const TDStorage *dst,
              bool *wrote_depth, bool *wrote_stencil);

/* storage of format f -> client (format,type) */
void tdPack(const TDFormat *f, const TDStorage *src, int w, int h, int d,
            GLenum format, GLenum type, bool swap_bytes,
            void *dst, size_t dst_row, size_t dst_img);

/* storage -> storage with value conversion (CopyTex* between different formats) */
void tdConvertStorage(const TDFormat *sf, const TDStorage *src, const TDFormat *df, const TDStorage *dst,
                      int w, int h, int d);

/* ETC2/EAC CPU decode of a w x h image (block data tightly packed) into the
 * decoded storage of f (RGBA8 / R16 / RG16, snorm for signed EAC). */
void tdDecodeETC(const TDFormat *f, const void *blocks, int w, int h, void *dst, size_t dst_row);

/* CPU decoder for the RGTC family (BC4 single / BC5 dual channel, unorm+snorm):
 * blocks -> decoded R8 / R8_SNORM / RG8 / RG8_SNORM pixels. */
void tdDecodeRGTC(const TDFormat *f, const void *blocks, int w, int h, void *dst, size_t dst_row);

/* compress w x h x d 8-bit samples (row pitch rw, image pitch iw) into the image's BC4 /
 * BC5 (RGTC1 / RGTC2, signedness per snorm) blocks; NULL on allocation failure. */
uint8_t *tdEncodeRGTC(const uint8_t *chan0, const uint8_t *chan1, int w, int h, int d,
                      size_t rw, size_t iw, bool two_channel, bool snorm);
/* BPTC encoders (BC6H mode 10 / BC7 mode 6, one subset): online compression of
 * uncompressed client data into compressed BPTC storage.  Input is interleaved
 * RGBA (floats for BC6H, bytes for BC7) with byte row/image pitches; the
 * return is a malloc'd block buffer (free() it). */
uint8_t *tdEncodeBPTCFloat(const float *rgba, int w, int h, int d, size_t row, size_t img, int isign);
uint8_t *tdEncodeBPTCUnorm(const uint8_t *rgba, int w, int h, int d, size_t row, size_t img);


/* half <-> float */
float    tdHalfToFloat(uint16_t h);
uint16_t tdFloatToHalf(float f);

#ifdef __cplusplus
}
#endif

#endif
