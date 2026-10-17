/* mgl_ext_texdata.h — private state and Metal hooks of the 'texdata' domain. Owned by agent 'texdata'.
 *
 * Texture storage model
 * ---------------------
 * GL-side every texture keeps, per face and level, its GL definition in
 * Texture.faces[f].levels[l] (width/height/depth, complete, internal format in
 * .mtl_format).  Pixel data lives either
 *   - on the GPU, in the realized MTLTexture (authoritative once a level is part
 *     of the realized shape: TDTexInfo.on_gpu bit set), or
 *   - in a CPU "shadow" (TextureLevel.data, storage layout of the Metal format,
 *     Metal geometry: 1D-array layers are slices) while the level is not part of
 *     the realized MTLTexture (not yet realized, or the shape changed).
 * Realization (MGLRenderer+TextureData.m) creates/recreates the MTLTexture from
 * the GL shape, blit-copies surviving levels from the previous MTLTexture,
 * uploads pending shadows, and reads back levels that drop out of the shape.
 *
 * Per-texture side info is indexed by Texture.dirty_on_gpu (1-based; that field
 * is otherwise unused) so it needs no change to the shared Texture struct.
 */
#ifndef MGL_EXT_TEXDATA_H
#define MGL_EXT_TEXDATA_H

#include "glm_context.h"
#include "mgl_ext.h"
#include "texdata_formats.h"

#define TD_MAX_LEVELS 16

typedef struct TDShape {
    GLenum target;          /* GL texture target of the object */
    GLenum internalformat;  /* GL sized internal format */
    GLuint width, height, depth;    /* Metal geometry of level 0: depth = 3D depth or slice count */
    GLuint levels;
    uint32_t swizzle;       /* packed MTLTextureSwizzleChannels (r | g<<8 | b<<16 | a<<24) */
} TDShape;

typedef struct TDTexInfo {
    GLboolean used;
    GLboolean realized;
    TDShape   shape;                    /* shape of the realized MTLTexture */
    uint32_t  on_gpu[6];                /* level bitmask: content is authoritative on the GPU */
    void     *orig[6][TD_MAX_LEVELS];   /* ETC CPU-decode fallback: original compressed blocks */
    size_t    orig_size[6][TD_MAX_LEVELS];
    void     *view;                     /* retained id<MTLTexture>: base texture when tex->mtl_data is a view */
} TDTexInfo;

/* GL-side state, reachable as ctx->state.ext_texdata; allocate in mgl_ext_init_texdata(). */
struct MGLExtState_texdata {
    TDTexInfo **infos;
    GLuint     num_infos, cap_infos;
    GLuint     free_hint;
    int        unpack_cb[4];            /* UNPACK_COMPRESSED_BLOCK_WIDTH/HEIGHT/DEPTH/SIZE */
    int        pack_cb[4];              /* PACK_COMPRESSED_BLOCK_WIDTH/HEIGHT/DEPTH/SIZE */
    Texture   *defaults[_MAX_TEXTURE_TYPES];   /* default texture objects (name 0) */
};

/* A region in Metal geometry: z = slice (array layer / cube-array layer-face) or 3D depth. */
typedef struct TDRegion {
    GLuint face, level;
    GLint  x, y, z;
    GLsizei w, h, d;
} TDRegion;

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texdata; filled by the renderer category. */
struct MGLExtMetal_texdata {
    /* upload storage-layout data into a realized level (GPU blit) */
    void (*upload)(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *src, bool depth, bool stencil);
    /* read storage-layout data of a region (synchronous) */
    bool (*download)(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *dst);
    /* GPU-decode a compressed region into RGBA float (4 floats per texel, rows of w) */
    bool (*download_decoded)(GLMContext ctx, Texture *tex, const TDRegion *r, float *dst);
    /* make sure the MTLTexture exists and matches the GL shape */
    bool (*realize)(GLMContext ctx, Texture *tex);
    /* copy w x h from the current read framebuffer at (x,y) into r (r->w/h ignored) */
    GLenum (*copy_from_read_fb)(GLMContext ctx, Texture *dst, const TDRegion *r, GLint x, GLint y, GLsizei w, GLsizei h, bool depth, bool stencil);
    /* raw texel copy between two realized textures (CopyImageSubData) */
    bool (*copy_image)(GLMContext ctx, Texture *src, const TDRegion *sr, Texture *dst, const TDRegion *dr);
    /* GPU mipmap generation from the base level */
    bool (*generate_mipmaps)(GLMContext ctx, Texture *tex, GLuint base, GLuint last);
    /* release Metal objects of a texture */
    void (*release)(GLMContext ctx, Texture *tex);
    /* information about the current read buffer: internal format, size; false if none */
    bool (*read_buffer_info)(GLMContext ctx, bool depth, bool stencil, GLenum *internalformat, GLuint *w, GLuint *h);
};

/* ---- C helpers shared with MGLRenderer+TextureData.m ---- */
#ifdef __cplusplus
extern "C" {
#endif
TDTexInfo *mgl_texdata_info(GLMContext ctx, Texture *tex);           /* allocates on first use */
TDTexInfo *mgl_texdata_find_info(GLMContext ctx, Texture *tex);      /* NULL if none */
bool  mgl_texdata_shape(GLMContext ctx, Texture *tex, TDShape *shape);   /* GL shape to realize */
const TDFormat *mgl_texdata_storage_format(GLenum internalformat);
void  mgl_texdata_level_geometry(Texture *tex, GLuint face, GLuint level, GLuint *w, GLuint *h, GLuint *d);
bool  mgl_texdata_shadow(GLMContext ctx, Texture *tex, GLuint face, GLuint level, bool alloc, TDStorage *st);
void  mgl_texdata_free_shadow(GLMContext ctx, Texture *tex, GLuint face, GLuint level);
GLuint mgl_texdata_num_faces(Texture *tex);
/* glPixelStore hook for the compressed block pnames; returns true if handled */
bool  mgl_texdata_pixel_store(GLMContext ctx, GLenum pname, GLint param);
/* write tight storage-layout data (Metal geometry) into a level: GPU if realized, else CPU shadow */
bool  mgl_texdata_write_region(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *src, bool depth, bool stencil);
#ifdef __cplusplus
}
#endif

#endif
