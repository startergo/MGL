/* mgl_ext_texext.h — private state and Metal hooks of the 'texext' domain. Owned by agent 'texext'.
 *
 * texext covers buffer textures (TexBuffer*), texture views (TextureView), multisample textures
 * (Tex*Multisample / TextureStorage*Multisample), texture invalidation and TextureBarrier.
 *
 * The core Texture struct is shared, so all texext-specific per-texture state lives in a side table
 * keyed by Texture* (MGLTexExtInfo).  Other domains read it through the accessors at the bottom
 * (e.g. texparam's glGetTexParameter TEXTURE_VIEW_*, texdata's glGetTexLevelParameter TEXTURE_SAMPLES).
 */
#ifndef MGL_EXT_TEXEXT_H
#define MGL_EXT_TEXEXT_H

#include <stdbool.h>
#include "glm_context.h"
#include "mgl_ext.h"

enum {
    MGL_TEXEXT_NONE = 0,        /* only invalidation bookkeeping */
    MGL_TEXEXT_BUFFER,          /* GL_TEXTURE_BUFFER with (optional) buffer data store */
    MGL_TEXEXT_VIEW,            /* created by glTextureView, shares storage with view_root */
    MGL_TEXEXT_MULTISAMPLE      /* GL_TEXTURE_2D_MULTISAMPLE(_ARRAY) */
};

typedef struct MGLTexExtInfo_t {
    Texture *tex;
    int kind;

    /* ---- buffer textures ---- */
    GLuint      buffer_name;        /* 0 = no data store attached */
    GLenum      buffer_format;      /* internalformat given to TexBuffer */
    GLintptr    buffer_offset;      /* TEXTURE_BUFFER_OFFSET */
    GLsizeiptr  buffer_size;        /* TEXTURE_BUFFER_SIZE requested (range) */
    GLboolean   buffer_ranged;      /* TexBufferRange vs TexBuffer (TexBuffer tracks buffer size) */
    GLuint      buffer_texel_size;  /* bytes per texel in the buffer */
    GLuint      buffer_components;  /* 1..4 */
    GLboolean   buffer_expand3;     /* RGB32F/I/UI: expanded to RGBA32 shadow (Metal has no RGB32) */
    /* Metal-side identity of what the current MTLTexture was built from (not retained) */
    void       *mtl_built_from;     /* id<MTLBuffer> of the GL buffer */
    GLsizeiptr  mtl_built_size;     /* buffer size at build time */
    GLboolean   mtl_uses_shadow;    /* texture lives in a converted shadow copy */
    void       *shadow_src;         /* malloc'd snapshot of the source bytes used for the shadow */
    size_t      shadow_src_size;

    /* ---- texture views ---- */
    Texture    *view_root;          /* texture owning the storage (never itself a view) */
    GLuint      view_min_level;     /* TEXTURE_VIEW_MIN_LEVEL (relative to root) */
    GLuint      view_num_levels;    /* TEXTURE_VIEW_NUM_LEVELS */
    GLuint      view_min_layer;     /* TEXTURE_VIEW_MIN_LAYER (relative to root) */
    GLuint      view_num_layers;    /* TEXTURE_VIEW_NUM_LAYERS */
    void       *view_built_from;    /* root id<MTLTexture> the view was made from (not retained) */
    GLboolean   has_format_views;   /* (root) some view reinterprets the pixel format */
    GLboolean   has_views;          /* (root) has views: keep render-target usage so views can be attached */
    int         view_refs;          /* (root) live views whose view_root is this texture: GL 4.6
                                     * 9.2 — a view keeps the parent's images alive, so the
                                     * root's delete is DEFERRED until the last view is gone
                                     * (see mglDeleteTextures / mgl_texext_view_unref_root) */
    GLboolean   pending_delete;     /* (root) glDeleteTextures ran with views still alive: the
                                     * texture is out of the hash (name = TEX_OBJ_RES_NAME)
                                     * but its storage stays valid for the views */

    /* ---- multisample ---- */
    GLsizei     samples_requested;
    GLsizei     samples;            /* TEXTURE_SAMPLES (actual Metal sample count) */
    GLboolean   fixed_sample_locations;

    /* ---- common ---- */
    GLuint      immutable_levels;   /* TEXTURE_IMMUTABLE_LEVELS for views / MS storage */
    GLuint      invalid_levels;     /* bit per level: contents undefined after InvalidateTexImage */

    struct MGLTexExtInfo_t *next;
} MGLTexExtInfo;

#define MGL_TEXEXT_BUCKETS 256

/* GL-side state, reachable as ctx->state.ext_texext; allocated in mgl_ext_init_texext(). */
struct MGLExtState_texext {
    MGLTexExtInfo *buckets[MGL_TEXEXT_BUCKETS];

    /* live MGL_TEXEXT_BUFFER textures with a data store: when > 0 the draw path
     * stamps read_serial on the backing buffers (a buffer sampled as a texture
     * buffer is outside the buffer-map lists, so without this stamp a queued
     * draw's read is invisible to the rename logic and a following
     * glBufferSubData / glMapBufferRange writes into memory the GPU still
     * samples — dEQP texture_buffer.render_modify) */
    GLuint buffer_texture_count;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texext; filled by MGLRenderer+TextureExt.m. */
struct MGLExtMetal_texext {
    /* GL TextureBarrier: make prior render-target writes visible to later fetches */
    void (*textureBarrier)(GLMContext ctx);
    /* level contents became undefined (whole level, all layers) */
    void (*invalidateTexImage)(GLMContext ctx, Texture *tex, GLint level);
    /* actual Metal sample count to use for a request (0 = unsupported) */
    GLsizei (*sampleCountFor)(GLMContext ctx, GLenum internalformat, GLsizei samples);
    /* root gained a view: make its Metal texture PixelFormatView (if reinterpreted) and
       RenderTarget capable, preserving contents */
    void (*prepareViewSource)(GLMContext ctx, Texture *root);
};

#ifdef __cplusplus
extern "C" {
#endif

/* side table */
MGLTexExtInfo *mgl_texext_info(GLMContext ctx, const Texture *tex);           /* NULL if none */
MGLTexExtInfo *mgl_texext_info_create(GLMContext ctx, Texture *tex);          /* find or create */
/* drop the side-table entry of a texture that is being destroyed.  The table is keyed by the
 * Texture pointer: without this, a later Texture allocated at the same address inherits the
 * old one's kind (view / buffer / multisample) and is then refused by the texdata path. */
void mgl_texext_info_free(GLMContext ctx, const Texture *tex);

/* a view of `root` is being destroyed: drop the root's view reference; returns true when the
 * caller must free the root NOW (its glDeleteTextures was deferred for this reference) */
bool mgl_texext_view_unref_root(GLMContext ctx, Texture *root);

/* buffer texture formats (GL 4.6 Table 8.16); returns false if not a legal buffer texture format */
bool mgl_texext_buffer_format_info(GLenum internalformat, GLuint *texel_size, GLuint *components, bool *expand3);

/* ---- accessors for other domains (return false if pname/texture not texext-managed) ----
 * tex parameters: GL_TEXTURE_VIEW_MIN_LEVEL, GL_TEXTURE_VIEW_NUM_LEVELS, GL_TEXTURE_VIEW_MIN_LAYER,
 *                 GL_TEXTURE_VIEW_NUM_LAYERS, GL_TEXTURE_IMMUTABLE_LEVELS (views / MS storage),
 *                 GL_TEXTURE_IMMUTABLE_FORMAT (views / MS storage)
 * level params:   GL_TEXTURE_SAMPLES, GL_TEXTURE_FIXED_SAMPLE_LOCATIONS, GL_TEXTURE_BUFFER_DATA_STORE_BINDING,
 *                 GL_TEXTURE_BUFFER_OFFSET, GL_TEXTURE_BUFFER_SIZE, GL_TEXTURE_WIDTH/HEIGHT/DEPTH and
 *                 GL_TEXTURE_INTERNAL_FORMAT for buffer textures */
bool mgl_texext_get_tex_parameter(GLMContext ctx, Texture *tex, GLenum pname, GLint *out);
bool mgl_texext_get_tex_level_parameter(GLMContext ctx, Texture *tex, GLint level, GLenum pname, GLint *out);

/* convenience */
bool mgl_texext_is_view(GLMContext ctx, const Texture *tex);
GLsizei mgl_texext_samples(GLMContext ctx, const Texture *tex);     /* 0 for single-sampled */

#ifdef __cplusplus
}
#endif

#endif
