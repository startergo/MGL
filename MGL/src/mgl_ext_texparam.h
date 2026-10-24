/* mgl_ext_texparam.h — private state and Metal hooks of the 'texparam' domain. Owned by agent 'texparam'.
 *
 * Covers texture / sampler parameters, texture level queries and internalformat queries.
 *
 * Other domains may record texture metadata the texparam queries report
 * (texture views, buffer textures, multisample storage, proxy targets):
 *     mgl_texparam_record_view / _record_buffer / _record_multisample
 *     mgl_texparam_proxy_texture   -> the Texture object that holds a proxy target's state
 * Texture names come from getNewName() and are never reused, so records are keyed by name.
 */
#ifndef MGL_EXT_TEXPARAM_H
#define MGL_EXT_TEXPARAM_H

#include "glm_context.h"
#include "mgl_ext.h"

/* per texture-name metadata recorded by other domains */
typedef struct MGLTexParamRecord {
    GLboolean   valid;
    GLboolean   is_view;
    GLuint      view_min_level, view_num_levels, view_min_layer, view_num_layers;
    GLuint      view_immutable_levels;  /* TEXTURE_IMMUTABLE_LEVELS of the ORIGINAL
                                         * texture (ARB_texture_view: the view reports
                                         * the parent's level count, not numlevels) */
    GLboolean   is_buffer;
    GLuint      buffer;
    GLintptr    buffer_offset;
    GLsizeiptr  buffer_size;            /* -1: whole buffer */
    GLsizei     samples;                /* 0: not multisample */
    GLboolean   fixed_sample_locations;
} MGLTexParamRecord;

/* Hard cap on the per-name texparam record array (names above this, and the
 * internal sentinel names >= TEX_OBJ_RES_NAME, get no record).  Real GL names
 * are small sequential integers from MGL's name allocator. */
#define MGL_TEXPARAM_MAX_RECORDS (1u << 22)

#define MGL_TEXPARAM_NUM_PROXY 11
/* GL-side state, reachable as ctx->state.ext_texparam; allocated in mgl_ext_init_texparam(). */
struct MGLExtState_texparam {
    MGLTexParamRecord *records;         /* indexed by texture name */
    GLuint             num_records;
    Texture           *proxy[MGL_TEXPARAM_NUM_PROXY];
};

/* Metal pixel format capabilities (derived from MTLDevice + Apple GPU family tables) */
typedef struct MGLTexParamFormatCaps {
    GLboolean supported;
    GLboolean filterable;
    GLboolean color_renderable;
    GLboolean depth_renderable;
    GLboolean stencil_renderable;
    GLboolean blendable;
    GLboolean msaa;
    GLboolean writable;        /* shader write */
    GLboolean atomic;
    GLint     num_sample_counts;
    GLint     sample_counts[8];   /* descending, all > 1 */
} MGLTexParamFormatCaps;

/* Properties of a realized Metal texture */
typedef struct MGLTexParamMtlInfo {
    GLuint    sample_count;
    GLuint    mipmap_level_count;
    GLuint    array_length;
    GLboolean is_view;
    GLuint    parent_level;
    GLuint    parent_slice;
    GLboolean has_buffer;
    GLsizeiptr buffer_offset;
    GLsizeiptr buffer_length;
    void      *mtl_buffer;      /* unretained id<MTLBuffer>, for name lookup */
} MGLTexParamMtlInfo;

/* Metal hooks, reachable as ctx->mtl_funcs.ext_texparam; filled by MGLRenderer+TextureParams.m. */
struct MGLExtMetal_texparam {
    bool (*formatCaps)(GLMContext ctx, GLuint mtl_pixel_format, MGLTexParamFormatCaps *caps);
    bool (*textureInfo)(GLMContext ctx, Texture *tex, MGLTexParamMtlInfo *info);
    void (*releaseObj)(GLMContext ctx, void *obj);   /* CFBridgingRelease without a command-buffer flush */
};

#ifdef __cplusplus
extern "C" {
#endif
/* default texture/sampler state (GL 4.6 tables 23.18/23.19) */
void mglTexParamInitDefaults(TextureParameter *p, GLenum target);
/* drop cached Metal sampler/view state after a parameter change */
void mglTexParamInvalidate(GLMContext ctx, TextureParameter *p);

/* metadata recording for other domains */
MGLTexParamRecord *mgl_texparam_record(GLMContext ctx, GLuint name, bool create);
void mgl_texparam_record_view(GLMContext ctx, Texture *tex, GLuint min_level, GLuint num_levels, GLuint min_layer, GLuint num_layers, GLuint immutable_levels);
void mgl_texparam_record_buffer(GLMContext ctx, Texture *tex, GLuint buffer, GLintptr offset, GLsizeiptr size);
void mgl_texparam_record_multisample(GLMContext ctx, Texture *tex, GLsizei samples, GLboolean fixed_sample_locations);
Texture *mgl_texparam_proxy_texture(GLMContext ctx, GLenum proxy_target);

/* level parameter core (used by Get{Tex,Texture}LevelParameter*) */
void mglTexParamGetLevel(GLMContext ctx, Texture *tex, GLenum target, GLint level, GLenum pname, GLint *iparams, GLfloat *fparams, bool dsa);

/* GL-level format description used by queries */
typedef struct MGLFormatDesc {
    GLenum  fmt;
    GLenum  base;
    GLubyte r, g, b, a, d, s, shared;
    GLenum  type;            /* component type of color / depth */
    GLboolean srgb;
    GLboolean compressed;
    GLubyte bw, bh;
    GLushort bbytes;
    GLenum  img_class;
    GLenum  view_class;
    GLenum  pix_format, pix_type;
    GLenum  preferred;       /* sized format used for generic/unsized formats */
} MGLFormatDesc;
const MGLFormatDesc *mglTexParamFormatDesc(GLenum internalformat);
#ifdef __cplusplus
}
#endif

#endif
