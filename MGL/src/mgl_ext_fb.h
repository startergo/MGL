/* mgl_ext_fb.h — private state and Metal hooks of the 'fb' domain. Owned by agent 'fb'.
 *
 * Framebuffer / renderbuffer objects are allocated by framebuffers.c as the
 * extended structs below (the public struct is the first member), so the
 * extra per-object state needs no change to glm_context.h.
 *
 * Row convention: Metal texture row r == GL window row y=r (bottom-up) for
 * every render target; readback/blits never flip rows, only present does.
 */
#ifndef MGL_EXT_FB_H
#define MGL_EXT_FB_H

#include "glm_context.h"
#include "mgl_ext.h"

#define MGL_FB_DEPTH_BIT    (1u << 8)
#define MGL_FB_STENCIL_BIT  (1u << 9)
#define MGL_FB_ATT_DEPTH    8   /* index into per-attachment arrays */
#define MGL_FB_ATT_STENCIL  9
#define MGL_FB_NUM_ATT      10

/* per framebuffer (FBO or default) draw/read selection */
typedef struct MGLFBDrawState {
    GLenum  draw_buffers[MAX_COLOR_ATTACHMENTS];
    GLsizei num_draw_buffers;
    GLenum  read_buffer;
    GLuint  invalid_mask;       /* bit i = color attachment i (default fb: bit 0), MGL_FB_DEPTH_BIT, MGL_FB_STENCIL_BIT */
} MGLFBDrawState;

typedef struct MGLFramebufferEx {
    Framebuffer     base;       /* must be first */
    MGLFBDrawState  ds;
    GLboolean       created;    /* object exists (bound once or created by CreateFramebuffers) */
    GLboolean       layered[MGL_FB_NUM_ATT];
} MGLFramebufferEx;

typedef struct MGLRenderbufferEx {
    Renderbuffer    base;       /* must be first */
    GLboolean       created;
    GLenum          internalformat;
    GLsizei         width, height;
    GLsizei         samples;    /* requested-and-rounded sample count, 0 = single sample */
} MGLRenderbufferEx;

/* value interpretation of a clear / read */
enum { MGL_FB_KIND_FLOAT = 0, MGL_FB_KIND_INT = 1, MGL_FB_KIND_UINT = 2 };

typedef struct MGLFBClearRequest {
    Framebuffer *fbo;                       /* NULL = default framebuffer */
    GLuint      color_slots;                /* bit i = draw-buffer slot i */
    int         color_kind[MAX_COLOR_ATTACHMENTS];
    union { GLfloat f[4]; GLint i[4]; GLuint u[4]; } color[MAX_COLOR_ATTACHMENTS];
    GLboolean   depth, stencil;
    GLfloat     depth_value;
    GLint       stencil_value;
} MGLFBClearRequest;

enum { MGL_FB_ASPECT_COLOR = 0, MGL_FB_ASPECT_DEPTH = 1, MGL_FB_ASPECT_STENCIL = 2 };

/* Read request: out receives width*height texels in GL row order (row 0 = y),
 * color: 4 x 32-bit (float/int/uint per returned kind), depth: float, stencil: GLuint.
 * Texels outside the attachment are left untouched. */
typedef struct MGLFBReadRequest {
    Framebuffer *fbo;
    GLenum      attachment;   /* GL_COLOR_ATTACHMENTi for FBOs; ignored for the default fb color */
    int         aspect;
    GLint       x, y;
    GLsizei     width, height;
    void        *out;
} MGLFBReadRequest;

typedef struct MGLFBBlitRequest {
    Framebuffer *read_fbo, *draw_fbo;
    GLint src[4], dst[4];       /* x0,y0,x1,y1 */
    GLbitfield mask;
    GLenum filter;
} MGLFBBlitRequest;

/* GL-side state, reachable as ctx->state.ext_fb; allocated in mgl_ext_init_fb(). */
struct MGLExtState_fb {
    MGLFBDrawState  default_ds;     /* default framebuffer */
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_fb; filled by MGLRenderer+Framebuffer.m. */
struct MGLExtMetal_fb {
    void *default_color;            /* retained id<MTLTexture> */
    void *default_depth_stencil;    /* retained id<MTLTexture> */
    /* bumped whenever the default framebuffer's backing textures are (re)created:
     * part of the render-pass-descriptor cache key (FBTarget.name), so a recycled
     * Metal texture address can never alias two generations of the default fb */
    uint64_t default_fb_gen;
    void *pipeline_cache;           /* retained NSMutableDictionary */
    GLuint default_width, default_height;
    GLint  max_samples;
    GLboolean scissor_initialized;

    /* returns default framebuffer size (creating the textures if needed) */
    void (*default_size)(GLMContext ctx, GLuint *w, GLuint *h);
    /* end the current render pass (drawing targets changed) */
    void (*targets_changed)(GLMContext ctx);
    void (*clear)(GLMContext ctx, const MGLFBClearRequest *req);
    /* returns MGL_FB_KIND_* or -1 on failure */
    int  (*read)(GLMContext ctx, const MGLFBReadRequest *req);
    void (*blit)(GLMContext ctx, const MGLFBBlitRequest *req);
    /* create the Metal storage of a renderbuffer; false = out of memory / unsupported */
    bool (*renderbuffer_storage)(GLMContext ctx, MGLRenderbufferEx *rb);
    void (*release_texture)(GLMContext ctx, Texture *tex);
    /* samples: actual Metal sample count; positions in GL convention (origin bottom-left) */
    bool (*sample_position)(GLMContext ctx, GLuint samples, GLuint index, GLfloat *xy);
    GLuint (*round_samples)(GLMContext ctx, GLuint samples);
    /* write CPU bytes into a GL buffer object (pixel pack buffer) */
    void (*write_buffer)(GLMContext ctx, Buffer *buf, size_t offset, size_t size, const void *src);
};

/* C helpers exported by framebuffers.c / framebuffer_ext.c for the Metal side */
MGLFBDrawState *mgl_fb_draw_state(GLMContext ctx, Framebuffer *fbo);
FBOAttachment *mgl_fb_attachment(Framebuffer *fbo, GLenum attachment); /* NULL if invalid enum */
Texture *mgl_fb_attachment_texture(GLMContext ctx, FBOAttachment *att);  /* NULL if nothing (validly) attached */
GLsizei mgl_fb_attachment_samples(GLMContext ctx, FBOAttachment *att);
bool mgl_fb_attachment_size(GLMContext ctx, FBOAttachment *att, GLuint *w, GLuint *h, GLuint *layers);
void mgl_fb_attachment_slice(GLMContext ctx, Framebuffer *fbo, FBOAttachment *att, GLuint *level, GLuint *slice, GLuint *depth_plane, GLuint *layer_count);
GLenum mgl_fb_status(GLMContext ctx, Framebuffer *fbo);

/* internal-format table */
typedef struct MGLFBFormatInfo {
    GLenum internalformat;
    GLenum base;            /* GL_RED, GL_RG, GL_RGB, GL_RGBA, GL_DEPTH_COMPONENT, GL_DEPTH_STENCIL, GL_STENCIL_INDEX */
    GLubyte r, g, b, a, d, s;
    GLenum component_type;  /* GL_UNSIGNED_NORMALIZED, GL_SIGNED_NORMALIZED, GL_FLOAT, GL_INT, GL_UNSIGNED_INT */
    GLboolean srgb;
    GLboolean color_renderable, depth_renderable, stencil_renderable;
} MGLFBFormatInfo;
const MGLFBFormatInfo *mgl_fb_format_info(GLenum internalformat);
bool mgl_fb_slot_lacks_alpha(GLMContext ctx, Framebuffer *fbo, GLuint slot); /* geom r58 */

#endif
