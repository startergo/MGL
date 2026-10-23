/* mgl_ext_vattrib.h — private state and Metal hooks of the 'vattrib' domain. Owned by agent 'vattrib'.
 *
 * GL 4.6 vertex-array model (§10.3): every VAO has MAX_ATTRIBS generic attribute *formats*
 * (size/type/normalized/integer/long/relative offset/binding index) and MGL_VA_MAX_BINDINGS
 * vertex buffer *bindings* (buffer/offset/stride/divisor).  The shared VertexArray struct in
 * glm_context.h only has a merged per-attribute record, so every VAO is allocated as a
 * MGLVertexArrayFull (VertexArray first, so a VertexArray* is a valid pointer to it) and the
 * legacy VertexArray.attrib[] fields are kept in sync (mgl_va_sync_attrib) for code in other
 * domains (draw_buffers.c validate_vao, buffers.c) that still reads them.
 */
#ifndef MGL_EXT_VATTRIB_H
#define MGL_EXT_VATTRIB_H

#include <stdint.h>
#include "glm_context.h"
#include "mgl_ext.h"

#define MGL_VA_MAX_BINDINGS         MAX_ATTRIBS     /* GL_MAX_VERTEX_ATTRIB_BINDINGS */
#define MGL_VA_MAX_RELATIVE_OFFSET  2047            /* GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET */
#define MGL_VA_MAX_STRIDE           2048            /* GL_MAX_VERTEX_ATTRIB_STRIDE */

/* Metal vertex buffer indices reserved by this domain (MGL's buffer map grows upward from 0).
 * Slot 30 belongs to the xfb capture scratch (MGL_XFB_MSL_BUFFER_INDEX): while transform
 * feedback is active, MGLRenderer+TransformFeedback.m binds it on the SAME render encoder
 * right after the constants, so the two must never share a slot (KHR-GL46
 * vertex_attrib_binding basic-input-case1/case2 in one process: case2's disabled attribute
 * read the scratch bytes — stale vertex data from case1 — instead of its current value).
 * Slot 30 is also MGL_TESS_VS_OUT_BUF (the tess raster pass reads it), but no tess draw
 * has vertex attributes, so only the xfb capture collides in practice. */
#define MGL_VA_CONST_BUFFER_INDEX   29              /* current generic values (constant step) */
#define MGL_VA_SHADOW_BUFFER_TOP    28              /* CPU-converted attribute streams, downward */

/* kind of a current generic attribute value */
enum { MGL_VA_KIND_FLOAT = 0, MGL_VA_KIND_INT = 1, MGL_VA_KIND_UINT = 2, MGL_VA_KIND_DOUBLE = 3 };

typedef struct MGLVertexAttribFormat_t {
    GLint       size;           /* 1..4 or GL_BGRA */
    GLenum      type;
    GLboolean   normalized;
    GLboolean   integer;        /* VertexAttribIPointer / IFormat */
    GLboolean   is_long;        /* VertexAttribLPointer / LFormat */
    GLuint      relativeoffset;
    GLuint      binding;        /* VERTEX_ATTRIB_BINDING */
    GLsizei     pointer_stride; /* VERTEX_ATTRIB_ARRAY_STRIDE (as given to *Pointer) */
    const void *pointer;        /* VERTEX_ATTRIB_ARRAY_POINTER */
} MGLVertexAttribFormat;

typedef struct MGLVertexBinding_t {
    Buffer     *buffer;
    GLuint      buffer_name;
    GLintptr    offset;
    GLsizei     stride;
    GLuint      divisor;
} MGLVertexBinding;

typedef struct MGLVertexArrayFull_t {
    VertexArray             base;   /* must be first */
    MGLVertexAttribFormat   fmt[MAX_ATTRIBS];
    MGLVertexBinding        binding[MGL_VA_MAX_BINDINGS];
} MGLVertexArrayFull;

static inline MGLVertexArrayFull *mgl_va_full(VertexArray *vao) { return (MGLVertexArrayFull *)vao; }

typedef struct MGLCurrentAttrib_t {
    union { GLfloat f[4]; GLint i[4]; GLuint u[4]; } v;
    GLdouble d[4];
    GLuint   kind;
} MGLCurrentAttrib;

/* per-attribute draw-time fetch mode chosen by the renderer */
enum { MGL_VA_FETCH_NONE = 0, MGL_VA_FETCH_DIRECT, MGL_VA_FETCH_SHADOW, MGL_VA_FETCH_CONST };

typedef struct MGLVattribDrawPlan_t {
    int         valid;
    int         binding_index[MGL_VA_MAX_BINDINGS];   /* Metal buffer index, -1 unused */
    int         fetch[MAX_ATTRIBS];
    int         shader_kind[MAX_ATTRIBS];             /* MGL_VA_KIND_* of the shader input */
    int         shader_comps[MAX_ATTRIBS];
    int         shadow_index[MAX_ATTRIBS];            /* Metal buffer index of a SHADOW stream */
    int         shadow_stride[MAX_ATTRIBS];           /* element stride of that stream */
    unsigned    const_mask;
    uint8_t     dbl[MAX_ATTRIBS];                     /* fp64 word slot: 0 none, 1 low (comp 0), 2 high (comp 2) */
    int         dbl_words[MAX_ATTRIBS];               /* u32 words this slot fetches */
    int         owner_gl[MAX_ATTRIBS];                /* GL attribute location feeding this internal slot
                                                         (== the index except for dvec3/dvec4-based inputs) */
} MGLVattribDrawPlan;

/* GL-side state, reachable as ctx->state.ext_vattrib; allocated in mgl_ext_init_vattrib(). */
struct MGLExtState_vattrib {
    MGLCurrentAttrib    current[MAX_ATTRIBS];
    uint8_t            *gen_names;      /* VAO names returned by GenVertexArrays and not deleted */
    size_t              gen_names_cap;
    MGLVattribDrawPlan  plan;           /* written by MGLRenderer+VertexAttrib.m */
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_vattrib; filled by the renderer category. */
struct MGLExtMetal_vattrib {
    void *reserved;
};

#ifdef __cplusplus
extern "C" {
#endif
/* vertex_arrays.c */
void mgl_va_sync_attrib(VertexArray *vao, GLuint attrib);
void mgl_va_sync_all(VertexArray *vao);
void mgl_va_element_buffer_bound(GLMContext ctx, Buffer *buf);   /* glBindBuffer(GL_ELEMENT_ARRAY_BUFFER) */
GLsizei mgl_va_element_size(GLenum type, GLint size);            /* bytes of one attribute element */
/* vertex_attrib.c: decode one element of attribute data at src into 4 components */
void mgl_va_fetch_float(const MGLVertexAttribFormat *fmt, const void *src, GLfloat out[4]);
void mgl_va_fetch_int(const MGLVertexAttribFormat *fmt, const void *src, GLint out[4]);
void mgl_va_fetch_dbl_words(const MGLVertexAttribFormat *fmt, const void *src, GLuint out[8]);
float mgl_va_uf11_to_float(GLuint v);
bool mgl_va_attrib_param(VertexArray *vao, GLuint index, GLenum pname, GLint64 *out);
float mgl_va_uf10_to_float(GLuint v);
#ifdef __cplusplus
}
#endif

#ifdef __OBJC__
#import <Metal/Metal.h>
#import "MGLRenderer.h"
@interface MGLRenderer (VertexAttrib)
- (bool) vattribMapVertexBuffers: (BufferMapList *) buffer_map;
- (MTLVertexDescriptor *) vattribGenerateVertexDescriptor;
@end
MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);
MTLVertexFormat mglVertexFormatForAttrib(GLenum type, GLint size, bool normalized, bool integer);
#endif

#endif
