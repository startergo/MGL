/* mgl_ext_xfb.h — private state and Metal hooks of the 'xfb' (transform feedback) domain.
 * Owned by agent 'xfb'.
 *
 * Mechanism (Metal has no transform feedback):
 *   - post-link: when a program has transform feedback varyings (glTransformFeedbackVaryings or
 *     in-shader xfb_buffer/xfb_offset qualifiers), the ORIGINAL vertex GLSL source is recompiled
 *     into a "capture variant": main() is renamed, a new main() calls it and then stores every
 *     captured output as raw 32-bit words into an SSBO (MSL buffer(MGL_XFB_MSL_BUFFER_INDEX)),
 *     indexed by (gl_VertexID - vstart) + instance * vcount.  The variant is built from source,
 *     so it is independent of any SPIR-V/MSL rewrite other domains apply to the program
 *     (e.g. rstate's clip-space y/z fixup): captured gl_Position is the GL value.
 *   - draw time: the Metal draw entry points (ctx->mtl_funcs.mtlDraw*) are wrapped.  While
 *     transform feedback is active and not paused, each draw first runs the capture variant in
 *     its own command buffer (rasterization disabled, no attachments), waits for it, then
 *     assembles primitives on the CPU exactly as GL orders them (strips/fans/loops/indices/
 *     primitive restart/instancing) and packs them into the bound TF buffers' CPU backing store
 *     (Buffer.data.buffer_data, kept coherent with the MTLBuffer), honouring buffer ranges and
 *     overflow (a primitive that does not fit stops writing).  The normal draw then runs unless
 *     GL_RASTERIZER_DISCARD is enabled.
 */
#ifndef MGL_EXT_XFB_H
#define MGL_EXT_XFB_H

#include <stdint.h>
#include <stdbool.h>
#include "glm_context.h"
#include "mgl_ext.h"

#define MGL_XFB_MAX_BUFFERS                 4
#define MGL_XFB_MAX_STREAMS                 4
#define MGL_XFB_MAX_INTERLEAVED_COMPONENTS  128
#define MGL_XFB_MAX_SEPARATE_COMPONENTS     4
#define MGL_XFB_MAX_SEPARATE_ATTRIBS        4

#define MGL_XFB_SSBO_BINDING                31   /* GLSL binding of the capture SSBO */
#define MGL_XFB_MSL_BUFFER_INDEX            30   /* Metal vertex buffer index it is pinned to */
#define MGL_XFB_HEADER_WORDS                4    /* vstart, vcount, instance base, record words */

/* one entry of the linked capture list (API varyings incl. gl_NextBuffer/gl_SkipComponents*,
 * or outputs carrying xfb_offset) */
typedef struct MGLXfbVarying_t {
    char    *name;          /* as reported by GetTransformFeedbackVarying */
    GLenum   type;          /* GL type, GL_NONE for gl_SkipComponents / gl_NextBuffer */
    GLint    size;          /* array size (1 for non arrays), N for gl_SkipComponentsN, 0 for gl_NextBuffer */
    GLuint   words;         /* 32-bit words captured (0 for skip / next buffer) */
    GLuint   buffer;        /* TF buffer index */
    GLuint   offset;        /* byte offset inside the buffer's vertex record */
    GLuint   src_word;      /* word offset inside the capture record */
    char    *expr;          /* GLSL expression reading the output (NULL for skip / next) */
    GLuint   comp_per_elem; /* words per array element */
    GLenum   comp_kind;     /* GL_FLOAT / GL_INT / GL_UNSIGNED_INT */
    GLuint   columns;       /* matrix columns (1 otherwise) */
    GLuint   rows;          /* vector size / matrix rows */
} MGLXfbVarying;

/* per program object */
typedef struct MGLXfbProgram_t {
    GLuint   name;
    Program *program;

    /* state set by glTransformFeedbackVaryings, applied at the next link */
    GLsizei  pending_count;
    char   **pending;
    GLenum   pending_mode;

    /* linked state */
    bool     linked;            /* a link has been processed */
    bool     link_failed;       /* xfb part of the link failed (see link_log) */
    char    *link_log;
    bool     from_shader;       /* layout(xfb_*) qualifiers decided the layout */
    GLenum   mode;              /* GL_INTERLEAVED_ATTRIBS / GL_SEPARATE_ATTRIBS */
    GLuint   count;             /* entries in vars[] */
    MGLXfbVarying *vars;
    GLint    max_name_length;   /* incl. NUL */
    GLuint   buffer_mask;       /* buffers written */
    GLuint   stride[MGL_XFB_MAX_BUFFERS];   /* bytes per vertex in each buffer */
    GLuint   record_words;      /* words per vertex in the capture record */

    /* capture variant */
    bool     vid_absolute;      /* gl_VertexID in the variant includes vertexStart (subtract vstart) */
    bool     iid_absolute;      /* gl_InstanceID in the variant includes baseInstance */
    char    *msl;
    char    *entry;
    void    *mtl_library;       /* owned by MGLRenderer+TransformFeedback.m */
    void    *mtl_function;
    void    *mtl_pipeline;
    void    *mtl_pipeline_key;
} MGLXfbProgram;

/* per transform feedback object (name 0 = default object) */
typedef struct MGLXfbObject_t {
    GLuint   name;
    bool     created;           /* bound once or made by CreateTransformFeedbacks (IsTransformFeedback) */
    bool     active, paused;
    GLenum   primitive_mode;
    Program *program;           /* program in use at Begin */
    GLuint   program_name;
    bool     ended_once;        /* EndTransformFeedback seen (DrawTransformFeedback) */

    /* binding points while this object is not bound (bound object lives in
     * ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER]) */
    BufferBaseTarget saved[MGL_XFB_MAX_BUFFERS];
    bool     is_base[MGL_XFB_MAX_BUFFERS];  /* bound with *Base (size follows the buffer) */

    GLsizeiptr write_offset[MGL_XFB_MAX_BUFFERS];   /* bytes written since Begin */
    uint64_t vertices[MGL_XFB_MAX_STREAMS];         /* vertices written since Begin */
    uint64_t draw_vertices[MGL_XFB_MAX_STREAMS];    /* snapshot at End, for DrawTransformFeedback */
    uint64_t prims_written[MGL_XFB_MAX_STREAMS];    /* since Begin */
    uint64_t prims_needed[MGL_XFB_MAX_STREAMS];
} MGLXfbObject;

/* GL-side state, reachable as ctx->state.ext_xfb; allocated in mgl_ext_init_xfb(). */
struct MGLExtState_xfb {
    MGLXfbObject   default_obj;
    MGLXfbObject **objs;        /* indexed by name, [0] unused */
    GLuint         obj_cap;
    MGLXfbProgram **progs;
    GLuint         prog_count, prog_cap;
    /* context-lifetime counters (monotonic), for queries */
    uint64_t       total_written[MGL_XFB_MAX_STREAMS];
    uint64_t       total_needed[MGL_XFB_MAX_STREAMS];
};

typedef void (*mgl_xfb_draw_arrays_fn)(GLMContext, GLenum, GLint, GLsizei);
typedef void (*mgl_xfb_draw_elements_fn)(GLMContext, GLenum, GLsizei, GLenum, const void *);

/* Metal hooks, reachable as ctx->mtl_funcs.ext_xfb; filled by the renderer category. */
struct MGLExtMetal_xfb {
    /* runs the capture variant of xp for vertices [vstart, vstart+vcount) x instances.
     * Returns the capture records (record_words words per vertex, instance-major) or NULL. */
    const uint32_t *(*capture)(GLMContext ctx, MGLXfbProgram *xp, GLint vstart, GLsizei vcount,
                               GLsizei instances, GLuint baseinstance);
    /* the CPU backing store of buf [offset, offset+size) changed: mirror it to the MTLBuffer */
    void (*buffer_written)(GLMContext ctx, Buffer *buf, size_t offset, size_t size);
    /* drop cached Metal objects of a program's capture variant */
    void (*release_program)(GLMContext ctx, MGLXfbProgram *xp);
    /* true if the instance index seen by the capture shader includes baseinstance */
    bool  instance_id_includes_base;

    /* original draw entry points, wrapped by mgl_xfb_install_draw_hooks() */
    struct GLMMetalFuncs orig;
};

#ifdef __cplusplus
extern "C" {
#endif
/* called by the renderer category after filling ctx->mtl_funcs.ext_xfb */
void mgl_xfb_install_draw_hooks(GLMContext ctx);

/* GetProgramiv TRANSFORM_FEEDBACK_VARYINGS / _VARYING_MAX_LENGTH / _BUFFER_MODE (for program.c).
 * Returns true when pname was handled. */
bool mgl_xfb_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params);
/* xfb part of linking failed (unknown varying, too many components, ...): program.c should
 * report LINK_STATUS false and append *log to the info log. */
bool mgl_xfb_link_failed(GLMContext ctx, Program *program, const char **log);
/* linked capture description (program introspection: TRANSFORM_FEEDBACK_VARYING interface) */
const MGLXfbProgram *mgl_xfb_program_info(GLMContext ctx, Program *program);

/* buffers.c hook: validates Bind{Buffer,Buffers}{Base,Range}(GL_TRANSFORM_FEEDBACK_BUFFER, ...)
 * (active TF, index < MAX_TRANSFORM_FEEDBACK_BUFFERS, 4-byte alignment) and records the
 * binding kind.  Raises the GL error and returns false when the bind must not happen. */
bool mgl_xfb_validate_bind_buffer(GLMContext ctx, GLenum target, GLuint index, GLuint buffer,
                                  GLintptr offset, GLsizeiptr size, bool range);

/* counters (context lifetime, monotonic) — primitives written / needed on a stream */
uint64_t mgl_xfb_primitives_written(GLMContext ctx, GLuint stream);
uint64_t mgl_xfb_primitives_needed(GLMContext ctx, GLuint stream);
/* true while a transform feedback object is active and not paused */
bool mgl_xfb_capturing(GLMContext ctx);
#ifdef __cplusplus
}
#endif

#endif
