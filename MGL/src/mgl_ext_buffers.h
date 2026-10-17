/* mgl_ext_buffers.h — private state and Metal hooks of the 'buffers' domain. Owned by agent 'buffers'.
 *
 * Buffer memory model (see MGLRenderer+Draw.m):
 *   Every Buffer that the GPU touches owns an MTLBuffer whose contents ALIAS
 *   Buffer.data.buffer_data (one copy of the data, unified / managed memory).
 *   CPU accesses are ordered against GPU work with mgl_buffers_prepare_cpu_read()
 *   (waits for pending GPU writes) and mgl_buffers_prepare_cpu_write() (waits for /
 *   renames away from pending GPU reads). Other domains that let the GPU read or
 *   write a buffer outside the draw/dispatch paths of this domain should call
 *   mgl_buffers_mark_gpu_use() so later CPU access synchronises correctly.
 */
#ifndef MGL_EXT_BUFFERS_H
#define MGL_EXT_BUFFERS_H

#include <stdint.h>
#include "glm_context.h"
#include "mgl_ext.h"

/* Metal primitive types (values of MTLPrimitiveType) used by the C draw code. */
enum {
    MGL_PRIM_POINT = 0,
    MGL_PRIM_LINE = 1,
    MGL_PRIM_LINE_STRIP = 2,
    MGL_PRIM_TRIANGLE = 3,
    MGL_PRIM_TRIANGLE_STRIP = 4,
};

/* per-buffer GPU usage tracking (open-addressed table keyed by Buffer*) */
typedef struct MGLBufferTrack_t {
    Buffer   *buf;          /* NULL = empty, (Buffer *)1 = tombstone */
    uint64_t  read_serial;  /* last command buffer serial that read the buffer */
    uint64_t  write_serial; /* last command buffer serial that may have written it */
} MGLBufferTrack;

/* GL-side state, reachable as ctx->state.ext_buffers; allocated in mgl_ext_init_buffers(). */
struct MGLExtState_buffers {
    Buffer *parameter_buffer;           /* GL_PARAMETER_BUFFER binding (no slot in STATE(buffers)) */

    MGLBufferTrack *track;              /* GPU usage table */
    uint32_t track_cap;                 /* power of two */
    uint32_t track_used;                /* live + tombstones */

    uint32_t range_mask[4];             /* indexed bindings made with *Range: UBO, SSBO, atomic, xfb */

    Sync **syncs;                       /* live sync objects (for IsSync / validation) */
    uint32_t sync_count, sync_cap;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_buffers; filled by MGLRenderer+Draw.m. */
struct MGLExtMetal_buffers {
    /* command stream */
    uint64_t (*current_serial)(GLMContext ctx);      /* serial of the command buffer being encoded */
    uint64_t (*completed_serial)(GLMContext ctx);    /* highest serial known complete */
    void (*flush)(GLMContext ctx);                   /* commit the current command buffer (if it has work) */
    void (*wait_serial)(GLMContext ctx, uint64_t serial); /* flush if needed and block until serial completed */
    void (*make_cpu_visible)(GLMContext ctx, Buffer *buf, uint64_t serial); /* wait + (discrete GPUs) synchronizeResource */
    void (*barrier)(GLMContext ctx, GLbitfield barriers, bool by_region); /* make shader writes visible to later commands */

    /* buffers */
    bool (*ensure_buffer)(GLMContext ctx, Buffer *buf);  /* create the aliasing MTLBuffer */
    bool (*rename_buffer)(GLMContext ctx, Buffer *buf);  /* give buf fresh storage (copy of contents) */
    void (*did_modify)(GLMContext ctx, Buffer *buf, size_t offset, size_t length); /* CPU wrote range */
    void (*release_buffer)(GLMContext ctx, Buffer *buf); /* drop Metal storage (delete / respecify) */

    /* draws: C side validates, emulates primitive types and calls these */
    bool (*begin_draw)(GLMContext ctx);              /* processGLState + encoder + resource binding */
    void (*draw_arrays)(GLMContext ctx, int prim, GLint first, GLsizei count, GLsizei instances, GLuint base_instance);
    /* index_buf != NULL: indices live in index_buf at byte offset; else cpu_indices (copied) */
    void (*draw_indexed)(GLMContext ctx, int prim, GLenum type, Buffer *index_buf, size_t offset, const void *cpu_indices,
                         GLsizei count, GLsizei instances, GLint base_vertex, GLuint base_instance);
    void (*draw_indirect)(GLMContext ctx, int prim, Buffer *indirect, size_t offset,
                          GLenum type, Buffer *index_buf); /* type == 0 -> arrays */
    void (*end_draw)(GLMContext ctx);                /* stamp buffers used by the draw */

    /* compute */
    bool (*dispatch)(GLMContext ctx, GLuint x, GLuint y, GLuint z, Buffer *indirect, size_t offset);

    /* GPU copies */
    void (*copy_buffer)(GLMContext ctx, Buffer *src, Buffer *dst, size_t src_off, size_t dst_off, size_t size);

    /* sync objects (fence state lives behind Sync.mtl_event) */
    void (*fence)(GLMContext ctx, Sync *sync);
    GLenum (*client_wait)(GLMContext ctx, Sync *sync, bool flush, GLuint64 timeout_ns);
    void (*server_wait)(GLMContext ctx, Sync *sync);
    bool (*is_signaled)(GLMContext ctx, Sync *sync);
    void (*delete_fence)(GLMContext ctx, Sync *sync);
};

#ifdef __cplusplus
extern "C" {
#endif
/* buffer synchronisation helpers (draw_ext.c) — safe to call from any domain */
void mgl_buffers_mark_gpu_use(GLMContext ctx, Buffer *buf, bool gpu_writes);
void mgl_buffers_prepare_cpu_read(GLMContext ctx, Buffer *buf);
void mgl_buffers_prepare_cpu_write(GLMContext ctx, Buffer *buf, bool whole_buffer);
void mgl_buffers_forget(GLMContext ctx, Buffer *buf);
MGLBufferTrack *mgl_buffers_track(GLMContext ctx, Buffer *buf, bool create);
uint32_t *mgl_buffers_range_mask(GLMContext ctx, int slot);
/* one draw without pre-draw / xfb hooks (called back by the xfb capture wrappers) */
void mgl_buffers_raw_draw(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                          GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance);
/* MultiDraw*Indirect[Count] core (draw_buffers.c); type == 0 -> arrays */
void mgl_buffers_multi_draw_indirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount,
                                     GLsizei stride, bool use_count, GLintptr count_offset);
bool mgl_buffers_sync_valid(GLMContext ctx, Sync *sync);
void mgl_buffers_sync_add(GLMContext ctx, Sync *sync);
void mgl_buffers_sync_remove(GLMContext ctx, Sync *sync);
#ifdef __cplusplus
}
#endif

#endif
