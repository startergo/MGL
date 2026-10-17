/* mgl_ext_gpuaudit.h — private state and Metal hooks of the 'gpuaudit' domain.
 *
 * This domain owns the machinery that keeps CPU work off the GL command path:
 *   - a render-pipeline-state cache keyed on the MTLRenderPipelineDescriptor
 *     contents (the Metal side lives in MGLRenderer+Gpuaudit.m),
 *   - caches for the CPU work the draw path used to redo on every call:
 *     the primitive-restart index scan and the rewritten / emulated / widened
 *     index lists (which are cached as MTLBuffers so a repeated draw encodes
 *     nothing on the CPU),
 *   - GPU implementations of buffer clears (blit fill / staged copies) and of
 *     glReadPixels into a PIXEL_PACK_BUFFER (texture -> buffer blit, no stall).
 *
 * Cache soundness: results are keyed on the *content generation* of the source
 * buffer.  Every path that changes a Buffer's CPU-visible contents must call
 * mgl_gpuaudit_invalidate() (see buffers.c bufferDidModify / initBufferData,
 * draw_ext.c mgl_buffers_prepare_cpu_read, queries.c).  Draws whose index
 * source is a persistently mapped buffer are never cached.
 */
#ifndef MGL_EXT_GPUAUDIT_H
#define MGL_EXT_GPUAUDIT_H

#include <stdint.h>
#include <stdbool.h>
#include "glm_context.h"
#include "mgl_ext.h"

#define MGL_GA_INDEX_CACHE 8      /* rewritten index lists kept as MTLBuffers */
#define MGL_GA_SCAN_CACHE  8      /* cached primitive-restart scan answers */
#define MGL_GA_SERIAL_CACHE 64    /* Buffer* -> content generation */

/* snapshot of the GL state that can change a rewrite decision or the rewritten list:
   the restart index (and whether restart is on at all), the provoking-vertex
   convention, and the bound program (flat-varying detection is per program). */
typedef struct MGLIndexKeyState {
    uint32_t  restart_index;
    uint8_t   restart_on;
    uint8_t   lastpv;
    void     *program;
} MGLIndexKeyState;

/* one cached "this draw needs an index rewrite: yes/no" answer */
typedef struct MGLScanCacheEnt {
    Buffer   *ebuf;             /* element buffer (NULL = never valid) */
    uint64_t  content;          /* its content generation when cached */
    uint32_t  offset;           /* byte offset of the index range */
    GLenum    mode, type;
    GLsizei   count;
    MGLIndexKeyState state;     /* GL state snapshot the answer was computed under */
    uint8_t   needed;           /* 1: rewrite needed, 0: not */
    uint64_t  lru;
} MGLScanCacheEnt;

/* one cached rewritten / emulated / widened index list, resident in an MTLBuffer */
typedef struct MGLIndexCacheEnt {
    Buffer   *ebuf;             /* element buffer, or NULL for glDrawArrays emulation */
    uint64_t  content;          /* content generation of ebuf (0 when ebuf == NULL) */
    uint32_t  offset;
    GLenum    mode, type;       /* GL draw parameters the list was built from */
    GLsizei   count;
    GLint     first;            /* arrays only */
    MGLIndexKeyState state;     /* GL state snapshot the list was built under */
    uint8_t   adjacency;        /* adjacency-mode emulation */
    /* result */
    GLenum    out_mode, out_type;
    GLsizei   out_count;
    void     *mtl_buffer;       /* retained id<MTLBuffer> holding the indices */
    uint64_t  lru;
} MGLIndexCacheEnt;

/* Buffer* -> content generation table (open addressing, generation never reused) */
typedef struct MGLSerialEnt {
    Buffer   *buf;              /* NULL = free */
    uint64_t  serial;
} MGLSerialEnt;

/* GL-side state, reachable as ctx->state.ext_gpuaudit; allocated in mgl_ext_init_gpuaudit(). */
struct MGLExtState_gpuaudit {
    MGLScanCacheEnt  scan[MGL_GA_SCAN_CACHE];
    MGLIndexCacheEnt index[MGL_GA_INDEX_CACHE];
    MGLSerialEnt    *serial;            /* open-addressed Buffer* -> generation table */
    uint32_t         serial_cap;        /* power of two, or 0 */
    uint64_t         serial_next;       /* fresh generation values */
    uint64_t         lru;

    /* statistics (exposed through glGetInteger*(GL_..._MGL_GPUAUDIT), see below) */
    uint32_t pso_hits, pso_misses;
    uint32_t scan_hits, scan_misses;
    uint32_t index_hits, index_misses;
    uint32_t pbo_readbacks, gpu_clears;
};

/* Metal hooks, reachable as ctx->mtl_funcs.ext_gpuaudit; filled by MGLRenderer+Gpuaudit.m. */
struct MGLExtMetal_gpuaudit {
    /* render pipeline states, keyed on descriptor content */
    void *(*pso_lookup)(GLMContext ctx, void *descriptor);           /* retained PSO or NULL */
    void  (*pso_store)(GLMContext ctx, void *descriptor, void *ps);  /* takes ownership of the reference */

    /* clear a buffer range with a repeating texel pattern on the GPU (async) */
    bool (*fill_buffer)(GLMContext ctx, Buffer *buf, GLintptr offset, GLsizeiptr size,
                        const uint8_t *texel, int texel_size);

    /* glReadPixels with a PIXEL_PACK_BUFFER bound: blit the read surface straight
     * into the PBO.  Returns false when the surface layout does not match
     * (format, type, pack parameters), in which case the CPU path runs. */
    bool (*read_pixels_to_buffer)(GLMContext ctx, Framebuffer *fbo, int aspect, GLenum attachment,
                                  GLint x, GLint y, GLsizei width, GLsizei height,
                                  GLenum format, GLenum type, Buffer *pbo, size_t pbo_offset,
                                  size_t row_bytes, size_t skip_bytes);

    /* draw with indices that already live in an MTLBuffer (a cached rewrite) */
    void (*draw_indexed_mtl)(GLMContext ctx, int prim, GLenum type, void *mtl_buffer, size_t offset,
                             GLsizei count, GLsizei instances, GLint base_vertex, GLuint base_instance);

    /* upload an index list into an MTLBuffer owned by the caller-free cache */
    void *(*index_buffer_new)(GLMContext ctx, const void *data, size_t bytes);  /* retained MTLBuffer or NULL */
    void  (*index_buffer_free)(GLMContext ctx, void *mtl_buffer);
};

/* glGet pnames handled by mgl_ext_get_gpuaudit() (all GLint counts).
 * GL_*_MGL_* vendor names, deliberately outside the GL core enum space. */
#define MGL_GPUAUDIT_PSO_HITS      0x8F00
#define MGL_GPUAUDIT_PSO_MISSES    0x8F01
#define MGL_GPUAUDIT_SCAN_HITS     0x8F02
#define MGL_GPUAUDIT_SCAN_MISSES   0x8F03
#define MGL_GPUAUDIT_INDEX_HITS    0x8F04
#define MGL_GPUAUDIT_INDEX_MISSES  0x8F05
#define MGL_GPUAUDIT_PBO_READBACKS 0x8F06
#define MGL_GPUAUDIT_GPU_CLEARS    0x8F07

#ifdef __cplusplus
extern "C" {
#endif

/* --- content generation -------------------------------------------------- */
/* every path that makes new CPU-visible contents of `buf` must call this */
void mgl_gpuaudit_invalidate(GLMContext ctx, Buffer *buf);
uint64_t mgl_gpuaudit_content(GLMContext ctx, Buffer *buf);   /* generation, or 0 when unknown */
void mgl_gpuaudit_invalidate_all(GLMContext ctx);             /* drop every cache entry */

/* --- primitive-restart scan cache ---------------------------------------- */
/* returns -1 when unknown: the caller must scan. Otherwise 1 (rewrite needed)
 * or 0 (not needed) for exactly this draw's index range and GL state. */
int mgl_gpuaudit_scan_lookup(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode, GLenum type,
                             GLsizei count, MGLIndexKeyState state);
void mgl_gpuaudit_scan_store(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode, GLenum type,
                             GLsizei count, MGLIndexKeyState state, int needed);

/* --- rewritten index list cache ------------------------------------------ */
/* look up a cached rewritten list for exactly this draw; returns the entry
 * (borrowed) with out_mode/out_type/out_count/mtl_buffer filled in, or NULL. */
MGLIndexCacheEnt *mgl_gpuaudit_index_lookup(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode,
                                            GLenum type, GLsizei count, GLint first,
                                            MGLIndexKeyState state, int adjacency);
/* store a rewritten list.  On success the cache owns an MTLBuffer copy of `data`
 * and the returned entry is ready to draw from; on failure it returns NULL and
 * `data` is untouched, so the caller keeps drawing from its own copy. */
MGLIndexCacheEnt *mgl_gpuaudit_index_store(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode,
                                           GLenum type, GLsizei count, GLint first,
                                           MGLIndexKeyState state, int adjacency,
                                           void *data, GLenum out_mode, GLenum out_type, GLsizei out_count);

/* hook the caches need: is this buffer's CPU view safe to cache (not mapped)? */
bool mgl_gpuaudit_cacheable(GLMContext ctx, Buffer *ebuf);

void mgl_ext_init_gpuaudit(GLMContext ctx);
bool mgl_ext_get_gpuaudit(GLMContext ctx, GLenum pname, GLuint type, void *data);
#ifdef __cplusplus
}
#endif

#endif
