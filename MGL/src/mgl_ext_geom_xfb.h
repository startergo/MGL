/* mgl_ext_geom_xfb.h — GPU pack of the geometry output into the transform feedback buffers.
 * Owned by agent 'geom' (part of the geometry emulation).
 *
 * Three small kernels run after the geometry kernel, in the same command buffer:
 *   1. a prefix sum over the per-thread CLOSED-PRIMITIVE counts (read from the record
 *      tags), giving every captured primitive its transform feedback index - GL records
 *      whole primitives, not loose vertices,
 *   2. a scatter writing every vertex of each closed primitive into the bound transform
 *      feedback buffers (a strip output decomposes into its line / triangle primitives,
 *      so shared strip vertices are captured once per primitive, as GL does), dropping
 *      primitives that do not fit entirely,
 *   3. a finish kernel advancing the write offsets by the captured vertices and folding the
 *      stream-0 primitive count in.
 * GL captures EVERY output stream (only stream 0 rasterizes): when the captured varyings
 * name more than one stream, 1+2 run once per captured stream, each counting only that
 * stream's records (the ordinals the geometry kernel stamps in the tags are per stream),
 * and writing only the buffers whose varyings belong to it.
 * The write offsets and the "since the last resolve" counters live in a small GPU block, so
 * no per-draw synchronisation happens; the CPU folds them in lazily at the points where the
 * application can observe the state (glEndTransformFeedback, pause/resume, binding another
 * object, ending or reading back a transform feedback / primitives-generated query). */
#ifndef MGL_EXT_GEOM_XFB_H
#define MGL_EXT_GEOM_XFB_H

#include "glm_context.h"
#include "mgl_ext_geom.h"

#define MGL_GEOM_XFB_MAX_VARS  32
/* geom r105d: 9 words — src_word, dst_off, words, buffer, stream, and the
 * padded-double-column copy spec (col_w0/col_w1/col_sstride/col_dstride;
 * col_dstride == 0 selects the plain contiguous copy) */
#define MGL_GEOM_XFB_WG        64

/* the GPU counter block: [0] object name, [1] vertices (all streams, block total),
 * [2] primitives (stream 0, block total), [3..6] write offsets, [8] the tess path's
 * per-draw vertex counter, [9..12] the per-draw captured vertices PER STREAM (cleared by
 * the finish kernel of the draw that produced them), [13..16] the block-total primitives
 * WRITTEN per stream, [17..20] the block-total primitives EMITTED per stream ("needed",
 * what GL_TRANSFORM_FEEDBACK_OVERFLOW compares against written), [21..24] the block-total
 * captured vertices per stream.  It is owned by the kernels while a pack is pending, and
 * only read by the CPU at a resolve (after a wait), so the CPU never races the GPU on it.
 * The per-draw inputs and the varying table live in a separate ring, one slot per draw in
 * flight, for the same reason. */
#define MGL_GEOM_XFB_BLOCK_WORDS  25
/* [0] threads, [1] maxv, [2] out_words, [3] nvars, [4..7] stride[4], [8..11] size[4],
 * [12] feedback-mode vertices per primitive, [13] geometry output type (0/1/2),
 * [14..17] per-buffer captured extent in bytes (max offset+size of its varyings),
 * [20] the stream this prefix/scatter pass packs, [21] the stream of each buffer
 * (2 bits per buffer: a buffer's write offset advances by ITS stream's vertices),
 * [22] the bitmask of the buffers this pass writes (the capacity test) */
#define MGL_GEOM_XFB_IN_WORDS     24
#define MGL_GEOM_XFB_VAR_WORDS    7
#define MGL_GEOM_XFB_SLOT_WORDS   (MGL_GEOM_XFB_IN_WORDS + MGL_GEOM_XFB_MAX_VARS * MGL_GEOM_XFB_VAR_WORDS)
#define MGL_GEOM_XFB_RING         4    /* draws in flight */

#ifdef __cplusplus
extern "C" {
#endif

/* try to encode the GPU pack for one geometry draw; false when the CPU pack must be used
 * (no Metal buffers for the transform feedback targets, or no varyings captured) */
bool mgl_geom_xfb_pack(GLMContext ctx, MGLGeomProgram *gp, struct Program_t *program,
                       uint32_t threads, uint32_t max_vertices, uint32_t out_words,
                       uint32_t vertex_cap);/* fold the pending GPU counters into the transform feedback object (waits for the command
 * buffer that produced them) and notify the transform feedback queries */
void mgl_geom_xfb_flush(GLMContext ctx);
/* a pack is pending: state the application may read is not up to date yet */
bool mgl_geom_xfb_pending(GLMContext ctx);
/* the bound transform feedback object is active but paused: nothing may be captured */
bool mgl_geom_xfb_paused(GLMContext ctx);
/* test-facing trace helpers (declared for the harness in tests/headless/mgl_geom_diag.h,
 * so no library header needs to expose them): log the buffer-tracking state (write /
 * completed serial) at a CPU read of a transform feedback buffer */
void mgl_geom_xfb_log_read_name(void *ctx, unsigned buf_name, const char *what);
void *mgl_geom_ctx(void);

#ifdef __cplusplus
}
#endif

#endif
