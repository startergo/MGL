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
 * The write offsets and the "since the last resolve" counters live in a small GPU block, so
 * no per-draw synchronisation happens; the CPU folds them in lazily at the points where the
 * application can observe the state (glEndTransformFeedback, pause/resume, binding another
 * object, ending or reading back a transform feedback / primitives-generated query). */
#ifndef MGL_EXT_GEOM_XFB_H
#define MGL_EXT_GEOM_XFB_H

#include "glm_context.h"
#include "mgl_ext_geom.h"

#define MGL_GEOM_XFB_MAX_VARS  32
#define MGL_GEOM_XFB_WG        64

/* the GPU counter block: [0] object name, [1] vertices, [2] primitives, [3..6] write
 * offsets, [8] the per-draw vertex counter.  It is owned by the kernels while a pack is
 * pending, and only read by the CPU at a resolve (after a wait), so the CPU never races the
 * GPU on it.
 * The per-draw inputs and the varying table live in a separate ring, one slot per draw in
 * flight, for the same reason. */
#define MGL_GEOM_XFB_BLOCK_WORDS  9
/* [0] threads, [1] maxv, [2] out_words, [3] nvars, [4..7] stride[4], [8..11] size[4],
 * [12] feedback-mode vertices per primitive, [13] geometry output type (0/1/2),
 * [14..17] per-buffer captured extent in bytes (max offset+size of its varyings) */
#define MGL_GEOM_XFB_IN_WORDS     20
#define MGL_GEOM_XFB_VAR_WORDS    4
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
