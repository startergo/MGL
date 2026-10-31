/* mgl_tess_diag.h — test-only diagnostics of the MGL tessellation domain.
 *
 * The counters are compiled into the library unconditionally; the GPU-time
 * instrumentation must be switched on once (mgl_tess_diag_enable) because it
 * attaches a completion handler to every tess command buffer. With diagnostics
 * off there is no per-draw cost beyond the plain counter increments, which only
 * happen on the (rare) CPU-side paths they measure.
 *
 * Counters are process-global (one GL context per test process). */
#ifndef MGL_TESS_DIAG_H
#define MGL_TESS_DIAG_H

#include <stdint.h>
#include <stdbool.h>

/* CPU-side work that the GPU paths must not do (all since library load):
 *   cpu_blob_builds    — per-patch data blobs assembled on the CPU
 *   cpu_generate_runs  — full CPU tessellations
 *   scan_waits         — GPU index-scan waits (must stay 0: round 14)
 *   cpu_index_reads    — element buffers read on the CPU (restart gapless build)
 *   restart_checks     — restart-presence scans (once per buffer content: round 15) */
unsigned mgl_tess_cpu_blob_builds(void);
unsigned mgl_tess_cpu_generate_runs(void);
unsigned mgl_tess_scan_waits(void);
unsigned mgl_tess_cpu_index_reads(void);
unsigned mgl_tess_restart_checks(void);

/* ---- tests-only capture access (see the MGLTessCapture contract) ----
 * mgl_tess_diag_capture enables TES-output capture for a program; after a
 * glFinish(), mgl_tess_diag_capture_info copies the published capture struct and
 * the GPU-written vertex count (the caller then reads records straight from the
 * Metal buffer, which is CPU-visible). */
void mgl_tess_diag_enable(int enable);
void mgl_tess_gpu_time_reset(void);
double mgl_tess_gpu_time_us(void);
unsigned mgl_tess_cbs_completed(void);
void mgl_tess_diag_capture(GLuint program, int enable);
bool mgl_tess_diag_capture_info(void **buffer, uint32_t *records_offset, uint32_t *stride_words,
                                uint32_t *prim, uint32_t *count);

/* Run the CPU (D3D11 reference) tessellator for ONE patch and return its vertex
 * and index counts: the emulated GPU stream and the query counters must produce
 * exactly these.  domain/spacing use MGL_TESS_DOMAIN_* / MGL_TESS_SPACING_*.
 * Test-only. */
bool mgl_tess_diag_cpu_counts(int domain, int spacing, int ccw, int point_mode,
                              const float levels[6], uint32_t *nvertices, uint32_t *nindices);

/* ... and the full stream: coords is 4 floats per vertex (u, v, w, 0), indices
 * the emitted primitive list (or the point list for point_mode).  Returns the
 * vertex count, -1 if the buffers are too small. */
int mgl_tess_diag_cpu_mesh(int domain, int spacing, int ccw, int point_mode,
                           const float levels[6], float *coords, int max_coords,
                           uint32_t *indices, int max_indices, int *out_index_count);

/* ---- r43 ring health / CPU-phase instrumentation --------------------------
 * mgl_tess_ring_spins: acquires that had to poll for a slot, plus the worst
 *   poll count and total time slept (a healthy hot loop stays at or near zero).
 * mgl_tess_ring_freed_big: blocks above 4 MB freed on retire instead of being
 *   pinned in a slot (the ring's memory bound; see TessRingBigBlockRecycle).
 * mgl_tess_phase_reset/us/cnt: arms per-phase CPU-time accounting of the
 *   hardware draw path (phases setup/index/ring/vsbind/gather/tcs/notcs/blob/
 *   order/render/post/total); zero cost until armed. */
unsigned mgl_tess_ring_spins(uint64_t *max_ns, uint64_t *total_ns);
unsigned long mgl_tess_ring_freed_big(void);
unsigned mgl_tess_ring_big_slots(void);
void mgl_tess_phase_reset(void);
double mgl_tess_phase_us(int phase);
uint64_t mgl_tess_phase_cnt(int phase);

#endif
