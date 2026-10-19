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

/* enable (1) / disable (0) the GPU-time instrumentation: sums
 * GPUEndTime - GPUStartTime over every tess command buffer and counts them.
 * Read after glFinish (the completion handlers have run by then). */
void mgl_tess_diag_enable(int enable);
void mgl_tess_gpu_time_reset(void);
double mgl_tess_gpu_time_us(void);
unsigned mgl_tess_cbs_completed(void);

#endif
