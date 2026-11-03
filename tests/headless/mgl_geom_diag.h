/* mgl_geom_diag.h — test-facing diagnostics of the geometry domain (mirrors
 * mgl_tess_diag.h).  These are trace helpers only: they print the buffer-tracking
 * state of a transform feedback buffer at a CPU read.  The library exports them so the
 * harness needs no private header; nothing else should call them. */
#ifndef MGL_GEOM_DIAG_H
#define MGL_GEOM_DIAG_H

#ifdef __cplusplus
extern "C" {
#endif

void *mgl_geom_ctx(void);
void mgl_geom_xfb_log_read_name(void *ctx, unsigned buf_name, const char *what);

/* r82: monotonic image-atomic sidecar counters of the current context.
 * which: 0 seeds, 1 flushes (MS raster), 2 compute resolves, 3 sidecar
 * allocations.  Used to prove the sidecar's zero cost when a program declares
 * no image atomics (and that each pass runs exactly once when one does). */
uint64_t mgl_geom_ms_stat(int which);

/* r100: capture-pass setup-cache counters (hit / miss) of the current process —
 * proves the cache engages on steady state and misses on state changes. */
uint64_t mgl_geom_capcache_stat(int which);

#ifdef __cplusplus
}
#endif

#endif
