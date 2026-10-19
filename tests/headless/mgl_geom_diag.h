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

#ifdef __cplusplus
}
#endif

#endif
