/*
 * compat_fbfetch.h — GL_EXT_shader_framebuffer_fetch lowering declarations
 * (compat_fbfetch.c). Kept separate from mgl_ext_compat.h because that header
 * re-defines the shared ERROR_* macros, which shaders.c / program.c must not see.
 *
 * code layout (Spirv.fbfetch, see compat_fbfetch.c):
 *   bits 0..7   colour locations fetched through `inout' outputs
 *   bits 8..11  gl_LastFragData: 0 = unused, 1..8 = highest literal index + 1,
 *               15 = dynamic index
 *   bit  31     the extension directive was present and enabled
 */
#ifndef MGL_COMPAT_FBFETCH_H
#define MGL_COMPAT_FBFETCH_H

#include <stdint.h>

uint32_t mgl_fbfetch_scan_source(const char *src);   /* Spirv.fbfetch code */
char *mgl_fbfetch_glsl_compat(unsigned type, const char *src);   /* GLSL pre-pass */
char *mgl_fbfetch_fix_msl(const char *msl, const char *entry, uint32_t code);  /* MSL patch */

#endif
