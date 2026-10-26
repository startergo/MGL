/*
 * msl_reserved.h — MSL reserved-identifier mangling for SPIRV-Cross.
 *
 * GLSL allows user identifiers (`uniform sampler2D sampler;`, a struct member named
 * `texture`, a function named `saturate`, ...) that are keywords, built-in type names or
 * Metal std-lib function names in MSL.  SPIRV-Cross copies SPIR-V names verbatim into the
 * MSL it emits (its own reserved-name fixup only covers a small keyword/macro list, not the
 * MSL types), so the Metal compiler rejects such shaders ("must use 'struct' tag to refer
 * to type 'sampler'", ambiguous reference, ...).
 *
 * mgl_msl_mangle_reserved() renames every *reserved* OpName / OpMemberName of a module
 * inside a live SPIRV-Cross compiler (suffix `_mgl`), before MSL emission.  It must run
 * AFTER GL introspection: GL introspection has to keep reporting the GLSL names, and it
 * reads the same compiler's names.
 */
#ifndef MGL_MSL_RESERVED_H
#define MGL_MSL_RESERVED_H

#include <stddef.h>
#include <stdint.h>
#include "spirv_cross_c.h"

/* true when `name` is an MSL keyword, built-in type or Metal std-lib function name that a
 * GLSL shader is allowed to use as an identifier but MSL is not. */
int mgl_msl_is_reserved_name(const char *name);

/* Rename every reserved OpName / OpMemberName of `words` (the same module `compiler` was
 * created from) to `<name>_mgl<N>` in the compiler.  Renames never touch `gl_`-prefixed
 * names and are chosen not to collide with any other identifier of the module. */
void mgl_msl_mangle_reserved(spvc_compiler compiler, const uint32_t *words, size_t nwords);

/*
 * SPIRV-Cross flattens a BuiltInCullDistance stage OUTPUT into individual
 * `float gl_CullDistance_N [[user(cullN)]]' struct members but — unlike
 * BuiltInClipDistance (spirv_msl.cpp: the [[clip_distance]] array special case) —
 * leaves the shader's stores pointing at the non-existent `out.gl_CullDistance[...]'
 * array and appends an "unflatten" tail reading an undeclared block local:
 *
 *     out.gl_CullDistance[i] = d[i];                                  // no array member
 *     out.gl_CullDistance_0 = _15._RESERVED_IDENTIFIER_FIXUP_gl_CullDistance[0];
 *
 * mgl_msl_fix_cull_distance() repairs the emitted MSL text of a vertex-function stage:
 * constant-index stores/loads are redirected to the flattened members, dynamic ones go
 * through a local array that is copied back before `return out', and the broken tail is
 * dropped.  Returns a malloc'd corrected copy, or NULL when nothing needed fixing.
 */
char *mgl_msl_fix_cull_distance(const char *msl);

#endif
