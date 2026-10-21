/*
 * MGL software fp64 — SPIR-V lowering pass.
 *
 * Apple GPUs have no fp64.  MGL implements GL 4.6's mandatory doubles the way
 * Mesa's Asahi driver does (nir_lower_fp64_full_software), but one level up:
 * at the SPIR-V level, because MGL's pipeline is
 * GLSL -> glslang -> SPIR-V -> (this pass) -> SPIRV-Cross -> MSL.
 *
 * Representation: a 'double' value becomes an OpTypeInt(64, 0) whose bits are
 * the IEEE-754 double pattern.  SPIRV-Cross maps that to MSL 'ulong', and
 * 64-bit integer arithmetic works on Apple GPUs (verified on M4 Pro).  Every
 * fp64 SPIR-V instruction is rewritten into an OpFunctionCall into the
 * soft-float64 library (MGL/src/fp64_softfloat.glsl: Berkeley SoftFloat 3e as
 * ported to GLSL by Mesa, plus composed entry points ported from Mesa's
 * nir_lower_double_ops.c).
 *
 * Why the bits-are-the-type trick works out:
 *   - OpConstant double needs no edit (the same literal words are a u64).
 *   - OpCompositeConstruct/Extract/Insert, OpPhi, OpSelect, OpCopyObject,
 *     OpVectorShuffle, loads/stores and access chains need no edit: component
 *     i of a dvecN is element i of the mapped ulongN, and the only structural
 *     change (matrix -> array of columns) keeps index order.
 *   - std140/std430 byte layouts of doubles (8/8, 16/16, 24|32/32, stride 32)
 *     are exactly SPIRV-Cross's MSL layout rules for ulong/ulongN vectors.
 *
 * Interface (stage in/out) variables are the exception: MSL has no vertex
 * format that can feed a ulongN attribute, so double vectors cross stage
 * boundaries as arrays of uint4 (two doubles = one uint4 = one 128-bit GL
 * location): dvec3 is array[2] of uint4 (2 locations, as GL requires) with a
 * bitcast-based conversion at every load/store.
 *
 * Row-major double matrices in buffer blocks are stored as arrays of row
 * vectors with the access-chain indices transposed.
 */
#ifndef MGL_FP64_H
#define MGL_FP64_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Lower every fp64 operation in a SPIR-V module.
 *
 * Returns a freshly malloc()ed module (caller frees), sets *lowered:
 *   false — module contains no fp64 type, input is returned untouched
 *   true  — module was rewritten (NULL + log filled on failure)
 */
uint32_t *mgl_fp64_lower_module(const uint32_t *words, size_t nwords, bool vertex_inputs,
                                bool *lowered, size_t *out_nwords, char *log, size_t logsz);

/*
 * Post-fix for MSL compiled from an fp64-lowered module.
 *
 * SPIRV-Cross renders every scalar (and vector) function parameter of the
 * soft-float library as `thread const T&`.  A `constant`-space buffer member
 * (default-block uniform, UBO/SSBO member) cannot bind to a thread reference,
 * so Metal rejects call sites like `_4499(mgl_DefaultUniforms_1.uniform_0, ..)`
 * with "no matching function for call to ...".  SPIRV-Cross copy-propagates a
 * materialized load straight back into the member expression (its C API
 * exposes no forced_temporaries), so the fix must be applied to the emitted
 * MSL: every `thread const T&` parameter becomes `T` by value.  That is
 * exactly the GLSL source semantics (the library declares every parameter by
 * value) and the SPIR-V semantics (OpFunctionCall arguments are rvalues, a
 * const ref can never be mutated), so it is safe for user functions of the
 * module too.  `device const`/`constant const` parameters are left alone.
 *
 * Returns a malloc()ed fixed copy (caller frees); NULL on allocation failure.
 */
char *mgl_fp64_fix_msl_scalar_params(const char *msl);

#endif /* MGL_FP64_H */
