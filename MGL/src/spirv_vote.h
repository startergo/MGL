/*
 * spirv_vote.h — SPV_KHR_subgroup_vote opcode rewrite for the MSL backend.
 *
 * glslang compiles GL_ARB_shader_group_vote calls (allInvocations /
 * anyInvocation / allInvocationsEqual, with or without the ARB suffix) to the
 * legacy SPV_KHR_subgroup_vote opcodes OpSubgroupAllKHR / OpSubgroupAnyKHR /
 * OpSubgroupAllEqualKHR (4428-4430).  SPIRV-Cross's MSL backend does not
 * implement those opcodes: it defers to the GLSL backend, which drops the
 * GLSL-only built-ins allInvocationsARB() / anyInvocationARB() /
 * allInvocationsEqualARB() into the Metal source — identifiers that do not
 * exist in MSL, so the Metal compiler rejects the shader
 * (KHR-GL46.shader_group_vote, all five cases).
 *
 * mgl_spirv_vote_khr_rewrite() rewrites the module to the core
 * OpGroupNonUniformAll/Any/AllEqual instructions (with an ExecutionScope of
 * Subgroup), which the MSL backend lowers to Metal's simd_all() / simd_any()
 * / spvSubgroupAllEqual().  Per ARB_shader_group_vote the "SIMD invocation
 * group" is implementation-defined; Metal's 32-lane simdgroup is a legal
 * group and satisfies the only guarantees the extension makes
 * (anyInvocation true when <value> is true, allInvocations false when
 * <value> is false, one answer for the whole group).
 */
#ifndef MGL_SPIRV_VOTE_H
#define MGL_SPIRV_VOTE_H

#include <stddef.h>
#include <stdint.h>

/* Rewrites OpSubgroupAllKHR / OpSubgroupAnyKHR / OpSubgroupAllEqualKHR to
 * OpGroupNonUniform{All,Any,AllEqual} with a Subgroup ExecutionScope.
 * Returns a malloc'd module, or NULL when the module has no such opcodes
 * (or cannot be walked — SPIRV-Cross then reports the original error).  */
uint32_t *mgl_spirv_vote_khr_rewrite(const uint32_t *in, size_t n, size_t *out_n);

#endif /* MGL_SPIRV_VOTE_H */
