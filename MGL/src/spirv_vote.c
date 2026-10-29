/*
 * spirv_vote.c — see spirv_vote.h.
 *
 * Rewrites the SPV_KHR_subgroup_vote AND SPV_KHR_shader_ballot opcodes that
 * glslang emits for GL_ARB_shader_group_vote / GL_ARB_shader_ballot into the
 * SPIR-V 1.3 OpGroupNonUniform* core forms, which are the only ones the
 * SPIRV-Cross MSL backend lowers (to Metal simdgroup operations):
 *
 *   OpSubgroupAllKHR/AnyKHR/AllEqualKHR  -> OpGroupNonUniformAll/Any/AllEqual
 *   OpSubgroupBallotKHR                  -> OpGroupNonUniformBallot
 *   OpSubgroupReadInvocationKHR          -> OpGroupNonUniformShuffle
 *   OpSubgroupFirstInvocationKHR         -> OpGroupNonUniformBroadcast lane 0
 *
 * (the KHR forms take no ExecutionScope operand, so each rewrite inserts the
 * %uint 3 (Subgroup) scope constant; the broadcast form additionally needs a
 * lane id, which we take as an existing %uint 0 constant or synthesize one).
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "spirv.h"
#include "spirv_vote.h"

/* growable SPIR-V word buffer */
typedef struct
{
    uint32_t *w;
    size_t n, cap;
    bool oom;
} VoteWordBuf;

static void vb_push(VoteWordBuf *b, uint32_t v)
{
    if (b->n == b->cap)
    {
        size_t ncap = b->cap ? b->cap * 2 : 1024;
        uint32_t *nw = (uint32_t *)realloc(b->w, ncap * sizeof(uint32_t));
        if (!nw)
        {
            b->oom = true;
            return;
        }
        b->w = nw;
        b->cap = ncap;
    }
    b->w[b->n++] = v;
}

static void vb_push_n(VoteWordBuf *b, const uint32_t *w, size_t n)
{
    for (size_t i = 0; i < n; i++)
        vb_push(b, w[i]);
}

static bool is_type_or_constant(uint32_t op)
{
    return (op >= SpvOpTypeVoid && op <= SpvOpTypeForwardPointer) ||   /* 19..32 */
           op == SpvOpConstant || op == SpvOpConstantTrue || op == SpvOpConstantFalse ||
           op == SpvOpConstantComposite || op == SpvOpConstantSampler ||
           op == SpvOpConstantNull ||
           (op >= SpvOpSpecConstantTrue && op <= SpvOpSpecConstantOp) ||  /* 41..53 */
           op == SpvOpUndef;
}

/* the scope constant's value: 3 = Subgroup (SPIR-V "Scope" enumeration) */
#define MGL_SCOPE_SUBGROUP 3u

uint32_t *mgl_spirv_vote_khr_rewrite(const uint32_t *in, size_t n, size_t *out_n)
{
    uint32_t bound;
    size_t pos;
    bool have_votes = false;
    bool have_ballot = false;       /* SPV_KHR_shader_ballot opcodes (ARB_shader_ballot) */
    bool have_shuffle = false;      /* readInvocationARB / readFirstInvocationARB */
    uint32_t uint_type_id = 0;      /* %uint (OpTypeInt 32 0), synthesized if absent */
    uint32_t scope_id = 0;          /* OpConstant %uint 3 */
    uint32_t lane0_id = 0;          /* OpConstant %uint 0 (broadcast lane for readFirstInvocationARB) */
    size_t insert_pos = 0;          /* end of the types/constants section */
    VoteWordBuf b = {0};
    uint32_t *result = NULL;

    if (!in || n < 5 || in[0] != SpvMagicNumber || !out_n)
        return NULL;
    bound = in[3];

    /* pass 1: scan the instruction stream */
    bool in_functions = false;         /* past the last types/constants instruction */
    for (pos = 5; pos < n; )
    {
        uint32_t op = in[pos] & 0xFFFFu, wc = in[pos] >> 16;
        if (wc == 0 || pos + wc > n)
            return NULL;                      /* malformed: leave the module alone */

        switch (op)
        {
            case SpvOpSubgroupAllKHR:
            case SpvOpSubgroupAnyKHR:
            case SpvOpSubgroupAllEqualKHR:
                have_votes = true;
                break;
            case SpvOpSubgroupBallotKHR:
                have_ballot = true;
                break;
            case SpvOpSubgroupReadInvocationKHR:
                have_ballot = true;
                have_shuffle = true;    /* -> OpGroupNonUniformShuffle */
                break;
            case SpvOpSubgroupFirstInvocationKHR:
                have_ballot = true;
                have_shuffle = true;    /* -> OpGroupNonUniformBroadcast lane 0 */
                break;
            case SpvOpTypeInt:
                if (wc == 4 && in[pos + 2] == 32 && in[pos + 3] == 0 && !uint_type_id)
                    uint_type_id = in[pos + 1];
                break;
            case SpvOpConstant:
                /* a constant of the uint type with value 3 can serve as the scope */
                if (wc == 4 && uint_type_id && in[pos + 1] == uint_type_id &&
                    in[pos + 3] == MGL_SCOPE_SUBGROUP && !scope_id)
                    scope_id = in[pos + 2];
                /* ... and a 0 of the uint type is the broadcast lane */
                if (wc == 4 && uint_type_id && in[pos + 1] == uint_type_id &&
                    in[pos + 3] == 0 && !lane0_id)
                    lane0_id = in[pos + 2];
                break;
            default:
                break;
        }

        /* the insertion point is the end of the types/constants section: after the
         * LAST type / constant / undef instruction, before the first function
         * definition (SPIR-V 2.4 logical layout).  glslang interleaves global
         * OpVariables with the types, so those do not end the section — but a
         * type defined after a global must precede the synthesized constant. */
        if (!in_functions)
        {
            if (is_type_or_constant(op))
                insert_pos = pos + wc;
            else if (op == SpvOpFunction)
                in_functions = true;
        }

        pos += wc;
    }

    if (!have_votes && !have_ballot)
        return NULL;                   /* nothing to rewrite */

    if (!insert_pos)
        insert_pos = n;                /* no functions at all: append */

    /* ids of the synthesized instructions (bound is the first free id) */
    uint32_t new_uint_type = 0, new_scope_const = 0, new_lane0_const = 0;

    if (!scope_id || (have_shuffle && !lane0_id))
    {
        if (!uint_type_id)
        {
            new_uint_type = bound++;
            if (!scope_id)
                new_scope_const = bound++;
            if (have_shuffle && !lane0_id)
                new_lane0_const = bound++;
        }
        else
        {
            if (!scope_id)
                new_scope_const = bound++;
            if (have_shuffle && !lane0_id)
                new_lane0_const = bound++;
        }
        if (!scope_id)
            scope_id = new_scope_const;      /* the operand every rewritten opcode uses */
        if (have_shuffle && !lane0_id)
            lane0_id = new_lane0_const;
    }
    uint32_t new_bound = bound;         /* bound now includes every synthesized id */

    /* pass 2: rebuild */
    vb_push_n(&b, in, 3);                       /* magic, version, generator */
    vb_push(&b, new_bound);                     /* patched bound */
    vb_push(&b, in[4]);                         /* schema */
    if (b.oom)
        goto done;

    for (pos = 5; pos < n; )
    {
        uint32_t op = in[pos] & 0xFFFFu, wc = in[pos] >> 16;

        if (pos == insert_pos)
        {
            /* the synthesized %uint type and the scope / lane-0 constants */
            if (new_uint_type)
            {
                vb_push(&b, (4u << 16) | SpvOpTypeInt);
                vb_push(&b, new_uint_type);
                vb_push(&b, 32);
                vb_push(&b, 0);
            }
            if (new_scope_const)
            {
                vb_push(&b, (4u << 16) | SpvOpConstant);
                vb_push(&b, uint_type_id ? uint_type_id : new_uint_type);
                vb_push(&b, new_scope_const);
                vb_push(&b, MGL_SCOPE_SUBGROUP);
            }
            if (new_lane0_const)
            {
                vb_push(&b, (4u << 16) | SpvOpConstant);
                vb_push(&b, uint_type_id ? uint_type_id : new_uint_type);
                vb_push(&b, new_lane0_const);
                vb_push(&b, 0);
            }
            if (b.oom)
                goto done;
        }

        switch (op)
        {
            case SpvOpSubgroupAllKHR:
            case SpvOpSubgroupAnyKHR:
            case SpvOpSubgroupAllEqualKHR:
            {
                /* OpSubgroup*KHR: <result_type> <result_id> <value>
                 * OpGroupNonUniform*: <result_type> <result_id> <scope> <value> */
                uint32_t new_op = (op == SpvOpSubgroupAllKHR) ? SpvOpGroupNonUniformAll :
                                  (op == SpvOpSubgroupAnyKHR) ? SpvOpGroupNonUniformAny :
                                                                SpvOpGroupNonUniformAllEqual;

                vb_push(&b, ((wc + 1u) << 16) | new_op);
                vb_push_n(&b, &in[pos + 1], 2);             /* result_type, result_id */
                vb_push(&b, scope_id);                      /* ExecutionScope = Subgroup */
                vb_push_n(&b, &in[pos + 3], wc - 3);        /* <value> */
                break;
            }

            case SpvOpSubgroupBallotKHR:
            {
                /* OpSubgroupBallotKHR: <result_type> <result_id> <predicate>
                 * OpGroupNonUniformBallot: <result_type> <result_id> <scope> <predicate> */
                vb_push(&b, ((wc + 1u) << 16) | SpvOpGroupNonUniformBallot);
                vb_push_n(&b, &in[pos + 1], 2);             /* result_type, result_id */
                vb_push(&b, scope_id);                      /* ExecutionScope = Subgroup */
                vb_push_n(&b, &in[pos + 3], wc - 3);        /* <predicate> */
                break;
            }

            case SpvOpSubgroupReadInvocationKHR:
            {
                /* OpSubgroupReadInvocationKHR: <result_type> <result_id> <value> <index>
                 * OpGroupNonUniformShuffle:   <result_type> <result_id> <scope> <value> <index> */
                vb_push(&b, ((wc + 1u) << 16) | SpvOpGroupNonUniformShuffle);
                vb_push_n(&b, &in[pos + 1], 2);             /* result_type, result_id */
                vb_push(&b, scope_id);                      /* ExecutionScope = Subgroup */
                vb_push_n(&b, &in[pos + 3], wc - 3);        /* <value> <index> */
                break;
            }

            case SpvOpSubgroupFirstInvocationKHR:
            {
                /* OpSubgroupFirstInvocationKHR: <result_type> <result_id> <value>
                 * OpGroupNonUniformBroadcast: <result_type> <result_id> <scope> <value> <id=0> */
                vb_push(&b, ((wc + 2u) << 16) | SpvOpGroupNonUniformBroadcast);
                vb_push_n(&b, &in[pos + 1], 2);             /* result_type, result_id */
                vb_push(&b, scope_id);                      /* ExecutionScope = Subgroup */
                vb_push_n(&b, &in[pos + 3], wc - 3);        /* <value> */
                vb_push(&b, lane0_id);                      /* lane 0 = first invocation */
                break;
            }

            case SpvOpCapability:
                /* the rewritten opcodes are covered by the GroupNonUniform* capabilities, not
                 * by the KHR ones; keep the module self-consistent for validators */
                if (wc == 2 && in[pos + 1] == SpvCapabilitySubgroupVoteKHR)
                {
                    vb_push(&b, (2u << 16) | SpvOpCapability);
                    vb_push(&b, SpvCapabilityGroupNonUniformVote);
                }
                else if (wc == 2 && in[pos + 1] == SpvCapabilitySubgroupBallotKHR)
                {
                    vb_push(&b, (2u << 16) | SpvOpCapability);
                    vb_push(&b, SpvCapabilityGroupNonUniformBallot);
                    if (have_shuffle)
                    {
                        vb_push(&b, (2u << 16) | SpvOpCapability);
                        vb_push(&b, SpvCapabilityGroupNonUniformShuffle);
                    }
                }
                else
                {
                    vb_push_n(&b, &in[pos], wc);
                }
                break;

            default:
                vb_push_n(&b, &in[pos], wc);
                break;
        }

        if (b.oom)
            goto done;
        pos += wc;
    }

    /* the insertion point may be the very end of the module */
    if (insert_pos >= n)
    {
        if (new_uint_type)
        {
            vb_push(&b, (4u << 16) | SpvOpTypeInt);
            vb_push(&b, new_uint_type);
            vb_push(&b, 32);
            vb_push(&b, 0);
        }
        if (new_scope_const)
        {
            vb_push(&b, (4u << 16) | SpvOpConstant);
            vb_push(&b, uint_type_id ? uint_type_id : new_uint_type);
            vb_push(&b, new_scope_const);
            vb_push(&b, MGL_SCOPE_SUBGROUP);
        }
        if (new_lane0_const)
        {
            vb_push(&b, (4u << 16) | SpvOpConstant);
            vb_push(&b, uint_type_id ? uint_type_id : new_uint_type);
            vb_push(&b, new_lane0_const);
            vb_push(&b, 0);
        }
    }

    if (!b.oom && b.w)
    {
        result = b.w;
        *out_n = b.n;
        b.w = NULL;                     /* ownership transferred */
        if (getenv("MGL_VOTE_DUMP"))
        {
            char path[256];
            snprintf(path, sizeof path, "%s/vote_stage.spv", getenv("MGL_VOTE_DUMP"));
            FILE *f = fopen(path, "wb");
            if (f) { fwrite(result, 4, b.n, f); fclose(f); }
        }
    }

done:
    free(b.w);
    return result;
}
