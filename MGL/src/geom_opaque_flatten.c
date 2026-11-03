/* geom_opaque_flatten.c — SPIR-V pass: arrays-of-arrays of opaque types -> flat array.
 *
 * SPIRV-Cross's MSL backend throws "Arrays of arrays of textures are not supported
 * in MSL" (spirv_msl.cpp image_type_glsl / sampler_type) for a GLSL array-of-arrays
 * of an opaque type (`uniform sampler2D u[2][3];`, ARB_arrays_of_arrays), which
 * fails the whole link (dEQP-GL45-ES31 program_interface_query
 * uniform.location.default_block.array.array.{sampler_2d,sampler_2d_array,
 * sampler_3d,isampler_2d,image_2d,iimage_2d_array,atomic_uint}).
 *
 * Metal CAN express the resource as ONE flat `array<texture2d<float>, 6>` spanning
 * 6 consecutive argument slots — exactly what MGL binds for an opaque uniform: the
 * GL unit of u[i][j] at base + i*3 + j is the flat element base + i*3 + j.  This
 * pass lowers ONLY the module handed to SPIRV-Cross: the declaration becomes the
 * flat array and every full-element access chain u[i0][i1].. is re-indexed with
 * the linearized  i0*D1*D2.. + i1*D2.. + ..  (constant indices fold to one
 * constant).  The GL view (Program::spirv[stage].ir, never run through this) keeps
 * the multi-dimensional declaration, so introspection still reports the per-row
 * names, sizes and locations GL prescribes.
 *
 * Returns a fresh module, or NULL when nothing needed flattening or an unsupported
 * use was met (the caller keeps the original and the old link error shows).
 *
 * Layout notes: the new OpTypeArray / OpTypePointer / OpConstant instructions are
 * inserted before the first global OpVariable (constants cannot live inside
 * function bodies); the OpIMul / OpIAdd linearization arithmetic is emitted in the
 * same basic block, immediately before the access chain it feeds.  The module
 * bound (header word 3) is updated to cover the new ids.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include "spirv.h"

typedef struct {
    uint32_t var, ptr, opaque;          /* variable id, its (old) pointer type, the element type */
    uint32_t dims[4];                   /* dimensions, OUTERMOST first */
    int      ndims;
    uint32_t total;                     /* product of all dimensions */
    uint32_t flat_t, flat_ptr;          /* new flat array type / new pointer type ids */
} FlOpaque;

typedef struct { uint32_t type; int64_t value; uint32_t id; } FlConst;   /* created constants */
typedef struct { uint32_t op; uint32_t w[4]; } FlPre;                    /* one pre-chain instruction (wc 5) */

typedef struct {
    const uint32_t *in;
    size_t n;
    struct { size_t off; uint32_t wc, op; } *insts;  /* per instruction: offset, word count, opcode */
    size_t ninst;
    uint32_t *def;                                   /* id -> defining instruction index (0xffffffff: none) */
    uint32_t *deft;                                  /* id -> its result type id */
    uint32_t bound;
    uint32_t next_id;
    FlConst consts[256];
    int nconsts;
} FlCtx;

/* value of an existing 32-bit integer OpConstant id, or -1 */
static int64_t fl_int_const(const FlCtx *c, uint32_t id)
{
    if (id >= c->bound || c->def[id] >= (uint32_t)c->ninst) return -1;
    size_t k = c->def[id];
    if (c->insts[k].op != SpvOpConstant || c->insts[k].wc < 4) return -1;
    uint32_t t = c->in[c->insts[k].off + 1];
    if (t >= c->bound || c->def[t] >= (uint32_t)c->ninst) return -1;
    size_t tk = c->def[t];
    if (c->insts[tk].op != SpvOpTypeInt || c->in[c->insts[tk].off + 2] != 32) return -1;
    return (int64_t)(int32_t)c->in[c->insts[k].off + 3];
}

/* some 32-bit integer OpTypeInt id of the module (glslang always emits one; the
 * original array lengths are constants of it) */
static uint32_t fl_int_type(const FlCtx *c)
{
    for (size_t k = 0; k < c->ninst; k++)
        if (c->insts[k].op == SpvOpTypeInt && c->in[c->insts[k].off + 2] == 32)
            return c->in[c->insts[k].off + 1];
    return 0;
}

/* an existing constant of (type, value), else 0 */
static uint32_t fl_find_const(const FlCtx *c, uint32_t type, int64_t value)
{
    for (size_t k = 0; k < c->ninst; k++)
    {
        if (c->insts[k].op != SpvOpConstant || c->insts[k].wc < 4) continue;
        if (c->in[c->insts[k].off + 1] == type &&
            (int32_t)c->in[c->insts[k].off + 3] == (int32_t)value)
            return c->in[c->insts[k].off + 2];
    }
    return 0;
}

/* the id of a constant (type, value): existing module constant, one created by an
 * earlier call, or a fresh one (appended to the output's constant section) */
static uint32_t fl_newconst(FlCtx *c, uint32_t type, int64_t value)
{
    if (!type) return 0;
    uint32_t found = fl_find_const(c, type, value);
    if (found) return found;
    for (int k = 0; k < c->nconsts; k++)
        if (c->consts[k].type == type && c->consts[k].value == value)
            return c->consts[k].id;
    if (c->nconsts >= (int)(sizeof c->consts / sizeof c->consts[0])) return 0;
    c->consts[c->nconsts].type = type;
    c->consts[c->nconsts].value = value;
    c->consts[c->nconsts].id = c->next_id++;
    return c->consts[c->nconsts++].id;
}

uint32_t *mgl_spirv_flatten_opaque_arrays(const uint32_t *in, size_t n, size_t *out_n)
{
    FlCtx c;
    uint32_t *out = NULL;
    FlOpaque cand[16];
    int ncand = 0;
    struct { size_t inst; int ci; FlPre pre[8]; int npre; uint32_t lin; } plan[512];
    int nplan = 0;

    *out_n = 0;
    if (n < 20 || in[0] != 0x07230203) return NULL;
    if (getenv("MGL_FLATTEN_DEBUG"))
        fprintf(stderr, "MGL flatten: called n=%zu bound=%u\n", n, in[3]);

    memset(&c, 0, sizeof c);
    c.in = in;
    c.n = n;
    c.bound = in[3];
    c.next_id = c.bound;
    c.insts = (void *)malloc(n * sizeof *c.insts);
    c.def = (uint32_t *)malloc(c.bound * sizeof *c.def);
    c.deft = (uint32_t *)calloc(c.bound, sizeof *c.deft);
    if (!c.insts || !c.def || !c.deft) goto bail;
    for (uint32_t k = 0; k < c.bound; k++) c.def[k] = 0xffffffffu;

    for (size_t o = 5; o < n; )
    {
        uint32_t wc = in[o] >> 16, op = in[o] & 0xffff;
        if (wc == 0 || o + wc > n) break;
        if (c.ninst >= n) break;
        c.insts[c.ninst].off = o; c.insts[c.ninst].wc = wc; c.insts[c.ninst].op = op;
        switch (op)
        {
            /* instructions with no result id: nothing to record */
            case SpvOpNop: case SpvOpUndef: case SpvOpSource: case SpvOpSourceContinued:
            case SpvOpSourceExtension: case SpvOpExtension: case SpvOpExtInstImport:
            case SpvOpMemoryModel: case SpvOpEntryPoint: case SpvOpExecutionMode:
            case SpvOpExecutionModeId: case SpvOpCapability: case SpvOpName:
            case SpvOpMemberName: case SpvOpLine: case SpvOpNoLine: case SpvOpDecorate:
            case SpvOpMemberDecorate: case SpvOpGroupDecorate: case SpvOpGroupMemberDecorate:
            case SpvOpDecorationGroup: case SpvOpStore: case SpvOpImageWrite:
            case SpvOpCopyMemory: case SpvOpCopyMemorySized: case SpvOpFunctionEnd:
            case SpvOpLabel: case SpvOpBranch: case SpvOpBranchConditional: case SpvOpKill:
            case SpvOpReturn: case SpvOpReturnValue: case SpvOpUnreachable:
            case SpvOpMemoryBarrier: case SpvOpControlBarrier: case SpvOpMemoryNamedBarrier:
            case SpvOpLoopMerge: case SpvOpSelectionMerge: case SpvOpTerminateInvocation:
            case SpvOpDemoteToHelperInvocation: case SpvOpTerminateRayKHR:
            case SpvOpIgnoreIntersectionKHR:
                break;
            default:
                if (op >= 19 && op <= 33)
                    c.def[in[o + 1]] = (uint32_t)c.ninst;          /* type declaration */
                else if (wc >= 3)
                {
                    c.def[in[o + 2]] = (uint32_t)c.ninst;
                    c.deft[in[o + 2]] = in[o + 1];
                }
                break;
        }
        c.ninst++;
        o += wc;
    }

    /* candidates: OpVariable UniformConstant of >= 2 array levels over image/sampler */
    for (size_t k = 0; k < c.ninst; k++)
    {
        if (c.insts[k].op != SpvOpVariable || c.insts[k].wc < 4) continue;
        if (in[c.insts[k].off + 3] != SpvStorageClassUniformConstant) continue;
        if (getenv("MGL_FLATTEN_DEBUG"))
            fprintf(stderr, "MGL flatten: UniformConstant var %u ptr=%u\n",
                    in[c.insts[k].off + 2], in[c.insts[k].off + 1]);
        uint32_t ptr_t = in[c.insts[k].off + 1];
        if (ptr_t >= c.bound || c.def[ptr_t] >= (uint32_t)c.ninst ||
            c.insts[c.def[ptr_t]].op != SpvOpTypePointer) continue;
        uint32_t t = in[c.insts[c.def[ptr_t]].off + 3];             /* pointee */
        if (getenv("MGL_FLATTEN_DEBUG"))
            fprintf(stderr, "MGL flatten:   pointee=%u\n", t);
        FlOpaque f;
        memset(&f, 0, sizeof f);
        f.var = in[c.insts[k].off + 2];
        f.ptr = ptr_t;
        while (f.ndims < 4)
        {
            if (t >= c.bound || c.def[t] >= (uint32_t)c.ninst) { t = 0; break; }
            if (c.insts[c.def[t]].op != SpvOpTypeArray) break;
            int64_t len = fl_int_const(&c, in[c.insts[c.def[t]].off + 3]);
            if (len <= 0) { t = 0; break; }                         /* spec-constant length */
            f.dims[f.ndims++] = (uint32_t)len;
            t = in[c.insts[c.def[t]].off + 2];                      /* element type */
        }
        if (getenv("MGL_FLATTEN_DEBUG"))
            fprintf(stderr, "MGL flatten:   walked ndims=%d t=%u\n", f.ndims, t);
        if (!t || f.ndims < 2) continue;
        if (t >= c.bound || c.def[t] >= (uint32_t)c.ninst) continue;
        {
            uint32_t etop = c.insts[c.def[t]].op;
            if (etop != SpvOpTypeImage && etop != SpvOpTypeSampler &&
                etop != SpvOpTypeSampledImage) continue;
        }
        f.opaque = t;
        f.total = 1;
        for (int q = 0; q < f.ndims; q++) f.total *= f.dims[q];
        if (f.total > 1024) continue;                               /* sanity: past any limit */
        cand[ncand++] = f;
    }
    if (!ncand) goto bail;

    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL flatten: %d opaque array-of-arrays candidate(s)\n", ncand);

    /* usage check: the variable may only be touched by name/decoration/entry-point
     * lists, its declaration, and FULL-element access chains — anything else
     * (partial chains, whole-array loads) is not linearizable here */
    for (size_t k = 0; k < c.ninst; k++)
    {
        uint32_t op = c.insts[k].op, wc = c.insts[k].wc;
        const uint32_t *w = in + c.insts[k].off;
        int hits = 0;
        for (uint32_t q = 1; q < wc; q++)
            for (int z = 0; z < ncand; z++)
                if (w[q] == cand[z].var) { hits++; break; }
        if (!hits) continue;
        bool ok = false;
        if (op == SpvOpName || op == SpvOpDecorate || op == SpvOpEntryPoint || op == SpvOpVariable)
            ok = true;
        else if (op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain)
        {
            int z = -1;
            for (int q = 0; q < ncand; q++) if (w[3] == cand[q].var) { z = q; break; }
            ok = z >= 0 && (uint32_t)(wc - 4) == (uint32_t)cand[z].ndims;
        }
        if (!ok) goto bail;
    }

    for (int z = 0; z < ncand; z++)
    {
        cand[z].flat_t = c.next_id++;
        cand[z].flat_ptr = c.next_id++;
    }

    /* plan every chain rewrite (all new ids are assigned here) */
    for (size_t k = 0; k < c.ninst && nplan < 512; k++)
    {
        uint32_t op = c.insts[k].op, wc = c.insts[k].wc;
        const uint32_t *w = in + c.insts[k].off;
        if (op != SpvOpAccessChain && op != SpvOpInBoundsAccessChain) continue;
        int ci = -1;
        for (int z = 0; z < ncand; z++) if (w[3] == cand[z].var) { ci = z; break; }
        if (ci < 0 || (uint32_t)(wc - 4) != (uint32_t)cand[ci].ndims) continue;

        uint32_t idx[4], itype[4];
        int64_t cval[4];
        bool allc = true;
        for (int q = 0; q < cand[ci].ndims; q++)
        {
            idx[q] = w[4 + q];
            itype[q] = (idx[q] < c.bound) ? c.deft[idx[q]] : 0;
            int64_t v = fl_int_const(&c, idx[q]);
            if (v >= 0) cval[q] = v;
            else { cval[q] = -1; allc = false; }
        }
        /* every NON-constant index shares one 32-bit integer type */
        uint32_t at = 0;
        for (int q = 0; q < cand[ci].ndims; q++)
        {
            if (cval[q] >= 0) continue;
            uint32_t t = itype[q];
            if (!t || t >= c.bound || c.def[t] >= (uint32_t)c.ninst ||
                c.insts[c.def[t]].op != SpvOpTypeInt || in[c.insts[c.def[t]].off + 2] != 32)
                goto bail;
            if (!at) at = t;
            else if (at != t) goto bail;
        }

        plan[nplan].inst = k;
        plan[nplan].ci = ci;
        plan[nplan].npre = 0;

        if (allc)
        {
            int64_t lin = 0;
            for (int q = 0; q < cand[ci].ndims; q++)
            {
                uint32_t stride = 1;
                for (int d = q + 1; d < cand[ci].ndims; d++) stride *= cand[ci].dims[d];
                lin += cval[q] * stride;
            }
            plan[nplan].lin = fl_newconst(&c, itype[0], lin);
            if (!plan[nplan].lin) goto bail;
        }
        else
        {
            uint32_t acc = 0;
            int nterms = 0;
            for (int q = 0; q < cand[ci].ndims; q++)
            {
                uint32_t stride = 1;
                for (int d = q + 1; d < cand[ci].ndims; d++) stride *= cand[ci].dims[d];
                uint32_t term;
                if (cval[q] >= 0)
                {
                    int64_t v = cval[q] * stride;
                    term = 0;
                    if (v != 0)
                    {
                        term = fl_newconst(&c, at, v);
                        if (!term) goto bail;
                    }
                }
                else if (stride == 1)
                {
                    term = idx[q];
                }
                else
                {
                    uint32_t sc = fl_newconst(&c, at, stride);
                    uint32_t m = c.next_id++;
                    if (!sc || plan[nplan].npre >= 8) goto bail;
                    plan[nplan].pre[plan[nplan].npre].op = SpvOpIMul;
                    plan[nplan].pre[plan[nplan].npre].w[0] = at;
                    plan[nplan].pre[plan[nplan].npre].w[1] = m;
                    plan[nplan].pre[plan[nplan].npre].w[2] = idx[q];
                    plan[nplan].pre[plan[nplan].npre].w[3] = sc;
                    plan[nplan].npre++;
                    term = m;
                }
                if (term)
                {
                    if (!nterms) acc = term;
                    else
                    {
                        uint32_t s = c.next_id++;
                        if (plan[nplan].npre >= 8) goto bail;
                        plan[nplan].pre[plan[nplan].npre].op = SpvOpIAdd;
                        plan[nplan].pre[plan[nplan].npre].w[0] = at;
                        plan[nplan].pre[plan[nplan].npre].w[1] = s;
                        plan[nplan].pre[plan[nplan].npre].w[2] = acc;
                        plan[nplan].pre[plan[nplan].npre].w[3] = term;
                        plan[nplan].npre++;
                        acc = s;
                    }
                    nterms++;
                }
            }
            plan[nplan].lin = acc;
        }
        nplan++;
    }
    if (!nplan) goto bail;

    /* emit */
    out = (uint32_t *)malloc((n + 64 + (size_t)c.nconsts * 4 + (size_t)ncand * 16 + (size_t)nplan * 40) * sizeof *out);
    if (!out) goto bail;
    {
        size_t o = 0;
        uint32_t ity = 0;
        for (int q = 0; q < 5; q++)
            out[o++] = in[q];            /* bound patched below: ids keep being made during the walk */

        bool types_done = false;
        for (size_t k = 0; k < c.ninst; k++)
        {
            uint32_t op = c.insts[k].op, wc = c.insts[k].wc;
            const uint32_t *w = in + c.insts[k].off;

            /* the new types + constants go right before the first function: after
             * every original type/constant (glslang interleaves global variables
             * between type groups, so "first variable" is too early) and still in
             * the module's declaration section (a constant cannot sit in a body) */
            if (!types_done && op == SpvOpFunction)
            {
                /* a 32-bit integer type for the flat-array length constant: the
                 * type of any existing 32-bit int constant (glslang always has one) */
                if (!ity) ity = fl_int_type(&c);
                for (int z = 0; z < ncand; z++)
                {
                    /* the length constant first: SPIRV-Cross's parser marks an
                     * array-length id used while parsing OpTypeArray, so the
                     * OpConstant must already exist by then */
                    uint32_t total_const = fl_newconst(&c, ity, cand[z].total);
                    if (!total_const) { free(out); out = NULL; goto bail; }
                    (void)total_const;
                }
                for (int q = 0; q < c.nconsts; q++)
                {
                    out[o++] = (4u << 16) | SpvOpConstant;
                    out[o++] = c.consts[q].type;
                    out[o++] = c.consts[q].id;
                    out[o++] = (uint32_t)(int32_t)c.consts[q].value;
                }
                for (int z = 0; z < ncand; z++)
                {
                    out[o++] = (4u << 16) | SpvOpTypeArray;
                    out[o++] = cand[z].flat_t;
                    out[o++] = cand[z].opaque;
                    out[o++] = fl_newconst(&c, ity, cand[z].total);
                }
                for (int z = 0; z < ncand; z++)
                {
                    out[o++] = (4u << 16) | SpvOpTypePointer;
                    out[o++] = cand[z].flat_ptr;
                    out[o++] = SpvStorageClassUniformConstant;
                    out[o++] = cand[z].flat_t;
                }
                types_done = true;
            }

            /* the rewritten chains */
            int pl = -1;
            for (int q = 0; q < nplan; q++)
                if (plan[q].inst == k) { pl = q; break; }
            if (pl >= 0)
            {
                for (int q = 0; q < plan[pl].npre; q++)
                {
                    out[o++] = (5u << 16) | plan[pl].pre[q].op;
                    for (uint32_t z = 0; z < 4; z++) out[o++] = plan[pl].pre[q].w[z];
                }
                out[o++] = (5u << 16) | op;          /* result type, result id, base, ONE index */
                out[o++] = w[1];
                out[o++] = w[2];
                out[o++] = w[3];
                out[o++] = plan[pl].lin;
                continue;
            }

            /* the variable's result type becomes the flat pointer */
            if (op == SpvOpVariable)
            {
                int z = -1;
                for (int q = 0; q < ncand; q++)
                    if (w[2] == cand[q].var) { z = q; break; }
                if (z >= 0)
                {
                    out[o++] = w[0];
                    out[o++] = cand[z].flat_ptr;
                    for (uint32_t q = 2; q < wc; q++) out[o++] = w[q];
                    continue;
                }
            }

            for (uint32_t q = 0; q < wc; q++)
                out[o++] = w[q];
        }

        out[3] = c.next_id;            /* module bound covers every id created above */
        *out_n = o;
    }

    free(c.insts); free(c.def); free(c.deft);
    return out;
bail:
    free(c.insts); free(c.def); free(c.deft);
    free(out);
    return NULL;
}

/* stage-neutral, miss-safe entry used by program.c compile_stage and
 * introspect_spirv.c msl_accepts for EVERY stage: any stage's MSL view cannot
 * carry arrays-of-arrays of an opaque type.  On a miss the pass has already
 * cleared *out_n — restore it, or the caller frees a module it thinks is empty
 * (the r97e regression), and the pass is NOT chained into the shared
 * mgl_spirv_prepare_for_msl (tess domain): a miss there clobbered the caller's
 * word count. */
uint32_t *mgl_spirv_flatten_opaque_aoa(const uint32_t *in, size_t n, size_t *out_n)
{
    uint32_t *out = mgl_spirv_flatten_opaque_arrays(in, n, out_n);
    if (!out)
        *out_n = n;
    return out;
}
