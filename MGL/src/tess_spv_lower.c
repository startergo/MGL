/* ------------------------------------------------------------------ SPIR-V lowering
 * SPIRV-Cross refuses dynamic indices that walk INTO an interface block's
 * array/matrix members for tessellation-evaluation inputs ("Trying to
 * dynamically index into an array interface variable in tessellation") —
 * Metal can only index staged attributes with constants.  MGL's TES reads
 * control points from a BUFFER after its canonicalization, so the module just
 * has to COMPILE: lower every such access chain by loading the whole member
 * array with constant indices into a function-local variable and indexing the
 * copy (local arrays are dynamically indexable in MSL).
 *
 *      inVariables[0].value[i]
 * becomes
 *      float tmp[N] = inVariables[0].value;    // constant-indexed copy
 *      ... gl_in.cp[0].value[i] -> tmp[i]      // same result id, new root
 *
 * The transform is id-local and straight-line: no control flow is touched.
 */

#include <stdint.h>
#include <stdlib.h>
#include "mgl_ext_tess.h"
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#define SPV_OP_TYPE_VECTOR      23
#define SPV_OP_TYPE_MATRIX      24
#define SPV_OP_TYPE_ARRAY       28
#define SPV_OP_TYPE_STRUCT      30
#define SPV_OP_TYPE_POINTER     32
#define SPV_OP_FUNCTION         54
#define SPV_OP_CONSTANT         43
#define SPV_OP_PHI              245
#define SPV_OP_COMPOSITE_EXTRACT 81
#define SPV_OP_VARIABLE         59
#define SPV_OP_LOAD             61
#define SPV_OP_STORE            62
#define SPV_OP_ACCESSCHAIN      65
#define SPV_OP_INBOUNDSAC       70
#define SPV_OP_LABEL            248
#define SPV_SC_INPUT            1
#define SPV_SC_FUNCTION         7

struct SpvLowerCtx
{
    const uint32_t *w;
    size_t n;
    uint32_t *off, *op, *wc, *def;  /* per instruction: offset, opcode, word count, result id */
    uint32_t ninst;
};

/* opcodes with no result id (result-id extraction must skip them) */
static bool spv_lower_no_result(uint32_t op)
{
    switch (op)
    {
        case 0:   /* OpNop */
        case 2:   /* OpSourceContinued */
        case 3:   /* OpSource */
        case 4:   /* OpSourceExtension */
        case 5:   /* OpName */
        case 6:   /* OpMemberName */
        case 8:   /* OpLine */
        case 10:  /* OpExtension */
        case 14:  /* OpMemoryModel */
        case 15:  /* OpEntryPoint */
        case 16:  /* OpExecutionMode */
        case 17:  /* OpCapability */
        case 56:  /* OpFunctionEnd */
        case 62:  /* OpStore */
        case 63:  /* OpCopyMemory */
        case 64:  /* OpCopyMemorySized */
        case 71:  /* OpDecorate */
        case 72:  /* OpMemberDecorate */
        case 74:  /* OpGroupDecorate */
        case 75:  /* OpGroupMemberDecorate */
        case 224: /* OpMemoryBarrier */
        case 225: /* OpControlBarrier */
        case 246: /* OpLoopMerge */
        case 247: /* OpSelectionMerge */
        case 249: /* OpBranch */
        case 250: /* OpBranchConditional */
        case 251: /* OpSwitch */
        case 252: /* OpKill */
        case 253: /* OpReturn */
        case 254: /* OpReturnValue */
        case 255: /* OpUnreachable */
            return true;
        default:
            return false;
    }
}

static bool spv_lower_setup(struct SpvLowerCtx *c, const uint32_t *words, size_t n)
{
    size_t i = 5;
    uint32_t cnt = 0;

    memset(c, 0, sizeof *c);
    if (!words || n < 5)
        return false;
    while (i < n)
    {
        uint32_t w0 = words[i] >> 16;
        if (w0 == 0 || i + w0 > n) return false;
        cnt++;
        i += w0;
    }
    c->off = (uint32_t *)malloc((cnt + 1) * sizeof(uint32_t));
    c->op = (uint32_t *)malloc((cnt + 1) * sizeof(uint32_t));
    c->wc = (uint32_t *)malloc((cnt + 1) * sizeof(uint32_t));
    c->def = (uint32_t *)calloc(cnt + 1, sizeof(uint32_t));
    if (!c->off || !c->op || !c->wc || !c->def)
    {
        free(c->off); free(c->op); free(c->wc); free(c->def);
        memset(c, 0, sizeof *c);
        return false;
    }
    i = 5;
    while (i < n)
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        c->off[c->ninst] = (uint32_t)i;
        c->op[c->ninst] = op;
        c->wc[c->ninst] = wc;
        /* type instructions (and OpString/OpExtInstImport) define their result
         * at +1; ordinary instructions at +2; the rest define nothing */
        if ((op >= 19 && op <= 33) || op == 7 || op == 11)
            c->def[c->ninst] = words[i + 1];
        else if (wc >= 3 && !spv_lower_no_result(op))
            c->def[c->ninst] = words[i + 2];
        c->ninst++;
        i += wc;
    }
    c->w = words;
    c->n = n;
    return true;
}

static void spv_lower_teardown(struct SpvLowerCtx *c)
{
    free(c->off); free(c->op); free(c->wc); free(c->def);
    memset(c, 0, sizeof *c);
}

static int spv_lower_find(const struct SpvLowerCtx *c, uint32_t id)
{
    for (uint32_t k = 0; k < c->ninst; k++)
        if (c->def[k] == id)
            return (int)k;
    return -1;
}

static int64_t spv_lower_const(const struct SpvLowerCtx *c, uint32_t id)
{
    int k = spv_lower_find(c, id);
    if (k < 0 || c->op[k] != SPV_OP_CONSTANT || c->wc[k] < 4)
        return -1;
    return (int32_t)c->w[c->off[k] + 3];
}

enum { SPV_T_OTHER = 0, SPV_T_ARRAY, SPV_T_MATRIX, SPV_T_STRUCT, SPV_T_VECTOR };

static int spv_lower_type_kind(const struct SpvLowerCtx *c, uint32_t type)
{
    int k = spv_lower_find(c, type);
    if (k < 0) return SPV_T_OTHER;
    switch (c->op[k])
    {
        case SPV_OP_TYPE_ARRAY:  return SPV_T_ARRAY;
        case SPV_OP_TYPE_MATRIX: return SPV_T_MATRIX;
        case SPV_OP_TYPE_STRUCT: return SPV_T_STRUCT;
        case SPV_OP_TYPE_VECTOR: return SPV_T_VECTOR;
        default:                 return SPV_T_OTHER;
    }
}

static uint32_t spv_lower_elem_type(const struct SpvLowerCtx *c, uint32_t type, uint32_t member)
{
    int k = spv_lower_find(c, type);
    if (k < 0) return 0;
    uint32_t off = c->off[k];
    if (c->op[k] == SPV_OP_TYPE_STRUCT)
        return (member + 2 < c->wc[k]) ? c->w[off + 2 + member] : 0;
    if (c->op[k] == SPV_OP_TYPE_ARRAY || c->op[k] == SPV_OP_TYPE_MATRIX)
        return c->w[off + 2];      /* [hdr, result, element, length] */
    if (c->op[k] == SPV_OP_TYPE_VECTOR)
        return c->w[off + 2];      /* [hdr, result, component type, count] */
    return 0;
}

static uint32_t spv_lower_pointee(const struct SpvLowerCtx *c, uint32_t ptr_type)
{
    int k = spv_lower_find(c, ptr_type);
    if (k < 0 || c->op[k] != SPV_OP_TYPE_POINTER || c->wc[k] < 4)
        return 0;
    return c->w[c->off[k] + 3];    /* [hdr, result, storage, pointee] */
}

static uint32_t spv_lower_ptr_find(const struct SpvLowerCtx *c, uint32_t storage, uint32_t pointee)
{
    for (uint32_t k = 0; k < c->ninst; k++)
        if (c->op[k] == SPV_OP_TYPE_POINTER && c->wc[k] >= 4 &&
            c->w[c->off[k] + 2] == storage && c->w[c->off[k] + 3] == pointee)
            return c->def[k];
    return 0;
}

struct SpvLowerSite
{
    uint32_t inst;      /* instruction index of the access chain */
    uint32_t k;         /* word position (relative) of the first dynamic index */
    uint32_t arr_type;  /* composite the dynamic index walks into */
    uint32_t elem_type; /* pointee of the chain's result (element type) */
    uint32_t func;      /* enclosing OpFunction instruction index */
    uint32_t root;      /* split chains: the root Input VARIABLE id (0 = own base) */
    uint32_t prefix[8]; /* split chains: the inner chains' CONSTANT indices, in
                         * application order (walked before this chain's own) */
    uint32_t prefix_n;
    uint8_t  is_patch;  /* per-patch array site (the top-level array index of a
                         * Patch-decorated Input variable — no control point) */
    uint8_t  leaf_mode; /* struct/array element: copy leaf-by-leaf (a whole
                         * element load would abort SPIRV-Cross) */
    uint8_t  dmap;      /* 1-based index into the dissolve map: the chain is
                         * rooted at a dissolved block and its leaf copies must
                         * use the dissolved per-(element,field) variables, so
                         * the original block variable stays unreferenced */
    uint32_t leaf_off;  /* first path in the shared leaf table */
    uint32_t leaf_n;    /* number of leaf paths of ONE element */
};

/* OpDecorate <id> Patch (decoration 15 in this toolchain's spirv.h) */
#define SPV_DEC_PATCH 15  /* this toolchain's spirv.h: NoPerspective=13, Patch=15 */

/* a leaf path of a composite: constant indices from the composite to a
 * scalar/vector/matrix (anything loadable as one stage-in attribute member) */
struct SpvLowerLeaf
{
    uint32_t idx[6];    /* constant indices, outermost first */
    uint32_t depth;
    uint32_t type;      /* leaf type */
};

/* Does this access chain dynamically index an array/matrix MEMBER of an Input
 * block?  (The first index — the control-point array — may be dynamic.)
 * glslang SPLITS long chains: OpAccessChain(%mid, block, 0, member) followed by
 * OpAccessChain(%res, %mid, i) — the dynamic index then sits in a chain whose
 * base is another chain, not the variable.  Resolve through up to four
 * intermediate chains; their indices must be constant (the rewrite re-roots the
 * element copy at the variable and concatenates them). */
static bool spv_lower_var_is_patch(const struct SpvLowerCtx *c, uint32_t var);
static uint32_t spv_lower_composite_len(const struct SpvLowerCtx *c, uint32_t type);

static bool spv_lower_offends(struct SpvLowerCtx *c, uint32_t inst, struct SpvLowerSite *site)
{
    uint32_t off = c->off[inst], wc = c->wc[inst];
    uint32_t base = c->w[off + 3];
    int var = spv_lower_find(c, base);
    uint32_t cur, j;
    uint32_t prefix[8], prefix_n = 0;

    site->root = 0;
    site->prefix_n = 0;
    for (int depth = 0; var >= 0 && c->op[var] != SPV_OP_VARIABLE && depth < 4; depth++)
    {
        uint32_t voff, vwc, q, rev[8], rn = 0;

        if (c->op[var] != SPV_OP_ACCESSCHAIN && c->op[var] != SPV_OP_INBOUNDSAC)
            return false;                       /* a load or something else: give up */
        voff = c->off[var];
        vwc = c->wc[var];
        for (q = 4; q < vwc; q++)
        {
            if (rn >= 8 || prefix_n + rn >= 8)
                return false;
            if (spv_lower_const(c, c->w[voff + q]) < 0)
                return false;                   /* dynamic in the inner chain: unsupported */
            rev[rn++] = c->w[voff + q];
        }
        while (rn)
            prefix[prefix_n++] = rev[--rn];     /* un-reverse into application order */
        base = c->w[voff + 3];
        var = spv_lower_find(c, base);
    }
    if (var < 0 || c->op[var] != SPV_OP_VARIABLE)
        return false;
    if (c->wc[var] < 4 || c->w[c->off[var] + 3] != SPV_SC_INPUT)
        return false;
    /* walk from the BASE VARIABLE's pointer type (the chain's own result type
     * points at the final element) */
    cur = spv_lower_pointee(c, c->w[c->off[var] + 1]);
    if (!cur)
        return false;

    /* apply the inner chains' constant indices first */
    for (uint32_t q = 0; q < prefix_n && cur; q++)
    {
        int64_t lit = spv_lower_const(c, prefix[q]);
        int kind = spv_lower_type_kind(c, cur);

        if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
            cur = spv_lower_elem_type(c, cur, 0);
        else if (kind == SPV_T_STRUCT)
            cur = spv_lower_elem_type(c, cur, (uint32_t)lit);
        else
            return false;
    }
    if (!cur)
        return false;

    j = off + 4;                                    /* first index operand */
    site->is_patch = spv_lower_var_is_patch(c, c->def[var]);
    if (prefix_n == 0 && spv_lower_type_kind(c, cur) == SPV_T_ARRAY)
    {
        if (site->is_patch)
        {
            /* a per-PATCH array ('patch in Blk b[2]'): the top-level index
             * selects the patch block, NOT a control point — SPIRV-Cross
             * flattens it into attribute leaves, so a dynamic index here is
             * an ordinary lowerable site */
            if (spv_lower_const(c, c->w[off + 4]) < 0)
            {
                site->inst = inst;
                site->k = 4;
                site->arr_type = cur;
                site->elem_type = 0;            /* filled by the caller */
                site->root = 0;
                site->prefix_n = 0;
                return true;
            }
            cur = spv_lower_elem_type(c, cur, 0);
            j++;
        }
        else
        {
            /* the control-point array: its index may be dynamic (gl_in[i]) */
            cur = spv_lower_elem_type(c, cur, 0);
            j++;
        }
        if (!cur || j >= off + wc)
            return false;
    }

    while (j < off + wc)
    {
        uint32_t idx = c->w[j];
        int64_t lit = spv_lower_const(c, idx);
        int kind = spv_lower_type_kind(c, cur);

        if (kind == SPV_T_VECTOR)
        {
            /* a component index behind the dynamic one (m[i].x): walks to the
             * component type; a dynamic VECTOR index past the offending one
             * still lowers (the copy covers it) */
            cur = spv_lower_elem_type(c, cur, 0);
        }
        else if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
        {
            if (lit < 0)
            {
                site->inst = inst;
                site->k = (uint32_t)(j - off);
                site->arr_type = cur;
                site->elem_type = 0;                /* filled by the caller (walks the rest) */
                if (prefix_n)
                {
                    site->root = c->def[var];
                    memcpy(site->prefix, prefix, prefix_n * sizeof(uint32_t));
                    site->prefix_n = prefix_n;
                }
                return true;
            }
            cur = spv_lower_elem_type(c, cur, 0);
        }
        else if (kind == SPV_T_STRUCT)
        {
            if (lit < 0)
                return false;                       /* dynamic member index: not supported */
            cur = spv_lower_elem_type(c, cur, (uint32_t)lit);
        }
        else
            return false;
        if (!cur)
            return false;
        j++;
    }
    return false;
}

/* pointee of the chain's own result type (what the OpLoad loads) */
static uint32_t spv_lower_chain_elem_type(struct SpvLowerCtx *c, const struct SpvLowerSite *s)
{
    uint32_t off = c->off[s->inst], wc = c->wc[s->inst];
    uint32_t cur = s->arr_type;

    for (uint32_t j = s->k; j < wc; j++)
    {
        uint32_t idx = c->w[off + j];
        int64_t lit = spv_lower_const(c, idx);
        int kind = spv_lower_type_kind(c, cur);
        if (kind == SPV_T_VECTOR)
            cur = spv_lower_elem_type(c, cur, 0);
        else if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
        {
            if (lit < 0)
                cur = spv_lower_elem_type(c, cur, 0);
            else
                cur = spv_lower_elem_type(c, cur, 0);
        }
        else if (kind == SPV_T_STRUCT)
        {
            if (lit < 0) return 0;
            cur = spv_lower_elem_type(c, cur, (uint32_t)lit);
        }
        else
            return 0;
        if (!cur)
            return 0;
    }
    return cur;
}

struct SpvNewType { uint32_t id, storage, pointee; };

static uint32_t spv_lower_type_alloc(struct SpvNewType *types, uint32_t *n, uint32_t cap,
                                     uint32_t *next_id, uint32_t storage, uint32_t pointee)
{
    for (uint32_t q = 0; q < *n; q++)
        if (types[q].storage == storage && types[q].pointee == pointee)
            return types[q].id;
    if (*n >= cap || !pointee)
        return 0;
    types[*n].id = (*next_id)++;
    types[*n].storage = storage;
    types[*n].pointee = pointee;
    (*n)++;
    return types[*n - 1].id;
}

/* resolve a pointer type from the module or from the freshly allocated list */
static uint32_t spv_lower_ptr_of(const struct SpvLowerCtx *c,
                                 const struct SpvNewType *types, uint32_t n_types,
                                 uint32_t storage, uint32_t pointee)
{
    uint32_t t = spv_lower_ptr_find(c, storage, pointee);
    if (t) return t;
    for (uint32_t q = 0; q < n_types; q++)
        if (types[q].storage == storage && types[q].pointee == pointee)
            return types[q].id;
    return 0;
}

/* array/matrix length of a composite type (literal) */
/* OpDecorate <id> Patch: the variable carries per-patch data (no control
 * point dimension) — SPIRV-Cross flattens its array into attribute leaves */
/* SPIRV-Cross's is_patch_block rule: a Block-decorated struct whose EVERY
 * member carries the Patch member-decoration — glslang's encoding of a GL
 * per-patch interface block ('patch out Blk { ... } name', the Patch lands on
 * the block MEMBERS, not the variable).  Returns 1 when `type` is such a
 * block. */
static int spv_patch_block_type(const uint32_t *words, size_t n, uint32_t type)
{
    int have_block = 0, n_members = 0, all_patch = 1;

    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == 71 /*OpDecorate*/ && wc >= 3 && words[k + 1] == type && words[k + 2] == 2 /*Block*/)
            have_block = 1;
        if (op == 30 /*OpTypeStruct*/ && wc >= 2 && words[k + 1] == type)
            n_members = wc - 2;
        k += wc;
    }
    if (!have_block || n_members <= 0)
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL dissolve: patch_block_type(%u): have_block=%d n_members=%d\n",
                    type, have_block, n_members);
        return 0;
    }
    for (int m = 0; m < n_members && all_patch; m++)
    {
        int found = 0;

        for (size_t k = 5; k < n && !found; )
        {
            uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

            if (wc == 0 || k + wc > n) break;
            if (op == 72 /*OpMemberDecorate*/ && wc >= 4 && words[k + 1] == type &&
                words[k + 2] == (uint32_t)m && words[k + 3] == SPV_DEC_PATCH)
                found = 1;
            k += wc;
        }
        if (!found)
        {
            all_patch = 0;
            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL dissolve: patch_block_type(%u): member %d has no Patch\n", type, m);
        }
    }
    return all_patch;
}

static bool spv_lower_var_is_patch(const struct SpvLowerCtx *c, uint32_t var)
{
    for (uint32_t k = 0; k < c->ninst; k++)
        if (c->op[k] == 71 /*OpDecorate*/ && c->wc[k] >= 3 &&
            c->w[c->off[k] + 1] == var && c->w[c->off[k] + 2] == SPV_DEC_PATCH)
            return true;
    /* glslang's per-patch interface BLOCKS carry Patch on the block members
     * (spv_patch_block_type): those variables are patch variables too, or the
     * dynamic-IO lower would rewrite their chains as per-vertex ones */
    {
        uint32_t pointee = 0, elem = 0;

        for (uint32_t k = 0; k < c->ninst; k++)
            if (c->op[k] == 59 /*OpVariable*/ && c->wc[k] >= 4 && c->w[c->off[k] + 2] == var)
            {
                for (uint32_t q = 0; q < c->ninst; q++)
                    if (c->op[q] == 32 /*OpTypePointer*/ && c->wc[q] >= 4 &&
                        c->w[c->off[q] + 1] == c->w[c->off[k] + 1])
                    {
                        pointee = c->w[c->off[q] + 3];
                        break;
                    }
                break;
            }
        if (!pointee)
            return false;
        /* one array level: 'patch out Blk b[2]' */
        for (uint32_t q = 0; q < c->ninst; q++)
            if (c->op[q] == 28 /*OpTypeArray*/ && c->wc[q] >= 4 && c->w[c->off[q] + 1] == pointee)
            {
                elem = c->w[c->off[q] + 2];
                break;
            }
        return spv_patch_block_type(c->w, c->n, elem ? elem : pointee) != 0;
    }
}

/* collect the leaf paths of `type`; returns how many were written to `out`
 * (0 = too many / unresolvable — the caller declines the site) */
static uint32_t spv_lower_leaves(const struct SpvLowerCtx *c, uint32_t type,
                                 uint32_t *path, uint32_t depth,
                                 struct SpvLowerLeaf *out, uint32_t cap)
{
    int kind = spv_lower_type_kind(c, type);

    if (kind != SPV_T_STRUCT && kind != SPV_T_ARRAY)
    {
        if (!cap || depth > 6)
            return 0;
        out[0].depth = depth;
        out[0].type = type;
        for (uint32_t q = 0; q < depth; q++)
            out[0].idx[q] = path[q];
        return 1;
    }
    if (depth >= 6)
        return 0;
    {
        uint32_t n = 0;
        uint32_t cnt = (kind == SPV_T_ARRAY) ? spv_lower_composite_len(c, type) : 0;
        if (kind == SPV_T_STRUCT)
        {
            int k = spv_lower_find(c, type);
            if (k < 0)
                return 0;
            cnt = c->wc[k] - 2;         /* member operands */
        }
        if (cnt > 64)
            return 0;
        for (uint32_t i = 0; i < cnt; i++)
        {
            uint32_t sub, elem = spv_lower_elem_type(c, type, i);
            if (!elem)
                return 0;
            path[depth] = i;
            sub = spv_lower_leaves(c, elem, path, depth + 1, out + n, cap - n);
            if (!sub)
                return 0;
            n += sub;
        }
        return n;
    }
}

static uint32_t spv_lower_composite_len(const struct SpvLowerCtx *c, uint32_t type)
{
    int k = spv_lower_find(c, type);
    if (k < 0) return 0;
    if (c->op[k] == SPV_OP_TYPE_ARRAY)
    {
        int64_t v = spv_lower_const(c, c->w[c->off[k] + 3]);
        return v > 0 && v <= 256 ? (uint32_t)v : 0;
    }
    if (c->op[k] == SPV_OP_TYPE_MATRIX)
        return c->w[c->off[k] + 3];
    return 0;
}

struct SpvNewConst { uint32_t id, type, value; };

/* an OpConstant id (module or pre-declared) of `value` with `type`, or 0 */
static uint32_t spv_lower_const_id(const struct SpvLowerCtx *c, uint32_t type, uint32_t value,
                                   const struct SpvNewConst *new_consts, uint32_t n_new_consts)
{
    for (uint32_t q = 0; q < c->ninst; q++)
        if (c->op[q] == SPV_OP_CONSTANT && c->wc[q] >= 4 &&
            c->w[c->off[q] + 1] == type && (int32_t)c->w[c->off[q] + 3] == (int32_t)value)
            return c->def[q];
    for (uint32_t q = 0; q < n_new_consts; q++)
        if (new_consts[q].type == type && new_consts[q].value == value)
            return new_consts[q].id;
    return 0;
}

/* GUARD (process-abort protection): SPIRV-Cross's
 * emit_tessellation_access_chain asserts (index != uint32_t(-1)) when a
 * post-tessellation access chain descends into an ARRAY-typed FIELD of an
 * Input interface variable — arrays cannot be flattened into tessellation
 * stage-in attributes, so the member never receives an interface member index.
 * Chains this lowering rewrites are removed from the module, but chains it
 * LEAVES (a constant trailing index, or a site dropped during validation)
 * stay and abort the process inside SPIRV-Cross (observed: dEQP-GL45-ES31
 * tessellation.user_defined_io.per_vertex_block.vertex_io_array_size_implicit
 * .triangles).  Returning true here makes the caller decline the whole
 * lowering: MGL then compiles the ORIGINAL module, which SPIRV-Cross rejects
 * cleanly ("Trying to dynamically index into an array interface variable in
 * tessellation"), so the link fails normally instead of crashing the runner. */
static bool spv_lower_fatal_remaining(const struct SpvLowerCtx *c,
                                      const struct SpvLowerSite *sites, uint32_t n_sites)
{
    for (uint32_t k = 0; k < c->ninst; k++)
    {
        uint32_t idx[96], nidx = 0, segs[8][2], nsegs = 0;

        if (c->op[k] != SPV_OP_ACCESSCHAIN && c->op[k] != SPV_OP_INBOUNDSAC)
            continue;
        bool is_site = false;
        for (uint32_t s = 0; s < n_sites; s++)
            if (sites[s].inst != 0xffffffff && sites[s].inst == k) { is_site = true; break; }
        if (is_site)
            continue;                                   /* rewritten: not in the output module */

        /* splice the chain and any inner chains into one index list; segments
         * are collected outermost..innermost and applied innermost-first */
        int cur = (int)k;
        uint32_t root_var = 0;
        for (int depth = 0; depth < 8; depth++)
        {
            uint32_t off = c->off[cur], wc = c->wc[cur];
            if (wc < 5)
                break;                                 /* no index operands */
            segs[nsegs][0] = off + 4;
            segs[nsegs][1] = off + wc;
            if (nsegs < 7) nsegs++;
            uint32_t base = c->w[off + 3];
            int b = spv_lower_find(c, base);
            if (b < 0) { root_var = 0; break; }
            if (c->op[b] == SPV_OP_ACCESSCHAIN || c->op[b] == SPV_OP_INBOUNDSAC) { cur = b; continue; }
            if (c->op[b] == SPV_OP_VARIABLE) root_var = base;
            break;
        }
        if (!root_var)
            continue;
        int rv = spv_lower_find(c, root_var);
        if (rv < 0 || c->op[rv] != SPV_OP_VARIABLE)
            continue;
        if (c->w[c->off[rv] + 3] != SPV_SC_INPUT)  /* OpVariable: type, result, storage */
            continue;                                   /* only stage-in interfaces matter */
        for (int s = (int)nsegs - 1; s >= 0 && nidx < 96; s--)
            for (uint32_t j = segs[s][0]; j < segs[s][1] && nidx < 96; j++)
                idx[nidx++] = c->w[j];
        if (!nidx)
            continue;

        /* walk the type path: the variable's OWN top-level array dimension
         * (teBlock[i] / gl_in[i]) is indexable; any ARRAY step after a struct
         * member descent is fatal */
        uint32_t t = spv_lower_pointee(c, c->w[c->off[rv] + 1]);
        bool member_depth = false;
        bool patch_var = spv_lower_var_is_patch(c, root_var);
        for (uint32_t q = 0; q < nidx && t; q++)
        {
            int kind = spv_lower_type_kind(c, t);
            if (kind == SPV_T_STRUCT)
            {
                int64_t lit = spv_lower_const(c, idx[q]);
                if (lit < 0)
                    return true;                       /* dynamic member index: unsupported too */
                t = spv_lower_elem_type(c, t, (uint32_t)lit);
                member_depth = true;
            }
            else if (kind == SPV_T_ARRAY)
            {
                if (member_depth)
                {
                    /* into an array FIELD: a DYNAMIC index aborts SPIRV-Cross
                     * (the flattened attribute leaves cannot be selected at
                     * run time); a constant index resolves into one leaf and
                     * is fine */
                    if (spv_lower_const(c, idx[q]) < 0)
                        return true;
                    t = spv_lower_elem_type(c, t, 0);
                    continue;
                }
                if (patch_var && spv_lower_const(c, idx[q]) < 0)
                    return true;                       /* patch-array index: fatal unless a site */
                t = spv_lower_elem_type(c, t, 0);
            }
            else if (kind == SPV_T_MATRIX || kind == SPV_T_VECTOR)
                t = spv_lower_elem_type(c, t, 0);       /* matrix members are legal in stage_in */
            else
                break;
        }
    }
    return false;
}

#define SPV_LOWER_MAX_LEAVES 2048
/* helper: push an OpName into a word buffer */
static void namebuf_words(char *buf, size_t *n, uint32_t id, const char *name);
static int64_t spv_dis_const(const uint32_t *words, size_t n, uint32_t id);
static uint32_t spv_dis_leaf_count(const uint32_t *words, size_t n, uint32_t type);
static void namebuf_words(char *buf, size_t *n, uint32_t id, const char *name)
{
    size_t need = strlen(name);
    size_t w4 = need / 4 + 1;      /* always at least one NUL byte (SPIR-V §2.2.1) */
    uint32_t *b = (uint32_t *)buf;

    b[(*n)++] = (uint32_t)(((2 + w4) << 16) | 5);
    b[(*n)++] = id;
    memset(b + *n, 0, w4 * 4);
    memcpy(b + *n, name, need);
    (*n) += w4;
}

/* ---- entry-copy emitter for the TES per-element form ----
 * The entry block copies each dissolved patch variable into the function-
 * local block array.  SPIRV-Cross's emit_tessellation_io_load can only
 * flatten LEAF loads (scalars/vectors) from a chain — loading a struct or
 * matrix needs stage_in_ptr_var_id, which is 0 when the TES has no per-vertex
 * inputs ("Cannot resolve expression type"), and whole-ARRAY member stores
 * lower to an MSL copy template that cannot match spvUnsafeArray against the
 * local's plain C array member.  So the copy walks the member's type tree and
 * emits one AccessChain/Load/Store per SCALAR-or-VECTOR leaf. */
struct TessCopyCtx
{
    const uint32_t *words;
    size_t n;
    uint32_t *ins_types;              /* new pointer types + int constants */
    size_t *n_types, cap_types;
    uint32_t *ins_entry;              /* the copy instructions */
    size_t *n_entry, cap_entry;
    uint32_t *next_id;
    uint32_t local_var, src_var;      /* dst base (Private) / src (Input) */
    uint32_t dst_sc;                  /* storage class of the dst chain's pointer type */
    uint32_t dst_idx[10], src_idx[10];
    int dst_n, src_n;
    uint32_t cst_id[24];              /* small int constants found in the module */
    uint32_t cst_val[24];
    int n_cst;
    int n_leaf;
};

static uint32_t tess_copy_cst(const struct TessCopyCtx *c, uint32_t v)
{
    for (int t = 0; t < c->n_cst; t++)
        if (c->cst_val[t] == v) return c->cst_id[t];
    return 0;
}

static uint32_t spv_lower_ptr_type(const uint32_t *words, size_t n, uint32_t *ins_types,
                                   size_t *n_types, size_t cap_types, uint32_t *next_id,
                                   uint32_t sc, uint32_t pointee);

/* returns 0 = done, -1 = decline (shape too deep / missing constant) */
static int tess_copy_walk(struct TessCopyCtx *c, uint32_t type, int depth)
{
    const uint32_t *words = c->words;
    size_t n = c->n;

    if (depth > 5 || c->dst_n >= 10 || c->src_n >= 10 || c->n_leaf > 512)
        return -1;
    for (size_t q = 5; q < n; )
    {
        uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

        if (wq == 0 || q + wq > n) break;
        if (oq == 28 /*OpTypeArray*/ && wq >= 4 && words[q + 1] == type)
        {
            uint32_t elem = words[q + 2], len_id = words[q + 3];
            int64_t len = spv_dis_const(words, n, len_id);
            uint32_t idx;

            if (len < 0 || len > 16)
                return -1;
            for (int64_t i = 0; i < len; i++)
            {
                idx = tess_copy_cst(c, (uint32_t)i);
                if (!idx) return -1;
                c->dst_idx[c->dst_n++] = idx;
                c->src_idx[c->src_n++] = idx;
                if (tess_copy_walk(c, elem, depth + 1)) return -1;
                c->dst_n--;
                c->src_n--;
            }
            return 0;
        }
        if (oq == 24 /*OpTypeMatrix*/ && wq >= 4 && words[q + 1] == type)
        {
            uint32_t col = words[q + 2], ncol = words[q + 3], idx;

            if (ncol < 2 || ncol > 4) return -1;
            for (uint32_t i = 0; i < ncol; i++)
            {
                idx = tess_copy_cst(c, i);
                if (!idx) return -1;
                c->dst_idx[c->dst_n++] = idx;
                c->src_idx[c->src_n++] = idx;
                if (tess_copy_walk(c, col, depth + 1)) return -1;
                c->dst_n--;
                c->src_n--;
            }
            return 0;
        }
        if (oq == 30 /*OpTypeStruct*/ && wq >= 2 && words[q + 1] == type)
        {
            uint32_t nmem = wq - 2, idx;

            if (nmem > 16) return -1;
            for (uint32_t j = 0; j < nmem; j++)
            {
                idx = tess_copy_cst(c, j);
                if (!idx) return -1;
                c->dst_idx[c->dst_n++] = idx;
                c->src_idx[c->src_n++] = idx;
                if (tess_copy_walk(c, words[q + 2 + j], depth + 1)) return -1;
                c->dst_n--;
                c->src_n--;
            }
            return 0;
        }
        q += wq;
    }
    /* leaf: scalar / vector — one chain pair + load + store.  The source
     * chain may be EMPTY (a plain scalar member reads the variable itself);
     * a zero-index OpAccessChain is invalid SPIR-V and aborts SPIRV-Cross's
     * active-builtin scan, so load the variable directly in that case. */
    {
        uint32_t ipf = spv_lower_ptr_type(words, n, c->ins_types, c->n_types,
                                          c->cap_types, c->next_id, c->dst_sc, type);
        uint32_t iaptr = (c->src_n > 0) ?
                         spv_lower_ptr_type(words, n, c->ins_types, c->n_types,
                                            c->cap_types, c->next_id, 1, type) : 0;
        uint32_t need = (uint32_t)(4 + c->dst_n) +
                        (c->src_n > 0 ? (uint32_t)(4 + c->src_n) : 0) + 4 + 3;
        uint32_t ac, acv, ld;

        if (!ipf) return -1;
        if (*c->n_entry + need > c->cap_entry) return -1;
        ac = (*c->next_id)++;
        ld = (*c->next_id)++;
        c->ins_entry[(*c->n_entry)++] = (uint32_t)(((4 + c->dst_n) << 16) | 65);
        c->ins_entry[(*c->n_entry)++] = ipf;
        c->ins_entry[(*c->n_entry)++] = ac;
        c->ins_entry[(*c->n_entry)++] = c->local_var;
        for (int k = 0; k < c->dst_n; k++) c->ins_entry[(*c->n_entry)++] = c->dst_idx[k];
        if (c->src_n > 0)
        {
            acv = (*c->next_id)++;
            c->ins_entry[(*c->n_entry)++] = (uint32_t)(((4 + c->src_n) << 16) | 65);
            c->ins_entry[(*c->n_entry)++] = iaptr;
            c->ins_entry[(*c->n_entry)++] = acv;
            c->ins_entry[(*c->n_entry)++] = c->src_var;
            for (int k = 0; k < c->src_n; k++) c->ins_entry[(*c->n_entry)++] = c->src_idx[k];
        }
        else
            acv = c->src_var;             /* load the variable directly */
        c->ins_entry[(*c->n_entry)++] = (4u << 16) | 61;      /* OpLoad */
        c->ins_entry[(*c->n_entry)++] = type;
        c->ins_entry[(*c->n_entry)++] = ld;
        c->ins_entry[(*c->n_entry)++] = acv;
        c->ins_entry[(*c->n_entry)++] = (3u << 16) | 62;      /* OpStore */
        c->ins_entry[(*c->n_entry)++] = ac;
        c->ins_entry[(*c->n_entry)++] = ld;
        c->n_leaf++;
    }
    return 0;
}

/* find-or-create a pointer type: search the module first, then the pending
 * insertion set, and only then mint a new id (0 = out of insertion room) */
static uint32_t spv_lower_ptr_type(const uint32_t *words, size_t n, uint32_t *ins_types,
                                   size_t *n_types, size_t cap_types, uint32_t *next_id,
                                   uint32_t sc, uint32_t pointee)
{
    for (size_t q = 5; q < n; )
    {
        uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

        if (wq == 0 || q + wq > n) break;
        if (oq == 32 /*OpTypePointer*/ && wq >= 4 && words[q + 2] == sc &&
            words[q + 3] == pointee)
            return words[q + 1];
        q += wq;
    }
    for (size_t q = 0; q + 4 <= *n_types; q += 4)
        if ((ins_types[q] & 0xffff) == 32 && ins_types[q + 2] == sc &&
            ins_types[q + 3] == pointee)
            return ins_types[q + 1];
    if (*n_types + 4 > cap_types)
        return 0;
    ins_types[(*n_types)++] = (4u << 16) | 32;
    ins_types[(*n_types)++] = *next_id;
    ins_types[(*n_types)++] = sc;
    ins_types[(*n_types)++] = pointee;
    return (*next_id)++;
}

/* ------------------------------------------------------------------ per-patch interface blocks
 * GL allows per-patch interface blocks, singly ('patch out Blk { ... } b;')
 * and as ARRAYS ('patch out Blk { ... } b[2];').  glslang encodes them as a
 * Block struct whose members carry the Patch decoration (spv_patch_block_type
 * — SPIRV-Cross's is_patch_block).  SPIRV-Cross's tessellation access chains
 * resolve such a variable's members through a member-index map that is only
 * correct for members that need no splitting, and the TCS compile of an
 * array-of-blocks dies with "Bad cast".  Rewrite: dissolve the block into one
 * plain PATCH variable per block member ('ppBlock_s' : S or array<S,2>,
 * 'ppBlock_fa' : float[3] or array<float[3],2>) — plain patch variables are
 * buffer-backed in the TCS kernel and attribute-flattened in the TES, and
 * SPIRV-Cross handles both correctly.
 *
 * Declines (NULL) on anything it cannot rewrite; the caller keeps the
 * original module, which fails the link cleanly. */
struct PArrJob
{
    uint32_t var, storage, block, len_id, base_loc;
    int array_form;                 /* 1 = 'Blk b[N]' (element index precedes the member), 0 = 'Blk b' */
    uint32_t n_members, member_type[32], member_slot[32], member_loc[32];
    char member_name[32][48];
    uint32_t new_var[32];
    /* TES-input form: one plain patch variable per (member, element) plus a
     * module-scope PRIVATE 'Blk[N]' copied at the entry block — the local
     * keeps dynamic element/member indices working (no variable selection
     * needed) and is visible from helper functions (SPIRV-Cross localizes
     * it into the entry point and passes it down to every callee) */
    int is_tes;
    uint32_t local_var, block_slots, n_elem, pointee;
    uint32_t new_var2[8][32];       /* [element][member] */
};

/* rewrite an OpEntryPoint's interface-id operand list into dst: the dissolved
 * block variables leave, their replacements join (SPIRV-Cross only recognizes
 * stage IO listed here).  Returns the words written to dst. */
static size_t spv_rewrite_entry_point(const uint32_t *words, size_t i,
                                      const struct PArrJob *jobs, int n_jobs,
                                      uint32_t *dst)
{
    uint32_t wc = words[i] >> 16;
    size_t p = 3, oh = 0, o = 0;

    while (p < wc)
    {
        uint32_t v = words[i + p];

        if (!(v & 0xff000000u) || !(v & 0xff0000u) || !(v & 0xff00u) || !(v & 0xffu))
        {
            p++;                     /* the word carrying the name's NUL */
            break;
        }
        p++;
    }
    memcpy(dst + o, words + i, p * 4);
    oh = o;
    o += p;
    for (uint32_t q = (uint32_t)p; q < wc; q++)
    {
        int drop = 0;

        for (int j = 0; j < n_jobs; j++)
            if (words[i + q] == jobs[j].var) drop = 1;
        if (!drop) dst[o++] = words[i + q];
    }
    for (int j = 0; j < n_jobs; j++)
    {
        if (jobs[j].is_tes)
        {
            for (uint32_t e = 0; e < jobs[j].n_elem; e++)
                for (uint32_t m2 = 0; m2 < jobs[j].n_members; m2++)
                    dst[o++] = jobs[j].new_var2[e][m2];
        }
        else
            for (uint32_t m2 = 0; m2 < jobs[j].n_members; m2++)
                dst[o++] = jobs[j].new_var[m2];
    }
    dst[oh] = (uint32_t)(((o - oh) << 16) | 15);
    return o;
}

/* execution model of the module's first OpEntryPoint (0 = none found) */
static uint32_t spv_entry_model(const uint32_t *words, size_t n)
{
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == 15 /*OpEntryPoint*/ && wc >= 4)
            return words[k + 1];
        if (op == 54 /*OpFunction*/)
            break;                  /* entry points precede the functions */
        k += wc;
    }
    return 0;
}

/* geom r105h: a LENGTH-1 array of a plain interface block as a TES OUTPUT
 * ('layout(xfb_stride=128) out Goku { ... } goku[1];' — enhanced_layouts
 * xfb_block_stride) trips SPIRV-Cross: it flattens the block into per-member stage
 * outputs but bridges them through the UNINDEXED instance ("out.goku_gohan =
 * goku.gohan;" while goku stays a spvUnsafeArray<Goku, 1>) and the MSL does not
 * compile.  Retype the variable to the plain block and splice the single constant
 * element index from its access chains — the form SPIRV-Cross emits correctly.
 * Returns a fresh module (caller frees) or NULL when nothing matched. */
static uint32_t *spv_strip_len1_out_blocks(const uint32_t *words, size_t n, size_t *out_n)
{
    /* var id -> the ptr-to-struct type to retarget to */
    uint32_t vars[8], newtys[8];
    uint32_t next_id = words[3];
    int n_vars = 0;
    uint32_t *buf = NULL;
    size_t o = 0;

    *out_n = 0;
    for (size_t i = 5; i < n && n_vars < 8; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

        if (wc == 0 || i + wc > n) break;
        if (op == 59 /*OpVariable*/ && wc >= 4 && words[i + 3] == 3 /*Output*/)
        {
            /* the pointer's pointee must be OpTypeArray of a Block struct with len 1 */
            uint32_t ptr = words[i + 1], arr = 0, elem = 0, len_cst = 0, len = 0, is_block = 0;

            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == 32 /*OpTypePointer*/ && wc2 >= 4 && words[k + 1] == ptr)
                    arr = words[k + 3];
                k += wc2;
            }
            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == 28 /*OpTypeArray*/ && wc2 >= 4 && words[k + 1] == arr)
                {
                    elem = words[k + 2];
                    len_cst = words[k + 3];
                }
                k += wc2;
            }
            if (elem)
                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 43 /*OpConstant*/ && wc2 >= 4 && words[k + 2] == len_cst)
                        len = words[k + 3];
                    if (op2 == 71 /*OpDecorate*/ && wc2 >= 3 && words[k + 1] == elem &&
                        words[k + 2] == 2 /*Block*/)
                        is_block = 1;
                    k += wc2;
                }
            if (is_block && len == 1)
            {
                vars[n_vars] = words[i + 2];
                newtys[n_vars] = 0;   /* filled while emitting (one new ptr type each) */
                n_vars++;
            }
        }
        i += wc;
    }
    if (!n_vars) return NULL;

    buf = malloc(n * 4 * 4 + 64);
    if (!buf) return NULL;
    for (uint32_t w = 0; w < 5; w++) buf[w] = words[w];   /* magic..schema */
    o = 5;
    for (size_t i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        int own = -1;

        if (wc == 0 || i + wc > n) { free(buf); return NULL; }

        /* retarget the variable's type to a fresh ptr(Block) — emitted right here */
        if (op == 59 /*OpVariable*/)
            for (int v = 0; v < n_vars; v++)
                if (vars[v] == words[i + 2] && !newtys[v])
                {
                    newtys[v] = next_id++;
                    own = v;
                    break;
                }

        if (own >= 0)
        {
            /* find the block struct id for this variable in two passes (the array
             * type usually PRECEDES its pointer in the module) */
            uint32_t arr2 = 0, elem2 = 0;

            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == 32 && wc2 >= 4 && words[k + 1] == words[i + 1]) arr2 = words[k + 3];
                k += wc2;
            }
            if (arr2)
                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 28 && wc2 >= 4 && words[k + 1] == arr2) elem2 = words[k + 2];
                    k += wc2;
                }
            buf[o++] = 4 << 16 | 32;            /* OpTypePointer */
            buf[o++] = newtys[own];
            buf[o++] = 3 /*Output*/;
            buf[o++] = elem2;
            /* the variable's result type becomes the fresh pointer */
            buf[o] = 4 << 16 | 59;               /* OpVariable */
            buf[o + 1] = newtys[own];
            buf[o + 2] = words[i + 2];
            buf[o + 3] = 3 /*Output*/;
            o += 4;
            i += wc;
            continue;
        }

        if ((op == 65 /*OpAccessChain*/ || op == 66 /*OpInBoundsAccessChain*/) && wc >= 4)
        {
            int v = -1;

            for (int q = 0; q < n_vars; q++)
                if (vars[q] == words[i + 3]) { v = q; break; }
            /* leading index must be the constant 0: its value id is the 4th word */
            if (v >= 0 && wc >= 5)
            {
                uint32_t zero_id = words[i + 4], is_zero = 0;

                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 43 /*OpConstant*/ && wc2 >= 4 && words[k + 2] == zero_id &&
                        words[k + 3] == 0)
                        is_zero = 1;
                    k += wc2;
                }
                if (is_zero)
                {
                    /* emit wc-1 words: header, result type, result id, base, then the
                     * indices AFTER the dropped constant 0 */
                    buf[o] = (wc - 1) << 16 | op;
                    buf[o + 1] = words[i + 1];
                    buf[o + 2] = words[i + 2];
                    buf[o + 3] = words[i + 3];
                    for (uint32_t w = 4; w + 1 < wc; w++)
                        buf[o + w] = words[i + w + 1];
                    o += wc - 1;
                    i += wc;
                    continue;
                }
            }
        }

        for (uint32_t w = 0; w < wc; w++) buf[o + w] = words[i + w];
        o += wc;
        i += wc;
    }

    /* the header's bound grows by the new ids */
    buf[3] = next_id;
    *out_n = o;
    if (getenv("MGL_TESS_DUMP_DISSOLVE"))
    {
        char path[512];
        static int seq = 0;

        snprintf(path, sizeof path, "%s/strip1_%d.spv", getenv("MGL_TESS_DUMP_DISSOLVE"), seq++);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(buf, 4, o, f); fclose(f); }
    }
    return buf;
}

uint32_t *mgl_tess_dissolve_patch_block_arrays(const uint32_t *words, size_t n, size_t *out_n)
{
    struct PArrJob jobs[8];
    int n_jobs = 0;
    uint32_t next_id = words[3];
    uint32_t *out = NULL;
    uint32_t model = spv_entry_model(words, n);

    *out_n = 0;
    if (!words || n < 8)
        return NULL;

    /* geom r105h: strip length-1 OUTPUT block arrays to the plain block first (see
     * spv_strip_len1_out_blocks) — only the evaluation stage declares those */
    if (model == 2 /*TessellationEvaluation*/)
    {
        size_t sn = 0;
        uint32_t *stripped = spv_strip_len1_out_blocks(words, n, &sn);

        if (stripped)
        {
            uint32_t *done = mgl_tess_dissolve_patch_block_arrays(stripped, sn, out_n);

            if (done) { free(stripped); return done; }
            /* no patch jobs: the stripped module is the result */
            if (getenv("MGL_TESS_DUMP_DISSOLVE"))
                fprintf(stderr, "MGL dissolve: len-1 output block stripped, no patch jobs\n");
            *out_n = sn;
            return stripped;
        }
    }

    if (getenv("MGL_TESS_DUMP_DISSOLVE"))
    {
        char path[512];
        static int seq = 0;

        snprintf(path, sizeof path, "%s/in_%d.spv", getenv("MGL_TESS_DUMP_DISSOLVE"), seq++);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(words, 4, n, f); fclose(f); }
    }

    /* 1) interface variables whose pointee is a patch block ('Blk b') or an
     *    array of one ('Blk b[N]') */
    for (size_t i = 5; i < n && n_jobs < 8; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

        if (wc == 0 || i + wc > n) break;
        if (op == 59 /*OpVariable*/ && wc >= 4 &&
            (words[i + 3] == 1 /*Input*/ || words[i + 3] == 3 /*Output*/))
        {
            uint32_t vid = words[i + 2];
            uint32_t ptr = words[i + 1];
            uint32_t pointee = 0, elem = 0, block = 0, len_id = 0, base_loc = 0xffffffffu;
            int array_form = 0, taken = 0, have_struct = 0;
            uint32_t n_elem_next = 1;

            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL dissolve: scan var %u storage %u ptr %u\n", vid, words[i + 3], ptr);

            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == 32 /*OpTypePointer*/ && wc2 >= 4 && words[k + 1] == ptr)
                {
                    pointee = words[k + 3];
                    break;
                }
                k += wc2;
            }
            if (pointee)
            {
                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 28 /*OpTypeArray*/ && wc2 >= 4 && words[k + 1] == pointee)
                    {
                        elem = words[k + 2];
                        len_id = words[k + 3];
                        break;
                    }
                    k += wc2;
                }
            }
            if (elem && spv_patch_block_type(words, n, elem))
            {
                array_form = 1;
                block = elem;
            }
            /* single patch blocks ('patch out Blk { ... } b;') are NOT
             * dissolved: SPIRV-Cross handles them natively (the flattened
             * member names map onto MGL's records) and rewriting them breaks
             * the MSL for nested struct/array members — only the array form
             * ('Blk b[N]') needs the rewrite */
            if (!block)
            {
                if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL dissolve: var %u: not an array-of-patch-block (pointee %u elem %u)\n",
                            vid, pointee, elem);
                i += wc;
                continue;
            }
            if (array_form)
            {
                int32_t len = -1;

                for (size_t k = 5; k < n && len < 0; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 43 /*OpConstant*/ && wc2 >= 4 && words[k + 2] == len_id)
                    {
                        len = (int32_t)words[k + 3];
                        break;
                    }
                    k += wc2;
                }
                if (len <= 0 || len > 8)
                {
                    if (getenv("MGL_TESS_DEBUG"))
                        fprintf(stderr, "MGL dissolve: var %u: block array length %d out of range\n", vid, len);
                    i += wc;
                    continue;
                }
                n_elem_next = (uint32_t)len;
            }
            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == 71 /*OpDecorate*/ && wc2 >= 4 &&
                    words[k + 1] == vid && words[k + 2] == 30 /*Location*/)
                {
                    base_loc = words[k + 3];
                    break;
                }
                k += wc2;
            }
            for (int j = 0; j < n_jobs; j++)
                if (jobs[j].var == vid)
                    taken = 1;
            if (taken || base_loc == 0xffffffffu)
            {
                if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL dissolve: var %u: taken=%d base_loc=%u\n", vid, taken, base_loc);
                /* no explicit location on the block: MGL assigns interface
                 * locations at link time elsewhere; dissolving without them
                 * would mis-place the attributes — leave the module alone */
                i += wc;
                continue;
            }
            {
                struct PArrJob *job = &jobs[n_jobs];

                memset(job, 0, sizeof *job);
                job->var = vid;
                job->storage = words[i + 3];
                job->block = block;
                job->len_id = len_id;
                job->base_loc = base_loc;
                job->array_form = array_form;
                job->n_elem = n_elem_next;
                job->pointee = pointee;
                /* TES inputs (storage Input in a TessellationEvaluation
                 * module) take the per-element + local-copy form: the member
                 * arrays of a dissolved variable cannot be emitted as
                 * attributes once they become arrays-of-arrays */
                job->is_tes = (model == 2 /*TessellationEvaluation*/ && words[i + 3] == 1 /*Input*/);
                if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL dissolve: job var %u is_tes=%d (model %u storage %u)\n",
                            vid, job->is_tes, model, words[i + 3]);
                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 30 /*OpTypeStruct*/ && wc2 >= 2 && words[k + 1] == block)
                    {
                        job->n_members = wc2 - 2;
                        if (job->n_members > 32)
                            job->n_members = 32;
                        for (uint32_t m = 0; m < job->n_members; m++)
                            job->member_type[m] = words[k + 2 + m];
                        have_struct = 1;
                        break;
                    }
                    k += wc2;
                }
                if (!have_struct || !job->n_members)
                {
                    i += wc;
                    continue;
                }
                /* member names / Locations / location slots */
                {
                    uint32_t slot = 0;

                    for (uint32_t m = 0; m < job->n_members; m++)
                    {
                        snprintf(job->member_name[m], sizeof job->member_name[m], "m%u", m);
                        job->member_loc[m] = 0xffffffffu;
                        job->member_slot[m] = slot;
                        slot += spv_dis_leaf_count(words, n, job->member_type[m]);
                    }
                    job->block_slots = slot;
                }
                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 72 /*OpMemberDecorate*/ && wc2 >= 5 && words[k + 1] == block &&
                        words[k + 3] == 30 /*Location*/ && words[k + 2] < job->n_members)
                        job->member_loc[words[k + 2]] = words[k + 4];
                    if (op2 == 6 /*OpMemberName*/ && wc2 >= 4 && words[k + 1] == block &&
                        words[k + 2] < job->n_members)
                    {
                        const uint32_t *w = words + k + 3;
                        const uint32_t *we = words + k + wc2;
                        size_t o = 0;
                        uint32_t mi = words[k + 2];

                        while (w < we && o + 4 < sizeof job->member_name[0])
                        {
                            for (int b = 0; b < 4; b++)
                            {
                                char ch = (char)((*w >> (8 * b)) & 255);

                                if (!ch) { w = we; break; }
                                job->member_name[mi][o++] = ch;
                            }
                            w++;
                        }
                        job->member_name[mi][o] = 0;
                    }
                    k += wc2;
                }
                n_jobs++;
            }
        }
        i += wc;
    }
    if (!n_jobs)
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL dissolve: no jobs after scan\n");
        return NULL;
    }

    /* 2) every chain rooted at a job var must have a CONSTANT member index;
     *    whole-variable / whole-element uses decline the transform.  Only the
     *    opcodes whose relevant operand can BE a variable id are inspected —
     *    matching raw word 3 would hit OpConstant literals and storage-class
     *    numbers that merely equal the variable's id. */
    for (size_t i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

        if (wc == 0 || i + wc > n) break;
        if (op == 65 /*OpAccessChain*/ || op == 70 /*OpInBoundsAccessChain*/)
        {
            for (int j = 0; j < n_jobs; j++)
            {
                if (words[i + 3] != jobs[j].var)
                    continue;
                if (wc < (uint32_t)(jobs[j].array_form ? 6 : 5))
                {
                    if (!jobs[j].is_tes)
                    {
                        if (getenv("MGL_TESS_DEBUG"))
                            fprintf(stderr, "MGL dissolve: var %u: chain wc %u too short (whole-element use)\n",
                                    jobs[j].var, wc);
                        return NULL;   /* chain [var, elem]: whole-element pointer */
                    }
                    /* TES: re-roots to the local — any chain length is fine */
                    break;
                }
                if (jobs[j].is_tes)
                    break;            /* the local composite takes any indices */
                {
                    uint32_t midx = words[i + (jobs[j].array_form ? 5 : 4)];
                    int64_t m = spv_dis_const(words, n, midx);

                    if (m < 0 || (uint32_t)m >= jobs[j].n_members)
                    {
                        if (getenv("MGL_TESS_DEBUG"))
                            fprintf(stderr, "MGL dissolve: var %u: chain op %u wc %u: member idx %u not const\n",
                                    jobs[j].var, op, wc, midx);
                        return NULL;   /* dynamic member index: leave */
                    }
                }
                break;
            }
        }
        else if (op == 61 /*OpLoad*/ || op == 62 /*OpStore*/ || op == 46 /*OpCopyMemory*/)
        {
            /* word 3 is the pointer for OpLoad, the object for OpStore /
             * OpCopyMemory (their pointer is word 2) */
            for (int j = 0; j < n_jobs; j++)
                if (words[i + 3] == jobs[j].var || words[i + 2] == jobs[j].var)
                {
                    if (getenv("MGL_TESS_DEBUG"))
                        fprintf(stderr, "MGL dissolve: var %u: op %u uses the variable directly\n",
                                jobs[j].var, op);
                    return NULL;
                }
        }
        i += wc;
    }

    /* 3) create the replacement variables (TCS: one plain patch array per
     *    member; TES: per-(member,element) + an entry-copied local) */
    {
        uint32_t ins_types[8 * 32 * 8], ins_vars[8 * 32 * 4], ins_deco[8 * 32 * 12];
        uint32_t ins_entry[8 * 40 * 12];
        size_t n_types = 0, n_vars = 0, n_deco = 0, n_entry = 0;
        char namebuf[8 * 32 * 6];
        size_t namebuf_n = 0;
        /* Input→Function pointer-type map for TES chains re-rooted at the
         * entry-copied local (filled by the pre-pass below block 3) */
        uint32_t job_rt_from[64], job_rt_to[64];
        int job_rt_n = 0;

        for (int j = 0; j < n_jobs; j++)
        {
            struct PArrJob *job = &jobs[j];
            char iname[128];
            size_t inl = 0;

            for (size_t q = 5; q < n; )
            {
                uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                if (wq == 0 || q + wq > n) break;
                if (oq == 5 /*OpName*/ && wq >= 3 && words[q + 1] == job->var)
                {
                    const uint32_t *w = words + q + 2;
                    const uint32_t *we = words + q + wq;

                    while (w < we && inl + 4 < sizeof iname)
                    {
                        for (int b = 0; b < 4; b++)
                        {
                            char ch = (char)((*w >> (8 * b)) & 255);

                            if (!ch) { w = we; break; }
                            iname[inl++] = ch;
                        }
                        w++;
                    }
                    break;
                }
                q += wq;
            }
            iname[inl] = 0;
            if (!inl)
                snprintf(iname, sizeof iname, "patcharr");

            if (job->is_tes)
            {
                /* ---- TES input form: per-(member,element) patch variables
                 * ---- + a module-scope PRIVATE 'Blk[N]' copied at the entry
                 * block.  The member variables are single-level composites —
                 * the attribute shape SPIRV-Cross can always emit — and the
                 * local keeps dynamic element/member indices working. */
                uint32_t int_type = 0;
                uint32_t consts_ids[16];   /* small int constants 0..15 */
                uint32_t consts_val[16];
                int n_cst = 0;

                for (size_t q = 5; q < n; )
                {
                    uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                    if (wq == 0 || q + wq > n) break;
                    if (oq == 21 /*OpTypeInt*/ && wq >= 4 && words[q + 2] == 32)
                    {
                        int_type = words[q + 1];
                        break;
                    }
                    q += wq;
                }
                if (!int_type)
                    return NULL;
                for (size_t q = 5; q < n && n_cst < 16; )
                {
                    uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                    if (wq == 0 || q + wq > n) break;
                    if (oq == 43 /*OpConstant*/ && wq >= 4 && words[q + 1] == int_type &&
                        words[q + 3] < 16)
                    {
                        int seen = 0;

                        for (int t = 0; t < n_cst; t++)
                            if (consts_val[t] == words[q + 3]) seen = 1;
                        if (!seen)
                        {
                            consts_ids[n_cst] = words[q + 2];
                            consts_val[n_cst] = words[q + 3];
                            n_cst++;
                        }
                    }
                    q += wq;
                }
                /* create the missing ones (e/m indices) */
                for (uint32_t v = 0; v < 16 && n_cst < 16; v++)
                {
                    int seen = 0;

                    for (int t = 0; t < n_cst; t++)
                        if (consts_val[t] == v) seen = 1;
                    if (!seen && v >= job->n_elem + job->n_members)
                        break;
                    if (seen)
                        continue;
                    if (n_types + 4 > sizeof ins_types / 4) return NULL;
                    ins_types[n_types++] = (4u << 16) | 43;   /* OpConstant */
                    ins_types[n_types++] = int_type;
                    consts_ids[n_cst] = next_id++;
                    ins_types[n_types++] = consts_ids[n_cst];
                    consts_val[n_cst] = v;
                    ins_types[n_types++] = v;
                    n_cst++;
                }

                /* module-scope PRIVATE 'Blk[N]' (the entry block only holds the
                 * copies into it).  SPIRV-Cross localizes Private globals into
                 * the entry function (localize_global_variables) and then
                 * plumbs them into every helper that touches them
                 * (extract_global_variables_from_functions — which SKIPS
                 * Function-storage locals, so a Function local referenced
                 * from a helper compiles to an undeclared identifier:
                 * KHR PIQ program_input patch_in.block_array.* reads the
                 * block from readInputs(), not main()).  Private storage
                 * keeps the re-rooted chains legal SPIR-V from any function. */
                {
                    uint32_t pp = 0;

                    for (size_t q = 5; q < n; )
                    {
                        uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                        if (wq == 0 || q + wq > n) break;
                        if (oq == 32 && wq >= 4 && words[q + 2] == 6 /*Private*/ &&
                            words[q + 3] == job->pointee)
                        {
                            pp = words[q + 1];
                            break;
                        }
                        q += wq;
                    }
                    if (!pp)
                    {
                        pp = next_id++;
                        if (n_types + 4 > sizeof ins_types / 4) return NULL;
                        ins_types[n_types++] = (4u << 16) | 32;
                        ins_types[n_types++] = pp;
                        ins_types[n_types++] = 6 /*Private*/;
                        ins_types[n_types++] = job->pointee;
                    }
                    job->local_var = next_id++;
                    if (n_vars + 4 > sizeof ins_vars / 4) return NULL;
                    ins_vars[n_vars++] = (4u << 16) | 59;   /* OpVariable */
                    ins_vars[n_vars++] = pp;
                    ins_vars[n_vars++] = job->local_var;
                    ins_vars[n_vars++] = 6 /*Private*/;
                }

                for (uint32_t e = 0; e < job->n_elem; e++)
                    for (uint32_t m = 0; m < job->n_members; m++)
                    {
                        uint32_t mt = job->member_type[m];
                        uint32_t aptr = 0, loc;
                        int cst_e = -1, cst_m = -1;

                        for (int t = 0; t < n_cst; t++)
                        {
                            if (consts_val[t] == e) cst_e = t;
                            if (consts_val[t] == m) cst_m = t;
                        }
                        /* ptr Input<memberType> for the per-element variable */
                        for (size_t q = 5; q < n; )
                        {
                            uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                            if (wq == 0 || q + wq > n) break;
                            if (oq == 32 && wq >= 4 && words[q + 2] == 1 /*Input*/ &&
                                words[q + 3] == mt)
                            {
                                aptr = words[q + 1];
                                break;
                            }
                            q += wq;
                        }
                        for (size_t q = 0; q + 4 <= n_types; q += 4)
                            if ((ins_types[q] & 0xffff) == 32 && ins_types[q + 2] == 1 &&
                                ins_types[q + 3] == mt)
                            {
                                aptr = ins_types[q + 1];
                                break;
                            }
                        if (!aptr)
                        {
                            aptr = next_id++;
                            if (n_types + 4 > sizeof ins_types / 4) return NULL;
                            ins_types[n_types++] = (4u << 16) | 32;
                            ins_types[n_types++] = aptr;
                            ins_types[n_types++] = 1 /*Input*/;
                            ins_types[n_types++] = mt;
                        }
                        /* ptr Input<memberType> was minted above (aptr); the
                         * local's store chain types are minted by
                         * tess_copy_walk with dst_sc (Private) storage */
                        job->new_var2[e][m] = next_id++;
                        if (n_vars + 4 > sizeof ins_vars / 4) return NULL;
                        ins_vars[n_vars++] = (4u << 16) | 59;
                        ins_vars[n_vars++] = aptr;
                        ins_vars[n_vars++] = job->new_var2[e][m];
                        ins_vars[n_vars++] = 1 /*Input*/;
                        loc = job->member_loc[m] != 0xffffffffu
                                  ? job->member_loc[m] + e * job->block_slots
                                  : job->base_loc + e * job->block_slots + job->member_slot[m];
                        if (n_deco + 7 > sizeof ins_deco / 4) return NULL;
                        ins_deco[n_deco++] = (3u << 16) | 71;
                        ins_deco[n_deco++] = job->new_var2[e][m];
                        ins_deco[n_deco++] = SPV_DEC_PATCH;
                        ins_deco[n_deco++] = (4u << 16) | 71;
                        ins_deco[n_deco++] = job->new_var2[e][m];
                        ins_deco[n_deco++] = 30 /*Location*/;
                        ins_deco[n_deco++] = loc;
                        {
                            char full[192];

                            snprintf(full, sizeof full, "mgl_ds_%s_%u", job->member_name[m], e);
                            if (namebuf_n + 1 + strlen(full) / 4 + 1 > sizeof namebuf / 4)
                                return NULL;
                            namebuf_words(namebuf, &namebuf_n, job->new_var2[e][m], full);
                        }
                        /* entry copy: local[e][m] = var, one leaf at a time
                         * (see struct TessCopyCtx above for why composites
                         * cannot be stored or loaded whole here) */
                        if (cst_e < 0 || cst_m < 0)
                            return NULL;
                        {
                            struct TessCopyCtx cc;

                            memset(&cc, 0, sizeof cc);
                            cc.words = words;
                            cc.n = n;
                            cc.ins_types = ins_types;
                            cc.n_types = &n_types;
                            cc.cap_types = sizeof ins_types / 4;
                            cc.ins_entry = ins_entry;
                            cc.n_entry = &n_entry;
                            cc.cap_entry = sizeof ins_entry / 4;
                            cc.next_id = &next_id;
                            cc.local_var = job->local_var;
                            cc.src_var = job->new_var2[e][m];
                            cc.dst_sc = 6 /*Private — matches the local's storage*/;
                            cc.dst_n = 2;
                            cc.dst_idx[0] = consts_ids[cst_e];
                            cc.dst_idx[1] = consts_ids[cst_m];
                            cc.src_n = 0;
                            memcpy(cc.cst_id, consts_ids, sizeof consts_ids);
                            memcpy(cc.cst_val, consts_val, sizeof consts_val);
                            cc.n_cst = n_cst;
                            if (tess_copy_walk(&cc, mt, 0))
                                return NULL;
                        }
                    }
                continue;
            }

            for (uint32_t m = 0; m < job->n_members; m++)
            {
                uint32_t at = job->member_type[m], aptr = 0;

                /* array form: array<memberType,N> with the block array's length */
                if (job->array_form)
                {
                    at = 0;
                    for (size_t q = 5; q < n; )
                    {
                        uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                        if (wq == 0 || q + wq > n) break;
                        if (oq == 28 && wq == 4 && words[q + 2] == job->member_type[m] &&
                            words[q + 3] == job->len_id)
                        {
                            at = words[q + 1];
                            break;
                        }
                        q += wq;
                    }
                    for (size_t q = 0; q + 4 <= n_types; q += 4)
                        if ((ins_types[q] & 0xffff) == 28 && ins_types[q + 2] == job->member_type[m] &&
                            ins_types[q + 3] == job->len_id)
                        {
                            at = ins_types[q + 1];
                            break;
                        }
                    if (!at)
                    {
                        at = next_id++;
                        if (n_types + 4 > sizeof ins_types / 4) return NULL;
                        ins_types[n_types++] = (4u << 16) | 28;
                        ins_types[n_types++] = at;
                        ins_types[n_types++] = job->member_type[m];
                        ins_types[n_types++] = job->len_id;
                    }
                }
                for (size_t q = 5; q < n; )
                {
                    uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                    if (wq == 0 || q + wq > n) break;
                    if (oq == 32 && wq >= 4 && words[q + 2] == job->storage &&
                        words[q + 3] == at)
                    {
                        aptr = words[q + 1];
                        break;
                    }
                    q += wq;
                }
                for (size_t q = 0; q + 4 <= n_types; q += 4)
                    if ((ins_types[q] & 0xffff) == 32 && ins_types[q + 2] == job->storage &&
                        ins_types[q + 3] == at)
                    {
                        aptr = ins_types[q + 1];
                        break;
                    }
                if (!aptr)
                {
                    aptr = next_id++;
                    if (n_types + 4 > sizeof ins_types / 4) return NULL;
                    ins_types[n_types++] = (4u << 16) | 32;
                    ins_types[n_types++] = aptr;
                    ins_types[n_types++] = job->storage;
                    ins_types[n_types++] = at;
                }
                job->new_var[m] = next_id++;
                if (n_vars + 4 > sizeof ins_vars / 4) return NULL;
                ins_vars[n_vars++] = (4u << 16) | 59;
                ins_vars[n_vars++] = aptr;
                ins_vars[n_vars++] = job->new_var[m];
                ins_vars[n_vars++] = job->storage;
                {
                    uint32_t loc = job->member_loc[m] != 0xffffffffu
                                       ? job->member_loc[m]
                                       : job->base_loc + job->member_slot[m];

                    if (n_deco + 7 > sizeof ins_deco / 4) return NULL;
                    ins_deco[n_deco++] = (3u << 16) | 71;
                    ins_deco[n_deco++] = job->new_var[m];
                    ins_deco[n_deco++] = SPV_DEC_PATCH;
                    ins_deco[n_deco++] = (4u << 16) | 71;
                    ins_deco[n_deco++] = job->new_var[m];
                    ins_deco[n_deco++] = 30 /*Location*/;
                    ins_deco[n_deco++] = loc;
                }
                {
                    char full[192];
                    size_t need, w4;

                    snprintf(full, sizeof full, "mgl_ds_%s", job->member_name[m]);
                    need = strlen(full);
                    w4 = need / 4 + 1;          /* keep in sync with namebuf_words */
                    if (namebuf_n + 1 + w4 > sizeof namebuf / 4) return NULL;
                    namebuf_words(namebuf, &namebuf_n, job->new_var[m], full);
                }
            }
        }

        /* TES chains re-rooted at the entry-copied local keep their result
         * type, whose storage class must follow the BASE (Private, not
         * Input).  Mint a Private counterpart for every rewritten chain's
         * Input pointer type up front — the rebuild looks the mapping up by
         * the old id (ins_types is emitted at the first OpFunction, before
         * any rewritten chain, so the new types dominate their uses). */
        {
            uint32_t rt_map_from[64], rt_map_to[64];
            int n_rt_map = 0;

            for (int j = 0; j < n_jobs; j++)
            {
                if (!jobs[j].is_tes)
                    continue;
                for (size_t i = 5; i < n; )
                {
                    uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
                    int known = 0;

                    if (wc == 0 || i + wc > n) break;
                    if ((op == 65 || op == 70) && wc >= 4 && words[i + 3] == jobs[j].var)
                    {
                        uint32_t rt = words[i + 1];

                        for (int t = 0; t < n_rt_map; t++)
                            if (rt_map_from[t] == rt) known = 1;
                        if (!known && n_rt_map < 64)
                        {
                            uint32_t pointee = 0, nt;

                            for (size_t q = 5; q < n; )
                            {
                                uint32_t wq = words[q] >> 16, oq = words[q] & 0xffff;

                                if (wq == 0 || q + wq > n) break;
                                if (oq == 32 && wq >= 4 && words[q + 1] == rt)
                                {
                                    pointee = words[q + 3];
                                    break;
                                }
                                q += wq;
                            }
                            nt = pointee ? spv_lower_ptr_type(words, n, ins_types, &n_types,
                                                             sizeof ins_types / 4, &next_id,
                                                             6 /*Private*/, pointee) : 0;
                            if (nt)
                            {
                                rt_map_from[n_rt_map] = rt;
                                rt_map_to[n_rt_map] = nt;
                                n_rt_map++;
                            }
                        }
                    }
                    i += wc;
                }
            }
            memcpy(job_rt_from, rt_map_from, sizeof rt_map_from);
            memcpy(job_rt_to, rt_map_to, sizeof rt_map_to);
            job_rt_n = n_rt_map;
        }

        /* 4) rebuild: names go into the DEBUG section (after the last existing
         *    OpName/OpMemberName), decorations into the ANNOTATION section
         *    (after the last existing OpDecorate/OpMemberDecorate), types+vars
         *    before the first function, chains rewritten (member dropped for
         *    the TCS per-member form; re-rooted to the entry-copied local for
         *    the TES per-element form), and the entry copy right after the
         *    entry block's OpLabel */
        {
            size_t first_fn = 0, last_name = 0, last_deco = 0, last_hdr = 0;
            size_t name_at, deco_at, o = 0;
            size_t entry_fn = 0, entry_fn_at = 0;
            size_t first_label = 0;    /* entry block of the ENTRY-POINT function */
            int want_entry = 0;

            for (int j = 0; j < n_jobs; j++)
                want_entry |= jobs[j].is_tes;

            for (size_t i = 5; i < n; )
            {
                uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

                if (wc == 0 || i + wc > n) break;
                if (op == 15 /*OpEntryPoint*/ && wc >= 4 && !entry_fn)
                    entry_fn = words[i + 2];      /* id of the entry function */
                if (op == 54 && !first_fn) first_fn = i;
                if (op == 54 && wc >= 4 && words[i + 2] == entry_fn && entry_fn)
                    entry_fn_at = i;              /* the entry point's OpFunction */
                if (want_entry && entry_fn_at && !first_label && op == 248 /*OpLabel*/)
                    first_label = i;
                if (!first_fn)
                {
                    if (op == 5 || op == 6)          /* OpName / OpMemberName */
                        last_name = i + wc;
                    if (op == 71 || op == 72)        /* OpDecorate / OpMemberDecorate */
                        last_deco = i + wc;
                    if (op == 14 || op == 15 || op == 16)  /* header block */
                        last_hdr = i + wc;
                }
                i += wc;
            }
            name_at = last_name ? last_name : (last_hdr ? last_hdr : 5);
            deco_at = last_deco ? last_deco : name_at;
            if (!first_fn || (want_entry && !first_label) || deco_at < name_at)
                return NULL;
            /* the entry copy goes after the entry block's OpVariable run —
             * function OpVariables must stay the first instructions of the
             * first block (SPIR-V §2.6) */
            if (want_entry)
            {
                size_t k = first_label + (words[first_label] >> 16);

                while (k < n && (words[k] & 0xffff) == 59 /*OpVariable*/)
                    k += words[k] >> 16;
                first_label = k;      /* insertion point (kept in first_label) */
            }

            out = (uint32_t *)malloc((n + namebuf_n + n_deco + n_types + n_vars +
                                      n_entry + 8 * 32 * 32 + 64) * 4);
            if (!out)
                return NULL;
            memcpy(out + o, words, 5 * 4);      /* module header */
            o = 5;
            for (size_t i = 5; i < name_at; )
            {
                uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

                if (wc == 0 || i + wc > name_at) break;
                if (op == 5 /*OpName*/ && wc >= 3)
                {
                    int dead = 0;

                    for (int j = 0; j < n_jobs; j++)
                        if (words[i + 1] == jobs[j].var) dead = 1;
                    if (dead)
                    {
                        i += wc;
                        continue;
                    }
                }
                if (op == 15 /*OpEntryPoint*/ && wc >= 4)
                    o += spv_rewrite_entry_point(words, i, jobs, n_jobs, out + o);
                else
                {
                    memcpy(out + o, words + i, wc * 4);
                    o += wc;
                }
                i += wc;
            }
            memcpy(out + o, namebuf, namebuf_n * 4);
            o += namebuf_n;
            /* between the debug and annotation anchors: drop decorations that
             * target the stripped block variables (a dangling OpDecorate on an
             * undefined id is invalid SPIR-V) */
            for (size_t i = name_at; i < deco_at; )
            {
                uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

                if (wc == 0 || i + wc > deco_at) break;
                if ((op == 71 || op == 72 || op == 5 || op == 6) && wc >= 3)
                {
                    int dead = 0;

                    for (int j = 0; j < n_jobs; j++)
                        if (words[i + 1] == jobs[j].var) dead = 1;
                    if (dead)
                    {
                        i += wc;
                        continue;
                    }
                }
                memcpy(out + o, words + i, wc * 4);
                o += wc;
                i += wc;
            }
            memcpy(out + o, ins_deco, n_deco * 4);
            o += n_deco;
            for (size_t i = deco_at; i < n; )
            {
                uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
                int rwj = -1, rw_tes = 0;
                int64_t m = -1;

                if (wc == 0 || i + wc > n) break;
                for (int j = 0; j < n_jobs; j++)
                    if (words[i + 3] == jobs[j].var && (op == 65 || op == 70))
                    {
                        if (jobs[j].is_tes)
                        {
                            /* any chain re-roots to the local with ALL its
                             * indices preserved */
                            if (wc >= 4)
                            {
                                rwj = j;
                                rw_tes = 1;
                            }
                            break;
                        }
                        if (wc >= (uint32_t)(jobs[j].array_form ? 6 : 5))
                        {
                            rwj = j;
                            m = spv_dis_const(words, n, words[i + (jobs[j].array_form ? 5 : 4)]);
                            break;
                        }
                    }
                if ((op == 71 /*OpDecorate*/ || op == 5 /*OpName*/) && wc >= 3)
                {
                    /* decorations/names of the stripped variable would dangle */
                    int dead = 0;

                    for (int j = 0; j < n_jobs; j++)
                        if (words[i + 1] == jobs[j].var) dead = 1;
                    if (dead)
                    {
                        i += wc;
                        continue;
                    }
                }
                if (op == 59 /*OpVariable*/ && wc >= 4)
                {
                    /* the dissolved block variable itself is DEAD: every use was
                     * rewritten to a member variable.  SPIRV-Cross force-emits
                     * Output variables regardless of the entry point's interface
                     * list, so an unstripped declaration would come back as an
                     * arrays-of-arrays struct member alongside the replacement
                     * (and duplicate the canonical record). */
                    int dead = 0;

                    for (int j = 0; j < n_jobs; j++)
                        if (words[i + 2] == jobs[j].var) dead = 1;
                    if (dead)
                    {
                        i += wc;
                        continue;
                    }
                }
                if (op == 54 && first_fn == i)
                {
                    memcpy(out + o, ins_types, n_types * 4);
                    o += n_types;
                    memcpy(out + o, ins_vars, n_vars * 4);
                    o += n_vars;
                }
                if (op == 15 /*OpEntryPoint*/ && wc >= 4)
                {
                    o += spv_rewrite_entry_point(words, i, jobs, n_jobs, out + o);
                    i += wc;
                    continue;
                }
                if (want_entry && first_label == i)
                {
                    memcpy(out + o, ins_entry, n_entry * 4);
                    o += n_entry;
                }
                if (rwj >= 0 && rw_tes)
                {
                    memcpy(out + o, words + i, wc * 4);
                    out[o + 3] = jobs[rwj].local_var;   /* re-root at the local */
                    /* the chain's result type must follow the base's storage
                     * class (Private): swap in the minted counterpart */
                    for (int t = 0; t < job_rt_n; t++)
                        if (job_rt_from[t] == words[i + 1])
                        {
                            out[o + 1] = job_rt_to[t];
                            break;
                        }
                    o += wc;
                    i += wc;
                    continue;
                }
                if (rwj >= 0 && m >= 0 && (uint32_t)m < jobs[rwj].n_members)
                {
                    uint32_t drop = jobs[rwj].array_form ? 5 : 4;   /* the member index */

                    out[o++] = ((wc - 1) << 16) | op;
                    out[o++] = words[i + 1];
                    out[o++] = words[i + 2];
                    out[o++] = jobs[rwj].new_var[m];
                    for (uint32_t q = 4; q < wc; q++)
                        if (q != drop)
                            out[o++] = words[i + q];
                    i += wc;
                    continue;
                }
                memcpy(out + o, words + i, wc * 4);
                o += wc;
                i += wc;
            }
            out[3] = next_id;
            *out_n = o;
            return out;
        }
    }
}


/* does this module contain a chain rooted at a per-vertex BLOCK variable that
 * ENDS at a direct member with a dynamic control-point index?  (SPIRV-Cross
 * asserts on those — "emit_tessellation_access_chain: index != -1" — when the
 * block also has composite members: the block's member-index map is polluted
 * by the flattened composites.  The dissolve normally rewrites these chains
 * away; if it did not run, compiling the module can abort the process.) */
bool mgl_tess_module_has_block_leafmember_chains(const uint32_t *words, size_t n)
{
    struct SpvLowerCtx c;

    if (!spv_lower_setup(&c, words, n))
        return false;

    for (uint32_t k = 0; k < c.ninst; k++)
    {
        uint32_t off, wc, base, idx[8], nidx = 0;
        int var, q;
        uint32_t t;
        bool member_depth = false;

        if (c.op[k] != SPV_OP_ACCESSCHAIN && c.op[k] != SPV_OP_INBOUNDSAC)
            continue;
        off = c.off[k];
        wc = c.wc[k];
        if (wc < 6)
            continue;
        base = c.w[off + 3];
        var = spv_lower_find(&c, base);
        if (var < 0 || c.op[var] != SPV_OP_VARIABLE || c.w[c.off[var] + 3] != SPV_SC_INPUT)
            continue;
        if (spv_lower_var_is_patch(&c, base))
            continue;
        t = spv_lower_pointee(&c, c.w[c.off[var] + 1]);
        if (spv_lower_type_kind(&c, t) != SPV_T_ARRAY)
            continue;
        t = spv_lower_elem_type(&c, t, 0);
        if (spv_lower_type_kind(&c, t) != SPV_T_STRUCT)
            continue;
        {
            bool is_block = false;

            for (uint32_t d = 0; d < c.ninst; d++)
                if (c.op[d] == 71 /*OpDecorate*/ && c.wc[d] >= 3 &&
                    c.w[c.off[d] + 1] == t && c.w[c.off[d] + 2] == 2 /*Block*/)
                {
                    is_block = true;
                    break;
                }
            if (!is_block)
                continue;
        }
        /* needs a Location (a user block; gl_PerVertex chains are fine) */
        {
            bool loc = false;

            for (uint32_t d = 0; d < c.ninst; d++)
                if (c.op[d] == 71 /*OpDecorate*/ && c.wc[d] >= 4 &&
                    c.w[c.off[d] + 1] == base && c.w[c.off[d] + 2] == 30 /*Location*/)
                {
                    loc = true;
                    break;
                }
            if (!loc)
                continue;
        }
        for (q = 4; q < (int)wc && nidx < 8; q++)
            idx[nidx++] = c.w[off + q];
        /* dynamic control point + the chain ends at a member: the risky shape */
        if (spv_lower_const(&c, idx[0]) >= 0)
            continue;
        for (q = 1; q < (int)nidx; q++)
        {
            int kind = spv_lower_type_kind(&c, t);

            if (kind == SPV_T_STRUCT)
            {
                int64_t lit = spv_lower_const(&c, idx[q]);

                if (lit < 0)
                    break;
                t = spv_lower_elem_type(&c, t, (uint32_t)lit);
                member_depth = true;
            }
            else if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
            {
                if (member_depth && spv_lower_const(&c, idx[q]) < 0)
                    break;              /* dynamic member-array index: throws cleanly */
                t = spv_lower_elem_type(&c, t, 0);
                member_depth = true;
            }
            else
                break;
            if (q + 1 == (int)nidx)
                return true;            /* ends at a member of the block */
        }
    }
    return false;
}

uint32_t *mgl_tess_lower_dynamic_io(const uint32_t *words, size_t n, size_t *out_n,
                                    const struct SpvDissolveMap *maps, uint32_t n_maps)
{
    struct SpvLowerCtx c;
    struct SpvLowerSite sites[64];
    struct SpvLowerLeaf leaf_tab[SPV_LOWER_MAX_LEAVES];
    uint32_t n_leaves = 0, n_sites = 0, next_id;
    struct SpvNewType new_types[192];
    uint32_t n_new_types = 0;
    struct SpvNewConst new_consts[256];
    uint32_t n_new_consts = 0;
    uint32_t *out = NULL;
    size_t out_cap = n + 16384, out_len = 0;
    uint32_t first_function = 0;
    uint32_t arrptr[64], loaded[64], fvar[64], elemc[64], idx_type = 0;

    if (!spv_lower_setup(&c, words, n))
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess lower: setup failed (%zu words)\n", n);
        return NULL;
    }
    next_id = words[3];

    for (uint32_t k = 0; k < c.ninst; k++)
    {
        if (c.op[k] == SPV_OP_FUNCTION && !first_function)
            first_function = c.off[k];
        if ((c.op[k] == SPV_OP_ACCESSCHAIN || c.op[k] == SPV_OP_INBOUNDSAC) &&
            n_sites < 64 && spv_lower_offends(&c, k, &sites[n_sites]))
        {
            for (int f = (int)k; f >= 0; f--)
                if (c.op[f] == SPV_OP_FUNCTION) { sites[n_sites].func = (uint32_t)f; break; }
            sites[n_sites].elem_type = spv_lower_chain_elem_type(&c, &sites[n_sites]);
            n_sites++;
        }
    }
    if (!n_sites)
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess lower: no dynamic-IO sites (%zu words)\n", n);
    }

    /* validate the sites: need a chain-elem type, an index type (from a
     * constant prefix member index) and the array length */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        uint32_t off = c.off[sites[s].inst];
        int ck = -1;

        if (!sites[s].elem_type || sites[s].k < 4 ||
            !spv_lower_composite_len(&c, sites[s].arr_type))
        {
            sites[s].inst = 0xffffffff;
            continue;
        }
        /* struct- or array-typed elements cannot be copied with one whole
         * element load (a struct load from a flattened stage-in aborts
         * SPIRV-Cross: "Cannot resolve expression type") — copy leaf by leaf
         * instead, one constant-indexed load per leaf (dEQP
         * user_defined_io.per_vertex_block.*, per_patch_block_array.*) */
        {
            uint32_t arr_elem = spv_lower_elem_type(&c, sites[s].arr_type, 0);
            int ek = spv_lower_type_kind(&c, arr_elem);
            if (ek == SPV_T_STRUCT || ek == SPV_T_ARRAY || sites[s].dmap)
            {
                uint32_t path[6];
                sites[s].leaf_mode = 1;
                sites[s].leaf_off = n_leaves;
                sites[s].leaf_n = spv_lower_leaves(&c, arr_elem, path, 0,
                                                   leaf_tab + n_leaves,
                                                   SPV_LOWER_MAX_LEAVES - n_leaves);
                n_leaves += sites[s].leaf_n;
                if (!sites[s].leaf_n || n_leaves > SPV_LOWER_MAX_LEAVES)
                {
                    sites[s].inst = 0xffffffff;
                    continue;
                }
            }
        }
        /* a site rooted at a DISSOLVED block's original variable: the chain
         * member index (words[off+5]) names the member; if that member was
         * dissolved per-(element,field), the leaf copies must read the
         * dissolved variables, not the block (the block's attribute span
         * would collide with the dissolved variables') */
        if (maps && n_maps)
        {
            uint32_t base = c.w[off + 3];
            int is_block_var = 0;

            for (uint32_t q = 0; q < n_maps; q++)
                if (maps[q].var == base)
                {
                    is_block_var = 1;
                    break;
                }
            if (is_block_var)
            {
                int64_t mval = (sites[s].k > 5) ? spv_lower_const(&c, c.w[off + 5]) : -1;

                for (uint32_t q = 0; q < n_maps; q++)
                    if (maps[q].var == base && maps[q].member == (uint32_t)mval &&
                        (maps[q].elems || maps[q].var0))
                    {
                        sites[s].dmap = (uint8_t)(q + 1);
                        break;
                    }
                if (!sites[s].dmap)
                {
                    /* the member was not leaf-dissolved: keeping the chain
                     * rooted at the block risks the attribute collision */
                    sites[s].inst = 0xffffffff;
                    continue;
                }
            }
        }

        /* the index type: from an index operand of the chain — this chain's
         * own first index (k > 4) or, for split chains whose first operand
         * already IS the dynamic index, the inner chain's.  A CONSTANT gives
         * it directly, but the control-point index is usually a loop Phi /
         * OpLoad (its result type is operand 1 for those too); rejecting
         * non-constant indices dropped every block-form site and SPIRV-Cross
         * refused the module (dEQP-GL45-ES31 tessellation.user_defined_io
         * .per_vertex_block.*). */
        idx_type = 0;
        if (sites[s].k > 4)
            ck = spv_lower_find(&c, c.w[off + 4]);
        else if (sites[s].prefix_n)
            ck = spv_lower_find(&c, sites[s].prefix[0]);
        if (ck < 0)
        {
            sites[s].inst = 0xffffffff;
            continue;
        }
        if (c.op[ck] == SPV_OP_CONSTANT || c.op[ck] == SPV_OP_PHI ||
            c.op[ck] == SPV_OP_LOAD || c.op[ck] == SPV_OP_VARIABLE ||
            c.op[ck] == SPV_OP_ACCESSCHAIN || c.op[ck] == SPV_OP_COMPOSITE_EXTRACT)
            idx_type = c.w[c.off[ck] + 1];
        if (!idx_type)
        {
            sites[s].inst = 0xffffffff;
            continue;
        }
    }

    /* guard 1 — ARRAY-field sites used to be declined wholesale.  The copy
     * chains descend into the field with CONSTANT indices, which SPIRV-Cross
     * resolves into the flattened attribute leaves; the historic failures
     * came from struct-typed elements (whole-element loads, now copied
     * leaf-by-leaf) and per-PATCH arrays mis-read as control-point arrays
     * (now sites of their own).  MGL_TESS_ARRAY_DECLINE restores the old
     * behaviour for A/B-testing. */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff)
            continue;
        if (spv_lower_type_kind(&c, sites[s].arr_type) == SPV_T_ARRAY &&
            getenv("MGL_TESS_ARRAY_DECLINE"))
        {
            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL tess lower: array-field site declined (MGL_TESS_ARRAY_DECLINE)\n");
            return NULL;
        }
    }

    /* guard 2: any access chain that still descends into an array-typed FIELD of
     * an Input interface (a constant trailing index the lowering does not
     * rewrite, or a site dropped during validation) aborts SPIRV-Cross —
     * decline the lowering entirely in that case */
    if (n_sites && spv_lower_fatal_remaining(&c, sites, n_sites))
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess lower: declining — remaining array-field access chain would abort SPIRV-Cross\n");
        return NULL;
    }

    /* per-site ids: element pointer, element value, function variable */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff) { fvar[s] = arrptr[s] = loaded[s] = elemc[s] = 0; continue; }
        arrptr[s] = next_id++;
        loaded[s] = next_id++;
        fvar[s] = next_id++;
        elemc[s] = next_id++;
    }

    /* constants 0..len-1 of the index type, shared across sites */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff) continue;
        uint32_t len = spv_lower_composite_len(&c, sites[s].arr_type);
        for (uint32_t v = 0; v < len; v++)
        {
            bool have = false;
            for (uint32_t q = 0; q < n_new_consts; q++)
                if (new_consts[q].type == idx_type && new_consts[q].value == v) { have = true; break; }
            /* an existing module constant with the same value also works */
            if (!have)
                for (uint32_t q = 0; q < c.ninst; q++)
                    if (c.op[q] == SPV_OP_CONSTANT && c.wc[q] >= 4 &&
                        c.w[c.off[q] + 1] == idx_type && (int32_t)c.w[c.off[q] + 3] == (int32_t)v)
                    { have = true; break; }
            if (!have && n_new_consts < 256)
            {
                new_consts[n_new_consts].id = next_id++;
                new_consts[n_new_consts].type = idx_type;
                new_consts[n_new_consts].value = v;
                n_new_consts++;
            }
        }
    }

    /* pointer types */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff) continue;
        uint32_t arr_elem = spv_lower_elem_type(&c, sites[s].arr_type, 0);
        if (!arr_elem) { sites[s].inst = 0xffffffff; continue; }
        (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_INPUT, arr_elem);
        (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_FUNCTION, arr_elem);
        (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_FUNCTION, sites[s].arr_type);
        (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_FUNCTION, sites[s].elem_type);
        if (sites[s].leaf_mode)
            for (uint32_t q = 0; q < sites[s].leaf_n; q++)
            {
                uint32_t lt = leaf_tab[sites[s].leaf_off + q].type;
                (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_INPUT, lt);
                (void)spv_lower_type_alloc(new_types, &n_new_types, 192, &next_id, SPV_SC_FUNCTION, lt);
            }
    }
    /* drop sites whose pointer types could not be provided */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff) continue;
        uint32_t arr_elem = spv_lower_elem_type(&c, sites[s].arr_type, 0);
        if (!spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_INPUT, arr_elem) ||
            !spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_FUNCTION, arr_elem) ||
            !spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_FUNCTION, sites[s].arr_type) ||
            !spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_FUNCTION, sites[s].elem_type))
            sites[s].inst = 0xffffffff;
        if (sites[s].leaf_mode && sites[s].inst != 0xffffffff)
            for (uint32_t q = 0; q < sites[s].leaf_n; q++)
            {
                uint32_t lt = leaf_tab[sites[s].leaf_off + q].type;
                if (!spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_INPUT, lt) ||
                    !spv_lower_ptr_of(&c, new_types, n_new_types, SPV_SC_FUNCTION, lt))
                {
                    sites[s].inst = 0xffffffff;
                    break;
                }
            }
    }

    /* helper to resolve a (storage, pointee) pointer type from the module or
       the freshly allocated ones */
    #define SPV_PTR_OF(_storage, _pointee) spv_lower_ptr_of(&c, new_types, n_new_types, (_storage), (_pointee))

    /* id bound: the copy loop below allocates TWO fresh ids per copied element
     * (arrptr/loaded) as it emits — the bound must cover them or SPIRV-Cross
     * indexes ids[] past its size and crashes in analyze_variable_scope
     * (headless test_tess_dynio: mat2 member of a tess INPUT block). */
    uint32_t bound = next_id;
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff) continue;
        bound += sites[s].leaf_mode
                     ? 3u * spv_lower_composite_len(&c, sites[s].arr_type) * sites[s].leaf_n
                     : 3u * spv_lower_composite_len(&c, sites[s].arr_type);
    }

    out = (uint32_t *)malloc(out_cap * sizeof(uint32_t));
    if (!out) { spv_lower_teardown(&c); return NULL; }

    /* where each owning function's OpVariables go: right after its FIRST label */
    uint32_t var_at[64];
    for (uint32_t s = 0; s < n_sites; s++)
    {
        var_at[s] = 0xffffffff;
        if (sites[s].inst == 0xffffffff) continue;
        for (uint32_t g = sites[s].func; g < c.ninst; g++)
            if (c.op[g] == SPV_OP_LABEL) { var_at[s] = g; break; }
    }

    #define LOW_PUSH(x) do { \
        if (out_len + 1 >= out_cap) { \
            out_cap *= 2; \
            out = (uint32_t *)realloc(out, out_cap * sizeof(uint32_t)); \
            if (!out) { spv_lower_teardown(&c); return NULL; } \
        } \
        out[out_len++] = (x); \
    } while (0)

    /* header (new bound) */
    for (int q = 0; q < 5; q++)
        LOW_PUSH(q == 3 ? bound : words[q]);

    for (uint32_t k = 0; k < c.ninst; k++)
    {
        uint32_t off = c.off[k], wc = c.wc[k], op = c.op[k];

        /* new types and constants: just before the first function */
        if (first_function && off == first_function)
        {
            for (uint32_t q = 0; q < n_new_types; q++)
            {
                LOW_PUSH((4u << 16) | SPV_OP_TYPE_POINTER);
                LOW_PUSH(new_types[q].id);
                LOW_PUSH(new_types[q].storage);
                LOW_PUSH(new_types[q].pointee);
            }
            for (uint32_t q = 0; q < n_new_consts; q++)
            {
                LOW_PUSH((4u << 16) | SPV_OP_CONSTANT);
                LOW_PUSH(new_consts[q].type);
                LOW_PUSH(new_consts[q].id);
                LOW_PUSH(new_consts[q].value);
            }
        }

        /* function variables: right after the entry block's OpLabel */
        if (op == SPV_OP_LABEL)
        {
            for (uint32_t s = 0; s < n_sites; s++)
            {
                if (sites[s].inst == 0xffffffff || var_at[s] != k) continue;
                uint32_t ptr = SPV_PTR_OF(SPV_SC_FUNCTION, sites[s].arr_type);
                if (!ptr) { sites[s].inst = 0xffffffff; continue; }
                LOW_PUSH((4u << 16) | SPV_OP_VARIABLE);
                LOW_PUSH(ptr);
                LOW_PUSH(fvar[s]);
                LOW_PUSH(SPV_SC_FUNCTION);
            }
        }

        /* an offending chain: constant-indexed element copy + dynamic index
         * on the copy (same result id) */
        int s = -1;
        for (uint32_t q = 0; q < n_sites; q++)
            if (sites[q].inst == k) { s = (int)q; break; }
        if (s >= 0)
        {
            struct SpvLowerSite *st = &sites[s];
            uint32_t arr_elem = spv_lower_elem_type(&c, st->arr_type, 0);
            uint32_t len = spv_lower_composite_len(&c, st->arr_type);
            uint32_t ptr_in = SPV_PTR_OF(SPV_SC_INPUT, arr_elem);
            uint32_t ptr_fn_e = SPV_PTR_OF(SPV_SC_FUNCTION, arr_elem);
            uint32_t ptr_chain = SPV_PTR_OF(SPV_SC_FUNCTION, st->elem_type);
            bool ok = ptr_in && ptr_fn_e && ptr_chain && arr_elem && len;

            if (st->leaf_mode)
            {
                const struct SpvDissolveMap *dm =
                    st->dmap ? &maps[st->dmap - 1] : NULL;

                /* leaf-by-leaf element copy: one constant-indexed load per
                 * leaf, stored into the function-local array.  Both sides
                 * descend with CONSTANT indices only, which SPIRV-Cross
                 * resolves into the flattened attribute leaves; the ORIGINAL
                 * dynamic index stays on the chain, re-rooted at the copy.
                 * A dissolved site reads the DISSOLVED per-(element,field)
                 * variables (base + [cp] + the leaf path below the field). */
                for (uint32_t e = 0; ok && e < len; e++)
                {
                    uint32_t ce = spv_lower_const_id(&c, idx_type, e, new_consts, n_new_consts);
                    if (!ce) { ok = false; break; }
                    for (uint32_t q = 0; q < st->leaf_n; q++)
                    {
                        const struct SpvLowerLeaf *lf = &leaf_tab[st->leaf_off + q];
                        uint32_t lt = lf->type;
                        uint32_t ptr_in_lf = SPV_PTR_OF(SPV_SC_INPUT, lt);
                        uint32_t ptr_fn_lf = SPV_PTR_OF(SPV_SC_FUNCTION, lt);
                        uint32_t aptr = next_id++, lval = next_id++, sptr = next_id++;
                        uint32_t lidx[6];
                        uint32_t srcvar = 0;

                        for (uint32_t p = 0; p < lf->depth; p++)
                        {
                            lidx[p] = spv_lower_const_id(&c, idx_type, lf->idx[p],
                                                         new_consts, n_new_consts);
                            if (!lidx[p]) { ok = false; break; }
                        }
                        if (!ok) break;
                        if (dm)
                        {
                            /* leaf-dissolved member: field = the leaf path's
                             * first index (depth 0: the single field of a
                             * plain array member); whole-member member: the
                             * element index stays an operand */
                            uint32_t f = lf->depth ? lf->idx[0] : 0;

                            if (dm->elems &&
                                (f >= dm->fields || e * dm->fields + f >= 96))
                            {
                                ok = false;
                                break;
                            }
                            srcvar = dm->elems ? dm->leaf[e * dm->fields + f] : dm->var0;
                        }

                        /* %aptr = AccessChain ptrIn_leaf <src base> ... */
                        LOW_PUSH(((dm ? (dm->elems ? (4u + (lf->depth ? lf->depth : 1u))
                                                   : (6u + lf->depth))
                                      : (st->k + 1u + st->prefix_n + lf->depth)) << 16) |
                                 SPV_OP_ACCESSCHAIN);
                        LOW_PUSH(ptr_in_lf);
                        LOW_PUSH(aptr);
                        if (st->root)
                        {
                            LOW_PUSH(st->root);
                            for (uint32_t p = 0; p < st->prefix_n; p++)
                                LOW_PUSH(st->prefix[p]);
                        }
                        else if (!dm)
                            LOW_PUSH(c.w[off + 3]);
                        if (dm)
                        {
                            /* the dissolved variable already carries the
                             * member (and, for leaf vars, the element) */
                            LOW_PUSH(srcvar);
                            LOW_PUSH(c.w[off + 4]);            /* control point */
                            if (dm->elems)
                            {
                                for (uint32_t p = 1; p < lf->depth; p++)
                                    LOW_PUSH(lidx[p]);
                            }
                            else
                            {
                                LOW_PUSH(ce);
                                for (uint32_t p = 0; p < lf->depth; p++)
                                    LOW_PUSH(lidx[p]);
                            }
                        }
                        else
                        {
                            for (uint32_t p = 4; p < st->k; p++)
                                LOW_PUSH(c.w[off + p]);
                            LOW_PUSH(ce);
                            for (uint32_t p = 0; p < lf->depth; p++)
                                LOW_PUSH(lidx[p]);
                        }

                        LOW_PUSH((4u << 16) | SPV_OP_LOAD);
                        LOW_PUSH(lt);
                        LOW_PUSH(lval);
                        LOW_PUSH(aptr);

                        /* %sptr = AccessChain ptrFn_leaf %fvar e leaf */
                        LOW_PUSH(((5u + lf->depth) << 16) | SPV_OP_ACCESSCHAIN);
                        LOW_PUSH(ptr_fn_lf);
                        LOW_PUSH(sptr);
                        LOW_PUSH(fvar[s]);
                        LOW_PUSH(ce);
                        for (uint32_t p = 0; p < lf->depth; p++)
                            LOW_PUSH(lidx[p]);
                        LOW_PUSH((3u << 16) | SPV_OP_STORE);
                        LOW_PUSH(sptr);
                        LOW_PUSH(lval);
                    }
                }
            }
            else for (uint32_t e = 0; ok && e < len; e++)
            {
                /* a constant with value e (module or fresh) */
                uint32_t ce = 0;
                for (uint32_t q = 0; q < c.ninst && !ce; q++)
                    if (c.op[q] == SPV_OP_CONSTANT && c.wc[q] >= 4 &&
                        c.w[c.off[q] + 1] == idx_type && (int32_t)c.w[c.off[q] + 3] == (int32_t)e)
                        ce = c.def[q];
                for (uint32_t q = 0; q < n_new_consts && !ce; q++)
                    if (new_consts[q].type == idx_type && new_consts[q].value == e)
                        ce = new_consts[q].id;
                if (!ce) { ok = false; break; }

                /* one fresh result id PER element: the same arrptr[s]/loaded[s]
                 * for every e defined id 317 len times ("Id 317 is defined more
                 * than once") whenever a module carried several such sites */
                arrptr[s] = next_id++;
                loaded[s] = next_id++;
                elemc[s] = next_id++;

                /* %eptr = AccessChain ptrIn_elem %base <prefix> %e
                 * words: hdr, rtype, result, base, [4, k) indices (+ split-chain
                 * inner-prefix indices before those), e */
                LOW_PUSH(((st->k + 1 + st->prefix_n) << 16) | SPV_OP_ACCESSCHAIN);
                LOW_PUSH(ptr_in);
                LOW_PUSH(arrptr[s]);
                if (st->root)
                {
                    LOW_PUSH(st->root);
                    for (uint32_t q = 0; q < st->prefix_n; q++)
                        LOW_PUSH(st->prefix[q]);
                }
                else
                    LOW_PUSH(c.w[off + 3]);
                for (uint32_t q = 4; q < st->k; q++)
                    LOW_PUSH(c.w[off + q]);
                LOW_PUSH(ce);

                /* %ev = Load %elem %eptr */
                LOW_PUSH((4u << 16) | SPV_OP_LOAD);
                LOW_PUSH(arr_elem);
                LOW_PUSH(loaded[s]);
                LOW_PUSH(arrptr[s]);

                /* %fp = AccessChain ptrFn_elem %fvar %e ; Store */
                LOW_PUSH((5u << 16) | SPV_OP_ACCESSCHAIN);
                LOW_PUSH(ptr_fn_e);
                LOW_PUSH(elemc[s]);
                LOW_PUSH(fvar[s]);
                LOW_PUSH(ce);
                LOW_PUSH((3u << 16) | SPV_OP_STORE);
                LOW_PUSH(elemc[s]);
                LOW_PUSH(loaded[s]);
            }
            if (ok)
            {
                /* the chain, re-rooted at the copy (same result id) */
                uint32_t rem = wc - st->k;
                LOW_PUSH(((4 + rem) << 16) | op);
                LOW_PUSH(ptr_chain);
                LOW_PUSH(c.w[off + 2]);
                LOW_PUSH(fvar[s]);
                for (uint32_t q = st->k; q < wc; q++)
                    LOW_PUSH(c.w[off + q]);
                continue;
            }
            /* fall through: emit the original chain unchanged */
        }

        for (uint32_t q = 0; q < wc; q++)
            LOW_PUSH(c.w[off + q]);
    }
    #undef LOW_PUSH
    #undef SPV_PTR_OF

    spv_lower_teardown(&c);
    *out_n = out_len;
    return out;
}

/* ===== legacy subgroup-vote opcodes -> SPIR-V 1.3 group-nonuniform =====
 *
 * OpSubgroupAllKHR / OpSubgroupAnyKHR / OpSubgroupAllEqualKHR
 * (SPV_KHR_shader_ballot) -> OpGroupNonUniformAll / Any / AllEqual with a
 * Subgroup-scope operand (see mgl_ext_tess.h).  Id-local, one word inserted
 * per rewritten instruction plus at most one OpTypeInt/OpConstant pair and
 * one OpCapability GroupNonUniformVote. */
#define SPV_OP_CAPABILITY       17
#define SPV_OP_TYPE_INT         21
#define SPV_CAP_GROUP_NONUNIFORM_VOTE 62
#define SPV_OP_SUBGROUP_ALL_KHR     4428
#define SPV_OP_SUBGROUP_ANY_KHR     4429
#define SPV_OP_SUBGROUP_ALL_EQ_KHR  4430
#define SPV_OP_GNU_ALL          334
#define SPV_OP_GNU_ANY          335
#define SPV_OP_GNU_ALL_EQUAL    336

uint32_t *mgl_spirv_lower_subgroup_vote(const uint32_t *words, size_t n, size_t *out_n)
{
    uint32_t votes = 0, type_u32 = 0, const3 = 0, have_cap = 0;
    size_t first_function = 0, i;

    *out_n = 0;
    if (!words || n < 5) return NULL;

    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_CAPABILITY)
        {
            if (wc >= 2 && words[i + 1] == SPV_CAP_GROUP_NONUNIFORM_VOTE)
                have_cap = 1;
        }
        else if (op == SPV_OP_TYPE_INT && wc == 4 && words[i + 2] == 32 && words[i + 3] == 0)
            type_u32 = words[i + 1];
        else if (op == SPV_OP_CONSTANT && wc == 4 && type_u32 && words[i + 1] == type_u32 &&
                 words[i + 3] == 3)
            const3 = words[i + 2];
        else if (op == SPV_OP_FUNCTION && !first_function)
            first_function = i;
        else if (op == SPV_OP_SUBGROUP_ALL_KHR || op == SPV_OP_SUBGROUP_ANY_KHR ||
                 op == SPV_OP_SUBGROUP_ALL_EQ_KHR)
            votes++;
        i += wc;
    }
    if (!votes) return NULL;

    uint32_t next_id = words[3];
    uint32_t new_type = 0, new_const = 0;
    if (!type_u32)
    {
        new_type = next_id++;
        if (!const3)
            new_const = next_id++;
    }
    else if (!const3)
        new_const = next_id++;

    size_t need = n + votes + (new_type ? 4 : 0) + (new_const ? 4 : 0) + (have_cap ? 0 : 2);
    uint32_t *out = (uint32_t *)malloc(need * sizeof(uint32_t));
    if (!out) return NULL;
    size_t o = 0;
    const3 = const3 ? const3 : new_const;

    /* header, with the raised bound */
    for (int q = 0; q < 5; q++)
        out[o++] = (q == 3) ? next_id : words[q];
    if (!have_cap)
    {
        out[o++] = (2u << 16) | SPV_OP_CAPABILITY;
        out[o++] = SPV_CAP_GROUP_NONUNIFORM_VOTE;
    }

    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff; if (wc == 0 || i + wc > n) { free(out); return NULL; }

        /* declarations for the scope constant: right before the functions */
        if (first_function && i == first_function)
        {
            if (new_type)
            {
                out[o++] = (4u << 16) | SPV_OP_TYPE_INT;
                out[o++] = new_type;
                out[o++] = 32;
                out[o++] = 0;
            }
            if (new_const)
            {
                out[o++] = (4u << 16) | SPV_OP_CONSTANT;
                out[o++] = new_type ? new_type : type_u32;
                out[o++] = new_const;
                out[o++] = 3;      /* ExecutionScopeSubgroup */
            }
        }

        if (op == SPV_OP_SUBGROUP_ALL_KHR || op == SPV_OP_SUBGROUP_ANY_KHR ||
            op == SPV_OP_SUBGROUP_ALL_EQ_KHR)
        {
            uint32_t core = (op == SPV_OP_SUBGROUP_ALL_KHR) ? SPV_OP_GNU_ALL :
                            (op == SPV_OP_SUBGROUP_ANY_KHR) ? SPV_OP_GNU_ANY :
                                                              SPV_OP_GNU_ALL_EQUAL;
            out[o++] = (5u << 16) | core;
            out[o++] = words[i + 1];   /* result type */
            out[o++] = words[i + 2];   /* result id */
            out[o++] = const3;         /* scope = Subgroup */
            out[o++] = words[i + 3];   /* predicate */
        }
        else
        {
            for (uint32_t q = 0; q < wc; q++)
                out[o++] = words[i + q];
        }
        i += wc;
    }

    *out_n = o;
    return out;
}

/* ===== BuiltIn DrawIndex (gl_DrawID) demoted to a Private variable =====
 *
 * SPIRV-Cross's MSL backend refuses BuiltInDrawIndex outright ("DrawIndex is
 * not supported in MSL") — Metal has no draw-id argument.  Demote the
 * variable to Private storage in the MSL-facing module; the stage MSL then
 * carries a plain `gl_DrawID` global whose value mgl_draw_params_rewrite_msl()
 * (tessellation.c) takes from MGL's draw-parameters buffer.  GL sees the
 * untouched module, so introspection still reports the built-in. */
#define SPV_OP_DECORATE         71
#define SPV_DEC_BUILTIN         11
#define SPV_BUILTIN_DRAW_INDEX  4426
#define SPV_SC_OUTPUT           3
#define SPV_SC_PRIVATE          6

uint32_t *mgl_spirv_lower_draw_index(const uint32_t *words, size_t n, size_t *out_n)
{
    uint32_t var = 0, hit = 0, old_ptr = 0, pointee = 0, new_ptr = 0, next_id;
    size_t i;

    *out_n = 0;
    if (!words || n < 5) return NULL;

    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_DECORATE && wc >= 4 && words[i + 2] == SPV_DEC_BUILTIN &&
            words[i + 3] == SPV_BUILTIN_DRAW_INDEX)
        {
            var = words[i + 1];
            hit = 1;
        }
        i += wc;
    }
    if (!hit) return NULL;

    next_id = words[3];
    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_VARIABLE && wc >= 4 && words[i + 2] == var)
        {
            old_ptr = words[i + 1];
            if (words[i + 3] != SPV_SC_INPUT) return NULL;            /* not an input builtin: leave it */
        }
        i += wc;
    }
    if (!old_ptr) return NULL;

    /* the pointee of the variable's pointer type, and any reusable Private
     * pointer of the same pointee (separate scans: the type is declared
     * before the variable) */
    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_TYPE_POINTER && wc == 4)
        {
            if (words[i + 1] == old_ptr)
                pointee = words[i + 3];
            else if (pointee && words[i + 2] == SPV_SC_PRIVATE && words[i + 3] == pointee)
                new_ptr = words[i + 1];
        }
        i += wc;
    }
    if (!pointee) return NULL;
    if (!new_ptr)
        new_ptr = next_id++;

    size_t need = n + 8 + (new_ptr >= words[3] ? 4 : 0);
    uint32_t *out = (uint32_t *)malloc(need * sizeof(uint32_t));
    if (!out) return NULL;
    size_t o = 0;

    for (int q = 0; q < 5; q++)
        out[o++] = (q == 3) ? next_id : words[q];

    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff; if (wc == 0 || i + wc > n) { free(out); return NULL; }

        if (op == SPV_OP_DECORATE && wc >= 4 && words[i + 1] == var &&
            words[i + 2] == SPV_DEC_BUILTIN && words[i + 3] == SPV_BUILTIN_DRAW_INDEX)
        {
            i += wc;                    /* drop the BuiltIn decoration */
            continue;
        }
        if (op == SPV_OP_VARIABLE && wc >= 4 && words[i + 2] == var)
        {
            if (new_ptr >= words[3])
            {
                /* the Private pointer type must precede its first use */
                out[o++] = (4u << 16) | SPV_OP_TYPE_POINTER;
                out[o++] = new_ptr;
                out[o++] = SPV_SC_PRIVATE;
                out[o++] = pointee;
            }
            out[o++] = words[i];
            out[o++] = new_ptr;
            out[o++] = var;
            out[o++] = SPV_SC_PRIVATE;
            for (uint32_t q = 4; q < wc; q++)
                out[o++] = words[i + q];
            i += wc;
            continue;
        }
        for (uint32_t q = 0; q < wc; q++)
            out[o++] = words[i + q];
        i += wc;
    }

    *out_n = o;
    return out;
}

/* the shared MSL preparation: every module handed to SPIRV-Cross's MSL
 * backend (program.c compile_stage, introspect_spirv.c msl_accepts) runs
 * through this so the acceptance gate and the real compile agree.  Returns a
 * fresh malloc'd module with the lowerings applied, or NULL when none
 * fired (the caller's module is unchanged). */
uint32_t *mgl_spirv_prepare_for_msl(const uint32_t *words, size_t n, size_t *n_out)
{
    uint32_t *mid = mgl_spirv_lower_subgroup_vote(words, n, n_out);
    const uint32_t *cur = mid ? mid : words;
    size_t cn = mid ? *n_out : n;

    /* arrays of per-patch interface blocks ("patch out Blk b[2]"): Vulkan
     * (and SPIRV-Cross's tessellation access chains) do not have them — the
     * Patch-decorated BLOCK variable's member-index map is broken and the
     * TCS compile dies with "Bad cast".  Dissolve into one plain patch ARRAY
     * variable per block member FIRST (plain patch variables are legal and
     * buffer/attribute-backed on both sides).  Fires only in tessellation
     * stages (the Patch decoration exists nowhere else). */
    {
        size_t pn = 0;
        uint32_t *pd = mgl_tess_dissolve_patch_block_arrays(cur, cn, &pn);

        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess prepare_for_msl: patch-block-array dissolve -> %s (%zu words)\n",
                    pd ? "ok" : "declined", pd ? pn : 0);
        if (pd)
        {
            if (mid)
                free(mid);
            mid = pd;
            cur = mid;
            cn = pn;
        }
    }

    uint32_t *out = mgl_spirv_lower_draw_index(cur, cn, n_out);
    if (!out)
        *n_out = cn;                    /* the pass clears n_out on a miss */
    else if (mid)
        free(mid);
    return out ? out : mid;
}

/* ------------------------------------------------------------------ dissolve
 * SPIRV-Cross flattens a per-vertex INPUT interface block of a tessellation
 * evaluation shader ("in TheBlock { S s; float fa[3]; S sa[2]; } vb[];") into
 * stage-in attributes, but its member-index resolution is only correct for
 * members that need no splitting: "vb[1].fa[2]" resolves through the ARRAY
 * member's un-split index and reads another member's attribute, and a
 * whole-member load ("S t = vb[0].s;") asserts in emit_tessellation_access_chain
 * (process abort).  Dissolve such a block into one plain per-vertex variable
 * per block member ("vb_s", "vb_fa", ...): SPIRV-Cross's plain-variable
 * splitting is correct, and the flattened leaf names it generates
 * ("vb_s_x", "vb_fa_0", "vb_sa_0_z_1", ...) are identical to the block's, so
 * MGL's record canonicalization matches them by name unchanged.  The block
 * variable itself stays in the module (name lookups keep working); it only
 * becomes unused.  Returns a fresh module, or NULL when nothing needed
 * dissolving or an unsupported pattern was met (caller keeps the original). */

/* kind of a TYPE id: SPV_T_* of the type instruction defining it */
static int spv_dis_kind(const uint32_t *words, size_t n, uint32_t type)
{
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op >= 19 && op <= 33 && words[k + 1] == type)
        {
            switch (op)
            {
                case SPV_OP_TYPE_ARRAY:  return SPV_T_ARRAY;
                case SPV_OP_TYPE_MATRIX: return SPV_T_MATRIX;
                case SPV_OP_TYPE_STRUCT: return SPV_T_STRUCT;
                case SPV_OP_TYPE_VECTOR: return SPV_T_VECTOR;
                default:                 return SPV_T_OTHER;
            }
        }
        k += wc;
    }
    return SPV_T_OTHER;
}

/* value of an int OpConstant id (-1 when not a 32-bit constant; callers that
 * store it into a uint32_t still see the 0xffffffffu sentinel) */
static int64_t spv_dis_const(const uint32_t *words, size_t n, uint32_t id)
{
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == SPV_OP_CONSTANT && wc == 4 && words[k + 2] == id)
            return words[k + 3];
        k += wc;
    }
    return -1;
}

/* the OpName string of an id ("" when unnamed) */
static void spv_dis_name(const uint32_t *words, size_t n, uint32_t id,
                         char *out, size_t cap)
{
    out[0] = 0;
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == 5 /* OpName */ && wc >= 3 && words[k + 1] == id)
        {
            const uint32_t *w = words + k + 2;
            const uint32_t *we = words + k + wc;
            size_t o = 0;

            for (; w < we && o + 4 < cap; w++)
                for (int b = 0; b < 4; b++)
                {
                    char ch = (char)((*w >> (8 * b)) & 255);

                    if (!ch) { w = we; break; }
                    out[o++] = ch;
                }
            out[o] = 0;
            return;
        }
        k += wc;
    }
}

/* how many stage-in LOCATIONS a type consumes (a struct/array counts its
 * leaves; vectors/matrices count their columns; doubles count double) */
static uint32_t spv_dis_leaf_count(const uint32_t *words, size_t n, uint32_t type)
{
    int kind = spv_dis_kind(words, n, type);

    if (kind == SPV_T_ARRAY)
    {
        uint32_t elem = 0, len_id = 0;

        for (size_t k = 5; k < n; )
        {
            uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

            if (wc == 0 || k + wc > n) break;
            if (op == SPV_OP_TYPE_ARRAY && wc >= 4 && words[k + 1] == type)
            {
                elem = words[k + 2];
                len_id = words[k + 3];
                break;
            }
            k += wc;
        }
        {
            uint32_t len = spv_dis_const(words, n, len_id);

            if (!elem || len == 0xffffffffu) return 1;
            return len * spv_dis_leaf_count(words, n, elem);
        }
    }
    if (kind == SPV_T_STRUCT)
    {
        uint32_t total = 0;

        for (size_t k = 5; k < n; )
        {
            uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

            if (wc == 0 || k + wc > n) break;
            if (op == SPV_OP_TYPE_STRUCT && wc >= 3 && words[k + 1] == type)
            {
                for (uint32_t q = 0; q + 2 < wc; q++)
                    total += spv_dis_leaf_count(words, n, words[k + 2 + q]);
                break;
            }
            k += wc;
        }
        return total ? total : 1;
    }
    if (kind == SPV_T_MATRIX)
    {
        uint32_t cols = 0;

        for (size_t k = 5; k < n; )
        {
            uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

            if (wc == 0 || k + wc > n) break;
            if (op == SPV_OP_TYPE_MATRIX && wc == 5 && words[k + 1] == type)
            {
                cols = words[k + 4];
                break;
            }
            k += wc;
        }
        return cols ? cols : 1;
    }
    /* scalar/vector: one location (fp64 counts two) */
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == 22 /* OpTypeFloat */ && wc >= 4 && words[k + 1] == type)
        {
            uint32_t bits = words[k + 3];

            return bits == 64 ? 2 : 1;
        }
        k += wc;
    }
    return 1;
}

/* OpMemberName of a struct field; false when unnamed */
static bool spv_dis_field_name(const uint32_t *words, size_t n, uint32_t stype,
                               uint32_t f, char *name, size_t name_sz)
{
    size_t k;

    for (k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == 6 /* OpMemberName */ && wc >= 4 && words[k + 1] == stype &&
            words[k + 2] == f)
        {
            size_t len = (wc - 3) * 4;

            snprintf(name, name_sz, "%.*s", (int)len, (const char *)(words + k + 3));
            for (size_t q = strlen(name) + 1; q < len; q++)
                if (((const char *)(words + k + 3))[q])
                    name[0] = 0;      /* embedded NUL: a packed multi-name word */
            return name[0] != 0;
        }
        k += wc;
    }
    return false;
}

/* dissolved per-vertex block member variables */
struct DissolveJob
{
    uint32_t var;               /* the block interface variable */
    uint32_t block_type;        /* the Block-decorated struct */
    uint32_t arr_len_id;        /* constant id of the control-point array length */
    uint32_t n_members;
    uint32_t member_type[64];
    char member_name[64][64];
    int32_t member_loc[64];     /* OpMemberDecorate Location, -1 when absent */
    uint32_t base_loc;          /* derived Location for the dissolved members */
    uint32_t member_slot[64];   /* first location slot of each member */
    uint32_t new_var[64];
    /* struct-ARRAY members dissolve to one variable per (element, field) leaf
     * instead: SPIRV-Cross mis-flattens per-vertex inputs whose element type
     * is a struct (wrong attribute for [cp][i].field chains) */
    uint32_t leaf_var[64][96];
    uint32_t leaf_elems[64];    /* element count, 0 = plain single-variable member */
    uint32_t leaf_fields[64];   /* field count of the element struct */
};

uint32_t *mgl_tess_dissolve_vertex_input_blocks(const uint32_t *words, size_t n, size_t *out_n,
                                                struct SpvDissolveMap *maps, uint32_t max_maps,
                                                uint32_t *n_maps_out, int *partial_out)
{
    struct DissolveJob jobs[8];
    int n_jobs = 0;
    uint32_t next_id;
    uint32_t *work = NULL;
    size_t i;
    uint32_t n_maps_local = 0;
    uint32_t names_out[8192], deco_out[4096], type_out[4096], var_out[8 * 64 * 5];
    size_t n_names = 0, n_deco = 0, n_type = 0, n_var = 0;

    *out_n = 0;
    if (n_maps_out)
        *n_maps_out = 0;
    if (partial_out)
        *partial_out = 0;
    if (!words || n < 8)
        return NULL;
    next_id = words[3];

    /* 1) block struct types (OpDecorate <type> Block) */
    {
        uint32_t blocks[16];
        int n_blocks = 0;

        for (i = 5; i < n; )
        {
            uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

            if (wc == 0 || i + wc > n) break;
            if (op == SPV_OP_DECORATE && wc >= 3 && words[i + 2] == 2 /* Block */ &&
                n_blocks < 16)
                blocks[n_blocks++] = words[i + 1];
            i += wc;
        }
        if (!n_blocks)
            return NULL;

        /* 2) INPUT variables of pointer-to-array<block> */
        for (i = 5; i < n && n_jobs < 8; )
        {
            uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

            if (wc == 0 || i + wc > n) break;
            if (op == SPV_OP_VARIABLE && wc >= 4 && words[i + 3] == SPV_SC_INPUT)
            {
                uint32_t ptr = words[i + 1], vid = words[i + 2];
                uint32_t pointee = 0, elem = 0, len_id = 0, block = 0;
                int is_patch = 0;

                /* per-PATCH arrays ('patch in Blk b[2]') are NOT per-vertex
                 * blocks: leave them alone (the dynamic-IO lower handles a
                 * dynamic patch-array index).  glslang decorates per-patch
                 * interface BLOCKS on the members (OpMemberDecorate <type> m
                 * Patch), not on the variable, so the variable-level scan
                 * alone misses 'patch in TheBlock {...} b[2]' and the block
                 * would dissolve as PER-VERTEX — its member variables would
                 * collide with the per-patch ones the patch-block-array
                 * dissolve creates later (tess_spv_lower.c) */
                for (size_t k = 5; k < n && !is_patch; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == 71 /*OpDecorate*/ && wc2 >= 3 &&
                        words[k + 1] == vid && words[k + 2] == SPV_DEC_PATCH)
                        is_patch = 1;
                    k += wc2;
                }

                for (size_t k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 >= 19 && op2 <= 33 && words[k + 1] == ptr)
                    {
                        pointee = words[k + 3];
                        break;
                    }
                    k += wc2;
                }
                if (pointee)
                    for (size_t k = 5; k < n; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 >= 19 && op2 <= 33 && words[k + 1] == pointee)
                        {
                            if (op2 == SPV_OP_TYPE_ARRAY && wc2 >= 4)
                            {
                                elem = words[k + 2];
                                len_id = words[k + 3];
                            }
                            break;
                        }
                        k += wc2;
                    }
                for (int b = 0; b < n_blocks; b++)
                    if (elem && elem == blocks[b]) { block = elem; break; }
                /* member-level Patch on the block type (glslang's form):
                 * a per-patch block array — skip it here */
                if (!is_patch && block && spv_patch_block_type(words, n, block))
                    is_patch = 1;
                if (is_patch)
                {
                    i += wc;
                    continue;
                }
                if (block)
                {
                    struct DissolveJob *job = &jobs[n_jobs];

                    memset(job, 0, sizeof *job);
                    bool skip = false;
                    uint32_t var_loc = 0;
                    bool have_var_loc = false;

                    memset(job, 0, sizeof *job);
                    job->var = vid;
                    job->block_type = block;
                    job->arr_len_id = len_id;

                    /* the block VARIABLE's Location (glslang decorates the
                     * variable, not the members): keep the dissolved members
                     * in declaration order after it */
                    for (size_t k = 5; k < n; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 == SPV_OP_DECORATE && wc2 >= 4 && words[k + 1] == vid &&
                            words[k + 2] == 30 /* Location */)
                        {
                            var_loc = words[k + 3];
                            have_var_loc = true;
                            break;
                        }
                        k += wc2;
                    }

                    for (size_t k = 5; k < n; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 == SPV_OP_TYPE_STRUCT && words[k + 1] == block && wc2 >= 2)
                        {
                            job->n_members = wc2 - 2;
                            for (uint32_t q = 0; q < job->n_members && q < 64; q++)
                                job->member_type[q] = words[k + 2 + q];
                            break;
                        }
                        k += wc2;
                    }
                    if (!job->n_members)
                        skip = true;
                    if (job->n_members > 64)
                        job->n_members = 64;
                    for (size_t k = 5; k < n && !skip; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 == 72 /* OpMemberDecorate */ && wc2 >= 4 &&
                            words[k + 1] == block && words[k + 3] == 11 /* BuiltIn */)
                        {
                            /* a built-in interface (gl_PerVertex): SPIRV-Cross's
                             * built-in handling must stay in charge */
                            skip = true;
                        }
                        k += wc2;
                    }
                    for (uint32_t q = 0; q < 64; q++) job->member_loc[q] = -1;
                    job->base_loc = have_var_loc ? var_loc : 0u;
                    {
                        uint32_t slot = 0;

                        for (uint32_t q = 0; q < job->n_members; q++)
                        {
                            job->member_slot[q] = slot;
                            slot += spv_dis_leaf_count(words, n, job->member_type[q]);
                        }
                    }

                    for (size_t k = 5; k < n; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 == 72 /* OpMemberDecorate */ && wc2 >= 4 &&
                            words[k + 1] == block)
                        {
                            uint32_t mi = words[k + 2];

                            if (mi < job->n_members)
                            {
                                if (words[k + 3] == 30 /* Location */ && wc2 >= 5)
                                    job->member_loc[mi] = (int32_t)words[k + 4];
                                else if (words[k + 3] == 0 /* Flat */ ||
                                         words[k + 3] == 16 /* NoPerspective */ ||
                                         words[k + 3] == 3 /* Centroid */)
                                    job->member_loc[mi] |= 0;  /* kept implicit */
                            }
                        }
                        if (op2 == 6 /* OpMemberName */ && wc2 >= 4 && words[k + 1] == block)
                        {
                            uint32_t mi = words[k + 2];

                            if (mi < job->n_members)
                            {
                                const uint32_t *w = words + k + 3;
                                const uint32_t *we = words + k + wc2;
                                size_t o = 0;

                                while (w < we && o + 4 < sizeof job->member_name[0])
                                {
                                    for (int b = 0; b < 4; b++)
                                    {
                                        char ch = (char)((*w >> (8 * b)) & 255);

                                        if (!ch) { w = we; break; }
                                        job->member_name[mi][o++] = ch;
                                    }
                                    w++;
                                }
                                job->member_name[mi][o] = 0;
                            }
                        }
                        k += wc2;
                    }
                    for (uint32_t q = 0; q < job->n_members; q++)
                        if (!job->member_name[q][0])
                            snprintf(job->member_name[q], sizeof job->member_name[q],
                                     "m%u", q);
                    if (!skip)
                        n_jobs++;
                }
            }
            i += wc;
        }
        if (!n_jobs)
            return NULL;
    }

    /* 3) which jobs actually need it: an access chain rooted at the block var
     *    descends into a member.  A chain that stops at the control-point
     *    index (whole-element load) cannot be dissolved — drop the job.  A
     *    dropped job whose whole access loads a STRUCT-typed aggregate is
     *    UNSAFE: SPIRV-Cross aborts (emit_tessellation_access_chain) on such
     *    chains when compiling the original block, so the whole module must
     *    be declined instead (see the tail of this function). */
    bool unsafe = false;
    for (int j = 0; j < n_jobs; j++)
    {
        bool whole = false, whole_struct = false, descend = false;

        for (i = 5; i < n; )
        {
            uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

            if (wc == 0 || i + wc > n) break;
            if ((op == SPV_OP_ACCESSCHAIN || op == SPV_OP_INBOUNDSAC) && wc >= 5 &&
                words[i + 3] == jobs[j].var)
            {
                if (wc >= 7)
                    descend = true;
                else if (wc == 6)
                {
                    /* a chain that stops AT a member: dissolvable when the
                     * member flattens to attributes — scalar/vector/matrix/
                     * array all do; only a STRUCT (or an array of structs)
                     * is a whole-element load SPIRV-Cross aborts on */
                    uint32_t mval = spv_dis_const(words, n, words[i + 5]);
                    int kind = SPV_T_OTHER;

                    if (mval != 0xffffffffu && mval < jobs[j].n_members)
                        kind = spv_dis_kind(words, n, jobs[j].member_type[mval]);
                    if (kind == SPV_T_STRUCT)
                    {
                        /* an ARRAY of structs is equally fatal (SPIRV-Cross
                         * mis-flattens it), anything else is flattenable */
                        whole = true;
                        whole_struct = true;
                    }
                    else
                        descend = true;
                }
                else
                {
                    whole = true;
                    whole_struct = true;    /* a whole block element is a struct load */
                }
            }
            i += wc;
        }
        if (whole || !descend)
        {
            if (whole_struct)
                unsafe = true;
            jobs[j] = jobs[n_jobs - 1];
            n_jobs--;
            j--;
        }
    }
    if (unsafe)
    {
        /* SPIRV-Cross cannot compile this module without aborting (whole
         * struct-typed interface loads); hand back a copy whose magic is
         * corrupted so every SPIRV-Cross compile fails CLEANLY (parser
         * error) and MGL reports a shader/link error instead of crashing.
         * Checked before the no-jobs exit: a dropped job is what set it. */
        uint32_t *bad = malloc(n * 4);

        if (!bad)
            return NULL;
        memcpy(bad, words, n * 4);
        bad[0] ^= 1;
        *out_n = n;
        return bad;
    }
    if (!n_jobs)
        return NULL;

    /* 4) plan the new instructions (names, decorations, types, variables) */
    for (int j = 0; j < n_jobs; j++)
    {
        struct DissolveJob *job = &jobs[j];
        char iname[64];

        spv_dis_name(words, n, job->var, iname, sizeof iname);
        if (!iname[0])
            snprintf(iname, sizeof iname, "blk%u", job->var);

        for (uint32_t m = 0; m < job->n_members; m++)
        {
            uint32_t arr = 0, ptr = 0;

            /* a member that is an ARRAY OF STRUCTS dissolves to one variable
             * per (element, field) leaf: SPIRV-Cross's per-vertex input
             * flattening mis-binds [cp][i].field chains for array<struct>
             * elements (it picks the field by the wrong index). Leaf names
             * "<inst>_mgl_ds_<member>_<i>_<field>" keep the same strings the
             * split names had, so the record keying is unchanged. */
            if (spv_dis_kind(words, n, job->member_type[m]) == SPV_T_ARRAY)
            {
                uint32_t elem = 0, len_id = 0, ecount = 0, efields = 0;
                size_t k;

                for (k = 5; k < n; )
                {
                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                    if (wc2 == 0 || k + wc2 > n) break;
                    if (op2 == SPV_OP_TYPE_ARRAY && wc2 == 4 && words[k + 1] == job->member_type[m])
                    {
                        elem = words[k + 2];
                        len_id = words[k + 3];
                        break;
                    }
                    k += wc2;
                }
                if (elem && spv_dis_kind(words, n, elem) == SPV_T_STRUCT)
                {
                    uint32_t lv = spv_dis_const(words, n, len_id);

                    ecount = (lv != 0xffffffffu) ? lv : 0;
                    for (k = 5; k < n; )
                    {
                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                        if (wc2 == 0 || k + wc2 > n) break;
                        if (op2 == SPV_OP_TYPE_STRUCT && wc2 >= 3 && words[k + 1] == elem)
                        {
                            efields = wc2 - 2;
                            break;
                        }
                        k += wc2;
                    }
                }
                if (ecount && efields && ecount * efields <= 96)
                {
                    bool ok = true;
                    uint32_t slot = job->member_loc[m] >= 0 ?
                                    (uint32_t)job->member_loc[m] :
                                    job->base_loc + job->member_slot[m];

                    /* validate every field first (scalar/vector/matrix or an
                     * array of them, NAMED): a bad field falls back to the
                     * whole-member variable for this member only */
                    for (uint32_t f = 0; f < efields && ok; f++)
                    {
                        char fname[64];
                        uint32_t ftype = 0;
                        int fkind;
                        size_t k2;

                        for (k2 = 5; k2 < n; )
                        {
                            uint32_t wc3 = words[k2] >> 16, op3 = words[k2] & 0xffff;

                            if (wc3 == 0 || k2 + wc3 > n) break;
                            if (op3 == SPV_OP_TYPE_STRUCT && wc3 >= 3 && words[k2 + 1] == elem)
                            {
                                ftype = words[k2 + 2 + f];
                                break;
                            }
                            k2 += wc3;
                        }
                        fkind = ftype ? spv_dis_kind(words, n, ftype) : SPV_T_STRUCT;
                        if (fkind == SPV_T_STRUCT ||
                            !spv_dis_field_name(words, n, elem, f, fname, sizeof fname))
                            ok = false;
                    }
                    if (!ok)
                        goto leaf_fallback;

                    for (uint32_t f = 0; f < efields; f++)
                    {
                        char fname[64];
                        uint32_t ftype = 0;
                        size_t k2;

                        for (k2 = 5; k2 < n; )
                        {
                            uint32_t wc3 = words[k2] >> 16, op3 = words[k2] & 0xffff;

                            if (wc3 == 0 || k2 + wc3 > n) break;
                            if (op3 == SPV_OP_TYPE_STRUCT && wc3 >= 3 && words[k2 + 1] == elem)
                            {
                                ftype = words[k2 + 2 + f];
                                break;
                            }
                            k2 += wc3;
                        }
                        if (!ftype || spv_dis_kind(words, n, ftype) == SPV_T_STRUCT ||
                            !spv_dis_field_name(words, n, elem, f, fname, sizeof fname))
                            ok = false;
                        if (ok)
                        {
                            /* an ARRAY field splits into several attributes,
                             * so a leaf's slot must advance by its leaf count */
                            uint32_t stride = spv_dis_leaf_count(words, n, ftype);

                            for (uint32_t e = 0; e < ecount; e++)
                            {
                                uint32_t larr = 0, lptr = 0;

                                for (size_t k3 = 5; k3 < n; )
                                {
                                    uint32_t wc4 = words[k3] >> 16, op4 = words[k3] & 0xffff;

                                    if (wc4 == 0 || k3 + wc4 > n) break;
                                    if (op4 == SPV_OP_TYPE_ARRAY && wc4 == 4 &&
                                        words[k3 + 2] == ftype && words[k3 + 3] == job->arr_len_id)
                                        larr = words[k3 + 1];
                                    if (op4 == SPV_OP_TYPE_POINTER && wc4 == 4 &&
                                        words[k3 + 2] == SPV_SC_INPUT && words[k3 + 3] == larr)
                                        lptr = words[k3 + 1];
                                    k3 += wc4;
                                }
                                if (!larr)
                                {
                                    larr = next_id++;
                                    type_out[n_type++] = (4u << 16) | SPV_OP_TYPE_ARRAY;
                                    type_out[n_type++] = larr;
                                    type_out[n_type++] = ftype;
                                    type_out[n_type++] = job->arr_len_id;
                                }
                                if (!lptr)
                                {
                                    lptr = next_id++;
                                    type_out[n_type++] = (4u << 16) | SPV_OP_TYPE_POINTER;
                                    type_out[n_type++] = lptr;
                                    type_out[n_type++] = SPV_SC_INPUT;
                                    type_out[n_type++] = larr;
                                }
                                job->leaf_var[m][e * efields + f] = next_id++;
                                var_out[n_var++] = (4u << 16) | SPV_OP_VARIABLE;
                                var_out[n_var++] = lptr;
                                var_out[n_var++] = job->leaf_var[m][e * efields + f];
                                var_out[n_var++] = SPV_SC_INPUT;
                                deco_out[n_deco++] = (4u << 16) | SPV_OP_DECORATE;
                                deco_out[n_deco++] = job->leaf_var[m][e * efields + f];
                                deco_out[n_deco++] = 30 /* Location */;
                                deco_out[n_deco++] = slot;
                                slot += stride;
                                {
                                    char full[160];
                                    uint32_t buf[40];
                                    size_t words_n = 0;

                                    snprintf(full, sizeof full, "%s_mgl_ds_%s_%u_%s",
                                             iname, job->member_name[m], e, fname);
                                    {
                                        const char *c = full;
                                        size_t rem = strlen(full) + 1;

                                        while (rem > 0 && words_n < 40)
                                        {
                                            uint32_t w = 0;

                                            for (int b = 0; b < 4 && rem > 0; b++)
                                            {
                                                w |= (uint32_t)(unsigned char)*c << (8 * b);
                                                c++;
                                                rem--;
                                            }
                                            buf[words_n++] = w;
                                        }
                                    }
                                    names_out[n_names++] = (uint32_t)((words_n + 2) << 16) | 5 /* OpName */;
                                    names_out[n_names++] = job->leaf_var[m][e * efields + f];
                                    for (size_t q = 0; q < words_n; q++) names_out[n_names++] = buf[q];
                                }
                            }
                        }
                    }
                    if (ok)
                    {
                        job->leaf_elems[m] = ecount;
                        job->leaf_fields[m] = efields;
                        continue;   /* no whole-member variable for this member */
                    }
leaf_fallback:
                    /* a bad leaf (nested struct field / unnamed member):
                     * this member keeps its whole-member variable; its chains
                     * keep the original indices (SPIRV-Cross flattens them) */
                    ;
                }
            }

            for (size_t k = 5; k < n; )
            {
                uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                if (wc2 == 0 || k + wc2 > n) break;
                if (op2 == SPV_OP_TYPE_ARRAY && wc2 == 4 &&
                    words[k + 2] == job->member_type[m] &&
                    words[k + 3] == job->arr_len_id)
                    arr = words[k + 1];
                if (op2 == SPV_OP_TYPE_POINTER && wc2 == 4 &&
                    words[k + 2] == SPV_SC_INPUT && words[k + 3] == arr)
                    ptr = words[k + 1];
                k += wc2;
            }
            if (!arr)
            {
                arr = next_id++;
                type_out[n_type++] = (4u << 16) | SPV_OP_TYPE_ARRAY;
                type_out[n_type++] = arr;
                type_out[n_type++] = job->member_type[m];
                type_out[n_type++] = job->arr_len_id;
            }
            if (!ptr)
            {
                ptr = next_id++;
                type_out[n_type++] = (4u << 16) | SPV_OP_TYPE_POINTER;
                type_out[n_type++] = ptr;
                type_out[n_type++] = SPV_SC_INPUT;
                type_out[n_type++] = arr;
            }
            job->new_var[m] = next_id++;
            var_out[n_var++] = (4u << 16) | SPV_OP_VARIABLE;
            var_out[n_var++] = ptr;
            var_out[n_var++] = job->new_var[m];
            var_out[n_var++] = SPV_SC_INPUT;
            if (job->member_loc[m] >= 0)
            {
                deco_out[n_deco++] = (4u << 16) | SPV_OP_DECORATE;
                deco_out[n_deco++] = job->new_var[m];
                deco_out[n_deco++] = 30 /* Location */;
                deco_out[n_deco++] = (uint32_t)job->member_loc[m];
            }
            else
            {
                /* SPIRV-Cross needs a Location to give the stage-in member an
                 * [[attribute(N)]]; any distinct value works — MGL binds by
                 * the emitted attribute numbers.  Space members by their leaf
                 * count so no two leaves land on the same slot. */
                deco_out[n_deco++] = (4u << 16) | SPV_OP_DECORATE;
                deco_out[n_deco++] = job->new_var[m];
                deco_out[n_deco++] = 30 /* Location */;
                deco_out[n_deco++] = job->base_loc + job->member_slot[m];
            }
            /* the name carries the block instance AND member so the record
             * canonicalization can strip the prefix and re-key by the block
             * member path (tess_annotate_keys); "mgl_ds" marks the variable
             * as dissolved there */
            {
                char full[160];
                uint32_t buf[40];
                size_t words_n;

                snprintf(full, sizeof full, "%s_mgl_ds_%s", iname, job->member_name[m]);
                words_n = (strlen(full) + 4) / 4;
                if (words_n <= 40)
                {
                    memset(buf, 0, sizeof buf);
                    memcpy(buf, full, strlen(full));
                    names_out[n_names++] = (uint32_t)((2 + words_n) << 16) | 5u;
                    names_out[n_names++] = job->new_var[m];
                    for (size_t q = 0; q < words_n; q++)
                        names_out[n_names++] = buf[q];
                }
            }
        }
    }

    /* 5) plan the chain rewrites (applied while emitting: the member index
     *    is dropped, so each rewritten instruction is ONE word shorter) */
    {
        /* export the member -> dissolved-variable map for the dynamic-IO
         * lower (leaf members: one variable per element/field) */
        for (int j = 0; j < n_jobs && maps && n_maps_local < max_maps; j++)
            for (int m = 0; m < (int)jobs[j].n_members &&
                            n_maps_local < max_maps; m++)
            {
                struct SpvDissolveMap *e = &maps[n_maps_local];

                e->var = jobs[j].var;
                e->member = (uint32_t)m;
                e->elems = jobs[j].leaf_elems[m];
                e->fields = jobs[j].leaf_fields[m];
                e->var0 = jobs[j].new_var[m];
                if (e->elems && e->fields && e->elems * e->fields <= 96)
                    for (uint32_t q = 0; q < e->elems * e->fields; q++)
                        e->leaf[q] = jobs[j].leaf_var[m][q];
                n_maps_local++;
            }
        if (n_maps_out)
            *n_maps_out = n_maps_local;

        uint32_t rw_off[512], rw_type[512], rw_base[512];
        uint8_t rw_drop2[512];      /* leaf rewrite: drop elem+field indices too */
        int n_rw = 0;

        for (i = 5; i < n && n_rw < 512; )
        {
            uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

            if (wc == 0 || i + wc > n) break;
            if ((op == SPV_OP_ACCESSCHAIN || op == SPV_OP_INBOUNDSAC) && wc >= 6)
            {
                for (int j = 0; j < n_jobs; j++)
                    if (words[i + 3] == jobs[j].var)
                    {
                        uint32_t mval = spv_dis_const(words, n, words[i + 5]);
                        int found_m = -1;

#define DISSOLVE_PARTIAL() do { if (partial_out) *partial_out = 1; goto next_chain; } while (0)

                        if (mval != 0xffffffffu && mval < jobs[j].n_members)
                            found_m = (int)mval;
                        if (found_m < 0)
                            DISSOLVE_PARTIAL();  /* dynamic member index: leave for the lower */
                        if (wc == 6)
                        {
                            /* stops AT the member: fine for everything that
                             * flattens to attributes; only a struct element
                             * (or array of structs, handled as leaves) can't */
                            int kind = spv_dis_kind(words, n, jobs[j].member_type[found_m]);

                            if (kind == SPV_T_STRUCT)
                                goto bail;
                        }

                        {
                            uint32_t pointee = jobs[j].member_type[found_m];
                            uint32_t lv_elem = 0xffffffffu, lv_field = 0xffffffffu;
                            bool leaf = jobs[j].leaf_elems[found_m] != 0;
                            uint32_t q = 6;

                            if (leaf)
                            {
                                /* the member path is [elem, field, ...]: fold
                                 * the two constants into the leaf variable */
                                uint32_t e = spv_dis_const(words, n, words[i + 6]);
                                uint32_t f = spv_dis_const(words, n, words[i + 7]);

                                if (wc < 8 || spv_dis_kind(words, n, pointee) != SPV_T_ARRAY)
                                    goto unsafe_exit;      /* whole struct element: SPIRV-Cross aborts */
                                if (e >= jobs[j].leaf_elems[found_m] ||
                                    f >= jobs[j].leaf_fields[found_m])
                                    DISSOLVE_PARTIAL();       /* dynamic index: the lower copies it */
                                /* first interior step must land on the struct */
                                {
                                    uint32_t elem_t = 0;

                                    for (size_t k = 5; k < n; )
                                    {
                                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                                        if (wc2 == 0 || k + wc2 > n) break;
                                        if (op2 == SPV_OP_TYPE_ARRAY && wc2 >= 4 &&
                                            words[k + 1] == pointee)
                                        {
                                            elem_t = words[k + 2];
                                            break;
                                        }
                                        k += wc2;
                                    }
                                    if (!elem_t || spv_dis_kind(words, n, elem_t) != SPV_T_STRUCT)
                                        DISSOLVE_PARTIAL();       /* unusable interior: leave for the lower */
                                }
                                lv_elem = e;
                                lv_field = f;
                                pointee = 0;        /* set after resolving the field below */
                                q = 8;              /* walk the deeper path only */
                                for (uint32_t d = 6; d <= 7; d++)
                                {
                                    uint32_t cval = spv_dis_const(words, n, words[i + d]);

                                    if (cval == 0xffffffffu) DISSOLVE_PARTIAL();   /* e/f const missing */
                                    for (size_t k = 5; k < n; )
                                    {
                                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                                        if (wc2 == 0 || k + wc2 > n) break;
                                        if (op2 == SPV_OP_TYPE_ARRAY && wc2 >= 4 &&
                                            words[k + 1] == jobs[j].member_type[found_m])
                                        {
                                            uint32_t st = words[k + 2];

                                            for (size_t k2 = 5; k2 < n; )
                                            {
                                                uint32_t wc3 = words[k2] >> 16, op3 = words[k2] & 0xffff;

                                                if (wc3 == 0 || k2 + wc3 > n) break;
                                                if (op3 == SPV_OP_TYPE_STRUCT && wc3 >= 3 &&
                                                    words[k2 + 1] == st)
                                                {
                                                    pointee = words[k2 + 2 + (d == 6 ? 0 : cval)];
                                                    /* d==6 resolves the ELEMENT (array
                                                     * index), d==7 the FIELD of it */
                                                    if (d == 6) pointee = st;
                                                    break;
                                                }
                                                k2 += wc3;
                                            }
                                            break;
                                        }
                                        k += wc2;
                                    }
                                }
                            }

                            for (; q < wc; q++)
                            {
                                int kind = spv_dis_kind(words, n, pointee);
                                uint32_t idx_id = words[i + q], nxt = 0;

                                if (kind == SPV_T_ARRAY)
                                {
                                    for (size_t k = 5; k < n; )
                                    {
                                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                                        if (wc2 == 0 || k + wc2 > n) break;
                                        if (op2 == SPV_OP_TYPE_ARRAY && wc2 >= 4 &&
                                            words[k + 1] == pointee)
                                        {
                                            nxt = words[k + 2];
                                            break;
                                        }
                                        k += wc2;
                                    }
                                    if (!nxt) DISSOLVE_PARTIAL();   /* leave for the lower */
                                }
                                else if (kind == SPV_T_STRUCT)
                                {
                                    uint32_t cval = spv_dis_const(words, n, idx_id);

                                    if (cval == 0xffffffffu) DISSOLVE_PARTIAL();
                                    for (size_t k = 5; k < n; )
                                    {
                                        uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                                        if (wc2 == 0 || k + wc2 > n) break;
                                        if (op2 == SPV_OP_TYPE_STRUCT && wc2 >= 3 &&
                                            words[k + 1] == pointee)
                                        {
                                            if (cval + 2 >= wc2) goto bail;
                                            nxt = words[k + 2 + cval];
                                            break;
                                        }
                                        k += wc2;
                                    }
                                    if (!nxt) goto bail;
                                }
                                pointee = nxt;
                            }
                            /* find-or-create an Input pointer for the new pointee */
                            {
                                uint32_t ptr = 0;

                                for (size_t k = 5; k < n; )
                                {
                                    uint32_t wc2 = words[k] >> 16, op2 = words[k] & 0xffff;

                                    if (wc2 == 0 || k + wc2 > n) break;
                                    if (op2 == SPV_OP_TYPE_POINTER && wc2 == 4 &&
                                        words[k + 2] == SPV_SC_INPUT &&
                                        words[k + 3] == pointee)
                                    {
                                        ptr = words[k + 1];
                                        break;
                                    }
                                    k += wc2;
                                }
                                if (!ptr)
                                {
                                    if (n_type + 4 > sizeof type_out / sizeof type_out[0])
                                        goto bail;
                                    ptr = next_id++;
                                    type_out[n_type++] = (4u << 16) | SPV_OP_TYPE_POINTER;
                                    type_out[n_type++] = ptr;
                                    type_out[n_type++] = SPV_SC_INPUT;
                                    type_out[n_type++] = pointee;
                                }
                                rw_off[n_rw] = (uint32_t)i;
                                rw_type[n_rw] = ptr;
                                rw_base[n_rw] = leaf ?
                                    jobs[j].leaf_var[found_m][lv_elem * jobs[j].leaf_fields[found_m] + lv_field] :
                                    jobs[j].new_var[found_m];
                                rw_drop2[n_rw] = leaf ? 1 : 0;
                                n_rw++;
                            }
                        }
                        break;
                    }
            }
next_chain: ;
            i += wc;
        }

        /* 6) emit the new module: names / decorations / types / variables
         *    inserted at the end of the debug-annotation block, the chain
         *    rewrites applied inline, the id bound bumped */
        {
            size_t insert_at = 0, insert_names = 0, first_func = 0;
            uint32_t *out;
            size_t o = 0, need;

            for (i = 5; i < n; )
            {
                uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;

                if (wc == 0 || i + wc > n) break;
                if (op == SPV_OP_FUNCTION)
                {
                    first_func = i;
                    break;
                }
                if (op >= 2 && op <= 8)            /* the debug section (names) */
                    insert_names = i + wc;
                if (op == SPV_OP_DECORATE || op == 72 /* OpMemberDecorate */ ||
                    op == 73 /* OpGroupDecorate */)
                    insert_at = i + wc;            /* the annotation section */
                i += wc;
            }
            if (!first_func)
                goto bail;
            if (getenv("MGL_TESS_DEBUG3"))
                fprintf(stderr, "dissolve: names@%zu annot=%zu first_func=%zu names=%zu deco=%zu type=%zu var=%zu\n",
                        insert_names, insert_at, first_func, n_names, n_deco, n_type, n_var);

            need = n + n_names + n_deco + n_type + n_var + 8;
            out = (uint32_t *)malloc(need * sizeof(uint32_t));
            if (!out)
                goto bail;

            for (i = 0; i < n; )
            {
                if (i == 0)
                {
                    /* the fixed 5-word header: magic, version, generator,
                     * id bound, schema */
                    for (uint32_t q = 0; q < 5; q++)
                        out[o++] = (q == 3) ? (next_id > words[3] ? next_id : words[3])
                                            : words[q];
                    i = 5;
                    continue;
                }
                if (i == insert_names)
                {
                    for (size_t q = 0; q < n_names; q++) out[o++] = names_out[q];
                }
                if (i == insert_at)
                {
                    for (size_t q = 0; q < n_deco; q++) out[o++] = deco_out[q];
                }
                if (i == first_func)
                {
                    /* all existing types, constants and globals precede the
                     * first function, so the new types and variables can
                     * reference anything defined above */
                    for (size_t q = 0; q < n_type; q++) out[o++] = type_out[q];
                    for (size_t q = 0; q < n_var; q++) out[o++] = var_out[q];
                }
                {
                    uint32_t wc = words[i] >> 16;
                    int rw = -1;

                    if (!wc) break;
                    for (int q = 0; q < n_rw; q++)
                        if (rw_off[q] == i) { rw = q; break; }
                    if (rw >= 0)
                    {
                        /* AccessChain minus the member index */
                        uint32_t op = words[i] & 0xffff;
                        uint32_t drop = rw_drop2[rw] ? 3 : 1;

                        out[o++] = ((wc - drop) << 16) | op;
                        out[o++] = rw_type[rw];
                        out[o++] = words[i + 2];
                        out[o++] = rw_base[rw];
                        out[o++] = words[i + 4];        /* the control-point index */
                        for (uint32_t q = 5 + drop; q < wc; q++)
                            out[o++] = words[i + q];
                        i += wc;
                    }
                    else
                    {
                        for (uint32_t q = 0; q < wc; q++)
                            out[o++] = words[i + q];
                        i += wc;
                    }
                }
            }
            *out_n = o;
            return out;
        }
    }

unsafe_exit:
    /* a chain loads a whole struct-typed aggregate of a block that would
     * otherwise dissolve: SPIRV-Cross ABORTS compiling such a module, so
     * decline it entirely — return a copy with corrupted magic so every
     * compile fails cleanly and MGL reports a shader/link error */
    free(work);
    {
        uint32_t *bad = malloc(n * 4);

        if (bad)
        {
            memcpy(bad, words, n * 4);
            bad[0] ^= 1;
            *out_n = n;
            return bad;
        }
        return NULL;
    }
bail:
    free(work);
    return NULL;
}
