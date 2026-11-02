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
};

/* Does this access chain dynamically index an array/matrix MEMBER of an Input
 * block?  (The first index — the control-point array — may be dynamic.)
 * glslang SPLITS long chains: OpAccessChain(%mid, block, 0, member) followed by
 * OpAccessChain(%res, %mid, i) — the dynamic index then sits in a chain whose
 * base is another chain, not the variable.  Resolve through up to four
 * intermediate chains; their indices must be constant (the rewrite re-roots the
 * element copy at the variable and concatenates them). */
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
    if (prefix_n == 0 && spv_lower_type_kind(c, cur) == SPV_T_ARRAY)
    {
        /* the control-point array: its index may be dynamic (gl_in[i]) */
        cur = spv_lower_elem_type(c, cur, 0);
        j++;
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
                    return true;                       /* into an array field: assert in SPIRV-Cross */
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

uint32_t *mgl_tess_lower_dynamic_io(const uint32_t *words, size_t n, size_t *out_n)
{
    struct SpvLowerCtx c;
    struct SpvLowerSite sites[64];
    uint32_t n_sites = 0, next_id;
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

    /* guard 1 — ARRAY-field sites: the copy loop lowers a site by emitting one
     * access chain PER ARRAY ELEMENT that descends into the field with a
     * constant index.  For MATRIX fields that is fine (Metal stage-in structs
     * allow matrix members and column indexing), but ARRAY fields cannot map to
     * tessellation stage-in attributes at all: SPIRV-Cross leaves the member
     * without an interface member index and asserts in
     * emit_tessellation_access_chain (process abort, kills cts-runner) instead
     * of reporting a compile error.  Decline such modules: MGL compiles the
     * original SPIR-V, which SPIRV-Cross rejects cleanly ("Trying to
     * dynamically index into an array interface variable in tessellation"),
     * so the link fails normally.  (dEQP-GL45-ES31
     * user_defined_io.per_vertex_block.vertex_io_array_size_implicit.*) */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        if (sites[s].inst == 0xffffffff)
            continue;
        if (spv_lower_type_kind(&c, sites[s].arr_type) == SPV_T_ARRAY)
        {
            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL tess lower: array-field site %s (MGL_TESS_ARRAY_LOWER=1 lowers it)\n",
                        getenv("MGL_TESS_ARRAY_LOWER") ? "accepted" : "declined");
            if (!getenv("MGL_TESS_ARRAY_LOWER"))
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
        bound += 2u * spv_lower_composite_len(&c, sites[s].arr_type);
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

            for (uint32_t e = 0; ok && e < len; e++)
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
 * fresh malloc'd module with both lowerings applied, or NULL when neither
 * fired (the caller's module is unchanged). */
uint32_t *mgl_spirv_prepare_for_msl(const uint32_t *words, size_t n, size_t *n_out)
{
    uint32_t *mid = mgl_spirv_lower_subgroup_vote(words, n, n_out);
    const uint32_t *cur = mid ? mid : words;
    size_t cn = mid ? *n_out : n;
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

/* value of an int OpConstant id (0xffffffff when not a 32-bit constant) */
static uint32_t spv_dis_const(const uint32_t *words, size_t n, uint32_t id)
{
    for (size_t k = 5; k < n; )
    {
        uint32_t wc = words[k] >> 16, op = words[k] & 0xffff;

        if (wc == 0 || k + wc > n) break;
        if (op == SPV_OP_CONSTANT && wc == 4 && words[k + 2] == id)
            return words[k + 3];
        k += wc;
    }
    return 0xffffffffu;
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

uint32_t *mgl_tess_dissolve_vertex_input_blocks(const uint32_t *words, size_t n, size_t *out_n)
{
    struct DissolveJob jobs[8];
    int n_jobs = 0;
    uint32_t next_id;
    uint32_t *work = NULL;
    size_t i;
    uint32_t names_out[8192], deco_out[4096], type_out[4096], var_out[8 * 64 * 5];
    size_t n_names = 0, n_deco = 0, n_type = 0, n_var = 0;

    *out_n = 0;
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

                        if (mval != 0xffffffffu && mval < jobs[j].n_members)
                            found_m = (int)mval;
                        if (found_m < 0)
                            goto bail;
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
                                    goto bail;              /* dynamic index: clean reject */
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
                                        goto bail;
                                }
                                lv_elem = e;
                                lv_field = f;
                                pointee = 0;        /* set after resolving the field below */
                                q = 8;              /* walk the deeper path only */
                                for (uint32_t d = 6; d <= 7; d++)
                                {
                                    uint32_t cval = spv_dis_const(words, n, words[i + d]);

                                    if (cval == 0xffffffffu) goto bail;
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
                                    if (!nxt) goto bail;
                                }
                                else if (kind == SPV_T_STRUCT)
                                {
                                    uint32_t cval = spv_dis_const(words, n, idx_id);

                                    if (cval == 0xffffffffu) goto bail;
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
