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

#define SPV_OP_TYPE_MATRIX      24
#define SPV_OP_TYPE_ARRAY       28
#define SPV_OP_TYPE_STRUCT      30
#define SPV_OP_TYPE_POINTER     32
#define SPV_OP_FUNCTION         54
#define SPV_OP_CONSTANT         43
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

enum { SPV_T_OTHER = 0, SPV_T_ARRAY, SPV_T_MATRIX, SPV_T_STRUCT };

static int spv_lower_type_kind(const struct SpvLowerCtx *c, uint32_t type)
{
    int k = spv_lower_find(c, type);
    if (k < 0) return SPV_T_OTHER;
    switch (c->op[k])
    {
        case SPV_OP_TYPE_ARRAY:  return SPV_T_ARRAY;
        case SPV_OP_TYPE_MATRIX: return SPV_T_MATRIX;
        case SPV_OP_TYPE_STRUCT: return SPV_T_STRUCT;
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
};

/* Does this access chain dynamically index an array/matrix MEMBER of an Input
 * block?  (The first index — the control-point array — may be dynamic.) */
static bool spv_lower_offends(struct SpvLowerCtx *c, uint32_t inst, struct SpvLowerSite *site)
{
    uint32_t off = c->off[inst], wc = c->wc[inst];
    uint32_t base = c->w[off + 3];
    int var = spv_lower_find(c, base);
    uint32_t cur, j;

    if (var < 0 || c->op[var] != SPV_OP_VARIABLE)
        return false;
    if (c->wc[var] < 4 || c->w[c->off[var] + 3] != SPV_SC_INPUT)
        return false;
    /* walk from the BASE VARIABLE's pointer type (the chain's own result type
     * points at the final element) */
    cur = spv_lower_pointee(c, c->w[c->off[var] + 1]);
    if (!cur)
        return false;

    j = off + 4;                                    /* first index operand */
    if (spv_lower_type_kind(c, cur) == SPV_T_ARRAY)
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

        if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
        {
            if (lit < 0)
            {
                site->inst = inst;
                site->k = (uint32_t)(j - off);
                site->arr_type = cur;
                site->elem_type = 0;                /* filled by the caller (walks the rest) */
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
        if (kind == SPV_T_ARRAY || kind == SPV_T_MATRIX)
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
        return NULL;
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
    }

    /* validate the sites: need a chain-elem type, an index type (from a
     * constant prefix member index) and the array length */
    for (uint32_t s = 0; s < n_sites; s++)
    {
        uint32_t off = c.off[sites[s].inst];
        if (!sites[s].elem_type || sites[s].k < 5 ||
            !spv_lower_composite_len(&c, sites[s].arr_type))
        {
            sites[s].inst = 0xffffffff;
            continue;
        }
        /* the index type: any constant used as an index in the prefix */
        int ck = spv_lower_find(&c, c.w[off + 4]);
        if (ck < 0 || c.op[ck] != SPV_OP_CONSTANT)
        {
            sites[s].inst = 0xffffffff;
            continue;
        }
        idx_type = c.w[c.off[ck] + 1];
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
        LOW_PUSH(q == 3 ? next_id : words[q]);

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

                /* %eptr = AccessChain ptrIn_elem %base <prefix> %e
                 * words: hdr, rtype, result, base, [4, k) indices, e -> k+1 */
                LOW_PUSH(((st->k + 1) << 16) | SPV_OP_ACCESSCHAIN);
                LOW_PUSH(ptr_in);
                LOW_PUSH(arrptr[s]);
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
    if (!words || n < 5)
        return NULL;

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
    if (!votes)
        return NULL;

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
    if (!out)
        return NULL;
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
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) { free(out); return NULL; }

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
    if (!words || n < 5)
        return NULL;

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
    if (!hit)
        return NULL;

    next_id = words[3];
    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_VARIABLE && wc >= 4 && words[i + 2] == var)
        {
            old_ptr = words[i + 1];
            if (words[i + 3] != SPV_SC_INPUT)
                return NULL;            /* not an input builtin: leave it */
        }
        i += wc;
    }
    if (!old_ptr)
        return NULL;

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
    if (!pointee)
        return NULL;
    if (!new_ptr)
        new_ptr = next_id++;

    size_t need = n + 8 + (new_ptr >= words[3] ? 4 : 0);
    uint32_t *out = (uint32_t *)malloc(need * sizeof(uint32_t));
    if (!out)
        return NULL;
    size_t o = 0;

    for (int q = 0; q < 5; q++)
        out[o++] = (q == 3) ? next_id : words[q];

    for (i = 5; i < n; )
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) { free(out); return NULL; }

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
