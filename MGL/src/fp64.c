/*
 * MGL software fp64 — SPIR-V lowering pass.  See fp64.h for the design.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * The soft-float64 library called by the rewritten instructions lives in
 * fp64_softfloat.glsl (Berkeley SoftFloat 3e via Mesa's float64.glsl, plus
 * entry points ported from Mesa's nir_lower_double_ops.c).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "spirv.h"
#include "GLSL.std.450.h"
#include "fp64.h"
#include "fp64_softfloat.h"
#include "fp64_df64.h"

struct Lib;   /* selected arithmetic library (defined in the library section) */

#if defined(__has_include)
#if __has_include(<glslang_c_interface.h>) && __has_include(<glslang_c_shader_types.h>)
#define MGL_FP64_GLSLANG 1
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#endif
#if __has_include(<spirv-tools/libspirv.h>)
#include <spirv-tools/libspirv.h>
#define MGL_FP64_SPVTOOLS 1
#endif
#endif

#if !defined(MGL_FP64_GLSLANG) || !defined(MGL_FP64_SPVTOOLS)
uint32_t *mgl_fp64_lower_module(const uint32_t *words, size_t nwords, bool *lowered,
                                size_t *out_nwords, char *log, size_t logsz)
{
    *lowered = false;
    if (out_nwords) *out_nwords = 0;
    if (words && nwords > 5) {
        for (size_t i = 5; i + 1 < nwords; i += (words[i] >> 16) ?: 1) {
            if ((words[i] & 0xffff) == SpvOpTypeFloat && words[i + 2] == 64) {
                *lowered = true;
                if (log && logsz) snprintf(log, logsz, "fp64: MGL built without glslang/SPIRV-Tools");
                return NULL;
            }
        }
    }
    return NULL;
}
#else /* full implementation */

/* ---------------------------------------------------------------- model -- */

enum { K_NONE = 0, K_VOID, K_BOOL, K_INT, K_FLOAT, K_VEC, K_MAT, K_ARR, K_STRUCT, K_PTR, K_FUNC, K_OTHER };

typedef struct {
    uint8_t kind;
    uint8_t is64, has64;
    uint32_t elem;           /* vec component / array element / ptr pointee */
    uint32_t n;              /* vec size / matrix columns */
    uint32_t width;          /* int/float width */
    int32_t sgn;             /* int signedness */
    size_t def;              /* instruction index of the definition */
} Ty;

typedef struct {
    uint16_t op, wc;
    size_t off;
} Ins;

typedef struct { uint32_t id, dec, a; } Dec;
typedef struct { uint32_t type, member, dec, a; } MDec;

typedef struct { uint32_t *w; size_t n, cap; } Buf;

typedef struct { uint32_t elem, n, id; } VecCache;

typedef struct {
    const uint32_t *w;
    size_t n;
    uint32_t bound;

    Ins *ins;
    size_t nins;
    size_t first_def;        /* first types/constants/global-variables instruction */
    size_t first_fn;         /* first OpFunction instruction */

    uint32_t cap_ids;
    Ty *ty;                  /* original module types (+ types created here) */
    uint32_t *vty;           /* value id -> result type id */
    uint64_t *cval;          /* constant id -> value */
    uint8_t  *isconst;

    Dec *decs; size_t ndecs;
    MDec *mdecs; size_t nmdecs;

    uint8_t *iface;          /* type id -> interface (stage in/out) repr */
    uint32_t *iforig;        /*   original double type it stands for */
    uint8_t *rowmajor;       /* type id -> row-major repr of a matrix */
    uint32_t *roworig;
    uint32_t *rowmap;        /* original matrix type -> row-major repr */

    Buf hdradd;              /* OpCapability / OpExtInstImport additions */
    Buf preann;              /* decorations for new types (annotation section) */
    Buf pre;                 /* new types/constants (start of def section) */
    Buf lead;                /* relocated module type defs, before `pre` */
    Buf defs;                /* rewritten def section */
    Buf body;                /* rewritten function bodies */
    Buf libdefs;             /* library types/constants (def section) */
    Buf libfns;              /* library functions (before user bodies) */

    VecCache *vc; size_t nvc;

    uint32_t next_id, id_limit;
    uint32_t libshift;       /* id base where the library is merged in */

    struct { uint32_t sid; uint32_t mem[32]; uint32_t n; } ifmem[32];
    size_t nifmem;

    /* stage-IO element wrappers (see iface_slot_struct) */
    struct { uint32_t id, orig; } pstruct[32]; size_t npstruct;
    struct { uint32_t id, orig; } mstruct[32]; size_t nmstruct;
    /* access-chain id -> original element type of an IO array the chain landed on */
    uint32_t *elemreg;

    char *log; size_t logsz; bool failed;

    uint32_t f64ty;          /* module's Float64 type id: becomes the u64 type */
    uint32_t u32ty, i32ty, boolty, u32ty_own;
    uint32_t k_u64_zero, k_u64_one;
    uint32_t uint4ty, uint2ty, ulong2ty;
    uint32_t modstruct_ty;   /* struct { u64; i32 } for ModfStruct/FrexpStruct */

    uint8_t  *attrslot;      /* pointer type id -> is a double vertex-attribute slot */
    uint32_t *attrn;         /* pointer type id -> attribute component count */
    uint32_t *attrhi;        /* slot0 variable id -> slot1 (hi half) variable id */
    uint32_t *attrhity;      /* hi variable id -> its pointer type */
    uint32_t  nattrhi;
    uint32_t *attrcomp;      /* access-chain id -> (attribute var << 4) | comp+1 */
    uint32_t *attrslotidx;   /* access-chain id -> slot index (arrays/matrices), ~0 = n/a */
    uint8_t  *elemslice;     /* chain id: 1 = lands on a member-SLICE (multi-slot element) */
    struct { uint32_t nslots, rows, id; } slotsc[64];
    size_t nslotsc;
    uint32_t *elemidx;       /* ... element index of the slice */
    uint32_t *elemstride;    /* ... slots per element */
    uint32_t *elemorig;      /* ... the element's original type */
    /* chain id -> the full index walk of a chain that lands on (a slice of) a
     * flat slot-array interface member.  The chain itself is NOT emitted: every
     * slot access becomes one complete OpAccessChain (see rw_access) —
     * SPIRV-Cross's tessellation stage-IO flattening consumes a second-level
     * chain's array index as a block-MEMBER index and walks off the struct. */
    struct { uint32_t base; uint32_t nidx; uint32_t idx[12];
             uint32_t sl0, sln; } *expc;
    uint32_t *attrsgeom;     /* attribute ptr type -> (nelem<<16)|(cols<<8)|rows, 0 = plain */
    uint32_t *attrorig;      /* attribute ptr type -> original pointee type id */
    uint8_t  *ioiface;       /* pointer type id -> stage IO with the u32-pair repr */
    uint32_t *arrlane;       /* access-chain id -> widened buffer-array element type
                              * (vec<lanes>(u64); loads extract lane 0, stores fill it) */
    uint8_t  *warr;          /* array type id -> lanes after widening (0 = untouched) */
    bool      vertex_inputs; /* this module is the vertex stage (Input == attributes) */

    struct { uint32_t storage, pointee, id; } ptrcache[64];
    size_t nptrcache;

    uint32_t moved[32];   /* module type defs relocated into `pre` */
    size_t nmoved;

    const struct Lib *lib; /* selected arithmetic library (soft-float / df64) */
} F64;

/* ------------------------------------------------------------------ out -- */

static void bpush(Buf *b, uint32_t v)
{
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 1024;
        b->w = realloc(b->w, b->cap * sizeof(uint32_t));
    }
    b->w[b->n++] = v;
}

static void binst(Buf *b, uint32_t op, const uint32_t *ops, uint32_t nops)
{
    bpush(b, ((uint32_t)nops + 1) << 16 | op);
    for (uint32_t i = 0; i < nops; i++) bpush(b, ops[i]);
}

static void binst2(Buf *b, uint32_t op, uint32_t a, uint32_t c) { uint32_t o[2] = {a, c}; binst(b, op, o, 2); }
static void binst3(Buf *b, uint32_t op, uint32_t a, uint32_t c, uint32_t d) { uint32_t o[3] = {a, c, d}; binst(b, op, o, 3); }
static void binst4(Buf *b, uint32_t op, uint32_t a, uint32_t c, uint32_t d, uint32_t e) { uint32_t o[4] = {a, c, d, e}; binst(b, op, o, 4); }

static void dump_module(const uint32_t *w, size_t n, const char *tag)
{
    const char *dir = getenv("MGL_FP64_DUMP");
    if (!dir) return;
    /* append a monotonic counter: several stages/programs are compiled per process
     * and overwriting made multi-stage diagnosis impossible */
    static unsigned dump_seq = 0;
    char path[512];
    snprintf(path, sizeof path, "%s/%02u_%s.spv", dir, dump_seq++, tag);
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fwrite(w, 4, n, fp);
    fclose(fp);
}

static void fail(F64 *f, const char *fmt, ...)
{
    va_list ap;
    if (f->failed) return;
    f->failed = true;
    dump_module(f->w, f->n, "failed");
    if (!f->log || !f->logsz) return;
    va_start(ap, fmt);
    vsnprintf(f->log, f->logsz, fmt, ap);
    va_end(ap);
}

/* ---------------------------------------------------------------- parse -- */

static spv_result_t fill_vty(void *user, const spv_parsed_instruction_t *inst)
{
    F64 *f = (F64 *)user;
    if (inst->result_id && inst->result_id < f->cap_ids)
        f->vty[inst->result_id] = inst->type_id;
    return SPV_SUCCESS;
}

static bool parse(F64 *f)
{
    const uint32_t *w = f->w;
    if (f->n < 5 || w[0] != SpvMagicNumber) return false;

    size_t i = 5, nins = 0;
    while (i < f->n) {
        uint32_t wc = w[i] >> 16;
        if (!wc || i + wc > f->n) return false;
        nins++;
        i += wc;
    }

    f->cap_ids = f->bound + 16;
    f->ty = calloc(f->cap_ids, sizeof(Ty));
    f->vty = calloc(f->cap_ids, sizeof(uint32_t));
    f->cval = calloc(f->cap_ids, sizeof(uint64_t));
    f->isconst = calloc(f->cap_ids, 1);
    f->iface = calloc(f->cap_ids, 1);
    f->iforig = calloc(f->cap_ids, sizeof(uint32_t));
    f->rowmajor = calloc(f->cap_ids, 1);
    f->roworig = calloc(f->cap_ids, sizeof(uint32_t));
    f->rowmap = calloc(f->cap_ids, sizeof(uint32_t));

    f->ins = calloc(nins + 1, sizeof(Ins));
    f->nins = nins;
    f->decs = calloc(nins + 8, sizeof(Dec));
    f->mdecs = calloc(nins + 8, sizeof(MDec));
    f->vc = calloc(nins + 64, sizeof(VecCache));

    i = 5;
    size_t k = 0;
    bool saw_def = false;
    while (i < f->n) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        f->ins[k].op = op;
        f->ins[k].wc = wc;
        f->ins[k].off = i;

        switch (op) {
        case SpvOpTypeVoid: case SpvOpTypeBool: case SpvOpTypeInt: case SpvOpTypeFloat:
        case SpvOpTypeVector: case SpvOpTypeMatrix: case SpvOpTypeArray: case SpvOpTypeRuntimeArray:
        case SpvOpTypeStruct: case SpvOpTypePointer: case SpvOpTypeFunction:
        case SpvOpTypeImage: case SpvOpTypeSampler: case SpvOpTypeSampledImage:
        case SpvOpConstant: case SpvOpConstantTrue: case SpvOpConstantFalse:
        case SpvOpConstantComposite: case SpvOpConstantNull: case SpvOpSpecConstant:
        case SpvOpSpecConstantTrue: case SpvOpSpecConstantFalse:
        case SpvOpSpecConstantComposite: case SpvOpSpecConstantOp:
        case SpvOpVariable: case SpvOpUndef:
            if (!saw_def) { saw_def = true; f->first_def = k; }
            break;
        case SpvOpFunction:
            if (!f->first_fn) f->first_fn = k;
            break;
        default:
            break;
        }
        k++;
        i += wc;
    }
    if (!f->first_fn) f->first_fn = nins;

    for (size_t j = 0; j < nins; j++) {
        uint32_t op = f->ins[j].op, wc = f->ins[j].wc;
        const uint32_t *iw = f->w + f->ins[j].off;
        uint32_t id = (wc >= 2) ? iw[1] : 0;
        if (!id || id >= f->cap_ids) continue;
        Ty *t = &f->ty[id];
        bool istype = false;
        switch (op) {
        case SpvOpTypeVoid:      istype = true; t->kind = K_VOID; break;
        case SpvOpTypeBool:      istype = true; t->kind = K_BOOL; break;
        case SpvOpTypeInt:       istype = true; t->kind = K_INT; t->width = iw[2]; t->sgn = iw[3]; break;
        case SpvOpTypeFloat:     istype = true; t->kind = K_FLOAT; t->width = iw[2]; t->n = 1; break;
        case SpvOpTypeVector:    istype = true; t->kind = K_VEC; t->elem = iw[2]; t->n = iw[3]; break;
        case SpvOpTypeMatrix:    istype = true; t->kind = K_MAT; t->elem = iw[2]; t->n = iw[3]; break;
        case SpvOpTypeArray:     istype = true; t->kind = K_ARR; t->elem = iw[2]; t->n = iw[3]; break;
        case SpvOpTypeRuntimeArray: istype = true; t->kind = K_ARR; t->elem = iw[2]; t->n = 0; break;
        case SpvOpTypeStruct:    istype = true; t->kind = K_STRUCT; break;
        case SpvOpTypePointer:   istype = true; t->kind = K_PTR; t->elem = iw[3]; t->n = iw[2]; break;
        case SpvOpTypeFunction:  istype = true; t->kind = K_FUNC; t->elem = iw[2]; break;
        default: break;
        }
        if (istype) t->def = j;

        if (op == SpvOpTypeFloat && iw[2] == 64 && !f->f64ty) f->f64ty = id;
        if (op == SpvOpTypeInt && iw[2] == 32) {
            if (iw[3] == 0 && !f->u32ty) f->u32ty = id;
            if (iw[3] == 1 && !f->i32ty) f->i32ty = id;
        }
        if (op == SpvOpTypeBool && !f->boolty) f->boolty = id;
    }
    for (size_t j = 0; j < nins; j++) {
        uint32_t op = f->ins[j].op, wc = f->ins[j].wc;
        const uint32_t *iw = f->w + f->ins[j].off;
        if ((op == SpvOpConstant || op == SpvOpSpecConstant) && wc >= 4) {
            uint32_t id = iw[2], ty = iw[1];
            if (id >= f->cap_ids) continue;
            f->isconst[id] = 1;
            f->vty[id] = ty;
            if (f->ty[ty].width == 64 && wc >= 5)
                f->cval[id] = (uint64_t)iw[3] | ((uint64_t)iw[4] << 32);
            else
                f->cval[id] = iw[3];
        }
    }

    /* value -> type, via SPIRV-Tools' operand classification */
    {
        spv_context ctx = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
        spv_diagnostic diag = NULL;
        if (spvBinaryParse(ctx, f, f->w, f->n, NULL, fill_vty, &diag) != SPV_SUCCESS) {
            if (diag && diag->error) fail(f, "fp64: module parse: %s", diag->error);
            else fail(f, "fp64: module parse failed");
            if (diag) spvDiagnosticDestroy(diag);
            spvContextDestroy(ctx);
            return !f->failed;
        }
        spvContextDestroy(ctx);
    }

    for (size_t j = 0; j < nins; j++) {
        uint32_t op = f->ins[j].op, wc = f->ins[j].wc;
        const uint32_t *iw = f->w + f->ins[j].off;
        if (op == SpvOpDecorate && wc >= 3) {
            f->decs[f->ndecs].id = iw[1];
            f->decs[f->ndecs].dec = iw[2];
            f->decs[f->ndecs].a = wc >= 4 ? iw[3] : 0;
            f->ndecs++;
        } else if (op == SpvOpMemberDecorate && wc >= 4) {
            f->mdecs[f->nmdecs].type = iw[1];
            f->mdecs[f->nmdecs].member = iw[2];
            f->mdecs[f->nmdecs].dec = iw[3];
            f->mdecs[f->nmdecs].a = wc >= 5 ? iw[4] : 0;
            f->nmdecs++;
        }
    }
    return true;
}

static bool has_dec(F64 *f, uint32_t id, uint32_t dec, uint32_t *val)
{
    for (size_t i = 0; i < f->ndecs; i++)
        if (f->decs[i].id == id && f->decs[i].dec == dec) {
            if (val) *val = f->decs[i].a;
            return true;
        }
    return false;
}

static bool has_mdec(F64 *f, uint32_t type, uint32_t member, uint32_t dec, uint32_t *val)
{
    for (size_t i = 0; i < f->nmdecs; i++)
        if (f->mdecs[i].type == type && f->mdecs[i].member == member && f->mdecs[i].dec == dec) {
            if (val) *val = f->mdecs[i].a;
            return true;
        }
    return false;
}

/* aggregate of Float64? (pointers/functions excluded) */
static bool is64(F64 *f, uint32_t id)
{
    if (id >= f->cap_ids) return false;
    Ty *t = &f->ty[id];
    if (t->has64) return t->is64;
    t->has64 = 1;
    switch (t->kind) {
    case K_FLOAT: t->is64 = (t->width == 64); break;
    case K_VEC: case K_MAT: case K_ARR:
        t->is64 = t->elem ? is64(f, t->elem) : false;
        break;
    case K_STRUCT: {
        uint32_t wc = f->ins[t->def].wc;
        const uint32_t *iw = f->w + f->ins[t->def].off;
        t->is64 = 0;
        for (uint32_t i = 2; i < wc; i++) t->is64 |= is64(f, iw[i]);
        break;
    }
    default: t->is64 = 0; break;
    }
    return t->is64;
}

static bool is_f64(F64 *f, uint32_t id)
{
    return id < f->cap_ids && f->ty[id].kind == K_FLOAT && f->ty[id].width == 64;
}

/* a double scalar or dvecN (the element/column types that flatten to words) */
static bool is_f64_scalar_or_vec(F64 *f, uint32_t id)
{
    Ty *t = &f->ty[id];
    return is_f64(f, id) ||
           (t->kind == K_VEC && t->elem && f->ty[t->elem].kind == K_FLOAT &&
            f->ty[t->elem].width == 64);
}

static uint32_t veclen(F64 *f, uint32_t id) { return id < f->cap_ids ? f->ty[id].n : 0; }

static uint32_t arrlen(F64 *f, uint32_t id)
{
    if (id >= f->cap_ids || f->ty[id].kind != K_ARR || !f->ty[id].n) return 0;
    uint32_t lid = f->ty[id].n;
    return (lid < f->cap_ids && f->isconst[lid]) ? (uint32_t)f->cval[lid] : 0;
}

static uint32_t struct_member(F64 *f, uint32_t sid, uint32_t i)
{
    const uint32_t *iw = f->w + f->ins[f->ty[sid].def].off;
    return iw[2 + i];
}

static uint32_t struct_nmembers(F64 *f, uint32_t sid)
{
    return f->ins[f->ty[sid].def].wc - 2;
}

/* member type of an (possibly interface-remapped) struct */
static uint32_t smember(F64 *f, uint32_t sid, uint32_t i)
{
    for (size_t k = 0; k < f->nifmem; k++)
        if (f->ifmem[k].sid == sid && i < f->ifmem[k].n)
            return f->ifmem[k].mem[i];
    return struct_member(f, sid, i);
}

static uint32_t snmembers(F64 *f, uint32_t sid)
{
    for (size_t k = 0; k < f->nifmem; k++)
        if (f->ifmem[k].sid == sid) return f->ifmem[k].n;
    return struct_nmembers(f, sid);
}

/* constant from a double bit pattern */
static uint32_t k_u64(F64 *f, uint64_t bits);
static uint32_t k_dbl(F64 *f, double v)
{
    union { double d; uint64_t u; } u;
    u.d = v;
    return k_u64(f, u.u);
}

/* ------------------------------------------------------------- new ids  -- */

static uint32_t nid(F64 *f)
{
    if (f->next_id >= f->id_limit) {
        fail(f, "fp64: out of SPIR-V id space");
        return 1;
    }
    return f->next_id++;
}

static uint32_t mk_int_type(F64 *f, uint32_t width, uint32_t sgn)
{
    uint32_t id = nid(f);
    f->ty[id].kind = K_INT; f->ty[id].width = width; f->ty[id].sgn = sgn;
    uint32_t o[3] = {id, width, sgn};
    binst(&f->pre, SpvOpTypeInt, o, 3);
    return id;
}

static uint32_t u32_for_pre(F64 *f);

static uint32_t k_u32(F64 *f, uint32_t v)
{
    uint32_t id = nid(f);
    uint32_t u32t = u32_for_pre(f);
    f->vty[id] = u32t;
    f->isconst[id] = 1;
    f->cval[id] = v;
    uint32_t o[3] = {u32t, id, v};
    binst(&f->pre, SpvOpConstant, o, 3);
    return id;
}

static uint32_t k_u64(F64 *f, uint64_t bits)
{
    if (bits == 0 && f->k_u64_zero) return f->k_u64_zero;
    if (bits == 0x3ff0000000000000ULL && f->k_u64_one) return f->k_u64_one;
    uint32_t id = nid(f);
    f->vty[id] = f->f64ty;
    f->isconst[id] = 1;
    f->cval[id] = bits;
    uint32_t o[4] = {f->f64ty, id, (uint32_t)bits, (uint32_t)(bits >> 32)};
    binst(&f->pre, SpvOpConstant, o, 4);
    if (bits == 0) f->k_u64_zero = id;
    if (bits == 0x3ff0000000000000ULL) f->k_u64_one = id;
    return id;
}

/* vector type; reuse module types when safe (no iface marking involved) */
static bool is_moved(F64 *f, uint32_t id)
{
    for (size_t i = 0; i < f->nmoved; i++) if (f->moved[i] == id) return true;
    return false;
}

static void mark_moved(F64 *f, uint32_t id)
{
    if (id && !is_moved(f, id) && f->nmoved < 32) f->moved[f->nmoved++] = id;
}

static uint32_t mkvec(F64 *f, uint32_t elem, uint32_t n, bool fresh)
{
    (void)fresh;
    for (size_t i = 0; i < f->nvc; i++)
        if (f->vc[i].elem == elem && f->vc[i].n == n) return f->vc[i].id;
    uint32_t id = nid(f);
    f->ty[id].kind = K_VEC; f->ty[id].elem = elem; f->ty[id].n = n;
    f->ty[id].has64 = 1; f->ty[id].is64 = is64(f, elem);
    f->vc[f->nvc].elem = elem; f->vc[f->nvc].n = n; f->vc[f->nvc].id = id;
    f->nvc++;
    uint32_t o[3] = {id, elem, n};
    binst(&f->pre, SpvOpTypeVector, o, 3);
    return id;
}

/* module vector type id for (elem, n), or 0 */
static uint32_t module_vec(F64 *f, uint32_t elem, uint32_t n)
{
    for (size_t i = 0; i < f->nins; i++)
        if (f->ins[i].op == SpvOpTypeVector && f->w[f->ins[i].off + 2] == elem &&
            f->w[f->ins[i].off + 3] == n)
            return f->w[f->ins[i].off + 1];
    return 0;
}

static uint32_t mkarray(F64 *f, uint32_t elem, uint32_t len)
{
    uint32_t l = k_u32(f, len);
    uint32_t id = nid(f);
    f->ty[id].kind = K_ARR; f->ty[id].elem = elem; f->ty[id].n = l;
    f->ty[id].has64 = 1; f->ty[id].is64 = is64(f, elem);
    uint32_t o[3] = {id, elem, l};
    binst(&f->pre, SpvOpTypeArray, o, 3);
    return id;
}

/* std140 packs arrays of *scalar* doubles with an ArrayStride of 16 while the
 * lowered element (u64) is 8 bytes. SPIRV-Cross widens such elements to a
 * vector itself when it repacks buffer blocks, but its scalar-u64 un-swizzle
 * never applies to our soft-float OpFunctionCall arguments ("no matching
 * function", ulong2 where ulong is expected), so widen them here instead:
 * the element becomes vec<n>(u64) whose natural MSL size equals the decorated
 * stride (layout stays exact, SPIRV-Cross needs no remapping), and access
 * chains landing on an element gain a trailing [0] so the value is the plain
 * u64 again. Returns the lane count, 0 when the array is not widened. */
static uint32_t widen_buffer_array(F64 *f, uint32_t arrid)
{
    if (arrid >= f->cap_ids || f->ty[arrid].kind != K_ARR) return 0;
    if (f->warr[arrid]) return f->warr[arrid];
    uint32_t stride = 0;
    if (!has_dec(f, arrid, SpvDecorationArrayStride, &stride)) return 0;
    if (stride <= 8 || stride > 32 || (stride & 7)) return 0;
    if (f->ty[arrid].elem != f->f64ty)
        return 0;    /* vector elements already match their natural stride */
    uint32_t lanes = stride / 8;
    uint32_t velem = module_vec(f, f->f64ty, lanes);
    if (!velem) velem = mkvec(f, f->f64ty, lanes, false);
    f->ty[arrid].elem = velem;
    f->ty[arrid].is64 = 1;
    f->warr[arrid] = (uint8_t)lanes;
    return lanes;
}


static uint32_t u32_for_pre(F64 *f)
{
    /* SPIR-V forbids duplicate non-aggregate types: reuse the module's u32
       (relocated into `pre`) when there is one */
    if (f->u32ty) {
        mark_moved(f, f->u32ty);
        return f->u32ty;
    }
    if (!f->u32ty_own) f->u32ty_own = mk_int_type(f, 32, 0);
    return f->u32ty_own;
}

static uint32_t uint2_of(F64 *f)
{
    if (f->uint2ty) return f->uint2ty;
    uint32_t u = u32_for_pre(f);
    uint32_t m = module_vec(f, u, 2);
    if (m) { mark_moved(f, m); f->uint2ty = m; return m; }
    f->uint2ty = mkvec(f, u, 2, true);
    return f->uint2ty;
}

static uint32_t uint4_of(F64 *f)
{
    if (f->uint4ty) return f->uint4ty;
    uint32_t u = u32_for_pre(f);
    uint32_t m = module_vec(f, u, 4);
    if (m) { mark_moved(f, m); f->uint4ty = m; return m; }
    f->uint4ty = mkvec(f, u, 4, true);
    return f->uint4ty;
}

static uint32_t ulong2_of(F64 *f)
{
    if (f->ulong2ty) return f->ulong2ty;
    uint32_t m = module_vec(f, f->f64ty, 2);
    if (m) { mark_moved(f, m); f->ulong2ty = m; return m; }
    f->ulong2ty = mkvec(f, f->f64ty, 2, true);
    return f->ulong2ty;
}

static void dec_stride(F64 *f, uint32_t typeid, uint32_t stride)
{
    uint32_t o[3] = {typeid, SpvDecorationArrayStride, stride};
    binst(&f->preann, SpvOpDecorate, o, 3);
}

/* get-or-create pointer type (may reuse a module type) */
static uint32_t ptr_for(F64 *f, uint32_t storage, uint32_t pointee)
{
    for (size_t i = 0; i < f->nins; i++)
        if (f->ins[i].op == SpvOpTypePointer && f->w[f->ins[i].off + 2] == storage &&
            f->w[f->ins[i].off + 3] == pointee)
            return f->w[f->ins[i].off + 1];
    for (size_t i = 0; i < f->nptrcache; i++)
        if (f->ptrcache[i].storage == storage && f->ptrcache[i].pointee == pointee)
            return f->ptrcache[i].id;
    uint32_t id = nid(f);
    f->ty[id].kind = K_PTR; f->ty[id].elem = pointee; f->ty[id].n = storage;
    uint32_t o[3] = {id, storage, pointee};
    binst(&f->pre, SpvOpTypePointer, o, 3);
    if (f->nptrcache < 64) {
        f->ptrcache[f->nptrcache].storage = storage;
        f->ptrcache[f->nptrcache].pointee = pointee;
        f->ptrcache[f->nptrcache].id = id;
        f->nptrcache++;
    }
    return id;
}

/* ---------------------------------------------------- interface repr     -- */
/* Stage in/out double values cross the boundary as uvec2 / uint4 / arrays of
 * uint4 (two doubles per uint4 = one 128-bit location), because MSL has no
 * vertex format able to feed a ulongN attribute. */

static uint32_t iface_ty(F64 *f, uint32_t orig);
static uint32_t iface_slots(F64 *f, uint32_t ety);

static uint32_t iface_mark(F64 *f, uint32_t id, uint32_t orig)
{
    f->iface[id] = 1;
    f->iforig[id] = orig;
    return id;
}

static uint32_t iface_slot_struct(F64 *f, uint32_t nslots, uint32_t rows)
{
    /* cache by (nslots, rows) — types must not multiply across statements */
    for (size_t k = 0; k < f->nslotsc; k++)
        if (f->slotsc[k].nslots == nslots && f->slotsc[k].rows == rows)
            return f->slotsc[k].id;
    if (nslots == 0 || nslots > 32) return uint4_of(f);
    uint32_t mty = (rows == 1) ? uint2_of(f) : uint4_of(f);
    uint32_t id = nid(f);
    uint32_t *o = malloc((nslots + 1) * 4);
    o[0] = id;
    for (uint32_t i = 0; i < nslots; i++) o[i + 1] = mty;
    binst(&f->pre, SpvOpTypeStruct, o, nslots + 1);
    free(o);
    f->ty[id].kind = K_STRUCT; f->ty[id].has64 = 0; f->ty[id].is64 = 0; f->ty[id].def = 0;
    /* register the members so smember()/struct_nmembers() work without a def */
    if (f->nifmem < 32) {
        f->ifmem[f->nifmem].sid = id;
        for (uint32_t i = 0; i < nslots && i < 32; i++) f->ifmem[f->nifmem].mem[i] = mty;
        f->ifmem[f->nifmem].n = nslots;
        f->nifmem++;
    }
    f->slotsc[f->nslotsc % 64].nslots = nslots;
    f->slotsc[f->nslotsc % 64].rows = rows;
    f->slotsc[f->nslotsc % 64].id = id;
    f->nslotsc++;
    return id;
}


/* struct { uint4 f[C*slots]; } — the per-element representation of a dmatNxR
 * inside a stage-IO ARRAY (same reason as iface_pair_struct). */
/* ------------------------------------------------ vertex-attribute slots  --
 * Metal vertex descriptors fetch at most 4 x 32 bits per attribute slot and
 * MSL has no double at all, so a double/dvecN vertex input is fetched as its
 * RAW u32 words (two per component) and reassembled in the shader:
 *     double -> uint2   at location k          (8 bytes)
 *     dvec2  -> uint4   at location k          (16 bytes)
 *     dvec3  -> uint4 @ k + uint2 @ k+1        (ARB_vertex_attrib_64bit:
 *     dvec4  -> uint4 @ k + uint4 @ k+1         dvec3/dvec4 span two locations)
 * The renderer writes the words: glVertexAttribLPointer feeds the double's
 * bits verbatim (bit-exact), glVertexAttribPointer converts per the spec
 * (GL_DOUBLE -> float32 -> double again, exact round trip of the mandated
 * conversion).  Loads of the original variable load both slots and assemble
 * the u64 register value with bitcasts (no arithmetic involved). */
static uint32_t attr_slot0_ty(F64 *f, uint32_t orig)
{
    Ty *t = &f->ty[orig];
    if (t->kind == K_FLOAT) return uint2_of(f);
    if (t->kind == K_VEC && t->n >= 2) return uint4_of(f);
    if (t->kind == K_VEC) return uint2_of(f);
    return 0;    /* arrays/matrices/structs of doubles are not vertex attributes */
}

static uint32_t attr_slot1_ty(F64 *f, uint32_t n)
{
    return (n == 3) ? uint2_of(f) : uint4_of(f);
}

/* is this a double scalar/vector, or a matrix of them? */
static bool is_f64_scalar_or_vec_or_mat(F64 *f, uint32_t id)
{
    if (is_f64_scalar_or_vec(f, id)) return true;
    return f->ty[id].kind == K_MAT && is_f64_scalar_or_vec(f, f->ty[id].elem);
}

/* slot geometry of an ARRAY-of-double or MATRIX vertex attribute:
 *   nelem  elements (each: cols columns of `rows` doubles)
 * Columns are slot-ALIGNED: a column with an odd row count pads its last slot
 * (a slot is one Metal location, uint4 = 2 doubles), so every (element, column)
 * maps to whole slots — the app-facing location layout ("location k+i*cols+c
 * holds column c of element i") is what the current-value and shadow-stream
 * deliveries feed. */
static bool attr_geom(F64 *f, uint32_t ty, uint32_t *nelem, uint32_t *cols, uint32_t *rows)
{
    *nelem = 1; *cols = 1; *rows = 1;
    if (f->ty[ty].kind == K_ARR)
    {
        if (!is_f64_scalar_or_vec_or_mat(f, f->ty[ty].elem)) return false;
        *nelem = arrlen(f, ty);
        ty = f->ty[ty].elem;
        if (f->ty[ty].kind == K_ARR) return false;   /* arrays of arrays: not attributes */
    }
    if (f->ty[ty].kind == K_MAT)
    {
        if (!is_f64_scalar_or_vec(f, f->ty[ty].elem)) return false;
        *cols = f->ty[ty].n;
        *rows = veclen(f, f->ty[ty].elem);
    }
    else if (f->ty[ty].kind == K_VEC) *rows = f->ty[ty].n;
    else if (f->ty[ty].kind != K_FLOAT) return false;
    if (!*nelem || !*cols || !*rows || *rows > 4) return false;
    return true;
}

static uint32_t attr_slots_per_col(uint32_t rows) { return (rows + 1) / 2; }

/* the slot representation of an ARRAY-of-double / double-matrix vertex
 * attribute: array of uint2/uint4, one element per Metal location.  SPIRV-Cross
 * flattens an array stage input into one [[attribute(k+j)]] per element, which
 * is exactly the delivery shape. */
static uint32_t attr_slotarr_ty(F64 *f, uint32_t orig)
{
    uint32_t nelem, cols, rows;

    if (!attr_geom(f, orig, &nelem, &cols, &rows)) return 0;
    uint32_t spc = attr_slots_per_col(rows);
    uint32_t slot = (rows == 1) ? uint2_of(f) : uint4_of(f);
    return mkarray(f, slot, nelem * cols * spc);
}

/* copy OpName/OpMemberName and the interface member decorations of an
   interface block onto its u32-slot mapping: SPIRV-Cross (and MGL's own
   geometry/tess reflection) find the block and its members by name, and the
   Block decoration is what marks the struct as an interface block. */
static void iface_copy_block_meta(F64 *f, uint32_t orig, uint32_t mapped)
{
    if (has_dec(f, orig, SpvDecorationBlock, NULL)) {
        uint32_t o[2] = {mapped, SpvDecorationBlock};
        binst(&f->preann, SpvOpDecorate, o, 2);
    }
    for (size_t j = 0; j < f->nins; j++) {
        uint32_t op = f->ins[j].op, wc = f->ins[j].wc;
        const uint32_t *iw = f->w + f->ins[j].off;
        if (op == SpvOpName && wc >= 3 && iw[1] == orig) {
            bpush(&f->preann, (wc << 16) | SpvOpName);
            bpush(&f->preann, mapped);
            for (uint32_t i = 2; i < wc; i++) bpush(&f->preann, iw[i]);
        } else if (op == SpvOpMemberName && wc >= 4 && iw[1] == orig) {
            bpush(&f->preann, (wc << 16) | SpvOpMemberName);
            bpush(&f->preann, mapped);
            for (uint32_t i = 2; i < wc; i++) bpush(&f->preann, iw[i]);
        }
    }
    for (size_t i = 0; i < f->nmdecs; i++) {
        if (f->mdecs[i].type != orig) continue;
        switch (f->mdecs[i].dec) {
        case SpvDecorationLocation: case SpvDecorationComponent:
        case SpvDecorationXfbBuffer: case SpvDecorationStream:
        case SpvDecorationBuiltIn: case SpvDecorationOffset: {
            uint32_t o[4] = {mapped, f->mdecs[i].member, f->mdecs[i].dec, f->mdecs[i].a};
            binst(&f->preann, SpvOpMemberDecorate, o, 4);   /* decorations with a literal */
            break;
        }
        case SpvDecorationFlat: case SpvDecorationNoPerspective:
        case SpvDecorationCentroid: case SpvDecorationSample: case SpvDecorationPatch: {
            uint32_t o[3] = {mapped, f->mdecs[i].member, f->mdecs[i].dec};
            binst(&f->preann, SpvOpMemberDecorate, o, 3);   /* no literal operand */
            break;
        }
        default: break;   /* RowMajor/MatrixStride do not apply to the mapping */
        }
    }
}

static uint32_t iface_struct(F64 *f, uint32_t orig)
{
    uint32_t nm = struct_nmembers(f, orig);
    uint32_t *mem = malloc((nm + 1) * sizeof(uint32_t));
    bool any = false;
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t mt = struct_member(f, orig, i);
        mem[i] = iface_ty(f, mt);
        if (mem[i] != mt) any = true;
    }
    if (!any) { free(mem); return orig; }
    uint32_t id = nid(f);
    f->ty[id].kind = K_STRUCT; f->ty[id].def = f->ty[orig].def; f->ty[id].has64 = 1;
    uint32_t *o = malloc((nm + 1) * sizeof(uint32_t));
    o[0] = id;
    for (uint32_t i = 0; i < nm; i++) o[i + 1] = mem[i];
    binst(&f->pre, SpvOpTypeStruct, o, nm + 1);
    free(o);
    if (nm < 32 && f->nifmem < 32) {
        f->ifmem[f->nifmem].sid = id;
        f->ifmem[f->nifmem].n = nm;
        for (uint32_t i = 0; i < nm; i++) f->ifmem[f->nifmem].mem[i] = mem[i];
        f->nifmem++;
    }
    free(mem);
    iface_copy_block_meta(f, orig, id);
    return iface_mark(f, id, orig);
}

static uint32_t iface_of_scalar_or_vec(F64 *f, uint32_t orig)
{
    uint32_t n = f->ty[orig].n;
    if (n <= 1) return iface_mark(f, uint2_of(f), orig);
    if (n == 2) return iface_mark(f, uint4_of(f), orig);
    /* dvec3/dvec4: a FLAT array of uint4 slots, one per 128-bit location.  No
     * nested struct may appear inside an interface block: MGL's tessellation
     * and geometry record layouts parse the MSL stage structs member-by-member
     * and splice canonical bodies, and a struct-typed member is invisible to
     * them.  Loads/stores fan out per slot (rw_load_store), so no whole-array
     * value is ever copied across address spaces (SPIRV-Cross's
     * spvArrayCopyFrom* helpers break in MGL's device-buffer stage IO). */
    return iface_mark(f, mkarray(f, uint4_of(f), (n + 1) / 2), orig);
}

static uint32_t iface_ty(F64 *f, uint32_t orig)
{
    Ty *t = &f->ty[orig];
    switch (t->kind) {
    case K_FLOAT:
        return iface_of_scalar_or_vec(f, orig);
    case K_VEC:
        if (is_f64(f, t->elem)) return iface_of_scalar_or_vec(f, orig);
        return orig;
    case K_MAT: {
        /* FLAT array of uint4, one per GL location (see iface_of_scalar_or_vec
         * for why the members must not be a struct): a dmatCxR column (dvecR)
         * takes ceil(R/2) locations, columns slot-aligned.  Matrices always use
         * uint4 slots — geometry.c's generated variants (geom_double_iface)
         * pack `uvec4 name[cols*ceil(rows/2)]` the same way, and the two
         * representations must match bit-for-bit where the stages meet. */
        if (!is_f64_scalar_or_vec(f, t->elem)) return orig;
        uint32_t slots = iface_slots(f, t->elem);
        return iface_mark(f, mkarray(f, uint4_of(f), t->n * slots), orig);
    }
    case K_ARR: {
        if (!t->n) return orig;
        if (is_f64_scalar_or_vec(f, t->elem)) {
            uint32_t n = (f->ty[t->elem].kind == K_FLOAT) ? 1 : f->ty[t->elem].n;
            if (n <= 2)
            {
                /* scalar/dvec2 elements stay a real array (uint2/uint4 elements):
                 * `x[i]` with a DYNAMIC i must stay expressible (slot fan-out
                 * needs a literal index), and these members are only ever stored
                 * element-wise */
                return iface_mark(f, mkarray(f, (n == 1) ? uint2_of(f) : uint4_of(f),
                                             arrlen(f, orig)), orig);
            }
            /* dvec3/dvec4 elements: one flat uint4 array for the whole member,
             * element i occupying slots [i*slots, i*slots+slots).  Element
             * indexes fold to that slot range (literal indexes only — see the
             * elemslice recording in rw_access). */
            uint32_t slots = iface_slots(f, t->elem);
            return iface_mark(f, mkarray(f, uint4_of(f), arrlen(f, orig) * slots), orig);
        }
        if (f->ty[t->elem].kind == K_MAT && is_f64_scalar_or_vec(f, f->ty[t->elem].elem)) {
            /* arrays of double matrices: the same one-flat-array rule (each
             * element's columns are slot-aligned inside it) */
            uint32_t mslots = f->ty[t->elem].n * iface_slots(f, f->ty[t->elem].elem);
            return iface_mark(f, mkarray(f, uint4_of(f), arrlen(f, orig) * mslots), orig);
        }
        if (f->ty[t->elem].kind == K_STRUCT && is64(f, t->elem)) {
            /* array of an interface block (TCS/GS/TES `in BLOCK name[]`): map
             * the block struct itself so member access lands on the u32 slots */
            uint32_t es = iface_ty(f, t->elem);
            if (es != t->elem)
                return iface_mark(f, mkarray(f, es, arrlen(f, orig)), orig);
        }
        return orig;
    }
    case K_STRUCT:
        return iface_struct(f, orig);
    default:
        return orig;
    }
}

/* ------------------------------------------------------ row-major repr  -- */

static bool struct_uses(F64 *f, uint32_t sid, uint32_t tid);

static uint32_t rowmajor_ty(F64 *f, uint32_t orig_mat)
{
    if (f->rowmap[orig_mat]) return f->rowmap[orig_mat];
    Ty *m = &f->ty[orig_mat];
    uint32_t rows = veclen(f, m->elem);          /* components per column */
    uint32_t rv = module_vec(f, f->f64ty, rows);
    if (rv) mark_moved(f, rv);
    uint32_t rowvec = rv ? rv : mkvec(f, f->f64ty, rows, true);
    uint32_t id = mkarray(f, rowvec, m->n);
    /* row stride follows the same std140/430 rule as any other double vector */
    uint32_t stride = (rows == 1) ? 8 : (rows == 2) ? 16 : 32;
    uint32_t ms = 0;
    for (size_t i = 0; i < f->nmdecs; i++)
        if (f->mdecs[i].type /* any struct */ && f->mdecs[i].dec == SpvDecorationMatrixStride &&
            struct_uses(f, f->mdecs[i].type, orig_mat)) { ms = f->mdecs[i].a; break; }
    dec_stride(f, id, ms ? ms : stride);
    f->rowmajor[id] = 1;
    f->roworig[id] = orig_mat;
    f->rowmap[orig_mat] = id;
    return id;
}

/* does struct `sid` (transitively) contain type `tid`? */
static bool struct_uses(F64 *f, uint32_t sid, uint32_t tid)
{
    if (sid == tid) return true;
    if (sid >= f->cap_ids || f->ty[sid].kind != K_STRUCT) return false;
    uint32_t nm = struct_nmembers(f, sid);
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t mt = struct_member(f, sid, i);
        if (mt == tid) return true;
        if (mt < f->cap_ids && f->ty[mt].kind == K_STRUCT && struct_uses(f, mt, tid)) return true;
    }
    return false;
}

/* ------------------------------------------------------- library        -- */

typedef struct { char name[48]; uint32_t id; } LibFn;

typedef struct Lib {
    uint32_t *w;
    size_t n;
    uint32_t bound;
    uint32_t main_id;
    LibFn fns[96];
    size_t nfns;
} Lib;

static Lib g_lib_soft, g_lib_fast;
static bool g_lib_tried[2] = { false, false };

/*
 * MGL_FP64_FAST=1 selects the double-single (float-float) arithmetic library
 * instead of exact SoftFloat.  Opt-in only: with the variable unset (or "0")
 * the exact SoftFloat path is used and the lowering is byte-for-byte the
 * historic one.  The two libraries expose the same entry-point names and
 * signatures, so nothing else in this pass changes.  Accuracy limits of the
 * fast library are documented in agents/tess_report.md (~48 significant bits,
 * |x| < 2^127, no double subnormals).
 */
static int fp64_lib_mode(void)
{
    const char *v = getenv("MGL_FP64_FAST");
    return (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
}

extern const glslang_resource_t *glslang_default_resource(void);

static bool compile_lib(Lib *out, const char *src_lib, char *log, size_t logsz)
{
    const char *wrap =
        "#version 460 core\n"
        "#extension GL_ARB_gpu_shader_int64 : require\n";
    size_t n1 = strlen(wrap), n2 = strlen(src_lib);    char *src = malloc(n1 + n2 + 1);
    memcpy(src, wrap, n1);
    memcpy(src + n1, src_lib, n2);
    src[n1 + n2] = 0;

    glslang_input_t input;
    memset(&input, 0, sizeof(input));
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_VERTEX;
    input.client = GLSLANG_CLIENT_OPENGL;
    input.client_version = GLSLANG_TARGET_OPENGL_450;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_0;
    input.code = src;
    input.default_version = 460;
    input.default_profile = GLSLANG_CORE_PROFILE;
    input.force_default_version_and_profile = 1;
    input.messages = GLSLANG_MSG_DEFAULT_BIT | GLSLANG_MSG_RELAXED_ERRORS_BIT;
    input.resource = glslang_default_resource();

    bool ok = false;
    glslang_shader_t *sh = glslang_shader_create(&input);
    if (!sh) {
        snprintf(log, logsz, "fp64: glslang_shader_create failed");
    } else if (!glslang_shader_preprocess(sh, &input)) {
        snprintf(log, logsz, "fp64 lib: %s", glslang_shader_get_info_log(sh));
    } else if (!glslang_shader_parse(sh, &input)) {
        snprintf(log, logsz, "fp64 lib: %s", glslang_shader_get_info_log(sh));
    } else {
        glslang_program_t *prog = glslang_program_create();
        glslang_program_add_shader(prog, sh);
        if (!glslang_program_link(prog, input.messages)) {
            snprintf(log, logsz, "fp64 lib link: %s", glslang_program_get_info_log(prog));
        } else {
            glslang_program_SPIRV_generate(prog, GLSLANG_STAGE_VERTEX);
            size_t n = glslang_program_SPIRV_get_size(prog);
            uint32_t *w = malloc(n * sizeof(uint32_t));
            glslang_program_SPIRV_get(prog, w);
            out->w = w;
            out->n = n;
            out->bound = w[3];
            ok = true;
        }
        glslang_program_delete(prog);
    }
    if (sh) glslang_shader_delete(sh);
    free(src);
    return ok;
}

/* collect entry point + "mgl_*" function names from the library module */
static void scan_lib(Lib *lib)
{
    size_t i = 5;
    while (i < lib->n) {
        uint32_t wc = lib->w[i] >> 16, op = lib->w[i] & 0xffff;
        if (op == SpvOpEntryPoint && lib->main_id == 0) lib->main_id = lib->w[i + 2];
        else if (op == SpvOpName && wc >= 3) {
            const char *s = (const char *)&lib->w[i + 2];
            if (!strncmp(s, "mgl_", 4) && lib->nfns < 96) {
                size_t j;
                for (j = 0; j + 1 < sizeof(lib->fns[0].name); j++) {
                    lib->fns[lib->nfns].name[j] = s[j];
                    if (!s[j] || s[j] == '(') break;
                }
                lib->fns[lib->nfns].name[j] = 0;
                lib->fns[lib->nfns].id = lib->w[i + 1];
                lib->nfns++;
            }
        }
        i += wc;
    }
}

/* select + compile the library for the current mode; NULL on failure */
static const Lib *ensure_lib(char *log, size_t logsz)
{
    int mode = fp64_lib_mode();
    Lib *lib = mode ? &g_lib_fast : &g_lib_soft;
    if (g_lib_tried[mode]) return lib->w ? lib : NULL;
    g_lib_tried[mode] = true;
    if (!compile_lib(lib, mode ? MGL_FP64_DF64_LIB_GLSL : MGL_FP64_LIB_GLSL, log, logsz)) {
        memset(lib, 0, sizeof(*lib));
        return NULL;
    }
    scan_lib(lib);
    return lib;
}

static uint32_t lib_fn(F64 *f, const char *fn)
{
    const Lib *lib = f->lib;
    size_t n = strlen(fn);
    for (size_t i = 0; i < lib->nfns; i++)
        if (!strncmp(lib->fns[i].name, fn, n) && (lib->fns[i].name[n] == 0 || lib->fns[i].name[n] == '('))
            return lib->fns[i].id + f->libshift;
    fail(f, "fp64: soft-float function %s missing", fn);
    return 0;
}

/* --------------------------------------------------------------- merge   -- */
/* Append the library to the module, shifting every library id by libshift.
 * The keep-alive main() and its globals are dropped. */

/* canonical key of a non-aggregate type instruction (SPIR-V forbids duplicate
 * non-aggregate type declarations; SPIRV-Cross's parser is not required to
 * survive them and did crash).  Library ids in the operands must be remapped
 * first; `c` is NULL when scanning the module's own types. */
typedef struct {
    uint32_t op, a, b, hash;
    uint32_t id;
} TyKey;

typedef struct {
    F64 *f;
    Buf *target;            /* libdefs until the first OpFunction, then libfns */
    uint32_t shift;
    bool dropping;
    uint32_t skipped[8];
    size_t nskipped;
    uint32_t remap_from[256], remap_to[256];
    size_t nremap;
    TyKey *tab; size_t ntab, tabcap;   /* the module's non-aggregate types */
} MergeCtx;

static uint32_t remap_of(MergeCtx *c, uint32_t id);

static uint32_t remap_of(MergeCtx *c, uint32_t id)
{
    if (!c) return id;
    for (size_t i = 0; i < c->nremap; i++)
        if (c->remap_from[i] == id) return c->remap_to[i];
    return id;
}

static bool ty_key(MergeCtx *c, uint32_t op, const uint32_t *iw, uint32_t wc, TyKey *k)
{
    k->op = op; k->a = k->b = k->hash = 0; k->id = 0;
    switch (op) {
    case SpvOpTypeVoid: case SpvOpTypeBool: case SpvOpTypeSampler:
    case SpvOpTypeEvent: case SpvOpTypeDeviceEvent: case SpvOpTypeReserveId:
    case SpvOpTypeQueue: case SpvOpTypePipe: case SpvOpTypeAccelerationStructureNV:
        return true;
    case SpvOpTypeInt: case SpvOpTypeFloat:
        k->a = iw[2];
        k->b = (op == SpvOpTypeInt) ? iw[3] : 0;
        if (op == SpvOpTypeFloat && iw[2] == 64) k->op = SpvOpTypeInt, k->b = 0; /* rewritten in place */
        return true;
    case SpvOpTypeVector: case SpvOpTypeMatrix:
        k->a = remap_of(c, iw[2]); k->b = iw[3]; return true;
    case SpvOpTypePointer:
        k->a = iw[2]; k->b = remap_of(c, iw[3]); return true;
    case SpvOpTypeFunction:
        k->a = remap_of(c, iw[2]);
        for (uint32_t i = 3; i < wc; i++) k->hash = k->hash * 31u + remap_of(c, iw[i]);
        return true;
    default:
        return false;       /* aggregates & exotics may repeat */
    }
}

static void tykey_scan(F64 *f, MergeCtx *c, const uint32_t *w, size_t wn,
                       TyKey **tab, size_t *n, size_t *cap)
{
    (void)f;
    if (!w) return;
    /* raw streams: 5-word header (module), 0 for a bare `pre` buffer */
    size_t start = (wn >= 5 && w[0] == SpvMagicNumber) ? 5 : 0;
    for (size_t k = start; k + 1 < wn; ) {        uint32_t wc = w[k] >> 16, op = w[k] & 0xffff;
        if (!wc) break;
        if (op >= SpvOpTypeVoid && op <= SpvOpTypeFunction) {
            TyKey key;
            if (ty_key(c, op, w + k, wc, &key) && key.op >= SpvOpTypeVoid && key.op <= SpvOpTypeFunction) {
                key.id = w[k + 1];
                if (*n == *cap) {
                    *cap = *cap ? *cap * 2 : 256;
                    *tab = realloc(*tab, *cap * sizeof **tab);
                }
                if (*tab) (*tab)[(*n)++] = key;
            }
        }
        k += wc;
    }
}

static bool is_id_operand(spv_operand_type_t t)
{
    return t == SPV_OPERAND_TYPE_ID || t == SPV_OPERAND_TYPE_TYPE_ID ||
           t == SPV_OPERAND_TYPE_RESULT_ID || t == SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID ||
           t == SPV_OPERAND_TYPE_SCOPE_ID || t == SPV_OPERAND_TYPE_OPTIONAL_ID;
}

static spv_result_t merge_cb(void *user, const spv_parsed_instruction_t *inst)
{
    MergeCtx *c = user;
    F64 *f = c->f;
    uint32_t op = inst->opcode;

    if (c->dropping) {
        if (op == SpvOpFunctionEnd) c->dropping = false;
        return SPV_SUCCESS;
    }

    switch (op) {
    case SpvOpCapability: case SpvOpExtension: case SpvOpExtInstImport:
    case SpvOpMemoryModel: case SpvOpEntryPoint: case SpvOpExecutionMode:
    case SpvOpExecutionModeId: case SpvOpSource: case SpvOpSourceContinued:
    case SpvOpSourceExtension: case SpvOpString: case SpvOpModuleProcessed:
    case SpvOpName: case SpvOpMemberName:
        return SPV_SUCCESS;                 /* handled by the caller / dropped */
    default:
        break;
    }

    uint32_t rid = inst->result_id;

    if (op == SpvOpFunction && c->target != &f->libfns) c->target = &f->libfns;
    if (op == SpvOpFunction && c->f->lib->main_id && rid == c->f->lib->main_id) {
        c->dropping = true;
        return SPV_SUCCESS;
    }
    if (op == SpvOpVariable && inst->num_words >= 4 && inst->words[3] != SpvStorageClassFunction)
        return SPV_SUCCESS;
    if (op == SpvOpDecorate || op == SpvOpMemberDecorate) {
        uint32_t target = (op == SpvOpDecorate) ? inst->words[1] : inst->words[1];
        for (size_t i = 0; i < c->nskipped; i++)
            if (c->skipped[i] == target) return SPV_SUCCESS;
    }

    /* non-aggregate types: reuse the module's declaration instead of emitting
     * a duplicate (SPIR-V forbids those and SPIRV-Cross's parser is not
     * required to survive them) */
    if (op >= SpvOpTypeVoid && op <= SpvOpTypeFunction) {
        TyKey key;
        if (ty_key(c, op, inst->words, inst->num_words, &key)) {
            for (size_t i = 0; i < c->ntab; i++) {
                if (c->tab[i].op == key.op && c->tab[i].a == key.a &&
                    c->tab[i].b == key.b && c->tab[i].hash == key.hash) {
                    if (c->nremap < 256) {
                        c->remap_from[c->nremap] = inst->result_id;
                        c->remap_to[c->nremap] = c->tab[i].id;
                        c->nremap++;
                    }
                    return SPV_SUCCESS;   /* deduped onto the module's type */
                }
            }
            /* New type the module does not declare (float in a double-only
             * module, say).  It is emitted below with its id shifted, so
             * register that mapping: derived keys (vectors, pointers,
             * functions) hash the base type through remap_of, and leaving the
             * raw library id there let it collide with an unrelated module id
             * (observed: vec2 keyed onto the module's v2ulong). */
            if (c->nremap < 256) {
                c->remap_from[c->nremap] = inst->result_id;
                c->remap_to[c->nremap] = inst->result_id + c->shift;
                c->nremap++;
            }
        }
    }

    size_t start = c->target->n;
    for (uint32_t i = 0; i < inst->num_words; i++) bpush(c->target, inst->words[i]);
    for (uint32_t i = 0; i < inst->num_operands; i++) {
        const spv_parsed_operand_t *o = &inst->operands[i];
        if (!is_id_operand(o->type)) continue;
        uint32_t v = inst->words[o->offset];
        uint32_t nv = v + c->shift;
        for (size_t j = 0; j < c->nremap; j++)
            if (c->remap_from[j] == v) nv = c->remap_to[j];
        c->target->w[start + o->offset] = nv;
    }
    return SPV_SUCCESS;
}

static void merge_lib(F64 *f)
{
    spv_context ctx = spvContextCreate(SPV_ENV_UNIVERSAL_1_0);
    spv_diagnostic diag = NULL;
    MergeCtx c;
    memset(&c, 0, sizeof(c));
    c.f = f;
    c.shift = f->libshift;
    c.target = &f->libdefs;

    /* the module's own non-aggregate types (original + pre-pass-created) —
     * the library's duplicates dedupe onto them during the merge */
    tykey_scan(f, &c, f->w, f->n, &c.tab, &c.ntab, &c.tabcap);
    tykey_scan(f, &c, f->pre.w, f->pre.n, &c.tab, &c.ntab, &c.tabcap);

    /* pre-scan: globals to drop (the keep-alive main's out variable); their
       decorations come earlier in the stream than the variables themselves */
    for (size_t i = 5; i < f->lib->n; i += f->lib->w[i] >> 16) {
        uint32_t op = f->lib->w[i] & 0xffff, wc = f->lib->w[i] >> 16;
        if (op == SpvOpVariable && wc >= 3 && f->lib->w[i + 3] != SpvStorageClassFunction) {
            if (c.nskipped < 8) c.skipped[c.nskipped++] = f->lib->w[i + 2];
        }
    }

    /* map library imports (GLSL.std.450) onto the module's own import;
       imports the module lacks are added to the header */
    for (size_t i = 5; i < f->lib->n; i += f->lib->w[i] >> 16) {
        uint32_t op = f->lib->w[i] & 0xffff;
        if (op != SpvOpExtInstImport) continue;
        const char *name = (const char *)&f->lib->w[i + 2];
        bool found = false;
        for (size_t j = 0; j < f->nins; j++) {
            if (f->ins[j].op != SpvOpExtInstImport) continue;
            const char *tname = (const char *)(f->w + f->ins[j].off + 2);
            if (!strcmp(name, tname)) {
                if (c.nremap < 8) {
                    c.remap_from[c.nremap] = f->lib->w[i + 1];
                    c.remap_to[c.nremap] = f->w[f->ins[j].off + 1];
                    c.nremap++;
                }
                found = true;
                break;
            }
        }
        if (!found && c.nremap < 8) {
            uint32_t newid = nid(f);
            size_t len = strlen(name) + 1;
            size_t words = (len + 3) / 4;
            bpush(&f->hdradd, ((uint32_t)words + 2) << 16 | SpvOpExtInstImport);
            bpush(&f->hdradd, newid);
            for (size_t k = 0; k < words * 4; k++) {
                uint32_t ch = k < len ? (uint8_t)name[k] : 0;
                if ((k & 3) == 0) bpush(&f->hdradd, ch);
                else f->hdradd.w[f->hdradd.n - 1] |= ch << ((k & 3) * 8);
            }
            c.remap_from[c.nremap] = f->lib->w[i + 1];
            c.remap_to[c.nremap] = newid;
            c.nremap++;
        }
    }

    if (spvBinaryParse(ctx, &c, f->lib->w, f->lib->n, NULL, merge_cb, &diag) != SPV_SUCCESS) {
        if (diag && diag->error) fail(f, "fp64: library parse: %s", diag->error);
        else fail(f, "fp64: library parse failed");
    }
    if (diag) spvDiagnosticDestroy(diag);
    spvContextDestroy(ctx);
}

/* ---------------------------------------------------- emit helpers      -- */
/* Everything below emits into f->body (function bodies). */

static uint32_t xcomp(F64 *f, uint32_t cty, uint32_t v, uint32_t idx)
{
    uint32_t rid = nid(f);
    uint32_t o[4] = {cty, rid, v, idx};
    binst(&f->body, SpvOpCompositeExtract, o, 4);
    return rid;
}

static uint32_t xcomp2(F64 *f, uint32_t cty, uint32_t v, uint32_t i, uint32_t j)
{
    uint32_t rid = nid(f);
    uint32_t o[5] = {cty, rid, v, i, j};
    binst(&f->body, SpvOpCompositeExtract, o, 5);
    return rid;
}

static uint32_t cstruct(F64 *f, uint32_t rty, const uint32_t *comps, uint32_t n)
{
    uint32_t rid = nid(f);
    uint32_t o[8] = {rty, rid, comps[0], comps[1], comps[2], comps[3], 0, 0};
    if (n > 4) n = 4;                        /* SPIR-V vectors/matrices are <= 4 */
    binst(&f->body, SpvOpCompositeConstruct, o, 2 + n);
    return rid;
}

static uint32_t cstruct_n(F64 *f, uint32_t rty, const uint32_t *comps, uint32_t n)
{
    uint32_t rid = nid(f);
    bpush(&f->body, ((uint32_t)n + 3) << 16 | SpvOpCompositeConstruct);
    bpush(&f->body, rty);
    bpush(&f->body, rid);
    for (uint32_t i = 0; i < n; i++) bpush(&f->body, comps[i]);
    return rid;
}

static uint32_t bitcast(F64 *f, uint32_t rty, uint32_t v)
{
    uint32_t rid = nid(f);
    uint32_t o[3] = {rty, rid, v};
    binst(&f->body, SpvOpBitcast, o, 3);
    return rid;
}

static uint32_t fcall(F64 *f, uint32_t rt, const char *fn, const uint32_t *args, uint32_t n)
{
    uint32_t fid = lib_fn(f, fn);
    if (!fid) return 0;
    uint32_t rid = nid(f);
    bpush(&f->body, ((uint32_t)n + 4) << 16 | SpvOpFunctionCall);
    bpush(&f->body, rt);
    bpush(&f->body, rid);
    bpush(&f->body, fid);
    for (uint32_t i = 0; i < n; i++) bpush(&f->body, args[i]);
    return rid;
}

static uint32_t fcall1(F64 *f, const char *fn, uint32_t a)
{
    uint32_t args[1] = {a};
    return fcall(f, f->f64ty, fn, args, 1);
}

static uint32_t fcall2(F64 *f, const char *fn, uint32_t a, uint32_t b)
{
    uint32_t args[2] = {a, b};
    return fcall(f, f->f64ty, fn, args, 2);
}

static uint32_t fcall3(F64 *f, const char *fn, uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t args[3] = {a, b, c};
    return fcall(f, f->f64ty, fn, args, 3);
}

/* the module may have no bool type of its own (e.g. a bare OpExtInst Step or
 * IsNan): emit one in the prelude */
static uint32_t bool_of(F64 *f)
{
    if (!f->boolty) {
        f->boolty = nid(f);
        f->ty[f->boolty].kind = K_BOOL;
        uint32_t o[1] = {f->boolty};
        binst(&f->pre, SpvOpTypeBool, o, 1);
    }
    return f->boolty;
}

/* library predicates return bool, not ulong — the call must be typed bool so
 * SPIRV-Cross sees a real OpSelect condition (otherwise it emits mix() on
 * integers, which MSL rejects) */
static uint32_t fcall_b2(F64 *f, const char *fn, uint32_t a, uint32_t b)
{
    uint32_t args[2] = {a, b};
    return fcall(f, bool_of(f), fn, args, 2);
}

static uint32_t lnot(F64 *f, uint32_t b)
{
    uint32_t rid = nid(f);
    uint32_t o[3] = {bool_of(f), rid, b};
    binst(&f->body, SpvOpLogicalNot, o, 3);
    return rid;
}

static uint32_t selb(F64 *f, uint32_t rty, uint32_t cond, uint32_t a, uint32_t b)
{
    uint32_t rid = nid(f);
    uint32_t o[5] = {rty, rid, cond, a, b};
    binst(&f->body, SpvOpSelect, o, 5);
    return rid;
}

static uint32_t ieql(F64 *f, uint32_t a, uint32_t b)
{
    uint32_t rid = nid(f);
    uint32_t o[4] = {bool_of(f), rid, a, b};
    binst(&f->body, SpvOpIEqual, o, 4);
    return rid;
}

/* during the body pass a value needs lowering when its type still contains the
 * Float64 type (scalar double, dvecN, dmatN, aggregates thereof). Pure-fp32
 * arithmetic in a program that merely happens to contain doubles must NOT be
 * lowered: the soft-float calls take u64 references and MSL would silently
 * truncate float arguments to integers. */
static bool is_f64_val(F64 *f, uint32_t id)
{
    if (id >= f->cap_ids) return false;
    return is64(f, f->vty[id]);
}

/* component-wise scalar binary op over a double shape */
static uint32_t cw2(F64 *f, uint32_t rty, uint32_t a, uint32_t b, const char *fn)
{
    Ty *t = &f->ty[rty];
    if (t->kind == K_VEC || t->kind == K_MAT || t->kind == K_ARR) {
        uint32_t cnt = (t->kind == K_ARR) ? arrlen(f, rty) : t->n;
        if (cnt > 16) { fail(f, "fp64: composite too large"); return 0; }
        uint32_t comps[16];
        for (uint32_t i = 0; i < cnt; i++) {
            uint32_t ai = xcomp(f, t->elem, a, i);
            uint32_t bi = xcomp(f, t->elem, b, i);
            comps[i] = cw2(f, t->elem, ai, bi, fn);
        }
        return cstruct_n(f, rty, comps, cnt);
    }
    return fcall2(f, fn, a, b);
}

static uint32_t cw1(F64 *f, uint32_t rty, uint32_t a, const char *fn)
{
    Ty *t = &f->ty[rty];
    if (t->kind == K_VEC || t->kind == K_MAT || t->kind == K_ARR) {
        uint32_t cnt = (t->kind == K_ARR) ? arrlen(f, rty) : t->n;
        if (cnt > 16) { fail(f, "fp64: composite too large"); return 0; }
        uint32_t comps[16];
        for (uint32_t i = 0; i < cnt; i++)
            comps[i] = cw1(f, t->elem, xcomp(f, t->elem, a, i), fn);
        return cstruct_n(f, rty, comps, cnt);
    }
    return fcall1(f, fn, a);
}

/* comparisons: brty is the (possibly vector) bool result type */
enum { CMP_OE, CMP_ONE, CMP_UE, CMP_UNE, CMP_OL, CMP_OG, CMP_OLE, CMP_OGE,
       CMP_UL, CMP_UG, CMP_ULE, CMP_UGE };

static uint32_t cmp_scalar(F64 *f, int kind, uint32_t a, uint32_t b)
{
    switch (kind) {
    case CMP_OE:  return fcall_b2(f, "mgl_feq64", a, b);
    case CMP_UNE: return fcall_b2(f, "mgl_fneu64", a, b);
    case CMP_UE:  return fcall_b2(f, "mgl_fneu64", a, b);
    case CMP_ONE: return lnot(f, fcall_b2(f, "mgl_feq64", a, b));
    case CMP_OL:  return fcall_b2(f, "mgl_flt64", a, b);
    case CMP_OG:  return fcall_b2(f, "mgl_flt64", b, a);
    case CMP_OLE: return fcall_b2(f, "mgl_fge64", b, a);   /* b >= a  <=>  a <= b */
    case CMP_OGE: return fcall_b2(f, "mgl_fge64", a, b);   /* operands were swapped */
    case CMP_UL:  return lnot(f, fcall_b2(f, "mgl_fge64", a, b));
    case CMP_UG:  return lnot(f, fcall_b2(f, "mgl_fge64", b, a));
    case CMP_ULE: return lnot(f, fcall_b2(f, "mgl_flt64", b, a));
    case CMP_UGE: return lnot(f, fcall_b2(f, "mgl_flt64", a, b));
    }
    return 0;
}

static uint32_t cw_cmp(F64 *f, uint32_t brty, uint32_t a, uint32_t b, uint32_t n, int kind)
{
    if (n <= 1) return cmp_scalar(f, kind, a, b);
    uint32_t comps[4];
    for (uint32_t i = 0; i < n; i++)
        comps[i] = cmp_scalar(f, kind, xcomp(f, f->f64ty, a, i), xcomp(f, f->f64ty, b, i));
    return cstruct(f, brty, comps, n);
}

/* vector length helper for a possibly-vector type */
static uint32_t vlen_of(F64 *f, uint32_t ty)
{
    if (ty < f->cap_ids && f->ty[ty].kind == K_VEC) return f->ty[ty].n;
    return 1;
}

/* ------------------------------------------- interface conversions       -- */

static uint32_t iface_to_reg(F64 *f, uint32_t v, uint32_t ity, uint32_t orig);
static uint32_t reg_to_iface(F64 *f, uint32_t v, uint32_t ity, uint32_t orig);

/* GL assigns ceil(R/2) 128-bit locations to a dvecR column/element, so the
 * flat interface representation of every composite is an array of uint4, one
 * per location (SPIRV-Cross rejects arrays-of-arrays in stage IO variables).
 * A double j of element i starts at word 2*j of location i*slots + j/2. */
static uint32_t iface_slots(F64 *f, uint32_t ety)
{
    uint32_t n = (f->ty[ety].kind == K_FLOAT) ? 1 : f->ty[ety].n;
    return (n + 1) / 2;
}

/* the u64 value of double (elem, comp) of the flat array v */
static uint32_t iface_word_to_u64(F64 *f, uint32_t v, uint32_t elem, uint32_t comp)
{
    uint32_t u4 = xcomp(f, uint4_of(f), v, elem);
    uint32_t lo = xcomp(f, u32_for_pre(f), u4, comp);
    uint32_t hi = xcomp(f, u32_for_pre(f), u4, comp + 1);
    uint32_t u2[2] = {lo, hi};
    return bitcast(f, f->f64ty, cstruct(f, uint2_of(f), u2, 2));
}

/* store the two doubles dlow/dhigh as the two words of flat element elem.
 * Build the uint4 from the two u32-pair BITCASTS directly: re-extracting the
 * four words as scalars lets SPIRV-Cross fold the construct into
 * uint4(uint2(ulong), ...) - a converting splat that truncates to zero. */
static void iface_flat_set(F64 *f, uint32_t elems[], uint32_t elem,
                           uint32_t dlow, uint32_t dhigh)
{
    uint32_t o[2] = { bitcast(f, uint2_of(f), dlow), bitcast(f, uint2_of(f), dhigh) };
    elems[elem] = cstruct(f, uint4_of(f), o, 2);
}

static uint32_t iface_pair_to_reg(F64 *f, uint32_t v, uint32_t ity, uint32_t orig)
{
    uint32_t n = f->ty[orig].n;
    if (n == 1) return bitcast(f, f->f64ty, v);
    if (n == 2) return bitcast(f, ulong2_of(f), v);
    /* n >= 3: v is array[2] of uint4 */
    uint32_t e[4];
    uint32_t b0 = bitcast(f, ulong2_of(f), xcomp(f, uint4_of(f), v, 0));
    e[0] = xcomp(f, f->f64ty, b0, 0);
    e[1] = xcomp(f, f->f64ty, b0, 1);
    uint32_t b1 = bitcast(f, ulong2_of(f), xcomp(f, uint4_of(f), v, 1));
    e[2] = xcomp(f, f->f64ty, b1, 0);
    e[3] = (n == 4) ? xcomp(f, f->f64ty, b1, 1) : k_u64(f, 0);
    return cstruct(f, orig, e, n);
}

static uint32_t reg_to_iface_pair(F64 *f, uint32_t v, uint32_t ity, uint32_t orig)
{
    uint32_t n = f->ty[orig].n;
    if (n == 1) return bitcast(f, uint2_of(f), v);
    if (n == 2) return bitcast(f, uint4_of(f), v);
    uint32_t e0 = xcomp(f, f->f64ty, v, 0);
    uint32_t e1 = xcomp(f, f->f64ty, v, 1);
    uint32_t e2 = xcomp(f, f->f64ty, v, 2);
    uint32_t e3 = (n == 4) ? xcomp(f, f->f64ty, v, 3) : k_u64(f, 0);
    uint32_t b0 = cstruct(f, ulong2_of(f), (uint32_t[]){e0, e1}, 2);
    uint32_t b1 = cstruct(f, ulong2_of(f), (uint32_t[]){e2, e3}, 2);
    uint32_t p0 = bitcast(f, uint4_of(f), b0);
    uint32_t p1 = bitcast(f, uint4_of(f), b1);
    return cstruct(f, ity, (uint32_t[]){p0, p1}, 2);
}

static uint32_t iface_to_reg(F64 *f, uint32_t v, uint32_t ity, uint32_t orig)
{
    Ty *t = &f->ty[orig];
    switch (t->kind) {
    case K_FLOAT:
    case K_VEC:
        return iface_pair_to_reg(f, v, ity, orig);
    case K_MAT:
    case K_ARR: {
        /* the flat array: cnt columns/elements, each slots uint4s.  Composite
         * elements flatten to raw words (double j of element i lives in
         * location i*slots + j/2, word j%2 of it). */
        uint32_t cnt = (t->kind == K_ARR) ? arrlen(f, orig) : t->n;
        uint32_t ety = t->elem;
        bool wrapped = false;
        uint32_t welem = 0;
        if (t->kind == K_ARR && f->ty[ity].kind == K_ARR &&
            f->ty[f->ty[ity].elem].kind == K_STRUCT) {
            /* wrapper-struct elements (dvec3/dvec4/dmatN): unwrap each */
            wrapped = true;
            welem = f->ty[ity].elem;
        }
        if (t->kind == K_ARR && f->ty[ity].kind == K_STRUCT) {
            /* the whole array is ONE slot struct (see iface_ty): extract each
             * element's member slice, convert it, rebuild the array value */
            uint32_t es, elemstruct = 0;
            if (f->ty[ety].kind == K_MAT) {
                uint32_t rows = veclen(f, f->ty[ety].elem);

                es = f->ty[ety].n * ((rows + 1) / 2);
                elemstruct = iface_slot_struct(f, es, rows);
            } else {
                uint32_t n = (f->ty[ety].kind == K_FLOAT) ? 1 : f->ty[ety].n;

                es = (n + 1) / 2;
                elemstruct = (es == 1) ? ((n == 1) ? uint2_of(f) : uint4_of(f))
                                       : iface_slot_struct(f, es, n);
            }
            {
                uint32_t mty = struct_member(f, ity, 0);
                uint32_t comps[16];

                for (uint32_t i = 0; i < cnt && i < 16; i++) {
                    if (es == 1)
                        comps[i] = iface_pair_to_reg(f, xcomp(f, mty, v, i), elemstruct, ety);
                    else
                    {
                        uint32_t mids[32];
                        uint32_t sv;

                        for (uint32_t q = 0; q < es && q < 32; q++)
                            mids[q] = xcomp(f, mty, v, i * es + q);
                        sv = cstruct_n(f, elemstruct, mids, es);
                        comps[i] = (f->ty[ety].kind == K_MAT)
                                     ? iface_to_reg(f, sv, elemstruct, ety)
                                     : iface_pair_to_reg(f, sv, elemstruct, ety);
                    }
                }
                return cstruct_n(f, orig, comps, cnt > 16 ? 16 : cnt);
            }
        }
        if (t->kind == K_ARR && f->ty[ity].kind == K_ARR &&
            f->ty[ety].kind == K_MAT && is_f64_scalar_or_vec(f, f->ty[ety].elem)) {
            /* the whole array is ONE flat slot array (see iface_ty): the same
             * per-element slice rule, element i owning slots
             * [i*es, i*es + es) of it */
            uint32_t rows = veclen(f, f->ty[ety].elem);
            uint32_t es = f->ty[ety].n * ((rows + 1) / 2);
            uint32_t elemstruct = iface_slot_struct(f, es, rows);
            uint32_t comps[16];

            for (uint32_t i = 0; i < cnt && i < 16; i++)
            {
                uint32_t mids[32];
                uint32_t sv;

                for (uint32_t q = 0; q < es && q < 32; q++)
                    mids[q] = xcomp(f, uint4_of(f), v, i * es + q);
                sv = cstruct_n(f, elemstruct, mids, es);
                comps[i] = iface_to_reg(f, sv, elemstruct, ety);
            }
            return cstruct_n(f, orig, comps, cnt > 16 ? 16 : cnt);
        }
        if (wrapped) {
            uint32_t comps[16];
            for (uint32_t i = 0; i < cnt && i < 16; i++) {
                uint32_t w = xcomp(f, welem, v, i);
                uint32_t flat = smember(f, welem, 0);
                uint32_t arr = xcomp(f, flat, w, 0);
                comps[i] = (f->ty[ety].kind == K_MAT)
                             ? iface_to_reg(f, arr, flat, ety)
                             : iface_pair_to_reg(f, arr, flat, ety);
            }
            return cstruct_n(f, orig, comps, cnt > 16 ? 16 : cnt);
        }
        uint32_t n = (f->ty[ety].kind == K_FLOAT) ? 1 : f->ty[ety].n;
        uint32_t slots = (n + 1) / 2;
        uint32_t comps[16];
        for (uint32_t i = 0; i < cnt && i < 16; i++) {
            uint32_t e[4];
            for (uint32_t j = 0; j < n; j++)
                e[j] = iface_word_to_u64(f, v, i * slots + j / 2, (j & 1) * 2);
            comps[i] = cstruct_n(f, ety, e, n);
        }
        return cstruct_n(f, orig, comps, cnt);
    }
    case K_STRUCT: {
        uint32_t nm = struct_nmembers(f, orig);
        uint32_t comps[64];
        for (uint32_t i = 0; i < nm && i < 64; i++) {
            uint32_t im = smember(f, ity, i);      /* the u32-slot member type */
            uint32_t om = struct_member(f, orig, i);
            comps[i] = iface_to_reg(f, xcomp(f, im, v, i), im, om);
        }
        return cstruct_n(f, orig, comps, nm > 64 ? 64 : nm);
    }
    default:
        fail(f, "fp64: bad interface type");
        return v;
    }
}

static uint32_t reg_to_iface(F64 *f, uint32_t v, uint32_t ity, uint32_t orig)
{
    Ty *t = &f->ty[orig];
    switch (t->kind) {
    case K_FLOAT:
    case K_VEC:
        return reg_to_iface_pair(f, v, ity, orig);
    case K_MAT:
    case K_ARR: {
        uint32_t cnt = (t->kind == K_ARR) ? arrlen(f, orig) : t->n;
        uint32_t ety = t->elem;
        bool wrapped = false;
        uint32_t welem = 0;
        if (t->kind == K_ARR && f->ty[ity].kind == K_ARR &&
            f->ty[f->ty[ity].elem].kind == K_STRUCT) {
            wrapped = true;
            welem = f->ty[ity].elem;
        }
        if (wrapped) {
            uint32_t comps[16];
            for (uint32_t i = 0; i < cnt && i < 16; i++) {
                uint32_t flat = smember(f, welem, 0);
                uint32_t col = xcomp(f, ety, v, i);
                uint32_t arr = (f->ty[ety].kind == K_MAT)
                                 ? reg_to_iface(f, col, flat, ety)
                                 : reg_to_iface_pair(f, col, flat, ety);
                comps[i] = cstruct(f, welem, &arr, 1);
            }
            return cstruct_n(f, ity, comps, cnt > 16 ? 16 : cnt);
        }
        if (t->kind == K_ARR && f->ty[ity].kind == K_ARR &&
            f->ty[ety].kind == K_MAT && is_f64_scalar_or_vec(f, f->ty[ety].elem)) {
            /* mirror of the load side in iface_to_reg: element i's slot struct
             * becomes slots [i*es, i*es+es) of the one flat array */
            uint32_t rows = veclen(f, f->ty[ety].elem);
            uint32_t es = f->ty[ety].n * ((rows + 1) / 2);
            uint32_t elemstruct = iface_slot_struct(f, es, rows);
            uint32_t total = cnt * es;
            uint32_t elems[64];

            for (uint32_t i = 0; i < cnt && i < 16; i++)
            {
                uint32_t m = reg_to_iface(f, xcomp(f, ety, v, i), elemstruct, ety);

                for (uint32_t q = 0; q < es && q < 32; q++)
                    if (i * es + q < 64)
                        elems[i * es + q] = xcomp(f, uint4_of(f), m, q);
            }
            return cstruct_n(f, ity, elems, total > 64 ? 64 : total);
        }
        uint32_t n = (f->ty[ety].kind == K_FLOAT) ? 1 : f->ty[ety].n;
        uint32_t slots = (n + 1) / 2;
        uint32_t total = cnt * slots;
        uint32_t elems[64];
        for (uint32_t i = 0; i < cnt && i < 16; i++) {
            uint32_t col = xcomp(f, ety, v, i);   /* ety: the column/element type */
            for (uint32_t j = 0; j < n; j += 2) {
                /* scalar components: the u64 type, not the column type */
                uint32_t d0 = xcomp(f, f->f64ty, col, j);
                uint32_t d1 = (j + 1 < n) ? xcomp(f, f->f64ty, col, j + 1) : k_u64(f, 0);
                iface_flat_set(f, elems, i * slots + j / 2, d0, d1);
            }
        }
        return cstruct_n(f, ity, elems, total > 64 ? 64 : total);
    }
    case K_STRUCT: {
        uint32_t nm = struct_nmembers(f, orig);
        uint32_t comps[64];
        for (uint32_t i = 0; i < nm && i < 64; i++) {
            uint32_t im = smember(f, ity, i);
            uint32_t om = struct_member(f, orig, i);
            comps[i] = reg_to_iface(f, xcomp(f, im, v, i), im, om);
        }
        return cstruct_n(f, ity, comps, nm > 64 ? 64 : nm);
    }
    default:
        fail(f, "fp64: bad interface type");
        return v;
    }
}

/* --------------------------------------------- row-major conversions     -- */

static uint32_t row_to_reg(F64 *f, uint32_t v, uint32_t rty, uint32_t orig)
{
    uint32_t C = f->ty[orig].n;                       /* columns */
    uint32_t R = veclen(f, f->ty[orig].elem);         /* rows */
    uint32_t cols[4];
    for (uint32_t c = 0; c < C; c++) {
        uint32_t col[4];
        for (uint32_t r = 0; r < R; r++)
            col[r] = xcomp2(f, f->f64ty, v, r, c);    /* rows are contiguous */
        cols[c] = cstruct(f, f->ty[orig].elem, col, R);
    }
    return cstruct(f, orig, cols, C);
}

static uint32_t reg_to_row(F64 *f, uint32_t v, uint32_t rty, uint32_t orig)
{
    uint32_t C = f->ty[orig].n;
    uint32_t R = veclen(f, f->ty[orig].elem);
    uint32_t rows[4];
    for (uint32_t r = 0; r < R; r++) {
        uint32_t row[4];
        for (uint32_t c = 0; c < C; c++)
            row[c] = xcomp2(f, f->f64ty, v, c, r);
        rows[r] = cstruct(f, f->ty[rty].elem, row, C);
    }
    return cstruct(f, rty, rows, R);
}


/* ---------------------------------------------------- matrix helpers    -- */

static uint32_t fsub2(F64 *f, uint32_t a, uint32_t b) { return fcall2(f, "mgl_fsub64", a, b); }
static uint32_t fmul2(F64 *f, uint32_t a, uint32_t b) { return fcall2(f, "mgl_fmul64", a, b); }
static uint32_t fadd2(F64 *f, uint32_t a, uint32_t b) { return fcall2(f, "mgl_fadd64", a, b); }

static uint32_t dot_val(F64 *f, uint32_t a, uint32_t b, uint32_t n)
{
    uint32_t acc = fmul2(f, xcomp(f, f->f64ty, a, 0), xcomp(f, f->f64ty, b, 0));
    for (uint32_t i = 1; i < n; i++)
        acc = fadd2(f, acc, fmul2(f, xcomp(f, f->f64ty, a, i), xcomp(f, f->f64ty, b, i)));
    return acc;
}

static uint32_t length_val(F64 *f, uint32_t x, uint32_t n)
{
    return fcall1(f, "mgl_fsqrt64", dot_val(f, x, x, n));
}

static uint32_t cw3(F64 *f, uint32_t rty, uint32_t a, uint32_t b, uint32_t c, const char *fn)
{
    Ty *t = &f->ty[rty];
    if (t->kind == K_VEC || t->kind == K_MAT || t->kind == K_ARR) {
        uint32_t cnt = (t->kind == K_ARR) ? arrlen(f, rty) : t->n;
        uint32_t comps[16];
        for (uint32_t i = 0; i < cnt && i < 16; i++)
            comps[i] = cw3(f, t->elem, xcomp(f, t->elem, a, i), xcomp(f, t->elem, b, i),
                           xcomp(f, t->elem, c, i), fn);
        return cstruct_n(f, rty, comps, cnt);
    }
    uint32_t args[3] = {a, b, c};
    return fcall(f, f->f64ty, fn, args, 3);
}

/* scalar component of a (possibly vector) double value */
static uint32_t dcomp(F64 *f, uint32_t v, uint32_t i)
{
    return xcomp(f, f->f64ty, v, i);
}

static uint32_t det_of(F64 *f, uint32_t m, uint32_t n);   /* m is a value of matrix type */

static uint32_t det2(F64 *f, uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    return fsub2(f, fmul2(f, a, d), fmul2(f, b, c));
}

static uint32_t det_of(F64 *f, uint32_t m, uint32_t n)
{
    if (n == 2) {
        uint32_t a = xcomp2(f, f->f64ty, m, 0, 0), b = xcomp2(f, f->f64ty, m, 0, 1);
        uint32_t c = xcomp2(f, f->f64ty, m, 1, 0), d = xcomp2(f, f->f64ty, m, 1, 1);
        return det2(f, a, b, c, d);
    }
    if (n == 3) {
        uint32_t e[3][3];
        for (uint32_t i = 0; i < 3; i++)
            for (uint32_t j = 0; j < 3; j++) e[i][j] = xcomp2(f, f->f64ty, m, i, j);
        return fadd2(f, fsub2(f, fmul2(f, e[0][0], det2(f, e[1][1], e[1][2], e[2][1], e[2][2])),
                              fmul2(f, e[0][1], det2(f, e[1][0], e[1][2], e[2][0], e[2][2]))),
                     fmul2(f, e[0][2], det2(f, e[1][0], e[1][1], e[2][0], e[2][1])));
    }
    if (n == 4) {
        uint32_t e[4][4];
        for (uint32_t i = 0; i < 4; i++)
            for (uint32_t j = 0; j < 4; j++) e[i][j] = xcomp2(f, f->f64ty, m, i, j);
        uint32_t t[4];
        for (uint32_t j = 0; j < 4; j++) {
            uint32_t s[3][3];
            for (uint32_t i = 1; i < 4; i++) {
                uint32_t k = 0;
                for (uint32_t jj = 0; jj < 4; jj++) {
                    if (jj == j) continue;
                    s[i - 1][k++] = e[i][jj];
                }
            }
            t[j] = fmul2(f, e[0][j], fadd2(f, fsub2(f,
                     fmul2(f, s[0][0], det2(f, s[1][1], s[1][2], s[2][1], s[2][2])),
                     fmul2(f, s[0][1], det2(f, s[1][0], s[1][2], s[2][0], s[2][2]))),
                     fmul2(f, s[0][2], det2(f, s[1][0], s[1][1], s[2][0], s[2][1]))));
        }
        uint32_t r = fsub2(f, t[0], t[1]);
        r = fadd2(f, r, t[2]);
        r = fsub2(f, r, t[3]);
        return r;
    }
    fail(f, "fp64: determinant of %u x %u matrix", n, n);
    return 0;
}

static uint32_t det_of3(F64 *f, uint32_t s[3][3]);

static uint32_t matrix_inverse(F64 *f, uint32_t rty, uint32_t m, uint32_t n)
{
    uint32_t e[4][4];
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < n; j++) e[i][j] = xcomp2(f, f->f64ty, m, i, j);
    uint32_t d = det_of(f, m, n);
    uint32_t cols[4];
    /* inv[j][i] = cofactor(i, j) / det */
    for (uint32_t j = 0; j < n; j++) {
        uint32_t col[4];
        for (uint32_t i = 0; i < n; i++) {
            uint32_t s[3][3];
            uint32_t si = 0;
            for (uint32_t ii = 0; ii < n; ii++) {
                if (ii == i) continue;
                uint32_t sj = 0;
                for (uint32_t jj = 0; jj < n; jj++) {
                    if (jj == j) continue;
                    s[si][sj++] = e[ii][jj];
                }
                si++;
            }
            uint32_t minor;
            if (n == 2) minor = s[0][0];
            else if (n == 3) minor = det2(f, s[0][0], s[0][1], s[1][0], s[1][1]);
            else minor = det_of3(f, s);
            uint32_t cof = ((i + j) & 1) ? fcall1(f, "mgl_fneg64", minor) : minor;
            col[i] = fcall2(f, "mgl_fdiv64", cof, d);
        }
        cols[j] = cstruct(f, f->ty[rty].elem, col, n);
    }
    return cstruct(f, rty, cols, n);
}

static uint32_t det_of3(F64 *f, uint32_t s[3][3])
{
    return fadd2(f, fsub2(f,
             fmul2(f, s[0][0], det2(f, s[1][1], s[1][2], s[2][1], s[2][2])),
             fmul2(f, s[0][1], det2(f, s[1][0], s[1][2], s[2][0], s[2][2]))),
             fmul2(f, s[0][2], det2(f, s[1][0], s[1][1], s[2][0], s[2][1])));
}

/* ---------------------------------------------------- conversions       -- */

static uint32_t conv_to_f64(F64 *f, uint32_t rt, uint32_t x, uint32_t xt)
{
    /* x: f32 scalar/vector (or i32/u32/i64/u64), rt: double scalar/vector */
    Ty *st = &f->ty[xt];
    uint32_t n = (st->kind == K_VEC) ? st->n : 1;
    uint32_t ety = st->kind == K_VEC ? st->elem : xt;
    uint32_t width = f->ty[ety].width;
    int sgn = f->ty[ety].sgn;
    const char *fn = NULL;
    if (f->ty[ety].kind == K_FLOAT) {
        fn = (width == 32) ? "mgl_fp32_to_fp64" : NULL;
        if (!fn) { fail(f, "fp64: only fp32 <-> fp64 conversion is supported"); return 0; }
    } else {
        if (width == 32) fn = sgn ? "mgl_int_to_fp64" : "mgl_uint_to_fp64";
        else if (width == 64) fn = sgn ? "mgl_int64_to_fp64" : "mgl_uint64_to_fp64";
        if (!fn) { fail(f, "fp64: unsupported integer width %u to double", width); return 0; }
    }
    if (n == 1) return fcall1(f, fn, x);
    uint32_t comps[4];
    for (uint32_t i = 0; i < n; i++) comps[i] = fcall1(f, fn, xcomp(f, ety, x, i));
    return cstruct(f, rt, comps, n);
}

static uint32_t conv_from_f64(F64 *f, uint32_t rt, uint32_t x, uint32_t xt)
{
    /* x: double scalar/vector, rt: f32 or int scalar/vector */
    Ty *st = &f->ty[xt];
    uint32_t n = (st->kind == K_VEC) ? st->n : 1;
    Ty *rtt = &f->ty[rt];
    uint32_t rety = rtt->kind == K_VEC ? rtt->elem : rt;
    uint32_t width = f->ty[rety].width;
    int sgn = f->ty[rety].sgn;
    const char *fn;
    if (f->ty[rety].kind == K_FLOAT) {
        if (width != 32) { fail(f, "fp64: only fp32 <-> fp64 conversion is supported"); return 0; }
        fn = "mgl_fp64_to_fp32";
    } else if (width == 32) fn = sgn ? "mgl_fp64_to_int" : "mgl_fp64_to_uint";
    else if (width == 64) fn = sgn ? "mgl_fp64_to_int64" : "mgl_fp64_to_uint64";
    else { fail(f, "fp64: unsupported conversion from double"); return 0; }

    if (n == 1) {
        /* mgl_fp64_to_fp32 returns float; the int64 variants return (u)i64, narrow ints i32 */
        uint32_t callty = (f->ty[rety].kind == K_FLOAT || width == 64) ? rety : f->i32ty;
        uint32_t c = fcall(f, callty, fn, &x, 1);
        if (f->ty[rety].kind != K_FLOAT && width != 32 && width != 64) return c;
        if (f->ty[rety].kind != K_FLOAT && width < 32) {
            uint32_t rid = nid(f);
            uint32_t o[3] = {rety, rid, c};
            binst(&f->body, sgn ? SpvOpSConvert : SpvOpUConvert, o, 3);
            return rid;
        }
        if (f->ty[rety].kind != K_FLOAT && width == 32) return c; /* i32 result */
        return c;                                          /* f32 result straight from the call */
    }
    uint32_t comps[4];
    for (uint32_t i = 0; i < n; i++) {
        uint32_t xi = xcomp(f, f->f64ty, x, i);
        uint32_t callty = (f->ty[rety].kind == K_FLOAT || width == 64) ? rety : f->i32ty;
        uint32_t c = fcall(f, callty, fn, &xi, 1);
        if (f->ty[rety].kind != K_FLOAT && width < 32) {
            uint32_t rid = nid(f);
            uint32_t o[3] = {rety, rid, c};
            binst(&f->body, sgn ? SpvOpSConvert : SpvOpUConvert, o, 3);
            c = rid;
        }
        comps[i] = c;
    }
    return cstruct(f, rt, comps, n);
}

/* widen narrow ints to i32/u32 before converting to double */
static uint32_t widen_int(F64 *f, uint32_t x, uint32_t ety)
{
    uint32_t width = f->ty[ety].width;
    if (width >= 32) return x;
    int sgn = f->ty[ety].sgn;
    uint32_t rid = nid(f);
    uint32_t o[3] = {sgn ? f->i32ty : f->u32ty, rid, x};
    binst(&f->body, sgn ? SpvOpSConvert : SpvOpUConvert, o, 3);
    return rid;
}

/* -------------------------------------------------- extinst dispatch    -- */

static uint32_t rw_extinst_val(F64 *f, const uint32_t *iw, uint32_t wc)
{
    uint32_t rt = iw[1], rid = iw[2];
    uint32_t num = iw[4];
    uint32_t a = (wc >= 6) ? iw[5] : 0;
    uint32_t b = (wc >= 7) ? iw[6] : 0;
    uint32_t c = (wc >= 8) ? iw[7] : 0;
    uint32_t xt = a ? f->vty[a] : 0;
    uint32_t res = 0;

    /* fp32 GLSL.std.450 math stays native; only double operands lower */
    if (!is64(f, rt) && !is64(f, xt)) return 0;

    switch (num) {
    case GLSLstd450Round:
        res = cw1(f, rt, a, "mgl_fround64"); break;
    case GLSLstd450RoundEven:
        /* round() is half-away-from-zero, roundEven() ties to even */
        res = cw1(f, rt, a, "mgl_froundeven64"); break;
    case GLSLstd450Trunc:
        res = cw1(f, rt, a, "mgl_ftrunc64"); break;
    case GLSLstd450FAbs:
        res = cw1(f, rt, a, "mgl_fabs64"); break;
    case GLSLstd450FSign:
        res = cw1(f, rt, a, "mgl_fsign64"); break;
    case GLSLstd450Floor:
        res = cw1(f, rt, a, "mgl_ffloor64"); break;
    case GLSLstd450Ceil:
        res = cw1(f, rt, a, "mgl_fceil64"); break;
    case GLSLstd450Fract:
        res = cw1(f, rt, a, "mgl_ffract64"); break;
    case GLSLstd450Sqrt:
        res = cw1(f, rt, a, "mgl_fsqrt64"); break;
    case GLSLstd450InverseSqrt:
        res = cw1(f, rt, a, "mgl_frsq64"); break;
    case GLSLstd450FMin:
    case GLSLstd450NMin:
        res = cw2(f, rt, a, b, "mgl_fmin64"); break;
    case GLSLstd450FMax:
    case GLSLstd450NMax:
        res = cw2(f, rt, a, b, "mgl_fmax64"); break;
    case GLSLstd450FClamp:
    case GLSLstd450NClamp:
        res = cw2(f, rt, cw2(f, rt, a, b, "mgl_fmax64"), c, "mgl_fmin64"); break;
    case GLSLstd450Fma:
        res = cw3(f, rt, a, b, c, "mgl_ffma64"); break;
    case GLSLstd450FMix: {
        /* operands (x, y, t): mix = fma(t, y - x, x) */
        uint32_t tt = f->vty[c];
        bool tscalar = f->ty[tt].kind == K_FLOAT && f->ty[rt].kind != K_FLOAT;
        if (f->ty[rt].kind == K_FLOAT) {
            res = fcall3(f, "mgl_ffma64", c, fsub2(f, b, a), a);
        } else if (tscalar) {
            uint32_t comps[4];
            uint32_t n = f->ty[rt].n;
            for (uint32_t i = 0; i < n; i++)
                comps[i] = fcall3(f, "mgl_ffma64", c, fsub2(f, dcomp(f, b, i), dcomp(f, a, i)), dcomp(f, a, i));
            res = cstruct(f, rt, comps, n);
        } else {
            res = cw3(f, rt, c, cw2(f, rt, b, a, "mgl_fsub64"), a, "mgl_ffma64");
        }
        break;
    }
    case GLSLstd450Step: {
        uint32_t n = vlen_of(f, rt);
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++) {
            /* GLSL step(edge, x) = x < edge ? 0.0 : 1.0  (x == edge gives 1.0;
             * the old `edge < x ? 1.0 : 0.0` returned 0 at the equal point) */
            uint32_t cond = fcall_b2(f, "mgl_flt64", dcomp(f, b, i), dcomp(f, a, i));
            comps[i] = selb(f, f->f64ty, cond, k_dbl(f, 0.0), k_dbl(f, 1.0));
        }
        res = (n > 1) ? cstruct(f, rt, comps, n) : comps[0];
        break;
    }
    case GLSLstd450SmoothStep: {
        uint32_t n = vlen_of(f, rt);
        uint32_t k2 = k_dbl(f, 2.0), k3 = k_dbl(f, 3.0);
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++) {
            uint32_t e0 = dcomp(f, a, i), e1 = dcomp(f, b, i), x = dcomp(f, c, i);
            uint32_t d = fsub2(f, e1, e0);
            uint32_t t = fcall2(f, "mgl_fdiv64", fsub2(f, x, e0), d);
            t = fcall2(f, "mgl_fmin64", fcall2(f, "mgl_fmax64", t, k_dbl(f, 0.0)), k_dbl(f, 1.0));
            comps[i] = fmul2(f, fmul2(f, t, t), fsub2(f, k3, fmul2(f, k2, t)));
        }
        res = (n > 1) ? cstruct(f, rt, comps, n) : comps[0];
        break;
    }
    case GLSLstd450Modf: {
        uint32_t t = cw1(f, rt, a, "mgl_ftrunc64");
        binst2(&f->body, SpvOpStore, b, t);
        res = cw2(f, rt, a, t, "mgl_fsub64");
        break;
    }
    case GLSLstd450ModfStruct: {
        /* struct { frac = x - trunc(x), integral = trunc(x) }:  BOTH members
         * have x's type (SPIR-V GLSL.std.450 spec; the old code narrowed
         * member 1 to int and Metal rejected the construct).  glslang emits
         * this for modf(x, out ip); the out param is stored from member 1. */
        uint32_t n = vlen_of(f, xt);
        uint32_t tr = cw1(f, xt, a, "mgl_ftrunc64");
        uint32_t fr = cw2(f, xt, a, tr, "mgl_fsub64");
        uint32_t comps[2] = { fr, tr };
        uint32_t o[5] = {rt, rid, comps[0], comps[1], 0};
        binst(&f->body, SpvOpCompositeConstruct, o, 4);
        return 0;
    }
    case GLSLstd450FrexpStruct: {
        /* struct { mantissa in [0.5, 1.0) (x's type), exponent (int/ivecN) }:
         * glslang emits this for frexp(x, out e); op 52 was rejected before. */
        uint32_t n = vlen_of(f, xt);
        uint32_t comps[2];
        comps[0] = cw1(f, xt, a, "mgl_frexp64_sig");
        if (n > 1) {
            uint32_t exps[4];
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ai = dcomp(f, a, i);
                exps[i] = fcall(f, f->i32ty, "mgl_frexp64_exp", &ai, 1);
            }
            uint32_t iv = module_vec(f, f->i32ty, n);
            if (!iv) iv = mkvec(f, f->i32ty, n, false);
            comps[1] = cstruct(f, iv, exps, n);
        } else {
            comps[1] = fcall(f, f->i32ty, "mgl_frexp64_exp", &a, 1);
        }
        uint32_t o[5] = {rt, rid, comps[0], comps[1], 0};
        binst(&f->body, SpvOpCompositeConstruct, o, 4);
        return 0;
    }
    case GLSLstd450Frexp: {
        uint32_t n = vlen_of(f, xt);
        if (n == 1) {
            uint32_t e = fcall(f, f->i32ty, "mgl_frexp64_exp", &a, 1);
            binst2(&f->body, SpvOpStore, b, e);
            res = fcall1(f, "mgl_frexp64_sig", a);
        } else {
            uint32_t sigs[4], exps[4];
            uint32_t evec_ty = f->ty[f->vty[b]].kind == K_PTR ? f->ty[f->vty[b]].elem : f->i32ty;
            for (uint32_t i = 0; i < n; i++) {
                uint32_t xi = dcomp(f, a, i);
                sigs[i] = fcall1(f, "mgl_frexp64_sig", xi);
                exps[i] = fcall(f, f->i32ty, "mgl_frexp64_exp", &xi, 1);
            }
            uint32_t evec = cstruct(f, evec_ty, exps, n);
            binst2(&f->body, SpvOpStore, b, evec);
            res = cstruct(f, rt, sigs, n);
        }
        break;
    }
    case GLSLstd450Ldexp: {
        uint32_t n = vlen_of(f, rt);
        if (n == 1) {
            uint32_t args[2] = {a, b};
            res = fcall(f, f->f64ty, "mgl_fldexp64", args, 2);
        } else {
            uint32_t ety = f->ty[f->vty[b]].kind == K_VEC ? f->ty[f->vty[b]].elem : f->vty[b];
            uint32_t comps[4];
            for (uint32_t i = 0; i < n; i++) {
                uint32_t args[2] = {dcomp(f, a, i), xcomp(f, ety, b, i)};
                comps[i] = fcall(f, f->f64ty, "mgl_fldexp64", args, 2);
            }
            res = cstruct(f, rt, comps, n);
        }
        break;
    }
    case GLSLstd450Length:
        res = length_val(f, a, veclen(f, xt)); break;
    case GLSLstd450Distance:
        res = length_val(f, cw2(f, xt, a, b, "mgl_fsub64"), veclen(f, xt)); break;
    case GLSLstd450Cross: {
        uint32_t ax = dcomp(f, a, 0), ay = dcomp(f, a, 1), az = dcomp(f, a, 2);
        uint32_t bx = dcomp(f, b, 0), by = dcomp(f, b, 1), bz = dcomp(f, b, 2);
        uint32_t comps[3] = {
            fsub2(f, fmul2(f, ay, bz), fmul2(f, az, by)),
            fsub2(f, fmul2(f, az, bx), fmul2(f, ax, bz)),
            fsub2(f, fmul2(f, ax, by), fmul2(f, ay, bx)),
        };
        res = cstruct(f, rt, comps, 3);
        break;
    }
    case GLSLstd450Normalize: {
        uint32_t n = veclen(f, xt);
        uint32_t inv = fcall1(f, "mgl_frcp64", length_val(f, a, n));
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++) comps[i] = fmul2(f, dcomp(f, a, i), inv);
        res = cstruct(f, rt, comps, n);
        break;
    }
    case GLSLstd450FaceForward: {
        uint32_t n = veclen(f, xt);
        /* faceforward(N, I, Nref) = dot(Nref, I) < 0 ? N : -N  (c = Nref, b = I) */
        uint32_t cond = fcall_b2(f, "mgl_flt64", dot_val(f, c, b, n), k_dbl(f, 0.0));
        uint32_t neg = cw1(f, xt, a, "mgl_fneg64");
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++)
            comps[i] = selb(f, f->f64ty, cond, dcomp(f, a, i), dcomp(f, neg, i));
        res = cstruct(f, rt, comps, n);
        break;
    }
    case GLSLstd450Reflect: {
        uint32_t n = veclen(f, xt);
        uint32_t t = fmul2(f, k_dbl(f, 2.0), dot_val(f, b, a, n));
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++)
            comps[i] = fsub2(f, dcomp(f, a, i), fmul2(f, t, dcomp(f, b, i)));
        res = cstruct(f, rt, comps, n);
        break;
    }
    case GLSLstd450Refract: {
        /* operands (I, N, eta): k = 1 - eta^2 (1 - dot(N,I)^2) */
        uint32_t n = veclen(f, xt);
        uint32_t d = dot_val(f, b, a, n);
        if (f->ty[f->vty[c]].kind == K_VEC) {
            fail(f, "fp64: refract with vector eta not supported");
            return 0;
        }
        uint32_t k = fsub2(f, k_dbl(f, 1.0),
                           fmul2(f, c, fmul2(f, c, fsub2(f, k_dbl(f, 1.0), fmul2(f, d, d)))));
        uint32_t cond = fcall_b2(f, "mgl_flt64", k, k_dbl(f, 0.0));
        uint32_t sk = fcall1(f, "mgl_fsqrt64", k);
        uint32_t t = fadd2(f, fmul2(f, c, d), sk);
        uint32_t comps[4];
        for (uint32_t i = 0; i < n; i++)
            comps[i] = selb(f, f->f64ty, cond, k_dbl(f, 0.0),
                            fsub2(f, fmul2(f, c, dcomp(f, a, i)), fmul2(f, t, dcomp(f, b, i))));
        res = cstruct(f, rt, comps, n);
        break;
    }
    case GLSLstd450Determinant:
        res = det_of(f, a, f->ty[xt].n); break;
    case GLSLstd450MatrixInverse:
        res = matrix_inverse(f, rt, a, f->ty[xt].n); break;
    case GLSLstd450PackDouble2x32:
        res = bitcast(f, f->f64ty, a); break;
    case GLSLstd450UnpackDouble2x32:
        res = bitcast(f, rt, a); break;
    default:
        fail(f, "fp64: GLSL.std.450 op %u on doubles is not supported", num);
        return 0;
    }
    return res;
}


/* -------------------------------------------------- access chains       -- */

static bool inst_has_result(uint32_t op)
{
    switch (op) {
    case SpvOpLoad: case SpvOpAccessChain: case SpvOpInBoundsAccessChain: case SpvOpPtrAccessChain:
    case SpvOpFunctionCall: case SpvOpCompositeExtract: case SpvOpCompositeInsert:
    case SpvOpCompositeConstruct: case SpvOpCopyObject: case SpvOpVectorShuffle:
    case SpvOpVectorExtractDynamic: case SpvOpVectorInsertDynamic: case SpvOpPhi:
    case SpvOpSelect: case SpvOpFunctionParameter: case SpvOpExtInst:
    case SpvOpFAdd: case SpvOpFSub: case SpvOpFMul: case SpvOpFDiv: case SpvOpFRem: case SpvOpFMod:
    case SpvOpFNegate: case SpvOpDot: case SpvOpVectorTimesScalar: case SpvOpMatrixTimesScalar:
    case SpvOpVectorTimesMatrix: case SpvOpMatrixTimesVector: case SpvOpMatrixTimesMatrix:
    case SpvOpOuterProduct: case SpvOpTranspose: case SpvOpIsNan: case SpvOpIsInf:
    case SpvOpFConvert: case SpvOpConvertFToS: case SpvOpConvertFToU:
    case SpvOpConvertSToF: case SpvOpConvertUToF: case SpvOpSpecConstantOp:
    case SpvOpFOrdEqual: case SpvOpFUnordEqual: case SpvOpFOrdNotEqual: case SpvOpFUnordNotEqual:
    case SpvOpFOrdLessThan: case SpvOpFUnordLessThan: case SpvOpFOrdGreaterThan: case SpvOpFUnordGreaterThan:
    case SpvOpFOrdLessThanEqual: case SpvOpFUnordLessThanEqual:
    case SpvOpFOrdGreaterThanEqual: case SpvOpFUnordGreaterThanEqual:
    case SpvOpConstant: case SpvOpSpecConstant: case SpvOpUndef: case SpvOpConstantNull:
        return true;
    default:
        return false;
    }
}

/* index*mult, in the index's own integer type (constant-folded when possible);
 * the multiply instruction lands in the body right before the access chain. */
static uint32_t io_scaled_index(F64 *f, uint32_t idx, uint32_t mult)
{
    if (mult <= 1) return idx;
    if (idx >= f->cap_ids) return idx;
    uint32_t ity = f->vty[idx];
    if (!ity) return idx;
    if (f->isconst[idx]) {
        uint32_t id = nid(f);
        f->vty[id] = ity; f->isconst[id] = 1; f->cval[id] = (uint32_t)f->cval[idx] * mult;
        uint32_t o[3] = {ity, id, (uint32_t)f->cval[id]};
        binst(&f->pre, SpvOpConstant, o, 3);
        return id;
    }
    uint32_t k = nid(f);
    f->vty[k] = ity; f->isconst[k] = 1; f->cval[k] = mult;
    uint32_t ok[3] = {ity, k, mult};
    binst(&f->pre, SpvOpConstant, ok, 3);
    uint32_t r = nid(f);
    f->vty[r] = ity;
    uint32_t om[5] = {ity, r, idx, k};
    binst(&f->body, SpvOpIMul, om, 4);
    return r;
}

/* walk the chain, transposing row-major indices and fixing the result type;
   emits the replacement instruction with the original result id */
static void rw_access(F64 *f, size_t j, bool inbounds)
{
    const uint32_t *iw = f->w + f->ins[j].off;
    uint32_t wc = f->ins[j].wc;
    uint32_t rtptr = iw[1], rid = iw[2], base = iw[3];
    uint32_t nidx = wc - 4;

    uint32_t baseptr = f->vty[base];
    if (!baseptr || f->ty[baseptr].kind != K_PTR) { fail(f, "fp64: access chain base is not a pointer"); return; }
    uint32_t storage = f->ty[baseptr].n;
    if (f->attrslot[baseptr]) {
        /* component access into a double attribute (glslang: pos.x as a chain).
         * Attributes are read-only, so the chain's only user is a load; drop
         * the chain here and let the load fetch the slot + extract. */
        if (f->ins[j].op == SpvOpAccessChain && wc == 5 && f->isconst[iw[4]] &&
            f->ty[baseptr].kind == K_PTR && f->ty[f->ty[baseptr].elem].kind == K_VEC &&
            (uint32_t)f->cval[iw[4]] < f->attrn[baseptr]) {
            f->attrcomp[iw[2]] = (base << 4) | ((uint32_t)f->cval[iw[4]] + 1);
            return;
        }
        /* ARRAY-of-double / double-matrix attribute: fold the (element, column,
         * row) constant indexes into ONE slot of the flattened slot array and a
         * pair inside that slot.  The chain's only user is a load (read-only),
         * so the chain is dropped and the load reads slot `slot` (recorded in
         * attrslotidx) + pair `row % 2`. */
        {
            uint32_t geom = f->attrsgeom[baseptr];

            if (geom) {
                uint32_t nelem = geom >> 16, cols = (geom >> 8) & 0xff, rows = geom & 0xff;
                uint32_t orig = f->attrorig[baseptr];
                uint32_t ei = 0, col = 0, row = 0, cur = orig;
                bool good = true;

                for (uint32_t k2 = 4; k2 < wc && good; k2++)
                {
                    if (!f->isconst[iw[k2]]) { good = false; break; }
                    uint32_t c = (uint32_t)f->cval[iw[k2]];

                    if (f->ty[cur].kind == K_ARR) { ei = c; cur = f->ty[cur].elem; }
                    else if (f->ty[cur].kind == K_MAT) { col = c; cur = f->ty[cur].elem; }
                    else if (f->ty[cur].kind == K_VEC) { row = c; cur = f->ty[cur].elem; }
                    else good = false;
                }
                if (good && f->ins[j].op == SpvOpAccessChain && ei < nelem && col < cols &&
                    row < rows && (f->ty[cur].kind == K_FLOAT ||
                                   (f->ty[cur].kind == K_VEC && veclen(f, cur) == 1)))
                {
                    uint32_t spc = attr_slots_per_col(rows);
                    uint32_t slot = ei * cols * spc + col * spc + row / 2;

                    f->attrslotidx[iw[2]] = slot;
                    f->attrcomp[iw[2]] = (base << 4) | ((row % 2) + 1);
                    return;
                }
                /* whole dvec2/dvec3/dvec4 array element (KHR-GL46.vertex_attrib_binding
                 * basic-inputL copies vs_in_attrib[i] whole, one dvec4 per element):
                 * record the element's FIRST slot and no pair — rw_load reads slot and
                 * slot+1 of the flattened slot array and reassembles the vector.  Only
                 * plain vectors land here: a matrix element (K_MAT) is still rejected */
                if (good && f->ins[j].op == SpvOpAccessChain && ei < nelem && col < cols &&
                    row < rows && f->ty[cur].kind == K_VEC &&
                    veclen(f, cur) >= 2 && veclen(f, cur) <= 4 &&
                    rows == veclen(f, cur))
                {
                    uint32_t spc = attr_slots_per_col(rows);
                    uint32_t slot = ei * cols * spc + col * spc;

                    f->attrslotidx[iw[2]] = slot;
                    f->attrcomp[iw[2]] = (base << 4);            /* low nibble 0: no pair */
                    return;
                }
            }
            fail(f, "fp64: element access into a double vertex attribute "
                    "is not supported (copy it to a local first)");
            return;
        }
    }
    if (f->ioiface[baseptr]) {
        /* constant component read of a double stage variable (glslang: v.x as a
         * chain): record it and let the load fetch the whole variable + extract.
         * Component WRITES are not supported (SPIR-V out vars are written whole).
         * An INTERFACE BLOCK base (struct or array-of-struct pointee) is not a
         * "component" here: its single const index selects a MEMBER, so those
         * chains fall through to the generic walk below, which maps the struct
         * members onto the u32-slot representation. */
        uint32_t mpt = f->ty[baseptr].elem;                     /* u32-slot repr */
        uint32_t opt = (mpt && mpt < f->cap_ids && f->iface[mpt]) ? f->iforig[mpt] : mpt;
        bool direct = is_f64_scalar_or_vec(f, opt);
        if (direct) {
            if (f->ins[j].op == SpvOpAccessChain && wc == 5 && f->isconst[iw[4]] &&
                (uint32_t)f->cval[iw[4]] < 4) {
                f->attrcomp[iw[2]] = (base << 4) | ((uint32_t)f->cval[iw[4]] + 1);
                return;
            }
            fail(f, "fp64: element access into a double stage variable "
                    "is not supported (copy it to a local first)");
            return;
        }
    }
    uint32_t cur = f->ty[baseptr].elem;
    bool ifmode = (storage == SpvStorageClassInput || storage == SpvStorageClassOutput) && is64(f, cur);
    uint32_t curi = ifmode ? iface_ty(f, cur) : cur;
    /* For an interface (ioiface) base the pointee is already the u32-slot repr;
     * walk the GL view (curo) alongside it so array/matrix indices can be
     * scaled and component steps folded into a slot + pair selection. */
    uint32_t curo = cur;
    if (f->ioiface[baseptr]) {
        ifmode = true;
        curo = (cur && cur < f->cap_ids && f->iface[cur] && f->iforig[cur]) ? f->iforig[cur] : cur;
        curi = cur;
    }

    uint32_t out[24];
    uint32_t nout = 0;
    bool patched = false;
    uint32_t elemreg = 0;      /* chain lands on one element of an IO array */
    uint32_t pair = 0xffffffff;/* ... on double #`pair` of a uint4 slot */

    for (uint32_t i = 0; i < nidx;) {
        Ty *t = &f->ty[curo];
        if (t->kind == K_STRUCT) {
            uint32_t midx = iw[4 + i];
            if (!midx || midx >= f->cap_ids || !f->isconst[midx]) {
                fail(f, "fp64: struct index in access chain must be constant");
                return;
            }
            uint32_t member = (uint32_t)f->cval[midx];
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            out[nout++] = midx;
            uint32_t mt = struct_member(f, curo, member);
            if (has_mdec(f, curo, member, SpvDecorationRowMajor, NULL) &&
                f->ty[mt].kind == K_MAT && is64(f, mt)) {
                if (i + 1 >= nidx) {
                    /* pointer to the whole (row-major) matrix */
                    curo = f->rowmap[mt] ? f->rowmap[mt] : rowmajor_ty(f, mt);
                    curi = curo;
                    patched = true;
                    i += 1;
                    continue;
                }
                if (i + 2 < nidx) {
                    /* m[c][r] becomes rows[r][c] */
                    if (nout + 2 > 24) { fail(f, "fp64: access chain too long"); return; }
                    out[nout++] = iw[4 + i + 2];
                    out[nout++] = iw[4 + i + 1];
                    i += 3;
                    curo = f->f64ty;
                    curi = curo;
                    patched = true;
                    continue;
                }
                fail(f, "fp64: pointer into a row-major double matrix column is not supported");
                return;
            }
            curo = mt;
            curi = ifmode ? smember(f, curi, member) : curo;
            if (curi != curo) patched = true;
            i += 1;
        } else if (t->kind == K_ARR && ifmode && is_f64_scalar_or_vec(f, t->elem) &&
                   f->ty[t->elem].n >= 3) {
            /* stage-IO array of dvec3/dvec4: ONE slot struct for the whole
             * member (named members); element i is the member slice
             * [i*slots, i*slots+slots).  Literal index only — record it and let
             * the load/store splice the slice. */
            if (nout >= 24 || !f->isconst[iw[4 + i]]) {
                fail(f, "fp64: dynamic index into a dvec3/dvec4 stage-IO array "
                        "is not supported");
                return;
            }
            uint32_t es = (f->ty[t->elem].n + 1) / 2;

            f->elemslice[rid] = 1;
            f->elemidx[rid] = (uint32_t)f->cval[iw[4 + i]];
            f->elemstride[rid] = es;
            f->elemorig[rid] = t->elem;
            curo = t->elem;
            /* curi stays the WHOLE-member struct: the chain lands there and the
             * load/store splices the recorded slice */
            elemreg = curo;
            patched = true;
            i += 1;
        } else if (t->kind == K_ARR && ifmode &&
                   f->ty[t->elem].kind == K_MAT && is_f64_scalar_or_vec(f, f->ty[t->elem].elem)) {
            /* arrays of double matrices: same slice rule (columns of each
             * element are slot-aligned inside the one big struct) */
            if (nout >= 24 || !f->isconst[iw[4 + i]]) {
                fail(f, "fp64: dynamic index into a double-matrix stage-IO array "
                        "is not supported");
                return;
            }
            uint32_t mslots = f->ty[t->elem].n * iface_slots(f, f->ty[t->elem].elem);

            f->elemslice[rid] = 1;
            f->elemidx[rid] = (uint32_t)f->cval[iw[4 + i]];
            f->elemstride[rid] = mslots;
            f->elemorig[rid] = t->elem;
            curo = t->elem;
            /* curi stays the WHOLE-member struct (see above) */
            elemreg = curo;
            patched = true;
            i += 1;
        } else if (t->kind == K_ARR && ifmode && is_f64_scalar_or_vec(f, t->elem)) {
            /* stage-IO array of doubles: elements are uint4 (scalar/dvec2) or
             * one wrapper struct each (dvec3/dvec4); the index stays 1:1 */
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            out[nout++] = iw[4 + i];
            curo = t->elem;
            curi = (f->ty[curi].kind == K_ARR) ? f->ty[curi].elem : uint4_of(f);
            elemreg = curo;
            /* a scalar-double element is the whole uint4 slot: elemreg covers its
             * loads AND stores.  Setting `pair` here would also set attrcomp, which
             * rw_load_store checks first and rejects for stores ("component
             * writes") — that broke `tv[gl_InvocationID] = x;` in a TCS. */
            patched = true;
            i += 1;
        } else if (t->kind == K_ARR && ifmode &&
                   (f->ty[t->elem].kind == K_MAT || f->ty[t->elem].kind == K_STRUCT)) {
            /* arrays of double matrices / of interface blocks: wrapper-struct
             * or mapped-struct elements, index 1:1 */
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            out[nout++] = iw[4 + i];
            curo = t->elem;
            curi = (f->ty[curi].kind == K_ARR) ? f->ty[curi].elem : curi;
            if (f->ty[curo].kind == K_MAT) elemreg = curo;
            if (curi != curo) patched = true;
            i += 1;
        } else if (t->kind == K_ARR) {
            uint32_t lanes = (!ifmode) ? widen_buffer_array(f, curo) : 0;
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            out[nout++] = iw[4 + i];
            curo = t->elem;
            if (lanes) {
                /* widened scalar-double buffer array: the chain now lands on
                 * vec<lanes>(u64); rw_load_store extracts / re-fills lane 0,
                 * keeping the value a plain u64 */
                f->arrlane[rid] = curo;
                curi = curo;
                curo = f->f64ty;
                patched = true;
            } else {
                curi = f->ty[curi].kind == K_ARR ? f->ty[curi].elem : curi;
                if (curi != curo) patched = true;
            }
            i += 1;
        } else if (t->kind == K_MAT && ifmode) {
            /* column of a flat-lying stage-IO matrix: the column spans
             * ceil(R/2) uint4 locations, so the index is scaled */
            uint32_t rows = veclen(f, t->elem);
            uint32_t slots = (rows + 1) / 2;
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            if (i + 1 < nidx && slots > 1) {
                /* m[c][r] of a wide matrix: the value is double r%2 of slot
                 * c*slots + r/2 of the flat slot array (constant indexes only —
                 * the whole column spans several slots and cannot be one chain) */
                uint32_t ci = iw[4 + i], ri = iw[4 + i + 1];

                if (!ci || ci >= f->cap_ids || !ri || ri >= f->cap_ids ||
                    !f->isconst[ci] || !f->isconst[ri] ||
                    (uint32_t)f->cval[ri] >= rows) {
                    fail(f, "fp64: element index into a wide double matrix stage "
                            "variable must be constant");
                    return;
                }
                out[nout++] = k_u32(f, (uint32_t)f->cval[ci] * slots +
                                           (uint32_t)f->cval[ri] / 2);
                pair = (uint32_t)f->cval[ri] % 2;
                curo = f->f64ty;
                curi = uint4_of(f);
                patched = true;
                i += 2;
                if (i < nidx) {
                    fail(f, "fp64: nested access into a double stage component "
                            "is not supported");
                    return;
                }
                continue;
            }
            out[nout++] = io_scaled_index(f, iw[4 + i], slots);
            /* an array-of-matrices slice: the element's slots start at
             * elemidx * elemstride inside the one flat member array */
            if (f->elemslice[rid])
            {
                uint32_t off = k_u32(f, f->elemidx[rid] * f->elemstride[rid]);
                uint32_t sum = nid(f);
                uint32_t o[4] = {u32_for_pre(f), sum, out[nout - 1], off};
                binst(&f->body, SpvOpIAdd, o, 4);
                out[nout - 1] = sum;
            }
            curo = t->elem;                 /* the dvecR column */
            curi = uint4_of(f);
            if (i + 1 >= nidx) {
                if (slots > 1) {
                    fail(f, "fp64: whole-column access into a wide double matrix "
                            "stage variable is not supported");
                    return;
                }
                elemreg = curo;             /* dvec2 column = one uint4 */
            }
            patched = true;
            i += 1;
        } else if (t->kind == K_MAT) {
            /* column-major matrix: [col][row] or [col] */
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            /* narrow matrices (dvec2 columns, one slot each) keep the column
             * index; a matrix-array slice shifts it by the element base */
            if (f->elemslice[rid] && f->elemstride[rid]) {
                uint32_t off = k_u32(f, f->elemidx[rid] * f->elemstride[rid]);
                uint32_t sum = nid(f);
                uint32_t o[4] = {u32_for_pre(f), sum, iw[4 + i], off};
                binst(&f->body, SpvOpIAdd, o, 4);
                out[nout++] = sum;
            } else {
                out[nout++] = iw[4 + i];
            }
            curo = t->elem;
            curi = f->ty[curi].kind == K_ARR ? f->ty[curi].elem : curi;
            if (curi != curo) patched = true;
            i += 1;
            if (i < nidx) {
                if (f->ty[curi].kind == K_VEC || f->ty[curi].kind == K_FLOAT) {
                    fail(f, "fp64: component access into a double matrix stage variable is not supported");
                    return;
                }
                if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
                out[nout++] = iw[4 + i];
                curo = f->f64ty;
                curi = curo;
                patched = true;
                i += 1;
            }
        } else if (t->kind == K_VEC && ifmode && is_f64(f, t->elem)) {
            /* component of a dvecN stage value: fold into slot + pair */
            uint32_t cidx = iw[4 + i];
            if (!cidx || cidx >= f->cap_ids || !f->isconst[cidx] ||
                (uint32_t)f->cval[cidx] >= t->n) {
                fail(f, "fp64: component index into a double stage variable must be constant");
                return;
            }
            uint32_t c = (uint32_t)f->cval[cidx];
            if (t->n <= 2) {
                /* one uint4 holds the whole value (scalar: pair 0) */
                pair = c;
                curi = uint4_of(f);
            } else {
                /* array uint4[2] member/variable or a pair-struct element */
                if (f->ty[curi].kind == K_STRUCT) {
                    if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
                    out[nout++] = k_u32(f, 0);
                    curi = smember(f, curi, 0);
                }
                if (f->ty[curi].kind == K_ARR) {
                    if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
                    /* an element of a dvec3/dvec4 stage-IO array: the flat slot
                     * index is elemidx * slots-per-element + c/2 (the slice the
                     * array step recorded) */
                    uint32_t sl = c / 2;
                    if (f->elemslice[rid]) sl += f->elemidx[rid] * f->elemstride[rid];
                    out[nout++] = k_u32(f, sl);
                    curi = f->ty[curi].elem;
                }
                pair = c % 2;
                curi = uint4_of(f);
            }
            patched = true;
            i += 1;
            if (i < nidx) {
                fail(f, "fp64: nested access into a double stage component is not supported");
                return;
            }
        } else if (t->kind == K_VEC) {
            /* indexing into a vector lvalue (glslang emits constant indices) */
            if (ifmode) {
                fail(f, "fp64: component access into a double stage variable "
                        "is not supported (copy it to a local first)");
                return;
            }
            if (nout >= 24) { fail(f, "fp64: access chain too long"); return; }
            out[nout++] = iw[4 + i];
            cur = t->elem;
            curi = cur;
            i += 1;
        } else {
            fail(f, "fp64: unexpected access chain step (kind %u)", t->kind);
            return;
        }
    }

    if (!patched) {
        /* copy verbatim */
        for (uint32_t i = 0; i < wc; i++) bpush(&f->body, iw[i]);
        return;
    }

    /* the result type may need patching when the storage repr of the final
       type differs from the in-place mapping */
    uint32_t newrt = rtptr;
    if (curi != curo) newrt = ptr_for(f, storage, curi);

    /* element/pair selections land on a ptr-to-uint4 (or to a wrapper struct):
     * rw_load_store converts through the recorded original type */
    if (elemreg) f->elemreg[rid] = elemreg;
    if (pair != 0xffffffff)
        f->attrcomp[rid] = (0 << 4) | (pair + 1);   /* base 0: the chain ptr itself is the slot */

    if (f->ty[curi].kind == K_ARR && f->iface[curi] && !elemreg) {
        /* whole multi-slot member/variable: NO chain instruction — record the
         * full index walk; rw_load_store emits one complete access chain per
         * slot (see the expc comment in the F64 struct) */
        f->vty[rid] = newrt;
        f->expc[rid].base = base;
        f->expc[rid].nidx = nout > 12 ? 12 : nout;
        for (uint32_t q = 0; q < nout && q < 12; q++) f->expc[rid].idx[q] = out[q];
        f->expc[rid].sl0 = 0;
        f->expc[rid].sln = arrlen(f, curi);
        return;
    }
    if (f->ty[curi].kind == K_ARR && f->iface[curi] && elemreg && f->elemslice[rid]) {
        /* a member slice of the flat slot array: the same treatment, the slot
         * window is [sl0, sl0+sln) */
        f->vty[rid] = newrt;
        f->expc[rid].base = base;
        f->expc[rid].nidx = nout > 12 ? 12 : nout;
        for (uint32_t q = 0; q < nout && q < 12; q++) f->expc[rid].idx[q] = out[q];
        f->expc[rid].sl0 = f->elemidx[rid] * f->elemstride[rid];
        f->expc[rid].sln = f->elemstride[rid];
        return;
    }

    /* the chain pointer's type must follow the rewritten indices: the load/store
     * path reads f->vty[] to pick the load result type (a stale original
     * ptr-to-double type would emit OpLoad v2uint from the uint4 slot pointer
     * and a size-violating v2uint -> v2ulong bitcast) */
    f->vty[rid] = newrt;
    bpush(&f->body, ((uint32_t)nout + 4) << 16 | (inbounds ? SpvOpInBoundsAccessChain : SpvOpAccessChain));
    bpush(&f->body, newrt);
    bpush(&f->body, rid);
    bpush(&f->body, base);
    for (uint32_t i = 0; i < nout; i++) bpush(&f->body, out[i]);
}

/* -------------------------------------------------- body streaming      -- */

static void copy_inst(F64 *f, size_t j)
{
    for (uint32_t i = 0; i < f->ins[j].wc; i++) bpush(&f->body, f->w[f->ins[j].off + i]);
}

static void emit_result(F64 *f, uint32_t rt, uint32_t rid, uint32_t res)
{
    uint32_t o[3] = {rt, rid, res};
    binst(&f->body, SpvOpCopyObject, o, 3);
}

/* "rows" of a multi-slot interface representation: the dvecR width the slots
 * carry (>= 2 — narrower values map to a plain uint2/uint4).  Selects the slot
 * type of the conversion struct (see iface_slot_struct). */
static uint32_t iface_repr_rows(F64 *f, uint32_t orig)
{
    Ty *t = &f->ty[orig];
    if (t->kind == K_FLOAT) return 1;
    if (t->kind == K_VEC) return t->n;
    if (t->kind == K_MAT) return veclen(f, t->elem);
    if (t->kind == K_ARR) {
        if (f->ty[t->elem].kind == K_MAT) return veclen(f, f->ty[t->elem].elem);
        return f->ty[t->elem].kind == K_FLOAT ? 1 : f->ty[t->elem].n;
    }
    return 2;
}

/* the slot element type a conversion struct built for `rows` carries.  Not
 * looked up through struct_member(): the ifmem registry is capped and full
 * long before the late conversion structs are created (soft-float modules
 * declare dozens of struct types), so member lookups there return garbage. */
static uint32_t iface_slot_ty(F64 *f, uint32_t rows)
{
    return (rows == 1) ? uint2_of(f) : uint4_of(f);
}

/* one slot of a flat slot ARRAY reached through `ptr`: either a complete
 * OpAccessChain replaying the recorded walk of a suppressed chain (see the expc
 * comment in the F64 struct) plus the slot index — the form SPIRV-Cross's
 * tessellation stage-IO flattening understands — or, for a direct
 * whole-variable access (no recorded chain), a second-level chain [ptr, j].
 * Whole-array loads and stores are avoided on purpose — SPIRV-Cross renders an
 * array value copy as an address-space-specific spvArrayCopyFrom* helper call,
 * which does not compile for the device/threadgroup stage IO MGL builds its
 * tessellation and geometry records from. */
static uint32_t iface_slot_ptr(F64 *f, uint32_t ptr, uint32_t arrty, uint32_t j)
{
    uint32_t sloty = f->ty[arrty].elem;

    if (f->expc[ptr].nidx)
    {
        uint32_t pty = f->vty[ptr];
        uint32_t storage = (pty && f->ty[pty].kind == K_PTR) ? f->ty[pty].n : SpvStorageClassOutput;
        uint32_t sptr = ptr_for(f, storage, sloty);
        uint32_t id = nid(f);
        uint32_t n = f->expc[ptr].nidx;

        bpush(&f->body, (n + 5) << 16 | SpvOpAccessChain);
        bpush(&f->body, sptr);
        bpush(&f->body, id);
        bpush(&f->body, f->expc[ptr].base);
        for (uint32_t k = 0; k < n; k++) bpush(&f->body, f->expc[ptr].idx[k]);
        bpush(&f->body, k_u32(f, j));
        f->vty[id] = sptr;
        return id;
    }
    {
        uint32_t pty = f->vty[ptr];
        uint32_t storage = (pty && f->ty[pty].kind == K_PTR) ? f->ty[pty].n : SpvStorageClassOutput;
        uint32_t sptr = ptr_for(f, storage, sloty);
        uint32_t id = nid(f);
        uint32_t o[5] = {sptr, id, ptr, k_u32(f, j)};
        binst(&f->body, SpvOpAccessChain, o, 4);
        f->vty[id] = sptr;
        return id;
    }
}


static void rw_load_store(F64 *f, size_t j, bool isload)
{
    const uint32_t *iw = f->w + f->ins[j].off;
    uint32_t ptr = isload ? iw[3] : iw[1];
    uint32_t pty = f->vty[ptr];
    if (!pty || f->ty[pty].kind != K_PTR) { copy_inst(f, j); return; }
    uint32_t storage = f->ty[pty].n;
    uint32_t pointee = f->ty[pty].elem;
    if (f->attrslot[pty]) {
        if (!isload) {
            fail(f, "fp64: vertex attributes are read-only");
            return;
        }
        /* double vertex attribute: the words were fetched into uint2/uint4
           slots; reassemble the u64 register value with pure bitcasts */
        uint32_t rt = iw[1], rid = iw[2];
        uint32_t var = ptr, comp = 0xffffffff;
        if (f->attrcomp[ptr]) {          /* component access: pos.x via OpAccessChain */
            var = f->attrcomp[ptr] >> 4;
            comp = (f->attrcomp[ptr] & 15) - 1;
        }
        uint32_t ptyv = f->vty[var];
        if (!ptyv || !f->attrslot[ptyv]) { fail(f, "fp64: internal: attribute pointer chain"); return; }
        uint32_t n = f->attrn[ptyv];
        uint32_t extra = (f->ins[j].wc > 4) ? 1 : 0;
        uint32_t lo = nid(f);
        uint32_t slotarr = f->attrslotidx[ptr];
        uint32_t o[4] = {f->ty[ptyv].elem, lo, var, extra ? iw[4] : 0};

        if (slotarr != 0xffffffffu)
        {
            /* array/matrix attribute: the slot variable is a flattened array of
             * uint2/uint4 (one per Metal location); read slot `slotarr` and take
             * the recorded pair.  Only element/component reads reach here (the
             * fold in rw_access rejects anything else). */
            uint32_t elemty = f->ty[f->ty[ptyv].elem].elem;    /* uint2 or uint4 */
            uint32_t jconst = nid(f);

            f->vty[jconst] = f->u32ty; f->isconst[jconst] = 1; f->cval[jconst] = slotarr;
            {
                uint32_t oc[4] = {f->u32ty, jconst, slotarr};
                binst(&f->pre, SpvOpConstant, oc, 3);
            }
            {
                uint32_t pty2 = ptr_for(f, storage, elemty);
                uint32_t ap = nid(f);
                uint32_t oa[5] = {pty2, ap, var, jconst};

                f->vty[ap] = pty2;
                binst(&f->body, SpvOpAccessChain, oa, 4);
                {
                    uint32_t ol[3] = {elemty, lo, ap};

                    binst(&f->body, SpvOpLoad, ol, 3);
                }
            }
            /* `lo` is now the slot's u32 vector */
            uint32_t res;

            if (comp == 0xffffffff) {
                if (elemty == uint2_of(f)) res = bitcast(f, f->f64ty, lo);
                else if (f->ty[rt].kind == K_VEC && f->ty[rt].n == 2) res = bitcast(f, ulong2_of(f), lo);
                else if (f->ty[rt].kind == K_VEC && (f->ty[rt].n == 3 || f->ty[rt].n == 4) &&
                         elemty == uint4_of(f))
                {
                    /* whole dvec3/dvec4 array element: two consecutive slots of the
                     * flattened slot array (each one Metal location = one dvec2, so the
                     * array's elements are uniformly uint4) — reassemble like the
                     * whole-attribute load above (the LAST slot of a dvec3 element is
                     * padded, only .x is meaningful) */
                    uint32_t jconst2 = nid(f), ap2 = nid(f), hiv = nid(f);
                    uint32_t pty2 = ptr_for(f, storage, elemty);
                    uint32_t e[4];

                    f->vty[jconst2] = f->u32ty;
                    f->isconst[jconst2] = 1;
                    f->cval[jconst2] = slotarr + 1;
                    {
                        uint32_t oc[4] = {f->u32ty, jconst2, slotarr + 1};
                        binst(&f->pre, SpvOpConstant, oc, 3);
                    }
                    f->vty[ap2] = pty2;
                    {
                        uint32_t oa[5] = {pty2, ap2, var, jconst2};
                        binst(&f->body, SpvOpAccessChain, oa, 4);
                    }
                    {
                        uint32_t ol[3] = {elemty, hiv, ap2};
                        binst(&f->body, SpvOpLoad, ol, 3);
                    }
                    {
                        uint32_t bl = bitcast(f, ulong2_of(f), lo);

                        e[0] = xcomp(f, f->f64ty, bl, 0);
                        e[1] = xcomp(f, f->f64ty, bl, 1);
                        if (f->ty[rt].n == 3)
                        {
                            uint32_t bh = bitcast(f, ulong2_of(f), hiv);
                            e[2] = xcomp(f, f->f64ty, bh, 0);
                        }
                        else
                        {
                            uint32_t bh = bitcast(f, ulong2_of(f), hiv);
                            e[2] = xcomp(f, f->f64ty, bh, 0);
                            e[3] = xcomp(f, f->f64ty, bh, 1);
                        }
                    }
                    res = cstruct(f, rt, e, f->ty[rt].n);
                }
                else { fail(f, "fp64: only scalar / dvec2 array attribute elements are read whole"); return; }
            } else if (elemty == uint2_of(f)) {
                /* the slot element IS one scalar double: the uint2 pair, not a uint4 */
                res = bitcast(f, f->f64ty, lo);
            } else {
                uint32_t bl = bitcast(f, ulong2_of(f), lo);

                res = xcomp(f, f->f64ty, bl, comp & 1);
            }
            emit_result(f, rt, rid, res);
            return;
        }
        binst(&f->body, SpvOpLoad, o, 3 + extra);
        uint32_t res;
        bool lo_pair = (o[0] == uint2_of(f));   /* scalar-double element: a uint2 pair, not a uint4 slot */
        if (comp == 0xffffffff) {
            if (n == 1 || lo_pair) {
                res = bitcast(f, f->f64ty, lo);
            } else if (n == 2) {
                res = bitcast(f, ulong2_of(f), lo);
            } else {
                uint32_t hi = f->attrhi[var];
                if (!hi) { fail(f, "fp64: internal: missing hi slot of a dvec3/dvec4 attribute"); return; }
                uint32_t hiv = nid(f);
                uint32_t oh[3] = {attr_slot1_ty(f, n), hiv, hi};
                binst(&f->body, SpvOpLoad, oh, 3);
                uint32_t bl = bitcast(f, ulong2_of(f), lo);
                uint32_t e[4];
                e[0] = xcomp(f, f->f64ty, bl, 0);
                e[1] = xcomp(f, f->f64ty, bl, 1);
                if (n == 3) {
                    /* slot1 is uint2: exactly one double */
                    e[2] = bitcast(f, f->f64ty, hiv);
                } else {
                    uint32_t bh = bitcast(f, ulong2_of(f), hiv);
                    e[2] = xcomp(f, f->f64ty, bh, 0);
                    e[3] = xcomp(f, f->f64ty, bh, 1);
                }
                res = cstruct(f, rt, e, n);
            }
        } else if (comp < 2 && !lo_pair) {
            uint32_t bl = bitcast(f, ulong2_of(f), lo);
            res = xcomp(f, f->f64ty, bl, comp);
        } else if (comp < 2) {
            res = bitcast(f, f->f64ty, lo);   /* the uint2 element IS component 0 */
        } else {
            uint32_t hi = f->attrhi[var];
            if (!hi || comp >= n) { fail(f, "fp64: internal: attribute component %u of %u", comp, n); return; }
            uint32_t hiv = nid(f);
            uint32_t oh[3] = {attr_slot1_ty(f, n), hiv, hi};
            binst(&f->body, SpvOpLoad, oh, 3);
            if (n == 3) {
                res = bitcast(f, f->f64ty, hiv);        /* uint2 slot: comp 2 is the whole slot */
            } else {
                uint32_t bh = bitcast(f, ulong2_of(f), hiv);
                res = xcomp(f, f->f64ty, bh, comp - 2);
            }
        }
        emit_result(f, rt, rid, res);
        return;
    }
    if (f->arrlane[ptr]) {
        /* load/store through a chain that landed on a widened scalar-double
         * buffer-array element (vec<lanes>(u64) with the std140 stride) */
        uint32_t vt = f->arrlane[ptr];
        uint32_t lanes = f->ty[vt].n;
        if (isload) {
            uint32_t tmp = nid(f);
            uint32_t o[3] = {vt, tmp, ptr};
            binst(&f->body, SpvOpLoad, o, 3);
            uint32_t res = xcomp(f, f->f64ty, tmp, 0);
            emit_result(f, iw[1], iw[2], res);
        } else {
            /* OpStore: [ptr, value] */
            uint32_t comps[4];
            comps[0] = iw[2];
            if (lanes < 1 || lanes > 4) { fail(f, "fp64: widened array store lanes %u", lanes); return; }
            for (uint32_t k = 1; k < lanes; k++) comps[k] = k_dbl(f, 0.0);
            uint32_t v = cstruct(f, vt, comps, lanes);
            binst2(&f->body, SpvOpStore, ptr, v);
        }
        return;
    }
    if (f->attrcomp[ptr] && (f->attrcomp[ptr] >> 4) == 0) {
        /* slot+pair selection recorded by rw_access: the chain pointer itself
         * names the uint4 slot; double #p of it is the value */
        if (!isload) {
            fail(f, "fp64: component writes to a double stage variable "
                    "are not supported (write the whole variable)");
            return;
        }
        uint32_t p = (f->attrcomp[ptr] & 15) - 1;
        uint32_t tmp = nid(f);
        uint32_t o[3] = {pointee, tmp, ptr};
        binst(&f->body, SpvOpLoad, o, 3);
        uint32_t pr = bitcast(f, ulong2_of(f), tmp);
        uint32_t res = xcomp(f, f->f64ty, pr, p);
        emit_result(f, iw[1], iw[2], res);
        return;
    }
    if (f->elemreg[ptr]) {
        /* one element of a stage-IO array of doubles (a uint2/uint4 slot for a
         * scalar/dvec2 element; for dvec3/dvec4/matrix elements the chain lands
         * on the whole member struct and the element is a member SLICE — see
         * the elemslice recording in rw_access) */
        uint32_t orig = f->elemreg[ptr];
        uint32_t fk = f->ty[orig].kind;
        uint32_t flat = 0;      /* the uint4[] member of a wrapper struct */
        uint32_t slice = f->elemslice[ptr];
        uint32_t sl0 = 0, sln = 0;

        if (slice)
        {
            uint32_t rows = (fk == K_MAT) ? veclen(f, f->ty[orig].elem) : f->ty[orig].n;
            uint32_t es = (fk == K_MAT) ? f->ty[orig].n * ((rows + 1) / 2) : (rows + 1) / 2;

            sl0 = f->elemidx[ptr] * es;
            sln = es;
            flat = iface_slot_struct(f, es, rows);   /* the element struct type */
        }
        else
        {
            if (fk == K_VEC && f->ty[orig].n >= 3) flat = smember(f, pointee, 0);
            if (fk == K_MAT) flat = smember(f, pointee, 0);
            if (f->ty[pointee].kind == K_ARR) {
                /* a flat slot ARRAY always carries multi-slot elements, which
                 * are addressed through a slice (see rw_access) */
                fail(f, "fp64: internal: slot-array element without a slice");
                return;
            }
        }
        if (isload) {
            uint32_t res;
            uint32_t v;

            if (f->ty[pointee].kind == K_ARR)
            {
                /* the member/variable is a flat slot ARRAY: the slice's slots
                 * are its elements — chain to each and load it (a whole-array
                 * load is not allowed, see iface_slot_ptr) */
                uint32_t sloty = f->ty[pointee].elem;
                uint32_t mids[16];

                for (uint32_t q = 0; q < sln && q < 16; q++)
                {
                    uint32_t c = iface_slot_ptr(f, ptr, pointee, slice ? sl0 + q : q);
                    uint32_t lv = nid(f);
                    uint32_t o[3] = {sloty, lv, c};

                    binst(&f->body, SpvOpLoad, o, 3);
                    mids[q] = lv;
                }
                v = cstruct_n(f, flat, mids, sln);
            }
            else
            {
                uint32_t tmp = nid(f);
                uint32_t o[3] = {pointee, tmp, ptr};
                binst(&f->body, SpvOpLoad, o, 3);
                v = tmp;
                if (slice)
                {
                    /* extract the element's slot members out of the whole member */
                    uint32_t mids[16];
                    uint32_t mty = (f->ty[flat].kind == K_STRUCT) ? struct_member(f, flat, 0) : uint4_of(f);

                    for (uint32_t q = 0; q < sln && q < 16; q++)
                        mids[q] = xcomp(f, mty, tmp, sl0 + q);
                    v = cstruct_n(f, flat, mids, sln);
                }
            }
            if (fk == K_FLOAT) {
                /* the element slot is a uint4 (16B, the canonical "two doubles per
                 * 128-bit location" slot) or a uint2 (8B — SPIRV-Cross's own
                 * rendering of a scalar-double stage variable).  Bitcast to the
                 * matching size: a size-mismatched OpBitcast would be emitted by
                 * SPIRV-Cross as a CONVERTING constructor (ulong2(x)) and silently
                 * zero-extend. */
                if (f->ty[pointee].kind == K_VEC && f->ty[pointee].n == 2 &&
                    f->ty[f->ty[pointee].elem].width == 32)
                    res = bitcast(f, f->f64ty, v);   /* uint2 (8B) -> u64 */
                else
                    res = xcomp(f, f->f64ty, bitcast(f, ulong2_of(f), v), 0);
            } else if (fk == K_VEC && f->ty[orig].n == 2) {
                res = bitcast(f, ulong2_of(f), v);
            } else if (fk == K_VEC) {
                uint32_t arr = slice ? v : xcomp(f, flat, v, 0);
                res = iface_pair_to_reg(f, arr, flat, orig);
            } else {   /* K_MAT */
                uint32_t arr = slice ? v : xcomp(f, flat, v, 0);
                res = iface_to_reg(f, arr, flat, orig);
            }
            emit_result(f, iw[1], iw[2], res);
        } else {
            uint32_t val = iw[2];
            uint32_t conv;
            if (fk == K_FLOAT) {
                if (f->ty[pointee].kind == K_VEC && f->ty[pointee].n == 2 &&
                    f->ty[f->ty[pointee].elem].width == 32)
                {
                    /* uint2 slot (the fp64 iface form): store the double's words */
                    conv = bitcast(f, uint2_of(f), val);
                }
                else
                {
                    /* uint4 slot (SPIRV-Cross's own rendering of a scalar-double
                     * stage variable): uint4(lo, hi) with the high words zero */
                    uint32_t z = k_u64(f, 0);
                    uint32_t b = bitcast(f, uint2_of(f), val);
                    uint32_t zz = bitcast(f, uint2_of(f), z);

                    conv = cstruct(f, uint4_of(f), (uint32_t[]){b, zz}, 2);
                }
            } else if (fk == K_VEC && f->ty[orig].n == 2) {
                conv = bitcast(f, uint4_of(f), val);
            } else if (fk == K_VEC) {
                uint32_t arr = reg_to_iface_pair(f, val, flat, orig);
                conv = slice ? arr : cstruct(f, pointee, &arr, 1);
            } else {   /* K_MAT */
                uint32_t arr = reg_to_iface(f, val, flat, orig);
                conv = slice ? arr : cstruct(f, pointee, &arr, 1);
            }
            if (slice)
            {
                if (f->ty[pointee].kind == K_ARR)
                {
                    /* the member is a flat slot ARRAY: the slice's slots are its
                     * own elements — store each directly, no read-modify-write */
                    uint32_t sloty = f->ty[pointee].elem;
                    uint32_t srows = (fk == K_MAT) ? veclen(f, f->ty[orig].elem)
                                                   : f->ty[orig].n;
                    uint32_t mty = iface_slot_ty(f, srows);

                    for (uint32_t q = 0; q < sln && q < 16; q++)
                        binst2(&f->body, SpvOpStore,
                               iface_slot_ptr(f, ptr, pointee, sl0 + q),
                               xcomp(f, mty, conv, q));
                    return;
                }
                /* splice the element's slots into the member struct: load the
                 * whole member, replace [sl0, sl0+sln), store it back */
                uint32_t nmem = struct_nmembers(f, pointee);
                uint32_t mids[64];
                uint32_t mty = struct_member(f, pointee, 0);
                uint32_t whole = nid(f);
                uint32_t ol[3] = {pointee, whole, ptr};

                if (nmem > 64) nmem = 64;
                binst(&f->body, SpvOpLoad, ol, 3);
                for (uint32_t q = 0; q < nmem; q++)
                    mids[q] = (q >= sl0 && q < sl0 + sln) ?
                              xcomp(f, mty, conv, q - sl0) :
                              xcomp(f, mty, whole, q);
                {
                    uint32_t ns = cstruct_n(f, pointee, mids, nmem);

                    binst2(&f->body, SpvOpStore, ptr, ns);
                }
                return;
            }
            binst2(&f->body, SpvOpStore, ptr, conv);
        }
        return;
    }
    if (f->iface[pointee]) {
        if (f->attrcomp[ptr]) {
            /* constant component read of a double stage variable (v.x):
             * fetch the whole variable, convert, extract the component */
            if (!isload) {
                fail(f, "fp64: component writes to a double stage variable "
                        "are not supported (write the whole variable)");
                return;
            }
            uint32_t var = f->attrcomp[ptr] >> 4, comp = (f->attrcomp[ptr] & 15) - 1;
            uint32_t vp = f->vty[var];
            if (!vp || !f->ioiface[vp]) { copy_inst(f, j); return; }
            uint32_t vpt = f->ty[vp].elem;
            uint32_t tmp = nid(f);
            uint32_t o[3] = {vpt, tmp, var};
            binst(&f->body, SpvOpLoad, o, 3);
            uint32_t reg = iface_to_reg(f, tmp, vpt, f->iforig[vpt]);
            uint32_t res = xcomp(f, f->f64ty, reg, comp);
            emit_result(f, iw[1], iw[2], res);
            return;
        }
        if (isload) {
            uint32_t rt = iw[1], rid = iw[2];
            if (f->ty[pointee].kind == K_ARR) {
                /* multi-slot flat representation (dvec3/dvec4, double matrices,
                 * arrays of them): load slot by slot, assemble the slot struct
                 * the conversions take, and never materialize the array value */
                uint32_t n = arrlen(f, pointee);
                uint32_t sloty = f->ty[pointee].elem;
                uint32_t orig = f->iforig[pointee];

                if (!n || n > 32) {
                    fail(f, "fp64: double stage variable spans %u slots", n);
                    return;
                }
                {
                    uint32_t sty = iface_slot_struct(f, n, iface_repr_rows(f, orig));
                    uint32_t mids[32];

                    for (uint32_t q = 0; q < n; q++) {
                        uint32_t c = iface_slot_ptr(f, ptr, pointee, q);
                        uint32_t v = nid(f);
                        uint32_t o[3] = {sloty, v, c};

                        binst(&f->body, SpvOpLoad, o, 3);
                        mids[q] = v;
                    }
                    {
                        uint32_t res = iface_to_reg(f, cstruct_n(f, sty, mids, n), sty, orig);

                        emit_result(f, rt, rid, res);
                    }
                }
                return;
            }
            uint32_t tmp = nid(f);
            uint32_t o[4] = {pointee, tmp, ptr};
            uint32_t n = (f->ins[j].wc > 4) ? 4 : 3;
            if (n == 4) o[3] = iw[4];
            binst(&f->body, SpvOpLoad, o, n);
            uint32_t res = iface_to_reg(f, tmp, pointee, f->iforig[pointee]);
            emit_result(f, rt, rid, res);
        } else {
            if (f->ty[pointee].kind == K_ARR) {
                uint32_t n = arrlen(f, pointee);
                uint32_t sloty = f->ty[pointee].elem;
                uint32_t orig = f->iforig[pointee];

                if (!n || n > 32) {
                    fail(f, "fp64: double stage variable spans %u slots", n);
                    return;
                }
                {
                    uint32_t sty = iface_slot_struct(f, n, iface_repr_rows(f, orig));
                    uint32_t conv = reg_to_iface(f, iw[2], sty, orig);
                    uint32_t mty = iface_slot_ty(f, iface_repr_rows(f, orig));

                    for (uint32_t q = 0; q < n; q++)
                        binst2(&f->body, SpvOpStore, iface_slot_ptr(f, ptr, pointee, q),
                               xcomp(f, mty, conv, q));
                }
                return;
            }
            uint32_t val = iw[2];
            uint32_t conv = reg_to_iface(f, val, pointee, f->iforig[pointee]);
            binst2(&f->body, SpvOpStore, ptr, conv);
        }
        return;
    }
    if (f->rowmajor[pointee]) {
        if (isload) {
            uint32_t rt = iw[1], rid = iw[2];
            uint32_t tmp = nid(f);
            uint32_t o[3] = {pointee, tmp, ptr};
            binst(&f->body, SpvOpLoad, o, 3);
            uint32_t res = row_to_reg(f, tmp, pointee, f->roworig[pointee]);
            emit_result(f, rt, rid, res);
        } else {
            uint32_t val = iw[2];
            uint32_t conv = reg_to_row(f, val, pointee, f->roworig[pointee]);
            binst2(&f->body, SpvOpStore, ptr, conv);
        }
        return;
    }
    copy_inst(f, j);
}

static bool rw_body_inst(F64 *f, size_t j)
{
    uint32_t op = f->ins[j].op;
    const uint32_t *iw = f->w + f->ins[j].off;
    uint32_t wc = f->ins[j].wc;

    switch (op) {
    case SpvOpLoad:
        rw_load_store(f, j, true);
        return true;
    case SpvOpStore:
        rw_load_store(f, j, false);
        return true;
    case SpvOpAccessChain:
        rw_access(f, j, false);
        return true;
    case SpvOpInBoundsAccessChain:
        rw_access(f, j, true);
        return true;
    case SpvOpCopyMemory: {
        uint32_t dst = iw[1], src = iw[2];
        uint32_t pty = f->vty[src];
        uint32_t pointee = (pty && f->ty[pty].kind == K_PTR) ? f->ty[pty].elem : 0;
        if (pointee && (f->iface[pointee] || f->rowmajor[pointee])) {
            uint32_t tmp = nid(f);
            uint32_t o[3] = {pointee, tmp, src};
            binst(&f->body, SpvOpLoad, o, 3);
            uint32_t v = (f->iface[pointee])
                ? iface_to_reg(f, tmp, pointee, f->iforig[pointee])
                : row_to_reg(f, tmp, pointee, f->roworig[pointee]);
            uint32_t conv = (f->iface[pointee])
                ? reg_to_iface(f, v, pointee, f->iforig[pointee])
                : reg_to_row(f, v, pointee, f->roworig[pointee]);
            binst2(&f->body, SpvOpStore, dst, conv);
            return true;
        }
        return false;
    }
    case SpvOpFAdd: case SpvOpFSub: case SpvOpFMul: case SpvOpFDiv:
    case SpvOpFRem: case SpvOpFMod: case SpvOpFNegate: {
        static const char *fns[] = { "mgl_fadd64", "mgl_fsub64", "mgl_fmul64", "mgl_fdiv64",
                                     "mgl_frem64", "mgl_fmod64", "mgl_fneg64" };
        static const uint32_t ops[] = { SpvOpFAdd, SpvOpFSub, SpvOpFMul, SpvOpFDiv,
                                        SpvOpFRem, SpvOpFMod, SpvOpFNegate };
        if (!is64(f, iw[1])) return false;                 /* plain fp32 math */
        const char *fn = NULL;
        for (int i = 0; i < 7; i++) if (ops[i] == op) fn = fns[i];
        uint32_t rt = iw[1], rid = iw[2];
        uint32_t res = (wc >= 5) ? cw2(f, rt, iw[3], iw[4], fn) : cw1(f, rt, iw[3], fn);
        emit_result(f, rt, rid, res);
        return true;
    }
    case SpvOpFOrdEqual: case SpvOpFUnordEqual:
    case SpvOpFOrdNotEqual: case SpvOpFUnordNotEqual:
    case SpvOpFOrdLessThan: case SpvOpFUnordLessThan:
    case SpvOpFOrdGreaterThan: case SpvOpFUnordGreaterThan:
    case SpvOpFOrdLessThanEqual: case SpvOpFUnordLessThanEqual:
    case SpvOpFOrdGreaterThanEqual: case SpvOpFUnordGreaterThanEqual: {
        int kind;
        switch (op) {
        case SpvOpFOrdEqual: kind = CMP_OE; break;
        case SpvOpFUnordEqual: kind = CMP_UE; break;
        case SpvOpFOrdNotEqual: kind = CMP_ONE; break;
        case SpvOpFUnordNotEqual: kind = CMP_UNE; break;
        case SpvOpFOrdLessThan: kind = CMP_OL; break;
        case SpvOpFUnordLessThan: kind = CMP_UL; break;
        case SpvOpFOrdGreaterThan: kind = CMP_OG; break;
        case SpvOpFUnordGreaterThan: kind = CMP_UG; break;
        case SpvOpFOrdLessThanEqual: kind = CMP_OLE; break;
        case SpvOpFUnordLessThanEqual: kind = CMP_ULE; break;
        case SpvOpFOrdGreaterThanEqual: kind = CMP_OGE; break;
        default: kind = CMP_UGE; break;
        }
        uint32_t rt = iw[1], rid = iw[2];
        if (!is_f64_val(f, iw[3])) return false;         /* plain fp32 compare */
        uint32_t res = cw_cmp(f, rt, iw[3], iw[4], vlen_of(f, rt), kind);
        emit_result(f, rt, rid, res);
        return true;
    }
    case SpvOpFConvert: {
        uint32_t rt = iw[1], rid = iw[2], x = iw[3];
        uint32_t xt = f->vty[x];
        if (!is64(f, rt) && !is64(f, xt)) return false;   /* f32 <-> f16 etc. */
        uint32_t res;
        if (is64(f, rt)) res = conv_to_f64(f, rt, x, xt);
        else res = conv_from_f64(f, rt, x, xt);
        emit_result(f, rt, rid, res);
        return true;
    }
    case SpvOpConvertFToS: case SpvOpConvertFToU: {
        uint32_t rt = iw[1], rid = iw[2], x = iw[3];
        if (!is64(f, f->vty[x])) return false;             /* plain float casts */
        emit_result(f, rt, rid, conv_from_f64(f, rt, x, f->vty[x]));
        return true;
    }
    case SpvOpConvertSToF: case SpvOpConvertUToF: {
        if (!is64(f, iw[1])) return false;                 /* int -> float */
        uint32_t rt = iw[1], rid = iw[2], x = iw[3];
        /* narrow ints widen to 32 bits first */
        {
            uint32_t xt = f->vty[x];
            Ty *st = &f->ty[xt];
            uint32_t ety = st->kind == K_VEC ? st->elem : xt;
            if (f->ty[ety].kind == K_INT && f->ty[ety].width < 32) {
                uint32_t n = st->kind == K_VEC ? st->n : 1;
                uint32_t w32 = f->ty[ety].sgn ? f->i32ty : f->u32ty;
                if (n == 1) x = widen_int(f, x, ety);
                else {
                    uint32_t comps[4];
                    for (uint32_t i = 0; i < n; i++) comps[i] = widen_int(f, xcomp(f, ety, x, i), ety);
                    uint32_t vty32 = mkvec(f, w32, n, false);
                    x = cstruct(f, vty32, comps, n);
                }
            }
        }
        emit_result(f, rt, rid, conv_to_f64(f, rt, x, f->vty[x]));
        return true;
    }
    case SpvOpIsNan: case SpvOpIsInf: {
        uint32_t rt = iw[1], rid = iw[2], x = iw[3];
        uint32_t xt = f->vty[x];
        if (!is64(f, xt)) return false;                 /* plain fp32 isnan/isinf */
        const char *fn = (op == SpvOpIsNan) ? "mgl_is_nan" : "mgl_isinf64";
        uint32_t n = vlen_of(f, xt);
        uint32_t res;
        if (n == 1) res = fcall(f, bool_of(f), fn, &x, 1);
        else {
            uint32_t comps[4];
            for (uint32_t i = 0; i < n; i++) {
                uint32_t xi = dcomp(f, x, i);
                comps[i] = fcall(f, bool_of(f), fn, &xi, 1);
            }
            res = cstruct(f, rt, comps, n);
        }
        emit_result(f, rt, rid, res);
        return true;
    }
    case SpvOpDot: {
        uint32_t rt = iw[1], rid = iw[2];
        if (!is64(f, rt)) return false;                 /* plain fp32 dot */
        uint32_t n = veclen(f, f->vty[iw[3]]);
        emit_result(f, rt, rid, dot_val(f, iw[3], iw[4], n));
        return true;
    }
    case SpvOpVectorTimesScalar: {
        uint32_t rt = iw[1], rid = iw[2];
        if (!is64(f, rt)) return false;                 /* plain fp32 vec*scalar */
        emit_result(f, rt, rid, cw2(f, rt, iw[3], iw[4], "mgl_fmul64"));
        return true;
    }
    case SpvOpMatrixTimesScalar: {
        uint32_t rt = iw[1], rid = iw[2];
        if (!is64(f, rt)) return false;                 /* plain fp32 mat*scalar */
        emit_result(f, rt, rid, cw2(f, rt, iw[3], iw[4], "mgl_fmul64"));
        return true;
    }
    case SpvOpMatrixTimesVector: {
        uint32_t rt = iw[1], rid = iw[2], m = iw[3], v = iw[4];
        if (!is64(f, rt)) return false;                 /* plain fp32 mat*vec */
        uint32_t C = veclen(f, rt);
        uint32_t R = veclen(f, f->ty[f->vty[m]].elem);
        uint32_t comps[4];
        for (uint32_t c = 0; c < C; c++) {
            uint32_t acc = fmul2(f, xcomp2(f, f->f64ty, m, c, 0), dcomp(f, v, 0));
            for (uint32_t r = 1; r < R; r++)
                acc = fadd2(f, acc, fmul2(f, xcomp2(f, f->f64ty, m, c, r), dcomp(f, v, r)));
            comps[c] = acc;
        }
        emit_result(f, rt, rid, cstruct(f, rt, comps, C));
        return true;
    }
    case SpvOpVectorTimesMatrix: {
        uint32_t rt = iw[1], rid = iw[2], v = iw[3], m = iw[4];
        if (!is64(f, rt)) return false;                 /* plain fp32 vec*mat */
        uint32_t R = veclen(f, rt);
        uint32_t C = f->ty[f->vty[m]].n;
        uint32_t comps[4];
        for (uint32_t r = 0; r < R; r++) {
            uint32_t acc = fmul2(f, dcomp(f, v, 0), xcomp2(f, f->f64ty, m, 0, r));
            for (uint32_t c = 1; c < C; c++)
                acc = fadd2(f, acc, fmul2(f, dcomp(f, v, c), xcomp2(f, f->f64ty, m, c, r)));
            comps[r] = acc;
        }
        emit_result(f, rt, rid, cstruct(f, rt, comps, R));
        return true;
    }
    case SpvOpMatrixTimesMatrix: {
        uint32_t rt = iw[1], rid = iw[2], A = iw[3], B = iw[4];
        if (!is64(f, rt)) return false;                 /* plain fp32 mat*mat */
        uint32_t C2 = f->ty[rt].n;
        uint32_t R1 = veclen(f, f->ty[rt].elem);
        uint32_t C1 = f->ty[f->vty[A]].n;
        uint32_t cols[4];
        for (uint32_t c2 = 0; c2 < C2; c2++) {
            uint32_t col[4];
            for (uint32_t r1 = 0; r1 < R1; r1++) {
                uint32_t acc = fmul2(f, xcomp2(f, f->f64ty, A, 0, r1), xcomp2(f, f->f64ty, B, c2, 0));
                for (uint32_t k = 1; k < C1; k++)
                    acc = fadd2(f, acc, fmul2(f, xcomp2(f, f->f64ty, A, k, r1),
                                              xcomp2(f, f->f64ty, B, c2, k)));
                col[r1] = acc;
            }
            cols[c2] = cstruct(f, f->ty[rt].elem, col, R1);
        }
        emit_result(f, rt, rid, cstruct(f, rt, cols, C2));
        return true;
    }
    case SpvOpOuterProduct: {
        uint32_t rt = iw[1], rid = iw[2];
        if (!is64(f, rt)) return false;                 /* plain fp32 outer */
        uint32_t N = f->ty[rt].n;               /* columns */
        uint32_t M = veclen(f, f->ty[rt].elem); /* rows */
        uint32_t cols[4];
        for (uint32_t c = 0; c < N; c++) {
            uint32_t col[4];
            for (uint32_t r = 0; r < M; r++)
                col[r] = fmul2(f, dcomp(f, iw[3], r), dcomp(f, iw[4], c));
            cols[c] = cstruct(f, f->ty[rt].elem, col, M);
        }
        emit_result(f, rt, rid, cstruct(f, rt, cols, N));
        return true;
    }
    case SpvOpTranspose: {
        uint32_t rt = iw[1], rid = iw[2], m = iw[3];
        if (!is64(f, rt)) return false;                 /* plain fp32 transpose */
        uint32_t C0 = f->ty[f->vty[m]].n;
        uint32_t R0 = veclen(f, f->ty[f->vty[m]].elem);
        uint32_t cols[4];
        for (uint32_t cp = 0; cp < R0; cp++) {
            uint32_t col[4];
            for (uint32_t rp = 0; rp < C0; rp++)
                col[rp] = xcomp2(f, f->f64ty, m, rp, cp);
            cols[cp] = cstruct(f, f->ty[rt].elem, col, C0);
        }
        emit_result(f, rt, rid, cstruct(f, rt, cols, R0));
        return true;
    }
    case SpvOpExtInst: {
        uint32_t rt = iw[1], rid = iw[2];
        uint32_t res = rw_extinst_val(f, iw, wc);
        if (res) emit_result(f, rt, rid, res);
        return true;
    }
    default:
        return false;
    }
}

/* -------------------------------------------------- def streaming       -- */

static uint32_t mat_stride_of(F64 *f, uint32_t mat)
{
    for (size_t i = 0; i < f->nmdecs; i++) {
        if (f->mdecs[i].dec != SpvDecorationMatrixStride) continue;
        uint32_t st = f->mdecs[i].type;
        if (st >= f->cap_ids || f->ty[st].kind != K_STRUCT) continue;
        if (f->mdecs[i].member < struct_nmembers(f, st) &&
            struct_member(f, st, f->mdecs[i].member) == mat)
            return f->mdecs[i].a;
    }
    uint32_t rows = veclen(f, f->ty[mat].elem);
    return (rows == 1) ? 8 : (rows == 2) ? 16 : 32;
}

static bool fold_specop(F64 *f, size_t j, uint32_t rid)
{
    const uint32_t *iw = f->w + f->ins[j].off;
    uint32_t wc = f->ins[j].wc;
    uint32_t rt = iw[1], op2 = iw[3];
    if (!is64(f, rt)) return false;

    double a = 0, b = 0;
    uint32_t na = 0;
    for (uint32_t i = 4; i < wc && na < 2; i++) {
        uint32_t id = iw[i];
        if (id >= f->cap_ids || !f->isconst[id]) { fail(f, "fp64: spec-constant op on non-constant"); return true; }
        uint32_t ty = f->vty[id];
        if (f->ty[ty].width == 64 && f->ty[ty].kind == K_FLOAT) {
            union { uint64_t u; double d; } u;
            u.u = f->cval[id];
            if (na == 0) a = u.d; else b = u.d;
        } else if (f->ty[ty].width == 32 && f->ty[ty].kind == K_FLOAT) {
            union { uint32_t u; float fl; } u;
            u.u = (uint32_t)f->cval[id];
            if (na == 0) a = (double)u.fl; else b = (double)u.fl;
        } else {
            fail(f, "fp64: unsupported spec-constant operand");
            return true;
        }
        na++;
    }

    double v = 0;
    switch (op2) {
    case SpvOpFAdd: v = a + b; break;
    case SpvOpFSub: v = a - b; break;
    case SpvOpFMul: v = a * b; break;
    case SpvOpFDiv: if (b == 0.0) { fail(f, "fp64: spec-constant division by zero"); return true; } v = a / b; break;
    case SpvOpFNegate: v = -a; break;
    case SpvOpFRem: v = fmod(a, b); break;
    case SpvOpFMod: v = a - b * floor(a / b); break;
    case SpvOpFConvert: v = a; break;
    default:
        fail(f, "fp64: spec-constant op %u not supported", op2);
        return true;
    }

    if (is_f64(f, rt)) {
        union { double d; uint64_t u; } u;
        u.d = v;
        uint32_t o[4] = {rt, rid, (uint32_t)u.u, (uint32_t)(u.u >> 32)};
        binst(&f->defs, SpvOpConstant, o, 4);
    } else {
        union { float fl; uint32_t u; } u;
        u.fl = (float)v;
        uint32_t o[3] = {rt, rid, u.u};
        binst(&f->defs, SpvOpConstant, o, 3);
    }
    return true;
}

static bool rw_def_inst(F64 *f, size_t j, uint32_t rid)
{
    uint32_t op = f->ins[j].op;
    const uint32_t *iw = f->w + f->ins[j].off;
    uint32_t wc = f->ins[j].wc;

    switch (op) {
    case SpvOpTypeFloat:
        if (iw[2] == 64)
            return true;   /* re-emitted as OpTypeInt at the head of `pre` */
        return false;
    case SpvOpTypeMatrix:
        if (is64(f, iw[2])) {
            uint32_t o[3] = {iw[1], iw[2], k_u32(f, iw[3])};
            binst(&f->defs, SpvOpTypeArray, o, 3);
            dec_stride(f, iw[1], mat_stride_of(f, iw[1]));
            return true;
        }
        return false;
    case SpvOpTypeArray: {
        if (!widen_buffer_array(f, iw[1]))
            return false;
        /* scalar-double buffer array widened to vec<lanes>(u64): re-emit with
         * the new element so the MSL natural stride equals the decorated one */
        uint32_t o[3] = {iw[1], f->ty[iw[1]].elem, f->ty[iw[1]].n};
        binst(&f->defs, SpvOpTypeArray, o, 3);
        return true;
    }
    case SpvOpTypeRuntimeArray: {
        if (!widen_buffer_array(f, iw[1]))
            return false;
        uint32_t o[2] = {iw[1], f->ty[iw[1]].elem};
        binst(&f->defs, SpvOpTypeRuntimeArray, o, 2);
        return true;
    }
    case SpvOpTypePointer:
        if ((iw[2] == SpvStorageClassInput || iw[2] == SpvStorageClassOutput) && is64(f, iw[3])) {
            /* the pre-pass (fp64_attr_slots) already chose the pointee: raw
               word slots for vertex attributes (vertex stage Input), the
               u32-pair iface representation (dvec3/dvec4 = array of uint4)
               everywhere else.  Reuse that choice verbatim: re-deriving it
               here with attr_slot0_ty() gave a bare uint4 for a fragment-side
               dvec3/dvec4 input while the body pass's load conversion kept
               indexing the array ("as_type cast from scalar 'unsigned int'
               to vector 'ulong2'"). */
            uint32_t pt = f->ty[iw[1]].elem ? f->ty[iw[1]].elem : iw[3];
            uint32_t o[3] = {iw[1], iw[2], pt};
            binst(&f->defs, SpvOpTypePointer, o, 3);
            return true;
        }
        return false;
    case SpvOpVariable:
        /* dvec3/dvec4 attributes: emit the hi-half slot variable right after
           the original (its Location decoration went to `preann` up front) */
        if (iw[3] == SpvStorageClassInput && iw[2] < f->cap_ids && f->attrhi[iw[2]]) {
            for (uint32_t i = 0; i < wc; i++) bpush(&f->defs, iw[i]);
            uint32_t ov[3] = {f->attrhity[f->attrhi[iw[2]]], f->attrhi[iw[2]], SpvStorageClassInput};
            binst(&f->defs, SpvOpVariable, ov, 3);
            return true;
        }
        return false;
    case SpvOpTypeStruct: {
        uint32_t nm = wc - 2;
        bool changed = false;
        for (uint32_t m = 0; m < nm; m++) {
            uint32_t mt = iw[2 + m];
            if (has_mdec(f, iw[1], m, SpvDecorationRowMajor, NULL) &&
                f->ty[mt].kind == K_MAT && is64(f, mt)) {
                if (!f->rowmap[mt]) rowmajor_ty(f, mt);
                changed = true;
            }
        }
        if (!changed) return false;
        uint32_t o[128];
        o[0] = iw[1];
        for (uint32_t m = 0; m < nm && m + 1 < 128; m++) {
            uint32_t mt = iw[2 + m];
            o[m + 1] = (has_mdec(f, iw[1], m, SpvDecorationRowMajor, NULL) &&
                        f->ty[mt].kind == K_MAT && is64(f, mt) && f->rowmap[mt]) ? f->rowmap[mt] : mt;
        }
        binst(&f->defs, SpvOpTypeStruct, o, (nm + 1 < 128 ? nm + 1 : 128));
        return true;
    }
    case SpvOpSpecConstantOp:
        return fold_specop(f, j, rid);
    default:
        return false;
    }
}

/* -------------------------------------------------- driver              -- */

/* build the attribute slot variables: must run before the body pass so the
   body pass sees the uint2/uint4 slot pointees through f->ty.
   `vertex_inputs` is true only for the vertex stage, where StorageClassInput
   really is a vertex attribute (everywhere else it is stage IO). */
static void fp64_attr_slots(F64 *f)
{
    for (size_t j = 0; j < f->nins; j++) {
        uint32_t op = f->ins[j].op;
        const uint32_t *iw = f->w + f->ins[j].off;
        if (op == SpvOpTypePointer && (iw[2] == SpvStorageClassInput || iw[2] == SpvStorageClassOutput) &&
            is64(f, iw[3])) {
            Ty *t = &f->ty[iw[3]];
            if (f->vertex_inputs && iw[2] == SpvStorageClassInput &&
                (t->kind == K_FLOAT || t->kind == K_VEC ||
                 t->kind == K_MAT || t->kind == K_ARR)) {
                uint32_t slot0 = (t->kind == K_FLOAT || t->kind == K_VEC)
                                     ? attr_slot0_ty(f, iw[3])
                                     : attr_slotarr_ty(f, iw[3]);
                if (slot0) {
                    /* vertex attribute: raw word slots (see attr_slot0_ty).  Arrays and
                     * matrices become one slot per Metal location (attr_slotarr_ty);
                     * attrn is the double count of ONE slot, so the dvec3/4 hi-variable
                     * synthesis below only fires for the plain dvec3/dvec4 case. */
                    f->ty[iw[1]].elem = slot0;   /* the def pass emits the same pointee */
                    f->attrslot[iw[1]] = 1;
                    if (t->kind == K_FLOAT || t->kind == K_VEC)
                        f->attrn[iw[1]] = (t->kind == K_VEC) ? t->n : 1;
                    else
                    {
                        uint32_t ne, cl, rw;

                        attr_geom(f, iw[3], &ne, &cl, &rw);
                        f->attrn[iw[1]] = (rw == 1) ? 1 : 2;   /* per slot */
                        f->attrsgeom[iw[1]] = (ne << 16) | (cl << 8) | rw;
                        f->attrorig[iw[1]] = iw[3];
                    }
                }
            } else {
                /* every other double crosses the stage boundary as its u32-word
                 * pairs (two per component; MSL has no doubles).  Rewriting the
                 * pointer's pointee here makes whole-variable OpLoad/OpStore take
                 * the iface conversion path; the def pass emits the same type. */
                uint32_t it = iface_ty(f, iw[3]);
                if (it) {
                    f->ty[iw[1]].elem = it;
                    f->ioiface[iw[1]] = 1;
                }
            }
        } else if (op == SpvOpVariable && iw[3] == SpvStorageClassInput) {
            uint32_t pty = f->vty[iw[2]];
            if (!pty || !f->attrslot[pty] || f->attrn[pty] < 3) continue;
            uint32_t n = f->attrn[pty];
            uint32_t loc = 0;
            if (!has_dec(f, iw[2], SpvDecorationLocation, &loc)) {
                fail(f, "fp64: dvec3/dvec4 vertex attribute without a location");
                return;
            }
            if (f->attrhi[iw[2]]) continue;
            uint32_t hi = nid(f);
            uint32_t hiptr = ptr_for(f, SpvStorageClassInput, attr_slot1_ty(f, n));
            f->attrhi[iw[2]] = hi;
            f->attrhity[hi] = hiptr;
            f->nattrhi++;
            /* Location k+1 (the second half of the two-location attribute) */
            uint32_t od[3] = {hi, SpvDecorationLocation, loc + 1};
            binst(&f->preann, SpvOpDecorate, od, 3);
            /* keep the name distinct from the attribute itself (debug section) */
            char nm[32];
            snprintf(nm, sizeof nm, "mgl_hi%u", loc);
            size_t nb = strlen(nm) + 1, nw = (nb + 3) / 4;
            uint32_t onw[2 + 8] = {0};
            onw[0] = (uint32_t)(2 + nw) << 16 | 5 /* OpName */;
            onw[1] = hi;
            memcpy(onw + 2, nm, nb);
            for (size_t i = 0; i < 2 + nw; i++) bpush(&f->hdradd, onw[i]);
        }
        if (f->failed) return;
    }
}

uint32_t *mgl_fp64_lower_module(const uint32_t *words, size_t nwords, bool vertex_inputs,
                                bool *lowered, size_t *out_nwords, char *log, size_t logsz)
{
    F64 f;
    memset(&f, 0, sizeof(f));
    f.w = words;
    f.n = nwords;
    f.bound = nwords >= 4 ? words[3] : 0;
    f.log = log;
    f.logsz = logsz;
    if (log && logsz) log[0] = 0;
    *lowered = true;
    if (out_nwords) *out_nwords = 0;

    uint32_t *outw = NULL;
    Buf out = {0};

    if (f.n < 5 || words[0] != SpvMagicNumber) {
        fail(&f, "fp64: not a SPIR-V module");
        goto done;
    }
    if (!parse(&f)) {
        if (!f.failed) fail(&f, "fp64: parse failed");
        goto done;
    }
    if (!f.f64ty) {
        *lowered = false;
        goto done;
    }
    if (!ensure_lib(log, logsz)) {
        fail(&f, "fp64: soft-float64 library unavailable%s%s",
             log && logsz && log[0] ? ": " : "", log && logsz && log[0] ? log : "");
        goto done;
    }
    f.lib = fp64_lib_mode() ? &g_lib_fast : &g_lib_soft;

    f.next_id = f.bound;
    f.libshift = f.bound + 4096;
    f.id_limit = f.libshift;
    /* id-indexed arrays must cover the FULL id space (module ids, ids the
     * pre/body passes create, and the merged library's ids).  parse() sized
     * them for the module only, so every id past bound+16 wrote out of
     * bounds — heap corruption that surfaced as flaky SPIRV-Cross parser
     * crashes and garbled lowered modules. */
    {
        uint32_t full = f.libshift + f.lib->bound + 2;
        void *p = NULL;
        #define F64_GROW(field, T) do { \
            p = realloc(f.field, (size_t)full * sizeof(T)); \
            if (!p) { fail(&f, "fp64: out of memory"); goto done; } \
            f.field = (T *)p; \
            memset(f.field + f.cap_ids, 0, (size_t)(full - f.cap_ids) * sizeof(T)); \
        } while (0)
        F64_GROW(ty, Ty);
        F64_GROW(vty, uint32_t);
        F64_GROW(cval, uint64_t);
        F64_GROW(isconst, uint8_t);
        F64_GROW(iface, uint8_t);
        F64_GROW(iforig, uint32_t);
        F64_GROW(rowmajor, uint8_t);
        F64_GROW(roworig, uint32_t);
        F64_GROW(rowmap, uint32_t);
        #undef F64_GROW
        f.cap_ids = full;
    }
    f.vertex_inputs = vertex_inputs;
    f.attrslot = calloc(f.libshift, 1);
    f.attrn = calloc(f.libshift, 4);
    f.attrhi = calloc(f.libshift, 4);
    f.attrhity = calloc(f.libshift, 4);
    f.attrcomp = calloc(f.libshift, 4);
    f.attrslotidx = malloc(f.libshift * 4);
    for (uint32_t q = 0; q < f.libshift; q++) f.attrslotidx[q] = 0xffffffffu;
    f.attrsgeom = calloc(f.libshift, 4);
    f.attrorig = calloc(f.libshift, 4);
    f.elemslice = calloc(f.libshift, 1);
    f.elemidx = calloc(f.libshift, 4);
    f.elemstride = calloc(f.libshift, 4);
    f.elemorig = calloc(f.libshift, 4);
    f.ioiface = calloc(f.libshift, 1);
    f.arrlane = calloc(f.libshift, 4);
    f.elemreg = calloc(f.libshift, 4);
    f.expc = calloc(f.libshift, sizeof *f.expc);
    f.warr = calloc(f.libshift, 1);
    if (!f.attrslot || !f.attrn || !f.attrhi || !f.attrhity || !f.attrcomp || !f.attrslotidx || !f.attrsgeom || !f.attrorig || !f.elemslice || !f.elemidx || !f.elemstride || !f.elemorig || !f.ioiface || !f.arrlane || !f.elemreg || !f.expc || !f.warr) {
        fail(&f, "fp64: out of memory");
        goto done;
    }
    /* the Float64 type id becomes the u64 type: define it first in `pre` so
       every pre-section constant/vector referencing it is well ordered.  Every
       module type `pre` may reference must be marked now, before the def
       stream runs (marks made later would leave duplicates). */
    mark_moved(&f, f.f64ty);
    if (f.u32ty) mark_moved(&f, f.u32ty);
    for (uint32_t n = 1; n <= 4; n++) {
        uint32_t v = module_vec(&f, f.f64ty, n);
        if (v) mark_moved(&f, v);
    }
    if (f.u32ty) {
        uint32_t v2 = module_vec(&f, f.u32ty, 2);
        uint32_t v4 = module_vec(&f, f.u32ty, 4);
        if (v2) mark_moved(&f, v2);
        if (v4) mark_moved(&f, v4);
    }
    /* pre-create the canonical word types (u32, uint2/4, ulong2) in `pre` so
       the library merge can dedupe its own onto them — duplicate
       non-aggregate type declarations crash SPIRV-Cross's parser */
    u32_for_pre(&f);
    uint2_of(&f);
    uint4_of(&f);
    ulong2_of(&f);
    merge_lib(&f);
    f.id_limit = f.libshift + f.lib->bound + 1;
    if (f.failed) goto done;

    /* Fast-mode marker: re-add OpNames (merge_lib drops them, SPIRV-Cross
       would rename the functions to _NNNN) with a df64_ prefix for every
       merged library function.  Any shader that performs double arithmetic
       calls at least one of them, so the generated MSL always carries a
       df64_* identifier, which MGLRenderer detects to compile the module
       with precise math: Metal's default fast-math reassociates the
       error-free transformations the double-single library is built from. */
    if (fp64_lib_mode() == 1) {
        for (size_t i = 0; i < f.lib->nfns; i++) {
            char nm[80];
            snprintf(nm, sizeof nm, "df64_%s", f.lib->fns[i].name);
            size_t nb = strlen(nm) + 1, nw = (nb + 3) / 4;
            uint32_t onw[2 + 24] = {0};
            onw[0] = (uint32_t)(2 + nw) << 16 | 5 /* OpName */;
            onw[1] = f.lib->fns[i].id + f.libshift;
            memcpy(onw + 2, nm, nb);
            for (size_t k = 0; k < 2 + nw; k++) bpush(&f.preann, onw[k]);
        }
    }

    /* vertex attributes -> raw word slots; other double stage IO -> u32 pairs */
    fp64_attr_slots(&f);
    if (f.failed) goto done;

    /* bodies */
    for (size_t j = f.first_fn; j < f.nins; j++) {
        if (f.failed) goto done;
        if (!rw_body_inst(&f, j)) copy_inst(&f, j);
    }
    /* def section */
    for (size_t j = f.first_def; j < f.first_fn; j++) {
        if (f.failed) goto done;
        const uint32_t *iw = f.w + f.ins[j].off;
        uint32_t r2 = (f.ins[j].wc >= 3 && inst_has_result(f.ins[j].op)) ? iw[2] : 0;
        if (!rw_def_inst(&f, j, r2)) {
            if (f.ins[j].op == SpvOpTypeFloat && iw[2] == 64)
                continue;                       /* re-emitted as OpTypeInt in `pre` */
            if (f.ins[j].op >= SpvOpTypeVoid && f.ins[j].op <= SpvOpTypeStruct) {
                uint32_t tid = iw[1];
                if (tid < f.cap_ids && is_moved(&f, tid)) continue;
            }
            for (uint32_t i = 0; i < f.ins[j].wc; i++) bpush(&f.defs, iw[i]);
        }
    }
    if (f.failed) goto done;

    /* relocated module types + the u64 redefinition lead `pre` */
    {
        Buf head = {0};
        for (size_t i = 0; i < f.nmoved; i++) {
            uint32_t tid = f.moved[i];
            if (tid == f.f64ty) {
                uint32_t o[3] = {f.f64ty, 64, 0};
                binst(&head, SpvOpTypeInt, o, 3);
                continue;
            }
            size_t dj = f.ty[tid].def;

            for (uint32_t i = 0; i < f.ins[dj].wc; i++) bpush(&head, f.w[f.ins[dj].off + i]);
        }
        if (head.n) {
            for (size_t i = 0; i < head.n; i++) bpush(&f.lead, head.w[i]);
        }
        free(head.w);
    }
    if (f.failed) goto done;

    /* header + annotations */
    {
        for (uint32_t i = 0; i < 5; i++) bpush(&out, f.w[i]);
        bool cap_done = false;
        for (size_t j = 0; j < f.first_def; j++) {
            uint32_t op = f.ins[j].op, wc = f.ins[j].wc;
            const uint32_t *iw = f.w + f.ins[j].off;
            if (!cap_done && op != SpvOpCapability) {
                uint32_t o[1] = {SpvCapabilityInt64};
                binst(&out, SpvOpCapability, o, 1);
                cap_done = true;
            }
            if (op == SpvOpMemoryModel && f.hdradd.n) {
                for (size_t i = 0; i < f.hdradd.n; i++) bpush(&out, f.hdradd.w[i]);
                f.hdradd.n = 0;   /* flushed: do not emit again past the annotations */
            }
            /* drop layout decorations that no longer apply to double matrices */
            if (op == SpvOpMemberDecorate && wc >= 5 &&
                (iw[3] == SpvDecorationRowMajor || iw[3] == SpvDecorationColMajor ||
                 iw[3] == SpvDecorationMatrixStride)) {
                uint32_t st = iw[1];
                if (st < f.cap_ids && f.ty[st].kind == K_STRUCT &&
                    iw[2] < struct_nmembers(&f, st)) {
                    uint32_t mt = struct_member(&f, st, iw[2]);
                    if (f.ty[mt].kind == K_MAT && is64(&f, mt))
                        continue;
                }
            }
            /* dvec3/dvec4 attributes: the hi slot variable joins the entry
               point's interface (SPIRV-Cross finds stage IO through it) */
            if (op == SpvOpEntryPoint && f.nattrhi) {
                uint32_t extra = 0;
                for (size_t v = 1; v < f.bound; v++)
                    if (f.attrhi[v] && f.attrhi[v] < f.libshift) extra++;
                if (extra) {
                    bpush(&out, (wc + extra) << 16 | op);
                    for (uint32_t i = 1; i < wc; i++) bpush(&out, iw[i]);
                    for (size_t v = 1; v < f.bound; v++)
                        if (f.attrhi[v] && f.attrhi[v] < f.libshift) bpush(&out, f.attrhi[v]);
                    continue;
                }
            }
            for (uint32_t i = 0; i < wc; i++) bpush(&out, iw[i]);
        }
    }
    if (f.hdradd.n) { /* no memory model seen (cannot happen) */
        for (size_t i = 0; i < f.hdradd.n; i++) bpush(&out, f.hdradd.w[i]);
    }

    for (size_t i = 0; i < f.preann.n; i++) bpush(&out, f.preann.w[i]);
    for (size_t i = 0; i < f.lead.n; i++) bpush(&out, f.lead.w[i]);
    for (size_t i = 0; i < f.pre.n; i++) bpush(&out, f.pre.w[i]);
    for (size_t i = 0; i < f.defs.n; i++) bpush(&out, f.defs.w[i]);
    for (size_t i = 0; i < f.libdefs.n; i++) bpush(&out, f.libdefs.w[i]);
    for (size_t i = 0; i < f.libfns.n; i++) bpush(&out, f.libfns.w[i]);
    for (size_t i = 0; i < f.body.n; i++) bpush(&out, f.body.w[i]);

    {
        uint32_t nb = f.next_id;
        if (f.libshift + f.lib->bound > nb) nb = f.libshift + f.lib->bound;
        out.w[3] = nb;
    }
    if (out_nwords) *out_nwords = out.n;
    if (out.n && getenv("MGL_FP64_DUMP"))
    {
        char path[256];
        static int seq;

        snprintf(path, sizeof path, "%s/lowered_%d.spv", getenv("MGL_FP64_DUMP"), seq++);
        FILE *fp = fopen(path, "wb");
        if (fp) { fwrite(out.w, 4, out.n, fp); fclose(fp); }
    }
    if (getenv("MGL_FP64_DUMP")) {
        dump_module(out.w, out.n, "lowered");
        dump_module(f.w, f.n, "input");
        fprintf(stderr, "[fp64] ins=%zu first_def=%zu first_fn=%zu nins=%zu "
                "ann_end_word=%zu lead=%zu preann=%zu pre=%zu defs=%zu libdefs=%zu libfns=%zu body=%zu\n",
                f.nins, f.first_def, f.first_fn, f.nins,
                (size_t)0, f.lead.n, f.preann.n, f.pre.n, f.defs.n, f.libdefs.n, f.libfns.n, f.body.n);
    }
    outw = out.w;
    out.w = NULL;

done:
    free(out.w);
    free((void *)f.ins);
    free(f.ty); free(f.vty); free(f.cval); free(f.isconst);
    free(f.decs); free(f.mdecs);
    free(f.iface); free(f.iforig); free(f.rowmajor); free(f.roworig); free(f.rowmap);
    free(f.attrslot); free(f.attrn); free(f.attrhi); free(f.attrhity); free(f.attrcomp); free(f.attrslotidx); free(f.attrsgeom); free(f.attrorig); free(f.elemslice); free(f.elemidx); free(f.elemstride); free(f.elemorig); free(f.ioiface); free(f.arrlane); free(f.elemreg); free(f.warr);
    free(f.vc);
    free(f.pre.w); free(f.preann.w); free(f.lead.w); free(f.defs.w); free(f.body.w);
    free(f.libdefs.w); free(f.libfns.w); free(f.hdradd.w);
    if (f.failed) {
        if (outw) { free(outw); outw = NULL; }
        if (log && logsz && !log[0]) snprintf(log, logsz, "fp64: lowering failed");
    }
    return outw;
}

#endif /* full implementation */

/* --------------------------------------------------- MSL scalar-param fix -- */

char *mgl_fp64_fix_msl_scalar_params(const char *msl)
{
    if (!msl) return NULL;

    char *fixed = strdup(msl);
    if (!fixed) return NULL;

    /* Rewrite every `thread const <T>& name` parameter of a thread-space
     * function to by-value `<T> name`.  The type must be one identifier
     * (ulong, ulong2, float4, a struct tag, ...); anything more complex
     * (templates, arrays) is left as a reference.  `device const`/`constant`
     * qualifiers never match the "thread const " prefix. */
    char *p = fixed;
    while ((p = strstr(p, "thread const ")) != NULL)
    {
        char *t = p + strlen("thread const ");       /* type name start */
        char *e = t;
        if (*e == '_' || (*e >= 'a' && *e <= 'z') || (*e >= 'A' && *e <= 'Z'))
        {
            while (*e == '_' || (*e >= '0' && *e <= '9') ||
                   (*e >= 'a' && *e <= 'z') || (*e >= 'A' && *e <= 'Z'))
                e++;
            if (e > t && e[0] == '&' && e[1] == ' ')
            {
                /* memmove the type over "thread const ", then the rest
                 * (from the space before the name) over "&" */
                size_t tl = (size_t)(e - t);
                memmove(p, t, tl);
                memmove(p + tl, e + 1, strlen(e + 1) + 1);
                p = p + tl;
                continue;
            }
        }
        p = t;                                       /* not a simple ref: skip */
    }
    return fixed;
}
