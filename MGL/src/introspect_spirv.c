/*
 * introspect_spirv.c
 * MGL — 'introspect' domain: SPIR-V level link processing and reflection.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * At post-link (see program_introspection.c) every stage's SPIR-V is rewritten:
 *
 *  1. Default-block uniforms.  glslang emits every loose non-opaque uniform as its own
 *     UniformConstant variable; SPIRV-Cross turns each into a separate Metal buffer argument,
 *     which cannot scale (16 bindable buffers) and gives no per-program storage.  All of them
 *     are moved into one Uniform-storage Block "mgl_DefaultUniforms" (natural Metal layout, so
 *     SPIRV-Cross emits no padding), loads / access chains are re-pointed into the block.
 *     The block is stored per program and per stage (MGLIProgram.dub[]).
 *  2. Interface locations.  Vertex inputs get their location from layout(location) >
 *     glBindAttribLocation > automatic assignment; fragment outputs from layout >
 *     glBindFragDataLocation[Indexed] (Index decoration -> Metal [[color(n), index(1)]]) >
 *     automatic; varyings without locations are matched by name between stages.
 *  3. Buffer ordering.  The renderer binds buffers in the order UBOs, uniform constants
 *     (default block), SSBOs and uses the list position as Metal buffer index, while
 *     SPIRV-Cross allocates indices in variable declaration order.  Buffer-class global
 *     variables are re-emitted in exactly that order so both agree.
 *
 * The patched words replace Program.spirv[stage].ir and MSL is regenerated through
 * program.c's parseSPIRVShaderToMetal(), so all MSL options remain owned by program.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CoreFoundation/CoreFoundation.h>

#include "spirv.h"
#include "spirv_cross_c.h"

#include "glm_context.h"
#include "buffers.h"
#include "mgl_ext_introspect.h"
#include "mgl_ext_program.h"
#include "mgl_ext_tess.h"        /* mgl_tess_prepare_spirv: isolines test-compile in msl_accepts */
#include <ctype.h>

#include "cts_glsl_compat.h"   /* bool/bvecN block members: GLSL type recovery (cts domain) */


#pragma mark - module model

enum { T_NONE = 0, T_VOID, T_BOOL, T_INT, T_FLOAT, T_VECTOR, T_MATRIX, T_ARRAY, T_RARRAY, T_STRUCT,
       T_POINTER, T_IMAGE, T_SAMPLER, T_SAMPLED_IMAGE, T_FUNCTION, T_OTHER };

typedef struct { uint32_t *w; uint32_t len; uint8_t dead; uint8_t owned; } SInst;

typedef struct { uint32_t key, id; } KV;

typedef struct {
    int kind;
    uint32_t width, sign, elem, count, storage;
    uint32_t dim, depth, arrayed, ms, sampled;
    uint32_t len; /* array length constant id (OpTypeArray) */
    uint32_t *mem; uint32_t nmem;
} SType;

typedef struct { uint32_t id; int32_t m; uint32_t d, lit; size_t inst; } SDec;
typedef struct { uint32_t id, m; const char *s; } SMName;

typedef struct SMod {
    uint32_t version, bound;
    uint32_t hdr[5];
    SInst *in; size_t n, cap;
    SType *t;
    const char **name;
    SMName *mn; size_t nmn, cmn;
    SDec *dec; size_t ndec, cdec;
    uint8_t *isconst; uint32_t *ctype; uint32_t *cval; size_t *cinst;
    uint8_t *isvar; uint32_t *vtype; uint32_t *vsc; uint32_t *vinit; size_t *vinst; uint8_t *vglobal;
    size_t first_func;
    uint32_t model;
    size_t ep_inst;
} SMod;

static void *zalloc(size_t n) { void *p = calloc(1, n ? n : 1); return p; }

static void mod_free(SMod *m)
{
    if (!m) return;
    for (size_t i = 0; i < m->n; i++) if (m->in[i].owned) free(m->in[i].w);
    free(m->in);
    if (m->t) { for (uint32_t i = 0; i < m->bound; i++) free(m->t[i].mem); }
    free(m->t); free(m->name); free(m->mn); free(m->dec);
    free(m->isconst); free(m->ctype); free(m->cval); free(m->cinst);
    free(m->isvar); free(m->vtype); free(m->vsc); free(m->vinit); free(m->vinst); free(m->vglobal);
    free(m);
}

static void add_dec(SMod *m, uint32_t id, int32_t mem, uint32_t d, uint32_t lit, size_t inst)
{
    if (m->ndec == m->cdec) { m->cdec = m->cdec ? m->cdec * 2 : 64; m->dec = realloc(m->dec, m->cdec * sizeof(SDec)); }
    m->dec[m->ndec++] = (SDec){ id, mem, d, lit, inst };
}

static void add_inst_raw(SMod *m, uint32_t *w, uint32_t len, int owned)
{
    if (m->n == m->cap) { m->cap = m->cap ? m->cap * 2 : 256; m->in = realloc(m->in, m->cap * sizeof(SInst)); }
    m->in[m->n++] = (SInst){ w, len, 0, (uint8_t)owned };
}

static SMod *mod_parse(const uint32_t *words, size_t nwords)
{
    if (!words || nwords < 5 || words[0] != 0x07230203) return NULL;
    SMod *m = zalloc(sizeof(SMod));
    memcpy(m->hdr, words, sizeof(m->hdr));
    m->version = words[1];
    m->bound = words[3];
    uint32_t b = m->bound + 1;
    m->t = zalloc(b * sizeof(SType));
    m->name = zalloc(b * sizeof(char *));
    m->isconst = zalloc(b); m->ctype = zalloc(b * 4); m->cval = zalloc(b * 4); m->cinst = zalloc(b * sizeof(size_t));
    m->isvar = zalloc(b); m->vtype = zalloc(b * 4); m->vsc = zalloc(b * 4); m->vinit = zalloc(b * 4);
    m->vinst = zalloc(b * sizeof(size_t)); m->vglobal = zalloc(b);
    m->first_func = (size_t)-1;
    m->ep_inst = (size_t)-1;

    /* private copy of the words so instructions can point into it */
    uint32_t *copy = malloc(nwords * 4);
    memcpy(copy, words, nwords * 4);
    add_inst_raw(m, copy, 0, 1); /* inst 0 owns the copy (len 0 = header placeholder) */

    size_t i = 5;
    while (i < nwords) {
        uint32_t wc = copy[i] >> 16, op = copy[i] & 0xffff;
        if (wc == 0 || i + wc > nwords) { mod_free(m); return NULL; }
        uint32_t *w = &copy[i];
        size_t idx = m->n;
        add_inst_raw(m, w, wc, 0);
#define ID_OK(x) ((x) < b)
        switch (op) {
            case SpvOpName: if (wc > 2 && ID_OK(w[1])) m->name[w[1]] = (const char *)&w[2]; break;
            case SpvOpMemberName:
                if (wc > 3) {
                    if (m->nmn == m->cmn) { m->cmn = m->cmn ? m->cmn * 2 : 64; m->mn = realloc(m->mn, m->cmn * sizeof(SMName)); }
                    m->mn[m->nmn++] = (SMName){ w[1], w[2], (const char *)&w[3] };
                }
                break;
            case SpvOpDecorate: if (wc >= 3) add_dec(m, w[1], -1, w[2], wc > 3 ? w[3] : 0, idx); break;
            case SpvOpMemberDecorate: if (wc >= 4) add_dec(m, w[1], (int32_t)w[2], w[3], wc > 4 ? w[4] : 0, idx); break;
            case SpvOpEntryPoint: if (m->ep_inst == (size_t)-1) { m->model = w[1]; m->ep_inst = idx; } break;
            case SpvOpTypeVoid: if (ID_OK(w[1])) m->t[w[1]].kind = T_VOID; break;
            case SpvOpTypeBool: if (ID_OK(w[1])) { m->t[w[1]].kind = T_BOOL; m->t[w[1]].width = 32; } break;
            case SpvOpTypeInt: if (ID_OK(w[1])) { m->t[w[1]].kind = T_INT; m->t[w[1]].width = w[2]; m->t[w[1]].sign = w[3]; } break;
            case SpvOpTypeFloat: if (ID_OK(w[1])) { m->t[w[1]].kind = T_FLOAT; m->t[w[1]].width = w[2]; } break;
            case SpvOpTypeVector: if (ID_OK(w[1])) { m->t[w[1]].kind = T_VECTOR; m->t[w[1]].elem = w[2]; m->t[w[1]].count = w[3]; } break;
            case SpvOpTypeMatrix: if (ID_OK(w[1])) { m->t[w[1]].kind = T_MATRIX; m->t[w[1]].elem = w[2]; m->t[w[1]].count = w[3]; } break;
            case SpvOpTypeImage:
                if (ID_OK(w[1])) { SType *t = &m->t[w[1]]; t->kind = T_IMAGE; t->elem = w[2]; t->dim = w[3]; t->depth = w[4];
                    t->arrayed = w[5]; t->ms = w[6]; t->sampled = w[7]; }
                break;
            case SpvOpTypeSampler: if (ID_OK(w[1])) m->t[w[1]].kind = T_SAMPLER; break;
            case SpvOpTypeSampledImage: if (ID_OK(w[1])) { m->t[w[1]].kind = T_SAMPLED_IMAGE; m->t[w[1]].elem = w[2]; } break;
            case SpvOpTypeArray:
                if (ID_OK(w[1])) { m->t[w[1]].kind = T_ARRAY; m->t[w[1]].elem = w[2]; m->t[w[1]].len = w[3];
                    m->t[w[1]].count = (ID_OK(w[3]) && m->isconst[w[3]]) ? m->cval[w[3]] : 1; }
                break;
            case SpvOpTypeRuntimeArray: if (ID_OK(w[1])) { m->t[w[1]].kind = T_RARRAY; m->t[w[1]].elem = w[2]; m->t[w[1]].count = 0; } break;
            case SpvOpTypeStruct:
                if (ID_OK(w[1])) { SType *t = &m->t[w[1]]; t->kind = T_STRUCT; t->nmem = wc - 2;
                    t->mem = malloc((wc - 2 + 1) * 4); memcpy(t->mem, &w[2], (wc - 2) * 4); }
                break;
            case SpvOpTypePointer: if (ID_OK(w[1])) { m->t[w[1]].kind = T_POINTER; m->t[w[1]].storage = w[2]; m->t[w[1]].elem = w[3]; } break;
            case SpvOpTypeFunction: if (ID_OK(w[1])) m->t[w[1]].kind = T_FUNCTION; break;
            case SpvOpTypeOpaque: case SpvOpTypeEvent: case SpvOpTypeDeviceEvent: case SpvOpTypeReserveId:
            case SpvOpTypeQueue: case SpvOpTypePipe: case SpvOpTypeForwardPointer:
                if (wc > 1 && ID_OK(w[1])) m->t[w[1]].kind = T_OTHER; break;
            case SpvOpConstant: case SpvOpSpecConstant:
                if (ID_OK(w[2])) { m->isconst[w[2]] = 1; m->ctype[w[2]] = w[1]; m->cval[w[2]] = wc > 3 ? w[3] : 0; m->cinst[w[2]] = idx; }
                break;
            case SpvOpConstantTrue: case SpvOpSpecConstantTrue:
                if (ID_OK(w[2])) { m->isconst[w[2]] = 1; m->ctype[w[2]] = w[1]; m->cval[w[2]] = 1; m->cinst[w[2]] = idx; } break;
            case SpvOpConstantFalse: case SpvOpSpecConstantFalse: case SpvOpConstantNull:
                if (ID_OK(w[2])) { m->isconst[w[2]] = 1; m->ctype[w[2]] = w[1]; m->cval[w[2]] = 0; m->cinst[w[2]] = idx; } break;
            case SpvOpConstantComposite: case SpvOpSpecConstantComposite:
                if (ID_OK(w[2])) { m->isconst[w[2]] = 2; m->ctype[w[2]] = w[1]; m->cinst[w[2]] = idx; } break;
            case SpvOpVariable:
                if (ID_OK(w[2])) { m->isvar[w[2]] = 1; m->vtype[w[2]] = w[1]; m->vsc[w[2]] = w[3];
                    m->vinit[w[2]] = wc > 4 ? w[4] : 0; m->vinst[w[2]] = idx; m->vglobal[w[2]] = m->first_func == (size_t)-1; }
                break;
            case SpvOpFunction: if (m->first_func == (size_t)-1) m->first_func = idx; break;
            default: break;
        }
#undef ID_OK
        i += wc;
    }
    if (m->first_func == (size_t)-1) m->first_func = m->n;
    return m;
}

static bool get_dec(const SMod *m, uint32_t id, int32_t mem, uint32_t d, uint32_t *lit)
{
    for (size_t i = 0; i < m->ndec; i++) {
        const SDec *x = &m->dec[i];
        if (x->id == id && x->m == mem && x->d == d && !m->in[x->inst].dead) { if (lit) *lit = x->lit; return true; }
    }
    return false;
}

static const char *member_name(const SMod *m, uint32_t id, uint32_t mem)
{
    for (size_t i = 0; i < m->nmn; i++) if (m->mn[i].id == id && m->mn[i].m == mem) return m->mn[i].s;
    return "";
}

static const char *id_name(const SMod *m, uint32_t id)
{
    return (id <= m->bound && m->name[id]) ? m->name[id] : "";
}

static uint32_t pointee(const SMod *m, uint32_t ptr_type)
{
    return (ptr_type <= m->bound && m->t[ptr_type].kind == T_POINTER) ? m->t[ptr_type].elem : 0;
}

static uint32_t strip_arrays(const SMod *m, uint32_t t)
{
    while (t <= m->bound && (m->t[t].kind == T_ARRAY || m->t[t].kind == T_RARRAY)) t = m->t[t].elem;
    return t;
}

static bool is_opaque(const SMod *m, uint32_t t)
{
    t = strip_arrays(m, t);
    if (t > m->bound) return false;
    switch (m->t[t].kind) {
        case T_IMAGE: case T_SAMPLER: case T_SAMPLED_IMAGE: case T_OTHER: return true;
        case T_STRUCT:
            for (uint32_t i = 0; i < m->t[t].nmem; i++) if (is_opaque(m, m->t[t].mem[i])) return true;
            return false;
        default: return false;
    }
}

static bool is_builtin_var(const SMod *m, uint32_t var)
{
    if (get_dec(m, var, -1, SpvDecorationBuiltIn, NULL)) return true;
    uint32_t t = strip_arrays(m, pointee(m, m->vtype[var]));
    if (t <= m->bound && m->t[t].kind == T_STRUCT)
        for (uint32_t i = 0; i < m->t[t].nmem; i++)
            if (get_dec(m, t, (int32_t)i, SpvDecorationBuiltIn, NULL)) return true;
    return false;
}

#pragma mark - instruction building

typedef struct { uint32_t *w; size_t n, cap; } WBuf;
static void wb_push(WBuf *b, uint32_t v)
{
    if (b->n == b->cap) { b->cap = b->cap ? b->cap * 2 : 1024; b->w = realloc(b->w, b->cap * 4); }
    b->w[b->n++] = v;
}
static void wb_inst(WBuf *b, const uint32_t *w, uint32_t len) { for (uint32_t i = 0; i < len; i++) wb_push(b, w[i]); }

typedef struct { WBuf names, decs, globals; } Inserts;

static void ins_op(WBuf *b, uint32_t op, int n, const uint32_t *ops)
{
    wb_push(b, ((uint32_t)(n + 1) << 16) | op);
    for (int i = 0; i < n; i++) wb_push(b, ops[i]);
}

static void ins_string_op(WBuf *b, uint32_t op, int n, const uint32_t *ops, const char *s)
{
    size_t sl = strlen(s) + 1, sw = (sl + 3) / 4;
    wb_push(b, ((uint32_t)(n + 1 + sw) << 16) | op);
    for (int i = 0; i < n; i++) wb_push(b, ops[i]);
    uint32_t tmp[64];
    for (size_t k = 0; k < sw; k++) {
        uint32_t v = 0;
        for (int c = 0; c < 4; c++) { size_t at = k * 4 + c; if (at < sl) v |= (uint32_t)(uint8_t)s[at] << (8 * c); }
        if (k < 64) tmp[k] = v;
        wb_push(b, v);
    }
    (void)tmp;
}

static void new_dec(SMod *m, Inserts *ins, uint32_t id, int32_t mem, uint32_t d, int has_lit, uint32_t lit)
{
    if (d == SpvDecorationLocation && getenv("MGL_LOC_DEBUG"))
        fprintf(stderr, "MGL newdec: id=%u mem=%d loc=%u\n", id, mem, lit);
    if (mem < 0) { uint32_t o[3] = { id, d, lit }; ins_op(&ins->decs, SpvOpDecorate, has_lit ? 3 : 2, o); }
    else { uint32_t o[4] = { id, (uint32_t)mem, d, lit }; ins_op(&ins->decs, SpvOpMemberDecorate, has_lit ? 4 : 3, o); }
    /* register for lookups (inst index 0 is never dead) */
    add_dec(m, id, mem, d, lit, 0);
}

static void kill_decs(SMod *m, uint32_t id, int32_t mem, uint32_t d)
{
    for (size_t i = 0; i < m->ndec; i++)
        if (m->dec[i].id == id && m->dec[i].m == mem && m->dec[i].d == d) {
            if (m->dec[i].inst) m->in[m->dec[i].inst].dead = 1;
            else m->dec[i].d = 0xffffffff; /* inserted decoration: hide from lookups */
        }
}

#pragma mark - Metal natural layout

static uint32_t align_up(uint32_t v, uint32_t a) { return a ? (v + a - 1) / a * a : v; }

/* declared with layout(<qual>) in the GLSL source? (defined near the link entry) */
static bool src_explicit_qual(const char *src, const char *name, const char *dir, const char *qual);
static bool src_explicit_location(const char *src, const char *name, const char *dir);

static uint32_t scalar_bytes(const SMod *m, uint32_t t)
{
    if (m->t[t].kind == T_BOOL) return 1;
    return m->t[t].width / 8;
}

/* size/alignment of a type in a Metal constant buffer; honours existing layout decorations */
static void msl_layout(const SMod *m, uint32_t t, uint32_t *size, uint32_t *align)
{
    *size = 0; *align = 1;
    if (t > m->bound) return;
    const SType *ty = &m->t[t];
    switch (ty->kind) {
        case T_BOOL: case T_INT: case T_FLOAT: *size = *align = scalar_bytes(m, t); break;
        case T_VECTOR: {
            uint32_t s = scalar_bytes(m, ty->elem), n = ty->count;
            *size = *align = (n == 3 ? 4 : n) * s;
            break;
        }
        case T_MATRIX: {
            uint32_t cs, ca; msl_layout(m, ty->elem, &cs, &ca);
            *size = ty->count * align_up(cs, ca); *align = ca;
            break;
        }
        case T_ARRAY: case T_RARRAY: {
            uint32_t es, ea, stride; msl_layout(m, ty->elem, &es, &ea);
            if (!get_dec(m, t, -1, SpvDecorationArrayStride, &stride)) stride = align_up(es, ea);
            *size = ty->count * stride; *align = ea;
            break;
        }
        case T_STRUCT: {
            uint32_t off = 0, end = 0, al = 1;
            for (uint32_t i = 0; i < ty->nmem; i++) {
                uint32_t ms, ma, mo;
                msl_layout(m, ty->mem[i], &ms, &ma);
                if (!get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo)) mo = align_up(off, ma);
                off = mo + ms; if (off > end) end = off;
                if (ma > al) al = ma;
            }
            *size = align_up(end, al); *align = al;
            break;
        }
        default: break;
    }
}

/* add natural-layout decorations (Offset / ArrayStride / MatrixStride / ColMajor) */
static void ensure_layout(SMod *m, Inserts *ins, uint32_t t, uint8_t *done);

static void decorate_matrix_member(SMod *m, Inserts *ins, uint32_t st, uint32_t i, uint32_t mt)
{
    uint32_t base = strip_arrays(m, mt);
    if (m->t[base].kind == T_MATRIX && !get_dec(m, st, (int32_t)i, SpvDecorationMatrixStride, NULL)) {
        uint32_t cs, ca; msl_layout(m, m->t[base].elem, &cs, &ca);
        new_dec(m, ins, st, (int32_t)i, SpvDecorationColMajor, 0, 0);
        new_dec(m, ins, st, (int32_t)i, SpvDecorationMatrixStride, 1, align_up(cs, ca));
    }
}

static void ensure_layout(SMod *m, Inserts *ins, uint32_t t, uint8_t *done)
{
    if (t > m->bound || done[t]) return;
    done[t] = 1;
    SType *ty = &m->t[t];
    if (ty->kind == T_ARRAY || ty->kind == T_RARRAY) {
        ensure_layout(m, ins, ty->elem, done);
        if (!get_dec(m, t, -1, SpvDecorationArrayStride, NULL)) {
            uint32_t es, ea; msl_layout(m, ty->elem, &es, &ea);
            new_dec(m, ins, t, -1, SpvDecorationArrayStride, 1, align_up(es, ea));
        }
    } else if (ty->kind == T_STRUCT) {
        uint32_t off = 0;
        for (uint32_t i = 0; i < ty->nmem; i++) {
            ensure_layout(m, ins, ty->mem[i], done);
            uint32_t ms, ma, mo;
            msl_layout(m, ty->mem[i], &ms, &ma);
            if (!get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo)) {
                mo = align_up(off, ma);
                new_dec(m, ins, t, (int32_t)i, SpvDecorationOffset, 1, mo);
            }
            decorate_matrix_member(m, ins, t, i, ty->mem[i]);
            off = mo + ms;
        }
    }
}

/* number of interface locations consumed by a type */
static uint32_t loc_count(const SMod *m, uint32_t t)
{
    if (t > m->bound) return 1;
    const SType *ty = &m->t[t];
    switch (ty->kind) {
        case T_VECTOR: return (scalar_bytes(m, ty->elem) == 8 && ty->count > 2) ? 2 : 1;
        case T_MATRIX: return ty->count * loc_count(m, ty->elem);
        case T_ARRAY: return (ty->count ? ty->count : 1) * loc_count(m, ty->elem);
        case T_STRUCT: { uint32_t n = 0; for (uint32_t i = 0; i < ty->nmem; i++) n += loc_count(m, ty->mem[i]); return n ? n : 1; }
        default: return 1;
    }
}

/* GL-visible location span of a VERTEX INPUT (ARB_vertex_attrib_64bit: "dvec3/dvec4
 * vertex inputs consume one location"; matrix columns use the same generic attributes
 * as the single-precision equivalent, one per column/element).  The SPIR-V Location
 * decoration (loc_count above) stays the internal slot span: two per dvec3/dvec4
 * element, which the fp64 word-pair interface and Metal attribute indices need. */
static uint32_t loc_count_gl(const SMod *m, uint32_t t)
{
    if (t > m->bound) return 1;
    const SType *ty = &m->t[t];
    switch (ty->kind) {
        case T_VECTOR: return 1;
        case T_MATRIX: return ty->count * loc_count_gl(m, ty->elem);
        case T_ARRAY: return (ty->count ? ty->count : 1) * loc_count_gl(m, ty->elem);
        case T_STRUCT: { uint32_t n = 0; for (uint32_t i = 0; i < ty->nmem; i++) n += loc_count_gl(m, ty->mem[i]); return n ? n : 1; }
        default: return 1;
    }
}

#pragma mark - GL type mapping

static GLenum gl_type_of(const SMod *m, uint32_t t)
{
    t = strip_arrays(m, t);
    if (t > m->bound) return GL_NONE;
    const SType *ty = &m->t[t];
    switch (ty->kind) {
        case T_BOOL: return GL_BOOL;
        case T_INT: return ty->sign ? GL_INT : GL_UNSIGNED_INT;
        case T_FLOAT: return ty->width == 64 ? GL_DOUBLE : GL_FLOAT;
        case T_VECTOR: {
            const SType *e = &m->t[ty->elem]; int n = (int)ty->count;
            static const GLenum fv[] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4 };
            static const GLenum dv[] = { GL_DOUBLE, GL_DOUBLE_VEC2, GL_DOUBLE_VEC3, GL_DOUBLE_VEC4 };
            static const GLenum iv[] = { GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4 };
            static const GLenum uv[] = { GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3, GL_UNSIGNED_INT_VEC4 };
            static const GLenum bv[] = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
            if (n < 1 || n > 4) return GL_NONE;
            if (e->kind == T_BOOL) return bv[n - 1];
            if (e->kind == T_INT) return e->sign ? iv[n - 1] : uv[n - 1];
            return e->width == 64 ? dv[n - 1] : fv[n - 1];
        }
        case T_MATRIX: {
            const SType *col = &m->t[ty->elem];
            int c = (int)ty->count, r = (int)col->count;
            bool d = m->t[col->elem].width == 64;
            static const GLenum fm[3][3] = { { GL_FLOAT_MAT2, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4 },
                                             { GL_FLOAT_MAT3x2, GL_FLOAT_MAT3, GL_FLOAT_MAT3x4 },
                                             { GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, GL_FLOAT_MAT4 } };
            static const GLenum dm[3][3] = { { GL_DOUBLE_MAT2, GL_DOUBLE_MAT2x3, GL_DOUBLE_MAT2x4 },
                                             { GL_DOUBLE_MAT3x2, GL_DOUBLE_MAT3, GL_DOUBLE_MAT3x4 },
                                             { GL_DOUBLE_MAT4x2, GL_DOUBLE_MAT4x3, GL_DOUBLE_MAT4 } };
            if (c < 2 || c > 4 || r < 2 || r > 4) return GL_NONE;
            return d ? dm[c - 2][r - 2] : fm[c - 2][r - 2];
        }
        case T_SAMPLED_IMAGE: case T_IMAGE: {
            const SType *img = ty->kind == T_SAMPLED_IMAGE ? &m->t[ty->elem] : ty;
            const SType *st = &m->t[img->elem];
            int k = (st->kind == T_INT) ? (st->sign ? 1 : 2) : 0; /* 0 float, 1 int, 2 uint */
            bool storage = ty->kind == T_IMAGE && img->sampled == 2;
            bool sh = img->depth == 1 && !storage && k == 0;
            bool ar = img->arrayed, ms = img->ms;
#define PICK(f, i, u) (k == 0 ? (f) : k == 1 ? (i) : (u))
            if (!storage) {
                switch (img->dim) {
                    case SpvDim1D: return ar ? (sh ? GL_SAMPLER_1D_ARRAY_SHADOW : PICK(GL_SAMPLER_1D_ARRAY, GL_INT_SAMPLER_1D_ARRAY, GL_UNSIGNED_INT_SAMPLER_1D_ARRAY))
                                             : (sh ? GL_SAMPLER_1D_SHADOW : PICK(GL_SAMPLER_1D, GL_INT_SAMPLER_1D, GL_UNSIGNED_INT_SAMPLER_1D));
                    case SpvDim2D:
                        if (ms) return ar ? PICK(GL_SAMPLER_2D_MULTISAMPLE_ARRAY, GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY, GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY)
                                          : PICK(GL_SAMPLER_2D_MULTISAMPLE, GL_INT_SAMPLER_2D_MULTISAMPLE, GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE);
                        return ar ? (sh ? GL_SAMPLER_2D_ARRAY_SHADOW : PICK(GL_SAMPLER_2D_ARRAY, GL_INT_SAMPLER_2D_ARRAY, GL_UNSIGNED_INT_SAMPLER_2D_ARRAY))
                                  : (sh ? GL_SAMPLER_2D_SHADOW : PICK(GL_SAMPLER_2D, GL_INT_SAMPLER_2D, GL_UNSIGNED_INT_SAMPLER_2D));
                    case SpvDim3D: return PICK(GL_SAMPLER_3D, GL_INT_SAMPLER_3D, GL_UNSIGNED_INT_SAMPLER_3D);
                    case SpvDimCube: return ar ? (sh ? GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW : PICK(GL_SAMPLER_CUBE_MAP_ARRAY, GL_INT_SAMPLER_CUBE_MAP_ARRAY, GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY))
                                               : (sh ? GL_SAMPLER_CUBE_SHADOW : PICK(GL_SAMPLER_CUBE, GL_INT_SAMPLER_CUBE, GL_UNSIGNED_INT_SAMPLER_CUBE));
                    case SpvDimRect: return sh ? GL_SAMPLER_2D_RECT_SHADOW : PICK(GL_SAMPLER_2D_RECT, GL_INT_SAMPLER_2D_RECT, GL_UNSIGNED_INT_SAMPLER_2D_RECT);
                    case SpvDimBuffer: return PICK(GL_SAMPLER_BUFFER, GL_INT_SAMPLER_BUFFER, GL_UNSIGNED_INT_SAMPLER_BUFFER);
                    default: return GL_NONE;
                }
            }
            switch (img->dim) {
                case SpvDim1D: return ar ? PICK(GL_IMAGE_1D_ARRAY, GL_INT_IMAGE_1D_ARRAY, GL_UNSIGNED_INT_IMAGE_1D_ARRAY) : PICK(GL_IMAGE_1D, GL_INT_IMAGE_1D, GL_UNSIGNED_INT_IMAGE_1D);
                case SpvDim2D:
                    if (ms) return ar ? PICK(GL_IMAGE_2D_MULTISAMPLE_ARRAY, GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY, GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY)
                                      : PICK(GL_IMAGE_2D_MULTISAMPLE, GL_INT_IMAGE_2D_MULTISAMPLE, GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE);
                    return ar ? PICK(GL_IMAGE_2D_ARRAY, GL_INT_IMAGE_2D_ARRAY, GL_UNSIGNED_INT_IMAGE_2D_ARRAY) : PICK(GL_IMAGE_2D, GL_INT_IMAGE_2D, GL_UNSIGNED_INT_IMAGE_2D);
                case SpvDim3D: return PICK(GL_IMAGE_3D, GL_INT_IMAGE_3D, GL_UNSIGNED_INT_IMAGE_3D);
                case SpvDimCube: return ar ? PICK(GL_IMAGE_CUBE_MAP_ARRAY, GL_INT_IMAGE_CUBE_MAP_ARRAY, GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY) : PICK(GL_IMAGE_CUBE, GL_INT_IMAGE_CUBE, GL_UNSIGNED_INT_IMAGE_CUBE);
                case SpvDimRect: return PICK(GL_IMAGE_2D_RECT, GL_INT_IMAGE_2D_RECT, GL_UNSIGNED_INT_IMAGE_2D_RECT);
                case SpvDimBuffer: return PICK(GL_IMAGE_BUFFER, GL_INT_IMAGE_BUFFER, GL_UNSIGNED_INT_IMAGE_BUFFER);
                default: return GL_NONE;
            }
#undef PICK
        }
        default: return GL_NONE;
    }
}

static void scalar_info(const SMod *m, uint32_t t, int *kind, int *cols, int *rows)
{
    t = strip_arrays(m, t);
    const SType *ty = &m->t[t];
    uint32_t s = t;
    *cols = 1; *rows = 1;
    if (ty->kind == T_MATRIX) { *cols = (int)ty->count; s = ty->elem; *rows = (int)m->t[s].count; s = m->t[s].elem; }
    else if (ty->kind == T_VECTOR) { *rows = (int)ty->count; s = ty->elem; }
    const SType *st = &m->t[s];
    if (st->kind == T_BOOL) *kind = MGLI_K_BOOL;
    else if (st->kind == T_INT) *kind = st->sign ? MGLI_K_INT : MGLI_K_UINT;
    else if (st->kind == T_FLOAT) *kind = st->width == 64 ? MGLI_K_DOUBLE : MGLI_K_FLOAT;
    else *kind = MGLI_K_OPAQUE;
}

#pragma mark - per-stage link context

typedef struct {
    uint32_t var;        /* original variable id */
    uint32_t type;       /* pointee type (as declared) */
    uint32_t gtype;      /* GPU storage type: bool->int, fp64->fp32 (== type unless rewritten) */
    uint32_t member;     /* member index in the default block */
    uint32_t offset;
    int      explicit_loc;
    uint8_t *leaf;       /* per leaf uniform (walk_default order): LEAF_WAS_* */
    int      nleaf;
    char     name[256];
} DubMember;

typedef struct {
    int       present;
    SMod     *m;               /* original module (patched in place) */
    SMod     *r;               /* re-parsed patched module (reflection) */
    DubMember *dub; uint32_t ndub;
    uint32_t  dub_size;
    uint32_t  dub_var;         /* id of the default block variable in r */
#define MGL_BUSED_UNKNOWN UINT64_MAX
    uint8_t  *active;          /* [bound+1] of r: active global variables */
    uint8_t  *dub_active;      /* [ndub] members actually accessed */
    uint64_t *bused;           /* [bound+1] of r: per UBO/SSBO variable, bitmask of
                                  top-level members accessed in this stage
                                  (SPIRV-Cross active buffer ranges); MGL_BUSED_UNKNOWN
                                  = usage unknown → declaration-based answer */
    /* location assignments to apply: var -> loc / index */
    struct { uint32_t var; uint32_t loc; int index; } *locs; int nlocs;
    MGLIProgram *ip;
    int       no_pack;
} StageCtx;

static void stage_set_loc(StageCtx *s, uint32_t var, uint32_t loc, int index)
{
    if (getenv("MGL_LOC_DEBUG")) fprintf(stderr, "MGL loc: set var=%u loc=%u ix=%d\n", var, loc, index);
    s->locs = realloc(s->locs, (s->nlocs + 1) * sizeof(*s->locs));
    s->locs[s->nlocs++] = (typeof(*s->locs)){ var, loc, index };
}

/* location bitmap helpers */
typedef struct { uint8_t used[1024]; uint32_t type[1024]; } LocMap;
static bool loc_free(const LocMap *lm, uint32_t base, uint32_t n)
{
    if (base + n > 1024) return false;
    for (uint32_t i = 0; i < n; i++) if (lm->used[base + i]) return false;
    return true;
}
static void loc_take(LocMap *lm, uint32_t base, uint32_t n) { for (uint32_t i = 0; i < n && base + i < 1024; i++) lm->used[base + i] = 1; }
static bool loc_taken(const LocMap *lm, uint32_t base, uint32_t n)
{ for (uint32_t i = 0; i < n && base + i < 1024; i++) if (lm->used[base + i]) return true; return false; }
static uint32_t loc_alloc(LocMap *lm, uint32_t n) { for (uint32_t b = 0; b < 1024; b++) if (loc_free(lm, b, n)) { loc_take(lm, b, n); return b; } return 0; }

/* compat r96e: base (scalar element) type id of a vertex-input type, for the
 * aliasing rule — variables sharing a location must have the same base type.
 * Unwraps arrays/pointers, matrices align like their column vector. */
static uint32_t attr_base_type(const SMod *m, uint32_t tid)
{
    uint32_t t = pointee(m, tid);
    for (int guard = 0; guard < 16 && t <= m->bound; guard++)
    {
        SType *ty = &m->t[t];
        if (ty->kind == T_ARRAY) { t = ty->elem; continue; }
        if (ty->kind == T_MATRIX) { t = ty->elem; continue; }
        if (ty->kind == T_VECTOR) { t = ty->elem; continue; }
        break;
    }
    return t;
}
static uint32_t attr_base_at(const SMod *m, const LocMap *lm, uint32_t gl)
{
    (void)m;
    return gl < 1024 ? lm->type[gl] : 0;
}

static bool stage_is_arrayed_io(int stage, bool input)
{
    if (stage == _GEOMETRY_SHADER || stage == _TESS_EVALUATION_SHADER) return input;
    if (stage == _TESS_CONTROL_SHADER) return true;
    return false;
}

static uint32_t io_loc_count(const SMod *m, int stage, uint32_t var, bool input)
{
    uint32_t t = pointee(m, m->vtype[var]);
    if (stage_is_arrayed_io(stage, input) && !get_dec(m, var, -1, SpvDecorationPatch, NULL) &&
        (m->t[t].kind == T_ARRAY || m->t[t].kind == T_RARRAY))
        t = m->t[t].elem;
    return loc_count(m, t);
}

/* GL name used to match an interface variable between stages */
static const char *io_match_name(const SMod *m, uint32_t var)
{
    uint32_t t = strip_arrays(m, pointee(m, m->vtype[var]));
    if (m->t[t].kind == T_STRUCT && get_dec(m, t, -1, SpvDecorationBlock, NULL)) return id_name(m, t);
    return id_name(m, var);
}

static void assign_interface_locations(StageCtx *sc, MGLIProgram *ip, const char *src)
{
    /* vertex inputs */
    if (sc[_VERTEX_SHADER].present) {
        SMod *m = sc[_VERTEX_SHADER].m;
        LocMap lm, lmg; memset(&lm, 0, sizeof lm); memset(&lmg, 0, sizeof lmg);
        /* per link: GL locations (lmg) and internal slots (lm) are (re)assigned;
           the SPIR-V Location decoration carries the internal slot (two per
           dvec3/dvec4 element for the fp64 word-pair interface), the GL location
           (one per element/column, ARB_vertex_attrib_64bit) is remembered in
           ip->attrib_gl and reported by introspection */
        free(ip->attrib_gl); ip->attrib_gl = NULL; ip->nattrib_gl = 0;
        struct { uint32_t v, gl, ng, n; } *vins = NULL;
        size_t nvins = 0;
        for (int pass = 0; pass < 3; pass++)
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != SpvStorageClassInput || is_builtin_var(m, v)) continue;
                uint32_t l, n = io_loc_count(m, _VERTEX_SHADER, v, true);
                uint32_t ng = loc_count_gl(m, pointee(m, m->vtype[v]));
                bool eloc = !!get_dec(m, v, -1, SpvDecorationLocation, &l);
                int bound = -1;
                if (!eloc)
                    for (int b = 0; b < ip->nattrib_binds; b++) if (!strcmp(ip->attrib_binds[b].name, id_name(m, v))) bound = (int)ip->attrib_binds[b].index;
                if (pass == 0 && !eloc) continue;
                if (pass == 1 && (eloc || bound < 0)) continue;   /* glBindAttribLocation */
                if (pass == 2 && (eloc || bound >= 0)) continue;  /* automatic */
                uint32_t gl;
                bool gl_fixed;   /* explicit / bound: the base was chosen, not allocated */
                if (eloc) { gl = l; gl_fixed = true; }
                else if (bound >= 0) { gl = (uint32_t)bound; gl_fixed = true; }
                else { gl = loc_alloc(&lmg, ng); gl_fixed = false; }
                /* compat r96e: vertex inputs may ALIAS — variables of the same base
                 * type may share a location and component range (GL 4.6 11.1.1;
                 * KHR-Single-GL46.enhanced_layouts.input_component_aliasing* expects
                 * the vertex variant to LINK).  Conflict only on overlapping
                 * components of a DIFFERENT base type. */
                if (gl_fixed && loc_taken(&lmg, gl, ng) && attr_base_type(m, m->vtype[v]) != attr_base_at(m, &lmg, gl)) {
                    ip->attrib_conflict = 1;
                    if (getenv("MGL_LOC_DEBUG")) fprintf(stderr, "MGL loc: conflict var=%u name=%s gl=%u ng=%u n=%u pass=%d\n", v, id_name(m, v), gl, ng, n, pass);
                }
                loc_take(&lmg, gl, ng);
                if (gl < 1024) lmg.type[gl] = attr_base_type(m, m->vtype[v]);
                vins = realloc(vins, (nvins + 1) * sizeof *vins);
                vins[nvins].v = v; vins[nvins].gl = gl; vins[nvins].ng = ng; vins[nvins].n = n;
                nvins++;
            }
        /* internal slot assignment, AFTER all GL locations are known: a double
           attribute needs n internal slots for ng GL locations (n = 2*ng for
           dvec3/dvec4 columns).  "Prefer the GL location" only works when the
           span fits (n <= ng) — otherwise it fragments [0, MAX_ATTRIBS) and a
           full 16-slot set overflows (KHR-GL46.vertex_attrib_64bit.limits_test).
           Wide spans are therefore packed first-fit first, everything else
           keeps its GL index when free. */
        for (int wide = 1; wide >= 0; wide--) {
            for (size_t i = 0; i < nvins; i++) {
                uint32_t v = vins[i].v, gl = vins[i].gl, ng = vins[i].ng, n = vins[i].n;
                uint32_t slot;
                if ((n > ng) != (wide == 1)) continue;
                if (n <= ng && loc_free(&lm, gl, n)) { slot = gl; loc_take(&lm, slot, n); }
                else { slot = loc_alloc(&lm, n); }   /* loc_alloc takes the range */
                if (slot + n > MAX_ATTRIBS || gl + ng > MAX_ATTRIBS) ip->attrib_overflow = 1;
                if (getenv("MGL_LOC_DEBUG")) fprintf(stderr, "MGL loc: attr var=%u name=%s gl=%u ng=%u slot=%u n=%u wide=%d\n", v, id_name(m, v), gl, ng, slot, n, wide);
                stage_set_loc(&sc[_VERTEX_SHADER], v, slot, -1);
                ip->attrib_gl = realloc(ip->attrib_gl, (size_t)(ip->nattrib_gl + 1) * sizeof(*ip->attrib_gl));
                ip->attrib_gl[ip->nattrib_gl].var = v;
                ip->attrib_gl[ip->nattrib_gl].gl = (GLint)gl;
                ip->nattrib_gl++;
            }
        }
        free(vins);
    }
    /* fragment outputs */
    if (sc[_FRAGMENT_SHADER].present) {
        SMod *m = sc[_FRAGMENT_SHADER].m;
        LocMap lm[2]; memset(lm, 0, sizeof lm);
        for (uint32_t v = 1; v <= m->bound; v++) {
            if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != SpvStorageClassOutput || is_builtin_var(m, v)) continue;
            uint32_t l, ix = 0; get_dec(m, v, -1, SpvDecorationIndex, &ix);
            if (get_dec(m, v, -1, SpvDecorationLocation, &l)) loc_take(&lm[ix ? 1 : 0], l, loc_count(m, pointee(m, m->vtype[v])));
        }
        for (int pass = 0; pass < 2; pass++)
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != SpvStorageClassOutput || is_builtin_var(m, v)) continue;
                if (get_dec(m, v, -1, SpvDecorationLocation, NULL)) continue;
                uint32_t n = loc_count(m, pointee(m, m->vtype[v]));
                int bound = -1, bidx = 0;
                for (int b = 0; b < ip->nfrag_binds; b++)
                    if (!strcmp(ip->frag_binds[b].name, id_name(m, v))) { bound = (int)ip->frag_binds[b].color; bidx = (int)ip->frag_binds[b].index; }
                if (pass == 0 && bound >= 0) { loc_take(&lm[bidx ? 1 : 0], (uint32_t)bound, n); stage_set_loc(&sc[_FRAGMENT_SHADER], v, (uint32_t)bound, bidx); }
                if (pass == 1 && bound < 0) stage_set_loc(&sc[_FRAGMENT_SHADER], v, loc_alloc(&lm[0], n), -1);
            }
    }
    /* varying interface of the stage being patched (mgl_introspect_patch_stage runs
     * per stage, so this is the ONLY place the stage's own interface numbers are
     * fixed up).  glslang's I/O mapper numbers location-less varyings sequentially
     * and IGNORES the width of explicitly-located matrices/arrays:
     *   layout(location=4) centroid out mat2x3 vtxVar4;   // spans locations 4..5
     *   flat out uvec3 var5;                              // glslang invents 5!
     * SPIRV-Cross splits the matrix into per-column out-struct members, so the
     * module has two outputs at locn5 and Metal rejects the function
     * ("duplicated user-defined name 'locn5' for vertex output declaration",
     * dEQP-GL45-ES31 separate_shader.random.116 / pipeline.different_uniform_*
     * link failures).  GL assigns location-less varyings at link time without
     * overlap; redo the invented numbers: an explicit (source layout(location))
     * variable keeps its location, an invented one keeps glslang's number when
     * the span fits and otherwise takes the first free span.  Declaration order
     * is glslang's numbering order, so a collision-free interface keeps every
     * number (no change), and the walk is deterministic in the variable set, so
     * a separable producer-only and consumer-only program whose interfaces match
     * still land on the same numbers and link up at draw time. */
    for (int st = _VERTEX_SHADER; st <= _FRAGMENT_SHADER; st++) {
        if (!sc[st].present) continue;
        SMod *m = sc[st].m;
        for (int input_side = 0; input_side < 2; input_side++) {
            bool input = input_side != 0;
            if (input && st == _VERTEX_SHADER) continue;    /* VS inputs: pass above */
            if (!input && st == _FRAGMENT_SHADER) continue; /* FS outputs: pass above */
            uint32_t want = input ? SpvStorageClassInput : SpvStorageClassOutput;
            LocMap lm; memset(&lm, 0, sizeof lm);
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != want || is_builtin_var(m, v)) continue;
                uint32_t l;
                if (!get_dec(m, v, -1, SpvDecorationLocation, &l)) continue;
                if (!src_explicit_location(src, io_match_name(m, v), input ? "in" : "out")) continue;
                loc_take(&lm, l, io_loc_count(m, st, v, input));
            }
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != want || is_builtin_var(m, v)) continue;
                uint32_t l = 0, n = io_loc_count(m, st, v, input);
                bool has = get_dec(m, v, -1, SpvDecorationLocation, &l) != 0;
                if (has && src_explicit_location(src, io_match_name(m, v), input ? "in" : "out")) continue;
                uint32_t keep = (has && loc_free(&lm, l, n)) ? l : loc_alloc(&lm, n);
                loc_take(&lm, keep, n);
                if (!has || keep != l) stage_set_loc(&sc[st], v, keep, -1);
            }
        }
    }

    /* varyings between consecutive graphics stages */
    int prev = -1;
    for (int st = _VERTEX_SHADER; st <= _FRAGMENT_SHADER; st++) {
        if (!sc[st].present) continue;
        if (prev >= 0) {
            SMod *a = sc[prev].m, *b = sc[st].m;
            LocMap lm; memset(&lm, 0, sizeof lm);
            for (uint32_t v = 1; v <= a->bound; v++) {
                if (!a->isvar[v] || !a->vglobal[v] || a->vsc[v] != SpvStorageClassOutput || is_builtin_var(a, v)) continue;
                uint32_t l; if (get_dec(a, v, -1, SpvDecorationLocation, &l)) loc_take(&lm, l, io_loc_count(a, prev, v, false));
            }
            for (uint32_t v = 1; v <= b->bound; v++) {
                if (!b->isvar[v] || !b->vglobal[v] || b->vsc[v] != SpvStorageClassInput || is_builtin_var(b, v)) continue;
                uint32_t l; if (get_dec(b, v, -1, SpvDecorationLocation, &l)) loc_take(&lm, l, io_loc_count(b, st, v, true));
            }
            /* consumer inputs without location: take the producer's location or a new shared one */
            for (uint32_t v = 1; v <= b->bound; v++) {
                if (!b->isvar[v] || !b->vglobal[v] || b->vsc[v] != SpvStorageClassInput || is_builtin_var(b, v)) continue;
                if (get_dec(b, v, -1, SpvDecorationLocation, NULL)) continue;
                const char *nm = io_match_name(b, v);
                uint32_t match = 0;
                for (uint32_t u = 1; u <= a->bound && !match; u++)
                    if (a->isvar[u] && a->vglobal[u] && a->vsc[u] == SpvStorageClassOutput && !is_builtin_var(a, u) &&
                        !strcmp(io_match_name(a, u), nm)) match = u;
                uint32_t l;
                if (match && get_dec(a, match, -1, SpvDecorationLocation, &l)) { stage_set_loc(&sc[st], v, l, -1); continue; }
                bool assigned = false;
                if (match) for (int k = 0; k < sc[prev].nlocs; k++) if (sc[prev].locs[k].var == match) { stage_set_loc(&sc[st], v, sc[prev].locs[k].loc, -1); assigned = true; }
                if (assigned) continue;
                uint32_t n = io_loc_count(b, st, v, true);
                if (match) { uint32_t n2 = io_loc_count(a, prev, match, false); if (n2 > n) n = n2; }
                l = loc_alloc(&lm, n);
                stage_set_loc(&sc[st], v, l, -1);
                if (match) stage_set_loc(&sc[prev], match, l, -1);
            }
            /* remaining producer outputs */
            for (uint32_t u = 1; u <= a->bound; u++) {
                if (!a->isvar[u] || !a->vglobal[u] || a->vsc[u] != SpvStorageClassOutput || is_builtin_var(a, u)) continue;
                if (get_dec(a, u, -1, SpvDecorationLocation, NULL)) continue;
                bool done = false;
                for (int k = 0; k < sc[prev].nlocs; k++) if (sc[prev].locs[k].var == u) done = true;
                if (!done) stage_set_loc(&sc[prev], u, loc_alloc(&lm, io_loc_count(a, prev, u, false)), -1);
            }
        }
        prev = st;
    }
}

#pragma mark - SPIR-V transform


static void mgrow(SMod *m, uint32_t id); /* id tables grow when types are inserted */

static uint32_t find_or_add_type_int(SMod *m, Inserts *ins, uint32_t *next_id)
{
    for (uint32_t i = 1; i <= m->bound; i++) if (m->t[i].kind == T_INT && m->t[i].width == 32 && m->t[i].sign == 1) return i;
    uint32_t id = (*next_id)++;
    uint32_t o[3] = { id, 32, 1 };
    ins_op(&ins->globals, SpvOpTypeInt, 3, o);
    mgrow(m, id);
    m->t[id].kind = T_INT; m->t[id].width = 32; m->t[id].sign = 1;
    return id;
}

static uint32_t find_or_add_type_float(SMod *m, Inserts *ins, uint32_t *next_id)
{
    for (uint32_t i = 1; i <= m->bound; i++) if (m->t[i].kind == T_FLOAT && m->t[i].width == 32) return i;
    uint32_t id = (*next_id)++;
    uint32_t o[2] = { id, 32 };
    ins_op(&ins->globals, SpvOpTypeFloat, 2, o);
    mgrow(m, id);
    m->t[id].kind = T_FLOAT; m->t[id].width = 32;
    return id;
}

/* MSL cannot lay `bool` out in buffers (natural alignment 0) and has no fp64 at all, so
 * default-block members get their storage type rewritten:  bool/bvec -> int/ivec,
 * double/dvec/dmat -> float/vec/mat (values are converted in the CPU-side shadow buffer).
 * The ORIGINAL type is remembered per leaf so GL introspection still reports GL_BOOL/GL_DOUBLE. */

enum { LEAF_AS_IS = 0, LEAF_WAS_BOOL = 1, LEAF_WAS_DOUBLE = 2 };

static uint8_t leaf_kind_of(const SMod *m, uint32_t t)
{
    t = strip_arrays(m, t);
    if (t > m->bound) return LEAF_AS_IS;
    const SType *ty = &m->t[t];
    if (ty->kind == T_BOOL) return LEAF_WAS_BOOL;
    if (ty->kind == T_FLOAT) return ty->width == 64 ? LEAF_WAS_DOUBLE : LEAF_AS_IS;
    if (ty->kind == T_VECTOR) {
        const SType *e = &m->t[ty->elem];
        if (e->kind == T_BOOL) return LEAF_WAS_BOOL;
        return (e->kind == T_FLOAT && e->width == 64) ? LEAF_WAS_DOUBLE : LEAF_AS_IS;
    }
    if (ty->kind == T_MATRIX) {
        const SType *e = &m->t[m->t[ty->elem].elem];
        return (e->kind == T_FLOAT && e->width == 64) ? LEAF_WAS_DOUBLE : LEAF_AS_IS;
    }
    return LEAF_AS_IS;
}


/* enumerate leaf uniforms in the exact order walk_default() creates them */
static void record_leaf_kinds(const SMod *m, uint32_t t, uint8_t **out, int *n)
{
    const SType *ty = &m->t[t];
    if (ty->kind == T_ARRAY || ty->kind == T_RARRAY) {
        uint32_t et = ty->elem;
        int base = m->t[et].kind != T_ARRAY && m->t[et].kind != T_RARRAY;
        if (base && m->t[et].kind != T_STRUCT) {
            *out = realloc(*out, (size_t)(*n + 1)); (*out)[(*n)++] = leaf_kind_of(m, et);
            return;
        }
        for (uint32_t i = 0; i < (ty->count ? ty->count : 1); i++) record_leaf_kinds(m, et, out, n);
        return;
    }
    if (ty->kind == T_STRUCT) {
        for (uint32_t i = 0; i < ty->nmem; i++) record_leaf_kinds(m, ty->mem[i], out, n);
        return;
    }
    *out = realloc(*out, (size_t)(*n + 1)); (*out)[(*n)++] = leaf_kind_of(m, t);
}

/* grow the id-indexed tables so freshly inserted types are usable by msl_layout()/get_dec() */
static void grow_arr(void **p, uint32_t old_n, uint32_t new_n, size_t es)
{
    *p = realloc(*p, (size_t)new_n * es);
    memset((char *)*p + old_n * es, 0, (size_t)(new_n - old_n) * es);
}

static void mgrow(SMod *m, uint32_t id)
{
    if (id < m->bound) return;
    uint32_t old_b = m->bound + 1;
    uint32_t nb = m->bound + 1;
    while (nb <= id) nb *= 2;
    grow_arr((void **)&m->t, old_b, nb, sizeof(SType));
    grow_arr((void **)&m->name, old_b, nb, sizeof(char *));
    grow_arr((void **)&m->isconst, old_b, nb, 1);
    grow_arr((void **)&m->ctype, old_b, nb, 4);
    grow_arr((void **)&m->cval, old_b, nb, 4);
    grow_arr((void **)&m->cinst, old_b, nb, sizeof(size_t));
    grow_arr((void **)&m->isvar, old_b, nb, 1);
    grow_arr((void **)&m->vtype, old_b, nb, 4);
    grow_arr((void **)&m->vsc, old_b, nb, 4);
    grow_arr((void **)&m->vinit, old_b, nb, 4);
    grow_arr((void **)&m->vinst, old_b, nb, sizeof(size_t));
    grow_arr((void **)&m->vglobal, old_b, nb, 1);
    m->bound = nb - 1;
}

/* insert a copy of type `t` with bool -> int and fp64 -> fp32 leaves; returns the new type id
 * (or `t` when nothing changed).  Inserted types are registered in the module model so layout
 * computation sees them. */
static uint32_t map_gpu_type(SMod *m, Inserts *ins, uint32_t *next_id, uint32_t t,
                             uint32_t int_t, uint32_t float_t, KV **cache, int *ncache)
{
    if (t > m->bound || !int_t) return t;
    for (int i = 0; i < *ncache; i++) if ((*cache)[i].key == t) return (*cache)[i].id;
    uint32_t r = t;
    SType *ty = &m->t[t];
    if (ty->kind == T_BOOL) r = int_t;
    else if (ty->kind == T_VECTOR || ty->kind == T_MATRIX) {
        uint32_t e = map_gpu_type(m, ins, next_id, ty->elem, int_t, float_t, cache, ncache);
        ty = &m->t[t]; /* recursion may have grown (realloc'd) the type table */
        if (e != ty->elem) {
            r = (*next_id)++;
            uint32_t o[3] = { r, e, ty->count };
            ins_op(&ins->globals, ty->kind == T_VECTOR ? SpvOpTypeVector : SpvOpTypeMatrix, 3, o);
            mgrow(m, r);
            m->t[r] = *ty; m->t[r].elem = e;
        }
    } else if (ty->kind == T_ARRAY || ty->kind == T_RARRAY) {
        uint32_t e = map_gpu_type(m, ins, next_id, ty->elem, int_t, float_t, cache, ncache);
        ty = &m->t[t]; /* recursion may have grown (realloc'd) the type table */
        if (e != ty->elem) {
            uint32_t len = ty->len, kind = ty->kind;
            r = (*next_id)++;
            if (kind == T_ARRAY) { uint32_t o[3] = { r, e, len }; ins_op(&ins->globals, SpvOpTypeArray, 3, o); }
            else { uint32_t o[2] = { r, e }; ins_op(&ins->globals, SpvOpTypeRuntimeArray, 2, o); }
            mgrow(m, r);
            m->t[r] = *ty; m->t[r].elem = e;
        }
    } else if (ty->kind == T_STRUCT && ty->nmem) {
        uint32_t nmem = ty->nmem;
        uint32_t *nm = alloca(nmem * sizeof(uint32_t));
        int any = 0;
        for (uint32_t i = 0; i < nmem; i++) {
            nm[i] = map_gpu_type(m, ins, next_id, m->t[t].mem[i], int_t, float_t, cache, ncache);
            any |= nm[i] != m->t[t].mem[i];
        }
        if (any) {
            ty = &m->t[t];
            r = (*next_id)++;
            uint32_t *o = alloca((nmem + 1) * sizeof(uint32_t));
            o[0] = r;
            for (uint32_t i = 0; i < nmem; i++) o[i + 1] = nm[i];
            ins_op(&ins->globals, SpvOpTypeStruct, (int)nmem + 1, o);
            mgrow(m, r);
            m->t[r] = *ty; m->t[r].nmem = nmem;
            m->t[r].mem = malloc((nmem + 1) * sizeof(uint32_t));
            memcpy(m->t[r].mem, nm, nmem * sizeof(uint32_t));
            /* keep names: reflection reads them back from the patched module */
            ins_string_op(&ins->names, SpvOpName, 1, &r, id_name(m, t));
            for (uint32_t i = 0; i < nmem; i++) {
                uint32_t o2[2] = { r, i };
                ins_string_op(&ins->names, SpvOpMemberName, 2, o2, member_name(m, t, i));
            }
        }
    }
    *cache = realloc(*cache, (size_t)(*ncache + 1) * sizeof(KV));
    (*cache)[*ncache] = (KV){ t, r }; (*ncache)++;
    return r;
}


static uint32_t kv_get(KV *kv, int n, uint32_t key) { for (int i = 0; i < n; i++) if (kv[i].key == key) return kv[i].id; return 0; }

static void flatten_const(const SMod *m, uint32_t cid, uint32_t **vals, int *n, int *wide, int depth)
{
    if (cid > m->bound || !m->isconst[cid] || depth > 16) return;
    uint32_t *w = m->in[m->cinst[cid]].w; uint32_t wc = m->in[m->cinst[cid]].len;
    uint32_t op = w[0] & 0xffff;
    if (m->isconst[cid] == 1) {
        const SType *ty = &m->t[m->ctype[cid]];
        int k = (ty->kind == T_FLOAT || ty->kind == T_INT) && ty->width == 64 ? 2 : 1;
        if (k == 2) *wide = 1;
        *vals = realloc(*vals, (size_t)(*n + k) * 4);
        if (op == SpvOpConstant) { (*vals)[(*n)++] = w[3]; if (k == 2) (*vals)[(*n)++] = wc > 4 ? w[4] : 0; }
        else { (*vals)[(*n)++] = m->cval[cid]; if (k == 2) (*vals)[(*n)++] = 0; }
        return;
    }
    for (uint32_t a = 3; a < wc; a++) flatten_const(m, w[a], vals, n, wide, depth + 1);
}

void mgli_free_inits(MGLIProgram *ip)
{
    for (int s = 0; s < MGLI_STAGES; s++) {
        for (int i = 0; i < ip->ninits[s]; i++) { free(ip->inits[s][i].name); free(ip->inits[s][i].vals); }
        free(ip->inits[s]); ip->inits[s] = NULL; ip->ninits[s] = 0;
    }
}

/* rewrite one stage; returns patched words (malloc) */
/* zero of any type (for int != 0 style bool conversion) */
static uint32_t null_const(SMod *m, Inserts *ins, uint32_t *next_id, uint32_t t)
{
    uint32_t id = (*next_id)++;
    uint32_t o[2] = { t, id };
    ins_op(&ins->globals, SpvOpConstantNull, 2, o);
    (void)m;
    return id;
}

/* produce an id of type `to` from `val` of type `from`, where from/to differ only in
 * bool<->int or fp64<->fp32 storage mapping of the leaves */
static uint32_t conv_val(SMod *m, WBuf *body, uint32_t *next_id, Inserts *ins, uint32_t val, uint32_t from, uint32_t to)
{
    if (from == to || to > m->bound || from > m->bound) return val;
    const SType *tt = &m->t[to];
    if (tt->kind == T_BOOL || (tt->kind == T_VECTOR && m->t[tt->elem].kind == T_BOOL)) {
        uint32_t r = (*next_id)++;
        uint32_t z = null_const(m, ins, next_id, from);
        wb_push(body, (5u << 16) | SpvOpINotEqual); wb_push(body, to); wb_push(body, r); wb_push(body, val); wb_push(body, z);
        return r;
    }
    if (tt->kind == T_FLOAT || tt->kind == T_VECTOR) { /* fp64 <- fp32 */
        uint32_t r = (*next_id)++;
        wb_push(body, (4u << 16) | SpvOpFConvert); wb_push(body, to); wb_push(body, r); wb_push(body, val);
        return r;
    }
    if (tt->kind == T_MATRIX) { /* dmat <- fmat: convert column by column */
        uint32_t fc = m->t[from].elem, tc2 = m->t[to].elem;
        uint32_t *cols = alloca(m->t[to].count * sizeof(uint32_t));
        for (uint32_t c = 0; c < m->t[to].count; c++) {
            uint32_t e = (*next_id)++;
            wb_push(body, (5u << 16) | SpvOpCompositeExtract); wb_push(body, fc); wb_push(body, e); wb_push(body, val); wb_push(body, c);
            cols[c] = conv_val(m, body, next_id, ins, e, fc, tc2);
        }
        uint32_t r = (*next_id)++;
        wb_push(body, ((2u + m->t[to].count) << 16) | SpvOpCompositeConstruct); wb_push(body, to); wb_push(body, r);
        for (uint32_t c = 0; c < m->t[to].count; c++) wb_push(body, cols[c]);
        return r;
    }
    if (tt->kind == T_STRUCT && tt->nmem) {
        uint32_t *mems = alloca(tt->nmem * sizeof(uint32_t));
        for (uint32_t i = 0; i < tt->nmem; i++) {
            uint32_t fm = m->t[from].mem[i];
            uint32_t e = (*next_id)++;
            wb_push(body, (5u << 16) | SpvOpCompositeExtract); wb_push(body, fm); wb_push(body, e); wb_push(body, val); wb_push(body, i);
            mems[i] = conv_val(m, body, next_id, ins, e, fm, tt->mem[i]);
        }
        uint32_t r = (*next_id)++;
        wb_push(body, ((2u + tt->nmem) << 16) | SpvOpCompositeConstruct); wb_push(body, to); wb_push(body, r);
        for (uint32_t i = 0; i < tt->nmem; i++) wb_push(body, mems[i]);
        return r;
    }
    { /* defensive: same-shape copy (arrays cannot be loaded from uniforms in GLSL) */
        uint32_t r = (*next_id)++;
        wb_push(body, (4u << 16) | SpvOpCopyObject); wb_push(body, to); wb_push(body, r); wb_push(body, val);
        return r;
    }
}

/* like conv_val but writes into a chosen result id (the id the original instruction used) */
static void emit_conv(SMod *m, WBuf *body, uint32_t *next_id, Inserts *ins, uint32_t val, uint32_t from, uint32_t to, uint32_t out)
{
    uint32_t r = conv_val(m, body, next_id, ins, val, from, to);
    if (r != out) {
        wb_push(body, (4u << 16) | SpvOpCopyObject); wb_push(body, to); wb_push(body, out); wb_push(body, r);
    }
}

/* Metal has no fp64 hardware: doubles now survive into the module that is reflected
 * (the GL view) and are lowered to software fp64 by fp64.c on the MSL side, so the
 * default-block layout below can use real 8-byte doubles and the CPU shadow stores
 * them without conversion. */

#pragma mark - stage-in/out arrays of matrices (SPIRV-Cross cannot emit them)

/* SPIRV-Cross's MSL backend flattens every stage-in/out variable and block into
 * an io struct with one member per LOCATION, and throws
 *     "MSL cannot emit arrays-of-matrices in input and output variables"
 * when a member/variable IS an array of matrices (dEQP-GL45-ES31
 * program_interface_query.program_input.type.interface_blocks.*.array.mat3x2
 * and separable_tess_eval.patch_var_array.mat3x2 link failures).  Metal has no
 * such restriction, so flatten the arrays at the SPIR-V level before the
 * translation: block member `T name[N]` (T a matrix) becomes N matrix members
 * name_0..name_{N-1} with consecutive locations, and every access [name][k]
 * (k constant) becomes the member access [name_k]; a plain `in T name[N]`
 * variable is re-typed to a one-member-per-element block.  Dynamic indices,
 * whole-array uses and value-level composites keep the original module (the
 * SPIRV-Cross error, i.e. the link fails) — the CTS interface shapes only use
 * constant indices.  The GL view reassembles the original single array
 * resource in reflect_io (ip->flat). */

typedef struct { uint32_t st, mi, mtype; } FlatCand;
typedef struct { uint32_t v, mtype; } FlatVar;

/* integer constant value of an access-chain index (int/uint), -1 if not one */
static int64_t flat_idx_const(const SMod *m, uint32_t id)
{
    if (!id || id > m->bound || !m->isconst[id] || m->isconst[id] == 2) return -1;
    if (m->t[m->ctype[id]].kind != T_INT) return -1;   /* sign in t.sign */
    return (int32_t)m->cval[id];
}

/* id of an existing int (or uint) constant with value `val`, else create one
 * (geom r88: member-index remaps for chains through block-array roots) */
static uint32_t flat_find_int_const(SMod *m, Inserts *ins, uint32_t *next_id, uint32_t val)
{
    for (uint32_t c = 1; c <= m->bound; c++)
        if (m->isconst[c] == 1 && m->t[m->ctype[c]].kind == T_INT && (uint32_t)(int32_t)m->cval[c] == val &&
            m->in[m->cinst[c]].w && (m->in[m->cinst[c]].w[0] & 0xffff) == SpvOpConstant) return c;
    /* need an int type; reuse the first one in the module */
    uint32_t it = 0;
    for (uint32_t c = 1; c <= m->bound && !it; c++)
        if (m->t[c].kind == T_INT && m->t[c].width == 32) it = c;
    if (!it) return 0;
    uint32_t id = (*next_id)++; mgrow(m, id);
    uint32_t o[3] = { it, id, val };
    ins_op(&ins->globals, SpvOpConstant, 3, o);
    m->isconst[id] = 1; m->ctype[id] = it; m->cval[id] = val; m->cinst[id] = 0;   /* 0 = header, never an OpConstant */
    return id;
}

/* replace instruction i with an owned copy of nw[0..len): works for both the
 * pre-function emission loop (which reads in[i].w/len) and the function-body
 * emission (which copies in[i].w/len verbatim) */
static void flat_replace_inst(SMod *m, size_t i, uint32_t *nw, uint32_t len)
{
    if (m->in[i].owned) free(m->in[i].w);
    m->in[i].w = nw;
    m->in[i].len = len;
    m->in[i].owned = 1;
}

static void flat_emit_member_name(SMod *m, Inserts *ins, uint32_t st, uint32_t mem, const char *name)
{
    uint32_t hdr[2] = { st, mem };
    ins_string_op(&ins->names, SpvOpMemberName, 2, hdr, name);
    /* model entry (strdup: read during this patch only) */
    if (m->nmn == m->cmn) { m->cmn = m->cmn ? m->cmn * 2 : 64; m->mn = realloc(m->mn, m->cmn * sizeof(SMName)); }
    m->mn[m->nmn].id = st; m->mn[m->nmn].m = mem; m->mn[m->nmn].s = strdup(name);
    m->nmn++;
}

static void flatten_io_matrix_arrays_dir(SMod *m, MGLIProgram *ip, int stage, Inserts *ins,
                                         uint32_t *next_id, const char *src, uint32_t want);

static void flatten_io_matrix_arrays(SMod *m, MGLIProgram *ip, int stage, Inserts *ins,
                                     uint32_t *next_id, const char *src)
{
    return flatten_io_matrix_arrays_dir(m, ip, stage, ins, next_id, src,
                                        (stage != _VERTEX_SHADER) ? SpvStorageClassInput : SpvStorageClassOutput);
}

/* geom r88: one call per storage direction.  r87 ran Input for every stage but
 * vertex (which ran Output): TES/GS/TCS OUTPUTS with arrays of matrices
 * (dEQP …program_output.type.separable_tess_eval.array.mat3x2, block arrays in
 * VS outputs) stayed unflattened and failed to translate. */
static void flatten_io_matrix_arrays_dir(SMod *m, MGLIProgram *ip, int stage, Inserts *ins,
                                         uint32_t *next_id, const char *src, uint32_t want)
{
    if (stage == _COMPUTE_SHADER || stage >= MGLI_STAGES) return;

    /* the pass runs again for the no_pack retry and once per direction: drop
     * this stage's records made for THIS direction only */
    int w = 0;
    for (int i = 0; i < ip->nflat; i++)
        if (ip->flat[i].stage != stage || ip->flat[i].sc != want) ip->flat[w++] = ip->flat[i];
    ip->nflat = w;

    FlatCand cand[16]; int ncand = 0;
    FlatVar pcand[8];  int npc = 0;
    /* geom r88: candidate structs reached through an ARRAY (interface block
     * arrays `out Block {..} inst[3];`, TCS/TE/GS per-vertex runtime arrays).
     * Chains into them are [root][elem][member][...]; pf==1 pointers below can
     * be either a block variable itself or the [root][elem] sub-chain. */
    uint32_t arr_root[8]; int narr_root = 0;

    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != want || is_builtin_var(m, v)) continue;
        bool patch = get_dec(m, v, -1, SpvDecorationPatch, NULL) != 0;
        uint32_t pt = pointee(m, m->vtype[v]);
        uint32_t st = pt;
        bool is_arr_root = false;
        if (m->t[pt].kind == T_ARRAY || m->t[pt].kind == T_RARRAY) {
            uint32_t et = m->t[pt].elem;
            if (m->t[et].kind == T_STRUCT) {
                st = et;   /* arrayed interface block instance */
                is_arr_root = true;
            } else {
                /* plain variable that IS an array of matrices.  Never a
                 * per-vertex interface: those are runtime arrays at the
                 * arrayed stages, and VS outputs / FS inputs are not arrayed.
                 * (geom r88: TES and GS outputs are not arrayed either —
                 * `layout(location=0) out mat3x2 x[2];` in a TES is a plain
                 * array over 2 and must flatten like a VS output.) */
                if (m->t[pt].kind == T_ARRAY && m->t[et].kind == T_MATRIX && npc < 8 &&
                    (stage == _FRAGMENT_SHADER || stage == _VERTEX_SHADER || patch ||
                     (want == SpvStorageClassOutput &&
                      (stage == _TESS_EVALUATION_SHADER || stage == _GEOMETRY_SHADER))))
                    pcand[npc].v = v, pcand[npc++].mtype = pt;
                continue;
            }
        }
        if (m->t[st].kind != T_STRUCT) continue;
        bool has_cand = false;
        for (uint32_t mi = 0; mi < m->t[st].nmem && ncand < 16; mi++) {
            uint32_t mt = m->t[st].mem[mi];
            if (m->t[mt].kind == T_ARRAY && m->t[m->t[mt].elem].kind == T_MATRIX && m->t[mt].count > 1) {
                cand[ncand].st = st, cand[ncand].mi = mi, cand[ncand++].mtype = mt;
                has_cand = true;
            }
        }
        if (is_arr_root && has_cand && narr_root < 8) arr_root[narr_root++] = v;
    }
    if (!ncand && !npc) {
        if (getenv("MGL_FLAT_DEBUG")) fprintf(stderr, "MGL flat st=%d sc=%u: no candidates\n", stage, want);
        return;
    }

    /* value type map: declared result type of every SSA value (composites) */
    uint32_t bound1 = m->bound + 1;
    uint32_t *vty = zalloc(bound1 * sizeof(uint32_t));
    for (size_t i = 1; i < m->n; i++) {
        uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
        if (wc < 3) continue;
        uint32_t res = 0;
        switch (op) {
            case SpvOpLoad: case SpvOpCompositeExtract: case SpvOpCompositeConstruct:
            case SpvOpCopyObject: case SpvOpFunctionCall: case SpvOpPhi: case SpvOpSelect:
            case SpvOpConstantNull: case SpvOpUndef: case SpvOpCompositeInsert:
            case SpvOpFunctionParameter:
                res = m->in[i].w[2]; break;
            default: break;
        }
        if (res && res < bound1) vty[res] = m->in[i].w[1];
    }

    /* pointer flags: 1 = pointer to a candidate struct, 2 = plain-var root,
     * 3 = arrayed root (block array / per-vertex runtime array of a candidate
     * struct; chains into it start with the element index) */
    uint8_t *pf = zalloc(bound1);
    bool arr_woelem[8] = {0};   /* whole-element pointer taken on an arrayed root */
    uint32_t arr_st[8];         /* the candidate struct of each arrayed root */
    for (int r = 0; r < narr_root; r++) {
        uint32_t at = pointee(m, m->vtype[arr_root[r]]);
        arr_st[r] = m->t[at].elem;
        pf[arr_root[r]] = 3;
    }
    for (int c = 0; c < ncand; c++) pf[cand[c].st] = 1;
    for (size_t i = 1; i < m->n; i++) {
        uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
        if (wc < 4) continue;
        if ((op == SpvOpVariable || op == SpvOpFunctionParameter) &&
            m->t[m->in[i].w[1]].kind == T_POINTER && pf[m->t[m->in[i].w[1]].elem])
            pf[m->in[i].w[2]] = 1;
    }
    /* chains whose result still points at a candidate struct (instance select) */
    for (size_t i = 1; i < m->n; i++) {
        uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5 &&
            pf[m->in[i].w[3]] == 1 &&
            m->t[m->in[i].w[1]].kind == T_POINTER && pf[m->t[m->in[i].w[1]].elem])
            pf[m->in[i].w[2]] = 1;
        else if (op == SpvOpCopyObject && wc >= 4 && pf[m->in[i].w[3]])
            pf[m->in[i].w[2]] = pf[m->in[i].w[3]];
        else if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5 &&
                 pf[m->in[i].w[3]] == 3) {
            /* [root][elem] → pointer to the block element; a chain stopping
             * there takes a whole-element pointer (infeasible for the group) */
            if (wc == 4 + 1) pf[m->in[i].w[2]] = 1;
            for (int r = 0; r < narr_root; r++)
                if (arr_root[r] == m->in[i].w[3] && wc == 5) arr_woelem[r] = true;
        }
    }
    for (int c = 0; c < npc; c++) pf[pcand[c].v] = 2;

    /* ---------- feasibility ---------- */
    int ok[16];
    for (int c = 0; c < ncand; c++) {
        uint32_t st = cand[c].st, mtype = cand[c].mtype;
        uint32_t mi = cand[c].mi;
        ok[c] = 1;
        /* 0. arrayed roots: no whole-element / whole-array uses of the block */
        for (int r = 0; r < narr_root && ok[c]; r++) {
            if (arr_st[r] != st) continue;
            if (arr_woelem[r]) ok[c] = 0;
            for (size_t i = 1; i < m->n && ok[c]; i++) {
                uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
                if (op == SpvOpLoad && wc >= 4 && m->in[i].w[3] == arr_root[r]) ok[c] = 0;
                else if (op == SpvOpStore && wc >= 3 && m->in[i].w[1] == arr_root[r]) ok[c] = 0;
                else if (op == SpvOpCopyMemory && wc >= 4 &&
                         (m->in[i].w[1] == arr_root[r] || m->in[i].w[2] == arr_root[r])) ok[c] = 0;
            }
        }
        if (!ok[c]) continue;
        /* 1. the array type must not be used anywhere else (composites, nulls,
         *    whole-array values, other pointers) */
        for (size_t i = 1; i < m->n && ok[c]; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if (op == SpvOpName || op == SpvOpMemberName || op == SpvOpDecorate ||
                op == SpvOpMemberDecorate || (op == SpvOpTypeArray && m->in[i].w[1] == mtype) ||
                (op == SpvOpTypeStruct && m->in[i].w[1] == st))
                continue;
            if (op == SpvOpCompositeConstruct && wc >= 4 && m->in[i].w[1] == st) {
                for (uint32_t k = 3; k < wc && ok[c]; k++)
                    if (vty[m->in[i].w[k]] == mtype) ok[c] = 0;
                continue;
            }
            for (uint32_t k = 1; k < wc && ok[c]; k++)
                if (m->in[i].w[k] == mtype) ok[c] = 0;
        }
        if (!ok[c]) continue;
        /* 2. member accesses: constant element index, never the whole array */
        for (size_t i = 1; i < m->n && ok[c]; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5 &&
                pf[m->in[i].w[3]] == 1) {
                int64_t mi_c = flat_idx_const(m, m->in[i].w[4]);
                if (mi_c < 0 || (uint32_t)mi_c != mi) continue;
                if (wc == 5) { ok[c] = 0; break; }             /* whole array ptr */
                if (flat_idx_const(m, m->in[i].w[5]) < 0) { ok[c] = 0; break; }
            } else if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 6 &&
                       pf[m->in[i].w[3]] == 3) {
                /* block array / per-vertex instance: [root][elem][member][...] */
                int64_t mi_c = flat_idx_const(m, m->in[i].w[5]);
                if (mi_c < 0) { ok[c] = 0; break; }            /* dynamic member */
                if ((uint32_t)mi_c != mi) continue;
                if (wc == 6) { ok[c] = 0; break; }             /* whole member ptr */
                if (flat_idx_const(m, m->in[i].w[6]) < 0) { ok[c] = 0; break; }
            } else if (op == SpvOpCompositeExtract && wc >= 5 && vty[m->in[i].w[3]] == st) {
                if (m->in[i].w[4] == mi && wc == 5) { ok[c] = 0; break; }
            } else if (op == SpvOpCompositeInsert && wc >= 6 && vty[m->in[i].w[4]] == st) {
                if (m->in[i].w[5] == mi && (wc == 6 || vty[m->in[i].w[3]] == mtype)) { ok[c] = 0; break; }
            }
        }
    }
    int pok[8];
    for (int c = 0; c < npc; c++) {
        uint32_t mtype = pcand[c].mtype, v = pcand[c].v;
        pok[c] = 1;
        for (size_t i = 1; i < m->n && pok[c]; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            /* the variable's own OpTypePointer(Input, array[]) mentions the
             * array type by construction — it is replaced.  A whole-array use
             * shows up as a chain [v] alone below, and a whole-array VALUE as
             * a Load/Store whose type or operand IS the array type here. */
            if (op == SpvOpName || op == SpvOpMemberName || op == SpvOpDecorate ||
                op == SpvOpMemberDecorate || op == SpvOpTypePointer ||
                (op == SpvOpTypeArray && m->in[i].w[1] == mtype))
                continue;
            for (uint32_t k = 1; k < wc && pok[c]; k++)
                if (m->in[i].w[k] == mtype) pok[c] = 0;
        }
        if (!pok[c]) continue;
        /* element index must be constant, and the whole array must never be
         * taken (chain [v] alone) */
        for (size_t i = 1; i < m->n && pok[c]; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 4 &&
                pf[m->in[i].w[3]] == 2 &&
                (wc == 4 || flat_idx_const(m, m->in[i].w[4]) < 0))
                pok[c] = 0;
        }
        (void)v;
    }

    /* ---------- member flattening (per struct, all members together) ---------- */
    for (int c0 = 0; c0 < ncand; c0++) {
        if (!ok[c0] || cand[c0].st == 0) continue;
        uint32_t st = cand[c0].st;
        /* snapshot member names: the re-emit loop below runs AFTER the module's
         * OpMemberName instructions for `st` are killed and the parsed-name
         * table is stripped of `st` — reading member_name() there lost every
         * surviving (non-group) member name, which SPIRV-Cross then MSL-names
         * _m0/_m1… (geom r88) */
        char memname[64][72];
        for (uint32_t mi = 0; mi < m->t[st].nmem && mi < 64; mi++) {
            const char *nm = member_name(m, st, mi);
            snprintf(memname[mi], sizeof memname[0], "%s", nm ? nm : "");
        }
        struct { uint32_t old_mi, new_mi, mtype; } grp[8]; int ngrp = 0;
        char grpname[8][64];
        for (int c = c0; c < ncand; c++) {
            if (cand[c].st != st || !ok[c]) continue;
            cand[c].st = 0;   /* consumed by this struct's rewrite */
            snprintf(grpname[ngrp], 64, "%s", memname[cand[c].mi]);
            grp[ngrp].old_mi = cand[c].mi;
            grp[ngrp].mtype = cand[c].mtype;
            ngrp++;
        }
        /* new member layout: groups expand in place, survivors shift */
        uint32_t final_mem[64]; int nf = 0;
        int newidx_of[64];
        uint32_t gstart[8];
        for (uint32_t mi = 0; mi < m->t[st].nmem && mi < 64; mi++) {
            int g = -1;
            for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
            if (g >= 0) {
                newidx_of[mi] = -1;
                gstart[g] = (uint32_t)nf;
                grp[g].new_mi = (uint32_t)nf;
                for (uint32_t e = 0; e < m->t[grp[g].mtype].count && nf < 64; e++)
                    final_mem[nf++] = m->t[grp[g].mtype].elem;
            } else {
                newidx_of[mi] = nf;
                final_mem[nf++] = m->t[st].mem[mi];
            }
        }
        /* records for reflect_io */
        ip->flat = realloc(ip->flat, (ip->nflat + ngrp) * sizeof *ip->flat);
        for (int g = 0; g < ngrp; g++) {
            memset(&ip->flat[ip->nflat], 0, sizeof ip->flat[0]);
            ip->flat[ip->nflat].stage = stage; ip->flat[ip->nflat].sc = want; ip->flat[ip->nflat].st = st;
            ip->flat[ip->nflat].start = grp[g].new_mi;
            ip->flat[ip->nflat].count = m->t[grp[g].mtype].count;
            ip->flat[ip->nflat].mtype = grp[g].mtype; ip->flat[ip->nflat].var = 0;
            ip->flat[ip->nflat].kind = 0;
            snprintf(ip->flat[ip->nflat].name, sizeof ip->flat[0].name, "%s", grpname[g]);
            ip->nflat++;
        }
        /* member decorations: kill + re-add with remapped indices; a flattened
         * group's Location becomes base + element * columns */
        struct { int32_t mem; uint32_t d; uint32_t lit; int has_lit; } dcs[128]; int ndcs = 0;
        for (size_t i = 0; i < m->ndec; i++) {
            const SDec *x = &m->dec[i];
            if (x->id == st && x->m >= 0 && !m->in[x->inst].dead && ndcs < 128)
                dcs[ndcs].mem = x->m, dcs[ndcs].d = x->d, dcs[ndcs].lit = x->lit,
                dcs[ndcs].has_lit = (x->lit != 0 || x->d == SpvDecorationLocation), ndcs++;
        }
        for (int i = 0; i < ndcs; i++) kill_decs(m, st, dcs[i].mem, dcs[i].d);
        for (int i = 0; i < ndcs; i++) {
            int g = -1;
            for (int gg = 0; gg < ngrp; gg++)
                if ((uint32_t)dcs[i].mem == grp[gg].old_mi) g = gg;
            if (g >= 0) {
                uint32_t N = m->t[grp[g].mtype].count;
                uint32_t cols = m->t[m->t[grp[g].mtype].elem].len;
                for (uint32_t e = 0; e < N; e++) {
                    uint32_t lit = dcs[i].lit;
                    if (dcs[i].d == SpvDecorationLocation) lit = dcs[i].lit + e * cols;
                    new_dec(m, ins, st, (int32_t)(grp[g].new_mi + e), dcs[i].d, dcs[i].has_lit, lit);
                }
            } else if (dcs[i].mem < 64 && newidx_of[dcs[i].mem] >= 0) {
                new_dec(m, ins, st, newidx_of[dcs[i].mem], dcs[i].d, dcs[i].has_lit, dcs[i].lit);
            }
        }
        /* member names */
        for (size_t i = 1; i < m->n; i++)
            if ((m->in[i].w[0] & 0xffff) == SpvOpMemberName && m->in[i].w[1] == st)
                m->in[i].dead = 1;
        {
            int w2 = 0;
            for (size_t i = 0; i < m->nmn; i++) if (m->mn[i].id != st) m->mn[w2++] = m->mn[i];
            m->nmn = w2;
        }
        for (uint32_t mi = 0; mi < m->t[st].nmem; mi++) {
            int g = -1;
            for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
            const char *nm = (mi < 64) ? memname[mi] : member_name(m, st, mi);
            if (!nm || !*nm) continue;
            if (g >= 0) {
                for (uint32_t e = 0; e < m->t[grp[g].mtype].count; e++) {
                    char n2[72];
                    snprintf(n2, sizeof n2, "%s_%u", nm, e);
                    flat_emit_member_name(m, ins, st, grp[g].new_mi + e, n2);
                }
            } else {
                flat_emit_member_name(m, ins, st, newidx_of[mi], nm);
            }
        }
        /* struct type members */
        free(m->t[st].mem);
        m->t[st].nmem = (uint32_t)nf;
        m->t[st].mem = malloc((nf + 1) * 4);
        memcpy(m->t[st].mem, final_mem, nf * 4);
        for (size_t i = 1; i < m->n; i++)
            if ((m->in[i].w[0] & 0xffff) == SpvOpTypeStruct && m->in[i].w[1] == st) {
                uint32_t *nw = malloc(4 * (nf + 2));
                nw[0] = ((uint32_t)(nf + 2) << 16) | SpvOpTypeStruct;
                nw[1] = st;
                memcpy(&nw[2], final_mem, nf * 4);
                flat_replace_inst(m, i, nw, (uint32_t)(nf + 2));
                break;
            }
        /* access chains / composites: [mi, k, rest] -> [mi_new + k, rest] */
        for (size_t i = 1; i < m->n; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5 &&
                pf[m->in[i].w[3]] == 1) {
                int64_t mi_c = flat_idx_const(m, m->in[i].w[4]);
                if (mi_c < 0) continue;
                uint32_t mi = (uint32_t)mi_c;
                int g = -1;
                for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
                if (g >= 0) {
                    if (wc < 6) continue;
                    int64_t k = flat_idx_const(m, m->in[i].w[5]);
                    if (k < 0 || (uint32_t)k >= m->t[grp[g].mtype].count) continue;
                    uint32_t *nw = malloc(4 * (wc - 1));
                    nw[0] = ((uint32_t)(wc - 1) << 16) | op;
                    memcpy(&nw[1], &m->in[i].w[1], 3 * 4);
                    nw[4] = m->in[i].w[5];   /* k IS the new member index value */
                    memcpy(&nw[5], &m->in[i].w[6], (wc - 6) * 4);
                    flat_replace_inst(m, i, nw, wc - 1);
                } else if (mi < 64 && newidx_of[mi] >= 0 && newidx_of[mi] != (int)mi) {
                    uint32_t cid = 0;
                    for (uint32_t cc = 1; cc <= m->bound; cc++)
                        if (m->isconst[cc] == 1 && (int32_t)m->cval[cc] == newidx_of[mi] &&
                            m->t[m->ctype[cc]].kind == T_INT) { cid = cc; break; }
                    if (!cid) continue;
                    uint32_t *nw = malloc(4 * wc);
                    memcpy(nw, m->in[i].w, 4 * wc);
                    nw[4] = cid;
                    flat_replace_inst(m, i, nw, wc);
                }
            } else if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 6 &&
                       pf[m->in[i].w[3]] == 3) {
                /* block-array / per-vertex instance: [root][elem][member][rest]
                 * → keep [root][elem], collapse [member][k] → [member'+k] */
                int64_t mi_c = flat_idx_const(m, m->in[i].w[5]);
                if (mi_c < 0) continue;
                uint32_t mi = (uint32_t)mi_c;
                int g = -1;
                for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
                if (g >= 0) {
                    if (wc < 7) continue;
                    int64_t k = flat_idx_const(m, m->in[i].w[6]);
                    if (k < 0 || (uint32_t)k >= m->t[grp[g].mtype].count) continue;
                    uint32_t *nw = malloc(4 * (wc - 1));
                    nw[0] = ((uint32_t)(wc - 1) << 16) | op;
                    memcpy(&nw[1], &m->in[i].w[1], 4 * 4);   /* result, ptr type, base, elem idx */
                    nw[5] = m->in[i].w[6];   /* k IS the new member index value */
                    memcpy(&nw[6], &m->in[i].w[7], (wc - 7) * 4);
                    flat_replace_inst(m, i, nw, wc - 1);
                } else if (mi < 64 && newidx_of[mi] >= 0 && newidx_of[mi] != (int)mi) {
                    uint32_t cid = flat_find_int_const(m, ins, next_id, (uint32_t)newidx_of[mi]);
                    if (!cid) continue;
                    uint32_t *nw = malloc(4 * wc);
                    memcpy(nw, m->in[i].w, 4 * wc);
                    nw[5] = cid;
                    flat_replace_inst(m, i, nw, wc);
                }
            } else if (op == SpvOpCompositeExtract && wc >= 5 && vty[m->in[i].w[3]] == st) {
                uint32_t mi = m->in[i].w[4];
                int g = -1;
                for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
                if (g < 0) {
                    if (mi < 64 && newidx_of[mi] >= 0 && newidx_of[mi] != (int)mi) {
                        uint32_t *nw = malloc(4 * wc);
                        memcpy(nw, m->in[i].w, 4 * wc);
                        nw[4] = (uint32_t)newidx_of[mi];
                        flat_replace_inst(m, i, nw, wc);
                    }
                } else if (wc >= 6) {
                    uint32_t *nw = malloc(4 * (wc - 1));
                    nw[0] = ((uint32_t)(wc - 1) << 16) | op;
                    memcpy(&nw[1], &m->in[i].w[1], 3 * 4);
                    nw[4] = grp[g].new_mi + m->in[i].w[5];
                    memcpy(&nw[5], &m->in[i].w[6], (wc - 6) * 4);
                    flat_replace_inst(m, i, nw, wc - 1);
                }
            } else if (op == SpvOpCompositeInsert && wc >= 6 && vty[m->in[i].w[4]] == st) {
                uint32_t mi = m->in[i].w[5];
                int g = -1;
                for (int gg = 0; gg < ngrp; gg++) if (grp[gg].old_mi == mi) g = gg;
                if (g < 0) {
                    if (mi < 64 && newidx_of[mi] >= 0 && newidx_of[mi] != (int)mi) {
                        uint32_t *nw = malloc(4 * wc);
                        memcpy(nw, m->in[i].w, 4 * wc);
                        nw[5] = (uint32_t)newidx_of[mi];
                        flat_replace_inst(m, i, nw, wc);
                    }
                } else if (wc >= 7 && vty[m->in[i].w[3]] == m->t[grp[g].mtype].elem) {
                    uint32_t *nw = malloc(4 * (wc - 1));
                    nw[0] = ((uint32_t)(wc - 1) << 16) | op;
                    memcpy(&nw[1], &m->in[i].w[1], 4 * 4);
                    nw[5] = grp[g].new_mi + m->in[i].w[6];
                    memcpy(&nw[6], &m->in[i].w[7], (wc - 7) * 4);
                    flat_replace_inst(m, i, nw, wc - 1);
                }
            }
        }
        (void)gstart;
    }

    /* ---------- plain-variable flattening ---------- */
    for (int c = 0; c < npc; c++) {
        if (getenv("MGL_FLAT_DEBUG"))
            fprintf(stderr, "MGL flat var cand v=%u mtype=%u pok=%d\n", pcand[c].v, pcand[c].mtype, pok[c]);
        if (!pok[c]) continue;
        uint32_t v = pcand[c].v, mtype = pcand[c].mtype;
        uint32_t N = m->t[mtype].count;
        uint32_t elem = m->t[mtype].elem;      /* the matrix type */
        const char *nm = id_name(m, v);
        if (!nm || !*nm) continue;

        uint32_t st2 = (*next_id)++; mgrow(m, st2);
        m->t[st2].kind = T_STRUCT;
        m->t[st2].nmem = N;
        m->t[st2].mem = malloc((N + 1) * 4);
        for (uint32_t e = 0; e < N; e++) m->t[st2].mem[e] = elem;
        {
            uint32_t o[70];
            o[0] = st2;
            for (uint32_t e = 0; e < N && e < 69; e++) o[1 + e] = elem;
            ins_op(&ins->globals, SpvOpTypeStruct, (int)N + 1, o);
        }
        {
            uint32_t id0 = st2;
            char tn[80];
            /* NOT the variable's own name: SPIRV-Cross uses the type name in
             * the MSL it generates and a same-named local variable in the
             * function body then shadows it ("cannot use dot operator on a
             * type") */
            snprintf(tn, sizeof tn, "mgl_%s_t", nm);
            ins_string_op(&ins->names, SpvOpName, 1, &id0, tn);
        }
        new_dec(m, ins, st2, -1, SpvDecorationBlock, 0, 0);
        for (uint32_t e = 0; e < N; e++) {
            char n2[72];
            snprintf(n2, sizeof n2, "%s_%u", nm, e);
            flat_emit_member_name(m, ins, st2, e, n2);
            /* interpolation qualifiers move from the variable to the members */
            if (get_dec(m, v, -1, SpvDecorationFlat, NULL))
                new_dec(m, ins, st2, (int32_t)e, SpvDecorationFlat, 0, 0);
            if (get_dec(m, v, -1, SpvDecorationCentroid, NULL))
                new_dec(m, ins, st2, (int32_t)e, SpvDecorationCentroid, 0, 0);
            if (get_dec(m, v, -1, SpvDecorationNoPerspective, NULL))
                new_dec(m, ins, st2, (int32_t)e, SpvDecorationNoPerspective, 0, 0);
            uint32_t base = 0;
            if (src_explicit_qual(src, nm, want == SpvStorageClassInput ? "in" : "out", "location") &&
                get_dec(m, v, -1, SpvDecorationLocation, &base)) {
                uint32_t cols = m->t[elem].len;
                new_dec(m, ins, st2, (int32_t)e, SpvDecorationLocation, 1, base + e * cols);
            }
        }
        /* pointer + retyped variable (same id: the element index k of every
         * access chain IS the new member index — no chain rewriting) */
        uint32_t ptr2 = (*next_id)++; mgrow(m, ptr2);
        m->t[ptr2].kind = T_POINTER;
        m->t[ptr2].storage = m->vsc[v];
        m->t[ptr2].elem = st2;
        {
            uint32_t o[4] = { ptr2, m->vsc[v], st2 };   /* [result, storage, pointee] */
            ins_op(&ins->globals, SpvOpTypePointer, 3, o);
            uint32_t ov[3] = { ptr2, v, m->vsc[v] };    /* [type, result, storage] */
            ins_op(&ins->globals, SpvOpVariable, 3, ov);
        }
        for (size_t i = 1; i < m->n; i++)
            if ((m->in[i].w[0] & 0xffff) == SpvOpVariable && m->in[i].w[2] == v) {
                m->in[i].dead = 1;
                break;
            }
        m->vtype[v] = ptr2;
        ip->flat = realloc(ip->flat, (ip->nflat + 1) * sizeof *ip->flat);
        memset(&ip->flat[ip->nflat], 0, sizeof ip->flat[0]);
        ip->flat[ip->nflat].stage = stage; ip->flat[ip->nflat].sc = want; ip->flat[ip->nflat].st = st2;
        ip->flat[ip->nflat].start = 0; ip->flat[ip->nflat].count = N;
        ip->flat[ip->nflat].mtype = mtype; ip->flat[ip->nflat].var = v;
        ip->flat[ip->nflat].kind = 0;
        snprintf(ip->flat[ip->nflat].name, sizeof ip->flat[0].name, "%s", nm);
        ip->nflat++;
    }
    free(vty);
    free(pf);
}

#pragma mark - arrays of patch interface blocks (SPIRV-Cross cannot emit them)

/* glslang emits `patch out Blk { M m; } inst[N];` as a Patch-decorated Output
 * variable of type array<Blk,N>.  SPIRV-Cross's MSL backend moves every patch
 * variable into the per-patch io struct and hits an unimplemented path there
 * (a Variant bad-cast that surfaces as the error string "nullptr"), so the
 * whole stage fails to translate — dEQP-GL45-ES31
 * program_interface_query.program_output.*.interface_blocks.patch_out.block_array.*.
 *
 * Metal has no such restriction once the array is gone: flatten inst[N] into
 * ONE patch block whose member k*nmem+mi is member mi of element k
 * (element-major — the order both GL and SPIRV-Cross assign locations),
 * rewrite every access [inst][k][mi][rest…] to [inst][k*nmem+mi][rest…], and
 * record the original shape for reflection (ip->flat kind 1: the GL view of an
 * interface block array is one resource per member, named <BlockName>.<member>
 * — the CTS's own reference notes "there is no array index specifier even if
 * the interface is declared as an array of instances").  TCS patch outputs and
 * TES patch inputs both apply.  Location decorations stay on the variable:
 * SPIRV-Cross assigns the members sequential locations from it, which for the
 * element-major member order equals GL's block-array packing (§9.2.5 /
 * ARB_enhanced_layouts), and reflection keeps reporting -1 / the explicit
 * source location only (§11.1.1). */
static void flatten_patch_block_arrays(SMod *m, MGLIProgram *ip, int stage, Inserts *ins,
                                       uint32_t *next_id, const char *src)
{
    (void)src;
    if (stage != _TESS_CONTROL_SHADER && stage != _TESS_EVALUATION_SHADER) return;
    uint32_t want = (stage == _TESS_CONTROL_SHADER) ? SpvStorageClassOutput : SpvStorageClassInput;

    /* the pass re-runs on the no_pack retry: drop this stage's kind-1 records */
    int w = 0;
    for (int i = 0; i < ip->nflat; i++)
        if (ip->flat[i].stage != stage || ip->flat[i].kind != 1) ip->flat[w++] = ip->flat[i];
    ip->nflat = w;

    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != want) continue;
        uint32_t pt = pointee(m, m->vtype[v]);
        if (m->t[pt].kind != T_ARRAY) continue;                        /* block arrays only */
        uint32_t st = m->t[pt].elem;
        if (m->t[st].kind != T_STRUCT || !get_dec(m, st, -1, SpvDecorationBlock, NULL)) continue;
        if (!get_dec(m, v, -1, SpvDecorationPatch, NULL)) {
            /* members must ALL be patch (SPIRV-Cross's is_patch_block rule) */
            if (!m->t[st].nmem) continue;
            bool all_patch = true;
            for (uint32_t mi = 0; mi < m->t[st].nmem && all_patch; mi++)
                if (!get_dec(m, st, (int32_t)mi, SpvDecorationPatch, NULL)) all_patch = false;
            if (!all_patch) continue;
        }
        if (is_builtin_var(m, v)) continue;
        uint32_t N = m->t[pt].count, nmem = m->t[st].nmem;
        if (!N || !nmem || N * nmem > 64) continue;
        /* arrays of matrices inside the members: left to the r87 pass, whose
         * records key on the struct id — a combined shape has no coverage */
        bool skip = false;
        for (uint32_t mi = 0; mi < nmem && !skip; mi++) {
            uint32_t mt = m->t[st].mem[mi];
            if (m->t[mt].kind == T_ARRAY && m->t[m->t[mt].elem].kind == T_MATRIX) skip = true;
        }
        for (int f = 0; f < ip->nflat && !skip; f++)
            if (ip->flat[f].stage == stage && ip->flat[f].st == st) skip = true;
        if (skip) continue;

        /* feasibility: the array type is only the variable's own type, and
         * every use of the variable is an access chain [v][k][mi][rest…] with
         * constant k and mi (no whole array / element / member pointers) */
        bool ok = true;
        for (size_t i = 1; i < m->n && ok; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if (op == SpvOpName || op == SpvOpMemberName || op == SpvOpDecorate ||
                op == SpvOpMemberDecorate)
                continue;
            if (op == SpvOpTypeArray && m->in[i].w[1] == pt) continue;
            if (op == SpvOpTypeStruct && m->in[i].w[1] == st) continue;
            if (op == SpvOpTypePointer && wc >= 4 && m->in[i].w[3] == pt) continue;   /* replaced */
            for (uint32_t k = 1; k < wc && ok; k++)
                if (m->in[i].w[k] == pt) ok = false;
        }
        for (size_t i = 1; i < m->n && ok; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 4 &&
                m->in[i].w[3] == v) {
                int64_t k = (wc >= 5) ? flat_idx_const(m, m->in[i].w[4]) : -1;
                int64_t mi = (wc >= 6) ? flat_idx_const(m, m->in[i].w[5]) : -1;
                if (wc < 6 || k < 0 || (uint64_t)k >= N || mi < 0) ok = false;
            } else if (op == SpvOpLoad && wc >= 4 && m->in[i].w[3] == v) ok = false;
            else if (op == SpvOpStore && wc >= 3 && m->in[i].w[1] == v) ok = false;
            else if (op == SpvOpCopyMemory && wc >= 4 && (m->in[i].w[1] == v || m->in[i].w[2] == v)) ok = false;
            else if (op == SpvOpCopyObject && wc >= 4 && m->in[i].w[3] == v) ok = false;
            else if (op == SpvOpPtrAccessChain && wc >= 4 && m->in[i].w[3] == v) ok = false;
        }
        if (!ok) {
            if (getenv("MGL_FLAT_DEBUG")) fprintf(stderr, "MGL pblock st=%d var %u '%s': shape not flattenable\n",
                                                  stage, v, id_name(m, v) ? id_name(m, v) : "?");
            continue;
        }

        /* member names / decorations snapshot (the originals are killed below) */
        char memname[64][72];
        struct { int32_t mem; uint32_t d, lit; int has_lit; } dcs[128]; int ndcs = 0;
        for (uint32_t mi = 0; mi < nmem; mi++) {
            const char *nm = member_name(m, st, mi);
            snprintf(memname[mi], sizeof memname[0], "%s", nm ? nm : "");
        }
        for (size_t i = 0; i < m->ndec; i++) {
            const SDec *x = &m->dec[i];
            if (x->id == st && x->m >= 0 && !m->in[x->inst].dead && ndcs < 128) {
                uint32_t d = x->d;
                int has_lit = (d == SpvDecorationLocation || d == SpvDecorationComponent ||
                               d == SpvDecorationOffset || d == SpvDecorationIndex ||
                               d == SpvDecorationBuiltIn);
                dcs[ndcs].mem = x->m, dcs[ndcs].d = d, dcs[ndcs].lit = x->lit,
                dcs[ndcs].has_lit = has_lit, ndcs++;
            }
        }

        /* the flattened struct: member k*nmem+mi = member mi of element k */
        uint32_t st2 = (*next_id)++; mgrow(m, st2);
        m->t[st2].kind = T_STRUCT;
        m->t[st2].nmem = N * nmem;
        m->t[st2].mem = malloc((N * nmem + 1) * 4);
        {
            uint32_t o[70];
            o[0] = st2;
            for (uint32_t e = 0; e < N * nmem && e < 69; e++)
                o[1 + e] = m->t[st].mem[e % nmem], m->t[st2].mem[e] = m->t[st].mem[e % nmem];
            ins_op(&ins->globals, SpvOpTypeStruct, (int)(N * nmem + 1), o);
        }
        {   /* keep the BLOCK name: reflect_io uses it as the resource prefix */
            uint32_t id0 = st2;
            ins_string_op(&ins->names, SpvOpName, 1, &id0, id_name(m, st) ? id_name(m, st) : "MGLPatchBlock");
        }
        new_dec(m, ins, st2, -1, SpvDecorationBlock, 0, 0);
        for (uint32_t k = 0; k < N; k++)
            for (uint32_t mi = 0; mi < nmem; mi++) {
                uint32_t ni = k * nmem + mi;
                char n2[80];
                snprintf(n2, sizeof n2, "%s_%u", memname[mi], k);
                flat_emit_member_name(m, ins, st2, ni, n2);
                for (int d = 0; d < ndcs; d++)
                    if ((uint32_t)dcs[d].mem == mi)
                        new_dec(m, ins, st2, (int32_t)ni, dcs[d].d, dcs[d].has_lit, dcs[d].lit);
            }

        /* pointer + retyped variable (same id) */
        uint32_t ptr2 = (*next_id)++; mgrow(m, ptr2);
        m->t[ptr2].kind = T_POINTER;
        m->t[ptr2].storage = m->vsc[v];
        m->t[ptr2].elem = st2;
        {
            uint32_t o[4] = { ptr2, m->vsc[v], st2 };
            ins_op(&ins->globals, SpvOpTypePointer, 3, o);
            uint32_t ov[3] = { ptr2, v, m->vsc[v] };
            ins_op(&ins->globals, SpvOpVariable, 3, ov);
        }
        for (size_t i = 1; i < m->n; i++)
            if ((m->in[i].w[0] & 0xffff) == SpvOpVariable && m->in[i].w[2] == v) {
                m->in[i].dead = 1;
                break;
            }
        m->vtype[v] = ptr2;

        /* access chains: [v][k][mi][rest] → [v][k*nmem+mi][rest] */
        for (size_t i = 1; i < m->n; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff, wc = m->in[i].w[0] >> 16;
            if ((op != SpvOpAccessChain && op != SpvOpInBoundsAccessChain) || wc < 6) continue;
            if (m->in[i].w[3] != v) continue;
            int64_t k = flat_idx_const(m, m->in[i].w[4]);
            int64_t mi = flat_idx_const(m, m->in[i].w[5]);
            if (k < 0 || (uint64_t)k >= N || mi < 0) continue;
            uint32_t cid = flat_find_int_const(m, ins, next_id, (uint32_t)(k * nmem + mi));
            if (!cid) continue;
            uint32_t *nw = malloc(4 * (wc - 1));
            nw[0] = ((uint32_t)(wc - 1) << 16) | op;
            memcpy(&nw[1], &m->in[i].w[1], 4 * 4);   /* result, ptr type, base, — */
            nw[4] = cid;
            memcpy(&nw[5], &m->in[i].w[6], (wc - 6) * 4);
            flat_replace_inst(m, i, nw, wc - 1);
        }

        /* records: one per original member; members of element k>0 of the same
         * original member sit at start + k*nmem */
        ip->flat = realloc(ip->flat, (ip->nflat + (int)nmem) * sizeof *ip->flat);
        for (uint32_t mi = 0; mi < nmem; mi++) {
            memset(&ip->flat[ip->nflat], 0, sizeof ip->flat[0]);
            ip->flat[ip->nflat].stage = stage; ip->flat[ip->nflat].sc = want;
            ip->flat[ip->nflat].st = st2; ip->flat[ip->nflat].start = mi;
            ip->flat[ip->nflat].count = N; ip->flat[ip->nflat].stride = nmem;
            ip->flat[ip->nflat].mtype = m->t[st].mem[mi]; ip->flat[ip->nflat].var = v;
            ip->flat[ip->nflat].kind = 1;
            snprintf(ip->flat[ip->nflat].name, sizeof ip->flat[0].name, "%s", memname[mi]);
            ip->nflat++;
        }
        if (getenv("MGL_FLAT_DEBUG"))
            fprintf(stderr, "MGL pblock st=%d var %u '%s': flattened %u x %u members -> struct %u\n",
                    stage, v, id_name(m, v) ? id_name(m, v) : "?", N, nmem, st2);
    }
}

static uint32_t *transform_stage(GLMContext ctx, StageCtx *sc, int stage, size_t *out_n, MGLIProgram *ip, const char *src)
{
    SMod *m = sc->m;
    if (getenv("MGL_FLAT_DEBUG")) fprintf(stderr, "MGL transform st=%d n=%zu\n", stage, m->n);
    Inserts ins; memset(&ins, 0, sizeof ins);
    uint32_t next_id = m->bound;
    uint32_t laid_n = m->bound + 1;
    uint8_t *laid = zalloc(laid_n);
    uint32_t **repls = zalloc(m->n * sizeof(uint32_t *)); /* fp64->fp32 constant replacements */
    uint32_t int_t = 0, float_t = 0;

    /* 0. flatten stage-in/out arrays of matrices (SPIRV-Cross cannot emit
     * them); everything below — locations, MSL, reflection — works on the
     * flattened module.  geom r88: both directions per stage (r87 only did
     * Input for non-vertex stages, so TES/GS OUTPUT arrays of matrices still
     * failed to translate).  Vertex INPUTS stay out: they are vertex
     * attributes, bound by location, and never arrays of matrices in GL. */
    flatten_io_matrix_arrays_dir(m, ip, stage, &ins, &next_id, src, SpvStorageClassOutput);
    if (stage != _VERTEX_SHADER)
        flatten_io_matrix_arrays_dir(m, ip, stage, &ins, &next_id, src, SpvStorageClassInput);
    /* 0b. flatten arrays of patch interface blocks (SPIRV-Cross cannot emit
     * those at all: "nullptr" bad-cast in its patch io-struct code) */
    flatten_patch_block_arrays(m, ip, stage, &ins, &next_id, src);
    KV *tc = NULL; int ntc = 0; /* mapped type cache */
    KV *mpt = NULL; int nmpt = 0; /* rewritten access-chain result id -> mapped pointee type */

    /* 1. interface locations */
    for (int i = 0; i < sc->nlocs; i++) {
        kill_decs(m, sc->locs[i].var, -1, SpvDecorationLocation);
        new_dec(m, &ins, sc->locs[i].var, -1, SpvDecorationLocation, 1, sc->locs[i].loc);
        if (sc->locs[i].index >= 0) {
            kill_decs(m, sc->locs[i].var, -1, SpvDecorationIndex);
            if (sc->locs[i].index > 0) new_dec(m, &ins, sc->locs[i].var, -1, SpvDecorationIndex, 1, (uint32_t)sc->locs[i].index);
        }
    }

    /* 2. default uniform block */
    uint32_t *old_of = zalloc((m->bound + 1) * 4);   /* var id -> member index + 1 */
    sc->ndub = 0;
    for (uint32_t v = 1; v <= m->bound; v++)
        if (m->isvar[v] && m->vglobal[v] && m->vsc[v] == SpvStorageClassUniformConstant && !is_opaque(m, pointee(m, m->vtype[v])))
            sc->ndub++;
    bool pack = sc->ndub > 0 && !sc->no_pack;
    sc->dub = zalloc(sc->ndub * sizeof(DubMember));
    {
        uint32_t k = 0;
        for (uint32_t v = 1; v <= m->bound; v++)
            if (m->isvar[v] && m->vglobal[v] && m->vsc[v] == SpvStorageClassUniformConstant && !is_opaque(m, pointee(m, m->vtype[v]))) {
                DubMember *d = &sc->dub[k];
                d->var = v; d->type = pointee(m, m->vtype[v]); d->member = k;
                d->gtype = d->type;
                uint32_t l; d->explicit_loc = get_dec(m, v, -1, SpvDecorationLocation, &l) ? (int)l : -1;
                snprintf(d->name, sizeof d->name, "%s", id_name(m, v));
                old_of[v] = k + 1;
                k++;
            }
        /* GL-level leaf types (bool/double) recorded before any GPU-side rewriting */
        for (uint32_t k = 0; k < sc->ndub; k++)
            record_leaf_kinds(m, sc->dub[k].type, &sc->dub[k].leaf, &sc->dub[k].nleaf);
    }
    /* Metal has no fp64: rewrite every double in the module to float (values stay doubles in
     * the CPU-side shadow and are converted on upload; GL introspection still reports the
     * original types via the recorded leaf kinds) */
    if (pack) {
        /* verify every use is a load or an access chain (otherwise keep the original module) */
        uint8_t *ptrs = zalloc(next_id + 1);
        for (uint32_t i = 0; i < sc->ndub; i++) ptrs[sc->dub[i].var] = 1;
        for (size_t i = m->first_func; i < m->n && pack; i++) {
            uint32_t *w = m->in[i].w, op = w[0] & 0xffff, wc = m->in[i].len;
#define P(x) ((x) <= m->bound && ptrs[(x)])
            switch (op) {
                case SpvOpLoad: break;
                case SpvOpAccessChain: case SpvOpInBoundsAccessChain: if (P(w[3])) ptrs[w[2]] = 1; break;
                case SpvOpPtrAccessChain: case SpvOpCopyObject: case SpvOpImageTexelPointer: if (P(w[3])) pack = false; break;
                case SpvOpStore: case SpvOpCopyMemory: case SpvOpCopyMemorySized: if (P(w[1]) || P(w[2])) pack = false; break;
                case SpvOpFunctionCall: for (uint32_t a = 4; a < wc; a++) if (P(w[a])) pack = false; break;
                case SpvOpSelect: if (P(w[4]) || P(w[5])) pack = false; break;
                case SpvOpPhi: for (uint32_t a = 3; a < wc; a += 2) if (P(w[a])) pack = false; break;
                case SpvOpPtrEqual: case SpvOpPtrNotEqual: case SpvOpPtrDiff: if (P(w[3]) || P(w[4])) pack = false; break;
                case SpvOpExtInst: for (uint32_t a = 5; a < wc; a++) if (P(w[a])) pack = false; break;
                default: break;
            }
#undef P
        }
        free(ptrs);
        if (!pack) {
            fprintf(stderr, "MGL introspect: stage %d: default uniforms used through pointers, keeping per-uniform buffers\n", stage);
            memset(old_of, 0, (m->bound + 1) * 4);
        }
    }

    uint32_t dub_struct = 0, dub_ptr = 0, dub_var = 0;
    KV *consts = NULL; int nconsts = 0;
    KV *uptr = NULL; int nuptr = 0;
    if (pack) {
        int_t = find_or_add_type_int(m, &ins, &next_id);
        /* MSL cannot lay out bools in buffers: storage type of bool/bvec members becomes int */
        float_t = find_or_add_type_float(m, &ins, &next_id);
        for (uint32_t k = 0; k < sc->ndub; k++)
            sc->dub[k].gtype = map_gpu_type(m, &ins, &next_id, sc->dub[k].type, int_t, float_t, &tc, &ntc);
        free(tc); tc = NULL; ntc = 0;
        if (m->bound + 1 > laid_n) {
            laid = realloc(laid, m->bound + 1);
            memset(laid + laid_n, 0, m->bound + 1 - laid_n);
            old_of = realloc(old_of, (m->bound + 1) * 4);
            memset(old_of + laid_n, 0, (m->bound + 1 - laid_n) * 4);
            laid_n = m->bound + 1;
        }
        /* member index constants */
        consts = zalloc(sc->ndub * sizeof(KV));
        for (uint32_t k = 0; k < sc->ndub; k++) {
            uint32_t id = 0;
            for (uint32_t c = 1; c <= m->bound && !id; c++)
                if (m->isconst[c] == 1 && m->ctype[c] == int_t && m->cval[c] == k && m->in[m->cinst[c]].w && (m->in[m->cinst[c]].w[0] & 0xffff) == SpvOpConstant) id = c;
            if (!id) { id = next_id++; uint32_t o[3] = { int_t, id, k }; ins_op(&ins.globals, SpvOpConstant, 3, o); }
            consts[nconsts++] = (KV){ k, id };
        }
        /* layout */
        uint32_t off = 0;
        for (uint32_t k = 0; k < sc->ndub; k++) {
            ensure_layout(m, &ins, sc->dub[k].gtype, laid);
            uint32_t s, a; msl_layout(m, sc->dub[k].gtype, &s, &a);
            if (a < 4) a = 4;
            off = align_up(off, a);
            sc->dub[k].offset = off;
            off += s;
        }
        sc->dub_size = align_up(off ? off : 16, 16);
        dub_struct = next_id++;
        {
            uint32_t *o = malloc((sc->ndub + 1) * 4);
            o[0] = dub_struct;
            for (uint32_t k = 0; k < sc->ndub; k++) o[k + 1] = sc->dub[k].gtype;
            ins_op(&ins.globals, SpvOpTypeStruct, (int)sc->ndub + 1, o);
            free(o);
        }
        uint32_t ds[1] = { dub_struct };
        ins_string_op(&ins.names, SpvOpName, 1, ds, "mgl_DefaultUniforms");
        new_dec(m, &ins, dub_struct, -1, SpvDecorationBlock, 0, 0);
        for (uint32_t k = 0; k < sc->ndub; k++) {
            uint32_t o[2] = { dub_struct, k };
            ins_string_op(&ins.names, SpvOpMemberName, 2, o, sc->dub[k].name);
            new_dec(m, &ins, dub_struct, (int32_t)k, SpvDecorationOffset, 1, sc->dub[k].offset);
            if (sc->dub[k].explicit_loc >= 0) new_dec(m, &ins, dub_struct, (int32_t)k, SpvDecorationLocation, 1, (uint32_t)sc->dub[k].explicit_loc);
            if (sc->ip && m->vinit[sc->dub[k].var]) {
                uint32_t *vals = NULL; int nv = 0, wide = 0;
                flatten_const(m, m->vinit[sc->dub[k].var], &vals, &nv, &wide, 0);
                MGLIProgram *ip = sc->ip;
                ip->inits[stage] = realloc(ip->inits[stage], (size_t)(ip->ninits[stage] + 1) * sizeof(*ip->inits[stage]));
                ip->inits[stage][ip->ninits[stage]].name = strdup(sc->dub[k].name);
                ip->inits[stage][ip->ninits[stage]].vals = vals;
                ip->inits[stage][ip->ninits[stage]].nvals = nv;
                ip->inits[stage][ip->ninits[stage]].wide = wide;
                ip->ninits[stage]++;
            }
            uint32_t base = strip_arrays(m, sc->dub[k].gtype);
            if (m->t[base].kind == T_MATRIX) {
                uint32_t cs, ca; msl_layout(m, m->t[base].elem, &cs, &ca);
                new_dec(m, &ins, dub_struct, (int32_t)k, SpvDecorationColMajor, 0, 0);
                new_dec(m, &ins, dub_struct, (int32_t)k, SpvDecorationMatrixStride, 1, align_up(cs, ca));
            }
        }
        dub_ptr = next_id++;
        { uint32_t o[3] = { dub_ptr, SpvStorageClassUniform, dub_struct }; ins_op(&ins.globals, SpvOpTypePointer, 3, o); }
        dub_var = next_id++;
        ins_string_op(&ins.names, SpvOpName, 1, &dub_var, "mgl_DefaultUniforms");
        new_dec(m, &ins, dub_var, -1, SpvDecorationDescriptorSet, 1, 0);
        new_dec(m, &ins, dub_var, -1, SpvDecorationBinding, 1, 0);
        /* drop the old variables and their annotations */
        for (uint32_t k = 0; k < sc->ndub; k++) {
            uint32_t v = sc->dub[k].var;
            m->in[m->vinst[v]].dead = 1;
            for (size_t d = 0; d < m->ndec; d++) if (m->dec[d].id == v && m->dec[d].inst) m->in[m->dec[d].inst].dead = 1;
            for (size_t i = 1; i < m->first_func; i++) {
                uint32_t op = m->in[i].w[0] & 0xffff;
                if (op == SpvOpName && m->in[i].w[1] == v) m->in[i].dead = 1;
            }
        }
    }

    /* uniform pointer type for pointee X (created on demand) */
#define UPTR(X) ({ uint32_t _x = (X), _r = 0; \
        for (uint32_t _i = 1; _i <= m->bound && !_r; _i++) \
            if (m->t[_i].kind == T_POINTER && m->t[_i].storage == SpvStorageClassUniform && m->t[_i].elem == _x) _r = _i; \
        if (!_r) _r = kv_get(uptr, nuptr, _x); \
        if (!_r) { _r = next_id++; uint32_t _o[3] = { _r, SpvStorageClassUniform, _x }; ins_op(&ins.globals, SpvOpTypePointer, 3, _o); \
                   uptr = realloc(uptr, (nuptr + 1) * sizeof(KV)); uptr[nuptr++] = (KV){ _x, _r }; } \
        _r; })

    /* pre-create all pointer types needed by the function rewrite (they must precede functions) */
    WBuf body; memset(&body, 0, sizeof body);
    uint8_t *retyped = zalloc(m->bound + 1);
    if (pack) {
        for (size_t i = m->first_func; i < m->n; i++) {
            uint32_t *w = m->in[i].w, op = w[0] & 0xffff, wc = m->in[i].len;
            if (op == SpvOpLoad && w[3] <= m->bound && old_of[w[3]]) {
                uint32_t mem = old_of[w[3]] - 1;
                uint32_t g = sc->dub[mem].gtype;
                uint32_t pt = UPTR(g);
                uint32_t nid = next_id++;
                uint32_t ac[5] = { (5u << 16) | SpvOpAccessChain, pt, nid, dub_var, kv_get(consts, nconsts, mem) };
                wb_inst(&body, ac, 5);
                if (g != w[1]) {
                    /* member storage type was rewritten: load the GPU type, convert back */
                    uint32_t t = next_id++;
                    wb_push(&body, (4u << 16) | SpvOpLoad); wb_push(&body, g); wb_push(&body, t); wb_push(&body, nid);
                    for (uint32_t a = 4; a < wc; a++) wb_push(&body, w[a]);
                    emit_conv(m, &body, &next_id, &ins, t, g, w[1], w[2]);
                } else {
                    wb_push(&body, w[0]); wb_push(&body, w[1]); wb_push(&body, w[2]); wb_push(&body, nid);
                    for (uint32_t a = 4; a < wc; a++) wb_push(&body, w[a]);
                }
                continue;
            }
            if (op == SpvOpLoad && w[3] <= m->bound && kv_get(mpt, nmpt, w[3])) {
                /* load through a rewritten access chain whose pointee was mapped */
                uint32_t g = kv_get(mpt, nmpt, w[3]);
                if (g != w[1]) {
                    uint32_t t = next_id++;
                    wb_push(&body, (4u << 16) | SpvOpLoad); wb_push(&body, g); wb_push(&body, t); wb_push(&body, w[3]);
                    for (uint32_t a = 4; a < wc; a++) wb_push(&body, w[a]);
                    emit_conv(m, &body, &next_id, &ins, t, g, w[1], w[2]);
                    continue;
                }
            }
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && w[3] <= m->bound && (old_of[w[3]] || retyped[w[3]])) {
                uint32_t p = pointee(m, w[1]);
                uint32_t mp = map_gpu_type(m, &ins, &next_id, p, int_t, float_t, &tc, &ntc);
                uint32_t pt = UPTR(mp);
                if (mp != p) { mpt = realloc(mpt, (size_t)(nmpt + 1) * sizeof(KV)); mpt[nmpt++] = (KV){ w[2], mp }; }
                if (w[2] <= m->bound) retyped[w[2]] = 1;
                if (old_of[w[3]]) {
                    uint32_t mem = old_of[w[3]] - 1;
                    wb_push(&body, ((wc + 1) << 16) | op); wb_push(&body, pt); wb_push(&body, w[2]); wb_push(&body, dub_var);
                    wb_push(&body, kv_get(consts, nconsts, mem));
                    for (uint32_t a = 4; a < wc; a++) wb_push(&body, w[a]);
                } else {
                    wb_push(&body, w[0]); wb_push(&body, pt);
                    for (uint32_t a = 2; a < wc; a++) wb_push(&body, w[a]);
                }
                continue;
            }
            wb_inst(&body, w, wc);
        }
    } else {
        for (size_t i = m->first_func; i < m->n; i++) wb_inst(&body, m->in[i].w, m->in[i].len);
    }
#undef UPTR

    /* SPIR-V >= 1.4: global interface lists contain every global; replace removed ids */
    if (pack && m->version >= 0x10400 && m->ep_inst != (size_t)-1) {
        /* handled at emission below */
    }

    /* 3. buffer-class global variable ordering */
    size_t *order = zalloc((m->n + 4) * sizeof(size_t)); size_t norder = 0;
    for (int cls = 0; cls < 3; cls++) {
        for (uint32_t v = 1; v <= m->bound; v++) {
            if (!m->isvar[v] || !m->vglobal[v] || m->in[m->vinst[v]].dead) continue;
            uint32_t pt = strip_arrays(m, pointee(m, m->vtype[v]));
            int c = -1;
            if (m->vsc[v] == SpvStorageClassUniform && get_dec(m, pt, -1, SpvDecorationBufferBlock, NULL)) c = 1;
            else if (m->vsc[v] == SpvStorageClassUniform) c = 0;
            else if (m->vsc[v] == SpvStorageClassStorageBuffer) c = 1;
            else if (m->vsc[v] == SpvStorageClassAtomicCounter) c = 2;
            if (c == cls) order[norder++] = m->vinst[v];
        }
        if (cls == 0 && pack) order[norder++] = (size_t)-2; /* the default block goes after the UBOs */
    }
    uint8_t *moved = zalloc(m->n + 1);
    for (size_t k = 0; k < norder; k++) if (order[k] != (size_t)-2) moved[order[k]] = 1;

    /* emission */
    size_t last_debug = 0, last_annot = 0, first_type = m->first_func;
    for (size_t i = 1; i < m->first_func; i++) {
        uint32_t op = m->in[i].w[0] & 0xffff;
        if (op == SpvOpSource || op == SpvOpSourceContinued || op == SpvOpSourceExtension || op == SpvOpName ||
            op == SpvOpMemberName || op == SpvOpString || op == SpvOpModuleProcessed) last_debug = i;
        if (op == SpvOpDecorate || op == SpvOpMemberDecorate || op == SpvOpDecorationGroup || op == SpvOpGroupDecorate ||
            op == SpvOpGroupMemberDecorate || op == SpvOpDecorateId || op == SpvOpDecorateString || op == SpvOpMemberDecorateString) last_annot = i;
    }
    for (size_t i = 1; i < m->first_func; i++) {
        uint32_t op = m->in[i].w[0] & 0xffff;
        if ((op >= SpvOpTypeVoid && op <= SpvOpTypeForwardPointer) || (op >= SpvOpConstantTrue && op <= SpvOpSpecConstantOp) ||
            op == SpvOpVariable || op == SpvOpUndef || op == SpvOpLine) { first_type = i; break; }
    }
    size_t names_at = last_debug ? last_debug + 1 : (last_annot ? 0 : first_type);
    size_t decs_at = last_annot ? last_annot + 1 : first_type;
    if (!names_at) {
        /* no debug section but annotations exist: names go before the first annotation */
        for (size_t i = 1; i < m->first_func; i++) {
            uint32_t op = m->in[i].w[0] & 0xffff;
            if (op == SpvOpDecorate || op == SpvOpMemberDecorate || op == SpvOpDecorationGroup) { names_at = i; break; }
        }
        if (!names_at) names_at = first_type;
    }

    WBuf out; memset(&out, 0, sizeof out);
    for (int i = 0; i < 5; i++) wb_push(&out, m->hdr[i]);
    for (size_t i = 1; i <= m->first_func; i++) {
        if (i == names_at) wb_inst(&out, ins.names.w, (uint32_t)ins.names.n);
        if (i == decs_at) wb_inst(&out, ins.decs.w, (uint32_t)ins.decs.n);
        if (i == m->first_func) {
            wb_inst(&out, ins.globals.w, (uint32_t)ins.globals.n);
            for (size_t k = 0; k < norder; k++) {
                if (order[k] == (size_t)-2) {
                    uint32_t o[4] = { (4u << 16) | SpvOpVariable, dub_ptr, dub_var, SpvStorageClassUniform };
                    wb_inst(&out, o, 4);
                } else {
                    wb_inst(&out, m->in[order[k]].w, m->in[order[k]].len);
                }
            }
            break;
        }
        if (m->in[i].dead || moved[i]) {
            if (repls[i]) wb_inst(&out, repls[i], repls[i][0] >> 16);
            continue;
        }
        uint32_t *w = m->in[i].w, op = w[0] & 0xffff;
        if (op == SpvOpEntryPoint && pack && m->version >= 0x10400) {
            /* drop removed interface ids, add the default block */
            size_t start = out.n;
            wb_push(&out, 0);
            uint32_t wc = m->in[i].len, a = 3;
            wb_push(&out, w[1]); wb_push(&out, w[2]);
            /* copy name string */
            while (a < wc) { uint32_t v = w[a]; wb_push(&out, v); a++; if (((v >> 24) & 0xff) == 0) break; }
            for (; a < wc; a++) if (!(w[a] <= m->bound && old_of[w[a]])) wb_push(&out, w[a]);
            wb_push(&out, dub_var);
            out.w[start] = ((uint32_t)(out.n - start) << 16) | SpvOpEntryPoint;
            continue;
        }
        wb_inst(&out, w, m->in[i].len);
    }
    wb_inst(&out, body.w, (uint32_t)body.n);
    out.w[3] = next_id;

    sc->dub_var = dub_var;
    free(order); free(moved); free(retyped); free(body.w); free(consts); free(uptr); free(old_of); free(laid);
    for (size_t i = 0; i < m->n; i++) free(repls[i]);
    free(repls); free(mpt); free(tc);
    free(ins.names.w); free(ins.decs.w); free(ins.globals.w);
    (void)ctx;
    *out_n = out.n;
    return out.w;
}

#pragma mark - reflection

typedef struct Refl {
    GLMContext ctx;
    MGLIProgram *ip;
    StageCtx *sc;
} Refl;

static char *xstrdup(const char *s) { return strdup(s ? s : ""); }

static int add_uniform(MGLIProgram *ip, const char *name, int *created)
{
    for (int i = 0; i < ip->nuniforms; i++) if (!strcmp(ip->uniforms[i].name, name)) { *created = 0; return i; }
    ip->uniforms = realloc(ip->uniforms, (ip->nuniforms + 1) * sizeof(MGLIUniform));
    MGLIUniform *u = &ip->uniforms[ip->nuniforms];
    memset(u, 0, sizeof *u);
    u->name = xstrdup(name);
    u->block_index = -1; u->offset = -1; u->array_stride = -1; u->matrix_stride = -1;
    u->atomic_buffer_index = -1; u->location = -1;
    *created = 1;
    return ip->nuniforms++;
}

/* GL 4.6 11.1.1: a uniform declared with an explicit location in one stage uses that
 * location for the merged program uniform; a same-named declaration without one (in
 * another stage) must not override it, and two different explicit locations are a
 * link error.  The location was left as the implicit -2-cursor sentinel when the
 * uniform was first created by a stage without an explicit location. */
static void merge_explicit_loc(MGLIProgram *ip, int ui, int explicit_loc, int cursor)
{
    if (explicit_loc < 0) return;
    MGLIUniform *u = &ip->uniforms[ui];
    int el = explicit_loc + cursor;
    if (el + u->size > ip->uniform_loc_max) ip->uniform_loc_overflow = 1;
    if (u->location >= 0)
    {
        if (u->location != el)
        {
            if (getenv("MGL_LOC_TRACE")) fprintf(stderr, "MGL loc: conflict %s: stored %d vs stage el=%d (explicit %d cursor %d)\n", u->name, u->location, el, explicit_loc, cursor);
            ip->uniform_loc_conflict = 1;
        }
    }
    else if (u->location < -1)
        u->location = el;            /* implicit so far: adopt the explicit location */
}

/* walk a default-block / opaque uniform variable and create GL uniforms for its leaves */
/* the GL type a rewritten leaf had originally (bool/bvecN, double/dvecN/dmatNxM) */
static GLenum leaf_gl_type(uint8_t kind, uint32_t cols, uint32_t rows)
{
    static const GLenum bv[4] = { GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
    static const GLenum dv[4] = { GL_DOUBLE, GL_DOUBLE_VEC2, GL_DOUBLE_VEC3, GL_DOUBLE_VEC4 };
    static const GLenum dm[3][3] = { { GL_DOUBLE_MAT2, GL_DOUBLE_MAT2x3, GL_DOUBLE_MAT2x4 },
                                     { GL_DOUBLE_MAT3x2, GL_DOUBLE_MAT3, GL_DOUBLE_MAT3x4 },
                                     { GL_DOUBLE_MAT4x2, GL_DOUBLE_MAT4x3, GL_DOUBLE_MAT4 } };
    if (kind == LEAF_WAS_BOOL) return (rows >= 1 && rows <= 4) ? bv[rows - 1] : GL_BOOL;
    if (kind == LEAF_WAS_DOUBLE) {
        if (cols > 1) return (cols >= 2 && cols <= 4 && rows >= 2 && rows <= 4) ? dm[cols - 2][rows - 2] : GL_DOUBLE;
        return (rows >= 1 && rows <= 4) ? dv[rows - 1] : GL_DOUBLE;
    }
    return GL_NONE;
}

static void walk_default(MGLIProgram *ip, const SMod *m, int stage, const char *name, uint32_t t, uint32_t offset,                         int explicit_loc, int *loc_cursor, uint32_t spv_id, int res_type, int *elem_cursor, int hidden_prefix,
                         const uint8_t *lk, int *lki)
{
    /* lk[] remembers, per leaf uniform in walk order, whether its GL type was rewritten for
     * the GPU (bool->int, double->float): introspection still reports GL_BOOL/GL_DOUBLE */
    const SType *ty = &m->t[t];
    /* a sampler/image member hoisted out of a default-block struct (see
     * mgl_hoist_struct_samplers: MSL has no textures in buffer-backed structs) is one
     * plain uniform here, named "u_mglhoist_b_mglhoist_s".  GL requires the dotted
     * spelling ("u.b.s") in every program-interface query, so swap in the recorded
     * GL path before anything derives a resource name from it.  An array of structs
     * (the uniform declaration itself arrayed) enumerates one resource per element
     * (GL 4.6 §7.3.1) — n_outer leading dimensions belong to the declaration, the
     * rest are the member's own bottom-level array. */
    char gl_name[512];
    const char *base_name = name;
    int n_outer = -1;
    if (strstr(name, "_mglhoist_") && !strchr(name, '.')) {
        /* the array's shape as SPIR-V sees it ("3", "2,3"): it selects the right
         * recorded spelling when the same uniform name was hoisted with more than
         * one shape in this process (glcts runs many cases in one process) */
        char shape[64];
        size_t so = 0;
        for (uint32_t at = t; so + 1 < sizeof shape; ) {
            if (m->t[at].kind != T_ARRAY && m->t[at].kind != T_RARRAY) break;
            if (so) shape[so++] = ',';
            so += (size_t)snprintf(shape + so, sizeof shape - so, "%u", m->t[at].count);
            at = m->t[at].elem;
        }
        shape[so] = 0;
        if (mgl_hoist_gl_path2(name, so ? shape : NULL, gl_name, sizeof gl_name, &n_outer))
            base_name = gl_name;
        else
            n_outer = -1;
    }
    name = base_name;
    if ((ty->kind == T_ARRAY || ty->kind == T_RARRAY) && n_outer > 0) {
        /* "S u[N];" hoisted to "u_mglhoist_m[N]": GL enumerates N resources "u[i].m" —
         * the element index belongs to the VARIABLE, so it is inserted before the
         * member path ("u[0].b.s"), and every dimension of the declaration expands
         * ("u[0][1].b.s").  Their locations are consecutive, exactly the layout one
         * flattened entry of size N*M would have taken, so glUniform1i on "u[i].m[j]"
         * and the sampler-unit shadow both keep working. */
        uint32_t dims[4], strides[4], total = 1;
        int nd = 0;
        uint32_t at = t;
        while (nd < 4 && (m->t[at].kind == T_ARRAY || m->t[at].kind == T_RARRAY) && nd < n_outer) {
            uint32_t st = 0;
            get_dec(m, at, -1, SpvDecorationArrayStride, &st);
            dims[nd] = m->t[at].count ? m->t[at].count : 1;
            strides[nd] = st;
            total *= dims[nd];
            at = m->t[at].elem;
            nd++;
        }
        const char *dot = strchr(name, '.');
        int hlen = dot ? (int)(dot - name) : (int)strlen(name);
        for (uint32_t k = 0; k < total; k++) {
            uint32_t rem = k, off = offset;
            char nm[512];
            snprintf(nm, sizeof nm, "%.*s", hlen, name);
            for (int i = 0; i < nd; i++) {
                uint32_t div = 1;
                for (int j = i + 1; j < nd; j++) div *= dims[j];
                uint32_t idx = (rem / div) % dims[i];
                rem -= idx * div;
                off += idx * strides[i];
                snprintf(nm + strlen(nm), sizeof nm - strlen(nm), "[%u]", idx);
            }
            snprintf(nm + strlen(nm), sizeof nm - strlen(nm), "%s", dot ? dot : "");
            walk_default(ip, m, stage, nm, at, off, explicit_loc, loc_cursor,
                         spv_id, res_type, elem_cursor, hidden_prefix, lk, lki);
        }
        return;
    }
    if ((ty->kind == T_ARRAY || ty->kind == T_RARRAY) ) {
        uint32_t base = m->t[ty->elem].kind == T_ARRAY ? 0 : 1;
        uint32_t es, ea, stride;
        msl_layout(m, ty->elem, &es, &ea);
        if (!get_dec(m, t, -1, SpvDecorationArrayStride, &stride)) stride = align_up(es, ea);
        uint32_t et = ty->elem;
        if (base && m->t[et].kind != T_STRUCT) {
            /* array of basic type: one uniform "name[0]" */
            char nm[512]; snprintf(nm, sizeof nm, "%s[0]", name);
            int created, ui = add_uniform(ip, nm, &created);
            MGLIUniform *u = &ip->uniforms[ui];
            if (created) {
                u->type = gl_type_of(m, et); u->size = (GLint)(ty->count ? ty->count : 1);
                scalar_info(m, et, &u->kind, &u->cols, &u->rows);
                if (lk && *lki < 8192) { uint8_t k2 = lk[(*lki)++]; if (k2) { u->type = leaf_gl_type(k2, u->cols, u->rows); u->kind = k2 == LEAF_WAS_BOOL ? MGLI_K_BOOL : MGLI_K_DOUBLE; } }
                u->opaque_class = is_opaque(m, et) ? (m->t[strip_arrays(m, et)].kind == T_IMAGE && m->t[strip_arrays(m, et)].sampled == 2 ? 2 : 1) : 0;
                if (u->opaque_class) u->kind = MGLI_K_INT;
                u->hidden = hidden_prefix;
                u->location = explicit_loc >= 0 ? explicit_loc + *loc_cursor : -2 - *loc_cursor; /* resolved later */
                if (explicit_loc >= 0 && explicit_loc + *loc_cursor + u->size > ip->uniform_loc_max)
                    ip->uniform_loc_overflow = 1;   /* GL 4.6 11.1.1: beyond GL_MAX_UNIFORM_LOCATIONS */
                *loc_cursor += u->size;
            }
            else
            {
                merge_explicit_loc(ip, ui, explicit_loc, *loc_cursor);
                *loc_cursor += u->size;   /* the merge path must advance too: a
                                           * second stage's leaves would otherwise
                                           * all see cursor 0 (false conflict) */
            }
            u->stages |= 1u << stage;
            u->st[stage].present = 1;
            u->st[stage].offset = offset; u->st[stage].array_stride = stride;
            uint32_t b = strip_arrays(m, et);
            if (m->t[b].kind == T_MATRIX) { uint32_t cs, ca; msl_layout(m, m->t[b].elem, &cs, &ca); u->st[stage].matrix_stride = align_up(cs, ca); }
            /* GPU storage: 4 bytes per scalar (int for bool), 8 for software-double leaves */
            u->st[stage].scalar_size = (u->kind == MGLI_K_DOUBLE) ? 8 : 4;
            u->st[stage].spv_id = spv_id; u->st[stage].res_type = res_type; u->st[stage].elem_base = *elem_cursor;
            *elem_cursor += (int)(ty->count ? ty->count : 1);
            return;
        }
        for (uint32_t i = 0; i < (ty->count ? ty->count : 1); i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[%u]", name, i);
            walk_default(ip, m, stage, nm, et, offset + i * stride, explicit_loc, loc_cursor, spv_id, res_type, elem_cursor, hidden_prefix, lk, lki);
        }
        return;
    }
    if (ty->kind == T_STRUCT) {
        for (uint32_t i = 0; i < ty->nmem; i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s.%s", name, member_name(m, t, i));
            uint32_t mo = 0; get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo);
            walk_default(ip, m, stage, nm, ty->mem[i], offset + mo, explicit_loc, loc_cursor, spv_id, res_type, elem_cursor, hidden_prefix, lk, lki);
        }
        return;
    }
    int created, ui = add_uniform(ip, name, &created);
    MGLIUniform *u = &ip->uniforms[ui];
    if (created) {
        u->type = gl_type_of(m, t); u->size = 1;
        scalar_info(m, t, &u->kind, &u->cols, &u->rows);
        if (lk && *lki < 8192) { uint8_t k2 = lk[(*lki)++]; if (k2) { u->type = leaf_gl_type(k2, u->cols, u->rows); u->kind = k2 == LEAF_WAS_BOOL ? MGLI_K_BOOL : MGLI_K_DOUBLE; } }
        u->opaque_class = is_opaque(m, t) ? (m->t[t].kind == T_IMAGE && m->t[t].sampled == 2 ? 2 : 1) : 0;
        if (u->opaque_class) u->kind = MGLI_K_INT;
        u->hidden = hidden_prefix;
        u->location = explicit_loc >= 0 ? explicit_loc + *loc_cursor : -2 - *loc_cursor;
        if (explicit_loc >= 0 && explicit_loc + *loc_cursor + 1 > ip->uniform_loc_max)
            ip->uniform_loc_overflow = 1;           /* GL 4.6 11.1.1: beyond GL_MAX_UNIFORM_LOCATIONS */
        *loc_cursor += 1;
    }
    else
    {
        merge_explicit_loc(ip, ui, explicit_loc, *loc_cursor);
        *loc_cursor += 1;                /* keep the leaf cursor in step across stages */
    }
    u->stages |= 1u << stage;
    u->st[stage].present = 1;
    if (getenv("MGL_FP64_DUMP")) { FILE*d_ = fopen("/tmp/fp64wd.txt","a"); if (d_) { fprintf(d_, "walk %s stage=%d off=%u\n", name, stage, offset); fclose(d_); } }
    u->st[stage].offset = offset; u->st[stage].array_stride = 0;
    if (ty->kind == T_MATRIX) { uint32_t cs, ca; msl_layout(m, ty->elem, &cs, &ca); u->st[stage].matrix_stride = align_up(cs, ca); }
    /* GPU storage: 4 bytes per scalar (int for bool), 8 for software-double leaves */
    u->st[stage].scalar_size = (u->kind == MGLI_K_DOUBLE) ? 8 : 4;
    u->st[stage].spv_id = spv_id; u->st[stage].res_type = res_type; u->st[stage].elem_base = *elem_cursor;
    *elem_cursor += 1;
}

/* size of a type under its explicit (block) layout */
static uint32_t block_type_size(const SMod *m, uint32_t t, uint32_t mstride, int row_major)
{
    const SType *ty = &m->t[t];
    switch (ty->kind) {
        case T_BOOL: return 4;
        case T_INT: case T_FLOAT: return ty->width / 8;
        case T_VECTOR: return ty->count * (m->t[ty->elem].kind == T_BOOL ? 4 : m->t[ty->elem].width / 8);
        case T_MATRIX: {
            uint32_t rows = m->t[ty->elem].count, cols = ty->count;
            uint32_t es = m->t[m->t[ty->elem].elem].width / 8;
            if (!mstride) mstride = rows * es;
            return row_major ? (rows - 1) * mstride + cols * es : (cols - 1) * mstride + rows * es;
        }
        case T_ARRAY: { uint32_t st = 0; get_dec(m, t, -1, SpvDecorationArrayStride, &st);
            return ty->count ? (ty->count - 1) * st + block_type_size(m, ty->elem, mstride, row_major) : 0; }
        case T_RARRAY: return 0;
        case T_STRUCT: {
            uint32_t end = 0;
            for (uint32_t i = 0; i < ty->nmem; i++) {
                uint32_t mo = 0, ms = 0; get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo);
                get_dec(m, t, (int32_t)i, SpvDecorationMatrixStride, &ms);
                uint32_t sz = block_type_size(m, ty->mem[i], ms, get_dec(m, t, (int32_t)i, SpvDecorationRowMajor, NULL));
                if (mo + sz > end) end = mo + sz;
            }
            return end;
        }
        default: return 0;
    }
}

typedef struct {
    MGLIProgram *ip; const SMod *m; int stage; int block; bool ssbo;
    int top_level; GLint top_size, top_stride;
    const MGLBoolPaths *bools;   /* GLSL-declared bool block members of this stage */
    /* REFERENCED_BY_*_SHADER is usage-based (GL 4.6 §9.2: a uniform block / SSBO counts
     * only when it "contains at least one variable referenced by the corresponding
     * shader").  used_mask bit i = top-level member i of the block is accessed in this
     * stage's module (SPIRV-Cross active buffer ranges); used_all = usage unknown
     * (reflection fallback / >64 members) → keep the declaration-based answer;
     * cur_used = the current recursion is under a used top-level member. */
    uint64_t used_mask; bool used_all, cur_used;
    uint32_t block_type;        /* id of the block's own struct type */
} BlockWalk;

/*
 * glslang stores bool/bvecN uniform- and buffer-block members as uint/uvecN (GLSL-to-SPIR-V
 * convention: "a 32-bit int where non-0 means true").  GL introspection must report the GLSL
 * type, so walk_block() checks the members declared bool in the stage source (cts domain:
 * cts_glsl_compat.c) and overrides the storage type of exactly those leaves.
 */
static void src_bool_leaf_type(const BlockWalk *bw, const char *name, uint32_t t,
                               GLenum *type, int *kind, int *cols, int *rows)
{
    if (!bw->bools || !mgl_cts_bool_paths_has(bw->bools, name))
        return;
    const SType *ty = &bw->m->t[t];
    int n = 1;
    uint32_t base = t;
    if (ty->kind == T_VECTOR) { n = (int)ty->count; base = ty->elem; }
    if (n < 1 || n > 4) return;
    /* only integer storage can represent a bool member (glslang's uint mapping) */
    if (bw->m->t[base].kind != T_INT) return;
    static const GLenum bt[5] = { GL_NONE, GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3, GL_BOOL_VEC4 };
    *type = bt[n];
    if (kind) *kind = MGLI_K_BOOL;
    if (cols) *cols = n;
    if (rows) *rows = 1;
}

static void add_member(MGLIBlock *blk, int ui)
{
    for (int i = 0; i < blk->member_count; i++) if (blk->members[i] == ui) return;
    blk->members = realloc(blk->members, (blk->member_count + 1) * sizeof(int));
    blk->members[blk->member_count++] = ui;
}

static void walk_block(BlockWalk *bw, const char *name, uint32_t t, uint32_t offset, uint32_t mstride, int row_major, int depth)
{
    const SMod *m = bw->m;
    const SType *ty = &m->t[t];
    if (ty->kind == T_ARRAY || ty->kind == T_RARRAY) {
        uint32_t stride = 0; get_dec(m, t, -1, SpvDecorationArrayStride, &stride);
        uint32_t et = ty->elem;
        bool basic = m->t[et].kind != T_STRUCT && m->t[et].kind != T_ARRAY && m->t[et].kind != T_RARRAY;
        if (depth == 0 && bw->ssbo && !basic) { bw->top_size = (GLint)ty->count; bw->top_stride = (GLint)stride; }
        if (basic) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[0]", name);
            if (bw->ssbo) {
                MGLIProgram *ip = bw->ip;
                int found = -1;
                for (int i = 0; i < ip->nbvars; i++) if (!strcmp(ip->bvars[i].name, nm) && ip->bvars[i].block_index == bw->block) found = i;
                if (found < 0) {
                    ip->bvars = realloc(ip->bvars, (ip->nbvars + 1) * sizeof(MGLIBufferVar));
                    MGLIBufferVar *b = &ip->bvars[ip->nbvars]; memset(b, 0, sizeof *b);
                    b->name = xstrdup(nm); b->type = gl_type_of(m, et); b->size = (GLint)ty->count; b->block_index = bw->block;
                    src_bool_leaf_type(bw, nm, et, &b->type, NULL, NULL, NULL);
                    b->offset = (GLint)offset; b->array_stride = (GLint)stride;
                    b->matrix_stride = m->t[et].kind == T_MATRIX ? (GLint)mstride : 0; b->is_row_major = m->t[et].kind == T_MATRIX ? row_major : 0;
                    b->top_size = depth == 0 ? 1 : bw->top_size; b->top_stride = depth == 0 ? 0 : bw->top_stride;
                    found = ip->nbvars++;
                    MGLIBlock *blk = &ip->ssbos[bw->block];
                    blk->members = realloc(blk->members, (blk->member_count + 1) * sizeof(int));
                    blk->members[blk->member_count++] = found;
                }
                if (bw->cur_used) ip->bvars[found].stages |= 1u << bw->stage;   /* usage-based (see BlockWalk) */
            } else {
                int created, ui = add_uniform(bw->ip, nm, &created);
                MGLIUniform *u = &bw->ip->uniforms[ui];
                if (created) {
                    u->type = gl_type_of(m, et); u->size = (GLint)(ty->count ? ty->count : 1); u->block_index = bw->block;
                    u->offset = (GLint)offset; u->array_stride = (GLint)stride;
                    u->matrix_stride = m->t[et].kind == T_MATRIX ? (GLint)mstride : 0; u->is_row_major = m->t[et].kind == T_MATRIX ? row_major : 0;
                    scalar_info(m, et, &u->kind, &u->cols, &u->rows);
                    src_bool_leaf_type(bw, nm, et, &u->type, &u->kind, &u->cols, &u->rows);
                }
                add_member(&bw->ip->ubos[bw->block], ui);
                if (bw->cur_used) u->stages |= 1u << bw->stage;   /* usage-based (see BlockWalk) */
            }
            return;
        }
        uint32_t n = ty->count ? ty->count : 1;
        if (bw->ssbo && depth == 0) n = 1; /* only the first element of a top-level array of aggregates */
        for (uint32_t i = 0; i < n; i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[%u]", name, i);
            walk_block(bw, nm, et, offset + i * stride, mstride, row_major, depth + 1);
        }
        return;
    }
    if (ty->kind == T_STRUCT) {
        bool is_block = (t == bw->block_type);   /* the block's own struct: its members are the top-level ones */
        for (uint32_t i = 0; i < ty->nmem; i++) {
            char nm[512];
            if (name[0]) snprintf(nm, sizeof nm, "%s.%s", name, member_name(m, t, i));
            else snprintf(nm, sizeof nm, "%s", member_name(m, t, i));
            uint32_t mo = 0, ms = 0; get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo);
            get_dec(m, t, (int32_t)i, SpvDecorationMatrixStride, &ms);
            int rm = get_dec(m, t, (int32_t)i, SpvDecorationRowMajor, NULL);
            if (depth == 0) { bw->top_size = 1; bw->top_stride = 0; }
            /* entering a top-level member of the block: usage is its mask bit
             * (SPIRV-Cross reports the top-level member containing any access);
             * nested members inherit their parent's answer */
            bool saved = bw->cur_used;
            if (is_block) bw->cur_used = bw->used_all || ((bw->used_mask >> i) & 1);
            /* depth 0 means "directly under the block's own struct": a member of a
             * NESTED struct is not a top-level member any more, so its arrays are
             * enumerated per element (GL 4.6 §7.3.1 / the CTS reference: only the
             * block member's own array collapses to name[0] for a buffer block).
             * The old "depth == 0 ? 0 : depth + 1" kept depth 0 for a member struct
             * as well, so an array inside it was collapsed too
             * (buffer_variable.resource_list.var_struct_array_*) */
            walk_block(bw, nm, ty->mem[i], offset + mo, ms, rm, is_block ? 0 : depth + 1);
            bw->cur_used = saved;
        }
        return;
    }
    /* leaf */
    if (bw->ssbo) {
        MGLIProgram *ip = bw->ip;
        int found = -1;
        for (int i = 0; i < ip->nbvars; i++) if (!strcmp(ip->bvars[i].name, name) && ip->bvars[i].block_index == bw->block) found = i;
        if (found < 0) {
            ip->bvars = realloc(ip->bvars, (ip->nbvars + 1) * sizeof(MGLIBufferVar));
            MGLIBufferVar *b = &ip->bvars[ip->nbvars]; memset(b, 0, sizeof *b);
            b->name = xstrdup(name); b->type = gl_type_of(m, t); b->size = 1; b->block_index = bw->block;
            src_bool_leaf_type(bw, name, t, &b->type, NULL, NULL, NULL);
            b->offset = (GLint)offset; b->array_stride = 0;
            b->matrix_stride = ty->kind == T_MATRIX ? (GLint)mstride : 0; b->is_row_major = ty->kind == T_MATRIX ? row_major : 0;
            b->top_size = bw->top_size; b->top_stride = bw->top_stride;
            found = ip->nbvars++;
            MGLIBlock *blk = &ip->ssbos[bw->block];
            blk->members = realloc(blk->members, (blk->member_count + 1) * sizeof(int));
            blk->members[blk->member_count++] = found;
        }
        ip->bvars[found].stages |= 1u << bw->stage;   /* SSBO members are walked only for stages that use the block */
        return;
    }
    int created, ui = add_uniform(bw->ip, name, &created);
    MGLIUniform *u = &bw->ip->uniforms[ui];
    if (created) {
        u->type = gl_type_of(m, t); u->size = 1; u->block_index = bw->block; u->offset = (GLint)offset; u->array_stride = 0;
        u->matrix_stride = ty->kind == T_MATRIX ? (GLint)mstride : 0; u->is_row_major = ty->kind == T_MATRIX ? row_major : 0;
        scalar_info(m, t, &u->kind, &u->cols, &u->rows);
        src_bool_leaf_type(bw, name, t, &u->type, &u->kind, &u->cols, &u->rows);
    }
    add_member(&bw->ip->ubos[bw->block], ui);
    if (bw->cur_used) u->stages |= 1u << bw->stage;   /* usage-based (see BlockWalk) */
}

static int add_block(MGLIBlock **arr, int *n, const char *name, int *created)
{
    for (int i = 0; i < *n; i++) if (!strcmp((*arr)[i].name, name)) { *created = 0; return i; }
    *arr = realloc(*arr, (*n + 1) * sizeof(MGLIBlock));
    MGLIBlock *b = &(*arr)[*n]; memset(b, 0, sizeof *b);
    b->name = xstrdup(name); b->array_element = -1;
    *created = 1;
    return (*n)++;
}

static void add_io(MGLIVar **arr, int *n, const SMod *m, int stage, const char *name, uint32_t t, GLint loc, GLint index, GLint comp, GLint patch, GLint slot)
{    for (int i = 0; i < *n; i++) if (!strcmp((*arr)[i].name, name)) { (*arr)[i].stages |= 1u << stage; return; }
    *arr = realloc(*arr, (*n + 1) * sizeof(MGLIVar));
    MGLIVar *v = &(*arr)[*n]; memset(v, 0, sizeof *v);
    v->name = xstrdup(name);
    v->type = gl_type_of(m, t);
    v->size = (m->t[t].kind == T_ARRAY) ? (GLint)m->t[t].count : 1;
    if (m->t[t].kind == T_RARRAY || (m->t[t].kind == T_ARRAY && m->t[t].count == 0)) {
        /* unsized arrayed io (tess/geometry per-vertex interfaces): the reported
         * GL_ARRAY_SIZE is the interface's vertex count (GL_MAX_PATCH_VERTICES
         * for tessellation; a geometry input primitive is queried per-vertex and
         * reported as a single element, as the CTS reference does for points) */
        v->size = (stage == _GEOMETRY_SHADER) ? 1 : MGL_TESS_MAX_PATCH_VERTICES;
    }
    v->location = loc; v->index = index; v->component = comp; v->is_per_patch = patch;
    v->slot = slot;
    v->stages = 1u << stage;
    (*n)++;
}

/* GL 4.6 §7.3.1 enumeration of a program input/output (a user interface-block
 * member or a plain varying) whose type may be nested structs / arrays:
 *   - every leaf field is its own resource, named by the dotted path from the
 *     variable down to the field ("Block.member.field");
 *   - a bottom-level array of basic types contributes ONE resource
 *     "name[0]" with the leaf type and GL_ARRAY_SIZE = element count;
 *   - arrays of aggregates (structs / arrays) enumerate EVERY element
 *     ("name[0].field", "name[1].field", ...), each of size 1.
 * This mirrors the Khronos CTS reference algorithm
 * (es31fProgramInterfaceDefinitionUtil.cpp generateVariableTypeResourceNames),
 * which follows the spec wording; without it a struct member was reported as
 * one resource of type GL_NONE (unknown to clients -> dEQP crashes).
 * `expl` tracks whether the location chain is explicit (a layout(location=N)
 * qualifier); with `need_expl` (inputs of stages other than the vertex shader,
 * outputs of stages other than the fragment shader, §11.1.1) a non-explicit
 * leaf reports GL_LOCATION -1 (ARB_enhanced_layouts / CTS LocationValidator).
 * Returns the location following the walked member. */
static GLint io_leaf_walk(MGLIVar **arr, int *narr, const SMod *m, int stage,
                          const char *name, uint32_t t, GLint loc, GLint comp, GLint patch, GLint slot,
                          bool expl, bool need_expl)
{
    const SType *ty = &m->t[t];
    GLint rep = (need_expl && !expl) ? -1 : loc;
    if (ty->kind == T_STRUCT) {
        for (uint32_t i = 0; i < ty->nmem; i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s.%s", name, member_name(m, t, i));
            uint32_t ml;
            bool mexpl = get_dec(m, t, (int32_t)i, SpvDecorationLocation, &ml) != 0;
            if (mexpl) { loc = (GLint)ml; expl = true; }
            loc = io_leaf_walk(arr, narr, m, stage, nm, ty->mem[i], loc, comp, patch, slot, expl, need_expl);
        }
        return loc;
    }
    if (ty->kind == T_ARRAY) {
        uint32_t et = ty->elem;
        bool basic = m->t[et].kind != T_STRUCT && m->t[et].kind != T_ARRAY && m->t[et].kind != T_RARRAY;
        if (basic) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[0]", name);
            add_io(arr, narr, m, stage, nm, t, rep, 0, comp, patch, slot);
            return loc + (GLint)loc_count(m, t);
        }
        uint32_t n = ty->count ? ty->count : 1;
        for (uint32_t i = 0; i < n; i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[%u]", name, i);
            loc = io_leaf_walk(arr, narr, m, stage, nm, et, loc, comp, patch, slot, expl, need_expl);
        }
        return loc;
    }
    add_io(arr, narr, m, stage, name, t, rep, 0, comp, patch, slot);
    return loc + (GLint)loc_count(m, t);
}

/* GL-visible name of a SPIR-V BuiltIn id (gl_PerVertex / tess / fragment members) */
static const char *builtin_gl_name(uint32_t b)
{
    switch (b) {
        case SpvBuiltInPosition: return "gl_Position";
        case SpvBuiltInPointSize: return "gl_PointSize";
        case SpvBuiltInClipDistance: return "gl_ClipDistance";
        case SpvBuiltInCullDistance: return "gl_CullDistance";
        case SpvBuiltInTessLevelOuter: return "gl_TessLevelOuter";
        case SpvBuiltInTessLevelInner: return "gl_TessLevelInner";
        case SpvBuiltInLayer: return "gl_Layer";
        case SpvBuiltInViewportIndex: return "gl_ViewportIndex";
        case SpvBuiltInPrimitiveId: return "gl_PrimitiveID";
        case SpvBuiltInFragDepth: return "gl_FragDepth";
        case SpvBuiltInSampleMask: return "gl_SampleMask";
        case 5014 /* SpvBuiltInFragStencilRefEXT */: return "gl_FragStencilRef";
        default: return NULL;
    }
}

/* is `v` an interface block whose members carry BuiltIn decorations (gl_PerVertex,
 * the tessellation per-patch block)?  glslang emits those as a Block struct type with
 * BuiltIn member decorations and no BuiltIn on the variable itself. */
static bool builtin_block_var(const SMod *m, uint32_t v, uint32_t *block_type)
{
    uint32_t t = pointee(m, m->vtype[v]);
    uint32_t st = strip_arrays(m, t);
    if (m->t[st].kind != T_STRUCT || !get_dec(m, st, -1, SpvDecorationBlock, NULL)) return false;
    for (uint32_t i = 0; i < m->t[st].nmem; i++)
        if (get_dec(m, st, (int32_t)i, SpvDecorationBuiltIn, NULL)) { if (block_type) *block_type = st; return true; }
    return false;
}

/* bitmask of struct members of built-in block variable `v` referenced in function
 * bodies; ~0ull when the whole block is referenced or an index is dynamic.  For an
 * arrayed block (gl_in[] / gl_out[]) the member index is the second AccessChain
 * index.  OpAccessChain is [op, ResultType, Result, Base, Index...]: Base is w[3],
 * the first index w[4]; constant indices are OpConstant ids. */
static uint64_t builtin_block_member_use(const SMod *m, uint32_t v, bool arrayed)
{
    uint64_t used = 0;
    if (m->first_func == (size_t)-1) return 0;
    for (size_t k = m->first_func; k < m->n; k++) {
        const uint32_t *w = m->in[k].w;
        uint32_t op = w[0] & 0xffff;
        if (op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) {
            if (m->in[k].len < 5 || w[3] != v) continue;
            uint32_t mi = w[4 + (arrayed ? 1 : 0)];
            if (m->in[k].len <= 4 + (arrayed ? 1 : 0)) { used = ~(uint64_t)0; continue; }
            if (mi <= m->bound && m->isconst[mi]) {
                uint32_t c = m->cval[mi];
                used |= (c < 64) ? (1ull << c) : ~(uint64_t)0;
            } else used = ~(uint64_t)0;
        } else if (op == SpvOpLoad && m->in[k].len >= 4 && w[3] == v) used = ~(uint64_t)0;
        else if (op == SpvOpStore && m->in[k].len >= 3 && w[1] == v) used = ~(uint64_t)0;
        else if (op == SpvOpCopyMemory && m->in[k].len >= 4 && (w[1] == v || w[2] == v)) used = ~(uint64_t)0;
    }
    return used;
}

/* highest constant index used on built-in array variable `v` (gl_SampleMask,
 * gl_ClipDistance): the reported GL_ARRAY_SIZE of a built-in array is the number of
 * active elements (GL 4.6 §9.2.1), not the glslang-padded declaration size. */
static uint32_t builtin_array_active_len(const SMod *m, uint32_t v, uint32_t count)
{
    uint32_t hi = 0;
    if (m->first_func == (size_t)-1) return count ? count : 1;
    for (size_t k = m->first_func; k < m->n; k++) {
        const uint32_t *w = m->in[k].w;
        uint32_t op = w[0] & 0xffff;
        if (op != SpvOpAccessChain && op != SpvOpInBoundsAccessChain) continue;
        if (m->in[k].len < 5 || w[3] != v) continue;
        uint32_t idx = w[4];
        if (idx <= m->bound && m->isconst[idx]) { if (m->cval[idx] > hi) hi = m->cval[idx]; }
        else return count ? count : 1;
    }
    return hi + 1;
}

static bool src_explicit_qual(const char *src, const char *name, const char *dir, const char *qual);

/* GL enum -> (columns, rows, is-double) of the leaf type; false for built-ins /
 * opaque / struct leaves (callers budget those as one 4-wide slot). */
static bool io_gl_shape(GLenum t, GLint *cols, GLint *rows, bool *dbl)
{
    static const GLenum singles[] = { GL_FLOAT, GL_FLOAT_VEC2, GL_FLOAT_VEC3, GL_FLOAT_VEC4,
                                      GL_INT, GL_INT_VEC2, GL_INT_VEC3, GL_INT_VEC4,
                                      GL_UNSIGNED_INT, GL_UNSIGNED_INT_VEC2, GL_UNSIGNED_INT_VEC3,
                                      GL_UNSIGNED_INT_VEC4, GL_BOOL, GL_BOOL_VEC2, GL_BOOL_VEC3,
                                      GL_BOOL_VEC4, GL_DOUBLE, GL_DOUBLE_VEC2 };
    static const GLint  srows[]   = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2 };
    static const GLenum dsingles[] = { GL_DOUBLE_VEC3, GL_DOUBLE_VEC4 };
    for (int i = 0; i < (int)(sizeof singles / sizeof singles[0]); i++) {
        if (singles[i] != t) continue;
        *cols = 1; *rows = srows[i]; *dbl = (t == GL_DOUBLE || t == GL_DOUBLE_VEC2);
        return true;
    }
    for (int i = 0; i < 2; i++) {
        if (dsingles[i] == t) { *cols = 1; *rows = i + 3; *dbl = true; return true; }
    }
    static const GLenum mats[] = { GL_FLOAT_MAT2, GL_FLOAT_MAT3, GL_FLOAT_MAT4,
                                   GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4, GL_FLOAT_MAT3x2,
                                   GL_FLOAT_MAT3x4, GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3,
                                   GL_DOUBLE_MAT2, GL_DOUBLE_MAT3, GL_DOUBLE_MAT4,
                                   GL_DOUBLE_MAT2x3, GL_DOUBLE_MAT2x4, GL_DOUBLE_MAT3x2,
                                   GL_DOUBLE_MAT3x4, GL_DOUBLE_MAT4x2, GL_DOUBLE_MAT4x3 };
    static const GLint  mcols[] = { 2, 3, 4, 2, 2, 3, 3, 4, 4, 2, 3, 4, 2, 2, 3, 3, 4, 4 };
    static const GLint  mrows[] = { 2, 3, 4, 3, 4, 2, 4, 2, 3, 2, 3, 4, 3, 4, 2, 4, 2, 3 };
    for (int i = 0; i < 18; i++) {
        if (mats[i] == t) { *cols = mcols[i]; *rows = mrows[i]; *dbl = i >= 9; return true; }
    }
    return false;
}

static void reflect_io(MGLIProgram *ip, const SMod *m, int stage, const uint8_t *active, bool input, MGLIVar **arr, int *narr, const char *src)
{
    uint32_t sc_want = input ? SpvStorageClassInput : SpvStorageClassOutput;
    /* GL 4.6 §11.1.1 / ARB_enhanced_layouts: only vertex-shader inputs and
     * fragment-shader outputs report link-assigned locations; every other
     * interface reports -1 unless declared with layout(location=N). */
    const bool need_expl = (input && stage != _VERTEX_SHADER) || (!input && stage != _FRAGMENT_SHADER);
    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != sc_want) continue;
        uint32_t bt_probe = 0;
        bool bblock = builtin_block_var(m, v, &bt_probe);
        if (!active[v] && !bblock) continue;   /* built-in blocks gate per member below */
        uint32_t t = pointee(m, m->vtype[v]);
        bool patch = get_dec(m, v, -1, SpvDecorationPatch, NULL);
        /* Arrayed interfaces (TCS/TE/GS per-vertex in/out) stay arrays here: the
         * resource is "<name>[0]" with GL_ARRAY_SIZE = the declared length
         * (§7.3.1); the link-side slot math uses the same base location. */
        uint32_t st = strip_arrays(m, t);
        bool is_block = m->t[st].kind == T_STRUCT && get_dec(m, st, -1, SpvDecorationBlock, NULL);
        uint32_t bi;
        if (get_dec(m, v, -1, SpvDecorationBuiltIn, &bi)) {
            /* individual built-in variable (gl_VertexID, gl_FragDepth, gl_SampleMask...).
             * Built-in arrays are named by their first element with the element type and
             * the number of ACTIVE elements as the array size (GL 4.6 §9.2.1). */
            if (m->t[t].kind == T_ARRAY || m->t[t].kind == T_RARRAY) {
                uint32_t cnt = (m->t[t].kind == T_ARRAY) ? m->t[t].count : 0;
                char nm[512];
                const char *cn = builtin_gl_name(bi); if (!cn) cn = id_name(m, v);
                snprintf(nm, sizeof nm, "%s[0]", cn);
                add_io(arr, narr, m, stage, nm, t, -1, -1, 0, patch, -1);
                (*arr)[*narr - 1].size = (GLint)builtin_array_active_len(m, v, cnt ? cnt : 1);
            } else {
                add_io(arr, narr, m, stage, id_name(m, v), t, -1, -1, 0, patch, -1);
            }
            continue;
        }
        if (is_block) {
            if (bblock) {
                uint32_t bt = bt_probe;
                /* gl_PerVertex / per-patch built-in block: its members ARE program
                 * inputs/outputs (GL 4.6 §11.1.1).  Arrayed instances (gl_in[]/gl_out[])
                 * name them "<block>.<member>", plain instances name them "<member>".
                 * Members only appear when this stage actually references them. */
                uint32_t vt = pointee(m, m->vtype[v]);
                bool arrayed = (m->t[vt].kind == T_ARRAY || m->t[vt].kind == T_RARRAY);
                const char *bname = id_name(m, bt);
                if (!bname || !*bname) bname = "gl_PerVertex";
                uint64_t used = builtin_block_member_use(m, v, arrayed);
                if (getenv("MGL_IO_DEBUG"))
                    fprintf(stderr, "IODBG st%d %s bblock var %u '%s' arrayed=%d used=%llx active=%d\n",
                            stage, input ? "in" : "out", v, id_name(m, v), arrayed,
                            (unsigned long long)used, active[v]);
                for (uint32_t i = 0; i < m->t[bt].nmem; i++) {
                    uint32_t mb;
                    if (!get_dec(m, bt, (int32_t)i, SpvDecorationBuiltIn, &mb)) continue;
                    if (used != ~(uint64_t)0 && !((used >> i) & 1)) continue;
                    const char *cn = member_name(m, bt, i);
                    const char *canon = builtin_gl_name(mb);
                    if (!cn || !*cn) cn = canon;
                    if (!cn) continue;
                    uint32_t mt = m->t[bt].mem[i];
                    bool mpatch = patch || get_dec(m, bt, (int32_t)i, SpvDecorationPatch, NULL);
                    if (m->t[mt].kind == T_ARRAY || m->t[mt].kind == T_RARRAY) {
                        char nm[560];
                        snprintf(nm, sizeof nm, "%s%s%s[0]", arrayed ? bname : "", arrayed ? "." : "", cn);
                        add_io(arr, narr, m, stage, nm, mt, -1, -1, 0, mpatch, -1);
                        if ((*arr)[*narr - 1].size < 1) (*arr)[*narr - 1].size = 1;   /* unsized gl_ClipDistance[] */
                    } else {
                        char nm[560];
                        snprintf(nm, sizeof nm, "%s%s%s", arrayed ? bname : "", arrayed ? "." : "", cn);
                        add_io(arr, narr, m, stage, nm, mt, -1, -1, 0, mpatch, -1);
                    }
                }
                continue;
            }
            /* user blocks list "Block.member" and, for struct members, every
             * leaf field "Block.member.field" (io_leaf_walk, §7.3.1) */
            uint32_t loc = 0; get_dec(m, v, -1, SpvDecorationLocation, &loc);
            /* the "<interface name>." prefix is used only when the block has an
             * instance name; an anonymous block's members keep their plain names
             * (glslang names the variable after the instance name, or after the
             * block type when there is no instance name) */
            const char *iname = id_name(m, v);
            char prefix[512];
            if (iname[0] && strcmp(iname, id_name(m, st))) snprintf(prefix, sizeof prefix, "%s", id_name(m, st));
            else prefix[0] = 0;
            /* a `patch in/out` block carries the Patch decoration on its MEMBERS
             * (glslang), not on the block variable */
            bool block_patch = patch;
            bool expl = false;
            for (uint32_t i = 0; i < m->t[st].nmem; i++) {
                if (get_dec(m, st, (int32_t)i, SpvDecorationPatch, NULL)) block_patch = true;
                if (get_dec(m, st, (int32_t)i, SpvDecorationLocation, NULL)) expl = true;
            }
            /* the block-level Location decoration glslang invents is only
             * meaningful (explicit) when the source has layout(location) */
            if (get_dec(m, v, -1, SpvDecorationLocation, NULL) &&
                src_explicit_qual(src, id_name(m, st), input ? "in" : "out", "location")) expl = true;
            if (getenv("MGL_IO_DEBUG"))
                fprintf(stderr, "IODBG st%d %s ublock var %u type %u '%s' iname='%s' loc=%u prefix='%s' patch=%d expl=%d\n",
                        stage, input ? "in" : "out", v, st, id_name(m, st), iname, loc, prefix, block_patch, expl);
            for (uint32_t i = 0; i < m->t[st].nmem; i++) {
                char nm[512];
                /* geom r88 kind 1: this struct is a flattened ARRAY OF PATCH
                 * INTERFACE BLOCKS; members start + k*stride are element k of
                 * the same original member.  GL sees ONE resource per original
                 * member (the CTS reference: "no array index specifier even if
                 * the interface is declared as an array of instances"). */
                int f1 = -1;
                for (int f = 0; f < ip->nflat; f++)
                    if (ip->flat[f].stage == stage && ip->flat[f].kind == 1 && ip->flat[f].st == st &&
                        i >= ip->flat[f].start && ip->flat[f].stride &&
                        (i - ip->flat[f].start) % ip->flat[f].stride == 0 &&
                        (i - ip->flat[f].start) / ip->flat[f].stride < ip->flat[f].count) { f1 = f; break; }
                if (f1 >= 0) {
                    if ((i - ip->flat[f1].start) / ip->flat[f1].stride == 0) {
                        if (prefix[0]) snprintf(nm, sizeof nm, "%s.%s", prefix, ip->flat[f1].name);
                        else snprintf(nm, sizeof nm, "%s", ip->flat[f1].name);
                        uint32_t ml; bool mexpl = get_dec(m, st, (int32_t)i, SpvDecorationLocation, &ml) != 0;
                        if (mexpl) loc = (GLint)ml;
                        loc = io_leaf_walk(arr, narr, m, stage, nm, ip->flat[f1].mtype, (GLint)loc, 0,
                                           block_patch, -1, expl || mexpl, need_expl);
                    }
                    /* elements >0 of the same member: same GL resource */
                    continue;
                }
                /* a flattened array-of-matrix group (SPIRV-Cross cannot emit
                 * them): reassemble the ORIGINAL single member resource from
                 * the recorded array type; the other members of the group are
                 * the same resource */
                const char *gname = NULL;
                uint32_t fl = 0, fmt = 0;
                for (int f = 0; f < ip->nflat; f++)
                    if (ip->flat[f].stage == stage && ip->flat[f].st == st &&
                        ip->flat[f].start == i) {
                        fl = ip->flat[f].count; fmt = ip->flat[f].mtype;
                        gname = ip->flat[f].name;
                        break;
                    }
                uint32_t mt = m->t[st].mem[i];
                int fvar = 0;
                for (int f = 0; f < ip->nflat; f++)
                    if (ip->flat[f].stage == stage && ip->flat[f].st == st &&
                        ip->flat[f].start == i && ip->flat[f].var) fvar = ip->flat[f].var;
                if (fl) {
                    mt = fmt;
                    /* a flattened plain variable is one resource named after
                     * the variable itself; a flattened block member keeps the
                     * "Block.member" prefix */
                    if (prefix[0] && !fvar) snprintf(nm, sizeof nm, "%s.%s", prefix, gname);
                    else snprintf(nm, sizeof nm, "%s", gname);
                } else {
                    bool inskip = false;
                    for (int f = 0; f < ip->nflat; f++)
                        if (ip->flat[f].stage == stage && ip->flat[f].st == st &&
                            i > ip->flat[f].start && i < ip->flat[f].start + ip->flat[f].count)
                            inskip = true;
                    if (inskip) continue;
                    if (prefix[0]) snprintf(nm, sizeof nm, "%s.%s", prefix, member_name(m, st, i));
                    else snprintf(nm, sizeof nm, "%s", member_name(m, st, i));
                }
                uint32_t ml; bool mexpl = get_dec(m, st, (int32_t)i, SpvDecorationLocation, &ml) != 0;
                if (mexpl) loc = (GLint)ml;
                loc = io_leaf_walk(arr, narr, m, stage, nm, mt, (GLint)loc, 0, block_patch, -1,
                                   expl || mexpl, need_expl);
                if (fl) i += fl - 1;   /* skip the rest of the group */
            }
            continue;
        }
        uint32_t loc = 0, idx = 0, comp = 0;
        get_dec(m, v, -1, SpvDecorationLocation, &loc);
        get_dec(m, v, -1, SpvDecorationIndex, &idx);
        get_dec(m, v, -1, SpvDecorationComponent, &comp);
        if (m->t[t].kind == T_STRUCT ||
            (m->t[t].kind == T_ARRAY &&
             (m->t[m->t[t].elem].kind == T_STRUCT || m->t[m->t[t].elem].kind == T_ARRAY ||
              m->t[m->t[t].elem].kind == T_RARRAY))) {
            /* struct / array-of-aggregate varying: enumerate the leaves
             * ("name.field", "name[0].field", §7.3.1).  Vertex inputs keep their
             * assigned GL location base (attrib_gl) like the basic-type path. */
            GLint gl = (GLint)loc;
            if (stage == _VERTEX_SHADER && input)
                for (int k = 0; k < ip->nattrib_gl; k++)
                    if (ip->attrib_gl[k].var == v) { gl = ip->attrib_gl[k].gl; break; }
            bool expl = get_dec(m, v, -1, SpvDecorationLocation, NULL) &&
                        src_explicit_qual(src, id_name(m, v), input ? "in" : "out", "location");
            io_leaf_walk(arr, narr, m, stage, id_name(m, v), t, gl, (GLint)comp, patch, -1, expl, need_expl);
            continue;
        }
        char nm[512];
        if (m->t[t].kind == T_ARRAY) snprintf(nm, sizeof nm, "%s[0]", id_name(m, v));
        else snprintf(nm, sizeof nm, "%s", id_name(m, v));
        /* vertex inputs: `loc` (the Location decoration) is the internal slot
           base; the GL location comes from the link-time assignment */
        GLint gl = (GLint)loc, slot = -1;
        if (stage == _VERTEX_SHADER && input) {
            slot = (GLint)loc;
            for (int k = 0; k < ip->nattrib_gl; k++)
                if (ip->attrib_gl[k].var == v) { gl = ip->attrib_gl[k].gl; break; }
        } else if (need_expl && !src_explicit_qual(src, id_name(m, v), input ? "in" : "out", "location")) {
            gl = -1;   /* §11.1.1: no effective location without layout(location) */
        }
        add_io(arr, narr, m, stage, nm, t,
               gl, (stage == _FRAGMENT_SHADER && !input) ? (GLint)idx : 0, (GLint)comp, patch, slot);
    }
}

/* active global variables and accessed default-block members via SPIRV-Cross */
static void compute_active(StageCtx *s, const uint32_t *words, size_t n)
{
    SMod *r = s->r;
    s->active = zalloc(r->bound + 1);
    s->dub_active = zalloc(s->ndub + 1);
    s->bused = zalloc((r->bound + 1) * sizeof(uint64_t));
    spvc_context c = NULL; spvc_parsed_ir ir = NULL; spvc_compiler comp = NULL;
    bool ok = false;
    if (spvc_context_create(&c) == SPVC_SUCCESS &&
        spvc_context_parse_spirv(c, words, n, &ir) == SPVC_SUCCESS &&
        spvc_context_create_compiler(c, SPVC_BACKEND_NONE, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) == SPVC_SUCCESS) {
        spvc_set set = NULL;
        if (spvc_compiler_get_active_interface_variables(comp, &set) == SPVC_SUCCESS) {
            spvc_resources res = NULL;
            if (spvc_compiler_create_shader_resources_for_active_variables(comp, &res, set) == SPVC_SUCCESS) {
                for (int rt = SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; rt <= SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE; rt++) {
                    const spvc_reflected_resource *list; size_t cnt;
                    if (spvc_resources_get_resource_list_for_type(res, (spvc_resource_type)rt, &list, &cnt) != SPVC_SUCCESS) continue;
                    for (size_t i = 0; i < cnt; i++) if (list[i].id <= r->bound) s->active[list[i].id] = 1;
                }
                ok = true;
            }
            /* built-in inputs/outputs are not in SPIRV-Cross resource lists: mark used ones
             * from the function bodies (Load's pointer is w[2], Store's pointer w[1],
             * an AccessChain's base w[2]; earlier code compared w[3] and missed reads).
             * Built-in BLOCK variables (gl_PerVertex / gl_in[]) have no BuiltIn decoration
             * at all — they are detected structurally. */
            for (uint32_t v = 1; v <= r->bound; v++)
                if (r->isvar[v] && r->vglobal[v] && (r->vsc[v] == SpvStorageClassInput || r->vsc[v] == SpvStorageClassOutput) &&
                    (get_dec(r, v, -1, SpvDecorationBuiltIn, NULL) || builtin_block_var(r, v, NULL))) {
                    /* conservative: referenced anywhere in function bodies
                     * (OpLoad's pointer is w[3], OpStore's pointer w[1], an
                     * AccessChain's base w[3] — both have a Result Type word) */
                    for (size_t k = r->first_func; k < r->n && !s->active[v]; k++) {
                        uint32_t op = r->in[k].w[0] & 0xffff;
                        uint32_t len = r->in[k].len;
                        if (op == SpvOpLoad && len >= 4 && r->in[k].w[3] == v) s->active[v] = 1;
                        if (op == SpvOpStore && len >= 3 && r->in[k].w[1] == v) s->active[v] = 1;
                        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && len >= 5 && r->in[k].w[3] == v) s->active[v] = 1;
                        if (op == SpvOpCopyMemory && len >= 4 && (r->in[k].w[1] == v || r->in[k].w[2] == v)) s->active[v] = 1;
                    }
                }
        }
        if (s->dub_var) {
            const spvc_buffer_range *ranges; size_t nr;
            if (spvc_compiler_get_active_buffer_ranges(comp, s->dub_var, &ranges, &nr) == SPVC_SUCCESS)
                for (size_t i = 0; i < nr; i++) if (ranges[i].index < s->ndub) s->dub_active[ranges[i].index] = 1;
        }
        /* per UBO/SSBO variable: which top-level members this stage's module actually
         * accesses (REFERENCED_BY_*_SHADER is usage-based, GL 4.6 §9.2 — a declared but
         * untouched block must report 0 for the stage).  SPIRV-Cross reports the
         * top-level member containing each access. */
        for (uint32_t v = 1; v <= r->bound; v++) {
            if (!r->isvar[v] || !r->vglobal[v]) continue;
            if (r->vsc[v] != SpvStorageClassUniform && r->vsc[v] != SpvStorageClassStorageBuffer) continue;
            const spvc_buffer_range *ranges; size_t nr;
            if (spvc_compiler_get_active_buffer_ranges(comp, v, &ranges, &nr) != SPVC_SUCCESS) {
                s->bused[v] = MGL_BUSED_UNKNOWN;
                continue;
            }
            for (size_t i = 0; i < nr; i++) {
                if (ranges[i].index >= 64) { s->bused[v] = MGL_BUSED_UNKNOWN; break; }  /* beyond the mask: conservative */
                s->bused[v] |= 1ull << ranges[i].index;
            }
        }
    }
    if (!ok) {
        for (uint32_t v = 1; v <= r->bound; v++) if (r->isvar[v] && r->vglobal[v]) s->active[v] = 1;
        for (uint32_t k = 0; k < s->ndub; k++) s->dub_active[k] = 1;
        for (uint32_t v = 1; v <= r->bound; v++) if (r->isvar[v] && r->vglobal[v]) s->bused[v] = MGL_BUSED_UNKNOWN;
    }
    /* Uniform blocks stay ACTIVE when they are declared but none of their members is
     * referenced: GL 4.6 §7.6.2 ("resources may still be consumed ... implementations
     * may consider them active"), and glslang does not strip the block from the SPIR-V,
     * so conformant drivers list them (KHR-GL46.shaders.uniform_block.random.* looks up
     * exactly such blocks with glGetUniformBlockIndex).  SPIRV-Cross's active-interface
     * analysis DCEs them — keep every declared UBO variable of the module.  (SSBOs and
     * default-block members keep the exact-active rules.) */
    for (uint32_t v = 1; v <= r->bound; v++) {
        if (!r->isvar[v] || !r->vglobal[v] || r->vsc[v] != SpvStorageClassUniform) continue;
        uint32_t st = strip_arrays(r, pointee(r, r->vtype[v]));
        if (get_dec(r, st, -1, SpvDecorationBufferBlock, NULL)) continue;   /* SSBO */
        s->active[v] = 1;
    }
    if (c) spvc_context_destroy(c);
}

static bool name_is_hidden(const char *n) { return !strncmp(n, "_mgl_sr_", 8); }

/* for an array-of-blocks variable: bitmask of ELEMENT indices referenced in
 * function bodies (~0ull when an element index is dynamic or the whole array is
 * copied).  GL 4.6 §9.2 makes REFERENCED_BY_*_SHADER per RESOURCE: each element
 * of a uniform-block array is its own resource (CTS uniform-block-types:
 * "TrickyBlock[1]" must report 0 when only e[0] is used). */
static uint64_t block_elem_use(const SMod *m, uint32_t v)
{
    uint64_t used = 0;
    if (m->first_func == (size_t)-1) return 0;
    for (size_t k = m->first_func; k < m->n; k++) {
        const uint32_t *w = m->in[k].w;
        uint32_t op = w[0] & 0xffff;
        if (op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain || op == SpvOpPtrAccessChain) {
            if (m->in[k].len < 5 || w[3] != v) continue;
            uint32_t e = w[4];
            if (e <= m->bound && m->isconst[e]) {
                uint32_t c = m->cval[e];
                used |= (c < 64) ? (1ull << c) : ~(uint64_t)0;
            } else used = ~(uint64_t)0;
        } else if (op == SpvOpLoad && m->in[k].len >= 4 && w[3] == v) used = ~(uint64_t)0;
        else if (op == SpvOpStore && m->in[k].len >= 3 && w[1] == v) used = ~(uint64_t)0;
        else if (op == SpvOpCopyMemory && m->in[k].len >= 4 && (w[1] == v || w[2] == v)) used = ~(uint64_t)0;
    }
    return used;
}

static void reflect_stage(MGLIProgram *ip, StageCtx *s, int stage, const char *src)
{
    const SMod *m = s->r;
    /* default block members */
    for (uint32_t k = 0; k < s->ndub && s->dub_var; k++) {
        /* GL 4.6 11.1.1 / ARB_explicit_uniform_location: a uniform declared with an
         * explicit location stays ACTIVE (queryable, settable) even when the shader
         * never references it — glslang keeps the OpVariable + Location decoration,
         * SPIRV-Cross just does not list it as an active buffer range.  Members
         * without an explicit location that are unreferenced stay dropped. */
        if (!s->dub_active[k] && s->dub[k].explicit_loc < 0) continue;
        DubMember *d = &s->dub[k];
        uint32_t mt = m->t[strip_arrays(m, pointee(m, m->vtype[s->dub_var]))].mem[k];
        int cursor = 0, elem = 0, lki = 0;
        if (getenv("MGL_FP64_DUMP")) { FILE*d_ = fopen("/tmp/fp64rs.txt","a"); if (d_) { fprintf(d_, "reflect %s d->offset=%u active=%d k=%u ndub=%u dubvar=%u\n", d->name, d->offset, s->dub_active ? s->dub_active[k] : -1, k, s->ndub, s->dub_var); fclose(d_); } }
        walk_default(ip, m, stage, d->name, mt, d->offset, d->explicit_loc, &cursor, 0, -1, &elem, name_is_hidden(d->name), d->leaf, &lki);
    }
    /* bool/bvecN block members: recover the GLSL-declared types from the stage source */
    MGLBoolPaths *bools = mgl_cts_bool_block_members(src);

    /* opaque uniforms + UBO + SSBO + atomic counters */
    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || !s->active[v]) continue;
        uint32_t t = pointee(m, m->vtype[v]);
        uint32_t st = strip_arrays(m, t);
        if (m->vsc[v] == SpvStorageClassUniformConstant) {
            if (!is_opaque(m, t)) {
                /* fallback path (uniform pointers escaped the packer): still reflect the
                 * uniform so GL queries and glUniform* work against the CPU shadow */
                if (s->dub_var) continue;
                int cursor = 0, elem = 0, lki = 0;
                const uint8_t *lk = NULL;
                for (uint32_t q = 0; q < s->ndub; q++) if (s->dub[q].var == v) lk = s->dub[q].leaf;
                walk_default(ip, m, stage, id_name(m, v), t, 0, -1, &cursor, v, -1, &elem, 0, lk, &lki);
                continue;
            }
            uint32_t l; int el = get_dec(m, v, -1, SpvDecorationLocation, &l) ? (int)l : -1;
            int cursor = 0, elem = 0;
            int rt = m->t[st].kind == T_IMAGE && m->t[st].sampled == 2 ? SPVC_RESOURCE_TYPE_STORAGE_IMAGE :
                     (m->t[st].kind == T_IMAGE ? SPVC_RESOURCE_TYPE_SEPARATE_IMAGE : SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
            int lki = 0;
            const uint8_t *lk = NULL;
            for (uint32_t q = 0; q < s->ndub; q++) if (s->dub[q].var == v) lk = s->dub[q].leaf;
            walk_default(ip, m, stage, id_name(m, v), t, 0, el, &cursor, v, rt, &elem, 0, lk, &lki);
            continue;
        }
        if (m->vsc[v] == SpvStorageClassAtomicCounter) {
            uint32_t binding = 0, off = 0;
            get_dec(m, v, -1, SpvDecorationBinding, &binding);
            get_dec(m, v, -1, SpvDecorationOffset, &off);
            /* glslang auto-mapped bindings are not explicit: GL defaults every atomic counter to binding 0 */
            if (binding && !src_explicit_qual(src, id_name(m, v), "uniform", "binding")) binding = 0;
            int ab = -1;
            for (int i = 0; i < ip->nabufs; i++) if (ip->abufs[i].binding == (GLint)binding) ab = i;
            if (ab < 0) {
                ip->abufs = realloc(ip->abufs, (ip->nabufs + 1) * sizeof(MGLIAtomicBuffer));
                memset(&ip->abufs[ip->nabufs], 0, sizeof(MGLIAtomicBuffer));
                ip->abufs[ip->nabufs].binding = (GLint)binding;
                ab = ip->nabufs++;
            }
            ip->abufs[ab].stages |= 1u << stage;
            char nm[512];
            uint32_t cnt = m->t[t].kind == T_ARRAY ? m->t[t].count : 1;
            if (m->t[t].kind == T_ARRAY) snprintf(nm, sizeof nm, "%s[0]", id_name(m, v)); else snprintf(nm, sizeof nm, "%s", id_name(m, v));
            int created, ui = add_uniform(ip, nm, &created);
            MGLIUniform *u = &ip->uniforms[ui];
            if (created) {
                u->type = GL_UNSIGNED_INT_ATOMIC_COUNTER; u->size = (GLint)cnt; u->offset = (GLint)off;
                u->array_stride = m->t[t].kind == T_ARRAY ? 4 : 0; u->matrix_stride = 0;
                u->atomic_buffer_index = ab; u->kind = MGLI_K_UINT; u->cols = u->rows = 1; u->opaque_class = 3;
                MGLIAtomicBuffer *a = &ip->abufs[ab];
                a->members = realloc(a->members, (a->member_count + 1) * sizeof(int));
                a->members[a->member_count++] = ui;
                if ((GLint)(off + 4 * cnt) > a->data_size) a->data_size = (GLint)(off + 4 * cnt);
            }
            u->stages |= 1u << stage;
            continue;
        }
        bool ubo = m->vsc[v] == SpvStorageClassUniform && !get_dec(m, st, -1, SpvDecorationBufferBlock, NULL);
        bool ssbo = (m->vsc[v] == SpvStorageClassUniform && get_dec(m, st, -1, SpvDecorationBufferBlock, NULL)) ||
                    m->vsc[v] == SpvStorageClassStorageBuffer;
        if (!ubo && !ssbo) continue;
        if (v == s->dub_var) continue;
        const char *bname = id_name(m, st);
        const char *iname = id_name(m, v);
        uint32_t binding = 0; get_dec(m, v, -1, SpvDecorationBinding, &binding);
        /* glslang auto-mapped bindings are not explicit: GL defaults un-bound blocks to binding point 0 */
        if (binding && !src_explicit_qual(src, bname, ubo ? "uniform" : "buffer", "binding")) binding = 0;
        uint32_t nel = m->t[t].kind == T_ARRAY ? m->t[t].count : 0;
        uint64_t eused = ~(uint64_t)0;
        if (nel) {
            eused = block_elem_use(m, v);
            if (eused == ~(uint64_t)0) { /* dynamic element index: every element counts */ }
        }

        /* GL 4.6 4.4.5 (ARB_enhanced_layouts): explicitly offset members of one block
         * must not overlap.  glslang accepts overlapping offsets (both compile and
         * link of the shader alone succeed), so the driver's link must reject the
         * program.  Overlap between an explicit member and an auto-laid-out one is
         * impossible (auto layout resumes after the previous member's end). */
        if (!ip->block_member_overlap && (ubo || ssbo) && m->t[st].kind == T_STRUCT && m->t[st].nmem >= 2) {
            for (uint32_t i = 0; i < m->t[st].nmem && !ip->block_member_overlap; i++) {
                uint32_t oi = 0, oj = 0;
                if (!get_dec(m, st, (int32_t)i, SpvDecorationOffset, &oi)) continue;
                int rmi = 0; uint32_t msi = 0;
                if (get_dec(m, st, (int32_t)i, SpvDecorationRowMajor, NULL)) rmi = 1;
                get_dec(m, st, (int32_t)i, SpvDecorationMatrixStride, &msi);
                uint32_t si = block_type_size(m, m->t[st].mem[i], msi, rmi);
                if (!si) continue;   /* runtime array: unbounded, overlap is not
                                      * decidable — glslang lays two runtime arrays of
                                      * one block at increasing offsets with the first
                                      * member still "open" */
                for (uint32_t j = i + 1; j < m->t[st].nmem && !ip->block_member_overlap; j++) {
                    if (!get_dec(m, st, (int32_t)j, SpvDecorationOffset, &oj)) continue;
                    int rmj = 0; uint32_t msj = 0;
                    if (get_dec(m, st, (int32_t)j, SpvDecorationRowMajor, NULL)) rmj = 1;
                    get_dec(m, st, (int32_t)j, SpvDecorationMatrixStride, &msj);
                    uint32_t sj = block_type_size(m, m->t[st].mem[j], msj, rmj);
                    if (!sj) continue;
                    if (oi < oj + sj && oj < oi + si) {
                        ip->block_member_overlap = 1;
                        snprintf(ip->block_overlap_msg, sizeof ip->block_overlap_msg,
                                 "block member offsets overlap: '%s' member %u "
                                 "(offset %u, size %u) overlaps member %u (offset %u, size %u)",
                                 bname[0] ? bname : "<unnamed block>",
                                 i, oi, si, j, oj, sj);
                    }
                }
            }
        }
        for (uint32_t e = 0; e < (nel ? nel : 1); e++) {
            char nm[512];
            if (nel) snprintf(nm, sizeof nm, "%s[%u]", bname, e); else snprintf(nm, sizeof nm, "%s", bname);
            int created;
            int bi = ubo ? add_block(&ip->ubos, &ip->nubos, nm, &created) : add_block(&ip->ssbos, &ip->nssbos, nm, &created);
            MGLIBlock *b = ubo ? &ip->ubos[bi] : &ip->ssbos[bi];
            if (created) {
                b->binding = (GLint)(binding + e);
                b->array_element = nel ? (int)e : -1;
                uint32_t sz = block_type_size(m, st, 0, 0);
                b->data_size = (GLint)(ubo ? align_up(sz, 16) : sz);
            }
            /* usage-based stage bits (GL 4.6 §9.2: REFERENCED_BY_X_SHADER is 1 only if
             * the block "contains at least one variable referenced by" X).  The block
             * itself stays listed for declared-but-unused UBOs (see compute_active). */
            uint64_t bused = (s->bused && v <= m->bound) ? s->bused[v] : MGL_BUSED_UNKNOWN;
            if (bused == 0) {
                /* belt and braces for accesses the member-range analysis cannot see
                 * (whole-block load/store): any function-body reference to the variable */
                for (size_t k = m->first_func; k < m->n && bused == 0; k++) {
                    uint32_t op = m->in[k].w[0] & 0xffff;
                    if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain || op == SpvOpPtrAccessChain ||
                         op == SpvOpLoad) && m->in[k].w[3] == v) bused = MGL_BUSED_UNKNOWN;
                    else if (op == SpvOpStore && m->in[k].w[1] == v) bused = MGL_BUSED_UNKNOWN;
                    else if (op == SpvOpCopyMemory && (m->in[k].w[1] == v || m->in[k].w[2] == v)) bused = MGL_BUSED_UNKNOWN;
                }
            }
            if (bused && (eused == ~(uint64_t)0 || ((eused >> e) & 1))) b->stages |= 1u << stage;
            b->spv_id[stage] = v;
            bool e_used = !nel || eused == ~(uint64_t)0 || ((eused >> e) & 1);
            BlockWalk bw = { ip, m, stage, bi, ssbo, 0, 1, 0, bools,
                             (bused == MGL_BUSED_UNKNOWN || !e_used) ? 0 : bused,
                             bused == MGL_BUSED_UNKNOWN && e_used, false, st };            {
                char prefix[512];
                /* GL names block members "<block type name>.<member>" only when the block
                 * has an instance name; anonymous block members keep their plain names
                 * (this is what the Khronos CTS reference layout expects) */
                if (iname[0]) snprintf(prefix, sizeof prefix, "%s", bname);
                else prefix[0] = 0;
                walk_block(&bw, prefix, st, 0, 0, 0, 0);
            }
        }
    }

    mgl_cts_bool_paths_free(bools);
}

/* assign locations to default-block / opaque uniforms (explicit first, then by order).
 * Fails (GL link error) when two different default-block uniforms end up on the same location. */
static bool assign_uniform_locations(MGLIProgram *ip)
{
    int maxloc = 0;
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->block_index >= 0 || u->opaque_class == 3 || u->hidden) continue;
        if (u->location >= 0 && u->location + u->size > maxloc) maxloc = u->location + u->size;
    }
    uint8_t *used = zalloc((size_t)maxloc + 1);
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->block_index >= 0 || u->opaque_class == 3 || u->hidden || u->location < 0) continue;
        for (int k = 0; k < u->size; k++) used[u->location + k] = 1;
    }
    int next = 0;
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->block_index >= 0 || u->opaque_class == 3 || u->hidden) { u->location = -1; continue; }
        if (u->location >= 0) continue;
        for (;;) {
            bool ok = true;
            for (int k = 0; k < u->size; k++) if (next + k < maxloc && used[next + k]) { ok = false; break; }
            if (ok) break;
            next++;
        }
        u->location = next;
        next += u->size;
    }
    free(used);
    int n = 0;
    for (int i = 0; i < ip->nuniforms; i++) if (ip->uniforms[i].location >= 0 && ip->uniforms[i].location + ip->uniforms[i].size > n)
        n = ip->uniforms[i].location + ip->uniforms[i].size;
    /* GL 4.6 11.1.1: the link fails when a uniform (explicit or implicitly
     * assigned) reaches past GL_MAX_UNIFORM_LOCATIONS — e.g. a 1024-element
     * array at location 0 leaves no room for one more implicit uniform */
    if (n > ip->uniform_loc_max) ip->uniform_loc_overflow = 1;
    ip->locs = zalloc((size_t)(n ? n : 1) * sizeof(*ip->locs));
    ip->nlocs = n;
    for (int l = 0; l < n; l++) ip->locs[l].uniform = -1;
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->location < 0) continue;
        for (int k = 0; k < u->size; k++) {
            if (ip->locs[u->location + k].uniform >= 0 && ip->locs[u->location + k].uniform != i) {
                free(ip->locs); ip->locs = NULL; ip->nlocs = 0;
                return false; /* GL 4.6: linking fails on colliding explicit uniform locations */
            }
            ip->locs[u->location + k].uniform = i;
            ip->locs[u->location + k].element = k;
        }
    }
    return true;
}


#pragma mark - link entry

/* compile the patched module once with SPIRV-Cross MSL; false (and a log line) if it is rejected */
/* does this module contain a Float64 type? (software fp64: kept in the GL
 * view, lowered to u64 + soft-float calls by fp64.c before MSL) */
static bool module_has_fp64(const uint32_t *words, size_t n)
{
    if (n < 6) return false;
    for (size_t i = 5; i + 2 < n; i += (words[i] >> 16) ? (words[i] >> 16) : 1)
        if ((words[i] & 0xffff) == SpvOpTypeFloat && words[i + 2] == 64)
            return true;
    return false;
}

static bool msl_accepts(const uint32_t *words, size_t n, int stage)
{
        fprintf(stderr, "MGL msl_accepts: stage=%d n=%zu\n", stage, n);
    /* SPIRV-Cross rejects double types in buffers, which the GL view of a
     * software-fp64 module legitimately contains (fp64.c lowers it later). */
    if (module_has_fp64(words, n)) return true;

    /* tess: the link driver compiles the TCS as a multi-patch compute kernel
     * (gl_in[] becomes a raw buffer: dynamic indexing is legal there), and an
     * isolines TES is swapped to the quad domain before MSL.  Test-compile with
     * those transforms so the acceptance gate matches the real pipeline —
     * otherwise the patched module (with the default-uniform block) is dropped
     * and the stage's uniforms never bind (tessellation max_in_out_attributes). */
    uint32_t *swapped = NULL;
    size_t swapped_n = 0;
    if (stage == _TESS_EVALUATION_SHADER &&
        (swapped = mgl_tess_prepare_spirv(words, n, &swapped_n)) != NULL)
    {
            fprintf(stderr, "MGL msl_accepts: swapped n=%zu first=%08x\n",
                    swapped_n, swapped_n ? swapped[0] : 0);
        if (mgl_tess_module_declined(swapped, swapped_n))
        {
                fprintf(stderr, "MGL msl_accepts: declined marker seen (n=%zu first=%08x)\n",
                        swapped_n, swapped_n ? swapped[0] : 0);
            /* the module needs transforms SPIRV-Cross cannot compile without
             * aborting (whole struct-typed interface loads): do not compile
             * anything here — the link reports a clean stage error */
            free(swapped);
            return false;
        }
        words = swapped;
        n = swapped_n;
    }
    {
        /* the same MSL normalizations program.c installs (vote opcodes,
         * gl_DrawID) — the gate must accept what the real compile accepts */
        uint32_t *norm = mgl_spirv_prepare_for_msl(words, n, &swapped_n);
        if (norm)
        {
            if (swapped) free(swapped);
            swapped = norm;          /* freed below with `swapped` */
            words = swapped;
            n = swapped_n;
        }
    }

    spvc_context c = NULL; spvc_parsed_ir ir = NULL; spvc_compiler comp = NULL; spvc_compiler_options opt = NULL;
    const char *res = NULL;
    bool ok = spvc_context_create(&c) == SPVC_SUCCESS &&
              spvc_context_parse_spirv(c, words, n, &ir) == SPVC_SUCCESS &&
              spvc_context_create_compiler(c, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) == SPVC_SUCCESS &&
              spvc_compiler_create_compiler_options(comp, &opt) == SPVC_SUCCESS &&
              spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3, 1, 0)) == SPVC_SUCCESS;
    if (ok && stage == _TESS_CONTROL_SHADER)
    {
        /* the same multi-patch options program.c installs for the real compile */
        ok = spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER,
                                            SPVC_TRUE) == SPVC_SUCCESS &&
             spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_MULTI_PATCH_WORKGROUP,
                                            SPVC_TRUE) == SPVC_SUCCESS &&
             spvc_compiler_install_compiler_options(comp, opt) == SPVC_SUCCESS &&
             spvc_compiler_compile(comp, &res) == SPVC_SUCCESS && res;
    }
    else if (ok)
    {
        ok = spvc_compiler_install_compiler_options(comp, opt) == SPVC_SUCCESS &&
             spvc_compiler_compile(comp, &res) == SPVC_SUCCESS && res;
    }
    if (!ok) fprintf(stderr, "MGL introspect: stage %d: patched SPIR-V rejected by SPIRV-Cross (%s), using the original module\n",
                     stage, c ? spvc_context_get_last_error_string(c) : "no context");
    if (c) spvc_context_destroy(c);
    free(swapped);
    return ok;
}

static void dump_words(const uint32_t *w, size_t n, Program *ptr, int stage, const char *tag)
{
    const char *dir = getenv("MGL_INTROSPECT_DUMP");
    if (!dir) return;
    char path[1024]; snprintf(path, sizeof path, "%s/prog%u_stage%d_%s.spv", dir, ptr->name, stage, tag);
    FILE *f = fopen(path, "wb"); if (f) { fwrite(w, 4, n, f); fclose(f); }
}


/* is `name` declared with layout(qual) as an `in`/`out`/`uniform`/`buffer` variable in the GLSL source? */
static bool src_explicit_qual(const char *src, const char *name, const char *dir, const char *qual)
{
    if (!src) return true;
    size_t ql = strlen(qual);
    size_t n = strlen(src);
    char *b = malloc(n + 1);
    /* strip comments */
    size_t o = 0;
    for (size_t i = 0; i < n; ) {
        if (src[i] == '/' && src[i + 1] == '/') { while (i < n && src[i] != '\n') i++; continue; }
        if (src[i] == '/' && src[i + 1] == '*') { i += 2; while (i < n && !(src[i] == '*' && src[i + 1] == '/')) i++; i += 2; continue; }
        b[o++] = src[i++];
    }
    b[o] = 0;
    bool found = false, expl = false;
    char *st = b;
    for (char *c = b; ; c++) {
        if (*c == ';' || *c == '{' || *c == '}' || *c == 0) {
            char save = *c; *c = 0;
            /* tokens */
            bool has_dir = false, has_qual = false; char last[256] = "";
            for (char *t = st; *t; ) {
                if (isalnum((unsigned char)*t) || *t == '_') {
                    char *e = t; while (isalnum((unsigned char)*e) || *e == '_') e++;
                    size_t l = (size_t)(e - t);
                    if (l == strlen(dir) && !strncmp(t, dir, l)) has_dir = true;
                    if (l == ql && !strncmp(t, qual, l)) has_qual = true;
                    if (l < sizeof last) { memcpy(last, t, l); last[l] = 0; }
                    t = e;
                } else if (*t == '[') { while (*t && *t != ']') t++; }
                else t++;
            }
            if (has_dir && !strcmp(last, name)) { found = true; expl = has_qual; }
            *c = save;
            if (!save) break;
            st = c + 1;
        }
    }
    free(b);
    return found ? expl : true;
}

/* is `name` declared with layout(location) as an `in`/`out` variable in the GLSL source? */
static bool src_explicit_location(const char *src, const char *name, const char *dir)
{
    return src_explicit_qual(src, name, dir, "location");
}

static const char *stage_source(GLMContext ctx, Program *program, int stage)
{
    Program *real = findProgram(ctx, program->name);
    if (!real) return NULL;
    MGLProgramObj *o = MGL_PROGRAM_OBJ(real);
    for (int i = 0; i < o->num_attached; i++)
        if (o->attached[i] && (int)o->attached[i]->glm_type == stage) return o->attached[i]->src;
    return NULL;
}

uint32_t *mgl_introspect_patch_stage(GLMContext ctx, Program *program, int stage, const uint32_t *words,
                                     size_t nwords, size_t *out_words)
{
    if (!program || stage < 0 || stage >= MGLI_STAGES) return NULL;
    MGLIProgram *ip = mgli_get(ctx, program->name);
    if (!ip) return NULL;
    StageCtx sc[MGLI_STAGES];
    uint32_t *out = NULL;
    for (int attempt = 0; attempt < 2 && !out; attempt++) {
        memset(sc, 0, sizeof sc);
        sc[stage].m = mod_parse(words, nwords);
        if (!sc[stage].m) return NULL;
        sc[stage].present = 1;
        sc[stage].ip = ip;
        sc[stage].no_pack = attempt;
        /* locations glslang's map_io invented are not "explicit": bindings may override them.
         * The same is true for uniforms: glslang numbers every loose uniform, but in GL only
         * layout(location=N) counts — invented numbers must not leak into the location table. */
        const char *src = stage_source(ctx, program, stage);
        SMod *m = sc[stage].m;
        if (src) {
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || !get_dec(m, v, -1, SpvDecorationLocation, NULL)) continue;
                if (m->vsc[v] == SpvStorageClassUniformConstant) {
                    if (!src_explicit_location(src, id_name(m, v), "uniform"))
                        kill_decs(m, v, -1, SpvDecorationLocation);
                    continue;
                }
                if (!(stage == _VERTEX_SHADER || stage == _FRAGMENT_SHADER)) continue;
                uint32_t want = stage == _VERTEX_SHADER ? SpvStorageClassInput : SpvStorageClassOutput;
                if (m->vsc[v] != want || is_builtin_var(m, v)) continue;
                if (!src_explicit_location(src, id_name(m, v), stage == _VERTEX_SHADER ? "in" : "out")) {
                    kill_decs(m, v, -1, SpvDecorationLocation);
                    kill_decs(m, v, -1, SpvDecorationIndex);
                }
            }
        }
        assign_interface_locations(sc, ip, src);
        /* initializers are captured again by transform_stage */
        for (int i = 0; i < ip->ninits[stage]; i++) { free(ip->inits[stage][i].name); free(ip->inits[stage][i].vals); }
        ip->ninits[stage] = 0;
        size_t n = 0;
        dump_words(words, nwords, program, stage, "orig");
        out = transform_stage(ctx, &sc[stage], stage, &n, ip, src);
        if (out) {
            dump_words(out, n, program, stage, attempt ? "patched_nopack" : "patched");
            /* the gate exists so a broken patch cannot poison the pipeline MSL — but the
             * geometry stage never becomes pipeline MSL (the geometry domain rewrites it
             * into a compute kernel), and SPIRV-Cross's MSL backend rejects geometry
             * modules that read gl_PrimitiveIDIn ("PrimitiveId is not supported in this
             * execution model"), which would otherwise drop the whole patched module —
             * including the default-uniform block rewrite — and leave the stage's plain
             * uniforms unbound (KHR-GL46.shader_storage_buffer_object basic-atomic-case1) */
            if (stage != _GEOMETRY_SHADER && !msl_accepts(out, n, stage)) { free(out); out = NULL; }
            else {
                *out_words = n;
                /* leaf kinds survive to the post-link reflection (StageCtx is freed below) */
                int total = 0;
                for (uint32_t k = 0; k < sc[stage].ndub; k++) total += sc[stage].dub[k].nleaf;
                free(ip->dub_leaf[stage]);
                ip->dub_leaf[stage] = malloc((size_t)(total ? total : 1));
                ip->dub_leaf_n[stage] = total;
                int o = 0;
                
                for (uint32_t k = 0; k < sc[stage].ndub; k++) {

                    memcpy(ip->dub_leaf[stage] + o, sc[stage].dub[k].leaf, (size_t)sc[stage].dub[k].nleaf);
                    o += sc[stage].dub[k].nleaf;
                }
                
            }
        }
        for (uint32_t k = 0; k < sc[stage].ndub; k++) free(sc[stage].dub[k].leaf);
        mod_free(sc[stage].m);
        free(sc[stage].dub); free(sc[stage].locs);
    }
    return out;
}

bool mgli_process_link(GLMContext ctx, Program *ptr, MGLIProgram *ip)
{
    StageCtx sc[MGLI_STAGES];
    ip->attrib_conflict = 0;   /* re-detected below per link */
    memset(sc, 0, sizeof sc);
    ip->stage_mask = 0;
    ip->first_stage = ip->last_stage = -1;

    for (int s = 0; s < MGLI_STAGES; s++) {
        if (!ptr->shader_slots[s] || !ptr->spirv[s].ir || !ptr->spirv[s].size) continue;
        if (getenv("MGL_FLAT_DEBUG"))
            fprintf(stderr, "MGL proc: prog=%u stage %d words=%zu first_word=%08x\n",
                    ptr->name, s, ptr->spirv[s].size,
                    ptr->spirv[s].size ? ptr->spirv[s].ir[5] : 0);
        sc[s].r = mod_parse(ptr->spirv[s].ir, ptr->spirv[s].size);
        if (!sc[s].r) { fprintf(stderr, "MGL introspect: stage %d: cannot parse SPIR-V\n", s); continue; }
        sc[s].present = 1;
        ip->stage_mask |= 1u << s;
        if (s != _COMPUTE_SHADER) { if (ip->first_stage < 0) ip->first_stage = s; ip->last_stage = s; }
    }
    if (ip->stage_mask & (1u << _COMPUTE_SHADER)) { ip->first_stage = ip->last_stage = _COMPUTE_SHADER; }
    if (!ip->stage_mask) return false;

    for (int s = 0; s < MGLI_STAGES; s++) {
        if (!sc[s].present) continue;
        SMod *r = sc[s].r;
        /* the default uniform block built by mgl_introspect_patch_stage */
        for (uint32_t v = 1; v <= r->bound && !sc[s].dub_var; v++)
            if (r->isvar[v] && r->vglobal[v] && r->vsc[v] == SpvStorageClassUniform && !strcmp(id_name(r, v), "mgl_DefaultUniforms"))
                sc[s].dub_var = v;
        if (sc[s].dub_var) {
            uint32_t st = pointee(r, r->vtype[sc[s].dub_var]);
            SType *ty = &r->t[st];
            sc[s].ndub = ty->nmem;
            sc[s].dub = zalloc((ty->nmem + 1) * sizeof(DubMember));
            for (uint32_t k = 0; k < ty->nmem; k++) {
                DubMember *d = &sc[s].dub[k];
                d->type = ty->mem[k]; d->member = k; d->gtype = ty->mem[k];
                { bool ok_ = get_dec(r, st, (int32_t)k, SpvDecorationOffset, &d->offset);
                  if (getenv("MGL_FP64_DUMP")) { FILE*d_ = fopen("/tmp/fp64off.txt","a"); if (d_) { fprintf(d_, "mem %d name=%s off=%u found=%d\n", k, member_name(r, st, k), d->offset, ok_); fclose(d_); } } }
                uint32_t l; d->explicit_loc = get_dec(r, st, (int32_t)k, SpvDecorationLocation, &l) ? (int)l : -1;
                snprintf(d->name, sizeof d->name, "%s", member_name(r, st, k));
            }
            /* attach the GL-level leaf kinds recorded at patch time (bool/double) */
            {
                int o = 0;
                for (uint32_t k = 0; k < ty->nmem; k++) {
                    uint8_t *cnt = NULL; int nc = 0;
                    record_leaf_kinds(r, sc[s].dub[k].type, &cnt, &nc);
                    free(cnt);sc[s].dub[k].nleaf = nc;
                    sc[s].dub[k].leaf = (o + nc <= ip->dub_leaf_n[s]) ? ip->dub_leaf[s] + o : NULL;
                    o += nc;
                }
            }
            uint32_t sz, al; msl_layout(r, st, &sz, &al);
            sc[s].dub_size = align_up(sz ? sz : 16, 16);
        }
        compute_active(&sc[s], ptr->spirv[s].ir, ptr->spirv[s].size);

        /* the default block is served from MGLIProgram.dub[s]: move it to the uniform-constant list */
        ip->dub_res_index[s] = -1;
        if (sc[s].dub_var) {
            SpirvResourceList *ub = &ptr->spirv_resources_list[s][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER];
            SpirvResourceList *uc = &ptr->spirv_resources_list[s][SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT];
            for (GLuint k = 0; k < ub->count; k++) {
                if (ub->list[k]._id != sc[s].dub_var) continue;
                SpirvResource res = ub->list[k];
                memmove(&ub->list[k], &ub->list[k + 1], (ub->count - k - 1) * sizeof(SpirvResource));
                ub->count--;
                uc->list = realloc(uc->list, (uc->count + 1) * sizeof(SpirvResource));
                memmove(&uc->list[1], &uc->list[0], uc->count * sizeof(SpirvResource));
                uc->list[0] = res;
                uc->count++;
                break;
            }
            for (GLuint k = 0; k < uc->count; k++) if (uc->list[k]._id == sc[s].dub_var) ip->dub_res_index[s] = (int)k;
            ip->dub_size[s] = sc[s].dub_size;
            Buffer *b = ip->dub[s] ? ip->dub[s] : newBuffer(ctx, GL_UNIFORM_BUFFER, 0);
            initBufferData(ctx, b, sc[s].dub_size, NULL, true);
            b->size = sc[s].dub_size;
            memset((void *)b->data.buffer_data, 0, sc[s].dub_size);
            b->data.dirty_bits |= DIRTY_BUFFER_DATA;
            ip->dub[s] = b;
        }
    }

    /* reflection */
    for (int s = 0; s < MGLI_STAGES; s++) if (sc[s].present && sc[s].r) reflect_stage(ip, &sc[s], s, stage_source(ctx, ptr, s));
    if (getenv("MGL_FP64_DUMP")) { for (int s2 = 0; s2 < MGLI_STAGES; s2++) { if (!sc[s2].dub_var || !ptr->spirv[s2].ir) continue; char path[256]; snprintf(path, sizeof path, "/tmp/fp64glview%d.spv", s2); FILE *f_ = fopen(path, "wb"); if (f_) { fwrite(ptr->spirv[s2].ir, 4, ptr->spirv[s2].size, f_); fclose(f_); } } }
    if (ip->first_stage >= 0 && sc[ip->first_stage].r)
        reflect_io(ip, sc[ip->first_stage].r, ip->first_stage, sc[ip->first_stage].active, true, &ip->inputs, &ip->ninputs,
                   stage_source(ctx, ptr, ip->first_stage));
    if (ip->last_stage >= 0 && ip->last_stage != _COMPUTE_SHADER && sc[ip->last_stage].r)
        reflect_io(ip, sc[ip->last_stage].r, ip->last_stage, sc[ip->last_stage].active, false, &ip->outputs, &ip->noutputs,
                   stage_source(ctx, ptr, ip->last_stage));
    for (int s = _GEOMETRY_SHADER; s >= _VERTEX_SHADER; s--)
        if (sc[s].present && sc[s].r) { reflect_io(ip, sc[s].r, s, sc[s].active, false, &ip->vtx_outputs, &ip->nvtx_outputs, stage_source(ctx, ptr, s)); break; }

    /* GL 4.6 11.1.1: the per-stage input/output component budgets.  glslang does
     * not check the consumed locations against them, so a program whose active
     * outputs (or the consumer stage's inputs) exceed MAX_*_(IN|OUT)PUT_COMPONENTS
     * must fail to link here (KHR-Single enhanced_layouts.varying_location_limit_*).
     * TCS outputs / TES inputs are arrays over the patch, budgeted PER ELEMENT. */
    {
        static const char *snames[MGLI_STAGES] = { "vertex", "tessellation control", "tessellation evaluation",
                                                   "geometry", "fragment", "compute" };
        for (int s = 0; s < MGLI_STAGES && !ip->io_components_overflow; s++) {
            if (!sc[s].present || !sc[s].r || s == _COMPUTE_SHADER) continue;
            for (int dir = 0; dir < 2; dir++) {
                MGLIVar *list = NULL; int nlist = 0;
                reflect_io(ip, sc[s].r, s, sc[s].active, dir == 0, &list, &nlist, stage_source(ctx, ptr, s));
                GLint lim = 0;
                if (dir == 0) {  /* inputs */
                    if (s == _VERTEX_SHADER) {   /* MAX_VERTEX_ATTRIBS: attrib_overflow above */
                        for (int i = 0; i < nlist; i++) free(list[i].name);
                        free(list); continue;
                    }
                    lim = 128;   /* GL_MAX_TESS_(CONTROL|EVALUATION)_INPUT_COMPONENTS as get.c reports */
                    if (s == _GEOMETRY_SHADER) lim = STATE_VAR(max_geometry_input_components) ? STATE_VAR(max_geometry_input_components) : 128;
                    else if (s == _FRAGMENT_SHADER) lim = STATE_VAR(max_fragment_input_components) ? STATE_VAR(max_fragment_input_components) : 128;
                } else {
                    lim = 128;   /* MAX_TESS_(CONTROL|EVALUATION)_OUTPUT_COMPONENTS: 128 */
                    if (s == _VERTEX_SHADER) lim = STATE_VAR(max_vertex_output_components) ? STATE_VAR(max_vertex_output_components) : 128;
                    else if (s == _GEOMETRY_SHADER) lim = STATE_VAR(max_geometry_output_components) ? STATE_VAR(max_geometry_output_components) : 128;
                }
                GLuint comps = 0;
                if (getenv("MGL_IO_BUDGET_DEBUG"))
                    for (int i = 0; i < nlist; i++)
                        fprintf(stderr, "io-budget stage=%d dir=%d var='%s' loc=%d size=%d type=0x%x patch=%d\n",
                                s, dir, list[i].name, (int)list[i].location, (int)list[i].size,
                                list[i].type, list[i].is_per_patch);
                for (int i = 0; i < nlist; i++) {
                    MGLIVar *v = &list[i];
                    if (v->location < 0) continue;      /* built-ins (gl_Position ...) are not budgeted */
                    GLint cols = 1, rows = 1; bool dbl = false;
                    if (!io_gl_shape(v->type, &cols, &rows, &dbl)) { comps += 4; continue; }
                    GLuint slots = (GLuint)cols * ((rows >= 3 && dbl) ? 2 : 1);
                    GLuint cnt = (GLuint)(v->size > 0 ? v->size : 1);
                    if (((s == _TESS_CONTROL_SHADER) ||
                         (s == _TESS_EVALUATION_SHADER && dir == 0)) && !v->is_per_patch) {
                        /* per-vertex patch interface: budgeted PER ELEMENT.  The TES
                         * input arrays come EXPANDED (one entry per patch vertex,
                         * element 0 at the array's base location) — count only the
                         * element-0 entry; TCS-side arrays stay one entry with the
                         * full size, so clamp it to a single element. */
                        const char *b = strchr(v->name, '[');
                        if (b && !(b[1] == '0' && b[2] == ']')) continue;
                        cnt = 1;
                    }
                    comps += slots * cnt * 4;
                    /* GL 4.6 11.1.1: the location space itself is capped too — an
                     * explicit location reaching past the budget's vectors is a link
                     * error even when only a few components are actually used */
                    if ((GLuint)(v->location + 1) * 4 > (GLuint)lim ||
                        (GLuint)(v->location + slots * cnt) * 4 > (GLuint)lim) {
                        comps = (GLuint)lim + 1;        /* force the error below */
                        snprintf(ip->block_overlap_msg, sizeof ip->block_overlap_msg,
                                 "%s '%s' of the %s stage uses location %d, beyond the "
                                 "%d-component (%d-vector) limit",
                                 dir == 0 ? "input" : "output", v->name, snames[s],
                                 (int)v->location, lim, lim / 4);
                    }
                }
                for (int i = 0; i < nlist; i++) free(list[i].name);
                free(list);
                if (comps > (GLuint)lim) {
                    ip->io_components_overflow = 1;
                    snprintf(ip->block_overlap_msg, sizeof ip->block_overlap_msg,
                             "active %s of the %s stage consume %u components, more than the %d-component limit",
                             dir == 0 ? "inputs" : "outputs", snames[s], comps, lim);
                }
            }
        }
    }

    if (!assign_uniform_locations(ip)) {
        fprintf(stderr, "MGL introspect: program %u: colliding explicit uniform locations\n", ip->prog ? ip->prog->name : 0);
        return false;
    }

    /* uniform storage + initializers + initial opaque bindings */
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->block_index >= 0 || u->opaque_class == 3) continue;
        size_t ss = (size_t)u->size * u->cols * u->rows * (u->kind == MGLI_K_DOUBLE ? 8 : 4);
        u->shadow = zalloc(ss);
        for (int s = 0; s < MGLI_STAGES; s++) {
            if (!u->st[s].present || !sc[s].r) continue;
            if (u->opaque_class) {
                /* default unit: layout(binding) of ANY stage that declares the uniform —
                 * the qualifier applies program-wide (GLSL 4.2 §4.3.2), so a stage
                 * without it must not overwrite the value (last-stage-wins sampled
                 * unit 0 for KHR-GL46.shading_language_420pack.binding_sampler_single).
                 * The maximum explicit binding wins; >= keeps the last stage's
                 * elem_base for the uniform all-stages-declare-it case. */
                uint32_t bmax = 0;
                int base = 0;
                for (int s2 = 0; s2 < MGLI_STAGES; s2++) {
                    if (!u->st[s2].present || !sc[s2].r) continue;
                    uint32_t b = 0;
                    const char *src = stage_source(ctx, ptr, s2);
                    if (get_dec(sc[s2].r, u->st[s2].spv_id, -1, SpvDecorationBinding, &b) &&
                        src && !src_explicit_qual(src, u->name, "uniform", "binding")) b = 0;
                    if (b >= bmax) { bmax = b; base = u->st[s2].elem_base; }
                }
                for (int e = 0; e < u->size; e++) ((uint32_t *)u->shadow)[e] = bmax + (uint32_t)(base + e);
                continue;
            }
            for (int k = 0; k < ip->ninits[s]; k++) {
                size_t ln = strlen(ip->inits[s][k].name);
                if (strcmp(ip->inits[s][k].name, u->name) && !(!strncmp(ip->inits[s][k].name, u->name, ln) && !strcmp(u->name + ln, "[0]"))) continue;
                size_t bytes = (size_t)ip->inits[s][k].nvals * 4;
                if (bytes > ss) bytes = ss;
                if ((u->kind == MGLI_K_DOUBLE) == (ip->inits[s][k].wide != 0)) memcpy(u->shadow, ip->inits[s][k].vals, bytes);
            }
        }
        mgli_upload_uniform(ctx, ip, u);
    }
    mgli_free_inits(ip);

    for (int s = 0; s < MGLI_STAGES; s++) {
        mod_free(sc[s].r);
        free(sc[s].dub); free(sc[s].active); free(sc[s].dub_active); free(sc[s].locs);
    }
    return true;
}
