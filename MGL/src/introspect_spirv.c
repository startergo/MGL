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
#include <ctype.h>


#pragma mark - module model

enum { T_NONE = 0, T_VOID, T_BOOL, T_INT, T_FLOAT, T_VECTOR, T_MATRIX, T_ARRAY, T_RARRAY, T_STRUCT,
       T_POINTER, T_IMAGE, T_SAMPLER, T_SAMPLED_IMAGE, T_FUNCTION, T_OTHER };

typedef struct { uint32_t *w; uint32_t len; uint8_t dead; uint8_t owned; } SInst;

typedef struct {
    int kind;
    uint32_t width, sign, elem, count, storage;
    uint32_t dim, depth, arrayed, ms, sampled;
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
                if (ID_OK(w[1])) { m->t[w[1]].kind = T_ARRAY; m->t[w[1]].elem = w[2];
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
    uint32_t type;       /* pointee type */
    uint32_t member;     /* member index in the default block */
    uint32_t offset;
    int      explicit_loc;
    char     name[256];
} DubMember;

typedef struct {
    int       present;
    SMod     *m;               /* original module (patched in place) */
    SMod     *r;               /* re-parsed patched module (reflection) */
    DubMember *dub; uint32_t ndub;
    uint32_t  dub_size;
    uint32_t  dub_var;         /* id of the default block variable in r */
    uint8_t  *active;          /* [bound+1] of r: active global variables */
    uint8_t  *dub_active;      /* [ndub] members actually accessed */
    /* location assignments to apply: var -> loc / index */
    struct { uint32_t var; uint32_t loc; int index; } *locs; int nlocs;
    MGLIProgram *ip;
    int       no_pack;
} StageCtx;

static void stage_set_loc(StageCtx *s, uint32_t var, uint32_t loc, int index)
{
    s->locs = realloc(s->locs, (s->nlocs + 1) * sizeof(*s->locs));
    s->locs[s->nlocs++] = (typeof(*s->locs)){ var, loc, index };
}

/* location bitmap helpers */
typedef struct { uint8_t used[1024]; } LocMap;
static bool loc_free(const LocMap *lm, uint32_t base, uint32_t n)
{
    if (base + n > 1024) return false;
    for (uint32_t i = 0; i < n; i++) if (lm->used[base + i]) return false;
    return true;
}
static void loc_take(LocMap *lm, uint32_t base, uint32_t n) { for (uint32_t i = 0; i < n && base + i < 1024; i++) lm->used[base + i] = 1; }
static uint32_t loc_alloc(LocMap *lm, uint32_t n) { for (uint32_t b = 0; b < 1024; b++) if (loc_free(lm, b, n)) { loc_take(lm, b, n); return b; } return 0; }

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

static void assign_interface_locations(StageCtx *sc, MGLIProgram *ip)
{
    /* vertex inputs */
    if (sc[_VERTEX_SHADER].present) {
        SMod *m = sc[_VERTEX_SHADER].m;
        LocMap lm; memset(&lm, 0, sizeof lm);
        for (uint32_t v = 1; v <= m->bound; v++) {
            if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != SpvStorageClassInput || is_builtin_var(m, v)) continue;
            uint32_t l; if (get_dec(m, v, -1, SpvDecorationLocation, &l)) loc_take(&lm, l, io_loc_count(m, _VERTEX_SHADER, v, true));
        }
        for (int pass = 0; pass < 2; pass++)
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != SpvStorageClassInput || is_builtin_var(m, v)) continue;
                if (get_dec(m, v, -1, SpvDecorationLocation, NULL)) continue;
                uint32_t n = io_loc_count(m, _VERTEX_SHADER, v, true);
                int bound = -1;
                for (int b = 0; b < ip->nattrib_binds; b++) if (!strcmp(ip->attrib_binds[b].name, id_name(m, v))) bound = (int)ip->attrib_binds[b].index;
                if (pass == 0 && bound >= 0) { loc_take(&lm, (uint32_t)bound, n); stage_set_loc(&sc[_VERTEX_SHADER], v, (uint32_t)bound, -1); }
                if (pass == 1 && bound < 0) stage_set_loc(&sc[_VERTEX_SHADER], v, loc_alloc(&lm, n), -1);
            }
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

static uint32_t find_or_add_type_int(SMod *m, Inserts *ins, uint32_t *next_id)
{
    for (uint32_t i = 1; i <= m->bound; i++) if (m->t[i].kind == T_INT && m->t[i].width == 32 && m->t[i].sign == 1) return i;
    uint32_t id = (*next_id)++;
    uint32_t o[3] = { id, 32, 1 };
    ins_op(&ins->globals, SpvOpTypeInt, 3, o);
    return id;
}

typedef struct { uint32_t key, id; } KV;

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
static uint32_t *transform_stage(GLMContext ctx, StageCtx *sc, int stage, size_t *out_n)
{
    SMod *m = sc->m;
    Inserts ins; memset(&ins, 0, sizeof ins);
    uint32_t next_id = m->bound;
    uint8_t *laid = zalloc(m->bound + 1);

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
    if (pack) {
        sc->dub = zalloc(sc->ndub * sizeof(DubMember));
        uint32_t k = 0;
        for (uint32_t v = 1; v <= m->bound; v++)
            if (m->isvar[v] && m->vglobal[v] && m->vsc[v] == SpvStorageClassUniformConstant && !is_opaque(m, pointee(m, m->vtype[v]))) {
                DubMember *d = &sc->dub[k];
                d->var = v; d->type = pointee(m, m->vtype[v]); d->member = k;
                uint32_t l; d->explicit_loc = get_dec(m, v, -1, SpvDecorationLocation, &l) ? (int)l : -1;
                snprintf(d->name, sizeof d->name, "%s", id_name(m, v));
                old_of[v] = k + 1;
                k++;
            }
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
            free(sc->dub); sc->dub = NULL; sc->ndub = 0;
            memset(old_of, 0, (m->bound + 1) * 4);
        }
    }

    uint32_t dub_struct = 0, dub_ptr = 0, dub_var = 0, int_t = 0;
    KV *consts = NULL; int nconsts = 0;
    KV *uptr = NULL; int nuptr = 0;
    if (pack) {
        int_t = find_or_add_type_int(m, &ins, &next_id);
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
            ensure_layout(m, &ins, sc->dub[k].type, laid);
            uint32_t s, a; msl_layout(m, sc->dub[k].type, &s, &a);
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
            for (uint32_t k = 0; k < sc->ndub; k++) o[k + 1] = sc->dub[k].type;
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
            uint32_t base = strip_arrays(m, sc->dub[k].type);
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
                uint32_t pt = UPTR(sc->dub[mem].type);
                uint32_t nid = next_id++;
                uint32_t ac[5] = { (5u << 16) | SpvOpAccessChain, pt, nid, dub_var, kv_get(consts, nconsts, mem) };
                wb_inst(&body, ac, 5);
                wb_push(&body, w[0]); wb_push(&body, w[1]); wb_push(&body, w[2]); wb_push(&body, nid);
                for (uint32_t a = 4; a < wc; a++) wb_push(&body, w[a]);
                continue;
            }
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && w[3] <= m->bound && (old_of[w[3]] || retyped[w[3]])) {
                uint32_t pt = UPTR(pointee(m, w[1]));
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
        if (m->in[i].dead || moved[i]) continue;
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

/* walk a default-block / opaque uniform variable and create GL uniforms for its leaves */
static void walk_default(MGLIProgram *ip, const SMod *m, int stage, const char *name, uint32_t t, uint32_t offset,
                         int explicit_loc, int *loc_cursor, uint32_t spv_id, int res_type, int *elem_cursor, int hidden_prefix)
{
    const SType *ty = &m->t[t];
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
                u->opaque_class = is_opaque(m, et) ? (m->t[strip_arrays(m, et)].kind == T_IMAGE && m->t[strip_arrays(m, et)].sampled == 2 ? 2 : 1) : 0;
                if (u->opaque_class) u->kind = MGLI_K_INT;
                u->hidden = hidden_prefix;
                u->location = explicit_loc >= 0 ? explicit_loc + *loc_cursor : -2 - *loc_cursor; /* resolved later */
                *loc_cursor += u->size;
            }
            u->stages |= 1u << stage;
            u->st[stage].present = 1;
            u->st[stage].offset = offset; u->st[stage].array_stride = stride;
            uint32_t b = strip_arrays(m, et);
            if (m->t[b].kind == T_MATRIX) { uint32_t cs, ca; msl_layout(m, m->t[b].elem, &cs, &ca); u->st[stage].matrix_stride = align_up(cs, ca); }
            u->st[stage].scalar_size = u->opaque_class ? 4 : (u->kind == MGLI_K_BOOL ? 1 : (u->kind == MGLI_K_DOUBLE ? 8 : 4));
            u->st[stage].spv_id = spv_id; u->st[stage].res_type = res_type; u->st[stage].elem_base = *elem_cursor;
            *elem_cursor += (int)(ty->count ? ty->count : 1);
            return;
        }
        for (uint32_t i = 0; i < (ty->count ? ty->count : 1); i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s[%u]", name, i);
            walk_default(ip, m, stage, nm, et, offset + i * stride, explicit_loc, loc_cursor, spv_id, res_type, elem_cursor, hidden_prefix);
        }
        return;
    }
    if (ty->kind == T_STRUCT) {
        for (uint32_t i = 0; i < ty->nmem; i++) {
            char nm[512]; snprintf(nm, sizeof nm, "%s.%s", name, member_name(m, t, i));
            uint32_t mo = 0; get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo);
            walk_default(ip, m, stage, nm, ty->mem[i], offset + mo, explicit_loc, loc_cursor, spv_id, res_type, elem_cursor, hidden_prefix);
        }
        return;
    }
    int created, ui = add_uniform(ip, name, &created);
    MGLIUniform *u = &ip->uniforms[ui];
    if (created) {
        u->type = gl_type_of(m, t); u->size = 1;
        scalar_info(m, t, &u->kind, &u->cols, &u->rows);
        u->opaque_class = is_opaque(m, t) ? (m->t[t].kind == T_IMAGE && m->t[t].sampled == 2 ? 2 : 1) : 0;
        if (u->opaque_class) u->kind = MGLI_K_INT;
        u->hidden = hidden_prefix;
        u->location = explicit_loc >= 0 ? explicit_loc + *loc_cursor : -2 - *loc_cursor;
        *loc_cursor += 1;
    }
    u->stages |= 1u << stage;
    u->st[stage].present = 1;
    u->st[stage].offset = offset; u->st[stage].array_stride = 0;
    if (ty->kind == T_MATRIX) { uint32_t cs, ca; msl_layout(m, ty->elem, &cs, &ca); u->st[stage].matrix_stride = align_up(cs, ca); }
    u->st[stage].scalar_size = u->opaque_class ? 4 : (u->kind == MGLI_K_BOOL ? 1 : (u->kind == MGLI_K_DOUBLE ? 8 : 4));
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
} BlockWalk;

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
                    b->offset = (GLint)offset; b->array_stride = (GLint)stride;
                    b->matrix_stride = m->t[et].kind == T_MATRIX ? (GLint)mstride : 0; b->is_row_major = m->t[et].kind == T_MATRIX ? row_major : 0;
                    b->top_size = depth == 0 ? 1 : bw->top_size; b->top_stride = depth == 0 ? 0 : bw->top_stride;
                    found = ip->nbvars++;
                    MGLIBlock *blk = &ip->ssbos[bw->block];
                    blk->members = realloc(blk->members, (blk->member_count + 1) * sizeof(int));
                    blk->members[blk->member_count++] = found;
                }
                ip->bvars[found].stages |= 1u << bw->stage;
            } else {
                int created, ui = add_uniform(bw->ip, nm, &created);
                MGLIUniform *u = &bw->ip->uniforms[ui];
                if (created) {
                    u->type = gl_type_of(m, et); u->size = (GLint)(ty->count ? ty->count : 1); u->block_index = bw->block;
                    u->offset = (GLint)offset; u->array_stride = (GLint)stride;
                    u->matrix_stride = m->t[et].kind == T_MATRIX ? (GLint)mstride : 0; u->is_row_major = m->t[et].kind == T_MATRIX ? row_major : 0;
                    scalar_info(m, et, &u->kind, &u->cols, &u->rows);
                }
                add_member(&bw->ip->ubos[bw->block], ui);
                u->stages |= 1u << bw->stage;
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
        for (uint32_t i = 0; i < ty->nmem; i++) {
            char nm[512];
            if (name[0]) snprintf(nm, sizeof nm, "%s.%s", name, member_name(m, t, i));
            else snprintf(nm, sizeof nm, "%s", member_name(m, t, i));
            uint32_t mo = 0, ms = 0; get_dec(m, t, (int32_t)i, SpvDecorationOffset, &mo);
            get_dec(m, t, (int32_t)i, SpvDecorationMatrixStride, &ms);
            int rm = get_dec(m, t, (int32_t)i, SpvDecorationRowMajor, NULL);
            if (depth == 0) { bw->top_size = 1; bw->top_stride = 0; }
            walk_block(bw, nm, ty->mem[i], offset + mo, ms, rm, depth == 0 ? 0 : depth + 1);
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
            b->offset = (GLint)offset; b->array_stride = 0;
            b->matrix_stride = ty->kind == T_MATRIX ? (GLint)mstride : 0; b->is_row_major = ty->kind == T_MATRIX ? row_major : 0;
            b->top_size = bw->top_size; b->top_stride = bw->top_stride;
            found = ip->nbvars++;
            MGLIBlock *blk = &ip->ssbos[bw->block];
            blk->members = realloc(blk->members, (blk->member_count + 1) * sizeof(int));
            blk->members[blk->member_count++] = found;
        }
        ip->bvars[found].stages |= 1u << bw->stage;
        return;
    }
    int created, ui = add_uniform(bw->ip, name, &created);
    MGLIUniform *u = &bw->ip->uniforms[ui];
    if (created) {
        u->type = gl_type_of(m, t); u->size = 1; u->block_index = bw->block; u->offset = (GLint)offset; u->array_stride = 0;
        u->matrix_stride = ty->kind == T_MATRIX ? (GLint)mstride : 0; u->is_row_major = ty->kind == T_MATRIX ? row_major : 0;
        scalar_info(m, t, &u->kind, &u->cols, &u->rows);
    }
    add_member(&bw->ip->ubos[bw->block], ui);
    u->stages |= 1u << bw->stage;
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

static void add_io(MGLIVar **arr, int *n, const SMod *m, int stage, const char *name, uint32_t t, GLint loc, GLint index, GLint comp, GLint patch)
{
    for (int i = 0; i < *n; i++) if (!strcmp((*arr)[i].name, name)) { (*arr)[i].stages |= 1u << stage; return; }
    *arr = realloc(*arr, (*n + 1) * sizeof(MGLIVar));
    MGLIVar *v = &(*arr)[*n]; memset(v, 0, sizeof *v);
    v->name = xstrdup(name);
    v->type = gl_type_of(m, t);
    v->size = (m->t[t].kind == T_ARRAY) ? (GLint)m->t[t].count : 1;
    v->location = loc; v->index = index; v->component = comp; v->is_per_patch = patch;
    v->stages = 1u << stage;
    (*n)++;
}

static void reflect_io(MGLIProgram *ip, const SMod *m, int stage, const uint8_t *active, bool input, MGLIVar **arr, int *narr)
{
    uint32_t sc_want = input ? SpvStorageClassInput : SpvStorageClassOutput;
    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != sc_want || !active[v]) continue;
        uint32_t t = pointee(m, m->vtype[v]);
        bool patch = get_dec(m, v, -1, SpvDecorationPatch, NULL);
        if (stage_is_arrayed_io(stage, input) && !patch && (m->t[t].kind == T_ARRAY || m->t[t].kind == T_RARRAY)) t = m->t[t].elem;
        uint32_t st = strip_arrays(m, t);
        bool is_block = m->t[st].kind == T_STRUCT && get_dec(m, st, -1, SpvDecorationBlock, NULL);
        uint32_t bi;
        if (get_dec(m, v, -1, SpvDecorationBuiltIn, &bi)) {
            add_io(arr, narr, m, stage, id_name(m, v), t, -1, -1, 0, patch);
            continue;
        }
        if (is_block) {
            /* built-in blocks (gl_PerVertex) are not enumerated; user blocks list "Block.member" */
            bool builtin = false;
            for (uint32_t i = 0; i < m->t[st].nmem; i++) if (get_dec(m, st, (int32_t)i, SpvDecorationBuiltIn, NULL)) builtin = true;
            if (builtin) continue;
            uint32_t loc = 0; get_dec(m, v, -1, SpvDecorationLocation, &loc);
            for (uint32_t i = 0; i < m->t[st].nmem; i++) {
                char nm[512]; snprintf(nm, sizeof nm, "%s.%s", id_name(m, st), member_name(m, st, i));
                uint32_t ml; if (get_dec(m, st, (int32_t)i, SpvDecorationLocation, &ml)) loc = ml;
                uint32_t mt = m->t[st].mem[i];
                if (m->t[mt].kind == T_ARRAY) { char n2[520]; snprintf(n2, sizeof n2, "%s[0]", nm); add_io(arr, narr, m, stage, n2, mt, (GLint)loc, 0, 0, patch); }
                else add_io(arr, narr, m, stage, nm, mt, (GLint)loc, 0, 0, patch);
                loc += loc_count(m, mt);
            }
            continue;
        }
        uint32_t loc = 0, idx = 0, comp = 0;
        get_dec(m, v, -1, SpvDecorationLocation, &loc);
        get_dec(m, v, -1, SpvDecorationIndex, &idx);
        get_dec(m, v, -1, SpvDecorationComponent, &comp);
        if (m->t[t].kind == T_STRUCT) {
            uint32_t l = loc;
            for (uint32_t i = 0; i < m->t[t].nmem; i++) {
                char nm[512]; snprintf(nm, sizeof nm, "%s.%s", id_name(m, v), member_name(m, t, i));
                add_io(arr, narr, m, stage, nm, m->t[t].mem[i], (GLint)l, 0, 0, patch);
                l += loc_count(m, m->t[t].mem[i]);
            }
            continue;
        }
        char nm[512];
        if (m->t[t].kind == T_ARRAY) snprintf(nm, sizeof nm, "%s[0]", id_name(m, v));
        else snprintf(nm, sizeof nm, "%s", id_name(m, v));
        add_io(arr, narr, m, stage, nm, t,
               (GLint)loc, (stage == _FRAGMENT_SHADER && !input) ? (GLint)idx : 0, (GLint)comp, patch);
    }
}

/* active global variables and accessed default-block members via SPIRV-Cross */
static void compute_active(StageCtx *s, const uint32_t *words, size_t n)
{
    SMod *r = s->r;
    s->active = zalloc(r->bound + 1);
    s->dub_active = zalloc(s->ndub + 1);
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
            /* built-in inputs/outputs are not in resource lists: mark used ones from the set */
            for (uint32_t v = 1; v <= r->bound; v++)
                if (r->isvar[v] && r->vglobal[v] && (r->vsc[v] == SpvStorageClassInput || r->vsc[v] == SpvStorageClassOutput) &&
                    get_dec(r, v, -1, SpvDecorationBuiltIn, NULL)) {
                    /* conservative: referenced anywhere in function bodies */
                    for (size_t k = r->first_func; k < r->n && !s->active[v]; k++) {
                        uint32_t op = r->in[k].w[0] & 0xffff;
                        if (op == SpvOpLoad && r->in[k].w[3] == v) s->active[v] = 1;
                        if (op == SpvOpStore && r->in[k].w[1] == v) s->active[v] = 1;
                        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && r->in[k].w[3] == v) s->active[v] = 1;
                    }
                }
        }
        if (s->dub_var) {
            const spvc_buffer_range *ranges; size_t nr;
            if (spvc_compiler_get_active_buffer_ranges(comp, s->dub_var, &ranges, &nr) == SPVC_SUCCESS)
                for (size_t i = 0; i < nr; i++) if (ranges[i].index < s->ndub) s->dub_active[ranges[i].index] = 1;
        }
    }
    if (!ok) {
        for (uint32_t v = 1; v <= r->bound; v++) if (r->isvar[v] && r->vglobal[v]) s->active[v] = 1;
        for (uint32_t k = 0; k < s->ndub; k++) s->dub_active[k] = 1;
    }
    if (c) spvc_context_destroy(c);
}

static bool name_is_hidden(const char *n) { return !strncmp(n, "_mgl_sr_", 8); }

static void reflect_stage(MGLIProgram *ip, StageCtx *s, int stage, int *loc_cursors_dummy)
{
    const SMod *m = s->r;
    (void)loc_cursors_dummy;
    /* default block members */
    for (uint32_t k = 0; k < s->ndub; k++) {
        if (!s->dub_active[k]) continue;
        DubMember *d = &s->dub[k];
        uint32_t mt = m->t[strip_arrays(m, pointee(m, m->vtype[s->dub_var]))].mem[k];
        int cursor = 0, elem = 0;
        walk_default(ip, m, stage, d->name, mt, d->offset, d->explicit_loc, &cursor, 0, -1, &elem, name_is_hidden(d->name));
    }
    /* opaque uniforms + UBO + SSBO + atomic counters */
    for (uint32_t v = 1; v <= m->bound; v++) {
        if (!m->isvar[v] || !m->vglobal[v] || !s->active[v]) continue;
        uint32_t t = pointee(m, m->vtype[v]);
        uint32_t st = strip_arrays(m, t);
        if (m->vsc[v] == SpvStorageClassUniformConstant) {
            if (!is_opaque(m, t)) continue; /* not packed (fallback): handled as per-uniform buffers */
            uint32_t l; int el = get_dec(m, v, -1, SpvDecorationLocation, &l) ? (int)l : -1;
            int cursor = 0, elem = 0;
            int rt = m->t[st].kind == T_IMAGE && m->t[st].sampled == 2 ? SPVC_RESOURCE_TYPE_STORAGE_IMAGE :
                     (m->t[st].kind == T_IMAGE ? SPVC_RESOURCE_TYPE_SEPARATE_IMAGE : SPVC_RESOURCE_TYPE_SAMPLED_IMAGE);
            walk_default(ip, m, stage, id_name(m, v), t, 0, el, &cursor, v, rt, &elem, 0);
            continue;
        }
        if (m->vsc[v] == SpvStorageClassAtomicCounter) {
            uint32_t binding = 0, off = 0;
            get_dec(m, v, -1, SpvDecorationBinding, &binding);
            get_dec(m, v, -1, SpvDecorationOffset, &off);
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
        uint32_t nel = m->t[t].kind == T_ARRAY ? m->t[t].count : 0;
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
            b->stages |= 1u << stage;
            b->spv_id[stage] = v;
            BlockWalk bw = { ip, m, stage, bi, ssbo, 0, 1, 0 };
            {
                char prefix[512];
                if (iname[0]) snprintf(prefix, sizeof prefix, "%s", bname); else prefix[0] = 0;
                walk_block(&bw, prefix, st, 0, 0, 0, 0);
            }
        }
    }
}

/* assign locations to default-block / opaque uniforms (explicit first, then by order) */
static void assign_uniform_locations(MGLIProgram *ip)
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
    ip->locs = zalloc((size_t)(n ? n : 1) * sizeof(*ip->locs));
    ip->nlocs = n;
    for (int l = 0; l < n; l++) ip->locs[l].uniform = -1;
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->location < 0) continue;
        for (int k = 0; k < u->size; k++) { ip->locs[u->location + k].uniform = i; ip->locs[u->location + k].element = k; }
    }
}


#pragma mark - link entry

/* compile the patched module once with SPIRV-Cross MSL; false (and a log line) if it is rejected */
static bool msl_accepts(const uint32_t *words, size_t n, int stage)
{
    spvc_context c = NULL; spvc_parsed_ir ir = NULL; spvc_compiler comp = NULL; spvc_compiler_options opt = NULL;
    const char *res = NULL;
    bool ok = spvc_context_create(&c) == SPVC_SUCCESS &&
              spvc_context_parse_spirv(c, words, n, &ir) == SPVC_SUCCESS &&
              spvc_context_create_compiler(c, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) == SPVC_SUCCESS &&
              spvc_compiler_create_compiler_options(comp, &opt) == SPVC_SUCCESS &&
              spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3, 1, 0)) == SPVC_SUCCESS &&
              spvc_compiler_install_compiler_options(comp, opt) == SPVC_SUCCESS &&
              spvc_compiler_compile(comp, &res) == SPVC_SUCCESS && res;
    if (!ok) fprintf(stderr, "MGL introspect: stage %d: patched SPIR-V rejected by SPIRV-Cross (%s), using the original module\n",
                     stage, c ? spvc_context_get_last_error_string(c) : "no context");
    if (c) spvc_context_destroy(c);
    return ok;
}

static void dump_words(const uint32_t *w, size_t n, Program *ptr, int stage, const char *tag)
{
    const char *dir = getenv("MGL_INTROSPECT_DUMP");
    if (!dir) return;
    char path[1024]; snprintf(path, sizeof path, "%s/prog%u_stage%d_%s.spv", dir, ptr->name, stage, tag);
    FILE *f = fopen(path, "wb"); if (f) { fwrite(w, 4, n, f); fclose(f); }
}


/* is `name` declared with layout(location) as an `in`/`out` variable in the GLSL source? */
static bool src_explicit_location(const char *src, const char *name, const char *dir)
{
    if (!src) return true;
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
            bool has_dir = false, has_loc = false; char last[256] = "";
            for (char *t = st; *t; ) {
                if (isalnum((unsigned char)*t) || *t == '_') {
                    char *e = t; while (isalnum((unsigned char)*e) || *e == '_') e++;
                    size_t l = (size_t)(e - t);
                    if (l == strlen(dir) && !strncmp(t, dir, l)) has_dir = true;
                    if (l == 8 && !strncmp(t, "location", 8)) has_loc = true;
                    if (l < sizeof last) { memcpy(last, t, l); last[l] = 0; }
                    t = e;
                } else if (*t == '[') { while (*t && *t != ']') t++; }
                else t++;
            }
            if (has_dir && !strcmp(last, name)) { found = true; expl = has_loc; }
            *c = save;
            if (!save) break;
            st = c + 1;
        }
    }
    free(b);
    return found ? expl : true;
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
        /* locations glslang's map_io invented are not "explicit": bindings may override them */
        const char *src = stage_source(ctx, program, stage);
        SMod *m = sc[stage].m;
        if (src && (stage == _VERTEX_SHADER || stage == _FRAGMENT_SHADER)) {
            uint32_t want = stage == _VERTEX_SHADER ? SpvStorageClassInput : SpvStorageClassOutput;
            for (uint32_t v = 1; v <= m->bound; v++) {
                if (!m->isvar[v] || !m->vglobal[v] || m->vsc[v] != want || is_builtin_var(m, v)) continue;
                if (!get_dec(m, v, -1, SpvDecorationLocation, NULL)) continue;
                if (!src_explicit_location(src, id_name(m, v), stage == _VERTEX_SHADER ? "in" : "out")) {
                    kill_decs(m, v, -1, SpvDecorationLocation);
                    kill_decs(m, v, -1, SpvDecorationIndex);
                }
            }
        }
        assign_interface_locations(sc, ip);
        /* initializers are captured again by transform_stage */
        for (int i = 0; i < ip->ninits[stage]; i++) { free(ip->inits[stage][i].name); free(ip->inits[stage][i].vals); }
        ip->ninits[stage] = 0;
        size_t n = 0;
        dump_words(words, nwords, program, stage, "orig");
        out = transform_stage(ctx, &sc[stage], stage, &n);
        if (out) {
            dump_words(out, n, program, stage, attempt ? "patched_nopack" : "patched");
            if (!msl_accepts(out, n, stage)) { free(out); out = NULL; }
            else *out_words = n;
        }
        mod_free(sc[stage].m);
        free(sc[stage].dub); free(sc[stage].locs);
    }
    return out;
}

bool mgli_process_link(GLMContext ctx, Program *ptr, MGLIProgram *ip)
{
    StageCtx sc[MGLI_STAGES];
    memset(sc, 0, sizeof sc);
    ip->stage_mask = 0;
    ip->first_stage = ip->last_stage = -1;

    for (int s = 0; s < MGLI_STAGES; s++) {
        if (!ptr->shader_slots[s] || !ptr->spirv[s].ir || !ptr->spirv[s].size) continue;
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
                d->type = ty->mem[k]; d->member = k;
                get_dec(r, st, (int32_t)k, SpvDecorationOffset, &d->offset);
                uint32_t l; d->explicit_loc = get_dec(r, st, (int32_t)k, SpvDecorationLocation, &l) ? (int)l : -1;
                snprintf(d->name, sizeof d->name, "%s", member_name(r, st, k));
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
    for (int s = 0; s < MGLI_STAGES; s++) if (sc[s].present && sc[s].r) reflect_stage(ip, &sc[s], s, NULL);
    if (ip->first_stage >= 0 && sc[ip->first_stage].r)
        reflect_io(ip, sc[ip->first_stage].r, ip->first_stage, sc[ip->first_stage].active, true, &ip->inputs, &ip->ninputs);
    if (ip->last_stage >= 0 && ip->last_stage != _COMPUTE_SHADER && sc[ip->last_stage].r)
        reflect_io(ip, sc[ip->last_stage].r, ip->last_stage, sc[ip->last_stage].active, false, &ip->outputs, &ip->noutputs);
    for (int s = _GEOMETRY_SHADER; s >= _VERTEX_SHADER; s--)
        if (sc[s].present && sc[s].r) { reflect_io(ip, sc[s].r, s, sc[s].active, false, &ip->vtx_outputs, &ip->nvtx_outputs); break; }
    assign_uniform_locations(ip);

    /* uniform storage + initializers + initial opaque bindings */
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->block_index >= 0 || u->opaque_class == 3) continue;
        size_t ss = (size_t)u->size * u->cols * u->rows * (u->kind == MGLI_K_DOUBLE ? 8 : 4);
        u->shadow = zalloc(ss);
        for (int s = 0; s < MGLI_STAGES; s++) {
            if (!u->st[s].present || !sc[s].r) continue;
            if (u->opaque_class) {
                uint32_t b = 0; /* default unit: layout(binding) */
                if (get_dec(sc[s].r, u->st[s].spv_id, -1, SpvDecorationBinding, &b))
                    for (int e = 0; e < u->size; e++) ((uint32_t *)u->shadow)[e] = b + (uint32_t)(u->st[s].elem_base + e);
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
