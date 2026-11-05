/*
 * mgl_layout_validate.c — GLSL 4.40+ / ARB_enhanced_layouts link-time layout
 * validation on the per-stage SPIR-V that glslang produced.
 *
 * glslang (the GLSL front end MGL uses) implements most of ARB_enhanced_layouts
 * but is missing two families of rules that GLSL 4.60 makes compile-/link-time
 * errors; Mesa enforces them in ast_to_hir.cpp ("layout qualifier offset
 * overlaps previous member", "offset must be a multiple of the base
 * alignment of %s") and in its linker.  GL lets an implementation defer any
 * compile-time error to link time, which is where MGL sees the SPIR-V:
 *
 *   R1  explicit interface-block member offsets (std140/std430) must be a
 *       multiple of the member's base alignment and must not overlap a
 *       previous member  (Mesa: ast_to_hir.cpp "offset overlaps previous
 *       member", "layout qualifier offset must be a multiple of the base
 *       alignment")  — KHR-Single-GL46.enhanced_layouts uniform_block_member_
 *       overlapping_offsets_* / ssb_member_overlapping_offsets_* (the double
 *       matrix cases produce aligned-but-overlapping offsets that glslang
 *       accepts; the single-precision cases are already caught by glslang's
 *       own alignment check).
 *
 *   R2  in/out (varying) locations must fit the stage's component limits —
 *       GL 4.6 11.1.1/16.1: "the amount of storage available for shader
 *       ... variables is limited" (Mesa: linker.cpp check_against_caps) —
 *       KHR-Single-GL46.enhanced_layouts.varying_location_limit_*.  The limits
 *       are the same constants MGL reports through glGet (glm_params.c /
 *       get.c): 128 output/input components per stage (32 locations) and
 *       16 vertex-attribute locations, each of them also applying to the
 *       next stage's inputs as the CTS's getLast*Location does.
 *
 * The validation runs per stage right after SPIR-V generation in program.c's
 * link path; a violation fails the link with a spec-cited message.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "spirv.h"

#include "mgl.h"
#include "glm_context.h"
#include "programs.h"

static const char *lv_stage_label(int stage)
{
    switch (stage)
    {
        case _VERTEX_SHADER:            return "vertex";
        case _FRAGMENT_SHADER:          return "fragment";
        case _GEOMETRY_SHADER:          return "geometry";
        case _TESS_CONTROL_SHADER:      return "tessellation control";
        case _TESS_EVALUATION_SHADER:   return "tessellation evaluation";
        case _COMPUTE_SHADER:           return "compute";
        default: return "shader";
    }
}

/* ------------------------------------------------------------------ tiny SPIR-V reader */

enum { LV_VOID, LV_BOOL, LV_INT, LV_FLOAT, LV_VEC, LV_MAT, LV_ARRAY, LV_RTARRAY, LV_STRUCT, LV_PTR, LV_OTHER };

typedef struct {
    uint32_t id;
    uint8_t  kind;
    uint8_t  bits;      /* scalar bits */
    uint8_t  n;         /* vector length / matrix columns */
    uint8_t  rows;      /* matrix rows (vector length of the column type) */
    uint32_t elem;      /* element type id (vec/mat/array/ptr) */
    uint32_t length;    /* array length */
    uint32_t length_id; /* OpConstant id of the array length */
    uint32_t members;   /* struct member type ids */
    int      n_members;
} LvType;

typedef struct { uint32_t id, deco, value; } LvDeco;
typedef struct { uint32_t id, member, deco, value; } LvMemberDeco;
typedef struct { uint32_t id; char name[128]; } LvName;
typedef struct { uint32_t id, member; char name[64]; } LvMemberName;

typedef struct {
    LvType       types[512];  int n_types;
    uint32_t     membertab[2048]; int n_membertab;
    LvDeco       decos[2048]; int n_decos;
    LvMemberDeco mdecos[4096]; int n_mdecos;
    LvName       names[1024]; int n_names;
    LvMemberName mnames[1024]; int n_mnames;
    struct { uint32_t id, v; } consts[512]; int n_consts;
    struct { uint32_t id, type, storage; } vars[512]; int n_vars;
} LvModule;

static LvType *lv_find(LvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_types; i++)
        if (m->types[i].id == id) return &m->types[i];
    return NULL;
}

static bool lv_deco(LvModule *m, uint32_t id, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == id && m->decos[i].deco == deco)
        {
            if (value) *value = m->decos[i].value;
            return true;
        }
    return false;
}

static bool lv_mdeco(LvModule *m, uint32_t type_id, uint32_t member, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_mdecos; i++)
        if (m->mdecos[i].id == type_id && m->mdecos[i].member == member && m->mdecos[i].deco == deco)
        {
            if (value) *value = m->mdecos[i].value;
            return true;
        }
    return false;
}

static const char *lv_name(LvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_names; i++)
        if (m->names[i].id == id) return m->names[i].name;
    return NULL;
}

static const char *lv_member_name(LvModule *m, uint32_t struct_id, uint32_t member)
{
    for (int i = 0; i < m->n_mnames; i++)
        if (m->mnames[i].id == struct_id && m->mnames[i].member == member)
            return m->mnames[i].name;
    return NULL;
}

static uint32_t lv_const(LvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_consts; i++)
        if (m->consts[i].id == id) return m->consts[i].v;
    return 1;
}

static uint32_t lv_array_length(LvModule *m, LvType *t)
{
    if (!t->length && t->length_id) return lv_const(m, t->length_id);
    return t->length ? t->length : 1;
}

/* parse the types / decorations / variables of a SPIR-V module */
static bool lv_parse(const uint32_t *ir, size_t words, LvModule *m)
{
    if (words < 5 || ir[0] != SpvMagicNumber) return false;

    size_t i = 5;   /* header */
    memset(m, 0, sizeof *m);

    while (i < words)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint16_t op = word & 0xffff;
        if (wc == 0 || i + wc > words) return false;
        const uint32_t *w = &ir[i + 1];

        switch (op)
        {
            case 6:  /* OpMemberName: struct id, member, name */
                if (m->n_mnames < (int)(sizeof m->mnames / sizeof m->mnames[0]))
                {
                    LvMemberName *n = &m->mnames[m->n_mnames++];
                    n->id = w[0];
                    n->member = w[1];
                    const char *p = (const char *)&w[2];
                    size_t o = 0;
                    while (*p && o + 1 < sizeof n->name)
                        n->name[o++] = *p++;
                    n->name[o] = 0;
                }
                break;

            case 5:  /* OpName */
                if (m->n_names < (int)(sizeof m->names / sizeof m->names[0]))
                {
                    LvName *n = &m->names[m->n_names++];
                    n->id = w[0];
                    const char *p = (const char *)&w[1];
                    size_t o = 0;
                    while (*p && o + 1 < sizeof n->name)
                        n->name[o++] = *p++;
                    n->name[o] = 0;
                }
                break;

            case 19: /* OpTypeVoid */
            case 20: /* OpTypeBool */
            case 21: /* OpTypeInt */
            case 22: /* OpTypeFloat */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t;
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = op == 19 ? LV_VOID : op == 20 ? LV_BOOL : op == 21 ? LV_INT : LV_FLOAT;
                    t.bits = (op == 19 || op == 20) ? 32 : w[1];   /* OpTypeInt/Float: width */
                    m->types[m->n_types++] = t;
                }
                break;
            }
            case 23: /* OpTypeVector */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t, *e = lv_find(m, w[1]);
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = LV_VEC;
                    t.elem = w[1];
                    t.rows = t.n = w[2];
                    t.bits = e ? e->bits : 32;
                    m->types[m->n_types++] = t;
                }
                break;
            }
            case 24: /* OpTypeMatrix */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t, *e = lv_find(m, w[1]);
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = LV_MAT;
                    t.elem = w[1];
                    t.n = w[2];               /* columns */
                    t.rows = e ? e->n : 1;    /* rows = column vector length */
                    t.bits = e ? e->bits : 32;
                    m->types[m->n_types++] = t;
                }
                break;
            }
            case 28: /* OpTypeArray */
            case 29: /* OpTypeRuntimeArray */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t;
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = op == 28 ? LV_ARRAY : LV_RTARRAY;
                    t.elem = w[1];
                    if (op == 28) t.length_id = w[2];
                    m->types[m->n_types++] = t;
                }
                break;
            }
            case 30: /* OpTypeStruct */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t;
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = LV_STRUCT;
                    t.members = w[1];   /* first member id in a growing member table */
                    m->types[m->n_types++] = t;
                }
                /* member type ids are recorded below (see the second pass) */
                break;
            }
            case 32: /* OpTypePointer */
            {
                if (m->n_types < (int)(sizeof m->types / sizeof m->types[0]))
                {
                    LvType t;
                    memset(&t, 0, sizeof t);
                    t.id = w[0];
                    t.kind = LV_PTR;
                    t.elem = w[2];   /* result, storage, type */
                    m->types[m->n_types++] = t;
                }
                break;
            }
            case 43: /* OpConstant: result type, result id, value */
                if (m->n_consts < (int)(sizeof m->consts / sizeof m->consts[0]))
                {
                    m->consts[m->n_consts].id = w[1];
                    m->consts[m->n_consts++].v = w[2];
                }
                break;
            case 71: /* OpDecorate */
                if (m->n_decos < (int)(sizeof m->decos / sizeof m->decos[0]))
                {
                    m->decos[m->n_decos].id = w[0];
                    m->decos[m->n_decos].deco = w[1];
                    m->decos[m->n_decos].value = wc > 3 ? w[2] : 0;
                    m->n_decos++;
                }
                break;
            case 72: /* OpMemberDecorate */
                if (m->n_mdecos < (int)(sizeof m->mdecos / sizeof m->mdecos[0]))
                {
                    m->mdecos[m->n_mdecos].id = w[0];
                    m->mdecos[m->n_mdecos].member = w[1];
                    m->mdecos[m->n_mdecos].deco = w[2];
                    m->mdecos[m->n_mdecos].value = wc > 4 ? w[3] : 0;
                    m->n_mdecos++;
                }
                break;
            case 59: /* OpVariable: id, ptr-type, storage */
                if (m->n_vars < (int)(sizeof m->vars / sizeof m->vars[0]))
                {
                    m->vars[m->n_vars].id = w[1];
                    m->vars[m->n_vars].type = w[0];
                    m->vars[m->n_vars].storage = w[2];
                    m->n_vars++;
                }
                break;
        }
        i += wc;
    }

    /* second pass: struct member type ids.  OpTypeStruct's operands are its
     * member type ids — record them in a side table keyed by the struct's
     * result id. */
    i = 5;
    while (i < words)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint16_t op = word & 0xffff;
        const uint32_t *w = &ir[i + 1];
        if (wc == 0 || i + wc > words) return false;
        if (op == 30)
        {
            LvType *t = lv_find(m, w[0]);
            if (t && t->kind == LV_STRUCT)
            {
                t->n_members = (int)wc - 2;
                /* members[n] holds the n-th member's type id: reuse .elem + a
                 * parallel member table below */
                t->members = m->n_membertab;
                for (uint32_t k = 0; k + 2 < wc; k++)
                {
                    if (m->n_membertab < (int)(sizeof m->membertab / sizeof m->membertab[0]))
                        m->membertab[m->n_membertab++] = w[1 + k];
                }
            }
        }
        i += wc;
    }
    return true;
}

/* ------------------------------------------------ std140 / std430 sizing (mirrors Mesa) */

static unsigned lv_round_up(unsigned x, unsigned a) { return a ? ((x + a - 1) / a) * a : x; }

static unsigned lv_member_align(LvModule *m, uint32_t type_id, bool std430, bool row_major);

/* base alignment of the element vector of a matrix (rules 1-3) */
static unsigned lv_vec_align(LvModule *m, uint32_t vec_type)
{
    LvType *v = lv_find(m, vec_type);
    unsigned bs = v ? v->bits / 8 : 4;
    unsigned n = v ? v->n : 1;
    if (n == 2) return 2 * bs;
    if (n == 3 || n == 4) return 4 * bs;
    return bs;
}

static unsigned lv_type_align(LvModule *m, LvType *t, bool std430, bool row_major)
{
    if (!t) return 4;

    switch (t->kind)
    {
        case LV_BOOL:
        case LV_INT:
        case LV_FLOAT:
            return t->bits / 8;
        case LV_VEC:
            return lv_vec_align(m, t->id);
        case LV_MAT:
        {
            /* rules 5/7: an array of column (row) vectors — a row-major matrix
             * is laid out as ROWS, each a vector of t->n (column count)
             * components (KHR-GL46 es_31_compatibility
             * shader_storage_buffer_object.basic-matrixOperations: row_major
             * mat2x3 aligns to 8, not to the vec3 column alignment) */
            unsigned elem;
            if (row_major)
            {
                LvType *e = lv_find(m, t->elem);
                unsigned bs = e ? e->bits / 8 : 4;
                unsigned n = t->n;
                elem = (n == 2) ? 2 * bs : (n == 3 || n == 4) ? 4 * bs : bs;
            }
            else
                elem = lv_vec_align(m, t->elem);
            return std430 ? elem : (elem < 16 ? 16 : elem);
        }
        case LV_ARRAY:
        case LV_RTARRAY:
        {
            LvType *e = lv_find(m, t->elem);
            if (e && (e->kind == LV_ARRAY || e->kind == LV_RTARRAY || e->kind == LV_STRUCT))
                return lv_type_align(m, e, std430, row_major);
            unsigned elem = e ? lv_type_align(m, e, std430, row_major) : 4;
            return std430 ? elem : (elem < 16 ? 16 : elem);   /* rule 4/9 rounding */
        }
        case LV_STRUCT:
        {
            unsigned a = 0;
            for (int i = 0; i < t->n_members; i++)
            {
                uint32_t mt = m->membertab[t->members + i];
                bool rm = row_major;
                if (lv_mdeco(m, t->id, i, SpvDecorationRowMajor, NULL)) rm = true;
                else if (lv_mdeco(m, t->id, i, SpvDecorationColMajor, NULL)) rm = false;
                unsigned ma = lv_member_align(m, mt, std430, rm);
                if (ma > a) a = ma;
            }
            return std430 ? a : (a < 16 ? 16 : a);            /* rule 9 rounding */
        }
        default:
            return 4;
    }
}

static unsigned lv_member_align(LvModule *m, uint32_t type_id, bool std430, bool row_major)
{
    LvType *t = lv_find(m, type_id);
    return lv_type_align(m, t, std430, row_major);
}

/* the std size of a matrix: number of columns (rows, row-major) x stride.
 * Prefer the module's own MatrixStride decoration: it is exactly what glslang
 * laid out (std140/430 + an explicit matrix_stride, when given). */
static unsigned lv_matrix_size(LvModule *m, LvType *t, bool std430, bool row_major)
{
    unsigned stride = 0;
    if (lv_deco(m, t->id, SpvDecorationMatrixStride, &stride)) { /* type decoration */ }
    if (!stride)
    {
        unsigned elem;
        if (row_major)
        {
            LvType *e = lv_find(m, t->elem);
            unsigned bs = e ? e->bits / 8 : 4;
            unsigned n = t->n;
            elem = (n == 2) ? 2 * bs : (n == 3 || n == 4) ? 4 * bs : bs;
        }
        else
            elem = lv_vec_align(m, t->elem);
        stride = std430 ? elem : (elem < 16 ? 16 : elem);
    }
    return (row_major ? t->rows : t->n) * stride;
}

static unsigned lv_type_size(LvModule *m, LvType *t, bool std430, bool row_major);

static unsigned lv_struct_size(LvModule *m, LvType *t, bool std430, bool row_major)
{
    unsigned size = 0, max_align = 0;
    for (int i = 0; i < t->n_members; i++)
    {
        uint32_t mt = m->membertab[t->members + i];
        LvType *me = lv_find(m, mt);
        bool rm = row_major;
        if (lv_mdeco(m, t->id, i, SpvDecorationRowMajor, NULL)) rm = true;
        else if (lv_mdeco(m, t->id, i, SpvDecorationColMajor, NULL)) rm = false;
        if (!me) continue;
        if (me->kind == LV_RTARRAY) continue;   /* unsized: ignored for sizing */
        unsigned a = lv_member_align(m, mt, std430, rm);
        size = lv_round_up(size, a);
        size += lv_type_size(m, me, std430, rm);
        if (a > max_align) max_align = a;
        if (me->kind == LV_STRUCT && i + 1 < t->n_members) size = lv_round_up(size, 16);
    }
    return lv_round_up(size, lv_type_align(m, t, std430, row_major));
}

static unsigned lv_type_size(LvModule *m, LvType *t, bool std430, bool row_major)
{
    if (!t) return 4;
    switch (t->kind)
    {
        case LV_BOOL:
        case LV_INT:
        case LV_FLOAT:
            return t->bits / 8;
        case LV_VEC:
            return t->n * (t->bits / 8);
        case LV_MAT:
            return lv_matrix_size(m, t, std430, row_major);
        case LV_ARRAY:
        {
            LvType *e = lv_find(m, t->elem);
            unsigned stride = 0;
            if (lv_deco(m, t->id, SpvDecorationArrayStride, &stride)) { }
            if (!stride)
            {
                unsigned elem = e ? lv_type_align(m, e, std430, row_major) : 4;
                stride = std430 ? elem : (elem < 16 ? 16 : elem);
            }
            return lv_array_length(m, t) * stride;
        }
        case LV_RTARRAY:
            return 0;
        case LV_STRUCT:
            return lv_struct_size(m, t, std430, row_major);
        default:
            return 4;
    }
}

/* ------------------------------------------------------------------ R1: block offsets */

/* check one interface block type: member offsets must be aligned and must not
 * overlap a previous member (Mesa ast_to_hir.cpp) */
static bool lv_check_block(LvModule *m, LvType *t, bool std430, char **log)
{
    uint32_t end = 0;           /* extent of the previous members */
    bool have = false;

    for (int i = 0; i < t->n_members; i++)
    {
        uint32_t mt = m->membertab[t->members + i];
        LvType *me = lv_find(m, mt);
        uint32_t off;
        bool rm = lv_mdeco(m, t->id, i, SpvDecorationRowMajor, NULL);

        if (!me) continue;
        /* array members are excluded: glslang sizes an implicitly-growing
         * array ("vec4 a[];" used at a[1]) beyond the layout extent its own
         * offsets imply — the r48 block-offset pass exists for exactly that —
         * so their extents cannot be validated against the decorations
         * (GL-level overlap of two array members is caught by glslang
         * itself; the CTS overlap families use plain scalar/vector/matrix
         * members). */
        if (me->kind == LV_ARRAY || me->kind == LV_RTARRAY) continue;
        /* only members the USER laid out participate: glslang assigns the
         * offsets of unqualified members itself (never overlapping), so
         * checking their extents would reject legal blocks */
        if (!lv_mdeco(m, t->id, i, SpvDecorationOffset, &off)) continue;

        unsigned align = lv_member_align(m, mt, std430, rm);
        unsigned size = lv_type_size(m, me, std430, rm);

        if (have && off < end)
        {
            if (log)
            {
                if (asprintf(log, "ERROR: 'layout' : offset of member '%s' of block '%s' "
                                  "overlaps previous member (offset %u, previous extent %u)\n",
                             lv_member_name(m, t->id, i) ? lv_member_name(m, t->id, i) : "?",
                             lv_name(m, t->id) ? lv_name(m, t->id) : "?",
                             off, end) < 0) { }
            }
            return false;
        }
        if (off % align)
        {
            if (log)
            {
                if (asprintf(log, "ERROR: 'layout' : offset of member '%s' of block '%s' "
                                  "must be a multiple of the base alignment %u (offset %u)\n",
                             lv_member_name(m, t->id, i) ? lv_member_name(m, t->id, i) : "?",
                             lv_name(m, t->id) ? lv_name(m, t->id) : "?",
                             align, off) < 0) { }
            }
            return false;
        }
        if (off + size > end) end = off + size;
        have = true;
    }
    return true;
}

/* ------------------------------------------------------------------ R2: location caps */

/* locations consumed by an in/out type (GL counts a dvec3/dvec4 as two) */
static unsigned lv_locations(LvModule *m, uint32_t type_id)
{
    LvType *t = lv_find(m, type_id);
    if (!t) return 1;
    switch (t->kind)
    {
        case LV_BOOL:
        case LV_INT:
        case LV_FLOAT:
        case LV_VEC:
        {
            if (t->bits == 64 && t->kind == LV_VEC && t->n >= 3) return 2;
            return 1;
        }
        case LV_MAT:
        {
            /* one location per column; a double column of 3-4 components
             * spans two (GL 4.6 11.1.1) */
            double per = (t->bits == 64 && t->rows > 2) ? 2 : 1;
            return (unsigned)((double)t->n * per);
        }
        case LV_ARRAY:
        {
            LvType *e = lv_find(m, t->elem);
            /* a per-vertex array (TCS/TES/GS in) counts once per vertex: the
             * limit is per-vertex, so count one element */
            if (e && (e->kind == LV_ARRAY || e->kind == LV_MAT || e->kind == LV_VEC || e->kind == LV_FLOAT))
                return lv_locations(m, t->elem);
            return lv_locations(m, t->elem);
        }
        case LV_STRUCT:
        {
            unsigned n = 0;
            for (int i = 0; i < t->n_members; i++)
                n += lv_locations(m, m->membertab[t->members + i]);
            return n ? n : 1;
        }
        default:
            return 1;
    }
}

/* the per-stage component caps, exactly as get.c reports them (ctx->state.var,
 * defaulted through LIMIT()) and capped by the neighbouring stage's side of
 * the interface — the CTS's getLastInput/OutputLocation() takes
 * min(param, paramPrev/Next).  Values are locations (components / 4, except
 * vertex attributes which are already locations). */
static int lv_lim(GLuint v, int def) { return v ? (int)v : def; }

static int lv_stage_location_limit(GLMContext ctx, int stage, bool input)
{
    int vs_out  = lv_lim(ctx->state.var.max_vertex_output_components, 128);
    int gs_in   = lv_lim(ctx->state.var.max_geometry_input_components, 128);
    int gs_out  = lv_lim(ctx->state.var.max_geometry_output_components, 128);
    int fs_in   = lv_lim(ctx->state.var.max_fragment_input_components, 128);
    /* tessellation caps are the constants get.c reports (128 components) */
    int tcs_in  = 128, tcs_out = 128, tes_in = 128, tes_out = 128;

    if (input)
    {
        switch (stage)
        {
            case _VERTEX_SHADER:          return 16;                                   /* attribs */
            case _TESS_CONTROL_SHADER:    return (tcs_in < vs_out ? tcs_in : vs_out) / 4;
            case _TESS_EVALUATION_SHADER: return (tes_in < tcs_out ? tes_in : tcs_out) / 4;
            case _GEOMETRY_SHADER:        return (gs_in < tes_out ? gs_in : tes_out) / 4;
            default:                      return (fs_in < gs_out ? fs_in : gs_out) / 4; /* fragment */
        }
    }
    else
    {
        switch (stage)
        {
            case _VERTEX_SHADER:          return (vs_out < tcs_in ? vs_out : tcs_in) / 4;
            case _TESS_CONTROL_SHADER:    return (tcs_out < tes_in ? tcs_out : tes_in) / 4;
            case _TESS_EVALUATION_SHADER: return tes_out / 4;
                /* compat r96d: the TES OUTPUT budget is its own cap (128 components = 32
                 * locations); the GS input cap applies to what the GS declares as inputs,
                 * not to TES outputs the GS does not consume — pairing them rejected
                 * TES-declared xfb captures (xfb_get_program_resource_api_dmat3/4, the
                 * dummy GS in those programs declares only its own varyings) */
            case _GEOMETRY_SHADER:        return (gs_out < fs_in ? gs_out : fs_in) / 4;
            default:                      return 1 << 30;   /* fragment: MAX_DRAW_BUFFERS */
        }
    }
}

/* check the in/out (varying) locations of one stage against its component caps */
/* storage class of the variable that instantiates block type `t` (a block
 * type has exactly one instance variable per module) */
static bool lv_block_storage(LvModule *m, LvType *t, uint32_t *storage_out)
{
    for (int v = 0; v < m->n_vars; v++)
    {
        LvType *ptr = lv_find(m, m->vars[v].type);
        if (ptr && ptr->kind == LV_PTR && ptr->elem == t->id)
        {
            if (storage_out) *storage_out = m->vars[v].storage;
            return true;
        }
    }
    return false;
}

/* MGL's per-vertex interface lowering (tess control out / tess eval + geometry
 * in) splits a block-instance array `out Goku {...} goku[];` into one variable
 * PER VERTEX, all sharing the OpName "goku", each with its own (shifted)
 * location: element i's members sit at 3*i+2, 3*i+4, ...  Per GL 4.6 11.1.2.1
 * the location cap applies PER VERTEX (the total is capped separately by
 * MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS), so only the lowest-location
 * element of such a split group may be checked (varying_block_member_locations
 * _tessellation_* regressed otherwise). */
/* tess r59: GL matches a producer OUTPUT to the next stage's INPUT by NAME
 * (GLSL 4.60 4.3.9); an output the next stage does not declare is INACTIVE —
 * it is given no interpolator storage and must not count against the
 * producer's location cap, however many of them transform feedback captures
 * (GL 4.6 13.2: "captured outputs need not be inputs of the subsequent
 * stage").  KHR-GL46.transform_feedback.get_xfb_varying links 30 unconsumed
 * mat2 vertex outputs (= 60 locations, all interleaved-captured, 120 <= 128
 * components) against a fragment stage that consumes nothing, and conformant
 * desktop GL links it — hardware interpolator budgets only cover the ACTIVE
 * (consumed) interface.  Returns true when `cm` (the next stage's module)
 * declares an input matching `var`: same variable/instance name, or the same
 * block TYPE name for interface blocks.  A NULL consumer means "unknown"
 * (separable stage): every output is treated as consumed — the pre-r59
 * behavior. */
static bool lv_consumed_by(LvModule *cm, LvModule *m, uint32_t var, LvType *bt)
{
    if (!cm) return true;
    const char *name = lv_name(m, var);
    bool is_block = bt && bt->kind == LV_STRUCT && lv_deco(m, bt->id, SpvDecorationBlock, NULL);
    const char *bname = is_block ? lv_name(m, bt->id) : NULL;

    if ((!name || !*name) && (!bname || !*bname)) return true;  /* unnamed: cannot prove unused */

    for (int v = 0; v < cm->n_vars; v++)
    {
        if (cm->vars[v].storage != SpvStorageClassInput) continue;
        LvType *cp = lv_find(cm, cm->vars[v].type);
        if (!cp || cp->kind != LV_PTR) continue;
        const char *cn = lv_name(cm, cm->vars[v].id);
        if (cn && name && *name && !strcmp(cn, name)) return true;
        if (bname && *bname)
        {
            LvType *cb = lv_find(cm, cp->elem);
            while (cb && cb->kind == LV_ARRAY) cb = lv_find(cm, cb->elem);
            if (cb && cb->kind == LV_STRUCT &&
                lv_deco(cm, cb->id, SpvDecorationBlock, NULL))
            {
                const char *cbn = lv_name(cm, cb->id);
                if (cbn && !strcmp(cbn, bname)) return true;
            }
        }
    }
    return false;
}

static bool lv_split_group_min(LvModule *m, uint32_t var, uint32_t *minloc,
                               unsigned *count)
{
    const char *name = lv_name(m, var);
    if (!name) return false;
    *count = 0;
    *minloc = 0xffffffffu;
    for (int v = 0; v < m->n_vars; v++)
    {
        if (m->vars[v].id == var) continue;
        const char *n2 = lv_name(m, m->vars[v].id);
        if (!n2 || strcmp(n2, name)) continue;
        uint32_t l = 0;
        if (lv_deco(m, m->vars[v].id, SpvDecorationLocation, &l) && l < *minloc)
            *minloc = l;
        (*count)++;
    }
    return *count > 0 && *minloc != 0xffffffffu;
}

/* compat r96e: component ALIASING of non-vertex inputs is a link error (GL
 * 4.6 11.1.1: "the components... declared for two inputs assigned to the same
 * location must not overlap" — vertex inputs may alias, that side is the
 * attribute path).  KHR-Single-GL46.enhanced_layouts.input_component_aliasing*
 * covers TCS/TES/GS/FS; the vertex variant must link. */
static char *lv_input_component_aliasing(LvModule *m, int stage)
{
    if (stage == _VERTEX_SHADER) return NULL;
    uint8_t mask[128] = {0};
    char who[128][128] = {{0}};
    for (int v = 0; v < m->n_vars; v++)
    {
        uint32_t var = m->vars[v].id;
        if (m->vars[v].storage != SpvStorageClassInput) continue;
        uint32_t loc, comp = 0;
        if (!lv_deco(m, var, SpvDecorationLocation, &loc)) continue;
        if (lv_deco(m, var, SpvDecorationComponent, &comp) && (comp > 3)) continue;  /* R2 covers */
        LvType *t = lv_find(m, m->vars[v].type);
        if (!t || t->kind != LV_PTR) continue;
        LvType *bt = lv_find(m, t->elem);
        while (bt && bt->kind == LV_ARRAY) bt = lv_find(m, bt->elem);
        if (!bt) continue;
        unsigned ncomp = 1, span = lv_locations(m, t->elem);
        if (bt->kind == LV_MAT) bt = lv_find(m, bt->elem);   /* align like a column */
        if (bt && bt->kind == LV_VEC) { ncomp = bt->n; if (bt->bits == 64) ncomp *= 2; }
        else if (bt && bt->bits == 64) ncomp = 2;
        if (ncomp > 4) ncomp = 4;
        uint32_t mk = ((1u << ncomp) - 1u) << comp;
        if (loc < 128)
        {
            if ((mask[loc] & mk) && strcmp(who[loc], lv_name(m, var) ? lv_name(m, var) : "?"))
            {
                char *l = NULL;
                if (asprintf(&l, "ERROR: 'component' : input '%s' aliases the components of "
                                  "'%s' at location %u of the %s stage's inputs\n",
                             lv_name(m, var) ? lv_name(m, var) : "?", who[loc], loc,
                             lv_stage_label(stage)) < 0) l = NULL;
                return l;
            }
            mask[loc] |= mk;
            snprintf(who[loc], sizeof who[loc], "%s", lv_name(m, var) ? lv_name(m, var) : "?");
            for (uint32_t s = 1; s < span && loc + s < 128; s++) mask[loc + s] = 0xF;
        }
    }
    return NULL;
}

static bool lv_check_locations(GLMContext ctx, LvModule *m, int stage, LvModule *cm, char **log)
{
    for (int v = 0; v < m->n_vars; v++)
    {
        uint32_t var = m->vars[v].id;
        uint32_t type = m->vars[v].type;
        uint32_t storage = m->vars[v].storage;
        bool input;
        uint32_t loc;

        LvType *ptr = lv_find(m, type);
        if (!ptr || ptr->kind != LV_PTR) continue;

        if (storage == SpvStorageClassInput) input = true;
        else if (storage == SpvStorageClassOutput) input = false;
        else continue;

        /* vertex inputs of every stage except the first are cross-stage
         * interfaces; every one of them is limited the same way */
        int limit = lv_stage_location_limit(ctx, stage, input);
        LvType *t = lv_find(m, ptr->elem);
        /* a block INSTANCE can be an array (per-vertex interfaces) — unwrap */
        LvType *bt = t;
        while (bt && bt->kind == LV_ARRAY) bt = lv_find(m, bt->elem);
        t = bt;

        /* an OUTPUT the next stage does not consume is inactive (no
         * interpolator storage): the location cap guards the ACTIVE
         * interface only — see lv_consumed_by above */
        if (!input && !lv_consumed_by(cm, m, var, t)) continue;

        /* interface BLOCK instances cannot be checked against the per-stage
         * location cap: glslang's AUTO_MAP_LOCATIONS (on for every GL shader
         * here, legacy shaders need it) INVENTS a Location for a block
         * instance when the user gave none — e.g. 16 for `out Goku {...}
         * goku;` in a tessellation evaluation stage whose 16 locations were
         * already used by other varyings (KHR-Single-GL46 enhanced_layouts
         * xfb_global_buffer_mat3x2/mat4/mat4x3).  Invented and user locations
         * are indistinguishable in SPIR-V, so the cap stays enforced for
         * plain varyings only (varying_location_limit_*).  Member locations,
         * when the user wrote them, are still checked below. */
        if (t && t->kind == LV_STRUCT && lv_deco(m, t->id, SpvDecorationBlock, NULL))
        {
            bool member_locs = false;
            for (int i = 0; i < t->n_members; i++)
                if (lv_mdeco(m, t->id, i, SpvDecorationLocation, NULL))
                { member_locs = true; break; }
            if (member_locs)
            {
                for (int i = 0; i < t->n_members; i++)
                {
                    uint32_t mloc = 0;
                    if (!lv_mdeco(m, t->id, i, SpvDecorationLocation, &mloc)) continue;
                    uint32_t mt = m->membertab[t->members + i];
                    unsigned slots = lv_locations(m, mt);
                    if (mloc + slots > (uint32_t)limit)
                    {
                        char *nl = NULL;
                        if (asprintf(&nl, "ERROR: 'location' : member %d of '%s' at location %u "
                                          "exceeds the %d available location(s) of the %s stage's %s\n",
                                     i, lv_name(m, t->id) ? lv_name(m, t->id) : "block",
                                     mloc, limit, lv_stage_label(stage),
                                     input ? "inputs" : "outputs") < 0)
                            *log = NULL;
                        else
                            *log = nl;
                        return false;
                    }
                }
            }
            continue;
        }

        /* a block instance whose members carry their own locations: the
         * MEMBERS' locations govern.  glslang's AUTO_MAP_LOCATIONS (on for
         * every GL shader here) invents a variable-level Location for the
         * instance array — e.g. 32 for `out Goku { layout(location=2).. }
         * goku[]` — which must not be checked against the cap. */
        if (lv_deco(m, var, SpvDecorationLocation, &loc))
        {
            unsigned slots = lv_locations(m, ptr->elem);
            uint32_t eff = loc;
            unsigned siblings = 0;
            uint32_t gmin = 0;
            if (lv_split_group_min(m, var, &gmin, &siblings) && gmin < eff)
                eff = gmin;   /* per-vertex split: cap applies per vertex */
            if (getenv("MGL_LV_DEBUG"))
                fprintf(stderr, "[LV] %sput var '%s' loc=%u eff=%u slots=%u limit=%d (stage %d)\n",
                        input ? "in" : "out", lv_name(m, var) ? lv_name(m, var) : "?",
                        loc, eff, slots, limit, stage);
            if (eff + slots > (uint32_t)limit)
            {
                if (log)
                {
                    if (asprintf(log, "ERROR: 'location' : %s '%s' at location %u exceeds "
                                      "the %d available location(s) of the %s stage's %sputs\n",
                                 input ? "input" : "output",
                                 lv_name(m, var) ? lv_name(m, var) : "?",
                                 loc, limit, lv_stage_label(stage), input ? "in" : "out") < 0) { }
                }
                return false;
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------ entry point */

/* ------------------------------------------------------------------ r97b: resource limits */

static uint64_t lv_type_bytes(LvModule *m, uint32_t id, int depth)
{
    LvType *t = lv_find(m, id);
    if (!t || depth > 8) return 0;
    switch (t->kind)
    {
    case LV_INT: case LV_FLOAT: case LV_BOOL:
        return t->bits ? (uint64_t)t->bits / 8 : 4;
    case LV_VEC:
        return (uint64_t)t->n * lv_type_bytes(m, t->elem, depth + 1);
    case LV_MAT:
        return (uint64_t)t->n * (uint64_t)t->rows * lv_type_bytes(m, t->elem, depth + 1);
    case LV_ARRAY:
    {
        uint64_t n = t->length;
        if (!n && t->length_id) n = lv_const(m, t->length_id);
        return n * lv_type_bytes(m, t->elem, depth + 1);
    }
    case LV_RTARRAY:
        return 0;
    case LV_STRUCT:
    {
        uint64_t s = 0;
        for (int i = 0; i < t->n_members; i++)
            s += lv_type_bytes(m, m->membertab[t->members + i], depth + 1);
        return s;
    }
    default:
        return 0;
    }
}


/* is this OpVariable a shader-storage block instance?  (glslang emits
 * StorageBuffer storage; older SPIR-V used Uniform + BufferBlock) */
static bool lv_is_ssbo_var(LvModule *m, int i)
{
    if (m->vars[i].storage == 12) return true;
    if (m->vars[i].storage != 2) return false;
    LvType *p = lv_find(m, m->vars[i].type);
    if (!p || p->kind != LV_PTR) return false;
    LvType *t = lv_find(m, p->elem);
    while (t && t->kind == LV_ARRAY) t = lv_find(m, t->elem);
    return t && t->kind == LV_STRUCT && lv_deco(m, t->id, SpvDecorationBufferBlock, NULL) != 0;
}

static char *lv_resource_limits(GLMContext ctx, LvModule *m, int stage, const uint32_t *ir, size_t words)
{
    /* --- compute local work-group size / shared memory ------------------- */
    if (stage == _COMPUTE_SHADER)
    {
        GLint mx = 0, my = 0, mz = 0, inv = 0, sh = 0;
        mglGetIntegeri_v(ctx, GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &mx);
        mglGetIntegeri_v(ctx, GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &my);
        mglGetIntegeri_v(ctx, GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &mz);
        mglGetIntegerv(ctx, GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &inv);
        for (size_t i = 5; i < words; )
        {
            uint32_t wc = ir[i] >> 16, op = ir[i] & 0xffff;
            if (wc == 0 || i + wc > words) break;
            if (op == 16 /*OpExecutionMode*/ && wc >= 6 && ir[i + 2] == 17 /*LocalSize*/)
            {
                uint32_t x = ir[i + 3], y = ir[i + 4], z = ir[i + 5];
                if ((mx && x > (uint32_t)mx) || (my && y > (uint32_t)my) || (mz && z > (uint32_t)mz))
                {
                    char *l = (char *)malloc(160);
                    if (l) snprintf(l, 160,
                        "ERROR: local work-group size (%u,%u,%u) exceeds GL_MAX_COMPUTE_WORK_GROUP_SIZE (%d,%d,%d)\n",
                        x, y, z, mx, my, mz);
                    return l;
                }
                if (inv && (uint64_t)x * y * z > (uint64_t)inv)
                {
                    char *l = (char *)malloc(160);
                    if (l) snprintf(l, 160,
                        "ERROR: local work-group size (%u,%u,%u) exceeds GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS (%d)\n",
                        x, y, z, inv);
                    return l;
                }
            }
            i += wc;
        }
        /* shared (Workgroup-storage) variables */
        mglGetIntegerv(ctx, GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &sh);
        if (sh)
        {
            uint64_t total = 0;
            for (int i = 0; i < m->n_vars; i++)
            {
                if (m->vars[i].storage != 4 /*Workgroup*/) continue;
                LvType *p = lv_find(m, m->vars[i].type);
                if (!p || p->kind != LV_PTR) continue;
                uint64_t b = lv_type_bytes(m, p->elem, 0);
                total += b;
                if (getenv("MGL_LV_DEBUG"))
                    printf("MGL lv: workgroup var %u bytes %llu\n", m->vars[i].id, (unsigned long long)b);
            }
            if (total > (uint64_t)sh)
            {
                char *l = (char *)malloc(160);
                if (l) snprintf(l, 160,
                    "ERROR: shared-memory usage (%llu bytes) exceeds GL_MAX_COMPUTE_SHARED_MEMORY_SIZE (%d)\n",
                    (unsigned long long)total, sh);
                return l;
            }
        }
    }

    /* --- per-stage uniform / shader-storage block counts ----------------- */
    {
        unsigned n_ub = 0, n_ssbo = 0;
        for (int i = 0; i < m->n_vars; i++)
        {
            if (m->vars[i].storage != 2 /*Uniform*/ && m->vars[i].storage != 12 /*StorageBuffer*/)
                continue;
            const char *nm = lv_name(m, m->vars[i].id);
            if (nm && nm[0] == 'm' && nm[1] == 'g' && nm[2] == 'l' && nm[3] == '_')
                continue;   /* MGL-injected emulation blocks are not user blocks */
            LvType *p = lv_find(m, m->vars[i].type);
            if (!p || p->kind != LV_PTR) continue;
            LvType *t = lv_find(m, p->elem);
            unsigned instances = 1;
            while (t && t->kind == LV_ARRAY)
            {
                uint64_t n = t->length;
                if (!n && t->length_id) n = lv_const(m, t->length_id);
                instances *= (unsigned)(n ? n : 1);
                t = lv_find(m, t->elem);
            }
            if (!t || t->kind != LV_STRUCT) continue;
            if (!lv_deco(m, t->id, SpvDecorationBlock, NULL) &&
                !lv_deco(m, t->id, SpvDecorationBufferBlock, NULL))
                continue;   /* plain (non-block) uniform */
            if (lv_is_ssbo_var(m, i)) n_ssbo += instances;
            else n_ub += instances;
        }
        static const GLenum ub_pname[_MAX_SHADER_TYPES] = {
            GL_MAX_VERTEX_UNIFORM_BLOCKS, GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS,
            GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS, GL_MAX_GEOMETRY_UNIFORM_BLOCKS,
            GL_MAX_FRAGMENT_UNIFORM_BLOCKS, GL_MAX_COMPUTE_UNIFORM_BLOCKS
        };
        static const GLenum ssbo_pname[_MAX_SHADER_TYPES] = {
            GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS,
            GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS,
            GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS
        };
        if (stage >= 0 && stage < _MAX_SHADER_TYPES)
        {
            GLint lim = 0;
            mglGetIntegerv(ctx, ub_pname[stage], &lim);
            if (lim && n_ub > (unsigned)lim)
            {
                char *l = (char *)malloc(160);
                if (l) snprintf(l, 160, "ERROR: %u uniform blocks exceed the stage limit (%d)\n", n_ub, lim);
                return l;
            }
            mglGetIntegerv(ctx, ssbo_pname[stage], &lim);
            if (lim && n_ssbo > (unsigned)lim)
            {
                char *l = (char *)malloc(160);
                if (l) snprintf(l, 160, "ERROR: %u shader storage blocks exceed the stage limit (%d)\n", n_ssbo, lim);
                return l;
            }
        }
    }
    return NULL;
}

/* combined shader-storage block count across all stages (r97b): Metal has no
 * combined limit, GL does — GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS.  Called from
 * program.c at link time with every compiled module. */
char *mgl_layout_combined_limits(GLMContext ctx, uint32_t **mods, size_t *sizes, unsigned mask)
{
    unsigned n_ssbo = 0;
    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        if (!(mask & (1u << s)) || !mods[s] || !sizes[s]) continue;
        LvModule *m = calloc(1, sizeof *m);
        if (!m || !lv_parse(mods[s], sizes[s], m)) { free(m); continue; }
        for (int i = 0; i < m->n_vars; i++)
        {
            if (!lv_is_ssbo_var(m, i)) continue;
            const char *nm = lv_name(m, m->vars[i].id);
            if (nm && nm[0] == 'm' && nm[1] == 'g' && nm[2] == 'l' && nm[3] == '_')
                continue;   /* MGL-injected emulation blocks are not user blocks */
            LvType *p = lv_find(m, m->vars[i].type);
            if (!p || p->kind != LV_PTR) continue;
            LvType *t = lv_find(m, p->elem);
            unsigned instances = 1;
            while (t && t->kind == LV_ARRAY)
            {
                uint64_t n = t->length;
                if (!n && t->length_id) n = lv_const(m, t->length_id);
                instances *= (unsigned)(n ? n : 1);
                t = lv_find(m, t->elem);
            }
            if (!t || t->kind != LV_STRUCT) continue;
            if (lv_deco(m, t->id, SpvDecorationBufferBlock, NULL) ||
                lv_deco(m, t->id, SpvDecorationBlock, NULL))
                n_ssbo += instances;
        }
        free(m);
    }
    GLint lim = 0;
    mglGetIntegerv(ctx, GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, &lim);
    if (lim && n_ssbo > (unsigned)lim)
    {
        char *l = (char *)malloc(160);
        if (l) snprintf(l, 160, "ERROR: %u shader storage blocks across stages exceed GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS (%d)\n",
                        n_ssbo, lim);
        return l;
    }
    return NULL;
}

char *mgl_layout_validate_stage(GLMContext ctx, int stage, const uint32_t *ir, size_t words,
                                const uint32_t *consumer_ir, size_t consumer_words)
{
    LvModule *m = calloc(1, sizeof *m);
    char *log = NULL;
    bool ok = true;

    if (!m) return NULL;
    if (!lv_parse(ir, words, m))
    {
        free(m);
        return NULL;    /* a module we cannot read: leave the decision to the
                           rest of the pipeline (SPIRV-Cross will complain) */
    }

    /* R0b (r97b): GL-reported resource limits are tighter than the Metal
     * hardware's — enforce them so over-limit shaders fail instead of running
     * (dEQP-GL45-ES31 debug.negative_coverage.{shader_storage,compute}.*:
     * local work-group size / invocations, per-stage uniform & shader-storage
     * block counts, shared-memory size) */
    {
        char *rl = lv_resource_limits(ctx, m, stage, ir, words);
        if (rl) { free(m); return rl; }
    }

    /* R0 (r96e): component aliasing of non-vertex inputs */
    {
        char *alias = lv_input_component_aliasing(m, stage);
        if (alias) { free(m); return alias; }
    }

    /* R1: interface blocks with explicit member offsets */
    for (int i = 0; i < m->n_types && ok; i++)
    {
        LvType *t = &m->types[i];
        bool std430;

        if (t->kind != LV_STRUCT || t->n_members <= 0) continue;
        if (lv_deco(m, t->id, SpvDecorationBlock, NULL)) std430 = false;
        else if (lv_deco(m, t->id, SpvDecorationBufferBlock, NULL)) std430 = true;
        else continue;

        /* only blocks that carry explicit member offsets are laid out by the
         * user — the implicit packings glslang generated are consistent by
         * construction */
        bool any_offset = false;
        for (int k = 0; k < t->n_members; k++)
            if (lv_mdeco(m, t->id, k, SpvDecorationOffset, NULL)) { any_offset = true; break; }
        if (!any_offset) continue;

        /* MGL's own injected blocks are laid out by MGL's packer (geometry.c:
         * the default-block uniform emulation "mgl_DefaultUniforms[_N]" /
         * "gl_DefaultUniformBlock", the std430 dub SSBO) and read back by the
         * same packer — they are not user GLSL and must not be held to the
         * 4.4.5 rules (KHR-GL46.program_interface_query.uniform-types
         * regressed: the packer's tight offsets put a 16-aligned member at
         * 184). */
        {
            const char *bn = lv_name(m, t->id);
            if (bn && (!strncmp(bn, "mgl_", 4) ||
                       !strcmp(bn, "gl_DefaultUniformBlock")))
                continue;
        }

        /* interface (xfb) blocks: an `out Block { layout(xfb_offset=..) ... }`
         * carries TRANSFORM-FEEDBACK offsets, not buffer-layout offsets — the
         * members' sizes there are the tight xfb sizes (a mat2 is 16 bytes),
         * so the 4.4.5 std140/std430 overlap rules do not apply and applying
         * them false-rejects legal programs (KHR-Single-GL46 enhanced_layouts
         * xfb_global_buffer_* regressed). */
        {
            uint32_t bstorage = 0;
            if (lv_block_storage(m, t, &bstorage) &&
                (bstorage == SpvStorageClassOutput || bstorage == SpvStorageClassInput))
                continue;
        }

        ok = lv_check_block(m, t, std430, &log);
    }

    /* R2: varying location limits.  consumer_ir = the next stage's module
     * (NULL for a separable stage or when there is none): an output it does
     * not declare is inactive and skips the cap (lv_consumed_by). */
    if (ok)
    {
        LvModule *cm = NULL;
        LvModule cmstore;
        if (consumer_ir && consumer_words >= 5 && consumer_ir[0] == SpvMagicNumber &&
            lv_parse(consumer_ir, consumer_words, &cmstore))
            cm = &cmstore;
        ok = lv_check_locations(ctx, m, stage, cm, &log);
    }

    free(m);
    return ok ? NULL : log;
}

/* ------------------------------------------------------------------ R3 (compat r96)
 * glslang's link-time auto-map numbers location-less per-vertex interface
 * arrays (TCS in/out, TES in, GS in) with VULKAN flattening: an
 * `in uint foo[];` occupies gl_MaxPatchVertices CONTIGUOUS slots, so the
 * invented Location lands far past the GL per-vertex budget while GL assigns
 * first-free (KHR-Single-GL46.enhanced_layouts.varying_components_gvec4_float:
 * `vs_tcs_result` is invented at 63 although the interface only uses locations
 * 0 and 31) — R2 above would then reject a program GL accepts.  Renumber, on
 * BOTH sides of every stage boundary so producer and consumer stay consistent,
 * every interface group whose invented location does not fit the boundary's
 * location budget; groups that fit keep glslang's (jointly consistent) number.
 * Explicitly located variables are never moved. */
void mgl_layout_renumber_io(GLMContext ctx, uint32_t **mods, size_t *sizes, unsigned mask);
void mgl_layout_renumber_io_srcs(GLMContext ctx, uint32_t **mods, size_t *sizes, unsigned mask,
                                 const char **srcs);

typedef struct {
    int       stage;         /* module the variable lives in */
    uint32_t  var;
    uint32_t  loc;
    bool      has_loc;
    unsigned  width;         /* GL locations per vertex */
    bool      input;         /* an input of its stage (else an output) */
    bool      user_loc;      /* `location` qualifier in the GLSL source (never renumbered) */
} RnVar;

typedef struct {
    char      name[128];
    RnVar     v[4];          /* producer + consumer (+ per-vertex split siblings) */
    int       nv;
    uint32_t  newloc;
    bool      renum;
    int       cap;           /* tightest per-vertex location budget of its entries */
} RnGroup;

static unsigned rn_width(LvModule *m, int stage, bool input, uint32_t var, uint32_t ptr_type)
{
    LvType *p = lv_find(m, ptr_type);
    if (!p || p->kind != LV_PTR) return 1;
    uint32_t t = p->elem;
    /* per-vertex interfaces strip ONE array level (io_loc_count rule) */
    bool arrayed = false;
    if (stage == _GEOMETRY_SHADER || stage == _TESS_EVALUATION_SHADER) arrayed = input;
    else if (stage == _TESS_CONTROL_SHADER) arrayed = true;
    if (arrayed && !lv_deco(m, var, 28 /* Patch */, NULL))
    {
        LvType *a = lv_find(m, t);
        if (a && (a->kind == LV_ARRAY || a->kind == LV_RTARRAY)) t = a->elem;
    }
    return lv_locations(m, t) ? lv_locations(m, t) : 1;
}

/* is the variable an interface-block INSTANCE (block type decoration)? */
static bool rn_block_instance(LvModule *m, uint32_t ptr_type)
{
    LvType *p = lv_find(m, ptr_type);
    if (!p || p->kind != LV_PTR) return false;
    LvType *t = lv_find(m, p->elem);
    while (t && t->kind == LV_ARRAY) t = lv_find(m, t->elem);
    return t && t->kind == LV_STRUCT &&
           (lv_deco(m, t->id, SpvDecorationBlock, NULL) ||
            lv_deco(m, t->id, SpvDecorationBufferBlock, NULL));
}

static void rn_set_loc(uint32_t *ir, size_t words, uint32_t var, uint32_t loc)
{
    for (size_t i = 5; i < words; )
    {
        uint32_t wc = ir[i] >> 16, op = ir[i] & 0xffff;
        if (wc == 0 || i + wc > words) return;
        if (op == SpvOpDecorate && wc >= 4 && ir[i + 1] == var && ir[i + 2] == 30 /* Location */)
            ir[i + 3] = loc;
        i += wc;
    }
}

/* compat r96c: names whose declaration in the GLSL source carries an explicit
 * `location` qualifier.  glslang's link-time auto-map INVENTS Location
 * decorations for location-less interfaces, so the decoration alone cannot say
 * whether a number past the budget is a user location (GL must fail the link)
 * or glslang's Vulkan flattening (GL accepts; R3 renumbers).  The source can.
 * A declaration is scanned from each `location` keyword to the terminating
 * ';' and its LAST identifier (the declared name, before any '[') is taken. */
static int rn_src_located(const char *src, char set[][80], int max)
{
    int n = 0;
    if (!src) return 0;
    for (const char *p = src; *p && n < max; p++)
    {
        if (strncmp(p, "location", 8) != 0) continue;
        if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) ||
            (isalnum((unsigned char)p[8]) || p[8] == '_')) continue;   /* first_input_location */
        const char *e = p;
        while (*e && *e != ';' && *e != '{') e++;
        const char *last = NULL;
        size_t lastlen = 0;
        for (const char *q = p; q < e; )
        {
            if (isalpha((unsigned char)*q) || *q == '_')
            {
                const char *s = q;
                while (q < e && (isalnum((unsigned char)*q) || *q == '_')) q++;
                last = s; lastlen = (size_t)(q - s);
            }
            else q++;
        }
        if (last && lastlen > 0 && lastlen < 80)
        {
            int dup = 0;
            for (int i = 0; i < n; i++) if (!strncmp(set[i], last, lastlen) && !set[i][lastlen]) { dup = 1; break; }
            if (!dup)
            {
                memcpy(set[n], last, lastlen);
                set[n][lastlen] = 0;
                n++;
            }
        }
        p = e ? e - 1 : p;   /* past the declaration (loop adds 1) */
    }
    return n;
}

void mgl_layout_renumber_io(GLMContext ctx, uint32_t **mods, size_t *sizes, unsigned mask)
{
    mgl_layout_renumber_io_srcs(ctx, mods, sizes, mask, NULL);
}

void mgl_layout_renumber_io_srcs(GLMContext ctx, uint32_t **mods, size_t *sizes, unsigned mask,
                                 const char **srcs)
{
    const bool dbg = getenv("MGL_RENUM_DEBUG") != NULL;
    if (getenv("MGL_NO_RENUM")) return;
    LvModule *mod[_MAX_SHADER_TYPES] = {0};
    int order[_MAX_SHADER_TYPES], norder = 0;
    char located[_MAX_SHADER_TYPES][64][80];
    int nlocated[_MAX_SHADER_TYPES] = {0};

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
        if (srcs && srcs[s]) nlocated[s] = rn_src_located(srcs[s], located[s], 64);

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        if (!(mask & (1u << s)) || !mods[s] || !sizes[s]) continue;
        if (s == _COMPUTE_SHADER) continue;
        mod[s] = calloc(1, sizeof *mod[s]);
        if (!mod[s] || !lv_parse(mods[s], sizes[s], mod[s])) { free(mod[s]); mod[s] = NULL; continue; }
        order[norder++] = s;
    }
    if (dbg) fprintf(stderr, "[RENUM] entry: norder=%d stages:", norder);
    if (dbg) for (int i = 0; i < norder; i++) fprintf(stderr, " %d", order[i]);
    if (dbg) fprintf(stderr, "\n");
    if (norder < 1)
    {
        for (int s = 0; s < _MAX_SHADER_TYPES; s++) free(mod[s]);
        return;
    }

    /* one interface walk per stage boundary (+ a solo walk for a stage edge
     * that has no partner in this program: a separable / single-stage link
     * still must fit the caps, and GL numbers its varyings first-free) */
    for (int b = 0; b < norder; b++)
    {
        int pa = order[b];
        int pb = (b + 1 < norder) ? order[b + 1] : -1;   /* consumer stage */

        for (int side = 0; side < 2; side++)
        {
            /* side 0: outputs of pa (paired with pb's inputs when pb >= 0)
             * side 1: inputs of pa when it is the FIRST stage (no producer) */
            int st, other;
            uint32_t want, other_want;
            if (side == 0)
            {
                if (pa == _FRAGMENT_SHADER) continue;
                st = pa; other = pb;
                want = SpvStorageClassOutput; other_want = SpvStorageClassInput;
            }
            else
            {
                if (b != 0 || pa == _VERTEX_SHADER) continue;
                st = pa; other = -1;
                want = SpvStorageClassInput; other_want = SpvStorageClassOutput;
            }
            int cap = lv_stage_location_limit(ctx, st, want == SpvStorageClassInput);
            RnGroup groups[128];
            int ng = 0;

            for (int which = 0; which < 2; which++)
            {
                int s2 = which ? other : st;
                uint32_t w2 = which ? other_want : want;
                bool in_side = (w2 == SpvStorageClassInput);
                LvModule *m = s2 >= 0 ? mod[s2] : NULL;
                if (!m) continue;
                for (int v = 0; v < m->n_vars; v++)
                {
                    uint32_t var = m->vars[v].id;
                    if (m->vars[v].storage != w2) continue;
                    if (lv_deco(m, var, SpvDecorationBuiltIn, NULL)) continue;
                    if (rn_block_instance(m, m->vars[v].type)) continue;
                    const char *nm = lv_name(m, var);
                    if (!nm || !*nm) continue;
                    uint32_t loc = 0;
                    bool has = lv_deco(m, var, 30 /* Location */, &loc);

                    RnGroup *g = NULL;
                    for (int i = 0; i < ng; i++)
                        if (!strcmp(groups[i].name, nm)) { g = &groups[i]; break; }
                    if (!g)
                    {
                        if (ng >= (int)(sizeof groups / sizeof groups[0])) continue;
                        g = &groups[ng++];
                        memset(g, 0, sizeof *g);
                        g->newloc = 0xffffffffu;
                        g->cap = 0x7fffffff;
                        snprintf(g->name, sizeof g->name, "%s", nm);
                    }
                    if (g->nv < (int)(sizeof g->v / sizeof g->v[0]))
                    {
                        RnVar *rv = &g->v[g->nv++];
                        rv->stage = s2; rv->var = var; rv->loc = loc; rv->has_loc = has;
                        rv->input = in_side;
                        rv->width = rn_width(m, s2, in_side, var, m->vars[v].type);
                        rv->user_loc = false;
                        for (int u = 0; u < nlocated[s2]; u++)
                            if (!strcmp(located[s2][u], nm)) { rv->user_loc = true; break; }
                        /* compat r96c: the budget that matters for a variable is its
                         * OWN stage's (a VS->GS boundary pairs 32-location VS outputs
                         * with 16-location GS inputs); use the tightest of the pair */
                        int vcap = lv_stage_location_limit(ctx, s2, in_side);
                        if (vcap < g->cap) g->cap = vcap;
                    }
                }
            }

            /* which groups must move: invented location past the interface
             * budget, or a producer/consumer disagreement (glslang keeps both
             * sides jointly consistent, so that is a safety net only)
             *
             * compat r96c: a USER-written location past the cap must FAIL the
             * link (GL 4.6 11.1.1; the R2 cap check rejects it), so only
             * locations glslang INVENTED may be renumbered.  Two shapes are
             * glslang's flattening: (a) the span starts inside the budget and
             * overflows by WIDTH (vs_tcs_result[] flattened from 0), and (b)
             * the span starts past the cap because earlier flattened groups
             * densely consumed the whole budget below it.  A lone high
             * location over a sparse interface is a user location. */
            {
                uint8_t all_used[1024] = {0};
                for (int i = 0; i < ng; i++)
                {
                    RnGroup *g = &groups[i];
                    for (int k = 0; k < g->nv; k++)
                    {
                        if (!g->v[k].has_loc) continue;
                        for (unsigned x = 0; x < g->v[k].width && g->v[k].loc + x < 1024; x++)
                            all_used[g->v[k].loc + x] = 1;
                    }
                }
                for (int i = 0; i < ng; i++)
                {
                    RnGroup *g = &groups[i];
                    bool user = false;
                    for (int k = 0; k < g->nv; k++) if (g->v[k].user_loc) user = true;
                    for (int k = 0; k < g->nv && !g->renum; k++)
                    {
                        if (!g->v[k].has_loc) continue;
                        if (user) break;   /* an explicit location never moves */
                        if (g->v[k].loc + g->v[k].width <= (uint32_t)g->cap) continue;
                        if (g->v[k].loc < (uint32_t)g->cap) { g->renum = true; break; }   /* (a) */
                        /* (b): every slot below min(loc, cap) is used — but NOT by this
                         * group's own span (a wide flattened array reaching past the cap
                         * is shape (a) and already handled) */
                        uint32_t lim = g->v[k].loc < (uint32_t)g->cap ? g->v[k].loc : (uint32_t)g->cap;
                        bool dense = true;
                        for (uint32_t s = 0; s < lim; s++)
                        {
                            if (all_used[s]) continue;
                            if (g->v[k].loc <= s && s < g->v[k].loc + g->v[k].width) continue;
                            dense = false; break;
                        }
                        if (dense) { g->renum = true; break; }
                    }
                    if (!g->renum && g->nv == 2 && g->v[0].has_loc && g->v[1].has_loc &&
                        g->v[0].loc != g->v[1].loc)
                        g->renum = true;
                }
            }

            /* span map from the groups that keep their numbers */
            uint8_t used[1024] = {0};
            for (int i = 0; i < ng; i++)
            {
                RnGroup *g = &groups[i];
                if (g->renum) continue;
                for (int k = 0; k < g->nv; k++)
                {
                    if (!g->v[k].has_loc) continue;
                    for (unsigned x = 0; x < g->v[k].width && g->v[k].loc + x < 1024; x++)
                        used[g->v[k].loc + x] = 1;
                }
            }
            /* first-fit for the moved groups, ordered by their old location */
            for (;;)
            {
                int pick = -1;
                for (int i = 0; i < ng; i++)
                {
                    RnGroup *g = &groups[i];
                    if (!g->renum || g->newloc != 0xffffffffu) continue;
                    if (pick < 0) { pick = i; continue; }
                    uint32_t a = g->v[0].has_loc ? g->v[0].loc : 0;
                    uint32_t b2 = groups[pick].v[0].has_loc ? groups[pick].v[0].loc : 0;
                    if (a < b2 || (a == b2 && strcmp(g->name, groups[pick].name) < 0)) pick = i;
                }
                if (pick < 0) break;
                RnGroup *g = &groups[pick];
                unsigned w = 1;
                for (int k = 0; k < g->nv; k++) if (g->v[k].width > w) w = g->v[k].width;
                uint32_t base = 0xffffffffu;
                for (uint32_t s = 0; s + w <= (uint32_t)g->cap; s++)
                {
                    bool free_span = true;
                    for (unsigned x = 0; x < w; x++) if (used[s + x]) { free_span = false; break; }
                    if (free_span) { base = s; break; }
                }
                if (base == 0xffffffffu)
                {
                    /* no room inside the budget: leave it alone; R2 rejects it */
                    g->renum = false;
                    continue;
                }
                for (unsigned x = 0; x < w; x++) used[base + x] = 1;
                g->newloc = base;
            }

            for (int i = 0; i < ng; i++)
            {
                RnGroup *g = &groups[i];
                if (!g->renum || g->newloc == 0xffffffffu) continue;
                for (int k = 0; k < g->nv; k++)
                {
                    if (!g->v[k].has_loc) continue;   /* introspect assigns those */
                    rn_set_loc(mods[g->v[k].stage], sizes[g->v[k].stage], g->v[k].var, g->newloc);
                    if (dbg)
                        fprintf(stderr, "[RENUM] '%s' stage %d var %u: %u -> %u (cap %d)\n",
                                g->name, g->v[k].stage, g->v[k].var, g->v[k].loc, g->newloc, cap);
                }
            }
        }
    }

    for (int s = 0; s < _MAX_SHADER_TYPES; s++) free(mod[s]);
}

/* ------------------------------------------------------------------ R4 (compat r96)
 * A hardware post-tessellation evaluation shader reads control-point and
 * per-patch inputs through [[attribute(N)]] whose index SPIRV-Cross takes from
 * the GLSL location.  Metal bounds those indices at 0..30, GL budgets 32
 * TCS-output / TES-input locations, so a legal `layout(location=31) in vec4`
 * fails the Metal compile ("attribute parameter is out of bounds").  The index
 * only keys MGL's own vertex descriptor (built from this same MSL text), so
 * renumbering the attributes densely in order of appearance keeps the data
 * flow — and also gives cp and per-patch inputs (two separate GL location
 * spaces) non-colliding descriptor slots. */
char *mgl_msl_compress_tes_attributes(const char *msl)
{
    if (!msl || getenv("MGL_NO_TES_ATTRCOMP")) return NULL;
    const char *p = msl;
    bool touched = false;

    /* pass 1: does anything actually need moving - an index past Metal's 0..30
     * bound, or the same index reused by two sites (the cp and per-patch
     * structs key one descriptor table)? */
    int seen[128], count[128], nseen = 0;
    while ((p = strstr(p, "[[attribute(")) != NULL)
    {
        int v = atoi(p + 12);
        if (v > 30) touched = true;
        bool known = false;
        for (int i = 0; i < nseen; i++)
            if (seen[i] == v) { count[i]++; known = true; }
        if (!known && nseen < 128) { seen[nseen] = v; count[nseen] = 1; nseen++; }
        p += 12;
    }
    for (int i = 0; i < nseen && !touched; i++)
        if (count[i] > 1) touched = true;
    if (!touched) return NULL;

    /* pass 2: rewrite.  Every [[attribute(N)]] site is a distinct stage-in
     * member (a member that spans two slots carries both, adjacent); the one
     * vertex-descriptor attribute table they key is shared by the cp and the
     * per-patch structs, so sites are numbered sequentially — unique by
     * construction, which also stops cp/per-patch location-space collisions. */
    char *out = malloc(strlen(msl) + 64);
    if (!out) return NULL;
    {
        const char *src = msl;
        char *dst = out;
        int next = 0;

        while (*src)
        {
            if (src[0] == '[' && src[1] == '[' && !strncmp(src, "[[attribute(", 12))
            {
                dst += sprintf(dst, "[[attribute(%d)", next++);
                src += 12;
                while (*src && *src != ')') src++;
                if (*src == ')') src++;
                continue;
            }
            *dst++ = *src++;
        }
        *dst = 0;
    }
    return out;
}

/* ===================================================================== *
 * Arrays-of-matrices OUTPUT lowering (compat r91)
 *
 * SPIRV-Cross: "MSL cannot emit arrays-of-matrices in input and output
 * variables" — Metal varyings cannot be arrays of matrices.  MGL already
 * lowers stage INPUTS that way (geometry.c mgl_spirv_lower_aom_input, hooked
 * in program.c); a VERTEX OUTPUT like
 *
 *     layout(location=0) flat out dmat2 goku[3];   (KHR-Single-GL46
 *     ...                                          enhanced_layouts.xfb_
 *     goku = uni_goku;                             explicit_location_dmat*)
 *
 * had no lowering and failed the stage's SPIR-V -> MSL translation.  This
 * pass rewrites every Output variable of array-of-matrix type into
 *   - one Output variable per array ELEMENT (matrix type), consecutive
 *     locations (cols * (2 for wide double columns)), and
 *   - a Private "shadow" array that the body keeps using under the original
 *     id, flushed to the element outputs before every terminator of the
 *     entry function.
 * xfb_* decorations of the array are transferred per element
 * (xfb_offset advanced by the array stride, xfb_buffer / xfb_stride kept) so
 * declared transform-feedback capture of the whole array still resolves.
 */

typedef struct { uint32_t *w; size_t n, cap; } AomOut;

static void aom_o_push(AomOut *o, uint32_t x)
{
    if (o->n == o->cap) { o->cap = o->cap ? o->cap * 2 : 4096; o->w = realloc(o->w, o->cap * 4); }
    o->w[o->n++] = x;
}

static void aom_o_inst(AomOut *o, uint16_t op, int wc, const uint32_t *w)
{
    /* wc = total word count including the opcode word */
    aom_o_push(o, ((uint32_t)wc << 16) | op);
    for (int i = 0; i < wc - 1; i++) aom_o_push(o, w[i]);
}

uint32_t *mgl_spirv_lower_aom_output(const uint32_t *in, size_t n, size_t *out_n)
{
    LvModule *m = calloc(1, sizeof *m);
    struct { uint32_t var, ptr_t, arr_t, elem_t, elem_ptr_t, loc, count, locs, stride, xfbb, xfbo, xfbs; bool xfb; } v[16];
    struct { uint32_t from, to; } map[256];      /* access-chain result -> element var */
    int nv = 0, nmap = 0;
    uint32_t entry = 0, model = 0;
    bool trace = getenv("MGL_AOM_DEBUG") != NULL;

    if (!m) return NULL;
    if (!lv_parse(in, n, m)) { if (trace) fprintf(stderr, "aom-out: parse fail\n"); free(m); return NULL; }

    for (int i = 0; i < m->n_vars && nv < 16; i++)
    {
        LvType *ptr = lv_find(m, m->vars[i].type);
        if (!ptr || ptr->kind != LV_PTR) continue;
        if (m->vars[i].storage != SpvStorageClassOutput) continue;
        LvType *arr = lv_find(m, ptr->elem);
        if (!arr || arr->kind != LV_ARRAY) continue;
        LvType *mat = lv_find(m, arr->elem);
        if (!mat || mat->kind != LV_MAT) continue;

        uint32_t loc;
        if (!lv_deco(m, m->vars[i].id, SpvDecorationLocation, &loc)) continue;
        /* locations consumed by one matrix element: one per column, two for
         * double columns of 3-4 components (GL 4.6 11.1.1) */
        uint32_t slot = (mat->bits == 64 && mat->rows >= 3) ? 2 : 1;
        uint32_t count = lv_array_length(m, arr);
        if (!count || count > 64) continue;

        memset(&v[nv], 0, sizeof v[0]);
        v[nv].var = m->vars[i].id;
        v[nv].ptr_t = m->vars[i].type;
        v[nv].arr_t = ptr->elem;
        v[nv].elem_t = arr->elem;
        v[nv].loc = loc;
        v[nv].count = count;
        v[nv].locs = mat->n * slot;
        lv_deco(m, arr->id, SpvDecorationArrayStride, &v[nv].stride);
        if (lv_deco(m, m->vars[i].id, SpvDecorationXfbBuffer, &v[nv].xfbb))
        {
            v[nv].xfb = true;
            lv_deco(m, m->vars[i].id, SpvDecorationOffset, &v[nv].xfbo);
            lv_deco(m, m->vars[i].id, SpvDecorationXfbStride, &v[nv].xfbs);
        }
        /* every access to the array must be through a constant element index
         * (glslang decomposes whole-array copies that way); a dynamic index
         * cannot be lowered — leave the whole variable alone in that case */
        bool dyn = false;
        for (size_t o = 5; o + 5 <= n && !dyn; )
        {
            uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
            if (wc == 0 || o + wc > n) break;
            const uint32_t *w = in + o;
            if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5 && w[3] == m->vars[i].id)
            {
                uint32_t idx = lv_const(m, w[4]);
                bool is_const = false;
                for (int c = 0; c < m->n_consts; c++) if (m->consts[c].id == w[4]) is_const = true;
                if (!is_const || idx >= count) dyn = true;
            }
            o += wc;
        }
        if (dyn) { if (trace) fprintf(stderr, "aom-out: var %u has dynamic index - skipped\n", m->vars[i].id); continue; }
        nv++;
    }

    /* the entry point id */
    {
        size_t o = 5;
        while (o + 4 <= n)
        {
            uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
            if (wc == 0 || o + wc > n) break;
            if (op == 15) { model = in[o + 1]; entry = in[o + 2]; break; }   /* OpEntryPoint */
            o += wc;
        }
    }
    if (!nv || !entry) { free(m); return NULL; }
    if (trace) fprintf(stderr, "aom-out: lowering %d array-of-matrix outputs (entry=%u)\n", nv, entry);

    uint32_t elem_ids[16][64];
    uint32_t next_id = in[3];
    for (int k = 0; k < nv; k++)
        for (uint32_t e = 0; e < v[k].count; e++)
            elem_ids[k][e] = next_id++;

    /* a store of the WHOLE array to the variable must be rewritten into a
     * store per element variable (OpCompositeExtract each element); a direct
     * OpLoad of the dropped variable cannot be expressed here - that aborts
     * the lowering (the caller keeps the original module). */
    bool array_load = false;
    for (size_t o = 5; o + 5 <= n; )
    {
        uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;
        if (op == SpvOpLoad && wc >= 4)
            for (int k = 0; k < nv; k++) if (in[o + 3] == v[k].var) array_load = true;
        o += wc;
    }
    if (array_load)
    {
        if (trace) fprintf(stderr, "aom-out: whole-array load present - lowering aborted\n");
        free(m);
        return NULL;
    }

    /* map: every constant-index access chain into the array that consumes
     * EXACTLY the array index resolves to the element variable itself (same
     * Output/matrix pointer type).  Deeper chains (into the matrix columns)
     * are rewritten in the emit loop: base -> element var, first index
     * dropped. */
    for (size_t o = 5; o + 5 <= n; )
    {
        uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;
        const uint32_t *w = in + o;
        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc == 5)
        {
            for (int k = 0; k < nv; k++)
                if (w[3] == v[k].var)
                {
                    uint32_t idx = lv_const(m, w[4]);
                    if (nmap < 256)
                    {
                        map[nmap].from = w[2];                  /* the access chain's result id */
                        map[nmap].to = elem_ids[k][idx];
                        nmap++;
                    }
                    break;
                }
        }
        o += wc;
    }

    AomOut out; memset(&out, 0, sizeof out);
    for (int i = 0; i < 5; i++) aom_o_push(&out, in[i]);

    bool globals_done = false, ann_done = false;
    for (size_t o = 5; o < n; )
    {
        uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;
        const uint32_t *w = in + o;

        /* OpTypePointer: result, storage, pointee — find the Output pointer
         * to the matrix type the body's access chains already use */
        if (op == SpvOpTypePointer && wc >= 4 && w[2] == SpvStorageClassOutput)
            for (int k = 0; k < nv; k++)
                if (!v[k].elem_ptr_t && w[3] == v[k].elem_t)
                    v[k].elem_ptr_t = w[1];

        /* the element decorations go into the annotation section (before the
         * type section starts at the first OpType*) */
        if (!ann_done && op >= 19 && op <= 37)
        {
            for (int k = 0; k < nv; k++)
                for (uint32_t e = 0; e < v[k].count; e++)
                {
                    uint32_t dv[3] = { elem_ids[k][e], SpvDecorationLocation, v[k].loc + e * v[k].locs };
                    aom_o_inst(&out, SpvOpDecorate, 4, dv);
                    if (v[k].xfb && v[k].stride)
                    {
                        uint32_t b1[3] = { elem_ids[k][e], SpvDecorationXfbBuffer, v[k].xfbb };
                        aom_o_inst(&out, SpvOpDecorate, 4, b1);
                        uint32_t b2[3] = { elem_ids[k][e], SpvDecorationOffset, v[k].xfbo + e * v[k].stride };
                        aom_o_inst(&out, SpvOpDecorate, 4, b2);
                        if (v[k].xfbs)
                        {
                            uint32_t b3[3] = { elem_ids[k][e], SpvDecorationXfbStride, v[k].xfbs };
                            aom_o_inst(&out, SpvOpDecorate, 4, b3);
                        }
                    }
                }
            ann_done = true;
        }

        /* element variables at the end of the global section */
        if (!globals_done && op == SpvOpFunction)
        {
            for (int k = 0; k < nv; k++)
            {
                for (uint32_t e = 0; e < v[k].count; e++)
                {
                    uint32_t vv[3] = { v[k].elem_ptr_t, elem_ids[k][e], SpvStorageClassOutput };
                    aom_o_inst(&out, SpvOpVariable, 4, vv);
                }
            }
            globals_done = true;
        }

        /* drop: the array variable itself, its name/decorations, and the
         * constant-index access chains (their results are re-pointed at the
         * element variables) */
        if (op == SpvOpVariable && wc >= 3)
        {
            bool drop = false;
            for (int k = 0; k < nv; k++) if (w[2] == v[k].var) drop = true;
            if (drop) { o += wc; continue; }
        }
        if ((op == SpvOpName || op == SpvOpDecorate) && wc >= 2)
        {
            bool drop = false;
            for (int k = 0; k < nv; k++) if (w[1] == v[k].var) drop = true;
            if (drop) { o += wc; continue; }
        }
        /* a store of the whole array to the dropped variable becomes a store
         * of each element: OpStore elem[e], OpCompositeExtract(value, e) */
        if (op == SpvOpStore && wc == 3)
        {
            int hit = -1;
            for (int k = 0; k < nv; k++) if (w[1] == v[k].var) hit = k;
            if (hit >= 0)
            {
                uint32_t val = w[2];
                for (uint32_t e = 0; e < v[hit].count; e++)
                {
                    uint32_t ext = next_id++;
                    uint32_t xw[4] = { v[hit].elem_t, ext, val, e };
                    aom_o_inst(&out, SpvOpCompositeExtract, 5, xw);
                    uint32_t sw[2] = { elem_ids[hit][e], ext };
                    aom_o_inst(&out, SpvOpStore, 3, sw);
                }
                o += wc;
                continue;
            }
        }
        if ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && wc >= 5)
        {
            bool drop = false;
            for (int mi = 0; mi < nmap; mi++) if (map[mi].from == w[2]) { drop = true; break; }
            if (drop) { o += wc; continue; }
            if (wc > 5)
            {
                for (int k = 0; k < nv; k++)
                    if (w[3] == v[k].var)
                    {
                        uint32_t idx = lv_const(m, w[4]);
                        uint32_t nw[64];
                        nw[0] = w[1]; nw[1] = w[2]; nw[2] = elem_ids[k][idx];
                        uint32_t nn = 3;
                        for (uint32_t q = 5; q < wc && nn < 64; q++) nw[nn++] = w[q];
                        aom_o_inst(&out, op, nn + 1, nw);
                        o += wc;
                        drop = true;    /* emitted already */
                        break;
                    }
                if (drop) continue;
            }
        }
        if (op == SpvOpEntryPoint && wc >= 3)
        {
            bool hit = false;
            for (size_t q = 3; q < (size_t)wc; q++)
                for (int k = 0; k < nv; k++) if (w[q] == v[k].var) hit = true;
            if (hit)
            {
                uint32_t nw[640]; int nn = 0;
                nw[nn++] = w[1]; nw[nn++] = w[2];
                for (size_t q = 3; q < (size_t)wc; q++)
                {
                    bool sub = false;
                    for (int k = 0; k < nv; k++) if (w[q] == v[k].var)
                    {
                        for (uint32_t e = 0; e < v[k].count && nn < 600; e++) nw[nn++] = elem_ids[k][e];
                        sub = true;
                    }
                    if (!sub && nn < 630) nw[nn++] = w[q];
                }
                aom_o_inst(&out, op, nn + 1, nw);
                o += wc;
                continue;
            }
        }

        /* The map (access-chain result -> element variable) must be applied to
         * ID operands only: rewriting every word corrupted non-id literals —
         * an OpExecutionMode whose mode literal collided with a mapped id was
         * rewritten into an element-variable id ("Invalid execution mode
         * operand", enhanced_layouts.xfb_stride_dmat2_te). */
        aom_o_push(&out, w[0]);
        bool subst_all_ids = (op == SpvOpPhi || op == SpvOpSelect);
        for (size_t q = 1; q < (size_t)wc; q++)
        {
            bool is_id =
                subst_all_ids ||
                (op == SpvOpLoad && q == 3) ||
                (op == SpvOpStore && q == 1) ||
                (op == SpvOpCopyMemory && (q == 1 || q == 2)) ||
                ((op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) && q == 3);
            uint32_t x = w[q];
            if (is_id)
                for (int mi = 0; mi < nmap; mi++) if (x == map[mi].from) { x = map[mi].to; break; }
            aom_o_push(&out, x);
        }
        o += wc;
    }

    free(m);
    if (out.n <= 5 || !globals_done) { free(out.w); return NULL; }
    out.w[3] = next_id;    /* new id bound */
    if (trace) { FILE *f = fopen("/tmp/aom_out.spv", "wb"); if (f) { fwrite(out.w, 4, out.n, f); fclose(f); } }
    *out_n = out.n;
    return out.w;
}