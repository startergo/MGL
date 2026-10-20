/*
 * geometry.c
 * MGL — geometry shaders: compute-based emulation, streams, layered output
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * Metal has no geometry stage, so a GL program with a geometry shader is run as
 *
 *      1. a rasterization-disabled render pass running the *vertex* stage in a "capture
 *         variant" (the technique of transform_feedback.c: the original GLSL main() is
 *         wrapped and stores every output per (vertex, instance) into an SSBO), followed by
 *      2. a compute pass running a *kernel* that is the geometry shader rewritten by a
 *         source transform: gl_in[] and the user input arrays are loaded from the captured
 *         records of the assembled input primitive, EmitVertex/EmitStreamVertex append the
 *         current outputs to an output vertex buffer with an atomic counter (honouring
 *         max_vertices) and EndPrimitive/EndStreamPrimitive close the strip and append its
 *         raster indices, and finally
 *      3. the ordinary raster pass with a generated pass-through vertex function that reads
 *         the GS output buffer at [[vertex_id]] (gl_Layer -> render_target_array_index,
 *         gl_ViewportIndex -> viewport_array_index) and feeds the program's fragment shader,
 *         drawn with a DrawIndexedPrimitives indirect argument buffer the kernel filled in.
 *
 * The choice of the *source-level* capture variant (rather than SPIRV-Cross's
 * "capture output to buffer" option) is deliberate: Metal's fixed-function vertex fetch
 * then handles every GL attribute format / stride / divisor / base offset, and the captured
 * coordinates stay in GL clip space — the rstate Metal clip-space fixup must only be applied
 * to the last stage before rasterization (here: the pass-through vertex function).
 */

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv_cross_c.h"
#include "spirv.h"

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_geom.h"
#include "mgl_ext_buffers.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_tess.h"     /* the TES-output capture producer (geometry behind tessellation) */
#include "mgl_ext_program.h"
#include "shaders.h"
#include "programs.h"
#include "msl_reserved.h"
#include "hash_table.h"
#include "mgl_ext_rstate.h"

/* shaders.c helpers (declared locally, as transform_feedback.c does) */
void initGLSLInput(GLMContext ctx, GLuint type, const char *src, glslang_input_t *input);
glslang_stage_t getGLSLStage(GLuint type);

#define GEOM()   ((struct MGLExtState_geom *)ctx->state.ext_geom)
#define METAL()  (ctx->mtl_funcs.ext_geom)

/* ------------------------------------------------------------------ helpers */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    assert(p);
    return p;
}

static char *xstrdup(const char *s)
{
    char *d = xmalloc(strlen(s) + 1);
    strcpy(d, s);
    return d;
}

/* growable text buffer */
typedef struct {
    char  *s;
    size_t len, cap;
    bool   oom;
} SB;

static void sb_reserve(SB *b, size_t extra)
{
    if (b->oom) return;
    if (b->len + extra + 1 > b->cap)
    {
        size_t c = b->cap ? b->cap * 2 : 4096;
        while (c < b->len + extra + 1) c *= 2;
        char *n = realloc(b->s, c);
        if (!n) { b->oom = true; return; }
        b->s = n;
        b->cap = c;
    }
}

static void sb_put(SB *b, const char *fmt, ...)
{
    va_list ap;
    int n;
    char tmp[512];

    if (b->oom) return;

    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) { b->oom = true; return; }
    if ((size_t)n < sizeof(tmp))
    {
        sb_reserve(b, (size_t)n);
        if (b->oom) return;
        memcpy(b->s + b->len, tmp, (size_t)n + 1);
        b->len += (size_t)n;
        b->s[b->len] = 0;
        return;
    }
    {
        char *big = xmalloc((size_t)n + 1);
        va_start(ap, fmt);
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        va_end(ap);
        sb_reserve(b, (size_t)n);
        if (!b->oom)
        {
            memcpy(b->s + b->len, big, (size_t)n + 1);
            b->len += (size_t)n;
            b->s[b->len] = 0;
        }
        free(big);
    }
}

static char *sb_take(SB *b)
{
    if (b->oom)
    {
        free(b->s);
        b->s = NULL;
        b->len = b->cap = 0;
        return NULL;
    }
    if (!b->s) return xstrdup("");
    return b->s;
}

static bool mgl_append_log(char **log, const char *fmt, ...)
{
    va_list ap;
    char buf[1024];

    if (!log) return false;

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    size_t len = *log ? strlen(*log) : 0;
    char *n = realloc(*log, len + strlen(buf) + 2);
    if (!n) return false;
    *log = n;
    memcpy(*log + len, buf, strlen(buf) + 1);
    return true;
}

/* ------------------------------------------------------------------ SPIR-V walker
 * Minimal reader for the module structure the emulation needs (types, interface
 * variables, decorations, execution modes).  Kept local so the transform does not
 * depend on reflection details of any backend. */

enum { SPVT_VOID, SPVT_BOOL, SPVT_INT, SPVT_FLOAT, SPVT_VEC, SPVT_MAT, SPVT_ARRAY, SPVT_STRUCT, SPVT_PTR };

typedef struct {
    uint32_t id;
    uint8_t  kind;
    uint32_t elem;          /* element type id for vec / matrix / array / pointer */
    uint32_t n;             /* vector size / matrix columns / array length */
    uint32_t bits;
    bool     is_signed;
    bool     len_is_spec;
    /* OpTypeStruct: the member type ids, in declaration order (an interface block's members) */
    uint32_t *members;
    int      n_members;
} SpvType;

typedef struct {
    uint32_t id;            /* the struct type */
    uint32_t member;        /* the member index */
    uint32_t deco;
    uint32_t value;
} SpvMemberDeco;

typedef struct {
    uint32_t id;            /* the struct type */
    uint32_t member;        /* member index */
    char     name[64];
} SpvMemberName;

/* one resource-binding pin for a generated MSL compile (see geom_build_pins) */
typedef struct { int binding; int msl; bool texture; } GeomPin;

typedef struct {
    uint32_t id;            /* target */
    uint32_t deco;
    uint32_t value;
} SpvDeco;

typedef struct {
    uint32_t id, type_id;
    uint32_t storage;
    bool     in_interface;
} SpvVar;

typedef struct {
    uint32_t id;
    char     name[160];
} SpvName;

/* per-member metadata of OpTypeStruct (interface blocks) */

typedef struct {
    const uint32_t *w;      /* words */
    size_t          n;
    SpvType        *types;
    int             n_types;
    SpvDeco        *decos;
    int             n_decos;
    SpvVar         *vars;
    int             n_vars;
    SpvName        *names;
    int             n_names;
    SpvMemberDeco  *mdecos;        /* OpMemberDecorate of struct types (interface blocks) */
    int             n_mdecos;
    SpvMemberName  *mnames;        /* OpMemberName */
    int             n_mnames;
    struct { uint32_t mode, value; } *modes;
    int             n_modes;
    struct { uint32_t id; uint64_t v; } *consts;
    int             n_consts;
    uint32_t        entry;
    bool            ok;
} SpvModule;

static void spv_push_type(SpvModule *m, const SpvType *t)
{
    m->types = realloc(m->types, sizeof(SpvType) * (size_t)(m->n_types + 1));
    m->types[m->n_types++] = *t;
}

static uint32_t spv_read_word(const uint32_t *in, size_t n, size_t pos)
{
    (void)n;
    return in[pos];
}

static void spv_get_string(const uint32_t *in, size_t wc, size_t pos, char *out, size_t out_len)
{
    size_t o = 0;
    out[0] = 0;
    for (size_t k = 0; k < wc; k++)
    {
        const char *p = (const char *)&in[pos + k];
        for (int b = 0; b < 4; b++)
        {
            char c = p[b];
            if (!c) { if (o < out_len) out[o] = 0; return; }
            if (o + 1 < out_len) out[o++] = c;
        }
    }
    out[o < out_len ? o : out_len - 1] = 0;
}

static SpvType *spv_find_type(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_types; i++)
        if (m->types[i].id == id) return &m->types[i];
    return NULL;
}

static uint64_t spv_const_value(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_consts; i++)
        if (m->consts[i].id == id) return m->consts[i].v;
    return 1;
}

static bool spv_deco(SpvModule *m, uint32_t id, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == id && m->decos[i].deco == deco)
        {
            if (value) *value = m->decos[i].value;
            return true;
        }
    return false;
}

static const char *spv_name(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_names; i++)
        if (m->names[i].id == id) return m->names[i].name;
    return NULL;
}

static int spv_binding(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == id && m->decos[i].deco == SpvDecorationBinding)
            return (int)m->decos[i].value;
    return -1;
}

static SpvVar *spv_find_var(SpvModule *m, uint32_t id) __attribute__((unused));
static SpvVar *spv_find_var(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_vars; i++)
        if (m->vars[i].id == id) return &m->vars[i];
    return NULL;
}

static bool spv_member_deco(SpvModule *m, uint32_t type_id, uint32_t member, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_mdecos; i++)
        if (m->mdecos[i].id == type_id && m->mdecos[i].member == member && m->mdecos[i].deco == deco)
        {
            if (value) *value = m->mdecos[i].value;
            return true;
        }
    return false;
}

static const char *spv_member_name(SpvModule *m, uint32_t type_id, uint32_t member)
{
    for (int i = 0; i < m->n_mnames; i++)
        if (m->mnames[i].id == type_id && m->mnames[i].member == member)
            return m->mnames[i].name;
    return NULL;
}

/* parse a SPIR-V module; returns NULL and sets *err on failure */
static SpvModule *spv_parse(const uint32_t *words, size_t nwords, const char **err)
{
    SpvModule *m = xmalloc(sizeof(SpvModule));
    memset(m, 0, sizeof(*m));

    if (nwords < 5 || words[0] != SpvMagicNumber)
    {
        *err = "invalid SPIR-V magic";
        free(m);
        return NULL;
    }
    m->w = words;
    m->n = nwords;

    size_t pos = 5;
    while (pos < nwords)
    {
        uint32_t first = spv_read_word(words, nwords, pos);
        uint32_t wc = first >> 16;
        uint32_t op = first & 0xffff;

        if (wc == 0 || pos + wc > nwords)
        {
            *err = "truncated SPIR-V instruction";
            break;
        }

        switch (op)
        {
            case SpvOpName:
            {
                char name[160];
                spv_get_string(words, wc - 2, pos + 2, name, sizeof(name));
                m->names = realloc(m->names, sizeof(SpvName) * (size_t)(m->n_names + 1));
                m->names[m->n_names].id = words[pos + 1];
                snprintf(m->names[m->n_names].name, sizeof(m->names[m->n_names].name), "%s", name);
                m->n_names++;
                break;
            }
            case SpvOpEntryPoint:
                m->entry = words[pos + 2];
                break;
            case SpvOpExecutionMode:
                m->modes = realloc(m->modes, sizeof(*m->modes) * (size_t)(m->n_modes + 1));
                m->modes[m->n_modes].mode = words[pos + 2];
                m->modes[m->n_modes].value = wc > 3 ? words[pos + 3] : 0;
                m->n_modes++;
                break;
            case SpvOpTypeVoid:
            case SpvOpTypeBool:
            case SpvOpTypeInt:
            case SpvOpTypeFloat:
            {
                SpvType t;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                if (op == SpvOpTypeVoid) t.kind = SPVT_VOID;
                else if (op == SpvOpTypeBool) { t.kind = SPVT_BOOL; t.bits = 32; }
                else if (op == SpvOpTypeInt) { t.kind = SPVT_INT; t.bits = words[pos + 2]; t.is_signed = words[pos + 3] != 0; }
                else { t.kind = SPVT_FLOAT; t.bits = words[pos + 2]; }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpTypeVector:
            case SpvOpTypeMatrix:
            case SpvOpTypeArray:
            case SpvOpTypeRuntimeArray:
            case SpvOpTypePointer:
            {
                SpvType t;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                if (op == SpvOpTypePointer)
                {
                    /* OpTypePointer: result, storage class, pointee type */
                    t.kind = SPVT_PTR;
                    t.elem = words[pos + 3];
                    t.n = words[pos + 2];
                }
                else
                {
                    t.elem = words[pos + 2];
                    if (op == SpvOpTypeVector) t.kind = SPVT_VEC;
                    else if (op == SpvOpTypeMatrix) t.kind = SPVT_MAT;
                    else
                    {
                        t.kind = SPVT_ARRAY;
                        if (op == SpvOpTypeRuntimeArray) { t.n = 1; t.len_is_spec = true; }
                        else { t.n = words[pos + 3]; t.len_is_spec = true; }
                    }
                    if (op == SpvOpTypeVector || op == SpvOpTypeMatrix) t.n = words[pos + 3];
                }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpTypeStruct:
            {
                /* the member type ids, in declaration order (an interface block) */
                SpvType t;
                int nm = (int)wc - 2;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                t.kind = SPVT_STRUCT;
                t.n_members = nm > 0 ? nm : 0;
                if (t.n_members)
                {
                    t.members = xmalloc(sizeof(uint32_t) * (size_t)t.n_members);
                    for (int k = 0; k < t.n_members; k++)
                        t.members[k] = words[pos + 2 + k];
                }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpMemberName:
            {
                char name[64];
                spv_get_string(words, wc - 3, pos + 3, name, sizeof(name));
                m->mnames = realloc(m->mnames, sizeof(SpvMemberName) * (size_t)(m->n_mnames + 1));
                m->mnames[m->n_mnames].id = words[pos + 1];
                m->mnames[m->n_mnames].member = words[pos + 2];
                snprintf(m->mnames[m->n_mnames].name, sizeof(m->mnames[m->n_mnames].name), "%s", name);
                m->n_mnames++;
                break;
            }
            case SpvOpMemberDecorate:
                m->mdecos = realloc(m->mdecos, sizeof(SpvMemberDeco) * (size_t)(m->n_mdecos + 1));
                m->mdecos[m->n_mdecos].id = words[pos + 1];
                m->mdecos[m->n_mdecos].member = words[pos + 2];
                m->mdecos[m->n_mdecos].deco = words[pos + 3];
                m->mdecos[m->n_mdecos].value = wc > 4 ? words[pos + 4] : 0;
                m->n_mdecos++;
                break;
            case SpvOpConstant:
                m->consts = realloc(m->consts, sizeof(*m->consts) * (size_t)(m->n_consts + 1));
                m->consts[m->n_consts].id = words[pos + 2];
                m->consts[m->n_consts].v = wc > 4 ? ((uint64_t)words[pos + 4] << 32) | words[pos + 3] : words[pos + 3];
                m->n_consts++;
                break;
            case SpvOpDecorate:
                m->decos = realloc(m->decos, sizeof(SpvDeco) * (size_t)(m->n_decos + 1));
                m->decos[m->n_decos].id = words[pos + 1];
                m->decos[m->n_decos].deco = words[pos + 2];
                m->decos[m->n_decos].value = wc > 3 ? words[pos + 3] : 0;
                m->n_decos++;
                break;
            case SpvOpVariable:
            {
                m->vars = realloc(m->vars, sizeof(SpvVar) * (size_t)(m->n_vars + 1));
                m->vars[m->n_vars].id = words[pos + 2];
                m->vars[m->n_vars].type_id = words[pos + 1];
                m->vars[m->n_vars].storage = wc > 3 ? words[pos + 3] : SpvStorageClassFunction;
                m->vars[m->n_vars].in_interface = false;
                m->n_vars++;
                break;
            }
            default:
                break;
        }

        pos += wc;
    }

    /* resolve array lengths given as OpConstant ids */
    for (int i = 0; i < m->n_types; i++)
        if (m->types[i].kind == SPVT_ARRAY && m->types[i].len_is_spec)
        {
            uint64_t v = spv_const_value(m, m->types[i].n);
            if (v && v < 0x10000) m->types[i].n = (uint32_t)v;
            m->types[i].len_is_spec = false;
        }

    m->ok = true;
    return m;
}

static void spv_free(SpvModule *m)
{
    if (!m) return;
    free(m->types);
    free(m->decos);
    free(m->vars);
    free(m->names);
    for (int i = 0; i < m->n_types; i++) free(m->types[i].members);
    free(m->mdecos);
    free(m->mnames);
    free(m->modes);
    free(m->consts);
    free(m);
}

/* strip pointer / array layers off a variable type */
static SpvType *spv_value_type(SpvModule *m, uint32_t type_id, uint32_t *array_elems)
{
    SpvType *t = spv_find_type(m, type_id);
    if (array_elems) *array_elems = 1;
    if (!t) return NULL;
    while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
    if (!t) return NULL;
    if (t->kind == SPVT_ARRAY)
    {
        uint32_t elems = t->n ? t->n : 1;
        t = spv_find_type(m, t->elem);
        while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
        if (!t) return NULL;
        if (array_elems) *array_elems = elems;
    }
    return t;
}

/* the scalar component type of a (possibly vector / matrix) type: a plain SPVT_VEC's kind
 * says nothing about float vs int — the component type is its element type.  Returning
 * GL_FLOAT for an ivec4 would make the kernel load it with uintBitsToFloat and the
 * generated assignment fail to compile ("cannot convert vec4 to ivec4"). */
static GLenum spv_scalar_base(SpvModule *m, const SpvType *t)
{
    const SpvType *s = t;
    if (!s) return GL_FLOAT;
    if (s->kind == SPVT_VEC || s->kind == SPVT_MAT) s = spv_find_type(m, s->elem);
    if (!s) return GL_FLOAT;
    if (s->kind == SPVT_FLOAT) return s->bits == 64 ? GL_DOUBLE : GL_FLOAT;
    if (s->kind == SPVT_INT) return s->is_signed ? GL_INT : GL_UNSIGNED_INT;
    if (s->kind == SPVT_BOOL) return GL_BOOL;
    return GL_FLOAT;
}

/* ------------------------------------------------------------------ varying layout */

static const char *glsl_type_name(GLenum base, GLuint rows, GLuint cols)
{
    static const char *scalar[5] = { "float", "double", "int", "uint", "bool" };
    static char buf[64];

    int k = base == GL_DOUBLE ? 1 : base == GL_INT ? 2 : base == GL_UNSIGNED_INT ? 3 : base == GL_BOOL ? 4 : 0;

    if (cols > 1)
    {
        if (cols == rows) snprintf(buf, sizeof(buf), "mat%d", cols);
        else snprintf(buf, sizeof(buf), "mat%dx%d", cols, rows);
        return buf;
    }
    if (rows == 1) return scalar[k];

    snprintf(buf, sizeof(buf), "%svec%d", base == GL_DOUBLE ? "d" : base == GL_INT ? "i" :
                                        base == GL_UNSIGNED_INT ? "u" : base == GL_BOOL ? "b" : "", rows);
    return buf;
}

static GLuint base_words(GLenum base)
{
    return base == GL_DOUBLE ? 2 : 1;
}

/* GLSL conversion applied when a value is stored into the word record */
static const char *word_store(GLenum base, const char *e)
{
    static char buf[256];
    if (base == GL_FLOAT) snprintf(buf, sizeof(buf), "floatBitsToUint(%s)", e);
    else if (base == GL_DOUBLE) snprintf(buf, sizeof(buf), "unpackDouble2x32(%s)", e);
    else snprintf(buf, sizeof(buf), "uint(%s)", e);
    return buf;
}

/* number of words a component of `base` occupies */
static GLuint comp_words(GLenum base) { return base == GL_DOUBLE ? 2 : 1; }

/* emit "name = <value>;" statements reconstructing v from the word buffer.
 * open/close delimit the word access: <open> + " + N" + <close>.  `index` is the vertex
 * index expression for geometry-stage inputs (which are arrays of the record per vertex). */
/* the access path of a varying inside the rewritten shader: a named interface block
 * member is reached through its instance ("b.c"); an anonymous block member (the
 * reflection records the member name itself) and a plain varying use the bare name */
static const char *geom_var_expr(char *buf, size_t cap, const MGLGeomVar *v)
{
    if (v->member[0] && strcmp(v->name, v->member))
        snprintf(buf, cap, "%s.%s", v->name, v->member);
    else
        snprintf(buf, cap, "%s", v->name);
    return buf;
}

static void gen_varying_load(SB *b, const MGLGeomVar *v, const char *open, const char *close,
                             const char *index)
{
    char prefix[192];

    if (index)
        snprintf(prefix, sizeof(prefix), "%s[%s]%s%s", v->name, index,
                 v->member[0] ? "." : "", v->member);
    else
        snprintf(prefix, sizeof(prefix), "%s%s%s", v->name, v->member[0] ? "." : "", v->member);

    for (GLuint e = 0; e < v->elems; e++)
    {
        char elem_prefix[64] = "";
        if (v->elems > 1) snprintf(elem_prefix, sizeof(elem_prefix), "[%u]", e);

        if (v->cols > 1)
        {
            /* matrix: one vecN per column */
            char mname[64];
            snprintf(mname, sizeof(mname), "%s", glsl_type_name(v->base, v->rows, v->cols));
            sb_put(b, "    %s%s = %s(", prefix, elem_prefix, mname);
            for (GLuint c = 0; c < v->cols; c++)
            {
                if (c) sb_put(b, ", ");
                sb_put(b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r = 0; r < v->rows; r++)
                {
                    GLuint w = v->offset + (e * v->cols * v->rows + c * v->rows + r) * comp_words(v->base);
                    if (r) sb_put(b, ", ");
                    if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                    else if (v->base == GL_DOUBLE) sb_put(b, "uint64BitsToDouble(uvec2(%s + %u%s, %s + %u%s))",
                                                          open, w, close, open, w + 1, close);
                    else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                    else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
                }
                sb_put(b, ")");
            }
            sb_put(b, ");\n");
        }
        else
        {
            sb_put(b, "    %s%s = ", prefix, elem_prefix);
            if (v->rows == 1 && v->base != GL_DOUBLE)
            {
                GLuint w = v->offset + e * v->rows * comp_words(v->base);
                if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
            }
            else
            {
                sb_put(b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r = 0; r < v->rows; r++)
                {
                    GLuint w = v->offset + (e * v->rows + r) * comp_words(v->base);
                    if (r) sb_put(b, ", ");
                    if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                    else if (v->base == GL_DOUBLE) sb_put(b, "uint64BitsToDouble(uvec2(%s + %u%s, %s + %u%s))",
                                                          open, w, close, open, w + 1, close);
                    else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                    else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
                }
                sb_put(b, ")");
            }
            sb_put(b, ";\n");
        }
    }
}

/* emit "dest[idx + off] = word;" statements storing v from the producer expression */
static void gen_varying_store(SB *b, const MGLGeomVar *v, const char *dest, const char *idx,
                              const char *src)
{
    static const char *comp[4] = { ".x", ".y", ".z", ".w" };

    for (GLuint e = 0; e < v->elems; e++)
    {
        char ep[32] = "";
        if (v->elems > 1) snprintf(ep, sizeof(ep), "[%u]", e);

        for (GLuint c = 0; c < v->cols; c++)
        {
            char cp[32] = "";
            if (v->cols > 1) snprintf(cp, sizeof(cp), "[%u]", c);

            for (GLuint r = 0; r < v->rows; r++)
            {
                char rp[8] = "";
                if (v->rows > 1) snprintf(rp, sizeof(rp), "%s", comp[r]);

                char expr[256];
                if (!src)
                    snprintf(expr, sizeof(expr), "0u");     /* undefined input: zeros */
                else
                    snprintf(expr, sizeof(expr), "%s%s%s%s", src, ep, cp, rp);

                GLuint w = v->offset + (e * v->cols * v->rows + c * v->rows + r) * comp_words(v->base);
                if (v->base == GL_DOUBLE)
                {
                    sb_put(b, "    { uvec2 mgl_w = unpackDouble2x32(%s); %s[%s + %u] = mgl_w.x; %s[%s + %u] = mgl_w.y; }\n",
                           expr, dest, idx, w, dest, idx, w + 1);
                }
                else
                {
                    sb_put(b, "    %s[%s + %u] = %s;\n", dest, idx, w, word_store(v->base, expr));
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ program table */

static void geom_free_prog(GLMContext ctx, MGLGeomProgram *gp);

static MGLGeomProgram *geom_prog_slot(GLMContext ctx, Program *p, bool create)
{
    struct MGLExtState_geom *g = GEOM();

    if (!g || !p || !p->name) return NULL;

    /* run_link builds every executable in a scratch object, so the Program pointer differs
     * between the compile hook and the post-link hook: fall back to the name */
    for (int i = 0; i < g->n_progs; i++)
    {
        MGLGeomProgram *gp = &g->progs[i];
        if (gp->name == p->name) return gp;
    }
    if (!create) return NULL;

    /* re-use slots whose program is gone (deleted, or a new object under the name) */
    for (int i = 0; i < g->n_progs; i++)
    {
        MGLGeomProgram *gp = &g->progs[i];
        if (gp->name && getProgram(ctx, gp->name)) continue;
        if (!gp->name) geom_free_prog(ctx, gp);
        memset(gp, 0, sizeof(*gp));
        gp->name = p->name;
        gp->program = getProgram(ctx, p->name);
        return gp;
    }

    /* no free slot: grow the table */
    if (g->n_progs >= g->progs_cap)
    {
        int cap = g->progs_cap ? g->progs_cap * 2 : MGL_GEOM_MAX_PROGRAMS;
        MGLGeomProgram *grown = realloc(g->progs, (size_t)cap * sizeof(*grown));
        if (!grown) return NULL;
        memset(grown + g->n_progs, 0, (size_t)(cap - g->n_progs) * sizeof(*grown));
        g->progs = grown;
        g->progs_cap = cap;
    }
    {
        MGLGeomProgram *gp = &g->progs[g->n_progs++];
        gp->name = p->name;
        gp->program = getProgram(ctx, p->name);
        return gp;
    }
}

static void geom_free_prog(GLMContext ctx, MGLGeomProgram *gp)
{
    if (gp->capture_function && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_function);
    if (gp->capture_library && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_library);
    if (gp->draw_function && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_function);
    if (gp->draw_library && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_library);
    free(gp->kernel_msl);
    free(gp->kernel_entry);
    free(gp->final_entry);
    free(gp->capture_msl);
    free(gp->capture_entry);
    free(gp->draw_msl);
    free(gp->draw_entry);
    free(gp->fail);
    memset(gp, 0, sizeof(*gp));
}

static void geom_fail(GLMContext ctx, MGLGeomProgram *gp, const char *fmt, ...)
{
    va_list ap;
    char buf[512];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    free(gp->fail);
    gp->fail = xstrdup(buf);
    gp->linked = false;
}

/* ------------------------------------------------------------------ glslang / SPIRV-Cross wrappers */

static bool geom_glsl_to_spirv(GLMContext ctx, GLuint glm_type, const char *src, int force_version,
                               uint32_t **ir, size_t *words, char **log)
{
    glslang_input_t input;
    glslang_shader_t *shader;
    glslang_program_t *prog;
    const char *msgs;
    int version = 330, options = GLSLANG_SHADER_VULKAN_RULES_RELAXED;

    memset(&input, 0, sizeof(input));
    initGLSLInput(ctx, glm_type, src, &input);
    if (force_version && input.default_version < force_version)
    {
        input.default_version = force_version;
        input.client_version = GLSLANG_TARGET_OPENGL_450;
    }

    {
        const char *vstr = strstr(src, "#version");
        if (vstr) sscanf(vstr, "#version %d", &version);
    }
    /* match the program's own compile (shaders.c mglCompileShader): GL assigns interface
     * locations and bindings at link time, so a 330+ shader whose outputs/inputs have no
     * layout(location) (legal GL, e.g. the CTS's "out uint vs_gs_index;") must compile here
     * too.  The invented locations never reach the record layout — the capture variant
     * stores outputs explicitly and the kernel reads them back by record offset — and the
     * invented bindings are re-pinned below (geom_spirv_to_msl), so auto-mapping is inert. */
    options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS | GLSLANG_SHADER_AUTO_MAP_BINDINGS;

    shader = glslang_shader_create(&input);
    if (!shader) { mgl_append_log(log, "ERROR: geometry: glslang shader creation failed\n"); return false; }
    glslang_shader_set_options(shader, options);

    if (!glslang_shader_preprocess(shader, &input) ||
        !glslang_shader_parse(shader, &input))
    {
        const char *info = glslang_shader_get_info_log(shader);
        const char *debug = glslang_shader_get_info_debug_log(shader);
        mgl_append_log(log, "ERROR: geometry: GLSL → SPIR-V failed\n%s\n%s", info ? info : "", debug ? debug : "");
        glslang_shader_delete(shader);
        return false;
    }

    prog = glslang_program_create();
    if (!prog) { glslang_shader_delete(shader); return false; }
    glslang_program_add_shader(prog, shader);

    if (!glslang_program_link(prog, GLSLANG_MSG_DEFAULT_BIT))
    {
        const char *info = glslang_program_get_info_log(prog);
        mgl_append_log(log, "ERROR: geometry: glslang program link failed\n%s", info ? info : "");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    /* run the I/O mapper (as program.c's link does): it is what actually assigns the
     * locations AUTO_MAP_LOCATIONS only promises — without it a shader whose in/out have
     * no layout(location) compiles but SPIRV-Cross then emits a stage_in struct whose
     * members carry no [[attribute(n)]], which Metal rejects */
    if (!glslang_program_map_io(prog))
    {
        const char *info = glslang_program_get_info_log(prog);
        mgl_append_log(log, "ERROR: geometry: glslang I/O mapping failed\n%s", info ? info : "");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    glslang_program_SPIRV_generate(prog, getGLSLStage(glm_type));

    msgs = glslang_program_SPIRV_get_messages(prog);
    if (msgs && *msgs) mgl_append_log(log, "%s\n", msgs);

    size_t n = glslang_program_SPIRV_get_size(prog);
    if (!n)
    {
        mgl_append_log(log, "ERROR: geometry: empty SPIR-V module\n");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    uint32_t *out = xmalloc(n * sizeof(uint32_t));
    glslang_program_SPIRV_get(prog, out);
    *ir = out;
    *words = n;

    glslang_program_delete(prog);
    glslang_shader_delete(shader);
    return true;
}

/* decompile SPIR-V back to GLSL (used when no source is available: ShaderBinary / program
 * binary) so that the same source transform can be applied */
static char *geom_spirv_to_glsl(const uint32_t *ir, size_t words, int version, char **log)
{
    spvc_context ctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_compiler_options opt;
    spvc_result r;
    const char *src = NULL;

    r = spvc_context_create(&ctx);
    if (r != SPVC_SUCCESS) return NULL;

    r = spvc_context_parse_spirv(ctx, ir, words, &pir);
    if (r != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V parse failed (%s)\n", spvc_context_get_last_error_string(ctx));
        spvc_context_destroy(ctx);
        return NULL;
    }

    r = spvc_context_create_compiler(ctx, SPVC_BACKEND_GLSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp);
    if (r != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → GLSL compiler creation failed\n");
        spvc_context_destroy(ctx);
        return NULL;
    }

    r = spvc_compiler_create_compiler_options(comp, &opt);
    if (r == SPVC_SUCCESS)
    {
        spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_GLSL_VERSION, (unsigned)version);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_EMIT_LINE_DIRECTIVES, SPVC_FALSE);
        spvc_compiler_install_compiler_options(comp, opt);
    }

    r = spvc_compiler_compile(comp, &src);
    if (r != SPVC_SUCCESS || !src)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → GLSL failed (%s)\n",
                       spvc_context_get_last_error_string(ctx));
        spvc_context_destroy(ctx);
        return NULL;
    }

    char *out = xstrdup(src);
    spvc_context_destroy(ctx);
    return out;
}

/* translate SPIR-V to MSL with the given buffer pins; returns the MSL source */
/* record the geometry stage's sampler/image array sizes: the kernel's Metal argument for
 * array<texture2d, N> spans N slots, so the binder needs the element count per resource */
static void geom_record_tex_arrays(const uint32_t *ir, size_t words, MGLGeomProgram *gp)
{
    spvc_context sctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_resources res;
    const spvc_reflected_resource *list = NULL;
    size_t count = 0;
    int types[2] = { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, SPVC_RESOURCE_TYPE_STORAGE_IMAGE };

    gp->n_tex_arrays = 0;
    if (spvc_context_create(&sctx) != SPVC_SUCCESS) return;
    if (spvc_context_parse_spirv(sctx, ir, words, &pir) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }
    if (spvc_context_create_compiler(sctx, SPVC_BACKEND_MSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }
    if (spvc_compiler_create_shader_resources(comp, &res) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }

    for (int t = 0; t < 2; t++)
    {
        const char *tn = NULL;
        size_t n = 0;

        if (types[t] == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE)
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, &list, &n);
        else
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_STORAGE_IMAGE, &list, &n);
        (void)tn; (void)count;
        for (size_t i = 0; i < n && gp->n_tex_arrays < 8; i++)
        {
            spvc_type ty = spvc_compiler_get_type_handle(comp, list[i].type_id);
            unsigned dims = spvc_type_get_num_array_dimensions(ty);

            if (!dims) continue;
            unsigned asz = (unsigned)spvc_type_get_array_dimension(ty, 0);
            if (!asz) continue;
            snprintf(gp->tex_arrays[gp->n_tex_arrays].name, sizeof gp->tex_arrays[0].name,
                     "%s", list[i].name);
            gp->tex_arrays[gp->n_tex_arrays].n = (int)asz;
            gp->n_tex_arrays++;
        }
    }
    spvc_context_destroy(sctx);
}

static char *geom_spirv_to_msl(GLMContext ctx, GLuint glm_type, const uint32_t *ir, size_t words,
                               const char *entry, bool disable_rasterization, char **log,
                               int pin_stage, const int *pins_glsl, const int *pins_msl, int pin_count,
                               const GeomPin *pins2, int npins2)
{
    spvc_context sctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_compiler_options opt;
    spvc_msl_resource_binding binding;
    spvc_result r;
    const char *msl = NULL;

    r = spvc_context_create(&sctx);
    if (r != SPVC_SUCCESS) return NULL;

    if (spvc_context_parse_spirv(sctx, ir, words, &pir) != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V parse failed (%s)\n", spvc_context_get_last_error_string(sctx));
        spvc_context_destroy(sctx);
        return NULL;
    }

    if (spvc_context_create_compiler(sctx, SPVC_BACKEND_MSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: MSL compiler creation failed\n");
        spvc_context_destroy(sctx);
        return NULL;
    }

    if (spvc_compiler_create_compiler_options(comp, &opt) == SPVC_SUCCESS)
    {
        spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_MSL_VERSION, 30000);
        if (disable_rasterization)
            spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION, SPVC_TRUE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, SPVC_FALSE);
        spvc_compiler_install_compiler_options(comp, opt);
    }

    spvc_compiler_msl_add_discrete_descriptor_set(comp, 3);

    if (pin_stage >= 0)
    {
        for (int i = 0; i < pin_count; i++)
        {
            memset(&binding, 0, sizeof(binding));
            binding.stage = (SpvExecutionModel)pin_stage;
            binding.desc_set = 0;
            binding.binding = (unsigned)pins_glsl[i];
            binding.msl_buffer = (unsigned)pins_msl[i];
            binding.msl_texture = 0;
            binding.msl_sampler = 0;
            if (spvc_compiler_msl_add_resource_binding(comp, &binding) != SPVC_SUCCESS)
            {
                mgl_append_log(log, "ERROR: geometry: could not pin MSL buffer %d\n", pins_msl[i]);
                spvc_context_destroy(sctx);
                return NULL;
            }
        }

        /* user resources of the stage (uniform / storage buffers, samplers): pinned to the
         * flat positions of the renderer's buffer map, so the buffer map built for the
         * program's own stage binds them verbatim for the emulation shaders too (see
         * geom_build_pins) */
        for (int i = 0; i < npins2; i++)
        {
            memset(&binding, 0, sizeof(binding));
            binding.stage = (SpvExecutionModel)pin_stage;
            binding.desc_set = 0;
            binding.binding = (unsigned)pins2[i].binding;
            if (pins2[i].texture)
            {
                binding.msl_texture = (unsigned)pins2[i].msl;
                binding.msl_sampler = (unsigned)pins2[i].msl;
            }
            else
            {
                binding.msl_buffer = (unsigned)pins2[i].msl;
            }
            if (spvc_compiler_msl_add_resource_binding(comp, &binding) != SPVC_SUCCESS)
            {
                mgl_append_log(log, "ERROR: geometry: could not pin MSL resource %d\n", pins2[i].binding);
                spvc_context_destroy(sctx);
                return NULL;
            }
        }
    }

    (void)glm_type;

    /* GLSL identifiers that collide with MSL keywords/types must be renamed (CTS shaders
     * declare e.g. `uniform sampler2D sampler;`); see msl_reserved.h. */
    mgl_msl_mangle_reserved(comp, ir, words);

    if (spvc_compiler_compile(comp, &msl) != SPVC_SUCCESS || !msl)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → MSL failed (%s)\n", spvc_context_get_last_error_string(sctx));
        spvc_context_destroy(sctx);
        return NULL;
    }

    char *out = xstrdup(msl);
    spvc_context_destroy(sctx);
    return out;
}

/* ------------------------------------------------------------------ source transform
 * Rewrites a geometry shader into the body of a compute shader:
 *   - global `in` / `out` declarations lose their storage qualifier (they become plain
 *     globals), input arrays get the input primitive size, and gl_PerVertex block
 *     declarations are dropped (replaced by the emulation's own `mgl_in[]`)
 *   - reserved identifiers are renamed (gl_Position, gl_in, EmitVertex, ...) so the
 *     transformed source can be wrapped by the emulation prelude
 */

typedef struct {
    const char *from, *to;
} Rename;

static const Rename geom_renames[] = {
    { "gl_PrimitiveIDIn",     "mgl_gs_prim_in" },
    { "gl_InvocationID",      "mgl_gs_invoc" },
    { "gl_ViewportIndex",     "mgl_gs_vp" },
    { "gl_PrimitiveID",       "mgl_gs_prim_out" },
    { "gl_ClipDistance",      "mgl_gs_clip" },
    { "gl_CullDistance",      "mgl_gs_cull" },
    { "gl_PointSize",         "mgl_gs_psize" },
    { "gl_Position",          "mgl_gs_pos" },
    { "gl_Layer",             "mgl_gs_layer" },
    { "gl_in",                "mgl_in" },
    { "EmitStreamVertex",     "mgl_gs_emit_stream_vertex" },
    { "EndStreamPrimitive",   "mgl_gs_end_stream_primitive" },
    { "EmitVertex",           "mgl_gs_emit_vertex" },
    { "EndPrimitive",         "mgl_gs_end_primitive" },
    { NULL, NULL }
};

static bool is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static size_t read_ident(const char *s, char *out, size_t cap)
{
    size_t n = 0;
    while (s[n] && is_ident_char(s[n]))
    {
        if (n + 1 < cap) out[n] = s[n];
        n++;
    }
    out[n < cap ? n : cap - 1] = 0;
    return n;
}

static const char *rename_ident(const char *id)
{
    for (int i = 0; geom_renames[i].from; i++)
        if (!strcmp(id, geom_renames[i].from)) return geom_renames[i].to;
    return NULL;
}

/* qualifiers that may precede in/out in a GLSL interface declaration */
static bool is_interface_qualifier(const char *w)
{
    return !strcmp(w, "layout") || !strcmp(w, "flat") || !strcmp(w, "smooth") ||
           !strcmp(w, "noperspective") || !strcmp(w, "centroid") || !strcmp(w, "sample") ||
           !strcmp(w, "precise") || !strcmp(w, "invariant") || !strcmp(w, "in") || !strcmp(w, "out");
}

/* skip whitespace and comments, return the new position */
static size_t skip_blank(const char *s, size_t i, size_t len)
{
    while (i < len)
    {
        if (isspace((unsigned char)s[i])) { i++; continue; }
        if (s[i] == '/' && i + 1 < len && s[i + 1] == '/')
        {
            while (i < len && s[i] != '\n') i++;
            continue;
        }
        if (s[i] == '/' && i + 1 < len && s[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(s[i] == '*' && s[i + 1] == '/')) i++;
            i += 2;
            continue;
        }
        break;
    }
    return i;
}

enum { STMT_OTHER = 0, STMT_IFACE_IN, STMT_IFACE_OUT };

typedef struct {
    int     kind;
    bool    is_struct;      /* leading token is `struct` */
    size_t  qual_end;       /* offset (from the statement start) after the in/out token */
} StmtInfo;

/* classify the top-level statement beginning at s[pos] (only leading tokens are read) */
static void classify_statement(const char *s, size_t pos, size_t len, StmtInfo *info)
{
    size_t i = pos;
    bool iface = false;

    memset(info, 0, sizeof(*info));

    while (i < len)
    {
        i = skip_blank(s, i, len);
        if (i >= len) break;

        if (is_ident_char(s[i]) && !isdigit((unsigned char)s[i]))
        {
            char id[64];
            size_t n = read_ident(s + i, id, sizeof(id));

            if (!strcmp(id, "in") || !strcmp(id, "out"))
            {
                iface = true;
                info->kind = !strcmp(id, "in") ? STMT_IFACE_IN : STMT_IFACE_OUT;
                info->qual_end = i + n - pos;
                i += n;
                continue;
            }
            if (!iface && is_interface_qualifier(id))
            {
                i += n;
                if (!strcmp(id, "layout"))
                {
                    while (i < len && s[i] != '(') i++;
                    if (i < len)
                    {
                        int d = 0;
                        do {
                            if (s[i] == '(') d++;
                            else if (s[i] == ')') d--;
                            i++;
                        } while (i < len && d > 0);
                    }
                }
                continue;
            }
            if (!iface && !strcmp(id, "struct")) info->is_struct = true;
            break;
        }
        break;
    }
}

/* end (one past) of the top-level statement beginning at pos: the ';' that follows any
 * balanced braces */
static size_t statement_end(const char *s, size_t pos, size_t len)
{
    size_t i = pos;
    int brace = 0;

    while (i < len)
    {
        char c = s[i];
        if (c == '/' && i + 1 < len && s[i + 1] == '/')
        {
            while (i < len && s[i] != '\n') i++;
            continue;
        }
        if (c == '/' && i + 1 < len && s[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(s[i] == '*' && s[i + 1] == '/')) i++;
            i += 2;
            continue;
        }
        if (c == '{') brace++;
        else if (c == '}') brace--;
        else if (c == ';' && brace == 0) return i + 1;
        i++;
    }
    return len;
}

/* append the identifier (renamed when it is one of the reserved names) */
static void emit_ident(SB *out, const char *id, bool *used_gl_in)
{
    const char *r = rename_ident(id);
    sb_put(out, "%s", r ? r : id);
    if (used_gl_in && r && !strcmp(id, "gl_in")) *used_gl_in = true;
}

/* copy a run of source text applying the identifier renames; stops (without consuming) at
 * one of stop_chars or at EOF.  Braces inside the copied run are counted into *depth. */
static void copy_renamed(SB *out, const char *s, size_t *pos, size_t len, const char *stop_chars,
                         int *brace_delta, bool *used_gl_in)
{
    while (*pos < len)
    {
        char c = s[*pos];

        if (stop_chars && strchr(stop_chars, c)) return;

        if (c == '/' && *pos + 1 < len && s[*pos + 1] == '/')
        {
            while (*pos < len && s[*pos] != '\n') { sb_put(out, "%c", s[*pos]); (*pos)++; }
            continue;
        }
        if (c == '/' && *pos + 1 < len && s[*pos + 1] == '*')
        {
            sb_put(out, "/*");
            *pos += 2;
            while (*pos + 1 < len && !(s[*pos] == '*' && s[*pos + 1] == '/')) { sb_put(out, "%c", s[*pos]); (*pos)++; }
            if (*pos + 1 < len) { sb_put(out, "*/"); *pos += 2; }
            continue;
        }

        if (is_ident_char(c) && !isdigit((unsigned char)c))
        {
            char id[128];
            size_t n = read_ident(s + *pos, id, sizeof(id));
            emit_ident(out, id, used_gl_in);
            *pos += n;
            continue;
        }

        if (c == '{' && brace_delta) (*brace_delta)++;
        if (c == '}' && brace_delta) (*brace_delta)--;
        sb_put(out, "%c", c);
        (*pos)++;
    }
}

/* emit the rewritten form of an interface declaration [pos, end) */
/* strip interface-only qualifiers from interface-block member declarations: a layout(...)
 * group and the interpolation/storage keywords are legal on an interface member but not on
 * the plain struct member the rewritten shader declares (locations are tracked separately
 * by the reflection, so nothing is lost) */
/* strip interface-only member qualifiers; `storage_too` also drops in/out — needed when
 * the members end up inside a plain struct (GLSL forbids storage qualifiers on struct
 * members), while re-declared-as-ordinary-declaration paths keep them */
static char *strip_member_quals2(const char *s, size_t n, bool storage_too)
{
    static const char *drop[] = { "flat", "smooth", "noperspective", "centroid", "sample",
                                  "invariant", "precise", NULL };
    static const char *storage[] = { "in", "out", NULL };
    SB out;
    size_t i = 0;

    memset(&out, 0, sizeof(out));
    while (i < n)
    {
        if (s[i] == '/' && i + 1 < n && s[i + 1] == '/')
        {
            while (i < n && s[i] != '\n') sb_put(&out, "%c", s[i++]);
            continue;
        }
        if (s[i] == '/' && i + 1 < n && s[i + 1] == '*')
        {
            sb_put(&out, "/*");
            i += 2;
            while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) sb_put(&out, "%c", s[i++]);
            if (i + 1 < n) { sb_put(&out, "*/"); i += 2; }
            continue;
        }

        size_t wl = 0;
        while (i + wl < n && (isalnum((unsigned char)s[i + wl]) || s[i + wl] == '_')) wl++;
        if (wl && i + wl < n && (isspace((unsigned char)s[i + wl]) || s[i + wl] == '('))
        {
            char w[32];
            int layout = wl == 6 && !strncmp(s + i, "layout", 6);
            int kw = 0;
            if ((size_t)wl < sizeof(w))
            {
                memcpy(w, s + i, wl);
                w[wl] = 0;
                for (int k = 0; drop[k]; k++)
                    if (!strcmp(w, drop[k])) kw = 1;
                if (storage_too)
                    for (int k = 0; storage[k]; k++)
                        if (!strcmp(w, storage[k])) kw = 1;
            }
            if (layout || kw)
            {
                i += wl;
                if (layout)
                {
                    while (i < n && isspace((unsigned char)s[i])) i++;
                    if (i < n && s[i] == '(')
                    {
                        int depth = 0;
                        while (i < n)
                        {
                            if (s[i] == '(') depth++;
                            else if (s[i] == ')') { depth--; if (!depth) { i++; break; } }
                            i++;
                        }
                    }
                }
                continue;
            }
        }
        sb_put(&out, "%c", s[i++]);
    }
    return out.s;
}

static char *strip_member_quals(const char *s, size_t n)
{
    return strip_member_quals2(s, n, false);
}

static void emit_interface(SB *out, const char *s, size_t pos, size_t end, StmtInfo *info,
                           unsigned in_verts, char *gl_in_name, size_t gl_in_cap, bool *input_iface)
{
    char *raw = xmalloc(end - pos + 1);
    size_t n = end - pos;

    memcpy(raw, s + pos, n);
    raw[n] = 0;

    /* everything up to and including in/out (plus its qualifiers) is dropped */
    const char *body = raw + info->qual_end;

    /* bare `layout(...) in;` — nothing to declare */
    {
        const char *p = body;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == ';')
        {
            free(raw);
            return;
        }
    }

    /* gl_PerVertex block: drop the declaration, remember the instance name */
    if (strstr(body, "gl_PerVertex"))
    {
        const char *close = strrchr(body, '}');
        if (close && close[1])
        {
            char id[64];
            if (read_ident(close + 1, id, sizeof(id)) && gl_in_name && !gl_in_name[0])
                snprintf(gl_in_name, gl_in_cap, "%s", id);
        }
        free(raw);
        return;
    }

    /* an interface block without an instance name puts its members into the enclosing scope:
     * emit them as ordinary declarations (with the block's array dimension folded into them
     * for an input), since a plain struct declaration would not bring the names into scope */
    {
        const char *close = strrchr(body, '}');
        if (close)
        {
            const char *after = close + 1;
            while (*after && isspace((unsigned char)*after)) after++;
            if (!*after || *after == ';')
            {
                const char *open = strchr(body, '{');
                if (open && open < close)
                {
                    SB decl;
                    /* members of an anonymous block may carry interface qualifiers
                     * (layout(location=...), xfb_offset, flat, ...) that are illegal on the
                     * ordinary declarations emitted here */
                    char *members = strip_member_quals(open + 1, (size_t)(close - open - 1));
                    memset(&decl, 0, sizeof(decl));
                    for (const char *p = members ? members : (open + 1); *p; )
                    {
                        if (info->kind == STMT_IFACE_IN && p[0] == '[' && p[1] == ']')
                        {
                            sb_put(&decl, "[%u]", in_verts);
                            p += 2;
                            continue;
                        }
                        sb_put(&decl, "%c", *p);
                        p++;
                    }
                    if (input_iface && info->kind == STMT_IFACE_IN) *input_iface = true;
                    /* the member text already carries its terminating ';' */
                    sb_put(out, "%s", decl.s ? decl.s : "");
                    free(decl.s);
                    free(members);
                    free(raw);
                    return;
                }
            }
        }
    }

    /* drop the qualifier of a variable declaration; size input arrays */
    if (info->kind == STMT_IFACE_IN)
    {
        if (input_iface) *input_iface = true;
        SB decl;
        const char *brace = strchr(body, '{');
        memset(&decl, 0, sizeof(decl));
        if (brace)
        {
            /* a named interface block keeps its struct: GLSL needs the "struct" keyword when
             * it is no longer an interface ("Blk { ... } v[3];" does not parse) */
            char bname[64] = "";
            const char *close = strrchr(body, '}');
            const char *bp = brace, *be;
            while (bp > body && isspace((unsigned char)bp[-1])) bp--;
            be = bp;
            while (be > body && (isalnum((unsigned char)be[-1]) || be[-1] == '_')) be--;
            if (bp - be > 0 && (size_t)(bp - be) < sizeof(bname))
            {
                memcpy(bname, be, bp - be);
                bname[bp - be] = 0;
            }
            if (close && bname[0] && strcmp(bname, "gl_PerVertex"))
            {
                char *members = strip_member_quals2(brace + 1, (size_t)(close - brace - 1), true);
                sb_put(&decl, "struct %s { %s }; %s ", bname, members ? members : "", bname);
                free(members);
                body = close + 1;
            }
        }
        for (const char *p = body; *p; )
        {
            if (p[0] == '[' && p[1] == ']')
            {
                sb_put(&decl, "[%u]", in_verts);
                p += 2;
                continue;
            }
            sb_put(&decl, "%c", *p);
            p++;
        }
        sb_put(out, "%s", decl.s ? decl.s : body);
        free(decl.s);
    }
    else
    {
        SB decl;
        const char *brace = strchr(body, '{');
        memset(&decl, 0, sizeof(decl));

        if (brace)
        {
            /* a named output interface block ("out B { ... } b;") becomes a plain struct
             * plus instance: the shader body refers to the members as b.c, which a plain
             * global of that struct type keeps working; interface-only member qualifiers
             * (layout(location/xfb_offset/stream), flat, ...) are dropped here — the
             * reflection keeps the real values.  GLSL matches the block to the next
             * stage's input block by BLOCK name, so the type name is preserved too. */
            char bname[64] = "";
            const char *close = strrchr(body, '}');
            const char *bp = brace, *be;
            while (bp > body && isspace((unsigned char)bp[-1])) bp--;
            be = bp;
            while (be > body && (isalnum((unsigned char)be[-1]) || be[-1] == '_')) be--;
            if (bp - be > 0 && (size_t)(bp - be) < sizeof(bname))
            {
                memcpy(bname, be, bp - be);
                bname[bp - be] = 0;
            }
            if (close && bname[0] && strcmp(bname, "gl_PerVertex"))
            {
                char *members = strip_member_quals2(brace + 1, (size_t)(close - brace - 1), true);
                sb_put(&decl, "struct %s { %s }; %s ", bname, members ? members : "", bname);
                free(members);
                body = close + 1;
            }
        }
        for (const char *p = body; *p; p++) sb_put(&decl, "%c", *p);
        sb_put(out, "%s", decl.s ? decl.s : body);
        free(decl.s);
    }

    free(raw);
}

/* rewrite a geometry shader into compute-shader source.  Returns the transformed text with
 * the `#version` line removed, or NULL on failure.  gl_in_name receives the instance name a
 * gl_PerVertex block declaration used when it is not "gl_in". */
static char *geom_transform_source(const char *src, unsigned in_verts, char *gl_in_name, size_t gl_in_cap,
                                   bool *input_iface, bool *output_iface)
{
    SB out;
    size_t pos = 0, len = strlen(src);
    int depth = 0;          /* >0 inside a function body */

    memset(&out, 0, sizeof(out));
    if (gl_in_name) gl_in_name[0] = 0;
    if (input_iface) *input_iface = false;
    if (output_iface) *output_iface = false;

    while (pos < len)
    {
        if (depth > 0)
        {
            /* copy the body verbatim (with renames); track braces */
            if (src[pos] == '}')
            {
                depth--;
                sb_put(&out, "}");
                pos++;
                continue;
            }
            if (src[pos] == '{')
            {
                depth++;
                sb_put(&out, "{");
                pos++;
                continue;
            }
            copy_renamed(&out, src, &pos, len, "{}", NULL, NULL);
            continue;
        }

        /* preprocessor directives are copied verbatim to the end of the (logical) line */
        if (src[pos] == '#')
        {
            while (pos < len)
            {
                char c = src[pos];
                sb_put(&out, "%c", c);
                pos++;
                if (c == '\n')
                {
                    /* line continuation */
                    size_t q = pos;
                    while (q > 1 && (src[q - 2] == ' ' || src[q - 2] == '\t' || src[q - 2] == '\r')) q--;
                    if (q >= 2 && src[q - 2] == '\\') continue;
                    break;
                }
            }
            continue;
        }

        /* top level: comments and blank text pass through */
        {
            size_t b = skip_blank(src, pos, len);
            while (pos < b)
            {
                /* copy a comment in place: copy_renamed() has no stop characters here and
                 * would run on to the end of the file, copying every declaration after the
                 * comment verbatim (the CTS limits shaders put a comment between the
                 * preamble and their declarations) */
                if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '/')
                {
                    while (pos < len && src[pos] != '\n') { sb_put(&out, "%c", src[pos]); pos++; }
                }
                else if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '*')
                {
                    sb_put(&out, "/*");
                    pos += 2;
                    while (pos + 1 < len && !(src[pos] == '*' && src[pos + 1] == '/'))
                    { sb_put(&out, "%c", src[pos]); pos++; }
                    if (pos + 1 < len) { sb_put(&out, "*/"); pos += 2; }
                }
                else { sb_put(&out, "%c", src[pos]); pos++; }
            }
            if (pos >= len) break;
            /* a preprocessor directive that follows the skipped blanks is handled by the
             * '#' branch above (it must not fall through to the statement copier, which
             * would swallow the following declaration: "#define A (0)\nflat in int x;")
             */
            if (src[pos] == '#') continue;
            /* likewise a comment: skip_blank does not cross it, and the statement copier
             * would run from the comment to the next ';' — swallowing any "#define" and
             * the declarations that follow it (the CTS limits shaders) */
            if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '/')
            {
                while (pos < len && src[pos] != '\n') { sb_put(&out, "%c", src[pos]); pos++; }
                continue;
            }
            if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '*')
            {
                sb_put(&out, "/*");
                pos += 2;
                while (pos + 1 < len && !(src[pos] == '*' && src[pos + 1] == '/'))
                { sb_put(&out, "%c", src[pos]); pos++; }
                if (pos + 1 < len) { sb_put(&out, "*/"); pos += 2; }
                continue;
            }
        }

        StmtInfo info;
        classify_statement(src, pos, len, &info);
        if (getenv("MGL_GEOM_TRACE"))
        {
            char dbg[80];
            size_t dn = len - pos < 60 ? len - pos : 60;
            memcpy(dbg, src + pos, dn); dbg[dn] = 0;
            fprintf(stderr, "MGL geom trace kind=%d qe=%u: [%s]\n", info.kind, info.qual_end, dbg);
        }

        if (info.kind != STMT_OTHER)
        {
            size_t end = statement_end(src, pos, len);
            emit_interface(&out, src, pos, end, &info, in_verts, gl_in_name, gl_in_cap, input_iface);
            if (info.kind == STMT_IFACE_OUT && output_iface) *output_iface = true;
            pos = end;
            continue;
        }

        /* plain statement (or a function signature): copy until ';' (counting statement
         * braces) or until the '{' that opens a function body */
        int brace = 0;
        while (pos < len)
        {
            char c = src[pos];
            if (c == '/' && pos + 1 < len && (src[pos + 1] == '/' || src[pos + 1] == '*'))
            {
                copy_renamed(&out, src, &pos, len, NULL, NULL, NULL);
                continue;
            }
            if (c == '{')
            {
                if (info.is_struct || brace > 0) { brace++; sb_put(&out, "{"); pos++; continue; }
                depth = 1;
                sb_put(&out, "{");
                pos++;
                break;
            }
            if (c == '}')
            {
                if (brace > 0) brace--;
                sb_put(&out, "}");
                pos++;
                continue;
            }
            if (c == ';' && brace == 0)
            {
                sb_put(&out, ";");
                pos++;
                break;
            }
            if (is_ident_char(c) && !isdigit((unsigned char)c))
            {
                char id[128];
                size_t n = read_ident(src + pos, id, sizeof(id));
                emit_ident(&out, id, NULL);
                pos += n;
                continue;
            }
            sb_put(&out, "%c", c);
            pos++;
        }
    }

    if (out.oom)
    {
        free(out.s);
        return NULL;
    }
    return sb_take(&out);
}

/* ------------------------------------------------------------------ reflection */

static bool geom_reflect_stage(SpvModule *m, bool inputs, MGLGeomVar *vars, int *count, char **log)
{
    int n = 0;
    uint32_t want = inputs ? SpvStorageClassInput : SpvStorageClassOutput;

    for (int i = 0; i < m->n_vars && n < MGL_GEOM_MAX_VARYINGS; i++)
    {
        SpvVar *v = &m->vars[i];
        uint32_t location = 0, tmp;
        MGLGeomVar *out;
        SpvType *t;
        uint32_t elems = 1;

        if (v->storage != want) continue;
        if (spv_deco(m, v->id, SpvDecorationBuiltIn, NULL)) continue;   /* builtins are handled separately */
        if (!spv_deco(m, v->id, SpvDecorationLocation, &location)) continue;   /* user varyings only */

        t = spv_value_type(m, v->type_id, &elems);
        if (!t) continue;
        if (inputs)
        {
            /* the outermost array of a geometry input is the primitive's vertices */
            uint32_t dummy = 1;
            t = spv_value_type(m, v->type_id, &dummy);
            elems = 1;
        }
        /* an interface block is a struct-typed interface variable: its members consume
         * consecutive locations starting at the block's location (glslang decorates the
         * variable, not the members).  With an instance name the members are reached through
         * the instance ("v[i].a"); without one the members are the arrays ("a[i]") and the
         * block's outer array dimension lives on them. */
        if (t->kind == SPVT_STRUCT)
        {
            const char *inst = spv_name(m, v->id);
            bool anon = !inst || !inst[0];
            GLuint next_loc = location;

            for (int mi = 0; mi < t->n_members && n < MGL_GEOM_MAX_VARYINGS; mi++)
            {
                const char *mn = spv_member_name(m, t->id, mi);
                uint32_t mloc = 0, mid = t->members[mi];
                SpvType *mt;
                GLenum mbase;
                GLuint mrows, mcols, consumed = 1;

                if (!mn) continue;

                /* the vertex dimension of an ANONYMOUS block input sits on the member itself
                 * (the variable is the bare struct), so one array layer is stripped there.
                 * With an instance name the variable's outer array was that dimension
                 * already (stripped above), and the member's array is its own ("vs_gs_out[16]"):
                 * stripping it here would collapse the member to one element. */
                mt = spv_find_type(m, mid);
                while (mt && mt->kind == SPVT_PTR) { mt = spv_find_type(m, mt->elem); mid = mt->id; }
                if (inputs && anon && mt && mt->kind == SPVT_ARRAY)
                {
                    mid = mt->elem;
                    mt = spv_find_type(m, mid);
                }
                /* an array member ("out B { vec4 c[2]; } b;"): the element type is stored and
                 * the length becomes elems — one layer only, the rewritten shader declares it
                 * as a struct member with its own [] dimension, which the generators unroll */
                GLuint melems = 1;
                if (mt && mt->kind == SPVT_ARRAY)
                {
                    melems = mt->n;
                    mid = mt->elem;
                    mt = spv_find_type(m, mid);
                }
                if (!mt || melems == 0 || (mt->kind == SPVT_ARRAY) ||
                    (mt->kind != SPVT_VEC && mt->kind != SPVT_MAT &&
                     mt->kind != SPVT_FLOAT && mt->kind != SPVT_INT && mt->kind != SPVT_BOOL))
                {
                    mgl_append_log(log, "ERROR: geometry: unsupported interface block member type for %s.%s\n",
                                   anon ? "?" : inst, mn);
                    return false;
                }

                mbase = mt->kind == SPVT_FLOAT ? (mt->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
                        mt->kind == SPVT_INT ? (mt->is_signed ? GL_INT : GL_UNSIGNED_INT) :
                        mt->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, mt);
                mrows = mt->kind == SPVT_VEC || mt->kind == SPVT_MAT ? mt->n : 1;
                mcols = mt->kind == SPVT_MAT ? spv_find_type(m, mt->elem)->n : 1;
                /* GLSL 4.60 4.4.1: a double (and its vectors) consumes two locations, a
                 * matrix one per column, an array one per element */
                consumed = melems * mcols * (mbase == GL_DOUBLE ? 2u : 1u);

                out = &vars[n++];
                memset(out, 0, sizeof(*out));
                snprintf(out->name, sizeof(out->name), "%s", anon ? mn : inst);
                snprintf(out->member, sizeof(out->member), "%s", anon ? "" : mn);
                out->elems = melems;
                out->xfb_buffer = -1;
                out->xfb_offset = 0xffffffffu;
                out->location = spv_member_deco(m, t->id, mi, SpvDecorationLocation, &mloc) ?
                                mloc : next_loc;
                next_loc = out->location + consumed;
                out->base = mbase;
                out->rows = mrows;
                out->cols = mcols;
                out->words = melems * mcols * mrows * base_words(mbase);
                out->flat = spv_member_deco(m, t->id, mi, SpvDecorationFlat, NULL) ||
                            spv_deco(m, v->id, SpvDecorationFlat, NULL);
                out->noperspective = spv_member_deco(m, t->id, mi, SpvDecorationNoPerspective, NULL) ||
                                     spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
                if (spv_member_deco(m, t->id, mi, SpvDecorationStream, &tmp)) out->stream = tmp;
                /* shader-declared transform feedback capture (xfb_buffer/xfb_offset): the
                 * member inherits the block's buffer, the offset is its own */
                if (spv_member_deco(m, t->id, mi, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
                else if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
                if (spv_member_deco(m, t->id, mi, SpvDecorationOffset, &tmp)) out->xfb_offset = tmp;
            }
            continue;
        }

        if (t->kind != SPVT_VEC && t->kind != SPVT_MAT &&
            t->kind != SPVT_FLOAT && t->kind != SPVT_INT && t->kind != SPVT_BOOL)
        {
            mgl_append_log(log, "ERROR: geometry: unsupported interface type for %s\n",
                           spv_name(m, v->id) ? spv_name(m, v->id) : "?");
            return false;
        }

        out = &vars[n++];
        memset(out, 0, sizeof(*out));
        snprintf(out->name, sizeof(out->name), "%s", spv_name(m, v->id) ? spv_name(m, v->id) : "");
        out->location = location;
        out->xfb_buffer = -1;
        out->xfb_offset = 0xffffffffu;
        out->base = t->kind == SPVT_FLOAT ? (t->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
                    t->kind == SPVT_INT ? (t->is_signed ? GL_INT : GL_UNSIGNED_INT) :
                    t->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, t);
        out->rows = t->kind == SPVT_VEC || t->kind == SPVT_MAT ? t->n : 1;
        out->cols = t->kind == SPVT_MAT ? spv_find_type(m, t->elem)->n : 1;
        out->elems = elems;
        out->words = elems * out->cols * out->rows * base_words(out->base);
        out->flat = spv_deco(m, v->id, SpvDecorationFlat, NULL);
        out->noperspective = spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
        out->centroid = spv_deco(m, v->id, SpvDecorationCentroid, NULL);
        out->sample = spv_deco(m, v->id, SpvDecorationSample, NULL);
        if (spv_deco(m, v->id, SpvDecorationStream, &tmp)) out->stream = tmp;
        if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
        if (spv_deco(m, v->id, SpvDecorationOffset, &tmp)) out->xfb_offset = tmp;
        (void)location;
    }

    *count = n;
    return true;
}

static int geom_find_location(const MGLGeomVar *vars, int count, GLuint location)
{
    for (int i = 0; i < count; i++)
        if (vars[i].location == location) return i;
    return -1;
}

/* ------------------------------------------------------------------ code generation */

/* remove the "#version ..." line of a source so a new one can be prepended */
static char *strip_version(const char *src)
{
    const char *v = strstr(src, "#version");
    if (!v) return xstrdup(src);
    const char *e = strchr(v, '\n');
    if (!e) return xstrdup(v && v > src ? src : "");
    size_t head = (size_t)(v - src);
    char *out = xmalloc(strlen(src) + 1);
    memcpy(out, src, head);
    strcpy(out + head, e + 1);
    return out;
}

/* dump helper (MGL_GEOM_DUMP=<dir>): writes the generated shaders for inspection */
/* the trace / behaviour switches are read once (this file is on the draw path) */
static struct { int read; bool draw, xfb, cpu_xfb, cpu_assembly; } gEnv = { 0, false, false, false, false };

static bool geom_env(int which)
{
    if (!gEnv.read)
    {
        gEnv.read = 1;
        gEnv.draw = getenv("MGL_GEOM_TRACE_DRAW") != NULL;
        gEnv.xfb = getenv("MGL_GEOM_XFB_TRACE") != NULL;
        gEnv.cpu_xfb = getenv("MGL_GEOM_CPU_XFB") != NULL;
        gEnv.cpu_assembly = getenv("MGL_GEOM_CPU_ASSEMBLY") != NULL;
    }
    return which == 0 ? gEnv.draw : which == 1 ? gEnv.xfb : which == 2 ? gEnv.cpu_xfb : gEnv.cpu_assembly;
}

static void geom_dump(const char *tag, const char *text)
{
    static int seq;
    const char *dir = getenv("MGL_GEOM_DUMP");
    char path[512];
    FILE *f;
    if (!dir || !text) return;
    snprintf(path, sizeof(path), "%s/%03d_%s", dir, ++seq, tag);
    f = fopen(path, "w");
    if (!f) return;
    fwrite(text, 1, strlen(text), f);
    fclose(f);
}

/* rename the entry point `main` of a shader source */
static char *rename_main(const char *src, const char *new_name)
{
    size_t len = strlen(src);
    char *out = xmalloc(len + strlen(new_name) + 8);
    size_t o = 0, i = 0;
    bool renamed = false;
    char last_ident[64] = "";

    while (i < len)
    {
        if (is_ident_char(src[i]) && !isdigit((unsigned char)src[i]))
        {
            char id[128];
            size_t n = read_ident(src + i, id, sizeof(id));
            if (!strcmp(id, "main") && !strcmp(last_ident, "void"))
            {
                memcpy(out + o, new_name, strlen(new_name));
                o += strlen(new_name);
                renamed = true;
            }
            else
            {
                memcpy(out + o, id, n);
                o += n;
            }
            snprintf(last_ident, sizeof(last_ident), "%s", id);
            i += n;
            continue;
        }
        /* whitespace between `void` and `main` does not break the sequence */
        if (!is_ident_char(src[i]) && !isspace((unsigned char)src[i])) last_ident[0] = 0;
        out[o++] = src[i++];
    }
    out[o] = 0;

    if (!renamed)
    {
        free(out);
        return NULL;
    }
    return out;
}

/* ---------------- 1. vertex capture variant ---------------- */

/* rewrite the vertex shader's "out" interface blocks into ordinary declarations: an
 * anonymous block's members become variables, a named block's instance becomes a struct.
 * The builtin gl_PerVertex block is left alone. */
static char *geom_plain_outputs(const char *src)
{
    SB out;
    size_t i = 0, len = strlen(src);
    int depth = 0;

    memset(&out, 0, sizeof(out));
    while (i < len)
    {
        /* comments pass through untouched */
        if (src[i] == '/' && i + 1 < len && (src[i + 1] == '/' || src[i + 1] == '*'))
        {
            if (src[i + 1] == '/')
            {
                while (i < len && src[i] != '\n') { sb_put(&out, "%c", src[i]); i++; }
            }
            else
            {
                sb_put(&out, "/*");
                i += 2;
                while (i + 1 < len && !(src[i] == '*' && src[i + 1] == '/')) { sb_put(&out, "%c", src[i]); i++; }
                if (i + 1 < len) { sb_put(&out, "*/"); i += 2; }
            }
            continue;
        }
        if (depth == 0 && src[i] == '#')
        {
            /* a preprocessor directive is line-scoped: it must not be swallowed into the
             * next declaration's head (a '#define X 64' feeding a later declaration's
             * array size would otherwise be dropped when that declaration is rewritten) */
            while (i < len && src[i] != '\n') { sb_put(&out, "%c", src[i]); i++; }
            continue;
        }
        if (depth == 0)
        {
            /* a global-scope declaration: collect the head (up to ';' or '{') and check
             * whether its last storage qualifier is "out" */
            size_t save = i, q = i;
            bool is_out = false, other_storage = false, done_head = false;
            bool line_start = true;

            while (q < len && src[q] != ';' && src[q] != '{')
            {
                if (src[q] == '\n') line_start = true;
                else if (src[q] != '#' && !isspace((unsigned char)src[q])) line_start = false;
                if (src[q] == '#' && line_start)
                {
                    /* a preprocessor directive between declarations: it cannot belong to
                     * the head (a '#define X 16u' feeding a later array size must survive
                     * the head being rewritten), so end the head here and pass it through */
                    for (size_t k = save; k < q; k++) sb_put(&out, "%c", src[k]);
                    while (q < len && src[q] != '\n') { sb_put(&out, "%c", src[q]); q++; }
                    if (q < len) { sb_put(&out, "\n"); q++; }   /* keep the line break */
                    save = q;
                    line_start = true;
                    continue;
                }
                if (src[q] == '(')
                {
                    q++;
                    while (q < len && src[q] != ')') q++;
                    if (q < len) q++;
                    continue;
                }
                if (isalnum((unsigned char)src[q]) || src[q] == '_')
                {
                    char word[64];
                    int n2 = 0;
                    while (q < len && (isalnum((unsigned char)src[q]) || src[q] == '_'))
                    {
                        if (n2 < 63) word[n2++] = src[q];
                        q++;
                    }
                    word[n2] = 0;
                    if (!strcmp(word, "out")) { is_out = true; other_storage = false; }
                    else if (!strcmp(word, "in") || !strcmp(word, "uniform") || !strcmp(word, "buffer") ||
                             !strcmp(word, "attribute") || !strcmp(word, "varying") ||
                             !strcmp(word, "struct") || !strcmp(word, "const"))
                        other_storage = true;
                    continue;
                }
                q++;
            }
            if (q < len && src[q] == '{' && is_out && !other_storage) done_head = true;
            (void)save;
            if (done_head)
            {
                size_t brace = q, close = brace;
                int d = 0;
                while (close < len)
                {
                    if (src[close] == '{') d++;
                    else if (src[close] == '}') { d--; if (!d) break; }
                    close++;
                }
                if (close < len)
                {
                    /* the block name is the identifier before '{' */
                    char bname[64] = "";
                    size_t bp = brace, be;
                    while (bp > 0 && isspace((unsigned char)src[bp - 1])) bp--;
                    be = bp;
                    while (be > 0 && (isalnum((unsigned char)src[be - 1]) || src[be - 1] == '_')) be--;
                    if (bp - be < sizeof(bname))
                    {
                        memcpy(bname, src + be, bp - be);
                        bname[bp - be] = 0;
                    }
                    if (strcmp(bname, "gl_PerVertex"))
                    {
                        char inst[64] = "";
                        size_t p = close + 1;
                        while (p < len && isspace((unsigned char)src[p])) p++;
                        while (p < len && (isalnum((unsigned char)src[p]) || src[p] == '_'))
                        {
                            size_t n2 = strlen(inst);
                            if (n2 + 1 < sizeof(inst)) { inst[n2] = src[p]; inst[n2 + 1] = 0; }
                            p++;
                        }
                        if (inst[0])
                        {
                            /* the block becomes a plain struct + instance; interface-only
                             * member qualifiers (location/xfb_offset/flat) and the storage
                             * qualifiers (in/out) must go: GLSL forbids them on struct
                             * members */
                            char *members = strip_member_quals2(src + brace + 1,
                                                               (size_t)(close - brace - 1), true);
                            sb_put(&out, "struct %s {%s}; %s %s;", bname,
                                   members ? members : "", bname, inst);
                            free(members);
                        }
                        else
                        {
                            /* anonymous block: the members become ordinary outputs of their
                             * own, so the same qualifier stripping applies */
                            char *members = strip_member_quals(src + brace + 1,
                                                               (size_t)(close - brace - 1));
                            sb_put(&out, "%s;", members ? members : "");
                            free(members);
                        }
                        i = p;
                        while (i < len && src[i] != ';') i++;
                        if (i < len) i++;
                        continue;
                    }
                }
            }
        }
        if (src[i] == '{') depth++;
        else if (src[i] == '}') depth--;
        sb_put(&out, "%c", src[i]);
        i++;
    }
    return out.s;
}

/* ---------------- plain (default-block) uniforms of the emulated stages ----------------
 *
 * The vertex capture variant and the geometry kernel are compiled from generated GLSL as
 * their own SPIR-V modules.  A plain "uniform T x;" there would become one
 * [[buffer(n)]] argument per variable (SPIRV-Cross has no notion of a default block),
 * laid out nothing like the program's default-uniform buffer, which introspect injects
 * as the single block "mgl_DefaultUniforms" for the raster stages.  We strip the plain
 * declarations from the transformed source and re-declare them as one std430 block whose
 * member offsets are the ones introspect baked into the program's SPIR-V, so the very
 * buffer that the renderer already builds for the stage can be bound verbatim for the
 * capture pass and the kernel (the renderer's buffer map puts the default block right
 * after the stage's uniform blocks — see mapGLBuffersToMTLBufferMap, which moves the
 * block to the head of the UNIFORM_CONSTANT list — and geomBuildPins below pins the
 * generated MSL to exactly those positions). */

typedef struct {
    char name[64];
    char type[96];      /* GLSL type as written, e.g. "ivec2", "float", "Foo" */
    char arr[48];       /* array suffix as written, e.g. "[4]" ("" for a scalar) */
} GeomPlainUni;

typedef struct {
    char     name[64];
    uint32_t offset;
} GeomDubMember;

static bool is_plain_uniform_type(const char *t)
{
    static const char *words[] = {
        "float", "int", "uint", "bool", "double",
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
        "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
        "dvec2", "dvec3", "dvec4",
        "mat2", "mat3", "mat4", "mat2x3", "mat2x4", "mat3x2", "mat3x4", "mat4x2", "mat4x3",
        "dmat2", "dmat3", "dmat4",
        "const", "highp", "mediump", "lowp",
        NULL
    };
    char tok[32];
    size_t n = 0;

    while (t[n] && !isspace((unsigned char)t[n]) && t[n] != '[' && n + 1 < sizeof(tok)) { tok[n] = t[n]; n++; }
    tok[n] = 0;
    if (!tok[0]) return false;
    for (int i = 0; words[i]; i++)
        if (!strcmp(tok, words[i])) return true;
    /* opaque types are NOT buffer-storable: they must stay where they were declared
     * (re-declaring a sampler inside the default-uniform block is illegal) */
    if (!strncmp(tok, "sampler", 7) || !strncmp(tok, "isampler", 8) ||
        !strncmp(tok, "usampler", 8) || !strncmp(tok, "image", 5) ||
        !strncmp(tok, "iimage", 6) || !strncmp(tok, "uimage", 6) ||
        !strncmp(tok, "atomic_uint", 11) || !strncmp(tok, "texture", 7))
        return false;
    /* anything else is a user-declared struct type (an identifier): also plain storage */
    for (size_t i = 0; i < n; i++)
        if (!is_ident_char((unsigned char)tok[i])) return false;
    return true;
}

/* Remove top-level plain (default-block) uniform declarations from src, recording them
 * (and the raw declarations, for the fallback) .  Declarations of blocks, samplers, images
 * and atomic counters are kept verbatim. */
/* replace C-style and line comments with spaces (newlines kept), so the plain-uniform
 * scanner's statement split is not confused: a comment has no ';' and would glue a
 * #define and the next uniform declaration into one "statement" */
static char *geom_blank_comments(const char *src)
{
    size_t n = strlen(src);
    char *out = xmalloc(n + 1);
    size_t i = 0, o = 0;

    while (i < n)
    {
        if (src[i] == '/' && i + 1 < n && src[i + 1] == '/')
        {
            while (i < n && src[i] != '\n') { out[o++] = ' '; i++; }
        }
        else if (src[i] == '/' && i + 1 < n && src[i + 1] == '*')
        {
            out[o++] = ' ';
            out[o++] = ' ';
            i += 2;
            while (i < n)
            {
                if (src[i] == '*' && i + 1 < n && src[i + 1] == '/')
                {
                    out[o++] = ' ';
                    out[o++] = ' ';
                    i += 2;
                    break;
                }
                out[o++] = (src[i] == '\n') ? '\n' : ' ';
                i++;
            }
        }
        else out[o++] = src[i++];
    }
    out[o] = 0;
    return out;
}

static char *geom_strip_plain_uniforms(const char *src, GeomPlainUni *uni, int *nuni, int max, char **removed)
{
    SB out;
    SB rm;
    size_t pos = 0, len = strlen(src);
    int depth = 0;

    memset(&out, 0, sizeof(out));
    memset(&rm, 0, sizeof(rm));
    *nuni = 0;

    while (pos < len)
    {
        if (depth > 0)
        {
            while (pos < len && depth > 0)
            {
                if (src[pos] == '{') depth++;
                else if (src[pos] == '}') depth--;
                sb_put(&out, "%c", src[pos]);
                pos++;
            }
            continue;
        }

        while (pos < len && isspace((unsigned char)src[pos])) { sb_put(&out, "%c", src[pos]); pos++; }
        if (pos >= len) break;
        if (src[pos] == '{')                    /* a function body starts here */
        {
            depth++;
            sb_put(&out, "{");
            pos++;
            continue;
        }
        if (src[pos] == '#')                 /* preprocessor line: verbatim (after the
                                              * whitespace skip — a '#' found at the raw
                                              * scan start made the statement scanner eat
                                              * the NEXT uniform declaration with the #define) */
        {
            size_t end = pos;
            while (end < len && src[end] != '\n') end++;
            /* honour a trailing backslash (line continuation): the directive continues
             * on the next line */
            for (;;)
            {
                size_t back = end;
                while (back > pos && (src[back - 1] == ' ' || src[back - 1] == '\t')) back--;
                if (back > pos && src[back - 1] == '\\' && end < len)
                {
                    end++;
                    while (end < len && src[end] != '\n') end++;
                }
                else break;
            }
            sb_put(&out, "%.*s", (int)(end - pos), src + pos);
            pos = end;
            continue;
        }

        /* the end of the statement: ';' at depth 0, or a '{' (a block declaration) */
        {
            size_t end = pos;
            int d = 0;
            while (end < len)
            {
                if (src[end] == '{') break;
                if (src[end] == ';' && d == 0) { end++; break; }
                if (src[end] == '(' || src[end] == '[') d++;
                else if (src[end] == ')' || src[end] == ']') d--;
                end++;
            }

            {
                char id[32];
                size_t n = read_ident(src + pos, id, sizeof(id));

                if (n && !strcmp(id, "uniform") && end < len && src[end - 1] == ';')
                {
                    const char *body = src + pos + n;   /* between "uniform" and ';' */
                    const char *bend = src + end - 1;
                    char type[96] = "";
                    bool keep = false;

                    while (body < bend && isspace((unsigned char)*body)) body++;
                    while (bend > body && isspace((unsigned char)bend[-1])) bend--;

                    /* "uniform [const] T a[4], b" — split declarators on ',' */
                    {
                        const char *p = body;
                        while (p < bend && !keep)
                        {
                            const char *c = p, *e;
                            while (c < bend && isspace((unsigned char)*c)) c++;
                            e = c;
                            while (e < bend && *e != ',') e++;
                            if (memchr(c, '=', (size_t)(e - c))) keep = true;  /* initializer */
                            else
                            {
                                const char *nb = e;   /* end of the declarator */
                                const char *ne;
                                char name[64];
                                char arr[48] = "";
                                size_t nn;

                                while (nb > c && isspace((unsigned char)nb[-1])) nb--;
                                if (nb > c && nb[-1] == ']')
                                {
                                    /* trailing [N]: keep the suffix text, the name ends before '[' */
                                    const char *ob = nb - 1;   /* at the ']' */

                                    while (ob > c && ob[-1] != '[') ob--;
                                    if (ob == c) keep = true;   /* ']' with no '[' */
                                    else
                                    {
                                        size_t alen = (size_t)(nb - (ob - 1));

                                        if (alen >= sizeof(arr)) keep = true;
                                        else
                                        {
                                            memcpy(arr, ob - 1, alen);
                                            arr[alen] = 0;
                                            nb = ob - 1;         /* the name ends here */
                                        }
                                    }
                                }
                                ne = nb;
                                while (ne > c && is_ident_char((unsigned char)ne[-1])) ne--;
                                nn = (size_t)(nb - ne);
                                if (!nn || ne == c) keep = true;   /* covers "a[2][3]" too */
                                else
                                {
                                    memcpy(name, ne, nn);
                                    name[nn] = 0;
                                    if (!type[0])
                                    {
                                        size_t t = (size_t)(ne - c);
                                        while (t > 0 && isspace((unsigned char)c[t-1])) t--;
                                        if (t == 0 || t >= sizeof(type)) keep = true;
                                        else
                                        {
                                            memcpy(type, c, t);
                                            type[t] = 0;
                                            /* drop precision/storage qualifiers so the
                                             * type name is what is_plain_uniform_type sees
                                             * ("lowp isampler2D" -> "isampler2D") */
                                            for (;;)
                                            {
                                                bool again = false;
                                                static const char *quals[] = { "lowp", "mediump", "highp", "precise", "const", NULL };

                                                for (int q = 0; quals[q]; q++)
                                                {
                                                    size_t l = strlen(quals[q]);
                                                    if (!strncmp(type, quals[q], l) &&
                                                        (type[l] == ' ' || type[l] == '\t'))
                                                    {
                                                        memmove(type, type + l + 1, strlen(type + l + 1) + 1);
                                                        again = true;
                                                        break;
                                                    }
                                                }
                                                if (!again) break;
                                            }
                                            if (!type[0]) keep = true;
                                        }
                                    }
                                    if (!keep)
                                    {
                                        if (*nuni >= max || !is_plain_uniform_type(type)) keep = true;
                                        else
                                        {
                                            snprintf(uni[*nuni].name, sizeof(uni[*nuni].name), "%s", name);
                                            snprintf(uni[*nuni].type, sizeof(uni[*nuni].type), "%s", type);
                                            snprintf(uni[*nuni].arr, sizeof(uni[*nuni].arr), "%s", arr);
                                            (*nuni)++;
                                        }
                                    }
                                }
                            }
                            p = e + 1;
                        }
                    }

                    if (keep)
                    {
                        /* opaque or unsupported: keep this declaration verbatim (the
                         * already-removed ones stay removed — they are re-declared from
                         * the default-uniform block, so they must NOT be reset here) */
                        sb_put(&out, "%.*s", (int)(end - pos), src + pos);
                    }
                    else
                    {
                        if (removed) sb_put(&rm, "%.*s\n", (int)(end - pos), src + pos);
                    }
                    pos = end;
                    continue;
                }
            }

            /* anything else: copy, tracking braces */
            {
                size_t p = pos;
                while (p < end)
                {
                    if (src[p] == '{') depth++;
                    sb_put(&out, "%c", src[p]);
                    p++;
                }
                if (end == pos && pos < len) { pos++; }     /* always make progress */
                else pos = end;
            }
        }
    }

    if (removed) *removed = sb_take(&rm);
    else free(rm.s);
    return sb_take(&out);
}

/* The layout introspect baked into the program's SPIR-V for a stage's default-uniform
 * block: find the injected "mgl_DefaultUniforms" STRUCT and read its member offsets.
 * The name is on both the struct and its instance variable (introspect names them the
 * same): the struct is the id that carries the OpMemberName entries. */
static bool geom_dub_layout(SpvModule *m, GeomDubMember *mem, int *nmem, int max)
{
    uint32_t sid = 0;
    int n = 0;

    *nmem = 0;
    if (!m) return false;
    for (int i = 0; i < m->n_names; i++)
    {
        if (strcmp(m->names[i].name, "mgl_DefaultUniforms")) continue;
        for (int k = 0; k < m->n_mnames; k++)
            if (m->mnames[k].id == m->names[i].id) { sid = m->names[i].id; break; }
        if (sid) break;
    }
    if (!sid) return false;

    for (int k = 0; n < max; k++)
    {
        bool found = false;

        for (int i = 0; i < m->n_mnames && n < max; i++)
        {
            if (m->mnames[i].id != sid || m->mnames[i].member != (uint32_t)k) continue;
            snprintf(mem[n].name, sizeof(mem[n].name), "%s", m->mnames[i].name);
            mem[n].offset = 0;
            found = true;
            break;
        }
        if (!found) break;
        for (int i = 0; i < m->n_mdecos; i++)
            if (m->mdecos[i].id == sid && m->mdecos[i].member == (uint32_t)k &&
                m->mdecos[i].deco == SpvDecorationOffset)
                mem[n].offset = m->mdecos[i].value;
        n++;
    }
    *nmem = n;
    return n > 0;
}

/* re-declare the stripped plain uniforms as one block matching that layout */
static void geom_emit_dub_block(SB *b, const GeomPlainUni *uni, int nuni,
                                const GeomDubMember *mem, int nmem)
{
    sb_put(b, "layout(std430, binding = %d) readonly buffer mgl_DefaultUniforms {\n", MGL_GEOM_SSBO_DUB);
    for (int i = 0; i < nuni; i++)
    {
        uint32_t off = 0;
        bool have = false;

        for (int k = 0; k < nmem; k++)
            if (!strcmp(mem[k].name, uni[i].name)) { off = mem[k].offset; have = true; break; }
        if (!have) continue;      /* introspect did not keep it (unused): skip */
        /* the array suffix is re-declared: std430 gives a 4-component-vector array a
         * 16-byte stride, the same layout introspect's msl_layout used for the offsets */
        sb_put(b, "    layout(offset = %u) %s %s%s;\n", off, uni[i].type, uni[i].name, uni[i].arr);
    }
    sb_put(b, "};\n");
}

/* pin the generated module's user resources to the flat buffer positions the renderer's
 * buffer map gives the same resources of the program's own stage:
 * mapGLBuffersToMTLBufferMap enumerates [uniform buffers, uniform constants (the default
 * block first), storage buffers, atomics] and binds each entry at its list position, which
 * is also the MSL buffer index in the program's stage module (SPIRV-Cross numbers uniform
 * buffers, then plain-uniform / uniform-constant args, then storage buffers).  prog is the
 * program's patched SPIR-V of the stage (it owns the ordering); mine is the generated
 * module (it owns the binding values that key the pins). */

/* Build the kernel-pinning map.  Returns the number of Metal argument slots the stage's
 * program resources span (0 on nothing to pin); a block/SSBO declared as an instance ARRAY
 * spans array_size consecutive slots, so the per-type counters advance by that — the bind
 * walk (geomBindStageBuffers / the capture pass) lays the buffers out the same way. */
static int geom_build_pins(SpvModule *prog, SpvModule *mine, bool pin_dub,
                           GeomPin *pins, int *npins, int max)
{
    uint32_t ub[32], uc[32], sb[32];
    int nub = 0, nuc = 0, nsb = 0, total = 0, ub_span = 0;
    bool has_dub = false;

    *npins = 0;

    for (int i = 0; i < prog->n_vars; i++)
    {
        SpvVar *v = &prog->vars[i];
        if (v->storage == SpvStorageClassUniform)
        {
            const char *nm = spv_name(prog, v->id);
            if (nm && !strcmp(nm, "mgl_DefaultUniforms")) { has_dub = true; continue; }
            if (nub < 32)
            {
                uint32_t asz = 1;

                ub[nub++] = v->id;
                spv_value_type(prog, v->type_id, &asz);   /* an instance array spans asz slots */
                ub_span += (int)asz;
            }
        }
        else if (v->storage == SpvStorageClassUniformConstant) { if (nuc < 32) uc[nuc++] = v->id; }
        else if (v->storage == SpvStorageClassStorageBuffer) { if (nsb < 32) sb[nsb++] = v->id; }
    }

    /* flat positions: ub j -> j; dub -> nub; uc k -> nub + has_dub + k; sb j -> nub + has_dub + nuc + j */
    {
        int mub = 0, muc = 0, msb = 0;

        for (int i = 0; i < mine->n_vars; i++)
        {
            SpvVar *v = &mine->vars[i];
            int flat;
            uint32_t asz = 1;

            /* block-instance arrays span asz consecutive slots (GL binding points base..base+asz-1) */
            spv_value_type(mine, v->type_id, &asz);

            if (v->storage == SpvStorageClassUniform)
            {
                if (mub >= nub) continue;
                flat = mub;
                mub += asz;
                total += asz;
            }
            else if (v->storage == SpvStorageClassUniformConstant)
            {
                if (muc >= nuc) continue;
                flat = ub_span + (has_dub ? 1 : 0) + muc;
                muc++;
                total++;
            }
            else if (v->storage == SpvStorageClassStorageBuffer)
            {
                if (msb >= nsb) continue;
                flat = ub_span + (has_dub ? 1 : 0) + nuc + msb;
                msb += asz;
                total += asz;
            }
            else continue;

            if (*npins < max)
            {
                pins[*npins].binding = spv_binding(mine, v->id);
                pins[*npins].msl = flat;
                pins[*npins].texture = (v->storage == SpvStorageClassUniformConstant);
                (*npins)++;
            }
        }
    }

    if (pin_dub && *npins < max)
    {
        pins[*npins].binding = MGL_GEOM_SSBO_DUB;
        pins[*npins].msl = ub_span;   /* right after the UBOs' expanded span */
        pins[*npins].texture = false;
        (*npins)++;
        total++;
    }
    return total;
}

static char *geom_gen_capture_glsl(GLMContext ctx, MGLGeomProgram *gp, const char *vs_src,
                                   const SpvModule *vs_m, char **log)
{
    char *renamed = rename_main(vs_src, "mgl_geom_vs_main");
    char *nover = NULL;
    char *noblk = NULL;
    char *removed_src = NULL;
    GeomPlainUni uni[MGL_GEOM_MAX_VARYINGS];
    GeomDubMember dub[MGL_GEOM_MAX_VARYINGS];
    int nuni_ = 0, ndub = 0;
    SB b;

    if (!renamed)
    {
        mgl_append_log(log, "ERROR: geometry: vertex shader has no main()\n");
        return NULL;
    }
    nover = strip_version(renamed);
    free(renamed);
    /* plain (default-block) uniforms are re-declared below as one block that matches the
     * program's default-uniform buffer (see geom_emit_dub_block) */
    nuni_ = 0;
    {
        char *stripped = geom_strip_plain_uniforms(nover, uni, &nuni_, MGL_GEOM_MAX_VARYINGS, &removed_src);
        if (!stripped) { free(nover); return NULL; }
        if (nuni_) { free(nover); nover = stripped; }
        else free(stripped);
    }
    /* the captured outputs must be ordinary variables: an "out" interface block would be
     * mapped to the vertex function's output struct (and SPIRV-Cross's copy-back of an
     * anonymous block reads a temporary), while the store below needs the members in scope */
    noblk = geom_plain_outputs(nover);
    free(nover);
    nover = noblk;

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version 460 core\n");

    /* re-declare the stripped plain uniforms (or put them back if the program's
     * default-uniform block could not be read) — ahead of the source, so the names are
     * in scope for it */
    if (nuni_ && geom_dub_layout((SpvModule *)vs_m, dub, &ndub, MGL_GEOM_MAX_VARYINGS))
        geom_emit_dub_block(&b, uni, nuni_, dub, ndub);
    else if (removed_src)
    {
        sb_put(&b, "%s", removed_src);
        nuni_ = 0;
    }
    free(removed_src);
    removed_src = NULL;

    sb_put(&b, "%s\n", nover);
    free(nover);

    sb_put(&b, "layout(std430, binding = %d) buffer MGLGeomVsOutBlock { uint mgl_hdr[8]; uint mgl_rec[]; };\n",
           MGL_GEOM_SSBO_VSOUT);
    sb_put(&b, "void main()\n{\n    mgl_geom_vs_main();\n");
    sb_put(&b, "    uint mgl_o = ((uint(gl_VertexID) - mgl_hdr[0]) + (uint(gl_InstanceID) - mgl_hdr[2]) * mgl_hdr[1]) * mgl_hdr[3];\n");

    if (gp->in_has_pos)
    {
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.x);\n", gp->in_pos_off + 0);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.y);\n", gp->in_pos_off + 1);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.z);\n", gp->in_pos_off + 2);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.w);\n", gp->in_pos_off + 3);
    }
    if (gp->in_has_psize)
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_PointSize);\n", gp->in_psize_off);
    if (gp->in_has_clip)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_ClipDistance[%u]);\n", gp->in_clip_off + i, i);
    if (gp->in_has_cull)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_CullDistance[%u]);\n", gp->in_cull_off + i, i);

    for (int i = 0; i < gp->n_in; i++)
    {
        MGLGeomVar *v = &gp->in_v[i];
        char open[64], close[8];
        snprintf(open, sizeof(open), "mgl_rec[mgl_o");
        snprintf(close, sizeof(close), "]");
        /* an input the vertex stage never writes stays undefined in GL: record zeros */
        gen_varying_store(&b, v, "mgl_rec", "mgl_o", v->expr[0] ? v->expr : NULL);
    }

    sb_put(&b, "}\n");
    return sb_take(&b);
}

static char *geom_gen_kernel_glsl(GLMContext ctx, MGLGeomProgram *gp, const char *gs_src,
                                  const SpvModule *gs_m, char **log)
{
    char gl_in_name[64];
    char *main_renamed = rename_main(gs_src, "mgl_gs_main");
    char *stripped_src = NULL;
    char *removed_src = NULL;
    GeomPlainUni uni[MGL_GEOM_MAX_VARYINGS];
    GeomDubMember dub[MGL_GEOM_MAX_VARYINGS];
    int nuni_ = 0, ndub = 0;
    char *transformed = geom_transform_source(main_renamed ? main_renamed : gs_src, gp->in_verts,
                                              gl_in_name, sizeof(gl_in_name), NULL, NULL);
    free(main_renamed);
    char *nover = NULL;
    SB b;
    GLuint out_type_code = gp->out_type == GL_POINTS ? 0 : gp->out_type == GL_LINE_STRIP ? 1 : 2;
    int version = 460;
    const char *vstr = strstr(gs_src, "#version");

    if (vstr) sscanf(vstr, "#version %d", &version);
    if (version < 450) version = 450;

    if (!transformed)
    {
        mgl_append_log(log, "ERROR: geometry: source transform failed\n");
        return NULL;
    }
    geom_dump("gs.transformed.glsl", transformed);
    nover = strip_version(transformed);
    free(transformed);
    {
        char *blanked = geom_blank_comments(nover);
        free(nover);
        nover = blanked;
    }
    /* plain (default-block) uniforms are re-declared below as one block that matches the
     * program's default-uniform buffer (see geom_emit_dub_block) */
    stripped_src = geom_strip_plain_uniforms(nover, uni, &nuni_, MGL_GEOM_MAX_VARYINGS, &removed_src);
    if (!stripped_src) { free(nover); return NULL; }
    if (nuni_) { free(nover); nover = stripped_src; }
    else free(stripped_src);

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version %d core\n", version);

    /* the source's LEADING preprocessor lines must come first: a re-declared uniform's
     * array size may be a macro (uniform ivec4 a[NUMBER];) that would otherwise not be
     * defined yet where the default-uniform block is emitted */
    {
        size_t nlen = strlen(nover), h = 0, last_pp = 0;

        while (h < nlen)
        {
            size_t line = h, s = h;

            while (line < nlen && nover[line] != '\n') line++;
            while (s < line && (nover[s] == ' ' || nover[s] == '\t')) s++;
            if (s < line && nover[s] == '#' && s + 1 < line)
            {
                /* only self-contained directives may be hoisted: a conditional block
                 * (#ifdef/#else/#endif) must stay intact around the code it guards */
                char d[16];
                size_t t = s + 1, n = 0;

                while (t < line && n < sizeof(d) - 1 &&
                       (isalnum((unsigned char)nover[t]) || nover[t] == '_'))
                    d[n++] = nover[t++];
                d[n] = 0;
                if (!strcmp(d, "define") || !strcmp(d, "pragma") || !strcmp(d, "extension"))
                    last_pp = line + 1;
                else break;
            }
            else if (s < line)
            {
                /* a one-line global precision statement is also order-independent and
                 * commonly sits between the #defines and the uniforms */
                if (!strncmp(nover + s, "precision", 9) && line > s + 2 &&
                    nover[line - 1] == ';')
                    last_pp = line + 1;
                else break;              /* first real statement */
            }
            h = line + 1;                          /* blank lines continue the scan */
        }
        h = last_pp;
        if (h)
        {
            sb_put(&b, "%.*s", (int)h, nover);
            char *rest = xstrdup(nover + h);
            free(nover);
            nover = rest;
        }
    }

    /* re-declare the stripped plain uniforms (or put them back if the program's
     * default-uniform block could not be read) — ahead of the source, so the names are
     * in scope for it */
    if (nuni_ && geom_dub_layout((SpvModule *)gs_m, dub, &ndub, MGL_GEOM_MAX_VARYINGS))
        geom_emit_dub_block(&b, uni, nuni_, dub, ndub);
    else if (removed_src)
    {
        sb_put(&b, "%s", removed_src);
        nuni_ = 0;
    }
    free(removed_src);
    removed_src = NULL;
    sb_put(&b, "layout(local_size_x = 32) in;\n\n");
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsPrimBlock { uint mgl_prim[]; };\n", MGL_GEOM_SSBO_PRIM);
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsVsOutBlock { uint mgl_vsout[]; };\n", MGL_GEOM_SSBO_VSOUT);
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsHdrBlock { uint mgl_hdr[]; };\n", MGL_GEOM_SSBO_HDR);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsOutBlock { uint mgl_out[]; };\n", MGL_GEOM_SSBO_OUTV);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsIdxBlock { uint mgl_idx[]; };\n", MGL_GEOM_SSBO_IDX);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsCtlBlock { uint mgl_ctl[]; };\n", MGL_GEOM_SSBO_CTL);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsCntBlock { uint mgl_cnt[]; };\n\n", MGL_GEOM_SSBO_CNT);

    sb_put(&b, "struct MGLGsInVert\n{\n");
    sb_put(&b, "    vec4 mgl_gs_pos;\n");
    if (gp->in_has_psize) sb_put(&b, "    float mgl_gs_psize;\n");
    if (gp->in_has_clip) sb_put(&b, "    float mgl_gs_clip[%u];\n", gp->in_clip_size);
    if (gp->in_has_cull) sb_put(&b, "    float mgl_gs_cull[%u];\n", gp->in_clip_size);
    sb_put(&b, "};\n");
    sb_put(&b, "MGLGsInVert mgl_in[%u];\n", gp->in_verts);
    if (gl_in_name[0] && strcmp(gl_in_name, "gl_in"))
        sb_put(&b, "#define %s mgl_in\n", gl_in_name);
    sb_put(&b, "\n");

    sb_put(&b, "vec4 mgl_gs_pos;\n");
    if (gp->out_has_psize) sb_put(&b, "float mgl_gs_psize;\n");
    if (gp->out_has_layer) sb_put(&b, "int mgl_gs_layer;\n");
    if (gp->out_has_vp) sb_put(&b, "int mgl_gs_vp;\n");
    if (gp->out_has_prim) sb_put(&b, "int mgl_gs_prim_out;\n");
    if (gp->out_has_clip) sb_put(&b, "float mgl_gs_clip[%u];\n", gp->out_clip_size);
    if (gp->out_has_cull) sb_put(&b, "float mgl_gs_cull[%u];\n", gp->out_clip_size);
    sb_put(&b, "int mgl_gs_prim_in;\nint mgl_gs_invoc;\n");
    sb_put(&b, "uint mgl_written, mgl_strip_n, mgl_seq, mgl_tag, mgl_stream, mgl_slot_base;\n");
    /* the vertices of the strip in emit order: threads interleave (one per primitive per
     * invocation), so the reserved slots are not consecutive */
    sb_put(&b, "uint mgl_strip_v[%u];\n\n", gp->max_vertices ? gp->max_vertices : 1);


    /* the rewritten geometry shader may call the emit helpers: declare them first */
    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s);\n");
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s);\n");
    sb_put(&b, "void mgl_gs_emit_vertex();\n");
    sb_put(&b, "void mgl_gs_end_primitive();\n\n");

    sb_put(&b, "%s\n\n", nover);
    free(nover);

    /* the rewritten geometry shader declares its inputs and outputs and defines the entry
     * point; the emulation helpers must be visible to it */
    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s);\n");
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s);\n");
    sb_put(&b, "void mgl_gs_emit_vertex();\n");
    sb_put(&b, "void mgl_gs_end_primitive();\n\n");
    /* load one input primitive */
    sb_put(&b, "void mgl_load_inputs(uint mgl_pi, uint mgl_inst)\n{\n");
    sb_put(&b, "    for (uint mgl_k = 0u; mgl_k < %uu; mgl_k++) {\n", gp->in_verts);
    sb_put(&b, "        uint mgl_r = (mgl_prim[mgl_pi * %uu + mgl_k] - mgl_hdr[12] + mgl_inst * mgl_hdr[6]) * %uu;\n",
           gp->in_verts, gp->in_words);
    if (gp->in_has_pos)
        sb_put(&b, "        mgl_in[mgl_k].mgl_gs_pos = vec4(uintBitsToFloat(mgl_vsout[mgl_r + %u]), uintBitsToFloat(mgl_vsout[mgl_r + %u]),"
                   " uintBitsToFloat(mgl_vsout[mgl_r + %u]), uintBitsToFloat(mgl_vsout[mgl_r + %u]));\n",
               gp->in_pos_off + 0, gp->in_pos_off + 1, gp->in_pos_off + 2, gp->in_pos_off + 3);
    if (gp->in_has_psize)
        sb_put(&b, "        mgl_in[mgl_k].mgl_gs_psize = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", gp->in_psize_off);
    if (gp->in_has_clip)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "        mgl_in[mgl_k].mgl_gs_clip[%u] = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", i, gp->in_clip_off + i);
    if (gp->in_has_cull)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "        mgl_in[mgl_k].mgl_gs_cull[%u] = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", i, gp->in_cull_off + i);
    for (int i = 0; i < gp->n_in; i++)
    {
        sb_put(&b, "        ");
        gen_varying_load(&b, &gp->in_v[i], "mgl_vsout[mgl_r", "]", "mgl_k");
    }
    sb_put(&b, "    }\n}\n\n");

    /* store one output vertex */
    sb_put(&b, "void mgl_store_output(uint slot)\n{\n    uint o = slot * %uu;\n", gp->out_words);
    sb_put(&b, "    mgl_out[o + 0u] = mgl_tag;\n    mgl_out[o + 1u] = (mgl_stream << 24u) | mgl_seq;\n");
    if (gp->out_has_pos)
    {
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.x);\n", gp->out_pos_off + 0);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.y);\n", gp->out_pos_off + 1);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.z);\n", gp->out_pos_off + 2);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.w);\n", gp->out_pos_off + 3);
    }
    if (gp->out_has_psize) sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_psize);\n", gp->out_psize_off);
    if (gp->out_has_layer) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_layer);\n", gp->out_layer_off);
    if (gp->out_has_vp) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_vp);\n", gp->out_vp_off);
    if (gp->out_has_prim) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_prim_out);\n", gp->out_prim_off);
    if (gp->out_has_clip)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_clip[%u]);\n", gp->out_clip_off + i, i);
    if (gp->out_has_cull)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_cull[%u]);\n", gp->out_cull_off + i, i);
    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char src[192];
        gen_varying_store(&b, v, "mgl_out", "o", geom_var_expr(src, sizeof(src), v));
    }
    sb_put(&b, "}\n\n");

    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s)\n{\n");
    sb_put(&b, "    mgl_stream = s;\n");
    sb_put(&b, "    if (mgl_written >= mgl_hdr[5]) { mgl_ctl[3] = 1u; return; }\n");
    sb_put(&b, "    uint slot = mgl_slot_base + mgl_seq;\n");
    sb_put(&b, "    if (slot >= mgl_hdr[3]) { mgl_ctl[3] = 1u; return; }\n");
    sb_put(&b, "    mgl_store_output(slot);\n");
    sb_put(&b, "    mgl_strip_v[mgl_strip_n] = slot;\n");
    sb_put(&b, "    mgl_strip_n++; mgl_written++; mgl_seq++;\n");
    sb_put(&b, "}\n\n");
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s)\n{\n");
    sb_put(&b, "    if (mgl_strip_n == 0u) return;\n");
    sb_put(&b, "    uint n = mgl_strip_n;\n");
    sb_put(&b, "    uint cnt = 0u;\n");
    if (out_type_code == 0)
        sb_put(&b, "    cnt = n;\n");
    else if (out_type_code == 1)
        sb_put(&b, "    if (n >= 2u) cnt = (n - 1u) * 2u;\n");
    else
        sb_put(&b, "    if (n >= 3u) cnt = (n - 2u) * 3u;\n");
    sb_put(&b, "    if (s == 0u && cnt > 0u) {\n");
    sb_put(&b, "        uint base = atomicAdd(mgl_ctl[1], cnt);\n");
    sb_put(&b, "        if (base + cnt > mgl_hdr[4]) { mgl_ctl[3] = 1u; }\n");
    sb_put(&b, "        else {\n");
    if (out_type_code == 0)
    {
        sb_put(&b, "            for (uint i = 0u; i < n; i++) mgl_idx[base + i] = mgl_strip_v[i];\n");
    }
    else if (out_type_code == 1)
    {
        sb_put(&b, "            for (uint i = 0u; i + 1u < n; i++) { mgl_idx[base + i * 2u] = mgl_strip_v[i];"
                   " mgl_idx[base + i * 2u + 1u] = mgl_strip_v[i + 1u]; }\n");
    }
    else
    {
        sb_put(&b, "            for (uint i = 0u; i + 2u < n; i++) {\n");
        sb_put(&b, "                uint a = mgl_strip_v[i], b2 = mgl_strip_v[i + 1u], c = mgl_strip_v[i + 2u];\n");
        sb_put(&b, "                if ((i & 1u) == 0u) { mgl_idx[base + i * 3u] = a; mgl_idx[base + i * 3u + 1u] = b2; mgl_idx[base + i * 3u + 2u] = c; }\n");
        sb_put(&b, "                else { mgl_idx[base + i * 3u] = b2; mgl_idx[base + i * 3u + 1u] = a; mgl_idx[base + i * 3u + 2u] = c; }\n");
        sb_put(&b, "            }\n");
    }
    sb_put(&b, "        }\n    }\n");
    sb_put(&b, "    atomicAdd(mgl_ctl[13u + s], 1u);\n");
    sb_put(&b, "    mgl_strip_n = 0u;\n");
    sb_put(&b, "}\n\n");
    sb_put(&b, "void mgl_gs_emit_vertex() { mgl_gs_emit_stream_vertex(0u); }\n");
    sb_put(&b, "void mgl_gs_end_primitive() { mgl_gs_end_stream_primitive(0u); }\n\n");

    sb_put(&b, "void main()\n{\n");
    sb_put(&b, "    uint mgl_t = gl_GlobalInvocationID.x;\n");
    sb_put(&b, "    uint mgl_invocations = mgl_hdr[2];\n");
    sb_put(&b, "    if (mgl_invocations == 0u) return;\n");
    sb_put(&b, "    uint mgl_p = mgl_t / mgl_invocations;\n");
    sb_put(&b, "    uint mgl_inv = mgl_t - mgl_p * mgl_invocations;\n");
    sb_put(&b, "    if (mgl_p >= mgl_hdr[0] * mgl_hdr[1]) return;\n");
    sb_put(&b, "    uint mgl_inst = mgl_p / mgl_hdr[0];\n");
    sb_put(&b, "    uint mgl_pi = mgl_p - mgl_inst * mgl_hdr[0];\n");
    sb_put(&b, "    mgl_load_inputs(mgl_pi, mgl_inst);\n");
    sb_put(&b, "    mgl_gs_prim_in = int(mgl_pi);\n    mgl_gs_invoc = int(mgl_inv);\n");
    sb_put(&b, "    mgl_tag = mgl_p; mgl_seq = 0u; mgl_written = 0u; mgl_strip_n = 0u; mgl_stream = 0u;\n");
    sb_put(&b, "    mgl_slot_base = mgl_t * mgl_hdr[5];\n");
    sb_put(&b, "    mgl_gs_pos = vec4(0.0);\n");
    if (gp->out_has_psize) sb_put(&b, "    mgl_gs_psize = 1.0;\n");
    if (gp->out_has_layer) sb_put(&b, "    mgl_gs_layer = 0;\n");
    if (gp->out_has_vp) sb_put(&b, "    mgl_gs_vp = 0;\n");
    if (gp->out_has_prim) sb_put(&b, "    mgl_gs_prim_out = int(mgl_pi);\n");
    if (gp->out_has_clip) sb_put(&b, "    for (int i = 0; i < %u; i++) mgl_gs_clip[i] = 0.0;\n", gp->out_clip_size);
    if (gp->out_has_cull) sb_put(&b, "    for (int i = 0; i < %u; i++) mgl_gs_cull[i] = 0.0;\n", gp->out_clip_size);
    for (int i = 0; i < gp->n_out; i++)
    {
        /* a named output block member is zero-initialized through the instance ("b.c");
         * a global instance in GLSL is zero-initialized anyway, this is belt and braces */
        char src[192];
        if (gp->out_v[i].member[0] && strcmp(gp->out_v[i].name, gp->out_v[i].member))
            snprintf(src, sizeof(src), "%s.%s", gp->out_v[i].name, gp->out_v[i].member);
        else
            snprintf(src, sizeof(src), "%s", gp->out_v[i].name);
        sb_put(&b, "    %s%s = %s(0);\n", src, gp->out_v[i].elems > 1 ? "[0]" : "",
               glsl_type_name(gp->out_v[i].base, gp->out_v[i].rows, gp->out_v[i].cols));
    }
    sb_put(&b, "    mgl_gs_main();\n");
    sb_put(&b, "    mgl_gs_end_stream_primitive(mgl_stream);\n");
    sb_put(&b, "    mgl_cnt[mgl_t] = mgl_written;\n");
    sb_put(&b, "    if (mgl_written != 0u) atomicAdd(mgl_ctl[0], mgl_written);\n");
    sb_put(&b, "}\n\n");


    return sb_take(&b);
}

/* ---------------- 3. pass-through raster vertex function ---------------- */

static char *geom_gen_draw_glsl(GLMContext ctx, MGLGeomProgram *gp, char **log)
{
    SB b;
    GLuint next_loc = 0;

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version 460 core\n\n");
    /* the record is read as a flat word array: the geometry kernel writes it with exactly
     * this layout, so no std430 struct padding can get in the way */
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsDrawBlock { uint mgl_draw[]; };\n\n",
           MGL_GEOM_SSBO_DRAW);

    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char arr[32] = "";
        char name[160];
        if (v->elems > 1) snprintf(arr, sizeof(arr), "[%u]", v->elems);
        /* a named output block member is emitted as a plain varying of its own: the
         * raster side links to the fragment stage's input block by location, so the
         * member's name is irrelevant here (and the instance name would collide) */
        if (v->member[0] && strcmp(v->name, v->member))
            snprintf(name, sizeof(name), "mgl_ob%d", i);
        else
            snprintf(name, sizeof(name), "%s", v->name);
        sb_put(&b, "layout(location = %u) %s%s%s out %s %s%s;\n", v->location,
               v->flat ? "flat " : "", v->noperspective ? "noperspective " : "",
               v->centroid ? "centroid " : "", glsl_type_name(v->base, v->rows, v->cols), name, arr);
        if (v->location + v->elems * (v->cols ? v->cols : 1) > next_loc)
            next_loc = v->location + v->elems * (v->cols ? v->cols : 1);
    }
    if (gp->out_has_layer)
    {
        gp->layer_loc = next_loc++;
        sb_put(&b, "layout(location = %u) flat out uint mgl_out_layer;\n", gp->layer_loc);
    }
    if (gp->out_has_vp)
    {
        gp->vp_loc = next_loc++;
        sb_put(&b, "layout(location = %u) flat out uint mgl_out_vp;\n", gp->vp_loc);
    }

    sb_put(&b, "\nvoid main()\n{\n");
    sb_put(&b, "    uint r = uint(gl_VertexID) * %uu;\n", gp->out_words);
    sb_put(&b, "    gl_Position = vec4(uintBitsToFloat(mgl_draw[r + %u]), uintBitsToFloat(mgl_draw[r + %u]),"
               " uintBitsToFloat(mgl_draw[r + %u]), uintBitsToFloat(mgl_draw[r + %u]));\n",
           gp->out_pos_off + 0, gp->out_pos_off + 1, gp->out_pos_off + 2, gp->out_pos_off + 3);
    if (gp->out_has_psize)
        sb_put(&b, "    gl_PointSize = uintBitsToFloat(mgl_draw[r + %u]);\n", gp->out_psize_off);
    if (gp->out_has_layer)
        sb_put(&b, "    mgl_out_layer = mgl_draw[r + %u];\n", gp->out_layer_off);
    if (gp->out_has_vp)
        sb_put(&b, "    mgl_out_vp = mgl_draw[r + %u];\n", gp->out_vp_off);
    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char name[160];
        if (v->member[0] && strcmp(v->name, v->member))
            snprintf(name, sizeof(name), "mgl_ob%d", i);
        else
            snprintf(name, sizeof(name), "%s", v->name);
        for (GLuint e = 0; e < (v->elems > 0 ? v->elems : 1u); e++)
        {
            char arr[32] = "";
            /* an array element's words are contiguous: cols * rows * base words each */
            GLuint base = v->offset + e * v->cols * v->rows * comp_words(v->base);
            if (v->elems > 1) snprintf(arr, sizeof(arr), "[%u]", e);
            sb_put(&b, "    %s%s = ", name, arr);
            if (v->cols > 1)
            {
                char mname[64];
                snprintf(mname, sizeof(mname), "%s", glsl_type_name(v->base, v->rows, v->cols));
                sb_put(&b, "%s(", mname);
                for (GLuint c = 0; c < v->cols; c++)
                {
                    if (c) sb_put(&b, ", ");
                    sb_put(&b, "%s(", glsl_type_name(v->base, v->rows, 1));
                    for (GLuint r2 = 0; r2 < v->rows; r2++)
                    {
                        GLuint w = base + (c * v->rows + r2) * comp_words(v->base);
                        if (r2) sb_put(&b, ", ");
                        if (v->base == GL_FLOAT) sb_put(&b, "uintBitsToFloat(mgl_draw[r + %u])", w);
                        else if (v->base == GL_DOUBLE) sb_put(&b, "uint64BitsToDouble(uvec2(mgl_draw[r + %u], mgl_draw[r + %u]))", w, w + 1);
                        else sb_put(&b, "%s(mgl_draw[r + %u])", v->base == GL_INT ? "int" : "uint", w);
                    }
                    sb_put(&b, ")");
                }
                sb_put(&b, ");\n");
            }
            else if (v->rows == 1 && v->base != GL_DOUBLE)
            {
                GLuint w = base;
                sb_put(&b, "%s(mgl_draw[r + %u]);\n", v->base == GL_FLOAT ? "uintBitsToFloat" :
                           v->base == GL_BOOL ? "uint" : (v->base == GL_INT ? "int" : "uint"), w);
            }
            else
            {
                sb_put(&b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r2 = 0; r2 < v->rows; r2++)
                {
                    GLuint w = base + r2 * comp_words(v->base);
                    if (r2) sb_put(&b, ", ");
                    if (v->base == GL_FLOAT) sb_put(&b, "uintBitsToFloat(mgl_draw[r + %u])", w);
                    else if (v->base == GL_DOUBLE) sb_put(&b, "uint64BitsToDouble(uvec2(mgl_draw[r + %u], mgl_draw[r + %u]))", w, w + 1);
                    else sb_put(&b, "%s(mgl_draw[r + %u])", v->base == GL_INT ? "int" : "uint", w);
                }
                sb_put(&b, ");\n");
            }
        }
    }
    sb_put(&b, "}\n");

    return sb_take(&b);
}

/* ------------------------------------------------------------------ link */

/* assign the word offsets of the emulation records (shared by the capture variant, the
 * geometry kernel and the pass-through vertex function) */
static void geom_layout(MGLGeomProgram *gp)
{
    int i;

    /* input record layout */
    {
        GLuint off = 0;
        gp->in_pos_off = off; off += 4;
        if (gp->in_has_psize) { gp->in_psize_off = off; off += 1; }
        if (gp->in_has_clip) { gp->in_clip_off = off; off += gp->in_clip_size; }
        if (gp->in_has_cull) { gp->in_cull_off = off; off += gp->in_clip_size; }
        for (i = 0; i < gp->n_in; i++) { gp->in_v[i].offset = off; off += gp->in_v[i].words; }
        gp->in_words = off;
    }
    /* output record layout */
    {
        GLuint off = 2;     /* tag + sequence */
        gp->out_pos_off = off; off += 4;
        if (gp->out_has_psize) { gp->out_psize_off = off; off += 1; }
        if (gp->out_has_layer) { gp->out_layer_off = off; off += 1; }
        if (gp->out_has_vp) { gp->out_vp_off = off; off += 1; }
        if (gp->out_has_prim) { gp->out_prim_off = off; off += 1; }
        if (gp->out_has_clip) { gp->out_clip_off = off; off += gp->out_clip_size; }
        if (gp->out_has_cull) { gp->out_cull_off = off; off += gp->out_clip_size; }
        for (i = 0; i < gp->n_out; i++) { gp->out_v[i].offset = off; off += gp->out_v[i].words; }
        gp->out_words = off;
    }
}

static const char *geom_gs_source(GLMContext ctx, MGLProgramObj *obj)
{
    for (GLuint i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (s && s->glm_type == _GEOMETRY_SHADER && s->src && s->src[0])
            return s->src;
    }
    return NULL;
}

static const char *geom_vs_source(GLMContext ctx, MGLProgramObj *obj)
{
    for (GLuint i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (s && s->glm_type == _VERTEX_SHADER && s->src && s->src[0])
            return s->src;
    }
    return NULL;
}

static GLuint geom_declared_clip_words(const char *src, const char *name)
{    /* the largest explicitly declared array size of gl_ClipDistance / gl_CullDistance */
    const char *p = src;
    GLuint best = 0;
    while ((p = strstr(p, name)))
    {
        const char *b = p + strlen(name);
        while (*b && isspace((unsigned char)*b)) b++;
        if (*b == '[')
        {
            unsigned v = 0;
            if (sscanf(b + 1, "%u", &v) == 1 && v > best) best = v;
        }
        p += strlen(name);
    }
    return best;
}

/* build the emulation state for one program's geometry stage; a failure here fails the link */
/* match every geometry-stage input varying against the vertex stage outputs of `vs_program`
 * and record the producing expression (the name the capture variant stores).  Returns false
 * with a message in `log` when an input is not written or its type differs. */
static bool geom_derive_exprs(MGLGeomProgram *gp, Program *vs_program, char **log)
{
    MGLGeomVar vs_out[MGL_GEOM_MAX_VARYINGS];
    int n_vs_out = 0;
    SpvModule *vm = NULL;
    const char *e2 = NULL;
    bool ok = false;

    if (!vs_program->spirv[_VERTEX_SHADER].ir)
    {
        mgl_append_log(log, "ERROR: geometry: no vertex stage to reflect the inputs from\n");
        return false;
    }
    vm = spv_parse(vs_program->spirv[_VERTEX_SHADER].ir, vs_program->spirv[_VERTEX_SHADER].size, &e2);
    if (!vm || !geom_reflect_stage(vm, false, vs_out, &n_vs_out, log))
    {
        mgl_append_log(log, "ERROR: geometry: could not reflect the vertex stage outputs\n");
        goto done;
    }
    for (int i = 0; i < gp->n_in; i++)
    {
        int j = geom_find_location(vs_out, n_vs_out, gp->in_v[i].location);
        if (j < 0)
        {
            /* GL leaves such an input undefined (its value may be anything): keep the
             * link valid — the record slot exists, the capture pass just never writes it */
            mgl_append_log(log, "WARNING: geometry: input '%s' at location %u is not written by the vertex stage\n",
                           gp->in_v[i].name, gp->in_v[i].location);
            gp->in_v[i].expr[0] = 0;
            continue;
        }
        if (vs_out[j].rows != gp->in_v[i].rows || vs_out[j].cols != gp->in_v[i].cols ||
            vs_out[j].base != gp->in_v[i].base || vs_out[j].elems != gp->in_v[i].elems)
        {
            mgl_append_log(log, "ERROR: geometry: type mismatch for '%s' (location %u)\n",
                           gp->in_v[i].name, gp->in_v[i].location);
            if (getenv("MGL_GEOM_TRACE"))
                fprintf(stderr, "MGL geom trace: mismatch at loc %u: GS(%s base=%u r=%u c=%u e=%u) "
                                "VS(%s base=%u r=%u c=%u e=%u)\n",
                        gp->in_v[i].location, gp->in_v[i].name, gp->in_v[i].base, gp->in_v[i].rows,
                        gp->in_v[i].cols, gp->in_v[i].elems,
                        vs_out[j].name, vs_out[j].base, vs_out[j].rows, vs_out[j].cols, vs_out[j].elems);
            goto done;
        }
        snprintf(gp->in_v[i].expr, sizeof(gp->in_v[i].expr), "%s%s%s", vs_out[j].name,
                 vs_out[j].member[0] ? "." : "", vs_out[j].member);
    }
    ok = true;

done:
    spv_free(vm);
    return ok;
}

bool mgl_geom_compile_stage(GLMContext ctx, Program *program, void *obj_p, int stage,                            const uint32_t *words, size_t nwords, const char *entry, char **log)
{
    MGLProgramObj *obj = (MGLProgramObj *)obj_p;
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, true);
    SpvModule *m = NULL;
    const char *err = NULL;
    bool ok = false;
    int i;

    if (!gp)
    {
        mgl_append_log(log, "ERROR: geometry: program table exhausted\n");
        return false;
    }
    geom_free_prog(ctx, gp);
    gp->name = program->name;
    gp->program = getProgram(ctx, program->name);

    m = spv_parse(words, nwords, &err);
    if (!m)
    {
        mgl_append_log(log, "ERROR: geometry: %s\n", err ? err : "SPIR-V parse failed");
        return false;
    }

    gp->in_type = 0;
    gp->out_type = 0;
    gp->in_verts = 0;
    gp->max_vertices = 0;
    gp->invocations = 1;
    bool have_in = false, have_out = false;
    for (i = 0; i < m->n_modes; i++)
    {
        switch (m->modes[i].mode)
        {
            case SpvExecutionModeInputPoints:           gp->in_type = GL_POINTS; gp->in_verts = 1; have_in = true; break;
            case SpvExecutionModeInputLines:            gp->in_type = GL_LINES; gp->in_verts = 2; have_in = true; break;
            case SpvExecutionModeInputLinesAdjacency:   gp->in_type = GL_LINES_ADJACENCY; gp->in_verts = 4; have_in = true; break;
            case SpvExecutionModeTriangles:             gp->in_type = GL_TRIANGLES; gp->in_verts = 3; have_in = true; break;
            case SpvExecutionModeInputTrianglesAdjacency: gp->in_type = GL_TRIANGLES_ADJACENCY; gp->in_verts = 6; have_in = true; break;
            case SpvExecutionModeOutputPoints:          gp->out_type = GL_POINTS; have_out = true; break;
            case SpvExecutionModeOutputLineStrip:       gp->out_type = GL_LINE_STRIP; have_out = true; break;
            case SpvExecutionModeOutputTriangleStrip:   gp->out_type = GL_TRIANGLE_STRIP; have_out = true; break;
            case SpvExecutionModeOutputVertices:        gp->max_vertices = m->modes[i].value; break;
            case SpvExecutionModeInvocations:           gp->invocations = m->modes[i].value ? m->modes[i].value : 1; break;
            default: break;
        }
    }

    if (!have_in || !have_out)     /* GL_POINTS is 0, so test the flags, not the type */
    {
        mgl_append_log(log, "ERROR: geometry: missing input or output primitive type\n");
        goto done;
    }
    if (gp->max_vertices > (GLuint)256)
    {
        mgl_append_log(log, "ERROR: geometry: max_vertices %u exceeds the limit %u\n",
                       gp->max_vertices, (unsigned)256);
        goto done;
    }
    if (gp->invocations > (GLuint)32)
    {
        mgl_append_log(log, "ERROR: geometry: invocations %u exceeds the limit %u\n",
                       gp->invocations, (unsigned)32);
        goto done;
    }

    if (!geom_reflect_stage(m, true, gp->in_v, &gp->n_in, log)) goto done;
    if (!geom_reflect_stage(m, false, gp->out_v, &gp->n_out, log)) goto done;

    /* a named output interface block ("out B { vec4 c; } b;") is rewritten to a plain
     * struct + instance in the generated kernel (the body keeps writing b.c) and its
     * members are re-emitted as ordinary location-addressed varyings on the raster side,
     * where the fragment stage's matching input block links by location */

    /* the entry names are needed even when the emulation is deferred */
    gp->kernel_entry = xstrdup("mgl_geom_gs");
    gp->capture_entry = xstrdup("main0");
    gp->draw_entry = xstrdup("main0");

    /* a separable geometry-only program has no vertex stage of its own: the reflection and
     * the record layout are kept (so the program queries work), the emulation is built at
     * the first draw, when the pipeline contributes the vertex stage.  The layout uses the
     * default clip distance size here; geom_build() refines it once the vertex stage is
     * known.  The builtin scan below still runs for such a program: the input record always
     * carries gl_Position, and the capture variant only stores what in_has_* says. */
    geom_layout(gp);
    gp->deferred = !program->shader_slots[_VERTEX_SHADER];

    /* builtins: driven by what the shader source mentions */
    {
        const char *src = geom_gs_source(ctx, obj);
        char *decompiled = NULL;
        if (!src)
        {
            decompiled = geom_spirv_to_glsl(words, nwords, 460, log);
            src = decompiled;
            if (!src)
            {
                mgl_append_log(log, "ERROR: geometry: no geometry shader source available\n");
                goto done;
            }
        }

        gp->in_has_pos = true;
        gp->in_has_psize = strstr(src, "gl_PointSize") != NULL;
        gp->in_has_clip = strstr(src, "gl_ClipDistance") != NULL;
        gp->in_has_cull = strstr(src, "gl_CullDistance") != NULL;
        gp->out_has_pos = true;
        gp->out_has_psize = strstr(src, "gl_PointSize") != NULL;
        gp->out_has_layer = strstr(src, "gl_Layer") != NULL;
        gp->out_has_vp = strstr(src, "gl_ViewportIndex") != NULL;
        gp->out_has_prim = strstr(src, "gl_PrimitiveID") != NULL;
        gp->out_has_clip = strstr(src, "gl_ClipDistance") != NULL;
        gp->out_has_cull = strstr(src, "gl_CullDistance") != NULL;

        {
            const char *vs_src = geom_vs_source(ctx, obj);
            char *vs_dec = NULL;
            GLuint n;
            if (!vs_src && program->spirv[_VERTEX_SHADER].ir)
            {
                vs_dec = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                            program->spirv[_VERTEX_SHADER].size, 460, log);
                vs_src = vs_dec;
            }
            if (!vs_src)
            {
                /* a separable geometry-only program has no vertex stage of its own: the clip
                 * sizes default to the maximum now and geom_build() refines them from the
                 * pipeline's vertex stage at the first draw */
                if (!gp->deferred)
                {
                    mgl_append_log(log, "ERROR: geometry: no vertex shader to capture from\n");
                    free(vs_dec);
                    free(decompiled);
                    goto done;
                }
                gp->in_clip_size = gp->in_has_clip ? MGL_GEOM_MAX_CLIP : 0;
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ? MGL_GEOM_MAX_CLIP : 0;
            }
            else
            {
                n = geom_declared_clip_words(vs_src, "gl_ClipDistance");
                gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
                n = geom_declared_clip_words(vs_src, "gl_CullDistance");
                if (gp->in_has_cull)
                {
                    GLuint nc2 = n ? n : MGL_GEOM_MAX_CLIP;
                    if (nc2 < gp->in_clip_size) gp->in_clip_size = nc2;
                }
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ?
                                    (gp->in_clip_size ? gp->in_clip_size : MGL_GEOM_MAX_CLIP) : 0;
            }
            free(vs_dec);
        /* the clip size of the real vertex stage is only known now */
        geom_layout(gp);
        }
        free(decompiled);
    }

    if (gp->deferred)
    {
        /* the emulation is built at the first draw, when the pipeline contributes the vertex
         * stage; everything above is reflection only */
        ok = true;
        goto done;
    }

    /* total output component limit */
    {
        GLuint comps = 0;
        for (i = 0; i < gp->n_out; i++)
            comps += gp->out_v[i].words * gp->max_vertices;
        comps += (4 + 1 + 1 + 1 + 1 + 2 * gp->out_clip_size) * gp->max_vertices;
        if (comps > (GLuint)2048)
        {
            mgl_append_log(log, "ERROR: geometry: %u total output components exceeds the limit %u\n",
                           comps, (unsigned)2048);
            goto done;
        }
    }

    gp->streams = 1;
    for (i = 0; i < gp->n_out; i++)
        if (gp->out_v[i].stream + 1 > gp->streams) gp->streams = gp->out_v[i].stream + 1;

    /* a tessellation stage in front of the geometry shader: the tess domain's TES-output
     * capture supplies the records, so there is no vertex stage to derive the inputs'
     * producing expressions from (the draw-time import repacks the capture instead) */
    gp->from_tess = (program->shader_slots[_TESS_EVALUATION_SHADER] != NULL);
    if (gp->from_tess && gp->in_has_clip)
    {
        mgl_append_log(log, "ERROR: geometry: gl_ClipDistance of a tessellation stage is not captured, "
                            "so the geometry stage cannot read gl_in[].gl_ClipDistance\n");
        goto done;
    }

    /* writer side: the vertex outputs feeding each input location.  A separable
     * geometry-only program has no vertex stage to reflect here; geom_ensure() does it from
     * the pipeline's composite once that is available. */
    if (!gp->deferred && !gp->from_tess)
    {
        if (!geom_derive_exprs(gp, program, log))
            goto done;
    }

    ok = true;

done:
    spv_free(m);
    if (!ok)
    {
        geom_fail(ctx, gp, "geometry shader emulation could not be built");
        return false;
    }
    return true;
}

/* patch a `[[user(locnN)]]` attribute of an MSL struct member into a Metal builtin
 * attribute (used to forward gl_Layer / gl_ViewportIndex).  Returns a newly allocated
 * string (the caller frees the original), or NULL if the member was not found. */
static char *msl_promote_attribute(const char *msl, const char *member, const char *attr)
{
    const char *p = strstr(msl, member);
    const char *u, *close;
    char *out;
    size_t head, tail;

    if (!p) return NULL;
    u = strstr(p, "user(locn");
    if (!u || (size_t)(u - p) > 96) return NULL;
    close = strchr(u, ')');
    if (!close) return NULL;
    close++;                        /* include the ')' */

    head = (size_t)(u - msl);
    tail = strlen(close);
    out = xmalloc(head + strlen(attr) + tail + 1);
    memcpy(out, msl, head);
    memcpy(out + head, attr, strlen(attr));
    memcpy(out + head + strlen(attr), close, tail + 1);
    return out;
}

/* post link: translate the three emulation shaders to MSL, replace the program's vertex
 * stage with the pass-through function */
/* SPIRV-Cross passes loose (explicit-location) uniforms of the default block as
 * `constant T&` arguments of the entry point, but a user function (our renamed shader
 * main) takes them as `thread const T&` — which Metal rejects.  Turn those parameters
 * into value parameters. */
static bool msl_value_parameters(char *msl, const char *entry)
{
    char *def = strstr(msl, entry);
    char *p, *close;
    size_t removed = 0;

    if (!def) return false;
    p = strchr(def, '(');
    if (!p) return false;

    /* find the end of the parameter list */
    close = p + 1;
    while (*close && *close != ')') close++;
    if (!*close) return false;

    /* rewrite "thread const T& name" as "T name" inside the list */
    p = p + 1;
    while (p < close)
    {
        if (!strncmp(p, "thread const ", 13))
        {
            char *amp = p + 13;
            while (amp < close && *amp != '&' && *amp != ',') amp++;
            if (amp < close && *amp == '&')
            {
                size_t tail = strlen(amp) + 1;
                /* drop the "thread const " qualifier */
                memmove(p, p + 13, strlen(p + 13) + 1);
                close -= 13;
                amp -= 13;
                /* drop the '&' */
                memmove(amp, amp + 1, tail);
                close -= 1;
                removed += 14;
                continue;
            }
        }
        p++;
    }
    return removed > 0;
}

/* build (or rebuild) the emulation for one geometry stage; `program` is the program that
 * owns the vertex + geometry stages (a linked program, or a pipeline's composite) */
static void geom_build(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);
    char *gs_src = NULL, *vs_src = NULL, *glsl, *msl;
    uint32_t *ir = NULL;
    size_t words = 0;
    char *log = NULL;
    int pins_gl[8], pins_msl[8];
    GeomPin pins2[8];
    int npins2 = 0;

#define GEOM_FAIL(...)  do { geom_fail(ctx, gp, __VA_ARGS__); \
                             if (!gp->deferred) obj->link_status = false; goto done; } while (0)

    if (!gp || !obj) return;
    if (!program->shader_slots[_GEOMETRY_SHADER]) return;

    gp->gs_exec = program->shader_slots[_GEOMETRY_SHADER];

    /* a tessellation stage in front of the geometry shader: the tess domain's TES-output
     * capture producer (mgl_ext_tess.h) feeds the geometry kernel instead of a vertex capture
     * variant — the records are repacked into this program's input record at draw time
     * (mgl_geom_route / the import kernel in MGLRenderer+Geometry.m) */
    gp->from_tess = (program->shader_slots[_TESS_EVALUATION_SHADER] != NULL);
    if (gp->from_tess && gp->in_has_clip)
        GEOM_FAIL("gl_ClipDistance of a tessellation stage is not captured: "
                  "the geometry stage cannot read gl_in[].gl_ClipDistance");
    if (!gp->from_tess &&
        (program->shader_slots[_TESS_CONTROL_SHADER] || program->shader_slots[_TESS_EVALUATION_SHADER]))
        GEOM_FAIL("a tessellation control shader without an evaluation shader cannot feed "
                  "the geometry stage");

    {
        const char *src = geom_gs_source(ctx, obj);
        gs_src = xstrdup(src ? src : "");
        if (!gs_src[0])
        {
            free(gs_src);
            gs_src = geom_spirv_to_glsl(program->spirv[_GEOMETRY_SHADER].ir,
                                        program->spirv[_GEOMETRY_SHADER].size, 460, &log);
            if (!gs_src) GEOM_FAIL("no geometry shader source: %s", log ? log : "?");
        }
    }

    if (gp->from_tess)
    {
        /* no vertex stage is captured: the tess domain publishes the records (with its own
         * member layout) and the import kernel repacks them into gp's input record */
        gp->vid_absolute = false;
        gp->iid_absolute = false;
        gp->in_clip_size = 0;
        gp->capture_entry = xstrdup("main0");
        geom_layout(gp);
        goto build;          /* skip the capture variant, the draw variant is built below */
    }

    {
        const char *src = geom_vs_source(ctx, obj);
        vs_src = xstrdup(src ? src : "");
        if (!vs_src[0])
        {
            free(vs_src);
            vs_src = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                        program->spirv[_VERTEX_SHADER].size, 460, &log);
            if (!vs_src) GEOM_FAIL("no vertex shader source: %s", log ? log : "?");
        }
    }

    /* ---- 1. vertex capture variant ---- */
    {
        const char *e2 = NULL;
        SpvModule *p_vs = spv_parse(program->spirv[_VERTEX_SHADER].ir,
                                    program->spirv[_VERTEX_SHADER].size, &e2);

        glsl = geom_gen_capture_glsl(ctx, gp, vs_src, p_vs, &log);
        if (!glsl) GEOM_FAIL("vertex capture variant generation failed");
        geom_dump("capture.glsl", glsl);
        if (!geom_glsl_to_spirv(ctx, GL_VERTEX_SHADER, glsl, 430, &ir, &words, &log))
            GEOM_FAIL("vertex capture variant compile failed: %s", log ? log : "?");
        free(glsl);
        {
            SpvModule *c_vs = spv_parse(ir, words, &e2);
            int span;

            npins2 = 0;
            span = geom_build_pins(p_vs, c_vs, true, pins2, &npins2, 8);
            if (span > MGL_GEOM_KERNEL_USER_SLOTS)
            {
                spv_free(c_vs);
                spv_free(p_vs);
                GEOM_FAIL("vertex stage resources span %d argument slots; the compute "
                          "emulation supports %d (MGL_GEOM_KERNEL_USER_SLOTS)",
                          span, (int)MGL_GEOM_KERNEL_USER_SLOTS);
            }
            spv_free(c_vs);
        }
        spv_free(p_vs);
    }
    pins_gl[0] = MGL_GEOM_SSBO_VSOUT;
    pins_msl[0] = MGL_GEOM_MSL_VSOUT;
    msl = geom_spirv_to_msl(ctx, GL_VERTEX_SHADER, ir, words, gp->capture_entry, true, &log,
                            SpvExecutionModelVertex, pins_gl, pins_msl, 1, pins2, npins2);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("vertex capture variant MSL failed: %s", log ? log : "?");
    /* the renamed vertex main is called from the generated entry point */
    msl_value_parameters(msl, "mgl_geom_vs_main(");
    gp->capture_msl = msl;
    geom_dump("capture.msl", msl);
    gp->vid_absolute = strstr(msl, "[[base_vertex]]") == NULL;
    gp->iid_absolute = strstr(msl, "[[base_instance]]") == NULL;

    /* the capture function is not part of the program's stages: compile it now */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
    {
        void *lib = NULL, *fn = NULL;
        char *err = NULL;
        if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, gp->capture_entry, &lib, &fn, &err))
        {
            GEOM_FAIL("vertex capture variant Metal compile failed: %s", err ? err : "?");
        }
        gp->capture_library = lib;
        gp->capture_function = fn;
    }
    else
        GEOM_FAIL("no Metal compile hook available");

    /* ---- 2. geometry kernel ---- */
build:
    /* the program's own geometry-stage module owns the resource order the renderer's
     * buffer map enumerates; pin this kernel's user resources to those positions */
    {
        const char *e2 = NULL;
        SpvModule *p_gs = spv_parse(program->spirv[_GEOMETRY_SHADER].ir,
                                    program->spirv[_GEOMETRY_SHADER].size, &e2);
        SpvModule *k_gs = NULL;

        glsl = geom_gen_kernel_glsl(ctx, gp, gs_src, p_gs, &log);
        if (!glsl) GEOM_FAIL("kernel generation failed");
        geom_dump("kernel.glsl", glsl);
        if (!geom_glsl_to_spirv(ctx, GL_COMPUTE_SHADER, glsl, 450, &ir, &words, &log))
            GEOM_FAIL("kernel compile failed: %s", log ? log : "?");
        geom_record_tex_arrays(ir, words, gp);
        free(glsl);
        k_gs = spv_parse(ir, words, &e2);
        {
            /* the kernel's own buffers live at fixed Metal slots (MGL_GEOM_MSL_*): the
             * stage's program resources must fit in the slots below them */
            int span = geom_build_pins(p_gs, k_gs, true, pins2, &npins2, 8);

            if (span > MGL_GEOM_KERNEL_USER_SLOTS)
                GEOM_FAIL("geometry stage resources span %d argument slots; the compute "
                          "emulation supports %d (MGL_GEOM_KERNEL_USER_SLOTS)",
                          span, (int)MGL_GEOM_KERNEL_USER_SLOTS);
        }
        spv_free(k_gs);
        spv_free(p_gs);
    }
    pins_gl[0] = MGL_GEOM_SSBO_PRIM;     pins_msl[0] = MGL_GEOM_MSL_PRIM;
    pins_gl[1] = MGL_GEOM_SSBO_VSOUT;    pins_msl[1] = MGL_GEOM_MSL_VSOUT;
    pins_gl[2] = MGL_GEOM_SSBO_HDR;      pins_msl[2] = MGL_GEOM_MSL_HDR;
    pins_gl[3] = MGL_GEOM_SSBO_OUTV;     pins_msl[3] = MGL_GEOM_MSL_OUTV;
    pins_gl[4] = MGL_GEOM_SSBO_IDX;      pins_msl[4] = MGL_GEOM_MSL_IDX;
    pins_gl[5] = MGL_GEOM_SSBO_CTL;      pins_msl[5] = MGL_GEOM_MSL_CTL;
    pins_gl[6] = MGL_GEOM_SSBO_CNT;      pins_msl[6] = MGL_GEOM_MSL_CNT;
    msl = geom_spirv_to_msl(ctx, GL_COMPUTE_SHADER, ir, words, gp->kernel_entry, false, &log,
                            SpvExecutionModelGLCompute, pins_gl, pins_msl, 7, pins2, npins2);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("kernel MSL failed: %s", log ? log : "?");

    /* the kernel must have a stable name: SPIRV-Cross always emits "kernel void main0(" */
    {
        const char *want = "kernel void main0(";
        char *e = strstr(msl, want);
        if (!e) GEOM_FAIL("kernel MSL has no main0 entry point");
        {
            size_t rest = strlen(e + strlen(want));
            memmove(e + strlen("kernel void mgl_geom_gs("), e + strlen(want), rest + 1);
            memcpy(e, "kernel void mgl_geom_gs(", strlen("kernel void mgl_geom_gs("));
        }
    }
    msl_value_parameters(msl, "mgl_gs_main(");
    gp->kernel_msl = msl;
    geom_dump("kernel.msl", msl);

    /* the geometry stage executable is the kernel */
    free(program->spirv[_GEOMETRY_SHADER].msl_str);
    program->spirv[_GEOMETRY_SHADER].msl_str = xstrdup(gp->kernel_msl);
    free(program->spirv[_GEOMETRY_SHADER].entry_point);
    program->spirv[_GEOMETRY_SHADER].entry_point = xstrdup(gp->kernel_entry);
    if (program->shader_slots[_GEOMETRY_SHADER])
        program->shader_slots[_GEOMETRY_SHADER]->entry_point = xstrdup(gp->kernel_entry);

    /* ---- 3. pass-through raster vertex function ---- */
    glsl = geom_gen_draw_glsl(ctx, gp, &log);
    if (!glsl) GEOM_FAIL("pass-through vertex function generation failed");
    geom_dump("draw.glsl", glsl);
    if (!geom_glsl_to_spirv(ctx, GL_VERTEX_SHADER, glsl, 460, &ir, &words, &log))
        GEOM_FAIL("pass-through vertex function compile failed: %s", log ? log : "?");
    free(glsl);
    pins_gl[0] = MGL_GEOM_SSBO_DRAW;     pins_msl[0] = MGL_GEOM_MSL_DRAW;
    msl = geom_spirv_to_msl(ctx, GL_VERTEX_SHADER, ir, words, gp->draw_entry, false, &log,
                            SpvExecutionModelVertex, pins_gl, pins_msl, 1, NULL, 0);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("pass-through vertex function MSL failed: %s", log ? log : "?");

    /* gl_Layer / gl_ViewportIndex become Metal rasterization controls */
    if (gp->out_has_layer)
    {
        char *n = msl_promote_attribute(msl, "mgl_out_layer", "render_target_array_index");
        if (!n) { free(msl); GEOM_FAIL("could not forward gl_Layer to render_target_array_index"); }
        free(msl);
        msl = n;
    }
    if (gp->out_has_vp)
    {
        char *n = msl_promote_attribute(msl, "mgl_out_vp", "viewport_array_index");
        if (!n) { free(msl); GEOM_FAIL("could not forward gl_ViewportIndex to viewport_array_index"); }
        free(msl);
        msl = n;
    }

    /* the rstate clip-space fixup belongs to the last stage before rasterization */
    {
        char *patched = mgl_rstate_patch_vertex_msl(msl, gp->draw_entry);
        if (patched)
        {
            free(msl);
            msl = patched;
        }
    }

    /* a point size is only meaningful for point output; some Metal implementations reject a
     * pipeline whose vertex function writes point_size together with an explicit
     * inputPrimitiveTopology (which layered / viewport-indexed output needs) */
    if (gp->out_type != GL_POINTS)
    {
        char *psz = strstr(msl, "[[point_size]]");
        if (psz)
        {
            size_t len = strlen("[[point_size]]");
            memmove(psz, psz + len, strlen(psz + len) + 1);
        }
    }

    gp->draw_msl = msl;
    geom_dump("draw.msl", msl);
    gp->linked = true;

    /* compile the pass-through function: it stands in for the vertex stage during the
     * raster pass.  A pipeline's composite is rebuilt from the stage programs on every
     * draw, so the substitution is made per draw rather than once at link time */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
    {
        void *lib = NULL, *fn = NULL;
        char *err = NULL;

        if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, gp->draw_entry, &lib, &fn, &err))
        {
            free(err);
            GEOM_FAIL("pass-through vertex function Metal compile failed: %s", err ? err : "?");
        }
        if (gp->draw_library) ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_library);
        if (gp->draw_function) ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_function);
        gp->draw_library = lib;
        gp->draw_function = fn;
    }
    else
        GEOM_FAIL("no Metal compile hook available");

#undef GEOM_FAIL

done:
    free(gs_src);
    free(vs_src);
    free(ir);
    free(log);
}

/* post-link hook: the emulation is built here unless the program has no vertex stage of
 * its own (a separable geometry-only program gets it from the pipeline at draw time) */
void mgl_geom_post_link(GLMContext ctx, Program *program)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);

    if (!gp || !obj) return;
    if (!program->shader_slots[_GEOMETRY_SHADER]) return;

    gp->gs_exec = program->shader_slots[_GEOMETRY_SHADER];
    gp->deferred = !program->shader_slots[_VERTEX_SHADER];
    if (gp->deferred) return;               /* separable: built at first draw */

    geom_build(ctx, gp, program);
}

/* make sure the emulation exists for the program that is about to draw */
static bool geom_ensure(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    if (!gp || !program) return false;
    if (gp->linked) return true;
    if (!program->shader_slots[_VERTEX_SHADER]) return false;

    /* a relink of the geometry program (glUseProgramStages relinks its stage programs)
     * leaves the slot reflected-empty: redo it from the program's own geometry stage */
    if ((!gp->out_type || !gp->out_words) && ctx->state.program_pipeline)
    {
        Program *gs = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_GEOMETRY_SHADER];
        if (gs && gs->spirv[_GEOMETRY_SHADER].ir)
            mgl_geom_compile_stage(ctx, gs, MGL_PROGRAM_OBJ(gs), _GEOMETRY_SHADER,
                                   gs->spirv[_GEOMETRY_SHADER].ir, gs->spirv[_GEOMETRY_SHADER].size,
                                   "main", NULL);
    }

    /* a separable geometry-only program reflected its own stage only: the inputs' producing
     * vertex outputs and the real clip sizes come from the composite the pipeline builds,
     * which carries the vertex stage.  geom_build() reads them from gp->in_v[].expr. */
    if (gp->deferred && program->spirv[_VERTEX_SHADER].ir && gp->n_in)
    {
        char *llog = NULL;

        if (!geom_derive_exprs(gp, program, &llog))
        {
            geom_fail(ctx, gp, "%s", llog ? llog : "geometry inputs could not be matched");
            free(llog);
            return false;
        }
        free(llog);
        {
            const char *vs_src = geom_vs_source(ctx, MGL_PROGRAM_OBJ(program));
            char *vs_dec = NULL;
            GLuint n;

            if (!vs_src)
            {
                vs_dec = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                            program->spirv[_VERTEX_SHADER].size, 460, &llog);
                vs_src = vs_dec;
            }
            if (vs_src)
            {
                n = geom_declared_clip_words(vs_src, "gl_ClipDistance");
                gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
                n = geom_declared_clip_words(vs_src, "gl_CullDistance");
                if (gp->in_has_cull)
                {
                    GLuint nc2 = n ? n : MGL_GEOM_MAX_CLIP;
                    if (nc2 < gp->in_clip_size) gp->in_clip_size = nc2;
                }
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ?
                                    (gp->in_clip_size ? gp->in_clip_size : MGL_GEOM_MAX_CLIP) : 0;
                geom_layout(gp);
            }
            free(vs_dec);
            free(llog);
        }
    }

    gp->deferred = false;
    geom_build(ctx, gp, program);
    return gp->linked;
}

bool mgl_geom_link_failed(GLMContext ctx, Program *program, const char **reason)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    if (!program->shader_slots[_GEOMETRY_SHADER]) return false;
    if (!gp || gp->linked || gp->deferred) return false;
    if (reason) *reason = gp->fail ? gp->fail : "geometry stage failed";
    return true;
}

/* like mgl_program_check_xfb_link: the post-link hook builds the emulation; a failure there
 * fails the link and is reported through the program info log */
bool mgl_program_check_geom_link(GLMContext ctx, Program *program)
{
    const char *glog = NULL;
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);

    if (!mgl_geom_link_failed(ctx, program, &glog)) return true;

    obj->link_status = false;
    if (glog && glog[0])
    {
        size_t len = strlen(glog) + (obj->info_log ? strlen(obj->info_log) : 0) + 8;
        char *old_log = obj->info_log;
        obj->info_log = malloc(len);
        if (obj->info_log)
            snprintf(obj->info_log, len, "%s%s%s", old_log ? old_log : "",
                     old_log ? "\n" : "", glog);
        free(old_log);
    }
    return false;
}

/* ------------------------------------------------------------------ program queries */

bool mgl_geom_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    if (!gp || !gp->linked) return false;

    switch (pname)
    {
        case GL_GEOMETRY_VERTICES_OUT:  *params = (GLint)gp->max_vertices; return true;
        case GL_GEOMETRY_INPUT_TYPE:    *params = (GLint)gp->in_type; return true;
        case GL_GEOMETRY_OUTPUT_TYPE:   *params = (GLint)gp->out_type; return true;
        case GL_GEOMETRY_SHADER_INVOCATIONS: *params = (GLint)gp->invocations; return true;
        default: return false;
    }
}

/* ------------------------------------------------------------------ draw */

static MGLGeomProgram *geom_for_program(GLMContext ctx, Program *p)
{
    if (!p || !p->shader_slots[_GEOMETRY_SHADER]) return NULL;
    return geom_prog_slot(ctx, p, false);
}

MGLGeomProgram *mgl_geom_current(GLMContext ctx)
{
    MGLGeomProgram *gp;

    if (ctx->state.program)
    {
        gp = geom_for_program(ctx, ctx->state.program);
        if (gp) return gp;
    }
    if (ctx->state.program_pipeline)
    {
        /* a pipeline resolves through the program that owns the geometry stage (the
         * composite it builds for the renderer is a fresh object every time) */
        Program *gs = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_GEOMETRY_SHADER];
        if (gs)
        {
            gp = geom_for_program(ctx, gs);
            if (gp) return gp;
        }
    }
    return NULL;
}

bool mgl_geom_present(GLMContext ctx)
{
    if (!ctx->state.ext_geom) return false;
    if (ctx->state.program && ctx->state.program->shader_slots[_GEOMETRY_SHADER]) return true;
    if (ctx->state.program_pipeline)
    {
        Program *c = mgl_program_pipeline_composite(ctx);
        if (c && c->shader_slots[_GEOMETRY_SHADER]) return true;
    }
    return false;
}

/* does the draw mode feed the geometry stage's input primitive type? */
static bool geom_mode_matches(GLenum in_type, GLenum mode)
{
    switch (in_type)
    {
        case GL_POINTS:                 return mode == GL_POINTS;
        case GL_LINES:                  return mode == GL_LINES || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP;
        case GL_LINES_ADJACENCY:        return mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY;
        case GL_TRIANGLES:              return mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN;
        case GL_TRIANGLES_ADJACENCY:    return mode == GL_TRIANGLES_ADJACENCY || mode == GL_TRIANGLE_STRIP_ADJACENCY;
        default: return false;
    }
}

/* number of input primitives of `count` vertices (per instance) */
static uint64_t geom_prim_count(GLenum gs_in, GLenum mode, uint64_t n)
{
    switch (mode)
    {
        case GL_POINTS:                    return n;
        case GL_LINES:                     return n / 2;
        case GL_LINE_STRIP:                return n >= 2 ? n - 1 : 0;
        case GL_LINE_LOOP:                 return n >= 2 ? n : 0;
        case GL_LINES_ADJACENCY:           return n / 4;
        case GL_LINE_STRIP_ADJACENCY:      return n >= 4 ? n - 3 : 0;
        case GL_TRIANGLES:                 return n / 3;
        case GL_TRIANGLE_STRIP:            return n >= 3 ? n - 2 : 0;
        case GL_TRIANGLE_FAN:              return n >= 3 ? n - 2 : 0;
        case GL_TRIANGLES_ADJACENCY:       return n / 6;
        case GL_TRIANGLE_STRIP_ADJACENCY:  return n >= 6 ? (n - 4) / 2 : 0;
        default: return 0;
    }
    (void)gs_in;
}

/* assemble one sequence of `n` vertex slots into input primitives */
static uint64_t geom_assemble(GLMContext ctx, MGLGeomProgram *gp, GLenum mode, const uint32_t *idx,
                              uint64_t n, uint32_t base, uint32_t *out, uint64_t cap)
{
    uint64_t np = 0;
    bool provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;
    GLuint k = gp->in_verts;

#define EMIT1(a)      do { if (np < cap) { out[np * k + 0] = idx[a] + base; np++; } } while (0)
#define EMIT2(a, b)   do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; np++; } } while (0)
#define EMIT3(a, b, c) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                out[np * k + 2] = idx[c] + base; np++; } } while (0)
#define EMIT4(a, b, c, d) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                  out[np * k + 2] = idx[c] + base; out[np * k + 3] = idx[d] + base; np++; } } while (0)
#define EMIT6(a, b, c, d, e, f) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                        out[np * k + 2] = idx[c] + base; out[np * k + 3] = idx[d] + base; \
                                                        out[np * k + 4] = idx[e] + base; out[np * k + 5] = idx[f] + base; np++; } } while (0)

    for (uint64_t i = 0; i + k <= n; i += (mode == GL_LINES || mode == GL_TRIANGLES ||
                                           mode == GL_LINES_ADJACENCY || mode == GL_TRIANGLES_ADJACENCY) ? k : 1)
    {
        switch (mode)
        {
            case GL_POINTS:
                EMIT1(i);
                break;
            case GL_LINES:
            case GL_LINE_STRIP:
                EMIT2(i, i + 1);
                break;
            case GL_LINE_LOOP:
                if (i + 1 < n) EMIT2(i, i + 1);
                else EMIT2(i, 0);
                break;
            case GL_LINES_ADJACENCY:
            case GL_LINE_STRIP_ADJACENCY:
                EMIT4(i, i + 1, i + 2, i + 3);
                break;
            case GL_TRIANGLES:
            case GL_TRIANGLE_FAN:
                if (mode == GL_TRIANGLE_FAN)
                {
                    if (provoking_first) EMIT3(i + 1, i + 2, 0);
                    else EMIT3(0, i + 1, i + 2);
                }
                else EMIT3(i, i + 1, i + 2);
                break;
            case GL_TRIANGLE_STRIP:
                if ((i & 1) == 0) EMIT3(i, i + 1, i + 2);
                else if (provoking_first) EMIT3(i + 2, i, i + 1);
                else EMIT3(i + 1, i, i + 2);
                break;
            case GL_TRIANGLES_ADJACENCY:
                EMIT6(i, i + 1, i + 2, i + 3, i + 4, i + 5);
                break;
            case GL_TRIANGLE_STRIP_ADJACENCY:
            {
                /* GL 4.6 table 10.4 (as the CTS encodes it): even slots are triangle
                 * vertices, odd ones adjacency; odd triangles flip winding; the first and
                 * last triangles have no preceding/following adjacency vertex. The strip
                 * advances two stream vertices per triangle. */
                uint64_t T = n >= 4 ? (n - 4) / 2 : 0;

                if (np < T)
                {
                    uint64_t b = 2 * np, e = (np & 1) == 0;
                    uint64_t v0 = e ? b : b + 2, v1 = e ? b + 2 : b, v2 = b + 4;
                    uint64_t a0 = np == 0 ? 1 : b - 2;
                    uint64_t a1 = np == T - 1 ? (e ? b + 5 : b + 3) : (e ? b + 6 : b + 3);
                    uint64_t a2 = np == T - 1 ? (e ? b + 3 : b + 5) : (e ? b + 3 : b + 6);

                    EMIT6(v0, a0, v1, a1, v2, a2);
                }
                break;
            }
            default:
                break;
        }
    }

#undef EMIT1
#undef EMIT2
#undef EMIT3
#undef EMIT4
#undef EMIT6

    return np;
}

/* the largest number of input primitives `n` vertices can form (the prep kernel's regions
 * are sized by this: the primitive count of the mode ignoring primitive restart) */
uint64_t mgl_geom_worst_prims(GLenum mode, uint64_t n)
{
    return geom_prim_count(GL_TRIANGLES, mode, n);
}

/* read the index list of an indexed draw as uint32 (handles restart by skipping runs) */
static uint32_t *geom_index_list(GLMContext ctx, GLenum type, size_t offset, GLsizei count, uint32_t *restart_out)
{
    Buffer *eb = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
    uint32_t *out = xmalloc(sizeof(uint32_t) * (size_t)(count > 0 ? count : 1));
    const uint8_t *p;
    GLuint restart = (type == GL_UNSIGNED_BYTE) ? 0xff : (type == GL_UNSIGNED_SHORT) ? 0xffff : 0xffffffffu;

    *restart_out = restart;

    mgl_buffers_prepare_cpu_read(ctx, eb);
    if (!eb->data.buffer_data)
    {
        free(out);
        return NULL;
    }
    p = (const uint8_t *)eb->data.buffer_data + offset;

    for (GLsizei i = 0; i < count; i++)
    {
        if (type == GL_UNSIGNED_BYTE) out[i] = p[i];
        else if (type == GL_UNSIGNED_SHORT) out[i] = ((const uint16_t *)p)[i];
        else out[i] = ((const uint32_t *)p)[i];
    }
    return out;
}

/* the GL draw mode as the prep kernel's code (see kGeomPrepMSL) */
int mgl_geom_mode_code(GLenum mode)
{
    switch (mode)
    {
        case GL_POINTS:                   return 0;
        case GL_LINES:                    return 1;
        case GL_LINE_STRIP:               return 2;
        case GL_LINE_LOOP:                return 3;
        case GL_LINES_ADJACENCY:          return 4;
        case GL_LINE_STRIP_ADJACENCY:     return 5;
        case GL_TRIANGLES:                return 6;
        case GL_TRIANGLE_STRIP:           return 7;
        case GL_TRIANGLE_FAN:             return 8;
        case GL_TRIANGLES_ADJACENCY:      return 9;
        case GL_TRIANGLE_STRIP_ADJACENCY: return 10;
        default: return -1;
    }
}

/* is the primitive restart index enabled and what is it? */
static uint32_t geom_restart_index(GLMContext ctx)
{
    if (!ctx->state.caps.primitive_restart) return 0xffffffffu;
    if (ctx->state.caps.primitive_restart_fixed_index) return 0xffffffffu;  /* the fixed index */
    return (uint32_t)ctx->state.var.primitive_restart_index;
}

static bool geom_route_cpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d, GLenum gs_in);
static bool geom_route_gpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d);

/* the last context that ran a geometry draw (the trace is a debugging aid) */
static GLMContext gTraceCtx;
void *mgl_geom_ctx(void) { return gTraceCtx; }

/* ---- tessellation in front of the geometry stage ---- */

/* the tess domain tessellates this draw with its TES-output capture enabled (the records and
 * the output index list are GPU-visible), then the geometry emulation consumes them: the
 * import kernel (MGLRenderer+Geometry.m) repacks the capture records into this program's
 * input record and builds the primitive slots, exactly what the prep kernel produces for a
 * plain draw. */
static bool geom_route_tess(GLMContext ctx, MGLGeomProgram *gp, bool indexed, GLenum type,
                            size_t offset, GLint first, GLsizei count, GLsizei instances,
                            GLint basevertex, GLuint baseinstance)
{
    Program *program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                   : ctx->state.program;
    const MGLTessCapture *c;
    MGLGeomDraw d;
    GLuint saved_discard;
    uint32_t want = gp->in_type;      /* the GS input primitive the capture must produce */
    static const uint32_t kof[3] = { 1u, 2u, 3u };     /* GL_POINTS, GL_LINES, GL_TRIANGLES */

    if (!program || !program->shader_slots[_TESS_EVALUATION_SHADER]) return false;

    /* the tess stage must not rasterize: the geometry stage owns rasterization, and the
     * fragment shader would otherwise see the tessellation outputs.  rstate discards in the
     * last pre-rasterization stage's vertex function, so the tess pipeline is rebuilt for it
     * (the geometry pass restores the value afterwards, before its own raster pass). */
    mgl_tess_set_capture(ctx, program, true);
    saved_discard = ctx->state.caps.rasterizer_discard;
    ctx->state.caps.rasterizer_discard = GL_TRUE;
    ctx->state.dirty_bits |= DIRTY_PROGRAM;
    mgl_tess_draw_patches(ctx, indexed, type, offset, first, count, instances,
                          basevertex, baseinstance);
    ctx->state.caps.rasterizer_discard = saved_discard;
    ctx->state.dirty_bits |= DIRTY_PROGRAM;

    c = mgl_tess_get_capture(ctx);
    if (geom_env(0))
        fprintf(stderr, "geom: trace tess capture c=%p valid=%d prim=%u want=%u members=%d\n",
                (const void *)c, c ? (int)c->valid : -1, c ? c->prim : 0u, want, c ? c->members : -1);
    if (!c || !c->valid || !c->buffer_object || !c->buffer)
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }
    if (c->prim != want)
    {
        /* triangles -> triangles, isolines -> lines, point_mode -> points */
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return false;
    }

    /* repack: for every geometry input location, the tess capture member at the same location
     * supplies the words (gl_Position is words 0..3 in both layouts) */
    memset(&d, 0, sizeof(d));
    d.tess_ring = c->buffer_object;     /* the MTLBuffer (buffer is its CPU address) */
    d.tess_recs_off = c->records_offset;
    d.tess_idx_off = c->index_offset;
    d.tess_cnt_off = c->count_offset;
    d.tess_stride = c->stride;
    d.tess_prim = c->prim;
    d.tess_pairs[d.tess_npairs++] = 0; d.tess_pairs[d.tess_npairs++] = 0;
    d.tess_pairs[d.tess_npairs++] = 1; d.tess_pairs[d.tess_npairs++] = 1;
    d.tess_pairs[d.tess_npairs++] = 2; d.tess_pairs[d.tess_npairs++] = 2;
    d.tess_pairs[d.tess_npairs++] = 3; d.tess_pairs[d.tess_npairs++] = 3;
    for (int i = 0; i < gp->n_in; i++)
    {
        int j = -1;

        /* the consumer matches the tess capture's members by name: a linkable program has the
         * same names for the evaluation stage's outputs and the geometry stage's inputs.
         * A block member arrives flattened by SPIRV-Cross as "<instance>_<member>" (the
         * instance names of the two stages may differ: blocks match by block name) */
        for (int m = 0; m < c->members; m++)
        {
            const char *want = gp->in_v[i].member[0] ? gp->in_v[i].member : gp->in_v[i].name;
            size_t wl = strlen(want), cl = strlen(c->member[m].name);
            int hit = !strcmp(c->member[m].name, want);
            if (!hit && gp->in_v[i].member[0] && cl > wl + 1 &&
                c->member[m].name[cl - wl - 1] == '_' &&
                !strcmp(c->member[m].name + cl - wl, want))
                hit = 1;               /* "<instance>_<member>" */
            if (hit && c->member[m].words >= gp->in_v[i].words)
            {
                j = m;
                break;
            }
        }
        if (j < 0)
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return false;
        }
        if (c->member[j].words != gp->in_v[i].words ||
            d.tess_npairs + 2 * gp->in_v[i].words > MGL_GEOM_TESS_MAX_WORDS)
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return false;
        }
        for (uint32_t w = 0; w < gp->in_v[i].words; w++)
        {
            d.tess_pairs[d.tess_npairs++] = c->member[j].offset + w;   /* word offset */
            d.tess_pairs[d.tess_npairs++] = gp->in_v[i].offset + w;
        }
    }

    /* sizing bounds: the capture ring block's record and index-list capacities (the tess
     * domain sizes them for this draw's patches).  The tess producer tessellates
     * patches * instances patches, so every instance is already in the capture and the
     * geometry stage sees a single instance of the assembled primitive stream. */
    d.gp = gp;
    d.program = program;
    d.mode = c->prim;                 /* the assembled primitives are of the capture's type */
    d.first = first;
    d.count = c->indices ? (GLsizei)c->indices : count;
    d.instances = 1;
    d.basevertex = basevertex;
    d.baseinstance = baseinstance;
    d.tess_rec_cap = c->records ? c->records : 1;
    d.tess_idx_cap = c->indices ? c->indices : 1;

    (void)kof;
    return METAL()->draw(ctx, &d);
}

void mgl_geom_route(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                    GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    MGLGeomProgram *gp;
    MGLGeomDraw d;
    GLenum gs_in;

    if (!mgl_geom_present(ctx)) return;

    gTraceCtx = ctx;
    gp = mgl_geom_current(ctx);
    if (!gp) return;

    if (geom_env(0))
        fprintf(stderr, "geom: trace route mode=%u indexed=%d first=%u count=%u inst=%u gp=%p linked=%d\n",
                mode, (int)indexed, first, count, instances, (void *)gp, (int)gp->linked);

    if (!gp->linked && !geom_ensure(ctx, gp,
                                    ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                                : ctx->state.program))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* a tessellation stage in front of the geometry shader: the tess domain tessellates (and
     * publishes its TES-output capture), then the geometry emulation consumes it.  The GL
     * draw mode is GL_PATCHES, so the usual input-type check does not apply — the capture's
     * output primitive type is validated against it instead. */
    if (gp->from_tess)
    {
        if (mode != GL_PATCHES)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        if (geom_route_tess(ctx, gp, indexed, type, offset, first, count, instances,
                            basevertex, baseinstance))
            return;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (!geom_mode_matches(gp->in_type, mode))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    gs_in = gp->in_type;

    memset(&d, 0, sizeof(d));
    d.gp = gp;
    d.program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx) : ctx->state.program;
    d.mode = mode;
    d.indexed = indexed;
    d.type = type;
    d.offset = offset;
    d.first = first;
    d.count = count;
    d.instances = instances;
    d.basevertex = basevertex;
    d.baseinstance = baseinstance;
    d.restart = geom_restart_index(ctx);
    d.provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;

    if (!METAL() || !METAL()->draw)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* the primitive assembly runs on the GPU whenever the element buffer is device-visible
     * and nothing forces the CPU path */
    if (!geom_env(3) &&
        METAL()->can_assemble_gpu && METAL()->can_assemble_gpu(ctx) &&
        geom_route_gpu(ctx, gp, &d))
        return;

    if (count <= 0 || instances <= 0) return;
    if (geom_route_cpu(ctx, gp, &d, gs_in)) return;

    METAL()->draw(ctx, &d);
}

/* CPU assembly: the fallback (MGL_GEOM_CPU_ASSEMBLY=1, or an element buffer the GPU cannot
 * read).  Returns true when it filled d with the lists it built. */
static bool geom_route_cpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d, GLenum gs_in)
{
    uint32_t *seq = NULL, *indices = NULL;
    uint64_t nseq = 0, nprims;
    uint32_t *prims = NULL;
    uint32_t restart = d->restart;
    uint32_t vmin = 0xffffffffu, vmax = 0;
    GLsizei index_count = 0;
    GLuint i;

    /* build the vertex sequence and (for indexed draws) the capture index list: primitive
     * restart entries are removed from the capture draw */
    if (d->indexed)
    {
        uint32_t *raw = geom_index_list(ctx, d->type, d->offset, d->count, &restart);
        if (!raw) return false;
        seq = xmalloc(sizeof(uint32_t) * (size_t)(d->count > 0 ? d->count : 1));
        indices = xmalloc(sizeof(uint32_t) * (size_t)(d->count > 0 ? d->count : 1));
        for (GLsizei j = 0; j < d->count; j++)
        {
            seq[j] = raw[j];
            if (raw[j] != restart) indices[index_count++] = raw[j];
        }
        free(raw);
        nseq = (uint64_t)index_count;
    }
    else
    {
        nseq = (uint64_t)d->count;
        index_count = 0;
        seq = xmalloc(sizeof(uint32_t) * (size_t)nseq);
        for (i = 0; i < (GLuint)nseq; i++) seq[i] = (uint32_t)(d->first + (GLint)i);
    }

    /* the vertex range the capture pass must cover.  The restart sentinel is not a vertex:
     * counting it would make vmax 0xffffffff and the capture record buffer would be sized by
     * 4 billion vertices (a >100 GB allocation, which is what stalled primitive restart) */
    for (i = 0; i < (GLuint)nseq; i++)
    {
        if (d->indexed && seq[i] == restart) continue;
        if (seq[i] < vmin) vmin = seq[i];
        if (seq[i] > vmax) vmax = seq[i];
    }
    if (vmin == 0xffffffffu)
    {
        free(seq);
        free(indices);
        return true;              /* nothing to draw */
    }
    if (d->indexed) vmin += (uint32_t)d->basevertex, vmax += (uint32_t)d->basevertex;

    /* split on primitive restart and assemble */
    {
        uint64_t s = 0;

        nprims = 0;
        while (s < nseq)
        {
            uint64_t e = s;
            while (e < nseq && seq[e] != restart) e++;
            nprims += geom_prim_count(gs_in, d->mode, e - s);
            s = e + 1;
        }
        if (!nprims)
        {
            free(seq);
            free(indices);
            return true;
        }

        prims = xmalloc(sizeof(uint32_t) * (size_t)(nprims * gp->in_verts));
        if (!prims)
        {
            free(seq);
            free(indices);
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return true;
        }

        {
            uint64_t np = 0;
            s = 0;
            while (s < nseq)
            {
                uint64_t e = s;
                while (e < nseq && seq[e] != restart) e++;
                if (e > s)
                {
                    uint32_t *sub = xmalloc(sizeof(uint32_t) * (size_t)(e - s));
                    uint64_t len = e - s;
                    for (uint64_t j = 0; j < len; j++) sub[j] = seq[s + j];
                    np += geom_assemble(ctx, gp, d->mode, sub, len, 0,
                                        prims + np * gp->in_verts, nprims - np);
                    free(sub);
                }
                s = e + 1;
            }
            (void)np;
        }
    }
    free(seq);

    d->indices = indices;
    d->index_count = index_count;
    d->prims = prims;
    d->prims_count = nprims;
    d->vstart = vmin;
    d->vcount = vmax - vmin + 1;

    METAL()->draw(ctx, d);

    free(prims);
    free(indices);
    return true;
}

/* ------------------------------------------------------------------ init / queries */

void mgl_ext_init_geom(GLMContext ctx)
{
    if (!ctx->state.ext_geom)
        ctx->state.ext_geom = calloc(1, sizeof(struct MGLExtState_geom));
    mgl_ext_register_post_link(mgl_geom_post_link);
}

bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    return false;
}

/* GPU assembly: the prep kernel derives everything from the draw command and the element
 * buffer, so the CPU never reads the element buffer or assembles the slot list */
static bool geom_route_gpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d)
{
    if (mgl_geom_mode_code(d->mode) < 0) return false;
    d->count_bound = (uint32_t)(d->count > 0 ? d->count : 1);
    if (d->count <= 0 || d->instances <= 0) return false;
    if (d->indexed && d->type == GL_UNSIGNED_INT && ((d->offset & 3u) != 0)) return false;
    if (d->indexed && d->type == GL_UNSIGNED_SHORT && ((d->offset & 1u) != 0)) return false;

    /* the prep kernel reads the command itself when the draw is indirect */
    d->cmd_layout = 0;
    d->cmd_buffer = NULL;
    d->cmd_offset = 0;

    METAL()->draw(ctx, d);
    return true;
}

/* an indirect draw: the prep kernel reads the command from the user's buffer, so the CPU
 * never sees the counts.  Returns false when the buffer sizes cannot be bounded without
 * reading the command (then the caller falls back to the CPU route). */
bool mgl_geom_route_indirect(GLMContext ctx, GLenum mode, GLenum type, Buffer *ibuf, size_t offset)
{
    MGLGeomProgram *gp;
    MGLGeomDraw d;
    uint32_t bound = 0;

    if (!mgl_geom_present(ctx)) return false;
    if (mgl_geom_mode_code(mode) < 0) return false;

    gp = mgl_geom_current(ctx);
    if (!gp) return false;
    if (!gp->linked && !geom_ensure(ctx, gp,
                                    ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                                : ctx->state.program))
        return false;
    if (!geom_mode_matches(gp->in_type, mode)) return false;

    /* bound the vertex count without reading the command: an indexed draw cannot use more
     * indices than the element buffer holds */
    if (type)
    {
        Buffer *eb = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
        if (!eb || !eb->data.mtl_data) return false;
        bound = (uint32_t)((eb->size - offset) / (type == GL_UNSIGNED_BYTE ? 1 :
                                                  type == GL_UNSIGNED_SHORT ? 2 : 4));
    }
    else
    {
        /* arrays: no more vertices than the smallest attribute buffer can supply */
        VertexArray *vao = ctx->state.vao;
        bound = 1u << 20;
        if (vao)
        {
            for (GLuint loc = 0; loc < MAX_ATTRIBS; loc++)
            {
                VertexAttrib *a = &vao->attrib[loc];
                Buffer *ab = a->buffer;
                uint32_t n;

                if (!(vao->enabled_attribs & (1u << loc))) continue;
                if (!ab || ab->size <= 0) continue;
                n = (uint32_t)((size_t)ab->size / (a->stride ? a->stride : 16));
                if (n && n < bound) bound = n;
            }
        }
        if (bound == (1u << 20)) return false;      /* unbounded: read the command instead */
    }
    if (!bound) return false;

    if (!METAL() || !METAL()->draw || !METAL()->can_assemble_gpu || !METAL()->can_assemble_gpu(ctx))
        return false;

    memset(&d, 0, sizeof(d));
    d.gp = gp;
    d.program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx) : ctx->state.program;
    d.mode = mode;
    d.indexed = type != 0;
    d.type = type;
    d.offset = 0;
    d.count = 0;
    d.instances = 1;                 /* the kernel reads the real values from the command */
    d.cmd_buffer = ibuf ? ibuf->data.mtl_data : NULL;
    d.cmd_offset = offset;
    d.cmd_layout = type ? 2 : 1;
    d.restart = geom_restart_index(ctx);
    d.provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;
    d.count_bound = bound;

    if (!d.cmd_buffer) return false;

    METAL()->draw(ctx, &d);
    return true;
}

/* ------------------------------------------------------------------ transform feedback of the geometry output
 * With a geometry stage active, GL captures the vertices the geometry shader emits to the
 * rasterization stream.  The emulation has them in the output vertex buffer already, so the
 * capture is a CPU pack of those records into the transform feedback buffers. */

static MGLXfbProgram *geom_xfb_prog(GLMContext ctx, Program *p, bool create)
{
    struct MGLExtState_xfb *x = ctx->state.ext_xfb;

    if (!x || !p) return NULL;
    for (GLuint i = 0; i < x->prog_count; i++)
    {
        MGLXfbProgram *xp = x->progs[i];
        if (xp->name == p->name)
        {
            if (xp->program != p) xp->program = p;
            return xp;
        }
    }
    if (!create) return NULL;
    if (x->prog_count == x->prog_cap)
    {
        GLuint cap = x->prog_cap ? x->prog_cap * 2 : 16;
        MGLXfbProgram **n = realloc(x->progs, cap * sizeof(*n));
        if (!n) return NULL;
        x->progs = n;
        x->prog_cap = cap;
    }
    MGLXfbProgram *xp = calloc(1, sizeof(MGLXfbProgram));
    if (!xp) return NULL;
    xp->name = p->name;
    xp->program = p;
    xp->pending_mode = GL_INTERLEAVED_ATTRIBS;
    xp->mode = GL_INTERLEAVED_ATTRIBS;
    x->progs[x->prog_count++] = xp;
    return xp;
}

static void geom_xfb_reset(GLMContext ctx, MGLXfbProgram *xp)
{
    if (xp->vars)
    {
        for (GLuint i = 0; i < xp->count; i++)
        {
            free(xp->vars[i].name);
            free(xp->vars[i].expr);
        }
        free(xp->vars);
    }
    xp->vars = NULL;
    xp->count = 0;
    xp->buffer_mask = 0;
    memset(xp->stride, 0, sizeof(xp->stride));
    xp->record_words = 0;
    free(xp->link_log);
    xp->link_log = NULL;
    xp->link_failed = false;
    xp->linked = true;
}

static GLenum geom_var_type(const MGLGeomVar *v)
{
    if (v->cols > 1)
    {
        if (v->base == GL_FLOAT) return v->cols == v->rows ? GL_FLOAT_MAT4 : GL_FLOAT_MAT4;   /* refined below */
        return GL_INT;
    }
    switch (v->base)
    {
        case GL_FLOAT:         return v->rows == 1 ? GL_FLOAT : GL_FLOAT_VEC2 + (v->rows - 2);
        case GL_INT:           return v->rows == 1 ? GL_INT : GL_INT_VEC2 + (v->rows - 2);
        case GL_UNSIGNED_INT:  return v->rows == 1 ? GL_UNSIGNED_INT : GL_UNSIGNED_INT_VEC2 + (v->rows - 2);
        case GL_DOUBLE:        return v->rows == 1 ? GL_DOUBLE : GL_DOUBLE_VEC2 + (v->rows - 2);
        default:               return GL_FLOAT;
    }
}

/* resolve the transform feedback varyings of a program with a geometry stage against the
 * geometry stage's outputs (called from the xfb domain's post-link hook) */
/* resolve the transform feedback varyings of one program against the outputs of its LAST
 * pre-rasterization stage: the geometry stage's out table, or (without a geometry shader)
 * the tessellation evaluation stage's SPIR-V.  The GL layout rules (gl_NextBuffer /
 * gl_SkipComponents, interleaved vs separate) are applied to the caller's pending list. */
/* record an xfb layout error as a link failure with an info-log message */
static void geom_xfb_fail(GLMContext ctx, MGLXfbProgram *xp, const char *msg)
{
    char *log = NULL;

    (void)ctx;
    mgl_append_info_log(&log, "ERROR: transform feedback: %s\n", msg);
    free(xp->link_log);
    xp->link_log = log;
    xp->link_failed = true;
}

static void geom_xfb_resolve(GLMContext ctx, MGLXfbProgram *xp,
                             const MGLGeomVar *vars, int nvars,
                             GLuint pos_word, GLuint psize_word,
                             const char *stage)
{
    GLuint dbuffer[64], doffset[64];
    GLsizei dn = 0;
    GLuint buffer = 0, offset = 0, n = 0, xn = 0;
    MGLXfbVarying *v;
    char dnames[64][192];
    char xnames[128][192];
    GLuint xsrc[128];                 /* SEPARATE: the pending name each expansion came from */
    GLuint sepo[128];                 /* SEPARATE: running byte offset per pending name */
    bool separate = xp->mode == GL_SEPARATE_ATTRIBS;

    memset(xsrc, 0, sizeof xsrc);
    memset(sepo, 0, sizeof sepo);

    geom_xfb_reset(ctx, xp);
    xp->mode = xp->pending_mode;
    if (!xp->pending_count)
    {
        /* shader-declared capture (xfb_buffer / xfb_offset on outputs, including interface
         * block members): GL 4.6 13.3.2 — with no API-specified varyings, every output that
         * declares an xfb_buffer is captured into that buffer at its declared offset */
        for (int k = 0; k < nvars && dn < 64; k++)
        {
            if (vars[k].xfb_buffer < 0) continue;
            if (vars[k].member[0] && strcmp(vars[k].name, vars[k].member))
                snprintf(dnames[dn], sizeof(dnames[dn]), "%s.%s", vars[k].name, vars[k].member);
            else
                snprintf(dnames[dn], sizeof(dnames[dn]), "%s", vars[k].name);
            dbuffer[dn] = (GLuint)vars[k].xfb_buffer;
            doffset[dn] = vars[k].xfb_offset;
            dn++;
        }
        if (!dn) return;             /* nothing captured for this program */
        /* capture in ascending (buffer, offset) order */
        for (GLsizei a = 1; a < dn; a++)
            for (GLsizei b = a; b > 0 && (dbuffer[b - 1] > dbuffer[b] ||
                 (dbuffer[b - 1] == dbuffer[b] && doffset[b - 1] > doffset[b])); b--)
            {
                char tn[192];
                GLuint tb = dbuffer[b], to = doffset[b];
                memcpy(tn, dnames[b], sizeof(tn));
                memcpy(dnames[b], dnames[b - 1], sizeof(dnames[b]));
                memcpy(dnames[b - 1], tn, sizeof(dnames[b - 1]));
                dbuffer[b] = dbuffer[b - 1]; doffset[b] = doffset[b - 1];
                dbuffer[b - 1] = tb; doffset[b - 1] = to;
            }
        n = (GLuint)dn;
    }
    else
    {
        /* API-specified varyings: an interface block instance name ("b") selects every
         * member of the block (GL 4.6 13.3.1), so expand the names before resolving */
        n = (GLuint)xp->pending_count;
        for (GLuint q = 0; q < n && xn < 128; q++)
        {
            const char *nm = xp->pending[q];
            int first = -1, members = 0;

            if (!strcmp(nm, "gl_NextBuffer") || !strncmp(nm, "gl_SkipComponents", 17))
            {
                xsrc[xn] = q;
                snprintf(xnames[xn++], sizeof(xnames[0]), "%s", nm);
                continue;
            }
            for (int k = 0; k < nvars; k++)
                if (vars[k].member[0] && !strcmp(vars[k].name, nm))
                { if (first < 0) first = k; members++; }
            if (members > 0)
            {
                for (int k = 0; k < nvars && xn < 128; k++)
                    if (vars[k].member[0] && !strcmp(vars[k].name, nm))
                    {
                        xsrc[xn] = q;
                        snprintf(xnames[xn++], sizeof(xnames[0]), "%s.%s",
                                 vars[k].name, vars[k].member);
                    }
            }
            else
            {
                xsrc[xn] = q;
                snprintf(xnames[xn++], sizeof(xnames[0]), "%s", nm);
            }
        }
        n = xn;
    }

    xp->vars = calloc(n, sizeof(MGLXfbVarying));
    if (!xp->vars) return;
    xp->count = n;

    for (GLuint i = 0; i < n; i++)
    {
        const char *name = xp->pending_count ? xnames[i] : dnames[i];

        if (!xp->pending_count)
        {
            buffer = dbuffer[i];
            offset = doffset[i];
        }
        v = &xp->vars[i];
        v->name = xstrdup(name);
        xp->max_name_length = (GLuint)(strlen(name) + 1 > (size_t)xp->max_name_length ?
                                        strlen(name) + 1 : (size_t)xp->max_name_length);

        if (!strcmp(name, "gl_NextBuffer"))
        {
            v->type = GL_NONE;
            v->size = 0;
            if (separate)
            {   /* GL 4.6 13.3.1: gl_NextBuffer is only defined for interleaved capture */
                geom_xfb_fail(ctx, xp, "gl_NextBuffer used with GL_SEPARATE_ATTRIBS");
                return;
            }
            xp->stride[buffer] = offset;
            buffer++;
            offset = 0;
            if (buffer >= MGL_XFB_MAX_BUFFERS) { xp->link_failed = true; return; }
            continue;
        }
        if (!strncmp(name, "gl_SkipComponents", 17))
        {
            GLuint n = (GLuint)atoi(name + 17);
            v->type = GL_NONE;
            v->size = (GLint)n;
            v->words = n;
            if (separate)
            {
                geom_xfb_fail(ctx, xp, "gl_SkipComponents used with GL_SEPARATE_ATTRIBS");
                return;
            }
            offset += n * 4;
            continue;
        }

        /* builtins */
        if (!strcmp(name, "gl_Position"))
        {
            v->type = GL_FLOAT_VEC4;
            v->comp_kind = GL_FLOAT;
            v->rows = 4;
            v->columns = 1;
            v->comp_per_elem = 1;
            v->words = 4;
            v->src_word = pos_word;
            if (separate) { v->buffer = xsrc[i]; v->offset = 0; xp->stride[xsrc[i]] = 16; }
            else          { v->buffer = buffer;  v->offset = offset; offset += 16; }
            xp->buffer_mask |= (1u << v->buffer);
            continue;
        }
        if (!strcmp(name, "gl_PointSize"))
        {
            v->type = GL_FLOAT;
            v->comp_kind = GL_FLOAT;
            v->rows = 1;
            v->columns = 1;
            v->comp_per_elem = 1;
            v->words = 1;
            v->src_word = psize_word;
            if (separate) { v->buffer = xsrc[i]; v->offset = 0; xp->stride[xsrc[i]] = 4; }
            else          { v->buffer = buffer;  v->offset = offset; offset += 4; }
            xp->buffer_mask |= (1u << v->buffer);
            continue;
        }

        /* a user output of the stage: either a plain varying, or an interface block member
         * written as instance.member ("b.c"); an interface block instance with a single
         * member also names that member */
        int j = -1;
        const char *dot = strchr(name, '.');
        for (int k = 0; k < nvars; k++)
        {
            if (dot)
            {
                size_t ilen = (size_t)(dot - name);
                if (!strncmp(vars[k].name, name, ilen) && vars[k].name[ilen] == 0 &&
                    !strcmp(vars[k].member, dot + 1))
                { j = k; break; }
            }
            else if (!strcmp(vars[k].name, name))
            { j = k; break; }
        }
        if (j < 0 && dot)
        {
            char *log = NULL;
            mgl_append_info_log(&log, "ERROR: transform feedback varying '%s' is not an output of the "
                                      "%s\n", name, stage);
            free(xp->link_log);
            xp->link_log = log;
            xp->link_failed = true;
            return;
        }
        if (j < 0)
        {
            char *log = NULL;
            mgl_append_info_log(&log, "ERROR: transform feedback varying '%s' is not an output of the "
                                      "%s\n", name, stage);
            free(xp->link_log);
            xp->link_log = log;
            xp->link_failed = true;
            return;
        }

        const MGLGeomVar *g = &vars[j];
        v->type = geom_var_type(g);
        v->comp_kind = g->base == GL_INT ? GL_INT : g->base == GL_UNSIGNED_INT ? GL_UNSIGNED_INT : GL_FLOAT;
        v->rows = g->rows;
        v->columns = g->cols;
        v->comp_per_elem = g->cols * g->rows * base_words(g->base);
        v->size = (GLint)g->elems;
        v->words = g->words;
        v->src_word = g->offset;
        if (separate)
        {
            /* each API-selected varying is its own binding point (GL 4.6 13.3.1); a name
             * that expanded to several block members captures them into that one buffer */
            v->buffer = xsrc[i];
            v->offset = sepo[xsrc[i]];
            sepo[xsrc[i]] += v->words * 4;
            xp->stride[xsrc[i]] = sepo[xsrc[i]];
        }
        else
        {
            v->buffer = buffer;
            v->offset = offset;
            if (xp->pending_count)
                offset += v->words * 4;
            else
            {
                /* declared capture: the stride of a buffer is the extent of its captures */
                GLuint end = offset + v->words * 4;
                if (end > xp->stride[buffer]) xp->stride[buffer] = end;
            }
        }
        xp->buffer_mask |= (1u << v->buffer);
    }

    if (xp->pending_count && !separate)
        xp->stride[buffer] = offset;
}

void mgl_geom_xfb_link(GLMContext ctx, Program *program)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, true);

    if (!gp || !xp) return;
    geom_xfb_resolve(ctx, xp, gp->out_v, gp->n_out, gp->out_pos_off, gp->out_psize_off,
                     "geometry stage");
}

/* transform feedback from a tessellation evaluation stage (no geometry shader): the TES's
 * SPIR-V outputs are the recorded stream.  The per-record word offsets of the tess output
 * capture differ from this table, so the pack refreshes src_word from the published member
 * table on every draw. */
void mgl_geom_tess_xfb_link(GLMContext ctx, Program *program)
{
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, true);
    MGLGeomVar vars[MGL_GEOM_MAX_VARYINGS];
    SpvModule *m = NULL;
    char *err = NULL;
    int n = 0;

    if (!xp || !program || !program->spirv[_TESS_EVALUATION_SHADER].ir) return;
    m = spv_parse(program->spirv[_TESS_EVALUATION_SHADER].ir,
                  program->spirv[_TESS_EVALUATION_SHADER].size, (const char **)&err);
    if (!m)
    {
        char *log = NULL;
        mgl_append_info_log(&log, "ERROR: transform feedback: TES reflection failed (%s)\n",
                            err ? err : "SPIR-V parse failed");
        free(xp->link_log);
        xp->link_log = log;
        xp->link_failed = true;
        return;
    }
    if (!geom_reflect_stage(m, false, vars, &n, NULL))
        n = 0;
    spv_free(m);
    /* gl_Position is the first word of every capture record; the tess output capture does not
     * record gl_PointSize (a TES writing it would be dropped) */
    geom_xfb_resolve(ctx, xp, vars, n, 0, 0, "tessellation evaluation stage");
}

/* the active transform feedback object (the xfb domain's current-object rule) */
static MGLXfbObject *geom_xfb_object(GLMContext ctx)
{
    struct MGLExtState_xfb *x = ctx->state.ext_xfb;
    GLuint name;

    if (!x) return NULL;
    name = ctx->state.transform_feedback ? ctx->state.transform_feedback->name : 0;
    if (name && name < x->obj_cap && x->objs[name]) return x->objs[name];
    return &x->default_obj;
}


/* ---------------- GPU pack of the geometry output ---------------- */

/* the test harness reaches the tracking state through here */
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);

void *mgl_geom_ctx(void);

/* trace helper: the buffer-tracking state of a transform feedback buffer at a CPU read */
void mgl_geom_xfb_log_read_name(void *ctxp, unsigned name, const char *what)
{
    GLMContext ctx = (GLMContext)ctxp;
    Buffer *buf = ctx ? (ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf) : NULL;
    struct MGLExtMetal_buffers *b = ctx ? ctx->mtl_funcs.ext_buffers : NULL;
    MGLBufferTrack *t = ctx ? mgl_buffers_track(ctx, buf, false) : NULL;
    uint64_t completed = (b && b->completed_serial) ? b->completed_serial(ctx) : 0;

    fprintf(stderr, "geom: read %-9s name=%u buf=%p ctx=%p write_serial=%llu completed=%llu (%s)\n",
            what, name, (void *)buf, (void *)ctx,
            (unsigned long long)(t ? t->write_serial : 0),
            (unsigned long long)completed,
            (!t || t->write_serial <= completed) ? "no wait" : "MUST wait");
}

/* the Metal side of the pack (MGLRenderer+Geometry.m) */
extern bool geom_xfb_gpu_encode(GLMContext ctx, MGLGeomProgram *gp, MGLXfbObject *o, MGLXfbProgram *xp,
                                BufferBaseTarget *tfb, uint32_t threads, uint32_t max_vertices,
                                uint32_t out_words, uint32_t vertex_cap);
extern bool geom_xfb_tess_encode(GLMContext ctx, MGLXfbObject *o, MGLXfbProgram *xp,
                                 BufferBaseTarget *tfb, const MGLGeomTessXfbSource *src);
extern void geom_xfb_gpu_read(GLMContext ctx, MGLXfbObject *o, uint32_t *dvert, uint32_t *dprim);
extern void geom_xfb_gpu_reset(GLMContext ctx);
extern bool geom_xfb_gpu_pending(void);
extern bool geom_xfb_gpu_valid(GLMContext ctx, GLuint name);
extern bool geom_xfb_gpu_ensure(GLMContext ctx, Buffer *buf);
extern void geom_xfb_gpu_wait(GLMContext ctx);

/* try to encode the GPU pack for one draw.  Returns false when the CPU pack must be used. */
bool mgl_geom_xfb_pack(GLMContext ctx, MGLGeomProgram *gp, Program *program,
                       uint32_t threads, uint32_t max_vertices, uint32_t out_words,
                       uint32_t vertex_cap)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, gp->program, false);
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;
    GLuint b;

    if (!o || !o->active || !xp || !xp->count) return false;
    /* the GPU pack is the default (the counters and write offsets stay on the GPU and are
     * only folded in where the application observes them); MGL_GEOM_CPU_XFB=1 asks for the
     * CPU pack, which advances the object's offsets at draw time */
    if (geom_env(2)) return false;
    if (mgl_geom_xfb_pending(ctx) && !geom_xfb_gpu_valid(ctx, o->name)) mgl_geom_xfb_flush(ctx);

    /* the varying table and the buffer strides/sizes must fit the fixed blocks; a transform
     * feedback buffer without Metal backing yet gets one from the renderer, so a block never
     * mixes the CPU pack (which advances the object's offsets on the CPU) with the GPU pack
     * (whose counter block holds its own) */
    for (b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
        if (xp->buffer_mask & (1u << b))
        {
            Buffer *buf = tfb[b].buf;
            if (!buf) return false;
            if (!buf->data.mtl_data && !geom_xfb_gpu_ensure(ctx, buf)) return false;
        }
    if (xp->count > MGL_GEOM_XFB_MAX_VARS) return false;

    return geom_xfb_gpu_encode(ctx, gp, o, xp, tfb, threads, max_vertices, out_words, vertex_cap);
}

/* fold the pending GPU counters into the transform feedback object.  Called at the points
 * where the application can observe them, so the command buffer has usually completed. */
void mgl_geom_xfb_flush(GLMContext ctx)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    uint32_t dvert, dprim;

    if (!mgl_geom_xfb_pending(ctx)) return;

    /* make the pack's results visible: wait for the command buffer it was encoded into */
    geom_xfb_gpu_wait(ctx);

    if (geom_env(1))
    {
        Buffer *b0 = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf;
        fprintf(stderr, "geom:   fold reads TF backing %p\n", b0 ? b0->data.mtl_data : NULL);
    }
    { extern void geom_xfb_dump(GLMContext ctx); geom_xfb_dump(ctx); }
    geom_xfb_gpu_read(ctx, o, &dvert, &dprim);
    if (geom_env(1))
    {
        Buffer *b0 = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf;
        float *f = b0 ? (float *)(void *)(vm_address_t)b0->data.buffer_data : NULL;
        fprintf(stderr, "geom: flush name=%u dvert=%u dprim=%u tf=%p words=[%f %f %f %f]\n",
                o ? o->name : 0, dvert, dprim, (void *)f,
                f ? f[0] : 0.0f, f ? f[1] : 0.0f, f ? f[2] : 0.0f, f ? f[3] : 0.0f);
    }

    if (o)
    {
        o->vertices[0] += dvert;
        o->prims_written[0] += dprim;
        o->prims_needed[0] += dprim;
    }
    if (ctx->state.ext_xfb)
    {
        struct MGLExtState_xfb *x = ctx->state.ext_xfb;
        x->total_written[0] += dprim;
        x->total_needed[0] += dprim;
    }

    /* the transform feedback primitive queries are counted by the queries domain */
    {
        typedef void (*geom_note_queries_fn)(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
        static geom_note_queries_fn note;
        static int looked;
        if (!looked)
        {
            note = (geom_note_queries_fn)dlsym(RTLD_DEFAULT, "mgl_queries_note_xfb_primitives");
            looked = 1;
        }
        if (note) note(ctx, 0, dprim, dprim);
    }

    geom_xfb_gpu_reset(ctx);
}

/* the GPU pack's own view of "paused": true when the bound transform feedback object is
 * active but paused (nothing is captured, the GPU counter block is left untouched) */
bool mgl_geom_xfb_paused(GLMContext ctx)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    return o && o->active && o->paused;
}

bool mgl_geom_xfb_pending(GLMContext ctx)
{
    (void)ctx;
    return geom_xfb_gpu_pending();
}

/* pack the emitted geometry output into the bound transform feedback buffers.  Called after
 * the geometry kernel has been synchronized, from the Metal side of the emulation.  The
 * kernel assigns every thread a deterministic slot range (thread * max_vertices), so the
 * emission order is simply the slot order. */
void mgl_geom_xfb_capture(GLMContext ctx, MGLGeomProgram *gp, const uint32_t *records, size_t words,
                          const uint32_t *ctl, const uint32_t *cnt, size_t cnt_words)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, gp->program, false);
    uint32_t count = ctl ? ctl[0] : 0;
    uint32_t prims = ctl ? ctl[13] : 0;
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;

    if (!o || !o->active || !xp || !xp->count || !count) return;
    if (!records || !cnt) return;

    for (uint32_t t = 0; t < cnt_words; t++)
    {
        uint32_t n = cnt[t];
        uint32_t base_slot = t * gp->max_vertices;

        if (base_slot + n > (uint32_t)(words / gp->out_words)) break;

        for (uint32_t s = 0; s < n; s++)
        {
            const uint32_t *rec = records + (size_t)(base_slot + s) * gp->out_words;
            GLuint stream = (rec[1] >> 24) & 0xff;

            /* GL captures the rasterization stream; MGL has no glRasterizationStream yet, so
             * it is stream 0 (the default) */
            if (stream != 0) continue;

            for (GLuint v = 0; v < xp->count; v++)
            {
                MGLXfbVarying *var = &xp->vars[v];
                Buffer *buf;
                uint32_t *dst;
                size_t pos;

                if (!var->words) continue;                    /* skip / next-buffer markers */
                if (!(xp->buffer_mask & (1u << var->buffer))) continue;
                buf = tfb[var->buffer].buf;
                if (!buf || !buf->data.buffer_data) continue;

                pos = (size_t)(o->write_offset[var->buffer] + var->offset);
                if (pos + var->words * 4 > (size_t)buf->size) continue;   /* full: drop, as GL does */

                dst = (uint32_t *)((uint8_t *)buf->data.buffer_data + pos);
                for (GLuint w = 0; w < var->words; w++)
                    dst[w] = rec[var->src_word + w];
            }

            for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
                if (xp->buffer_mask & (1u << b))
                    o->write_offset[b] += xp->stride[b];
            o->vertices[0]++;
        }
    }

    o->prims_written[0] += prims;
    o->prims_needed[0] += prims;
    if (ctx->state.ext_xfb)
    {
        struct MGLExtState_xfb *x = ctx->state.ext_xfb;
        x->total_written[0] += prims;
        x->total_needed[0] += prims;
    }

    /* the transform feedback primitive queries are counted by the queries domain */
    {
        typedef void (*geom_note_queries_fn)(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
        static geom_note_queries_fn note;
        static int looked;
        if (!looked)
        {
            note = (geom_note_queries_fn)dlsym(RTLD_DEFAULT, "mgl_queries_note_xfb_primitives");
            looked = 1;
        }
        if (note) note(ctx, 0, prims, prims);
    }
}

/* ------------------------------------------------------------------ xfb from tessellation
 * Without a geometry shader the tessellation evaluation stage is the LAST
 * pre-rasterization stage, so transform feedback records its outputs.  The tess domain's
 * output capture (the same one the geometry stage consumes) is the recorded stream: the
 * draw runs with the capture enabled and still rasterizes normally, then this pack writes
 * the captured records into the transform feedback buffers. */

bool mgl_geom_tess_xfb_route(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                             GLint first, GLsizei count, GLsizei instances,
                             GLint basevertex, GLuint baseinstance)
{
    Program *program = ctx->state.program ? getProgram(ctx, ctx->state.program->name) : NULL;
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, false);
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;
    const MGLTessCapture *c;
    MGLGeomTessXfbSource src;
    GLuint b;
    bool ok;

    if (!program || !o || !o->active || !xp || !xp->count) return false;
    if (geom_env(2)) return false;
    if (mgl_geom_xfb_pending(ctx) && !geom_xfb_gpu_valid(ctx, o->name)) mgl_geom_xfb_flush(ctx);
    if (xp->count > MGL_GEOM_XFB_MAX_VARS) return false;
    for (b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
        if (xp->buffer_mask & (1u << b))
        {
            Buffer *buf = tfb[b].buf;
            if (!buf) return false;
            if (!buf->data.mtl_data && !geom_xfb_gpu_ensure(ctx, buf)) return false;
        }

    /* the tess domain records its output for us and rasterizes the patch itself */
    mgl_tess_set_capture(ctx, program, true);
    ok = mgl_tess_draw_patches(ctx, indexed, type, offset, first, count, instances,
                               basevertex, baseinstance);
    if (!ok) return false;

    c = mgl_tess_get_capture(ctx);
    if (!c || !c->valid || !c->buffer_object || !c->stride) return true;   /* nothing captured */

    /* the capture record's word offsets come from the published member table: they follow the
     * tess output capture's layout, not the geometry stage's table the link-time resolution
     * used */
    for (GLsizei i = 0; i < xp->count; i++)
    {
        MGLXfbVarying *v = &xp->vars[i];
        const char *nm = v->name ? v->name : "";
        uint32_t w = v->words;
        bool found = false;

        if (!strcmp(nm, "gl_Position")) { v->src_word = 0; continue; }   /* always first */
        for (uint32_t k = 0; k < c->members; k++)
            if (!strcmp(c->member[k].name, nm))
            {
                v->src_word = c->member[k].offset;
                if (!w) w = c->member[k].words;
                found = true;
                break;
            }
        if (!found && v->type != GL_NONE)
        {
            /* a varying the capture does not record (e.g. gl_PointSize): record nothing for it
             * rather than capturing the wrong words */
            v->words = 0;
            v->src_word = 0;
        }
    }

    memset(&src, 0, sizeof src);
    src.ring = c->buffer_object;
    src.records = c->records_offset;
    src.counts = c->count_offset;
    src.stride = c->stride;
    src.capacity = c->records ? c->records : 1;
    /* GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN counts primitives: k captured vertices each */
    src.per_prim = c->prim == GL_TRIANGLES ? 3 : c->prim == GL_LINES ? 2 : 1;

    return geom_xfb_tess_encode(ctx, o, xp, tfb, &src);
}
