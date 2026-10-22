/*
 * tessellation.c
 * MGL — tessellation shaders (TCS/TES) and GL_PATCHES: GL side.
 *
 * The Metal side of this domain is MGLRenderer+Tessellation.m; see mgl_ext_tess.h
 * for the overall design. This file owns:
 *   - the MSL rewrites of the tessellation stages (called from program.c during link)
 *   - SPIR-V parsing (tessellation execution modes, control point layouts)
 *   - the CPU tessellator (GL primitive generation: triangles/quads/isolines)
 *   - the GL state, glGet hooks, program queries and the GL_PATCHES draw entry
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>

#include "glm_context.h"
#include "mgl.h"
#include "mgl_ext.h"
#include "mgl_ext_tess.h"
#include "mgl_ext_rstate.h"
#include "mgl_ext_queries.h"
#include "mgl_ext_program.h"
#include "spirv_cross_c.h"
#include "glcorearb.h"

/* ------------------------------------------------------------------ helpers */

/* Trace flag, evaluated once: getenv() per draw (or per shader rewrite) is not
 * free, and this file sits on the draw path. MGL_TESS_DEBUG=1 enables tracing. */
static int tess_trace_flag = -1;
static bool tess_trace(void)
{
    if (tess_trace_flag < 0)
        tess_trace_flag = (getenv("MGL_TESS_DEBUG") != NULL) ? 1 : 0;
    return tess_trace_flag != 0;
}

/* MGL_TESS_CPU (force the CPU reference path), evaluated once like tess_trace();
 * MGLRenderer.m consults it when it picks the pipeline's vertex function */
static int tess_cpu_flag = -1;
bool mgl_tess_cpu_reference(void)
{
    if (tess_cpu_flag < 0)
        tess_cpu_flag = (getenv("MGL_TESS_CPU") != NULL) ? 1 : 0;
    return tess_cpu_flag != 0;
}

/* failure exit of the TES->plain-MSL rewrite: never silent, but only verbose with
 * MGL_TESS_DEBUG (the step number localises the failing parse in rewrite_tes) */
#define TES_REWRITE_FAIL(step)                                                    \
    do {                                                                          \
        if (tess_trace())                                                         \
            fprintf(stderr, "MGL tess: plain-TES rewrite failed at step %d\n",    \
                    (step));                                                      \
        return NULL;                                                              \
    } while (0)

static void tess_reset_scratch(struct MGLExtState_tess *t)
{
    t->tess_vertex_count = 0;
    t->tess_index_count = 0;
    t->patch_count = 0;
}

/* ------------------------------------------------------------------ SPIR-V
 * Minimal walker: execution modes of the tessellation stages.
 */

#define SPV_OP_ENTRYPOINT      15
#define SPV_OP_EXECUTIONMODE   16
#define SPV_OP_CAPABILITY      17

/* values from spirv.h (SpvExecutionMode*) */
#define SPV_EM_SPACING_EQUAL                1
#define SPV_EM_SPACING_FRACTIONAL_EVEN      2
#define SPV_EM_SPACING_FRACTIONAL_ODD       3
#define SPV_EM_VERTEX_ORDER_CW              4
#define SPV_EM_VERTEX_ORDER_CCW             5
#define SPV_EM_POINT_MODE                   10
#define SPV_EM_TRIANGLES                    22
#define SPV_EM_QUADS                        24
#define SPV_EM_ISOLINES                     25
#define SPV_EM_OUTPUT_VERTICES              26

/* collect execution modes of the tessellation entry point (model-filtered) */
static bool spv_tess_modes(const uint32_t *ir, size_t n, unsigned model_wanted,
                           int modes[16], int *count, int *out_vertices)
{
    size_t i = 5;
    uint32_t entry_id = 0;

    *count = 0;
    if (out_vertices) *out_vertices = 0;
    if (!ir || n < 5 || ir[0] != 0x07230203)
        return false;

    while (i < n)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint32_t op = word & 0xffff;

        if (wc == 0 || i + wc > n)
            break;

        if (op == SPV_OP_ENTRYPOINT && wc >= 2 && ir[i + 1] == model_wanted)
            entry_id = ir[i + 2];
        i += wc;
    }

    if (!entry_id)
        return false;

    i = 5;
    while (i < n)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint32_t op = word & 0xffff;

        if (wc == 0 || i + wc > n)
            break;

        if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && ir[i + 1] == entry_id)
        {
            uint32_t mode = ir[i + 2];
            if (mode == SPV_EM_OUTPUT_VERTICES && wc >= 4 && out_vertices)
                *out_vertices = (int)ir[i + 3];
            if (*count < 16)
                modes[(*count)++] = (int)mode;
        }
        i += wc;
    }
    return true;
}

static int spv_tess_domain(const uint32_t *ir, size_t n, int *spacing, int *ccw, int *point_mode)
{
    int modes[16], count = 0, domain = -1;

    if (!spv_tess_modes(ir, n, 2 /* SpvExecutionModelTessellationEvaluation */, modes, &count, NULL))
        return -1;

    *spacing = MGL_TESS_SPACING_EQUAL;
    *ccw = 1;
    *point_mode = 0;
    domain = MGL_TESS_DOMAIN_TRIANGLES;      /* SPIR-V default when no domain mode is given */
    for (int i = 0; i < count; i++)
    {
        switch (modes[i])
        {
            case SPV_EM_TRIANGLES: domain = MGL_TESS_DOMAIN_TRIANGLES; break;
            case SPV_EM_QUADS:     domain = MGL_TESS_DOMAIN_QUADS; break;
            case SPV_EM_ISOLINES:  domain = MGL_TESS_DOMAIN_ISOLINES; break;
            case SPV_EM_SPACING_EQUAL:           *spacing = MGL_TESS_SPACING_EQUAL; break;
            case SPV_EM_SPACING_FRACTIONAL_EVEN: *spacing = MGL_TESS_SPACING_FRACTIONAL_EVEN; break;
            case SPV_EM_SPACING_FRACTIONAL_ODD:  *spacing = MGL_TESS_SPACING_FRACTIONAL_ODD; break;
            case SPV_EM_VERTEX_ORDER_CCW: *ccw = 1; break;
            case SPV_EM_VERTEX_ORDER_CW:  *ccw = 0; break;
            case SPV_EM_POINT_MODE: *point_mode = 1; break;
            default: break;
        }
    }
    return domain;
}

/* isolines are emulated on the quad domain: SPIRV-Cross refuses ExecutionModeIsolines.
 * returns a copy of the module with Isolines replaced by Quads, or NULL. */
uint32_t *mgl_tess_prepare_spirv(const uint32_t *words, size_t n, size_t *out_n)
{
    bool has_isolines = false;
    uint32_t entry_id = 0, *copy;
    size_t i = 5;

    *out_n = 0;
    if (!words || n < 5)
        return NULL;

    while (i < n)
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_ENTRYPOINT && wc >= 2 && words[i + 1] == 2 /* TessEvaluation */)
            entry_id = words[i + 2];
        i += wc;
    }
    if (!entry_id)
        return NULL;

    i = 5;
    while (i < n)
    {
        uint32_t wc = words[i] >> 16, op = words[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && words[i + 1] == entry_id &&
            words[i + 2] == SPV_EM_ISOLINES)
            has_isolines = true;
        i += wc;
    }

    if (!has_isolines)
        return NULL;

    copy = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!copy)
        return NULL;
    memcpy(copy, words, n * sizeof(uint32_t));

    i = 5;
    while (i < n)
    {
        uint32_t wc = copy[i] >> 16, op = copy[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && copy[i + 1] == entry_id &&
            copy[i + 2] == SPV_EM_ISOLINES)
            copy[i + 2] = SPV_EM_QUADS;     /* isolines -> quad domain emulation */
        i += wc;
    }

    *out_n = n;
    return copy;
}

struct MslStruct;
static bool tess_cp_record_name(const char *msl, char *out, size_t out_len);
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len);
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len);
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out, char *name, size_t name_len);
static bool msl_tcs_record_names(const char *msl, char *in_name, size_t in_len,
                                 char *out_name, size_t out_len);
static void tess_canonicalize_records(GLMContext ctx, MGLTessProgram *tp);

/* ------------------------------------------------------------------ MSL struct layout
 * Parse an MSL interface struct declaration and compute the byte layout Metal
 * uses for it (C++ rules: natural alignment, arrays padded to element alignment).
 */

struct MslStruct {
    char body[8192];                       /* full declaration text */
    MGLTessPatchMember members[MGL_TESS_MAX_PATCH_MEMBERS];
    int count;
    size_t size;
    int cp_ref_member;                     /* index of the patch_control_point member */
};

/* element alignment/size of a base MSL type name (no array suffix) */
static bool msl_scalar_type(const char *t, size_t *align, size_t *size)
{
    struct { const char *name; size_t a, s; } tab[] = {
        { "float",  4, 4 }, { "half",  2, 2 },
        { "int",    4, 4 }, { "uint",  4, 4 }, { "bool", 1, 1 },
        { "short",  2, 2 }, { "ushort", 2, 2 }, { "char", 1, 1 }, { "uchar", 1, 1 },
        { "float2", 8, 8 }, { "half2",  4, 4 }, { "int2",  8, 8 }, { "uint2", 8, 8 },
        { "float3", 16, 16 }, { "half3",  8, 8 }, { "int3", 16, 16 }, { "uint3", 16, 16 },
        { "float4", 16, 16 }, { "half4",  8, 8 }, { "int4", 16, 16 }, { "uint4", 16, 16 },
        { "double", 8, 8 }, { "long", 8, 8 }, { "ulong", 8, 8 },
        { "double2", 16, 16 }, { "long2", 16, 16 }, { "ulong2", 16, 16 },
        { "double3", 32, 32 }, { "long3", 32, 32 }, { "ulong3", 32, 32 },
        { "double4", 32, 32 }, { "long4", 32, 32 }, { "ulong4", 32, 32 },
        { NULL, 0, 0 }
    };
    for (int i = 0; tab[i].name; i++)
        if (!strcmp(t, tab[i].name)) { *align = tab[i].a; *size = tab[i].s; return true; }
    return false;
}

/* matrix floatCxR: named like float4x4 (4 columns of float4) */
static bool msl_matrix_type(const char *t, size_t *align, size_t *size)
{
    int c, r;
    if (strlen(t) != 8 + 1 || t[5] != 'x' || strncmp(t, "float", 5))
        return false;
    c = t[6] - '0'; r = t[7] - '0';
    if (c < 2 || c > 4 || r < 2 || r > 4)
        return false;
    static const size_t colalign[5] = { 0, 4, 8, 16, 16 };
    static const size_t colsize[5]  = { 0, 4, 8, 16, 16 };
    *align = colalign[r];                 /* alignment of the column vector type */
    *size = (size_t)c * colsize[r];       /* C columns, each padded to column size */
    return true;
}

static size_t round_up(size_t v, size_t a) { return a ? ((v + a - 1) / a) * a : v; }

/* alignment/size of a possibly-arrayed element type: "float3", "spvUnsafeArray<float3, 4>", "float4x4" */
static bool msl_element_type(const char *t, size_t *align, size_t *size)
{
    if (!strncmp(t, "spvUnsafeArray<", 15))
    {
        char inner[64];
        long n = 0;
        /* spvUnsafeArray<T, N> */
        const char *comma = strchr(t + 15, ',');
        const char *close = strchr(t, '>');
        if (!comma || !close || close < comma || (size_t)(comma - (t + 15)) >= sizeof(inner))
            return false;
        memcpy(inner, t + 15, comma - (t + 15));
        inner[comma - (t + 15)] = 0;
        n = strtol(comma + 1, NULL, 10);
        if (n <= 0)
            return false;
        size_t ia, is;
        if (!msl_element_type(inner, &ia, &is))
            return false;
        *align = ia;
        *size = round_up((size_t)n * round_up(is, ia), ia);
        return true;
    }
    if (msl_scalar_type(t, align, size))
        return true;
    if (msl_matrix_type(t, align, size))
        return true;
    return false;
}

/* find "struct NAME { ... };" in msl; parse members and layout */
static bool msl_parse_struct(const char *msl, const char *name, struct MslStruct *out)
{
    char head[256];
    const char *p, *open, *close;
    size_t off = 0;
    size_t cursor = 0;

    snprintf(head, sizeof head, "struct %s\n{", name);
    p = strstr(msl, head);
    if (!p)
    {
        /* tolerate single-line form */
        snprintf(head, sizeof head, "struct %s {", name);
        p = strstr(msl, head);
        if (!p)
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: struct %s not found\n", name);
            return false;
        }
    }
    open = strchr(p, '{');
    close = strchr(open, '}');
    if (!open || !close || (size_t)(close - open) >= sizeof(out->body))
        return false;

    memcpy(out->body, open + 1, close - open - 1);
    out->body[close - open - 1] = 0;
    out->count = 0;
    out->cp_ref_member = -1;
    out->size = 0;

    /* split members on ';' */
    while (off < strlen(out->body))
    {
        char member[512], type[256], namebuf[128];
        const char *semi = strchr(out->body + off, ';');
        const char *attr;
        long len;
        char *last;
        int i;

        if (!semi)
            break;
        len = semi - (out->body + off);
        if (len <= 0 || len >= (long)sizeof(member))
        {
            off = (semi - out->body) + 1;
            continue;
        }
        memcpy(member, out->body + off, (size_t)len);
        member[len] = 0;
        off = (semi - out->body) + 1;

        /* strip [[...]] attributes, recording an [[attribute(N)]] number */
        int attr_num = -1;
        attr = strstr(member, "[[");
        if (attr)
        {
            const char *aend = strstr(attr, "]]");
            if (!aend) continue;
            {
                const char *adec = strstr(attr, "attribute(");
                if (adec && adec < aend)
                    attr_num = (int)strtol(adec + 10, NULL, 10);
            }
            memmove((void *)attr, (void *)(aend + 2), strlen(aend + 2) + 1);
        }

        /* trim whitespace on both ends: "type name" or "type name[N]" */
        {
            char *end = member + strlen(member);
            while (end > member && (end[-1] == ' ' || end[-1] == '\n' || end[-1] == '\t'))
                *--end = 0;
            while (*member == ' ' || *member == '\n' || *member == '\t')
                memmove(member, member + 1, strlen(member));
        }
        last = strrchr(member, ' ');
        if (!last) continue;
        snprintf(type, sizeof type, "%.*s", (int)(last - member), member);
        snprintf(namebuf, sizeof namebuf, "%s", last + 1);

        /* trailing array "name[4]" */
        char *brk = strchr(namebuf, '[');
        long arr = 1;
        if (brk)
        {
            arr = strtol(brk + 1, NULL, 10);
            if (arr <= 0) arr = 1;
            *brk = 0;
            /* move the count into the element type: "T[N]" */
            char witharr[300];
            snprintf(witharr, sizeof witharr, "spvUnsafeArray<%s, %ld>", type, arr);
            snprintf(type, sizeof type, "%s", witharr);
        }

        if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS)
            break;

        if (!strncmp(type, "patch_control_point<", 20))
        {
            out->cp_ref_member = out->count;
            out->members[out->count].attr = -1;
            out->members[out->count].is_tess_level = 0;
            snprintf(out->members[out->count].type, sizeof(out->members[out->count].type), "%s", type);
            snprintf(out->members[out->count].name, sizeof(out->members[out->count].name), "%s", namebuf);
            out->members[out->count].offset = 0;
            out->members[out->count].size = 0;      /* not part of the blob */
            out->count++;
            continue;
        }

        size_t align = 0, size = 0;
        if (!msl_element_type(type, &align, &size))
            continue;   /* unsupported member type: skip (shader did not use it in a layout we know) */

        cursor = round_up(cursor, align);
        out->members[out->count].offset = cursor;
        out->members[out->count].size = size;
        out->members[out->count].attr = attr_num;
        out->members[out->count].array_idx = 0;     /* set by tess_annotate_keys */
        out->members[out->count].is_tess_level =
            !strcmp(namebuf, "gl_TessLevel") || !strcmp(namebuf, "gl_TessLevelOuter") ? 1 :
            !strcmp(namebuf, "gl_TessLevelInner") ? 2 : 0;
        snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", type);
        snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s", namebuf);
        cursor += size;
        out->count++;
        (void)i;
    }

    out->size = cursor;
    return true;        /* the struct was found; count may be 0 (no data members) */
}

/* ------------------------------------------------------------------ canonical records
 * The per-vertex control point record and the per-patch record are written by one
 * stage (the VS kernel / the TCS kernel) and read by another (the TCS kernel / the
 * TES). SPIRV-Cross emits a struct per stage holding exactly the members THAT stage
 * statically uses, in declaration order — so a consumer that declares-but-does-not-
 * use a varying, or declares its inputs in a different order, would read every
 * following member at a shifted offset. The fix is one canonical record layout per
 * interface, derived from the UNION of the stages' members keyed by GL varying
 * location (never by name or by what SPIRV-Cross happens to keep):
 *
 *   - user varyings      key = Location decoration (from the program's reflected
 *                        SPIR-V resources of that stage)
 *   - builtins           key = fixed slots (gl_Position, gl_PointSize, ...) so every
 *                        stage agrees regardless of declaration order
 *   - unreflectable      key = stable hash of the member name (interface blocks whose
 *                        members SPIRV-Cross flattens but whose locations the resource
 *                        list does not carry)
 *
 * Every stage's MSL struct is then rewritten to the canonical member sequence (the
 * stage's own member names for members it has, never-referenced pads otherwise), and
 * the record the stages exchange is byte-identical for producer and consumer.
 */

/* fixed builtin slots: after user locations, before name hashes, in a stable order */
static uint32_t tess_builtin_key(const char *name)
{
    static const struct { const char *name; uint32_t key; } tab[] = {
        { "gl_Position",     0x80010000u },
        { "gl_PointSize",    0x80010001u },
        { "gl_ClipDistance", 0x80010002u },
        { "gl_CullDistance", 0x80010003u },
        { "gl_TessLevel",      0x80020000u },
        { "gl_TessLevelOuter", 0x80020000u },   /* packed half4 alias of the same slot */
        { "gl_TessLevelInner", 0x80020001u },
        { NULL, 0 }
    };
    for (int i = 0; tab[i].name; i++)
        if (!strcmp(name, tab[i].name))
            return tab[i].key;
    return 0;
}

static uint32_t tess_name_hash(const char *s)
{
    uint32_t h = 2166136261u;
    while (*s)
    {
        h ^= (uint8_t)*s++;
        h *= 16777619u;
    }
    return 0x40000000u | (h & 0x3fffffffu);
}

/* attach the canonical key to every member of a parsed stage struct, using the
 * stage's reflected SPIR-V interface (name -> location). Location 0 is valid, so
 * "not found" is tracked separately from the key value. Arrayed varyings are
 * flattened by SPIRV-Cross into "name_0", "name_1" members: they resolve to the
 * base varying's location with their element index recorded separately */
/* does the stage's GL-view module declare a variable or interface-block
 * member of this name?  (The reflection lists carry the block INSTANCE name
 * only, so members are resolved through OpMemberName/OpName.  Block members
 * carry no Location decorations in the module, so the canonical key for them
 * is a stable hash of the resolved name — identical on every side of a
 * linkable program.) */
static bool tess_stage_var_named(GLMContext ctx, MGLTessProgram *tp, int stage,
                                  const char *name)
{
    Program *prog = tp ? tp->program : NULL;
    uint32_t *ir;
    size_t n;

    if (!ctx || !prog || !name || stage < 0 || stage >= _MAX_SHADER_TYPES)
        return false;
    ir = prog->spirv[stage].ir;
    n = prog->spirv[stage].size;
    if (!ir || n < 8)
        return false;

    /* 1) find the name: OpMemberName -> (struct type, member), OpName -> id */
    {
        size_t m = 5;
        int found_member = -1;
        uint32_t found_type = 0, found_id = 0;

        while (m + 3 < n)
        {
            uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
            if (wc == 0) break;
            if (op == 5 /* OpName */ || op == 6 /* OpMemberName */)
            {
                char buf[128];
                size_t o = 0;
                const uint32_t *w = ir + m + (op == 5 ? 2 : 3);
                const uint32_t *we = ir + m + wc;

                for (; w < we && o + 4 < sizeof buf; w++)
                    for (int b = 0; b < 4; b++)
                    {
                        char c = (char)((*w >> (8 * b)) & 255);
                        if (!c) { w = we; break; }
                        buf[o++] = c;
                    }
                buf[o] = 0;
                if (!strcmp(buf, name))
                {
                    if (op == 6)
                    {
                        found_member = (int)ir[m + 2];
                        found_type = ir[m + 1];
                    }
                    else
                        found_id = ir[m + 1];
                    break;
                }
            }
            m += wc;
        }
        if (!found_type && !found_id)
            return false;

    }
    return true;
}

static void tess_annotate_keys(GLMContext ctx, MGLTessProgram *tp, int stage,
                               struct MslStruct *s, const SpirvResourceList *rl)
{
    for (int i = 0; i < s->count; i++)
    {
        MGLTessPatchMember *m = &s->members[i];
        uint32_t key = 0;
        bool found = false;
        char base[128];                 /* the bare GL name of the varying */
        uint32_t idx = m->array_idx;

        /* builtins first: gl_Position & co must key to their fixed slots on every
         * side of the record.  The generic path below would otherwise name-hash
         * them whenever the stage module declares the variable at top level
         * (the vertex stage's gl_Position output), splitting the record into a
         * producer-written and a consumer-read copy that never alias. */
        key = tess_builtin_key(m->name);
        if (key) { m->key = key; continue; }

        base[0] = 0;
        if (rl)
            for (GLuint j = 0; j < rl->count; j++)
                if (rl->list && !strcmp(rl->list[j].name, m->name))
                {
                    key = (uint32_t)rl->list[j].location;
                    found = true;
                    break;
                }
        if (!found)
        {
            /* flattened array element ("<name>_<q>") and/or interface-block
             * member reached with the instance prefix ("<instance>_<member>",
             * anonymous blocks as "m_<id>_<member>").  GL varying names
             * commonly end in digits themselves ("variable0"), so both the
             * plain and the element-stripped reading are tried, validated
             * against the stage module's declared names — the same varying
             * then annotates to the same key on every side of the record and
             * the array/flattened merge in tess_canon_union can fire */
            uint32_t fidx = m->array_idx;

            base[0] = 0;
            for (const char *b = m->name; !base[0] && (b = strchr(b, '_')); b++)
            {
                const char *suf = b + 1;
                size_t n = strlen(suf);
                char b2[128];

                if (tess_stage_var_named(ctx, tp, stage, suf))
                {
                    snprintf(base, sizeof base, "%s", suf);
                    break;
                }
                while (n > 1 && suf[n - 1] >= '0' && suf[n - 1] <= '9')
                    n--;
                if (n > 1 && suf[n - 1] == '_')
                {
                    snprintf(b2, sizeof b2, "%.*s", (int)(n - 1), suf);
                    if (tess_stage_var_named(ctx, tp, stage, b2))
                    {
                        snprintf(base, sizeof base, "%s", b2);
                        fidx = (uint32_t)atoi(suf + n);
                    }
                }
            }
            if (!base[0])
            {
                char b2[128];
                size_t n = strlen(m->name);

                if (tess_stage_var_named(ctx, tp, stage, m->name))
                    snprintf(base, sizeof base, "%s", m->name);
                else
                {
                    while (n > 1 && m->name[n - 1] >= '0' && m->name[n - 1] <= '9')
                        n--;
                    if (n > 1 && m->name[n - 1] == '_')
                    {
                        snprintf(b2, sizeof b2, "%.*s", (int)(n - 1), m->name);
                        if (tess_stage_var_named(ctx, tp, stage, b2))
                        {
                            snprintf(base, sizeof base, "%s", b2);
                            fidx = (uint32_t)atoi(m->name + n);
                        }
                    }
                }
            }
            if (base[0])
            {
                /* the split base may itself be a reflected varying: key by its
                 * location so the producer's whole-array member (keyed by the
                 * same location) aliases these flattened elements */
                bool matched = false;
                if (rl)
                    for (GLuint j = 0; j < rl->count; j++)
                    if (rl->list)
                    {
                        const char *n = rl->list[j].name;
                        size_t nl = n ? strlen(n) : 0, bl = strlen(base);
                        if (!n || !strcmp(n, base) ||
                            (nl == bl + 3 && !strncmp(n, base, bl) && !strcmp(n + bl, "[0]")))
                        {
                            key = (uint32_t)rl->list[j].location;   /* 0 is a valid location */
                            matched = true;
                            break;
                        }
                    }
                if (!matched) key = tess_name_hash(base);
                m->array_idx = fidx;
                found = true;
            }
        }
        if (!found)
            key = tess_builtin_key(m->name);
        if (!found && !key)
            key = tess_name_hash(m->name);   /* unreflected (block member / gl_ extra) */
        m->key = key;
    }
}

/* Interface blocks reach the MSL stage structs as flattened members whose names
 * SPIRV-Cross prefixes differently on the two sides of an interface
 * ("<instance>_<member>" on the consumer, "m_<id>_<member>" for an anonymous
 * block on the producer), so neither the reflected names nor the name hash
 * line up: the producer's and the consumer's copy of the same GL member
 * become two different record slots and the value never crosses the stage.
 * Pair such leftover members positionally — SPIRV-Cross emits block members in
 * declaration order and both stages declare the same block — requiring the
 * same MSL type, the same array element, and a common suffix that starts at a
 * '_' boundary in both names.  GL requires matching blocks across stages, so
 * the order (and types) agree. */
static void tess_pair_iface_members(struct MslStruct *a, struct MslStruct *b)
{
    static uint32_t next_pair_key = 0x60000000u;
    char used[MGL_TESS_MAX_PATCH_MEMBERS];

    if (!a || !b || a == b) return;
    memset(used, 0, sizeof used);
    for (int i = 0; i < a->count; i++)
    {
        MGLTessPatchMember *ma = &a->members[i];
        size_t na;

        if ((ma->key & 0xc0000000u) != 0x40000000u) continue;   /* only unreflected members */
        na = strlen(ma->name);
        for (int j = 0; j < b->count; j++)
        {
            MGLTessPatchMember *mb = &b->members[j];
            size_t nb, k;

            if (used[j] || (mb->key & 0xc0000000u) != 0x40000000u) continue;
            if (strcmp(ma->type, mb->type) != 0) continue;
            if (ma->array_idx != mb->array_idx) continue;
            nb = strlen(mb->name);
            k = 0;
            while (k < na && k < nb && ma->name[na - 1 - k] == mb->name[nb - 1 - k]) k++;
            if (!k) continue;
            /* the shared suffix must be a whole component of both names: it
             * consumes a name entirely, starts at a '_' separator, or the
             * character before it is the separator */
            if (!((k == na || ma->name[na - k] == '_' || ma->name[na - k - 1] == '_') &&
                  (k == nb || mb->name[nb - k] == '_' || mb->name[nb - k - 1] == '_'))) continue;
            ma->key = mb->key = next_pair_key++;
            used[j] = 1;
            if (tess_trace())
                fprintf(stderr, "MGL tess: paired block member %s <-> %s\n", ma->name, mb->name);
            break;
        }
    }
}

struct CanonRec {
    MGLTessPatchMember m[MGL_TESS_MAX_PATCH_MEMBERS];
    int    n;
    size_t size;        /* record stride, tail-padded to the struct alignment */
};

/* canonical record = union of the stages' members by key, sorted by key; each key's
 * slot takes the largest member any stage declares there (the producer's data must
 * fit even if a consumer declares a narrower type it never reads). A member whose
 * type is an arrayed varying covers the same slots as another stage's flattened
 * "name_i" elements of that varying: the two merge into the wider (array) entry */
static bool tess_canon_union(struct CanonRec *out, struct MslStruct *const *in, int nin)
{
    memset(out, 0, sizeof *out);
    for (int s = 0; s < nin; s++)
    {
        if (!in[s])
            continue;
        for (int i = 0; i < in[s]->count; i++)
        {
            MGLTessPatchMember *m = &in[s]->members[i];
            bool m_arr = strstr(m->type, "spvUnsafeArray") != NULL;
            int found = -1;
            bool absorbed = false;

            if (m->size == 0)
                continue;               /* patch_control_point reference and friends */
            for (int j = 0; j < out->n; j++)
            {
                bool o_arr = strstr(out->m[j].type, "spvUnsafeArray") != NULL;

                if (out->m[j].key != m->key)
                    continue;
                if (o_arr || m_arr ||
                    (!o_arr && !m_arr && out->m[j].array_idx == m->array_idx))
                {
                    found = j;          /* same varying (whole array or an element) */
                    if (!o_arr && m_arr)
                    {
                        /* the array covers the elements already registered —
                         * including the other flattened elements of the same
                         * varying registered as their own slots */
                        out->m[j] = *m;
                        out->m[j].offset = 0;
                        for (int q = 0; q < out->n; q++)
                        {
                            if (q == j || out->m[q].key != m->key) continue;
                            if (strstr(out->m[q].type, "spvUnsafeArray") != NULL) continue;
                            memmove(&out->m[q], &out->m[q + 1],
                                    (out->n - q - 1) * sizeof(out->m[0]));
                            out->n--;
                            if (q < j) j--;
                            q--;
                        }
                    }
                    break;
                }
                if (o_arr && !m_arr && m->array_idx != 0)
                    absorbed = true;   /* element of an array slot already present */
            }
            if (absorbed)
                continue;
            if (found < 0)
            {
                if (out->n >= MGL_TESS_MAX_PATCH_MEMBERS)
                    return false;
                out->m[out->n] = *m;
                out->m[out->n].offset = 0;
                out->n++;
            }
            else if (m->size > out->m[found].size)
            {
                /* keep the wider type, remember the member that provides it */
                snprintf(out->m[found].type, sizeof out->m[found].type, "%s", m->type);
                snprintf(out->m[found].name, sizeof out->m[found].name, "%s", m->name);
                out->m[found].size = m->size;
                out->m[found].array_idx = m->array_idx;
            }
        }
    }

    /* sort by key (insertion sort: n is bounded by MGL_TESS_MAX_PATCH_MEMBERS) */
    for (int i = 1; i < out->n; i++)
    {
        MGLTessPatchMember tmp = out->m[i];
        int j = i - 1;
        while (j >= 0 && out->m[j].key > tmp.key)
        {
            out->m[j + 1] = out->m[j];
            j--;
        }
        out->m[j + 1] = tmp;
    }

    /* byte layout: C++ natural alignment, struct tail-padded to its alignment */
    size_t cursor = 0, maxalign = 1;
    for (int i = 0; i < out->n; i++)
    {
        size_t align = 1, size = 0;
        if (!msl_element_type(out->m[i].type, &align, &size))
            return false;               /* unknown member type: keep the stage structs */
        out->m[i].offset = round_up(cursor, align);
        cursor = out->m[i].offset + size;
        out->m[i].size = size;
        if (align > maxalign)
            maxalign = align;
    }
    out->size = round_up(cursor, maxalign);
    return true;
}

/* element count of an MSL arrayed varying type ("spvUnsafeArray<T, N>"), 1 otherwise */
static uint32_t msl_array_len(const char *type)
{
    const char *at = strstr(type, "spvUnsafeArray<");

    if (!at)
        return 1;
    at = strrchr(at, ',');
    while (at && *at && (*at < '0' || *at > '9'))
        at++;
    return at ? (uint32_t)strtoul(at, NULL, 10) : 1;
}

/* does the stage's own struct already match the canonical layout? */
static bool tess_canon_matches(const struct MslStruct *own, const struct CanonRec *canon)
{
    if (!own || own->count != canon->n)
        return false;
    for (int i = 0; i < canon->n; i++)
        if (own->members[i].key != canon->m[i].key ||
            strcmp(own->members[i].type, canon->m[i].type))
            return false;
    return true;
}

/* element type of "spvUnsafeArray<T, N>" (text between '<' and the last ','), with the
 * inner template name already renamed when rename_arr; false when type is not an array */
static bool tess_canon_element_type(const char *type, bool rename_arr, char *out, size_t cap)
{
    const char *at = strstr(type, "spvUnsafeArray<");
    const char *end;
    size_t n;

    if (!at)
        return false;
    at += strlen("spvUnsafeArray<");
    end = strrchr(type, ',');
    if (!end || end < at)
        return false;
    n = (size_t)(end - at);
    while (n > 0 && at[n - 1] == ' ')
        n--;
    if (n >= cap)
        return false;
    memcpy(out, at, n);
    out[n] = 0;
    if (rename_arr)
        for (char *p = out; (p = strstr(p, "spvUnsafeArray")) != NULL; )
        {
            memcpy(p, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
            p += strlen("spvUnsafeArray");
        }
    return true;
}

/* copy of a member type with spvUnsafeArray renamed to our own template (the plain-TES
 * MSL declares mgl_spv_arr in the prelude, SPIRV-Cross's template only exists in the
 * kernel forms) */
static void tess_type_renamed(const char *type, bool rename_arr, char *out, size_t cap)
{
    snprintf(out, cap, "%s", type);
    if (!rename_arr)
        return;
    for (char *p = out; (p = strstr(p, "spvUnsafeArray")) != NULL; )
    {
        memcpy(p, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
        p += strlen("spvUnsafeArray");
    }
}

/* one canonical member as struct-body lines for a stage. stage/nstage: the members the
 * stage's own struct declares for this key. A canonical slot that is an ARRAY while the
 * stage flattened it into per-element members ("name_i", array_idx = element index) is
 * re-expanded to those members — they tile the slot exactly, and the stage's code refers
 * to them by their own names and (scalar/vector) types. rename_arr: see above. */
static void str_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...);

static void tess_canon_member_lines(char **buf, size_t *len, size_t *cap,
                                    const MGLTessPatchMember *cm,
                                    const MGLTessPatchMember *stage, int nstage,
                                    bool rename_arr, bool *first)
{
    char type[160];
    bool canon_arr = strstr(cm->type, "spvUnsafeArray") != NULL;
    int nflat = 0;

    tess_type_renamed(cm->type, rename_arr, type, sizeof type);

    if (stage)
        for (int j = 0; j < nstage; j++)
            if (stage[j].key == cm->key && stage[j].size > 0 &&
                strstr(stage[j].type, "spvUnsafeArray") == NULL)
                nflat++;
    if (canon_arr && nflat > 0)
    {
        /* the canonical slot is the whole array, the stage uses flattened elements */
        char elem[120];
        uint32_t n = msl_array_len(cm->type);

        if (tess_canon_element_type(cm->type, rename_arr, elem, sizeof elem))
            for (uint32_t e = 0; e < n; e++)
            {
                const char *nm = NULL;
                char pad[64];

                if (stage)
                    for (int j = 0; j < nstage; j++)
                        if (stage[j].key == cm->key && stage[j].size > 0 &&
                            stage[j].array_idx == e &&
                            strstr(stage[j].type, "spvUnsafeArray") == NULL)
                        {
                            nm = stage[j].name;
                            break;
                        }
                if (!nm)
                {
                    snprintf(pad, sizeof pad, "mgl_k%08x_%u", cm->key, e);
                    nm = pad;
                }
                str_append(buf, len, cap, "%s    %s %s;\n", *first ? "\n" : "", elem, nm);
                *first = false;
            }
        return;
    }

    {
        const char *nm = NULL;
        char pad[48];

        if (stage)
            for (int j = 0; j < nstage; j++)
                if (stage[j].key == cm->key && stage[j].size > 0)
                {
                    nm = stage[j].name;
                    break;
                }
        if (!nm)
        {
            snprintf(pad, sizeof pad, "mgl_k%08x", cm->key);
            nm = pad;
        }
        str_append(buf, len, cap, "%s    %s %s;\n", *first ? "\n" : "", type, nm);
        *first = false;
    }
}

/* canonical struct body text for one stage: the stage's member names for members it
 * declares, syntactically-valid pads for the rest (never referenced by its code) */
/* A scalar-double VS output renders as uint2 in the VS's own MSL but occupies
 * the full uint4 slot in the canonical control-point record (the consumer
 * side's repr, "one uint4 = one 128-bit GL location").  Widen the whole-
 * variable stores `out.<name> = <expr>;` so the canonicalized VS kernel still
 * compiles.  (Reads of such members always go through `.xy` and work either
 * way.) */
static void tess_widen_double_stores(char **msl_p, const struct MslStruct *stage,
                                     const struct CanonRec *canon)
{
    char pat[160];

    for (int i = 0; i < canon->n; i++)
    {
        const MGLTessPatchMember *cm = &canon->m[i];
        const MGLTessPatchMember *sm = NULL;

        for (int j = 0; j < stage->count; j++)
            if (stage->members[j].key == cm->key) { sm = &stage->members[j]; break; }
        if (!sm || !sm->name[0])
            continue;
        if (!strstr(cm->type, "uint4") || !strstr(sm->type, "uint2"))
            continue;
        if (strstr(sm->type, "spvUnsafeArray") || strstr(cm->type, "spvUnsafeArray"))
            continue;   /* per-element stores only happen on the TCS side */

        snprintf(pat, sizeof pat, "out.%s = ", sm->name);
        {
            char *hit = strstr(*msl_p, pat);

            while (hit)
            {
                size_t e0 = (size_t)(hit - *msl_p) + strlen(pat);
                const char *sc = strchr(*msl_p + e0, ';');
                size_t elen, add, olen;
                char *grown;

                if (!sc)
                    break;
                elen = (size_t)(sc - (*msl_p + e0));
                add = strlen("uint4(") + strlen(", 0u, 0u)");
                olen = strlen(*msl_p);
                grown = (char *)realloc(*msl_p, olen + add + 1);
                if (!grown)
                    return;
                *msl_p = grown;
                /* layout: "uint4(" expr ", 0u, 0u)" <tail from ';'> — the tail
                 * moves first (it is the only region beyond the expr), then the
                 * expr itself shifts right past "uint4(" */
                memmove(*msl_p + e0 + 6 + elen + 9, *msl_p + e0 + elen,
                        olen - (e0 + elen) + 1);
                memmove(*msl_p + e0 + 6, *msl_p + e0, elen);
                memcpy(*msl_p + e0, "uint4(", 6);
                memcpy(*msl_p + e0 + 6 + elen, ", 0u, 0u)", 9);
                hit = strstr(*msl_p + e0 + 6 + elen + 9, pat);
            }
        }
    }
}

static void tess_canon_body(char *out, size_t cap, const struct MslStruct *own,                            const struct CanonRec *canon)
{
    char *buf = NULL;
    size_t len = 0, bc = 0;
    bool first = true;

    out[0] = 0;
    for (int i = 0; i < canon->n; i++)
    {
        /* not renamed here: msl_rewrite_struct_body renames (and injects the template)
         * only when the original MSL does not define spvUnsafeArray before the struct */
        tess_canon_member_lines(&buf, &len, &bc, &canon->m[i],
                                own ? own->members : NULL, own ? own->count : 0,
                                false, &first);
        if (!buf)
            return;
    }
    snprintf(out, cap, "%s", buf);
    free(buf);
}

/* replace the body of "struct NAME { ... };" in msl with new_body; returns a malloc'd
 * copy, or NULL when the struct is absent (nothing to rewrite). The spvUnsafeArray
 * template is injected when the canonical body needs it and the original declares it
 * later (SPIRV-Cross emits it after the stage's own structs in kernel form) */
static const char kSpvUnsafeArray[] =
    "template <typename T, size_t Num>\n"
    "struct mgl_spv_arr\n"
    "{\n"
    "    T elements[Num ? Num : 1];\n"
    "    \n"
    "    thread T& operator [] (size_t pos) thread\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const thread T& operator [] (size_t pos) const thread\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    device T& operator [] (size_t pos) device\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const device T& operator [] (size_t pos) const device\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    constexpr const constant T& operator [] (size_t pos) const constant\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    threadgroup T& operator [] (size_t pos) threadgroup\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "};\n";

static char *msl_rewrite_struct_body(const char *msl, const char *name, const char *body)
{
    char head[256];
    const char *p, *open, *close;
    const char *tmpl = NULL;
    char *out, *new_body;
    size_t prefix, tail;

    snprintf(head, sizeof head, "struct %s\n{", name);
    p = strstr(msl, head);
    if (!p)
    {
        snprintf(head, sizeof head, "struct %s {", name);
        p = strstr(msl, head);
    }
    if (!p)
        return NULL;
    open = strchr(p, '{');
    close = open ? strchr(open, '}') : NULL;
    if (!open || !close)
        return NULL;

    /* array members reference spvUnsafeArray: if its template is not already defined
     * ahead of this struct (SPIRV-Cross emits it after the stage structs in kernel
     * form), use our own name and declare it in front — the two never collide */
    new_body = strdup(body);
    if (new_body && strstr(new_body, "spvUnsafeArray"))
    {
        bool before_us = false, before_orig = false;

        for (const char *q = msl; q < p; q++)
        {
            if (!before_orig && !strncmp(q, "struct spvUnsafeArray", 21))
                before_orig = true;
            if (!before_us && !strncmp(q, "struct mgl_spv_arr", 18))
                before_us = true;
        }
        if (!before_orig)
        {
            for (char *at = new_body; (at = strstr(at, "spvUnsafeArray")) != NULL; )
            {
                memcpy(at, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
                at += strlen("spvUnsafeArray");
            }
            if (!before_us)
                tmpl = kSpvUnsafeArray;
        }
    }
    if (!new_body)
        return NULL;

    prefix = (size_t)(open + 1 - msl);
    tail = strlen(close);                 /* "};\n..." to the end */
    out = (char *)malloc(prefix + strlen(new_body) + tail + 1 + (tmpl ? strlen(tmpl) : 0));
    if (!out)
    {
        free(new_body);
        return NULL;
    }
    if (tmpl)
    {
        /* the template goes in front of the whole struct declaration */
        size_t decl = (size_t)(p - msl);

        memcpy(out, msl, decl);
        memcpy(out + decl, tmpl, strlen(tmpl));
        memcpy(out + decl + strlen(tmpl), msl + decl, prefix - decl);
        memcpy(out + prefix + strlen(tmpl), new_body, strlen(new_body));
        memcpy(out + prefix + strlen(tmpl) + strlen(new_body), close, tail + 1);
    }
    else
    {
        memcpy(out, msl, prefix);
        memcpy(out + prefix, new_body, strlen(new_body));
        memcpy(out + prefix + strlen(new_body), close, tail + 1);
    }
    free(new_body);
    return out;
}

/* ------------------------------------------------------------------ MSL rewrites
 * Called from program.c's compile_stage right after SPIRV-Cross produced the MSL.
 */

static void str_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap, ap2;
    int need;
    char tmp[2048];
    char *heap = NULL;
    const char *src;
    char *nb;

    va_start(ap, fmt);
    va_copy(ap2, ap);
    need = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (need < 0)
    {
        va_end(ap2);
        return;
    }
    if ((size_t)need >= sizeof tmp)
    {
        /* long appends (whole shader bodies) do not fit the stack buffer: format
         * into the heap so they are not truncated */
        heap = (char *)malloc((size_t)need + 1);
        if (!heap)
        {
            va_end(ap2);
            return;
        }
        vsnprintf(heap, (size_t)need + 1, fmt, ap2);
    }
    va_end(ap2);
    src = heap ? heap : tmp;
    if (*len + (size_t)need + 1 > *cap)
    {
        size_t nc = (*len + (size_t)need + 1) * 2;

        nb = (char *)realloc(*buf, nc);
        if (!nb)
        {
            free(heap);
            return;
        }
        *buf = nb;
        *cap = nc;
    }
    memcpy(*buf + *len, src, (size_t)need + 1);
    *len += (size_t)need;
    free(heap);
}

/* replace an exact occurrence of old with new inside a growable buffer */
static void buf_replace(char **buf, size_t *len, size_t *cap, const char *old, const char *new)
{
    char *at = strstr(*buf, old);
    size_t oldlen = strlen(old), newlen = strlen(new);

    if (!at)
        return;
    if (newlen > oldlen)
    {
        size_t prefix = (size_t)(at - *buf);
        *cap = *len + newlen - oldlen + 1;
        *buf = (char *)realloc(*buf, *cap);
        at = *buf + prefix;
    }
    memmove(at + newlen, at + oldlen, *len - (size_t)(at - *buf) - oldlen + 1);
    memcpy(at, new, newlen);
    *len += newlen - oldlen;
}

/* VS kernel (SPIRV-Cross VERTEX_FOR_TESSELLATION):
 *  - route the base vertex/instance through our params buffer (GL: gl_VertexID includes
 *    base vertex, gl_InstanceID excludes base instance)
 *  - drop the [[grid_size]] / [[grid_origin]] parameters and provide them as locals from
 *    the params buffer: fetching stage-in attributes silently returns zeros when a kernel
 *    also takes [[grid_size]] (AGX driver bug, verified with a bisected probe)
 */
static char *rewrite_vs_kernel(const char *msl)
{
    char *out = NULL;
    size_t len = 0, cap = 0;

    str_append(&out, &len, &cap, "%s", msl);

    /* parameters -> locals (indices must not collide with the vertex descriptor) */
    buf_replace(&out, &len, &cap, ", uint3 spvStageInputSize [[grid_size]]", "");
    buf_replace(&out, &len, &cap, ", uint3 spvDispatchBase [[grid_origin]]", "");

    /* insert the locals right after the opening brace.  The buffer is grown with realloc,
     * so the insertion point is kept as an offset — a moved allocation invalidated the
     * pointer before, the prologue landed in freed memory and the kernel compiled with
     * undeclared spvStageInputSize / mgl_vs_params (KHR-GL46 tessellation clusters) */
    {
        char *open = strstr(out, "kernel void ");
        open = open ? strchr(open, '{') : NULL;
        if (!open)
        {
            free(out);
            return NULL;
        }

        char prologue[512];
        int n = snprintf(prologue, sizeof prologue,
                         "\n    uint3 spvStageInputSize = uint3(mgl_vs_params[1], mgl_vs_params[2], 1u);\n"
                         "    uint3 spvDispatchBase = uint3(mgl_vs_params[0], mgl_vs_params[3], 0u);\n");
        (void)n;
        size_t plen = strlen(prologue), open_off = (size_t)(open + 1 - out);
        size_t taillen = len - open_off;
        char *grown = (char *)realloc(out, len + plen + 1);
        if (!grown) { free(out); return NULL; }
        out = grown; cap = len + plen + 1;
        char *ins = out + open_off;
        memmove(ins + plen, ins, taillen + 1);
        memcpy(ins, prologue, plen);
        len += plen;
    }

    /* GL id semantics (replaces the uses of the removed locals) */
    buf_replace(&out, &len, &cap,
                "uint gl_VertexID = as_uint(gl_GlobalInvocationID.x + spvDispatchBase.x);",
                "uint gl_VertexID = as_uint(gl_GlobalInvocationID.x + mgl_vs_params[0]);");
    buf_replace(&out, &len, &cap,
                "uint gl_VertexID = gl_GlobalInvocationID.x + spvDispatchBase.x;",
                "uint gl_VertexID = gl_GlobalInvocationID.x + mgl_vs_params[0];");
    buf_replace(&out, &len, &cap,
                "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y + spvDispatchBase.y);",
                "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y);");
    buf_replace(&out, &len, &cap,
                "uint gl_InstanceID = gl_GlobalInvocationID.y + spvDispatchBase.y;",
                "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y);");

    /* append the params parameter to the kernel signature */
    {
        char *p = strstr(out, "kernel void ");
        char *paren = p ? strchr(p, '(') : NULL;
        int depth = 0;
        char *close = NULL;

        if (!paren)
        {
            free(out);
            return NULL;
        }
        for (char *q = paren; *q; q++)
        {
            if (q[0] == '[' && q[1] == '[') { const char *e = strstr(q, "]]"); if (e) { q = (char *)e; continue; } }
            if (*q == '(') depth++;
            else if (*q == ')') { depth--; if (depth == 0) { close = q; break; } }
        }
        if (!close)
        {
            free(out);
            return NULL;
        }

        char add[128];
        size_t addlen, taillen, close_off;
        snprintf(add, sizeof add, ", constant uint* mgl_vs_params [[buffer(%d)]]", MGL_TESS_VS_PARAMS);
        addlen = strlen(add);
        taillen = strlen(close);
        close_off = (size_t)(close - out);          /* offset: realloc may move the buffer */
        char *grown = (char *)realloc(out, len + addlen + 1);
        if (!grown) { free(out); return NULL; }
        out = grown; cap = len + addlen + 1;
        close = out + close_off;
        memmove(close + addlen, close, taillen + 1);
        memcpy(close, add, addlen);
        len += addlen;
    }

    return out;
}

/* ------------------------------------------------------------------ TES rewrite
 * SPIRV-Cross emits the TES as a post-tessellation vertex function
 *   [[patch(triangle, N)]] vertex out_t entry(in_patch_t patchIn [[stage_in]],
 *                                            uint gl_PrimitiveID [[patch_id]],
 *                                            float3 gl_TessCoord [[position_in_patch]])
 * This rewrite turns it into a plain vertex function reading the tessellation
 * coordinates / control points / patch data from buffers indexed by [[vertex_id]],
 * which works on GPUs without hardware tessellation:
 *   vertex out_t entry(device const float4* coords [[buffer(27)]],
 *                      device const in_t* cp [[buffer(28)]],
 *                      device const uchar* patch [[buffer(29)]],
 *                      constant uint* params [[buffer(26)]],
 *                      uint mgl_vid [[vertex_id]])
 */
static const char *kTessPrelude =
    "template <typename T>\n"
    "struct mgl_cp_ref\n"
    "{\n"
    "    device const T *mgl_p;\n"
    "    uint mgl_n;\n"
    "    device const T & operator[] (size_t i) const { return mgl_p[i]; }\n"
    "    uint size() const { return mgl_n; }\n"
    "};\n\n";

static char *rewrite_tes(const char *msl, const MGLTessProgram *tp)
{
    const char *sig = strstr(msl, "[[ patch(");
    if (!sig)
        sig = strstr(msl, "[[patch(");
    if (!sig)
        { TES_REWRITE_FAIL(9); }

    const char *vertex_kw = strstr(sig, "vertex ");
    if (!vertex_kw || (size_t)(vertex_kw - sig) > 64)
        { TES_REWRITE_FAIL(13); }
    const char *sig_open = strchr(vertex_kw, '(');
    if (!sig_open)
        { TES_REWRITE_FAIL(16); }

    /* matching close paren of the parameter list */
    const char *sig_close = NULL;
    int depth = 0;
    for (const char *q = sig_open; *q; q++)
    {
        if (q[0] == '[' && q[1] == '[')
        {
            const char *e = strstr(q, "]]");
            if (e) { q = e; continue; }
        }
        if (*q == '(') depth++;
        else if (*q == ')') { if (--depth == 0) { sig_close = q; break; } }
    }
    if (!sig_close)
        { TES_REWRITE_FAIL(32); }

    /* the opening brace of the function body */
    const char *body_open = strchr(sig_close, '{');
    if (!body_open)
        { TES_REWRITE_FAIL(37); }

    /* return type and entry name: the entry name is the last identifier before '(' */
    char entry_ret[128], entry_name[128];
    {
        const char *name_start = sig_open;
        while (name_start > vertex_kw && *(name_start - 1) != ' ' && *(name_start - 1) != '\n' && *(name_start - 1) != '\t')
            name_start--;
        if (name_start <= vertex_kw + 7 || name_start == sig_open)
            return NULL;
        snprintf(entry_name, sizeof entry_name, "%.*s", (int)(sig_open - name_start), name_start);
        snprintf(entry_ret, sizeof entry_ret, "%.*s", (int)(name_start - vertex_kw - 8), vertex_kw + 7);
    }

    /* patchIn struct type name: first word of the parameter list. A TES that reads
     * neither its control points nor any per-patch varying gets no stage_in struct at
     * all (the first parameter is then gl_TessCoord or patch_id) — flagged and every
     * patchIn-related rewrite is skipped */
    char patchin_name[128];
    bool no_patchin = false;
    {
        const char *p = sig_open + 1;
        while (*p == ' ' || *p == '\n') p++;
        const char *e = p;
        while (*e && *e != ' ' && *e != '\n') e++;
        if (e == p || (size_t)(e - p) >= sizeof patchin_name)
            { TES_REWRITE_FAIL(62); }
        snprintf(patchin_name, sizeof patchin_name, "%.*s", (int)(e - p), p);

        char head[160];
        snprintf(head, sizeof head, "struct %s\n{", patchin_name);
        if (!strstr(msl, head))
        {
            snprintf(head, sizeof head, "struct %s {", patchin_name);
            if (!strstr(msl, head))
                no_patchin = true;      /* no stage_in struct: no inputs to rebuild */
        }
    }

    /* control point struct type: from patch_control_point<T> in the patchIn struct */
    char cp_type[128];
    {
        char head[160];
        const char *p = NULL;
        const char *pcp, *t0, *t1;

        if (!no_patchin)
        {
            snprintf(head, sizeof head, "struct %s\n{", patchin_name);
            p = strstr(msl, head);
            if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
        }
        pcp = p ? strstr(p, "patch_control_point<") : NULL;
        if (pcp && pcp < strchr(p, '}'))
        {
            t0 = pcp + strlen("patch_control_point<");
            t1 = strchr(t0, '>');
            if (!t1 || (size_t)(t1 - t0) >= sizeof cp_type)
                { TES_REWRITE_FAIL(82); }
            memcpy(cp_type, t0, (size_t)(t1 - t0));
            cp_type[t1 - t0] = 0;
        }
        else
        {
            /* no control points referenced: the type is unused, but must name a type */
            snprintf(cp_type, sizeof cp_type, "mgl_cp_rec");
        }
    }

    struct MslStruct patch_in;
    memset(&patch_in, 0, sizeof patch_in);
    if (!no_patchin && !msl_parse_struct(msl, patchin_name, &patch_in))
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: rewrite_tes: patchIn struct parse failed (%s)\n", patchin_name);
        { TES_REWRITE_FAIL(92); }
    }
    if (tess_trace())
        fprintf(stderr, "MGL tess: rewrite_tes: sig ok, cp=%s patchin=%s members=%d cpref=%d\n",
                cp_type, patchin_name, patch_in.count, patch_in.cp_ref_member);

    /* ---- assemble the rewritten source ---- */
    char *out = NULL;
    size_t len = 0, cap = 0;

    const char *ins = strstr(msl, "using namespace metal;");
    if (!ins)
        { TES_REWRITE_FAIL(104); }
    ins += strlen("using namespace metal;");
    str_append(&out, &len, &cap, "%.*s%s", (int)(ins - msl), msl, kTessPrelude);

    /* everything between the prelude and "[[ patch(", with the canonical control point
     * record type declared immediately after the prelude, BEFORE the stage's own
     * structs (the patched patchIn struct references it) */
    if (tp && tp->num_cp_record > 0)
    {
        bool needs_arr = false;

        for (int j = 0; j < tp->num_cp_record && !needs_arr; j++)
            if (strstr(tp->cp_record[j].type, "spvUnsafeArray"))
                needs_arr = true;
        if (needs_arr)
            str_append(&out, &len, &cap, "%s", kSpvUnsafeArray);
        str_append(&out, &len, &cap, "\nstruct mgl_cp_rec\n{\n");
        {
            bool first = true;

            for (int j = 0; j < tp->num_cp_record; j++)
            {
                /* renamed: the template above is ours; array slots the TES flattened
                 * ("tc_arr_0", "tc_arr_1") are re-expanded so the record's members have
                 * the names and element types the TES code accesses them by */
                tess_canon_member_lines(&out, &len, &cap, &tp->cp_record[j],
                                        tp->tes_cp, tp->num_tes_cp, true, &first);
            }
        }
        str_append(&out, &len, &cap, "};\n");
    }
    else
    {
        /* no canonical record (TES reads neither control points nor per-patch varyings,
         * or pairing is off): the entry signature still names the control point buffer,
         * so the fallback type must be declared here — a TES with no cp inputs gets
         * cp_type == "mgl_cp_rec" and nothing would declare it ("unknown type name
         * 'mgl_cp_rec'" from newLibraryWithSource) */
        str_append(&out, &len, &cap, "\nstruct mgl_cp_rec\n{\n    float4 gl_Position;\n};\n");
    }
    str_append(&out, &len, &cap, "%.*s", (int)(sig - ins), ins);

    /* new signature.  The original entry point's buffer/texture/sampler
     * parameters must survive: everything except the patch stage_in struct,
     * gl_TessCoord / patch_id, which the prologue rebuilds from the coordinate
     * stream (a TES reading a uniform block would otherwise lose it —
     * "use of undeclared identifier" in the generated body). */
    char *kept = NULL;
    size_t klen = 0, kcap = 0;
    {
        const char *ps = sig_open + 1;
        while (ps < sig_close)
        {
            const char *e = ps;
            int depth = 0;

            while (e < sig_close && (depth || *e != ','))
            {
                if (*e == '(' || *e == '<') depth++;
                else if (*e == ')' || *e == '>') depth--;
                e++;
            }
            {
                size_t n = (size_t)(e - ps);
                char ent[512];

                if (n >= sizeof ent) n = sizeof ent - 1;
                memcpy(ent, ps, n);
                ent[n] = 0;
                if (!strstr(ent, "[[stage_in]]") && !strstr(ent, "position_in_patch") &&
                    !strstr(ent, "[[patch_id]]"))
                    str_append(&kept, &klen, &kcap, "%s%s", klen ? ", " : "", ent);
            }
            ps = e < sig_close ? e + 1 : e;
            while (ps < sig_close && (*ps == ' ' || *ps == '\n' || *ps == '\t')) ps++;
        }
    }
    str_append(&out, &len, &cap,
               "vertex %s %s(device const float4* mgl_tess_coords [[buffer(%d)]], "
               "device const %s* mgl_tess_cp [[buffer(%d)]], "
               "device const uchar* mgl_tess_patch [[buffer(%d)]], "
               "constant uint* mgl_tess_params [[buffer(%d)]], "
               "uint mgl_vid [[vertex_id]]%s%s)",
               entry_ret, entry_name, MGL_TESS_TES_COORDS,
               (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type, MGL_TESS_TES_CP,
               MGL_TESS_TES_PATCH, MGL_TESS_TES_PARAMS,
               kept && klen ? ", " : "", kept ? kept : "");
    free(kept);

    /* prologue: rebuild patchIn from the buffers. Offsets and the record stride come
     * from the canonical per-patch record (tp->tes_patch_in), so the blob layout is
     * identical no matter which members the TES happens to declare */
    size_t patch_stride = patch_in.size;
    {
        const MGLTessProgram *ctp = tp;
        if (ctp && ctp->num_tes_patch_in)
            patch_stride = ctp->tes_patch_in_size;
    }
    if (!no_patchin)
    {
        str_append(&out, &len, &cap, "\n{\n    %s patchIn;\n", patchin_name);
        for (int i = 0; i < patch_in.count; i++)
        {
            size_t moff = patch_in.members[i].offset;

            if (i == patch_in.cp_ref_member)
                continue;
            if (tp)
                for (int j = 0; j < tp->num_tes_patch_in; j++)
                    if (!strcmp(tp->tes_patch_in[j].name, patch_in.members[i].name) &&
                        !strcmp(tp->tes_patch_in[j].type, patch_in.members[i].type))
                    {
                        moff = tp->tes_patch_in[j].offset;   /* canonical offset */
                        break;
                    }
            /* per-patch record: the blob holds total_patches records of the canonical
             * patch size, indexed by the patch id the tessellator stored in the
             * coordinate stream */
            str_append(&out, &len, &cap,
                       "    patchIn.%s = *(device const %s*)(mgl_tess_patch + "
                       "(size_t)mgl_tess_coords[mgl_vid].w * %zu + %zu);\n",
                       patch_in.members[i].name, patch_in.members[i].type, patch_stride, moff);
        }
        if (patch_in.cp_ref_member >= 0)
            str_append(&out, &len, &cap,
                       "    patchIn.%s = mgl_cp_ref<%s>{ &mgl_tess_cp[(uint)mgl_tess_coords[mgl_vid].w * mgl_tess_params[0]], mgl_tess_params[0] };\n",
                       patch_in.members[patch_in.cp_ref_member].name,
                       (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type);
    }
    else
        str_append(&out, &len, &cap, "\n{\n");

    /* debug hook: dump control point N as the vertex position (MGL_TESS_CPDBG=N) */
    if (tp && tp->num_cp_record > 0)
    {
        const char *n = getenv("MGL_TESS_CPDBG");
        if (n && *n)
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(mgl_tess_cp[%s].gl_Position.xy, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret, n);
        else if (getenv("MGL_TESS_PARAMDBG"))
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(float(mgl_tess_params[0]) * 0.2 - 0.9, float(mgl_tess_params[1]) * 0.2 - 0.9, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret);
        else if (getenv("MGL_TESS_COORDDBG"))
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(mgl_tess_coords[mgl_vid].xy * 1.6 - 0.8, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret);
    }

    /* direct builtin parameters of the original entry point */
    {
        char params[768];
        snprintf(params, sizeof params, "%.*s", (int)(sig_close - sig_open - 1), sig_open + 1);
        if (strstr(params, "[[patch_id]]"))
            str_append(&out, &len, &cap, "    uint gl_PrimitiveID = (uint)mgl_tess_coords[mgl_vid].w;\n");
        if (strstr(params, "float2 gl_TessCoordIn [[position_in_patch]]"))
            str_append(&out, &len, &cap, "    float2 gl_TessCoordIn = mgl_tess_coords[mgl_vid].xy;\n");
        else if (strstr(params, "float3 gl_TessCoord [[position_in_patch]]"))
            str_append(&out, &len, &cap, "    float3 gl_TessCoord = mgl_tess_coords[mgl_vid].xyz;\n");
    }

    /* original body */
    str_append(&out, &len, &cap, "%s", body_open + 1);
    if (getenv("MGL_TESS_STEPS")) { FILE *fq = fopen("/tmp/tessstep0.txt", "w"); if (fq) { fputs(out, fq); fclose(fq); } }

    /* strip [[attribute(N)]] from the two struct declarations and replace
       patch_control_point<T> by mgl_cp_ref<mgl_cp_rec> so the types work outside
       stage_in and the control point stride is the canonical record's */
    const char *decl_names[2] = { no_patchin ? "" : patchin_name, cp_type };
    for (int d = 0; d < 2; d++)
    {
        char head[160];
        const char *p;

        if (!decl_names[d][0])
            continue;                       /* no patchIn struct at all */
        snprintf(head, sizeof head, "struct %s\n{", decl_names[d]);
        p = strstr(out, head);
        if (!p) { snprintf(head, sizeof head, "struct %s {", decl_names[d]); p = strstr(out, head); }
        if (!p)
            continue;
        const char *open = strchr(p, '{');
        const char *close = strchr(open, '}');
        if (!open || !close)
            continue;

        char canon[4096];
        char *canon_buf = NULL;
        const char *newbody = NULL;

        if (d == 1 && tp && tp->num_cp_record > 0)
        {
            /* every TES member must be present in the canonical record (it is, unless
             * the union bailed out) — otherwise keep the stage's own layout */
            bool complete = true;

            for (int k = 0; k < tp->num_tes_cp && complete; k++)
            {
                bool found = false;
                for (int j = 0; j < tp->num_cp_record; j++)
                    if (tp->cp_record[j].key == tp->tes_cp[k].key)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                    complete = false;
            }
            if (complete)
            {
                bool first = true;
                size_t blen = 0, bcap = 0;

                canon[0] = 0;
                for (int j = 0; j < tp->num_cp_record; j++)
                {
                    /* renamed: the plain TES declares our own template in the prelude
                     * (kSpvUnsafeArray), SPIRV-Cross's may not exist in vertex form */
                    tess_canon_member_lines(&canon_buf, &blen, &bcap, &tp->cp_record[j],
                                            tp->tes_cp, tp->num_tes_cp, true, &first);
                    if (!canon_buf)
                        break;
                }
                newbody = canon_buf;
            }
        }

        if (!newbody)
        {
            size_t bodylen = (size_t)(close - open - 1);
            char *body = (char *)malloc(bodylen + 1);
            if (!body)
                continue;
            memcpy(body, open + 1, bodylen);
            body[bodylen] = 0;

            for (char *at = body; (at = strstr(at, "[[")) != NULL; )
            {
                char *e = strstr(at, "]]");
                if (!e) break;
                memmove(at, e + 2, strlen(e + 2) + 1);
            }
            for (char *at = body; (at = strstr(at, "patch_control_point<")) != NULL; )
            {
                size_t plen = strlen("patch_control_point<");
                char *close_ = strchr(at + plen, '>');
                char repl[192];

                snprintf(repl, sizeof repl, "mgl_cp_ref<%s>",
                         (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type);
                if (!close_)
                    break;
                /* drop the original template argument, keep the member name after '>' */
                memmove(at + strlen(repl), close_ + 1, strlen(close_ + 1) + 1);
                memcpy(at, repl, strlen(repl));
                at += strlen(repl);
            }
            newbody = body;
        }

        {
            /* splice "struct X { OLD }" -> "struct X { NEW }".  Offsets are
             * recomputed from strlen(out) (ground truth) because the tracked
             * `len` drifts when the fallback path above already shrank its
             * copied body.  The buffer is only ever GROWN: a shrinking realloc
             * would truncate below `slen`, and the memmove source tail
             * [off+1+bodylen, slen) would read past the allocation. */
            size_t bodylen = (size_t)(close - open - 1);
            size_t slen = strlen(out);
            size_t off = (size_t)(open - out);
            size_t newlen = slen - bodylen + strlen(newbody);
            const char *hold = newbody;

            if (cap < slen + 1 || cap < newlen + 1)
            {
                size_t want = (slen > newlen ? slen : newlen) + 1;
                char *grown = (char *)realloc(out, want);

                if (!grown)
                    return NULL;
                out = grown;
                cap = want;
            }
            memmove((void *)(out + off + 1 + strlen(newbody)),
                    out + off + 1 + bodylen, slen - (off + 1 + bodylen));
            memcpy((void *)(out + off + 1), newbody, strlen(newbody));
            out[newlen] = 0;   /* explicit terminator: never read out[slen] */
            len = newlen;
            if (getenv("MGL_TESS_STEPS")) { char pth[64]; snprintf(pth, sizeof pth, "/tmp/tessstep%d.txt", d + 1); FILE *fq = fopen(pth, "w"); if (fq) { fputs(out, fq); fclose(fq); }
                const char *z = memchr(out, 0, len + 1);
                fprintf(stderr, "SPLICE-POST d=%d newlen=%zu len=%zu strlen=%zu cap=%zu nul_at=%ld ctx=[%.40s]\n",
                        d, newlen, len, strlen(out), cap, z ? (long)(z - out) : -1L, z ? z - 20 : "");
                if (z) { fprintf(stderr, "  hex@nul-8:"); for (int q = -8; q <= 2; q++) fprintf(stderr, " %02x", (unsigned char)z[q]); fprintf(stderr, "\n"); } }
            if (hold != canon)
                free((void *)hold);
        }
    }

    return out;
}

/* hardware TES: SPIRV-Cross's post-tessellation vertex function, with the control point
 * count of [[patch(domain, 0)]] fixed up from the TCS (Metal ignores a mismatch with
 * drawPatches, but gl_PatchVerticesIn reads it) */
static char *rewrite_tes_hardware(const char *msl, int cp_count)
{
    const char *sig = strstr(msl, "[[ patch(");
    char *out = NULL;
    size_t len = 0, cap = 0;

    if (!sig)
        sig = strstr(msl, "[[patch(");
    if (!sig)
        return NULL;

    /* "[[ patch(triangle, 0)]] vertex" -> "[[ patch(triangle, N)]] vertex" */
    {
        const char *comma = strchr(sig, ',');
        const char *digits, *rest;
        if (!comma)
            return NULL;
        digits = comma + 1;
        while (*digits == ' ')
            digits++;
        rest = digits;
        while (*rest >= '0' && *rest <= '9')
            rest++;

        str_append(&out, &len, &cap, "%.*s%d%s", (int)(digits - msl), msl,
                   cp_count > 0 ? cp_count : 3, rest);
    }
    return out;
}

/* public re-substitution used when GL_PATCH_VERTICES changes after link */
char *mgl_tess_rewrite_tes_hw(const char *msl, int n)
{
    return rewrite_tes_hardware(msl, n);
}

/* entry from program.c: rewrite the MSL of a tessellation-pipeline stage */
char *mgl_tess_rewrite_stage_msl(GLMContext ctx, int stage, bool tess_pipeline, const char *msl)
{
    if (!msl)
        return NULL;

    if (stage == _TESS_EVALUATION_SHADER)
    {
        /* keep the native post-tessellation vertex function (hardware path), only
         * fixing the baked control point count; the plain-vertex variant used for
         * isolines / point_mode is generated from it on demand (mgl_tess_plain_msl) */
        int cp_count = 3;
        if (ctx->state.program && ctx->state.program->spirv[_TESS_CONTROL_SHADER].ir)
        {
            int modes[16], count = 0, out_vertices = 0;
            if (spv_tess_modes(ctx->state.program->spirv[_TESS_CONTROL_SHADER].ir,
                               ctx->state.program->spirv[_TESS_CONTROL_SHADER].size,
                               1 /* TessellationControl */, modes, &count, &out_vertices) &&
                out_vertices > 0)
                cp_count = out_vertices;
        }
        return rewrite_tes_hardware(msl, cp_count);
    }

    if (stage == _VERTEX_SHADER && tess_pipeline)
        return rewrite_vs_kernel(msl);

    return NULL;
}

/* plain-vertex variant of the TES (isolines / point_mode): generated from the stored
 * hardware MSL and cached. The caller gets a copy and frees it. */
char *mgl_tess_plain_msl(GLMContext ctx, MGLTessProgram *tp)
{
    if (!tp || !tp->program)
        return NULL;
    if (!tp->plain_msl)
    {
        const char *msl = tp->program->spirv[_TESS_EVALUATION_SHADER].msl_str;
        char *plain = rewrite_tes(msl, tp);
        if (getenv("MGL_DUMP_MSL") && plain)
            fprintf(stderr, "==== plain TES MSL:\n%s\n==== end\n", plain);
        if (!plain)
            return NULL;
        tp->plain_msl = plain;
    }
    return strdup(tp->plain_msl);
}

/* ------------------------------------------------------------------ CPU tessellator
 * GL 4.6 §11.3.4 "Tessellation": generate the tessellation coordinates and the
 * connectivity of the tessellated patch. Vertex arrays are laid out per patch:
 *   coordinates: float4 (u, v, w, patch id) per tessellated vertex
 *   indices:     triangle/line/point lists in the same order
 */

struct TessMesh {
    float    *coords;      /* 4 per vertex */
    unsigned ncoords, coord_cap;
    uint32_t *indices;
    unsigned nindices, index_cap;
};

static bool mesh_reserve(struct TessMesh *m, unsigned more_coords, unsigned more_indices)
{
    if (m->ncoords + more_coords > m->coord_cap)
    {
        unsigned cap = (m->ncoords + more_coords) * 2 + 64;
        float *c = (float *)realloc(m->coords, cap * sizeof(float) * 4);
        if (!c) return false;
        m->coords = c;
        m->coord_cap = cap;
    }
    if (m->nindices + more_indices > m->index_cap)
    {
        unsigned cap = (m->nindices + more_indices) * 2 + 64;
        uint32_t *i = (uint32_t *)realloc(m->indices, cap * sizeof(uint32_t));
        if (!i) return false;
        m->indices = i;
        m->index_cap = cap;
    }
    return true;
}

static unsigned mesh_vertex(struct TessMesh *m, float u, float v, float w, float pid)
{
    if (!mesh_reserve(m, 1, 0))
        return 0;
    m->coords[m->ncoords * 4 + 0] = u;
    m->coords[m->ncoords * 4 + 1] = v;
    m->coords[m->ncoords * 4 + 2] = w;
    m->coords[m->ncoords * 4 + 3] = pid;
    return m->ncoords++;
}

static void mesh_index(struct TessMesh *m, unsigned a, unsigned b, unsigned c)
{
    if (!mesh_reserve(m, 0, 3))
        return;
    m->indices[m->nindices++] = a;
    m->indices[m->nindices++] = b;
    m->indices[m->nindices++] = c;
}

/* spacing rules (GL 4.6 table 11.2): round a level to an integer segment count */
static int tess_round_level(float level, int spacing)
{
    if (isnan(level) || level < 1.0f)
        level = 1.0f;
    if (level > (float)MGL_TESS_MAX_LEVEL)
        level = (float)MGL_TESS_MAX_LEVEL;

    switch (spacing)
    {
        case MGL_TESS_SPACING_EQUAL:
            return (int)ceilf(level);
        case MGL_TESS_SPACING_FRACTIONAL_EVEN:
        {
            if (level < 2.0f) level = 2.0f;
            int n = (int)ceilf(level);
            return (n & 1) ? n + 1 : n;         /* round up to even */
        }
        default:                                /* fractional odd */
        {
            if (level > 63.0f) level = 63.0f;
            int n = (int)ceilf(level);
            return (n & 1) ? n : n + 1;         /* round up to odd */
        }
    }
}

/* signed area of the domain-space triangle; triangles/quads use different 2D charts:
 *  triangles (barycentric u,v,w): 2D point = (w, u)  [corner 0 top, 1 left, 2 right]
 *  quads:                          2D point = (u, v)  [u right, v up]
 */
static void tess_triangles(struct TessMesh *m, const float outer[3], float inner, int spacing,
                           float pid, bool ccw)
{
    int n[3], N;
    int rows_capacity;

    for (int i = 0; i < 3; i++)
        n[i] = tess_round_level(outer[i], spacing);
    N = tess_round_level(inner, spacing);
    for (int i = 0; i < 3; i++)
        if (n[i] > N) N = n[i];
    if (N < 1) N = 1;

    rows_capacity = N + 1;

    /* row j: points with barycentric u = j/N from edge 0 up to corner 0.
     * row 0 is the domain edge 0 (between corners 1 and 2), subdivided exactly n[0] times. */
    unsigned *row_prev = (unsigned *)malloc((rows_capacity + 1) * sizeof(unsigned));
    unsigned *row_cur = (unsigned *)malloc((rows_capacity + 1) * sizeof(unsigned));
    float p[3];

    /* row 0: exact outer edge 0 from corner 1 (0,1,0) to corner 2 (0,0,1) */
    int prev_count = 0;
    for (int k = 0; k <= n[0]; k++)
    {
        float t = (n[0] > 0) ? (float)k / (float)n[0] : 0.0f;
        p[0] = 0.0f; p[1] = 1.0f - t; p[2] = t;
        row_prev[prev_count++] = mesh_vertex(m, p[0], p[1], p[2], pid);
    }

    for (int j = 1; j <= N; j++)
    {
        float u = (float)j / (float)N;
        int count = N - j + 1;
        int cur_count = 0;

        for (int k = 0; k < count; k++)
        {
            float t = (count > 1) ? (float)k / (float)(count - 1) : 0.0f;
            /* point on the segment from (u, 1-u, 0) to (u, 0, 1-u) */
            p[0] = u;
            p[1] = (1.0f - u) * (1.0f - t);
            p[2] = (1.0f - u) * t;
            row_cur[cur_count++] = mesh_vertex(m, p[0], p[1], p[2], pid);
        }

        /* stitch row_prev (n+1 points) to row_cur (n points) */
        int a = 0, b = 0;
        while (a < prev_count - 1 || b < cur_count - 1)
        {
            float fa = (prev_count > 1) ? (float)a / (float)(prev_count - 1) : 1.0f;
            float fb = (cur_count > 1) ? (float)b / (float)(cur_count - 1) : 1.0f;

            if (a >= prev_count - 1)
            {
                if (ccw) mesh_index(m, row_prev[a], row_cur[b], row_cur[b + 1]);
                else     mesh_index(m, row_prev[a], row_cur[b + 1], row_cur[b]);
                b++;
            }
            else if (b >= cur_count - 1)
            {
                if (ccw) mesh_index(m, row_prev[a], row_prev[a + 1], row_cur[b]);
                else     mesh_index(m, row_prev[a], row_cur[b], row_prev[a + 1]);
                a++;
            }
            else if (fa <= fb)
            {
                if (ccw) mesh_index(m, row_prev[a], row_prev[a + 1], row_cur[b]);
                else     mesh_index(m, row_prev[a], row_cur[b], row_prev[a + 1]);
                a++;
            }
            else
            {
                if (ccw) mesh_index(m, row_prev[a], row_cur[b], row_cur[b + 1]);
                else     mesh_index(m, row_prev[a], row_cur[b + 1], row_cur[b]);
                b++;
            }
        }

        memcpy(row_prev, row_cur, cur_count * sizeof(unsigned));
        prev_count = cur_count;
    }

    free(row_prev);
    free(row_cur);
}

static void tess_quads(struct TessMesh *m, const float outer[4], const float inner[2], int spacing,
                       float pid, bool ccw)
{
    int nu = tess_round_level(inner[0], spacing);
    int nv = tess_round_level(inner[1], spacing);
    if (tess_round_level(outer[2], spacing) > nu) nu = tess_round_level(outer[2], spacing);
    if (tess_round_level(outer[3], spacing) > nu) nu = tess_round_level(outer[3], spacing);
    if (tess_round_level(outer[0], spacing) > nv) nv = tess_round_level(outer[0], spacing);
    if (tess_round_level(outer[1], spacing) > nv) nv = tess_round_level(outer[1], spacing);
    if (nu < 1) nu = 1;
    if (nv < 1) nv = 1;

    unsigned *grid = (unsigned *)malloc((size_t)(nu + 1) * (nv + 1) * sizeof(unsigned));
    for (int j = 0; j <= nv; j++)
        for (int i = 0; i <= nu; i++)
            grid[j * (nu + 1) + i] = mesh_vertex(m, (float)i / (float)nu, (float)j / (float)nv, 0.0f, pid);

    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++)
        {
            unsigned a = grid[j * (nu + 1) + i];
            unsigned b = grid[j * (nu + 1) + i + 1];
            unsigned c = grid[(j + 1) * (nu + 1) + i + 1];
            unsigned dd = grid[(j + 1) * (nu + 1) + i];
            if (ccw)
            {
                mesh_index(m, a, b, c);
                mesh_index(m, a, c, dd);
            }
            else
            {
                mesh_index(m, a, c, b);
                mesh_index(m, a, dd, c);
            }
        }
    free(grid);
}

static void tess_isolines(struct TessMesh *m, const float outer[2], int spacing, float pid)
{
    int n0 = (int)ceilf(outer[0] < 1.0f ? 1.0f : (outer[0] > 64.0f ? 64.0f : outer[0]));
    int n1 = tess_round_level(outer[1], spacing);

    for (int i = 0; i < n0; i++)
    {
        float v = ((float)i + 0.5f) / (float)n0;
        unsigned prev = 0;
        for (int k = 0; k <= n1; k++)
        {
            float u = (n1 > 0) ? (float)k / (float)n1 : 0.0f;
            unsigned vtx = mesh_vertex(m, u, v, 0.0f, pid);
            if (k == 0) { prev = vtx; continue; }
            if (!mesh_reserve(m, 0, 2)) return;
            m->indices[m->nindices++] = prev;
            m->indices[m->nindices++] = vtx;
            prev = vtx;
        }
    }
}

/* generate the tessellated geometry of every patch of a draw.
 * factors: per patch {outer0..3, inner0..1} (6 floats per patch), patch ids are
 * consecutive from 0. Point mode emits one point per tessellated vertex. */
static bool tess_generate(GLMContext ctx, MGLTessProgram *tp, const float *factors,
                          unsigned patches)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    struct TessMesh mesh = { NULL, 0, 0, NULL, 0, 0 };

    tess_reset_scratch(S);
    if (!patches)
        return true;

    for (unsigned p = 0; p < patches; p++)
    {
        const float *f = factors + p * 6;
        switch (tp->domain)
        {
            case MGL_TESS_DOMAIN_TRIANGLES:
                tess_triangles(&mesh, f, f[4], tp->spacing, (float)p, tp->vertex_order_ccw != 0);
                break;
            case MGL_TESS_DOMAIN_ISOLINES:
                tess_isolines(&mesh, f, tp->spacing, (float)p);
                break;
            default:
                tess_quads(&mesh, f, f + 4, tp->spacing, (float)p, tp->vertex_order_ccw != 0);
                break;
        }
    }

    if (tp->point_mode && mesh.ncoords)
    {
        /* point mode: one point per tessellated vertex */
        free(mesh.indices);
        mesh.indices = NULL;
        mesh.nindices = 0;
        mesh.index_cap = 0;
        if (!mesh_reserve(&mesh, 0, mesh.ncoords))
        {
            free(mesh.coords);
            return false;
        }
        for (unsigned i = 0; i < mesh.ncoords; i++)
            mesh.indices[mesh.nindices++] = i;
    }

    free(S->coord_buf);
    free(S->index_buf);
    S->coord_buf = mesh.coords;
    S->index_buf = mesh.indices;
    S->tess_vertex_count = (GLsizei)mesh.ncoords;
    S->tess_index_count = (GLsizei)mesh.nindices;
    S->patch_count = (GLsizei)patches;

    /* query bookkeeping (GL_PRIMITIVES_GENERATED etc. see the exact tessellated output) */
    if (mesh.nindices)
    {
        GLuint prims;
        if (tp->point_mode || tp->domain == MGL_TESS_DOMAIN_ISOLINES)
            prims = (GLuint)(tp->point_mode ? mesh.ncoords : mesh.nindices / 2);
        else
            prims = (GLuint)(mesh.nindices / 3);
        mgl_tess_note_draw(ctx, patches, (GLuint)mesh.ncoords, prims);
    }
    return true;
}

/* build the per-patch data blob consumed by the rewritten TES (TES patchIn layout):
 * - members written by the TCS are copied from the read-back patch output struct
 * - the tessellation level members are filled from the factors
 * - anything else stays zero (undefined per GL when no TCS writes it)
 *
 * CPU form: used by the MGL_TESS_CPU reference path and the GPU-tessellator
 * fallback only — the GPU paths assemble the blob with the mgl_tess_blob kernel
 * (see MGLTessKernels.h) from the same map (mgl_tess_build_blob_map). */
static unsigned tess_cpu_blob_builds;   /* diagnostic: CPU blob builds since load */
static unsigned tess_cpu_generate_runs; /* diagnostic: CPU tessellator runs since load */
static unsigned tess_scan_waits;        /* diagnostic: GPU index-scan waits since load */
static unsigned tess_cpu_index_reads;   /* diagnostic: CPU element-buffer reads since load */
static unsigned tess_restart_checks;    /* diagnostic: CPU restart-presence scans since load */

unsigned mgl_tess_cpu_blob_builds(void)   { return tess_cpu_blob_builds; }
unsigned mgl_tess_cpu_generate_runs(void) { return tess_cpu_generate_runs; }
unsigned mgl_tess_scan_waits(void)        { return tess_scan_waits; }
unsigned mgl_tess_cpu_index_reads(void)   { return tess_cpu_index_reads; }
unsigned mgl_tess_restart_checks(void)    { return tess_restart_checks; }
void mgl_tess_note_scan_wait(void)      { tess_scan_waits++; }
void mgl_tess_note_cpu_index_read(void) { tess_cpu_index_reads++; }
void mgl_tess_note_restart_check(void)  { tess_restart_checks++; }

static void tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                                  const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                                  const float *factors, unsigned patches)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    size_t need = (size_t)tp->tes_patch_in_size * patches;

    tess_cpu_blob_builds++;
    if (!need)
        return;
    if (S->patch_cap < need)
    {
        unsigned char *p = (unsigned char *)realloc(S->patch_data, need);
        if (!p)
            return;
        S->patch_data = p;
        S->patch_cap = need;
    }
    memset(S->patch_data, 0, need);

    for (unsigned pi = 0; pi < patches; pi++)
    {
        unsigned char *dst = S->patch_data + (size_t)pi * tp->tes_patch_in_size;
        const unsigned char *src = tcs_patch_out ? tcs_patch_out + (size_t)pi * tcs_patch_stride : NULL;
        const float *f = factors ? factors + pi * 6 : NULL;

        for (int i = 0; i < tp->num_tes_patch_in; i++)
        {
            const MGLTessPatchMember *m = &tp->tes_patch_in[i];

            if (!strcmp(m->name, "gl_TessLevel") || !strcmp(m->name, "gl_TessLevelOuter"))
            {
                if (!f) continue;
                float lv[4] = { f[0], f[1], f[2], f[3] };
                if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
                    lv[3] = f[4];                    /* triangles pack inner[0] into .w */
                memcpy(dst + m->offset, lv, sizeof lv);
            }
            else if (!strcmp(m->name, "gl_TessLevelInner"))
            {
                if (!f) continue;
                float lv[2] = { f[4], f[5] };
                memcpy(dst + m->offset, lv, sizeof lv);
            }
            else if (src)
            {
                /* copy from the TCS patch output by varying location (canonical key) */
                for (int j = 0; j < tp->num_tcs_patch_out; j++)
                {
                    if (tp->tcs_patch_out[j].key == m->key &&
                        tp->tcs_patch_out[j].key != 0)
                    {
                        memcpy(dst + m->offset, src + tp->tcs_patch_out[j].offset, m->size);
                        break;
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ per-program state */

static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out,
                                   char *struct_name, size_t struct_len);
MGLTessProgram *mgl_tess_program(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return NULL;
    for (tp = S->programs[(((uintptr_t)program) >> 4) % MGL_TESS_PROGRAM_HASH]; tp; tp = tp->next)
        if (tp->program == program)
            return tp;
    return NULL;
}

static void tess_free_program(GLMContext ctx, MGLTessProgram *tp)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram **head = &S->programs[(((uintptr_t)tp->program) >> 4) % MGL_TESS_PROGRAM_HASH];

    while (*head && *head != tp)
        head = &(*head)->next;
    if (*head)
        *head = tp->next;
    /* release the bridged Metal objects of the compile-once variants */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
    {
        void **owned[8] = { &tp->capture_function, &tp->capture_library, &tp->capture_pso,
                            &tp->plain_function, &tp->tcs_function, &tp->tcs_library,
                            &tp->vs_function, &tp->vs_library };
        void *extra[2] = { tp->plain_library, NULL };

        for (int i = 0; i < 8; i++)
            if (*owned[i])
            {
                ctx->mtl_funcs.ext_program->release_obj(ctx, *owned[i]);
                *owned[i] = NULL;
            }
        if (extra[0])
            ctx->mtl_funcs.ext_program->release_obj(ctx, extra[0]);
    }
    free(tp->plain_msl);
    free(tp);
}

void mgl_ext_delete_program_tess(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return;
    while ((tp = mgl_tess_program(ctx, program)) != NULL)
        tess_free_program(ctx, tp);
}

/* post-link: parse the tessellation modes and the patch data layouts */
static void mgl_ext_post_link_tess(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;
    int domain, spacing, ccw, point_mode, out_vertices = 0;
    bool has_tcs, has_tes;

    if (!S || !program)
        return;

    has_tcs = program->spirv[_TESS_CONTROL_SHADER].ir != NULL;
    has_tes = program->spirv[_TESS_EVALUATION_SHADER].ir != NULL;
    if (!has_tes)
        return;                             /* not (a usable) tessellation program */

    domain = spv_tess_domain(program->spirv[_TESS_EVALUATION_SHADER].ir,
                             program->spirv[_TESS_EVALUATION_SHADER].size,
                             &spacing, &ccw, &point_mode);
    if (domain < 0)
        return;

    if (has_tcs)
    {
        int modes[16], count = 0;
        if (!spv_tess_modes(program->spirv[_TESS_CONTROL_SHADER].ir,
                            program->spirv[_TESS_CONTROL_SHADER].size,
                            1 /* SpvExecutionModelTessellationControl */, modes, &count, &out_vertices) ||
            out_vertices < 1 || out_vertices > MGL_TESS_MAX_PATCH_VERTICES)
            out_vertices = 3;
    }

    /* replace any stale entry (relink) */
    mgl_ext_delete_program_tess(ctx, program);

    tp = (MGLTessProgram *)calloc(1, sizeof(MGLTessProgram));
    if (!tp)
        return;
    tp->program = program;
    tp->program_name = program->name;
    tp->has_tcs = has_tcs;
    tp->domain = domain;
    tp->spacing = spacing;
    tp->vertex_order_ccw = ccw;
    tp->point_mode = point_mode;
    tp->tcs_out_vertices = has_tcs ? out_vertices : 0;
    tp->tessellation_active = 1;

    {
        MGLTessProgram **head = &S->programs[(((uintptr_t)program) >> 4) % MGL_TESS_PROGRAM_HASH];
        tp->next = *head;
        *head = tp;

    }

    /* patch data layouts, parsed from the generated MSL (raw per-stage structs; the
     * canonical records below give the offsets actually used at draw time) */
    if (has_tcs)
    {
        struct MslStruct po;
        char po_name[160];
        if (msl_parse_struct(program->spirv[_TESS_CONTROL_SHADER].msl_str, "main0_patchOut", &po) ||
            msl_parse_struct_entry(program->spirv[_TESS_CONTROL_SHADER].msl_str, &po, po_name, sizeof po_name))
        {
            tp->num_tcs_patch_out = po.count;
            tp->tcs_patch_out_size = po.size;
            for (int i = 0; i < po.count && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
                tp->tcs_patch_out[i] = po.members[i];
        }
    }

    /* TES interface layouts (from the stored hardware-form MSL): control point members
     * and patch-in members with their [[attribute(N)]] indices build the render
     * pipeline's vertex descriptor at draw time */
    {
        struct MslStruct pi;
        const char *msl = program->spirv[_TESS_EVALUATION_SHADER].msl_str;
        char name[160];

        if (msl_patch_in_name(msl, name, sizeof name) && msl_parse_struct(msl, name, &pi))
        {
            int m = 0;
            for (int i = 0; i < pi.count && m < MGL_TESS_MAX_PATCH_MEMBERS; i++)
            {
                if (i == pi.cp_ref_member)
                    continue;
                tp->tes_patch_in[m++] = pi.members[i];
            }
            tp->num_tes_patch_in = m;
            tp->tes_patch_in_size = pi.size;

            struct MslStruct cp;
            char cpname[128];
            if (msl_cp_struct_name(msl, name, cpname, sizeof cpname) &&
                msl_parse_struct(msl, cpname, &cp))
            {
                tp->cp_record_size = cp.size;
                tp->num_tes_cp = cp.count < MGL_TESS_MAX_PATCH_MEMBERS ? cp.count : MGL_TESS_MAX_PATCH_MEMBERS;
                for (int i = 0; i < tp->num_tes_cp; i++)
                    tp->tes_cp[i] = cp.members[i];
            }
        }

        /* the baked control point count of [[patch(domain, N)]] */
        tp->hw_cp_count = tp->has_tcs ? tp->tcs_out_vertices : 3;

        /* one canonical record layout per interface (control points / per-patch),
         * shared by producer and consumer stages; recompiles the VS / TCS kernels
         * when their SPIRV-Cross struct differs (see "canonical records") */
        tess_canonicalize_records(ctx, tp);
    }

    /* the GPU blob kernel's member map (levels + TCS patch-output copies) */
    mgl_tess_build_blob_map(ctx, tp);
}

/* ------------------------------------------------------------------ GL hooks */

void mgl_ext_init_tess(GLMContext ctx)
{
    struct MGLExtState_tess *S;

    if (ctx->state.ext_tess)
        return;
    S = (struct MGLExtState_tess *)calloc(1, sizeof(struct MGLExtState_tess));
    if (!S)
        return;
    ctx->state.ext_tess = S;

    mgl_ext_register_post_link(mgl_ext_post_link_tess);
}

bool mgl_ext_get_tess(GLMContext ctx, GLenum pname, GLenum type, void *data)
{
    /* limits are served by get.c; GL_PATCH_VERTICES / default levels by rstate */
    return false;
}

/* note exact tessellation statistics to the query domain */
/* query bookkeeping for GPU-path draws (no CPU tessellation): the tessellated output
 * counts, computed from the levels with exactly the arithmetic of the GPU tessellator's
 * mgl_patch_counts (MGLTessKernels.h) and the CPU tessellator. */
void mgl_tess_note_counts(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches)
{
    GLuint ncoords = 0, nindex = 0;

    for (unsigned p = 0; p < patches; p++)
    {
        const float *f = factors + p * 6;
        GLuint pc = 0, pi = 0;

        if (tp->domain == MGL_TESS_DOMAIN_ISOLINES)
        {
            int n0 = (int)ceilf(f[0] < 1.0f ? 1.0f : (f[0] > 64.0f ? 64.0f : f[0]));
            int n1 = tess_round_level(f[1], tp->spacing);

            pc = (GLuint)n0 * (GLuint)(n1 + 1);
            pi = 2u * (GLuint)n0 * (GLuint)n1;
        }
        else if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
        {
            int n[3], N, j;
            unsigned prev, cur;

            n[0] = tess_round_level(f[0], tp->spacing);
            n[1] = tess_round_level(f[1], tp->spacing);
            n[2] = tess_round_level(f[2], tp->spacing);
            N = tess_round_level(f[4], tp->spacing);
            for (int i = 0; i < 3; i++)
                if (n[i] > N)
                    N = n[i];
            if (N < 1)
                N = 1;

            prev = (unsigned)n[0] + 1u;
            pc = prev;
            for (j = 1; j <= N; j++)
            {
                cur = (unsigned)(N - j + 1);
                pc += cur;
                pi += (prev - 1u) + (cur - 1u);
                prev = cur;
            }
            pi *= 3u;
        }
        else
        {
            int nu = tess_round_level(f[4], tp->spacing), nv = tess_round_level(f[5], tp->spacing);
            int o;

            o = tess_round_level(f[2], tp->spacing); if (o > nu) nu = o;
            o = tess_round_level(f[3], tp->spacing); if (o > nu) nu = o;
            o = tess_round_level(f[0], tp->spacing); if (o > nv) nv = o;
            o = tess_round_level(f[1], tp->spacing); if (o > nv) nv = o;

            pc = (GLuint)(nu + 1) * (GLuint)(nv + 1);
            pi = 6u * (GLuint)nu * (GLuint)nv;
        }

        if (tp->point_mode)
            pi = pc;
        ncoords += pc;
        nindex += pi;
    }

    if (ncoords || nindex)
    {
        GLuint prims;

        if (tp->point_mode)
            prims = ncoords;
        else if (tp->domain == MGL_TESS_DOMAIN_ISOLINES)
            prims = nindex / 2u;
        else
            prims = nindex / 3u;
        mgl_tess_note_draw(ctx, patches, ncoords, prims);
    }
}

/* words a packed MSL type occupies (geometry.c's rule: elems*cols*rows*base) */
static bool tess_msl_type_words(const char *type, uint32_t *words)
{
    uint32_t base = 1, cols = 1, rows = 1;
    const char *d;

    if (!strncmp(type, "spvUnsafeArray<", 15) && (d = strchr(type + 15, ',')) != NULL)
    {
        /* a flat slot array: N elements of the element type */
        char elem[32];
        uint32_t n = 0, w = 0;
        int k = (int)(d - (type + 15));

        if (k < 0 || k >= (int)sizeof elem) return false;
        snprintf(elem, sizeof elem, "%.*s", k, type + 15);
        if (sscanf(d + 1, "%u", &n) != 1) return false;
        if (!tess_msl_type_words(elem, &w)) return false;
        *words = w * n;
        return true;
    }
    if (!strncmp(type, "float", 5) || !strncmp(type, "int", 3) || !strncmp(type, "uint", 4) ||
        !strncmp(type, "bool", 4) || !strncmp(type, "short", 5) || !strncmp(type, "half", 4))
        base = 1;
    else if (!strncmp(type, "double", 6))
        base = 2;
    else
        return false;               /* arrays / structs are not captured */

    d = type;
    while (*d && !(*d >= '0' && *d <= '9'))
        d++;
    if (*d)
    {
        cols = (uint32_t)strtoul(d, NULL, 10);
        while (*d && *d != 'x')
            d++;
        if (*d == 'x')
            rows = (uint32_t)strtoul(d + 1, NULL, 10);
    }
    *words = base * cols * rows;
    return true;
}

/* the capture-kernel MSL: the plain TES vertex function rewritten into a compute
 * kernel storing the packed output record (see the MGLTessCapture contract).
 * Fills tp->cap_members[] / tp->cap_stride; NULL when the output interface has
 * something unsupported. Caller frees. */
/* number of doubles behind a TES output member, from the GL-view module: the
 * capture record's MSL repr pads each dvecR column to whole uint4 slots, but
 * every consumer of the capture (the geometry domain's import and the xfb
 * publish) reads the doubles compactly, two words each.  0 = not a double
 * varying (u32/float members keep their natural size). */
static uint32_t tess_out_doubles(GLMContext ctx, MGLTessProgram *tp, const char *name)
{
    Program *prog = tp->program;
    uint32_t *ir;
    size_t n, m;

    if (!prog || !name || !*name)
        return 0;
    ir = prog->spirv[_TESS_EVALUATION_SHADER].ir;
    n = prog->spirv[_TESS_EVALUATION_SHADER].size;
    if (!ir || n < 8 || !ctx)
        return 0;

    /* find the member/variable by name: OpName (variables) or OpMemberName
     * (interface block members) */
    m = 5;
    while (m + 3 < n)
    {
        uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
        uint32_t tid = 0;

        if (wc == 0) break;
        if (op == 5 /* OpName */)
            tid = ir[m + 1];
        else if (op == 6 /* OpMemberName */)
            tid = 0;                 /* handled below: need the member index */
        if (op == 5 || op == 6)
        {
            char buf[128];
            size_t o = 0;
            const uint32_t *w = ir + m + (op == 5 ? 2 : 3);
            const uint32_t *we = ir + m + wc;

            for (; w < we && o + 4 < sizeof buf; w++)
            {
                for (int b = 0; b < 4; b++)
                {
                    char c = (char)((*w >> (8 * b)) & 255);
                    if (!c) { w = we; break; }
                    buf[o++] = c;
                }
            }
            buf[o] = 0;
            if (!strcmp(buf, name))
            {
                /* resolve the type: the variable itself, or the struct member */
                uint32_t ty = 0;

                if (op == 5)
                {
                    /* OpName names the variable; find its OpVariable */
                    size_t v = 5;
                    while (v + 4 < n)
                    {
                        uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                        if (wv == 0) break;
                        if (ov == 59 && ir[v + 2] == tid && ir[v + 3] == SpvStorageClassOutput)
                        {
                            ty = ir[v + 1];   /* the pointer type */
                            break;
                        }
                        v += wv;
                    }
                }
                else
                {
                    /* OpMemberName: (struct type, member index) -> member type */
                    size_t v = 5;
                    uint32_t midx = ir[m + 2];

                    while (v + 3 < n)
                    {
                        uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                        if (wv == 0) break;
                        if (ov == 30 /* OpTypeStruct */ && ir[v + 1] == tid)
                        {
                            if (midx + 2 < wv)
                                ty = ir[v + 2 + midx];
                            break;
                        }
                        v += wv;
                    }
                }
                /* walk the (pointer-stripped) type: array * matrix columns *
                 * vector rows doubles */
                {
                    uint32_t elems = 1, cols = 1, rows = 1, isdbl = 0, pass = 0;

                    while (ty && pass++ < 8)
                    {
                        size_t v = 5;

                        while (v + 3 < n)
                        {
                            uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                            if (wv == 0) break;
                            if (ov >= 19 && ov <= 33 && ir[v + 1] == ty)
                            {
                                if (ov == 32) { ty = ir[v + 3]; break; }             /* pointer */
                                if (ov == 28)                                        /* array */
                                {
                                    size_t c = 5;
                                    uint32_t alen = 1;
                                    while (c + 3 < n)
                                    {
                                        uint32_t wcc = ir[c] >> 16, oc = ir[c] & 0xffff;
                                        if (wcc == 0) break;
                                        if (oc == 43 && ir[c + 2] == ir[v + 3])
                                        {
                                            alen = ir[c + 3];
                                            break;
                                        }
                                        c += wcc;
                                    }
                                    elems *= alen;
                                    ty = ir[v + 2];
                                    break;
                                }
                                if (ov == 24) { cols = ir[v + 3]; ty = ir[v + 2]; break; }   /* matrix */
                                if (ov == 23) { rows = ir[v + 3]; ty = ir[v + 2]; break; }   /* vector */
                                if (ov == 22) { isdbl = (ir[v + 2] == 64); ty = 0; break; }  /* float */
                                ty = 0;
                                break;
                            }
                            v += wv;
                        }
                    }
                    if (isdbl)
                        return elems * cols * rows;
                    return 0;
                }
            }
        }
        m += wc;
    }
    return 0;
}

char *mgl_tess_capture_msl(GLMContext ctx, MGLTessProgram *tp)
{
    char *plain = mgl_tess_plain_msl(ctx, tp);
    char *out = NULL, *k = NULL;
    const char *sig, *name, *open, *vid, *body, *p;
    char types[MGL_TESS_CAPTURE_MAX][48];
    uint32_t dbls[MGL_TESS_CAPTURE_MAX];    /* per member: double count (0 = plain u32/float member) */
    uint32_t slotsz[MGL_TESS_CAPTURE_MAX];  /* ... words per fp64 slot: 2 (uint2) or 4 (uint4) */
    uint32_t slotn[MGL_TESS_CAPTURE_MAX];   /* ... slots in the MSL repr (1 for a bare uint2/uint4) */
    char cap_retname[96];
    struct MslStruct rec;
    const char *cursor;
    uint32_t off = 4;
    size_t cap, len = 0;

    if (!plain)
        return NULL;

    sig = strstr(plain, "vertex ");
    name = open = vid = body = NULL;
    if (sig)
    {
        /* the return type is the first token after "vertex " */
        const char *rt = sig + 7 + strspn(sig + 7, " \t\n");
        const char *rte = rt + strcspn(rt, " \t\n");

        name = rte + strspn(rte, " \t\n");
        open = name ? strchr(name, '(') : NULL;
        vid = plain ? strstr(plain, "uint mgl_vid [[vertex_id]]") : NULL;
        body = vid ? strchr(vid, '{') : NULL;
        if (rt >= rte || (size_t)(rte - rt) >= sizeof cap_retname)
        {
            rt = NULL;
            name = NULL;
        }
        else
            snprintf(cap_retname, sizeof cap_retname, "%.*s", (int)(rte - rt), rt);
    }
    if (!sig || !name || !open || !vid || !body || open > vid)
        { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 1 sig=%p name=%p open=%p vid=%p body=%p\n", (const void*)sig, (const void*)name, (const void*)open, (const void*)vid, (const void*)body); goto done; }

    /* "vertex <Ret> name(...)": the record type is between 'vertex ' and the name */
    {
        if (!msl_parse_struct(plain, cap_retname, &rec))
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 3 (%s)\n", cap_retname); goto done; }
    }

    tp->cap_members_n = 0;
    for (int i = 0; i < rec.count && tp->cap_members_n < MGL_TESS_CAPTURE_MAX; i++)
    {
        const char *m = rec.members[i].name;

        /* the rstate fixup injects a synthetic point-size member into the last
         * pre-raster stage's output record; it is not a shader output, skip it */
        if (!strcmp(m, "mgl_rs_PointSize"))
            continue;
        uint32_t w;

        dbls[tp->cap_members_n] = 0;
        slotsz[tp->cap_members_n] = 0;
        slotn[tp->cap_members_n] = 1;
        if (!strcmp(m, "gl_Position"))
        {
            w = 4;
            snprintf(types[tp->cap_members_n], 48, "float4");
            tp->cap_members[tp->cap_members_n].offset = 0;
        }
        else if (!strncmp(m, "gl_", 3))
            continue;               /* other builtins are not captured */
        else if (!tess_msl_type_words(rec.members[i].type, &w))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: capture: unsupported varying %s of type %s\n",
                        m, rec.members[i].type);
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 4\n"); goto done; }
        }
        else
        {
            const char *t = rec.members[i].type;
            uint32_t d = tess_out_doubles(ctx, tp, m);

            snprintf(types[tp->cap_members_n], 48, "%s", t);
            tp->cap_members[tp->cap_members_n].offset = off;
            if (d)
            {
                /* fp64 repr: the doubles sit two per uint4 slot (one per uint2
                 * slot), possibly with a padded tail — the capture record
                 * carries them COMPACTLY, matching the geometry import and the
                 * xfb publish */
                uint32_t wsz = strstr(t, "uint2") ? 2 : 4;

                dbls[tp->cap_members_n] = d;
                slotsz[tp->cap_members_n] = wsz;
                {
                    const char *nm = strstr(t, "spvUnsafeArray<");
                    uint32_t cnt = 1;

                    if (nm && sscanf(strchr(nm, ',') + 1, "%u", &cnt) != 1)
                        cnt = 1;
                    slotn[tp->cap_members_n] = cnt;
                }
                w = 2 * d;
            }
        }

        snprintf(tp->cap_members[tp->cap_members_n].name, 64, "%s", m);
        tp->cap_members[tp->cap_members_n].words = w;
        if (strcmp(m, "gl_Position"))
            off += w;
        tp->cap_members_n++;
    }
    {
        bool have_pos = false;

        for (int i = 0; i < tp->cap_members_n; i++)
            if (!strcmp(tp->cap_members[i].name, "gl_Position"))
                have_pos = true;
        if (!have_pos)
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 5\n"); goto done; }               /* gl_Position must be present */
    }
    /* Pad the record to a multiple of 4 words (16 bytes): every record starts with the
     * gl_Position float4 store, and a device float4 store at a non-16-byte-aligned
     * address is undefined on Metal - the tessellated vertices past the first would
     * come back as garbage.  The consumers (the geometry domain's import kernel and the
     * diagnostic capture query) read tp->cap_stride, so they follow the padding. */
    tp->cap_stride = (off + 3u) & ~3u;

    if (tess_trace())
        fprintf(stderr, "MGL tess: capture: %s (stride %u, %d members)\n",
                "layout built", tp->cap_stride, tp->cap_members_n);
    cap = strlen(plain) + 4096;
    k = (char *)malloc(cap);
    if (!k)
        { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 6\n"); goto done; }

    str_append(&k, &len, &cap, "%.*skernel void ", (int)(sig - plain), plain);
    str_append(&k, &len, &cap, "%.*s_cap(", (int)(open - name), name);
    str_append(&k, &len, &cap, "%.*s", (int)(vid - (open + 1)), open + 1);
    /* the plain signature's tail after the mgl_vid parameter: the original
     * entry's buffers (uniform blocks etc.) — they must stay in the kernel's
     * signature, the body reads them */
    {
        const char *vp = vid + strlen("uint mgl_vid [[vertex_id]]");
        const char *close = NULL;
        int depth = 0;

        for (const char *q = open; *q; q++)
        {
            if (q[0] == '[' && q[1] == '[')
            {
                const char *e = strstr(q, "]]");
                if (e) { q = e; continue; }
            }
            if (*q == '(') depth++;
            else if (*q == ')' && --depth == 0) { close = q; break; }
        }
        while (*vp == ' ') vp++;
        if (*vp == ',' && close)
        {
            const char *t = vp + 1;

            while (*t == ' ' || *t == '\n') t++;
            str_append(&k, &len, &cap, "%.*s, ", (int)(close - t), t);
        }
    }
    str_append(&k, &len, &cap, "uint mgl_vid [[thread_position_in_grid]], "
                               "device const uint* mgl_cap_count [[buffer(%d)]], "
                               "device uint* mgl_cap [[buffer(%d)]])\n",
               MGL_TESS_TES_CAP_COUNT, MGL_TESS_TES_CAP_RECS);
    /* the original signature's tail ("uint mgl_vid [[vertex_id]])") is dropped: the
     * record store replaces the return value, and thread_position_in_grid is mgl_vid */
    str_append(&k, &len, &cap, "{");
    str_append(&k, &len, &cap, "\n    if (mgl_vid >= mgl_cap_count[0]) return;\n");

    /* replace every "return <var>;" with the packed record store: `cursor` walks the
     * original body starting just after its '{' (the '{' was emitted above) */
    cursor = body + 1;
    while ((p = strstr(cursor, "return ")) != NULL)
    {
        const char *semi = strchr(p, ';');
        char varname[32], store[2048];
        size_t vn = semi ? (size_t)(semi - (p + 7)) : 0;
        int slen = 0;

        if (!semi)
            break;
        if (vn >= sizeof varname)
            vn = sizeof varname - 1;
        snprintf(varname, sizeof varname, "%.*s", (int)vn, p + 7);

        str_append(&k, &len, &cap, "%.*s", (int)(p - cursor), cursor);
        if (vn == 0)
        {
            /* a bare "return;" inside the body is legal in the kernel too */
            str_append(&k, &len, &cap, "return;");
            cursor = semi + 1;
            continue;
        }
        for (int i = 0; i < tp->cap_members_n; i++)
        {
            const MGLTessCaptureMember *cm = &tp->cap_members[i];
            const char *t = types[i];
            const char *arr = strstr(t, "spvUnsafeArray<");

            str_append(&k, &len, &cap, "%s", i == 0 ? "{ " : "");
            if (dbls[i])
            {
                /* the doubles sit two per uint4 slot (one per uint2 slot) with
                 * a possibly padded tail: copy them out compactly, two words
                 * per double */
                uint32_t sw = slotsz[i];     /* words per slot: 2 or 4 */
                uint32_t spdbl = sw / 2;     /* doubles per slot: 1 or 2 */

                str_append(&k, &len, &cap,
                           "device uint* mgl_w = (device uint*)((device uchar*)mgl_cap + "
                           "(size_t)mgl_vid * %uu + %uu); (void)mgl_w;\n",
                           tp->cap_stride * 4u, cm->offset * 4u);
                for (uint32_t d = 0; d < dbls[i]; d++)
                {
                    uint32_t slot = d / spdbl;
                    const char *lo = (spdbl == 2 && (d & 1)) ? ".z" : ".x";
                    const char *hi = (spdbl == 2 && (d & 1)) ? ".w" : ".y";
                    char ix[16];

                    ix[0] = 0;
                    if (arr) snprintf(ix, sizeof ix, "[%u]", slot);
                    str_append(&k, &len, &cap, "mgl_w[%u] = %s.%s%s%s; mgl_w[%u] = %s.%s%s%s;\n",
                               2 * d, varname, cm->name, ix, lo,
                               2 * d + 1, varname, cm->name, ix, hi);
                }
            }
            else if (arr)
            {
                /* plain (non-double) slot array: element-wise copy — a struct
                 * copy across address spaces would not compile */
                str_append(&k, &len, &cap,
                           "device uint* mgl_w = (device uint*)((device uchar*)mgl_cap + "
                           "(size_t)mgl_vid * %uu + %uu);\n",
                           tp->cap_stride * 4u, cm->offset * 4u);
                for (uint32_t q = 0; q < slotn[i] && q < 32; q++)
                    str_append(&k, &len, &cap, "mgl_w[%u] = %s.%s[%u];\n", q, varname, cm->name, q);
            }
            else
                str_append(&k, &len, &cap,
                           "*(device %s*)((device uchar*)mgl_cap + (size_t)mgl_vid * %uu + %uu) = %s.%s;\n",
                           t, tp->cap_stride * 4u, cm->offset * 4u, varname, cm->name);
        }
        str_append(&k, &len, &cap, "}\nreturn;\n");
        cursor = semi + 1;
    }
    str_append(&k, &len, &cap, "%s", cursor);

    /* the record struct was written for a vertex function: stage attributes such as
     * [[position]] / [[user(locn0)]] are not legal on a kernel local, strip them */
    {
        char head[160];
        const char *at;
        char *decl;

        snprintf(head, sizeof head, "struct %s\n{", cap_retname);
        at = strstr(k, head);
        if (!at) { snprintf(head, sizeof head, "struct %s {", cap_retname); at = strstr(k, head); }
        if (at)
        {
            char *ob = strchr(at, '{');
            char *cb = ob ? strchr(ob, '}') : NULL;

            if (ob && cb)
                for (char *a = ob; (a = strstr(a, "[[")) != NULL && a < cb; )
                {
                    char *e = strstr(a, "]]");
                    if (!e || e > cb)
                        break;
                    memmove(a, e + 2, strlen(e + 2) + 1);
                    cb -= (size_t)(e + 2 - a);
                    len -= (size_t)(e + 2 - a);
                }
            (void)decl;
        }
    }
    out = k;

done:
    if (!out)
    {
        tp->cap_members_n = 0;
        tp->cap_stride = 0;
    }
    free(plain);
    return out;
}

/* True when the current tessellated draw rasterizes through the plain TES
 * (MGLRenderer+Tessellation.m's emulated path: GPU tessellator + indirect
 * draw, or the CPU reference) instead of the hardware post-tessellation
 * pipeline.  Transform-feedback capture forces this path as well — the
 * captured records must be GPU-visible — and so do isolines / point_mode.
 * With a geometry shader the geometry pass-through renders instead, so the
 * plain TES is never the raster vertex function there. */
bool mgl_tess_plain_render(GLMContext ctx)
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);

    if (!tp || !tp->plain_function)
        return false;
    if (ctx->state.program && ctx->state.program->shader_slots[_GEOMETRY_SHADER])
        return false;
    return tp->domain == MGL_TESS_DOMAIN_ISOLINES || tp->point_mode || tp->capture ||
           mgl_tess_cpu_reference();
}

void mgl_tess_set_capture(GLMContext ctx, Program *program, bool enable){
    MGLTessProgram *tp = mgl_tess_active_program(ctx);

    if (!tp)
        return;
    (void)program;
    tp->capture = enable;
}

const MGLTessCapture *mgl_tess_get_capture(GLMContext ctx)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    return (S && S->capture.valid) ? &S->capture : NULL;
}

void mgl_tess_note_draw(GLMContext ctx, GLuint patches, GLuint tes_invocations, GLuint tessellated_primitives)
{
    mgl_queries_note_tess(ctx, tes_invocations, tessellated_primitives);
}

/* the TCS kernel's two record struct names: the types of the spvIn / spvOut
 * parameters of its entry point */
static bool msl_tcs_record_names(const char *msl, char *in_name, size_t in_len,
                                 char *out_name, size_t out_len)
{
    bool got_in = false, got_out = false;
    const char *at = msl;

    while ((at = strstr(at, "device ")) != NULL && !(got_in && got_out))
    {
        const char *star = strchr(at, '*');
        const char *b = at + strlen("device ");
        const char *nl = strchr(at, '\n');

        /* "device const T* spvIn" / "device T* spvOut" (kernel parameter list) */
        if (star && nl && star < nl)
        {
            const char *param = star + 1;
            bool is_in, is_out;
            const char *e = star;

            if (!strncmp(b, "const ", 6))
                b += 6;                 /* device const T* spvIn */
            while (param < at + strlen(at) && *param == ' ')
                param++;
            is_in = !strncmp(param, "spvIn", 5);
            is_out = !strncmp(param, "spvOut", 6);
            while (e > b && *(e - 1) == ' ')
                e--;
            if ((is_in || is_out) && e > b)
            {
                char name[128];
                snprintf(name, sizeof name, "%.*s", (int)(e - b), b);
                if (is_in && !got_in)
                {
                    snprintf(in_name, in_len, "%s", name);
                    got_in = true;
                }
                else if (is_out && !got_out)
                {
                    snprintf(out_name, out_len, "%s", name);
                    got_out = true;
                }
            }
        }
        at += strlen("device ");
    }
    return got_in && got_out;
}

/* ------------------------------------------------------------------ record canonicalization
 * Parses every stage's interface struct, builds the canonical control-point and
 * per-patch records (union by varying location — see "canonical records" above),
 * rewrites the producer MSL structs to them and recompiles those kernels. Consumers
 * read the records through the canonical layout: the TCS kernel through its rewritten
 * spvIn/spvOut structs, the plain TES through its rewritten control-point struct and
 * blob offsets, the hardware TES through the vertex descriptor offsets (tp->tes_cp /
 * tp->tes_patch_in are already canonical after this call). */
static void tess_canonicalize_records(GLMContext ctx, MGLTessProgram *tp)
{
    Program *program = tp->program;
    const char *vs_msl = program->spirv[_VERTEX_SHADER].msl_str;
    const char *tcs_msl = tp->has_tcs ? program->spirv[_TESS_CONTROL_SHADER].msl_str : NULL;
    const char *tes_msl = program->spirv[_TESS_EVALUATION_SHADER].msl_str;

    struct MslStruct vs_out, tcs_in, tcs_out, tes_cp, tcs_po, tes_pi;
    char vs_name[160] = "", tcs_in_name[160] = "", tcs_out_name[160] = "";
    char po_name[160] = "", pi_name[160] = "", cp_name[160] = "";
    bool have_vs = false, have_tcs_in = false, have_tcs_out = false;
    bool have_tcs_po = false, have_pi = false, have_cp = false;
    struct CanonRec canon, canon_tcs, canon_patch;

    memset(&vs_out, 0, sizeof vs_out);
    memset(&tcs_in, 0, sizeof tcs_in);
    memset(&tcs_out, 0, sizeof tcs_out);
    memset(&tes_cp, 0, sizeof tes_cp);
    memset(&tcs_po, 0, sizeof tcs_po);
    memset(&tes_pi, 0, sizeof tes_pi);

    if (vs_msl && tess_cp_record_name(vs_msl, vs_name, sizeof vs_name))
        have_vs = msl_parse_struct(vs_msl, vs_name, &vs_out);
    if (tcs_msl)
    {
        if (msl_tcs_record_names(tcs_msl, tcs_in_name, sizeof tcs_in_name,
                                 tcs_out_name, sizeof tcs_out_name))
        {
            have_tcs_in = msl_parse_struct(tcs_msl, tcs_in_name, &tcs_in);
            have_tcs_out = msl_parse_struct(tcs_msl, tcs_out_name, &tcs_out);
        }
        have_tcs_po = msl_parse_struct_entry(tcs_msl, &tcs_po, po_name, sizeof po_name);
    }
    if (tes_msl)
    {
        if (msl_patch_in_name(tes_msl, pi_name, sizeof pi_name))
            have_pi = msl_parse_struct(tes_msl, pi_name, &tes_pi);
        if (have_pi && msl_cp_struct_name(tes_msl, pi_name, cp_name, sizeof cp_name))
            have_cp = msl_parse_struct(tes_msl, cp_name, &tes_cp);
    }

    /* attach varying locations to every member */
    tess_annotate_keys(ctx, tp, _VERTEX_SHADER, &vs_out, have_vs ?
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tp, _VERTEX_SHADER, &tcs_in, have_tcs_in ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, tp, _TESS_CONTROL_SHADER, &tcs_out, have_tcs_out ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tp, _TESS_CONTROL_SHADER, &tes_cp, have_cp ?
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, tp, _TESS_CONTROL_SHADER, &tcs_po, have_tcs_po ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tp, _TESS_EVALUATION_SHADER, &tes_pi, have_pi ?
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);

    /* pair flattened interface-block members that no reflected name matched.
     * KNOWN-INCOMPLETE (round 5): with the pairing active the canonical-record
     * build of the plain TES still emits invalid MSL (heap garbage in the
     * epilogue — "MSL source is not valid UTF-8"); the values do not cross
     * stages yet, so it is off by default.  Enable with MGL_TESS_PAIR_BLOCKS=1
     * to resume debugging: the pairing itself works (see the trace), the
     * overflow is in the canon path it enables (rewrite_tes epilogue). */
    if (getenv("MGL_TESS_PAIR_BLOCKS"))
    {
        if (have_tcs_in)  tess_pair_iface_members(&vs_out, &tcs_in);
        else if (have_cp) tess_pair_iface_members(&vs_out, &tes_cp);
        if (have_cp)      tess_pair_iface_members(&tcs_out, &tes_cp);
    }

    /* propagate the keys to the member copies stored in tp (filled earlier at link
     * time from the same structs, without annotations) */
    {
        struct MslStruct *parsed[3] = { have_tcs_po ? &tcs_po : NULL, have_pi ? &tes_pi : NULL,
                                        have_cp ? &tes_cp : NULL };
        for (int s = 0; s < 3; s++)
        {
            if (!parsed[s])
                continue;
            for (int i = 0; i < parsed[s]->count; i++)
            {
                MGLTessPatchMember *src = &parsed[s]->members[i];
                MGLTessPatchMember *dstarr[3] = { NULL, NULL, NULL };
                int ndst = 0, ndstmax[3] = { 0, 0, 0 };

                if (parsed[s] == &tcs_po)
                {
                    dstarr[0] = tp->tcs_patch_out;
                    ndstmax[0] = tp->num_tcs_patch_out;
                    ndst = 1;
                }
                else if (parsed[s] == &tes_pi)
                {
                    dstarr[0] = tp->tes_patch_in;
                    ndstmax[0] = tp->num_tes_patch_in;
                    ndst = 1;
                }
                else
                {
                    dstarr[0] = tp->tes_cp;
                    ndstmax[0] = tp->num_tes_cp;
                    ndst = 1;
                }
                for (int d = 0; d < ndst; d++)
                    for (int j = 0; j < ndstmax[d]; j++)
                        if (!strcmp(dstarr[d][j].name, src->name) &&
                            !strcmp(dstarr[d][j].type, src->type))
                        {
                            dstarr[d][j].key = src->key;
                            dstarr[d][j].array_idx = src->array_idx;
                            break;
                        }
            }
        }
    }

    /* ---- canonical control point records: one per interface ----
     * vertex -> TCS: union of the VS output struct and the TCS input struct
     * TCS -> TES:    union of the TCS output struct and the TES control point struct
     * (no TCS: the vertex kernel's output feeds the TES directly, one interface)
     * The pair is keyed by varying location, so a consumer that drops or reorders
     * members never shifts the record its producer writes. */
    if (tp->has_tcs)
    {
        struct MslStruct *vs_side[2] = { have_vs ? &vs_out : NULL, have_tcs_in ? &tcs_in : NULL };
        struct MslStruct *tes_side[2] = { have_tcs_out ? &tcs_out : NULL, have_cp ? &tes_cp : NULL };

        if (!tess_canon_union(&canon, vs_side, 2) ||
            !tess_canon_union(&canon_tcs, tes_side, 2))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: canonical record union failed; layouts kept per-stage\n");
            return;
        }
    }
    else
    {
        struct MslStruct *stages[2] = { have_vs ? &vs_out : NULL, have_cp ? &tes_cp : NULL };

        if (!tess_canon_union(&canon, stages, 2))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: canonical record union failed; layouts kept per-stage\n");
            return;
        }
        canon_tcs = canon;
    }
    /* ---- canonical per-patch record: union of the TCS patch outputs and TES patch
     * inputs (tessellation levels included: the blob and the vertex descriptor read
     * them from the factor record, but they occupy the same slot here) */
    {
        struct MslStruct *stages[2] = { have_tcs_po ? &tcs_po : NULL, have_pi ? &tes_pi : NULL };
        if (!tess_canon_union(&canon_patch, stages, 2))
            return;
    }

    if (tess_trace())
    {
        fprintf(stderr, "MGL tess: canonical vs->tcs record (%d members, %zu bytes):\n",
                canon.n, canon.size);
        for (int i = 0; i < canon.n; i++)
            fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                    canon.m[i].key, canon.m[i].type, canon.m[i].name,
                    canon.m[i].offset, canon.m[i].size);
        if (tp->has_tcs)
        {
            fprintf(stderr, "MGL tess: canonical tcs->tes record (%d members, %zu bytes):\n",
                    canon_tcs.n, canon_tcs.size);
            for (int i = 0; i < canon_tcs.n; i++)
                fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                        canon_tcs.m[i].key, canon_tcs.m[i].type, canon_tcs.m[i].name,
                        canon_tcs.m[i].offset, canon_tcs.m[i].size);
        }
        fprintf(stderr, "MGL tess: canonical patch record (%d members, %zu bytes):\n",
                canon_patch.n, canon_patch.size);
        for (int i = 0; i < canon_patch.n; i++)
            fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                    canon_patch.m[i].key, canon_patch.m[i].type, canon_patch.m[i].name,
                    canon_patch.m[i].offset, canon_patch.m[i].size);
    }

    /* the control point stream the TES reads: the TCS's output record, or the vertex
     * kernel's output record when no TCS runs */
    {
        const struct CanonRec *rec = tp->has_tcs ? &canon_tcs : &canon;

        tp->num_cp_record = rec->n;
        tp->cp_record_full_size = rec->size;
        tp->cp_record_size = rec->size;
        for (int i = 0; i < rec->n && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
            tp->cp_record[i] = rec->m[i];

        /* TES control point members read at the canonical offsets (attribute indices
         * unchanged: they belong to the TES's own struct). Flattened elements of an
         * arrayed varying point into the array member's element slots */
        for (int i = 0; i < tp->num_tes_cp; i++)
        {
            for (int j = 0; j < tp->num_cp_record; j++)
                if (tp->tes_cp[i].key == rec->m[j].key)
                {
                    tp->tes_cp[i].offset = rec->m[j].offset;
                    if (tp->tes_cp[i].array_idx && strstr(rec->m[j].type, "spvUnsafeArray"))
                    {
                        size_t align = 1, size = 0;

                        if (msl_element_type(rec->m[j].type, &align, &size))
                            tp->tes_cp[i].offset = rec->m[j].offset +
                                tp->tes_cp[i].array_idx * (size / msl_array_len(rec->m[j].type));
                    }
                    break;
                }
            if (tess_trace())
                fprintf(stderr, "MGL tess: tes_cp[%d] %-22s attr=%d key=%08x idx=%u -> offset %zu\n",
                        i, tp->tes_cp[i].name, tp->tes_cp[i].attr, tp->tes_cp[i].key,
                        tp->tes_cp[i].array_idx, tp->tes_cp[i].offset);
        }
    }

    /* TES patch-in members at canonical offsets (levels too: the blob is laid out
     * this way; the hardware descriptor overrides the level members to the factor
     * buffer at draw time) */
    for (int i = 0; i < tp->num_tes_patch_in; i++)
        for (int j = 0; j < canon_patch.n; j++)
            if (tp->tes_patch_in[i].key == canon_patch.m[j].key)
            {
                tp->tes_patch_in[i].offset = canon_patch.m[j].offset;
                break;
            }
    tp->tes_patch_in_size = canon_patch.size;
    if (canon_patch.size)
        tp->tcs_patch_out_size = canon_patch.size;  /* the TCS patchOut is rewritten below */

    /* the TCS patch output members (the rewritten kernel's canonical patchOut struct)
     * sit at the canonical offsets as well */
    for (int i = 0; i < tp->num_tcs_patch_out; i++)
        for (int j = 0; j < canon_patch.n; j++)
            if (tp->tcs_patch_out[i].key == canon_patch.m[j].key)
            {
                tp->tcs_patch_out[i].offset = canon_patch.m[j].offset;
                break;
            }

    /* ---- producer rewrites: whichever kernel writes a record whose SPIRV-Cross
     * struct differs from the canonical layout is recompiled with the canonical
     * struct (member names it uses are kept; pads are never referenced) */
    {
        char body[4096];

        if (have_vs && !tess_canon_matches(&vs_out, &canon))
        {
            char *nw;
            tess_canon_body(body, sizeof body, &vs_out, &canon);
            nw = msl_rewrite_struct_body(vs_msl, vs_name, body);
            /* scalar-double outputs: uint2 store in the VS body, uint4 slot in
             * the canonical record — widen the stores before compiling */
            if (nw)
                tess_widen_double_stores(&nw, &vs_out, &canon);
            if (getenv("MGL_DUMP_MSL") && nw)
                fprintf(stderr, "==== tess VS canonical MSL:\n%s\n==== end\n", nw);
            if (nw && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, nw,
                        program->spirv[_VERTEX_SHADER].entry_point, &lib, &fn, &err))
                {
                    tp->vs_function = fn;
                    tp->vs_library = lib;
                    if (tess_trace())
                        fprintf(stderr, "MGL tess: VS kernel canonicalized (record %zu b)\n",
                                canon.size);
                }
                else
                {
                    fprintf(stderr, "MGL ERROR: tess: VS canonical-record recompile failed: %s\n",
                            err ? err : "?");
                    free(err);
                }
            }
            free(nw);
        }

        if (tp->has_tcs && tcs_msl)
        {
            char *msl = strdup(tcs_msl);
            bool changed = false;

            if (have_tcs_in && !tess_canon_matches(&tcs_in, &canon))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_in, &canon);
                nw = msl_rewrite_struct_body(msl, tcs_in_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }
            if (have_tcs_out && !tess_canon_matches(&tcs_out, &canon_tcs))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_out, &canon_tcs);
                nw = msl_rewrite_struct_body(msl, tcs_out_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }
            if (have_tcs_po && !tess_canon_matches(&tcs_po, &canon_patch))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_po, &canon_patch);
                nw = msl_rewrite_struct_body(msl, po_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }

            /* triangle domains: write MTLTriangleTessellationFactorsHalf (8-byte factor
             * records) instead of the quad struct (see mgl_tess_factor_stride) */
            if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
            {
                char *step = strstr(msl, "device MTLQuadTessellationFactorsHalf*");
                if (step)
                {
                    size_t oldlen = strlen("device MTLQuadTessellationFactorsHalf*");
                    size_t newlen = strlen("device MTLTriangleTessellationFactorsHalf*");
                    /* remember the offset before the realloc: realloc may move the
                     * block and `step` would then dangle (use-after-free read, and on a
                     * moved block a corrupted kernel source) */
                    size_t at = (size_t)(step - msl);
                    char *grown = (char *)realloc(msl, strlen(msl) + newlen - oldlen + 1);
                    if (grown)
                    {
                        char *p;
                        msl = grown;
                        p = msl + at;
                        memmove(p + newlen, p + oldlen, strlen(p + oldlen) + 1);
                        memcpy(p, "device MTLTriangleTessellationFactorsHalf*", newlen);
                        for (char *w = msl; (w = strstr(w, ".insideTessellationFactor[0]")) != NULL; )
                        {
                            size_t tail = strlen(w + strlen(".insideTessellationFactor[0]")) + 1;
                            memmove(w + strlen(".insideTessellationFactor"),
                                    w + strlen(".insideTessellationFactor[0]"), tail);
                            w += strlen(".insideTessellationFactor");
                        }
                        /* gl_TessLevelInner[1] is meaningless for triangles; map it onto the
                         * scalar inside factor MTLTriangleTessellationFactorsHalf has */
                        for (char *w = msl; (w = strstr(w, ".insideTessellationFactor[1]")) != NULL; )
                        {
                            size_t tail = strlen(w + strlen(".insideTessellationFactor[1]")) + 1;
                            memmove(w + strlen(".insideTessellationFactor"),
                                    w + strlen(".insideTessellationFactor[1]"), tail);
                            w += strlen(".insideTessellationFactor");
                        }
                        /* gl_TessLevelOuter[3] is unused for triangles and the factor record
                         * only has three edges: drop such stores (keeping the right-hand
                         * side's side effects) */
                        for (char *w = msl; (w = strstr(w, ".edgeTessellationFactor[3]")) != NULL; )
                        {
                            char *ls = w, *semi = strchr(w, ';'), *eq, *rhs;
                            char repl[512];
                            while (ls > msl && ls[-1] != '\n' && ls[-1] != ';' &&
                                   ls[-1] != '{' && ls[-1] != '}')
                                ls--;
                            if (!semi) { w += strlen(".edgeTessellationFactor"); continue; }
                            eq = semi;
                            while (eq > ls && eq[-1] != '=') eq--;
                            if (eq == ls) { w = semi; continue; }   /* not a plain assignment */
                            rhs = eq;
                            while (*rhs == ' ' || *rhs == '=') rhs++;
                            snprintf(repl, sizeof repl, "(void)(%.*s);", (int)(semi - rhs), rhs);
                            size_t span = (size_t)(semi - ls + 1);
                            memset(ls, ' ', span);
                            memcpy(ls, repl, strlen(repl) <= span ? strlen(repl) : span);
                            w = ls + span;
                        }
                        changed = true;
                    }
                }
            }

            if (changed && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                if (getenv("MGL_DUMP_MSL"))
                    fprintf(stderr, "==== canonical TCS MSL:\n%s\n==== end\n", msl);
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl,
                        program->spirv[_TESS_CONTROL_SHADER].entry_point, &lib, &fn, &err))
                {
                    tp->tcs_function = fn;
                    tp->tcs_library = lib;
                    if (tess_trace())
                        fprintf(stderr, "MGL tess: TCS kernel canonicalized (cp %zu b, patch %zu b)\n",
                                canon.size, canon_patch.size);
                }
                else
                {
                    fprintf(stderr, "MGL ERROR: tess: TCS canonical-record recompile failed: %s\n",
                            err ? err : "?");
                    free(err);
                }
            }
            free(msl);
        }
    }
}

/* ------------------------------------------------------------------ MSL name helpers */

/* name of the TES patch input struct: the type of the "patchIn" parameter (hardware
 * form: "<patchIn> patchIn [[stage_in]]") or of the prologue local (plain form) */
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len)
{
    const char *decl = strstr(msl, " patchIn [[stage_in]]");

    if (!decl)
        decl = strstr(msl, " patchIn;");
    if (!decl)
        return false;

    const char *e = decl;
    while (e > msl && *(e - 1) == ' ')
        e--;
    const char *b = e;
    while (b > msl && *(b - 1) != ' ' && *(b - 1) != '\n' && *(b - 1) != '(' && *(b - 1) != ',')
        b--;
    if (b == e)
        return false;
    snprintf(out, out_len, "%.*s", (int)(e - b), b);
    return true;
}

/* control point struct name of the TES ("patch_control_point<T> gl_in;" in the
 * hardware form, "mgl_cp_ref<T> gl_in;" in the plain form) */
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len)
{
    char head[160];
    const char *p;

    snprintf(head, sizeof head, "struct %s\n{", patchin_name);
    p = strstr(msl, head);
    if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
    if (!p)
        return false;
    const char *ref = strstr(p, "patch_control_point<");
    if (!ref)
        ref = strstr(p, "mgl_cp_ref<");
    if (!ref || ref > strchr(p, '}'))
        return false;
    const char *t0 = ref + (strstr(ref, "<") == ref + strlen("patch_control_point<") - 1
                            ? strlen("patch_control_point<") : strlen("mgl_cp_ref<"));
    /* t0 currently points at the start of the template argument either way */
    {
        const char *lt = strchr(p, '<');
        if (!lt) return false;
        t0 = lt + 1;
    }
    const char *t1 = strchr(t0, '>');
    if (!t1 || (size_t)(t1 - t0) >= out_len)
        return false;
    memcpy(out, t0, (size_t)(t1 - t0));
    out[t1 - t0] = 0;
    return true;
}


/* name of the control point record struct: the TCS's spvIn element type, or the VS
 * kernel's spvOut element type */
static bool tess_cp_record_name(const char *msl, char *out, size_t out_len)
{
    const char *mark = strstr(msl, "device ");
    const char *star = NULL;
    char word[128];
    const char *b, *e;

    while (mark)
    {
        /* device TYPE* spvIn / device TYPE* spvOut */
        if (!strncmp(mark + 7, "const ", 6))
            mark = strstr(mark + 1, "device ");
        star = strchr(mark, '*');
        if (star && star - mark < 120)
        {
            const char *after = star + 1;
            while (*after == ' ') after++;
            if (!strncmp(after, "spvIn", 5) || !strncmp(after, "spvOut", 6))
            {
                b = mark + 7;
                if (!strncmp(b, "const ", 6)) b += 6;
                e = star;
                while (e > b && *(e - 1) == ' ') e--;
                if (e > b && (size_t)(e - b) < out_len)
                {
                    snprintf(out, out_len, "%.*s", (int)(e - b), b);
                    return true;
                }
            }
        }
        mark = strstr(mark + 1, "device ");
        (void)word;
    }
    return false;
}


/* find the "<name>_patchOut" struct of a TCS (any name) */
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out,
                                   char *struct_name, size_t struct_len)
{
    const char *at = msl;

    if (struct_name && struct_len)
        struct_name[0] = 0;
    while ((at = strstr(at, "_patchOut")) != NULL)
    {
        const char *b = at;
        char name[160];

        while (b > msl && *(b - 1) != ' ' && *(b - 1) != '\n')
            b--;
        if (b == at || b == msl || strncmp(b - 7, "struct ", 7))
        {
            at += 9;
            continue;
        }
        snprintf(name, sizeof name, "%.*s", (int)(at + 9 - b), b);
        if (msl_parse_struct(msl, name, out))
        {
            if (struct_name && struct_len)
                snprintf(struct_name, struct_len, "%s", name);
            return true;
        }
        at += 9;
    }
    return false;
}

/* ------------------------------------------------------------------ GL_PATCHES draw
 * Called from draw_buffers.c once the draw is validated: hands the draw to the
 * Metal side (see MGLRenderer+Tessellation.m).
 */
bool mgl_tess_draw_patches(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                           GLint first, GLsizei count, GLsizei instances,
                           GLint basevertex, GLint baseinstance)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return false;

    tp = mgl_tess_active_program(ctx);
    if (!tp)
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: draw rejected: no active tess program (prog=%p)\n",
                    (void *)ctx->state.program);
        return false;
    }

    if (!ctx->mtl_funcs.ext_tess || !ctx->mtl_funcs.ext_tess->draw_patches)
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: draw rejected: no draw hook\n");
        return false;
    }

    return ctx->mtl_funcs.ext_tess->draw_patches(ctx, indexed, type, offset, first, count,
                                                 instances ? instances : 1, basevertex, baseinstance);
}

/* the tessellation program of the current GL state (program object or separable pipeline) */
MGLTessProgram *mgl_tess_active_program(GLMContext ctx)
{
    Program *tes = NULL;

    if (ctx->state.program)
    {
        tes = ctx->state.program;
    }
    else if (ctx->state.program_pipeline)
    {
        if (ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER])
            tes = ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER];
        else if (ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER])
            tes = ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER];
    }

    if (!tes)
        return NULL;
    return mgl_tess_program(ctx, tes);
}

/* record size of the TES control-point struct (rewritten TES MSL), so the Metal side
 * can size its scratch buffers without re-parsing the shader source */
bool mgl_tess_cp_record_size(const char *msl, char *patchin_name, size_t patchin_len,
                             char *cp_name, size_t cp_len, size_t *record_size)
{
    struct MslStruct cp;
    char pin[160], cpn[128];

    if (!msl || !msl_patch_in_name(msl, pin, sizeof pin))
        return false;
    if (!msl_cp_struct_name(msl, pin, cpn, sizeof cpn))
        return false;
    if (!msl_parse_struct(msl, cpn, &cp))
        return false;
    if (patchin_name && patchin_len)
        snprintf(patchin_name, patchin_len, "%s", pin);
    if (cp_name && cp_len)
        snprintf(cp_name, cp_len, "%s", cpn);
    *record_size = cp.size;
    return cp.size > 0;
}

/* exported for MGLRenderer+Tessellation.m */
bool mgl_tess_generate(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches)
{
    tess_cpu_generate_runs++;
    return tess_generate(ctx, tp, factors, patches);
}

/* precompute, once per program, how the GPU blob kernel must assemble each patch's
 * TES patch-in record: for every TES patch-in member, either the tessellation
 * levels (from the half factor record) or a byte range of the TCS patch output,
 * matched by varying name exactly like tess_build_patch_blob does on the CPU. */
void mgl_tess_build_blob_map(GLMContext ctx, MGLTessProgram *tp)
{
    int n = 0;

    (void)ctx;
    tp->blob_map_n = 0;
    for (int i = 0; i < tp->num_tes_patch_in && n < MGL_TESS_BLOB_MAX; i++)
    {
        const MGLTessPatchMember *m = &tp->tes_patch_in[i];
        uint32_t kind = 0, src = 0, size = 0;

        if (!strcmp(m->name, "gl_TessLevel") || !strcmp(m->name, "gl_TessLevelOuter"))
            kind = MGL_TESS_BLOB_OUTER;
        else if (!strcmp(m->name, "gl_TessLevelInner"))
            kind = MGL_TESS_BLOB_INNER;
        else
        {
            /* copy from the TCS patch output record, matched by varying location
             * (the canonical key — names may differ between the stages) */
            bool found = false;
            for (int j = 0; j < tp->num_tcs_patch_out && !found; j++)
            {
                if (tp->tcs_patch_out[j].key == m->key)
                {
                    src = (uint32_t)tp->tcs_patch_out[j].offset;
                    size = (uint32_t)m->size;
                    kind = MGL_TESS_BLOB_COPY;
                    found = true;
                }
            }
            if (!found)
                continue;               /* no producer: stays zero (undefined per GL) */
        }

        tp->blob_kind[n] = kind;
        tp->blob_dst[n] = (uint32_t)m->offset;
        tp->blob_src[n] = src;
        tp->blob_size[n] = size;
        n++;
    }
    tp->blob_map_n = n;
}

void mgl_tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                               const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                               const float *factors, unsigned patches)
{
    tess_build_patch_blob(ctx, tp, tcs_patch_out, tcs_patch_stride, factors, patches);
}

GLsizei mgl_tess_patch_vertices_in(GLMContext ctx, MGLTessProgram *tp)
{
    if (tp->has_tcs)
        return tp->tcs_out_vertices;
    if (ctx->state.ext_rstate)
        return ((struct MGLExtState_rstate *)ctx->state.ext_rstate)->patch_vertices;
    return 3;
}

/* the tessellator's factor record stride: 8 bytes for triangles (3 edge + 1 inside
 * halfs), 12 for quads/isolines (4 edge + 2 inside) */
size_t mgl_tess_factor_stride(GLMContext ctx, MGLTessProgram *tp)
{
    if (tp && tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
        return 8;
    return 12;
}

/* post-link: fix the TCS factor-struct type for triangle programs and recompile the
 * kernel. SPIRV-Cross types the factor buffer as MTLQuadTessellationFactorsHalf (12-
 * byte records) because the TCS alone doesn't know the domain, and the domain is only
 * known once the TES linked. The corrected function lives on the MGLTessProgram and
 * is preferred by the draw path. */

/* ---- tests-only capture access (mgl_tess_diag.h) ---- */
extern GLMContext _ctx;

void mgl_tess_diag_capture(GLuint program, int enable)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)(_ctx ? _ctx->state.ext_tess : NULL);

    (void)program;
    if (S && _ctx)
    {
        MGLTessProgram *tp = mgl_tess_active_program(_ctx);
        if (tess_trace())
            fprintf(stderr, "MGL tess: diag_capture(enable=%d): _ctx=%p tp=%p\n",
                    enable, (void *)_ctx, (void *)tp);
        if (tp)
            tp->capture = enable != 0;
    }
}

bool mgl_tess_diag_capture_info(void **buffer, uint32_t *records_offset, uint32_t *stride_words,
                                uint32_t *prim, uint32_t *count)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)(_ctx ? _ctx->state.ext_tess : NULL);

    if (!S || !S->capture.valid || !buffer)
        return false;
    *buffer = S->capture.buffer;
    if (records_offset) *records_offset = (uint32_t)S->capture.records_offset;
    if (stride_words)   *stride_words = S->capture.stride;
    if (prim)           *prim = S->capture.prim;
    if (count)
    {
        const uint32_t *w = (const uint32_t *)((const uint8_t *)S->capture.buffer +
                                               S->capture.count_offset);
        *count = *w;
        if (tess_trace())
            fprintf(stderr, "MGL tess: diag capture_info: buf=%p count_off=%zu val=%u "
                            "rec_off=%zu idx_off=%zu\n",
                    S->capture.buffer, (size_t)S->capture.count_offset, *w,
                    (size_t)S->capture.records_offset, (size_t)S->capture.index_offset);
    }
    return true;
}
