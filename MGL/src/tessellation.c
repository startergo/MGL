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
#include "glcorearb.h"

/* ------------------------------------------------------------------ helpers */

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
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len);
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len);
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out);

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
            if (getenv("MGL_TESS_DEBUG"))
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

        /* strip [[...]] attribute (keep its number for later use) */
        attr = strstr(member, "[[");
        if (attr)
        {
            const char *aend = strstr(attr, "]]");
            if (!aend) continue;
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
        snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", type);
        snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s", namebuf);
        cursor += size;
        out->count++;
        (void)i;
    }

    out->size = cursor;
    return true;        /* the struct was found; count may be 0 (no data members) */
}

/* ------------------------------------------------------------------ MSL rewrites
 * Called from program.c's compile_stage right after SPIRV-Cross produced the MSL.
 */

static void str_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap;
    int need;
    char tmp[2048];

    va_start(ap, fmt);
    need = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (need < 0)
        return;
    if (*len + (size_t)need + 1 > *cap)
    {
        *cap = (*len + (size_t)need + 1) * 2;
        *buf = (char *)realloc(*buf, *cap);
    }
    memcpy(*buf + *len, tmp, (size_t)need + 1);
    *len += (size_t)need;
}

/* replace an exact occurrence of old with new inside a buffer (new may be shorter/longer) */
static bool str_replace_exact(char *buf, size_t cap, const char *old, const char *new)
{
    char *at = strstr(buf, old);
    size_t oldlen = strlen(old), newlen = strlen(new), taillen;

    if (!at || strlen(buf) + newlen + 1 > cap + oldlen)   /* cap check is advisory; buffer is heap grown below */
        return false;
    taillen = strlen(at + oldlen);
    memmove(at + newlen, at + oldlen, taillen + 1);
    memcpy(at, new, newlen);
    return true;
}

/* VS kernel: route base vertex/instance through our params buffer.
 * GL: gl_VertexID includes base vertex; gl_InstanceID excludes base instance. */
static char *rewrite_vs_kernel(const char *msl)
{
    static const char *vtxid = "uint gl_VertexID = as_uint(gl_GlobalInvocationID.x + spvDispatchBase.x);";
    static const char *vtxid2 = "uint gl_VertexID = gl_GlobalInvocationID.x + spvDispatchBase.x;";
    static const char *instid = "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y + spvDispatchBase.y);";
    static const char *instid2 = "uint gl_InstanceID = gl_GlobalInvocationID.y + spvDispatchBase.y;";
    const char *kern = strstr(msl, "kernel void ");
    char *out;
    size_t len = 0, cap = 0;

    if (!kern)
        return NULL;

    out = NULL;
    str_append(&out, &len, &cap, "%s", msl);

    if (strstr(out, vtxid))
        str_replace_exact(out, cap, vtxid, "uint gl_VertexID = as_uint(gl_GlobalInvocationID.x + mgl_vs_params[0]);");
    else if (strstr(out, vtxid2))
        str_replace_exact(out, cap, vtxid2, "uint gl_VertexID = gl_GlobalInvocationID.x + mgl_vs_params[0];");
    if (strstr(out, instid))
        str_replace_exact(out, cap, instid, "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y);");
    else if (strstr(out, instid2))
        str_replace_exact(out, cap, instid2, "uint gl_InstanceID = gl_GlobalInvocationID.y;");

    if (!strstr(out, "mgl_vs_params"))
    {
        /* shader did not use the ids: still add the parameter for uniformity */
        free(out);
        return strdup(msl);
    }

    /* append the params parameter to the kernel signature */
    {
        char *sig_end = strchr(out, ')');
        /* find the signature end: first ')' after "kernel void <name>(" that is not inside "[[ ]]" */
        char *p = strstr(out, "kernel void ");
        char *paren = p ? strchr(p, '(') : NULL;
        int depth = 0;
        char *close = NULL;
        if (!paren) { free(out); return strdup(msl); }

        for (char *q = paren; q && *q; q++)
        {
            if (*q == '[' && q[1] == '[') { const char *e = strstr(q, "]]"); if (e) q = (char *)e; continue; }
            if (*q == '(') depth++;
            else if (*q == ')') { depth--; if (depth == 0) { close = q; break; } }
        }
        if (!close)
        {
            free(out);
            return strdup(msl);
        }
        (void)sig_end;

        /* insert ", constant uint* mgl_vs_params [[buffer(N)]]" before close */
        {
            char add[128];
            size_t addlen, taillen;
            snprintf(add, sizeof add, ", constant uint* mgl_vs_params [[buffer(%d)]]", MGL_TESS_VS_PARAMS);
            addlen = strlen(add);
            taillen = strlen(close);
            out = (char *)realloc(out, len + addlen + 1);
            memmove(close + addlen, close, taillen + 1);
            memcpy(close, add, addlen);
        }
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

static char *rewrite_tes(const char *msl)
{
    const char *sig = strstr(msl, "[[ patch(");
    if (!sig)
        sig = strstr(msl, "[[patch(");
    if (!sig)
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 9); return NULL; }

    const char *vertex_kw = strstr(sig, "vertex ");
    if (!vertex_kw || (size_t)(vertex_kw - sig) > 64)
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 13); return NULL; }
    const char *sig_open = strchr(vertex_kw, '(');
    if (!sig_open)
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 16); return NULL; }

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
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 32); return NULL; }

    /* the opening brace of the function body */
    const char *body_open = strchr(sig_close, '{');
    if (!body_open)
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 37); return NULL; }

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

    /* patchIn struct type name: first word of the parameter list */
    char patchin_name[128];
    {
        const char *p = sig_open + 1;
        while (*p == ' ' || *p == '\n') p++;
        const char *e = p;
        while (*e && *e != ' ' && *e != '\n') e++;
        if (e == p || (size_t)(e - p) >= sizeof patchin_name)
            { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 62); return NULL; }
        snprintf(patchin_name, sizeof patchin_name, "%.*s", (int)(e - p), p);
    }

    /* control point struct type: from patch_control_point<T> in the patchIn struct */
    char cp_type[128];
    {
        char head[160];
        const char *p;
        snprintf(head, sizeof head, "struct %s\n{", patchin_name);
        p = strstr(msl, head);
        if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
        if (!p)
            { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 75); return NULL; }
        const char *pcp = strstr(p, "patch_control_point<");
        if (!pcp || pcp > strchr(p, '}'))
            { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 78); return NULL; }
        const char *t0 = pcp + strlen("patch_control_point<");
        const char *t1 = strchr(t0, '>');
        if (!t1 || (size_t)(t1 - t0) >= sizeof cp_type)
            { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 82); return NULL; }
        memcpy(cp_type, t0, (size_t)(t1 - t0));
        cp_type[t1 - t0] = 0;
    }

    struct MslStruct patch_in;
    if (!msl_parse_struct(msl, patchin_name, &patch_in))
    {
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess: rewrite_tes: patchIn struct parse failed (%s)\n", patchin_name);
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 92); return NULL; }
    }
    if (getenv("MGL_TESS_DEBUG"))
        fprintf(stderr, "MGL tess: rewrite_tes: sig ok, cp=%s patchin=%s members=%d cpref=%d\n",
                cp_type, patchin_name, patch_in.count, patch_in.cp_ref_member);

    /* ---- assemble the rewritten source ---- */
    char *out = NULL;
    size_t len = 0, cap = 0;

    const char *ins = strstr(msl, "using namespace metal;");
    if (!ins)
        { fprintf(stderr, "MGL tess: rewrite_tes NULL at rel-line %d\n", 104); return NULL; }
    ins += strlen("using namespace metal;");
    str_append(&out, &len, &cap, "%.*s%s", (int)(ins - msl), msl, kTessPrelude);

    /* everything between the prelude and "[[ patch(" */
    str_append(&out, &len, &cap, "%.*s", (int)(sig - ins), ins);

    /* new signature */
    str_append(&out, &len, &cap,
               "vertex %s %s(device const float4* mgl_tess_coords [[buffer(%d)]], "
               "device const %s* mgl_tess_cp [[buffer(%d)]], "
               "device const uchar* mgl_tess_patch [[buffer(%d)]], "
               "constant uint* mgl_tess_params [[buffer(%d)]], "
               "uint mgl_vid [[vertex_id]])",
               entry_ret, entry_name, MGL_TESS_TES_COORDS, cp_type, MGL_TESS_TES_CP,
               MGL_TESS_TES_PATCH, MGL_TESS_TES_PARAMS);

    /* prologue: rebuild patchIn from the buffers */
    str_append(&out, &len, &cap, "\n{\n    %s patchIn;\n", patchin_name);
    for (int i = 0; i < patch_in.count; i++)
    {
        if (i == patch_in.cp_ref_member)
            continue;
        str_append(&out, &len, &cap,
                   "    patchIn.%s = *(device const %s*)(mgl_tess_patch + %zu);\n",
                   patch_in.members[i].name, patch_in.members[i].type, patch_in.members[i].offset);
    }
    if (patch_in.cp_ref_member >= 0)
        str_append(&out, &len, &cap,
                   "    patchIn.%s = mgl_cp_ref<%s>{ &mgl_tess_cp[(uint)mgl_tess_coords[mgl_vid].w * mgl_tess_params[0]], mgl_tess_params[0] };\n",
                   patch_in.members[patch_in.cp_ref_member].name, cp_type);

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

    /* strip [[attribute(N)]] from the two struct declarations and replace
       patch_control_point<T> by mgl_cp_ref<T> so the types work outside stage_in */
    const char *decl_names[2] = { patchin_name, cp_type };
    for (int d = 0; d < 2; d++)
    {
        char head[160];
        const char *p;
        snprintf(head, sizeof head, "struct %s\n{", decl_names[d]);
        p = strstr(out, head);
        if (!p) { snprintf(head, sizeof head, "struct %s {", decl_names[d]); p = strstr(out, head); }
        if (!p)
            continue;
        const char *open = strchr(p, '{');
        const char *close = strchr(open, '}');
        if (!open || !close)
            continue;

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
            memmove(at + strlen("mgl_cp_ref<"), at + plen, strlen(at + plen) + 1);
            memcpy(at, "mgl_cp_ref<", strlen("mgl_cp_ref<"));
            at += strlen("mgl_cp_ref<");
        }

        size_t prefix = (size_t)(open + 1 - out);
        size_t newlen = len - bodylen + strlen(body);
        out = (char *)realloc(out, newlen + 1);
        if (!out) { free(body); return NULL; }
        memmove(out + prefix + strlen(body), open + 1 + bodylen, len - (prefix + bodylen) + 1);
        memcpy(out + prefix, body, strlen(body));
        len = newlen;
        free(body);
    }

    return out;
}

/* entry from program.c: rewrite the MSL of a tessellation-pipeline stage */
char *mgl_tess_rewrite_stage_msl(GLMContext ctx, int stage, bool tess_pipeline, const char *msl)
{
    if (!msl)
        return NULL;


    if (stage == _TESS_EVALUATION_SHADER)
        return rewrite_tes(msl);

    if (stage == _VERTEX_SHADER && tess_pipeline)
        return rewrite_vs_kernel(msl);

    return NULL;
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
 * - anything else stays zero (undefined per GL when no TCS writes it) */
static void tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                                  const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                                  const float *factors, unsigned patches)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    size_t need = (size_t)tp->tes_patch_in_size * patches;

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
                /* copy from the TCS patch output by member name */
                for (int j = 0; j < tp->num_tcs_patch_out; j++)
                {
                    if (!strcmp(tp->tcs_patch_out[j].name, m->name) &&
                        tp->tcs_patch_out[j].size == m->size)
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
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess: program %u linked domain=%d spacing=%d ccw=%d point=%d tcs=%d outverts=%d\n",
                    program->name, tp->domain, tp->spacing, tp->vertex_order_ccw, tp->point_mode,
                    tp->has_tcs, tp->tcs_out_vertices);
    }

    /* patch data layouts, parsed from the generated MSL */
    if (has_tcs)
    {
        struct MslStruct po;
        if (msl_parse_struct(program->spirv[_TESS_CONTROL_SHADER].msl_str, "main0_patchOut", &po) ||
            msl_parse_struct_entry(program->spirv[_TESS_CONTROL_SHADER].msl_str, &po))
        {
            tp->num_tcs_patch_out = po.count;
            tp->tcs_patch_out_size = po.size;
            for (int i = 0; i < po.count && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
                tp->tcs_patch_out[i] = po.members[i];
        }
    }

    {
        struct MslStruct pi;
        const char *msl = program->spirv[_TESS_EVALUATION_SHADER].msl_str;
        char name[160];
        bool pin_ok = msl_patch_in_name(msl, name, sizeof name);
        (void)pin_ok;
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess: post-link: TES msl=%p patchin_name=%s ok=%d\n",
                    (const void *)msl, pin_ok ? name : "(none)", pin_ok);
        if (pin_ok && msl_parse_struct(msl, name, &pi))
        {
            tp->num_tes_patch_in = pi.count;
            tp->tes_patch_in_size = pi.size;
            int m = 0;
            for (int i = 0; i < pi.count && m < MGL_TESS_MAX_PATCH_MEMBERS; i++)
            {
                if (i == pi.cp_ref_member)
                {
                    tp->num_tes_patch_in--;
                    continue;
                }
                tp->tes_patch_in[m++] = pi.members[i];
            }
            /* control point record size from the cp struct */
            struct MslStruct cp;
            char cpname[128];
            if (msl_cp_struct_name(msl, name, cpname, sizeof cpname))
            {
                if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL tess: post-link TES cp struct=%s\n", cpname);
                if (msl_parse_struct(msl, cpname, &cp))
                {
                    tp->cp_record_size = cp.size;
                    if (getenv("MGL_TESS_DEBUG"))
                        fprintf(stderr, "MGL tess: cp record size=%zu (%d members)\n", cp.size, cp.count);
                }
            }
        }
    }
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
void mgl_tess_note_draw(GLMContext ctx, GLuint patches, GLuint tes_invocations, GLuint tessellated_primitives)
{
    mgl_queries_note_tess(ctx, tes_invocations, tessellated_primitives);
}

/* ------------------------------------------------------------------ MSL name helpers */

/* name of the TES patch input struct: the type of the local "patchIn" the rewritten
 * entry point declares in its prologue ("<patchIn> patchIn;") */
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len)
{
    const char *decl = strstr(msl, " patchIn;");

    if (!decl)
        return false;
    const char *e = decl;
    while (e > msl && *(e - 1) == ' ')
        e--;
    const char *b = e;
    while (b > msl && *(b - 1) != ' ' && *(b - 1) != '\n')
        b--;
    if (b == e)
        return false;
    snprintf(out, out_len, "%.*s", (int)(e - b), b);
    return true;
}

/* control point struct name of a rewritten TES ("mgl_cp_ref<T> gl_in;") */
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len)
{
    char head[160];
    const char *p;

    snprintf(head, sizeof head, "struct %s\n{", patchin_name);
    p = strstr(msl, head);
    if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
    if (!p)
        return false;
    const char *ref = strstr(p, "mgl_cp_ref<");
    if (!ref)
        return false;
    const char *t0 = ref + strlen("mgl_cp_ref<");
    const char *t1 = strchr(t0, '>');
    if (!t1 || (size_t)(t1 - t0) >= out_len)
        return false;
    memcpy(out, t0, (size_t)(t1 - t0));
    out[t1 - t0] = 0;
    return true;
}

/* find the "<name>_patchOut" struct of a TCS (any name) */
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out)
{
    const char *at = msl;

    while ((at = strstr(at, "_patchOut")) != NULL)
    {
        /* must be a struct declaration: "struct <name>_patchOut" */
        const char *s = at;
        while (s > msl && *(s - 1) != ' ' && *(s - 1) != '\n')
            s--;
        if (!s || strncmp(s - 7 < msl ? "" : s - 7, "struct ", 7))
        {
            at += 9;
            continue;
        }
        char name[160];
        snprintf(name, sizeof name, "%.*s", (int)(at + 9 - s), s);
        if (msl_parse_struct(msl, name, out))
            return true;
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

    if (getenv("MGL_TESS_DEBUG"))
        fprintf(stderr, "MGL tess: draw_patches entered: state=%p program=%p pipeline=%p hooks=%p\n",
                (void *)S, (void *)ctx->state.program, (void *)ctx->state.program_pipeline,
                (void *)(ctx->mtl_funcs.ext_tess ? ctx->mtl_funcs.ext_tess->draw_patches : NULL));

    if (!S)
        return false;

    tp = mgl_tess_active_program(ctx);
    if (!tp)
        return false;

    if (!ctx->mtl_funcs.ext_tess || !ctx->mtl_funcs.ext_tess->draw_patches)
        return false;

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
    return tess_generate(ctx, tp, factors, patches);
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
