/*
 * msl_reserved.c — rename GLSL identifiers that collide with MSL reserved words.
 * See msl_reserved.h.  (KHR-GL46 cluster #1: 736 pixelstoragemodes cases et al.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include "spirv.h"
#include "msl_reserved.h"

/*
 * MSL reserved identifiers: every MSL keyword / reserved word (Metal Shading Language
 * specification §4.1), every built-in scalar/vector/matrix/texture/sampler type, and the
 * Metal std-lib function names GLSL shaders commonly shadow.  GLSL reserves almost none of
 * these, so a conforming GLSL shader may use any of them as a variable, function, block or
 * member name; the MSL SPIRV-Cross emits then does not compile.
 */
static const char *const msl_reserved[] = {
    /* --- MSL / C++ keywords and reserved words --- */
    "array", "bool", "break", "bfloat", "case", "catch", "char", "class", "const",
    "constexpr", "continue", "default", "delete", "discard_fragment", "do", "double",
    "dynamic_cast", "else", "enum", "extern", "false", "float", "for", "fragment",
    "friend", "goto", "half", "if", "inline", "int", "long", "metal", "namespace", "new",
    "operator", "private", "protected", "public", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "template", "this", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "while",
    /* function qualifiers, address spaces, attribute / argument qualifiers */
    "kernel", "vertex", "compute", "device", "constant", "thread", "threadgroup",
    "threadgroup_imageblock", "imageblock", "ray_data", "polygon_data", "quad_data",
    "mesh", "object", "object_data", "mesh_grid_properties", "amplification", "attribute",
    "uniform", "buffer", "in", "out", "inout", "restrict", "read_only", "write_only",
    "read_write", "packed", "unorm", "snorm", "fixed", "noinline", "indirect",
    "early_fragment_tests", "no_early_z", "sample", "point", "line", "triangle",
    "triangle_strip", "patch", "quad", "flat", "smooth", "noperspective", "centroid",
    "nointerpolation", "perprimitive", "pervertex", "color", "depth", "stencil",
    /* --- built-in scalar, vector and matrix types --- */
    "size_t", "ptrdiff_t", "intptr_t", "uintptr_t", "vector", "matrix",
    "bool2", "bool3", "bool4",
    "char2", "char3", "char4", "uchar2", "uchar3", "uchar4",
    "short2", "short3", "short4", "ushort2", "ushort3", "ushort4",
    "int2", "int3", "int4", "uint2", "uint3", "uint4",
    "long2", "long3", "long4", "ulong2", "ulong3", "ulong4",
    "half2", "half3", "half4", "float2", "float3", "float4",
    "bfloat2", "bfloat3", "bfloat4",
    "packed_char2", "packed_char3", "packed_short2", "packed_short3",
    "packed_uchar2", "packed_uchar3", "packed_ushort2", "packed_ushort3",
    "packed_half2", "packed_half3", "packed_bfloat2", "packed_bfloat3",
    "packed_float2", "packed_float3", "packed_int2", "packed_int3",
    "packed_uint2", "packed_uint3", "packed_long2", "packed_ulong2",
    "half2x2", "half2x3", "half2x4", "half3x2", "half3x3", "half3x4",
    "half4x2", "half4x3", "half4x4",
    "float2x2", "float2x3", "float2x4", "float3x2", "float3x3", "float3x4",
    "float4x2", "float4x3", "float4x4",
    "bfloat2x2", "bfloat2x3", "bfloat2x4", "bfloat3x2", "bfloat3x3", "bfloat3x4",
    "bfloat4x2", "bfloat4x3", "bfloat4x4",
    /* --- texture / sampler / ray-tracing built-in types --- */
    "sampler", "texture1d", "texture1d_array", "texture2d", "texture2d_array", "texture3d",
    "texturecube", "texturecube_array", "texture2d_ms", "texture2d_ms_array",
    "texture_buffer", "depth2d", "depth2d_array", "depth3d", "depthcube",
    "depthcube_array", "depth2d_ms", "depth2d_ms_array", "texture", "depth",
    "ray", "intersector", "intersection", "acceleration_structure",
    "acceleration_structure_ref", "instancing_args", "visible_function_table",
    /* --- Metal std-lib functions GLSL shaders commonly shadow --- */
    "abs", "fabs", "acos", "acospi", "addsat", "all", "any", "asin", "asinpi", "atan",
    "atan2", "atan2pi", "atanpi", "ceil", "clamp", "copysign", "cos", "cosh", "cospi",
    "cross", "degrees", "determinant", "distance", "dot", "equal", "exp", "exp2", "exp10",
    "faceforward", "fast_normalize", "floor", "fma", "fmax", "fmax3", "fmin", "fmin3",
    "fmod", "fract", "frexp", "gather", "isfinite", "isinf", "isnan", "isnormal",
    "isordered", "isunordered", "ldexp", "length", "log", "log2", "log10", "mad", "max",
    "median3", "min", "mix", "modf", "mul24", "mad24", "normalize", "not", "not_equal",
    "outerProduct", "popcount", "pow", "pown", "powr", "radians", "rcp", "reflect",
    "refract", "remainder", "reverse_bits", "rint", "rootn", "round", "rsqrt", "sample",
    "saturate", "select", "shuffle", "shuffle_down", "shuffle_up", "shuffle_xor", "sign",
    "sin", "sinh", "sinpi", "smoothstep", "sqrt", "step", "tan", "tanh", "tanpi",
    "transpose", "trunc", "simd_shuffle", "simd_shuffle_down", "simd_shuffle_up",
    "simd_shuffle_xor", "simd_sum", "simd_prefix_sum",
    /* GLSL built-in function names that survive into generated MSL namespaces */
    "texture", "texelFetch", "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual",
    "notEqual", "matrixCompMult", "EmitVertex", "EndPrimitive",
    /* MSL preprocessor / macro namespace (also fixed up by SPIRV-Cross, keep for safety) */
    "assert", "is_function_constant_defined", "CHAR_BIT", "SCHAR_MAX", "SCHAR_MIN",
    "UCHAR_MAX", "CHAR_MAX", "CHAR_MIN", "USHRT_MAX", "SHRT_MAX", "SHRT_MIN", "UINT_MAX",
    "INT_MAX", "INT_MIN", "FLT_DIG", "FLT_MANT_DIG", "FLT_MAX_10_EXP", "FLT_MAX_EXP",
    "FLT_MIN_10_EXP", "FLT_MIN_EXP", "FLT_RADIX", "FLT_MAX", "FLT_MIN", "FLT_EPSILON",
    "DBL_DIG", "DBL_MANT_DIG", "DBL_MAX_10_EXP", "DBL_MAX_EXP", "DBL_MIN_10_EXP",
    "DBL_MIN_EXP", "DBL_RADIX", "DBL_MAX", "DBL_MIN", "DBL_EPSILON", "M_E_F", "M_LOG2E_F",
    "M_LOG10E_F", "M_LN2_F", "M_LN10_F", "M_PI_F", "M_PI_2_F", "M_PI_4_F", "M_1_PI_F",
    "M_2_PI_F", "M_2_SQRTPI_F", "M_SQRT2_F", "M_SQRT1_2_F", "INFINITY", "NAN",
};

int mgl_msl_is_reserved_name(const char *name)
{
    if (!name || !*name)
        return 0;
    for (size_t i = 0; i < sizeof msl_reserved / sizeof msl_reserved[0]; i++)
        if (!strcmp(msl_reserved[i], name))
            return 1;
    return 0;
}

/* ------------------------------------------------------------------ name set (uniqueness)
 * open-addressing hash set of strdup'd names, grown on load > 2/3 */
typedef struct {
    char **slots;
    size_t cap;
    size_t count;
} NameSet;

static unsigned name_hash(const char *s)
{
    unsigned h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

static void nameset_rehash(NameSet *ns)
{
    size_t new_cap = ns->cap ? ns->cap * 2 : 256;
    char **slots = (char **)calloc(new_cap, sizeof(char *));
    if (!slots)
        return;                                   /* degraded: duplicates stay possible */
    for (size_t i = 0; i < ns->cap; i++)
    {
        if (!ns->slots[i])
            continue;
        size_t j = name_hash(ns->slots[i]) & (new_cap - 1);
        while (slots[j]) j = (j + 1) & (new_cap - 1);
        slots[j] = ns->slots[i];
    }
    free(ns->slots);
    ns->slots = slots;
    ns->cap = new_cap;
}

static void nameset_add(NameSet *ns, const char *name)
{
    if ((ns->count + 1) * 3 > ns->cap * 2)
        nameset_rehash(ns);
    if (!ns->cap)
        return;
    size_t j = name_hash(name) & (ns->cap - 1);
    while (ns->slots[j])
    {
        if (!strcmp(ns->slots[j], name))
            return;
        j = (j + 1) & (ns->cap - 1);
    }
    ns->slots[j] = strdup(name);
    ns->count++;
}

static int nameset_has(NameSet *ns, const char *name)
{
    if (!ns->cap)
        return 0;
    size_t j = name_hash(name) & (ns->cap - 1);
    while (ns->slots[j])
    {
        if (!strcmp(ns->slots[j], name))
            return 1;
        j = (j + 1) & (ns->cap - 1);
    }
    return 0;
}

static void nameset_free(NameSet *ns)
{
    for (size_t i = 0; i < ns->cap; i++) free(ns->slots[i]);
    free(ns->slots);
}


/* ------------------------------------------------------------------ SPIR-V name instructions */

/* decode the literal string starting at word `from` of an instruction */
static void spv_lit_string(const uint32_t *ins, size_t wc, size_t from, char *buf, size_t cap)
{
    const char *p = (const char *)&ins[from];
    size_t avail = (wc - from) * 4;
    size_t n = 0;
    while (n < avail && n + 1 < cap && p[n])
        n++;
    memcpy(buf, p, n);
    buf[n] = 0;
}

void mgl_msl_mangle_reserved(spvc_compiler compiler, const uint32_t *words, size_t nwords)
{
    NameSet ns = {0};
    char name[512];

    if (!compiler || !words || nwords < 5 || words[0] != SpvMagicNumber)
        return;

    /* pass 1: collect every identifier of the module (collision-free mangling) */
    for (size_t pos = 5; pos < nwords; )
    {
        uint32_t wc = words[pos] >> 16;
        uint32_t op = words[pos] & 0xFFFF;
        if (wc == 0 || pos + wc > nwords)
            break;
        if (op == SpvOpName && wc >= 3)
        {
            spv_lit_string(words + pos, wc, 2, name, sizeof name);
            if (*name) nameset_add(&ns, name);
        }
        else if (op == SpvOpMemberName && wc >= 4)
        {
            spv_lit_string(words + pos, wc, 3, name, sizeof name);
            if (*name) nameset_add(&ns, name);
        }
        pos += wc;
    }

    /* pass 2: rename the reserved ones in the live compiler */
    for (size_t pos = 5; pos < nwords; )
    {
        uint32_t wc = words[pos] >> 16;
        uint32_t op = words[pos] & 0xFFFF;
        if (wc == 0 || pos + wc > nwords)
            break;

        if (op == SpvOpName && wc >= 3)
        {
            spv_lit_string(words + pos, wc, 2, name, sizeof name);
            if (*name && strncmp(name, "gl_", 3) && mgl_msl_is_reserved_name(name))
            {
                char mangled[560];
                for (int n = 0; n < 100; n++)
                {
                    if (n) snprintf(mangled, sizeof mangled, "%s_mgl%d", name, n);
                    else   snprintf(mangled, sizeof mangled, "%s_mgl", name);
                    if (!nameset_has(&ns, mangled))
                    {
                        nameset_add(&ns, mangled);
                        spvc_compiler_set_name(compiler, words[pos + 1], mangled);
                        break;
                    }
                }
            }
        }
        else if (op == SpvOpMemberName && wc >= 4)
        {
            spv_lit_string(words + pos, wc, 3, name, sizeof name);
            if (*name && strncmp(name, "gl_", 3) && mgl_msl_is_reserved_name(name))
            {
                char mangled[560];
                for (int n = 0; n < 100; n++)
                {
                    if (n) snprintf(mangled, sizeof mangled, "%s_mgl%d", name, n);
                    else   snprintf(mangled, sizeof mangled, "%s_mgl", name);
                    if (!nameset_has(&ns, mangled))
                    {
                        nameset_add(&ns, mangled);
                        spvc_compiler_set_member_name(compiler, words[pos + 1],
                                                      words[pos + 2], mangled);
                        break;
                    }
                }
            }
        }
        pos += wc;
    }

    nameset_free(&ns);
}

/* ===== BuiltInCullDistance stage OUTPUT flattening =========================
 *
 * See msl_reserved.h.  Shapes repaired in the emitted MSL of a vertex-function stage:
 *
 *   out.gl_CullDistance[3] = x;                          -> out.gl_CullDistance_3 = x;
 *   out.gl_CullDistance[i] = x[i];                       -> mgl_cull_out_fix[i] = x[i];
 *                                                          (+ copy-back before `return out;')
 *   out.gl_CullDistance_0 = _15._RESERVED_IDENTIFIER_FIXUP_gl_CullDistance[0];
 *                                                        -> out.gl_CullDistance_0 = out.gl_CullDistance_0;
 *                                                          (the tail's undeclared local is gone;
 *                                                          the self-assignment is a no-op)
 *
 * Constant-index accesses (stores AND reads) are redirected to the flattened
 * `NAME_i [[user(cull<i>)]]' struct members parsed out of the MSL; a dynamic index is
 * routed through a local array copied back to the members before `return out'. */
/* the sanitized member name SPIRV-Cross gives the local gl_CullDistance array in the
 * broken unflatten tail (`out.X_0 = _15._RESERVED_IDENTIFIER_FIXUP_gl_CullDistance[0];') */
#define CULL_FIXUP_PREFIX "._RESERVED_IDENTIFIER_FIXUP_gl_CullDistance["
#define CULL_FIXUP_LEN (sizeof CULL_FIXUP_PREFIX - 1)

/* the flattened-member table: element i -> "gl_CullDistance_i" style member name */
static int cull_parse_table(const char *msl, char elemname[][40])
{
    int nelem = 0;
    for (const char *p = strstr(msl, "[[user(cull"); p; p = strstr(p + 1, "[[user(cull"))
    {
        int idx;
        if (sscanf(p + strlen("[[user(cull"), "%d]]", &idx) != 1 || idx < 0 || idx >= 64)
            continue;
        /* the member name ends right where the ` [[user(cull' attribute begins */
        const char *a = p;
        while (a > msl && (a[-1] == ' ' || a[-1] == '\t')) a--;
        const char *n = a;
        while (n > msl && (isalnum((unsigned char)n[-1]) || n[-1] == '_')) n--;
        size_t nl = (size_t)(a - n);
        if (nl == 0 || nl >= 40)
            continue;
        if (idx + 1 > nelem) nelem = idx + 1;
        memcpy(elemname[idx], n, nl);
        elemname[idx][nl] = 0;
    }
    for (int i = 0; i < nelem; i++)
        if (!elemname[i][0])
            return 0;                /* incomplete table: caller must not touch the MSL */
    return nelem;
}

/* tiny formatted append for the rewriters below (no SB dependency here) */
static void cull_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap, ap2;
    int need;
    char *nb;

    va_start(ap, fmt);
    va_copy(ap2, ap);
    need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (need <= 0) { va_end(ap2); return; }
    if (*len + (size_t)need + 1 > *cap)
    {
        *cap = (*len + (size_t)need + 1) * 2;
        nb = realloc(*buf, *cap);
        if (!nb) { va_end(ap2); return; }
        *buf = nb;
    }
    vsnprintf(*buf + *len, *cap - *len, fmt, ap2);
    va_end(ap2);
    *len += (size_t)need;
}

/* BARE array arguments: when the stage main was renamed (geometry capture variant) and
 * its builtin gl_CullDistance global hoisted into a reference parameter, the generated
 * entry passes `out.gl_CullDistance' — an array the flattened output struct no longer
 * has.  Route the argument through a local array and copy back to the flattened members
 * after the call statement.  Returns a new string, or NULL when the shape is absent. */
static char *cull_fix_bare_arg(const char *msl, char elemname[][40], int nelem)
{
    const char *needle = "out.gl_CullDistance";
    const size_t nl = strlen(needle);
    const char *p;
    char *out = NULL;
    size_t done = 0;      /* bytes of the original consumed into the output */
    size_t scan = 0;      /* independent scan cursor (skips must not move `done') */

    while ((p = strstr(msl + scan, needle)) != NULL)
    {
        size_t k = (size_t)(p - msl);
        if (p[nl] == '[' || isalnum((unsigned char)p[nl]) || p[nl] == '_')
        {
            /* indexed store or a longer identifier: handled by the caller's main pass */
            scan = k + 1;
            continue;
        }

        /* the statement boundaries around the call */
        size_t stmt_s = k;
        while (stmt_s > done && msl[stmt_s-1] != ';' && msl[stmt_s-1] != '{' &&
               msl[stmt_s-1] != '}') stmt_s--;
        size_t stmt_e = k + nl, depth = 0;
        while (stmt_e < strlen(msl))
        {
            char c = msl[stmt_e];
            if (c == '(') depth++;
            else if (c == ')')
            {
                if (depth == 0) { stmt_e++; break; }     /* the call's own close */
                depth--;
            }
            else if (c == ';' && depth == 0) break;
            stmt_e++;
        }
        if (stmt_e >= strlen(msl)) { free(out); return NULL; }

        {
            char *b = NULL;
            size_t bl = 0, bc = 0;
            char *stmt, *q;
            cull_append(&b, &bl, &bc, "%.*s", (int)(stmt_s - done), msl + done);
            /* the statement with the argument swapped for the local */
            stmt = malloc(stmt_e - stmt_s + 64);
            if (!stmt) { free(out); free(b); return NULL; }
            memcpy(stmt, msl + stmt_s, stmt_e - stmt_s);
            stmt[stmt_e - stmt_s] = 0;
            q = strstr(stmt, needle);
            if (!q) { free(stmt); free(b); free(out); return NULL; }
            {
                size_t off = (size_t)(q - stmt);
                memmove(stmt + off + strlen("mgl_cull_arg_fix"), stmt + off + nl,
                        strlen(stmt) - off - nl + 1);
                memcpy(stmt + off, "mgl_cull_arg_fix", strlen("mgl_cull_arg_fix"));
            }
            {
                char zeros[8*4] = "0";
                for (int z = 1; z < (nelem > 8 ? 8 : nelem); z++)
                    strcpy(zeros + strlen(zeros), ",0");
                cull_append(&b, &bl, &bc, "    float mgl_cull_arg_fix[%d] = {%s};\n    ",
                            nelem > 8 ? 8 : nelem, zeros);
            }
            cull_append(&b, &bl, &bc, "%s", stmt);
            free(stmt);
            if (msl[stmt_e] == ';')
            {
                cull_append(&b, &bl, &bc, ";");
                stmt_e++;
            }
            for (int e = 0; e < nelem && e < 8; e++)
                cull_append(&b, &bl, &bc, "\n    out.%s = mgl_cull_arg_fix[%d];", elemname[e], e);
            cull_append(&b, &bl, &bc, "\n");

            free(out);
            out = b;                       /* keep the pieces written so far */
            done = stmt_e;
            scan = stmt_e;
        }
    }
    if (!out) return NULL;
    {
        /* the remainder after the last fixed statement */
        size_t ol = strlen(out);
        char *fin = malloc(ol + strlen(msl) - done + 1);
        if (!fin) { free(out); return NULL; }
        memcpy(fin, out, ol);
        strcpy(fin + ol, msl + done);
        free(out);
        return fin;
    }
}

/* the main pass: rewrite indexed accesses of the flattened members (constant indices
 * become member writes, dynamic indices copy through a local array) */
static char *mgl_msl_fix_cull_distance_indexed(const char *msl);

char *mgl_msl_fix_cull_distance(const char *msl)
{
    char elemname[64][40];          /* gl_CullDistance has at most 8 elements in GL 4.6 */
    int nelem;
    char *bare;

    if (!msl || !strstr(msl, "[[user(cull"))
        return NULL;
    /* 1. element i -> flattened member name (from the struct declarations) */
    nelem = cull_parse_table(msl, elemname);
    if (nelem == 0)
        return NULL;

    /* 1b. repair the hoisted-array call arguments first */
    bare = cull_fix_bare_arg(msl, elemname, nelem);
    if (bare)
    {
        char *indexed = mgl_msl_fix_cull_distance_indexed(bare);
        if (indexed) { free(bare); return indexed; }
        return bare;         /* the bare repair alone fixed the source */
    }
    return mgl_msl_fix_cull_distance_indexed(msl);
}

/* the main pass: rewrite indexed accesses of the flattened members (constant indices
 * become member writes, dynamic indices copy through a local array) */
static char *mgl_msl_fix_cull_distance_indexed(const char *msl)
{
    char elemname[64][40];
    int nelem = cull_parse_table(msl, elemname);

    if (nelem == 0)
        return NULL;

    const size_t len = strlen(msl);
    size_t cap = len + 4096;
    char *o = malloc(cap);
    if (!o)
        return NULL;
    size_t n = 0;
    int dynamic = 0, changed = 0;

    /* pick a local-array name that cannot collide with shader identifiers */
    const char *localname = "mgl_cull_out_fix";
    if (strstr(msl, localname))
        localname = "mgl_cull_out_fix_mgl";

    /* 2. rewrite both access shapes anywhere in the text */
    size_t i = 0;
    while (i < len)
    {
        size_t idx = 0;
        int have = 0;
        if (i + 20 <= len && !strncmp(msl + i, "out.gl_CullDistance[", 20))
        {
            idx = i + 20;
            have = 1;
        }
        else if (strncmp(msl + i, CULL_FIXUP_PREFIX, CULL_FIXUP_LEN))
        {
            o[n++] = msl[i++];
            if (n + 4096 > cap) { cap *= 2; o = realloc(o, cap); if (!o) return NULL; }
            continue;
        }
        else
        {
            /* `ID' + CULL_FIXUP_PREFIX: the leading identifier is already in `o' */
            idx = i + CULL_FIXUP_LEN;
            have = 2;
        }
        (void)have;

        /* index text until ']' */
        size_t j = idx, k = j;
        while (k < len && msl[k] != ']' && msl[k] != ';' && msl[k] != '\n') k++;
        int isconst = (k > j && k < len && msl[k] == ']');
        for (size_t d = j; d < k && isconst; d++)
            if (!isdigit((unsigned char)msl[d]))
                isconst = 0;
        if (isconst)
        {
            int e = atoi(msl + j);
            if (e >= 0 && e < nelem)
            {
                /* `out.gl_CullDistance[': nothing of the token is in `o' yet;
                 * `ID._RESERVED_...[': trim the already-copied ID first */
                if (have == 2)
                    while (n > 0 && (isalnum((unsigned char)o[n - 1]) || o[n - 1] == '_'))
                        n--;
                int rl = snprintf(o + n, 64, "out.%s", elemname[e]);
                n += (size_t)rl;
                i = k + 1;
                changed = 1;
                continue;
            }
        }
        /* dynamic index (or one past the array): route through the local array;
         * `out.gl_CullDistance[' and `ID._RESERVED_...[' both become `<local>[' */
        if (have == 2)
            while (n > 0 && (isalnum((unsigned char)o[n - 1]) || o[n - 1] == '_'))
                n--;
        memcpy(o + n, localname, strlen(localname));
        n += strlen(localname);
        i = idx - 1;               /* the '[' is copied from the source text */
        o[n++] = msl[i++];
        dynamic = 1;
        changed = 1;
    }
    o[n] = 0;

    if (!changed)
    {
        free(o);
        return NULL;
    }

    if (dynamic)
    {
        /* declare the local after the entry function's opening brace, copy it back
         * before the last `return out;' */
        const char *vfn = NULL;
        for (const char *p = o; *p; )
        {
            const char *eol = strchr(p, '\n');
            size_t fl = eol ? (size_t)(eol - p) : strlen(p);
            if (fl >= 7 && (!strncmp(p, "vertex ", 7) || !strncmp(p, "kernel ", 7)))
                vfn = p;
            p = eol ? eol + 1 : p + fl;
        }
        const char *brace = vfn ? strchr(vfn, '{') : NULL;
        if (!brace)
        {
            free(o);
            return NULL;
        }
        size_t at = (size_t)(brace - o) + 1;
        char *ret = NULL;
        for (char *p = o; (p = strstr(p, "return out;")); p += 5)
            ret = p;
        if (!ret)
        {
            free(o);
            return NULL;
        }
        char decl[96], copies[64 * 64];
        snprintf(decl, sizeof decl, "\n    float %s[%d] = {};", localname, nelem);
        size_t dlen = strlen(decl), clen = 0;
        for (int e = 0; e < nelem; e++)
            clen += (size_t)snprintf(copies + clen, sizeof copies - clen,
                                     "    out.%s = %s[%d];\n", elemname[e], localname, e);
        char *fin = malloc(n + dlen + clen + 8);
        if (!fin)
        {
            free(o);
            return NULL;
        }
        memcpy(fin, o, at);
        memcpy(fin + at, decl, dlen);
        memcpy(fin + at + dlen, o + at, (size_t)(ret - o) - at);
        memcpy(fin + at + dlen + (size_t)(ret - o) - at, copies, clen);
        memcpy(fin + at + dlen + (size_t)(ret - o) - at + clen, ret, n - (size_t)(ret - o) + 1);
        free(o);
        o = fin;
    }
    return o;
}

/* ===== uint-in-MSL / int-in-GL integer builtins =============================================
 *
 * SPIRV-Cross emits Metal's `gl_VertexID' / `gl_InstanceID' / `gl_BaseVertex' /
 * `gl_BaseInstance' / `gl_PrimitiveID' builtins for the SPIR-V BuiltIn of the same GL
 * name.  The MSL builtins are `uint' while GL(SL) declares them `int', and SPIRV-Cross
 * does not insert the narrowing cast: any std-lib call mixing the builtin with int
 * operands then fails to compile with an ambiguous overload —
 *
 *     out.dataOut = b.m[min(4, gl_VertexID)];   // error: call to 'min' is ambiguous
 *                                              // candidate min(int,int) / min(uint,uint)
 *                                              // (KHR-GL46.draw_indirect.advanced-twoPass-
 *                                              //  transformFeedback-*)
 *
 * mgl_msl_fix_int_builtins() wraps every *use* of these builtins in an explicit
 * `int(...)' cast, restoring GL's signed semantics (also for `/', `%', comparisons).
 * Declaration sites of same-named locals — MGL's own emulation kernels declare
 * `uint gl_VertexID = ...;' — and member accesses (`in.gl_VertexID') are left alone;
 * uses already inside a cast (`uint(gl_VertexID)') are skipped for the same reason.
 * SPIRV-Cross also hoists entry-point bodies into a static helper that receives the
 * builtin through a `thread uint& gl_VertexID' parameter; a wrapped use could not bind
 * to that reference, so such parameters are first rewritten in place (length-preserving)
 * to a by-value `int' parameter — the builtin is read-only, so passing a copy is exact.
 * Returns a malloc'd corrected copy, or NULL when nothing needed fixing. */
static const char *const msl_int_builtins[] = {
    "gl_VertexID", "gl_InstanceID", "gl_BaseVertex", "gl_BaseInstance", "gl_PrimitiveID",
};

static int msl_is_decl_type_token(const char *p, size_t len)
{
    static const char *const types[] = {
        "uint", "int", "ushort", "short", "ulong", "long", "uchar", "char", "bool",
        "float", "half", "size_t", "ptrdiff_t", "const", "static", "auto",
    };
    for (size_t i = 0; i < sizeof types / sizeof types[0]; i++)
        if (strlen(types[i]) == len && !memcmp(p, types[i], len))
            return 1;
    return 0;
}

/* match a builtin name at `p' (whole identifier) */
static size_t msl_builtin_at(const char *p, size_t avail)
{
    for (size_t k = 0; k < sizeof msl_int_builtins / sizeof msl_int_builtins[0]; k++)
    {
        size_t l = strlen(msl_int_builtins[k]);
        if (avail >= l && !memcmp(p, msl_int_builtins[k], l))
        {
            char next = p[l];
            if (!isalnum((unsigned char)next) && next != '_')
                return l;
        }
    }
    return 0;
}

/* rewrite `uint& NAME' function-parameter declarations to `int  NAME' (in place,
 * length-preserving: "uint" -> "int ", '&' -> ' '), so wrapped call arguments
 * (`int(gl_VertexID)') can bind.  Returns the number of rewrites. */
static int msl_fix_ref_params(char *buf)
{
    int n = 0;
    size_t len = strlen(buf);
    for (size_t i = 0; i < len; )
    {
        size_t l = msl_builtin_at(buf + i, len - i);
        if (!l)
        {
            i++;
            continue;
        }
        /* back over whitespace and '&' sigils */
        size_t j = i, amp = 0;
        while (j > 0 && (isspace((unsigned char)buf[j - 1]) || buf[j - 1] == '&'))
        {
            if (buf[j - 1] == '&')
                amp++;
            j--;
        }
        if (!amp || j == 0)
        {
            i += l;
            continue;
        }
        size_t e = j;
        while (e > 0 && (isalnum((unsigned char)buf[e - 1]) || buf[e - 1] == '_'))
            e--;
        /* forward: parameter end must be reached (',' or ')') */
        size_t k = i + l;
        while (k < len && isspace((unsigned char)buf[k]))
            k++;
        if (j - e == 4 && !memcmp(buf + e, "uint", 4) && k < len && (buf[k] == ',' || buf[k] == ')'))
        {
            memcpy(buf + e, "int ", 4);
            for (size_t a = e + 4; a < i; a++)
                if (buf[a] == '&')
                    buf[a] = ' ';
            n++;
        }
        i += l;
    }
    return n;
}

char *mgl_msl_fix_int_builtins(const char *msl)
{
    if (!msl)
        return NULL;

    bool any = false;
    for (size_t i = 0; i < sizeof msl_int_builtins / sizeof msl_int_builtins[0]; i++)
        if (strstr(msl, msl_int_builtins[i]))
        {
            any = true;
            break;
        }
    if (!any)
        return NULL;

    char *work = strdup(msl);
    if (!work)
        return NULL;
    int ref_fixed = msl_fix_ref_params(work);

    const size_t len = strlen(work);
    size_t cap = len + 1024;
    char *o = malloc(cap);
    if (!o)
    {
        free(work);
        return NULL;
    }
    size_t n = 0, i = 0;
    int changed = ref_fixed;

    while (i < len)
    {
        if (isalpha((unsigned char)work[i]) || work[i] == '_')
        {
            size_t s = i;
            while (i < len && (isalnum((unsigned char)work[i]) || work[i] == '_'))
                i++;
            size_t w = i - s;
            if (!msl_builtin_at(work + s, w))
            {
                if (n + w + 1 > cap)
                {
                    free(o);
                    free(work);
                    return NULL;
                }
                memcpy(o + n, work + s, w);
                n += w;
                continue;
            }
            /* previous token: a declaration (`uint gl_VertexID = ...', after the
             * ref-rewrite above also `int  gl_VertexID,'), an existing cast
             * (`uint(gl_VertexID)') or a member (`in.gl_VertexID') must not be wrapped */
            size_t j = s;
            while (j > 0 && (isspace((unsigned char)work[j - 1]) || work[j - 1] == '&' || work[j - 1] == '*'))
                j--;
            bool skip = (j > 0 && (work[j - 1] == '.' || work[j - 1] == '>'));
            if (!skip && j > 0 && (work[j - 1] == '&' || work[j - 1] == '*'))
                skip = true;    /* declaration sigil: `uint& name' / `int* name' */
            if (!skip && j > 0)
            {
                size_t e = j;
                while (e > 0 && (isalnum((unsigned char)work[e - 1]) || work[e - 1] == '_'))
                    e--;
                if (e < j && msl_is_decl_type_token(work + e, j - e))
                    skip = true;
            }
            if (skip)
            {
                if (n + w + 1 > cap)
                {
                    free(o);
                    free(work);
                    return NULL;
                }
                memcpy(o + n, work + s, w);
                n += w;
                continue;
            }
            if (n + w + 8 > cap)
            {
                free(o);
                free(work);
                return NULL;
            }
            o[n++] = 'i'; o[n++] = 'n'; o[n++] = 't'; o[n++] = '(';
            memcpy(o + n, work + s, w);
            n += w;
            o[n++] = ')';
            changed++;
        }
        else
        {
            if (n + 2 > cap)
            {
                free(o);
                free(work);
                return NULL;
            }
            o[n++] = work[i++];
        }
    }
    o[n] = 0;
    free(work);

    if (!changed)
    {
        free(o);
        return NULL;
    }
    return o;
}

/* See the header.  GLSL's interpolateAtOffset(x, o) takes the offset from the
 * pixel CENTER; Metal's interpolate_at_offset takes it from the pixel's
 * upper-left corner (verified on this GPU: the SPIRV-Cross bias below maps GL
 * (0,0) to 7/16 from the corner = 1/16 px short of the center), quantized to
 * the 1/16 pixel grid.  SPIRV-Cross emits `.interpolate_at_offset(o + 0.4375)'
 * (0.5 - 1/16, its closest stand-in for the center), which leaves a systematic
 * 1/16-pixel error — KHR-GL46.shader_multisample_interpolation.render.
 * interpolate_at_offset_check (offset = gl_SamplePosition - 0.5, tolerance
 * 1/255) rejects it.  GL offsets live in (-0.5, 0.5) and sample positions sit
 * on the 1/16 grid, so `+ 0.5' is the exact translation; the hardware accepts
 * the full [0, 1) corner range (measured up to 13/16).  Rewrite the bias in
 * every interpolate_at_offset call tail. */
char *mgl_msl_fix_interpolate_at_offset(const char *msl)
{
    static const char kCall[] = ".interpolate_at_offset(";
    static const char kBias[] = " + 0.4375)";
    static const char kFix[]  = " + 0.5)";

    if (!msl) return NULL;
    const char *p = msl;
    bool any = false;
    while ((p = strstr(p, kCall)) != NULL)
    {
        const char *open = p + strlen(kCall) - 1;      /* the argument '(' */
        int depth = 1;
        const char *q = open + 1;
        while (*q && depth > 0)
        {
            if (*q == '(') depth++;
            else if (*q == ')') depth--;
            q++;
        }
        if (depth == 0)
        {
            size_t blen = strlen(kBias);
            if ((size_t)(q - p) > blen && strncmp(q - blen, kBias, blen) == 0)
                any = true;
        }
        p = q;
    }
    if (!any)
        return NULL;

    /* one replacement pass */
    char *out = malloc(strlen(msl) + 1);
    if (!out) return NULL;
    const char *s = msl;
    char *d = out;
    p = msl;
    while ((p = strstr(p, kCall)) != NULL)
    {
        const char *open = p + strlen(kCall) - 1;
        int depth = 1;
        const char *q = open + 1;
        while (*q && depth > 0)
        {
            if (*q == '(') depth++;
            else if (*q == ')') depth--;
            q++;
        }
        size_t blen = strlen(kBias);
        if (depth == 0 && (size_t)(q - p) > blen && strncmp(q - blen, kBias, blen) == 0)
        {
            size_t head = (size_t)(q - s) - blen;
            memcpy(d, s, head);
            d += head;
            memcpy(d, kFix, strlen(kFix));
            d += strlen(kFix);
            s = q;
            p = q;
        }
        else
            p = q;
    }
    strcpy(d, s);
    return out;
}

/* See the header.  KHR-GL46.draw_indirect.advanced-twoPass-transformFeedback-*
 * compiles `min(4, gl_VertexID)' — the mixed int/uint overload is ambiguous. */
char *mgl_msl_fix_vertex_id_builtins(const char *msl)
{
    static const char *kFrom[]  = { "uint gl_VertexID [[vertex_id]]",
                                    "uint gl_InstanceID [[instance_id]]" };
    static const char *kParam[] = { "uint mgl_vid_ [[vertex_id]]",
                                    "uint mgl_iid_ [[instance_id]]" };
    static const char *kLocal[] = { "int gl_VertexID = int(mgl_vid_);\n    ",
                                    "int gl_InstanceID = int(mgl_iid_);\n    " };
    char *out = NULL;

    if (!msl) return NULL;
    for (unsigned k = 0; k < 2; k++)
    {
        const char *hit = strstr(out ? out : msl, kFrom[k]);
        if (!hit) continue;
        if (hit > (out ? out : msl))
        {
            char prev = (out ? out : msl)[(size_t)(hit - (out ? out : msl)) - 1];
            if (prev == '_' || isalnum((unsigned char)prev)) continue;
        }
        {
            const char *src = out ? out : msl;
            size_t off = (size_t)(hit - src), len = strlen(src);
            size_t fl = strlen(kFrom[k]), pl = strlen(kParam[k]), ll = strlen(kLocal[k]);
            char *ns = malloc(len - fl + pl + ll + 1);
            if (!ns) { free(out); return NULL; }
            memcpy(ns, src, off);
            memcpy(ns + off, kParam[k], pl);
            memcpy(ns + off + pl, src + off + fl, len - off - fl + 1);
            free(out);
            out = ns;

            /* insert the int local right after the entry function's opening brace;
               SPIRV-Cross writes single-line entry signatures: ")\n{" then the body */
            {
                char *brace = strstr(out + off, ")\n{");
                size_t ins = 0;
                if (brace) ins = (size_t)(brace - out) + 3;
                else
                {
                    char *brace2 = strstr(out + off, ") {");
                    if (brace2) ins = (size_t)(brace2 - out) + 3;
                }
                if (ins)
                {
                    size_t nlen = strlen(out);
                    char *ns2 = realloc(out, nlen + ll + 1);
                    if (ns2)
                    {
                        out = ns2;
                        memmove(out + ins + ll, out + ins, nlen - ins + 1);
                        memcpy(out + ins, kLocal[k], ll);
                    }
                }
            }
        }
    }
    return out;
}
