/*
 * msl_reserved.c — rename GLSL identifiers that collide with MSL reserved words.
 * See msl_reserved.h.  (KHR-GL46 cluster #1: 736 pixelstoragemodes cases et al.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
