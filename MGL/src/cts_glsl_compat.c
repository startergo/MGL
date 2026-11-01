/*
 * cts_glsl_compat.c — GLSL source compatibility rewriting before glslang.
 *
 * GL allows (and the Khronos CTS exercises) constructs that are legal GLSL but that
 * glslang refuses when generating SPIR-V.  These are rewritten to an equivalent program
 * that glslang accepts; nothing that changes GL semantics is touched.
 *
 *   layout(shared) / layout(packed) — GL-only block packings (KHR-GL46
 *     shaders.uniform_block.*.shared/packed, 500+ cases: "'shared' : not allowed when
 *     generating SPIR-V").  Both mean "implementation-defined, introspectable layout".
 *     Dropping the qualifier leaves the block with the default packing, which MGL lays
 *     out itself (std140-equivalent, offsets reported through introspection) — exactly
 *     what a GL implementation may do for shared/packed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include "cts_glsl_compat.h"

static int is_ident_char(int c) { return isalnum(c) || c == '_'; }

typedef struct {
    char *buf;
    size_t len, cap;
} Str;

static void str_push(Str *s, char c)
{
    if (s->len + 1 >= s->cap)
    {
        s->cap = s->cap ? s->cap * 2 : 4096;
        s->buf = realloc(s->buf, s->cap);
    }
    s->buf[s->len++] = c;
}

static void str_append(Str *s, const char *p, size_t n)
{
    for (size_t i = 0; i < n; i++) str_push(s, p[i]);
}

/* is `src[i..)` the standalone identifier `word`? */
static int at_word(const char *src, size_t len, size_t i, const char *word)
{
    size_t wl = strlen(word);
    if (i + wl > len || strncmp(src + i, word, wl)) return 0;
    return (i == 0 || !is_ident_char((unsigned char)src[i - 1])) &&
           (i + wl >= len || !is_ident_char((unsigned char)src[i + wl]));
}

/* skip whitespace and both comment forms */
static size_t skip_ws_comments(const char *src, size_t len, size_t i)
{
    for (;;)
    {
        while (i < len && isspace((unsigned char)src[i])) i++;
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '/')
        {
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(src[i] == '*' && src[i + 1] == '/')) i++;
            i = i + 1 < len ? i + 2 : len;
            continue;
        }
        return i;
    }
}

/* whitespace / comments WITHIN the directive's own line only: a whole-expression
 * macro test (`#ifndef GL_ARB_cull_distance') must have nothing else on the line,
 * and the general helper above crosses newlines (isspace eats '\n'), which made
 * the very next line's `#error' look like part of the directive and refused the
 * fold (KHR-GL46.cull_distance.functional's `#ifndef ... #error' guard). */
static size_t skip_ws_line(const char *src, size_t len, size_t i)
{
    for (;;)
    {
        while (i < len && (src[i] == ' ' || src[i] == '\t' || src[i] == '\r')) i++;
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '/')
        {
            while (i < len && src[i] != '\n') i++;
            return i;
        }
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '*')
        {
            size_t j = i + 2;
            while (j + 1 < len && !(src[j] == '*' && src[j + 1] == '/') && src[j] != '\n') j++;
            if (j + 1 < len && src[j] == '*' && src[j + 1] == '/') { i = j + 2; continue; }
            return j;      /* unterminated / newline-terminated: stop there */
        }
        return i;
    }
}

/* index of the ')' closing the '(' at `open` (len if unbalanced) */
static size_t match_paren(const char *src, size_t len, size_t open)
{
    int depth = 0;
    for (size_t i = open; i < len; i++)
    {
        if (src[i] == '(') depth++;
        else if (src[i] == ')')
        {
            if (--depth == 0) return i;
        }
    }
    return len;
}

/* a plain identifier entry "shared"/"packed" inside a layout list? */
static int entry_is_gl_packing(const char *src, size_t start, size_t end)
{
    while (start < end && isspace((unsigned char)src[start])) start++;
    while (end > start && isspace((unsigned char)src[end - 1])) end--;
    size_t n = start;
    while (n < end && is_ident_char((unsigned char)src[n])) n++;
    while (n < end && isspace((unsigned char)src[n])) n++;   /* no "= value" */
    if (n != end) return 0;
    size_t l = end - start;
    return (l == 6 && !strncmp(src + start, "shared", 6)) || (l == 6 && !strncmp(src + start, "packed", 6));
}

/*
 * Rewrite `src`: drop shared/packed from every layout(...) qualifier list.
 * Returns a malloc'd copy (caller frees), or NULL when nothing needed changing.
 */
static char *strip_gl_block_packings(const char *src)
{
    size_t len = strlen(src);
    Str out = {0};
    int changed = 0;
    size_t i = 0;

    while (i < len)
    {
        /* copy preprocessor lines verbatim */
        if (src[i] == '#')
        {
            size_t e = i;
            while (e < len && src[e] != '\n')
            {
                if (src[e] == '\\') { while (e < len && src[e] != '\n') e++; continue; }
                e++;
            }
            str_append(&out, src + i, e - i);
            i = e;
            continue;
        }
        if (at_word(src, len, i, "layout"))
        {
            size_t open = skip_ws_comments(src, len, i + 6);
            if (open < len && src[open] == '(')
            {
                size_t close = match_paren(src, len, open);
                if (close < len)
                {
                    /* split the list at top-level commas */
                    size_t nent = 0, kept = 0;
                    size_t e = open + 1;
                    while (e < close)
                    {
                        size_t start = e;
                        int depth = 0;
                        while (e < close && (depth || src[e] != ','))
                        {
                            if (src[e] == '(') depth++;
                            else if (src[e] == ')') depth--;
                            e++;
                        }
                        nent++;
                        if (!entry_is_gl_packing(src, start, e)) kept++;
                        if (e < close) e++;   /* skip the comma */
                    }
                    if (kept < nent)
                    {
                        changed = 1;
                        if (kept == 0)
                        {
                            /* nothing left: drop the whole layout(...) statement and one
                               whitespace run so "layout(shared) uniform" stays valid */
                            i = close + 1;
                            while (i < len && isspace((unsigned char)src[i]) && src[i] != '\n') i++;
                            continue;
                        }
                        str_append(&out, "layout(", 7);
                        e = open + 1;
                        int first = 1;
                        while (e < close)
                        {
                            size_t start = e;
                            int depth = 0;
                            while (e < close && (depth || src[e] != ','))
                            {
                                if (src[e] == '(') depth++;
                                else if (src[e] == ')') depth--;
                                e++;
                            }
                            if (!entry_is_gl_packing(src, start, e))
                            {
                                if (!first) str_append(&out, ", ", 2);
                                str_append(&out, src + start, e - start);
                                first = 0;
                            }
                            if (e < close) e++;
                        }
                        str_push(&out, ')');
                        i = close + 1;
                        continue;
                    }
                }
            }
        }
        /* ordinary character: copy, but jump over comments/strings so a "layout" inside
           one is not rewritten */
        if (i + 1 < len && src[i] == '/' && (src[i + 1] == '/' || src[i + 1] == '*'))
        {
            size_t j = skip_ws_comments(src, len, i);
            str_append(&out, src + i, j - i);
            i = j;
            continue;
        }
        if (src[i] == '"')
        {
            size_t j = i + 1;
            while (j < len && src[j] != '"')
            {
                if (src[j] == '\\' && j + 1 < len) j++;
                j++;
            }
            j = j < len ? j + 1 : len;
            str_append(&out, src + i, j - i);
            i = j;
            continue;
        }
        str_push(&out, src[i]);
        i++;
    }

    if (!changed)
    {
        free(out.buf);
        return NULL;
    }
    str_push(&out, 0);
    return out.buf;
}

/* GLSL 4.60 4.4.3: a global layout-qualifier declaration for the *buffer* interface
 * without a block ("layout(binding = 1) buffer;") is not valid OpenGL GLSL — it only
 * parses as Vulkan GLSL, which glslang implements.  A rewrite that keeps it would let
 * an ill-formed program link, so the declaration is rejected with a #error line
 * (KHR-GL46.shader_storage_buffer_object negative-glsl-compileTime; `uniform;` IS
 * legal and is left alone). */
static char *reject_buffer_layout_decl(const char *src)
{
    const char *q = src, *emitted = src;
    bool changed = false;
    Str out = { NULL, 0, 0 };

    if (!src) return NULL;

    while ((q = strstr(q, "buffer")) != NULL)
    {
        const char *b = q;                 /* token start */
        q += strlen("buffer");
        if (!(b == src || !is_ident_char((unsigned char)b[-1])) ||
            is_ident_char((unsigned char)*q))
            continue;                      /* part of an identifier */

        const char *pb = b;
        while (pb > src && isspace((unsigned char)pb[-1])) pb--;
        const char *fa = q;
        while (*fa && isspace((unsigned char)*fa)) fa++;

        /* "layout(...)buffer;" — layout qualifier list directly before, semicolon
         * right after (a block would name the interface between them).  A global
         * buffer-interface layout declaration is legal GLSL for the packing /
         * matrix-order qualifiers (GLSL 4.60 4.4.3; the CTS's "layout(shared,
         * row_major) buffer;"), but not for a binding point — reject only those. */
        if (!(pb > src && pb[-1] == ')' && *fa == ';'))
            continue;

        {
            const char *pe = pb - 1;             /* the ')' */
            int depth = 0;
            const char *po = pe;                 /* walk back to the matching '(' */

            while (po > src)
            {
                if (*po == ')') depth++;
                else if (*po == '(') { if (depth-- == 0) break; }
                po--;
            }
            if (po <= src || *po != '(') continue;

            /* does the qualifier list mention a binding point? */
            {
                const char *scan = po + 1;
                bool has_binding = false;

                for (; scan + 7 <= pe; scan++)
                {
                    if ((scan == po + 1 || !is_ident_char((unsigned char)scan[-1])) &&
                        strncmp(scan, "binding", 7) == 0 &&
                        !is_ident_char((unsigned char)scan[7]))
                    {
                        has_binding = true;
                        break;
                    }
                }
                if (!has_binding) continue;
            }
        }

        str_append(&out, emitted, (size_t)(b - emitted));
        str_append(&out, "buffer;", 7);
        /* the #error must be the first token on its own line */
        str_append(&out, "\n#error MGL: a buffer interface layout declaration"
                         " is not valid in OpenGL\n", 63);
        emitted = fa + 1;                  /* past the ';' */
        q = emitted;
        changed = true;
    }

    if (!changed)
    {
        free(out.buf);
        return NULL;
    }
    str_append(&out, emitted, strlen(emitted));
    str_push(&out, 0);
    return out.buf;
}

static bool gather_offset_is_literal(const char *e)
{
    /* blank constructor type keywords, then allow digits/signs/commas/hex only */
    char tmp[512];
    size_t n = strlen(e);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, e, n); tmp[n] = 0;
    for (char *p = tmp; (p = strstr(p, "ivec2")) != NULL; ) memset(p, ' ', 6);
    for (char *p = tmp; (p = strstr(p, "ivec3")) != NULL; ) memset(p, ' ', 6);
    for (char *p = tmp; *p; p++)
    {
        if (isalpha((unsigned char)*p))
        {
            /* 'x' of a hex literal only right after a digit */
            if ((*p == 'x' || *p == 'X') && p > tmp && isdigit((unsigned char)p[-1])) continue;
            return false;              /* a runtime identifier */
        }
    }
    return true;
}

/* the declared sampler dimension for a (simple) sampler expression */
typedef enum { GATHER_T2D, GATHER_T2DARRAY, GATHER_T2D_SHADOW, GATHER_T2DARRAY_SHADOW, GATHER_T_UNKNOWN } GatherDim;

static GatherDim gather_dim_of(const char *src, const char *expr, size_t len)
{
    char name[128];
    size_t n = len < sizeof name - 1 ? len : sizeof name - 1;
    memcpy(name, expr, n); name[n] = 0;
    /* strip a trailing array/struct selection: "uSampler[0]" -> "uSampler" */
    for (size_t k = 0; k < n; k++)
        if (name[k] == '[' || name[k] == '.') { name[k] = 0; break; }
    if (!name[0]) return GATHER_T_UNKNOWN;

    static const struct { const char *kw; GatherDim d; } dims[] = {
        { "sampler2DArrayShadow", GATHER_T2DARRAY_SHADOW },
        { "sampler2DShadow",      GATHER_T2D_SHADOW },
        { "sampler2DArray",       GATHER_T2DARRAY },
        { "sampler2D",            GATHER_T2D },
    };
    for (unsigned d = 0; d < sizeof dims / sizeof dims[0]; d++)
    {
        const char *p = strstr(src, dims[d].kw);
        while (p)
        {
            const char *q = p + strlen(dims[d].kw);
            while (q < src + strlen(src) && (*q == ' ' || *q == '\t' || *q == '\n')) q++;
            size_t kl = strlen(name);
            if (strncmp(q, name, kl) == 0 && !is_ident_char((unsigned char)q[kl]))
                return dims[d].d;          /* exact declaration "kw name" */
            p = strstr(p + 1, dims[d].kw);
        }
    }
    return GATHER_T_UNKNOWN;
}

static char *rewrite_gather_var_offset(const char *src)
{
    if (getenv("MGL_NO_GATHER_RW")) return NULL;
    if (!strstr(src, "textureGatherOffset") && !strstr(src, "textureGather(")) return NULL;
    size_t len = strlen(src);
    Str out = {0};
    size_t i = 0;
    bool changed = false;

    while (i < len)
    {
        bool plain_gather = at_word(src, len, i, "textureGather");
        if (!at_word(src, len, i, "textureGatherOffset") && !plain_gather)
        {
            str_push(&out, src[i++]);
            continue;
        }
        size_t w = i + strlen(plain_gather ? "textureGather" : "textureGatherOffset");
        size_t open = skip_ws_comments(src, len, w);
        size_t close = (open < len && src[open] == '(') ? match_paren(src, len, open) : len;
        if (close >= len)
        {
            str_append(&out, src + i, w - i);
            i = w;
            continue;
        }

        /* split the top-level arguments */
        char args[5][512];
        int nargs = 0;
        {
            size_t start = open + 1, depth = 0;
            for (size_t j = open + 1; j <= close; j++)
            {
                if (j < close && (src[j] == '(' || src[j] == '[')) depth++;
                else if (j < close && (src[j] == ')' || src[j] == ']')) depth--;
                if (j == close || (src[j] == ',' && depth == 0))
                {
                    if (nargs < 5)
                    {
                        size_t s = start, e = j;
                        while (s < e && isspace((unsigned char)src[s])) s++;
                        while (e > s && isspace((unsigned char)src[e - 1])) e--;
                        size_t n = e - s;
                        if (n >= sizeof args[0]) n = sizeof args[0] - 1;
                        memcpy(args[nargs], src + s, n);
                        args[nargs][n] = 0;
                    }
                    nargs++;
                    start = j + 1;
                }
            }
        }

        GatherDim dim = (nargs >= 3 && nargs <= 5) ? gather_dim_of(src, args[0], strlen(args[0])) : GATHER_T_UNKNOWN;
        bool shadow = (dim == GATHER_T2D_SHADOW || dim == GATHER_T2DARRAY_SHADOW);

        if (plain_gather)
        {
            /* plain textureGather: only the shadow-with-component form needs help:
             * glslang rejects textureGather(S, P, refZ, component) although GL 4.6
             * accepts and IGNORES the component for shadow samplers.  Everything
             * else (and any color form) compiles natively. */
            if (shadow && nargs == 4)
            {
                char buf[512];
                snprintf(buf, sizeof buf, "textureGather(%s, %s, %s)", args[0], args[1], args[2]);
                str_append(&out, buf, strlen(buf));
                changed = true;
            }
            else
                str_append(&out, src + i, close + 1 - i);
            i = close + 1;
            continue;
        }

        /* textureGatherOffset: (S,P,off[,comp]) | (S,P,ref,off) | (S,P,ref,off,comp) */
        int off_arg = shadow ? 3 : 2;
        if (dim == GATHER_T_UNKNOWN || off_arg >= nargs ||
            gather_offset_is_literal(args[off_arg]))
        {
            /* constant offsets reach Metal through SPIRV-Cross natively; leave */
            str_append(&out, src + i, close + 1 - i);
            i = close + 1;
            continue;
        }

        char buf[1400];
        char shift[640];
        /* the gather footprint depends only on floor(P*size - 0.5); snap to it and
         * add the integer offset (float-exact at texel centers, where a plain
         * coordinate shift can round one texel short) */
        if (dim == GATHER_T2D || dim == GATHER_T2D_SHADOW)
            snprintf(shift, sizeof shift,
                     "((floor((%s) * vec2(textureSize(%s, 0)) - vec2(0.5)) + vec2(%s) + vec2(0.5)) / vec2(textureSize(%s, 0)))",
                     args[1], args[0], args[off_arg], args[0]);
        else
            snprintf(shift, sizeof shift,
                     "vec3(((floor((%s).xy * vec2(textureSize(%s, 0).xy) - vec2(0.5)) + vec2(%s) + vec2(0.5)) / vec2(textureSize(%s, 0).xy)), (%s).z)",
                     args[1], args[0], args[off_arg], args[0], args[1]);
        if (shadow)
            /* keep the reference value after the shifted coordinate:
             * textureGather(S, P', refZ).  glslang rejects BOTH shadow forms that
             * carry a component selector (5-arg textureGatherOffset and 4-arg
             * textureGather) although GL 4.6 allows and ignores it there
             * (KHR-GL46.gpu_shader5 texture_gather_offset_depth_*), so the
             * component is dropped for shadow samplers. */
            snprintf(buf, sizeof buf, "textureGather(%s, %s, %s)", args[0], shift, args[2]);
        else if (nargs == 4)
            snprintf(buf, sizeof buf, "textureGather(%s, %s, %s)", args[0], shift, args[3]);
        else
            snprintf(buf, sizeof buf, "textureGather(%s, %s)", args[0], shift);
        str_append(&out, buf, strlen(buf));
        i = close + 1;
        changed = true;
    }
    if (!changed) { free(out.buf); return NULL; }
    str_push(&out, 0);          /* Str never terminates: produce a real C string */
    return out.buf;
}

static char *rewrite_gather_offsets(const char *src);

/* SSBO binding-point range checking (GLSL 4.60 4.4.5 / ARB_shading_language_420pack):
 * "layout(binding = N) buffer Block { ... } name[M];" is a compile-time error when
 * N >= GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS or when N + M exceeds it.  glslang
 * does not check that for OpenGL clients, so the declaration is rejected with a
 * #error line here (KHR-GL46.shader_storage_buffer_object negative-glsl-compileTime;
 * keep this in sync with MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS in get.c). */
#define MGL_CTS_MAX_SSBO_BINDINGS 8

/* walk backward over whitespace, comments and the block qualifier words that may
 * sit between "layout(...)" and the "buffer" keyword ("readonly buffer B {...}") */
static const char *back_over_qualifiers(const char *src, const char *p)
{
    static const char *const quals[] = { "coherent", "volatile", "restrict", "readonly",
                                         "writeonly", "std140", "std430", "shared", "packed", NULL };
    for (;;)
    {
        while (p > src && isspace((unsigned char)p[-1])) p--;
        if (p == src || p[-1] != ')') return p;
        bool cut = false;
        for (int i = 0; quals[i]; i++)
        {
            size_t n = strlen(quals[i]);
            if ((size_t)(p - src) >= n && strncmp(p - n, quals[i], n) == 0 &&
                (p - n == src || !is_ident_char((unsigned char)p[-n - 1])))
            {
                p -= n;
                cut = true;
                break;
            }
        }
        if (!cut) return p;
    }
}

static int layout_binding_value(const char *lp, const char *rp)
{
    /* the integer of "binding = N" inside [lp, rp) — -1 when absent */
    for (const char *q = lp; q + 7 < rp; q++)
    {
        if ((q == lp || !is_ident_char((unsigned char)q[-1])) &&
            strncmp(q, "binding", 7) == 0 && !is_ident_char((unsigned char)q[7]))
        {
            const char *e = q + 7;
            while (e < rp && isspace((unsigned char)*e)) e++;
            if (e < rp && *e == '=')
            {
                e++;
                while (e < rp && isspace((unsigned char)*e)) e++;
                if (e < rp && isdigit((unsigned char)*e))
                {
                    int v = 0;
                    while (e < rp && isdigit((unsigned char)*e)) v = v * 10 + (*e++ - '0');
                    return v;
                }
            }
            return -1;
        }
    }
    return -1;
}

/* index of the '}' matching src[open]=='{' (len when unbalanced) */
static size_t match_brace(const char *src, size_t len, size_t open)
{
    int depth = 0;
    for (size_t i = open; i < len; i++)
    {
        if (src[i] == '{') depth++;
        else if (src[i] == '}')
        {
            if (--depth == 0) return i;
        }
    }
    return len;
}

static char *reject_ssbo_binding_range(const char *src)
{
    const char *q = src, *emitted = src;
    bool changed = false;
    Str out = { NULL, 0, 0 };

    if (!src) return NULL;

    while ((q = strstr(q, "buffer")) != NULL)
    {
        const char *b = q;
        q += strlen("buffer");
        if (!(b == src || !is_ident_char((unsigned char)b[-1])) ||
            is_ident_char((unsigned char)*q))
            continue;                              /* inside an identifier */

        /* a block: "buffer Name {" (anonymous blocks are named too) */
        const char *p = q;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || !(isalpha((unsigned char)*p) || *p == '_')) continue;
        while (*p && is_ident_char((unsigned char)*p)) p++;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != '{') continue;

        /* the closing brace of the block, then the optional instance array size */
        long array_size = 0;
        const char *scan_from = q;                 /* where strstr() resumes */
        {
            size_t len = strlen(p);
            size_t close = match_brace(p, len, 0);
            if (close >= len) continue;
            const char *after = p + close + 1;
            const char *a = after;
            while (*a && isspace((unsigned char)*a)) a++;
            while (*a && is_ident_char((unsigned char)*a)) a++;   /* instance name */
            while (*a && isspace((unsigned char)*a)) a++;
            if (*a == '[')
            {
                const char *d = a + 1;
                while (*d && isspace((unsigned char)*d)) d++;
                if (isdigit((unsigned char)*d))
                {
                    array_size = strtol(d, (char **)&d, 10);
                    while (*d && isspace((unsigned char)*d)) d++;
                    if (*d != ']') array_size = 0;                /* not a plain literal */
                }
            }
            scan_from = after;                     /* past the declaration either way */
        }

        /* layout(...) list in front of the declaration */
        const char *pq = back_over_qualifiers(src, b);
        if (pq == src || pq[-1] != ')') continue;
        {
            int depth = 0;
            const char *po = pq - 1;                 /* at ')' */
            while (po > src)
            {
                if (*po == ')') depth++;
                else if (*po == '(' && --depth == 0) break;
                po--;
            }
            if (po <= src || *po != '(') continue;
            /* "layout" must directly precede the '(' */
            if (!((size_t)(po - src) >= 6 && strncmp(po - 6, "layout", 6) == 0 &&
                  (po - 6 == src || !is_ident_char((unsigned char)po[-7]))))
                continue;

            int binding = layout_binding_value(po + 1, pq - 1);
            if (binding < 0) continue;
            if (binding < MGL_CTS_MAX_SSBO_BINDINGS &&
                (array_size <= 0 || binding + array_size <= MGL_CTS_MAX_SSBO_BINDINGS))
                continue;

            /* reject: emit everything up to "layout" and raise a preprocessor error */
            str_append(&out, emitted, (size_t)(po - 6 - emitted));
            if (getenv("MGL_SSBO_BIND_DEBUG"))
                fprintf(stderr, "MGL: reject ssbo binding=%d array=%ld at off=%ld (emitted off=%ld)\n",
                        binding, array_size, (long)(po - 6 - src), (long)(emitted - src));
            str_append(&out, "\n#error MGL: ", 13);
            {
                char msg[128];
                int n = snprintf(msg, sizeof msg,
                                 "binding point %d%s%s is out of range (0..%d)\n", binding,
                                 array_size > 0 ? " + array size " : "",
                                 array_size > 0 ? "..." : "",
                                 MGL_CTS_MAX_SSBO_BINDINGS - 1);
                str_append(&out, msg, (size_t)n);
            }
            emitted = po - 6;
            q = scan_from;                          /* never walk back: no infinite loop */
            changed = true;
        }
    }

    if (!changed)
    {
        free(out.buf);
        return NULL;
    }
    str_append(&out, emitted, strlen(emitted));
    str_push(&out, 0);
    return out.buf;
}

/* "layout(std430) uniform Block { ... }" is not valid GLSL — std430 applies to
 * buffer blocks only (GLSL 4.60 4.4.5).  glslang accepts it with a warning, so the
 * declaration is rejected here (KHR-GL46.shader_storage_buffer_object
 * negative-glsl-compileTime). */
static bool layout_has_std430(const char *lp, const char *rp)
{
    for (const char *q = lp; q + 6 <= rp; q++)
    {
        if ((q == lp || !is_ident_char((unsigned char)q[-1])) &&
            strncmp(q, "std430", 6) == 0 && (q + 6 == rp || !is_ident_char((unsigned char)q[6])))
            return true;
    }
    return false;
}

static char *reject_std430_uniform_block(const char *src)
{
    const char *q = src, *emitted = src;
    bool changed = false;
    Str out = { NULL, 0, 0 };

    if (!src) return NULL;

    while ((q = strstr(q, "uniform")) != NULL)
    {
        const char *b = q;
        q += strlen("uniform");
        if (!(b == src || !is_ident_char((unsigned char)b[-1])) ||
            is_ident_char((unsigned char)*q))
            continue;                              /* inside an identifier */

        /* a block: "uniform Name {" */
        const char *p = q;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || !(isalpha((unsigned char)*p) || *p == '_')) continue;
        while (*p && is_ident_char((unsigned char)*p)) p++;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != '{') continue;

        const char *scan_from = q;
        {
            size_t len = strlen(p);
            size_t close = match_brace(p, len, 0);
            if (close < len) scan_from = p + close + 1;
        }

        const char *pq = back_over_qualifiers(src, b);
        if (pq == src || pq[-1] != ')') continue;
        {
            int depth = 0;
            const char *po = pq - 1;                 /* at ')' */
            while (po > src)
            {
                if (*po == ')') depth++;
                else if (*po == '(' && --depth == 0) break;
                po--;
            }
            if (po <= src || *po != '(') continue;
            if (!((size_t)(po - src) >= 6 && strncmp(po - 6, "layout", 6) == 0 &&
                  (po - 6 == src || !is_ident_char((unsigned char)po[-7]))))
                continue;
            if (!layout_has_std430(po + 1, pq - 1)) continue;

            str_append(&out, emitted, (size_t)(po - 6 - emitted));
            str_append(&out, "\n#error MGL: layout(std430) cannot be used with a uniform block\n", 61);
            emitted = po - 6;
            q = scan_from;
            changed = true;
        }
    }

    if (!changed)
    {
        free(out.buf);
        return NULL;
    }
    str_append(&out, emitted, strlen(emitted));
    str_push(&out, 0);
    return out.buf;
}

/* GLSL 4.20 (ARB_shading_language_420pack) allows 'invariant' on *input* declarations
 * and in any qualifier order; glslang rejects "'invariant': can only apply to an output".
 * Drop it from input declarations — for SPIR-V it is a no-op there anyway.  Handles both
 * inline qualifiers ("highp in layout(...) smooth invariant vec4 v;") and the standalone
 * form ("invariant v;"). */
static char *strip_invariant_on_input(const char *src)
{
    if (!src) return NULL;
    size_t len = strlen(src);
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, src, len + 1);
    bool changed = false;
    int depth = 0;
    for (size_t i = 0; i < len; )
    {
        char c = out[i];
        if (c == '/' && i + 1 < len && out[i + 1] == '/')
        {
            while (i < len && out[i] != '\n') i++;
            continue;
        }
        if (c == '/' && i + 1 < len && out[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(out[i] == '*' && out[i + 1] == '/')) i++;
            i = (i + 2 > len) ? len : i + 2;
            continue;
        }
        if (c == '"')
        {
            i++;
            while (i < len && out[i] != '"') i++;
            i++;
            continue;
        }
        if (c == '{') { depth++; i++; continue; }
        if (c == '}') { depth--; i++; continue; }
        if (depth >= 1 && at_word(out, len, i, "invariant"))
        {
            /* interface-block MEMBER: GLSL 4.2 (ARB_shading_language_420pack)
             * allows invariant on members of an INPUT block (ignored); glslang
             * rejects it there ("can only apply to an output").  The block
             * header decides: strip only inside `in' blocks — output members
             * keep the qualifier (KHR-GL46.shading_language_420pack.
             * qualifier_order_block) */
            size_t b = i;
            int d = depth;
            while (b > 0)
            {
                if (out[b] == '}') d++;
                else if (out[b] == '{')
                {
                    d--;
                    if (d == 0) break;
                }
                b--;
            }
            size_t hs = b;
            while (hs > 0 && out[hs - 1] != ';' && out[hs - 1] != '{' && out[hs - 1] != '}') hs--;
            bool is_in = false, is_out = false;
            for (size_t k = hs; k < b; k++)
            {
                if (at_word(out, len, k, "in")) is_in = true;
                else if (at_word(out, len, k, "out")) is_out = true;
                else if (at_word(out, len, k, "uniform") || at_word(out, len, k, "buffer")) is_out = true;
            }
            if (is_in && !is_out)
            {
                /* splice the token out */
                size_t e = i + strlen("invariant");
                while (e < len && (out[e] == ' ' || out[e] == '\t')) e++;
                memmove(out + i, out + e, len - e + 1);
                len -= e - i;
                changed = true;
                continue;
            }
            i += strlen("invariant");
            continue;
        }
        if (depth == 0 && at_word(out, len, i, "invariant"))
        {
            size_t ds = i;
            while (ds > 0 && out[ds - 1] != ';' && out[ds - 1] != '{' && out[ds - 1] != '}') ds--;
            size_t de = i;
            while (de < len && out[de] != ';') de++;
            bool has_out = false, has_type = false;
            for (size_t k = ds; k < de; k++)
            {
                if (at_word(out, len, k, "out")) has_out = true;
                else if (at_word(out, len, k, "vec4") || at_word(out, len, k, "vec3") ||
                         at_word(out, len, k, "vec2") || at_word(out, len, k, "float") ||
                         at_word(out, len, k, "int") || at_word(out, len, k, "uint") ||
                         at_word(out, len, k, "mat4") || at_word(out, len, k, "mat3") ||
                         at_word(out, len, k, "mat2")) has_type = true;
            }
            bool strip = false;
            if (has_type)
                strip = !has_out;   /* inputs, and direction-less leftovers (kernel-wrapped sources) */
            else
            {
                /* standalone "invariant NAME;": NAME's declaration direction decides */
                char name[128] = "";
                if (sscanf(out + i + 9, " %127[a-zA-Z0-9_]", name) == 1)
                {
                    size_t nl = strlen(name);
                    for (size_t k = 0; k + nl + 1 < len; k++)
                    {
                        if (at_word(src, len, k, name))
                        {
                            size_t ns = k;
                            while (ns > 0 && src[ns - 1] != ';' && src[ns - 1] != '{' && src[ns - 1] != '}') ns--;
                            size_t ne = k;
                            while (ne < len && src[ne] != ';') ne++;
                            bool pin = false, pout = false;
                            for (size_t q = ns; q < ne; q++)
                            {
                                if (at_word(src, len, q, "in")) pin = true;
                                else if (at_word(src, len, q, "out")) pout = true;
                            }
                            if (pin && !pout) { strip = true; break; }
                            if (pout) break;
                        }
                    }
                }
            }
            if (strip)
            {
                memset(out + i, ' ', 9);
                changed = true;
            }
            i += 9;
            continue;
        }
        i++;
    }
    if (!changed) { free(out); return NULL; }
    return out;
}

/* ================================================================= GL preprocessor strictness
 *
 * glslang is more permissive than the GLSL spec in four shapes the CTS negative
 * cases exercise (KHR-GL46.shaders.preprocessor.*):
 *   - a #version directive that is not the first significant statement
 *     (version_not_first_statement_1/2: glslang accepts "precision ...; #version 330")
 *   - a #version number that is not a real GLSL version (version_is_less: #version 329
 *     — MGL's legacy upgrade path rewrote every number below 330 to 330)
 *   - extra tokens after #if / #elif / #else / #endif (invalid_conditionals.*: the GLSL
 *     preprocessor requires the directive to end there; glslang ignores the tail)
 *   - the C stringize operator '#' inside a macro body (basic.stringification: GLSL
 *     supports ## but has no '#'; glslang only complains when the macro is expanded)
 * Each is scanned comment- and continuation-aware; on a hit a leading "#error" line is
 * injected so the compile fails with a spec-shaped message.  Anything the scan cannot
 * decide is left to glslang. */

typedef struct {
    const char *p;      /* token start */
    size_t n;           /* token length */
    int kind;           /* 0 punctuation, 1 identifier, 2 number */
} PpTok;

/* tokenize the tail of a conditional directive; returns the token count */
static int pp_tokenize(const char *p, const char *end, PpTok *tok, int max)
{
    int n = 0;
    while (p < end && n < max)
    {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { p++; continue; }
        if (c == '/' && p + 1 < end && p[1] == '*')          /* trailing block comment */
        {
            p += 2;
            while (p + 1 < end && !(p[0] == '*' && p[1] == '/')) p++;
            p = (p + 2 > end) ? end : p + 2;
            continue;
        }
        if (c == '/' && p + 1 < end && p[1] == '/') break;   /* line comment ends it */
        if (is_ident_char((unsigned char)c) && !isdigit((unsigned char)c))
        {
            const char *s = p;
            while (p < end && is_ident_char((unsigned char)*p)) p++;
            tok[n].p = s; tok[n].n = (size_t)(p - s); tok[n].kind = 1; n++;
            continue;
        }
        if (isdigit((unsigned char)c))
        {
            const char *s = p;
            while (p < end && (isalnum((unsigned char)*p) || *p == '.' || *p == '_')) p++;
            tok[n].p = s; tok[n].n = (size_t)(p - s); tok[n].kind = 2; n++;
            continue;
        }
        /* punctuation: two-character operators first */
        if (p + 1 < end && ((c == '&' && p[1] == '&') || (c == '|' && p[1] == '|') ||
            (c == '<' && p[1] == '<') || (c == '>' && p[1] == '>') ||
            (c == '<' && p[1] == '=') || (c == '>' && p[1] == '=') ||
            (c == '=' && p[1] == '=') || (c == '!' && p[1] == '=')))
        {
            tok[n].p = p; tok[n].n = 2; tok[n].kind = 0; n++; p += 2;
            continue;
        }
        tok[n].p = p; tok[n].n = 1; tok[n].kind = 0; n++; p++;
    }
    return n;
}

/* recursive-descent structure parse of a pp-expression; returns false when the token
 * stream is not a well-formed expression (then the caller stays silent and glslang
 * decides), true when parsed — *leftover reports tokens after the expression */
static bool pp_parse_expr(PpTok *t, int n, int *pos, bool *leftover);

static bool pp_parse_primary(PpTok *t, int n, int *pos)
{
    if (*pos >= n) return false;
    PpTok *k = &t[*pos];
    if (k->kind == 1)
    {
        (*pos)++;
        if (k->n == 7 && strncmp(k->p, "defined", 7) == 0)
        {
            if (*pos < n && t[*pos].kind == 1) { (*pos)++; return true; }
            if (*pos + 1 < n && t[*pos].n == 1 && t[*pos].p[0] == '(' &&
                t[*pos + 1].kind == 1 && *pos + 2 < n && t[*pos + 2].n == 1 && t[*pos + 2].p[0] == ')')
            { *pos += 3; return true; }
            return false;
        }
        /* a function-like macro reference in a pp-expression (expands before the
         * expression is evaluated): consume a balanced argument list */
        if (*pos < n && t[*pos].n == 1 && t[*pos].p[0] == '(')
        {
            int depth = 0;
            while (*pos < n)
            {
                if (t[*pos].n == 1 && t[*pos].p[0] == '(') depth++;
                else if (t[*pos].n == 1 && t[*pos].p[0] == ')') { depth--; (*pos)++; if (!depth) break; continue; }
                (*pos)++;
            }
            if (depth) return false;
        }
        return true;
    }
    if (k->kind == 2) { (*pos)++; return true; }
    if (k->n == 1 && k->p[0] == '(')
    {
        (*pos)++;
        bool dummy = false;
        if (!pp_parse_expr(t, n, pos, &dummy)) return false;
        if (*pos >= n || t[*pos].n != 1 || t[*pos].p[0] != ')') return false;
        (*pos)++;
        return true;
    }
    return false;
}

static bool pp_is_binop(PpTok *k)
{
    if (k->kind != 0) return false;
    if (k->n == 2) return true;                    /* && || << >> <= >= == != */
    switch (k->p[0])
    {
        case '*': case '/': case '%': case '+': case '-':
        case '<': case '>': case '&': case '^': case '|': case ',': return true;
        default: return false;
    }
}

static bool pp_parse_unary(PpTok *t, int n, int *pos)
{
    while (*pos < n && t[*pos].kind == 0 && t[*pos].n == 1 &&
           (t[*pos].p[0] == '!' || t[*pos].p[0] == '~' ||
            t[*pos].p[0] == '+' || t[*pos].p[0] == '-'))
        (*pos)++;
    return pp_parse_primary(t, n, pos);
}

static bool pp_parse_expr(PpTok *t, int n, int *pos, bool *leftover)
{
    if (!pp_parse_unary(t, n, pos)) return false;
    while (*pos < n && pp_is_binop(&t[*pos]))
    {
        (*pos)++;
        if (!pp_parse_unary(t, n, pos)) return false;
    }
    if (*pos < n && t[*pos].kind == 0 && t[*pos].n == 1 && t[*pos].p[0] == '?')
    {
        (*pos)++;
        if (!pp_parse_expr(t, n, pos, leftover)) return false;
        if (*pos >= n || t[*pos].n != 1 || t[*pos].p[0] != ':') return false;
        (*pos)++;
        if (!pp_parse_expr(t, n, pos, leftover)) return false;
    }
    *leftover = (*pos < n);
    return true;
}

/* the directive line starting at p (already past '#'), spanning to its end (handles
 * backslash-newline continuations and stops at a line comment); *word points at the
 * directive's first letter, *tail at the first character after it */
static const char *pp_directive_end(const char *p, const char *end, const char **word, const char **tail)
{
    while (p < end && (*p == ' ' || *p == '\t')) p++;
    const char *w = p;
    while (p < end && is_ident_char((unsigned char)*p)) p++;
    *word = w;
    *tail = p;
    const char *q = p;
    while (q < end && *q != '\n')
    {
        if (*q == '/' && q + 1 < end && q[1] == '/') break;
        if (*q == '\\' && q + 1 < end && q[1] == '\n') { q += 2; continue; }
        q++;
    }
    return q;
}

static bool pp_word_is(const char *w, size_t n, const char *word)
{
    return n == strlen(word) && strncmp(w, word, n) == 0;
}

/* a '#' begins a directive when only blanks separate it from the line start */
static bool pp_at_line_start(const char *src, const char *p)
{
    while (p > src && (p[-1] == ' ' || p[-1] == '\t')) p--;
    return p == src || p[-1] == '\n';
}

/* inject a leading "#error" line (keeps the original text for the info log) */
static char *pp_error_source(const char *src, const char *msg)
{
    size_t n = strlen(src);
    char *out = (char *)malloc(n + 64);
    if (!out) return NULL;
    snprintf(out, n + 64, "#error %s\n%s", msg, src);
    return out;
}

/* a leftover token ends a #if/#elif/#else/#endif directive; only reject when no macro
 * expansion could legally consume it: numbers never can, identifiers only when they are
 * not defined as macros (an object-like macro may expand to nothing, and a macro may
 * open a bracket the following punctuation closes — expressions.parenthesis_tricky) */
static bool pp_leftover_is_error(PpTok *t, const char **macros, int nmacros)
{
    if (t->kind == 2) return true;               /* number: never expandable */
    if (t->kind == 1)
    {
        for (int i = 0; i < nmacros; i++)
            if (strlen(macros[i]) == t->n && strncmp(macros[i], t->p, t->n) == 0)
                return false;                    /* a macro: expansion may consume it */
        return true;
    }
    return false;                                /* punctuation: a macro may bracket it */
}

static char *pp_error_source(const char *src, const char *msg);
static size_t mglc_token(const char *s, size_t len, size_t i, char *out, size_t outn);

/* ===== reserved words in positions glslang still accepts ====================
 *
 * KHR-GL46.CommonBugs.CommonBug_ReservedNames compiles ~3600 shaders that
 * declare an identifier with a GLSL keyword / reserved word and requires the
 * compile to be REJECTED.  glslang (desktop GLSL) rejects nearly all of them
 * but still accepts:
 *   - subroutine TYPE names and subroutine UNIFORM names (2003 of the cases):
 *     'subroutine void attribute(inout vec4);' / 'subroutine uniform void ...;'
 *   - 'invariant' as a declarator / function-argument name
 *   - 'subroutine' as a uniform/in/out BLOCK name
 * Those (and only those) are rejected here, spec-shaped, before any rewrite. */

static const char *const mglc_reserved_words[] = {
    "active", "asm", "atomic_uint", "attribute", "bool", "break", "buffer",
    "bvec2", "bvec3", "bvec4", "case", "cast", "centroid", "class",
    "coherent", "common", "const", "continue", "default", "discard",
    "dmat2", "dmat2x2", "dmat2x3", "dmat2x4", "dmat3", "dmat3x2", "dmat3x3",
    "dmat3x4", "dmat4", "dmat4x2", "dmat4x3", "dmat4x4", "do", "double",
    "dvec2", "dvec3", "dvec4", "else", "enum", "extern", "external", "false",
    "filter", "fixed", "flat", "float", "for", "fvec2", "fvec3", "fvec4",
    "goto", "half", "highp", "hvec2", "hvec3", "hvec4", "if", "iimage1D",
    "iimage1DArray", "iimage2D", "iimage2DArray", "iimage2DMS",
    "iimage2DMSArray", "iimage2DRect", "iimage3D", "iimageBuffer", "iimageCube",
    "iimageCubeArray", "image1D", "image1DArray", "image1DArrayShadow",
    "image1DShadow", "image2D", "image2DArray", "image2DArrayShadow",
    "image2DMS", "image2DMSArray", "image2DRect", "image2DShadow", "image3D",
    "imageBuffer", "imageCube", "imageCubeArray", "in", "inline", "inout",
    "input", "int", "interface", "invariant", "isampler1D", "isampler1DArray",
    "isampler2D", "isampler2DArray", "isampler2DMS", "isampler2DMSArray",
    "isampler2DRect", "isampler3D", "isamplerBuffer", "isamplerCube",
    "isamplerCubeArray", "ivec2", "ivec3", "ivec4", "layout", "long", "lowp",
    "mat2", "mat2x2", "mat2x3", "mat2x4", "mat3", "mat3x2", "mat3x3", "mat3x4",
    "mat4", "mat4x2", "mat4x3", "mat4x4", "mediump", "namespace", "noinline",
    "noperspective", "out", "output", "packed", "partition", "patch",
    "precise", "precision", "public", "readonly", "resource", "restrict",
    "return", "row_major", "sample", "sampler1D", "sampler1DArray",
    "sampler1DArrayShadow", "sampler1DShadow", "sampler2D", "sampler2DArray",
    "sampler2DArrayShadow", "sampler2DMS", "sampler2DMSArray", "sampler2DRect",
    "sampler2DRectShadow", "sampler2DShadow", "sampler3D", "sampler3DRect",
    "samplerBuffer", "samplerCube", "samplerCubeArray",
    "samplerCubeArrayShadow", "samplerCubeShadow", "shared", "short",
    "sizeof", "smooth", "static", "struct", "subroutine", "superp", "switch",
    "template", "this", "true", "typedef", "uimage1D", "uimage1DArray",
    "uimage2D", "uimage2DArray", "uimage2DMS", "uimage2DMSArray",
    "uimage2DRect", "uimage3D", "uimageBuffer", "uimageCube",
    "uimageCubeArray", "uint", "uniform", "union", "unsigned", "usampler1D",
    "usampler1DArray", "usampler2D", "usampler2DArray", "usampler2DMS",
    "usampler2DMSArray", "usampler2DRect", "usampler3D", "usamplerBuffer",
    "usamplerCube", "usamplerCubeArray", "using", "uvec2", "uvec3", "uvec4",
    "varying", "vec2", "vec3", "vec4", "void", "volatile", "while",
    "writeonly",
};

static int mglc_is_reserved(const char *w)
{
    int lo = 0, hi = (int)(sizeof mglc_reserved_words / sizeof mglc_reserved_words[0]) - 1;
    while (lo <= hi) {                       /* the table above is sorted */
        int mid = (lo + hi) / 2, c = strcmp(w, mglc_reserved_words[mid]);
        if (!c) return 1;
        if (c < 0) hi = mid - 1; else lo = mid + 1;
    }
    return 0;
}

char *mgl_reject_reserved_declarators(const char *src);   /* shaders.c: before the
                                                            * introspect subroutine rewrite */
static char *reject_reserved_declarators(const char *src)
{
    if (!src) return NULL;
    size_t len = strlen(src);
    size_t i = 0;
    while (i < len) {
        char tok[64];
        size_t j = i;
        /* identifier-sized token at a word boundary? */
        j = skip_ws_comments(src, len, j);
        if (j >= len) break;
        if (!is_ident_char((unsigned char)src[j])) { i = j + 1; continue; }
        size_t n = 0;
        while (j < len && is_ident_char((unsigned char)src[j]) && n + 1 < sizeof tok)
            tok[n++] = src[j++];
        tok[n] = 0;
        if (n >= sizeof tok - 1) { i = j; continue; }   /* overlong: not a word we track */
        i = j;

        /* (uniform|in|out|buffer) subroutine {  ->  block named 'subroutine' */
        if (!strcmp(tok, "uniform") || !strcmp(tok, "in") || !strcmp(tok, "out") ||
            !strcmp(tok, "buffer")) {
            char n2[64];
            size_t k = mglc_token(src, len, j, n2, sizeof n2);
            if (!strcmp(n2, "subroutine")) {
                size_t k2 = skip_ws_comments(src, len, k);
                if (k2 < len && src[k2] == '{')
                    return pp_error_source(src, "reserved word 'subroutine' "
                                                 "used as an interface block name");
            }
            continue;
        }
        /* 'struct <name> {' — the declared struct TYPE name (glslang accepts
         * 'struct invariant {...}' — GLSL 4.6 3.7; found by the full reserved
         * word x slot battery, tests/headless/test_reserved.c) */
        if (!strcmp(tok, "struct")) {
            char n2[64];
            mglc_token(src, len, j, n2, sizeof n2);
            if (n2[0] && mglc_is_reserved(n2))
                return pp_error_source(src, "reserved word used as a struct name");
            continue;
        }
        /* 'invariant' used as a declarator / argument NAME: the token AFTER it
         * ends the declaration (qualifier position is always followed by an
         * identifier) */
        if (!strcmp(tok, "invariant")) {
            size_t k = skip_ws_comments(src, len, j);
            if (k < len && (src[k] == ';' || src[k] == ',' || src[k] == '=' ||
                            src[k] == '[' || src[k] == ')'))
                return pp_error_source(src, "reserved word 'invariant' used as a name");
            continue;
        }
        /* subroutine declaration: [ '(' type-list ')' ] [uniform] type... NAME ( | ;
         * — the declared subroutine TYPE / subroutine UNIFORM name must not be
         * reserved (glslang accepts those; GLSL 4.6 3.7) */
        if (!strcmp(tok, "subroutine")) {
            size_t k = j;
            k = skip_ws_comments(src, len, k);
            if (k < len && src[k] == '(') {          /* subroutine (types...) decl */
                int d = 0;
                size_t e = k;
                while (e < len) {
                    if (src[e] == '(') d++;
                    else if (src[e] == ')') { d--; if (!d) { e++; break; } }
                    e++;
                }
                /* every identifier inside the list names a subroutine type */
                size_t q = k + 1;
                while (q < e) {
                    char nm[64];
                    size_t q2 = mglc_token(src, e, q, nm, sizeof nm);
                    if (nm[0] && mglc_is_reserved(nm))
                        return pp_error_source(src, "reserved word used as a "
                                                     "subroutine type name");
                    q = q2 > q ? q2 : q + 1;
                }
                k = skip_ws_comments(src, len, e);
            }
            /* optional 'uniform', then the declared name: the last identifier
             * before the terminating '(' (type decl) or ';' (uniform decl) */
            char n2[64];
            size_t k2 = mglc_token(src, len, k, n2, sizeof n2);
            if (!strcmp(n2, "uniform")) { k = k2; k2 = mglc_token(src, len, k, n2, sizeof n2); }
            char last[64] = "";
            for (;;) {
                size_t kw = skip_ws_comments(src, len, k2);
                if (kw >= len || src[kw] == '(' || src[kw] == ';') break;
                char n3[64];
                size_t k3 = mglc_token(src, len, kw, n3, sizeof n3);
                if (n3[0]) { snprintf(last, sizeof last, "%s", n3); k2 = k3; }
                else k2 = kw + 1;          /* punctuation noise */
            }
            if (last[0] && mglc_is_reserved(last))
                return pp_error_source(src, "reserved word used as a subroutine "
                                             "type or subroutine uniform name");
            continue;
        }
    }
    return NULL;
}

/* integer mix(genIType, genIType, genBType) exists in desktop GLSL only from 4.50
 * on (or with GL_EXT_shader_integer_mix); glslang's built-in mix() accepts the
 * integer forms at every version, so KHR-GL46.shaders.shader_integer_mix.
 * prototypes-negative (a #version 330 shader full of mix(ivec2(...), ...)) links
 * instead of failing.  Rejected when the declared version is older than 450 and
 * the extension is not enabled.  (Constructor-argument forms only - a mix() over
 * integer VARIABLES is not detectable without type information.) */
static char *reject_integer_mix(const char *src)
{
    if (!src) return NULL;
    size_t len = strlen(src);
    unsigned version = 0;
    int ext_enabled = 0, have_ext_directive = 0;
    for (size_t i = 0; i < len; ) {
        i = skip_ws_comments(src, len, i);
        if (i >= len) break;
        if (src[i] == '#') {
            char dir[32];
            size_t j = mglc_token(src, len, i + 1, dir, sizeof dir);
            if (!strcmp(dir, "version")) {
                char vs[16];
                mglc_token(src, len, j, vs, sizeof vs);
                version = (unsigned)atoi(vs);
            } else if (!strcmp(dir, "extension")) {
                char nm[96], beh[24];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                k = skip_ws_comments(src, len, k);
                if (k < len && src[k] == ':') k = mglc_token(src, len, k + 1, beh, sizeof beh);
                if (!strcmp(nm, "GL_EXT_shader_integer_mix")) {
                    have_ext_directive = 1;
                    if (!strcmp(beh, "enable") || !strcmp(beh, "require") || !strcmp(beh, "warn"))
                        ext_enabled = 1;
                }
            }
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        i++;
    }
    if (!version || version >= 450) return NULL;      /* core from 4.50 on */
    if (ext_enabled) return NULL;
    if (have_ext_directive) return NULL;              /* disable/warn'd: glslang decides */
    /* mix( <i|u|b>vecN ( ... */
    for (size_t i = 0; i < len; ) {
        i = skip_ws_comments(src, len, i);
        if (i >= len) break;
        if (is_ident_char((unsigned char)src[i])) {
            char tok[32];
            size_t n = 0, j = i;
            while (j < len && is_ident_char((unsigned char)src[j]) && n + 1 < sizeof tok)
                tok[n++] = src[j++];
            tok[n] = 0;
            if (!strcmp(tok, "mix")) {
                size_t k = skip_ws_comments(src, len, j);
                if (k < len && src[k] == '(') {
                    char t2[16];
                    size_t k2 = mglc_token(src, len, k + 1, t2, sizeof t2);
                    size_t k3 = skip_ws_comments(src, len, k2);
                    if ((!strncmp(t2, "ivec", 4) || !strncmp(t2, "uvec", 4) ||
                         !strncmp(t2, "bvec", 4)) && isdigit((unsigned char)t2[4]) &&
                        k3 < len && src[k3] == '(')
                        return pp_error_source(src, "integer mix() requires GLSL 4.50 or "
                                                    "GL_EXT_shader_integer_mix");
                }
            }
            i = j;
            continue;
        }
        i++;
    }
    return NULL;
}

/* ================================================================== GL_ARB_cull_distance
 *
 * Cull distance has been core since GLSL 4.50 and glslang carries no ARB
 * extension entry for it at all (Versions.h: "present for 4.5, but need
 * extension control over block members"), so a #version 150/420 source with
 * `#extension GL_ARB_cull_distance : require' — what KHR-GL46.cull_distance.*
 * compiles against a 4.6 context that advertises the extension — dies with
 * "extension not supported" before anything else.  The directive is dropped
 * and the version lifted to 4.50, where gl_CullDistance / gl_MaxCullDistances
 * / gl_MaxCombinedClipAndCullDistances are core built-ins.  A `#define
 * GL_ARB_cull_distance 1' stand-in is not allowed (GL_* is reserved), so the
 * whole-expression macro tests (`#ifndef GL_ARB_cull_distance',
 * `#if defined(GL_ARB_cull_distance)') are folded to #if 0 / #if 1 — the CTS
 * guards its error paths with exactly those.  `: disable' only drops the
 * directive.  Fires only when the directive is present, so unrelated sources
 * are untouched. */
/* `#ifndef GL_ARB_cull_distance' and friends: the macro test is rewritten to a
 * constant — GL_* names are reserved and may not be #defined (glslang rejects
 * `#define GL_ARB_cull_distance 1'), so the conditionals that test the
 * extension macro are folded instead: ifdef/defined(...) -> #if 1, ifndef/
 * !defined(...) -> #if 0.  Only a WHOLE-expression test is folded; anything
 * combined with && or || is left to the preprocessor.  Returns 0 = no match,
 * 1 = fold to true, -1 = fold to false. */
static int cull_conditional_fold(const char *src, size_t len, size_t j)
{
    /* j: first token position after the directive word */
    char t1[24];
    size_t k = mglc_token(src, len, j, t1, sizeof t1);
    int negate = 0;
    if (!strcmp(t1, "!"))
    {
        negate = 1;
        k = mglc_token(src, len, k, t1, sizeof t1);
    }
    if (!strcmp(t1, "defined"))
    {
        char t2[64];
        size_t k2 = mglc_token(src, len, k, t2, sizeof t2);
        if (!strcmp(t2, "("))
        {
            k2 = mglc_token(src, len, k2, t2, sizeof t2);
            size_t k3 = mglc_token(src, len, k2, t1, sizeof t1);   /* ')' */
            if (strcmp(t1, ")")) return 0;
            k = k3;
        }
        else
            k = k2;
        /* nothing but whitespace/comment may follow the whole-expression test */
        size_t r = skip_ws_line(src, len, k);
        if (r < len && src[r] != '\n') return 0;
        int is_cull = !strcmp(t2, "GL_ARB_cull_distance");
        return is_cull ? (negate ? -1 : 1) : 0;
    }
    return 0;
}

/* ===== GL_ARB_shader_group_vote / bare subgroup-vote spellings =====
 *
 * KHR-GL46.shader_group_vote specialises its compute shaders for a 4.6 core
 * context to the extension-less spellings of the vote built-ins
 * (allInvocations / anyInvocation / allInvocationsEqual) and adds no
 * #extension directive; glslang only knows them under
 * GL_ARB_shader_group_vote as the ARB-suffixed names (the KHR subgroup
 * extensions are Vulkan-only there, even when the #extension line is
 * accepted).  Rename the bare (and EXT-suffixed) calls to the ARB spellings
 * and enable the extension; MGL lowers OpGroupNonUniformAll/Any/AllEqual to
 * Metal simd_all/simd_any/spvSubgroupAllEqual (SPIRV-Cross). */
static char *rewrite_subgroup_vote_ext(const char *src)
{
    static const char *const names[3] = { "allInvocations", "anyInvocation", "allInvocationsEqual" };
    if (!src) return NULL;
    /* all three names carry "nvocation" */
    if (!strstr(src, "nvocation")) return NULL;

    /* does a call site exist (identifier-boundary name followed by '(')? */
    int calls = 0;
    for (int n = 0; n < 3 && !calls; n++)
    {
        const char *p = src;
        size_t nl = strlen(names[n]);
        while ((p = strstr(p, names[n])) != NULL)
        {
            size_t i = (size_t)(p - src);
            size_t after = i + nl;
            int pre_ok = (i == 0) || !is_ident_char((unsigned char)src[i - 1]);
            /* skip spaces/tabs between the name and the '(' */
            size_t j = after;
            while (src[j] == ' ' || src[j] == '\t') j++;
            if (pre_ok && (src[j] == '('))
            {
                /* already suffixed (allInvocationsARB / ...EXT)? */
                if (!(after + 3 <= strlen(src) &&
                      (strncmp(src + after, "ARB", 3) == 0 || strncmp(src + after, "EXT", 3) == 0)))
                    calls = 1;
            }
            p += nl;
        }
    }
    if (!calls) return NULL;

    /* is the ARB extension already enabled (or explicitly disabled)? */
    int have_arb = 0, arb_disabled = 0;
    {
        const char *p = src;
        while ((p = strstr(p, "#extension")) != NULL)
        {
            const char *e = p + 10;
            while (*e == ' ' || *e == '\t') e++;
            if (!strncmp(e, "GL_ARB_shader_group_vote", 24))
            {
                const char *b = e + 24;
                while (*b == ' ' || *b == '\t') b++;
                if (*b == ':')
                {
                    b++;
                    while (*b == ' ' || *b == '\t') b++;
                    if (!strncmp(b, "disable", 7)) arb_disabled = 1;
                    else have_arb = 1;   /* enable / require / warn */
                }
                else
                    have_arb = 1;
            }
            p += 10;
        }
    }

    if (arb_disabled) return NULL;      /* app turned it off: leave the (now
                                         * illegal) bare spellings alone */

    /* emit: rename the calls, inject the #extension after #version */
    size_t len = strlen(src);
    Str out = {0};
    size_t i = 0;
    size_t version_end = 0;              /* insert the directive after this */
    if (!have_arb && !arb_disabled)
    {
        const char *v = strstr(src, "#version");
        if (v)
        {
            const char *nl = strchr(v, '\n');
            version_end = nl ? (size_t)(nl - src) + 1 : len;
        }
    }
    while (i < len)
    {
        int replaced = 0;
        if (version_end && i == version_end && !have_arb)
        {
            static const char dir[] = "#extension GL_ARB_shader_group_vote : enable\n";
            str_append(&out, dir, sizeof dir - 1);
            version_end = 0;             /* once */
        }
        for (int n = 0; n < 3 && !replaced; n++)
        {
            const char *name = names[n];
            size_t nl = strlen(name);
            if (!strncmp(src + i, name, nl))
            {
                size_t after = i + nl;
                int pre_ok = (i == 0) || !is_ident_char((unsigned char)src[i - 1]);
                if (!pre_ok) break;
                if (after + 3 <= len &&
                    (strncmp(src + after, "ARB", 3) == 0 || strncmp(src + after, "EXT", 3) == 0))
                    break;               /* already suffixed */
                size_t j = after;
                while (j < len && (src[j] == ' ' || src[j] == '\t')) j++;
                if (j < len && src[j] == '(')
                {
                    str_append(&out, src + i, nl);
                    str_append(&out, "ARB", 3);
                    i = after;
                    replaced = 1;
                }
            }
        }
        if (!replaced)
        {
            str_push(&out, src[i]);
            i++;
        }
    }
    if (!out.buf) return NULL;
    str_push(&out, '\0');
    return out.buf;
}

/* GL_ARB_shader_ballot's desktop-GL surface needs two adjustments through
 * glslang -> SPIR-V -> SPIRV-Cross/MSL:
 *  - gl_SubGroupSizeARB stays a default-uniform (not the SubgroupSize builtin),
 *    which would read 0: replace the uses with an injected helper returning the
 *    Metal simdgroup width (32 on every Apple GPU; the TES pass folds it to 1).
 *  - the gl_SubGroup*MaskARB builtins are lowered by SPIRV-Cross to
 *    simd_ballot(true), which only sets the bits of ACTIVE lanes; a partially
 *    active wave (e.g. the 4 vertices of a quad in the vertex-stage compute
 *    kernel) then disagrees with the spec/CTS formula "all subgroup indices i
 *    with i OP gl_SubGroupInvocationARB".  The masks are pure functions of the
 *    invocation index, so replace them with helpers computing that formula
 *    (uint64_t flavor for the bare use, uvec4 flavor when a swizzle follows,
 *    since the desktop profile declares the masks uint64_t while SPIR-V/MSL
 *    expose them as uvec4 — glslang accepts both spellings of the builtin).
 * The TES pass (tessellation.c) rewrites gl_SubGroupInvocationARB to 0 inside
 * these helper bodies, giving the size-1 subgroup it must emulate. */
static char *rewrite_ballot_subgroup_size(const char *src)
{
    static const char name[] = "gl_SubGroupSizeARB";
    static const char *const masks[5] = { "gl_SubGroupEqMaskARB", "gl_SubGroupGeMaskARB",
                                          "gl_SubGroupGtMaskARB", "gl_SubGroupLeMaskARB",
                                          "gl_SubGroupLtMaskARB" };
    const size_t len = strlen(src);

    int uses = 0, mask_uses = 0;
    for (size_t i = 0; i + sizeof name - 1 <= len; i++)
        if (at_word(src, len, i, name)) { uses++; i += sizeof name - 2; }
    for (int k = 0; k < 5; k++)
    {
        size_t ml = strlen(masks[k]);
        for (size_t i = 0; i + ml <= len; i++)
            if (at_word(src, len, i, masks[k])) { mask_uses++; i += ml - 1; }
    }
    if (!uses && !mask_uses) return NULL;

    /* inject the helper after the last #extension directive (and after
     * #version), so it never precedes the preprocessor preamble */
    size_t insert = 0;
    {
        const char *v = strstr(src, "#version");
        if (v)
        {
            const char *nl = strchr(v, '\n');
            insert = nl ? (size_t)(nl - src) + 1 : len;
        }
        const char *p = src;
        while ((p = strstr(p, "#extension")) != NULL)
        {
            const char *nl = strchr(p, '\n');
            if (!nl) break;
            size_t at = (size_t)(nl - src) + 1;
            if (at > insert) insert = at;
            p = nl;
        }
    }

    static const char pre[] =
        "#extension GL_ARB_gpu_shader_int64 : enable\n"
        "uint mgl_ballot_subgroup_size() { return 32u; }"
        " /* MGL: simdgroup width, see rewrite_ballot_subgroup_size */\n"
        /* lanes(): bits [0, size) — the TES/VS pass folds the size call to 1u,
         * which clamps every mask to the emulated size-1 subgroup */
        "uint64_t mgl_ballot_lanes() { return (1ul << mgl_ballot_subgroup_size()) - 1ul; }\n"
        "uint64_t mgl_ballot_eq() { return mgl_ballot_lanes() & (1ul << gl_SubGroupInvocationARB); }\n"
        "uint64_t mgl_ballot_ge() { return mgl_ballot_lanes() & (~0ul << gl_SubGroupInvocationARB); }\n"
        "uint64_t mgl_ballot_gt() { return mgl_ballot_lanes() & (~0ul << (gl_SubGroupInvocationARB + 1u)); }\n"
        "uint64_t mgl_ballot_le() { return mgl_ballot_lanes() & ((2ul << gl_SubGroupInvocationARB) - 1ul); }\n"
        "uint64_t mgl_ballot_lt() { return mgl_ballot_lanes() & ((1ul << gl_SubGroupInvocationARB) - 1ul); }\n"
        "uvec4 mgl_ballot_v4(uint64_t m) { return uvec4(uint(m), uint(m >> 32ul), 0u, 0u); }\n"
        "uvec4 mgl_ballot_eq4() { return mgl_ballot_v4(mgl_ballot_eq()); }\n"
        "uvec4 mgl_ballot_ge4() { return mgl_ballot_v4(mgl_ballot_ge()); }\n"
        "uvec4 mgl_ballot_gt4() { return mgl_ballot_v4(mgl_ballot_gt()); }\n"
        "uvec4 mgl_ballot_le4() { return mgl_ballot_v4(mgl_ballot_le()); }\n"
        "uvec4 mgl_ballot_lt4() { return mgl_ballot_v4(mgl_ballot_lt()); }\n";
    static const char repl[] = "mgl_ballot_subgroup_size()";
    static const char *const maskrepl[5] = { "mgl_ballot_eq", "mgl_ballot_ge", "mgl_ballot_gt",
                                             "mgl_ballot_le", "mgl_ballot_lt" };

    Str out = {0};
    for (size_t i = 0; i <= len; i++)
    {
        if (insert && i == insert)
        {
            str_append(&out, pre, sizeof pre - 1);
            insert = 0;                     /* once */
        }
        int took = 0;
        if (i < len && at_word(src, len, i, name))
        {
            str_append(&out, repl, sizeof repl - 1);
            i += sizeof name - 2;
            took = 1;
        }
        else if (i < len)
        {
            for (int k = 0; k < 5 && !took; k++)
            {
                size_t ml = strlen(masks[k]);
                if (at_word(src, len, i, masks[k]))
                {
                    /* swizzled use (uvec4 flavor) or bare use (uint64_t flavor) */
                    size_t j = i + ml;
                    while (j < len && (src[j] == ' ' || src[j] == '\t' || src[j] == '\n')) j++;
                    int v4 = (j < len && src[j] == '.');
                    const char *r = v4 ? maskrepl[k] : maskrepl[k];   /* base name */
                    size_t rl = strlen(r);
                    str_append(&out, r, rl);
                    if (v4) { str_push(&out, '4'); }
                    str_push(&out, '(');
                    str_push(&out, ')');
                    i += ml - 1;
                    took = 1;
                }
            }
        }
        if (!took && i < len)
            str_push(&out, src[i]);
    }
    if (out.buf) str_push(&out, '\0');
    if (getenv("MGL_BALLOT_DUMP") && out.buf) fprintf(stderr, "[cts ballot source]\n%s\n[/cts ballot]\n", out.buf);
    return out.buf;
}

static char *rewrite_cull_distance_ext(const char *src)
{
    if (!src || !strstr(src, "GL_ARB_cull_distance"))
        return NULL;
    size_t len = strlen(src);
    int enable_seen = 0, disable_seen = 0;
    unsigned version = 0;
    int have_version = 0;

    /* pass 1: find the directives (only at a line start, after blanks) */
    for (size_t i = 0; i < len; )
    {
        while (i < len && (src[i] == ' ' || src[i] == '\t')) i++;   /* directive column */
        if (i >= len) break;
        if (src[i] == '#')
        {
            char dir[24];
            size_t j = mglc_token(src, len, i + 1, dir, sizeof dir);
            if (!strcmp(dir, "version"))
            {
                char vs[16];
                mglc_token(src, len, j, vs, sizeof vs);
                version = (unsigned)atoi(vs);
                have_version = 1;
            }
            else if (!strcmp(dir, "extension"))
            {
                char nm[64], beh[24];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                k = skip_ws_comments(src, len, k);
                if (k < len && src[k] == ':') k = mglc_token(src, len, k + 1, beh, sizeof beh);
                else beh[0] = 0;
                if (!strcmp(nm, "GL_ARB_cull_distance"))
                {
                    if (!strcmp(beh, "disable")) disable_seen = 1;
                    else enable_seen = 1;         /* require/enable/warn/broken */
                }
            }
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        while (i < len && src[i] != '\n') i++;
        if (i < len) i++;
    }
    if (!enable_seen && !disable_seen)
        return NULL;
    int bump = enable_seen && (!have_version || version < 450);

    /* pass 2: emit */
    Str out = {0};
    size_t i = 0;
    while (i < len)
    {
        size_t ls = i;
        while (i < len && (src[i] == ' ' || src[i] == '\t')) i++;
        if (i < len && src[i] == '#')
        {
            char dir[24];
            size_t j = mglc_token(src, len, i + 1, dir, sizeof dir);
            size_t e = j;
            while (e < len && src[e] != '\n') e++;
            if (!strcmp(dir, "version") && bump)
            {
                /* replace only the number token, keep `core'/`compatibility' */
                char vs[16];
                size_t k = mglc_token(src, len, j, vs, sizeof vs);
                str_append(&out, src + ls, j - ls);
                str_append(&out, " 450", 4);
                str_append(&out, src + k, e - k);
                i = e;
                continue;
            }
            if (!strcmp(dir, "extension"))
            {
                char nm[64], beh[24];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                size_t k2 = skip_ws_comments(src, len, k);
                if (k2 < len && src[k2] == ':') k2 = mglc_token(src, len, k2 + 1, beh, sizeof beh);
                else beh[0] = 0;
                if (!strcmp(nm, "GL_ARB_cull_distance") && beh[0])
                {
                    /* the directive disappears: the extension is core from
                     * 4.50 on (and the version is lifted), a macro definition
                     * is not allowed on the reserved GL_ prefix */
                    i = e;
                    continue;
                }
            }
            if (!strcmp(dir, "ifndef") || !strcmp(dir, "ifdef"))
            {
                char nm[64];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                size_t r = skip_ws_line(src, len, k);
                if (!strcmp(nm, "GL_ARB_cull_distance") && (r >= len || src[r] == '\n'))
                {
                    str_append(&out, src + ls, i - ls);
                    /* "#if 0"/"#if 1" are FIVE bytes: six would copy the literal's
                     * NUL into the buffer and truncate everything after it */
                    str_append(&out, !strcmp(dir, "ifndef") ? "#if 0" : "#if 1", 5);
                    i = e;
                    continue;
                }
            }
            if (!strcmp(dir, "if") || !strcmp(dir, "elif"))
            {
                int fold = cull_conditional_fold(src, len, j);
                if (fold)
                {
                    str_append(&out, src + ls, i - ls);
                    str_append(&out, dir[0] == 'e' ? "#elif " : "#if ", dir[0] == 'e' ? 6 : 4);
                    str_append(&out, fold > 0 ? "1" : "0", 1);
                    i = e;
                    continue;
                }            }
            str_append(&out, src + ls, e - ls);
            i = e;
            continue;
        }
        /* ordinary line: copy through the newline */
        while (i < len && src[i] != '\n') i++;
        if (i < len) i++;
        str_append(&out, src + ls, i - ls);
    }
    str_push(&out, '\0');
    return out.buf;
}

/* GL_ARB_texture_query_levels is core in GLSL 4.30 and absent from glslang's
 * desktop extension tables (unlike its sibling ARB_texture_query_lod), so a
 * `#version 400 core / #extension GL_ARB_texture_query_levels: require` shader
 * (KHR-GL46.texture_query_levels.*) fails to compile with "extension not
 * supported".  textureQueryLevels() parses from 4.30 on, so lift the version to
 * 430 and drop the directive — the same treatment rewrite_cull_distance_ext
 * gives GL_ARB_cull_distance (core 450).  #ifdef guards fold like the CTS
 * expects (MGL always exposes 4.6 core features). */
static char *rewrite_texture_query_levels_ext(const char *src)
{
    if (!src || !strstr(src, "GL_ARB_texture_query_levels"))
        return NULL;
    size_t len = strlen(src);
    int enable_seen = 0, disable_seen = 0;
    unsigned version = 0;
    int have_version = 0;

    for (size_t i = 0; i < len; )
    {
        while (i < len && (src[i] == ' ' || src[i] == '\t')) i++;
        if (i >= len) break;
        if (src[i] == '#')
        {
            char dir[24];
            size_t j = mglc_token(src, len, i + 1, dir, sizeof dir);
            if (!strcmp(dir, "version"))
            {
                char vs[16];
                mglc_token(src, len, j, vs, sizeof vs);
                version = (unsigned)atoi(vs);
                have_version = 1;
            }
            else if (!strcmp(dir, "extension"))
            {
                char nm[64], beh[24];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                k = skip_ws_comments(src, len, k);
                if (k < len && src[k] == ':') k = mglc_token(src, len, k + 1, beh, sizeof beh);
                else beh[0] = 0;
                if (!strcmp(nm, "GL_ARB_texture_query_levels"))
                {
                    if (!strcmp(beh, "disable")) disable_seen = 1;
                    else enable_seen = 1;
                }
            }
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        while (i < len && src[i] != '\n') i++;
        if (i < len) i++;
    }
    if (!enable_seen && !disable_seen)
        return NULL;
    int bump = enable_seen && (!have_version || version < 430);

    Str out = {0};
    size_t i = 0;
    while (i < len)
    {
        size_t ls = i;
        while (i < len && (src[i] == ' ' || src[i] == '\t')) i++;
        if (i < len && src[i] == '#')
        {
            char dir[24];
            size_t j = mglc_token(src, len, i + 1, dir, sizeof dir);
            size_t e = j;
            while (e < len && src[e] != '\n') e++;
            if (!strcmp(dir, "version") && bump)
            {
                /* replace only the number token, keep `core'/`compatibility' */
                char vs[16];
                size_t k = mglc_token(src, len, j, vs, sizeof vs);
                str_append(&out, src + ls, j - ls);
                str_append(&out, " 430", 4);
                str_append(&out, src + k, e - k);
                i = e;
                continue;
            }
            if (!strcmp(dir, "extension"))
            {
                char nm[64], beh[24];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                size_t k2 = skip_ws_comments(src, len, k);
                if (k2 < len && src[k2] == ':') k2 = mglc_token(src, len, k2 + 1, beh, sizeof beh);
                else beh[0] = 0;
                if (!strcmp(nm, "GL_ARB_texture_query_levels") && beh[0])
                {
                    /* the directive disappears: the built-in is core from 4.30
                     * on (and the version is lifted); a GL_-prefixed macro
                     * definition is not allowed */
                    i = e;
                    continue;
                }
            }
            if (!strcmp(dir, "ifndef") || !strcmp(dir, "ifdef"))
            {
                char nm[64];
                size_t k = mglc_token(src, len, j, nm, sizeof nm);
                size_t r = skip_ws_line(src, len, k);
                if (!strcmp(nm, "GL_ARB_texture_query_levels") && (r >= len || src[r] == '\n'))
                {
                    str_append(&out, src + ls, i - ls);
                    /* "#if 0"/"#if 1" are FIVE bytes: six would copy the
                     * literal's NUL and truncate everything after it */
                    str_append(&out, !strcmp(dir, "ifndef") ? "#if 0" : "#if 1", 5);
                    i = e;
                    continue;
                }
            }
            str_append(&out, src + ls, e - ls);
            i = e;
            continue;
        }
        while (i < len && src[i] != '\n') i++;
        if (i < len) i++;
        str_append(&out, src + ls, i - ls);
    }
    str_push(&out, '\0');
    return out.buf;
}

static char *reject_pp_strictness(const char *src)
{
    if (!src) return NULL;
    const char *p = src;
    const char *end = src + strlen(src);
    bool seen_significant = false;      /* any token before the first #version */
    bool have_version = false;

    /* defined macro names, for the leftover check above */
    const char *macros[256];
    int nmacros = 0;
    {
        const char *q = src;
        while (q < end && nmacros < 256)
        {
            if (q[0] == '#' && pp_at_line_start(src, q))
            {
                const char *w, *tail, *dend = pp_directive_end(q + 1, end, &w, &tail);
                size_t wn = (size_t)(tail - w);
                if (pp_word_is(w, wn, "define"))
                {
                    const char *b = tail;
                    while (b < dend && (*b == ' ' || *b == '\t')) b++;
                    const char *ns = b;
                    while (b < dend && is_ident_char((unsigned char)*b)) b++;
                    if (b > ns) macros[nmacros++] = ns;
                }
                q = (*dend == '\n') ? dend + 1 : dend;
                continue;
            }
            q++;
        }
    }

    while (p < end)
    {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { p++; continue; }
        if (c == '/' && p + 1 < end && p[1] == '*')
        {
            p += 2;
            while (p + 1 < end && !(p[0] == '*' && p[1] == '/')) p++;
            p = (p + 2 > end) ? end : p + 2;
            continue;
        }
        if (c == '/' && p + 1 < end && p[1] == '/')
        {
            while (p < end && *p != '\n') p++;
            continue;
        }
        if (c == '#' && pp_at_line_start(src, p))
        {
            const char *w, *tail, *dend;
            dend = pp_directive_end(p + 1, end, &w, &tail);
            size_t wn = (size_t)(tail - w);

            if (pp_word_is(w, wn, "version"))
            {
                if (seen_significant)
                    return pp_error_source(src, "#version must be the first statement of the shader");
                int v = 0;
                char prof[32] = { 0 };
                char line[128];
                size_t ln = (size_t)(dend - w);
                if (ln >= sizeof line) ln = sizeof line - 1;
                memcpy(line, w, ln); line[ln] = 0;
                sscanf(line, "version %d %31s", &v, prof);
                static const int known[] = { 100, 110, 120, 130, 140, 150, 300, 310, 320,
                                             330, 400, 410, 420, 430, 440, 450, 460 };
                bool ok = false;
                for (size_t i = 0; i < sizeof known / sizeof known[0]; i++)
                    if (known[i] == v) ok = true;
                if (!ok)
                    return pp_error_source(src, "unsupported #version number");
                have_version = true;
            }
            else if (pp_word_is(w, wn, "define"))
            {
                /* a single '#' (not '##') inside a macro body: GLSL has no stringize */
                const char *b = tail;
                while (b < dend && (*b == ' ' || *b == '\t')) b++;
                while (b < dend && is_ident_char((unsigned char)*b)) b++;   /* macro name */
                if (b < dend && *b == '(')                                  /* parameter list */
                {
                    int d = 0;
                    do { if (*b == '(') d++; else if (*b == ')') d--; b++; } while (b < dend && d > 0);
                }
                for (const char *q = b; q < dend; q++)
                    if (q[0] == '#' && !(q + 1 < dend && q[1] == '#') &&
                        !(q > src && q[-1] == '#'))
                        return pp_error_source(src, "'#' stringize operator is not supported in GLSL");
                seen_significant = true;
            }
            else if (pp_word_is(w, wn, "if") || pp_word_is(w, wn, "elif"))
            {
                PpTok tok[128];
                int n = pp_tokenize(tail, dend, tok, 128);
                int pos = 0;
                bool leftover = false;
                if (n > 0 && n < 128 && pp_parse_expr(tok, n, &pos, &leftover) &&
                    leftover && pp_leftover_is_error(&tok[pos], macros, nmacros))
                    return pp_error_source(src, "extra tokens after the preprocessor directive");
                seen_significant = true;
            }
            else if (pp_word_is(w, wn, "else") || pp_word_is(w, wn, "endif"))
            {
                PpTok tok[8];
                int n = pp_tokenize(tail, dend, tok, 8);
                if (n > 0 && pp_leftover_is_error(&tok[0], macros, nmacros))
                    return pp_error_source(src, "extra tokens after the preprocessor directive");
                seen_significant = true;
            }
            else
                seen_significant = true;

            p = (*dend == '\n') ? dend + 1 : dend;
            (void)have_version;
            continue;
        }
        seen_significant = true;
        /* skip an ordinary token */
        if (is_ident_char((unsigned char)c)) { while (p < end && is_ident_char((unsigned char)*p)) p++; }
        else if (c == '"') { p++; while (p < end && *p != '"') p++; p++; }
        else if (c == '\'') { p++; while (p < end && *p != '\'') p++; p++; }
        else p++;
    }
    return NULL;
}

/* ---------------------------------------------------- layout(binding=N) range check
 *
 * GLSL 4.60 4.4.5: for uniform blocks, buffer blocks, atomic counters and opaque
 * types, "any binding greater than or equal to the implementation-dependent limit
 * (for uniform blocks, GL_MAX_UNIFORM_BUFFER_BINDINGS, etc.) is a compile-time error".
 * glslang does not check buffer blocks against these limits, so KHR-GL46.layout_binding
 * binding_compilation_errors (which compiles a block with binding == the reported
 * maximum) links successfully.  Scan the declaration headers and inject a #error. */

typedef struct {
    const char *p;
    const char *end;
} BScan;

static void bs_ws(BScan *s)
{
    for (;;)
    {
        while (s->p < s->end && isspace((unsigned char)*s->p)) s->p++;
        if (s->p + 1 < s->end && s->p[0] == '/' && s->p[1] == '*')
        {
            s->p += 2;
            while (s->p + 1 < s->end && !(s->p[0] == '*' && s->p[1] == '/')) s->p++;
            s->p = s->p + 1 < s->end ? s->p + 2 : s->end;
            continue;
        }
        if (s->p + 1 < s->end && s->p[0] == '/' && s->p[1] == '/')
        {
            while (s->p < s->end && *s->p != '\n') s->p++;
            continue;
        }
        break;
    }
}

/* read the identifier at the cursor (NULL when none) */
static const char *bs_ident(BScan *s, size_t *len)
{
    bs_ws(s);
    if (s->p < s->end && (isalpha((unsigned char)*s->p) || *s->p == '_'))
    {
        const char *w = s->p;
        while (s->p < s->end && is_ident_char((unsigned char)*s->p)) s->p++;
        *len = (size_t)(s->p - w);
        return w;
    }
    return NULL;
}

static bool bs_word_is(const char *w, size_t n, const char *word)
{
    return w && n == strlen(word) && strncmp(w, word, n) == 0;
}

/* value of a literal (with unary +/- and integer constants) after "binding" '=' */
static bool bs_binding_value(BScan *s, unsigned *val)
{
    bs_ws(s);
    int sign = 1;
    if (s->p < s->end && (*s->p == '+' || *s->p == '-'))
    {
        if (*s->p == '-') sign = -1;
        s->p++;
    }
    bs_ws(s);
    if (s->p >= s->end || !isdigit((unsigned char)*s->p)) return false;
    unsigned long v = 0;
    const char *d = s->p;
    if (d + 1 < s->end && d[0] == '0' && (d[1] == 'x' || d[1] == 'X'))
    {
        s->p += 2;
        while (s->p < s->end && isxdigit((unsigned char)*s->p))
        {
            v = v * 16 + (unsigned)(strtoul(s->p, (char **)&s->p, 16));
            if (s->p >= s->end || s->p == d) break;
        }
    }
    else if (d + 1 < s->end && d[0] == '0')          /* octal (CTS 0112 = 74) */
    {
        s->p++;
        while (s->p < s->end && *s->p >= '0' && *s->p <= '7')
            v = v * 8 + (unsigned)(*s->p++ - '0');
    }
    else
    {
        while (s->p < s->end && isdigit((unsigned char)*s->p))
        {
            v = v * 10 + (unsigned)(*s->p - '0');
            s->p++;
        }
    }
    *val = (unsigned)(sign < 0 ? -(long)v : v);
    return true;
}

/* declared element count of the declaration being classified: the "[N]" after the
 * instance name — for a block instance array that name follows the block body, so skip
 * to the matching '}' first.  1 when the declaration is not arrayed. */
static unsigned bs_decl_count(const char *p, const char *end, char nxt)
{
    unsigned long n = 1;

    if (!p || p >= end) return 1;
    if (nxt == '{')
    {
        int depth = 0;
        while (p < end)
        {
            if (*p == '{') depth++;
            else if (*p == '}')
            {
                if (--depth == 0) { p++; break; }
            }
            p++;
        }
        while (p < end && (isspace((unsigned char)*p) || is_ident_char((unsigned char)*p))) p++;
    }
    else
    {
        while (p < end && is_ident_char((unsigned char)*p)) p++;
    }
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p < end && *p == '[')
    {
        p++;
        while (p < end && isspace((unsigned char)*p)) p++;
        if (p < end && isdigit((unsigned char)*p))
        {
            unsigned long v = 0;
            while (p < end && isdigit((unsigned char)*p)) v = v * 10 + (unsigned)(*p++ - '0');
            while (p < end && isspace((unsigned char)*p)) p++;
            if (p < end && *p == ']') n = v ? v : 1;
        }
    }
    return (unsigned)n;
}

static char *reject_binding_range(const char *src, const MGLCompatBindingLimits *lim)
{
    BScan s = { src, src + strlen(src) };

    while (s.p < s.end)
    {
        size_t n;
        const char *w = bs_ident(&s, &n);
        if (!w)
        {
            bs_ws(&s);
            if (s.p < s.end) s.p++;
            continue;
        }
        if (!bs_word_is(w, n, "layout")) continue;

        bs_ws(&s);
        if (s.p >= s.end || *s.p != '(') continue;
        s.p++;

        /* qualifiers inside layout(...) */
        unsigned binding = 0;
        bool have_binding = false;
        for (;;)
        {
            bs_ws(&s);
            if (s.p < s.end && *s.p == ')') { s.p++; break; }
            if (s.p < s.end && *s.p == ',') { s.p++; continue; }
            size_t qn;
            const char *q = bs_ident(&s, &qn);
            if (!q) { if (s.p < s.end) s.p++; continue; }   /* shared/... punctuation */
            if (bs_word_is(q, qn, "binding"))
            {
                bs_ws(&s);
                if (s.p < s.end && *s.p == '=')
                {
                    s.p++;
                    if (bs_binding_value(&s, &binding)) have_binding = true;
                }
            }
        }

        if (!have_binding) continue;

        /* classify the declaration that follows the layout(...) */
        bs_ws(&s);
        size_t a_n, b_n;
        const char *a = bs_ident(&s, &a_n);
        for (;;)
        {
            if (bs_word_is(a, a_n, "readonly") || bs_word_is(a, a_n, "writeonly") ||
                bs_word_is(a, a_n, "coherent") || bs_word_is(a, a_n, "restrict") ||
                bs_word_is(a, a_n, "volatile") || bs_word_is(a, a_n, "precisely"))
            {
                bs_ws(&s);
                a = bs_ident(&s, &a_n);            /* skip storage qualifiers */
                continue;
            }
            break;
        }
        const char *b = a ? bs_ident(&s, &b_n) : NULL;
        for (;;)
        {
            if (bs_word_is(b, b_n, "readonly") || bs_word_is(b, b_n, "writeonly") ||
                bs_word_is(b, b_n, "coherent") || bs_word_is(b, b_n, "restrict") ||
                bs_word_is(b, b_n, "volatile") || bs_word_is(b, b_n, "precisely") ||
                bs_word_is(b, b_n, "highp") || bs_word_is(b, b_n, "mediump") ||
                bs_word_is(b, b_n, "lowp"))
            {
                bs_ws(&s);
                b = bs_ident(&s, &b_n);            /* qualifiers after "uniform" */
                continue;
            }
            break;
        }
        bs_ws(&s);
        char nxt = s.p < s.end ? *s.p : 0;

        char err[96];
        unsigned limit = 0;
        bool per_element_binding = true;   /* block/sampler/image arrays take binding..binding+N-1 */
        if (bs_word_is(a, a_n, "uniform") && bs_word_is(b, b_n, "atomic_uint"))
        {
            limit = lim->atomic_bindings;
            /* atomic_uint ARRAYS stay in ONE buffer binding point at consecutive
             * offsets (GLSL 4.60 §4.4.2.3 / ARB_shader_atomic_counters) — only
             * binding itself has to be < GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS.
             * The r86 count rule here wrongly rejected KHR-GL46
             * layout_binding.atomic_uint_layout_binding_atomic_*_array_size
             * (binding=0, atomic_uint[N]) and geometry_shader.api.max_atomic_counters. */
            per_element_binding = false;
        }
        else if (bs_word_is(a, a_n, "uniform") && b && nxt == '{')
            limit = lim->ubo_bindings;                     /* uniform block */
        else if (bs_word_is(a, a_n, "uniform") && b_n >= 5 && strncmp(b, "image", 5) == 0)
            limit = lim->image_units;
        else if (bs_word_is(a, a_n, "uniform") && b_n >= 7 && strncmp(b, "sampler", 7) == 0)
            limit = lim->combined_texture_units;
        /* buffer blocks are handled by reject_ssbo_binding_range (array extents included) */

        if (limit)
        {
            /* an instance array occupies binding .. binding+N-1: the LAST index must stay
             * under the limit too (GLSL 4.60 4.4.5 / dEQP layout_binding.negative
             * *_binding_over_max_array, which picks binding = limit + 1 - N) */
            unsigned count = per_element_binding ? bs_decl_count(s.p, s.end, nxt) : 1;
            if (binding >= limit || (count > 1 && binding + count - 1 >= limit))
            {
                snprintf(err, sizeof err,
                         "layout(binding=%u) exceeds the implementation limit (%u)", binding, limit);
                return pp_error_source(src, err);
            }
        }
    }
    return NULL;
}

/* ------------------------------------------------ atomic counter layout rules
 *
 * GLSL 4.60 4.4.2.4 / ARB_shader_atomic_counters: an atomic counter declaration
 * with an offset qualifier must also carry a binding qualifier (CTS
 * ac-binding-examples 3 and 4: "layout(offset=8) uniform atomic_uint ac;" must
 * not compile), and once any declaration uses a binding qualifier every atomic
 * counter declaration in the shader must (ac-binding-examples 3).  glslang
 * accepts both shapes; scan the declaration headers and inject a #error. */

typedef struct {
    bool has_binding, has_offset;
} AtomicQuals;

/* read consecutive layout(...) groups preceding a declaration */
static bool bs_layout_quals(BScan *s, AtomicQuals *q)
{
    bool any = false;
    memset(q, 0, sizeof *q);
    for (;;)
    {
        BScan save = *s;
        bs_ws(s);
        size_t n;
        const char *w = bs_ident(s, &n);
        if (!bs_word_is(w, n, "layout")) { *s = save; return any; }
        bs_ws(s);
        if (s->p >= s->end || *s->p != '(') { *s = save; return any; }
        s->p++;
        any = true;
        for (;;)
        {
            bs_ws(s);
            if (s->p < s->end && *s->p == ')') { s->p++; break; }
            if (s->p < s->end && *s->p == ',') { s->p++; continue; }
            size_t qn;
            const char *qq = bs_ident(s, &qn);
            if (!qq) { if (s->p < s->end) s->p++; continue; }
            if (bs_word_is(qq, qn, "binding")) q->has_binding = true;
            if (bs_word_is(qq, qn, "offset")) q->has_offset = true;
        }
    }
}

static char *reject_atomic_layout_rules(const char *src)
{
    BScan s = { src, src + strlen(src) };
    bool any_binding = false;
    bool any_unbound = false;

    while (s.p < s.end)
    {
        BScan decl = s;
        AtomicQuals q;
        if (!bs_layout_quals(&s, &q)) { s = decl; }

        /* storage qualifiers may precede the declaration */
        for (;;)
        {
            bs_ws(&s);
            size_t st_n;
            BScan sv = s;
            const char *st = bs_ident(&s, &st_n);
            if (bs_word_is(st, st_n, "readonly") || bs_word_is(st, st_n, "writeonly") ||
                bs_word_is(st, st_n, "coherent") || bs_word_is(st, st_n, "restrict") ||
                bs_word_is(st, st_n, "volatile"))
                continue;
            s = sv;
            break;
        }

        bs_ws(&s);
        size_t u_n;
        const char *u = bs_ident(&s, &u_n);
        if (bs_word_is(u, u_n, "uniform"))
        {
            size_t t_n;
            const char *t = bs_ident(&s, &t_n);
            if (bs_word_is(t, t_n, "atomic_uint"))
            {
                if (q.has_offset && !q.has_binding)
                    return pp_error_source(src,
                        "atomic counter offset qualifier requires a binding qualifier");
                if (q.has_binding) any_binding = true;
                else any_unbound = true;
            }
        }
        if (s.p == decl.p)           /* no progress: skip a token */
        {
            if (!bs_ident(&s, &(size_t){0})) { bs_ws(&s); if (s.p < s.end) s.p++; }
        }
    }
    if (any_binding && any_unbound)
        return pp_error_source(src,
            "atomic counter declarations must all carry the binding qualifier when any does");
    return NULL;
}

char *mgl_cts_glsl_compat_generated(const char *src)
{
    /* the geometry / tess / xfb capture variants are MGL-authored GLSL: their
     * buffer blocks sit at the renumbered binding points MGL chose for its own
     * record layout (well past GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS), and they
     * are never handed to the app — the compile-time binding-range and std430
     * checks do not apply to them. */
    if (!src) return NULL;
    /* chain-owned copy currently installed in src0 (each pass returns a fresh
     * malloc'd string or NULL when it changed nothing) */
    char *owned0 = NULL;
    const char *src0 = src;
    char *r;
    r = rewrite_cull_distance_ext(src0);
    if (r) { owned0 = r; src0 = owned0; }
    /* GL_ARB_texture_query_levels: core in 4.30 but unknown to glslang's desktop
     * extension table — lift the version, drop the directive
     * (KHR-GL46.texture_query_levels.*) */
    r = rewrite_texture_query_levels_ext(src0);
    if (r) { free(owned0); owned0 = r; src0 = owned0; }
    /* the extension-less group-vote spellings must be renamed to the ARB ones in
     * the generated variants too (the geometry kernel embeds the user's source) */
    r = rewrite_subgroup_vote_ext(src0);
    if (r) { free(owned0); owned0 = r; src0 = owned0; }
    /* GL_ARB_shader_ballot must not reach SPIR-V's SubgroupSize/Subgroup*Mask
     * builtins in the generated variants: the geometry kernel compiles the user's
     * source as a compute kernel, and SPIRV-Cross hands `gl_SubgroupSize` to the
     * renamed main as a plain parameter it never declares in the kernel entry —
     * "use of undeclared identifier 'gl_SubgroupSize'" (KHR-GL46.shader_ballot_tests'
     * geometry-stage pipelines rendered nothing).  The injected helpers return the
     * simdgroup width and the spec's mask formula instead (see
     * rewrite_ballot_subgroup_size). */
    r = rewrite_ballot_subgroup_size(src0);
    if (r) { free(owned0); owned0 = r; src0 = owned0; }
    char *hoisted0 = mgl_hoist_struct_samplers(src0);
    if (hoisted0) { free(owned0); owned0 = hoisted0; src0 = owned0; }
    char *cur = strip_invariant_on_input(src0);
    if (!cur && src0 != src) cur = (char *)src0;
    char *next = strip_gl_block_packings(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    next = reject_buffer_layout_decl(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    char *g = rewrite_gather_offsets(cur ? cur : src);
    if (g)
    {
        if (cur && cur != src) free(cur);
        cur = g;
    }
    g = rewrite_gather_var_offset(cur ? cur : src);
    if (g)
    {
        if (cur && cur != src) free(cur);
        cur = g;
    }
    return (cur && cur != src) ? cur : NULL;
}

/* ===== hoist opaque members out of default-uniform-block structs =====
 *
 * GLSL (4.4+ / ARB-style struct uniforms) allows sampler and image members in
 * structs used for default-block uniforms:
 *     struct S { float a; vec3 b; sampler2D c; };
 *     uniform S s;                       ... texture(s.c, uv) ...
 * SPIRV-Cross's MSL backend cannot lower that: it emits `texture2d<float> c`
 * as a member of the [[buffer]] struct but never the paired `sampler cSmplr`
 * member ("no member named 'cSmplr' in 'S'"), and reflection never reports the
 * sampler, so no binding is assigned either.  GLSL-side rewrite before
 * compilation:
 *     struct S { float a; vec3 b; };
 *     uniform S s;
 *     uniform sampler2D s__c;            ... texture(s__c, uv) ...
 * with arrays preserved in order (`uniform S s[2]` + member c  ->
 * s_mglhoist_c[2], `s[1].c` -> `s_mglhoist_c[1]`; member array
 * `sampler2D c[2]` -> `s.c[1]` -> `s_mglhoist_c[1]`).  Nested structs flatten
 * into the name: `s.b.a` -> `s_mglhoist_b_mglhoist_a` (consecutive
 * underscores are reserved identifiers in GLSL).
 * glGetUniformLocation("s.c") / ("s[0].c") is translated back by
 * mgl_hoist_lookup_name() (uniforms.c) — the active-uniform ENUMERATION keeps
 * reporting the hoisted spelling (a small introspection infidelity). */

#define MGLC_MAX_STRUCTS 48
#define MGLC_MAX_MEMBERS 64

typedef struct {
    char mname[80];
    char mtype[80];      /* member base type: "sampler2D", "vec3", "S", ... */
    int  opaque;
    char mdims[64];      /* raw "[2][3]" suffix, "" when scalar */
    size_t decl_off, decl_end;   /* span of "type name dims ;" inside the body */
} MGLCMember;

typedef struct {
    char tname[80];
    MGLCMember m[MGLC_MAX_MEMBERS];
    int nm;
    int has_opaque;               /* an opaque member somewhere (recursively) */
    size_t body_off, body_end;    /* span between the struct's { and } */
} MGLCStruct;

static int mglc_is_opaque_type(const char *t)
{
    return !strncmp(t, "sampler", 7) || !strncmp(t, "isampler", 8) ||
           !strncmp(t, "usampler", 8) || !strncmp(t, "image", 5) ||
           !strncmp(t, "iimage", 6) || !strncmp(t, "uimage", 6);
}

static int mglc_is_qualifier(const char *t)
{
    return !strcmp(t, "lowp") || !strcmp(t, "mediump") || !strcmp(t, "highp") ||
           !strcmp(t, "precision") || !strcmp(t, "precise") || !strcmp(t, "invariant");
}

/* scan a token: identifier-ish or number (no separator handling) */
static size_t mglc_token(const char *s, size_t len, size_t i, char *out, size_t outn)
{
    i = skip_ws_comments(s, len, i);
    size_t n = 0;
    while (i < len && is_ident_char((unsigned char)s[i]) && n + 1 < outn)
        out[n++] = s[i++];
    out[n] = 0;
    return i;
}

/* copy "[..][..]" suffix starting at i (ws-skipped) into dims; returns end */
static size_t mglc_dims(const char *s, size_t len, size_t i, char *dims, size_t dimn)
{
    dims[0] = 0;
    size_t dn = 0;
    for (;;) {
        size_t j = skip_ws_comments(s, len, i);
        if (j >= len || s[j] != '[') return i;
        int depth = 0;
        size_t k = j;
        while (k < len) {
            if (s[k] == '[') depth++;
            else if (s[k] == ']') { depth--; if (!depth) { k++; break; } }
            k++;
        }
        if (k > len) return i;
        for (size_t x = j; x < k && dn + 1 < dimn; x++) dims[dn++] = s[x];
        dims[dn] = 0;
        i = k;
    }
}

static int mglc_struct_find(const MGLCStruct *st, int nst, const char *name)
{
    for (int i = 0; i < nst; i++) if (!strcmp(st[i].tname, name)) return i;
    return -1;
}

/* pass A: collect top-level `struct N { ... };` definitions */
static int mglc_collect_structs(const char *s, size_t len, MGLCStruct *st, int max)
{
    int nst = 0;
    int depth = 0;
    for (size_t i = 0; i < len; ) {
        i = skip_ws_comments(s, len, i);
        if (i >= len) break;
        if (s[i] == '{') { depth++; i++; continue; }
        if (s[i] == '}') { depth--; i++; continue; }
        if (depth == 0 && at_word(s, len, i, "struct")) {
            size_t j = mglc_token(s, len, i + 6, st[nst].tname, sizeof st[nst].tname);
            if (st[nst].tname[0] && j < len) {
                j = skip_ws_comments(s, len, j);
                if (j < len && s[j] == '{' && nst < max - 1) {
                    st[nst].nm = 0;
                    st[nst].has_opaque = 0;
                    st[nst].body_off = j + 1;
                    int d2 = 1;
                    size_t k = j + 1;
                    while (k < len && d2 > 0) {
                        if (s[k] == '{') d2++;
                        else if (s[k] == '}') { d2--; if (!d2) break; }
                        k++;
                    }
                    st[nst].body_end = k;
                    /* members: split the body on ';' */
                    size_t m = st[nst].body_off;
                    while (m < st[nst].body_end) {
                        size_t e = m;
                        int bd = 0;
                        while (e < st[nst].body_end) {
                            if (s[e] == '[' || s[e] == '(') bd++;
                            else if (s[e] == ']' || s[e] == ')') bd--;
                            else if (s[e] == ';' && bd == 0) break;
                            e++;
                        }
                        if (e > st[nst].body_end) e = st[nst].body_end;
                        /* parse one member: [qualifiers] TYPE NAME [dims] */
                        char tok[80];
                        size_t p = mglc_token(s, e, m, tok, sizeof tok);
                        while (tok[0] && mglc_is_qualifier(tok))
                            p = mglc_token(s, e, p, tok, sizeof tok);
                        char type[80] = "";
                        if (tok[0] && !mglc_is_qualifier(tok)) {
                            snprintf(type, sizeof type, "%s", tok);
                            size_t q = mglc_token(s, e, p, st[nst].m[st[nst].nm].mname,
                                                  sizeof st[nst].m[0].mname);
                            if (st[nst].m[st[nst].nm].mname[0] &&
                                st[nst].nm < MGLC_MAX_MEMBERS) {
                                MGLCMember *mm = &st[nst].m[st[nst].nm];
                                snprintf(mm->mtype, sizeof mm->mtype, "%s", type);
                                mm->opaque = mglc_is_opaque_type(type);
                                if (mm->opaque) st[nst].has_opaque = 1;
                                q = mglc_dims(s, e, q, mm->mdims, sizeof mm->mdims);
                                mm->decl_off = m;
                                mm->decl_end = e + 1 < st[nst].body_end ? e + 1 : e;
                                st[nst].nm++;
                            }
                        }
                        m = e + 1;
                    }
                    nst++;
                    i = k + 1;
                    /* skip the trailing ';' */
                    size_t j2 = skip_ws_comments(s, len, i);
                    if (j2 < len && s[j2] == ';') i = j2 + 1;
                    continue;
                }
            }
            i += 6;
            continue;
        }
        /* skip other tokens wholesale */
        while (i < len && !isspace((unsigned char)s[i]) && s[i] != '{' && s[i] != '}') i++;
    }
    /* propagate has_opaque through nested struct members */
    for (int pass = 0; pass < 4; pass++)
        for (int a = 0; a < nst; a++)
            for (int b = 0; b < st[a].nm; b++) {
                if (st[a].m[b].opaque) continue;
                int sub = mglc_struct_find(st, nst, st[a].m[b].mtype);
                if (sub >= 0 && st[sub].has_opaque) st[a].has_opaque = 1;
            }
    return nst;
}

/* Hoisted-name → GL-name table.  The hoist (a source-level rewrite: default-block
 * structs cannot hold samplers in MSL) renames "u.b.s" to "u_mglhoist_b_mglhoist_s",
 * which the SPIR-V-level introspection then reports verbatim — but GL requires the
 * dotted spelling in every program-interface query (dEQP PIQ *.sampler_struct).
 * Every hoisted declaration records its GL path here, so introspection can name the
 * resource back.  n_outer = how many leading array dimensions of the hoisted array
 * came from the uniform declaration (an array of structs enumerates per element,
 * GL 4.6 §7.3.1; member arrays are bottom-level arrays reported as name[0]). */
typedef struct MGLHoistRec {
    struct MGLHoistRec *next;
    char hoisted[160];   /* "u_mglhoist_b_mglhoist_s" */
    char gl[192];        /* "u.b.s" (no array suffix) */
    int  n_outer;
    char dims[64];       /* declared array dims, outer-first ("[3]", "[2][3]") */
    unsigned seq;        /* record order: the newest matching shape wins */
} MGLHoistRec;

static MGLHoistRec *g_hoist_recs;
static unsigned g_hoist_seq;

static void mgl_hoist_record(const char *hoisted, const char *prefix, const char *member,
                             int n_outer, const char *dims)
{
    MGLHoistRec *r;
    char gl[192];
    size_t o = 0;

    /* "u_mglhoist_b" + ".s" -> "u.b.s" */
    for (const char *p = prefix; *p && o + 2 < sizeof gl; p++) {
        if (!strncmp(p, "_mglhoist_", 10)) { gl[o++] = '.'; p += 9; }
        else gl[o++] = *p;
    }
    if (o + strlen(member) + 1 >= sizeof gl) return;
    gl[o] = 0;
    snprintf(gl + o, sizeof gl - o, ".%s", member);

    for (r = g_hoist_recs; r; r = r->next)
        if (!strcmp(r->hoisted, hoisted) && !strcmp(r->dims, dims) && r->n_outer == n_outer) return;
    r = malloc(sizeof *r);
    if (!r) return;
    snprintf(r->hoisted, sizeof r->hoisted, "%s", hoisted);
    snprintf(r->gl, sizeof r->gl, "%s", gl);
    snprintf(r->dims, sizeof r->dims, "%s", dims);
    r->n_outer = n_outer;
    r->seq = ++g_hoist_seq;
    r->next = g_hoist_recs;
    g_hoist_recs = r;
}

/* GL dotted spelling of a hoisted uniform name ("u_mglhoist_b_mglhoist_s" -> "u.b.s").
 * dims (may be NULL) is the array's shape as SPIR-V sees it, outer-first and
 * comma-separated ("3", "2,3"); among the spellings recorded for the name the newest
 * one whose shape matches wins, so a program linking an older shader still resolves
 * after a later shader re-used the same uniform name with a different shape.
 * *n_outer receives the number of leading dimensions that belong to the uniform
 * declaration (an array of structs enumerates per element, GL 4.6 §7.3.1).
 * Returns false when the name was not hoisted. */
bool mgl_hoist_gl_path2(const char *hoisted, const char *dims, char *out, size_t outn, int *n_outer)
{
    MGLHoistRec *best = NULL;
    char cand[64];

    for (MGLHoistRec *r = g_hoist_recs; r; r = r->next) {
        if (strcmp(r->hoisted, hoisted)) continue;
        if (!best || r->seq > best->seq) best = r;
        /* candidate shape: "[2][3]" -> "2,3" */
        cand[0] = 0;
        {
            size_t o = 0;
            for (const char *p = r->dims; *p; p++) {
                const char *e;
                if (*p != '[' || !(e = strchr(p, ']'))) continue;
                size_t n = (size_t)(e - p - 1);
                if (o) cand[o++] = ',';
                if (o + n + 1 >= sizeof cand) { cand[0] = 0; break; }
                memcpy(cand + o, p + 1, n);
                o += n;
                p = e;
            }
            cand[o] = 0;
        }
        if (dims && !strcmp(cand, dims) && (!best || r->seq > best->seq)) best = r;
    }
    if (!best) return false;
    snprintf(out, outn, "%s", best->gl);
    if (n_outer) *n_outer = best->n_outer;
    return true;
}

bool mgl_hoist_gl_path(const char *hoisted, char *out, size_t outn)
{
    return mgl_hoist_gl_path2(hoisted, NULL, out, outn, NULL);
}

/* number of leading dimensions of a hoisted array that belong to the uniform
 * declaration ("S u[2];" with member "sampler2D m[3]" -> 1), 0 when the uniform
 * itself is not an array; -1 when the name was not hoisted */
int mgl_hoist_outer_dims(const char *hoisted)
{
    for (MGLHoistRec *r = g_hoist_recs; r; r = r->next)
        if (!strcmp(r->hoisted, hoisted)) return r->n_outer;
    return -1;
}

/* enumerate the opaque member paths of struct si; appends "NAME__m1[..]__mk"
 * declarations to out, dims accumulating outer-uniform dims first */
static void mglc_enum_paths(const MGLCStruct *st, int nst, int si, const char *prefix,
                            const char *dims, int n_outer, Str *out)
{
    for (int k = 0; k < st[si].nm; k++) {
        const MGLCMember *m = &st[si].m[k];
        char np[256];
        snprintf(np, sizeof np, "%s_mglhoist_%s", prefix, m->mname);
        char nd[192];
        snprintf(nd, sizeof nd, "%s%s", dims, m->mdims);
        if (m->opaque) {
            /* remember the GL spelling of this hoisted uniform: introspection must
             * report "u.b.sampler" for the hoisted "u_mglhoist_b_mglhoist_sampler"
             * (GL 4.6 §7.3.1 dotted names), and n_outer leading array dimensions of
             * the hoisted array belong to the uniform declaration itself (an array
             * of structs expands per element, the member arrays do not).  See
             * mgl_hoist_gl_path2(). */
            mgl_hoist_record(np, prefix, m->mname, n_outer, nd);
            str_append(out, "\nuniform ", 9);
            str_append(out, m->mtype, strlen(m->mtype));
            str_append(out, " ", 1);
            str_append(out, np, strlen(np));
            str_append(out, nd, strlen(nd));
            str_append(out, ";", 1);
        } else {
            int sub = mglc_struct_find(st, nst, m->mtype);
            if (sub >= 0 && st[sub].has_opaque)
                mglc_enum_paths(st, nst, sub, np, nd, n_outer, out);
        }
    }
}

typedef struct {
    char var[80];
    char name[256];   /* hoisted base name (path joined by __) */
    int  npath;
    char path[8][80]; /* member identifiers along the path */
} MGLCHoist;

static int mglc_path_split(const char *name, MGLCHoist *h)  /* "s_mglhoist_b_mglhoist_a" */
{
    snprintf(h->name, sizeof h->name, "%s", name);
    h->npath = 0;
    const char *p = name;
    while (*p && h->npath < 8) {
        const char *q = strstr(p, "_mglhoist_");
        size_t n = q ? (size_t)(q - p) : strlen(p);
        if (!n || n >= sizeof h->path[0]) return 0;
        memcpy(h->path[h->npath], p, n);
        h->path[h->npath][n] = 0;
        h->npath++;
        p = q ? q + 10 : p + n;
    }
    snprintf(h->var, sizeof h->var, "%s", h->path[0]);
    return h->npath >= 2;
}

/* the whole hoisting pass; returns a malloc'd rewritten source or NULL */
char *mgl_hoist_struct_samplers(const char *src)
{
    if (!src) return NULL;
    size_t len = strlen(src);
    MGLCStruct st[MGLC_MAX_STRUCTS];
    int nst = mglc_collect_structs(src, len, st, MGLC_MAX_STRUCTS);
    if (getenv("MGL_HOIST_DEBUG")) {
        fprintf(stderr, "HOIST: nst=%d\n", nst);
        for (int q = 0; q < nst; q++) {
            fprintf(stderr, "HOIST: struct %s members=%d opaque=%d:", st[q].tname, st[q].nm, st[q].has_opaque);
            for (int w = 0; w < st[q].nm; w++)
                fprintf(stderr, " %s:%s%s", st[q].m[w].mtype, st[q].m[w].mname, st[q].m[w].mdims);
            fprintf(stderr, "\n");
        }
    }
    /* collect the hoisted paths of every default-block uniform whose struct
     * type owns an opaque member (they drive the use rewriter) */
    MGLCHoist hs[128];
    int nh = 0;
    int depth = 0;
    for (size_t i = 0; i < len; ) {
        i = skip_ws_comments(src, len, i);
        if (i >= len) break;
        if (src[i] == '{') { depth++; i++; continue; }
        if (src[i] == '}') { depth--; i++; continue; }
        if (depth == 0 && at_word(src, len, i, "uniform")) {
            char type[80], var[80], dims[64];
            size_t j = mglc_token(src, len, i + 7, type, sizeof type);
            j = skip_ws_comments(src, len, j);
            if (j < len && src[j] == '(') {
                size_t k = match_paren(src, len, j);
                if (k < len) { j = k + 1; j = mglc_token(src, len, j, type, sizeof type); }
            }
            j = mglc_token(src, len, j, var, sizeof var);
            if (type[0] && var[0]) {
                j = mglc_dims(src, len, j, dims, sizeof dims);
                int si = mglc_struct_find(st, nst, type);
                if (getenv("MGL_HOIST_DEBUG")) fprintf(stderr, "HOIST: uniform type=%s var=%s dims=%s si=%d\n", type, var, dims, si);
                if (si >= 0 && st[si].has_opaque) {
                    Str decls = {0};
                    int n_outer = 0;
                    for (const char *d = dims; *d; d++) n_outer += (*d == '[');
                    mglc_enum_paths(st, nst, si, var, dims, n_outer, &decls);
                    if (decls.buf) {
                        str_push(&decls, 0);
                        char *p = decls.buf;
                        while (nh < 128) {
                            char *q = strstr(p, "uniform ");
                            if (!q) break;
                            char *e = strchr(q, ';');
                            if (!e) break;
                            char nm[256] = "";
                            char *sp = strchr(q + 8, ' ');
                            if (sp && sp < e) {
                                size_t n = 0;
                                sp++;
                                while (*sp && *sp != '[' && *sp != ';' &&
                                       n + 1 < sizeof nm) nm[n++] = *sp++;
                                nm[n] = 0;
                            }
                            if (nm[0] && mglc_path_split(nm, &hs[nh])) nh++;
                            p = e + 1;
                        }
                        free(decls.buf);
                    }
                }
                while (j < len && src[j] != ';') j++;
                if (j < len) j++;
                i = j;
                continue;
            }
            i += 7;
            continue;
        }
        while (i < len && !isspace((unsigned char)src[i]) && src[i] != '{' && src[i] != '}') i++;
    }

    if (getenv("MGL_HOIST_DEBUG")) fprintf(stderr, "HOIST: nh=%d\n", nh);
    if (!nh) return NULL;

    /* single verbatim walk: uses of a hoisted path become the joined name, the
     * hoisted declarations follow their uniform, and opaque members are cut
     * from the struct bodies (spans refer to this same text) */
    Str out = {0};
    int wd = 0;
    char prev = 0;
    size_t i = 0;
    while (i < len) {
        if (src[i] == '/' && i + 1 < len && src[i + 1] == '/') {          /* line comment */
            while (i < len && src[i] != '\n') str_push(&out, src[i++]);
            continue;
        }
        if (src[i] == '/' && i + 1 < len && src[i + 1] == '*') {          /* block comment */
            str_push(&out, src[i++]); str_push(&out, src[i++]);
            while (i < len && !(src[i - 1] == '*' && src[i] == '/')) str_push(&out, src[i++]);
            if (i < len) str_push(&out, src[i++]);
            continue;
        }
        if (src[i] == '{') { wd++; str_push(&out, src[i++]); prev = '{'; continue; }
        if (src[i] == '}') { wd--; str_push(&out, src[i++]); prev = '}'; continue; }

        if (!is_ident_char((unsigned char)prev) && is_ident_char((unsigned char)src[i])) {
            /* hoisted use: VAR [idx] .m1 [idx] ... */
            char tok[80];
            size_t n = 0, j = i;
            while (j < len && is_ident_char((unsigned char)src[j]) && n + 1 < sizeof tok)
                tok[n++] = src[j++];
            tok[n] = 0;
            char idxs[6][128];
            char idents[8][80];
            int nidx = 0, nid = 0, matched = -1;
            size_t k = j;
            for (;;) {
                size_t k2 = skip_ws_comments(src, len, k);
                if (k2 < len && src[k2] == '[' && nidx < 6) {
                    int d = 0; size_t e = k2;
                    while (e < len) {
                        if (src[e] == '[') d++;
                        else if (src[e] == ']') { d--; if (!d) { e++; break; } }
                        e++;
                    }
                    if (e > len) break;
                    size_t nn = e - k2 - 2;
                    if (nn >= sizeof idxs[0]) break;
                    memcpy(idxs[nidx], src + k2 + 1, nn);
                    idxs[nidx][nn] = 0;
                    nidx++;
                    k = e;
                } else if (k2 < len && src[k2] == '.' && nid < 8) {
                    size_t k3 = mglc_token(src, len, k2 + 1, idents[nid], sizeof idents[0]);
                    if (!idents[nid][0]) break;
                    nid++;
                    k = k3;
                } else break;
            }
            for (int h = 0; h < nh && matched < 0; h++) {
                if (strcmp(hs[h].var, tok)) continue;
                if (hs[h].npath - 1 != nid) continue;
                int ok = 1;
                for (int q = 0; q < nid; q++)
                    if (strcmp(hs[h].path[q + 1], idents[q])) { ok = 0; break; }
                if (ok) matched = h;
            }
            if (matched >= 0) {
                str_append(&out, hs[matched].name, strlen(hs[matched].name));
                for (int q = 0; q < nidx; q++) {
                    char b2[140];
                    snprintf(b2, sizeof b2, "[%s]", idxs[q]);
                    str_append(&out, b2, strlen(b2));
                }
                i = k;
                prev = ']';
                continue;
            }
            /* struct definition of a hoisted type: filter the opaque members */
            if (wd == 0 && !strcmp(tok, "struct")) {
                char tn[80];
                size_t j2 = mglc_token(src, len, j, tn, sizeof tn);
                size_t j3 = skip_ws_comments(src, len, j2);
                if (tn[0] && j3 < len && src[j3] == '{') {
                    int si = mglc_struct_find(st, nst, tn);
                    int d2 = 1;
                    size_t k = j3 + 1;
                    while (k < len) {
                        if (src[k] == '{') d2++;
                        else if (src[k] == '}') { d2--; if (!d2) break; }
                        k++;
                    }
                    str_append(&out, src + i, j3 + 1 - i);        /* "struct N {" */
                    if (si >= 0 && st[si].has_opaque) {
                        int any_kept = 0;
                        for (int mi = 0; mi < st[si].nm; mi++)
                            if (!st[si].m[mi].opaque) any_kept = 1;
                        for (int mi = 0; mi < st[si].nm; mi++) {
                            const MGLCMember *m = &st[si].m[mi];
                            if (!m->opaque)
                                str_append(&out, src + m->decl_off, m->decl_end - m->decl_off);
                            else if (!any_kept && mi == 0) {
                                /* a struct whose members are all opaque still needs one
                                 * non-opaque member (GLSL forbids an empty struct).
                                 * highp is spelled out: the pad is injected into every
                                 * stage, and glslang's cross-stage symbol validation
                                 * rejects the same uniform with a different default
                                 * precision ("have precision conflict cross stage") —
                                 * an ES fragment shader defaults float to mediump.
                                 * (The old fixed length passed here was two bytes
                                 * short, which truncated the closing of the emitted
                                 * block comment, and the following '}' + ';' landed
                                 * inside it: every shader whose uniform struct held
                                 * only samplers or images then failed with "End of
                                 * input in comment" — dEQP PIQ *.sampler_struct.) */
                                static const char pad[] = "\n    highp float mgl_hoist_pad; /* opaque-only struct */\n";
                                str_append(&out, pad, sizeof(pad) - 1);
                            }
                        }
                    } else {
                        str_append(&out, src + j3 + 1, k - (j3 + 1));
                    }
                    str_push(&out, '}');
                    i = k + 1;
                    size_t j4 = skip_ws_comments(src, len, i);
                    if (j4 < len && src[j4] == ';') { str_push(&out, ';'); i = j4 + 1; }
                    prev = ';';
                    continue;
                }
            }
            /* uniform of a hoisted struct type: append the hoisted samplers */
            if (wd == 0 && !strcmp(tok, "uniform")) {
                char type[80], var[80], dims[64];
                size_t j2 = mglc_token(src, len, j, type, sizeof type);
                size_t j3 = skip_ws_comments(src, len, j2);
                if (j3 < len && src[j3] == '(') {
                    size_t k = match_paren(src, len, j3);
                    if (k < len) { j2 = mglc_token(src, len, k + 1, type, sizeof type); j3 = skip_ws_comments(src, len, j2); }
                }
                size_t j4 = mglc_token(src, len, j3, var, sizeof var);
                if (type[0] && var[0]) {
                    size_t j5 = mglc_dims(src, len, j4, dims, sizeof dims);
                    size_t j6 = j5;
                    while (j6 < len && src[j6] != ';') j6++;
                    if (j6 < len) j6++;
                    int si = mglc_struct_find(st, nst, type);
                    str_append(&out, src + i, j6 - i);            /* the declaration */
                    if (si >= 0 && st[si].has_opaque) {
                        Str decls = {0};
                        int n_outer = 0;
                        for (const char *d = dims; *d; d++) n_outer += (*d == '[');
                        mglc_enum_paths(st, nst, si, var, dims, n_outer, &decls);
                        if (decls.buf) {
                            str_push(&decls, 0);
                            str_append(&out, decls.buf, decls.len - 1);  /* no NUL inside */
                            free(decls.buf);
                        }
                    }
                    i = j6;
                    prev = ';';
                    continue;
                }
            }
            /* plain identifier */
            str_append(&out, src + i, j - i);
            prev = src[j - 1];
            i = j;
            continue;
        }
        str_push(&out, src[i]);
        prev = src[i];
        i++;
    }
    str_push(&out, 0);   /* Str is not NUL-terminated by construction */
    return out.buf;
}

/* "s.c" -> "s_mglhoist_c", "s[0].c" -> "s_mglhoist_c[0]",
 * "s.c[1]" -> "s_mglhoist_c[1]", "s.b.a" -> "s_mglhoist_b_mglhoist_a":
 * dotted uniform paths become the hoisted spelling with
 * the array indices moved behind the joined name.  Returns 1 when out holds a
 * different name than the input. */
int mgl_hoist_lookup_name(const char *name, char *out, size_t outn)
{
    if (!name || !strchr(name, '.') || !out || outn < 2) return 0;
    const char *p = name;
    char seg[96];
    size_t on = 0;
    int nseg = 0, first = 1;
    char idxbuf[128];
    size_t in_ = 0;
    idxbuf[0] = 0;
    while (*p) {
        /* identifier */
        size_t n = 0;
        while (*p && is_ident_char((unsigned char)*p) && n + 1 < sizeof seg)
            seg[n++] = *p++;
        seg[n] = 0;
        if (!n) return 0;
        if (!first) {
            const char sep[] = "_mglhoist_";
            if (on + sizeof sep - 1 >= outn) return 0;
            memcpy(out + on, sep, sizeof sep - 1);
            on += sizeof sep - 1;
        }
        if (on + n >= outn) return 0;
        memcpy(out + on, seg, n);
        on += n;
        first = 0;
        nseg++;
        /* indices belong after the full name: collect and continue */
        while (*p == '[') {
            const char *q = strchr(p, ']');
            if (!q) return 0;
            size_t l = (size_t)(q - p + 1);
            if (in_ + l >= sizeof idxbuf) return 0;
            memcpy(idxbuf + in_, p, l);
            in_ += l;
            idxbuf[in_] = 0;
            p = q + 1;
        }
        if (*p == '.') { p++; continue; }
        if (*p) return 0;
    }
    if (nseg < 2 || on + in_ + 1 > outn) return 0;
    memcpy(out + on, idxbuf, in_);
    out[on + in_] = 0;
    return strcmp(out, name) != 0;
}


char *mgl_cts_glsl_compat(const char *src)
{
    return mgl_cts_glsl_compat_limited(src, NULL);
}

char *mgl_reject_reserved_declarators(const char *src)
{
    return reject_reserved_declarators(src);
}

char *mgl_cts_glsl_compat_limited(const char *src, const MGLCompatBindingLimits *limits)
{
    if (!src) return NULL;
    /* spec-shaped rejections first (see reject_pp_strictness): once one fires the
     * compile must fail, and the other passes are pointless */
    char *strict = reject_pp_strictness(src);
    if (strict) return strict;
    {
        /* reserved words where glslang is still lenient (see the table above) */
        char *res = reject_reserved_declarators(src);
        if (res) return res;
        res = reject_integer_mix(src);
        if (res) return res;
    }
    {
        /* atomic counter layout rules are context-free: run them for generated
         * sources too */
        char *atomic = reject_atomic_layout_rules(src);
        if (atomic) return atomic;
    }
    if (limits)
    {
        char *bad = reject_binding_range(src, limits);
        if (bad) return bad;
    }
    char *owned = NULL;    /* chain-owned copy currently installed in `src' */
    {
        /* GL_ARB_cull_distance is core in GLSL 4.50 and absent from glslang's
         * extension table: lift the version and turn the directive into the
         * macro the CTS's #ifdef guards expect (see the pass above) */
        char *culled = rewrite_cull_distance_ext(src);
        if (culled) { owned = culled; src = owned; }
    }
    {
        /* GL_ARB_texture_query_levels: core in GLSL 4.30 but likewise absent
         * from glslang's desktop extension table (KHR-GL46.texture_query_levels
         * compiles `#version 400 core` + `#extension ...: require`): lift the
         * version, drop the directive */
        char *tql = rewrite_texture_query_levels_ext(src);
        if (tql) { free(owned); owned = tql; src = owned; }
    }
    {
        /* the extension-less spellings of the GL_ARB_shader_group_vote
         * built-ins (allInvocations/anyInvocation/allInvocationsEqual) are
         * not in glslang's desktop-GL tables: rename to the ARB names and
         * enable the extension (see rewrite_subgroup_vote_ext) */
        char *voted = rewrite_subgroup_vote_ext(src);
        if (voted) { free(owned); owned = voted; src = owned; }
    }
    {
        /* glslang's OpenGL profile keeps GL_ARB_shader_ballot's
         * gl_SubGroupSizeARB a default-uniform (not the SubgroupSize builtin),
         * which would read 0 through mgl_DefaultUniforms: give the uses an
         * injected helper returning the simdgroup width */
        char *ballot = rewrite_ballot_subgroup_size(src);
        if (ballot) { free(owned); owned = ballot; src = owned; }
    }
    {
        /* samplers inside default-block structs cannot be lowered to MSL
         * (see mgl_hoist_struct_samplers) - hoist them out first so every
         * later pass sees plain sampler uniforms */
        char *hoisted = mgl_hoist_struct_samplers(src);
        if (hoisted) { free(owned); owned = hoisted; src = owned; }
    }
    /* every step returns a malloc'd copy or NULL when it changed nothing, so the
     * chain keeps the latest non-NULL text and falls back to the original when
     * no rewrite fired at all — an owned copy must stay alive even then */
    char *cur = strip_invariant_on_input(src);
    if (!cur && owned) cur = owned;
    char *next = strip_gl_block_packings(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    next = reject_buffer_layout_decl(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    next = reject_ssbo_binding_range(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    next = reject_std430_uniform_block(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    char *g = rewrite_gather_offsets(cur ? cur : src);
    if (g)
    {
        free(cur);
        cur = g;
    }
    g = rewrite_gather_var_offset(cur ? cur : src);
    if (g)
    {
        free(cur);
        return g;
    }
    return cur;
}

/* ================================================================== textureGatherOffsets
 *
 * SPIRV-Cross's MSL backend silently DROPS the ConstOffsets image operand of
 * OpImageGather/OpImageDrefGather (Metal has no multi-offset gather), so a
 * textureGatherOffsets call would sample the wrong texels in every stage.
 * The call is therefore rewritten at the GLSL level into four
 * textureGatherOffset calls: per ARB_texture_gather, the i-th result channel
 * is the selected component of the texel at (base + offset[i]), which is
 * exactly channel (.w == (x0,y0)) of the single-offset gather of the block at
 * that offset.  textureGatherOffset keeps a (runtime) offset in SPIR-V, and
 * MGL's gather emulation for the vertex/compute stages already handles it.
 *
 * Limitations (same as the GLSL construct itself): the offsets argument must
 * be an array expression indexable with [0..3], the component argument is
 * inlined (so it must be constant, which GL requires anyway), and the sampler
 * and coordinate expressions are evaluated four times — fine for the uniform
 * and parameter expressions GL allows here. */

/* constructor matching the sampler's channel type: vec4 / ivec4 / uvec4.
 * The base identifier of the sampler argument is looked up in the source and
 * its declared type's leading 'i'/'u' decides; anything unrecognized is float.
 */
static const char *gather_offsets_ctor(const char *src, size_t len, const char *arg0, size_t a0len)
{
    char id[128];
    size_t idlen = 0;
    for (size_t i = 0; i < a0len && idlen + 1 < sizeof id; i++)
        if (is_ident_char((unsigned char)arg0[i])) id[idlen++] = arg0[i];
        else if (idlen) break;
    id[idlen] = 0;
    if (!idlen) return "vec4";

    /* find `<type> <id>' (uniform declarations and function parameters alike) */
    char pat[192];
    snprintf(pat, sizeof pat, " %s", id);
    const char *p = src;
    while ((p = strstr(p, pat)) != NULL)
    {
        if (p + strlen(pat) <= src + len &&
            (p[strlen(pat)] == ';' || p[strlen(pat)] == '[' || p[strlen(pat)] == ',' ||
             p[strlen(pat)] == '=' || p[strlen(pat)] == ')' || isspace((unsigned char)p[strlen(pat)])))
        {
            /* walk back over whitespace to the type token */
            const char *t = p;
            while (t > src && isspace((unsigned char)t[-1])) t--;
            const char *te = t;
            while (te > src && is_ident_char((unsigned char)te[-1])) te--;
            if (te < t && strncmp(te, "isampler", 8) == 0) return "ivec4";
            if (te < t && strncmp(te, "usampler", 8) == 0) return "uvec4";
            if (te < t && strncmp(te, "sampler", 7) == 0) return "vec4";
        }
        p += strlen(pat);
    }
    return "vec4";
}

/* try to resolve `arr' as a constant ivec2[4] initializer and return the four
 * offset expressions (allocated, NULL when not resolvable):
 *     const ivec2 arr[4] = ivec2[](E0, E1, E2, E3);
 *     const ivec2 arr[4] = ivec2[4](E0, E1, E2, E3);
 *     const ivec2 arr[4] = { E0, E1, E2, E3 };      (GLSL 4.20)
 * SPIRV-Cross only forwards compile-time-constant gather offsets to MSL, so the
 * array elements must be inlined into the four textureGatherOffset calls. */
static char **gather_offsets_resolve(const char *src, size_t len, const char *arr, size_t alen)
{
    if (alen == 0 || alen >= 128) return NULL;
    char id[130];
    memcpy(id, arr, alen);
    id[alen] = 0;

    char pat[160];
    snprintf(pat, sizeof pat, "%s[4]", id);
    const char *p = strstr(src, pat);
    if (!p) return NULL;
    p += strlen(pat);

    while (p < src + len && *p != '=') p++;
    if (p >= src + len) return NULL;
    p++;

    /* skip the constructor head up to its '(' (ivec2[] / ivec2[4]( / { ) */
    const char *open = NULL;
    for (const char *q = p; q < src + len; q++)
    {
        if (*q == '(' || *q == '{') { open = q; break; }
        if (*q == ';') break;
    }
    if (!open) return NULL;
    size_t close = match_paren(src, len, (size_t)(open - src));
    if (close >= len) return NULL;

    char **offs = calloc(4, sizeof(char *));
    if (!offs) return NULL;
    int n = 0;
    size_t start = (size_t)(open - src) + 1, depth = 0;
    for (size_t j = start; j <= close && n < 4; j++)
    {
        if (j < close && (src[j] == '(' || src[j] == '[')) depth++;
        else if (j < close && (src[j] == ')' || src[j] == ']')) depth--;
        if (j == close || (src[j] == ',' && depth == 0))
        {
            size_t s = start, e = j;
            while (s < e && isspace((unsigned char)src[s])) s++;
            while (e > s && isspace((unsigned char)src[e - 1])) e--;
            offs[n] = malloc(e - s + 1);
            if (!offs[n]) { for (int k = 0; k < 4; k++) free(offs[k]); free(offs); return NULL; }
            memcpy(offs[n], src + s, e - s);
            offs[n][e - s] = 0;
            n++;
            start = j + 1;
        }
    }
    if (n != 4)
    {
        for (int k = 0; k < 4; k++) free(offs[k]);
        free(offs);
        return NULL;
    }
    return offs;
}

/* an INLINE constant offsets argument: "ivec2[4](E0, E1, E2, E3)" (or ivec2[](...)
 * / a brace list) handed straight to textureGatherOffsets.  Returns the four
 * element expressions, or NULL when the argument is not a constant constructor
 * (dynamic offsets keep the documented approximation below).
 * dEQP-GL45-ES31 texture.gather.offsets.* passes the four offsets inline. */
static char **gather_offsets_inline(const char *arg, size_t alen)
{
    if (alen == 0 || alen >= 1024) return NULL;
    if (strncmp(arg, "ivec2", 5) != 0) return NULL;
    const char *p = arg + 5;
    const char *end = arg + alen;
    if (p < end && *p == '[')
    {
        p++;
        if (p < end && *p == ']') p++;                        /* ivec2[](...) */
        else while (p < end && isdigit((unsigned char)*p)) p++;   /* ivec2[4](...) */
        if (p >= end || *p != ']') return NULL;
        p++;
    }
    while (p < end && isspace((unsigned char)*p)) p++;
    if (p >= end || *p != '(') return NULL;

    /* split the four top-level element expressions of the constructor — the
     * same walk gather_offsets_resolve() does for a named const array: the
     * constructor's own '(' and ')' are the walk bounds, an element's parens
     * keep depth > 0 across its commas, and a split happens at a top-level
     * ',' or at the constructor's closing ')' */
    size_t open_off = (size_t)(p - arg);
    size_t close_off = match_paren(arg, alen, open_off);
    if (close_off >= alen) return NULL;
    char **offs = calloc(4, sizeof(char *));
    if (!offs) return NULL;
    int n = 0;
    size_t start = open_off + 1, depth = 0;
    for (size_t j = start; j <= close_off && n < 4; j++)
    {
        if (j < close_off && (arg[j] == '(' || arg[j] == '[')) depth++;
        else if (j < close_off && (arg[j] == ')' || arg[j] == ']')) depth--;
        if (j == close_off || (arg[j] == ',' && depth == 0))
        {
            const char *s = arg + start, *e = arg + j;
            while (s < e && isspace((unsigned char)s[0])) s++;
            while (e > s && isspace((unsigned char)e[-1])) e--;
            if (s >= e) { for (int k = 0; k < 4; k++) free(offs[k]); free(offs); return NULL; }
            offs[n] = malloc(e - s + 1);
            if (!offs[n]) { for (int k = 0; k < 4; k++) free(offs[k]); free(offs); return NULL; }
            memcpy(offs[n], s, e - s);
            offs[n][e - s] = 0;
            n++;
            start = j + 1;
        }
    }
    if (n != 4)
    {
        for (int k = 0; k < 4; k++) free(offs[k]);
        free(offs);
        return NULL;
    }
    return offs;
}

static char *rewrite_gather_offsets(const char *src)
{
    if (!strstr(src, "textureGatherOffsets")) return NULL;
    size_t len = strlen(src);
    Str out = {0};
    size_t i = 0;

    while (i < len)
    {
        if (!at_word(src, len, i, "textureGatherOffsets"))
        {
            str_push(&out, src[i++]);
            continue;
        }
        size_t w = i + strlen("textureGatherOffsets");
        size_t open = skip_ws_comments(src, len, w);
        if (open >= len || src[open] != '(')
        {
            str_append(&out, src + i, w - i);
            i = w;
            continue;
        }
        size_t close = match_paren(src, len, open);
        if (close >= len)
        {
            str_append(&out, src + i, w - i);
            i = w;
            continue;
        }

        /* split the top-level arguments */
        char args[4][512];
        int nargs = 0;
        {
            size_t a0 = open + 1, depth = 0;
            size_t start = a0;
            for (size_t j = a0; j <= close; j++)
            {
                if (j < close && (src[j] == '(' || src[j] == '[')) depth++;
                else if (j < close && (src[j] == ')' || src[j] == ']')) depth--;
                if (j == close || (src[j] == ',' && depth == 0))
                {
                    if (nargs < 4)
                    {
                        size_t s = start, e = j;
                        while (s < e && isspace((unsigned char)src[s])) s++;
                        while (e > s && isspace((unsigned char)src[e - 1])) e--;
                        size_t n = e - s;
                        if (n >= sizeof args[0]) n = sizeof args[0] - 1;
                        memcpy(args[nargs], src + s, n);
                        args[nargs][n] = 0;
                    }
                    nargs++;
                    start = j + 1;
                }
            }
        }

        if (nargs < 3)
        {
            /* leave anything unexpected (including the optional sampler+coord
             * overload split into two calls) untouched */
            str_append(&out, src + i, close + 1 - i);
            i = close + 1;
            continue;
        }

        const char *ctor = gather_offsets_ctor(src, len, args[0], strlen(args[0]));

        /* argument order: non-shadow (S, P, offsets[, component]), shadow
         * (S, P, refZ, offsets) — find which argument is the const array.
         * off_arg may only move when that argument really resolved: leaving it
         * at 3 after BOTH attempts failed swapped the component and offsets
         * arguments and emitted "textureGatherOffset(S, P, <ctor>, 0[0])" —
         * invalid GLSL ("left of '[' is not of type array..."), which failed
         * every 4-arg textureGatherOffsets call (dEQP-GL45-ES31
         * texture.gather.offsets.* min_required_offset / implementation_offset) */
        char **offs = NULL;
        int off_arg = 2;
        int last_arg = (nargs >= 4) ? 3 : 2;
        for (int a = 2; a <= last_arg && !offs; a++)
        {
            offs = gather_offsets_resolve(src, len, args[a], strlen(args[a]));
            if (!offs)
                offs = gather_offsets_inline(args[a], strlen(args[a]));
            if (offs)
                off_arg = a;
        }
        const char *comp = (nargs >= 4) ? args[off_arg == 2 ? 3 : 2] : "0";
        const char *arr = (nargs >= 4) ? args[off_arg] : args[2];

        /* constant offsets: inline the four literal expressions so SPIRV-Cross
         * forwards them (Metal's gather offset must be compile-time constant) */
        char buf[2400];
        if (offs)
        {
            /* non-shadow: (S, P, offset, [comp]); shadow: (S, P, refZ, offset) */
            if (off_arg == 3)
                snprintf(buf, sizeof buf,
                         "%s(textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w)",
                         ctor,
                         args[0], args[1], comp, offs[0],
                         args[0], args[1], comp, offs[1],
                         args[0], args[1], comp, offs[2],
                         args[0], args[1], comp, offs[3]);
            else
                snprintf(buf, sizeof buf,
                         "%s(textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w,"
                         " textureGatherOffset(%s, %s, %s, %s).w)",
                         ctor,
                         args[0], args[1], offs[0], comp,
                         args[0], args[1], offs[1], comp,
                         args[0], args[1], offs[2], comp,
                         args[0], args[1], offs[3], comp);
            for (int k = 0; k < 4; k++) free(offs[k]);
            free(offs);
        }
        else
        {
            /* dynamic offsets: SPIRV-Cross drops them (Metal cannot gather with
             * a runtime offset), so approximate with the block at offset[0] —
             * documented limitation, no CTS case reaches this */
            if (off_arg == 3)
                snprintf(buf, sizeof buf,
                         "%s(textureGatherOffset(%s, %s, %s, %s[0]).w,"
                         " textureGatherOffset(%s, %s, %s, %s[1]).w,"
                         " textureGatherOffset(%s, %s, %s, %s[2]).w,"
                         " textureGatherOffset(%s, %s, %s, %s[3]).w)",
                         ctor,
                         args[0], args[1], comp, arr,
                         args[0], args[1], comp, arr,
                         args[0], args[1], comp, arr,
                         args[0], args[1], comp, arr);
            else
                snprintf(buf, sizeof buf,
                         "%s(textureGatherOffset(%s, %s, %s[0], %s).w,"
                         " textureGatherOffset(%s, %s, %s[1], %s).w,"
                         " textureGatherOffset(%s, %s, %s[2], %s).w,"
                         " textureGatherOffset(%s, %s, %s[3], %s).w)",
                         ctor,
                         args[0], args[1], arr, comp,
                         args[0], args[1], arr, comp,
                         args[0], args[1], arr, comp,
                         args[0], args[1], arr, comp);
        }
        str_append(&out, buf, strlen(buf));
        i = close + 1;
    }

    if (!out.buf) { free(out.buf); return NULL; }
    str_push(&out, '\0');
    return out.buf;
}

/* ================================================================== bool block members
 *
 * glslang stores bool/bvecN uniform- and buffer-block members as uint/uvecN (GLSL-to-SPIR-V
 * convention; see header).  GL introspection must still report GL_BOOL/GL_BOOL_VEC*, so the
 * declared types are recovered from the source.  Only block members are of interest, and
 * only bool-ness, so anything the scanner does not recognize degrades to "not bool" — never
 * a false positive.  Paths are produced exactly the way walk_block() (introspect_spirv.c)
 * builds them, so the two sides agree on arrays and nested structs.
 */

typedef struct SmType SmType;
typedef struct {
    char *name;
    SmType *type;
} SmMember;

struct SmType {
    int is_struct;
    SmMember *members;
    int nmembers;
    int is_bool;              /* basic type is bool/bvecN ... */
    int bvec;                 /* ... with N components (1 = plain bool) */
    int is_array;
    int dim;                  /* outermost array length (0 = unsized) */
    SmType *elem;
};

typedef struct {
    const char *s;
    size_t n, i;
    char tok[128];            /* current token: identifier or one of { } ; , [ ] = */
    int tt;                   /* token class: 0 EOF, 1 ident, 2 punctuation */
    int pushed;
} P;

/* declared struct types (name -> SmType), so members typed by a struct name resolve */
typedef struct {
    char *name;
    SmType *type;
} SmSym;

typedef struct {
    P p;
    SmSym *syms;
    int nsyms, symcap;
    SmType **all;             /* every SmType created, for cleanup */
    int nall, allcap;
} Ctx;

/* --- tokens -------------------------------------------------------------- */

static void p_raw_advance(P *p)
{
    for (;;)
    {
        while (p->i < p->n && isspace((unsigned char)p->s[p->i])) p->i++;
        if (p->i + 1 < p->n && p->s[p->i] == '/' && p->s[p->i + 1] == '/')
        {
            while (p->i < p->n && p->s[p->i] != '\n') p->i++;
            continue;
        }
        if (p->i + 1 < p->n && p->s[p->i] == '/' && p->s[p->i + 1] == '*')
        {
            p->i += 2;
            while (p->i + 1 < p->n && !(p->s[p->i] == '*' && p->s[p->i + 1] == '/')) p->i++;
            p->i = p->i + 1 < p->n ? p->i + 2 : p->n;
            continue;
        }
        if (p->i < p->n && p->s[p->i] == '#')       /* preprocessor line incl. continuations */
        {
            /* comments are removed before directive processing, so a block comment
             * starting on a directive line continues it across the newline
             * (CTS function_redefinition_3: a '#define' whose body holds a comment
             * that spans the line end) */
            while (p->i < p->n)
            {
                if (p->s[p->i] == '/' && p->i + 1 < p->n && p->s[p->i + 1] == '*')
                {
                    p->i += 2;
                    while (p->i + 1 < p->n && !(p->s[p->i] == '*' && p->s[p->i + 1] == '/')) p->i++;
                    p->i = p->i + 1 < p->n ? p->i + 2 : p->n;
                    continue;
                }
                if (p->s[p->i] == '/' && p->i + 1 < p->n && p->s[p->i + 1] == '/')
                {
                    while (p->i < p->n && p->s[p->i] != '\n') p->i++;
                    break;
                }
                if (p->s[p->i] == '\\' && p->i + 1 < p->n && p->s[p->i + 1] == '\n') { p->i += 2; continue; }
                if (p->s[p->i] == '\n') break;
                p->i++;
            }
            continue;
        }
        return;
    }
}

static int p_next(P *p)
{
    if (p->pushed) { p->pushed = 0; return p->tt; }
    p_raw_advance(p);
    if (p->i >= p->n) { p->tt = 0; p->tok[0] = 0; return 0; }
    char c = p->s[p->i];
    if (is_ident_char((unsigned char)c))
    {
        size_t l = 0;
        while (p->i < p->n && is_ident_char((unsigned char)p->s[p->i]) && l + 1 < sizeof p->tok)
            p->tok[l++] = p->s[p->i++];
        p->tok[l] = 0;
        p->tt = 1;
        return 1;
    }
    p->i++;
    p->tok[0] = c;
    p->tok[1] = 0;
    p->tt = 2;
    return 2;
}

static void p_push(P *p) { p->pushed = 1; }

static int tok_is(P *p, const char *w) { return p->tt == 1 && !strcmp(p->tok, w); }
static int tok_is_punct(P *p, char c) { return p->tt == 2 && p->tok[0] == c; }

/* skip a balanced group starting at the current '(' / '{' / '[' */
static void skip_group(P *p, char open, char close)
{
    int depth = 0;
    do {
        if (p->tt == 2 && p->tok[0] == open) depth++;
        else if (p->tt == 2 && p->tok[0] == close) depth--;
        p_next(p);
    } while (p->tt && depth > 0);
}

/* member/declaration qualifiers that may precede a type in a block body or declaration */
static int is_qualifier(P *p)
{
    static const char *const q[] = { "layout", "highp", "mediump", "lowp", "flat", "smooth",
        "noperspective", "centroid", "sample", "patch", "invariant", "precise", "coherent",
        "volatile", "restrict", "readonly", "writeonly", "uniform", "buffer", "in", "out",
        "attribute", "varying", "shared", "packed", NULL };
    if (p->tt != 1) return 0;
    for (int i = 0; q[i]; i++) if (!strcmp(p->tok, q[i])) return 1;
    return 0;
}

/* --- little type model ---------------------------------------------------- */

static SmType *sm_basic(int is_bool, int bvec)
{
    SmType *t = calloc(1, sizeof *t);
    t->is_bool = is_bool;
    t->bvec = bvec;
    return t;
}

static SmType *sm_array(SmType *elem, int dim)
{
    SmType *t = calloc(1, sizeof *t);
    t->is_array = 1;
    t->dim = dim;
    t->elem = elem;
    return t;
}

static void sm_free(SmType *t)
{
    if (!t) return;
    for (int i = 0; i < t->nmembers; i++) { free(t->members[i].name); sm_free(t->members[i].type); }
    free(t->members);
    sm_free(t->elem);
    free(t);
}

static SmType *ctx_track(Ctx *c, SmType *t)
{
    if (c->nall == c->allcap)
    {
        c->allcap = c->allcap ? c->allcap * 2 : 32;
        c->all = realloc(c->all, (size_t)c->allcap * sizeof(SmType *));
    }
    c->all[c->nall++] = t;
    return t;
}

static void ctx_add_sym(Ctx *c, const char *name, SmType *t)
{
    for (int i = 0; i < c->nsyms; i++) if (!strcmp(c->syms[i].name, name)) { c->syms[i].type = t; return; }
    if (c->nsyms == c->symcap)
    {
        c->symcap = c->symcap ? c->symcap * 2 : 16;
        c->syms = realloc(c->syms, (size_t)c->symcap * sizeof(SmSym));
    }
    c->syms[c->nsyms].name = strdup(name);
    c->syms[c->nsyms].type = t;
    c->nsyms++;
}

static SmType *ctx_lookup_sym(Ctx *c, const char *name)
{
    for (int i = 0; i < c->nsyms; i++) if (!strcmp(c->syms[i].name, name)) return c->syms[i].type;
    return NULL;
}

static void sm_add_member(SmType *s, const char *name, SmType *type)
{
    s->members = realloc(s->members, (size_t)(s->nmembers + 1) * sizeof(SmMember));
    s->members[s->nmembers].name = strdup(name);
    s->members[s->nmembers].type = type;
    s->nmembers++;
}

/* parse "[N]*" array suffixes, wrapping `base`; leaves the next non-'[' token current */
static SmType *parse_array_suffix(P *p, SmType *base)
{
    int dims[8], nd = 0;
    while (tok_is_punct(p, '['))
    {
        p_next(p);
        int d = 0;
        if (p->tt == 1) d = atoi(p->tok);           /* integer literal */
        while (p->tt && !tok_is_punct(p, ']')) p_next(p);
        if (p->tt) p_next(p);
        if (nd < 8) dims[nd++] = d;
    }
    for (int i = nd - 1; i >= 0; i--) base = sm_array(base, dims[i]);
    return base;
}

/* parse the body { ... } of a struct/block into `s`; '{' is the current token on entry,
 * the token after '}' is current on exit */
static void parse_member_list(Ctx *c, SmType *s)
{
    P *p = &c->p;
    p_next(p);                                      /* enter the body */
    while (p->tt)
    {
        if (tok_is_punct(p, '}')) { p_next(p); return; }

        while (is_qualifier(p))
        {
            if (tok_is(p, "layout"))
            {
                p_next(p);
                if (tok_is_punct(p, '(')) skip_group(p, '(', ')');
                else if (p->tt) p_next(p);
            }
            else if (p->tt) p_next(p);
        }
        if (!p->tt) return;

        SmType *base = NULL;
        if (tok_is(p, "struct"))
        {
            p_next(p);
            SmType *inner = ctx_track(c, calloc(1, sizeof *inner));
            inner->is_struct = 1;
            if (p->tt == 1)
            {
                ctx_add_sym(c, p->tok, inner);      /* named struct type */
                p_next(p);
            }
            if (tok_is_punct(p, '{')) parse_member_list(c, inner);
            base = inner;
        }
        else
        {
            char tyname[128];
            snprintf(tyname, sizeof tyname, "%s", p->tok);
            int is_bool = 0, bvec = 0;
            if (!strcmp(tyname, "bool")) { is_bool = 1; bvec = 1; }
            else if (!strcmp(tyname, "bvec2")) { is_bool = 1; bvec = 2; }
            else if (!strcmp(tyname, "bvec3")) { is_bool = 1; bvec = 3; }
            else if (!strcmp(tyname, "bvec4")) { is_bool = 1; bvec = 4; }
            p_next(p);
            if (is_bool) base = ctx_track(c, sm_basic(is_bool, bvec));
            else base = ctx_lookup_sym(c, tyname);  /* declared struct type, else unknown */
        }

        /* declarators: name [dims] (, name [dims])* ; */
        while (p->tt)
        {
            if (p->tt == 1)
            {
                char name[128];
                snprintf(name, sizeof name, "%s", p->tok);
                p_next(p);
                SmType *t = parse_array_suffix(p, base);
                sm_add_member(s, name, t);
            }
            else if (tok_is_punct(p, ',')) p_next(p);
            else if (tok_is_punct(p, ';')) { p_next(p); break; }
            else break;                              /* unexpected: abandon this list */
        }
    }
}

/* --- collect bool paths --------------------------------------------------- */

struct MGLBoolPaths {
    char **paths;
    int n, cap;
};

static void paths_add(MGLBoolPaths *out, const char *block, const char *path)
{
    char full[512];
    if (block[0]) snprintf(full, sizeof full, "%s.%s", block, path);
    else snprintf(full, sizeof full, "%s", path);   /* anonymous block: no prefix */
    for (int i = 0; i < out->n; i++) if (!strcmp(out->paths[i], full)) return;
    if (out->n == out->cap)
    {
        out->cap = out->cap ? out->cap * 2 : 32;
        out->paths = realloc(out->paths, (size_t)out->cap * sizeof(char *));
    }
    out->paths[out->n++] = strdup(full);
}

/* mirror walk_block()'s naming: arrays of basic type collapse to "[0]", arrays of
 * aggregates expand per element, structs append ".member" */
static void emit_paths(const char *block, const char *prefix, SmType *t, MGLBoolPaths *out)
{
    if (!t) return;
    if (t->is_array)
    {
        SmType *e = t->elem;
        if (e && !e->is_array && !e->is_struct)
        {
            if (e->is_bool)
            {
                char p2[512];
                snprintf(p2, sizeof p2, "%s[0]", prefix);
                paths_add(out, block, p2);
            }
            return;
        }
        int n = t->dim ? t->dim : 1;
        for (int i = 0; i < n; i++)
        {
            char p2[512];
            snprintf(p2, sizeof p2, "%s[%d]", prefix, i);
            emit_paths(block, p2, e, out);
        }
        return;
    }
    if (t->is_struct)
    {
        for (int i = 0; i < t->nmembers; i++)
        {
            char p2[512];
            if (prefix[0]) snprintf(p2, sizeof p2, "%s.%s", prefix, t->members[i].name);
            else snprintf(p2, sizeof p2, "%s", t->members[i].name);
            emit_paths(block, p2, t->members[i].type, out);
        }
        return;
    }
    if (t->is_bool) paths_add(out, block, prefix);
}

/* skip one statement at the current nesting level (handles nested groups) */
static void skip_statement(P *p)
{
    while (p->tt)
    {
        if (p->tt == 2 && (p->tok[0] == '{' || p->tok[0] == '(' || p->tok[0] == '['))
        {
            char o = p->tok[0];
            skip_group(p, o, o == '{' ? '}' : (o == '(' ? ')' : ']'));
            continue;
        }
        if (p->tt == 2 && (p->tok[0] == '}' || p->tok[0] == ')' || p->tok[0] == ']'))
        {
            /* entered AT a closer: leave it (and any pending push) to the caller —
             * consuming/re-pushing here cannot make progress and used to loop forever */
            if (!p->pushed) p_push(p);
            return;
        }
        int done = (p->tt == 2 && p->tok[0] == ';');
        p_next(p);
        if (done) return;
        if (p->tt == 2 && (p->tok[0] == '}' || p->tok[0] == ')' || p->tok[0] == ']')) { p_push(p); return; }
    }
}

MGLBoolPaths *mgl_cts_bool_block_members(const char *src)
{
    if (!src) return NULL;
    MGLBoolPaths *out = calloc(1, sizeof *out);
    Ctx c;
    memset(&c, 0, sizeof c);
    c.p.s = src;
    c.p.n = strlen(src);
    p_next(&c.p);

    while (c.p.tt)
    {
        P *p = &c.p;
        if (tok_is(p, "layout"))
        {
            p_next(p);
            if (tok_is_punct(p, '(')) skip_group(p, '(', ')');
            else skip_statement(p);
            continue;
        }
        if (tok_is(p, "struct"))
        {
            /* record a global struct definition so block members typed by it resolve */
            p_next(p);
            SmType *inner = ctx_track(&c, calloc(1, sizeof *inner));
            inner->is_struct = 1;
            if (p->tt == 1) { ctx_add_sym(&c, p->tok, inner); p_next(p); }
            if (tok_is_punct(p, '{')) parse_member_list(&c, inner);
            skip_statement(p);                        /* optional declarators to ';' */
            continue;
        }
        if (tok_is(&c.p, "uniform") || tok_is(&c.p, "buffer"))
        {
            int is_block = 0;
            P save = c.p;
            p_next(p);
            while (is_qualifier(p) && !tok_is(p, "uniform") && !tok_is(p, "buffer")) p_next(p);
            if (p->tt == 1)                             /* block type name or plain type */
            {
                char blockname[128];
                snprintf(blockname, sizeof blockname, "%s", p->tok);
                p_next(&c.p);
                if (tok_is_punct(&c.p, '{'))
                {
                    is_block = 1;
                    SmType root;
                    memset(&root, 0, sizeof root);
                    root.is_struct = 1;
                    parse_member_list(&c, &root);
                    /* GL prefixes member names with the block name only when the block has
                     * an instance name; walk_block() must agree (see introspect_spirv.c) */
                    if (c.p.tt != 1) blockname[0] = 0;
                    emit_paths(blockname, "", &root, out);
                }
            }
            if (!is_block) { c.p = save; skip_statement(&c.p); }
            else
            {
                /* skip the instance declarator to ';' */
                while (c.p.tt && !tok_is_punct(&c.p, ';'))
                {
                    if (c.p.tt == 2 && (c.p.tok[0] == '{' || c.p.tok[0] == '('))
                    {
                        char o = c.p.tok[0];
                        skip_group(&c.p, o, o == '{' ? '}' : ')');
                        continue;
                    }
                    p_next(&c.p);
                }
                if (c.p.tt) p_next(&c.p);
            }
            continue;
        }
        /* a stray closer at top level (unbalanced text left over from a macro body,
         * e.g. CTS function_redefinition_3): drop it and move on — skip_statement
         * deliberately parks closers for its caller */
        if (c.p.tt == 2 && (c.p.tok[0] == '}' || c.p.tok[0] == ')' || c.p.tok[0] == ']'))
        {
            c.p.pushed = 0;
            p_next(&c.p);
            continue;
        }
        skip_statement(&c.p);
    }

    for (int i = 0; i < c.nall; i++) { for (int k = 0; k < c.all[i]->nmembers; k++) { free(c.all[i]->members[k].name); } free(c.all[i]->members); free(c.all[i]->elem); free(c.all[i]); }
    for (int i = 0; i < c.nsyms; i++) free(c.syms[i].name);
    free(c.syms); free(c.all);

    if (!out->n) { mgl_cts_bool_paths_free(out); return NULL; }
    return out;
}

int mgl_cts_bool_paths_has(const MGLBoolPaths *p, const char *path)
{
    if (!p || !path) return 0;
    for (int i = 0; i < p->n; i++) if (!strcmp(p->paths[i], path)) return 1;
    return 0;
}

void mgl_cts_bool_paths_free(MGLBoolPaths *p)
{
    if (!p) return;
    for (int i = 0; i < p->n; i++) free(p->paths[i]);
    free(p->paths);
    free(p);
}

/* debug helper (used by tests / MGL_DEBUG_BOOL_PATHS) */
#include <stdio.h>
void mgl_cts_bool_paths_debug(const MGLBoolPaths *p)
{
    if (!p) { printf("bool paths: (none)\n"); return; }
    printf("bool paths: %d\n", p->n);
    for (int i = 0; i < p->n; i++) printf("  %s\n", p->paths[i]);
}
