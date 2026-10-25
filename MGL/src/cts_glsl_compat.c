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

    char pat[160];
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
    if (!strstr(src, "textureGatherOffset")) return NULL;
    size_t len = strlen(src);
    Str out = {0};
    size_t i = 0;
    bool changed = false;

    while (i < len)
    {
        if (!at_word(src, len, i, "textureGatherOffset"))
        {
            str_push(&out, src[i++]);
            continue;
        }
        size_t w = i + strlen("textureGatherOffset");
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

        GatherDim dim = (nargs >= 3 && nargs <= 4) ? gather_dim_of(src, args[0], strlen(args[0])) : GATHER_T_UNKNOWN;
        bool shadow = (dim == GATHER_T2D_SHADOW || dim == GATHER_T2DARRAY_SHADOW);
        int off_arg = shadow ? nargs - 1 : 2;              /* (S,P,off[,comp]) | (S,P,ref,off) */
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
            snprintf(buf, sizeof buf, "textureGather(%s, %s, %s)", args[0], shift, args[nargs - 2]);
        else if (nargs == 4)
            snprintf(buf, sizeof buf, "textureGather(%s, %s, %s)", args[0], shift, args[3]);
        else
            snprintf(buf, sizeof buf, "textureGather(%s, %s)", args[0], shift);
        str_append(&out, buf, strlen(buf));
        i = close + 1;
        changed = true;
    }
    if (!changed) { free(out.buf); return NULL; }
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
        if (depth == 0 && at_word(out, len, i, "invariant"))
        {
            /* declaration span: from the previous ; { } to the next ; */
            size_t ds = i;
            while (ds > 0 && out[ds - 1] != ';' && out[ds - 1] != '{' && out[ds - 1] != '}') ds--;
            size_t de = i;
            while (de < len && out[de] != ';') de++;
            bool has_in = false, has_out = false, has_type = false;
            for (size_t k = ds; k < de; k++)
            {
                if (at_word(out, len, k, "in")) has_in = true;
                else if (at_word(out, len, k, "out")) has_out = true;
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
        if (p + 1 < end && (c == '&' && p[1] == '&') || (c == '|' && p[1] == '|') ||
            (c == '<' && p[1] == '<') || (c == '>' && p[1] == '>') ||
            (c == '<' && p[1] == '=') || (c == '>' && p[1] == '=') ||
            (c == '=' && p[1] == '=') || (c == '!' && p[1] == '='))
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
        bs_ws(&s);
        char nxt = s.p < s.end ? *s.p : 0;

        char err[96];
        unsigned limit = 0;
        if (bs_word_is(a, a_n, "uniform") && bs_word_is(b, b_n, "atomic_uint"))
            limit = lim->atomic_bindings;
        else if (bs_word_is(a, a_n, "uniform") && b && nxt == '{')
            limit = lim->ubo_bindings;                     /* uniform block */
        else if (bs_word_is(a, a_n, "uniform") && b_n >= 5 && strncmp(b, "image", 5) == 0)
            limit = lim->image_units;
        else if (bs_word_is(a, a_n, "uniform") && b_n >= 7 && strncmp(b, "sampler", 7) == 0)
            limit = lim->combined_texture_units;
        /* buffer blocks are handled by reject_ssbo_binding_range (array extents
         * included); block instance arrays are not checked here */

        if (limit && binding >= limit)
        {
            snprintf(err, sizeof err,
                     "layout(binding=%u) exceeds the implementation limit (%u)", binding, limit);
            return pp_error_source(src, err);
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
    char *cur = strip_invariant_on_input(src);
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

char *mgl_cts_glsl_compat(const char *src)
{
    return mgl_cts_glsl_compat_limited(src, NULL);
}

char *mgl_cts_glsl_compat_limited(const char *src, const MGLCompatBindingLimits *limits)
{
    if (!src) return NULL;
    /* spec-shaped rejections first (see reject_pp_strictness): once one fires the
     * compile must fail, and the other passes are pointless */
    char *strict = reject_pp_strictness(src);
    if (strict) return strict;
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
    /* every step returns a malloc'd copy or NULL when it changed nothing, so the
     * chain keeps the latest non-NULL text and falls back to the original when
     * no rewrite fired at all */
    char *cur = strip_invariant_on_input(src);
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
         * (S, P, refZ, offsets) — find which argument is the const array */
        char **offs = NULL;
        int off_arg = 2;
        if (nargs >= 4)
        {
            offs = gather_offsets_resolve(src, len, args[2], strlen(args[2]));
            if (!offs)
            {
                offs = gather_offsets_resolve(src, len, args[3], strlen(args[3]));
                off_arg = 3;
            }
        }
        else
        {
            offs = gather_offsets_resolve(src, len, args[2], strlen(args[2]));
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
