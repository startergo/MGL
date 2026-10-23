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

static char *rewrite_gather_offsets(const char *src);

char *mgl_cts_glsl_compat(const char *src)
{
    if (!src) return NULL;
    /* every step returns a malloc'd copy or NULL when it changed nothing, so the
     * chain keeps the latest non-NULL text and falls back to the original when
     * no rewrite fired at all */
    char *cur = strip_gl_block_packings(src);
    char *next = reject_buffer_layout_decl(cur ? cur : src);
    if (next && cur) free(cur);
    if (next) cur = next;
    char *g = rewrite_gather_offsets(cur ? cur : src);
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
            while (p->i < p->n && (p->s[p->i] != '\n' || (p->i && p->s[p->i - 1] == '\\'))) p->i++;
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
