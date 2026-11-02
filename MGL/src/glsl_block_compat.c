/* glsl_block_compat.c — GLSL-interface-block grammar glslang cannot parse.
 *
 * Two GL 4.x-legal interface-block forms are rejected outright by the pinned
 * glslang grammar (verified with glslangValidator 11.16.1):
 *
 *   1. an ANONYMOUS block — `buffer { vec4 a; float b; } blk;` (any of
 *      uniform / buffer / in / out): "syntax error, unexpected LEFT_BRACE".
 *   2. an INLINE STRUCT DECLARATION inside a block's member list —
 *      `buffer Block { vec4 a; Struct { vec4 sb; float sa[2]; } bs[2]; } blk;`
 *      — "syntax error, unexpected IDENTIFIER".
 *
 * Both appear in dEQP-GL45-ES31 program_interface_query (e.g.
 * shader_storage_block.buffer_data_size.unnamed_block / .named_block, which
 * build TargetInterface exactly like that).  This pass rewrites the source
 * before glslang sees it, without changing GL semantics:
 *
 *   1. the block gets a NAME: the instance name when one follows the `}`
 *      (for an anonymous block the GL resource name IS the instance name, and
 *      the block-type name is invisible to GL, so making them equal preserves
 *      every query), else a synthetic `mgl_anon_block_N` (an anonymous block
 *      WITHOUT an instance name exposes its members as globals; the block
 *      resource then has no GL name and MGL reports the synthetic one).
 *   2. the inline struct is hoisted in front of the block as
 *      `struct NAME { ... };` and the member keeps `NAME declarators;`.
 *
 * Everything else (comments, preprocessor lines, layout lists) is copied
 * verbatim; the pass returns NULL when nothing needed rewriting.
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *s; size_t n, cap; } BStr;

static void bs_put(BStr *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap)
    {
        b->cap = (b->n + n + 1) * 2 + 256;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

static void bs_puts(BStr *b, const char *s) { bs_put(b, s, strlen(s)); }

/* --- source scanning helpers (comment-aware) -------------------------------- */

static bool is_idc(int c) { return isalnum(c) || c == '_'; }

static size_t skip_ws(const char *s, size_t len, size_t i)
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
            i = (i + 1 < len) ? i + 2 : len;
            continue;
        }
        break;
    }
    return i;
}

static size_t skip_word(const char *s, size_t len, size_t i)
{
    while (i < len && is_idc(s[i])) i++;
    return i;
}

static bool at_word(const char *s, size_t len, size_t i, const char *w)
{
    size_t wl = strlen(w);
    if (i + wl > len || strncmp(s + i, w, wl)) return false;
    return (i + wl >= len || !is_idc(s[i + wl])) && (i == 0 || !is_idc(s[i - 1]));
}

/* matching ')' / ']' / '}' for the opener at i */
static size_t match_close(const char *s, size_t len, size_t i)
{
    char open = s[i], close = (open == '(') ? ')' : (open == '[') ? ']' : '}';
    int depth = 0;
    for (size_t p = i; p < len; p++)
    {
        if (s[p] == '/' && p + 1 < len && (s[p + 1] == '/' || s[p + 1] == '*'))
        {
            p = skip_ws(s, len, p);      /* returns past the comment */
            if (p >= len) return len;
        }
        if (s[p] == open) depth++;
        else if (s[p] == close && --depth == 0) return p;
    }
    return len;
}

/* --- edit list --------------------------------------------------------------- */

typedef struct { size_t pos; size_t len; const char *text; size_t textlen; } Edit;

static int edit_cmp(const void *a, const void *b)
{
    const Edit *x = a, *y = b;
    if (x->pos != y->pos) return (x->pos < y->pos) ? -1 : 1;
    if (x->len != y->len) return (x->len < y->len) ? -1 : 1;
    return 0;
}

/* the qualifier words that may sit between `uniform`/`buffer`/... and the block */
static bool is_block_qualifier(const char *s, size_t len, size_t i, size_t *next)
{
    static const char *q[] = { "uniform", "buffer", "in", "out", "layout", "shared", "packed",
                               "std140", "std430", "row_major", "column_major", "binding",
                               "readonly", "writeonly", "smooth", "flat", "noperspective",
                               "centroid", "sample", "patch", "invariant", NULL };
    if (!isalpha((unsigned char)s[i]) && s[i] != '_') return false;
    size_t e = skip_word(s, len, i);
    for (int k = 0; q[k]; k++)
        if (e - i == (size_t)strlen(q[k]) && !strncmp(s + i, q[k], e - i))
        {
            size_t n = skip_ws(s, len, e);
            if (n < len && s[n] == '(')                 /* layout(...) */
                n = skip_ws(s, len, match_close(s, len, n) + 1);
            else if (n < len && s[n] == '=')            /* binding = N */
            {
                n = skip_ws(s, len, n + 1);
                while (n < len && (isdigit((unsigned char)s[n]) || is_idc(s[n]))) n++;
                n = skip_ws(s, len, n);
            }
            *next = n;
            return true;
        }
    return false;
}

char *mgl_glsl_block_compat(const char *src)
{
    size_t len = strlen(src);
    Edit edits[128];
    int nedits = 0;
    size_t i = 0;
    int anon = 0;
    BStr hoist = {0};

    while (i < len)
    {
        /* preprocessor lines pass through */
        if (src[i] == '#')
        {
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        if (isspace((unsigned char)src[i]) || src[i] == '/' || !isalpha((unsigned char)src[i]))
        {
            i++;
            continue;
        }
        /* consume the qualifier chain (layout(...) std430 buffer ...), then
         * NAME { ... } (nothing to do) or { ... } (anonymous — name it) */
        size_t chain_start = i;
        if (!is_block_qualifier(src, len, i, &i)) { i = skip_word(src, len, i); continue; }
        size_t lastq_start = chain_start;        /* start of the last qualifier matched */
        {
            size_t nx;
            while (is_block_qualifier(src, len, i, &nx))
            {
                lastq_start = i;
                i = nx;
            }
        }
        size_t head = i;
        size_t bopen = skip_ws(src, len, i);
        bool anonymous = false;
        size_t name_pos = 0, name_len = 0;
        if (bopen < len && src[bopen] == '{')
        {
            /* `uniform readonly { ... }` (writeonly/invariant/...) puts a RESERVED
             * word where the block NAME goes — glslang must reject the source, so
             * do not hand it a synthetic name.  A block is anonymous only when the
             * qualifier chain ENDS with a storage qualifier; a memory/interpolation
             * qualifier right before the `{` is either that illegal form or
             * `readonly buffer {...}` (chain ends in `buffer` — still handled). */
            size_t qe = skip_word(src, len, lastq_start);
            bool storage = (qe - lastq_start == 7 && !strncmp(src + lastq_start, "uniform", 7)) ||
                           (qe - lastq_start == 6 && !strncmp(src + lastq_start, "buffer", 6)) ||
                           (qe - lastq_start == 2 && !strncmp(src + lastq_start, "in", 2)) ||
                           (qe - lastq_start == 3 && !strncmp(src + lastq_start, "out", 3));
            if (!storage) continue;
            anonymous = true;
        }
        else if (bopen < len && isalpha((unsigned char)src[bopen]))
        {
            name_pos = bopen;
            name_len = skip_word(src, len, bopen) - bopen;
            size_t after = skip_ws(src, len, name_pos + name_len);
            if (after < len && src[after] == '{') bopen = after;
            else continue;                      /* an ordinary declaration */
        }
        else continue;

        size_t bclose = match_close(src, len, bopen);
        if (bclose >= len) continue;

        if (anonymous)
        {
            /* instance name (optional) after the closing brace, before `;` */
            size_t p = skip_ws(src, len, bclose + 1);
            size_t inst_pos = p, inst_len = 0;
            if (p < len && (isalpha((unsigned char)src[p]) || src[p] == '_'))
                inst_len = skip_word(src, len, p) - p;
            (void)inst_pos;
            /* the synthetic name must differ from the instance name (glslang:
             * "block instance name redefinition"); introspection maps
             * mgl_anon_block_N back to the instance name for GL queries */
            char nm[128], text[144];
            snprintf(nm, sizeof nm, "mgl_anon_block_%d", anon++);
            snprintf(text, sizeof text, " %s", nm);
            if (nedits < 128)
            {
                edits[nedits].pos = bopen;
                edits[nedits].len = 0;
                edits[nedits].text = strdup(text);
                edits[nedits].textlen = strlen(text);
                nedits++;
            }
        }

        /* inline struct declarations among the members: IDENT { ... } declarators ; */
        {
            size_t p = skip_ws(src, len, bopen + 1);
            while (p < bclose)
            {
                if (isalpha((unsigned char)src[p]) || src[p] == '_')
                {
                    size_t we = skip_word(src, len, p);
                    size_t q = skip_ws(src, len, we);
                    if (q < bclose && src[q] == '{')
                    {
                        size_t qclose = match_close(src, len, q);
                        if (qclose < bclose)
                        {
                            /* struct NAME { body } -> hoist, member keeps NAME */
                            if (nedits < 126)
                            {
                                char *decl = malloc(64 + (qclose - q + 1));
                                snprintf(decl, 64 + (qclose - q + 1), "struct %.*s %.*s; ",
                                         (int)(we - p), src + p, (int)(qclose - q + 1), src + q);
                                edits[nedits].pos = chain_start;   /* before the qualifier chain */
                                edits[nedits].len = 0;
                                edits[nedits].text = decl;
                                edits[nedits].textlen = strlen(decl);
                                nedits++;
                                char *rep = strndup(src + p, we - p);
                                edits[nedits].pos = p;
                                edits[nedits].len = qclose + 1 - p;
                                edits[nedits].text = rep;
                                edits[nedits].textlen = strlen(rep);
                                nedits++;
                            }
                            p = skip_ws(src, len, qclose + 1);
                            continue;
                        }
                    }
                    /* skip the rest of this member up to its `;` (brace/paren aware) */
                    p = we;
                    while (p < bclose)
                    {
                        if (src[p] == '{' || src[p] == '(' || src[p] == '[')
                            p = match_close(src, len, p) + 1;
                        else if (src[p] == ';') { p++; break; }
                        else p++;
                    }
                    p = skip_ws(src, len, p);
                }
                else if (src[p] == ';') p = skip_ws(src, len, p + 1);
                else p++;
            }
        }
        i = bclose + 1;
    }

    if (!nedits)
    {
        free(hoist.s);
        return NULL;
    }
    qsort(edits, nedits, sizeof edits[0], edit_cmp);

    BStr out = {0};
    size_t o = 0;
    for (int k = 0; k < nedits; k++)
    {
        size_t pos = edits[k].pos;
        size_t end = edits[k].pos + edits[k].len;
        if (pos < o) pos = o;                      /* (edits never overlap) */
        if (pos > o) bs_put(&out, src + o, pos - o);
        bs_put(&out, edits[k].text, edits[k].textlen);
        if (end > o) o = end;
    }
    if (o < len) bs_put(&out, src + o, len - o);

    for (int k = 0; k < nedits; k++) free((void *)edits[k].text);
    free(hoist.s);
    return out.s;
}
