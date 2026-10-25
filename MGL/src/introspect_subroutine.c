/*
 * introspect_subroutine.c
 * MGL — 'introspect' domain: GLSL subroutine emulation.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * glslang cannot generate SPIR-V for GLSL subroutines, so they are rewritten at the source
 * level before compilation (mgl_introspect_shader_source(), called from mglCompileShader):
 *
 *   subroutine vec4 ColorFn(vec2 uv);                -> removed (signature recorded)
 *   layout(index=N) subroutine(ColorFn) vec4 red(..)  -> qualifier removed, red gets index N
 *   layout(location=L) subroutine uniform ColorFn u;  -> layout(location=39xx) uniform int _mgl_sr_u;
 *                                                       + prototype of _mgl_sr_call_u(...)
 *   u(args) / u[i](args)                              -> _mgl_sr_call_u(args) / _mgl_sr_call_u(i, args)
 *
 * and a dispatcher "switch (_mgl_sr_u) { case <index>: return red(args); ... }" is appended for
 * every subroutine uniform.  The selector is an ordinary (hidden) default-block uniform, so
 * glUniformSubroutinesuiv only writes ints into the program's default uniform block.
 * Line numbers are preserved (removed text is replaced by spaces).
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl_ext_introspect.h"

#define SR_SELECTOR_LOCATION_BASE 3900

typedef struct { char *name; char *ret; char *params; } SrType;
typedef struct { char *name; char **compat; int ncompat; int index; char *ret; char *params; } SrFunc;
typedef struct { char *name; char *type; int size; int location; size_t at; int ndims; int dims[8]; } SrUni;

typedef struct { char *s; size_t n, cap; } Str;
static void sputn(Str *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2; b->s = realloc(b->s, b->cap); }
    memcpy(b->s + b->n, s, n); b->n += n; b->s[b->n] = 0;
}
static void sput(Str *b, const char *s) { sputn(b, s, strlen(s)); }
static void sprintf_(Str *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#include <stdarg.h>
static void sprintf_(Str *b, const char *fmt, ...)
{
    char tmp[4096]; va_list ap; va_start(ap, fmt); vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap); sput(b, tmp);
}

static char *dupn(const char *s, size_t n) { char *r = malloc(n + 1); memcpy(r, s, n); r[n] = 0; return r; }

static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = 0;
    return s;
}

static bool idc(char c) { return isalnum((unsigned char)c) || c == '_'; }

typedef struct { char *b; uint8_t *code; size_t n; } Src;

/* mark comments and preprocessor lines as non-code */
static void build_mask(Src *s)
{
    s->code = malloc(s->n + 1);
    memset(s->code, 1, s->n + 1);
    bool line_start = true;
    for (size_t i = 0; i < s->n; ) {
        char c = s->b[i];
        if (c == '/' && i + 1 < s->n && s->b[i + 1] == '/') { while (i < s->n && s->b[i] != '\n') s->code[i++] = 0; continue; }
        if (c == '/' && i + 1 < s->n && s->b[i + 1] == '*') {
            s->code[i++] = 0; s->code[i++] = 0;
            while (i < s->n && !(s->b[i] == '*' && i + 1 < s->n && s->b[i + 1] == '/')) s->code[i++] = 0;
            if (i < s->n) { s->code[i++] = 0; s->code[i++] = 0; }
            continue;
        }
        if (line_start && c == '#') {
            while (i < s->n && s->b[i] != '\n') {
                if (s->b[i] == '\\' && i + 1 < s->n && s->b[i + 1] == '\n') { s->code[i++] = 0; }
                s->code[i++] = 0;
            }
            continue;
        }
        if (c == '\n') line_start = true; else if (!isspace((unsigned char)c)) line_start = false;
        i++;
    }
}

static size_t skip_ws(const Src *s, size_t i) { while (i < s->n && (isspace((unsigned char)s->b[i]) || !s->code[i])) i++; return i; }

static size_t read_ident(const Src *s, size_t i, size_t *len)
{
    i = skip_ws(s, i);
    size_t j = i;
    while (j < s->n && s->code[j] && idc(s->b[j])) j++;
    *len = j - i;
    return i;
}

static size_t match_close(const Src *s, size_t open, char oc, char cc)
{
    int depth = 0;
    for (size_t i = open; i < s->n; i++) {
        if (!s->code[i]) continue;
        if (s->b[i] == oc) depth++;
        else if (s->b[i] == cc) { if (--depth == 0) return i; }
    }
    return s->n;
}

static void blank(Src *s, size_t a, size_t b)
{
    for (size_t i = a; i < b && i < s->n; i++) if (s->b[i] != '\n') s->b[i] = ' ';
}

/* "layout(...)" immediately before position kw; returns its start or kw; fills key values */
static size_t layout_before(const Src *s, size_t kw, int *loc, int *index)
{
    size_t i = kw;
    while (i > 0 && (isspace((unsigned char)s->b[i - 1]) || !s->code[i - 1])) i--;
    if (i == 0 || s->b[i - 1] != ')') return kw;
    int depth = 0; size_t j = i - 1;
    for (;; j--) {
        if (s->code[j]) { if (s->b[j] == ')') depth++; else if (s->b[j] == '(' && --depth == 0) break; }
        if (j == 0) return kw;
    }
    size_t open = j;
    size_t k = open;
    while (k > 0 && isspace((unsigned char)s->b[k - 1])) k--;
    if (k < 6 || strncmp(s->b + k - 6, "layout", 6) || (k > 6 && idc(s->b[k - 7]))) return kw;
    char *inner = dupn(s->b + open + 1, i - 1 - open - 1);
    for (char *t = strtok(inner, ","); t; t = strtok(NULL, ",")) {
        char key[64] = ""; int v = -1;
        if (sscanf(t, " %63[a-z_] = %d", key, &v) == 2) {
            if (!strcmp(key, "location") && loc) *loc = v;
            if (!strcmp(key, "index") && index) *index = v;
        }
    }
    free(inner);
    return k - 6;
}

/* split a parameter list into declarations with generated names */
static int split_params(const char *params, char ***decl)
{
    char *p = strdup(params ? params : "");
    char *t = trim(p);
    *decl = NULL;
    if (!*t || !strcmp(t, "void")) { free(p); return 0; }
    int n = 0, depth = 0;
    char *start = t;
    for (char *c = t; ; c++) {
        if (*c == '(' || *c == '[') depth++;
        if (*c == ')' || *c == ']') depth--;
        if ((*c == ',' && depth == 0) || *c == 0) {
            char save = *c; *c = 0;
            char *one = trim(start);
            /* array suffix */
            char suffix[64] = "";
            char *lb = strchr(one, '[');
            if (lb) { snprintf(suffix, sizeof suffix, "%s", lb); *lb = 0; one = trim(one); }
            /* drop a trailing parameter name (needs at least a type before it) */
            static const char *quals[] = { "in", "out", "inout", "const", "highp", "mediump", "lowp", "precise", NULL };
            int ntok = 0; char *last = NULL;
            char *copy = strdup(one);
            for (char *tk = strtok(copy, " \t\n"); tk; tk = strtok(NULL, " \t\n")) {
                bool q = false; for (int k = 0; quals[k]; k++) if (!strcmp(tk, quals[k])) q = true;
                if (!q) { ntok++; last = tk; }
            }
            Str d = { 0 };
            if (ntok >= 2 && last) {
                char *pos = strstr(one, last);
                char *lastpos = pos;
                while (pos) { lastpos = pos; pos = strstr(pos + 1, last); }
                sputn(&d, one, (size_t)(lastpos - one));
            } else {
                sput(&d, one); sput(&d, " ");
            }
            sprintf_(&d, "_mgl_a%d%s", n, suffix);
            free(copy);
            *decl = realloc(*decl, (size_t)(n + 1) * sizeof(char *));
            (*decl)[n++] = d.s;
            *c = save;
            if (!save) break;
            start = c + 1;
        }
    }
    free(p);
    return n;
}

/* Normalize one generated parameter declaration ("in vec4 _mgl_a0") for comparison:
 * collapse whitespace, default-qualify a bare type. */
static char *norm_param(const char *decl)
{
    static const char *dirs[] = { "in", "out", "inout", NULL };
    Str d = { 0 };
    char *copy = strdup(decl ? decl : "");
    bool first = true, qualified = false;
    for (char *tk = strtok(copy, " \t\n"); tk; tk = strtok(NULL, " \t\n")) {
        bool dir = false;
        for (int k = 0; dirs[k]; k++) if (!strcmp(tk, dirs[k])) dir = true;
        if (first && dir) qualified = true;
        if (!first) sput(&d, " ");
        sput(&d, tk);
        first = false;
    }
    free(copy);
    char *r;
    if (!qualified && d.n) {
        char *q = malloc(d.n + 4);
        snprintf(q, d.n + 4, "in %s", d.s ? d.s : "");
        free(d.s);
        r = q;
    } else {
        r = d.s ? d.s : strdup("");
    }
    return r;
}

static char *norm_ret(const char *ret)
{
    char *copy = strdup(ret ? ret : "");
    char *t = trim(copy);
    /* collapse interior whitespace runs */
    Str d = { 0 };
    for (char *tk = strtok(t, " \t\n"); tk; tk = strtok(NULL, " \t\n")) { if (d.n) sput(&d, " "); sput(&d, tk); }
    free(copy);
    return d.s ? d.s : strdup("");
}

/* GL 4.6 §4.7: a subroutine function's signature (return type + parameter types)
 * must match EVERY subroutine type it is declared compatible with. */
static bool sig_matches(const char *fret, const char *fparams, const char *tret, const char *tparams)
{
    char *fr = norm_ret(fret), *tr = norm_ret(tret);
    bool ok = !strcmp(fr, tr);
    free(fr); free(tr);
    if (!ok) return false;
    char **fd = NULL, **td = NULL;
    int nf = split_params(fparams, &fd), nt = split_params(tparams, &td);
    if (nf != nt) ok = false;
    for (int i = 0; ok && i < nf; i++) {
        char *a = norm_param(fd[i]), *b = norm_param(td[i]);
        if (strcmp(a, b)) ok = false;
        free(a); free(b);
    }
    for (int i = 0; i < nf; i++) free(fd ? fd[i] : NULL);
    for (int i = 0; i < nt; i++) free(td ? td[i] : NULL);
    free(fd); free(td);
    return ok;
}

void mgli_free_stage_sub(MGLIStageSub *s)
{
    if (!s) return;
    for (int i = 0; i < s->nfuncs; i++) {
        free(s->funcs[i].name);
        for (int k = 0; k < s->funcs[i].ncompat; k++) free(s->funcs[i].compat[k]);
        free(s->funcs[i].compat);
    }
    free(s->funcs);
    for (int i = 0; i < s->nunis; i++) { free(s->unis[i].name); free(s->unis[i].type_name); free(s->unis[i].compat); free(s->unis[i].selector); }
    free(s->unis);
    free(s->selection);
    memset(s, 0, sizeof *s);
}

/* Strip "#extension ..." directives for extensions glslang does not know but MGL
 * advertises (subroutine syntax is rewritten away below; arrays-of-arrays syntax works
 * once the version is raised to 430).  A #version line older than a construct the
 * source uses is raised in step (GL 4.6 accepts the shader, so the rewrite must make
 * glslang accept it too).  Returns a malloc'd copy, or NULL when nothing changed. */
char *mgli_strip_subroutine_extension(const char *src)
{
    if (!src) return NULL;
    static const char *unknown_exts[] = { "GL_ARB_shader_subroutine", "GL_ARB_arrays_of_arrays", NULL };
    bool any = false, needs_430 = false;
    for (int e = 0; unknown_exts[e]; e++) {
        for (const char *p = src; (p = strstr(p, "#extension")); ) {
            const char *ls = p;
            while (ls > src && (ls[-1] == ' ' || ls[-1] == '\t')) ls--;
            if (ls == src || ls[-1] == '\n') {
                const char *en = strchr(p, '\n');
                size_t n = en ? (size_t)(en - p) : strlen(p);
                if (memmem(p, n, unknown_exts[e], strlen(unknown_exts[e]))) {
                    any = true;
                    if (e == 1) needs_430 = true;
                }
            }
            p += 10;
        }
    }
    if (!any) return NULL;
    size_t n = strlen(src);
    char *out = malloc(n + 1);
    memcpy(out, src, n + 1);
    /* re-scan on the copy and blank each matching directive line */
    for (int e = 0; unknown_exts[e]; e++) {
        for (char *p = out; (p = strstr(p, "#extension")); ) {
            char *ls = p;
            while (ls > out && (ls[-1] == ' ' || ls[-1] == '\t')) ls--;
            if ((ls == out || ls[-1] == '\n') && memmem(p, strcspn(p, "\n"), unknown_exts[e], strlen(unknown_exts[e]))) {
                while (*p && *p != '\n') *p++ = ' ';
            } else {
                p += 10;
            }
        }
    }
    /* raise #version: "binding" layout qualifiers need 420 (glslang), arrays of
     * arrays 430.  The rewritten directive keeps the shader's profile word. */
    {
        int need = needs_430 ? 430 : 0;
        if (!need) {
            for (const char *p = src; (p = strstr(p, "layout")); p += 6) {
                const char *open = p + 6;
                while (*open == ' ' || *open == '\t') open++;
                if (*open != '(') continue;
                const char *close = strchr(open, ')');
                if (!close) continue;
                for (const char *b = open; b < close; b++) {
                    if (!strncmp(b, "binding", 7) && (b == open || !isalnum((unsigned char)b[-1])) &&
                        !isalnum((unsigned char)b[7])) { need = 420; break; }
                }
                if (need) break;
            }
        }
        if (need && !strncmp(out, "#version", 8)) {
            char *nl = strchr(out, '\n');
            size_t linelen = nl ? (size_t)(nl - out) : strlen(out);
            char line[128];
            snprintf(line, sizeof line, "%.*s", (int)(linelen < 127 ? linelen : 127), out);
            int ver = 0, pos = 0;
            char prof[24] = "", profile[24] = "core";
            if (sscanf(line, "#version %d %23s%n", &ver, prof, &pos) >= 1 && ver < need && ver >= 140) {
                if (pos > 0 && (size_t)pos < strlen(line) && prof[0] &&
                    !strcmp(trim(prof), "compatibility"))
                    snprintf(profile, sizeof profile, "compatibility");
                char head[64];
                snprintf(head, sizeof head, "#version %d %s\n", need, profile);
                char *res = malloc(strlen(head) + (nl ? strlen(nl + 1) : 0) + 2);
                strcpy(res, head);
                strcat(res, nl ? nl + 1 : "");
                free(out);
                out = res;
            }
        }
    }
    return out;
}

char *mgli_rewrite_subroutines(const char *src, MGLIStageSub *out)
{
    memset(out, 0, sizeof *out);
    if (!src || !strstr(src, "subroutine")) return NULL;
    Src s; s.n = strlen(src); s.b = dupn(src, s.n);
    build_mask(&s);

    SrType *types = NULL; int ntypes = 0;
    SrFunc *funcs = NULL; int nfuncs = 0;
    SrUni *unis = NULL; int nunis = 0;
    bool found = false;

    for (size_t i = 0; i < s.n; i++) {
        if (!s.code[i] || strncmp(s.b + i, "subroutine", 10) || (i > 0 && idc(s.b[i - 1])) || idc(s.b[i + 10])) continue;
        found = true;
        int loc = -1, index = -1;
        size_t start = layout_before(&s, i, &loc, &index);
        size_t j = skip_ws(&s, i + 10);
        size_t len;
        if (j < s.n && s.b[j] == '(') {
            /* implementation: subroutine(T1, T2) ret name(...) */
            size_t close = match_close(&s, j, '(', ')');
            char *list = dupn(s.b + j + 1, close - j - 1);
            SrFunc f = { 0 }; f.index = index;
            for (char *t = strtok(list, ", \t\n"); t; t = strtok(NULL, ", \t\n")) {
                f.compat = realloc(f.compat, (size_t)(f.ncompat + 1) * sizeof(char *));
                f.compat[f.ncompat++] = strdup(t);
            }
            free(list);
            /* function name: identifier right before the next '(' */
            size_t k = close + 1, lastid = 0, lastlen = 0, paren = s.n;
            while (k < s.n) {
                k = skip_ws(&s, k);
                if (k < s.n && s.b[k] == '(') { paren = k; break; }
                size_t a = read_ident(&s, k, &len);
                if (!len) { k++; continue; }
                lastid = a; lastlen = len; k = a + len;
            }
            f.name = dupn(s.b + lastid, lastlen);
            /* signature (for the GL §4.7 compatibility check below) */
            if (paren < s.n && lastid > close + 1) {
                char *ret = dupn(s.b + close + 1, lastid - (close + 1));
                f.ret = norm_ret(ret);
                free(ret);
                size_t pc = match_close(&s, paren, '(', ')');
                f.params = dupn(s.b + paren + 1, pc - paren - 1);
            }
            funcs = realloc(funcs, (size_t)(nfuncs + 1) * sizeof(SrFunc));
            funcs[nfuncs++] = f;
            blank(&s, start, close + 1);
            i = close;
            continue;
        }
        size_t a = read_ident(&s, j, &len);
        if (len == 7 && !strncmp(s.b + a, "uniform", 7)) {
            size_t ta = read_ident(&s, a + len, &len); char *tname = dupn(s.b + ta, len);
            size_t na = read_ident(&s, ta + len, &len); char *uname = dupn(s.b + na, len);
            size_t k = skip_ws(&s, na + len);
            /* arrays (incl. arrays of arrays): routine[4][4] */
            int dims[8] = { 0 }, ndims = 0;
            while (k < s.n && s.b[k] == '[' && ndims < 8) {
                size_t cb = match_close(&s, k, '[', ']');
                dims[ndims] = atoi(s.b + k + 1);
                if (dims[ndims] < 1) dims[ndims] = 1;
                ndims++;
                k = skip_ws(&s, cb + 1);
            }
            int size = 1;
            for (int d = 0; d < ndims; d++) size *= dims[d];
            if (!ndims) { dims[0] = 1; ndims = 1; }
            while (k < s.n && s.b[k] != ';') k++;
            SrUni u = { 0 };
            u.name = uname; u.type = tname; u.size = size; u.location = loc; u.at = start;
            u.ndims = ndims;
            for (int d = 0; d < ndims; d++) u.dims[d] = dims[d];
            unis = realloc(unis, (size_t)(nunis + 1) * sizeof(SrUni));
            unis[nunis++] = u;
            blank(&s, start, k + 1);
            i = k;
            continue;
        }
        /* type declaration: subroutine ret Name(params); */
        size_t k = j, lastid = 0, lastlen = 0;
        while (k < s.n) {
            k = skip_ws(&s, k);
            if (k < s.n && s.b[k] == '(') break;
            size_t b = read_ident(&s, k, &len);
            if (!len) { k++; continue; }
            lastid = b; lastlen = len; k = b + len;
        }
        if (k >= s.n) break;
        size_t close = match_close(&s, k, '(', ')');
        SrType t;
        t.name = dupn(s.b + lastid, lastlen);
        char *ret = dupn(s.b + j, lastid - j); t.ret = strdup(trim(ret)); free(ret);
        t.params = dupn(s.b + k + 1, close - k - 1);
        types = realloc(types, (size_t)(ntypes + 1) * sizeof(SrType));
        types[ntypes++] = t;
        size_t semi = close; while (semi < s.n && s.b[semi] != ';') semi++;
        blank(&s, i, semi + 1);
        i = semi;
    }
    if (!found) { free(s.b); free(s.code); return NULL; }

    /* subroutine indices: explicit first, then lowest unused */
    for (int f = 0; f < nfuncs; f++) if (funcs[f].index < 0) {
        for (int c = 0; ; c++) {
            bool used = false; for (int g = 0; g < nfuncs; g++) if (funcs[g].index == c) used = true;
            if (!used) { funcs[f].index = c; break; }
        }
    }
    /* uniform locations: explicit first, then lowest free range */
    for (int u = 0; u < nunis; u++) if (unis[u].location < 0) {
        for (int c = 0; ; c++) {
            bool clash = false;
            for (int v = 0; v < nunis; v++) if (v != u && unis[v].location >= 0 &&
                c < unis[v].location + unis[v].size && unis[v].location < c + unis[u].size) clash = true;
            if (!clash) { unis[u].location = c; break; }
        }
    }

    /* output */
    Str o = { 0 };
    size_t ui = 0;
    /* uniforms sorted by position (they are recorded in source order) */
    for (size_t i = 0; i < s.n; ) {
        if (ui < (size_t)nunis && unis[ui].at == i) {
            SrUni *u = &unis[ui];
            SrType *t = NULL; for (int k = 0; k < ntypes; k++) if (!strcmp(types[k].name, u->type)) t = &types[k];
            /* No layout(location=...): explicit uniform locations need GLSL 430 /
             * GL_ARB_explicit_uniform_location and the selector is written by name
             * (write_selectors), so a plain default-block uniform is enough. */
            if (u->size > 1) sprintf_(&o, "uniform int _mgl_sr_%s[%d]; ", u->name, u->size);
            else sprintf_(&o, "uniform int _mgl_sr_%s; ", u->name);
            char **pd = NULL; int np = split_params(t ? t->params : "", &pd);
            sprintf_(&o, "%s _mgl_sr_call_%s(", t ? t->ret : "void", u->name);
            bool first = true;
            if (u->size > 1) { sput(&o, "uint _mgl_i"); first = false; }
            for (int k = 0; k < np; k++) { if (!first) sput(&o, ", "); sput(&o, pd[k]); first = false; free(pd[k]); }
            free(pd);
            sput(&o, ");");
            ui++;
        }
        /* call sites */
        if (s.code[i] && idc(s.b[i]) && (i == 0 || (!idc(s.b[i - 1]) && s.b[i - 1] != '.'))) {
            size_t j = i; while (j < s.n && idc(s.b[j])) j++;
            int hit = -1;
            for (int k = 0; k < nunis; k++) if (strlen(unis[k].name) == j - i && !strncmp(s.b + i, unis[k].name, j - i)) hit = k;
            if (hit >= 0) {
                size_t k = skip_ws(&s, j);
                /* index expressions (arrays of arrays: routine[j][i]) */
                size_t ia[8], ib[8];
                int nidx = 0;
                while (k < s.n && s.b[k] == '[' && nidx < 8) {
                    size_t cb = match_close(&s, k, '[', ']');
                    ia[nidx] = k + 1; ib[nidx] = cb; nidx++;
                    k = skip_ws(&s, cb + 1);
                }
                if (k < s.n && s.b[k] == '(') {
                    size_t after = skip_ws(&s, k + 1);
                    bool noargs = after < s.n && s.b[after] == ')';
                    sprintf_(&o, "_mgl_sr_call_%s(", unis[hit].name);
                    if (unis[hit].size > 1) {
                        if (!nidx) {
                            sput(&o, "0u");
                        } else {
                            /* flatten routine[j][i] -> j * dims[1] + i */
                            for (int m = 0; m < nidx; m++) {
                                if (m) sput(&o, " + ");
                                sputn(&o, s.b + ia[m], ib[m] - ia[m]);
                                for (int d = m + 1; d < nidx; d++) sprintf_(&o, " * %d", unis[hit].dims[d]);
                            }
                        }
                        if (!noargs) sput(&o, ", ");
                    }
                    i = k + 1;
                    continue;
                }
                /* member access: .length() on (a slice of) the uniform array */
                if (k < s.n && s.b[k] == '.') {
                    size_t mlen = 0;
                    size_t m = read_ident(&s, k + 1, &mlen);
                    if (mlen == 6 && !strncmp(s.b + m, "length", 6)) {
                        size_t p = skip_ws(&s, m + mlen);
                        if (p < s.n && s.b[p] == '(' && nidx < unis[hit].ndims) {
                            sprintf_(&o, "%d", unis[hit].dims[nidx]);
                            i = match_close(&s, p, '(', ')') + 1;   /* skip ".length()" */
                            continue;
                        }
                    }
                }
            }
            sputn(&o, s.b + i, j - i);
            i = j;
            continue;
        }
        sputn(&o, s.b + i, 1);
        i++;
    }

    /* dispatchers */
    sput(&o, "\n");
    for (int u = 0; u < nunis; u++) {
        SrType *t = NULL; for (int k = 0; k < ntypes; k++) if (!strcmp(types[k].name, unis[u].type)) t = &types[k];
        const char *ret = t ? t->ret : "void";
        bool is_void = !strcmp(ret, "void");
        char **pd = NULL; int np = split_params(t ? t->params : "", &pd);
        sprintf_(&o, "%s _mgl_sr_call_%s(", ret, unis[u].name);
        bool first = true;
        if (unis[u].size > 1) { sput(&o, "uint _mgl_i"); first = false; }
        for (int k = 0; k < np; k++) { if (!first) sput(&o, ", "); sput(&o, pd[k]); first = false; free(pd[k]); }
        free(pd);
        Str args = { 0 }; sput(&args, "");
        for (int k = 0; k < np; k++) sprintf_(&args, "%s_mgl_a%d", k ? ", " : "", k);
        sput(&o, ") {\n");
        if (unis[u].size > 1) sprintf_(&o, "    switch (_mgl_sr_%s[_mgl_i]) {\n", unis[u].name);
        else sprintf_(&o, "    switch (_mgl_sr_%s) {\n", unis[u].name);
        int first_compat = -1;
        for (int f = 0; f < nfuncs; f++) {
            bool ok = false; for (int c = 0; c < funcs[f].ncompat; c++) if (!strcmp(funcs[f].compat[c], unis[u].type)) ok = true;
            if (!ok) continue;
            if (first_compat < 0) first_compat = f;
            if (is_void) sprintf_(&o, "    case %d: %s(%s); return;\n", funcs[f].index, funcs[f].name, args.s);
            else sprintf_(&o, "    case %d: return %s(%s);\n", funcs[f].index, funcs[f].name, args.s);
        }
        if (first_compat >= 0) {
            if (is_void) sprintf_(&o, "    default: %s(%s); return;\n", funcs[first_compat].name, args.s);
            else sprintf_(&o, "    default: return %s(%s);\n", funcs[first_compat].name, args.s);
            sput(&o, "    }\n}\n");
        } else {
            sput(&o, "    default: break;\n    }\n");
            if (!is_void) sprintf_(&o, "    return %s(0);\n", ret);
            sput(&o, "}\n");
        }
        free(args.s);
    }

    /* GL §4.7: a subroutine's signature must match every type it is compatible
     * with; a mismatch is a compile-time error, surfaced as #error in the
     * rewritten source (KHR-GL46.shader_subroutine.subroutines_incompatible_with_subroutine_type) */
    for (int f = 0; f < nfuncs; f++) {
        if (!funcs[f].ret) continue;
        for (int c = 0; c < funcs[f].ncompat; c++) {
            SrType *t = NULL;
            for (int k = 0; k < ntypes; k++) if (!strcmp(types[k].name, funcs[f].compat[c])) t = &types[k];
            if (!t) continue;
            if (!sig_matches(funcs[f].ret, funcs[f].params, t->ret, t->params))
                sprintf_(&o, "\n#error subroutine '%s': signature does not match subroutine type '%s'\n",
                         funcs[f].name, t->name);
        }
    }

    /* reflection record */
    out->nfuncs = nfuncs;
    out->funcs = calloc((size_t)nfuncs + 1, sizeof(MGLISubFunc));
    for (int f = 0; f < nfuncs; f++) {
        out->funcs[f].name = funcs[f].name; out->funcs[f].index = funcs[f].index;
        out->funcs[f].compat = funcs[f].compat; out->funcs[f].ncompat = funcs[f].ncompat;
        free(funcs[f].ret); free(funcs[f].params);
    }
    out->nunis = nunis;
    out->unis = calloc((size_t)nunis + 1, sizeof(MGLISubUniform));
    for (int u = 0; u < nunis; u++) {
        MGLISubUniform *d = &out->unis[u];
        d->name = unis[u].name; d->type_name = unis[u].type; d->location = unis[u].location; d->size = unis[u].size;
        d->ndims = unis[u].ndims;
        for (int m = 0; m < unis[u].ndims && m < 8; m++) d->dims[m] = unis[u].dims[m];
        char sel[300]; snprintf(sel, sizeof sel, "_mgl_sr_%s", unis[u].name); d->selector = strdup(sel);
        d->compat = calloc((size_t)nfuncs + 1, sizeof(int));
        for (int f = 0; f < nfuncs; f++)
            for (int c = 0; c < funcs[f].ncompat; c++) if (!strcmp(funcs[f].compat[c], unis[u].type)) { d->compat[d->ncompat++] = funcs[f].index; break; }
        if (d->location + d->size > out->nlocs) out->nlocs = d->location + d->size;
    }
    for (int k = 0; k < ntypes; k++) { free(types[k].name); free(types[k].ret); free(types[k].params); }
    free(types); free(funcs); free(unis);
    free(s.b); free(s.code);
    return o.s;
}
