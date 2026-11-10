/*
 * compat_fbfetch.c — GL_EXT_shader_framebuffer_fetch (compat domain, r101).
 *
 * "the fragment shader can read the current contents of the framebuffer"
 * through `inout` fragment outputs and the gl_LastFragData[] builtin.
 *
 * Metal mechanism: a fragment entry-point PARAMETER annotated [[color(n)]]
 * reads the current tile-memory value of color attachment n (Apple-GPU
 * framebuffer fetch; the same mechanism GL_KHR_blend_equation_advanced uses,
 * see mgl_advblend_rewrite). MGL owns the whole GLSL -> MSL path, so the
 * extension is lowered in two passes:
 *
 *   1. GLSL (mgl_fbfetch_glsl_compat, called from mglCompileShader):
 *        - the #extension directive (unknown to glslang) is blanked,
 *        - `layout(location=n) inout T o;` becomes `out` (glslang accepts
 *          reads of an out variable; SPIR-V OpLoad of an Output) — the
 *          MSL local `out.<member>` then starts at zero and is preloaded
 *          with the fetched colour,
 *        - gl_LastFragData becomes the default-block uniform array mgl_lfd[]
 *          (highp: required by ES fragment shaders, ignored by desktop GLSL).
 *      In ES 1.00 / GLSL < 130 nothing is rewritten, so `inout` at global
 *      scope stays the compile error the spec demands (dEQP negcov
 *      framebuffer_fetch.invalid_inout_version).
 *   2. MSL (mgl_fbfetch_fix_msl, called from compile_stage): the fragment
 *      signature gains `<type> mgl_fbfetch_n [[color(n)]]` parameters and the
 *      first statement of the body preloads every fetched output:
 *      `out.<member> = mgl_fbfetch_n;`. mgl_lfd[i] reads are rewritten to the
 *      matching parameter (literal index) or an index select (dynamic).
 *
 * State: the scan result rides on Spirv.fbfetch (bit per fetched location +
 * the gl_LastFragData range) and is recomputed from the shader source at
 * every link, so no extra per-shader state is needed.
 *
 * Not covered: EXT_shader_framebuffer_fetch_non_coherent
 * (glFramebufferFetchBarrierEXT — every MGL draw is its own Metal encoder,
 * so MGL is always coherent), ARB_fragment_coord_conventions-style float
 * attachment reads of gl_LastFragData (the builtin is vec4; integer
 * attachments fetch garbage, as in GL ES).
 */

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdbool.h>
#include "glm_context.h"
#include "compat_fbfetch.h"

#define MGL_FBF_EXT_NAME   "GL_EXT_shader_framebuffer_fetch"
#define MGL_FBF_LFD_NAME   "gl_LastFragData"
#define MGL_FBF_LFD_MSL    "mgl_lfd"
/* the extension's array is gl_MaxDrawBuffers long; MGL reports 8 */
#define MGL_FBF_MAX_LFD    8
#define MGL_FBF_MAX_INOUT  16

/* code layout (Spirv.fbfetch):
 *   bits 0..7   colour locations fetched through `inout' outputs
 *   bits 8..11  gl_LastFragData: 0 = unused, 1..8 = highest literal index + 1,
 *               15 = dynamic index (fetch every attachment)
 *   bit  31     the extension directive was present and enabled
 */
#define MGL_FBF_LOC_MASK   0x000000ffu
#define MGL_FBF_LFD_SHIFT  8
#define MGL_FBF_LFD_MASK   0x00000f00u
#define MGL_FBF_LFD_DYNAMIC 0xfu
#define MGL_FBF_ENABLED    0x80000000u

static bool fbf_idc(int c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static bool fbf_word(const char *p, const char *w)
{
    size_t n = strlen(w);
    return strncmp(p, w, n) == 0 && !fbf_idc((unsigned char)p[n]);
}

/* span of the identifier at/after p */
static const char *fbf_ident_end(const char *p)
{
    while (*p && fbf_idc((unsigned char)*p))
        p++;
    return p;
}

typedef struct {
    uint32_t code;          /* MGL_FBF_* bits */
    int      has_directive; /* the #extension line exists (any behaviour) */
    int      enabled;       /* behaviour is not disable */
    int      rewrite_io;    /* inout outputs may be rewritten (GLSL version) */
    size_t   dir_start, dir_end;    /* directive span, newline excluded */
    size_t   inject;                /* offset just past the #version line */
    size_t   inout_at[MGL_FBF_MAX_INOUT];   /* 'inout' token offsets */
    uint32_t inout_loc[MGL_FBF_MAX_INOUT];  /* location (explicit or ordinal) */
    int      n_inout;
    size_t   lfd_at[64];
    int      n_lfd;
    int      lfd_literal;   /* highest literal index seen, or -1 (dynamic) */
} FbfScan;

/* location = N inside the declaration head before an `inout' token */
static int fbf_head_location(const char *src, size_t inout_pos, uint32_t ordinal)
{
    size_t start = inout_pos;
    while (start > 0 && src[start - 1] != ';' && src[start - 1] != '{' &&
           src[start - 1] != '}' && src[start - 1] != '\n')
    {
        /* preprocessor lines belong to earlier declarations */
        if (src[start - 1] == '\n')
            break;
        start--;
    }
    /* find 'location' <ws>* '=' <ws>* <int> */
    for (size_t i = start; i < inout_pos; i++)
    {
        if (src[i] == 'l' && fbf_word(src + i, "location"))
        {
            size_t q = i + 8;
            while (q < inout_pos && (src[q] == ' ' || src[q] == '\t'))
                q++;
            if (q < inout_pos && src[q] == '=')
            {
                q++;
                while (q < inout_pos && (src[q] == ' ' || src[q] == '\t'))
                    q++;
                if (q < inout_pos && isdigit((unsigned char)src[q]))
                {
                    int v = 0;
                    while (q < inout_pos && isdigit((unsigned char)src[q]))
                        v = v * 10 + (src[q++] - '0');
                    return v < MGL_FBF_MAX_LFD ? v : -1;
                }
            }
            i = q;
        }
    }
    return (int)ordinal;
}

static void fbf_scan(const char *src, FbfScan *s)
{
    size_t i = 0;
    int depth = 0;              /* braces + parens: global scope is depth 0 */
    int in_block = 0, in_line = 0;
    int version = 0, es = 0;

    memset(s, 0, sizeof *s);
    s->lfd_literal = -2;        /* -2 = unused, -1 = dynamic, >=0 literal max */
    s->inject = 0;

    while (src[i])
    {
        char c = src[i];

        if (in_line)
        {
            if (c == '\n') in_line = 0;
            i++;
            continue;
        }
        if (in_block)
        {
            if (c == '*' && src[i + 1] == '/') { in_block = 0; i += 2; continue; }
            i++;
            continue;
        }
        if (c == '/' && src[i + 1] == '/') { in_line = 1; i += 2; continue; }
        if (c == '/' && src[i + 1] == '*') { in_block = 1; i += 2; continue; }

        if (c == '#' && (i == 0 || src[i - 1] == '\n'))
        {
            size_t e = i;
            while (src[e] && src[e] != '\n')
                e++;
            /* directive text between # and end of line (comments aside) */
            if (fbf_word(src + i + 1, "version"))
            {
                int v = 0;
                char prof[32] = {0};
                sscanf(src + i, "#version %d %31s", &v, prof);
                if (v) { version = v; es = (strcmp(prof, "es") == 0); }
                s->inject = src[e] ? e + 1 : e;
            }
            else if (fbf_word(src + i + 1, "extension"))
            {
                size_t q = i + 10;
                while (src[q] == ' ' || src[q] == '\t') q++;
                if (fbf_word(src + q, MGL_FBF_EXT_NAME))
                {
                    size_t r = q + strlen(MGL_FBF_EXT_NAME);
                    while (r < e && (src[r] == ' ' || src[r] == '\t')) r++;
                    s->has_directive = 1;
                    s->enabled = 1;      /* no ': disable' found below */
                    s->dir_start = i;
                    s->dir_end = e;
                    if (r < e && src[r] == ':')
                    {
                        r++;
                        while (r < e && (src[r] == ' ' || src[r] == '\t')) r++;
                        if (fbf_word(src + r, "disable"))
                            s->enabled = 0;
                    }
                }
            }
            i = e;
            continue;
        }

        if (c == '{' || c == '(') { depth++; i++; continue; }
        if (c == '}' || c == ')') { if (depth) depth--; i++; continue; }

        if (isalpha((unsigned char)c) || c == '_')
        {
            const char *id = src + i;
            const char *end = fbf_ident_end(id);
            size_t n = (size_t)(end - id);

            if (depth == 0 && n == 5 && strncmp(id, "inout", 5) == 0 &&
                !(i > 0 && (fbf_idc((unsigned char)src[i - 1]) || src[i - 1] == '.')) &&
                s->n_inout < MGL_FBF_MAX_INOUT)
            {
                int loc = fbf_head_location(src, i, (uint32_t)s->n_inout);
                if (loc >= 0)
                {
                    s->inout_at[s->n_inout] = i;
                    s->inout_loc[s->n_inout] = (uint32_t)loc;
                    s->n_inout++;
                }
            }
            else if (n == strlen(MGL_FBF_LFD_NAME) &&
                     strncmp(id, MGL_FBF_LFD_NAME, n) == 0 &&
                     !(i > 0 && (fbf_idc((unsigned char)src[i - 1]) || src[i - 1] == '.')))
            {
                if (s->n_lfd < 64) s->lfd_at[s->n_lfd++] = i;
                /* trailing [ <index> ] */
                size_t q = (size_t)(end - src);
                while (src[q] == ' ' || src[q] == '\t') q++;
                if (src[q] == '[')
                {
                    q++;
                    while (src[q] == ' ' || src[q] == '\t') q++;
                    if (isdigit((unsigned char)src[q]))
                    {
                        int v = 0;
                        while (isdigit((unsigned char)src[q])) v = v * 10 + (src[q++] - '0');
                        if (v >= MGL_FBF_MAX_LFD) v = MGL_FBF_MAX_LFD - 1;
                        if (s->lfd_literal < v) s->lfd_literal = v;
                    }
                    else
                    {
                        s->lfd_literal = -1;     /* dynamic index */
                    }
                }
                /* else: array decay (`gl_LastFragData' passed whole) — dynamic */
            }
            i = (size_t)(end - src);
            continue;
        }

        i++;
    }

    /* ES >= 3.00 and desktop >= 3.30 carry the inout-output syntax */
    s->rewrite_io = s->enabled &&
                    ((es && version >= 300) || (!es && (version == 0 || version >= 330)));

    if (s->enabled && s->rewrite_io)
        for (int k = 0; k < s->n_inout; k++)
            s->code |= 1u << (s->inout_loc[k] & 7);

    if (s->enabled && s->n_lfd)
    {
        int max = s->lfd_literal >= 0 ? s->lfd_literal : MGL_FBF_MAX_LFD - 1;
        s->code |= ((uint32_t)(max + 1) & 0xf) << MGL_FBF_LFD_SHIFT;
    }

    if (s->enabled)
        s->code |= MGL_FBF_ENABLED;
}

/* public scan used at link time (Spirv.fbfetch code) */
uint32_t mgl_fbfetch_scan_source(const char *src)
{
    FbfScan s;
    if (!src)
        return 0;
    fbf_scan(src, &s);
    return s.code;
}

/*
 * GLSL pre-pass. Returns a malloc'd rewritten fragment source, or NULL when
 * the source has nothing to do (the common case — one strstr-style check
 * keeps the hot path cheap).
 */
char *mgl_fbfetch_glsl_compat(GLuint type, const char *src)
{
    FbfScan s;
    char *out;
    size_t cap, o = 0;
    size_t last = 0;

    if (!src || type != GL_FRAGMENT_SHADER)
        return NULL;
    if (!strstr(src, MGL_FBF_EXT_NAME))
        return NULL;

    fbf_scan(src, &s);
    if (!s.has_directive && !s.n_lfd)
        return NULL;

    /* worst case: directive blanked, inout->out, gl_LastFragData->mgl_lfd,
     * plus the injected uniform line */
    cap = strlen(src) + 64;
    out = (char *)malloc(cap);
    if (!out)
        return NULL;

    /* one ordered sweep: blank the directive, rename tokens, inject */
    for (;;)
    {
        size_t next = strlen(src);
        size_t emit_end = next;
        const char *rep = NULL;
        size_t rep_len = 0;

        if (s.has_directive && s.dir_start >= last && s.dir_start < next)
        {
            next = s.dir_start;
            rep = "";
            rep_len = 0;
            emit_end = s.dir_end;       /* blank the whole directive line */
        }
        for (int k = 0; k < s.n_inout; k++)
        {
            if (!s.rewrite_io) break;
            if (s.inout_at[k] >= last && s.inout_at[k] < next)
            {
                next = s.inout_at[k];
                rep = "out";
                rep_len = 3;
                emit_end = s.inout_at[k] + 5;
            }
        }
        for (int k = 0; k < s.n_lfd; k++)
        {
            if (!s.enabled) break;
            if (s.lfd_at[k] >= last && s.lfd_at[k] < next)
            {
                next = s.lfd_at[k];
                rep = MGL_FBF_LFD_MSL;
                rep_len = strlen(MGL_FBF_LFD_MSL);
                emit_end = s.lfd_at[k] + strlen(MGL_FBF_LFD_NAME);
            }
        }

        if (!rep)
            break;

        if (o + (next - last) + rep_len + 1 > cap)
        {
            char *grown;
            cap = cap * 2 + (next - last) + rep_len + 1;
            grown = (char *)realloc(out, cap);
            if (!grown) { free(out); return NULL; }
            out = grown;
        }
        memcpy(out + o, src + last, next - last);
        o += next - last;
        if (rep_len)
        {
            memcpy(out + o, rep, rep_len);
            o += rep_len;
        }
        last = emit_end;

        /* the injected lfd uniform goes right after #version */
        if (s.n_lfd && s.enabled && last >= s.inject && s.inject != (size_t)-1)
        {
            static const char decl[] = "\nuniform highp vec4 " MGL_FBF_LFD_MSL "[8];";
            if (o + sizeof decl > cap)
            {
                char *grown = (char *)realloc(out, cap + sizeof decl);
                if (!grown) { free(out); return NULL; }
                out = grown;
                cap += sizeof decl;
            }
            memcpy(out + o, decl, sizeof decl - 1);
            o += sizeof decl - 1;
            s.inject = (size_t)-1;     /* once */
        }
    }

    if (o + (strlen(src) - last) + 1 > cap)
    {
        char *grown = (char *)realloc(out, o + (strlen(src) - last) + 1);
        if (!grown) { free(out); return NULL; }
        out = grown;
    }
    memcpy(out + o, src + last, strlen(src) - last);
    o += strlen(src) - last;
    out[o] = 0;
    return out;
}

/* ---- MSL patch ---- */

typedef struct {
    char *s;
    size_t len, cap;
} FbfStr;

static void fbf_put(FbfStr *b, const char *p, size_t n)
{
    if (b->len + n + 1 > b->cap)
    {
        b->cap = (b->len + n + 1) * 2;
        b->s = (char *)realloc(b->s, b->cap);
    }
    memcpy(b->s + b->len, p, n);
    b->len += n;
    b->s[b->len] = 0;
}

static void fbf_puts(FbfStr *b, const char *p)
{
    fbf_put(b, p, strlen(p));
}

static const char *fbf_match(const char *open)
{
    int depth = 0;
    for (const char *p = open; *p; p++)
    {
        if (*p == '(' || *p == '{') depth++;
        else if (*p == ')' || *p == '}')
        {
            if (--depth == 0)
                return p;
        }
    }
    return NULL;
}

/* find `<entry>(' preceded by an identifier (the entry definition) */
static const char *fbf_find_entry(const char *msl, const char *entry)
{
    size_t n = strlen(entry);
    const char *p = msl;
    while ((p = strstr(p, entry)) != NULL)
    {
        if ((p == msl || !fbf_idc((unsigned char)p[-1])) && p[n] == '(')
            return p;
        p += n;
    }
    return NULL;
}

/* the identifier before p (skipping whitespace) */
static const char *fbf_ident_before(const char *base, const char *p)
{
    while (p > base && (p[-1] == ' ' || p[-1] == '\t' || p[-1] == '\n'))
        p--;
    const char *e = p;
    while (p > base && fbf_idc((unsigned char)p[-1]))
        p--;
    return p == e ? NULL : p;
}

/* one output-struct member: type + name at colour attachment loc */
typedef struct {
    int loc;
    char type[32];
    char name[128];
} FbfMember;

static int fbf_struct_members(const char *msl, const char *struct_name, FbfMember *out, int max)
{
    char decl[192];
    int n = 0;
    snprintf(decl, sizeof decl, "struct %s", struct_name);
    const char *st = strstr(msl, decl);
    if (!st)
        return 0;
    const char *open = strchr(st + strlen(decl), '{');
    if (!open)
        return 0;
    const char *close = fbf_match(open);
    if (!close)
        return 0;

    const char *p = open + 1;
    while (p < close)
    {
        const char *semi = memchr(p, ';', (size_t)(close - p));
        if (!semi)
            break;
        const char *col = p;
        while ((col = strstr(col, "color(")) != NULL && col < semi)
        {
            int loc = atoi(col + 6);
            const char *br = col;
            while (br > open && br[-1] == ' ')
                br--;
            if (br - open >= 2 && br[-1] == '[' && br[-2] == '[')
            {
                const char *name = fbf_ident_before(msl, br - 2);
                if (name && name > open)
                {
                    const char *type_start = p;
                    while (type_start < name && (*type_start == ' ' || *type_start == '\t' || *type_start == '\n'))
                        type_start++;
                    size_t tlen = (size_t)(name - type_start);
                    size_t nlen = (size_t)(fbf_ident_end(name) - name);
                    if (tlen > 0 && tlen < sizeof out->type && nlen > 0 && nlen < sizeof out->name && n < max)
                    {
                        memcpy(out[n].type, type_start, tlen);
                        out[n].type[tlen] = 0;
                        memcpy(out[n].name, name, nlen);
                        out[n].name[nlen] = 0;
                        out[n].loc = loc;
                        n++;
                    }
                }
            }
            col += 6;
        }
        p = semi + 1;
    }
    return n;
}

/*
 * Patch the fragment MSL: add [[color(n)]] input parameters and preload the
 * output locals / mgl_lfd reads. Returns malloc'd MSL or NULL when the entry
 * shape is not SPIRV-Cross's (caller keeps the original).
 */
char *mgl_fbfetch_fix_msl(const char *msl, const char *entry, uint32_t code)
{
    FbfStr b = { NULL, 0, 0 };
    FbfMember members[8];
    int n_members;
    uint32_t loc_mask = code & MGL_FBF_LOC_MASK;
    uint32_t lfd_n = (code & MGL_FBF_LFD_MASK) >> MGL_FBF_LFD_SHIFT;
    char param[256];
    char init[512];
    if (!msl || !entry || !code || !(code & MGL_FBF_ENABLED))
        return NULL;

    const char *name = fbf_find_entry(msl, entry);
    if (!name)
        return NULL;
    /* `fragment <ret> <entry>(' */
    const char *ret = fbf_ident_before(msl, name);
    if (!ret)
        return NULL;
    const char *kw = fbf_ident_before(msl, ret);
    if (!kw || strncmp(kw, "fragment", 8) != 0 || fbf_idc((unsigned char)kw[8]))
        return NULL;

    const char *params_open = name + strlen(entry);
    while (*params_open == ' ' || *params_open == '\t' || *params_open == '\n')
        params_open++;
    if (*params_open != '(')
        return NULL;
    const char *params_close = fbf_match(params_open);
    if (!params_close)
        return NULL;
    const char *body_open = params_close + 1;
    while (*body_open == ' ' || *body_open == '\t' || *body_open == '\n')
        body_open++;
    if (*body_open != '{')
        return NULL;

    /* a shader with no default-block uniforms has an EMPTY parameter list
     * (SPIRV-Cross emits `name()'): the added fetch parameters then need no
     * separator before the first one */
    bool empty_params = false;
    {
        const char *q = params_open + 1;
        while (q < params_close && (*q == ' ' || *q == '\t' || *q == '\n'))
            q++;
        empty_params = (q >= params_close);
    }
    bool any_added = false;

    char retname[128];
    size_t rn = (size_t)(fbf_ident_end(ret) - ret);
    if (rn == 0 || rn >= sizeof retname)
        return NULL;
    memcpy(retname, ret, rn);
    retname[rn] = 0;
    n_members = fbf_struct_members(msl, retname, members, 8);
    if (!n_members && !lfd_n)
        return NULL;

    /* find `<rettype> out = ... ;' — the local whose members are preloaded */
    const char *ins = NULL;
    {
        char pat[160];
        snprintf(pat, sizeof pat, "%s ", retname);
        const char *d = strstr(body_open, pat);
        while (d)
        {
            const char *eol = strchr(d, ';');
            if (!eol)
                break;
            /* `RETTYPE out = {};' or `RETTYPE out;' */
            const char *id = d + strlen(pat);
            while (*id == ' ' || *id == '\t')
                id++;
            if (isalpha((unsigned char)*id) || *id == '_')
            {
                ins = eol + 1;
                break;
            }
            d = strstr(eol, pat);
        }
        if (!ins)
        {
            free(b.s);
            return NULL;    /* no output local: not SPIRV-Cross's shape */
        }
    }

    /* copy the signature + parameters, then up to the preload point */
    fbf_put(&b, msl, (size_t)(params_close - msl));

    for (int n = 0; n <= 7; n++)
    {
        bool want = false;
        const char *type = "float4";
        char typebuf[32];

        if (loc_mask & (1u << n))
        {
            for (int k = 0; k < n_members; k++)
                if (members[k].loc == n)
                {
                    /* struct member types are the MSL scalar-vector types a
                     * [[color]] argument accepts (float4/int4/uint4/...) */
                    snprintf(typebuf, sizeof typebuf, "%s", members[k].type);
                    type = typebuf;
                    want = true;
                    break;
                }
        }
        else if (lfd_n && (uint32_t)n < lfd_n)
        {
            want = true;    /* gl_LastFragData is vec4 */
        }
        if (!want)
            continue;
        /* an existing parameter already reads this attachment (advanced
         * blending): reuse rather than duplicate the attribute */
        char attr[32];
        snprintf(attr, sizeof attr, "color(%d)", n);
        if (memmem(params_open, (size_t)(params_close - params_open), attr, strlen(attr)))
            continue;

        /* the separator is only needed when a parameter is already there
         * (uniform struct or an earlier fetch parameter) */
        snprintf(param, sizeof param, "%s%s mgl_fbfetch_%d [[color(%d)]]",
                 (!any_added && empty_params) ? "" : ", ", type, n, n);
        fbf_puts(&b, param);
        any_added = true;
    }

    if (!any_added)
    {
        free(b.s);
        return NULL;    /* every wanted attachment was already fetched */
    }

    fbf_put(&b, params_close, (size_t)(ins - params_close));

    /* preload: `out.<member> = mgl_fbfetch_n;' */
    init[0] = 0;
    {
        size_t ilen = 0;
        for (int n = 0; n <= 7; n++)
        {
            if (!(loc_mask & (1u << n)))
                continue;
            for (int k = 0; k < n_members; k++)
                if (members[k].loc == n)
                {
                    char one[192];
                    size_t on = (size_t)snprintf(one, sizeof one, "\n    out.%s = mgl_fbfetch_%d;",
                                                 members[k].name, n);
                    if (ilen + on + 1 > sizeof init)
                        break;
                    memcpy(init + ilen, one, on);
                    ilen += on;
                }
        }
        init[ilen] = 0;
    }
    fbf_puts(&b, init);
    fbf_put(&b, ins, strlen(ins));

    /* mgl_lfd[i] reads -> the fetched parameters. The array member access is
     * `mgl_DefaultUniforms_1.mgl_lfd[i]': rewrite the whole postfix. */
    if (lfd_n)
    {
        char *fixed;
        size_t from = 0;

        while ((fixed = strstr(b.s + from, MGL_FBF_LFD_MSL "[")) != NULL)
        {
            const char *acc = fixed;
            /* back over `.mgl_lfd[' — the member access start (acc sits on
             * the first identifier char: `acc == fixed' is fine when the char
             * before the token is the '.') */
            while (acc > b.s && fbf_idc((unsigned char)acc[-1]))
                acc--;
            if (acc == b.s || acc[-1] != '.')
            {
                from = (size_t)(fixed - b.s) + strlen(MGL_FBF_LFD_MSL);
                continue;
            }
            const char *member = acc;
            /* member start includes the uniform reference before the dot */
            while (member > b.s && (fbf_idc((unsigned char)member[-1]) || member[-1] == '.'))
                member--;
            const char *close = strchr(fixed, ']');
            if (!close)
                break;
            const char *idx = fixed + strlen(MGL_FBF_LFD_MSL) + 1;
            while (idx < close && (*idx == ' ' || *idx == '\t'))
                idx++;

            char sel[1024];
            size_t sl = 0;
            bool literal = isdigit((unsigned char)*idx);
            int li = -1;
            if (literal)
            {
                li = atoi(idx);
                if (li < 0 || li > 7) li = 0;
            }
            if (literal)
            {
                sl = (size_t)snprintf(sel, sizeof sel, "mgl_fbfetch_%d", li);
            }
            else
            {
                size_t ilen = (size_t)(close - idx);
                char iexpr[128];
                if (ilen >= sizeof iexpr) ilen = sizeof iexpr - 1;
                memcpy(iexpr, idx, ilen);
                iexpr[ilen] = 0;
                sl = 0;
                for (uint32_t n = 0; n < lfd_n && n < 8; n++)
                {
                    if (n)
                    {
                        memcpy(sel + sl, " : ", 3);
                        sl += 3;
                    }
                    sl += (size_t)snprintf(sel + sl, sizeof sel - sl, "(%s) == %uu ? mgl_fbfetch_%u", iexpr, n, n);
                }
                if (lfd_n)
                {
                    memcpy(sel + sl, " : float4(0.0)", 14);
                    sl += 14;
                }
                /* wrap the chain */
                memmove(sel + 1, sel, sl);
                sel[0] = '(';
                sel[sl + 1] = ')';
                sl += 2;
            }

            /* splice: [member_start, close+1) -> sel */
            size_t at = (size_t)(member - b.s);
            size_t cut = (size_t)(close + 1 - member);
            size_t tail = b.len - (at + cut);
            char *res = (char *)malloc(b.len - cut + sl + 1);
            if (!res) { free(b.s); return NULL; }
            memcpy(res, b.s, at);
            memcpy(res + at, sel, sl);
            memcpy(res + at + sl, member + cut, tail);
            res[at + sl + tail] = 0;
            free(b.s);
            b.s = res;
            b.len = at + sl + tail;
            b.cap = b.len + 1;
            from = at + sl;
        }
    }

    return b.s;
}
