/*
 * compat_border.c — GL_CLAMP_TO_BORDER with a border colour Metal cannot express.
 *
 * Metal samplers offer exactly three fixed border colours (transparent black,
 * opaque black, opaque white) for MTLSamplerAddressModeClampToBorderColor, so a
 * GL border colour like int (255,255,255,255) on an R32I texture — which the
 * hardware clamps to 1 — can never come back right.  This is emulated the way
 * the whole compat domain works: the *program's* MSL is rewritten, per draw, so
 * the affected sample calls compute GL's border weight themselves:
 *
 *   weight(axis) = NEAREST : 0 outside [0,1], 1 inside
 *                  LINEAR  : clamp(0.5 + min(c, 1-c) * size, 0, 1)   (GL 4.6 8.14)
 *
 * A full border hit returns the exact colour; a partial (LINEAR, half-texel)
 * hit is compensated against the colour the Metal sampler actually returns:
 * sample() with hardware border B_hw yields (1-b)*S_in + b*B_hw, and GL wants
 * (1-b)*S_in + b*BC, so r + b*(BC - B_hw) is exact for any B_hw.
 *
 * The table of slots needing emulation (mgl_border_scan) is derived from the
 * currently bound texture + effective sampler of each program resource; the
 * MSL variant is compiled lazily and cached by the renderer (exactly like the
 * rstate PSS variant).  When no active unit uses CLAMP_TO_BORDER the whole
 * path costs two integer compares per draw.
 *
 * Known gaps (documented in agents/compat_report.md): shadow samplers
 * (.sample_compare), gather, cube and rectangle textures, the vertex and
 * compute stages, and mixed NEAREST/LINEAR min+mag filters (the LINEAR weight
 * form is used when either filter is LINEAR) keep the hardware border.
 */
#include "glm_context.h"
#include "mgl_ext_rstate.h"
#include "texdata_formats.h"
#include "spirv_cross_c.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MGL_BEMU_MARKER "mgl_border:"

/* ------------------------------------------------------------------ scanning */

/* does this unit's effective sampler clamp to border on any axis? (cheap gate) */
static bool border_axes(const TextureParameter *p, unsigned *axes)
{
    unsigned a = 0;
    if (!p) return false;
    if (p->wrap_s == GL_CLAMP_TO_BORDER) a |= 1;
    if (p->wrap_t == GL_CLAMP_TO_BORDER) a |= 2;
    if (p->wrap_r == GL_CLAMP_TO_BORDER) a |= 4;
    *axes = a;
    return a != 0;
}

/* any active texture unit asking for GL_CLAMP_TO_BORDER? */
bool mgl_border_wanted(GLMContext ctx)
{
    for (int w = 0; w < 4; w++)
    {
        unsigned bits = ctx->state.active_texture_mask[w];
        while (bits)
        {
            int u = w * 32 + __builtin_ctz(bits);
            bits &= bits - 1;
            Sampler *sam = ctx->state.texture_samplers[u];
            Texture *tex = ctx->state.active_textures[u];
            unsigned a;
            if (border_axes(sam ? &sam->params : (tex ? &tex->params : NULL), &a))
                return true;
        }
    }
    return false;
}

/* the three border colours Metal can produce, in the float domain */
static bool fixed_border_f(const GLfloat b[4])
{
    if (b[0] == 0.0f && b[1] == 0.0f && b[2] == 0.0f && (b[3] == 0.0f || b[3] == 1.0f)) return true;
    return b[0] == 1.0f && b[1] == 1.0f && b[2] == 1.0f && b[3] == 1.0f;
}
static bool fixed_border_i(const GLint b[4])
{
    if (b[0] == 0 && b[1] == 0 && b[2] == 0 && (b[3] == 0 || b[3] == 1)) return true;
    return b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1;
}

/* the border colour the Metal sampler actually returns (MGLRenderer+TextureParams
 * picks the nearest of the three fixed colours from the float form) */
static void hw_border(const GLfloat b[4], float out[4])
{
    float a = b[3];
    float l = (b[0] + b[1] + b[2]) / 3.0f;
    if (a < 0.5f)          { out[0] = out[1] = out[2] = out[3] = 0.0f; }
    else if (l < 0.5f)     { out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f; }
    else                   { out[0] = out[1] = out[2] = out[3] = 1.0f; }
}

/* quantize a float border colour the way the texture's storage rounds it, so
 * the emulated colour matches what a hardware border fetch would deliver */
static float quantize(float c, uint8_t kind)
{
    switch (kind)
    {
        case TDK_UN8:  { if (c < 0) c = 0; if (c > 1) c = 1; return floorf(c * 255.0f + 0.5f) / 255.0f; }
        case TDK_SN8:  { if (c < -1) c = -1; if (c > 1) c = 1; return floorf(c * 127.0f + 0.5f) / 127.0f; }
        case TDK_UN16: { if (c < 0) c = 0; if (c > 1) c = 1; return floorf(c * 65535.0f + 0.5f) / 65535.0f; }
        case TDK_SN16: { if (c < -1) c = -1; if (c > 1) c = 1; return floorf(c * 32767.0f + 0.5f) / 32767.0f; }
        default:       return c;
    }
}

/* print a float so MSL parses the exact value (a trailing h keeps half precision
 * for half textures; the half4 constructor rounds) */
static const char *fmt_float(float v, uint8_t kind)
{
    static char buf[4][40];
    static int i = 0;
    char *b = buf[i];
    i = (i + 1) & 3;
    snprintf(b, 40, "%.9g%s", v, kind == TDK_F16 ? "h" : "");
    return b;
}

/* build the MSL literal of a 4-component colour */
static void color_literal(char *out, size_t cap, const char *type, const float f[4],
                          const GLint *iv, const GLuint *uv, uint8_t kind)
{
    if (iv)      snprintf(out, cap, "%s(%d, %d, %d, %d)", type, iv[0], iv[1], iv[2], iv[3]);
    else if (uv) snprintf(out, cap, "%s(%uu, %uu, %uu, %uu)", type, uv[0], uv[1], uv[2], uv[3]);
    else snprintf(out, cap, "%s(%s, %s, %s, %s)", type, fmt_float(f[0], kind), fmt_float(f[1], kind),
                  fmt_float(f[2], kind), fmt_float(f[3], kind));
}

/*
 * Collect the MSL texture slots of `stage` whose bound texture + effective
 * sampler need border emulation.  Returns the entry count (<= MGL_BORDER_MAX).
 */
int mgl_border_scan(GLMContext ctx, int stage, MGLBorderEmu out[MGL_BORDER_MAX])
{
    Program *prog = ctx->state.program;
    int n = 0;

    if (!prog || !mgl_border_wanted(ctx))
        return 0;

    const SpirvResourceList *rl = &prog->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
    for (GLuint i = 0; i < rl->count && n < MGL_BORDER_MAX; i++)
    {
        const SpirvResource *r = &rl->list[i];
        GLuint unit = r->binding;
        if (unit >= (GLuint)TEXTURE_UNITS) continue;
        Texture *tex = ctx->state.active_textures[unit];
        if (!tex) continue;

        const TextureParameter *p = ctx->state.texture_samplers[unit] ?
                                    &ctx->state.texture_samplers[unit]->params : &tex->params;

        unsigned axes;
        if (!border_axes(p, &axes)) continue;

        /* an array texture never takes the border on its layer selector */
        if (tex->target == GL_TEXTURE_2D_ARRAY) axes &= ~4u;

        GLenum sized = tdResolveInternalFormat(tex->internalformat, 0, 0);
        const TDFormat *f = sized ? tdFormatInfo(sized) : NULL;
        if (!f) continue;

        bool integer = (f->flags & TDF_INTEGER) != 0 ||
                       f->comp_type == GL_INT || f->comp_type == GL_UNSIGNED_INT;
        bool hw;
        if (integer)
        {
            if (f->comp_type == GL_INT) hw = fixed_border_i(p->border_color_i);
            else                        hw = fixed_border_i((const GLint *)p->border_color_ui);
        }
        else
            hw = fixed_border_f(p->border_color);

        if (hw)
            continue;   /* the Metal sampler border is already exact */

        MGLBorderEmu *e = &out[n];
        memset(e, 0, sizeof *e);
        e->msl_slot = (r->msl_binding != 0xffffffffu) ? r->msl_binding : unit;
        e->axes = (uint8_t)axes;

        /* LINEAR border weights only make sense (and only work — see mgl_badd)
         * for filterable, non-integer textures */
        bool lin_within = (p->min_filter == GL_LINEAR || p->min_filter == GL_LINEAR_MIPMAP_NEAREST ||
                           p->min_filter == GL_LINEAR_MIPMAP_LINEAR);
        e->linear = (!integer && (f->flags & TDF_FILTERABLE) &&
                     (p->mag_filter == GL_LINEAR || lin_within)) ? 1 : 0;

        if (integer)
        {
            if (f->comp_type == GL_INT)
            {
                e->kind = 2;
                color_literal(e->bc, sizeof e->bc, "int4", NULL, p->border_color_i, NULL, 0);
                color_literal(e->bh, sizeof e->bh, "int4", (const float[4]){0, 0, 0, 0}, NULL, NULL, 0);
            }
            else
            {
                e->kind = 3;
                color_literal(e->bc, sizeof e->bc, "uint4", NULL, NULL, p->border_color_ui, 0);
                color_literal(e->bh, sizeof e->bh, "uint4", (const float[4]){0, 0, 0, 0}, NULL, NULL, 0);
            }
        }
        else
        {
            float q[4];
            float hb[4];
            hw_border(p->border_color, hb);
            for (int k = 0; k < 4; k++) q[k] = quantize(p->border_color[k], f->kind);
            e->kind = (f->kind == TDK_F16) ? 1 : 0;
            color_literal(e->bc, sizeof e->bc, e->kind ? "half4" : "float4", q, NULL, NULL, f->kind);
            color_literal(e->bh, sizeof e->bh, e->kind ? "half4" : "float4", hb, NULL, NULL, f->kind);
        }
        n++;
    }
    return n;
}

uint64_t mgl_border_sig(const MGLBorderEmu *e, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++)
    {
        h ^= e[i].msl_slot; h *= 1099511628211ULL;
        h ^= (uint64_t)(e[i].axes << 8 | e[i].linear << 4 | e[i].kind); h *= 1099511628211ULL;
        h ^= 0x9e3779b97f4a7c15ULL; h *= 1099511628211ULL;
        for (const char *p = e[i].bc; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
        for (const char *p = e[i].bh; *p; p++) { h ^= (unsigned char)*p; h *= 1099511628211ULL; }
    }
    return n ? h : 0;
}

/* ---------------------------------------------------------------- MSL rewrite */

static bool idchar(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

/* one emulated texture argument found in the MSL */
struct BTex {
    char name[128];
    char helper[24];    /* mgl_bsamp2 / mgl_bsamp2a / mgl_bsamp3 */
    const MGLBorderEmu *emu;
};

/* find `[[texture(N)]]` declarations and collect the emulated ones */
static int collect_tex(const char *msl, const MGLBorderEmu *emu, int n, struct BTex out[MGL_BORDER_MAX])
{
    int found = 0;
    const char *p = msl;

    while (found < MGL_BORDER_MAX && (p = strstr(p, "[[texture(")) != NULL)
    {
        const char *attr = p;                       /* "[[texture(N)]]" start */
        const char *np = p + 10;
        unsigned long slot = 0;
        while (*np >= '0' && *np <= '9') { slot = slot * 10 + (*np - '0'); np++; }
        p = np;

        const MGLBorderEmu *e = NULL;
        for (int i = 0; i < n; i++) if (emu[i].msl_slot == slot) { e = &emu[i]; break; }
        if (!e) continue;

        /* MSL declares `TYPE NAME [[texture(N)]]`: the name is left of the
         * attribute, the type left of the name */
        const char *t = attr;
        while (t > msl && (t[-1] == ' ' || t[-1] == '\t' || t[-1] == '\n' || t[-1] == '\r')) t--;
        const char *ne = t;                     /* end of the parameter name */
        const char *ns = ne;
        while (ns > msl && idchar((unsigned char)ns[-1])) ns--;
        size_t nl = (size_t)(ne - ns);
        if (nl == 0 || nl >= sizeof out[found].name) continue;

        const char *ts = ns;
        while (ts > msl && (ts[-1] == ' ' || ts[-1] == '\t' || ts[-1] == '\n' || ts[-1] == '\r')) ts--;
        const char *tend = ts;
        while (tend > msl && tend[-1] != ',' && tend[-1] != '(') tend--;
        if (tend > msl && (*tend == ',' || *tend == '(')) tend++;   /* past the separator */
        while (*tend == ' ' || *tend == '\t' || *tend == '\n' || *tend == '\r') tend++;
        char type[64];
        size_t tl = (size_t)(ts - tend);
        if (tl == 0 || tl >= sizeof type) continue;
        memcpy(type, tend, tl); type[tl] = 0;

        const char *helper = NULL;
        if (strncmp(type, "texture2d<", 10) == 0) helper = "mgl_bsamp2";
        else if (strncmp(type, "texture2d_array<", 16) == 0) helper = "mgl_bsamp2a";
        else if (strncmp(type, "texture3d<", 10) == 0) helper = "mgl_bsamp3";
        if (!helper) continue;   /* MS / cube / depth: the hardware border stays */

        memcpy(out[found].name, ns, nl); out[found].name[nl] = 0;
        snprintf(out[found].helper, sizeof out[found].helper, "%s", helper);
        out[found].emu = e;
        found++;
    }
    return found;
}

/* length of a balanced (...) argument list starting at open ('('), or 0 */
static size_t arglist_len(const char *open)
{
    size_t depth = 0;
    for (const char *q = open; *q; q++)
    {
        if (*q == '(' || *q == '[') depth++;
        else if (*q == ')' || *q == ']')
        {
            if (--depth == 0) return (size_t)(q - open) + 1;
        }
    }
    return 0;
}

/* split the argument list "(a, b, c)" at top-level commas */
static int split_args(const char *open, size_t len, char args[6][192])
{
    int n = 0, depth = 0;
    const char *start = open + 1;
    for (size_t i = 1; i + 1 < len; i++)
    {
        char c = open[i];
        if (c == '(' || c == '[') depth++;
        else if (c == ')' || c == ']') depth--;
        else if (c == ',' && depth == 0)
        {
            if (n < 6)
            {
                size_t l = (size_t)(&open[i] - start);
                if (l >= 192) l = 191;
                memcpy(args[n], start, l); args[n][l] = 0;
                n++;
            }
            start = &open[i + 1];
        }
    }
    if (n < 6)
    {
        size_t l = (size_t)((open + len - 1) - start);
        if (l >= 192) l = 191;
        memcpy(args[n], start, l); args[n][l] = 0;
        n++;
    }
    return n;
}

static const char *k_border_helpers =
"// mgl_border: GL_CLAMP_TO_BORDER emulation for border colours Metal's samplers\n"
"// cannot express (only transparent/opaque black and opaque white exist)\n"
"static inline __attribute__((always_inline)) float mgl_bwn(float c) { return (c < 0.0 || c >= 1.0) ? 0.0 : 1.0; }\n"
"static inline __attribute__((always_inline)) float mgl_bwl(float c, float n) { return clamp(0.5 + min(c, 1.0 - c) * n, 0.0, 1.0); }\n"
"static inline __attribute__((always_inline)) float4 mgl_badd(float4 r, float4 d, float b) { return r + b * d; }\n"
"static inline __attribute__((always_inline)) half4  mgl_badd(half4 r, half4 d, float b) { return r + (half)b * d; }\n"
"static inline __attribute__((always_inline)) int4   mgl_badd(int4 r, int4 d, float b) { (void)b; (void)d; return r; }\n"
"static inline __attribute__((always_inline)) uint4  mgl_badd(uint4 r, uint4 d, float b) { (void)b; (void)d; return r; }\n"
"template <typename T, typename... Rest>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_bsamp2(texture2d<T> t, sampler s, float2 c, uint ax, bool lin, vec<T,4> bc, vec<T,4> bh, Rest... rest)\n"
"{\n"
"    float w = 1.0;\n"
"    if (ax & 1u) w *= lin ? mgl_bwl(c.x, float(t.get_width())) : mgl_bwn(c.x);\n"
"    if (ax & 2u) w *= lin ? mgl_bwl(c.y, float(t.get_height())) : mgl_bwn(c.y);\n"
"    if (w <= 0.0) return bc;\n"
"    if (!lin)\n"
"    {\n"
"        /* GL NEAREST samples texel floor(c*N) clamped inside: c == 1.0 is the\n"
"         * last texel, never the border — keep the coordinate off Metal's */\n"
"        float2 n = float2(float(t.get_width()), float(t.get_height()));\n"
"        float2 cc = float2((ax & 1u) ? clamp(c.x, 0.0, (n.x - 0.5) / n.x) : c.x,\n"
"                           (ax & 2u) ? clamp(c.y, 0.0, (n.y - 0.5) / n.y) : c.y);\n"
"        return t.sample(s, cc, rest...);\n"
"    }\n"
"    /* sample() with hardware border B_hw returns (1-b)*S_in + b*B_hw; GL wants\n"
"     * (1-b)*S_in + b*BC, so compensate: r + b*(BC - B_hw) */\n"
"    return mgl_badd(t.sample(s, c, rest...), bc - bh, 1.0 - w);\n"
"}\n"
"template <typename T, typename... Rest>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_bsamp2a(texture2d_array<T> t, sampler s, float2 c, uint ax, bool lin, vec<T,4> bc, vec<T,4> bh, Rest... rest)\n"
"{\n"
"    float w = 1.0;\n"
"    if (ax & 1u) w *= lin ? mgl_bwl(c.x, float(t.get_width())) : mgl_bwn(c.x);\n"
"    if (ax & 2u) w *= lin ? mgl_bwl(c.y, float(t.get_height())) : mgl_bwn(c.y);\n"
"    if (w <= 0.0) return bc;\n"
"    if (!lin)\n"
"    {\n"
"        float2 n = float2(float(t.get_width()), float(t.get_height()));\n"
"        float2 cc = float2((ax & 1u) ? clamp(c.x, 0.0, (n.x - 0.5) / n.x) : c.x,\n"
"                           (ax & 2u) ? clamp(c.y, 0.0, (n.y - 0.5) / n.y) : c.y);\n"
"        return t.sample(s, cc, rest...);\n"
"    }\n"
"    return mgl_badd(t.sample(s, c, rest...), bc - bh, 1.0 - w);\n"
"}\n"
"template <typename T, typename... Rest>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_bsamp3(texture3d<T> t, sampler s, float3 c, uint ax, bool lin, vec<T,4> bc, vec<T,4> bh, Rest... rest)\n"
"{\n"
"    float w = 1.0;\n"
"    if (ax & 1u) w *= lin ? mgl_bwl(c.x, float(t.get_width())) : mgl_bwn(c.x);\n"
"    if (ax & 2u) w *= lin ? mgl_bwl(c.y, float(t.get_height())) : mgl_bwn(c.y);\n"
"    if (ax & 4u) w *= lin ? mgl_bwl(c.z, float(t.get_depth())) : mgl_bwn(c.z);\n"
"    if (w <= 0.0) return bc;\n"
"    if (!lin)\n"
"    {\n"
"        float3 n = float3(float(t.get_width()), float(t.get_height()), float(t.get_depth()));\n"
"        float3 cc = float3((ax & 1u) ? clamp(c.x, 0.0, (n.x - 0.5) / n.x) : c.x,\n"
"                           (ax & 2u) ? clamp(c.y, 0.0, (n.y - 0.5) / n.y) : c.y,\n"
"                           (ax & 4u) ? clamp(c.z, 0.0, (n.z - 0.5) / n.z) : c.z);\n"
"        return t.sample(s, cc, rest...);\n"
"    }\n"
"    return mgl_badd(t.sample(s, c, rest...), bc - bh, 1.0 - w);\n"
"}\n";

/*
 * Rewrite `msl` so plain `.sample()` / `.sample_level()` calls on the emulated
 * texture arguments go through the helpers.  Returns malloc'd MSL (the input is
 * untouched), or NULL when no emulated texture argument is present.
 */
char *mgl_border_rewrite(const char *msl, const MGLBorderEmu *emu, int n)
{
    struct BTex bt[MGL_BORDER_MAX];
    int nbt = collect_tex(msl, emu, n, bt);
    if (getenv("MGL_BORDER_DEBUG"))
    {
        const char *tpos = strstr(msl, "[[texture(");
        fprintf(stderr, "MGL border: collect_tex=%d attr_at=%d\n", nbt, (int)(tpos ? tpos - msl : -1));
        if (tpos) fprintf(stderr, "MGL border: attrctx=[%.100s]\n", tpos - 25);
        for (int i = 0; i < nbt; i++)
            fprintf(stderr, "MGL border: bt[%d] name '%s' helper '%s' slot %u\n",
                    i, bt[i].name, bt[i].helper, bt[i].emu->msl_slot);
    }
    if (!nbt) return NULL;
    if (getenv("MGL_BORDER_DUMP"))
    {
        FILE *f = fopen(getenv("MGL_BORDER_DUMP"), "wb");
        if (f) { fwrite(msl, 1, strlen(msl), f); fclose(f); }
    }

    size_t cap = strlen(msl) + strlen(k_border_helpers) + (size_t)n * 512 + 4096;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t at = 0;
    const char *src = msl;
    int rewrites = 0;

    while (*src)
    {
        /* the earliest plain sample call of any emulated texture object */
        const char *hit = NULL;
        struct BTex *hitb = NULL;
        size_t hit_ml = 0;
        for (int i = 0; i < nbt; i++)
        {
            size_t nl = strlen(bt[i].name);
            const char *q = src;
            while ((q = strstr(q, bt[i].name)) != NULL)
            {
                const char *dot = q + nl;
                if (q > msl && idchar((unsigned char)q[-1])) { q += nl; continue; }
                size_t ml = 0;
                if (strncmp(dot, ".sample_level(", 14) == 0) ml = 14;
                else if (strncmp(dot, ".sample(", 8) == 0) ml = 8;
                if (!ml) { q += nl; continue; }
                if (!hit || q < hit) { hit = q; hitb = &bt[i]; hit_ml = ml; }
                break;
            }
        }
        if (!hit)
        {
            strcpy(out + at, src);
            break;
        }
        if (getenv("MGL_BORDER_DEBUG"))
            fprintf(stderr, "MGL border: hit at %d '%.60s' ml=%zu\n", (int)(hit - msl), hit, hit_ml);

        /* everything before the call, then the rewritten call */
        if (at + (size_t)(hit - src) + 1 > cap) { free(out); return NULL; }
        memcpy(out + at, src, (size_t)(hit - src));
        at += (size_t)(hit - src);

        const char *open = hit + strlen(hitb->name) + hit_ml - 1;   /* the '(' */
        size_t alen = arglist_len(open);
        /* ".sample(" already carries the '(' that arglist_len counts again */
        size_t call_len = strlen(hitb->name) + hit_ml + alen - 1;
        if (alen == 0)
        {
            /* unterminated: copy the rest verbatim */
            strcpy(out + at, hit);
            break;
        }

        char args[6][192];
        int nargs = split_args(open, alen, args);
        if (nargs < 2)
        {
            if (at + call_len + 1 > cap) { free(out); return NULL; }
            memcpy(out + at, hit, call_len); at += call_len;
            src = hit + call_len;
            out[at] = 0;
            continue;
        }

        const MGLBorderEmu *e = hitb->emu;
        at += (size_t)snprintf(out + at, cap - at, "%s(%s, %s, %s, mgl_ba_%u, mgl_bl_%u, mgl_bc_%u, mgl_bh_%u",
                               hitb->helper, hitb->name, args[0], args[1],
                               e->msl_slot, e->msl_slot, e->msl_slot, e->msl_slot);
        for (int a = 2; a < nargs; a++)
            at += (size_t)snprintf(out + at, cap - at, ", %s", args[a]);
        at += (size_t)snprintf(out + at, cap - at, ")");
        if (at >= cap - 1) { free(out); return NULL; }
        src = hit + call_len;
        out[at] = 0;
        rewrites++;
    }

    if (!rewrites)
    {
        free(out);
        if (getenv("MGL_BORDER_DEBUG")) fprintf(stderr, "MGL border: zero rewrites\n");
        return NULL;
    }

    /* per-slot constants + helpers, after the using-declaration */
    {
        char *pre = (char *)malloc(strlen(k_border_helpers) + (size_t)n * 256 + 128);
        if (!pre) { free(out); return NULL; }
        size_t p = 0;
        p += (size_t)snprintf(pre + p, 128, "// %s GL_CLAMP_TO_BORDER custom border colours\n", MGL_BEMU_MARKER);
        for (int i = 0; i < n; i++)
            p += (size_t)snprintf(pre + p, 256,
                                  "constant const uint mgl_ba_%u = %uu;\n"
                                  "constant const bool mgl_bl_%u = %s;\n"
                                  "constant const %s mgl_bc_%u = %s;\n"
                                  "constant const %s mgl_bh_%u = %s;\n",
                                  emu[i].msl_slot, (unsigned)emu[i].axes,
                                  emu[i].msl_slot, emu[i].linear ? "true" : "false",
                                  emu[i].kind == 2 ? "int4" : emu[i].kind == 3 ? "uint4" : emu[i].kind == 1 ? "half4" : "float4",
                                  emu[i].msl_slot, emu[i].bc,
                                  emu[i].kind == 2 ? "int4" : emu[i].kind == 3 ? "uint4" : emu[i].kind == 1 ? "half4" : "float4",
                                  emu[i].msl_slot, emu[i].bh);
        if (p > strlen(k_border_helpers) + (size_t)n * 256 + 128 - 8) { free(pre); free(out); return NULL; }
        memcpy(pre + p, k_border_helpers, strlen(k_border_helpers) + 1);

        const char *anchor = strstr(out, "using namespace metal;");
        if (!anchor) { free(pre); free(out); if (getenv("MGL_BORDER_DEBUG")) fprintf(stderr, "MGL border: no anchor, out head '%.60s'\n", out); return NULL; }
        size_t at_anchor = (size_t)(anchor - out) + strlen("using namespace metal;");

        char *full = (char *)malloc(strlen(out) + strlen(pre) + 8);
        if (!full) { free(pre); free(out); return NULL; }
        memcpy(full, out, at_anchor);
        memcpy(full + at_anchor, "\n", 1);
        memcpy(full + at_anchor + 1, pre, strlen(pre));
        strcpy(full + at_anchor + 1 + strlen(pre), out + at_anchor);
        free(pre);
        free(out);
        return full;
    }
}
