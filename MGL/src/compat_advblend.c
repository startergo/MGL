/*
 * compat_advblend.c — GL_KHR_blend_equation_advanced (GL 4.6 core §17.3.8),
 * the compat domain's implementation.  All 15 advanced blend equations
 * (MULTIPLY..HSL_LUMINOSITY, GL_*_KHR) have no Metal fixed-function
 * equivalent, but Apple GPUs support programmable blending through
 * framebuffer fetch: a fragment entry-point parameter with the [[color(0)]]
 * attribute reads the attachment-0 value out of tile memory, raster-ordered
 * and coherent (this is why GL_KHR_blend_equation_advanced_coherent holds
 * without barriers).
 *
 * The equation therefore runs inside a rewritten copy of the program's
 * fragment MSL (mgl_advblend_rewrite): the entry gains a
 * `float4 mglAdvDst [[color(0)]]` argument and every `return <out>;` first
 * composites the written color with mglAdvDst.  Metal fixed-function
 * blending is disabled for attachment 0 for that pipeline
 * (MGLRenderer+RenderState.m) — the same emulation shape Mesa uses
 * (gl_nir_lower_blend_equation_advanced.c) on hardware without the unit.
 *
 * GL-side pieces live here as well:
 *   - enum<->mode table (mgl_advblend_mode; glBlendEquation accepts the
 *     enums — state.c), qualifier mask scanned from the FS source at link
 *     time (post-link hook), draw-time validation (pre-draw hook:
 *     INVALID_OPERATION for >1 draw buffer or a missing qualifier bit),
 *     and the query-only GL_BLEND_ADVANCED_COHERENT_KHR cap.
 *
 * Cost when no advanced equation is selected: one table lookup in the
 * pre-draw hook and one in the pipeline paths.  All rewriting/compiling
 * happens only while an advanced equation is actually active.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_compat.h"
#include "mgl_ext_fb.h"
#include "mgl_ext_program.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

/* ---------------------------------------------------------------- enums */

static const GLenum adv_mode_enum[MGL_ADVBLEND_MODES] = {
    GL_MULTIPLY_KHR, GL_SCREEN_KHR, GL_OVERLAY_KHR, GL_DARKEN_KHR, GL_LIGHTEN_KHR,
    GL_COLORDODGE_KHR, GL_COLORBURN_KHR, GL_HARDLIGHT_KHR, GL_SOFTLIGHT_KHR,
    GL_DIFFERENCE_KHR, GL_EXCLUSION_KHR,
    GL_HSL_HUE_KHR, GL_HSL_SATURATION_KHR, GL_HSL_COLOR_KHR, GL_HSL_LUMINOSITY_KHR
};

GLenum mgl_advblend_enum(int mode)
{
    if (mode < 0 || mode >= MGL_ADVBLEND_MODES)
        return 0;
    return adv_mode_enum[mode];
}

static const char *adv_mode_name[MGL_ADVBLEND_MODES] = {
    "multiply", "screen", "overlay", "darken", "lighten", "colordodge", "colorburn",
    "hardlight", "softlight", "difference", "exclusion", "hsl_hue",
    "hsl_saturation", "hsl_color", "hsl_luminosity"
};

const char *mgl_advblend_mode_name(int mode)
{
    return (mode >= 0 && mode < MGL_ADVBLEND_MODES) ? adv_mode_name[mode] : "(none)";
}

/* the GLSL layout qualifier spelling glslang accepts (TQualifier::getBlendEquationString) */
static const char *adv_mode_qual[MGL_ADVBLEND_MODES] = {
    "blend_support_multiply", "blend_support_screen", "blend_support_overlay",
    "blend_support_darken", "blend_support_lighten", "blend_support_colordodge",
    "blend_support_colorburn", "blend_support_hardlight", "blend_support_softlight",
    "blend_support_difference", "blend_support_exclusion", "blend_support_hsl_hue",
    "blend_support_hsl_saturation", "blend_support_hsl_color", "blend_support_hsl_luminosity"
};

/* O(1) enum -> mode via a lookup window over the KHR block (0x9294..0x92B0,
 * sparse: e.g. 0x929D / 0x929F / 0x92A1..0x92AC are not blend equations).
 * Encoded mode+1 so the zero-filled unused slots read "not advanced". */
#define ADV_BASE GL_MULTIPLY_KHR
#define ADV_SPAN (GL_HSL_LUMINOSITY_KHR + 1)
static const uint8_t adv_lookup[ADV_SPAN - ADV_BASE] = {
    [GL_MULTIPLY_KHR - ADV_BASE] = 1,
    [GL_SCREEN_KHR - ADV_BASE] = 2,
    [GL_OVERLAY_KHR - ADV_BASE] = 3,
    [GL_DARKEN_KHR - ADV_BASE] = 4,
    [GL_LIGHTEN_KHR - ADV_BASE] = 5,
    [GL_COLORDODGE_KHR - ADV_BASE] = 6,
    [GL_COLORBURN_KHR - ADV_BASE] = 7,
    [GL_HARDLIGHT_KHR - ADV_BASE] = 8,
    [GL_SOFTLIGHT_KHR - ADV_BASE] = 9,
    [GL_DIFFERENCE_KHR - ADV_BASE] = 10,
    [GL_EXCLUSION_KHR - ADV_BASE] = 11,
    [GL_HSL_HUE_KHR - ADV_BASE] = 12,
    [GL_HSL_SATURATION_KHR - ADV_BASE] = 13,
    [GL_HSL_COLOR_KHR - ADV_BASE] = 14,
    [GL_HSL_LUMINOSITY_KHR - ADV_BASE] = 15,
};

int mgl_advblend_mode(GLenum eq)
{
    if (eq < ADV_BASE || eq >= ADV_SPAN)
        return -1;
    return adv_lookup[eq - ADV_BASE] - 1;
}

/* Called by the advanced-equation setters (state.c) the first time an
 * advanced equation is installed.  This is the ONLY place the extension
 * touches global machinery: until then no per-draw hook is registered (the
 * draw path's mgl_ext_pre_draw loop stays empty, exactly the pre-advblend
 * behaviour) and the pipeline/specialize checks below stop at a single
 * load+branch on ext_compat->advblend_seen. */
void mgl_advblend_armed(GLMContext ctx)
{
    struct MGLExtState_compat *cm = ctx->state.ext_compat;
    if (!cm)
    {
        /* core context selecting an advanced equation: materialise the slot
         * (mgl_ext_init_compat does NOT allocate one for core contexts —
         * perf s31 zero-footprint rule).  is_compat stays FALSE. */
        cm = mgl_compat_slot_lazy(ctx);
        if (!cm)
            return;
    }
    if (!cm->advblend_seen)
    {
        cm->advblend_seen = GL_TRUE;
        mgl_ext_register_pre_draw(mgl_compat_advblend_pre_draw);
    }
}

/* Fast gate for the per-PSO-build checks (MGLRenderer+RenderState.m): one
 * flag load, no equation read, no table.  False for every context that never
 * selected an advanced blend equation. */
GLboolean mgl_advblend_seen(GLMContext ctx)
{
    struct MGLExtState_compat *cm = ctx->state.ext_compat;
    return cm && cm->advblend_seen;
}

/* GL-state gate: is an advanced equation currently in effect for draw buffer 0?
 * (blend enabled there + the same advanced enum for RGB and A — the equation
 * setters only ever install the pair together.)  No Metal work, no allocation. */
int mgl_advblend_active(GLMContext ctx)
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    GLenum eq = ctx->state.var.blend_equation_rgb[0];

    if (!rs || !rs->blend[0] || mgl_advblend_mode(eq) < 0)
        return -1;
    if (ctx->state.var.blend_equation_alpha[0] != eq)
        return -1;
    /* no fragment stage -> nothing composites */
    if (!ctx->state.program || !ctx->state.program->shader_slots[_FRAGMENT_SHADER])
        return -1;
    return mgl_advblend_mode(eq);
}

/* the effective fragment program (a bound program wins over pipeline stages) */
static Program *adv_fs_program(GLMContext ctx)
{
    if (ctx->state.program)
        return ctx->state.program;
    if (ctx->state.program_pipeline)
        return MGL_PIPELINE_OBJ(ctx->state.program_pipeline)->base.stage_programs[_FRAGMENT_SHADER];
    return NULL;
}

/* --------------------------------------------------- blend_support_* mask */

uint32_t mgl_advblend_scan_src(const char *src)
{
    uint32_t mask = 0;

    if (!src)
        return 0;

    if (strstr(src, "blend_support_all_equations"))
        return (1u << MGL_ADVBLEND_MODES) - 1;

    for (int i = 0; i < MGL_ADVBLEND_MODES; i++)
    {
        const char *p, *name = adv_mode_qual[i];
        size_t n = strlen(name);
        for (p = strstr(src, name); p; p = strstr(p + n, name))
        {
            /* whole identifier (blend_support_multiply must not match
             * blend_support_multiply_foo, nor inside a longer token) */
            char prev = p > src ? p[-1] : 0;
            char next = p[n];
            if (isalnum((unsigned char)prev) || prev == '_' ||
                isalnum((unsigned char)next) || next == '_')
                continue;
            mask |= 1u << i;
            break;
        }
    }
    return mask;
}

/* Which equations does the CURRENT fragment stage support?  Scanned on demand
 * from the retained GLSL source (Shader->src): this runs only on the
 * advanced-blend draw paths, and keeping no per-program cache means no field in
 * the shared Program/Spirv layout and no link-time hook at all (perf s30). */
static uint32_t adv_blend_mask(GLMContext ctx)
{
    Program *p = adv_fs_program(ctx);
    Shader *fs = p ? p->shader_slots[_FRAGMENT_SHADER] : NULL;
    return fs ? mgl_advblend_scan_src(fs->src) : 0;
}

/* ------------------------------------------------------ draw-time checks */

bool mgl_compat_advblend_pre_draw(GLMContext ctx, GLenum mode, GLsizei vertex_count,
                                  GLsizei instance_count, bool indexed, GLenum type,
                                  size_t offset)
{
    (void)mode; (void)vertex_count; (void)instance_count;
    (void)indexed; (void)type; (void)offset;

    int m = mgl_advblend_active(ctx);
    if (m < 0)
        return true;

    /* GL 4.6 §17.3.8: advanced blending draws to one color buffer only.
     * Count the active (non-NONE) draw buffers of the current framebuffer. */
    {
        int n = 0;
        MGLFBDrawState *ds = mgl_fb_draw_state(ctx, ctx->state.framebuffer);
        if (ds)
        {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
            {
                if (ds->draw_buffers[i] == GL_NONE) continue;
                if (ds->draw_buffers[i] == GL_FRONT_AND_BACK) { n += 2; continue; }
                n++;
            }
        }
        if (n > 1)
        {
            ctx->error_func(ctx, __func__, GL_INVALID_OPERATION);
            return false;
        }
    }

    /* and the fragment shader must declare the matching blend_support_* */
    if (!(adv_blend_mask(ctx) & (1u << m)))
    {
        ctx->error_func(ctx, __func__, GL_INVALID_OPERATION);
        return false;
    }

    return true;
}

/* ------------------------------------------------------ MSL rewriting */

typedef struct {
    char *s;
    size_t len, cap;
} AdvStr;

static void adv_put(AdvStr *b, const char *txt, size_t n)
{
    if (b->len + n + 1 > b->cap)
    {
        b->cap = (b->len + n + 1) * 2;
        b->s = realloc(b->s, b->cap);
    }
    memcpy(b->s + b->len, txt, n);
    b->len += n;
    b->s[b->len] = 0;
}

static void adv_puts(AdvStr *b, const char *txt)
{
    adv_put(b, txt, strlen(txt));
}

static bool adv_ident(int c)
{
    return isalnum((unsigned char)c) || c == '_';
}

/* the MSL text of mgl_adv_unp/mgl_adv_lum/.../mgl_adv_f for one equation */
static bool adv_blend_math(AdvStr *b, int mode)
{
    adv_puts(b,
        "/* ===== MGL advanced blending (GL_KHR_blend_equation_advanced, compat_advblend.c) =====\n"
        "   Reference: GL 4.6 17.3.8 / dEQU glcBlendEquationAdvancedTests Blend(): work on\n"
        "   un-premultiplied colors, then composite RGB = f*p0 + Cs'*p1 + Cd'*p2,\n"
        "   A = p0+p1+p2 with p0 = As*Ad, p1 = As*(1-Ad), p2 = Ad*(1-As). */\n"
        "static inline float3 mgl_adv_unp(float4 c)\n"
        "{\n"
        "    /* C' = C.rgb / C.a, or 0 when C.a == 0 (dEQU ToNormal; Mesa's extra\n"
        "     * rgb == a -> 1 per-component special case is NOT in the reference\n"
        "     * math and shows up as a 1-channel mismatch on premultiplied-ish\n"
        "     * test colours, e.g. src RGBA(191,143,96,191) under MULTIPLY) */\n"
        "    return select(float3(0.0), c.rgb / c.a, c.a != 0.0);\n"
        "}\n"
        "static inline float mgl_adv_lum(float3 c) { return 0.30 * c.r + 0.59 * c.g + 0.11 * c.b; }\n"
        "static inline float mgl_adv_min3(float3 c) { return min(min(c.r, c.g), c.b); }\n"
        "static inline float mgl_adv_max3(float3 c) { return max(max(c.r, c.g), c.b); }\n"
        "static inline float3 mgl_adv_setlum(float3 cbase, float3 clum)\n"
        "{\n"
        "    float lbase = mgl_adv_lum(cbase);\n"
        "    float llum = mgl_adv_lum(clum);\n"
        "    float3 color = cbase + float3(llum - lbase);\n"
        "    float3 vllum = float3(llum);\n"
        "    float3 lo = vllum + ((color - vllum) * llum) / (llum - mgl_adv_min3(color));\n"
        "    float3 hi = vllum + ((color - vllum) * (1.0 - llum)) / (mgl_adv_max3(color) - llum);\n"
        "    float3 r = select(color, hi, mgl_adv_max3(color) > 1.0);\n"
        "    return select(r, lo, mgl_adv_min3(color) < 0.0);\n"
        "}\n"
        "static inline float3 mgl_adv_setlumsat(float3 cbase, float3 csat, float3 clum)\n"
        "{\n"
        "    float minbase = mgl_adv_min3(cbase);\n"
        "    float sbase = mgl_adv_max3(cbase) - minbase;\n"
        "    float ssat = mgl_adv_max3(csat) - mgl_adv_min3(csat);\n"
        "    float3 color = select(float3(0.0), (cbase - float3(minbase)) * ssat / sbase, sbase > 0.0);\n"
        "    return mgl_adv_setlum(color, clum);\n"
        "}\n");

    switch (mode)
    {
        case 0:  /* MULTIPLY */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return s * d; }\n"); break;
        case 1:  /* SCREEN */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return s + d - s * d; }\n"); break;
        case 2:  /* OVERLAY */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return select(2.0 * s * d, 1.0 - 2.0 * (1.0 - s) * (1.0 - d), d > 0.5); }\n");
            break;
        case 3:  /* DARKEN */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return min(s, d); }\n"); break;
        case 4:  /* LIGHTEN */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return max(s, d); }\n"); break;
        case 5:  /* COLORDODGE */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{\n"
                "    /* dEQU order: d <= 0 -> 0 FIRST, then s < 1 -> min(1, d/(1-s)),\n"
                "     * else 1 (a d=0 lane must stay 0 even when s >= 1) */\n"
                "    float3 r = select(min(float3(1.0), d / (1.0 - s)), float3(1.0), s >= 1.0);\n"
                "    return select(float3(0.0), r, d > 0.0);\n"
                "}\n");
            break;
        case 6:  /* COLORBURN */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{\n"
                "    float3 r = select(float3(0.0), float3(1.0) - min(float3(1.0), (1.0 - d) / s), s > 0.0);\n"
                "    return select(r, float3(1.0), d >= 1.0);\n"
                "}\n");
            break;
        case 7:  /* HARDLIGHT */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return select(2.0 * s * d, 1.0 - 2.0 * (1.0 - s) * (1.0 - d), s > 0.5); }\n");
            break;
        case 8:  /* SOFTLIGHT */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{\n"
                "    float3 a = d - (1.0 - 2.0 * s) * d * (1.0 - d);\n"
                "    float3 b = d + (2.0 * s - 1.0) * d * ((16.0 * d - 12.0) * d + 3.0);\n"
                "    float3 c = d + (2.0 * s - 1.0) * (sqrt(d) - d);\n"
                "    float3 r = select(b, c, d > 0.25);\n"
                "    return select(a, r, s > 0.5);\n"
                "}\n");
            break;
        case 9:  /* DIFFERENCE */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return abs(s - d); }\n"); break;
        case 10: /* EXCLUSION */
            adv_puts(b, "static inline float3 mgl_adv_f(float3 s, float3 d) { return s + d - 2.0 * s * d; }\n"); break;
        case 11: /* HSL_HUE */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return mgl_adv_setlumsat(s, d, d); }\n");
            break;
        case 12: /* HSL_SATURATION */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return mgl_adv_setlumsat(d, s, d); }\n");
            break;
        case 13: /* HSL_COLOR */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return mgl_adv_setlum(s, d); }\n");
            break;
        case 14: /* HSL_LUMINOSITY */
            adv_puts(b,
                "static inline float3 mgl_adv_f(float3 s, float3 d)\n"
                "{ return mgl_adv_setlum(d, s); }\n");
            break;
        default:
            return false;
    }

    adv_puts(b,
        "static inline float4 mgl_adv_blend(float4 S, float4 D)\n"
        "{\n"
        "    float3 s = mgl_adv_unp(S);\n"
        "    float3 d = mgl_adv_unp(D);\n"
        "    float p0 = S.a * D.a;\n"
        "    float p1 = S.a * (1.0 - D.a);\n"
        "    float p2 = D.a * (1.0 - S.a);\n"
        "    float3 rgb = mgl_adv_f(s, d) * p0 + s * p1 + d * p2;\n"
        "    return float4(rgb, p0 + p1 + p2);\n"
        "}\n");
    return true;
}

/* find `<name>(` used as a standalone identifier (not a member selection,
 * not a longer identifier); returns the identifier start, or NULL */
static const char *adv_find_call(const char *s, const char *name)
{
    size_t n = strlen(name);
    const char *p = s;

    while ((p = strstr(p, name)) != NULL)
    {
        if (p > s && (adv_ident(p[-1]) || p[-1] == '.')) { p += n; continue; }
        const char *q = p + n;
        while (*q == ' ' || *q == '\t' || *q == '\n') q++;
        if (*q == '(')
            return p;
        p += n;
    }
    return NULL;
}

/* index of the identifier that ends just before `p' (skipping whitespace) */
static const char *adv_ident_before(const char *s, const char *p)
{
    while (p > s && (p[-1] == ' ' || p[-1] == '\t' || p[-1] == '\n'))
        p--;
    const char *e = p;
    while (p > s && adv_ident(p[-1]))
        p--;
    return (p < e) ? p : NULL;
}

/* index just past the identifier starting at `p' */
static const char *adv_ident_end(const char *p)
{
    while (*p && adv_ident(*p))
        p++;
    return p;
}

/* matching close paren/brace from an opener */
static const char *adv_match(const char *open)
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

/*
 * Rewrite a fragment-stage MSL (SPIRV-Cross output) so the entry itself
 * performs `mode' over the destination read through a [[color(0)]] fragment
 * argument.  Returns malloc'd MSL (entry name unchanged, so the rest of the
 * pipeline machinery finds it), or NULL when the entry shape is not the
 * expected one — the caller then falls back to no blending.
 */
char *mgl_advblend_rewrite(const char *msl, const char *entry, int mode)
{
    if (!msl || !entry || mode < 0 || mode >= MGL_ADVBLEND_MODES)
        return NULL;
    if (strstr(msl, "mgl_adv_blend"))
        return NULL;   /* already rewritten */

    /* entry definition: `fragment <rettype> <entry>(<params>) { body }` */
    const char *name = adv_find_call(msl, entry);
    if (!name)
        return NULL;

    /* the `fragment' qualifier keyword in front of the return type */
    const char *ret = adv_ident_before(msl, name);
    if (!ret)
        return NULL;
    const char *kw = adv_ident_before(msl, ret);
    if (!kw || strncmp(kw, "fragment", 8) != 0 || adv_ident(kw[8]))
        return NULL;

    const char *params_open = name + strlen(entry);
    while (*params_open == ' ' || *params_open == '\t' || *params_open == '\n')
        params_open++;
    if (*params_open != '(')
        return NULL;
    const char *params_close = adv_match(params_open);
    if (!params_close)
        return NULL;

    const char *body_open = params_close + 1;
    while (*body_open == ' ' || *body_open == '\t' || *body_open == '\n')
        body_open++;
    if (*body_open != '{')
        return NULL;
    const char *body_close = adv_match(body_open);
    if (!body_close)
        return NULL;

    /* the color(0) member of the return struct (SPIRV-Cross emits
     * `struct <entry>_out { float4 m [[color(0)]]; ... };`) */
    char retname[128];
    size_t rn = adv_ident_end(ret) - ret;
    if (rn == 0 || rn >= sizeof retname)
        return NULL;
    memcpy(retname, ret, rn);
    retname[rn] = 0;

    char decl[192];
    snprintf(decl, sizeof decl, "struct %s", retname);
    const char *st = strstr(msl, decl);
    if (!st)
        return NULL;
    const char *stopen = strchr(st + strlen(decl), '{');
    if (!stopen)
        return NULL;
    const char *stclose = adv_match(stopen);
    if (!stclose)
        return NULL;
    const char *col = stopen;
    const char *member = NULL;
    while ((col = strstr(col, "color(0)")) != NULL && col < stclose)
    {
        const char *br = col;
        while (br > stopen && br[-1] == ' ')
            br--;
        if (br - stopen >= 2 && br[-1] == '[' && br[-2] == '[')
        {
            const char *m = adv_ident_before(msl, br - 2);
            if (m && m > stopen)
            {
                member = m;
                break;
            }
        }
        col += 8;
    }
    if (!member)
        return NULL;
    {
        /* the member's type must be `float4': anything else (half4, a short
         * vector, an integer output) cannot take a [[color(0)]] float4 fetch */
        const char *scan = member;
        while (scan > stopen + 1 && scan[-1] != ';' && scan[-1] != '{' && scan[-1] != '\n')
            scan--;
        while (scan < member && (*scan == ' ' || *scan == '\t'))
            scan++;
        if (scan >= member || strncmp(scan, "float4", 6) != 0 || adv_ident(scan[6]))
            return NULL;
    }
    char membername[128];
    size_t mn = adv_ident_end(member) - member;
    if (mn == 0 || mn >= sizeof membername)
        return NULL;
    memcpy(membername, member, mn);
    membername[mn] = 0;

    /* assemble: [0,kw) + math + [kw,params_close) + ", float4 dst [[color(0)]]"
     * + [params_close, ...) with every `return <var>;' inside the entry body
     * first blending <var>.member with the fetched destination */
    AdvStr b = { NULL, 0, 0 };

    adv_put(&b, msl, (size_t)(kw - msl));

    if (!adv_blend_math(&b, mode))
    {
        free(b.s);
        return NULL;
    }

    adv_put(&b, kw, (size_t)(params_close - kw));

    /* empty parameter list -> no leading comma */
    {
        const char *q = params_open + 1;
        while (q < params_close && (*q == ' ' || *q == '\t' || *q == '\n'))
            q++;
        adv_puts(&b, (q < params_close) ? ", float4 mglAdvDst [[color(0)]]"
                                        : "float4 mglAdvDst [[color(0)]]");
    }

    /* copy the body, patching `return <var>;' statements */
    const char *p = params_close;
    const char *end = body_close;
    while (p < end)
    {
        const char *hit = p;
        const char *ret_kw = NULL;
        while ((hit = strstr(hit, "return")) != NULL && hit < end)
        {
            bool okb = hit == msl || !(adv_ident(hit[-1]) || hit[-1] == '.');
            const char *after = hit + 6;
            if (okb && !(adv_ident(*after)))
            {
                ret_kw = hit;
                break;
            }
            hit += 6;
        }
        if (!ret_kw)
        {
            adv_put(&b, p, (size_t)(end - p));
            break;
        }
        adv_put(&b, p, (size_t)(ret_kw - p));

        const char *q = ret_kw + 6;
        while (*q == ' ' || *q == '\t') q++;
        const char *v0 = q;
        const char *v1 = adv_ident_end(q);
        const char *q2 = v1;
        while (*q2 == ' ' || *q2 == '\t') q2++;

        if (v1 == v0 || *q2 != ';' || (v1 - v0) > 64 ||
            strncmp(v0, "discard_fragment", 16) == 0)
        {
            /* not `return <var>;' (a constructor, a nested call, ...): keep it */
            adv_put(&b, ret_kw, (size_t)(q2 + 1 - ret_kw));
            p = q2 + 1;
            continue;
        }

        char var[72];
        memcpy(var, v0, (size_t)(v1 - v0));
        var[v1 - v0] = 0;
        char ins[256];
        snprintf(ins, sizeof ins, "{ %s.%s = mgl_adv_blend(%s.%s, mglAdvDst); return %s; }",
                 var, membername, var, membername, var);
        adv_puts(&b, ins);
        p = q2 + 1;
    }

    adv_put(&b, end, strlen(end));
    return b.s;
}
