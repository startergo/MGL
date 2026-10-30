/*
 * test_compat_advblend.c — GL_KHR_blend_equation_advanced (GL 4.6 core
 * §17.3.8) emulated on Metal by the framebuffer-fetch fragment rewrite
 * (compat_advblend.c): a [[color(0)]] argument reads the destination out of
 * tile memory and the rewritten entry composites every return.  Reference
 * math transcribed from glcBlendEquationAdvancedTests.cpp (dEQU) and the GL
 * 4.6 spec (Rs' = Rs/As etc.); pixel checks over RGBA8.
 */
#include "mgl_test.h"
#include <string.h>
#include <stdbool.h>
#include <math.h>

#ifndef GL_MULTIPLY_KHR
#define GL_MULTIPLY_KHR 0x9294
#define GL_SCREEN_KHR 0x9295
#define GL_OVERLAY_KHR 0x9296
#define GL_DARKEN_KHR 0x9297
#define GL_LIGHTEN_KHR 0x9298
#define GL_COLORDODGE_KHR 0x9299
#define GL_COLORBURN_KHR 0x929A
#define GL_HARDLIGHT_KHR 0x929B
#define GL_SOFTLIGHT_KHR 0x929C
#define GL_DIFFERENCE_KHR 0x929E
#define GL_EXCLUSION_KHR 0x92A0
#define GL_HSL_HUE_KHR 0x92AD
#define GL_HSL_SATURATION_KHR 0x92AE
#define GL_HSL_COLOR_KHR 0x92AF
#define GL_HSL_LUMINOSITY_KHR 0x92B0
#define GL_BLEND_ADVANCED_COHERENT_KHR 0x9509
#endif

#define W 16
#define H 16

static const GLenum k_modes[15] = {
    GL_MULTIPLY_KHR, GL_SCREEN_KHR, GL_OVERLAY_KHR, GL_DARKEN_KHR, GL_LIGHTEN_KHR,
    GL_COLORDODGE_KHR, GL_COLORBURN_KHR, GL_HARDLIGHT_KHR, GL_SOFTLIGHT_KHR,
    GL_DIFFERENCE_KHR, GL_EXCLUSION_KHR,
    GL_HSL_HUE_KHR, GL_HSL_SATURATION_KHR, GL_HSL_COLOR_KHR, GL_HSL_LUMINOSITY_KHR
};
static const char *k_mode_names[15] = {
    "multiply", "screen", "overlay", "darken", "lighten", "colordodge", "colorburn",
    "hardlight", "softlight", "difference", "exclusion", "hsl_hue", "hsl_saturation",
    "hsl_color", "hsl_luminosity"
};
static const char *k_mode_quals[15] = {
    "blend_support_multiply", "blend_support_screen", "blend_support_overlay",
    "blend_support_darken", "blend_support_lighten", "blend_support_colordodge",
    "blend_support_colorburn", "blend_support_hardlight", "blend_support_softlight",
    "blend_support_difference", "blend_support_exclusion", "blend_support_hsl_hue",
    "blend_support_hsl_saturation", "blend_support_hsl_color", "blend_support_hsl_luminosity"
};

/* ------------- reference math (dEQU glcBlendEquationAdvancedTests.cpp) ---- */

typedef struct { float c[4]; } V4;

static float fmin3(V4 c) { float m = fminf(c.c[0], c.c[1]); return fminf(m, c.c[2]); }
static float fmax3(V4 c) { float m = fmaxf(c.c[0], c.c[1]); return fmaxf(m, c.c[2]); }
static float sat(V4 c) { return fmax3(c) - fmin3(c); }
static float lum(V4 c) { return 0.30f * c.c[0] + 0.59f * c.c[1] + 0.11f * c.c[2]; }

static V4 clip_color(V4 c)
{
    float l = lum(c), mn = fmin3(c), mx = fmax3(c);
    if (mn < 0.0f)
        for (int i = 0; i < 3; i++) c.c[i] = l + (c.c[i] - l) * l / (l - mn);
    if (mx > 1.0f)
        for (int i = 0; i < 3; i++) c.c[i] = l + (c.c[i] - l) * (1.0f - l) / (mx - l);
    return c;
}

static V4 set_lum(V4 cbase, V4 clum)
{
    float lbase = lum(cbase), llum = lum(clum);
    for (int i = 0; i < 3; i++) cbase.c[i] += (llum - lbase);
    return clip_color(cbase);
}

static V4 set_lum_sat(V4 cbase, V4 csat, V4 clum)
{
    float minbase = fmin3(cbase), sbase = sat(cbase), ssat = sat(csat);
    V4 color;
    for (int i = 0; i < 3; i++)
        color.c[i] = (sbase > 0.0f) ? (cbase.c[i] - minbase) * ssat / sbase : 0.0f;
    color.c[3] = 0.0f;
    return set_lum(color, clum);
}

static V4 to_normal(V4 p)
{
    V4 n = p;
    if (p.c[3] == 0.0f)
    {
        n.c[0] = n.c[1] = n.c[2] = 0.0f;
        return n;
    }
    for (int i = 0; i < 3; i++) n.c[i] = p.c[i] / p.c[3];
    return n;
}

static V4 mode_rgb(int mode, V4 s, V4 d)
{
    V4 r;
    for (int i = 0; i < 3; i++)
    {
        float a = s.c[i], b = d.c[i], v = 0.0f;
        switch (mode)
        {
            case 0:  v = a * b; break;
            case 1:  v = a + b - a * b; break;
            case 2:  v = (b <= 0.5f) ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b); break;
            case 3:  v = fminf(a, b); break;
            case 4:  v = fmaxf(a, b); break;
            case 5:  v = (b <= 0.0f) ? 0.0f : ((a < 1.0f) ? fminf(1.0f, b / (1.0f - a)) : 1.0f); break;
            case 6:  v = (b >= 1.0f) ? 1.0f : ((a > 0.0f) ? 1.0f - fminf(1.0f, (1.0f - b) / a) : 0.0f); break;
            case 7:  v = (a <= 0.5f) ? 2.0f * a * b : 1.0f - 2.0f * (1.0f - a) * (1.0f - b); break;
            case 9:  v = fabsf(b - a); break;
            case 10: v = a + b - 2.0f * a * b; break;
            default: break;   /* softlight & HSL handled below */
        }
        r.c[i] = v;
    }
    r.c[3] = 0.0f;

    if (mode == 8)   /* softlight */
    {
        for (int i = 0; i < 3; i++)
        {
            float s1 = s.c[i], d1 = d.c[i];
            if (s1 <= 0.5f)
                r.c[i] = d1 - (1.0f - 2.0f * s1) * d1 * (1.0f - d1);
            else if (d1 <= 0.25f)
                r.c[i] = d1 - (1.0f - 2.0f * s1) * d1 * ((16.0f * d1 - 12.0f) * d1 + 3.0f);
            else
                r.c[i] = d1 + (2.0f * s1 - 1.0f) * (sqrtf(d1) - d1);
        }
    }
    return r;
}

static V4 blend_ref(int mode, V4 src, V4 dst)
{
    V4 sn = to_normal(src), dn = to_normal(dst);
    V4 rgb;
    if (mode == 11)      rgb = set_lum_sat(src, dst, dst);
    else if (mode == 12) rgb = set_lum_sat(dst, src, dst);
    else if (mode == 13) rgb = set_lum(src, dst);
    else if (mode == 14) rgb = set_lum(dst, src);
    else                 rgb = mode_rgb(mode, sn, dn);

    float p0 = src.c[3] * dst.c[3];
    float p1 = src.c[3] * (1.0f - dst.c[3]);
    float p2 = dst.c[3] * (1.0f - src.c[3]);
    V4 out;
    for (int i = 0; i < 3; i++)
        out.c[i] = p0 * rgb.c[i] + p1 * sn.c[i] + p2 * dn.c[i];
    out.c[3] = p0 + p1 + p2;
    return out;
}

/* RGBA8 round-trip */
static float q8(float x)
{
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return floorf(x * 255.0f + 0.5f) / 255.0f;
}

/* ---------------------------------------------------------------- helpers */

static const char *vs_quad =
    "#version 150\n"
    "out vec2 vUV;\n"
    "void main() {\n"
    "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "    vUV = p;\n"
    "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

/* qualifier: layout string or NULL for "no qualifier" */
static GLuint make_program(const char *qual, const char *directive)
{
    char fs[1024];
    snprintf(fs, sizeof fs,
        "#version 430\n"
        "%s\n"
        "%s\n"
        "layout (location = 0) out vec4 oCol;\n"
        "uniform vec4 uSrc;\n"
        "in vec2 vUV;\n"
        "void main() {\n"
        "    vec4 c = uSrc;\n"
        "    if (uSrc.x > 10.0) { oCol = vec4(1.0, 0.0, 0.0, 1.0); return; }   /* an early return */\n"
        "    oCol = c;\n"
        "}\n",
        directive ? directive : "/* no extension directive */",
        qual ? qual : "/* no qualifier */");

    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p = glCreateProgram();
    const char *src;
    src = vs_quad;
    glShaderSource(v, 1, &src, NULL);
    glCompileShader(v);
    src = fs;
    glShaderSource(f, 1, &src, NULL);
    glCompileShader(f);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

static bool program_ok(GLuint p)
{
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link failed: %s\n", log);
    }
    return ok != 0;
}

static GLuint tex_rgba8(int n)
{
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    (void)n;
    return t;
}

static GLuint fbo1, tex1;
static GLuint fbo2, tex2a, tex2b;

static void setup_fbos(void)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);   /* core context: draws need a VAO */
    glBindVertexArray(vao);

    tex1 = tex_rgba8(0);
    glGenFramebuffers(1, &fbo1);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex1, 0);

    tex2a = tex_rgba8(1);
    tex2b = tex_rgba8(2);
    glGenFramebuffers(1, &fbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex2a, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex2b, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
}

/* draw one quad with `src' over the cleared `dst' using `mode' */
static void draw_one(GLuint prog, int mode, V4 src, V4 dst)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
    GLenum db = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &db);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glClearColor(dst.c[0], dst.c[1], dst.c[2], dst.c[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glUniform4fv(glGetUniformLocation(prog, "uSrc"), 1, src.c);
    glEnable(GL_BLEND);
    glBlendEquation(k_modes[mode]);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

/* ------------------------------------------------------------------ tests */

TEST(ext_and_state)
{
    /* extension strings on the CORE context (this test runs core by default) */
    GLint prof = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &prof);
    EXPECT_EQ_INT(prof, GL_CONTEXT_CORE_PROFILE_BIT);

    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    EXPECT_TRUE(n > 0);
    bool base = false, coh = false;
    for (int i = 0; i < n; i++)
    {
        const GLubyte *e = glGetStringi(GL_EXTENSIONS, i);
        if (!e) continue;
        if (!strcmp((const char *)e, "GL_KHR_blend_equation_advanced")) base = true;
        if (!strcmp((const char *)e, "GL_KHR_blend_equation_advanced_coherent")) coh = true;
    }
    EXPECT_TRUE(base);
    EXPECT_TRUE(coh);

    /* glBlendEquation accepts the advanced enums; the Separate forms do not */
    EXPECT_NO_GL_ERROR();
    glBlendEquation(GL_MULTIPLY_KHR);
    EXPECT_NO_GL_ERROR();
    GLint eq = 0;
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &eq);
    EXPECT_EQ_INT(eq, GL_MULTIPLY_KHR);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eq);
    EXPECT_EQ_INT(eq, GL_MULTIPLY_KHR);

    glBlendEquationSeparate(GL_MULTIPLY_KHR, GL_FUNC_ADD);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_MULTIPLY_KHR);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBlendEquationSeparatei(0, GL_MULTIPLY_KHR, GL_FUNC_ADD);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &eq);   /* unchanged by the failures */
    EXPECT_EQ_INT(eq, GL_MULTIPLY_KHR);

    glBlendEquationi(0, GL_SCREEN_KHR);
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &eq);
    EXPECT_EQ_INT(eq, GL_SCREEN_KHR);

    /* disabled blend must not touch the pipeline path */
    glBlendEquation(GL_FUNC_ADD);
    EXPECT_NO_GL_ERROR();
}

TEST(coherent_cap)
{
    EXPECT_TRUE(glIsEnabled(GL_BLEND_ADVANCED_COHERENT_KHR));
    EXPECT_NO_GL_ERROR();
    glDisable(GL_BLEND_ADVANCED_COHERENT_KHR);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(!glIsEnabled(GL_BLEND_ADVANCED_COHERENT_KHR));
    GLint i = 2;
    glGetIntegerv(GL_BLEND_ADVANCED_COHERENT_KHR, &i);
    EXPECT_EQ_INT(i, 0);
    glEnable(GL_BLEND_ADVANCED_COHERENT_KHR);
    EXPECT_TRUE(glIsEnabled(GL_BLEND_ADVANCED_COHERENT_KHR));
    glGetIntegerv(GL_BLEND_ADVANCED_COHERENT_KHR, &i);
    EXPECT_EQ_INT(i, 1);
    EXPECT_NO_GL_ERROR();
}

TEST(glsl_qualifiers)
{
    /* require + matching qualifier links */
    char qual[64];
    snprintf(qual, sizeof qual, "layout (%s) out;", k_mode_quals[0]);
    GLuint p = make_program(qual, "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));
    glDeleteProgram(p);

    /* blend_support_all_equations */
    p = make_program("layout (blend_support_all_equations) out;",
                     "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));
    glDeleteProgram(p);

    /* the preprocessor macro exists and is 1 */
    const char *fs =
        "#version 430\n"
        "#extension GL_KHR_blend_equation_advanced : require\n"
        "layout (blend_support_multiply) out;\n"
        "layout (location = 0) out vec4 oCol;\n"
        "void main() { oCol = vec4(GL_KHR_blend_equation_advanced == 1 ? 0.0 : 1.0); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    const char *src;
    src = vs_quad;
    glShaderSource(v, 1, &src, NULL);
    glCompileShader(v);
    src = fs;
    glShaderSource(f, 1, &src, NULL);
    glCompileShader(f);
    GLint ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048];
        glGetShaderInfoLog(f, sizeof log, NULL, log);
        printf("  macro shader failed: %s\n", log);
    }
    EXPECT_TRUE(ok);
    glDeleteShader(v);
    glDeleteShader(f);

    /* `: disable' must reject the qualifier (glslang enforces it) */
    GLuint d = make_program(qual, "#extension GL_KHR_blend_equation_advanced : disable");
    EXPECT_TRUE(!program_ok(d));
    glDeleteProgram(d);
}

TEST(blend_math)
{
    /* src alpha 1 pairs + one dst-alpha pair; HSL modes get valid colors.
     * Case 1 is a CTS-style premultiplied-ish src (rgb <= a, one channel ==
     * a) — it locks the ToNormal plain-division semantics (no rgb==a -> 1
     * special case) that blend_all.GL_MULTIPLY_KHR_all_qualifier caught. */
    struct { V4 s, d; bool hsl_ok; } cases[] = {
        { { 0.20f, 0.40f, 0.60f, 1.0f }, { 0.80f, 0.60f, 0.40f, 1.0f }, true },
        { { 0.75f, 0.75f, 0.50f, 0.75f }, { 0.00f, 0.00f, 0.00f, 0.75f }, false },
        { { 1.00f, 0.75f, 0.50f, 1.0f }, { 0.00f, 0.00f, 0.00f, 0.0f }, false },
        { { 0.25f, 0.50f, 0.75f, 1.0f }, { 0.10f, 0.90f, 0.40f, 0.5f }, false },
        { { 0.00f, 0.00f, 0.00f, 1.0f }, { 1.00f, 1.00f, 1.00f, 1.0f }, true },
        { { 1.00f, 0.50f, 0.25f, 1.0f }, { 0.75f, 0.25f, 1.00f, 1.0f }, true },
    };

    for (int m = 0; m < 15; m++)
    {
        char qual[64];
        snprintf(qual, sizeof qual, "layout (%s) out;", k_mode_quals[m]);
        GLuint p = make_program(qual, "#extension GL_KHR_blend_equation_advanced : require");
        if (!program_ok(p))
        {
            glDeleteProgram(p);
            continue;
        }

        for (unsigned ci = 0; ci < sizeof cases / sizeof cases[0]; ci++)
        {
            if ((m >= 11) && !cases[ci].hsl_ok) continue;   /* HSL needs valid colors */
            V4 dst = cases[ci].d, src = cases[ci].s;
            dst.c[0] = q8(dst.c[0]); dst.c[1] = q8(dst.c[1]);
            dst.c[2] = q8(dst.c[2]); dst.c[3] = q8(dst.c[3]);
            draw_one(p, m, src, dst);

            GLubyte px[4];
            glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            V4 ref = blend_ref(m, src, dst);
            float rr = q8(ref.c[0]), gg = q8(ref.c[1]), bb = q8(ref.c[2]), aa = q8(ref.c[3]);
            bool ok =
                fabsf(px[0] / 255.0f - rr) <= 1.6f / 255.0f &&
                fabsf(px[1] / 255.0f - gg) <= 1.6f / 255.0f &&
                fabsf(px[2] / 255.0f - bb) <= 1.6f / 255.0f &&
                fabsf(px[3] / 255.0f - aa) <= 1.6f / 255.0f;
            mgl_test_checks++;
            if (!ok)
                MGL_FAIL("%s case %u: got (%d,%d,%d,%d) want (%d,%d,%d,%d)",
                         k_mode_names[m], ci, px[0], px[1], px[2], px[3],
                         (int)(rr * 255.0f + 0.5f), (int)(gg * 255.0f + 0.5f),
                         (int)(bb * 255.0f + 0.5f), (int)(aa * 255.0f + 0.5f));
        }
        glDeleteProgram(p);
    }
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
    EXPECT_NO_GL_ERROR();
}

TEST(all_equations_variant)
{
    /* one program cycling equations: each is a different patched library */
    GLuint p = make_program("layout (blend_support_all_equations) out;",
                            "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));

    V4 src = { 0.25f, 0.50f, 0.75f, 1.0f }, dst = { 0.50f, 0.25f, 0.10f, 1.0f };
    dst.c[0] = q8(dst.c[0]); dst.c[1] = q8(dst.c[1]);
    dst.c[2] = q8(dst.c[2]); dst.c[3] = q8(dst.c[3]);
    for (int m = 0; m < 15; m++)
    {
        draw_one(p, m, src, dst);
        GLubyte px[4];
        glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        V4 ref = blend_ref(m, src, dst);
        mgl_test_checks++;
        if (fabsf(px[0] / 255.0f - q8(ref.c[0])) > 1.6f / 255.0f ||
            fabsf(px[1] / 255.0f - q8(ref.c[1])) > 1.6f / 255.0f ||
            fabsf(px[2] / 255.0f - q8(ref.c[2])) > 1.6f / 255.0f ||
            fabsf(px[3] / 255.0f - q8(ref.c[3])) > 1.6f / 255.0f)
            MGL_FAIL("all_equations %s: got (%d,%d,%d,%d)", k_mode_names[m],
                     px[0], px[1], px[2], px[3]);
    }
    glDeleteProgram(p);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
}

TEST(coherency_two_draws)
{
    /* no barrier between draws — the tile read must see the previous result */
    GLuint p = make_program("layout (blend_support_all_equations) out;",
                            "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));

    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
    GLenum db = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &db);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(p);
    glEnable(GL_BLEND);

    V4 first = { 0.50f, 0.50f, 0.50f, 1.0f };
    V4 second = { 1.00f, 0.50f, 0.25f, 1.0f };
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBlendEquation(GL_MULTIPLY_KHR);
    glUniform4fv(glGetUniformLocation(p, "uSrc"), 1, first.c);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    glBlendEquation(GL_DIFFERENCE_KHR);
    glUniform4fv(glGetUniformLocation(p, "uSrc"), 1, second.c);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    GLubyte px[4];
    glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);

    /* dst2 = multiply(0.5,1) = 0.5 ; then difference(0.25.. wait: second blend
     * uses |Cs - Cd| = |(1,0.5,0.25) - 0.5| = (0.5,0,0.25) */
    V4 d1 = { q8(0.5f), q8(0.5f), q8(0.5f), 1.0f };
    V4 ref = blend_ref(9, second, d1);
    mgl_test_checks++;
    if (fabsf(px[0] / 255.0f - q8(ref.c[0])) > 1.6f / 255.0f ||
        fabsf(px[1] / 255.0f - q8(ref.c[1])) > 1.6f / 255.0f ||
        fabsf(px[2] / 255.0f - q8(ref.c[2])) > 1.6f / 255.0f)
        MGL_FAIL("coherency: got (%d,%d,%d,%d)", px[0], px[1], px[2], px[3]);

    glDeleteProgram(p);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
}

TEST(missing_qualifier)
{
    GLuint p = make_program(NULL, "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));

    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
    GLenum db = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &db);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform4f(glGetUniformLocation(p, "uSrc"), 1.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendEquation(GL_MULTIPLY_KHR);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* contents untouched */
    GLubyte px[4];
    glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    mgl_test_checks++;
    if (px[0] != 0 || px[1] != 255 || px[2] != 0 || px[3] != 255)
        MGL_FAIL("missing qualifier changed contents: (%d,%d,%d,%d)", px[0], px[1], px[2], px[3]);

    glDeleteProgram(p);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
}

TEST(mrt_draw)
{
    GLuint p = make_program("layout (blend_support_all_equations) out;",
                            "#extension GL_KHR_blend_equation_advanced : require");
    EXPECT_TRUE(program_ok(p));

    glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    GLenum dbs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, dbs);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform4f(glGetUniformLocation(p, "uSrc"), 1.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendEquation(GL_DARKEN_KHR);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* both attachments untouched (white) */
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    GLubyte px[4];
    glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    mgl_test_checks++;
    if (px[0] != 255 || px[1] != 255 || px[2] != 255 || px[3] != 255)
        MGL_FAIL("MRT att0 changed: (%d,%d,%d,%d)", px[0], px[1], px[2], px[3]);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    mgl_test_checks++;
    if (px[0] != 255 || px[1] != 255 || px[2] != 255 || px[3] != 255)
        MGL_FAIL("MRT att1 changed: (%d,%d,%d,%d)", px[0], px[1], px[2], px[3]);

    /* single draw buffer again: the equation works */
    GLenum one = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &one);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(p);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo1);
}

int main(void)
{
    setup_fbos();
    RUN(ext_and_state);
    RUN(coherent_cap);
    RUN(glsl_qualifiers);
    RUN(blend_math);
    RUN(all_equations_variant);
    RUN(coherency_two_draws);
    RUN(missing_qualifier);
    RUN(mrt_draw);
    return mgl_test_summary();
}
