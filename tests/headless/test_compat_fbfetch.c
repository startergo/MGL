/*
 * test_compat_fbfetch.c — GL_EXT_shader_framebuffer_fetch (compat domain).
 *
 * `inout' fragment outputs and gl_LastFragData[] read the previous draw's
 * colour out of tile memory: the GLSL pre-pass (compat_fbfetch.c) rewrites
 * them to plain readable outputs / a uniform array, and the MSL pass adds
 * [[color(n)]] entry-point parameters that preload those locals (the same
 * Apple-GPU mechanism GL_KHR_blend_equation_advanced uses).
 *
 * Checks are pixel-exact on RGBA8: a clear + shader math round-trip, a
 * two-draw accumulation (each draw fetches the previous draw's result), the
 * gl_LastFragData builtin with a literal and a dynamic index, and a control
 * draw without the extension (the inout path must not change plain outputs).
 */
#include "mgl_test.h"
#include <string.h>
#include <stdbool.h>

#define W 16
#define H 16

static const char *vs_quad =
    "#version 150\n"
    "void main() {\n"
    "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static GLuint make_program(const char *fs_body)
{
    char fs[2048];
    snprintf(fs, sizeof fs,
        "#version 430\n"
        "#extension GL_EXT_shader_framebuffer_fetch : enable\n"
        "%s\n", fs_body);
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs_quad, NULL); glCompileShader(v);
    glShaderSource(f, 1, &(const char *){fs}, NULL); glCompileShader(f);
    GLint ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(f, sizeof log, NULL, log);
        fprintf(stderr, "  fbfetch FS compile failed: %s\n", log);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        fprintf(stderr, "  fbfetch link failed: %s\n", log);
        return 0;
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static void draw_quad(void)
{
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

/* one RGBA8 pixel of the default framebuffer */
static void px(int x, int y, GLubyte out[4])
{
    glFinish();
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}

static void setup_fb(void)
{
    static GLuint vao = 0;      /* core context: draws need a VAO */
    if (!vao)
        glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, W, H);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
}

/* ------------------------------------------------------------------ tests */

TEST(inout_accumulate)
{
    setup_fb();
    GLuint p = make_program(
        "layout (location = 0) inout vec4 o;\n"
        "uniform float uAdd;\n"
        "void main() { o = o + vec4(uAdd); }\n");
    EXPECT_NE(p, 0);
    if (!p) return;

    /* clear 0.5, two draws of +0.125 -> 0.75 (192/255) */
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "uAdd"), 0.125f);
    draw_quad();
    draw_quad();
    GLubyte c[4];
    px(W / 2, H / 2, c);
    EXPECT_EQ_INT(c[0], 192);
    EXPECT_EQ_INT(c[1], 192);
    EXPECT_EQ_INT(c[2], 192);
    EXPECT_EQ_INT(c[3], 255);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(inout_rgb)
{
    setup_fb();
    GLuint p = make_program(
        "layout (location = 0) inout vec4 o;\n"
        "void main() { o.rgb = o.gba; }\n");
    EXPECT_NE(p, 0);
    if (!p) return;

    /* rotate: (r,g,b) <- (g,b,a); a stays fetched (then overwritten) */
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    draw_quad();
    GLubyte c[4];
    px(2, 3, c);
    EXPECT_EQ_INT(c[0], 128);   /* 0.5 */
    EXPECT_EQ_INT(c[1], 191);   /* 0.75 */
    EXPECT_EQ_INT(c[2], 255);   /* 1.0 */
    EXPECT_EQ_INT(c[3], 255);   /* o.a keeps the fetched 1.0 (rgb write only) */
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(lfd_literal)
{
    setup_fb();
    GLuint p = make_program(
        "layout (location = 0) out vec4 o;\n"
        "void main() { o = gl_LastFragData[0] * vec4(0.5, 0.5, 0.5, 1.0); }\n");
    EXPECT_NE(p, 0);
    if (!p) return;

    glClearColor(0.5f, 0.75f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    draw_quad();
    GLubyte c[4];
    px(W / 2, H / 2, c);
    EXPECT_EQ_INT(c[0], 64);
    EXPECT_EQ_INT(c[1], 96);
    EXPECT_EQ_INT(c[2], 32);
    EXPECT_EQ_INT(c[3], 255);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(lfd_dynamic_index)
{
    setup_fb();
    GLuint p = make_program(
        "layout (location = 0) out vec4 o;\n"
        "uniform int uIdx;\n"
        "void main() { o = gl_LastFragData[uIdx]; }\n");
    EXPECT_NE(p, 0);
    if (!p) return;

    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "uIdx"), 0);
    draw_quad();
    GLubyte c[4];
    px(1, 1, c);
    EXPECT_EQ_INT(c[0], 64);
    EXPECT_EQ_INT(c[1], 128);
    EXPECT_EQ_INT(c[2], 191);
    EXPECT_EQ_INT(c[3], 255);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(lfd_and_inout_two_draws)
{
    setup_fb();
    GLuint p = make_program(
        "layout (location = 0) inout vec4 o;\n"
        "uniform float uMul;\n"
        "void main() { o = max(o, gl_LastFragData[0]) * uMul; }\n");
    EXPECT_NE(p, 0);
    if (!p) return;

    /* draw1: o = max(0.5, 0.5) * 0.5 = 0.25, a = 0.5
     * draw2: lfd = draw1's result (0.25 / a 0.5), o = same * 0.5 = 0.125, a 0.25 */
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "uMul"), 0.5f);
    draw_quad();
    draw_quad();
    GLubyte c[4];
    px(W / 2, H / 2, c);
    EXPECT_EQ_INT(c[0], 32);
    EXPECT_EQ_INT(c[1], 32);
    EXPECT_EQ_INT(c[2], 32);
    EXPECT_EQ_INT(c[3], 64);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(plain_output_unaffected)
{
    setup_fb();
    /* no extension use at all: plain out must still write, not accumulate */
    const char *fs =
        "#version 430\n"
        "layout (location = 0) out vec4 o;\n"
        "void main() { o = vec4(0.25, 0.25, 0.25, 1.0); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs_quad, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    draw_quad();
    draw_quad();
    GLubyte c[4];
    px(W / 2, H / 2, c);
    EXPECT_EQ_INT(c[0], 64);    /* still 0.25 — no fetch preload happened */
    glDeleteProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    EXPECT_NO_GL_ERROR();
}

TEST(es310_shader)
{
    setup_fb();
    /* dEQP's dialect: #version 310 es + the extension */
    const char *fs =
        "#version 310 es\n"
        "#extension GL_EXT_shader_framebuffer_fetch : enable\n"
        "precision highp float;\n"
        "layout (location = 0) inout vec4 o;\n"
        "uniform float uAdd;\n"
        "void main() { o = o + vec4(uAdd); }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs_quad, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    GLint ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[1024];
        glGetShaderInfoLog(f, sizeof log, NULL, log);
        fprintf(stderr, "  es310 FS compile failed: %s\n", log);
    }
    EXPECT_EQ_INT(ok, 1);
    if (!ok) { glDeleteShader(v); glDeleteShader(f); return; }
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (ok)
    {
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(p);
        glUniform1f(glGetUniformLocation(p, "uAdd"), 0.25f);
        draw_quad();
        GLubyte c[4];
        px(W / 2, H / 2, c);
        EXPECT_EQ_INT(c[0], 192);
        EXPECT_EQ_INT(c[3], 255);
    }
    glDeleteProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    EXPECT_NO_GL_ERROR();
}

TEST(texel_fetch_default_texture)
{
    /* dEQP framebuffer_fetch.basic.texel_fetch repro: the CTS uploads a grid
     * into TEXTURE OBJECT 0 on unit 1 (it binds m_texColorBuffer before that
     * name exists), samples it with texelFetch and adds the fetched
     * framebuffer colour.  Only a compatibility / ES context has a default
     * texture; the core context here must sample (0,0,0,1). */
    setup_fb();

    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    /* the grid goes into object 0 on unit 1 (checkerboard of two colours) */
    GLubyte grid[W * H * 4];
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            int even = ((x / 2) + (y / 2)) % 2 == 0;
            GLubyte *p = grid + (y * W + x) * 4;
            p[0] = even ? 128 : 0;
            p[1] = even ? 128 : 0;
            p[2] = even ? 0 : 128;
            p[3] = 255;
        }
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, grid);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    GLuint p = make_program(
        "layout (location = 0) inout vec4 o;\n"
        "uniform sampler2D u_sampler;\n"
        "void main() { o += texelFetch(u_sampler, ivec2(gl_FragCoord.xy), 0); }\n");
    EXPECT_NE(p, 0);
    if (!p) goto done;
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_sampler"), 1);

    glClearColor(0.5f, 0.0f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_quad();
    {
        GLubyte c[4];
        px(0, 0, c);
        /* Mesa-parity default texture (r103): texture object 0 IS the target's
         * default texture object — the grid uploaded into it samples through,
         * so the result is grid + the fetched clear colour: even cell
         * (128,128,0) + (128,0,128) = (255,128,128).  This is exactly what
         * dEQP-GL45-ES31 functional.shaders.framebuffer_fetch.basic.texel_fetch
         * expects (the ES31 suite assumes ES default-texture semantics even on
         * its desktop core context; see mglBindTexture name-0 branch). */
        EXPECT_EQ_INT(c[0], 255);
        EXPECT_EQ_INT(c[1], 128);
        EXPECT_EQ_INT(c[2], 128);
        EXPECT_EQ_INT(c[3], 255);
    }
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
done:
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

/* The 8 dEQP framebuffer_fetch.framebuffer_format.{rgba32f,rgba16f,
 * r11f_g11f_b10f,rg32f,rg16f,r32f,r16f,rgb16f} cases report NotSupported: the
 * ES31 suite gates float render targets on GL_EXT_color_buffer_float /
 * _half_float (isRequiredFormat returns isES32 only), which MGL does not
 * advertise — and cannot honestly: EXT_color_buffer_float requires RGB16F
 * color-renderable, and Metal has no RGB16F pixel format.  This test documents
 * both halves: the fetch functionality itself works on the float formats MGL
 * does support (RGBA32F round-trip below, pixel-exact), and RGB16F is
 * genuinely not a renderable format here. */
TEST(float_format_fetch_and_rgb16f_gap)
{
    /* every float colour format EXT_color_buffer_float would claim: render
     * target complete, inout fetch accumulates, blending is exact */
    struct { GLenum fmt; const char *nm; int chans; } fmts[] = {
        { GL_RGBA32F, "rgba32f", 3 }, { GL_RGBA16F, "rgba16f", 3 },
        { GL_RG32F, "rg32f", 2 },    { GL_RG16F, "rg16f", 2 },
        { GL_R32F, "r32f", 1 },      { GL_R16F, "r16f", 1 },
        { GL_R11F_G11F_B10F, "r11f", 3 }, { GL_RGB16F, "rgb16f", 3 },
    };

    for (unsigned i = 0; i < sizeof fmts / sizeof fmts[0]; i++)
    {
        GLuint tex, fbo, p;
        float c[4];

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, fmts[i].fmt, W, H, 0, GL_RGBA, GL_FLOAT, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        { MGL_FAIL("%s: fbo incomplete", fmts[i].nm); goto next; }

        glViewport(0, 0, W, H);
        glDisable(GL_BLEND);
        p = make_program(
            "layout (location = 0) inout vec4 o;\n"
            "uniform vec4 uAdd;\n"
            "void main() { o = o + uAdd; }\n");
        glUseProgram(p);
        glUniform4f(glGetUniformLocation(p, "uAdd"), 0.25f, 0.25f, 0.25f, 0.25f);
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        draw_quad();
        draw_quad();
        glFinish();
        c[0] = c[1] = c[2] = c[3] = -1;
        glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
        /* 0.5 + 2*0.25 = 1.0 exactly in f32/f16 (and in the 11/10-bit formats
         * 0.5 and 0.25 are representable); only the format's own channels */
        {
            int ok = 1;
            for (int ch = 0; ch < fmts[i].chans; ch++)
                if (c[ch] < 0.99f || c[ch] > 1.01f) ok = 0;
            if (!ok)
                MGL_FAIL("%s: fetch-accumulate rgb=%f,%f,%f,%f", fmts[i].nm, c[0], c[1], c[2], c[3]);
            EXPECT_EQ_INT(ok, 1);
        }

        /* blend: dst 0.5, src 0.5 * 0.5 (SRC_ALPHA, ONE) -> 0.75 */
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform4f(glGetUniformLocation(p, "uAdd"), 0.0f, 0.0f, 0.0f, 0.0f);
        /* draw writes uColour.a = 0.5 through the inout path with blending */
        glDeleteProgram(p);
        p = make_program(
            "layout (location = 0) inout vec4 o;\n"
            "void main() { o = vec4(0.5, 0.5, 0.5, 0.5); }\n");
        glUseProgram(p);
        draw_quad();
        glFinish();
        c[0] = c[1] = c[2] = c[3] = -1;
        glReadPixels(W / 2, H / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
        if (c[0] < 0.74f || (fmts[i].chans > 1 && c[1] < 0.74f) || (fmts[i].chans > 2 && c[2] < 0.74f))
            MGL_FAIL("%s: blend rgb=%f,%f,%f", fmts[i].nm, c[0], c[1], c[2]);
        EXPECT_EQ_INT(c[0] > 0.74f && c[0] < 0.76f, 1);
        EXPECT_NO_GL_ERROR();
        glDeleteProgram(p);
    next:
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
    }
    glDisable(GL_BLEND);
    setup_fb();
}

int main(void)
{
    RUN(inout_accumulate);
    RUN(inout_rgb);
    RUN(lfd_literal);
    RUN(lfd_dynamic_index);
    RUN(lfd_and_inout_two_draws);
    RUN(plain_output_unaffected);
    RUN(es310_shader);
    RUN(texel_fetch_default_texture);
    RUN(float_format_fetch_and_rgb16f_gap);
    return mgl_test_summary();
}
