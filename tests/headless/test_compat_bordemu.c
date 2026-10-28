/*
 * test_compat_bordemu.c — GL_CLAMP_TO_BORDER with border colours Metal's
 * samplers cannot express (integer colours like (255,255,255,255) on R32I,
 * arbitrary float colours), emulated by the per-draw MSL rewrite
 * (compat_border.c).  Regression for KHR-GL46.texture_border_clamp
 * Texture{2D,2DArray,3D}R32{I,UI}.
 */
#include "mgl_test.h"
#include <string.h>
#include <stdbool.h>

static const char *k_vs =
    "#version 460 core\n"
    "in vec2 a; out vec2 uv;\n"
    "void main() { uv = a * 1.5 + 0.25; gl_Position = vec4(a, 0.0, 1.0); }\n";

static const char *k_fs_i2d =
    "#version 460 core\n"
    "uniform isampler2D s; in vec2 uv; out ivec4 o;\n"
    "void main() { o = texture(s, uv); }\n";
static const char *k_fs_u2d =
    "#version 460 core\n"
    "uniform usampler2D s; in vec2 uv; out uvec4 o;\n"
    "void main() { o = texture(s, uv); }\n";
static const char *k_fs_f2d =
    "#version 460 core\n"
    "uniform sampler2D s; in vec2 uv; out vec4 o;\n"
    "void main() { o = texture(s, uv); }\n";
static const char *k_fs_i2darr =
    "#version 460 core\n"
    "uniform isampler2DArray s; in vec2 uv; out ivec4 o;\n"
    "void main() { o = texture(s, vec3(uv, 0.0)); }\n";
static const char *k_fs_i3d =
    "#version 460 core\n"
    "uniform isampler3D s; in vec2 uv; out ivec4 o;\n"
    "void main() { o = texture(s, vec3(uv, 0.5)); }\n";

#define W 96
#define H 96

static GLuint g_vao, g_vbo, g_prog[5], g_tex[5], g_samp, g_fbo;

static GLuint build(const char *fs)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &k_vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    glBindAttribLocation(p, 0, "a");
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok)
    {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("link failed: %s\n", log);
    }
    return p;
}

static GLuint fbo_with(GLenum ifmt)
{
    GLuint fbo, tex;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    if (ifmt == GL_RGBA16I || ifmt == GL_RGBA16UI)
        glTexImage2D(GL_TEXTURE_2D, 0, ifmt, W, H, 0, GL_RGBA_INTEGER, GL_INT, NULL);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, ifmt, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    EXPECT_NO_GL_ERROR();
    return fbo;
}

static void setup(void)
{
    static const float quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    static GLint  tex_i[1] = { 7 };
    static GLuint tex_u[1] = { 7 };
    static unsigned char tex_f[8 * 8 * 4];

    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    glGenBuffers(1, &g_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    g_prog[0] = build(k_fs_i2d);
    g_prog[1] = build(k_fs_u2d);
    g_prog[2] = build(k_fs_f2d);
    g_prog[3] = build(k_fs_i2darr);
    g_prog[4] = build(k_fs_i3d);

    glGenTextures(5, g_tex);
    glBindTexture(GL_TEXTURE_2D, g_tex[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1, 1, 0, GL_RED_INTEGER, GL_INT, tex_i);
    glBindTexture(GL_TEXTURE_2D, g_tex[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 1, 1, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, tex_u);
    for (int i = 0; i < 8 * 8 * 4; i += 4) { tex_f[i] = 0; tex_f[i + 1] = 0; tex_f[i + 2] = 0; tex_f[i + 3] = 255; }
    glBindTexture(GL_TEXTURE_2D, g_tex[2]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_f);
    {
        GLint v[64];
        for (int i = 0; i < 64; i++) v[i] = 7;
        glBindTexture(GL_TEXTURE_3D, g_tex[3]);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32I, 4, 4, 4, 0, GL_RED_INTEGER, GL_INT, v);
    }
    glGenSamplers(1, &g_samp);
    EXPECT_NO_GL_ERROR();
}

/* uv of a pixel centre: the quad's uv = ndc * 1.5 + 0.25, spanning [-1.25,1.75] */
static float uv_at(int x) { return 1.5f * (2.0f * ((float)x + 0.5f) / (float)W - 1.0f) + 0.25f; }
static bool inside(int x, int y) { float u = uv_at(x), v = uv_at(y); return u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f; }

static void draw_quad(void)
{
    glViewport(0, 0, W, H);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
}

static GLint read_i(int x, int y)
{
    GLint px[4] = { -999, -999, -999, -999 };
    glReadPixels(x, y, 1, 1, GL_RGBA_INTEGER, GL_INT, px);
    EXPECT_NO_GL_ERROR();
    return px[0];
}

static void border_sampler(GLint wrap_r)
{
    glSamplerParameteri(g_samp, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(g_samp, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(g_samp, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(g_samp, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glSamplerParameteri(g_samp, GL_TEXTURE_WRAP_R, wrap_r);
    EXPECT_NO_GL_ERROR();
}

/* check every sampled pixel: inside -> 7, border -> want */
static void check_grid(int want, const char *tag)
{
    int bad = 0;
    for (int y = 0; y < H; y += 7)
        for (int x = 0; x < W; x += 7)
        {
            int w = inside(x, y) ? 7 : want;
            int got = read_i(x, y);
            if (got != w) { if (bad < 4) printf("%s (%d,%d) want %d got %d\n", tag, x, y, w, got); bad++; }
        }
    EXPECT_EQ_INT(bad, 0);
}

/* ---- R32I / R32UI + sampler Iiv/Iuiv border, NEAREST ---------------------- */
TEST(int_border_2d)
{
    setup();
    g_fbo = fbo_with(GL_RGBA16I);
    static const GLint bcol_i[4] = { 255, 255, 255, 255 };

    glUseProgram(g_prog[0]);
    glUniform1i(glGetUniformLocation(g_prog[0], "s"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_tex[0]);
    glBindSampler(0, g_samp);
    border_sampler(GL_CLAMP_TO_BORDER);
    glSamplerParameterIiv(g_samp, GL_TEXTURE_BORDER_COLOR, bcol_i);
    draw_quad();
    check_grid(255, "R32I");

    /* border-colour change on the *same* sampler between draws must re-bake
     * the pipeline variant (rs_pre_draw signature check) */
    static const GLint b2[4] = { 42, 42, 42, 42 };
    glSamplerParameterIiv(g_samp, GL_TEXTURE_BORDER_COLOR, b2);
    draw_quad();
    check_grid(42, "R32I-reborder");

    /* plain texture params (no sampler object): the texture's own border */
    glBindSampler(0, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterIiv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, b2);
    draw_quad();
    check_grid(42, "R32I-texparam");

    /* R32UI through the same table (unsigned literal) */
    g_fbo = fbo_with(GL_RGBA16UI);
    glUseProgram(g_prog[1]);
    glUniform1i(glGetUniformLocation(g_prog[1], "s"), 0);
    glBindSampler(0, g_samp);
    border_sampler(GL_CLAMP_TO_BORDER);
    static const GLuint bu2[4] = { 65001, 65001, 65001, 65001 };
    glSamplerParameterIuiv(g_samp, GL_TEXTURE_BORDER_COLOR, bu2);
    glBindTexture(GL_TEXTURE_2D, g_tex[1]);
    draw_quad();
    check_grid(65001, "R32UI");
}

/* ---- R32I 2D array and 3D -------------------------------------------------- */
TEST(int_border_array_3d)
{
    g_fbo = fbo_with(GL_RGBA16I);

    /* 1x1x1 array of 7 on its own object (NULL would leave the texels zero) */
    {
        GLint a7[1] = { 7 };
        glBindTexture(GL_TEXTURE_2D_ARRAY, g_tex[4]);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32I, 1, 1, 1, 0, GL_RED_INTEGER, GL_INT, a7);
        EXPECT_NO_GL_ERROR();
    }
    glUseProgram(g_prog[3]);
    glUniform1i(glGetUniformLocation(g_prog[3], "s"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, g_tex[4]);
    glBindSampler(0, g_samp);
    border_sampler(GL_REPEAT);   /* arrays ignore R */
    static const GLint bcol[4] = { 255, 255, 255, 255 };
    glSamplerParameterIiv(g_samp, GL_TEXTURE_BORDER_COLOR, bcol);
    draw_quad();
    check_grid(255, "2DArray");

    /* 3D with r inside the volume: s/t borders only */
    glUseProgram(g_prog[4]);
    glUniform1i(glGetUniformLocation(g_prog[4], "s"), 0);
    glBindTexture(GL_TEXTURE_3D, g_tex[3]);
    border_sampler(GL_CLAMP_TO_BORDER);
    draw_quad();
    check_grid(255, "3D");
}

/* ---- RGBA8 float border colour (unrepresentable) + NEAREST ---------------- */
TEST(float_border_color)
{
    g_fbo = fbo_with(GL_RGBA8);
    glUseProgram(g_prog[2]);
    glUniform1i(glGetUniformLocation(g_prog[2], "s"), 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, g_tex[2]);
    glBindSampler(2, g_samp);
    border_sampler(GL_CLAMP_TO_BORDER);
    static const GLfloat bcol[4] = { 63.0f / 255.0f, 127.0f / 255.0f, 191.0f / 255.0f, 1.0f };
    glSamplerParameterfv(g_samp, GL_TEXTURE_BORDER_COLOR, bcol);
    draw_quad();

    int bad = 0;
    for (int y = 0; y < H; y += 5)
        for (int x = 0; x < W; x += 5)
        {
            GLubyte want[4] = { 0, 0, 0, 255 }, got[4] = { 9, 9, 9, 9 };
            if (!inside(x, y)) { want[0] = 63; want[1] = 127; want[2] = 191; }
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
            if (memcmp(want, got, 4)) { if (bad < 4) printf("RGBA8 (%d,%d) want %d,%d,%d,%d got %d,%d,%d,%d\n", x, y, want[0], want[1], want[2], want[3], got[0], got[1], got[2], got[3]); bad++; }
        }
    EXPECT_EQ_INT(bad, 0);

    /* a Metal-representable colour stays on the hardware path (REPEAT: the
     * uniform 8x8 texture reads back its fill colour everywhere) */
    glSamplerParameterf(g_samp, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameterf(g_samp, GL_TEXTURE_WRAP_T, GL_REPEAT);
    draw_quad();
    bad = 0;
    for (int y = 0; y < H; y += 5)
        for (int x = 0; x < W; x += 5)
        {
            GLubyte want[4] = { 0, 0, 0, 255 }, got[4];
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
            if (memcmp(want, got, 4)) { bad++; }
        }
    EXPECT_EQ_INT(bad, 0);
}

int main(void)
{
    RUN(int_border_2d);
    RUN(int_border_array_3d);
    RUN(float_border_color);
    return mgl_test_summary();
}
