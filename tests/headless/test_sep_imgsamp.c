#include "mgl_test.h"
#include <stdio.h>
#include <stdbool.h>
#include "spv_vsfs.h"
#include "spv_sepimg.h"

/* Vulkan-style SEPARATE images and samplers (OpTypeImage + OpTypeSampler, MSL
 * [[texture(n)]] and [[sampler(m)]]) loaded through glShaderBinary: the image
 * (binding 2) must read texture unit 2 and the sampler (binding 3) must take its
 * state from unit 3 — a sampler object bound there, else the paired texture's
 * own parameters.  Before the fix the render stages bound neither argument (the
 * sample read zeros) and the compute stage ignored both resource classes.
 *
 * Texture: 2x1 RGBA8, red | green, LINEAR, sampled at u = 0.6:
 *   LINEAR  -> green weight (0.6 - 0.25) / 0.5 = 0.7, red 0.3
 *   NEAREST -> pure green */

static GLuint g_tex, g_nearest;

static void setup_texture(void)
{
    static const GLubyte px[2][4] = { { 255, 0, 0, 255 }, { 0, 255, 0, 255 } };
    glGenTextures(1, &g_tex);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, g_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glActiveTexture(GL_TEXTURE0);

    glGenSamplers(1, &g_nearest);
    glSamplerParameteri(g_nearest, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(g_nearest, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(g_nearest, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(g_nearest, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    EXPECT_NO_GL_ERROR();
}

static GLuint spirv_shader(GLenum type, const uint32_t *words, int nwords)
{
    GLuint sh = glCreateShader(type);
    GLint st = 0;
    glShaderBinary(1, &sh, GL_SHADER_BINARY_FORMAT_SPIR_V, words, nwords * 4);
    glSpecializeShader(sh, "main", 0, NULL, NULL);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &st);
    EXPECT_EQ_INT(st, GL_TRUE);
    return sh;
}

static GLuint link_program(GLuint a, GLuint b)
{
    GLuint p = glCreateProgram();
    GLint link = 0;
    glAttachShader(p, a);
    if (b)
        glAttachShader(p, b);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link != GL_TRUE)
    {
        char log[1024] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("link log: %s\n", log);
    }
    return p;
}

static void expect_linear(const float c[4], const char *what)
{
    printf("%s (texture LINEAR): %.3f %.3f %.3f %.3f\n", what, c[0], c[1], c[2], c[3]);
    EXPECT_NEAR(c[0], 0.3, 0.06);
    EXPECT_NEAR(c[1], 0.7, 0.06);
    EXPECT_NEAR(c[2], 0.0, 0.02);
}

static void expect_nearest(const float c[4], const char *what)
{
    printf("%s (sampler object NEAREST on unit 3): %.3f %.3f %.3f %.3f\n", what, c[0], c[1], c[2], c[3]);
    EXPECT_NEAR(c[0], 0.0, 0.02);
    EXPECT_NEAR(c[1], 1.0, 0.02);
    EXPECT_NEAR(c[2], 0.0, 0.02);
}

TEST(SeparateImageSamplerDraw)
{
    GLuint vs = spirv_shader(GL_VERTEX_SHADER, spv_vs, spv_vs_WORDS);
    GLuint fs = spirv_shader(GL_FRAGMENT_SHADER, spv_sep_fs, (int)(sizeof spv_sep_fs / 4));
    GLuint p = link_program(vs, fs);
    GLuint vao, b[2];
    GLint vp[4];
    float c[4];
    const float pos[4][4] = { { -1,-1,0,1 }, { 1,-1,0,1 }, { 1,1,0,1 }, { -1,1,0,1 } };
    const float col[4][4] = { { 0,0,1,1 }, { 0,0,1,1 }, { 0,0,1,1 }, { 0,0,1,1 } };

    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    glGetIntegerv(GL_VIEWPORT, vp);

    /* no sampler object: the sampler pairs with the image's texture (LINEAR) */
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glFinish();
    glReadPixels(vp[2] / 2, vp[3] / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
    expect_linear(c, "draw");

    /* a sampler object on the SAMPLER's unit (3) must apply at the sampler slot */
    glBindSampler(3, g_nearest);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glFinish();
    glReadPixels(vp[2] / 2, vp[3] / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
    expect_nearest(c, "draw");
    glBindSampler(3, 0);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(2, b);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

static void run_compute(GLuint p, GLuint ssbo, float out[4])
{
    const float zero[4] = { -1, -1, -1, -1 };
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof zero, zero);
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16, out);
}

TEST(SeparateImageSamplerCompute)
{
    GLuint cs = spirv_shader(GL_COMPUTE_SHADER, spv_sep_cs, (int)(sizeof spv_sep_cs / 4));
    GLuint p = link_program(cs, 0);
    GLuint ssbo;
    float c[4];

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16, NULL, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    run_compute(p, ssbo, c);
    expect_linear(c, "compute");

    glBindSampler(3, g_nearest);
    run_compute(p, ssbo, c);
    expect_nearest(c, "compute");
    glBindSampler(3, 0);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(1, &ssbo);
    glDeleteProgram(p);
    glDeleteShader(cs);
}

TEST(Setup) { setup_texture(); }

int main(void)
{
    RUN(Setup);
    RUN(SeparateImageSamplerDraw);
    RUN(SeparateImageSamplerCompute);
    return mgl_test_summary();
}
