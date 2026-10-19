#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "spv_vsfs.h"

/* GL 4.6 requires SPIR-V as a shader binary format: GL_NUM_SHADER_BINARY_FORMATS must
 * be 1, GL_SHADER_BINARY_FORMATS must return GL_SHADER_BINARY_FORMAT_SPIR_V,
 * GL_ARB_gl_spirv / GL_ARB_spirv_extensions must be advertised, and
 * glShaderBinary + glSpecializeShader must link and draw. */

static bool has_extension(const char *name)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
    {
        const GLubyte *e = glGetStringi(GL_EXTENSIONS, (GLuint)i);
        if (e && !strcmp((const char *)e, name))
            return true;
    }
    return false;
}

TEST(ShaderBinaryFormats)
{
    GLint num = 0, fmt[4] = { 0, 0, 0, 0 };

    glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS, &num);
    printf("GL_NUM_SHADER_BINARY_FORMATS = %d\n", (int)num);
    EXPECT_EQ_INT(num, 1);

    glGetIntegerv(GL_SHADER_BINARY_FORMATS, fmt);
    printf("GL_SHADER_BINARY_FORMATS[0] = 0x%x (SPIR-V is 0x%x)\n", (int)fmt[0],
           (int)GL_SHADER_BINARY_FORMAT_SPIR_V);
    EXPECT_EQ_INT(fmt[0], GL_SHADER_BINARY_FORMAT_SPIR_V);

    EXPECT_TRUE(has_extension("GL_ARB_gl_spirv"));
    EXPECT_TRUE(has_extension("GL_ARB_spirv_extensions"));

    {
        GLint ns = 0;
        glGetIntegerv(GL_NUM_SPIR_V_EXTENSIONS, &ns);
        printf("GL_NUM_SPIR_V_EXTENSIONS = %d\n", (int)ns);
        EXPECT_TRUE(ns >= 1);
        for (GLint i = 0; i < ns; i++)
        {
            const GLubyte *e = glGetStringi(GL_SPIR_V_EXTENSIONS, (GLuint)i);
            printf("  SPIR-V extension %d: %s\n", (int)i, e ? (const char *)e : "(null)");
            EXPECT_TRUE(e != NULL && *e != 0);
        }
    }
}

TEST(ShaderBinarySpecializeDraw)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p = glCreateProgram();
    GLint link = 0;
    GLuint vao, b[2];

    /* binary upload of pre-generated SPIR-V, then specialization to the entry point */
    glShaderBinary(1, &vs, GL_SHADER_BINARY_FORMAT_SPIR_V, spv_vs, spv_vs_WORDS * 4);
    EXPECT_NO_GL_ERROR();
    glGetShaderiv(vs, GL_SHADER_TYPE, &link);
    EXPECT_EQ_INT(link, GL_VERTEX_SHADER);
    glSpecializeShader(vs, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();
    {
        GLint st = 0;
        glGetShaderiv(vs, GL_COMPILE_STATUS, &st);
        EXPECT_EQ_INT(st, GL_TRUE);
    }

    glShaderBinary(1, &fs, GL_SHADER_BINARY_FORMAT_SPIR_V, spv_fs, spv_fs_WORDS * 4);
    EXPECT_NO_GL_ERROR();
    glSpecializeShader(fs, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();

    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link != GL_TRUE)
    {
        char log[1024] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("link log: %s\n", log);
    }
    glUseProgram(p);

    /* a quad carrying per-vertex colours; the FS passes v_col through */
    {
        const float pos[4][4] = { { -1,-1,0,1 }, { 1,-1,0,1 }, { 1,1,0,1 }, { -1,1,0,1 } };
        const float col[4][4] = { { 0,1,0,1 }, { 0,1,0,1 }, { 0,1,0,1 }, { 0,1,0,1 } };
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
    }

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();
    glFinish();

    {
        GLint vp[4];
        float c[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        glReadPixels(vp[2] / 2, vp[3] / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
        printf("centre pixel: %.2f %.2f %.2f\n", c[0], c[1], c[2]);
        EXPECT_TRUE(c[1] > 0.9f && c[0] < 0.1f && c[2] < 0.1f);
    }
    glDeleteProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

TEST(ShaderBinaryErrors)
{
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);

    /* an unknown binary format is GL_INVALID_ENUM */
    glShaderBinary(1, &sh, 0xdeadbeef, spv_vs, spv_vs_WORDS * 4);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_ENUM);

    /* negative count/length */
    glShaderBinary(-1, &sh, GL_SHADER_BINARY_FORMAT_SPIR_V, spv_vs, spv_vs_WORDS * 4);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);
    while (glGetError())
        ;
    glDeleteShader(sh);
}

int main(void)
{
    RUN(ShaderBinaryFormats);
    RUN(ShaderBinarySpecializeDraw);
    RUN(ShaderBinaryErrors);
    return mgl_test_summary();
}
