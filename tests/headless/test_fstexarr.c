/* probe: FS sampler ARRAY (2 elements) fetch — textures-domain baseline */
#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kFs =
    "#version 450 core\n"
    "uniform sampler2D tex[2];\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ float a = texelFetch(tex[0], ivec2(0), 0).r; float b = texelFetch(tex[1], ivec2(0), 0).g;\n"
    "  c0 = vec4(a, b, 0, 1); }\n";
static const char *kFs2 =
    "#version 450 core\n"
    "uniform sampler2D tex0;\n"
    "uniform sampler2D tex1;\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ float a = texelFetch(tex0, ivec2(0), 0).r; float b = texelFetch(tex1, ivec2(0), 0).g;\n"
    "  c0 = vec4(a, b, 0, 1); }\n";

static GLuint mkprog(const char *fs)
{
    GLuint p = glCreateProgram();
    const char *s[2] = { kVs, fs };
    GLenum t[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 2; i++)
    {
        GLuint sh = glCreateShader(t[i]);
        glShaderSource(sh, 1, &s[i], NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                   fprintf(stderr, "compile: %s\n", log); return 0; }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
               fprintf(stderr, "link: %s\n", log); return 0; }
    return p;
}

static GLuint g_fbo, g_ctex, g_vao, g_vbo, g_tex[2];

static void setup_common(void)
{
    glGenTextures(1, &g_ctex);
    glBindTexture(GL_TEXTURE_2D, g_ctex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_ctex, 0);
    glViewport(0, 0, 64, 64);

    static float d0[4] = { 1.0f, 0, 0, 1.0f }, d1[4] = { 0, 1.0f, 0, 1.0f };
    for (int i = 0; i < 2; i++)
    {
        glGenTextures(1, &g_tex[i]);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, g_tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, i ? d1 : d0);

    }
    glActiveTexture(GL_TEXTURE0);

    /* read both textures back CPU-side to prove their contents */
    for (int i = 0; i < 2; i++)
    {
        float rb[4] = { -1, -1, -1, -1 };
        glActiveTexture(GL_TEXTURE0 + i);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, rb);
        printf("unit %d readback = %g %g %g %g\n", i, rb[0], rb[1], rb[2], rb[3]);
    }
    glActiveTexture(GL_TEXTURE0);

    glGenVertexArrays(1, &g_vao);
    glGenBuffers(1, &g_vbo);
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    float q[8] = { -1,-1, 1,-1, -1,1, 1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static GLubyte draw_and_read(GLuint p)
{
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    GLubyte c[4] = { 0, 0, 0, 0 };
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    return c[1];
}

TEST(FsTexArray)
{
    setup_common();
    GLuint p = mkprog(kFs);
    if (!p) { MGL_FAIL("array program"); return; }
    GLint l0 = glGetUniformLocation(p, "tex");
    if (l0 >= 0) glProgramUniform1i(p, l0, 0);
    GLubyte g = draw_and_read(p);
    printf("fs array fetch green = %d\n", g);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(g > 100);
}

TEST(FsTwoSamplers)
{
    GLuint p = mkprog(kFs2);
    if (!p) { MGL_FAIL("two-sampler program"); return; }
    GLint l0 = glGetUniformLocation(p, "tex0");
    GLint l1 = glGetUniformLocation(p, "tex1");
    printf("loc tex0=%d tex1=%d\n", l0, l1);
    if (l0 >= 0) glProgramUniform1i(p, l0, 0);
    if (l1 >= 0) glProgramUniform1i(p, l1, 1);
    GLubyte g = draw_and_read(p);
    printf("fs two-sampler fetch green = %d\n", g);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(g > 100);
}

int main(void)
{
    RUN(FsTexArray);
    RUN(FsTwoSamplers);
    return mgl_test_summary();
}
