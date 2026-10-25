/* test_csdbg.c — debug harness for the encoder-worker thread
 * (MGL_ENCODER_THREAD=1): a trimmed copy of test_geom.c's block_vs_fs_control.
 * Run under the flag (and with MGL_CS_STATS=1) to check the recorded path
 * renders the same as the direct one. */
#include "mgl_test.h"

#include <stdio.h>

static GLuint program(const char *vs, const char *gs, const char *fs)
{
    char log[4096];
    GLint ok = 0;
    GLuint p = glCreateProgram();
    GLuint sh[3];
    int n = 0;
    struct { GLenum t; const char *src; } stages[3] = {
        { GL_VERTEX_SHADER, vs }, { GL_GEOMETRY_SHADER, gs }, { GL_FRAGMENT_SHADER, fs }
    };
    for (int i = 0; i < 3; i++)
    {
        if (!stages[i].src)
            continue;
        sh[n] = glCreateShader(stages[i].t);
        glShaderSource(sh[n], 1, &stages[i].src, NULL);
        glCompileShader(sh[n]);
        glGetShaderiv(sh[n], GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            glGetShaderInfoLog(sh[n], sizeof log, NULL, log);
            MGL_FAIL("compile: %s", log);
            return 0;
        }
        glAttachShader(p, sh[n]);
        n++;
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link: %s", log);
        return 0;
    }
    return p;
}

static GLuint fbo, tex;
static void block_fbo(int w, int h)
{
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        MGL_FAIL("fbo incomplete");
}

static void px(int x, int y, GLubyte *c)
{
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    EXPECT_NO_GL_ERROR();
}

static void setup(GLuint n, const float *v)
{
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n * 2 * sizeof(float), v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

TEST(block_vs_fs_control)
{
    block_fbo(256, 256);
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out VOut { vec4 a; vec2 b; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); a = vec4(pos, 0.5, 1.0); b = pos * 0.5; }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in VOut { vec4 a; vec2 b; };\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 0.5 + b.x, 0.25, 1.0); }\n";
    GLuint p = program(vs, NULL, fs);
    glUseProgram(p);
    glViewport(0, 0, 256, 256);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    { static const float t[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f }; setup(6, t); }
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(128, 128, c);
    printf("px(128,128) = %d %d %d %d\n", c[0], c[1], c[2], c[3]);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(block_vs_fs_control);
    return mgl_test_summary();
}
