/*
 * test_compat_pipefbo.c — PSO fast-path vs a re-attached FBO texture (compat
 * domain; the rule geom introduced at e767e6a, pinned here for the compat
 * pipeline cache too).
 *
 * The pipeline cache key covers the framebuffer POINTER and the attachment
 * FORMATS, so re-attaching a texture with an identical pixel format leaves
 * the key unchanged.  When DIRTY_FBO arrives through such a re-attach, the
 * descriptor path is what re-binds the new attachment to the render pass; a
 * cache HIT that skips it keeps drawing into the previous texture.
 * Sequence: FBO A renders green into T1, then re-attaches T2 (same RGBA8
 * format) and renders yellow.  T2 must be yellow and T1 must still be green.
 */
#include "mgl_test.h"

static const char *kVS =
    "#version 460 core\n"
    "layout(location = 0) in vec2 p;\n"
    "void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char *kFS =
    "#version 460 core\n"
    "uniform vec4 col;\n"
    "layout(location = 0) out vec4 o;\n"
    "void main() { o = col; }\n";

static GLuint prog, vao, vbo, fboA, t1, t2;

static void draw_full(GLubyte r, GLubyte g, GLubyte b)
{
    glUseProgram(prog);
    glUniform4f(glGetUniformLocation(prog, "col"), r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void read_attachment(GLuint tex, GLubyte *px)
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glDeleteFramebuffers(1, &fbo);
}

TEST(reattach_same_format_bind)
{
    const GLuint sh[2] = { glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER) };
    const char *src[2] = { kVS, kFS };
    for (int i = 0; i < 2; i++)
    {
        glShaderSource(sh[i], 1, &src[i], NULL);
        glCompileShader(sh[i]);
        GLint ok = 0; glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
        EXPECT_TRUE(ok);
    }
    prog = glCreateProgram();
    glAttachShader(prog, sh[0]);
    glAttachShader(prog, sh[1]);
    glLinkProgram(prog);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);

    static const GLfloat quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    /* T1 starts blue so a "no draw at all" outcome is distinguishable */
    static GLubyte fill1[4 * 4 * 4] = { 0 }, fill2[4 * 4 * 4] = { 0 };
    for (int i = 0; i < 4 * 4; i++)
    {
        fill1[i * 4 + 2] = 255; fill1[i * 4 + 3] = 255;   /* T1: blue */
        fill2[i * 4 + 0] = 255; fill2[i * 4 + 3] = 255;   /* T2: red */
    }
    glGenFramebuffers(1, &fboA);
    glGenTextures(1, &t1);
    glGenTextures(1, &t2);
    glBindTexture(GL_TEXTURE_2D, t1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, fill1);
    glBindTexture(GL_TEXTURE_2D, t2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, fill2);

    glBindFramebuffer(GL_FRAMEBUFFER, fboA);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t1, 0);
    EXPECT_TRUE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);

    /* 1. green into T1 */
    draw_full(0, 255, 0);
    EXPECT_NO_GL_ERROR();

    /* 2. re-attach T2 — same pixel format, so the PSO key does not change */
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t2, 0);
    EXPECT_TRUE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    /* 3. yellow must land in T2 */
    draw_full(255, 255, 0);
    EXPECT_NO_GL_ERROR();

    GLubyte p1[4] = { 0 }, p2[4] = { 0 };
    read_attachment(t1, p1);
    read_attachment(t2, p2);
    EXPECT_TRUE(p1[1] == 255 && p1[2] == 0 && p1[0] == 0);
    EXPECT_TRUE(p2[0] == 255 && p2[1] == 255 && p2[2] == 0);
    if (!(p1[1] == 255 && p1[2] == 0))
        MGL_FAIL("T1 = (%d,%d,%d): expected green (yellow leaked into the old attachment)",
                 p1[0], p1[1], p1[2]);
    if (!(p2[0] == 255 && p2[1] == 255 && p2[2] == 0))
        MGL_FAIL("T2 = (%d,%d,%d): expected yellow — the re-attached texture was not bound",
                 p2[0], p2[1], p2[2]);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fboA);
    glDeleteTextures(1, &t1);
    glDeleteTextures(1, &t2);
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

int main(void)
{
    RUN(reattach_same_format_bind);
    return mgl_test_summary();
}
