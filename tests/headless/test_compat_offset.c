/* test_compat_offset.c — does an offset-only glVertexAttribPointer re-spec
 * reach the encoder?  Two meshes in one VBO at different offsets; the second
 * draw re-specifies the pointer.  If the per-draw fast path skipped the
 * rebind, mesh B would draw with mesh A's data. */
#include "mgl_test.h"

extern void mgl_profile_dump(void);

static const char *vs_src =
    "#version 410 core\n"
    "layout(location=0) in vec3 aPosition;\n"
    "uniform vec4 uColor;\n"
    "void main() { gl_Position = vec4(aPosition, 1.0); }\n";
static const char *fs_src =
    "#version 410 core\n"
    "uniform vec4 uColor; out vec4 fragColor;\n"
    "void main() { fragColor = uColor; }\n";

TEST(offset_respec)
{
    GLuint prog = mgl_make_program(vs_src, fs_src);
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    GLint loc = glGetUniformLocation(prog, "uColor");

    GLuint fbo, color;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glViewport(0, 0, 64, 64);

    /* mesh A (left, red) at offset 0; mesh B (right, green) at offset 512 */
    float meshA[9] = { -0.9f,-0.9f,0,  -0.1f,-0.9f,0,  -0.5f,0.9f,0 };
    float meshB[9] = {  0.1f,-0.9f,0,   0.9f,-0.9f,0,   0.5f,0.9f,0 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 1024, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof meshA, meshA);
    glBufferSubData(GL_ARRAY_BUFFER, 512, sizeof meshB, meshB);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);

    /* warm the pipeline with mesh A */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUniform4f(loc, 1, 0, 0, 1);
    for (int i = 0; i < 30; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glClear(GL_COLOR_BUFFER_BIT);

    /* A: draw with offset 0 */
    glUniform4f(loc, 1, 0, 0, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 16, 255, 0, 0, 255, 4);
    EXPECT_PIXEL(48, 48, 0, 0, 0, 255, 4);

    /* B: same VBO, same stride/type — only the pointer offset changes */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)512);
    glUniform4f(loc, 0, 1, 0, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(48, 16, 0, 255, 0, 255, 4);

    /* and back to A once more (proves the rebind is not one-way) */
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glUniform4f(loc, 1, 0, 0, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 48, 255, 0, 0, 255, 4);

    /* repeat the respec pattern so the profiler buckets average over it:
     * A, respec(B), B — the [ideas] shape (one VBO, moving pointer) */
    for (int f = 0; f < 120; f++)
    {
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)512);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    mgl_profile_dump();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(offset_respec);
    return mgl_test_summary();
}
