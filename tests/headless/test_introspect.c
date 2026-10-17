#include "mgl_test.h"

static GLuint g_vao, g_vbo;
static void setup_quad(void)
{
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    if (!g_vao) {
        glGenVertexArrays(1, &g_vao); glBindVertexArray(g_vao);
        glGenBuffers(1, &g_vbo); glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    }
    glBindVertexArray(g_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

TEST(draw_uniforms_vs_fs)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "uniform float k;\n"
        "uniform vec2 offs;\n"
        "void main(){ gl_Position = vec4(pos * k + offs, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 tint;\n"
        "uniform float scale;\n"
        "void main(){ color = tint * scale; }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    setup_quad();
    glUseProgram(p);
    GLint lk = glGetUniformLocation(p, "k"), lo = glGetUniformLocation(p, "offs");
    GLint lt = glGetUniformLocation(p, "tint"), ls = glGetUniformLocation(p, "scale");
    EXPECT_TRUE(lk >= 0); EXPECT_TRUE(lo >= 0); EXPECT_TRUE(lt >= 0); EXPECT_TRUE(ls >= 0);
    glUniform1f(lk, 1.0f); glUniform2f(lo, 0.0f, 0.0f);
    glUniform4f(lt, 0.0f, 1.0f, 0.5f, 1.0f); glUniform1f(ls, 1.0f);
    EXPECT_NO_GL_ERROR();
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 10, 0, 255, 128, 255, 3);
    /* change uniforms between draws: k=0 collapses the triangle */
    glUniform4f(lt, 1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 3);
    glUniform1f(lk, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(10, 10, 0, 0, 0, 255, 3);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
}

int main(void)
{
    RUN(draw_uniforms_vs_fs);
    return mgl_test_summary();
}
