#include "mgl_test.h"

TEST(version_strings)
{
    EXPECT_TRUE(glGetString(GL_VERSION) != NULL);
    EXPECT_TRUE(glGetString(GL_SHADING_LANGUAGE_VERSION) != NULL);
    EXPECT_NO_GL_ERROR();
}

TEST(clear_color)
{
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_PIXEL(0, 0, 255, 0, 0, 255, 2);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_PIXEL(0, 0, 0, 0, 255, 255, 2);
    EXPECT_NO_GL_ERROR();
}

TEST(draw_triangle)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint prog = mgl_make_program(vs, fs);
    if (!prog) return;
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(10, 10, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(version_strings);
    RUN(clear_color);
    RUN(draw_triangle);
    return mgl_test_summary();
}
