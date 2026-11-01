/* test_prst86.c — primitive restart with the adjacency list modes: the restart index
 * ends the group and assembly resumes at the NEXT index (a full 4/6-index group is
 * still required), and the drawn primitive uses the group's REAL vertices (i1,i2 for
 * lines, i1,i3,i5 for triangles) - not the adjacency ones.  Geometry and spot values
 * mirror KHR-GL46.primitive_restart.restart_mode. */
#include "mgl_test.h"
#include <stdbool.h>

static const char *VS = "#version 430\n"
    "layout(location=0) in vec3 position;\n"
    "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }\n";
static const char *FS = "#version 430\n"
    "out vec4 o;\n"
    "void main() { o = vec4(1.0); }\n";

/* the CTS vertices: quad, mid point, point spots */
static const float verts[6][3] = {
    { 1.0f, -1.0f, -2.0f }, { 1.0f, 1.0f, -2.0f }, { -1.0f, -1.0f, -2.0f },
    { -1.0f, 1.0f, -2.0f }, { 0.0f, 0.0f, -2.0f }, { 0.5f, -0.5f, -2.0f },
};
static const GLuint idxLA[9] = { 4, 4, 3, 3, 0xFFFFu, 4, 4, 0, 0 };
static const GLuint idxTA[13] = { 0, 0, 1, 1, 2, 2, 0xFFFFu, 2, 2, 1, 1, 3, 3 };

static int spots(GLenum mode, const GLuint *idx, int n)
{
    GLuint prog, vbo, ibo, vao;
    int hit = 0;
    static const float sp[3][2] = { { 0.5f, 0.5f }, { 0.25f, 0.75f }, { 0.75f, 0.25f } };

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    prog = glCreateProgram();
    {
        GLuint s = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
        GLint ok = 0;
        glShaderSource(s, 1, &VS, NULL); glCompileShader(s);
        glShaderSource(f, 1, &FS, NULL); glCompileShader(f);
        glAttachShader(prog, s); glAttachShader(prog, f);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        glDeleteShader(s); glDeleteShader(f);
    }
    glUseProgram(prog);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(n * 4), idx, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    {
        GLint loc = glGetAttribLocation(prog, "position");
        glEnableVertexAttribArray((GLuint)loc);
        glVertexAttribPointer((GLuint)loc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFu);
    EXPECT_NO_GL_ERROR();
    glDrawElements(mode, n, GL_UNSIGNED_INT, 0);
    EXPECT_NO_GL_ERROR();
    for (int i = 0; i < 3; i++)
    {
        float px[4];
        int x = (int)(0.5f + sp[i][0] * 63), y = (int)(0.5f + sp[i][1] * 63);
        int ok = 0;
        /* the CTS testSpotLine tolerates +-1 pixel */
        int cand[5][2] = { { x, y }, { x - 1, y }, { x + 1, y }, { x, y - 1 }, { x, y + 1 } };
        for (int c = 0; c < 5 && !ok; c++)
        {
            if (cand[c][0] < 0 || cand[c][0] > 63 || cand[c][1] < 0 || cand[c][1] > 63) continue;
            glReadPixels(cand[c][0], cand[c][1], 1, 1, GL_RGBA, GL_FLOAT, px);
            if (px[0] > 0.9f) ok = 1;
        }
        if (ok) hit++;
        printf("      spot %d (%g,%g): %s\n", i, sp[i][0], sp[i][1], ok ? "hit" : "MISS");
    }
    glDisable(GL_PRIMITIVE_RESTART);
    glDeleteProgram(prog); glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &ibo);
    glDeleteVertexArrays(1, &vao);
    return hit;
}

TEST(restart_adj_exact)
{
    int h;
    glViewport(0, 0, 64, 64);
    h = spots(GL_LINES_ADJACENCY, idxLA, 9);
    printf("   LINES_ADJACENCY: %d/3 spots\n", h);
    EXPECT_EQ_INT(h, 3);
    h = spots(GL_LINE_STRIP_ADJACENCY, idxLA, 9);
    printf("   LINE_STRIP_ADJACENCY: %d/3 spots\n", h);
    EXPECT_EQ_INT(h, 3);
    h = spots(GL_TRIANGLES_ADJACENCY, idxTA, 13);
    printf("   TRIANGLES_ADJACENCY: %d/3 spots\n", h);
    EXPECT_EQ_INT(h, 3);
    h = spots(GL_TRIANGLE_STRIP_ADJACENCY, idxTA, 13);
    printf("   TRIANGLE_STRIP_ADJACENCY: %d/3 spots\n", h);
    EXPECT_EQ_INT(h, 3);
}

int main(void)
{
    RUN(restart_adj_exact);
    return mgl_test_summary();
}
