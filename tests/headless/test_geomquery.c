/* queries around batches of geometry draws: GL_TIME_ELAPSED / glQueryCounter(GL_TIMESTAMP)
 * resolves must not hang when no glFinish() separates the draws from the readback.
 * Regression test for the extBufWaitSerial hang (waiting on a command buffer that was
 * never committed — a query split can leave the tracked/last buffer open). */
#include "mgl_test.h"

#define VERTS    (30 * 1000)
#define IDX_CNT  (90 * 1000)
#define FRAMES   20

static GLuint prog;

static void setup(void)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ v_pos = pos; gl_Position = vec4(pos * 0.02, 0.0, 1.0); }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) out vec2 g_pos;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_pos = v_pos[i]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 g_pos;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(g_pos * 0.5 + 0.5, 0.0, 1.0); }\n";
    GLuint s, vao, vbo, ebo, q = prog;
    float *verts = malloc(sizeof(float) * 2 * VERTS);
    GLuint *idx = malloc(sizeof(GLuint) * IDX_CNT);

    for (int i = 0; i < VERTS; i++)
    {
        verts[i * 2 + 0] = (float)(i % 977);
        verts[i * 2 + 1] = (float)((i * 31) % 977);
    }
    for (int i = 0; i < IDX_CNT; i++)
        idx[i] = (GLuint)(((GLuint)i * 7919u) % VERTS);   /* scattered */

    q = glCreateProgram();
    s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &vs, NULL); glCompileShader(s); glAttachShader(q, s);
    s = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(s, 1, &gs, NULL); glCompileShader(s); glAttachShader(q, s);
    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 1, &fs, NULL); glCompileShader(s); glAttachShader(q, s);
    glLinkProgram(q);
    {
        GLint ok = 0;
        glGetProgramiv(q, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
    }
    prog = q;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * VERTS, verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * IDX_CNT, idx, GL_STATIC_DRAW);
    glUseProgram(prog);
    free(verts);
    free(idx);
}

TEST(elapsed_no_finish)
{
    GLuint q = 0;
    GLuint64 ns = 0;

    glGenQueries(1, &q);
    glBeginQuery(GL_TIME_ELAPSED, q);
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, IDX_CNT, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TIME_ELAPSED);
    /* no glFinish: the driver must make the result available itself */
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
    printf("  TIME_ELAPSED over %d GS draws: %.1f us/frame\n", FRAMES, (double)ns / 1000.0 / FRAMES);
    EXPECT_TRUE(ns > 0);
    glDeleteQueries(1, &q);
    EXPECT_NO_GL_ERROR();
}

TEST(elapsed_then_more_draws)
{
    /* resolve while later draws (new command buffers) are already encoded */
    GLuint q = 0;
    GLuint64 ns = 0;

    glGenQueries(1, &q);
    glBeginQuery(GL_TIME_ELAPSED, q);
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, IDX_CNT, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TIME_ELAPSED);
    for (int f = 0; f < 4; f++)
        glDrawElements(GL_TRIANGLES, IDX_CNT, GL_UNSIGNED_INT, NULL);
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
    printf("  TIME_ELAPSED resolved after 4 more draws: %.1f us/frame\n", (double)ns / 1000.0 / FRAMES);
    EXPECT_TRUE(ns > 0);
    glDeleteQueries(1, &q);
    EXPECT_NO_GL_ERROR();
}

TEST(timestamp_counters)
{
    GLuint t0 = 0, t1 = 0;
    GLuint64 a = 0, b = 0;

    glGenQueries(1, &t0);
    glGenQueries(1, &t1);
    glQueryCounter(t0, GL_TIMESTAMP);
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, IDX_CNT, GL_UNSIGNED_INT, NULL);
    glQueryCounter(t1, GL_TIMESTAMP);
    glGetQueryObjectui64v(t0, GL_QUERY_RESULT, &a);
    glGetQueryObjectui64v(t1, GL_QUERY_RESULT, &b);
    printf("  TIMESTAMP pair: t0=%llu t1=%llu delta=%.1f us/frame\n",
           (unsigned long long)a, (unsigned long long)b, (double)(b - a) / 1000.0 / FRAMES);
    EXPECT_TRUE(b >= a);
    glDeleteQueries(1, &t0);
    glDeleteQueries(1, &t1);
    EXPECT_NO_GL_ERROR();
}

TEST(available_poll)
{
    /* GL_QUERY_RESULT_AVAILABLE must eventually flip to GL_TRUE without a finish */
    GLuint q = 0;
    GLint avail = 0;
    GLuint64 ns = 0;
    int polls = 0;

    glGenQueries(1, &q);
    glBeginQuery(GL_TIME_ELAPSED, q);
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, IDX_CNT, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TIME_ELAPSED);
    do {
        glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE, &avail);
    } while (!avail && ++polls < 2000000);
    EXPECT_TRUE(avail);
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &ns);
    EXPECT_TRUE(ns > 0);
    printf("  availability after %d polls, result %.1f us/frame\n", polls, (double)ns / 1000.0 / FRAMES);
    glDeleteQueries(1, &q);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    setup();
    RUN(elapsed_no_finish);
    RUN(elapsed_then_more_draws);
    RUN(timestamp_counters);
    RUN(available_poll);
    return mgl_test_summary();
}
