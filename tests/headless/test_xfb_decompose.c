/* r76: transform feedback of strips/fans/loops records the DECOMPOSED
 * independent primitives (GL 4.6 13.3.2.2: with primitiveMode TRIANGLES/LINES
 * the strip/fan/loop is captured as independent triangles/lines, in the
 * 10.1 decomposition order INCLUDING the odd-triangle winding flip).
 *
 * compat r108 claimed "4-vertex strip -> 6 records instead of per-vertex" is
 * a bug; per spec it is CORRECT.  This test asserts the count AND the vertex
 * order for every mode, plus the adjacency-mode error (no GS + xfb active ->
 * INVALID_OPERATION, 13.3.1.1) and a geometry-shader adjacency capture. */
#include "mgl_test.h"
#include <string.h>

static const char *kVs =
    "#version 460 core\n"
    "layout(location=0) in float a_id;\n"
    "out float v_id;\n"
    "void main(){\n"
    "  float x = (a_id / 6.0) * 2.0 - 1.0;\n"
    "  gl_Position = vec4(x, x * 0.5, 0.0, 1.0);\n"
    "  v_id = a_id;\n"
    "}\n";
static const char *kVsAdj =
    "#version 460 core\n"
    "layout(location=0) in float a_id;\n"
    "out float v_id;\n"
    "void main(){ gl_Position = vec4(0.0, 0.0, 0.0, 1.0); v_id = a_id; }\n";
static const char *kGsAdj =
    "#version 460 core\n"
    "layout(triangles_adjacency) in;\n"
    "layout(line_strip, max_vertices=4) out;\n"
    "in float v_id[];\n"
    "out float g_id;\n"
    "void main(){\n"
    "  for (int i = 0; i < 4; i++) { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); g_id = v_id[i]; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 460 core\n"
    "layout(location=0) out vec4 o;\n"
    "void main(){ o = vec4(0,1,0,1); }\n";

static int capture_ids(GLenum draw_mode, GLenum capture_mode, const char *gs,
                       int count, float *got, int maxgot)
{
    const GLenum with_gs[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const GLenum no_gs[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    const GLenum *types = gs ? with_gs : no_gs;
    const char *srcs[3] = { kVs, gs, kFs };
    int nsrc = gs ? 3 : 2;    GLuint p, vao, b, xb;
    GLint link = 0;
    int n = 0;

    for (int i = 0; i < maxgot; i++) got[i] = -1.0f;
    while (glGetError() != GL_NO_ERROR) {}
    p = glCreateProgram();
    for (int i = 0; i < nsrc; i++)
    {
        GLuint so = glCreateShader(types[i]);
        const char *src = srcs[i] ? srcs[i] : kFs;
        glShaderSource(so, 1, &src, NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    {
        const char *v = gs ? "g_id" : "v_id";
        glTransformFeedbackVaryings(p, 1, &v, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE)
    {
        char lg[2048] = {0}; GLsizei nn = 0;
        glGetProgramInfoLog(p, sizeof lg - 1, &nn, lg);
        printf("  LINK FAIL: %.200s\n", lg);
        glDeleteProgram(p);
        return -1;
    }
    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    {
        float ids[16];
        for (int i = 0; i < count && i < 16; i++) ids[i] = (float)(i + 1);
        glBufferData(GL_ARRAY_BUFFER, count * 4, ids, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, NULL);
        glEnableVertexAttribArray(0);
    }
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, maxgot * 4 + 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBeginTransformFeedback(capture_mode);
    glDrawArrays(draw_mode, 0, count);
    glEndTransformFeedback();
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    glBindBuffer(GL_COPY_READ_BUFFER, xb);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, maxgot * 4, got);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glFinish();
    while (n < maxgot && got[n] > 0.0f && got[n] < 16.0f) n++;
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    return n;
}

static void check_mode(const char *name, GLenum mode, GLenum capture,
                       const float *want, int nwant)
{
    float got[64];
    int n = capture_ids(mode, capture, NULL, 6, got, 64);
    printf("  %-15s n=%2d got:", name, n);
    for (int i = 0; i < n && i < 16; i++) printf(" %.0f", got[i]);
    printf("\n");
    EXPECT_EQ_INT(n, nwant);
    if (n == nwant)
        for (int i = 0; i < nwant; i++)
            EXPECT_NEAR(got[i], want[i], 1e-5);
}

TEST(XfbStripDecomposition)
{
    glViewport(0, 0, 64, 64);

    {
        const float w[] = { 1,2,3, 4,5,6 };
        check_mode("TRIANGLES", GL_TRIANGLES, GL_TRIANGLES, w, 6);
    }
    {
        /* the odd triangles flip winding: (2,1,3), (4,3,5) */
        const float w[] = { 1,2,3, 3,2,4, 3,4,5, 5,4,6 };
        check_mode("TRIANGLE_STRIP", GL_TRIANGLE_STRIP, GL_TRIANGLES, w, 12);
    }
    {
        const float w[] = { 1,2,3, 1,3,4, 1,4,5, 1,5,6 };
        check_mode("TRIANGLE_FAN", GL_TRIANGLE_FAN, GL_TRIANGLES, w, 12);
    }
    {
        const float w[] = { 1,2, 3,4, 5,6 };
        check_mode("LINES", GL_LINES, GL_LINES, w, 6);
    }
    {
        const float w[] = { 1,2, 2,3, 3,4, 4,5, 5,6 };
        check_mode("LINE_STRIP", GL_LINE_STRIP, GL_LINES, w, 10);
    }
    {
        const float w[] = { 1,2, 2,3, 3,4, 4,5, 5,6, 6,1 };
        check_mode("LINE_LOOP", GL_LINE_LOOP, GL_LINES, w, 12);
    }

    /* adjacency without a GS while xfb is active: INVALID_OPERATION (13.3.1.1) */
    while (glGetError() != GL_NO_ERROR) {}
    {
        GLuint p = glCreateProgram(), vao, b, xb;
        float got[8];
        GLint link = 0;
        const GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
        const char *srcs[2] = { kVsAdj, kFs };
        for (int i = 0; i < 2; i++)
        {
            GLuint so = glCreateShader(types[i]);
            glShaderSource(so, 1, &srcs[i], NULL);
            glCompileShader(so);
            glAttachShader(p, so);
            glDeleteShader(so);
        }
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &link);
        EXPECT_EQ_INT(link, GL_TRUE);
        if (!link) return;
        glUseProgram(p);
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &b);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        {
            float ids[6] = { 1, 2, 3, 4, 5, 6 };
            glBufferData(GL_ARRAY_BUFFER, sizeof ids, ids, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, NULL);
            glEnableVertexAttribArray(0);
        }
        glGenBuffers(1, &xb);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof got, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
        while (glGetError() != GL_NO_ERROR) {}
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES_ADJACENCY, 0, 6);
        EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
        glEndTransformFeedback();
        while (glGetError() != GL_NO_ERROR) {}
        glDeleteBuffers(1, &xb);
        glDeleteProgram(p);
    }

    /* with a GS consuming triangles_adjacency: the GS's OUTPUT primitives are
     * captured (4 line vertices per input adjacency primitive) */
    {
        float got[16];
        int n = capture_ids(GL_TRIANGLES_ADJACENCY, GL_LINES, kGsAdj, 12, got, 16);
        /* the GS emits a 4-vertex line_strip per adjacency primitive; captured
         * with primitiveMode GL_LINES the strip decomposes to 3 line pairs
         * each: (1,2)(2,3)(3,4) then (7,8)(8,9)(9,10) */
        const float w[] = { 1,2, 2,3, 3,4, 7,8, 8,9, 9,10 };
        printf("  GS+ADJACENCY     n=%2d got:", n);
        for (int i = 0; i < n && i < 12; i++) printf(" %.0f", got[i]);
        printf("\n");
        EXPECT_EQ_INT(n, 12);
        for (int i = 0; i < 12; i++)
            EXPECT_NEAR(got[i], w[i], 1e-5);
    }
}

int main(void) { RUN(XfbStripDecomposition); return mgl_test_summary(); }
