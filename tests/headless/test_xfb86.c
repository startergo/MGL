/* test_xfb86.c — geometry-shader transform feedback capture round r86:
 *  1. gl_Position captured from a GS: GL records the strip DECOMPOSED into its
 *     triangles, the odd ones transposed (the GL strip-decomposition rule; the reference is
 *     KHR-GL46.transform_feedback.geometry's own reference_data).
 *  2. streams + gl_SkipComponents4 / gl_NextBuffer markers (KHR-GL46.transform_feedback3.
 *     multiple_streams): stream 0 and stream 1 capture into two binding points, skip
 *     markers pad the records, and a marker must never overwrite a captured varying.
 */
#include "mgl_test.h"

static const char *kVS = "#version 400\n"
    "in vec4 vertex;\n"
    "void main (void) { gl_Position = vertex; }\n";
static const char *kFS = "#version 400\n"
    "in vec4 color;\n"
    "out vec4 outColor;\n"
    "void main (void) { outColor = color; }\n";

static GLuint link_xfb(const char *gs, const char *const *vars, int nvars, GLenum mode)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint g = glCreateShader(GL_GEOMETRY_SHADER), p = glCreateProgram();
    GLint ok = 0;

    glShaderSource(vs, 1, &kVS, NULL); glCompileShader(vs);
    glShaderSource(fs, 1, &kFS, NULL); glCompileShader(fs);
    glShaderSource(g, 1, &gs, NULL); glCompileShader(g);
    glAttachShader(p, vs); glAttachShader(p, g); glAttachShader(p, fs);
    glDeleteShader(vs); glDeleteShader(fs); glDeleteShader(g);
    glTransformFeedbackVaryings(p, nvars, vars, mode);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    glUseProgram(p);
    return p;
}

static GLuint setup_points(GLuint prog, const float *data, int n)
{
    GLuint vbo, vao;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(n * 16), data, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    {
        GLint loc = glGetAttribLocation(prog, "vertex");
        EXPECT_TRUE(loc >= 0);
        glEnableVertexAttribArray((GLuint)loc);
        glVertexAttribPointer((GLuint)loc, 4, GL_FLOAT, GL_FALSE, 0, 0);
    }
    return vbo;
}

TEST(xfb_geom_position)
{
    static const char *gs = "#version 400\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 5) out;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "    gl_Position = gl_in[0].gl_Position; color = vec4(1.0); EmitVertex();\n"
        "    gl_Position = gl_in[1].gl_Position; color = vec4(1.0); EmitVertex();\n"
        "    gl_Position = (gl_in[0].gl_Position+gl_in[1].gl_Position+gl_in[2].gl_Position)/3;\n"
        "    color = vec4(1.0); EmitVertex();\n"
        "    gl_Position = gl_in[2].gl_Position; color = vec4(1.0); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const float in_data[12] = {
        1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 2.0f, 1.0f, 1.0f, -1.0f, 3.0f, 1.0f,
    };
    /* the 4-vertex strip is captured as its two triangles, the second transposed:
     * (v0, v1, C) then (C, v1, v2) */
    static const float ref[24] = {
        1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 2.0f, 1.0f,
        1.0f / 3.0f, 1.0f / 3.0f, 2.0f, 1.0f,
        1.0f / 3.0f, 1.0f / 3.0f, 2.0f, 1.0f, -1.0f, 1.0f, 2.0f, 1.0f,
        1.0f, -1.0f, 3.0f, 1.0f,
    };
    const char *var = "gl_Position";
    float out[24];
    GLuint p = link_xfb(gs, &var, 1, GL_SEPARATE_ATTRIBS);
    GLuint vbo = setup_points(p, in_data, 3), xfb, q[2] = { 0, 0 };

    glGenBuffers(1, &xfb);
    glBindBuffer(GL_ARRAY_BUFFER, xfb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(out), NULL, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);

    glGenQueries(2, q);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q[0]);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q[1]);
    glBeginTransformFeedback(GL_TRIANGLES);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    EXPECT_NO_GL_ERROR();
    {
        GLint gen = 0, written = 0;
        glGetQueryObjectiv(q[0], GL_QUERY_RESULT, &gen);
        glGetQueryObjectiv(q[1], GL_QUERY_RESULT, &written);
        EXPECT_EQ_INT(gen, 2);      /* one strip = two decomposed triangles */
        EXPECT_EQ_INT(written, 2);
    }

    memset(out, 0x7f, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(out), out);
    for (int i = 0; i < 24; i++)
        EXPECT_NEAR(out[i], ref[i], 1e-4);

    glDeleteQueries(2, q);
    glDeleteBuffers(1, &xfb);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(p);
}

TEST(xfb_streams_markers)
{
    static const char *gs = "#version 400\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 8) out;\n"
        "layout(stream=0) out vec4 pos0;\n"
        "layout(stream=1) out vec4 pos1;\n"
        "void main() {\n"
        "    pos0 = vec4(0.1, 0.2, 0.3, 0.4) * gl_in[0].gl_Position;\n"
        "    EmitStreamVertex(0);\n"
        "    EndStreamPrimitive(0);\n"
        "    pos1 = vec4(-0.1, -0.2, -0.3, -0.4) * gl_in[0].gl_Position;\n"
        "    EmitStreamVertex(1);\n"
        "    EndStreamPrimitive(1);\n"
        "}\n";
    static const float in_data[12] = {
        -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -2.0f, 1.0f, -1.0f, 1.0f, -3.0f, 1.0f,
    };
    static const char *vars[5] = {
        "pos0", "gl_SkipComponents4", "gl_NextBuffer", "gl_SkipComponents4", "pos1",
    };
    /* buffer 0's record = [pos0][skip 4], buffer 1's = [skip 4][pos1] */
    static const float ref0[6][4] = {
        { -0.1f, -0.2f, -0.3f, 0.4f }, { 0, 0, 0, 0 },
        { 0.1f, -0.2f, -0.6f, 0.4f }, { 0, 0, 0, 0 },
        { -0.1f, 0.2f, -0.9f, 0.4f }, { 0, 0, 0, 0 },
    };
    static const float ref1[6][4] = {
        { 0, 0, 0, 0 }, { 0.1f, 0.2f, 0.3f, -0.4f },
        { 0, 0, 0, 0 }, { -0.1f, 0.2f, 0.6f, -0.4f },
        { 0, 0, 0, 0 }, { 0.1f, -0.2f, 0.9f, -0.4f },
    };
    GLuint p = link_xfb(gs, vars, 5, GL_INTERLEAVED_ATTRIBS);
    GLuint vbo = setup_points(p, in_data, 3), xfb[2], q = 0;
    float b0[24], b1[24];

    glGenBuffers(2, xfb);
    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_ARRAY_BUFFER, xfb[i]);
        glBufferData(GL_ARRAY_BUFFER, 96, NULL, GL_STATIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb[0]);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, xfb[1]);

    glGenQueries(1, &q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_POINTS, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    EXPECT_NO_GL_ERROR();
    {
        GLint written = 0;
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
        /* glBeginQuery == glBeginQueryIndexed(index 0) and this target's index IS the
         * vertex stream (the GL stream-query rule), so the unindexed query counts stream 0 only:
         * 3 points.  Stream 1's 3 points are visible in buffer 1 below (and via
         * glBeginQueryIndexed(..., 1)). */
        EXPECT_EQ_INT(written, 3);
    }

    memset(b0, 0x7f, sizeof b0);
    memset(b1, 0x7f, sizeof b1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[0]);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(b0), b0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb[1]);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(b1), b1);
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 4; c++)
        {
            EXPECT_NEAR(b0[r * 4 + c], ref0[r][c], 1e-5);
            EXPECT_NEAR(b1[r * 4 + c], ref1[r][c], 1e-5);
        }

    glDeleteQueries(1, &q);
    glDeleteBuffers(2, xfb);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(xfb_geom_position);
    RUN(xfb_streams_markers);
    return mgl_test_summary();
}
