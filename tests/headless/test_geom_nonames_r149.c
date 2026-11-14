/* r149 Task A: user varyings named with a _mgl_ PREFIX report their REAL names
 * through a GS declared capture (the exact-list rule replaced the prefix rule) */
#include "mgl_test.h"
#include <string.h>
#include <stdio.h>

static const char *gs_prefixed =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "layout(xfb_buffer = 0, xfb_stride = 20) out Blk {\n"
    "   layout(xfb_offset = 0) vec4 _mgl_out_user;\n"
    "   layout(xfb_offset = 16) float _mgl_t_user;\n"
    "};\n"
    "void main() {\n"
    "  gl_Position = gl_in[0].gl_Position;\n"
    "  _mgl_out_user = vec4(1.0, 0.5, 0.25, 0.125);\n"
    "  _mgl_t_user = 0.9;\n"
    "  EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";

TEST(r149_prefixed_names_report)
{
    const char *vs = "#version 450 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
    GLuint p = mgl_make_program5(vs, NULL, NULL, gs_prefixed,
        "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1);}\n");
    GLint count = -1, maxlen = -1;
    char nm[128];
    GLsizei len = -1;
    float out[8] = {0};

    if (!p) return;
    glUseProgram(p);
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 2048, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
        glDeleteBuffers(1, &bo);
    }
    EXPECT_NEAR(out[0], 1.0, 1e-5);          /* the capture itself is unchanged */

    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
    EXPECT_EQ_INT(count, 2);
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH, &maxlen);
    EXPECT_EQ_INT(maxlen, (int)strlen("_mgl_out_user") + 1);   /* was 1: both blanked */
    nm[0] = '?';
    glGetTransformFeedbackVarying(p, 0, sizeof nm, &len, NULL, NULL, nm);
    EXPECT_EQ_INT(len, (GLsizei)strlen("_mgl_out_user"));
    EXPECT_EQ_INT(strcmp(nm, "_mgl_out_user"), 0);
    nm[0] = '?';
    glGetTransformFeedbackVarying(p, 1, sizeof nm, &len, NULL, NULL, nm);
    EXPECT_EQ_INT(strcmp(nm, "_mgl_t_user"), 0);
    /* and the fence's own name-less module still blanks (the r147/r148 rule
     * stays: exact list, not prefix) */
    glDeleteProgram(p);
}

/* r149 Task B: deleting a program must not leave its xfb table entry behind for
 * the next program with the same (recycled) GL name */
static GLuint decl_capture_program(int variant)
{
    char gs[1024];
    GLuint p;

    snprintf(gs, sizeof gs,
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "layout(xfb_buffer = 0) out V {\n"
        "   layout(xfb_offset = 0) vec4 v0;\n"
        "%s"
        "};\n"
        "void main() {\n"
        "  gl_Position = gl_in[0].gl_Position;\n"
        "  v0 = vec4(1.0);\n"
        "  EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n",
        variant ? "   layout(xfb_offset = 16) vec4 v1;\n   layout(xfb_offset = 32) vec4 v2;\n" : "");

    p = mgl_make_program5("#version 450 core\nvoid main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n",
                          NULL, NULL, gs,
                          "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1);}\n");
    if (p) glUseProgram(p);
    return p;
}

TEST(r149_delete_program_frees_slot)
{
    GLint count = -1;
    GLuint a, name, b;

    /* program A: 3 captured varyings, then delete */
    a = decl_capture_program(1);
    EXPECT_TRUE(a);
    if (!a) return;
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 2048, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();
        glDeleteBuffers(1, &bo);
    }
    name = a;
    glDeleteProgram(a);

    /* program B reuses A's GL name with ONE captured varying */
    b = decl_capture_program(0);
    EXPECT_TRUE(b);
    if (!b) return;
    /* MGL hands out the lowest free GL name, so b == name when recycling
     * happened; if it ever differs, retry the fence on a new pair so it never
     * passes vacuously */
    if (b != name)
    {
        glDeleteProgram(b);
        b = decl_capture_program(0);
        if (!b) return;
    }
    {
        GLuint bo = 0;

        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 2048, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();
        glDeleteBuffers(1, &bo);
    }
    glGetProgramiv(b, GL_TRANSFORM_FEEDBACK_VARYINGS, &count);
    EXPECT_EQ_INT(count, 1);              /* was 3: A's stale slot surfaced */
    /* and B's capture matches its own single varying */
    {
        GLuint q = 0, bo = 0;
        GLint written = -1;
        float out[4] = {0};

        glGenQueries(1, &q);
        glGenBuffers(1, &bo);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 2048, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, bo);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
        EXPECT_EQ_INT(written, 1);
        EXPECT_NEAR(out[0], 1.0, 1e-5);
        glDeleteQueries(1, &q);
        glDeleteBuffers(1, &bo);
    }
    glDeleteProgram(b);
}

int main(void)
{
    GLuint vao = 0;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    RUN(r149_prefixed_names_report);
    RUN(r149_delete_program_frees_slot);
    return mgl_test_summary();
}
