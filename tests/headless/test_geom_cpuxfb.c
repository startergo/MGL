/* GS transform feedback through the CPU pack (MGL_GEOM_CPU_XFB): several primitives
 * per draw and several draws into one buffer.  Each primitive lands at
 * write_offset + prim*per*stride; the CPU pack used to advance write_offset per
 * primitive as well, which put every primitive after the first at twice its offset
 * (and the next draw after the doubled end).  patch 0004. */
#include "mgl_test.h"

static GLuint mk_prog(const char *vs, const char *gs)
{
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *s; } sh[2] = {
        { GL_VERTEX_SHADER, vs }, { GL_GEOMETRY_SHADER, gs } };
    for (int i = 0; i < 2; i++)
    {
        GLuint h = glCreateShader(sh[i].t);
        GLint ok = 0;
        glShaderSource(h, 1, &sh[i].s, NULL);
        glCompileShader(h);
        glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
        if (ok != 1)
        {
            char log[2048]; GLsizei len = 0;
            glGetShaderInfoLog(h, sizeof log, &len, log);
            fprintf(stderr, "  stage %d compile failed:\n%s\n", i, log);
        }
        glAttachShader(prog, h);
        glDeleteShader(h);
    }
    static const char *names[] = { "val" };
    glTransformFeedbackVaryings(prog, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  LINK failed:\n%s\n", log);
    }
    return prog;
}

static const char *vs_src =
    "#version 460 core\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";

/* one 4-vertex strip per input point: two triangles, captured as v0,v1,v2 | v2,v1,v3 */
static const char *gs_strip_src =
    "#version 460 core\n"
    "layout(points) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n"
    "uniform float base;\n"
    "out float val;\n"
    "void main() {\n"
    "    for (int k = 0; k < 4; k++) {\n"
    "        val = base + float(gl_PrimitiveIDIn) * 100.0 + float(k);\n"
    "        gl_Position = vec4(float(k & 1), float(k >> 1), 0.0, 1.0);\n"
    "        EmitVertex();\n"
    "    }\n"
    "    EndPrimitive();\n"
    "}\n";

/* one point primitive per input point, two vertices each emitted as separate points */
static const char *gs_points_src =
    "#version 460 core\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 2) out;\n"
    "uniform float base;\n"
    "out float val;\n"
    "void main() {\n"
    "    for (int k = 0; k < 2; k++) {\n"
    "        val = base + float(gl_PrimitiveIDIn) * 10.0 + float(k);\n"
    "        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "        EmitVertex();\n"
    "        EndPrimitive();\n"
    "    }\n"
    "}\n";

#define CAP 64

static void run_case(const char *tag, const char *gs, GLenum xfb_mode,
                     const int *draw_points, const float *bases, int ndraws,
                     const float *expect, int nexpect, GLuint expect_prims)
{
    GLuint prog = mk_prog(vs_src, gs);
    GLuint vao, buf, tfb, q;
    float init[CAP];

    for (int i = 0; i < CAP; i++) init[i] = -1.0f;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof init, init, GL_STREAM_READ);
    glGenTransformFeedbacks(1, &tfb);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glGenQueries(1, &q);
    glUseProgram(prog);
    GLint ub = glGetUniformLocation(prog, "base");
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(xfb_mode);
    for (int d = 0; d < ndraws; d++)
    {
        glUniform1f(ub, bases[d]);
        glDrawArrays(GL_POINTS, 0, draw_points[d]);
    }
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLuint prims = 0;
    glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
    mgl_test_checks++;
    if (prims != expect_prims)
        MGL_FAIL("%s: PRIMITIVES_WRITTEN == %u, expected %u", tag, prims, expect_prims);

    float got[CAP];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof got, got);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    for (int i = 0; i < CAP; i++)
    {
        float want = i < nexpect ? expect[i] : -1.0f;   /* nothing lands past the run */
        mgl_test_checks++;
        if (fabsf(got[i] - want) > 0.001f)
            MGL_FAIL("%s: word %d == %f, expected %f", tag, i, got[i], want);
    }

    glDeleteTransformFeedbacks(1, &tfb);
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &buf);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

/* 2 points then 1 point: 4 + 2 triangles, back to back */
TEST(cpu_xfb_strip_multi_prim_multi_draw)
{
    static const int pts[2] = { 2, 1 };
    static const float bases[2] = { 0.0f, 1000.0f };
    static const float expect[18] = {
        0, 1, 2,   2, 1, 3,
        100, 101, 102,   102, 101, 103,
        1000, 1001, 1002,   1002, 1001, 1003,
    };
    run_case("strip", gs_strip_src, GL_TRIANGLES, pts, bases, 2, expect, 18, 6);
}

/* 3 points then 2 points: 6 + 4 point primitives, back to back */
TEST(cpu_xfb_points_multi_prim_multi_draw)
{
    static const int pts[2] = { 3, 2 };
    static const float bases[2] = { 0.0f, 500.0f };
    static const float expect[10] = {
        0, 1, 10, 11, 20, 21,
        500, 501, 510, 511,
    };
    run_case("points", gs_points_src, GL_POINTS, pts, bases, 2, expect, 10, 10);
}

int main(void)
{
    /* force the CPU pack (read once, on the first geometry draw) */
    setenv("MGL_GEOM_CPU_XFB", "1", 1);
    RUN(cpu_xfb_strip_multi_prim_multi_draw);
    RUN(cpu_xfb_points_multi_prim_multi_draw);
    return mgl_test_summary();
}
