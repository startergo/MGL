/* test_gsindirect2.c — CTS-exact geometry indirect setup: one shared STATIC
 * buffer (NULL + SubData) bound to ARRAY/ELEMENT/DRAW_INDIRECT, commands at
 * byte offsets, basevertex/baseinstance = 1 */
#include "mgl_test.h"

static GLuint compile1(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok; char log[4096];
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile: %s\n", log); }
    EXPECT_TRUE(ok);
    return sh;
}

static GLuint mgl_make_program_gs(const char *vs, const char *gs, const char *fs)
{
    GLuint p = glCreateProgram(), s1 = compile1(GL_VERTEX_SHADER, vs),
           s2 = compile1(GL_GEOMETRY_SHADER, gs), s3 = compile1(GL_FRAGMENT_SHADER, fs);
    GLint ok; char log[4096];
    glAttachShader(p, s1); glAttachShader(p, s2); glAttachShader(p, s3);
    glLinkProgram(p);
    ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "link: %s\n", log); }
    EXPECT_TRUE(ok);
    return p;
}

static const char *k_vs =
    "#version 330\n"
    "layout(location=0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_gs =
    "#version 150\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 9) out;\n"
    "void main() {\n"
    "    for (int p = 0; p < 3; p++) {\n"
    "        for (int i = 0; i < 3; i++) {\n"
    "            gl_Position = vec4(gl_in[i].gl_Position.xy * (0.5 - p * 0.1), 0.0, 1.0);\n"
    "            EmitVertex();\n"
    "        }\n"
    "        EndPrimitive();\n"
    "    }\n"
    "}\n";
static const char *k_fs =
    "#version 330\n"
    "out vec4 o;\n"
    "void main() { o = vec4(1.0, 0.0, 0.0, 1.0); }\n";

static const float k_tri[9] = { -1,-1, 1,-1, 0,1 };

static void check(GLuint q, GLuint expected_min, GLuint expected_max, const char *what)
{
    GLuint64 v = 0;
    glFinish();
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &v);
    mgl_test_checks++;
    if (!(v >= expected_min && v <= expected_max))
        MGL_FAIL("%s: query=%llu expected [%u..%u]", what, (unsigned long long)v, expected_min, expected_max);
}

TEST(geometry_stats_cts_shape)
{
    struct { GLuint count, instances, first, baseInstance; } acmd;
    struct { GLuint count, instances, firstidx, basevertex, baseinstance; } ecmd;
    GLuint prog = 0, vao = 0, bo = 0, q_emit = 0, q_inv = 0;
    GLuint idx[3] = { 0, 1, 2 };
    size_t vert_off = 0, idx_off = 256, acmd_off = 512, ecmd_off = 544;
    size_t bo_size = 640;
    void *base;

    prog = mgl_make_program_gs(k_vs, k_gs, k_fs);
    EXPECT_TRUE(prog);
    if (!prog) return;

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);

    /* ONE static buffer for everything, filled with SubData (the CTS pattern) */
    glGenBuffers(1, &bo);
    glBindBuffer(GL_ARRAY_BUFFER, bo);
    glBufferData(GL_ARRAY_BUFFER, bo_size, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, vert_off, sizeof k_tri, k_tri);
    glBufferSubData(GL_ARRAY_BUFFER, idx_off, sizeof idx, idx);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bo);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)vert_off);

    glGenQueries(1, &q_emit);
    glGenQueries(1, &q_inv);
    glUseProgram(prog);
    glViewport(0, 0, 8, 8);

    /* count = 3 (1 triangle input), instanceCount = 4, basevertex/baseinstance = 1 */
    ecmd.count = 3; ecmd.instances = 4; ecmd.firstidx = (GLuint)idx_off; ecmd.basevertex = 1; ecmd.baseinstance = 1;
    glBufferSubData(GL_ARRAY_BUFFER, ecmd_off, sizeof ecmd, &ecmd);
    while (glGetError() != GL_NO_ERROR) {}

    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q_emit);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q_inv);
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, (void *)ecmd_off);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    { GLenum e = glGetError(); if (e) MGL_FAIL("drawElementsIndirect error 0x%x", e); }

    check(q_inv, 4, 4, "cts elements-indirect invocations");
    check(q_emit, 12, 12, "cts elements-indirect emitted");

    /* the same via arrays command (reference: works) */
    acmd.count = 3; acmd.instances = 4; acmd.first = 0; acmd.baseInstance = 1;
    glBufferSubData(GL_ARRAY_BUFFER, acmd_off, sizeof acmd, &acmd);
    while (glGetError() != GL_NO_ERROR) {}
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q_emit);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q_inv);
    glDrawArraysIndirect(GL_TRIANGLES, (void *)acmd_off);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    { GLenum e = glGetError(); if (e) MGL_FAIL("drawArraysIndirect error 0x%x", e); }

    check(q_inv, 4, 4, "cts arrays-indirect invocations");
    check(q_emit, 12, 12, "cts arrays-indirect emitted");

    (void)base;
}

int main(void)
{
    RUN(geometry_stats_cts_shape);
    return mgl_test_summary();
}
