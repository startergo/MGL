/* test_gsindirect.c — geometry-stage pipeline statistics for indirect draws
 * (CTS functional_geometry_shader_queries: GEOMETRY_SHADER_PRIMITIVES_EMITTED
 * came out as one instance's worth for glDrawElementsIndirect). */
#include "mgl_test.h"

static const char *k_vs =
    "#version 330\n"
    "layout(location=0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_gs =
    "#version 330\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 12) out;\n"
    "void main() {\n"
    "    /* the CTS GS emits several primitives per invocation */\n"
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

static GLuint compile_one(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok = 0;
    char log[4096];
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile: %s\n", log); }
    EXPECT_TRUE(ok);
    return sh;
}

static GLuint make_prog(void)
{
    GLuint p, vs, gs, fs;
    GLint ok = 0;
    char log[4096];
    vs = compile_one(GL_VERTEX_SHADER, k_vs);
    gs = compile_one(GL_GEOMETRY_SHADER, k_gs);
    fs = compile_one(GL_FRAGMENT_SHADER, k_fs);
    EXPECT_TRUE(vs && gs && fs);
    p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, gs); glAttachShader(p, fs);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); fprintf(stderr, "link: %s\n", log); }
    EXPECT_TRUE(ok);
    return p;
}

static void check(GLuint q, GLuint expected_min, GLuint expected_max, const char *what)
{
    fprintf(stderr, "PHASE check %s\n", what);
    GLuint64 v = 0;
    GLint avail = 0;
    int spins = 0;
    glFinish();
    while (!avail && spins++ < 200) glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE, &avail);
    glGetQueryObjectui64v(q, GL_QUERY_RESULT, &v);
    mgl_test_checks++;
    if (!(v >= expected_min && v <= expected_max))
        MGL_FAIL("%s: query=%llu expected [%u..%u]", what, (unsigned long long)v,
                 expected_min, expected_max);
}

TEST(geometry_stats_indirect)
{
    GLuint prog, vao, vbo, ibo, abuf, q_emit, q_inv;
    struct { GLuint count, instances, first, baseInstance; } acmd;
    struct { GLuint count, instances, firstidx, basevertex, baseinstance; } ecmd;
    GLushort idx[3] = { 0, 1, 2 };

    prog = make_prog();
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_tri, k_tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glGenBuffers(1, &abuf);
    glGenQueries(1, &q_emit);
    glGenQueries(1, &q_inv);
    glUseProgram(prog);
    glViewport(0, 0, 8, 8);

    fprintf(stderr, "PHASE arrays-indirect begin\n");
    /* ---- drawArraysIndirect: 1 triangle x 4 instances ---- */
    acmd.count = 3; acmd.instances = 4; acmd.first = 0; acmd.baseInstance = 0;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, abuf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof acmd, &acmd, GL_STATIC_DRAW);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q_emit);
    { GLenum e = glGetError(); if (e) MGL_FAIL("beginQuery emitted: 0x%x", e); }
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q_inv);
    { GLenum e = glGetError(); if (e) MGL_FAIL("beginQuery invocations: 0x%x", e); }
    glDrawArraysIndirect(GL_TRIANGLES, NULL);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    while (glGetError() != GL_NO_ERROR) {}
    check(q_inv, 4, 4, "arrays-indirect invocations");
    check(q_emit, 12, 12, "arrays-indirect emitted");

    fprintf(stderr, "PHASE elements-indirect begin\n");
    /* ---- drawElementsIndirect: the failing CTS case ---- */
    ecmd.count = 3; ecmd.instances = 4; ecmd.firstidx = 0; ecmd.basevertex = 0; ecmd.baseinstance = 0;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, abuf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof ecmd, &ecmd, GL_STATIC_DRAW);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q_emit);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q_inv);
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, NULL);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    while (glGetError() != GL_NO_ERROR) {}
    check(q_inv, 4, 4, "elements-indirect invocations");
    check(q_emit, 12, 12, "elements-indirect emitted");

    /* ---- CTS shape: basevertex/baseinstance 1, command updated via SubData ---- */
    ecmd.count = 3; ecmd.instances = 4; ecmd.firstidx = 0; ecmd.basevertex = 1; ecmd.baseinstance = 1;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, abuf);
    glBufferSubData(GL_DRAW_INDIRECT_BUFFER, 0, sizeof ecmd, &ecmd);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q_emit);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q_inv);
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, NULL);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    while (glGetError() != GL_NO_ERROR) {}
    check(q_inv, 4, 4, "elements-indirect bvi invocations");
    check(q_emit, 12, 12, "elements-indirect bvi emitted");

    glDeleteQueries(2, (GLuint[]){q_emit, q_inv});
}

int main(void)
{
    RUN(geometry_stats_indirect);
    return mgl_test_summary();
}
