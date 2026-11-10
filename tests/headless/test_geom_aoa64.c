/* geom r109 — f64 leaves in arrays-of-arrays stage IO.
 *
 * r108's aoa pass refused any leaf containing a 64-bit float, so every GL
 * 4.3+ interface like `flat out dvec4 v[2][2]` (and dmat/double/struct-with-
 * doubles leaves, their geometry-shader counterparts and their transform
 * feedback capture) failed the whole link.  r109 lifts the refusal: the
 * Private shadow twin keeps real doubles (fp64 only retargets Input/Output
 * POINTER types; Private doubles lower to soft-float u64 values), so every
 * per-leaf stage variable is a plain dvec/dmat/struct the fp64 pass has
 * always converted at its load/store boundaries.
 *
 * Also covered: the fp64 reg_to_iface K_STRUCT fix (whole-struct stores of a
 * struct with a scalar-double member used to record zero for it), and the
 * geometry-shader interface (kernel record leaves + capture writer + the
 * pass-through draw slots) now carrying arrays-of-arrays of doubles.
 */
#include "mgl_test.h"

static GLuint mk_prog(const char *vs, const char *fs, const char *tag)
{
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *s; } sh[2] = {
        { GL_VERTEX_SHADER, vs }, { GL_FRAGMENT_SHADER, fs } };
    for (int i = 0; i < 2; i++)
    {
        GLuint h = glCreateShader(sh[i].t);
        glShaderSource(h, 1, &sh[i].s, NULL);
        glCompileShader(h);
        GLint ok = 0;
        glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
        if (ok != 1)
        {
            char log[2048]; GLsizei len = 0;
            glGetShaderInfoLog(h, sizeof log, &len, log);
            fprintf(stderr, "  [%s] stage %d compile FAILED:\n%s\n", tag, i, log);
        }
        glAttachShader(prog, h);
        glDeleteShader(h);
    }
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [%s] LINK FAILED:\n%s\n", tag, log);
    }
    return prog;
}

static GLuint mk_prog3(const char *vs, const char *gs, const char *fs, const char *tag)
{
    GLuint prog = glCreateProgram();
    const char *srcs[3] = { vs, gs, fs };
    GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint h = glCreateShader(types[i]);
        glShaderSource(h, 1, &srcs[i], NULL);
        glCompileShader(h);
        GLint ok = 0;
        glGetShaderiv(h, GL_COMPILE_STATUS, &ok);
        if (ok != 1)
        {
            char log[2048]; GLsizei len = 0;
            glGetShaderInfoLog(h, sizeof log, &len, log);
            fprintf(stderr, "  [%s] stage %d compile FAILED:\n%s\n", tag, i, log);
        }
        glAttachShader(prog, h);
        glDeleteShader(h);
    }
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [%s] LINK FAILED:\n%s\n", tag, log);
    }
    return prog;
}

static void draw_fullscreen(void)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
}

static void expect_px(const char *what, const GLubyte *expect, int tol)
{
    GLubyte px[4];
    mgl_read_rgba(8, 8, px);
    for (int c = 0; c < 4; c++)
    {
        mgl_test_checks++;
        if (abs((int)px[c] - (int)expect[c]) > tol)
            MGL_FAIL("%s: channel %d == %d, expected ~%d", what, c, px[c], expect[c]);
    }
}

/* ---------------- VS -> FS: leaf shapes ---------------- */

TEST(aoa64_dvec4)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(v[0][0].y), float(v[1][0].z), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_dvec4");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* v[1][1].x = 4/20 ; v[0][0].y = 0.25 ; v[1][0].z = 0.5 */
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_dvec4", e, 2);
    glDeleteProgram(prog);
}

TEST(aoa64_dvec3)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec3 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec3(double(a*2+b+1) / 20.0, double(a*2+b+2) / 20.0, double(a*2+b+3) / 20.0);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec3 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(v[0][0].y), float(v[1][0].z), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_dvec3");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* v[1][1] = (4,5,6)/20 ; v[0][0] = (1,2,3)/20 ; v[1][0] = (3,4,5)/20 */
    const GLubyte e[4] = { 51, 26, 64, 255 };
    expect_px("aoa64_dvec3", e, 3);
    glDeleteProgram(prog);
}

TEST(aoa64_dscalar)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out double d[2][3];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 3; b++)\n"
        "            d[a][b] = double(a*3+b+1) / 20.0;\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in double d[2][3];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(d[1][2]), float(d[0][0]), float(d[1][0]), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_dscalar");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* d[1][2] = 6/20 ; d[0][0] = 1/20 ; d[1][0] = 4/20 */
    const GLubyte e[4] = { 77, 13, 51, 255 };
    expect_px("aoa64_dscalar", e, 3);
    glDeleteProgram(prog);
}

TEST(aoa64_dmat)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dmat2 m[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            m[a][b] = dmat2(dvec2(double(a*4+b+1) / 40.0, double(a*4+b+2) / 40.0),\n"
        "                            dvec2(double(a*4+b+3) / 40.0, double(a*4+b+4) / 40.0));\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dmat2 m[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(m[1][1][0].x), float(m[1][1][1].x), float(m[1][0][0].x), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_dmat");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* m[1][1]: col0.x=6/40, col1.x=8/40 ; m[1][0]: col0.x=5/40 */
    const GLubyte e[4] = { 38, 51, 32, 255 };
    expect_px("aoa64_dmat", e, 3);
    glDeleteProgram(prog);
}

TEST(aoa64_struct)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { dvec3 a; double b; };\n"
        "layout(location = 0) flat out Inner o[2][2];\n"
        "void main() {\n"
        "    for (int i = 0; i < 2; i++)\n"
        "        for (int j = 0; j < 2; j++)\n"
        "        {\n"
        "            o[i][j].a = dvec3(double(i*2+j+1) / 20.0, double(i*2+j+2) / 20.0, 0.5);\n"
        "            o[i][j].b = double(i*2+j+3) / 20.0;\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { dvec3 a; double b; };\n"
        "layout(location = 0) flat in Inner o[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(o[1][1].a.x), float(o[0][0].a.y), float(o[1][0].b), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_struct");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* o[1][1].a.x = 4/20 ; o[0][0].a.y = 2/20 ; o[1][0].b = 5/20 */
    const GLubyte e[4] = { 51, 26, 64, 255 };
    expect_px("aoa64_struct", e, 3);
    glDeleteProgram(prog);
}

/* ---------------- transform feedback ---------------- */

static void xfb_capture_and_check(const char *tag, const char *vs, const char *fs,
                                  const double *want, GLuint stride_doubles)
{
    GLuint prog = mk_prog(vs, fs, tag);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }

    GLuint tfo, buf;
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    const GLdouble *d = (const GLdouble *)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
    for (GLuint i = 0; i < stride_doubles; i++)
    {
        mgl_test_checks++;
        if (fabs(d[i] - want[i]) > 1e-6)
            MGL_FAIL("%s: double %u == %.6f, expected %.6f", tag, i, d[i], want[i]);
    }
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
}

TEST(aoa64_xfb_dvec4)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0, xfb_buffer = 0, xfb_stride = 128, xfb_offset = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    double want[16];
    for (int leaf = 0; leaf < 4; leaf++)
    {
        want[leaf * 4 + 0] = (leaf + 1) / 20.0;
        want[leaf * 4 + 1] = 0.25;
        want[leaf * 4 + 2] = 0.5;
        want[leaf * 4 + 3] = 0.75;
    }
    xfb_capture_and_check("aoa64_xfb_dvec4", vs, fs, want, 16);
}

TEST(aoa64_xfb_api_whole)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec2 v[2][3];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 3; b++)\n"
        "            v[a][b] = dvec2(double(a*3+b+1) / 20.0, 0.5);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    GLuint prog = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    glAttachShader(prog, v);
    glDeleteShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glAttachShader(prog, f);
    glDeleteShader(f);
    const char *names[1] = { "v" };
    glTransformFeedbackVaryings(prog, 1, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [aoa64_xfb_api_whole] LINK FAILED:\n%s\n", log);
        glDeleteProgram(prog);
        return;
    }

    GLuint tfo, buf;
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    GLuint va = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    const GLdouble *d = (const GLdouble *)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
    for (int leaf = 0; leaf < 6; leaf++)
    {
        mgl_test_checks++;
        if (fabs(d[leaf * 2] - (leaf + 1) / 20.0) > 1e-6)
            MGL_FAIL("aoa64_xfb_api_whole: leaf %d x == %.6f, expected %.6f",
                     leaf, d[leaf * 2], (leaf + 1) / 20.0);
        mgl_test_checks++;
        if (fabs(d[leaf * 2 + 1] - 0.5) > 1e-9)
            MGL_FAIL("aoa64_xfb_api_whole: leaf %d y == %.6f, expected 0.5", leaf, d[leaf * 2 + 1]);
    }
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
}

/* ---------------- geometry shader interface ---------------- */

static const char *AOA64_GS_TRI_STRIP_OUT =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 4) out;\n";

TEST(aoa64_gs_out)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(-1 + 2 * float(gl_VertexID & 1), -1 + 2 * float(gl_VertexID >> 1), 0, 1); }\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(-1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(-1, 1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, 1, 0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(v[0][0].y), float(v[1][0].z), 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa64_gs_out");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_gs_out", e, 2);
    glDeleteProgram(prog);
}

TEST(aoa64_gs_in)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(-1 + 2 * float(gl_VertexID & 1), -1 + 2 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    /* the geometry input is <verts>[a][b]: v[vert][a][b] */
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location = 0) flat in dvec4 v[3][2][2];\n"
        "layout(location = 8) flat out vec4 pick;\n"
        "void main() {\n"
        "    pick = vec4(float(v[1][1][1].x), float(v[0][0][0].y), float(v[1][0][0].z), 1.0);\n"
        "    gl_Position = vec4(-1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(-1, 1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, 1, 0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 8) flat in vec4 pick;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = pick; }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa64_gs_in");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* v[1][1][1].x = 4/20 ; v[0][0][0].y = 0.25 ; v[1][0][0].z = 0.5 */
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_gs_in", e, 2);
    glDeleteProgram(prog);
}

TEST(aoa64_gs_xfb)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(-1 + 2 * float(gl_VertexID & 1), -1 + 2 * float(gl_VertexID >> 1), 0, 1); }\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location = 0, xfb_buffer = 0, xfb_stride = 64, xfb_offset = 0) flat out dvec2 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec2(double(gl_PrimitiveIDIn * 4 + a*2+b+1) / 20.0, 0.5);\n"
        "    gl_Position = vec4(-1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, -1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(0, 1, 0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa64_gs_xfb");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }

    GLuint tfo, buf;
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    GLuint va = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    /* one emitted vertex carries 4 leaves: prim 0 -> 1/20,2/20,3/20,4/20 */
    const GLdouble *d = (const GLdouble *)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
    for (int leaf = 0; leaf < 4; leaf++)
    {
        mgl_test_checks++;
        if (fabs(d[leaf * 2] - (leaf + 1) / 20.0) > 1e-6)
            MGL_FAIL("aoa64_gs_xfb: leaf %d x == %.6f, expected %.6f", leaf,
                     d[leaf * 2], (leaf + 1) / 20.0);
        mgl_test_checks++;
        if (fabs(d[leaf * 2 + 1] - 0.5) > 1e-9)
            MGL_FAIL("aoa64_gs_xfb: leaf %d y == %.6f, expected 0.5", leaf, d[leaf * 2 + 1]);
    }
    glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
}


/* r107/r108 documented gap, closed by r109's fp64 reg_to_iface member fix:
 * a whole-struct FUNCTION COPY of a struct containing an array-of-matrices
 * plus a scalar double used to record garbage for the double (slot-typed
 * extraction from the register-form struct). */
TEST(aoa64_aomstruct_fn_copy)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "struct S { dmat2 m[2]; double d; };\n"
        "layout(location = 0) flat out S s;\n"
        "S mk(double x) { S r; r.m[0] = dmat2(x); r.m[1] = dmat2(x*2.0); r.d = x*3.0; return r; }\n"
        "void main() {\n"
        "    s = mk(0.25);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct S { dmat2 m[2]; double d; };\n"
        "layout(location = 0) flat in S s;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(s.m[1][0].x), float(s.d), float(s.m[0][1].y), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_aomstruct_fn_copy");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* m[1][0].x = 0.5 ; d = 0.75 ; m[0][1].y = 0.25 */
    const GLubyte e[4] = { 128, 191, 64, 255 };
    expect_px("aoa64_aomstruct_fn_copy", e, 3);
    glDeleteProgram(prog);
}

/* the FS consumes the aoa with DYNAMIC indices -- stays in the Private shadow
 * (the per-leaf stage variables are only read once by the prologue) */
TEST(aoa64_dynamic_read)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "    double acc = 0.0;\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            acc += v[a][b].x;\n"
        "    color = vec4(float(acc * 20.0 / 10.0), 0.0, 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_dynamic_read");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* (1+2+3+4)/20 * 20 / 10 = 1.0 */
    const GLubyte e[4] = { 255, 0, 0, 255 };
    expect_px("aoa64_dynamic_read", e, 2);
    glDeleteProgram(prog);
}

/* three array levels, still within the pass's leaf budget */
TEST(aoa64_three_levels)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec2 v[2][2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            for (int c = 0; c < 2; c++)\n"
        "                v[a][b][c] = dvec2(double(a*4+b*2+c+1) / 20.0, 0.5);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec2 v[2][2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1][0].x), float(v[1][1][1].x), float(v[0][1][1].x), 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa64_three_levels");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* v[1][1][0].x = 7/20 ; v[1][1][1].x = 8/20 ; v[0][1][1].x = 4/20 */
    const GLubyte e[4] = { 89, 102, 51, 255 };
    expect_px("aoa64_three_levels", e, 3);
    glDeleteProgram(prog);
}

/* ---------------- r110: TES output f64 aoa leaves (flat-array collapse) ------- */

static const char *k_tes_vs =
    "#version 460 core\n"
    "layout(location = 0) in vec4 pos;\n"
    "void main() { gl_Position = pos; }\n";
static const char *k_tes_tcs =
    "#version 460 core\n"
    "layout(vertices = 3) out;\n"
    "void main() { if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1; gl_TessLevelOuter[1] = 1; gl_TessLevelOuter[2] = 1; gl_TessLevelInner[0] = 1; }\n"
    "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position; }\n";
static const char *k_tes_tes =
    "#version 460 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location = 0) flat out dvec4 v[2][2];\n"
    "void main() {\n"
    "    for (int a = 0; a < 2; a++)\n"
    "        for (int b = 0; b < 2; b++)\n"
    "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
    "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "}\n";

/* a 5-stage-ready builder: custom TES, optional GS, FS */
static GLuint tes_prog_tes(const char *tes, const char *gs, const char *fs, const char *tag)
{
    GLuint p = glCreateProgram();
    const char *srcs[5] = { k_tes_vs, k_tes_tcs, tes, gs, fs };
    GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER,
                        GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 5; i++)
    {
        if (!srcs[i]) continue;
        GLuint h = glCreateShader(types[i]);
        glShaderSource(h, 1, &srcs[i], NULL);
        glCompileShader(h);
        glAttachShader(p, h);
        glDeleteShader(h);
    }
    glLinkProgram(p);
    GLint lc = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "  [%s] LINK FAILED:\n%s\n", tag, log);
    }
    return p;
}

static GLuint tes_prog(const char *gs, const char *fs, const char *tag)
{
    return tes_prog_tes(k_tes_tes, gs, fs, tag);
}

/* one GL_PATCHES draw of a triangle covering the whole viewport */
static void tes_draw(void)
{
    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float tri[3][4] = { { -1, -1, 0, 1 }, { 3, -1, 0, 1 }, { -1, 3, 0, 1 } };
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glDrawArrays(GL_PATCHES, 0, 3);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

/* plain raster: the flat-array TES output feeds the FS through the tess
 * raster vertex function */
TEST(aoa64_tes_out)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(v[0][0].y), float(v[1][0].z), 1.0); }\n";
    GLuint prog = tes_prog(NULL, fs, "aoa64_tes_out");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    tes_draw();
    EXPECT_NO_GL_ERROR();
    /* v[1][1].x = 4/20 ; v[0][0].y = 0.25 ; v[1][0].z = 0.5 */
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_tes_out", e, 2);
    glDeleteProgram(prog);
}

/* through a geometry stage: the TES output is CAPTURED (the tess capture
 * kernel packs a compact record) and the geometry import repacks it; the fp64
 * slot members "v_0".."v_7" must pair with the GL view "v" by name */
TEST(aoa64_tes_gs)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in; layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 0) flat in dvec4 v[][2][2];\n"
        "layout(location = 0) flat out dvec4 vo;\n"
        "void main() { vo = v[0][1][1]; gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  vo = v[1][0][1]; gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  vo = v[2][1][0]; gl_Position = gl_in[2].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v.x), float(v.y), float(v.z), 1.0); }\n";
    GLuint prog = tes_prog(gs, fs, "aoa64_tes_gs");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    tes_draw();
    EXPECT_NO_GL_ERROR();
    /* the strip's provoking (last) vertex wrote v[1][1][0]: x = 4/20, y = .25, z = .5 */
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_tes_gs", e, 2);
    glDeleteProgram(prog);
}

/* whole-aoa transform feedback of the TES output (the tess capture + the
 * geometry-domain xfb repack): the record must carry all 16 doubles per
 * tessellated vertex, in row-major flat element order */
TEST(aoa64_tes_xfb)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    GLuint prog = tes_prog(NULL, fs, "aoa64_tes_xfb");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    static const char *vars[] = { "v" };
    glTransformFeedbackVaryings(prog, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    GLuint tfb = 0, q = 0;
    glGenBuffers(1, &tfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glGenQueries(1, &q);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_PATCHES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    tes_draw();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    GLint n = 0;
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &n);
    EXPECT_EQ_INT(n, 3);    /* equal_spacing, outer levels 1,1,1 -> one triangle */
    double rec[16];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
    for (int i = 0; i < 4; i++)
    {
        /* element (a,b) flat = a*2+b: x = (a*2+b+1)/20, y = 0.25, z = 0.5, w = 0.75 */
        EXPECT_NEAR(rec[i * 4 + 0], (i + 1) / 20.0, 1e-9);
        EXPECT_NEAR(rec[i * 4 + 1], 0.25, 1e-9);
        EXPECT_NEAR(rec[i * 4 + 2], 0.5, 1e-9);
        EXPECT_NEAR(rec[i * 4 + 3], 0.75, 1e-9);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &tfb);
    glDeleteProgram(prog);
}

/* matrix leaves keep the r109 refusal when fp64 cannot lower the whole
 * variable (a DYNAMIC leaf index): the aom-output pass would re-split a
 * flattened dmat array per column and rename the elements, so the tess
 * capture's name pairing would drop the doubles — fail the link LOUDLY
 * instead of rendering wrong values.  (Constant-index dmat aoa leaves lower
 * fine unsplit and are covered by aoa64_dmat.) */
TEST(aoa64_tes_refuse_dmat)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location = 0) flat out dmat2 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dmat2(double(a * 2 + b + 1) / 20.0);\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dmat2 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1][0].x)); }\n";
    GLuint p = glCreateProgram();
    const char *srcs[4] = { k_tes_vs, k_tes_tcs, tes, fs };
    GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 4; i++)
    {
        GLuint h = glCreateShader(types[i]);
        glShaderSource(h, 1, &srcs[i], NULL);
        glCompileShader(h);
        glAttachShader(p, h);
        glDeleteShader(h);
    }
    glLinkProgram(p);
    GLint lc = 1;
    glGetProgramiv(p, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 0);
    if (lc == 0)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        mgl_test_checks++;
        if (!strstr(log, "fp64"))
            MGL_FAIL("aoa64_tes_refuse_dmat: expected an fp64 refusal in the log, got:\n%s", log);
    }
    glDeleteProgram(p);
    while (glGetError() != GL_NO_ERROR) {}
}

/* 3 levels of dvec2 leaves — the collapse flattens 2x2x2 into 8 elements */
TEST(aoa64_tes_3levels)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location = 0) flat out dvec2 v[2][2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++)\n"
        "        v[a][b][c] = dvec2(double(a*4+b*2+c+1) / 40.0, 0.5);\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec2 v[2][2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1][1].x), float(v[0][1][0].y), float(v[1][0][1].x), 1.0); }\n";
    GLuint prog = tes_prog_tes(tes, NULL, fs, "aoa64_tes_3levels");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    tes_draw();
    EXPECT_NO_GL_ERROR();
    /* v[1][1][1].x = 8/40 ; v[0][1][0].y = 0.5 ; v[1][0][1].x = 6/40 */
    const GLubyte e[4] = { 51, 128, 38, 255 };
    expect_px("aoa64_tes_3levels", e, 2);
    glDeleteProgram(prog);
}

/* explicit xfb layout: the aoa at offset 0 (row-major flat element order),
 * a scalar double at 128, stride 144 — render AND capture */
TEST(aoa64_tes_xfb_explicit)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(xfb_buffer = 0, xfb_stride = 144) out;\n"
        "layout(xfb_buffer = 0, xfb_offset = 0) flat out dvec4 v[2][2];\n"
        "layout(xfb_buffer = 0, xfb_offset = 128) flat out double tail;\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)\n"
        "        v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    tail = 0.125LF;\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 8) flat in double tail;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(tail), float(v[0][0].z), 1.0); }\n";
    GLuint prog = tes_prog_tes(tes, NULL, fs, "aoa64_tes_xfb_explicit");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    static const char *vars[] = { "v", "tail" };
    glTransformFeedbackVaryings(prog, 2, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    tes_draw();
    EXPECT_NO_GL_ERROR();
    /* v[1][1].x = 0.2 ; tail = 0.125 ; v[0][0].z = 0.5 */
    const GLubyte e[4] = { 51, 32, 128, 255 };
    expect_px("aoa64_tes_xfb_explicit", e, 2);

    GLuint tfb = 0;
    glGenBuffers(1, &tfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_PATCHES);
    tes_draw();
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    double rec[18];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
    for (int e2 = 0; e2 < 4; e2++)
    {
        EXPECT_NEAR(rec[e2 * 4 + 0], (e2 + 1) / 20.0, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 1], 0.25, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 2], 0.5, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 3], 0.75, 1e-9);
    }
    EXPECT_NEAR(rec[16], 0.125, 1e-9);
    glDeleteBuffers(1, &tfb);
    glDeleteProgram(prog);
}

/* TWO flattened outputs (v at location 0, w at 8), SEPARATE_ATTRIBS xfb */
TEST(aoa64_tes_two_outputs_sep)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "layout(location = 8) flat out dvec2 w[2][3];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)\n"
        "        v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    for (int a = 0; a < 2; a++) for (int b = 0; b < 3; b++)\n"
        "        w[a][b] = dvec2(double(a*3+b+1) / 30.0, 0.75);\n"
        "    gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 8) flat in dvec2 w[2][3];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(w[1][2].x), float(w[0][0].y), 1.0); }\n";
    GLuint prog = tes_prog_tes(tes, NULL, fs, "aoa64_tes_two_outputs_sep");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    static const char *vars[] = { "v", "w" };
    glTransformFeedbackVaryings(prog, 2, vars, GL_SEPARATE_ATTRIBS);
    glLinkProgram(prog);
    lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    tes_draw();
    EXPECT_NO_GL_ERROR();
    /* v[1][1].x = 0.2 ; w[1][2].x = 6/30 = 0.2 ; w[0][0].y = 0.75 */
    const GLubyte e[4] = { 51, 51, 191, 255 };
    expect_px("aoa64_tes_two_outputs_sep", e, 2);

    GLuint tb[2] = { 0, 0 };
    glGenBuffers(2, tb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb[0]);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tb[0]);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb[1]);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tb[1]);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_PATCHES);
    tes_draw();
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    double rec[24];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb[0]);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 4 * 4 * sizeof(double), rec);
    for (int e2 = 0; e2 < 4; e2++)
    {
        EXPECT_NEAR(rec[e2 * 4 + 0], (e2 + 1) / 20.0, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 1], 0.25, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 2], 0.5, 1e-9);
        EXPECT_NEAR(rec[e2 * 4 + 3], 0.75, 1e-9);
    }
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb[1]);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 6 * 2 * sizeof(double), rec);
    for (int e2 = 0; e2 < 6; e2++)
    {
        EXPECT_NEAR(rec[e2 * 2 + 0], (e2 + 1) / 30.0, 1e-9);
        EXPECT_NEAR(rec[e2 * 2 + 1], 0.75, 1e-9);
    }
    glDeleteBuffers(2, tb);
    glDeleteProgram(prog);
}

/* ---------------- documented gap: TES output (refused, loud) ---------------- */

/* r110: a geometry stage with an aoa INPUT *and* an aoa OUTPUT used to produce
 * a module with SSA ids defined twice (the shadow prologue and the terminator
 * flush shared chain/val ids; several whole-array stores to one variable
 * reused one id batch) — "fp64: module parse: Id N is defined more than
 * once".  Pixel-verified pass-through of a dvec4 aoa through both sides of
 * the geometry stage, plus whole-array copies (vo = v[i]). */
TEST(aoa64_gs_in_and_out)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0, 0, 1);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in; layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location = 0) flat in dvec4 v[3][2][2];\n"
        "layout(location = 12) flat out dvec4 vo[2][2];\n"
        "void main() {\n"
        "    vo = v[0];\n"
        "    gl_Position = vec4(-1, -1, 0, 1); EmitVertex();\n"
        "    vo = v[1];\n"
        "    gl_Position = vec4(1, -1, 0, 1); EmitVertex();\n"
        "    vo = v[2];\n"
        "    gl_Position = vec4(-1, 1, 0, 1); EmitVertex();\n"
        "    gl_Position = vec4(1, 1, 0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 12) flat in dvec4 vo[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(vo[1][1].x), float(vo[0][0].y), float(vo[1][0].z), 1.0); }\n";
    GLuint p = glCreateProgram();
    const char *srcs[3] = { vs, gs, fs };
    GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint h = glCreateShader(types[i]);
        glShaderSource(h, 1, &srcs[i], NULL);
        glCompileShader(h);
        glAttachShader(p, h);
        glDeleteShader(h);
    }
    glLinkProgram(p);
    GLint lc = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "  [aoa64_gs_in_and_out] LINK FAILED:\n%s\n", log);
        glDeleteProgram(p);
        return;
    }
    glUseProgram(p);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
    /* every VS vertex writes the same aoa, so any provoking vertex gives
     * vo[1][1].x = 4/20 = .2, vo[0][0].y = .25, vo[1][0].z = .5 */
    const GLubyte e[4] = { 51, 64, 128, 255 };
    expect_px("aoa64_gs_in_and_out", e, 2);
    glDeleteProgram(p);
}

/* r110: TES aoa output consumed by a geometry stage and captured by xfb: the
 * tess capture packs the flattened TES record, the geometry import repacks it
 * into the GS input, the GS re-emits an aoa and xfb captures it */
TEST(aoa64_tes_gs_xfb)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in; layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 0) flat in dvec4 v[][2][2];\n"
        "layout(location = 0) flat out dvec4 vo[2][2];\n"
        "void main() { vo = v[0]; gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  vo = v[1]; gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  vo = v[2]; gl_Position = gl_in[2].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 vo[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(vo[1][1].x)); }\n";
    GLuint prog = tes_prog(gs, fs, "aoa64_tes_gs_xfb");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    static const char *vars[] = { "vo" };
    glTransformFeedbackVaryings(prog, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    GLuint tfb = 0;
    glGenBuffers(1, &tfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glUseProgram(prog);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_PATCHES);
    tes_draw();
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();
    double rec[48];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tfb);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
    /* 3 emitted vertices, each copying the tessellated input's v: all four
     * leaves, in flat element order */
    for (int i = 0; i < 3; i++)
        for (int e = 0; e < 4; e++)
        {
            EXPECT_NEAR(rec[(i * 4 + e) * 4 + 0], (e + 1) / 20.0, 1e-9);
            EXPECT_NEAR(rec[(i * 4 + e) * 4 + 1], 0.25, 1e-9);
            EXPECT_NEAR(rec[(i * 4 + e) * 4 + 2], 0.5, 1e-9);
            EXPECT_NEAR(rec[(i * 4 + e) * 4 + 3], 0.75, 1e-9);
        }
    glDeleteBuffers(1, &tfb);
    glDeleteProgram(prog);
}

TEST(aoa64_tes_gap)
{
    while (glGetError() != GL_NO_ERROR) {}
    static const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(-1 + 2 * float(gl_VertexID & 1), -1 + 2 * float(gl_VertexID >> 1), 0, 1); }\n";
    static const char *tcs =
        "#version 460 core\n"
        "layout(vertices = 3) out;\n"
        "void main() { if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1; gl_TessLevelOuter[1] = 1; gl_TessLevelOuter[2] = 1; gl_TessLevelInner[0] = 1; }\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position; }\n";
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location = 0) flat out dvec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = dvec4(double(a*2+b+1) / 20.0, 0.25, 0.5, 0.75);\n"
        "    gl_Position = gl_in[0].gl_Position;\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in dvec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(float(v[1][1].x), float(v[0][0].y), float(v[1][0].z), 1.0); }\n";
    GLuint prog = glCreateProgram();
    const char *srcs[4] = { vs, tcs, tes, fs };
    GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 4; i++)
    {
        GLuint h = glCreateShader(types[i]);
        glShaderSource(h, 1, &srcs[i], NULL);
        glCompileShader(h);
        glAttachShader(prog, h);
        glDeleteShader(h);
    }
    glLinkProgram(prog);
    GLint lc = 1;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    /* r110 closed the r109 gap: a TES OUTPUT aoa with scalar/vector-of-double
     * leaves collapses into one flat array variable that keeps the GL name (the
     * tess capture pairs its fp64 slots "<name>_<slot>" with the GL view by
     * name), so the link now SUCCEEDS and the doubles flow through — checked
     * pixel-exact in aoa64_tes_out below.  This variant keeps the r109 shape
     * (the loop's dynamic v[a][b] stores) but writes a degenerate gl_Position,
     * so only the link status is asserted here. */
    EXPECT_EQ_INT(lc, 1);
    glDeleteProgram(prog);
    while (glGetError() != GL_NO_ERROR) {}
}

int main(void)
{
    RUN(aoa64_dvec4);
    RUN(aoa64_dvec3);
    RUN(aoa64_dscalar);
    RUN(aoa64_dmat);
    RUN(aoa64_struct);
    RUN(aoa64_xfb_dvec4);
    RUN(aoa64_xfb_api_whole);
    RUN(aoa64_gs_out);
    RUN(aoa64_gs_in);
    RUN(aoa64_gs_xfb);
    RUN(aoa64_aomstruct_fn_copy);
    RUN(aoa64_dynamic_read);
    RUN(aoa64_three_levels);
    RUN(aoa64_tes_out);
    RUN(aoa64_tes_gs);
    RUN(aoa64_tes_xfb);
    RUN(aoa64_tes_3levels);
    RUN(aoa64_tes_xfb_explicit);
    RUN(aoa64_tes_two_outputs_sep);
    RUN(aoa64_tes_refuse_dmat);
    RUN(aoa64_gs_in_and_out);
    RUN(aoa64_tes_gs_xfb);
    RUN(aoa64_tes_gap);
    return mgl_test_summary();
}
