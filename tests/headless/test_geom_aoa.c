/* geom r108 — arrays-of-arrays stage IO.
 *
 * SPIRV-Cross refuses a stage variable with more than one array dimension
 * ("MSL cannot emit arrays-of-arrays in input and output variables"), so any
 * GL 4.3+ interface like `flat out vec4 v[2][3]` failed the whole link.
 * geometry.c's mgl_spirv_lower_aoa_stage_io splits such variables into one
 * stage variable per leaf element plus a Private shadow twin (dynamic indexing
 * and helper functions keep working); the xfb side captures the leaves in
 * row-major order.  These tests pixel-verify the render path and word-verify
 * the capture path. */
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

/* vec4 leaves: render through, constant and dynamic indices */
TEST(aoa_vec4)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out vec4 v[2][3];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 3; b++)\n"
        "            v[a][b] = vec4(float(a*3+b+1) / 20.0);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in vec4 v[2][3];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "    int j = 1 + 1;   /* a dynamic index into the shadow */\n"
        "    color = vec4(v[1][2].x / 5.0, v[0][0].y, v[1][1].z + v[j][1].w * 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_prog(vs, fs, "aoa_vec4");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* v[1][2].x = 6/20, /5 -> 0.06 -> 15 ; v[0][0].y = 1/20 -> 13 ; v[1][1].z = 5/20 -> 64 */
    const GLubyte expect[4] = { 15, 13, 64, 255 };
    expect_px("aoa_vec4", expect, 2);
    glDeleteProgram(prog);
}

/* matrix leaves: each mat4 leaf spans 4 locations */
TEST(aoa_mat)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out mat4 m[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            m[a][b] = mat4(float(a*2+b+1));\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0, 1);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in mat4 m[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(m[1][1][3].w / 8.0, m[0][1][0].x / 4.0, m[1][0][2].z / 6.0, 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa_mat");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* m[1][1] = 4, m[0][1] = 2, m[1][0] = 3 */
    const GLubyte expect[4] = { 128, 128, 128, 255 };
    expect_px("aoa_mat", expect, 2);
    glDeleteProgram(prog);
}

/* struct leaves with an array-of-matrix member: composes with the
 * aom-struct pass (mgl_spirv_lower_aom_struct_member) that runs after */
TEST(aoa_struct)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location = 0) flat out Inner o[2][3];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 3; b++) {\n"
        "            float s = float(a * 3 + b + 1);\n"
        "            o[a][b].m[0] = mat2x3(s, s, s, s, s, s);\n"
        "            o[a][b].m[1] = mat2x3(s*2, s*2, s*2, s*2, s*2, s*2);\n"
        "            o[a][b].f = s;\n"
        "        }\n"
        "    gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(location = 0) flat in Inner i[2][3];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(i[1][2].f / 7.0, i[0][0].m[0][1].x, i[1][1].m[1][0].z * 0.05, 1.0); }\n";
    GLuint prog = mk_prog(vs, fs, "aoa_struct");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen();
    EXPECT_NO_GL_ERROR();
    /* i[1][2].f = 6 -> 6/7 -> 219 ; m[0][1].x of s=1 -> 1 ; i[1][1] s=5, m[1] = 10 -> 0.5 */
    const GLubyte expect[4] = { 219, 255, 128, 255 };
    expect_px("aoa_struct", expect, 2);
    glDeleteProgram(prog);
}

/* whole-variable xfb capture of a vec4[2][2] via xfb_offset: the leaves land
 * in the record in row-major order */
static GLuint mk_xfb_prog(const char *vs, const char *tag)
{
    return mk_prog(vs, "#version 460 core\nlayout(location=0) out vec4 c; void main() { c = vec4(1); }", tag);
}

static GLuint begin_capture(GLuint prog, GLuint *tfo, GLuint *buf)
{
    glUseProgram(prog);
    glGenTransformFeedbacks(1, tfo);
    glGenBuffers(1, buf);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, *tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, *buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    return 0;
}

TEST(aoa_xfb_vec)
{
    static const char *vs =
        "#version 460 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout(xfb_buffer=0, xfb_offset=0, location=0) flat out vec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++) {\n"
        "            float s = float(a * 2 + b + 1);\n"
        "            v[a][b] = vec4(s, s * 10.0, s * 100.0, s * 1000.0);\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_xfb_prog(vs, "aoa_xfb_vec");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    GLuint tfo, buf;
    begin_capture(prog, &tfo, &buf);
    float out[64];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int e = 0; e < 4; e++)
    {
        float s = (float)(e + 1);
        float ex[4] = { s, s * 10.f, s * 100.f, s * 1000.f };
        for (int c = 0; c < 4; c++)
        {
            mgl_test_checks++;
            if (fabsf(out[e * 4 + c] - ex[c]) > 0.01f)
                MGL_FAIL("aoa_xfb_vec: leaf %d comp %d: %.2f, expected %.2f", e, c, out[e * 4 + c], ex[c]);
        }
    }
    glDeleteBuffers(1, &buf);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteProgram(prog);
}

/* struct leaves + whole-variable capture: one varying per leaf member,
 * packed with scalar alignment (GL 4.6 13.2.2) */
TEST(aoa_xfb_struct)
{
    static const char *vs =
        "#version 460 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "struct Inner { mat2x3 m[2]; float f; };\n"
        "layout(xfb_buffer=0, xfb_offset=0, location=0) flat out Inner o[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++) {\n"
        "            float s = float(a * 2 + b + 1);\n"
        "            o[a][b].m[0] = mat2x3(s, s, s, s, s, s);\n"
        "            o[a][b].m[1] = mat2x3(s*10, s*10, s*10, s*10, s*10, s*10);\n"
        "            o[a][b].f = s;\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_xfb_prog(vs, "aoa_xfb_struct");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    GLint n = 0;
    glGetProgramiv(prog, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
    EXPECT_EQ_INT(n, 8);
    char nm[128]; GLsizei len = 0; GLenum ty = 0; GLint sz = 0;
    glGetTransformFeedbackVarying(prog, 6, sizeof nm, &len, &sz, &ty, nm);
    EXPECT_STREQ(nm, "o[1][1].m");
    EXPECT_EQ_INT(sz, 2);
    EXPECT_EQ_INT(ty, GL_FLOAT_MAT2x3);
    glGetTransformFeedbackVarying(prog, 7, sizeof nm, &len, &sz, &ty, nm);
    EXPECT_STREQ(nm, "o[1][1].f");
    EXPECT_EQ_INT(ty, GL_FLOAT);
    GLuint tfo, buf;
    begin_capture(prog, &tfo, &buf);
    float out[64];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    /* element span: m[2] (2 x 6 words) + f, packed -> 13 words */
    for (int e = 0; e < 4; e++)
    {
        float s = (float)(e + 1);
        for (int w = 0; w < 6; w++)
        {
            mgl_test_checks++;
            if (fabsf(out[e * 13 + w] - s) > 0.01f)
                MGL_FAIL("aoa_xfb_struct: elem %d m0 w%d: %.2f, expected %.2f", e, w, out[e * 13 + w], s);
        }
        for (int w = 0; w < 6; w++)
        {
            mgl_test_checks++;
            if (fabsf(out[e * 13 + 6 + w] - s * 10.f) > 0.05f)
                MGL_FAIL("aoa_xfb_struct: elem %d m1 w%d: %.2f, expected %.2f", e, w, out[e * 13 + 6 + w], s * 10.f);
        }
        mgl_test_checks++;
        if (fabsf(out[e * 13 + 12] - s) > 0.01f)
            MGL_FAIL("aoa_xfb_struct: elem %d f: %.2f, expected %.2f", e, out[e * 13 + 12], s);
    }
    glDeleteBuffers(1, &buf);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteProgram(prog);
}

/* the API path: glTransformFeedbackVaryings("v") captures the whole
 * arrays-of-arrays variable, "v[1][1]" selects one leaf */
TEST(aoa_xfb_api)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out vec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++) {\n"
        "            float s = float(a * 2 + b + 1);\n"
        "            v[a][b] = vec4(s);\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_xfb_prog(vs, "aoa_xfb_api");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    const char *names[2] = { "v", "v[1][1]" };
    glTransformFeedbackVaryings(prog, 2, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    {
        GLint n = 0;
        glGetProgramiv(prog, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
        EXPECT_EQ_INT(n, 2);
        char nm[128]; GLsizei len = 0; GLenum ty = 0; GLint sz = 0;
        glGetTransformFeedbackVarying(prog, 0, sizeof nm, &len, &sz, &ty, nm);
        EXPECT_STREQ(nm, "v");
        EXPECT_EQ_INT(sz, 4);
        EXPECT_EQ_INT(ty, GL_FLOAT_VEC4);
        glGetTransformFeedbackVarying(prog, 1, sizeof nm, &len, &sz, &ty, nm);
        EXPECT_STREQ(nm, "v[1][1]");
        EXPECT_EQ_INT(sz, 1);
    }
    GLuint tfo, buf;
    begin_capture(prog, &tfo, &buf);
    float out[64];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int e = 0; e < 4; e++)
        for (int c = 0; c < 4; c++)
        {
            mgl_test_checks++;
            if (fabsf(out[e * 4 + c] - (float)(e + 1)) > 0.01f)
                MGL_FAIL("aoa_xfb_api: whole leaf %d comp %d: %.2f, expected %d", e, c, out[e * 4 + c], e + 1);
        }
    for (int c = 0; c < 4; c++)
    {
        mgl_test_checks++;
        if (fabsf(out[16 + c] - 4.0f) > 0.01f)
            MGL_FAIL("aoa_xfb_api: v[1][1] comp %d: %.2f, expected 4", c, out[16 + c]);
    }
    glDeleteBuffers(1, &buf);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteProgram(prog);
}

/* ---------------- geometry shader interplay ---------------- */

static GLuint mk_prog3(const char *vs, const char *gs, const char *fs, const char *tag){
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *s; } sh[3] = {
        { GL_VERTEX_SHADER, vs }, { GL_GEOMETRY_SHADER, gs }, { GL_FRAGMENT_SHADER, fs } };
    for (int i = 0; i < 3; i++)
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

static void draw3(void)
{
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
}

/* a one-level array of structs into a GS — the writer-side capture expression
 * used to drop the element subscript ("o.m" instead of "o[1].m") and the GS
 * input side never unrolled array elements at all */
TEST(aoa_gs_one_level_struct)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { vec2 v; float f; };\n"
        "layout(location = 0) flat out Inner o[2];\n"
        "void main() {\n"
        "    o[0].v = vec2(1.0, 0.5); o[0].f = 0.25;\n"
        "    o[1].v = vec2(0.75, 0.125); o[1].f = 0.5;\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { vec2 v; float f; };\n"
        "layout(location = 0) flat in Inner o[3][2];\n"
        "layout(location = 5) flat out float pick;\n"
        "void main() {\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        pick = o[i][1].f + o[i][0].v.y * 0.5;\n"
        "        gl_Position = gl_in[i].gl_Position;\n"
        "        EmitVertex();\n"
        "    }\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 5) flat in float pick;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(pick, 1.0, 0.5, 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa_gs_one_level_struct");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw3();
    EXPECT_NO_GL_ERROR();
    const GLubyte expect[4] = { 191, 255, 128, 255 };
    expect_px("aoa_gs_one_level_struct", expect, 3);
    glDeleteProgram(prog);
}

/* GS consumes an arrays-of-arrays VS output (vec4 leaves) */
TEST(aoa_gs_in_vec)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out vec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++) {\n"
        "            float s = float(a * 2 + b + 1);\n"
        "            v[a][b] = vec4(s * 0.1);\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 0) flat in vec4 v[3][2][2];\n"
        "layout(location = 4) flat out float pick;\n"
        "void main() {\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        pick = v[i][1][1].x + v[i][0][0].w;\n"
        "        gl_Position = gl_in[i].gl_Position;\n"
        "        EmitVertex();\n"
        "    }\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 4) flat in float pick;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(pick, 1.0, 0.5, 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa_gs_in_vec");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw3();
    EXPECT_NO_GL_ERROR();
    const GLubyte expect[4] = { 128, 255, 128, 255 };
    expect_px("aoa_gs_in_vec", expect, 2);
    glDeleteProgram(prog);
}

/* GS consumes an arrays-of-arrays of STRUCTS with a matrix member */
TEST(aoa_gs_in_struct)
{
    static const char *vs =
        "#version 460 core\n"
        "struct Inner { mat2 m; float f; };\n"
        "layout(location = 0) flat out Inner o[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++) {\n"
        "            float s = float(a * 2 + b + 1);\n"
        "            o[a][b].m = mat2(s, s, s, s);\n"
        "            o[a][b].f = s;\n"
        "        }\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "struct Inner { mat2 m; float f; };\n"
        "layout(location = 0) flat in Inner o[3][2][2];\n"
        "layout(location = 13) flat out float pick;\n"
        "void main() {\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        pick = o[i][1][1].f + o[i][0][1].m[1].x * 0.1;\n"
        "        gl_Position = gl_in[i].gl_Position;\n"
        "        EmitVertex();\n"
        "    }\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 13) flat in float pick;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(pick / 8.0, 1.0, 0.5, 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa_gs_in_struct");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw3();
    EXPECT_NO_GL_ERROR();
    const GLubyte expect[4] = { 134, 255, 128, 255 };
    expect_px("aoa_gs_in_struct", expect, 2);
    glDeleteProgram(prog);
}

/* GS EMITS an arrays-of-arrays output (matrix + scalar leaves) to the FS */
TEST(aoa_gs_out)
{
    static const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 0) flat out mat4 m[2][2];\n"
        "layout(location = 16) flat out float f[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            m[a][b] = mat4(float(a * 2 + b + 1));\n"
        "    f[0][0] = 0.25; f[0][1] = 0.5; f[1][0] = 0.75; f[1][1] = 1.0;\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        gl_Position = gl_in[i].gl_Position;\n"
        "        EmitVertex();\n"
        "    }\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in mat4 m[2][2];\n"
        "layout(location = 16) flat in float f[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(m[1][1][3].w / 8.0, f[1][1], f[0][1], 1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa_gs_out");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    draw3();
    EXPECT_NO_GL_ERROR();
    const GLubyte expect[4] = { 128, 255, 128, 255 };
    expect_px("aoa_gs_out", expect, 2);
    glDeleteProgram(prog);
}

/* xfb of a GS-emitted arrays-of-arrays output (stream capture of the record) */
TEST(aoa_gs_xfb)
{
    static const char *vs =
        "#version 460 core\n"
        "void main() {\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    static const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location = 0, xfb_buffer = 0, xfb_offset = 0) flat out vec4 v[2][2];\n"
        "void main() {\n"
        "    for (int i = 0; i < 3; i++) {\n"
        "        v[0][0] = vec4(float(i + 1));\n"
        "        v[0][1] = vec4(float(i + 1) * 10.0);\n"
        "        v[1][0] = vec4(float(i + 1) * 100.0);\n"
        "        v[1][1] = vec4(float(i + 1) * 1000.0);\n"
        "        gl_Position = gl_in[i].gl_Position;\n"
        "        EmitVertex();\n"
        "    }\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    GLuint prog = mk_prog3(vs, gs, fs, "aoa_gs_xfb");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    GLuint tfo, buf;
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(1, &buf);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    float out[64];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int vert = 0; vert < 3; vert++)
        for (int e = 0; e < 4; e++)
        {
            float ex = (float)(vert + 1);
            for (int q = 0; q < e; q++) ex *= 10.f;
            for (int c = 0; c < 4; c++)
            {
                mgl_test_checks++;
                if (fabsf(out[vert * 16 + e * 4 + c] - ex) > 0.5f)
                    MGL_FAIL("aoa_gs_xfb: vert %d leaf %d comp %d: %.2f, expected %.2f",
                             vert, e, c, out[vert * 16 + e * 4 + c], ex);
            }
        }
    glDeleteBuffers(1, &buf);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

/* TES OUTPUT arrays-of-arrays (a TES output is a plain stage variable; the
 * per-vertex TES INPUTS stay the tess domain's to route) */
static GLuint mk_prog4(const char *vs, const char *tcs, const char *tes, const char *fs, const char *tag)
{
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *s; } sh[4] = {
        { GL_VERTEX_SHADER, vs }, { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes }, { GL_FRAGMENT_SHADER, fs } };
    for (int i = 0; i < 4; i++)
    {
        if (!sh[i].s) continue;
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

TEST(aoa_tes_out)
{
    static const char *vs =
        "#version 460 core\n"
        "void main() { gl_Position = vec4(float(gl_VertexID & 1) * 4.0 - 1.0, float(gl_VertexID >> 1) * 4.0 - 1.0, 0.0, 1.0); }\n";
    static const char *tcs =
        "#version 460 core\n"
        "layout(vertices = 3) out;\n"
        "void main() {\n"
        "    if (gl_InvocationID == 0) { gl_TessLevelInner[0] = 3.0; gl_TessLevelOuter = float[4](2.0, 2.0, 2.0, 2.0); }\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "}\n";
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(location = 0) flat out vec4 v[2][2];\n"
        "void main() {\n"
        "    for (int a = 0; a < 2; a++)\n"
        "        for (int b = 0; b < 2; b++)\n"
        "            v[a][b] = vec4(float(a * 2 + b + 1) / 10.0);\n"
        "    gl_Position = vec4(gl_TessCoord.xy * 2.0 - 1.0, 0.0, 1.0);\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "layout(location = 0) flat in vec4 v[2][2];\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(v[1][1].x, v[0][0].y, v[1][0].z, 1.0); }\n";
    GLuint prog = mk_prog4(vs, tcs, tes, fs, "aoa_tes_out");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    glViewport(0, 0, 32, 32);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    /* v[1][1].x = 4/10 -> 102 ; v[0][0].y = 1/10 -> 26 ; v[1][0].z = 3/10 -> 77 */
    const GLubyte expect[4] = { 102, 26, 77, 255 };
    expect_px("aoa_tes_out", expect, 3);
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

/* separate attribs: each aoa varying into its own buffer */
TEST(aoa_xfb_separate)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(location = 0) flat out vec4 a[1][1];\n"
        "layout(location = 1) flat out vec4 b[1][1];\n"
        "void main() {\n"
        "    a[0][0] = vec4(1.0, 2.0, 3.0, 4.0);\n"
        "    b[0][0] = vec4(0.5, 1.5, 2.5, 3.5);\n"
        "    gl_Position = vec4(-1 + 2.0 * float(gl_VertexID & 1), -1 + 2.0 * float(gl_VertexID >> 1), 0.0, 1.0);\n"
        "}\n";
    GLuint prog = mk_prog(vs, "#version 460 core\nlayout(location=0) out vec4 c; void main() { c = vec4(1); }", "aoa_xfb_separate");
    GLint lc = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    const char *names[2] = { "a", "b" };
    glTransformFeedbackVaryings(prog, 2, names, GL_SEPARATE_ATTRIBS);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &lc);
    if (lc != 1)
    {
        char log[4096]; GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "  [aoa_xfb_separate] RELINK FAILED:\n%s\n", log);
    }
    EXPECT_EQ_INT(lc, 1);
    if (lc != 1) { glDeleteProgram(prog); return; }
    glUseProgram(prog);
    GLuint tfo, buf[2];
    glGenTransformFeedbacks(1, &tfo);
    glGenBuffers(2, buf);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    for (int i = 0; i < 2; i++)
    {
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, buf[i]);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    }
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    float out[2][64];
    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf[i]);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out[0], out[i]);
    }
    for (int e = 0; e < 1; e++)
        for (int c = 0; c < 4; c++)
        {
            static const float exa[4] = { 1.f, 2.f, 3.f, 4.f }, exb[4] = { 0.5f, 1.5f, 2.5f, 3.5f };
            mgl_test_checks++;
            if (fabsf(out[0][c] - exa[c]) > 0.01f)
                MGL_FAIL("aoa_xfb_separate: buf0 comp %d: %.2f, expected %.2f", c, out[0][c], exa[c]);
            mgl_test_checks++;
            if (fabsf(out[1][c] - exb[c]) > 0.01f)
                MGL_FAIL("aoa_xfb_separate: buf1 comp %d: %.2f, expected %.2f", c, out[1][c], exb[c]);
        }
    glDeleteBuffers(2, buf);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(aoa_vec4);
    RUN(aoa_mat);
    RUN(aoa_struct);
    RUN(aoa_xfb_vec);
    RUN(aoa_xfb_struct);
    RUN(aoa_xfb_api);
    RUN(aoa_gs_one_level_struct);
    RUN(aoa_gs_in_vec);
    RUN(aoa_gs_in_struct);
    RUN(aoa_gs_out);
    RUN(aoa_gs_xfb);
    RUN(aoa_tes_out);
    RUN(aoa_xfb_separate);
    return mgl_test_summary();
}
