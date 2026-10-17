#include "mgl_test.h"
#include <stdbool.h>

/* ------------------------------------------------------------------ helpers */

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        GLsizei len = 0;
        log[0] = 0;
        glGetShaderInfoLog(s, sizeof log, &len, log);
        printf("compile %x failed:\n%s\n", type, log);
    }
    return s;
}

static GLuint program(const char *vs, const char *gs, const char *fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    if (gs) glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        GLsizei len = 0;
        log[0] = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        printf("link failed:\n%s\n", log);
    }
    return p;
}

static const char *plain_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec2 v_pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_pos = pos; }\n";

static const char *plain_fs =
    "#version 450 core\n"
    "layout(location=0) in vec2 v_pos;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";

static GLuint vao, vbo;
static const float quad[] = { -1.0f, -1.0f,   1.0f, -1.0f,   -1.0f, 1.0f,   1.0f, 1.0f };

static void setup(GLuint n, const float *v)
{
    if (!vao)
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static void clear_black(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
}

static void px(int x, int y, GLubyte *out)
{
    glFinish();
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}

static bool is_green(GLubyte *p) { return p[0] < 32 && p[1] > 200 && p[2] < 32; }
static bool is_black(GLubyte *p) { return p[0] < 8 && p[1] < 8 && p[2] < 8; }
static bool is_yellow(GLubyte *p) { return p[0] > 200 && p[1] > 200 && p[2] < 32; }

/* count pixels in a 7x7 block around (x,y) matching `match` */
static int count_block(int x, int y, bool (*match)(GLubyte *))
{
    int n = 0;
    for (int dy = -3; dy <= 3; dy++)
        for (int dx = -3; dx <= 3; dx++)
        {
            GLubyte c[4];
            px(x + dx, y + dy, c);
            if (match(c)) n++;
        }
    return n;
}

/* ------------------------------------------------------------------ tests */

TEST(control_no_geometry_shader)
{
    GLuint p = program(plain_vs, NULL, plain_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    glUseProgram(p);
    clear_black();
    setup(8, quad);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(20, 20, c);
    printf("control pixel = %d,%d,%d,%d\n", c[0], c[1], c[2], c[3]);
    EXPECT_TRUE(is_green(c));
    glDeleteProgram(p);
}

TEST(control_varying_no_gs)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out float v_shade;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_shade = 0.5; }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float v_shade;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, v_shade, 0.0, 1.0); }\n";
    GLuint p = program(vs, NULL, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    glUseProgram(p);
    clear_black();
    setup(8, quad);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(60, 60, c);
    EXPECT_TRUE(c[1] > 100 && c[1] < 155);
    glDeleteProgram(p);
}

TEST(passthrough_triangles)
{
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
    GLuint p = program(plain_vs, gs, plain_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLint v = -1;
    glGetProgramiv(p, GL_GEOMETRY_VERTICES_OUT, &v); EXPECT_EQ_INT(v, 3);
    glGetProgramiv(p, GL_GEOMETRY_INPUT_TYPE, &v); EXPECT_EQ_INT(v, GL_TRIANGLES);
    glGetProgramiv(p, GL_GEOMETRY_OUTPUT_TYPE, &v); EXPECT_EQ_INT(v, GL_TRIANGLE_STRIP);
    glGetProgramiv(p, GL_GEOMETRY_SHADER_INVOCATIONS, &v); EXPECT_EQ_INT(v, 1);

    glUseProgram(p);
    clear_black();
    setup(8, quad);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    EXPECT_TRUE(count_block(60, 60, is_green) > 20);
    EXPECT_TRUE(count_block(260, 260, is_black) > 20);
    glDeleteProgram(p);
}

TEST(point_to_quad_amplification)
{
    /* every input point becomes a full-screen quad */
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location=0) out vec2 g_pos;\n"
        "void main(){\n"
        "  g_pos = vec2(0.0);\n"
        "  gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 1.0, -1.0, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-1.0,  1.0, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 1.0,  1.0, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, plain_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    glUseProgram(p);
    static const float one[2] = { 0.0f, 0.0f };
    setup(2, one);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLubyte c[4];
    px(160, 160, c);
    EXPECT_TRUE(is_green(c));
    px(10, 10, c);
    EXPECT_TRUE(is_green(c));
    glDeleteProgram(p);
}

TEST(primitive_id_and_invocations)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles, invocations = 2) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) out float shade;\n"
        "void main(){\n"
        "  if (gl_PrimitiveIDIn != gl_InvocationID) return;\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    shade = float(gl_InvocationID) * 0.5 + 0.25;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float shade;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, shade, 0.0, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLint inv = 0;
    glGetProgramiv(p, GL_GEOMETRY_SHADER_INVOCATIONS, &inv);
    EXPECT_EQ_INT(inv, 2);

    glUseProgram(p);
    clear_black();
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    /* invocation 0 writes shade = 0.25 (== 64), invocation 1 emits nothing;
     * the triangle is (-1,-1),(1,-1),(0,1) so (100,150) is inside it */
    GLubyte c[4] = {0, 0, 0, 0};
    px(100, 150, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 40 && c[1] < 100 && c[2] < 32);
    glDeleteProgram(p);
}

TEST(lines_adjacency_input)
{
    /* adjacency vertices are visible in gl_in[] but only the middle edge is drawn */
    const char *gs =
        "#version 450 core\n"
        "layout(lines_adjacency) in;\n"
        "layout(line_strip, max_vertices = 4) out;\n"
        "void main(){\n"
        "  gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[2].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[0].gl_Position * 0.5; EmitVertex();\n"
        "  gl_Position = gl_in[3].gl_Position * 0.5; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0, 1.0, 0.0, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    glUseProgram(p);
    clear_black();
    /* horizontal line from -0.5 to 0.5, adjacency at +/-0.9 */
    static const float l[8] = { -0.9f, 0.0f,  -0.5f, 0.0f,   0.5f, 0.0f,   0.9f, 0.0f };
    setup(8, l);
    glDrawArrays(GL_LINES_ADJACENCY, 0, 4);
    EXPECT_NO_GL_ERROR();

    /* the middle of the row must be covered by the emitted line strip */
    GLubyte c[4];
    px(100, 100, c);
    EXPECT_TRUE(count_block(100, 100, is_yellow) > 3);
    glDeleteProgram(p);
}

TEST(simple_varying_from_gs)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) out float shade;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; shade = 0.5; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float shade;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.2, shade, 0.0, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    clear_black();
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4] = {0,0,0,0};
    px(100, 150, c);
    EXPECT_TRUE(c[1] > 100 && c[1] < 150);
    glDeleteProgram(p);
}

TEST(draw_mode_mismatch)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "void main(){ gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    glUseProgram(p);
    setup(6, quad);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

TEST(link_error_no_varying)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "layout(location=3) in vec4 missing[];\n"
        "void main(){ gl_Position = missing[0]; EmitVertex(); EndPrimitive(); }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 1;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(control_no_geometry_shader);
    RUN(control_varying_no_gs);
    RUN(passthrough_triangles);
    RUN(point_to_quad_amplification);
    RUN(primitive_id_and_invocations);
    RUN(lines_adjacency_input);
    RUN(simple_varying_from_gs);
    RUN(draw_mode_mismatch);
    RUN(link_error_no_varying);
    return mgl_test_summary();
}
