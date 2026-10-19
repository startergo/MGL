#include "mgl_test.h"
/* from MGL/src/geometry.c: trace the buffer tracking of a TF buffer at a CPU read,
 * looked up by buffer name (test-facing diagnostics, see mgl_geom_diag.h) */
#include "mgl_geom_diag.h"
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
static bool is_greenish(GLubyte *p) { return p[0] < 32 && p[1] > 200; }
static bool is_green_mid(GLubyte *p) { return p[0] < 32 && p[1] > 100 && p[1] < 200; }

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


/* ------------------------------------------------------------------ more coverage */

static const char *tri_fs =
    "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";

TEST(end_primitive_splitting)
{
    /* two separate strips from one input triangle */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 6) out;\n"
        "void main(){\n"
        "  gl_Position = vec4(-0.9, -0.4, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.1, -0.4, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.9,  0.4, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_Position = vec4( 0.1, -0.4, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.9, -0.4, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4( 0.9,  0.4, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, tri_fs);
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

    EXPECT_TRUE(count_block(30, 100, is_green) > 20);
    EXPECT_TRUE(count_block(170, 100, is_green) > 20);
    EXPECT_TRUE(count_block(100, 100, is_green) == 0);
    glDeleteProgram(p);
}

TEST(amplification_max_vertices)
{
    /* one input point becomes three side-by-side triangles */
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 9) out;\n"
        "layout(location=0) out float col;\n"
        "void main(){\n"
        "  for (int k = 0; k < 3; k++) {\n"
        "    float x = -0.9 + float(k) * 0.7;\n"
        "    gl_Position = vec4(x, -0.4, 0.0, 1.0); col = float(k) * 0.5; EmitVertex();\n"
        "    gl_Position = vec4(x + 0.6, -0.4, 0.0, 1.0); col = float(k) * 0.5; EmitVertex();\n"
        "    gl_Position = vec4(x, 0.4, 0.0, 1.0); col = float(k) * 0.5; EmitVertex();\n"
        "    EndPrimitive();\n"
        "  }\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float col;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 1.0, col, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLint mv = 0;
    glGetProgramiv(p, GL_GEOMETRY_VERTICES_OUT, &mv);
    EXPECT_EQ_INT(mv, 9);

    glUseProgram(p);
    clear_black();
    static const float one[2] = { 0.0f, 0.0f };
    setup(2, one);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    /* each amplified triangle is (x,-0.4),(x+0.6,-0.4),(x,0.4): at the vertical middle it
     * spans [x, x+0.3] */
    EXPECT_TRUE(count_block(20, 100, is_greenish) > 20);
    EXPECT_TRUE(count_block(95, 100, is_greenish) > 20);
    EXPECT_TRUE(count_block(165, 100, is_greenish) > 20);
    EXPECT_TRUE(count_block(60, 100, is_black) > 20);
    glDeleteProgram(p);
}

TEST(triangles_adjacency_input)
{
    /* the even vertices form the triangle, the odd ones are adjacent */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles_adjacency) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) out float adj;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i * 2].gl_Position;\n"
        "    adj = gl_in[i * 2 + 1].gl_Position.x + 0.5;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float adj;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(adj, adj, adj, 1.0); }\n";
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
    /* triangle (-1,-1),(1,-1),(0,1) with adjacency x = 0.2, 0.4, 0.6 */
    static const float v[12] = { -1.0f, -1.0f,   0.2f, 0.0f,
                                  1.0f, -1.0f,   0.4f, 0.0f,
                                  0.0f,  1.0f,   0.6f, 0.0f };
    setup(12, v);
    glDrawArrays(GL_TRIANGLES_ADJACENCY, 0, 6);
    EXPECT_NO_GL_ERROR();

    /* the triangle is drawn flat-shaded with the last adjacency vertex (0.6 + 0.5) */
    GLubyte c[4] = {0,0,0,0};
    px(100, 150, c);
    printf("tri adjacency pixel = %d,%d,%d\n", c[0], c[1], c[2]);
    EXPECT_TRUE(c[0] > 200 && c[1] > 200);
    glDeleteProgram(p);
}

TEST(indexed_and_instanced)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) out float inst;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; inst = v_pos[0].y; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float inst;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, inst, 0.0, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ v_pos = vec2(0.0, 0.5); gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint vao2 = 0, vbo2 = 0, ebo = 0;
    static const float verts[8] = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f };
    static const GLushort idx[3] = { 0, 1, 2 };
    static const GLushort idx2[3] = { 1, 2, 0 };

    setup(8, verts);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glBindVertexArray(vao);

    glUseProgram(p);

    clear_black();
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(count_block(60, 120, is_green_mid) > 10);

    /* base vertex */
    clear_black();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx2, idx2, GL_STATIC_DRAW);
    glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL, 1);
    EXPECT_NO_GL_ERROR();

    /* instanced: every instance writes the same triangle */
    clear_black();
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(count_block(60, 120, is_green_mid) > 10);

    glDeleteBuffers(1, &ebo);
    glDeleteProgram(p);
}

TEST(indirect_draws)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, tri_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glUseProgram(p);

    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, buf);
    {
        const GLuint args[4] = { 3, 1, 0, 0 };     /* count, primcount, first, baseinstance */
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof args, args, GL_STATIC_DRAW);
    }
    clear_black();
    glDrawArraysIndirect(GL_TRIANGLES, NULL);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(count_block(100, 150, is_green) > 10);

    /* indexed indirect */
    GLuint ebo = 0;
    static const GLushort idx[3] = { 0, 1, 2 };
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    {
        const GLuint args[5] = { 3, 1, 0, 0, 0 };  /* count, inst, first, basevertex, baseinst */
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof args, args, GL_STATIC_DRAW);
    }
    clear_black();
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_SHORT, NULL);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(count_block(100, 150, is_green) > 10);

    glDeleteBuffers(1, &buf);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(p);
}

TEST(layered_rendering)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main(){\n"
        "  gl_Layer = 1;\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, tri_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint tex = 0, fbo = 0, rfbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 200, 200, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glUseProgram(p);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    GLubyte c[4] = {0,0,0,0};
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, 0);
    glFinish();
    glReadPixels(100, 150, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    printf("layer0 = %d,%d,%d  ", c[0], c[1], c[2]);
    EXPECT_TRUE(is_black(c));
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, 1);
    glFinish();
    glReadPixels(100, 150, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    printf("layer1 = %d,%d,%d\n", c[0], c[1], c[2]);
    EXPECT_TRUE(is_green(c));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

TEST(viewport_index)
{
    /* two viewports side by side; the geometry picks the right one */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main(){\n"
        "  gl_ViewportIndex = 1;\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, tri_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    glUseProgram(p);
    glViewportIndexedf(0, 0.0f, 0.0f, 100.0f, 200.0f);
    glViewportIndexedf(1, 100.0f, 0.0f, 100.0f, 200.0f);
    clear_black();
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    /* viewport 1 covers x in [100,200] */
    EXPECT_TRUE(count_block(150, 100, is_green) > 20);
    EXPECT_TRUE(count_block(50, 100, is_green) == 0);
    glViewport(0, 0, 200, 200);
    glDeleteProgram(p);
}

TEST(separable_pipeline_with_gs)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint pv = glCreateProgram(), pg = glCreateProgram(), pf = glCreateProgram(), pipe = 0;
    GLint ok = 0;
    GLuint sv = compile(GL_VERTEX_SHADER, vs);
    GLuint sg = compile(GL_GEOMETRY_SHADER, gs);
    GLuint sf = compile(GL_FRAGMENT_SHADER, tri_fs);

    glProgramParameteri(pv, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glProgramParameteri(pg, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glProgramParameteri(pf, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(pv, sv);
    glAttachShader(pg, sg);
    glAttachShader(pf, sf);
    glLinkProgram(pv);
    glGetProgramiv(pv, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_TRUE);
    glLinkProgram(pg);
    glGetProgramiv(pg, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glLinkProgram(pf);
    glGetProgramiv(pf, GL_LINK_STATUS, &ok); EXPECT_EQ_INT(ok, GL_TRUE);

    glUseProgram(0);
    glGenProgramPipelines(1, &pipe);
    glBindProgramPipeline(pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, pv);
    glUseProgramStages(pipe, GL_GEOMETRY_SHADER_BIT, pg);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, pf);
    EXPECT_NO_GL_ERROR();

    clear_black();
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(count_block(100, 150, is_green) > 10);

    /* the geometry draw must not corrupt the vertex program it shares a Shader with: draw
     * pv alone (no geometry stage) through a second pipeline — its own vertex function has
     * to feed the fragment stage again */
    {
        GLuint plain = 0;
        glGenProgramPipelines(1, &plain);
        glBindProgramPipeline(plain);
        glUseProgramStages(plain, GL_VERTEX_SHADER_BIT, pv);
        glUseProgramStages(plain, GL_FRAGMENT_SHADER_BIT, pf);
        EXPECT_NO_GL_ERROR();
        clear_black();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        EXPECT_TRUE(count_block(100, 100, is_green) > 10);
        /* the pixel is outside the triangle: the geometry stage must not be leaking into
         * the plain vertex program's draws */
        EXPECT_TRUE(count_block(5, 190, is_black) > 10);

        /* and the geometry pipeline still draws afterwards: its composite program is rebuilt
         * from the stage programs on every scope entry, so the pass-through override has to
         * win again rather than the composite's own vertex stage */
        glBindProgramPipeline(pipe);
        clear_black();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        EXPECT_TRUE(count_block(100, 150, is_green) > 10);

        glDeleteProgramPipelines(1, &plain);
    }

    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(pv); glDeleteProgram(pg); glDeleteProgram(pf);
    glDeleteShader(sv); glDeleteShader(sg); glDeleteShader(sf);
}

TEST(primitive_queries)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, tri_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    glUseProgram(p);
    setup(8, quad);

    GLuint q1 = 0, q2 = 0, q3 = 0;
    GLint v1 = -1, v2 = -1, v3 = -1;
    glGenQueries(1, &q1);
    glGenQueries(1, &q2);
    glGenQueries(1, &q3);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q1);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q2);
    glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, q3);
    clear_black();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
    EXPECT_NO_GL_ERROR();

    glGetQueryObjectiv(q1, GL_QUERY_RESULT, &v1);
    glGetQueryObjectiv(q2, GL_QUERY_RESULT, &v2);
    glGetQueryObjectiv(q3, GL_QUERY_RESULT, &v3);
    printf("primitives generated=%d emitted=%d invocations=%d\n", v1, v2, v3);
    EXPECT_EQ_INT(v1, 2);
    EXPECT_EQ_INT(v2, 2);
    EXPECT_EQ_INT(v3, 2);

    glDeleteQueries(1, &q1);
    glDeleteQueries(1, &q2);
    glDeleteQueries(1, &q3);
    glDeleteProgram(p);
}


/* ------------------------------------------------------------------ transform feedback of the geometry output */

TEST(xfb_of_geometry_output)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) out vec2 g_pos;\n"
        "layout(location=1) out vec4 g_col;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    g_pos = v_pos[i] * 2.0;\n"
        "    g_col = vec4(0.25, 0.5, 0.75, 1.0);\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_pos = pos; }\n";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));

    static const char *vars[3] = { "g_pos", "g_col", NULL };
    glTransformFeedbackVaryings(p, 2, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint q = 0, tbuf = 0;
    glGenQueries(1, &q);
    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);

    glUseProgram(p);
    {
        /* nine vertices: three independent triangles */
        static const float nine[18] = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
                                        -0.8f, -0.6f, 0.8f, -0.6f, -0.8f, 0.4f,
                                         0.6f, -0.4f, 0.9f,  0.2f, 0.6f, 0.4f };
        setup(18, nine);
    }
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glDrawArrays(GL_TRIANGLES, 0, 6);          /* two input triangles */
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLint written = -1;
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
    printf("xfb written = %d\n", written);
    EXPECT_EQ_INT(written, 2);

    struct { float x, y; float r, g, b, a; } out[8];
    /* read back with no glFinish: the read path must wait for the pack itself */
    memset(out, 0, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    printf("  immediate read (no glFinish): %5.2f %5.2f %5.2f %5.2f\n", out[0], out[1], out[6], out[7]);
    for (int i = 0; i < 6; i++)
        printf("  v%d = %f %f | %f %f %f\n", i, out[i].x, out[i].y, out[i].r, out[i].g, out[i].b);

    /* the first input triangle is (-1,-1),(1,-1),(-1,1): the geometry doubles it */
    EXPECT_NEAR(out[0].x, -2.0, 1e-5);
    EXPECT_NEAR(out[0].y, -2.0, 1e-5);
    EXPECT_NEAR(out[1].x,  2.0, 1e-5);
    EXPECT_NEAR(out[2].x, -2.0, 1e-5);
    EXPECT_NEAR(out[3].r, 0.25, 1e-5);
    EXPECT_NEAR(out[3].g, 0.5, 1e-5);
    EXPECT_NEAR(out[3].b, 0.75, 1e-5);

    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &tbuf);
    glDeleteProgram(p);
}

TEST(xfb_streams)
{
    /* stream 0 is captured; the vertices emitted to stream 1 are not */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 6) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) out vec2 g_pos;\n"
        "layout(stream=1, location=1) out vec2 g_other;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = gl_in[i].gl_Position;\n"
        "    g_pos = v_pos[i];\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "  g_other = vec2(9.0);\n"
        "  EmitStreamVertex(1);\n"
        "  EndStreamPrimitive(1);\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_pos = pos; }\n";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    GLuint gsh = compile(GL_GEOMETRY_SHADER, gs);
    GLint cok = 0;
    glGetShaderiv(gsh, GL_COMPILE_STATUS, &cok);
    if (!cok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetShaderInfoLog(gsh, sizeof l, &n, l);
        printf("gs compile log: %s\n", l);
        return;
    }
    glAttachShader(p, gsh);
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));

    static const char *vars[2] = { "g_pos", NULL };
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof l, &n, l);
        printf("stream link log: %s\n", l);
        return;
    }

    GLuint tbuf = 0;
    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    {
        static const float zeros[16] = { 0 };
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof zeros, zeros, GL_STREAM_READ);
        /* the allocation may recycle memory another buffer used: clear it on the GPU so the
         * "nothing else was captured" expectations below are meaningful */
    }
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);

    glUseProgram(p);
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[16];
    memset(out, 0, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int i = 0; i < 6; i++) printf("  s%d = %f %f\n", i, out[i * 2], out[i * 2 + 1]);

    /* stream 0 got three vertices; the stream 1 vertex must not appear in the capture */
    EXPECT_NEAR(out[0], -1.0, 1e-5);
    EXPECT_NEAR(out[1], -1.0, 1e-5);
    EXPECT_NEAR(out[4], 0.0, 1e-5);
    EXPECT_NEAR(out[5], 1.0, 1e-5);

    /* only the three stream-0 vertices are captured: the query is authoritative (the
     * buffer's tail may recycle another buffer's memory, so it cannot be asserted on) */
    {
        GLuint q = 0;
        GLint written = -1;
        glGenQueries(1, &q);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
        printf("  stream-0 primitives written = %d\n", written);
        EXPECT_EQ_INT(written, 1);
        glDeleteQueries(1, &q);
    }

    glDeleteBuffers(1, &tbuf);
    glDeleteProgram(p);
}


/* ---- geometry behind a tessellation stage ----------------------------------------
 * The tess domain's TES-output capture feeds the geometry kernel ("from tess"): there is no
 * vertex capture pass, the import kernel repacks the captured records into this program's
 * input records and the tessellator's index list becomes the primitive slots.  A filled
 * triangle domain (non-indexed and indexed), isolines and point_mode are covered.
 *
 * NOTE: every TES here really uses its input varyings.  SPIRV-Cross drops unused ones from
 * the TES control-point struct, and the emulated TES reads the TCS output with that struct:
 * a TES that ignores a varying the TCS writes reads shifted control points (tess domain
 * limitation, see the round-21 report). */

static void setup4(GLuint n, const float *v)
{
    if (!vao)
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

static const char *tess_vs =
    "#version 450 core\n"
    "layout(location=0) in vec4 pos;\n"
    "layout(location=0) out vec4 v_uv;\n"
    "void main(){ gl_Position = vec4(pos.xy, 0.0, 1.0); v_uv = pos; }\n";

static const char *tri_tcs =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_uv[];\n"
    "layout(location=0) out vec4 tc_uv[];\n"
    "void main(){\n"
    "  tc_uv[gl_InvocationID] = v_uv[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "  }\n"
    "}\n";

static const char *tri_tes =
    "#version 450 core\n"
    "layout(triangles) in;\n"
    "layout(location=0) in vec4 tc_uv[];\n"
    "layout(location=0) out float shade;\n"
    "void main(){\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position"
    " + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  shade = 0.75 * tc_uv[0].w;\n"
    "}\n";

static const char *tess_fs =
    "#version 450 core\n"
    "layout(location=0) in float gshade;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(0.0, gshade, 0.0, 1.0); }\n";

static GLuint tess_gs_prog(const char *tcs, const char *tes, const char *gs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, tess_vs));
    glAttachShader(p, compile(GL_TESS_CONTROL_SHADER, tcs));
    glAttachShader(p, compile(GL_TESS_EVALUATION_SHADER, tes));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, tess_fs));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        GLsizei len = 0;
        glGetProgramInfoLog(p, sizeof log, &len, log);
        printf("tess+gs link log: %s\n", log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    return ok ? p : 0;
}

static const float tess_tri[12] =
    { -0.5f, -0.5f, 0.0f, 1.0f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.5f, 0.0f, 1.0f };

TEST(tessellation_triangles_into_gs)
{
    /* the geometry stage shrinks every tessellated triangle: the centre pixel is inside the
     * shrunken triangle, the left edge only inside the unshrunk one */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in float shade[];\n"
        "layout(location=0) out float gshade;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = vec4(gl_in[i].gl_Position.xy * 0.5, 0.0, 1.0);"
        " gshade = shade[i]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint p = tess_gs_prog(tri_tcs, tri_tes, gs);
    if (!p) return;
    glUseProgram(p);
    clear_black();
    setup4(12, tess_tri);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(100, 100, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100 && c[2] < 32);
    px(20, 100, c);
    EXPECT_TRUE(c[0] == 0 && c[1] == 0);
    glDeleteProgram(p);
}

TEST(tessellation_indexed_into_gs)
{
    /* the same tessellated patch through an index buffer (a cyclic permutation keeps the
     * winding): the primitive slots come from the tessellator's index list either way */
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in float shade[];\n"
        "layout(location=0) out float gshade;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = vec4(gl_in[i].gl_Position.xy * 0.75, 0.0, 1.0);"
        " gshade = shade[i]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint p = tess_gs_prog(tri_tcs, tri_tes, gs);
    if (!p) return;
    glUseProgram(p);
    clear_black();
    setup4(12, tess_tri);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    GLuint ibo = 0;
    static const GLuint idx[3] = { 2, 0, 1 };
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glDrawElements(GL_PATCHES, 3, GL_UNSIGNED_INT, 0);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(100, 100, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100 && c[2] < 32);
    px(15, 100, c);
    EXPECT_TRUE(c[0] == 0 && c[1] == 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &ibo);
    glDeleteProgram(p);
}

TEST(tessellation_isolines_into_gs)
{
    /* isolines produce line primitives: the geometry stage thickens the single tessellated
     * isoline into a quad (triangle_strip) */
    const char *tcs =
        "#version 450 core\n"
        "layout(vertices=2) out;\n"
        "layout(location=0) in vec4 v_uv[];\n"
        "layout(location=0) out vec4 tc_uv[];\n"
        "void main(){\n"
        "  tc_uv[gl_InvocationID] = v_uv[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
        "  }\n"
        "}\n";
    const char *tes =
        "#version 450 core\n"
        "layout(isolines) in;\n"
        "layout(location=0) in vec4 tc_uv[];\n"
        "layout(location=0) out float shade;\n"
        "void main(){\n"
        "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
        "  shade = 0.75 * tc_uv[0].w;\n"
        "}\n";
    const char *gs =
        "#version 450 core\n"
        "layout(lines) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location=0) in float shade[];\n"
        "layout(location=0) out float gshade;\n"
        "void main(){\n"
        "  vec2 a = gl_in[0].gl_Position.xy, b = gl_in[1].gl_Position.xy;\n"
        "  vec2 d = normalize(b - a);\n"
        "  vec2 n = vec2(-d.y, d.x) * 0.05;\n"
        "  gshade = shade[0];\n"
        "  gl_Position = vec4(a - n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(b - n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(a + n, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(b + n, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    static const float line[8] = { -0.5f, 0.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 1.0f };
    GLuint p = tess_gs_prog(tcs, tes, gs);
    if (!p) return;
    glUseProgram(p);
    clear_black();
    setup4(8, line);
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glDrawArrays(GL_PATCHES, 0, 2);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(100, 100, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100 && c[2] < 32);
    px(100, 175, c);
    EXPECT_TRUE(c[0] == 0 && c[1] == 0);
    glDeleteProgram(p);
}

TEST(tessellation_point_mode_into_gs)
{
    /* point_mode tessellation emits points: the geometry stage turns each into a small
     * triangle, so a point at the top control point covers a pixel there */
    const char *tes =
        "#version 450 core\n"
        "layout(triangles, point_mode) in;\n"
        "layout(location=0) in vec4 tc_uv[];\n"
        "layout(location=0) out float shade;\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position"
        " + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  shade = 0.75 * tc_uv[0].w;\n"
        "}\n";
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in float shade[];\n"
        "layout(location=0) out float gshade;\n"
        "void main(){\n"
        "  vec2 p = gl_in[0].gl_Position.xy;\n"
        "  gshade = shade[0];\n"
        "  gl_Position = vec4(p + vec2(-0.1, -0.1), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.1, -0.1), 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(p + vec2( 0.0,  0.1), 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint p = tess_gs_prog(tri_tcs, tes, gs);
    if (!p) return;
    glUseProgram(p);
    clear_black();
    setup4(12, tess_tri);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(100, 150, c);                 /* the top control point (0, 0.5) */
    EXPECT_TRUE(c[0] < 32 && c[1] > 100 && c[2] < 32);
    px(100, 25, c);                  /* far below the bottom edge */
    EXPECT_TRUE(c[0] == 0 && c[1] == 0);
    glDeleteProgram(p);
}

/* primitive restart with a geometry shader: the restarted runs are separate primitives and
 * the capture pass never sees the restart index */
TEST(primitive_restart_with_gs)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) out float prim;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; prim = float(gl_PrimitiveIDIn); EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float prim;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(prim / 2.0, 0.0, 0.0, 1.0); }\n";
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
    /* two triangles separated by a restart index */
    static const float v[12] = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
                                 -1.0f, -0.2f, 1.0f, -0.2f, -1.0f, 0.8f };
    static const GLuint idx[7] = { 0, 1, 2, 0xffffffffu, 3, 4, 5 };
    GLuint vao2 = 0, vbo2 = 0, ebo = 0;
    glGenVertexArrays(1, &vao2);
    glGenBuffers(1, &vbo2);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao2);
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xffffffffu);
    glDrawElements(GL_TRIANGLES, 7, GL_UNSIGNED_INT, NULL);
    EXPECT_NO_GL_ERROR();
    glDisable(GL_PRIMITIVE_RESTART);

    /* primitive 0 shades red = 0, primitive 1 = 0.5 */
    int scan = 0;
    for (int yy = 5; yy < 195 && scan < 3; yy += 5)
        for (int xx = 5; xx < 195 && scan < 3; xx += 5)
        {
            GLubyte c[4] = {0,0,0,0};
            px(xx, yy, c);
            if (c[0] || c[1] || c[2]) { printf("  (%d,%d) = %d,%d,%d\n", xx, yy, c[0], c[1], c[2]); scan++; }
        }
    EXPECT_TRUE(scan >= 2);

    glBindVertexArray(vao);
    glDeleteBuffers(1, &vbo2);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao2);
    glDeleteProgram(p);
}


/* several draws in one transform feedback block, then pause / resume: the GPU pack keeps the
 * write offsets and counters on the GPU and only folds them in at the points where the
 * application observes them.  The captured bytes must be identical to a single block that
 * saw the same draws. */
TEST(xfb_gpu_pack_multiple_draws_and_pause)
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
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_pos = pos; }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    static const char *vars[2] = { "g_pos", NULL };
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint qa = 0, qb = 0, tbuf = 0, ref = 0;
    float out[64], expect[64];

    glGenQueries(1, &qa);
    glGenQueries(1, &qb);
    glGenBuffers(1, &tbuf);
    glGenBuffers(1, &ref);

    glUseProgram(p);
    {
        /* nine vertices: three independent triangles */
        static const float nine[18] = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
                                        -0.8f, -0.6f, 0.8f, -0.6f, -0.8f, 0.4f,
                                         0.6f, -0.4f, 0.9f,  0.2f, 0.6f, 0.4f };
        setup(18, nine);
    }
    glEnable(GL_RASTERIZER_DISCARD);

    /* reference: one block, three draws, no pause */
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, ref);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof expect, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, ref);
    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qa);
    glBeginQuery(GL_PRIMITIVES_GENERATED, qb);
    for (int d = 0; d < 3; d++)
        glDrawArrays(GL_TRIANGLES, (GLint)(d * 3), 3);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();
    memset(expect, 0, sizeof expect);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, ref);
    if (getenv("MGL_GEOM_XFB_TRACE")) mgl_geom_xfb_log_read_name(mgl_geom_ctx(), ref, "reference");
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof expect, expect);
    {
        GLint wa = -1, ga = -1;
        glGetQueryObjectiv(qa, GL_QUERY_RESULT, &wa);
        glGetQueryObjectiv(qb, GL_QUERY_RESULT, &ga);
        printf("reference: written=%d generated=%d\n", wa, ga);
        EXPECT_EQ_INT(wa, 3);
        EXPECT_EQ_INT(ga, 3);
    }

    /* the same three draws, split by a pause / resume and interleaved with query reads */
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof out, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);
    memset(out, 0, sizeof out);

    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qa);
    glBeginQuery(GL_PRIMITIVES_GENERATED, qb);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDrawArrays(GL_TRIANGLES, 3, 3);
    glPauseTransformFeedback();
    {
        /* while paused and the query still active, the counters are folded in on demand */
        GLint avail = -1, w = -1;
        glGetQueryObjectiv(qa, GL_QUERY_RESULT_AVAILABLE, &avail);
        glGetQueryObjectiv(qa, GL_QUERY_RESULT, &w);
        printf("paused: available=%d written so far=%d\n", avail, w);
    }
    glResumeTransformFeedback();
    glDrawArrays(GL_TRIANGLES, 6, 3);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndTransformFeedback();

    memset(out, 0, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);

    {
        GLint wa = -1, ga = -1;
        glGetQueryObjectiv(qa, GL_QUERY_RESULT, &wa);
        glGetQueryObjectiv(qb, GL_QUERY_RESULT, &ga);
        printf("paused block: written=%d generated=%d\n", wa, ga);
        EXPECT_EQ_INT(wa, 3);
        EXPECT_EQ_INT(ga, 3);
    }

    printf("  ref:");
    for (int i = 0; i < 18; i++) printf(" %5.2f", expect[i]);
    printf("\n  got:");
    for (int i = 0; i < 18; i++) printf(" %5.2f", out[i]);
    printf("\n");
    for (int i = 0; i < 18; i++)
    {
        if (i < 6 || out[i] != expect[i])
            printf("  [%02d] got %8.3f want %8.3f%s\n", i, out[i], expect[i],
                   out[i] == expect[i] ? "" : "   <-- differs");
        EXPECT_NEAR(out[i], expect[i], 1e-6);
    }

    glDisable(GL_RASTERIZER_DISCARD);
    glDeleteQueries(1, &qa);
    glDeleteQueries(1, &qb);
    glDeleteBuffers(1, &tbuf);
    glDeleteBuffers(1, &ref);
    glDeleteProgram(p);
}

/* ---- arrays of blocks with xfb_offset, vertex stage only ---- */
TEST(xfb_arrays_of_blocks_xfb_offset)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(xfb_buffer = 0, xfb_stride = 64) out OutBlk {\n"
        "    layout(xfb_offset = 0) vec4 a;\n"
        "    layout(xfb_offset = 16) vec4 b;\n"
        "} blk[2];\n"
        "void main(){\n"
        "  gl_Position = vec4(pos, 0.0, 1.0);\n"
        "  blk[0].a = vec4(pos, 0.0, 1.0);\n"
        "  blk[0].b = vec4(1.0, 2.0, 3.0, 4.0);\n"
        "  blk[1].a = vec4(5.0, 6.0, 7.0, 8.0);\n"
        "  blk[1].b = vec4(9.0, 10.0, 11.0, 12.0);\n"
        "}\n";
    GLuint p = program(vs, NULL, plain_fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    setup(6, tri);
    GLuint tb = 0;
    glGenBuffers(1, &tb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tb);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    {
        float out[32];
        memset(out, 0, sizeof out);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tb);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
        /* record 0: blk[0].a, blk[0].b at stride 32, then blk[1].a, blk[1].b */
        EXPECT_NEAR(out[0], -1.0f, 1e-4);
        EXPECT_NEAR(out[1], -1.0f, 1e-4);
        EXPECT_NEAR(out[4], 1.0f, 1e-4);
        EXPECT_NEAR(out[5], 2.0f, 1e-4);
        EXPECT_NEAR(out[8], 5.0f, 1e-4);
        EXPECT_NEAR(out[12], 9.0f, 1e-4);
        printf("vs blocks: %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f | %.1f %.1f %.1f %.1f\n",
               out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
               out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
    }
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
    RUN(end_primitive_splitting);
    RUN(amplification_max_vertices);
    RUN(triangles_adjacency_input);
    RUN(indexed_and_instanced);
    RUN(indirect_draws);
    RUN(layered_rendering);
    RUN(viewport_index);
    RUN(separable_pipeline_with_gs);
    RUN(primitive_queries);
    RUN(xfb_of_geometry_output);
    RUN(xfb_streams);
    RUN(xfb_arrays_of_blocks_xfb_offset);
    RUN(tessellation_triangles_into_gs);
    RUN(tessellation_indexed_into_gs);
    RUN(tessellation_isolines_into_gs);
    RUN(tessellation_point_mode_into_gs);
    RUN(primitive_restart_with_gs);
    RUN(xfb_gpu_pack_multiple_draws_and_pause);
    return mgl_test_summary();
}
