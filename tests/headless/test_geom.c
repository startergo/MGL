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

/* a private framebuffer so block tests are immune to the viewport / layer state left by
 * earlier tests */
static GLuint block_fbo(GLuint *texout, int w, int h)
{
    GLuint tex, fbo;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    *texout = tex;
    return fbo;
}

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
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* 256x256 so the (160,160) read is inside */
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 200, 200); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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
    GLint ok = 0;
    /* a geometry input the vertex stage never writes stays undefined in GL — the link
     * must succeed (KHR-GL46.geometry_shader.output.vertex_emit_at_end) */
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 200, 200); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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


/* GL captures EVERY output stream, each into the buffers its varyings name (GL 4.6
 * 13.3.2): the CTS draw_xfb_stream_test shape - a geometry shader emitting three points to
 * stream 0 and three to stream 1, captured into two buffers with gl_NextBuffer, the
 * per-stream PRIMITIVES_GENERATED / TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN queries, and
 * glDrawTransformFeedbackStream rasterizing each stream's capture. */
TEST(xfb_multi_stream_capture)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 6) out;\n"
        "layout(stream = 1) out vec4 position;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    gl_Position = vec4( 1.0, -1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    gl_Position = vec4(-1.0,  1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    position = vec4( 1.0, -1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "    position = vec4(-1.0,  1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "    position = vec4( 1.0,  1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "}\n";
    const char *vs_blank =
        "#version 450 core\nvoid main() { gl_Position = vec4(0.0); }\n";
    const char *fs =
        "#version 450 core\nlayout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0); }\n";
    const char *vs_pass =
        "#version 450 core\nlayout(location=0) in vec4 position;\n"
        "void main() { gl_Position = position; }\n";

    const char *vars[3] = { "gl_Position", "gl_NextBuffer", "position" };
    GLuint pgen = glCreateProgram();
    glAttachShader(pgen, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(pgen, compile(GL_VERTEX_SHADER, vs_blank));
    glAttachShader(pgen, compile(GL_FRAGMENT_SHADER, fs));
    glTransformFeedbackVaryings(pgen, 3, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(pgen);
    GLint ls = 0;
    glGetProgramiv(pgen, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);
    if (!ls)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(pgen, sizeof l, &n, l);
        printf("  multi-stream link log: %s\n", l);
        return;
    }

    GLuint pdraw = glCreateProgram();
    glAttachShader(pdraw, compile(GL_VERTEX_SHADER, vs_pass));
    glAttachShader(pdraw, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(pdraw);
    ls = 0;
    glGetProgramiv(pdraw, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);

    GLuint xfb = 0, bo[2], rbo = 0, fbo = 0, vao = 0;
    glGenTransformFeedbacks(1, &xfb);
    glGenBuffers(2, bo);
    glGenRenderbuffers(1, &rbo);
    glGenFramebuffers(1, &fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, xfb);
    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo[i]);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, bo[i]);
    }
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_R8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 8, 8);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    GLuint q[8];
    glGenQueries(8, q);
    glUseProgram(pgen);
    for (int s = 0; s < 4; s++)
    {
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, s, q[s]);
        glBeginQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, s, q[4 + s]);
    }
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    for (int s = 0; s < 4; s++)
    {
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, s);
        glEndQueryIndexed(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, s);
    }
    EXPECT_NO_GL_ERROR();
    for (int s = 0; s < 4; s++)
    {
        GLint gen = 0, wr = 0;
        glGetQueryObjectiv(q[s], GL_QUERY_RESULT, &gen);
        glGetQueryObjectiv(q[4 + s], GL_QUERY_RESULT, &wr);
        printf("  stream %d: generated=%d written=%d\n", s, gen, wr);
        EXPECT_EQ_INT(gen, s < 2 ? 3 : 0);
        EXPECT_EQ_INT(wr, s < 2 ? 3 : 0);
    }

    /* each buffer holds its own stream's three points, in emission order */
    for (int s = 0; s < 2; s++)
    {
        float m[12];
        memset(m, 0, sizeof m);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo[s]);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof m, m);
        printf("  stream %d:", s);
        for (int i = 0; i < 3; i++) printf(" (%.1f,%.1f)", m[i * 4], m[i * 4 + 1]);
        printf("\n");
        if (s == 0)
        {
            EXPECT_NEAR(m[0], -1.0, 1e-5); EXPECT_NEAR(m[1], -1.0, 1e-5);
            EXPECT_NEAR(m[4],  1.0, 1e-5); EXPECT_NEAR(m[5], -1.0, 1e-5);
            EXPECT_NEAR(m[8], -1.0, 1e-5); EXPECT_NEAR(m[9],  1.0, 1e-5);
        }
        else
        {
            EXPECT_NEAR(m[0],  1.0, 1e-5); EXPECT_NEAR(m[1], -1.0, 1e-5);
            EXPECT_NEAR(m[4], -1.0, 1e-5); EXPECT_NEAR(m[5],  1.0, 1e-5);
            EXPECT_NEAR(m[8],  1.0, 1e-5); EXPECT_NEAR(m[9],  1.0, 1e-5);
        }
    }

    /* each stream's capture rasterizes as the triangles its three points form */
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(pdraw);
    for (int s = 0; s < 2; s++)
    {
        glBindBuffer(GL_ARRAY_BUFFER, bo[s]);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        glEnableVertexAttribArray(0);
        glDrawTransformFeedbackStream(GL_TRIANGLES, xfb, s);
        EXPECT_NO_GL_ERROR();
    }
    {
        unsigned char pix[8 * 8 * 4];
        int dark = 0;
        glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, pix);
        for (int i = 0; i < 8 * 8; i++) if (pix[i * 4] < 200) dark++;
        printf("  coverage: %d/64 dark\n", dark);
        EXPECT_EQ_INT(dark, 0);
    }

    glDeleteQueries(8, q);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteBuffers(2, bo);
    glDeleteTransformFeedbacks(1, &xfb);
    glDeleteProgram(pgen);
    glDeleteProgram(pdraw);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/* the CTS draw_xfb_stream_instanced_test shape: stream 0 captures a quad's positions and
 * stream 1 its colors; glDrawTransformFeedbackStreamInstanced draws the capture four times
 * with per-instance matrices, the four quarter quads covering the framebuffer. */
TEST(xfb_multi_stream_instanced)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 8) out;\n"
        "layout(stream = 1) out vec4 color;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    gl_Position = vec4( 1.0, -1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    gl_Position = vec4(-1.0,  1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    gl_Position = vec4( 1.0,  1.0, 0.0, 1.0);\n"
        "    EmitStreamVertex(0);\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "    color = vec4(1.0, 1.0, 1.0, 1.0);\n"
        "    EmitStreamVertex(1);\n"
        "}\n";
    const char *vs_blank =
        "#version 450 core\nvoid main() { }\n";
    const char *fs_blank =
        "#version 450 core\nlayout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0); }\n";
    const char *vs_draw =
        "#version 450 core\n"
        "uniform MatrixBlock\n"
        "{\n"
        "    mat4 transformation_0;\n"
        "    mat4 transformation_1;\n"
        "    mat4 transformation_2;\n"
        "    mat4 transformation_3;\n"
        "};\n"
        "layout(location=0) in  vec4 position;\n"
        "layout(location=1) in  vec4 color;\n"
        "layout(location=0) out vec4 colour;\n"
        "void main()\n"
        "{\n"
        "    switch(gl_InstanceID % 4)\n"
        "    {\n"
        "    case 0: gl_Position = position * transformation_0; break;\n"
        "    case 1: gl_Position = position * transformation_1; break;\n"
        "    case 2: gl_Position = position * transformation_2; break;\n"
        "    case 3: gl_Position = position * transformation_3; break;\n"
        "    }\n"
        "    colour = color;\n"
        "}\n";
    const char *fs_draw =
        "#version 450 core\nlayout(location=0) in vec4 colour;\nlayout(location=0) out vec4 pixel;\n"
        "void main(){ pixel = colour; }\n";
    static const float uniforms[64] = {
        0.5f, 0.0f, 0.0f, 0.5f,  0.0f, 0.5f, 0.0f, 0.5f,  0.0f, 0.0f, 0.5f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, -0.5f, 0.0f, 0.5f, 0.0f, 0.5f,  0.0f, 0.0f, 0.5f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, -0.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, 0.5f,  0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f,  0.0f, 0.0f, 0.0f, 1.0f};

    const char *vars[3] = { "gl_Position", "gl_NextBuffer", "color" };
    GLuint pgen = glCreateProgram();
    glAttachShader(pgen, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(pgen, compile(GL_VERTEX_SHADER, vs_blank));
    glAttachShader(pgen, compile(GL_FRAGMENT_SHADER, fs_blank));
    glTransformFeedbackVaryings(pgen, 3, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(pgen);
    GLint ls = 0;
    glGetProgramiv(pgen, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);
    if (!ls)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(pgen, sizeof l, &n, l);
        printf("  multi-stream instanced link log: %s\n", l);
        return;
    }

    GLuint pdraw = glCreateProgram();
    glAttachShader(pdraw, compile(GL_VERTEX_SHADER, vs_draw));
    glAttachShader(pdraw, compile(GL_FRAGMENT_SHADER, fs_draw));
    glLinkProgram(pdraw);
    ls = 0;
    glGetProgramiv(pdraw, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);

    GLuint xfb = 0, bo[2], ubo = 0, rbo = 0, fbo = 0, vao = 0;
    glGenTransformFeedbacks(1, &xfb);
    glGenBuffers(2, bo);
    glGenBuffers(1, &ubo);
    glGenRenderbuffers(1, &rbo);
    glGenFramebuffers(1, &fbo);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, xfb);
    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo[i]);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4 * 4 * sizeof(float), NULL, GL_DYNAMIC_COPY);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, bo[i]);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof uniforms, uniforms, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    {
        GLuint bi = glGetUniformBlockIndex(pdraw, "MatrixBlock");
        EXPECT_TRUE(bi != GL_INVALID_INDEX);
        if (bi != GL_INVALID_INDEX) glUniformBlockBinding(pdraw, bi, 0);
    }

    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_R8, 4, 4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    glUseProgram(pgen);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 4);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    /* four positions per stream; stream 1 holds the four white colors */
    for (int s = 0; s < 2; s++)
    {
        float m[16];
        memset(m, 0, sizeof m);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, bo[s]);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof m, m);
        if (s == 0)
        {
            EXPECT_NEAR(m[0], -1.0, 1e-5);
            EXPECT_NEAR(m[12],  1.0, 1e-5);
            EXPECT_NEAR(m[15],  1.0, 1e-5);
        }
        else
        {
            EXPECT_NEAR(m[0], 1.0, 1e-5);
            EXPECT_NEAR(m[15], 1.0, 1e-5);
        }
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, bo[0]);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, bo[1]);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(1);
    glUseProgram(pdraw);
    glDrawTransformFeedbackStreamInstanced(GL_TRIANGLE_STRIP, xfb, 0, 4);
    EXPECT_NO_GL_ERROR();
    {
        unsigned char pix[4 * 4 * 4];
        int dark = 0;
        glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, pix);
        for (int i = 0; i < 4 * 4; i++) if (pix[i * 4] < 200) dark++;
        printf("  instanced coverage: %d/16 dark\n", dark);
        EXPECT_EQ_INT(dark, 0);
    }

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(2, bo);
    glDeleteTransformFeedbacks(1, &xfb);
    glDeleteProgram(pgen);
    glDeleteProgram(pdraw);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

TEST(xfb_strip_decomposition)
{
    /* GL captures whole primitives: a strip output decomposes into its line / triangle
     * primitives, each captured with ALL of its vertices (shared strip vertices repeated),
     * and a primitive that does not fit the buffer is not captured at all.  Checks the
     * captured layout, the written count with a large buffer, and the half-full cut - the
     * shape of the CTS primitive_queries cases. */
    struct
    {
        const char *layout;
        GLenum mode;
        int n;          /* emitted strip vertices */
        int prim;       /* primitives the strip decomposes into */
        int per;        /* vertices per captured primitive */
    } cases[2] = {
        { "line_strip",     GL_LINES,     6, 5, 2 },
        { "triangle_strip", GL_TRIANGLES, 5, 3, 3 },
    };

    for (int c = 0; c < 2; c++)
    {
        char gs[1024];
        snprintf(gs, sizeof gs,
                 "#version 450 core\n"
                 "layout(points) in;\n"
                 "layout(%s, max_vertices = 8) out;\n"
                 "layout(location=0) out vec2 g_pos;\n"
                 "void main(){\n"
                 "  for (int i = 0; i < %d; i++) { g_pos = vec2(float(i), float(i) + 10.0); EmitVertex(); }\n"
                 "  EndPrimitive();\n"
                 "}\n", cases[c].layout, cases[c].n);
        const char *fs =
            "#version 450 core\n"
            "layout(location=0) out vec4 color;\n"
            "void main(){ color = vec4(0.0); }\n";
        const char *vs =
            "#version 450 core\n"
            "layout(location=0) in vec2 pos;\n"
            "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
        GLuint p = glCreateProgram();
        glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
        glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
        glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
        static const char *vars[2] = { "g_pos", NULL };
        glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) return;

        GLuint big = 0, small = 0;
        glGenBuffers(1, &big);
        glGenBuffers(1, &small);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, big);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, cases[c].prim * cases[c].per * 8, NULL, GL_STREAM_READ);
        /* half the primitives' worth of space: the whole-primitive rule must cut there */
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, small);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, (cases[c].prim / 2) * cases[c].per * 8, NULL, GL_STREAM_READ);

        glUseProgram(p);
        static const float pt[2] = { 0.0f, 0.0f };
        setup(2, pt);
        GLuint qw = 0, qg = 0;
        glGenQueries(1, &qw);
        glGenQueries(1, &qg);

        for (int round = 0; round < 2; round++)
        {
            GLuint buf = round ? small : big;
            int want = round ? cases[c].prim / 2 : cases[c].prim;

            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);
            glEnable(GL_RASTERIZER_DISCARD);
            glBeginTransformFeedback(cases[c].mode);
            glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qw);
            glBeginQuery(GL_PRIMITIVES_GENERATED, qg);
            glDrawArrays(GL_POINTS, 0, 1);
            glEndQuery(GL_PRIMITIVES_GENERATED);
            glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
            glEndTransformFeedback();
            glDisable(GL_RASTERIZER_DISCARD);
            EXPECT_NO_GL_ERROR();

            GLint written = -1, gen = -1;
            glGetQueryObjectiv(qw, GL_QUERY_RESULT, &written);
            glGetQueryObjectiv(qg, GL_QUERY_RESULT, &gen);
            printf("  %s round %d: written = %d generated = %d (want %d/%d)\n",
                   cases[c].layout, round, written, gen, want, cases[c].prim);
            EXPECT_EQ_INT(written, want);
            EXPECT_EQ_INT(gen, cases[c].prim);      /* generated is not buffer-limited */

            /* the captured stream repeats the shared strip vertices: primitive p holds
             * vertices p..p+per-1 of the emitted strip, with the ODD triangle-strip
             * primitives transposed (the GL strip-decomposition rule; KHR-GL46.transform_feedback.geometry
             * records a 4-vertex strip as v0,v1,v2 | v2,v1,v3) */
            float out[32];
            memset(out, 0, sizeof out);
            glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
            glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0,
                               (GLsizeiptr)(want * cases[c].per * 8), out);
            for (int pr = 0; pr < want; pr++)
                for (int j = 0; j < cases[c].per; j++)
                {
                    int sj = (cases[c].per == 3 && (pr & 1) && j < 2) ? (pr + 1 - j) : (pr + j);
                    EXPECT_NEAR(out[(pr * cases[c].per + j) * 2 + 0], (float)sj, 1e-5);
                    EXPECT_NEAR(out[(pr * cases[c].per + j) * 2 + 1], (float)sj + 10.0, 1e-5);
                }
        }

        glDeleteQueries(1, &qw);
        glDeleteQueries(1, &qg);
        glDeleteBuffers(1, &big);
        glDeleteBuffers(1, &small);
        glDeleteProgram(p);
    }
}

TEST(pipeline_vs_gs_capture)
{
    /* the CTS api.program_pipeline_vs_gs_capture shape: a pipeline whose GS-only program
     * captures transform feedback varyings from the geometry output (pass 1), then the
     * geometry stage is dropped from the pipeline and the VS-only program captures its own
     * varyings into the same buffer (pass 2).  Separable programs assign automatic varying
     * locations by their own declaration order, so the stages interface-match by name. */
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 pos;\n"
        "flat out ivec4 out_vs_1;\n"
        "flat out int vslot;\n"
        "void main()\n"
        "{\n"
        "    vslot = gl_VertexID;\n"
        "    out_vs_1 = ivec4(gl_VertexID, gl_VertexID + 1, gl_VertexID + 2, gl_VertexID + 3);\n"
        "    gl_Position = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "}\n";
    /* the inputs are declared in the OPPOSITE order to the VS outputs on purpose */
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "flat in int vslot[];\n"
        "flat in ivec4 out_vs_1[];\n"
        "out vec4 out_gs_1;\n"
        "void main()\n"
        "{\n"
        "    out_gs_1 = vec4(float(out_vs_1[0].x * 2), float(out_vs_1[0].y * 2 + 1),"
        " float(out_vs_1[0].z * 2 + 2), float(out_vs_1[0].w * 2 + 3));\n"
        "    gl_Position = vec4(0, 0, 0, 1);\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint g = compile(GL_GEOMETRY_SHADER, gs);
    GLuint vpo = glCreateProgram();
    GLuint gpo = glCreateProgram();
    glProgramParameteri(vpo, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glProgramParameteri(gpo, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(vpo, v);
    glAttachShader(gpo, g);
    static const char *gv[1] = { "out_gs_1" };
    static const char *vv[1] = { "out_vs_1" };
    glTransformFeedbackVaryings(gpo, 1, gv, GL_INTERLEAVED_ATTRIBS);
    glTransformFeedbackVaryings(vpo, 1, vv, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(vpo);
    glLinkProgram(gpo);
    GLint lv = 0, lg = 0;
    glGetProgramiv(vpo, GL_LINK_STATUS, &lv);
    glGetProgramiv(gpo, GL_LINK_STATUS, &lg);
    EXPECT_EQ_INT(lv, GL_TRUE);
    EXPECT_EQ_INT(lg, GL_TRUE);
    if (!lv || !lg) return;

    GLuint ppo = 0;
    glGenProgramPipelines(1, &ppo);
    glUseProgramStages(ppo, GL_VERTEX_SHADER_BIT, vpo);
    glUseProgramStages(ppo, GL_GEOMETRY_SHADER_BIT, gpo);
    glUseProgram(0);
    glBindProgramPipeline(ppo);
    EXPECT_NO_GL_ERROR();

    setup(2, (const float[]){ 0.0f, 0.0f });
    GLuint tfo = 0;
    glGenBuffers(1, &tfo);
    glBindBuffer(GL_ARRAY_BUFFER, tfo);
    glBufferData(GL_ARRAY_BUFFER, 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfo);

    /* pass 1: the geometry stage captures its own output */
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    glDisable(GL_RASTERIZER_DISCARD);

    float out[4] = { 0, 0, 0, 0 };
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 16, out);
    printf("  gs pass captured: %f %f %f %f (want 0 3 6 9)\n", out[0], out[1], out[2], out[3]);
    EXPECT_NEAR(out[0], 0.0, 1e-5);
    EXPECT_NEAR(out[1], 3.0, 1e-5);
    EXPECT_NEAR(out[2], 6.0, 1e-5);
    EXPECT_NEAR(out[3], 9.0, 1e-5);

    /* pass 2: drop the geometry stage - the vertex program captures its own varying */
    glUseProgramStages(ppo, GL_GEOMETRY_SHADER_BIT, 0);
    EXPECT_NO_GL_ERROR();
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfo);
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();

    int out2[4] = { -1, -1, -1, -1 };
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 16, out2);
    printf("  vs pass captured: %d %d %d %d (want 0 1 2 3)\n", out2[0], out2[1], out2[2], out2[3]);
    EXPECT_EQ_INT(out2[0], 0);
    EXPECT_EQ_INT(out2[1], 1);
    EXPECT_EQ_INT(out2[2], 2);
    EXPECT_EQ_INT(out2[3], 3);

    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &ppo);
    glDeleteProgram(vpo);
    glDeleteProgram(gpo);
    glDeleteShader(v);
    glDeleteShader(g);
    glDeleteBuffers(1, &tfo);
}

TEST(vs_conditional_declarations)
{
    /* the CTS nonarray_input shape: the vertex source declares its outputs inside
     * #ifndef / #else / #endif groups (including an interface block in the inactive
     * branch), and the geometry source does the same.  The capture-variant and kernel
     * rewrites must keep the preprocessor structure balanced - comments directly before a
     * declaration are their own statement, and a head scan never emits anything. */
    static const char *kVsPreamble = "#version 460 core\n";
    static const char *kVsBodyNoBlock =
        "#ifndef USE_BLOCK\n"
        "    #define V1 v1\n"
        "    #define V2 v2\n"
        "\n"
        "    out vec4 v1;\n"
        "    out vec4 v2;\n"
        "#else\n"
        "    #extension GL_ARB_shader_interface_blocks : require\n"
        "    out VS_GS\n"
        "    {\n"
        "        vec4 v1;\n"
        "        vec4 v2;\n"
        "    } interface_block;\n"
        "#endif\n"
        "\n"
        "void main()\n"
        "{\n"
        "    V1 = vec4(-1, -1, 0, 1);\n"
        "    V2 = vec4(-1,  1, 0, 1);\n"
        "}\n";
    static const char *kVsBodyBlock =
        "#ifndef USE_BLOCK\n"
        "    out vec4 v1;\n"
        "    out vec4 v2;\n"
        "#else\n"
        "    out VS_GS\n"
        "    {\n"
        "        vec4 v1;\n"
        "        vec4 v2;\n"
        "    } interface_block;\n"
        "#endif\n"
        "\n"
        "void main()\n"
        "{\n"
        "    interface_block.v1 = vec4(-1, -1, 0, 1);\n"
        "    interface_block.v2 = vec4(-1,  1, 0, 1);\n"
        "}\n";
    static const char *kDefineUseBlock = "#define USE_BLOCK\n";
    static const char *kGs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=4) out;\n"
        "in vec4 v1[];\n"
        "in vec4 v2[];\n"
        "void main()\n"
        "{\n"
        "    gl_Position = v1[0]; EmitVertex();\n"
        "    gl_Position = v2[0]; EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    static const char *kGsBlock =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices=4) out;\n"
        "in VS_GS\n"
        "{\n"
        "    vec4 v1;\n"
        "    vec4 v2;\n"
        "} ib[];\n"
        "void main()\n"
        "{\n"
        "    gl_Position = ib[0].v1; EmitVertex();\n"
        "    gl_Position = ib[0].v2; EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 c;\n"
        "void main(){ c = vec4(0, 1, 0, 1); }\n";

    struct { const char *name; const char *const *parts; int n; const char *gs; } cases[2];
    const char *noBlock[2] = { kVsPreamble, kVsBodyNoBlock };
    const char *block[3] = { kVsPreamble, kDefineUseBlock, kVsBodyBlock };
    cases[0].name = "vs-no-block"; cases[0].parts = noBlock; cases[0].n = 2; cases[0].gs = kGs;
    cases[1].name = "vs-block";    cases[1].parts = block;   cases[1].n = 3; cases[1].gs = kGsBlock;

    for (int i = 0; i < 2; i++)
    {
        GLuint v = glCreateShader(GL_VERTEX_SHADER);
        GLuint g = glCreateShader(GL_GEOMETRY_SHADER);
        GLuint f = compile(GL_FRAGMENT_SHADER, fs);
        GLuint p = glCreateProgram();
        GLint ok = 0;
        glShaderSource(v, cases[i].n, cases[i].parts, NULL);
        glCompileShader(v);
        glShaderSource(g, 1, &cases[i].gs, NULL);
        glCompileShader(g);
        glAttachShader(p, v);
        glAttachShader(p, g);
        glAttachShader(p, f);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char l[1024] = "";
            glGetProgramInfoLog(p, sizeof l, NULL, l);
            printf("  %s link log: %s\n", cases[i].name, l);
        }
        EXPECT_EQ_INT(ok, GL_TRUE);
        glDeleteProgram(p);
        glDeleteShader(v);
        glDeleteShader(g);
        glDeleteShader(f);
    }
}

TEST(gs_atomic_counter_link_fails)
{
    /* geometry-stage atomic counters are implemented on the emulation kernel (geometry.c:
     * mgl_msl_inject_atomic_counters), so a geometry shader using one LINKS and the
     * counter counts every invocation; a shader exceeding the reported limits still
     * fails the link (the CTS linking.more_ACs_in_GS_than_supported shape) */
    const char *gs =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "layout(binding = 0) uniform atomic_uint counter0;\n"
        "void main()\n"
        "{\n"
        "    atomicCounterIncrement(counter0);\n"
        "    gl_Position = vec4(0);\n"
        "    EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    const char *gs_over =
        "#version 460 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices=1) out;\n"
        "layout(binding = 0) uniform atomic_uint c0;\n"
        "layout(binding = 1) uniform atomic_uint c1;\n"
        "layout(binding = 2) uniform atomic_uint c2;\n"
        "layout(binding = 3) uniform atomic_uint c3;\n"
        "layout(binding = 4) uniform atomic_uint c4;\n"
        "layout(binding = 5) uniform atomic_uint c5;\n"
        "layout(binding = 6) uniform atomic_uint c6;\n"
        "layout(binding = 7) uniform atomic_uint c7;\n"
        "layout(binding = 8) uniform atomic_uint c8;\n"
        "void main()\n"
        "{\n"
        "    if (atomicCounter(c0) == 0u) { gl_Position = vec4(0); EmitVertex(); }\n"
        "    EndPrimitive();\n"
        "}\n";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER,
        "#version 460 core\nlayout(location=0) in vec4 pos;\nvoid main(){ gl_Position = pos; }\n"));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER,
        "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(1); }\n"));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok)
    {
        char l[512] = "";
        glGetProgramInfoLog(p, sizeof l, NULL, l);
        printf("  link log: %s", l);
    }

    /* the counter must count every geometry invocation */
    GLuint acb = 0, seed = 0;
    glGenBuffers(1, &acb);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, acb);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &seed, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, acb);
    glUseProgram(p);
    { static const float pts[4] = { 0,0,0,0 }; setup(4, pts); }
    glDrawArrays(GL_POINTS, 0, 4);
    EXPECT_NO_GL_ERROR();
    unsigned *m = (unsigned *)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 0, 4, GL_MAP_READ_BIT);
    EXPECT_TRUE(m && *m == 4);
    printf("  gs atomic counter = %u (want 4)\n", m ? *m : 0xffffffffu);
    if (m) glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    glDeleteProgram(p);

    /* nine atomic counter buffers exceeds MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS (8) */
    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER,
        "#version 460 core\nlayout(location=0) in vec4 pos;\nvoid main(){ gl_Position = pos; }\n"));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs_over));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER,
        "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(1); }\n"));
    glLinkProgram(p);
    ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    if (ok)
        printf("  link log: (linked, expected failure)\n");
    else
    {
        char l[512] = "";
        glGetProgramInfoLog(p, sizeof l, NULL, l);
        printf("  link log: %s", l);
    }
    glDeleteProgram(p);
    glDeleteBuffers(1, &acb);
}

TEST(block_vs_fs_control)
{
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out VOut { vec4 a; vec2 b; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); a = vec4(pos, 0.5, 1.0); b = pos * 0.5; }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in VOut { vec4 a; vec2 b; };\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 0.5 + b.x, 0.25, 1.0); }\n";
    GLuint p = program(vs, NULL, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char l[2048]=""; GLsizei n=0; glGetProgramInfoLog(p, sizeof l, &n, l); printf("blk ctrl log: %s\n", l); }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    /* the pixel maths below assume a 256x256 viewport centred on (128,128) */
    glViewport(0, 0, 256, 256);
    clear_black();
    { static const float t[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f }; setup(6, t); }
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(128, 128, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100);
    glDeleteProgram(p);
}

TEST(gs_struct_interface_vars)
{
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out VOut { vec4 a; vec2 b; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); a = vec4(pos, 0.5, 1.0); b = pos * 0.5; }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in VOut { vec4 a; vec2 b; } v[];\n"
        "layout(location=0) out GOut { vec4 c; vec2 d; };\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = v[i].a;\n"
        "    c = vec4(v[i].a.xy * 0.5, 0.0, 1.0);\n"
        "    d = v[i].b;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in GOut { vec4 c; vec2 d; };\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 0.5 + d.x, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof l, &n, l);
        printf("struct link log: %s\n", l);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    /* the pixel maths below assume a 256x256 viewport centred on (128,128) */
    glViewport(0, 0, 256, 256);
    clear_black();
    { static const float t[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f }; setup(6, t); }
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(128, 128, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100);
    glDeleteProgram(p);
}

/* control: a block between the vertex and fragment stages without a geometry shader */
TEST(gs_struct_member_order)
{
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
    /* member order inside the block is free: the record offsets follow the block's
     * declaration, not the location order */
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out VOut { vec2 b; vec4 a; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); b = pos * 0.5; a = vec4(pos, 0.5, 1.0); }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in VOut { vec2 b; vec4 a; } v[];\n"
        "layout(location=0) out GOut { vec2 d; vec4 e; };\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = v[i].a;\n"
        "    d = v[i].b;\n"
        "    e = v[i].a;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in GOut { vec2 d; vec4 e; };\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0, 0.5 + d.x, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof l, &n, l);
        printf("member order link log: %s\n", l);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;
    glUseProgram(p);
    /* the pixel maths below assume a 256x256 viewport centred on (128,128) */
    glViewport(0, 0, 256, 256);
    clear_black();
    { static const float t[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f }; setup(6, t); }
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte c[4];
    px(128, 128, c);
    EXPECT_TRUE(c[0] < 32 && c[1] > 100);
    glDeleteProgram(p);
}

TEST(gs_named_output_block_roundtrip)
{
    /* named interface blocks on every edge: an anonymous VS block, a named geometry input
     * block (instance array), a named geometry output block, a named fragment input block.
     * This used to be rejected at link time ("named output blocks are not supported");
     * the block is now rewritten to a plain struct + instance in the generated kernel */
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out VOut { vec4 a; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); a = vec4(pos * 0.5 + 0.5, 0.5, 1.0); }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in VOut { vec4 a; } v[];\n"
        "layout(location=0) out GOut { vec4 c; } g;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g.c = v[i].a; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in GOut { vec4 c; } g;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = g.c; }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    {
        GLuint btex, bfbo = block_fbo(&btex, 64, 64);
        setup(8, quad);
        glUseProgram(p);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        {
            GLubyte c[4];
            px(48, 48, c);      /* NDC (0.5, 0.5) -> a = (0.75, 0.75, 0.5) -> (191, 191, 128) */
            EXPECT_TRUE(abs(c[0] - 191) < 10 && abs(c[1] - 191) < 10 && abs(c[2] - 128) < 12);
        }
        glDeleteFramebuffers(1, &bfbo);
        glDeleteTextures(1, &btex);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);  /* do not leave a deleted framebuffer bound */
        glViewport(0, 0, 200, 200);
    }
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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
    { GLuint _t; GLuint _f = block_fbo(&_t, 200, 200); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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

/* transform feedback of a tessellation stage WITHOUT a geometry shader: the TES is the last
 * pre-rasterization stage, so its captured output is what GL records.  A level-1 triangle
 * patch tessellates to exactly its three corners, so the capture holds three vertices. */
TEST(xfb_of_tes_output)
{
    const char *tes =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(location=0) in vec4 tc_uv[];\n"
        "layout(location=0) out vec2 tf_pos;\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position"
        " + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  tf_pos = gl_Position.xy + 0.5 * tc_uv[0].w;\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 tf_pos;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(tf_pos, 0.0, 1.0); }\n";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, tess_vs));
    glAttachShader(p, compile(GL_TESS_CONTROL_SHADER, tri_tcs));
    glAttachShader(p, compile(GL_TESS_EVALUATION_SHADER, tes));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    {
        static const char *vars[2] = { "tf_pos", NULL };
        glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof l, &n, l);
        printf("tes xfb link log: %s\n", l);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint tbuf = 0, q = 0;
    GLint written = -1;
    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);

    glUseProgram(p);
    setup4(12, tess_tri);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_RASTERIZER_DISCARD);
    glGenQueries(1, &q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[8];
    memset(out, 0, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int i = 0; i < 3; i++) printf("  tes xfb v%d = %f %f\n", i, out[i * 2], out[i * 2 + 1]);

    /* the three tessellated corners: tc_uv[0].w is 1, so each is shifted by 0.5.  The
     * tessellator's vertex order is not specified, so the corners are matched as a set */
    {
        static const float want[3][2] = { { 1.0f, 0.0f }, { 0.5f, 1.0f }, { 0.0f, 0.0f } };
        for (int i = 0; i < 3; i++)
        {
            bool found = false;
            for (int j = 0; j < 3; j++)
                if (fabsf(out[j * 2] - want[i][0]) < 1e-5f && fabsf(out[j * 2 + 1] - want[i][1]) < 1e-5f)
                    found = true;
            EXPECT_TRUE(found);
        }
    }

    glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
    printf("  tes xfb primitives written = %d\n", written);
    EXPECT_EQ_INT(written, 1);            /* three captured vertices = one triangle */
    glDeleteQueries(1, &q);
    glDeleteProgram(p);
}

/* the same tessellated patch behind a geometry shader: transform feedback records the
 * GEOMETRY stage's output instead - the shrunken triangle, not the TES corners */
TEST(xfb_of_tes_gs_output)
{
    const char *tes =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(location=0) in vec4 tc_uv[];\n"
        "layout(location=0) out vec2 tf_pos;\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position"
        " + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  tf_pos = gl_Position.xy + 0.5 * tc_uv[0].w;\n"
        "}\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in vec2 tf_pos[];\n"
        "layout(location=0) out vec2 g_pos;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    gl_Position = vec4(gl_in[i].gl_Position.xy * 0.5, 0.0, 1.0);\n"
        "    g_pos = tf_pos[i] * 0.5;\n"
        "    EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 g_pos;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(g_pos, 0.0, 1.0); }\n";
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, tess_vs));
    glAttachShader(p, compile(GL_TESS_CONTROL_SHADER, tri_tcs));
    glAttachShader(p, compile(GL_TESS_EVALUATION_SHADER, tes));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    {
        static const char *vars[2] = { "g_pos", NULL };
        glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char l[2048] = ""; GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof l, &n, l);
        printf("tes+gs xfb link log: %s\n", l);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint tbuf = 0, q = 0;
    GLint written = -1;
    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);

    glUseProgram(p);
    setup4(12, tess_tri);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_RASTERIZER_DISCARD);
    glGenQueries(1, &q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float out[8];
    memset(out, 0, sizeof out);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    for (int i = 0; i < 3; i++) printf("  tes+gs xfb v%d = %f %f\n", i, out[i * 2], out[i * 2 + 1]);

    /* the geometry stage halved every TES corner */
    {
        static const float want[3][2] = { { 0.5f, 0.0f }, { 0.25f, 0.5f }, { 0.0f, 0.0f } };
        for (int i = 0; i < 3; i++)
        {
            bool found = false;
            for (int j = 0; j < 3; j++)
                if (fabsf(out[j * 2] - want[i][0]) < 1e-5f && fabsf(out[j * 2 + 1] - want[i][1]) < 1e-5f)
                    found = true;
            EXPECT_TRUE(found);
        }
    }

    glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
    printf("  tes+gs xfb primitives written = %d\n", written);
    EXPECT_EQ_INT(written, 1);
    glDeleteQueries(1, &q);
    glDeleteProgram(p);
}

TEST(tessellation_point_mode_into_gs)
{
    { GLuint _t; GLuint _f = block_fbo(&_t, 200, 200); (void)_f; }   /* private 256x256 FBO: pixel checks read up to (255,255) */
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

TEST(primitive_restart_points_gl_primitive_id_in)
{
    /* POINTS + restart mid-list (dEQP geometry_shading.basic.primitive_id_in_restarted):
     * indices [3, 2, restart, 1] give 3 points with gl_PrimitiveIDIn 0,1,2 — the
     * CPU-assembly path used the restart-free count as the sequence length and
     * dropped the point after the restart */
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(points, max_vertices = 1) out;\n"
        "layout(location=0) out float id;\n"
        "void main(){ gl_Position = gl_in[0].gl_Position; id = float(gl_PrimitiveIDIn);"
        " EmitVertex(); EndPrimitive(); }\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in float id;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(1.0, id / 2.0, 0.0, 1.0); }\n";
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    GLuint p = program(vs, gs, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    /* own 256x256 framebuffer: the default drawable is 1024x768 and the point
     * coordinates below are exact only for a known viewport */
    GLint saved_vp[4] = { 0, 0, 0, 0 };
    glGetIntegerv(GL_VIEWPORT, saved_vp);
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 256, 256);
    glUseProgram(p);
    clear_black();
    glPointSize(32.0f);
    static const float v[8] = { -0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f };
    static const GLushort idx16[4] = { 3, 2, 0xffff, 1 };
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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx16, idx16, GL_STATIC_DRAW);

    /* one restarting draw (GL_PRIMITIVE_RESTART with the ushort index) */
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xffff);
    glDrawElements(GL_POINTS, 4, GL_UNSIGNED_SHORT, 0);
    glDisable(GL_PRIMITIVE_RESTART);
    EXPECT_NO_GL_ERROR();

    /* points for vertices 3, 2, 1 (drawn in that order): gl_PrimitiveIDIn 0, 1, 2 */
    struct { int x, y; float g; } want[3] = { { 192, 64, 0.0f }, { 64, 64, 0.5f }, { 192, 192, 1.0f } };
    for (int i = 0; i < 3; i++)
    {
        unsigned char px[4];
        glFinish();
        glReadPixels(want[i].x, want[i].y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        EXPECT_TRUE(px[0] > 200);
        EXPECT_TRUE(px[1] == (unsigned char)(want[i].g * 255.0f + 0.5f));
    }
    /* vertex 0's quadrant stays empty */
    unsigned char px[4];
    glFinish();
    glReadPixels(64, 192, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_TRUE(px[0] == 0 && px[1] == 0);

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao2);
    glDeleteBuffers(1, &vbo2);
    glDeleteBuffers(1, &ebo);
    /* restore the state the next test sees: point size, framebuffer, viewport */
    glPointSize(1.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(saved_vp[0], saved_vp[1], saved_vp[2], saved_vp[3]);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

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

/* read a pixel from the framebuffer of the test's bound FBO */
static void geom_read_u8(int x, int y, GLubyte out[4])
{
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}
#define EXPECT_U8(x, y, r, g, b, a) do { GLubyte _p[4] = { 1, 2, 3, 4 }; geom_read_u8(x, y, _p); mgl_test_checks++; \
        if (_p[0] != (GLubyte)(r) || _p[1] != (GLubyte)(g) || _p[2] != (GLubyte)(b) || _p[3] != (GLubyte)(a)) \
            MGL_FAIL("(%d,%d) is %u,%u,%u,%u, expected %d,%d,%d,%d", x, y, _p[0], _p[1], _p[2], _p[3], r, g, b, a); } while (0)

TEST(pool_rotation_hazard)
{
    /* the geometry scratch pools rotate over a few buffers because the CPU writes the
     * next draw's inputs while the GPU may still be reading the previous draws' records;
     * a pool that reused its one buffer immediately would corrupt batched draws.  Draw
     * many small geometry draws back to back — no glFinish between them, so the GPU lags
     * the CPU — each producing a quad at its own position, and verify every quad. */
    const char *vs = "#version 450 core\n"
                     "layout(location=0) in vec2 pos;\n"
                     "layout(location=0) out vec2 v_pos;\n"
                     "void main(){ v_pos = pos; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *gs = "#version 450 core\n"
                     "layout(points) in;\n"
                     "layout(location=0) in vec2 v_pos[];\n"
                     "layout(triangle_strip, max_vertices=4) out;\n"
                     "layout(location=0) out vec2 g_uv;\n"
                     "void main(){\n"
                     "  vec2 p = v_pos[0], e = vec2(0.035, 0.035);\n"
                     "  g_uv = vec2(0); gl_Position = vec4(p.x - e.x, p.y - e.y, 0.0, 1.0); EmitVertex();\n"
                     "  g_uv = vec2(1); gl_Position = vec4(p.x + e.x, p.y - e.y, 0.0, 1.0); EmitVertex();\n"
                     "  g_uv = vec2(1); gl_Position = vec4(p.x - e.x, p.y + e.y, 0.0, 1.0); EmitVertex();\n"
                     "  g_uv = vec2(0); gl_Position = vec4(p.x + e.x, p.y + e.y, 0.0, 1.0); EmitVertex();\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "layout(location=0) in vec2 g_uv;\n"
                     "layout(location=0) out vec4 c0;\n"
                     "void main(){ if (min(g_uv.x, g_uv.y) < 0.25) discard; c0 = vec4(0, 1, 0, 1); }\n";
    GLuint p = program(vs, gs, fs);
    if (!p) return;

    GLuint tex, fbo, vao, vbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 256, 256);
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const int N = 96;
    float px[N], py[N];
    for (int i = 0; i < N; i++)
    {
        /* a grid of quad centres in NDC; remember the framebuffer pixel to check */
        px[i] = -0.95f + 1.9f * ((i % 12) / 11.0f);
        py[i] = -0.95f + 1.9f * ((i / 12) / 7.0f);
    }
    glFinish();
    for (int i = 0; i < N; i++)
    {
        float v[2] = { px[i], py[i] };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof v, v);
        glDrawArrays(GL_POINTS, 0, 1);          /* no flush: the GPU lags behind */
    }
    glFinish();

    for (int i = 0; i < N; i++)
    {
        int x = (int)((px[i] * 0.5f + 0.5f) * 256), y = (int)((py[i] * 0.5f + 0.5f) * 256);
        if (x > 0 && x < 255 && y > 0 && y < 255)
        {
            EXPECT_U8(x, y, 0, 255, 0, 255);
            EXPECT_U8(x, y + 6, 0, 0, 0, 255);       /* the gap between quads */
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

TEST(wild_index_survives_cpu_assembly)
{
    /* an out-of-range index (0x10000 with a four-vertex buffer) stretches the capture
     * ring's vertex window (vcount = vmax - vmin + 1); a corrupted element store reading
     * near-0xffffffff indices used to size it at 24 * 4e9 bytes, the allocation failed,
     * the geometry pipelines were torn down and every following draw fell back.  The ring
     * is budget-capped now; this draw is undefined, but MGL must stay standing and the
     * next draw must render exactly as before. */
    const char *vs = "#version 450 core\n"
                     "layout(location=0) in vec2 pos;\n"
                     "layout(location=0) out vec2 v_pos;\n"
                     "void main(){ v_pos = pos; gl_Position = vec4(pos * 0.4, 0.0, 1.0); }\n";
    const char *gs = "#version 450 core\n"
                     "layout(points) in;\n"
                     "layout(location=0) in vec2 v_pos[];\n"
                     "layout(points, max_vertices = 1) out;\n"
                     "layout(location=0) out vec2 g_pos;\n"
                     "void main(){ g_pos = v_pos[0]; gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }\n";
    const char *fs = "#version 450 core\n"
                     "layout(location=0) in vec2 g_pos;\n"
                     "layout(location=0) out vec4 c0;\n"
                     "void main(){ c0 = vec4(1, 0, 0, 1); }\n";
    GLuint p = program(vs, gs, fs);
    if (!p) return;
    {
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
    }

    GLuint tex, fbo, vao, vbo, ebo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 64, 64);
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * 4, NULL, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);



    /* the out-of-range index draw (undefined result, MGL must survive it) */
    {
        float v[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
        GLuint wild[3] = { 0, 0x10000u, 0 };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof v, v);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof wild, wild, GL_STATIC_DRAW);
        glDrawElements(GL_POINTS, 3, GL_UNSIGNED_INT, NULL);
        glFinish();
    }

    /* the next draw must be unaffected */
    {
        float v[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
        GLuint idx[3] = { 0, 1, 2 };
        GLuint ok[1] = { 3 };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof v, v);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glDrawElements(GL_POINTS, 3, GL_UNSIGNED_INT, NULL);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof ok, ok);
        glDrawElements(GL_POINTS, 1, GL_UNSIGNED_INT, NULL);
        glFinish();
    }
    /* the points land at NDC (±0.4, ±0.4): pixels (19,19), (44,19), (19,44), (44,44) */
    EXPECT_U8(19, 19, 255, 0, 0, 255);
    EXPECT_U8(44, 44, 255, 0, 0, 255);
    EXPECT_U8(8, 56, 0, 0, 0, 255);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

TEST(element_store_tail_integrity)
{
    /* regression test for the "0xfffed312 at a fixed tail index" report: a 300032-word
     * element store whose scattered indices are built with (GLuint)((i * 7919) % VERTS)
     * must contain exactly those words at every index — in particular from i = 271182,
     * where i * 7919 overflows a signed int and the naive signed expression produces
     * (GLuint)(-77038) = 0xfffed312.  The index is well inside the store (271182 <
     * 300032), so what the assembly reads must be what was uploaded; the driver must not
     * truncate or mangle the tail. */
    enum { WORDS = 300032, VERTS2 = 100000 };
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "layout(location=0) in vec2 v_pos[];\n"
                     "layout(location=0) out vec2 g_pos;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_pos = v_pos[i]; EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    GLuint p = program(plain_vs, gs, plain_fs);
    GLuint ebo = 0, q = 0;
    GLuint *idx = malloc(sizeof(GLuint) * WORDS);
    GLuint *back = malloc(sizeof(GLuint) * WORDS);
    GLint emitted = 0;

    if (!p) { free(idx); free(back); return; }

    for (GLuint i = 0; i < WORDS; i++)
        idx[i] = (GLuint)((i * 7919u) % VERTS2);   /* unsigned: no overflow for any i */

    /* the VAO must be bound before the element buffer (element-array binding is per-VAO),
     * and every index must be in range so the draw is fully defined */
    {
        float *verts = malloc(sizeof(float) * 2 * VERTS2);
        for (int i = 0; i < VERTS2; i++)
        {
            verts[i * 2 + 0] = -1.0f + 2.0f * (float)(i % 977) / 976.0f;
            verts[i * 2 + 1] = -1.0f + 2.0f * (float)((i * 31) % 977) / 976.0f;
        }
        setup(VERTS2, verts);
        free(verts);
    }

    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * WORDS, idx, GL_STATIC_DRAW);

    /* read the store back and compare word by word */
    glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(GLuint) * WORDS, back);
    EXPECT_NO_GL_ERROR();
    {
        int bad = 0;
        for (GLuint i = 0; i < WORDS; i++)
            if (back[i] != (GLuint)((i * 7919u) % VERTS2)) bad++;
        EXPECT_EQ_INT(bad, 0);
        /* the exact words that a signed (int)(i * 7919) % VERTS would corrupt */
        EXPECT_EQ_INT(back[271181], (GLuint)(271181u * 7919u) % VERTS2);
        EXPECT_EQ_INT(back[271182], 90258u);
        EXPECT_EQ_INT(back[271182] == 0xfffed312u, 0);
        EXPECT_EQ_INT(back[WORDS - 1], (GLuint)(((GLuint)(WORDS - 1)) * 7919u) % VERTS2);
    }

    /* and the assembly must accept the store: draw through the GS and count primitives */
    glUseProgram(p);
    glGenQueries(1, &q);
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q);
    glDrawElements(GL_TRIANGLES, WORDS, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &emitted);
    EXPECT_EQ_INT(emitted, WORDS / 3);

    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(p);
    free(idx);
    free(back);
    EXPECT_NO_GL_ERROR();
}

TEST(tiled_restart_gpu_assembly)
{
    /* the tile-parallel prep chain (draws above MGL_GEOM_TILED_MIN) used to scan only
     * 1/32nd of the element store (tile index taken from the global thread id), so with
     * primitive restart almost every run record was stale: out_prims came out as the
     * candidate count (count/3) or garbage, and the geometry kernel dispatched up to
     * 100x the threadgroups it needed (the 50 ms/frame restart regression).  This test
     * drives a large restart-heavy indexed draw through the GPU assembly path (warm-up
     * draws first, so the element store is Metal-backed) and checks the exact number of
     * primitives the geometry stage emitted via transform feedback. */
    enum { COUNT = 60000, VERTS3 = 1000, RUNS = COUNT / 6 };   /* 10000 triangles */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "layout(location=0) in vec2 v_pos[];\n"
                     "layout(location=0) out vec2 g_pos;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_pos = v_pos[i]; EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\nlayout(location=0) out vec4 c;\nvoid main(){ c = vec4(0.0); }\n";
    const char *vs = "#version 450 core\nlayout(location=0) in vec2 pos;\nlayout(location=0) out vec2 v_pos;\nvoid main(){ gl_Position = vec4(pos * 0.02, 0.0, 1.0); v_pos = pos; }\n";
    static const char *vars[2] = { "g_pos", NULL };
    GLuint p = glCreateProgram();
    GLuint ebo = 0, tbuf = 0, q = 0;
    GLuint *idx = malloc(sizeof(GLuint) * COUNT);
    float *verts = malloc(sizeof(float) * 2 * VERTS3);
    GLint written = -1;

    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    {
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) { free(idx); free(verts); return; }
    }

    for (int i = 0; i < VERTS3; i++)
    {
        verts[i * 2 + 0] = (float)(i % 97);
        verts[i * 2 + 1] = (float)(i % 89);
    }
    for (int i = 0; i < COUNT; i++)
        idx[i] = (GLuint)(((GLuint)i * 7919u) % VERTS3);

    setup(VERTS3, verts);                      /* binds the VAO and vertex buffer */
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * COUNT, idx, GL_STATIC_DRAW);
    glUseProgram(p);

    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 6 * sizeof(float) * 3 * RUNS, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);
    glGenQueries(1, &q);

    /* warm-up without restart: promotes the element store, so the draws below use the
     * GPU assembly (the tile-parallel chain) */
    glEnable(GL_RASTERIZER_DISCARD);
    for (int i = 0; i < 3; i++)
        glDrawElements(GL_TRIANGLES, COUNT, GL_UNSIGNED_INT, NULL);
    glFinish();

    /* restart every 6th index: RUNS runs of five indices -> one triangle each */
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xffffffffu);
    for (int i = 5; i < COUNT; i += 6) idx[i] = 0xffffffffu;
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * COUNT, idx, GL_STATIC_DRAW);

    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glDrawElements(GL_TRIANGLES, COUNT, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_PRIMITIVE_RESTART);
    EXPECT_NO_GL_ERROR();

    glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
    printf("  tiled restart xfb written = %d (expect %d)\n", written, RUNS);
    EXPECT_EQ_INT(written, RUNS);

    /* every record must be a real vertex of the mesh (no stale/garbage records);
     * the only captured varying is g_pos, so a record is two floats */
    {
        float *out = malloc(2 * sizeof(float) * 3 * RUNS);
        memset(out, 0, 2 * sizeof(float) * 3 * RUNS);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 2 * sizeof(float) * 3 * RUNS, out);
        int bad = 0;
        for (int r = 0; r < 3 * RUNS; r++)
        {
            float x = out[r * 2 + 0], y = out[r * 2 + 1];
            if (x < 0.0f || x > 96.0f || y < 0.0f || y > 88.0f ||
                x != (float)(int)x || y != (float)(int)y)
                bad++;
        }
        printf("  records: %d of %d out of range/non-integral\n", bad, 3 * RUNS);
        EXPECT_EQ_INT(bad, 0);
        free(out);
    }

    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &tbuf);
    glDeleteBuffers(1, &ebo);
    glDeleteProgram(p);
    free(idx);
    free(verts);
    EXPECT_NO_GL_ERROR();
}

/* ---------- named output interface blocks ---------- */

TEST(gs_output_block_instance_name)
{
    /* out Blk { vec4 c; } b; in the geometry stage, in Blk { vec4 c; } b; in the fragment
     * stage: the block is rewritten to a plain struct + instance in the generated kernel
     * (the body keeps writing b.c), re-emitted as a location-addressed varying for
     * rasterization, and matched by location to the fragment stage's input block */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { vec4 c; } b;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.c = vec4(0.0, 1.0, 0.0, 1.0); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "in Blk { vec4 c; } b;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ o = b.c; }\n";
    GLuint p = program(plain_vs, gs, fs);

    if (!p) return;
    GLuint btex, bfbo = block_fbo(&btex, 64, 64);
    setup(8, quad);
    glUseProgram(p);
    clear_black();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_EQ_INT(count_block(32, 32, is_green), 49);
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &bfbo);
    glDeleteTextures(1, &btex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);      /* do not leave a deleted framebuffer bound */
    glViewport(0, 0, 200, 200);
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_two_members_and_plain)
{
    /* two block members plus a plain varying beside them: the member locations follow the
     * declaration order and the plain varying takes the next free one */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { vec4 a; vec2 b; } blk;\n"
                     "layout(location = 4) out float plain;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; blk.a = vec4(0.0, 1.0, 0.0, 1.0); blk.b = vec2(0.5); plain = 0.25; EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "in Blk { vec4 a; vec2 b; } blk;\n"
                     "layout(location = 4) in float plain;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ o = blk.a * (blk.b.x + plain); }\n";
    GLuint p = program(plain_vs, gs, fs);

    if (!p) return;
    GLuint btex, bfbo = block_fbo(&btex, 64, 64);
    setup(8, quad);
    glUseProgram(p);
    clear_black();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    {
        GLubyte c[4];
        px(32, 32, c);          /* 0.75 * green = 191 */
        EXPECT_TRUE(c[0] < 32 && c[1] > 170 && c[1] < 210 && c[2] < 32);
    }
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &bfbo);
    glDeleteTextures(1, &btex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);      /* do not leave a deleted framebuffer bound */
    glViewport(0, 0, 200, 200);
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_array_member)
{
    /* vec4 c[2] inside the block: two consecutive locations, both elements carried */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { layout(location = 2) vec4 c[2]; } b;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.c[0] = vec4(0.0, 1.0, 0.0, 1.0); b.c[1] = vec4(0.0, 0.0, 1.0, 1.0); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "in Blk { layout(location = 2) vec4 c[2]; } b;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ o = b.c[0] * 0.5 + b.c[1] * 0.5; }\n";
    GLuint p = program(plain_vs, gs, fs);

    if (!p) return;
    GLuint btex, bfbo = block_fbo(&btex, 64, 64);
    setup(8, quad);
    glUseProgram(p);
    clear_black();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    {
        GLubyte c[4];
        px(32, 32, c);          /* half green + half blue */
        EXPECT_TRUE(c[0] < 32 && c[1] > 100 && c[1] < 155 && c[2] > 100 && c[2] < 155);
    }
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &bfbo);
    glDeleteTextures(1, &btex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);      /* do not leave a deleted framebuffer bound */
    glViewport(0, 0, 200, 200);
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_flat_and_matrix_members)
{
    /* flat interpolation and a mat2 member survive the struct rewrite */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { flat vec4 f; mat2 m; } b;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.f = vec4(1.0, 1.0, 0.0, 1.0); b.m = mat2(1.0, 0.25, 0.25, 1.0); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "in Blk { flat vec4 f; mat2 m; } b;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ vec2 v = b.m * vec2(0.25, 0.25); o = b.f * vec4(v, 0.0, 1.0); }\n";
    GLuint p = program(plain_vs, gs, fs);

    if (!p) return;
    GLuint btex, bfbo = block_fbo(&btex, 64, 64);
    setup(8, quad);
    glUseProgram(p);
    clear_black();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    {
        GLubyte c[4];
        px(32, 32, c);          /* yellow * (0.3125, 0.3125) -> (80, 80, 0) */
        EXPECT_TRUE(c[0] > 60 && c[0] < 100 && c[1] > 60 && c[1] < 100 && c[2] < 32);
    }
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &bfbo);
    glDeleteTextures(1, &btex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);      /* do not leave a deleted framebuffer bound */
    glViewport(0, 0, 200, 200);
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_anonymous)
{
    /* an interface block without an instance name puts its members in the enclosing
     * scope; the fragment stage may still use an instance name of its own */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { vec4 c; };\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; c = vec4(0.0, 1.0, 0.0, 1.0); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\n"
                     "in Blk { vec4 c; } i;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ o = i.c; }\n";
    GLuint p = program(plain_vs, gs, fs);

    if (!p) return;
    GLuint btex, bfbo = block_fbo(&btex, 64, 64);
    setup(8, quad);
    glUseProgram(p);
    clear_black();
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_EQ_INT(count_block(32, 32, is_green), 49);
    glDeleteProgram(p);
    glDeleteFramebuffers(1, &bfbo);
    glDeleteTextures(1, &btex);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);      /* do not leave a deleted framebuffer bound */
    glViewport(0, 0, 200, 200);
    EXPECT_NO_GL_ERROR();
}

TEST(xfb_of_gs_output_block_members)
{
    /* transform feedback of block members named as instance.member, both through the API
     * (interleaved, sequential offsets) and through shader-declared xfb_buffer/xfb_offset
     * (GL 4.6 13.3.2: the in-shader qualifiers capture when no API list is given) */
    const char *vs = plain_vs;
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { vec4 c; vec2 d; } b;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.c = vec4(0.125, 0.25, 0.5, 1.0); b.d = vec2(0.75, 0.875); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = plain_fs;
    static const char *names[2] = { "b.c", "b.d" };
    GLuint p, tfb, tbo;
    GLfloat rec[10];

    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glTransformFeedbackVaryings(p, 2, names, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    {
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) { glDeleteProgram(p); return; }
    }
    setup(8, quad);
    glUseProgram(p);
    glGenTransformFeedbacks(1, &tfb);
    glGenBuffers(1, &tbo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glFinish();

    memset(rec, 0xcc, sizeof rec);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
    {
        int good = fabsf(rec[0] - 0.125f) < 1e-4f && fabsf(rec[1] - 0.25f) < 1e-4f &&
                   fabsf(rec[2] - 0.5f) < 1e-4f && fabsf(rec[3] - 1.0f) < 1e-4f &&
                   fabsf(rec[4] - 0.75f) < 1e-4f && fabsf(rec[5] - 0.875f) < 1e-4f;
        EXPECT_TRUE(good);
        printf("  api-named block member xfb: %s\n", good ? "ok" : "BAD");
    }
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfb);
    glDeleteBuffers(1, &tbo);
    glDeleteProgram(p);

    /* shader-declared capture: no glTransformFeedbackVaryings call, the layout qualifiers
     * pick the buffer (1) and the offsets (16 and 32) */
    {
        const char *gs2 = "#version 450 core\n"
                          "layout(triangles) in;\n"
                          "layout(triangle_strip, max_vertices = 3) out;\n"
                          "layout(xfb_buffer = 1) out Blk { layout(xfb_offset = 16) vec4 c; layout(xfb_offset = 32) vec2 d; } b;\n"
                          "void main(){\n"
                          "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.c = vec4(0.125, 0.25, 0.5, 1.0); b.d = vec2(0.75, 0.875); EmitVertex(); }\n"
                          "  EndPrimitive();\n"
                          "}\n";
        p = glCreateProgram();
        glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
        glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs2));
        glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
        glLinkProgram(p);
        {
            GLint ok = 0;
            glGetProgramiv(p, GL_LINK_STATUS, &ok);
            EXPECT_EQ_INT(ok, GL_TRUE);
            if (!ok) { glDeleteProgram(p); return; }
        }
        glUseProgram(p);
        glGenTransformFeedbacks(1, &tfb);
        glGenBuffers(1, &tbo);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, tbo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glEnable(GL_RASTERIZER_DISCARD);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        glFinish();

        memset(rec, 0xcc, sizeof rec);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
        {
            /* c at byte 16, d at byte 32: words 4..7 and 8..9 */
            int good = fabsf(rec[4] - 0.125f) < 1e-4f && fabsf(rec[5] - 0.25f) < 1e-4f &&
                       fabsf(rec[6] - 0.5f) < 1e-4f && fabsf(rec[7] - 1.0f) < 1e-4f &&
                       fabsf(rec[8] - 0.75f) < 1e-4f && fabsf(rec[9] - 0.875f) < 1e-4f;
            EXPECT_TRUE(good);
            printf("  shader-declared xfb_buffer/offset: %s\n", good ? "ok" : "BAD");
        }
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
        glDeleteTransformFeedbacks(1, &tfb);
        glDeleteBuffers(1, &tbo);
        glDeleteProgram(p);
    }
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_after_tessellation)
{
    /* a tessellated pipeline whose geometry stage declares a named output block: the
     * evaluation stage also feeds the geometry stage through a named block (matched by
     * block name, with different instance names on both sides) */
    const char *vs = "#version 450 core\nlayout(location=0) in vec2 pos;\nlayout(location=0) out vec2 v_pos;\nvoid main(){ v_pos = pos; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *tc = "#version 450 core\nlayout(vertices = 3) out;\nlayout(location=0) in vec2 v_pos[];\nlayout(location=0) out vec2 tc_pos[];\nvoid main(){ gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0; gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0; tc_pos[gl_InvocationID] = v_pos[gl_InvocationID]; gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position; }\n";
    const char *te = "#version 450 core\nlayout(triangles, equal_spacing, ccw) in;\nout VBlk { vec2 p; } vout;\nvoid main(){ gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position; vout.p = gl_Position.xy; }\n";
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "in VBlk { vec2 p; } vin[];\n"
                     "out GBlk { vec4 g; } gout;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; gout.g = vec4(vin[i].p * 0.5 + 0.5, 0.0, 1.0); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = "#version 450 core\nin GBlk { vec4 g; } gin;\nlayout(location=0) out vec4 o;\nvoid main(){ o = gin.g; }\n";
    GLuint p, s;
    GLint ok = 0;

    p = glCreateProgram();
    s = compile(GL_VERTEX_SHADER, vs); glAttachShader(p, s);
    s = compile(GL_TESS_CONTROL_SHADER, tc); glAttachShader(p, s);
    s = compile(GL_TESS_EVALUATION_SHADER, te); glAttachShader(p, s);
    s = compile(GL_GEOMETRY_SHADER, gs); glAttachShader(p, s);
    s = compile(GL_FRAGMENT_SHADER, fs); glAttachShader(p, s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    {
        static const float tri[] = { -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f };
        setup(6, tri);
    }
    {
        GLuint btex, bfbo = block_fbo(&btex, 32, 32);
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        glUseProgram(p);
        glDrawArrays(GL_PATCHES, 0, 3);
        {
            GLubyte c[4];
            px(16, 16, c);      /* NDC (0.031, 0.031) -> p*0.5+0.5 -> (132, 132, 0) */
            EXPECT_TRUE(abs(c[0] - 132) < 14 && abs(c[1] - 132) < 14 && c[2] < 32);
        }
        glDeleteFramebuffers(1, &bfbo);
        glDeleteTextures(1, &btex);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);  /* do not leave a deleted framebuffer bound */
        glViewport(0, 0, 200, 200);
    }
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

TEST(gs_output_block_xfb_errors)
{
    /* a member that does not exist must fail the link; a whole-block name selects every
     * member (the capture then contains both, sequentially) */
    const char *gs = "#version 450 core\n"
                     "layout(triangles) in;\n"
                     "layout(triangle_strip, max_vertices = 3) out;\n"
                     "out Blk { vec4 c; vec2 d; } b;\n"
                     "void main(){\n"
                     "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; b.c = vec4(0.125, 0.25, 0.5, 1.0); b.d = vec2(0.75, 0.875); EmitVertex(); }\n"
                     "  EndPrimitive();\n"
                     "}\n";
    const char *fs = plain_fs;
    static const char *bad[1] = { "b.nope" };
    static const char *whole[1] = { "b" };
    GLuint p;
    GLint ok = 0;

    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, plain_vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glTransformFeedbackVaryings(p, 1, bad, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    glDeleteProgram(p);

    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, plain_vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glTransformFeedbackVaryings(p, 1, whole, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (ok)
    {
        GLuint tfb, tbo;
        GLfloat rec[6];
        setup(8, quad);
        glUseProgram(p);
        glGenTransformFeedbacks(1, &tfb);
        glGenBuffers(1, &tbo);
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfb);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
        glEnable(GL_RASTERIZER_DISCARD);
        glBeginTransformFeedback(GL_TRIANGLES);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        glFinish();
        memset(rec, 0xcc, sizeof rec);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbo);
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof rec, rec);
        EXPECT_TRUE(fabsf(rec[0] - 0.125f) < 1e-4f && fabsf(rec[4] - 0.75f) < 1e-4f &&
                    fabsf(rec[5] - 0.875f) < 1e-4f);
        printf("  whole-block xfb selection: c then d captured\n");
        glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
        glDeleteTransformFeedbacks(1, &tfb);
        glDeleteBuffers(1, &tbo);
    }
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

TEST(pool_wrap_stress_tiny)
{
    /* The scratch pools' slot-reuse rule must be safe however far the GPU lags the CPU.
     * The pool knobs are read once, at the first geometry draw of the process, and this
     * test runs first: it forces tiny pools (16 KiB per slot, 2 slots), so the allocator
     * has to wrap, reclaim and wait every few draws — and every geometry test below keeps
     * running under the same tiny pools.  The batch below is the purest hazard case:
     * many geometry draws with NOTHING between them (no state change, no buffer update,
     * no finish), so the draws pile into one command buffer while the CPU keeps handing
     * out ranges — a reuse rule that reasons about GPU timing instead of range ownership
     * corrupts exactly this.  A tess->GS draw is interleaved every few quads: the tess
     * import consumes the pools from a command buffer the tessellation stage itself
     * commits, the other consumer path a slot can have. */
    setenv("MGL_GEOM_POOL_CAP", "16384", 1);
    setenv("MGL_GEOM_POOL_SLOTS", "2", 1);
    { GLuint _t; GLuint _f = block_fbo(&_t, 256, 256); (void)_f; }

    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "layout(location=0) out vec2 g_uv;\n"
        "void main(){\n"
        "  vec2 p = v_pos[0], e = vec2(0.02, 0.02);\n"
        "  g_uv = vec2(0); gl_Position = vec4(p.x - e.x, p.y - e.y, 0.0, 1.0); EmitVertex();\n"
        "  g_uv = vec2(1); gl_Position = vec4(p.x + e.x, p.y - e.y, 0.0, 1.0); EmitVertex();\n"
        "  g_uv = vec2(1); gl_Position = vec4(p.x - e.x, p.y + e.y, 0.0, 1.0); EmitVertex();\n"
        "  g_uv = vec2(0); gl_Position = vec4(p.x + e.x, p.y + e.y, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 g_uv;\n"
        "layout(location=0) out vec4 c0;\n"
        "void main(){ if (min(g_uv.x, g_uv.y) < 0.25) discard; c0 = vec4(0, 1, 0, 1); }\n";
    const char *tgs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(location=0) in float shade[];\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) out float gshade;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = vec4(gl_in[i].gl_Position.xy * 0.06 + vec2(0.86, 0.86),"
        " 0.0, 1.0); gshade = shade[i]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint p = program(plain_vs, gs, fs);
    GLuint tp = tess_gs_prog(tri_tcs, tri_tes, tgs);
    if (!p || !tp) return;

    /* the point batch: 96 distinct positions in one static buffer, one draw each */
    const int N = 96;
    static float pts[N][2];
    for (int i = 0; i < N; i++)
    {
        pts[i][0] = -0.95f + 1.9f * ((i % 12) / 11.0f);
        pts[i][1] = -0.95f + 1.9f * ((i / 12) / 7.0f);
    }
    GLuint pvao, pvbo, tvao, tvbo;
    glGenVertexArrays(1, &pvao);
    glBindVertexArray(pvao);
    glGenBuffers(1, &pvbo);
    glBindBuffer(GL_ARRAY_BUFFER, pvbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pts, pts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glGenVertexArrays(1, &tvao);
    glBindVertexArray(tvao);
    glGenBuffers(1, &tvbo);
    glBindBuffer(GL_ARRAY_BUFFER, tvbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tess_tri, tess_tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glUseProgram(p);                /* the batch below starts with point draws */
    clear_black();
    glFinish();
    for (int i = 0; i < N; i++)
    {
        if ((i & 7) == 4)
        {
            /* interleaved tess->GS consumer (commits its own command buffer) */
            glUseProgram(tp);
            glBindVertexArray(tvao);
            glDrawArrays(GL_PATCHES, 0, 3);
            glUseProgram(p);
        }
        glBindVertexArray(pvao);
        glDrawArrays(GL_POINTS, i, 1);          /* nothing between the draws */
    }
    glFinish();
    EXPECT_NO_GL_ERROR();

    int missing = 0;
    for (int i = 0; i < N; i++)
    {
        int x = (int)((pts[i][0] * 0.5f + 0.5f) * 256), y = (int)((pts[i][1] * 0.5f + 0.5f) * 256);
        if (x <= 0 || x >= 255 || y <= 0 || y >= 255) continue;
        GLubyte c[4];
        px(x, y, c);
        if (!is_green(c)) missing++;
    }
    printf("pool-wrap stress (16 KiB x 2 slots): %d/%d quads missing\n", missing, N);
    EXPECT_TRUE(missing == 0);

    /* the tess blob is green too: the interleaved import path kept working */
    {
        GLubyte c[4];
        px(236, 236, c);
        EXPECT_TRUE(c[1] > 100);
    }
    glDeleteProgram(p);
    glDeleteProgram(tp);
    glDeleteVertexArrays(1, &pvao);
    glDeleteVertexArrays(1, &tvao);
    glDeleteBuffers(1, &pvbo);
    glDeleteBuffers(1, &tvbo);
}


/* ===================== round 31 regression tests (CTS geometry_shader fixes) =========== */

static const char *r31_vs = "#version 450 core\n"
    "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n";
static const char *r31_fs = "#version 450 core\n"
    "out vec4 c; void main() { c = vec4(0.2); }\n";

/* a separable geometry-only program links and answers the four geometry queries before it is
 * ever drawn (KHR-GL46.geometry_shader.api.getProgramiv3 builds one with glCreateShaderProgramv) */
TEST(r31_gs_only_spo_queries)
{
    const char *gs =
        "#version 450 core\n"
        "layout(triangles, invocations = 7) in;\n"
        "layout(triangle_strip, max_vertices = 33) out;\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    GLuint s = compile(GL_GEOMETRY_SHADER, gs);
    char log[4096] = "";
    GLsizei len = 0;
    char *src[1] = { (char *)gs };
    GLuint p = glCreateShaderProgramv(GL_GEOMETRY_SHADER, 1, (const char **)src);
    (void)s;
    GLint ls = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    if (!ls) { glGetProgramInfoLog(p, sizeof log, &len, log); printf("spo link: %s\n", log); }
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { glDeleteProgram(p); return; }

    GLint v = -1;
    glGetProgramiv(p, GL_GEOMETRY_INPUT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_TRIANGLES);
    v = -1; glGetProgramiv(p, GL_GEOMETRY_OUTPUT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_TRIANGLE_STRIP);
    v = -1; glGetProgramiv(p, GL_GEOMETRY_VERTICES_OUT, &v);
    EXPECT_EQ_INT(v, 33);
    v = -1; glGetProgramiv(p, GL_GEOMETRY_SHADER_INVOCATIONS, &v);
    EXPECT_EQ_INT(v, 7);
    glDeleteProgram(p);
}

/* every SSBO bound with glBindBufferRange must land in its own slice of the buffer
 * (KHR-GL46.geometry_shader.api.max_shader_storage_blocks binds one buffer 8 times) */
TEST(r31_ssbo_range_bindings)
{
    GLint align = 0, maxb = 0;
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &align);
    glGetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, &maxb);
    EXPECT_TRUE(align >= 4 && maxb >= 2);
    int n = maxb < 8 ? maxb : 8;

    char gs[4096];
    int w = 0;
    w += snprintf(gs + w, sizeof gs - w, "#version 450 core\n"
        "layout(points) in;\nlayout(points, max_vertices = 1) out;\nint counter = 0;\n");
    for (int i = 0; i < n; i++)
        w += snprintf(gs + w, sizeof gs - w,
            "layout(binding = %d) buffer ssb%d { int value; } S_SSB%d;\n", i, i, i);
    w += snprintf(gs + w, sizeof gs - w, "void main() {\n");
    for (int i = 0; i < n; i++)
        w += snprintf(gs + w, sizeof gs - w, "  counter += S_SSB%d.value++;\n", i);
    w += snprintf(gs + w, sizeof gs - w,
        "  gl_Position = vec4(float(counter), 0.0, 0.0, 1.0);\n EmitVertex();\n EndPrimitive();\n}\n");

    GLuint p = program(r31_vs, gs, r31_fs);
    GLint ls = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { glDeleteProgram(p); return; }

    int ints = align / 4;
    GLint data[8 * 256];
    for (int i = 0; i < n; i++) data[i * ints] = i + 1;
    GLuint ssbo = 0;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, n * align, data, GL_DYNAMIC_COPY);
    for (int i = 0; i < n; i++)
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, i, ssbo, i * align, 4);

    GLuint tf = 0, vao = 0;
    glGenBuffers(1, &tf);
    glGenVertexArrays(1, &vao);
    float init[4] = {0,0,0,0};
    glBindBuffer(GL_ARRAY_BUFFER, tf);
    glBufferData(GL_ARRAY_BUFFER, 16, init, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tf);
    glBindVertexArray(vao);
    glUseProgram(p);
    const char *vars[] = {"gl_Position"};
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float *m = glMapBufferRange(GL_ARRAY_BUFFER, 0, 16, GL_MAP_READ_BIT);
    float want = 0;
    for (int i = 0; i < n; i++) want += (float)(i + 1);
    EXPECT_TRUE(m && m[0] == want);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    int *s = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, n * align, GL_MAP_READ_BIT);
    int bad = -1;
    for (int i = 0; i < n; i++)
        if (s && s[i * ints] != i + 2) { bad = i; break; }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    EXPECT_EQ_INT(bad, -1);
    glDeleteProgram(p);
    glDeleteBuffers(1, &ssbo);
    glDeleteBuffers(1, &tf);
    glDeleteVertexArrays(1, &vao);
}

/* blending into a layered 3D attachment: the raster pass must build its pipeline from the
 * pass-through vertex function (no point_size write) with triangle topology
 * (KHR-GL46.geometry_shader.layered_framebuffer.blending_support) */
TEST(r31_layered_blending)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 64) out;\n"
        "void main() {\n"
        "  for (int n = 0; n < 4; ++n) {\n"
        "    gl_Layer = n; gl_Position = vec4( 1,  1, 0, 1); EmitVertex();\n"
        "    gl_Layer = n; gl_Position = vec4( 1, -1, 0, 1); EmitVertex();\n"
        "    gl_Layer = n; gl_Position = vec4(-1,  1, 0, 1); EmitVertex();\n"
        "    gl_Layer = n; gl_Position = vec4(-1, -1, 0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "  }\n"
        "}\n";
    GLuint p = program(r31_vs, gs, r31_fs);
    GLint ls = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { glDeleteProgram(p); return; }

    const int W = 4, H = 4, D = 4;
    unsigned char buf[4][H * W * 4], ref[4][H * W * 4];
    for (int s = 0; s < D; s++)
        for (int n = 0; n < W * H; n++)
        {
            unsigned char *px = &buf[s][n * 4];
            px[0] = 0; px[1] = (unsigned char)(s * 255 / 4);
            px[2] = (unsigned char)(s * 255 / 8); px[3] = (unsigned char)(s * 255 / 12);
            for (int c = 0; c < 4; c++) {
                float d = px[c] / 255.0f, t = d * d + 0.8f * 0.2f;
                if (t < 0) t = 0; if (t > 1) t = 1;
                ref[s][n * 4 + c] = (unsigned char)(t * 255.0f);
            }
        }
    GLuint tex = 0, fbo = 0, vao = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_3D, tex);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, W, H, D);
    for (int s = 0; s < D; s++)
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, s, W, H, 1, GL_RGBA, GL_UNSIGNED_BYTE, buf[s]);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, W, H);
    glBlendFunc(GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR);
    glEnable(GL_BLEND);
    glUseProgram(p);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLuint rf = 0;
    glGenFramebuffers(1, &rf);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, rf);
    for (int layer = 0; layer < D; layer++)
    {
        unsigned char out[H * W * 4];
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, layer);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, out);
        for (int i = 0; i < W * H; i++)
            for (int c = 0; c < 4; c++)
                if (abs((int)out[i * 4 + c] - (int)ref[layer][i * 4 + c]) > 1)
                {
                    printf("layer %d px %d ch %d: %d vs %d\n", layer, i, c,
                           out[i * 4 + c], ref[layer][i * 4 + c]);
                    EXPECT_TRUE(0);
                    c = 4; i = W * H; layer = D;
                }
    }
    glDisable(GL_BLEND);
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &rf);
    glDeleteVertexArrays(1, &vao);
}

/* a framebuffer with no attachments but FRAMEBUFFER_DEFAULT_LAYERS still rasterizes per layer,
 * so the fragment shader's imageStore reaches every layer
 * (KHR-GL46.geometry_shader.layered_rendering_fbo_no_attachment) */
TEST(r31_layered_fbo_no_attachment)
{
    const char *gs =
        "#version 450 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 16) out;\n"
        "out vec2 uv;\n"
        "flat out int layer_id;\n"
        "void main() {\n"
        "  for (int n = 0; n < 4; ++n) {\n"
        "    gl_Position = vec4( 1, -1, 0, 1); gl_Layer = n; layer_id = n; uv = vec2(1, 0); EmitVertex();\n"
        "    gl_Position = vec4( 1,  1, 0, 1); gl_Layer = n; layer_id = n; uv = vec2(1, 1); EmitVertex();\n"
        "    gl_Position = vec4(-1, -1, 0, 1); gl_Layer = n; layer_id = n; uv = vec2(0, 0); EmitVertex();\n"
        "    gl_Position = vec4(-1,  1, 0, 1); gl_Layer = n; layer_id = n; uv = vec2(0, 1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "  }\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "precision highp int;\n"
        "layout(rgba32i) writeonly uniform highp iimage2DArray img;\n"
        "in vec2 uv;\n"
        "flat in int layer_id;\n"
        "void main() { imageStore(img, ivec3(ivec2(uv * 32.0), layer_id), ivec4(0, 255, 0, 0)); }\n";
    GLuint p = program(r31_vs, gs, fs);
    GLint ls = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { glDeleteProgram(p); return; }

    const int W = 32, H = 32, NL = 4;
    GLint *data = malloc(NL * W * H * 4 * sizeof(int));
    for (int i = 0; i < NL * W * H; i++) {
        data[i*4+0] = 255; data[i*4+1] = 0; data[i*4+2] = 0; data[i*4+3] = 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA32I, W, H, NL);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, W, H, NL, GL_RGBA_INTEGER, GL_INT, data);
    free(data);

    GLuint fbo = 0, vao = 0;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, W);
    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, H);
    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_LAYERS, NL);
    GLint q = -1;
    glGetFramebufferParameteriv(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_LAYERS, &q);
    EXPECT_EQ_INT(q, NL);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glBindImageTexture(0, tex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    glViewport(0, 0, W, H);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLint *layer = malloc(W * H * 4 * sizeof(int));
    int bad = 0;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    for (int n = 0; n < NL && !bad; n++)
    {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, n);
        if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { bad = n + 1; break; }
        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
        glReadPixels(0, 0, W, H, GL_RGBA_INTEGER, GL_INT, layer);
        for (int i = 0; i < W * H && !bad; i++)
            if (layer[i*4+0] != 0 || layer[i*4+1] != 255) { bad = n + 1; break; }
    }
    free(layer);
    EXPECT_EQ_INT(bad, 0);
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteVertexArrays(1, &vao);
}


/* KHR-GL46.viewport_array.scissor_clear — glClear of a layered attachment with
 * the scissor test enabled must clear the scissor-0 box on EVERY array layer
 * (the scissored clear is a full-screen draw in MGL; its vertex function must
 * write render_target_array_index and the draw must run once per layer). */
TEST(r52_layered_scissored_clear)
{
    const int W = 8, H = 8, NL = 16;
    GLint *data = malloc(NL * W * H * sizeof(int));
    for (int i = 0; i < NL * W * H; i++) data[i] = -1;

    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32I, W, H, NL, 0, GL_RED_INTEGER, GL_INT, data);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    /* 4x4 grid of 2x2 viewports / scissor boxes over the 8x8 image (CTS shape) */
    GLfloat vps[16 * 4];
    GLint scs[16 * 4];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            int i = y * 4 + x;
            vps[i*4+0] = (float)(x * 2); vps[i*4+1] = (float)(y * 2);
            vps[i*4+2] = 2.0f;           vps[i*4+3] = 2.0f;
            scs[i*4+0] = x * 2; scs[i*4+1] = y * 2;
            scs[i*4+2] = 2;     scs[i*4+3] = 2;
        }
    glViewportArrayv(0, 16, vps);
    glScissorArrayv(0, 16, scs);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, W, H);
    glEnable(GL_SCISSOR_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    memset(data, 0x7f, NL * W * H * sizeof(int));
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED_INTEGER, GL_INT, data);
    int bad_layer = -1;
    for (int n = 0; n < NL && bad_layer < 0; n++)
    {
        const GLint *layer = data + n * W * H;
        for (int y = 0; y < H && bad_layer < 0; y++)
            for (int x = 0; x < W; x++) {
                int inside_scissor0 = (x < 2 && y < 2);
                int want = inside_scissor0 ? 0 : -1;
                if (layer[y * W + x] != want) { bad_layer = n; break; }
            }
    }
    EXPECT_EQ_INT(bad_layer, -1);

    /* full clear path (no scissor): Metal load-action clear must cover all layers too */
    glDisable(GL_SCISSOR_TEST);
    glClear(GL_COLOR_BUFFER_BIT);
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED_INTEGER, GL_INT, data);
    bad_layer = -1;
    for (int n = 0; n < NL * W * H; n++)
        if (data[n] != 0) { bad_layer = n / (W * H); break; }
    EXPECT_EQ_INT(bad_layer, -1);

    free(data);
    glDeleteTextures(1, &tex);
    glDeleteFramebuffers(1, &fbo);
    glDisable(GL_SCISSOR_TEST);
}

/* KHR-GL46.texture_size_promotion.functional — two regressions found r52:
 * (a) sampler2DRect uniforms resolved their texture through the unit's
 *     GL_TEXTURE_2D binding after the Rect->2D MSL rewrite (an unrelated 2D
 *     texture bound at the same unit got sampled instead of the RECTANGLE one);
 * (b) a single-component fragment output ("out float result") wrote 0 to R on
 *     an RGBA attachment (Metal needs every channel; SPIRV-Cross now pads). */
TEST(r52_rect_sampler_and_scalar_out)
{
    const float src[4] = {0.125f, 0.25f, 0.5f, 0.75f};
    GLuint stex = 0, dtex = 0, fbo = 0, prog = 0, vao = 0;
    float pixel = -1.0f;

    const char *vs = "#version 140\nvoid main(){ switch(gl_VertexID){case 0: gl_Position=vec4(-1,-1,0,1);break;case 1: gl_Position=vec4( 1,-1,0,1);break;case 2: gl_Position=vec4(-1, 1,0,1);break;case 3: gl_Position=vec4( 1, 1,0,1);break;} }\n";
    const char *fs[] = {
        "#version 140\nout float result;\nuniform sampler2DRect data;\nvoid main(){ result = texelFetch(data, ivec2(0)).r; }\n",
        "#version 140\nout float result;\nuniform sampler2DRect data;\nvoid main(){ result = texture(data, vec2(0.5, 0.5)).r; }\n",
        "#version 140\nout vec4 result;\nuniform sampler2DRect data;\nvoid main(){ result = vec4(texture(data, vec2(0.5, 0.5)).r, 0.0, 0.0, 1.0); }\n",
    };

    glGenTextures(1, &stex);
    glBindTexture(GL_TEXTURE_RECTANGLE, stex);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R8, 1, 1, 0, GL_RGBA, GL_FLOAT, src);

    glGenTextures(1, &dtex);
    glBindTexture(GL_TEXTURE_2D, dtex);   /* also occupies unit 0's 2D slot: the trap */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_FLOAT, NULL);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dtex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 1, 1);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    for (int v = 0; v < 3; v++)
    {
        GLuint sh = glCreateShader(GL_VERTEX_SHADER), fh = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(sh, 1, &vs, NULL); glCompileShader(sh);
        glShaderSource(fh, 1, &fs[v], NULL); glCompileShader(fh);
        if (prog) glDeleteProgram(prog);
        prog = glCreateProgram();
        glAttachShader(prog, sh); glAttachShader(prog, fh); glLinkProgram(prog);
        glDeleteShader(sh); glDeleteShader(fh);
        GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, 1);
        if (!ok) break;

        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "data"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_RECTANGLE, stex);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        EXPECT_NO_GL_ERROR();
        glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, &pixel);
        EXPECT_NEAR(pixel, 0.125f, 1.0f / 128.0f);
    }

    glDeleteProgram(prog);
    glDeleteTextures(1, &stex);
    glDeleteTextures(1, &dtex);
    glDeleteFramebuffers(1, &fbo);
}

/* the geometry kernel reads image units like the compute path does
 * (KHR-GL46.geometry_shader.api.max_image_uniforms) */
TEST(r31_gs_image_reads)
{
    GLint maximg = 0;
    glGetIntegerv(GL_MAX_GEOMETRY_IMAGE_UNIFORMS, &maximg);
    EXPECT_TRUE(maximg >= 1);
    int n = maximg < 8 ? maximg : 8;

    char gs[4096];
    int w = 0;
    w += snprintf(gs + w, sizeof gs - w, "#version 450 core\n"
        "layout(points) in;\nlayout(points, max_vertices = 1) out;\nint counter = 0;\n");
    for (int i = 0; i < n; i++)
        w += snprintf(gs + w, sizeof gs - w,
            "layout(r32i, binding = %d) readonly uniform highp iimage2D img%d;\n", i, i);
    w += snprintf(gs + w, sizeof gs - w, "void main() {\n");
    for (int i = 0; i < n; i++)
        w += snprintf(gs + w, sizeof gs - w, "  counter += imageLoad(img%d, ivec2(0, 0)).x;\n", i);
    w += snprintf(gs + w, sizeof gs - w,
        "  gl_Position = vec4(float(counter), 0.0, 0.0, 1.0);\n EmitVertex();\n EndPrimitive();\n}\n");

    GLuint p = program(r31_vs, gs, r31_fs);
    GLint ls = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, 1);
    if (!ls) { glDeleteProgram(p); return; }

    GLuint tex[8], fbo[8];
    glGenTextures(n, tex);
    glGenFramebuffers(n, fbo);
    for (int i = 0; i < n; i++)
    {
        GLint one = i + 1;
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32I, 1, 1);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED_INTEGER, GL_INT, &one);
        glBindImageTexture(i, tex[i], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32I);
    }

    GLuint tf = 0, vao = 0;
    glGenBuffers(1, &tf);
    glGenVertexArrays(1, &vao);
    float init[4] = {0,0,0,0};
    glBindBuffer(GL_ARRAY_BUFFER, tf);
    glBufferData(GL_ARRAY_BUFFER, 16, init, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tf);
    glBindVertexArray(vao);
    glUseProgram(p);
    const char *vars[] = {"gl_Position"};
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float want = 0;
    for (int i = 0; i < n; i++) want += (float)(i + 1);
    float *m = glMapBufferRange(GL_ARRAY_BUFFER, 0, 16, GL_MAP_READ_BIT);
    EXPECT_TRUE(m && m[0] == want);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glDeleteProgram(p);
    glDeleteTextures(n, tex);
    glDeleteFramebuffers(n, fbo);
    glDeleteBuffers(1, &tf);
    glDeleteVertexArrays(1, &vao);
}

/* the sample_variables builtins: glslang does not declare gl_NumSamples for desktop GLSL
 * (an OES_sample_variables builtin there), so MGL injects it as the constant GL says it is
 * (baked from the current draw framebuffer's sample count).  KHR-GL46.sample_variables
 * needs all five in one program. */
TEST(sample_variables_builtins)
{
    const char *fs =
        "#version 440 core\n"
        "layout(location = 0) out vec4 o_color;\n"
        "uniform uint mask;\n"
        "void main()\n"
        "{\n"
        "    for (int i = 0; i < (gl_NumSamples + 31) / 32; ++i)\n"
        "        gl_SampleMask[i] = int(mask);\n"
        "    o_color = vec4(gl_SamplePosition, float(gl_SampleID) / 32.0, 1.0);\n"
        "    o_color.a = float(gl_SampleMaskIn[0]) / 255.0;\n"
        "}\n";
    const char *vs =
        "#version 440 core\n"
        "void main() { gl_Position = vec4(0.0); }\n";
    GLuint p = glCreateProgram();
    GLint ls = 0;

    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);
    glDeleteProgram(p);
}

/* KHR-GL46.sample_variables mask.*.samples_0: with SAMPLE_BUFFERS == 0 the fragment
 * shader's gl_SampleMask write must NOT gate rasterization (GL 4.6 13.3: the sample
 * mask output is ignored) — every mask value renders the full quad.  Implemented by
 * specializing the patched MSL with the pipeline's sample count
 * (MGL_RS_FC_PIPELINE_SAMPLES): mgl_rs_sample_mask() drops the shader-written mask
 * only for single-sample pipelines.  (Metal caps MSAA at 4 samples on Apple GPUs, so
 * the CTS's samples_8 cases are honest NotSupported results.) */
TEST(sample_mask_single_sample_ignored)
{
    const char *fs =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o_color;\n"
        "uniform int u_sampleMask;\n"
        "void main()\n"
        "{\n"
        "    for (int i = 0; i < (gl_NumSamples + 31) / 32; ++i)\n"
        "        gl_SampleMask[i] = u_sampleMask & gl_SampleMaskIn[i];\n"
        "    o_color = vec4(1, 0, 0, 1);\n"
        "}\n";
    const char *vs =
        "#version 460 core\n"
        "in vec2 a_position;\n"
        "void main() { gl_Position = vec4(a_position, 0.0, 1.0); }\n";
    GLuint p = glCreateProgram(), tex, fbo;
    GLint ls = 0;

    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ls);
    EXPECT_EQ_INT(ls, GL_TRUE);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 16, 16);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 16, 16);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    float pos[] = { -1, -1, -1, 1, 1, -1, 1, 1 };
    static const GLushort idx[] = { 0, 1, 2, 2, 1, 3 };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glUseProgram(p);
    GLint loc = glGetUniformLocation(p, "u_sampleMask");
    const int masks[] = { 0, 1, 2, 3, 4, 5, 6, 7, 0 };
    for (unsigned m = 0; m < sizeof masks / sizeof masks[0]; m++) {
        glClearBufferfv(GL_COLOR, 0, (const float[4]){0, 1, 0, 1});
        glUniform1i(loc, masks[m]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, idx);
        int bad = 0;
        GLubyte px[4];
        for (int y = 2; y < 14 && !bad; y++) {
            mgl_read_rgba(8, y, px);
            if (px[0] < 250 || px[1] > 5) bad++;
        }
        if (bad) printf("   mask=%d: (%d,%d,%d,%d)\n", masks[m], px[0], px[1], px[2], px[3]);
        EXPECT_EQ_INT(bad, 0);
    }
    EXPECT_NO_GL_ERROR();
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, 512, 512);
}

/* r48: SSBO block offsets after glslang's implicit array growth.
 *
 * A non-final unsized buffer-block member is sized by the largest constant index
 * that touches it (glslang's updateImplicitArraySize), but glslang does not move the
 * offsets of the members that follow it — the block then self-overlaps and the second
 * runtime array reads the wrong bytes.  mgl_spirv_fix_block_offsets pushes the later
 * members past the grown array's declared span (KHR-GL46.shader_storage_buffer_object
 * basic-syntax / basic-syntaxSSO variant 8). */
TEST(r48_ssbo_two_runtime_arrays)
{
    static const char *vs =
        "#version 460 core\n"
        "layout(std430) coherent buffer Buffer {\n"
        "  coherent vec4 position01[];\n"
        "  vec4 position2[];\n"
        "} g_buffer;\n"
        "void main() {\n"
        "  switch (gl_VertexID) {\n"
        "    case 0: gl_Position = g_buffer.position01[0]; break;\n"
        "    case 1: gl_Position = g_buffer.position01[1]; break;\n"
        "    case 2: gl_Position = g_buffer.position2[gl_VertexID - 2]; break;\n"
        "  }\n"
        "}\n";
    static const char *fs =
        "#version 460 core\nlayout(location = 0) out vec4 o_color;\n"
        "void main() { o_color = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const float data[12] = {-1.0f, -1.0f, 0.0f, 1.0f,   /* position01[0] */
                            3.0f, -1.0f, 0.0f, 1.0f,    /* position01[1] */
                            -1.0f, 3.0f, 0.0f, 1.0f};   /* position2[0]  */
    GLuint vao, buf, p = mgl_make_program(vs, fs);

    if (!p) return;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof data, data, GL_STATIC_DRAW);

    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    /* the whole viewport is covered only when position2[0] reads the third vec4 */
    GLubyte px[4];
    mgl_read_rgba(0, 0, px);
    mgl_read_rgba(1, 1, px);
    EXPECT_TRUE(px[1] == 255 && px[0] == 0 && px[2] == 0);

    glDeleteBuffers(1, &buf);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

/* r48: the SSBO binding-point range and std430-on-uniform-block are compile-time
 * errors (glslang does not check them for OpenGL clients) */
TEST(r48_ssbo_layout_compile_errors)
{
    const char *bad1[] = { "#version 460 core\n",
                           "layout(binding = 8) buffer Buffer { int x; };\n"
                           "void main() { x = 0; }" };
    const char *bad2[] = { "#version 460 core\n",
                           "layout(binding = 6) buffer Buffer { int x; } g[4];\n"
                           "void main() { g[0].x = 0; }" };
    const char *bad3[] = { "#version 460 core\n",
                           "layout(std430) uniform UBO { int x; };\n"
                           "void main() { }" };
    const char *good[] = { "#version 460 core\n",
                           "layout(std430, binding = 2) readonly buffer Buffer { int x; } g[4];\n"
                           "void main() { if (g[0].x == 0) discard; }" };
    GLuint s;

    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 2, bad1, NULL);
    glCompileShader(s);
    { GLint ok_ = 1; glGetShaderiv(s, GL_COMPILE_STATUS, &ok_); EXPECT_TRUE(ok_ == 0); }
    glDeleteShader(s);

    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 2, bad2, NULL);
    glCompileShader(s);
    { GLint ok_ = 1; glGetShaderiv(s, GL_COMPILE_STATUS, &ok_); EXPECT_TRUE(ok_ == 0); }
    glDeleteShader(s);

    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 2, bad3, NULL);
    glCompileShader(s);
    { GLint ok_ = 1; glGetShaderiv(s, GL_COMPILE_STATUS, &ok_); EXPECT_TRUE(ok_ == 0); }
    glDeleteShader(s);

    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 2, good, NULL);
    glCompileShader(s);
    { GLint ok_ = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok_); EXPECT_TRUE(ok_ == 1); }
    glDeleteShader(s);
}

/* r48: TES-side SSBO writes must land in the GL buffer.  The evaluation shader is
 * the render pipeline's vertex function, so its SSBO slots start at 0 — the vertex
 * array's attribute streams may not be bound to the render encoder, and a small SSBO
 * must alias the GL buffer instead of the read-only argument ring
 * (KHR-GL46.shader_storage_buffer_object basic-atomic-case2). */
TEST(r48_tes_ssbo_atomics)
{
    static const char *vs = "#version 460 core\n"
                            "layout(location = 0) in vec4 p;\n"
                            "void main() { gl_Position = p; }\n";
    static const char *tcs = "#version 460 core\n"
                             "layout(vertices = 1) out;\n"
                             "void main() {\n"
                             "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
                             "  if (gl_InvocationID == 0) {\n"
                             "    gl_TessLevelInner[0] = 1.0;\n"
                             "    gl_TessLevelInner[1] = 1.0;\n"
                             "    gl_TessLevelOuter[0] = 1.0;\n"
                             "    gl_TessLevelOuter[1] = 1.0;\n"
                             "    gl_TessLevelOuter[2] = 1.0;\n"
                             "    gl_TessLevelOuter[3] = 1.0;\n"
                             "  }\n"
                             "}\n";
    static const char *tes = "#version 460 core\n"
                             "layout(quads, point_mode) in;\n"
                             "layout(std430, binding = 0) buffer TCSuint { uint out_u[1]; };\n"
                             "layout(std430, binding = 1) buffer TCSint { int out_i[1]; };\n"
                             "void main() {\n"
                             "  gl_Position = gl_in[0].gl_Position;\n"
                             "  out_u[0] = 7u;\n"
                             "  out_i[0] = 9;\n"
                             "}\n";
    static const char *fs = "#version 460 core\n"
                            "layout(location = 0) out vec4 o;\n"
                            "void main() { o = vec4(0, 1, 0, 1); }\n";
    GLuint vao, vbo, sbuf[2], p;
    GLuint v = glCreateShader(GL_VERTEX_SHADER), tc = glCreateShader(GL_TESS_CONTROL_SHADER),
           te = glCreateShader(GL_TESS_EVALUATION_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    p = glCreateProgram();
    glShaderSource(v, 1, &vs, NULL);  glCompileShader(v);
    glShaderSource(tc, 1, &tcs, NULL); glCompileShader(tc);
    glShaderSource(te, 1, &tes, NULL); glCompileShader(te);
    glShaderSource(f, 1, &fs, NULL);  glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, tc); glAttachShader(p, te); glAttachShader(p, f);
    for (GLuint sh_ = 0; sh_ < 4; sh_++)
    {
        GLuint so_ = (GLuint[]){v, tc, te, f}[sh_];
        GLint ok_ = 0; glGetShaderiv(so_, GL_COMPILE_STATUS, &ok_);
        if (!ok_) { char log_[2048] = ""; glGetShaderInfoLog(so_, sizeof log_, NULL, log_);
                    MGL_FAIL("shader %u compile: %s", so_, log_); return; }
    }
    glLinkProgram(p);
    { GLint ok_ = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok_);
      if (!ok_) { char log_[2048] = ""; glGetProgramInfoLog(p, sizeof log_, NULL, log_);
                  MGL_FAIL("link: %s", log_); return; } }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    {
        const float quad[2] = {0.0f, 0.0f};
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    }
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(2, sbuf);
    for (int i = 0; i < 2; i++)
    {
        const GLuint zero = 0;
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, sbuf[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 4, &zero, GL_DYNAMIC_DRAW);
    }

    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    glDrawArrays(GL_PATCHES, 0, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    {
        GLuint u = 0, i = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbuf[0]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &u);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbuf[1]);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 4, &i);
        EXPECT_TRUE(u == 7 && i == 9);
    }

    glDeleteBuffers(2, sbuf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    glDeleteShader(v); glDeleteShader(tc); glDeleteShader(te); glDeleteShader(f);
}

/* r54: MSL's gl_VertexID / gl_InstanceID / gl_PrimitiveID are `uint' while GL declares
 * them `int'; SPIRV-Cross emits them uncast, so `min(4, gl_VertexID)' compiles to an
 * ambiguous Metal overload (KHR-GL46.draw_indirect.advanced-twoPass-transformFeedback-*).
 * Exercise the main link path AND the transform-feedback capture variant. */
TEST(r54_int_builtins_minmax)
{
    const char *vs =
        "#version 400 core\n"
        "flat out uint dataOut;\n"
        "layout(std140) uniform BLOCK { uint m[5]; } b;\n"
        "in vec4 i_vertex;\n"
        "void main() {\n"
        "  dataOut = b.m[min(4, gl_VertexID)] + max(0, 10 - gl_VertexID);\n"
        "  gl_Position = i_vertex;\n"
        "}";
    const char *fs =
        "#version 400 core\n"
        "flat in uint dataOut; out vec4 c;\n"
        "void main() { c = vec4(float(dataOut) / 64.0, 0.0, 0.0, 1.0); }";

    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    static const char *vars[2] = { "dataOut", NULL };
    glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint ubo = 0;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    /* std140 uint array: one uint per 16-byte slot */
    GLuint m5[5 * 4] = { 0 };
    m5[0] = 10; m5[4] = 20; m5[8] = 30; m5[12] = 40; m5[16] = 50;
    glBufferData(GL_UNIFORM_BUFFER, sizeof m5, m5, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

    GLuint tbuf = 0;
    glGenBuffers(1, &tbuf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    /* gl_VertexID has no base-vertex here: 0,1,2 -> m[0..2] = 10,20,30 and +10,+9,+8 */
    float quad[12] = { -1, -1, 1, -1, 1, 1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    GLuint out[4] = { 0, 0, 0, 0 };
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    EXPECT_EQ_INT(out[0], 20);
    EXPECT_EQ_INT(out[1], 29);
    EXPECT_EQ_INT(out[2], 38);

    /* rasterized path of the same program: the pipeline must also compile */
    glDisable(GL_RASTERIZER_DISCARD);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(p);
    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(1, &tbuf);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* r54: gl_PrimitiveID (int in GL, uint in MSL) in an int min() in the fragment stage */
TEST(r54_fs_primitiveid_min)
{
    const char *vs =
        "#version 400 core\n"
        "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
    const char *fs =
        "#version 400 core\n"
        "out vec4 c;\n"
        "void main() { c = vec4(float(min(gl_PrimitiveID, 1)) + 0.25, 0.0, 0.0, 1.0); }";
    GLuint p = program(vs, NULL, fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return;

    GLuint fbo = 0, tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 8, 8);
    glUseProgram(p);
    /* one primitive: gl_PrimitiveID = 0 -> min(0,1) = 0 -> 0.25 red */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 64, 0, 0, 255, 2);

    glDeleteProgram(p);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
}

/* r54: readback of an sRGB-encoded color attachment must return the RAW encoded
 * bytes — the GL sRGB decoder applies only when sampling from a shader.  The r52
 * "srgb_read" linearization for FLOAT clients broke
 * KHR-GL46.packed_pixels.rectangle.srgb8 / pbo_rectangle.srgb8 (and 10 more). */
TEST(r54_srgb_readback_raw)
{
    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_RECTANGLE, tex);
    GLubyte data[4 * 4 * 3];
    for (int i = 0; i < 16; i++)
    {
        data[i * 3 + 0] = (GLubyte)(i * 16);
        data[i * 3 + 1] = (GLubyte)(255 - i * 16);
        data[i * 3 + 2] = 36;
    }
    glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_SRGB8, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    float out[16 * 4] = {0};
    glReadPixels(0, 0, 4, 4, GL_BGRA, GL_FLOAT, out);
    EXPECT_NO_GL_ERROR();
    /* texel 0: stored bytes (0, 255, 36) — GL_BGRA float read gives (36/255, 1.0, 0, 1) */
    EXPECT_NEAR(out[0], 36 / 255.0, 1e-5);
    EXPECT_NEAR(out[1], 1.0, 1e-5);
    EXPECT_NEAR(out[2], 0.0, 1e-5);
    EXPECT_NEAR(out[3], 1.0, 1e-5);
    /* texel 2: stored (32, 223, 36) */
    EXPECT_NEAR(out[8], 36 / 255.0, 1e-5);
    EXPECT_NEAR(out[9], 223 / 255.0, 1e-5);
    EXPECT_NEAR(out[10], 32 / 255.0, 1e-5);

    /* the PBO path must match */
    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, sizeof out, NULL, GL_STREAM_READ);
    glReadPixels(0, 0, 4, 4, GL_BGRA, GL_FLOAT, NULL);
    EXPECT_NO_GL_ERROR();
    float *m = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, sizeof out, GL_MAP_READ_BIT);
    EXPECT_TRUE(m != NULL);
    if (m)
    {
        EXPECT_NEAR(m[0], 36 / 255.0, 1e-5);
        EXPECT_NEAR(m[1], 1.0, 1e-5);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    glDeleteBuffers(1, &pbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

/* r54: the specific compressed internal formats (table 8.5) are not supported for
 * 3-dimensional textures — glTexImage3D on TEXTURE_3D / TEXTURE_2D_ARRAY with an
 * RGTC (or BPTC/S3TC/ETC2) internalformat is INVALID_OPERATION
 * (KHR-GL46.packed_pixels.varied_rectangle.compressed_*_rgtc*, whose
 * GL_UNPACK_IMAGE_HEIGHT cases upload to GL_TEXTURE_2D_ARRAY).  2D targets stay
 * legal (uncompressed client data takes the decoded-storage path). */
TEST(r54_compressed_no_3d)
{
    /* GL 4.6 spec, TexImage3D errors: RGTC/EAC/ETC2 compressed internalformats
     * are INVALID_OPERATION unless target is TEXTURE_2D_ARRAY (Mesa
     * _mesa_target_can_be_compressed); S3TC has no 3D support either, BPTC has
     * but cannot be online-compressed from GL_RED client data here.  On
     * TEXTURE_2D_ARRAY the spec-valid combos must SUCCEED —
     * KHR-GL46.copy_image.incompatible_formats_compression uploads an
     * uncompressed (online-compressed) RGTC2 to a 2D array (compat s27):
     *   - RGTC1/SIGNED_RGTC1 + external RED/UNSIGNED_BYTE  -> no error
     *   - RGTC2/SIGNED_RGTC2 + RED (external is RG)        -> format mismatch
     *   - S3TC/BPTC + RED (external RGBA / none)           -> no online compression
     */
    const GLenum bad[] = {
        GL_COMPRESSED_RED_RGTC1, GL_COMPRESSED_SIGNED_RED_RGTC1,
        GL_COMPRESSED_RG_RGTC2, GL_COMPRESSED_SIGNED_RG_RGTC2,
        GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_BPTC_UNORM,
    };
    GLubyte data[64];
    for (int i = 0; i < 64; i++) data[i] = (GLubyte)(i * 3);

    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++)
    {
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
        while (glGetError() != GL_NO_ERROR) {}
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, bad[i], 4, 4, 2, 0, GL_RED, GL_UNSIGNED_BYTE, data);
        /* all four RGTC formats online-compress (decoded storage); the CTS
         * packed_pixels isFormatValid accepts a client format with fewer
         * components than the base format (GL_RED into an RG base), so these
         * all succeed.  S3TC/BPTC have no CPU encoder -> "no online
         * compression" INVALID_OPERATION. */
        if (bad[i] == GL_COMPRESSED_RGB_S3TC_DXT1_EXT || bad[i] == GL_COMPRESSED_RGBA_BPTC_UNORM)
            EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
        else
            EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
        glBindTexture(GL_TEXTURE_3D, tex);
        glTexImage3D(GL_TEXTURE_3D, 0, bad[i], 4, 4, 2, 0, GL_RED, GL_UNSIGNED_BYTE, data);
        EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
        /* 2D must still succeed for the RGTC formats (online compression ->
         * decoded storage; the CTS rectangle.compressed_*_rgtc1 cases rely on it);
         * S3TC/BPTC stay rejected on RECTANGLE like the GL requires */
        glDeleteTextures(1, &tex);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_RECTANGLE, tex);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, bad[i], 4, 4, 0, GL_RED, GL_UNSIGNED_BYTE, data);
        GLenum e2d = glGetError();
        if (bad[i] == GL_COMPRESSED_RED_RGTC1 || bad[i] == GL_COMPRESSED_SIGNED_RED_RGTC1 ||
            bad[i] == GL_COMPRESSED_RG_RGTC2 || bad[i] == GL_COMPRESSED_SIGNED_RG_RGTC2)
            EXPECT_EQ_INT(e2d, GL_NO_ERROR);
        else
            EXPECT_TRUE(e2d == GL_NO_ERROR || e2d == GL_INVALID_OPERATION);
        glDeleteTextures(1, &tex);
    }
}

/* r55: glDrawElementsIndirect through the geometry emulation — the prep kernel must
 * honour the command's firstIndex (the element buffer binds at its base for indirect
 * draws, so the assembly read vertex data as indices: garbage vmin/vmax truncated the
 * emission), and the output budget must cover the command's primcount (the GPU-only
 * sizing assumed a single instance).  One shared buffer carries vertices, indices and
 * both indirect commands at distinct offsets, exactly like the CTS. */
TEST(r55_indirect_firstindex_and_primcount)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    static const float vertex_data[] = {
        -0.1f, 0.2f, 0.2f, -0.7f, 0.5f, -0.5f, 0.1f, -0.2f, -0.2f, 0.7f, -0.5f, 0.5f,
    };
    static const GLuint index_data[] = { 2, 1, 0 };

    size_t vsize = sizeof vertex_data, isize = sizeof index_data;
    GLuint voff = 0, ioff = (GLuint)vsize, aoff = ioff + (GLuint)isize, eoff = aoff + 16;
    GLuint firstindex = ioff / 4;

    GLuint bo;
    glGenBuffers(1, &bo);
    glBindBuffer(GL_ARRAY_BUFFER, bo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bo);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bo);
    glBufferData(GL_ARRAY_BUFFER, eoff + 20, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, voff, vsize, vertex_data);
    glBufferSubData(GL_ARRAY_BUFFER, ioff, isize, index_data);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)voff);
    glEnableVertexAttribArray(0);

    struct { GLenum mode; GLuint count; const char *in; } cases[] = {
        { GL_POINTS, 1, "points" },
        { GL_LINES, 2, "lines" },
        { GL_LINES_ADJACENCY, 4, "lines_adjacency" },
        { GL_TRIANGLES, 3, "triangles" },
        { GL_TRIANGLES_ADJACENCY, 6, "triangles_adjacency" },
    };

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        GLuint arrays_args[4] = { cases[c].count, 4, 0, 1 };
        GLuint elements_args[5] = { cases[c].count, 4, firstindex, 1, 1 };
        glBufferSubData(GL_ARRAY_BUFFER, aoff, 16, arrays_args);
        glBufferSubData(GL_ARRAY_BUFFER, eoff, 20, elements_args);

        /* points -> 1 vertex per emit; the GS emits 3 primitives per invocation and the
         * command carries primcount=4: 12 emitted primitives, 12 raster points */
        char gs[1024];
        snprintf(gs, sizeof gs,
            "#version 150\n"
            "layout(%s) in;\n"
            "layout(points, max_vertices=8) out;\n"
            "void main() {\n"
            "    for (int i = 0; i < 3; i++) {\n"
            "        gl_Position = vec4(-0.9 + 0.3 * float(i), 0.25, 0.0, 1.0);\n"
            "        EmitVertex();\n"
            "    }\n"
            "    EndPrimitive();\n"
            "}\n", cases[c].in);
        GLuint prog = glCreateProgram();
        GLuint vs = glCreateShader(GL_VERTEX_SHADER), g = glCreateShader(GL_GEOMETRY_SHADER),
               fs = glCreateShader(GL_FRAGMENT_SHADER);
        const char *vsrc = "#version 150\nin vec4 i_p; void main() { gl_Position = i_p; }\n";
        const char *fsrc = "#version 150\nout vec4 c; void main() { c = vec4(1.0); }\n";
        glShaderSource(vs, 1, &vsrc, NULL); glCompileShader(vs);
        const char *gsrc = gs;
        glShaderSource(g, 1, &gsrc, NULL); glCompileShader(g);
        glShaderSource(fs, 1, &fsrc, NULL); glCompileShader(fs);
        glAttachShader(prog, vs); glAttachShader(prog, g); glAttachShader(prog, fs);
        glLinkProgram(prog);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok)
        {
            char log[2048];
            glGetProgramInfoLog(prog, sizeof log, NULL, log);
            glDeleteProgram(prog);
            MGL_FAIL("%s: link failed: %s", cases[c].in, log);
        }
        glUseProgram(prog);

        glClear(GL_COLOR_BUFFER_BIT);
        GLuint q = 0;
        glGenQueries(1, &q);
        glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q);
        glDrawElementsIndirect(cases[c].mode, GL_UNSIGNED_INT, (void *)(uintptr_t)eoff);
        glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
        GLuint64 emitted = 0;
        glGetQueryObjectui64v(q, GL_QUERY_RESULT, &emitted);
        GLuint64 invocations = 0;
        GLuint qi = 0;
        glGenQueries(1, &qi);      /* a query object's target is immutable: separate object */
        glBeginQuery(GL_GEOMETRY_SHADER_INVOCATIONS, qi);
        glDrawElementsIndirect(cases[c].mode, GL_UNSIGNED_INT, (void *)(uintptr_t)eoff);
        glEndQuery(GL_GEOMETRY_SHADER_INVOCATIONS);
        glGetQueryObjectui64v(qi, GL_QUERY_RESULT, &invocations);
        glDeleteQueries(1, &q);
        glDeleteQueries(1, &qi);
        glDeleteProgram(prog);

        EXPECT_EQ_INT(emitted, 12);
        EXPECT_EQ_INT(invocations, 4);
    }

    glDeleteBuffers(1, &bo);
    glDeleteVertexArrays(1, &vao);
}

/* ---- r99: the capture pass must feed the same vertex fetch plan the raster pass feeds
 * (converted shadow streams, current generic values, renumbered fp64 word slots) ---- */

static const char *r99_pass_tri_gs =
    "#version 450 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "layout(location = 0) in vec4 v_col[];\n"
    "layout(location = 0) out vec4 g_col;\n"
    "void main(){\n"
    "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_col = v_col[i]; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";

static const char *r99_col_fs =
    "#version 450 core\n"
    "layout(location = 0) in vec4 g_col;\n"
    "layout(location = 0) out vec4 color;\n"
    "void main(){ color = g_col; }\n";

static const float r99_quad[] = { -1.0f, -1.0f,   1.0f, -1.0f,   -1.0f, 1.0f,   1.0f, 1.0f };

/* vs must output the tested color at location 0; setup() specifies attribute 1 on the
 * bound VAO.  Renders with and without the geometry shader; both must give exp. */
static void r99_draw_case(const char *vs_src, const char *name, int exp_r, int exp_g, int exp_b,
                          void (*setup)(void))
{
    GLuint p_nogs = program(vs_src, NULL, r99_col_fs);
    GLuint p_gs   = program(vs_src, r99_pass_tri_gs, r99_col_fs);
    GLuint vao, vbo;

    (void)name;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    /* earlier tests leave an 8x8 viewport / scissor / discard behind: reset the bits
     * this case depends on (it reads the full-screen quad at (32,32)) */
    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, 64, 64);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof r99_quad, r99_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
    if (setup) setup();

    for (int with_gs = 0; with_gs <= 1; with_gs++)
    {
        GLuint p = with_gs ? p_gs : p_nogs;
        glUseProgram(p);
        glClearColor(0, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        EXPECT_PIXEL(32, 32, exp_r, exp_g, exp_b, 255, 4);   /* gs and no-gs must agree */
    }
    glUseProgram(0);
    glDeleteProgram(p_nogs);
    glDeleteProgram(p_gs);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    EXPECT_NO_GL_ERROR();
}

TEST(r99_gs_const_attrib)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in vec4 col;\n"
        "layout(location=0) out vec4 v_col;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_col = col; }\n";
    /* attribute 1 stays disabled: the current generic value must reach the capture
     * variant (const slot 29 on the capture encoder) */
    glVertexAttrib4f(1, 0.25f, 0.5f, 0.75f, 1.0f);
    r99_draw_case(vs, "r99_gs_const_attrib", 64, 128, 191, NULL);
}

static unsigned char r99_mis[4 * 10];
static void r99_setup_misaligned(void)
{
    for (int i = 0; i < 4; i++)
    {
        float xy[2] = { 1.0f, 0.5f };
        memcpy(&r99_mis[i * 10], xy, 8);
    }
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof r99_mis, r99_mis, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    /* stride 10: not a multiple of 4 -> fixed-function fetch impossible, the CPU
     * converted (shadow) stream must be bound on the capture encoder */
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 10, (void *)0);
}

TEST(r99_gs_shadow_attrib)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in vec2 v;\n"
        "layout(location=0) out vec4 v_col;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_col = vec4(v.x, v.y, 0.5, 1.0); }\n";
    r99_draw_case(vs, "r99_gs_shadow_attrib", 255, 128, 128, r99_setup_misaligned);
}

static double r99_dbls[4 * 2];
static void r99_setup_dvec4_short(void)
{
    for (int i = 0; i < 4; i++) { r99_dbls[i * 2] = 0.25; r99_dbls[i * 2 + 1] = 0.5; }
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof r99_dbls, r99_dbls, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    /* size-2 double array feeding a dvec4: z,w come from the current generic value,
     * and the fp64 word slots are renumbered past the plain inputs — the capture
     * variant must be pinned to the program's slots (r99 pin fix) */
    glVertexAttribLPointer(1, 2, GL_DOUBLE, 16, (void *)0);
    glVertexAttribL4d(1, 9.0, 9.0, 0.75, 1.0);
}

TEST(r99_gs_dvec4_padded)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in dvec4 d;\n"
        "layout(location=0) out vec4 v_col;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_col = vec4(float(d.x), float(d.y), float(d.z), 1.0); }\n";
    r99_draw_case(vs, "r99_gs_dvec4_padded", 64, 128, 191, r99_setup_dvec4_short);
}

/* ---- r100: the capture-pass setup cache must hit on steady draws and miss (and still
 * render correctly) whenever an input of the cached map/descriptor/plan changes ---- */

static void r100_draw_once(GLuint p, int exp_r, int exp_g, int exp_b, const char *tag)
{
    glUseProgram(p);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_PIXEL(32, 32, exp_r, exp_g, exp_b, 255, 4);
    (void)tag;
}

TEST(r100_capcache_invalidation)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in vec2 v;\n"
        "layout(location=0) out vec4 v_col;\n"
        "layout(std140, binding=0) uniform UBO { vec4 tint; };\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_col = vec4(v.x, v.y, 0.5, 1.0) * tint; }\n";
    GLuint p_gs = program(vs, r99_pass_tri_gs, r99_col_fs);
    GLuint vao, vbo, ubo;

    glDisable(GL_RASTERIZER_DISCARD);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, 64, 64);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ubo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof r99_quad, r99_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
    /* attribute 1: aligned stream, converted (shadow) is not needed — plain direct */
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof r99_quad, r99_quad, GL_STATIC_DRAW);   /* separate VBO below */
    GLuint vbo2;
    glGenBuffers(1, &vbo2);
    static const float ones[8] = { 1,1, 1,1, 1,1, 1,1 };
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glBufferData(GL_ARRAY_BUFFER, sizeof ones, ones, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    {
        float tint[4] = { 1, 1, 1, 1 };
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof tint, tint, GL_STATIC_DRAW);
    }
    glUseProgram(p_gs);

    uint64_t m0 = mgl_geom_capcache_stat(1);
    r100_draw_once(p_gs, 255, 255, 128, "first");
    uint64_t h1 = mgl_geom_capcache_stat(0), m1 = mgl_geom_capcache_stat(1);
    EXPECT_TRUE(m1 > m0);                              /* the first draw is always a miss */
    /* steady state: nothing changed, every following draw must hit the snapshot */
    for (int i = 0; i < 4; i++) r100_draw_once(p_gs, 255, 255, 128, "steady");
    uint64_t h2 = mgl_geom_capcache_stat(0), m2 = mgl_geom_capcache_stat(1);
    EXPECT_TRUE(h2 > h1);
    EXPECT_EQ_INT((int)m2, (int)m1);                   /* no extra miss while state holds */

    /* (a) attribute-format change: the fetch plan / descriptor must be rebuilt —
     * re-upload attribute 1's data packed at stride 10 first, so the value stays
     * (1,1) after the misaligned (CPU-converted) layout takes over */
    {
        unsigned char packed[4 * 10];
        for (int i = 0; i < 4; i++)
        {
            float xy[2] = { 1.0f, 1.0f };
            memcpy(&packed[i * 10], xy, 8);
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo2);
        glBufferData(GL_ARRAY_BUFFER, sizeof packed, packed, GL_STATIC_DRAW);
    }
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 10, (void *)0);   /* misaligned: shadow */
    r100_draw_once(p_gs, 255, 255, 128, "fmt");
    uint64_t h3 = mgl_geom_capcache_stat(0), m3 = mgl_geom_capcache_stat(1);
    EXPECT_TRUE(m3 > m2);
    (void)h3;

    /* (b) UBO rebind at the same index with different content: the map must rebuild
     * (the row hash catches the pointer change) and the new uniform must apply */
    GLuint ubo2;
    glGenBuffers(1, &ubo2);
    {
        float tint[4] = { 0.5f, 1, 1, 1 };
        glBindBuffer(GL_UNIFORM_BUFFER, ubo2);
        glBufferData(GL_UNIFORM_BUFFER, sizeof tint, tint, GL_STATIC_DRAW);
    }
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo2);
    r100_draw_once(p_gs, 128, 255, 128, "ubo");
    EXPECT_TRUE(mgl_geom_capcache_stat(1) > m3);

    glDeleteBuffers(1, &ubo2);
    glDeleteBuffers(1, &vbo2);
    glDeleteBuffers(1, &ubo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p_gs);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    /* TEST_ONLY=<name> runs a single test (debugging a state cascade between tests) */
    const char *only = getenv("TEST_ONLY");
#define RUN_IF(name) do { if (!only || !strcmp(only, #name)) RUN(name); } while (0)
    RUN_IF(pool_wrap_stress_tiny);     /* first: forces the tiny pools for the whole binary */
    RUN_IF(block_vs_fs_control);
    RUN_IF(gs_struct_interface_vars);
    RUN_IF(gs_struct_member_order);
    RUN_IF(gs_named_output_block_roundtrip);
    RUN_IF(control_no_geometry_shader);
    RUN_IF(control_varying_no_gs);
    RUN_IF(passthrough_triangles);
    RUN_IF(point_to_quad_amplification);
    RUN_IF(primitive_id_and_invocations);
    RUN_IF(lines_adjacency_input);
    RUN_IF(simple_varying_from_gs);
    RUN_IF(draw_mode_mismatch);
    RUN_IF(link_error_no_varying);
    RUN_IF(end_primitive_splitting);
    RUN_IF(amplification_max_vertices);
    RUN_IF(triangles_adjacency_input);
    RUN_IF(indexed_and_instanced);
    RUN_IF(indirect_draws);
    RUN_IF(layered_rendering);
    RUN_IF(viewport_index);
    RUN_IF(separable_pipeline_with_gs);
    RUN_IF(primitive_queries);
    RUN_IF(xfb_of_geometry_output);
    RUN_IF(xfb_streams);
    RUN_IF(sample_variables_builtins);
    RUN_IF(sample_mask_single_sample_ignored);
    RUN_IF(xfb_multi_stream_capture);
    RUN_IF(xfb_multi_stream_instanced);
    RUN_IF(xfb_strip_decomposition);
    RUN_IF(pipeline_vs_gs_capture);
    RUN_IF(vs_conditional_declarations);
    RUN_IF(gs_atomic_counter_link_fails);
    RUN_IF(xfb_arrays_of_blocks_xfb_offset);
    RUN_IF(tessellation_triangles_into_gs);
    RUN_IF(tessellation_indexed_into_gs);
    RUN_IF(tessellation_isolines_into_gs);
    RUN_IF(tessellation_point_mode_into_gs);
    RUN_IF(xfb_of_tes_output);
    RUN_IF(xfb_of_tes_gs_output);
    RUN_IF(primitive_restart_points_gl_primitive_id_in);
    RUN_IF(primitive_restart_with_gs);
    RUN_IF(pool_rotation_hazard);
    RUN_IF(wild_index_survives_cpu_assembly);
    RUN_IF(xfb_gpu_pack_multiple_draws_and_pause);
    RUN_IF(element_store_tail_integrity);
    RUN_IF(tiled_restart_gpu_assembly);
    RUN_IF(gs_output_block_instance_name);
    RUN_IF(gs_output_block_two_members_and_plain);
    RUN_IF(gs_output_block_array_member);
    RUN_IF(gs_output_block_flat_and_matrix_members);
    RUN_IF(gs_output_block_anonymous);
    RUN_IF(xfb_of_gs_output_block_members);
    RUN_IF(gs_output_block_after_tessellation);
    RUN_IF(gs_output_block_xfb_errors);
    RUN_IF(r31_gs_only_spo_queries);
    RUN_IF(r31_ssbo_range_bindings);
    RUN_IF(r31_layered_blending);
    RUN_IF(r31_layered_fbo_no_attachment);
    RUN_IF(r31_gs_image_reads);
    RUN_IF(r48_ssbo_two_runtime_arrays);
    RUN_IF(r48_ssbo_layout_compile_errors);
    RUN_IF(r48_tes_ssbo_atomics);
    RUN_IF(r52_layered_scissored_clear);
    RUN_IF(r52_rect_sampler_and_scalar_out);
    RUN_IF(r54_int_builtins_minmax);
    RUN_IF(r54_fs_primitiveid_min);
    RUN_IF(r54_srgb_readback_raw);
    RUN_IF(r54_compressed_no_3d);
    RUN_IF(r55_indirect_firstindex_and_primcount);
    RUN_IF(r99_gs_const_attrib);
    RUN_IF(r99_gs_shadow_attrib);
    RUN_IF(r99_gs_dvec4_padded);
    RUN_IF(r100_capcache_invalidation);
    return mgl_test_summary();
}
