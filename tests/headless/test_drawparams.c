/* test_drawparams — GL_ARB_shader_draw_parameters / core 4.60
 * (KHR-GL46.shader_draw_parameters_tests, 13/13 Pass)
 *
 * gl_BaseVertex/gl_BaseInstance/gl_DrawID reach the vertex stage through
 * tessellation.c's draw-parameters buffer: Metal's [[base_vertex]] equals
 * vertexStart (GL's `first') for arrays draws where GL specifies 0, and
 * BuiltIn DrawIndex is refused by SPIRV-Cross's MSL backend entirely
 * (demoted to Private in tess_spv_lower.c, value from the buffer).
 * draw_buffers.c stamps the id/base-vertex per single draw — including every
 * record of a multi-draw(-indirect) — and MGLRenderer+Draw.m binds the bytes.
 */
#include "mgl_test.h"

static const char *vs = R"(#version 460
in vec2 a_p;
out vec3 v_dp;
void main()
{
    v_dp = vec3(float(gl_BaseVertex), float(gl_BaseInstance), float(gl_DrawID));
    gl_Position = vec4(a_p, 0.0, 1.0);
}
)";
static const char *fs = R"(#version 460
in vec3 v_dp;
out vec4 col;
void main() { col = vec4(v_dp / 16.0, 1.0); }
)";

static GLuint prog;

static void setup(void)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    prog = glCreateProgram();
    glAttachShader(prog, v); glAttachShader(prog, f);
    glLinkProgram(prog);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok == GL_TRUE);
}

static void draw_read(const char *tag, int first, int count, int base_vertex_indexed,
                      int base_instance, int instances, int want_bv, int want_bi, int want_id)
{
    float xs[16] = { -1.0f, -0.3f,  -0.66f, 0.3f,  -0.33f, -0.3f,  0.0f, 0.3f,
                      0.33f, -0.3f,  0.66f, 0.3f,  0.9f, -0.3f,  1.0f, 0.3f };
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof xs, xs, GL_STREAM_DRAW);
    GLuint vao; glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLint loc = glGetAttribLocation(prog, "a_p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);

    if (base_vertex_indexed >= 0)
    {
        GLuint ibo; glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        unsigned short idx[3] = { 0, 1, 2 };
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STREAM_DRAW);
        if (base_instance)
            glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, 0, instances, base_vertex_indexed, base_instance);
        else
            glDrawElementsBaseVertex(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, 0, base_vertex_indexed);
        glDeleteBuffers(1, &ibo);
    }
    else if (instances > 1 || base_instance)
    {
        if (base_instance)
            glDrawArraysInstancedBaseInstance(GL_TRIANGLES, first, count, instances, base_instance);
        else
            glDrawArraysInstanced(GL_TRIANGLES, first, count, instances);
    }
    else
    {
        glDrawArrays(GL_TRIANGLES, first, count);
    }

    GLubyte px[4] = { 0, 0, 0, 0 };
    for (int x = 0; x < 64; x++)
    {
        glReadPixels(x, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        if (px[3] == 255) break;
    }
    int bv = px[0] * 16 / 255, bi = px[1] * 16 / 255, di = px[2] * 16 / 255;
    printf("  [%s] bv=%d(want %d) bi=%d(want %d) id=%d(want %d)\n", tag, bv, want_bv, bi, want_bi, di, want_id);
    EXPECT_TRUE(px[3] == 255 && bv == want_bv && bi == want_bi && di == want_id);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* GL: arrays `first' is NOT a base vertex — gl_BaseVertex is 0 (Metal's
 * [[base_vertex]] would read vertexStart == first) */
TEST(arrays_first3)   { draw_read("arrays first=3", 3, 3, -1, 0, 0, 0, 0, 0); }
TEST(arrays_first0)   { draw_read("arrays first=0", 0, 3, -1, 0, 0, 0, 0, 0); }
TEST(indexed_bv3)     { draw_read("indexed bv=3", 0, 3, 3, 0, 0, 3, 0, 0); }
TEST(inst_arr_bi1)    { draw_read("arrays inst bi=1", 0, 3, -1, 1, 2, 0, 1, 0); }
TEST(inst_idx_bv3bi2) { draw_read("indexed inst bv=3 bi=2", 0, 3, 3, 2, 2, 3, 2, 0); }

/* multi-draw: draw ids 0..n-1 */
TEST(multi_draw_ids)
{
    float xs[16] = { -0.8f, -0.3f,  -0.5f, 0.3f,  -0.2f, -0.3f,   0.2f, 0.3f,
                      0.5f, -0.3f,   0.8f, 0.3f,   0.9f, -0.3f,   1.0f, 0.3f };
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof xs, xs, GL_STREAM_DRAW);
    GLuint vao; glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLint loc = glGetAttribLocation(prog, "a_p");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);

    GLint first[2] = { 0, 3 };
    GLsizei cnt[2] = { 3, 3 };
    glMultiDrawArrays(GL_TRIANGLES, first, cnt, 2);

    GLubyte px0[4], px1[4];
    glReadPixels(12, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px0);
    glReadPixels(47, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px1);
    printf("  [multi] draw0 id=%d draw1 id=%d\n", px0[2] * 16 / 255, px1[2] * 16 / 255);
    EXPECT_TRUE(px0[3] == 255 && px0[2] == 0);
    EXPECT_TRUE(px1[3] == 255 && px1[2] == 16);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* multi-draw-ARRAYS-indirect (the CTS MultiDrawArraysIndirect shape): three
 * records; per-record draw id, arrays base vertex 0, baseInstance in vOffset */
static const char *vs_mdi = R"(#version 460
in vec3 vertex;
out vec3 color;
void main()
{
    float hOffset = float(gl_BaseVertex) / 5.0;
    float vOffset = float(gl_BaseInstance) / 5.0;
    color = vec3(0.0);
    if (gl_DrawID % 3 == 0) color.r = 1;
    else if (gl_DrawID % 3 == 1) color.g = 1;
    else if (gl_DrawID % 3 == 2) color.b = 1;
    gl_Position = vec4(vertex + vec3(hOffset, vOffset, 0), 1);
}
)";
static const char *fs_mdi = R"(#version 460
in vec3 color;
out vec4 fragColor;
void main() { fragColor = vec4(color, 1.0); }
)";

typedef struct { GLuint count, primCount, first, baseInstance; } ArraysCmd;

TEST(multi_arrays_indirect)
{
    const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, -1.0f, -0.8f, 0.0f, -0.9f, -1.0f, 0.0f,
        -0.9f, -0.8f, 0.0f, -0.8f, -1.0f, 0.0f, -0.8f, -0.8f, 0.0f,
    };
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
    GLuint vao; glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    ArraysCmd cmds[3] = { { 5, 1, 1, 0 }, { 4, 1, 0, 0 }, { 3, 1, 0, 1 } };
    GLuint ibo; glGenBuffers(1, &ibo);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ibo);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof cmds, cmds, GL_STATIC_DRAW);

    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vs_mdi, NULL); glCompileShader(v);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs_mdi, NULL); glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok == GL_TRUE);

    glViewport(0, 0, 256, 256);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, NULL, 3, 0);

    struct { int x, y, r, g, b; } pts[] = {
        { 6, 6,   0, 255, 0 },   /* draw 1 only: green */
        { 6, 19,  255, 255, 0 }, /* draws 0+1: yellow */
        { 19, 6,  255, 0, 0 },   /* draw 0 only: red */
        { 19, 19, 255, 0, 0 },
    };
    for (int i = 0; i < 4; i++)
    {
        GLubyte px[4];
        glReadPixels(pts[i].x, pts[i].y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        printf("  [mdi %d,%d] got %d,%d,%d\n", pts[i].x, pts[i].y, px[0], px[1], px[2]);
        EXPECT_TRUE(px[0] == pts[i].r && px[1] == pts[i].g && px[2] == pts[i].b);
    }
    glDeleteProgram(p); glDeleteShader(v); glDeleteShader(f);
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao); glDeleteBuffers(1, &ibo);
}

int main(void)
{
    setup();
    RUN(arrays_first3);
    RUN(arrays_first0);
    RUN(indexed_bv3);
    RUN(inst_arr_bi1);
    RUN(inst_idx_bv3bi2);
    RUN(multi_draw_ids);
    RUN(multi_arrays_indirect);
    return 0;
}
