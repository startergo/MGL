/*
 * test_ssbo_switch.c — an SSBO bound to the vertex stage must be re-read by
 * every draw: glBindBufferBase / glBindBufferRange switching between draws of
 * the same program.  Regression for
 * KHR-GL46.shader_storage_buffer_object.advanced-switchBuffers (and the
 * bindRange half of advanced-indirectAddressing-case2).
 */
#include "mgl_test.h"
#include <string.h>

static const char *vs_src =
    "#version 450\n"
    "struct VertexData { vec2 position; vec3 color; };\n"
    "layout(binding = 0, std430) buffer Input { VertexData vertex[]; } g_vs_in;\n"
    "out StageData { vec3 color; } g_vs_out;\n"
    "void main() {\n"
    "  gl_Position = vec4(g_vs_in.vertex[gl_VertexID].position, 0, 1);\n"
    "  g_vs_out.color = g_vs_in.vertex[gl_VertexID].color;\n"
    "}\n";
static const char *fs_src =
    "#version 450\n"
    "in StageData { vec3 color; } g_fs_in;\n"
    "layout(location = 0) out vec4 g_fs_out;\n"
    "void main() { g_fs_out = vec4(g_fs_in.color, 1); }\n";

/* each quad's 4 vertices: pos.xy, color.rgb (+pad) — the test's std430 layout */
static float vdata[4][32];

static void fill_vdata(void)
{
    static const float org[4][5] = { /* x, y, r, g, b */
        {-0.9f, -0.9f, 1, 0, 0},
        {0.1f, -0.9f, 0, 1, 0},
        {-0.9f, 0.1f, 0, 0, 1},
        {0.1f, 0.1f, 1, 1, 0},
    };
    for (int q = 0; q < 4; q++)
        for (int v = 0; v < 4; v++)
        {
            float dx = (v & 1) ? 0.8f : 0.0f, dy = (v & 2) ? 0.8f : 0.0f;
            float *e = &vdata[q][v * 8];      /* 32-byte record */
            e[0] = org[q][0] + dx;            /* position.xy  @ +0 */
            e[1] = org[q][1] + dy;
            e[4] = org[q][2];                 /* color.rgb    @ +16 (std430 vec3 align) */
            e[5] = org[q][3];
            e[6] = org[q][4];
        }
}

static GLuint prog, tex, fbo, vao, ssbo[5];

static void check_quads(const char *tag)
{
    static const int r[4] = {255, 0, 0, 255}, g[4] = {0, 255, 0, 255}, b[4] = {0, 0, 255, 0};
    static const int sx[4] = {25, 75, 25, 75}, sy[4] = {25, 25, 75, 75};
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    for (int q = 0; q < 4; q++)
    {
        GLubyte px[4];
        mgl_read_rgba(sx[q], sy[q], px);
        mgl_test_checks++;
        if (abs(px[0] - r[q]) > 8 || abs(px[1] - g[q]) > 8 || abs(px[2] - b[q]) > 8)
            MGL_FAIL("quadrant %d (%s): got %u,%u,%u want %d,%d,%d", q, tag,
                     px[0], px[1], px[2], r[q], g[q], b[q]);
    }
}

TEST(ssbo_switch_buffers)
{
    GLint ok;
    GLuint shv = glCreateShader(GL_VERTEX_SHADER), shf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shv, 1, &vs_src, NULL);
    glCompileShader(shv);
    glGetShaderiv(shv, GL_COMPILE_STATUS, &ok);
    if (!ok) { EXPECT_EQ_INT(ok, 1); return; }
    glShaderSource(shf, 1, &fs_src, NULL);
    glCompileShader(shf);
    glGetShaderiv(shf, GL_COMPILE_STATUS, &ok);
    if (!ok) { EXPECT_EQ_INT(ok, 1); return; }
    prog = glCreateProgram();
    glAttachShader(prog, shv);
    glAttachShader(prog, shf);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { EXPECT_EQ_INT(ok, 1); return; }

    fill_vdata();
    glGenBuffers(5, ssbo);
    for (int i = 0; i < 4; i++)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vdata[0]), vdata[i], GL_STATIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(vdata), NULL, GL_STATIC_DRAW);
    for (int i = 0; i < 4; i++)
    {
        glBindBuffer(GL_COPY_READ_BUFFER, ssbo[i]);
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_SHADER_STORAGE_BUFFER, 0, i * sizeof(vdata[0]),
                            sizeof(vdata[0]));
    }

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 100, 100);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glViewport(0, 0, 100, 100);
    glUseProgram(prog);

    /* loop 1: glBindBufferBase switch between draws of one program */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 4; i++)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[i]);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    check_quads("bindBase");

    /* loop 2: glBindBufferRange into the concatenated buffer */
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 4; i++)
    {
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, ssbo[4], i * sizeof(vdata[0]), sizeof(vdata[0]));
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, 1);
    }
    check_quads("bindRange");

    /* loop 3: same draws with a finish between (encoder churn) */
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 4; i++)
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo[i]);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glFinish();
    }
    check_quads("finish");
}

int main(void)
{
    RUN(ssbo_switch_buffers);
    return mgl_test_summary();
}
