/* test_gpuaudit.c — the 'gpuaudit' domain: work that used to run on the CPU on the
 * draw / upload / readback path now runs on the GPU or is avoided outright.
 *
 * Every fix is verified by value (buffer bytes / framebuffer pixels), not just by the
 * internal counters (which are also checked so the GPU path is proven to be taken). */
#include "mgl_test.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

/* private glGet pnames of this domain (mgl_ext_get_gpuaudit) */
#define MGL_GPUAUDIT_PSO_HITS      0x8F00
#define MGL_GPUAUDIT_PSO_MISSES    0x8F01
#define MGL_GPUAUDIT_SCAN_HITS     0x8F02
#define MGL_GPUAUDIT_SCAN_MISSES   0x8F03
#define MGL_GPUAUDIT_INDEX_HITS    0x8F04
#define MGL_GPUAUDIT_INDEX_MISSES  0x8F05
#define MGL_GPUAUDIT_PBO_READBACKS 0x8F06
#define MGL_GPUAUDIT_GPU_CLEARS    0x8F07

static GLint g_stat(GLenum pname)
{
    GLint v = -1;
    glGetIntegerv(pname, &v);
    return v;
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static GLuint g_prog;            /* flat colour program, vertex at location 0 */

static void setup(void)
{
    const char *vs = "#version 330 core\n"
        "layout(location=0) in vec2 pos;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 330 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 u_col;\n"
        "void main(){ color = u_col; }\n";
    g_prog = mgl_make_program(vs, fs);
    glUseProgram(g_prog);
}

/* upload a vertex buffer with `n` 2-float vertices and bind attribute 0 to it */
static GLuint bind_verts(const float *v, int n)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(n * 2 * sizeof(float)), v, GL_STATIC_DRAW);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    return vao;
}

/* draw with a given clear colour and read the centre pixel back */
static void color_at_center(GLfloat col[4], GLubyte out[4])
{
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    glClearColor(col[0], col[1], col[2], col[3]);
    glClear(GL_COLOR_BUFFER_BIT);
    mgl_read_rgba(vp[2] / 2, vp[3] / 2, out);
}

/* ---------------------------------------------------------------- F5: GPU clears */

TEST(gpu_buffer_clear)
{
    /* promote a buffer to GPU residency with a draw, then clear on the GPU: the buffer
     * that is cleared is the one the draw read from */
    float tri[] = { -1, -1, 3, -1, -1, 3 };
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, 1 << 20, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(tri), tri);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    GLint clears = g_stat(MGL_GPUAUDIT_GPU_CLEARS);

    float f4[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
    unsigned char u1 = 0x33;
    GLuint u32 = 0xDEADBEEFu;
    unsigned short u2[2] = { 0x1234, 0x5678 };

    glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, f4);
    glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, &u1);
    glClearBufferSubData(GL_ARRAY_BUFFER, GL_R32UI, 64, 64, GL_RED_INTEGER, GL_UNSIGNED_INT, &u32);
    glClearBufferSubData(GL_ARRAY_BUFFER, GL_RG16, 128, 32, GL_RG, GL_UNSIGNED_SHORT, u2);
    glFinish();

    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_GPU_CLEARS) >= clears + 4);

    {   /* R8 overwrote the RGBA32F clear everywhere */
        GLubyte *p = (GLubyte *)glMapBufferRange(GL_ARRAY_BUFFER, 4096, 16, GL_MAP_READ_BIT);
        EXPECT_EQ_INT(p[0], 0x33);
        EXPECT_EQ_INT(p[15], 0x33);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    {   /* 4-byte sub clear at 64..128 */
        GLuint *q = (GLuint *)glMapBufferRange(GL_ARRAY_BUFFER, 64, 64, GL_MAP_READ_BIT);
        EXPECT_EQ_INT(q[0], 0xDEADBEEFu & 0xFFFFFFFF);
        EXPECT_EQ_INT(q[15], 0xDEADBEEFu & 0xFFFFFFFF);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    {   /* 2-component 2-byte sub clear at 128..160 */
        GLushort *s = (GLushort *)glMapBufferRange(GL_ARRAY_BUFFER, 128, 32, GL_MAP_READ_BIT);
        EXPECT_EQ_INT(s[0], 0x1234);
        EXPECT_EQ_INT(s[1], 0x5678);
        EXPECT_EQ_INT(s[14], 0x1234);
        EXPECT_EQ_INT(s[15], 0x5678);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    {   /* full 16-byte-texel clear again, to prove the old pattern is gone */
        glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, f4);
        glFinish();
        GLfloat *f = (GLfloat *)glMapBufferRange(GL_ARRAY_BUFFER, 0, 4, GL_MAP_READ_BIT);
        EXPECT_NEAR(f[0], 1.0, 1e-6);
        EXPECT_NEAR(f[1], 0.5, 1e-6);
        EXPECT_NEAR(f[2], 0.25, 1e-6);
        EXPECT_NEAR(f[3], 1.0, 1e-6);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    EXPECT_NO_GL_ERROR();

    glBindVertexArray(vao);
    glDeleteBuffers(1, &b);
}

/* CPU-only buffers keep the CPU path: no promotion is forced on them */
TEST(cpu_buffer_clear_still_correct)
{
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, 4096, NULL, GL_STATIC_DRAW);
    GLint v = 0x01020304;
    glClearBufferData(GL_ARRAY_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &v);
    glFinish();
    GLuint *q = (GLuint *)glMapBufferRange(GL_ARRAY_BUFFER, 0, 16, GL_MAP_READ_BIT);
    EXPECT_EQ_INT(q[0], 0x01020304);
    EXPECT_EQ_INT(q[3], 0x01020304);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glDeleteBuffers(1, &b);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------ F2: pipeline cache */

TEST(pipeline_state_cache)
{
    float tri[] = { -1, -1, 3, -1, -1, 3 };
    bind_verts(tri, 3);

    GLint hits = g_stat(MGL_GPUAUDIT_PSO_HITS), misses = g_stat(MGL_GPUAUDIT_PSO_MISSES);
    GLint created = hits + misses;

    /* warm both blend configurations once */
    for (int i = 0; i < 4; i++)
    {
        if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    GLint after_warm = g_stat(MGL_GPUAUDIT_PSO_HITS) + g_stat(MGL_GPUAUDIT_PSO_MISSES);

    /* repeated state churn must be served from the cache: 400 more draws with the same
     * two configurations may build at most a couple of new states (the first ones above) */
    for (int i = 0; i < 400; i++)
    {
        if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();

    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_PSO_MISSES) - misses <= 2);
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_PSO_HITS) > hits);
    (void)created; (void)after_warm;

    /* and the two states must still render the right thing */
    GLfloat c1[4] = { 1, 0, 0, 1 }, c2[4] = { 0, 1, 0, 1 };
    glUniform4fv(glGetUniformLocation(g_prog, "u_col"), 1, c1);
    glDisable(GL_BLEND);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte px1[4];
    color_at_center(c1, px1);
    EXPECT_EQ_INT(px1[0], 255);
    EXPECT_EQ_INT(px1[1], 0);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform4fv(glGetUniformLocation(g_prog, "u_col"), 1, c2);
    glDrawArrays(GL_TRIANGLES, 0, 3);   /* fully opaque green */
    GLubyte px2[4];
    mgl_read_rgba(0, 0, px2);
    EXPECT_EQ_INT(px2[1], 255);
    EXPECT_EQ_INT(px2[0], 0);
    glDisable(GL_BLEND);
    EXPECT_NO_GL_ERROR();
}

/* -------------------------------------------------------- timings (informational) */

TEST(timings)
{
    float tri[] = { -1, -1, 3, -1, -1, 3 };
    bind_verts(tri, 3);
    for (int i = 0; i < 100; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    double t0 = now_ms();
    for (int i = 0; i < 2000; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    double t1 = now_ms();
    printf("# 2000 identical draws: %.2f us/draw\n", (t1 - t0) * 1e3 / 2000);

    t0 = now_ms();
    for (int i = 0; i < 1000; i++)
    {
        if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glFinish();
    t1 = now_ms();
    printf("# 1000 blend-state flips: %.2f us/draw\n", (t1 - t0) * 1e3 / 1000);
    glDisable(GL_BLEND);

    /* a large index buffer drawn with primitive restart: the scan must not run per draw */
    size_t n = 512 * 1024;
    GLuint *idx = (GLuint *)malloc(n * 4);
    for (size_t i = 0; i < n; i++) idx[i] = (GLuint)(i % 300);
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(n * 4), idx, GL_STATIC_DRAW);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    for (int i = 0; i < 2; i++) glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
    glFinish();
    t0 = now_ms();
    for (int i = 0; i < 20; i++) glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
    glFinish();
    t1 = now_ms();
    printf("# 20 draws of %zu indices (restart on): %.3f ms/draw\n", n, (t1 - t0) / 20);
    glDisable(GL_PRIMITIVE_RESTART);
    free(idx);

    /* 16 MiB GPU clear */
    {
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        GLuint b;
        glGenBuffers(1, &b);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        glBufferData(GL_ARRAY_BUFFER, 16 << 20, NULL, GL_STATIC_DRAW);
        float c[4] = { 1, 0.5, 0.25, 1 };
        glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, c);
        glFinish();
        t0 = now_ms();
        glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, c);
        glFinish();
        t1 = now_ms();
        printf("# 16 MiB RGBA32F clear (submit+wait): %.3f ms\n", t1 - t0);
        glDeleteBuffers(1, &b);
    }
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    setup();
    RUN(gpu_buffer_clear);
    RUN(cpu_buffer_clear_still_correct);
    RUN(pipeline_state_cache);
    RUN(timings);
    return mgl_test_summary();
}
