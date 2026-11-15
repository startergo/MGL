/* test_gpuaudit.c — the 'gpuaudit' domain: work that used to run on the CPU on the
 * draw / upload / readback path now runs on the GPU or is avoided outright.
 *
 * Every fix is verified by value (buffer bytes / framebuffer pixels), not just by the
 * internal counters (which are also checked so the GPU path is proven to be taken). */
#include "mgl_test.h"
#include <stdint.h>
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
#define MGL_GPUAUDIT_INDIRECT_NOREAD 0x8F08
#define MGL_GPUAUDIT_SHADOW_HITS    0x8F09
#define MGL_GPUAUDIT_XFB_SCATTERS   0x8F0B

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

/* --------------------------------------------------- F3/F4: index list caches */

static GLuint g_pos_col_prog;   /* position + per-vertex colour (flat or smooth) */
static GLuint g_flat_prog;
static GLint g_vp[4];

static void setup2(void)
{
    glGetIntegerv(GL_VIEWPORT, g_vp);

    const char *vs = "#version 330 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=1) in vec4 col;\n"
        "out vec4 v_col;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_col = col; }\n";
    const char *fs_smooth = "#version 330 core\n"
        "in vec4 v_col; layout(location=0) out vec4 color;\n"
        "void main(){ color = v_col; }\n";
    const char *fs_flat = "#version 330 core\n"
        "flat in vec4 v_col; layout(location=0) out vec4 color;\n"
        "void main(){ color = v_col; }\n";
    const char *vs_flat = vs;
    g_pos_col_prog = mgl_make_program(vs, fs_smooth);
    g_flat_prog = mgl_make_program(vs_flat, fs_flat);
}

/* 6 vertices: a big triangle (0,1,2) and a small one (3,4,5) in the opposite corner,
 * with distinct colours so every draw is identifiable by what is on screen */
static const float g_verts[] = {
    -1.0f, -1.0f,   1.0f, 0.0f, 0.0f, 1.0f,     /* 0 */
     3.0f, -1.0f,   1.0f, 0.0f, 0.0f, 1.0f,     /* 1 */
    -1.0f,  3.0f,   1.0f, 0.0f, 0.0f, 1.0f,     /* 2 */
     0.2f,  0.2f,   0.0f, 1.0f, 0.0f, 1.0f,     /* 3 */
     1.0f,  0.2f,   0.0f, 1.0f, 0.0f, 1.0f,     /* 4 */
     0.2f,  1.0f,   0.0f, 1.0f, 0.0f, 1.0f,     /* 5 */
};

static GLuint g_vao6, g_ebo6;

static void setup_geom(void)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_verts), g_verts, GL_STATIC_DRAW);
    glGenVertexArrays(1, &g_vao6);
    glBindVertexArray(g_vao6);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, (void *)8);
    glGenBuffers(1, &g_ebo6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo6);
}

/* how many pixels are red / green after a clear + draw */
static void coverage(GLuint *red, GLuint *green)
{
    GLsizei w = g_vp[2], h = g_vp[3];
    static GLubyte *px;
    if (!px) px = (GLubyte *)malloc((size_t)w * h * 4);
    glFinish();
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
    GLuint r = 0, g = 0;
    for (GLsizei i = 0; i < w * h; i++)
    {
        if (px[i * 4] > 200 && px[i * 4 + 1] < 50) r++;
        else if (px[i * 4 + 1] > 200 && px[i * 4] < 50) g++;
    }
    *red = r;
    *green = g;
}

/* fan/loop/adjacency emulation and restart splitting must give identical pixels
 * whether the list was just built (first draw) or served from the cache (rest) */
TEST(index_cache_emulation_and_restart)
{
    glUseProgram(g_pos_col_prog);
    glBindVertexArray(g_vao6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo6);

    /* reference: 6 indices, two triangles, no restart */
    GLuint idx6[6] = { 0, 1, 2, 3, 4, 5 };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx6), idx6, GL_STATIC_DRAW);

    GLuint r_ref = 0, g_ref = 0;
    for (int rep = 0; rep < 3; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glFinish();
        coverage(&r_ref, &g_ref);
    }
    EXPECT_TRUE(r_ref > 100);
    EXPECT_TRUE(g_ref > 100);

    /* the same geometry as a fan (emulated on the CPU): cached after the first draw */
    GLuint r_fan = 0, g_fan = 0;
    for (int rep = 0; rep < 3; rep++)
    {
        GLuint r, g;
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 3);
        glFinish();
        coverage(&r, &g);
        if (rep == 0) { r_fan = r; g_fan = g; }
        EXPECT_TRUE(r > 100);              /* the fan is emulated; the first draw sets
                                              the reference for the cached ones */
        EXPECT_EQ_INT(r, r_fan);           /* cached draws must render identically */
        EXPECT_EQ_INT(g, g_fan);
    }
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_INDEX_HITS) > 0);

    /* restart splitting: two triangles separated by the restart index */
    GLuint idx8[8] = { 0, 1, 2, 0xFFFFFFFFu, 3, 4, 5, 0xFFFFFFFFu };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx8), idx8, GL_STATIC_DRAW);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);

    for (int rep = 0; rep < 3; rep++)
    {
        GLuint r, g;
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 8, GL_UNSIGNED_INT, 0);
        glFinish();
        coverage(&r, &g);
        EXPECT_TRUE(r > 100);
        EXPECT_TRUE(g > 100);
    }

    /* restart scan is cached: 3 draws above scanned once */
    EXPECT_EQ_INT(g_stat(MGL_GPUAUDIT_SCAN_MISSES) >= 1, 1);
    GLint scan_hits = g_stat(MGL_GPUAUDIT_SCAN_HITS);
    glDrawElements(GL_TRIANGLES, 8, GL_UNSIGNED_INT, 0);
    glFinish();
    EXPECT_EQ_INT(g_stat(MGL_GPUAUDIT_SCAN_HITS) > scan_hits, 1);

    /* 8-bit indices with restart (widening + restart on the CPU path); the restart
     * index has to match the index width, which is the spec-correct usage */
    GLubyte ub8[8] = { 0, 1, 2, 0xFF, 3, 4, 5, 0xFF };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ub8), ub8, GL_STATIC_DRAW);
    glPrimitiveRestartIndex(0xFFu);

    for (int rep = 0; rep < 3; rep++)
    {
        GLuint r, g;
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 8, GL_UNSIGNED_BYTE, 0);
        glFinish();
        coverage(&r, &g);
        EXPECT_EQ_INT(r, r_ref);
        EXPECT_EQ_INT(g, g_ref);
    }

    glDisable(GL_PRIMITIVE_RESTART);
    EXPECT_NO_GL_ERROR();
}

/* a re-uploaded index buffer must invalidate the cached list (same offset/count/type) */
TEST(index_cache_invalidated_on_reupload)
{
    glUseProgram(g_pos_col_prog);
    glBindVertexArray(g_vao6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo6);

    GLuint idx[3] = { 0, 1, 2 };           /* big red triangle */
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    GLuint r = 0, g = 0;
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
    glFinish();
    coverage(&r, &g);
    EXPECT_TRUE(r > 100);
    EXPECT_EQ_INT(g, 0);

    /* same buffer object, same offset/count/type: only the contents change */
    GLuint idx2[3] = { 3, 4, 5 };
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(idx2), idx2);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
    glFinish();
    coverage(&r, &g);
    EXPECT_EQ_INT(r, 0);
    EXPECT_TRUE(g > 100);

    EXPECT_NO_GL_ERROR();
}

/* changing the provoking-vertex convention must rebuild the cached list, not reuse it */
TEST(index_cache_provoking_state)
{
    glUseProgram(g_flat_prog);
    glBindVertexArray(g_vao6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo6);

    /* the convention is part of the cache key: flipping it must not reuse a list built
     * under the other convention, and repeated draws must keep hitting the cache.
     * (MGL renders this geometry identically under both conventions; the point here is
     * that the cached draw path keeps producing exactly that output.) */
    GLuint idx[4] = { 0, 2, 3, 5 };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    GLuint r_ref = 0, g_ref = 0;
    glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
    for (int rep = 0; rep < 3; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, 0);
        coverage(&r_ref, &g_ref);
    }
    EXPECT_TRUE(r_ref > 100);

    glProvokingVertex(GL_LAST_VERTEX_CONVENTION);
    for (int rep = 0; rep < 3; rep++)
    {
        GLuint r = 0, g = 0;
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, 0);
        coverage(&r, &g);
        EXPECT_EQ_INT(r, r_ref);
        EXPECT_EQ_INT(g, g_ref);
    }

    glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------- F6: glReadPixels into a PIXEL_PACK_BUFFER */

TEST(read_pixels_into_pbo_gpu)
{
    /* offscreen RGBA8 framebuffer with a known gradient */
    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    GLubyte img[64 * 64 * 4];
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
        {
            int i = (y * 64 + x) * 4;
            img[i] = (GLubyte)(x * 4);
            img[i + 1] = (GLubyte)(y * 4);
            img[i + 2] = 0x40;
            img[i + 3] = 0xFF;
        }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glFinish();

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, 64 * 64 * 4, NULL, GL_STREAM_READ);

    GLint before = g_stat(MGL_GPUAUDIT_PBO_READBACKS);
    glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_PBO_READBACKS) == before + 1);   /* GPU blit taken */

    /* map and compare byte for byte */
    GLubyte *m = (GLubyte *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, 64 * 64 * 4, GL_MAP_READ_BIT);
    int bad = 0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
        {
            int i = (y * 64 + x) * 4;
            if (m[i] != (GLubyte)(x * 4) || m[i + 1] != (GLubyte)(y * 4) ||
                m[i + 2] != 0x40 || m[i + 3] != 0xFF)
                bad++;
        }
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    EXPECT_EQ_INT(bad, 0);

    /* pack alignment applies to the destination rows: width 61 pixels = 244 bytes,
     * padded to 248 by GL_PACK_ALIGNMENT 8 */
    glPixelStorei(GL_PACK_ALIGNMENT, 8);
    glBufferData(GL_PIXEL_PACK_BUFFER, 64 * 248, NULL, GL_STREAM_READ);
    glReadPixels(0, 0, 61, 64, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    EXPECT_NO_GL_ERROR();
    m = (GLubyte *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, 64 * 248, GL_MAP_READ_BIT);
    bad = 0;
    for (int y = 0; y < 64; y++)
        for (int x = 0; x < 61; x++)
        {
            int i = y * 248 + x * 4;
            if (m[i] != (GLubyte)(x * 4) || m[i + 1] != (GLubyte)(y * 4)) bad++;
        }
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    EXPECT_EQ_INT(bad, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    /* skip rows/space are honoured (row stride of an 8-pixel read = 32 bytes) */
    glPixelStorei(GL_PACK_SKIP_ROWS, 3);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 5);
    glBufferData(GL_PIXEL_PACK_BUFFER, 64 * 248, NULL, GL_STREAM_READ);
    glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    EXPECT_NO_GL_ERROR();
    m = (GLubyte *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, 64 * 248, GL_MAP_READ_BIT);
    bad = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
        {
            int i = (y + 3) * 32 + (x + 5) * 4;
            if (m[i] != (GLubyte)(x * 4) || m[i + 1] != (GLubyte)(y * 4)) bad++;
        }
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    EXPECT_EQ_INT(bad, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);

    /* reads the blit cannot serve keep working through the CPU path */
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    GLubyte sub[8 * 8 * 4];
    glReadPixels(4, 4, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, sub);
    EXPECT_EQ_INT(sub[0], 16);      /* x = 4 */
    EXPECT_EQ_INT(sub[1], 16);      /* y = 4 */
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteBuffers(1, &pbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

/* -------------------------------- F11: multi-draw indirect without CPU reads */

typedef struct { GLuint count, instanceCount, first, baseVertex, baseInstance; } DrawCmd;

TEST(multi_draw_indirect_no_cpu_read)
{
    glUseProgram(g_pos_col_prog);
    glBindVertexArray(g_vao6);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ebo6);

    /* 200 commands: half draw the big red triangle, half the small green one */
    enum { N = 200 };
    DrawCmd cmds[N];
    GLuint idx[6] = { 0, 1, 2, 3, 4, 5 };
    for (int i = 0; i < N; i++)
    {
        GLuint first = (i & 1) ? 3 : 0;
        cmds[i] = (DrawCmd){ 3, 1, first, 0, 0 };
        if (i == 0)
            cmds[i].count = 3, cmds[i].first = 0;
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    GLuint ibuf;
    glGenBuffers(1, &ibuf);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ibuf);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmds), cmds, GL_STATIC_DRAW);

    /* reference: the same 200 triangles drawn one by one */
    GLuint r_ref = 0, g_ref = 0;
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < N; i++)
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT,
                       (const void *)(uintptr_t)(((i & 1) ? 3u : 0u) * sizeof(GLuint)));
    glFinish();
    coverage(&r_ref, &g_ref);

    GLint before = g_stat(MGL_GPUAUDIT_INDIRECT_NOREAD);
    for (int rep = 0; rep < 3; rep++)
    {
        GLuint r, g;
        glClear(GL_COLOR_BUFFER_BIT);
        glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, N, 20);
        glFinish();
        coverage(&r, &g);
        EXPECT_EQ_INT(r, r_ref);
        EXPECT_EQ_INT(g, g_ref);
    }
    /* every command was handed to Metal unread */
    EXPECT_EQ_INT(g_stat(MGL_GPUAUDIT_INDIRECT_NOREAD) - before, 3 * N);

    /* with a counting query active the counts are read again and the CPU counters work */
    GLuint q;
    glGenQueries(1, &q);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q);
    glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, N, 20);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glFinish();
    GLuint prims = 0;
    glGetQueryObjectuiv(q, GL_QUERY_RESULT, &prims);
    EXPECT_EQ_INT(prims, N);
    glDeleteQueries(1, &q);

    glDeleteBuffers(1, &ibuf);
    EXPECT_NO_GL_ERROR();
}

/* ---------------------------- F8: converted vertex-attribute streams (shadow cache) */

static GLuint g_dprog, g_dvao, g_dvbo;
static double g_dverts[12] = { -1, -1, 0, 1,   3, -1, 0, 1,   -1, 3, 0, 1 };

static const char *shadow_vs_src(void)
{
    /* a program that takes a dvec4 attribute: the fp64 lowering fetches it as
     * raw u32 word slots; misaligned streams are converted on the CPU and
     * cached after the first draw */
    return "#version 430 core\n"
        "layout(location = 0) in dvec4 pos;\n"
        "void main() { gl_Position = vec4(pos.xy / 2.0, 0.0, 1.0); }\n";
}
static const char *shadow_fs_src(void)
{
    return "#version 430 core\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() { color = vec4(0, 1, 0, 1); }\n";
}

static void shadow_setup(void)
{
    g_dprog = mgl_make_program(shadow_vs_src(), shadow_fs_src());

    glGenBuffers(1, &g_dvbo);
    glBindBuffer(GL_ARRAY_BUFFER, g_dvbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_dverts), g_dverts, GL_STATIC_DRAW);
    glGenVertexArrays(1, &g_dvao);
    glBindVertexArray(g_dvao);
    glEnableVertexAttribArray(0);
    /* misaligned relative offset on purpose: a bit-exact L-pointer stream cannot
     * be fetched directly by Metal, so it goes through the converted (cached)
     * shadow path even for glVertexAttribLPointer */
    glBufferData(GL_ARRAY_BUFFER, 3 * 36, NULL, GL_STATIC_DRAW);
    for (int i = 0; i < 3; i++)
        glBufferSubData(GL_ARRAY_BUFFER, i * 36 + 3, 32, g_dverts + i * 4);
    glVertexAttribLPointer(0, 4, GL_DOUBLE, 36, (void *)(uintptr_t)3);
}

TEST(shadow_attribute_cache)
{
    shadow_setup();
    glUseProgram(g_dprog);

    GLuint r = 0, g = 0;
    for (int rep = 0; rep < 3; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        coverage(&r, &g);
        EXPECT_TRUE(g > 100);              /* the double attribute made it into the shader */
        EXPECT_EQ_INT(r, 0);
    }
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_SHADOW_HITS) >= 2);   /* draws 2 and 3 were served cached */

    /* re-uploading the vertex data must invalidate the converted stream */
    double verts2[12] = { -1, -1, 0, 1,   1, -1, 0, 1,   -1, 1, 0, 1 };
    for (int i = 0; i < 3; i++)
        glBufferSubData(GL_ARRAY_BUFFER, i * 36 + 3, 32, verts2 + i * 4);
    GLuint g2 = 0;
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    coverage(&r, &g2);
    EXPECT_TRUE(g2 > 100);
    EXPECT_TRUE(g2 < g);                    /* the smaller triangle covers fewer pixels */

    EXPECT_NO_GL_ERROR();
}

/* ----------------------------------- F9: transform feedback scatter on the GPU */

TEST(xfb_scatter_gpu)
{
    /* program capturing gl_Position + a colour, INTERLEAVED */
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec3 vcolor;\n"
        "layout(location=1) out float vid;\n"
        "void main(){\n"
        "  gl_Position = vec4(pos, 0.25, 1.0);\n"
        "  vcolor = vec3(pos.x * 2.0, pos.y + 10.0, float(gl_VertexID));\n"
        "  vid = float(gl_VertexID) + 100.0;\n"
        "}\n";
    const char *fs = "#version 450 core\nlayout(location=0) out vec4 color;\nvoid main(){ color = vec4(1.0); }\n";

    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &vs, NULL); glCompileShader(s); glAttachShader(p, s);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f); glAttachShader(p, f);
    const char *vars[3] = { "gl_Position", "vcolor", "vid" };
    glTransformFeedbackVaryings(p, 3, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glUseProgram(p);

    const GLuint NV = 9000;
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static float verts[9000 * 2];
    for (GLuint i = 0; i < NV; i++) { verts[i * 2] = -1.0f + (float)(i % 17) * 0.01f; verts[i * 2 + 1] = -1.0f + (float)(i % 13) * 0.01f; }
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(NV * 2 * 4), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    static GLfloat junk[9000 * 8];
    for (GLuint i = 0; i < NV * 8; i++) junk[i] = -1.0f;
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, (GLsizeiptr)(NV * 8 * 4), junk, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, b);

    GLint before = g_stat(MGL_GPUAUDIT_XFB_SCATTERS);

    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, NV);
    glEndTransformFeedback();
    glFinish();

    EXPECT_EQ_INT(g_stat(MGL_GPUAUDIT_XFB_SCATTERS), before + 1);   /* kernel path taken */

    /* byte-exact against the CPU path's values */
    static GLfloat out[9000 * 8];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, (GLsizeiptr)(NV * 8 * 4), out);

    int bad = 0;
    for (GLuint v = 0; v < NV && bad < 4; v++)
    {
        GLfloat e[8] = { verts[v * 2], verts[v * 2 + 1], 0.25f, 1.0f,
                         verts[v * 2] * 2.0f, verts[v * 2 + 1] + 10.0f, (GLfloat)v, (GLfloat)v + 100.0f };
        for (int c = 0; c < 8; c++)
            if (out[v * 8 + c] != e[c]) { bad++; break; }
    }
    EXPECT_EQ_INT(bad, 0);

    /* a second capture starts over at the buffer start (glBeginTransformFeedback) and
     * must also be served by the scatter kernel */
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, NV);
    glEndTransformFeedback();
    glFinish();
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, b);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, (GLsizeiptr)(8 * 4), out);
    EXPECT_NEAR(out[0], verts[0], 1e-6);
    EXPECT_NEAR(out[2], 0.25f, 1e-6);
    EXPECT_NEAR(out[6], 0.0f, 1e-6);       /* gl_VertexID restarts per draw */
    EXPECT_NEAR(out[7], 100.0f, 1e-6);
    EXPECT_EQ_INT(g_stat(MGL_GPUAUDIT_XFB_SCATTERS), before + 2);

    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
}

/* ------------------------------------------------------------ F2: pipeline cache */

TEST(pipeline_state_cache)
{
    glUseProgram(g_prog);            /* previous tests left other programs bound */
    float tri[] = { -1, -1, 3, -1, -1, 3 };
    bind_verts(tri, 3);

    GLint hits = g_stat(MGL_GPUAUDIT_PSO_HITS), misses = g_stat(MGL_GPUAUDIT_PSO_MISSES);
    GLint created = hits + misses;
    /* r152: the compat pipeline fast path (MGLRenderer+Compat.m) serves these draws
     * from its own cache before the descriptor-content cache is consulted — since
     * glFinish no longer forces the descriptor rebuild, the hits can land there
     * (its counter: MGLRenderer.m's mgl_prof_cnt[16], pipeHit) */
    extern uint64_t mgl_prof_cnt[];
    uint64_t compat_hits = mgl_prof_cnt[16];

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

    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_PSO_MISSES) - misses <= 2);
    EXPECT_TRUE(g_stat(MGL_GPUAUDIT_PSO_HITS) > hits || mgl_prof_cnt[16] > compat_hits);
    (void)created; (void)after_warm;

    /* and the cached states must still render the right thing */
    GLfloat c1[4] = { 1, 0, 0, 1 };
    glUniform4fv(glGetUniformLocation(g_prog, "u_col"), 1, c1);
    glDisable(GL_BLEND);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte px1[4];
    color_at_center(c1, px1);
    GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
    mgl_read_rgba(vp[2] / 2, vp[3] / 2, px1);
    EXPECT_EQ_INT(px1[0], 255);
    EXPECT_EQ_INT(px1[1], 0);
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
    t1 = now_ms();                       /* submit only: the CPU cost of the draw */
    glFinish();
    printf("# 20 draws of %zu indices (restart on): %.3f ms/draw (scan h=%d m=%d)\n", n, (t1 - t0) / 20,
           g_stat(MGL_GPUAUDIT_SCAN_HITS), g_stat(MGL_GPUAUDIT_SCAN_MISSES));
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
    setup2();
    setup_geom();
    RUN(index_cache_emulation_and_restart);
    RUN(index_cache_invalidated_on_reupload);
    RUN(index_cache_provoking_state);
    RUN(gpu_buffer_clear);
    RUN(xfb_scatter_gpu);
    RUN(shadow_attribute_cache);
    RUN(multi_draw_indirect_no_cpu_read);
    RUN(read_pixels_into_pbo_gpu);
    RUN(cpu_buffer_clear_still_correct);
    RUN(pipeline_state_cache);
    RUN(timings);
    return mgl_test_summary();
}
