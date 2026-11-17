/* r172: a dirty Metal-backed buffer whose writes all reported their range
 * (bufferDidModify / the compat ring -> didModifyRange of that range) skips the
 * whole-buffer didModifyRange at the next draw; writers that do not report a
 * range (default-uniform block, readback into a buffer, ...) keep it.
 *   1. constant ring through unsynchronized glMapNamedBufferRange per draw
 *   2. glNamedBufferSubData into a large uniform buffer per draw
 *   3. a default-uniform vec4 changed per draw (unsynced writer: full syncs)
 *   4. glReadPixels into a buffer, then that buffer read as a uniform block
 * Counts mgl_buf_sync_full / mgl_buf_sync_skipped; every draw's colour checked. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_buf_sync_full, mgl_buf_sync_skipped;

#define W 64
#define CELL 8
#define SLOT 256

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs_ubo =
    "#version 450\nlayout(std140, binding=1) uniform C { vec4 col; } c;\nout vec4 o;\nvoid main(){ o = vec4(c.col.rgb, 1.0); }\n";
static const char *fs_dub =
    "#version 450\nlayout(location=1) uniform vec4 col;\nout vec4 o;\nvoid main(){ o = vec4(col.rgb, 1.0); }\n";

static GLuint mk(const char *f)
{
    const char *s[2] = {vs_cell, f}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) {
        GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(sh, sizeof log, NULL, log); printf("  [compile] %s\n", log); }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    return p;
}
static unsigned char img[W * W * 4];
static void grab(void) { glFinish(); glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, img); }
static int cell_is(int cell, int r, int g, int b, const char *what)
{
    int cx = cell % 8, cy = cell / 8;
    const unsigned char *p = &img[4 * ((cy * CELL + CELL / 2) * W + cx * CELL + CELL / 2)];
    int ok = abs(p[0] - r) <= 3 && abs(p[1] - g) <= 3 && abs(p[2] - b) <= 3;
    if (!ok) printf("  [%s] cell %d = %d %d %d, want %d %d %d\n", what, cell, p[0], p[1], p[2], r, g, b);
    return ok ? 0 : 1;
}
static void place(int cell) { glUniform2f(0, (float)(cell % 8), (float)(cell / 8)); }
static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }
static void col_of(int i, float *c) { c[0] = (float)(i % 4) / 3.0f; c[1] = (float)((i / 4) % 4) / 3.0f; c[2] = (i & 16) ? 1.0f : 0.0f; c[3] = 1.0f; }
static int u8(float f) { return (int)(f * 255.0f + 0.5f); }

int main(void)
{
    int bad = 0;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    float unitq[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof unitq, unitq, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint pu = mk(fs_ubo), pd = mk(fs_dub);
    uint64_t f0, s0;

    /* ---- 1. constant ring, unsynchronized maps */
    glUseProgram(pu);
    GLuint ring; glCreateBuffers(1, &ring);
    glNamedBufferStorage(ring, 1 << 20, NULL, GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
    for (int i = 0; i < 2; i++) {   /* warm-up: the buffer gets its Metal storage */
        float *w = glMapNamedBufferRange(ring, 0, 16, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
        col_of(0, w); glUnmapNamedBuffer(ring);
        glBindBufferRange(GL_UNIFORM_BUFFER, 1, ring, 0, 16); place(63); quad();
    }
    glFinish();
    f0 = mgl_buf_sync_full; s0 = mgl_buf_sync_skipped;
    for (int i = 0; i < 16; i++) {
        GLintptr off = (GLintptr)(i + 1) * SLOT;
        float *w = glMapNamedBufferRange(ring, off, 16, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
        col_of(i, w); glUnmapNamedBuffer(ring);
        glBindBufferRange(GL_UNIFORM_BUFFER, 1, ring, off, 16);
        place(i); quad();
    }
    printf("  section 1: whole-buffer syncs %llu, range-synced skips %llu over 16 draws\n",
           (unsigned long long)(mgl_buf_sync_full - f0), (unsigned long long)(mgl_buf_sync_skipped - s0));
    bad += (mgl_buf_sync_full - f0) != 0 || (mgl_buf_sync_skipped - s0) < 15;

    /* ---- 2. glNamedBufferSubData into a large uniform buffer */
    GLuint big; glCreateBuffers(1, &big);
    glNamedBufferData(big, 64 * 1024, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, big, 0, 16);
    { float c[4]; col_of(0, c); glNamedBufferSubData(big, 0, 16, c); place(63); quad(); }
    glFinish();
    f0 = mgl_buf_sync_full; s0 = mgl_buf_sync_skipped;
    for (int i = 0; i < 16; i++) {
        float c[4]; col_of(i + 16, c);
        glNamedBufferSubData(big, (GLintptr)i * SLOT, 16, c);
        glBindBufferRange(GL_UNIFORM_BUFFER, 1, big, (GLintptr)i * SLOT, 16);
        place(16 + i); quad();
    }
    printf("  section 2: whole-buffer syncs %llu, range-synced skips %llu over 16 draws\n",
           (unsigned long long)(mgl_buf_sync_full - f0), (unsigned long long)(mgl_buf_sync_skipped - s0));
    bad += (mgl_buf_sync_full - f0) != 0;

    /* ---- 3. default-uniform vec4 per draw: an unsynced writer */
    glUseProgram(pd);
    glFinish();
    f0 = mgl_buf_sync_full;
    for (int i = 0; i < 8; i++) {
        float c[4]; col_of(32 + i, c);
        glUniform4fv(1, 1, c);
        place(32 + i); quad();
    }
    printf("  section 3: whole-buffer syncs %llu over 8 default-uniform draws\n", (unsigned long long)(mgl_buf_sync_full - f0));

    /* ---- 4. readback into a buffer, then read as a uniform block */
    {
        GLuint rb; glCreateBuffers(1, &rb);
        glNamedBufferData(rb, 4096, NULL, GL_DYNAMIC_COPY);
        glFinish();
        glBindBuffer(GL_PIXEL_PACK_BUFFER, rb);
        /* cell 32 holds draw 32's colour; read one RGBA32F texel of it */
        glReadPixels(32 % 8 * CELL + CELL / 2, 32 / 8 * CELL + CELL / 2, 1, 1, GL_RGBA, GL_FLOAT, (void *)0);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glUseProgram(pu);
        glBindBufferRange(GL_UNIFORM_BUFFER, 1, rb, 0, 16);
        place(48); quad();
    }

    grab();
    for (int i = 0; i < 16; i++) { float c[4]; col_of(i, c); bad += cell_is(i, u8(c[0]), u8(c[1]), u8(c[2]), "1 ring"); }
    for (int i = 0; i < 16; i++) { float c[4]; col_of(i + 16, c); bad += cell_is(16 + i, u8(c[0]), u8(c[1]), u8(c[2]), "2 sub-data"); }
    for (int i = 0; i < 8; i++) { float c[4]; col_of(32 + i, c); bad += cell_is(32 + i, u8(c[0]), u8(c[1]), u8(c[2]), "3 default uniform"); }
    { float c[4]; col_of(32, c); bad += cell_is(48, u8(c[0]), u8(c[1]), u8(c[2]), "4 readback buffer as uniforms"); }
    EXPECT_TRUE(bad == 0);
    printf("== r172 range-synced buffers: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
