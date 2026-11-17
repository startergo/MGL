/* r173: the separate image / sampler binder keeps a per-texture resolution cache
 * (Texture.sep_img / sep_samp), and the texture partial latches only the moved
 * units.  Same SPIR-V as r171: image+sampler pairs at units 2 and 5; output is
 * unit 2's colour plus unit 5's texel at u = 1.25 (REPEAT: black, CLAMP: blue).
 * Each section changes something the cache keys on, between draws:
 *   1. unit 2 alternates red / green (textures come back: cache hits expected,
 *      and the texture partial latches without the vertex-array group)
 *   2. unit 2's texture is re-uploaded (glTexSubImage2D) red <-> green
 *   3. unit 2's texture is re-specified (glTexImage2D) red <-> green
 *   4. no sampler object on unit 5: the TEXTURE's wrap mode alternates
 *   5. a render pass on another framebuffer between every draw
 *   6. unit 5's sampler object deleted and re-created between draws
 * Pixels checked per cell. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "spv_sepsampler.h"

extern uint64_t mgl_compat_sep_hit, mgl_compat_sep_miss, mgl_compat_latch_part;

#define W 64
#define CELL 8

static GLuint spv(GLenum ty, const unsigned char *b, size_t n)
{
    GLuint s = glCreateShader(ty);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, b, (GLsizei)n);
    glSpecializeShader(s, "main", 0, NULL, NULL);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, sizeof log, NULL, log); printf("  [specialize] %s\n", log); }
    return s;
}
static GLuint tex_px(int w, const unsigned char *px)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}
static GLuint samp(GLint wrap)
{
    GLuint s; glGenSamplers(1, &s);
    glSamplerParameteri(s, GL_TEXTURE_WRAP_S, wrap);
    glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return s;
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
static void cell_quad(int cell)
{
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)((size_t)cell * 8 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

int main(void)
{
    int bad = 0;
    GLuint fbo, ct, fbo2, ct2;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenTextures(1, &ct2); glBindTexture(GL_TEXTURE_2D, ct2); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo2); glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct2, 0);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);

    float quads[64 * 8];
    for (int c = 0; c < 64; c++) {
        float x0 = (float)(c % 8) * 0.25f - 1.0f, y0 = (float)(c / 8) * 0.25f - 1.0f;
        float q[8] = {x0, y0, x0 + 0.25f, y0, x0, y0 + 0.25f, x0 + 0.25f, y0 + 0.25f};
        memcpy(&quads[c * 8], q, sizeof q);
    }
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof quads, quads, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint p = glCreateProgram();
    glAttachShader(p, spv(GL_VERTEX_SHADER, spv_sep_vert, sizeof spv_sep_vert));
    glAttachShader(p, spv(GL_FRAGMENT_SHADER, spv_sep_frag, sizeof spv_sep_frag));
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); return 1; }
    glUseProgram(p);

    unsigned char R[4] = {255, 0, 0, 255}, G[4] = {0, 255, 0, 255};
    unsigned char KB[8] = {0, 0, 0, 255, 0, 0, 255, 255};    /* black | blue */
    GLuint red = tex_px(1, R), green = tex_px(1, G), kb = tex_px(2, KB), up = tex_px(1, R);
    GLuint srep = samp(GL_REPEAT), s0 = samp(GL_REPEAT);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);  glBindSampler(2, s0);
    glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, kb);   glBindSampler(5, srep);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, green); cell_quad(63);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, up);    cell_quad(63);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);   cell_quad(63);
    glClear(GL_COLOR_BUFFER_BIT);

    /* ---- 1. unit 2 alternates red / green */
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    cell_quad(0);
    uint64_t h0 = mgl_compat_sep_hit, mi0 = mgl_compat_sep_miss, l0 = mgl_compat_latch_part;
    for (int i = 0; i < 8; i++) {
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, (i & 1) ? green : red);
        glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4);   /* same vertex arrays: no rebind */
    }
    printf("  section 1: cache hits %llu, misses %llu, partial latches %llu over 8 draws\n",
           (unsigned long long)(mgl_compat_sep_hit - h0), (unsigned long long)(mgl_compat_sep_miss - mi0),
           (unsigned long long)(mgl_compat_latch_part - l0));
    bad += (mgl_compat_sep_hit - h0) < 8 || (mgl_compat_latch_part - l0) < 6;

    /* ---- 2. unit 2's texture re-uploaded every draw */
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, up);
    for (int i = 0; i < 8; i++) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, (i & 1) ? G : R);
        cell_quad(8 + i);
    }
    /* ---- 3. unit 2's texture re-specified every draw */
    for (int i = 0; i < 8; i++) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, (i & 1) ? G : R);
        cell_quad(16 + i);
    }
    glBindTexture(GL_TEXTURE_2D, red);
    cell_quad(63);

    /* ---- 4. no sampler object on unit 5: the texture's own wrap mode decides */
    glBindSampler(5, 0);
    glActiveTexture(GL_TEXTURE5);
    for (int i = 0; i < 8; i++) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, (i & 1) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        cell_quad(24 + i);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glBindSampler(5, srep);

    /* ---- 5. a pass on another framebuffer between every draw */
    for (int i = 0; i < 8; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, (i & 1) ? red : green);
        cell_quad(0);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glBindTexture(GL_TEXTURE_2D, (i & 1) ? green : red);
        cell_quad(32 + i);
    }

    /* ---- 6. unit 5's sampler object deleted / re-created (REPEAT <-> CLAMP) */
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);
    for (int i = 0; i < 8; i++) {
        glDeleteSamplers(1, &srep);
        srep = samp((i & 1) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glBindSampler(5, srep);
        cell_quad(40 + i);
    }

    grab();
    for (int i = 0; i < 8; i++)
        bad += cell_is(i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "1 alternate");
    for (int i = 0; i < 8; i++)
        bad += cell_is(8 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "2 re-upload");
    for (int i = 0; i < 8; i++)
        bad += cell_is(16 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "3 re-specify");
    for (int i = 0; i < 8; i++)
        bad += cell_is(24 + i, 255, 0, (i & 1) ? 255 : 0, "4 texture wrap");
    for (int i = 0; i < 8; i++)
        bad += cell_is(32 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "5 pass break");
    for (int i = 0; i < 8; i++)
        bad += cell_is(40 + i, 255, 0, (i & 1) ? 255 : 0, "6 sampler re-created");
    EXPECT_TRUE(bad == 0);
    printf("== r173 separate resolve cache / partial latch: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
