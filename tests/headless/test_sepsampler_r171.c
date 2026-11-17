/* r171: programs with Vulkan-style SEPARATE images and samplers (rsvga's D3D9 and
 * DX10+ SPIR-V: OpTypeImage + OpTypeSampler per register, image N and sampler N
 * at the same GL unit) take the per-unit paths:
 *   - the texture partial walk re-binds only the moved units
 *   - the walk realizes only the units the program samples
 *   - the draw gate looks only at those units
 * Sixteen units stay bound (some never realized); the fragment shader samples
 * image/sampler pairs at units 2 and 5.  Between draws:
 *   1. the texture on unit 2 switches (red <-> green)
 *   2. the sampler object on unit 5 switches (REPEAT <-> CLAMP_TO_EDGE: decides
 *      whether t1, sampled at u = 1.25, reads its black or its blue texel)
 *   3. a parameter of unit 5's bound sampler object changes
 *   4. a texture on an unsampled unit switches (empty-mask partial: nothing to re-bind)
 *   5. a texture parameter of unit 2's texture changes (swizzle R <- G: a view)
 * Pixels checked per cell; sections 1-4 count moved-units-only rebinds and misses. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "spv_sepsampler.h"

extern uint64_t mgl_compat_fast_texunits, mgl_compat_fast_miss, mgl_compat_fast_tex;

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
/* one quad per cell, 64 quads in one Metal-backed buffer: moving the attribute
 * offset is a vertex-array rebind (same alignment class) */
static void cell_quad(int cell)
{
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)((size_t)cell * 8 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

int main(void)
{
    int bad = 0;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
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

    unsigned char R[4] = {255, 0, 0, 255}, G[4] = {0, 255, 0, 255}, Y[4] = {255, 255, 0, 255};
    unsigned char KB[8] = {0, 0, 0, 255, 0, 0, 255, 255};    /* black | blue */
    GLuint red = tex_px(1, R), green = tex_px(1, G), yellow = tex_px(1, Y), kb = tex_px(2, KB);
    GLuint other[16];
    for (int u = 0; u < 16; u++) {
        if (u == 2 || u == 5) continue;
        glGenTextures(1, &other[u]);
        glActiveTexture(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, other[u]);
        if (u != 12) {   /* unit 12 keeps a texture that is never given storage nor sampled */
            unsigned char px[4] = {(unsigned char)(u * 10), 0, 0, 255};
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
        }
    }
    GLuint srep = samp(GL_REPEAT), sclamp = samp(GL_CLAMP_TO_EDGE), s0 = samp(GL_REPEAT);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);  glBindSampler(2, s0);
    glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, kb);   glBindSampler(5, srep);
    /* realize every texture the program will sample before counting */
    cell_quad(63);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, green); cell_quad(63);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, yellow); cell_quad(63);
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);   cell_quad(63);
    glClear(GL_COLOR_BUFFER_BIT);

    uint64_t u0, m0, t0;
    /* ---- 1. unit 2's texture switches every draw */
    u0 = mgl_compat_fast_texunits; m0 = mgl_compat_fast_miss; t0 = mgl_compat_fast_tex;
    for (int i = 0; i < 8; i++) {
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, (i & 1) ? green : red);
        cell_quad(i);
    }
    printf("  section 1: texture partials %llu, moved-units-only %llu, misses %llu over 8 draws\n",
           (unsigned long long)(mgl_compat_fast_tex - t0), (unsigned long long)(mgl_compat_fast_texunits - u0),
           (unsigned long long)(mgl_compat_fast_miss - m0));
    bad += (mgl_compat_fast_texunits - u0) < 7 || (mgl_compat_fast_miss - m0) > 1;
    glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);      /* sections 2-4 sample red at unit 2 */
    cell_quad(63);
    /* ---- 2. unit 5's sampler object switches every draw */
    u0 = mgl_compat_fast_texunits; m0 = mgl_compat_fast_miss;
    for (int i = 0; i < 8; i++) {
        glBindSampler(5, (i & 1) ? sclamp : srep);
        cell_quad(8 + i);
    }
    printf("  section 2: moved-units-only %llu, misses %llu over 8 draws\n",
           (unsigned long long)(mgl_compat_fast_texunits - u0), (unsigned long long)(mgl_compat_fast_miss - m0));
    bad += (mgl_compat_fast_texunits - u0) < 7 || (mgl_compat_fast_miss - m0) > 1;
    /* ---- 3. a parameter of unit 5's sampler object changes every draw */
    glBindSampler(5, srep);
    u0 = mgl_compat_fast_texunits; m0 = mgl_compat_fast_miss;
    for (int i = 0; i < 8; i++) {
        glSamplerParameteri(srep, GL_TEXTURE_WRAP_S, (i & 1) ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        cell_quad(16 + i);
    }
    printf("  section 3: moved-units-only %llu, misses %llu over 8 draws\n",
           (unsigned long long)(mgl_compat_fast_texunits - u0), (unsigned long long)(mgl_compat_fast_miss - m0));
    bad += (mgl_compat_fast_texunits - u0) < 7 || (mgl_compat_fast_miss - m0) > 1;
    glSamplerParameteri(srep, GL_TEXTURE_WRAP_S, GL_REPEAT);
    /* ---- 4. a texture on an unsampled unit switches */
    m0 = mgl_compat_fast_miss; u0 = mgl_compat_fast_texunits;
    for (int i = 0; i < 8; i++) {
        glActiveTexture(GL_TEXTURE9); glBindTexture(GL_TEXTURE_2D, (i & 1) ? other[3] : other[9]);
        cell_quad(24 + i);
    }
    printf("  section 4: moved-units-only (empty) %llu, misses %llu over 8 draws\n",
           (unsigned long long)(mgl_compat_fast_texunits - u0), (unsigned long long)(mgl_compat_fast_miss - m0));
    bad += (mgl_compat_fast_texunits - u0) < 7 || (mgl_compat_fast_miss - m0) > 1;
    /* ---- 5. swizzle on unit 2's texture (red): R <- G gives black, back gives red */
    for (int i = 0; i < 4; i++) {
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, red);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, (i & 1) ? GL_GREEN : GL_RED);
        cell_quad(32 + i);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);

    grab();
    for (int i = 0; i < 8; i++)
        bad += cell_is(i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "1 unit-2 texture");
    for (int i = 0; i < 8; i++)
        bad += cell_is(8 + i, 255, 0, (i & 1) ? 255 : 0, "2 unit-5 sampler object");
    for (int i = 0; i < 8; i++)
        bad += cell_is(16 + i, 255, 0, (i & 1) ? 255 : 0, "3 sampler parameter");
    for (int i = 0; i < 8; i++)
        bad += cell_is(24 + i, 255, 0, 0, "4 unsampled unit switches");
    for (int i = 0; i < 4; i++)
        bad += cell_is(32 + i, (i & 1) ? 0 : 255, 0, 0, "5 swizzle");
    EXPECT_TRUE(bad == 0);
    printf("== r171 separate images / samplers: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
