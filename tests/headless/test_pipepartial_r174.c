/* r174: render-state changes on an unchanged program take a partial walk instead
 * of the full one.  A pipeline-baked change (blend, colour mask) looks the
 * pipeline up in the compat cache and binds it; an encoder-only change (cull,
 * scissor) re-applies the encoder state.  Between every two draws:
 *   1. blend func alternates (ONE,ZERO) / (ONE,ONE) over a blue background
 *   2. colour mask alternates all / no-red
 *   3. cull face alternates BACK / FRONT (the quads are front-facing)
 *   4. blend func and the sampled texture change on the same draw
 *   5. blend func and the vertex-array offset change on the same draw (rebind)
 *   6. the scissor box alternates the whole target / one pixel in a corner
 * Pixels checked per cell; full-walk misses and pipeline partials counted. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_miss, mgl_compat_fast_pipe;

#define W 64
#define CELL 8

static const char *vs =
    "#version 450\nlayout(location=0) in vec2 p;\n"
    "void main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *fs =
    "#version 450\nlayout(binding=0) uniform sampler2D t;\nout vec4 o;\n"
    "void main(){ o = vec4(texture(t, vec2(0.5)).rgb, 1.0); }\n";

static GLuint mk(void)
{
    const char *s[2] = {vs, fs}; GLenum ty[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) {
        GLuint sh = glCreateShader(ty[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(sh, sizeof log, NULL, log); printf("  [compile] %s\n", log); }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    return p;
}
static GLuint tex1(unsigned char r, unsigned char g, unsigned char b)
{
    unsigned char px[4] = {r, g, b, 255};
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
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
static void at_first(int cell) { glDrawArrays(GL_TRIANGLE_STRIP, cell * 4, 4); }
static void at_offset(int cell)
{
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)((size_t)cell * 8 * sizeof(float)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static uint64_t m0, p0;
static void mark(void) { m0 = mgl_compat_fast_miss; p0 = mgl_compat_fast_pipe; }
static int report(const char *what, int want_pipe)
{
    uint64_t mi = mgl_compat_fast_miss - m0, pp = mgl_compat_fast_pipe - p0;
    printf("  %s: full-walk misses %llu, pipeline partials %llu over 8 draws\n", what,
           (unsigned long long)mi, (unsigned long long)pp);
    return mi > 1 || (want_pipe && pp < 6);
}

int main(void)
{
    int bad = 0;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 128.0f / 255.0f, 1); glClear(GL_COLOR_BUFFER_BIT);

    float quads[64 * 8];
    for (int c = 0; c < 64; c++) {
        float x0 = (float)(c % 8) * 0.25f - 1.0f, y0 = (float)(c / 8) * 0.25f - 1.0f;
        float q[8] = {x0, y0, x0 + 0.25f, y0, x0, y0 + 0.25f, x0 + 0.25f, y0 + 0.25f};
        memcpy(&quads[c * 8], q, sizeof q);
    }
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof quads, quads, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
    GLuint p = mk(); glUseProgram(p);
    GLuint red = tex1(255, 0, 0), green = tex1(0, 255, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, red);

    /* build both blend pipelines (and the masked one) before counting */
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ZERO); at_first(63);
    glBlendFunc(GL_ONE, GL_ONE);  at_first(63);
    glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_TRUE); at_first(63);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glBlendFunc(GL_ONE, GL_ZERO); at_first(63);
    glClear(GL_COLOR_BUFFER_BIT);
    at_first(63);

    /* ---- 1. blend func */
    mark();
    for (int i = 0; i < 8; i++) {
        glBlendFunc(GL_ONE, (i & 1) ? GL_ONE : GL_ZERO);
        at_first(i);
    }
    bad += report("section 1 blend", 1);
    glBlendFunc(GL_ONE, GL_ZERO); at_first(63);

    /* ---- 2. colour mask */
    mark();
    for (int i = 0; i < 8; i++) {
        glColorMask((i & 1) ? GL_FALSE : GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        at_first(8 + i);
    }
    bad += report("section 2 colour mask", 1);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); at_first(63);

    /* ---- 3. cull face (encoder state only) */
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK); at_first(63);
    mark();
    for (int i = 0; i < 8; i++) {
        glCullFace((i & 1) ? GL_FRONT : GL_BACK);
        at_first(16 + i);
    }
    bad += report("section 3 cull", 0);
    glDisable(GL_CULL_FACE); at_first(63);

    /* ---- 4. blend + texture on the same draw */
    mark();
    for (int i = 0; i < 8; i++) {
        glBlendFunc(GL_ONE, (i & 1) ? GL_ONE : GL_ZERO);
        glBindTexture(GL_TEXTURE_2D, (i & 1) ? green : red);
        at_first(24 + i);
    }
    bad += report("section 4 blend + texture", 1);
    glBlendFunc(GL_ONE, GL_ZERO); glBindTexture(GL_TEXTURE_2D, red); at_first(63);

    /* ---- 5. blend + vertex-array offset on the same draw (rebind) */
    at_offset(63);
    mark();
    for (int i = 0; i < 8; i++) {
        glBlendFunc(GL_ONE, (i & 1) ? GL_ONE : GL_ZERO);
        at_offset(32 + i);
    }
    bad += report("section 5 blend + rebind", 1);
    glBlendFunc(GL_ONE, GL_ZERO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); at_first(63);

    /* ---- 6. scissor box (encoder state only) */
    glEnable(GL_SCISSOR_TEST); glScissor(0, 0, W, W); at_first(63);
    mark();
    for (int i = 0; i < 8; i++) {
        if (i & 1) glScissor(0, 0, 1, 1); else glScissor(0, 0, W, W);
        at_first(40 + i);
    }
    bad += report("section 6 scissor", 0);
    glScissor(0, 0, W, W); glDisable(GL_SCISSOR_TEST);

    grab();
    for (int i = 0; i < 8; i++)
        bad += cell_is(i, 255, 0, (i & 1) ? 128 : 0, "1 blend");
    for (int i = 0; i < 8; i++)
        bad += cell_is(8 + i, (i & 1) ? 0 : 255, 0, 0, "2 colour mask");
    for (int i = 0; i < 8; i++)
        bad += cell_is(16 + i, (i & 1) ? 0 : 255, 0, (i & 1) ? 128 : 0, "3 cull");
    for (int i = 0; i < 8; i++)
        bad += cell_is(24 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, (i & 1) ? 128 : 0, "4 blend + texture");
    for (int i = 0; i < 8; i++)
        bad += cell_is(32 + i, 255, 0, (i & 1) ? 128 : 0, "5 blend + rebind");
    for (int i = 0; i < 8; i++)
        bad += cell_is(40 + i, (i & 1) ? 0 : 255, 0, (i & 1) ? 128 : 0, "6 scissor");
    EXPECT_TRUE(bad == 0);
    printf("== r174 render-state partials: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
