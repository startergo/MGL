/* r168: switching between small vertex buffers (< 4096 bytes: no Metal storage,
 * streamed by value) with an unchanged layout takes the vertex-array rebind path,
 * not the full walk.  The D3D7/9 stream pattern rsvga produces:
 *   1. eight small vertex buffers (float2 position + BGRA8 normalized colour,
 *      stride 12), one quad each; every draw switches the vertex buffer AND the
 *      index buffer (eight small index buffers)
 *   2. the same, with the buffer's contents rewritten (glBufferSubData) right
 *      before it is drawn — a dirty small stream on the rebind path
 *   3. the same with the sampled texture switched every draw (rebind folded into
 *      the texture partial walk)
 *   4. a switch from a small buffer to a Metal-backed one (>= 4096 bytes) still
 *      takes the full walk and draws right
 * Pixels checked per cell; sections 1-3 count fast-path misses. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_miss;

#define W 64
#define CELL 8

typedef struct { float x, y; uint8_t bgra[4]; } Vtx;   /* 12 bytes */

static const char *vs_src =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=1) in vec4 c;\nout vec4 vc;\n"
    "void main(){ gl_Position = vec4(p, 0.5, 1.0); vc = c; }\n";
static const char *fs_src =
    "#version 450\nin vec4 vc;\nout vec4 o;\nvoid main(){ o = vc; }\n";
static const char *fs_tex_src =
    "#version 450\nin vec4 vc;\nlayout(binding=1) uniform sampler2D t;\nout vec4 o;\n"
    "void main(){ o = vec4(vc.rgb + texture(t, vec2(0.5)).rgb, 1.0); }\n";

static GLuint mk(const char *fs)
{
    const char *s[2] = {vs_src, fs}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
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

/* quad for cell c with colour (r,g,b) in BGRA order */
static void make_quad(Vtx *q, int cell, uint8_t r, uint8_t g, uint8_t b)
{
    float x0 = (float)(cell % 8) * 0.25f - 1.0f, y0 = (float)(cell / 8) * 0.25f - 1.0f;
    float xs[4] = {x0, x0 + 0.25f, x0, x0 + 0.25f}, ys[4] = {y0, y0, y0 + 0.25f, y0 + 0.25f};
    for (int v = 0; v < 4; v++) { q[v].x = xs[v]; q[v].y = ys[v]; q[v].bgra[0] = b; q[v].bgra[1] = g; q[v].bgra[2] = r; q[v].bgra[3] = 255; }
}
static void colour_of(int i, uint8_t *r, uint8_t *g, uint8_t *b) { *r = (uint8_t)(i * 4); *g = (uint8_t)(255 - i * 4); *b = (uint8_t)((i & 1) ? 200 : 40); }

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

static GLuint vbs[8], ibs[8];
static void attribs(void)
{
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vtx), (void *)0);
    glVertexAttribPointer(1, GL_BGRA, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vtx), (void *)8);
}

/* run 64 draws; mode 0 plain, 1 rewrite before draw, 2 texture switch; returns misses over draws 2..63 */
static uint64_t run(int mode, GLuint *tex)
{
    uint64_t m0 = 0;
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 64; i++) {
        if (i == 2) m0 = mgl_compat_fast_miss;
        int k = i % 8;
        glBindBuffer(GL_ARRAY_BUFFER, vbs[k]);
        Vtx q[4]; uint8_t r, g, b; colour_of(i, &r, &g, &b); make_quad(q, i, r, g, b);
        if (mode == 1 || i < 8 || mode == 2)
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof q, q);     /* this draw's quad into buffer k */
        attribs();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibs[k]);
        if (mode == 2) { glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, tex[i % 2]); }
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)0);
    }
    return mgl_compat_fast_miss - m0;
}

int main(void)
{
    int bad = 0;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);
    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);

    glGenBuffers(8, vbs); glGenBuffers(8, ibs);
    for (int k = 0; k < 8; k++) {
        glBindBuffer(GL_ARRAY_BUFFER, vbs[k]);
        glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(Vtx), NULL, GL_DYNAMIC_DRAW);   /* 48 bytes */
        /* index buffers differ in content (different winding) but all draw the quad */
        uint16_t idx[6] = {0, 1, 2, 2, 1, 3};
        if (k & 1) { idx[0] = 1; idx[1] = 3; idx[2] = 0; idx[3] = 0; idx[4] = 3; idx[5] = 2; }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibs[k]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    }
    GLuint prog = mk(fs_src), ptex = mk(fs_tex_src);

    /* ---- 1. eight small vertex buffers, switched every draw (contents fixed after the first 8) */
    glUseProgram(prog);
    /* fill: buffer k holds the quad of the LAST draw that uses it in a pass of 64 → rewrite per pass */
    uint64_t m1 = run(1, NULL);   /* warm-up pass with rewrites: correctness only */
    grab();
    for (int i = 0; i < 64; i++) { uint8_t r, g, b; colour_of(i, &r, &g, &b); bad += cell_is(i, r, g, b, "1a rewrite per draw"); }
    printf("  section 2 (rewrite before each draw): fast-path misses %llu over 62 draws\n", (unsigned long long)m1);
    /* section 1 proper: buffers hold draws 56..63's quads; draw them 8 times over, switching each draw */
    {
        uint64_t m0 = 0;
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < 64; i++) {
            if (i == 2) m0 = mgl_compat_fast_miss;
            int k = i % 8;
            glBindBuffer(GL_ARRAY_BUFFER, vbs[k]); attribs();
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibs[(k + 3) % 8]);   /* index buffers switch independently */
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)0);
        }
        uint64_t m = mgl_compat_fast_miss - m0;
        printf("  section 1 (switch only): fast-path misses %llu over 62 draws\n", (unsigned long long)m);
        bad += m > 2;
        grab();
        for (int k = 0; k < 8; k++) { uint8_t r, g, b; colour_of(56 + k, &r, &g, &b); bad += cell_is(56 + k, r, g, b, "1 switch only"); }
        for (int c = 0; c < 56; c++) bad += cell_is(c, 0, 0, 0, "1 untouched cell");
    }
    bad += m1 > 2;

    /* ---- 3. with a texture switch every draw */
    {
        GLuint tx[2]; glGenTextures(2, tx);
        for (int t = 0; t < 2; t++) {
            unsigned char px[4] = {0, 0, 0, 255};
            glBindTexture(GL_TEXTURE_2D, tx[t]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }
        glUseProgram(ptex);
        uint64_t m = run(2, tx);
        printf("  section 3 (texture switch too): fast-path misses %llu over 62 draws\n", (unsigned long long)m);
        bad += m > 2;
        grab();
        for (int i = 0; i < 64; i++) { uint8_t r, g, b; colour_of(i, &r, &g, &b); bad += cell_is(i, r, g, b, "3 texture switch"); }
    }

    /* ---- 4. small -> Metal-backed buffer: full walk, right pixels */
    {
        glUseProgram(prog);
        GLuint big; glGenBuffers(1, &big); glBindBuffer(GL_ARRAY_BUFFER, big);
        glBufferData(GL_ARRAY_BUFFER, 8192, NULL, GL_STATIC_DRAW);
        Vtx q[4]; make_quad(q, 20, 255, 255, 255);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof q, q);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, vbs[0]); attribs(); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibs[0]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, big); attribs();
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void *)0);
        grab();
        bad += cell_is(20, 255, 255, 255, "4 small -> Metal-backed");
    }

    EXPECT_TRUE(bad == 0);
    printf("== r168 small vertex buffer switches: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
