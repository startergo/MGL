/* r163: glClearColor / glClearDepth / glClearStencil and glActiveTexture no longer
 * dirty draw state.
 *   1. a run of identical draws with those calls between each draw stays on the
 *      draw fast path (counted with mgl_compat_fast_miss) and draws correctly
 *   2. clear values set between draws are still honoured by the next glClear
 *      (colour, depth and stencil, checked through depth / stencil tests)
 *   3. a unit selected with glActiveTexture then bound is sampled; selecting
 *      another unit without binding changes nothing; rebinding the unit does
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_miss;

#define W 64
#define CELL 8

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "layout(location=2) uniform float z;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, z, 1.0); }\n";
static const char *fs_tex =
    "#version 450\nlayout(binding=1) uniform sampler2D t;\nout vec4 o;\nvoid main(){ o = vec4(texture(t, vec2(0.5)).rgb, 1.0); }\n";
static const char *fs_col =
    "#version 450\nlayout(location=3) uniform vec4 c;\nout vec4 o;\nvoid main(){ o = c; }\n";

static GLuint mk(const char *v, const char *f)
{
    const char *s[2] = {v, f}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) {
        GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(sh, sizeof log, NULL, log); printf("  [compile] %s\n", log); }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); }
    return p;
}

static GLuint tex_rgb(unsigned char r, unsigned char g, unsigned char b)
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
static void place(int cell) { glUniform2f(0, (float)(cell % 8), (float)(cell / 8)); }
static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

int main(void)
{
    int bad = 0;
    GLuint fbo, ct, ds;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenRenderbuffers(1, &ds); glBindRenderbuffer(GL_RENDERBUFFER, ds);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, ds);
    EXPECT_TRUE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, W, W);
    float unitq[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof unitq, unitq, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint red = tex_rgb(255, 0, 0), green = tex_rgb(0, 255, 0), blue = tex_rgb(0, 0, 255);
    GLuint ptex = mk(vs_cell, fs_tex), pcol = mk(vs_cell, fs_col);

    /* ---- 1. clear-value and active-unit calls between identical draws */
    {
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(ptex); glUniform1f(2, 0.0f);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, green);
        glActiveTexture(GL_TEXTURE0);
        place(0); quad();                              /* warm-up: latches the fast-path snapshot */
        place(1); quad();
        uint64_t miss0 = mgl_compat_fast_miss;
        for (int i = 2; i < 64; i++) {
            glClearColor(0.25f, 0.5f, 0.75f, 1.0f);   /* D3D9 Clear setup, no glClear here */
            glClearDepth(1.0);
            glClearStencil(0);
            glActiveTexture(GL_TEXTURE3);              /* stage select without a bind */
            glActiveTexture(GL_TEXTURE0);
            place(i); quad();
        }
        uint64_t misses = mgl_compat_fast_miss - miss0;
        grab();
        for (int i = 0; i < 64; i++) bad += cell_is(i, 0, 255, 0, "1 draws");
        printf("  fast-path misses over 62 draws: %llu\n", (unsigned long long)misses);
        EXPECT_TRUE(misses <= 1);
        bad += misses > 1;
    }

    /* ---- 2. clear values set between draws are honoured by the next glClear */
    {
        glUseProgram(pcol);
        glClearColor(0, 0, 0, 1); glClearDepth(1.0); glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glUniform1f(2, 0.0f); glUniform4f(3, 1, 1, 1, 1); place(0); quad();
        glClearColor(0.0f, 0.0f, 1.0f, 1.0f);          /* set between draws ... */
        place(1); quad();
        glClear(GL_COLOR_BUFFER_BIT);                  /* ... used here */
        grab();
        bad += cell_is(0, 0, 0, 255, "2 colour clear");
        bad += cell_is(9, 0, 0, 255, "2 colour clear");

        /* depth: clear to 0.25, then a quad at depth 0.5 (z=0) fails GL_LESS,
         * a quad at depth 0.1 (z=-0.8) passes */
        glClearDepth(0.25);
        place(2); quad();                              /* a draw between set and clear */
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS);
        glUniform4f(3, 1, 0, 0, 1); glUniform1f(2, 0.0f); place(3); quad();
        glUniform4f(3, 0, 1, 0, 1); glUniform1f(2, -0.8f); place(4); quad();
        glDisable(GL_DEPTH_TEST);

        /* stencil: clear to 5, EQUAL 5 passes, EQUAL 6 fails */
        glClearStencil(5);
        glUniform1f(2, 0.0f); glUniform4f(3, 1, 1, 1, 1); place(5); quad();
        glClear(GL_STENCIL_BUFFER_BIT);
        glEnable(GL_STENCIL_TEST); glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilFunc(GL_EQUAL, 5, 0xff); glUniform4f(3, 1, 1, 0, 1); place(6); quad();
        glStencilFunc(GL_EQUAL, 6, 0xff); glUniform4f(3, 1, 0, 1, 1); place(7); quad();
        glDisable(GL_STENCIL_TEST);
        grab();
        bad += cell_is(2, 255, 255, 255, "2 draw between set and depth clear");
        bad += cell_is(3, 0, 0, 255, "2 depth clear 0.25 rejects 0.5");
        bad += cell_is(4, 0, 255, 0, "2 depth clear 0.25 passes 0.1");
        bad += cell_is(6, 255, 255, 0, "2 stencil clear 5 EQUAL 5");
        bad += cell_is(7, 0, 0, 255, "2 stencil clear 5 rejects EQUAL 6");
    }

    /* ---- 3. active-unit selection with and without binds */
    {
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(ptex); glUniform1f(2, 0.0f);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, blue);   /* unit 0: not sampled */
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, red);
        place(0); quad();                                                    /* red */
        glActiveTexture(GL_TEXTURE0);                                        /* select only */
        place(1); quad();                                                    /* still red */
        glBindTexture(GL_TEXTURE_2D, green);                                 /* unit 0: not sampled */
        place(2); quad();                                                    /* still red */
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, blue);
        place(3); quad();                                                    /* blue */
        glActiveTexture(GL_TEXTURE2);
        place(4); quad();                                                    /* still blue */
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, green);
        glActiveTexture(GL_TEXTURE5);
        place(5); quad();                                                    /* green */
        GLint b1 = -1; glActiveTexture(GL_TEXTURE1); glGetIntegerv(GL_TEXTURE_BINDING_2D, &b1);
        EXPECT_TRUE(b1 == (GLint)green); bad += b1 != (GLint)green;
        grab();
        bad += cell_is(0, 255, 0, 0, "3 unit1 red");
        bad += cell_is(1, 255, 0, 0, "3 select unit0 only");
        bad += cell_is(2, 255, 0, 0, "3 bind on unsampled unit0");
        bad += cell_is(3, 0, 0, 255, "3 unit1 blue");
        bad += cell_is(4, 0, 0, 255, "3 select unit2 only");
        bad += cell_is(5, 0, 255, 0, "3 unit1 green then select unit5");
    }

    EXPECT_TRUE(bad == 0);
    printf("== r163 clear/activetex: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
