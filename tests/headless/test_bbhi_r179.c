/* r179: the draw gate, the latch and the bind-map cache compare / copy the
 * indexed binding tables only up to the highest binding point ever set
 * (buffer_base_hi).  Every change between two draws of one program must still
 * reach the second draw:
 *   1. binding point 70 bound for the first time (the mark grows from 2 to 71)
 *      between two draws: the first reads an unbound block (black)
 *   2. two buffers alternated at binding point 70
 *   3. one buffer, its range at point 70 moved every draw (red / green halves)
 *   4. the buffer at point 70 deleted and a new one bound
 *   5. point 70 unbound again (back to black), then a block at point 1 moved
 * Points 0 and 1 hold a constant block throughout (the D3D9 pattern).
 * Pixels checked per cell. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#define W 64
#define CELL 8
#define SLOT 256

static const char *vs =
    "#version 450\nlayout(location=0) in vec2 p;\n"
    "void main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *fs =
    "#version 450\n"
    "layout(std140, binding=1) uniform A { vec4 k; } a;\n"
    "layout(std140, binding=70) uniform B { vec4 col; } b;\n"
    "out vec4 o;\nvoid main(){ o = vec4(b.col.rgb + a.k.rgb, 1.0); }\n";

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
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); }
    return p;
}
static GLuint ubo_col(float r, float g, float b)
{
    float c[4] = {r, g, b, 1};
    GLuint u; glGenBuffers(1, &u); glBindBuffer(GL_UNIFORM_BUFFER, u);
    glBufferData(GL_UNIFORM_BUFFER, sizeof c, c, GL_STATIC_DRAW);
    return u;
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
static void cell(int c) { glDrawArrays(GL_TRIANGLE_STRIP, c * 4, 4); }

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

    /* constants at points 0 and 1: zero (point 0 unused by the shader, bound like D3D9's VS block) */
    GLuint k0 = ubo_col(0, 0, 0), k1 = ubo_col(0, 0, 0), kb = ubo_col(0, 0, 1.0f);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, k0);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, k1);
    GLuint p = mk(); glUseProgram(p);
    cell(63); cell(63);                     /* the snapshot is latched with point 70 unbound */

    GLuint red = ubo_col(1, 0, 0), green = ubo_col(0, 1, 0);
    /* ---- 1. point 70 bound for the first time between two draws */
    cell(0);                                /* unbound: black */
    glBindBufferBase(GL_UNIFORM_BUFFER, 70, red);
    cell(1);                                /* red */
    /* ---- 2. two buffers alternated at point 70 */
    for (int i = 0; i < 6; i++) {
        glBindBufferBase(GL_UNIFORM_BUFFER, 70, (i & 1) ? green : red);
        cell(2 + i);
    }
    /* ---- 3. one buffer, its range at point 70 moved every draw */
    GLuint big; glGenBuffers(1, &big); glBindBuffer(GL_UNIFORM_BUFFER, big);
    glBufferData(GL_UNIFORM_BUFFER, 2 * SLOT, NULL, GL_DYNAMIC_DRAW);
    { float r[4] = {1, 0, 0, 1}, g[4] = {0, 1, 0, 1};
      glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, r); glBufferSubData(GL_UNIFORM_BUFFER, SLOT, 16, g); }
    for (int i = 0; i < 8; i++) {
        glBindBufferRange(GL_UNIFORM_BUFFER, 70, big, (i & 1) ? SLOT : 0, 16);
        cell(8 + i);
    }
    /* ---- 4. delete the buffer at point 70, bind a new one */
    glBindBufferBase(GL_UNIFORM_BUFFER, 70, red);
    cell(16);
    glDeleteBuffers(1, &red);               /* unbinds point 70: black */
    cell(17);
    GLuint white = ubo_col(1, 1, 1);
    glBindBufferBase(GL_UNIFORM_BUFFER, 70, white);
    cell(18);
    /* ---- 5. point 70 unbound, then the block at point 1 moved */
    glBindBufferBase(GL_UNIFORM_BUFFER, 70, 0);
    cell(19);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, kb);
    cell(20);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, k1);
    cell(21);

    grab();
    bad += cell_is(0, 0, 0, 0, "1 unbound");
    bad += cell_is(1, 255, 0, 0, "1 first bind");
    for (int i = 0; i < 6; i++)
        bad += cell_is(2 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "2 alternate");
    for (int i = 0; i < 8; i++)
        bad += cell_is(8 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "3 range");
    bad += cell_is(16, 255, 0, 0, "4 before delete");
    bad += cell_is(17, 0, 0, 0, "4 deleted");
    bad += cell_is(18, 255, 255, 255, "4 new buffer");
    bad += cell_is(19, 0, 0, 0, "5 unbound");
    bad += cell_is(20, 0, 0, 255, "5 point 1 moved");
    bad += cell_is(21, 0, 0, 0, "5 point 1 back");
    EXPECT_TRUE(bad == 0);
    printf("== r179 bounded binding tables: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
