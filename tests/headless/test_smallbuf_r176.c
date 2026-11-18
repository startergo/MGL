/* r176: small (< 4 KB) buffers get Metal storage at their first bind instead of
 * being streamed through the argument ring at every bind.  Every way of changing
 * such a buffer between draws must still reach the next draw:
 *   1. glBufferSubData on the colour stream before each draw
 *   2. glMapBufferRange (write, invalidate range) on the colour stream
 *   3. glBufferData re-specification (same size, new contents)
 *   4. glBufferData re-specification to another size (8 vertices <-> 4)
 *   5. a small uniform block rewritten with glBufferSubData before each draw
 *   6. two small colour buffers alternated, neither rewritten
 * Each draw fills one cell; pixels checked per cell; promotions counted. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_small_promoted;

#define W 64
#define CELL 8

static const char *vs =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=1) in vec4 c;\n"
    "layout(std140, binding=0) uniform U { vec4 tint; } u;\n"
    "layout(location=0) out vec4 vc;\n"
    "void main(){ vc = c * u.tint; gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *fs =
    "#version 450\nlayout(location=0) in vec4 vc;\nout vec4 o;\nvoid main(){ o = vc; }\n";

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
static void fill4(float *c, float r, float g, float b, int n)
{
    for (int i = 0; i < n; i++) { c[4*i] = r; c[4*i+1] = g; c[4*i+2] = b; c[4*i+3] = 1; }
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

    /* positions: 64 quads in one large buffer; cell chosen with `first` */
    float quads[64 * 8];
    for (int c = 0; c < 64; c++) {
        float x0 = (float)(c % 8) * 0.25f - 1.0f, y0 = (float)(c / 8) * 0.25f - 1.0f;
        float q[8] = {x0, y0, x0 + 0.25f, y0, x0, y0 + 0.25f, x0 + 0.25f, y0 + 0.25f};
        memcpy(&quads[c * 8], q, sizeof q);
    }
    GLuint vao, vpos; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vpos); glBindBuffer(GL_ARRAY_BUFFER, vpos); glBufferData(GL_ARRAY_BUFFER, sizeof quads, quads, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    /* colour stream: a SMALL buffer, 4 vertices x vec4, read at vertex 0..3 via base vertex */
    float col[8 * 4];
    GLuint vcol, vcol2; glGenBuffers(1, &vcol); glGenBuffers(1, &vcol2);
    fill4(col, 1, 0, 0, 4);
    glBindBuffer(GL_ARRAY_BUFFER, vcol); glBufferData(GL_ARRAY_BUFFER, 4 * 16, col, GL_DYNAMIC_DRAW);
    fill4(col, 0, 0, 1, 4);
    glBindBuffer(GL_ARRAY_BUFFER, vcol2); glBufferData(GL_ARRAY_BUFFER, 4 * 16, col, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vcol);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(1);

    /* small uniform block */
    float tint[4] = {1, 1, 1, 1};
    GLuint ub; glGenBuffers(1, &ub); glBindBuffer(GL_UNIFORM_BUFFER, ub);
    glBufferData(GL_UNIFORM_BUFFER, sizeof tint, tint, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ub);

    GLuint p = mk(); glUseProgram(p);
    /* the colour attribute reads vertices 0..3 whatever cell is drawn: positions come
     * from `first`, so give the colour attribute divisor 0 and draw via base vertex
     * on an index list instead */
    uint16_t idx[4] = {0, 1, 2, 3};
    GLuint ib; glGenBuffers(1, &ib); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    /* positions per cell: re-point attribute 0 at the cell's quad */
#define CELLDRAW(c) do { glBindBuffer(GL_ARRAY_BUFFER, vpos); \
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)((size_t)(c) * 8 * sizeof(float))); \
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0); } while (0)

    CELLDRAW(63); CELLDRAW(63);                 /* first binds: promoted on a stream walk */
    uint64_t pr0 = mgl_small_promoted;

    /* ---- 1. glBufferSubData before each draw (red / green) */
    for (int i = 0; i < 8; i++) {
        fill4(col, (i & 1) ? 0.f : 1.f, (i & 1) ? 1.f : 0.f, 0, 4);
        glBindBuffer(GL_ARRAY_BUFFER, vcol); glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * 16, col);
        CELLDRAW(i);
    }
    /* ---- 2. map / write / unmap */
    for (int i = 0; i < 8; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, vcol);
        float *m = glMapBufferRange(GL_ARRAY_BUFFER, 0, 4 * 16, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT);
        if (!m) { printf("  map failed\n"); bad++; break; }
        fill4(m, (i & 1) ? 0.f : 1.f, 0, (i & 1) ? 1.f : 0.f, 4);
        glUnmapBuffer(GL_ARRAY_BUFFER);
        CELLDRAW(8 + i);
    }
    /* ---- 3. glBufferData, same size */
    for (int i = 0; i < 8; i++) {
        fill4(col, 0, (i & 1) ? 0.f : 1.f, (i & 1) ? 1.f : 0.f, 4);
        glBindBuffer(GL_ARRAY_BUFFER, vcol); glBufferData(GL_ARRAY_BUFFER, 4 * 16, col, GL_DYNAMIC_DRAW);
        CELLDRAW(16 + i);
    }
    /* ---- 4. glBufferData, size alternating 8 / 4 vertices */
    for (int i = 0; i < 8; i++) {
        int n = (i & 1) ? 4 : 8;
        fill4(col, 1, (i & 1) ? 1.f : 0.f, (i & 1) ? 0.f : 1.f, n);
        glBindBuffer(GL_ARRAY_BUFFER, vcol); glBufferData(GL_ARRAY_BUFFER, n * 16, col, GL_DYNAMIC_DRAW);
        CELLDRAW(24 + i);
    }
    fill4(col, 1, 1, 1, 4);
    glBindBuffer(GL_ARRAY_BUFFER, vcol); glBufferData(GL_ARRAY_BUFFER, 4 * 16, col, GL_DYNAMIC_DRAW);
    /* ---- 5. small uniform block rewritten before each draw */
    for (int i = 0; i < 8; i++) {
        float t[4] = {(i & 1) ? 0.f : 1.f, (i & 1) ? 1.f : 0.f, 0, 1};
        glBindBuffer(GL_UNIFORM_BUFFER, ub); glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof t, t);
        CELLDRAW(32 + i);
    }
    glBindBuffer(GL_UNIFORM_BUFFER, ub); glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof tint, tint);
    /* ---- 6. two small colour buffers alternated (white vcol / blue vcol2) */
    for (int i = 0; i < 8; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, (i & 1) ? vcol2 : vcol);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
        CELLDRAW(40 + i);
    }

    grab();
    for (int i = 0; i < 8; i++)
        bad += cell_is(i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "1 subdata");
    for (int i = 0; i < 8; i++)
        bad += cell_is(8 + i, (i & 1) ? 0 : 255, 0, (i & 1) ? 255 : 0, "2 map");
    for (int i = 0; i < 8; i++)
        bad += cell_is(16 + i, 0, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, "3 bufferdata");
    for (int i = 0; i < 8; i++)
        bad += cell_is(24 + i, 255, (i & 1) ? 255 : 0, (i & 1) ? 0 : 255, "4 resize");
    for (int i = 0; i < 8; i++)
        bad += cell_is(32 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "5 uniform block");
    for (int i = 0; i < 8; i++)
        bad += cell_is(40 + i, (i & 1) ? 0 : 255, (i & 1) ? 0 : 255, 255, "6 alternate");
    printf("  small-buffer promotions: %llu (before the sections: %llu)\n",
           (unsigned long long)mgl_small_promoted, (unsigned long long)pr0);
    bad += mgl_small_promoted == 0;   /* promoted on a walk that binds the stream */
    EXPECT_TRUE(bad == 0);
    printf("== r176 small buffers on the GPU: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
