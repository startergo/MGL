/* r166: a uniform-buffer re-bind to the SAME buffer at a new offset (a D3D9-style
 * constant stream) takes a partial path instead of the full state walk, and every
 * draw still reads its own range.
 *   1. VS block at point 0 and FS block at point 300 (the overflow table, where
 *      rsvga puts its pixel constants) re-bound at a new offset per draw
 *   2. the same with the vertex array re-specified at a new offset per draw
 *      (vertex-array rebind + uniform ranges in one partial)
 *   3. a small buffer (no Metal storage: the argument ring) at moving offsets
 *   4. content written with glBufferSubData AND the range moved between draws
 *   5. a re-bind to a DIFFERENT buffer object still takes the full walk and reads
 *      the new buffer
 *   6. the D3D9 pattern in one draw: vertex array moved, the sampled texture
 *      switched and both ranges moved (vertex-array rebind folded into the
 *      texture partial walk)
 *   7. the constant RING rsvga will use: one 1 MB buffer, each draw's constants
 *      written through glMapNamedBufferRange(WRITE | INVALIDATE_RANGE |
 *      UNSYNCHRONIZED) at a fresh offset, then bound there.  No storage rename
 *      (new Metal buffers), every draw partial, every cell right.
 * Each section checks every cell's colour; section 1 also counts the partial
 * verdicts (mgl_compat_fast_uboofs) and the fast-path misses. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_uboofs, mgl_compat_fast_miss, mgl_bufpool_stat_alloc;

#define W 64
#define CELL 8
#define SLOT 256          /* GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT-safe stride */

static const char *vs_src =
    "#version 450\nlayout(location=0) in vec2 p;\n"
    "layout(std140, binding=0) uniform VSC { vec4 cell; vec4 tint; } vc;\nout vec4 vt;\n"
    "void main(){ vec2 q = (vc.cell.xy + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); vt = vc.tint; }\n";
static const char *fs_src =
    "#version 450\nlayout(std140, binding=300) uniform PSC { vec4 col; } pc;\nin vec4 vt;\nout vec4 o;\n"
    "void main(){ o = vec4(pc.col.rgb + vt.rgb, 1.0); }\n";

static const char *fs_tex_src =
    "#version 450\nlayout(std140, binding=300) uniform PSC { vec4 col; } pc;\nlayout(binding=2) uniform sampler2D t;\n"
    "in vec4 vt;\nout vec4 o;\n"
    "void main(){ o = vec4(pc.col.rgb + vt.rgb + texture(t, vec2(0.5)).rgb, 1.0); }\n";

static GLuint mk_fs(const char *fs)
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
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); }
    return p;
}
static GLuint mk(void) { return mk_fs(fs_src); }

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

/* draw i: VS slot = cell (i%8, i/8) + tint (i%4)*0.25 red; FS slot = blue i/63 */
static void slot_vs(float *f, int i) { memset(f, 0, 32); f[0] = (float)(i % 8); f[1] = (float)(i / 8); f[4] = (float)(i % 4) * 0.25f; }
static void slot_fs(float *f, int i) { memset(f, 0, 16); f[2] = (float)i / 63.0f; f[1] = (i & 1) ? 1.0f : 0.0f; }
static int want_r(int i) { return (int)((float)(i % 4) * 0.25f * 255.0f + 0.5f); }
static int want_g(int i) { return (i & 1) ? 255 : 0; }
static int want_b(int i) { return (int)((float)i / 63.0f * 255.0f + 0.5f); }

static int check_all(int n, const char *what)
{
    int bad = 0;
    grab();
    for (int i = 0; i < n; i++) bad += cell_is(i, want_r(i), want_g(i), want_b(i), what);
    return bad;
}

int main(void)
{
    int bad = 0;
    GLint align = 0; glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &align);
    EXPECT_TRUE(align > 0 && SLOT % align == 0);
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    glClearColor(0, 0, 0, 1);

    /* vertex data: 64 copies of the unit quad, so the VAO can move per draw */
    float quadv[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 64 * sizeof quadv, NULL, GL_STATIC_DRAW);
    for (int i = 0; i < 64; i++) glBufferSubData(GL_ARRAY_BUFFER, i * sizeof quadv, sizeof quadv, quadv);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    /* scratch constant buffer: 64 VS slots then 64 FS slots */
    unsigned char *cb = calloc(1, 128 * SLOT);
    for (int i = 0; i < 64; i++) { slot_vs((float *)(cb + i * SLOT), i); slot_fs((float *)(cb + (64 + i) * SLOT), i); }
    GLuint ub; glGenBuffers(1, &ub); glBindBuffer(GL_UNIFORM_BUFFER, ub);
    glBufferData(GL_UNIFORM_BUFFER, 128 * SLOT, cb, GL_STATIC_DRAW);

    GLuint prog = mk(); glUseProgram(prog);

    /* ---- 1. same buffer, new offset, both stages */
    glClear(GL_COLOR_BUFFER_BIT);
    uint64_t u0 = 0, m0 = 0;
    for (int i = 0; i < 64; i++) {
        if (i == 2) { u0 = mgl_compat_fast_uboofs; m0 = mgl_compat_fast_miss; }
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub, (GLintptr)i * SLOT, 32);
        glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub, (GLintptr)(64 + i) * SLOT, 16);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    uint64_t partial = mgl_compat_fast_uboofs - u0, misses = mgl_compat_fast_miss - m0;
    printf("  section 1: ubo-range partials %llu, fast-path misses %llu over 62 draws\n",
           (unsigned long long)partial, (unsigned long long)misses);
    EXPECT_TRUE(partial >= 60 && misses <= 2);
    bad += !(partial >= 60 && misses <= 2);
    bad += check_all(64, "1 ranges");

    /* ---- 2. vertex array moved too */
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 64; i++) {
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)(((i * 7) % 64) * sizeof quadv));
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub, (GLintptr)i * SLOT, 32);
        glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub, (GLintptr)(64 + i) * SLOT, 16);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    bad += check_all(64, "2 ranges + vertex array");
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* ---- 3. a small buffer (argument ring) at moving offsets: 4 slots */
    {
        GLuint sb; glGenBuffers(1, &sb); glBindBuffer(GL_UNIFORM_BUFFER, sb);
        unsigned char small[8 * SLOT];
        memset(small, 0, sizeof small);
        glBufferData(GL_UNIFORM_BUFFER, sizeof small, NULL, GL_DYNAMIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < 64; i++) {
            int k = i % 4;
            slot_vs((float *)(small + k * SLOT), i); slot_fs((float *)(small + (4 + k) * SLOT), i);
            glBufferSubData(GL_UNIFORM_BUFFER, k * SLOT, 32, small + k * SLOT);
            glBufferSubData(GL_UNIFORM_BUFFER, (4 + k) * SLOT, 16, small + (4 + k) * SLOT);
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, sb, (GLintptr)k * SLOT, 32);
            glBindBufferRange(GL_UNIFORM_BUFFER, 300, sb, (GLintptr)(4 + k) * SLOT, 16);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        bad += check_all(64, "3 small buffer, content + range");
    }

    /* ---- 4. big buffer: content rewritten at the new range before each draw */
    glBindBuffer(GL_UNIFORM_BUFFER, ub);
    glClear(GL_COLOR_BUFFER_BIT);
    for (int i = 0; i < 64; i++) {
        int k = 63 - i;                               /* the slot that held draw k's data */
        float vsd[8], fsd[4];
        slot_vs(vsd, i); slot_fs(fsd, i);
        glBufferSubData(GL_UNIFORM_BUFFER, (GLintptr)k * SLOT, sizeof vsd, vsd);
        glBufferSubData(GL_UNIFORM_BUFFER, (GLintptr)(64 + k) * SLOT, sizeof fsd, fsd);
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub, (GLintptr)k * SLOT, 32);
        glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub, (GLintptr)(64 + k) * SLOT, 16);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    bad += check_all(64, "4 content + range");

    /* ---- 5. a different buffer object: full walk, new contents */
    {
        GLuint ub2; glGenBuffers(1, &ub2); glBindBuffer(GL_UNIFORM_BUFFER, ub2);
        unsigned char *cb2 = calloc(1, 2 * SLOT);
        slot_vs((float *)cb2, 9);   /* cell 9 */
        float *f = (float *)(cb2 + SLOT); f[0] = 1.0f; f[1] = 1.0f; f[2] = 1.0f;   /* white */
        glBufferData(GL_UNIFORM_BUFFER, 2 * SLOT, cb2, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub, 0, 32);
        glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub, 64 * SLOT, 16);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);                                   /* cell 0, draw-0 colours */
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub2, 0, 32);
        glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub2, SLOT, 16);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);                                   /* cell 9, white */
        grab();
        bad += cell_is(0, want_r(0), want_g(0), want_b(0), "5 first buffer");
        bad += cell_is(9, 255, 255, 255, "5 another buffer");
        free(cb2);
    }

    /* ---- 6. vertex array + texture + ranges on every draw */
    {
        GLuint pt = mk_fs(fs_tex_src);
        glUseProgram(pt);
        /* textures add 0 / 64 / 128 to green: the draw's green is (i&1)*255 + that, so use only odd
         * cells' red/blue and check green separately for even cells */
        GLuint tx[3]; glGenTextures(3, tx);
        for (int k = 0; k < 3; k++) {
            unsigned char px[4] = {0, (unsigned char)(64 * k), 0, 255};
            glBindTexture(GL_TEXTURE_2D, tx[k]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }
        glBindBuffer(GL_UNIFORM_BUFFER, ub);
        glBufferData(GL_UNIFORM_BUFFER, 128 * SLOT, cb, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        uint64_t u0 = 0, m0 = 0;
        for (int i = 0; i < 64; i++) {
            if (i == 2) { u0 = mgl_compat_fast_uboofs; m0 = mgl_compat_fast_miss; }
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void *)(uintptr_t)(((i * 5) % 64) * sizeof quadv));
            glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, tx[i % 3]);
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, ub, (GLintptr)i * SLOT, 32);
            glBindBufferRange(GL_UNIFORM_BUFFER, 300, ub, (GLintptr)(64 + i) * SLOT, 16);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        printf("  section 6: ubo-range partials %llu, fast-path misses %llu over 62 draws\n",
               (unsigned long long)(mgl_compat_fast_uboofs - u0), (unsigned long long)(mgl_compat_fast_miss - m0));
        int b6 = (mgl_compat_fast_miss - m0) > 2;
        grab();
        for (int i = 0; i < 64; i++) {
            int g = want_g(i) + 64 * (i % 3); if (g > 255) g = 255;
            b6 += cell_is(i, want_r(i), g, want_b(i), "6 vertex array + texture + ranges");
        }
        bad += b6;
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    }

    /* ---- 7. constant ring: unsynchronized map at a fresh offset per draw */
    {
        glUseProgram(prog);
        GLuint ring; glCreateBuffers(1, &ring);
        const GLsizeiptr RING = 1 << 20;
        glNamedBufferStorage(ring, RING, NULL, GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
        glClear(GL_COLOR_BUFFER_BIT);
        GLintptr head = 0;
        uint64_t u0 = 0, m0 = 0, a0 = 0;
        for (int i = 0; i < 64; i++) {
            if (i == 2) { u0 = mgl_compat_fast_uboofs; m0 = mgl_compat_fast_miss; a0 = mgl_bufpool_stat_alloc; }
            GLintptr vo = head, fo = head + SLOT;
            head += 2 * SLOT;                         /* 64 draws x 512 B: no wrap in 1 MB */
            unsigned char *w = glMapNamedBufferRange(ring, vo, 2 * SLOT,
                                                     GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
            if (!w) { printf("  [7] map failed\n"); bad++; break; }
            slot_vs((float *)w, i); slot_fs((float *)(w + SLOT), i);
            glUnmapNamedBuffer(ring);
            glBindBufferRange(GL_UNIFORM_BUFFER, 0, ring, vo, 32);
            glBindBufferRange(GL_UNIFORM_BUFFER, 300, ring, fo, 16);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
        uint64_t partial = mgl_compat_fast_uboofs - u0, misses = mgl_compat_fast_miss - m0,
                 allocs = mgl_bufpool_stat_alloc - a0;
        printf("  section 7: ubo-range partials %llu, fast-path misses %llu, new Metal buffers %llu over 62 draws\n",
               (unsigned long long)partial, (unsigned long long)misses, (unsigned long long)allocs);
        int b7 = !(partial >= 60 && misses <= 2 && allocs == 0);
        b7 += check_all(64, "7 constant ring");
        bad += b7;
    }

    free(cb);
    EXPECT_TRUE(bad == 0);
    printf("== r166 uniform-buffer ranges: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
