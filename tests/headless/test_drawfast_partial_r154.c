/* r154: the per-draw fast path's partial walks render what a full walk would.
 *
 * Every section draws many small quads into one render pass (no readback in
 * between, so the draw-fast gate sees the same encoder / pipeline and takes its
 * partial verdicts), each quad into its own 8x8 cell of a 64x64 target, then
 * reads all cells back once:
 *   1. texture-binding-only changes between draws (four textures, glBindTexture
 *      with a same-VAO re-bind and a same-UBO glBindBufferBase in between — the
 *      0x4029 dirty mask a D3D9 translator produces)
 *   2. sampler-object and texture-parameter changes (address mode: REPEAT vs
 *      CLAMP_TO_EDGE decides the colour)
 *   3. vertex-stage texture switches (VS samples, FS passes the colour through)
 *   4. vertex-buffer content updates between draws (glBufferSubData on a small
 *      ring-backed VBO and on a >= 4 KB MTLBuffer-backed one, and an orphaning
 *      same-size glBufferData)
 *   5. uniform-buffer content updates while the previous draw still references
 *      the storage (pool rename: same buffer object, new Metal storage)
 *   6. texture + uniform buffer + vertex buffer all moving on every draw
 *   7. encoder-state (stencil reference) changes with a uniform-buffer update
 *      on every draw; 7b the same with the blend constant (the rstate partial
 *      path must flush the listed buffers)
 *   2 also covers a glSamplerParameter on the BOUND sampler object: its new
 *      sampler-state cache can land at the released one's address (r154 adds
 *      TextureParameter.mtl_serial to every identity cache keyed on it)
 *   8. texture content update without a rebind (must still show)
 * Then checks that the partial verdicts were actually taken (r154 counters). */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_tex, mgl_compat_fast_bufs, mgl_compat_fast_moved, mgl_compat_fast_rs;

#define W 64
#define CELL 8

/* quad from the bound VBO (unit square 0..1), placed in cell `cell` */
static const char *vs_cell =
    "#version 450\n"
    "layout(location=0) in vec2 p;\n"
    "layout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs_tex =
    "#version 450\nlayout(binding=0) uniform sampler2D t;\nlayout(location=1) uniform vec2 uv;\n"
    "out vec4 o;\nvoid main(){ o = texture(t, uv); }\n";
static const char *fs_ubo =
    "#version 450\nlayout(std140, binding=2) uniform C { vec4 col; };\nout vec4 o;\nvoid main(){ o = col; }\n";
static const char *fs_tex_ubo =
    "#version 450\nlayout(binding=0) uniform sampler2D t;\nlayout(std140, binding=2) uniform C { vec4 col; };\n"
    "out vec4 o;\nvoid main(){ o = texture(t, vec2(0.5)) + col; }\n";
/* vertex-stage texture: colour fetched in the VS */
static const char *vs_vtex =
    "#version 450\n"
    "layout(location=0) in vec2 p;\n"
    "layout(location=0) uniform vec2 cell;\n"
    "layout(binding=1) uniform sampler2D vt;\n"
    "out vec4 c;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0);\n"
    "             c = textureLod(vt, vec2(0.5), 0.0); }\n";
static const char *fs_pass =
    "#version 450\nin vec4 c;\nout vec4 o;\nvoid main(){ o = c; }\n";
/* VBO carries absolute NDC positions (section 4) */
static const char *vs_abs =
    "#version 450\nlayout(location=0) in vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *fs_col =
    "#version 450\nlayout(location=2) uniform vec4 col;\nout vec4 o;\nvoid main(){ o = col; }\n";

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

static GLuint tex_px(int w, int h, const unsigned char *px)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static unsigned char img[W * W * 4];

static void grab(void)
{
    glFinish();
    glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, img);
}

static int cell_is(int cell, int r, int g, int b, const char *what)
{
    int cx = cell % 8, cy = cell / 8;
    const unsigned char *p = &img[4 * ((cy * CELL + CELL / 2) * W + cx * CELL + CELL / 2)];
    int ok = abs(p[0] - r) <= 2 && abs(p[1] - g) <= 2 && abs(p[2] - b) <= 2;
    if (!ok) printf("  [%s] cell %d = %d %d %d, want %d %d %d\n", what, cell, p[0], p[1], p[2], r, g, b);
    return ok ? 0 : 1;
}

static void place(int cell) { glUniform2f(0, (float)(cell % 8), (float)(cell / 8)); }
static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

static const unsigned char COL[4][4] = {
    {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 0, 255}
};

int main(void)
{
    int bad = 0;
    uint64_t tex0 = mgl_compat_fast_tex, bufs0 = mgl_compat_fast_bufs, mov0 = mgl_compat_fast_moved,
             rs0 = mgl_compat_fast_rs;

    GLuint fbo, ct, ds;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenTextures(1, &ds); glBindTexture(GL_TEXTURE_2D, ds); glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, ds, 0);
    glViewport(0, 0, W, W);

    float unit[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof unit, unit, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint tx[4];
    for (int i = 0; i < 4; i++) tx[i] = tex_px(1, 1, COL[i]);

    GLuint ubo; glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    float zero[4] = {0, 0, 0, 0};
    glBufferData(GL_UNIFORM_BUFFER, 16, zero, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, ubo);

    GLuint pt = mk(vs_cell, fs_tex), pu = mk(vs_cell, fs_ubo), ptu = mk(vs_cell, fs_tex_ubo);
    GLuint pv = mk(vs_vtex, fs_pass), pa = mk(vs_abs, fs_col);

    /* ---- 1. texture-binding-only switches (+ redundant VAO / UBO rebinds) */
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(pt); glUniform2f(1, 0.5f, 0.5f);
    glActiveTexture(GL_TEXTURE0);
    for (int c = 0; c < 32; c++) {
        glBindVertexArray(vao);                       /* same VAO: DIRTY_VAO, redundant */
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, ubo);  /* same binding: buffer-base dirt, redundant */
        glBindTexture(GL_TEXTURE_2D, tx[(c * 3 + c / 4) % 4]);
        place(c); quad();
    }
    grab();
    for (int c = 0; c < 32; c++) {
        const unsigned char *k = COL[(c * 3 + c / 4) % 4];
        bad += cell_is(c, k[0], k[1], k[2], "tex switch");
    }

    /* ---- 2. sampler object / texture parameter changes (2x1 red|green, u = 1.25) */
    {
        unsigned char rg[8] = {255, 0, 0, 255, 0, 255, 0, 255};
        GLuint t2 = tex_px(2, 1, rg);
        GLuint smp; glGenSamplers(1, &smp);
        glSamplerParameteri(smp, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(smp, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, t2);
        glUniform2f(1, 1.25f, 0.5f);                  /* REPEAT -> u 0.25 red, CLAMP -> green */
        int want[16];
        for (int c = 0; c < 16; c++) {
            int mode = c % 4;
            if (mode == 0) { glBindSampler(0, 0); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); want[c] = 0; }
            if (mode == 1) { glBindSampler(0, 0); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); want[c] = 1; }
            if (mode == 2) { glSamplerParameteri(smp, GL_TEXTURE_WRAP_S, GL_REPEAT); glBindSampler(0, smp); want[c] = 0; }
            if (mode == 3) { glSamplerParameteri(smp, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); want[c] = 1; }
            place(c); quad();
        }
        glBindSampler(0, 0);
        grab();
        for (int c = 0; c < 16; c++)
            bad += cell_is(c, want[c] ? 0 : 255, want[c] ? 255 : 0, 0, "sampler/param");
        glDeleteSamplers(1, &smp);
        glDeleteTextures(1, &t2);
    }

    /* ---- 3. vertex-stage texture switches */
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(pv);
    glActiveTexture(GL_TEXTURE1);
    for (int c = 0; c < 16; c++) {
        glBindTexture(GL_TEXTURE_2D, tx[(c + 1) % 4]);
        place(c); quad();
    }
    glActiveTexture(GL_TEXTURE0);
    grab();
    for (int c = 0; c < 16; c++) {
        const unsigned char *k = COL[(c + 1) % 4];
        bad += cell_is(c, k[0], k[1], k[2], "vs texture switch");
    }

    /* ---- 4. vertex-buffer content updates between draws */
    for (int big = 0; big < 2; big++) {
        GLuint vb; glGenBuffers(1, &vb);
        size_t sz = big ? 8192 : 64;                  /* >= 4 KB: MTLBuffer-backed; small: ring */
        float *init = calloc(1, sz);
        glBindBuffer(GL_ARRAY_BUFFER, vb); glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sz, init, GL_DYNAMIC_DRAW);
        free(init);
        GLuint va2; glGenVertexArrays(1, &va2); glBindVertexArray(va2);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
        glUseProgram(pa);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int c = 0; c < 24; c++) {
            float x0 = (c % 8) * 0.25f - 1.0f, y0 = (c / 8) * 0.25f - 1.0f;
            float q[8] = {x0, y0, x0 + 0.25f, y0, x0, y0 + 0.25f, x0 + 0.25f, y0 + 0.25f};
            if (c % 6 == 5)       /* orphaning same-size respecification */
            {
                float *full = calloc(1, sz); memcpy(full, q, sizeof q);
                glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sz, full, GL_DYNAMIC_DRAW);
                free(full);
            }
            else
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof q, q);
            const unsigned char *k = COL[c % 4];
            glUniform4f(2, k[0] / 255.0f, k[1] / 255.0f, k[2] / 255.0f, 1.0f);
            quad();
        }
        grab();
        for (int c = 0; c < 24; c++) {
            const unsigned char *k = COL[c % 4];
            bad += cell_is(c, k[0], k[1], k[2], big ? "vbo update (mtl)" : "vbo update (ring)");
        }
        glBindVertexArray(vao);
        glDeleteVertexArrays(1, &va2);
        glDeleteBuffers(1, &vb);
    }

    /* ---- 5. uniform-buffer updates while the previous draw references it */
    glBindVertexArray(vao);
    glUseProgram(pu);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    for (int c = 0; c < 32; c++) {
        const unsigned char *k = COL[(c + 2) % 4];
        float col[4] = {k[0] / 255.0f, k[1] / 255.0f, k[2] / 255.0f, 1.0f};
        if (c % 5 == 4) glBufferData(GL_UNIFORM_BUFFER, 16, col, GL_DYNAMIC_DRAW);   /* orphan */
        else glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, col);
        place(c); quad();
    }
    grab();
    for (int c = 0; c < 32; c++) {
        const unsigned char *k = COL[(c + 2) % 4];
        bad += cell_is(c, k[0], k[1], k[2], "ubo update");
    }

    /* ---- 5b. (r157) an 8 KB fragment-stage UBO re-specified with glBufferData
     *      before every draw: the re-specification releases its Metal storage,
     *      the buffers path must allocate again and bind the new contents */
    {
        GLuint big; glGenBuffers(1, &big); glBindBuffer(GL_UNIFORM_BUFFER, big);
        static float blk[2048];
        memset(blk, 0, sizeof blk);
        glBufferData(GL_UNIFORM_BUFFER, sizeof blk, blk, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, big);
        glClear(GL_COLOR_BUFFER_BIT);
        uint64_t mv0 = mgl_compat_fast_moved;
        for (int c = 0; c < 24; c++) {
            const unsigned char *k = COL[(c + 1) % 4];
            blk[0] = k[0] / 255.0f; blk[1] = k[1] / 255.0f; blk[2] = k[2] / 255.0f; blk[3] = 1.0f;
            glBufferData(GL_UNIFORM_BUFFER, sizeof blk, blk, GL_DYNAMIC_DRAW);
            place(c); quad();
        }
        grab();
        for (int c = 0; c < 24; c++) {
            const unsigned char *k = COL[(c + 1) % 4];
            bad += cell_is(c, k[0], k[1], k[2], "8 KB ubo re-specified");
        }
        const char *off = getenv("MGL_DRAWFAST_R154");
        if (!(off && off[0] == '0'))
            EXPECT_TRUE(mgl_compat_fast_moved - mv0 >= 16);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glDeleteBuffers(1, &big);
    }

    /* ---- 6. texture + UBO + (cell) uniform all moving per draw */
    glUseProgram(ptu);
    glClear(GL_COLOR_BUFFER_BIT);
    for (int c = 0; c < 32; c++) {
        glBindTexture(GL_TEXTURE_2D, tx[c % 4]);                 /* r/g/b/yellow */
        float add[4] = {0, 0, (c / 4) % 2 ? 1.0f : 0.0f, 0};     /* blue on/off */
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, add);
        place(c); quad();
    }
    grab();
    for (int c = 0; c < 32; c++) {
        const unsigned char *k = COL[c % 4];
        int b = (c / 4) % 2 ? 255 : k[2];
        bad += cell_is(c, k[0], k[1], b, "tex+ubo");
    }

    /* ---- 7. stencil reference (encoder state) + UBO update per draw */
    glUseProgram(pu);
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_STENCIL_TEST);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    for (int c = 0; c < 16; c++) {
        glStencilFunc(GL_ALWAYS, c + 1, 0xff);
        const unsigned char *k = COL[(c + 3) % 4];
        float col[4] = {k[0] / 255.0f, k[1] / 255.0f, k[2] / 255.0f, 1.0f};
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, col);
        place(c); quad();
    }
    glDisable(GL_STENCIL_TEST);
    grab();
    for (int c = 0; c < 16; c++) {
        const unsigned char *k = COL[(c + 3) % 4];
        bad += cell_is(c, k[0], k[1], k[2], "stencil ref + ubo");
    }
    {
        static unsigned char st[W * W];
        glReadPixels(0, 0, W, W, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, st);
        for (int c = 0; c < 16; c++) {
            int v = st[(c / 8 * CELL + 4) * W + (c % 8) * CELL + 4];
            if (v != c + 1) { printf("  [stencil] cell %d = %d, want %d\n", c, v, c + 1); bad++; }
        }
    }

    /* ---- 7b. blend constant (encoder state only) + UBO update per draw */
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_CONSTANT_COLOR, GL_ZERO);          /* rgb = col * 1 */
    for (int c = 0; c < 16; c++) {
        glBlendColor(1.0f, 1.0f, 1.0f, (c & 1) ? 0.5f : 1.0f);
        const unsigned char *k = COL[(c + 1) % 4];
        float col[4] = {k[0] / 255.0f, k[1] / 255.0f, k[2] / 255.0f, 1.0f};
        glBufferSubData(GL_UNIFORM_BUFFER, 0, 16, col);
        place(c); quad();
    }
    glDisable(GL_BLEND);
    grab();
    for (int c = 0; c < 16; c++) {
        const unsigned char *k = COL[(c + 1) % 4];
        bad += cell_is(c, k[0], k[1], k[2], "blend color + ubo");
    }

    /* ---- 8. texture content update without a rebind */
    glUseProgram(pt); glUniform2f(1, 0.5f, 0.5f);
    glClear(GL_COLOR_BUFFER_BIT);
    GLuint tu = tex_px(1, 1, COL[0]);
    glBindTexture(GL_TEXTURE_2D, tu);
    for (int c = 0; c < 8; c++) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, COL[c % 4]);
        place(c); quad();
    }
    grab();
    for (int c = 0; c < 8; c++) {
        const unsigned char *k = COL[c % 4];
        bad += cell_is(c, k[0], k[1], k[2], "texsubimage");
    }

    if (glGetError() != GL_NO_ERROR) { printf("  [gl error]\n"); bad++; }
    EXPECT_EQ_INT(bad, 0);

    /* the partial verdicts were exercised (unless r154 is switched off) */
    const char *off = getenv("MGL_DRAWFAST_R154");
    if (!(off && off[0] == '0')) {
        printf("  r154 partial: textures=%llu buffers=%llu moved=%llu rstate=%llu\n",
               (unsigned long long)(mgl_compat_fast_tex - tex0), (unsigned long long)(mgl_compat_fast_bufs - bufs0),
               (unsigned long long)(mgl_compat_fast_moved - mov0), (unsigned long long)(mgl_compat_fast_rs - rs0));
        EXPECT_TRUE(mgl_compat_fast_tex - tex0 >= 32);
        EXPECT_TRUE(mgl_compat_fast_moved - mov0 >= 8);
    }
    return mgl_test_summary();
}
