/* r170: a glUniform* write that leaves the value unchanged no longer uploads.
 *   1. sampler uniform re-sent with the same unit before every draw, with an
 *      unsampled, never-realized texture left bound on another unit: the texture
 *      partial walk keeps re-binding only the moved units (mgl_compat_fast_texunits)
 *   2. sampler uniform changed: the new unit is sampled
 *   3. vec4 uniform re-sent with the same value: no new Metal buffers, draws on
 *      the fast path; changed value takes effect
 *   4. float array: an identical sub-range write, then one changed element
 *   5. glProgramUniform on a program that is not current, then used
 *   6. relink: values reset to defaults; a write of the default value (0) after
 *      link still draws right
 * Pixels checked for every draw. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_texunits, mgl_compat_fast_miss, mgl_bufpool_stat_alloc;

#define W 64
#define CELL 8

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs_tex =
    "#version 450\nuniform sampler2D t;\nout vec4 o;\nvoid main(){ o = vec4(texture(t, vec2(0.5)).rgb, 1.0); }\n";
static const char *fs_col =
    "#version 450\nuniform vec4 col;\nuniform float k[4];\nout vec4 o;\n"
    "void main(){ o = vec4(col.rgb + vec3(k[0], k[1], k[2]) * k[3], 1.0); }\n";

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
static GLuint tex_rgb(unsigned char r, unsigned char g, unsigned char b)
{
    unsigned char px[4] = {r, g, b, 255};
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
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

    GLuint red = tex_rgb(255, 0, 0), green = tex_rgb(0, 255, 0), blue = tex_rgb(0, 0, 255);

    /* 1 + 2: sampler uniform */
    GLuint pt = mk(fs_tex);
    GLint lt = glGetUniformLocation(pt, "t");
    glUseProgram(pt);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, red);
    glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, blue);
    glUniform1i(lt, 3);
    place(0); quad();
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, green);   /* realize green once */
    place(1); quad();
    uint64_t u0 = mgl_compat_fast_texunits;
    for (int i = 0; i < 8; i++) {
        glUniform1i(lt, 3);                                   /* same unit, every draw */
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, (i & 1) ? green : red);
        place(8 + i); quad();
    }
    uint64_t masked = mgl_compat_fast_texunits - u0;
    printf("  section 1: moved-units-only rebinds %llu of 8\n", (unsigned long long)masked);
    bad += masked < 7;
    glUniform1i(lt, 5);                                       /* changed: unit 5 (blue) */
    place(16); quad();

    /* 3 + 4: plain uniforms */
    GLuint pc = mk(fs_col);
    GLint lcol = glGetUniformLocation(pc, "col"), lk = glGetUniformLocation(pc, "k");
    glUseProgram(pc);
    float k[4] = {0, 0, 0, 0};
    glUniform1fv(lk, 4, k);
    glUniform4f(lcol, 0.0f, 0.0f, 1.0f, 1.0f);
    place(24); quad(); place(25); quad();
    uint64_t a0 = mgl_bufpool_stat_alloc, m0 = mgl_compat_fast_miss;
    for (int i = 0; i < 6; i++) {
        glUniform4f(lcol, 0.0f, 0.0f, 1.0f, 1.0f);           /* same value */
        place(26 + i); quad();
    }
    uint64_t allocs = mgl_bufpool_stat_alloc - a0, misses = mgl_compat_fast_miss - m0;
    printf("  section 3: new Metal buffers %llu, fast-path misses %llu over 6 redundant writes\n",
           (unsigned long long)allocs, (unsigned long long)misses);
    bad += allocs != 0 || misses > 1;
    glUniform4f(lcol, 1.0f, 0.0f, 0.0f, 1.0f);                /* changed */
    place(32); quad();
    glUniform1fv(lk, 4, k);                                   /* identical array write */
    place(33); quad();
    float k2[2] = {1.0f, 1.0f};                               /* k[2] = 1 (blue), k[3] = 1 */
    glUniform1fv(glGetUniformLocation(pc, "k[2]"), 2, k2);
    place(34); quad();

    /* 5. glProgramUniform on a non-current program */
    glUseProgram(pt);
    glProgramUniform4f(pc, lcol, 0.0f, 1.0f, 0.0f, 1.0f);
    glProgramUniform1fv(pc, lk, 4, k);
    glUseProgram(pc);
    place(40); quad();

    /* 6. relink: defaults back; writing the default value still draws right */
    glLinkProgram(pc);
    glUseProgram(pc);
    lcol = glGetUniformLocation(pc, "col"); lk = glGetUniformLocation(pc, "k");
    glUniform4f(lcol, 0.0f, 0.0f, 0.0f, 0.0f);                /* = default */
    glUniform1fv(lk, 4, k);
    place(48); quad();
    glUniform4f(lcol, 1.0f, 1.0f, 0.0f, 1.0f);
    place(49); quad();

    grab();
    bad += cell_is(0, 255, 0, 0, "1 unit 3 red");
    bad += cell_is(1, 0, 255, 0, "1 unit 3 green");
    for (int i = 0; i < 8; i++) bad += cell_is(8 + i, (i & 1) ? 0 : 255, (i & 1) ? 255 : 0, 0, "1 same sampler unit re-sent");
    bad += cell_is(16, 0, 0, 255, "2 sampler moved to unit 5");
    bad += cell_is(24, 0, 0, 255, "3 vec4");
    for (int i = 0; i < 6; i++) bad += cell_is(26 + i, 0, 0, 255, "3 vec4 re-sent");
    bad += cell_is(32, 255, 0, 0, "3 vec4 changed");
    bad += cell_is(33, 255, 0, 0, "4 identical array write");
    bad += cell_is(34, 255, 0, 255, "4 array element changed");
    bad += cell_is(40, 0, 255, 0, "5 glProgramUniform");
    bad += cell_is(48, 0, 0, 0, "6 default after relink");
    bad += cell_is(49, 255, 255, 0, "6 changed after relink");
    EXPECT_TRUE(bad == 0);
    printf("== r170 redundant uniform writes: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
