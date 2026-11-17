/* r165: the full state walk realizes only the texture units the bound program
 * samples.  A texture on a unit nobody samples is left alone — and must be
 * right the moment a program does sample it.  Sixteen units stay bound
 * throughout (D3D9 style); each case changes a texture while its unit is NOT
 * sampled, then samples it:
 *   1. glTexSubImage2D new contents
 *   2. glTexImage2D re-specification (new size)
 *   3. glTexParameteri wrap / filter change (sampled outside [0,1])
 *   4. a sampler uniform re-pointed at the unit with glUniform1i (same program)
 *   5. a sampler array whose element is re-pointed at the unit
 *   6. a geometry shader sampling the unit (emulated stage)
 *   7. a sampler object bound to the unit while unsampled
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define W 64
#define CELL 8

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\nout vec2 uv;\n"
    "void main(){ uv = p; vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
/* samples at a fixed coordinate given by a uniform (location 2) */
static const char *fs_one =
    "#version 450\nlayout(location=1) uniform sampler2D t;\nlayout(location=2) uniform vec2 at;\nin vec2 uv;\nout vec4 o;\n"
    "void main(){ o = vec4(texture(t, at).rgb, 1.0); }\n";
static const char *fs_arr =
    "#version 450\nlayout(location=1) uniform sampler2D ts[2];\nin vec2 uv;\nout vec4 o;\n"
    "void main(){ o = vec4(texture(ts[1], vec2(0.5)).rgb, 1.0); }\n";
static const char *vs_pass =
    "#version 450\nlayout(location=0) in vec2 p;\nout vec2 vp;\nvoid main(){ vp = p; gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *gs_tex =
    "#version 450\nlayout(triangles) in;\nlayout(triangle_strip, max_vertices=3) out;\n"
    "layout(binding=9) uniform sampler2D gt;\nlayout(location=0) uniform vec2 cell;\nin vec2 vp[];\nout vec3 gc;\n"
    "void main(){ vec3 c = textureLod(gt, vec2(0.5), 0.0).rgb;\n"
    "  for (int i = 0; i < 3; i++) { vec2 q = (cell + vp[i]) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); gc = c; EmitVertex(); }\n"
    "  EndPrimitive(); }\n";
static const char *fs_gs =
    "#version 450\nin vec3 gc;\nout vec4 o;\nvoid main(){ o = vec4(gc, 1.0); }\n";

static GLuint mk(const char *v, const char *g, const char *f)
{
    const char *s[3] = {v, g, f}; GLenum t[3] = {GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 3; i++) {
        if (!s[i]) continue;
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

static void fill(GLuint t, int w, int h, unsigned char r, unsigned char g, unsigned char b)
{
    unsigned char *px = malloc((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) { px[4*i] = r; px[4*i+1] = g; px[4*i+2] = b; px[4*i+3] = 255; }
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    free(px);
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
static void unit_bind(int u, GLuint t) { glActiveTexture(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, t); }

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

    GLuint tex[16];
    glGenTextures(16, tex);
    for (int u = 0; u < 16; u++) { fill(tex[u], 4, 4, (unsigned char)(10 * u), 0, 0); unit_bind(u, tex[u]); }

    GLuint p1 = mk(vs_cell, NULL, fs_one);
    glUseProgram(p1); glUniform2f(2, 0.5f, 0.5f);

    /* every case draws first with the program on unit 0, so units 3..15 are unsampled */
    #define WARM() do { glUseProgram(p1); glUniform1i(1, 0); glUniform2f(2, 0.5f, 0.5f); place(63); quad(); } while (0)

    /* 1. contents change while unsampled */
    WARM();
    { unsigned char px[4 * 4 * 4]; for (int i = 0; i < 16; i++) { px[4*i] = 0; px[4*i+1] = 255; px[4*i+2] = 0; px[4*i+3] = 255; }
      glActiveTexture(GL_TEXTURE3); glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px); }
    place(63); quad();                                   /* still unsampled: one more draw */
    glUniform1i(1, 3); place(0); quad();
    /* 2. re-specification (new size) while unsampled */
    WARM();
    fill(tex[4], 8, 8, 0, 0, 255); glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, tex[4]);
    place(63); quad();
    glUniform1i(1, 4); place(1); quad();
    /* 3. parameters while unsampled: a 2x1 texture, left red / right blue; REPEAT at x=1.25 reads
     *    the left texel, CLAMP_TO_EDGE reads the right one */
    WARM();
    { unsigned char px[8] = {255, 0, 0, 255, 0, 0, 255, 255};
      glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, tex[5]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); }
    glUniform1i(1, 5); glUniform2f(2, 1.25f, 0.5f); place(2); quad();          /* repeat: red */
    WARM();
    glActiveTexture(GL_TEXTURE5); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    place(63); quad();
    glUniform1i(1, 5); glUniform2f(2, 1.25f, 0.5f); place(3); quad();          /* clamp: blue */
    /* 4. sampler uniform re-pointed at a unit that was never sampled */
    WARM();
    fill(tex[6], 4, 4, 255, 255, 0); unit_bind(6, tex[6]);
    place(63); quad();
    glUniform1i(1, 6); glUniform2f(2, 0.5f, 0.5f); place(4); quad();
    /* 5. sampler array element re-pointed */
    {
        GLuint pa = mk(vs_cell, NULL, fs_arr);
        glUseProgram(pa); GLint su[2] = {0, 1}; glUniform1iv(1, 2, su); place(63); quad();
        fill(tex[7], 4, 4, 0, 255, 255); unit_bind(7, tex[7]);
        place(63); quad();
        su[1] = 7; glUniform1iv(1, 2, su); place(5); quad();
    }
    /* 6. geometry shader sampling unit 9 */
    {
        WARM();
        fill(tex[9], 4, 4, 255, 0, 255); unit_bind(9, tex[9]);
        place(63); quad();
        GLuint pg = mk(vs_pass, gs_tex, fs_gs);
        glUseProgram(pg); place(6); quad();
    }
    /* 7. a sampler object bound to an unsampled unit: REPEAT texture, CLAMP sampler */
    {
        WARM();
        unsigned char px[8] = {255, 0, 0, 255, 0, 0, 255, 255};
        glActiveTexture(GL_TEXTURE10); glBindTexture(GL_TEXTURE_2D, tex[10]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        GLuint so; glGenSamplers(1, &so);
        glSamplerParameteri(so, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(so, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(so, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindSampler(10, so);
        place(63); quad();
        glUniform1i(1, 10); glUniform2f(2, 1.25f, 0.5f); place(7); quad();    /* sampler clamps: blue */
    }

    grab();
    bad += cell_is(0, 0, 255, 0, "1 sub-image while unsampled");
    bad += cell_is(1, 0, 0, 255, "2 re-specified while unsampled");
    bad += cell_is(2, 255, 0, 0, "3 REPEAT");
    bad += cell_is(3, 0, 0, 255, "3 CLAMP set while unsampled");
    bad += cell_is(4, 255, 255, 0, "4 sampler uniform to a new unit");
    bad += cell_is(5, 0, 255, 255, "5 sampler array element");
    bad += cell_is(6, 255, 0, 255, "6 geometry shader unit");
    bad += cell_is(7, 0, 0, 255, "7 sampler object bound while unsampled");
    EXPECT_TRUE(bad == 0);
    printf("== r165 texture walk (sampled units): %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
