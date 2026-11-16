/* r158: the texture-only partial walk re-binds only the units that moved — and
 * still everything a full walk would change.  One render pass per section, many
 * small quads into 8x8 cells of a 64x64 target, one readback per section:
 *   1. a sampler uniform re-pointed at another (unchanged) unit between draws
 *      (glUniform1i on the sampler; no texture binding changes at all)
 *   2. eight units bound, one unit's texture switched per draw (the per-unit
 *      rebind path; the other seven must keep their bindings)
 *   3. a sampler ARRAY whose element units are re-pointed between draws
 *   4. the vertex and fragment stages sampling different units, one switched
 *   5. a unit switched to an INCOMPLETE texture (fallback black) and back
 *   6. a unit switched to a texture whose sampler object changes on the same draw
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern uint64_t mgl_compat_fast_tex;

#define W 64
#define CELL 8

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs_one =
    "#version 450\nlayout(location=1) uniform sampler2D t;\nout vec4 o;\nvoid main(){ o = texture(t, vec2(0.5)); }\n";
/* eight samplers summed with weights: each unit contributes its own channel bit */
static const char *fs_eight =
    "#version 450\nlayout(binding=0) uniform sampler2D t0;\nlayout(binding=1) uniform sampler2D t1;\n"
    "layout(binding=2) uniform sampler2D t2;\nlayout(binding=3) uniform sampler2D t3;\n"
    "layout(binding=4) uniform sampler2D t4;\nlayout(binding=5) uniform sampler2D t5;\n"
    "layout(binding=6) uniform sampler2D t6;\nlayout(binding=7) uniform sampler2D t7;\n"
    "out vec4 o;\nvoid main(){ vec2 c = vec2(0.5);\n"
    "  float r = texture(t0,c).r*0.5 + texture(t1,c).r*0.25 + texture(t2,c).r*0.125;\n"
    "  float g = texture(t3,c).r*0.5 + texture(t4,c).r*0.25 + texture(t5,c).r*0.125;\n"
    "  float b = texture(t6,c).r*0.5 + texture(t7,c).r*0.25;\n"
    "  o = vec4(r, g, b, 1.0); }\n";
static const char *fs_arr =
    "#version 450\nlayout(location=1) uniform sampler2D ts[2];\nout vec4 o;\n"
    "void main(){ o = vec4(texture(ts[0], vec2(0.5)).r, texture(ts[1], vec2(0.5)).r, 0.0, 1.0); }\n";
static const char *vs_vtex =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "layout(binding=2) uniform sampler2D vt;\nout float vr;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0);\n"
    "             vr = textureLod(vt, vec2(0.5), 0.0).r; }\n";
static const char *fs_vtex =
    "#version 450\nlayout(binding=3) uniform sampler2D ft;\nin float vr;\nout vec4 o;\n"
    "void main(){ o = vec4(vr, texture(ft, vec2(0.5)).r, 0.0, 1.0); }\n";

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

/* 1x1 texture, red channel = v */
static GLuint tex_r(unsigned char v)
{
    unsigned char px[4] = {v, 0, 0, 255};
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
    uint64_t tex0 = mgl_compat_fast_tex;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    float unitq[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof unitq, unitq, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint on = tex_r(255), off = tex_r(0), mid = tex_r(128);

    /* ---- 1. sampler uniform re-pointed between draws */
    {
        GLuint p = mk(vs_cell, fs_one);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, off);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, on);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, mid);
        glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
        for (int c = 0; c < 24; c++) { glUniform1i(1, c % 3); place(c); quad(); }
        grab();
        static const int want[3] = {0, 255, 128};
        for (int c = 0; c < 24; c++) bad += cell_is(c, want[c % 3], 0, 0, "sampler uniform");
    }

    /* ---- 2. eight units, one switched per draw */
    {
        GLuint p = mk(vs_cell, fs_eight);
        glUseProgram(p);
        int state[8] = {0};
        for (int u = 0; u < 8; u++) { glActiveTexture(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, off); }
        glClear(GL_COLOR_BUFFER_BIT);
        int wr[48], wg[48], wb[48];
        for (int c = 0; c < 48; c++) {
            int u = (c * 5) % 8;
            state[u] ^= 1;
            glActiveTexture(GL_TEXTURE0 + u); glBindTexture(GL_TEXTURE_2D, state[u] ? on : off);
            wr[c] = (int)(255 * (state[0] * 0.5 + state[1] * 0.25 + state[2] * 0.125) + 0.5);
            wg[c] = (int)(255 * (state[3] * 0.5 + state[4] * 0.25 + state[5] * 0.125) + 0.5);
            wb[c] = (int)(255 * (state[6] * 0.5 + state[7] * 0.25) + 0.5);
            place(c); quad();
        }
        grab();
        for (int c = 0; c < 48; c++) bad += cell_is(c, wr[c], wg[c], wb[c], "eight units");
    }

    /* ---- 3. sampler array element units re-pointed */
    {
        GLuint p = mk(vs_cell, fs_arr);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, on);
        glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, off);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int c = 0; c < 16; c++) {
            GLint u[2] = { (c & 1) ? 5 : 4, (c & 2) ? 4 : 5 };
            glUniform1iv(1, 2, u);
            place(c); quad();
        }
        grab();
        for (int c = 0; c < 16; c++)
            bad += cell_is(c, (c & 1) ? 0 : 255, (c & 2) ? 255 : 0, 0, "sampler array");
    }

    /* ---- 4. vertex- and fragment-stage units, one switched per draw */
    {
        GLuint p = mk(vs_vtex, fs_vtex);
        glUseProgram(p);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, off);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, off);
        glClear(GL_COLOR_BUFFER_BIT);
        int sv = 0, sf = 0, wv[16], wf[16];
        for (int c = 0; c < 16; c++) {
            if (c & 1) { sv ^= 1; glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, sv ? on : off); }
            else       { sf ^= 1; glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, sf ? on : off); }
            wv[c] = sv ? 255 : 0; wf[c] = sf ? 255 : 0;
            place(c); quad();
        }
        grab();
        for (int c = 0; c < 16; c++) bad += cell_is(c, wv[c], wf[c], 0, "vs/fs units");
    }

    /* ---- 5. switch to an incomplete texture (samples black) and back */
    {
        GLuint inc; glGenTextures(1, &inc); glBindTexture(GL_TEXTURE_2D, inc);
        unsigned char px[16] = {255, 255, 255, 255};
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
        /* default MIN_FILTER needs mipmaps: incomplete */
        GLuint p = mk(vs_cell, fs_one);
        glUseProgram(p); glUniform1i(1, 6);
        glActiveTexture(GL_TEXTURE6);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int c = 0; c < 12; c++) { glBindTexture(GL_TEXTURE_2D, (c & 1) ? inc : on); place(c); quad(); }
        grab();
        for (int c = 0; c < 12; c++) bad += cell_is(c, (c & 1) ? 0 : 255, 0, 0, "incomplete");
        glDeleteTextures(1, &inc);
    }

    /* ---- 6. texture and sampler object changed on the same draw (wrap decides) */
    {
        unsigned char rg[8] = {255, 0, 0, 255, 0, 0, 0, 255};     /* red | black */
        GLuint t2; glGenTextures(1, &t2); glBindTexture(GL_TEXTURE_2D, t2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, rg);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        static const char *fs_uv =
            "#version 450\nlayout(location=1) uniform sampler2D t;\nout vec4 o;\nvoid main(){ o = texture(t, vec2(1.25, 0.5)); }\n";
        GLuint p = mk(vs_cell, fs_uv);
        glUseProgram(p); glUniform1i(1, 7);
        GLuint srep, sclamp; glGenSamplers(1, &srep); glGenSamplers(1, &sclamp);
        GLuint ss[2] = {srep, sclamp};
        for (int i = 0; i < 2; i++) {
            glSamplerParameteri(ss[i], GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glSamplerParameteri(ss[i], GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        glSamplerParameteri(srep, GL_TEXTURE_WRAP_S, GL_REPEAT);            /* u 0.25: red */
        glSamplerParameteri(sclamp, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);   /* edge: black */
        glActiveTexture(GL_TEXTURE7);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int c = 0; c < 12; c++) {
            glBindTexture(GL_TEXTURE_2D, (c % 3 == 2) ? on : t2);
            glBindSampler(7, (c & 1) ? sclamp : srep);
            place(c); quad();
        }
        glBindSampler(7, 0);
        grab();
        for (int c = 0; c < 12; c++) {
            int r = (c % 3 == 2) ? 255 : ((c & 1) ? 0 : 255);
            bad += cell_is(c, r, 0, 0, "texture + sampler");
        }
    }

    glActiveTexture(GL_TEXTURE0);
    if (glGetError() != GL_NO_ERROR) { printf("  [gl error]\n"); bad++; }
    EXPECT_EQ_INT(bad, 0);
    const char *offenv = getenv("MGL_DRAWFAST_R154");
    if (!(offenv && offenv[0] == '0')) {
        printf("  texture partial walks: %llu\n", (unsigned long long)(mgl_compat_fast_tex - tex0));
        EXPECT_TRUE(mgl_compat_fast_tex - tex0 >= 60);
    }
    return mgl_test_summary();
}
