/* r169: a bound, realized texture with no level / data change skips the realize
 * walk at bind time, and the draw-fast snapshot tracks the indexed buffer tables
 * and image units by generation.  Every change between draws must still show:
 *   1. glTexSubImage2D into the bound texture
 *   2. glTexImage2D re-specifying level 0 at a new size
 *   3. glGenerateMipmap, sampled at an explicit LOD
 *   4. glCopyTexSubImage2D from the framebuffer
 *   5. GL_TEXTURE_BASE_LEVEL moved to a level with other contents
 *   6. glBindImageTexture re-pointing an image unit between draws (imageLoad)
 * Each case: draw, change, draw again; both cells checked. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define W 64
#define CELL 8

static const char *vs_cell =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs_lod =
    "#version 450\nlayout(binding=0) uniform sampler2D t;\nlayout(location=1) uniform float lod;\nout vec4 o;\n"
    "void main(){ o = vec4(textureLod(t, vec2(0.5), lod).rgb, 1.0); }\n";
static const char *fs_img =
    "#version 450\nlayout(binding=0, rgba8) readonly uniform image2D im;\nout vec4 o;\n"
    "void main(){ o = vec4(imageLoad(im, ivec2(0, 0)).rgb, 1.0); }\n";

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
    return p;
}
static void solid(GLenum target, GLint level, int w, int h, unsigned char r, unsigned char g, unsigned char b, int spec)
{
    unsigned char *px = malloc((size_t)w * h * 4);
    for (int i = 0; i < w * h; i++) { px[4*i] = r; px[4*i+1] = g; px[4*i+2] = b; px[4*i+3] = 255; }
    if (spec) glTexImage2D(target, level, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    else glTexSubImage2D(target, level, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
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

    GLuint pl = mk(vs_cell, fs_lod);
    glUseProgram(pl); glUniform1f(1, 0.0f);
    GLuint t; glGenTextures(1, &t);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    solid(GL_TEXTURE_2D, 0, 8, 8, 255, 0, 0, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    /* 1. sub-image */
    place(0); quad(); place(0); quad();                            /* realized, then the fast path */
    solid(GL_TEXTURE_2D, 0, 8, 8, 0, 255, 0, 0);
    place(1); quad();
    /* 2. re-specification at a new size */
    solid(GL_TEXTURE_2D, 0, 16, 16, 0, 0, 255, 1);
    place(2); quad();
    /* 3. mipmaps: level 0 white, generated chain, sample LOD 2 (white too); then level 2 replaced */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
    solid(GL_TEXTURE_2D, 0, 16, 16, 255, 255, 255, 0);
    glGenerateMipmap(GL_TEXTURE_2D);
    glUniform1f(1, 2.0f); place(3); quad();
    solid(GL_TEXTURE_2D, 2, 4, 4, 255, 0, 255, 0);
    place(4); quad();
    /* 4. copy from the framebuffer: cell 1 (green) into level 0 */
    glUniform1f(1, 0.0f);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 4, 4, 1 * CELL, 0, CELL, CELL);   /* texels 4..11 cover the sample at 0.5 */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    place(5); quad();
    /* 5. base level 2 (magenta, from case 3) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 2);
    place(6); quad();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);

    /* 6. image unit re-pointed between draws */
    {
        GLuint pi = mk(vs_cell, fs_img);
        GLuint ia, ib; glGenTextures(1, &ia); glGenTextures(1, &ib);
        glBindTexture(GL_TEXTURE_2D, ia); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
        solid(GL_TEXTURE_2D, 0, 1, 1, 255, 128, 0, 0);
        glBindTexture(GL_TEXTURE_2D, ib); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
        solid(GL_TEXTURE_2D, 0, 1, 1, 0, 128, 255, 0);
        glUseProgram(pi);
        glBindImageTexture(0, ia, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        place(8); quad(); place(8); quad();
        glBindImageTexture(0, ib, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        place(9); quad();
        glBindImageTexture(0, ia, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
        place(10); quad();
    }

    grab();
    bad += cell_is(0, 255, 0, 0, "1 before sub-image");
    bad += cell_is(1, 0, 255, 0, "1 after sub-image");
    bad += cell_is(2, 0, 0, 255, "2 re-specified");
    bad += cell_is(3, 255, 255, 255, "3 generated mip LOD 2");
    bad += cell_is(4, 255, 0, 255, "3 level 2 replaced");
    bad += cell_is(5, 0, 255, 0, "4 copied from the framebuffer");
    bad += cell_is(6, 255, 0, 255, "5 base level 2");
    bad += cell_is(8, 255, 128, 0, "6 image unit A");
    bad += cell_is(9, 0, 128, 255, "6 image unit B");
    bad += cell_is(10, 255, 128, 0, "6 image unit A again");
    EXPECT_TRUE(bad == 0);
    printf("== r169 fast realize + table generations: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
