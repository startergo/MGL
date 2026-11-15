/* r152: redundant GL calls are no-ops (no dirty state, no new render pass), and the
 * real changes that follow them still take effect.  Covers glBindTexture,
 * glEnable/glDepthFunc/glBlendFunc/glColorMask/glViewport, glBindFramebuffer,
 * glDrawBuffers and glFramebufferTexture2D — including re-attaching a texture
 * after it was re-specified at a different size (must render into the NEW image)
 * and blending across a redundant re-attach inside one render pass. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static const char *vs =
    "#version 450\nlayout(location=0) in vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
static const char *fs_col =
    "#version 450\nlayout(location=0) uniform vec4 col;\nout vec4 o;\nvoid main(){ o = col; }\n";
static const char *fs_tex =
    "#version 450\nlayout(binding=0) uniform sampler2D t;\nout vec4 o;\nvoid main(){ o = texture(t, vec2(0.5)); }\n";

static GLuint mk(const char *f)
{
    const char *s[2] = {vs, f}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) { GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh); glAttachShader(p, sh); glDeleteShader(sh); }
    glLinkProgram(p);
    return p;
}

static GLuint tex_rgba(int w, int h, const unsigned char *px)
{
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

static int px_is(int x, int y, int r, int g, int b, const char *what)
{
    unsigned char p[4] = {0};
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    int ok = abs(p[0] - r) <= 2 && abs(p[1] - g) <= 2 && abs(p[2] - b) <= 2;
    if (!ok) printf("  [%s] (%d,%d) = %d %d %d, want %d %d %d\n", what, x, y, p[0], p[1], p[2], r, g, b);
    return ok ? 0 : 1;
}

int main(void)
{
    int bad = 0;
    GLuint pc = mk(fs_col), pt = mk(fs_tex);
    float v[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    GLuint ct = tex_rgba(32, 32, NULL), dt, fbo;
    glGenTextures(1, &dt); glBindTexture(GL_TEXTURE_2D, dt); glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, 32, 32);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, dt, 0);
    glViewport(0, 0, 32, 32);

    /* 1. redundant texture binds, then a real switch */
    unsigned char red[4] = {255, 0, 0, 255}, green[4] = {0, 255, 0, 255};
    GLuint tr = tex_rgba(1, 1, red), tg = tex_rgba(1, 1, green);
    glUseProgram(pt);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tr); quad();
    glBindTexture(GL_TEXTURE_2D, tr); glBindTexture(GL_TEXTURE_2D, tr); quad();
    bad += px_is(16, 16, 255, 0, 0, "tex same");
    glBindTexture(GL_TEXTURE_2D, tg); quad();
    bad += px_is(16, 16, 0, 255, 0, "tex switch after redundant");
    /* updating the bound texture's contents without a rebind must show */
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, red); quad();
    bad += px_is(16, 16, 255, 0, 0, "texsubimage without rebind");

    /* 2. redundant state, then a real change */
    glUseProgram(pc);
    glDisable(GL_BLEND); glDisable(GL_BLEND); glColorMask(1, 1, 1, 1); glColorMask(1, 1, 1, 1);
    glUniform4f(0, 0.0f, 0.0f, 1.0f, 1.0f); quad();
    bad += px_is(16, 16, 0, 0, 255, "state same");
    glColorMask(1, 0, 1, 1);
    glUniform4f(0, 1.0f, 1.0f, 0.0f, 1.0f); quad();          /* green masked: r=255 g=0(kept) b=0 */
    bad += px_is(16, 16, 255, 0, 0, "colormask change");
    glColorMask(1, 1, 1, 1);
    glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glClear(GL_DEPTH_BUFFER_BIT);
    glUniform4f(0, 0.0f, 1.0f, 0.0f, 1.0f); quad();          /* passes (0.5 < 1) */
    glDepthFunc(GL_LESS); glDepthFunc(GL_LESS);
    glUniform4f(0, 0.0f, 0.0f, 1.0f, 1.0f); quad();          /* fails (0.5 < 0.5 false) */
    bad += px_is(16, 16, 0, 255, 0, "depthfunc same");
    glDepthFunc(GL_LEQUAL);
    quad();                                                  /* passes now */
    bad += px_is(16, 16, 0, 0, 255, "depthfunc change");
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, 32, 32); glViewport(0, 0, 32, 32);
    glViewport(0, 0, 16, 16); glUniform4f(0, 1.0f, 1.0f, 1.0f, 1.0f); quad();
    bad += px_is(4, 4, 255, 255, 255, "viewport change (inside)");
    bad += px_is(24, 24, 0, 0, 255, "viewport change (outside)");
    glViewport(0, 0, 32, 32);

    /* 2b. indexed setters (glEnablei / glViewportIndexedf / glScissorIndexed /
     *     glDepthRangeIndexed): redundant, then a real change */
    glEnablei(GL_SCISSOR_TEST, 0); glScissorIndexed(0, 0, 0, 32, 32); glScissorIndexed(0, 0, 0, 32, 32);
    glViewportIndexedf(0, 0, 0, 32, 32); glViewportIndexedf(0, 0, 0, 32, 32);
    glDepthRangeIndexed(0, 0.0, 1.0); glDepthRangeIndexed(0, 0.0, 1.0);
    glUniform4f(0, 0.0f, 0.0f, 0.0f, 1.0f); quad();
    bad += px_is(24, 24, 0, 0, 0, "indexed same");
    glScissorIndexed(0, 0, 0, 16, 16);
    glUniform4f(0, 1.0f, 1.0f, 0.0f, 1.0f); quad();
    bad += px_is(4, 4, 255, 255, 0, "scissor indexed change (inside)");
    bad += px_is(24, 24, 0, 0, 0, "scissor indexed change (outside)");
    glDisablei(GL_SCISSOR_TEST, 0); glDisablei(GL_SCISSOR_TEST, 0);
    glEnablei(GL_BLEND, 0); glEnablei(GL_BLEND, 0); glBlendFunci(0, GL_ONE, GL_ONE);
    glUniform4f(0, 0.0f, 0.0f, 1.0f, 0.0f); quad();          /* adds blue */
    bad += px_is(24, 24, 0, 0, 255, "enablei blend");
    glDisablei(GL_BLEND, 0); glDisablei(GL_BLEND, 0);
    glUniform4f(0, 0.0f, 1.0f, 0.0f, 1.0f); quad();          /* replaces */
    bad += px_is(24, 24, 0, 255, 0, "disablei blend");
    glBlendFunci(0, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /* 3. redundant FBO bind / draw buffers / re-attach inside one pass, with blending
     *    accumulating across them */
    glEnable(GL_BLEND); glBlendFunc(GL_ONE, GL_ONE);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    GLenum db = GL_COLOR_ATTACHMENT0;
    for (int i = 0; i < 4; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glDrawBuffers(1, &db);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, dt, 0);
        glUniform4f(0, 0.25f, 0.0f, 0.0f, 0.0f); quad();
    }
    bad += px_is(16, 16, 255, 0, 0, "blend across redundant fb calls");
    glDisable(GL_BLEND);

    /* 4. draw buffers really change */
    GLuint ct2 = tex_rgba(32, 32, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, ct2, 0);
    GLenum db2[1] = {GL_COLOR_ATTACHMENT1};   /* fragment output 0 -> attachment 1 */
    glDrawBuffers(1, db2);
    glUniform4f(0, 0.0f, 1.0f, 1.0f, 1.0f); quad();
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    bad += px_is(16, 16, 0, 255, 255, "drawbuffers change (att1)");
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    bad += px_is(16, 16, 255, 0, 0, "drawbuffers change (att0 untouched)");
    glDrawBuffers(1, &db);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);

    /* 5. re-specify the attached texture at another size, then re-attach it */
    glBindTexture(GL_TEXTURE_2D, ct);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, dt);
    GLuint dt8; glGenTextures(1, &dt8); glBindTexture(GL_TEXTURE_2D, dt8); glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, 8, 8);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, dt8, 0);
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) { printf("  [respecify] status 0x%x\n", st); bad++; }
    glViewport(0, 0, 8, 8);
    glUniform4f(0, 1.0f, 0.0f, 1.0f, 1.0f); quad();
    bad += px_is(4, 4, 255, 0, 255, "respecified attachment renders");
    {
        unsigned char img[8 * 8 * 4]; memset(img, 0, sizeof img);
        glBindTexture(GL_TEXTURE_2D, ct);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        if (img[4 * (4 * 8 + 4) + 0] != 255 || img[4 * (4 * 8 + 4) + 2] != 255) {
            printf("  [respecify] texture readback %d %d %d\n", img[4 * 36], img[4 * 36 + 1], img[4 * 36 + 2]); bad++; }
    }

    /* 6. default framebuffer round trip after redundant binds */
    glBindFramebuffer(GL_FRAMEBUFFER, 0); glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glUniform4f(0, 0.0f, 1.0f, 0.0f, 1.0f); quad();
    bad += px_is(4, 4, 0, 255, 0, "fbo round trip");

    if (glGetError() != GL_NO_ERROR) { printf("  [gl error]\n"); bad++; }
    EXPECT_EQ_INT(bad, 0);
    return mgl_test_summary();
}
