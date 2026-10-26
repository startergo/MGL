/* test_xctx10.c — compat profile: GL_RGB texture upload (glmark2 caustics are
 * 24-bit PNGs uploaded as GL_RGB; RGBA path is verified separately) */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static const char *k_vs =
    "attribute vec2 pos;\n"
    "varying vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "precision mediump float;\n"
    "uniform sampler2D tex;\n"
    "varying vec2 uv;\n"
    "void main() { gl_FragColor = vec4(texture2D(tex, vec2(96.5 / 256.0, 96.5 / 256.0)).rgb, 1.0); }\n";
static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

TEST(compat_rgb_upload)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    FILE *f = fopen("/tmp/xctx/caust01.rgb", "rb");
    GLubyte *rgb, back[256 * 256 * 4];
    GLuint tex, prog, vao, vbo;
    GLint loc, apos;

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    if (!f) { fprintf(stderr, "  note: caust01.rgb missing, skip\n"); return; }
    rgb = (GLubyte *)malloc(256 * 256 * 3);
    if (fread(rgb, 1, 256 * 256 * 3, f) != 256 * 256 * 3) { fclose(f); return; }
    fclose(f);

    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    while (glGetError() != GL_NO_ERROR) {}

    /* readback: RGB must arrive in RGB with alpha 255 */
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back);
    while (glGetError() != GL_NO_ERROR) {}
    {
        int bad = 0;
        for (int i = 0; i < 256 * 256; i++)
        {
            if (abs(back[i * 4] - rgb[i * 3]) > 1 ||
                abs(back[i * 4 + 1] - rgb[i * 3 + 1]) > 1 ||
                abs(back[i * 4 + 2] - rgb[i * 3 + 2]) > 1)
                bad++;
        }
        if (bad) fprintf(stderr, "  note: %d/%d texels wrong; (0,0) got %d,%d,%d want 186\n",
                         bad, 256 * 256, back[0], back[1], back[2]);
        EXPECT_EQ_INT(bad, 0);
    }

    /* sampled draw of a known texel through the exact scene math */
    prog = mgl_make_program(k_vs, k_fs);
    EXPECT_TRUE(prog);
    apos = glGetAttribLocation(prog, "pos");
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)apos);
    glVertexAttribPointer((GLuint)apos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    loc = glGetUniformLocation(prog, "tex"); if (loc >= 0) glUniform1i(loc, 0);
    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    {
        GLubyte px[4];
        while (glGetError() != GL_NO_ERROR) {}
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        {
            int o = (96 * 256 + 96) * 3;
            fprintf(stderr, "  note: sampled px = %d,%d,%d,%d; texel(96,96) = %d,%d,%d\n",
                    px[0], px[1], px[2], px[3], rgb[o], rgb[o+1], rgb[o+2]);
            EXPECT_NEAR(px[0], rgb[o], 3);
            EXPECT_NEAR(px[1], rgb[o+1], 3);
            EXPECT_NEAR(px[2], rgb[o+2], 3);
        }
    }

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_rgb_upload);
    return mgl_test_summary();
}
