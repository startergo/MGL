/* test_xctx5.c — compat profile: real glmark2 jellyfish256.png RGBA texture
 * upload + readback + blended sampling (jellyfish rendered too bright/solid) */
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
    "uniform sampler2D uSampler;\n"
    "varying vec2 uv;\n"
    "void main() {\n"
    "    /* sample texel (96,96)'s centre directly: no viewport mapping needed */\n"
    "    vec4 colorMap = texture2D(uSampler, vec2(96.5 / 256.0, 96.5 / 256.0));\n"
    "    gl_FragColor = vec4(colorMap.rgb, colorMap.a + 0.5 - 0.3);\n"   /* scene math shape */
    "}\n";
static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

static void read_file_texel(GLubyte *out, const GLubyte *data, int w, int x, int y)
{
    memcpy(out, data + (y * w + x) * 4, 4);
}

TEST(compat_jellyfish_texture)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    FILE *f;
    long len;
    GLubyte *rgba;
    GLuint tex = 0, prog = 0, vao = 0, vbo = 0;
    GLint loc, apos;
    GLubyte back[256 * 256 * 4];

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    f = fopen("/tmp/xctx/jelly256.rgba", "rb");
    if (!f) { fprintf(stderr, "  note: /tmp/xctx/jelly256.rgba missing, skip\n"); destroyGLMContext(c); return; }
    fseek(f, 0, SEEK_END); len = ftell(f); fseek(f, 0, SEEK_SET);
    rgba = (GLubyte *)malloc(len);
    if (fread(rgba, 1, len, f) != (size_t)len) { fclose(f); free(rgba); return; }
    fclose(f);
    EXPECT_EQ_INT(len, 256 * 256 * 4);

    /* 1. upload + readback */
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back);
    while (glGetError() != GL_NO_ERROR) {}

    int bad = 0;
    for (int i = 0; i < 256 * 256 * 4; i++)
        if (rgba[i] != back[i]) bad++;
    if (bad) {
        GLubyte a[4], b[4];
        read_file_texel(a, rgba, 256, 128, 128);
        read_file_texel(b, back, 256, 128, 128);
        fprintf(stderr, "  note: readback differs in %d bytes; texel(128,128) got %d,%d,%d,%d want %d,%d,%d,%d\n",
                bad, b[0],b[1],b[2],b[3], a[0],a[1],a[2],a[3]);
    }
    EXPECT_EQ_INT(bad, 0);

    /* 2. blended sampling with the scene's alpha math */
    prog = mgl_make_program(k_vs, k_fs);
    EXPECT_TRUE(prog);
    apos = glGetAttribLocation(prog, "pos");
    EXPECT_TRUE(apos >= 0);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)apos);
    glVertexAttribPointer((GLuint)apos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    loc = glGetUniformLocation(prog, "uSampler"); if (loc >= 0) glUniform1i(loc, 0);

    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    {
        GLubyte px2[4];
        while (glGetError() != GL_NO_ERROR) {}
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px2);
        /* texel (96,96): rgb (215,134,65), a=209 -> transparency clamps to 1 */
        fprintf(stderr, "  note: blended center px = %d,%d,%d,%d\n", px2[0],px2[1],px2[2],px2[3]);
        EXPECT_NEAR(px2[0], 215, 6);
        EXPECT_NEAR(px2[1], 134, 6);
        EXPECT_NEAR(px2[2], 65, 6);
    }

    glDeleteProgram(prog); glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
    free(rgba);
    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_jellyfish_texture);
    return mgl_test_summary();
}
