/* test_xctx3.c — compat-profile GLSL 120 blending (glmark2 jellyfish washed out:
 * the scene writes vec4(rgb, alpha) with GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA) */
#include "mgl_test.h"

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static const char *k_vs120 =
    "attribute vec2 pos;\n"
    "varying vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs120 =
    "precision mediump float;\n"
    "uniform sampler2D tex;\n"
    "uniform float alpha;\n"
    "varying vec2 uv;\n"
    "void main() { vec4 c = texture2D(tex, uv); gl_FragColor = vec4(c.rgb, c.a * alpha); }\n";

static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

static void draw_and_read(GLubyte out[4], float alpha)
{
    GLubyte img[4*4*4];
    GLuint tex = 0, prog = 0, vao = 0, vbo = 0;
    GLint loc;

    for (int i = 0; i < 16; i++) { img[i*4]=0; img[i*4+1]=0; img[i*4+2]=255; img[i*4+3]=255; }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    prog = mgl_make_program(k_vs120, k_fs120);
    EXPECT_TRUE(prog);
    GLint apos = glGetAttribLocation(prog, "pos");
    EXPECT_TRUE(apos >= 0);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)apos);
    glVertexAttribPointer((GLuint)apos, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    loc = glGetUniformLocation(prog, "tex");
    if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(prog, "alpha");
    if (loc >= 0) glUniform1f(loc, alpha);

    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(1, 1, 1, 1);         /* white background */
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    mgl_read_rgba(1, 1, out);
    while (glGetError() != GL_NO_ERROR) {}

    glDeleteProgram(prog); glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
}

TEST(compat_profile_blend_120)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLubyte px[4];
    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    /* half-transparent blue over white must land at mid blue */
    draw_and_read(px, 0.5f);
    EXPECT_NEAR(px[0], 128, 10);
    EXPECT_NEAR(px[1], 128, 10);
    EXPECT_NEAR(px[2], 255, 6);

    /* nearly transparent: stays white */
    draw_and_read(px, 0.1f);
    EXPECT_NEAR(px[0], 230, 12);
    EXPECT_NEAR(px[2], 255, 6);

    /* opaque blue */
    draw_and_read(px, 1.0f);
    EXPECT_NEAR(px[0], 0, 8);
    EXPECT_NEAR(px[2], 255, 6);

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_profile_blend_120);
    return mgl_test_summary();
}
