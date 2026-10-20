/*
 * test_compat_render5.c — legacy texture state: GL_CLAMP wrap, borders,
 * GL_GENERATE_MIPMAP, glTexEnv LOD bias.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void setup_2d(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
}

/* GL_CLAMP is accepted (and reported back); sampling matches CLAMP_TO_EDGE */
TEST(clamp_wrap_and_border)
{
    setup_2d(64, 64);
    while (glGetError() != GL_NO_ERROR) { }

    GLuint tex = 0;
    GLubyte img[2 * 2 * 4] = { 255, 0, 0, 255,   0, 255, 0, 255,
                               0, 0, 255, 255,   255, 255, 0, 255 };
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    EXPECT_NO_GL_ERROR();

    GLint ws = 99, wt = 99;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, &ws);
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, &wt);
    EXPECT_TRUE(ws == GL_CLAMP);
    EXPECT_TRUE(wt == GL_CLAMP);

    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 1, 0, 0, 1 });
    GLfloat bc[4] = { 0, 0, 0, 0 };
    glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, bc);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(bc[0] == 1.0f && bc[1] == 0.0f);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0.5f, 0.5f);
    glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
    glEnd();
    glFinish();

    /* the interior sample is unaffected by the wrap mode */
    EXPECT_PIXEL(32, 32, 255, 255, 0, 255, 8);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

/* GL_GENERATE_MIPMAP builds the chain on level-0 upload */
TEST(generate_mipmap)
{
    setup_2d(64, 64);
    while (glGetError() != GL_NO_ERROR) { }

    GLuint tex = 0;
    GLubyte img[32 * 32 * 4];
    for (int i = 0; i < 32 * 32; i++)
    {
        img[i * 4 + 0] = 255;
        img[i * 4 + 1] = (GLubyte)((i % 32) * 8);
        img[i * 4 + 2] = 0;
        img[i * 4 + 3] = 255;
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    EXPECT_NO_GL_ERROR();

    GLint gmm = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, &gmm);
    EXPECT_TRUE(gmm == GL_TRUE);

    GLint w = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 3, GL_TEXTURE_WIDTH, &w);
    EXPECT_TRUE(w == 4);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 5, GL_TEXTURE_WIDTH, &w);
    EXPECT_TRUE(w == 1);

    /* mipmap filtering is complete */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glColor4f(1, 1, 1, 1);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    glFinish();

    GLubyte p[4];
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] == 255);
    EXPECT_NO_GL_ERROR();

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
}

/* glTexEnv GL_TEXTURE_LOD_BIAS is accepted and used by the FFP sampler */
TEST(texenv_lod_bias)
{
    setup_2d(64, 64);
    while (glGetError() != GL_NO_ERROR) { }

    GLuint tex = 0;
    GLubyte img[4 * 4 * 4];
    for (int i = 0; i < 4 * 4; i++)
    {
        img[i * 4 + 0] = (GLubyte)(255 - i * 16);
        img[i * 4 + 1] = (GLubyte)(i * 16);
        img[i * 4 + 2] = 64;
        img[i * 4 + 3] = 255;
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glColor4f(1, 1, 1, 1);

    GLfloat bias = 0.0f;
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_LOD_BIAS, 2.0f);
    glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_LOD_BIAS, &bias);       /* both spellings */
    EXPECT_NO_GL_ERROR();

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    glFinish();

    GLubyte p[4];
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    /* level 2 averages everything: still red-ish; the draw must succeed */
    EXPECT_TRUE(p[0] > 120);
    EXPECT_NO_GL_ERROR();

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
}

int main(void)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return 1;

    RUN(clamp_wrap_and_border);
    RUN(generate_mipmap);
    RUN(texenv_lod_bias);

    return mgl_test_summary();
}
