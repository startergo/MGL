/*
 * test_compat_render4.c — point sprites, point size attenuation, legacy
 * texture formats (LUMINANCE/ALPHA/INTENSITY incl. readback), texture
 * residency queries.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

#ifndef GL_INTENSITY
#define GL_INTENSITY 0x8049
#endif

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
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_LINE_STIPPLE);
    glPointSize(1.0f);
}

/* 2x2 texture, one unique colour per texel; point sprite maps gl_PointCoord
 * (0,0) to the top-left texel of the point */
TEST(point_sprite_orientation)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint tex = 0;
    GLubyte img[2 * 2 * 4] = {
        255, 0, 0, 255,     /* texel (0,0) red   */
          0, 255, 0, 255,   /* texel (1,0) green */
          0, 0, 255, 255,   /* texel (0,1) blue  */
        255, 255, 0, 255,   /* texel (1,1) yellow */
    };
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glPointSize(32.0f);

    glBegin(GL_POINTS);
    glVertex2f(32, 32);
    glEnd();

    glDisable(GL_POINT_SPRITE);
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    glFinish();

    /* gl_PointCoord (0,0) is the upper-left corner of the point */
    EXPECT_PIXEL(24, 40, 255, 0, 0, 255, 8);
    EXPECT_PIXEL(40, 40, 0, 255, 0, 255, 8);
    EXPECT_PIXEL(24, 24, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(40, 24, 255, 255, 0, 255, 8);
}

/* distance attenuation: size' = clamp(size * sqrt(1/(a + b*d + c*d^2)) */
TEST(point_size_attenuation)
{
    while (glGetError() != GL_NO_ERROR) { }   /* start clean */

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1, 1, -1, 1, 1, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPointSize(16.0f);
    glPointParameterf(GL_POINT_SIZE_MIN, 1.0f);
    glPointParameterf(GL_POINT_SIZE_MAX, 64.0f);
    glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, (GLfloat[3]){ 0.0f, 1.0f, 0.0f });
    glColor4f(1, 1, 1, 1);

    /* 16 * sqrt(1/d) for d = 4, 9, 16 -> 8, ~5.3, 4 pixels */
    float zs[3] = { 4.0f, 9.0f, 16.0f };
    int want[3] = { 8, 5, 4 };

    for (int i = 0; i < 3; i++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_POINTS);
        glVertex3f(0, 0, -zs[i]);
        glEnd();
        glFinish();

        int x0 = 64, x1 = -1, y0 = 64, y1 = -1;
        for (int y = 20; y < 44; y++)
            for (int x = 20; x < 44; x++)
            {
                GLubyte p[4];
                glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
                if (p[0] > 100)
                {
                    if (x < x0) x0 = x;
                    if (x > x1) x1 = x;
                    if (y < y0) y0 = y;
                    if (y > y1) y1 = y;
                }
            }

        int w = x1 - x0 + 1, h = y1 - y0 + 1;
        EXPECT_TRUE(w >= want[i] - 1 && w <= want[i] + 1);
        EXPECT_TRUE(h >= want[i] - 1 && h <= want[i] + 1);
    }

    /* identity attenuation restores the plain glPointSize size */
    glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION, (GLfloat[3]){ 1.0f, 0.0f, 0.0f });
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, -9);
    glEnd();
    glFinish();

    GLubyte p[4];
    glReadPixels(24, 24, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] > 200);
    glReadPixels(15, 15, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] == 0);
}

/* legacy internal formats expand to the right RGBA values on sampling */
TEST(legacy_texture_formats)
{
    struct { GLenum fmt; const char *name; GLubyte want[4]; } cases[] = {
        { GL_LUMINANCE,       "LUMINANCE",       { 128, 128, 128, 255 } },
        { GL_LUMINANCE_ALPHA, "LUMINANCE_ALPHA", { 128, 128, 128, 64 } },
        { GL_ALPHA,           "ALPHA",           { 0, 0, 0, 128 } },
        { GL_INTENSITY,       "INTENSITY",       { 128, 128, 128, 128 } },
    };

    while (glGetError() != GL_NO_ERROR) { }   /* start clean */

    for (unsigned i = 0; i < 4; i++)
    {
        setup_2d(64, 64);
        glClear(GL_COLOR_BUFFER_BIT);
        while (glGetError() != GL_NO_ERROR) { }

        GLuint tex = 0;
        GLubyte data[4] = { 128, 64, 32, 128 };     /* L, A, ?, ? per kind */

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, cases[i].fmt, 1, 1, 0, cases[i].fmt, GL_UNSIGNED_BYTE, data);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glEnable(GL_TEXTURE_2D);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        glBegin(GL_QUADS);
        glTexCoord2f(0.5f, 0.5f);
        glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
        glEnd();

        glFinish();
        EXPECT_PIXEL(32, 32, cases[i].want[0], cases[i].want[1], cases[i].want[2], cases[i].want[3], 8);

        /* readback packs back to the legacy layout */
        GLubyte back[4] = { 0, 0, 0, 0 };
        glGetTexImage(GL_TEXTURE_2D, 0, cases[i].fmt, GL_UNSIGNED_BYTE, back);
        EXPECT_NO_GL_ERROR();

        GLint ifmt = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &ifmt);
        EXPECT_TRUE(ifmt != 0);

        if (cases[i].fmt == GL_LUMINANCE_ALPHA)
        {
            EXPECT_TRUE(back[0] > 120 && back[0] < 136);
            EXPECT_TRUE(back[1] > 56 && back[1] < 72);
        }
        else
        {
            EXPECT_TRUE(back[0] > 120 && back[0] < 136);
        }

        glDisable(GL_TEXTURE_2D);
        glDeleteTextures(1, &tex);
    }
}

TEST(texture_residency)
{
    GLuint t[3] = { 0, 0, 0 };
    glGenTextures(3, t);

    GLclampf prio[3] = { 1.0f, 0.5f, 0.1f };
    glPrioritizeTextures(3, t, prio);
    EXPECT_NO_GL_ERROR();

    GLboolean res[3] = { GL_FALSE, GL_FALSE, GL_FALSE };
    GLboolean all = glAreTexturesResident(3, t, res);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(all == GL_TRUE || all == GL_FALSE);
    EXPECT_TRUE(res[0] == GL_TRUE || res[0] == GL_FALSE);

    /* a name of 0 is not a texture */
    GLuint bad[1] = { 0 };
    GLboolean rbad[1] = { GL_TRUE };
    glAreTexturesResident(1, bad, rbad);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteTextures(3, t);
}

int main(void)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return 1;

    RUN(point_sprite_orientation);
    RUN(point_size_attenuation);
    RUN(legacy_texture_formats);
    RUN(texture_residency);

    return mgl_test_summary();
}
