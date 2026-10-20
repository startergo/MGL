/*
 * test_compat_render3.c — line stipple, point size, texgen sphere/reflection
 * maps, COMBINE variants, CopyPixels, pixel maps.
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
    glDisable(GL_POLYGON_STIPPLE);
    glDisable(GL_LINE_STIPPLE);
    glPixelZoom(1, 1);
}

/* 16 pixels per pattern repeat: 0x00FF = first 8 drawn, next 8 dropped */
TEST(line_stipple)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(1, 0x00FF);
    glColor4f(0, 1, 0, 1);
    glLineWidth(1);

    glBegin(GL_LINES);
    glVertex2f(2, 32);      /* 32 pixels long: pixels 0..7 and 16..23 drawn */
    glVertex2f(34, 32);
    glEnd();

    glDisable(GL_LINE_STIPPLE);
    glFinish();

    GLubyte p[3];
    int lit = 0, dark = 0;

    for (int x = 2; x < 34; x++)
    {
        glReadPixels(x, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
        if (p[1] > 200 || p[0] > 200) lit++;
        else if (p[0] == 0 && p[1] == 0) dark++;
    }
    EXPECT_TRUE(lit >= 12 && lit <= 20);
    EXPECT_TRUE(lit + dark == 32);
}

/* glPointSize(8): the point covers about 8x8 pixels */
TEST(point_size)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(8.0f);
    glColor4f(1, 1, 0, 1);
    glBegin(GL_POINTS);
    glVertex2f(32, 32);
    glEnd();

    glFinish();
    glPointSize(1.0f);

    GLubyte p[3];
    int x0 = 64, x1 = -1, y0 = 64, y1 = -1;

    for (int y = 24; y < 40; y++)
        for (int x = 24; x < 40; x++)
        {
            glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
            if (p[0] > 200 && p[1] > 200)
            {
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
            }
        }

    EXPECT_TRUE(x1 - x0 >= 5 && x1 - x0 <= 8);
    EXPECT_TRUE(y1 - y0 >= 5 && y1 - y0 <= 8);
    EXPECT_TRUE(x0 > 24 && y0 > 24);   /* roughly centred on (32,32) */
}

/* NORMAL_MAP / REFLECTION_MAP texgen with identity modelview */
TEST(texgen_normal_reflection_map)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* 4x1 texture: texel x unique colour, sampled at (s, 0.5) */
    GLuint tex = 0;
    GLubyte img[4 * 4] = { 255, 0, 0, 255,   0, 255, 0, 255,   0, 0, 255, 255,   255, 255, 0, 255 };
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glColor4f(1, 1, 1, 1);

    /* normal (0,0,1), identity modelview: s = normal.x = 0 (red), t = 1 */
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP);
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);

    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glVertex2f(8, 8); glVertex2f(28, 8); glVertex2f(28, 28); glVertex2f(8, 28);
    glEnd();

    /* reflection: u = normalize(eye) = (0.94, 0.34, 0) at the sample point,
     * r = u - 2n(n.u) = u (n = (0,0,1), n.u = 0) -> s = 0.94 -> texel 3 (yellow) */
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);

    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glVertex2f(36, 8); glVertex2f(56, 8); glVertex2f(56, 28); glVertex2f(36, 28);
    glEnd();

    glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    glFinish();

    EXPECT_PIXEL(16, 16, 255, 0, 0, 255, 8);      /* NORMAL_MAP: s = n.x = 0 */
    EXPECT_PIXEL(44, 16, 255, 255, 0, 255, 8);    /* REFLECTION_MAP: s = u.x = 0.94 */
}

/* SPHERE_MAP with a tilted normal */
TEST(texgen_sphere_map)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint tex = 0;
    GLubyte img[4 * 4];
    for (int i = 0; i < 4; i++)
        img[i * 4] = (GLubyte)(i * 64), img[i * 4 + 1] = 128, img[i * 4 + 2] = 0, img[i * 4 + 3] = 255;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);

    /* normal (1,0,0), identity modelview:
     * u = n = (1,0,0); r = u - 2n(n.u) = (-1,0,0)
     * m = 2 sqrt(rx^2 + ry^2 + (rz+1)^2) = 2*sqrt(2)
     * s = rx/(2m) + 0.5 = -1/(4*sqrt(2)) + 0.5 = 0.3232 -> texel 1 (s, 4 texels, NEAREST) */
    glBegin(GL_QUADS);
    glNormal3f(1, 0, 0);
    glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
    glEnd();

    glDisable(GL_TEXTURE_GEN_S);
    glDisable(GL_TEXTURE_GEN_T);
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    glFinish();

    /* texel 1 = (64, 128, 0) */
    EXPECT_PIXEL(32, 32, 64, 128, 0, 255, 12);
}

/* COMBINE: INTERPOLATE / SUBTRACT / ADD_SIGNED + operand ONE_MINUS_SRC_COLOR */
TEST(texenv_combine_variants)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint tex = 0;
    GLubyte img[2 * 2 * 4] = {
        255, 128,   0, 255,     /* texel 0 */
          0,  64, 255, 128,     /* texel 1 */
    };
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (GLfloat[4]){ 0, 0, 0, 1 });

    glEnable(GL_TEXTURE_2D);
    glColor4f(0.5f, 1.0f, 0.25f, 1.0f);

    /* interpolate: Arg0*t + Arg1*(1-t) with t = texture alpha (texel 0: a=1) */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_INTERPOLATE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_TEXTURE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_PRIMARY_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE2_RGB, GL_TEXTURE);
    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND2_RGB, GL_SRC_ALPHA);

    glBegin(GL_QUADS);
    glTexCoord2f(0.25f, 0.25f);
    glVertex2f(8, 8); glVertex2f(28, 8); glVertex2f(28, 28); glVertex2f(8, 28);
    glEnd();

    /* subtract: PRIMARY_COLOR - TEXTURE (texel 1: (0, 64/255, 1)) */
    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_SUBTRACT);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_PRIMARY_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_TEXTURE);

    glBegin(GL_QUADS);
    glTexCoord2f(0.75f, 0.25f);
    glVertex2f(36, 8); glVertex2f(56, 8); glVertex2f(56, 28); glVertex2f(36, 28);
    glEnd();

    /* add signed, scale 2: 2*(primary + texture - 0.5), texel 0 (255,128,0) */
    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_ADD_SIGNED);
    glTexEnvi(GL_TEXTURE_ENV, GL_RGB_SCALE, 2);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE0_RGB, GL_TEXTURE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SOURCE1_RGB, GL_PRIMARY_COLOR);

    glBegin(GL_QUADS);
    glTexCoord2f(0.25f, 0.25f);
    glVertex2f(8, 36); glVertex2f(28, 36); glVertex2f(28, 56); glVertex2f(8, 56);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    glFinish();

    /* interpolate: rgb = tex * 1 + primary * 0 = (255,128,0) */
    EXPECT_PIXEL(16, 16, 255, 128, 0, 255, 12);
    /* subtract: primary - tex = (0.5, 1-0.25, 0.25-1) = (128, 191, 0);
     * alpha: COMBINE_ALPHA default MODULATE -> 1 * 128/255 = 128 */
    EXPECT_PIXEL(44, 16, 128, 191, 0, 128, 12);
    /* add signed x2: 2*((1+0.5)-0.5, (0.5+1)-0.5, (0+0.25)-0.5) = (255,255,0) */
    EXPECT_PIXEL(16, 44, 255, 255, 0, 255, 12);
}

/* glCopyPixels copies a region */
TEST(copy_pixels)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor4f(0, 0, 1, 1);
    glBegin(GL_QUADS);
    glVertex2f(4, 4); glVertex2f(12, 4); glVertex2f(12, 12); glVertex2f(4, 12);
    glEnd();

    glRasterPos2i(20, 20);
    glCopyPixels(4, 4, 8, 8, GL_COLOR);
    glFinish();

    EXPECT_PIXEL(24, 24, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(27, 27, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(19, 19, 0, 0, 0, 255, 8);
}

/* glPixelMap affects DrawPixels when GL_MAP_COLOR is on */
TEST(pixel_map)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* 2-entry maps: 0 -> 0, 1 -> 0.5 (halves every channel); the alpha map
     * stays identity-sized (single entry = 1.0) */
    GLfloat map[2] = { 0.0f, 0.5f };
    glPixelMapfv(GL_PIXEL_MAP_R_TO_R, 2, map);
    glPixelMapfv(GL_PIXEL_MAP_G_TO_G, 2, map);
    glPixelMapfv(GL_PIXEL_MAP_B_TO_B, 2, map);
    glPixelTransferi(GL_MAP_COLOR, GL_TRUE);

    GLubyte px[4] = { 255, 255, 0, 255 };
    glColor4f(1, 1, 1, 1);      /* DrawPixels modulates by the raster colour */
    glRasterPos2i(8, 8);
    glDrawPixels(2, 2, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glFinish();

    EXPECT_PIXEL(8, 8, 128, 128, 0, 255, 8);

    glPixelTransferi(GL_MAP_COLOR, GL_FALSE);
}

int main(void)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return 1;

    RUN(line_stipple);
    RUN(point_size);
    RUN(texgen_normal_reflection_map);
    RUN(texgen_sphere_map);
    RUN(texenv_combine_variants);
    RUN(copy_pixels);
    RUN(pixel_map);

    return mgl_test_summary();
}
