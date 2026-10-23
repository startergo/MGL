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
    /* exactly the first 8 pixels of each 16 (LSB-first pattern) */
    EXPECT_TRUE(lit == 16);
    EXPECT_TRUE(lit + dark == 32);

    glReadPixels(2, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[1] > 200);
    glReadPixels(10, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] == 0 && p[1] == 0);
    glReadPixels(17, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] == 0 && p[1] == 0);
    glReadPixels(18, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[1] > 200);
    glReadPixels(25, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[1] > 200);
    glReadPixels(26, 31, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
    EXPECT_TRUE(p[0] == 0 && p[1] == 0);
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

/* ------------------------------------------------------------------ */
/* line stipple                                                        */
/* ------------------------------------------------------------------ */

/* pattern 0x00FF, factor 1, immediate GL_LINES: fragments 0..7 of every line
 * pass, 8..15 fail, 16..23 pass ... (LSB-first, see Mesa's line stipple) */
TEST(line_stipple_per_line_restart)
{
    setup_2d(64, 64);
    glColor4f(0, 1, 0, 1);
    glLineStipple(1, 0x00FF);
    glEnable(GL_LINE_STIPPLE);

    glClear(GL_COLOR_BUFFER_BIT);
    /* horizontal line covers window row y-1 (diamond rule, see line_diamond_rule) */
    glBegin(GL_LINES); glVertex2f(4, 16); glVertex2f(60, 16); glEnd();

    /* distances 0..7 on */
    EXPECT_PIXEL(4, 15, 0, 255, 0, 255, 4);
    EXPECT_PIXEL(10, 15, 0, 255, 0, 255, 4);
    /* distances 8..15 off */
    EXPECT_PIXEL(12, 15, 0, 0, 0, 255, 4);
    EXPECT_PIXEL(18, 15, 0, 0, 0, 255, 4);
    /* distances 16..23 on */
    EXPECT_PIXEL(21, 15, 0, 255, 0, 255, 4);

    /* second line in the same glBegin: the counter restarts per line */
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES);
    glVertex2f(40, 16); glVertex2f(60, 16);
    glVertex2f(4, 32); glVertex2f(24, 32);
    glEnd();
    EXPECT_PIXEL(44, 15, 0, 255, 0, 255, 4);   /* dist 4: on */
    EXPECT_PIXEL(52, 15, 0, 0, 0, 255, 4);     /* dist 12: off */
    EXPECT_PIXEL(8, 31, 0, 255, 0, 255, 4);    /* new line: dist 4 on again */

    glDisable(GL_LINE_STIPPLE);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(4, 16); glVertex2f(60, 16); glEnd();
    EXPECT_PIXEL(12, 15, 0, 255, 0, 255, 4);   /* disabled: continuous */

    EXPECT_NO_GL_ERROR();
}

/* pattern 0x0001, factor 4: fragments 0..3 on, everything else off */
TEST(line_stipple_factor)
{
    setup_2d(64, 64);
    glColor4f(0, 1, 0, 1);
    glLineStipple(4, 0x0001);
    glEnable(GL_LINE_STIPPLE);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(4, 16); glVertex2f(60, 16); glEnd();

    EXPECT_PIXEL(4, 15, 0, 255, 0, 255, 4);    /* dist 0..3 */
    EXPECT_PIXEL(6, 15, 0, 255, 0, 255, 4);
    EXPECT_PIXEL(8, 15, 0, 0, 0, 255, 4);      /* dist 4..63: bit 1..15 of 0x0001 */
    EXPECT_PIXEL(30, 15, 0, 0, 0, 255, 4);
    EXPECT_PIXEL(58, 15, 0, 0, 0, 255, 4);

    /* GL_LINE_STRIP: distances accumulate across the whole strip */
    glLineStipple(1, 0x00FF);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINE_STRIP);
    glVertex2f(4, 16); glVertex2f(20, 16);   /* 16 px */
    glVertex2f(20, 40);                      /* 24 px more */
    glEnd();
    /* segment 1: dist 0..15 → x 4..19 all on (bits 0..7 and 8..15? 0x00FF: bits 0..7 only) */
    EXPECT_PIXEL(4, 15, 0, 255, 0, 255, 4);    /* dist 0 */
    EXPECT_PIXEL(11, 15, 0, 255, 0, 255, 4);   /* dist 7 */
    EXPECT_PIXEL(12, 15, 0, 0, 0, 255, 4);     /* dist 8 */
    /* vertical segment: fragment centers x=19 (diamond rule), dist 16..39 */
    EXPECT_PIXEL(19, 19, 0, 255, 0, 255, 4);   /* dist 16+3=19 → bit 3: on */
    EXPECT_PIXEL(19, 27, 0, 0, 0, 255, 4);     /* dist ~27 → bit 11: off */

    glDisable(GL_LINE_STIPPLE);
    EXPECT_NO_GL_ERROR();
}

/* client arrays: glDrawArrays(GL_LINES) with stipple */
TEST(line_stipple_arrays)
{
    setup_2d(64, 64);
    glColor4f(0, 1, 0, 1);
    glLineStipple(1, 0x00FF);
    glEnable(GL_LINE_STIPPLE);

    const float v[] = { 4, 16,  60, 16,   4, 40,  60, 40 };
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, v);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 4);

    EXPECT_PIXEL(6, 15, 0, 255, 0, 255, 4);
    EXPECT_PIXEL(14, 15, 0, 0, 0, 255, 4);
    EXPECT_PIXEL(6, 39, 0, 255, 0, 255, 4);    /* second line restarts */
    EXPECT_PIXEL(14, 39, 0, 0, 0, 255, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_LINE_STIPPLE);
    EXPECT_NO_GL_ERROR();
}

/* indexed glDrawElements: the stipple distance follows INDEX order, not the
 * order the vertices happen to be stored in the array */
TEST(line_stipple_indexed)
{
    setup_2d(64, 64);
    glColor4f(0, 1, 0, 1);
    glLineStipple(1, 0x00FF);
    glEnable(GL_LINE_STIPPLE);

    /* vertices stored in increasing-x order; the strip runs right-to-left */
    const float v[] = { 4, 16,  20, 16,  40, 16,  56, 16 };
    const GLushort idx[] = { 3, 2, 1, 0 };
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, v);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_LINE_STRIP, 4, GL_UNSIGNED_SHORT, idx);

    /* fragments run x = 56,55,...,41 (dist 0..15), 40,...,21 (dist 16..35), 20,...,4 */
    EXPECT_PIXEL(52, 15, 0, 255, 0, 255, 4);   /* dist ~4: on */
    EXPECT_PIXEL(50, 15, 0, 255, 0, 255, 4);   /* dist ~6: on */
    EXPECT_PIXEL(46, 15, 0, 0, 0, 255, 4);     /* dist ~10: off */
    EXPECT_PIXEL(42, 15, 0, 0, 0, 255, 4);     /* dist ~14: off */
    /* second segment dist 16..: bit 0 of 0x00FF again */
    EXPECT_PIXEL(37, 15, 0, 255, 0, 255, 4);   /* dist ~21: on */
    EXPECT_PIXEL(35, 15, 0, 255, 0, 255, 4);   /* dist ~23: on */
    EXPECT_PIXEL(31, 15, 0, 0, 0, 255, 4);     /* dist ~27: off */
    EXPECT_PIXEL(23, 15, 0, 255, 0, 255, 4);   /* third segment: dist ~48: on */

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_LINE_STIPPLE);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* GL_GENERATE_MIPMAP (SGIS_generate_mipmap compat semantics)          */
/* ------------------------------------------------------------------ */

TEST(generate_mipmap_on_teximage)
{
    setup_2d(64, 64);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    GLint gmm = -1;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, &gmm);
    EXPECT_EQ_INT(gmm, GL_FALSE);

    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    gmm = -1;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, &gmm);
    EXPECT_EQ_INT(gmm, GL_TRUE);

    /* 8x8: four solid quadrants; mip level 3 is 1x1 = the average of all */
    GLubyte img[8 * 8 * 3];
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
        {
            GLubyte *p = &img[(y * 8 + x) * 3];
            if (y < 4) { if (x < 4) { p[0]=255; p[1]=0;   p[2]=0; }   /* red */
                         else       { p[0]=0;   p[1]=255; p[2]=0; } } /* green */
            else       { if (x < 4) { p[0]=0;   p[1]=0;   p[2]=255; } /* blue */
                         else       { p[0]=255; p[1]=255; p[2]=255; } } /* white */
        }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, img);

    /* the mipmap chain must exist: level 3 present */
    GLint w = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 3, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 1);

    /* complete texture (mipmap filter + full chain): draw 1 pixel of level 3 =
     * the average of all quadrants (0.5, 0.5, 0.5) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(1, 0);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(0, 1);
    glEnd();
    EXPECT_PIXEL(0, 0, 128, 128, 128, 255, 10);

    /* GENERATE_MIPMAP off: a level-0 upload on a fresh texture must NOT add levels */
    GLuint tex2 = 0;
    glGenTextures(1, &tex2);
    glBindTexture(GL_TEXTURE_2D, tex2);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_FALSE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
    w = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 0);    /* level 1 does not exist */

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &tex2);
    EXPECT_NO_GL_ERROR();
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

    /* added 2026-10-23: per-line restart, factor, arrays, indexed, GENERATE_MIPMAP */
    RUN(line_stipple_per_line_restart);
    RUN(line_stipple_factor);
    RUN(line_stipple_arrays);
    RUN(line_stipple_indexed);
    RUN(generate_mipmap_on_teximage);

    return mgl_test_summary();
}
