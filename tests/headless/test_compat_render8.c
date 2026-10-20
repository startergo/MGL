/*
 * test_compat_render8.c — indexed line stipple (compat domain).
 *
 * The line-stipple distance is a per-vertex attribute, but glDrawElements can
 * reuse a vertex at several places of the index stream, where it carries
 * different distances from the start of the strip.  MGL de-indexes stippled
 * indexed line draws (gathering every enabled client array in draw order).
 *
 * Geometry: vertices A(0,32) B(20,32) C(40,32); indices 0,1,2, 2,1,0 draw the
 * strip forward and then backward, so the pattern runs 0..40 and 40..80.
 * Stipple 0xF00F factor 1 (4 on, 8 off, 4 on per 16 pixels):
 *   forward  lit at x mod 16 in {0..3, 12..15}
 *   backward lit at x mod 16 in {0..4, 13..15}
 *   union:   [0..4] [12..20] [28..36]; a line at y=32 lights row 31 (diamond
 *   rule).  Per-vertex (wrong) distances would light 0-3,12-19,28-35 instead.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void *compat_ctx;

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

static const float verts[3 * 4] = {
    0, 32, 0, 1,
    20, 32, 0, 1,
    40, 32, 0, 1,
};
static const GLubyte cols[3 * 4] = {
    255, 0, 0, 255,
    255, 0, 0, 255,
    255, 0, 0, 255,
};
static const GLushort idx[6] = { 0, 1, 2, 2, 1, 0 };

static int lit(int x, int y)
{
    GLubyte p[4] = { 0, 0, 0, 0 };

    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    return p[0] > 200 && p[1] < 50 && p[2] < 50 && p[3] > 200;
}

TEST(draw_elements_line_stipple)
{
    setup_2d(64, 64);

    glVertexPointer(4, GL_FLOAT, 0, verts);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, cols);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(1, 0xF00F);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_LINE_STRIP, 6, GL_UNSIGNED_SHORT, idx);

    /* 0xF00F lights pattern bits 0-3 and 12-15 (LSB first); both passes run
     * 0..80 pixels, and 80 is a multiple of 16, so the union is
     * x mod 16 in {0..3, 12..15} for x in 0..39.  (Per-vertex distances
     * instead of draw-order ones would also light x=4 and x=36.) */
    EXPECT_TRUE(lit(0, 31));
    EXPECT_TRUE(lit(3, 31));
    EXPECT_TRUE(lit(12, 31));
    EXPECT_TRUE(lit(16, 31));
    EXPECT_TRUE(lit(19, 31));
    EXPECT_TRUE(lit(28, 31));
    EXPECT_TRUE(lit(32, 31));
    EXPECT_TRUE(lit(35, 31));

    EXPECT_TRUE(!lit(4, 31));   /* discriminates against per-vertex distances */
    EXPECT_TRUE(!lit(8, 31));
    EXPECT_TRUE(!lit(11, 31));
    EXPECT_TRUE(!lit(20, 31));
    EXPECT_TRUE(!lit(24, 31));
    EXPECT_TRUE(!lit(36, 31));  /* discriminates against per-vertex distances */
    EXPECT_TRUE(!lit(38, 31));
    EXPECT_TRUE(!lit(41, 31));  /* beyond the geometry */
    EXPECT_TRUE(!lit(0, 32));   /* the row below the line */
    EXPECT_TRUE(!lit(0, 30));

    glDisable(GL_LINE_STIPPLE);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    EXPECT_NO_GL_ERROR();
}

/* the same indices without stipple must still draw the plain strip */
TEST(draw_elements_lines_unstippled)
{
    setup_2d(64, 64);

    glVertexPointer(4, GL_FLOAT, 0, verts);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, cols);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_LINE_STRIP, 6, GL_UNSIGNED_SHORT, idx);

    for (int x = 0; x <= 39; x++)
        EXPECT_TRUE(lit(x, 31));
    EXPECT_TRUE(!lit(40, 31));
    EXPECT_TRUE(!lit(41, 31));

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    EXPECT_NO_GL_ERROR();
}

/* GL_LINES: the distance restarts for every segment, also when indexed */
TEST(draw_elements_lines_segments)
{
    setup_2d(64, 64);

    static const float v2[4 * 4] = {
        0, 32, 0, 1,
        10, 32, 0, 1,
        10, 32, 0, 1,
        24, 32, 0, 1,
    };
    static const GLubyte c2[4 * 4] = {
        255, 0, 0, 255,  255, 0, 0, 255,
        255, 0, 0, 255,  255, 0, 0, 255,
    };
    static const GLushort i2[4] = { 0, 1, 2, 3 };

    glVertexPointer(4, GL_FLOAT, 0, v2);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, c2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glEnable(GL_LINE_STIPPLE);
    glLineStipple(1, 0x00FF);   /* 8 on, 8 off, restarting per segment */

    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_LINES, 4, GL_UNSIGNED_SHORT, i2);

    /* GL_LINES restarts the pattern at every segment: the first 8 pixels of
     * each segment are lit (0x00FF, LSB first) */
    EXPECT_TRUE(lit(0, 31));
    EXPECT_TRUE(lit(7, 31));
    EXPECT_TRUE(lit(10, 31));
    EXPECT_TRUE(lit(17, 31));
    EXPECT_TRUE(!lit(8, 31));
    EXPECT_TRUE(!lit(9, 31));
    EXPECT_TRUE(!lit(19, 31));
    EXPECT_TRUE(!lit(23, 31));

    glDisable(GL_LINE_STIPPLE);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    RUN(draw_elements_line_stipple);
    RUN(draw_elements_lines_unstippled);
    RUN(draw_elements_lines_segments);

    return mgl_test_summary();
}
