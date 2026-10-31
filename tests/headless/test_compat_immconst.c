/*
 * test_compat_immconst.c — pixel-exact coverage of the session-37 immediate
 * mode "varying stream" upload gate (compat_vb.c flush_imm):
 *   - a stream whose recorded values are constant across the batch must read
 *     the generic current value (constant attrib), not stale array data;
 *   - attribute arrays enabled by an earlier batch must be disabled again
 *     when a later batch does not feed them (stale-disable path);
 *   - texcoord / normal / color each in both roles (varying and constant),
 *     with the fixed-function pipeline (lighting, texturing) consuming them.
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
    glShadeModel(GL_SMOOTH);
    glClearColor(0, 0, 0, 1);
}

static void quad_at(float x0, float y0, float x1, float y1)
{
    glBegin(GL_QUADS);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

/* batch A: per-vertex colors (varying stream, smooth shade)
 * batch B: one glColor inside the batch (constant stream -> constant attrib)
 * the two batches overlap nothing; center pixels must be the vertex colours */
TEST(imm_color_constant_vs_varying)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* varying color stream: red left column, green right column */
    glBegin(GL_QUADS);
    glColor3f(1, 0, 0);
    glVertex2f(2, 2);
    glVertex2f(30, 2);
    glColor3f(0, 1, 0);
    glVertex2f(30, 30);
    glVertex2f(2, 30);
    glEnd();

    /* constant color stream: single glColor inside the batch */
    glBegin(GL_QUADS);
    glColor3f(0, 0, 1);
    glVertex2f(34, 2);
    glVertex2f(62, 2);
    glVertex2f(62, 30);
    glVertex2f(34, 30);
    glEnd();

    /* constant color set BEFORE the batch (same constant path, cur_* route) */
    glBegin(GL_QUADS);
    glColor3f(1, 1, 0);
    glVertex2f(34, 34);
    glVertex2f(62, 34);
    glVertex2f(62, 62);
    glVertex2f(34, 62);
    glEnd();

    /* smooth shade: exact barycentric weights of the two quad triangles
     * (v0=v1 red, v2=v3 green): w_v0=21/98, w_v2=77/98 at these samples */
    EXPECT_PIXEL(6, 24, 55, 200, 0, 255, 12);
    EXPECT_PIXEL(26, 24, 55, 200, 0, 255, 12);
    EXPECT_PIXEL(48, 24, 0, 0, 255, 255, 8);   /* constant-in-batch: blue */
    EXPECT_PIXEL(48, 56, 255, 255, 0, 255, 8); /* constant-pre-batch: yellow */
}

/* stale-disable: batch 1 uses texture coords (array enabled), batch 2 does
 * not — batch 2 must render with unit-0 texture coords untouched by batch 1's
 * still-bound array */
TEST(imm_texcoord_stale_disable)
{
    GLuint tex;
    GLubyte data[4 * 4 * 4];

    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* 4x4 texture: left half red, right half green */
    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            GLubyte *px = &data[(y * 4 + x) * 4];
            px[0] = x < 2 ? 255 : 0;
            px[1] = x < 2 ? 0 : 255;
            px[2] = 0;
            px[3] = 255;
        }
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glEnable(GL_TEXTURE_2D);

    /* batch 1: texcoords varying (right half of the texture) */
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    glTexCoord2f(0.5f, 0.0f); glVertex2f(2, 2);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(30, 2);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(30, 30);
    glTexCoord2f(0.5f, 1.0f); glVertex2f(2, 30);
    glEnd();

    /* batch 2: no texcoords in the batch -> constant current texcoord.  NOTE
     * the current texcoord is the one batch 1 last set, so reset it first;
     * the texcoord array from batch 1 must be disabled, not still bound. */
    glTexCoord2f(0.0f, 0.0f);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    glVertex2f(34, 2);
    glVertex2f(62, 2);
    glVertex2f(62, 30);
    glVertex2f(34, 30);
    glEnd();

    EXPECT_PIXEL(8, 24, 0, 255, 0, 255, 8);   /* batch 1 right-half green */
    EXPECT_PIXEL(24, 24, 0, 255, 0, 255, 8);
    EXPECT_PIXEL(48, 24, 255, 0, 0, 255, 8);  /* batch 2 constant (0,0) red */
    EXPECT_PIXEL(58, 24, 255, 0, 0, 255, 8);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
}

/* normals: lighting on, constant normal inside the batch vs varying normals
 * across batches — the lit colours must track the normal actually used */
TEST(imm_normal_constant_lighting)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_FALSE);
    glShadeModel(GL_FLAT);

    /* directional light straight at the viewer */
    {
        float pos[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
        float amb[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        float dif[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
    }
    {
        float spec[4] = { 0, 0, 0, 1 };
        float emi[4] = { 0, 0, 0, 1 };
        glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
        glMaterialfv(GL_FRONT, GL_EMISSION, emi);
    }
    glColor3f(1, 1, 1);

    /* batch 1: normal faces the light -> fully lit white */
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glVertex2f(2, 2);
    glVertex2f(30, 2);
    glVertex2f(30, 30);
    glVertex2f(2, 30);
    glEnd();

    /* batch 2: no glNormal -> constant current (0,0,1), still fully lit;
     * batch 1's varying-free normal array must not leak a dark normal in */
    glBegin(GL_QUADS);
    glVertex2f(34, 2);
    glVertex2f(62, 2);
    glVertex2f(62, 30);
    glVertex2f(34, 30);
    glEnd();

    /* batch 3: normal turned away -> only ambient (black) */
    glBegin(GL_QUADS);
    glNormal3f(0, 0, -1);
    glVertex2f(34, 34);
    glVertex2f(62, 34);
    glVertex2f(62, 62);
    glVertex2f(34, 62);
    glEnd();

    /* default material diffuse 0.8 + light-model ambient 0.2*0.2 = 0.84 */
    EXPECT_PIXEL(16, 16, 214, 214, 214, 255, 10); /* facing light */
    EXPECT_PIXEL(48, 16, 214, 214, 214, 255, 10); /* constant normal path */
    EXPECT_PIXEL(48, 48, 0, 0, 0, 255, 12);      /* back-facing normal */

    glDisable(GL_LIGHTING);
}

/* single-vertex batches (GL_POINTS) exercise the n < 2 early-out: color and
 * position only, no arrays besides position */
TEST(imm_points_single_vertex)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glPointSize(3.0f);
    glBegin(GL_POINTS);
    glColor3f(0, 1, 0);
    glVertex2f(8, 8);
    glEnd();
    glBegin(GL_POINTS);
    glColor3f(1, 0, 1);
    glVertex2f(40, 40);
    glEnd();

    EXPECT_PIXEL(8, 8, 0, 255, 0, 255, 8);
    EXPECT_PIXEL(40, 40, 255, 0, 255, 255, 8);
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    if (!compat_ctx)
        MGL_FAIL("compat context required");

    RUN(imm_color_constant_vs_varying);
    RUN(imm_texcoord_stale_disable);
    RUN(imm_normal_constant_lighting);
    RUN(imm_points_single_vertex);

    return mgl_test_summary();
}
