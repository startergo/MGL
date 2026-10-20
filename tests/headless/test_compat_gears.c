/*
 * test_compat_gears.c — end-to-end "classic app" test (compat domain).
 *
 * A glxgears-style frame: three gears built in immediate mode inside display
 * lists (quads, triangle fans, per-vertex normals), a perspective projection,
 * matrix stacks, two directional lights, materials and depth testing.  The
 * rendered frame is compared against a stored reference:
 *   - structural checks (background, coverage, per-gear colour presence), and
 *   - 10x10-block colour averages of the whole frame vs. compat_gears_ref.h.
 *
 * The reference can be regenerated (after an intended rendering change) with
 *   ./test_compat_gears --dump > tests/headless/compat_gears_ref.h
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"
#include <stdlib.h>

#ifndef MGLC_GEARS_NO_REF
#include "compat_gears_ref.h"
#endif

#define W 160
#define H 120

static void *compat_ctx;
static GLuint gear_lists[3];

/* one gear, drawn in immediate mode (design after the classic glxgears gear):
 * inner_radius..outer_radius ring, `teeth` trapezoid teeth of tooth_depth,
 * thickness `width`.  The faces use per-vertex normals (smooth shading), the
 * teeth flanks are flat-shaded quads. */
static void gear(GLfloat inner_radius, GLfloat outer_radius, GLfloat width,
                 GLint teeth, GLfloat tooth_depth)
{
    GLint i;
    GLfloat r0 = inner_radius;
    GLfloat r1 = outer_radius - tooth_depth / 2.0f;
    GLfloat r2 = outer_radius + tooth_depth / 2.0f;
    GLfloat da = 2.0f * (GLfloat)M_PI / teeth;

    /* front side (fan over ring + teeth, normals in the xy plane) */
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, width * 0.5f);
    for (i = 0; i <= teeth; i++)
    {
        GLfloat a = i * 2.0f * (GLfloat)M_PI / teeth;

        glVertex3f(cosf(a) * r2, sinf(a) * r2, width * 0.5f);
        glVertex3f(cosf(a + da / 3.0f) * r1, sinf(a + da / 3.0f) * r1, width * 0.5f);
        glVertex3f(cosf(a + 2.0f * da / 3.0f) * r1, sinf(a + 2.0f * da / 3.0f) * r1, width * 0.5f);
        glVertex3f(cosf(a + da) * r2, sinf(a + da) * r2, width * 0.5f);
    }
    glEnd();

    /* back side */
    glBegin(GL_TRIANGLE_FAN);
    glNormal3f(0.0f, 0.0f, -1.0f);
    glVertex3f(0.0f, 0.0f, -width * 0.5f);
    for (i = 0; i <= teeth; i++)
    {
        GLfloat a = i * 2.0f * (GLfloat)M_PI / teeth;

        glVertex3f(cosf(a) * r2, sinf(a) * r2, -width * 0.5f);
        glVertex3f(cosf(a + da) * r2, sinf(a + da) * r2, -width * 0.5f);
        glVertex3f(cosf(a + 2.0f * da / 3.0f) * r1, sinf(a + 2.0f * da / 3.0f) * r1, -width * 0.5f);
        glVertex3f(cosf(a + da / 3.0f) * r1, sinf(a + da / 3.0f) * r1, -width * 0.5f);
    }
    glEnd();

    /* teeth flanks (flat normals) */
    glBegin(GL_QUADS);
    da = 2.0f * (GLfloat)M_PI / teeth;
    for (i = 0; i < teeth; i++)
    {
        GLfloat a = -i * da;
        GLfloat b = a + da / 3.0f;
        GLfloat c = a + 2.0f * da / 3.0f;
        GLfloat d = a + da;
        GLfloat u[2], v[2];

        u[0] = cosf(b); u[1] = sinf(b);
        v[0] = cosf(a); v[1] = sinf(a);

        glNormal3f(v[0], v[1], 0.0f);
        glVertex3f(v[0] * r2, v[1] * r2, -width * 0.5f);
        glVertex3f(v[0] * r2, v[1] * r2, width * 0.5f);
        glNormal3f(u[0], u[1], 0.0f);
        glVertex3f(u[0] * r2, u[1] * r2, width * 0.5f);
        glVertex3f(u[0] * r2, u[1] * r2, -width * 0.5f);
        glNormal3f(0.0f, 0.0f, 1.0f);
        glVertex3f(u[0] * r2, u[1] * r2, width * 0.5f);
        glVertex3f(u[0] * r1, u[1] * r1, width * 0.5f);
        glNormal3f(0.0f, 0.0f, -1.0f);
        glVertex3f(u[0] * r1, u[1] * r1, -width * 0.5f);
        glVertex3f(u[0] * r2, u[1] * r2, -width * 0.5f);

        u[0] = cosf(d); u[1] = sinf(d);
        v[0] = cosf(c); v[1] = sinf(c);

        glNormal3f(v[0], v[1], 0.0f);
        glVertex3f(v[0] * r2, v[1] * r2, -width * 0.5f);
        glVertex3f(v[0] * r2, v[1] * r2, width * 0.5f);
        glNormal3f(u[0], u[1], 0.0f);
        glVertex3f(u[0] * r2, u[1] * r2, width * 0.5f);
        glVertex3f(u[0] * r2, u[1] * r2, -width * 0.5f);
        glNormal3f(0.0f, 0.0f, 1.0f);
        glVertex3f(u[0] * r2, u[1] * r2, width * 0.5f);
        glVertex3f(u[0] * r1, u[1] * r1, width * 0.5f);
        glNormal3f(0.0f, 0.0f, -1.0f);
        glVertex3f(u[0] * r1, u[1] * r1, -width * 0.5f);
        glVertex3f(u[0] * r2, u[1] * r2, -width * 0.5f);
    }

    /* inside rim */
    glNormal3f(0.0f, 0.0f, 1.0f);
    glVertex3f(r0, 0.0f, -width * 0.5f);
    glVertex3f(r0, 0.0f, width * 0.5f);
    glNormal3f(-1.0f, 0.0f, 0.0f);
    glVertex3f(-r0, 0.0f, width * 0.5f);
    glVertex3f(-r0, 0.0f, -width * 0.5f);
    glEnd();
}

static void build_lists(void)
{
    static const GLfloat red[4] = { 0.8f, 0.1f, 0.0f, 1.0f };
    static const GLfloat green[4] = { 0.0f, 0.7f, 0.2f, 1.0f };
    static const GLfloat blue[4] = { 0.2f, 0.2f, 0.8f, 1.0f };
    static const GLfloat pos0[4] = { 5.0f, 5.0f, 10.0f, 0.0f };
    static const GLfloat pos1[4] = { -5.0f, 5.0f, 10.0f, 0.0f };

    glLightfv(GL_LIGHT0, GL_POSITION, pos0);
    glLightfv(GL_LIGHT1, GL_POSITION, pos1);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[4]){ 0.1f, 0.1f, 0.1f, 1.0f });

    gear_lists[0] = glGenLists(3);
    gear_lists[1] = gear_lists[0] + 1;
    gear_lists[2] = gear_lists[0] + 2;

    glNewList(gear_lists[0], GL_COMPILE);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, red);
    glMaterialfv(GL_FRONT, GL_SPECULAR, (GLfloat[4]){ 0.5f, 0.5f, 0.5f, 1.0f });
    glMaterialf(GL_FRONT, GL_SHININESS, 20.0f);
    gear(1.0f, 4.0f, 2.0f, 20, 0.7f);       /* red */
    glEndList();

    glNewList(gear_lists[1], GL_COMPILE);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, green);
    glMaterialfv(GL_FRONT, GL_SPECULAR, (GLfloat[4]){ 0.5f, 0.5f, 0.5f, 1.0f });
    glMaterialf(GL_FRONT, GL_SHININESS, 20.0f);
    gear(0.5f, 2.0f, 2.0f, 10, 0.7f);       /* green */
    glEndList();

    glNewList(gear_lists[2], GL_COMPILE);
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, blue);
    glMaterialfv(GL_FRONT, GL_SPECULAR, (GLfloat[4]){ 0.5f, 0.5f, 0.5f, 1.0f });
    glMaterialf(GL_FRONT, GL_SHININESS, 20.0f);
    gear(1.5f, 3.0f, 1.5f, 16, 0.7f);       /* blue */
    glEndList();

    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

/* one frame at the classic t=0 gear angles */
static void draw_frame(void)
{
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-1.0, 1.0, -(double)H / W, (double)H / W, 5.0, 60.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -40.0f);

    glPushMatrix();
    glRotatef(20.0f, 1.0f, 0.0f, 0.0f);
    glRotatef(30.0f, 0.0f, 1.0f, 0.0f);

    glPushMatrix();
    glTranslatef(-3.0f, -2.0f, 0.0f);
    glRotatef(0.0f, 0.0f, 0.0f, 1.0f);
    glCallList(gear_lists[0]);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(3.1f, -2.0f, 0.0f);
    glRotatef(-9.0f, 0.0f, 0.0f, 1.0f);
    glCallList(gear_lists[1]);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(-3.1f, 4.2f, 0.0f);
    glRotatef(-25.0f, 0.0f, 0.0f, 1.0f);
    glCallList(gear_lists[2]);
    glPopMatrix();

    glPopMatrix();
}

/* 10x10-block RGB averages: 16 x 12 blocks */
#define BLK 10
#define BW (W / BLK)
#define BH (H / BLK)

static void block_average(const GLubyte *img, GLubyte out[BW * BH * 3])
{
    for (int by = 0; by < BH; by++)
        for (int bx = 0; bx < BW; bx++)
        {
            int r = 0, g = 0, b = 0;

            for (int y = 0; y < BLK; y++)
                for (int x = 0; x < BLK; x++)
                {
                    const GLubyte *p = &img[((by * BLK + y) * W + bx * BLK + x) * 4];

                    r += p[0]; g += p[1]; b += p[2];
                }
            out[(by * BW + bx) * 3 + 0] = (GLubyte)(r / (BLK * BLK));
            out[(by * BW + bx) * 3 + 1] = (GLubyte)(g / (BLK * BLK));
            out[(by * BW + bx) * 3 + 2] = (GLubyte)(b / (BLK * BLK));
        }
}

TEST(gears_frame)
{
    GLubyte *img = (GLubyte *)malloc(W * H * 4);
    GLubyte blocks[BW * BH * 3];
    int covered = 0, r_gear = 0, g_gear = 0, b_gear = 0;

    draw_frame();
    glFinish();

    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, img);

    /* structure: background is the clear colour, the gears cover a good part
     * of the frame and all three material hues are present */
    EXPECT_NEAR(img[((H - 1 - 0) * W + 0) * 4 + 0], 13, 4);
    EXPECT_NEAR(img[((H - 1 - 0) * W + 0) * 4 + 2], 26, 4);

    for (int i = 0; i < W * H; i++)
    {
        int r = img[i * 4], g = img[i * 4 + 1], b = img[i * 4 + 2];

        if (r > 60 || g > 60 || b > 60) covered++;
        if (r > 60 && r > 2 * g) r_gear++;
        if (g > 60 && g > 2 * r) g_gear++;
        if (b > 60 && b > 2 * g) b_gear++;
    }
    EXPECT_TRUE(covered > W * H / 4 && covered < W * H * 2 / 3);
    EXPECT_TRUE(r_gear > 200);
    EXPECT_TRUE(g_gear > 100);
    EXPECT_TRUE(b_gear > 100);

    block_average(img, blocks);

#ifndef MGLC_GEARS_NO_REF
    /* pixel comparison against the stored reference (block averages, small
     * tolerance for GPU rounding) */
    {
        int bad = 0, worst = 0;

        for (int i = 0; i < BW * BH * 3; i++)
        {
            int d = blocks[i] - (int)compat_gears_ref[i];

            if (d < 0) d = -d;
            if (d > worst) worst = d;
            if (d > 4) bad++;
        }
        EXPECT_TRUE(bad == 0);
        EXPECT_TRUE(worst <= 4);
    }
#endif

    free(img);
    EXPECT_NO_GL_ERROR();
}

int main(int argc, char **argv)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    glViewport(0, 0, W, H);
    build_lists();

    if (argc > 1 && strcmp(argv[1], "--dump") == 0)
    {
        GLubyte *img = (GLubyte *)malloc(W * H * 4);
        GLubyte blocks[BW * BH * 3];

        draw_frame();
        glFinish();
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, img);
        block_average(img, blocks);

        printf("/* generated by: test_compat_gears --dump > tests/headless/compat_gears_ref.h */\n");
        printf("#ifndef COMPAT_GEARS_REF_H\n#define COMPAT_GEARS_REF_H\n");
        printf("/* 10x10-block RGB averages of the %dx%d glxgears-style frame */\n", W, H);
        printf("static const unsigned char compat_gears_ref[%d] = {\n", BW * BH * 3);
        for (int i = 0; i < BW * BH * 3; i++)
            printf("%d,%s", blocks[i], (i % 16 == 15) ? "\n" : "");
        printf("\n};\n#endif\n");
        free(img);
        return 0;
    }

    RUN(gears_frame);
    return mgl_test_summary();
}
