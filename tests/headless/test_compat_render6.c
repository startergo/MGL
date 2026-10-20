/*
 * test_compat_render6.c — GL_CLAMP / border colour behaviour (compat domain).
 *
 * GL_CLAMP and GL_CLAMP_TO_BORDER need border texel fetches.  Apple GPUs fetch
 * a border only for the three fixed Metal border colours, and never blend it
 * GL_CLAMP-style at the edge.  MGL therefore:
 *   - hands GL_CLAMP (NEAREST) / CLAMP_TO_BORDER to the sampler when the
 *     border colour is transparent/opaque black or opaque white (hardware
 *     MTLSamplerAddressModeClampToBorderColor), and
 *   - emulates everything else in the generated fixed-function shader: exact
 *     border colour, GL_CLAMP's 50/50 edge blend for LINEAR.
 *
 * The 4x4 texture is uniformly red; texel centres are at s = 1/8, 3/8, 5/8, 7/8.
 * Quads map window x to s so that s = -1/2, -1/8, 0 and +1/8 land exactly on
 * the centres of pixels 0, 6, 8 and 10.
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
    glPixelZoom(1, 1);
}

static GLuint red_tex(void)
{
    GLuint t;
    GLubyte px[4 * 4 * 4];
    for (int i = 0; i < 16; i++)
    {
        px[i * 4 + 0] = 255; px[i * 4 + 1] = 0;
        px[i * 4 + 2] = 0; px[i * 4 + 3] = 255;
    }
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return t;
}

/* s = -0.53125 .. 3.46875 across the quad: pixel centres 0/6/8/10 get
 * s = -0.5, -1/8, 0, +1/8 (see the header comment); t constant 0.5 */
static void draw_linear_quad(void)
{
    glBegin(GL_QUADS);
    glTexCoord2f(-0.53125f, 0.5f); glVertex2f(0, 0);
    glTexCoord2f(3.46875f, 0.5f); glVertex2f(64, 0);
    glTexCoord2f(3.46875f, 0.5f); glVertex2f(64, 64);
    glTexCoord2f(-0.53125f, 0.5f); glVertex2f(0, 64);
    glEnd();
}

static void draw_nearest_quad(void)
{
    glBegin(GL_QUADS);
    glTexCoord2f(-0.25f, -0.25f); glVertex2f(0, 0);
    glTexCoord2f(1.25f, -0.25f); glVertex2f(64, 0);
    glTexCoord2f(1.25f, 1.25f); glVertex2f(64, 64);
    glTexCoord2f(-0.25f, 1.25f); glVertex2f(0, 64);
    glEnd();
}

/* GL_CLAMP + NEAREST + arbitrary border colour: border outside [0,1] */
TEST(clamp_nearest_arbitrary_border)
{
    setup_2d(64, 64);
    GLuint t = red_tex();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 0, 0, 1, 1 });

    GLint q = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, &q);
    EXPECT_TRUE(q == GL_CLAMP);

    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_nearest_quad();

    /* s at the pixel centres: 0 -> -0.238 (outside), 10 -> -0.0039 (outside),
     * 11 -> 0.0195 (texel 0), 63 -> 1.238 (outside) */
    EXPECT_PIXEL(0, 32, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(63, 32, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(32, 0, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(32, 63, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(10, 32, 0, 0, 255, 255, 2);
    EXPECT_PIXEL(11, 32, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

/* GL_CLAMP + LINEAR + arbitrary border: GL clamps the coordinate to [0,1]
 * first, so the border never enters with more than 50%: 50/50 exactly at the
 * edge (s=0) and everywhere outside, full texel from half a texel in */
TEST(clamp_linear_arbitrary_border)
{
    setup_2d(64, 64);
    GLuint t = red_tex();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 0, 0, 1, 1 });

    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_linear_quad();

    EXPECT_PIXEL(0, 32, 128, 0, 128, 255, 4);    /* s = -0.5: saturated 50/50 */
    EXPECT_PIXEL(6, 32, 128, 0, 128, 255, 4);    /* s = -1/8: still 50/50 */
    EXPECT_PIXEL(8, 32, 128, 0, 128, 255, 4);    /* s = 0: the 50/50 blend */
    EXPECT_PIXEL(10, 32, 255, 0, 0, 255, 4);     /* s = +1/8: full texel 0 */
    /* halfway between s=0 and s=1/8 (pixel 9, s=1/16): 75/25 */
    EXPECT_PIXEL(9, 32, 191, 0, 64, 255, 6);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

/* GL_CLAMP + NEAREST + a Metal-native border colour goes to the hardware
 * sampler (CLAMP_TO_BORDER address mode); switching colours flips between the
 * hardware and the shader path at runtime */
TEST(clamp_nearest_hardware_border)
{
    setup_2d(64, 64);
    GLuint t = red_tex();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 0, 0, 0, 1 });

    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_nearest_quad();

    EXPECT_PIXEL(0, 32, 0, 0, 0, 255, 2);
    EXPECT_PIXEL(63, 32, 0, 0, 0, 255, 2);
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);
    EXPECT_PIXEL(11, 32, 255, 0, 0, 255, 2);

    /* switching to an arbitrary colour flips the unit to the shader path */
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 0, 1, 0, 1 });
    glClear(GL_COLOR_BUFFER_BIT);
    draw_nearest_quad();
    EXPECT_PIXEL(0, 32, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);

    /* ...and back to hardware (opaque white) */
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 1, 1, 1, 1 });
    glClear(GL_COLOR_BUFFER_BIT);
    draw_nearest_quad();
    EXPECT_PIXEL(0, 32, 255, 255, 255, 255, 2);
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

/* GL_CLAMP_TO_BORDER with an arbitrary colour: full border half a texel
 * outside (no saturation at 50/50 as GL_CLAMP has) */
TEST(clamp_to_border_linear_arbitrary)
{
    setup_2d(64, 64);
    GLuint t = red_tex();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, (GLfloat[4]){ 0, 0, 1, 1 });

    glEnable(GL_TEXTURE_2D);
    glClear(GL_COLOR_BUFFER_BIT);
    draw_linear_quad();

    EXPECT_PIXEL(8, 32, 128, 0, 128, 255, 4);    /* s = 0: 50/50 as in GL_CLAMP */
    EXPECT_PIXEL(10, 32, 255, 0, 0, 255, 4);     /* s = +1/8: full texel 0 */
    EXPECT_PIXEL(6, 32, 0, 0, 255, 255, 4);      /* s = -1/8: full border now */
    EXPECT_PIXEL(0, 32, 0, 0, 255, 255, 4);      /* s = -0.5: still full border */

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &t);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    RUN(clamp_nearest_arbitrary_border);
    RUN(clamp_linear_arbitrary_border);
    RUN(clamp_nearest_hardware_border);
    RUN(clamp_to_border_linear_arbitrary);

    return mgl_test_summary();
}
