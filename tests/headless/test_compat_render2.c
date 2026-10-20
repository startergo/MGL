/*
 * test_compat_render2.c — more pixel-exact coverage of the compat domain:
 * matrix transforms, positional light + glColorMaterial, texenv modes
 * (REPLACE/DECAL/BLEND/ADD/COMBINE), glRasterPos/glDrawPixels/glBitmap,
 * glAccum, polygon stipple, evaluators and select/feedback.
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
    glDisable(GL_CLIP_PLANE0);
    glShadeModel(GL_SMOOTH);
    glPixelZoom(1, 1);
}

TEST(matrix_transforms)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor4f(1, 1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();
    glTranslatef(16, 16, 0);
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(16, 0);
    glVertex2f(16, 16);
    glVertex2f(0, 16);
    glEnd();
    glPopMatrix();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(24, 24, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(4, 4, 0, 0, 0, 255, 8);
    EXPECT_PIXEL(40, 40, 0, 0, 0, 255, 8);

    /* rotate 90 degrees about z: (x,y) -> (-y,x) around the origin */
    glClear(GL_COLOR_BUFFER_BIT);
    glPushMatrix();
    glTranslatef(32, 32, 0);
    glRotatef(90, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex2f(4, 4);
    glVertex2f(20, 4);
    glVertex2f(20, 20);
    glVertex2f(4, 20);
    glEnd();
    glPopMatrix();

    EXPECT_NO_GL_ERROR();
    /* (4..20)^2 rotated 90 deg CCW around (32,32) -> x 12..28, y 36..52 */
    EXPECT_PIXEL(20, 44, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(52, 12, 0, 0, 0, 255, 8);
    EXPECT_PIXEL(12, 12, 0, 0, 0, 255, 8);

    /* scale */
    glClear(GL_COLOR_BUFFER_BIT);
    glPushMatrix();
    glScalef(2, 2, 1);
    glBegin(GL_QUADS);
    glVertex2f(4, 4);
    glVertex2f(12, 4);
    glVertex2f(12, 12);
    glVertex2f(4, 12);
    glEnd();
    glPopMatrix();

    /* (4..12)^2 scaled by 2 -> 8..24 */
    EXPECT_PIXEL(20, 20, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(10, 10, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(4, 4, 0, 0, 0, 255, 8);
}

TEST(positional_light_and_colormaterial)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat[4]){ 32, 8, 10, 1 });   /* positional */
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat[4]){ 1, 1, 1, 1 });

    /* color material: material diffuse tracks the current color */
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT, GL_DIFFUSE);

    glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });

    /* fixed-function lighting is per-vertex (Gouraud): subdivide so the vertex
     * under the light is bright; compare against the CPU reference below */
    glColor4f(1, 0, 0, 1);
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    for (int ty = 0; ty < 8; ty++)
        for (int tx = 0; tx < 8; tx++)
        {
            float x0 = tx * 8.0f, y0 = ty * 8.0f;
            glVertex3f(x0, y0, 0);
            glVertex3f(x0 + 8, y0, 0);
            glVertex3f(x0 + 8, y0 + 8, 0);
            glVertex3f(x0, y0 + 8, 0);
        }
    glEnd();

    EXPECT_NO_GL_ERROR();
    /* the vertex at (32,8) sits exactly under the light: fully lit */
    EXPECT_PIXEL(32, 8, 235, 0, 0, 255, 40);

    /* CPU reference: NdotL at the vertex (32,16), the tile just above the light */
    {
        float ex = 32.0f - 32.0f, ey = 8.0f - 16.0f, ez = 10.0f;
        float len = sqrtf(ex * ex + ey * ey + ez * ez);
        float ndotl = ez / len;
        EXPECT_PIXEL(32, 16, (int)(ndotl * 255), 0, 0, 255, 12);
        EXPECT_TRUE(ndotl > 0.6f && ndotl <= 1.0f);
    }

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
}

TEST(texenv_modes)
{
    setup_2d(64, 64);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    GLubyte img[2 * 2 * 4] = {
        255, 0, 0, 255,     0, 255, 0, 128,
        0, 255, 0, 128,   255, 0, 0, 255,
    };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glEnable(GL_TEXTURE_2D);

    /* REPLACE: the texture replaces everything */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    EXPECT_PIXEL(8, 8, 255, 0, 0, 255, 8);
    EXPECT_PIXEL(56, 8, 0, 255, 0, 128, 8);

    /* DECAL: rgb from the texture, alpha from the fragment color */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glColor4f(0, 0, 0, 64.0f / 255.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    EXPECT_PIXEL(8, 8, 255, 0, 0, 64, 8);
    /* DECAL blends rgb by the texture alpha: 0.5 * 255 = 128, alpha from the fragment */
    EXPECT_PIXEL(56, 8, 0, 128, 0, 64, 8);

    /* BLEND: rgb mix(fragment, env color, tex rgb); alpha = frag.a * tex.a */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
    glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, (GLfloat[4]){ 0, 0, 1, 1 });
    glColor4f(1, 1, 0, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    /* texel red (1,0,0): rgb = mix(frag, env, tex) = (0, 1, 0); a = 1*1 */
    EXPECT_PIXEL(8, 8, 0, 255, 0, 255, 8);
    /* texel green (0,1,0): rgb = mix(frag, env, tex) = (1, 0, 0); a = 1*0.5 */
    EXPECT_PIXEL(56, 8, 255, 0, 0, 128, 8);

    /* ADD: rgb = frag + tex; alpha = frag * tex */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_ADD);
    glColor4f(0, 0, 0.5, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    EXPECT_PIXEL(8, 8, 255, 0, 128, 255, 8);
    EXPECT_PIXEL(56, 8, 0, 255, 128, 128, 8);

    /* COMBINE: rgb = arg0*arg1 (modulate), alpha = arg0 (replace) */
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_COMBINE);
    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_RGB, GL_MODULATE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_RGB, GL_TEXTURE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SRC1_RGB, GL_PRIMARY_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_RGB, GL_SRC_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND1_RGB, GL_SRC_COLOR);
    glTexEnvi(GL_TEXTURE_ENV, GL_COMBINE_ALPHA, GL_REPLACE);
    glTexEnvi(GL_TEXTURE_ENV, GL_SRC0_ALPHA, GL_TEXTURE);
    glTexEnvi(GL_TEXTURE_ENV, GL_OPERAND0_ALPHA, GL_SRC_ALPHA);
    glColor4f(0.5f, 1.0f, 1.0f, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();
    EXPECT_PIXEL(8, 8, 128, 0, 0, 255, 8);
    EXPECT_PIXEL(56, 8, 0, 255, 0, 128, 8);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

TEST(rasterpos_drawpixels)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLubyte px[4 * 4 * 4];
    for (int i = 0; i < 16; i++)
    {
        px[i * 4 + 0] = 0;
        px[i * 4 + 1] = 0;
        px[i * 4 + 2] = 255;
        px[i * 4 + 3] = 255;
    }

    glRasterPos2i(8, 8);
    glDrawPixels(4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(20, 20, 0, 0, 0, 255, 8);

    /* zoom x2 */
    glClear(GL_COLOR_BUFFER_BIT);
    glPixelZoom(2, 2);
    glRasterPos2i(4, 4);
    glDrawPixels(4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glPixelZoom(1, 1);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(10, 10, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(2, 2, 0, 0, 0, 255, 8);

    /* current raster position queries */
    GLfloat rp[4];
    glGetFloatv(GL_CURRENT_RASTER_POSITION, rp);
    EXPECT_NEAR(rp[0], 4.0, 1e-4);
    GLint valid = 0;
    glGetIntegerv(GL_CURRENT_RASTER_POSITION_VALID, &valid);
    EXPECT_EQ_INT(valid, 1);
}

TEST(bitmap)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* 8x8 pattern: a 4x4 solid block in the lower-left */
    GLubyte bits[8] = { 0xF0, 0xF0, 0xF0, 0xF0, 0, 0, 0, 0 };

    glColor4f(1, 1, 0, 1);
    glRasterPos2i(0, 0);
    glBitmap(8, 8, 0, 0, 0, 0, bits);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(1, 1, 255, 255, 0, 255, 8);
    EXPECT_PIXEL(5, 5, 0, 0, 0, 255, 8);
}

TEST(accum)
{
    setup_2d(64, 64);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* draw red, load into accum */
    glColor4f(1, 0, 0, 1);
    glRectf(0, 0, 32, 32);
    glClearAccum(0, 0, 0, 0);
    glAccum(GL_LOAD, 1.0f);
    EXPECT_NO_GL_ERROR();

    /* clear the color buffer, draw green, accumulate on top */
    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0, 1, 0, 1);
    glRectf(32, 0, 64, 32);
    glAccum(GL_ACCUM, 1.0f);

    /* return the accumulation buffer */
    glClear(GL_COLOR_BUFFER_BIT);
    glAccum(GL_RETURN, 1.0f);

    EXPECT_NO_GL_ERROR();
    /* left half: red; right half: green */
    EXPECT_PIXEL(8, 8, 255, 0, 0, 255, 16);
    EXPECT_PIXEL(56, 8, 0, 255, 0, 255, 16);
    EXPECT_PIXEL(40, 40, 0, 0, 0, 255, 16);
}

TEST(polygon_stipple)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* every other bit in x set: vertical stripes */
    GLubyte mask[32 * 4];
    memset(mask, 0, sizeof mask);
    for (int row = 0; row < 32; row++)
        for (int b = 0; b < 4; b++)
            mask[row * 4 + b] = 0xAA;   /* 10101010 */

    glPolygonStipple(mask);
    glEnable(GL_POLYGON_STIPPLE);

    glColor4f(1, 1, 1, 1);
    glRectf(0, 0, 64, 64);

    glDisable(GL_POLYGON_STIPPLE);
    EXPECT_NO_GL_ERROR();

    /* striped: x % 4 in {0,1} drawn, {2,3} discarded (window space) */
    int drawn = 0, empty = 0;
    for (int x = 0; x < 64; x++)
    {
        GLubyte p[3];
        glReadPixels(x, 32, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, p);
        if (p[0] > 128) drawn++;
        else empty++;
    }
    EXPECT_TRUE(drawn >= 28 && drawn <= 36);
    EXPECT_TRUE(empty >= 28 && empty <= 36);
}

TEST(evaluators)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* a 1D map: straight line along y at x=16 */
    GLfloat pts[2 * 3] = { 16, 4, 0,  16, 60, 0 };
    glColor4f(0, 1, 0, 1);
    glMap1f(GL_MAP1_VERTEX_3, 0.0f, 1.0f, 0, 2, pts);   /* stride 0 = tight (bytes) */
    glEnable(GL_MAP1_VERTEX_3);
    glMapGrid1f(16, 0.0f, 1.0f);
    glEvalMesh1(GL_LINE, 0, 16);
    glDisable(GL_MAP1_VERTEX_3);

    EXPECT_NO_GL_ERROR();
    /* Vertical line at x=16: x=16 is the RIGHT corner of column 15's diamonds
     * and the LEFT corner of column 16's.  GL's diamond-exit rule counts the
     * right corner and not the left one, so exactly column 15 is lit — verified
     * against a CORE context drawing the same line (identical pixels). */
    {
        GLubyte a[4] = { 0, 0, 0, 0 }, b[4] = { 0, 0, 0, 0 };
        glReadPixels(15, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);
        glReadPixels(16, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);
        EXPECT_TRUE(a[1] > 200);
        EXPECT_TRUE(a[0] == 0 && a[2] == 0);
        EXPECT_TRUE(b[1] == 0);
    }
    EXPECT_PIXEL(32, 32, 0, 0, 0, 255, 8);

    /* 2D evaluator covering the quad */
    glClear(GL_COLOR_BUFFER_BIT);
    GLfloat q[3 * 4] = {
        4, 4, 0,   60, 4, 0,
        4, 60, 0,  60, 60, 0,
    };
    glColor4f(1, 0, 1, 1);
    glMap2f(GL_MAP2_VERTEX_3, 0, 1, 0, 2, 0, 1, 0, 2, q); /* strides 0 = tight (bytes) */
    glEnable(GL_MAP2_VERTEX_3);
    glMapGrid2f(4, 0, 1, 4, 0, 1);
    glEvalMesh2(GL_FILL, 0, 4, 0, 4);
    glDisable(GL_MAP2_VERTEX_3);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 255, 0, 255, 255, 16);
    EXPECT_PIXEL(1, 1, 0, 0, 0, 255, 4);
}

TEST(select_mode)
{
    setup_2d(64, 64);

    GLuint sel[64];
    glSelectBuffer(64, sel);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(1);

    /* one quad in the middle */
    glColor4f(1, 1, 1, 1);
    glRectf(16, 16, 48, 48);

    glPopName();
    GLint hits = glRenderMode(GL_RENDER);

    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    /* record: name count, zmin, zmax, name */
    EXPECT_TRUE(hits == 1);
    if (hits == 1)
    {
        EXPECT_EQ_INT((int)sel[0], 1);      /* one name on the stack */
        EXPECT_TRUE(sel[3] == 1);           /* the pushed name */
    }

    /* a quad outside the view volume produces no hit */
    GLuint sel2[64];
    glSelectBuffer(64, sel2);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(7);
    glRectf(200, 200, 300, 300);
    glPopName();
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, 0);
}

TEST(feedback_mode)
{
    setup_2d(64, 64);

    GLfloat fb[256];
    glFeedbackBuffer(256, GL_3D, fb);
    glRenderMode(GL_FEEDBACK);

    glColor4f(1, 0, 0, 1);
    glBegin(GL_TRIANGLES);
    glVertex2f(8, 8);
    glVertex2f(56, 8);
    glVertex2f(8, 56);
    glEnd();

    GLint count = glRenderMode(GL_RENDER);

    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    /* GL_POLYGON_TOKEN(3D: 3 floats each) */
    EXPECT_TRUE(count >= 1 + 9);
    if (count >= 10)
    {
        EXPECT_EQ_INT((int)fb[0], GL_POLYGON_TOKEN);
        EXPECT_EQ_INT((int)fb[1], 3);
        /* first vertex window position: x=8, y=8 */
        EXPECT_NEAR(fb[2], 8.0, 0.5);
        EXPECT_NEAR(fb[3], 8.0, 0.5);
    }
}

/* specular exponent: the scalar uniform mglMatShininess[0] (float array next
 * to vec4 arrays in the generated shader) must reach the GPU.  The vertex at
 * (32.5,32.5) with the light at (32.5,36.5,10):
 *   VP = (0,.3714,.9285), H = normalize(VP+(0,0,1)) -> NdH = .9819
 * pow(.9819,128)=.0965, pow(.9819,16)=..7458, pow(.9819,2)=.9641 */
static float specular_at_center(float shininess)
{
    setup_2d(64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat[4]){ 32.5f, 36.5f, 10, 1 });
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat[4]){ 0, 0, 0, 1 });
    glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat[4]){ 1, 1, 1, 1 });

    glMaterialfv(GL_FRONT, GL_SPECULAR, (GLfloat[4]){ 1, 1, 1, 1 });
    glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat[4]){ 0, 0, 0, 1 });
    glMaterialf(GL_FRONT, GL_SHININESS, shininess);

    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    for (int ty = 0; ty < 16; ty++)
        for (int tx = 0; tx < 16; tx++)
        {
            float x0 = tx * 4.0f + 0.5f, y0 = ty * 4.0f + 0.5f;
            glVertex3f(x0, y0, 0);
            glVertex3f(x0 + 4, y0, 0);
            glVertex3f(x0 + 4, y0 + 4, 0);
            glVertex3f(x0, y0 + 4, 0);
        }
    glEnd();

    GLubyte px[4];
    mgl_read_rgba(32, 32, px);   /* pixel center = the vertex under the light */
    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    return px[0] / 255.0f;
}

TEST(specular_shininess)
{
    EXPECT_NEAR(specular_at_center(128.0f), 0.0965f, 0.02f);
    EXPECT_NEAR(specular_at_center(16.0f), 0.7458f, 0.03f);
    EXPECT_NEAR(specular_at_center(2.0f), 0.9641f, 0.02f);
    EXPECT_NO_GL_ERROR();
}

/* GL line rasterization (diamond-exit rule): an exact vertical/horizontal
 * line on a pixel boundary lights the pixel whose RIGHT / TOP corner the line
 * touches (column 15 for x=16, row 15 for y=16); a line on a pixel center
 * (16.5) lights that column/row only.  Verified against a CORE context
 * drawing the same segments through a VBO (identical coverage). */
TEST(line_diamond_rule)
{
    setup_2d(64, 64);
    glColor4f(0, 1, 0, 1);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(16, 4); glVertex2f(16, 60); glEnd();
    EXPECT_PIXEL(15, 32, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(16, 32, 0, 0, 0, 255, 2);
    EXPECT_PIXEL(15, 4, 0, 255, 0, 255, 2);    /* first/last diamond corners */
    EXPECT_PIXEL(15, 59, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(15, 3, 0, 0, 0, 255, 2);      /* just outside the segment */
    EXPECT_PIXEL(15, 60, 0, 0, 0, 255, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(4, 16); glVertex2f(60, 16); glEnd();
    EXPECT_PIXEL(32, 15, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(32, 16, 0, 0, 0, 255, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(16.5f, 4); glVertex2f(16.5f, 60); glEnd();
    EXPECT_PIXEL(16, 32, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(15, 32, 0, 0, 0, 255, 2);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_LINES); glVertex2f(4, 16.5f); glVertex2f(60, 16.5f); glEnd();
    EXPECT_PIXEL(32, 16, 0, 255, 0, 255, 2);
    EXPECT_PIXEL(32, 15, 0, 0, 0, 255, 2);

    EXPECT_NO_GL_ERROR();
}

/* feedback must record per-vertex color / texcoords of client arrays, in
 * draw order (also for indexed draws, which revisit vertices) */
TEST(feedback_per_vertex_data)
{
    setup_2d(64, 64);

    static const float pos[4][4] = {
        { 4, 4, 0, 1 }, { 30, 4, 0, 1 }, { 30, 20, 0, 1 }, { 4, 20, 0, 1 },
    };
    static const GLubyte col[4][4] = {
        { 255, 0, 0, 255 }, { 0, 255, 0, 255 }, { 0, 0, 255, 255 }, { 128, 128, 0, 255 },
    };
    static const float tex[4][4] = {
        { 0, 0, 0, 1 }, { 1, 0, 0, 1 }, { 1, 1, 0, 1 }, { 0, 1, 0, 1 },
    };
    static const GLushort idx[3] = { 3, 2, 1 };     /* reversed draw order */

    GLfloat fb[128];
    glFeedbackBuffer(128, GL_3D_COLOR_TEXTURE, fb);
    glRenderMode(GL_FEEDBACK);

    glVertexPointer(4, GL_FLOAT, 0, pos);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, col);
    glTexCoordPointer(4, GL_FLOAT, 0, tex);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, idx);

    GLint count = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    /* POLYGON_TOKEN n x11 (3 coords + color + tex) */
    EXPECT_TRUE(count == 2 + 3 * 11);
    if (count >= 13)
    {
        EXPECT_EQ_INT((int)fb[0], GL_POLYGON_TOKEN);
        EXPECT_EQ_INT((int)fb[1], 3);

        /* first recorded vertex is index 3: (4,20), color (128,128,0,255),
         * texcoord (0,1,0,1) */
        EXPECT_NEAR(fb[2], 4.0, 0.5);
        EXPECT_NEAR(fb[3], 20.0, 0.5);
        EXPECT_NEAR(fb[5], 128.0f / 255.0f, 0.01);
        EXPECT_NEAR(fb[6], 128.0f / 255.0f, 0.01);
        EXPECT_NEAR(fb[7], 0.0, 0.01);
        EXPECT_NEAR(fb[8], 1.0, 0.01);
        EXPECT_NEAR(fb[9], 0.0, 0.01);
        EXPECT_NEAR(fb[10], 1.0, 0.01);

        /* second vertex is index 2: (30,20), color (0,0,255,255), tex (1,1,0,1) */
        EXPECT_NEAR(fb[13], 30.0, 0.5);
        EXPECT_NEAR(fb[14], 20.0, 0.5);
        EXPECT_NEAR(fb[16], 0.0, 0.01);
        EXPECT_NEAR(fb[17], 0.0, 0.01);
        EXPECT_NEAR(fb[18], 1.0, 0.01);
        EXPECT_NEAR(fb[20], 1.0, 0.01);
        EXPECT_NEAR(fb[21], 1.0, 0.01);
    }
}

/* without a color/texcoord array the current color / texcoord are recorded */
TEST(feedback_current_values)
{
    setup_2d(64, 64);

    static const float pos[3][4] = {
        { 8, 8, 0, 1 }, { 40, 8, 0, 1 }, { 8, 40, 0, 1 },
    };

    GLfloat fb[128];
    glFeedbackBuffer(128, GL_3D_COLOR_TEXTURE, fb);
    glRenderMode(GL_FEEDBACK);

    glVertexPointer(4, GL_FLOAT, 0, pos);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor4f(0.25f, 0.5f, 0.75f, 1.0f);
    glTexCoord4f(0.1f, 0.2f, 0.3f, 1.0f);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    GLint count = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDisableClientState(GL_VERTEX_ARRAY);

    EXPECT_TRUE(count == 2 + 3 * 11);
    if (count >= 13)
    {
        EXPECT_NEAR(fb[5], 0.25, 0.01);
        EXPECT_NEAR(fb[6], 0.5, 0.01);
        EXPECT_NEAR(fb[7], 0.75, 0.01);
        EXPECT_NEAR(fb[8], 1.0, 0.01);
        EXPECT_NEAR(fb[9], 0.1, 0.01);
        EXPECT_NEAR(fb[10], 0.2, 0.01);
    }
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    RUN(matrix_transforms);
    RUN(positional_light_and_colormaterial);
    RUN(specular_shininess);
    RUN(texenv_modes);
    RUN(rasterpos_drawpixels);
    RUN(bitmap);
    RUN(accum);
    RUN(polygon_stipple);
    RUN(evaluators);
    RUN(line_diamond_rule);
    RUN(select_mode);
    RUN(feedback_mode);
    RUN(feedback_per_vertex_data);
    RUN(feedback_current_values);

    return mgl_test_summary();
}
