/*
 * test_compat_pipeline.c — fixed-function rendering through the compat domain:
 * immediate mode, the generated fixed-function vertex/fragment pipeline,
 * quads/polygons, flat shading, lighting, texturing (texenv), alpha test,
 * fog, clip planes, client arrays with VAO 0, display lists, glRectf and
 * compatibility GLSL.
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
}

TEST(immediate_triangle)
{
    setup_2d(64, 64);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor4f(1, 0, 0, 1);
    glBegin(GL_TRIANGLES);
    glVertex2f(8, 8);
    glVertex2f(56, 8);
    glVertex2f(8, 56);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 10, 255, 0, 0, 255, 8);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 8);      /* outside */
    EXPECT_PIXEL(30, 30, 255, 0, 0, 255, 40);   /* inside (barycentric) */
}

TEST(quad_and_polygon)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor4f(0, 1, 0, 1);
    glBegin(GL_QUADS);
    glVertex2f(10, 10);
    glVertex2f(40, 10);
    glVertex2f(40, 40);
    glVertex2f(10, 40);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(25, 25, 0, 255, 0, 255, 8);
    EXPECT_PIXEL(5, 5, 0, 0, 0, 255, 8);
    EXPECT_PIXEL(45, 45, 0, 0, 0, 255, 8);

    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0, 0, 1, 1);
    glBegin(GL_POLYGON);
    glVertex2f(32, 8);
    glVertex2f(56, 32);
    glVertex2f(32, 56);
    glVertex2f(8, 32);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 0, 0, 255, 255, 8);
    EXPECT_PIXEL(8, 8, 0, 0, 0, 255, 8);
}

TEST(smooth_and_flat_shading)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* smooth: interpolated color across the triangle */
    glShadeModel(GL_SMOOTH);
    glBegin(GL_TRIANGLES);
    glColor4f(1, 0, 0, 1); glVertex2f(4, 4);
    glColor4f(0, 1, 0, 1); glVertex2f(60, 4);
    glColor4f(0, 0, 1, 1); glVertex2f(4, 60);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 255, 0, 0, 255, 70);
    EXPECT_PIXEL(52, 8, 0, 255, 0, 255, 90);
    EXPECT_PIXEL(8, 52, 0, 0, 255, 255, 90);

    /* flat: all fragments take the provoking (last) vertex color */
    glClear(GL_COLOR_BUFFER_BIT);
    glShadeModel(GL_FLAT);
    glBegin(GL_TRIANGLES);
    glColor4f(1, 0, 0, 1); glVertex2f(4, 4);
    glColor4f(0, 1, 0, 1); glVertex2f(60, 4);
    glColor4f(0, 0, 1, 1); glVertex2f(4, 60);
    glEnd();

    EXPECT_PIXEL(8, 8, 0, 0, 255, 255, 12);
    EXPECT_PIXEL(40, 8, 0, 0, 255, 255, 12);
    glShadeModel(GL_SMOOTH);
}

TEST(texenv_modulate)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* 2x2 texture: red/green checkerboard */
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    GLubyte img[2 * 2 * 4] = {
        255, 0, 0, 255,   0, 255, 0, 255,
        0, 255, 0, 255, 255, 0, 0, 255,
    };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    EXPECT_NO_GL_ERROR();

    glEnable(GL_TEXTURE_2D);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();

    EXPECT_NO_GL_ERROR();
    /* texel (0,0) = red in the bottom-left quadrant */
    EXPECT_PIXEL(8, 8, 255, 0, 0, 255, 8);
    /* texel (1,0) = green */
    EXPECT_PIXEL(56, 8, 0, 255, 0, 255, 8);

    /* modulate with a dark vertex color */
    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(0.5, 0.5, 0.5, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0, 0);
    glTexCoord2f(1, 0); glVertex2f(64, 0);
    glTexCoord2f(1, 1); glVertex2f(64, 64);
    glTexCoord2f(0, 1); glVertex2f(0, 64);
    glEnd();

    EXPECT_PIXEL(8, 8, 128, 0, 0, 255, 12);

    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &tex);
    EXPECT_NO_GL_ERROR();
}

TEST(alpha_test)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.5f);

    glBegin(GL_TRIANGLES);
    glColor4f(1, 1, 1, 1.0f); glVertex2f(0, 0);
    glColor4f(1, 1, 1, 0.0f); glVertex2f(64, 0);
    glColor4f(1, 1, 1, 1.0f); glVertex2f(0, 64);
    glEnd();

    EXPECT_NO_GL_ERROR();
    /* alpha 1 at the corner -> kept; alpha 0 at (64,0) corner -> discarded */
    EXPECT_PIXEL(2, 2, 255, 255, 255, 230, 30);
    EXPECT_PIXEL(60, 2, 0, 0, 0, 255, 30);

    glDisable(GL_ALPHA_TEST);
}

TEST(fog_linear)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_FOG);
    glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_START, 0.0f);
    glFogf(GL_FOG_END, 1000.0f);   /* everything well inside: fully fogged */
    glFogfv(GL_FOG_COLOR, (GLfloat[4]){ 0, 0, 1, 1 });

    glColor4f(1, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex3f(0, 0, 0);
    glVertex3f(64, 0, 0);
    glVertex3f(64, 64, 0);
    glVertex3f(0, 64, 0);
    glEnd();

    EXPECT_NO_GL_ERROR();
    /* eye distance ~45 << end 1000 -> fog factor 1.0 -> pure fog color */
    EXPECT_PIXEL(32, 32, 8, 0, 247, 255, 10);

    /* short fog: everything beyond the end -> no fog at all */
    glClear(GL_COLOR_BUFFER_BIT);
    glFogf(GL_FOG_END, 4.0f);
    glBegin(GL_QUADS);
    glVertex3f(0, 0, 0);
    glVertex3f(64, 0, 0);
    glVertex3f(64, 64, 0);
    glVertex3f(0, 64, 0);
    glEnd();
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 6);

    glDisable(GL_FOG);
}

TEST(clip_plane)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* keep x <= 32: dot(plane, eye) = -x + 32 >= 0 */
    const GLdouble plane[4] = { -1, 0, 0, 32 };
    glClipPlane(GL_CLIP_PLANE0, plane);
    glEnable(GL_CLIP_PLANE0);

    glColor4f(1, 1, 0, 1);
    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glVertex2f(64, 0);
    glVertex2f(64, 64);
    glVertex2f(0, 64);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 32, 255, 255, 0, 255, 8);  /* kept (x <= 32) */
    EXPECT_PIXEL(50, 32, 0, 0, 0, 255, 8);      /* clipped */

    glDisable(GL_CLIP_PLANE0);
}

TEST(lighting_directional)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_LIGHTING);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat[4]){ 0, 0, 1, 0 });
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat[4]){ 1, 1, 1, 1 });

    glMaterialfv(GL_FRONT, GL_DIFFUSE, (GLfloat[4]){ 1, 0, 0, 1 });
    glMaterialfv(GL_FRONT, GL_AMBIENT, (GLfloat[4]){ 0, 0, 0, 1 });

    /* flat quad facing the light: fully lit red */
    glBegin(GL_QUADS);
    glNormal3f(0, 0, 1);
    glVertex3f(8, 8, 0);
    glVertex3f(56, 8, 0);
    glVertex3f(56, 56, 0);
    glVertex3f(8, 56, 0);
    glEnd();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 16);

    /* now tilt the normal away: diffuse drops to 0 */
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
    glNormal3f(1, 0, 0);
    glVertex3f(8, 8, 0);
    glVertex3f(56, 8, 0);
    glVertex3f(56, 56, 0);
    glVertex3f(8, 56, 0);
    glEnd();

    EXPECT_PIXEL(32, 32, 0, 0, 0, 255, 16);

    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
}

TEST(client_arrays_vao0)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    static const GLfloat verts[3 * 3] = {
        8, 8, 0,
        56, 8, 0,
        8, 56, 0,
    };
    static const GLfloat colors[3 * 4] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    };

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, verts);
    glColorPointer(4, GL_FLOAT, 0, colors);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(10, 10, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(50, 50, 0, 0, 0, 255, 8);

    /* VAO 0 is the default vertex array: it must report as bound */
    GLint vao = 12345;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
    EXPECT_EQ_INT(vao, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

TEST(client_indexed_draw)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    static const GLfloat verts[4 * 2] = { 8, 8, 56, 8, 56, 56, 8, 56 };
    static const GLubyte idx[6] = { 0, 1, 2, 0, 2, 3 };

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, verts);

    glColor4f(0, 1, 1, 1);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, idx);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 0, 255, 255, 255, 8);
    EXPECT_PIXEL(2, 2, 0, 0, 0, 255, 8);

    glDisableClientState(GL_VERTEX_ARRAY);
}

TEST(display_list_replay)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint list = glGenLists(1);
    EXPECT_TRUE(list != 0);

    glNewList(list, GL_COMPILE);
    glColor4f(1, 0, 1, 1);
    glBegin(GL_QUADS);
    glVertex2f(8, 8);
    glVertex2f(56, 8);
    glVertex2f(56, 56);
    glVertex2f(8, 56);
    glEnd();
    glEndList();

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 0, 0, 0, 255, 8);      /* not drawn while compiling */

    glCallList(list);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 255, 0, 255, 255, 8);

    /* the list replays the recorded state too */
    glClear(GL_COLOR_BUFFER_BIT);
    glColor4f(1, 1, 1, 1);
    glCallList(list);
    EXPECT_PIXEL(32, 32, 255, 0, 255, 255, 8);

    glDeleteLists(list, 1);
}

TEST(push_pop_attrib)
{
    setup_2d(64, 64);

    glColor4f(1, 0, 0, 1);
    glPushAttrib(GL_CURRENT_BIT | GL_TRANSFORM_BIT);
    glColor4f(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslated(10, 0, 0);
    glPopAttrib();

    GLfloat v[4];
    glGetFloatv(GL_CURRENT_COLOR, v);
    EXPECT_NEAR(v[0], 1.0, 1e-6);   /* GL_CURRENT_BIT restores the pushed color */

    EXPECT_NO_GL_ERROR();

    GLint depth = 0;
    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);        /* GL_TRANSFORM_BIT popped the matrix too */
}

TEST(rect)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor4f(1, 1, 1, 1);
    glRectf(10, 10, 50, 50);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(30, 30, 255, 255, 255, 255, 8);
    EXPECT_PIXEL(4, 4, 0, 0, 0, 255, 8);
}

TEST(compat_glsl_shader)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    /* #version 120 style fixed-function-input shader */
    const char *vs =
        "#version 120\n"
        "varying vec4 col;\n"
        "void main() {\n"
        "    gl_Position = ftransform();\n"
        "    col = gl_Color;\n"
        "    gl_TexCoord[0] = vec4(0.5, 0.5, 0.0, 1.0);\n"
        "}\n";
    const char *fs =
        "#version 120\n"
        "varying vec4 col;\n"
        "void main() {\n"
        "    gl_FragColor = col;\n"
        "}\n";

    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL);
    glCompileShader(v);
    GLint ok = 0;
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = { 0 };
        glGetShaderInfoLog(v, sizeof log - 1, NULL, log);
        printf("VS log:\n%s\n", log);
    }
    EXPECT_EQ_INT(ok, 1);
    glShaderSource(f, 1, &fs, NULL);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = { 0 };
        glGetShaderInfoLog(f, sizeof log - 1, NULL, log);
        printf("FS log:\n%s\n", log);
    }
    EXPECT_EQ_INT(ok, 1);

    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);

    if (!ok)
        return;

    glUseProgram(p);

    static const GLfloat verts[3 * 3] = { 8, 8, 0, 56, 8, 0, 8, 56, 0 };
    static const GLfloat colors[3 * 4] = { 1, 0.5f, 0.25f, 1, 1, 0.5f, 0.25f, 1, 1, 0.5f, 0.25f, 1 };

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, verts);
    glColorPointer(4, GL_FLOAT, 0, colors);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(12, 12, 255, 128, 64, 255, 16);

    glUseProgram(0);
    glDeleteProgram(p);
}

int main(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    if (!compat_ctx)
        return 1;

    RUN(immediate_triangle);
    RUN(quad_and_polygon);
    RUN(smooth_and_flat_shading);
    RUN(texenv_modulate);
    RUN(alpha_test);
    RUN(fog_linear);
    RUN(clip_plane);
    RUN(lighting_directional);
    RUN(client_arrays_vao0);
    RUN(client_indexed_draw);
    RUN(display_list_replay);
    RUN(push_pop_attrib);
    RUN(rect);
    RUN(compat_glsl_shader);

    return mgl_test_summary();
}
