/*
 * test_compat_edge.c — compatibility-profile spec corner cases that legacy
 * apps lean on: matrix-stack limits and per-unit texture stacks, the
 * transpose matrix entry points, immediate-mode error behaviour, display
 * list query state + nesting limit + list base, and the (client) attribute
 * stack limits.  Everything here is also a regression test for the GLX
 * shim's compat-context creation (profile mask, not version string).
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void *compat_ctx;

/* ------------------------------------------------------------------ */
/* matrix stacks                                                       */

TEST(matrix_stack_depths_and_limits)
{
    GLint depth = -1, maxd = -1;

    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    EXPECT_NO_GL_ERROR();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);                    /* bottom counts as one */
    glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH, &maxd);
    EXPECT_TRUE(maxd >= 32);                    /* spec minimum */

    /* fill the stack exactly to the reported depth */
    for (int i = 1; i < maxd; i++)
        glPushMatrix();
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, maxd);

    /* one more push must overflow and change nothing */
    glPushMatrix();
    EXPECT_GL_ERROR(GL_STACK_OVERFLOW);
    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, maxd);

    /* drain again: pop back to depth 1, then underflow */
    for (int i = 1; i < maxd; i++)
        glPopMatrix();
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);
    glPopMatrix();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);
    glGetIntegerv(GL_MODELVIEW_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);

    /* the projection and texture stacks have their own counters */
    glGetIntegerv(GL_PROJECTION_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);
    glGetIntegerv(GL_TEXTURE_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 1);
    EXPECT_NO_GL_ERROR();
}

TEST(matrix_mode_errors_and_persistence)
{
    GLint mode = 0;

    glMatrixMode(GL_MODELVIEW);
    glMatrixMode(0xBEEF);                       /* invalid */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetIntegerv(GL_MATRIX_MODE, &mode);
    EXPECT_EQ_INT(mode, GL_MODELVIEW);          /* unchanged */

    /* matrix ops on a valid but unloaded stack still work; identity resets */
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glTranslatef(1, 2, 3);
    GLfloat m[16];
    glGetFloatv(GL_TEXTURE_MATRIX, m);
    EXPECT_NEAR(m[12], 1, 1e-6);
    EXPECT_NEAR(m[13], 2, 1e-6);
    EXPECT_NEAR(m[14], 3, 1e-6);

    /* pop below the bottom of the texture stack */
    glPopMatrix();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);

    glMatrixMode(GL_MODELVIEW);
    EXPECT_NO_GL_ERROR();
}

TEST(per_unit_texture_stacks)
{
    GLfloat m0[16], m1[16];

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glActiveTexture(GL_TEXTURE1);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();

    /* push + load a distinctive matrix on unit 1 only */
    glActiveTexture(GL_TEXTURE1);
    glPushMatrix();
    glScalef(2, 2, 2);

    glActiveTexture(GL_TEXTURE0);
    glGetFloatv(GL_TEXTURE_MATRIX, m0);
    glActiveTexture(GL_TEXTURE1);
    glGetFloatv(GL_TEXTURE_MATRIX, m1);
    EXPECT_NEAR(m0[0], 1, 1e-6);
    EXPECT_NEAR(m1[0], 2, 1e-6);

    /* popping on unit 0 must not touch unit 1's stack */
    glActiveTexture(GL_TEXTURE0);
    glPopMatrix();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);        /* unit 0 stack is empty */

    glActiveTexture(GL_TEXTURE1);
    glPopMatrix();                              /* the push from above */
    EXPECT_NO_GL_ERROR();
    glGetFloatv(GL_TEXTURE_MATRIX, m1);
    EXPECT_NEAR(m1[0], 1, 1e-6);

    glActiveTexture(GL_TEXTURE0);
    glMatrixMode(GL_MODELVIEW);
    EXPECT_NO_GL_ERROR();
}

TEST(transpose_matrix_entry_points)
{
    /* row-major input to the *Transpose entry points ==
     * column-major input of the transposed matrix */
    const GLfloat rowmajor[16] = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    GLfloat colmajor[16], out[16];

    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++)
            colmajor[c * 4 + r] = rowmajor[r * 4 + c];

    glMatrixMode(GL_MODELVIEW);
    glLoadTransposeMatrixf(rowmajor);
    glGetFloatv(GL_MODELVIEW_MATRIX, out);
    EXPECT_NO_GL_ERROR();
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(out[i], colmajor[i], 1e-6);

    glLoadIdentity();
    glLoadMatrixf(colmajor);
    GLfloat a[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, a);

    glLoadIdentity();
    glMultTransposeMatrixf(rowmajor);
    glGetFloatv(GL_MODELVIEW_MATRIX, out);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(out[i], a[i], 1e-6);        /* ident * M == M */

    glLoadIdentity();
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* immediate mode                                                       */

TEST(immediate_mode_error_behaviour)
{
    GLint bad = 0;

    glEnd();                                    /* no glBegin */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBegin(GL_QUADS);
    glBegin(GL_TRIANGLES);                      /* nested begin */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glEnd();
    EXPECT_NO_GL_ERROR();

    glBegin(0x1000);                            /* not a primitive */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glEnd();                                    /* begin never opened (Mesa parity) */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* vertex data commands outside begin/end set current state and are
     * NOT errors (glVertex itself is ignored) */
    glColor3f(1, 0, 0);
    glNormal3f(0, 0, 1);
    glVertex3f(1, 2, 3);
    EXPECT_NO_GL_ERROR();

    GLfloat c[4];
    glGetFloatv(GL_CURRENT_COLOR, c);
    EXPECT_NEAR(c[0], 1, 1e-6);
    EXPECT_NEAR(c[1], 0, 1e-6);

    /* degenerate: begin+end with no vertices draws nothing, no error */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 16, 0, 16, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
    glEnd();
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 0, 0, 0, 255, 4);

    (void)bad;
}

TEST(glRect_pixel_check)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 32, 0, 32, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 32, 32);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0, 1, 0);
    glRectf(4, 4, 12, 12);                      /* x1<y2 order must not matter */
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 0, 255, 0, 255, 6);
    EXPECT_PIXEL(2, 2, 0, 0, 0, 255, 4);

    glClear(GL_COLOR_BUFFER_BIT);
    glRectf(12, 12, 4, 4);                      /* reversed corners */
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 0, 255, 0, 255, 6);
    EXPECT_PIXEL(14, 14, 0, 0, 0, 255, 4);
}

/* ------------------------------------------------------------------ */
/* display lists                                                        */

static int list_render_probe(GLuint list, int x, int y)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glCallList(list);
    GLubyte p[4];
    mgl_read_rgba(x, y, p);
    return p[0] | p[1] << 8 | p[2] << 16;
}

TEST(display_list_query_state)
{
    GLint mode = 7, index = -1;

    glNewList(5, GL_COMPILE);
    glGetIntegerv(GL_LIST_MODE, &mode);
    EXPECT_EQ_INT(mode, GL_COMPILE);
    glGetIntegerv(GL_LIST_INDEX, &index);
    EXPECT_EQ_INT(index, 5);
    glEndList();
    EXPECT_NO_GL_ERROR();

    glGetIntegerv(GL_LIST_MODE, &mode);
    EXPECT_EQ_INT(mode, 0);                     /* 0 outside recording */
    glGetIntegerv(GL_LIST_INDEX, &index);
    EXPECT_EQ_INT(index, 0);

    /* new list while recording: error, first list keeps recording */
    glNewList(6, GL_COMPILE);
    glNewList(7, GL_COMPILE);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetIntegerv(GL_LIST_INDEX, &index);
    EXPECT_EQ_INT(index, 6);
    glEndList();
    EXPECT_NO_GL_ERROR();

    /* glEndList with nothing open */
    glEndList();
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    EXPECT_EQ_INT(glIsList(6), GL_TRUE);
    EXPECT_EQ_INT(glIsList(7), GL_FALSE);
    EXPECT_EQ_INT(glIsList(0), GL_FALSE);

    /* list 0 and undefined lists are rejected/ignored */
    glNewList(0, GL_COMPILE);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glCallList(9999);
    EXPECT_NO_GL_ERROR();

    EXPECT_EQ_INT(glGenLists(0), 0);
    EXPECT_NO_GL_ERROR();
    glDeleteLists(1, 0);
    EXPECT_NO_GL_ERROR();
}

TEST(display_list_nesting_limit)
{
    GLint maxnest = 0;
    glGetIntegerv(GL_MAX_LIST_NESTING, &maxnest);
    EXPECT_TRUE(maxnest >= 64);                 /* spec minimum */

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 8, 0, 8, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 8, 8);

    /* a call chain maxnest deep (list i calls list i+1); the deepest list
     * draws.  Building it must not error and calling the head must render. */
    GLuint base = glGenLists(maxnest + 1);
    EXPECT_TRUE(base != 0);
    for (int i = 0; i < maxnest; i++)
    {
        glNewList(base + i, GL_COMPILE);
        if (i == maxnest - 1)
        {
            glColor3f(1, 0, 0);
            glBegin(GL_QUADS);
            glVertex2f(0, 0); glVertex2f(8, 0);
            glVertex2f(8, 8); glVertex2f(0, 8);
            glEnd();
        }
        else
            glCallList(base + i + 1);
        glEndList();
    }
    EXPECT_NO_GL_ERROR();

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glCallList(base);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 8);

    /* one level deeper than the limit: the innermost call is dropped
     * (silently, like Mesa) rather than corrupting state */
    GLuint over = glGenLists(maxnest + 1);
    EXPECT_TRUE(over != 0);
    for (int i = 0; i < maxnest + 1; i++)
    {
        glNewList(over + i, GL_COMPILE);
        if (i == maxnest)
        {
            glColor3f(0, 0, 1);
            glBegin(GL_QUADS);
            glVertex2f(0, 0); glVertex2f(8, 0);
            glVertex2f(8, 8); glVertex2f(0, 8);
            glEnd();
        }
        else
            glCallList(over + i + 1);
        glEndList();
    }
    EXPECT_NO_GL_ERROR();
    glClear(GL_COLOR_BUFFER_BIT);
    glCallList(over);                           /* deepest level dropped */
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(4, 4, 0, 0, 0, 255, 8);

    glDeleteLists(base, maxnest + 1);
    glDeleteLists(over, maxnest + 1);
    EXPECT_NO_GL_ERROR();
}

TEST(display_list_list_base_and_calllists)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 16, 0, 16, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 16, 16);

    GLuint base = glGenLists(3);
    EXPECT_TRUE(base != 0);

    /* lists draw squares of increasing height, keyed by colour */
    for (int i = 0; i < 3; i++)
    {
        glNewList(base + i, GL_COMPILE);
        glColor3f(1, (float)i / 4, 0);
        glBegin(GL_QUADS);
        glVertex2f(0, 0); glVertex2f(16, 0);
        glVertex2f(16, 16); glVertex2f(0, 16);
        glEnd();
        glEndList();
    }

    GLint b = 0;
    glGetIntegerv(GL_LIST_BASE, &b);
    EXPECT_EQ_INT(b, 0);

    glListBase(base);
    glGetIntegerv(GL_LIST_BASE, &b);
    EXPECT_EQ_INT(b, (int)base);

    /* call the middle list through glCallLists (offsets are base-relative) */
    GLubyte idx = 1;
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glCallLists(1, GL_UNSIGNED_BYTE, &idx);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(8, 8, 255, 64, 0, 255, 6);

    /* negative n */
    glCallLists(-1, GL_UNSIGNED_BYTE, &idx);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteLists(base, 3);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* attribute stacks                                                     */

TEST(attrib_stack_limits)
{
    GLint depth = -1, maxd = -1;

    glGetIntegerv(GL_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 0);
    glGetIntegerv(GL_MAX_ATTRIB_STACK_DEPTH, &maxd);
    EXPECT_TRUE(maxd >= 16);

    for (int i = 0; i < maxd; i++)
        glPushAttrib(GL_LIGHTING_BIT | GL_TRANSFORM_BIT);
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, maxd);

    glPushAttrib(GL_CURRENT_BIT);
    EXPECT_GL_ERROR(GL_STACK_OVERFLOW);

    for (int i = 0; i < maxd; i++)
        glPopAttrib();
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 0);

    glPopAttrib();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);
}

TEST(attrib_stack_restores_state)
{
    /* a known pre-push colour so the restore expectation is exact */
    glColor3f(1, 0.25f, 0.5f);
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT | GL_TRANSFORM_BIT);

    GLfloat amb[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb);
    glColor3f(1, 0, 1);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glTranslatef(1, 2, 3);
    glEnable(GL_LIGHTING);
    glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 30);

    glPopAttrib();
    EXPECT_NO_GL_ERROR();

    GLfloat q[4] = { 9, 9, 9, 9 };
    glGetFloatv(GL_LIGHT_MODEL_AMBIENT, q);
    EXPECT_NEAR(q[0], 0.2f, 1e-6);              /* GL default ambient = (0.2,0.2,0.2,1) */
    EXPECT_NEAR(q[1], 0.2f, 1e-6);
    EXPECT_NEAR(q[2], 0.2f, 1e-6);

    GLfloat c[4];
    glGetFloatv(GL_CURRENT_COLOR, c);
    EXPECT_NEAR(c[0], 1, 1e-6);
    EXPECT_NEAR(c[1], 0.25f, 1e-6);
    EXPECT_NEAR(c[2], 0.5f, 1e-6);

    EXPECT_EQ_INT(glIsEnabled(GL_LIGHTING), GL_FALSE);

    GLfloat cutoff = -1;
    glGetLightfv(GL_LIGHT0, GL_SPOT_CUTOFF, &cutoff);
    EXPECT_NEAR(cutoff, 180, 1e-6);             /* default spot cutoff */

    /* the projection push/pop pair above is inside the attrib push, so the
     * stack must be back at its base depth */
    GLint d = -1;
    glGetIntegerv(GL_PROJECTION_STACK_DEPTH, &d);
    EXPECT_EQ_INT(d, 1);
    EXPECT_NO_GL_ERROR();
}

TEST(client_attrib_stack_limits)
{
    GLint depth = -1, maxd = -1;

    glGetIntegerv(GL_CLIENT_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 0);
    glGetIntegerv(GL_MAX_CLIENT_ATTRIB_STACK_DEPTH, &maxd);
    EXPECT_TRUE(maxd >= 16);

    for (int i = 0; i < maxd; i++)
        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_CLIENT_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, maxd);

    glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
    EXPECT_GL_ERROR(GL_STACK_OVERFLOW);

    for (int i = 0; i < maxd; i++)
        glPopClientAttrib();
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_CLIENT_ATTRIB_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 0);

    glPopClientAttrib();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);
}

TEST(client_attrib_stack_restores_arrays)
{
    static GLfloat verts[3 * 3] = { 0 };
    static GLubyte colors[3 * 3] = { 0 };

    glDisableClientState(GL_COLOR_ARRAY);
    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors);
    glVertexPointer(3, GL_FLOAT, 0, verts);
    glEnableClientState(GL_VERTEX_ARRAY);

    glPopClientAttrib();
    EXPECT_NO_GL_ERROR();

    EXPECT_EQ_INT(glIsEnabled(GL_VERTEX_ARRAY), GL_FALSE);
    EXPECT_EQ_INT(glIsEnabled(GL_COLOR_ARRAY), GL_FALSE);
    GLint size = -1;
    glGetIntegerv(GL_COLOR_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 4);                     /* default */
}

/* ------------------------------------------------------------------ */
/* registry-constant regression tests (see the s38 enum audit: four of
 * these constants were wrong for months and only self-consistent MGL
 * constants exercised the paths)                                      */

TEST(generate_mipmap_registry_constant)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    /* the value in every app's gl.h (SGIS_generate_mipmap) */
    glTexParameteri(GL_TEXTURE_2D, 0x8191 /* GL_GENERATE_MIPMAP */, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    GLint v = -1;
    glGetTexParameteriv(GL_TEXTURE_2D, 0x8191, &v);
    EXPECT_EQ_INT(v, GL_TRUE);

    GLubyte img[8 * 8 * 3];
    memset(img, 0x80, sizeof img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
    GLint w = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 3, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 1);                        /* the mip chain was generated */
    glDeleteTextures(1, &tex);
}

TEST(feedback_registry_constants)
{
    GLfloat buf[64];

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -10, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /* GL_2D = 0x600 in the registry; MGL had 0x1400 (= GL_POINTS) and
     * glFeedbackBuffer(GL_2D) raised GL_INVALID_ENUM for every real app */
    glFeedbackBuffer(64, 0x600 /* GL_2D */, buf);
    EXPECT_NO_GL_ERROR();
    glFeedbackBuffer(64, 0x601 /* GL_3D */, buf);
    EXPECT_NO_GL_ERROR();
    glFeedbackBuffer(64, 0x604 /* GL_4D_COLOR_TEXTURE */, buf);
    EXPECT_NO_GL_ERROR();

    glFeedbackBuffer(64, 0x600, buf);
    glRenderMode(GL_FEEDBACK);
    glPassThrough(42.0f);
    glBegin(GL_POINTS);
    glVertex3f(0.5f, 0.25f, 1.0f);
    glEnd();
    GLint n = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(n, 5);                        /* token+value, token, 2 floats */
    if (n >= 5)
    {
        EXPECT_NEAR(buf[0], 0x700, 0.5);        /* GL_PASS_THROUGH_TOKEN */
        EXPECT_NEAR(buf[1], 42.0, 1e-5);
        EXPECT_NEAR(buf[2], 0x701, 0.5);        /* GL_POINT_TOKEN */
    }

    /* 3D feedback: three floats per vertex */
    glFeedbackBuffer(64, 0x601, buf);
    glRenderMode(GL_FEEDBACK);
    glBegin(GL_POINTS);
    glVertex3f(0.5f, 0.25f, 1.0f);
    glEnd();
    n = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(n, 4);                        /* token + 3 floats */
    if (n >= 4)
        EXPECT_NEAR(buf[0], 0x701, 0.5);
    EXPECT_NO_GL_ERROR();
}

TEST(edge_flag_and_cull_face_queries)
{
    GLboolean b = 2;

    /* GL_EDGE_FLAG = 0x0B43 (was misdefined 0x0B44 = GL_CULL_FACE, so
     * glGetBooleanv(GL_CULL_FACE) answered the edge flag) */
    glGetBooleanv(GL_EDGE_FLAG, &b);
    EXPECT_EQ_INT(b, GL_TRUE);                  /* default edge flag */
    glEdgeFlag(GL_FALSE);
    glGetBooleanv(GL_EDGE_FLAG, &b);
    EXPECT_EQ_INT(b, GL_FALSE);
    glEdgeFlag(GL_TRUE);

    glDisable(GL_CULL_FACE);
    glGetBooleanv(GL_CULL_FACE, &b);
    EXPECT_EQ_INT(b, GL_FALSE);
    glEnable(GL_CULL_FACE);
    glGetBooleanv(GL_CULL_FACE, &b);
    EXPECT_EQ_INT(b, GL_TRUE);
    glDisable(GL_CULL_FACE);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* client-array queries (new surface, exercised with registry values)  */

TEST(client_array_descriptor_queries)
{
    static const GLfloat verts[12] = { 0 };
    static const GLubyte colors[9] = { 0 };
    static const GLfloat tex[8] = { 0 };

    GLint size = -1, type = -1, stride = -1;

    /* spec defaults before any pointer is set */
    glGetIntegerv(GL_VERTEX_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 4);
    glGetIntegerv(GL_VERTEX_ARRAY_TYPE, &type);
    EXPECT_EQ_INT(type, GL_FLOAT);
    glGetIntegerv(GL_VERTEX_ARRAY_STRIDE, &stride);
    EXPECT_EQ_INT(stride, 0);
    glGetIntegerv(GL_COLOR_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 4);
    glGetIntegerv(GL_NORMAL_ARRAY_TYPE, &type);
    EXPECT_EQ_INT(type, GL_FLOAT);

    glVertexPointer(3, GL_FLOAT, 12, verts);
    glColorPointer(3, GL_UNSIGNED_BYTE, 3, colors);
    glTexCoordPointer(2, GL_FLOAT, 0, tex);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glGetIntegerv(GL_VERTEX_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 3);
    glGetIntegerv(GL_VERTEX_ARRAY_STRIDE, &stride);
    EXPECT_EQ_INT(stride, 12);
    glGetIntegerv(GL_COLOR_ARRAY_TYPE, &type);
    EXPECT_EQ_INT(type, GL_UNSIGNED_BYTE);
    glGetIntegerv(GL_COLOR_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 3);
    glGetIntegerv(GL_TEXTURE_COORD_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 2);

    EXPECT_EQ_INT(glIsEnabled(GL_VERTEX_ARRAY), GL_TRUE);
    EXPECT_EQ_INT(glIsEnabled(GL_COLOR_ARRAY), GL_TRUE);
    EXPECT_EQ_INT(glIsEnabled(GL_NORMAL_ARRAY), GL_FALSE);

    /* the server enable path (Mesa compat parity) toggles the same state */
    glDisable(GL_COLOR_ARRAY);
    EXPECT_EQ_INT(glIsEnabled(GL_COLOR_ARRAY), GL_FALSE);
    glEnable(GL_COLOR_ARRAY);
    EXPECT_EQ_INT(glIsEnabled(GL_COLOR_ARRAY), GL_TRUE);

    /* texcoord arrays are per client-active-texture unit */
    glClientActiveTexture(GL_TEXTURE1);
    EXPECT_EQ_INT(glIsEnabled(GL_TEXTURE_COORD_ARRAY), GL_FALSE);
    glGetIntegerv(GL_TEXTURE_COORD_ARRAY_SIZE, &size);
    EXPECT_EQ_INT(size, 4);                     /* unit 1 still at defaults */
    glClientActiveTexture(GL_TEXTURE0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* fixed-function state roundtrips legacy apps query every frame       */

TEST(light_position_snapshot_semantics)
{
    GLfloat p[4] = { 1, 2, 3, 1 }, q[4] = { -9, -9, -9, -9 };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(1, 0, 0);                      /* modelview = T(1,0,0) */
    glLightfv(GL_LIGHT0, GL_POSITION, p);       /* stored as eye coords NOW */

    /* a later modelview change must not move the stored light */
    glTranslatef(100, 100, 100);
    glGetLightfv(GL_LIGHT0, GL_POSITION, q);
    EXPECT_NEAR(q[0], 2, 1e-6);                 /* (2,2,3,1) = T(1,0,0)*(1,2,3,1) */
    EXPECT_NEAR(q[1], 2, 1e-6);
    EXPECT_NEAR(q[2], 3, 1e-6);
    EXPECT_NEAR(q[3], 1, 1e-6);

    /* a directional light (w=0) keeps its direction: rotating +90 deg about
     * +X maps -Z to +Y (right-hand rule) */
    GLfloat d[4] = { 0, 0, -1, 0 };
    glLoadIdentity();
    glRotatef(90, 1, 0, 0);
    glLightfv(GL_LIGHT1, GL_POSITION, d);
    glLoadIdentity();
    glGetLightfv(GL_LIGHT1, GL_POSITION, q);
    EXPECT_NEAR(q[0], 0, 1e-6);
    EXPECT_NEAR(q[1], 1, 1e-5);
    EXPECT_NEAR(q[2], 0, 1e-5);
    EXPECT_NEAR(q[3], 0, 1e-6);
    EXPECT_NO_GL_ERROR();
}

TEST(clip_plane_get_roundtrip)
{
    const GLdouble eq[4] = { 1, 2, 3, 4 };
    GLdouble out[4] = { -9, -9, -9, -9 };

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClipPlane(GL_CLIP_PLANE1, eq);
    EXPECT_NO_GL_ERROR();
    glGetClipPlane(GL_CLIP_PLANE1, out);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR((GLfloat)out[i], (GLfloat)eq[i], 1e-9);

    /* the stored equation is eye space: a modelview change at get time
     * must not re-transform it */
    glTranslatef(5, 5, 5);
    glGetClipPlane(GL_CLIP_PLANE1, out);
    EXPECT_NEAR((GLfloat)out[0], 1, 1e-9);
    EXPECT_NEAR((GLfloat)out[3], 4, 1e-9);
    glLoadIdentity();

    glClipPlane(0x3009, eq);                    /* GL_CLIP_PLANE9: beyond MAX */
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_NO_GL_ERROR();
}

TEST(color_index_state_in_rgba_mode)
{
    GLint iv = -1;
    GLfloat f = -1;

    glGetIntegerv(GL_INDEX_MODE, &iv);
    EXPECT_EQ_INT(iv, GL_FALSE);
    glGetIntegerv(GL_RGBA_MODE, &iv);
    EXPECT_EQ_INT(iv, GL_TRUE);

    glGetFloatv(GL_CURRENT_INDEX, &f);
    EXPECT_NEAR(f, 1, 1e-6);                    /* spec default index = 1 */
    glIndexi(5);
    glGetFloatv(GL_CURRENT_INDEX, &f);
    EXPECT_NEAR(f, 5, 1e-6);

    /* colour-index-only state exists as state but has no RGBA effect */
    glIndexMask(3);
    EXPECT_NO_GL_ERROR();
    GLint mask = -1;
    glGetIntegerv(GL_INDEX_WRITEMASK, &mask);
    EXPECT_EQ_INT(mask, 3);
    glIndexMask(~0u);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(matrix_stack_depths_and_limits);
    RUN(matrix_mode_errors_and_persistence);
    RUN(per_unit_texture_stacks);
    RUN(transpose_matrix_entry_points);
    RUN(immediate_mode_error_behaviour);
    RUN(glRect_pixel_check);
    RUN(display_list_query_state);
    RUN(display_list_nesting_limit);
    RUN(display_list_list_base_and_calllists);
    RUN(attrib_stack_limits);
    RUN(attrib_stack_restores_state);
    RUN(client_attrib_stack_limits);
    RUN(client_attrib_stack_restores_arrays);
    RUN(generate_mipmap_registry_constant);
    RUN(feedback_registry_constants);
    RUN(edge_flag_and_cull_face_queries);
    RUN(client_array_descriptor_queries);
    RUN(light_position_snapshot_semantics);
    RUN(clip_plane_get_roundtrip);
    RUN(color_index_state_in_rgba_mode);
    return mgl_test_summary();
}
