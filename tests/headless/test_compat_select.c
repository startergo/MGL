/*
 * test_compat_select.c — GL_SELECT / GL_FEEDBACK conformance for the
 * compatibility profile (spec 5.4 + Mesa main/feedback.c, drawpix.c):
 *   - glRenderMode into GL_SELECT/GL_FEEDBACK without a buffer is
 *     GL_INVALID_OPERATION; bad mode is GL_INVALID_ENUM;
 *   - glSelectBuffer/glFeedbackBuffer negative size, NULL buffer, bad type,
 *     and "already in that mode" errors;
 *   - glPassThrough and the name-stack commands are silent no-ops outside
 *     their mode (no GL_INVALID_OPERATION);
 *   - every primitive drawn with the SAME names merges into ONE hit record
 *     whose zmin/zmax span all of them (Mesa HitFlag deferral); records are
 *     closed by name-stack changes and by leaving the mode;
 *   - hit z values are scaled by 2^32-1;
 *   - overflowing either buffer makes glRenderMode return -1 (tokens/records
 *     are written until the buffer is full, then counted past the end);
 *   - glBitmap/glDrawPixels/glCopyPixels in feedback emit
 *     GL_BITMAP_TOKEN/GL_DRAW_PIXEL_TOKEN/GL_COPY_PIXEL_TOKEN plus the raster
 *     position vertex when the raster position is valid, rasterize nothing
 *     outside GL_RENDER, and glBitmap still advances the raster position;
 *   - glPopName/glLoadName are compiled into display lists.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

/* feedback tokens (gl.h values; not in the core header) */
#define GL_BITMAP_TOKEN     0x0704
#define GL_DRAW_PIXEL_TOKEN 0x0705
#define GL_COPY_PIXEL_TOKEN 0x0706

static void *compat_ctx;

static void sel_begin(void)
{
    compat_ctx = mglCompatCreateContext();
    EXPECT_TRUE(compat_ctx != NULL);
    glViewport(0, 0, 64, 64);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void quad(float x0, float y0, float x1, float y1, float z)
{
    glBegin(GL_QUADS);
    glVertex3f(x0, y0, z);
    glVertex3f(x1, y0, z);
    glVertex3f(x1, y1, z);
    glVertex3f(x0, y1, z);
    glEnd();
}

#define ZWORD(z) ((GLuint)((double)(z) * 4294967295.0))

/* ------------------------------------------------------------------ */

TEST(render_mode_and_buffer_errors)
{
    GLuint sbuf[64];
    GLfloat fbuf[64];

    sel_begin();

    glRenderMode(GL_SELECT);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);      /* no glSelectBuffer yet */
    glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();

    glRenderMode(GL_FEEDBACK);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);      /* no glFeedbackBuffer yet */
    glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();

    glRenderMode(0xBEEF);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    EXPECT_NO_GL_ERROR();

    glSelectBuffer(-1, sbuf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glSelectBuffer(8, sbuf);
    EXPECT_NO_GL_ERROR();
    glRenderMode(GL_SELECT);
    EXPECT_NO_GL_ERROR();
    glSelectBuffer(8, sbuf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);      /* already selecting */
    glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();

    glFeedbackBuffer(-1, GL_3D, fbuf);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glFeedbackBuffer(8, GL_3D, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glFeedbackBuffer(8, 0x1234, fbuf);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFeedbackBuffer(8, GL_3D, fbuf);
    EXPECT_NO_GL_ERROR();
    glFeedbackBuffer(8, GL_3D, fbuf);
    EXPECT_NO_GL_ERROR();                       /* not in feedback yet */
    glRenderMode(GL_FEEDBACK);
    EXPECT_NO_GL_ERROR();
    glFeedbackBuffer(8, GL_3D, fbuf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);      /* already in feedback */
    glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();
}

TEST(silent_noops_outside_mode)
{
    sel_begin();

    /* glPassThrough outside feedback: no error, no effect */
    glPassThrough(42.0f);
    EXPECT_NO_GL_ERROR();

    /* name-stack commands outside GL_SELECT: no error, no state change */
    glPushName(1);
    glPopName();
    glLoadName(2);
    glInitNames();
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT((int)glGetError(), GL_NO_ERROR);
}

TEST(name_stack_limits)
{
    GLuint sbuf[512];
    GLint max = 0, depth = 99;

    sel_begin();
    glGetIntegerv(GL_MAX_NAME_STACK_DEPTH, &max);
    EXPECT_EQ_INT(max, 64);

    glSelectBuffer(512, sbuf);
    glRenderMode(GL_SELECT);

    glLoadName(1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);      /* empty stack */
    glPopName();
    EXPECT_GL_ERROR(GL_STACK_UNDERFLOW);

    for (int i = 0; i < max; i++)
        glPushName((GLuint)i + 100);
    EXPECT_NO_GL_ERROR();
    glPushName(999);
    EXPECT_GL_ERROR(GL_STACK_OVERFLOW);         /* depth cap = 64 */

    glGetIntegerv(GL_NAME_STACK_DEPTH, &depth);
    EXPECT_EQ_INT(depth, 64);
    glRenderMode(GL_RENDER);
}

/* two batches with the same name must merge into ONE hit record whose
 * zmin/zmax span both; a name change closes the record */
TEST(one_record_per_name)
{
    GLuint sbuf[64];
    GLint hits;

    sel_begin();
    glSelectBuffer(64, sbuf);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(7);
    quad(-0.5f, -0.5f, 0.0f, 0.5f, 0.2f);       /* win z 0.6 */
    quad(0.0f, -0.5f, 0.5f, 0.5f, -0.2f);       /* win z 0.4, another batch */
    glLoadName(9);
    quad(-0.9f, -0.9f, 0.9f, 0.9f, 0.0f);       /* win z 0.5 */
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, 2);

    /* record 1: [count=1, zmin, zmax, 7] spanning BOTH quads */
    EXPECT_EQ_INT((int)sbuf[0], 1);
    EXPECT_EQ_INT((int)sbuf[3], 7);
    EXPECT_TRUE(llabs((long long)sbuf[1] - (long long)ZWORD(0.4f)) < 16);
    EXPECT_TRUE(llabs((long long)sbuf[2] - (long long)ZWORD(0.6f)) < 16);

    /* record 2: [count=1, zmin=0.5, zmax=0.5, 9] */
    EXPECT_EQ_INT((int)sbuf[4], 1);
    EXPECT_EQ_INT((int)sbuf[7], 9);
    EXPECT_TRUE(llabs((long long)sbuf[5] - (long long)ZWORD(0.5f)) < 16);
    EXPECT_TRUE(llabs((long long)sbuf[6] - (long long)ZWORD(0.5f)) < 16);
}

TEST(hit_with_empty_name_stack)
{
    GLuint sbuf[64];
    GLint hits;

    sel_begin();
    glSelectBuffer(64, sbuf);
    glRenderMode(GL_SELECT);
    glInitNames();                              /* no names pushed at all */
    quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f);
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, 1);
    EXPECT_EQ_INT((int)sbuf[0], 0);             /* zero names */
    EXPECT_TRUE(llabs((long long)sbuf[1] - (long long)ZWORD(0.5f)) < 16);
}

TEST(select_overflow_returns_minus_one)
{
    GLuint sbuf[2];
    GLint hits;

    sel_begin();
    glSelectBuffer(2, sbuf);                    /* a record needs 4 words */
    glRenderMode(GL_SELECT);
    glPushName(3);
    quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f);
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, -1);
    /* the first two words still made it into the buffer */
    EXPECT_EQ_INT((int)sbuf[0], 1);

    /* and the mode state is clean again */
    glSelectBuffer(64, sbuf);
    glRenderMode(GL_SELECT);
    glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();
}

TEST(feedback_overflow_returns_minus_one)
{
    GLfloat fbuf[4];
    GLint n;

    sel_begin();
    glFeedbackBuffer(4, GL_3D, fbuf);
    glRenderMode(GL_FEEDBACK);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, 0);
    glVertex3f(0.1f, 0, 0);
    glEnd();
    n = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(n, -1);                       /* 8 floats wanted, 4 stored */

    glFeedbackBuffer(64, GL_3D, fbuf);
    glRenderMode(GL_FEEDBACK);
    glBegin(GL_POINTS);
    glVertex3f(0, 0, 0);
    glEnd();
    n = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(n, 4);                        /* token + 3 coords */
    EXPECT_NEAR(fbuf[0], (GLfloat)GL_POINT_TOKEN, 0);
    EXPECT_NEAR(fbuf[1], 32.0f, 0.01f);
    EXPECT_NEAR(fbuf[2], 32.0f, 0.01f);
}

/* ------------------------------------------------------------------ */
/* pixel operations in feedback / select                               */

static GLubyte bits[8] = { 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA };

TEST(feedback_pixel_tokens)
{
    GLfloat fbuf[128];
    GLint n;
    GLfloat px[4 * 4] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

    sel_begin();
    glFeedbackBuffer(128, GL_3D_COLOR, fbuf);
    glRenderMode(GL_FEEDBACK);

    glColor4f(1, 0, 0, 1);
    glRasterPos3f(32.0f / 64.0f * 2.0f - 1.0f, 0.0f, 0.0f);   /* window (32,32) */
    glBitmap(8, 8, 0, 0, 4.0f, 2.0f, bits);     /* advances to (36,34) */
    glDrawPixels(4, 4, GL_RGBA, GL_FLOAT, px);
    glCopyPixels(0, 0, 4, 4, GL_COLOR);
    n = glRenderMode(GL_RENDER);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(n, 3 * (1 + 7));              /* 3D_COLOR = 7 floats/vertex */

    EXPECT_NEAR(fbuf[0], (GLfloat)GL_BITMAP_TOKEN, 0);
    EXPECT_NEAR(fbuf[1], 32.0f, 0.01f);
    EXPECT_NEAR(fbuf[2], 32.0f, 0.01f);
    EXPECT_NEAR(fbuf[3], 0.5f, 0.01f);
    EXPECT_NEAR(fbuf[4], 1.0f, 0.01f);          /* raster colour red */
    EXPECT_NEAR(fbuf[5], 0.0f, 0.01f);
    EXPECT_NEAR(fbuf[6], 0.0f, 0.01f);
    EXPECT_NEAR(fbuf[7], 1.0f, 0.01f);

    EXPECT_NEAR(fbuf[8], (GLfloat)GL_DRAW_PIXEL_TOKEN, 0);
    EXPECT_NEAR(fbuf[9], 36.0f, 0.01f);         /* raster pos advanced */
    EXPECT_NEAR(fbuf[10], 34.0f, 0.01f);

    /* [8]=token + 7 floats -> the copy token lands at 16 */
    EXPECT_NEAR(fbuf[16], (GLfloat)GL_COPY_PIXEL_TOKEN, 0);
    EXPECT_NEAR(fbuf[17], 36.0f, 0.01f);        /* copypixels does not advance */
}

TEST(feedback_pixel_invalid_raster_pos)
{
    GLfloat fbuf[64];
    GLint n;

    sel_begin();
    glFeedbackBuffer(64, GL_3D_COLOR, fbuf);
    glRenderMode(GL_FEEDBACK);
    glRasterPos4f(0, 0, 0, 0);                  /* clip w = 0 -> invalid */
    glBitmap(8, 8, 0, 0, 0, 0, bits);
    n = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(n, 0);                        /* no token when invalid */
    EXPECT_NO_GL_ERROR();
}

TEST(pixel_ops_do_nothing_in_select)
{
    GLuint sbuf[64];
    GLfloat px[4] = { 1, 1, 1, 1 };
    GLint hits;

    sel_begin();
    glSelectBuffer(64, sbuf);
    glRenderMode(GL_SELECT);
    glPushName(1);
    glRasterPos3f(0, 0, 0);
    glDrawPixels(2, 2, GL_RGBA, GL_FLOAT, px);
    glBitmap(8, 8, 0, 0, 0, 0, bits);
    glCopyPixels(0, 0, 2, 2, GL_COLOR);
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, 0);                     /* no hits from pixel ops */
    EXPECT_NO_GL_ERROR();
}

/* ------------------------------------------------------------------ */
/* display lists compile the name-stack commands                       */

TEST(name_stack_compiles_into_lists)
{
    GLuint sbuf[64], list;
    GLint hits;

    sel_begin();

    list = glGenLists(1);
    glNewList(list, GL_COMPILE);
    glPushName(5);
    glLoadName(6);
    quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f);
    glPopName();
    glEndList();
    EXPECT_NO_GL_ERROR();

    glSelectBuffer(64, sbuf);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(1);                              /* stack: 1 */
    glCallList(list);                           /* push 5, load 6, draw, pop */
    quad(-0.5f, -0.5f, 0.5f, 0.5f, 0.0f);       /* still name 1 */
    hits = glRenderMode(GL_RENDER);
    EXPECT_EQ_INT(hits, 2);

    /* record 1 has the two names {1,6}; record 2 (5 floats) has {1} */
    EXPECT_EQ_INT((int)sbuf[0], 2);
    EXPECT_EQ_INT((int)sbuf[3], 1);
    EXPECT_EQ_INT((int)sbuf[4], 6);
    EXPECT_TRUE(llabs((long long)sbuf[2] - (long long)ZWORD(0.5f)) < 16);
    EXPECT_EQ_INT((int)sbuf[5], 1);
    EXPECT_TRUE(llabs((long long)sbuf[6] - (long long)ZWORD(0.5f)) < 16);
    EXPECT_EQ_INT((int)sbuf[8], 1);
}

int main(void)
{
    RUN(render_mode_and_buffer_errors);
    RUN(silent_noops_outside_mode);
    RUN(name_stack_limits);
    RUN(one_record_per_name);
    RUN(hit_with_empty_name_stack);
    RUN(select_overflow_returns_minus_one);
    RUN(feedback_overflow_returns_minus_one);
    RUN(feedback_pixel_tokens);
    RUN(feedback_pixel_invalid_raster_pos);
    RUN(pixel_ops_do_nothing_in_select);
    RUN(name_stack_compiles_into_lists);
    return mgl_test_summary();
}
