/* ab_compat_draw.c — compatibility-profile per-draw CPU microbench (compat
 * domain A/B).  Exercises the legacy paths this domain owns: immediate-mode
 * quads, matrix pushes, attrib push/pop and the compat enable/get surface,
 * reporting CPU ms per 1000 draw calls (min of N runs is taken by the driver
 * shell).  MGL_AB_CPU=1 switches to CPU-time accounting like ab_patches.c. */
#define GL_GLEXT_PROTOTYPES 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gl_compat.h"
#include "mgl_compat.h"

static double now_s(void)
{
    struct timespec ts;
    if (getenv("MGL_AB_CPU"))
        /* thread CPU only: Metal's compile/JIT helper threads otherwise show
         * up in the process total and swamp the per-draw numbers */
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    else
        clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void frame_imm(int quads)
{
    for (int i = 0; i < quads; i++)
    {
        glPushMatrix();
        glRotatef(1.0f, 0, 0, 1);
        glBegin(GL_QUADS);
        glColor3f(1, 0.5f, 0.25f);
        glVertex2f(-1, -1);
        glVertex2f(1, -1);
        glVertex2f(1, 1);
        glVertex2f(-1, 1);
        glEnd();
        glPopMatrix();
    }
}

static void frame_state(int quads)
{
    /* attrib push/pop of cheap groups (no FF pipeline flip): colour + matrix */
    for (int i = 0; i < quads; i++)
    {
        glPushAttrib(GL_CURRENT_BIT | GL_TRANSFORM_BIT);
        glColor3f(0.9f, 0.8f, 0.7f);
        glTranslatef(0.001f, 0, 0);
        glBegin(GL_QUADS);
        glVertex2f(-1, -1); glVertex2f(1, -1);
        glVertex2f(1, 1);   glVertex2f(-1, 1);
        glEnd();
        glPopAttrib();
    }
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    void *cc = mglCompatCreateContext();
    if (!cc) { fprintf(stderr, "compat context failed\n"); return 1; }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, 256, 256);
    glDisable(GL_DEPTH_TEST);

    /* warm-up + pipeline compile */
    frame_imm(64);
    frame_state(64);
    glFinish();

    double t0 = now_s();
    for (int i = 0; i < iters; i++)
        frame_imm(8);
    glFinish();
    double t1 = now_s();
    printf("compat imm-quads submit %.3f\n", (t1 - t0) * 1000.0 * 1000.0 / (iters * 8));

    t0 = now_s();
    for (int i = 0; i < iters; i++)
        frame_state(8);
    glFinish();
    t1 = now_s();
    printf("compat attrib-lit submit %.3f\n", (t1 - t0) * 1000.0 * 1000.0 / (iters * 8));

    /* state query loop (glGet of legacy state + enable toggles): pure CPU,
     * no Metal traffic - the stable regression signal for state paths */
    GLfloat v;
    GLint iv;
    GLboolean b;
    t0 = now_s();
    for (int i = 0; i < iters * 8; i++)
    {
        glGetFloatv(GL_LIGHT_MODEL_AMBIENT, &v);
        glGetIntegerv(GL_VERTEX_ARRAY_SIZE, &iv);
        glGetBooleanv(GL_COLOR_ARRAY, &b);
        glEnable(GL_LIGHTING);
        glDisable(GL_LIGHTING);
        glIsEnabled(GL_LIGHT0);
    }
    t1 = now_s();
    printf("compat legacy get %.3f\n", (t1 - t0) * 1000.0 * 1000.0 / (iters * 8 * 6));
    return 0;
}
