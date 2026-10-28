/* abb_prof.c — ab_bench mode 0/1 loop + MGL_PROFILE bucket dump via dlsym.
 * Same draw loop as tests/headless/ab_bench.c; after the timed loop calls the
 * library's own mgl_profile_dump() so per-draw CPU buckets can be compared
 * between two libmgl builds without WindowServer noise. */
#define GL_GLEXT_PROTOTYPES 1
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <dlfcn.h>
#include "glcorearb.h"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const char *kVS =
    "#version 330 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kFS =
    "#version 330 core\n"
    "out vec4 col;\n"
    "void main() { col = vec4(0.2, 0.6, 1.0, 1.0); }\n";

static GLuint mkprog(void)
{
    GLuint p = glCreateProgram();
    const char *srcs[2] = { kVS, kFS };
    GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], NULL);
        glCompileShader(s);
        glAttachShader(p, s);
    }
    glLinkProgram(p);
    return p;
}

int main(int argc, char **argv)
{
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    int iters = argc > 2 ? atoi(argv[2]) : 2000;
    void (*dump)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_profile_dump");
    if (!dump) { fprintf(stderr, "no mgl_profile_dump\n"); return 2; }

    GLuint prog = mkprog();
    glUseProgram(prog);
    float quad[] = { -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    GLuint vbo, vao;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    /* warmup + first dump (ignored: includes pipeline compile) */
    for (int i = 0; i < 200; i++) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();
    dump();

    double t0 = now_s();
    if (mode == 0)
        for (int i = 0; i < iters; i++) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    else
        for (int i = 0; i < iters; i++) {
            if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
    glFinish();
    double t = now_s() - t0;
    fprintf(stderr, "RESULT mode%d x%d %.4f ms\n", mode, iters, t * 1e3);
    dump();   /* steady-state buckets over exactly `iters` draws */
    return 0;
}
