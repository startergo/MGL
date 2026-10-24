/* scaling probe: plain drawArrays submit cost vs vertex count / mode (headless) */
#include "mgl_test.h"
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const char *kVS =
    "#version 330 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kFS =
    "#version 330 core\n"
    "out vec4 col;\n"
    "void main() { col = vec4(0.2, 0.6, 1.0, 1.0); }\n";

int main(void)
{
    GLuint prog = mgl_make_program(kVS, kFS);
    glUseProgram(prog);

    const int N = 1 << 20;
    float *v = malloc(sizeof(float) * 2 * N);
    for (int i = 0; i < N; i++) { v[i*2] = -1.0f + 2.0f * (i % 1024) / 1024.0f; v[i*2+1] = -1.0f + 2.0f * (i / 1024) / 1024.0f; }
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * N, v, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glViewport(0, 0, 800, 600);
    glFinish();

    int counts[] = { 1024, 16384, 65536, 262144, 1048576 };
    GLenum modes[] = { GL_POINTS, GL_TRIANGLES };
    const char *mn[] = { "points ", "tris   " };
    for (unsigned m = 0; m < 2; m++)
        for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); c++)
        {
            int n = counts[c], rep = 60;
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < rep; i++)
                glDrawArrays(modes[m], 0, n);
            double t1 = now_s();
            glFinish();
            double t2 = now_s();
            printf("%s %8d verts: submit %8.1f us/draw, finish-wait %6.1f us/draw\n",
                   mn[m], n, (t1 - t0) * 1e6 / rep, (t2 - t1) * 1e6 / rep);
        }
    free(v);
    return 0;
}
