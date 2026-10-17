/* ab_bench.c — A/B benchmark for the gpuaudit domain: run against two builds of
 * libmgl (rpath build/ab_old, build/ab_new) and compare. Not part of the suite. */
#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

static GLuint mkquad(void)
{
    float quad[] = { -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    GLuint vbo, vao;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    return vao;
}

int main(int argc, char **argv)
{
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    int iters = argc > 2 ? atoi(argv[2]) : 1000;

    GLuint prog = mgl_make_program(kVS, kFS);
    glUseProgram(prog);
    mkquad();
    GLuint idx[] = { 0, 1, 2, 2, 1, 3 };
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    double t;
    char label[128];

    switch (mode)
    {
        case 0: {  /* identical drawArrays */
            for (int i = 0; i < 100; i++) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++) glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "drawArrays x%d", iters);
            break;
        }
        case 1: {  /* blend state flip between draws (pipeline state churn) */
            for (int i = 0; i < 200; i++) {
                if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++) {
                if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            }
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "blend-flip draws x%d", iters);
            break;
        }
        case 2: {  /* triangle fan, per-draw CPU emulation */
            for (int i = 0; i < 100; i++) glDrawArrays(GL_TRIANGLE_FAN, 0, 64);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++) glDrawArrays(GL_TRIANGLE_FAN, 0, 64);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "drawArrays FAN64 x%d", iters);
            break;
        }
        case 3: {  /* restart-enabled indexed draws: index scan */
            glEnable(GL_PRIMITIVE_RESTART);
            glPrimitiveRestartIndex(0xFFFFFFFFu);
            size_t n = 1024 * 1024;
            uint32_t *big = malloc(n * 4);
            for (size_t i = 0; i < n; i++) big[i] = (uint32_t)(i % 1000);
            GLuint b2;
            glGenBuffers(1, &b2);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b2);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, n * 4, big, GL_STATIC_DRAW);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "1M-index restart draws x%d", iters);
            glDisable(GL_PRIMITIVE_RESTART);
            break;
        }
        case 4: {  /* 8-bit indices, per-draw widening */
            GLubyte ub[6] = { 0, 1, 2, 2, 1, 3 };
            GLuint b3;
            glGenBuffers(1, &b3);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b3);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6, ub, GL_STATIC_DRAW);
            for (int i = 0; i < 100; i++) glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++) glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "ubyte index draws x%d", iters);
            break;
        }
        case 5: {  /* glClearBufferData */
            GLuint b4;
            glGenBuffers(1, &b4);
            glBindBuffer(GL_ARRAY_BUFFER, b4);
            glBufferData(GL_ARRAY_BUFFER, 16 << 20, NULL, GL_STATIC_DRAW);
            float c[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
            int sub = argc > 3 ? atoi(argv[3]) : 0;
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glClearBufferData(GL_ARRAY_BUFFER, sub ? GL_R8 : GL_RGBA32F, sub ? GL_RED : GL_RGBA, GL_FLOAT, c);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "clearBufferData 16MB %s x%d", sub ? "R8" : "RGBA32F", iters);
            break;
        }
        case 6: {  /* readPixels into a PBO */
            GLuint fbo, tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
            GLuint pbo;
            glGenBuffers(1, &pbo);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
            glBufferData(GL_PIXEL_PACK_BUFFER, 512 * 512 * 4, NULL, GL_STREAM_READ);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glReadPixels(0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            glFinish();
            t = now_s() - t0;
            snprintf(label, sizeof(label), "readPixels PBO 512x512 x%d", iters);
            break;
        }
        default: return 1;
    }

    printf("RESULT %s %.4f\n", label, t * 1e3);
    return 0;
}
