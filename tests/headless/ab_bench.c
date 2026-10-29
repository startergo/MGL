/* ab_bench.c — A/B benchmark for the gpuaudit domain: run against two builds of
 * libmgl (rpath build/ab_old, build/ab_new) and compare. Not part of the suite. */
#include "mgl_test.h"
#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static double now_s(void)
{
    struct timespec ts;
    if (getenv("MGL_AB_CPU"))
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);   /* GPU-contention-immune */
    else
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
        case 11: {  /* blend flip, ONE draw per frame: MGL_PROFILE dumps per 120 frames */
            for (int i = 0; i < iters; i++) {
                if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                glFinish();
            }
            t = 0; snprintf(label, sizeof(label), "blend-flip framed x%d", iters);
            {   /* MGL_PROFILE buckets for the whole loop (mgl_prof_frame never
                 * fires headless: the compat swap path is not taken) */
                void (*dump)(void) = (void (*)(void))dlsym(RTLD_DEFAULT, "mgl_profile_dump");
                if (dump) dump();
            }
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
        case 7: {  /* same as 3 but submit-only (the CPU cost of the draw) */
            glEnable(GL_PRIMITIVE_RESTART);
            glPrimitiveRestartIndex(0xFFFFFFFFu);
            size_t n = 512 * 1024;
            uint32_t *big = malloc(n * 4);
            for (size_t i = 0; i < n; i++) big[i] = (uint32_t)(i % 1000);
            GLuint b2;
            glGenBuffers(1, &b2);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b2);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, n * 4, big, GL_STATIC_DRAW);
            for (int i = 0; i < 2; i++) glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
            t = now_s() - t0;
            glFinish();
            snprintf(label, sizeof(label), "1M-index restart submits x%d", iters);
            glDisable(GL_PRIMITIVE_RESTART);
            break;
        }
        case 8: {  /* clear of a GPU-resident buffer, submit only */
            /* the cleared buffer is itself the vertex source of a draw, so it is
             * GPU-resident (has an MTLBuffer) before the clears start */
            GLuint b4, va;
            glGenBuffers(1, &b4);
            glBindBuffer(GL_ARRAY_BUFFER, b4);
            glBufferData(GL_ARRAY_BUFFER, 16 << 20, NULL, GL_STATIC_DRAW);
            glGenVertexArrays(1, &va);
            glBindVertexArray(va);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glFinish();

            float c[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
            glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, c);   /* promote */
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, c);
            t = now_s() - t0;
            glFinish();
            snprintf(label, sizeof(label), "16MB RGBA32F clear submits x%d", iters);
            glBindVertexArray(va);
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
        case 9: {  /* 1000-command multi-draw indirect, submit only */
            enum { N = 1000 };
            typedef struct { GLuint count, instanceCount, first, baseVertex, baseInstance; } DrawCmd;
            DrawCmd cmds[N];
            for (int i = 0; i < N; i++) cmds[i] = (DrawCmd){ 3, 1, 0, 0, 0 };
            GLuint ibuf;
            glGenBuffers(1, &ibuf);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ibuf);
            glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(cmds), cmds, GL_STATIC_DRAW);
            for (int i = 0; i < 5; i++) glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, N, 20);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, N, 20);
            t = now_s() - t0;
            glFinish();
            snprintf(label, sizeof(label), "1000-cmd multiDrawIndirect x%d", iters);
            break;
        }
        case 10: {  /* many small texture uploads (pooled staging) */
            GLuint tex;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            GLubyte *tile = (GLubyte *)malloc(512 * 512 * 4);
            for (int i = 0; i < 512 * 512 * 4; i++) tile[i] = (GLubyte)(i * 7);
            for (int i = 0; i < 20; i++)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, tile);
            glFinish();
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, tile);
            t = now_s() - t0;
            glFinish();
            snprintf(label, sizeof(label), "big texSubImage 512x512 x%d", iters);
            glDeleteTextures(1, &tex);
            break;
        }
        default: return 1;
    }

    printf("RESULT %s %.4f\n", label, t * 1e3);
    return 0;
}
