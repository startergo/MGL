/* scratch benchmark for the gpuaudit domain (not part of the suite) */
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
    "uniform float u_off;\n"
    "void main() { gl_Position = vec4(pos + u_off, 0.0, 1.0); }\n";
static const char *kFS =
    "#version 330 core\n"
    "out vec4 col;\n"
    "void main() { col = vec4(0.2, 0.6, 1.0, 1.0); }\n";

int main(void)
{
    GLuint prog = mgl_make_program(kVS, kFS);
    glUseProgram(prog);
    GLint off = glGetUniformLocation(prog, "u_off");

    float quad[] = { -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint idx[] = { 0, 1, 2, 2, 1, 3 };
    GLuint ebo;
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glFinish();

    /* (a) 2000 identical draws */
    double t0 = now_s();
    for (int i = 0; i < 2000; i++)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();
    double t1 = now_s();
    printf("2000 identical drawArrays: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);

    t0 = now_s();
    for (int i = 0; i < 2000; i++)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glFinish();
    t1 = now_s();
    printf("2000 identical drawElements: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);

    /* (b) uniform update between draws */
    t0 = now_s();
    for (int i = 0; i < 2000; i++) {
        glUniform1f(off, (float)(i & 7) * 0.001f);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glFinish();
    t1 = now_s();
    printf("2000 draws + uniform: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);

    /* (c) blend state flip between draws (PSO-relevant state) */
    t0 = now_s();
    for (int i = 0; i < 1000; i++) {
        if (i & 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glFinish();
    t1 = now_s();
    printf("1000 draws + blend flip: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 1000);

    /* (d) draw with restart enabled + list mode (index scan path) */
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFFFFFFu);
    t0 = now_s();
    for (int i = 0; i < 2000; i++)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glFinish();
    t1 = now_s();
    printf("2000 drawElements restart+list: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);

    /* (e) large restart scan: 1M indices */
    {
        size_t n = 1024 * 1024;
        uint32_t *big = malloc(n * 4);
        for (size_t i = 0; i < n; i++) big[i] = (uint32_t)(i % 1000);
        GLuint b2;
        glGenBuffers(1, &b2);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b2);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, n * 4, big, GL_STATIC_DRAW);
        t0 = now_s();
        for (int i = 0; i < 100; i++)
            glDrawElements(GL_TRIANGLES, (GLsizei)n, GL_UNSIGNED_INT, 0);
        glFinish();
        t1 = now_s();
        printf("100 draws of 1M indices (restart+list): %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 100);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    }
    glDisable(GL_PRIMITIVE_RESTART);

    /* (f) fan/loop emulation */
    t0 = now_s();
    for (int i = 0; i < 2000; i++)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 64);
    glFinish();
    t1 = now_s();
    printf("2000 drawArrays FAN 64 verts: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);

    /* (g) uint8 indices */
    {
        GLubyte ub[6] = { 0, 1, 2, 2, 1, 3 };
        GLuint b3;
        glGenBuffers(1, &b3);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b3);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6, ub, GL_STATIC_DRAW);
        t0 = now_s();
        for (int i = 0; i < 2000; i++)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);
        glFinish();
        t1 = now_s();
        printf("2000 drawElements UBYTE: %.3f ms (%.1f us/draw)\n", (t1 - t0) * 1e3, (t1 - t0) * 1e6 / 2000);
    }

    /* (h) clearBufferData 16MB */
    {
        GLuint b4;
        glGenBuffers(1, &b4);
        glBindBuffer(GL_ARRAY_BUFFER, b4);
        glBufferData(GL_ARRAY_BUFFER, 16 << 20, NULL, GL_STATIC_DRAW);
        float c[4] = { 1.0f, 0.5f, 0.25f, 1.0f };
        t0 = now_s();
        glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32F, GL_RGBA, GL_FLOAT, c);
        glFinish();
        t1 = now_s();
        printf("clearBufferData 16MB RGBA32F: %.3f ms\n", (t1 - t0) * 1e3);
        t0 = now_s();
        glClearBufferData(GL_ARRAY_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, c);
        glFinish();
        t1 = now_s();
        printf("clearBufferData 16MB R8: %.3f ms\n", (t1 - t0) * 1e3);
    }

    /* (i) readpixels 512x512 */
    {
        void *px = malloc(512 * 512 * 4);
        glFinish();
        t0 = now_s();
        glReadPixels(0, 0, 512, 512, GL_RGBA, GL_UNSIGNED_BYTE, px);
        t1 = now_s();
        printf("readPixels 512x512 RGBA/UBYTE: %.3f ms\n", (t1 - t0) * 1e3);
        free(px);
    }

    printf("done\n");
    return 0;
}
