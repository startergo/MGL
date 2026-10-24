/*
 * test_compat_thru.c — vertex/fragment throughput sanity (glmark2 [buffer] was
 * 10x below native on a single 100k-tri draw: find which part of the vertex
 * path costs).  Not a correctness test: prints ms/frame for a few variants.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void bench(const char *what, int indexed, int attrs, int verts_per_quad)
{
    const int N = 50000;                 /* quads */
    int vcount = N * verts_per_quad;

    const char *vs = "#version 460 core\nlayout(location=0) in vec2 p; out vec2 v;\n"
                     "void main(){ v = p*0.5+0.5; gl_Position = vec4(p,0,1); }\n";
    const char *fs = "#version 460 core\nin vec2 v; out vec4 o;\n"
                     "void main(){ o = vec4(v,0.5,1); }\n";
    char log[512] = { 0 };
    GLuint prog = glCreateProgram();
    GLuint sh[2] = { glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER) };
    glShaderSource(sh[0], 1, &vs, NULL); glCompileShader(sh[0]);
    glShaderSource(sh[1], 1, &fs, NULL); glCompileShader(sh[1]);
    glAttachShader(prog, sh[0]); glAttachShader(prog, sh[1]);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(prog, sizeof log, NULL, log); printf("   %s LINK FAIL %s\n", what, log); return; }
    glUseProgram(prog);

    int stride = attrs == 2 ? 24 : 8;
    float *data = malloc((size_t)vcount * stride);
    for (int i = 0; i < vcount; i++)
    {
        float x = -1.f + 2.f * ((i % 1250) / 1250.0f);
        float y = -1.f + 2.f * ((i / 1250) / 800.0f);
        float *e = data + i * (stride / 4);
        e[0] = x; e[1] = y;
        if (attrs == 2) { e[2] = 1; e[3] = 1; e[4] = 1; e[5] = 1; }
    }
    GLuint vbo, ibo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)vcount * stride, data, GL_DYNAMIC_DRAW);
    free(data);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void *)0);
    if (attrs == 2)
    {
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void *)16);
    }
    if (indexed)
    {
        unsigned *ix = malloc((size_t)N * 6 * sizeof *ix);
        for (int i = 0; i < N; i++)
        {
            unsigned b = (unsigned)(i * 4);
            ix[i * 6 + 0] = b;     ix[i * 6 + 1] = b + 1; ix[i * 6 + 2] = b + 2;
            ix[i * 6 + 3] = b;     ix[i * 6 + 4] = b + 2; ix[i * 6 + 5] = b + 3;
        }
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)N * 6 * 4, ix, GL_STATIC_DRAW);
        free(ix);
    }

    int count = indexed ? N * 6 : vcount;
    for (int i = 0; i < 3; i++)     /* warm pipelines */
    {
        if (indexed) glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL);
        else glDrawArrays(GL_TRIANGLES, 0, count);
    }
    glFinish();

    double t0 = now_ms();
    const int FRAMES = 60;
    for (int f = 0; f < FRAMES; f++)
    {
        if (indexed) glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL);
        else glDrawArrays(GL_TRIANGLES, 0, count);
    }
    glFinish();
    double dt = (now_ms() - t0) / FRAMES;
    printf("   %-30s %6.2f ms/frame  (%.1fM tri/s)\n", what, dt, 100000.0 / (dt * 1000.0));

    if (ibo) glDeleteBuffers(1, &ibo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteShader(sh[0]); glDeleteShader(sh[1]);
}

TEST(compat_thru_bench)
{
    bench("100k tris arrays pos-only", 0, 1, 4);
    bench("100k tris indexed pos-only", 1, 1, 4);
    bench("100k tris indexed pos+color", 1, 2, 4);
    bench("100k tris arrays pos+color", 0, 2, 4);
    EXPECT_NO_GL_ERROR();
}

/* index-type isolation: same 100k tris, ushort vs uint indices */
TEST(compat_thru_idxtype)
{
    const int N = 50000;
    const char *vs = "#version 460 core\nlayout(location=0) in vec2 p; out vec2 v;\n"
                     "void main(){ v = p*0.5+0.5; gl_Position = vec4(p,0,1); }\n";
    const char *fs = "#version 460 core\nin vec2 v; out vec4 o;\n"
                     "void main(){ o = vec4(v,0.5,1); }\n";
    GLuint prog = glCreateProgram();
    GLuint sh[2] = { glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER) };
    glShaderSource(sh[0], 1, &vs, NULL); glCompileShader(sh[0]);
    glShaderSource(sh[1], 1, &fs, NULL); glCompileShader(sh[1]);
    glAttachShader(prog, sh[0]); glAttachShader(prog, sh[1]);
    glLinkProgram(prog);
    glUseProgram(prog);

    float *data = malloc((size_t)N * 4 * 8);
    for (int i = 0; i < N * 4; i++)
    {
        data[i * 2 + 0] = -1.f + 2.f * ((i % 1250) / 1250.0f);
        data[i * 2 + 1] = -1.f + 2.f * ((i / 1250) / 800.0f);
    }
    GLuint vbo, ibo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)N * 4 * 8, data, GL_DYNAMIC_DRAW);
    free(data);
    unsigned short *ix16 = malloc((size_t)N * 6 * 2);
    unsigned int *ix32 = malloc((size_t)N * 6 * 4);
    for (int i = 0; i < N; i++)
    {
        unsigned b = (unsigned)(i * 4);
        unsigned short b16 = (unsigned short)(i * 4);
        for (int k = 0; k < 6; k++) { ix16[i * 6 + k] = b16 + k - (k > 2 ? 1 : 0); ix32[i * 6 + k] = b + k - (k > 2 ? 1 : 0); }
    }
    ix16[2] = 2; ix16[5] = 3; ix32[2] = 2; ix32[5] = 3;   /* simple 0,1,2 / 0,2,3 */

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    const char *names[2] = { "uint16 idx", "uint32 idx" };
    for (int t = 0; t < 2; t++)
    {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)N * 6 * (t ? 4 : 2), t ? (void *)ix32 : (void *)ix16, GL_STATIC_DRAW);
        for (int i = 0; i < 3; i++)
            glDrawElements(GL_TRIANGLES, N * 6, t ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, NULL);
        glFinish();
        double t0 = now_ms();
        for (int f = 0; f < 60; f++)
            glDrawElements(GL_TRIANGLES, N * 6, t ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, NULL);
        glFinish();
        printf("   %-30s %6.2f ms/frame\n", names[t], (now_ms() - t0) / 60);
    }
    free(ix16); free(ix32);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    void *c = mglCompatCreateContext();
    EXPECT_TRUE(c != NULL);
    RUN(compat_thru_bench);
    RUN(compat_thru_idxtype);
    return mgl_test_summary();
}
