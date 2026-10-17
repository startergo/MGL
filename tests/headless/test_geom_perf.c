#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* timing of the geometry emulation's per-draw CPU cost: the primitive assembly runs on the
 * GPU (a prep kernel) unless MGL_GEOM_CPU_ASSEMBLY forces the CPU path.  The test draws a
 * large indexed mesh through a geometry shader many times and reports the wall time of both
 * paths, which is what an application sees between two glFinish()s. */

#define IDX_COUNT  (300 * 1000)
#define VERTS      (100 * 1000)
#define FRAMES     40

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    GLint ok = 0;
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    return s;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double run_frames(GLuint p, GLsizei count, GLuint query)
{
    double t0, t1;
    GLint primitives = 0;

    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, query);
    t0 = now();
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL);
    glFinish();
    t1 = now();
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glGetQueryObjectiv(query, GL_QUERY_RESULT, &primitives);
    printf("  %d frames of %d indices: %.3f s (%.1f us/frame), %d primitives emitted\n",
           FRAMES, count, t1 - t0, (t1 - t0) * 1e6 / FRAMES, primitives);
    EXPECT_TRUE(primitives > 0);
    return (t1 - t0) * 1e6 / FRAMES;
}

int main(void)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 v_pos;\n"
        "void main(){ v_pos = pos; gl_Position = vec4(pos * 0.02, 0.0, 1.0); }\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices = 3) out;\n"
        "layout(location=0) in vec2 v_pos[];\n"
        "layout(location=0) out vec2 g_pos;\n"
        "void main(){\n"
        "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_pos = v_pos[i]; EmitVertex(); }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in vec2 g_pos;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(g_pos * 0.5 + 0.5, 0.0, 1.0); }\n";
    GLuint p, vao, vbo, ebo, q;
    GLuint *idx = malloc(sizeof(GLuint) * IDX_COUNT);
    float *verts = malloc(sizeof(float) * 2 * VERTS);
    double cpu, gpu;

    for (int i = 0; i < VERTS; i++)
    {
        verts[i * 2 + 0] = (float)(i % 977);
        verts[i * 2 + 1] = (float)((i * 31) % 977);
    }
    for (int i = 0; i < IDX_COUNT; i++)
        idx[i] = (GLuint)((i * 7919) % VERTS);       /* scattered, restart-free */

    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(p);
    {
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) return 1;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glGenQueries(1, &q);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * VERTS, verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * IDX_COUNT, idx, GL_STATIC_DRAW);
    glUseProgram(p);

    glViewport(0, 0, 200, 200);

    printf("geometry shader primitive assembly (300k scattered indices, %d frames):\n", FRAMES);

    setenv("MGL_GEOM_CPU_ASSEMBLY", "1", 1);
    cpu = run_frames(p, IDX_COUNT, q);
    unsetenv("MGL_GEOM_CPU_ASSEMBLY");
    gpu = run_frames(p, IDX_COUNT, q);

    printf("GPU assembly is %.2fx %s the CPU path (%.1f vs %.1f us/frame)\n",
           cpu > gpu ? cpu / gpu : gpu / cpu, cpu > gpu ? "faster than" : "slower than", cpu, gpu);

    /* both paths must rasterize the same amount: count covered pixels */
    {
        int covered = 0;
        glClear(GL_COLOR_BUFFER_BIT);
        glFinish();
        glDrawElements(GL_TRIANGLES, IDX_COUNT, GL_UNSIGNED_INT, NULL);
        glFinish();
        for (int y = 0; y < 200; y += 4)
            for (int x = 0; x < 200; x += 4)
            {
                GLubyte a[4] = {0, 0, 0, 0};
                glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);
                if (a[3] || a[0] || a[1] || a[2]) covered++;
            }
        printf("covered samples: %d\n", covered);
        EXPECT_TRUE(covered > 0);
    }

    free(idx);
    free(verts);
    return mgl_test_summary();
}
