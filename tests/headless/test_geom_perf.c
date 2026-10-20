#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IDX_COUNT  (300 * 1000)
#define VERTS      (100 * 1000)
#define FRAMES     40

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    GLint ok = 0;
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    (void)&ok;
    return s;
}

/* timing of the geometry emulation's per-draw cost: the primitive assembly runs on the
 * GPU (a prep kernel) unless MGL_GEOM_CPU_ASSEMBLY forces the CPU path.  The test draws a
 * large indexed mesh through a geometry shader many times and reports, per case, the wall
 * time of the draws between two glFinish()s (what an application sees) and the GPU time
 * measured with a GL_TIME_ELAPSED query around the same batch.  The env knobs the driver
 * reads are cached at the FIRST draw of the process, so they must be set before the test
 * starts: MGL_GEOM_SERIAL_PREP=1 replaces the tile-parallel restart prep with the serial
 * run walk, MGL_GEOM_WORST_CASE_DISPATCH=1 dispatches the geometry kernel with the
 * worst-case (count / in_verts) thread count instead of the exact count the prep kernel
 * computed, MGL_GEOM_TILED_MIN=n forces the tile-parallel chain for draws > n indices. */

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double gpu_time_frames(GLsizei count)
{
    GLuint tq = 0;
    GLuint64 ns = 0;

    glGenQueries(1, &tq);
    glBeginQuery(GL_TIME_ELAPSED, tq);
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TIME_ELAPSED);
    glGetQueryObjectui64v(tq, GL_QUERY_RESULT, &ns);   /* no glFinish: the resolve waits */
    glDeleteQueries(1, &tq);
    return (double)ns / 1000.0 / FRAMES;   /* us per frame */
}

static void run_frames(const char *label, GLsizei count, GLuint query)
{
    double t0, t1, gpu;
    GLint primitives = 0;

    /* wall time between two finishes */
    glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, query);
    t0 = now();
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL);
    glFinish();
    t1 = now();
    glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
    glGetQueryObjectiv(query, GL_QUERY_RESULT, &primitives);
    EXPECT_TRUE(primitives > 0);

    /* GPU time of the same batch (dominated by the prep / assembly chain) */
    gpu = gpu_time_frames(count);

    /* NB: the emitted count below is the CPU-side approximation the query domain uses
     * (count / in_verts): with primitive restart the real count is lower (count/6 here,
     * verified exactly via transform feedback in test_geom's tiled_restart_gpu_assembly);
     * the exact GPU counters are only plumbed into the xfb queries, not this one */

    printf("  %-28s %9.1f us/frame wall, %9.1f us/frame GPU, %d primitives emitted\n",
           label, (t1 - t0) * 1e6 / FRAMES, gpu, primitives);
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

    for (int i = 0; i < VERTS; i++)
    {
        verts[i * 2 + 0] = (float)(i % 977);
        verts[i * 2 + 1] = (float)((i * 31) % 977);
    }
    for (int i = 0; i < IDX_COUNT; i++)
        idx[i] = (GLuint)(((GLuint)i * 7919u) % VERTS);  /* scattered, restart-free; unsigned:
                                                   i * 7919 overflows int for i >= 271182 */

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

    printf("geometry shader primitive assembly (300k scattered indices, %d frames):\n", FRAMES);

    /* the wall number below is the GPU path (set MGL_GEOM_CPU_ASSEMBLY=1 in the
     * environment before the process starts to A/B the CPU assembly path) */
    run_frames("assembly (no restart)", IDX_COUNT, q);
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

    /* primitive restart: the real primitive count (count/6 with a restart every 6th index)
     * is only known after the restart scan, so the prep chain (tiles -> prefix -> runs ->
     * reduce -> emit) runs instead of the single no-restart prep kernel; the geometry
     * kernel itself still dispatches with the exact count that chain produced.  The env
     * knobs (worst-case dispatch, serial prep) are cached at the first draw, so they only
     * take effect when set before the process starts. */
    {
        const GLuint restart = 0xffffffffu;
        glEnable(GL_PRIMITIVE_RESTART);
        glPrimitiveRestartIndex(restart);
        for (int i = 5; i < IDX_COUNT; i += 6) idx[i] = restart;
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * IDX_COUNT, idx, GL_STATIC_DRAW);
        printf("primitive restart (every 6th index):\n");
        run_frames("restart", IDX_COUNT, q);
        for (int i = 0; i < IDX_COUNT; i++) idx[i] = (GLuint)(((GLuint)i * 7919u) % VERTS);
        glDisable(GL_PRIMITIVE_RESTART);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * IDX_COUNT, idx, GL_STATIC_DRAW);
    }

    /* indirect draws: the prep kernel reads the command from the app's buffer */
    {
        GLuint ib = 0;
        const GLuint args[5] = { IDX_COUNT, 1, 0, 0, 0 };   /* count, inst, first, basevertex, baseinst */
        double t0, t1;
        GLint prims = 0;

        glGenBuffers(1, &ib);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, ib);
        glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof args, args, GL_STATIC_DRAW);

        glBeginQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED, q);
        t0 = now();
        for (int f = 0; f < FRAMES; f++)
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
        glFinish();
        t1 = now();
        glEndQuery(GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &prims);
        EXPECT_TRUE(prims > 0);
        {
            GLuint tq = 0;
            GLuint64 ns = 0;
            glGenQueries(1, &tq);
            glBeginQuery(GL_TIME_ELAPSED, tq);
            for (int f = 0; f < FRAMES; f++)
                glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL);
            glEndQuery(GL_TIME_ELAPSED);
            glGetQueryObjectui64v(tq, GL_QUERY_RESULT, &ns);
            glDeleteQueries(1, &tq);
            printf("  %-28s %9.1f us/frame wall, %9.1f us/frame GPU, %d primitives emitted\n",
                   "indirect", (t1 - t0) * 1e6 / FRAMES, (double)ns / 1000.0 / FRAMES, prims);
        }

        glDeleteBuffers(1, &ib);
    }

    free(idx);
    free(verts);
    return mgl_test_summary();
}
