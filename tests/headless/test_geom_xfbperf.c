// Timing of the geometry emulation's transform feedback pack: CPU (sync + memcpy) vs the
// GPU pack kernels.  Run with MGL_GEOM_GPU_XFB=1 to time the GPU path.
#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FRAMES 200

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    return s;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = vec4(0.0); }\n";
    GLuint p, vao, vbo, ebo, tbuf, q;
    GLuint *idx = malloc(sizeof(GLuint) * 30000);
    double t0, t1;
    GLint written = 0;

    for (int i = 0; i < 30000; i++) idx[i] = (GLuint)((i * 7919) % 100000);

    p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
    {
        static const char *vars[2] = { "g_pos", NULL };
        glTransformFeedbackVaryings(p, 1, vars, GL_INTERLEAVED_ATTRIBS);
    }
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
    glGenBuffers(1, &tbuf);
    glGenQueries(1, &q);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    {
        static float verts[200000];
        for (int i = 0; i < 100000; i++) { verts[i * 2] = (float)(i % 977); verts[i * 2 + 1] = (float)((i * 31) % 977); }
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    }
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 30000 * 4, idx, GL_STATIC_DRAW);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, tbuf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8 << 20, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbuf);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);

    /* one block with many draws: the CPU pack syncs after every draw, the GPU pack only
     * resolves the counters once at the end */
    glBeginTransformFeedback(GL_TRIANGLES);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    t0 = now();
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_TRIANGLES, 30000, GL_UNSIGNED_INT, NULL);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndTransformFeedback();
    t1 = now();

    glGetQueryObjectiv(q, GL_QUERY_RESULT, &written);
    printf("%s pack: %d frames, %d primitives written, %.3f s (%.1f us/frame)\n",
           getenv("MGL_GEOM_GPU_XFB") ? "GPU" : "CPU", FRAMES, written, t1 - t0,
           (t1 - t0) * 1e6 / FRAMES);
    EXPECT_TRUE(written > 0);

    free(idx);
    return mgl_test_summary();
}
