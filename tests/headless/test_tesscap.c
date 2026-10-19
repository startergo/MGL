#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Round 15, step 2 measurement: the GPU-driven indexed path dispatches the vertex
 * kernel over the full attribute capacity x instances (the GPU-written vcount guards
 * the real work). What does that cost when a tiny draw (2 patches, 6 indices) comes
 * from a 1M-vertex buffer, compared with a 6-vertex buffer? Reported as ms/frame
 * between two glFinish()s, 200 frames each, and the ratio. */

#define FRAMES   200
#define BIG_VERTS (1000 * 1000)

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec4 a_pos;\n"
    "layout(location=1) in vec4 a_col;\n"
    "layout(location=0) out vec4 v_col;\n"
    "void main() { gl_Position = a_pos; v_col = a_col; }\n";
static const char *kTcs =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0]=2.0; gl_TessLevelOuter[1]=2.0; gl_TessLevelOuter[2]=2.0;\n"
    "    gl_TessLevelInner[0]=2.0; }\n"
    "}\n";
static const char *kTes =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = tc_col[0];\n"
    "}\n";
static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec4 f_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";

static double time_frames(int frames)
{
    double t0 = now();
    for (int f = 0; f < frames; f++)
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
    glFinish();
    return (now() - t0) * 1000.0 / frames;
}

TEST(IndexedCapacityCost)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[4] = {
        { GL_VERTEX_SHADER, kVs }, { GL_TESS_CONTROL_SHADER, kTcs },
        { GL_TESS_EVALUATION_SHADER, kTes }, { GL_FRAGMENT_SHADER, kFs } };
    for (int i = 0; i < 4; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL); glCompileShader(so); glAttachShader(p, so); glDeleteShader(so);
    }
    glLinkProgram(p);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);

    GLuint vao, b[3];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(3, b);

    /* the big buffer: 1M vertices; the drawn 6 indices reference only vertices
       900000..900005, so the capacity-based dispatch covers a million threads */
    float *pos = malloc((size_t)BIG_VERTS * 16);
    float *col = malloc((size_t)BIG_VERTS * 16);
    for (int i = 0; i < BIG_VERTS; i++)
    {
        pos[i * 4 + 0] = -1.0f + 2.0f * ((i % 977) / 977.0f);
        pos[i * 4 + 1] = -1.0f + 2.0f * ((i % 733) / 733.0f);
        pos[i * 4 + 2] = 0.0f;
        pos[i * 4 + 3] = 1.0f;
        col[i * 4 + 0] = 1.0f; col[i * 4 + 1] = 0.5f; col[i * 4 + 2] = 0.25f; col[i * 4 + 3] = 1.0f;
    }
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, (size_t)BIG_VERTS * 16, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, (size_t)BIG_VERTS * 16, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    const unsigned int idx[6] = { 900000u, 900001u, 900002u, 900003u, 900004u, 900005u };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* warm-up both PSOs and caches, then measure */
    time_frames(20);
    double t_big = time_frames(FRAMES);

    /* same draw from a 6-vertex buffer (min + 0): the dispatch covers 6 threads */
    {
        float spos[6][4] = {
            { -1, 1, 0, 1 }, { -1, -1, 0, 1 }, { 0, -1, 0, 1 },
            {  0, 1, 0, 1 }, {  0, -1, 0, 1 }, { 1, -1, 0, 1 },
        };
        float scol[6][4] = { {1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1} };
        GLuint sb[2];
        glGenBuffers(2, sb);
        glBindBuffer(GL_ARRAY_BUFFER, sb[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof spos, spos, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, sb[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof scol, scol, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        const unsigned int sidx[6] = { 0, 1, 2, 3, 4, 5 };
        GLuint eb;
        glGenBuffers(1, &eb);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof sidx, sidx, GL_STATIC_DRAW);
        time_frames(20);
        double t_small = time_frames(FRAMES);
        printf("indexed tiny-draw cost: big-buffer %.4f ms/frame, small-buffer %.4f ms/frame, ratio %.2fx\n",
               t_big, t_small, t_small > 0 ? t_big / t_small : 0.0);
        glDeleteBuffers(2, sb);
        glDeleteBuffers(1, &eb);
    }
    free(pos);
    free(col);
    glDeleteProgram(p);
}

int main(void) { RUN(IndexedCapacityCost); return mgl_test_summary(); }
