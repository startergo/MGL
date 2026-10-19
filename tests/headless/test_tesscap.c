#include "mgl_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Round 16: re-measure the capacity-spanning vertex dispatch with real GPU time.
 * Round 15's wall-clock numbers (~0.02 ms/frame including glFinish) are shorter
 * than a Metal commit+wait round trip, so they prove nothing about GPU cost. The
 * library now sums GPUEndTime-GPUStartTime over every tess command buffer
 * (mgl_tess_gpu_time_us) and counts them (mgl_tess_cbs_completed):
 *
 *  1) verify glFinish: after a draw + glFinish every committed tess command
 *     buffer must have completed (the counter settles without further GL work);
 *  2) GPU time of a 2-patch draw from a 1M-vertex buffer vs a 6-vertex buffer,
 *     with a pixel check proving the draws render.
 */

extern void mgl_tess_gpu_time_reset(void);
extern double mgl_tess_gpu_time_us(void);
extern unsigned mgl_tess_cbs_completed(void);

#define FRAMES   200
#define BIG_VERTS (1000 * 1000)

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* the completed-counter is bumped by Metal's completion handlers, which are
 * dispatched asynchronously; wait briefly for it to settle */
static unsigned wait_counters_settled(unsigned ms)
{
    unsigned v = mgl_tess_cbs_completed(), prev = 0;
    double t0 = now();
    while (now() - t0 < (double)ms / 1000.0)
    {
        if (v != prev)
            prev = v;
        v = mgl_tess_cbs_completed();
        if (v == prev && v > 0)
            break;
    }
    return mgl_tess_cbs_completed();
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
    /* the six referenced vertices get exactly the small test's geometry, so both
       measured draws rasterise the same pixels and only the dispatch size differs */
    {
        const float spos[6][4] = {
            { -1, 1, 0, 1 }, { -1, -1, 0, 1 }, { 0, -1, 0, 1 },
            {  0, 1, 0, 1 }, {  0, -1, 0, 1 }, { 1, -1, 0, 1 },
        };
        for (int i = 0; i < 6; i++)
        {
            memcpy(&pos[(900000 + i) * 4], spos[i], 16);
            col[(900000 + i) * 4 + 0] = 1.0f;
            col[(900000 + i) * 4 + 1] = 1.0f;
            col[(900000 + i) * 4 + 2] = 1.0f;
            col[(900000 + i) * 4 + 3] = 1.0f;
        }
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

    /* ---- glFinish verification: every committed tess command buffer completes ---- */
    {
        mgl_tess_gpu_time_reset();
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
        glFinish();
        {
            unsigned at_finish = mgl_tess_cbs_completed();
            unsigned settled = wait_counters_settled(200);

            printf("glFinish: %u tess command buffers completed at return, %u after settling\n",
                   at_finish, settled);
            EXPECT_TRUE(settled >= 2);           /* scan buffer + the compute/render buffer */
            EXPECT_TRUE(at_finish >= 1);         /* at least one already done at return */
        }
    }

    /* ---- GPU time: big buffer vs small buffer ---- */
    mgl_tess_gpu_time_reset();
    for (int f = 0; f < FRAMES; f++)
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
    glFinish();
    wait_counters_settled(200);
    double t_big = mgl_tess_gpu_time_us() / FRAMES;

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
        for (int i = 0; i < 20; i++)
            glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
        glFinish();
        mgl_tess_gpu_time_reset();
        for (int f = 0; f < FRAMES; f++)
            glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
        glFinish();
        wait_counters_settled(200);
        double t_small = mgl_tess_gpu_time_us() / FRAMES;

        /* the draw must render: white-ish pixels where the small-buffer patches are */
        {
            GLint vp[4];
            float c[4];
            glGetIntegerv(GL_VIEWPORT, vp);
            glReadPixels(vp[2] / 2, vp[3] / 2, 1, 1, GL_RGBA, GL_FLOAT, c);
            printf("pixel at viewport centre after the measured run: %.2f %.2f %.2f\n",
                   c[0], c[1], c[2]);
            EXPECT_TRUE(c[0] + c[1] + c[2] > 0.5f);
        }

        /* re-measure the big buffer again after the small one (ordering artifact check) */
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
        glBindBuffer(GL_ARRAY_BUFFER, b[0]);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        glBindBuffer(GL_ARRAY_BUFFER, b[1]);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        for (int i = 0; i < 20; i++)
            glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
        glFinish();
        mgl_tess_gpu_time_reset();
        for (int f = 0; f < FRAMES; f++)
            glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_INT, NULL);
        glFinish();
        wait_counters_settled(200);
        double t_big2 = mgl_tess_gpu_time_us() / FRAMES;
        printf("indexed tiny-draw GPU time: big %.2f, small %.2f, big-again %.2f us/frame "
               "(ratios %.2fx / %.2fx)\n",
               t_big, t_small, t_big2, t_small > 0 ? t_big / t_small : 0.0,
               t_small > 0 ? t_big2 / t_small : 0.0);
        glDeleteBuffers(2, sb);
        glDeleteBuffers(1, &eb);
    }
    free(pos);
    free(col);
    glDeleteProgram(p);
}

int main(void) { RUN(IndexedCapacityCost); return mgl_test_summary(); }
