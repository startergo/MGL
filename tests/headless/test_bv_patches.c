/* r75: instanced drawElements*BaseVertex with GL_PATCHES on the GPU-scan path.
 *
 * The gather kernel maps an element occurrence to the vertex kernel's output
 * record as [instance * vstride + vertex].  On the GPU-driven (scan) path the
 * vertex kernel's instance stride is the GPU-decided vcount (vsparams[1] =
 * maxv+1), but the CPU bound the gather's vstride to the whole-buffer
 * capacity -- with instances > 1 and basevertex != 0 the later instances read
 * the wrong records (KHR-GL46.draw_elements_base_vertex_tests.AEP_shader_
 * stages regressed at tess r66; fixed by binding vsparams[1] itself).
 *
 * This test forces that path: many vertices (capacity >> the drawn window),
 * a 3-index window at the buffer start, basevertex 10, 3 instances, and
 * compares against the equivalent reference draw with pre-shifted indices. */
#include "mgl_test.h"
#include <string.h>

static const char *kVs =
    "#version 460 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=1) in vec4 a_col;\n"
    "out vec4 v_col;\n"
    "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); v_col = a_col; }\n";
static const char *kTcs =
    "#version 460 core\n"
    "layout(vertices=3) out;\n"
    "in vec4 v_col[];\n"
    "out vec4 tc_col[];\n"
    "void main(){\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "  gl_TessLevelInner[0] = 1.0;\n"
    "}\n";
static const char *kTes =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "in vec4 tc_col[];\n"
    "out vec4 te_col;\n"
    "void main(){\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  te_col = gl_TessCoord.x * tc_col[0] + gl_TessCoord.y * tc_col[1] + gl_TessCoord.z * tc_col[2];\n"
    "}\n";
static const char *kGs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "in vec4 te_col[];\n"
    "out vec4 g_col;\n"
    "void main(){\n"
    "  for (int i = 0; i < 3; i++) { g_col = te_col[i]; gl_Position = gl_in[i].gl_Position; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 460 core\n"
    "in vec4 g_col;\n"
    "layout(location=0) out vec4 o;\n"
    "void main(){ o = g_col; }\n";

#define NV 220
#define BASE 10
static float g_pos[NV][2];
static float g_col[NV][4];

static unsigned long long draw_hash(int mode)
{
    /* mode 0: glDrawElementsInstancedBaseVertex (the GPU-scan gather path --
     *          the regression)
     * mode 1: glDrawArraysInstanced(first = 10) — no index buffer, no gather:
     *          the ground truth (GL: first is the base vertex for arrays)
     * mode 2: glDrawElementsInstanced with pre-shifted indices {10,11,12} */
    static float img[200 * 200 * 4];
    unsigned long long h = 1469598103934665603ULL;
    GLuint ibo, eb[4];

    for (int i = 0; i < 4; i++)
        eb[i] = (GLuint)((i < 3 ? i : 0) + (mode == 2 ? BASE : 0));
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (mode == 1)
        glDrawArraysInstanced(GL_PATCHES, BASE, 3, 3);
    else
    {
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof eb, eb, GL_STATIC_DRAW);
        if (mode == 2)
            glDrawElementsInstanced(GL_PATCHES, 3, GL_UNSIGNED_INT, NULL, 3);
        else
            glDrawElementsInstancedBaseVertex(GL_PATCHES, 3, GL_UNSIGNED_INT, NULL, 3, BASE);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &ibo);
    }
    glFinish();
    glReadPixels(0, 0, 200, 200, GL_RGBA, GL_FLOAT, img);
    for (int i = 0; i < 200 * 200; i++)
        for (int k = 0; k < 4; k++)
        {
            unsigned u;
            memcpy(&u, &img[i*4+k], 4);
            h ^= u; h *= 1099511628211ULL;
        }
    return h;
}

TEST(TessBaseVertexInstancedGpuScan)
{
    const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[5] = { kVs, kTcs, kTes, kGs, kFs };
    GLuint p, vao, b;
    GLint link = 0;
    unsigned long long hb, hr;

    for (int i = 0; i < NV; i++)
    {
        float a = (float)(i * 6 + 1) / 40.0f;
        g_pos[i][0] = 0.92f * cosf(a);
        g_pos[i][1] = 0.92f * sinf(a);
        g_col[i][0] = (i * 37 % 256) / 255.0f;
        g_col[i][1] = (i * 91 % 256) / 255.0f;
        g_col[i][2] = (i * 163 % 256) / 255.0f;
        g_col[i][3] = 1.0f;
    }
    p = glCreateProgram();
    for (int i = 0; i < 5; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof g_pos + sizeof g_col, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof g_pos, g_pos);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof g_pos, sizeof g_col, g_col);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, (void *)sizeof g_pos);
    glEnableVertexAttribArray(1);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);

    hb = draw_hash(0);
    hr = draw_hash(1);
    {
        unsigned long long h2 = draw_hash(2);
        printf("  basevertex=%llu drawArrays=%llu pre-shifted=%llu\n", hb, hr, h2);
        EXPECT_TRUE(hb == hr);
        EXPECT_TRUE(hb == h2);
    }
    /* the deterministic RED: fill the vertex-record region with a DIFFERENT
     * window's data first (indices {100,101,102}); the pre-fix gather read
     * records at inst*capacity -- unwritten ring memory -- rendering black or
     * stale geometry where the correct draw shows the {10,11,12} triangle */
    {
        GLuint ibo, warm[3] = { 100, 101, 102 };
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof warm, warm, GL_STATIC_DRAW);
        glDrawElementsInstanced(GL_PATCHES, 3, GL_UNSIGNED_INT, NULL, 3);
        glFinish();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &ibo);
    }
    hb = draw_hash(0);
    hr = draw_hash(2);
    printf("  after warm-up: basevertex=%llu pre-shifted=%llu %s\n", hb, hr, hb == hr ? "MATCH" : "MISMATCH");
    EXPECT_TRUE(hb == hr);
    glDeleteProgram(p);
}

int main(void) { RUN(TessBaseVertexInstancedGpuScan); return mgl_test_summary(); }
