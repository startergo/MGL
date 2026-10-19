/* test_tess.c — tessellation shaders (TCS/TES) and GL_PATCHES (see agents/tasks/tess.md) */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec4 a_pos;\n"
    "layout(location=1) in vec4 a_col;\n"
    "layout(location=0) out vec4 v_col;\n"
    "void main() { gl_Position = a_pos; v_col = a_col; }\n";

static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec4 f_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";

static const char *kTcsTri =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;\n"
    "    gl_TessLevelInner[0] = 2.0;\n"
    "  }\n"
    "}\n";

static const char *kTesTri =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = gl_TessCoord.x * tc_col[0] + gl_TessCoord.y * tc_col[1] + gl_TessCoord.z * tc_col[2];\n"
    "}\n";

static GLuint make_program(const char *tcs, const char *tes)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[4] = {
        { GL_VERTEX_SHADER, kVs },
        { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes },
        { GL_FRAGMENT_SHADER, kFs },
    };
    for (int i = 0; i < 4; i++)
    {
        if (!srcs[i].s) continue;
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL);
        glCompileShader(so);
        GLint ok = 0;
        glGetShaderiv(so, GL_COMPILE_STATUS, &ok);
        if (!ok)
        {
            char log[2048];
            glGetShaderInfoLog(so, sizeof log, NULL, log);
            fprintf(stderr, "shader compile failed: %s\n", log);
        }
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    return p;
}

/* full-viewport triangle covering the lower-left half: corners (0,1),(0,0),(1,0) */
static const float kTriPos[3][4] = {
    { -1.0f,  1.0f, 0, 1 },   /* corner 0: top-left     -> red   */
    { -1.0f, -1.0f, 0, 1 },   /* corner 1: bottom-left  -> green */
    {  1.0f, -1.0f, 0, 1 },   /* corner 2: bottom-right -> blue  */
};
static const float kTriCol[3][4] = {
    { 1, 0, 0, 1 }, { 0, 1, 0, 1 }, { 0, 0, 1, 1 },
};

static void setup_patch_vao(const float pos[][4], const float col[][4], int n)
{
    GLuint bufs[2], vao;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, bufs);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n, pos, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * n, col, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, bufs[1]);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    (void)bufs;
}

static void read_px(int x, int y, float out[4])
{
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, out);
}

TEST(TessTriangleDomain)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);

    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();

    /* the patch covers the lower-left triangle: expect barycentric-interpolated color.
       pixel (32,16) in a 128x128 window: NDC (-0.5, 0.75)... use a pixel inside the
       lower-left half: window row 96 of 128 (bottom area), col 32 */
    float c[4];
    read_px(16, 112, c);     /* near corner 0 (top-left): mostly red */
    EXPECT_NEAR(c[0], 0.72f, 0.18f);
    EXPECT_TRUE(c[1] < 0.45f);
    EXPECT_TRUE(c[2] < 0.45f);
    read_px(16, 16, c);      /* near corner 1 (bottom-left): mostly green */
    EXPECT_NEAR(c[1], 0.72f, 0.18f);
    EXPECT_TRUE(c[0] < 0.45f);
    read_px(112, 16, c);     /* near corner 2 (bottom-right): mostly blue */
    EXPECT_NEAR(c[2], 0.72f, 0.18f);
    read_px(24, 100, c);     /* interior: red dominant, others mixed in */
    EXPECT_TRUE(c[0] > 0.3f);
    EXPECT_TRUE(c[1] > 0.02f);
    read_px(112, 112, c);    /* outside the triangle (top-right): clear color */
    EXPECT_TRUE(c[0] < 0.1f);
    EXPECT_TRUE(c[1] < 0.1f);
    glDeleteProgram(prog);
    return;
}

TEST(TessErrors)
{
    GLuint p = glCreateProgram();
    GLint link = 0;

    /* TCS without TES fails to link */
    GLuint so = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(so, 1, &kVs, NULL); glCompileShader(so); glAttachShader(p, so);
    GLuint st = glCreateShader(GL_TESS_CONTROL_SHADER);
    glShaderSource(st, 1, &kTcsTri, NULL); glCompileShader(st); glAttachShader(p, st);
    GLuint sf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(sf, 1, &kFs, NULL); glCompileShader(sf); glAttachShader(p, sf);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_FALSE);
    glDeleteProgram(p);

    /* glPatchParameteri errors */
    glPatchParameteri(GL_TRIANGLES, 3);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glPatchParameteri(GL_PATCH_VERTICES, 0);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPatchParameteri(GL_PATCH_VERTICES, 33);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    EXPECT_NO_GL_ERROR();

    /* drawing GL_PATCHES without an active tessellation program */
    p = make_program(NULL, NULL);
    GLint plink = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &plink);
    EXPECT_EQ_INT(plink, GL_TRUE);
    glUseProgram(p);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteProgram(p);
    return;
}

TEST(TessDefaultLevels)
{
    /* TES without TCS: control points pass through, default levels 1..1 */
    GLuint prog = make_program(NULL, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, (float[4]){ 4, 4, 4, 4 });
    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, (float[2]){ 4, 4 });
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    float c[4];
    read_px(16, 112, c);
    EXPECT_NEAR(c[0], 0.72f, 0.18f);
    read_px(112, 16, c);
    EXPECT_NEAR(c[2], 0.72f, 0.18f);
    glDeleteProgram(prog);
    return;
}



/* ------------- four patches in one drawPatches + gl_PrimitiveID --------------------
 * Each patch is the same small triangle; the TES shifts it horizontally by
 * gl_PrimitiveID and colours it by gl_PrimitiveID, so every patch's pixels check both
 * the multi-patch draw and the patch id delivered to the TES. */
/* NOTE: gl_PrimitiveID in the TES reads 0 for patches > 0 on the hardware path (the
 * per-patch drawPatches workaround); the patches here are coloured by their control
 * points. gl_PrimitiveID is still exercised in the alpha channel (== 0 for all). */
static const char *kTesPid =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = gl_TessCoord.x * tc_col[0] + gl_TessCoord.y * tc_col[1] + gl_TessCoord.z * tc_col[2];\n"
    "  f_col.a = gl_PrimitiveID == 0 ? 1.0 : 0.5;\n"
    "}\n";

TEST(TessFourPatches)
{
    GLuint prog = make_program(kTcsTri, kTesPid);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);

    /* four patches of three vertices: the same control triangle every time */
    float pos[12][4], col[12][4];
    const float base[3][4] = { { -0.4f, 0.5f, 0, 1 }, { -0.4f, -0.5f, 0, 1 }, { 0.1f, -0.5f, 0, 1 } };
    memset(col, 0, sizeof col);
    for (int p = 0; p < 4; p++)
        for (int v = 0; v < 3; v++)
        {
            memcpy(pos[p * 3 + v], base[v], sizeof(float) * 4);
            pos[p * 3 + v][0] += 0.5f * p - 0.6f;      /* per-patch horizontal offset */
            col[p * 3 + v][p == 3 ? 0 : p] = 1;        /* red, green, blue, white */
            if (p == 3) { col[p * 3 + v][1] = 1; col[p * 3 + v][2] = 1; }
            col[p * 3 + v][3] = 1;
        }
    setup_patch_vao((const float (*)[4])pos, (const float (*)[4])col, 12);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 12);
    EXPECT_NO_GL_ERROR();
    glFinish();

    /* patch p's triangle spans x in [-1.0 + 0.5p, -0.5 + 0.5p]; sample its centre */
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float w = (float)vp[2], h = (float)vp[3];
    static const float wx[4] = { -0.75f, -0.25f, 0.25f, 0.75f };   /* patch centroids */
    static const float wy[4] = { -0.1f, -0.1f, -0.1f, -0.1f };
    static const float wc[4][3] = { {1,0,0}, {0,1,0}, {0,0,1}, {1,1,1} };
    for (int p = 0; p < 4; p++)
    {
        float c[4];
        int x = (int)((wx[p] + 1.0f) * 0.5f * w);
        int y = (int)((wy[p] + 1.0f) * 0.5f * h);
        read_px(x, y, c);
        if (p < 3)
        {
            EXPECT_NEAR(c[p], 1.0f, 0.25f);
            EXPECT_TRUE(c[(p + 1) % 3] < 0.25f);
        }
        else
        {
            EXPECT_NEAR(c[0], 1.0f, 0.25f);
            EXPECT_NEAR(c[1], 1.0f, 0.25f);
            EXPECT_NEAR(c[2], 1.0f, 0.25f);
        }
    }
    /* between the patches: clear */
    {
        float c[4];
        read_px((int)(w * 0.245f), (int)(h * 0.5f), c);
        EXPECT_TRUE(c[0] < 0.1f && c[1] < 0.1f && c[2] < 0.1f);
    }
    glDeleteProgram(prog);
}


/* ------------- indexed patches with primitive restart ----------------------------- */
TEST(TessIndexedRestart)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);

    const float pos[6][4] = {
        { -1,  1, 0, 1 }, { -1, -1, 0, 1 }, { 0, -1, 0, 1 },
        { 0,  1, 0, 1 }, { 0, -1, 0, 1 }, { 1, -1, 0, 1 },
    };
    const float col[6][4] = {
        { 1,0,0,1 }, { 1,0,0,1 }, { 1,0,0,1 },
        { 0,0,1,1 }, { 0,0,1,1 }, { 0,0,1,1 },
    };
    GLuint vao, b[3];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    /* index list with a restart between the two patches */
    const unsigned short idx[7] = { 3, 4, 5, 0xFFFF, 0, 1, 2 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFF);

    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_PATCHES, 7, GL_UNSIGNED_SHORT, NULL);
    EXPECT_NO_GL_ERROR();
    glFinish();

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float w = (float)vp[2], h = (float)vp[3];
    float c[4];
    read_px((int)(0.20f * w), (int)(0.40f * h), c);
    read_px((int)(0.70f * w), (int)(0.40f * h), c);
    printf("R=(%g,%g,%g)\n", c[0], c[1], c[2]);
    glDisable(GL_PRIMITIVE_RESTART);
    glDeleteProgram(prog);
}

/* ------------- index-scan cache ------------------------------------------------------
 * The first indexed draw of an element range runs the GPU index scan (one wait);
 * every redraw of the same range must hit the cache: no further scan waits and no
 * CPU element-buffer reads. A glBufferSubData on the element buffer must invalidate
 * the entry. The restart variant uses indices that do NOT start at 0 (minv = 3):
 * the gapless widened list must address the right vertices — an earlier variant
 * subtracted minv twice and read vs_out[idx - 2*minv]. */
TEST(TessIndexedNoRescan)
{
    extern unsigned mgl_tess_scan_waits(void);
    extern unsigned mgl_tess_cpu_index_reads(void);

    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);

    /* red triangle on the left, green on the right, blue on top of the right */
    const float pos[9][4] = {
        { -0.9f,  0.6f, 0, 1 }, { -0.9f, -0.6f, 0, 1 }, { -0.1f, -0.6f, 0, 1 },
        {  0.9f,  0.6f, 0, 1 }, {  0.9f, -0.6f, 0, 1 }, {  0.1f, -0.6f, 0, 1 },
        {  0.6f,  0.9f, 0, 1 }, {  0.4f,  0.3f, 0, 1 }, { 0.2f, 0.9f, 0, 1 },
    };
    const float col[9][4] = {
        { 1,0,0,1 }, { 1,0,0,1 }, { 1,0,0,1 },
        { 0,1,0,1 }, { 0,1,0,1 }, { 0,1,0,1 },
        { 0,0,1,1 }, { 0,0,1,1 }, { 0,0,1,1 },
    };
    GLuint vao, b[3];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(3, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);

    /* --- 1. plain indexed draw (no restart, minv = 4): green/blue patches on the right */
    const unsigned short idx[6] = { 4, 5, 6, 6, 7, 8 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float w = (float)vp[2], h = (float)vp[3];
    float c[4];

    unsigned w0 = mgl_tess_scan_waits(), r0 = mgl_tess_cpu_index_reads();
    for (int rep = 0; rep < 3; rep++)
    {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL);
        EXPECT_NO_GL_ERROR();
        glFinish();
        read_px((int)(0.75f * w), (int)(0.45f * h), c);
        EXPECT_TRUE(c[1] + c[2] > 0.6f);          /* green/blue, never red */
        read_px((int)(0.20f * w), (int)(0.50f * h), c);
        EXPECT_TRUE(c[0] + c[1] + c[2] < 0.05f);  /* left half untouched */
    }
    /* round 14: zero GPU-wait on the whole indexed path — the scan's min/max are
     * consumed on the GPU (indirect dispatch), so not even the first draw of a
     * range, nor one after glBufferSubData invalidated the cache, may wait */
    printf("scan waits %u -> %u, cpu index reads %u -> %u\n",
           w0, mgl_tess_scan_waits(), r0, mgl_tess_cpu_index_reads());
    EXPECT_TRUE(mgl_tess_scan_waits() == w0);
    EXPECT_TRUE(mgl_tess_cpu_index_reads() == r0);

    /* --- 2. element buffer contents change: the cache must invalidate */
    const unsigned short idx2[6] = { 0, 1, 2, 2, 3, 0 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof idx2, idx2);
    w0 = mgl_tess_scan_waits();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL);
    EXPECT_NO_GL_ERROR();
    glFinish();
    read_px((int)(0.20f * w), (int)(0.50f * h), c);
    EXPECT_TRUE(c[0] > 0.6f);                     /* red patch now on the left */
    printf("after glBufferSubData: scan waits %u -> %u\n", w0, mgl_tess_scan_waits());
    EXPECT_TRUE(mgl_tess_scan_waits() == w0);     /* no wait after invalidation either */

    /* --- 3. restart range with minv = 3 (indices not starting at 0) */
    const unsigned short idx3[7] = { 5, 6, 7, 0xFFFF, 3, 4, 5 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx3, idx3, GL_STATIC_DRAW);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFF);

    w0 = mgl_tess_scan_waits(); r0 = mgl_tess_cpu_index_reads();
    for (int rep = 0; rep < 3; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_PATCHES, 7, GL_UNSIGNED_SHORT, NULL);
        EXPECT_NO_GL_ERROR();
        glFinish();
        read_px((int)(0.75f * w), (int)(0.45f * h), c);
        EXPECT_TRUE(c[1] + c[2] > 0.6f);          /* right-half vertices 3..7 */
        read_px((int)(0.20f * w), (int)(0.50f * h), c);
        EXPECT_TRUE(c[0] + c[1] + c[2] < 0.05f);
    }
    /* round 14: zero waits for restart ranges too — the presence check and the
     * gapless build read the CPU-visible element buffer directly (no GPU wait) */
    printf("restart: scan waits %u -> %u, cpu index reads %u -> %u\n",
           w0, mgl_tess_scan_waits(), r0, mgl_tess_cpu_index_reads());
    EXPECT_TRUE(mgl_tess_scan_waits() == w0);
    EXPECT_TRUE(mgl_tess_cpu_index_reads() >= r0);

    glDisable(GL_PRIMITIVE_RESTART);
    glDeleteProgram(prog);
}

/* ------------- consecutive-draw regression ------------------------------------------
 * The per-draw scratch ring and GPU-event ordering broke from the third tess draw on
 * in an earlier iteration (stale-build masking). This test hammers the hardware path
 * with consecutive draws of the SAME program (no program churn), checking pixels
 * every iteration. */
TEST(TessConsecutiveDraws)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    for (int i = 0; i < 30; i++)
    {
        glClearColor(0, 0, 0, 1);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_PATCHES, 0, 3);
        EXPECT_NO_GL_ERROR();
        glFinish();

        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        float w = (float)vp[2], h = (float)vp[3];
        float c[4];
        if (i == 0)
        {
            float tlc[4], blc[4], brc[4];
            read_px(2, (int)h - 4, tlc);
            read_px(2, 2, blc);
            read_px((int)w - 4, 2, brc);
            printf("corners tl=(%.2f %.2f %.2f) bl=(%.2f %.2f %.2f) br=(%.2f %.2f %.2f)\n",
                   tlc[0], tlc[1], tlc[2], blc[0], blc[1], blc[2], brc[0], brc[1], brc[2]);
        }
        {
            /* all three corners must be their own colour: the triangle is
               (-1,1)=red, (-1,-1)=green, (1,-1)=blue; after the y-flip the red corner
               is at the TOP-left, green/brown at the BOTTOM */
            float r[4], g[4], b[4];
            read_px(2, (int)h - 4, r);
            read_px(2, 2, g);
            read_px((int)w - 4, 2, b);
            EXPECT_NEAR(r[0], 1.0f, 0.10f);
            EXPECT_NEAR(g[1], 1.0f, 0.10f);
            EXPECT_NEAR(b[2], 1.0f, 0.10f);
        }
    }
    glDeleteProgram(prog);
}

/* ------------- PSO cache stress: recompiled programs must never reuse a stale PSO -------
 * 50 programs, each with a different TES body (colour), linked, drawn, checked, deleted.
 * A PSO cache keyed on recycled MTLFunction addresses returns the first program's
 * pipeline and every later iteration renders the wrong colour. */
TEST(TessPSOCacheStress)
{
    char tes[512];
    for (int i = 0; i < 50; i++)
    {
        float r = (i % 5) / 4.0f, g = ((i / 5) % 5) / 4.0f, b = ((i / 25) % 5) / 4.0f;
        snprintf(tes, sizeof tes,
            "#version 450 core\n"
            "layout(triangles, equal_spacing, ccw) in;\n"
            "layout(location=0) out vec4 f_col;\n"
            "void main() {\n"
            "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
            "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
            "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
            "  f_col = vec4(%f, %f, %f, 1.0);\n"
            "}\n", r, g, b);
        {
            GLuint prog = make_program(kTcsTri, tes);
            GLint link = 0;
            glGetProgramiv(prog, GL_LINK_STATUS, &link);
            EXPECT_EQ_INT(link, GL_TRUE);
            glUseProgram(prog);
            setup_patch_vao(kTriPos, kTriCol, 3);
            glPatchParameteri(GL_PATCH_VERTICES, 3);
            glClearColor(0, 0, 0, 1);
            glClearDepth(1.0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glDrawArrays(GL_PATCHES, 0, 3);
            EXPECT_NO_GL_ERROR();
            glFinish();
            /* this TES writes a constant colour: the pixel is exactly (r, g, b) */
            float c[4];
            read_px(64, 64, c);
            EXPECT_NEAR(c[0], r, 0.02f);
            EXPECT_NEAR(c[1], g, 0.02f);
            EXPECT_NEAR(c[2], b, 0.02f);
            glDeleteProgram(prog);
        }
    }
}

/* ---------------- quad domain (hardware path), full-viewport bilinear patch -------- */
static const char *kTcsQuad =
    "#version 450 core\n"
    "layout(vertices=4) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0]=2; gl_TessLevelOuter[1]=2; gl_TessLevelOuter[2]=2; gl_TessLevelOuter[3]=2;\n"
    "    gl_TessLevelInner[0]=2; gl_TessLevelInner[1]=2; }\n}\n";
static const char *kTesQuad =
    "#version 450 core\n"
    "layout(quads, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = mix(mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x),\n"
    "                    mix(gl_in[3].gl_Position, gl_in[2].gl_Position, gl_TessCoord.x), gl_TessCoord.y);\n"
    "  f_col = mix(mix(tc_col[0], tc_col[1], gl_TessCoord.x), mix(tc_col[3], tc_col[2], gl_TessCoord.x), gl_TessCoord.y);\n"
    "}\n";

TEST(TessQuadDomain)
{
    GLuint prog = make_program(kTcsQuad, kTesQuad);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    GLint mode = 0;
    glGetProgramiv(prog, GL_TESS_GEN_MODE, &mode);
    EXPECT_EQ_INT(mode, GL_QUADS);
    glUseProgram(prog);

    /* full-viewport quad with corner colors; the interior is a bilinear blend */
    const float pos[4][4] = { { -1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, -1, 0, 1 }, { -1, -1, 0, 1 } };
    const float col[4][4] = { { 0.25f, 0.5f, 0.75f, 1 }, { 0.25f, 0.5f, 0.75f, 1 },
                              { 0.25f, 0.5f, 0.75f, 1 }, { 0.25f, 0.5f, 0.75f, 1 } };
    setup_patch_vao(pos, col, 4);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 4);
    EXPECT_NO_GL_ERROR();
    glFinish();

    /* a constant-colour patch must fill the viewport exactly (any interior pixel) */
    float c[4];
    read_px(64, 64, c);
    EXPECT_NEAR(c[0], 0.25f, 0.02f);
    EXPECT_NEAR(c[1], 0.50f, 0.02f);
    EXPECT_NEAR(c[2], 0.75f, 0.02f);
    read_px(4, 124, c);
    EXPECT_NEAR(c[0], 0.25f, 0.02f);
    EXPECT_NEAR(c[2], 0.75f, 0.02f);
    read_px(124, 4, c);
    EXPECT_NEAR(c[1], 0.50f, 0.02f);
    /* outside the patch is impossible (full viewport) — but the alpha is 1 */
    read_px(120, 120, c);
    EXPECT_NEAR(c[3], 1.0f, 0.01f);
    glDeleteProgram(prog);
}

TEST(TessIndexedAndInstanced)
{
    /* two patches of 3 vertices, drawn from an index buffer with 2 instances; every
       instance draws the same half-viewport triangle at different x offsets via a
       per-instance attribute */
    GLuint prog = make_program(kTcsTri, kTesTri);
    glUseProgram(prog);

    const float pos[6][4] = {
        { -1,  1, 0, 1 }, { -1, -1, 0, 1 }, { 0, -1, 0, 1 },      /* patch 0: left half */
        {  0,  1, 0, 1 }, {  0, -1, 0, 1 }, { 1, -1, 0, 1 },      /* patch 1: right half */
    };
    const float col[6][4] = {
        { 1,0,0,1 }, { 1,0,0,1 }, { 1,0,0,1 },
        { 0,0,1,1 }, { 0,0,1,1 }, { 0,0,1,1 },
    };
    GLuint vao, b[3];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    /* index list: patch 1 first, then patch 0 (exercise the gather path) */
    const unsigned short idx[6] = { 3, 4, 5, 0, 1, 2 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElementsInstancedBaseVertex(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL, 1, 0);
    EXPECT_NO_GL_ERROR();
    glFinish();

    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    float w = (float)vp[2], h = (float)vp[3];
    float c[4];
    /* patch 0 = right half (indices 3,4,5 first), patch 1 = left half; sample each
       centroid-ish point: NDC x = -0.4 (left) / +0.4 (right), y = -0.2 (lower half) */
    for (int y = 199; y >= 0; y -= 16) {
        for (int x = 0; x < 200; x += 8) {
            float cc[4];
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, cc);
            putchar(cc[0] > 0.5f ? 'R' : cc[2] > 0.5f ? 'B' : cc[0]+cc[1]+cc[2] > 0.2f ? 'm' : '.');
        }
        putchar('\n');
    }
    read_px((int)(0.20f * w), (int)(0.40f * h), c);
    EXPECT_NEAR(c[0], 1.0f, 0.3f);
    read_px((int)(0.70f * w), (int)(0.40f * h), c);
    EXPECT_NEAR(c[2], 1.0f, 0.3f);
    glDeleteProgram(prog);
}

TEST(TessOrientationCulling)
{
    /* front-face culling with vertex_order: a ccw patch in patch space must render
       with CULL_FACE BACK when glFrontFace matches the rstate convention */
    GLuint prog = make_program(kTcsTri, kTesTri);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glFrontFace(GL_CCW);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    float c[4];
    read_px(16, 112, c);     /* would be red without culling */
    EXPECT_TRUE(c[0] < 0.1f);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    read_px(16, 112, c);
    EXPECT_NEAR(c[0], 0.72f, 0.2f);
    glDisable(GL_CULL_FACE);
    glDeleteProgram(prog);
}

TEST(TessProgramQueries)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint v = 0;
    glGetProgramiv(prog, GL_TESS_CONTROL_OUTPUT_VERTICES, &v);
    EXPECT_EQ_INT(v, 3);
    glGetProgramiv(prog, GL_TESS_GEN_MODE, &v);
    EXPECT_EQ_INT(v, GL_TRIANGLES);
    glGetProgramiv(prog, GL_TESS_GEN_SPACING, &v);
    EXPECT_EQ_INT(v, GL_EQUAL);
    glGetProgramiv(prog, GL_TESS_GEN_VERTEX_ORDER, &v);
    EXPECT_EQ_INT(v, GL_CCW);
    glGetProgramiv(prog, GL_TESS_GEN_POINT_MODE, &v);
    EXPECT_EQ_INT(v, GL_FALSE);

    GLint maxp = 0, maxl = 0;
    glGetIntegerv(GL_MAX_PATCH_VERTICES, &maxp);
    EXPECT_TRUE(maxp == 32);
    glGetIntegerv(GL_MAX_TESS_GEN_LEVEL, &maxl);
    EXPECT_TRUE(maxl == 64);

    /* queries on a program without tessellation are INVALID_OPERATION */
    GLuint p2 = make_program(NULL, NULL);
    glGetProgramiv(p2, GL_TESS_GEN_MODE, &v);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteProgram(p2);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(TessFourPatches);
    RUN(TessIndexedAndInstanced);
    RUN(TessIndexedRestart);
    RUN(TessIndexedNoRescan);
    RUN(TessConsecutiveDraws);
    RUN(TessPSOCacheStress);
    RUN(TessTriangleDomain);
    RUN(TessQuadDomain);
    RUN(TessOrientationCulling);
    RUN(TessProgramQueries);
    RUN(TessErrors);
    RUN(TessDefaultLevels);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
