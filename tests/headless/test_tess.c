/* test_tess.c — tessellation shaders (TCS/TES) and GL_PATCHES (see agents/tasks/tess.md) */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <dlfcn.h>
#include "glcorearb.h"
#include "mgl_test.h"
#include "mgl_tess_diag.h"

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

static GLuint make_program_full(const char *vs, const char *tcs, const char *tes)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[4] = {
        { GL_VERTEX_SHADER, vs },
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

static GLuint make_program(const char *tcs, const char *tes)
{
    return make_program_full(kVs, tcs, tes);
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
    glViewport(0, 0, 200, 200);   /* these pixel checks assume a 200x200 window */
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

/* ------------------------------------------------------------------ canonical records
 * The control point / per-patch records must be laid out by GL varying location for
 * every stage of the pipeline: a TES that declares but never uses a varying, declares
 * its inputs in a different order than the TCS writes them, per-patch varyings with
 * different names, or arrayed varyings must never shift the data. Each test renders
 * the reference triangle and checks the image strictly. */

/* full-image check: every pixel of the lower-left half inside the triangle must be
 * lit, everything above the anti-diagonal must stay clear; the two-pixel band around
 * the diagonal itself is ignored (edge sampling). The reference triangle's corners
 * are (-1,1),(-1,-1),(1,-1): its hypotenuse is the anti-diagonal x + y = w */
static void check_strict_half(const char *tag, int min_covered)
{
    static float img[512 * 512 * 4];
    GLint vp[4] = { 0, 0, 0, 0 };
    int w, h, covered = 0, bad = 0;

    glGetIntegerv(GL_VIEWPORT, vp);
    w = vp[2] > 0 ? vp[2] : 128;
    h = vp[3] > 0 ? vp[3] : 128;
    if (w > 512 || h > 512)
        w = h = 512;
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, img);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            const float *px = &img[(y * w + x) * 4];
            float lum = px[0] + px[1] + px[2];
            if (x + y <= w - 3)
            {
                if (lum > 0.2f) covered++;
            }
            else if (x + y >= w + 1 && lum > 0.2f)
                bad++;
        }
    if (bad || covered < min_covered)
    {
        int minx = 999, miny = 999, maxx = -1, maxy = -1, cnt = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (img[(y * w + x) * 4] > 0.5f)
                {
                    cnt++;
                    if (x < minx) minx = x;
                    if (x > maxx) maxx = x;
                    if (y < miny) miny = y;
                    if (y > maxy) maxy = y;
                }
        fprintf(stderr, "  [%s] covered=%d bad=%d lit=%d bbox=(%d,%d)-(%d,%d) vp=%dx%d\n",
                tag, covered, bad, cnt, minx, miny, maxx, maxy, w, h);
    }
    EXPECT_TRUE(bad == 0);
    EXPECT_TRUE(covered >= min_covered);
}

/* point_mode: every rasterized point must lie inside the triangle */
static void check_points_inside(const char *tag, int min_points)
{
    static float img[512 * 512 * 4];
    GLint vp[4] = { 0, 0, 0, 0 };
    int w, h, points = 0, bad = 0;

    glGetIntegerv(GL_VIEWPORT, vp);
    w = vp[2] > 0 ? vp[2] : 128;
    h = vp[3] > 0 ? vp[3] : 128;
    if (w > 512 || h > 512)
        w = h = 512;
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, img);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (img[(y * w + x) * 4] + img[(y * w + x) * 4 + 1] + img[(y * w + x) * 4 + 2] > 0.2f)
            {
                points++;
                if (x + y > w + 1)
                    bad++;
            }
    if (bad || points < min_points)
        fprintf(stderr, "  [%s] points=%d bad=%d vp=%dx%d\n", tag, points, bad, w, h);
    EXPECT_TRUE(bad == 0);
    EXPECT_TRUE(points >= min_points);
}

/* interior reference triangle (point_mode tests): no control point or tessellated
 * vertex lands exactly on a clip boundary, so every lattice point rasterizes */
static const float kIntPos[3][4] = {
    { -0.6f,  0.6f, 0, 1 },
    { -0.6f, -0.6f, 0, 1 },
    {  0.6f, -0.6f, 0, 1 },
};

static void draw_reference_patch(void)
{
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
}

/* pixel near reference triangle corner 0 (top-left), 1 (bottom-left), 2 (bottom-right),
 * moved diagonally inside so it is not on the hypotenuse (x + y = 0) */
static void corner_px(int corner, int *x, int *y)
{
    GLint vp[4] = { 0, 0, 0, 0 };
    int w, h;

    glGetIntegerv(GL_VIEWPORT, vp);
    w = vp[2] > 0 ? vp[2] : 128;
    h = vp[3] > 0 ? vp[3] : 128;
    *x = (corner == 2) ? w - w / 8 : w / 16;
    *y = (corner == 0) ? h - h / 8 : h / 16;
}

/* TCS writes two varyings; the TES declares both but reads only gl_Position */
static const char *kTcsUnused =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "layout(location=1) out vec4 tc_extra[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  tc_extra[gl_InvocationID] = vec4(0.5, 0.5, 0.5, 1.0);\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
    "    gl_TessLevelInner[0] = 3.0;\n"
    "  }\n"
    "}\n";

static const char *kTesUnused(const char *layout)   /* layout: with/without point_mode */
{
    static char buf[1024];
    snprintf(buf, sizeof buf,
        "#version 450 core\n"
        "layout(%s) in;\n"
        "layout(location=0) in vec4 tc_col[];\n"
        "layout(location=1) in vec4 tc_extra[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = vec4(1, 1, 1, 1);\n"
        "}\n", layout);
    return buf;
}

/* the same, but the TCS also never writes what the VS sent (v_col unused by the TCS) */
static const char *kTcsUnusedIn =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=1) in vec4 v_unused[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = vec4(1.0);\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
    "    gl_TessLevelInner[0] = 3.0;\n"
    "  }\n"
    "}\n";

TEST(TessUnusedTesInput)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    /* hardware path (drawPatches) */
    GLuint prog = make_program(kTcsUnused, kTesUnused("triangles, equal_spacing, ccw"));
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    draw_reference_patch();
    check_strict_half("unused-input hw", 18000);
    glDeleteProgram(prog);

    /* emulated path (point_mode renders through the plain TES): interior corners, so
     * every tessellated vertex of the level-3 lattice (10 points) rasterizes */
    prog = make_program(kTcsUnused, kTesUnused("triangles, equal_spacing, ccw, point_mode"));
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kIntPos, kTriCol, 3);
    draw_reference_patch();
    check_points_inside("unused-input emu", 10);
    glDeleteProgram(prog);
}

TEST(TessUnusedTcsInput)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    /* the TCS declares a VS output it never reads: its input record must still match
     * the vertex kernel's output record */
    GLuint prog = make_program(kTcsUnusedIn, kTesUnused("triangles, equal_spacing, ccw"));
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    draw_reference_patch();
    check_strict_half("unused-tcs-input hw", 18000);
    glDeleteProgram(prog);
}

/* ---- GL 4.6 7.4.1 (shader interface matching, r67) ----------------------
 *
 * "A link error is generated if any statically referenced input variable or
 * block does not have a matching output" (inside ONE program object).  glslang
 * auto-maps the orphan and reports nothing, so the consumer would read a record
 * nobody writes. */

static const char *kTcsNoExtra =
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

TEST(TessLinkUnmatchedInput)
{
    /* statically used TES input with no TCS output: link error */
    const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "in vec4 tc_missing[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = tc_missing[0];\n"
        "}\n";
    GLuint p = make_program(kTcsNoExtra, tes);
    GLint link = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_FALSE);
    char log[1024] = "";
    glGetProgramInfoLog(p, sizeof log - 1, NULL, log);
    EXPECT_TRUE(strstr(log, "has no matching output") != NULL);
    glDeleteProgram(p);

    /* the same declaration never used: NOT a link error (7.4.1 only rejects
     * statically referenced inputs) */
    const char *tes_unused =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "in vec4 tc_missing[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = vec4(1.0);\n"
        "}\n";
    p = make_program(kTcsNoExtra, tes_unused);
    link = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glDeleteProgram(p);
    while (glGetError() != GL_NO_ERROR) { }
}

TEST(TessLinkUnmatchedByLocation)
{
    /* explicit locations: the TCS writes 0, the TES also reads 5 */
    const char *tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in vec4 v_col[];\n"
        "layout(location=0) out vec4 tc_a[];\n"
        "void main() {\n"
        "  tc_a[gl_InvocationID] = v_col[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location=0) in vec4 tc_a[];\n"
        "layout(location=5) in vec4 tc_b[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = tc_a[0] + tc_b[1];\n"
        "}\n";
    GLuint p = make_program(tcs, tes);
    GLint link = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_FALSE);
    glDeleteProgram(p);
    while (glGetError() != GL_NO_ERROR) { }
}

TEST(TessLinkBlockInstanceNames)
{
    /* interface blocks match by BLOCK name: the TCS's instance-less out block
     * feeds the TES's differently-named in block instance */
    const char *tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in vec4 v_col[];\n"
        "out StageOut { vec4 col; } tcs_o[];\n"
        "void main() {\n"
        "  tcs_o[gl_InvocationID].col = v_col[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "in StageOut { vec4 col; } tc_in[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = gl_TessCoord.x * tc_in[0].col + gl_TessCoord.y * tc_in[1].col\n"
        "        + gl_TessCoord.z * tc_in[2].col;\n"
        "}\n";
    GLuint p = make_program(tcs, tes);
    GLint link = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link)
    {
        glUseProgram(p);
        glViewport(0, 0, 512, 512);
        setup_patch_vao(kTriPos, kTriCol, 3);
        draw_reference_patch();
        check_strict_half("block-instance-names", 18000);
    }
    glDeleteProgram(p);
}

TEST(TessLinkCacheAfterFailure)
{
    /* glslang's TIntermediate error count is sticky and lives on the shared
     * TShader: a FAILED link used to poison the compile cache so the next
     * program linking the same sources died with an empty "GLSL link failed"
     * (dEQP shaders.linkage.gl45.tessellation.varying.rules).  A failed link
     * must not break the next one. */
    const char *tcs_bad =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in float v_col[];\n"     /* type mismatch vs the VS */
        "layout(location=0) out float tc_col[];\n"
        "void main() {\n"
        "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 2.0; gl_TessLevelOuter[1] = 2.0; gl_TessLevelOuter[2] = 2.0;\n"
        "    gl_TessLevelInner[0] = 2.0;\n"
        "  }\n"
        "}\n";
    const char *tes_bad =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location=0) in float tc_col[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = vec4(tc_col[0]);\n"
        "}\n";
    GLuint bad = make_program(tcs_bad, tes_bad);
    GLint link = -1;
    glGetProgramiv(bad, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_FALSE);                 /* the type mismatch itself */
    glDeleteProgram(bad);

    /* the same VS and FS sources must still link a good TCS/TES pair */
    GLuint good = make_program(kTcsTri, kTesTri);
    link = -1;
    glGetProgramiv(good, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glDeleteProgram(good);
    while (glGetError() != GL_NO_ERROR) { }
}

/* TCS writes locations 1, 0, 2 in that source order; the TES declares 2, 0, 1 */
static const char *kTcsOrder =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=1) out vec4 tc_b[];\n"
    "layout(location=0) out vec4 tc_a[];\n"
    "layout(location=2) out vec4 tc_c[];\n"
    "void main() {\n"
    "  tc_b[gl_InvocationID] = vec4(0.0, 1.0, 0.0, 1.0);\n"
    "  tc_a[gl_InvocationID] = vec4(1.0, 0.0, 0.0, 1.0);\n"
    "  tc_c[gl_InvocationID] = vec4(0.0, 0.0, 1.0, 1.0);\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 4.0; gl_TessLevelOuter[1] = 4.0; gl_TessLevelOuter[2] = 4.0;\n"
    "    gl_TessLevelInner[0] = 4.0;\n"
    "  }\n"
    "}\n";




static const char *kTesOrder =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=2) in vec4 t_c[];\n"     /* different names than the TCS, */
    "layout(location=0) in vec4 t_a[];\n"     /* declared in another order */
    "layout(location=1) in vec4 t_b[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = gl_TessCoord.x * t_a[0] + gl_TessCoord.y * t_b[1] + gl_TessCoord.z * t_c[2];\n"
    "}\n";

TEST(TessReorderedInputs)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    GLuint prog = make_program(kTcsOrder, kTesOrder);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    draw_reference_patch();

    /* the corners take their colour from the varying at the matching location,
     * whatever order the two stages declare them in */
    float c[4];
    int px, py;
    corner_px(0, &px, &py);
    read_px(px, py, c);
    corner_px(1, &px, &py);
    read_px(px, py, c);
    corner_px(2, &px, &py);
    read_px(px, py, c);
    fprintf(stderr, " c2=(%.2f,%.2f,%.2f)\n", c[0], c[1], c[2]);
    corner_px(0, &px, &py);
    read_px(px, py, c);      /* corner 0: location 0 = red */
    EXPECT_TRUE(c[0] > 0.6f);
    EXPECT_TRUE(c[1] < 0.4f);
    corner_px(1, &px, &py);
    read_px(px, py, c);      /* corner 1: location 1 = green */
    EXPECT_TRUE(c[1] > 0.6f);
    EXPECT_TRUE(c[0] < 0.4f);
    corner_px(2, &px, &py);
    read_px(px, py, c);      /* corner 2: location 2 = blue */
    EXPECT_TRUE(c[2] > 0.6f);
    EXPECT_TRUE(c[0] < 0.4f);
    check_strict_half("reordered hw", 18000);
    glDeleteProgram(prog);
}

/* per-patch varyings: the names differ between the stages and one is never read */
static const char *kTcsPatch =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "layout(location=2) patch out vec4 p_col;\n"
    "layout(location=3) patch out vec4 p_extra;\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    p_col = vec4(0.75, 0.5, 0.25, 1.0);\n"
    "    p_extra = vec4(1.0, 0.0, 0.0, 1.0);\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
    "    gl_TessLevelInner[0] = 3.0;\n"
    "  }\n"
    "}\n";

static const char *kTesPatch =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_col[];\n"
    "layout(location=3) patch in vec4 p_unused;\n"
    "layout(location=2) patch in vec4 t_col;   /* different name, same location */\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = t_col;\n"
    "}\n";

TEST(TessPerPatchVaryings)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    GLuint prog = make_program(kTcsPatch, kTesPatch);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    draw_reference_patch();

    /* every lit pixel carries the per-patch colour (0.75, 0.5, 0.25) */
    float c[4];
    {
        GLint vp[4] = { 0, 0, 0, 0 };
        glGetIntegerv(GL_VIEWPORT, vp);
        read_px(vp[2] / 4, vp[3] / 2, c);   /* interior of the triangle */
    }
    EXPECT_NEAR(c[0], 0.75f, 0.02f);
    EXPECT_NEAR(c[1], 0.50f, 0.02f);
    EXPECT_NEAR(c[2], 0.25f, 0.02f);
    check_strict_half("per-patch hw", 18000);
    glDeleteProgram(prog);
}

/* arrayed varying: the record holds a 2-element array at one location */
static const char *kTcsArr =
    "#version 450 core\n"
    "#extension GL_ARB_arrays_of_arrays : enable\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_arr[][2];\n"
    "void main() {\n"
    "  tc_arr[gl_InvocationID][0] = vec4(1.0, 0.0, 0.0, 1.0);\n"
    "  tc_arr[gl_InvocationID][1] = vec4(0.0, 0.0, 1.0, 1.0);\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
    "    gl_TessLevelInner[0] = 3.0;\n"
    "  }\n"
    "}\n";

static const char *kTesArr =
    "#version 450 core\n"
    "#extension GL_ARB_arrays_of_arrays : enable\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec4 tc_arr[][2];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  f_col = gl_TessCoord.x * tc_arr[0][1] + gl_TessCoord.z * tc_arr[2][0];\n"
    "}\n";

TEST(TessArrayedVarying)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    GLuint prog = make_program(kTcsArr, kTesArr);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    draw_reference_patch();

    /* corner 0 reads the array's second element (blue), corner 2 the first (red) */
    float c[4];
    int px, py;
    corner_px(0, &px, &py);
    read_px(px, py, c);
    EXPECT_TRUE(c[2] > 0.6f);
    EXPECT_TRUE(c[0] < 0.4f);
    corner_px(2, &px, &py);
    read_px(px, py, c);
    EXPECT_TRUE(c[0] > 0.6f);
    EXPECT_TRUE(c[2] < 0.4f);
    check_strict_half("arrayed hw", 18000);
    glDeleteProgram(prog);
}

/* ---- patch block ARRAY (dEQP user_defined_io per_patch_block_array shape) ----
 * every leaf of a 2-element patch block array is compared against the value the
 * TCS wrote; the TES colours grey with the FIRST FAILING leaf index (green when
 * everything matches) so a bad mapping shows up leaf-exactly in one pixel. */
static const char *kTcsBlkArr =
    "#version 450 core\n"
    "struct S { int x; vec4 y; float z[2]; };\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "patch out TheBlock {\n"
    "  float blockFa[3];\n"
    "  S     blockSa[2];\n"
    "  float blockF;\n"
    "} tcBlock[2];\n"
    "void main() {\n"
    "  for (int e = 0; e < 2; e++) {\n"
    "    tcBlock[e].blockFa = float[3](1.5 + float(e), 2.5 + float(e), 3.5 + float(e));\n"
    "    tcBlock[e].blockSa[0].x = 3 + e;  tcBlock[e].blockSa[1].x = 5 + e;\n"
    "    tcBlock[e].blockSa[0].y = vec4(4.0 + float(e), 5.0, 6.0, 7.0);\n"
    "    tcBlock[e].blockSa[1].y = vec4(8.0 + float(e), 9.0, 10.0, 11.0);\n"
    "    tcBlock[e].blockSa[0].z = float[2](0.5 + float(e), 1.5);\n"
    "    tcBlock[e].blockSa[1].z = float[2](2.5 + float(e), 3.5);\n"
    "    tcBlock[e].blockF = 12.5 + float(e);\n"
    "  }\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
    "    gl_TessLevelInner[0] = 3.0;\n"
    "  }\n"
    "}\n";

static const char *kTesBlkArr =
    "#version 450 core\n"
    "struct S { int x; vec4 y; float z[2]; };\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "patch in TheBlock {\n"
    "  float blockFa[3];\n"
    "  S     blockSa[2];\n"
    "  float blockF;\n"
    "} teBlock[2];\n"
    "layout(location=0) out vec4 f_col;\n"
    "int g_fail = -1;\n"
    "void chk(float got, float want, int idx) { if (g_fail < 0 && got != want) g_fail = idx; }\n"
    "void chk4(vec4 got, vec4 want, int idx) { if (g_fail < 0 && got != want) g_fail = idx; }\n"
    "void main() {\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
    "  for (int e = 0; e < 2; e++) {\n"
    "    chk(teBlock[e].blockFa[0], 1.5 + float(e), e * 14 + 0);\n"
    "    chk(teBlock[e].blockFa[1], 2.5 + float(e), e * 14 + 1);\n"
    "    chk(teBlock[e].blockFa[2], 3.5 + float(e), e * 14 + 2);\n"
    "    chk(float(teBlock[e].blockSa[0].x), float(3 + e), e * 14 + 3);\n"
    "    chk4(teBlock[e].blockSa[0].y, vec4(4.0 + float(e), 5.0, 6.0, 7.0), e * 14 + 4);\n"
    "    chk(teBlock[e].blockSa[0].z[0], 0.5 + float(e), e * 14 + 5);\n"
    "    chk(teBlock[e].blockSa[0].z[1], 1.5, e * 14 + 6);\n"
    "    chk(float(teBlock[e].blockSa[1].x), float(5 + e), e * 14 + 7);\n"
    "    chk4(teBlock[e].blockSa[1].y, vec4(8.0 + float(e), 9.0, 10.0, 11.0), e * 14 + 8);\n"
    "    chk(teBlock[e].blockSa[1].z[0], 2.5 + float(e), e * 14 + 9);\n"
    "    chk(teBlock[e].blockSa[1].z[1], 3.5, e * 14 + 10);\n"
    "    chk(teBlock[e].blockF, 12.5 + float(e), e * 14 + 11);\n"
    "  }\n"
    "  if (g_fail < 0) f_col = vec4(0.0, 1.0, 0.0, 1.0);\n"
    "  else            f_col = vec4(0.1 + float(g_fail) * 0.02, 0.0, 0.0, 1.0);\n"
    "}\n";

TEST(TessPatchBlockArrayValues)
{
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
    GLuint prog = make_program(kTcsBlkArr, kTesBlkArr);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link)
    {
        glUseProgram(prog);
        setup_patch_vao(kTriPos, kTriCol, 3);
        draw_reference_patch();
        float c[4];
        {
            int px, py;
            corner_px(0, &px, &py);
            read_px(px, py, c);   /* inside the triangle near corner 0 */
        }
        /* red-ish = the first failing leaf: (r - 0.1) / 0.02 */
        EXPECT_NEAR(c[0], 0.0f, 0.02f);
        EXPECT_NEAR(c[1], 1.0f, 0.02f);
        if (c[0] > 0.05f)
            fprintf(stderr, "TessPatchBlockArrayValues: first failing leaf %d\n",
                    (int)((c[0] - 0.1f) / 0.02f + 0.5f));
        check_strict_half("patch block array hw", 18000);
    }
    glDeleteProgram(prog);
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
    glViewport(0, 0, 200, 200);   /* these pixel checks assume a 200x200 window */
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

/* ------------- restart-presence cache -------------------------------------------------
 * With primitive restart enabled, whether the range contains the restart value is
 * decided on the CPU (it re-forms patches, so the draw encoding depends on it) —
 * but at most ONCE per buffer content: the result is cached like the round-13
 * scan results, keyed on the element buffer and its gpuaudit content serial. */
TEST(TessRestartCheckCached)
{
    unsigned c0, r0;

    GLuint prog = make_program(kTcsTri, kTesTri);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);

    const float pos[6][4] = {
        { -1,  1, 0, 1 }, { -1, -1, 0, 1 }, { 0, -1, 0, 1 },
        {  0,  1, 0, 1 }, {  0, -1, 0, 1 }, { 1, -1, 0, 1 },
    };
    const float col[6][4] = {
        { 1,0,0,1 }, { 1,0,0,1 }, { 1,0,0,1 },
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
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glEnable(GL_PRIMITIVE_RESTART);
    glPrimitiveRestartIndex(0xFFFF);

    /* a range WITHOUT the restart value: the presence check runs once, never again */
    const unsigned short idx[6] = { 0, 1, 2, 3, 4, 5 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);

    c0 = mgl_tess_restart_checks();
    for (int rep = 0; rep < 3; rep++)
    {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL);
        EXPECT_NO_GL_ERROR();
    }
    glFinish();
    printf("restart checks (no restart values): %u -> %u\n", c0, mgl_tess_restart_checks());
    EXPECT_TRUE(mgl_tess_restart_checks() - c0 == 1);

    /* new contents: one more check */
    const unsigned short idx2[6] = { 3, 4, 5, 0, 1, 2 };
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof idx2, idx2);
    c0 = mgl_tess_restart_checks();
    for (int rep = 0; rep < 2; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL);
        EXPECT_NO_GL_ERROR();
    }
    glFinish();
    printf("restart checks (after glBufferSubData): %u -> %u\n", c0, mgl_tess_restart_checks());
    EXPECT_TRUE(mgl_tess_restart_checks() - c0 == 1);

    /* a range WITH a restart value: one presence check, one widened build, then cached */
    const unsigned short idx3[7] = { 3, 4, 5, 0xFFFF, 0, 1, 2 };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx3, idx3, GL_STATIC_DRAW);
    c0 = mgl_tess_restart_checks();
    r0 = mgl_tess_cpu_index_reads();
    for (int rep = 0; rep < 3; rep++)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawElements(GL_PATCHES, 7, GL_UNSIGNED_SHORT, NULL);
        EXPECT_NO_GL_ERROR();
    }
    glFinish();
    printf("restart checks (restart range): %u -> %u, index reads %u -> %u\n",
           c0, mgl_tess_restart_checks(), r0, mgl_tess_cpu_index_reads());
    EXPECT_TRUE(mgl_tess_restart_checks() - c0 == 1);
    EXPECT_TRUE(mgl_tess_cpu_index_reads() - r0 == 1);

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
    glViewport(0, 0, 512, 512);   /* these pixel checks assume a 512x512 window */
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


/* xfb capture of tessellated output with GL_POINTS: GL 4.6 13.4.2 - POINTS
 * captures one record per emitted vertex from ANY tessellated primitive type.
 * The record COUNT and PRIMITIVES_WRITTEN come from the transform feedback
 * primitive mode the app selected, not from the TES domain (r25: a triangle
 * domain with GL_POINTS capture wrote 3x too few records - one per triangle -
 * and KHR-GL46.tessellation_shader.vertex.vertex_spacing spun forever in its
 * vertex classification loop). */
TEST(TessXfbPoints)
{
    struct Src { int t; const char *s; } srcs[] = {
        { GL_VERTEX_SHADER, kVs },
        { GL_TESS_CONTROL_SHADER,
          "#version 450 core\n"
          "layout(vertices=3) out;\n"
          "void main() {\n"
          "  if (gl_InvocationID == 0) {\n"
          "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
          "    gl_TessLevelInner[0] = 3.0;\n"
          "  }\n"
          "}\n" },
        { GL_TESS_EVALUATION_SHADER,
          "#version 450 core\n"
          "layout(triangles, equal_spacing, ccw) in;\n"
          "out vec3 tcoord;\n"
          "void main() { tcoord = gl_TessCoord; gl_Position = vec4(gl_TessCoord, 1.0); }\n" },
    };
    GLuint p = glCreateProgram();
    for (unsigned i = 0; i < sizeof srcs / sizeof srcs[0]; i++) {
        GLuint so = glCreateShader(srcs[i].t);
        glShaderSource(so, 1, &srcs[i].s, NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    const char *xfb[] = { "tcoord" };
    glTransformFeedbackVaryings(p, 1, xfb, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    GLuint xb, q;
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 1 << 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glGenQueries(1, &q);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    GLint wr = -1;
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &wr);
    glDeleteQueries(1, &q);
    /* outer=inner=3: 13 triangles are generated, each contributing 3 points */
    EXPECT_EQ_INT(wr, 39);

    glBindBuffer(GL_ARRAY_BUFFER, xb);
    float *m = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    EXPECT_TRUE(m != NULL);
    if (m) {
        /* the distinct tess coordinates: 3 corners + 6 edge points + 3 interior */
        int uniq = 0, seen_corners = 0;
        for (int i = 0; i < wr; i++) {
            const float *c = m + i * 3;
            int dup = 0;
            for (int j = 0; j < i; j++)
                if (m[j*3] == c[0] && m[j*3+1] == c[1] && m[j*3+2] == c[2]) { dup = 1; break; }
            if (!dup) {
                uniq++;
                if ((c[0] > 0.999f && c[1] < 0.001f) || (c[1] > 0.999f && c[0] < 0.001f) ||
                    (c[2] > 0.999f && c[0] < 0.001f))
                    seen_corners++;
            }
        }
        EXPECT_EQ_INT(uniq, 12);          /* 3 + 6 + 3 */
        EXPECT_EQ_INT(seen_corners, 3);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
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

/* ------------------------------------------------------------------ instanced patches
 * glDrawArraysInstanced(GL_PATCHES, ...): the TCS kernel writes one half-factor
 * record per (instance, patch) — the GLOBAL patch index — while gl_PrimitiveID
 * restarts at 0 for every instance (GL 4.6 table 11.1).  A regression in the
 * MSL rewrite (the record subscript matched only `spvTessLevel[gl_PrimitiveID]`,
 * not the `int(...)`-cast spelling SPIRV-Cross also emits) wrote every patch's
 * factors to record [0]: instance 1 was culled by the hardware tessellator and
 * rendered nothing.  These tests render two instances with a per-instance
 * attribute and check the image exactly. */

static const char *kInstVs =
    "#version 450 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=2) in vec4 a_ic;        /* xy offset (divisor 1), z colour */\n"
    "out vec2 v_pos; out float v_sel;\n"
    "void main() { v_pos = a_pos * 0.25 + a_ic.xy; v_sel = a_ic.z; }\n";
static const char *kInstTcs =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec2 v_pos[]; layout(location=1) in float v_sel[];\n"
    "layout(location=0) out vec2 tc_pos[]; layout(location=1) out float tc_sel[];\n"
    "void main() {\n"
    "  tc_pos[gl_InvocationID] = v_pos[gl_InvocationID];\n"
    "  tc_sel[gl_InvocationID] = v_sel[gl_InvocationID];\n"
    "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0]=3.0; gl_TessLevelOuter[1]=3.0;\n"
    "    gl_TessLevelOuter[2]=3.0; gl_TessLevelInner[0]=3.0; }\n"
    "}\n";
static const char *kInstTes =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec2 tc_pos[]; layout(location=1) in float tc_sel[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  vec2 p = gl_TessCoord.x * tc_pos[0] + gl_TessCoord.y * tc_pos[1] + gl_TessCoord.z * tc_pos[2];\n"
    "  gl_Position = vec4(p, 0.0, 1.0);\n"
    "  float s = gl_TessCoord.x * tc_sel[0] + gl_TessCoord.y * tc_sel[1] + gl_TessCoord.z * tc_sel[2];\n"
    "  f_col = (s > 0.5) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);\n"
    "}\n";
static const char *kInstTesNoTcs =
    "#version 450 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(location=0) in vec2 v_pos[]; layout(location=1) in float v_sel[];\n"
    "layout(location=0) out vec4 f_col;\n"
    "void main() {\n"
    "  vec2 p = gl_TessCoord.x * v_pos[0] + gl_TessCoord.y * v_pos[1] + gl_TessCoord.z * v_pos[2];\n"
    "  gl_Position = vec4(p, 0.0, 1.0);\n"
    "  float s = gl_TessCoord.x * v_sel[0] + gl_TessCoord.y * v_sel[1] + gl_TessCoord.z * v_sel[2];\n"
    "  f_col = (s > 0.5) ? vec4(0, 1, 0, 1) : vec4(1, 0, 0, 1);\n"
    "}\n";

/* control triangle: pos*0.25 gives legs of 1.0 in NDC; instance 0 shifts it to
 * (-0.35,-0.35) -> red triangle {x>=-0.6, y>=-0.6, x+y<=-0.2}; instance 1 shifts
 * to (0.35,0.35) and draws second -> green {x>=0.1, y>=0.1, x+y<=1.2} */
static const float kInstPos[3][2] = { { -1, -1 }, { 3, -1 }, { -1, 3 } };
static const float kInstIc[2][4] = {
    { -0.35f, -0.35f, 0.0f, 0.0f },
    {  0.35f,  0.35f, 1.0f, 0.0f },
};

/* exact image check for the two-instance draw: every pixel centre strictly
 * inside a triangle must carry that instance's colour, everything outside both
 * stays clear.  A 1.5-pixel band around each triangle edge is skipped (shared
 * edge sampling / rasterization rules). */
static void check_two_instance_image(const char *tag, int with_green)
{
    static float img[512 * 512 * 4];
    GLint vp[4] = { 0, 0, 0, 0 };
    int w, h, bad = 0, red = 0, green = 0;

    glGetIntegerv(GL_VIEWPORT, vp);
    w = vp[2] > 0 ? vp[2] : 96;
    h = vp[3] > 0 ? vp[3] : 96;
    if (w > 512 || h > 512) w = h = 512;
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, img);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            const float *c = &img[(y * w + x) * 4];
            float nx = (x + 0.5f) / w * 2 - 1, ny = (y + 0.5f) / h * 2 - 1;
            /* margins: ~1.5 px in NDC units; skip pixels within any boundary band */
            float mx = 3.0f / w, my = 3.0f / h;
            int in_r = (nx >= -0.6f + mx) && (ny >= -0.6f + my) && (nx + ny <= -0.2f - mx - my);
            int in_g = with_green && (nx >= 0.1f + mx) && (ny >= 0.1f + my) && (nx + ny <= 1.2f - mx - my);
            int near_r = (fabsf(nx + 0.6f) < mx) || (fabsf(ny + 0.6f) < my) ||
                         (fabsf(nx + ny + 0.2f) < mx + my);
            int near_g = with_green && ((fabsf(nx - 0.1f) < mx) || (fabsf(ny - 0.1f) < my) ||
                         (fabsf(nx + ny - 1.2f) < mx + my));
            int lit = (c[0] + c[1] + c[2]) > 0.2f;

            if (near_r || near_g) continue;                  /* skip edge bands */
            if (c[0] > 0.9f && c[1] < 0.1f) red++;
            if (c[1] > 0.9f && c[0] < 0.1f) green++;
            if (in_r && !(c[0] > 0.5f && c[1] < 0.5f))
            { if (bad++ < 6) printf("  %s: want red at %d,%d got %.2f %.2f %.2f\n", tag, x, y, c[0], c[1], c[2]); }
            else if (in_g && !(c[1] > 0.5f && c[0] < 0.5f))
            { if (bad++ < 6) printf("  %s: want green at %d,%d got %.2f %.2f %.2f\n", tag, x, y, c[0], c[1], c[2]); }
            else if (!in_r && !in_g && lit)
            { if (bad++ < 6) printf("  %s: want clear at %d,%d got %.2f %.2f %.2f\n", tag, x, y, c[0], c[1], c[2]); }
        }
    printf("  %s: red=%d green=%d bad=%d\n", tag, red, green, bad);
    EXPECT_TRUE(red > 200);
    if (with_green) EXPECT_TRUE(green > 200);
    EXPECT_EQ_INT(bad, 0);
}

static void setup_inst_vao(void)
{
    GLuint vao, b[2];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof kInstPos, kInstPos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof kInstIc, kInstIc, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(2);
}

TEST(TessInstancedPerInstanceAttrib)
{
    glViewport(0, 0, 96, 96);
    GLuint prog = make_program_full(kInstVs, kInstTcs, kInstTes);
    glUseProgram(prog);
    setup_inst_vao();
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(GL_PATCHES, 0, 3, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    check_two_instance_image("tcs-2inst", 1);
    glDeleteProgram(prog);
}

TEST(TessInstancedNoTcsDefaultLevels)
{
    /* no TCS: the TES reads the vertex kernel's output record directly (its
       inputs must match the VS outputs by name); the default levels come from
       glPatchParameterfv.  Both instances must still render. */
    glViewport(0, 0, 96, 96);
    GLuint prog = make_program_full(kInstVs, NULL, kInstTesNoTcs);
    glUseProgram(prog);
    setup_inst_vao();
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, (float[4]){ 3, 3, 3, 3 });
    glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, (float[2]){ 3, 3 });
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(GL_PATCHES, 0, 3, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    check_two_instance_image("notcs-2inst", 1);
    glDeleteProgram(prog);
}

TEST(TessInstancedPerPatchLevels)
{
    /* gl_PrimitiveID restarts at 0 for every instance (GL 4.6): two patches per
       instance, patch 1 writes level 3 and patch 0 writes 0 (culled).  Every
       instance must therefore render exactly one triangle — its patch 1.  If
       gl_PrimitiveID counted across instances, instance 1's patch 1 would be
       patch 3 overall and cull, leaving one triangle. */
    static const char *tcs_pid =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in vec2 v_pos[]; layout(location=1) in float v_sel[];\n"
        "layout(location=0) out vec2 tc_pos[]; layout(location=1) out float tc_sel[];\n"
        "void main() {\n"
        "  tc_pos[gl_InvocationID] = v_pos[gl_InvocationID];\n"
        "  tc_sel[gl_InvocationID] = v_sel[gl_InvocationID];\n"
        "  if (gl_InvocationID == 0) {\n"
        "    float lv = (gl_PrimitiveID == 1) ? 3.0 : 0.0;\n"
        "    gl_TessLevelOuter[0]=lv; gl_TessLevelOuter[1]=lv; gl_TessLevelOuter[2]=lv;\n"
        "    gl_TessLevelInner[0]=lv; }\n"
        "}\n";
    glViewport(0, 0, 96, 96);
    GLuint prog = make_program_full(kInstVs, tcs_pid, kInstTes);
    glUseProgram(prog);
    /* 6 vertices = 2 patches per instance, both the same base triangle: patch 1
       (level 3) renders where patch 0 (level 0, culled) would be, so the image
       is exactly the two-instance reference below */
    GLuint vao, b[2];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, b);
    float pos6[6][2];
    for (int i = 0; i < 6; i++)
    {
        pos6[i][0] = kInstPos[i % 3][0];
        pos6[i][1] = kInstPos[i % 3][1];
    }
    const float ic[2][4] = { { -0.35f, -0.35f, 0.0f, 0.0f }, { 0.35f, 0.35f, 1.0f, 0.0f } };
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos6, pos6, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof ic, ic, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(2);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(GL_PATCHES, 0, 6, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    check_two_instance_image("primid-2x2", 1);
    glDeleteProgram(prog);
}

TEST(TessInstancedIndexedTwoInstances)
{
    /* indexed patches with instanceCount 2: the per-instance offset must reach
       both instances; patch 1 (indices 3..5) is the shifted triangle */
    glViewport(0, 0, 96, 96);
    GLuint prog = make_program_full(kInstVs, kInstTcs, kInstTes);
    glUseProgram(prog);
    GLuint vao, b[3];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(3, b);
    float pos6[6][2];
    for (int i = 0; i < 3; i++)
    {
        pos6[i][0] = kInstPos[i][0];  pos6[i][1] = kInstPos[i][1];
        pos6[3 + i][0] = kInstPos[i][0]; pos6[3 + i][1] = kInstPos[i][1];
    }
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos6, pos6, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    const float ic[2][4] = {
        { -0.35f, -0.35f, 0.0f, 0.0f },
        {  0.35f,  0.35f, 1.0f, 0.0f },
    };
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof ic, ic, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(2);
    const unsigned short ix[6] = { 0, 1, 2, 3, 4, 5 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ix, ix, GL_STATIC_DRAW);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElementsInstanced(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL, 2);
    EXPECT_NO_GL_ERROR();
    glFinish();
    check_two_instance_image("indexed-2inst", 1);
    glDeleteProgram(prog);
}

/* ------------------------------------------------------------------ GPU/CPU parity
 * The MGL_TESS_CPU reference path must render pixel-identical images to the
 * hardware drawPatches path — it is the arbiter for every "which side is wrong"
 * investigation.  The env flag is read once per process, so the test re-execs
 * itself with MGL_TESS_CPU=1 and compares full-image hashes over a small case
 * family (plain, instanced, no-TCS). */
#include <mach-o/dyld.h>
#include <unistd.h>

static void parity_render(const char *tag, unsigned long long *hash, int *lit)
{
    const int W = 96;
    GLuint vao, b[2];
    glViewport(0, 0, W, W);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, b);
    glBindBuffer(GL_ARRAY_BUFFER, b[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof kInstPos, kInstPos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, b[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof kInstIc, kInstIc, GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(2);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(0, 0, 0, 1);

    struct { const char *name; const char *tcs; const char *tes; int instances; } cases[] = {
        { "tri",      kInstTcs, kInstTes,      1 },
        { "tri_inst", kInstTcs, kInstTes,      2 },
        { "notcs",    NULL,     kInstTesNoTcs, 2 },
    };
    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++)
    {
        GLuint prog = make_program_full(kInstVs, cases[c].tcs, cases[c].tes);
        glUseProgram(prog);
        glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, (float[4]){ 3, 3, 3, 3 });
        glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, (float[2]){ 3, 3 });
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArraysInstanced(GL_PATCHES, 0, 3, cases[c].instances);
        glFinish();
        while (glGetError() != GL_NO_ERROR) {}
        static float img[96 * 96 * 4];
        unsigned long long h = 1469598103934665603ULL;
        int l = 0;
        glReadPixels(0, 0, W, W, GL_RGBA, GL_FLOAT, img);
        for (int i = 0; i < W * W; i++) {
            if (img[i*4] + img[i*4+1] + img[i*4+2] > 0.2f) l++;
            for (int k = 0; k < 4; k++) {
                unsigned u;
                memcpy(&u, &img[i*4+k], 4);
                h ^= u; h *= 1099511628211ULL;
            }
        }
        if (hash) hash[c] = h;
        if (lit) lit[c] = l;
        printf("PARITY %s_%s lit=%d hash=%llu\n", tag, cases[c].name, l, h);
        glDeleteProgram(prog);
    }
    glDeleteBuffers(2, b);
    glDeleteVertexArrays(1, &vao);
}

static int tess_parity_child(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "--cpu-parity-child") != 0)
        return 0;
    parity_render("cpu", NULL, NULL);
    fflush(stdout);
    exit(0);
}

TEST(TessCpuReferenceParity)
{
    char self[4096];
    uint32_t sz = sizeof self;
    unsigned long long own_h[3];
    int own_l[3];
    int fds[2];
    char line[256];

    if (_NSGetExecutablePath(self, &sz) != 0)
    { printf("  parity: self path failed\n"); return; }

    parity_render("gpu", own_h, own_l);         /* this process: hardware path */

    fflush(stdout);
    if (pipe(fds) != 0) { EXPECT_TRUE(0); return; }
    pid_t pid = fork();
    if (pid == 0)
    {
        char *const env[] = { (char *)"MGL_TESS_CPU=1", NULL };
        dup2(fds[1], 1);
        close(fds[0]); close(fds[1]);
        execve(self, (char *const[]){ self, (char *)"--cpu-parity-child", NULL }, env);
        _exit(127);
    }
    close(fds[1]);
    FILE *f = fdopen(fds[0], "r");
    int n = 0, ok = 0;
    while (fgets(line, sizeof line, f))
    {
        if (strncmp(line, "PARITY cpu_", 11) == 0 && n < 3)
        {
            int l = 0;
            unsigned long long h = 0;
            if (sscanf(line + 11, "%*[^=]=%d hash=%llu", &l, &h) == 2)
            {
                int match = (l == own_l[n]) && (h == own_h[n]);
                printf("  parity case %d: cpu lit=%d hash=%llu vs gpu lit=%d hash=%llu %s\n",
                       n, l, h, own_l[n], own_h[n], match ? "MATCH" : "MISMATCH");
                ok += match;
            }
            n++;
        }
    }
    fclose(f);
    int st = 0;
    waitpid(pid, &st, 0);
    EXPECT_EQ_INT(n, 3);
    EXPECT_EQ_INT(ok, 3);
}

TEST(TessOrientationCulling)
{    glViewport(0, 0, 200, 200);   /* these pixel checks assume a 200x200 window */
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

/* Counting queries on tessellated output must stay EXACT while active: the draw
 * then takes the CPU reference path (which notes its own counts), and the GPU
 * paths skip mgl_tess_note_counts entirely (r43: its only consumer was a count
 * query).  Lock both: a minimal patch (all levels 1) and a fully culled one. */
static const char *kTcsLvl1 =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "  }\n"
    "}\n";
static const char *kTcsCull =
    "#version 450 core\n"
    "layout(vertices=3) out;\n"
    "layout(location=0) in vec4 v_col[];\n"
    "layout(location=0) out vec4 tc_col[];\n"
    "void main() {\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  if (gl_InvocationID == 0) {\n"
    "    gl_TessLevelOuter[0] = 0.0; gl_TessLevelOuter[1] = 0.0; gl_TessLevelOuter[2] = 0.0;\n"
    "    gl_TessLevelInner[0] = 0.0;\n"
    "  }\n"
    "}\n";

static void query_pair_draw(const char *tcs, GLint *prims, GLint *invocations)
{
    GLuint p = make_program(tcs, kTesTri);
    GLuint q1 = 0, q2 = 0;

    glUseProgram(p);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glGenQueries(1, &q1);
    glGenQueries(1, &q2);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q1);
    glBeginQuery(GL_TESS_EVALUATION_SHADER_INVOCATIONS, q2);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndQuery(GL_TESS_EVALUATION_SHADER_INVOCATIONS);
    *prims = *invocations = -1;
    glGetQueryObjectiv(q1, GL_QUERY_RESULT, prims);
    glGetQueryObjectiv(q2, GL_QUERY_RESULT, invocations);
    glDeleteQueries(1, &q1);
    glDeleteQueries(1, &q2);
    glDeleteProgram(p);
}

TEST(TessCountQueriesExact)
{
    GLint pg = -1, inv = -1;

    query_pair_draw(kTcsLvl1, &pg, &inv);
    printf("  [levels 1] PRIMITIVES_GENERATED=%d TES_INVOCATIONS=%d\n", pg, inv);
    EXPECT_EQ_INT(pg, 1);       /* one triangle, three tessellated vertices */
    EXPECT_EQ_INT(inv, 3);

    query_pair_draw(kTcsCull, &pg, &inv);
    printf("  [culled]   PRIMITIVES_GENERATED=%d TES_INVOCATIONS=%d\n", pg, inv);
    EXPECT_EQ_INT(pg, 0);       /* all outer levels <= 0: the patch is culled */
    EXPECT_EQ_INT(inv, 0);
}

/* r65: the same counting queries while a TES-output transform-feedback capture
 * owns the draw (no geometry stage).  The draw must still count exactly — the
 * CPU reference path publishes the capture too, so forcing it for the query
 * keeps xfb working (XFB_PRIMITIVES_WRITTEN below) while every tess count query
 * (invocations, generated, clipping in/out) reads the exact number instead of 0
 * (the GPU-emulated path publishes count words but only a geometry draw folds
 * them into the query accumulator). */
TEST(TessCountQueriesWithXfbCapture)
{
    const char *tes_xfb =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location=0) in vec4 tc_col[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "layout(xfb_buffer=0, xfb_offset=0) out vec4 tes_col;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  tes_col = tc_col[0];\n"
        "  f_col = tc_col[0];\n"
        "}\n";
    GLuint prog = make_program(kTcsLvl1, tes_xfb);
    GLuint tfo = 0, buf = 0, q[6] = { 0 };
    GLint patches = -1, inv = -1, written = -1, pg = -1, cin = -1, cout = -1;

    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glGenBuffers(1, &buf);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, buf);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glGenTransformFeedbacks(1, &tfo);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf);

    for (int i = 0; i < 6; i++) glGenQueries(1, &q[i]);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginQuery(GL_TESS_CONTROL_SHADER_PATCHES, q[0]);
    glBeginQuery(GL_TESS_EVALUATION_SHADER_INVOCATIONS, q[1]);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q[2]);
    glBeginQuery(GL_PRIMITIVES_GENERATED, q[3]);
    glBeginQuery(GL_CLIPPING_INPUT_PRIMITIVES, q[4]);
    glBeginQuery(GL_CLIPPING_OUTPUT_PRIMITIVES, q[5]);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glEndQuery(GL_TESS_CONTROL_SHADER_PATCHES);
    glEndQuery(GL_TESS_EVALUATION_SHADER_INVOCATIONS);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    glEndQuery(GL_CLIPPING_INPUT_PRIMITIVES);
    glEndQuery(GL_CLIPPING_OUTPUT_PRIMITIVES);
    glDisable(GL_RASTERIZER_DISCARD);

    glGetQueryObjectiv(q[0], GL_QUERY_RESULT, &patches);
    glGetQueryObjectiv(q[1], GL_QUERY_RESULT, &inv);
    glGetQueryObjectiv(q[2], GL_QUERY_RESULT, &written);
    glGetQueryObjectiv(q[3], GL_QUERY_RESULT, &pg);
    glGetQueryObjectiv(q[4], GL_QUERY_RESULT, &cin);
    glGetQueryObjectiv(q[5], GL_QUERY_RESULT, &cout);
    printf("  [xfb capture] patches=%d TES_INVOCATIONS=%d XFB_WRITTEN=%d PG=%d CLIP_IN=%d CLIP_OUT=%d\n",
           patches, inv, written, pg, cin, cout);
    EXPECT_EQ_INT(patches, 1);
    EXPECT_EQ_INT(inv, 3);      /* one triangle at all-1 levels: three vertices */
    EXPECT_EQ_INT(written, 1);
    EXPECT_EQ_INT(pg, 1);
    EXPECT_EQ_INT(cin, 1);
    EXPECT_EQ_INT(cout, 1);
    for (int i = 0; i < 6; i++) glDeleteQueries(1, &q[i]);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glDeleteTransformFeedbacks(1, &tfo);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
}

/* Ring memory bound: a burst of draws whose emulated scratch exceeds
 * TESS_RING_BIG (4 MB) must not pin one buffer per slot — a freed draw reuses
 * a warm big slot (at most a couple grow for in-flight work), and retired big
 * blocks beyond two are freed (mgl_tess_ring_freed_big).  32 quad patches in
 * point_mode need ~5.3 MB of tessellator scratch each. */
TEST(TessRingBigBlockRecycle)
{
    const char *tes =
        "#version 450 core\n"
        "layout(quads, equal_spacing, ccw, point_mode) in;\n"
        "layout(location=0) in vec4 tc_col[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() {\n"
        "  gl_Position = vec4(gl_TessCoord.x * 2.0 - 1.0, gl_TessCoord.y * 2.0 - 1.0, 0.0, 1.0);\n"
        "  f_col = vec4(1, 0.5, 0.2, 1);\n"
        "}\n";
    GLuint p;

    p = make_program(kTcsQuad, tes);
    glUseProgram(p);
    {
        const float pos[4][4] = { { -1, 1, 0, 1 }, { 1, 1, 0, 1 }, { 1, -1, 0, 1 }, { -1, -1, 0, 1 } };
        const float col[4][4] = { { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 } };
        setup_patch_vao(pos, col, 4);
    }
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    for (int i = 0; i < 40; i++)
    {
        glDrawArrays(GL_PATCHES, 0, 4 * 32);
        if ((i % 8) == 7)
            glFinish();      /* let blocks retire so the recycle can run */
    }
    glFinish();
    printf("  freed_big=%lu big_slots=%u after 40 big emulated draws\n",
           mgl_tess_ring_freed_big(), mgl_tess_ring_big_slots());
    /* free draws reuse a warm big slot: a couple grow for in-flight work at
     * most — a slot per draw would pin 40 x 5.3 MB and fail this */
    EXPECT_TRUE(mgl_tess_ring_big_slots() <= 4);
    glDeleteProgram(p);
}

/* ------------- r44: ring generation protocol under real timing ------------------
 * 600 mixed-size patch draws (3-vertex hardware shape and 64-patch hardware draws)
 * with glFinish on a prime stride: slot reuse across generations while the twin
 * retires (completion handler + GL-thread poll) race in every possible order.  A
 * double release would hand one block to two draws and corrupt a frame; every
 * checked frame must stay pixel-exact, and the ring must never wedge (the
 * acquire escape hatch only fires after 2 s of no progress — max poll < 50 ms).
 * mgl_tess_ring_dup_retires() counts the twins the generation guard no-opped. */
TEST(TessRingGenerationHammer)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    GLuint pnos;      /* TES-only (no TCS): a second shape, different ring sizes */
    GLint link = 0;
    int checked = 0;
    uint64_t maxns = 0, totns = 0;

    pnos = make_program(NULL, kTesTri);
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    for (int i = 0; i < 600; i++)
    {
        GLint vp[4];
        float w, h, r[4], g[4], b[4];

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        if ((i % 3) == 0)
        {
            glUseProgram(prog);            /* TCS+TES, 1 patch */
            glDrawArrays(GL_PATCHES, 0, 3);
        }
        else if ((i % 3) == 1)
        {
            glUseProgram(prog);            /* 64 patches in one draw */
            glDrawArraysInstanced(GL_PATCHES, 0, 3 * 64, 1);
        }
        else
        {
            glUseProgram(pnos);            /* no TCS: default levels path */
            glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL,
                               (const float[]){ 2, 2, 2, 2 });
            glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL,
                               (const float[]){ 2, 2 });
            glDrawArrays(GL_PATCHES, 0, 3);
        }
        if (i % 17 == 0)                   /* prime stride: vary slot lifetimes */
        {
            glFinish();
            glGetIntegerv(GL_VIEWPORT, vp);
            w = (float)vp[2];
            h = (float)vp[3];
            read_px(2, (int)h - 4, r);
            read_px(2, 2, g);
            read_px((int)w - 4, 2, b);
            EXPECT_NEAR(r[0], 1.0f, 0.10f);
            EXPECT_NEAR(g[1], 1.0f, 0.10f);
            EXPECT_NEAR(b[2], 1.0f, 0.10f);
            checked++;
        }
    }
    glFinish();
    EXPECT_TRUE(checked > 30);
    mgl_tess_ring_spins(&maxns, &totns);
    printf("  hammer: %d frames checked, dup retires no-opped by the generation guard: %u\n",
           checked, mgl_tess_ring_dup_retires());
    /* the 2 s acquire escape (spin 19999) must never fire: 50 ms of worst-case
       polling is already 200 acquire rounds under load */
    EXPECT_TRUE(maxns < 50000ull * 1000ull);
    glDeleteProgram(prog);
    glDeleteProgram(pnos);
}

/* ------------- r44: un-executable patch counts must drop, not corrupt -------------
 * patches * instances overflows a GLsizei (1024 patches x 2^22 instances = 2^32):
 * before the r44 guard that was signed-overflow UB whose wrapped value sized the
 * ring block (a negative/zero total made the kernels dispatch with garbage
 * grids and Metal validation fail); now the draw is dropped before any GPU work
 * and the ring stays healthy for everything after it. */
TEST(TessHugePatchCountDropped)
{
    GLuint prog = make_program(kTcsTri, kTesTri);
    float big[1024 * 3][4];
    GLint link = 0;

    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);
    for (int i = 0; i < 1024 * 3; i++)
    {
        big[i][0] = kTriPos[i % 3][0];
        big[i][1] = kTriPos[i % 3][1];
        big[i][2] = kTriPos[i % 3][2];
        big[i][3] = kTriPos[i % 3][3];
    }
    {
        GLuint bufs[2], vao;

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(2, bufs);
        glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(big), big, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, bufs[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(big), big, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, bufs[1]);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    }
    glPatchParameteri(GL_PATCH_VERTICES, 3);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(GL_PATCHES, 0, 1024 * 3, 1 << 22);   /* 2^32 total patches */
    EXPECT_NO_GL_ERROR();
    glFinish();
    {
        GLint vp[4];
        float c[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        read_px(vp[2] / 2, vp[3] / 2, c);
        EXPECT_NEAR(c[0], 0.0f, 0.02f);     /* dropped: nothing was drawn */
        EXPECT_NEAR(c[1], 0.0f, 0.02f);
    }
    /* and the ring is still healthy: ordinary draws keep rendering exactly */
    for (int i = 0; i < 40; i++)
        glDrawArraysInstanced(GL_PATCHES, 0, 9, 1);
    glFinish();
    setup_patch_vao(kTriPos, kTriCol, 3);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    {
        GLint vp[4];
        float r[4], g[4], b[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        read_px(2, vp[3] - 4, r);
        read_px(2, 2, g);
        read_px(vp[2] - 4, 2, b);
        EXPECT_NEAR(r[0], 1.0f, 0.10f);
        EXPECT_NEAR(g[1], 1.0f, 0.10f);
        EXPECT_NEAR(b[2], 1.0f, 0.10f);
    }
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

/* Multi-patch GL_PATCHES through a geometry shader: the tess domain captures
 * the TES output in EMISSION order for the geometry kernel (the transform-
 * feedback order), so the geometry import must use identity primitive slots —
 * re-indexing through the tessellator's coordinate list permutes twice and the
 * second patch collapses (KHR-GL46.shader_ballot's 2-patch quad geometry).
 * Two disjoint triangle patches; both halves must end up white. */
static const char *kGsPass =
    "#version 450 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=3) out;\n"
    "layout(location=0) in vec4 f_col[];\n"
    "layout(location=0) out vec4 g_col;\n"
    "void main() {\n"
    "  for (int i = 0; i < 3; i++) {\n"
    "    gl_Position = gl_in[i].gl_Position;\n"
    "    g_col = f_col[i];\n"
    "    EmitVertex();\n"
    "  }\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFsGs =
    "#version 450 core\n"
    "layout(location=0) in vec4 g_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = g_col; }\n";

TEST(TessGeometryMultiPatch)
{
    /* patch 1: left half triangle; patch 2: right half triangle */
    static const float pos[6][4] = {
        { -0.9f, -0.8f, 0, 1 }, { -0.9f, 0.8f, 0, 1 }, { -0.1f, 0.0f, 0, 1 },
        {  0.1f, 0.0f, 0, 1 }, {  0.9f, -0.8f, 0, 1 }, {  0.9f, 0.8f, 0, 1 },
    };
    static const float col[6][4] = {
        { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 },
        { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 },
    };
    static const GLushort idx[6] = { 0, 1, 2, 3, 4, 5 };
    glViewport(0, 0, 200, 200);
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[5] = {
        { GL_VERTEX_SHADER, kVs },
        { GL_TESS_CONTROL_SHADER, kTcsTri },
        { GL_TESS_EVALUATION_SHADER, kTesTri },
        { GL_GEOMETRY_SHADER, kGsPass },
        { GL_FRAGMENT_SHADER, kFsGs },
    };
    for (int i = 0; i < 5; i++)
    {
        GLuint sh = glCreateShader(srcs[i].t);
        glShaderSource(sh, 1, &srcs[i].s, NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_TRUE(link);
    if (!link) { glDeleteProgram(p); return; }

    setup_patch_vao(pos, col, 6);
    /* indexed 2-patch draw: patch 2 uses vertices 3,4,5 */
    {
        GLuint ibo;
        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    }
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_PATCHES, 6, GL_UNSIGNED_SHORT, NULL);
    EXPECT_NO_GL_ERROR();

    float c[4];
    glFinish();
    read_px(60, 100, c);   /* inside patch 1 */
    EXPECT_PIXEL(60, 100, 255, 255, 255, 255, 4);
    read_px(140, 100, c);  /* inside patch 2 */
    EXPECT_PIXEL(140, 100, 255, 255, 255, 255, 4);
    read_px(100, 190, c);  /* between the patches: untouched clear */
    EXPECT_PIXEL(100, 190, 255, 0, 0, 255, 4);
    glDeleteProgram(p);
}

/* --- matrix varyings through the TES capture into the geometry stage ------
 * The geometry import pairs capture members with geometry inputs by name; a
 * flattened MATRIX arrives as "<name>_0","<name>_1" columns, and the
 * single-digit suffix used to fall through the pairing ("no capture member
 * for te_out/") — dEQP shaders.linkage.gl45.tessellation_geometry.varying.
 * types.mat*. */
static GLuint make_5stage_program(const char *vs, const char *tcs, const char *tes,
                                  const char *gs, const char *fs)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } srcs[5] = {
        { GL_VERTEX_SHADER, vs }, { GL_TESS_CONTROL_SHADER, tcs },
        { GL_TESS_EVALUATION_SHADER, tes }, { GL_GEOMETRY_SHADER, gs },
        { GL_FRAGMENT_SHADER, fs },
    };
    for (int i = 0; i < 5; i++)
    {
        GLuint sh = glCreateShader(srcs[i].t);
        glShaderSource(sh, 1, &srcs[i].s, NULL);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char b[2048];
            glGetShaderInfoLog(sh, sizeof b, NULL, b);
            fprintf(stderr, "  %s stage compile: %s\n",
                    srcs[i].t == GL_VERTEX_SHADER ? "vertex" :
                    srcs[i].t == GL_TESS_CONTROL_SHADER ? "tcs" :
                    srcs[i].t == GL_TESS_EVALUATION_SHADER ? "tes" :
                    srcs[i].t == GL_GEOMETRY_SHADER ? "gs" : "fs", b);
        }
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint link = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (!link) {
        char b[2048];
        glGetProgramInfoLog(p, sizeof b, NULL, b);
        fprintf(stderr, "  5-stage link: %s\n", b);
    }
    return p;
}

TEST(TessGeometryMatrixVarying)
{
    const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "layout(location=1) in vec4 a_col;\n"
        "layout(location=0) out mat2 v_m;\n"
        "void main() {\n"
        "  gl_Position = a_pos;\n"
        "  v_m = mat2(a_col.xy, a_col.zw);\n"
        "}\n";
    const char *tcs =
        "#version 450 core\n"
        "layout(vertices=3) out;\n"
        "layout(location=0) in mat2 v_m[];\n"
        "layout(location=0) out mat2 tc_m[];\n"
        "void main() {\n"
        "  tc_m[gl_InvocationID] = v_m[gl_InvocationID];\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) {\n"
        "    gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 3.0; gl_TessLevelOuter[2] = 3.0;\n"
        "    gl_TessLevelInner[0] = 3.0;\n"
        "  }\n"
        "}\n";
    const char *tes =
        "#version 450 core\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location=0) in mat2 tc_m[];\n"
        "layout(location=0) out mat2 te_m;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  te_m = tc_m[0] * gl_TessCoord.x + tc_m[1] * gl_TessCoord.y\n"
        "       + tc_m[2] * gl_TessCoord.z;\n"
        "}\n";
    const char *gs =
        "#version 450 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=3) out;\n"
        "layout(location=0) in mat2 te_m[];\n"
        "layout(location=0) out mat2 g_m;\n"
        "void main() {\n"
        "  for (int i = 0; i < 3; i++) {\n"
        "    g_m = te_m[i];\n"
        "    gl_Position = gl_in[i].gl_Position; EmitVertex();\n"
        "  }\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) in mat2 g_m;\n"
        "layout(location=0) out vec4 f_col;\n"
        "void main() { f_col = vec4(g_m[0], g_m[1]); }\n";

    GLuint p = make_5stage_program(vs, tcs, tes, gs, fs);
    EXPECT_TRUE(p != 0);
    if (p) {
        glUseProgram(p);
        /* a_col = (1, 0.5, 0.25, 0.75) -> mat2 columns (1, .5) and (.25, .75):
         * the FS receives them back through all five stages */
        const float pos[3][4] = { { -1,-1,0,1 }, { 0,1,0,1 }, { 1,-1,0,1 } };
        const float col[3][4] = { { 1,0.5f,0.25f,0.75f },
                                   { 1,0.5f,0.25f,0.75f },
                                   { 1,0.5f,0.25f,0.75f } };
        setup_patch_vao(pos, col, 3);
        draw_reference_patch();
        EXPECT_PIXEL(100, 190, 255, 128, 64, 191, 6);
        EXPECT_PIXEL(100, 5,   255, 128, 64, 191, 6);
    }
    glDeleteProgram(p);
}

/* --- gl_Layer from the vertex / evaluation stage (ARB_shader_viewport_layer_array)
 *
 * Metal needs MTLRenderPipelineDescriptor.inputPrimitiveTopology once the final
 * pre-rasterization function writes [[render_target_array_index]] (gl_Layer), and
 * rejects a [[point_size]] write under a non-Point topology — the rstate patcher
 * therefore skips the synthetic point-size member for layer writers.  Both guards
 * (tessConfigureLayeredTopology + state_ext.c) are exercised here: VS-only and
 * TES (the post-tessellation vertex function is the TES). */
static GLuint make_layered_fbo(GLuint *tex, int layers)
{
    GLuint fbo;
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, *tex);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 64, 64, layers);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, *tex, 0);
    GLenum db = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &db);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 64, 64);
    return fbo;
}

static float read_layer(GLuint tex, int layer, int x, int y)
{
    GLuint rfbo;
    float c[4] = { 0, 0, 0, 0 };
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, layer);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
    glDeleteFramebuffers(1, &rfbo);
    return c[0];
}

TEST(TessLayerFromVertexStage)
{
    const char *vs =
        "#version 450 core\n"
        "#extension GL_ARB_shader_viewport_layer_array: require\n"
        "layout(location=0) in vec4 a_pos;\n"
        "layout(location=0) out vec4 v_col;\n"
        "uniform int u_layer;\n"
        "void main()\n"
        "{\n"
        "  gl_Position = a_pos;\n"
        "  gl_Layer = u_layer;\n"
        "  v_col = vec4(1, 0, 0, 1);\n"
        "}\n";
    static const float quad[4][4] = {
        { -1, -1, 0, 1 }, { -1, 1, 0, 1 }, { 1, -1, 0, 1 }, { 1, 1, 0, 1 },
    };
    GLuint p = mgl_make_program5(vs, NULL, NULL, NULL, kFs);
    EXPECT_TRUE(p != 0);
    if (!p) return;

    GLuint tex, fbo = make_layered_fbo(&tex, 4);
    setup_patch_vao(quad, quad, 4);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_layer"), 2);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    printf("  [vs layer] layer0=%.2f layer2=%.2f (want 0.0 1.0)\n",
           read_layer(tex, 0, 32, 32), read_layer(tex, 2, 32, 32));
    EXPECT_NEAR(read_layer(tex, 2, 32, 32), 1.0f, 1e-4);
    EXPECT_NEAR(read_layer(tex, 0, 32, 32), 0.0f, 1e-4);
    EXPECT_NEAR(read_layer(tex, 3, 32, 32), 0.0f, 1e-4);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

TEST(TessLayerFromEvaluationStage)
{
    const char *tcs = kTcsTri;
    const char *tes =
        "#version 450 core\n"
        "#extension GL_ARB_shader_viewport_layer_array: require\n"
        "layout(triangles, equal_spacing, ccw) in;\n"
        "layout(location=0) in vec4 tc_col[];\n"
        "layout(location=0) out vec4 f_col;\n"
        "uniform int u_layer;\n"
        "void main() {\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position\n"
        "               + gl_TessCoord.y * gl_in[1].gl_Position\n"
        "               + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  f_col = vec4(1, 0, 0, 1);\n"
        "  gl_Layer = u_layer;\n"
        "}\n";
    GLuint p = mgl_make_program5(kVs, tcs, tes, NULL, kFs);
    EXPECT_TRUE(p != 0);
    if (!p) return;

    GLuint tex, fbo = make_layered_fbo(&tex, 4);
    setup_patch_vao(kTriPos, kTriCol, 3);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_layer"), 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    printf("  [tes layer] layer0=%.2f layer1=%.2f (want 0.0 1.0)\n",
           read_layer(tex, 0, 16, 16), read_layer(tex, 1, 16, 16));
    EXPECT_NEAR(read_layer(tex, 1, 16, 16), 1.0f, 1e-4);
    EXPECT_NEAR(read_layer(tex, 0, 16, 16), 0.0f, 1e-4);
    EXPECT_NEAR(read_layer(tex, 3, 16, 16), 0.0f, 1e-4);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

/* ------------------------------------------------------------------ post_depth_coverage
 * KHR-GL46.post_depth_coverage_tests guards.  GL_ARB_post_depth_coverage is
 * deliberately NOT advertised (r41): Metal's [[sample_mask,
 * post_depth_coverage]] misses stencil-failed samples, so no shader may use
 * the feature — the CTS group is extension-gated (2 NotSupported) and
 * layout(post_depth_coverage) fails the link (tessellation.c rejects the
 * PostDepthCoverage execution mode).
 *
 * Test 1 — the extension string is gone, and the PostDepthShader programs:
 * `require`/`enable` without the layout still compile (glslang knows the
 * extension name; nothing depends on it), while every
 * layout(post_depth_coverage) program must FAIL the link with a clear log.
 * Test 2 — the plain gl_SampleMaskIn rendering (GL 4.0 core, no extension):
 * a 1x1 4x-MSAA depth/stencil FBO; a "near" triangle stamps depth/stencil,
 * a "far" triangle fails both tests everywhere, and — without
 * post_depth_coverage — the mask reflects COVERAGE, so it must read
 * {1,1,1,1} for every sample in both the depth and the stencil case,
 * resolved per-sample into a 1x4 texture. */
TEST(FragPostDepthShader)
{
    const char *vs =
        "#version 450\n"
        "in vec3 vertex;\n"
        "in vec2 inTexCoord;\n"
        "out vec2 texCoord;\n"
        "void main() { texCoord = inTexCoord; gl_Position = vec4(vertex, 1); }\n";
    const char *f1 =
        "#version 450\n"
        "#extension GL_ARB_post_depth_coverage : require\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    const char *f2 =
        "#version 450\n"
        "#extension GL_ARB_post_depth_coverage : enable\n"
        "#ifndef GL_ARB_post_depth_coverage\n"
        "  #error GL_ARB_post_depth_coverage not defined\n"
        "#else\n"
        "  #if (GL_ARB_post_depth_coverage != 1)\n"
        "    #error GL_ARB_post_depth_coverage wrong defined\n"
        "  #endif\n"
        "#endif\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    const char *f3 =
        "#version 450\n"
        "#extension GL_ARB_post_depth_coverage : enable\n"
        "layout(early_fragment_tests) in;\n"
        "layout(post_depth_coverage) in;\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";
    const char *f4 =
        "#version 450\n"
        "#extension GL_ARB_post_depth_coverage : enable\n"
        "layout(post_depth_coverage) in;\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    /* the extension must NOT be in GL_EXTENSIONS / glGetStringi */
    {
        GLint n = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        EXPECT_TRUE(n > 0);
        for (GLint i = 0; i < n; i++)
        {
            const GLubyte *e = glGetStringi(GL_EXTENSIONS, (GLuint)i);
            EXPECT_TRUE(e != NULL);
            if (e && strcmp((const char *)e, "GL_ARB_post_depth_coverage") == 0)
                MGL_FAIL("GL_ARB_post_depth_coverage is still advertised");
        }
    }

    /* require/enable without the feature use: glslang accepts the name, and
     * nothing in the program depends on the unsupported behaviour */
    GLuint p1 = mgl_make_program5(vs, NULL, NULL, NULL, f1);
    GLuint p2 = mgl_make_program5(vs, NULL, NULL, NULL, f2);
    EXPECT_TRUE(p1 != 0); EXPECT_TRUE(p2 != 0);
    glDeleteProgram(p1); glDeleteProgram(p2);

    /* every layout(post_depth_coverage) program must fail the link */
    for (int i = 0; i < 2; i++)
    {
        GLuint vs2 = glCreateShader(GL_VERTEX_SHADER);
        GLuint fs2 = glCreateShader(GL_FRAGMENT_SHADER);
        GLuint p = glCreateProgram();
        const char *fs_src = i ? f4 : f3;
        GLint ok = 0;
        char log[512] = "";

        glShaderSource(vs2, 1, &vs, NULL);
        glCompileShader(vs2);
        glShaderSource(fs2, 1, &fs_src, NULL);
        glCompileShader(fs2);
        glAttachShader(p, vs2);
        glAttachShader(p, fs2);
        glLinkProgram(p);
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        if (!ok) glGetProgramInfoLog(p, sizeof log, NULL, log);
        if (ok)
            MGL_FAIL("layout(post_depth_coverage) program %d linked although "
                     "GL_ARB_post_depth_coverage is not supported", i);
        if (!strstr(log, "post_depth_coverage"))
            MGL_FAIL("post_depth_coverage link rejection lacks a clear log: %s", log);
        glDeleteProgram(p);
        glDeleteShader(vs2);
        glDeleteShader(fs2);
    }
    EXPECT_NO_GL_ERROR();
}

TEST(FragPostDepthCoverageMask)
{
    const char *vs =
        "#version 450\n"
        "in vec3 vertex;\n"
        "in vec2 inTexCoord;\n"
        "out vec2 texCoord;\n"
        "void main() { texCoord = inTexCoord; gl_Position = vec4(vertex, 1); }\n";
    const char *body =
        "in vec2 texCoord;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "    int s1 = (gl_SampleMaskIn[0] & 0x1);\n"
        "    int s2 = (gl_SampleMaskIn[0] & 0x2) / 2;\n"
        "    int s3 = (gl_SampleMaskIn[0] & 0x4) / 4;\n"
        "    int s4 = (gl_SampleMaskIn[0] & 0x8) / 8;\n"
        "    fragColor = vec4(s1, s2, s3, s4);\n"
        "}\n";
    char fPlain[1024], fPdc[1024];
    snprintf(fPlain, sizeof fPlain, "#version 450\n%s", body);
    snprintf(fPdc, sizeof fPdc,
             "#version 450\n#extension GL_ARB_post_depth_coverage : enable\n"
             "layout(post_depth_coverage) in;\n%s", body);
    const char *fResolve =
        "#version 450\n"
        "in vec2 texCoord;\n"
        "out vec4 fragColor;\n"
        "uniform sampler2DMS tex;\n"
        "void main() { fragColor = texelFetch(tex, ivec2(0, 0), int(texCoord.y * 4)); }\n";

    GLuint pA = mgl_make_program5(vs, NULL, NULL, NULL, fPlain);
    GLuint pR = mgl_make_program5(vs, NULL, NULL, NULL, fResolve);
    EXPECT_TRUE(pA != 0); EXPECT_TRUE(pR != 0);
    if (!pA || !pR) return;

    /* the post_depth_coverage program must fail the link (r41: the extension
     * is not supported and is not advertised) */
    {
        GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
        GLuint pB = glCreateProgram();
        GLint ok = 0;
        char log[512] = "";
        const char *fsrc = fPdc;
        glShaderSource(s, 1, &fsrc, NULL);
        glCompileShader(s);
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        EXPECT_TRUE(ok);
        glAttachShader(pB, s);
        glLinkProgram(pB);
        glGetProgramiv(pB, GL_LINK_STATUS, &ok);
        if (ok) glGetProgramInfoLog(pB, sizeof log, NULL, log);
        EXPECT_TRUE(!ok);
        if (ok) MGL_FAIL("layout(post_depth_coverage) linked although the extension is not supported (%s)", log);
        glDeleteProgram(pB);
        glDeleteShader(s);
    }
    EXPECT_NO_GL_ERROR();

    /* 1x1 4x-MSAA target + depth-stencil, and a 1x4 single-sample resolve target */
    GLuint rb = 0, texMS = 0, fboMS = 0, tex = 0, fbo = 0;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_STENCIL, 1, 1);
    EXPECT_NO_GL_ERROR();
    glGenTextures(1, &texMS);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texMS);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, 1, 1, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    glGenFramebuffers(1, &fboMS);
    glBindFramebuffer(GL_FRAMEBUFFER, fboMS);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, texMS, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 4);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* geometry: near covers only the samples with sx+sy < 0.7 (sample 0),
     * far covers the whole pixel, post is the full-screen resolve quad */
    static const GLfloat nearV[] = { -1,-1,0.25f,  0.4f,-1,0.25f,  -1,0.25f,0.25f };
    static const GLfloat farV[]  = { -1,-1,0.75f,  3,-1,0.75f,     -1,3,0.75f };
    static const GLfloat postV[] = { -1,-1,0,1,  1,-1,0,1,  -1,1,0,1,  1,1,0,1 };
    static const GLfloat tcs[]   = { 0,0, 1,0, 0,1, 1,1 };
    GLuint vao = 0, vbo = 0, tbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tcs, tcs, GL_STATIC_DRAW);

    GLint aVertex = glGetAttribLocation(pA, "vertex");
    GLint aTexCoord = glGetAttribLocation(pA, "inTexCoord");
    EXPECT_TRUE(aVertex >= 0 && aTexCoord >= 0);

    for (int stencilCase = 0; stencilCase <= 1; stencilCase++)
    {
        GLuint prog = pA;
        glBindFramebuffer(GL_FRAMEBUFFER, fboMS);
        glViewport(0, 0, 1, 1);
        glEnable(GL_MULTISAMPLE);
        if (stencilCase) glEnable(GL_STENCIL_TEST); else glEnable(GL_DEPTH_TEST);
        glClearColor(0, 0, 0, 0);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        if (stencilCase) {
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilOp(GL_ZERO, GL_REPLACE, GL_REPLACE);
            glStencilMask(0xFF);
        }

        /* without post_depth_coverage the sample mask is plain coverage
         * (GL 4.0 core): every sample of the covered pixel reads 1,
         * no matter what the depth/stencil tests did to the fragments */
        GLboolean expected[4] = { GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE };

        glBindVertexArray(vao);
        glUseProgram(prog);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof nearV, nearV, GL_STATIC_DRAW);
        glEnableVertexAttribArray((GLuint)aVertex);
        glVertexAttribPointer((GLuint)aVertex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, tbo);
        glEnableVertexAttribArray((GLuint)aTexCoord);
        glVertexAttribPointer((GLuint)aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);            /* near: stamps z/stencil */

        glClear(GL_COLOR_BUFFER_BIT);
        if (stencilCase) { glStencilFunc(GL_NOTEQUAL, 1, 0xFF); glStencilMask(0x00); }

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof farV, farV, GL_STATIC_DRAW);
        glVertexAttribPointer((GLuint)aVertex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);            /* far: fails where near wrote */

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);

        /* resolve each sample into its own row */
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, 1, 4);
        glUseProgram(pR);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, texMS);
        glUniform1i(glGetUniformLocation(pR, "tex"), 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof postV, postV, GL_STATIC_DRAW);
        glVertexAttribPointer((GLuint)aVertex, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisable(GL_MULTISAMPLE);

        GLubyte px[16];
        memset(px, 0, sizeof px);
        glReadPixels(0, 0, 1, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);
        printf("    rows (sample0..3): %d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d\n",
               px[0], px[1], px[2], px[3], px[4], px[5], px[6], px[7],
               px[8], px[9], px[10], px[11], px[12], px[13], px[14], px[15]);
        GLboolean got[4] = { GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE };
        for (int i = 0; i < 4; i++)
            for (int n = 0; n < 4; n++)
                got[n] = got[n] || (px[4 * i + n] != 0);
        int ok = 1;
        for (int n = 0; n < 4; n++) if (got[n] != expected[n]) ok = 0;
        mgl_test_checks++;
        if (!ok)
            MGL_FAIL("sample mask coverage (%s): mask %d%d%d%d expected %d%d%d%d",
                     stencilCase ? "stencil" : "depth",
                     got[0], got[1], got[2], got[3],
                     expected[0], expected[1], expected[2], expected[3]);
        else
            printf("  [coverage %s] mask %d%d%d%d\n",
                   stencilCase ? "stencil" : "depth", got[0], got[1], got[2], got[3]);
        EXPECT_NO_GL_ERROR();
    }

    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo); glDeleteBuffers(1, &tbo);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fboMS); glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &texMS); glDeleteTextures(1, &tex);
    glDeleteRenderbuffers(1, &rb);
    glDeleteProgram(pA); glDeleteProgram(pR);
}

/* KHR-GL46.shader_image_load_store.early-fragment-tests regression guard
 * (GL 4.6 §17.3.4): a fragment shader that writes to memory and does NOT
 * declare layout(early_fragment_tests) must execute before the depth test —
 * its imageStore side effect happens even when the fragment later fails the
 * depth test.  Apple GPUs cull depth-failed fragments before shading (HSR);
 * MGL emulates the late test with a guarded discard_fragment() + [[depth(any)]]
 * output variant (tessellation.c / MGLRenderer+RenderState.m).  With
 * early_fragment_tests declared the shader must NOT run for depth-failed
 * fragments.  Both the plain and the geometry-shader draw path are covered. */
static void lft_draw(GLuint prog, const float *pos3, int count, GLenum mode)
{
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(count * 3 * sizeof(float)), pos3, GL_STATIC_DRAW);
    GLint aV = glGetAttribLocation(prog, "a_pos");
    if (aV >= 0) {
        glEnableVertexAttribArray((GLuint)aV);
        glVertexAttribPointer((GLuint)aV, 3, GL_FLOAT, GL_FALSE, 0, 0);
    }
    glDrawArrays(mode, 0, count);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

TEST(FragLateTestsSideEffects)
{
    const char *vs =
        "#version 400 core\n"
        "in vec3 a_pos;\n"
        "void main() { gl_Position = vec4(a_pos, 1); }\n";
    const char *gs =
        "#version 400 core\n"
        "layout(points) in;\n"
        "layout(triangle_strip, max_vertices = 4) out;\n"
        "void main() {\n"
        "    gl_Position = vec4(-1,-1,0.5,1); EmitVertex();\n"
        "    gl_Position = vec4( 3,-1,0.5,1); EmitVertex();\n"
        "    gl_Position = vec4(-1, 3,0.5,1); EmitVertex();\n"
        "    gl_Position = vec4( 3, 3,0.5,1); EmitVertex();\n"
        "    EndPrimitive();\n"
        "}\n";
    const char *fs_head =
        "#version 400 core\n"
        "#extension GL_ARB_shader_image_load_store : require\n"
        "layout(r32f) uniform image2D u_image;\n";
    const char *fs_tail =
        "void main() {\n"
        "    imageStore(u_image, ivec2(gl_FragCoord.xy), vec4(1.0, 0, 0, 0));\n"
        "    discard;\n"
        "}\n";
    char fs_late[512], fs_early[512];
    snprintf(fs_late, sizeof fs_late, "%s%s", fs_head, fs_tail);
    snprintf(fs_early, sizeof fs_early, "%slayout(early_fragment_tests) in;\n%s", fs_head, fs_tail);

    GLuint pLate = mgl_make_program5(vs, NULL, NULL, NULL, fs_late);
    GLuint pEarly = mgl_make_program5(vs, NULL, NULL, NULL, fs_early);
    GLuint pGsLate = mgl_make_program3(vs, gs, fs_late);
    GLuint pGsEarly = mgl_make_program3(vs, gs, fs_early);
    EXPECT_TRUE(pLate && pEarly && pGsLate && pGsEarly);
    if (!pLate || !pEarly || !pGsLate || !pGsEarly) return;

    GLuint tex = 0, rb = 0, rbds = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, 1, 1);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_R32F, 1, 1);
    glGenRenderbuffers(1, &rbds);
    glBindRenderbuffer(GL_RENDERBUFFER, rbds);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 1, 1);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbds);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 1, 1);

    static const GLfloat tri[] = { -1,-1,0.5f,  3,-1,0.5f,  -1,3,0.5f };  /* z = cleared 0.5 */
    static const GLfloat pts[] = { 0,0,0 };
    float zero = 0.0f, one;

    for (int gs = 0; gs <= 1; gs++)
    for (int early = 0; early <= 1; early++)
    {
        GLuint prog = gs ? (early ? pGsEarly : pGsLate) : (early ? pEarly : pLate);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RED, GL_FLOAT, &zero);
        glUseProgram(prog);
        glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
        glUniform1i(glGetUniformLocation(prog, "u_image"), 0);
        glClearDepth(0.5f);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);               /* z 0.5 < cleared 0.5 fails */
        if (gs) lft_draw(prog, pts, 1, GL_POINTS);
        else lft_draw(prog, tri, 3, GL_TRIANGLES);
        glDisable(GL_DEPTH_TEST);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glBindTexture(GL_TEXTURE_2D, tex);
        one = -1;
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, &one);
        mgl_test_checks++;
        int want = early ? 0 : 1;
        if ((int)one != want) {
            MGL_FAIL("late fragment tests (%s, early=%d): image red = %g, want %d",
                     gs ? "GS path" : "direct", early, one, want);
        } else {
            printf("  [lft %s early=%d] image red = %g\n", gs ? "GS" : "direct", early, one);
        }
        EXPECT_NO_GL_ERROR();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb); glDeleteRenderbuffers(1, &rbds);
    glDeleteTextures(1, &tex);
    glDeleteProgram(pLate); glDeleteProgram(pEarly); glDeleteProgram(pGsLate);
    glDeleteProgram(pGsEarly);
}

/* ---- gl_DrawID across the link cache (r40) --------------------------------
 * A program whose vertex stage reads gl_DrawID / gl_BaseVertex binds the
 * draw-parameters buffer per draw (execTessBindDrawParams).  The whole-link
 * executable cache (program.c) clones a linked program for an identical
 * source set; exec_clone must carry uses_draw_params or the clone never binds
 * the buffer and every draw sees gl_DrawID == 0 - KHR-GL46
 * shader_draw_parameters_tests.ShaderMultiDraw* (the single-draw cases pass
 * because gl_DrawID 0 is their expected value only in draw 0).  Sequence that
 * fails: link+draw a first program, then relink the same sources and draw a
 * multi-draw with it. */
TEST(DrawIdSurvivesLinkCache)
{
    static const char *vs =
        "#version 460\n"
        "in vec3 vertex;\n"
        "out vec3 color;\n"
        "void main()\n"
        "{\n"
        "    float hOffset = float(gl_BaseVertex) / 5.0;\n"
        "    float vOffset = float(gl_BaseInstance) / 5.0;\n"
        "    color = vec3(0.0);\n"
        "    if (gl_DrawID % 3 == 0) color.r = 1;\n"
        "    else if (gl_DrawID % 3 == 1) color.g = 1;\n"
        "    else if (gl_DrawID % 3 == 2) color.b = 1;\n"
        "    gl_Position = vec4(vertex + vec3(hOffset, vOffset, 0), 1);\n"
        "}\n";
    static const char *fs =
        "#version 460\n"
        "in vec3 color;\n"
        "out vec4 fragColor;\n"
        "void main() { fragColor = vec4(color, 1.0); }\n";
    static const GLfloat verts[] = {
        -1.0f, -1.0f, 0.0f, -1.0f, -0.8f, 0.0f, -0.9f, -1.0f, 0.0f,
        -0.9f, -0.8f, 0.0f, -0.8f, -1.0f, 0.0f, -0.8f, -0.8f, 0.0f,
    };

    /* program 1: linked fresh (draw-params flag set on the link path) */
    GLuint p1 = mgl_make_program(vs, fs);
    EXPECT_TRUE(p1 != 0);
    glUseProgram(p1);
    GLuint vbo; glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    GLuint vao; glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLint loc = glGetAttribLocation(p1, "vertex");
    glEnableVertexAttribArray((GLuint)loc);
    glVertexAttribPointer((GLuint)loc, 3, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 256, 256);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glClearColor(0, 0, 0, 1);

    /* the single draw first (gl_DrawID == 0 there) */
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 1, 4);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p1);
    p1 = 0;

    /* program 2: SAME sources - served from the whole-link cache */
    GLuint p2 = mgl_make_program(vs, fs);
    EXPECT_TRUE(p2 != 0);
    glUseProgram(p2);
    loc = glGetAttribLocation(p2, "vertex");
    glEnableVertexAttribArray((GLuint)loc);
    glVertexAttribPointer((GLuint)loc, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    const GLint dFirst[] = {0, 1};
    const GLsizei dCount[] = {4, 4};
    glMultiDrawArrays(GL_TRIANGLE_STRIP, dFirst, dCount, 2);
    EXPECT_NO_GL_ERROR();

    /* (-0.95,-0.85): covered by both draws -> red+green; (-0.85,-0.95): draw 1 only */
    GLubyte both[4], green[4], red[4], none[4];
    mgl_read_rgba(6, 19, both);
    mgl_read_rgba(19, 6, green);
    mgl_read_rgba(6, 6, red);
    mgl_read_rgba(19, 19, none);
    printf("  [drawid cache-hit] both=%d,%d,%d green=%d,%d,%d red=%d,%d,%d none=%d,%d,%d\n",
           both[0], both[1], both[2], green[0], green[1], green[2],
           red[0], red[1], red[2], none[0], none[1], none[2]);
    EXPECT_EQ_INT(both[0], 255); EXPECT_EQ_INT(both[1], 255); EXPECT_EQ_INT(both[2], 0);
    EXPECT_EQ_INT(green[0], 0);  EXPECT_EQ_INT(green[1], 255);
    EXPECT_EQ_INT(red[0], 255);  EXPECT_EQ_INT(red[1], 0);
    EXPECT_EQ_INT(none[0], 0);   EXPECT_EQ_INT(none[1], 0);

    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(p2);
}

/* ---- gl_PerVertex block rewrite below GLSL 4.50 (r40) ----------------------
 * A #version 130 vertex shader redeclaring gl_ClipDistance loose must link:
 * the loose member is folded into an out gl_PerVertex block that may not carry
 * gl_CullDistance below GLSL 4.50 ("block redeclaration has extra members").
 * The negative form (size gl_MaxClipDistances+1) must fail to compile - the
 * fold must not clamp an over-range size to 8.  Both from KHR-GL46
 * clip_distance.functional/coverage/negative. */
TEST(ClipDistance130Link)
{
    static const char *vs_ok =
        "#version 130\n"
        "out float gl_ClipDistance[1];\n"
        "float f(int i) { return 0.0; }\n"
        "in vec4 position;\n"
        "void main()\n"
        "{\n"
        "    for(int i = 0; i < 1; i++) gl_ClipDistance[i] = f(i);\n"
        "    gl_Position = position;\n"
        "}\n";
    static const char *vs_neg =
        "#version 130\n"
        "out float gl_ClipDistance[gl_MaxClipDistances + 1];\n"
        "void main()\n"
        "{\n"
        "    gl_ClipDistance[0] = 0.0;\n"
        "    gl_Position = vec4(1.0);\n"
        "}\n";
    static const char *fs =
        "#version 130\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    GLuint p = mgl_make_program(vs_ok, fs);
    EXPECT_TRUE(p != 0);
    if (p)
    {
        glUseProgram(p);
        EXPECT_NO_GL_ERROR();
        glDeleteProgram(p);
    }

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vs_neg, NULL);
    glCompileShader(vs);
    GLint ok = 0; glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_FALSE);          /* over-maximum redeclaration */
    glDeleteShader(vs);
    while (glGetError() != GL_NO_ERROR) {}
}

/* ---- CPU-fallback regressions (dEQP-GL45-ES31 tessellation.primitive_discard) ----
 *
 * The CTS shape: hundreds of patches in one draw.  The GPU emulated tessellator's
 * worst-case scratch (>= ~165 KB/patch) busts its 64 MB budget around 400 patches,
 * so the draw falls back to the CPU reference path AFTER the VS/TCS kernels ran.
 * Three independent bugs lived on that path, all fixed:
 *   1. the CPU blob builder skipped TCS patch outputs whose canonical key is 0
 *      (GL location 0) — the blob member stayed zero and the TES produced garbage
 *      positions (all-black image);
 *   2. the plain TES variant was not compiled for the LATE fallback (the condition
 *      only knew gpu_emulated || use_cpu_reference), so the plain indexed draw ran
 *      with a post-tessellation pipeline + tessellation vertex descriptor and
 *      rasterized nothing;
 *   3. transform-feedback capture recorded one record per UNIQUE vertex instead of
 *      one per EMITTED vertex in emission order (wrong record count).
 * GL requires XFB capture AND rasterization from the same draw: the image checks
 * hold with glBeginTransformFeedback active. */
static const char *kPdVs =
    "#version 450\n"
    "out gl_PerVertex { vec4 gl_Position; };\n"
    "in float in_v_attr;\n"
    "out float in_tc_attr;\n"
    "void main() { in_tc_attr = in_v_attr; }\n";
static const char *kPdFs =
    "#version 450\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() { o_color = vec4(1.0); }\n";
static const char *kPdTcs =
    "#version 450\n"
    "layout(vertices = 1) out;\n"
    "in float in_tc_attr[];\n"
    "patch out vec2 in_te_scale;    /* implicit location 0: canonical key 0 */\n"
    "patch out vec2 in_te_offset;\n"
    "void main() {\n"
    "    in_te_scale  = vec2(in_tc_attr[6], in_tc_attr[7]);\n"
    "    in_te_offset = vec2(in_tc_attr[8], in_tc_attr[9]);\n"
    "    gl_TessLevelInner[0] = in_tc_attr[0];\n"
    "    gl_TessLevelInner[1] = in_tc_attr[1];\n"
    "    gl_TessLevelOuter[0] = in_tc_attr[2];\n"
    "    gl_TessLevelOuter[1] = in_tc_attr[3];\n"
    "    gl_TessLevelOuter[2] = in_tc_attr[4];\n"
    "    gl_TessLevelOuter[3] = in_tc_attr[5];\n"
    "}\n";
static const char *kPdTes =
    "#version 450\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "patch in vec2 in_te_scale;\n"
    "patch in vec2 in_te_offset;\n"
    "out vec3 out_te_tessCoord;\n"
    "void main() {\n"
    "    out_te_tessCoord = gl_TessCoord;\n"
    "    gl_Position = vec4(gl_TessCoord.xy * in_te_scale + in_te_offset, 0.0, 1.0);\n"
    "}\n";

#define PD_COLS 32
#define PD_ROWS 16
#define PD_PATCHES (PD_COLS * PD_ROWS)   /* 512: beyond the 64 MB emulated budget */

static GLuint pd_setup(void)
{
    /* per-vertex float: [0..5] levels, [6,7] scale, [8,9] offset — 10 verts/patch */
    static float attr[PD_PATCHES * 10];
    GLuint vao, buf;

    for (int i = 0; i < PD_PATCHES; i++) {
        float *p = attr + i * 10;
        p[0] = 1; p[1] = 1;              /* inner: minimal grid */
        p[2] = 3; p[3] = 3; p[4] = 3; p[5] = 3;   /* outer */
        p[6] = 2.0f / PD_COLS; p[7] = 2.0f / PD_ROWS;
        p[8] = (float)(i % PD_COLS) / PD_COLS * 2.0f - 1.0f;
        p[9] = (float)(i / PD_COLS) / PD_ROWS * 2.0f - 1.0f;
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &buf);
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, sizeof attr, attr, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, NULL);
    glPatchParameteri(GL_PATCH_VERTICES, 10);
    return vao;
}

/* pixel inside grid cell (col,row) of a 256x256 viewport: the triangle-domain
 * patch covers the lower-left TRIANGLE of its cell (u+v<=1), so probe at a
 * quarter point — the cell centre lies exactly on the hypotenuse (u=v=0.5),
 * which the rasterizer's fill rule may or may not cover */
static void pd_cell(int col, int row, float out[4])
{
    int x = (int)((col + 0.25f) * (256.0f / PD_COLS));
    int y = (int)((row + 0.25f) * (256.0f / PD_ROWS));

    glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, out);
}

TEST(TessCpuFallbackPatchBlobKey0)
{
    /* state isolation: earlier suite tests leave cull / depth / scissor / stencil
     * enabled, and can leave a PIXEL_PACK_BUFFER / read-FBO bound (glReadPixels
     * then lands in the buffer, not the probed array) — the pixel probes below
     * assume an unfiltered raster and a clean readback path */
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glViewport(0, 0, 256, 256);
    GLuint vao = pd_setup();
    GLuint prog = glCreateProgram();
    struct { int t; const char *s; } st[4] = {
        { GL_VERTEX_SHADER, kPdVs }, { GL_FRAGMENT_SHADER, kPdFs },
        { GL_TESS_CONTROL_SHADER, kPdTcs }, { GL_TESS_EVALUATION_SHADER, kPdTes },
    };
    for (int i = 0; i < 4; i++) {
        GLuint sh = glCreateShader(st[i].t);
        glShaderSource(sh, 1, &st[i].s, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(prog);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link != GL_TRUE) {
        char log[1024] = "";
        glGetProgramInfoLog(prog, sizeof log, NULL, log);
        printf("TessCpuFallbackPatchBlobKey0 link: %s\n", log);
        glDeleteProgram(prog);
        glDeleteVertexArrays(1, &vao);
        return;
    }
    glUseProgram(prog);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, PD_PATCHES * 10);
    EXPECT_NO_GL_ERROR();

    /* every cell must be lit by its patch: the blob carries the per-patch scale
     * and offset (the location-0 member was the all-black bug) */
    float c[4];
    int lit = 0, probes = 0;
    for (int row = 0; row < PD_ROWS; row++)
        for (int col = 0; col < PD_COLS; col++) {
            pd_cell(col, row, c);
            probes++;
            if (c[0] > 0.5f) lit++;
        }
    EXPECT_EQ_INT(lit, probes);
    pd_cell(0, 0, c);          EXPECT_TRUE(c[0] > 0.5f);
    pd_cell(PD_COLS - 1, PD_ROWS - 1, c); EXPECT_TRUE(c[0] > 0.5f);

    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
}

TEST(TessXfbCpuFallbackRasterizes)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glViewport(0, 0, 256, 256);
    GLuint vao = pd_setup();
    GLuint prog = glCreateProgram();
    struct { int t; const char *s; } st[4] = {
        { GL_VERTEX_SHADER, kPdVs }, { GL_FRAGMENT_SHADER, kPdFs },
        { GL_TESS_CONTROL_SHADER, kPdTcs }, { GL_TESS_EVALUATION_SHADER, kPdTes },
    };
    for (int i = 0; i < 4; i++) {
        GLuint sh = glCreateShader(st[i].t);
        glShaderSource(sh, 1, &st[i].s, NULL);
        glCompileShader(sh);
        glAttachShader(prog, sh);
        glDeleteShader(sh);
    }
    const char *xfb_v[] = { "out_te_tessCoord" };
    glTransformFeedbackVaryings(prog, 1, xfb_v, GL_SEPARATE_ATTRIBS);
    glLinkProgram(prog);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (link != GL_TRUE) { glDeleteProgram(prog); glDeleteVertexArrays(1, &vao); return; }
    glUseProgram(prog);

    GLuint xfb, qgen, qwr;
    glGenBuffers(1, &xfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 1 << 22, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glGenQueries(1, &qgen);
    glGenQueries(1, &qwr);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBeginQuery(GL_PRIMITIVES_GENERATED, qgen);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, qwr);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_PATCHES, 0, PD_PATCHES * 10);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glEndQuery(GL_PRIMITIVES_GENERATED);
    EXPECT_NO_GL_ERROR();

    GLint gen = -1, wr = -1;
    glGetQueryObjectiv(qgen, GL_QUERY_RESULT, &gen);
    glGetQueryObjectiv(qwr, GL_QUERY_RESULT, &wr);
    EXPECT_TRUE(gen > 0);
    EXPECT_TRUE(wr > 0);
    EXPECT_EQ_INT(gen, wr);

    /* records: one per EMITTED vertex in emission order — 3 per triangle, and the
     * three tcoords of a triangle are distinct barycentrics (a UNIQUE-vertex-order
     * stream repeats the same coordinates) */
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb);
    const float *rec = (const float *)glMapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, GL_READ_ONLY);
    EXPECT_TRUE(rec != NULL);
    if (rec) {
        int ntri = wr < 4096 ? wr : 4096;
        int bad = 0;
        for (int t = 0; t < ntri; t++) {
            const float *a = rec + (size_t)t * 9, *b = a + 3, *c = b + 3;
            for (int k = 0; k < 9; k++)
                if (!(rec[(size_t)t * 9 + k] >= 0.0f && rec[(size_t)t * 9 + k] <= 1.0f)) bad++;
            if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) bad++;   /* degenerate dup */
            if (b[0] == c[0] && b[1] == c[1] && b[2] == c[2]) bad++;
            if (a[0] == c[0] && a[1] == c[1] && a[2] == c[2]) bad++;
        }
        EXPECT_EQ_INT(bad, 0);
        glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
    }
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);

    /* GL 4.6: transform feedback does NOT disable rasterization — the image must
     * be identical to the non-capturing draw (this was the all-black failure) */
    float c[4];
    int lit = 0, probes = 0;
    for (int row = 0; row < PD_ROWS; row++)
        for (int col = 0; col < PD_COLS; col++) {
            pd_cell(col, row, c);
            probes++;
            if (c[0] > 0.5f) lit++;
        }
    EXPECT_EQ_INT(lit, probes);

    glDeleteQueries(1, &qgen);
    glDeleteQueries(1, &qwr);
    glDeleteBuffers(1, &xfb);
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
}

    /* ---- GL_ARB_ES3_2_compatibility ----------------------------------------- */

/* glPrimitiveBoundingBoxARB is exported and dlsym-resolvable (the CTS binds
 * entry points by name), the state is queryable in every glGet type, and the
 * initial value is the spec's (-1,-1,-1,1,1,1,1,1).  It is a rendering hint
 * only (spec 23.10 / §14.1pbb: fragments outside the box "may or may not be
 * generated"), so no rendering effect is expected or verified. */
TEST(TessES32EntryAndState)
{
    typedef void (*pbb_t)(float, float, float, float, float, float, float, float);
    pbb_t fn = (pbb_t)dlsym(RTLD_DEFAULT, "glPrimitiveBoundingBoxARB");

    EXPECT_NE((void *)fn, NULL);
    if (!fn)
        return;

    /* initial value */
    float v[8] = { 0 };
    glGetFloatv(GL_PRIMITIVE_BOUNDING_BOX_ARB, v);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (v[0] != -1 || v[1] != -1 || v[2] != -1 || v[3] != 1 ||
        v[4] != 1 || v[5] != 1 || v[6] != 1 || v[7] != 1)
    {
        printf("  bad initial bbox: %g %g %g %g / %g %g %g %g\n",
               v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        MGL_FAIL("initial GL_PRIMITIVE_BOUNDING_BOX_ARB wrong");
    }

    fn(-2.0f, -0.75f, -1.5f, 1.25f, 0.5f, 0.25f, 2.5f, 2.0f);
    glGetFloatv(GL_PRIMITIVE_BOUNDING_BOX_ARB, v);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (v[0] != -2.0f || v[1] != -0.75f || v[2] != -1.5f || v[3] != 1.25f ||
        v[4] != 0.5f || v[5] != 0.25f || v[6] != 2.5f || v[7] != 2.0f)
        MGL_FAIL("glPrimitiveBoundingBoxARB state not stored");

    /* same state through the other glGet types (dEQP state_query group) */
    GLboolean b[8];
    glGetBooleanv(GL_PRIMITIVE_BOUNDING_BOX_ARB, b);
    mgl_test_checks++;
    if (b[0] != GL_TRUE || b[7] != GL_TRUE)
        MGL_FAIL("GetBooleanv(GL_PRIMITIVE_BOUNDING_BOX_ARB)");

    GLint i[8] = { 0 };
    glGetIntegerv(GL_PRIMITIVE_BOUNDING_BOX_ARB, i);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (i[0] != -2 || i[4] != 1)     /* GL rounds float->int to nearest */
    {
        printf("  int = %d %d %d %d / %d %d %d %d\n", i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]);
        MGL_FAIL("GetIntegerv(GL_PRIMITIVE_BOUNDING_BOX_ARB)");
    }

    GLint64 q[8] = { 0 };
    glGetInteger64v(GL_PRIMITIVE_BOUNDING_BOX_ARB, q);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (q[0] != -2)
    {
        printf("  int64 = %lld\n", (long long)q[0]);
        MGL_FAIL("GetInteger64v(GL_PRIMITIVE_BOUNDING_BOX_ARB)");
    }

    /* restore the default for other tests */
    fn(-1, -1, -1, 1, 1, 1, 1, 1);
}

TEST(TessES32LineWidthQueries)
{
    float r[2] = { 0, 0 }, g = 0;

    glGetFloatv(GL_MULTISAMPLE_LINE_WIDTH_RANGE_ARB, r);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (!(r[0] >= 1.0f && r[1] >= r[0]))
    {
        printf("  ms line width range = %g..%g\n", r[0], r[1]);
        MGL_FAIL("GL_MULTISAMPLE_LINE_WIDTH_RANGE_ARB < spec minimum (1,1)");
    }

    glGetFloatv(GL_MULTISAMPLE_LINE_WIDTH_GRANULARITY_ARB, &g);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (!(g > 0.0f))
        MGL_FAIL("GL_MULTISAMPLE_LINE_WIDTH_GRANULARITY_ARB must be positive");
}

/* desktop GLSL + '#extension GL_ARB_ES3_2_compatibility : require' + the
 * gl_BoundingBox[] per-patch output (the dEQP primitive_bounding_box TCS
 * specialization for a core context) links and draws; the bounding box is a
 * no-effect hint (see mgl_tess_es32_compat). */
TEST(TessES32DesktopBoundingBoxShader)
{
    static const char *vs =
        "#version 460 core\n"
        "out vec4 vs_col;\n"
        "void main()\n"
        "{\n"
        "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "    vs_col = vec4(0, 1, 0, 1);\n"
        "}\n";
    static const char *tcs =
        "#version 460 core\n"
        "#extension GL_ARB_ES3_2_compatibility : require\n"
        "layout(vertices=3) out;\n"
        "in vec4 vs_col[];\n"
        "out vec4 tcs_col[];\n"
        "void main()\n"
        "{\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "    tcs_col[gl_InvocationID] = vs_col[gl_InvocationID];\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "    gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n"
        "    gl_BoundingBox[0] = vec4(-1.0, -1.0, -1.0, 1.0);\n"
        "    gl_BoundingBox[1] = vec4( 1.0,  1.0,  1.0, 1.0);\n"
        "}\n";
    static const char *tes =
        "#version 460 core\n"
        "layout(triangles, equal_spacing) in;\n"
        "in vec4 tcs_col[];\n"
        "out vec4 tes_col;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                   gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                   gl_in[2].gl_Position * gl_TessCoord.z);\n"
        "    tes_col = tcs_col[0];\n"
        "}\n";
    static const char *fs =
        "#version 460 core\n"
        "in vec4 tes_col; out vec4 o;\n"
        "void main(){ o = tes_col; }\n";
    GLuint p = mgl_make_program5(vs, tcs, tes, NULL, fs);

    EXPECT_NE(p, 0);
    if (!p)
        return;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte px[3] = { 0, 0, 0 };
    glReadPixels(8, 8, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    mgl_test_checks++;
    if (px[1] < 200)
        MGL_FAIL("expected green");
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
}

/* ESSL 3.20 (#version 320 es): gl_BoundingBox is a core builtin there and
 * glslang lowers it to a plain patch out array, which MGL captures in the
 * TCS record like any per-patch array (no tess-factor semantics). */
TEST(TessES32Essl320Tess)
{
    static const char *vs =
        "#version 320 es\n"
        "out highp vec4 vs_col;\n"
        "void main()\n"
        "{\n"
        "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
        "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
        "    vs_col = vec4(0, 1, 0, 1);\n"
        "}\n";
    static const char *tcs =
        "#version 320 es\n"
        "layout(vertices=3) out;\n"
        "in highp vec4 vs_col[];\n"
        "out highp vec4 tcs_col[];\n"
        "void main()\n"
        "{\n"
        "    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "    tcs_col[gl_InvocationID] = vs_col[gl_InvocationID];\n"
        "    gl_TessLevelOuter[0] = 1.0;\n"
        "    gl_TessLevelOuter[1] = 1.0;\n"
        "    gl_TessLevelOuter[2] = 1.0;\n"
        "    gl_TessLevelInner[0] = 1.0;\n"
        "    gl_BoundingBox[0] = vec4(-1.0, -1.0, -1.0, 1.0);\n"
        "    gl_BoundingBox[1] = vec4( 1.0,  1.0,  1.0, 1.0);\n"
        "}\n";
    static const char *tes =
        "#version 320 es\n"
        "layout(triangles, equal_spacing) in;\n"
        "in highp vec4 tcs_col[];\n"
        "out highp vec4 tes_col;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = (gl_in[0].gl_Position * gl_TessCoord.x +\n"
        "                   gl_in[1].gl_Position * gl_TessCoord.y +\n"
        "                   gl_in[2].gl_Position * gl_TessCoord.z);\n"
        "    tes_col = tcs_col[0];\n"
        "}\n";
    static const char *fs =
        "#version 320 es\n"
        "in highp vec4 tes_col; out highp vec4 o;\n"
        "void main(){ o = tes_col; }\n";
    GLuint p = mgl_make_program5(vs, tcs, tes, NULL, fs);

    EXPECT_NE(p, 0);
    if (!p)
        return;

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    EXPECT_NO_GL_ERROR();
    GLubyte px[3] = { 0, 0, 0 };
    glReadPixels(8, 8, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    mgl_test_checks++;
    if (px[1] < 200)
        MGL_FAIL("expected green");
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
}

/* ESSL 3.20 audit: every shader stage accepts '#version 320 es' (geometry and
 * tessellation are ES 3.2 core; no EXT extension directives needed) */
TEST(TessES32Essl320Stages)
{
    static const char *gs =
        "#version 320 es\n"
        "layout(points) in; layout(points, max_vertices=1) out;\n"
        "void main(){ gl_Position = vec4(0); EmitVertex(); }\n";
    static const char *cs =
        "#version 320 es\n"
        "layout(local_size_x=1) in;\n"
        "layout(binding=0) buffer B { highp vec4 v[]; };\n"
        "void main(){ v[0] = vec4(1.0); }\n";
    GLuint s = glCreateShader(GL_GEOMETRY_SHADER);
    GLint ok = 0;

    glShaderSource(s, 1, &gs, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_NE(ok, 0);
    glDeleteShader(s);

    s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_NE(ok, 0);
    glDeleteShader(s);
}

/* ------------------------------------------------------------------ aliased wide lines
 * (r70, the last item of the r65 gap list).  Metal rasterizes only 1-px lines;
 * MGL expands the last pre-raster stage's output into screen-space quads on the
 * GPU.  A width-N line must cover EXACTLY N pixel rows/columns (the dEQP
 * LineRenderCase demand): the expansion snaps to pixel centres. */

static const char *kVsLine =
    "#version 460 core\n"
    "layout(location=0) in vec4 a_pos;\n"
    "layout(location=0) out vec4 v_col;\n"
    "void main() { gl_Position = a_pos; v_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";

static const char *kFsId =
    "#version 460 core\n"
    "layout(location=0) in vec4 f_col;\n"
    "layout(location=0) out vec4 o_col;\n"
    "void main() { o_col = f_col; }\n";

static void line_vao(const float *pos, int n)
{
    GLuint vao, b;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(n * 2 * sizeof(float)), pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
}

/* count the pixels lit in row y / column x of the green channel */
static int wl_row_lit(int y, int x0, int x1)
{
    float c[4];
    int n = 0;

    for (int x = x0; x <= x1; x++)
    {
        read_px(x, y, c);
        if (c[1] > 0.5f) n++;
    }
    return n;
}

static int wl_col_lit(int x, int y0, int y1)
{
    float c[4];
    int n = 0;

    for (int y = y0; y <= y1; y++)
    {
        read_px(x, y, c);
        if (c[1] > 0.5f) n++;
    }
    return n;
}

TEST(TessWideLineRange)
{
    /* the emulation advertises [1, 64] for aliased lines */
    GLfloat range[2] = { 0, 0 };
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, range);
    EXPECT_NEAR(range[0], 1.0, 0.001);
    EXPECT_TRUE(range[1] >= 5.0f);
    /* glLineWidth beyond the range clamps, width 1 stays the native path */
    glLineWidth(1.0f);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    return;
}

/* compile+link an arbitrary stage set (wideline tests need their own varyings) */
static GLuint wl_stages(const GLenum *types, const char **srcs, int n)
{
    GLuint p = glCreateProgram();

    for (int i = 0; i < n; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
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

static GLuint wl_program(const char *vs, const char *fs)
{
    const GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[2] = { vs, fs };

    return wl_stages(types, srcs, 2);
}

static GLuint wl_tess_program(const char *vs, const char *tcs, const char *tes, const char *fs)
{
    const GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[4] = { vs, tcs, tes, fs };

    return wl_stages(types, srcs, 4);
}

static GLuint wl_program_unused_(const char *vs, const char *fs)
{
    struct { GLenum t; const char *s; } st[2] = { { GL_VERTEX_SHADER, vs }, { GL_FRAGMENT_SHADER, fs } };
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++)
    {
        GLuint so = glCreateShader(st[i].t);
        glShaderSource(so, 1, &st[i].s, NULL);
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

TEST(TessWideLineVerticalHorizontal)
{
    glViewport(0, 0, 200, 200);
    GLuint prog = wl_program(kVsLine, kFsId);
    GLint link = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(prog);

    /* vertical line at NDC x=0 with width 5: snapped to the pixel-centre grid,
     * exactly 5 columns light up and nothing else */
    const float vpos[] = { 0.0f, -1.0f, 0.0f, 1.0f };
    line_vao(vpos, 2);
    glLineWidth(5.0f);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_row_lit(100, 0, 199), 5);
    EXPECT_EQ_INT(wl_col_lit(98, 10, 190), 181);
    EXPECT_EQ_INT(wl_col_lit(102, 10, 190), 181);
    EXPECT_EQ_INT(wl_col_lit(97, 10, 190), 0);
    EXPECT_EQ_INT(wl_col_lit(103, 10, 190), 0);

    /* width 1 immediately after: the native 1-px path, exactly one pixel */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_row_lit(100, 0, 199), 1);

    /* back to width 5: horizontal line at NDC y=0 */
    glLineWidth(5.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    const float hpos[] = { -1.0f, 0.0f, 1.0f, 0.0f };
    line_vao(hpos, 2);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 5);

    /* a strip: connected arms, each exactly 5 thick, corner included */
    glClear(GL_COLOR_BUFFER_BIT);
    const float spos[] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f };
    line_vao(spos, 3);
    glDrawArrays(GL_LINE_STRIP, 0, 3);
    glFinish();
    {
        /* thickness of the horizontal arm: scan a column inside its run */
        int lo = -1, hi = -1;
        float c[4];

        for (int y = 30; y <= 70; y++)
        {
            read_px(100, y, c);
            if (c[1] > 0.5f) { if (lo < 0) lo = y; hi = y; }
        }
        EXPECT_EQ_INT(hi - lo + 1, 5);
        EXPECT_EQ_INT(wl_row_lit((lo + hi) / 2, 60, 100), 41);  /* continuous */
    }
    /* thickness of the vertical arm */
    EXPECT_EQ_INT(wl_row_lit(100, 140, 160), 5);
    EXPECT_EQ_INT(wl_col_lit(150, 60, 100), 41);
    EXPECT_EQ_INT(wl_col_lit(147, 60, 100), 0);
    EXPECT_EQ_INT(wl_col_lit(153, 60, 100), 0);

    /* a clear after wide draws must wipe everything, and the native path must
     * still work (the replay pipeline must not leak into later draws) */
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    EXPECT_EQ_INT(wl_row_lit(100, 0, 199), 0);
    EXPECT_EQ_INT(wl_col_lit(150, 60, 100), 0);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    /* the strip's first segment as a 1-px native line: one of the two rows holds
     * the full ~100-pixel run */
    EXPECT_EQ_INT(wl_row_lit(49, 0, 199) + wl_row_lit(50, 0, 199), 100);

    glDeleteProgram(prog);
    return;
}

/* ------------------------------------------------------------------ isolines + wide lines
 * (r70): the emulated isolines path takes the TES-capture + expansion + replay
 * pipeline (mgl_ext_tess.h) when lineWidth > 1. */

TEST(TessWideLineIsolines)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *tcs =
        "#version 460 core\n"
        "layout(vertices=2) out;\n"
        "void main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; }\n"
        "}\n";
    const char *tes =
        "#version 460 core\n"
        "layout(isolines) in;\n"
        "layout(location=0) out vec4 v_col;\n"
        "void main(){\n"
        "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
        "  v_col = vec4(0.0, 1.0, 0.0, 1.0);\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) in vec4 f_col;\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = f_col; }\n";
    GLuint p = wl_tess_program(vs, tcs, tes, fs);
    GLint link = 0;

    if (!p)
        return;
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };   /* vec2 pairs (line_vao) */
    line_vao(pos, 2);
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* width 1 first: the plain emulated path, one pixel row (the row-flip's
     * half-pixel convention decides 99 vs 100 — scan both) */
    glLineWidth(1.0f);
    glDrawArrays(GL_PATCHES, 0, 2);
    glFinish();
    EXPECT_TRUE(wl_row_lit(99, 0, 199) + wl_row_lit(100, 0, 199) >= 99);

    /* width 5: the expansion path, exactly five rows */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(5.0f);
    glDrawArrays(GL_PATCHES, 0, 2);
    glFinish();
    if (getenv("MGL_WL_DEBUG"))
    {
        float img[200 * 200 * 4];
        int n = 0, minx = 999, maxx = -1, miny = 999, maxy = -1;
        glReadPixels(0, 0, 200, 200, GL_RGBA, GL_FLOAT, img);
        for (int y = 0; y < 200; y++)
            for (int x = 0; x < 200; x++)
                if (img[(y * 200 + x) * 4 + 1] > 0.5f)
                {
                    n++;
                    if (x < minx) minx = x;
                    if (x > maxx) maxx = x;
                    if (y < miny) miny = y;
                    if (y > maxy) maxy = y;
                }
        printf("iso w5 bbox x=%d..%d y=%d..%d n=%d\n", minx, maxx, miny, maxy, n);
        fflush(stdout);
    }
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 5);
    EXPECT_TRUE(wl_row_lit(100, 60, 140) >= 80);
    EXPECT_EQ_INT(wl_col_lit(47, 60, 140), 0);
    EXPECT_EQ_INT(wl_col_lit(153, 60, 140), 0);

    /* back to width 1 after the expansion path */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(1.0f);
    glDrawArrays(GL_PATCHES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 1);
    glDeleteProgram(p);
    return;
}

/* ------------------------------------------------------------------ geometry output + wide lines
 * (r70/r71): the geometry-kernel widening (extTessWidelineGeomReserve +
 * extTessWidelineGeomPrepare joining the kernel encoder). */

TEST(TessWideLineGeometryOutput)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(lines) in;\n"
        "layout(line_strip, max_vertices = 2) out;\n"
        "void main(){\n"
        "  gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[3] = { vs, gs, fs };
    GLuint p = wl_stages(types, srcs, 3);
    GLint link = 0;

    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };
    line_vao(pos, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    /* width 1: the geometry raster pass's native line */
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_TRUE(wl_row_lit(99, 0, 199) + wl_row_lit(100, 0, 199) >= 99);

    /* width 5: the expansion, exactly five rows */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(5.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 5);
    EXPECT_TRUE(wl_row_lit(100, 60, 140) >= 80);

    /* and back to 1 */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_TRUE(wl_row_lit(99, 0, 199) + wl_row_lit(100, 0, 199) >= 99);

    glDeleteProgram(p);
    return;
}

/* a multi-segment GS output: an L-shaped line_strip (three emitted vertices,
 * two segments sharing the corner).  Neighbouring segments must snap their
 * shared endpoint identically (the expansion kernel's pixel-centre snap) or the
 * widened strip shows gaps/overlaps at the corner — the shape the dEQP
 * LineRenderCase grids draw. */
TEST(TessWideLineGeometryPolyline)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(lines) in;\n"
        "layout(line_strip, max_vertices = 3) out;\n"
        "void main(){\n"
        "  gl_Position = vec4(-0.6, 0.0, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.2, 0.0, 0.0, 1.0); EmitVertex();\n"
        "  gl_Position = vec4(-0.2, 0.4, 0.0, 1.0); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[3] = { vs, gs, fs };
    GLuint p = wl_stages(types, srcs, 3);
    GLint link = 0;
    int rows5 = 0, cols5 = 0, r, c, pg_ref = -1;

    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };  /* one GL_LINES primitive */
    line_vao(pos, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    /* the native 1-px path first (the widened draw must not change it) */
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();
    EXPECT_TRUE(wl_row_lit(99, 42, 76) >= 33);

    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(5.0f);
    glDrawArrays(GL_LINES, 0, 2);
    glFinish();

    /* horizontal arm: clip y=0 -> readback row 100; arm spans x in [40, 80].
     * Exactly five rows carry the arm (measure away from the corner). */
    for (r = 97; r <= 103; r++)
        if (wl_row_lit(r, 42, 76) >= 33) rows5++;
    EXPECT_EQ_INT(rows5, 5);
    /* vertical arm: clip x=-0.2 -> column 80; the arm spans readback rows
     * 102..139 (MGL's row order: +y up).  Exactly five columns carry it. */
    for (c = 77; c <= 83; c++)
        if (wl_col_lit(c, 104, 136) >= 30) cols5++;
    EXPECT_EQ_INT(cols5, 5);
    /* the corner itself is covered (the arms join): (80, 100) */
    {
        float px[4];
        read_px(80, 100, px);
        EXPECT_TRUE(px[1] > 0.5f);
    }

    /* queries count the geometry stage's output, never the substituted quads:
     * one line_strip primitive emitted, regardless of line width */
    {
        GLuint q;
        GLint pg = -1, clip_in = -1, clip_out = -1;
        glGenQueries(1, &q);
        glClear(GL_COLOR_BUFFER_BIT);
        glLineWidth(1.0f);       /* the native path must agree exactly */
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_LINES, 0, 2);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &pg_ref);
        glClear(GL_COLOR_BUFFER_BIT);
        glLineWidth(5.0f);
        glBeginQuery(GL_PRIMITIVES_GENERATED, q);
        glDrawArrays(GL_LINES, 0, 2);
        glEndQuery(GL_PRIMITIVES_GENERATED);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &pg);
        EXPECT_EQ_INT(pg, pg_ref);   /* two segments: a 3-vertex strip counts its segments */
        EXPECT_EQ_INT(pg, 2);
        glBeginQuery(GL_CLIPPING_INPUT_PRIMITIVES, q);
        glDrawArrays(GL_LINES, 0, 2);
        glEndQuery(GL_CLIPPING_INPUT_PRIMITIVES);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &clip_in);
        EXPECT_EQ_INT(clip_in, 2);
        glBeginQuery(GL_CLIPPING_OUTPUT_PRIMITIVES, q);
        glDrawArrays(GL_LINES, 0, 2);
        glEndQuery(GL_CLIPPING_OUTPUT_PRIMITIVES);
        glGetQueryObjectiv(q, GL_QUERY_RESULT, &clip_out);
        EXPECT_EQ_INT(clip_out, 2);
        glDeleteQueries(1, &q);
    }

    /* a widened line is still a line primitive to GL: never culled.  Enable
     * front-face culling and redraw — the quads must render identically. */
    glClear(GL_COLOR_BUFFER_BIT);
    glCullFace(GL_FRONT);
    glEnable(GL_CULL_FACE);
    glDrawArrays(GL_LINES, 0, 2);
    glDisable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFinish();
    rows5 = cols5 = 0;
    for (r = 97; r <= 103; r++)
        if (wl_row_lit(r, 42, 76) >= 33) rows5++;
    for (c = 77; c <= 83; c++)
        if (wl_col_lit(c, 104, 136) >= 30) cols5++;
    EXPECT_EQ_INT(rows5, 5);
    EXPECT_EQ_INT(cols5, 5);

    glDeleteProgram(p);
    return;
}

/* transform feedback of a geometry stage's line output WHILE the output is
 * widened: xfb captures the emitted (pre-rasterizer) vertices — the pack joins
 * the kernel encoder after the expansion dispatch in the same encoder */
TEST(TessWideLineGeometryXfb)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *gs =
        "#version 460 core\n"
        "layout(lines) in;\n"
        "layout(line_strip, max_vertices = 2) out;\n"
        "out vec2 vpos;\n"
        "void main(){\n"
        "  gl_Position = gl_in[0].gl_Position; vpos = gl_Position.xy; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; vpos = gl_Position.xy; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[3] = { vs, gs, fs };
    const char *xfbv[] = { "vpos" };
    GLuint p = glCreateProgram(), sh[3], xb, q;
    GLint link = 0, wr = -1;
    float *m;

    for (int i = 0; i < 3; i++)
    {
        sh[i] = glCreateShader(types[i]);
        glShaderSource(sh[i], 1, &srcs[i], NULL);
        glCompileShader(sh[i]);
        glAttachShader(p, sh[i]);
        glDeleteShader(sh[i]);
    }
    glTransformFeedbackVaryings(p, 1, xfbv, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };
    line_vao(pos, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4096, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glGenQueries(1, &q);

    glLineWidth(5.0f);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_LINES);
    glDrawArrays(GL_LINES, 0, 2);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glFinish();

    /* the widened line still rasterizes exactly five rows */
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 5);
    EXPECT_TRUE(wl_row_lit(100, 60, 140) >= 80);

    /* and xfb captured the two EMITTED endpoints, not the corner records */
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &wr);
    EXPECT_EQ_INT(wr, 1);
    glBindBuffer(GL_ARRAY_BUFFER, xb);
    m = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    EXPECT_TRUE(m != NULL);
    if (m)
    {
        EXPECT_NEAR(m[0], -0.5f, 1e-4);
        EXPECT_NEAR(m[1], 0.0f, 1e-4);
        EXPECT_NEAR(m[2], 0.5f, 1e-4);
        EXPECT_NEAR(m[3], 0.0f, 1e-4);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    return;
}

/* INSTANCED wide isolines (r71): the emulated patch stream folds instances in
 * (total_patches = patches * instances), so instanced isolines widen through
 * the same capture + expansion + replay path — two instances offset in y by a
 * per-instance attribute, each arm exactly `width' rows (array + indexed) */
TEST(TessWideLineIsolinesInstanced)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec2 a_pos;\n"
        "layout(location=1) in float a_yoff;\n"
        "out float v_yoff;\n"
        "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); v_yoff = a_yoff; }\n";
    const char *tcs =
        "#version 460 core\n"
        "layout(vertices=2) out;\n"
        "in float v_yoff[];\n"
        "out float t_yoff[];\n"
        "void main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  t_yoff[gl_InvocationID] = v_yoff[gl_InvocationID];\n"
        "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; }\n"
        "}\n";
    const char *tes =
        "#version 460 core\n"
        "layout(isolines) in;\n"
        "in float t_yoff[];\n"
        "void main(){\n"
        "  vec4 p = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
        "  p.y += mix(t_yoff[0], t_yoff[1], gl_TessCoord.x);\n"
        "  gl_Position = p;\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[4] = { vs, tcs, tes, fs };
    GLuint p = wl_stages(types, srcs, 4);
    GLint link = 0;
    static const float pos[4] = { -0.5f, 0.0f, 0.5f, 0.0f };
    static const float yoff[2] = { 0.3f, -0.3f };
    GLuint vao, vbo, ibo;
    int arm_hi, arm_lo, r;

    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ARRAY_BUFFER, ibo);
    glBufferData(GL_ARRAY_BUFFER, sizeof yoff, yoff, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4, NULL);
    glVertexAttribDivisor(1, 1);
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);

    /* 1-px reference: one row per instance at rows ~69 and ~129 */
    glLineWidth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArraysInstanced(GL_PATCHES, 0, 2, 2);
    glFinish();
    EXPECT_TRUE(wl_row_lit(69, 10, 190) >= 90);
    EXPECT_TRUE(wl_row_lit(129, 10, 190) >= 90);

    /* width 5, array path: two arms of exactly 5 rows each */
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(5.0f);
    glDrawArraysInstanced(GL_PATCHES, 0, 2, 2);
    glFinish();
    arm_hi = arm_lo = 0;
    for (r = 122; r <= 138; r++)
        if (wl_row_lit(r, 10, 190) >= 90) arm_hi++;
    for (r = 62; r <= 78; r++)
        if (wl_row_lit(r, 10, 190) >= 90) arm_lo++;
    EXPECT_EQ_INT(arm_hi, 5);
    EXPECT_EQ_INT(arm_lo, 5);

    /* the same through an element buffer (the GPU index scan path) */
    {
        static const GLuint idx[2] = { 1, 0 };
        GLuint ebo;
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElementsInstanced(GL_PATCHES, 2, GL_UNSIGNED_INT, NULL, 2);
        glFinish();
        arm_hi = arm_lo = 0;
        for (r = 122; r <= 138; r++)
            if (wl_row_lit(r, 10, 190) >= 90) arm_hi++;
        for (r = 62; r <= 78; r++)
            if (wl_row_lit(r, 10, 190) >= 90) arm_lo++;
        EXPECT_EQ_INT(arm_hi, 5);
        EXPECT_EQ_INT(arm_lo, 5);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &ebo);
    }
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    return;
}

/* xfb capture alongside the isolines widening: both consume the TES capture
 * stream (the widening replays it, the pack copies it out) — the captured
 * vertices must be the TES's own positions, and the line still rasterizes
 * `width' rows wide (widening is a rasterization feature, never a stream one) */
TEST(TessWideLineIsolinesXfb)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *tcs =
        "#version 460 core\n"
        "layout(vertices=2) out;\n"
        "void main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 3.0; gl_TessLevelOuter[1] = 1.0; }\n"
        "}\n";
    const char *tes =
        "#version 460 core\n"
        "layout(isolines) in;\n"
        "out vec2 tpos;\n"
        "void main(){\n"
        "  vec4 p = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
        "  tpos = p.xy;\n"
        "  gl_Position = p;\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[4] = { vs, tcs, tes, fs };
    const char *xfbv[] = { "tpos" };
    GLuint p = glCreateProgram(), sh[4], xb, q;
    GLint link = 0, wr = -1;
    float *m;
    int rows = 0;

    for (int i = 0; i < 4; i++)
    {
        sh[i] = glCreateShader(types[i]);
        glShaderSource(sh[i], 1, &srcs[i], NULL);
        glCompileShader(sh[i]);
        glAttachShader(p, sh[i]);
        glDeleteShader(sh[i]);
    }
    glTransformFeedbackVaryings(p, 1, xfbv, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };
    line_vao(pos, 2);
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glGenQueries(1, &q);

    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(5.0f);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_LINES);
    glDrawArrays(GL_PATCHES, 0, 2);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glFinish();
    glGetQueryObjectiv(q, GL_QUERY_RESULT, &wr);
    glBindBuffer(GL_ARRAY_BUFFER, xb);
    m = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    EXPECT_TRUE(m != NULL);

    /* the widened line: exactly five rows at the centre */
    for (int r = 95; r <= 105; r++)
        if (wl_row_lit(r, 60, 140) >= 60) rows++;
    EXPECT_EQ_INT(rows, 5);

    /* widening must never alter the captured stream: re-capture at width 1
     * and compare float for float */
    if (m)
    {
        float *m2;
        GLint wr2 = -1;
        GLuint xb2, q2;
        glGenBuffers(1, &xb2);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb2);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb2);
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
        glGenQueries(1, &q2);
        glLineWidth(1.0f);
        glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q2);
        glBeginTransformFeedback(GL_LINES);
        glDrawArrays(GL_PATCHES, 0, 2);
        glEndTransformFeedback();
        glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
        glFinish();
        glGetQueryObjectiv(q2, GL_QUERY_RESULT, &wr2);
        EXPECT_EQ_INT(wr2, wr);
        glBindBuffer(GL_ARRAY_BUFFER, xb2);
        m2 = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (m2)
        {
            for (int i = 0; i < 2 * wr; i++)
                EXPECT_NEAR(m2[i], m[i], 1e-6);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glDeleteQueries(1, &q2);
        glDeleteBuffers(1, &xb2);
        glBindBuffer(GL_ARRAY_BUFFER, xb);
    }

    if (m)
    {
        /* GL isolines semantics (dEQP esextcTessellationShaderIsolines: "the
         * number of isolines is derived from the FIRST outer level, the
         * number of segments in each isoline from the SECOND"): outer={3,1}
         * is THREE lines at v = 0, 1/3, 2/3, each ONE full-span segment
         * u in [0,1].  The TES ignores v so the lines coincide; the captured
         * stream repeats the two control points once per line. */
        int bad = 0;
        EXPECT_EQ_INT(wr, 3);
        for (int i = 0; i < 2 * wr; i += 2)
            if (m[i + 1] != 0.0f) bad++;      /* every captured y = 0 */
        EXPECT_EQ_INT(bad, 0);
        for (int k = 0; k < wr; k++)
        {
            EXPECT_NEAR(m[2 * (2 * k + 0)], -0.5f, 1e-4);   /* line k, u=0 */
            EXPECT_NEAR(m[2 * (2 * k + 1)], 0.5f, 1e-4);    /* line k, u=1 */
        }
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    glDeleteQueries(1, &q);
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    return;
}

/* isolines feeding a line_strip geometry stage: both hooks in one pipeline
 * (TES capture -> GS records -> expansion) — the dEQP
 * primitive_bounding_box.wide_lines vertex_tessellation_geometry family */
TEST(TessWideLineIsolinesIntoGeometry)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "void main(){ gl_Position = a_pos; }\n";
    const char *tcs =
        "#version 460 core\n"
        "layout(vertices=2) out;\n"
        "void main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
        "  if (gl_InvocationID == 0) { gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; }\n"
        "}\n";
    const char *tes =
        "#version 460 core\n"
        "layout(isolines) in;\n"
        "void main(){\n"
        "  gl_Position = mix(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_TessCoord.x);\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(lines) in;\n"
        "layout(line_strip, max_vertices = 2) out;\n"
        "void main(){\n"
        "  gl_Position = gl_in[0].gl_Position; EmitVertex();\n"
        "  gl_Position = gl_in[1].gl_Position; EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "layout(location=0) out vec4 o_col;\n"
        "void main(){ o_col = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER,
                              GL_FRAGMENT_SHADER };
    const char *srcs[5] = { vs, tcs, tes, gs, fs };
    GLuint p = wl_stages(types, srcs, 5);
    GLint link = 0;

    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    const float pos[] = { -0.5f, 0.0f, 0.5f, 0.0f };
    line_vao(pos, 2);
    glPatchParameteri(GL_PATCH_VERTICES, 2);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glLineWidth(5.0f);
    glDrawArrays(GL_PATCHES, 0, 2);
    glFinish();
    EXPECT_EQ_INT(wl_col_lit(100, 0, 199), 5);
    EXPECT_TRUE(wl_row_lit(100, 60, 140) >= 80);

    glDeleteProgram(p);
    return;
}

/* ------------------------------------------------------------------ dEQP vtg guards (r72)
 * The primitive_bounding_box.vertex_tessellation_geometry_fragment family:
 * (1) its link pattern — VS+TCS+TES+FS links first, then VS+TCS+TES+GS+FS
 *     from the SAME shader objects (the glslang intermediate cache share the
 *     same_set handling in program.c guards; the Nov-4 CTS failures predate
 *     that Nov-5 fix), and
 * (2) the pipeline itself — the TCS permutes gl_Position (.wzxy), the TES
 *     un-permutes (.zwyx -- an identity pair) with `precise`, patch and flat
 *     varyings, and the GS 1->3 amplification.  The saved CTS evidence showed
 *     a black viewport; the replica renders the full expected grid. */

static const char *kVtgVs =
    "#version 460 core\n"
    "layout(location=0) in vec2 a_pos;\n"
    "layout(location=1) in vec4 a_col;\n"
    "out vec4 v_col;\n"
    "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); v_col = a_col; }\n";
static const char *kVtgTcs =
    "#version 460 core\n"
    "layout(vertices=3) out;\n"
    "in vec4 v_col[];\n"
    "out vec4 tc_col[];\n"
    "patch out vec3 p_dat;\n"
    "void main(){\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position.wzxy;\n"
    "  tc_col[gl_InvocationID] = v_col[gl_InvocationID];\n"
    "  p_dat = vec3(1.0);\n"
    "  gl_TessLevelOuter[0] = 2.8; gl_TessLevelOuter[1] = 2.8; gl_TessLevelOuter[2] = 2.8;\n"
    "  gl_TessLevelInner[0] = 2.8;\n"
    "}\n";
static const char *kVtgTes =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "in vec4 tc_col[];\n"
    "out vec4 te_col;\n"
    "patch in vec3 p_dat;\n"
    "flat out vec3 te_flat;\n"
    "precise gl_Position;\n"
    "void main(){\n"
    "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position.zwyx\n"
    "               + gl_TessCoord.y * gl_in[1].gl_Position.zwyx\n"
    "               + gl_TessCoord.z * gl_in[2].gl_Position.zwyx;\n"
    "  te_col = tc_col[0];\n"
    "  te_flat = p_dat;\n"
    "}\n";
static const char *kVtgGs =
    "#version 460 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices=9) out;\n"
    "in vec4 te_col[];\n"
    "flat in vec3 te_flat[];\n"
    "out vec4 g_col;\n"
    "void main(){\n"
    "  vec4 p0 = gl_in[0].gl_Position;\n"
    "  vec4 p1 = gl_in[1].gl_Position;\n"
    "  vec4 p2 = gl_in[2].gl_Position;\n"
    "  vec4 c = vec4((p0.xyz + p1.xyz + p2.xyz) / 3.0, 1.0);\n"
    "  for (int k = 0; k < 3; k++)\n"
    "  {\n"
    "    vec4 a = (k == 0) ? p0 : ((k == 1) ? p1 : p2);\n"
    "    vec4 b = (k == 0) ? p1 : ((k == 1) ? p2 : p0);\n"
    "    g_col = te_col[0] * te_flat[0].x;\n"
    "    gl_Position = a; EmitVertex();\n"
    "    g_col = te_col[0] * te_flat[0].x;\n"
    "    gl_Position = b; EmitVertex();\n"
    "    g_col = te_col[0] * te_flat[0].x;\n"
    "    gl_Position = c; EmitVertex();\n"
    "    EndPrimitive();\n"
    "  }\n"
    "}\n";

/* the dEQP link pattern: one compile per stage, the TES link first, the
 * tess+geometry link second from the same objects */
static int vtg_draw(int with_gs, GLuint shared[3])
{
    static const float pos[6] = { -0.7f, -0.7f, 0.8f, -0.5f, -0.1f, 0.8f };
    static const float col[3][4] = { {1,0,0,1}, {0,1,0,1}, {0,0,1,1} };
    static const char *kFsT = "#version 460 core\nin vec4 te_col;\nlayout(location=0) out vec4 o;\nvoid main(){ o = te_col; }\n";
    static const char *kFsG = "#version 460 core\nin vec4 g_col;\nlayout(location=0) out vec4 o;\nvoid main(){ o = g_col; }\n";
    const char *srcs[3] = { kVtgVs, kVtgTcs, kVtgTes };
    const GLenum types[3] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER };
    GLuint p, vao, b;
    GLint link = 0;
    int lit = 0;
    float c[4];

    for (int i = 0; i < 3; i++)
    {
        if (!shared[i])
        {
            shared[i] = glCreateShader(types[i]);
            glShaderSource(shared[i], 1, &srcs[i], NULL);
            glCompileShader(shared[i]);
        }
    }
    p = glCreateProgram();
    for (int i = 0; i < 3; i++)
        glAttachShader(p, shared[i]);
    if (with_gs)
    {
        GLuint g = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(g, 1, &kVtgGs, NULL);
        glCompileShader(g);
        glAttachShader(p, g);
        glDeleteShader(g);
    }
    {
        const char *fs = with_gs ? kFsG : kFsT;
        GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fs, NULL);
        glCompileShader(f);
        glAttachShader(p, f);
        glDeleteShader(f);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE) { glDeleteProgram(p); return -1; }
    glUseProgram(p);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    for (int y = 0; y < 200; y++) for (int x = 0; x < 200; x++)
    {
        glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
        if (c[0] + c[1] + c[2] > 0.2f) lit++;
    }
    glDeleteProgram(p);
    return lit;
}

TEST(TessVtgSharedShaders)
{
    GLuint shared[3] = {0, 0, 0};
    int vtf = vtg_draw(0, shared);
    int vtg = vtg_draw(1, shared);
    EXPECT_TRUE(vtf > 500);
    /* the three GS strips tile the input triangle: same coverage as the plain
     * TES render (the wzxy/zwyx pair is the identity permutation) */
    EXPECT_TRUE(vtg > vtf * 8 / 10);
    EXPECT_TRUE(vtg < vtf * 12 / 10);
    for (int i = 0; i < 3; i++) glDeleteShader(shared[i]);
    return;
}

/* the full dEQP pipeline verbatim-in-shape: VS pattern transform, TCS .wzxy
 * + bbox patch varyings, TES un-permute + precise + mirrorY(patternCenter),
 * GS mirrorX + 1->3 amplification, FS's fragmentInsideTheBBox(gl_FragCoord).
 * u_posScale chosen so the mirrored 16x16-cell grid lands exactly on the
 * [-0.8,0.8] square: 25600 lit pixels, nothing blue. */
TEST(TessVtgExactPipeline)
{
    const char *vs =
        "#version 460 core\n"
        "layout(location=0) in vec4 a_pos;\n"
        "layout(location=1) in vec4 a_col;\n"
        "out vec4 vtx_color;\n"
        "uniform vec4 u_posScale;\n"
        "void main(){\n"
        "  vec2 off = u_posScale.xy, sc = u_posScale.zw;\n"
        "  gl_Position = vec4(a_pos.xy * sc + off, a_pos.z, a_pos.w);\n"
        "  vtx_color = a_col;\n"
        "}\n";
    const char *tcs =
        "#version 460 core\n"
        "layout(vertices=3) out;\n"
        "in vec4 vtx_color[];\n"
        "out vec4 tess_ctrl_color[];\n"
        "uniform float u_tessellationLevel;\n"
        "uniform vec4 u_posScale;\n"
        "uniform vec4 u_primitiveBBoxMin;\n"
        "uniform vec4 u_primitiveBBoxMax;\n"
        "patch out float vp_bbox_expansionSize;\n"
        "patch out vec3 vp_bbox_clipMin;\n"
        "patch out vec3 vp_bbox_clipMax;\n"
        "void main(){\n"
        "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position.wzxy;\n"
        "  tess_ctrl_color[gl_InvocationID] = vtx_color[gl_InvocationID];\n"
        "  gl_TessLevelOuter[0] = u_tessellationLevel;\n"
        "  gl_TessLevelOuter[1] = u_tessellationLevel;\n"
        "  gl_TessLevelOuter[2] = u_tessellationLevel;\n"
        "  gl_TessLevelInner[0] = u_tessellationLevel;\n"
        "  vp_bbox_expansionSize = 0.0;\n"
        "  vp_bbox_clipMin = min(u_primitiveBBoxMin.xyz / u_primitiveBBoxMin.w, u_primitiveBBoxMin.xyz / u_primitiveBBoxMax.w);\n"
        "  vp_bbox_clipMax = max(u_primitiveBBoxMax.xyz / u_primitiveBBoxMin.w, u_primitiveBBoxMax.xyz / u_primitiveBBoxMax.w);\n"
        "}\n";
    const char *tes =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "in vec4 tess_ctrl_color[];\n"
        "out vec4 tess_color;\n"
        "uniform vec4 u_posScale;\n"
        "patch in float vp_bbox_expansionSize;\n"
        "patch in vec3 vp_bbox_clipMin;\n"
        "patch in vec3 vp_bbox_clipMax;\n"
        "flat out float v_geo_bbox_expansionSize;\n"
        "flat out vec3 v_geo_bbox_clipMin;\n"
        "flat out vec3 v_geo_bbox_clipMax;\n"
        "precise gl_Position;\n"
        "vec4 mirrorY(vec4 p){\n"
        "  vec2 off = u_posScale.xy, sc = u_posScale.zw;\n"
        "  vec2 ctr = off + sc * 0.5;\n"
        "  return vec4(p.x, 2.0 * ctr.y - p.y, p.z, p.w);\n"
        "}\n"
        "void main(){\n"
        "  gl_Position = mirrorY(gl_TessCoord.x * gl_in[0].gl_Position.zwyx +\n"
        "                        gl_TessCoord.y * gl_in[1].gl_Position.zwyx +\n"
        "                        gl_TessCoord.z * gl_in[2].gl_Position.zwyx);\n"
        "  tess_color = tess_ctrl_color[0];\n"
        "  v_geo_bbox_expansionSize = vp_bbox_expansionSize;\n"
        "  v_geo_bbox_clipMin = vp_bbox_clipMin;\n"
        "  v_geo_bbox_clipMax = vp_bbox_clipMax;\n"
        "}\n";
    const char *gs =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(triangle_strip, max_vertices=9) out;\n"
        "in vec4 tess_color[3];\n"
        "out vec4 geo_color;\n"
        "uniform vec4 u_posScale;\n"
        "flat in float v_geo_bbox_expansionSize[3];\n"
        "flat in vec3 v_geo_bbox_clipMin[3];\n"
        "flat in vec3 v_geo_bbox_clipMax[3];\n"
        "flat out vec3 v_bbox_clipMin;\n"
        "flat out vec3 v_bbox_clipMax;\n"
        "flat out float v_bbox_expansionSize;\n"
        "vec4 mirrorX(vec4 p){\n"
        "  vec2 off = u_posScale.xy, sc = u_posScale.zw;\n"
        "  vec2 ctr = off + sc * 0.5;\n"
        "  return vec4(2.0 * ctr.x - p.x, p.y, p.z, p.w);\n"
        "}\n"
        "void setVis(){ v_bbox_expansionSize = v_geo_bbox_expansionSize[0]; v_bbox_clipMin = v_geo_bbox_clipMin[0]; v_bbox_clipMax = v_geo_bbox_clipMax[0]; }\n"
        "void main(){\n"
        "  vec4 p0 = mirrorX(gl_in[0].gl_Position);\n"
        "  vec4 p1 = mirrorX(gl_in[1].gl_Position);\n"
        "  vec4 p2 = mirrorX(gl_in[2].gl_Position);\n"
        "  vec4 pc = vec4((p0.xyz + p1.xyz + p2.xyz) / 3.0, 1.0);\n"
        "  vec4 tc = tess_color[0];\n"
        "  gl_Position = p0; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = p1; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = pc; geo_color = tc; setVis(); EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_Position = p1; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = p2; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = pc; geo_color = tc; setVis(); EmitVertex();\n"
        "  EndPrimitive();\n"
        "  gl_Position = p2; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = p0; geo_color = tc; setVis(); EmitVertex();\n"
        "  gl_Position = pc; geo_color = tc; setVis(); EmitVertex();\n"
        "  EndPrimitive();\n"
        "}\n";
    const char *fs =
        "#version 460 core\n"
        "in vec4 geo_color;\n"
        "layout(location=0) out vec4 o_color;\n"
        "uniform ivec2 u_viewportPos;\n"
        "uniform ivec2 u_viewportSize;\n"
        "flat in float v_bbox_expansionSize;\n"
        "flat in vec3 v_bbox_clipMin;\n"
        "flat in vec3 v_bbox_clipMax;\n"
        "bool insideBBox(float depth){\n"
        "  vec4 wc = vec4(floor((v_bbox_clipMin.x * 0.5 + 0.5) * float(u_viewportSize.x) - v_bbox_expansionSize/2.0),\n"
        "                 floor((v_bbox_clipMin.y * 0.5 + 0.5) * float(u_viewportSize.y) - v_bbox_expansionSize/2.0),\n"
        "                 ceil((v_bbox_clipMax.x * 0.5 + 0.5) * float(u_viewportSize.x) + v_bbox_expansionSize/2.0),\n"
        "                 ceil((v_bbox_clipMax.y * 0.5 + 0.5) * float(u_viewportSize.y) + v_bbox_expansionSize/2.0));\n"
        "  if (gl_FragCoord.x < float(u_viewportPos.x) + wc.x || gl_FragCoord.x > float(u_viewportPos.x) + wc.z ||\n"
        "      gl_FragCoord.y < float(u_viewportPos.y) + wc.y || gl_FragCoord.y > float(u_viewportPos.y) + wc.w)\n"
        "    return false;\n"
        "  if (depth*2.0-1.0 < v_bbox_clipMin.z - 0.001 || depth*2.0-1.0 > v_bbox_clipMax.z + 0.001)\n"
        "    return false;\n"
        "  return true;\n"
        "}\n"
        "void main(){\n"
        "  float blue = insideBBox(gl_FragCoord.z) ? 0.0 : 1.0;\n"
        "  o_color = vec4(geo_color.r, geo_color.g, blue, geo_color.a);\n"
        "}\n";
    const GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                              GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[5] = { vs, tcs, tes, gs, fs };
    static float pos[512 * 24], col[512 * 24];
    GLuint p, vao, b;
    GLint link = 0, loc;
    int lit = 0, blue = 0, n = 0;
    float c[4];

    for (int gy = 0; gy < 16; gy++)
        for (int gx = 0; gx < 16; gx++)
        {
            float x0 = -0.8f + gx * 0.1f, y0 = -0.8f + gy * 0.1f, x1 = x0 + 0.1f, y1 = y0 + 0.1f;
            float cr = ((gx + gy) & 1) ? 0.1f : 0.9f, cg = ((gx + gy) & 1) ? 0.9f : 0.1f;
            float quad[6][2] = { {x0,y0}, {x1,y0}, {x0,y1}, {x1,y0}, {x1,y1}, {x0,y1} };
            for (int v = 0; v < 6; v++)
            {
                pos[n*4+0] = quad[v][0]; pos[n*4+1] = quad[v][1];
                pos[n*4+2] = 0.0f;       pos[n*4+3] = 1.0f;
                col[n*4+0] = cr; col[n*4+1] = cg; col[n*4+2] = 0.0f; col[n*4+3] = 1.0f;
                n++;
            }
        }
    p = wl_stages(types, srcs, 5);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    EXPECT_EQ_INT(link, GL_TRUE);
    if (!link) return;
    glUseProgram(p);
    /* patternCenter = off + sc/2; final = off + sc*(1 - x): with off=(-1,-1),
     * sc=(1,1) the [-0.8,0.8] square maps onto itself */
    loc = glGetUniformLocation(p, "u_posScale");
    if (loc >= 0) glUniform4f(loc, -1.0f, -1.0f, 1.0f, 1.0f);
    loc = glGetUniformLocation(p, "u_tessellationLevel");
    if (loc >= 0) glUniform1f(loc, 2.8f);
    loc = glGetUniformLocation(p, "u_primitiveBBoxMin");
    if (loc >= 0) glUniform4f(loc, -2.0f, -2.0f, -2.0f, 1.0f);
    loc = glGetUniformLocation(p, "u_primitiveBBoxMax");
    if (loc >= 0) glUniform4f(loc, 2.0f, 2.0f, 2.0f, 1.0f);
    loc = glGetUniformLocation(p, "u_viewportPos");
    if (loc >= 0) glUniform2i(loc, 0, 0);
    loc = glGetUniformLocation(p, "u_viewportSize");
    if (loc >= 0) glUniform2i(loc, 200, 200);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, NULL);
    glEnableVertexAttribArray(1);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, n);
    glFinish();
    for (int y = 0; y < 200; y++) for (int x = 0; x < 200; x++)
    {
        glReadPixels(x, y, 1, 1, GL_RGBA, GL_FLOAT, c);
        if (c[0] + c[1] + c[2] > 0.05f) lit++;
        if (c[2] > 0.5f) blue++;
    }
    /* the mirrored grid covers the [-0.8,0.8] square exactly; the generous
     * bbox marks nothing blue */
    EXPECT_TRUE(lit > 25000);
    EXPECT_EQ_INT(blue, 0);
    glDeleteProgram(p);
    return;
}

/* ------------------------------------------------------------------ TES xfb of matrices (r72)
 * The KHR enhanced_layouts.xfb_explicit_location_{,d}mat*_tessellation_eval
 * family: a matrix varying with an explicit xfb_offset captured from the TES.
 * GL packs it column-major, R components per column, contiguous.  (The
 * std140 uniform must be uploaded with padded columns — a plain {1,2,3,4}
 * upload puts cols 3,4 into col0's padding.) */
static const char *kXfbMatVs = "#version 460 core\nlayout(location=0) in vec2 a_pos;\nvoid main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }\n";
static const char *kXfbMatTcs = "#version 460 core\nlayout(vertices=3) out;\nvoid main(){\n"
    "  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;\n"
    "  gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "  gl_TessLevelInner[0] = 1.0;\n}\n";
static const char *kXfbMatFs = "#version 460 core\nlayout(location=0) out vec4 o;\nvoid main(){ o = vec4(0,1,0,1); }\n";

static int xfb_mat_case(const char *gltype, const void *ubuf, size_t ubytes,
                        const float *expect, int nexpect, const char *tag)
{
    char tes[1024];
    const GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    static const float pos[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f };
    GLuint p, vao, b, xb;
    GLint link = 0;
    int ok = 1;
    const char *srcs[4];

    snprintf(tes, sizeof tes,
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout(std140, binding = 0) uniform Goku { %s uni_goku; };\n"
        "layout (location = 0, xfb_offset = 0) flat out %s goku;\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  goku = uni_goku;\n"
        "}\n", gltype, gltype);
    srcs[0] = kXfbMatVs; srcs[1] = kXfbMatTcs; srcs[2] = tes; srcs[3] = kXfbMatFs;
    p = glCreateProgram();
    for (int i = 0; i < 4; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE) { glDeleteProgram(p); return 0; }
    glUseProgram(p);
    {
        GLuint idx = glGetUniformBlockIndex(p, "Goku"), ubo;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, ubytes, ubuf, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
        glUniformBlockBinding(p, idx, 0);
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 8192, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glFinish();
    glBindBuffer(GL_ARRAY_BUFFER, xb);
    {
        float *m = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (m)
        {
            for (int i = 0; i < nexpect; i++)
                if (memcmp(&m[i], &expect[i], 4) != 0) ok = 0;
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        else ok = 0;
    }
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    return ok;
}

TEST(TessXfbMatrices)
{
    /* std140 mat2: two vec4-padded columns; capture = 4 floats column-major */
    static const float ub2[8] = { 1,2,0,0,  3,4,0,0 };
    static const float ex2[4] = { 1,2,3,4 };
    /* std140 mat3: three vec4-padded columns; capture = 9 floats */
    static const float ub3[12] = { 1,2,0,0,  3,4,0,0,  5,6,0,0 };
    static const float ex3[9] = { 1,2,0, 3,4,0, 5,6,0 };
    /* std140 dmat2: two dvec2 columns; capture = 4 doubles bit-exact */
    static const double ubd[4] = { 1.0, 2.0, 3.0, 4.0 };
    float exd[8];
    memcpy(exd, ubd, sizeof ubd);

    EXPECT_TRUE(xfb_mat_case("mat2", ub2, sizeof ub2, ex2, 4, "mat2"));
    EXPECT_TRUE(xfb_mat_case("mat3", ub3, sizeof ub3, ex3, 9, "mat3"));
    EXPECT_TRUE(xfb_mat_case("dmat2", ubd, sizeof ubd, exd, 8, "dmat2"));
    return;
}

/* ------------------------------------------------------------------ TES xfb: block stride, struct, matrix array (r72)
 * The remaining KHR enhanced_layouts shapes: an xfb BLOCK with xfb_stride=128
 * (members pack per the block layout; the RECORD is padded to the stride),
 * a struct varying at xfb_offset=16 (a 16 B gap), and a matrix ARRAY
 * (column-major, contiguous per matrix).  */
static int xfb_shape_case(const char *tes, const float *expect, int nexpect, const char *tag)
{
    const GLenum types[4] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER };
    const char *srcs[4] = { kXfbMatVs, kXfbMatTcs, tes, kXfbMatFs };
    static const float pos[6] = { -0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f };
    GLuint p, vao, b, xb;
    GLint link = 0;
    int ok = 1;

    p = glCreateProgram();
    for (int i = 0; i < 4; i++)
    {
        GLuint so = glCreateShader(types[i]);
        glShaderSource(so, 1, &srcs[i], NULL);
        glCompileShader(so);
        glAttachShader(p, so);
        glDeleteShader(so);
    }
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &link);
    if (link != GL_TRUE)
    {
        char lg[2048] = {0}; GLsizei nn = 0;
        glGetProgramInfoLog(p, sizeof lg - 1, &nn, lg);
        printf("  %s LINK FAIL: %.200s\n", tag, lg);
        glDeleteProgram(p);
        return 0;
    }
    glUseProgram(p);
    {
        static const float uni[8] = { 1,2,3,4,  9,10,11,12 };   /* uni_gohan, uni_chichi */
        GLuint idx = glGetUniformBlockIndex(p, "block0"), ubo;
        if (idx != GL_INVALID_INDEX)
        {
            glGenBuffers(1, &ubo);
            glBindBuffer(GL_UNIFORM_BUFFER, ubo);
            glBufferData(GL_UNIFORM_BUFFER, sizeof uni, uni, GL_STATIC_DRAW);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
            glUniformBlockBinding(p, idx, 0);
        }
    }
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, NULL);
    glEnableVertexAttribArray(0);
    glPatchParameteri(GL_PATCH_VERTICES, 3);
    glViewport(0, 0, 200, 200);
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 16384, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 3);
    glEndTransformFeedback();
    glFinish();
    glBindBuffer(GL_ARRAY_BUFFER, xb);
    {
        float *m = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
        if (m)
        {
            for (int i = 0; i < nexpect; i++)
                if (m[i] != expect[i]) { ok = 0; break; }
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        else ok = 0;
    }
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    return ok;
}

TEST(TessXfbBlockStrideStructArray)
{
    /* A. xfb block with xfb_stride = 128: members at bytes 0/16/32, the record
     * padded to 128 (words 12..31 zero, vertex 1 at word 32) */
    static float exb[44];
    for (int i = 0; i < 44; i++) exb[i] = 0;
    exb[0] = 1; exb[1] = 2; exb[2] = 3; exb[3] = 4;
    exb[4] = 5; exb[5] = 6; exb[6] = 7; exb[7] = 8;
    exb[8] = 9; exb[9] = 10; exb[10] = 11; exb[11] = 12;
    for (int i = 0; i < 12; i++) exb[32 + i] = exb[i];
    const char *tesb =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout (xfb_offset = 0, xfb_stride = 128) out Goku {\n"
        "    vec4 gohan;\n"
        "    vec4 goten;\n"
        "    vec4 chichi;\n"
        "} goku;\n"
        "layout(binding = 0, std140) uniform block0 { vec4 uni_gohan; vec4 uni_chichi; };\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  goku.gohan = uni_gohan;\n"
        "  goku.goten = vec4(5.0, 6.0, 7.0, 8.0);\n"
        "  goku.chichi = uni_chichi;\n"
        "}\n";
    /* B. struct at xfb_offset = 16: words 0..3 zero, the members at 4..15 */
    static float exs[16];
    exs[4] = 1; exs[5] = 2; exs[6] = 3; exs[7] = 4;
    exs[8] = 5; exs[9] = 6; exs[10] = 7; exs[11] = 8;
    exs[12] = 9; exs[13] = 10; exs[14] = 11; exs[15] = 12;
    const char *tess =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "struct Goku { vec4 gohan; vec4 goten; vec4 chichi; };\n"
        "layout (xfb_offset = 16) out Goku goku;\n"
        "layout(binding = 0, std140) uniform block0 { vec4 uni_gohan; vec4 uni_chichi; };\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  goku.gohan = uni_gohan;\n"
        "  goku.goten = vec4(5.0, 6.0, 7.0, 8.0);\n"
        "  goku.chichi = uni_chichi;\n"
        "}\n";
    /* C. matrix array: two column-major mat2 = 16 floats */
    static const float exa[16] = { 1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4 };
    const char *tesa =
        "#version 460 core\n"
        "layout(triangles) in;\n"
        "layout (location = 0, xfb_offset = 0) flat out mat2 goku[2];\n"
        "void main(){\n"
        "  gl_Position = gl_TessCoord.x * gl_in[0].gl_Position + gl_TessCoord.y * gl_in[1].gl_Position + gl_TessCoord.z * gl_in[2].gl_Position;\n"
        "  goku[0] = mat2(1.0, 2.0, 3.0, 4.0);\n"
        "  goku[1] = mat2(1.0, 2.0, 3.0, 4.0);\n"
        "}\n";

    EXPECT_TRUE(xfb_shape_case(tesb, exb, 44, "block128"));
    EXPECT_TRUE(xfb_shape_case(tess, exs, 16, "struct16"));
    EXPECT_TRUE(xfb_shape_case(tesa, exa, 16, "mat2arr"));
    return;
}

int main(int argc, char **argv)
{
    if (tess_parity_child(argc, argv))
        return 0;
    RUN(TessWideLineRange);
    RUN(TessWideLineIsolines);
    RUN(TessWideLineIsolinesInstanced);
    RUN(TessWideLineIsolinesXfb);
    RUN(TessVtgSharedShaders);
    RUN(TessVtgExactPipeline);
    RUN(TessXfbMatrices);
    RUN(TessXfbBlockStrideStructArray);
    RUN(TessWideLineGeometryOutput);
    RUN(TessWideLineGeometryPolyline);
    RUN(TessWideLineGeometryXfb);
    RUN(TessWideLineIsolinesIntoGeometry);
    RUN(TessWideLineVerticalHorizontal);
    RUN(TessInstancedPerInstanceAttrib);
    RUN(TessInstancedNoTcsDefaultLevels);
    RUN(TessInstancedPerPatchLevels);
    RUN(TessInstancedIndexedTwoInstances);

    if (getenv("GRID")) {
        GLuint prog = make_program(kTcsTri, kTesTri);
        glUseProgram(prog);
        setup_patch_vao(kTriPos, kTriCol, 3);
        glPatchParameteri(GL_PATCH_VERTICES, 3);
        glClearColor(0, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_PATCHES, 0, 3);
        for (int y = 7; y >= 0; y--) {
            for (int x = 0; x < 8; x++) {
                float c[4];
                glReadPixels(x * 16 + 4, y * 16 + 4, 1, 1, GL_RGBA, GL_FLOAT, c);
                putchar(c[0] > 0.5f ? 'R' : c[1] > 0.5f ? 'G' : c[2] > 0.5f ? 'b' : '.');
            }
            putchar('\n');
        }
        return 0;
    }
    if (getenv("ONE")) {
        const char *one = getenv("ONE");
        if (!strcmp(one, "tri")) RUN(TessTriangleDomain);
        if (!strcmp(one, "psocache")) RUN(TessPSOCacheStress);
        if (!strcmp(one, "deflvl")) RUN(TessDefaultLevels);
        if (!strcmp(one, "pdcshader")) RUN(FragPostDepthShader);
        if (!strcmp(one, "pdcmask")) RUN(FragPostDepthCoverageMask);
        if (!strcmp(one, "lft")) RUN(FragLateTestsSideEffects);
        printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
        return mgl_test_failures != 0;
    }
    RUN(TessFourPatches);
    RUN(TessXfbPoints);
    RUN(TessGeometryMultiPatch);
    RUN(TessGeometryMatrixVarying);
    RUN(TessIndexedAndInstanced);
    RUN(TessCpuReferenceParity);
    RUN(TessIndexedRestart);
    RUN(TessIndexedNoRescan);
    RUN(TessRestartCheckCached);
    RUN(TessConsecutiveDraws);
    RUN(TessPSOCacheStress);
    RUN(TessTriangleDomain);
    RUN(TessQuadDomain);
    RUN(TessOrientationCulling);
    RUN(TessUnusedTesInput);
    RUN(TessUnusedTcsInput);
    RUN(TessLinkUnmatchedInput);
    RUN(TessLinkUnmatchedByLocation);
    RUN(TessLinkBlockInstanceNames);
    RUN(TessLinkCacheAfterFailure);
    RUN(TessReorderedInputs);
    RUN(TessPerPatchVaryings);
    RUN(TessArrayedVarying);
    RUN(TessPatchBlockArrayValues);
    RUN(TessProgramQueries);
    RUN(TessCountQueriesExact);
    RUN(TessCountQueriesWithXfbCapture);
    RUN(TessRingBigBlockRecycle);
    RUN(TessRingGenerationHammer);
    RUN(TessHugePatchCountDropped);
    RUN(TessErrors);
    RUN(TessCpuFallbackPatchBlobKey0);
    RUN(TessXfbCpuFallbackRasterizes);
    RUN(TessDefaultLevels);
    RUN(TessLayerFromVertexStage);
    RUN(TessLayerFromEvaluationStage);
    RUN(FragPostDepthShader);
    RUN(FragPostDepthCoverageMask);
    RUN(FragLateTestsSideEffects);
    RUN(DrawIdSurvivesLinkCache);
    RUN(ClipDistance130Link);

    RUN(TessES32EntryAndState);
    RUN(TessES32LineWidthQueries);
    RUN(TessES32DesktopBoundingBoxShader);
    RUN(TessES32Essl320Tess);
    RUN(TessES32Essl320Stages);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}

