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
    glClear(GL_COLOR_BUFFER_BIT);
    if (!getenv("MGL_NO_DRAW")) glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();

    /* the patch covers the lower-left triangle: expect barycentric-interpolated color.
       pixel (32,16) in a 128x128 window: NDC (-0.5, 0.75)... use a pixel inside the
       lower-left half: window row 96 of 128 (bottom area), col 32 */
    float c[4];
    read_px(16, 16, c);      /* near corner 0 (top-left): mostly red */
    EXPECT_NEAR(c[0], 0.72f, 0.18f);
    EXPECT_TRUE(c[1] < 0.45f);
    EXPECT_TRUE(c[2] < 0.45f);
    read_px(16, 112, c);     /* near corner 1 (bottom-left): mostly green */
    EXPECT_NEAR(c[1], 0.72f, 0.18f);
    EXPECT_TRUE(c[0] < 0.45f);
    read_px(112, 112, c);    /* near corner 2 (bottom-right): mostly blue */
    EXPECT_NEAR(c[2], 0.72f, 0.18f);
    read_px(64, 96, c);      /* interior: all three mixed */
    EXPECT_TRUE(c[0] > 0.05f);
    EXPECT_TRUE(c[1] > 0.05f);
    read_px(112, 16, c);     /* outside the triangle: clear color */
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
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_PATCHES, 0, 3);
    glFinish();
    float c[4];
    read_px(16, 16, c);
    EXPECT_NEAR(c[0], 0.72f, 0.18f);
    read_px(112, 112, c);
    EXPECT_NEAR(c[2], 0.72f, 0.18f);
    glDeleteProgram(prog);
    return;
}

int main(void)
{
    RUN(TessTriangleDomain);
    RUN(TessErrors);
    RUN(TessDefaultLevels);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
