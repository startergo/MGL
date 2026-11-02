/*
 * test_compat_xfblayout.c — shader-declared transform feedback layout
 * (ARB_enhanced_layouts xfb_buffer / xfb_offset / xfb_stride).
 *
 * KHR-Single-GL46.enhanced_layouts.xfb_stride_of_empty_list_and_api:
 *   layout (xfb_buffer = 0, xfb_stride = 64) out vec4 gs_fs1;   <- no offset: NOT captured
 *   layout (xfb_buffer = 1, xfb_offset = 0)  out vec4 gs_fs2;   <- captured into buffer 1
 * with glTransformFeedbackVaryings(gs_fs1, gs_fs2, INTERLEAVED) — GL 4.6 11.1.2.1:
 * when the shader uses xfb layout qualifiers, the API-specified varyings are IGNORED.
 */
#include "mgl_test.h"
#include <stdbool.h>

static const char *vs =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "void main() {}\n";
static const char *tcs =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(vertices = 1) out;\n"
    "void main() {\n"
    "    gl_TessLevelOuter[0] = 1.0;\n"
    "    gl_TessLevelOuter[1] = 1.0;\n"
    "    gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelOuter[3] = 1.0;\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "    gl_TessLevelInner[1] = 1.0;\n"
    "}\n";
static const char *tes =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(isolines, point_mode) in;\n"
    "void main() {}\n";
static const char *gs =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(points)                   in;\n"
    "layout(points, max_vertices = 3) out;\n"
    "layout (xfb_buffer = 0, xfb_stride = 64) out vec4 gs_fs1;\n"
    "layout (xfb_buffer = 1, xfb_offset = 0)  out vec4 gs_fs2;\n"
    "layout(binding    = 0) uniform gs_block {\n"
    "    vec4 uni_gs;\n"
    "};\n"
    "void main() {\n"
    "    gs_fs1 = -uni_gs;\n"
    "    gs_fs2 = uni_gs;\n"
    "    gl_Position  = vec4(0, 0, 0, 1);\n"
    "    EmitVertex();\n"
    "}\n";
static const char *fs =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "in  vec4 gs_fs2;\n"
    "out vec4 fs_out;\n"
    "void main() { fs_out = vec4(gs_fs2); }\n";

static const char *gs_api =
    "#version 430 core\n"
    "layout(points)                   in;\n"
    "layout(points, max_vertices = 3) out;\n"
    "layout(binding    = 0) uniform gs_block {\n"
    "    vec4 uni_gs;\n"
    "};\n"
    "out vec4 gs_fs1;\n"
    "out vec4 gs_fs2;\n"
    "void main() {\n"
    "    gs_fs1 = -uni_gs;\n"
    "    gs_fs2 = uni_gs;\n"
    "    gl_Position  = vec4(0, 0, 0, 1);\n"
    "    EmitVertex();\n"
    "}\n";

static const float uni[4] = {0.125f, -0.75f, 0.5f, 1.0f};

/* build the 5-stage pipeline with the given transform-feedback setup */
static GLuint build_xfb_program3(const char *gs_src, const char *const *names, GLsizei count, GLenum mode, bool with_tess)
{
    GLuint p = glCreateProgram();
    typedef struct { GLenum t; const char *s; } Stage;
    Stage all[] = {
        {GL_VERTEX_SHADER, vs}, {GL_TESS_CONTROL_SHADER, tcs},
        {GL_TESS_EVALUATION_SHADER, tes}, {GL_GEOMETRY_SHADER, gs_src}, {GL_FRAGMENT_SHADER, fs}};
    Stage stages[5];
    unsigned nstages = 0;
    for (unsigned i = 0; i < 5; i++) {
        if (!with_tess && (all[i].t == GL_TESS_CONTROL_SHADER || all[i].t == GL_TESS_EVALUATION_SHADER)) continue;
        stages[nstages++] = all[i];
    }
    for (unsigned i = 0; i < nstages; i++) {
        GLuint sh = glCreateShader(stages[i].t);
        glShaderSource(sh, 1, &stages[i].s, NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
            printf("  stage %u compile failed: %s", i, log); }
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    if (count) glTransformFeedbackVaryings(p, count, names, mode);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[4096]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("  link failed: %s", log); }
    while (glGetError() != GL_NO_ERROR) {}
    return p;
}

static GLuint build_xfb_program(const char *const *names, GLsizei count, GLenum mode)
{
    return build_xfb_program3(gs, names, count, mode, true);
}

static GLuint xfb0, xfb1;   /* transform feedback binding buffers */

static void setup_tf(void)
{
    GLbyte b0[64], b1[16];
    memset(b0, 0, sizeof b0);
    memset(b1, 0, sizeof b1);
    glGenBuffers(1, &xfb0);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb0);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof b0, b0, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfb0);
    glGenBuffers(1, &xfb1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb1);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof b1, b1, GL_DYNAMIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, xfb1);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(xfb_layout_overrides_api)
{
    const char *names[2] = {"gs_fs1", "gs_fs2"};
    GLuint p = build_xfb_program(names, 2, GL_INTERLEAVED_ATTRIBS);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    EXPECT_EQ_INT(linked, GL_TRUE);

    /* only gs_fs2 is captured: buffer 1, offset 0 */
    GLint n = -1;
    glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
    EXPECT_EQ_INT(n, 1);

    setup_tf();
    glUseProgram(p);
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, uni, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    while (glGetError() != GL_NO_ERROR) {}

    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 1);   /* one vertex per patch (CTS ctx) */
    glBeginTransformFeedback(GL_POINTS);
    EXPECT_NO_GL_ERROR();                      /* FIRST_MISSING of the CTS case: must succeed */
    glDrawArrays(GL_PATCHES, 0, 1);
    EXPECT_NO_GL_ERROR();
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();

    /* buffer 0 must be untouched (gs_fs1 has no xfb_offset) */
    float out0[16];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb0);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out0, out0);
    EXPECT_EQ_INT(out0[0] == 0.0f && out0[1] == 0.0f, 1);

    /* buffer 1 holds uni_gs once */
    float out1[4];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb1);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out1, out1);
    EXPECT_NEAR(out1[0], uni[0], 1e-6);
    EXPECT_NEAR(out1[1], uni[1], 1e-6);
    EXPECT_NEAR(out1[2], uni[2], 1e-6);
    EXPECT_NEAR(out1[3], uni[3], 1e-6);
    glDeleteProgram(p);
    glDeleteBuffers(1, &xfb0);
    glDeleteBuffers(1, &xfb1);
}

TEST(xfb_api_separate)
{
    fprintf(stderr, "### T2 start\n");
    const char *names[2] = {"gs_fs1", "gs_fs2"};
    GLuint p = build_xfb_program3(gs_api, names, 2, GL_SEPARATE_ATTRIBS, true);
    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    EXPECT_EQ_INT(linked, GL_TRUE);
    fprintf(stderr, "### T1 start\n");
    setup_tf();
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, uni, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    glUseProgram(p);
    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 1);   /* one vertex per patch (CTS ctx) */
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 1);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    GLuint q = 0; glGenQueries(1, &q);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, q);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_PATCHES, 0, 1);
    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    GLint prims = -1; glGetQueryObjectiv(q, GL_QUERY_RESULT, &prims);
    printf("  xfb primitives written = %d", prims);
    glDeleteQueries(1, &q);
    float out1[4];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb1);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out1, out1);
    printf("  api-separate buf1 = %g %g %g %g", out1[0], out1[1], out1[2], out1[3]);
    EXPECT_NEAR(out1[0], uni[0], 1e-6);
    EXPECT_NEAR(out1[1], uni[1], 1e-6);
    glDeleteProgram(p);
    glDeleteBuffers(1, &xfb0);
    glDeleteBuffers(1, &xfb1);
}

TEST(xfb_gs_points_no_tess)
{
    fprintf(stderr, "### T3 start\n");
    const char *names[2] = {"gs_fs1", "gs_fs2"};
    GLuint p = build_xfb_program3(gs_api, names, 2, GL_SEPARATE_ATTRIBS, false);
    GLint linked = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &linked);
    EXPECT_EQ_INT(linked, GL_TRUE);
    setup_tf();
    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, uni, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    glUseProgram(p);
    GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glPatchParameteri(GL_PATCH_VERTICES, 1);   /* one vertex per patch (CTS ctx) */
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    EXPECT_NO_GL_ERROR();
    float out1[4];
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xfb1);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out1, out1);
    printf("  gs-points buf1 = %g %g %g %g", out1[0], out1[1], out1[2], out1[3]);
    EXPECT_NEAR(out1[0], uni[0], 1e-6);
    glDeleteProgram(p);
    glDeleteBuffers(1, &xfb0);
    glDeleteBuffers(1, &xfb1);
}

int main(void)
{
    RUN(xfb_layout_overrides_api);
    RUN(xfb_api_separate);
    RUN(xfb_gs_points_no_tess);
    return mgl_test_summary();
}
