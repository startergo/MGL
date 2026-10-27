/* test_vs_gather — vertex-stage gathers through the xfb capture path
 * (KHR-GL46.gpu_shader5.texture_gather_offset_depth_repeat / offsets_color_*):
 *   1. the capture pipeline must SPECIALIZE the shadow/wrap function constants
 *      (an unspecialized function aborts Metal validation — 5 CTS crashes);
 *   2. read()-based gather taps must apply the sampler's address mode
 *      themselves (REPEAT wraps OOB taps to the far edge).
 */
#include "mgl_test.h"
#include <string.h>

static const char *kVsInt =
    "#version 450 core\n"
    "precision highp isampler2D;\n"
    "uniform isampler2D smplr;\n"
    "layout(location = 0) in ivec2 offsets;\n"
    "layout(location = 1) in vec2 texCoords;\n"
    "flat out ivec4 without_offset_0;\n"
    "flat out ivec4 without_offset_1;\n"
    "flat out ivec4 with_offset_0;\n"
    "flat out ivec4 with_offset_1;\n"
    "void main() {\n"
    "    vec2 fractTexCoords = fract(texCoords);\n"
    "    without_offset_0 = textureGather(smplr, fractTexCoords, 0);\n"
    "    without_offset_1 = textureGather(smplr, fractTexCoords, 1);\n"
    "    with_offset_0 = textureGatherOffset(smplr, texCoords, offsets, 0);\n"
    "    with_offset_1 = textureGatherOffset(smplr, texCoords, offsets, 1);\n"
    "    gl_Position = vec4(texCoords * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static const char *kVsShadow =
    "#version 450 core\n"
    "uniform sampler2DShadow dtex;\n"
    "layout(location = 0) in vec2 in_uv;\n"
    "out vec4 g_shadow;\n"
    "out vec4 g_color;\n"
    "uniform sampler2D ctex;\n"
    "void main() {\n"
    "   gl_Position = vec4(in_uv, 0.0, 1.0);\n"
    "   g_shadow = textureGatherOffset(dtex, in_uv, 0.5625, ivec2(1, 2));\n"
    "   g_color = textureGatherOffset(ctex, in_uv, ivec2(1, 2));\n"
    "}\n";

TEST(int_gather_wrap_and_capture)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    char log[2048] = "";
    GLint ok = 0;
    glShaderSource(v, 1, &kVsInt, NULL);
    glCompileShader(v);
    glAttachShader(p, v);
    const char *vars[4] = { "without_offset_0", "without_offset_1", "with_offset_0", "with_offset_1" };
    glTransformFeedbackVaryings(p, 4, vars, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    glDeleteShader(v);
    glGetProgramInfoLog(p, sizeof log, NULL, log);
    if (log[0]) printf("  link: %s", log);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    const int SZ = 64;
    static int data[64 * 64 * 4];
    for (int y = 0; y < SZ; y++)
        for (int x = 0; x < SZ; x++)
            for (int c = 0; c < 4; c++) data[(y * SZ + x) * 4 + c] = x + c * 64;
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32I, SZ, SZ, 0, GL_RGBA_INTEGER, GL_INT, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const struct { int32_t ox, oy; float u, v; } vdata[1] = { { 4, 2, 0.0f, 0.0f } };
    glBufferData(GL_ARRAY_BUFFER, sizeof vdata, vdata, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribIPointer(0, 2, GL_INT, 16, (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void *)8);

    GLuint xb;
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4 * 16, NULL, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);

    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "smplr"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    int r[16];
    glBindBuffer(GL_COPY_READ_BUFFER, xb);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof r, r);
    /* with offset (4,2) from uv (0,0): base = (-1,-1)+(4,2) = (3,1);
     * taps (x0,y1)=(3,2) (x1,y1)=(4,2) (x1,y0)=(4,1) (x0,y0)=(3,1) */
    EXPECT_EQ_INT(r[8 + 0], 3);   /* with_offset_0.x */
    EXPECT_EQ_INT(r[8 + 1], 4);
    EXPECT_EQ_INT(r[8 + 2], 4);
    EXPECT_EQ_INT(r[8 + 3], 3);
    /* component 1 values are 64 higher */
    EXPECT_EQ_INT(r[12 + 0], 67);
    EXPECT_EQ_INT(r[12 + 1], 68);
    /* without offset at uv (0,0): base (-1,-1) wraps (REPEAT) to (63,63) /
     * (63,0) / (0,63) / (0,0) */
    EXPECT_EQ_INT(r[0], 63);
    EXPECT_EQ_INT(r[3], 63);
    EXPECT_EQ_INT(r[1], 0);
    EXPECT_EQ_INT(r[2], 0);

    glDeleteBuffers(1, &xb);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(1, &tex);
    glDeleteProgram(p);
}

TEST(shadow_and_color_gather_capture)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    char log[2048] = "";
    GLint ok = 0;
    glShaderSource(v, 1, &kVsShadow, NULL);
    glCompileShader(v);
    glAttachShader(p, v);
    const char *vars[2] = { "g_shadow", "g_color" };
    glTransformFeedbackVaryings(p, 2, vars, GL_SEPARATE_ATTRIBS);
    glLinkProgram(p);
    glDeleteShader(v);
    glGetProgramInfoLog(p, sizeof log, NULL, log);
    if (log[0]) printf("  link: %s", log);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) { glDeleteProgram(p); return; }

    const int SZ = 8;
    float depth[64];
    for (int i = 0; i < 64; i++) depth[i] = (i % SZ) / 8.0f;
    GLuint dtex;
    glGenTextures(1, &dtex);
    glBindTexture(GL_TEXTURE_2D, dtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, SZ, SZ, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    float col[64 * 4];
    for (int i = 0; i < 64; i++) { col[i * 4] = (i % SZ) / 8.0f; col[i * 4 + 1] = col[i * 4 + 2] = col[i * 4 + 3] = 1; }
    GLuint ctex;
    glGenTextures(1, &ctex);
    glBindTexture(GL_TEXTURE_2D, ctex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SZ, SZ, 0, GL_RGBA, GL_FLOAT, col);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float uv[1][2] = { { 0.5f, 0.5f } };
    glBufferData(GL_ARRAY_BUFFER, sizeof uv, uv, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    GLuint xb[2];
    glGenBuffers(2, xb);
    for (int q = 0; q < 2; q++) {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb[q]);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 4 * 16, NULL, GL_STREAM_READ);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, q, xb[q]);
    }

    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "dtex"), 0);
    glUniform1i(glGetUniformLocation(p, "ctex"), 1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, dtex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ctex);

    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    float ra[16] = { -1 }, rb[16] = { -1 };
    glBindBuffer(GL_COPY_READ_BUFFER, xb[0]);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof ra, ra);
    glBindBuffer(GL_COPY_READ_BUFFER, xb[1]);
    glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof rb, rb);
    /* x gradient with ref 0.5, LEQUAL: columns >= 4 pass */
    printf("  shadow gather: %.3f %.3f %.3f %.3f\n", ra[0], ra[1], ra[2], ra[3]);
    printf("  color gather:  %.3f %.3f %.3f %.3f\n", rb[0], rb[1], rb[2], rb[3]);
    int ones = 0;
    for (int i = 0; i < 4; i++) ones += ra[i] > 0.5f;
    EXPECT_TRUE(ones >= 1 && ones <= 3);
    /* color gather of the x gradient: some taps > 0 */
    int nonzero = 0;
    for (int i = 0; i < 4; i++) nonzero += rb[i] != 0.0f;
    EXPECT_TRUE(nonzero == 4);

    glDeleteBuffers(2, xb);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(2, &dtex);
    glDeleteProgram(p);
}

int main(void)
{
    RUN(int_gather_wrap_and_capture);
    RUN(shadow_and_color_gather_capture);
    return mgl_test_summary();
}
