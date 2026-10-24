/* test_vab_slots — KHR-GL46.vertex_attrib_binding regression (round 44).
 *
 * Root cause of the CTS group's shared-batch failures: Metal vertex buffer slot 30 was
 * claimed by BOTH MGL_VA_CONST_BUFFER_INDEX (disabled-attribute current values, bound by
 * extVattribPreDraw) and MGL_XFB_MSL_BUFFER_INDEX (the xfb capture scratch, bound on the
 * SAME render encoder after it).  During a capture draw the scratch binding wins, so a
 * disabled attribute read the scratch bytes — stale vertex data left by an earlier
 * capture in the same process — instead of its current generic value.  The CTS cases
 * passed standalone (fresh zero scratch) and failed in a shared batch.
 *
 * This runs two captures in one process: the first leaves real vertex data in the
 * scratch, the second has a DISABLED attribute whose current value (7,7,7,7) differs
 * from both zero and the stale data (1,2,3,1), so neither luck nor poison passes.
 */
#include "mgl_test.h"

static const char *vs =
    "#version 430 core\n"
    "layout(location = 0) in vec4 vs_in_attrib[16];\n"
    "out StageData { vec4 attrib[16]; } vs_out;\n"
    "void main() {\n"
    "  for (int i = 0; i < vs_in_attrib.length(); ++i)\n"
    "    vs_out.attrib[i] = vs_in_attrib[i];\n"
    "}\n";

static char vnames[16][32];

/* capture `vcount` vertices with the current VAO, compare the 16 captured slots of
 * vertex `vi` against `want` (16 vec4, NULL = all current values from `cur`) */
/* local label variant: mgl_test.h has no EXPECT_NEAR with a label */
#define EXPECT_NEAR_L(lbl, a, b, eps) do { mgl_test_checks++; double _a = (a), _b = (b); \
    if (!((_a > _b - (eps)) && (_a < _b + (eps)))) MGL_FAIL("%s: %g != %g", lbl, _a, _b); } while (0)

static void capture(GLuint po, GLenum mode, GLint vcount, const float *cur)
{
    float zero[64 * 4] = {0};
    GLuint xfbo;
    glGenBuffers(1, &xfbo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfbo);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof zero, zero, GL_DYNAMIC_DRAW);

    glUseProgram(po);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(mode);
    glDrawArrays(mode, 0, vcount);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    EXPECT_NO_GL_ERROR();

    static float data[64 * 4];
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof data, data);
    /* vertex 0's attrib[i] is xfb slot i; disabled slots must read their current
     * generic value even while the xfb capture scratch rides the same encoder */
    for (int i = 0; i < 16; i++)
    {
        char label[64];
        snprintf(label, sizeof label, "attrib[%d]", i);
        EXPECT_NEAR_L(label, data[i * 4 + 0], cur[i * 4 + 0], 1e-5);
        EXPECT_NEAR_L(label, data[i * 4 + 1], cur[i * 4 + 1], 1e-5);
        EXPECT_NEAR_L(label, data[i * 4 + 2], cur[i * 4 + 2], 1e-5);
        EXPECT_NEAR_L(label, data[i * 4 + 3], cur[i * 4 + 3], 1e-5);
    }
    glDeleteBuffers(1, &xfbo);
}

/* case1 of the CTS group: one enabled array binding, everything else current values.
 * Leaves real vertex data (1,2,3,1) in the capture scratch buffer. */
TEST(case1_poison)
{
    GLuint po = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    glAttachShader(po, sh);
    const char *v[16];
    for (int i = 0; i < 16; i++) { snprintf(vnames[i], 32, "StageData.attrib[%d]", i); v[i] = vnames[i]; }
    glTransformFeedbackVaryings(po, 16, v, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(po);
    GLint ok = 0;
    glGetProgramiv(po, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) return;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    /* distinctive currents: the stale-scratch poison value (1,2,3,1) must never match */
    float cur[16 * 4];
    for (int i = 0; i < 16; i++)
    {
        glVertexAttrib4f(i, -8 - i, 0.5, 0.25, 1);
        cur[i * 4 + 0] = -8 - i; cur[i * 4 + 1] = 0.5f; cur[i * 4 + 2] = 0.25f; cur[i * 4 + 3] = 1;
    }

    const float d0[3] = {1, 2, 3}, d1[3] = {4, 5, 6};
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 24, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 12, d0);
    glBufferSubData(GL_ARRAY_BUFFER, 12, 12, d1);

    glBindVertexArray(vao);
    glBindVertexBuffer(0, vbo, 0, 12);
    glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(1, 0);
    glEnableVertexAttribArray(1);

    cur[1 * 4 + 0] = 1; cur[1 * 4 + 1] = 2; cur[1 * 4 + 2] = 3; cur[1 * 4 + 3] = 1;
    capture(po, GL_POINTS, 2, cur);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(po);
}

/* case2: four enabled attributes on one binding, slot 2 disabled with current (7,7,7,7).
 * With the slot-30 collision the disabled slot read the scratch (stale (1,2,3,1) from
 * case1, or (0,0,0,0) from a fresh buffer) — both wrong now. */
TEST(case2_disabled_current)
{
    GLuint po = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &vs, NULL);
    glCompileShader(sh);
    glAttachShader(po, sh);
    const char *v[16];
    for (int i = 0; i < 16; i++) { snprintf(vnames[i], 32, "StageData.attrib[%d]", i); v[i] = vnames[i]; }
    glTransformFeedbackVaryings(po, 16, v, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(po);
    GLint ok = 0;
    glGetProgramiv(po, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    if (!ok) return;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    float cur[16 * 4];
    for (int i = 0; i < 16; i++)
    {
        glVertexAttrib4f(i, 0, 0, 0, 0);
        cur[i * 4 + 0] = cur[i * 4 + 1] = cur[i * 4 + 2] = 0; cur[i * 4 + 3] = 0;
    }
    /* the poisoned slot must have a current value no scratch content can fake */
    glVertexAttrib4f(2, 7, 7, 7, 7);
    cur[2 * 4 + 0] = cur[2 * 4 + 1] = cur[2 * 4 + 2] = 7; cur[2 * 4 + 3] = 7;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    { const float a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
      glBufferData(GL_ARRAY_BUFFER, 24, NULL, GL_STATIC_DRAW);
      glBufferSubData(GL_ARRAY_BUFFER, 0, 12, a);
      glBufferSubData(GL_ARRAY_BUFFER, 12, 12, b); }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(vao);
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribFormat(7, 1, GL_FLOAT, GL_FALSE, 8);
    glVertexAttribFormat(15, 2, GL_FLOAT, GL_FALSE, 4);
    glVertexAttribBinding(0, 0);
    glVertexAttribBinding(1, 0);
    glVertexAttribBinding(7, 0);
    glVertexAttribBinding(15, 0);
    glBindVertexBuffer(0, vbo, 0, 12);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(7);
    glEnableVertexAttribArray(15);

    /* vertex 0 expectations: overlapping formats on one 12-byte stride binding */
    cur[0 * 4 + 0] = 1; cur[0 * 4 + 1] = 2; cur[0 * 4 + 3] = 1;
    cur[1 * 4 + 0] = 1; cur[1 * 4 + 1] = 2; cur[1 * 4 + 2] = 3; cur[1 * 4 + 3] = 1;
    cur[7 * 4 + 0] = 3; cur[7 * 4 + 3] = 1;
    cur[15 * 4 + 0] = 2; cur[15 * 4 + 1] = 3; cur[15 * 4 + 3] = 1;
    capture(po, GL_POINTS, 1, cur);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(po);
}

/* ------------------------------------------------------------------ round 45 --
 * KHR-GL46.vertex_attrib_binding basic-inputL / advanced-bindingUpdate.
 *
 * (a) whole dvec4 ARRAY elements (`vs_in_attrib[i]`) in the fp64 lowering: the fp64 pass
 *     flattens `dvec4 a[8]` into 16 uint4 Metal slots (two per element), and a size 3/2/1
 *     glVertexAttribLFormat feeding one of them must read the components the array does
 *     not supply from the CURRENT generic attribute value (GL 4.6 §10.5) — through ONE
 *     converted shadow stream for the whole attribute, never a raw-word slot next to a
 *     converted one.  The double at binding 6 (stride 0) reads the same element for every
 *     vertex.
 * (b) a program mixing dvec3@3 + ivec2@4: the interface-location pass gives the dvec3 the
 *     internal slots 0-1 and moves the vec4 position to MSL slot 5, so the draw plan must
 *     map every internal slot back to its OWNING GL attribute (slot 5 -> attrib 0, not the
 *     disabled attrib 5) — this used to feed the position the constant (0,0,0,0) and draw
 *     nothing.
 */

TEST(case3_dvec_element_pad)
{
    static const char *dvs =
        "#version 430 core\n"
        "layout(location = 0) in dvec4 vs_in_attrib[8];\n"
        "out StageData { dvec4 attrib[8]; } vs_out;\n"
        "void main() {\n"
        "  vs_out.attrib[0] = vs_in_attrib[0];\n"
        "  vs_out.attrib[1] = vs_in_attrib[1];\n"
        "  vs_out.attrib[2] = vs_in_attrib[2];\n"
        "  vs_out.attrib[3] = vs_in_attrib[3];\n"
        "  vs_out.attrib[4] = vs_in_attrib[4];\n"
        "  vs_out.attrib[5] = vs_in_attrib[5];\n"
        "  vs_out.attrib[6] = vs_in_attrib[6];\n"
        "  vs_out.attrib[7] = vs_in_attrib[7];\n"
        "}\n";   /* unrolled like the CTS: whole dvec4 array elements, const indices */

    GLuint po = glCreateProgram();
    GLuint sh = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sh, 1, &dvs, NULL);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, 1);
    glAttachShader(po, sh);
    {
        const GLchar *const v[8] = {
            "StageData.attrib[0]", "StageData.attrib[1]", "StageData.attrib[2]", "StageData.attrib[3]",
            "StageData.attrib[4]", "StageData.attrib[5]", "StageData.attrib[6]", "StageData.attrib[7]",
        };
        glTransformFeedbackVaryings(po, 8, v, GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(po);
    GLint st = 0; glGetProgramiv(po, GL_LINK_STATUS, &st);
    EXPECT_EQ_INT(st, 1);
    if (!st) { char log[4096]; GLsizei ll = 0; log[0] = 0;
        glGetProgramInfoLog(po, sizeof log, &ll, log); fprintf(stderr, "  [case3] link: %s\n", log); }
    if (st) {
        /* three dvec4 elements; the CTS layout: attrib0 = LFormat(4)@binding0(stride 32),
         * attrib3 = LFormat(3, reloff 8)@binding2(stride 16), attrib5 = LFormat(2)@binding0,
         * attrib7 = LFormat(1, reloff 24)@binding6(offset 16, stride 0) */
        double elem[3][4] = { {1,2,3,4}, {5,6,7,8}, {9,10,11,12} };
        GLuint vbo; glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof elem, elem, GL_STATIC_DRAW);

        GLuint vao; glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        for (int i = 0; i < 8; i++) glVertexAttribL4d(i, 0, 0, 0, 0);
        glVertexAttribLFormat(0, 4, GL_DOUBLE, 0);
        glVertexAttribLFormat(3, 3, GL_DOUBLE, 8);
        glVertexAttribLFormat(5, 2, GL_DOUBLE, 0);
        glVertexAttribLFormat(7, 1, GL_DOUBLE, 24);
        glVertexAttribBinding(0, 0);
        glVertexAttribBinding(3, 2);
        glVertexAttribBinding(5, 0);
        glVertexAttribBinding(7, 6);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(3);
        glEnableVertexAttribArray(5);
        glEnableVertexAttribArray(7);
        glBindVertexBuffer(0, vbo, 0, 32);
        glBindVertexBuffer(2, vbo, 0, 16);
        glBindVertexBuffer(6, vbo, 16, 0);

        double out[32 * 4];
        memset(out, 0, sizeof out);
        GLuint xfbo; glGenBuffers(1, &xfbo);
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xfbo);
        glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, sizeof out, NULL, GL_STREAM_READ);

        glUseProgram(po);
        glEnable(GL_RASTERIZER_DISCARD);
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, 2);
        glEndTransformFeedback();
        glDisable(GL_RASTERIZER_DISCARD);
        EXPECT_NO_GL_ERROR();
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);

        /* vertex 0: attrib0 = elem0, attrib3 = doubles at 8/16/24, attrib5 = elem0.xy,
         *           attrib7 = the double at 16+24 = 6 (stride 0: same for vertex 1) */
        EXPECT_NEAR_L("v0 attrib0.x", out[0], 1, 1e-9);
        EXPECT_NEAR_L("v0 attrib0.w", out[3], 4, 1e-9);
        EXPECT_NEAR_L("v0 attrib3.x", out[12], 2, 1e-9);
        EXPECT_NEAR_L("v0 attrib3.z", out[14], 4, 1e-9);
        EXPECT_NEAR_L("v0 attrib3.w", out[15], 0, 1e-9);   /* padded from the current value */
        EXPECT_NEAR_L("v0 attrib5.x", out[20], 1, 1e-9);
        EXPECT_NEAR_L("v0 attrib5.y", out[21], 2, 1e-9);
        EXPECT_NEAR_L("v0 attrib5.z", out[22], 0, 1e-9);   /* padded */
        EXPECT_NEAR_L("v0 attrib7.x", out[28], 6, 1e-9);
        /* vertex 1: attrib0 = elem1, attrib3 = doubles at 24/32/40, attrib5 = elem1.xy */
        EXPECT_NEAR_L("v1 attrib0.x", out[32], 5, 1e-9);
        EXPECT_NEAR_L("v1 attrib0.w", out[35], 8, 1e-9);
        EXPECT_NEAR_L("v1 attrib3.x", out[44], 4, 1e-9);
        EXPECT_NEAR_L("v1 attrib3.z", out[46], 6, 1e-9);
        EXPECT_NEAR_L("v1 attrib5.x", out[52], 5, 1e-9);
        EXPECT_NEAR_L("v1 attrib5.y", out[53], 6, 1e-9);
        EXPECT_NEAR_L("v1 attrib7.x", out[60], 6, 1e-9);

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &xfbo);
    }
    glDeleteShader(sh);
    glDeleteProgram(po);
}

TEST(case4_renumbered_slots)
{
    static const char *vs4 =
        "#version 430 core\n"
        "layout(location = 0) in vec4 vs_in_position;\n"
        "layout(location = 1) in vec2 vs_in_color_rg;\n"
        "layout(location = 2) in float vs_in_color_b;\n"
        "layout(location = 3) in dvec3 vs_in_data0;\n"
        "layout(location = 4) in ivec2 vs_in_data1;\n"
        "out StageData { vec2 color_rg; float color_b; dvec3 data0; ivec2 data1; } vs_out;\n"
        "void main() {\n"
        "  vs_out.data1 = vs_in_data1;\n"
        "  vs_out.data0 = vs_in_data0;\n"
        "  vs_out.color_b = vs_in_color_b;\n"
        "  vs_out.color_rg = vs_in_color_rg;\n"
        "  gl_Position = vs_in_position; }\n";
    static const char *fs4 =
        "#version 430 core\n"
        "in StageData { vec2 color_rg; float color_b; flat dvec3 data0; flat ivec2 data1; } fs_in;\n"
        "layout(location = 0) out vec4 fs_out_color;\n"
        "uniform dvec3 g_expected_data0;\n"
        "uniform ivec2 g_expected_data1;\n"
        "void main() {\n"
        "  fs_out_color = vec4(fs_in.color_rg, fs_in.color_b, 1);\n"
        "  if (fs_in.data0 != g_expected_data0) fs_out_color = vec4(1);\n"
        "  if (fs_in.data1 != g_expected_data1) fs_out_color = vec4(1); }\n";

    GLuint po = mgl_make_program(vs4, fs4);
    EXPECT_TRUE(po != 0);
    if (po) {
        /* one 52-byte stride binding: pos vec2@0, rg vec2@8, b float@16, dvec3@20, ivec2@44 */
        unsigned char buf[4 * 52];
        static const float pos[4][2] = { {-1,-1}, {1,-1}, {1,1}, {-1,1} };
        memset(buf, 0, sizeof buf);
        for (int i = 0; i < 4; i++) {
            memcpy(buf + i * 52 + 0, pos[i], 8);
            float rg[2] = { 0.0f, 1.0f }; memcpy(buf + i * 52 + 8, rg, 8);
            float b = 0.0f; memcpy(buf + i * 52 + 16, &b, 4);
            double d0[3] = { 1.0, 2.0, 3.0 }; memcpy(buf + i * 52 + 20, d0, 24);
            int i2[2] = { 1, 2 }; memcpy(buf + i * 52 + 44, i2, 8);
        }
        GLuint vbo; glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof buf, buf, GL_STATIC_DRAW);
        GLushort idx[4] = { 0, 1, 3, 2 };
        GLuint ebo; glGenBuffers(1, &ebo);

        GLuint vao; glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glBindVertexBuffer(0, vbo, 0, 52);
        glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8);
        glVertexAttribFormat(2, 1, GL_FLOAT, GL_FALSE, 16);
        glVertexAttribLFormat(3, 3, GL_DOUBLE, 20);
        glVertexAttribIFormat(4, 2, GL_INT, 44);
        for (int i = 0; i < 5; i++) { glVertexAttribBinding(i, 0); glEnableVertexAttribArray(i); }

        glUseProgram(po);
        glUniform3d(glGetUniformLocation(po, "g_expected_data0"), 1.0, 2.0, 3.0);
        glUniform2i(glGetUniformLocation(po, "g_expected_data1"), 1, 2);
        glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, NULL, 1, 0, 0);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(2, 2, 0, 255, 0, 255, 2);

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteProgram(po);
    }
}


/* ------------------------------------------------------ round 45, case 5 -----
 * The two CPU-converted attribute paths (double word slots and plain float/int
 * streams) allocate their Metal vertex buffer indices from one counter; the double
 * path pre-decremented while the float path post-decremented, so the first converted
 * stream of each kind shared an index and one read the other's bytes.  This is the
 * KHR-GL46.vertex_attrib_binding advanced-bindingUpdate draw-2 layout: an UNALIGNED
 * stride (62) forces every attribute through a converted stream, including the dvec3. */

TEST(case5_converted_double_and_float)
{
    static const char *vs5 =
        "#version 430 core\n"
        "layout(location = 0) in vec4 vs_in_position;\n"
        "layout(location = 1) in vec2 vs_in_color_rg;\n"
        "layout(location = 2) in float vs_in_color_b;\n"
        "layout(location = 3) in dvec3 vs_in_data0;\n"
        "layout(location = 4) in ivec2 vs_in_data1;\n"
        "out StageData { vec2 color_rg; float color_b; dvec3 data0; ivec2 data1; } vs_out;\n"
        "void main() {\n"
        "  vs_out.data1 = vs_in_data1;\n"
        "  vs_out.data0 = vs_in_data0;\n"
        "  vs_out.color_b = vs_in_color_b;\n"
        "  vs_out.color_rg = vs_in_color_rg;\n"
        "  gl_Position = vs_in_position; }\n";
    static const char *fs5 =
        "#version 430 core\n"
        "in StageData { vec2 color_rg; float color_b; flat dvec3 data0; flat ivec2 data1; } fs_in;\n"
        "layout(location = 0) out vec4 fs_out_color;\n"
        "uniform dvec3 g_expected_data0;\n"
        "uniform ivec2 g_expected_data1;\n"
        "void main() {\n"
        "  dvec3 d0 = fs_in.data0;\n"
        "  ivec2 d1 = fs_in.data1;\n"
        "  fs_out_color = vec4(fs_in.color_rg, fs_in.color_b, 1);\n"
        "  if (d0 != g_expected_data0) fs_out_color = vec4(1);\n"
        "  if (d1 != g_expected_data1) fs_out_color = vec4(1); }\n";

    GLuint po = mgl_make_program(vs5, fs5);
    EXPECT_TRUE(po != 0);
    if (po) {
        /* binding 0 = vbo at offset 8, stride 62 (unaligned: all attributes converted) */
        unsigned char buf[4 * 62];
        static const float pos[4][2] = { {-1,1}, {1,1}, {1,-1}, {-1,-1} };
        memset(buf, 0, sizeof buf);
        for (int i = 0; i < 4; i++) {
            memcpy(buf + 8 + i * 62 + 0, pos[i], 8);
            float rg[2] = { 0.0f, 0.0f }; memcpy(buf + 8 + i * 62 + 8, rg, 8);
            float b = 1.0f; memcpy(buf + 8 + i * 62 + 16, &b, 4);
            double d0[3] = { 4.0, 5.0, 6.0 }; memcpy(buf + 8 + i * 62 + 20, d0, 24);
            int i2[2] = { 3, 4 }; memcpy(buf + 8 + i * 62 + 44, i2, 8);
        }
        GLuint vbo; glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof buf, buf, GL_STATIC_DRAW);
        GLuint idx[4] = { 3, 2, 0, 1 };
        GLuint ebo; glGenBuffers(1, &ebo);

        GLuint vao; glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glBindVertexBuffer(0, vbo, 8, 62);
        glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexAttribFormat(1, 2, GL_FLOAT, GL_FALSE, 8);
        glVertexAttribFormat(2, 1, GL_FLOAT, GL_FALSE, 16);
        glVertexAttribLFormat(3, 3, GL_DOUBLE, 20);
        glVertexAttribIFormat(4, 2, GL_INT, 44);
        for (int i = 0; i < 5; i++) { glVertexAttribBinding(i, 0); glEnableVertexAttribArray(i); }

        glUseProgram(po);
        glUniform3d(glGetUniformLocation(po, "g_expected_data0"), 4.0, 5.0, 6.0);
        glUniform2i(glGetUniformLocation(po, "g_expected_data1"), 3, 4);
        glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_INT, NULL, 1, 0, 0);
        EXPECT_NO_GL_ERROR();
        EXPECT_PIXEL(2, 2, 0, 0, 255, 255, 2);

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteProgram(po);
    }
}

int main(void)
{
    RUN(case1_poison);
    RUN(case2_disabled_current);
    RUN(case3_dvec_element_pad);
    RUN(case4_renumbered_slots);
    RUN(case5_converted_double_and_float);
    return mgl_test_summary();
}
