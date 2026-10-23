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

int main(void)
{
    RUN(case1_poison);
    RUN(case2_disabled_current);
    return mgl_test_summary();
}
