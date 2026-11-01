/* geom r87: separable_tess_eval.patch_var_array.mat3x2 — a plain TES patch
 * input that IS an array of matrices must flatten to a member-per-element block
 * (SPIRV-Cross cannot emit arrays-of-matrices in io variables). */
#include "mgl_test.h"
#include <stdbool.h>
#include <string.h>

static const char *tes_pv = "#version 450\n"
"in gl_PerVertex { vec4 gl_Position; } gl_in[];\n"
"out gl_PerVertex { vec4 gl_Position; };\n"
"layout(triangles, point_mode) in;\n"
"patch in highp mat3x2 target[3];\n"
"highp uniform vec4 unusedZero;\n"
"highp vec4 readInputs()\n"
"{\n"
"    highp vec4 retValue = unusedZero;\n"
"    retValue += vec4(float(target[0][0][0]));\n"
"    retValue += vec4(float(target[1][0][0]));\n"
"    retValue += vec4(float(target[2][0][0]));\n"
"    return retValue;\n"
"}\n"
"void main()\n"
"{\n"
"    gl_Position = readInputs();\n"
"}\n";

TEST(regress_tes_patch_var_array)
{
    GLuint tes = glCreateShader(GL_TESS_EVALUATION_SHADER);
    GLuint p;
    GLint ok = GL_FALSE;
    char log[4096]; GLsizei len = 0;
    GLenum props[3] = { GL_TYPE, GL_ARRAY_SIZE, GL_LOCATION };
    GLint vals[3] = { 0, 0, 0 };
    GLint idx = -1;
    GLsizei plen = 0;

    glShaderSource(tes, 1, &tes_pv, NULL);
    glCompileShader(tes);
    glGetShaderiv(tes, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);

    p = glCreateProgram();
    glAttachShader(p, tes);
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "tes patch_var_array link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    /* GL view: ONE resource "target[0]", element type FLOAT_MAT3x2, size 3 */
    idx = glGetProgramResourceIndex(p, GL_PROGRAM_INPUT, "target[0]");
    EXPECT_TRUE(idx >= 0);
    if (idx >= 0) {
        glGetProgramResourceiv(p, GL_PROGRAM_INPUT, idx, 3, props, 3, &plen, vals);
        EXPECT_NO_GL_ERROR();
        EXPECT_EQ_INT(vals[0], GL_FLOAT_MAT3x2);
        EXPECT_EQ_INT(vals[1], 3);
    }

    glDeleteProgram(p);
    glDeleteShader(tes);
}

int main(void)
{
    RUN(regress_tes_patch_var_array);
    return mgl_test_summary();
}
