/* tess r30: dynamic index into a matrix member of a tessellation INPUT block
 * (KHR-GL46.shading_language_420pack.length_of_vector_and_matrix fails the TES
 * SPIR-V -> MSL translation: "Trying to dynamically index into an array
 * interface variable in tessellation").  The lowering in tess_spv_lower.c
 * copies the member into a function-local and indexes the copy. */

#include "mgl_test.h"

static const char *vs =
    "#version 400\n#extension GL_ARB_shading_language_420pack : require\n"
    "out VSOutputBlock { vec4 vs_tcs_variable; } output_block;\n"
    "out vec4 vs_tcs_result;\n"
    "void main() {\n"
    "    output_block.vs_tcs_variable = vec4(0.125);\n"
    "    vs_tcs_result = vec4(0, 1, 0, 1);\n"
    "}\n";

/* one control point, identity factors */
static const char *tcs =
    "#version 400\n#extension GL_ARB_shading_language_420pack : require\n"
    "layout(vertices = 1) out;\n"
    "in VSOutputBlock { vec4 vs_tcs_variable; } input_block[];\n"
    "out TCSOutputBlock { mat2 tcs_tes_variable; } output_block[];\n"
    "out vec4 tcs_tes_result[];\n"
    "void main() {\n"
    "    output_block[gl_InvocationID].tcs_tes_variable = mat2(0.125, 0.25, 0.375, 0.5);\n"
    "    tcs_tes_result[gl_InvocationID] = vec4(0,1,0,1);\n"
    "    gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0; gl_TessLevelOuter[2] = 1.0;\n"
    "    gl_TessLevelInner[0] = 1.0;\n"
    "}\n";

/* dynamic index into input_block[0].tcs_tes_variable[i] AND the output member */
static const char *tes =
    "#version 400\n#extension GL_ARB_shading_language_420pack : require\n"
    "layout(triangles, equal_spacing) in;\n"
    "in  vec4 tcs_tes_result[];\n"
    "out vec4 tes_gs_result;\n"
    "in  TCSOutputBlock { mat2 tcs_tes_variable; } input_block[];\n"
    "out TCSOutputBlock { mat2 tes_gs_variable; } output_block;\n"
    "void main() {\n"
    "    vec4 result = vec4(0, 1, 0, 1);\n"
    "    const uint input_member_length = input_block[0].tcs_tes_variable.length();\n"
    "    const uint output_member_length = output_block.tes_gs_variable.length();\n"
    "    float array_in [input_block[0].tcs_tes_variable.length()];\n"
    "    float array_out[output_block.tes_gs_variable.length()];\n"
    "    for (uint i = 0; i < input_member_length; ++i)\n"
    "        array_in[i]  = input_block[0].tcs_tes_variable[i].x;\n"
    "    for (uint i = 0; i < output_member_length; ++i)\n"
    "        array_out[i] = input_block[0].tcs_tes_variable[i].y;\n"
    "    float sum_in = 0, sum_out = 0;\n"
    "    for (uint i = 0; i < array_in.length(); ++i)  sum_in  += array_in[i];\n"
    "    for (uint i = 0; i < array_out.length(); ++i) sum_out += array_out[i];\n"
    "    if (0.5 != sum_in || 0.75 != sum_out) result = vec4(1, 0, 0, 1);\n"
    "    else if (vec4(0, 1, 0, 1) != tcs_tes_result[0]) result = vec4(1, 0, 0, 1);\n"
    "    tes_gs_result = result;\n"
    "    gl_Position = vec4(gl_TessCoord.xy * 2.0 - 1.0, 0.0, 1.0);\n"
    "    output_block.tes_gs_variable = input_block[0].tcs_tes_variable;\n"
    "}\n";

static const char *fs =
    "#version 400\n#extension GL_ARB_shading_language_420pack : require\n"
    "in vec4 tes_gs_result;\n"
    "out vec4 fragColor;\n"
    "void main() { fragColor = tes_gs_result; }\n";

TEST(TessDynamicIoMatrixMember)
{
    GLuint prog = glCreateProgram();
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &vs, NULL); glCompileShader(s); glAttachShader(prog, s);
    s = glCreateShader(GL_TESS_CONTROL_SHADER);
    glShaderSource(s, 1, &tcs, NULL); glCompileShader(s); glAttachShader(prog, s);
    s = glCreateShader(GL_TESS_EVALUATION_SHADER);
    glShaderSource(s, 1, &tes, NULL); glCompileShader(s); glAttachShader(prog, s);
    s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 1, &fs, NULL); glCompileShader(s); glAttachShader(prog, s);
    {
        GLint cs = 0;
        glGetShaderiv(glGetAttachedShaders ? 0 : 0, 0, 0);
    }
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    {
        char log[4096];
        GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof log, &len, log);
        fprintf(stderr, "LINK FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_TRUE(ok == GL_TRUE);

    glUseProgram(prog);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_PATCHES, 0, 1);

    /* green = dynamic indices lowered and computed correctly */
    EXPECT_PIXEL(32, 32, 0, 255, 0, 255, 4);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(TessDynamicIoMatrixMember);
    printf("== %s: %d checks, %d failures\n", __FILE__, mgl_test_checks, mgl_test_failures);
    return mgl_test_failures != 0;
}
