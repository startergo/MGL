/* ARB_gl_spirv: SPIR-V-binary compute shaders must link.
 *
 * Two CTS regressions:
 *  - KHR-GL46.gl_spirv.spirv_glsl_to_spirv_builtin_functions_test: a program built
 *    from a SPIR-V compute module failed with "no local work group size was
 *    specified" although the module carries OpExecutionMode LocalSize 1 1 1 (the
 *    detection scanned for a bare opcode in the packed first word and read the mode
 *    literal one operand too far).
 *  - KHR-GL46.gl_spirv.spirv_modules_error_verification_test: glCompileShader on a
 *    shader whose SPIR_V_BINARY_ARB state is TRUE must generate GL_INVALID_OPERATION.
 */
#include "mgl_test.h"
#include <stdio.h>
#include "spv_builtinfn.h"

TEST(spirv_binary_compile_status)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, builtinfn_spv, (GLsizei)builtinfn_spv_len);
    EXPECT_NO_GL_ERROR();
    GLint sb = 0;
    glGetShaderiv(s, GL_SPIR_V_BINARY_ARB, &sb);
    EXPECT_EQ_INT(1, sb);
    glSpecializeShader(s, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(1, ok);
    glDeleteShader(s);
}

TEST(spirv_binary_local_size_link)   /* the gl_spirv builtin-functions module links */
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, builtinfn_spv, (GLsizei)builtinfn_spv_len);
    EXPECT_NO_GL_ERROR();
    glSpecializeShader(s, "main", 0, NULL, NULL);
    EXPECT_NO_GL_ERROR();
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    GLint lok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &lok);
    if (!lok) {
        char log[2048] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %.1500s", log);
    }
    /* GL_COMPUTE_WORK_GROUP_SIZE reports the module's LocalSize (GL 4.6 20.11) */
    GLint wg[3] = {0, 0, 0};
    glGetProgramiv(p, GL_COMPUTE_WORK_GROUP_SIZE, &wg[0]);
    glGetProgramiv(p, GL_COMPUTE_WORK_GROUP_SIZE + 1, &wg[1]);
    glGetProgramiv(p, GL_COMPUTE_WORK_GROUP_SIZE + 2, &wg[2]);
    EXPECT_EQ_INT(1, wg[0]);
    EXPECT_EQ_INT(1, wg[1]);
    EXPECT_EQ_INT(1, wg[2]);
    glDeleteProgram(p);
    glDeleteShader(s);
}

TEST(spirv_binary_link_run)          /* and it dispatches without error */
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, builtinfn_spv, (GLsizei)builtinfn_spv_len);
    glSpecializeShader(s, "main", 0, NULL, NULL);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    GLint lok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &lok);
    EXPECT_EQ_INT(1, lok);

    /* the module's main computes GLSL.std.450 builtin results into a local and
     * returns; executing it must simply work */
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteShader(s);
}

TEST(spirv_compile_shader_error)     /* CompileShader on a SPIR-V shader: INVALID_OPERATION */
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, builtinfn_spv, (GLsizei)builtinfn_spv_len);
    EXPECT_NO_GL_ERROR();
    glCompileShader(s);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLint ok = 1;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(0, ok);
    glDeleteShader(s);
}

int main(void) {
    RUN(spirv_binary_compile_status);
    RUN(spirv_binary_local_size_link);
    RUN(spirv_binary_link_run);
    RUN(spirv_compile_shader_error);
    return mgl_test_summary();
}
