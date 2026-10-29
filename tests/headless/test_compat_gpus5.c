/* test_compat_gpus5.c — GL_ARB_gpu_shader5 (r28 sweep: KHR-GL46.gpu_shader5_gl
 * 3 NotSupported cases: implicit_conversions, function_overloading,
 * float_encoding).
 *
 * The extension's functionality is GL 4.0 core, which MGL implements (the
 * core gpu_shader5 group already renders fma/precise/indexing cases); glslang
 * implements the extension itself (implicit conversions ParseHelper:8135,
 * profileRequires 400-or-ARB_gpu_shader5), and shaders.c upgrades #version 150
 * sources to 330 core while keeping the #extension line.  Only the extension
 * string was missing.  These checks mirror the three CTS cases. */
#include "mgl_test.h"

static int ext_listed(const char *name)
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (strcmp((const char *)glGetStringi(GL_EXTENSIONS, i), name) == 0)
            return 1;
    return 0;
}

static int compile_status(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = "";
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        fprintf(stderr, "    [gpus5] compile log: %.400s\n", log);
    }
    glDeleteShader(s);
    while (glGetError() != GL_NO_ERROR) {}
    return ok;
}

/* gl_VertexID quad, at #version 150 so the shaders.c legacy upgrade path runs */
#define VS_HEAD \
    "#version 150\n"                            \
    "#extension GL_ARB_gpu_shader5 : require\n"

static const char *fs_passthrough =
    "#version 150 core\n"
    "in  vec4 result;\n"
    "out vec4 color;\n"
    "void main() { color = result; }\n";

/* one implicit-conversion case, CTS style: v = DEST(u2) - u1 over uniforms of
 * the source type; white iff the conversion is exact (u1 == u2) */
#define CONV_VS(dst, src, zero)                                          \
    VS_HEAD                                                             \
    "uniform " src " u1;\n"                                              \
    "uniform " src " u2;\n"                                              \
    "out vec4 result;\n"                                                 \
    "void main() {\n"                                                    \
    "    " dst " v = " dst "(u2) - u1;\n"                                \
    "    result = vec4(0.0);\n"                                          \
    "    if (" zero " == v) result = vec4(1.0);\n"                       \
    "    switch (gl_VertexID) {\n"                                       \
    "      case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"    \
    "      case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"    \
    "      case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"    \
    "      case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"    \
    "    }\n"                                                            \
    "}\n"

static void draw_full(void)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDeleteVertexArrays(1, &vao);
}

TEST(extension_advertised)
{
    EXPECT_TRUE(ext_listed("GL_ARB_gpu_shader5"));
    EXPECT_NO_GL_ERROR();
}

TEST(implicit_conversions)
{
    /* int -> uint (needs the extension; int -> float is GLSL 1.20) */
    static const char *vs_i2u = CONV_VS("uint", "int", "0u");
    GLuint p = mgl_make_program(vs_i2u, fs_passthrough);
    EXPECT_TRUE(p != 0);
    if (p) {
        glUseProgram(p);
        glUniform1i(glGetUniformLocation(p, "u1"), 7);
        glUniform1i(glGetUniformLocation(p, "u2"), 7);
        glViewport(0, 0, 4, 4);
        draw_full();
        EXPECT_PIXEL(1, 1, 255, 255, 255, 255, 0);
        glDeleteProgram(p);
    }

    /* ivec4 -> uvec4 */
    static const char *vs_i2uv = CONV_VS("uvec4", "ivec4", "uvec4(0u)");
    p = mgl_make_program(vs_i2uv, fs_passthrough);
    EXPECT_TRUE(p != 0);
    if (p) {
        glUseProgram(p);
        glUniform4i(glGetUniformLocation(p, "u1"), 1, 2, 3, 4);
        glUniform4i(glGetUniformLocation(p, "u2"), 1, 2, 3, 4);
        draw_full();
        EXPECT_PIXEL(1, 1, 255, 255, 255, 255, 0);
        glDeleteProgram(p);
    }

    /* uvec4 -> vec4 (float compare against converted zero) */
    static const char *vs_u2v = CONV_VS("vec4", "uvec4", "vec4(0.0)");
    p = mgl_make_program(vs_u2v, fs_passthrough);
    EXPECT_TRUE(p != 0);
    if (p) {
        glUseProgram(p);
        glUniform4ui(glGetUniformLocation(p, "u1"), 9, 8, 7, 6);
        glUniform4ui(glGetUniformLocation(p, "u2"), 9, 8, 7, 6);
        draw_full();
        EXPECT_PIXEL(1, 1, 255, 255, 255, 255, 0);
        glDeleteProgram(p);
    }
    EXPECT_NO_GL_ERROR();
}

TEST(function_overloading)
{
    /* CTS GPUShader5FunctionOverloadingTest: f(vec4,vec4) vs f(uvec4,uvec4),
     * called with (ivec4, uvec4): the ivec4->uvec4 implicit conversion is the
     * gpu_shader5 part.  u1 == u2 as bits => black (a - b == 0), so invert. */
    static const char *vs =
        VS_HEAD
        "uniform ivec4 u1;\n"
        "uniform uvec4 u2;\n"
        "out     vec4  result;\n"
        "vec4 f(in vec4 a, in vec4 b) { return a * b; }\n"
        "vec4 f(in uvec4 a, in uvec4 b) { return vec4(a - b); }\n"
        "void main() {\n"
        "    result = f(u1, u2);\n"
        "    if (result == vec4(0.0)) result = vec4(1.0);\n"
        "    switch (gl_VertexID) {\n"
        "      case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"
        "      case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"
        "      case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"
        "      case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"
        "    }\n"
        "}\n";
    GLuint p = mgl_make_program(vs, fs_passthrough);
    EXPECT_TRUE(p != 0);
    if (p) {
        glUseProgram(p);
        glUniform4i(glGetUniformLocation(p, "u1"), 5, 6, 7, 8);
        glUniform4ui(glGetUniformLocation(p, "u2"), 5, 6, 7, 8);
        glViewport(0, 0, 4, 4);
        draw_full();
        EXPECT_PIXEL(1, 1, 255, 255, 255, 255, 0);
        glDeleteProgram(p);
        EXPECT_NO_GL_ERROR();
    }
}

TEST(float_encoding)
{
    /* CTS GPUShader5FloatEncodingTest: floatBitsToInt/floatBitsToUint/
     * intBitsToFloat at #version 150 need gpu_shader5 (3.30 core otherwise);
     * the upgraded-to-330 shader must still honour the extension require */
    static const char *vs =
        VS_HEAD
        "uniform ivec4 expected_value;\n"
        "uniform vec4  value;\n"
        "out vec4 result;\n"
        "void main() {\n"
        "    result = vec4(1.0);\n"
        "    ivec4 ret_val = floatBitsToInt(value);\n"
        "    if (expected_value != ret_val) result = vec4(0.0);\n"
        "    if (uintBitsToFloat(floatBitsToUint(value)) != value) result = vec4(0.0);\n"
        "    if (intBitsToFloat(floatBitsToInt(value)) != value) result = vec4(0.0);\n"
        "    switch (gl_VertexID) {\n"
        "      case 0: gl_Position = vec4(-1.0, 1.0, 0.0, 1.0); break;\n"
        "      case 1: gl_Position = vec4( 1.0, 1.0, 0.0, 1.0); break;\n"
        "      case 2: gl_Position = vec4(-1.0,-1.0, 0.0, 1.0); break;\n"
        "      case 3: gl_Position = vec4( 1.0,-1.0, 0.0, 1.0); break;\n"
        "    }\n"
        "}\n";
    GLuint p = mgl_make_program(vs, fs_passthrough);
    EXPECT_TRUE(p != 0);
    if (p) {
        const float f[4] = { -1.0f, -1234.0f, 1.0f, 1234.0f };
        const int i[4] = { (int)0xbf800000, (int)0xc49a4000, 0x3f800000, 0x449a4000 };
        glUseProgram(p);
        glUniform4fv(glGetUniformLocation(p, "value"), 1, f);
        glUniform4iv(glGetUniformLocation(p, "expected_value"), 1, i);
        glViewport(0, 0, 4, 4);
        draw_full();
        EXPECT_PIXEL(1, 1, 255, 255, 255, 255, 0);
        glDeleteProgram(p);
        EXPECT_NO_GL_ERROR();
    }
}

TEST(int_to_uint_rejected_without_extension)
{
    static const char *src =
        "#version 330 core\n"
        "out vec4 color;\n"
        "uniform int u1;\n"
        "void main() {\n"
        "    uint v = 1u - u1;\n" /* implicit int -> uint: not core GLSL */
        "    color = vec4(float(v));\n"
        "}\n";
    EXPECT_TRUE(!compile_status(GL_FRAGMENT_SHADER, src));
}

int main(void)
{
    RUN(extension_advertised);
    RUN(implicit_conversions);
    RUN(function_overloading);
    RUN(float_encoding);
    RUN(int_to_uint_rejected_without_extension);
    return mgl_test_summary();
}
