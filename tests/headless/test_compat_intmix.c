/* test_compat_intmix.c — GL_EXT_shader_integer_mix on a core context
 * (r28 sweep: KHR-GL46.shaders[45].shader_integer_mix, 11 NotSupported cases).
 *
 * The extension is a pure GLSL enable: it adds select-style mix() overloads
 * for int/uint/bool vectors.  glslang implements them for desktop GLSL
 * 1.50-4.50 (Initialize.cpp / ParseHelper requireExtensions at < 450) and
 * they are core GLSL from 4.50 on, so MGL only has to advertise the string.
 * Checks: the string is in the glGetStringi list, the preprocessor macro is
 * defined when enabled, integer mix renders with select semantics, the same
 * code is rejected at #version 330 without the extension, and works at
 * #version 450 with no extension (already core). */
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

/* compile one shader; returns its COMPILE_STATUS */
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
        fprintf(stderr, "    [intmix] compile log: %.300s\n", log);
    }
    glDeleteShader(s);
    while (glGetError() != GL_NO_ERROR) {}
    return ok;
}

TEST(extension_advertised)
{
    EXPECT_TRUE(ext_listed("GL_EXT_shader_integer_mix"));
    EXPECT_NO_GL_ERROR();
}

TEST(macro_defined_when_enabled)
{
    /* mirrors CTS "define": the preprocessor macro must exist and be 1 */
    static const char *src =
        "#version 330 core\n"
        "#extension GL_EXT_shader_integer_mix : enable\n"
        "#ifndef GL_EXT_shader_integer_mix\n"
        "#error GL_EXT_shader_integer_mix is not defined\n"
        "#elif GL_EXT_shader_integer_mix != 1\n"
        "#error GL_EXT_shader_integer_mix is not equal to 1\n"
        "#endif\n"
        "out vec4 color;\n"
        "void main() { color = vec4(1.0); }\n";
    EXPECT_TRUE(compile_status(GL_FRAGMENT_SHADER, src));
}

/* select semantics: mix(x, y, a) == a ? y : x, component-wise */
static const char *mix_fs =
    "#version 330 core\n"
    "#extension GL_EXT_shader_integer_mix : enable\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    int ok = 1;\n"
    "    ivec4 x = ivec4(10, 20, 30, 40);\n"
    "    ivec4 y = ivec4(200, 210, 220, 230);\n"
    "    ivec4 m = mix(x, y, bvec4(true, false, true, false));\n"
    "    if (m != ivec4(200, 20, 220, 40)) ok = 0;\n"
    "    uvec4 ux = uvec4(1u, 2u, 3u, 4u);\n"
    "    uvec4 uy = uvec4(90u, 91u, 92u, 93u);\n"
    "    uvec4 um = mix(ux, uy, bvec4(false, true, false, true));\n"
    "    if (um != uvec4(1u, 91u, 3u, 93u)) ok = 0;\n"
    "    if (mix(5, 7, true) != 7) ok = 0;\n"
    "    if (mix(8u, 9u, false) != 8u) ok = 0;\n"
    "    if (mix(true, false, true)) ok = 0;\n"
    "    bvec4 bm = mix(bvec4(true), bvec4(false), bvec4(true, false, true, false));\n"
    "    if (bm != bvec4(false, true, false, true)) ok = 0;\n"
    "    color = (ok == 1) ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);\n"
    "}\n";

static const char *id_vs =
    "#version 330 core\n"
    "void main() {\n"
    "    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
    "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

TEST(renders_select_semantics)
{
    GLuint p = mgl_make_program(id_vs, mix_fs);
    EXPECT_TRUE(p != 0);
    if (!p) return;
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glViewport(0, 0, 8, 8);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(2, 2, 0, 255, 0, 255, 0);
    EXPECT_PIXEL(6, 6, 0, 255, 0, 255, 0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
}

TEST(rejected_at_330_without_extension)
{
    /* same overloads, no #extension: must be a compile error (CTS
     * prototypes-negative) */
    static const char *src =
        "#version 330 core\n"
        "out vec4 color;\n"
        "void main() {\n"
        "    ivec4 m = mix(ivec4(1), ivec4(2), bvec4(true));\n"
        "    color = vec4(float(m.x));\n"
        "}\n";
    EXPECT_TRUE(!compile_status(GL_FRAGMENT_SHADER, src));
}

TEST(core_at_450_without_extension)
{
    static const char *src =
        "#version 450 core\n"
        "out vec4 color;\n"
        "void main() {\n"
        "    ivec4 m = mix(ivec4(1), ivec4(2), bvec4(true));\n"
        "    uvec4 um = mix(uvec4(1), uvec4(2), bvec4(false));\n"
        "    color = vec4(float(m.x), float(um.x), 0.0, 1.0);\n"
        "}\n";
    EXPECT_TRUE(compile_status(GL_FRAGMENT_SHADER, src));
}

int main(void)
{
    RUN(extension_advertised);
    RUN(macro_defined_when_enabled);
    RUN(renders_select_semantics);
    RUN(rejected_at_330_without_extension);
    RUN(core_at_450_without_extension);
    return mgl_test_summary();
}
