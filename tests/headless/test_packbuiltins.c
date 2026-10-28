/* CTS gl_spirv.spirv_glsl_to_spirv_builtin_functions: the GLSL variant of the
 * pack/unpack builtin shader must compile (MGL failed it with a warnings-only log). */
#include "mgl_test.h"

static const char *cs =
    "#version 450\n"
    "layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
    "void main()\n"
    "{\n"
    "    vec2 v2val = vec2(0.1, 0.2);\n"
    "    vec4 v4val = vec4(0.1, 0.2, 0.3, 0.4);\n"
    "    uint uival1 = packUnorm2x16(v2val);\n"
    "    uint uival2 = packSnorm2x16(v2val);\n"
    "    uint uival3 = packUnorm4x8(v4val);\n"
    "    uint uival4 = packSnorm4x8(v4val);\n"
    "    v2val = unpackUnorm2x16(uival1);\n"
    "    v2val = unpackSnorm2x16(uival2);\n"
    "    v4val = unpackUnorm4x8(uival3);\n"
    "    v4val = unpackSnorm4x8(uival4);\n"
    "    uvec2 uv2val = uvec2(10, 20);\n"
    "    double dval = packDouble2x32(uv2val);\n"
    "    uv2val = unpackDouble2x32(dval);\n"
    "    uint uival5 = packHalf2x16(v2val);\n"
    "    v2val = unpackHalf2x16(uival5);\n"
    "}\n";

TEST(pack_compile)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = "";
        GLsizei n = 0;
        glGetShaderInfoLog(s, sizeof log, &n, log);
        MGL_FAIL("compile failed: %.2500s", log);
    }
    glDeleteShader(s);
}

TEST(pack_link)
{
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs, NULL);
    glCompileShader(s);
    GLuint p = glCreateProgram();
    glAttachShader(p, s);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = "";
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %.2500s", log);
    }
    glUseProgram(p);
    glDispatchCompute(1, 1, 1);
    EXPECT_NO_GL_ERROR();
    glDeleteProgram(p);
    glDeleteShader(s);
}

int main(void) {
    RUN(pack_compile);
    RUN(pack_link);
    return mgl_test_summary();
}
