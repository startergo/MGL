/* test_compat_ssbo_unused — declared-but-unused SSBO blocks must stay ACTIVE
 * (dEQP ES31 ssbo.layout.random.* queries every declared block; glslang may
 * strip unused ones from the SPIR-V — MGL must keep them introspectable).
 */
#include "mgl_test.h"

static const char *kCs =
    "#version 450\n"
    "layout(local_size_x = 1) in;\n"
    "layout(binding = 0) buffer BlockB { uvec2 a; float b; };\n"
    "layout(binding = 1) buffer BlockC { int c; ivec4 d; float e; } blockC;\n"
    "layout(shared, binding = 2) buffer BlockD { vec2 f; vec2 g; };\n"
    "layout(binding = 3) buffer BlockE { uvec2 h; } blockE;\n"
    "void main() {\n"
    "    a = uvec2(1u, 2u);\n"
    "    b = 3.0;\n"
    "    blockC.c = 5;\n"
    "    blockC.d = ivec4(-4, 2, -1, 9);\n"
    "}\n";

TEST(compat_ssbo_unused)
{
    GLuint p = glCreateProgram(), s = glCreateShader(GL_COMPUTE_SHADER);
    GLint ok = 0;
    glShaderSource(s, 1, &kCs, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, s);
    glDeleteShader(s);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);

    GLint nblocks = 0;
    glGetProgramInterfaceiv(p, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &nblocks);
    /* all four declared blocks are active: the used ones plus BlockD (unused) and
     * BlockE (instantiated but no member referenced) */
    EXPECT_EQ_INT(nblocks, 4);

    GLuint d = glGetProgramResourceIndex(p, GL_SHADER_STORAGE_BLOCK, "BlockD");
    EXPECT_TRUE(d != GL_INVALID_INDEX);

    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(compat_ssbo_unused);
    return mgl_test_summary();
}
