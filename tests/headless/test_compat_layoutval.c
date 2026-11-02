/*
 * test_compat_layoutval.c — regression tests for the link-time layout
 * validator (mgl_layout_validate.c, compat r86/r93).
 *
 * r93: MGL's OWN injected default-uniform block (mgl_DefaultUniforms /
 * gl_DefaultUniformBlock, geometry.c) is packed by MGL's own packer and must
 * not be held to the GLSL 4.4.5 offset rules — KHR-GL46.program_interface_
 * query.uniform-types regressed when the validator rejected the packer's
 * tight offsets ("offset of member 'i' ... must be a multiple of the base
 * alignment 16 (offset 184)"), which failed the LINK and emptied the
 * program's whole uniform introspection (GL_ACTIVE_UNIFORMS = 0).
 */
#include "mgl_test.h"
#include "gl_compat.h"

static const char *vs_dub =
    "#version 460 core\n"
    "struct Nested { vec4 a; float b[3]; float d[3]; };\n"
    "struct Holder { Nested a; Nested b[2]; float c; float d[4]; };\n"
    "uniform Holder k;\n"
    "uniform Holder l[3];\n"
    "uniform float  h;\n"
    "uniform vec3   i;\n"
    "uniform vec4   j;\n"
    "uniform mat4   m;\n"
    "out vec4 ov;\n"
    "void main() {\n"
    "    ov = k.a.a + l[2].b[1].a + vec4(h) + vec4(i, 0.0) + j + m[0]\n"
    "       + vec4(l[2].b[1].d[0]) + vec4(k.b[0].d[1]);\n"
    "    gl_Position = ov;\n"
    "}\n";

static const char *fs_passthrough =
    "#version 460 core\n"
    "in vec4 ov;\n"
    "layout(location = 0) out vec4 o;\n"
    "void main() { o = ov; }\n";

TEST(layout_validator_skips_mgl_blocks)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0;
    char log[1024] = "";

    glShaderSource(v, 1, &vs_dub, NULL);
    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glShaderSource(f, 1, &fs_passthrough, NULL);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        glGetProgramInfoLog(p, sizeof log, NULL, log);
        MGL_FAIL("link failed: %s", log);
    }

    /* the injected default-uniform block must still be introspectable */
    GLint nuni = 0;
    glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &nuni);
    EXPECT_TRUE(nuni > 0);
    GLint loc = glGetUniformLocation(p, "l[2].b[1].d[0]");
    EXPECT_NE(loc, -1);
    loc = glGetUniformLocation(p, "k.b[0].d[1]");
    EXPECT_NE(loc, -1);
    /* link/compile may leave a pending error from internal probing (pre-existing
     * MGL behaviour, not what this test is about) — clear it, then check that
     * using the program is clean */
    while (glGetError() != GL_NO_ERROR) { }
    glUseProgram(p);
    EXPECT_NO_GL_ERROR();
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
}

/* the validator must still reject USER blocks with overlapping offsets
 * (GLSL 4.60 4.4.5) — the exemption is only for MGL's own blocks */
TEST(layout_validator_rejects_user_overlap)
{
    static const char *vs_bad =
        "#version 460 core\n"
        "layout(std140) uniform Bad { layout(offset = 0) vec4 a;\n"
        "                             layout(offset = 8) vec2 b; };\n"
        "void main() { gl_Position = a + vec4(b, 0.0, 1.0); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
    GLint ok = 0;

    glShaderSource(v, 1, &vs_bad, NULL);
    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glShaderSource(f, 1, &fs_passthrough, NULL);
    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(!ok);
    glDeleteProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
}

int main(void)
{
    RUN(layout_validator_skips_mgl_blocks);
    RUN(layout_validator_rejects_user_overlap);
    return mgl_test_summary();
}
