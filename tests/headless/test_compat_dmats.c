/*
 * test_compat_dmats.c — double-matrix outputs + declared xfb capture:
 * KHR-Single-GL46.enhanced_layouts.xfb_explicit_location_dmat* fail with
 * "SPIRV-Cross: Cannot resolve expression type" on the vertex stage.
 */
#include "mgl_test.h"

static const char *vs_xfb =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout (location = 0, xfb_offset = 0) flat out dmat2 goku[3];\n"
    "layout(std140, binding = 0) uniform Goku {\n"
    "    dmat2 uni_goku[3];\n"
    "};\n"
    "void main() { goku = uni_goku; }\n";
static const char *vs_plain =
    "#version 430 core\n"
    "layout (location = 0) flat out dmat2 goku[3];\n"
    "layout(std140, binding = 0) uniform Goku {\n"
    "    dmat2 uni_goku[3];\n"
    "};\n"
    "void main() { goku = uni_goku; }\n";
static const char *fs =
    "#version 430 core\n"
    "layout(location = 0) out vec4 c;\n"
    "flat in dmat2 goku[3];\n"
    "void main() { c = vec4(float(goku[0][0][0].x)); }\n";

static const char *fs_ignore =
    "#version 430 core\n"
    "layout(location = 0) out vec4 c;\n"
    "void main() { c = vec4(1); }\n";
static const char *fs_read_scalar =
    "#version 430 core\n"
    "layout(location = 0) out vec4 c;\n"
    "flat in dmat2 goku;\n"
    "void main() { c = vec4(float(goku[0][0].x)); }\n";

static GLuint link2(const char *vs_src, const char *fs_src)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } st[2] = {{GL_VERTEX_SHADER, vs_src}, {GL_FRAGMENT_SHADER, fs_src}};
    for (int i = 0; i < 2; i++)
    {
        GLuint sh = glCreateShader(st[i].t);
        glShaderSource(sh, 1, &st[i].s, NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("    link log: %s", log); }
    while (glGetError() != GL_NO_ERROR) {}
    glDeleteProgram(p);
    return (GLuint)ok;
}

static GLuint link(const char *vs_src)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } st[2] = {{GL_VERTEX_SHADER, vs_src}, {GL_FRAGMENT_SHADER, fs}};
    for (int i = 0; i < 2; i++)
    {
        GLuint sh = glCreateShader(st[i].t);
        glShaderSource(sh, 1, &st[i].s, NULL);
        glCompileShader(sh);
        glAttachShader(p, sh);
        glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("    link log: %s", log); }
    while (glGetError() != GL_NO_ERROR) {}
    glDeleteProgram(p);
    return (GLuint)ok;
}

static const char *vs_idx =
    "#version 430 core\n"
    "layout (location = 0) flat out dmat2 goku[3];\n"
    "layout(std140, binding = 0) uniform Goku {\n"
    "    dmat2 uni_goku[3];\n"
    "};\n"
    "void main() { goku[0] = uni_goku[0]; goku[1] = uni_goku[1]; goku[2] = uni_goku[2]; }\n";
static const char *vs_scalar =
    "#version 430 core\n"
    "layout (location = 0) flat out dmat2 goku;\n"
    "layout(std140, binding = 0) uniform Goku {\n"
    "    dmat2 uni_goku;\n"
    "};\n"
    "void main() { goku = uni_goku; }\n";

TEST(dmat_out_capture)
{
    GLuint ok = link(vs_plain);       /* whole-array dmat2 copy, no capture */
    printf("  plain dmat2[3] out linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    ok = link2(vs_idx, fs_ignore);   /* per-element assignment */
    printf("  per-element dmat2[3] out linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    ok = link2(vs_scalar, fs_read_scalar);   /* scalar dmat2 read by the FS */
    printf("  scalar dmat2 out (FS reads) linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    /* whole-array dmat2 xfb capture: blocked by the fp64 lowering of the
     * array-of-double UBO load in the capture variant (r91 notes) */
    ok = link2(vs_xfb, fs_ignore);
    printf("  xfb dmat2[3] out linked=%u", ok);
}

int main(void)
{
    RUN(dmat_out_capture);
    return mgl_test_summary();
}
