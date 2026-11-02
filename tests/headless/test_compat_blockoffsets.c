/*
 * test_compat_blockoffsets.c — GLSL 4.40+ explicit block-member offset rules
 * (ARB_enhanced_layouts): alignment + no-overlap validation, diagnosed through
 * COMPILE status + info log (GLSL 4.6 §4.4.5).
 */
#include "mgl_test.h"

static GLuint try_compile(GLenum type, const char *src, char *log, GLsizei logsz)
{
    /* the layout rules are validated when MGL sees the stage's SPIR-V, i.e. at
     * link time (GL lets an implementation defer compile-time errors there) */
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLuint p = glCreateProgram();
    glAttachShader(p, sh);
    if (type != GL_FRAGMENT_SHADER)
    {
        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        static const char *fs_src = "#version 430 core\nout vec4 c; void main() { c = vec4(1); }\n";
        glShaderSource(fs, 1, &fs_src, NULL);
        glCompileShader(fs);
        glAttachShader(p, fs);
        glDeleteShader(fs);
    }
    glLinkProgram(p);
    GLint ok = -1;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (log) { glGetProgramInfoLog(p, logsz, NULL, log); log[logsz-1] = 0;
        GLint cok = 1; glGetShaderiv(sh, GL_COMPILE_STATUS, &cok);
        if (!cok) glGetShaderInfoLog(sh, logsz, NULL, log); }
    glDeleteProgram(p);
    glDeleteShader(sh);
    while (glGetError() != GL_NO_ERROR) {}
    return (GLuint)ok;
}

static const char *ssbo_mat2_overlap =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(std140) buffer Block {\n"
    "    layout (offset = 16) mat2 b;\n"
    "    layout (offset = 16) mat2 a;\n"
    "} block;\n"
    "void main() { block.b = mat2(1); block.a = mat2(0); }\n";

static const char *ssbo_dmat2_overlap =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(std140) buffer Block {\n"
    "    layout (offset = 16) dmat2 b;\n"
    "    layout (offset = 16) dmat2 a;\n"
    "} block;\n"
    "void main() { block.b = dmat2(1); block.a = dmat2(0); }\n";

static const char *ssbo_dmat2_overlap_gap =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(std140) buffer Block {\n"
    "    layout (offset = 0) dmat2 b;\n"
    "    layout (offset = 32) dmat2 a;\n"
    "} block;\n"
    "void main() { block.b = dmat2(1); block.a = dmat2(0); }\n";

static const char *ssbo_badalign =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(std140) buffer Block {\n"
    "    layout (offset = 3) vec4 a;\n"
    "} block;\n"
    "void main() { block.a = vec4(1); }\n";

TEST(block_offsets_compile_time)
{
    char log[2048];
    GLuint ok = try_compile(GL_VERTEX_SHADER, ssbo_mat2_overlap, log, sizeof log);
    printf("  mat2 overlap compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);        /* overlapping offsets: compile error */

    ok = try_compile(GL_VERTEX_SHADER, ssbo_dmat2_overlap, log, sizeof log);
    printf("  dmat2 overlap compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);

    ok = try_compile(GL_VERTEX_SHADER, ssbo_badalign, log, sizeof log);
    printf("  bad alignment compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);

    ok = try_compile(GL_VERTEX_SHADER, ssbo_dmat2_overlap_gap, log, sizeof log);
    printf("  dmat2 non-overlap compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_TRUE);         /* legal: 0..32, 32..64 */
}

/* GL 4.6 11.1.1: link error when an in/out location exceeds the stage's
 * component limits (128 components = 32 locations for varyings, 16 vertex
 * attribute locations) — KHR-Single-GL46.enhanced_layouts.varying_location_limit */
static GLuint link_vs_fs(const char *vs_src, const char *fs_src)
{
    GLuint p = glCreateProgram(), vs = glCreateShader(GL_VERTEX_SHADER),
           fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs, 1, &vs_src, NULL); glCompileShader(vs);
    glShaderSource(fs, 1, &fs_src, NULL); glCompileShader(fs);
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = -1; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("    log: %s", log); }
    glDeleteShader(vs); glDeleteShader(fs);
    while (glGetError() != GL_NO_ERROR) {}
    GLuint save = p;
    glDeleteProgram(p);
    return (GLuint)ok ? 1 : 0 ? 0 : (ok ? 1u : 0u) + (save ? 0 : 0);
}

/* GLSL 4.60 4.4.5: offset/align on a member require the block to be std140 or
 * std430 — shared/packed blocks with such members must fail (the CTS conflict
 * families) while packing-legal shared blocks must still compile */
TEST(layout_qualifier_conflict)
{
    char log[2048];
    GLuint ok = try_compile(GL_VERTEX_SHADER,
        "#version 430 core\n"
        "layout (shared) buffer Block {\n"
        "    layout (offset = 16) vec4 b;\n"
        "    layout (align = 64) vec4 a;\n"
        "} block;\n"
        "void main() { block.b = vec4(1); }\n", log, sizeof log);
    printf("  ssb shared + member offset/align compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);

    ok = try_compile(GL_VERTEX_SHADER,
        "#version 430 core\n"
        "layout (packed) uniform Block {\n"
        "    layout (offset = 16) vec4 b;\n"
        "} block;\n"
        "void main() { block.b = vec4(1); }\n", log, sizeof log);
    printf("  ubo packed + member offset compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);

    ok = try_compile(GL_VERTEX_SHADER,
        "#version 430 core\n"
        "layout (shared) uniform Block {\n"
        "    vec4 b;\n"
        "    vec4 a;\n"
        "} block;\n"
        "void main() { gl_Position = block.b + block.a; }\n", log, sizeof log);
    printf("  shared (no member quals) compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_TRUE);
}

TEST(varying_location_limits)
{
    GLuint ok = link_vs_fs(
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 32) flat out vec4 x;\n"
        "void main() { x = vec4(1); gl_Position = vec4(0); }\n",
        "#version 430 core\nlayout(location = 0) out vec4 c;\n"
        "layout(location = 32) flat in vec4 x;\nvoid main() { c = x; }\n");
    EXPECT_EQ_INT(ok, 0);          /* 33rd location: out of range */

    ok = link_vs_fs(
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 31) flat out vec4 x;\n"
        "void main() { x = vec4(1); gl_Position = vec4(0); }\n",
        "#version 430 core\nlayout(location = 0) out vec4 c;\n"
        "layout(location = 31) flat in vec4 x;\nvoid main() { c = x; }\n");
    EXPECT_EQ_INT(ok, 1);          /* the last legal location */

    ok = link_vs_fs(
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 31) out dvec3 x;\n"       /* 2 slots -> 33 */
        "void main() { x = dvec3(1); gl_Position = vec4(0); }\n",
        "#version 430 core\nlayout(location = 0) out vec4 c;\n"
        "layout(location = 31) in dvec3 x;\nvoid main() { c = vec4(x); }\n");
    EXPECT_EQ_INT(ok, 0);

    ok = link_vs_fs(
        "#version 430 core\n"
        "#extension GL_ARB_enhanced_layouts : require\n"
        "layout (location = 16) in vec4 a;\n"
        "void main() { gl_Position = a; }\n",
        "#version 430 core\nlayout(location = 0) out vec4 c;\nvoid main() { c = vec4(1); }\n");
    EXPECT_EQ_INT(ok, 0);          /* MAX_VERTEX_ATTRIBS = 16 */
}

int main(void)
{
    RUN(block_offsets_compile_time);
    RUN(layout_qualifier_conflict);
    RUN(varying_location_limits);
    return mgl_test_summary();
}
