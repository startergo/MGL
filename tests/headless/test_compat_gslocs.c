/*
 * test_compat_gslocs.c — varying location limits for geometry-stage inputs
 * (arrays per vertex) and outputs; GL 4.6 11.1.1 caps.
 */
#include "mgl_test.h"

static const char *vs =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout (location = 15) flat out float gohan;\n"
    "void main() { gl_Position = vec4(0); gohan = 1.0; }\n";
static const char *fs =
    "#version 430 core\n"
    "layout(location = 0) out vec4 c;\n"
    "void main() { c = vec4(1); }\n";
static const char *gs_in_over =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "layout (location = 32) flat in float gohan[];\n"
    "layout (location = 0) out float fs_fs;\n"
    "void main() { fs_fs = gohan[0]; EmitVertex(); EndPrimitive(); }\n";
static const char *gs_in_ok =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "layout (location = 15) flat in float gohan[];\n"
    "layout (location = 0) out float fs_fs;\n"
    "void main() { fs_fs = gohan[0]; EmitVertex(); EndPrimitive(); }\n";
static const char *gs_in_dmat_over =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "layout (location = 31) flat in dmat2 gohan[];\n"
    "layout (location = 0) out float fs_fs;\n"
    "void main() { fs_fs = float(gohan[0][0][0].x); EmitVertex(); EndPrimitive(); }\n";
static const char *gs_out_over =
    "#version 430 core\n"
    "#extension GL_ARB_enhanced_layouts : require\n"
    "layout(points) in;\n"
    "layout(points, max_vertices = 1) out;\n"
    "layout (location = 32) flat out vec4 goku;\n"
    "void main() { goku = vec4(1); EmitVertex(); EndPrimitive(); }\n";

static GLuint link(const char *gs_src)
{
    GLuint p = glCreateProgram();
    struct { GLenum t; const char *s; } st[4] = {
        {GL_VERTEX_SHADER, vs}, {GL_GEOMETRY_SHADER, gs_src}, {GL_FRAGMENT_SHADER, fs}};
    for (int i = 0; i < 3; i++)
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
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log);
        printf("    link log: %s", log); }
    glDeleteProgram(p);
    while (glGetError() != GL_NO_ERROR) {}
    return (GLuint)ok;
}

TEST(gs_location_limits)
{
    /* order matters for a latent MGL bug, not the rule under test: a glslang
     * cross-stage layout-mismatch failure (the over-limit cases below pair the
     * VS's gohan@15 with gohan@32/31) leaves the shader-object cache poisoned
     * — the NEXT link of those sources fails with an empty "GLSL link failed".
     * The legal case links first, the three illegal ones after it all expect
     * failure anyway (repro: .agent/probe_order.c). */
    GLuint ok = link(gs_in_ok);                /* float input at the last legal
                                                 location (MAX_GEOMETRY_INPUT_
                                                 COMPONENTS=64/4-1=15); the VS
                                                 writes the matching gohan@15 so
                                                 the only limit in play is the
                                                 location cap (GL 4.6 7.4.1: an
                                                 unmatched statically used input
                                                 would be a link error) */
    printf("  gs in float @31 linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    ok = link(gs_in_over);      /* float input at location 32 */
    printf("  gs in float @32 linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    ok = link(gs_in_dmat_over);         /* dmat2 input (2 locs) at 31 */
    printf("  gs in dmat2 @31 linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
    ok = link(gs_out_over);             /* vec4 output at location 32 */
    printf("  gs out vec4 @32 linked=%u", ok);
    EXPECT_EQ_INT(ok, GL_FALSE);
}

int main(void)
{
    RUN(gs_location_limits);
    return mgl_test_summary();
}
