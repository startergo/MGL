/*
 * test_compat_esconst.c — ES 3.1 semantics preserved by MGL's '#version N es'
 * path (ARB_ES3_1_compatibility style): dynamic offsets to textureGatherOffset
 * etc. are compile errors in ES but legal in desktop GL
 * (dEQP-GL45-ES31 functional.debug.negative_coverage.callbacks.shader_function).
 */
#include "mgl_test.h"

static const char *es_gather_dyn =
    "#version 310 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform mediump sampler2D s;\n"
    "uniform highp ivec2 o;\n"
    "out vec4 c;\n"
    "void main() { c = textureGatherOffset(s, vec2(0.5), o); }\n";
static const char *gl_gather_dyn =
    "#version 450 core\n"
    "uniform sampler2D s;\n"
    "uniform ivec2 o;\n"
    "out vec4 c;\n"
    "void main() { c = textureGatherOffset(s, vec2(0.5), o); }\n";
static const char *es_gather_const =
    "#version 310 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform mediump sampler2D s;\n"
    "out vec4 c;\n"
    "void main() { c = textureGatherOffset(s, vec2(0.5), ivec2(1)); }\n";

static GLuint compile(GLenum type, const char *src, char *log, GLsizei n)
{
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = -1;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (log) { glGetShaderInfoLog(sh, n, NULL, log); log[n-1] = 0; }
    glDeleteShader(sh);
    while (glGetError() != GL_NO_ERROR) {}
    return (GLuint)ok;
}

static const char *es_fsq_noprec =
    "#version 310 es\n"
    "uniform sampler2D s;\n"          /* ES: sampler precision must be declared */
    "out vec4 c;\n"
    "void main() { c = texture(s, vec2(0.5)); }\n";

TEST(es_profile_is_es)
{
    char log[2048];
    GLuint ok = compile(GL_FRAGMENT_SHADER, es_fsq_noprec, log, sizeof log);
    printf("  es FS missing sampler precision compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);      /* proves the profile really is ES */
}

TEST(es_constant_offset_rules)
{
    char log[2048];
    GLuint ok = compile(GL_VERTEX_SHADER, es_gather_dyn, log, sizeof log);
    printf("  es dynamic gather offset compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_FALSE);      /* ES 3.1: offset must be a constant */

    ok = compile(GL_VERTEX_SHADER, es_gather_const, log, sizeof log);
    printf("  es const gather offset compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_TRUE);

    ok = compile(GL_VERTEX_SHADER, gl_gather_dyn, log, sizeof log);
    printf("  gl dynamic gather offset compile=%u log=%s", ok, log);
    EXPECT_EQ_INT(ok, GL_TRUE);       /* desktop GL: dynamic offsets are fine */
}

int main(void)
{
    RUN(es_profile_is_es);
    RUN(es_constant_offset_rules);
    return mgl_test_summary();
}
