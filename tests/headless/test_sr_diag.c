/* subroutine diagnostics (round 19 scratch) */
#include "mgl_test.h"

static const char *vs =
    "#version 400\n"
    "#extension GL_ARB_shader_subroutine : require\n"
    "\n"
    "subroutine vec4 colorGetter(vec4 in_value);\n"
    "\n"
    "subroutine(colorGetter) vec4 getter0(vec4 in_value)\n"
    "{\n"
    "    return vec4(0.25, 0.5, 0.75, 1.0);\n"
    "}\n"
    "\n"
    "subroutine(colorGetter) vec4 getter1(vec4 in_value)\n"
    "{\n"
    "    return vec4(1.0, 0.5, 0.25, 1.0);\n"
    "}\n"
    "\n"
    "subroutine uniform colorGetter colorGetterUniform;\n"
    "\n"
    "out vec4 result;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "    result      = colorGetterUniform(vec4(1.0));\n"
    "}\n";

static const char *fs =
    "#version 400\n"
    "layout(location = 0) out vec4 out_color;\n"
    "void main() { out_color = vec4(1.0); }\n";

TEST(sr_plain_link)
{
    GLuint vsid = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vsid, 1, &vs, NULL);
    glCompileShader(vsid);
    GLint ok = 0; glGetShaderiv(vsid, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096] = ""; glGetShaderInfoLog(vsid, sizeof log, NULL, log);
               fprintf(stderr, "VS compile log:\n%s\n", log); }
    EXPECT_EQ_INT(ok, GL_TRUE);

    GLuint fsid = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fsid, 1, &fs, NULL);
    glCompileShader(fsid);
    glGetShaderiv(fsid, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096] = ""; glGetShaderInfoLog(fsid, sizeof log, NULL, log);
               fprintf(stderr, "FS compile log:\n%s\n", log); }
    EXPECT_EQ_INT(ok, GL_TRUE);

    GLuint p = glCreateProgram();
    glAttachShader(p, vsid); glAttachShader(p, fsid);
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (!GL_TRUE == lk) { char log[4096] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               fprintf(stderr, "plain link log:\n%s\n", log); }
    EXPECT_EQ_INT(lk, GL_TRUE);
    glDeleteProgram(p); glDeleteShader(vsid); glDeleteShader(fsid);
}

TEST(sr_xfb_link)
{
    GLuint vsid = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vsid, 1, &vs, NULL);
    glCompileShader(vsid);
    GLint ok = 0; glGetShaderiv(vsid, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);

    GLuint p = glCreateProgram();
    glAttachShader(p, vsid);
    const char *xfb[] = { "result" };
    glTransformFeedbackVaryings(p, 1, xfb, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    if (lk != GL_TRUE) { char log[4096] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               fprintf(stderr, "xfb link log:\n%s\n", log); }
    EXPECT_EQ_INT(lk, GL_TRUE);
    glDeleteProgram(p); glDeleteShader(vsid);
}

int main(void)
{
    RUN(sr_plain_link);
    RUN(sr_xfb_link);
    return mgl_test_summary();
}
