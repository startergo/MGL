/* geom r87: separate_shader.pipeline.different_uniform_separate_programs — glslang
 * I/O mapping of the monolithic twin fails when the same sources were first linked
 * as single-stage programs (each io-mapping assigned its own uniform location 0). */
#include "mgl_test.h"
#include <stdbool.h>
#include <string.h>

static const char *vsdu = "#version 310 es\n"
"uniform highp float vtxScale;\n"
"const vec2 triangle[3] = vec2[3](\n"
"	vec2( 0.2500,  0.8750),\n"
"	vec2(-0.6250,  0.7500),\n"
"	vec2( 0.8125, -0.6250)\n"
");\n"
"void main (void)\n"
"{\n"
"	gl_Position = vec4(vtxScale * triangle[gl_VertexID], 0.0, 1.0);\n"
"}\n";

static const char *fsdu = "#version 310 es\n"
"precision highp float;\n"
"out vec4 fragColor;\n"
"uniform highp float frgScale;\n"
"int imod (int n, int d)\n"
"{\n"
"	return (n < 0 ? d - 1 - (-1 - n) % d : n % d);\n"
"}\n"
"vec4 hsv (vec3 hsv){\n"
"	float h = hsv.x * 3.0;\n"
"	float r = max(0.0, 1.0 - h) + max(0.0, h - 2.0);\n"
"	float g = max(0.0, 1.0 - abs(h - 1.0));\n"
"	float b = max(0.0, 1.0 - abs(h - 2.0));\n"
"	vec3 hs = mix(vec3(1.0), vec3(r, g, b), hsv.y);\n"
"	return vec4(hsv.z * hs, 1.0);\n"
"}\n"
"void main (void)\n"
"{\n"
"	fragColor = vec4(vec3(frgScale), 1.0);\n"
"}\n";

static GLuint compile1(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok = GL_FALSE;
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    return sh;
}

static GLuint link_stage(GLenum type, const char *src)
{
    GLuint sh = compile1(type, src);
    GLuint p = glCreateProgram();
    GLint ok = GL_FALSE;
    char log[4096]; GLsizei len = 0;
    glAttachShader(p, sh);
    glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "stage program link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    return p;
}

TEST(regress_sep_single_stage_then_mono)
{
    GLint ok = GL_FALSE;
    char log[4096]; GLsizei len = 0;
    GLuint mono, vs, fs;

    /* exactly the CTS order: single-stage programs of each source first */
    GLuint vprog = link_stage(GL_VERTEX_SHADER, vsdu);
    GLuint fprog = link_stage(GL_FRAGMENT_SHADER, fsdu);
    EXPECT_NO_GL_ERROR();

    /* then the monolithic twin from the same sources */
    vs = compile1(GL_VERTEX_SHADER, vsdu);
    fs = compile1(GL_FRAGMENT_SHADER, fsdu);
    mono = glCreateProgram();
    glAttachShader(mono, vs);
    glAttachShader(mono, fs);
    glLinkProgram(mono);
    glGetProgramiv(mono, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(mono, sizeof log, &len, log);
        fprintf(stderr, "monolithic link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(mono); glDeleteProgram(vprog); glDeleteProgram(fprog);
    glDeleteShader(vs); glDeleteShader(fs);
}

int main(void)
{
    RUN(regress_sep_single_stage_then_mono);
    return mgl_test_summary();
}
