/* geom r87: separate_shader.pipeline.different_uniform_separate_programs — the
 * monolithic equivalent of a VS(uniform vtxScale)+FS(uniform frgScale) pair with
 * NO varyings fails glslang I/O mapping ("ERROR: GLSL I/O mapping failed"). */
#include "mgl_test.h"
#include <stdbool.h>

static const char *vsdu = "#version 310 es\n"
"uniform highp float vtxScale;\n"
"void main (void)\n"
"{\n"
"	gl_Position = vec4(vtxScale * vec2(0.5, 0.5), 0.0, 1.0);\n"
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
static void link_and_check(const char *tag, bool separable)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p;
    char log[4096];
    GLsizei len = 0;
    GLint ok = GL_FALSE;

    glShaderSource(vs, 1, &vsdu, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    glShaderSource(fs, 1, &fsdu, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);

    p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    if (separable) glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "%s link FAILED (separable=%d): %.*s\n", tag, (int)separable, (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    glDeleteProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

/* the CTS shape: first the separable stage-only programs, then the monolithic
 * equivalent — the compile cache shared glslang intermediates across the links
 * and the monolithic link's I/O mapping failed with an empty glslang log */
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

TEST(regress_sep_monolithic_uniforms)
{
    link_and_check("different_uniforms", false);
    link_and_check("different_uniforms_separable", true);

    /* pipeline shape exactly as the CTS: glCreateShaderProgramv stage programs
     * (compile, link, detach, DELETE the shader), then the monolithic twin */
    GLint ok = GL_FALSE;
    char log[4096]; GLsizei len = 0;
    const char *vsrc = vsdu, *fsrc = fsdu;
    GLuint vprog = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &vsrc);
    GLuint fprog = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &fsrc);
    glGetProgramiv(vprog, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(vprog, sizeof log, &len, log);
        fprintf(stderr, "vprog link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    glGetProgramiv(fprog, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(fprog, sizeof log, &len, log);
        fprintf(stderr, "fprog link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);

    /* manual creation exactly as glu::ShaderProgram: shader objects are owned by
     * the wrapper and DELETED right after the link while still attached */
    GLuint vs1 = compile1(GL_VERTEX_SHADER, vsdu);
    GLuint fs1 = compile1(GL_FRAGMENT_SHADER, fsdu);
    GLuint vprog2 = glCreateProgram(), fprog2 = glCreateProgram();
    GLint ok2 = GL_FALSE;
    glAttachShader(vprog2, vs1);
    glProgramParameteri(vprog2, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(vprog2);
    glGetProgramiv(vprog2, GL_LINK_STATUS, &ok2);
    EXPECT_EQ_INT(ok2, GL_TRUE);
    glAttachShader(fprog2, fs1);
    glProgramParameteri(fprog2, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(fprog2);
    glGetProgramiv(fprog2, GL_LINK_STATUS, &ok2);
    EXPECT_EQ_INT(ok2, GL_TRUE);
    glDeleteShader(vs1);
    glDeleteShader(fs1);

    /* the CTS binds the pipeline and sets both programs' uniforms through
     * glUseProgram BEFORE creating the monolithic twin */
    GLuint ppl = 0;
    glGenProgramPipelines(1, &ppl);
    glUseProgramStages(ppl, GL_VERTEX_SHADER_BIT, vprog);
    glUseProgramStages(ppl, GL_FRAGMENT_SHADER_BIT, fprog);
    glBindProgramPipeline(ppl);
    EXPECT_NO_GL_ERROR();
    glUseProgram(vprog);
    glUniform1f(glGetUniformLocation(vprog, "vtxScale"), 0.9375f);
    glUseProgram(fprog);
    glUniform1f(glGetUniformLocation(fprog, "frgScale"), 1.0f);
    glUseProgram(0);
    EXPECT_NO_GL_ERROR();

    /* the monolithic twin, same sources, separable = false */
    GLuint vs2 = compile1(GL_VERTEX_SHADER, vsdu);
    GLuint fs2 = compile1(GL_FRAGMENT_SHADER, fsdu);
    GLuint mono = glCreateProgram();
    glAttachShader(mono, vs2);
    glAttachShader(mono, fs2);
    glLinkProgram(mono);
    glGetProgramiv(mono, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(mono, sizeof log, &len, log);
        fprintf(stderr, "mono link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);

    glDeleteProgram(mono); glDeleteProgram(vprog); glDeleteProgram(fprog);
    glDeleteProgram(vprog2); glDeleteProgram(fprog2);
    glDeleteProgramPipelines(1, &ppl);
    glDeleteShader(vs2); glDeleteShader(fs2);
}

int main(void)
{
    RUN(regress_sep_monolithic_uniforms);
    return mgl_test_summary();
}
