/* test_compat_pipeline_no_ffp — compat-as-superset audit regression
 * (compat r106): on a compatibility-profile context, a draw with
 * glUseProgram(0) + a bound PROGRAM PIPELINE must run the pipeline's stages.
 * The compat array-draw wrapper (bind_ffp) used to install the generated
 * fixed-function program whenever no user program was current, so
 * KHR-GL46.direct_state_access.program_pipelines_functional rendered nothing
 * (black) on a compat context while passing on core.
 */
#include "mgl_test.h"
#include "mgl_compat.h"

static GLuint mksep(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    char log[512];
    GLint ok = 0;
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(s, sizeof log, NULL, log); MGL_FAIL("compile: %s", log); }
    glProgramParameteri(s, 0, GL_PROGRAM_SEPARABLE);   /* name ignored for shaders */
    return s;
}

TEST(pipeline_draws_on_compat_context)
{
    void *cc = mglCompatCreateContext();
    EXPECT_TRUE(cc != NULL);
    EXPECT_TRUE(mglCurrentContextIsCompat());

    /* separable VS/FS pair */
    GLuint vs = glCreateShaderProgramv(GL_VERTEX_SHADER, 1,
        (const char *const[]){ "#version 450 core\n"
                               "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                               "void main(){ gl_Position = vec4(q[gl_VertexID], 0, 1); }" });
    GLuint fs = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1,
        (const char *const[]){ "#version 450 core\n"
                               "layout(location=0) out vec4 o;\n"
                               "void main(){ o = vec4(0, 0, 1, 1); }" });
    EXPECT_TRUE(vs != 0);
    EXPECT_TRUE(fs != 0);

    GLuint pipe = 0;
    glGenProgramPipelines(1, &pipe);
    glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, vs);
    glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, fs);
    EXPECT_NO_GL_ERROR();

    GLuint vao, fbo, rb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 32, 32);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 32, 32);

    glUseProgram(0);
    glBindProgramPipeline(pipe);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);       /* full-screen quad via gl_VertexID */
    EXPECT_NO_GL_ERROR();

    GLubyte px[4] = { 0, 0, 0, 0 };
    glReadPixels(16, 16, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(px[2], 255);                  /* blue through the pipeline's FS */

    glBindProgramPipeline(0);
    glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(vs);
    glDeleteProgram(fs);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    (void)mksep;
}
int main(void){ RUN(pipeline_draws_on_compat_context); return mgl_test_summary(); }
