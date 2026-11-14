/* test_compat_unused_ms_sampler — r109 Task A regression (compat r107 fallout):
 * KHR-GL46.es_31_compatibility.sample_variables samples_0 cases fail AFTER a
 * preceding case + the CTS state reset.  Mechanism: the reset legally empties
 * the DEFAULT 2D_MULTISAMPLE object (glTexImage2DMultisample(..., 0, 0)), the
 * next case's shader declares an UNUSED `uniform sampler2DMS` whose unit holds
 * that level-less default object, and the draw must still render normally
 * (the unused sampler's slot gets the incomplete fallback).
 */
#include "mgl_test.h"

static GLuint mkprog(void)
{
    const char *vs = "#version 450\n"
                     "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                     "void main(){ gl_Position = vec4(q[gl_VertexID], 0, 1); }\n";
    const char *fs = "#version 450\n"
                     "layout(location=0) out vec4 o;\n"
                     "uniform sampler2DMS u_ms;      /* declared, never executed */\n"
                     "uniform sampler2D  u_tex;\n"
                     "void main(){ o = texture(u_tex, vec2(0.5)) + vec4(0, 0, texelFetch(u_ms, ivec2(0), 0).r, 0); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    char log[512];
    GLint ok = 0;
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }
    return p;
}

static int covered(GLuint fbo)
{
    GLubyte px[4] = { 0 };
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return px[1] > 128;    /* green channel */
}

TEST(unused_ms_sampler_after_reset)
{
    /* the state-reset sequence that precedes every CTS case */
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    while (glGetError()) {}
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA8, 0, 0, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0);
    while (glGetError()) {}
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 1, GL_RGBA8, 0, 0, 0, GL_TRUE);
    while (glGetError()) {}

    /* a real texture on unit 0 (the used sampler), the emptied default MS
     * object stays on unit 1 (the unused sampler2DMS) */
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    static const GLubyte green[4 * 4 * 4] = { 0 };
    /* all green */
    GLubyte g[4 * 4 * 4];
    for (int i = 0; i < 16; i++) { g[i * 4 + 0] = 0; g[i * 4 + 1] = 255; g[i * 4 + 2] = 0; g[i * 4 + 3] = 255; }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, g);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    GLuint prog = mkprog();
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "u_tex"), 0);
    glUniform1i(glGetUniformLocation(prog, "u_ms"), 1);
    EXPECT_TRUE(glGetUniformLocation(prog, "u_ms") >= 0);   /* the MS sampler exists */

    GLuint vao, fbo, rb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 4, 4);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);   /* the default object (still whole) */
    glActiveTexture(GL_TEXTURE0);

    /* case-1 draw: the default MS object is still level-less-but-fresh */
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(covered(fbo));

    /* the between-cases reset empties the default MS objects (r107 semantics) */
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    while (glGetError()) {}
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 1, GL_RGBA8, 0, 0, GL_TRUE);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 0);
    while (glGetError()) {}
    glTexImage3DMultisample(GL_TEXTURE_2D_MULTISAMPLE_ARRAY, 1, GL_RGBA8, 0, 0, 0, GL_TRUE);
    while (glGetError()) {}
    glActiveTexture(GL_TEXTURE0);
    glClear(GL_COLOR_BUFFER_BIT);

    /* case-2 draw: same program, same state — must render identically */
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(covered(fbo));                     /* RED pre-fix: nothing renders */

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    glDeleteProgram(prog);
    glDeleteTextures(1, &tex);
    (void)green;
}
int main(void){ RUN(unused_ms_sampler_after_reset); return mgl_test_summary(); }
