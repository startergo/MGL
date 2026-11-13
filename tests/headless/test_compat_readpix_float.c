/* test_compat_readpix_float — glReadPixels compatibility rules for FLOAT
 * attachments (compat r104/r108):
 *
 *   - an INTEGER attachment still requires an integer FORMAT (GL 4.6 8.2:
 *     the client format must be integer exactly when the buffer is);
 *   - a FLOAT-component attachment accepts EVERY table-valid (format,type)
 *     pair — desktop GL converts, including the integer-typed readbacks
 *     (GL_RED/GL_INT of R32F etc.).  compat r104 briefly rejected the
 *     integer-typed pairs per GLES-3.0 table 3.2 (what the ES31 negcov
 *     read_pixels_fbo_format_mismatch case wants when GL_EXT_color_buffer_
 *     float is exposed), but KHR-GL46.packed_pixels (27 cases) requires the
 *     opposite on the very same calls — with the string exposed (user
 *     decision) the two packages contradict; r108 restored the permissive
 *     desktop behaviour.  The rule keeps FLOAT-typed and UBYTE reads legal
 *     and verifies a real conversion VALUE.
 */
#include "mgl_test.h"

static GLuint fbo_with(GLint ifmt, GLenum cfmt, GLenum ctype)
{
    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, 8, 8, 0, cfmt, ctype, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

TEST(readpix_float_attachment_rules)
{
    GLuint f32 = fbo_with(GL_RGBA32F, GL_RGBA, GL_FLOAT);
    GLuint f16 = fbo_with(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    GLuint i32 = fbo_with(GL_RGBA32I, GL_RGBA_INTEGER, GL_INT);
    float out[4];
    GLubyte ub[4];
    GLint iv[4];

    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* store a known value in the float FBO: clear 0.25 */
    glBindFramebuffer(GL_FRAMEBUFFER, f32);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /* float-typed, byte-typed AND integer-typed reads all succeed and convert
     * (r108 permissive rule; packed_pixels relies on every pair) */
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, out);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(out[0], 0.25f, 1e-6);
    glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, out);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(out[0], 0.25f, 1e-6);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ub);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(ub[0], 64, 1);          /* 0.25 */
    for (int k = 0; k < 5; k++)
    {
        static const GLenum types[5] = { GL_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT };
        while (glGetError()) {}
        glReadPixels(0, 0, 1, 1, GL_RGBA, types[k], out);
        EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    }
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RED, GL_INT, iv);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* same for half-float */
    glBindFramebuffer(GL_FRAMEBUFFER, f16);
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RED, GL_INT, out);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* INTEGER attachment keeps the format rule: non-integer format errors,
     * RGBA_INTEGER works */
    glBindFramebuffer(GL_FRAMEBUFFER, i32);
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT, (GLuint *) out);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, (GLuint *) out);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &f32);
    glDeleteFramebuffers(1, &f16);
    glDeleteFramebuffers(1, &i32);
}

int main(void)
{
    RUN(readpix_float_attachment_rules);
    return mgl_test_summary();
}
