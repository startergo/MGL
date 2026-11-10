/* test_compat_readpix_float — glReadPixels incompatibility rule for
 * FLOAT-component attachments (compat r104):
 *
 *   reading an attachment whose component type is GL_FLOAT (R16F/R32F/RGBA32F/
 *   10F_11F_11F) with a plain integer-typed readback (GL_BYTE/GL_SHORT/
 *   GL_INT/... and a non-integer format) is GL_INVALID_OPERATION — GLES 3.0
 *   table 3.2, enforced by dEQP-GLES31 negative_coverage
 *   get_error.buffer.read_pixels_fbo_format_mismatch, whose float sections
 *   became active once GL_EXT_color_buffer_float was advertised.
 *
 *   FLOAT-typed reads (GL_RED/GL_FLOAT, GL_RGBA/GL_HALF_FLOAT) and the classic
 *   GL_RGBA/GL_UNSIGNED_BYTE conversion stay legal (KHR-GL46 packed_pixels
 *   reads GL_RED/GL_FLOAT of R32F; fbo tests read UBYTE).
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

TEST(readpix_float_attachment_integer_type)
{
    GLuint f32 = fbo_with(GL_RGBA32F, GL_RGBA, GL_FLOAT);
    GLuint f16 = fbo_with(GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT);
    GLuint i32 = fbo_with(GL_RGBA32I, GL_RGBA_INTEGER, GL_INT);
    float out[4];
    GLubyte ub[4];
    GLint e;

    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* integer-typed reads of float attachments: INVALID_OPERATION */
    glBindFramebuffer(GL_FRAMEBUFFER, f32);
    for (int k = 0; k < 5; k++)
    {
        static const GLenum types[5] = { GL_BYTE, GL_SHORT, GL_UNSIGNED_SHORT, GL_INT, GL_UNSIGNED_INT };
        while (glGetError()) {}
        glReadPixels(0, 0, 1, 1, GL_RGBA, types[k], out);
        e = glGetError();
        EXPECT_EQ_INT(e, GL_INVALID_OPERATION);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, f16);
    glReadPixels(0, 0, 1, 1, GL_RED, GL_INT, out);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);

    /* float-typed and byte-typed reads stay legal */
    glBindFramebuffer(GL_FRAMEBUFFER, f32);
    while (glGetError()) {}
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, out);
    EXPECT_NO_GL_ERROR();
    glReadPixels(0, 0, 1, 1, GL_RED, GL_FLOAT, out);
    EXPECT_NO_GL_ERROR();
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ub);
    EXPECT_NO_GL_ERROR();
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_HALF_FLOAT, out);
    EXPECT_NO_GL_ERROR();

    /* integer attachment keeps the format rule: non-integer format errors,
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
    RUN(readpix_float_attachment_integer_type);
    return mgl_test_summary();
}
