/* test_readbuf.c — glReadBuffer against the READ framebuffer binding (glmark2
 * scene-gl41 blit path reported INVALID_OPERATION) */
#include "mgl_test.h"

TEST(read_buffer_on_fbo)
{
    GLuint fbo = 0, tex = 0;
    GLint prev = -1;

    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* the glmark2 path: READ_FRAMEBUFFER bound, read buffer = COLOR_ATTACHMENT0 */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_NO_GL_ERROR();

    /* restoring to the default framebuffer: GL_BACK must be accepted */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glReadBuffer(GL_BACK);
    EXPECT_NO_GL_ERROR();
    glGetIntegerv(GL_READ_BUFFER, &prev);
    EXPECT_EQ_INT(prev, GL_BACK);

    /* GL_NONE is legal on the default framebuffer too */
    glReadBuffer(GL_NONE);
    EXPECT_NO_GL_ERROR();

    /* COLOR_ATTACHMENT0 while the DEFAULT framebuffer is the read target:
     * INVALID_OPERATION per table 17.4 */
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

int main(void)
{
    RUN(read_buffer_on_fbo);
    return mgl_test_summary();
}
