/* Supervisor r75 follow-up RED/GREEN: STENCIL_INDEX is invalid for a texture
 * without a stencil plane for EVERY op, readback included.  r75 limited the
 * STENCIL_INDEX rejection to TexImage/SubImage, so glGetTexImage of an R16F /
 * R32F texture with STENCIL_INDEX stopped raising GL_INVALID_OPERATION and the
 * 27 KHR-GL46.packed_pixels float-format cases failed.  The combined
 * depth-stencil readback leniency (stencil plane of DEPTH24_STENCIL8 via
 * STENCIL_INDEX) must stay legal. */
#include "mgl_test.h"
#include <string.h>

TEST(StencilIndexReadOfNonStencilTextureRejected)
{
    GLuint t;
    static float data[64];
    static unsigned char out[256];
    const GLint ifmts[3] = { GL_R16F, GL_R32F, GL_RGBA8 };
    const GLenum fmts[3] = { GL_RED, GL_RED, GL_RGBA };
    const GLenum types[3] = { GL_FLOAT, GL_FLOAT, GL_UNSIGNED_BYTE };

    memset(data, 0, sizeof data);
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    for (int i = 0; i < 3; i++)
    {
        while (glGetError() != GL_NO_ERROR) {}
        glTexImage2D(GL_TEXTURE_2D, 0, ifmts[i], 2, 2, 0, fmts[i], types[i], data);
        EXPECT_NO_GL_ERROR();
        while (glGetError() != GL_NO_ERROR) {}
        glGetTexImage(GL_TEXTURE_2D, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, out);
        EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    }

    /* the leniency r75 kept: the stencil plane of a combined depth-stencil
     * texture reads back as STENCIL_INDEX */
    while (glGetError() != GL_NO_ERROR) {}
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 2, 2, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, data);
    EXPECT_NO_GL_ERROR();
    while (glGetError() != GL_NO_ERROR) {}
    glGetTexImage(GL_TEXTURE_2D, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, out);
    EXPECT_NO_GL_ERROR();
    while (glGetError() != GL_NO_ERROR) {}
    glDeleteTextures(1, &t);
}

int main(void) { RUN(StencilIndexReadOfNonStencilTextureRejected); return mgl_test_summary(); }
