/* r75 RED/GREEN: GL 4.6 table 8.11 — the STENCIL_INDEX client format pairs
 * only with the stencil-only internal formats.  The combined depth-stencil
 * internal formats (DEPTH_STENCIL, DEPTH24_STENCIL8, DEPTH32F_STENCIL8) take
 * DEPTH_STENCIL client data only; STENCIL_INDEX data must be rejected with
 * GL_INVALID_OPERATION (KHR-GL46.packed_pixels.*_format_stencil_index).
 * Regressed when the r46/r47 STENCIL_INDEX8 texture support made the combo
 * reachable (first-bad 8554b98). */
#include "mgl_test.h"
#include <string.h>

TEST(PackedPixelsStencilIndexRejected)
{
    GLuint t;
    static const unsigned char data[16] = {0};
    const GLint ds_ifmts[3] = { GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_DEPTH32F_STENCIL8 };

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();   /* baseline sanity */

    /* the rejection: every combined depth-stencil internal format must refuse
     * STENCIL_INDEX client data */
    for (int i = 0; i < 3; i++)
    {
        while (glGetError() != GL_NO_ERROR) {}
        glTexImage2D(GL_TEXTURE_2D, 0, ds_ifmts[i], 4, 4, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, data);
        EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
        while (glGetError() != GL_NO_ERROR) {}
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, ds_ifmts[i], 4, 4, 2, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, data);
        EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    }

    /* the legal neighbours stay legal: STENCIL_INDEX8 + STENCIL_INDEX data
     * (UNSIGNED_BYTE) and the depth-stencil internals + DEPTH_STENCIL data */
    while (glGetError() != GL_NO_ERROR) {}
    glTexImage2D(GL_TEXTURE_2D, 0, GL_STENCIL_INDEX8, 4, 4, 0, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();
    while (glGetError() != GL_NO_ERROR) {}
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 4, 4, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, data);
    EXPECT_NO_GL_ERROR();
    while (glGetError() != GL_NO_ERROR) {}
    glDeleteTextures(1, &t);
}

int main(void) { RUN(PackedPixelsStencilIndexRejected); return mgl_test_summary(); }
