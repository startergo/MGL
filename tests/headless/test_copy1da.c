/*
 * test_copy1da.c — glCopyImageSubData bounds on GL_TEXTURE_1D_ARRAY: Y addresses
 * the single row of the 1-high image (must stay within row 0) and Z selects the
 * layer (bounded by the layer count).  Regression for
 * KHR-GL46.copy_image.exceeding_boundaries.
 */
#include "mgl_test.h"
#include <string.h>

static GLuint mk1da(GLubyte fill)
{
    GLuint t;
    GLubyte v[16 * 16];
    memset(v, fill, sizeof v);
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_1D_ARRAY, t);
    /* 16 texels x 16 layers; complete without mipmaps */
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_R8, 16, 16, 0, GL_RED, GL_UNSIGNED_BYTE, v);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    return t;
}

TEST(copy_1darray_bounds)
{
    GLuint a = mk1da(0x11), b = mk1da(0x22);

    /* valid: one layer, row 0 */
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 3, b, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 3, 4, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    /* y beyond the single row -> GL_INVALID_VALUE */
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 0, 14, 0, b, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 0, 4, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 0, b, GL_TEXTURE_1D_ARRAY, 0, 0, 14, 0, 4, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);

    /* z beyond the layer count -> GL_INVALID_VALUE */
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 16, b, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 0, 4, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 14, b, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 0, 4, 1, 4);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);

    /* x beyond the texel row */
    glCopyImageSubData(a, GL_TEXTURE_1D_ARRAY, 0, 14, 0, 0, b, GL_TEXTURE_1D_ARRAY, 0, 0, 0, 0, 4, 1, 1);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);

    /* the valid copy landed: layer 0 of b keeps its original fill */
    GLubyte px[16 * 16];
    memset(px, 0, sizeof px);
    glGetTexImage(GL_TEXTURE_1D_ARRAY, 0, GL_RED, GL_UNSIGNED_BYTE, px);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_EQ_INT(px[0], 0x22);
    EXPECT_EQ_INT(px[3 * 16], 0x11);   /* layer 3 received a's fill */

    glDeleteTextures(1, &a);
    glDeleteTextures(1, &b);
}

int main(void)
{
    RUN(copy_1darray_bounds);
    return mgl_test_summary();
}
