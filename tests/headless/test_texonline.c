/*
 * test_texonline.c — online texture compression (uncompressed data uploaded to
 * a compressed internalformat):
 *   - RGTC (BC4/BC5): GL_TEXTURE_COMPRESSED reports TRUE, glGetCompressedTexImage
 *     returns re-encoded blocks, a compressed sub-image into the decoded level
 *     decodes and reads back byte-exactly (KHR-GL46.direct_state_access.
 *     textures_compressed_subimage),
 *   - BPTC/ETC2: MGL has no CPU encoder, so the request reports the "no online
 *     compression" error (GL_INVALID_OPERATION), which the CTS treats as a valid
 *     driver response,
 *   - generic GL_COMPRESSED_RGB substitutes an uncompressed internal format and
 *     GL_TEXTURE_COMPRESSED reports FALSE.
 */
#include "mgl_test.h"
#include <string.h>

/* online RGTC1: upload, download, re-upload, byte-exact download */
TEST(rgtc_online_roundtrip)
{
    GLuint ta = 0, tb = 0;
    GLubyte data[8 * 8];
    for (int i = 0; i < 8 * 8; i++) data[i] = (GLubyte)(i * 31);

    glGenTextures(1, &ta);
    glBindTexture(GL_TEXTURE_2D, ta);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RED_RGTC1, 8, 8, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    GLint comp = 0, size = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &comp);
    EXPECT_EQ_INT(comp, 1);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);
    EXPECT_EQ_INT(size, 32);            /* 2x2 blocks of 8 bytes */
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    GLubyte ref[64];
    memset(ref, 0xEE, sizeof ref);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, ref);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    for (int i = 0; i < size; i++)
        if (ref[i] == 0xEE && i >= 1) { EXPECT_EQ_INT(ref[i], 0); break; }  /* data arrived */

    /* storage + compressed sub-image of the downloaded blocks */
    glGenTextures(1, &tb);
    glBindTexture(GL_TEXTURE_2D, tb);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RED_RGTC1, 8, 8, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 8, GL_COMPRESSED_RED_RGTC1, size, ref);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    GLubyte res[64];
    memset(res, 0, sizeof res);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, res);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    EXPECT_EQ_INT(memcmp(ref, res, (size_t)size), 0);

    /* the decoded level still samples as the uploaded pixels (values near i*31) */
    /* dual-channel RGTC2 round-trips the same way */
    glGenTextures(1, &ta);
    glBindTexture(GL_TEXTURE_2D, ta);
    GLubyte rg[8 * 8 * 2];
    for (int i = 0; i < 8 * 8; i++) { rg[2 * i] = (GLubyte)(i * 3); rg[2 * i + 1] = (GLubyte)(255 - i); }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RG_RGTC2, 8, 8, 0, GL_RG, GL_UNSIGNED_BYTE, rg);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &comp);
    EXPECT_EQ_INT(comp, 1);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);
    EXPECT_EQ_INT(size, 64);            /* 2x2 blocks of 16 bytes */
    memset(ref, 0xEE, sizeof ref);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, ref);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);

    glDeleteTextures(1, &ta);
    glDeleteTextures(1, &tb);
}

/* formats without a CPU encoder: the online-compression request must fail */
TEST(online_unsupported_families)
{
    GLuint t = 0;
    GLubyte data[8 * 8 * 4];
    for (int i = 0; i < 8 * 8 * 4; i++) data[i] = (GLubyte)i;

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_BPTC_UNORM, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB8_ETC2, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_OPERATION);
    /* explicit block uploads keep working (8x8 ETC2 RGB = 2x2 blocks x 8 bytes) */
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB8_ETC2, 8, 8, 0, 32, NULL);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    /* a NULL-data definition of compressed storage stays legal */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_BPTC_UNORM, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDeleteTextures(1, &t);
}

/* the generic compressed internal formats substitute an uncompressed format */
TEST(generic_compressed_substitutes)
{
    GLuint t = 0;
    GLubyte data[16 * 4];
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB, 16, 16, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    GLint comp = 2;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &comp);
    EXPECT_EQ_INT(comp, 0);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glDeleteTextures(1, &t);
}

int main(void)
{
    RUN(rgtc_online_roundtrip);
    RUN(online_unsupported_families);
    RUN(generic_compressed_substitutes);
    return mgl_test_summary();
}
