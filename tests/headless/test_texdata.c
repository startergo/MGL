/* test_texdata.c — texture image specification, pixel transfer, compressed
 * images, copies, clears and readback ('texdata' domain). */
#include "mgl_test.h"

#ifndef GL_COMPRESSED_RGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#endif

static GLuint new_tex(GLenum target)
{
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(target, t);
    return t;
}

TEST(rgba8_roundtrip)
{
    GLubyte src[4 * 4 * 4], dst[4 * 4 * 4];
    for (int i = 0; i < (int)sizeof src; i++) src[i] = (GLubyte)(i * 7 + 3);
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
    EXPECT_NO_GL_ERROR();
    memset(dst, 0, sizeof dst);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, dst);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(memcmp(src, dst, sizeof src) == 0);
    /* BGRA readback swaps red and blue */
    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, dst);
    EXPECT_EQ_INT(dst[0], src[2]);
    EXPECT_EQ_INT(dst[2], src[0]);
    EXPECT_EQ_INT(dst[3], src[3]);
    glDeleteTextures(1, &t);
}

/* upload bytes in (format,type), read back in the same (format,type) */
static int roundtrip(GLenum ifmt, GLenum format, GLenum type, const void *data, size_t bytes, int w, int h)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    unsigned char out[256];
    memset(out, 0xcd, sizeof out);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, format, type, data);
    GLenum e1 = glGetError();
    glGetTexImage(GL_TEXTURE_2D, 0, format, type, out);
    GLenum e2 = glGetError();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glDeleteTextures(1, &t);
    if (e1 || e2) { fprintf(stderr, "  [roundtrip] 0x%x: errors 0x%x 0x%x\n", ifmt, e1, e2); return 0; }
    if (memcmp(out, data, bytes)) {
        fprintf(stderr, "  [roundtrip] 0x%x mismatch:", ifmt);
        for (size_t i = 0; i < bytes && i < 16; i++) fprintf(stderr, " %02x/%02x", ((const unsigned char *)data)[i], out[i]);
        fprintf(stderr, "\n");
        return 0;
    }
    return 1;
}

TEST(format_roundtrips)
{
    GLubyte r8[3] = { 1, 128, 255 };
    EXPECT_TRUE(roundtrip(GL_R8, GL_RED, GL_UNSIGNED_BYTE, r8, 3, 3, 1));
    GLubyte rg8[6] = { 1, 2, 3, 4, 5, 6 };
    EXPECT_TRUE(roundtrip(GL_RG8, GL_RG, GL_UNSIGNED_BYTE, rg8, 6, 3, 1));
    GLubyte srgb[8] = { 10, 20, 30, 40, 200, 210, 220, 230 };
    EXPECT_TRUE(roundtrip(GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, srgb, 8, 2, 1));
    GLushort h16[2] = { 0x3c00, 0xc000 }; /* 1.0, -2.0 */
    EXPECT_TRUE(roundtrip(GL_R16F, GL_RED, GL_HALF_FLOAT, h16, 4, 2, 1));
    GLfloat rg32[4] = { 1.5f, -3.25f, 1e6f, 0.125f };
    EXPECT_TRUE(roundtrip(GL_RG32F, GL_RG, GL_FLOAT, rg32, 16, 2, 1));
    GLuint u32[4] = { 1, 0xffffffffu, 12345678, 7 };
    EXPECT_TRUE(roundtrip(GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, u32, 16, 1, 1));
    GLbyte i8[2] = { -128, 127 };
    EXPECT_TRUE(roundtrip(GL_R8I, GL_RED_INTEGER, GL_BYTE, i8, 2, 2, 1));
    GLushort u16[4] = { 0, 1000, 65535, 32768 };
    EXPECT_TRUE(roundtrip(GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT, u16, 8, 1, 1));
    GLuint a2[2] = { 0xC0000000u | (1023u << 20) | (512u << 10) | 1u, 0x40000000u | 77u };
    EXPECT_TRUE(roundtrip(GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, a2, 8, 2, 1));
    GLuint uf[1] = { 0x3c0u | (0x3c0u << 11) | (0x1e0u << 22) }; /* 1.0, 1.0, 1.0 */
    EXPECT_TRUE(roundtrip(GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, uf, 4, 1, 1));
    GLuint e5[1] = { 256u | (128u << 9) | (64u << 18) | (16u << 27) };
    EXPECT_TRUE(roundtrip(GL_RGB9_E5, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, e5, 4, 1, 1));
    GLfloat d32[3] = { 0.0f, 0.5f, 1.0f };
    EXPECT_TRUE(roundtrip(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, d32, 12, 3, 1));
    GLuint ds[2] = { (0x800000u << 8) | 0x5a, (0xffffffu << 8) | 0x01 };
    EXPECT_TRUE(roundtrip(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, ds, 8, 2, 1));
    EXPECT_NO_GL_ERROR();
}

TEST(format_conversions)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    GLubyte px[4] = { 0, 51, 102, 255 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    GLfloat f[4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, f);
    EXPECT_NEAR(f[1], 0.2, 1e-6);
    EXPECT_NEAR(f[2], 0.4, 1e-6);
    EXPECT_NEAR(f[3], 1.0, 1e-6);

    /* RGB8: alpha reads back as 1, green-only read */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLubyte[]){ 9, 8, 7, 6 });
    GLubyte o[4] = { 0 };
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0], 9);
    EXPECT_EQ_INT(o[3], 255);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_GREEN, GL_UNSIGNED_BYTE, o);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    EXPECT_EQ_INT(o[0], 8);

    /* packed 5_6_5 into RGB8 */
    GLushort p565 = (31u << 11) | (0u << 5) | 31u;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 1, 1, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &p565);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0], 255);
    EXPECT_EQ_INT(o[1], 0);
    EXPECT_EQ_INT(o[2], 255);
    GLushort back[2] = { 0, 0 };
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, back);
    EXPECT_EQ_INT(back[0], p565);

    /* float into R8 clamps */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 2, 1, 0, GL_RED, GL_FLOAT, (GLfloat[]){ -1.0f, 2.0f });
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, o);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    EXPECT_EQ_INT(o[0], 0);
    EXPECT_EQ_INT(o[1], 255);

    /* swap bytes */
    GLushort s[2] = { 0x1234, 0xabcd }, so[2] = { 0, 0 };
    glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, 2, 1, 0, GL_RED, GL_UNSIGNED_SHORT, s);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_SHORT, so);
    EXPECT_EQ_INT(so[0], 0x3412);
    EXPECT_EQ_INT(so[1], 0xcdab);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(unpack_pack_state)
{
    /* 5x4 RGB source; upload the 2x2 block at (1,1) with ROW_LENGTH/SKIP and alignment 1 */
    GLubyte src[4][5][3];
    for (int y = 0; y < 4; y++) for (int x = 0; x < 5; x++) { src[y][x][0] = x; src[y][x][1] = y; src[y][x][2] = 100; }
    GLuint t = new_tex(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 5);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 1);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, src);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    EXPECT_NO_GL_ERROR();

    GLubyte o[2][2][4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0][0][0], 1);
    EXPECT_EQ_INT(o[0][0][1], 1);
    EXPECT_EQ_INT(o[1][1][0], 2);
    EXPECT_EQ_INT(o[1][1][1], 2);
    EXPECT_EQ_INT(o[1][1][3], 255);

    /* default alignment 4 with 3-byte rows: row stride 8 for a 2-pixel RGB row */
    GLubyte padded[2][8];
    memset(padded, 0, sizeof padded);
    padded[0][0] = 11; padded[0][3] = 12; padded[1][0] = 21; padded[1][3] = 22;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, padded);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0][1][0], 12);
    EXPECT_EQ_INT(o[1][0][0], 21);
    EXPECT_EQ_INT(o[1][1][0], 22);

    /* PACK_ROW_LENGTH + PACK_SKIP_PIXELS on readback */
    GLubyte big[2][4][4];
    memset(big, 0, sizeof big);
    glPixelStorei(GL_PACK_ROW_LENGTH, 4);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, big);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    EXPECT_EQ_INT(big[0][0][0], 0);
    EXPECT_EQ_INT(big[0][1][0], 11);
    EXPECT_EQ_INT(big[1][2][0], 22);

    GLint v = 0;
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 7);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &v);
    EXPECT_EQ_INT(v, 7);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(subimage_after_gpu)
{
    GLubyte red[16 * 4], o[16 * 4];
    for (int i = 0; i < 16; i++) { red[i * 4] = 255; red[i * 4 + 1] = 0; red[i * 4 + 2] = 0; red[i * 4 + 3] = 255; }
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, red);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0], 255);
    GLubyte green[4] = { 0, 255, 0, 255 };
    glTexSubImage2D(GL_TEXTURE_2D, 0, 2, 3, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, green);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[(3 * 4 + 2) * 4 + 1], 255);
    EXPECT_EQ_INT(o[(3 * 4 + 2) * 4 + 0], 0);
    EXPECT_EQ_INT(o[(3 * 4 + 1) * 4 + 0], 255);
    /* TextureSubImage2D (DSA) */
    glTextureSubImage2D(t, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, green);
    GLubyte one[4];
    glGetTextureSubImage(t, 0, 0, 0, 0, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, sizeof one, one);
    EXPECT_EQ_INT(one[1], 255);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(mip_levels_and_generate)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    GLubyte l0[4 * 4 * 4], l1[2 * 2 * 4], l2[4] = { 1, 2, 3, 4 }, o[64];
    memset(l0, 200, sizeof l0);
    memset(l1, 100, sizeof l1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, l0);
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, l1);
    glTexImage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, l2);
    EXPECT_NO_GL_ERROR();
    glGetTexImage(GL_TEXTURE_2D, 1, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0], 100);
    glGetTexImage(GL_TEXTURE_2D, 2, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[3], 4);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[60], 200);

    /* GenerateMipmap from a uniform base gives the same colour at every level */
    glGenerateMipmap(GL_TEXTURE_2D);
    EXPECT_NO_GL_ERROR();
    glGetTexImage(GL_TEXTURE_2D, 2, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_NEAR(o[0], 200, 1);
    GLint w = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 2);

    /* TexStorage with a full chain, levels > 0 usable */
    GLuint t2 = new_tex(GL_TEXTURE_2D);
    glTexStorage2D(GL_TEXTURE_2D, 3, GL_RGBA8, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 2, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, l2);
    glGetTexImage(GL_TEXTURE_2D, 2, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[2], 3);
    glDeleteTextures(1, &t2);

    /* integer textures cannot be mipmapped */
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, 4, 4, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, l0);
    glGenerateMipmap(GL_TEXTURE_2D);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGenerateMipmap(GL_TEXTURE_RECTANGLE);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDeleteTextures(1, &t);
}

TEST(cube_map_faces)
{
    GLuint t = new_tex(GL_TEXTURE_CUBE_MAP);
    for (int f = 0; f < 6; f++)
    {
        GLubyte px[4 * 4];
        for (int i = 0; i < 4; i++) { px[i * 4] = 10 * (f + 1); px[i * 4 + 1] = f; px[i * 4 + 2] = 0; px[i * 4 + 3] = 255; }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    }
    EXPECT_NO_GL_ERROR();
    for (int f = 0; f < 6; f++)
    {
        GLubyte o[16];
        glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
        EXPECT_EQ_INT(o[0], 10 * (f + 1));
        EXPECT_EQ_INT(o[13], f);
    }
    /* DSA read of the whole cube: six consecutive faces */
    GLubyte all[6 * 16];
    glGetTextureImage(t, 0, GL_RGBA, GL_UNSIGNED_BYTE, sizeof all, all);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(all[5 * 16], 60);
    EXPECT_EQ_INT(all[2 * 16 + 1], 2);
    /* non-square face is an error */
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA8, 2, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    /* GetTexImage on CUBE_MAP itself is invalid */
    glGetTexImage(GL_TEXTURE_CUBE_MAP, 0, GL_RGBA, GL_UNSIGNED_BYTE, all);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDeleteTextures(1, &t);
}

TEST(arrays_and_3d)
{
    GLuint t = new_tex(GL_TEXTURE_2D_ARRAY);
    GLubyte img[3][2][2][4];
    for (int l = 0; l < 3; l++) for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++)
    { img[l][y][x][0] = l * 50; img[l][y][x][1] = y; img[l][y][x][2] = x; img[l][y][x][3] = 255; }
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 2, 2, 3, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    GLubyte blue[4] = { 0, 0, 255, 255 };
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 1, 0, 1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, blue);
    EXPECT_NO_GL_ERROR();
    GLubyte o[3][2][2][4];
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[2][0][0][0], 100);
    EXPECT_EQ_INT(o[1][0][1][2], 255);
    EXPECT_EQ_INT(o[1][0][0][0], 50);
    GLubyte one[4];
    glGetTextureSubImage(t, 0, 1, 1, 2, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 4, one);
    EXPECT_EQ_INT(one[0], 100);
    EXPECT_EQ_INT(one[1], 1);
    glDeleteTextures(1, &t);

    /* 1D array: rows are layers */
    t = new_tex(GL_TEXTURE_1D_ARRAY);
    GLubyte rows[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_R8, 3, 2, 0, GL_RED, GL_UNSIGNED_BYTE, rows);
    glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 1, 1, 1, 1, GL_RED, GL_UNSIGNED_BYTE, (GLubyte[]){ 99 });
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    GLubyte ro[2][4];
    memset(ro, 0, sizeof ro);
    glGetTexImage(GL_TEXTURE_1D_ARRAY, 0, GL_RED, GL_UNSIGNED_BYTE, ro);   /* rows padded to 4 */
    EXPECT_EQ_INT(ro[0][2], 3);
    EXPECT_EQ_INT(ro[1][0], 4);
    EXPECT_EQ_INT(ro[1][1], 99);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);

    /* 3D texture with SKIP_IMAGES */
    t = new_tex(GL_TEXTURE_3D);
    GLubyte vol[4][2][2];   /* 4 images, we upload images 1..2 */
    for (int z = 0; z < 4; z++) for (int y = 0; y < 2; y++) for (int x = 0; x < 2; x++) vol[z][y][x] = z * 10 + y * 2 + x;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_SKIP_IMAGES, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, 2, 2, 2, 0, GL_RED, GL_UNSIGNED_BYTE, vol);
    glPixelStorei(GL_UNPACK_SKIP_IMAGES, 0);
    GLubyte vo[2][2][2];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_UNSIGNED_BYTE, vo);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    EXPECT_EQ_INT(vo[0][0][0], 10);
    EXPECT_EQ_INT(vo[1][1][1], 23);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(getn_bufsize)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    GLubyte buf[64];
    glGetnTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 63, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetnTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, 64, buf);
    EXPECT_NO_GL_ERROR();
    glGetTextureImage(t, 0, GL_RGBA, GL_UNSIGNED_BYTE, 10, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_INT_8_8_8_8, buf);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteTextures(1, &t);
}

TEST(pixel_buffer_objects)
{
    GLubyte src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 }, out[8] = { 0 };
    GLuint pbo[2];
    glGenBuffers(2, pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo[0]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 16, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 4, 8, src);
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void *)4);
    EXPECT_NO_GL_ERROR();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void *)4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);   /* would read past the end */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void *)4);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, 16, NULL, GL_STREAM_READ);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void *)8);
    EXPECT_NO_GL_ERROR();
    glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 8, 8, out);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    EXPECT_TRUE(memcmp(out, src, 8) == 0);
    glDeleteBuffers(2, pbo);
    glDeleteTextures(1, &t);
}

/* ---- compressed ---------------------------------------------------------- */

static void bits_put(GLubyte *b, int pos, int n, unsigned v)
{
    for (int i = 0; i < n; i++, pos++)
        if (v & (1u << i)) b[pos / 8] |= (GLubyte)(1u << (pos % 8));
}

static void compressed_roundtrip(GLenum fmt, const GLubyte *block, int bsize, const GLubyte expect[4], int tol, const char *what)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, fmt, 4, 4, 0, bsize, block);
    GLenum e = glGetError();
    if (e) MGL_FAIL("%s: CompressedTexImage2D error 0x%x", what, e);
    GLubyte back[16];
    memset(back, 0, sizeof back);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, back);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    if (memcmp(back, block, bsize)) MGL_FAIL("%s: GetCompressedTexImage differs", what);
    if (fmt != GL_COMPRESSED_RGBA_S3TC_DXT1_EXT)   /* S3TC is an extension: not in the core query table */
    {
        GLint c = 0, sz = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &c);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &sz);
        EXPECT_EQ_INT(c, GL_TRUE);
        EXPECT_EQ_INT(sz, bsize);
    }
    /* decompressed readback */
    GLubyte px[16 * 4];
    memset(px, 0, sizeof px);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    mgl_test_checks++;
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 4; k++)
            if (abs(px[i * 4 + k] - expect[k]) > tol)
            {
                MGL_FAIL("%s: texel %d = %d,%d,%d,%d expected %d,%d,%d,%d", what, i, px[i * 4], px[i * 4 + 1], px[i * 4 + 2], px[i * 4 + 3],
                         expect[0], expect[1], expect[2], expect[3]);
                i = 16;
                break;
            }
    glDeleteTextures(1, &t);
}

TEST(compressed_formats)
{
    /* BC1: color0 = pure red (565), color1 = blue, all indices 0 */
    GLubyte bc1[8] = { 0x00, 0xF8, 0x1F, 0x00, 0, 0, 0, 0 };
    compressed_roundtrip(GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, bc1, 8, (GLubyte[]){ 255, 0, 0, 255 }, 0, "BC1");
    /* BC4: red0 = 255, red1 = 0, indices 0 */
    GLubyte bc4[8] = { 255, 0, 0, 0, 0, 0, 0, 0 };
    compressed_roundtrip(GL_COMPRESSED_RED_RGTC1, bc4, 8, (GLubyte[]){ 255, 0, 0, 255 }, 0, "BC4");
    /* BC5 */
    GLubyte bc5[16] = { 255, 0, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 0, 0 };
    compressed_roundtrip(GL_COMPRESSED_RG_RGTC2, bc5, 16, (GLubyte[]){ 255, 128, 0, 255 }, 1, "BC5");
    /* BC7 mode 6: R=254 G=0 B=0 A=254 (p bits 0), indices 0 */
    GLubyte bc7[16] = { 0 };
    bits_put(bc7, 0, 7, 0x40);
    bits_put(bc7, 7, 7, 127); bits_put(bc7, 14, 7, 127);
    bits_put(bc7, 49, 7, 127); bits_put(bc7, 56, 7, 127);
    compressed_roundtrip(GL_COMPRESSED_RGBA_BPTC_UNORM, bc7, 16, (GLubyte[]){ 254, 0, 0, 254 }, 0, "BC7");
    /* ETC2 RGB individual mode: base 0xF/0/0, table 0, indices 0 -> +2 */
    GLubyte etc2[8] = { 0xFF, 0x00, 0x00, 0x00, 0, 0, 0, 0 };
    compressed_roundtrip(GL_COMPRESSED_RGB8_ETC2, etc2, 8, (GLubyte[]){ 255, 2, 2, 255 }, 0, "ETC2 native");
    /* same block through the CPU decoder (forced fallback) */
    setenv("MGL_FORCE_ETC_DECODE", "1", 1);
    compressed_roundtrip(GL_COMPRESSED_RGB8_ETC2, etc2, 8, (GLubyte[]){ 255, 2, 2, 255 }, 0, "ETC2 CPU");
    /* RGBA8 ETC2 + EAC alpha: base 200, mul 1, table 13 index 4 -> +0 */
    GLubyte etc2a[16] = { 200, 0x1D, 0x92, 0x49, 0x24, 0x92, 0x49, 0x24, 0xFF, 0x00, 0x00, 0x00, 0, 0, 0, 0 };
    compressed_roundtrip(GL_COMPRESSED_RGBA8_ETC2_EAC, etc2a, 16, (GLubyte[]){ 255, 2, 2, 200 }, 0, "ETC2+EAC CPU");
    unsetenv("MGL_FORCE_ETC_DECODE");
    compressed_roundtrip(GL_COMPRESSED_RGBA8_ETC2_EAC, etc2a, 16, (GLubyte[]){ 255, 2, 2, 200 }, 0, "ETC2+EAC native");

    /* EAC R11: native vs CPU decode agree */
    GLubyte r11[8] = { 128, 0x00, 0, 0, 0, 0, 0, 0 };
    GLfloat a[16], b[16];
    GLuint t = new_tex(GL_TEXTURE_2D);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_R11_EAC, 4, 4, 0, 8, r11);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, a);
    setenv("MGL_FORCE_ETC_DECODE", "1", 1);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_R11_EAC, 4, 4, 0, 8, r11);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, b);
    unsetenv("MGL_FORCE_ETC_DECODE");
    EXPECT_NEAR(a[0], b[0], 1.0 / 1024);
    EXPECT_NEAR(b[0], 1025.0 / 2047.0, 1.0 / 1024);
    EXPECT_NO_GL_ERROR();
    glDeleteTextures(1, &t);
}

TEST(compressed_subimage_and_errors)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RED_RGTC1, 8, 8);
    EXPECT_NO_GL_ERROR();
    GLubyte blk[8] = { 255, 0, 0, 0, 0, 0, 0, 0 };
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 4, 4, 4, 4, GL_COMPRESSED_RED_RGTC1, 8, blk);
    EXPECT_NO_GL_ERROR();
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 2, 0, 4, 4, GL_COMPRESSED_RED_RGTC1, 8, blk);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_COMPRESSED_RED_RGTC1, 7, blk);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glCompressedTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_COMPRESSED_RG_RGTC2, 16, blk);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLubyte all[32];
    glGetCompressedTextureSubImage(t, 0, 4, 4, 0, 4, 4, 1, sizeof all, all);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(all[0], 255);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA, 4, 4, 0, 16, all);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);       /* generic formats are not allowed */
    glGetnCompressedTexImage(GL_TEXTURE_2D, 0, 4, all);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);  /* bufSize too small */
    glDeleteTextures(1, &t);

    /* unpack compressed block state: pick the 2nd block of an 8x4 image */
    GLubyte two[16] = { 10, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 0 };
    t = new_tex(GL_TEXTURE_2D);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_WIDTH, 4);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_HEIGHT, 4);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_SIZE, 8);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 8);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 4);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RED_RGTC1, 4, 4, 0, 8, two);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_WIDTH, 0);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_HEIGHT, 0);
    glPixelStorei(GL_UNPACK_COMPRESSED_BLOCK_SIZE, 0);
    EXPECT_NO_GL_ERROR();
    GLubyte back[8];
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, back);
    EXPECT_EQ_INT(back[0], 200);
    GLint v = -1;
    glGetIntegerv(GL_UNPACK_COMPRESSED_BLOCK_SIZE, &v);
    EXPECT_EQ_INT(v, 0);
    glDeleteTextures(1, &t);
}

/* ---- copies / clears ---------------------------------------------------- */

TEST(copy_tex_sub_image_from_fbo)
{
    GLuint ct = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glViewport(0, 0, 8, 8);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint dt = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 1, 1, 2, 2, 2, 2);
    EXPECT_NO_GL_ERROR();
    GLubyte o[4][4][4];
    memset(o, 0, sizeof o);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[1][1][1], 255);
    EXPECT_EQ_INT(o[2][2][1], 255);
    EXPECT_EQ_INT(o[2][2][0], 0);

    /* CopyTexImage2D defines a new level, with format conversion to RGBA32F */
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 0, 0, 2, 2, 0);
    EXPECT_NO_GL_ERROR();
    GLfloat f[4 * 4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, f);
    EXPECT_NEAR(f[1], 1.0, 1e-6);
    EXPECT_NEAR(f[0], 0.0, 1e-6);

    /* CopyTextureSubImage2D (DSA) after a new clear */
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glCopyTextureSubImage2D(dt, 0, 0, 0, 0, 0, 1, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, f);
    EXPECT_NEAR(f[0], 1.0, 1e-6);
    EXPECT_NEAR(f[4 + 1], 1.0, 1e-6);   /* (1,0) still green */

    /* integer destination from a normalized read buffer */
    GLuint it = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, 2, 2, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    GLuint ts[3] = { ct, dt, it };
    glDeleteTextures(3, ts);
}

TEST(copy_into_array_layer)
{
    GLuint ct = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, 4, 4);
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint at = new_tex(GL_TEXTURE_2D_ARRAY);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 4, 4, 3);
    glCopyTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 2, 0, 0, 4, 4);
    EXPECT_NO_GL_ERROR();
    GLubyte o[4];
    glGetTextureSubImage(at, 0, 3, 3, 2, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 4, o);
    EXPECT_EQ_INT(o[2], 255);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    GLuint ts[2] = { ct, at };
    glDeleteTextures(2, ts);
}

TEST(copy_image_sub_data)
{
    GLubyte src[4 * 4 * 4];
    for (int i = 0; i < (int)sizeof src; i++) src[i] = (GLubyte)i;
    GLuint a = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
    GLuint b = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glCopyImageSubData(a, GL_TEXTURE_2D, 0, 1, 1, 0, b, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    EXPECT_NO_GL_ERROR();
    GLubyte o[4 * 4 * 4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0], src[(1 * 4 + 1) * 4]);
    EXPECT_EQ_INT(o[(1 * 4 + 1) * 4 + 3], src[(2 * 4 + 2) * 4 + 3]);

    /* reinterpretation RGBA8 -> R32UI */
    GLuint c = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, 4, 4, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glCopyImageSubData(a, GL_TEXTURE_2D, 0, 0, 0, 0, c, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    EXPECT_NO_GL_ERROR();
    GLuint u[16];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, u);
    EXPECT_EQ_INT(u[1], 0x07060504u);

    /* BC1 block <-> RG32UI texel */
    GLuint d = new_tex(GL_TEXTURE_2D);
    GLubyte bc1[8] = { 0x00, 0xF8, 0x1F, 0x00, 0, 0, 0, 0 };
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 4, 4, 0, 8, bc1);
    GLuint e = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32UI, 1, 1, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, NULL);
    glCopyImageSubData(d, GL_TEXTURE_2D, 0, 0, 0, 0, e, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    EXPECT_NO_GL_ERROR();
    GLuint rg[2] = { 0, 0 };
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, rg);
    EXPECT_EQ_INT(rg[0], 0x001FF800u);

    /* errors */
    glCopyImageSubData(a, GL_TEXTURE_2D, 0, 3, 3, 0, b, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glCopyImageSubData(a, GL_TEXTURE_3D, 0, 0, 0, 0, b, GL_TEXTURE_2D, 0, 0, 0, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    GLuint f = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, 4, 4, 0, GL_RG, GL_UNSIGNED_BYTE, NULL);
    glCopyImageSubData(a, GL_TEXTURE_2D, 0, 0, 0, 0, f, GL_TEXTURE_2D, 0, 0, 0, 0, 1, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLuint ts[6] = { a, b, c, d, e, f };
    glDeleteTextures(6, ts);
}

TEST(clear_tex_image)
{
    GLuint t = new_tex(GL_TEXTURE_2D);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 4, 4);
    GLubyte col[4] = { 10, 20, 30, 40 };
    glClearTexImage(t, 0, GL_RGBA, GL_UNSIGNED_BYTE, col);
    GLfloat half[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
    glClearTexSubImage(t, 0, 1, 1, 0, 2, 2, 1, GL_RGBA, GL_FLOAT, half);
    EXPECT_NO_GL_ERROR();
    GLubyte o[4][4][4];
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0][0][0], 10);
    EXPECT_EQ_INT(o[0][0][3], 40);
    EXPECT_EQ_INT(o[1][1][0], 128);
    EXPECT_EQ_INT(o[2][2][2], 128);
    EXPECT_EQ_INT(o[3][3][1], 20);
    glClearTexImage(t, 1, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGetTexImage(GL_TEXTURE_2D, 1, GL_RGBA, GL_UNSIGNED_BYTE, o);
    EXPECT_EQ_INT(o[0][0][0], 0);
    EXPECT_EQ_INT(o[0][0][3], 0);
    glClearTexSubImage(t, 0, 3, 3, 0, 2, 2, 1, GL_RGBA, GL_UNSIGNED_BYTE, col);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glClearTexImage(t, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, col);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glClearTexImage(12345, 0, GL_RGBA, GL_UNSIGNED_BYTE, col);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteTextures(1, &t);
}

/* ---- objects and errors ------------------------------------------------- */

TEST(object_errors)
{
    GLuint t;
    glGenTextures(1, &t);
    EXPECT_TRUE(!glIsTexture(t));
    glBindTexture(GL_TEXTURE_2D, t);
    EXPECT_TRUE(glIsTexture(t));
    glBindTexture(GL_TEXTURE_3D, t);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindTexture(0x1234, t);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindTexture(GL_TEXTURE_2D, 0xFFFFFF);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glActiveTexture(GL_TEXTURE0 + 100000);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    GLint b = -1;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &b);
    EXPECT_EQ_INT(b, t);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, -1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, 0x1234, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexImage2D(GL_TEXTURE_2D, 0, 0x1234, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_6_5, NULL);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexImage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_NO_GL_ERROR();   /* any level may be specified first */
    glTexSubImage2D(GL_TEXTURE_2D, 1, 1, 1, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLuint t2 = new_tex(GL_TEXTURE_2D);
    glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA8, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);   /* too many levels */
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);        /* unsized */

    /* proxy */
    glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA8, 1 << 20, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    EXPECT_NO_GL_ERROR();
    GLint w = -1;
    glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 0);
    glTexImage2D(GL_PROXY_TEXTURE_2D, 0, GL_RGBA8, 64, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    EXPECT_EQ_INT(w, 64);

    /* BindTextures / BindTextureUnit and deletion */
    GLuint pair[2] = { t, t2 };
    glBindTextures(2, 2, pair);
    EXPECT_NO_GL_ERROR();
    glActiveTexture(GL_TEXTURE3);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &b);
    EXPECT_EQ_INT(b, t2);
    glActiveTexture(GL_TEXTURE0);
    glBindTextureUnit(5, 0xFFFFFF);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteTextures(2, pair);
    EXPECT_TRUE(!glIsTexture(t));
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &b);
    EXPECT_EQ_INT(b, 0);

    GLuint ct;
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &ct);
    EXPECT_TRUE(glIsTexture(ct));
    glTextureStorage2D(ct, 1, GL_RGBA8, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTextureStorage3D(ct, 1, GL_RGBA8, 4, 4, 2);
    EXPECT_NO_GL_ERROR();
    GLint imm = 0;
    glGetTextureParameteriv(ct, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
    EXPECT_EQ_INT(imm, GL_TRUE);
    GLuint bad;
    glCreateTextures(0x1234, 1, &bad);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDeleteTextures(1, &ct);
}

/* ---- sampling ----------------------------------------------------------- */

TEST(sample_compressed_and_uploaded)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "layout(location=0) out vec2 uv;\n"
        "void main(){ uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) in vec2 uv; layout(binding=0) uniform sampler2D tex;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = texture(tex, uv); }\n";
    GLuint prog = mgl_make_program(vs, fs);
    if (!prog) return;

    GLuint target = new_tex(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
    glViewport(0, 0, 4, 4);

    static const float tri[] = { -1, -1, 3, -1, -1, 3 };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);

    GLuint src = new_tex(GL_TEXTURE_2D);   /* unit 0 */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLubyte bc1[8] = { 0x00, 0xF8, 0x1F, 0x00, 0, 0, 0, 0 };
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 4, 4, 0, 8, bc1);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    GLubyte o[16 * 4];
    glGetTextureImage(target, 0, GL_RGBA, GL_UNSIGNED_BYTE, sizeof o, o);
    EXPECT_EQ_INT(o[0], 255);
    EXPECT_EQ_INT(o[2], 0);
    EXPECT_EQ_INT(o[63], 255);

    GLubyte cyan[4 * 4 * 4];
    for (int i = 0; i < 16; i++) { cyan[i * 4] = 0; cyan[i * 4 + 1] = 255; cyan[i * 4 + 2] = 255; cyan[i * 4 + 3] = 255; }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, cyan);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGetTextureImage(target, 0, GL_RGBA, GL_UNSIGNED_BYTE, sizeof o, o);
    EXPECT_EQ_INT(o[0], 0);
    EXPECT_EQ_INT(o[1], 255);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    GLuint ts[2] = { target, src };
    glDeleteTextures(2, ts);
}

int main(void)
{
    RUN(rgba8_roundtrip);
    RUN(format_roundtrips);
    RUN(format_conversions);
    RUN(unpack_pack_state);
    RUN(subimage_after_gpu);
    RUN(mip_levels_and_generate);
    RUN(cube_map_faces);
    RUN(arrays_and_3d);
    RUN(getn_bufsize);
    RUN(pixel_buffer_objects);
    RUN(compressed_formats);
    RUN(compressed_subimage_and_errors);
    RUN(copy_tex_sub_image_from_fbo);
    RUN(copy_into_array_layer);
    RUN(copy_image_sub_data);
    RUN(clear_tex_image);
    RUN(object_errors);
    RUN(sample_compressed_and_uploaded);
    return mgl_test_summary();
}
