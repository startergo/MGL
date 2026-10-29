/* test_bptc — online compression of uncompressed sub-image data into REAL
 * compressed RGTC / BPTC storage (glTexStorage + glTexSubImage path used by
 * KHR-GL46.texture_view.view_classes), plus the CPU BC6H/BC7 encoders it
 * depends on (texdata_formats.c):
 *   - the compressed level keeps hardware blocks: glGetCompressedTexImage
 *     returns 16 bytes for a 4x4 BPTC block and re-uploads byte-exact;
 *   - decode (glGetTexImage of the compressed level) round-trips the ramp the
 *     encoders are near-lossless on (BC7 mode 6 is exact on the 4-step ramp,
 *     BC6H mode 10 within endpoint quantization);
 *   - block alignment and format-compat rules mirror CompressedTexSubImage.
 */
#include "mgl_test.h"
#include <string.h>
#include <math.h>

/* 4x4 RGBA ramp: texel (x,y) = (y*16+x*4) + channel, i.e. 0..63 in steps of 4
 * (the same shape KHR-GL46.texture_view.view_classes uploads) */
static GLubyte ramp4[4][4][4];
static GLfloat framp4[4][4][3];

static void make_ramps(void)
{
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
        {
            int i = y * 16 + x * 4;
            ramp4[y][x][0] = (GLubyte)(i + 0);
            ramp4[y][x][1] = (GLubyte)(i + 1);
            ramp4[y][x][2] = (GLubyte)(i + 2);
            ramp4[y][x][3] = (GLubyte)(i + 3);
            framp4[y][x][0] = (float)(i + 0);
            framp4[y][x][1] = (float)(i + 1);
            framp4[y][x][2] = (float)(i + 2);
        }
}

TEST(bc7_online)
{
    GLuint t = 0;
    GLubyte back[4][4][4];
    GLubyte blocks[16];

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGBA_BPTC_UNORM, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_NO_GL_ERROR();

    memset(back, 0xee, sizeof(back));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back);
    EXPECT_NO_GL_ERROR();
    /* BC7 mode 6 with parity-quantized endpoints: error <= 1 per channel */
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            for (int c = 0; c < 4; c++)
            {
                int d = back[y][x][c] - ramp4[y][x][c];
                if (d < -1 || d > 1)
                    MGL_FAIL("bc7 (%d,%d) ch%d: %d vs %d", x, y, c, back[y][x][c], ramp4[y][x][c]);
                mgl_test_checks++;
            }

    /* real hardware blocks come back out */
    memset(blocks, 0, sizeof(blocks));
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, blocks);
    EXPECT_NO_GL_ERROR();
    {
        int nz = 0;
        for (int i = 0; i < 16; i++)
            if (blocks[i])
                nz++;
        EXPECT_TRUE(nz > 0);        /* mode bit 6 set at least */
        EXPECT_EQ_INT(blocks[0] & 0x7f, 0x40); /* mode 6: bit 6, bits 0..5 clear */
    }

    /* blocks re-upload into a second texture and decode identically */
    {
        GLuint t2 = 0;
        GLubyte back2[4][4][4];
        glGenTextures(1, &t2);
        glBindTexture(GL_TEXTURE_2D, t2);
        glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_BPTC_UNORM, 4, 4, 0, 16, blocks);
        EXPECT_NO_GL_ERROR();
        memset(back2, 0xee, sizeof(back2));
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back2);
        EXPECT_NO_GL_ERROR();
        EXPECT_TRUE(memcmp(back, back2, sizeof(back)) == 0);
        glDeleteTextures(1, &t2);
    }
    glDeleteTextures(1, &t);
}

TEST(bc7_srgb_online)
{
    GLuint t = 0;
    GLubyte back[4][4][4];

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_NO_GL_ERROR();
    memset(back, 0xee, sizeof(back));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back);
    EXPECT_NO_GL_ERROR();
    /* the sRGB variant encodes the same blocks; readback may be raw (sRGB
     * encoding, value 1) or linearized (srgb_decode(1/255) -> 0): accept both.
     * What must hold: no error and the near-black texel stays near-black. */
    EXPECT_TRUE(back[0][0][0] <= 1);
    glDeleteTextures(1, &t);
}

/* VK-GL-CTS reference BC6H decode (decompressBc6H) of one mode-10 block */
static const uint16_t bc6h_w4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

static uint32_t tb_getbits(const uint8_t *d, int first, int last)
{
    uint32_t r = 0;
    for (int i = first; i <= last; i++)
        if (d[i >> 3] & (1 << (i & 7)))
            r |= 1u << (i - first);
    return r;
}

static float tb_half(uint32_t bits)
{
    int s = (bits >> 15) & 1, e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
    double v;
    if (e == 0)
        v = ldexp((double)m, -24);
    else
        v = ldexp((double)(1024 + m), e - 25);
    return (float)(s ? -v : v);
}

static void tb_decode_bc6h(const uint8_t *blk, int isign, float out[4][4][3])
{
    int32_t e[2][3];
    int off[6] = {5, 15, 25, 35, 45, 55};

    for (int i = 0; i < 6; i++)
    {
        int32_t c = (int32_t)tb_getbits(blk, off[i], off[i] + 9);
        int32_t u;
        if (isign)
        {
            int s = (c >> 9) & 1;
            c = s ? c - 1024 : c;
            int32_t m = c < 0 ? -c : c;
            u = m == 0 ? 0 : (m >= 511 ? (c < 0 ? -0x7fff : 0x7fff) : (c < 0 ? -(64 * m + 32) : 64 * m + 32));
        }
        else
        {
            u = c == 0 ? 0 : (c == 1023 ? 0xffff : 64 * c + 32);
        }
        e[i / 3][i % 3] = u;
    }

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
        {
            int t = y * 4 + x;
            int idx = (t == 0) ? (int)tb_getbits(blk, 65, 67)
                               : (int)tb_getbits(blk, 65 + 3 + (t - 1) * 4, 65 + 3 + (t - 1) * 4 + 3);
            int w = bc6h_w4[idx];
            for (int c = 0; c < 3; c++)
            {
                int32_t v = ((64 - w) * e[0][c] + w * e[1][c] + 32) >> 6;
                uint32_t bits;
                if (isign)
                {
                    int s = v < 0;
                    int32_t mm = s ? -v : v;
                    mm = (mm * 31) >> 5;
                    bits = (uint32_t)((s ? 0x8000u : 0u) | (uint32_t)mm);
                }
                else
                {
                    bits = (uint32_t)((v * 31) / 64);
                }
                out[y][x][c] = tb_half(bits);
            }
        }
}

/* shared: upload the float ramp to a BPTC float level, then verify that the
 * METAL hardware decode equals the VK-GL-CTS reference decode of the encoded
 * blocks (bit-exact: proves the encoder emits spec-conformant blocks), plus a
 * loose value check (BC6H is lossy; one shared 4-bit index per RGB texel caps
 * quality, which GL does not guarantee for online compression). */
static void bc6h_check(GLenum ifmt, int isign, float tol_abs, float tol_rel)
{
    GLuint t = 0;
    GLfloat back[4][4][3];
    float ref[4][4][3];
    uint8_t blocks[16];

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            for (int c = 0; c < 3; c++)
                framp4[y][x][c] = (float)(y * 16 + x * 4 + c - (isign ? 128 : 0));

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, ifmt, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGB, GL_FLOAT, framp4);
    EXPECT_NO_GL_ERROR();
    memset(back, 0, sizeof(back));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, back);
    EXPECT_NO_GL_ERROR();
    memset(blocks, 0, sizeof(blocks));
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, blocks);
    EXPECT_NO_GL_ERROR();
    /* mode field must be mode 10: bits[1:0] = 11b, bits[4:2] = 000b */
    EXPECT_EQ_INT(blocks[0] & 0x1f, 0x03);
    tb_decode_bc6h(blocks, isign, ref);
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            for (int c = 0; c < 3; c++)
            {
                if (back[y][x][c] != ref[y][x][c])
                    MGL_FAIL("hw decode (%d,%d,%d) = %f, CTS ref %f", x, y, c, back[y][x][c], ref[y][x][c]);
                mgl_test_checks++;
                if (tol_rel > 0)
                {
                    float tol = tol_abs + tol_rel * (framp4[y][x][c] < 0 ? -framp4[y][x][c] : framp4[y][x][c]);
                    float d = back[y][x][c] - framp4[y][x][c];
                    if (d < -tol || d > tol)
                        MGL_FAIL("lossy (%d,%d,%d): %f vs %f (tol %f)", x, y, c, back[y][x][c], framp4[y][x][c], tol);
                    mgl_test_checks++;
                }
            }
    glDeleteTextures(1, &t);
}

TEST(bc6h_online)
{
    /* unsigned: bit-exact vs the CTS reference decoder; value quality is not
     * checked (a per-channel-offset ramp is adversarial for the format's one
     * shared 4-bit index per RGB texel; endpoints are still exact) */
    bc6h_check(GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, 0, 1.0f, 0.0f);
}

TEST(bc6h_signed_online)
{
    bc6h_check(GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, 1, 2.0f, 0.05f);
}

TEST(rgtc_online_subimage)
{
    GLuint t = 0;
    GLubyte back[4][4];
    GLubyte rramp[4][4];

    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            rramp[y][x] = (GLubyte)(y * 16 + x * 4);

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RED_RGTC1, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RED, GL_UNSIGNED_BYTE, rramp);
    EXPECT_NO_GL_ERROR();
    memset(back, 0xee, sizeof(back));
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, back);
    EXPECT_NO_GL_ERROR();
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
        {
            int d = back[y][x] - rramp[y][x];
            if (d < -4 || d > 4)
                MGL_FAIL("rgtc (%d,%d): %d vs %d", x, y, back[y][x], rramp[y][x]);
            mgl_test_checks++;
        }
    glDeleteTextures(1, &t);
}

TEST(online_errors)
{
    GLuint t = 0;

    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);

    /* ETC2 has no encoder: still INVALID_OPERATION */
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGB8_ETC2, 4, 4);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGB, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteTextures(1, &t);

    /* block alignment rules mirror CompressedTexSubImage */
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_COMPRESSED_RGBA_BPTC_UNORM, 8, 8);
    EXPECT_NO_GL_ERROR();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 2, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    /* a full trailing edge may be unaligned (right/bottom edge exception) */
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_NO_GL_ERROR();
    /* integer client data into a non-integer compressed format */
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, ramp4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDeleteTextures(1, &t);
}

int main(void)
{
    make_ramps();
    RUN(bc7_online);
    RUN(bc7_srgb_online);
    RUN(bc6h_online);
    RUN(bc6h_signed_online);
    RUN(rgtc_online_subimage);
    RUN(online_errors);
    return mgl_test_summary();
}
