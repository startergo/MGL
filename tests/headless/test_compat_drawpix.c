/*
 * test_compat_drawpix.c — glDrawPixels payload formats (compat domain).
 *
 * GL 4.6 §16.4: the DrawPixels path copies its payload into the framebuffer
 * verbatim — the sRGB framebuffer encoding applies to shader-written
 * fragments, not to the pixel path.  MGL blits through a textured quad, so an
 * sRGB draw target must see the payload through an sRGB view (decode on
 * sample + encode on write = byte identity).  Compressed payloads (RGTC) are
 * decoded per block on the CPU.  Reading back from an sRGB framebuffer
 * converts to linear (§18.2.2), which the test models with its own decoder.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"
#include <math.h>

/* GL sRGB transfer (IEC 61966-2-1) */
static float srgb_decode(float v)
{
    if (v <= 0.04045f) return v / 12.92f;
    return powf((v + 0.055f) / 1.055f, 2.4f);
}

static GLuint make_fbo(GLenum ifmt)
{
    GLuint fbo, tex;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, 16, 16, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    return fbo;   /* objects are per-test scope; the context is shared */
}

/* 1: GL_SRGB8 target, GL_RED payload: the stored byte must equal the payload
 * byte (readback decodes sRGB->linear, so compare against srgb_decode) */
TEST(srgb8_target_verbatim)
{
    make_fbo(GL_SRGB8);

    GLubyte data[16 * 16];
    for (int i = 0; i < 16 * 16; i++) data[i] = (GLubyte)(i * 4 + 4);
    glDrawPixels(16, 16, GL_RED, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();

    GLubyte out[16 * 16 * 4];
    glReadPixels(0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, out);
    EXPECT_NO_GL_ERROR();
    int bad = 0;
    for (int i = 0; i < 16 * 16; i++)
    {
        int w = (int)(srgb_decode(data[i] / 255.0f) * 255.0f + 0.5f);
        if (abs((int)out[i * 4] - w) > 1)
            bad++;
    }
    EXPECT_TRUE(bad == 0);
    if (bad)
        MGL_FAIL("%d/%d pixels not byte-preserved", bad, 16 * 16);
}

/* 2: linear R8 target, GL_SRGB8 payload: bytes pass through untouched */
TEST(srgb8_payload_passthrough)
{
    make_fbo(GL_R8);

    GLubyte data[16 * 16 * 3];
    for (int i = 0; i < 16 * 16; i++)
    {
        data[i * 3 + 0] = (GLubyte)(i * 4 + 4);
        data[i * 3 + 1] = (GLubyte)(i * 2 + 1);
        data[i * 3 + 2] = (GLubyte)(i * 3 + 7);
    }
    glDrawPixels(16, 16, GL_SRGB8, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();

    GLubyte out[16 * 16 * 4];
    glReadPixels(0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, out);
    EXPECT_NO_GL_ERROR();
    int bad = 0;
    for (int i = 0; i < 16 * 16; i++)
        if (out[i * 4] != data[i * 3 + 0] || out[i * 4 + 1] != 0 ||
            out[i * 4 + 2] != 0 || out[i * 4 + 3] != 255)   /* R8 target: G=B=0, A=1 */
            bad++;
    EXPECT_TRUE(bad == 0);
    if (bad)
        MGL_FAIL("%d/%d pixels not raw", bad, 16 * 16);
}

/* 3: GL_COMPRESSED_RED_RGTC1 payload, one 8-byte 4x4 block decoded on the CPU
 * (endpoints 240/16, codes 0..3 row-major: palette 240,16,184,156) */
TEST(rgtc1_payload)
{
    make_fbo(GL_R8);

    GLubyte block[8];
    block[0] = 240; block[1] = 16;
    memset(block + 2, 0, 6);
    for (unsigned i = 0; i < 16; i++)
    {
        unsigned bit = 2 * i;
        block[2 + (bit >> 3)] |= (GLubyte)((i % 4) << (bit & 7));
    }
    glDrawPixels(4, 4, GL_COMPRESSED_RED_RGTC1, GL_UNSIGNED_BYTE, block);
    EXPECT_NO_GL_ERROR();

    GLubyte out[16 * 16 * 4];
    glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, out);
    EXPECT_NO_GL_ERROR();
    static const int pal[4] = { 240, 16, 184, 156 };
    int bad = 0;
    for (int i = 0; i < 16; i++)
        if (abs((int)out[i * 4] - pal[i % 4]) > 1)
            bad++;
    EXPECT_TRUE(bad == 0);
    if (bad)
        MGL_FAIL("%d/16 texels not decoded (row0: %d,%d,%d,%d)",
                 bad, out[0], out[4], out[8], out[12]);
}

int main(void)
{
    void *cctx = mglCompatCreateContext();
    EXPECT_TRUE(cctx != NULL);
    if (!cctx)
        return 1;
    RUN(srgb8_target_verbatim);
    RUN(srgb8_payload_passthrough);
    RUN(rgtc1_payload);
    return mgl_test_summary();
}
