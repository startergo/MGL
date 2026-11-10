/* test_compat_blit_srgb — GL_FRAMEBUFFER_SRGB semantics of glBlitFramebuffer
 * colour copies between sRGB-encoded and linear framebuffers (GL 4.6 §18.3.1;
 * Mesa blit.c keeps the source's sRGB format — i.e. reads DECODE — only while
 * ctx->Color.sRGBEnabled):
 *
 *  - cap OFF: a blit passes the stored bytes through unchanged, whatever the
 *    two sides' encodings are (MGL blits through plain-UNORM views).
 *  - cap ON:  an sRGB source is DECODED to linear in flight; an sRGB draw
 *    framebuffer RE-ENCODES on write; a linear draw framebuffer keeps the
 *    linear value.
 *
 * (MGL's byte model: clears and ReadPixels of an sRGB attachment are RAW — no
 * conversion, as the KHR-GL46.direct_state_access renderbuffer tests verify —
 * so a source cleared to 0.25 holds the bytes 0.25 and the cap-ON blit must
 * decode them to 0.0508.  dEQP-GL45-ES31 functional.fbo.srgb_write_control.
 * framebuffer_srgb_enabled_copy_to_linear failed before r104: the blit always
 * passed bytes through.)
 */
#include "mgl_test.h"

static GLuint mkfbo(GLenum ifmt)
{
    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return fbo;
}

static float center(GLuint fbo)
{
    float px[4] = { -1, -1, -1, -1 };
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(8, 8, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    return px[0];
}

/* sRGB transfer, exactly as GL defines it (IEC 61966-2-1) */
static float enc(float x)
{
    return x <= 0.0031308f ? x * 12.92f : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
}
static float dec(float x)
{
    return x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

TEST(blit_srgb_write_control)
{
    GLuint srgb = mkfbo(GL_SRGB8_ALPHA8);
    GLuint lin = mkfbo(GL_RGBA8);
    GLuint srgb2 = mkfbo(GL_SRGB8_ALPHA8);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* both sRGB framebuffers hold the raw bytes 0.25 (clear is unconverted) */
    glBindFramebuffer(GL_FRAMEBUFFER, srgb);
    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, srgb2);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, lin);
    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /* readback is raw too */
    EXPECT_NEAR(center(srgb), 0.25f, 1.0f / 255.0f);

    /* 1. cap OFF: SRGB -> LINEAR passes the bytes through */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, srgb);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lin);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(lin), 0.25f, 1.5f / 255.0f);

    /* 2. cap OFF: LINEAR -> SRGB passes the bytes through */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, lin);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, srgb2);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(srgb2), 0.25f, 1.5f / 255.0f);

    /* 3. cap ON: SRGB -> LINEAR DECODES (the dEQP case) */
    glEnable(GL_FRAMEBUFFER_SRGB);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, srgb);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lin);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(lin), dec(0.25f), 1.5f / 255.0f);

    /* 4. cap ON: SRGB -> SRGB round-trips (decode then re-encode) */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, srgb);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, srgb2);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(srgb2), 0.25f, 1.5f / 255.0f);

    /* 5. cap ON, LINEAR source into the sRGB dst: ENCODED on write */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, lin);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, srgb2);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    /* the linear dst of case 3 holds dec(0.25); encode(dec(0.25)) == 0.25 bytes */
    EXPECT_NEAR(center(srgb2), enc(dec(0.25f)), 1.5f / 255.0f);

    /* 6. scaled blit with the cap ON also converts (sample() filters decoded values) */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, srgb);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lin);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(lin), dec(0.25f), 1.5f / 255.0f);
    glDisable(GL_FRAMEBUFFER_SRGB);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &srgb);
    glDeleteFramebuffers(1, &lin);
    glDeleteFramebuffers(1, &srgb2);
}

TEST(blit_srgb_msaa_resolve)
{
    /* the multisample branch of the same rule: resolving an sRGB MSAA source
     * into a linear single-sample framebuffer decodes while GL_FRAMEBUFFER_SRGB
     * is on, passes bytes through while it is off (Mesa blit.c: the MS source
     * keeps its sRGB format only while sRGBEnabled). */
    GLuint src = 0, lin = 0, stex = 0, ltex = 0;

    glGenTextures(1, &stex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, stex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 2, GL_SRGB8_ALPHA8, 32, 32, GL_TRUE);
    glGenFramebuffers(1, &src);
    glBindFramebuffer(GL_FRAMEBUFFER, src);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, stex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* store raw 0.25 bytes in every sample (clear is unconverted) */
    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glGenTextures(1, &ltex);
    glBindTexture(GL_TEXTURE_2D, ltex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &lin);
    glBindFramebuffer(GL_FRAMEBUFFER, lin);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ltex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* resolve with the cap OFF: raw bytes */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, lin);
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(lin), 0.25f, 1.5f / 255.0f);

    /* resolve with the cap ON: decoded */
    glEnable(GL_FRAMEBUFFER_SRGB);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);   /* center() rebound the read FB */
    glClearBufferfv(GL_COLOR, 0, (const float[4]){ 0, 0, 0, 1 });
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(center(lin), dec(0.25f), 1.5f / 255.0f);
    glDisable(GL_FRAMEBUFFER_SRGB);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &src);
    glDeleteFramebuffers(1, &lin);
    glDeleteTextures(1, &stex);
    glDeleteTextures(1, &ltex);
}

int main(void)
{
    RUN(blit_srgb_write_control);
    RUN(blit_srgb_msaa_resolve);
    return mgl_test_summary();
}
