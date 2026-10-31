/* test_compat_mscaps.c — multisample capability queries (supervisor 16:22:
 * "the multisample sample-variable / image-size NotSupported group").
 *
 * Two findings this locks in:
 *  1. glGetIntegerv(GL_MAX_SAMPLES) used to return 0 (it was collapsed into the
 *     GL_MAX_IMAGE_SAMPLES "no MS image stores in MSL" case in get.c) while the
 *     renderbuffer validator and glGetFramebufferParameteriv(GL_MAX_SAMPLES)
 *     always used the real device cap.  GL 4.6 requires MAX_SAMPLES >= 4; with
 *     0 the CTS gated whole groups with "Test sample count greater than
 *     MAX_SAMPLES" (sample_variables samples_8) / "Not supported".
 *  2. GL_MAX_IMAGE_SAMPLES is the device sample count (4): MS image LOADS are
 *     Metal texture2d_ms reads and MS image stores/atomics run through the geom
 *     image-atomic sidecar (r79+, rasterized back at the barrier); the
 *     word-addressable-only store support is declared honestly through
 *     texture_query's GL_SHADER_IMAGE_STORE = NONE.
 *
 * samples_8 CTS cases remain NotSupported on this hardware (Apple M4 Pro:
 * supportsTextureSampleCount 8 -> NO, 4 -> YES) which is spec-correct
 * behaviour, not a defect.  samples_1/2/4 sample-variable cases already Pass.
 */
#include "mgl_test.h"

#define MS_CHECK(cond, fmt, ...) do { mgl_test_checks++; \
    if (!(cond)) MGL_FAIL(fmt, ##__VA_ARGS__); } while (0)

TEST(max_samples_query)
{
    GLint v = -1;
    glGetIntegerv(GL_MAX_SAMPLES, &v);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(v >= 4, "GL_MAX_SAMPLES %d < 4 (GL 4.6 minimum)", v);

    GLint fbs = -1;
    glGetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &fbs);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(fbs == v, "GL_MAX_FRAMEBUFFER_SAMPLES %d != GL_MAX_SAMPLES %d", fbs, v);

    /* NOTE: GL_MAX_SAMPLES is NOT a glGetFramebufferParameteriv pname in
     * GL 4.6 (only GL_FRAMEBUFFER_DEFAULT_* for user FBOs), so the only public
     * query is the global glGetIntegerv above; MGL correctly raises
     * GL_INVALID_VALUE for the FBO-parameter spelling (kept out of this test).
     * framebuffer_ext.c's ext-slot GL_MAX_SAMPLES handler (device cap) is the
     * pre-existing fallback and now agrees with get.c (4). */
}

TEST(image_samples_zero_and_sync)
{
    GLint v = -1;
    glGetIntegerv(GL_MAX_IMAGE_SAMPLES, &v);
    EXPECT_NO_GL_ERROR();
    /* geom r79 (MS image sidecar store path): stores and atomics on MS images
     * are rerouted through the image-atomic sidecar and rasterized back into
     * the texture at the barrier (test_geom_msimg verifies stores per sample,
     * layout collisions, loads and imageAtomicAdd), so the cap is the device
     * sample count again — no longer 0. */
    MS_CHECK(v == 4, "GL_MAX_IMAGE_SAMPLES %d != 4 (MS image stores work via the geom sidecar)", v);

    /* the related caps that DO describe MSAA support all report >= 4 */
    GLint cts = -1, dts = -1, is = -1;
    glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &cts);
    glGetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES, &dts);
    glGetIntegerv(GL_MAX_INTEGER_SAMPLES, &is);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(cts >= 4 && dts >= 4 && is >= 4, "MS texture samples caps %d/%d/%d (all must be >= 4)", cts, dts, is);

    /* internalformat sample-count enumeration for an MS texture target lists
     * the device's real counts, highest first, count >= 2 ({1,4} here) */
    GLint n = -1;
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_NUM_SAMPLE_COUNTS, 1, &n);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(n >= 2, "GL_NUM_SAMPLE_COUNTS %d < 2 on TEXTURE_2D_MULTISAMPLE", n);

    GLint counts[8] = {0};
    glGetInternalformativ(GL_TEXTURE_2D_MULTISAMPLE, GL_RGBA8, GL_SAMPLES, 8, counts);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(counts[0] >= 4, "RGBA8 MS sample counts: first=%d (expected 4 first)", counts[0]);
}

TEST(renderbuffer_four_samples_works)
{
    /* the cap is not a lie: a 4x RGBA8 renderbuffer FBO must be complete and
     * clearable — this is exactly what sample_variables samples_4 cases do */
    GLuint rbo = 0, fbo = 0;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, 64, 64);
    EXPECT_NO_GL_ERROR();

    GLint rb_samples = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &rb_samples);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(rb_samples == 4, "GL_RENDERBUFFER_SAMPLES %d != 4", rb_samples);

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_NO_GL_ERROR();
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    MS_CHECK(st == GL_FRAMEBUFFER_COMPLETE, "4x RGBA8 renderbuffer FBO not complete (0x%x)", st);

    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    GLint fb_samples = 0;
    glGetIntegerv(GL_SAMPLES, &fb_samples);
    EXPECT_NO_GL_ERROR();
    MS_CHECK(fb_samples == 4, "bound FBO GL_SAMPLES %d != 4", fb_samples);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
    while (glGetError() != GL_NO_ERROR) {}
}

TEST(multisample_texture_rejects_unsupported)
{
    /* samples beyond the device cap must be an error, not silent success:
     * Apple M4 Pro supports {1,2,4}; 8 is invalid (this is what makes the CTS
     * samples_8 cases *correctly* skip on this hardware) */
    GLint maxs = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &maxs);
    EXPECT_NO_GL_ERROR();

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs, GL_RGBA8, 32, 32, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, maxs * 2 + 1, GL_RGBA8, 32, 32, GL_TRUE);
    GLenum e = glGetError();
    MS_CHECK(e != GL_NO_ERROR,
             "TexImage2DMultisample(samples=%d > MAX_SAMPLES %d) accepted without error", maxs * 2 + 1, maxs);

    glDeleteTextures(1, &tex);
    while (glGetError() != GL_NO_ERROR) {}
}

int main(void)
{
    RUN(max_samples_query);
    RUN(image_samples_zero_and_sync);
    RUN(renderbuffer_four_samples_works);
    RUN(multisample_texture_rejects_unsupported);
    return mgl_test_summary();
}
