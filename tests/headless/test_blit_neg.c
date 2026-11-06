/* glBlitFramebuffer depth/stencil format-match rules (GL 4.6 18.3.1):
 *  - read D32FS8 -> draw D24S8 (single-sample) : GL_INVALID_OPERATION (formats
 *    do not match), for both the DEPTH and the STENCIL bit
 *  - read without depth -> draw with depth      : silent no-op of the aspect
 *  - multisample read (resolve)                 : formats may convert, no error
 * compat r99: the D32FS8 -> D24S8 case now errors (dEQP ES31 debug.negative_coverage
 * get_error.buffer.blit_framebuffer expects it) — GL compares internalformats even
 * when MGL shares one Metal storage for both.
 *  - MS D24S8 read -> D32FS8 draw               : resolves and converts (no error)
 */
#include "mgl_test.h"

static GLuint fbo[2], rbo[2], tex[2];

TEST(a_ss_ds_mismatch_errors)
{
    glGenFramebuffers(2, fbo);
    glGenRenderbuffers(2, rbo);
    glGenTextures(2, tex);

    /* packed depth-stencil formats all share MGL's Metal storage
     * (Depth32Float_Stencil8), so D32FS8 <-> D24S8 blits are legal; a packed
     * DS attachment vs a depth-only attachment is a genuine mismatch */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[0]);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 32, 32);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);

    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* depth-only format mismatch: DC16 vs DC24 */
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, 32, 32);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    /* matching packed formats are fine — but BOTH sides must be the same format:
     * a D32FS8 read vs a D24S8 draw is a format mismatch and errors per GL 4.6
     * 18.3.1 (dEQP ES31 debug.negative_coverage get_error.buffer.blit_framebuffer),
     * even though MGL stores both as Depth32Float_Stencil8.  Make BOTH sides
     * D24S8: */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[0]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[0]);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_NO_ERROR);
}

TEST(b_missing_read_aspect_noop)
{
    /* read has color only; draw has color+depth+stencil; DEPTH/STENCIL bits are
     * silently dropped (KHR-GL46 color-config test) */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
    glBindTexture(GL_TEXTURE_2D, tex[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[0], 0);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);

    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16,
                      GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_NO_ERROR);
}

TEST(c_ms_resolve_converts)
{
    /* multisampled D24S8 read resolves into a D32FS8 draw (KHR-GL46 depth-config
     * test blits into the CTS's D32FS8 default framebuffer) */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo[0]);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[0]);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[0]);
    EXPECT_NO_GL_ERROR();
    GLenum st = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) MGL_FAIL("read fbo status=0x%x", (unsigned)st);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo[1]);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo[1]);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, 32, 32);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo[1]);

    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_NO_ERROR);
    glBlitFramebuffer(0, 0, 32, 32, 0, 0, 32, 32, GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_NO_ERROR);
}

int main(void)
{
    RUN(a_ss_ds_mismatch_errors);
    RUN(b_missing_read_aspect_noop);
    RUN(c_ms_resolve_converts);
    return mgl_test_summary();
}

