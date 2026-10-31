/* compat s36: GL_RGB9_E5 renderbuffer emulation (RGBA16F backing).
 *
 * Apple Metal has MTLPixelFormatRGB9E5Float but it is sample-only (no render
 * target), and GL 4.6 table 8.13 lists RGB9_E5 color-renderable = N — yet
 * KHR-GL46.internalformat.renderbuffer.rgb9_e5 wants a WORKING RGB9_E5 colour
 * renderbuffer (storage -> complete FBO -> draw with depth test -> readPixels),
 * so MGL emulates it: the renderbuffer is allocated as RGBA16Float and the
 * 5-9-9-9 shared-exponent lattice (GL 4.6 2.3.5.2) is re-applied to the backing
 * before ANY read / copy-out consumes it (fbQuantizeRGB9E5).  Every lattice
 * value is exactly representable in half float (min 2^-24, max 65504), so the
 * quantised backing IS the exact 5-9-9-9 image.
 *
 * These tests pin:
 *   - storage + FRAMEBUFFER_COMPLETE (+ depth attachment alongside)
 *   - clear -> readback exact in GL_FLOAT and GL_UNSIGNED_INT_5_9_9_9_REV
 *   - fragment write (values NOT half-representable) -> stored value lies
 *     exactly on the 5-9-9-9 lattice (readback float == decode(encode(float)))
 *   - blit copy-out quantised
 *   - caps honest: COLOR_RENDERABLE/FB_RENDERABLE TRUE for GL_RENDERBUFFER
 *     target, FALSE for texture targets; RGB9_E5 TEXTURE attachments still
 *     incomplete; attachment component sizes 9/9/9
 */
#include "mgl_test.h"

/* GL 4.6 2.3.5.2 float -> shared exponent (N=9, B=15, E_max=31) */
static uint32_t enc5(float r, float g, float b)
{
    const float sharedExpMax = 65504.0f;   /* 511/512 * 2^16 */
    float rc = fminf(fmaxf(r, 0.0f), sharedExpMax);
    float gc = fminf(fmaxf(g, 0.0f), sharedExpMax);
    float bc = fminf(fmaxf(b, 0.0f), sharedExpMax);
    float mx = fmaxf(rc, fmaxf(gc, bc));
    float exp_p = fmaxf(-16.0f, floorf(log2f(mx))) + 16.0f;
    float scale = exp2f(exp_p - 24.0f);
    float max_s = floorf(mx / scale + 0.5f);
    if (!(max_s >= 0.0f && max_s < 512.0f)) scale *= 2.0f;
    uint32_t rr = (uint32_t)floorf(rc / scale + 0.5f) & 511;
    uint32_t gg = (uint32_t)floorf(gc / scale + 0.5f) & 511;
    uint32_t bb = (uint32_t)floorf(bc / scale + 0.5f) & 511;
    uint32_t ee = (uint32_t)floorf(exp_p + ((max_s >= 0.0f && max_s < 512.0f) ? 0.0f : 1.0f)) & 31;
    return rr | (gg << 9) | (bb << 18) | (ee << 27);
}

static void dec5(uint32_t v, float *rgb)
{
    float sc = ldexpf(1.0f, (int)(v >> 27) - 24);
    rgb[0] = (float)(v & 0x1ff) * sc;
    rgb[1] = (float)((v >> 9) & 0x1ff) * sc;
    rgb[2] = (float)((v >> 18) & 0x1ff) * sc;
}

TEST(rgb9e5_rbo_storage_complete_clear)
{
    GLuint fbo = 0, rbo = 0, depth = 0;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glGenRenderbuffers(1, &depth);

    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB9_E5, 16, 16);
    EXPECT_NO_GL_ERROR();

    glBindRenderbuffer(GL_RENDERBUFFER, depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, 16, 16);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    EXPECT_NO_GL_ERROR();
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("rgb9e5 fbo incomplete"); return; }

    GLint rs = -1, gs = -1, enc = -1;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &rs);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &gs);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &enc);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(rs, 9);
    EXPECT_EQ_INT(gs, 9);
    EXPECT_EQ_INT(enc, GL_LINEAR);

    /* all clear-color values are exactly representable in half AND on the
     * 5-9-9-9 lattice -> readback must be exact, in both read types */
    glClearColor(0.5f, 0.25f, 0.125f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(px[0], 0.5f, 0.0f);
    EXPECT_NEAR(px[1], 0.25f, 0.0f);
    EXPECT_NEAR(px[2], 0.125f, 0.0f);

    GLuint e5 = 0;
    glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, &e5);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(e5, enc5(0.5f, 0.25f, 0.125f));

    /* corner pixel: the whole attachment was cleared */
    glReadPixels(15, 15, 1, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, &e5);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(e5, enc5(0.5f, 0.25f, 0.125f));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &depth);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);
}

static const char *kFs =
    "#version 330 core\n"
    "uniform vec3 uColor;\n"
    "out vec4 oColor;\n"
    "void main() { oColor = vec4(uColor, 1.0); }\n";
static const char *kVs =
    "#version 330 core\n"
    "void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));"
    " gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

TEST(rgb9e5_rbo_fragment_write_on_lattice)
{
    GLuint fbo = 0, rbo = 0, prog = 0, vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB9_E5, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_NO_GL_ERROR();
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("rgb9e5 fbo incomplete"); return; }

    GLuint vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs, 1, &kVs, NULL); glCompileShader(vs);
    glShaderSource(fs, 1, &kFs, NULL); glCompileShader(fs);
    prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs); glLinkProgram(prog);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) { MGL_FAIL("link failed"); return; }
    glUseProgram(prog);

    /* 0.1/0.2/0.4 are NOT representable in half float: the store rounds to
     * half, the emulation then snaps to the 5-9-9-9 lattice.  What the test can
     * pin exactly: the stored image lies ON the lattice, i.e. float readback ==
     * decode(5-9-9-9 readback), and both are within 2^-8 of the written color. */
    glUniform3f(glGetUniformLocation(prog, "uColor"), 0.1f, 0.2f, 0.4f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    GLuint e5 = 0;
    glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, &e5);
    EXPECT_NO_GL_ERROR();

    float lat[3];
    dec5(e5, lat);
    EXPECT_NEAR(px[0], lat[0], 0.0f);
    EXPECT_NEAR(px[1], lat[1], 0.0f);
    EXPECT_NEAR(px[2], lat[2], 0.0f);
    EXPECT_NEAR(px[0], 0.1f, 1.0f / 256.0f);
    EXPECT_NEAR(px[1], 0.2f, 1.0f / 256.0f);
    EXPECT_NEAR(px[2], 0.4f, 1.0f / 256.0f);

    /* large values exercise a non-zero shared exponent: 100.0/200.0/64768.0 */
    glUniform3f(glGetUniformLocation(prog, "uColor"), 100.0f, 200.0f, 64768.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, &e5);
    float px2[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px2);
    EXPECT_NO_GL_ERROR();
    float lat2[3];
    dec5(e5, lat2);
    EXPECT_NEAR(px2[0], lat2[0], 0.0f);
    EXPECT_NEAR(px2[2], lat2[2], 0.0f);
    EXPECT_NEAR(px2[2], 64768.0f, 64768.0f / 256.0f);

    glDeleteProgram(prog);
    glDeleteShader(vs); glDeleteShader(fs);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteVertexArrays(1, &vao);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);
}

TEST(rgb9e5_blit_copy_out_quantised)
{
    GLuint sfbo = 0, srbo = 0, dfbo = 0, dtex = 0;
    glGenFramebuffers(1, &sfbo);
    glGenRenderbuffers(1, &srbo);
    glBindRenderbuffer(GL_RENDERBUFFER, srbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB9_E5, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, sfbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, srbo);
    glClearColor(0.3f, 0.6f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    glGenTextures(1, &dtex);
    glBindTexture(GL_TEXTURE_2D, dtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 8, 8, 0, GL_RGBA, GL_FLOAT, NULL);
    glGenFramebuffers(1, &dfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, dfbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, dtex, 0);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, sfbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dfbo);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 8, 8, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, dfbo);
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    /* 0.3/0.6/0.9 are half-exact (0.3000488, 0.60009766, 0.89990234); the blit
     * source must already be on the lattice, so the copied half equals the
     * decode of the spec encode of the half-rounded clear colour */
    float lat[3];
    dec5(enc5(0.30004883f, 0.60009766f, 0.89990234f), lat);
    EXPECT_NEAR(px[0], lat[0], 0.0f);
    EXPECT_NEAR(px[1], lat[1], 0.0f);
    EXPECT_NEAR(px[2], lat[2], 0.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &dfbo);
    glDeleteTextures(1, &dtex);
    glDeleteFramebuffers(1, &sfbo);
    glDeleteRenderbuffers(1, &srbo);
}

TEST(rgb9e5_caps_and_texture_attachments)
{
    GLint v = -1;
    glGetInternalformativ(GL_RENDERBUFFER, GL_RGB9_E5, GL_COLOR_RENDERABLE, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_TRUE);
    v = -1;
    glGetInternalformativ(GL_RENDERBUFFER, GL_RGB9_E5, GL_FRAMEBUFFER_RENDERABLE, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_FULL_SUPPORT);
    v = -1;
    glGetInternalformativ(GL_RENDERBUFFER, GL_RGB9_E5, GL_INTERNALFORMAT_RED_TYPE, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_FLOAT);

    /* texture images of the format remain non-renderable (honest caps: Metal
     * cannot render to RGB9E5Float and MGL does not emulate texture images) */
    v = -1;
    glGetInternalformativ(GL_TEXTURE_2D, GL_RGB9_E5, GL_COLOR_RENDERABLE, 1, &v);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(v, GL_FALSE);

    GLuint fbo = 0, tex = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB9_E5, 8, 8, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, NULL);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteTextures(1, &tex);
    glDeleteFramebuffers(1, &fbo);
}

int main(void)
{
    RUN(rgb9e5_rbo_storage_complete_clear);
    RUN(rgb9e5_rbo_fragment_write_on_lattice);
    RUN(rgb9e5_blit_copy_out_quantised);
    RUN(rgb9e5_caps_and_texture_attachments);
    return 0;
}
