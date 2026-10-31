/* compat s34: sRGB renderbuffer attachments must not get Metal's automatic
 * clear/write encode while GL_FRAMEBUFFER_SRGB is off — the r52 fix swapped
 * only TEXTURE attachments to a plain-UNORM view; renderbuffers were handed to
 * Metal in their _sRGB format, so glClear stored encode(color) and readback
 * (raw, per packed_pixels) returned the encoded bytes
 * (KHR-GL46.direct_state_access.renderbuffers_storage{,_multisample}: clear
 * 0.25 -> fetched 0.537255 = sRGB(0.25)). */
#include "mgl_test.h"

/* sRGB transfer, float -> encoded byte lattice -> float, as GL/Mesa do it */
static float srgb_encode(float l)
{
    return l <= 0.0031308f ? l * 12.92f : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f;
}

TEST(srgb_rbo_clear_reads_back_clear_color)
{
    GLuint fbo = 0, rbo = 0;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_SRGB8_ALPHA8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_NO_GL_ERROR();
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("fbo incomplete"); return; }

    /* the value 0.25 is in the linear part where encode != identity: encode(0.25) ~ 0.537 */
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    /* readback returns the stored bytes raw (packed_pixels model); the clear must
     * have stored the clear color itself, not its sRGB encoding */
    EXPECT_NEAR(px[0], 0.25f, 1.5f / 255.0f);
    EXPECT_NEAR(px[1], 0.50f, 1.5f / 255.0f);
    EXPECT_NEAR(px[2], 0.75f, 1.5f / 255.0f);
    EXPECT_NEAR(px[3], 1.00f, 1.5f / 255.0f);

    /* the whole attachment, not just pixel 0 */
    float row[8 * 4];
    glReadPixels(0, 7, 8, 1, GL_RGBA, GL_FLOAT, row);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(row[8 * 4 - 4], 0.25f, 1.5f / 255.0f);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
}

TEST(srgb_tex_clear_control)
{
    /* control: a TEXTURE attachment with the same sRGB internal format already
     * stores the clear color raw (r52 plain-view swap) */
    GLuint fbo = 0, tex = 0;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_NO_GL_ERROR();
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("fbo incomplete"); return; }

    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(px[0], 0.25f, 1.5f / 255.0f);
    EXPECT_NEAR(px[2], 0.75f, 1.5f / 255.0f);

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
}

TEST(srgb_rbo_render_writes_linear_when_srgb_off)
{
    /* fragment writes into an sRGB renderbuffer with GL_FRAMEBUFFER_SRGB off
     * must store the linear (raw) color, same rule as texture attachments */
    GLuint fbo = 0, rbo = 0, prog = 0;
    const char *vs = "#version 330 core\n"
                     "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);"
                     " gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
    const char *fs = "#version 330 core\n"
                     "layout(location = 0) out vec4 c;\n"
                     "void main() { c = vec4(0.25, 0.5, 0.75, 1.0); }";
    prog = glCreateProgram();
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s1, 1, &vs, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &fs, NULL); glCompileShader(s2);
    glAttachShader(prog, s1); glAttachShader(prog, s2); glLinkProgram(prog);
    glDeleteShader(s1); glDeleteShader(s2);
    EXPECT_NO_GL_ERROR();

    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_SRGB8_ALPHA8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    glViewport(0, 0, 8, 8);
    glDisable(GL_DEPTH_TEST);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    /* raw linear bytes, NOT encode(0.25) ~ 0.537 */
    EXPECT_NEAR(px[0], 0.25f, 1.5f / 255.0f);
    EXPECT_NEAR(px[2], 0.75f, 1.5f / 255.0f);

    glUseProgram(0);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
}

TEST(srgb_rbo_roundtrip_with_srgb_enabled)
{
    /* with GL_FRAMEBUFFER_SRGB on, MGL renders/clears through the _sRGB view;
     * the clear color is still stored raw (Mesa parity: glClear applies no
     * transfer), so clear -> read must still roundtrip */
    GLuint fbo = 0, rbo = 0;
    glGenFramebuffers(1, &fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_SRGB8_ALPHA8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    glDisable(GL_FRAMEBUFFER_SRGB);
    EXPECT_NO_GL_ERROR();
    EXPECT_NEAR(px[0], 0.25f, 1.5f / 255.0f);
    EXPECT_NEAR(px[1], 0.50f, 1.5f / 255.0f);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
}


TEST(srgb_rbo_multisample_resolve_roundtrip)
{
    /* the DSA renderbuffers_storage_multisample chain: clear a 4x sRGB rbo,
     * glBlitFramebuffer-resolve into a single-sample sRGB rbo, read raw back */
    GLuint msfbo = 0, msrbo = 0, sfbo = 0, srbo = 0;
    glGenFramebuffers(1, &msfbo);
    glGenRenderbuffers(1, &msrbo);
    glBindRenderbuffer(GL_RENDERBUFFER, msrbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_SRGB8_ALPHA8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, msfbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msrbo);

    glGenFramebuffers(1, &sfbo);
    glGenRenderbuffers(1, &srbo);
    glBindRenderbuffer(GL_RENDERBUFFER, srbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_SRGB8_ALPHA8, 8, 8);
    glBindFramebuffer(GL_FRAMEBUFFER, sfbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, srbo);
    EXPECT_NO_GL_ERROR();
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("resolve fbo incomplete"); return; }

    glBindFramebuffer(GL_FRAMEBUFFER, msfbo);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_READ_FRAMEBUFFER, msfbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, sfbo);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 8, 8, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, sfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sfbo);

    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_NO_GL_ERROR();
    /* raw bytes through the resolve — not encode(0.25) ~ 0.537, not decode(0.25) ~ 0.051 */
    EXPECT_NEAR(px[0], 0.25f, 1.5f / 255.0f);
    EXPECT_NEAR(px[1], 0.50f, 1.5f / 255.0f);
    EXPECT_NEAR(px[2], 0.75f, 1.5f / 255.0f);
    EXPECT_NEAR(px[3], 1.00f, 1.5f / 255.0f);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &msfbo);
    glDeleteRenderbuffers(1, &msrbo);
    glDeleteFramebuffers(1, &sfbo);
    glDeleteRenderbuffers(1, &srbo);
}

int main(void)
{
    RUN(srgb_rbo_clear_reads_back_clear_color);
    RUN(srgb_tex_clear_control);
    RUN(srgb_rbo_render_writes_linear_when_srgb_off);
    RUN(srgb_rbo_roundtrip_with_srgb_enabled);
    RUN(srgb_rbo_multisample_resolve_roundtrip);
    return mgl_test_summary();
}
