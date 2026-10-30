/*
 * test_alphaless — GL formats with no alpha channel (R3_G3_B2, RGB*, ...)
 * live on an RGBA Metal backing. GL 4.6: writes to the missing channel are
 * discarded and READS RETURN ONE — sampling (texture() and texelFetch()),
 * after an upload and after rendering into the texture.
 *
 * KHR-GL46.texture_size_promotion.functional: renders into an
 * R3_G3_B2 multisample texture and texelFetch()es .a from sampler2DMS.
 */
#include "mgl_test.h"

static const char *vs_src =
    "#version 460 core\n"
    "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"
    "  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

static const char *fs_store =
    "#version 460 core\n"
    "layout(location = 0) out vec4 o;\n"
    "void main() { o = vec4(0.25, 0.5, 0.75, 0.125); }\n";

static const char *fs_ms_src(int sample)
{
    static char buf[256];
    snprintf(buf, sizeof buf,
             "#version 460 core\n"
             "layout(location = 0) out vec4 o;\n"
             "uniform sampler2DMS s;\n"
             "void main() { o = vec4(texelFetch(s, ivec2(0), %d)); }\n", sample);
    return buf;
}

static const char *fs_tex =
    "#version 460 core\n"
    "layout(location = 0) out vec4 o;\n"
    "uniform sampler2D s;\n"
    "void main() { o = texture(s, vec2(0.5)); }\n";

static GLuint prog(const char *fs)
{
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER), p = glCreateProgram();
    GLint ok = 0;
    char log[512];
    glShaderSource(v, 1, &vs_src, NULL); glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(v, sizeof log, NULL, log); MGL_FAIL("vs: %s", log); return 0; }
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(f, sizeof log, NULL, log); MGL_FAIL("fs: %s", log); return 0; }
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(f, sizeof log, NULL, log); glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); return 0; }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

static GLuint g_vao;

static GLuint read_rgba8(void)
{
    GLubyte px[4];
    glFinish();
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return (GLuint)px[0] | ((GLuint)px[1] << 8) | ((GLuint)px[2] << 16) | ((GLuint)px[3] << 24);
}

/* render a full-screen quad writing (0.25, 0.5, 0.75, 0.125) into the current FBO */
static void draw_quad(void)
{
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

TEST(ms_r3g3b2_texelfetch_alpha_one)
{
    GLint max_samples = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
    if (max_samples < 2) max_samples = 2;
    GLuint tex, fbo, p = prog(fs_ms_src(max_samples - 1));
    if (!p) return;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, max_samples - 1, GL_R3_G3_B2, 1, 1, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    GLuint sp = prog(fs_store);
    glUseProgram(sp);
    glViewport(0, 0, 1, 1);
    draw_quad();   /* alpha 0.125 must be discarded */
    EXPECT_NO_GL_ERROR();

    /* read it back through the attachment: alpha must read 1 */
    GLuint dst, dfbo;
    glGenTextures(1, &dst);
    glBindTexture(GL_TEXTURE_2D, dst);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &dfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, dfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
    glViewport(0, 0, 1, 1);

    glUseProgram(p);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glUniform1i(glGetUniformLocation(p, "s"), 0);
    draw_quad();
    EXPECT_NO_GL_ERROR();

    GLuint px = read_rgba8();
    printf("   [ms alpha probe] max_samples=%d px=%08x\n", max_samples, px);
    EXPECT_EQ_INT(px & 0xff, 64);        /* 0.25 */
    EXPECT_EQ_INT((px >> 8) & 0xff, 128);/* 0.5 */
    EXPECT_EQ_INT((px >> 16) & 0xff, 191);/* 0.75 */
    EXPECT_EQ_INT((px >> 24) & 0xff, 255);/* missing alpha reads 1 */

    glDeleteFramebuffers(2, (GLuint[]){fbo, dfbo});
    glDeleteTextures(2, (GLuint[]){tex, dst});
    glDeleteProgram(p); glDeleteProgram(sp);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

TEST(render_into_rgb8_sample_alpha_one)
{
    GLuint tex, fbo, p = prog(fs_store), s = prog(fs_tex);
    if (!p || !s) return;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 8, 8);
    glUseProgram(p);
    draw_quad();  /* alpha 0.125 discarded */
    EXPECT_NO_GL_ERROR();

    GLuint dst, dfbo;
    glGenTextures(1, &dst);
    glBindTexture(GL_TEXTURE_2D, dst);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &dfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, dfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
    glUseProgram(s);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(s, "s"), 0);
    draw_quad();
    GLuint px = read_rgba8();
    EXPECT_EQ_INT(px & 0xff, 64);
    EXPECT_EQ_INT((px >> 8) & 0xff, 128);
    EXPECT_EQ_INT((px >> 16) & 0xff, 191);
    EXPECT_EQ_INT((px >> 24) & 0xff, 255);

    glDeleteFramebuffers(2, (GLuint[]){fbo, dfbo});
    glDeleteTextures(2, (GLuint[]){tex, dst});
    glDeleteProgram(p); glDeleteProgram(s);
}

TEST(r3g3b2_upload_swizzle_a)
{
    /* upload with GL_RGBA data whose alpha is 0 — GL discards it; SWIZZLE_A
     * sourcing A must read 1 (texture_swizzle.functional composition) */
    GLuint tex;
    GLubyte data[4 * 4 * 4];
    for (int i = 0; i < 16; i++)
    {
        data[4 * i + 0] = 255; data[4 * i + 1] = 0; data[4 * i + 2] = 0; data[4 * i + 3] = 0;
    }
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R3_G3_B2, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    EXPECT_NO_GL_ERROR();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
    EXPECT_NO_GL_ERROR();
    GLint a = 7;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, &a);
    EXPECT_EQ_INT(a, GL_ALPHA);

    const char *fsswz =
        "#version 460 core\n"
        "layout(location = 0) out vec4 o;\n"
        "uniform sampler2D s;\n"
        "void main() { o = texture(s, vec2(0.5)); }\n";
    GLuint p = prog(fsswz), fbo, dst;
    if (!p) return;
    glGenTextures(1, &dst);
    glBindTexture(GL_TEXTURE_2D, dst);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
    glViewport(0, 0, 4, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glUseProgram(p);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(p, "s"), 0);
    draw_quad();
    GLuint px = read_rgba8();
    EXPECT_EQ_INT(px & 0xff, 255);
    EXPECT_EQ_INT((px >> 8) & 0xff, 0);
    EXPECT_EQ_INT((px >> 16) & 0xff, 0);
    EXPECT_EQ_INT((px >> 24) & 0xff, 255);   /* SWIZZLE_A=ALPHA of a no-alpha format reads 1 */

    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(2, (GLuint[]){tex, dst});
    glDeleteProgram(p);
}

int main(void)
{
    glGenVertexArrays(1, &g_vao);
    glBindVertexArray(g_vao);
    RUN(ms_r3g3b2_texelfetch_alpha_one);
    RUN(render_into_rgb8_sample_alpha_one);
    RUN(r3g3b2_upload_swizzle_a);
    return mgl_test_summary();
}
