/* test_texswizzle.c — regression tests for KHR-GL46.texture_swizzle.functional
 * (geom r58):
 *   1. GL_TEXTURE_SWIZZLE_* composes with GL_DEPTH_TEXTURE_MODE (GL 4.6 8.26:
 *      the swizzle selectors apply to the depth-mode-splatted vector, they do
 *      not override the mode), and DEPTH_TEXTURE_MODE defaults to GL_RED.
 *   2. rendering into an attachment whose format has no alpha channel must not
 *      store the fill's alpha (GL discards writes to missing channels, reads
 *      return one) — Metal's RGBA backing otherwise leaks it.
 */
#include "mgl_test.h"
#include <math.h>

#ifndef GL_DEPTH_TEXTURE_MODE
#define GL_DEPTH_TEXTURE_MODE 0x884B
#endif

static const char *kVS =
    "#version 460 core\n"
    "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); "
    "gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";

TEST(depth_mode_swizzle_compose)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint prog = mgl_make_program(kVS,
        "#version 460 core\n"
        "uniform sampler2D tex; out vec4 col;\n"
        "void main() { col = texelFetch(tex, ivec2(0, 0), 0); }\n");

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, 4, 4, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    /* mode default must be GL_RED (not LUMINANCE) */
    GLint mode = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, &mode);
    EXPECT_TRUE(mode == GL_RED);
    /* swizzle [A,B,G,R] over the RED-mode splat (d,0,0,1) -> (1,0,0,d) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_ALPHA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_RED);
    EXPECT_NO_GL_ERROR();

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0);
    glClearDepth(0.5);
    glClear(GL_DEPTH_BUFFER_BIT);
    EXPECT_NO_GL_ERROR();

    GLuint out;
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
    glViewport(0, 0, 1, 1);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    /* depth16 quantizes 0.5 -> 32768/65535 */
    EXPECT_TRUE(px[0] == 1.0f && px[1] == 0.0f && px[2] == 0.0f && fabsf(px[3] - 0.5f) < 0.001f);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &out);
    glDeleteVertexArrays(1, &vao);
}

TEST(alphaless_attachment_masks_alpha_writes)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_R3_G3_B2, 8, 8, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    /* fill the MS texture with alpha = 0; R3_G3_B2 has no alpha channel so the
     * write must be discarded and a later swizzle red <- alpha must read one */
    GLuint fp = mgl_make_program(kVS,
        "#version 460 core\nout vec4 c; void main() { c = vec4(0.5, 0.5, 0.5, 0.0); }\n");
    GLuint fill;
    glGenFramebuffers(1, &fill);
    glBindFramebuffer(GL_FRAMEBUFFER, fill);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    glViewport(0, 0, 8, 8);
    glUseProgram(fp);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    GLuint prog = mgl_make_program(kVS,
        "#version 460 core\nuniform sampler2DMS tex; out vec4 col;\n"
        "void main() { col = texelFetch(tex, ivec2(0, 0), 0); }\n");
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLuint out;
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, NULL);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    glViewport(0, 0, 1, 1);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_R, GL_ALPHA);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_G, GL_BLUE);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
    glTexParameteri(GL_TEXTURE_2D_MULTISAMPLE, GL_TEXTURE_SWIZZLE_A, GL_RED);
    glUniform1i(glGetUniformLocation(prog, "tex"), 0);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();
    float px[4] = { -1, -1, -1, -1 };
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
    EXPECT_TRUE(px[0] == 1.0f);        /* red <- alpha of (r,g,b,1) */
    EXPECT_TRUE(fabsf(px[1] - 0.5f) < 0.05f); /* green <- blue (3-bit) */

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(fp);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fill);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &out);
    glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(depth_mode_swizzle_compose);
    RUN(alphaless_attachment_masks_alpha_writes);
    return mgl_test_summary();
}
