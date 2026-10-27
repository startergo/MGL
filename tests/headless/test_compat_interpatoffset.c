/* test_compat_interpatoffset.c — interpolateAtOffset(v, gl_SamplePosition-0.5)
 * must equal the sample-qualified interpolation of the same varying
 * (KHR-GL46.shader_multisample_interpolation.render.interpolate_at_offset_check). */
#include "mgl_test.h"

static const char *vs =
    "#version 440 core\n"
    "in vec2 a_position;\n"
    "in vec4 a_color;\n"
    "out vec4 v_colorBase;\n"
    "out vec4 v_color;\n"
    "void main() { gl_Position = vec4(a_position, 0.0, 1.0);\n"
    "  v_colorBase = a_color; v_color = a_color; }\n";

static const char *fs =
    "#version 440 core\n"
    "in vec4 v_colorBase;\n"
    "sample in vec4 v_color;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() {\n"
    "  vec4 temp = interpolateAtOffset(v_colorBase, gl_SamplePosition - 0.5);\n"
    "  bool ok = all(lessThan(abs(temp - v_color), vec4(1.0/255.0)));\n"
    "  o_color = vec4(ok ? 1.0 : 0.0);\n"
    "}\n";

static void one_case(int samples)
{
    GLuint tex;
    glGenTextures(1, &tex);
    if (samples > 1)
    {
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
        glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA8, 8, 8, GL_FALSE);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           samples > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, tex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("  samples=%d: FBO incomplete\n", samples);
        EXPECT_TRUE(0);
        return;
    }
    glViewport(0, 0, 8, 8);

    GLuint prog = mgl_make_program(vs, fs);
    EXPECT_TRUE(prog != 0);
    if (!prog)
        return;
    glUseProgram(prog);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pos[8] = { -1,-1, -1,1, 1,-1, 1,1 };
    float col[16] = { 0,0,0,1, 0,1,0,1, 1,0,0,1, 1,1,0,1 };
    float both[24];
    memcpy(both, pos, sizeof pos);
    memcpy(both + 8, col, sizeof col);
    glBufferData(GL_ARRAY_BUFFER, sizeof both, both, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 16, (void *)32);

    if (samples > 1)
        glEnable(GL_SAMPLE_SHADING);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    glDisable(GL_SAMPLE_SHADING);

    /* resolve + read */
    GLuint rfbo, rtex;
    glGenTextures(1, &rtex);
    glBindTexture(GL_TEXTURE_2D, rtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtex, 0);
    glViewport(0, 0, 8, 8);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 8, 8, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, rfbo);

    GLubyte p[4] = { 0, 0, 0, 0 };
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    printf("  samples=%d pixel = %d %d %d %d\n", samples, p[0], p[1], p[2], p[3]);
    EXPECT_TRUE(p[0] > 200);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &rtex);
}

TEST(interpolate_at_offset_check)
{
    one_case(1);
    one_case(2);
    one_case(4);
}

int main(void)
{
    RUN(interpolate_at_offset_check);
    return mgl_test_summary();
}
