/* test_compat_sampleshading.c — KHR-GL46.sample_shading.render repro: with
 * GL_SAMPLE_SHADING + minSampleShading(1), a gradient quad rendered into an
 * MSAA texture must give per-sample interpolation (each sample of a pixel a
 * different color); without it, all samples of a pixel share one color. */
#include "mgl_test.h"

static const char *vs_src =
    "#version 400 core\n"
    "in vec2 a_position;\n"
    "in vec4 a_color;\n"
    "out vec4 v_color;\n"
    "void main() { gl_Position = vec4(a_position, 0.0, 1.0); v_color = a_color; }\n";

static const char *fs_src =
    "#version 400 core\n"
    "in vec4 v_color;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() { o_color = vec4(v_color.x, v_color.y, 0.0, 0.0); }\n";

#define W 16
#define H 16

TEST(sample_shading_interpolation)
{
    GLint maxSamples = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
    printf("  GL_MAX_SAMPLES = %d\n", (int)maxSamples);
    int samples = maxSamples < 4 ? maxSamples : 4;
    EXPECT_TRUE(samples >= 2);

    GLuint prog = mgl_make_program(vs_src, fs_src);
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);
    glTexStorage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA8, W, H, GL_FALSE);

    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, W, H);

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

    glMinSampleShading(1.0f);
    glEnable(GL_SAMPLE_SHADING);
    EXPECT_EQ_INT(glIsEnabled(GL_SAMPLE_SHADING), GL_TRUE);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();

    glMinSampleShading(0.0f);
    glDisable(GL_SAMPLE_SHADING);

    /* unroll the samples into a W*samples wide single-sample target */
    const char *uv = "#version 400 core\nin vec2 p; void main(){ gl_Position=vec4(p,0,1); }";
    const char *uf = "#version 400 core\n"
                     "uniform sampler2DMS u_tex; uniform int u_samples;\n"
                     "layout(location=0) out vec4 o;\n"
                     "void main(){ ivec2 c = ivec2(int(gl_FragCoord.x)/u_samples, gl_FragCoord.y);\n"
                     "  o = texelFetch(u_tex, c, int(gl_FragCoord.x)%u_samples); }";
    GLuint rp = mgl_make_program(uv, uf);
    EXPECT_TRUE(rp != 0);
    glUseProgram(rp);

    GLuint rbo, fbo2;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, W * samples, H);
    glGenFramebuffers(1, &fbo2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, W * samples, H);
    glUniform1i(glGetUniformLocation(rp, "u_samples"), samples);
    glUniform1i(glGetUniformLocation(rp, "u_tex"), 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, tex);

    /* reuse the position-only attribs */
    glDisableVertexAttribArray(1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();

    /* count unique pixels */
    static unsigned char px[W * 4 * H * 4];
    glReadPixels(0, 0, W * samples, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int unique = 0, seen[W * H * 4][2];
    for (int i = 0; i < W * samples * H; i++)
    {
        int a = px[i * 4 + 0], b = px[i * 4 + 1];
        int dup = 0;
        for (int j = 0; j < unique; j++)
            if (seen[j][0] == a && seen[j][1] == b) { dup = 1; break; }
        if (!dup) { seen[unique][0] = a; seen[unique][1] = b; unique++; }
    }
    printf("  samples=%d unique colors=%d (need >= %d)\n", samples, unique, W * H * samples);
    EXPECT_TRUE(unique >= W * H * samples);   /* full sample shading: distinct per-sample values */
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(sample_shading_interpolation);
    return mgl_test_summary();
}
