/* map Metal's interpolate_at_offset behavior over offsets with +0.5 vs +0.4375:
 * a linear gradient (4x4 viewport); probe GL offsets -0.375, -0.25, 0, 0.25, 0.375
 * and print the resulting gradient value.  center=0.125, 1px = 0.25 in gradient. */
#include "mgl_test.h"
#include <stdio.h>

static const char *vs =
    "#version 440 core\n"
    "in vec2 a_position;\n"
    "out vec4 v_pos;\n"
    "void main() { gl_Position = vec4(a_position, 0.0, 1.0);\n"
    "  v_pos = vec4((a_position + 1.0) * 0.5, 0.0, 1.0); }\n";

static const char *fs =
    "#version 440 core\n"
    "in vec4 v_pos;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() {\n"
    "  o_color = vec4(interpolateAtOffset(v_pos, vec2(OFFSET)).xy, 0.0, 1.0);\n"
    "}\n";

TEST(offset_map)
{
    const int W = 4, H = 4;
    GLuint fbo, tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, W, H);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pos[8] = { -1,-1, -1,1, 1,-1, 1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, 0);

    const float offs[] = { -0.375f, -0.25f, 0.0f, 0.25f, 0.375f };
    for (unsigned i = 0; i < sizeof offs / sizeof offs[0]; i++)
    {
        char fsrc[1024];
        char num[32];
        snprintf(num, sizeof num, "%ff", offs[i]);
        const char *p = strstr(fs, "OFFSET");
        snprintf(fsrc, (size_t)(p - fs) + 1, "%s", fs);
        strcat(fsrc, num);
        strcat(fsrc, p + strlen("OFFSET"));
        GLuint prog = mgl_make_program(vs, fsrc);
        EXPECT_TRUE(prog != 0);
        if (!prog) continue;
        glUseProgram(prog);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        float o[4] = { -9, -9, -9, -9 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, o);
        float expected = 0.125f + offs[i] * 0.25f;   /* 1px = 0.25 in gradient units */
        printf("  offset %+.3f -> %f (GL exact %.6f)\n", offs[i], o[0], expected);
        EXPECT_TRUE(fabsf(o[0] - expected) < 0.002f);
        EXPECT_TRUE(fabsf(o[1] - expected) < 0.002f);
    }
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(offset_map);
    return mgl_test_summary();
}
