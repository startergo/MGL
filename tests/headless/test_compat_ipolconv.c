/* measure: interpolateAtOffset(v, (0,0)) vs the sample-qualified v — how far
 * off is MGL/SPIRV-Cross's translate?  prints the delta as a color gradient
 * value (x = r, y = g) so the magnitude identifies the convention. */
#include "mgl_test.h"

static const char *vs =
    "#version 440 core\n"
    "in vec2 a_position;\n"
    "out vec4 v_pos;\n"
    "void main() { gl_Position = vec4(a_position, 0.0, 1.0);\n"
    "  v_pos = vec4((a_position + 1.0) * 0.5, 0.0, 1.0); }\n";

static const char *fs =
    "#version 440 core\n"
    "in vec4 v_pos;\n"
    "sample in vec4 v_samp;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() {\n"
    "  vec4 at0 = interpolateAtOffset(v_pos, vec2(0.0));\n"
    "  o_color = vec4(at0.xy, 0.0, 1.0);\n"
    "}\n";

static const char *fs2 =
    "#version 440 core\n"
    "sample in vec4 v_samp;\n"
    "layout(location=0) out vec4 o_color;\n"
    "void main() { o_color = vec4(v_samp.xy, 0.0, 1.0); }\n";

static const char *vs2 =
    "#version 440 core\n"
    "in vec2 a_position;\n"
    "out vec4 v_samp;\n"
    "void main() { gl_Position = vec4(a_position, 0.0, 1.0);\n"
    "  v_samp = vec4((a_position + 1.0) * 0.5, 0.0, 1.0); }\n";

TEST(offset_convention)
{
    /* 4x4 single-sample target: at pixel (0,0) the center value of the linear
     * (position+1)/2 gradient is ((-1 + 1/8) + 1)/2 = 7/16 = 0.4375 (x and y).
     * Print what interpolateAtOffset(0) and the sample qualifier produce. */
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

    float o0[4] = { -9, -9, -9, -9 }, o1[4] = { -9, -9, -9, -9 };

    GLuint p1 = mgl_make_program(vs, fs);
    EXPECT_TRUE(p1 != 0);
    glUseProgram(p1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, o0);

    GLuint p2 = mgl_make_program(vs2, fs2);
    EXPECT_TRUE(p2 != 0);
    glUseProgram(p2);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, o1);

    printf("  interpolateAtOffset(0,0) = %f %f   sample-qualified = %f %f   (center = 0.4375)\n",
           o0[0], o0[1], o1[0], o1[1]);
    /* 1/16 px in a 4px viewport = 1/64 = 0.015625; 0.4375 px = 0.109375 */
    EXPECT_TRUE(fabsf(o0[0] - o1[0]) < 0.02f && fabsf(o0[1] - o1[1]) < 0.02f);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(offset_convention);
    return mgl_test_summary();
}
