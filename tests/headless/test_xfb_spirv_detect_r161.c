/* r161: an ARB_gl_spirv program declares transform-feedback capture only through
 * the Xfb execution mode / XfbBuffer / XfbStride — not through the Offset
 * decorations every uniform-block member carries.  A SPIR-V vertex shader with a
 * uniform block must link without running the xfb link (which decompiled the
 * stage and compiled it through glslang on every glLinkProgram: rsvga paid
 * ~50 ms per link and ~600 ms for the first); a shader that does declare capture
 * still gets it. */
#include "mgl_test.h"
#include <stdint.h>
#include "spv_xfb_detect.h"

extern uint64_t mgl_xfb_link_count;

static GLuint spv_shader(GLenum type, const unsigned char *bin, size_t n)
{
    GLuint sh = glCreateShader(type);
    glShaderBinary(1, &sh, GL_SHADER_BINARY_FORMAT_SPIR_V, bin, (GLsizei)n);
    glSpecializeShader(sh, "main", 0, NULL, NULL);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    return sh;
}

static GLuint spv_program(const unsigned char *vs, size_t vn)
{
    GLuint p = glCreateProgram();
    GLuint v = spv_shader(GL_VERTEX_SHADER, vs, vn);
    GLuint f = spv_shader(GL_FRAGMENT_SHADER, spv_pass_frag, sizeof spv_pass_frag);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

TEST(uniform_block_is_not_a_capture)
{
    uint64_t before = mgl_xfb_link_count;
    GLuint p = spv_program(spv_ubo_vert, sizeof spv_ubo_vert);
    EXPECT_EQ_INT(mgl_xfb_link_count - before, 0);
    GLint n = -1; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
    EXPECT_EQ_INT(n, 0);
    GLint blocks = 0; glGetProgramiv(p, GL_ACTIVE_UNIFORM_BLOCKS, &blocks);
    EXPECT_EQ_INT(blocks, 1);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

TEST(declared_capture_still_links)
{
    uint64_t before = mgl_xfb_link_count;
    GLuint p = spv_program(spv_xfb_vert, sizeof spv_xfb_vert);
    EXPECT_TRUE(mgl_xfb_link_count - before >= 1);
    GLint n = -1; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_VARYINGS, &n);
    EXPECT_EQ_INT(n, 1);
    GLint stride = 0; glGetProgramiv(p, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &stride);
    EXPECT_EQ_INT(stride, GL_INTERLEAVED_ATTRIBS);

    /* capture one point: gl_Position = pos, col = 2 * pos */
    float pt[4] = {0.25f, -0.5f, 0.75f, 1.0f};
    GLuint vao, vbo, tfb;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
    glGenBuffers(1, &tfb); glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 16, NULL, GL_STATIC_READ);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    float out[4] = {0};
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 16, out);
    for (int i = 0; i < 4; i++) EXPECT_NEAR(out[i], 2.0f * pt[i], 1e-5);
    glUseProgram(0);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(uniform_block_is_not_a_capture);
    RUN(declared_capture_still_links);
    return mgl_test_summary();
}
