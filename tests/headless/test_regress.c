/* Cross-domain regressions found during integration. */
#include "mgl_test.h"

/* glDeleteBuffers used the GL target enum as an index into the binding arrays
 * (out-of-bounds write) and left dangling pointers in VAOs and indexed bindings. */
TEST(delete_bound_buffers)
{
    GLuint vao, bufs[4];
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(4, bufs);
    static const float tri[] = { -1,-1, 3,-1, -1,3 };
    glBindBuffer(GL_ARRAY_BUFFER, bufs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    static const GLuint idx[] = { 0, 1, 2 };
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufs[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 256, NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, bufs[2]);
    glBindBuffer(GL_UNIFORM_BUFFER, bufs[3]);
    glBufferData(GL_UNIFORM_BUFFER, 256, NULL, GL_DYNAMIC_DRAW);
    glBindBufferRange(GL_UNIFORM_BUFFER, 1, bufs[3], 0, 64);
    EXPECT_NO_GL_ERROR();

    glDeleteBuffers(4, bufs);
    EXPECT_NO_GL_ERROR();

    GLint v = -1;
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &v);          EXPECT_EQ_INT(v, 0);
    glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &v);  EXPECT_EQ_INT(v, 0);
    glGetIntegerv(GL_SHADER_STORAGE_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, 0);
    glGetIntegerv(GL_UNIFORM_BUFFER_BINDING, &v);        EXPECT_EQ_INT(v, 0);
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 3, &v); EXPECT_EQ_INT(v, 0);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 1, &v);   EXPECT_EQ_INT(v, 0);
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &v); EXPECT_EQ_INT(v, 0);
    for (int i = 0; i < 4; i++) EXPECT_TRUE(!glIsBuffer(bufs[i]));
    EXPECT_NO_GL_ERROR();

    /* context still healthy: repeat create/delete churn, then clear + read back */
    for (int r = 0; r < 64; r++) {
        GLuint b; glGenBuffers(1, &b);
        glBindBuffer(r & 1 ? GL_SHADER_STORAGE_BUFFER : GL_ARRAY_BUFFER, b);
        glBufferData(r & 1 ? GL_SHADER_STORAGE_BUFFER : GL_ARRAY_BUFFER, 64 + r, NULL, GL_STREAM_DRAW);
        glDeleteBuffers(1, &b);
    }
    glDisableVertexAttribArray(0);
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_PIXEL(4, 4, 0, 255, 0, 255, 2);
    EXPECT_NO_GL_ERROR();
}

/* fb + rstate: default framebuffer size must seed every viewport/scissor index. */
TEST(default_viewport_scissor_seeded)
{
    GLint vp[4] = {0}, sc[4] = {0};
    glGetIntegerv(GL_VIEWPORT, vp);
    glGetIntegerv(GL_SCISSOR_BOX, sc);
    EXPECT_TRUE(vp[2] > 1 && vp[3] > 1);
    EXPECT_EQ_INT(sc[2], vp[2]);
    EXPECT_EQ_INT(sc[3], vp[3]);
    GLint vp5[4] = {0};
    glGetIntegeri_v(GL_VIEWPORT, 5, vp5);
    EXPECT_EQ_INT(vp5[2], vp[2]);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(default_viewport_scissor_seeded);
    RUN(delete_bound_buffers);
    return mgl_test_summary();
}
