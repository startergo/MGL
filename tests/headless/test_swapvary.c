#include "mgl_test.h"
/* Does MGL line up VS outputs with FS inputs by NAME (GL link semantics) when the two
 * stages declare the varyings in different orders?  glslang's default I/O resolver
 * assigns locations per stage sequentially, so the orders must not matter here. */
TEST(swap_order_varyings)
{
    static const char *vs =
        "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "out vec4 v_red;\n"
        "out vec4 v_green;\n"
        "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_red = vec4(1.0, 0.0, 0.0, 1.0); v_green = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    /* note: declares the varyings in the OPPOSITE order to the vertex shader */
    static const char *fs =
        "#version 450 core\n"
        "in vec4 v_green;\n"
        "in vec4 v_red;\n"
        "layout(location=0) out vec4 color;\n"
        "void main(){ color = v_red; }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint vao, vbo, fbo, tex;
    const float quad[] = { -1, -1, 3, -1, -1, 3 };
    GLint ok = 0;

    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok == GL_TRUE);
    if (ok != GL_TRUE) return;
    glUseProgram(p);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 64, 64);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(swap_order_varyings);
    return mgl_test_summary();
}
