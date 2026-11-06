/* test_compat_minfb — minimal 256x256 FBO quad render (sanity for imgcast2 setup) */
#include "mgl_test.h"

TEST(compat_minfb)
{
    GLuint vao = 0, vbo = 0, prog = 0, rfbo = 0, rtex = 0;
    static const float quad[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,  1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 0.0f, 1.0f,
    };
    const char *vs = "#version 460 core\nlayout(location=0) in vec4 p;\nlayout(r32ui, binding=0) coherent uniform uimage2D g_image;\nvoid main(){gl_Position=p;}\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 c;\nvoid main(){c=vec4(1.0);}\n";

    prog = glCreateProgram();
    {
        GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
        glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
        glAttachShader(prog, v); glAttachShader(prog, f);
        glLinkProgram(prog);
        GLint ok = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        EXPECT_TRUE(ok);
        glDeleteShader(v); glDeleteShader(f);
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, 0);

    glGenTextures(1, &rtex);
    glBindTexture(GL_TEXTURE_2D, rtex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 256);
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_FRAMEBUFFER, rfbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glUseProgram(prog);
    glViewport(0, 0, 256, 256);
    GLuint imgtex; glGenTextures(1,&imgtex); glBindTexture(GL_TEXTURE_2D,imgtex); glTexStorage2D(GL_TEXTURE_2D,1,GL_R32UI,4,4); glBindImageTexture(0,imgtex,0,GL_FALSE,0,GL_READ_WRITE,GL_R32UI); glBindTexture(GL_TEXTURE_2D,0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();

    GLubyte px[4] = {0};
    glReadPixels(128, 128, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_EQ_INT(px[0], 255);
    EXPECT_EQ_INT(px[3], 255);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &rtex);
    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(compat_minfb);
    return mgl_test_summary();
}
