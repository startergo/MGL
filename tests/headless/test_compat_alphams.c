/* compat s34: repro for KHR-GL46.texture_size_promotion.functional
 * "Promotion from internal format GL_RGB8I ... alpha channel ... target
 *  GL_TEXTURE_2D_MULTISAMPLE. Expected value = 1 read value = 0":
 * the alpha of an alpha-less texture must read 1 through the whole
 * 2D -> (render) -> 2DMS -> (texelFetch .a) chain. */
#include "mgl_test.h"

static GLuint make_prog(const char *fs)
{
    GLuint p = glCreateProgram();
    const char *vs = "#version 330 core\n"
                     "void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);"
                     " gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
    GLuint s1 = glCreateShader(GL_VERTEX_SHADER), s2 = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s1, 1, &vs, NULL); glCompileShader(s1);
    glShaderSource(s2, 1, &fs, NULL); glCompileShader(s2);
    glAttachShader(p, s1); glAttachShader(p, s2); glLinkProgram(p);
    glDeleteShader(s1); glDeleteShader(s2);
    return p;
}

TEST(alphaless_2d_ms_chain)
{
    /* 1. alpha-less 2D RGB8I texture with data */
    GLuint src = 0;
    glGenTextures(1, &src);
    glBindTexture(GL_TEXTURE_2D, src);
    GLint datai[4] = { -1, 2, -3, 4 };
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8I, 1, 1, 0, GL_RGB_INTEGER, GL_INT, datai);
    EXPECT_NO_GL_ERROR();

    /* 1a. sampling the alpha-less 2D texture directly must give 1 (r77 case) */
    {
        GLuint fbo = 0, dst = 0;
        glGenTextures(1, &dst);
        glBindTexture(GL_TEXTURE_2D, dst);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1, 1, 0, GL_RED_INTEGER, GL_INT, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
        GLuint prog = make_prog("#version 330 core\n"
                                "layout(location = 0) out ivec4 o;\n"
                                "uniform isampler2D t;\n"
                                "void main() { o = ivec4(texelFetch(t, ivec2(0), 0).a); }");
        GLuint vao = 0; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glViewport(0, 0, 1, 1);
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "t"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        GLint px = 5;
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_INT, &px);
        EXPECT_NO_GL_ERROR();
        EXPECT_EQ_INT(px, 1);
        glUseProgram(0);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(prog);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &dst);
    }

    /* 2. render the 2D texture's full vec4 into an RGB8I 2D-multisample texture */
    GLuint ms = 0;
    glGenTextures(1, &ms);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB8I, 1, 1, GL_TRUE);
    EXPECT_NO_GL_ERROR();
    {
        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, ms, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { MGL_FAIL("ms fbo incomplete"); return; }
        GLuint prog = make_prog("#version 330 core\n"
                                "#extension GL_ARB_texture_multisample : enable\n"
                                "layout(location = 0) out ivec4 o;\n"
                                "uniform isampler2D t;\n"
                                "void main() { o = texelFetch(t, ivec2(0), 0); }");
        GLuint vao = 0; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glBindTexture(GL_TEXTURE_2D, src);
        glViewport(0, 0, 1, 1);
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "t"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        glUseProgram(0);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(prog);
        glDeleteFramebuffers(1, &fbo);
    }

    /* 3. texelFetch the MS texture's alpha — must be 1 */
    {
        GLuint fbo = 0, dst = 0;
        glGenTextures(1, &dst);
        glBindTexture(GL_TEXTURE_2D, dst);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, 1, 1, 0, GL_RED_INTEGER, GL_INT, NULL);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dst, 0);
        GLuint prog = make_prog("#version 330 core\n"
                                "#extension GL_ARB_texture_multisample : enable\n"
                                "layout(location = 0) out ivec4 o;\n"
                                "uniform isampler2DMS t;\n"
                                "void main() { o = ivec4(texelFetch(t, ivec2(0), 0).a); }");
        GLuint vao = 0; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glViewport(0, 0, 1, 1);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
        glUseProgram(prog);
        glUniform1i(glGetUniformLocation(prog, "t"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_NO_GL_ERROR();
        GLint px = 5;
        glReadPixels(0, 0, 1, 1, GL_RED_INTEGER, GL_INT, &px);
        EXPECT_NO_GL_ERROR();
        EXPECT_EQ_INT(px, 1);
        glUseProgram(0);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(prog);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &dst);
    }

    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
    glDeleteTextures(1, &ms);
    glDeleteTextures(1, &src);
}

int main(void)
{
    RUN(alphaless_2d_ms_chain);
    return mgl_test_summary();
}
