/* test_compat_default_texture_nolevels — glmark2 [shadow]/[jellyfish] regression
 * (compat r104b, supervisor hold):
 *
 * A texture unit that had name 0 bound to it (glmark2's state cleanup) holds the
 * target's DEFAULT texture object (compat r103 Mesa parity, name 0) — which has
 * NO levels until someone uploads storage into it.  Drawing with a program whose
 * sampler reads such a unit used to send the level-less default object into
 * createMTLTextureFromGLTexture from BOTH the active_texture_mask pre-pass and
 * the sampler bind loop ("Assertion hit in MGLRenderer.m" + "emergency fallback
 * texture 0x0x0").  Such a texture is INCOMPLETE: the sample must be (0,0,0,1),
 * with no GL error and no Metal fallback noise.  A complete texture on the same
 * unit must still sample normally through the reordered bind path.
 */
#include "mgl_test.h"

static GLuint mkprog_sampler(void)
{
    const char *vs = "#version 450\n"
                     "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                     "void main(){ gl_Position = vec4(q[gl_VertexID],0,1); }\n";
    const char *fs = "#version 450\n"
                     "layout(location=0) out vec4 o;\n"
                     "uniform sampler2D s;\n"
                     "void main(){ o = texture(s, vec2(0.5)); }\n";
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    return p;
}

static void draw_and_read(GLuint fbo, const char *tag)
{
    GLubyte px[4] = { 9, 9, 9, 9 };
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();                       /* no assert fallback path, no error */
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NO_GL_ERROR();
    printf("  %-28s sampled rgba(%d,%d,%d,%d)\n", tag, px[0], px[1], px[2], px[3]);
    if (px[0] != 0 || px[1] != 0 || px[2] != 0)
        MGL_FAIL("expected a black sample (incomplete texture), got %d %d %d %d",
                 px[0], px[1], px[2], px[3]);
    /* the incomplete fallback carries GL 8.17's (0,0,0,1) */
    EXPECT_EQ_INT(px[3], 255);
    (void)fbo;
}

TEST(default_texture_levelless_unit_samples_black)
{
    GLuint prog, fbo, rb, vao;
    static const GLubyte white[8 * 8 * 4] = {[0 ... 8 * 8 * 4 - 1] = 255 };

    prog = mkprog_sampler();
    glUseProgram(prog);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 64, 64);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 64, 64);

    glUniform1i(glGetUniformLocation(prog, "s"), 1);   /* sampler on unit 1 */
    glActiveTexture(GL_TEXTURE1);

    /* 1. explicit bind of name 0 (Mesa parity: binds the target's default
     *    object) — the glmark2 shadow/jellyfish case: level-less, INCOMPLETE,
     *    samples (0,0,0,1) and never reaches the texture creator */
    glBindTexture(GL_TEXTURE_2D, 0);
    EXPECT_NO_GL_ERROR();
    draw_and_read(fbo, "bind-name-0");

    /* 2. a complete texture on the same unit still binds and samples WHITE
     *    through the reordered bind path */
    {
        GLuint t;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        EXPECT_NO_GL_ERROR();
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        EXPECT_NO_GL_ERROR();
        GLubyte px[4] = { 0, 0, 0, 0 };
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        EXPECT_NO_GL_ERROR();
        EXPECT_EQ_INT(px[0], 255);   /* white */
        EXPECT_EQ_INT(px[1], 255);
        EXPECT_EQ_INT(px[2], 255);
        EXPECT_EQ_INT(px[3], 255);
        printf("  %-28s sampled rgba(%d,%d,%d,%d)\n", "real-texture", px[0], px[1], px[2], px[3]);

        /* 3. unbind again: back to the level-less default object */
        glBindTexture(GL_TEXTURE_2D, 0);
        draw_and_read(fbo, "after-unbind");
        glDeleteTextures(1, &t);
    }

    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
}

int main(void)
{
    RUN(default_texture_levelless_unit_samples_black);
    return mgl_test_summary();
}
