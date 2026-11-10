/* test_tex1da_3d — GL 4.6 §8.5.1: 1D array textures are allocated and updated
 * through the *3D* texture commands (glTexImage3D/glTexStorage3D with target
 * GL_TEXTURE_1D_ARRAY: width = elements, height = layers, depth 1;
 * glTexSubImage3D: x = element, y = layer, z = 0).  MGL previously accepted
 * only the 2D spelling here and returned GL_INVALID_ENUM for the spec'd 3D one
 * (found by compat r104 while writing the multi_bind regression test).
 * The internal layout is identical (height = layers), so both spellings now
 * work; this test exercises the 3D spelling end-to-end through a shader.
 */
#include "mgl_test.h"

static GLuint mkprog(const char *fs_body)
{
    const char *vs = "#version 450\n"
                     "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                     "out vec2 v;\n"
                     "void main(){ v = q[gl_VertexID]; gl_Position = vec4(q[gl_VertexID],0,1); }\n";
    const char *fs_fmt = "#version 450\n"
                         "in vec2 v;\n"
                         "layout(location=0) out vec4 o;\n"
                         "uniform sampler1DArray s;\n"
                         "void main(){ %s }\n";
    char fs[512];
    snprintf(fs, sizeof fs, fs_fmt, fs_body);
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &(const char *){ fs }, NULL); glCompileShader(f);
    GLint ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    EXPECT_TRUE(ok);
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_TRUE(ok);
    return p;
}

TEST(tex1da_3d_spelling)
{
    /* storage + per-layer upload through the 3D commands */
    GLuint tex, prog, fbo, rb;
    GLubyte rows[4][16];
    for (int l = 0; l < 4; l++)
        for (int x = 0; x < 16; x++) rows[l][x] = (GLubyte)(l * 60 + 10 + x);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_1D_ARRAY, tex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexStorage3D(GL_TEXTURE_1D_ARRAY, 1, GL_R8, 16, 4, 1);
    EXPECT_NO_GL_ERROR();
    for (int l = 0; l < 4; l++)
    {
        glTexSubImage3D(GL_TEXTURE_1D_ARRAY, 0, 0, l, 0, 16, 1, 1, GL_RED, GL_UNSIGNED_BYTE, rows[l]);
        EXPECT_NO_GL_ERROR();
    }
    /* level-0 dims through the GL query */
    GLint w = -1, h = -1;
    glGetTexLevelParameteriv(GL_TEXTURE_1D_ARRAY, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_1D_ARRAY, 0, GL_TEXTURE_HEIGHT, &h);
    EXPECT_EQ_INT(w, 16);
    EXPECT_EQ_INT(h, 4);            /* height = layers for a 1D array */

    prog = mkprog("o = vec4(texelFetch(s, ivec2(int((v.x+1.0)*8.0), int((v.y+1.0)*2.0)), 0).r);");
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "s"), 0);

    /* offscreen FB + a VAO (MGL core contexts require a bound VAO to draw) */
    GLuint vao;
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
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();

    /* v=(0,0) centre -> texel (8,2): rows[2][8] = 2*60+10+8 = 138 */
    GLubyte px[4] = { 0, 0, 0, 0 };
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NEAR(px[0], 138, 1);

    /* the glTexImage3D spelling (mutable texture, full-image upload, layers as rows) */
    GLubyte full[4][16];
    for (int l = 0; l < 4; l++)
        for (int x = 0; x < 16; x++) full[l][x] = (GLubyte)(200 + l);

    /* depth != 1 is GL_INVALID_VALUE for a 1D array (GL 4.6 §8.5.1) */
    glTexImage3D(GL_TEXTURE_1D_ARRAY, 0, GL_R8, 16, 4, 2, 0, GL_RED, GL_UNSIGNED_BYTE, full);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glTexStorage3D(GL_TEXTURE_1D_ARRAY, 1, GL_R8, 16, 4, 2);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteTextures(1, &tex);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_1D_ARRAY, tex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage3D(GL_TEXTURE_1D_ARRAY, 0, GL_R8, 16, 4, 1, 0, GL_RED, GL_UNSIGNED_BYTE, full);
    EXPECT_NO_GL_ERROR();
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_NEAR(px[0], 202, 1);

    /* DSA: glTextureStorage3D on a 1D-array texture object */
    GLuint dsa;
    glGenTextures(1, &dsa);
    glBindTexture(GL_TEXTURE_1D_ARRAY, dsa);
    glTextureStorage3D(dsa, 1, GL_R8, 8, 2, 1);
    EXPECT_NO_GL_ERROR();

    /* negative: depth > 1 is invalid for a 1D array (texture is 2D-layout) */
    glTexSubImage3D(GL_TEXTURE_1D_ARRAY, 0, 0, 0, 1, 16, 1, 1, GL_RED, GL_UNSIGNED_BYTE, rows[0]);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    glDeleteTextures(1, &tex);
    glDeleteTextures(1, &dsa);
    glDeleteProgram(prog);
    glDeleteVertexArrays(1, &vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

int main(void)
{
    RUN(tex1da_3d_spelling);
    return mgl_test_summary();
}
