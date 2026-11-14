/* test_compat_ubo_high_binding — r111 Task A: rsvga's D3D9 translation places
 * its fragment constant block at binding 256 (the D3D9 PS constant space after
 * the VS's).  GL uniform-buffer BINDING POINTS are pure indirection (the Metal
 * buffer slot is per linked program), so MGL advertises
 * MGL_MAX_UNIFORM_BUFFER_BINDINGS = 512 and a block at binding 256 must
 * actually RECEIVE ITS DATA in a draw (RED before the limit raise:
 * glBindBufferBase(GL_UNIFORM_BUFFER, 256, ...) raised GL_INVALID_VALUE
 * because 256 >= the reported 84).
 */
#include "mgl_test.h"

static GLuint mkframe(GLuint *ubo_out, GLuint *fbo_out, GLuint *col_out)
{
    const char *vs = "#version 460 core\n"
                     "const vec2 q[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));\n"
                     "void main(){ gl_Position = vec4(q[gl_VertexID], 0, 1); }\n";
    /* the fragment colour comes from a uniform block at binding 256 */
    const char *fs = "#version 460 core\n"
                     "layout(binding = 256) uniform C { vec4 colour; };\n"
                     "layout(location = 0) out vec4 o;\n"
                     "void main(){ o = colour; }\n";
    char log[512];
    GLint ok = 0;
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(f, sizeof log, NULL, log); MGL_FAIL("fs: %s", log); }
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }

    GLuint ubo;
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, (const float[4]){ 0, 1, 0, 1 }, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 256, ubo);   /* RED pre-fix: 0x501 */
    EXPECT_NO_GL_ERROR();

    GLuint col, fbo, vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenTextures(1, &col);
    glBindTexture(GL_TEXTURE_2D, col);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    EXPECT_NO_GL_ERROR();
    *ubo_out = ubo;
    *fbo_out = fbo;
    *col_out = col;
    return p;
}

TEST(ubo_binding_256_receives_data)
{
    GLint lim = 0;
    glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &lim);
    printf("  GL_MAX_UNIFORM_BUFFER_BINDINGS = %d\n", lim);
    EXPECT_TRUE(lim >= 256);

    GLuint ubo, fbo, col;
    GLuint p = mkframe(&ubo, &fbo, &col);

    GLubyte px[4] = { 0, 0, 0, 0 };
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("  sampled rgba(%d,%d,%d,%d)\n", px[0], px[1], px[2], px[3]);
    EXPECT_EQ_INT(px[1], 255);          /* the block's colour reached the draw */
    EXPECT_EQ_INT(px[3], 255);

    /* binding 511 (the last point) works, 512 errors */
    while (glGetError()) {}
    glBindBufferBase(GL_UNIFORM_BUFFER, 511, ubo);
    EXPECT_EQ_INT(glGetError(), GL_NO_ERROR);
    glBindBufferBase(GL_UNIFORM_BUFFER, 512, ubo);
    EXPECT_EQ_INT(glGetError(), GL_INVALID_VALUE);

    glDeleteProgram(p);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &col);
    glDeleteBuffers(1, &ubo);
}
int main(void){ RUN(ubo_binding_256_receives_data); return mgl_test_summary(); }
