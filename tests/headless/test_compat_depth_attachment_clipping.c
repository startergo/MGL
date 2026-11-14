/* test_compat_depth_attachment_clipping — r110 (rsvga follow-up): coverage must
 * not depend on whether a DEPTH attachment is bound when GL_DEPTH_TEST is off.
 * The twin's data: positions (±1,±1,0.5,1), matrix rows c303=(2,0,0,0),
 * c304=(0,-2,0,0), c305=(0,0,1,0), c306=(-1,1,0,1) — so w = -x+y+1 and the
 * (1,-1) vertex gets w = -1 <= 0: GL 4.6 13.7 clips it against the near plane
 * (-w <= z fails), and near/far clipping applies WHATEVER the depth-attachment
 * state (only GL_DEPTH_CLAMP disables it).  MGL measured 15/16 without the
 * attachment vs 14/16 with it — the no-attachment path failed to near-clip.
 */
#include "mgl_test.h"

static const float kPos[4][4] = {
    { -1, -1, 0.5f, 1 }, { 1, -1, 0.5f, 1 }, { 1, 1, 0.5f, 1 }, { -1, 1, 0.5f, 1 },
};

static GLuint mkframe(GLenum depth_ifmt, GLuint *depth_tex, GLuint *col_tex)
{
    const char *vs = "#version 460 core\n"
                     "layout(binding = 0) uniform C { vec4 r[308]; };\n"
                     "layout(location = 0) in vec4 a_pos;\n"
                     "layout(location = 0) out vec4 v_col;\n"
                     "layout(location = 2) out vec4 v_t2;\n"
                     "layout(location = 19) out vec4 v_t19;\n"
                     "void main()\n"
                     "{\n"
                     "    v_col = r[307];\n"
                     "    v_t2 = r[1];\n"
                     "    v_t19 = r[2];\n"
                     "    gl_Position = vec4(dot(a_pos, r[303]), dot(a_pos, r[304]),\n"
                     "                       dot(a_pos, r[305]), dot(a_pos, r[306]));\n"
                     "}\n";
    const char *fs = "#version 460 core\n"
                     "layout(location = 0) in vec4 v_col;\n"
                     "layout(location = 0) out vec4 o;\n"
                     "void main(){ o = v_col; }\n";
    char log[512];
    GLint ok = 0;
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, NULL); glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(v, sizeof log, NULL, log); MGL_FAIL("vs: %s", log); }
    glShaderSource(f, 1, &fs, NULL); glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(f, sizeof log, NULL, log); MGL_FAIL("fs: %s", log); }
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { glGetProgramInfoLog(p, sizeof log, NULL, log); MGL_FAIL("link: %s", log); }

    float c[308][4] = { 0 };
    c[303][0] = 2; c[304][1] = -2; c[305][2] = 1;
    c[306][0] = -1; c[306][1] = 1; c[306][3] = 1;
    c[307][0] = 1; c[307][3] = 1;

    GLuint vao, vbo, ubo, fbo, col;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof kPos, kPos, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, (void *) 0);
    glGenBuffers(1, &ubo);
    glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, sizeof c, c, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);

    glGenTextures(1, &col);
    glBindTexture(GL_TEXTURE_2D, col);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
    *depth_tex = 0;
    if (depth_ifmt)
    {
        glGenTextures(1, depth_tex);
        glBindTexture(GL_TEXTURE_2D, *depth_tex);
        if (depth_ifmt == GL_DEPTH24_STENCIL8)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, 4, 4, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, depth_ifmt, 4, 4, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, *depth_tex, 0);
    }
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    glViewport(0, 0, 4, 4);          /* the twin's 0.5-offset viewport maps the
                                      * same pixels for a 4x4 target */
    glDisable(GL_DEPTH_TEST);
    glDepthRange(0.0, 1.0);
    if (depth_ifmt)
        glClearDepth(1.0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | (depth_ifmt ? GL_DEPTH_BUFFER_BIT : 0));
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    EXPECT_NO_GL_ERROR();
    *col_tex = col;
    return fbo;
}

static int covered(GLuint fbo)
{
    GLubyte px[16 * 4];
    int n = 0;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px);
    for (int i = 0; i < 16; i++)
        if (px[i * 4] > 128) n++;
    return n;
}

TEST(depth_attachment_does_not_change_coverage)
{
    GLuint d1, d2, c1, c2;
    GLuint f0 = mkframe(0, &d1, &c1);
    GLuint f1 = mkframe(GL_DEPTH_COMPONENT16, &d1, &c1);
    GLuint f2 = mkframe(GL_DEPTH24_STENCIL8, &d2, &c2);

    int n0 = covered(f0);
    int n1 = covered(f1);
    int n2 = covered(f2);
    printf("  covered: no-depth %d/16, D16 %d/16, D24S8 %d/16\n", n0, n1, n2);

    /* GL 4.6 13.7: near/far (and w<=0) clipping applies regardless of the
     * depth attachment — all three must agree */
    EXPECT_EQ_INT(n1, n0);
    EXPECT_EQ_INT(n2, n0);

    glDeleteFramebuffers(1, &f0);
    glDeleteFramebuffers(1, &f1);
    glDeleteFramebuffers(1, &f2);
    if (d1) glDeleteTextures(1, &d1);
    if (d2) glDeleteTextures(1, &d2);
    glDeleteTextures(1, &c1);
    glDeleteTextures(1, &c2);
}
int main(void){ RUN(depth_attachment_does_not_change_coverage); return mgl_test_summary(); }
