/* test_xctx7.c — compat profile: mat4 uniform reaches the GPU intact
 * (jellyfish lighting matrices; write/readback already verified) */
#include "mgl_test.h"

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static const char *k_vs =
    "attribute vec2 pos;\n"
    "uniform mat4 uM;\n"
    "uniform bool uUseM;\n"
    "void main() { gl_Position = uUseM ? (uM * vec4(pos, 0.0, 1.0)) : vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "precision mediump float;\n"
    "void main() { gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0); }\n";
static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

static int coverage(GLuint prog, const GLfloat m[16])
{
    GLubyte px[4];
    int x, y, hit = 0;
    GLint loc = glGetUniformLocation(prog, "uM");
    if (loc >= 0) glUniformMatrix4fv(loc, 1, GL_FALSE, m);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++)
        {
            glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            if (px[1] > 100) hit++;
        }
    while (glGetError() != GL_NO_ERROR) {}
    return hit;
}

TEST(compat_mat4_uniform_gpu)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLuint prog, vao, vbo;
    GLint apos;
    GLfloat m[16];
    int full, half, trans, nom;

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    prog = mgl_make_program(k_vs, k_fs);
    EXPECT_TRUE(prog);
    apos = glGetAttribLocation(prog, "pos");
    EXPECT_TRUE(apos >= 0);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)apos);
    glVertexAttribPointer((GLuint)apos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    glViewport(0, 0, 16, 16);
    glDisable(GL_BLEND);

    /* without the matrix: all 256 pixels */
    { GLint u = glGetUniformLocation(prog, "uUseM"); if (u >= 0) glUniform1i(u, 0); }
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    nom = coverage(prog, m);
    EXPECT_EQ_INT(nom, 256);
    { GLint u = glGetUniformLocation(prog, "uUseM"); if (u >= 0) glUniform1i(u, 1); }

    /* identity: all 256 pixels */
    for (int i = 0; i < 16; i++) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    full = coverage(prog, m);
    EXPECT_EQ_INT(full, 256);

    /* scale 0.5: exactly 64 pixels (8x8), centred */
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = 0.5f; m[5] = 0.5f; m[10] = 1.0f; m[15] = 1.0f;
    half = coverage(prog, m);
    EXPECT_EQ_INT(half, 64);

    /* translation 0.75,0.75 (last column): quad moves up-right and clips */
    for (int i = 0; i < 16; i++) m[i] = 0.0f;
    m[0] = 1.0f; m[5] = 1.0f; m[10] = 1.0f; m[12] = 0.75f; m[13] = 0.75f; m[15] = 1.0f;
    trans = coverage(prog, m);
    /* x in [-0.25, 1.75], clipped to [-1,1]: pixels 6..15 (10 cols) = 100 */
    EXPECT_EQ_INT(trans, 100);

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_mat4_uniform_gpu);
    return mgl_test_summary();
}
