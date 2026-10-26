/* test_xctx4.c — compat profile: two samplers on two texture units
 * (glmark2 jellyfish: unit0 body RGBA + unit1 caustics RGB; wrong unit
 * resolution would brighten the jellyfish body) */
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
    "varying vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "precision mediump float;\n"
    "uniform sampler2D uSampler;\n"
    "uniform sampler2D uSampler1;\n"
    "varying vec2 uv;\n"
    "void main() {\n"
    "    vec4 a = texture2D(uSampler, uv);\n"
    "    vec4 b = texture2D(uSampler1, uv);\n"
    "    gl_FragColor = vec4(a.rgb * 0.5 + b.rgb * 0.5, 1.0);\n"
    "}\n";
static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };

TEST(compat_two_units)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLubyte px[4];
    GLubyte body[4*4*4], caust[4*4*3];
    GLuint t0 = 0, t1 = 0, prog = 0, vao = 0, vbo = 0;
    GLint loc, apos;

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    for (int i = 0; i < 16; i++) { body[i*4]=255; body[i*4+1]=0;   body[i*4+2]=0;   body[i*4+3]=128; }
    for (int i = 0; i < 16; i++) { caust[i*3]=0;     caust[i*3+1]=0; caust[i*3+2]=255; }

    glGenTextures(1, &t0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, body);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &t1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, t1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, caust);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    /* the glmark2 pattern: bind 0 on a unit between frames */
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, t1);

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
    loc = glGetUniformLocation(prog, "uSampler");  if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(prog, "uSampler1"); if (loc >= 0) glUniform1i(loc, 1);

    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    mgl_read_rgba(1, 1, px);
    while (glGetError() != GL_NO_ERROR) {}

    /* half red (unit 0) + half blue (unit 1) = purple; green stays 0 */
    EXPECT_NEAR(px[0], 128, 10);
    EXPECT_NEAR(px[1], 0, 10);
    EXPECT_NEAR(px[2], 128, 10);

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_two_units);
    return mgl_test_summary();
}
