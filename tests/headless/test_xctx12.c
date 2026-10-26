/* test_xctx12.c — compat: alternating programs across frames (jellyfish draws a
 * texture-less gradient quad, then a textured program; the next frame starts
 * with the gradient again).  Each draw must use ITS OWN program. */
#include "mgl_test.h"

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static const char *k_gvs =
    "attribute vec2 pos;\n"
    "varying float v;\n"
    "void main() { v = pos.y * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_gfs =
    "precision mediump float;\n"
    "varying float v;\n"
    "uniform vec3 color1;\n"
    "uniform vec3 color2;\n"
    "void main() { gl_FragColor = vec4(mix(color1, color2, v), 1.0); }\n";

static const char *k_tvs =
    "attribute vec2 pos;\n"
    "varying vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_tfs =
    "precision mediump float;\n"
    "uniform sampler2D uSampler;\n"
    "uniform sampler2D uSampler1;\n"
    "varying vec2 uv;\n"
    "void main() {\n"
    "    vec4 a = texture2D(uSampler, uv);\n"
    "    vec4 b = texture2D(uSampler1, uv * 3.0);\n"
    "    gl_FragColor = vec4(a.rgb * 0.5 + b.rgb * 0.5, 1.0);\n"
    "}\n";

static const float k_quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };
static const GLushort k_idx[6] = { 0, 1, 2, 0, 2, 3 };

TEST(alternating_programs)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLuint gprog, tprog, t0, t1, vao, vbo;
    GLint gpos, tpos, loc;
    GLubyte img[4 * 4 * 4], px[4];

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    for (int i = 0; i < 16; i++)
    { img[i * 4] = 255; img[i * 4 + 1] = 0; img[i * 4 + 2] = 0; img[i * 4 + 3] = 255; }
    glGenTextures(1, &t0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    for (int i = 0; i < 16; i++)
    { img[i * 4] = 0; img[i * 4 + 1] = 0; img[i * 4 + 2] = 255; img[i * 4 + 3] = 255; }
    glGenTextures(1, &t1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, t1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    gprog = mgl_make_program(k_gvs, k_gfs);
    tprog = mgl_make_program(k_tvs, k_tfs);
    EXPECT_TRUE(gprog && tprog);
    if (!gprog || !tprog) return;

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    { GLuint ibo; glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof k_idx, k_idx, GL_STATIC_DRAW); }

    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    gpos = glGetAttribLocation(gprog, "pos");
    tpos = glGetAttribLocation(tprog, "pos");

    for (int frame = 0; frame < 3; frame++)
    {
        /* 1. the gradient quad (no textures) */
        glUseProgram(gprog);
        glEnableVertexAttribArray((GLuint)gpos);
        glVertexAttribPointer((GLuint)gpos, 2, GL_FLOAT, GL_FALSE, 0, 0);
        loc = glGetUniformLocation(gprog, "color1"); if (loc >= 0) glUniform3f(loc, 0.2f, 0.4f, 0.9f);
        loc = glGetUniformLocation(gprog, "color2"); if (loc >= 0) glUniform3f(loc, 0.1f, 0.2f, 0.5f);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        while (glGetError() != GL_NO_ERROR) {}
        /* mid gradient = (0.15,0.3,0.7) — must NOT contain texture reds/blues */
        EXPECT_NEAR(px[0], 38, 8);
        EXPECT_NEAR(px[1], 77, 8);
        EXPECT_NEAR(px[2], 191, 8);   /* mix at v=0.375 */

        /* 2. the textured quad */
        glDisableVertexAttribArray((GLuint)gpos);
        glUseProgram(tprog);
        glEnableVertexAttribArray((GLuint)tpos);
        glVertexAttribPointer((GLuint)tpos, 2, GL_FLOAT, GL_FALSE, 0, 0);
        loc = glGetUniformLocation(tprog, "uSampler");  if (loc >= 0) glUniform1i(loc, 0);
        loc = glGetUniformLocation(tprog, "uSampler1"); if (loc >= 0) glUniform1i(loc, 1);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        while (glGetError() != GL_NO_ERROR) {}
        EXPECT_NEAR(px[0], 128, 8);   /* half red + half blue */
        EXPECT_NEAR(px[2], 128, 8);
        glDisableVertexAttribArray((GLuint)tpos);
    }

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(alternating_programs);
    return mgl_test_summary();
}
