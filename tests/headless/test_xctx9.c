/* test_xctx9.c — compat profile, real glmark2 jellyfish program: rotating
 * caustic textures on unit 1 + per-frame uCurrentTime updates (the caustic
 * light web over the bell was missing/pale in glmark2).  Each "frame" must
 * sample exactly the caustic texture bound for that frame. */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned type2,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n; char *s;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    s = (char *)calloc(1, n + 1);
    if (fread(s, 1, n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    fclose(f);
    return s;
}

static GLuint compile1(GLenum type, const char *src)
{
    GLuint sh = glCreateShader(type);
    GLint ok; char log[8192];
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { glGetShaderInfoLog(sh, sizeof log, NULL, log); fprintf(stderr, "compile %x: %s\n", type, log); }
    EXPECT_TRUE(ok);
    return sh;
}

/* the vertex shader outputs the varyings the frag needs; we feed a fixed quad
 * and verify the FRAG's caustic sample directly with a trimmed program: same
 * uniform set, same texture math */
static const char *k_vs =
    "attribute vec2 pos;\n"
    "varying vec4 vWorld;\n"
    "void main() { vWorld = vec4(pos * 8.0, 1.0, 1.0); gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "precision mediump float;\n"
    "uniform sampler2D uSampler;\n"
    "uniform sampler2D uSampler1;\n"
    "uniform float uCurrentTime;\n"
    "varying vec4 vWorld;\n"
    "void main() {\n"
    "    vec4 caustics = texture2D(uSampler1, vec2(vWorld.x / 24.0 + uCurrentTime / 20.0,\n"
    "                                              (vWorld.z - vWorld.y) / 48.0 + uCurrentTime / 17.0));\n"
    "    gl_FragColor = vec4(caustics.rgb, 1.0);\n"
    "}\n";

TEST(jellyfish_caustics_rotation)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLuint tex[40], prog = 0, vao = 0, vbo = 0;
    GLint loc, apos;
    const float quad[12] = { -1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1 };
    int bad = 0;

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    /* 40 textures: unit-0 body (name 1) + 33 caustic frames (names 2..34),
     * each frame a flat distinct colour */
    GLubyte img[4 * 4 * 4];
    for (int i = 0; i < 40; i++) tex[i] = 0;
    glGenTextures(40, tex);
    for (int i = 0; i < 40; i++)
    {
        for (int p = 0; p < 16; p++)
        {
            img[p * 4 + 0] = (GLubyte)(i == 0 ? 20 : 10 + i * 6);
            img[p * 4 + 1] = (GLubyte)(i == 0 ? 30 : 5 + i * 3);
            img[p * 4 + 2] = (GLubyte)(i == 0 ? 40 : 200 - i * 2);
            img[p * 4 + 3] = 255;
        }
        glActiveTexture(GL_TEXTURE0 + (i == 0 ? 0 : 1));
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    while (glGetError() != GL_NO_ERROR) {}

    prog = mgl_make_program(k_vs, k_fs);
    EXPECT_TRUE(prog);
    if (!prog) return;
    apos = glGetAttribLocation(prog, "pos");
    EXPECT_TRUE(apos >= 0);
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)apos);
    glVertexAttribPointer((GLuint)apos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(prog);
    loc = glGetUniformLocation(prog, "uSampler");  if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(prog, "uSampler1"); if (loc >= 0) glUniform1i(loc, 1);
    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);

    /* 33 frames: rotate the caustic texture across the boundary names
     * (31/32/33) and advance uCurrentTime each frame */
    for (int f = 0; f < 33; f++)
    {
        int i = 1 + (f % 33);                 /* caustic frame index (texture tex[i]) */
        float t = (float)f / 10.0f;
        GLubyte px[4];
        loc = glGetUniformLocation(prog, "uCurrentTime");
        if (loc >= 0) glUniform1f(loc, t);
        glActiveTexture(GL_TEXTURE0);     glBindTexture(GL_TEXTURE_2D, tex[0]);
        glActiveTexture(GL_TEXTURE1);     glBindTexture(GL_TEXTURE_2D, tex[i]);
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        while (glGetError() != GL_NO_ERROR) {}
        glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        if (abs(px[0] - (10 + i * 6)) > 4 || abs(px[2] - (200 - i * 2)) > 4)
        {
            bad++;
            fprintf(stderr, "  note: frame %d sampled tex[%d] (name %u): got %d,%d,%d want %d,%d,%d\n",
                    f, i, tex[i], px[0], px[1], px[2], 10 + i * 6, 5 + i * 3, 200 - i * 2);
        }
    }
    EXPECT_EQ_INT(bad, 0);

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(jellyfish_caustics_rotation);
    return mgl_test_summary();
}
