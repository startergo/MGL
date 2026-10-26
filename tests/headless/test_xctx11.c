/* test_xctx11.c — real glmark2 jellyfish.vert uniform-buffer layout on the
 * compat path: several mat4 + float uniforms must reach the GPU in their own
 * slots (CPU readback is not enough; the caustic UV comes from uWorld) */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static const char *k_fs =
    "precision mediump float;\n"
    "varying vec4 vWorld;\n"
    "void main() { gl_FragColor = vec4(vWorld.x / 24.0, (vWorld.z - vWorld.y) / 48.0, 0.0, 1.0); }\n";

/* quad in clip space; aVertex* fed as four-attribute interleaved data */
static const float k_vert[4 * 11] = {
    /* pos(3)      normal(3)    color(3)     uv(2) */
    -1.f, -1.f, 0.f,  0.f, 0.f, 1.f,  1.f, 0.f, 0.f,  0.f, 0.f,
     1.f, -1.f, 0.f,  0.f, 1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f,
     1.f,  1.f, 0.f,  1.f, 0.f, 0.f,  0.f, 0.f, 1.f,  1.f, 1.f,
    -1.f,  1.f, 0.f,  0.f, 1.f, 1.f,  1.f, 1.f, 0.f,  0.f, 1.f,
};
static const GLushort k_idx[6] = { 0, 1, 2, 0, 2, 3 };

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

TEST(jellyfish_vert_uniforms_gpu)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    char *vsrc = slurp("/opt/homebrew/share/glmark2/shaders/jellyfish.vert");
    GLuint prog, vs, fs, vao, vbo, ibo;
    GLint lp, ln, lc, lt, lu[10];
    GLfloat uWorld[16], uVP[16], uWIT[16];
    float t = 1.7f;
    GLubyte px[4];

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);
    if (!vsrc) { fprintf(stderr, "  note: jellyfish.vert missing, skip\n"); return; }

    vs = compile1(GL_VERTEX_SHADER, vsrc);
    fs = compile1(GL_FRAGMENT_SHADER, k_fs);
    prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    { GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok); EXPECT_TRUE(ok); }

    lp = glGetAttribLocation(prog, "aVertexPosition");
    ln = glGetAttribLocation(prog, "aVertexNormal");
    lc = glGetAttribLocation(prog, "aVertexColor");
    lt = glGetAttribLocation(prog, "aTextureCoord");
    EXPECT_TRUE(lp >= 0 && ln >= 0 && lc >= 0 && lt >= 0);

    /* uWorld: scale (3, 5, 2) — makes vWorld large enough to expose slot mixups */
    for (int i = 0; i < 16; i++) uWorld[i] = 0.0f;
    uWorld[0] = 3.0f; uWorld[5] = 5.0f; uWorld[10] = 2.0f; uWorld[15] = 1.0f;
    for (int i = 0; i < 16; i++) uVP[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < 16; i++) uWIT[i] = (i % 5 == 0) ? 1.0f : 0.0f;

    glUseProgram(prog);
    lu[0] = glGetUniformLocation(prog, "uWorld");          if (lu[0] >= 0) glUniformMatrix4fv(lu[0], 1, GL_FALSE, uWorld);
    lu[1] = glGetUniformLocation(prog, "uWorldViewProj");  if (lu[1] >= 0) glUniformMatrix4fv(lu[1], 1, GL_FALSE, uVP);
    lu[2] = glGetUniformLocation(prog, "uWorldInvTranspose"); if (lu[2] >= 0) glUniformMatrix4fv(lu[2], 1, GL_FALSE, uWIT);
    lu[3] = glGetUniformLocation(prog, "uLightPos");       if (lu[3] >= 0) glUniform3f(lu[3], 3.f, 5.f, 10.f);
    lu[4] = glGetUniformLocation(prog, "uLightRadius");    if (lu[4] >= 0) glUniform1f(lu[4], 10.f);
    lu[5] = glGetUniformLocation(prog, "uLightCol");       if (lu[5] >= 0) glUniform4f(lu[5], 0.1f, 0.2f, 0.3f, 1.f);
    lu[6] = glGetUniformLocation(prog, "uAmbientCol");     if (lu[6] >= 0) glUniform4f(lu[6], 0.1f, 0.1f, 0.1f, 1.f);
    lu[7] = glGetUniformLocation(prog, "uFresnelPower");   if (lu[7] >= 0) glUniform1f(lu[7], 1.f);
    lu[8] = glGetUniformLocation(prog, "uFresnelCol");     if (lu[8] >= 0) glUniform4f(lu[8], 0.2f, 0.2f, 0.2f, 1.f);
    lu[9] = glGetUniformLocation(prog, "uCurrentTime");    if (lu[9] >= 0) glUniform1f(lu[9], t);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_vert, k_vert, GL_STATIC_DRAW);
    {
        GLsizei st = 11 * (GLsizei)sizeof(float);
        glEnableVertexAttribArray((GLuint)lp);
        glVertexAttribPointer((GLuint)lp, 3, GL_FLOAT, GL_FALSE, st, (void *)(0 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)ln);
        glVertexAttribPointer((GLuint)ln, 3, GL_FLOAT, GL_FALSE, st, (void *)(3 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)lc);
        glVertexAttribPointer((GLuint)lc, 3, GL_FLOAT, GL_FALSE, st, (void *)(6 * sizeof(float)));
        glEnableVertexAttribArray((GLuint)lt);
        glVertexAttribPointer((GLuint)lt, 2, GL_FLOAT, GL_FALSE, st, (void *)(9 * sizeof(float)));
    }
    glGenBuffers(1, &ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof k_idx, k_idx, GL_STATIC_DRAW);

    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, NULL);
    { GLenum e = glGetError(); if (e) MGL_FAIL("draw error 0x%x", e); }
    while (glGetError() != GL_NO_ERROR) {}
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "  note: px = %d,%d,%d,%d\n", px[0], px[1], px[2], px[3]);

    /* reference at pixel (1,1) = NDC (-0.25,-0.25): interpolated worldPos.x =
     * -0.25 -> vWorld.x = 3 * -0.25 = -0.75 -> /24 < 0 -> clamps to 0; and
     * vWorld.z - vWorld.y = 2*0 - 5*(-0.25) = 1.25 -> /48*255 = 6.6.  A
     * mixed-up matrix slot (identity or zero) gives 0 for the second channel. */
    EXPECT_NEAR(px[0], 0, 3);
    EXPECT_NEAR(px[1], 7, 3);

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(jellyfish_vert_uniforms_gpu);
    return mgl_test_summary();
}
