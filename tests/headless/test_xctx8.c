/* test_xctx8.c — compat profile: interleaved multi-attribute VBO
 * (jellyfish mesh: position+normal+color+texcoord in one buffer; a wrong
 * stride/offset would scramble lighting and animation) */
#include "mgl_test.h"

extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

/* per-vertex: pos(3f) normal(3f) color(3f) uv(2f) = 11 floats */
static const char *k_vs =
    "attribute vec3 aVertexPosition;\n"
    "attribute vec3 aVertexNormal;\n"
    "attribute vec3 aVertexColor;\n"
    "attribute vec2 aTextureCoord;\n"
    "varying vec3 vC;\n"
    "varying vec2 vT;\n"
    "void main() {\n"
    "    vC = aVertexNormal * 0.5 + 0.5;   /* normals arrive in [-1,1] */\n"
    "    vT = aTextureCoord;\n"
    "    gl_Position = vec4(aVertexPosition.xy, 0.0, 1.0);\n"
    "}\n";
/* pack the normal map and uv product into RGBA; checked by readback */
static const char *k_fs2 =
    "precision mediump float;\n"
    "varying vec3 vC;\n"
    "varying vec2 vT;\n"
    "void main() {\n"
    "    gl_FragColor = vec4(vC, vT.x * vT.y);\n"
    "}\n";

static const float k_vert[3 * 11] = {
    /* pos          normal      color      uv */
    -1.f, -1.f, 0.f,  0.f, 0.f, 1.f,  0.1f, 0.2f, 0.3f,  0.25f, 0.25f,
     1.f, -1.f, 0.f,  0.f, 1.f, 0.f,  0.4f, 0.5f, 0.6f,  0.75f, 0.25f,
     0.f,  1.f, 0.f,  1.f, 0.f, 0.f,  0.7f, 0.8f, 0.9f,  0.5f,  0.75f,
};

TEST(compat_interleaved_attributes)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                      GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
    GLuint prog, vao, vbo;
    GLint lp, ln, lc, lt;
    GLubyte px[4];
    GLsizei stride = 11 * (GLsizei)sizeof(float);

    EXPECT_TRUE(c != NULL);
    if (!c) return;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);

    prog = mgl_make_program(k_vs, k_fs2);
    EXPECT_TRUE(prog);
    lp = glGetAttribLocation(prog, "aVertexPosition");
    ln = glGetAttribLocation(prog, "aVertexNormal");
    lc = glGetAttribLocation(prog, "aVertexColor");
    lt = glGetAttribLocation(prog, "aTextureCoord");
    EXPECT_TRUE(lp >= 0 && ln >= 0 && lc >= 0 && lt >= 0);
    fprintf(stderr, "  note: locs %d %d %d %d\n", lp, ln, lc, lt);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_vert, k_vert, GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)lp);
    glVertexAttribPointer((GLuint)lp, 3, GL_FLOAT, GL_FALSE, stride, (void *)(0 * sizeof(float)));
    glEnableVertexAttribArray((GLuint)ln);
    glVertexAttribPointer((GLuint)ln, 3, GL_FLOAT, GL_FALSE, stride, (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray((GLuint)lc);
    glVertexAttribPointer((GLuint)lc, 3, GL_FLOAT, GL_FALSE, stride, (void *)(6 * sizeof(float)));
    glEnableVertexAttribArray((GLuint)lt);
    glVertexAttribPointer((GLuint)lt, 2, GL_FLOAT, GL_FALSE, stride, (void *)(9 * sizeof(float)));

    glUseProgram(prog);
    glViewport(0, 0, 4, 4);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    while (glGetError() != GL_NO_ERROR) {}

    /* read pixel (1,1); compute the same barycentric interpolation in C and
     * compare every channel exactly */
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    fprintf(stderr, "  note: px = %d,%d,%d,%d\n", px[0], px[1], px[2], px[3]);
    {
        /* pixel centre in NDC */
        float cx = (1.5f / 4.0f) * 2.0f - 1.0f, cy = (1.5f / 4.0f) * 2.0f - 1.0f;
        float ax = -1, ay = -1, bx = 1, by = -1, cxv = 0, cyv = 1;
        float v0x = bx - ax, v0y = by - ay, v1x = cxv - ax, v1y = cyv - ay;
        float v2x = cx - ax, v2y = cy - ay;
        float d00 = v0x * v0x + v0y * v0y, d01 = v0x * v1x + v0y * v1y;
        float d11 = v1x * v1x + v1y * v1y, d20 = v2x * v0x + v2y * v0y;
        float d21 = v2x * v1x + v2y * v1y;
        float den = d00 * d11 - d01 * d01;
        /* P = A + u*(B-A) + v*(C-A): v multiplies (C-A), u multiplies (B-A) */
        float wC = (d00 * d21 - d01 * d20) / den;   /* weight of vertex C */
        float wB = (d11 * d20 - d01 * d21) / den;   /* weight of vertex B */
        float wA = 1.0f - wC - wB;
        float nx = (0 * wA + 0 * wB + 1 * wC) * 0.5f + 0.5f;
        float ny = (0 * wA + 1 * wB + 0 * wC) * 0.5f + 0.5f;
        float nz = (1 * wA + 0 * wB + 0 * wC) * 0.5f + 0.5f;
        float tx = 0.25f * wA + 0.75f * wB + 0.5f * wC;
        float ty = 0.25f * wA + 0.25f * wB + 0.75f * wC;
        fprintf(stderr, "  note: ref = %.1f,%.1f,%.1f alpha=%.1f\n",
                nx * 255, ny * 255, nz * 255, tx * ty * 255);
        EXPECT_NEAR(px[0], nx * 255, 4);
        EXPECT_NEAR(px[1], ny * 255, 4);
        EXPECT_NEAR(px[2], nz * 255, 4);
        EXPECT_NEAR(px[3], tx * ty * 255, 4);
    }

    destroyGLMContext(c);
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(compat_interleaved_attributes);
    return mgl_test_summary();
}
