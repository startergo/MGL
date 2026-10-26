/*
 * test_xctx.c — cross-context isolation (glmark2 bug, session 14).
 *
 * glmark2 destroys and recreates the GL context for every scene and GL object
 * names restart at 1.  Nothing of context A may be observable in context B:
 * textures, shaders/programs, VAOs and per-draw caches must all be per-context.
 * This test runs a full "scene" in context A (jellyfish-like: >32 textures so
 * the name-indexed tables must grow), destroys it, runs a second scene in a
 * fresh context B and pixel-checks B's output.
 */
#include "mgl_test.h"

/* libmgl internal context management (see glm_context.h / MGLRenderer.m) */
extern void *createGLMContextProfile(unsigned format, unsigned type,
                                     unsigned depth_format, unsigned depth_type,
                                     unsigned stencil_format, unsigned stencil_type,
                                     unsigned profile);
extern void destroyGLMContext(void *ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);

static void *make_ctx(void)
{
    void *c = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                      GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                      GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE, 0);
    if (!c) return NULL;
    CppCreateMGLRendererHeadless(c);
    MGLsetCurrentContext(c);
    return c;
}

static const char *k_vs =
    "#version 330\n"
    "layout(location=0) in vec2 pos;\n"
    "out vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *k_fs =
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec2 uv;\n"
    "out vec4 o;\n"
    "void main() { o = texture(tex, uv); }\n";

static const float k_quad[18] = {
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
};

/* a "scene": N textures all filled with one color, a program that samples
 * texture #sample_index (1-based among the ones generated HERE), a VAO, one
 * quad.  Returns the sampled pixel. */
static void run_scene(GLubyte out[4], int n_textures, int sample_index,
                       GLubyte r, GLubyte g, GLubyte b)
{
    GLuint tex[64] = { 0 };
    GLubyte img[4 * 4 * 4];
    GLuint prog, vao, vbo;
    GLint loc;

    for (int i = 0; i < 4 * 4; i++) {
        img[i * 4 + 0] = r; img[i * 4 + 1] = g; img[i * 4 + 2] = b; img[i * 4 + 3] = 255;
    }

    glGenTextures(n_textures, tex);
    for (int i = 0; i < n_textures; i++) {
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    /* every generated name at and past the initial hash-table size (32) must be
     * a first-class citizen */
    EXPECT_TRUE(sample_index >= 1 && sample_index <= n_textures);
    EXPECT_NO_GL_ERROR();
    for (int i = 28; i <= 34 && i <= n_textures; i++)
        EXPECT_TRUE(glIsTexture(tex[i - 1]));

    prog = mgl_make_program(k_vs, k_fs);
    EXPECT_TRUE(prog);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof k_quad, k_quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex[sample_index - 1]);
    loc = glGetUniformLocation(prog, "tex");
    if (loc >= 0) glUniform1i(loc, 0);

    glViewport(0, 0, 4, 4);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    mgl_read_rgba(1, 1, out);
    if (getenv("XCTX_DEBUG"))
        fprintf(stderr, "scene %d tex%d: px=%d,%d,%d,%d\n", n_textures, sample_index, out[0],out[1],out[2],out[3]);
    while (glGetError() != GL_NO_ERROR) {}

    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(n_textures, tex);
}

TEST(two_contexts_isolated)
{
    void *a, *b;
    GLubyte px[4];

    /* scene A ("ideas"): 8 textures, red, sample texture 3 */
    a = make_ctx();
    EXPECT_TRUE(a != NULL);
    if (a) {
        run_scene(px, 8, 3, 255, 0, 0);
        EXPECT_NEAR(px[0], 255, 8);
        EXPECT_NEAR(px[1], 0, 8);
        EXPECT_NEAR(px[2], 0, 8);
        destroyGLMContext(a);
    }

    /* scene B ("jellyfish"): 33 textures (> the initial 32-entry tables),
     * green, sample textures 31/32/33 in turn — every one must be green */
    b = make_ctx();
    EXPECT_TRUE(b != NULL);
    if (b) {
        int idx[3] = { 31, 32, 33 };
        for (int i = 0; i < 3; i++) {
            run_scene(px, 33, idx[i], 0, 255, 0);
            EXPECT_NEAR(px[0], 0, 8);
            EXPECT_NEAR(px[1], 255, 8);
            EXPECT_NEAR(px[2], 0, 8);
        }
        destroyGLMContext(b);
    }

    /* a third context after B must also be isolated (A was destroyed first,
     * B most recently — check neither leaks in) */
    void *c = make_ctx();
    EXPECT_TRUE(c != NULL);
    if (c) {
        run_scene(px, 4, 2, 0, 0, 255);
        EXPECT_NEAR(px[0], 0, 8);
        EXPECT_NEAR(px[1], 0, 8);
        EXPECT_NEAR(px[2], 255, 8);
        destroyGLMContext(c);
    }

    /* the harness' auto-created context must still work afterwards */
    MGLsetCurrentContext(NULL);
}

int main(void)
{
    RUN(two_contexts_isolated);
    return mgl_test_summary();
}
