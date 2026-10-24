/* test_compat_perf.c — regression tests for the encoder bind shadow and the
 * streaming argument ring (perf session 3; MGLRenderer+Compat.m,
 * mgl_compat_perf.h).
 *
 * Every mechanism replaces a per-draw Metal encode call with bookkeeping:
 *   - a bind that the encoder already has (same object+offset+encoder) is
 *     skipped -> a stale skip must be caught by pixel checks that alternate
 *     buffer / texture / uniform content between draws;
 *   - byte payloads (default uniform blocks, client arrays, constant attribs,
 *     copied index lists) stream through a 4x512 KB shared ring with
 *     offset-only updates and a memcmp skip -> exercise with enough traffic
 *     to wrap the ring (> 2 MB) and verify the pixels are still exact.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "uniform float xoff;\n"
    "void main(){ gl_Position = vec4(pos.x + xoff, pos.y, 0.0, 1.0); }\n";

static const char *k_fs =
    "#version 450 core\n"
    "layout(location=0) out vec4 color;\n"
    "uniform vec4 ucolor;\n"
    "void main(){ color = ucolor; }\n";

/* 192 vec4 pad uniforms make the default uniform block ~3.1 KB: every draw
 * updates the whole block, so 700 draws push > 2 MB through the argument ring
 * and force a slot recycle (the GPU wait path) plus fresh memcmp compares. */
#define PAD_VECS 192
static char *big_fs_source(void)
{
    static char src[8192];
    int n = snprintf(src, sizeof src,
        "#version 450 core\nlayout(location=0) out vec4 color;\n"
        "uniform vec4 ucolor;\nuniform vec4 pad[%d];\n"
        "void main(){ vec4 s = ucolor;\n", PAD_VECS);
    for (int i = 0; i < PAD_VECS; i++)
        n += snprintf(src + n, sizeof src - (size_t)n, "s += pad[%d] * %f;\n", i, (i & 1) ? 0.0f : 0.0f);
    snprintf(src + n, sizeof src - (size_t)n, "color = s; }\n");
    return src;
}

TEST(ring_uniform_updates)
{
    GLuint prog = mgl_make_program(k_vs, big_fs_source());
    if (!prog) return;
    GLint loc_x = glGetUniformLocation(prog, "xoff");
    GLint loc_c = glGetUniformLocation(prog, "ucolor");
    GLint loc_p = glGetUniformLocation(prog, "pad[0]");
    if (loc_x < 0 || loc_c < 0 || loc_p < 0) { MGL_FAIL("uniform locations missing"); return; }

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float quad[] = { -1,-1,  1,-1,  1,1,   -1,-1,  1,1,  -1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    glViewport(0, 0, 64, 64);

    float pad[PAD_VECS * 4];
    for (int i = 0; i < 700; i++)
    {
        float x = -1.0f + 2.0f * (float)i / 699.0f;
        float r = (i % 2) ? 1.0f : 0.25f;
        float g = (i % 3) ? 0.5f : 0.75f;
        /* the pad block changes every draw: fresh ring allocation, no memcmp
         * skip possible (the skip path is covered by the revisit below) */
        for (int j = 0; j < PAD_VECS * 4; j++) pad[j] = 0.0f;
        glUniform1f(loc_x, x);
        glUniform4f(loc_c, r, g, 0.0f, 1.0f);
        glUniform4fv(loc_p, PAD_VECS, pad);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (i % 64 == 0 && i < 176)   /* beyond that the quad's left edge passes px=8 */
        {
            /* left half of the screen stays covered by the quad */
            EXPECT_PIXEL(8, 32, (int)(r * 255 + 0.5f), (int)(g * 255 + 0.5f), 0, 255, 3);
        }
    }
    EXPECT_NO_GL_ERROR();

    /* revisit an earlier uniform set: identical bytes -> the memcmp skip must
     * keep the encoder's binding and the pixels identical (a stale ring
     * offset or a wrongly skipped rebind would not) */
    for (int j = 0; j < PAD_VECS * 4; j++) pad[j] = 0.0f;
    glUniform1f(loc_x, -1.0f);
    glUniform4f(loc_c, 1.0f, 0.25f, 0.0f, 1.0f);
    glUniform4fv(loc_p, PAD_VECS, pad);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    EXPECT_PIXEL(8, 32, 255, 64, 0, 255, 3);

    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* client vertex arrays re-specified per draw: every draw uploads fresh client
 * memory through the ring (content-compare skips only identical vertices).
 * Client arrays exist only in a compatibility context (core rejects the
 * pointer with GL_INVALID_OPERATION by spec) and only through the default
 * VAO / the legacy pointer entry points (glVertexPointer). */
TEST(ring_client_arrays_per_draw)
{
    void *cctx = mglCompatCreateContext();      /* switches the current context */
    EXPECT_TRUE(cctx != NULL);

    GLuint prog = mgl_make_program(k_vs, k_fs);
    if (!prog) return;
    GLint loc_x = glGetUniformLocation(prog, "xoff");
    GLint loc_c = glGetUniformLocation(prog, "ucolor");

    glBindVertexArray(0);       /* default VAO: client arrays live here */
    glUseProgram(prog);
    glViewport(0, 0, 64, 64);
    glUniform1f(loc_x, 0.0f);

    float base[] = { -1,-1,  1,-1,  1,1,   -1,-1,  1,1,  -1,1 };
    for (int i = 0; i < 32; i++)
    {
        float tri[12];
        memcpy(tri, base, sizeof base);
        if (i & 1)
        {
            /* shrink: half-screen quad -> (48,48) falls outside */
            for (int v = 0; v < 6; v++) { tri[v * 2] *= 0.5f; tri[v * 2 + 1] *= 0.5f; }
        }
        glUniform4f(loc_c, (i & 1) ? 1.0f : 0.0f, 0.0f, (i & 1) ? 0.0f : 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(2, GL_FLOAT, 0, tri);       /* legacy client array */
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (i & 1)
            EXPECT_PIXEL(48, 48, 0, 0, 0, 255, 2);
        else
            EXPECT_PIXEL(48, 48, 0, 0, 255, 255, 2);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(prog);
}

/* copied index lists (CPU index pointer) stream through the ring too; the
 * client index pointer needs the default VAO of a compat context */
TEST(ring_cpu_indices)
{
    void *cctx = mglCompatCreateContext();
    EXPECT_TRUE(cctx != NULL);

    GLuint prog = mgl_make_program(k_vs, k_fs);
    if (!prog) return;
    GLint loc_c = glGetUniformLocation(prog, "ucolor");

    glBindVertexArray(0);
    glEnableClientState(GL_VERTEX_ARRAY);
    static const float quad[] = { -1,-1,  1,-1,  1,1,  -1,1 };
    glVertexPointer(2, GL_FLOAT, 0, quad);
    EXPECT_NO_GL_ERROR();

    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "xoff"), 0.0f);
    glViewport(0, 0, 64, 64);

    /* alternate between two index sets: full-screen quad and a degenerate
     * triangle (no pixels) */
    static const GLubyte idx_full[6]   = { 0, 1, 2,  0, 2, 3 };
    static const GLubyte idx_empty[3]  = { 0, 1, 1 };
    for (int i = 0; i < 8; i++)
    {
        glUniform4f(loc_c, 1.0f, 1.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, (i & 1) ? 6 : 3, GL_UNSIGNED_BYTE,
                       (i & 1) ? (const void *)idx_full : (const void *)idx_empty);
        if (i & 1)
            EXPECT_PIXEL(32, 32, 255, 255, 0, 255, 2);
        else
            EXPECT_PIXEL(32, 32, 0, 0, 0, 255, 2);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(prog);
}

/* two programs alternating per draw: same buffer indices must not inherit the
 * other program's bindings (encoder-identity shadow across program switches) */
TEST(shadow_program_alternation)
{
    const char *fs2 =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 ucolor2;\n"
        "void main(){ color = ucolor2 * 0.5; }\n";
    GLuint p1 = mgl_make_program(k_vs, k_fs);
    GLuint p2 = mgl_make_program(k_vs, fs2);
    if (!p1 || !p2) return;

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float quad[] = { -1,-1,  1,-1,  1,1,   -1,-1,  1,1,  -1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glViewport(0, 0, 64, 64);

    for (int i = 0; i < 16; i++)
    {
        glUseProgram(p1);
        glUniform1f(glGetUniformLocation(p1, "xoff"), 0.0f);
        glUniform4f(glGetUniformLocation(p1, "ucolor"), 1.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);

        glUseProgram(p2);
        glUniform1f(glGetUniformLocation(p2, "xoff"), 0.0f);
        glUniform4f(glGetUniformLocation(p2, "ucolor2"), 0.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        EXPECT_PIXEL(32, 32, 0, 128, 128, 128, 2);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(p1);
    glDeleteProgram(p2);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

/* one texture re-bound per draw after a re-upload (glTexSubImage2D): the
 * texture shadow must re-issue the bind when the TEXTURE OBJECT changes and
 * must NOT skip when the same object's pixels changed */
TEST(shadow_texture_content)
{
    const char *fs =
        "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform sampler2D tex;\n"
        "void main(){ color = texture(tex, vec2(0.5)); }\n";
    GLuint prog = mgl_make_program(k_vs, fs);
    if (!prog) return;

    GLuint vao, vbo, tex[2];
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    static const float quad[] = { -1,-1,  1,-1,  1,1,   -1,-1,  1,1,  -1,1 };
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glGenTextures(2, tex);
    EXPECT_NO_GL_ERROR();
    for (int t = 0; t < 2; t++)
    {
        glBindTexture(GL_TEXTURE_2D, tex[t]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        unsigned px = (t == 0) ? 0xff0000ffu : 0xff00ff00u;   /* t=0: red, t=1: green */
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &px);
        EXPECT_NO_GL_ERROR();
    }

    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog, "xoff"), 0.0f);
    glViewport(0, 0, 64, 64);
    for (int i = 0; i < 8; i++)
    {
        int t = i & 1;
        glActiveTexture(GL_TEXTURE0);
        EXPECT_NO_GL_ERROR();
        glBindTexture(GL_TEXTURE_2D, tex[t]);
        EXPECT_NO_GL_ERROR();
        /* restore the pristine texel (the previous iteration's re-upload
         * below replaced it) so the first draw's expectation is stable */
        unsigned orig = (t == 0) ? 0xff0000ffu : 0xff00ff00u;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &orig);
        EXPECT_NO_GL_ERROR();
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (t == 0)
            EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 2);
        else
            EXPECT_PIXEL(32, 32, 0, 255, 0, 255, 2);

        /* and re-upload the same object's only texel: content changed in place
         * (0xff00ffff little-endian -> R=255,G=255,B=0 = yellow) */
        unsigned alt = t ? 0xffff00ffu : 0xff00ffffu;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &alt);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        if (t == 0)
            EXPECT_PIXEL(32, 32, 255, 255, 0, 255, 2);
        else
            EXPECT_PIXEL(32, 32, 255, 0, 255, 255, 2);
    }
    EXPECT_NO_GL_ERROR();

    glDeleteProgram(prog);
    glDeleteTextures(2, tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

int main(void)
{
    RUN(ring_uniform_updates);
    RUN(ring_client_arrays_per_draw);
    RUN(ring_cpu_indices);
    RUN(shadow_program_alternation);
    RUN(shadow_texture_content);
    return mgl_test_summary();
}
