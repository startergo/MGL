/* test_tessbuf.c — texture-buffer render / modify / re-render (dEQP-GL45-ES31
 * functional.texture.texture_buffer.render_modify reproducer: a texture buffer
 * rendered, then modified by glBufferSubData / glMapBufferRange without an
 * intervening glFinish, then rendered again must sample the MODIFIED contents.
 * Mirrors the CTS: the sampler uniform points at unit 1, the texture is bound
 * to unit 1 per draw, and the element buffer / TEXTURE_BUFFER binding is
 * detached again after the modify.)
 */
#include "mgl_test.h"

static const char *k_vs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";

static GLuint g_vao, g_vbo;

static void fullscreen_setup(void)
{
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    if (!g_vao) {
        glGenVertexArrays(1, &g_vao);
        glGenBuffers(1, &g_vbo);
        glBindVertexArray(g_vao);
        glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
    }
    /* rebinding forces DIRTY_VAO -> a fresh render encoder: works around the core draw path not
       recreating an encoder after a flush (glFinish / buffer realloc) when only DIRTY_STATE is set */
    glBindVertexArray(0);
    glBindVertexArray(g_vao);
}

static void draw_fullscreen(GLuint prog)
{
    if (!getenv("CTS_DRAW")) { fullscreen_setup(); glUseProgram(prog); glDrawArrays(GL_TRIANGLES, 0, 3); return; }
    /* CTS renderGL: a FRESH VAO + VBO per draw (fullscreen triangle), deleted afterwards */
    GLuint vao, vbo;
    static const float tri[] = { -1,-1,  3,-1,  -1,3 };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(0);
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}

#define TB_TEXELS 4096   /* 16384 bytes of RGBA8, like the CTS case */

/* glTexBuffer on buf; the texture goes to unit 1 like the CTS (u_fragSampler=1) */
static GLuint make_tb(GLuint *buf_out, const GLubyte *texels)
{
    GLuint buf, tex;
    glGenBuffers(1, &buf);
    *buf_out = buf;
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    glBufferData(GL_TEXTURE_BUFFER, TB_TEXELS * 4, texels, GL_STATIC_DRAW);
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, buf);
    glBindTexture(GL_TEXTURE_BUFFER, 0);   /* CTS leaves it unbound after setup */
    return tex;
}

/* draw the column-fetch pattern and check pixels x0..x0+7 against texels[x..] */
static void render_and_check(GLuint prog, GLuint tex, const GLubyte texels[TB_TEXELS][4], int x0, const char *what)
{
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    draw_fullscreen(prog);
    for (int k = 0; k < 8; k++) {
        int x = x0 + k;
        EXPECT_PIXEL(x, 3, texels[x][0], texels[x][1], texels[x][2], texels[x][3], 2);
    }
    (void)what;
}

/* deqp fbo surface: a texture-backed FBO as the default framebuffer */
static GLuint g_fbo, g_fbotex, g_fbods;
static void fbo_setup(int w, int h)
{
    glGenTextures(1, &g_fbotex);
    glBindTexture(GL_TEXTURE_2D, g_fbotex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenTextures(1, &g_fbods);
    glBindTexture(GL_TEXTURE_2D, g_fbods);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, w, h, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_fbotex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, g_fbods, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) MGL_FAIL("fbo incomplete");
    glViewport(0, 0, w, h);
}

static const char *k_tb_fs =
    "#version 450 core\n"
    "uniform samplerBuffer tb;\n"
    "uniform int uOff;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = texelFetch(tb, (int(gl_FragCoord.x) + uOff) % 4096); }\n";

TEST(buffer_render_modify_subdata)
{
    static GLubyte t0[TB_TEXELS][4], t1[TB_TEXELS][4];
    for (int i = 0; i < TB_TEXELS; i++) {
        t0[i][0] = (GLubyte)(i * 7); t0[i][1] = (GLubyte)(i * 13); t0[i][2] = (GLubyte)(i * 29); t0[i][3] = 255;
        t1[i][0] = (GLubyte)(255 - i); t1[i][1] = (GLubyte)(i * 3); t1[i][2] = (GLubyte)(i * 11); t1[i][3] = 128;
    }

    GLuint buf;
    GLuint tex = make_tb(&buf, (const GLubyte *)t0);
    GLuint p = mgl_make_program(k_vs, k_tb_fs);
    if (!p) return;
    GLint loc = glGetUniformLocation(p, "tb");
    EXPECT_NE(loc, -1);
    if (loc == -1) return;
    GLint off_loc = glGetUniformLocation(p, "uOff");
    EXPECT_NE(off_loc, -1);
    glUseProgram(p);
    glUniform1i(loc, 1);
    glUniform1i(off_loc, 0);
    fbo_setup(256, 256);
    EXPECT_NO_GL_ERROR();

    /* draw #1 with NO readback in between: the modify below must run while
     * draw #1 is still queued on the GPU (the CTS verifies only at the end) */
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    draw_fullscreen(p);
    EXPECT_NO_GL_ERROR();

    /* the CTS modify: a partial write starting at byte 2010 (texel 502.5) —
     * texels 0..501 stay t0, texels 504.. become t1 */
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    EXPECT_NO_GL_ERROR();
    glBufferSubData(GL_TEXTURE_BUFFER, 2010, 12267, (const GLubyte *)t1 + 2010);
    EXPECT_NO_GL_ERROR();
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    EXPECT_NO_GL_ERROR();

    /* draw #2 into the SAME open encoder (no clear, no state change — the CTS
     * clears once at init only) and only now read back */
    draw_fullscreen(p);
    for (int k = 0; k < 8; k++) {
        int x = k;
        EXPECT_PIXEL(x, 3, t0[x][0], t0[x][1], t0[x][2], t0[x][3], 2);        /* unchanged region */
    }
    glUniform1i(off_loc, 600);
    draw_fullscreen(p);
    for (int k = 0; k < 8; k++) {
        int x = 600 + k;
        EXPECT_PIXEL(k, 3, t1[x][0], t1[x][1], t1[x][2], t1[x][3], 2);        /* modified region */
    }
    EXPECT_NO_GL_ERROR();
}

TEST(buffer_render_modify_map_write)
{
    static GLubyte t0[TB_TEXELS][4], t1[TB_TEXELS][4];
    for (int i = 0; i < TB_TEXELS; i++) {
        t0[i][0] = (GLubyte)(i * 5); t0[i][1] = 60; t0[i][2] = (GLubyte)(i * 17); t0[i][3] = 255;
        t1[i][0] = (GLubyte)(i * 2); t1[i][1] = 200; t1[i][2] = (GLubyte)(i * 23); t1[i][3] = 255;
    }

    GLuint buf;
    GLuint tex = make_tb(&buf, (const GLubyte *)t0);
    GLuint p = mgl_make_program(k_vs, k_tb_fs);
    if (!p) return;
    GLint loc = glGetUniformLocation(p, "tb");
    EXPECT_NE(loc, -1);
    if (loc == -1) return;
    GLint off_loc = glGetUniformLocation(p, "uOff");
    EXPECT_NE(off_loc, -1);
    glUseProgram(p);
    glUniform1i(loc, 1);
    glUniform1i(off_loc, 0);
    fbo_setup(256, 256);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, tex);
    draw_fullscreen(p);
    EXPECT_NO_GL_ERROR();

    /* partial write-only map at 2010..14277 while draw #1 is queued */
    glBindBuffer(GL_TEXTURE_BUFFER, buf);
    void *m = glMapBufferRange(GL_TEXTURE_BUFFER, 2010, 12267, GL_MAP_WRITE_BIT);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(m != NULL);
    if (m) {
        memcpy(m, (const GLubyte *)t1 + 2010, 12267);
        EXPECT_EQ_INT(glUnmapBuffer(GL_TEXTURE_BUFFER), GL_TRUE);
    }
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    EXPECT_NO_GL_ERROR();

    glClear(GL_COLOR_BUFFER_BIT);
    draw_fullscreen(p);
    for (int k = 0; k < 8; k++) {
        int x = k;
        EXPECT_PIXEL(x, 3, t0[x][0], t0[x][1], t0[x][2], t0[x][3], 2);
    }
    glUniform1i(off_loc, 600);
    draw_fullscreen(p);
    for (int k = 0; k < 8; k++) {
        int x = 600 + k;
        EXPECT_PIXEL(k, 3, t1[x][0], t1[x][1], t1[x][2], t1[x][3], 2);
    }
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(buffer_render_modify_subdata);
    RUN(buffer_render_modify_map_write);
    return mgl_test_summary();
}
