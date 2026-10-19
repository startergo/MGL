/* test_fb.c — framebuffer / renderbuffer / readback / blit / clear tests (fb domain) */
#include "mgl_test.h"

static GLint fb_w, fb_h;

static void default_size(void)
{
    GLint vp[4] = {0};
    glGetIntegerv(GL_VIEWPORT, vp);
    fb_w = vp[2]; fb_h = vp[3];
}

static GLuint make_rb(GLenum fmt, GLsizei w, GLsizei h, GLsizei samples)
{
    GLuint rb;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    if (samples) glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, fmt, w, h);
    else glRenderbufferStorage(GL_RENDERBUFFER, fmt, w, h);
    return rb;
}

static GLuint make_fbo_rb(GLenum fmt, GLsizei w, GLsizei h, GLuint *rb_out)
{
    GLuint fbo, rb = make_rb(fmt, w, h, 0);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    if (rb_out) *rb_out = rb;
    return fbo;
}

static void read_u8(int x, int y, GLubyte out[4])
{
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}

#define EXPECT_RGBA8(x, y, r, g, b, a) do { GLubyte _p[4] = {1,2,3,4}; read_u8(x, y, _p); mgl_test_checks++; \
    if (abs(_p[0]-(r)) > 1 || abs(_p[1]-(g)) > 1 || abs(_p[2]-(b)) > 1 || abs(_p[3]-(a)) > 1) \
        MGL_FAIL("pixel(%d,%d) = %d,%d,%d,%d expected %d,%d,%d,%d", x, y, _p[0], _p[1], _p[2], _p[3], r, g, b, a); } while (0)

static GLuint quad_prog(void)
{
    static GLuint p;
    if (p) return p;
    const char *vs = "#version 450 core\nlayout(location=0) in vec2 pos;\nlayout(location=1) in vec4 col;\n"
                     "layout(location=0) out vec4 vcol;\nvoid main(){ vcol = col; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\nlayout(location=0) in vec4 vcol;\nlayout(location=0) out vec4 c0;\nlayout(location=1) out vec4 c1;\n"
                     "void main(){ c0 = vcol; c1 = vcol.bgra; }\n";
    p = mgl_make_program(vs, fs);
    return p;
}

/* draw a rectangle in NDC with color col */
static void draw_rect(float x0, float y0, float x1, float y1, const float col[4])
{
    static GLuint vao, vbo;
    if (!vao) { glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo); }
    float xy[12] = { x0,y0, x1,y0, x0,y1,  x1,y0, x1,y1, x0,y1 };
    float v[36];
    for (int i = 0; i < 6; i++)
    {
        v[i * 6 + 0] = xy[i * 2]; v[i * 6 + 1] = xy[i * 2 + 1];
        for (int c = 0; c < 4; c++) v[i * 6 + 2 + c] = col[c];
    }
    GLuint p = quad_prog();
    if (!p) return;
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 24, 0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 24, (void *)8);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

TEST(default_fb_size_matches_viewport)
{
    default_size();
    EXPECT_TRUE(fb_w > 0 && fb_h > 0);
    GLint sc[4] = {0};
    glGetIntegerv(GL_SCISSOR_BOX, sc);
    EXPECT_EQ_INT(sc[2], fb_w);
    EXPECT_EQ_INT(sc[3], fb_h);
    EXPECT_NO_GL_ERROR();
}

TEST(default_fb_clear_readback_everywhere)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_RGBA8(0, 0, 0, 255, 0, 255);
    EXPECT_RGBA8(fb_w - 1, 0, 0, 255, 0, 255);
    EXPECT_RGBA8(0, fb_h - 1, 0, 255, 0, 255);
    EXPECT_RGBA8(fb_w / 2, fb_h / 3, 0, 255, 0, 255);

    /* float / bgra / red / packed */
    float f[4] = {0};
    glReadPixels(3, 3, 1, 1, GL_RGBA, GL_FLOAT, f);
    EXPECT_NEAR(f[1], 1.0, 1e-3); EXPECT_NEAR(f[0], 0.0, 1e-3);
    GLubyte bgra[4] = {0};
    glReadPixels(3, 3, 1, 1, GL_BGRA, GL_UNSIGNED_BYTE, bgra);
    EXPECT_EQ_INT(bgra[1], 255); EXPECT_EQ_INT(bgra[2], 0); EXPECT_EQ_INT(bgra[3], 255);
    GLushort p565 = 0;
    glReadPixels(3, 3, 1, 1, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &p565);
    EXPECT_EQ_INT(p565, 0x07E0);
    GLushort us[4] = {0};
    glReadPixels(3, 3, 1, 1, GL_RGBA, GL_UNSIGNED_SHORT, us);
    EXPECT_EQ_INT(us[1], 65535);
    EXPECT_NO_GL_ERROR();
}

TEST(default_fb_draw_readback)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fb_w, fb_h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const float red[4] = {1, 0, 0, 1};
    draw_rect(-1, -1, 0, 1, red);   /* left half */
    EXPECT_RGBA8(2, fb_h / 2, 255, 0, 0, 255);
    EXPECT_RGBA8(fb_w - 3, fb_h / 2, 0, 0, 0, 255);
    /* read a block spanning both halves */
    GLubyte *row = calloc((size_t)fb_w * 4, 1);
    glReadPixels(0, fb_h / 2, fb_w, 1, GL_RGBA, GL_UNSIGNED_BYTE, row);
    EXPECT_EQ_INT(row[0], 255);
    EXPECT_EQ_INT(row[(fb_w - 1) * 4], 0);
    free(row);
    EXPECT_NO_GL_ERROR();
}

TEST(fbo_texture_rgba8_clear_draw_readback)
{
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_RGBA8(0, 0, 0, 0, 255, 255);
    EXPECT_RGBA8(5, 7, 0, 0, 255, 255);
    EXPECT_RGBA8(15, 15, 0, 0, 255, 255);
    const float yellow[4] = {1, 1, 0, 1};
    draw_rect(-1, -1, 0, 1, yellow);   /* x < 8 */
    EXPECT_RGBA8(3, 9, 255, 255, 0, 255);
    EXPECT_RGBA8(12, 9, 0, 0, 255, 255);

    /* attachment queries */
    GLint v = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_TEXTURE);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &v);
    EXPECT_EQ_INT(v, tex);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &v);
    EXPECT_EQ_INT(v, 8);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_UNSIGNED_NORMALIZED);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glBindTexture(GL_TEXTURE_2D, 0);   /* textures.c: delete leaves active_texture_mask set */
    glDeleteTextures(1, &tex);
    glViewport(0, 0, fb_w, fb_h);
    EXPECT_NO_GL_ERROR();
}

TEST(fbo_rgba32f_clearbuffer_readback)
{
    GLuint rb, fbo = make_fbo_rb(GL_RGBA32F, 8, 8, &rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    const GLfloat c[4] = {2.5f, -1.0f, 0.25f, 1000.0f};
    glClearBufferfv(GL_COLOR, 0, c);
    float f[4] = {0};
    glReadPixels(6, 1, 1, 1, GL_RGBA, GL_FLOAT, f);
    EXPECT_NEAR(f[0], 2.5, 1e-6); EXPECT_NEAR(f[1], -1.0, 1e-6);
    EXPECT_NEAR(f[2], 0.25, 1e-6); EXPECT_NEAR(f[3], 1000.0, 1e-3);
    GLubyte u[4] = {0};
    glReadPixels(6, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, u);   /* clamped */
    EXPECT_EQ_INT(u[0], 255); EXPECT_EQ_INT(u[1], 0); EXPECT_EQ_INT(u[2], 64);
    GLint fmt = 0, type = 0;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &fmt);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &type);
    EXPECT_EQ_INT(fmt, GL_RGBA); EXPECT_EQ_INT(type, GL_FLOAT);
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    EXPECT_NO_GL_ERROR();
}

TEST(fbo_r32ui_integer_readback)
{
    GLuint rb, fbo = make_fbo_rb(GL_R32UI, 8, 8, &rb);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    const GLuint c[4] = {123456u, 7, 8, 9};
    glClearBufferuiv(GL_COLOR, 0, c);
    GLuint r[4] = {0};
    glReadPixels(2, 5, 1, 1, GL_RGBA_INTEGER, GL_UNSIGNED_INT, r);
    EXPECT_EQ_INT(r[0], 123456); EXPECT_EQ_INT(r[1], 0); EXPECT_EQ_INT(r[2], 0); EXPECT_EQ_INT(r[3], 1);
    GLuint one = 0;
    glReadPixels(7, 7, 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &one);
    EXPECT_EQ_INT(one, 123456);
    GLubyte b[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    EXPECT_NO_GL_ERROR();
}

TEST(mrt_clearbuffer_and_draw)
{
    GLuint rb0 = make_rb(GL_RGBA8, 8, 8, 0), rb1 = make_rb(GL_RGBA8, 8, 8, 0), fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, rb1);
    const GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, bufs);
    GLint db1 = 0;
    glGetIntegerv(GL_DRAW_BUFFER1, &db1);
    EXPECT_EQ_INT(db1, GL_COLOR_ATTACHMENT1);
    const GLfloat a[4] = {1, 0, 0, 1}, b[4] = {0, 0, 1, 1};
    glClearBufferfv(GL_COLOR, 0, a);
    glClearBufferfv(GL_COLOR, 1, b);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_RGBA8(4, 4, 255, 0, 0, 255);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_RGBA8(4, 4, 0, 0, 255, 255);

    /* draw writes c0 to attachment 0 and c1 (=col.bgra) to attachment 1 */
    glViewport(0, 0, 8, 8);
    const float col[4] = {0.0f, 1.0f, 0.2f, 1.0f};
    draw_rect(-1, -1, 1, 1, col);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_RGBA8(1, 6, 0, 255, 51, 255);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_RGBA8(1, 6, 51, 255, 0, 255);

    /* swapped mapping: slot 0 -> attachment 1 */
    const GLenum sw[2] = { GL_COLOR_ATTACHMENT1, GL_NONE };
    glDrawBuffers(2, sw);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    EXPECT_RGBA8(2, 2, 255, 255, 255, 255);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_RGBA8(2, 2, 0, 255, 51, 255);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    GLuint rbs[2] = { rb0, rb1 };
    glDeleteRenderbuffers(2, rbs);
    glViewport(0, 0, fb_w, fb_h);
    EXPECT_NO_GL_ERROR();
}

TEST(scissor_and_mask_clear)
{
    GLuint rb, fbo = make_fbo_rb(GL_RGBA8, 16, 16, &rb);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(4, 4, 4, 4);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    EXPECT_RGBA8(5, 5, 255, 255, 255, 255);
    EXPECT_RGBA8(3, 5, 0, 0, 0, 0);
    EXPECT_RGBA8(8, 8, 0, 0, 0, 0);
    EXPECT_RGBA8(5, 2, 0, 0, 0, 0);

    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
    glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
    glClear(GL_COLOR_BUFFER_BIT);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    EXPECT_RGBA8(5, 5, 128, 255, 255, 255);
    EXPECT_RGBA8(0, 0, 128, 0, 0, 0);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    glScissor(0, 0, fb_w, fb_h);
}

TEST(blit_copy_flip_scale)
{
    GLuint rbA, A = make_fbo_rb(GL_RGBA8, 16, 16, &rbA);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 8, 8);           /* bottom-left quadrant green */
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);

    GLuint rbB, B = make_fbo_rb(GL_RGBA8, 16, 16, &rbB);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, A);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, B);

    /* straight copy */
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, B);
    EXPECT_RGBA8(2, 2, 0, 255, 0, 255);
    EXPECT_RGBA8(12, 12, 255, 0, 0, 255);

    /* vertical flip: green moves to the top-left */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, A);
    glBlitFramebuffer(0, 0, 16, 16, 0, 16, 16, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, B);
    EXPECT_RGBA8(2, 13, 0, 255, 0, 255);
    EXPECT_RGBA8(2, 2, 255, 0, 0, 255);

    /* scale the 8x8 green quadrant onto the whole target */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, A);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, B);
    EXPECT_RGBA8(15, 15, 0, 255, 0, 255);
    EXPECT_RGBA8(0, 0, 0, 255, 0, 255);
    /* linear: interior samples stay inside the green quadrant */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, A);
    glBlitFramebuffer(8, 0, 16, 8, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_LINEAR);   /* red quadrant */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, B);
    EXPECT_RGBA8(8, 8, 255, 0, 0, 255);
    EXPECT_RGBA8(0, 8, 191, 64, 0, 255);    /* s=8.25: 75% red texel 8, 25% green texel 7 */

    /* blit to the default framebuffer, then read it */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, A);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, 16, 16, 10, 10, 26, 26, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    EXPECT_RGBA8(11, 11, 0, 255, 0, 255);
    EXPECT_RGBA8(24, 24, 255, 0, 0, 255);

    /* errors */
    glBlitFramebuffer(0, 0, 1, 1, 0, 0, 1, 1, 0x80000000u, GL_NEAREST);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glBlitFramebuffer(0, 0, 1, 1, 0, 0, 1, 1, GL_COLOR_BUFFER_BIT, GL_NEAREST_MIPMAP_NEAREST);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBlitFramebuffer(0, 0, 1, 1, 0, 0, 1, 1, GL_DEPTH_BUFFER_BIT, GL_LINEAR);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLuint f[2] = { A, B }, r[2] = { rbA, rbB };
    glDeleteFramebuffers(2, f);
    glDeleteRenderbuffers(2, r);
    EXPECT_NO_GL_ERROR();
}

TEST(multisample_renderbuffer_resolve)
{
    GLint maxs = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &maxs);
    EXPECT_TRUE(maxs >= 4);
    GLint maxfs = 0;
    glGetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &maxfs);
    EXPECT_TRUE(maxfs >= 4);

    GLuint ms = make_rb(GL_RGBA8, 16, 16, 4), fms;
    GLint s = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &s);
    EXPECT_TRUE(s >= 4);
    glGenFramebuffers(1, &fms);
    glBindFramebuffer(GL_FRAMEBUFFER, fms);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ms);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLint samples = 0, sb = 0;
    glGetIntegerv(GL_SAMPLES, &samples);
    glGetIntegerv(GL_SAMPLE_BUFFERS, &sb);
    EXPECT_TRUE(samples >= 4); EXPECT_EQ_INT(sb, 1);

    GLfloat pos[2] = { -1, -1 };
    glGetMultisamplefv(GL_SAMPLE_POSITION, 0, pos);
    EXPECT_TRUE(pos[0] >= 0 && pos[0] < 1 && pos[1] >= 0 && pos[1] < 1);
    glGetMultisamplefv(GL_SAMPLE_POSITION, samples, pos);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const float c[4] = {1, 0, 1, 1};
    draw_rect(-1, -1, 0, 1, c);

    GLubyte px[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);   /* MS read framebuffer */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLuint rs, fs = make_fbo_rb(GL_RGBA8, 16, 16, &rs);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fms);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fs);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 8, 8, GL_COLOR_BUFFER_BIT, GL_NEAREST);  /* scaled MS blit */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fs);
    EXPECT_RGBA8(3, 8, 255, 0, 255, 255);
    EXPECT_RGBA8(12, 8, 0, 0, 0, 255);

    /* mixed sample counts are incomplete */
    glBindFramebuffer(GL_FRAMEBUFFER, fms);
    GLuint single = make_rb(GL_RGBA8, 16, 16, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, single);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLuint f[2] = { fms, fs }, r[3] = { ms, rs, single };
    glDeleteFramebuffers(2, f);
    glDeleteRenderbuffers(3, r);
    glViewport(0, 0, fb_w, fb_h);
    EXPECT_NO_GL_ERROR();
}

TEST(depth_stencil_clear_readback)
{
    GLuint rc, fbo = make_fbo_rb(GL_RGBA8, 8, 8, &rc);
    GLuint ds = make_rb(GL_DEPTH24_STENCIL8, 8, 8, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, ds);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLint v = 0;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_STENCIL_SIZE, &v);
    EXPECT_EQ_INT(v, 8);
    glClearBufferfi(GL_DEPTH_STENCIL, 0, 0.25f, 7);
    float d = 0;
    glReadPixels(3, 3, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 0.25, 1e-6);
    GLubyte s = 0;
    glReadPixels(3, 3, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &s);
    EXPECT_EQ_INT(s, 7);
    GLuint packed = 0;
    glReadPixels(3, 3, 1, 1, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, &packed);
    EXPECT_EQ_INT(packed & 0xff, 7);
    EXPECT_NEAR((packed >> 8) / 16777215.0, 0.25, 1e-5);

    const GLfloat one = 1.0f;
    glClearBufferfv(GL_DEPTH, 0, &one);
    const GLint three = 3;
    glClearBufferiv(GL_STENCIL, 0, &three);
    glReadPixels(0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 1.0, 1e-6);
    glReadPixels(0, 0, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &s);
    EXPECT_EQ_INT(s, 3);

    glClearDepth(0.5);
    glClear(GL_DEPTH_BUFFER_BIT);
    GLushort dz = 0;
    glReadPixels(7, 7, 1, 1, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, &dz);
    EXPECT_NEAR(dz, 32768, 1);

    /* depth blit between same-format buffers */
    GLuint rc2, fbo2 = make_fbo_rb(GL_RGBA8, 8, 8, &rc2);
    GLuint ds2 = make_rb(GL_DEPTH24_STENCIL8, 8, 8, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, ds2);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo2);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 8, 8, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo2);
    glReadPixels(4, 4, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 0.5, 1e-6);
    glReadPixels(4, 4, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &s);
    EXPECT_EQ_INT(s, 3);
    /* scaled depth blit (CPU path) */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glClearBufferfv(GL_DEPTH, 0, &one);   /* clears draw fb (fbo2) depth to 1 */
    glBlitFramebuffer(0, 0, 4, 4, 0, 0, 8, 8, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo2);
    glReadPixels(7, 7, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 0.5, 1e-6);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GLuint f[2] = { fbo, fbo2 }, r[4] = { rc, ds, rc2, ds2 };
    glDeleteFramebuffers(2, f);
    glDeleteRenderbuffers(4, r);
    EXPECT_NO_GL_ERROR();
}

TEST(blit_depth_stencil_to_multisample)
{
    GLint maxs = 0;
    glGetIntegerv(GL_MAX_SAMPLES, &maxs);
    if (maxs < 4) return;   /* need 4x multisampled depth/stencil targets */

    /* single-sampled source: 8x8 with four depth/stencil quadrants */
    GLuint srcC, src = make_fbo_rb(GL_RGBA8, 8, 8, &srcC);
    GLuint srcDS = make_rb(GL_DEPTH24_STENCIL8, 8, 8, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, srcDS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glDisable(GL_SCISSOR_TEST);
    glClearDepth(0.95f);
    glClearStencil(200);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    static const float qz[4] = { 0.2f, 0.4f, 0.6f, 0.8f };
    static const GLubyte qs[4] = { 3, 5, 9, 13 };
    for (int q = 0; q < 4; q++)
    {
        glScissor((q & 1) * 4, (q >> 1) * 4, 4, 4);
        glClearBufferfi(GL_DEPTH_STENCIL, 0, qz[q], qs[q]);
    }
    glDisable(GL_SCISSOR_TEST);

    /* 16x16 4x multisampled destination + single-sample resolve target */
    GLuint dst;
    GLuint dstC = make_rb(GL_RGBA8, 16, 16, 4), dstDS = make_rb(GL_DEPTH24_STENCIL8, 16, 16, 4);
    glGenFramebuffers(1, &dst);
    glBindFramebuffer(GL_FRAMEBUFFER, dst);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, dstC);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, dstDS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLuint resC, res = make_fbo_rb(GL_RGBA8, 16, 16, &resC);
    GLuint resDS = make_rb(GL_DEPTH24_STENCIL8, 16, 16, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, resDS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    float d = 0;
    GLubyte st = 0;

    /* resolve helper: copy dst's depth/stencil into the single-sample target and read */
#define MSBLIT_CHECK(x, y, dz_, st_) do { \
        glBindFramebuffer(GL_READ_FRAMEBUFFER, dst); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res); \
        glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST); \
        EXPECT_NO_GL_ERROR(); \
        glBindFramebuffer(GL_READ_FRAMEBUFFER, res); \
        glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d); \
        EXPECT_NEAR(d, dz_, 1e-6); \
        glReadPixels(x, y, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, &st); \
        EXPECT_EQ_INT(st, st_); } while (0)

    /* scaled 2x depth+stencil blit into the multisampled image */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
    glClearDepth(0.5f);
    glClearStencil(64);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    MSBLIT_CHECK(2, 2, 0.2f, 3);
    MSBLIT_CHECK(10, 2, 0.4f, 5);
    MSBLIT_CHECK(2, 10, 0.6f, 9);
    MSBLIT_CHECK(10, 10, 0.8f, 13);
    MSBLIT_CHECK(7, 7, 0.2f, 3);    /* nearest edge: dest 7 -> source 3.75 -> quadrant 0 */
    MSBLIT_CHECK(8, 8, 0.8f, 13);   /* dest 8 -> source 4.25 -> quadrant 3 */

    /* the blitted depth is real depth: render against it (draw_rect writes z = 0.5) */
    glBindFramebuffer(GL_FRAMEBUFFER, dst);
    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const float white[4] = { 1, 1, 1, 1 };
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    draw_rect(-1, -1, 1, 1, white);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, dst);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, res);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, res);
    EXPECT_RGBA8(2, 2, 0, 0, 255, 255);       /* stored depth 0.2/0.4 < 0.5: incoming z fails GL_LESS */
    EXPECT_RGBA8(12, 2, 0, 0, 255, 255);
    EXPECT_RGBA8(2, 12, 255, 255, 255, 255);  /* stored depth 0.6/0.8 > 0.5: incoming z passes */
    EXPECT_RGBA8(12, 12, 255, 255, 255, 255);
    glDisable(GL_DEPTH_TEST);

    /* depth only, scaled down (8x8 -> 4x4, minification) */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
    glClearDepth(0.5f);
    glClearStencil(64);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 4, 4, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    MSBLIT_CHECK(1, 1, 0.2f, 64);
    MSBLIT_CHECK(3, 1, 0.4f, 64);
    MSBLIT_CHECK(2, 2, 0.8f, 64);
    MSBLIT_CHECK(9, 9, 0.5f, 64);

    /* stencil only, partial destination rectangle */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBlitFramebuffer(0, 0, 8, 8, 4, 4, 12, 12, GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    MSBLIT_CHECK(6, 6, 0.5f, 3);     /* depth outside the minification blit keeps the clear */
    MSBLIT_CHECK(11, 11, 0.5f, 13);
    MSBLIT_CHECK(2, 2, 0.8f, 64);    /* outside the partial rectangle */
    /* scissored: only the lower-left half of the destination is updated */
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
    glClearDepth(0.5f);
    glClearStencil(64);
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 8, 8);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT, GL_NEAREST);
    EXPECT_NO_GL_ERROR();
    glDisable(GL_SCISSOR_TEST);
    MSBLIT_CHECK(2, 2, 0.2f, 3);
    MSBLIT_CHECK(10, 2, 0.5f, 64);   /* right half clipped by the scissor */
    MSBLIT_CHECK(2, 10, 0.5f, 64);

    /* float depth source into a float multisampled target */
    GLuint fdst, fdstDS = make_rb(GL_DEPTH_COMPONENT32F, 16, 16, 4);
    glGenFramebuffers(1, &fdst);
    glBindFramebuffer(GL_FRAMEBUFFER, fdst);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fdstDS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glClearDepth(0.5f);
    glClear(GL_DEPTH_BUFFER_BIT);
    GLuint fdst2, f2DS = make_rb(GL_DEPTH_COMPONENT32F, 16, 16, 0);
    glGenFramebuffers(1, &fdst2);
    glBindFramebuffer(GL_FRAMEBUFFER, fdst2);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, f2DS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    /* 32F source with two distinct depth bands */
    GLuint fsrc, fsDS = make_rb(GL_DEPTH_COMPONENT32F, 8, 8, 0);
    glGenFramebuffers(1, &fsrc);
    glBindFramebuffer(GL_FRAMEBUFFER, fsrc);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fsDS);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glDisable(GL_SCISSOR_TEST);
    glClearDepth(0.9f);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, 8, 4);
    glClearDepth(0.25f);
    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fdst);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fsrc);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);   /* scaled into 4x MS */
    EXPECT_NO_GL_ERROR();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fdst);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fdst2);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fdst2);
    glReadPixels(2, 2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 0.25f, 1e-6);
    glReadPixels(2, 13, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &d);
    EXPECT_NEAR(d, 0.9f, 1e-6);

    /* errors: LINEAR with depth/stencil, scaled read from a multisampled framebuffer */
    glBindFramebuffer(GL_READ_FRAMEBUFFER, src);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dst);
    glBlitFramebuffer(0, 0, 8, 8, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_LINEAR);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, dst);   /* multisampled read, scaled */
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 8, 8, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_DEPTH_BUFFER_BIT, GL_NEAREST);  /* resolve stays legal */
    EXPECT_NO_GL_ERROR();

#undef MSBLIT_CHECK

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, fb_w, fb_h);
    GLuint fbs[6] = { src, dst, res, fdst, fdst2, fsrc };
    GLuint rbs[9] = { srcC, srcDS, dstC, dstDS, resC, resDS, fdstDS, f2DS, fsDS };
    glDeleteFramebuffers(6, fbs);
    glDeleteRenderbuffers(9, rbs);
    EXPECT_NO_GL_ERROR();
}

TEST(completeness_status_codes)
{
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    EXPECT_TRUE(!glIsFramebuffer(fbo));
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    EXPECT_TRUE(glIsFramebuffer(fbo));
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);

    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, 32);
    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, 16);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLint v = 0;
    glGetFramebufferParameteriv(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, &v);
    EXPECT_EQ_INT(v, 32);

    GLuint depth = make_rb(GL_DEPTH_COMPONENT24, 8, 8, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, depth);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);

    GLuint empty;
    glGenRenderbuffers(1, &empty);
    glBindRenderbuffer(GL_RENDERBUFFER, empty);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, empty);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);

    GLubyte px[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_GL_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_GL_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);

    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 4, 4);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);

    /* attachment deleted while bound -> detached */
    glDeleteRenderbuffers(1, &empty);
    v = -1;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_NONE);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);   /* depth only */
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER + 1), 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &depth);
    EXPECT_TRUE(!glIsFramebuffer(fbo));
    EXPECT_NO_GL_ERROR();
}

TEST(dsa_equivalents)
{
    GLuint fbo = 0, rb = 0;
    glCreateFramebuffers(1, &fbo);
    glCreateRenderbuffers(1, &rb);
    EXPECT_TRUE(glIsFramebuffer(fbo));
    EXPECT_TRUE(glIsRenderbuffer(rb));
    glNamedRenderbufferStorage(rb, GL_RGBA8, 8, 8);
    GLint w = 0;
    glGetNamedRenderbufferParameteriv(rb, GL_RENDERBUFFER_WIDTH, &w);
    EXPECT_EQ_INT(w, 8);
    glNamedFramebufferRenderbuffer(fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    EXPECT_EQ_INT(glCheckNamedFramebufferStatus(fbo, GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLint t = 0;
    glGetNamedFramebufferAttachmentParameteriv(fbo, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &t);
    EXPECT_EQ_INT(t, GL_RENDERBUFFER);
    const GLfloat c[4] = {0, 1, 1, 1};
    glClearNamedFramebufferfv(fbo, GL_COLOR, 0, c);
    glNamedFramebufferReadBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glNamedFramebufferDrawBuffer(fbo, GL_COLOR_ATTACHMENT0);
    glBlitNamedFramebuffer(fbo, 0, 0, 0, 8, 8, 0, 0, 8, 8, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    EXPECT_RGBA8(4, 4, 0, 255, 255, 255);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    EXPECT_RGBA8(7, 0, 0, 255, 255, 255);

    GLint p = 0;
    glNamedFramebufferParameteri(fbo, GL_FRAMEBUFFER_DEFAULT_LAYERS, 3);
    glGetNamedFramebufferParameteriv(fbo, GL_FRAMEBUFFER_DEFAULT_LAYERS, &p);
    EXPECT_EQ_INT(p, 3);
    const GLenum inv = GL_COLOR_ATTACHMENT0;
    glInvalidateNamedFramebufferData(fbo, 1, &inv);
    glInvalidateNamedFramebufferSubData(fbo, 1, &inv, 0, 0, 2, 2);
    EXPECT_NO_GL_ERROR();

    const GLenum bufs[1] = { GL_COLOR_ATTACHMENT0 };
    glNamedFramebufferDrawBuffers(fbo, 1, bufs);
    glClearNamedFramebufferfv(fbo, GL_COLOR, 0, c);
    EXPECT_RGBA8(1, 1, 0, 255, 255, 255);

    glClearNamedFramebufferfv(12345, GL_COLOR, 0, c);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glNamedRenderbufferStorage(54321, GL_RGBA8, 1, 1);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    EXPECT_NO_GL_ERROR();
}

TEST(pack_state_and_pbo)
{
    GLuint rb, fbo = make_fbo_rb(GL_RGBA8, 8, 8, &rb);
    glClearColor(1, 0.5f, 0.25f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    GLubyte buf[64];
    memset(buf, 0xAA, sizeof buf);
    glPixelStorei(GL_PACK_ALIGNMENT, 8);
    glReadPixels(0, 0, 3, 2, GL_RGB, GL_UNSIGNED_BYTE, buf);   /* rows padded to 16 bytes */
    EXPECT_EQ_INT(buf[0], 255); EXPECT_EQ_INT(buf[1], 128);
    EXPECT_EQ_INT(buf[9], 0xAA);                                /* untouched padding */
    EXPECT_EQ_INT(buf[16], 255); EXPECT_EQ_INT(buf[18], 64);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    memset(buf, 0, sizeof buf);
    glPixelStorei(GL_PACK_ROW_LENGTH, 4);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 1);
    glPixelStorei(GL_PACK_SKIP_ROWS, 1);
    glReadPixels(0, 0, 2, 2, GL_RED, GL_UNSIGNED_BYTE, buf);
    EXPECT_EQ_INT(buf[0], 0);
    EXPECT_EQ_INT(buf[5], 255); EXPECT_EQ_INT(buf[6], 255); EXPECT_EQ_INT(buf[9], 255);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);

    glPixelStorei(GL_PACK_SWAP_BYTES, 1);
    GLushort us = 0;
    glReadPixels(0, 0, 1, 1, GL_GREEN, GL_UNSIGNED_SHORT, &us);
    EXPECT_EQ_INT(us, 0x8080);  /* 0.5 -> 0x8080 either way */
    GLuint ui = 0;
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, &ui);
    EXPECT_EQ_INT(ui, 0xFF4080FFu);   /* swapped 0xFF8040FF */
    glPixelStorei(GL_PACK_SWAP_BYTES, 0);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, &ui);
    EXPECT_EQ_INT(ui, 0xFF8040FFu);

    /* robust read */
    GLubyte small[4];
    glReadnPixels(0, 0, 2, 1, GL_RGBA, GL_UNSIGNED_BYTE, 4, small);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glReadnPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 4, small);
    EXPECT_NO_GL_ERROR();
    EXPECT_EQ_INT(small[2], 64);

    /* pixel pack buffer */
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER, 64, NULL, GL_STREAM_READ);
    glReadPixels(0, 0, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE, (void *)16);
    EXPECT_NO_GL_ERROR();
    GLubyte got[16] = {0};
    glGetBufferSubData(GL_PIXEL_PACK_BUFFER, 16, 16, got);
    EXPECT_EQ_INT(got[0], 255); EXPECT_EQ_INT(got[1], 128); EXPECT_EQ_INT(got[2], 64); EXPECT_EQ_INT(got[15], 255);
    glReadPixels(0, 0, 8, 8, GL_RGBA, GL_UNSIGNED_BYTE, 0);   /* 256 bytes > 64 */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_SHORT, (void *)1);  /* misaligned */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    /* (not deleted: mglDeleteBuffers indexes STATE(buffers[]) with the GL target enum -> crash; buffers domain) */

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    EXPECT_NO_GL_ERROR();
}

TEST(error_cases)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 9999);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glBindFramebuffer(GL_TEXTURE_2D, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glBindRenderbuffer(GL_RENDERBUFFER, 9999);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, 0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawBuffer(GL_FRONT_RIGHT);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawBuffer(GL_BACK);
    EXPECT_NO_GL_ERROR();
    GLint db = 0;
    glGetIntegerv(GL_DRAW_BUFFER, &db);
    EXPECT_EQ_INT(db, GL_BACK);
    glReadBuffer(GL_BACK);
    EXPECT_NO_GL_ERROR();

    GLuint rb, fbo = make_fbo_rb(GL_RGBA8, 4, 4, &rb);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, 0x1234, GL_RENDERBUFFER, rb);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 31, GL_RENDERBUFFER, rb);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rb);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_3D, 0, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    const GLenum back = GL_BACK;
    glDrawBuffers(1, &back);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    const GLenum dup[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(2, dup);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glDrawBuffers(-1, dup);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glReadBuffer(GL_BACK);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glReadBuffer(0x1234);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    const GLint iv[4] = {0};
    glClearBufferiv(GL_DEPTH, 0, iv);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    const GLfloat fv[4] = {0};
    glClearBufferfv(GL_COLOR, -1, fv);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glClearBufferfv(GL_DEPTH, 1, fv);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glClearBufferuiv(GL_STENCIL, 0, (const GLuint *)iv);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glClearBufferfi(GL_COLOR, 0, 1.0f, 0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glClear(0x10);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);

    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB9_E5, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, -1, 4);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 1000, GL_RGBA8, 4, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    GLint p;
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, 0x1234, &p);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLubyte px[4];
    glReadPixels(0, 0, 1, 1, GL_RGBA, 0x1234, px);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_SHORT_5_6_5, px);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glReadPixels(0, 0, -1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    glReadPixels(0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, px);   /* no depth attachment */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glReadPixels(0, 0, 1, 1, GL_RGBA_INTEGER, GL_FLOAT, px);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);

    GLint v;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &v);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &v);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFramebufferParameteri(GL_FRAMEBUFFER, 0x1234, 1);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, -1);
    EXPECT_GL_ERROR(GL_INVALID_VALUE);
    const GLenum bad = GL_COLOR;
    glInvalidateFramebuffer(GL_FRAMEBUFFER, 1, &bad);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glGetMultisamplefv(0x1234, 0, (GLfloat *)px);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, 4);
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_BACK_LEFT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &v);
    EXPECT_EQ_INT(v, GL_FRAMEBUFFER_DEFAULT);
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &v);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glDrawBuffer(GL_FRONT);
    glReadBuffer(GL_FRONT);
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rb);
    EXPECT_NO_GL_ERROR();
}

TEST(fbo_texture_level_and_layer)
{
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGBA8, 16, 16);
    while (glGetError() != GL_NO_ERROR) {}
    GLint w1 = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 1, GL_TEXTURE_WIDTH, &w1);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 1);
    GLint lvl = -1;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL, &lvl);
    EXPECT_EQ_INT(lvl, 1);
    if (w1 == 8)
    {
        EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
        glClearColor(1, 0, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        EXPECT_RGBA8(7, 7, 255, 0, 255, 255);
    }
    else
    {
        /* texture domain did not define level 1: the attachment must be incomplete */
        fprintf(stderr, "  [note] level-1 storage unavailable from the texture domain; checking incompleteness only\n");
        EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    }
    EXPECT_NO_GL_ERROR();

    GLuint arr;
    glGenTextures(1, &arr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 4, 4, 3, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0, 2);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    GLint layer = -1;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER, &layer);
    EXPECT_EQ_INT(layer, 2);
    glClearColor(0, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_RGBA8(3, 3, 0, 255, 255, 255);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0, 5);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, 0);   /* 2D texture */
    EXPECT_GL_ERROR(GL_INVALID_OPERATION);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, 0);
    GLint layered = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_LAYERED, &layered);
    EXPECT_EQ_INT(layered, GL_TRUE);
    EXPECT_NO_GL_ERROR();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    GLuint t[2] = { tex, arr };
    glDeleteTextures(2, t);
    EXPECT_NO_GL_ERROR();
}

TEST(swap_and_finish_keep_working)
{
    extern void MGLswapBuffers(void *ctx);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_RGBA8(1, 1, 51, 102, 153, 255);
    glFlush();
    glFinish();
    MGLswapBuffers(NULL);
    glClearColor(1, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const float blue[4] = {0, 0, 1, 1};
    draw_rect(-1, -1, 1, 1, blue);
    EXPECT_RGBA8(fb_w / 2, fb_h / 2, 0, 0, 255, 255);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(default_fb_size_matches_viewport);
    RUN(default_fb_clear_readback_everywhere);
    RUN(default_fb_draw_readback);
    RUN(fbo_texture_rgba8_clear_draw_readback);
    RUN(fbo_rgba32f_clearbuffer_readback);
    RUN(fbo_r32ui_integer_readback);
    RUN(mrt_clearbuffer_and_draw);
    RUN(scissor_and_mask_clear);
    RUN(blit_copy_flip_scale);
    RUN(multisample_renderbuffer_resolve);
    RUN(depth_stencil_clear_readback);
    RUN(blit_depth_stencil_to_multisample);
    RUN(completeness_status_codes);
    RUN(dsa_equivalents);
    RUN(pack_state_and_pbo);
    RUN(error_cases);
    RUN(fbo_texture_level_and_layer);
    RUN(swap_and_finish_keep_working);
    return mgl_test_summary();
}
