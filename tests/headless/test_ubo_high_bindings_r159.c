/* r159: uniform-buffer binding points above 84 (compat r111 advertises 512).
 *   1. glBindBuffersBase / glBindBuffersRange over the whole advertised range
 *      round-trips through glGetIntegeri_v (KHR-GL46.multi_bind functional_bind_
 *      buffers_base/_range: an 8-entry overflow table lost everything but the
 *      highest 8 points)
 *   2. unbinding one high point leaves the others; deleting a buffer clears the
 *      high points it was bound to
 *   3. drawing from blocks at binding 256 and 300, the buffer at 256 swapped
 *      between draws of one render pass (the draw-fast gate must see the
 *      re-pointed high binding) */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define W 64
#define CELL 8

static const char *vs =
    "#version 450\nlayout(location=0) in vec2 p;\nlayout(location=0) uniform vec2 cell;\n"
    "void main(){ vec2 q = (cell + p) * (2.0 / 8.0) - 1.0; gl_Position = vec4(q, 0.5, 1.0); }\n";
static const char *fs =
    "#version 450\nlayout(std140, binding=256) uniform A { vec4 a; };\n"
    "layout(std140, binding=300) uniform B { vec4 b; };\nout vec4 o;\nvoid main(){ o = a + b; }\n";

static GLuint mk(void)
{
    const char *s[2] = {vs, fs}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) {
        GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[1024]; glGetShaderInfoLog(sh, sizeof log, NULL, log); printf("  [compile] %s\n", log); }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[1024]; glGetProgramInfoLog(p, sizeof log, NULL, log); printf("  [link] %s\n", log); }
    return p;
}

static unsigned char img[W * W * 4];

TEST(multi_bind_whole_range)
{
    GLint max = 0;
    glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &max);
    EXPECT_TRUE(max >= 256);
    GLuint *names = calloc((size_t)max, sizeof(GLuint));
    GLintptr *offs = calloc((size_t)max, sizeof(GLintptr));
    GLsizeiptr *sizes = calloc((size_t)max, sizeof(GLsizeiptr));
    glGenBuffers(max, names);
    for (int i = 0; i < max; i++) {
        glBindBuffer(GL_UNIFORM_BUFFER, names[i]);
        glBufferData(GL_UNIFORM_BUFFER, 512, NULL, GL_STATIC_DRAW);
        offs[i] = (i % 2) * 256; sizes[i] = 256;
    }
    glBindBuffersBase(GL_UNIFORM_BUFFER, 0, max, names);
    EXPECT_NO_GL_ERROR();
    int bad = 0;
    for (int i = 0; i < max; i++) {
        GLint v = -1; glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, (GLuint)i, &v);
        if (v != (GLint)names[i] && bad++ < 4) printf("  [base] index %d = %d, want %u\n", i, v, names[i]);
    }
    EXPECT_EQ_INT(bad, 0);

    glBindBuffersRange(GL_UNIFORM_BUFFER, 0, max, names, offs, sizes);
    EXPECT_NO_GL_ERROR();
    bad = 0;
    for (int i = 0; i < max; i++) {
        GLint v = -1; GLint64 o = -1, s = -1;
        glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, (GLuint)i, &v);
        glGetInteger64i_v(GL_UNIFORM_BUFFER_START, (GLuint)i, &o);
        glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, (GLuint)i, &s);
        if ((v != (GLint)names[i] || o != offs[i] || s != sizes[i]) && bad++ < 4)
            printf("  [range] index %d = %d/%lld/%lld\n", i, v, (long long)o, (long long)s);
    }
    EXPECT_EQ_INT(bad, 0);

    /* unbind one high point; the neighbours stay */
    glBindBufferBase(GL_UNIFORM_BUFFER, 200, 0);
    GLint v199 = 0, v200 = -1, v201 = 0;
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 199, &v199);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 200, &v200);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 201, &v201);
    EXPECT_EQ_INT(v199, names[199]); EXPECT_EQ_INT(v200, 0); EXPECT_EQ_INT(v201, names[201]);

    /* deleting a buffer clears its high binding */
    glDeleteBuffers(1, &names[300]);
    GLint v300 = -1; glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 300, &v300);
    EXPECT_EQ_INT(v300, 0);

    glBindBuffersBase(GL_UNIFORM_BUFFER, 0, max, NULL);    /* unbind all */
    GLint v0 = -1, vlast = -1;
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, 0, &v0);
    glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, (GLuint)max - 1, &vlast);
    EXPECT_EQ_INT(v0, 0); EXPECT_EQ_INT(vlast, 0);
    names[300] = 0;
    glDeleteBuffers(max, names);
    free(names); free(offs); free(sizes);
    EXPECT_NO_GL_ERROR();
}

TEST(draw_from_high_bindings)
{
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    float q[8] = {0, 0, 1, 0, 0, 1, 1, 1};
    GLuint vao, vbo; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);

    float red[4] = {1, 0, 0, 1}, green[4] = {0, 1, 0, 1}, blue[4] = {0, 0, 1, 0};
    GLuint ur, ug, ub;
    glGenBuffers(1, &ur); glBindBuffer(GL_UNIFORM_BUFFER, ur); glBufferData(GL_UNIFORM_BUFFER, 16, red, GL_STATIC_DRAW);
    glGenBuffers(1, &ug); glBindBuffer(GL_UNIFORM_BUFFER, ug); glBufferData(GL_UNIFORM_BUFFER, 16, green, GL_STATIC_DRAW);
    glGenBuffers(1, &ub); glBindBuffer(GL_UNIFORM_BUFFER, ub); glBufferData(GL_UNIFORM_BUFFER, 16, blue, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 300, ub);

    GLuint p = mk(); glUseProgram(p);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    for (int c = 0; c < 16; c++) {
        glBindBufferBase(GL_UNIFORM_BUFFER, 256, (c & 1) ? ug : ur);
        glUniform2f(0, (float)(c % 8), (float)(c / 8));
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glFinish();
    glReadPixels(0, 0, W, W, GL_RGBA, GL_UNSIGNED_BYTE, img);
    int bad = 0;
    for (int c = 0; c < 16; c++) {
        const unsigned char *px = &img[4 * (((c / 8) * CELL + 4) * W + (c % 8) * CELL + 4)];
        int wr = (c & 1) ? 0 : 255, wg = (c & 1) ? 255 : 0;
        if (abs(px[0] - wr) > 2 || abs(px[1] - wg) > 2 || abs(px[2] - 255) > 2) {
            if (bad++ < 4) printf("  [draw] cell %d = %d %d %d, want %d %d 255\n", c, px[0], px[1], px[2], wr, wg);
        }
    }
    EXPECT_EQ_INT(bad, 0);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(multi_bind_whole_range);
    RUN(draw_from_high_bindings);
    return mgl_test_summary();
}
