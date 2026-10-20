/* orphaning: glBufferData must give the buffer a NEW data store — commands already
 * issued (draws queued, not finished) must still see the OLD contents, and a
 * glBufferSubData on in-flight work must be ordered as if serialized.  Draws with the
 * geometry CPU/GPU assembly read the element store, so this exercises the path. */
#include "mgl_test.h"
#include <stdlib.h>
#include <string.h>

static void readpx(int x, int y, GLubyte out[4])
{
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}
#define CHK(x, y, r, g, b) do { GLubyte _p[4] = { 9, 9, 9, 9 }; readpx(x, y, _p); mgl_test_checks++; \
        if (_p[0] != (GLubyte)(r) || _p[1] != (GLubyte)(g) || _p[2] != (GLubyte)(b)) \
            MGL_FAIL("(%d,%d) is %u,%u,%u,%u, expected %d,%d,%d", x, y, _p[0], _p[1], _p[2], _p[3], r, g, b); } while (0)

int main(void)
{
    /* quad 0 = left half of a 64x64 FBO from indices {0,1,2}, quad 1 = right half from
     * {3,4,5}: the second upload re-uses the SAME element indices with different data,
     * so a stale/overwritten store is visible in the pixels */
    const char *vs = "#version 450 core\n"
                     "layout(location=0) in vec2 pos;\n"
                     "layout(location=1) in float side;\n"
                     "out float v_side;\n"
                     "void main(){ v_side = side; gl_Position = vec4(pos, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
                     "in float v_side;\n"
                     "layout(location=0) out vec4 c;\n"
                     "void main(){ c = v_side < 0.5 ? vec4(1,0,0,1) : vec4(0,1,0,1); }\n";
    GLuint p = mgl_make_program(vs, fs);
    EXPECT_TRUE(p != (GLuint)0);
    if (!p) return 1;
    {
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        EXPECT_EQ_INT(ok, GL_TRUE);
        if (!ok) return 1;
    }

    GLuint tex, fbo, vao, vbo, ebo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 64, 64);

    /* vertices: two quads (left, right); 'side' picks the colour (0 = red, 1 = green).
     * Slots 0..2 = left/red, 3..5 = right/green, 6..8 = left/green, 9..11 = right/red, so
     * the same index triple {0,1,2} vs {6,7,8} (or {3,4,5} vs {9,10,11}) draws the same
     * region in either colour and stale vs fresh stores are told apart by pixels alone. */
    float verts[] = {
        -1, -1, 0,   0, -1, 0,   -1, 1, 0,    /* 0..2:  left, red   */
         0, -1, 1,   1, -1, 1,   0,  1, 1,    /* 3..5:  right, green */
        -1, -1, 1,   0, -1, 1,   -1, 1, 1,    /* 6..8:  left, green  */
         0, -1, 0,   1, -1, 0,   0,  1, 0,    /* 9..11: right, red   */
    };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 12, 0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 12, (void *)8);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glUseProgram(p);

    /* ---- 1. glBufferData orphaning: draw L (red), re-upload with data that would draw
     *          green instead, no finish between; the first draw must still be red ---- */
    GLuint ia[3] = { 0, 1, 2 };
    GLuint ib[3] = { 6, 7, 8 };
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ia, ia, GL_STATIC_DRAW);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);      /* left, red — queued */
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ib, ib, GL_STATIC_DRAW);   /* re-spec */
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);      /* left, green */
    glFinish();
    CHK(4, 32, 0, 255, 0);        /* the second draw saw the NEW store */
    CHK(60, 32, 0, 0, 0);

    /* ---- 2. glBufferSubData on in-flight work: serialised as if in order ---- */
    glClear(GL_COLOR_BUFFER_BIT);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ia, ia, GL_STATIC_DRAW);
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);      /* left, red — queued */
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof ib, ib);  /* sub-update */
    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);      /* left, green */
    glFinish();
    CHK(4, 32, 0, 255, 0);
    CHK(60, 32, 0, 0, 0);

    /* ---- 3. a plain vertex-array buffer: same orphaning contract ---- */
    glClear(GL_COLOR_BUFFER_BIT);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof ia, ia, GL_STATIC_DRAW);
    {
        float left[9] = { -1, -1, 0, 0, -1, 0, -1, 1, 0 };
        float right[9] = { 0, -1, 1, 1, -1, 1, 0, 1, 1 };
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof left, left);   /* positions of 0..2 */
        glBufferSubData(GL_ARRAY_BUFFER, 9 * sizeof(float), sizeof right, right);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);   /* left, red — queued */
        /* re-spec the whole vertex store with the right-hand quad in slots 0..2 */
        float swap[18];
        memcpy(swap, right, sizeof right);
        memcpy(swap + 9, left, sizeof left);
        glBufferData(GL_ARRAY_BUFFER, sizeof swap, swap, GL_STATIC_DRAW);
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL);   /* right, green now */
        glFinish();
    }
    {
        int green = 0, red = 0;
        GLubyte img[64 * 64 * 4];
        glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, img);
        for (int i = 0; i < 64 * 64; i++)
        {
            if (img[i * 4 + 1] > 200 && img[i * 4] < 50) green++;
            if (img[i * 4] > 200 && img[i * 4 + 1] < 50) red++;
        }
        printf("test3: %d red, %d green pixels\n", red, green);
        EXPECT_TRUE(green > 100);
    }
    CHK(4, 32, 255, 0, 0);        /* the queued draw kept the OLD vertex store */

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    return 0;
}
