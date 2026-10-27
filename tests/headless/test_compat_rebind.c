/* test_compat_rebind.c — session 19's buffer-identity-only respec tier
 * (compatRebindClassify / compatDrawFastRebind) must draw exactly what a full
 * state walk would draw for every respec pattern that can reach it, and must
 * NOT reach it for patterns it cannot handle (stride/divisor/alignment/class
 * changes, ring-backed buffers, shadow streams).  Each pattern draws to a
 * 64x64 FBO and checks exact pixels. */
#include "mgl_test.h"
#include "mgl_compat.h"
#include <stdint.h>

static const char *vs_src =
    "#version 410 core\n"
    "layout(location=0) in vec3 aPosition;\n"
    "layout(location=1) in vec2 aTex;\n"
    "uniform mat4 uMVP;\n"
    "out vec2 vT;\n"
    "void main() { vT = aTex; gl_Position = uMVP * vec4(aPosition, 1.0); }\n";
static const char *fs_src =
    "#version 410 core\n"
    "in vec2 vT; out vec4 fragColor;\n"
    "void main() { fragColor = vec4(0.25 + 0.5 * vT.x, 0.25 + 0.5 * vT.y, 0.5, 1.0); }\n";

static GLuint prog, fbo, color;
static GLint loc_mvp;

/* a 3-vertex triangle at x0 (its centroid), every vertex with tex (tx,ty):
 * 15 floats pos3+tex2 interleaved */
static void mesh15(float *v, float x0, float tx, float ty)
{
    const float x[3] = { -0.15f + x0, 0.15f + x0, x0 };
    const float y[3] = { -0.9f, -0.9f, 0.9f };
    for (int i = 0; i < 3; i++)
    {
        v[i * 5 + 0] = x[i];
        v[i * 5 + 1] = y[i];
        v[i * 5 + 2] = 0.0f;
        v[i * 5 + 3] = tx;
        v[i * 5 + 4] = ty;
    }
}

static void setup(void)
{
    void *cc = mglCompatCreateContext();
    EXPECT_TRUE(cc != NULL);
    prog = mgl_make_program(vs_src, fs_src);
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);
    loc_mvp = glGetUniformLocation(prog, "uMVP");

    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glViewport(0, 0, 64, 64);
    glClearColor(0, 0, 0, 1);
    float mvp[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
}

static GLuint big_vbo(const void *data, size_t bytes)
{
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferData(GL_ARRAY_BUFFER, 32768, NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, data);
    return b;
}

static void begin(void) { glClear(GL_COLOR_BUFFER_BIT); }

static void spec(GLuint vbo, int stride, int off0, int off1)
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void *)(intptr_t)off0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void *)(intptr_t)off1);
}

/* tex colors: (0,0)->(64,64,128)  (1,1)->(191,191,128) */
#define C00 64, 64, 128, 255
#define C11 191, 191, 128, 255

TEST(rebind_patterns)
{
    setup();
    if (!prog)
        return;

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    /* L: triangle at -0.45, tex (0,0).  R: triangle at +0.45, tex (1,1) */
    float mR[15], mL[15];
    mesh15(mR, 0.45f, 1.0f, 1.0f);
    mesh15(mL, -0.45f, 0.0f, 0.0f);
    GLuint vbR = big_vbo(mR, sizeof mR), vbL = big_vbo(mL, sizeof mL);

    /* warm the pipeline with R's own respec */
    spec(vbR, 20, 0, 12);
    for (int i = 0; i < 20; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    /* 1. respec to L, draw: the LEFT mesh, tex (0,0) */
    begin();
    spec(vbL, 20, 0, 12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 32, 64, 64, 128, 255, 6);
    EXPECT_PIXEL(48, 32, 0, 0, 0, 255, 6);

    /* 2. respec back to R */
    begin();
    spec(vbR, 20, 0, 12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);

    /* 3. alternate L,R,L fast: each respec must land */
    for (int rep = 0; rep < 3; rep++)
    {
        begin();
        spec(vbL, 20, 0, 12);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_PIXEL(16, 32, 64, 64, 128, 255, 6);

        begin();
        spec(vbR, 20, 0, 12);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);
    }

    /* 4. offset-only respec: same VBO, mesh R at 0, mesh L at 120 */
    GLuint vb2 = big_vbo(mR, sizeof mR);
    glBindBuffer(GL_ARRAY_BUFFER, vb2);
    glBufferSubData(GL_ARRAY_BUFFER, 120, sizeof mL, mL);
    begin();
    spec(vb2, 20, 0, 12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);
    begin();
    spec(vb2, 20, 120, 132);                        /* mesh L now */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 32, 64, 64, 128, 255, 6);
    EXPECT_PIXEL(48, 32, 0, 0, 0, 255, 6);

    /* 5. one binding moves: pos from L, tex from R */
    begin();
    glBindBuffer(GL_ARRAY_BUFFER, vbL);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, 0);
    glBindBuffer(GL_ARRAY_BUFFER, vbR);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, (void *)(intptr_t)12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 32, 191, 191, 128, 255, 6);                   /* left position, right tex */

    /* 6. stride change (must NOT rebind-classify: new PSO key) */
    float packed[3 * 8];                            /* pos3 tex2 pad3 per vertex */
    for (int i = 0; i < 3; i++)
    {
        packed[i * 8 + 0] = mR[i * 5 + 0];
        packed[i * 8 + 1] = mR[i * 5 + 1];
        packed[i * 8 + 2] = 0.0f;
        packed[i * 8 + 3] = 1.0f;
        packed[i * 8 + 4] = 1.0f;
        packed[i * 8 + 5] = packed[i * 8 + 6] = packed[i * 8 + 7] = 0.0f;
    }
    GLuint vb3 = big_vbo(packed, sizeof packed);
    begin();
    spec(vb3, 32, 0, 12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);

    /* 7. divisor respec (must full-walk): instanced attrib */
    begin();
    spec(vbR, 20, 0, 12);
    glVertexAttribDivisor(1, 1);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, 2);
    EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);
    glVertexAttribDivisor(1, 0);

    /* 8. misaligned-offset respec (alignment class change -> full walk): the
     *    tex pair really sits at byte 13 of each 24-byte vertex */
    unsigned char raw[3 * 24];
    memset(raw, 0, sizeof raw);
    for (int i = 0; i < 3; i++)
    {
        float pos[3] = { mR[i * 5 + 0], mR[i * 5 + 1], 0.0f };
        float tex[2] = { 1.0f, 1.0f };
        memcpy(&raw[i * 24], pos, 12);
        memcpy(&raw[i * 24 + 13], tex, 8);
    }
    GLuint vb4 = big_vbo(raw, sizeof raw);
    begin();
    spec(vb4, 24, 0, 13);                           /* 13 % 4 != 0 */
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(48, 32, 191, 191, 128, 255, 6);

    /* 9. back to a plain respec after all of that */
    begin();
    spec(vbL, 20, 0, 12);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_PIXEL(16, 32, 64, 64, 128, 255, 6);

    EXPECT_NO_GL_ERROR();
}

TEST(rebind_uniform_updates)
{
    /* uniforms must still land when a respec takes the rebind tier: the mesh
     * AND the fresh uniform value are both applied on the same draw */
    setup();
    if (!prog)
        return;
    float mvp[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

    glEnableVertexAttribArray(0);
    glDisableVertexAttribArray(1);                   /* vT = const (0,0) */

    float tR[15], tL[15], mR[9], mL[9];              /* pos3 only, stride 12 */
    mesh15(tR, 0.45f, 1.0f, 1.0f);
    mesh15(tL, -0.45f, 0.0f, 0.0f);
    /* mesh15 interleaves pos3+tex2; repack to pos3 (the spec below uses
     * stride 12 and only attrib 0 is enabled) */
    for (int i = 0; i < 3; i++)
    {
        mR[i * 3 + 0] = tR[i * 5 + 0]; mR[i * 3 + 1] = tR[i * 5 + 1]; mR[i * 3 + 2] = tR[i * 5 + 2];
        mL[i * 3 + 0] = tL[i * 5 + 0]; mL[i * 3 + 1] = tL[i * 5 + 1]; mL[i * 3 + 2] = tL[i * 5 + 2];
    }
    GLuint vbR = big_vbo(mR, sizeof mR), vbL = big_vbo(mL, sizeof mL);

    spec(vbR, 12, 0, 0);
    for (int i = 0; i < 10; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();

    for (int rep = 0; rep < 3; rep++)
    {
        /* respec to L + shift +0.9: L lands where R was (right half) */
        begin();
        spec(vbL, 12, 0, 0);
        mvp[12] = 0.9f;
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        {
            GLubyte p[4] = { 9, 9, 9, 9 };
            glReadPixels(48, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);

            EXPECT_TRUE(p[0] > 48 && p[0] < 80 && p[1] > 48 && p[1] < 80);
            glReadPixels(16, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
            EXPECT_TRUE(p[0] < 10 && p[1] < 10);
        }

        /* respec to R + shift -0.9: R lands where L was (left half) */
        begin();
        spec(vbR, 12, 0, 0);
        mvp[12] = -0.9f;
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        {
            GLubyte p[4];
            glReadPixels(16, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
            EXPECT_TRUE(p[0] > 48 && p[0] < 80 && p[1] > 48 && p[1] < 80);
            glReadPixels(48, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
            EXPECT_TRUE(p[0] < 10 && p[1] < 10);
        }
    }
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    if (getenv("UNIFORM_FIRST")) { RUN(rebind_uniform_updates); RUN(rebind_patterns); return mgl_test_summary(); } RUN(rebind_patterns);
    RUN(rebind_uniform_updates);
    return mgl_test_summary();
}
