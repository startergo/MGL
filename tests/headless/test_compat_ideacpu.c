/* test_compat_ideacpu.c — [ideas]-shaped per-draw CPU micro-benchmark.
 *
 * Not a correctness test: it replicates glmark2 [ideas]' draw shape on the
 * compat domain (89 VAO-switch state walks + ~353 draws per frame, one
 * program, per-frame uniform updates, indexed triangle draws) and prints the
 * MGL_PROFILE per-draw buckets so per-draw CPU work can be measured without
 * launching glmark2 (headless, no window, no present).  Run directly:
 *
 *   MGL_PROFILE=1 build/tests/test_compat_ideacpu 2>&1 | grep 'MGL PROFILE'
 *
 * Variants (env MGL_IDEACPU): 0 = the ideas shape above (default).
 */
#include "mgl_test.h"

extern void mgl_profile_dump(void);

#ifndef MGL_IDEACPU_VAOS
#define MGL_IDEACPU_VAOS 89
#endif
#define DRAWS_PER_WALK 4                 /* 89 x 4 = 356 draws/frame like [ideas] */

static const char *vs_src =
    "#version 410 core\n"
    "layout(location=0) in vec3 aPosition;\n"
    "layout(location=1) in vec2 aTexcoord;\n"
    "layout(location=2) in vec4 aColor;\n"
    "uniform mat4 uMVP;\n"
    "out vec2 vTex; out vec4 vColor;\n"
    "void main() { vTex = aTexcoord; vColor = aColor;\n"
    "  gl_Position = uMVP * vec4(aPosition, 1.0); }\n";

static const char *fs_src =
    "#version 410 core\n"
    "in vec2 vTex; in vec4 vColor; out vec4 fragColor;\n"
    "uniform sampler2D uTex0; uniform sampler2D uTex1;\n"
    "void main() { fragColor = vColor * (texture(uTex0, vTex) * 0.75 + texture(uTex1, vTex) * 0.25); }\n";

TEST(ideas_shape_cpu)
{
    GLuint prog = mgl_make_program(vs_src, fs_src);
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);

    /* two small textures (the program samples both: num_samplers > 1) */
    GLuint tex[2];
    glGenTextures(2, tex);
    unsigned char px[16 * 16 * 4];
    for (int i = 0; i < (int)sizeof px; i++) px[i] = (unsigned char)(i * 7);
    for (int t = 0; t < 2; t++)
    {
        glBindTexture(GL_TEXTURE_2D, tex[t]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }

    /* a framebuffer so the draws go somewhere real but no present is needed */
    GLuint fbo, color;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    EXPECT_EQ_INT(glCheckFramebufferStatus(GL_FRAMEBUFFER), GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 800, 600);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    /* 89 VAOs, identical layout, distinct VBO + element buffers (the [ideas]
     * pattern: every mesh re-specifies its vertex arrays) */
    GLuint vao[MGL_IDEACPU_VAOS], vbo[MGL_IDEACPU_VAOS], ebo[MGL_IDEACPU_VAOS];
    glGenVertexArrays(MGL_IDEACPU_VAOS, vao);
    glGenBuffers(MGL_IDEACPU_VAOS, vbo);
    glGenBuffers(MGL_IDEACPU_VAOS, ebo);

    float verts[8 * 9];                   /* 8 vertices: pos3 tex2 color4 */
    for (int v = 0; v < 8; v++)
    {
        float *p = &verts[v * 9];
        p[0] = (float)(v & 1) * 0.5f - 0.25f;
        p[1] = (float)((v >> 1) & 1) * 0.5f - 0.25f;
        p[2] = (float)(v & 2) * 0.25f;
        p[3] = (float)(v & 1);
        p[4] = (float)((v >> 1) & 1);
        p[5] = 0.2f; p[6] = 0.4f; p[7] = 0.6f; p[8] = 1.0f;
    }
    unsigned short idx[12 * 3];
    for (int t = 0; t < 12; t++)
    {
        int base = (t * 3) % 6;
        idx[t * 3 + 0] = (unsigned short)base;
        idx[t * 3 + 1] = (unsigned short)(base + 1);
        idx[t * 3 + 2] = (unsigned short)(base + 2);
    }

    for (int i = 0; i < MGL_IDEACPU_VAOS; i++)
    {
        glBindVertexArray(vao[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
        glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[i]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, (void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 36, (void *)12);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 36, (void *)20);
    }

    GLint loc_mvp = glGetUniformLocation(prog, "uMVP");
    GLint loc_t0 = glGetUniformLocation(prog, "uTex0");
    GLint loc_t1 = glGetUniformLocation(prog, "uTex1");
    glUniform1i(loc_t0, 0);
    glUniform1i(loc_t1, 1);

    float mvp[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, tex[1]);

    /* warm up: compile pipelines, fill caches */
    for (int f = 0; f < 60; f++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < MGL_IDEACPU_VAOS; i++)
        {
            glBindVertexArray(vao[i]);
            for (int d = 0; d < DRAWS_PER_WALK; d++)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
        }
    }
    glFinish();

    /* the measured frames: 89 walks (VAO switch) + 356 draws, one uniform
     * update per walk like [ideas]' per-object matrix upload */
    for (int f = 0; f < 120; f++)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        for (int i = 0; i < MGL_IDEACPU_VAOS; i++)
        {
            glBindVertexArray(vao[i]);
            mvp[12] = (float)i * 0.01f;
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
            for (int d = 0; d < DRAWS_PER_WALK; d++)
                glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
        }
    }
    glFinish();

    mgl_profile_dump();

    /* sanity: something was drawn */
    GLubyte out[4] = { 0, 0, 0, 0 };
    glReadPixels(400, 300, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
    EXPECT_TRUE(out[0] != 26 || out[1] != 26 || out[2] != 26);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(ideas_shape_cpu);
    return mgl_test_summary();
}
