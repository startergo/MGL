/* test_compat_ideaspec.c — exact [ideas] respec shape on a compat context:
 * one VAO, one VBO, per-mesh glVertexAttribPointer with a DIFFERENT byte
 * offset (glmark2 SceneIdeas/SGILogo pattern), indexed + arrays draws.
 * Prints the MGL_PROFILE per-draw buckets (run directly, not via run.sh):
 *
 *   MGL_PROFILE=1 build/tests/test_compat_ideaspec 2>&1 | grep 'MGL PROFILE'
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include <time.h>

extern void mgl_profile_dump(void);

#ifndef MGL_IDEASPEC_MESHES
#define MGL_IDEASPEC_MESHES 89
#endif
#ifndef MGL_IDEASPEC_EXTRA
#define MGL_IDEASPEC_EXTRA 3         /* draws per mesh without respec */
#endif

static const char *vs_src =
    "#version 410 core\n"
    "layout(location=0) in vec3 aPosition;\n"
    "layout(location=1) in vec3 aNormal;\n"
    "uniform mat4 uMVP; uniform mat4 uNormalMat;\n"
    "out vec3 vN;\n"
    "void main() { vN = mat3(uNormalMat) * aNormal;\n"
    "  gl_Position = uMVP * vec4(aPosition, 1.0); }\n";

static const char *fs_src =
    "#version 410 core\n"
    "in vec3 vN; out vec4 fragColor;\n"
    "uniform vec4 uColor; uniform sampler2D uTex;\n"
    "void main() { fragColor = uColor * (0.5 + 0.5 * vN.z) + 0.001 * texture(uTex, vec2(0.5)); }\n";

TEST(ideas_respec_shape_cpu)
{
    void *cc = getenv("IDEASPEC_CORE") ? NULL : mglCompatCreateContext();
    EXPECT_TRUE(getenv("IDEASPEC_CORE") != NULL || cc != NULL);
    if (!cc && !getenv("IDEASPEC_CORE"))
        return;

    /* offset-only respec on a COMPAT context: the same VBO, a different byte
     * offset.  The VAO-unchanged hash does not cover the pointer offset, so
     * this must still reach the encoder (the drawn mesh must change). */
    if (getenv("IDEASPEC_OFFSETTEST"))
    {
        static const char *ovs =
            "#version 410 core\nlayout(location=0) in vec3 aPosition;\n"
            "uniform vec4 uColor; out vec3 vN;\n"
            "void main() { gl_Position = vec4(aPosition, 1.0); }\n";
        static const char *ofs_ =
            "#version 410 core\nin vec3 vN; uniform vec4 uColor; out vec4 fragColor;\n"
            "void main() { fragColor = uColor; }\n";
        GLuint p2 = mgl_make_program(ovs, ofs_);
        EXPECT_TRUE(p2 != 0);
        glUseProgram(p2);
        GLint lc = glGetUniformLocation(p2, "uColor");

        GLuint f2, c2;
        glGenTextures(1, &c2);
        glBindTexture(GL_TEXTURE_2D, c2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glGenFramebuffers(1, &f2);
        glBindFramebuffer(GL_FRAMEBUFFER, f2);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, c2, 0);
        glViewport(0, 0, 64, 64);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        float meshA[9] = { -0.9f,-0.9f,0,  -0.1f,-0.9f,0,  -0.5f,0.9f,0 };
        float meshB[9] = {  0.1f,-0.9f,0,   0.9f,-0.9f,0,   0.5f,0.9f,0 };
        GLuint v2;
        glGenBuffers(1, &v2);
        glBindBuffer(GL_ARRAY_BUFFER, v2);
        glBufferData(GL_ARRAY_BUFFER, 1024, NULL, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof meshA, meshA);
        glBufferSubData(GL_ARRAY_BUFFER, 512, sizeof meshB, meshB);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glUniform4f(lc, 1, 0, 0, 1);
        for (int i = 0; i < 30; i++) glDrawArrays(GL_TRIANGLES, 0, 3);
        glFinish();
        EXPECT_PIXEL(16, 16, 255, 0, 0, 255, 8);        /* left triangle */
        EXPECT_PIXEL(48, 16, 0, 0, 0, 255, 8);          /* nothing right yet */

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)512);
        glUniform4f(lc, 0, 1, 0, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        EXPECT_PIXEL(48, 16, 0, 255, 0, 255, 8);        /* mesh B now */

        /* repeat the respec pattern so the drawfast stats are readable */
        for (int f = 0; f < 120; f++)
        {
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)512);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }
        glFinish();
        mgl_profile_dump();
        printf("  offsettest done\n");
        return;
    }

    GLuint prog = mgl_make_program(vs_src, fs_src);
    EXPECT_TRUE(prog != 0);
    glUseProgram(prog);

    GLuint tex;
    glGenTextures(1, &tex);
    unsigned char px[16 * 16 * 4];
    for (int i = 0; i < (int)sizeof px; i++) px[i] = (unsigned char)(i * 3);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint fbo, color;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glViewport(0, 0, 800, 600);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    /* one interleaved VBO per mesh (the real [ideas] shape: each mesh has its
     * own buffer object, identical layout, offset 0 — so the PSO key is
     * CONSTANT and the walks are driven by buffer-identity changes) */
    GLuint vbo[MGL_IDEASPEC_MESHES], ebo;
    const int verts_per_mesh = 24;
    glGenBuffers(MGL_IDEASPEC_MESHES, vbo);
    for (int m = 0; m < MGL_IDEASPEC_MESHES; m++)
    {
        float v[verts_per_mesh * 6];
        for (int i = 0; i < verts_per_mesh; i++)
        {
            float *p = &v[i * 6];
            p[0] = ((i % 3) - 1) * 0.03f + 0.005f * m;
            p[1] = ((i % 5) - 2) * 0.02f;
            p[2] = ((i % 7) - 3) * 0.05f;
            p[3] = p[4] = p[5] = 0.5f;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo[m]);
        glBufferData(GL_ARRAY_BUFFER, 32768, NULL, GL_STATIC_DRAW);   /* real MTL buffer, like [ideas]' meshes */
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof v, v);
    }
    unsigned short idx[12 * 3];
    for (int t = 0; t < 12; t++)
    {
        idx[t * 3 + 0] = (unsigned short)((t * 3) % 21);
        idx[t * 3 + 1] = (unsigned short)((t * 3 + 1) % 21);
        idx[t * 3 + 2] = (unsigned short)((t * 3 + 2) % 21);
    }
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 32768, NULL, GL_STATIC_DRAW);   /* real MTL buffer */
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof idx, idx);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void *)12);

    GLint loc_mvp = glGetUniformLocation(prog, "uMVP");
    GLint loc_nm = glGetUniformLocation(prog, "uNormalMat");
    GLint loc_c = glGetUniformLocation(prog, "uColor");
    GLint loc_t = glGetUniformLocation(prog, "uTex");
    glUniform1i(loc_t, 0);
    float mvp[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };

    int warm = 60, frames = 120;
    uint64_t wall0 = 0;
    for (int f = -warm; f < frames; f++)
    {
        if (f == 0)
            wall0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        glClear(GL_COLOR_BUFFER_BIT);
        for (int m = 0; m < MGL_IDEASPEC_MESHES; m++)
        {
            /* the [ideas] respec: identical layout, different buffer object */
            glBindBuffer(GL_ARRAY_BUFFER, vbo[m]);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, 0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void *)12);
            mvp[12] = (float)m * 0.01f;
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
            glUniformMatrix4fv(loc_nm, 1, GL_FALSE, mvp);
            glUniform4f(loc_c, 0.4f, 0.5f, 0.6f, 1.0f);
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, 0);
#ifdef IDEASPEC_ARRAYS
            for (int d = 1; d < MGL_IDEASPEC_EXTRA; d++)
                glDrawArrays(GL_TRIANGLES, d * 6, 12);
#endif
        }
    }
    uint64_t wall1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    printf("WALL %.3f ms/frame (%d frames)\n", (double)(wall1 - wall0) / 1e6 / frames, frames);
    glFinish();

    if (getenv("IDEASPEC_PIXDUMP"))
    {
        GLubyte p[4];
        for (int y = 580; y > 0; y -= 100)
        {
            for (int x = 20; x < 800; x += 60)
            {
                glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
                printf("%3d,%3d,%3d ", p[0], p[1], p[2]);
            }
            printf("\n");
        }
    }

    mgl_profile_dump();

    GLubyte out[4] = { 0, 0, 0, 0 };
    /* the meshes are interleaved 3-float pairs around the origin; the pixel at
     * (0.2/800-ish right of center) is covered by some triangle, so the buffer
     * must not be uniformly the clear color */
    int nonclear = 0;
    for (int i = 0; i < 40; i++)
    {
        glReadPixels(300 + i * 5, 300, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
        if (out[0] != 26 || out[1] != 26 || out[2] != 26)
            nonclear++;
    }
    EXPECT_TRUE(nonclear >= 1);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    RUN(ideas_respec_shape_cpu);
    return mgl_test_summary();
}
