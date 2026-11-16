/* r153: per-buffer pool of retired Metal storages.  A CPU write into a uniform
 * buffer the GPU still reads (rename) and a glBufferData re-specification both give
 * the buffer fresh storage; that storage now comes from the buffer's own retired
 * storages once their command buffer completed, instead of a new MTLBuffer.
 * Checks: every frame renders the LAST value written (no stale or recycled
 * contents), a partial write keeps the latest contents of the rest of the buffer,
 * the pool is actually reused, re-specifying to another size and deleting a buffer
 * with a full pool are safe. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

static const char *vs_src =
    "#version 450\nlayout(location=0) in vec4 pos;\n"
    "void main(){ gl_Position = pos; }\n";
/* colour = a.c[0].rgb picked by b.sel: the fragment reads the FAR end of the
 * buffer too (a.c[255]) so a partial write must keep it */
static const char *fs_src =
    "#version 450\nout vec4 o;\n"
    "layout(std140, binding=0) uniform A { vec4 c[256]; } a;\n"
    "void main(){ o = vec4(a.c[0].r, a.c[255].g, a.c[0].b, 1.0); }\n";

static GLuint ubo;

static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

static int px_is(int r, int g, int b, const char *what, int frame)
{
    unsigned char p[4] = {0};
    glReadPixels(8, 8, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    int ok = abs(p[0] - r) <= 2 && abs(p[1] - g) <= 2 && abs(p[2] - b) <= 2;
    if (!ok) printf("  [%s frame %d] %d %d %d, want %d %d %d\n", what, frame, p[0], p[1], p[2], r, g, b);
    return ok ? 0 : 1;
}

int main(void)
{
    uint64_t *st_alloc = dlsym(RTLD_DEFAULT, "mgl_bufpool_stat_alloc");
    uint64_t *st_reuse = dlsym(RTLD_DEFAULT, "mgl_bufpool_stat_reuse");
    int bad = 0;
    GLuint ct, fbo, vao, vbo, prog;
    glCreateTextures(GL_TEXTURE_2D, 1, &ct); glTextureStorage2D(ct, 1, GL_RGBA8, 16, 16);
    glCreateFramebuffers(1, &fbo); glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, ct, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, 16, 16);
    const char *s[2] = {vs_src, fs_src}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    prog = glCreateProgram();
    for (int i = 0; i < 2; i++) { GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh); glAttachShader(prog, sh); glDeleteShader(sh); }
    glLinkProgram(prog); glUseProgram(prog);
    float v[16] = {-1,-1,0,1, 1,-1,0,1, -1,1,0,1, 1,1,0,1};
    glCreateVertexArrays(1, &vao); glBindVertexArray(vao);
    glCreateBuffers(1, &vbo); glNamedBufferStorage(vbo, sizeof v, v, 0);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, 16);
    glVertexArrayAttribFormat(vao, 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vao, 0, 0); glEnableVertexArrayAttrib(vao, 0);

    static float data[256 * 4];
    memset(data, 0, sizeof data);
    data[255 * 4 + 1] = 0.5f;                       /* far end: green 128 */
    glCreateBuffers(1, &ubo);
    glNamedBufferData(ubo, sizeof data, data, GL_DYNAMIC_DRAW);   /* 4 KB: a real Metal buffer */
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo);
    uint64_t r0 = st_reuse ? *st_reuse : 0;

    /* 1. rename: 30 frames x 50 partial writes (first vec4 only) + draws; the far
     *    end written once must survive every rename, and each frame shows the last
     *    value written */
    for (int f = 0; f < 30; f++)
    {
        int last = 0;
        for (int d = 0; d < 50; d++)
        {
            last = (f * 50 + d) % 250;
            float c0[4] = { (float)last / 250.0f, 0.0f, (float)(250 - last) / 250.0f, 1.0f };
            glNamedBufferSubData(ubo, 0, sizeof c0, c0);
            quad();
        }
        glFlush();
        if ((f % 5) == 4)
        {
            glFinish();
            bad += px_is((int)(last / 250.0f * 255.0f + 0.5f), 128, (int)((250 - last) / 250.0f * 255.0f + 0.5f), "rename", f);
        }
    }
    glFinish();
    if (st_reuse && *st_reuse == r0) { printf("  [rename] pool never reused a storage\n"); bad++; }

    /* 2. re-specification (glBufferData every draw) */
    for (int f = 0; f < 10; f++)
    {
        int last = 0;
        for (int d = 0; d < 40; d++)
        {
            last = (f * 40 + d) % 250;
            data[0] = (float)last / 250.0f; data[2] = 0.0f; data[255 * 4 + 1] = 1.0f;
            glNamedBufferData(ubo, sizeof data, data, GL_DYNAMIC_DRAW);
            quad();
        }
        glFlush();
        glFinish();
        bad += px_is((int)(last / 250.0f * 255.0f + 0.5f), 255, 0, "respec", f);
    }

    /* 3. re-specify to another size: the pooled storages of the old size go */
    static float big[1024 * 4];
    memset(big, 0, sizeof big);
    big[0] = 1.0f; big[255 * 4 + 1] = 0.0f; big[2] = 1.0f;
    for (int d = 0; d < 20; d++) { glNamedBufferData(ubo, sizeof big, big, GL_DYNAMIC_DRAW); quad(); }
    glFinish();
    bad += px_is(255, 0, 255, "resize", 0);

    /* 4. deleting a buffer whose pool holds storages */
    glDeleteBuffers(1, &ubo);
    glFinish();
    if (glGetError() != GL_NO_ERROR) { printf("  [gl error]\n"); bad++; }
    if (st_alloc && st_reuse)
        printf("  pool: %llu allocations, %llu reuses\n", (unsigned long long)*st_alloc, (unsigned long long)(*st_reuse));

    EXPECT_EQ_INT(bad, 0);
    return mgl_test_summary();
}
