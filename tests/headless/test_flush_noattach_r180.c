/* r180: a flush-type call while a framebuffer with NO attachments is bound (and
 * no VAO, with dirty state) is legal GL, and there is nothing for a render pass
 * to carry: no targets, so no clear either. MGL's non-draw state pass
 * (processGLState:false) still tried to build a render encoder for it and
 * logged, every time:
 *   MGL ERROR: no render targets for the bound draw framebuffer
 *   MGL WARNING: Failed to create render encoder - continuing with degraded functionality
 * (rsvga: 2,500 of them under AquaMark3 / a DirectDraw test, 11 in 3DMark06's
 * start-up, 2026-11-20.) Every caller that reaches that pass is exercised here
 * with stderr captured: glFlush, glFinish, glFenceSync (+ glClientWaitSync),
 * glGetSynciv, glDeleteTextures, glFlushMappedBufferRange. None may log.
 * Last section: a clear pending on a REAL framebuffer still lands when a flush
 * on the empty one comes in between. */
#include "mgl_test.h"
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>

#define W 16
static int saved_err = -1;
static char cap_path[] = "/tmp/mgl_r180_capXXXXXX";
static int cap_fd = -1;

static void cap_begin(void)
{
    fflush(stderr);
    if (cap_fd < 0) cap_fd = mkstemp(cap_path);
    ftruncate(cap_fd, 0); lseek(cap_fd, 0, SEEK_SET);
    saved_err = dup(2);
    dup2(cap_fd, 2);
}
/* returns the number of MGL encoder complaints logged since cap_begin */
static int cap_end(void)
{
    fflush(stderr);
    dup2(saved_err, 2); close(saved_err);
    char buf[65536]; ssize_t n = pread(cap_fd, buf, sizeof buf - 1, 0);
    if (n < 0) n = 0;
    buf[n] = 0;
    int hits = 0;
    for (const char *p = buf; (p = strstr(p, "no render targets for the bound draw framebuffer")); p++) hits++;
    for (const char *p = buf; (p = strstr(p, "Failed to create render encoder")); p++) hits++;
    return hits;
}

static GLuint empty_fbo;
/* the state rsvga is in: an attachment-less draw framebuffer, no VAO, dirty state */
static void arm(void)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, empty_fbo);
    glBindVertexArray(0);
    static int flip;
    if (flip ^= 1) glEnable(GL_BLEND); else glDisable(GL_BLEND);
}

static int check(const char *what, int hits)
{
    GLenum e = glGetError();
    int ok = hits == 0 && e == GL_NO_ERROR;
    printf("  %-44s logged %d encoder complaint(s), GL error %#x  %s\n", what, hits, e, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main(void)
{
    int bad = 0;
    glCreateFramebuffers(1, &empty_fbo);
    GLenum st = glCheckNamedFramebufferStatus(empty_fbo, GL_FRAMEBUFFER);
    printf("empty framebuffer status %#x (incomplete: missing attachment is expected)\n", st);
    while (glGetError()) {}

    printf("section 1: flush-type calls with an attachment-less framebuffer bound\n");
    arm(); cap_begin(); glFlush(); bad += check("glFlush", cap_end());
    arm(); cap_begin(); glFinish(); bad += check("glFinish", cap_end());
    arm(); cap_begin();
    GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glClientWaitSync(s, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
    bad += check("glFenceSync + glClientWaitSync", cap_end());
    arm(); cap_begin();
    GLint sv = 0; glGetSynciv(s, GL_SYNC_STATUS, 1, NULL, &sv);
    bad += check("glGetSynciv", cap_end());
    glDeleteSync(s);
    GLuint t; glCreateTextures(GL_TEXTURE_2D, 1, &t); glTextureStorage2D(t, 1, GL_RGBA8, W, W);
    arm(); cap_begin(); glDeleteTextures(1, &t); bad += check("glDeleteTextures", cap_end());
    GLuint b; glCreateBuffers(1, &b);
    glNamedBufferStorage(b, 256, NULL, GL_MAP_WRITE_BIT | GL_DYNAMIC_STORAGE_BIT);
    void *m = glMapNamedBufferRange(b, 0, 256, GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT);
    if (m) memset(m, 7, 256);
    arm(); cap_begin(); glFlushMappedNamedBufferRange(b, 0, 256); bad += check("glFlushMappedBufferRange", cap_end());
    glUnmapNamedBuffer(b); glDeleteBuffers(1, &b);
    while (glGetError()) {}

    printf("section 2: a clear pending on a real framebuffer survives a flush on the empty one\n");
    GLuint ct, fbo;
    glCreateTextures(GL_TEXTURE_2D, 1, &ct); glTextureStorage2D(ct, 1, GL_RGBA8, W, W);
    glCreateFramebuffers(1, &fbo); glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, ct, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, W);
    glClearColor(1, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);    /* deferred into the next pass */
    arm(); cap_begin(); glFlush(); int hits = cap_end();
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    unsigned char px[4] = { 0 }; glFinish();
    glReadPixels(W / 2, W / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    int ok = px[0] == 255 && px[1] == 0 && px[2] == 0 && hits == 0;
    printf("  %-44s %d %d %d %d, logged %d  %s\n", "red clear, then flush on the empty fbo", px[0], px[1], px[2], px[3], hits,
           ok ? "ok" : "FAIL");
    bad += !ok;

    unlink(cap_path);
    printf("%s: %d failure(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
