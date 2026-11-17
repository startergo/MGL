/* r164: MGL_PROFILE call-site histogram.  Under MGL_PROFILE=1 the profile dump
 * names who ended render passes, flushed, started command buffers and forced a
 * full state re-apply.  Each cause is produced a known number of times between
 * draws, the dump is captured, and the named sites are checked:
 *   glClear            -> pass-end from the clear path
 *   glTexSubImage2D    -> pass-end from the texture upload path
 *   FBO re-attach      -> pass-end from the target-change path
 *   glFinish           -> flush
 *   glFinish           -> pass-end + re-apply from the flush
 *   glCopyBufferSubData / glReadPixels -> no "MGL TRAIL" debug line (those
 *                          now need MGL_TRAIL=1)
 */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

__attribute__((constructor(101))) static void prof_env(void) { setenv("MGL_PROFILE", "1", 1); unsetenv("MGL_TRAIL"); }
extern void mgl_profile_dump(void);

#define W 32

static GLuint mk(void)
{
    const char *v = "#version 450\nlayout(location=0) in vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
    const char *f = "#version 450\nlayout(binding=0) uniform sampler2D t;\nout vec4 o;\nvoid main(){ o = texture(t, vec2(0.5)); }\n";
    const char *s[2] = {v, f}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) { GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh); glAttachShader(p, sh); glDeleteShader(sh); }
    glLinkProgram(p);
    return p;
}

static char out[1 << 20];

/* run fn with stderr captured into out[] */
static void capture(void (*fn)(void))
{
    char path[] = "/tmp/mgl_r164_XXXXXX";
    int fd = mkstemp(path);
    fflush(stderr);
    int saved = dup(2);
    dup2(fd, 2);
    fn();
    fflush(stderr);
    dup2(saved, 2); close(saved);
    lseek(fd, 0, SEEK_SET);
    ssize_t n = read(fd, out, sizeof out - 1);
    out[n > 0 ? n : 0] = 0;
    close(fd); unlink(path);
}

/* the line after "MGL PROFILE sites <kind>/frame" block containing needle */
static int site_has(const char *kind, const char *needle)
{
    char hdr[64]; snprintf(hdr, sizeof hdr, "MGL PROFILE sites %s/frame", kind);
    const char *p = strstr(out, hdr);
    if (!p) return 0;
    const char *end = strstr(p + 1, "MGL PROFILE sites ");
    const char *hit = strstr(p, needle);
    return hit && (!end || hit < end);
}

static GLuint fbo, ct[2], tex, prog, vao, vbo, bsrc, bdst;
static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

static void work(void)
{
    unsigned char px[4 * 4 * 4], rb[4 * 4 * 4];
    memset(px, 200, sizeof px);
    for (int i = 0; i < 6; i++) {
        quad(); glClear(GL_COLOR_BUFFER_BIT);                                       /* clear */
        quad(); glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, px); /* upload */
        quad(); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct[(i + 1) & 1], 0); /* re-attach */
        quad(); glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, 64);  /* copy */
        quad(); glFinish();                                                         /* flush */
        quad(); glReadPixels(0, 0, 4, 4, GL_RGBA, GL_UNSIGNED_BYTE, rb);              /* readback */
    }
}
static void dump(void) { mgl_profile_dump(); }

int main(void)
{
    int bad = 0;
    for (int i = 0; i < 2; i++) {
        glGenTextures(1, &ct[i]); glBindTexture(GL_TEXTURE_2D, ct[i]); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    }
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct[0], 0);
    glViewport(0, 0, W, W);
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    float q[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0); glEnableVertexAttribArray(0);
    glGenBuffers(1, &bsrc); glBindBuffer(GL_COPY_READ_BUFFER, bsrc); glBufferData(GL_COPY_READ_BUFFER, 256, NULL, GL_STATIC_DRAW);
    glGenBuffers(1, &bdst); glBindBuffer(GL_COPY_WRITE_BUFFER, bdst); glBufferData(GL_COPY_WRITE_BUFFER, 256, NULL, GL_STATIC_DRAW);
    prog = mk(); glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tex);
    quad(); glFinish();
    capture(dump);                         /* drop the setup counts */

    capture(work);
    int trail = strstr(out, "MGL TRAIL") != NULL;
    if (trail) printf("  [trail] debug trace printed without MGL_TRAIL\n");
    bad += trail;

    capture(dump);
    /* print the site blocks for the log */
    for (char *l = strtok(out, "\n"); l; l = strtok(NULL, "\n"))
        if (strstr(l, "MGL PROFILE sites") || strstr(l, "MGL PROFILE   "))
            printf("  %s\n", l);
    capture(dump);   /* re-capture is empty now: re-run work + dump for the checks */
    capture(work);
    capture(dump);

    struct { const char *kind, *needle, *what; } want[] = {
        { "pass-end", "fbClear",            "glClear ends the pass" },
        { "pass-end", "texdata",            "texture upload ends the pass" },
        { "pass-end", "fbTargetsChanged",   "re-attach ends the pass" },
        { "pass-end", "extBufFlush",        "a flush ends the pass" },
        { "flush",    "c_mtl_flush",        "glFinish flushes" },
        { "reapply",  "extBufFlush",        "a flush forces re-apply" },
        { "new-cmdbuf", "extBufEnsureCommandBuffer", "the next draw starts a command buffer" },
    };
    for (unsigned i = 0; i < sizeof want / sizeof want[0]; i++)
        if (!site_has(want[i].kind, want[i].needle)) {
            printf("  [site] %s: no '%s' under sites %s\n", want[i].what, want[i].needle, want[i].kind);
            bad++;
        }
    EXPECT_TRUE(bad == 0);
    printf("== r164 profile sites: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
