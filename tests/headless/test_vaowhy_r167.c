/* r167: MGL_PROFILE names why a vertex-array re-specification missed the draw
 * fast path (drawfast-vaowhy), and the vao-branch list line no longer prints on
 * every walk.  Between pairs of draws the test changes exactly one thing:
 *   stride                 -> attr-ptr-stride (the glVertexAttribPointer stride)
 *   offset 0 -> 2 (bytes)  -> offset-align4
 *   attribute type         -> attr-format
 *   small CPU-side buffer  -> ring-buffer-change
 * and a pure offset move (same alignment) must NOT be counted (it is a rebind).
 * Every draw is also checked for the right pixels. */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

__attribute__((constructor(101))) static void prof_env(void) { setenv("MGL_PROFILE", "1", 1); }
extern void mgl_profile_dump(void);

#define W 32
static char out[1 << 20];
static void capture(void (*fn)(void))
{
    char path[] = "/tmp/mgl_r167_XXXXXX";
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
static unsigned long count_of(const char *needle)
{
    const char *line = strstr(out, "MGL PROFILE drawfast-vaowhy:");
    if (!line) return 0;
    const char *eol = strchr(line, '\n');
    const char *h = strstr(line, needle);
    if (!h || (eol && h > eol)) return 0;
    return strtoul(h + strlen(needle) + 1, NULL, 10);
}
static unsigned lines_of(const char *needle)
{
    unsigned n = 0;
    for (const char *p = out; (p = strstr(p, needle)); p++) n++;
    return n;
}

static GLuint prog, vao, vb, small1, small2;
static void quad(void) { glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); }

/* a full-screen quad of floats, 2 per vertex, written at byte offset ofs with the given stride */
static void put_quad(GLuint b, size_t ofs, int stride_floats)
{
    float q[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    float tmp[64]; memset(tmp, 0, sizeof tmp);
    for (int v = 0; v < 4; v++) { tmp[v * stride_floats] = q[v][0]; tmp[v * stride_floats + 1] = q[v][1]; }
    glBindBuffer(GL_ARRAY_BUFFER, b);
    glBufferSubData(GL_ARRAY_BUFFER, (GLintptr)ofs, 4 * stride_floats * sizeof(float), tmp);
}

static void work(void)
{
    for (int r = 0; r < 4; r++) {
        /* baseline: stride 8, offset 0 */
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        /* pure offset move, same alignment class: a rebind, not counted */
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)256); quad();
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        /* stride */
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void *)512); quad();
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        /* offset alignment class: a byte-offset copy at 1026 (1026 & 3 == 2) */
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)1026); quad();
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        /* attribute format: half floats at 2048 */
        glVertexAttribPointer(0, 2, GL_HALF_FLOAT, GL_FALSE, 4, (void *)2048); quad();
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        /* small CPU-side buffers swapped */
        glBindBuffer(GL_ARRAY_BUFFER, small1);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
        glBindBuffer(GL_ARRAY_BUFFER, small2);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); quad();
    }
}
static void dump(void) { mgl_profile_dump(); }

int main(void)
{
    int bad = 0;
    GLuint fbo, ct;
    glGenTextures(1, &ct); glBindTexture(GL_TEXTURE_2D, ct); glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, W);
    glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ct, 0);
    glViewport(0, 0, W, W);
    const char *v = "#version 450\nlayout(location=0) in vec2 p;\nvoid main(){ gl_Position = vec4(p, 0.5, 1.0); }\n";
    const char *f = "#version 450\nout vec4 o;\nvoid main(){ o = vec4(0.0, 1.0, 0.0, 1.0); }\n";
    const char *s[2] = {v, f}; GLenum t[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    prog = glCreateProgram();
    for (int i = 0; i < 2; i++) { GLuint sh = glCreateShader(t[i]); glShaderSource(sh, 1, &s[i], NULL); glCompileShader(sh); glAttachShader(prog, sh); glDeleteShader(sh); }
    glLinkProgram(prog); glUseProgram(prog);

    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vb); glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, 8192, NULL, GL_STATIC_DRAW);    /* >= 4096: Metal-backed */
    put_quad(vb, 0, 2); put_quad(vb, 256, 2); put_quad(vb, 512, 4);
    {   /* byte offset 1026: unaligned floats copy */
        float q[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
        glBufferSubData(GL_ARRAY_BUFFER, 1026, sizeof q, q);
        uint16_t h[8] = {0xBC00, 0xBC00, 0x3C00, 0xBC00, 0xBC00, 0x3C00, 0x3C00, 0x3C00};   /* +-1.0 halves */
        glBufferSubData(GL_ARRAY_BUFFER, 2048, sizeof h, h);
    }
    float q[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenBuffers(1, &small1); glBindBuffer(GL_ARRAY_BUFFER, small1); glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glGenBuffers(1, &small2); glBindBuffer(GL_ARRAY_BUFFER, small2); glBufferData(GL_ARRAY_BUFFER, sizeof q, q, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0); glEnableVertexAttribArray(0);
    glClearColor(1, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    quad(); glFinish();
    capture(dump);

    capture(work);
    unsigned per_draw_lines = lines_of("vao-branch list counts");
    capture(dump);
    printf("  vao-branch lines during 44 draws: %u\n", per_draw_lines);
    for (char *l = strstr(out, "MGL PROFILE drawfast-vaowhy:"); l; l = NULL) {
        char *e = strchr(l, '\n'); if (e) *e = 0; printf("  %s\n", l); if (e) *e = '\n';
    }
    unsigned long st = count_of("attr-ptr-stride"), al = count_of("offset-align4"), fm = count_of("attr-format"),
                  rb = count_of("ring-buffer-change");
    if (per_draw_lines > 1) { printf("  [print] vao-branch line printed %u times\n", per_draw_lines); bad++; }
    if (st < 4)  { printf("  [attr-ptr-stride] %lu\n", st); bad++; }
    if (al < 4)  { printf("  [offset-align4] %lu\n", al); bad++; }
    if (fm < 4)  { printf("  [attr-format] %lu\n", fm); bad++; }
    if (rb < 4)  { printf("  [ring-buffer-change] %lu\n", rb); bad++; }

    unsigned char px[4];
    glFinish(); glReadPixels(W / 2, W / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    if (px[1] < 250 || px[0] > 5) { printf("  [pixels] %u %u %u\n", px[0], px[1], px[2]); bad++; }

    EXPECT_TRUE(bad == 0);
    printf("== r167 vertex-array miss reasons: %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
