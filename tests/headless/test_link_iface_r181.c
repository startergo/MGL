/* r181: rsvga's shader warm-up (slowdraw 8098e47, tiny_partner()) linked a guest
 * pixel shader that reads a varying at location 3 with a throwaway vertex shader
 * that writes gl_Position only, and drew with it. MGL accepted the link; Metal
 * then refused the render pipeline ("Link failed: fragment input user(locn3) was
 * not found in vertex shader outputs"), and MGL's pipeline error handling called
 * recoverFromMetalError -> resetMetalState: the shared command queue was
 * recreated ("MGL AGX RECOVERY") - 578 times in one glmark2 run, the score down
 * to 916 from 960. A refused pipeline is the program's problem, not a GPU fault.
 *
 * Per case (SPIR-V, rsvga's form; GLSL), five times in a row, stderr captured:
 *   - the link status is reported (GL 4.6 7.4.1: a fragment input that is read
 *     but has no matching vertex output fails the link)
 *   - a draw with the program: if the link failed, GL_INVALID_OPERATION and
 *     nothing reaches Metal; never a command-queue recreation either way
 *   - then a matching program draws green: later draws still work */
#include "mgl_test.h"
#include <stdint.h>
#include <unistd.h>
#include "spv_link_iface.h"

#define W 8
static int saved_err = -1, cap_fd = -1;
static char cap_path[] = "/tmp/mgl_r181_capXXXXXX";
static void cap_begin(void)
{
    fflush(stderr);
    if (cap_fd < 0) cap_fd = mkstemp(cap_path);
    ftruncate(cap_fd, 0); lseek(cap_fd, 0, SEEK_SET);
    saved_err = dup(2); dup2(cap_fd, 2);
}
static int cap_count(const char *needle, char *buf)
{
    int n = 0;
    for (const char *p = buf; (p = strstr(p, needle)); p++) n++;
    return n;
}
static char capbuf[1 << 20];
static void cap_end(void)
{
    fflush(stderr); dup2(saved_err, 2); close(saved_err);
    ssize_t n = pread(cap_fd, capbuf, sizeof capbuf - 1, 0);
    capbuf[n > 0 ? n : 0] = 0;
}

static GLuint spv(GLenum ty, const unsigned char *b, size_t n)
{
    GLuint s = glCreateShader(ty);
    glShaderBinary(1, &s, GL_SHADER_BINARY_FORMAT_SPIR_V, b, (GLsizei)n);
    glSpecializeShader(s, "main", 0, NULL, NULL);
    return s;
}
static GLuint glsl(GLenum ty, const char *src)
{
    GLuint s = glCreateShader(ty); glShaderSource(s, 1, &src, NULL); glCompileShader(s);
    return s;
}
static GLuint link2(GLuint vs, GLuint fs, GLint *status)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, status);
    return p;
}

static GLuint vao, fbo, rt;
static void draw_with(GLuint prog)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, W);
    glBindVertexArray(vao);
    glUseProgram(prog);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
static int green_ok(GLuint good)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glClearColor(1, 0, 1, 1); glClear(GL_COLOR_BUFFER_BIT);
    draw_with(good);
    unsigned char px[4]; glFinish();
    glReadPixels(W / 2, W / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return px[0] < 3 && px[1] > 252 && px[2] < 3;
}

static int run_case(const char *name, GLuint bad_vs, GLuint fs, GLuint good)
{
    int bad = 0;
    GLint st = -1;
    cap_begin();
    GLenum draw_err[5];
    for (int i = 0; i < 5; i++) {
        GLuint p = link2(bad_vs, fs, &st);
        while (glGetError()) {}
        draw_with(p);
        draw_err[i] = glGetError();
        glFinish();
        glDeleteProgram(p);
    }
    cap_end();
    int rec = cap_count("AGX RECOVERY", capbuf);
    int pfail = cap_count("Pipeline creation failed", capbuf);
    int ok_draw = st ? 1 : (draw_err[0] == GL_INVALID_OPERATION);
    printf("  %-8s link %s, draw GL error %#x; logged: %d \"AGX RECOVERY\", %d \"Pipeline creation failed\"\n",
           name, st ? "SUCCEEDED" : "failed", draw_err[0], rec, pfail);
    if (rec) { printf("  %-8s FAIL: the command queue was recreated\n", name); bad++; }
    if (!ok_draw) { printf("  %-8s FAIL: draw with an unlinked program must be GL_INVALID_OPERATION\n", name); bad++; }
    int g = green_ok(good);
    printf("  %-8s a matching program afterwards draws green: %s\n", name, g ? "ok" : "FAIL");
    bad += !g;
    return bad;
}

int main(void)
{
    int bad = 0;
    glGenVertexArrays(1, &vao);
    glCreateTextures(GL_TEXTURE_2D, 1, &rt); glTextureStorage2D(rt, 1, GL_RGBA8, W, W);
    glCreateFramebuffers(1, &fbo); glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, rt, 0);

    GLint st;
    GLuint fs_spv = spv(GL_FRAGMENT_SHADER, spv_need3_frag, sizeof spv_need3_frag);
    GLuint good_spv = link2(spv(GL_VERTEX_SHADER, spv_good_vert, sizeof spv_good_vert), fs_spv, &st);
    if (!st) { printf("  setup: the matching SPIR-V pair failed to link\n"); bad++; }
    printf("section 1: SPIR-V (rsvga's form): vertex writes gl_Position only, fragment reads location 3\n");
    bad += run_case("SPIR-V", spv(GL_VERTEX_SHADER, spv_tiny_vert, sizeof spv_tiny_vert), fs_spv, good_spv);

    const char *vs_tiny = "#version 450\nvoid main(){ vec2 q = vec2((gl_VertexID&1)*4-1, (gl_VertexID&2)*2-1); gl_Position = vec4(q,0,1); }\n";
    const char *vs_good = "#version 450\nlayout(location=3) out vec4 v;\nvoid main(){ vec2 q = vec2((gl_VertexID&1)*4-1, (gl_VertexID&2)*2-1); gl_Position = vec4(q,0,1); v = vec4(0,1,0,1); }\n";
    const char *fs3 = "#version 450\nlayout(location=3) in vec4 v;\nlayout(location=0) out vec4 o;\nvoid main(){ o = v; }\n";
    GLuint fs_glsl = glsl(GL_FRAGMENT_SHADER, fs3);
    GLuint good_glsl = link2(glsl(GL_VERTEX_SHADER, vs_good), fs_glsl, &st);
    if (!st) { printf("  setup: the matching GLSL pair failed to link\n"); bad++; }
    printf("section 2: GLSL, the same pair\n");
    bad += run_case("GLSL", glsl(GL_VERTEX_SHADER, vs_tiny), fs_glsl, good_glsl);

    unlink(cap_path);
    printf("%s: %d failure(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
