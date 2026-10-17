/* scratch diagnostics — not committed */
#include "mgl_test.h"

TEST(vs_uniform_debug)
{
    const char *vs = "#version 450 core\n"
        "layout(location=0) in vec2 pos;\n"
        "uniform float k;\n"
        "uniform vec2 offs;\n"
        "void main(){ gl_Position = vec4(pos * k + offs, 0.0, 1.0); }\n";
    const char *fs = "#version 450 core\n"
        "layout(location=0) out vec4 color;\n"
        "uniform vec4 tint;\n"
        "uniform float scale;\n"
        "void main(){ color = tint * scale; }\n";
    GLuint p = mgl_make_program(vs, fs);
    if (!p) return;
    glUseProgram(p);
    GLint n = 0;
    glGetProgramInterfaceiv(p, GL_UNIFORM, GL_ACTIVE_RESOURCES, &n);
    fprintf(stderr, "  [dbg] ACTIVE_UNIFORMS=%d\n", n);
    for (int i = 0; i < n; i++) {
        char nm[128]; GLsizei len; GLint sz; GLenum ty; GLint loc;
        nm[0] = '?'; nm[1] = 0;
        glGetActiveUniform(p, i, sizeof nm, &len, &sz, &ty, nm);
        fprintf(stderr, "  [dbg] got nm='%s' len=%d sz=%d ty=%06x err=0x%x\n", nm, len, sz, ty, glGetError());
        loc = glGetUniformLocation(p, nm);
        fprintf(stderr, "  [dbg] [%d] %-10s type=%06x size=%d loc=%d\n", i, nm, ty, sz, loc);
    }
    GLint lk = glGetUniformLocation(p, "k");
    fprintf(stderr, "  [dbg] loc(k)=%d err=0x%x\n", lk, glGetError());
    glUniform1f(lk, 1.0f);
    fprintf(stderr, "  [dbg] after glUniform1f(k): err=0x%x\n", glGetError());
    GLint lo = glGetUniformLocation(p, "offs");
    glUniform2f(lo, 0.0f, 0.0f);
    fprintf(stderr, "  [dbg] after glUniform2f(offs): err=0x%x\n", glGetError());
    GLint lt = glGetUniformLocation(p, "tint");
    glUniform4f(lt, 0, 1, 0.5f, 1);
    fprintf(stderr, "  [dbg] after glUniform4f(tint): err=0x%x\n", glGetError());
}

int main(void)
{
    RUN(vs_uniform_debug);
    return mgl_test_summary();
}
