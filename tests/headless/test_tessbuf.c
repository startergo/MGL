/* test_tessbuf.c — tess control/eval SSBO write then map read (CTS const-expr executor) */
#include "mgl_test.h"
#include <string.h>

static const char *vs =
    "#version 460 core\n"
    "void main() { gl_Position = vec4(float(gl_VertexID), 0.0, 0.0, 1.0); }\n";

static const char *tcs =
    "#version 460 core\n"
    "layout(vertices = 1) out;\n"
    "layout(std430) buffer Out { float o; };\n"
    "void main() {\n"
    "    gl_TessLevelInner[0] = gl_TessLevelInner[1] = 1.0;\n"
    "    gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = 1.0;\n"
    "    gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = 1.0;\n"
    "    if (gl_InvocationID == 0) o = 42.0;\n"
    "}\n";

static const char *tes =
    "#version 460 core\n"
    "layout(triangles, equal_spacing, ccw) in;\n"
    "layout(std430) buffer Out { float o; };\n"
    "void main() { if (gl_TessCoord.x == 0.0) o = 42.0; }\n";

static int run_case(const char *tcs_src, const char *tes_src, float *out)
{
    GLuint prog = glCreateProgram();
    struct { GLenum t; const char *src; } st[3] = {
        { GL_VERTEX_SHADER, vs },
        { GL_TESS_CONTROL_SHADER, tcs_src },
        { GL_TESS_EVALUATION_SHADER, tes_src },
    };
    for (int i = 0; i < 3; i++)
    {
        GLuint s = glCreateShader(st[i].t);
        const char *src = st[i].src;
        glShaderSource(s, 1, &src, NULL);
        glCompileShader(s);
        GLint cok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &cok);
        if (!cok)
        {
            char slog[2048] = "";
            glGetShaderInfoLog(s, sizeof slog - 1, NULL, slog);
            fprintf(stderr, "  shader 0x%x compile: %s", st[i].t, slog);
        }
        glAttachShader(prog, s);
        glDeleteShader(s);
    }
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        glGetProgramInfoLog(prog, sizeof log - 1, NULL, log);
        fprintf(stderr, "  link: %s", log);
        glDeleteProgram(prog);
        return -1;
    }

    GLuint buf = 0;
    glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 64, NULL, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);

    GLuint va = 0;
    glGenVertexArrays(1, &va);
    glBindVertexArray(va);

    glUseProgram(prog);
    glPatchParameteri(GL_PATCH_VERTICES, 1);
    if (getenv("MGL_TESSBUF_NODISCARD") == NULL) glEnable(GL_RASTERIZER_DISCARD);
    glDrawArrays(GL_PATCHES, 0, 1);
    GLenum derr = glGetError();
    if (derr != GL_NO_ERROR) fprintf(stderr, "  draw error 0x%x\n", derr);
    glDisable(GL_RASTERIZER_DISCARD);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    float *q = (float *)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 64, GL_MAP_READ_BIT);
    GLenum merr = glGetError();
    if (merr != GL_NO_ERROR) fprintf(stderr, "  map error 0x%x\n", merr);
    if (!q) { fprintf(stderr, "  map returned NULL (err 0x%x)\n", merr); }
    else { *out = q[0]; glUnmapBuffer(GL_SHADER_STORAGE_BUFFER); }

    glDeleteVertexArrays(1, &va);
    glDeleteBuffers(1, &buf);
    glDeleteProgram(prog);
    return merr ? -2 : 0;
}

TEST(tess_control_ssbo_map_read)
{
    float out = 0.0f;
    int rc = run_case(tcs, tes, &out);
    EXPECT_EQ_INT(rc, 0);
    EXPECT_EQ_INT((int)out, 42);
}

int main(void)
{
    RUN(tess_control_ssbo_map_read);
    return mgl_test_summary();
}
