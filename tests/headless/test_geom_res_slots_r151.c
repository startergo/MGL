/* r151: geometry-shader user resources in the compute emulation's argument table.
 * (1) more than 8 storage blocks in a GS (GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS = 16):
 *     the pin table held 8, the rest collided on Metal slots 0..7, the kernel failed to
 *     compile and the draw was dropped (KHR-GL46.geometry_shader.api.max_shader_storage_blocks).
 * (2) a GS uniform block at binding 20..28: the emulation's own SSBOs used GLSL bindings
 *     18..28, inside the legal UBO binding range (0..511 since compat r111). */
#include "mgl_test.h"
#include <stdio.h>
#include <string.h>

static const char *vs = "#version 450\nvoid main(){ gl_Position = vec4(0.0); }\n";
static const char *fs = "#version 450\nout vec4 c; void main(){ c = vec4(1.0); }\n";
static char gs[16384];

static GLuint link_gs(void)
{
    const char *src[3] = {gs, vs, fs};
    GLenum st[3] = {GL_GEOMETRY_SHADER, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
    GLuint p = glCreateProgram();
    for (int i = 0; i < 3; i++) {
        GLuint s = glCreateShader(st[i]);
        glShaderSource(s, 1, &src[i], NULL); glCompileShader(s); glAttachShader(p, s); glDeleteShader(s);
    }
    const char *fv[1] = {"gl_Position"};
    glTransformFeedbackVaryings(p, 1, fv, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char l[1024] = ""; glGetProgramInfoLog(p, 1024, NULL, l); printf("  [link] %s\n", l); glDeleteProgram(p); return 0; }
    return p;
}

/* one point through the GS with XFB + rasterizer discard; returns gl_Position.x */
static float draw_point(GLuint p)
{
    float init[4] = {-1, -1, -1, -1}, x = -1;
    GLuint vao, tfb;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &tfb); glBindBuffer(GL_ARRAY_BUFFER, tfb); glBufferData(GL_ARRAY_BUFFER, 16, init, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tfb);
    glUseProgram(p); glEnable(GL_RASTERIZER_DISCARD); glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback(); glDisable(GL_RASTERIZER_DISCARD);
    float *t = glMapBufferRange(GL_ARRAY_BUFFER, 0, 16, GL_MAP_READ_BIT);
    if (t) x = t[0];
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glDeleteBuffers(1, &tfb); glDeleteVertexArrays(1, &vao);
    return x;
}

static int ssbo_case(int n)
{
    size_t o = 0;
    o += snprintf(gs + o, sizeof gs - o, "#version 450\nlayout(points) in;\nlayout(points, max_vertices = 1) out;\nint counter = 0;\n");
    for (int i = 0; i < n; i++) o += snprintf(gs + o, sizeof gs - o, "layout(binding = %d) buffer ssb%d { int value; } S%d;\n", i, i, i);
    o += snprintf(gs + o, sizeof gs - o, "void main(){\n");
    for (int i = 0; i < n; i++) o += snprintf(gs + o, sizeof gs - o, "  counter += S%d.value++;\n", i);
    o += snprintf(gs + o, sizeof gs - o, "  gl_Position = vec4(float(counter), 0.0, 0.0, 1.0);\n  EmitVertex();\n}\n");
    GLuint p = link_gs();
    if (!p) return 1;
    GLint align = 4; glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &align);
    int ia = align / 4, size = n * align;
    static GLint data[4096];
    memset(data, 0, sizeof data);
    for (int i = 0; i < n; i++) data[i * ia] = i + 1;
    GLuint buf; glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, GL_DYNAMIC_COPY);
    for (int i = 0; i < n; i++) glBindBufferRange(GL_SHADER_STORAGE_BUFFER, i, buf, i * align, 4);
    float counter = draw_point(p);
    int bad = (int)(counter + 0.5f) != n * (n + 1) / 2;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    GLint *m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_READ_BIT);
    for (int i = 0; i < n; i++) if (!m || m[i * ia] != i + 2) bad = 1;
    if (bad) printf("  [ssbo n=%d] counter=%g first=%d\n", n, counter, m ? m[0] : -99);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glDeleteBuffers(1, &buf); glDeleteProgram(p);
    return bad;
}

static int ubo_case(int binding)
{
    snprintf(gs, sizeof gs, "#version 450\nlayout(points) in;\nlayout(points, max_vertices = 1) out;\n"
             "layout(std140, binding = %d) uniform U { vec4 v; } u;\n"
             "void main(){ gl_Position = u.v; EmitVertex(); }\n", binding);
    GLuint p = link_gs();
    if (!p) return 1;
    float v[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    GLuint ubo; glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
    glBufferData(GL_UNIFORM_BUFFER, 16, v, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, binding, ubo);
    float x = draw_point(p);
    int bad = x != 1.5f;
    if (bad) printf("  [ubo binding=%d] got %g\n", binding, x);
    glDeleteBuffers(1, &ubo); glDeleteProgram(p);
    return bad;
}

int main(void)
{
    GLint n = 0, b = 0;
    glGetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, &n);
    glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS, &b);
    int bad = 0;
    int ns[] = {1, 8, 9, n};
    for (unsigned i = 0; i < sizeof ns / sizeof ns[0]; i++) bad += ssbo_case(ns[i]);
    int bs[] = {0, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 300, b - 1};
    for (unsigned i = 0; i < sizeof bs / sizeof bs[0]; i++) bad += ubo_case(bs[i]);
    EXPECT_EQ_INT(bad, 0);
    return mgl_test_summary();
}
