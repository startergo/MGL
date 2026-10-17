#include "mgl_test.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/* a program binary of a geometry program carries the per-stage SPIR-V, so reloading it
 * links from executables with no GLSL source: the emulation then has to decompile the
 * geometry stage back to GLSL.  The link, the program queries and a draw must all work. */

static const char *vs_src =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec2 v_pos;\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); v_pos = pos; }\n";

static const char *fs_src =
    "#version 450 core\n"
    "layout(location=0) in vec2 g_pos;\n"
    "layout(location=0) out vec4 color;\n"
    "void main(){ color = vec4(0.0, 1.0, 0.0, 1.0); }\n";

static const char *gs_src =
    "#version 450 core\n"
    "layout(triangles) in;\n"
    "layout(triangle_strip, max_vertices = 3) out;\n"
    "layout(location=0) in vec2 v_pos[];\n"
    "layout(location=0) out vec2 g_pos;\n"
    "void main(){\n"
    "  for (int i = 0; i < 3; i++) { gl_Position = gl_in[i].gl_Position; g_pos = v_pos[i]; EmitVertex(); }\n"
    "  EndPrimitive();\n"
    "}\n";

static GLuint compile(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = "";
        GLsizei len = 0;
        glGetShaderInfoLog(s, sizeof log, &len, log);
        printf("compile: %s\n", log);
    }
    return s;
}

static void px(int x, int y, GLubyte *out)
{
    glFinish();
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}

static bool is_green(GLubyte *p) { return p[0] < 32 && p[1] > 200 && p[2] < 32; }

int main(void)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, compile(GL_VERTEX_SHADER, vs_src));
    glAttachShader(p, compile(GL_GEOMETRY_SHADER, gs_src));
    glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs_src));
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok) return 1;

    /* ask for a retrievable binary, then re-link and read it back */
    glProgramParameteri(p, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    glLinkProgram(p);

    GLint len = 0;
    glGetProgramiv(p, GL_PROGRAM_BINARY_LENGTH, &len);
    EXPECT_TRUE(len > 0);
    if (len <= 0) return 1;

    void *blob = calloc(1, (size_t)len);
    GLenum fmt = 0;
    GLsizei got = 0;
    glGetProgramBinary(p, len, &got, &fmt, blob);
    EXPECT_NO_GL_ERROR();
    EXPECT_TRUE(got > 0);
    if (got <= 0) return 1;
    glDeleteProgram(p);

    /* reload: no shaders are attached, so the geometry stage comes from SPIR-V */
    p = glCreateProgram();
    glProgramBinary(p, fmt, blob, got);
    free(blob);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    EXPECT_EQ_INT(ok, GL_TRUE);
    if (!ok)
    {
        char log[4096] = "";
        GLsizei n = 0;
        glGetProgramInfoLog(p, sizeof log, &n, log);
        printf("binary link log:\n%s\n", log);
        return 1;
    }

    {
        GLint v = 0;
        glGetProgramiv(p, GL_GEOMETRY_VERTICES_OUT, &v); EXPECT_EQ_INT(v, 3);
        glGetProgramiv(p, GL_GEOMETRY_INPUT_TYPE, &v);   EXPECT_EQ_INT(v, GL_TRIANGLES);
        glGetProgramiv(p, GL_GEOMETRY_OUTPUT_TYPE, &v);  EXPECT_EQ_INT(v, GL_TRIANGLE_STRIP);
        glGetProgramiv(p, GL_GEOMETRY_SHADER_INVOCATIONS, &v); EXPECT_EQ_INT(v, 1);
    }

    glUseProgram(p);
    GLuint vao = 0, vbo = 0;
    static const float tri[6] = { -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 1.0f };
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof tri, tri, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    EXPECT_NO_GL_ERROR();

    {
        GLubyte c[4] = {0, 0, 0, 0};
        px(100, 150, c);
        printf("binary-gs pixel = %d,%d,%d\n", c[0], c[1], c[2]);
        EXPECT_TRUE(is_green(c));
    }

    glDeleteProgram(p);
    return mgl_test_summary();
}
