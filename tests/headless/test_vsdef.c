/* probe: VS with a #define feeding an out-array size + GS (capture-variant rewrite path) */
#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

static const char *kVs =
    "#version 450 core\n"
    "\n"
    "\n"
    "// definition of TSZ\n"
    "#define TSZ 2u\n"
    "layout(location=0) in vec2 pos;\n"
    "out VS_OUT { vec2 v_pos[TSZ]; };\n"
    "void main(){ gl_Position = vec4(pos, 0.0, 1.0); }\n";
static const char *kGs =
    "#version 450 core\n"
    "layout(points) in;\n"
    "in VS_OUT { vec2 v_pos[2]; } vs_in[1];\n"
    "layout(triangle_strip, max_vertices=4) out;\n"
    "layout(location=0) out vec2 g_uv;\n"
    "void main(){\n"
    "  vec2 p = vs_in[0].v_pos[0], e = vec2(0.2, 0.2);\n"
    "  g_uv = vec2(0); gl_Position = vec4(p.x-e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x+e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x-e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(0); gl_Position = vec4(p.x+e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec2 g_uv;\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ if (min(g_uv.x, g_uv.y) < 0.25) discard; c0 = vec4(0, 1, 0, 1); }\n";

TEST(VsDefineArray)
{
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 128, 128);

    GLuint p = glCreateProgram();
    const char *s[3] = { kVs, kGs, kFs };
    GLenum t[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint sh = glCreateShader(t[i]);
        glShaderSource(sh, 1, &s[i], NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                   MGL_FAIL("compile %x: %s", t[i], log); return; }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("link: %s", log); return; }

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pt[2] = { 0, 0 };
    glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glUseProgram(p);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLubyte c[4];
    glReadPixels(64, 64, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);
    printf("vs-define-array pixel = %d,%d,%d\n", c[0], c[1], c[2]);
    EXPECT_TRUE(c[1] > 100);
}

int main(void)
{
    RUN(VsDefineArray);
    return mgl_test_summary();
}
