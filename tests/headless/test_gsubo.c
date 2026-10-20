/* GS with a uniform-block instance ARRAY: the kernel spans N consecutive slots */
#include <stdio.h>
#include "glcorearb.h"
#include "mgl_test.h"

#define NB 14
#define MGL_S_(x) #x
#define MGL_S(x) MGL_S_(x)

static const char *kVs =
    "#version 450 core\n"
    "layout(location=0) in vec2 pos;\n"
    "layout(location=0) out vec2 v_pos;\n"
    "void main(){ v_pos = pos; gl_Position = vec4(pos, 0.0, 1.0); }\n";

static const char *kGs =
    "#version 450 core\n"
    "layout(points) in;\n"
    "layout(location=0) in vec2 v_pos[];\n"
    "layout(triangle_strip, max_vertices=4) out;\n"
    "layout(location=0) out vec2 g_uv;\n"
    "layout(location=1) out float g_s;\n"
    "layout(std140) uniform UB { vec4 c; } ub[" MGL_S(NB) "];\n"
    "void main(){\n"
    "  float s = 0.0;\n"
    "  for (int i = 0; i < " MGL_S(NB) "; i++) s += ub[i].c.x;\n"
    "  g_s = s;\n"
    "  vec2 p = v_pos[0], e = vec2(0.3, 0.3);\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x-e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x+e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x-e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x+e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";

static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec2 g_uv;\n"
    "layout(location=1) in float g_s;\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ c0 = vec4(g_s / 105.0, 0, 0, 1); }\n";

TEST(GsUBOArray)
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

    GLuint ubos[NB];
    double want = 0.0;
    for (int i = 0; i < NB; i++)
    {
        float data[4] = { (float)(i + 1), 0, 0, 1.0f };   /* distinct: a missing block shows */
        want += i + 1;
        glGenBuffers(1, &ubos[i]);
        glBindBuffer(GL_UNIFORM_BUFFER, ubos[i]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, i, ubos[i]);
    }
    printf("expected sum = %.0f\n", want);

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
    double got = c[0] / 255.0 * 105.0;
    printf("gs-ubo-array sum = %.1f\n", got);
    EXPECT_TRUE(got > 104.0 && got < 106.0);
}

/* GS with a uniform-block instance ARRAY spanning N consecutive kernel slots; the sum
 * covers every element, so a shadowed slot (the old init-kernel slot-0 collision)
 * reads 0 and shows up. */
int main(void)
{
    RUN(GsUBOArray);
    return mgl_test_summary();
}
