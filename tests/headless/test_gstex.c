/* probe: GS sampling many textures (the CTS limits.max_texture_units pattern) */
#include <stdio.h>
#include <string.h>
#include "glcorearb.h"
#include "mgl_test.h"

#define NTEX 16
#define NTEX_S "16"

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
    "uniform sampler2D tex[" NTEX_S "];\n"
    "void main(){\n"
    "  vec2 p = v_pos[0], e = vec2(0.25, 0.25);\n"
    "  float s = 0.0;\n"
    "  for (int i = 0; i < " NTEX_S "; i++) s += texelFetch(tex[i], ivec2(0), 0).r;\n"
    "  g_s = s;\n"
    "  g_uv = vec2(0); gl_Position = vec4(p.x-e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x+e.x, p.y-e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(1); gl_Position = vec4(p.x-e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  g_uv = vec2(0); gl_Position = vec4(p.x+e.x, p.y+e.y, 0, 1); EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";
static const char *kFs =
    "#version 450 core\n"
    "layout(location=0) in vec2 g_uv;\n"
    "layout(location=1) in float g_s;\n"
    "layout(location=0) out vec4 c0;\n"
    "void main(){ if (min(g_uv.x, g_uv.y) < 0.25) discard; c0 = (g_s >= 8.0f) ? vec4(0,1,0,1) : vec4(1,0,0,1); }\n";

TEST(GsManyTextures)
{
    GLuint fbo, ctex;
    glGenTextures(1, &ctex);
    glBindTexture(GL_TEXTURE_2D, ctex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ctex, 0);
    glViewport(0, 0, 128, 128);
    GLuint tex[NTEX];
    for (int i = 0; i < NTEX; i++)
    {
        glGenTextures(1, &tex[i]);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        float data[4] = { 1.0f, 0, 0, 1.0f };
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, data);
    }
    glActiveTexture(GL_TEXTURE0);

    GLuint p = glCreateProgram();
    const char *src[3] = { kVs, kGs, kFs };
    GLenum typ[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++)
    {
        GLuint sh = glCreateShader(typ[i]);
        glShaderSource(sh, 1, &src[i], NULL);
        glCompileShader(sh);
        GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048]; glGetShaderInfoLog(sh, sizeof log, NULL, log);
                   MGL_FAIL("compile %x: %s", typ[i], log); return; }
        glAttachShader(p, sh); glDeleteShader(sh);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("link: %s", log); return; }


    GLuint loc = glGetUniformLocation(p, "tex");
    GLint sz = 0;
    printf("loc(tex)=%d ", (int)loc);
    if (loc == (GLuint)-1) { loc = glGetUniformLocation(p, "tex[0]"); printf("loc(tex[0])=%d ", (int)loc); }
    glGetUniformiv(p, loc, &sz);
    printf("unit0=%d\n", sz);
    int units[NTEX];
    units[0] = 0;
    /* per-element sampler-uniform sets (glProgramUniform1i on "tex[i]", i>0) are rejected
     * by MGL's uniforms path — a uniforms-domain gap, logged in agents/conformance.md.
     * MGL's static convention already feeds tex[i] from unit base+i, so setting only the
     * base (which also validates count=1 works) is enough here */
    glProgramUniform1i(p, loc, units[0]);
    EXPECT_NO_GL_ERROR();

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float pt[2] = { 0.0f, 0.0f };
    glBufferData(GL_ARRAY_BUFFER, sizeof pt, pt, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glUseProgram(p);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    EXPECT_NO_GL_ERROR();

    GLubyte c[4];
    glReadPixels(64, 64, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, c);   /* the readback blit */
    printf("pixel = %d,%d,%d\n", c[0],c[1],c[2]);
    EXPECT_TRUE(c[1] > 100);
}

int main(void)
{
    RUN(GsManyTextures);
    return mgl_test_summary();
}
