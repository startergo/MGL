/* tess->GS pipeline isolation (round 19 scratch) */
#include "mgl_test.h"
#include <stdbool.h>
#include <string.h>

static GLuint mksh(GLenum t, const char *src)
{
    GLuint s = glCreateShader(t);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               fprintf(stderr, "compile %x log:\n%s\n", t, log); }
    return s;
}

static const char *VS =
    "#version 400\n"
    "out VS_DATA { vec4 vs_data; } result;\n"
    "void main() { gl_Position = vec4(-0.5,-0.5,0,1); result.vs_data = vec4(1,2,3,4); }\n";
static const char *TC =
    "#version 400\n"
    "layout (vertices = 1) out;\n"
    "in VS_DATA { vec4 vs_data; } vs[];\n"
    "out TC_DATA { vec4 tc_data; vec4 vs_data; } result[];\n"
    "void main() {\n"
    " gl_TessLevelInner[0] = 1.0; gl_TessLevelInner[1] = 1.0;\n"
    " gl_TessLevelOuter[0] = 1.0; gl_TessLevelOuter[1] = 1.0;\n"
    " gl_TessLevelOuter[2] = 1.0; gl_TessLevelOuter[3] = 1.0;\n"
    " result[gl_InvocationID].tc_data = vec4(2,3,4,5);\n"
    " result[gl_InvocationID].vs_data = vs[gl_InvocationID].vs_data; }\n";
static const char *TE_QUADS =
    "#version 400\n"
    "layout (quads) in;\n"
    "in TC_DATA { vec4 tc_data; vec4 vs_data; } tc[];\n"
    "out TE_DATA { vec4 tc_data; vec4 te_data; vec4 vs_data; } result;\n"
    "void main() { gl_Position = vec4(gl_TessCoord.xy, 0, 1); result.vs_data = tc[0].vs_data;"
    " result.tc_data = tc[0].tc_data; result.te_data = vec4(3,4,5,6); }\n";
static const char *GS =
    "#version 400\n"
    "layout(triangles) in;\nlayout(triangle_strip, max_vertices = 4) out;\n"
    "in TE_DATA { vec4 tc_data; vec4 te_data; vec4 vs_data; } te[];\n"
    "out GS_DATA { vec4 gs_data; vec4 tc_data; vec4 te_data; vec4 vs_data; } result;\n"
    "void main() {\n"
    " result.gs_data = vec4(4,5,6,7);\n"
    " result.tc_data = te[0].tc_data; result.te_data = te[0].te_data; result.vs_data = te[0].vs_data;\n"
    " gl_Position = vec4(1,-1,0,1); EmitVertex();\n"
    " gl_Position = vec4(-1,-1,0,1); EmitVertex();\n"
    " gl_Position = vec4(1,1,0,1); EmitVertex();\n"
    " gl_Position = vec4(-1,1,0,1); EmitVertex(); EndPrimitive(); }\n";
static const char *FS =
    "#version 400\n"
    "in GS_DATA { vec4 gs_data; vec4 tc_data; vec4 te_data; vec4 vs_data; } gs;\n"
    "out vec4 result;\n"
    "void main() { result = gs.gs_data + gs.tc_data + gs.te_data + gs.vs_data; }\n";

TEST(tess_gs_quads)
{
    GLuint sh[5] = { mksh(GL_VERTEX_SHADER, VS), mksh(GL_TESS_CONTROL_SHADER, TC),
                     mksh(GL_TESS_EVALUATION_SHADER, TE_QUADS), mksh(GL_GEOMETRY_SHADER, GS),
                     mksh(GL_FRAGMENT_SHADER, FS) };
    GLuint p = glCreateProgram();
    for (int i = 0; i < 5; i++) glAttachShader(p, sh[i]);
    glLinkProgram(p);
    GLint lk = 0; glGetProgramiv(p, GL_LINK_STATUS, &lk);
    EXPECT_EQ_INT(lk, GL_TRUE);
    if (lk == GL_TRUE) {
        GLuint fb = 0, tex = 0, vao = 0;
        glGenFramebuffers(1, &fb); glGenTextures(1, &tex); glGenVertexArrays(1, &vao);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindFramebuffer(GL_FRAMEBUFFER, fb);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glBindVertexArray(vao);
        glUseProgram(p);
        glPatchParameteri(GL_PATCH_VERTICES, 1);
        float clear[4] = { -1, -1, -1, -1 };
        glClearBufferfv(GL_COLOR, 0, clear);
        glDrawArrays(GL_PATCHES, 0, 1);
        float px[4] = { 0 };
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT, px);
        fprintf(stderr, "quads->GS px = %g %g %g %g (want 10 14 18 22)\n", px[0], px[1], px[2], px[3]);
        EXPECT_EQ_INT((int)px[0], 10);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fb); glDeleteTextures(1, &tex); glDeleteVertexArrays(1, &vao);
    }
    glDeleteProgram(p);
    for (int i = 0; i < 5; i++) glDeleteShader(sh[i]);
}

int main(void)
{
    RUN(tess_gs_quads);
    return mgl_test_summary();
}
