/* geom r87: separate_shader.random.116 regression — the CTS vertex shader with an
 * explicit-location mat2x3 (locations 4+5) plus implicit-location varyings made
 * MGL emit two vertex outputs with the same [[user(locn5)]], which Metal rejects
 * ("duplicated user-defined name 'locn5' for vertex output declaration"). */
#include "mgl_test.h"

static const char *vs116 = "#version 310 es\n"
"uniform highp float vtxScale;\n"
"flat out highp float var0;\n"
"layout(location=1) smooth out highp vec2 vtxVar1;\n"
"centroid out highp vec3 var2;\n"
"centroid out highp vec2 var3;\n"
"layout(location=4) centroid out highp mat2x3 vtxVar4;\n"
"flat out highp uvec3 var5;\n"
"out highp vec4 var6;\n"
"layout(location=8) flat out highp uvec2 vtxVar7;\n"
"void main (void)\n"
"{\n"
"	gl_Position = vec4(vtxScale * vec2(0.5, 0.5), 0.0, 1.0);\n"
"	var0 = 1.0; vtxVar1 = vec2(1.0); var2 = vec3(1.0); var3 = vec2(1.0);\n"
"	vtxVar4 = mat2x3(1.0,2.0,3.0,4.0,5.0,6.0);\n"
"	var5 = uvec3(1u); var6 = vec4(1.0); vtxVar7 = uvec2(2u);\n"
"}\n";

static const char *fs116 = "#version 310 es\n"
"flat in highp float var0;\n"
"layout(location=1) smooth in highp vec2 vtxVar1;\n"
"centroid in highp vec3 var2;\n"
"centroid in highp vec2 var3;\n"
"layout(location=4) centroid in highp mat2x3 vtxVar4;\n"
"flat in highp uvec3 var5;\n"
"in highp vec4 var6;\n"
"layout(location=8) flat in highp uvec2 vtxVar7;\n"
"layout(location=0) out highp vec4 frag0;\n"
"void main (void)\n"
"{\n"
"	frag0 = vec4(var0 + vtxVar1.x + var2.x + var3.x + vtxVar4[0].x,\n"
"	             float(var5.x) + var6.x + float(vtxVar7.x), 0.0, 1.0);\n"
"}\n";

TEST(regress_sep_116_locations)
{
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint p;
    char log[4096];
    GLsizei len = 0;
    GLint ok = GL_FALSE;

    glShaderSource(vs, 1, &vs116, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
    EXPECT_NO_GL_ERROR();
    if (ok != GL_TRUE) {
        glGetShaderInfoLog(vs, sizeof log, &len, log);
        fprintf(stderr, "vs116 compile failed: %s\n", log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);

    glShaderSource(fs, 1, &fs116, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
    EXPECT_NO_GL_ERROR();
    if (ok != GL_TRUE) {
        glGetShaderInfoLog(fs, sizeof log, &len, log);
        fprintf(stderr, "fs116 compile failed: %s\n", log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);

    p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE) {
        glGetProgramInfoLog(p, sizeof log, &len, log);
        fprintf(stderr, "sep116 link FAILED: %.*s\n", (int)len, log);
    }
    EXPECT_EQ_INT(ok, GL_TRUE);
    glDeleteProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

int main(void)
{
    RUN(regress_sep_116_locations);
    return mgl_test_summary();
}
