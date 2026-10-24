/*
 * test_compat_glsl.c — compatibility-profile GLSL through the whole stack:
 * #version 110/120/150-compat shaders, fixed-function builtins (matrices,
 * light/material/fog uniforms, texture matrix), gl_FragData, ftransform.
 */
#include "mgl_test.h"
#include "mgl_compat.h"
#include "gl_compat.h"

static void setup_2d(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glDisable(GL_FOG);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_TEXTURE_2D);
    glPixelZoom(1, 1);
}

static GLuint build(const char *vs_src, const char *fs_src, char *log, GLsizei logsize)
{
    GLuint p = glCreateProgram();
    GLuint v = glCreateShader(GL_VERTEX_SHADER);
    GLuint f = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(v, 1, &vs_src, NULL);
    glCompileShader(v);
    GLint ok = 0;
    glGetShaderiv(v, GL_COMPILE_STATUS, &ok);
    if (!ok && log) { glGetShaderInfoLog(v, logsize, NULL, log); return 0; }

    glShaderSource(f, 1, &fs_src, NULL);
    glCompileShader(f);
    ok = 0;
    glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
    if (!ok && log) { glGetShaderInfoLog(f, logsize, NULL, log); return 0; }

    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok && log) { glGetProgramInfoLog(p, logsize, NULL, log); return 0; }

    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}


/* deprecated texture builtins on 1D/2D/3D/cube samplers incl. Proj/Lod/bias
 * and the shadow family (translator: they map onto texture()/textureProj/
 * textureLod; the shadow helpers return vec4(vec3(r)) for GLSL 1.10) */
TEST(glsl_deprecated_texture_builtins)
{
    const char *vs =
        "#version 120\n"
        "uniform sampler1D t1;\n"
        "uniform sampler1DShadow s1;\n"
        "uniform sampler2D t2;\n"
        "uniform vec4 c4;\n"
        "void main() {\n"
        "  gl_Position = ftransform();\n"
        "  gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "  /* Lod + ProjLod variants are vertex-legal in GLSL 1.10/1.20 */\n"
        "  vec4 a = texture1DLod(t1, 0.5, 0.0);\n"
        "  vec4 b = texture2DProjLod(t2, c4, 0.0);\n"
        "  vec4 c = shadow1DLod(s1, vec3(0.5, 0.0, 0.5), 0.0);\n"
        "  gl_Position += a + b + c;\n"
        "}\n";
    const char *fs =
        "#version 110\n"
        "uniform sampler1D t1;\n"
        "uniform sampler2D t2;\n"
        "uniform sampler2DShadow s2;\n"
        "void main() {\n"
        "  vec4 a = texture1D(t1, gl_TexCoord[0].x);\n"
        "  vec4 b = texture1D(t1, gl_TexCoord[0].x, 0.5);\n"   /* bias overload */
        "  vec4 c = texture2DProj(t2, gl_TexCoord[0]);\n"
        "  vec4 d = shadow2D(s2, vec3(0.5, 0.5, 0.5));\n"
        "  vec4 e = shadow2DProj(s2, vec4(0.5, 0.5, 0.5, 1.0), 0.25);\n" /* bias */
        "  gl_FragColor = (a + b + c + d + e) * 0.2;\n"
        "}\n";
    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("     build log: %s\n", log); return; }

    /* gl_MaxVaryingFloats is translated to gl_MaxVaryingComponents */
    const char *vs2 =
        "#version 120\n"
        "void main() { gl_Position = ftransform();\n"
        "  if (gl_MaxVaryingFloats < 8) gl_Position.z += 0.1; }\n";
    GLuint p2 = build(vs2, "#version 120\nvoid main(){ gl_FragColor = vec4(1); }", log, sizeof log);
    EXPECT_TRUE(p2 != 0);
    if (!p2) printf("     maxvarying log: %s\n", log);

    glUseProgram(p);
    EXPECT_TRUE(glGetUniformLocation(p, "t1") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "t2") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "s2") >= 0);
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteProgram(p2);
}

/* a 1D texture actually sampled through texture1D(): red on the left half of
 * the quad, green on the right (1D textures are 2D textures of height 1 in
 * Metal via SPIRV-Cross's 1D-as-2D option) */
TEST(glsl_sampler1d_renders)
{
    unsigned char tex[2][4] = { { 255, 0, 0, 255 }, { 0, 255, 0, 255 } };
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_1D, t);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    const char *vs =
        "#version 120\n"
        "void main() { gl_Position = ftransform(); gl_TexCoord[0] = gl_MultiTexCoord0; }\n";
    const char *fs =
        "#version 110\n"
        "uniform sampler1D tex;\n"
        "void main() { gl_FragColor = texture1D(tex, gl_TexCoord[0].x); }\n";
    char log[512] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("     build log: %s\n", log); glDeleteTextures(1, &t); return; }

    setup_2d(32, 32);
    glUseProgram(p);
    GLint loc = glGetUniformLocation(p, "tex");
    EXPECT_TRUE(loc >= 0);
    glUniform1i(loc, 0);

    glBegin(GL_QUADS);
    glTexCoord1f(0.0f); glVertex2f(0, 0);
    glTexCoord1f(0.0f); glVertex2f(32, 0);
    glTexCoord1f(1.0f); glVertex2f(32, 32);
    glTexCoord1f(1.0f); glVertex2f(0, 32);
    glEnd();
    EXPECT_TRUE(glGetError() == GL_NO_ERROR);

    unsigned char a[4] = { 0 }, b[4] = { 0 };
    glReadPixels(2, 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);
    glReadPixels(29, 29, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);
    EXPECT_TRUE(a[0] > 200 && a[1] < 50);    /* texel 0: red */
    EXPECT_TRUE(b[1] > 200 && b[0] < 50);    /* texel 1: green */

    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
}


/* #version 150 with an `out gl_PerVertex` redeclaration — legal through
 * GL_ARB_separate_shader_objects (accepted by Mesa and Apple's 4.1 core;
 * glmark2's gl41-pipeline ships exactly this shader).  The compiler enables
 * the extension when it sees the redeclaration below GLSL 4.10. */
TEST(glsl_150_gl_per_vertex_redeclaration)
{
    const char *vs =
        "#version 150\n"
        "out gl_PerVertex { vec4 gl_Position; };\n"
        "in vec3 position;\n"
        "out vec3 vN;\n"
        "void main() { vN = position; gl_Position = vec4(position, 1.0); }\n";
    const char *fs =
        "#version 150\nout vec4 FragColor;\n"
        "in vec3 vN;\nvoid main() { FragColor = vec4(normalize(vN) * 0.5 + 0.5, 1.0); }\n";
    char log[512] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) printf("     build log: %s\n", log);
    glDeleteProgram(p);

    /* without the redeclaration the same shader must of course still work */
    const char *vs2 =
        "#version 150\nin vec3 position;\nvoid main() { gl_Position = vec4(position, 1.0); }\n";
    GLuint p2 = build(vs2, fs, log, sizeof log);
    EXPECT_TRUE(p2 != 0);
    glDeleteProgram(p2);
}

/* #version 110 with ftransform, gl_Color, gl_TexCoord and gl_FragData[0] */
TEST(glsl_110_ftransform_fragdata)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    const char *vs =
        "#version 110\n"
        "varying vec4 col;\n"
        "void main() {\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "    col = gl_Color;\n"
        "    gl_TexCoord[0] = gl_MultiTexCoord0;\n"
        "}\n";
    const char *fs =
        "#version 110\n"
        "varying vec4 col;\n"
        "void main() { gl_FragData[0] = col; }\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    glUseProgram(p);
    glColor4f(0, 1, 0, 1);
    glBegin(GL_QUADS);
    glMultiTexCoord2f(GL_TEXTURE0, 0, 0);
    glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
    glEnd();
    glUseProgram(0);
    glDeleteProgram(p);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 0, 255, 0, 255, 8);
    EXPECT_PIXEL(4, 4, 0, 0, 0, 255, 8);
}

/* #version 120: ftransform(), gl_NormalMatrix, gl_FogFragCoord, texture2D */
TEST(glsl_120_builtins)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint tex = 0;
    GLubyte img[4] = { 255, 0, 0, 255 };
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    const char *vs =
        "#version 120\n"
        "varying vec4 col;\n"
        "varying vec4 tc;\n"
        "void main() {\n"
        "    gl_Position = ftransform();\n"
        "    col = gl_Color;\n"
        "    tc = gl_TextureMatrix[0] * gl_MultiTexCoord0;\n"
        "    gl_FogFragCoord = 0.0;\n"
        "}\n";
    const char *fs =
        "#version 120\n"
        "varying vec4 col;\n"
        "varying vec4 tc;\n"
        "uniform sampler2D s;\n"
        "void main() {\n"
        "    gl_FragColor = texture2D(s, tc.xy) * col;\n"
        "}\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    glUseProgram(p);
    GLint loc = glGetUniformLocation(p, "s");
    EXPECT_TRUE(loc >= 0);
    glUniform1i(loc, 0);

    /* scale the texture matrix so u,v = 0.5 always (1x1 texture sample) */
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glScalef(0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);

    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glMultiTexCoord2f(GL_TEXTURE0, 0.3f, 0.9f);
    glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
    glEnd();
    glUseProgram(0);
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);

    EXPECT_NO_GL_ERROR();
    EXPECT_PIXEL(32, 32, 255, 0, 0, 255, 8);
}

/* #version 150 compatibility: gl_Vertex/gl_Color + gl_FragColor + uniforms
 * fed from fixed-function state (gl_LightSource, gl_FrontMaterial, gl_Fog) */
TEST(glsl_150_compat_state_uniforms)
{
    setup_2d(64, 64);
    glClear(GL_COLOR_BUFFER_BIT);

    glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat[4]){ 0, 1, 0, 1 });
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat[4]){ 0, 0, 1, 0 });
    glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat[4]){ 0, 0, 1, 1 });
    glFogfv(GL_FOG_COLOR, (GLfloat[4]){ 0, 0, 0, 1 });

    const char *vs =
        "#version 150 compatibility\n"
        "out vec4 col;\n"
        "void main() {\n"
        "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
        "    col = gl_Color * gl_LightSource[0].diffuse;\n"
        "}\n";
    const char *fs =
        "#version 150 compatibility\n"
        "in vec4 col;\n"
        "void main() {\n"
        "    gl_FragColor = col + gl_FrontMaterial.emission + gl_Fog.color;\n"
        "}\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    glUseProgram(p);
    glColor4f(1, 0, 0, 1);
    glBegin(GL_QUADS);
    glVertex2f(8, 8); glVertex2f(56, 8); glVertex2f(56, 56); glVertex2f(8, 56);
    glEnd();
    glUseProgram(0);
    glDeleteProgram(p);

    EXPECT_NO_GL_ERROR();
    /* red * light diffuse (0,1,0) = (0,0,0); + emission blue + fog black */
    EXPECT_PIXEL(32, 32, 0, 0, 255, 255, 8);

    glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat[4]){ 0, 0, 0, 1 });
}

/* builtin uniform locations are queryable (active builtins resolve) */
TEST(glsl_builtin_uniform_locations)
{
    const char *vs =
        "#version 120\n"
        "void main() {\n"
        "    gl_Position = gl_ModelViewMatrix * gl_ProjectionMatrix * gl_Vertex;\n"
        "    gl_Position += vec4(gl_NormalMatrix * vec3(0.0), 0.0);\n"
        "}\n";
    const char *fs =
        "#version 120\n"
        "void main() { gl_FragColor = vec4(1.0); }\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    glUseProgram(p);
    EXPECT_TRUE(glGetUniformLocation(p, "gl_ModelViewMatrix") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "gl_ProjectionMatrix") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "gl_NormalMatrix") >= 0);
    EXPECT_TRUE(glGetUniformLocation(p, "gl_TextureMatrix[0]") < 0);   /* unused -> -1 */
    glUseProgram(0);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

/* scalar default-block uniforms next to vec4s must round-trip and reach the
 * GPU (the fixed-function generator declares float/vec2 uniforms of its own);
 * checked in readback and in rendered pixels, via #version 110 (translated)
 * and glProgramUniform* on a program that is not current */
TEST(glsl_scalar_uniform_sandwich)
{
    setup_2d(8, 8);
    glClear(GL_COLOR_BUFFER_BIT);

    const char *vs = "#version 110\nvoid main(){ gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0); }\n";
    const char *fs = "#version 110\n"
        "uniform vec4 a0;\n"
        "uniform float s1;\n"
        "uniform vec4 a1;\n"
        "uniform float s2;\n"
        "void main(){ gl_FragColor = vec4(a0.x + s1 + a1.y + s2, 0.0, 0.0, 1.0); }\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    glUseProgram(p);
    glUniform4f(glGetUniformLocation(p, "a0"), 0.25f, 0, 0, 0);
    glUniform1f(glGetUniformLocation(p, "s1"), 0.25f);
    glUniform4f(glGetUniformLocation(p, "a1"), 0, 0.25f, 0, 0);
    glUniform1f(glGetUniformLocation(p, "s2"), 0.25f);

    GLfloat rb = -1.0f;
    glGetUniformfv(p, glGetUniformLocation(p, "s1"), &rb);
    EXPECT_NEAR(rb, 0.25f, 1e-6);

    glBegin(GL_QUADS);
    glVertex2f(-1, -1); glVertex2f(1, -1); glVertex2f(1, 1); glVertex2f(-1, 1);
    glEnd();
    EXPECT_PIXEL(4, 4, 255, 0, 0, 255, 2);
    glUseProgram(0);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

TEST(glsl_program_uniform_compat)
{
    setup_2d(8, 8);
    glClear(GL_COLOR_BUFFER_BIT);

    const char *vs = "#version 460 core\nlayout(location=0) in vec2 p; void main(){ gl_Position=vec4(p,0,1); }\n";
    const char *fs = "#version 460 core\nlayout(location=0) out vec4 o; uniform float s1; void main(){ o=vec4(s1,0,0,1); }\n";

    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("log: %s\n", log); return; }

    /* the program is NOT current: only glProgramUniform* may touch it */
    glProgramUniform1f(p, glGetUniformLocation(p, "s1"), 0.5f);
    GLfloat rb = -1.0f;
    glGetUniformfv(p, glGetUniformLocation(p, "s1"), &rb);
    EXPECT_NEAR(rb, 0.5f, 1e-6);

    glUseProgram(p);
    glBegin(GL_QUADS);
    glVertex2f(-1, -1); glVertex2f(1, -1); glVertex2f(1, 1); glVertex2f(-1, 1);
    glEnd();
    EXPECT_PIXEL(4, 4, 128, 0, 0, 255, 3);
    glUseProgram(0);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

/* glmark2 [terrain] crash regression: a sampler that is declared but never
 * sampled is stripped from the MSL by SPIRV-Cross (msl_binding = 0xffffffff
 * sentinel, program.c).  A texture bound to that unit must not be handed to
 * setFragmentTexture:atIndex:0xffffffff — that indexed the AGX bind table out
 * of range and killed the driver (SIGSEGV inside AGXMetalG16X setTexture). */
TEST(glsl_unused_sampler_no_crash)
{
    setup_2d(8, 8);
    glClear(GL_COLOR_BUFFER_BIT);

    const char *vs = "#version 120\n"
                     "void main() { gl_Position = ftransform(); gl_TexCoord[0] = gl_MultiTexCoord0; }\n";
    const char *fs = "#version 110\n"
                     "uniform sampler2D used_tex;   /* sampled */\n"
                     "uniform sampler2D unused_tex; /* declared, never read */\n"
                     "void main() { gl_FragColor = texture2D(used_tex, gl_TexCoord[0].xy); }\n";
    char log[1024] = { 0 };
    GLuint p = build(vs, fs, log, sizeof log);
    EXPECT_TRUE(p != 0);
    if (!p) { printf("     build log: %s\n", log); return; }
    glUseProgram(p);

    GLuint tex[2];
    glGenTextures(2, tex);
    GLubyte px[4] = { 40, 90, 160, 255 };
    for (int i = 0; i < 2; i++)          /* a real texture on BOTH units */
    {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, tex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    }
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(p, "used_tex"), 0);
    glUniform1i(glGetUniformLocation(p, "unused_tex"), 1);

    EXPECT_NO_GL_ERROR();

    setup_2d(8, 8);
    glBegin(GL_QUADS);
    glTexCoord2f(0.5f, 0.5f); glVertex2f(0, 0);
    glTexCoord2f(0.5f, 0.5f); glVertex2f(8, 0);
    glTexCoord2f(0.5f, 0.5f); glVertex2f(8, 8);
    glTexCoord2f(0.5f, 0.5f); glVertex2f(0, 8);
    glEnd();
    EXPECT_PIXEL(4, 4, 40, 90, 160, 255, 3);

    glDeleteTextures(2, tex);
    glUseProgram(0);
    glDeleteProgram(p);
    EXPECT_NO_GL_ERROR();
}

int main(void)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return 1;

    RUN(glsl_150_gl_per_vertex_redeclaration);
    RUN(glsl_deprecated_texture_builtins);
    RUN(glsl_sampler1d_renders);
    RUN(glsl_110_ftransform_fragdata);
    RUN(glsl_120_builtins);
    RUN(glsl_150_compat_state_uniforms);
    RUN(glsl_builtin_uniform_locations);
    RUN(glsl_scalar_uniform_sandwich);
    RUN(glsl_program_uniform_compat);
    RUN(glsl_unused_sampler_no_crash);

    return mgl_test_summary();
}
