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

int main(void)
{
    void *ctx = mglCompatCreateContext();
    EXPECT_TRUE(ctx != NULL);
    if (!ctx)
        return 1;

    RUN(glsl_110_ftransform_fragdata);
    RUN(glsl_120_builtins);
    RUN(glsl_150_compat_state_uniforms);
    RUN(glsl_builtin_uniform_locations);
    RUN(glsl_scalar_uniform_sandwich);
    RUN(glsl_program_uniform_compat);

    return mgl_test_summary();
}
