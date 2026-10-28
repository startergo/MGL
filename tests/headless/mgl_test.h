/*
 * mgl_test.h — minimal headless test harness for MGL.
 *
 * libmgl.dylib creates a headless Metal-backed context in a load-time
 * constructor, so a test is just a C program that calls GL entry points.
 * Each test file is one executable; tests/headless/run.sh builds and runs all.
 *
 *   #include "mgl_test.h"
 *   TEST(clear_red) {
 *       glClearColor(1,0,0,1); glClear(GL_COLOR_BUFFER_BIT); glFinish();
 *       GLubyte px[4]; glReadPixels(0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE,px);
 *       EXPECT_EQ_INT(px[0], 255);
 *       EXPECT_NO_GL_ERROR();
 *   }
 *   int main(void) { RUN(clear_red); return mgl_test_summary(); }
 */
#ifndef MGL_TEST_H
#define MGL_TEST_H

#define GL_GLEXT_PROTOTYPES 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "glcorearb.h"

static int mgl_test_failures = 0;
static int mgl_test_checks = 0;
static const char *mgl_test_current = "";

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
        int _before = mgl_test_failures; mgl_test_current = #name; \
        while (glGetError() != GL_NO_ERROR) {} \
        test_##name(); \
        printf("%s %s\n", mgl_test_failures == _before ? "PASS" : "FAIL", #name); \
        fflush(stdout); \
    } while (0)

#define MGL_FAIL(fmt, ...) do { mgl_test_failures++; \
        fprintf(stderr, "  [%s] %s:%d: " fmt "\n", mgl_test_current, __FILE__, __LINE__, ##__VA_ARGS__); } while (0)

#define EXPECT_TRUE(c) do { mgl_test_checks++; if (!(c)) MGL_FAIL("expected true: %s", #c); } while (0)
#define EXPECT_EQ_INT(a, b) do { mgl_test_checks++; long long _a = (long long)(a), _b = (long long)(b); \
        if (_a != _b) MGL_FAIL("%s == %lld, expected %lld", #a, _a, _b); } while (0)
#define EXPECT_NEAR(a, b, eps) do { mgl_test_checks++; double _a = (a), _b = (b); \
        if (fabs(_a - _b) > (eps)) MGL_FAIL("%s == %f, expected %f", #a, _a, _b); } while (0)
#define EXPECT_STREQ(a, b) do { mgl_test_checks++; const char *_a = (const char *)(a), *_b = (b); \
        if (!_a || strcmp(_a, _b)) MGL_FAIL("%s == \"%s\", expected \"%s\"", #a, _a ? _a : "(null)", _b); } while (0)
#define EXPECT_GL_ERROR(e) do { mgl_test_checks++; GLenum _e = glGetError(); \
        if (_e != (e)) MGL_FAIL("glGetError() == 0x%04x, expected %s (0x%04x)", _e, #e, (e)); \
        while (glGetError() != GL_NO_ERROR) {} } while (0)
#define EXPECT_NO_GL_ERROR() EXPECT_GL_ERROR(GL_NO_ERROR)

/* Read one RGBA8 pixel from the current read framebuffer. */
static inline void mgl_read_rgba(int x, int y, GLubyte out[4])
{
    glFinish();
    glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
}

#define EXPECT_PIXEL(x, y, r, g, b, a, tol) do { GLubyte _p[4]; mgl_read_rgba(x, y, _p); \
        mgl_test_checks++; \
        if (abs(_p[0]-(r)) > (tol) || abs(_p[1]-(g)) > (tol) || abs(_p[2]-(b)) > (tol) || abs(_p[3]-(a)) > (tol)) \
            MGL_FAIL("pixel(%d,%d) = %d,%d,%d,%d expected %d,%d,%d,%d", x, y, _p[0], _p[1], _p[2], _p[3], r, g, b, a); \
    } while (0)

/* Compile + link a 5-stage or compute program; returns 0 and reports on failure. */
static inline GLuint mgl_make_program5(const char *vs_src, const char *tcs_src, const char *tes_src,
                                       const char *gs_src, const char *fs_src)
{
    GLuint p = glCreateProgram();
    const char *srcs[5] = { vs_src, tcs_src, tes_src, gs_src, fs_src };
    GLenum types[5] = { GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER,
                        GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 5; i++) {
        if (!srcs[i]) continue;
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], NULL);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
                   MGL_FAIL("shader compile failed: %s", log); return 0; }
        glAttachShader(p, s);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("program link failed: %s", log); return 0; }
    return p;
}

static inline GLuint mgl_make_program_cs(const char *cs_src)
{
    GLuint p = glCreateProgram();
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &cs_src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
               MGL_FAIL("shader compile failed: %s", log); return 0; }
    glAttachShader(p, s);
    glLinkProgram(p);
    ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("program link failed: %s", log); return 0; }
    glDeleteShader(s);
    return p;
}

/* Compile + link a VS/FS program; returns 0 and reports on failure. */
static inline GLuint mgl_make_program(const char *vs_src, const char *fs_src)
{
    GLuint p = glCreateProgram();
    const char *srcs[2] = { vs_src, fs_src };
    GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 2; i++) {
        if (!srcs[i]) continue;
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], NULL);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
                   MGL_FAIL("shader compile failed: %s", log); return 0; }
        glAttachShader(p, s);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("program link failed: %s", log); return 0; }
    return p;
}

static inline GLuint mgl_make_program3(const char *vs_src, const char *gs_src, const char *fs_src)
{
    GLuint p = glCreateProgram();
    const char *srcs[3] = { vs_src, gs_src, fs_src };
    GLenum types[3] = { GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 3; i++) {
        if (!srcs[i]) continue;
        GLuint s = glCreateShader(types[i]);
        glShaderSource(s, 1, &srcs[i], NULL);
        glCompileShader(s);
        GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char log[2048] = ""; glGetShaderInfoLog(s, sizeof log, NULL, log);
                   MGL_FAIL("shader compile failed: %s", log); return 0; }
        glAttachShader(p, s);
    }
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048] = ""; glGetProgramInfoLog(p, sizeof log, NULL, log);
               MGL_FAIL("program link failed: %s", log); return 0; }
    return p;
}

static inline int mgl_test_summary(void)
{
    printf("== %d checks, %d failures\n", mgl_test_checks, mgl_test_failures);
    return mgl_test_failures ? 1 : 0;
}

#endif
