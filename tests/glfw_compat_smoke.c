/*
 * tests/glfw_compat_smoke.c — manual smoke test (NOT part of run.sh; it opens a
 * real window on the screen): GLFW compatibility-profile contexts through the
 * MGL glfw platform (external/glfw/src/mgl_context.m).
 *
 * Build & run from the repo root:
 *   cc -Iexternal/glfw/include tests/glfw_compat_smoke.c -o build/glfw_compat_smoke \
 *      -Lbuild -lglfw -Wl,-rpath,"$(pwd)/build" && ./build/glfw_compat_smoke
 *
 * Cases (each opens a window for ~0.3 s):
 *   1. no hints at all (GLFW defaults: version 1.0, any profile) -> compat context
 *   2. explicit GLFW_OPENGL_COMPAT_PROFILE at 3.2                 -> compat context
 *      (GLFW rejects profiles below 3.2 by design; 3.0+compat is invalid)
 *   3. explicit GLFW_OPENGL_COMPAT_PROFILE at 4.6                 -> compat context
 *   4. explicit GLFW_OPENGL_CORE_PROFILE at 4.6                   -> core context
 *      (must report the core bit and reject glBegin with GL_INVALID_OPERATION)
 * A compat case passes when glGetString(GL_VERSION) is 4.6, the profile mask
 * has GL_CONTEXT_COMPATIBILITY_PROFILE_BIT, GL_ARB_compatibility is advertised
 * and an immediate-mode quad actually rasterizes (center pixel != clear color).
 * Exit code 0 = all cases green.
 */
#define GL_GLEXT_PROTOTYPES 1
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GL_CONTEXT_CORE_PROFILE_BIT
#define GL_CONTEXT_CORE_PROFILE_BIT           0x00000001
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT  0x00000002
#endif
#ifndef GL_CONTEXT_PROFILE_MASK
#define GL_CONTEXT_PROFILE_MASK               0x9126
#endif

static int fails = 0;
#define CHECK(cond, ...) do { int _c = (cond); \
    printf("%s %s:%d: ", _c ? "ok  " : "FAIL", __func__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); if (!_c) fails++; } while (0)

/* draw an immediate-mode quad; returns 1 when the viewport center is covered
 * (i.e. legacy glBegin/glEnd rasterized something), 0 when it was a no-op */
static int draw_quad_reads_back(GLFWwindow *w)
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(w, &width, &height);

    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    while (glGetError() != GL_NO_ERROR) {}

    glBegin(GL_QUADS);
    glColor3f(1, 0, 0); glVertex2f(-0.5f, -0.5f);
    glColor3f(0, 1, 0); glVertex2f( 0.5f, -0.5f);
    glColor3f(0, 0, 1); glVertex2f( 0.5f,  0.5f);
    glColor3f(1, 1, 0); glVertex2f(-0.5f,  0.5f);
    glEnd();
    GLenum imm_err = glGetError();

    glFinish();
    GLubyte px[3] = { 0, 0, 0 };
    glReadPixels(width / 2, height / 2, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, px);
    glfwSwapBuffers(w);

    /* quad center = average of the four corner colors (0.5, 0.5, 0.25);
     * clear color = (0.1, 0.2, 0.3).  Green channel separates them cleanly. */
    int covered = px[1] > 100 && px[0] < 200;
    printf("     center pixel %d,%d,%d (glBegin error 0x%04x)\n", px[0], px[1], px[2], imm_err);
    return covered ? 1 : (imm_err ? 2 : 0);   /* 1 drew, 2 rejected, 0 no-op */
}

static void run_case(const char *name, int hint_version, int hint_profile)
{
    printf("== %s\n", name);
    if (hint_version) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, hint_version / 10);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, hint_version % 10);
    }
    if (hint_profile)
        glfwWindowHint(GLFW_OPENGL_PROFILE, hint_profile);
    else
        glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_ANY_PROFILE);

    GLFWwindow *w = glfwCreateWindow(320, 240, name, NULL, NULL);
    CHECK(w != NULL, "window created");
    if (!w) return;
    glfwMakeContextCurrent(w);

    const char *ver = (const char *)glGetString(GL_VERSION);
    CHECK(ver && strncmp(ver, "4.6", 3) == 0, "GL_VERSION = \"%s\"", ver ? ver : "(null)");

    GLint mask = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    int compat = (mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) != 0;
    int core   = (mask & GL_CONTEXT_CORE_PROFILE_BIT) != 0;

    const char *exts = (const char *)glGetString(GL_EXTENSIONS);
    int arb_compat = exts && strstr(exts, "GL_ARB_compatibility") != NULL;

    int res = draw_quad_reads_back(w);
    if (hint_profile == GLFW_OPENGL_CORE_PROFILE) {
        CHECK(core && !compat, "profile mask = 0x%x (core expected)", mask);
        CHECK(!arb_compat, "GL_ARB_compatibility not advertised (has=%d)", arb_compat);
        CHECK(res == 2, "glBegin rejected in core (res=%d)", res);
    } else {
        CHECK(compat && !core, "profile mask = 0x%x (compat expected)", mask);
        CHECK(arb_compat, "GL_ARB_compatibility advertised");
        CHECK(res == 1, "immediate-mode quad rasterized (res=%d)", res);
    }

    glfwDestroyWindow(w);
}

int main(void)
{
    setbuf(stdout, NULL);
    if (!glfwInit()) {
        printf("FAIL glfwInit\n");
        return 1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);

    run_case("default hints (1.0, any profile) -> compat", 0, 0);
    run_case("GLFW_OPENGL_COMPAT_PROFILE @ 3.2 (min with profiles)", 32, GLFW_OPENGL_COMPAT_PROFILE);
    run_case("GLFW_OPENGL_COMPAT_PROFILE @ 4.6", 46, GLFW_OPENGL_COMPAT_PROFILE);
    run_case("GLFW_OPENGL_CORE_PROFILE @ 4.6 (must stay core)", 46, GLFW_OPENGL_CORE_PROFILE);

    glfwTerminate();
    printf("%s: %d failed checks\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
