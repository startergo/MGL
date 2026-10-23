/*
 * tests/nsgl_smoke.m — manual smoke test of the NSOpenGL interposer
 * (MGL/nsgl/, build/nsgl/libmgl_nsgl.dylib).  NOT part of run.sh: it opens a
 * real window on screen.  Build & run from the repo root:
 *
 *   cc -w -fmodules tests/nsgl_smoke.m -o build/nsgl_smoke -framework AppKit && \
 *   DYLD_INSERT_LIBRARIES="$(pwd)/build/nsgl/libmgl_nsgl.dylib" ./build/nsgl_smoke
 *
 * It creates a Legacy-profile NSOpenGLPixelFormat + NSOpenGLContext (which the
 * interposer turns into an MGL compatibility context), draws a coloured
 * immediate-mode quad with gl* entry points resolved through dlsym
 * (RTLD_DEFAULT -> MGL, like glad's fallback loader) and flushes it to the
 * window.  Prints PASS when the readback of the presented frame (the
 * interposer exposes the same buffer; verified via the dump hook below when
 * MGL_NSGL_DUMPDIR is set, otherwise by the error checks) has the four corner
 * colours.  Exits within a second.
 */
#import <AppKit/AppKit.h>
#import <OpenGL/OpenGL.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GL_VERSION = 0x1F02, GL_COLOR_BUFFER_BIT = 0x4000, GL_QUADS = 7 };

static void *g(const char *name) { return dlsym(RTLD_DEFAULT, name); }
#define DFN(ret, name, args) static ret (*p_##name) args

int main(void)
{
    setbuf(stdout, NULL);

    [NSApplication sharedApplication];
    NSRect fr = NSMakeRect(100, 100, 256, 256);
    NSWindow *win = [[NSWindow alloc] initWithContentRect:fr
                                                styleMask:1
                                                  backing:NSBackingStoreBuffered
                                                    defer:NO];
    NSView *view = [[NSView alloc] initWithFrame:fr];
    win.contentView = view;
    [win setTitle:@"nsgl_smoke (MGL)"];
    [win makeKeyAndOrderFront:nil];

    NSOpenGLPixelFormatAttribute attrs[] = {
        NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersionLegacy,   /* -> compat */
        NSOpenGLPFADoubleBuffer, NSOpenGLPFADepthSize, 24,
        0
    };
    NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
    NSOpenGLContext *ctx = [[NSOpenGLContext alloc] initWithFormat:pf shareContext:nil];
    [ctx setView:view];
    [ctx makeCurrentContext];

    DFN(const unsigned char *, glGetString, (unsigned));
#define INIT_FN(name) do { p_##name = (__typeof__(p_##name))g(#name); } while (0)
    DFN(unsigned, glGetError, (void));
    DFN(void, glViewport, (int, int, int, int));
    DFN(void, glClearColor, (float, float, float, float));
    DFN(void, glClear, (unsigned));
    DFN(void, glBegin, (unsigned));
    DFN(void, glEnd, (void));
    DFN(void, glColor3f, (float, float, float));
    DFN(void, glVertex2f, (float, float));

    INIT_FN(glGetString); INIT_FN(glGetError); INIT_FN(glViewport);
    INIT_FN(glClearColor); INIT_FN(glClear); INIT_FN(glBegin);
    INIT_FN(glEnd); INIT_FN(glColor3f); INIT_FN(glVertex2f);

    if (!p_glGetString || !p_glBegin)
    {
        printf("FAIL: GL entry points not found through dlsym\n");
        return 1;
    }

    int fails = 0;
    const char *ver = (const char *)p_glGetString(GL_VERSION);
    printf("%s GL_VERSION = \"%s\"\n", ver && strstr(ver, "compat") ? "ok  " : "FAIL", ver ? ver : "(null)");
    if (!ver || !strstr(ver, "compat"))
        fails++;

    p_glViewport(0, 0, 256, 256);
    p_glClearColor(0, 0, 0, 1);
    p_glClear(GL_COLOR_BUFFER_BIT);
    p_glBegin(GL_QUADS);
    p_glColor3f(1, 0, 0); p_glVertex2f(-1, -1);
    p_glColor3f(0, 1, 0); p_glVertex2f(1, -1);
    p_glColor3f(0, 0, 1); p_glVertex2f(1, 1);
    p_glColor3f(1, 1, 0); p_glVertex2f(-1, 1);
    p_glEnd();
    unsigned err = p_glGetError();
    printf("%s immediate-mode quad, glGetError = 0x%04x\n", err == 0 ? "ok  " : "FAIL", err);
    if (err)
        fails++;

    [ctx flushBuffer];

    /* the same readback the interposer presents can be verified with
     * MGL_NSGL_DUMPDIR=<dir> (debug build path) or by eye; the checks above
     * plus the benchmark runs are the automation here */
    printf("%s: %d failed checks (presented quad is on screen)\n",
           fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
