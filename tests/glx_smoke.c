/*
 * tests/glx_smoke.c — manual smoke test of the GLX shim (MGL/glx/, build/glx/
 * libGL.1.dylib).  NOT part of run.sh: it needs a running X server
 * (XQuartz).  Build & run from the repo root:
 *
 *   cc -I/opt/local/include tests/glx_smoke.c -o build/glx_smoke \
 *      build/glx/libGL.1.dylib -Lbuild -lmgl -L/opt/local/lib -lX11 -lXext \
 *      -Wl,-rpath,"$(pwd)/build" -Wl,-rpath,"$(pwd)/build/glx" && \
 *      # (-lmgl: apps that link the shim directly at build time also bind the
 *      #  re-exported gl* symbols to libmgl itself)
 *   DISPLAY=:0 ./build/glx_smoke
 *
 * Checks: glXQueryExtension/Version, glXChooseVisual, a compatibility context
 * via glXCreateContext (glBegin works, profile mask compat bit), a core
 * context via glXCreateContextAttribsARB (glBegin rejected), immediate-mode
 * rendering pushed to the X window by glXSwapBuffers and read back through
 * XGetImage (verifies the BGRA readback + y-flip + XPutImage path), and
 * glXUseXFont.  Exits 0 when everything passes.
 */
#define GL_GLEXT_PROTOTYPES 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>

/* the local glx.h has the pre-GLX-1.2 four-argument glXUseXFont; the shim
 * exports the standard five-argument one, so call it through its address */
static void (*p_glXUseXFont)(Display *, XID, int, int, int);
extern GLXContext glXCreateContextAttribsARB(Display *dpy, GLXFBConfig config,
    GLXContext share_context, Bool direct, const int *attrib_list);
#ifndef GLX_CONTEXT_MAJOR_VERSION_ARB
#define GLX_CONTEXT_MAJOR_VERSION_ARB          0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB          0x2092
#define GLX_CONTEXT_PROFILE_MASK_ARB           0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB       0x00000001
#define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
#endif

static int fails = 0;
#define CHECK(cond, ...) do { int _c = (cond); \
    printf("%s %s:%d: ", _c ? "ok  " : "FAIL", __func__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); if (!_c) fails++; } while (0)

#ifndef GL_CONTEXT_COMPATIBILITY_PROFILE_BIT
#define GL_CONTEXT_CORE_PROFILE_BIT          0x00000001
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT 0x00000002
#define GL_CONTEXT_PROFILE_MASK              0x9126
#endif

int main(void)
{
    setbuf(stdout, NULL);

    Display *dpy = XOpenDisplay(NULL);
    CHECK(dpy != NULL, "XOpenDisplay(%s)", getenv("DISPLAY") ? getenv("DISPLAY") : "(null)");
    if (!dpy)
        return 1;

    int err_base = 0, evt_base = 0;
    CHECK(glXQueryExtension(dpy, &err_base, &evt_base), "glXQueryExtension (err=%d evt=%d)", err_base, evt_base);
    int major = 0, minor = 0;
    CHECK(glXQueryVersion(dpy, &major, &minor) && major == 1 && minor >= 3,
          "glXQueryVersion = %d.%d", major, minor);
    CHECK(glXIsDirect(dpy, NULL) == True, "glXIsDirect");
    CHECK(strstr(glXQueryExtensionsString(dpy, DefaultScreen(dpy)), "GLX_ARB_create_context") != NULL,
          "GLX_ARB_create_context advertised");

    /* ---- visual + window -------------------------------------------- */
    int attribs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, None };
    XVisualInfo *vis = glXChooseVisual(dpy, DefaultScreen(dpy), attribs);
    CHECK(vis != NULL, "glXChooseVisual (visual %#lx depth %d)", vis ? vis->visualid : 0, vis ? vis->depth : 0);
    if (!vis)
        return 1;

    int rgba = -1, dbl = -1, depth = -1;
    glXGetConfig(dpy, vis, GLX_RGBA, &rgba);
    glXGetConfig(dpy, vis, GLX_DOUBLEBUFFER, &dbl);
    glXGetConfig(dpy, vis, GLX_DEPTH_SIZE, &depth);
    CHECK(rgba == True && dbl == True && depth >= 24, "glXGetConfig rgba=%d dbl=%d depth=%d", rgba, dbl, depth);

    XSetWindowAttributes swa;
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vis->screen), vis->visual, AllocNone);
    swa.event_mask = ExposureMask | StructureNotifyMask;
    Window win = XCreateWindow(dpy, RootWindow(dpy, vis->screen), 0, 0, 300, 200, 0,
                               vis->depth, InputOutput, vis->visual, CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "glx_smoke (MGL)");
    XMapWindow(dpy, win);
    XSync(dpy, False);

    /* ---- legacy context: compatibility profile ---------------------- */
    GLXContext ctx = glXCreateContext(dpy, vis, NULL, True);
    CHECK(ctx != NULL, "glXCreateContext");
    CHECK(glXMakeCurrent(dpy, win, ctx) == True, "glXMakeCurrent");
    CHECK(glXGetCurrentContext() == ctx && glXGetCurrentDrawable() == (GLXDrawable)win,
          "glXGetCurrentContext/Drawable");
    CHECK(glXGetCurrentDisplay() == dpy, "glXGetCurrentDisplay");

    const char *ver = (const char *)glGetString(GL_VERSION);
    CHECK(ver && strncmp(ver, "4.6", 3) == 0, "GL_VERSION = \"%s\"", ver ? ver : "(null)");
    CHECK(ver && strstr(ver, "compat") != NULL, "compat version string");
    GLint mask = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    CHECK((mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) != 0, "profile mask = 0x%x", mask);

    /* ---- draw with immediate mode, swap, verify pixels on the X side - */
    glViewport(0, 0, 300, 200);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    glColor3f(0, 1, 0);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
    glEnd();
    GLenum imm_err = glGetError();
    CHECK(imm_err == GL_NO_ERROR, "glBegin/glEnd triangle (glGetError 0x%04x)", imm_err);
    glXSwapBuffers(dpy, win);
    XSync(dpy, False);

    /* read the CENTER pixel from the X server: it must be the triangle */
    XImage *img = XGetImage(dpy, win, 0, 0, 300, 200, AllPlanes, ZPixmap);
    int cx = -1, cy = -1;
    if (img)
    {
        unsigned long p = XGetPixel(img, 150, 100);
        cx = (int)((p >> 16) & 0xff);   /* TrueColor 24-bit: r g b */
        cy = (int)((p >> 8) & 0xff);
        XDestroyImage(img);
    }
    CHECK(img != NULL && cy > 150 && cx < 100, "window center pixel r=%d g=%d (green expected)", cx, cy);

    /* a top pixel must be the clear colour (y-flip sanity) */
    img = XGetImage(dpy, win, 0, 0, 300, 200, AllPlanes, ZPixmap);
    int top_g = -1;
    if (img)
    {
        unsigned long p = XGetPixel(img, 150, 1);
        top_g = (int)((p >> 8) & 0xff);
        XDestroyImage(img);
    }
    CHECK(top_g >= 40 && top_g <= 70, "top pixel g=%d (clear colour 51)", top_g);

    /* ---- glXUseXFont: build 2 display lists from the default font ---- */
    XFontStruct *fs = XLoadQueryFont(dpy, "fixed");
    CHECK(fs != NULL, "XLoadQueryFont");
    p_glXUseXFont = (void (*)(Display *, XID, int, int, int))glXGetProcAddress((const GLubyte *)"glXUseXFont");
    CHECK(p_glXUseXFont != NULL, "glXGetProcAddress(glXUseXFont)");
    if (fs)
    {
        if (p_glXUseXFont) p_glXUseXFont(dpy, fs->fid, 'A', 2, 1000);
        CHECK(glGetError() == GL_NO_ERROR, "glXUseXFont (error 0x%04x)", glGetError());
        (void)glGetError();
        glNewList(2000, GL_COMPILE);
        glRasterPos2f(-0.9f, 0.8f);
        glCallList(1000);
        glEndList();
        glCallList(2000);
        CHECK(glGetError() == GL_NO_ERROR, "font display list (error 0x%04x)", glGetError());
        XFreeFont(dpy, fs);
    }

    /* ---- fbconfig API + a core context via attribs ------------------- */
    int nfbc = 0;
    GLXFBConfig *fbcs = glXChooseFBConfig(dpy, DefaultScreen(dpy), NULL, &nfbc);
    CHECK(fbcs != NULL && nfbc > 0, "glXChooseFBConfig (%d configs)", nfbc);
    if (fbcs && nfbc > 0)
    {
        int v = -1;
        glXGetFBConfigAttrib(dpy, fbcs[0], GLX_DOUBLEBUFFER, &v);
        CHECK(v == True, "first fbconfig double buffered (v=%d)", v);
        XVisualInfo *fvis = glXGetVisualFromFBConfig(dpy, fbcs[0]);
        CHECK(fvis != NULL, "glXGetVisualFromFBConfig");
        if (fvis)
            XFree(fvis);
        XFree(fbcs);
    }

    int core_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB, 6,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };
    GLXContext core = glXCreateContextAttribsARB(dpy, NULL, NULL, True, core_attribs);
    CHECK(core != NULL, "glXCreateContextAttribsARB (4.6 core)");
    if (core)
    {
        CHECK(glXMakeCurrent(dpy, win, core) == True, "make core current");
        glBegin(GL_TRIANGLES);
        glVertex2f(0, 0); glVertex2f(1, 0); glVertex2f(0, 1);
        glEnd();
        CHECK(glGetError() == GL_INVALID_OPERATION, "glBegin rejected in core (0x%04x)", glGetError());
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
        CHECK((mask & GL_CONTEXT_CORE_PROFILE_BIT) != 0, "core profile mask = 0x%x", mask);
        glXDestroyContext(dpy, core);
    }

    /* compat profile via attribs as well */
    int compat_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 3,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
        None
    };
    GLXContext compat = glXCreateContextAttribsARB(dpy, NULL, NULL, True, compat_attribs);
    CHECK(compat != NULL, "glXCreateContextAttribsARB (3.3 compat)");
    if (compat)
        glXDestroyContext(dpy, compat);

    /* core below 3.2 must be rejected */
    int bad_attribs[] = {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 1,
        GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
        None
    };
    GLXContext bad = glXCreateContextAttribsARB(dpy, NULL, NULL, True, bad_attribs);
    CHECK(bad == NULL, "3.1 core rejected");

    /* unbind + destroy */
    CHECK(glXMakeCurrent(dpy, None, NULL) == True, "glXMakeCurrent(None, NULL)");
    CHECK(glXGetCurrentContext() == NULL, "current context cleared");
    glXDestroyContext(dpy, ctx);

    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);

    printf("%s: %d failed checks\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
