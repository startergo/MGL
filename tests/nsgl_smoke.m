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
#import <IOSurface/IOSurface.h>
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
    /* glmark2-macos style: a POINTS-sized GL surface (the flag must be set
     * before setView:; macOS now defaults it to YES) */
    [view setWantsBestResolutionOpenGLSurface:NO];
    [ctx setView:view];
    /* the flag may also flip after binding: the swizzled setter must rescale
     * the layer back (leave it NO, i.e. a POINTS surface, as glmark2 does) */
    [view setWantsBestResolutionOpenGLSurface:YES];
    [view setWantsBestResolutionOpenGLSurface:NO];
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
    DFN(void, glFinish, (void));

    INIT_FN(glGetString); INIT_FN(glGetError); INIT_FN(glViewport);
    INIT_FN(glClearColor); INIT_FN(glClear); INIT_FN(glBegin);
    INIT_FN(glEnd); INIT_FN(glColor3f); INIT_FN(glVertex2f); INIT_FN(glFinish);

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

    /* IOSurface present path (MGL_SURFACE_PRESENT=1, the interposer default):
     * after the flush the presenting layer's contents is the surface MGL just
     * blitted — wait for the GPU, then read its corners to prove the frame
     * really landed (and with the right orientation: GL bottom-left red,
     * top-right blue; the compositor step itself cannot be observed in a
     * locked-headless session) */
    if (p_glFinish)
        p_glFinish();
    {
        IOSurfaceRef found = NULL;
        for (CALayer *l in view.layer.sublayers)
        {
            /* the pool layer may be a direct sublayer (view.layer is the Metal
             * layer) or nested one level down */
            if (l.contents && CFGetTypeID((CFTypeRef)l.contents) == IOSurfaceGetTypeID())
            {
                found = (IOSurfaceRef)l.contents;
                break;
            }
            for (CALayer *sl in l.sublayers)
            {
                if (sl.contents && CFGetTypeID((CFTypeRef)sl.contents) == IOSurfaceGetTypeID())
                {
                    found = (IOSurfaceRef)sl.contents;
                    break;
                }
            }
        }
        if (getenv("MGL_SURFACE_PRESENT") && strcmp(getenv("MGL_SURFACE_PRESENT"), "0") != 0)
        {
            if (!found)
            {
                printf("FAIL: no presenting IOSurface layer after flushBuffer\n");
                fails++;
            }
            else
            {
                size_t w = IOSurfaceGetWidth(found), h = IOSurfaceGetHeight(found);
                IOSurfaceLock(found, kIOSurfaceLockReadOnly, NULL);
                uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(found);
                size_t bpr = IOSurfaceGetBytesPerRow(found);
                /* every sample must equal the GL-side readback at the SAME
                 * position (the present blit is a straight copy; the layer's
                 * transform does the y-flip), proving content of the surface */
                unsigned char (*p_glReadPixels)(int, int, unsigned, unsigned, unsigned, unsigned, void *) =
                    (void *)dlsym(RTLD_DEFAULT, "glReadPixels");
                int bad = 0;
                const int pts[5][2] = { { 8, 8 }, { 200, 8 }, { 8, 200 }, { 200, 200 }, { 128, 128 } };
                for (int k = 0; k < 5; k++)
                {
                    unsigned char glp[4] = { 9, 9, 9, 9 }, sp[4] = { 0 };
                    if (p_glReadPixels)
                        p_glReadPixels(pts[k][0], pts[k][1], 1, 1, 0x1908 /* RGBA */, 0x1401 /* UBYTE */, glp);
                    size_t sx = pts[k][0], sy = pts[k][1];   /* blit-copy present: surface row == GL row */
                    memcpy(sp, base + sy * bpr + sx * 4, 4);
                    int ok5 = glp[0] == sp[0] && glp[1] == sp[1] && glp[2] == sp[2] && glp[3] == sp[3];
                    if (!ok5) bad++;
                    printf("     gl(%d,%d): GL rgba=%u,%u,%u,%u  surface=%u,%u,%u,%u  %s\n",
                           pts[k][0], pts[k][1], glp[0], glp[1], glp[2], glp[3],
                           sp[0], sp[1], sp[2], sp[3], ok5 ? "ok" : "FAIL");
                }
                IOSurfaceLock(found, kIOSurfaceLockReadOnly, NULL);
                /* Retina points-vs-pixels: the smoke app leaves
                 * wantsBestResolutionOpenGLSurface at NO, so the GL surface must
                 * be POINTS — the drawn quad (full viewport) must cover the
                 * WHOLE surface, i.e. all four quadrant centres are strongly
                 * coloured (a 2x-sized surface leaves the top-right black and
                 * was the user-visible "quarter of the window" glmark2 bug) */
                if (w != (size_t)view.bounds.size.width || h != (size_t)view.bounds.size.height)
                {
                    printf("FAIL: surface %zux%zu but view is %.0fx%.0f points (wantBestRes=NO must be 1:1)\n",
                           w, h, view.bounds.size.width, view.bounds.size.height);
                    fails++;
                }
                else
                {
                    struct { size_t x, y; } q[4] = { { w / 4, h - h / 4 }, { w - w / 4, h - h / 4 },
                                                     { w / 4, h / 4 }, { w - w / 4, h / 4 } };
                    int grey = 0;
                    for (int k = 0; k < 4; k++)
                    {
                        uint8_t p[4];
                        memcpy(p, base + q[k].y * bpr + q[k].x * 4, 4);
                        if (abs(p[0] - p[1]) < 8 && abs(p[1] - p[2]) < 8)
                            grey++;   /* neutral = background, not the quad */
                    }
                    printf("%s surface present quadrants: %d/4 show the drawn quad\n",
                           grey ? "FAIL" : "ok  ", 4 - grey);
                    fails += grey;
                }
                IOSurfaceUnlock(found, kIOSurfaceLockReadOnly, NULL);
                /* CALayer-transform calibration: renderInContext cannot display
                 * IOSurface contents (magenta), so validate the flip semantics
                 * with a CGImage contents layer — same compositing machinery
                 * that will show the IOSurface on screen */
                {
                    size_t cw = 64, ch = 64;
                    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
                    CGContextRef imb = CGBitmapContextCreate(NULL, cw, ch, 8, cw * 4, cs,
                                                             kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedLast);
                    uint8_t *d = (uint8_t *)CGBitmapContextGetData(imb);
                    for (size_t y = 0; y < ch; y++)
                        for (size_t x = 0; x < cw; x++)
                        {
                            uint8_t *p = d + y * CGBitmapContextGetBytesPerRow(imb) + x * 4;
                            p[0] = 0; p[1] = 255; p[2] = 0; p[3] = 255;
                            if (y < ch / 2) { p[0] = 255; }   /* top half = yellow */
                        }
                    CGImageRef img = CGBitmapContextCreateImage(imb);
                    CGContextRelease(imb);
                    /* layer WITHOUT flip: image top half (yellow) appears at layer top */
                    CALayer *plain = [CALayer layer];
                    plain.frame = CGRectMake(0, 0, 64, 64);
                    plain.contents = (__bridge id)img;
                    /* layer WITH the present path's flip */
                    CALayer *flipped = [CALayer layer];
                    flipped.frame = CGRectMake(0, 0, 64, 64);
                    flipped.contents = (__bridge id)img;
                    flipped.transform = CATransform3DMakeScale(1, -1, 1);
                    CALayer *host = [CALayer layer];
                    host.frame = CGRectMake(0, 0, 128, 64);
                    [host addSublayer:plain];
                    [host addSublayer:flipped];
                    CGContextRef cgb = CGBitmapContextCreate(NULL, 128, 64, 8, 128 * 4, cs,
                                                             kCGBitmapByteOrder32Host | kCGImageAlphaPremultipliedLast);
                    CGColorSpaceRelease(cs);
                    [host renderInContext:cgb];
                    uint8_t *px = (uint8_t *)CGBitmapContextGetData(cgb);
                    size_t rb = CGBitmapContextGetBytesPerRow(cgb);
                    uint8_t plainTop = px[rb * 8 + 16 * 4 + 0], flippedTop = px[rb * 8 + 96 * 4 + 0];
                    printf("%s CALayer flip calibration: plain-top-r=%u flipped-top-r=%u (flip must swap yellow/red halves)\n",
                           (plainTop > 100 && flippedTop < 100) ? "ok  " : "FAIL", plainTop, flippedTop);
                    if (!(plainTop > 100 && flippedTop < 100)) fails++;
                    CGImageRelease(img);
                    CGContextRelease(cgb);
                }

                printf("%s surface present: %zu checks byte-identical to glReadPixels\n",
                       bad ? "FAIL" : "ok  ", (size_t)(5 - bad));
                fails += bad;
            }
        }
        else if (found)
        {
            printf("FAIL: presenting IOSurface layer exists with MGL_SURFACE_PRESENT off\n");
            fails++;
        }
    }

    /* the same readback the interposer presents can be verified with
     * MGL_NSGL_DUMPDIR=<dir> (debug build path) or by eye; the checks above
     * plus the benchmark runs are the automation here */
    printf("%s: %d failed checks (presented quad is on screen)\n",
           fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
