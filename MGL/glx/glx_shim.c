/*
 * glx_shim.c — a GLX 1.4 front end for MGL: X11 apps that link libGL get
 * real GL 4.6 (core or compatibility) on Metal through this shim.
 *
 *   MGL/glx/glx_shim.c  (this file)  — contexts, drawables, make-current, swap
 *   MGL/glx/glx_config.c            — fbconfig/visual tables + attribute logic
 *
 * Design (after Mesa's xlib GLX driver, MIT licensed):
 *  - a GLXContext is an MGL context (compatibility profile for the legacy
 *    glXCreateContext / glXCreateNewContext; glXCreateContextAttribsARB honours
 *    GLX_CONTEXT_PROFILE_MASK_ARB) with a headless MGLRenderer bound to it;
 *  - GL renders into the MGL default framebuffer, which for headless contexts
 *    sizes itself to the union of the glViewport()s ever used.  The shim sizes
 *    it exactly by setting the viewport to the drawable size the first time a
 *    context is made current (GLX spec behaviour);
 *  - glXSwapBuffers = glFinish + glReadPixels(BGRA) + (XShm)PutImage with the
 *    GL->X y flip into the window.  Depth/stencil live in the same offscreen
 *    framebuffer, so X visuals are always the server's TrueColor 24-bit one.
 *
 * Licensed under the Apache License, Version 2.0 (see the repository LICENSE).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sysctl.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xlibint.h>
#include <X11/extensions/XShm.h>

#define GL_GLEXT_PROTOTYPES 1
#include "glcorearb.h"      /* core GL entry points (re-exported from libmgl) */
#include "gl_compat.h"     /* compatibility-profile entry points (compat domain) */

/* GLX shim types (X-free header; the X11 headers above provide the
 * Display/Window types used by the entry points below) */
#include "glx_types.h"

/* MGL exports (declared locally: MGLContext.h has C++ default arguments) */
extern void *createGLMContextProfile(GLenum format, GLenum type,
                                     GLenum depth_format, GLenum depth_type,
                                     GLenum stencil_format, GLenum stencil_type,
                                     GLenum profile);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *MGLgetCurrentContext(void);

#ifndef GL_CONTEXT_CORE_PROFILE_BIT
#define GL_CONTEXT_CORE_PROFILE_BIT          0x00000001
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT 0x00000002
#endif

/* ------------------------------------------------------------------ */
/* per-display state                                                   */

#define MGLX_MAX_DRAWABLES 128

enum { MGLX_DRAW_WINDOW = 1, MGLX_DRAW_PIXMAP, MGLX_DRAW_PBUFFER };

typedef struct MGLXDrawableRec {
    XID xid;                /* the X resource */
    int type;
    unsigned w, h;          /* last known geometry */
    unsigned fb_w, fb_h;    /* union of viewports seen (readback clamp) */
    unsigned long event_mask;
    /* swap image (created lazily, resized on geometry change) */
    XImage *img;
    char *img_data;         /* non-shm data buffer */
    int shm;
    XShmSegmentInfo shminfo;
    unsigned char *read_buf;    /* glReadPixels staging (row 0 = bottom) */
    size_t read_buf_size;
    GC gc;
} MGLXDrawableRec;

typedef struct MGLXDisplayRec {
    Display *dpy;
    int screen;
    VisualID visualid;
    int depth;
    MGLXFBConfigRec configs[MGLX_NCONFIGS];
    XExtCodes *codes;       /* error base for GLX errors */
    MGLXDrawableRec drawables[MGLX_MAX_DRAWABLES];
} MGLXDisplayRec;

struct MGLXContextRec {
    void *glm;              /* GLMContext */
    int is_compat;
    int render_type;        /* GLX_RGBA_TYPE */
    int screen;
    MGLXFBConfigRec *cfg;
    MGLXDisplayRec *dpy_state;
    int ever_current;
};

#define MGLX_MAX_DISPLAYS 8
static MGLXDisplayRec s_displays[MGLX_MAX_DISPLAYS];
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;

/* the GLX "current" record */
static struct {
    Display *dpy;
    MGLXDrawableRec *draw;
    MGLXDrawableRec *read;
    MGLXContext ctx;
} s_current;

/* libmgl's auto-created core context: target for glXMakeCurrent(dpy,None,NULL)
 * so that stray GL calls after unbinding do not dereference NULL */
static void *s_init_ctx;

/* live-context registry: XID (the MGLXContext pointer) -> context, used by the
 * GLX_EXT_import_context entry points (glXImportContextEXT has to turn the
 * XID glXGetContextIDEXT returned back into a GLXContext) */
#define MGLX_MAX_CONTEXTS 64
static MGLXContext s_contexts[MGLX_MAX_CONTEXTS];
static int s_ncontexts;

static void mglx_register_context(MGLXContext c)
{
    for (int i = 0; i < MGLX_MAX_CONTEXTS; i++)
        if (!s_contexts[i])
        {
            s_contexts[i] = c;
            s_ncontexts++;
            return;
        }
}

static void mglx_unregister_context(MGLXContext c)
{
    for (int i = 0; i < MGLX_MAX_CONTEXTS; i++)
        if (s_contexts[i] == c)
        {
            s_contexts[i] = NULL;
            s_ncontexts--;
            return;
        }
}

static MGLXContext mglx_find_context_by_xid(XID xid)
{
    for (int i = 0; i < MGLX_MAX_CONTEXTS; i++)
        if (s_contexts[i] && (XID)(uintptr_t)s_contexts[i] == xid)
            return s_contexts[i];
    return NULL;
}

__attribute__((constructor))
static void mglx_init(void)
{
    s_init_ctx = MGLgetCurrentContext();
}

/* ------------------------------------------------------------------ */
/* errors                                                             */

/*
 * GLX errors are reported through the return value of the entry points, not
 * as X protocol errors: synthesizing an X error from client code confuses
 * Xlib's sequence bookkeeping (the reader side of _XError expects to run from
 * the reply loop) and would kill apps that rely on the default error handler.
 * Documented deviation: an app that installs an X error handler and expects
 * GLXBadFBConfig/GLXBadAttribute codes will not see them.
 */
static int mglx_send_error(Display *dpy, int glx_code, XID resource, int minor)
{
    (void)dpy; (void)glx_code; (void)resource; (void)minor;
    return 0;
}

static MGLXDisplayRec *mglx_display(Display *dpy)
{
    MGLXDisplayRec *free_slot = NULL;

    if (!dpy)
        return NULL;

    for (int i = 0; i < MGLX_MAX_DISPLAYS; i++)
    {
        if (s_displays[i].dpy == dpy)
            return &s_displays[i];
        if (!s_displays[i].dpy && !free_slot)
            free_slot = &s_displays[i];
    }
    if (!free_slot)
        return NULL;

    /* new display: pick the screen's TrueColor 24-bit visual */
    int screen = DefaultScreen(dpy);
    XVisualInfo tmpl, *vis;
    int n = 0;
    memset(&tmpl, 0, sizeof tmpl);
    tmpl.screen = screen;
    tmpl.depth = 24;
    tmpl.class = TrueColor;
    vis = XGetVisualInfo(dpy, VisualScreenMask | VisualDepthMask | VisualClassMask,
                         &tmpl, &n);
    if (!vis || n == 0)
    {
        /* fall back to the default visual if it is usable */
        Visual *dv = DefaultVisual(dpy, screen);
        if (dv && (dv->class == TrueColor || dv->class == DirectColor))
        {
            memset(&tmpl, 0, sizeof tmpl);
            tmpl.screen = screen;
            tmpl.visualid = XVisualIDFromVisual(dv);
            vis = XGetVisualInfo(dpy, VisualScreenMask | VisualIDMask, &tmpl, &n);
        }
    }
    if (!vis || n == 0)
        return NULL;    /* no TrueColor visual: no GLX */

    memset(free_slot, 0, sizeof *free_slot);
    free_slot->dpy = dpy;
    free_slot->screen = screen;
    free_slot->visualid = vis[0].visualid;
    free_slot->depth = vis[0].depth;
    XFree(vis);

    mglx_build_configs(free_slot->configs, MGLX_NCONFIGS, screen,
                       free_slot->visualid, free_slot->depth, 0x101);

    /* register the GLX extension so GLX errors have a code base */
    free_slot->codes = XAddExtension(dpy);
    return free_slot;
}

static MGLXDrawableRec *mglx_find_drawable(MGLXDisplayRec *ds, XID xid)
{
    if (!ds || !xid)
        return NULL;
    for (int i = 0; i < MGLX_MAX_DRAWABLES; i++)
        if (ds->drawables[i].xid == xid)
            return &ds->drawables[i];
    return NULL;
}

static MGLXDrawableRec *mglx_add_drawable(MGLXDisplayRec *ds, XID xid, int type,
                                          unsigned w, unsigned h)
{
    MGLXDrawableRec *d;

    if (!ds || !xid || !w || !h)
        return NULL;
    if ((d = mglx_find_drawable(ds, xid)))
        return d;

    for (int i = 0; i < MGLX_MAX_DRAWABLES; i++)
        if (!ds->drawables[i].xid)
        {
            d = &ds->drawables[i];
            memset(d, 0, sizeof *d);
            d->xid = xid;
            d->type = type;
            d->w = w;
            d->h = h;
            d->fb_w = w;    /* grows with every viewport seen */
            d->fb_h = h;
            return d;
        }
    return NULL;
}

/* refresh w/h from the server (windows can be resized) */
static void mglx_refresh_geometry(Display *dpy, MGLXDrawableRec *d)
{
    Window root;
    int x, y;
    unsigned w, h, bw, depth;

    if (!d || d->type == MGLX_DRAW_PBUFFER)
        return;
    if (!XGetGeometry(dpy, d->xid, &root, &x, &y, &w, &h, &bw, &depth))
        return;
    d->w = w;
    d->h = h;
}

/* ------------------------------------------------------------------ */
/* GL context creation                                                */

static MGLXContext mglx_create_context(Display *dpy, MGLXFBConfigRec *cfg,
                                       int compat, int render_type)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXContext c;

    if (!ds)
        return NULL;
    if (!cfg)
        cfg = &ds->configs[0];

    c = (MGLXContext)calloc(1, sizeof *c);
    if (!c)
        return NULL;

    /* an MGL context with depth 24 + stencil 8; the headless renderer owns
     * the offscreen default framebuffer */
    c->glm = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                     GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                     GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                     compat ? GL_CONTEXT_COMPATIBILITY_PROFILE_BIT
                                            : GL_CONTEXT_CORE_PROFILE_BIT);
    if (!c->glm)
    {
        free(c);
        return NULL;
    }
    CppCreateMGLRendererHeadless(c->glm);

    /* verify the profile took effect: query the profile mask, NOT the version
     * string (GL_VERSION is the same "4.6 Metal - 90.5" in both profiles since
     * the s14 identification-string work, and substring matching it broke every
     * legacy glXCreateContext/glXCreateNewContext call) */
    void *prev = MGLgetCurrentContext();
    MGLsetCurrentContext(c->glm);
    GLint mask = 0;
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &mask);
    while (glGetError() != GL_NO_ERROR) {}
    int got_compat = (mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) != 0;
    MGLsetCurrentContext(prev ? prev : s_init_ctx);

    if (compat && !got_compat)
    {
        free(c);
        return NULL;
    }

    c->is_compat = got_compat;
    c->render_type = render_type;
    c->screen = ds->screen;
    c->cfg = cfg;
    c->dpy_state = ds;
    mglx_register_context(c);
    return c;
}

static void mglx_destroy_drawable_images(MGLXDisplayRec *ds, MGLXDrawableRec *d)
{
    if (d->img)
    {
        if (d->shm)
        {
            XShmDetach(ds->dpy, &d->shminfo);
            if (d->img->data)
                shmdt(d->shminfo.shmaddr);
            XDestroyImage(d->img);
        }
        else
        {
            XDestroyImage(d->img);  /* frees img->data == img_data */
        }
        d->img = NULL;
        d->img_data = NULL;
    }
    free(d->read_buf);
    d->read_buf = NULL;
    d->read_buf_size = 0;
    if (d->gc)
    {
        XFreeGC(ds->dpy, d->gc);
        d->gc = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* swap: read the default framebuffer back and push it to X           */

static int mglx_ensure_image(MGLXDisplayRec *ds, MGLXDrawableRec *d, unsigned w, unsigned h)
{
    Display *dpy = ds->dpy;

    if (d->img && d->img->width == (int)w && d->img->height == (int)h)
        return 1;
    mglx_destroy_drawable_images(ds, d);

    if (!d->gc)
        d->gc = XCreateGC(dpy, d->xid, 0, NULL);

    /* depth follows the drawable: X11 depth-24 ZPixmap packs pixels into 24
     * bits with 32-bit scanline padding (3 bytes/pixel + pad), while depth-32
     * drawables take a full BGRA quad.  Xlib computes the protocol stride
     * (bytes_per_line) and the bits per pixel; we allocate to match. */
    Window root;
    int x, y;
    unsigned iw, ih, bw, depth = 0;
    if (!XGetGeometry(dpy, d->xid, &root, &x, &y, &iw, &ih, &bw, &depth))
        return 0;

    d->img = XCreateImage(dpy, NULL, (unsigned int)depth, ZPixmap, 0, NULL,
                          w, h, 32, 0);
    if (!d->img)
        return 0;
    d->img_data = (char *)malloc((size_t)d->img->bytes_per_line * h);
    if (!d->img_data)
    {
        XFree(d->img);
        d->img = NULL;
        return 0;
    }
    d->img->data = d->img_data;
    d->shm = 0;

    d->read_buf_size = (size_t)w * h * 4;
    d->read_buf = (unsigned char *)realloc(d->read_buf, d->read_buf_size);
    if (!d->read_buf)
        return 0;
    return 1;
}

static void mglx_swap_buffers(Display *dpy, MGLXDrawableRec *d)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    unsigned w, h, rw, rh;

    if (!ds || !d)
        return;
    mglx_refresh_geometry(dpy, d);

    w = d->w;
    h = d->h;
    if (!w || !h)
        return;

    /* pbuffers have no X surface to present to */
    if (d->type == MGLX_DRAW_PBUFFER)
        return;

    /* clamp to what the GL framebuffer actually covers: its size is the union
     * of the viewports used, tracked in fb_w/fb_h */
    rw = w < d->fb_w ? w : d->fb_w;
    rh = h < d->fb_h ? h : d->fb_h;
    if (!rw || !rh)
        return;

    if (!mglx_ensure_image(ds, d, w, h))
        return;

    /* GPU-side barrier before the CPU readback: a fence is enough (glFinish
     * would flush every pending command buffer on the queue) */
    GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (fence)
    {
        while (glGetError() != GL_NO_ERROR) {}
        GLenum fs;
        do {
            fs = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000ULL /* 1 ms */);
        } while (fs == GL_TIMEOUT_EXPIRED);
        glDeleteSync(fence);
        while (glGetError() != GL_NO_ERROR) {}
    }

    /* BGRA + 8_8_8_8_REV is exactly the X11 TrueColor byte order on
     * little-endian; fall back to RGBA + CPU swizzle if a server needs MSB */
    GLenum fmt = GL_BGRA, type = GL_UNSIGNED_INT_8_8_8_8_REV;
    glReadPixels(0, 0, (GLsizei)rw, (GLsizei)rh, fmt, type, d->read_buf);
    if (glGetError() != GL_NO_ERROR)
    {
        fmt = GL_RGBA;
        type = GL_UNSIGNED_BYTE;
        glReadPixels(0, 0, (GLsizei)rw, (GLsizei)rh, fmt, type, d->read_buf);
        while (glGetError() != GL_NO_ERROR) {}
    }

    /* copy row-by-row with the GL->X flip: X row 0 is the top, GL row 0 the
     * bottom, so GL row r lands in X row rh-1-r.  The image is anchored at
     * the bottom-left of the drawable.  Depth-24 drawables take 3 bytes/pixel
     * (B,G,R), depth-32 take BGRA; an RGBA fallback read is swizzled. */
    int bpp24 = d->img->bits_per_pixel <= 24;
    int msn = d->img->byte_order == MSBFirst;
    for (unsigned r = 0; r < rh; r++)
    {
        const unsigned char *src = d->read_buf + (size_t)r * rw * 4;
        unsigned char *dst = (unsigned char *)d->img->data +
                             (size_t)(h - 1 - r) * d->img->bytes_per_line;
        if (fmt == GL_BGRA && !msn)
        {
            if (!bpp24)
                memcpy(dst, src, (size_t)rw * 4);
            else
                for (unsigned x = 0; x < rw; x++)
                {
                    dst[x * 3 + 0] = src[x * 4 + 0];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 2];
                }
        }
        else if (fmt == GL_BGRA && msn)
        {
            for (unsigned x = 0; x < rw; x++)
            {
                unsigned char o[4] = { src[x*4+3], src[x*4+2], src[x*4+1], src[x*4+0] };
                dst[x * (bpp24 ? 3 : 4) + 0] = o[0];
                dst[x * (bpp24 ? 3 : 4) + 1] = o[1];
                dst[x * (bpp24 ? 3 : 4) + 2] = o[2];
                if (!bpp24) dst[x * 4 + 3] = o[3];
            }
        }
        else    /* RGBA source: swizzle to BGR(A) */
        {
            for (unsigned x = 0; x < rw; x++)
            {
                dst[x * (bpp24 ? 3 : 4) + 0] = src[x * 4 + 2];
                dst[x * (bpp24 ? 3 : 4) + 1] = src[x * 4 + 1];
                dst[x * (bpp24 ? 3 : 4) + 2] = src[x * 4 + 0];
                if (!bpp24) dst[x * 4 + 3] = src[x * 4 + 3];
            }
        }
    }

    XPutImage(dpy, d->xid, d->gc, d->img, 0, (int)(h - rh), 0, (int)(h - rh),
              rw, rh);
    XSync(dpy, False);
}

/* ------------------------------------------------------------------ */
/* GLX 1.0/1.1 entry points                                           */

XVisualInfo *glXChooseVisual(Display *dpy, int screen, int *attrib_list)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    const char *bad = NULL;

    if (!ds)
        return NULL;
    if (screen != ds->screen)
        return NULL;

    for (int i = 0; i < MGLX_NCONFIGS; i++)
    {
        int m = mglx_config_matches(&ds->configs[i], attrib_list, 0, &bad);
        if (m < 0)
        {
            mglx_send_error(dpy, GLX_BAD_ATTRIBUTE, None, 0);
            return NULL;
        }
        if (m > 0)
        {
            XVisualInfo tmpl;
            int n = 0;
            memset(&tmpl, 0, sizeof tmpl);
            tmpl.visualid = ds->configs[i].visualid;
            tmpl.screen = screen;
            XVisualInfo *vis = XGetVisualInfo(dpy, VisualIDMask | VisualScreenMask,
                                              &tmpl, &n);
            if (vis && n > 0)
                return vis;     /* caller frees with XFree */
            if (vis)
                XFree(vis);
            return NULL;
        }
    }
    return NULL;
}

int glXGetConfig(Display *dpy, XVisualInfo *visual, int attribute, int *value)
{
    MGLXDisplayRec *ds = mglx_display(dpy);

    if (!value)
        return GLX_BAD_VALUE;
    *value = 0;

    if (!ds || !visual)
        return GLX_BAD_VISUAL;

    if (visual->visualid != ds->visualid)
        return GLX_BAD_VISUAL;  /* only our chosen visual has GLX support */

    if (attribute == GLX_USE_GL)
    {
        *value = 1;
        return 0;
    }

    int v = mglx_config_attrib(&ds->configs[0], attribute);
    if (v == GLX_NONE)
        return GLX_BAD_ATTRIBUTE;
    *value = v;
    return 0;
}

GLXContext glXCreateContext(Display *dpy, XVisualInfo *visual,
                            GLXContext share_list, Bool direct)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !visual || visual->visualid != ds->visualid)
    {
        mglx_send_error(dpy, GLX_BAD_VISUAL, None, 0);
        return NULL;
    }
    /* share_list is accepted but display lists/objects stay per-context in
     * MGL (documented gap: no share pools between GLX contexts) */
    return mglx_create_context(dpy, &ds->configs[0], 1 /* compat */, GLX_RGBA_TYPE);
}

void glXDestroyContext(Display *dpy, GLXContext ctx)
{
    (void)dpy;
    if (ctx == s_current.ctx)
    {
        s_current.ctx = NULL;
        s_current.dpy = NULL;
        s_current.draw = s_current.read = NULL;
        MGLsetCurrentContext(s_init_ctx);
    }
    /* the MGL context itself is intentionally kept alive: MGL has no context
     * teardown path, and apps create a bounded number of contexts */
    mglx_unregister_context(ctx);
    free(ctx);
}

static Bool mglx_make_current(Display *dpy, MGLXDrawableRec *draw,
                              MGLXDrawableRec *read, MGLXContext ctx)
{
    pthread_mutex_lock(&s_lock);

    if (!ctx)
    {
        s_current.dpy = NULL;
        s_current.draw = s_current.read = NULL;
        s_current.ctx = NULL;
        MGLsetCurrentContext(s_init_ctx);
        pthread_mutex_unlock(&s_lock);
        return True;
    }

    if (!draw || !read)
    {
        pthread_mutex_unlock(&s_lock);
        return False;
    }

    MGLsetCurrentContext(ctx->glm);

    /* first make-current of a context: viewport/scissor = drawable size (GLX
     * spec); this also sizes MGL's offscreen default framebuffer exactly */
    if (!ctx->ever_current)
    {
        glViewport(0, 0, (GLsizei)draw->w, (GLsizei)draw->h);
        glScissor(0, 0, (GLsizei)draw->w, (GLsizei)draw->h);
        ctx->ever_current = 1;
    }

    /* track the framebuffer the GL side can cover: MGL sizes its default
     * framebuffer to the union of the viewports ever used */
    {
        GLint vp[4] = { 0, 0, 0, 0 };
        glGetIntegerv(GL_VIEWPORT, vp);
        while (glGetError() != GL_NO_ERROR) {}
        unsigned vw = vp[2] > 0 ? (unsigned)vp[2] : draw->w;
        unsigned vh = vp[3] > 0 ? (unsigned)vp[3] : draw->h;
        if (vw > draw->fb_w) draw->fb_w = vw;
        if (vh > draw->fb_h) draw->fb_h = vh;
    }

    s_current.dpy = dpy;
    s_current.draw = draw;
    s_current.read = read;
    s_current.ctx = ctx;
    pthread_mutex_unlock(&s_lock);
    return True;
}

Bool glXMakeCurrent(Display *dpy, GLXDrawable drawable, GLXContext ctx)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d;

    if (!ctx && drawable == None)
        return mglx_make_current(dpy, NULL, NULL, NULL);

    if (!ctx || !ds)
    {
        mglx_send_error(dpy, GLX_BAD_CONTEXT, (XID)(uintptr_t)ctx, 0);
        return False;
    }

    mglx_refresh_geometry(dpy, mglx_find_drawable(ds, drawable));
    d = mglx_add_drawable(ds, drawable, MGLX_DRAW_WINDOW, 1, 1);
    if (!d)
    {
        mglx_send_error(dpy, GLX_BAD_DRAWABLE, drawable, 0);
        return False;
    }
    mglx_refresh_geometry(dpy, d);
    if (d->w == 0 || d->h == 0) { d->w = 1; d->h = 1; }

    return mglx_make_current(dpy, d, d, ctx);
}

Bool glXMakeContextCurrent(Display *dpy, GLXDrawable draw, GLXDrawable read,
                           GLXContext ctx)
{
    MGLXDisplayRec *ds = mglx_display(dpy);

    if (!ctx && draw == None && read == None)
        return mglx_make_current(dpy, NULL, NULL, NULL);
    if (!ctx || !ds)
    {
        mglx_send_error(dpy, GLX_BAD_CONTEXT, (XID)(uintptr_t)ctx, 0);
        return False;
    }

    MGLXDrawableRec *dd = mglx_find_drawable(ds, draw);
    if (!dd)
    {
        dd = mglx_add_drawable(ds, draw, MGLX_DRAW_WINDOW, 1, 1);
        if (dd) mglx_refresh_geometry(dpy, dd);
    }
    MGLXDrawableRec *rd = dd;
    if (read != draw)
    {
        rd = mglx_find_drawable(ds, read);
        if (!rd)
            rd = mglx_add_drawable(ds, read, MGLX_DRAW_WINDOW, 1, 1);
    }
    if (!dd || !rd)
    {
        mglx_send_error(dpy, GLX_BAD_DRAWABLE, draw, 0);
        return False;
    }
    return mglx_make_current(dpy, dd, rd, ctx);
}

GLXContext glXGetCurrentContext(void)
{
    return s_current.ctx;
}

GLXDrawable glXGetCurrentDrawable(void)
{
    return s_current.draw ? s_current.draw->xid : None;
}

GLXDrawable glXGetCurrentReadDrawable(void)
{
    return s_current.read ? s_current.read->xid : None;
}

Display *glXGetCurrentDisplay(void)
{
    return s_current.dpy;
}

Bool glXQueryExtension(Display *dpy, int *error_base, int *event_base)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds)
        return False;
    if (error_base)
        *error_base = ds->codes ? ds->codes->first_error : 0;
    if (event_base)
        *event_base = ds->codes ? ds->codes->first_event : 0;
    return True;
}

Bool glXQueryVersion(Display *dpy, int *major, int *minor)
{
    if (!mglx_display(dpy))
        return False;
    if (major) *major = 1;
    if (minor) *minor = 4;
    return True;
}

Bool glXIsDirect(Display *dpy, GLXContext ctx)
{
    (void)dpy; (void)ctx;
    return True;    /* rendering happens in this process, on Metal */
}

void glXSwapBuffers(Display *dpy, GLXDrawable drawable)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d;

    if (!ds)
        return;
    d = mglx_find_drawable(ds, drawable);
    if (!d)
    {
        mglx_refresh_geometry(dpy, NULL);
        d = mglx_add_drawable(ds, drawable, MGLX_DRAW_WINDOW, 1, 1);
        if (d)
            mglx_refresh_geometry(dpy, d);
    }
    if (!d || d->type == MGLX_DRAW_PBUFFER)
    {
        mglx_send_error(dpy, GLX_BAD_DRAWABLE, drawable, 0);
        return;
    }

    /* keep the readback clamp fresh: the viewport may have grown since */
    if (s_current.ctx && s_current.draw == d)
    {
        GLint vp[4];
        glGetIntegerv(GL_VIEWPORT, vp);
        while (glGetError() != GL_NO_ERROR) {}
        if (vp[2] > 0 && (unsigned)vp[2] > d->fb_w) d->fb_w = (unsigned)vp[2];
        if (vp[3] > 0 && (unsigned)vp[3] > d->fb_h) d->fb_h = (unsigned)vp[3];
    }

    mglx_swap_buffers(dpy, d);
}

const char *glXQueryExtensionsString(Display *dpy, int screen)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || screen != ds->screen)
        return NULL;
    return mglx_extensions_string();
}

const char *glXQueryServerString(Display *dpy, int screen, int name)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || screen != ds->screen)
        return NULL;
    return mglx_client_string(name);
}

const char *glXGetClientString(Display *dpy, int name)
{
    if (!mglx_display(dpy))
        return NULL;
    return mglx_client_string(name);
}

void glXWaitGL(void)
{
    glFinish();
}

void glXWaitX(void)
{
    if (s_current.dpy)
        XSync(s_current.dpy, False);
}

/*
 * glXUseXFont: rasterize each glyph through the X server (draw the character
 * into a 1-bit pixmap with the font, read it back) and record a glBitmap
 * display list, exactly what legacy X11 apps expect (Mesa's implementation
 * does the same server round trip).
 */
void glXUseXFont(Display *dpy, Font font, int first, int count, int list_base)
{
    XFontStruct *fs;

    if (!dpy || !count || !s_current.ctx)
        return;
    fs = XQueryFont(dpy, font);
    if (!fs)
        return;

    Window root = RootWindow(dpy, DefaultScreen(dpy));
    for (int i = first; i < first + count; i++)
    {
        XCharStruct *cs;
        char ch[2];

        ch[0] = (char)(i & 0xff);
        ch[1] = 0;
        if (fs->per_char &&
            i >= fs->min_char_or_byte2 && i <= fs->max_char_or_byte2)
            cs = &fs->per_char[i - fs->min_char_or_byte2];
        else
            cs = &fs->min_bounds;

        int w = cs->rbearing - cs->lbearing;
        int h = cs->ascent + cs->descent;
        if (w < 0) w = 0;
        if (h < 0) h = 0;

        unsigned char *bits = NULL;
        if (w > 0 && h > 0)
        {
            Pixmap pm = XCreatePixmap(dpy, root, (unsigned)w, (unsigned)h, 1);
            GC gc = XCreateGC(dpy, pm, 0, NULL);
            XSetForeground(dpy, gc, 0);
            XFillRectangle(dpy, pm, gc, 0, 0, (unsigned)w, (unsigned)h);
            XSetForeground(dpy, gc, 1);
            XSetFont(dpy, gc, font);
            XDrawString(dpy, pm, gc, -cs->lbearing, cs->ascent, ch, 1);
            XFreeGC(dpy, gc);

            XImage *img = XGetImage(dpy, pm, 0, 0, (unsigned)w, (unsigned)h,
                                    1, XYPixmap);
            if (img)
            {
                int bpr = (w + 7) / 8;
                bits = (unsigned char *)calloc((size_t)bpr * h, 1);
                if (bits)
                {
                    /* glBitmap rows are bottom-to-top, bits MSB-first */
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            if (XGetPixel(img, x, y))
                                bits[(size_t)(h - 1 - y) * bpr + x / 8] |=
                                    (unsigned char)(0x80u >> (x & 7));
                }
                XDestroyImage(img);
            }
            XFreePixmap(dpy, pm);
        }

        glNewList((GLuint)(list_base + i - first), GL_COMPILE);
        if (bits)
            glBitmap((GLsizei)w, (GLsizei)h,
                     (GLfloat)-cs->lbearing, (GLfloat)cs->descent,
                     (GLfloat)cs->width, 0, (const GLubyte *)bits);
        else
            glBitmap(0, 0, 0, 0, (GLfloat)cs->width, 0, NULL);
        glEndList();
        free(bits);
    }
    XFree(fs);
}

/* ------------------------------------------------------------------ */
/* GLX 1.3 fbconfig entry points                                      */

GLXFBConfig *glXChooseFBConfig(Display *dpy, int screen,
                               const int *attrib_list, int *nelements)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    GLXFBConfig *out;
    int n = 0;
    const char *bad = NULL;

    if (!nelements)
        return NULL;
    *nelements = 0;
    if (!ds || screen != ds->screen)
        return NULL;

    out = (GLXFBConfig *)malloc(sizeof(GLXFBConfig) * MGLX_NCONFIGS);
    if (!out)
        return NULL;

    for (int i = 0; i < MGLX_NCONFIGS; i++)
    {
        int m = mglx_config_matches(&ds->configs[i], attrib_list, 1, &bad);
        if (m < 0)
        {
            free(out);
            mglx_send_error(dpy, GLX_BAD_ATTRIBUTE, None, 0);
            return NULL;
        }
        if (m > 0)
            out[n++] = &ds->configs[i];
    }
    if (!n)
    {
        free(out);
        return NULL;
    }
    *nelements = n;
    return out;     /* freed with XFree */
}

int glXGetFBConfigAttrib(Display *dpy, GLXFBConfig config,
                         int attribute, int *value)
{
    if (!value)
        return GLX_BAD_VALUE;
    *value = 0;
    if (!config)
        return GLX_BAD_ATTRIBUTE;
    int v = mglx_config_attrib(config, attribute);
    if (v == GLX_NONE)
    {
        /* a couple of attributes need GL-determined answers */
        if (attribute == 0x20B2 /* GLX_FRAMEBUFFER_SRGB_CAPABLE_ARB */)
        {
            *value = 0;
            return 0;
        }
        return GLX_BAD_ATTRIBUTE;
    }
    *value = v;
    return 0;
}

XVisualInfo *glXGetVisualFromFBConfig(Display *dpy, GLXFBConfig config)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !config)
        return NULL;
    XVisualInfo tmpl;
    int n = 0;
    memset(&tmpl, 0, sizeof tmpl);
    tmpl.visualid = config->visualid;
    tmpl.screen = config->screen;
    XVisualInfo *vis = XGetVisualInfo(dpy, VisualIDMask | VisualScreenMask,
                                      &tmpl, &n);
    if (vis && n > 0)
        return vis;
    if (vis)
        XFree(vis);
    return NULL;
}

GLXFBConfig glXGetFBConfigFromVisualSGIX(Display *dpy, XVisualInfo *visual)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !visual || visual->visualid != ds->visualid)
        return NULL;
    return &ds->configs[0];
}

/*
 * GLX 1.3: every FBConfig of the screen (no attribute filtering).  The table
 * is already sorted best-first (mglx_config_compare order), so the array is
 * returned in table order like glXChooseFBConfig(NULL).
 */
GLXFBConfig *glXGetFBConfigs(Display *dpy, int screen, int *nelements)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    GLXFBConfig *out;

    if (!nelements)
        return NULL;
    *nelements = 0;
    if (!ds || screen != ds->screen)
        return NULL;

    out = (GLXFBConfig *)malloc(sizeof(GLXFBConfig) * MGLX_NCONFIGS);
    if (!out)
        return NULL;
    for (int i = 0; i < MGLX_NCONFIGS; i++)
        out[i] = &ds->configs[i];
    *nelements = MGLX_NCONFIGS;
    return out;     /* freed with XFree */
}

GLXWindow glXCreateWindow(Display *dpy, GLXFBConfig config, Window window,
                          const int *attrib_list)
{
    (void)attrib_list;
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !config)
        return None;
    mglx_refresh_geometry(dpy, NULL);
    MGLXDrawableRec *d = mglx_add_drawable(ds, (XID)window, MGLX_DRAW_WINDOW, 1, 1);
    if (d)
        mglx_refresh_geometry(dpy, d);
    return (GLXWindow)window;
}

void glXDestroyWindow(Display *dpy, GLXWindow window)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d = mglx_find_drawable(ds, window);
    if (d && d->type == MGLX_DRAW_WINDOW)
    {
        mglx_destroy_drawable_images(ds, d);
        d->xid = None;
    }
}

GLXPixmap glXCreateGLXPixmap(Display *dpy, XVisualInfo *visual, Pixmap pixmap)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !visual || visual->visualid != ds->visualid)
        return None;
    MGLXDrawableRec *d = mglx_add_drawable(ds, (XID)pixmap, MGLX_DRAW_PIXMAP, 1, 1);
    if (d)
        mglx_refresh_geometry(dpy, d);
    return (GLXPixmap)pixmap;
}

GLXPixmap glXCreatePixmap(Display *dpy, GLXFBConfig config, Pixmap pixmap,
                          const int *attrib_list)
{
    (void)attrib_list;
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || !config)
        return None;
    MGLXDrawableRec *d = mglx_add_drawable(ds, (XID)pixmap, MGLX_DRAW_PIXMAP, 1, 1);
    if (d)
        mglx_refresh_geometry(dpy, d);
    return (GLXPixmap)pixmap;
}

void glXDestroyPixmap(Display *dpy, GLXPixmap pixmap)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d = mglx_find_drawable(ds, pixmap);
    if (d && d->type == MGLX_DRAW_PIXMAP)
    {
        mglx_destroy_drawable_images(ds, d);
        d->xid = None;
    }
}

void glXDestroyGLXPixmap(Display *dpy, GLXPixmap pixmap)
{
    glXDestroyPixmap(dpy, pixmap);
}

GLXPbuffer glXCreatePbuffer(Display *dpy, GLXFBConfig config,
                            const int *attrib_list)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    unsigned w = 64, h = 64;
    Bool largest = False;

    if (!ds || !config)
        return None;
    if (attrib_list)
        for (int i = 0; attrib_list[i]; i += 2)
        {
            switch (attrib_list[i])
            {
                case GLX_PBUFFER_WIDTH:  w = (unsigned)attrib_list[i + 1]; break;
                case GLX_PBUFFER_HEIGHT: h = (unsigned)attrib_list[i + 1]; break;
                case GLX_LARGEST_PBUFFER: largest = attrib_list[i + 1] != 0; break;
                default: break;
            }
        }
    if (largest)
    {
        if (w > (unsigned)config->max_pbuffer_width)  w = (unsigned)config->max_pbuffer_width;
        if (h > (unsigned)config->max_pbuffer_height) h = (unsigned)config->max_pbuffer_height;
    }
    if (!w || !h)
        return None;

    /* the pbuffer is an unmapped X window: a real drawable XID with geometry,
     * never presented to the screen */
    Window root = RootWindow(dpy, ds->screen);
    Window win = XCreateSimpleWindow(dpy, root, 0, 0, w, h, 0, 0, 0);
    if (!win)
        return None;
    XSetWindowAttributes swa;
    memset(&swa, 0, sizeof swa);
    swa.override_redirect = True;
    XChangeWindowAttributes(dpy, win, CWOverrideRedirect, &swa);
    mglx_add_drawable(ds, (XID)win, MGLX_DRAW_PBUFFER, w, h);
    return (GLXPbuffer)win;
}

void glXDestroyPbuffer(Display *dpy, GLXPbuffer pbuffer)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d = mglx_find_drawable(ds, pbuffer);
    if (d && d->type == MGLX_DRAW_PBUFFER)
    {
        mglx_destroy_drawable_images(ds, d);
        XDestroyWindow(dpy, pbuffer);
        d->xid = None;
    }
}

void glXQueryDrawable(Display *dpy, GLXDrawable drawable, int attribute,
                      unsigned int *value)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d;

    if (!value)
        return;
    *value = 0;
    if (!ds)
        return;
    d = mglx_find_drawable(ds, drawable);
    if (d)
        mglx_refresh_geometry(dpy, d);

    switch (attribute)
    {
        case GLX_WIDTH:  *value = d ? d->w : 0; break;
        case GLX_HEIGHT: *value = d ? d->h : 0; break;
        case GLX_PRESERVED_CONTENTS: *value = 1; break;
        case GLX_LARGEST_PBUFFER:    *value = 0; break;
        case GLX_FBCONFIG_ID:        *value = d ? 0x101u : 0; break;
        case GLX_SWAP_INTERVAL_EXT:  *value = 1; break;
        case GLX_MAX_SWAP_INTERVAL_EXT: *value = 1; break;
        default: break;
    }
}

GLXContext glXCreateNewContext(Display *dpy, GLXFBConfig config,
                               int render_type, GLXContext share_list,
                               Bool direct)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds)
        return NULL;
    if (render_type && render_type != GLX_RGBA_TYPE)
    {
        mglx_send_error(dpy, GLX_BAD_ATTRIBUTE, None, 0);
        return NULL;
    }
    /* legacy entry point: compatibility profile (X11 apps are legacy by
     * definition); share_list accepted but not implemented (gap) */
    return mglx_create_context(dpy, config, 1, GLX_RGBA_TYPE);
}

int glXQueryContext(Display *dpy, GLXContext ctx, int attribute, int *value)
{
    if (!ctx || !value)
        return GLX_BAD_CONTEXT;
    switch (attribute)
    {
        case GLX_FBCONFIG_ID: *value = ctx->cfg ? ctx->cfg->fbconfig_id : 0x101; break;
        case GLX_RENDER_TYPE: *value = ctx->render_type; break;
        case GLX_SCREEN:      *value = ctx->screen; break;
        default: return GLX_BAD_ATTRIBUTE;
    }
    return 0;
}

void glXSelectEvent(Display *dpy, GLXDrawable drawable, unsigned long mask)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d = mglx_find_drawable(ds, drawable);
    if (d)
        d->event_mask = mask;
}

void glXGetSelectedEvent(Display *dpy, GLXDrawable drawable,
                         unsigned long *mask)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXDrawableRec *d = mglx_find_drawable(ds, drawable);
    if (mask)
        *mask = d ? d->event_mask : 0;
}

/* ------------------------------------------------------------------ */
/* GLX_ARB_create_context                                             */

GLXContext glXCreateContextAttribsARB(Display *dpy, GLXFBConfig config,
                                      GLXContext share_list, Bool direct,
                                      const int *attrib_list)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    int major = 1, minor = 0, profile_bits = 0, flags = 0, have_profile = 0;

    if (!ds)
        return NULL;

    if (attrib_list)
        for (int i = 0; attrib_list[i]; i += 2)
        {
            switch (attrib_list[i])
            {
                case GLX_CONTEXT_MAJOR_VERSION_ARB: major = attrib_list[i + 1]; break;
                case GLX_CONTEXT_MINOR_VERSION_ARB: minor = attrib_list[i + 1]; break;
                case GLX_CONTEXT_PROFILE_MASK_ARB:  profile_bits = attrib_list[i + 1]; have_profile = 1; break;
                case GLX_CONTEXT_FLAGS_ARB:         flags = attrib_list[i + 1]; break;
                case GLX_CONTEXT_RESET_NOTIFICATION_STRATEGY_ARB: break;   /* accepted, ignored */
                default:
                    mglx_send_error(dpy, GLX_BAD_ATTRIBUTE, None, 0);
                    return NULL;
            }
        }

    if (major < 1 || major > 4 || (major == 4 && minor > 6))
    {
        mglx_send_error(dpy, 13 /* GLXBadFBConfig */, None, 0);
        return NULL;
    }

    int compat;
    if (!have_profile)
        /* GLX_ARB_create_context_profile: absent mask defaults to core for
         * 3.2+ requests, compatibility below (where core does not exist) */
        compat = (major < 3 || (major == 3 && minor < 2));
    else if (profile_bits & GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB)
        compat = 1;
    else if (profile_bits & GLX_CONTEXT_CORE_PROFILE_BIT_ARB)
    {
        if (major < 3 || (major == 3 && minor < 2))
        {
            mglx_send_error(dpy, 13 /* GLXBadFBConfig */, None, 0);
            return NULL;
        }
        compat = 0;
    }
    else
    {
        mglx_send_error(dpy, 13 /* GLXBadFBConfig */, None, 0);
        return NULL;
    }

    /* forward-compatible flag is meaningless on a compatibility context */
    if ((flags & GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB) && compat)
    {
        mglx_send_error(dpy, 13 /* GLXBadFBConfig */, None, 0);
        return NULL;
    }

    return mglx_create_context(dpy, config, compat, GLX_RGBA_TYPE);
}

/* ------------------------------------------------------------------ */
/* swap-control extensions (interval is fixed at 1: one present per
 * glXSwapBuffers; the state is real so glXQueryDrawable reports it) */

int glXSwapIntervalSGI(int interval)
{
    return interval >= 0 ? 0 : GLX_BAD_VALUE;
}

void glXSwapIntervalEXT(Display *dpy, GLXDrawable drawable, int interval)
{
    (void)dpy; (void)drawable; (void)interval;
}

int glXSwapIntervalMESA(unsigned int interval)
{
    (void)interval;
    return 0;
}

int glXGetSwapIntervalMESA(void)
{
    return 1;
}

/* ------------------------------------------------------------------ */
/* GLX_SGIX_fbconfig / GLX_SGIX_pbuffer / GLX_SGI_make_current_read      */
/* aliases: X11 apps link these symbol names directly (not through       */
/* glXGetProcAddress), so they must exist as exported functions; the     */
/* semantics are the modern GLX 1.3 entry points they were folded into.  */

GLXFBConfig *glXChooseFBConfigSGIX(Display *dpy, int screen,
                                   const int *attrib_list, int *nelements)
{
    return glXChooseFBConfig(dpy, screen, attrib_list, nelements);
}

int glXGetFBConfigAttribSGIX(Display *dpy, GLXFBConfig config,
                             int attribute, int *value)
{
    return glXGetFBConfigAttrib(dpy, config, attribute, value);
}

XVisualInfo *glXGetVisualFromFBConfigSGIX(Display *dpy, GLXFBConfig config)
{
    return glXGetVisualFromFBConfig(dpy, config);
}

GLXContext glXCreateContextWithConfigSGIX(Display *dpy, GLXFBConfig config,
                                          int render_type,
                                          GLXContext share_list, Bool direct)
{
    return glXCreateNewContext(dpy, config, render_type, share_list, direct);
}

GLXPixmap glXCreateGLXPixmapWithConfigSGIX(Display *dpy, GLXFBConfig config,
                                           Pixmap pixmap)
{
    return glXCreatePixmap(dpy, config, pixmap, NULL);
}

void glXDestroyGLXPbufferSGIX(Display *dpy, GLXPbuffer pbuffer)
{
    glXDestroyPbuffer(dpy, pbuffer);
}

void glXSelectEventSGIX(Display *dpy, GLXDrawable drawable, unsigned long mask)
{
    glXSelectEvent(dpy, drawable, mask);
}

void glXGetSelectedEventSGIX(Display *dpy, GLXDrawable drawable,
                             unsigned long *mask)
{
    glXGetSelectedEvent(dpy, drawable, mask);
}

Bool glXMakeCurrentReadSGI(Display *dpy, GLXDrawable draw, GLXDrawable read,
                           GLXContext ctx)
{
    return glXMakeContextCurrent(dpy, draw, read, ctx);
}

int glXQueryContextInfoEXT(Display *dpy, GLXContext ctx, int attribute,
                           int *value)
{
    return glXQueryContext(dpy, ctx, attribute, value);
}

/* ------------------------------------------------------------------ */
/* GLX_EXT_import_context                                              */

XID glXGetContextIDEXT(const GLXContext ctx)
{
    /* the context pointer IS the XID in this client-side implementation */
    return (XID)(uintptr_t)ctx;
}

GLXContext glXImportContextEXT(Display *dpy, XID context_id)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    MGLXContext c = mglx_find_context_by_xid(context_id);
    if (!c || !ds)
    {
        mglx_send_error(dpy, GLX_BAD_CONTEXT, context_id, 0);
        return NULL;
    }
    return c;
}

void glXFreeContextEXT(Display *dpy, GLXContext ctx)
{
    glXDestroyContext(dpy, ctx);
}

Display *glXGetCurrentDisplayEXT(void)
{
    return glXGetCurrentDisplay();
}

/* ------------------------------------------------------------------ */
/* GLX_MESA_query_renderer                                             */

#define GLX_RENDERER_VENDOR_ID_MESA              0x8183
#define GLX_RENDERER_DEVICE_ID_MESA              0x8184
#define GLX_RENDERER_VERSION_MESA                0x8185
#define GLX_RENDERER_ACCELERATED_MESA            0x8186
#define GLX_RENDERER_VIDEO_MEMORY_MESA           0x8187
#define GLX_RENDERER_UNIFIED_MEMORY_ARCHITECTURE_MESA 0x8188
#define GLX_RENDERER_PREFERRED_PROFILE_MESA      0x8189
#define GLX_RENDERER_OPENGL_CORE_PROFILE_VERSION_MESA        0x818A
#define GLX_RENDERER_OPENGL_COMPATIBILITY_PROFILE_VERSION_MESA 0x818B
#define GLX_RENDERER_OPENGL_ES_PROFILE_VERSION_MESA          0x818C
#define GLX_RENDERER_OPENGL_ES2_PROFILE_VERSION_MESA         0x818D

/* the GL renderer strings only exist while a context is current; cache copies
 * so the non-current entry points can answer without one */
static struct {
    int   valid;
    char  vendor[128], renderer[128], version[128];
} s_renderer_strings;

static void mglx_refresh_renderer_strings(void)
{
    if (!s_current.ctx)
        return;
    const struct { GLenum e; char *dst; size_t n; } q[3] = {
        { GL_VENDOR,   s_renderer_strings.vendor,   sizeof s_renderer_strings.vendor },
        { GL_RENDERER, s_renderer_strings.renderer, sizeof s_renderer_strings.renderer },
        { GL_VERSION,  s_renderer_strings.version,  sizeof s_renderer_strings.version },
    };
    for (int i = 0; i < 3; i++)
    {
        const GLubyte *s = glGetString(q[i].e);
        while (glGetError() != GL_NO_ERROR) {}
        if (!s)
            return;
        snprintf(q[i].dst, q[i].n, "%s", (const char *)s);
    }
    s_renderer_strings.valid = 1;
}

static Bool mglx_query_renderer_integer(int attribute, unsigned int *value)
{
    if (!value)
        return False;

    switch (attribute)
    {
        case GLX_RENDERER_VENDOR_ID_MESA:
        case GLX_RENDERER_DEVICE_ID_MESA:
            /* no PCI ids behind an MTLDevice on Apple silicon: unknown, and
             * reported as such rather than guessed */
            value[0] = 0;
            return True;
        case GLX_RENDERER_VERSION_MESA:
            value[0] = 4; value[1] = 6; value[2] = 0;
            return True;
        case GLX_RENDERER_ACCELERATED_MESA:
            value[0] = 1;      /* Metal: always GPU-rendered */
            return True;
        case GLX_RENDERER_VIDEO_MEMORY_MESA:
            /* on unified memory (Apple silicon) the GPU shares system memory:
             * report the machine's RAM, the number Metal's working-set limits
             * are derived from.  Discrete GPUs: not reported (unknown). */
            mglx_refresh_renderer_strings();
            if (s_renderer_strings.valid &&
                strncmp(s_renderer_strings.renderer, "Apple", 5) == 0)
            {
                uint64_t ram = 0;
                size_t sz = sizeof ram;
                if (sysctlbyname("hw.memsize", &ram, &sz, NULL, 0) == 0 &&
                    ram >= 1024 * 1024)
                {
                    value[0] = (unsigned int)(ram / (1024 * 1024));  /* MiB */
                    return True;
                }
            }
            return False;
        case GLX_RENDERER_OPENGL_CORE_PROFILE_VERSION_MESA:
        case GLX_RENDERER_OPENGL_COMPATIBILITY_PROFILE_VERSION_MESA:
            value[0] = 4; value[1] = 6; value[2] = 0;
            return True;
        case GLX_RENDERER_OPENGL_ES_PROFILE_VERSION_MESA:
        case GLX_RENDERER_OPENGL_ES2_PROFILE_VERSION_MESA:
            value[0] = 0; value[1] = 0; value[2] = 0;   /* no GLES contexts */
            return True;
        case GLX_RENDERER_UNIFIED_MEMORY_ARCHITECTURE_MESA:
            /* Apple-silicon GPUs (renderer "Apple ...") share system memory;
             * discrete GPUs in older Macs do not */
            mglx_refresh_renderer_strings();
            value[0] = s_renderer_strings.valid &&
                       strncmp(s_renderer_strings.renderer, "Apple", 5) == 0;
            return True;
        case GLX_RENDERER_PREFERRED_PROFILE_MESA:
            /* the GLX shim hands X11 (legacy) apps compatibility contexts */
            value[0] = GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB;
            return True;
        default:
            return False;
    }
}

static const char *mglx_query_renderer_string(int attribute)
{
    mglx_refresh_renderer_strings();
    if (!s_renderer_strings.valid)
        return NULL;
    switch (attribute)
    {
        case GLX_RENDERER_VENDOR_ID_MESA: return s_renderer_strings.vendor;
        case GLX_RENDERER_DEVICE_ID_MESA: return s_renderer_strings.renderer;
        case GLX_RENDERER_VERSION_MESA:   return s_renderer_strings.version;
        default: return NULL;
    }
}

Bool glXQueryRendererIntegerMESA(Display *dpy, int screen, int renderer,
                                 int attribute, unsigned int *value)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || screen != ds->screen || renderer != 0)
        return False;
    return mglx_query_renderer_integer(attribute, value);
}

Bool glXQueryCurrentRendererIntegerMESA(int attribute, unsigned int *value)
{
    if (!s_current.ctx)
        return False;
    return mglx_query_renderer_integer(attribute, value);
}

const char *glXQueryRendererStringMESA(Display *dpy, int screen, int renderer,
                                       int attribute)
{
    MGLXDisplayRec *ds = mglx_display(dpy);
    if (!ds || screen != ds->screen || renderer != 0)
        return NULL;
    return mglx_query_renderer_string(attribute);
}

const char *glXQueryCurrentRendererStringMESA(int attribute)
{
    if (!s_current.ctx)
        return NULL;
    return mglx_query_renderer_string(attribute);
}

/* ------------------------------------------------------------------ */
/* glXGetProcAddress                                                   */

void (*glXGetProcAddress(const GLubyte *procname))(void);
void (*glXGetProcAddressARB(const GLubyte *procname))(void);
void glXCopyContext(Display *dpy, GLXContext src, GLXContext dst,
                    unsigned long mask);
/* SGIX/EXT/MESA entry points defined further down in this file */
GLXFBConfig *glXGetFBConfigs(Display *dpy, int screen, int *nelements);
GLXFBConfig *glXChooseFBConfigSGIX(Display *dpy, int screen,
                                   const int *attrib_list, int *nelements);
int glXGetFBConfigAttribSGIX(Display *dpy, GLXFBConfig config,
                             int attribute, int *value);
XVisualInfo *glXGetVisualFromFBConfigSGIX(Display *dpy, GLXFBConfig config);
GLXContext glXCreateContextWithConfigSGIX(Display *dpy, GLXFBConfig config,
                                          int render_type,
                                          GLXContext share_list, Bool direct);
GLXPixmap glXCreateGLXPixmapWithConfigSGIX(Display *dpy, GLXFBConfig config,
                                           Pixmap pixmap);
void glXDestroyGLXPbufferSGIX(Display *dpy, GLXPbuffer pbuffer);
void glXSelectEventSGIX(Display *dpy, GLXDrawable drawable, unsigned long mask);
void glXGetSelectedEventSGIX(Display *dpy, GLXDrawable drawable,
                             unsigned long *mask);
Bool glXMakeCurrentReadSGI(Display *dpy, GLXDrawable draw, GLXDrawable read,
                           GLXContext ctx);
int glXQueryContextInfoEXT(Display *dpy, GLXContext ctx, int attribute,
                           int *value);
XID glXGetContextIDEXT(const GLXContext ctx);
GLXContext glXImportContextEXT(Display *dpy, XID context_id);
void glXFreeContextEXT(Display *dpy, GLXContext ctx);
Display *glXGetCurrentDisplayEXT(void);
Bool glXQueryRendererIntegerMESA(Display *dpy, int screen, int renderer,
                                 int attribute, unsigned int *value);
Bool glXQueryCurrentRendererIntegerMESA(int attribute, unsigned int *value);
const char *glXQueryRendererStringMESA(Display *dpy, int screen, int renderer,
                                       int attribute);
const char *glXQueryCurrentRendererStringMESA(int attribute);

static const struct { const char *name; void *fn; } mglx_procs[] = {
    { "glXChooseVisual", (void *)glXChooseVisual },
    { "glXGetConfig", (void *)glXGetConfig },
    { "glXCopyContext", (void *)glXCopyContext },
    { "glXCreateContext", (void *)glXCreateContext },
    { "glXCreateNewContext", (void *)glXCreateNewContext },
    { "glXCreateContextAttribsARB", (void *)glXCreateContextAttribsARB },
    { "glXDestroyContext", (void *)glXDestroyContext },
    { "glXMakeCurrent", (void *)glXMakeCurrent },
    { "glXMakeContextCurrent", (void *)glXMakeContextCurrent },
    { "glXGetCurrentContext", (void *)glXGetCurrentContext },
    { "glXGetCurrentDrawable", (void *)glXGetCurrentDrawable },
    { "glXGetCurrentReadDrawable", (void *)glXGetCurrentReadDrawable },
    { "glXGetCurrentDisplay", (void *)glXGetCurrentDisplay },
    { "glXQueryExtension", (void *)glXQueryExtension },
    { "glXQueryVersion", (void *)glXQueryVersion },
    { "glXQueryContext", (void *)glXQueryContext },
    { "glXQueryDrawable", (void *)glXQueryDrawable },
    { "glXIsDirect", (void *)glXIsDirect },
    { "glXSwapBuffers", (void *)glXSwapBuffers },
    { "glXWaitGL", (void *)glXWaitGL },
    { "glXWaitX", (void *)glXWaitX },
    { "glXUseXFont", (void *)glXUseXFont },
    { "glXQueryExtensionsString", (void *)glXQueryExtensionsString },
    { "glXQueryServerString", (void *)glXQueryServerString },
    { "glXGetClientString", (void *)glXGetClientString },
    { "glXGetProcAddress", (void *)glXGetProcAddress },
    { "glXGetProcAddressARB", (void *)glXGetProcAddressARB },
    { "glXChooseFBConfig", (void *)glXChooseFBConfig },
    { "glXChooseFBConfigSGIX", (void *)glXChooseFBConfig },
    { "glXGetFBConfigs", (void *)glXGetFBConfigs },
    { "glXGetFBConfigAttrib", (void *)glXGetFBConfigAttrib },
    { "glXGetFBConfigAttribSGIX", (void *)glXGetFBConfigAttrib },
    { "glXGetFBConfigFromVisualSGIX", (void *)glXGetFBConfigFromVisualSGIX },
    { "glXGetVisualFromFBConfig", (void *)glXGetVisualFromFBConfig },
    { "glXGetVisualFromFBConfigSGIX", (void *)glXGetVisualFromFBConfig },
    { "glXCreateContextWithConfigSGIX", (void *)glXCreateContextWithConfigSGIX },
    { "glXCreateGLXPixmapWithConfigSGIX", (void *)glXCreateGLXPixmapWithConfigSGIX },
    { "glXDestroyGLXPbufferSGIX", (void *)glXDestroyGLXPbufferSGIX },
    { "glXCreateWindow", (void *)glXCreateWindow },
    { "glXDestroyWindow", (void *)glXDestroyWindow },
    { "glXCreatePixmap", (void *)glXCreatePixmap },
    { "glXDestroyPixmap", (void *)glXDestroyPixmap },
    { "glXCreateGLXPixmap", (void *)glXCreateGLXPixmap },
    { "glXDestroyGLXPixmap", (void *)glXDestroyGLXPixmap },
    { "glXCreatePbuffer", (void *)glXCreatePbuffer },
    { "glXDestroyPbuffer", (void *)glXDestroyPbuffer },
    { "glXSelectEvent", (void *)glXSelectEvent },
    { "glXGetSelectedEvent", (void *)glXGetSelectedEvent },
    { "glXSelectEventSGIX", (void *)glXSelectEventSGIX },
    { "glXGetSelectedEventSGIX", (void *)glXGetSelectedEventSGIX },
    { "glXMakeCurrentReadSGI", (void *)glXMakeCurrentReadSGI },
    { "glXQueryContextInfoEXT", (void *)glXQueryContextInfoEXT },
    { "glXGetContextIDEXT", (void *)glXGetContextIDEXT },
    { "glXImportContextEXT", (void *)glXImportContextEXT },
    { "glXFreeContextEXT", (void *)glXFreeContextEXT },
    { "glXGetCurrentDisplayEXT", (void *)glXGetCurrentDisplayEXT },
    { "glXQueryRendererIntegerMESA", (void *)glXQueryRendererIntegerMESA },
    { "glXQueryCurrentRendererIntegerMESA", (void *)glXQueryCurrentRendererIntegerMESA },
    { "glXQueryRendererStringMESA", (void *)glXQueryRendererStringMESA },
    { "glXQueryCurrentRendererStringMESA", (void *)glXQueryCurrentRendererStringMESA },
    { "glXSwapIntervalSGI", (void *)glXSwapIntervalSGI },
    { "glXSwapIntervalEXT", (void *)glXSwapIntervalEXT },
    { "glXSwapIntervalMESA", (void *)glXSwapIntervalMESA },
    { "glXGetSwapIntervalMESA", (void *)glXGetSwapIntervalMESA },
    { NULL, NULL }
};

void (*glXGetProcAddressARB(const GLubyte *procname))(void)
{
    if (!procname)
        return NULL;
    for (int i = 0; mglx_procs[i].name; i++)
        if (strcmp((const char *)procname, mglx_procs[i].name) == 0)
            return (void (*)(void))mglx_procs[i].fn;
    /* not one of ours: the GL entry points are re-exported from libmgl, so a
     * global lookup finds them (RTLD_DEFAULT includes our dependencies) */
    return (void (*)(void))dlsym(RTLD_DEFAULT, (const char *)procname);
}

void (*glXGetProcAddress(const GLubyte *procname))(void)
{
    return glXGetProcAddressARB(procname);
}

/*
 * glXCopyContext: copies GL state between contexts.  MGL keeps GL state
 * private inside a context, so a true copy is not possible; per the GLX 1.4
 * spec the destination context must not be current — we honour that and
 * report the limitation.  (No X11 app in practice uses glXCopyContext.)
 */
void glXCopyContext(Display *dpy, GLXContext src, GLXContext dst,
                    unsigned long mask)
{
    (void)mask;
    if (!src || !dst || dst == s_current.ctx)
        mglx_send_error(dpy, GLX_BAD_CONTEXT, (XID)(uintptr_t)dst, 0);
    /* documented gap: no state copy */
}
