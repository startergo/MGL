/*
 * mgl_nsgl.m — NSOpenGL interposer that runs NSOpenGL apps (glmark2-macos and
 * friends) on MGL.  It is loaded with DYLD_INSERT_LIBRARIES:
 *
 *   make nsgl     ->  build/nsgl/libmgl_nsgl.dylib
 *   DYLD_INSERT_LIBRARIES=<abs>/build/nsgl/libmgl_nsgl.dylib ./glmark2-macos ...
 *
 * How it works:
 *  1. DYLD_INTERPOSE on dlopen/dlsym: a dlopen of
 *     /System/Library/Frameworks/OpenGL.framework returns libmgl.dylib, so
 *     glad-style loaders resolve every gl* entry point to MGL (a full 4.6 +
 *     compatibility implementation); RTLD_DEFAULT gl* lookups also land in
 *     MGL through this library.
 *  2. Method swizzling of NSOpenGLPixelFormat -initWithAttributes: and the
 *     NSOpenGLContext lifecycle: an NSOpenGLContext becomes an MGL context
 *     (compatibility profile for NSOpenGLProfileVersionLegacy or a request
 *     without a profile attribute; core for 3_2Core/4_1Core).  The original
 *     NSOpenGLContext initializer is never called, so no CGL context is
 *     created; -dealloc chains to the original with zeroed ivars.
 *  3. Rendering happens in MGL's headless default framebuffer, sized to the
 *     view (setView:/update set the viewport to the view size; the union of
 *     all viewports never shrinks, exactly like the GLX shim).  flushBuffer
 *     reads the framebuffer back (BGRA) and blits it into the view's layer.
 *
 * Licensed under the Apache License, Version 2.0 (see the repository LICENSE).
 */
#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CoreGraphics.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach-o/dyld.h>   /* _NSGetExecutablePath */
#import <pthread.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

#define GL_GLEXT_PROTOTYPES 1
#import "glcorearb.h"
#import "gl_compat.h"

/* MGL exports (declared locally: MGLContext.h has C++ default arguments) */
extern void *createGLMContextProfile(GLenum format, GLenum type,
                                     GLenum depth_format, GLenum depth_type,
                                     GLenum stencil_format, GLenum stencil_type,
                                     GLenum profile);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);
extern void MGLsetCurrentContext(void *ctx);
extern void *MGLgetCurrentContext(void);
extern void MGLswapBuffers(void *ctx);

/* MGLRenderer lives in libmgl (no public header): the windowed bind is what
 * the GLFW platform uses — it creates the CAMetalLayer on the view and makes
 * MGLswapBuffers present the drawable (zero-copy present path) */
@protocol MGLRendererBind <NSObject>
- (void)createMGLRendererAndBindToContext:(void *)glm_ctx view:(NSView *)view;
@end

#ifndef GL_CONTEXT_CORE_PROFILE_BIT
#define GL_CONTEXT_CORE_PROFILE_BIT          0x00000001
#define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT 0x00000002
#endif

/* ------------------------------------------------------------------ */
/* dlopen/dlsym interposition                                          */

static void *s_mgl_handle;    /* real dlopen handle of libmgl.dylib */

static void nsgl_find_mgl(void)
{
    if (s_mgl_handle)
        return;
    /* libmgl sits next to this interposer (build/nsgl/libmgl.dylib) or is
     * already loaded as our dependency */
    Dl_info info;
    if (dladdr((const void *)&nsgl_find_mgl, &info) && info.dli_fname)
    {
        char path[PATH_MAX];
        snprintf(path, sizeof path, "%s", info.dli_fname);
        char *slash = strrchr(path, '/');
        if (slash)
        {
            strcpy(slash + 1, "libmgl.dylib");
            s_mgl_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
            if (s_mgl_handle)
                return;
            s_mgl_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (s_mgl_handle)
                return;
        }
    }
    s_mgl_handle = dlopen("build/libmgl.dylib", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (!s_mgl_handle)
        s_mgl_handle = dlopen("libmgl.dylib", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
}

/* only the APPLICATION's own dlopen of OpenGL.framework should land in MGL;
 * AppKit and libmgl itself must keep using the real framework (libmgl probes
 * CGL for its default limits, AppKit's NSOpenGLPixelFormat init resolves
 * private libGL symbols like gliInitializeLibrary from it) */
static char s_exec_path[PATH_MAX];   /* canonical path of the main executable */

static void nsgl_cache_exec_path(void)
{
    char buf[PATH_MAX];
    uint32_t size = sizeof buf;
    if (s_exec_path[0])
        return;
    if (_NSGetExecutablePath(buf, &size) == 0)
    {
        char *rp = realpath(buf, NULL);
        if (rp)
        {
            snprintf(s_exec_path, sizeof s_exec_path, "%s", rp);
            free(rp);
        }
    }
}

static int nsgl_caller_is_main_from(void *retaddr)
{
    Dl_info info;
    if (!s_exec_path[0])
        return 0;
    if (!dladdr(retaddr, &info) || !info.dli_fname)
        return 0;
    char *rp = realpath(info.dli_fname, NULL);
    if (!rp)
        return 0;
    int is_main = strcmp(rp, s_exec_path) == 0;
    free(rp);
    return is_main;
}

static void *nsgl_dlopen(const char *path, int mode)
{
    if (path && strstr(path, "OpenGL.framework"))
    {
        if (nsgl_caller_is_main_from(__builtin_return_address(0)))
        {
            nsgl_find_mgl();
            if (s_mgl_handle)
                return s_mgl_handle;    /* the app now believes this is OpenGL */
        }
    }
    return dlopen(path, mode);
}

static void *nsgl_dlsym(void *handle, const char *name)
{
    /* the application's own GL symbol lookups go to MGL: this catches the
     * glad-style loaders that probe RTLD_DEFAULT (Apple's real GL is only
     * 4.1 and would fail their version checks); libraries' internal lookups
     * (AppKit, libmgl itself) are left untouched */
    if (name && strncmp(name, "gl", 2) == 0 && strlen(name) > 3 &&
        nsgl_caller_is_main_from(__builtin_return_address(0)))
    {
        nsgl_find_mgl();
        if (s_mgl_handle)
        {
            void *sym = dlsym(s_mgl_handle, name);
            if (sym)
                return sym;
        }
    }
    return dlsym(handle, name);
}

#define DYLD_INTERPOSE(_replacement, _replacee)                              \
    __attribute__((used)) static struct {                                    \
        const void *replacement; const void *replacee;                      \
    } _interpose_##_replacee                                                 \
    __attribute__((section("__DATA,__interpose"))) = {                      \
        (const void *)(unsigned long)&_replacement,                          \
        (const void *)(unsigned long)&_replacee };

DYLD_INTERPOSE(nsgl_dlopen, dlopen)
DYLD_INTERPOSE(nsgl_dlsym, dlsym)

/* ------------------------------------------------------------------ */
/* per-object state: boxed in NSMutableData / NSData via associated
 * objects (associated objects must be real objects, the structs are not) */

typedef struct {
    int compat;
    int double_buffered;
    int depth, stencil, samples, alpha, color;
} NSGLPixelFormatInfo;

typedef struct {
    void *glm;
    int is_compat;
    __unsafe_unretained NSView *view;
    unsigned view_w, view_h;
    unsigned fb_w, fb_h;       /* union of viewports (readback clamp) */
    int swap_interval;
    int renderer_bound;        /* 1 once a renderer exists for glm */
    int windowed;              /* 1 when bound to the view via MGLRenderer */
    unsigned char *read_buf;
    size_t read_buf_size;
    int param_values[512];     /* setValues params recorded for getValues */
} NSGLState;

static const void *kNSGLPFKey = &kNSGLPFKey;
static const void *kNSGLCtxKey = &kNSGLCtxKey;

static NSGLPixelFormatInfo *nsgl_pf_info(NSOpenGLPixelFormat *pf)
{
    NSData *d = objc_getAssociatedObject(pf, kNSGLPFKey);
    return d ? (NSGLPixelFormatInfo *)[d bytes] : NULL;
}

static NSGLState *nsgl_ctx_state(NSOpenGLContext *ctx)
{
    NSMutableData *d = objc_getAssociatedObject(ctx, kNSGLCtxKey);
    return d ? (NSGLState *)[d mutableBytes] : NULL;
}

static const void *kNSGLRendererKey = &kNSGLRendererKey;

/* the renderer object is retained as an associated object of the context so
 * it outlives the bind call (MGL keeps no list of renderers) */
/* vsync: interval 0 turns the layer's display sync off (see the swap-interval
 * parameter notes at nsgl_ctx_setValues) */
static void nsgl_apply_swap_interval(NSGLState *st)
{
    if (st->view)
    {
        CALayer *layer = [st->view layer];
        if ([layer isKindOfClass:[CAMetalLayer class]])
            ((CAMetalLayer *)layer).displaySyncEnabled = (st->swap_interval != 0);
    }
}

static void nsgl_bind_windowed_renderer(NSOpenGLContext *self, NSGLState *st, NSView *view)
{
    id<MGLRendererBind> r = (id<MGLRendererBind>)[[objc_getClass("MGLRenderer") alloc] init];
    if (!r)
        return;
    [r createMGLRendererAndBindToContext:st->glm view:view];
    objc_setAssociatedObject(self, kNSGLRendererKey, (id)r, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    st->renderer_bound = 1;
    st->windowed = 1;

    nsgl_apply_swap_interval(st);
}

static void nsgl_bind_headless_renderer(NSGLState *st)
{
    if (st->renderer_bound)
        return;
    CppCreateMGLRendererHeadless(st->glm);
    st->renderer_bound = 1;
    st->windowed = 0;
}

static void *s_init_ctx;                  /* libmgl's auto core context */
static __unsafe_unretained NSOpenGLContext *s_current_nsgl;   /* what +currentContext reports; not retained: storing/overwriting it must not release (and thereby dealloc) a context while s_lock is held */
static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;

/* original implementations, chained where the AppKit object must stay valid */
static IMP s_pf_orig_init;
static IMP s_ctx_orig_init;
static IMP s_ctx_orig_dealloc;

/* AppKit pixel-format attribute tags (values are public and stable) */
enum {
    kNSGL_PFADoubleBuffer = 5, kNSGL_PFAColorSize = 8, kNSGL_PFAAlphaSize = 11,
    kNSGL_PFADepthSize = 12, kNSGL_PFAStencilSize = 13, kNSGL_PFAAccumSize = 14,
    kNSGL_PFAMinimumPolicy = 51, kNSGL_PFAMaximumPolicy = 52,
    kNSGL_PFAOffScreen = 53, kNSGL_PFASampleBuffers = 55, kNSGL_PFASamples = 56,
    kNSGL_PFARendererID = 70, kNSGL_PFAOpenGLProfile = 99
};
enum {
    kNSGL_ProfileLegacy = 0x1000,
    kNSGL_Profile3_2Core = 0x3200,
    kNSGL_Profile4_1Core = 0x4100
};

/* ------------------------------------------------------------------ */
/* NSOpenGLPixelFormat                                                 */

static id nsgl_pf_initWithAttributes(NSOpenGLPixelFormat *self, SEL _cmd,
                                     const NSOpenGLPixelFormatAttribute *attrs)
{
    /* remember the requested profile: Legacy (or no profile attribute at all)
     * means the app wants compatibility GL; 3_2Core/4_1Core means core.
     * Everything else (depth/stencil/samples/accum) is provided by MGL
     * unconditionally, so a request is never rejected. */
    NSGLPixelFormatInfo info;
    memset((void *)&info, 0, sizeof info);
    info.compat = 1;
    if (attrs)
        for (int i = 0; attrs[i]; i++)
        {
            int attr = (int)attrs[i];
            switch (attr)
            {
                case kNSGL_PFAOpenGLProfile:
                    info.compat = ((int)attrs[i + 1] == kNSGL_ProfileLegacy);
                    i++;    /* boolean attributes carry no value */
                    break;
                case kNSGL_PFADoubleBuffer:
                    info.double_buffered = 1;
                    break;
                case kNSGL_PFADepthSize:   info.depth = (int)attrs[i + 1]; i++; break;
                case kNSGL_PFAStencilSize: info.stencil = (int)attrs[i + 1]; i++; break;
                case kNSGL_PFASamples:     info.samples = (int)attrs[i + 1]; i++; break;
                case kNSGL_PFAAlphaSize:   info.alpha = (int)attrs[i + 1]; i++; break;
                case kNSGL_PFAColorSize:   info.color = (int)attrs[i + 1]; i++; break;
                default: break;
            }
        }
    objc_setAssociatedObject(self, kNSGLPFKey,
                             [NSData dataWithBytes:&info length:sizeof info],
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    /* let AppKit build its own pixel format too (harmless, keeps any later
     * -getValues:forAttribute: honest); if the attributes are one of the
     * combinations modern CGL rejects, keep the object anyway — MGL will
     * back the context regardless */
    if (s_pf_orig_init)
    {
        id r = ((id (*)(id, SEL, const NSOpenGLPixelFormatAttribute *))s_pf_orig_init)
            (self, _cmd, attrs);
        if (r)
            return r;
    }
    /* the original rejected the attributes: our object has NO CGL pixel
     * format, so -getValues:forAttribute: (if the app uses it) cannot work —
     * neutralize the query methods too */
    return self;
}

/* ------------------------------------------------------------------ */
/* NSOpenGLContext                                                     */

static id nsgl_ctx_initWithFormat(NSOpenGLContext *self, SEL _cmd,
                                  NSOpenGLPixelFormat *format,
                                  NSOpenGLContext *share)
{
    NSGLPixelFormatInfo *info = nsgl_pf_info(format);
    NSGLState st = { 0 };
    for (int i = 0; i < 512; i++) st.param_values[i] = -1;

    (void)share;    /* share lists not implemented (documented gap) */
    st.is_compat = info ? info->compat : 1;
    st.swap_interval = 1;

    st.glm = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                     GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                     GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                     st.is_compat ? GL_CONTEXT_COMPATIBILITY_PROFILE_BIT
                                                  : GL_CONTEXT_CORE_PROFILE_BIT);
    if (!st.glm)
        return nil;
    /* the renderer is bound lazily: to the view when one is attached (the
     * windowed Metal path, like GLFW), headless as a fallback for GL calls
     * that happen before any setView: */

    objc_setAssociatedObject(self, kNSGLCtxKey,
                             [NSMutableData dataWithBytes:&st length:sizeof st],
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    /* deliberately no call to the original initializer: it would create a
     * real CGL context we never use */
    return self;
}

static SEL nsgl_dealloc_sel(void) { return sel_registerName("dealloc"); }

/* __unsafe_unretained: chaining to the original dealloc frees the object;
 * an ARC-owned strong 'self' would release it a second time on return */
static void nsgl_ctx_dealloc(NSOpenGLContext *__unsafe_unretained self, SEL _cmd)
{
    NSGLState *st = nsgl_ctx_state(self);
    if (st)
    {
        pthread_mutex_lock(&s_lock);
        if (s_current_nsgl == self)
        {
            s_current_nsgl = nil;
            MGLsetCurrentContext(s_init_ctx);
        }
        pthread_mutex_unlock(&s_lock);
        free(st->read_buf);
        st->read_buf = NULL;
    }
    objc_setAssociatedObject(self, kNSGLCtxKey, nil, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    /* chain to the original dealloc: ivars are zeroed (our init never ran),
     * so it only releases the (nil) CGL context and the object itself.
     * Called through the saved IMP: objc_msgSend would recurse. */
    if (s_ctx_orig_dealloc)
        ((void (*)(id, SEL))s_ctx_orig_dealloc)(self, nsgl_dealloc_sel());
}

/* view -> context registry for setWantsBestResolutionOpenGLSurface: tracking */
#define NSGL_MAX_VIEWS 8
static NSView *s_views[NSGL_MAX_VIEWS];
static NSOpenGLContext *s_view_ctx[NSGL_MAX_VIEWS];
static pthread_mutex_t s_view_lock = PTHREAD_MUTEX_INITIALIZER;

static void nsgl_register_view(NSView *view, NSOpenGLContext *ctx)
{
    pthread_mutex_lock(&s_view_lock);
    int slot = -1;
    for (int i = 0; i < NSGL_MAX_VIEWS; i++)
    {
        if (s_views[i] == view) { slot = i; break; }
        if (!s_views[i] && slot < 0) slot = i;
    }
    if (slot >= 0)
    {
        s_views[slot] = view;
        s_view_ctx[slot] = ctx;
    }
    pthread_mutex_unlock(&s_view_lock);
}

static void nsgl_unregister_view(NSOpenGLContext *ctx)
{
    pthread_mutex_lock(&s_view_lock);
    for (int i = 0; i < NSGL_MAX_VIEWS; i++)
        if (s_view_ctx[i] == ctx) { s_views[i] = nil; s_view_ctx[i] = nil; }
    pthread_mutex_unlock(&s_view_lock);
}

/* rescale the Metal layer of a bound context to the view's
 * wantsBestResolutionOpenGLSurface: NO (glmark2-macos) -> POINTS surface
 * (contentsScale 1), YES -> backing pixels.  Called from setView: and from the
 * swizzled NSView setter — apps flip the flag after binding, and the macOS
 * default for the property is now YES. */
static void nsgl_sync_surface_scale(NSView *view)
{
    if (!view)
        return;
    BOOL best = [view respondsToSelector:@selector(wantsBestResolutionOpenGLSurface)]
                ? view.wantsBestResolutionOpenGLSurface : NO;
    CGFloat scale = best ? (view.window ? view.window.backingScaleFactor
                                         : [[NSScreen mainScreen] backingScaleFactor])
                         : (CGFloat)1.0;
    if (scale <= 0) scale = 1.0;
    CALayer *layer = [view layer];
    if ([layer isKindOfClass:[CAMetalLayer class]] && layer.contentsScale != scale)
        layer.contentsScale = scale;    /* drives drawableSize -> default framebuffer */
}

static IMP s_view_orig_setBestRes;

/* NSView.wantsBestResolutionOpenGLSurface decides POINTS vs PIXELS for the GL
 * surface; the swizzled setter keeps a bound context's layer in sync */
static void nsgl_view_setWantsBestResolution(NSView *self, SEL _cmd, BOOL best)
{
    if (s_view_orig_setBestRes)
        ((void (*)(id, SEL, BOOL))s_view_orig_setBestRes)(
            self, @selector(setWantsBestResolutionOpenGLSurface:), best);
    nsgl_sync_surface_scale(self);
}

static void nsgl_ctx_setView(NSOpenGLContext *self, SEL _cmd, NSView *view)
{
    NSGLState *st = nsgl_ctx_state(self);
    if (!st)
        return;
    st->view = view;
    if (view)
    {
        /* bind the windowed MGL renderer: it hosts its own CAMetalLayer on
         * the view and sizes the default framebuffer from the drawable */
        nsgl_bind_windowed_renderer(self, st, view);
        nsgl_register_view(view, self);
        NSRect b = [view bounds];
        st->view_w = (unsigned)b.size.width;
        st->view_h = (unsigned)b.size.height;
        if (!st->view_w) st->view_w = 1;
        if (!st->view_h) st->view_h = 1;
    }
}



static NSView *nsgl_ctx_view(NSOpenGLContext *self, SEL _cmd)
{
    NSGLState *st = nsgl_ctx_state(self);
    return st ? st->view : nil;
}

static void nsgl_ctx_makeCurrent(NSOpenGLContext *self, SEL _cmd)
{
    NSGLState *st = nsgl_ctx_state(self);
    pthread_mutex_lock(&s_lock);
    s_current_nsgl = st ? self : nil;
    MGLsetCurrentContext(st ? st->glm : s_init_ctx);
    pthread_mutex_unlock(&s_lock);
    if (st)
        nsgl_bind_headless_renderer(st);   /* only when setView: never came */
}

static void nsgl_ctx_clearCurrent(NSOpenGLContext *self, SEL _cmd)
{
    pthread_mutex_lock(&s_lock);
    s_current_nsgl = nil;
    MGLsetCurrentContext(s_init_ctx);
    pthread_mutex_unlock(&s_lock);
}

static NSOpenGLContext *nsgl_ctx_current(NSOpenGLContext *self, SEL _cmd)
{
    return s_current_nsgl;
}

static void nsgl_ctx_update(NSOpenGLContext *self, SEL _cmd)
{
    {
        NSGLState *st = nsgl_ctx_state(self);
        if (st && st->view)
        {
            /* rescale if the app flipped wantsBestResolutionOpenGLSurface (the
             * swizzled setter already does; update re-syncs after window changes) */
            nsgl_sync_surface_scale(st->view);
        }
    }
    NSGLState *st = nsgl_ctx_state(self);
    if (!st || !st->view)
        return;
    NSRect b = [st->view bounds];
    st->view_w = (unsigned)b.size.width;
    st->view_h = (unsigned)b.size.height;
    if (!st->windowed && st->view_w && st->view_h &&
        (st->view_w > st->fb_w || st->view_h > st->fb_h))
    {
        /* headless fallback: the view grew beyond what any viewport covers */
        MGLsetCurrentContext(st->glm);
        glViewport(0, 0, (GLsizei)st->view_w, (GLsizei)st->view_h);
        if (st->view_w > st->fb_w) st->fb_w = st->view_w;
        if (st->view_h > st->fb_h) st->fb_h = st->view_h;
    }
}

/* NSOpenGLContext -setValues:forParameter: / -getValues:forParameter: parameter
 * names.  The SDK's NSOpenGLContextParameterSwapInterval is 222 (NSOpenGL.h);
 * 196 is the legacy NSOpenGLCPSwapInterval constant that older code (and code
 * compiled against the deprecated CGL-level name kCGLCPSwapInterval) still
 * passes — both must toggle the layer's display sync, or an interval-0
 * request (glmark2) is silently ignored and everything is pinned to vsync. */
#define NSGL_PARAM_SWAP_INTERVAL_LEGACY 196  /* NSOpenGLCPSwapInterval */
#define NSGL_PARAM_SWAP_INTERVAL        222  /* NSOpenGLContextParameterSwapInterval */

static void nsgl_ctx_setValues(NSOpenGLContext *self, SEL _cmd,
                               const GLint *vals, NSUInteger param)
{
    NSGLState *st = nsgl_ctx_state(self);
    if (!st || !vals)
        return;
    if (param == NSGL_PARAM_SWAP_INTERVAL || param == NSGL_PARAM_SWAP_INTERVAL_LEGACY)
    {
        st->swap_interval = *vals;
        nsgl_apply_swap_interval(st);
    }
    else if (param == 235 /* SurfaceOrder */ || param == 201 /* SwapRectangleEnable */ ||
             param == 306 /* SurfaceVolatile */)
    {
        /* compositing parameters of the real NSOpenGL surface, which the
         * interposer never creates: recorded so -getValues:forParameter:
         * reports the state back, nothing to apply */
        st->param_values[param] = *vals;
    }
    else if (param == 236 /* SurfaceOpacity */)
    {
        /* opacity IS applied: the CAMetalLayer is opaque by default (matching
         * the NSOpenGL default); SurfaceOpacity = 0 makes the compositor blend
         * the surface alpha with what is behind it */
        st->param_values[236] = *vals;
        if (st->view)
        {
            CALayer *l = [st->view layer];
            for (CALayer *c in [l.sublayers copy])
                if ([c isKindOfClass:[CAMetalLayer class]])
                    c.opaque = (*vals != 0);
            if ([l isKindOfClass:[CAMetalLayer class]])
                l.opaque = (*vals != 0);
        }
    }
    /* SurfaceBackingSize (304, 2 values) and SwapRectangle (200, 4 values):
     * recorded without effect for the same reason */
    else if ((param == 304 || param == 200) && vals[0] >= 0)
    {
        st->param_values[param] = vals[0];
    }
}

static void nsgl_ctx_getValues(NSOpenGLContext *self, SEL _cmd,
                               GLint *vals, NSUInteger param)
{
    NSGLState *st = nsgl_ctx_state(self);
    if (!st || !vals)
        return;
    if (param == NSGL_PARAM_SWAP_INTERVAL || param == NSGL_PARAM_SWAP_INTERVAL_LEGACY)
        *vals = st->swap_interval;
    else if (param == 235 /* SurfaceOrder */)
        *vals = st->param_values[235] != -1 ? st->param_values[235] : 1;
    else if (param == 236 /* SurfaceOpacity */)
        *vals = st->param_values[236] != -1 ? st->param_values[236] : 1;
    else if (param == 314 /* HasDrawable */)
        *vals = (st->view != nil);
    else if (param == 304 /* SurfaceBackingSize */ && st->view)
    {
        vals[0] = (GLint)st->view_w;
        vals[1] = (GLint)st->view_h;
    }
    else if (st->param_values[param] != -1 && param < 512)
        *vals = st->param_values[param];
}

static void nsgl_ctx_flushBuffer(NSOpenGLContext *self, SEL _cmd)
{
    NSGLState *st = nsgl_ctx_state(self);
    if (!st || !st->view)
        return;

    /* windowed renderer bound to the view: present the CAMetalLayer drawable
     * (zero copy; same path glfwSwapBuffers takes on the MGL platform) */
    if (st->windowed)
    {
        MGLswapBuffers(st->glm);
        return;
    }

    /* clamp the readback to what the GL framebuffer covers (union of
     * viewports, tracked in fb_w/fb_h) */
    GLint vp[4] = { 0, 0, 0, 0 };
    glGetIntegerv(GL_VIEWPORT, vp);
    while (glGetError() != GL_NO_ERROR) {}
    if (vp[2] > 0 && (unsigned)vp[2] > st->fb_w) st->fb_w = (unsigned)vp[2];
    if (vp[3] > 0 && (unsigned)vp[3] > st->fb_h) st->fb_h = (unsigned)vp[3];

    unsigned rw = st->view_w < st->fb_w ? st->view_w : st->fb_w;
    unsigned rh = st->view_h < st->fb_h ? st->view_h : st->fb_h;
    if (!rw || !rh)
        return;

    size_t bytes = (size_t)rw * rh * 4;
    if (st->read_buf_size < bytes)
    {
        unsigned char *nb = realloc(st->read_buf, bytes);
        if (!nb)
            return;
        st->read_buf = nb;
        st->read_buf_size = bytes;
    }

    glFinish();
    while (glGetError() != GL_NO_ERROR) {}
    glReadPixels(0, 0, (GLsizei)rw, (GLsizei)rh, GL_BGRA,
                 GL_UNSIGNED_INT_8_8_8_8_REV, st->read_buf);
    while (glGetError() != GL_NO_ERROR) {}

    CALayer *layer = st->view.layer;
    if (!layer)
        return;

    /* blit into the view's layer: rows flipped (GL origin is bottom-left),
     * BGRA bytes map straight onto a little-endian CGImage */
    size_t row_bytes = (size_t)rw * 4;
    unsigned char *flipped = malloc(row_bytes * rh);
    if (!flipped)
        return;
    for (unsigned r = 0; r < rh; r++)
        memcpy(flipped + (size_t)r * row_bytes,
               st->read_buf + (size_t)(rh - 1 - r) * row_bytes,
               row_bytes);

    CGDataProviderRef prov = CGDataProviderCreateWithData(NULL, flipped,
                                                          row_bytes * rh, NULL);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGImageRef img = prov && cs
        ? CGImageCreate(rw, rh, 8, 32, row_bytes, cs,
                        kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst,
                        prov, NULL, false, kCGRenderingIntentDefault)
        : NULL;
    if (cs) CGColorSpaceRelease(cs);
    if (prov) CGDataProviderRelease(prov);
    if (!img)
    {
        free(flipped);
        return;
    }
    layer.contents = (__bridge id)img;
    CGImageRelease(img);
    free(flipped);


}

/* ------------------------------------------------------------------ */
/* install the swizzles                                                */

static IMP nsgl_swizzle(Class cls, SEL sel, IMP imp, BOOL is_class_method)
{
    Method m = is_class_method ? class_getClassMethod(cls, sel)
                               : class_getInstanceMethod(cls, sel);
    if (!m)
        return NULL;
    return method_setImplementation(m, imp);
}

__attribute__((constructor))
static void nsgl_init(void)
{
    nsgl_cache_exec_path();

    /* The present path uses IOSurface + layer.contents (no drawable blocking)
     * — see MGLRenderer+Compat.m.  This MUST run before MGLgetCurrentContext()
     * below: that call initialises libmgl and the renderer caches this env var
     * on first use (compatSurfacePresentEnabled).  When the setenv came last,
     * the early init call latched "disabled" for the whole process and every
     * swap fell through to -[CAMetalLayer nextDrawable], which pins the frame
     * rate to the display refresh (glmark2 full run: every scene 120-121 FPS
     * in a visible window / 27-40 FPS occluded; root cause of the mglR7/mglR8
     * regression, NOT the direct-present path it was first blamed on). */
    if (!getenv("MGL_SURFACE_PRESENT"))
        setenv("MGL_SURFACE_PRESENT", "1", 0);

    s_init_ctx = MGLgetCurrentContext();

    Class pf = objc_getClass("NSOpenGLPixelFormat");
    Class ct = objc_getClass("NSOpenGLContext");
    if (!pf || !ct)
        return;

    s_pf_orig_init = nsgl_swizzle(pf, @selector(initWithAttributes:),
                                  (IMP)nsgl_pf_initWithAttributes, NO);
    s_ctx_orig_init = nsgl_swizzle(ct, @selector(initWithFormat:shareContext:),
                                   (IMP)nsgl_ctx_initWithFormat, NO);
    s_ctx_orig_dealloc = nsgl_swizzle(ct, nsgl_dealloc_sel(),
                                      (IMP)nsgl_ctx_dealloc, NO);
    nsgl_swizzle(ct, @selector(setView:), (IMP)nsgl_ctx_setView, NO);

    /* POINTS vs PIXELS: keep bound Metal layers in sync with the app's
     * wantsBestResolutionOpenGLSurface (glmark2-macos flips it after setView:) */
    {
        Class nsview = objc_getClass("NSView");
        s_view_orig_setBestRes = nsgl_swizzle(nsview, @selector(setWantsBestResolutionOpenGLSurface:),
                                              (IMP)nsgl_view_setWantsBestResolution, NO);
    }
    nsgl_swizzle(ct, @selector(view), (IMP)nsgl_ctx_view, NO);
    nsgl_swizzle(ct, @selector(makeCurrentContext), (IMP)nsgl_ctx_makeCurrent, NO);
    nsgl_swizzle(ct, @selector(clearCurrentContext), (IMP)nsgl_ctx_clearCurrent, NO);
    nsgl_swizzle(ct, @selector(currentContext), (IMP)nsgl_ctx_current, YES);
    nsgl_swizzle(ct, @selector(update), (IMP)nsgl_ctx_update, NO);
    nsgl_swizzle(ct, @selector(setValues:forParameter:),
                 (IMP)nsgl_ctx_setValues, NO);
    nsgl_swizzle(ct, @selector(getValues:forParameter:),
                 (IMP)nsgl_ctx_getValues, NO);
    nsgl_swizzle(ct, @selector(flushBuffer), (IMP)nsgl_ctx_flushBuffer, NO);

    fprintf(stderr, "mgl_nsgl: NSOpenGL -> MGL interposer active\n");
}
