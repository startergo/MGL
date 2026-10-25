/* mtlcount.m — Metal API call counter for A/B comparison of native OpenGL
 * (GLEngine -> AppleMetalOpenGLRenderer -> Metal) vs MGL.
 *
 * Observation tool only (supervisor instruction: observe, don't copy):
 * swizzles a fixed list of Metal framework instance methods with counting
 * IMPs that forward to the original implementation.  No behaviour change.
 *
 * Loaded via DYLD_INSERT_LIBRARIES.  Prints
 *     MTLCOUNT interval <n> <name> <count-delta>
 * every $MTL_COUNT_INTERVAL seconds (if set), and at exit
 *     MTLCOUNT total <name> <count>
 * Interval deltas / benchmark FPS give per-frame rates.
 *
 * Build:
 *   clang -O2 -fobjc-arc -dynamiclib ab/mtlcount.m -o build/mtlcount.dylib \
 *     -framework Metal -framework Foundation
 *
 * Use:
 *   native: DYLD_INSERT_LIBRARIES=build/mtlcount.dylib glmark2-macos ...
 *   MGL:    DYLD_INSERT_LIBRARIES="build/nsgl/libmgl_nsgl.dylib:build/mtlcount.dylib" ...
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#define MAXC 128
static _Atomic uint64_t g_cnt[MAXC];
static _Atomic uint64_t g_prev[MAXC];
static char g_names[MAXC][80];
static int g_ncnt = 0;
static int g_interval = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static Class g_done[64];
static int g_ndone = 0;

enum {
    S_VOID_SEL,        /* (void)(id,SEL)                                          */
    S_RET_SEL,         /* (id)(id,SEL)                                            */
    S_RET_DESC,        /* (id)(id,SEL,id)                                         */
    S_RET_N2,          /* (id)(id,SEL,NSUInteger,NSUInteger)                      */
    S_RET_BYTES,       /* (id)(id,SEL,const void*,NSUInteger,NSUInteger)          */
    S_RET_DESC_OPT,    /* (id)(id,SEL,id,NSUInteger,id) options+handler           */
    S_RET_DESC_ERR,    /* (id)(id,SEL,id,NSError**)  ...error: out-param          */
    S_VOID_BUF_STRIDE, /* (void)(id,SEL,id,NSUInteger,NSUInteger,NSUInteger)      */
    S_VOID_OBJ,        /* (void)(id,SEL,id)                                       */
    S_VOID_BYTES,      /* (void)(id,SEL,const void*,NSUInteger,NSUInteger)        */
    S_VOID_BUF,        /* (void)(id,SEL,id,NSUInteger,NSUInteger)                 */
    S_VOID_OBJ_IDX,    /* (void)(id,SEL,id,NSUInteger)                            */
    S_VOID_N1,         /* (void)(id,SEL,NSUInteger)                               */
    S_VOID_N1_R,       /* (void)(id,SEL,NSUInteger,NSUInteger) offset:atIndex:    */
    S_DRAW3,           /* drawPrimitives:vertexStart:vertexCount:                 */
    S_DRAW4,           /* + instanceCount:                                        */
    S_IDX5,            /* drawIndexedPrimitives:indexCount:indexType:indexBuffer: */
    S_IDX6,            /* + instanceCount:                                        */
    S_IDX8,            /* + instanceCount:baseVertex:baseInstance:                */
    S_USE_RES          /* useResource:usage:                                      */
};

typedef struct { const char *proto; const char *sel; int sig; } Entry;

static const Entry g_entries[] = {
    { "MTLCommandBuffer", "renderCommandEncoderWithDescriptor:", S_RET_DESC },
    { "MTLCommandBuffer", "computeCommandEncoderWithDescriptor:", S_RET_DESC },
    { "MTLCommandBuffer", "blitCommandEncoder", S_RET_SEL },
    { "MTLCommandBuffer", "commit", S_VOID_SEL },
    { "MTLRenderCommandEncoder", "endEncoding", S_VOID_SEL },
    { "MTLRenderCommandEncoder", "setRenderPipelineState:", S_VOID_OBJ },
    { "MTLRenderCommandEncoder", "setDepthStencilState:", S_VOID_OBJ },
    { "MTLRenderCommandEncoder", "setVertexBytes:length:atIndex:", S_VOID_BYTES },
    { "MTLRenderCommandEncoder", "setFragmentBytes:length:atIndex:", S_VOID_BYTES },
    { "MTLRenderCommandEncoder", "setVertexBuffer:offset:atIndex:", S_VOID_BUF },
    { "MTLRenderCommandEncoder", "setFragmentBuffer:offset:atIndex:", S_VOID_BUF },
    { "MTLRenderCommandEncoder", "setVertexBuffer:offset:attributeStride:atIndex:", S_VOID_BUF_STRIDE },
    { "MTLRenderCommandEncoder", "setFragmentTexture:atIndex:", S_VOID_OBJ_IDX },
    { "MTLRenderCommandEncoder", "setVertexTexture:atIndex:", S_VOID_OBJ_IDX },
    { "MTLRenderCommandEncoder", "setFragmentSamplerState:atIndex:", S_VOID_OBJ_IDX },
    { "MTLRenderCommandEncoder", "setVertexBufferOffset:atIndex:", S_VOID_N1_R },
    { "MTLRenderCommandEncoder", "setFragmentBufferOffset:atIndex:", S_VOID_N1_R },
    { "MTLRenderCommandEncoder", "setVertexSamplerState:atIndex:", S_VOID_OBJ_IDX },
    { "MTLRenderCommandEncoder", "setTriangleFillMode:", S_VOID_N1 },
    { "MTLRenderCommandEncoder", "setCullMode:", S_VOID_N1 },
    { "MTLRenderCommandEncoder", "setFrontFacingWinding:", S_VOID_N1 },
    { "MTLRenderCommandEncoder", "setStencilReferenceValue:", S_VOID_N1 },
    { "MTLRenderCommandEncoder", "useResource:usage:", S_USE_RES },
    { "MTLRenderCommandEncoder", "drawPrimitives:vertexStart:vertexCount:", S_DRAW3 },
    { "MTLRenderCommandEncoder", "drawPrimitives:vertexStart:vertexCount:instanceCount:", S_DRAW4 },
    { "MTLRenderCommandEncoder", "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:", S_IDX5 },
    { "MTLRenderCommandEncoder", "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:", S_IDX6 },
    { "MTLRenderCommandEncoder", "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:", S_IDX8 },
    { "MTLComputeCommandEncoder", "endEncoding", S_VOID_SEL },
    { "MTLComputeCommandEncoder", "setComputePipelineState:", S_VOID_OBJ },
    { "MTLDevice", "newRenderPipelineStateWithDescriptor:", S_RET_DESC_ERR },
    { "MTLDevice", "newRenderPipelineStateWithDescriptor:options:completionHandler:", S_RET_DESC_OPT },
    { "MTLDevice", "newDepthStencilStateWithDescriptor:", S_RET_DESC_ERR },
    { "MTLDevice", "newSamplerStateWithDescriptor:", S_RET_DESC_ERR },
    { "MTLDevice", "newBufferWithLength:options:", S_RET_N2 },
    { "MTLDevice", "newBufferWithBytes:length:options:", S_RET_BYTES },
};

/* trampoline macro: params = block parameter list (self + args, NO _cmd —
 * imp_implementationWithBlock does not pass the selector); the original IMP
 * is invoked with the selector captured at swizzle time; call = statement
 * invoking orig (its `return' propagates), then count */
#define DEF_TRAMP_FULL(tag, rettype, params, call)                              \
    static IMP tramp_##tag(IMP orig, SEL sel, _Atomic uint64_t *cnt)            \
    {                                                                           \
        rettype (^b) params = ^rettype params { call; atomic_fetch_add(cnt, 1); }; \
        return imp_implementationWithBlock(b);                                  \
    }

DEF_TRAMP_FULL(void_sel, void, (id self),
    ((void (*)(id, SEL))orig)(self, sel))
DEF_TRAMP_FULL(ret_sel, id, (id self),
    return ((id (*)(id, SEL))orig)(self, sel))
DEF_TRAMP_FULL(ret_desc, id, (id self, id a),
    return ((id (*)(id, SEL, id))orig)(self, sel, a))
DEF_TRAMP_FULL(ret_n2, id, (id self, NSUInteger a, NSUInteger b),
    return ((id (*)(id, SEL, NSUInteger, NSUInteger))orig)(self, sel, a, b))
DEF_TRAMP_FULL(ret_bytes, id, (id self, const void *a, NSUInteger b, NSUInteger c),
    return ((id (*)(id, SEL, const void *, NSUInteger, NSUInteger))orig)(self, sel, a, b, c))
DEF_TRAMP_FULL(void_obj, void, (id self, id a),
    ((void (*)(id, SEL, id))orig)(self, sel, a))
DEF_TRAMP_FULL(void_bytes, void, (id self, const void *a, NSUInteger b, NSUInteger c),
    ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))orig)(self, sel, a, b, c))
DEF_TRAMP_FULL(ret_desc_err, id, (id self, id a, NSError **err),
    return ((id (*)(id, SEL, id, NSError **))orig)(self, sel, a, err))
DEF_TRAMP_FULL(ret_desc_opt, id, (id self, id a, NSUInteger b, id c),
    return ((id (*)(id, SEL, id, NSUInteger, id))orig)(self, sel, a, b, c))
DEF_TRAMP_FULL(void_buf_stride, void, (id self, id a, NSUInteger b, NSUInteger c, NSUInteger d),
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger, NSUInteger))orig)(self, sel, a, b, c, d))
DEF_TRAMP_FULL(void_buf, void, (id self, id a, NSUInteger b, NSUInteger c),
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))orig)(self, sel, a, b, c))
DEF_TRAMP_FULL(void_obj_idx, void, (id self, id a, NSUInteger b),
    ((void (*)(id, SEL, id, NSUInteger))orig)(self, sel, a, b))
DEF_TRAMP_FULL(void_n1_r, void, (id self, NSUInteger a, NSUInteger b),
    ((void (*)(id, SEL, NSUInteger, NSUInteger))orig)(self, sel, a, b))
DEF_TRAMP_FULL(void_n1, void, (id self, NSUInteger a),
    ((void (*)(id, SEL, NSUInteger))orig)(self, sel, a))
DEF_TRAMP_FULL(draw3, void, (id self, NSUInteger a, NSUInteger b, NSUInteger c),
    ((void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger))orig)(self, sel, a, b, c))
DEF_TRAMP_FULL(draw4, void, (id self, NSUInteger a, NSUInteger b, NSUInteger c, NSUInteger d),
    ((void (*)(id, SEL, NSUInteger, NSUInteger, NSUInteger, NSUInteger))orig)(self, sel, a, b, c, d))
DEF_TRAMP_FULL(idx5, void, (id self, NSUInteger a, NSUInteger b, id c, NSUInteger d),
    ((void (*)(id, SEL, NSUInteger, NSUInteger, id, NSUInteger))orig)(self, sel, a, b, c, d))
DEF_TRAMP_FULL(idx6, void, (id self, NSUInteger a, NSUInteger b, id c, NSUInteger d, NSUInteger e),
    ((void (*)(id, SEL, NSUInteger, NSUInteger, id, NSUInteger, NSUInteger))orig)(self, sel, a, b, c, d, e))
DEF_TRAMP_FULL(idx8, void, (id self, NSUInteger a, NSUInteger b, id c, NSUInteger d, NSUInteger e, NSInteger f, NSUInteger g),
    ((void (*)(id, SEL, NSUInteger, NSUInteger, id, NSUInteger, NSUInteger, NSInteger, NSUInteger))orig)(self, sel, a, b, c, d, e, f, g))
DEF_TRAMP_FULL(use_res, void, (id self, id a, NSUInteger b),
    ((void (*)(id, SEL, id, NSUInteger))orig)(self, sel, a, b))

static IMP make_tramp(int sig, IMP orig, SEL sel, _Atomic uint64_t *cnt)
{
    switch (sig) {
    case S_VOID_SEL: return tramp_void_sel(orig, sel, cnt);
    case S_RET_SEL: return tramp_ret_sel(orig, sel, cnt);
    case S_RET_DESC: return tramp_ret_desc(orig, sel, cnt);
    case S_RET_N2: return tramp_ret_n2(orig, sel, cnt);
    case S_RET_BYTES: return tramp_ret_bytes(orig, sel, cnt);
    case S_RET_DESC_OPT: return tramp_ret_desc_opt(orig, sel, cnt);
    case S_RET_DESC_ERR: return tramp_ret_desc_err(orig, sel, cnt);
    case S_VOID_BUF_STRIDE: return tramp_void_buf_stride(orig, sel, cnt);
    case S_VOID_OBJ: return tramp_void_obj(orig, sel, cnt);
    case S_VOID_BYTES: return tramp_void_bytes(orig, sel, cnt);
    case S_VOID_BUF: return tramp_void_buf(orig, sel, cnt);
    case S_VOID_OBJ_IDX: return tramp_void_obj_idx(orig, sel, cnt);
    case S_VOID_N1: return tramp_void_n1(orig, sel, cnt);
    case S_VOID_N1_R: return tramp_void_n1_r(orig, sel, cnt);
    case S_DRAW3: return tramp_draw3(orig, sel, cnt);
    case S_DRAW4: return tramp_draw4(orig, sel, cnt);
    case S_IDX5: return tramp_idx5(orig, sel, cnt);
    case S_IDX6: return tramp_idx6(orig, sel, cnt);
    case S_IDX8: return tramp_idx8(orig, sel, cnt);
    case S_USE_RES: return tramp_use_res(orig, sel, cnt);
    }
    return NULL;
}

static bool group_enabled(const char *tag)
{
    const char *sk = getenv("MTL_COUNT_SKIP");
    if (!sk) return true;
    return !strstr(sk, tag);
}

static void try_class(Class c, const char *tag)
{
    if (!group_enabled(tag)) return;
    for (size_t e = 0; e < sizeof g_entries / sizeof g_entries[0]; e++) {
        if (strcmp(g_entries[e].proto, tag) != 0)
            continue;
        const char *only = getenv("MTL_COUNT_ONLY");
        if (only && !strstr(g_entries[e].sel, only))
            continue;
        SEL sel = sel_registerName(g_entries[e].sel);
        Method m = class_getInstanceMethod(c, sel);
        if (!m || g_ncnt >= MAXC)
            continue;
        IMP orig = method_getImplementation(m);
        if (!orig)
            continue;
        IMP t = make_tramp(g_entries[e].sig, orig, sel, &g_cnt[g_ncnt]);
        if (!t)
            continue;
        method_setImplementation(m, t);
        if (getenv("MTL_COUNT_DEBUG"))
            fprintf(stderr, "MTLCOUNT swizzle %s -> %s\n", class_getName(c), g_entries[e].sel);
        snprintf(g_names[g_ncnt], sizeof g_names[0], "%s|%s", tag, g_entries[e].sel);
        g_ncnt++;
    }
}

static void scan_classes(void)
{
    pthread_mutex_lock(&g_mu);
    unsigned int n = 0;
    Class *all = (Class *)objc_copyClassList(&n);
    pthread_mutex_unlock(&g_mu);
    if (!all)
        return;
    for (unsigned int i = 0; i < n; i++) {
        Class c = all[i];
        const char *tag = NULL;
        if (class_conformsToProtocol(c, @protocol(MTLRenderCommandEncoder)))
            tag = "MTLRenderCommandEncoder";
        else if (class_conformsToProtocol(c, @protocol(MTLComputeCommandEncoder)))
            tag = "MTLComputeCommandEncoder";
        else if (class_conformsToProtocol(c, @protocol(MTLCommandBuffer)))
            tag = "MTLCommandBuffer";
        else if (class_conformsToProtocol(c, @protocol(MTLDevice)))
            tag = "MTLDevice";
        if (!tag)
            continue;
        bool seen = false;
        pthread_mutex_lock(&g_mu);
        for (int k = 0; k < g_ndone; k++)
            if (g_done[k] == c) { seen = true; break; }
        if (!seen && g_ndone < (int)(sizeof g_done / sizeof g_done[0]))
            g_done[g_ndone++] = c;
        pthread_mutex_unlock(&g_mu);
        if (!seen)
            try_class(c, tag);
    }
    free(all);
}

static void report(const char *kind)
{
    fprintf(stderr, "MTLCOUNT %s begin\n", kind);
    for (int i = 0; i < g_ncnt; i++)
        fprintf(stderr, "MTLCOUNT %s %s %llu\n", kind, g_names[i],
                (unsigned long long)atomic_load(&g_cnt[i]));
    fprintf(stderr, "MTLCOUNT %s end\n", kind);
    fflush(stderr);
}

static void report_total(void)
{
    static int done;
    if (done++)
        return;
    report("total");
}

static void *poll_thread(void *arg)
{
    (void)arg;
    for (int i = 0; i < 600; i++) {   /* 2 min of late-load watching */
        scan_classes();
        usleep(200000);
    }
    return NULL;
}

static void interval_report(void)
{
    static int n;
    n++;
    fprintf(stderr, "MTLCOUNT interval %d begin\n", n);
    for (int i = 0; i < g_ncnt; i++) {
        uint64_t c = atomic_load(&g_cnt[i]);
        fprintf(stderr, "MTLCOUNT interval %d %s %llu\n", n, g_names[i],
                (unsigned long long)(c - atomic_load(&g_prev[i])));
        atomic_store(&g_prev[i], c);
    }
    fprintf(stderr, "MTLCOUNT interval %d end\n", n);
    fflush(stderr);
}

static void *interval_thread(void *arg)
{
    (void)arg;
    for (;;) {
        sleep((unsigned)g_interval);
        interval_report();
    }
    return NULL;
}

__attribute__((constructor)) static void mtlcount_init(void)
{
    const char *iv = getenv("MTL_COUNT_INTERVAL");
    if (iv && atoi(iv) > 0) {
        g_interval = atoi(iv);
        pthread_t t;
        pthread_create(&t, NULL, interval_thread, NULL);
        pthread_detach(t);
    }
    scan_classes();
    pthread_t t;
    pthread_create(&t, NULL, poll_thread, NULL);
    pthread_detach(t);
    atexit(report_total);
}
