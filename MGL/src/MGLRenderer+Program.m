/*
 * MGLRenderer+Program.m — Metal side of the 'program' domain.
 *
 * Compiles a linked stage's MSL into an MTLLibrary/MTLFunction at link time so that
 * Metal compile errors are reported through the program info log, and releases the
 * retained Metal objects of program executables.
 */

#import "MGLRenderer_Private.h"
#include "mgl_ext_program.h"

@interface MGLRenderer (Program)
- (bool) extProgramCompileMSL:(const char *)msl entry:(const char *)entry
                      library:(void **)library function:(void **)function error:(char **)error_log;
/* MSL-source -> MTLLibrary cache (shared with -compileShader:); compiles on miss */
- (id<MTLLibrary>) extProgramLibraryForMSL:(const char *)msl key:(NSString *)key error:(char **)error_log;
@end

static bool mtlProgramCompileMSL(GLMContext glm_ctx, const char *msl, const char *entry,
                                 void **library, void **function, char **error_log)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    return [renderer extProgramCompileMSL:msl entry:entry library:library function:function error:error_log];
}

static void mtlProgramReleaseObj(GLMContext glm_ctx, void *obj)
{
    (void)glm_ctx;
    if (obj)
        CFBridgingRelease(obj);
}

@implementation MGLRenderer (Program)

/* MSL-source -> compiled library / entry-function cache.  CTS-style suites relink the
 * same shaders thousands of times (texture_swizzle.smoke alone compiles ~10k programs);
 * glslang+SPIRV-Cross and especially newLibraryWithSource dominate the run time, so the
 * immutable results are reused.  Keyed by device pointer + MSL text (functions are
 * default-specialized exactly as the uncached path did). */
static NSMutableDictionary<NSString *, NSMutableDictionary<NSString *, id> *> *msl_lib_cache;
static NSLock *msl_lib_cache_lock = nil;
static NSLock *mslLibCacheLock(void)
{
    static dispatch_once_t once;
    dispatch_once(&once, ^{ msl_lib_cache_lock = [[NSLock alloc] init]; });
    return msl_lib_cache_lock;
}

/* the entry-cache dictionary for an MSL source on a device, creating it if needed
 * (caller holds the lock; returning a stored autoreleased reference is fine under ARC) */
static NSMutableDictionary *mglEntryCacheForKey(NSMutableDictionary *cache, const char *msl, id<MTLDevice> dev)
{
    NSString *key = [NSString stringWithFormat:@"%p %s", dev, msl];
    NSMutableDictionary *e = cache[key];
    if (!e)
    {
        e = [NSMutableDictionary new];
        cache[key] = e;
    }
    return e;
}

/* function-pointer -> (MSL text + entry) content id: the compat pipeline cache uses
 * stage-function POINTERS as MSL content identity ("equal pointers mean equal code"),
 * which holds while this cache keeps every function alive forever.  This registry
 * makes that assumption checkable: a released-and-recycled address gets a new id,
 * and the audit in MGLRenderer+Compat.m drops any pipeline-cache entry whose stored
 * id no longer matches (a wrong-PSO hit would render with foreign shader code). */
#define MGL_FNID_CAP 32768  /* power of two; ids are never removed (was 8192: CTS
                             * suites that relink >2k programs saturated it and the
                             * fast path silently degraded to decline-everything) */
static void      *fnid_key[MGL_FNID_CAP];
static uint64_t   fnid_val[MGL_FNID_CAP];
static int        fnid_used;

static uint64_t fnidHash(void *fn)
{
    uintptr_t v = (uintptr_t)fn;
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    return (uint64_t)v;
}

/* caller holds mslLibCacheLock() */
static void fnidRegister(void *fn, uint64_t id)
{
    if (!fn) return;
    if (fnid_used * 4 > MGL_FNID_CAP * 3) return;   /* full: audit degrades to no-op */
    uint64_t mask = MGL_FNID_CAP - 1;
    uint64_t i = fnidHash(fn) & mask;
    while (fnid_key[i] && fnid_key[i] != fn) i = (i + 1) & mask;
    if (!fnid_key[i]) { fnid_key[i] = fn; fnid_used++; }
    fnid_val[i] = id;
}

/* Register <function>'s MSL content identity (the FULL MSL text hash) so the
 * compat pipeline fast path can key on it.  extProgramCompileMSL registers its
 * functions itself; bindMTLProgram installs functions that came from the
 * library cache directly (compileShader:), and those must be registered too or
 * every draw of such a program declines the fast path (glmark2 ideas -63%). */
void mglProgramRegisterMSLIdentity(const void *function, const char *msl, const char *entry)
{
    if (!function || !msl)
        return;
    fnidRegister((void *)function, fnidOf(msl, entry ? entry : "main"));
}

uint64_t mglProgramFunctionId(void *fn)
{
    if (!fn) return 0;
    NSLock *lock = mslLibCacheLock();
    [lock lock];
    uint64_t mask = MGL_FNID_CAP - 1;
    uint64_t i = fnidHash(fn) & mask;
    uint64_t id = 0;
    while (fnid_key[i])
    {
        if (fnid_key[i] == fn) { id = fnid_val[i]; break; }
        i = (i + 1) & mask;
    }
    [lock unlock];
    return id;
}

static uint64_t fnidOf(const char *msl, const char *entry)
{
    uint64_t h = 1469598103934665603ULL;
    for (const char *p = entry; *p; p++) { h = (h ^ (uint8_t)*p) * 1099511628211ULL; }
    h = (h ^ 0x2f) * 1099511628211ULL;
    for (const char *p = msl; *p; p++) { h = (h ^ (uint8_t)*p) * 1099511628211ULL; }
    return h ? h : 1;
}

- (bool) extProgramCompileMSL:(const char *)msl entry:(const char *)entry
                      library:(void **)library function:(void **)function error:(char **)error_log
{
    @autoreleasepool {
        id<MTLLibrary> lib;
        id<MTLFunction> fn;
        NSString *src;

        *library = NULL;
        *function = NULL;

        if (!_device || !msl || !entry)
        {
            if (error_log) *error_log = strdup("no Metal device");
            return false;
        }

        src = [NSString stringWithUTF8String:msl];
        if (!src)
        {
            if (error_log) *error_log = strdup("MSL source is not valid UTF-8");
            return false;
        }

        NSString *name = [NSString stringWithUTF8String:entry];
        NSLock *lock = mslLibCacheLock();

        [lock lock];
        if (!msl_lib_cache)
            msl_lib_cache = [NSMutableDictionary new];
        NSMutableDictionary *entry_cache = mglEntryCacheForKey(msl_lib_cache, msl, _device);
        id<MTLFunction> cached_fn = entry_cache[name];
        if (cached_fn)
        {
            id<MTLLibrary> cached_lib = entry_cache[@"__lib"];
            fnidRegister((void *)(__bridge void *)cached_fn, fnidOf(msl, entry));
            [lock unlock];
            *library = (void *)CFBridgingRetain(cached_lib);
            *function = (void *)CFBridgingRetain(cached_fn);
            return true;
        }
        [lock unlock];

        lib = [self extProgramLibraryForMSL: msl key: nil error: error_log];
        if (!lib)
        {
            if (getenv("MGL_MSL_DEBUG"))
                fprintf(stderr, "MGL msl: library compile failed: %s entry %s\n",
                        (error_log && *error_log) ? *error_log : "?", entry);
            return false;
        }

        /* specialized with no values, like -bindMTLProgram: functions carrying function
        /* specialized with no values, like -bindMTLProgram: functions carrying function
         * constants (rstate fixups) take their GL-default branches.  constantValues
         * must be non-nil: Metal validation asserts on nil (test_atomic_counters
         * SIGTRAPped in the full suite; 'failure -[MGLRenderer processGLState:]' logs
         * otherwise) — an empty values object selects the same default branches. */
        fn = [lib newFunctionWithName:name
                        constantValues:[[MTLFunctionConstantValues alloc] init]
                                  error:nil];
        if (!fn)
        {
            if (getenv("MGL_MSL_DEBUG"))
                fprintf(stderr, "MGL msl: entry '%s' not found in library\n", entry);
            if (error_log)
                asprintf(error_log, "entry point '%s' not found in the compiled Metal library", entry);
            return false;
        }

        *library = (void *)CFBridgingRetain(lib);
        *function = (void *)CFBridgingRetain(fn);

        {
            NSLock *lock2 = mslLibCacheLock();
            [lock2 lock];
            fnidRegister(*function, fnidOf(msl, entry));
            [lock2 unlock];
        }

        [lock lock];
        entry_cache[@"__lib"] = lib;
        entry_cache[name] = fn;
        [lock unlock];
        return true;
    }
}

/* shared cache lookup / insert used by -extProgramCompileMSL: and the renderer's
 * -compileShader: (which used to re-run newLibraryWithSource per program link,
 * e.g. for every rstate re-specialization) */
- (id<MTLLibrary>) extProgramLibraryForMSL: (const char *)msl key: (NSString *)key error: (char **)error_log
{
    @autoreleasepool {
        NSLock *lock = mslLibCacheLock();
        NSString *src;

        if (!msl)
            return nil;

        [lock lock];
        if (msl_lib_cache)
        {
            NSMutableDictionary *entry_cache = mglEntryCacheForKey(msl_lib_cache, msl, _device);
            id<MTLLibrary> cached = entry_cache[@"__lib"];
            if (cached)
            {
                [lock unlock];
                return cached;
            }
        }
        else
            msl_lib_cache = [NSMutableDictionary new];
        [lock unlock];

        src = [NSString stringWithUTF8String: msl];
        if (!src)
            return nil;

        /* tess domain: precise math when the MSL embeds the fp64 double-single library */
        NSError *error = nil;
        id<MTLLibrary> lib = [_device newLibraryWithSource: src
                                                   options: [self mslCompileOptionsForSource: src]
                                                     error: &error];
        if (!lib)
        {
            if (error_log)
                *error_log = strdup(error ? [[error localizedDescription] UTF8String]
                                          : "unknown Metal compiler error");
            return nil;
        }

        [lock lock];
        mglEntryCacheForKey(msl_lib_cache, msl, _device)[@"__lib"] = lib;
        [lock unlock];
        return lib;
    }
}

- (void) bindExtProgramFuncsToGLMContext: (GLMContext) glm_ctx
{
    static struct MGLExtMetal_program funcs = {
        .compile_msl = mtlProgramCompileMSL,
        .release_obj = mtlProgramReleaseObj,
    };

    glm_ctx->mtl_funcs.ext_program = &funcs;
}

@end
