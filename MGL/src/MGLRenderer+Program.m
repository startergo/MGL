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

- (bool) extProgramCompileMSL:(const char *)msl entry:(const char *)entry
                      library:(void **)library function:(void **)function error:(char **)error_log
{
    @autoreleasepool {
        NSError *error = nil;
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

        lib = [_device newLibraryWithSource:src options:nil error:&error];
        if (!lib)
        {
            if (error_log)
                *error_log = strdup(error ? [[error localizedDescription] UTF8String] : "unknown Metal compiler error");
            return false;
        }

        fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
        if (!fn)
        {
            if (error_log)
                asprintf(error_log, "entry point '%s' not found in the compiled Metal library", entry);
            return false;
        }

        *library = (void *)CFBridgingRetain(lib);
        *function = (void *)CFBridgingRetain(fn);
        return true;
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
