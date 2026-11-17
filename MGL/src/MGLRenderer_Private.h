/*
 * MGLRenderer_Private.h — internal interface of MGLRenderer.
 *
 * Category files (MGLRenderer+<Domain>.m) import this to reach renderer state.
 * A category may implement
 *     - (void) bindExt<Domain>FuncsToGLMContext: (GLMContext) glm_ctx;
 * which -bindObjFuncsToGLMContext: calls automatically (objc runtime scan), to
 * install its mtl_funcs hooks without editing MGLRenderer.m.
 *
 * ext hook methods — a category may also implement any of (called for every domain):
 *     - (void) ext<Domain>RenderPassDescriptor: (MTLRenderPassDescriptor *) desc;  // before encoder creation
 *     - (void) ext<Domain>RenderEncoderCreated: (id<MTLRenderCommandEncoder>) enc;
 *     - (void) ext<Domain>PreDraw: (id<MTLRenderCommandEncoder>) enc;            // end of processGLState(true)
 *     - (void) ext<Domain>WillCommit: (id<MTLCommandBuffer>) cmdbuf;              // in commitCommandBufferWithAGXRecovery
 */
#ifndef MGLRenderer_Private_h
#define MGLRenderer_Private_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>

#import "MGLRenderer.h"
#import "glm_context.h"
#import "spirv_cross_c.h"

typedef struct SyncList_t {
    GLuint count;
    GLuint  size;
    Sync **list;
} SyncList;

MTLPixelFormat mtlPixelFormatForGLTex(Texture * gl_tex);

/* MGL_PROFILE=1 per-draw cost attribution (MGLRenderer.m); mgl_profile_dump is
 * called from the swap path / interposer to log rolling per-frame averages */
uint64_t mgl_prof_begin(int id);
void mgl_prof_end(int id, uint64_t t0);
void mgl_prof_slow(const char *what, uint64_t t0, const char *text);   /* r160 */
/* r164: MGL_PROFILE call-site histogram.  Records the two return addresses
 * `skip` frames above the instrumented function (no symbol lookup on the hot
 * path); mgl_profile_dump names them with dladdr and prints the top sites per
 * frame.  Kinds: */
enum { MGL_SITE_PASS_END = 0,    /* a render pass ended (an encoder existed)       */
       MGL_SITE_FLUSH,           /* extBufFlush committed the command buffer       */
       MGL_SITE_NEW_CMDBUF,      /* a new command buffer was started               */
       MGL_SITE_REAPPLY,         /* all draw state marked for re-apply (r152 mask) */
       MGL_SITE_KINDS };
void mgl_prof_site(int kind, int skip);
/* r164: the "MGL TRAIL" command-buffer status traces are debug output; they ran
 * unconditionally (NSLog on every readback / blit / copy).  MGL_TRAIL=1 enables. */
int mgl_trail_on(void);
#define MGL_TRAIL_LOG(...) do { if (mgl_trail_on()) NSLog(__VA_ARGS__); } while (0)
void mgl_profile_dump(void);
void mgl_prof_frame(void);

/* MGLRenderer+VertexAttrib.m — binds the vattrib plan's per-attribute vertex
 * buffers to an encoder.  Declared here because the transform-feedback capture
 * path has no processGLState pass and must call it directly. */
@interface MGLRenderer (VertexAttribPreDraw)
- (void) extVattribPreDraw: (id<MTLRenderCommandEncoder>) enc;
@end

/* Resolve the texture a sampler/image uniform of GL target <gl_target> on texture
 * unit <unit> must sample.  GL semantics: the sampler's dimension picks the unit's
 * binding for THAT target (a sampler2DMS reads GL_TEXTURE_2D_MULTISAMPLE even when
 * GL_TEXTURE_2D of the same unit was bound last — KHR-GL46.direct_state_access
 * textures_storage_multisample does exactly that through glBindTextureUnit).
 * Falls back to the unit's "most recently bound" texture (active_textures) when
 * the per-target binding is empty or the reflected target is unknown, which keeps
 * array-of-samplers and pre-existing behaviour working. */
static inline Texture * mgl_texture_for_sampler(GLMContext ctx, GLuint unit, GLuint gl_target)
{
    if (unit >= TEXTURE_UNITS)
        return NULL;
    if (gl_target)
    {
        GLuint idx = _MAX_TEXTURE_TYPES;
        switch (gl_target)
        {
            case GL_TEXTURE_BUFFER: idx = _TEXTURE_BUFFER_TARGET; break;
            case GL_TEXTURE_1D: idx = _TEXTURE_1D; break;
            case GL_TEXTURE_2D: idx = _TEXTURE_2D; break;
            case GL_TEXTURE_3D: idx = _TEXTURE_3D; break;
            case GL_TEXTURE_RECTANGLE: idx = _TEXTURE_RECTANGLE; break;
            case GL_TEXTURE_1D_ARRAY: idx = _TEXTURE_1D_ARRAY; break;
            case GL_TEXTURE_2D_ARRAY: idx = _TEXTURE_2D_ARRAY; break;
            case GL_TEXTURE_CUBE_MAP: idx = _TEXTURE_CUBE_MAP; break;
            case GL_TEXTURE_CUBE_MAP_ARRAY: idx = _TEXTURE_CUBE_MAP_ARRAY; break;
            case GL_TEXTURE_2D_MULTISAMPLE: idx = _TEXTURE_2D_MULTISAMPLE; break;
            case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: idx = _TEXTURE_2D_MULTISAMPLE_ARRAY; break;
            default: break;
        }
        if (idx < _MAX_TEXTURE_TYPES)
        {
            Texture *t = ctx->state.texture_units[unit].textures[idx];
            if (t)
                return t;
        }
    }
    return ctx->state.active_textures[unit];
}

typedef struct MGLDrawable_t {
    GLuint width;
    GLuint height;
    id<MTLTexture> drawbuffer;
    id<MTLTexture> depthbuffer;
    id<MTLTexture> stencilbuffer;
} MGLDrawable;

enum {
    _FRONT,
    _BACK,
    _FRONT_LEFT,
    _FRONT_RIGHT,
    _BACK_LEFT,
    _BACK_RIGHT,
    _MAX_DRAW_BUFFERS
};

// CRITICAL SECURITY: Safe Metal object validation helper
static inline id<NSObject> SafeMetalBridge(void *ptr, Class expectedClass, const char *objectName) {
    if (!ptr) {
        NSLog(@"MGL SECURITY ERROR: NULL pointer for %s", objectName);
        return nil;
    }

    id<NSObject> obj = (__bridge id<NSObject>)(ptr);
    if (!obj) {
        NSLog(@"MGL SECURITY ERROR: Metal bridge cast returned nil for %s", objectName);
        return nil;
    }

    if (expectedClass && [obj isKindOfClass:expectedClass] == NO) {
        NSLog(@"MGL SECURITY ERROR: Metal object is not valid %s (got %@)", objectName, NSStringFromClass([obj class]));
        return nil;
    }

    return obj;
}


/* geom: true when the program's VERTEX MSL emulates shadow sampling at Metal texture slot N
 * (mgl_fix_shadow_vs); those slots bind the compare-stripped sampler */
static inline __attribute__((always_inline)) bool mgl_geom_vs_shadow_slot(Program *p, GLuint slot)
{
    return p && slot < 32 && (p->spirv[_VERTEX_SHADER].shadow_slots & (1u << slot));
}

@interface MGLRenderer ()
{
    NSView *_view;

    CAMetalLayer *_layer;
    id<CAMetalDrawable> _drawable;

    GLMContext  ctx;    // context macros need this exact name

    id<MTLDevice> _device;

    // CRITICAL FIX: Thread synchronization to prevent race conditions
    NSLock *_metalStateLock;

    // AGX GPU Error Tracking - Prevent command queue from entering error state
    NSUInteger _consecutiveGPUErrors;
    NSTimeInterval _lastGPUErrorTime;
    BOOL _gpuErrorRecoveryMode;

    // PROACTIVE TEXTURE STORAGE - Essential textures created during initialization

    MGLDrawable _drawBuffers[_MAX_DRAW_BUFFERS];


    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _commandQueue;

    // The render pipeline generated from the vertex and fragment shaders in the .metal shader file.
    id<MTLRenderPipelineState> _pipelineState;

    // render pass descriptor containts the binding information for VAO's and such
    MTLRenderPassDescriptor *_renderPassDescriptor;

    // each pass a new command buffer is created
    id<MTLCommandBuffer> _currentCommandBuffer;
    SyncList  *_currentCommandBufferSyncList;

    id<MTLRenderCommandEncoder> _currentRenderEncoder;

    GLuint _blitOperationComplete;

    id<MTLEvent> _currentEvent;
    GLsizei _currentSyncName;
}


- (void) bindMTLBuffer:(Buffer *) ptr;- (bool) mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage: (int) stage;
- (bool) mapBuffersToMTL;
- (Buffer *) nullShaderBuffer;   /* MGLRenderer.m: zero-filled dummy for unbound [[buffer(n)]] slots */
- (bool) updateDirtyBuffer:(Buffer *)ptr;
- (bool) checkForDirtyBufferData: (BufferMapList *)buffer_map_list;
- (bool) updateDirtyBaseBufferList: (BufferMapList *)buffer_map_list;
- (bool) bindVertexBuffersToCurrentRenderEncoder;
- (bool) bindFragmentBuffersToCurrentRenderEncoder;
- (int) getVertexBufferIndexWithAttributeSet: (int) attribute;
- (void)swizzleTexDesc:(MTLTextureDescriptor *)tex_desc forTex:(Texture*)tex;
- (id<MTLTexture>) createMTLTextureFromGLTexture:(Texture *) tex;
- (id<MTLTexture>) createFallbackMTLTexture:(Texture *) tex;
- (NSUInteger)bytesPerPixelForFormat:(GLenum)internalformat;
- (id<MTLSamplerState>) createMTLSamplerForTexParam:(TextureParameter *)tex_param target:(GLuint)target;
- (bool) bindTexturesToCurrentRenderEncoder;
- (bool) bindTexturesToCurrentRenderEncoderUnits: (const unsigned *) unit_mask;   /* r158 */
-(void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(size_t)srcX0 srcY0:(size_t)srcY0 srcX1:(size_t)srcX1 srcY1:(size_t)srcY1 dstX0:(size_t)dstX0 dstY0:(size_t)dstY0 dstX1:(size_t)dstX1 dstY1:(size_t)dstY1 mask:(size_t)mask filter:(GLuint)filter;
- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment;
- (bool)bindMTLTexture:(Texture *)tex;
- (bool)bindActiveTexturesToMTL;
- (bool) texwalkSampledUnits: (unsigned *) want;   /* r165: units the bound program can sample */
- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer;
- (int) getProgramBindingCount: (int) stage type: (int) type;
- (int) getProgramBinding: (int) stage type: (int) type index: (int) index;
- (int) getProgramLocation: (int) stage type: (int) type index: (int) index;
- (id<MTLLibrary>) compileShader: (const char *) str;

-(bool)bindMTLProgram:(Program *)ptr;
- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil customSize:(CGSize)size;
- (bool) checkDrawBufferSize:(GLuint) index;
- (void) updateCurrentRenderEncoder;
- (bool) newRenderEncoder;
- (bool) newCommandBuffer;
- (bool) newCommandBufferAndRenderEncoder;
-(MTLRenderPipelineDescriptor *)generatePipelineDescriptor;
- (MTLVertexDescriptor *)generateVertexDescriptor;
-(void)bindBlendStateToPipelineStateDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor;
-(bool)bindFramebufferAttachmentTextures;
- (void) endRenderEncoding;
- (void) emergencyResetMetalState;
- (bool) processGLState: (bool) draw_command;

- (bool) bindBuffersToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder;
- (bool) bindTexturesToComputeEncoder:(id <MTLComputeCommandEncoder>) computeCommandEncoder;
-(bool)processCompute:(id <MTLComputeCommandEncoder>) computeCommandEncoder;
-(void)mtlDispatchCompute:(GLMContext)glm_ctx groupsX:(GLuint)groups_x groupsY:(GLuint)groups_y groupsZ:(GLuint)groups_z;
-(void)mtlDispatchComputeIndirect:(GLMContext)glm_ctx indirect:(GLintptr)indirect;
-(bool) processBuffer:(Buffer*)ptr;
-(void) flushCommandBuffer: (bool) finish;
-(void) mtlDeleteMTLObj:(GLMContext) glm_ctx buffer: (void *)obj;
-(void) mtlGetSync:(GLMContext) glm_ctx sync: (Sync *)sync;
-(void) mtlWaitForSync:(GLMContext) glm_ctx sync: (Sync *)sync;
-(void) mtlFlush:(GLMContext) glm_ctx finish:(bool)finish;
-(void) mtlSwapBuffers:(GLMContext) glm_ctx;
-(void) mtlClearBuffer:(GLMContext) glm_ctx type:(GLuint) type mask:(GLbitfield) mask;
-(void) mtlBufferSubData:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t)offset size:(size_t)size ptr:(const void *)ptr;
-(void *) mtlMapUnmapBuffer:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset size:(size_t) size access:(GLenum) access map:(bool)map;
-(void) mtlFlushMappedBufferRange:(GLMContext) glm_ctx buf:(Buffer *)buf offset:(size_t) offset length:(size_t) length;
-(void)mtlGenerateMipmaps:(GLMContext)glm_ctx forTexture:(Texture *) tex;
-(void)mtlTexSubImage:(GLMContext)glm_ctx tex:(Texture *)tex buf:(Buffer *)buf src_offset:(size_t)src_offset src_pitch:(size_t)src_pitch src_image_size:(size_t)src_image_size src_size:(size_t)src_size slice:(GLuint)slice level:(GLuint)level width:(size_t)width height:(size_t)height depth:(size_t)depth xoffset:(size_t)xoffset yoffset:(size_t)yoffset zoffset:(size_t)zoffset;
-(void) mtlDrawArrays: (GLMContext) ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count;
-(void) mtlDrawElements: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices;
-(void) mtlDrawRangeElements: (GLMContext) glm_ctx mode:(GLenum) mode start:(GLuint) start end:(GLuint) end count: (GLsizei) count type: (GLenum) type indices:(const void *)indices;
-(void) mtlDrawArraysInstanced: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount;
-(void) mtlDrawElementsInstanced: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount;
-(void) mtlDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex;
-(void) mtlDrawRangeElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode start: (GLuint) start end: (GLuint) end type: (GLenum) type indices:(const void *)indices basevertex:(GLint) basevertex;
-(void) mtlDrawElementsInstancedBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count:(GLuint) count type: (GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex;
-(void) mtlDrawArraysIndirect: (GLMContext) glm_ctx mode:(GLenum) mode indirect: (const void *) indirect;
-(void) mtlDrawElementsIndirect: (GLMContext) glm_ctx mode:(GLenum) mode type:(GLenum) type indirect: (const void *) indirect;
-(void) mtlDrawArraysInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode first: (GLint) first count: (GLsizei) count instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance;
-(void) mtlDrawElementsInstancedBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount baseinstance:(GLuint) baseinstance;
-(void) mtlDrawElementsInstancedBaseVertexBaseInstance: (GLMContext) glm_ctx mode:(GLenum) mode count: (GLsizei) count type:(GLenum) type indices:(const void *)indices instancecount:(GLsizei) instancecount basevertex:(GLint) basevertex baseinstance:(GLuint) baseinstance;
-(void) mtlMultiDrawArrays: (GLMContext)glm_ctx mode:(GLenum) mode first:(const GLint *)first count:(const GLsizei *)count drawcount:(GLsizei) drawcount;
-(void) mtlMultiDrawElements: (GLMContext)glm_ctx mode:(GLenum) mode count:(const GLsizei *)count type:(GLenum)type indices:(const void *const*)indices drawcount:(GLsizei) drawcount;
-(void) mtlMultiDrawElementsBaseVertex: (GLMContext) glm_ctx mode:(GLenum) mode count: (const GLsizei *) count type: (GLenum) type indices:(const void *const *)indices drawcount:(GLsizei) drawcount basevertex:(const GLint *) basevertex;
-(void) mtlMultiDrawArraysIndirect: (GLMContext)glm_ctx mode:(GLenum) mode indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride;
-(void) mtlMultiDrawElementsIndirect: (GLMContext)glm_ctx mode:(GLenum) mode type:(GLenum)type indirect:(const void *)indirect drawcount:(GLsizei) drawcount stride:(GLsizei)stride;
- (void) bindObjFuncsToGLMContext: (GLMContext) glm_ctx;
- (void) createMGLRendererAndBindToContext: (GLMContext) glm_ctx view: (NSView *) view;
- (MTLCaptureDescriptor *)setupCaptureToFile: (id<MTLDevice>)device;
- (void)startCapture:(MTLCaptureDescriptor *) descriptor;
- (void)stopCapture;
- (void)dealloc;
- (BOOL)validateMetalObjects;
- (BOOL)recoverFromMetalError:(NSError *)error operation:(NSString *)operation;
- (void)clearTextureCache;
- (void)cleanupCommandBuffer;
- (void)resetMetalState;
- (void)commitCommandBufferWithAGXRecovery:(id<MTLCommandBuffer>)commandBuffer;
- (BOOL)shouldSkipGPUOperations;
- (void)clearProblematicGPUState;
- (void)enableMinimalFunctionalityMode;
- (void)recordGPUError;
- (void)recordGPUSuccess;
- (NSUInteger)getOptimalAlignmentForPixelFormat:(MTLPixelFormat)format;

/* geom r60: the geometry-emulation bind paths (MGLRenderer+Geometry.m) share the
 * renderer's 1x1 black fallback for GL 8.17 incomplete-texture sampling */
- (id<MTLTexture>) computeFallbackTexture;
/* GL 8.17 incomplete-texture fallback matching the MSL slot's texture type */
- (id<MTLTexture>) computeFallbackTextureForType: (MTLTextureType)type;
- (id<MTLSamplerState>) computeFallbackSampler;
@end

/* Methods shared across the renderer but implemented in its domain categories
 * (MGLRenderer+Geometry.m, +Tessellation.m, +Program.m, +Draw.m).  They are
 * declared in a category of their own, not in the class extension above: a
 * class-extension declaration promises a definition in the primary
 * @implementation (MGLRenderer.m), which made clang warn twice per method
 * (-Wincomplete-implementation there, -Wobjc-protocol-method-implementation
 * at the category definition). */
@interface MGLRenderer (DomainMethods)
/* geometry domain (MGLRenderer+Geometry.m): Metal requires inputPrimitiveTopology when the
 * vertex stage writes render_target_array_index / viewport_array_index */
- (void) geomConfigureTopology: (MTLRenderPipelineDescriptor *) pd;
/* geom: the Metal texture an image unit (glBindImageTexture) must expose to a shader —
 * a view selecting the unit's mip level and, for a non-layered binding, its layer;
 * returns `base` unchanged when no slicing is needed.  Shared by the render, compute,
 * geometry and tessellation image-binding paths. */
- (id<MTLTexture>) geomImageUnitView: (const ImageUnit *) iu base: (id<MTLTexture>) base;
/* geom: non-layered image2D bindings of 3D textures alias one z-slice through
 * a 2D shadow plane (Metal has no 3D<->2D view/blit).  Sync at draw start,
 * flush pending stores at commit and before reads of the 3D texture. */
- (void) geom3DImageShadowSync;
- (void) geom3DImageShadowFlush;
- (void) geom3DImageShadowFlushTexture: (Texture *) tex;
/* geom: like geomImageUnitView, then reinterpret the texels as the image unit's `format`
 * argument (glBindImageTexture) through a Metal pixel-format view when it differs from
 * the texture's internal format — GL 4.6 §8.26 lets a unit read, say, an R32I texture as
 * rg16f; elem_class (SpirvResource.elem_class, 0xffffffff unknown) keeps the plain view
 * when the SPIR-V image's sampled type disagrees with that format */
- (id<MTLTexture>) geomImageUnitViewRes: (const ImageUnit *) iu base: (id<MTLTexture>) base
                              elemClass: (GLuint) elem_class;
/* geom: realize a stage's sampled images and image units (deferred texture uploads) while
 * NO encoder is open on the current command buffer — bindMTLTexture's upload path opens a
 * blit encoder, which Metal forbids while an encoder is encoding.  Call before the first
 * encoder of a draw that binds stage textures after opening it (tess kernel path) */
- (void) geomRealizeStageTextures: (Program *) program stage: (int) stage;
/* geom: shared 1x1 zero texture for an image unit bound to an incomplete texture
 * (no image): reads are zero and stores drop, GL 4.6 §8.26 */
- (id<MTLTexture>) geomImageEmptyTexture: (GLenum) internalformat;
/* geom: disable rasterization when the pipeline's vertex stage is void (no outputs) —
 * Metal rejects a rasterizing pipeline for such functions (see the implementation) */
- (void) geomRasterizationForVertexStage: (MTLRenderPipelineDescriptor *) desc;
/* geom: SPIRV-Cross's image-atomic emulation — bind one zero-initialized sidecar
 * MTLBuffer per image unit at the `<name>_atomic` argument slots the stage's storage
 * images reflected (SpirvResource.msl_atomic; see mgl_msl_fix_image_atomics).  Called
 * from bindTextures / bindTexturesToComputeEncoder after the image-unit textures. */
- (bool) geomBindImageAtomicBuffersRender: (int) stage;
- (bool) geomBindImageAtomicBuffersCompute: (id <MTLComputeCommandEncoder>) encoder;
/* geom: create + seed the image-atomic sidecars of the pending compute dispatch
 * (own compute pass, before the dispatch's encoder is opened) */
- (void) geomPrepareImageAtomicsCompute;
- (void) geomPrepareImageAtomicsRender;   /* geom: seed MS image sidecars pre-encoder */
/* geom: the (program, stage) twins for the tessellation emulation's VS / TCS / TES
 * kernels, which run as compute with their own stage's resource list */
- (bool) geomBindStageImageAtomicsCompute: (Program *) program
                                   encoder: (id <MTLComputeCommandEncoder>) encoder
                                     stage: (int) stage;
- (void) geomPrepareStageImageAtomicsCompute: (Program *) program stage: (int) stage;
- (id<MTLBuffer>) geomImageAtomicBuffer: (GLuint) unit view: (id<MTLTexture>) view;
- (void) geomReleaseImageAtomics;
- (id<MTLComputePipelineState>) geomAtomResolvePso: (MTLTextureType) tt format: (MTLPixelFormat) pf;
- (void) geomResolveImageAtomics;
/* geom r88: DEFINE a freshly created multisample texture's contents as zero —
 * Metal leaves private MS storage undefined, and the image-atomic sidecar seed
 * (and any unwritten-texel read) must not observe recycled garbage.  Called
 * from texextCreateMultisampleTexture (MGLRenderer+TextureExt.m). */
- (void) geomDefineMultisampleTexture: (id<MTLTexture>) t;
/* geom: SSBO runtime-array .length() — bind SPIRV-Cross's spvBufferSizeConstants
 * (bound-range byte size of the SSBO at each Metal buffer index) to the render or
 * compute tables; no-op when the stage MSL does not reference it */
- (void) geomBindSSBOSizes: (BufferMapList *) list stage: (int) stage vertex: (bool) vertex;
- (void) geomBindSSBOSizesCompute: (id <MTLComputeCommandEncoder>) enc;
/* geom: a cached empty fragment function (no outputs) used for vertex-only pipelines —
 * Apple GPUs reject rasterizationEnabled=NO with a non-void vertex function, so a
 * program without a fragment stage renders nothing (no attachments) through it while
 * its vertex-stage side effects (SSBO / image / atomic writes) still run */
- (id<MTLFunction>) geomEmptyFragmentFunction;
/* geom: when the pass-through vertex override is active, a specialized copy of it (the same
 * rstate vertex function constants — it is the last pre-rasterization stage); NULL when the
 * override is inactive, so the caller leaves the descriptor's vertex function alone */
- (id<MTLFunction>) geomSpecializeVertexOverride;
/* geom: the vertex stage's shadow gather (spirv[VS].shadow_slots) reads the compare
 * function of the unit as a function constant, see geomShadowFCValues below *//* geom: fill the shadow-compare function constants (32010+slot) of a stage's MSL from the
 * GL_TEXTURE_COMPARE_FUNC of the units feeding its shadow-sampled texture slots, and mix a
 * checksum of the values into *hash (the rstate specialization cache key must cover them) */
- (void) geomShadowFCValues:(MTLFunctionConstantValues *)cv stage:(int)stage hash:(uint64_t *)hash;
/* geom: re-specialize the vertex function when a shadow-sampled unit's compare function
 * changed since the last specialization (called on the texture-binding path, which runs
 * before the PSO build of the same draw, so dirtying the program takes effect there) */
- (void) geomShadowCheckStale:(GLuint)unit texture:(Texture *)tex slot:(GLuint)slot;
/* program domain: MSL-source -> MTLLibrary cache (compileShader: uses it) */
- (id<MTLLibrary>) extProgramLibraryForMSL: (const char *)msl key: (NSString *)key error: (char **)error_log;
/* tess domain: precise-math options for MSL carrying the fp64 df64 library */
/* tess r70/r71 — aliased wide lines through the geometry raster pass: the geometry
 * kernel's flat-word line records are widened into corner records by the shared
 * expansion kernel (mgl_ext_tess.h).  extTessWidelineGeomReserve sizes + acquires
 * the ring block BEFORE the draw's first encoder opens (a ring acquire may commit
 * the current command buffer; committing with an open encoder wedges the ring) and
 * extTessWidelineGeomPrepare dispatches the kernel into the geometry kernel's own
 * encoder (dispatches are ordered only within one encoder).  The raster pass then
 * swaps prim to Triangle and binds the substitute buffers; extTessWidelineGeomRetire
 * releases the ring slot. */
- (bool) extTessWidelineGeomReserve: (void *) gp idxCap: (NSUInteger) idx_cap;
- (bool) extTessWidelineGeomPrepare: (void *) gp
                               outv: (id<MTLBuffer>) outv outvOff: (size_t) outvOff
                                idx: (id<MTLBuffer>) idx ctl: (id<MTLBuffer>) ctl
                                enc: (id<MTLComputeCommandEncoder>) enc;
- (void) extTessWidelineGeomRetire: (int) slot;
- (MTLCompileOptions *) mslCompileOptionsForSource: (NSString *) src;
/* tess domain (MGLRenderer+Tessellation.m): true when the current program/pipeline has
 * a tessellation evaluation shader — the render pipeline's vertex function is then the
 * (rewritten) TES and the vertex stage runs as a compute kernel before the render pass */
- (bool) extTessActive;
/* r153: per-buffer pool of retired Metal storages (MGLRenderer+Draw.m) */
- (id<MTLBuffer>) extBufPoolTake: (Buffer *) buf length: (NSUInteger) length options: (MTLResourceOptions) options;
- (void) extTessBindDrawParams: (id<MTLRenderCommandEncoder>) enc;   /* gl_DrawID (r28) */
- (MTLVertexDescriptor *) extTessVertexDescriptor;
- (void) extTessConfigurePipeline: (MTLRenderPipelineDescriptor *) desc;
/* inputPrimitiveTopology for GS-less pipelines whose final pre-raster stage writes
 * gl_Layer / gl_ViewportIndex (ARB_shader_viewport_layer_array): Metal requires it
 * once the vertex function has [[render_target_array_index]] / [[viewport_array_index]] */
- (void) tessConfigureLayeredTopology: (MTLRenderPipelineDescriptor *) pd;
/* geom r60: the compute bind path binds samplers at their own MSL index */
- (NSUInteger) geomComputeSamplerSlotForImage: (const char *)img_name textureSlot: (NSUInteger)tslot;
@end

/* texparam domain: texture/sampler parameters on Metal (MGLRenderer+TextureParams.m) */
@interface MGLRenderer (TextureParams)
- (id<MTLSamplerState>) texparamCreateSampler:(TextureParameter *)p target:(GLenum)target;
- (id<MTLSamplerState>) texparamSamplerForUnit:(GLuint)unit texture:(Texture *)tex;
/* texparam: the sampler state cached in a TextureParameter's mtl_data — which holds an
 * MGLTexParamCache (not a bare sampler) once texparamCacheFor has run; casting blindly
 * is a type-confusion crash */
- (id<MTLSamplerState>) texparamCachedSampler:(TextureParameter *)p;
- (id<MTLTexture>) texparamTextureForTex:(Texture *)tex base:(id<MTLTexture>)base;
- (void) texparamPrepareTextureDescriptor:(MTLTextureDescriptor *)desc forTex:(Texture *)tex;
- (MTLTextureSwizzleChannels) texparamSwizzleForParams:(TextureParameter *)p;
@end

MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

#endif /* MGLRenderer_Private_h */
