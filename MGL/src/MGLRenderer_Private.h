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
    NSMutableArray *_proactiveTextures;

    MGLDrawable _drawBuffers[_MAX_DRAW_BUFFERS];

    MTLBlendFactor _src_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_rgb_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _src_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendFactor _dst_blend_alpha_factor[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _rgb_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLBlendOperation _alpha_blend_operation[MAX_COLOR_ATTACHMENTS];
    MTLColorWriteMask _color_mask[MAX_COLOR_ATTACHMENTS];

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

/* geometry domain (MGLRenderer+Geometry.m): Metal requires inputPrimitiveTopology when the
 * vertex stage writes render_target_array_index / viewport_array_index */
- (void) geomConfigureTopology: (MTLRenderPipelineDescriptor *) pd;
/* geom: when the pass-through vertex override is active, a specialized copy of it (the same
 * rstate vertex function constants — it is the last pre-rasterization stage); NULL when the
 * override is inactive, so the caller leaves the descriptor's vertex function alone */
- (id<MTLFunction>) geomSpecializeVertexOverride;

- (void) bindMTLBuffer:(Buffer *) ptr;- (bool) mapGLBuffersToMTLBufferMap:(BufferMapList *)buffer_map stage: (int) stage;
- (bool) mapBuffersToMTL;
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
-(void)mtlBlitFramebuffer:(GLMContext)glm_ctx srcX0:(size_t)srcX0 srcY0:(size_t)srcY0 srcX1:(size_t)srcX1 srcY1:(size_t)srcY1 dstX0:(size_t)dstX0 dstY0:(size_t)dstY0 dstX1:(size_t)dstX1 dstY1:(size_t)dstY1 mask:(size_t)mask filter:(GLuint)filter;
- (Texture *)framebufferAttachmentTexture: (FBOAttachment *)fbo_attachment;
- (bool)bindMTLTexture:(Texture *)tex;
- (bool)bindActiveTexturesToMTL;
- (bool)bindFramebufferTexture:(FBOAttachment *)fbo_attachment isDrawBuffer:(bool) isDrawBuffer;
- (int) getProgramBindingCount: (int) stage type: (int) type;
- (int) getProgramBinding: (int) stage type: (int) type index: (int) index;
- (int) getProgramLocation: (int) stage type: (int) type index: (int) index;
- (id<MTLLibrary>) compileShader: (const char *) str;
-(bool)bindMTLProgram:(Program *)ptr;
- (id)newDrawBuffer:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil;
- (id)newDrawBufferWithCustomSize:(MTLPixelFormat)pixelFormat isDepthStencil:(bool)depthStencil customSize:(CGSize)size;
- (bool) checkDrawBufferSize:(GLuint) index;
- (MTLStencilOperation) mtlStencilOpForGLOp:(GLenum) op;
- (void) updateCurrentRenderEncoder;
- (bool) newRenderEncoder;
- (bool) newCommandBuffer;
- (bool) newCommandBufferAndRenderEncoder;
-(MTLRenderPipelineDescriptor *)generatePipelineDescriptor;
- (MTLVertexDescriptor *)generateVertexDescriptor;
- (MTLBlendFactor) blendFactorFromGL:(GLenum)gl_blend;
- (MTLBlendOperation) blendOperationFromGL:(GLenum)gl_blend_op;
- (void) updateBlendStateCache;
-(void)bindBlendStateToPipelineStateDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor;
-(bool)bindFramebufferAttachmentTextures;
- (void) endRenderEncoding;
- (void) emergencyResetMetalState;
- (bool) processGLState: (bool) draw_command;

/* tess domain (MGLRenderer+Tessellation.m): true when the current program/pipeline has
 * a tessellation evaluation shader — the render pipeline's vertex function is then the
 * (rewritten) TES and the vertex stage runs as a compute kernel before the render pass */
- (bool) extTessActive;
- (MTLVertexDescriptor *) extTessVertexDescriptor;
- (void) extTessConfigurePipeline: (MTLRenderPipelineDescriptor *) desc;
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
-(void) mtlReadDrawable:(GLMContext) glm_ctx pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region;
-(void) mtlGetTexImage:(GLMContext) glm_ctx tex: (Texture *)tex pixelBytes:(void *)pixelBytes bytesPerRow:(NSUInteger)bytesPerRow bytesPerImage:(NSUInteger)bytesPerImage fromRegion:(MTLRegion)region mipmapLevel:(NSUInteger)level slice:(NSUInteger)slice;
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
- (void)createProactiveTextures;
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

@end

/* texparam domain: texture/sampler parameters on Metal (MGLRenderer+TextureParams.m) */
@interface MGLRenderer (TextureParams)
- (id<MTLSamplerState>) texparamCreateSampler:(TextureParameter *)p target:(GLenum)target;
- (id<MTLSamplerState>) texparamSamplerForUnit:(GLuint)unit texture:(Texture *)tex;
- (id<MTLTexture>) texparamTextureForTex:(Texture *)tex base:(id<MTLTexture>)base;
- (void) texparamPrepareTextureDescriptor:(MTLTextureDescriptor *)desc forTex:(Texture *)tex;
- (MTLTextureSwizzleChannels) texparamSwizzleForParams:(TextureParameter *)p;
@end

MTLVertexFormat glTypeSizeToMtlType(GLuint type, GLuint size, bool normalized);

#endif /* MGLRenderer_Private_h */
