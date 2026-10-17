/* MGLRenderer+Framebuffer.h — fb-domain renderer category (MGLRenderer+Framebuffer.m). Owned by agent 'fb'. */
#ifndef MGLRenderer_Framebuffer_h
#define MGLRenderer_Framebuffer_h

#import "MGLRenderer_Private.h"
#include "mgl_ext_fb.h"

@interface MGLRenderer (Framebuffer)
- (void) bindExtFramebufferFuncsToGLMContext: (GLMContext) glm_ctx;
/* render pass for the bound draw framebuffer -> _renderPassDescriptor (called by -newRenderEncoder) */
- (bool) fbBuildRenderPassDescriptor;
/* attachment pixel formats + raster sample count matching that render pass (called by -generatePipelineDescriptor) */
- (void) fbApplyPipelineFormats: (MTLRenderPipelineDescriptor *) desc;
/* default framebuffer creation + initial viewport/scissor (called once at renderer creation) */
- (void) fbInitDefaultFramebuffer;
- (id<MTLTexture>) fbEnsureDefaultFramebuffer;
- (void) fbDefaultSizeW: (GLuint *) w h: (GLuint *) h;
- (void) fbClear: (const MGLFBClearRequest *) req;
- (int)  fbRead: (const MGLFBReadRequest *) req;
- (void) fbBlit: (const MGLFBBlitRequest *) req;
- (void) fbSwapBuffers;
- (void) fbFlush: (bool) finish;
- (bool) fbRenderbufferStorage: (MGLRenderbufferEx *) rb;
- (bool) fbSamplePosition: (GLuint) samples index: (GLuint) index out: (GLfloat *) xy;
- (GLuint) fbRoundSamples: (GLuint) samples;
@end

#endif
