/* MGLRenderer+Gpuaudit.h — gpuaudit renderer category entry points used by other files.
 * Owned by agent 'gpuaudit' (see mgl_ext_gpuaudit.h). */
#ifndef MGLRenderer_Gpuaudit_h
#define MGLRenderer_Gpuaudit_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer (Gpuaudit)
/* render pipeline state cache (called from -processGLState:) */
- (id<MTLRenderPipelineState>) gpuauditPipelineForDescriptor:(MTLRenderPipelineDescriptor *)desc;
- (void) gpuauditStorePipeline:(id<MTLRenderPipelineState>)ps forDescriptor:(MTLRenderPipelineDescriptor *)desc;

@end

#endif
