/* MGLRenderer+RenderState.h — rstate category entry points used by MGLRenderer.m. Owned by 'rstate'. */
#import "MGLRenderer_Private.h"

@interface MGLRenderer (RenderState)
/* apply all GL encoder-level state (viewport, scissor, depth/stencil, raster) to enc */
- (void) rstateApplyEncoderState:(id<MTLRenderCommandEncoder>) enc;
/* blend, masks, rasterizer discard, default-FB attachment formats, shader fixup specialization */
- (void) rstateConfigurePipelineDescriptor:(MTLRenderPipelineDescriptor *)desc;
/* compat perf: content hash of everything baked into the pipeline (PSO cache key) */
- (uint64_t) rstatePipelineContentHash;
@end
