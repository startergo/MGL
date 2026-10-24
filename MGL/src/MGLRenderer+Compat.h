/* MGLRenderer+Compat.h — compat-domain renderer category (MGLRenderer+Compat.m) */
#ifndef MGLRenderer_Compat_h
#define MGLRenderer_Compat_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer (Compat)
/* IOSurface present path (CGL-like async swap): enabled per renderer with
 * MGL_SURFACE_PRESENT=1 and only while a windowed Metal layer exists; see
 * MGLRenderer+Compat.m.  compatSurfacePresent: blits the default framebuffer
 * into a pooled IOSurface texture, commits and shows it with layer.contents —
 * returns true when it handled the frame (the drawable path must be skipped). */
- (bool) compatSurfacePresentEnabled;
- (id<MTLTexture>) compatSurfaceWrite:(NSUInteger)w h:(NSUInteger)h;
- (bool) compatPipelineBindNeeded;
- (bool) compatPipelineFastPath;
- (void) compatPipelineStore;
- (bool) compatSurfacePresent:(id<MTLTexture>)color;
/* perf session 3: encoder bind shadow + streaming upload ring (see
 * mgl_compat_perf.h).  Every per-draw Metal argument bind in the shared draw
 * path routes through these: an identical bind already present in the current
 * encoder is skipped, and small byte payloads (uniform blocks, client arrays,
 * constant attribs, copied indices) stream through the ring with offset-only
 * updates.  MGL_BIND_SHADOW=0 disables both (kill switch). */
- (void) compatBindVertexBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)offset atIndex:(NSUInteger)idx;
- (void) compatBindFragmentBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)offset atIndex:(NSUInteger)idx;
- (void) compatSetVertexTexture:(id<MTLTexture>)tex atIndex:(NSUInteger)idx;
- (void) compatSetFragmentTexture:(id<MTLTexture>)tex atIndex:(NSUInteger)idx;
- (void) compatSetVertexSamplerState:(id<MTLSamplerState>)samp atIndex:(NSUInteger)idx;
- (void) compatSetFragmentSamplerState:(id<MTLSamplerState>)samp atIndex:(NSUInteger)idx;
- (void) compatRingVertexBytes:(const void *)p length:(NSUInteger)len atIndex:(NSUInteger)idx;
- (void) compatRingFragmentBytes:(const void *)p length:(NSUInteger)len atIndex:(NSUInteger)idx;
- (id<MTLBuffer>) compatRingIndexBytes:(const void *)p length:(NSUInteger)len offset:(NSUInteger *)out_off;
- (void) compatBindInvalidate;
- (void) compatBindProfileDump;
- (void) compatBindProfileReset;
/* perf session 8: per-draw texture-resolution cache — bindTexturesToCurrentRenderEncoder
 * calls compatTexResolveHit before the validate/realize/sampler/view chain and
 * compatTexResolveRecord after it (see mgl_compat_perf.h) */
- (bool) compatTexResolveHit:(bool)frag unit:(GLuint)unit slot:(GLuint)slot tex:(Texture *)tex;
- (void) compatTexResolveRecord:(bool)frag unit:(GLuint)unit slot:(GLuint)slot tex:(Texture *)tex
                           view:(id<MTLTexture>)view sampler:(id<MTLSamplerState>)samp;
@end

#endif
