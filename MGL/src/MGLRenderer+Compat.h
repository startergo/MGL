/* MGLRenderer+Compat.h — compat-domain renderer category (MGLRenderer+Compat.m) */
#ifndef MGLRenderer_Compat_h
#define MGLRenderer_Compat_h

#import "MGLRenderer_Private.h"

/* compat perf session 10: single encoder (CS) worker thread
 * (MGLRenderer+CompatCS.m).  MGL_ENCODER_THREAD=1 makes newRenderEncoder()
 * install a recording proxy as _currentRenderEncoder: every encoder method the
 * shared code calls becomes a compact packet replayed in order by one worker
 * thread, so GL-thread time is not spent inside Metal encoding.  Points that
 * must touch Metal directly (encoder/encoder teardown, new command buffer,
 * commit, compute/blit encoders, geometry/tess passes, renderer teardown) call
 * compatCSDrain first; buffer writes keep their record-time serial stamping, so
 * the existing wait/rename tracking orders them against deferred draws. */
@interface MGLRenderer (CompatCS)
- (bool) compatCSOn;
- (id<MTLRenderCommandEncoder>) compatCSEncoderProxyFor: (id<MTLRenderCommandEncoder>) real;
- (void) compatCSDrain;
- (void) compatCSFrameStats;
- (void) compatCSStop;
@end

/* C bridge (GL-side buffer write paths etc.) — no-op unless the env flag is on */
void mgl_compat_cs_drain(GLMContext glm_ctx);

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
/* perf session 13: uniform/storage buffer-map prefix cache.  mapBuffersToMTL
 * calls compatBindmapFast first — true means both stage lists' prefix was
 * served from the cache (only the vertex-stage attribute suffix re-ran);
 * false means the caller runs the full enumeration and then calls
 * compatBindmapStore to re-arm the cache. */
- (bool) compatBindmapFast: (int) vertex_stage;
- (void) compatBindmapStore: (int) vertex_stage;
/* perf session 12 verdicts of compatDrawGate: */
enum { MGL_DF_FULL = 0, MGL_DF_SKIP, MGL_DF_BUFFERS, MGL_DF_RSTATE };

/* perf session 12: per-draw state-walk fast path.  extBufBeginDraw calls
 * compatDrawGate first: when every state group the walk reads still matches
 * the exact-content snapshot taken by the last full walk (compatDrawLatch),
 * the whole walk is a proven no-op and is skipped; when only listed-buffer
 * content changed, compatDrawFastBuffers re-streams just the buffers.
 * extBufEndDraw does the same for the GPU-serial stamps. */
- (int)  compatDrawGate: (bool) allowDirtyListBuffers;
- (bool) compatDrawFastBuffers;
- (bool) compatDrawFastRState;
- (void) compatDrawLatch;
- (bool) compatEndDrawFast;
- (void) compatEndDrawLatch;
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
