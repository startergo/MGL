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

/* GL_EXT_texture_sRGB_decode (compat r102): SKIP_DECODE sampling binds a plain
 * (non-sRGB) view of the texture's storage.  Called right after
 * texparamTextureForTex at every sampled-image bind site; returns <base>
 * unchanged unless the unit's effective decode state is GL_SKIP_DECODE_EXT. */
@interface MGLRenderer (CompatSRGBDecode)
- (id<MTLTexture>) compatSRGBDecodeViewForTex: (Texture *)tex base: (id<MTLTexture>)base unit: (GLuint)unit;
@end

/* C bridge (GL-side buffer write paths etc.) — no-op unless the env flag is on */
void mgl_compat_cs_drain(GLMContext glm_ctx);

/* compat_ssbo.c — aliased shader-storage-block support (GL 4.6 §7.12): the
 * pairs of the stage's storage-block MSL arguments that are bound to the
 * identical buffer range right now (0 = nothing aliases, -1 = too many), and
 * the MSL rewrite folding each secondary block onto the primary's pointer
 * (returns NULL when the module cannot be rewritten — caller keeps the base
 * function, i.e. the documented no-alias behaviour) */
int  mgl_compat_ssbo_alias_pairs(GLMContext ctx, BufferMapList *map,
                                 unsigned *from_msl, unsigned *to_msl, int max_pairs);
char *mgl_compat_msl_alias_ssbo(const char *msl, const char *entry,
                                const unsigned *from_msl, const unsigned *to_msl,
                                int npairs);

/* MGL_NO_PIPE_FAST=1 disables the compat pipeline fast path (debug switch; the
 * getenv happens once, this is the draw path) */
int mglNoPipeFast(void);
/* function-pointer -> MSL content id (MGLRenderer+Program.m): the content
 * identity the pipeline-cache key hashes instead of raw function addresses */
uint64_t mglProgramFunctionId(void *fn);
/* register a program stage function's MSL content identity (call wherever a
 * compiled function is installed on a program — see MGLRenderer+Program.m) */
void mglProgramRegisterMSLIdentity(const void *function, const char *msl, const char *entry);

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
/* compat r100c: compute-function variant whose ALIASED shader storage blocks
 * (two blocks bound to the SAME buffer range, GL 4.6 §7.12) share one Metal
 * pointer — Metal assumes distinct [[buffer(n)]] args never alias and would
 * forward the first block's store to the later load inside one invocation.
 * nil when no storage blocks alias (costs one small C walk; the variants are
 * cached per signature on the base function).  See compat_ssbo.c. */
- (id<MTLFunction>) compatSSBOAliasComputeFunction: (Shader *) cs program: (Program *) program
                                                                   map: (BufferMapList *) map;
/* perf session 13: uniform/storage buffer-map prefix cache.  mapBuffersToMTL
 * calls compatBindmapFast first — true means both stage lists' prefix was
 * served from the cache (only the vertex-stage attribute suffix re-ran);
 * false means the caller runs the full enumeration and then calls
 * compatBindmapStore to re-arm the cache. */
- (bool) compatBindmapFast: (int) vertex_stage;
- (void) compatBindmapStore: (int) vertex_stage;
/* perf session 12 verdicts of compatDrawGate: */
enum { MGL_DF_FULL = 0, MGL_DF_SKIP, MGL_DF_BUFFERS, MGL_DF_RSTATE, MGL_DF_REBIND,
       MGL_DF_TEXTURES /* r154: only sampled-texture bindings moved */ };

/* perf session 12: per-draw state-walk fast path.  extBufBeginDraw calls
 * compatDrawGate first: when every state group the walk reads still matches
 * the exact-content snapshot taken by the last full walk (compatDrawLatch),
 * the whole walk is a proven no-op and is skipped; when only listed-buffer
 * content changed, compatDrawFastBuffers re-streams just the buffers.
 * extBufEndDraw does the same for the GPU-serial stamps. */
- (int)  compatDrawGate: (bool) allowDirtyListBuffers;
- (bool) compatDrawFastBuffers;
- (bool) compatDrawFastRebind;

/* compat perf (session 19): true when the compat shadow already has this
 * (buffer, offset) bound at this index on this encoder — used by
 * extBufBindList to skip the whole storage/ring dispatch for unchanged slots */
bool compatVbufAlreadyBound(GLMContext ctx, unsigned idx, const void *mtl, size_t offset, void *enc_ptr);
- (bool) compatDrawFastRState;
/* r154: texture-binding-only partial walk; false = run the full walk */
- (bool) compatDrawFastTextures;
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

/* compat: GL_KHR_blend_equation_advanced (GL 4.6 core §17.3.8) — when an
 * advanced equation is active, the rewritten fragment variant that composites
 * via a [[color(0)]] framebuffer fetch (compat_advblend.c).  Returns false
 * when the normal fixed-function path should be kept (no Metal work then). */
- (bool) compatAdvancedBlendVariant: (Shader *)shader
                                msl: (const char **)out_msl
                            library: (id<MTLLibrary> *)out_lib
                          function: (id<MTLFunction> *)out_fn;
@end

#endif
