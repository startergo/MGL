/* MGLRenderer+TextureExt.h — 'texext' domain category interface (owned by agent 'texext').
 * Hook methods called from MGLRenderer.m (texture creation / binding / pipeline set-up). */
#ifndef MGLRenderer_TextureExt_h
#define MGLRenderer_TextureExt_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer (TextureExt)
/* true if texext creates the Metal object for this texture (buffer textures, views, multisample) */
- (BOOL) texextOwnsTexture: (Texture *) tex;
- (id<MTLTexture>) texextCreateMTLTexture: (Texture *) tex;
/* adjust a descriptor built by -createMTLTextureFromGLTexture: (e.g. PixelFormatView usage for view sources) */
- (void) texextAdjustTextureDescriptor: (MTLTextureDescriptor *) desc forTexture: (Texture *) tex;
/* mark tex dirty if its Metal object no longer matches its source (buffer reallocated, view root recreated) */
- (void) texextValidateTexture: (Texture *) tex;
/* rasterSampleCount for multisample framebuffer attachments */
- (void) texextAdjustPipelineDescriptor: (MTLRenderPipelineDescriptor *) desc;
@end

#endif
