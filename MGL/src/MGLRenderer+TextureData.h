/* MGLRenderer+TextureData.h — 'texdata' domain category interface (owned by agent 'texdata'). */
#ifndef MGLRenderer_TextureData_h
#define MGLRenderer_TextureData_h

#import "MGLRenderer_Private.h"

@interface MGLRenderer (TextureData)
/* realize (create / reshape / upload) a regular texture and its sampler; false if the
 * texture is not handled by texdata (buffer, view, multisample, renderbuffer) */
- (bool) texdataBindMTLTexture:(Texture *)tex;
- (bool) texdataRealize:(Texture *)tex;
@end

/* implemented in MGLRenderer+TextureData.m's (TextureDataBinding) block */
@interface MGLRenderer (TextureDataBinding)
/* texdata: publish device-dependent format caps (ETC2/BC CPU decode, ASTC) to the
 * texdata format table; MGLRenderer.m calls it once _device exists, because the
 * auto-called bindExt* binders run before the device is attached */
- (void) texdataSetDeviceCaps;
@end

#endif
