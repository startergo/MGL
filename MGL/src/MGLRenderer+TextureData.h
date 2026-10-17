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

#endif
