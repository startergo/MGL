/*
 * MGLRenderer+TextureParams.m — texparam domain: Metal side of texture
 * parameters, sampler objects and format capability queries.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * GL -> Metal mapping
 *   sampler state  -> MTLSamplerState (cached in Sampler->mtl_data, or in the
 *                     MGLTexParamCache stored in Texture->params.mtl_data)
 *   swizzle, BASE/MAX_LEVEL, DEPTH_STENCIL_TEXTURE_MODE
 *                  -> texture view of the bound texture (level range, swizzle,
 *                     X32_Stencil8 / X24_Stencil8 pixel format), cached likewise
 *   border color   -> nearest of Metal's transparent black / opaque black / opaque white
 *   LOD bias       -> not representable in MTLSamplerDescriptor (tracked + queryable only)
 */

#import "MGLRenderer_Private.h"
#include "mgl_ext_texparam.h"
#include "tex_param_internal.h"

@interface MGLTexParamCache : NSObject
@property (nonatomic, strong) id<MTLSamplerState> sampler;
@property (nonatomic, strong) id<MTLTexture> view;
@property (nonatomic) uintptr_t viewParent;
@property (nonatomic) uint64_t viewKey;
@end

@implementation MGLTexParamCache
@end

@interface MGLRenderer (TextureParamsPrivate)
- (id<MTLDevice>) texparamDevice;
@end

#pragma mark C hooks

static MTLTextureSwizzle texparamSwizzle(GLenum s, MTLTextureSwizzle identity)
{
    switch (s)
    {
        case GL_RED:   return MTLTextureSwizzleRed;
        case GL_GREEN: return MTLTextureSwizzleGreen;
        case GL_BLUE:  return MTLTextureSwizzleBlue;
        case GL_ALPHA: return MTLTextureSwizzleAlpha;
        case GL_ZERO:  return MTLTextureSwizzleZero;
        case GL_ONE:   return MTLTextureSwizzleOne;
    }
    return identity;
}

static bool texparamIsInteger(MTLPixelFormat f)
{
    switch (f)
    {
        case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint:
        case MTLPixelFormatR16Uint: case MTLPixelFormatR16Sint:
        case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
        case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGB10A2Uint:
        case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint:
        case MTLPixelFormatStencil8:
        case MTLPixelFormatX32_Stencil8: case MTLPixelFormatX24_Stencil8:
            return true;
        default:
            return false;
    }
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static bool texparamIsCompressed(MTLPixelFormat f)
{
    return (f >= MTLPixelFormatBC1_RGBA && f <= MTLPixelFormatBC7_RGBAUnorm_sRGB) ||
           (f >= MTLPixelFormatPVRTC_RGB_2BPP && f <= MTLPixelFormatPVRTC_RGBA_4BPP_sRGB) ||
           (f >= MTLPixelFormatEAC_R11Unorm && f <= MTLPixelFormatETC2_RGB8A1_sRGB) ||
           (f >= MTLPixelFormatASTC_4x4_sRGB && f <= MTLPixelFormatASTC_12x12_HDR);
}
#pragma clang diagnostic pop

static bool texparamIsSRGB(MTLPixelFormat f)
{
    switch (f)
    {
        case MTLPixelFormatR8Unorm_sRGB: case MTLPixelFormatRG8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatBGRA8Unorm_sRGB:
            return true;
        default:
            return false;
    }
}

static bool texparamFormatCaps(GLMContext glm_ctx, GLuint mtl_pixel_format, MGLTexParamFormatCaps *caps)
{
    MGLRenderer *renderer = (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
    id<MTLDevice> dev = [renderer texparamDevice];
    MTLPixelFormat f = (MTLPixelFormat)mtl_pixel_format;

    bzero(caps, sizeof(*caps));

    if (!dev || f == MTLPixelFormatInvalid)
        return false;

    bool apple = [dev supportsFamily:MTLGPUFamilyApple1];
    bool compressed = texparamIsCompressed(f);
    bool depth = (f == MTLPixelFormatDepth16Unorm || f == MTLPixelFormatDepth32Float ||
                  f == MTLPixelFormatDepth24Unorm_Stencil8 || f == MTLPixelFormatDepth32Float_Stencil8);
    bool stencil = (f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth24Unorm_Stencil8 ||
                    f == MTLPixelFormatDepth32Float_Stencil8);
    bool integer = texparamIsInteger(f);
    bool f32 = (f == MTLPixelFormatR32Float || f == MTLPixelFormatRG32Float || f == MTLPixelFormatRGBA32Float);
    bool packed16 = (f == MTLPixelFormatB5G6R5Unorm || f == MTLPixelFormatA1BGR5Unorm ||
                     f == MTLPixelFormatABGR4Unorm || f == MTLPixelFormatBGR5A1Unorm);

    /* availability */
    if (f >= MTLPixelFormatBC1_RGBA && f <= MTLPixelFormatBC7_RGBAUnorm_sRGB)
    {
        if (![dev supportsBCTextureCompression]) return false;
    }
    else if (compressed || packed16)
    {
        if (!apple) return false;      /* ETC2/EAC/ASTC/PVRTC and 16-bit packed: Apple GPU families */
    }
    if (f == MTLPixelFormatDepth24Unorm_Stencil8 && ![dev isDepth24Stencil8PixelFormatSupported])
        return false;

    caps->supported = GL_TRUE;

    caps->filterable = !integer && !(f32 && ![dev supports32BitFloatFiltering]) &&
                       f != MTLPixelFormatStencil8;

    caps->color_renderable = !compressed && !depth && !stencil &&
                             !(f == MTLPixelFormatRGB9E5Float && !apple);
    caps->depth_renderable = depth;
    caps->stencil_renderable = stencil;
    caps->blendable = caps->color_renderable && !integer;
    caps->writable = caps->color_renderable && !texparamIsSRGB(f) && !packed16 && f != MTLPixelFormatRGB9E5Float;
    caps->atomic = (f == MTLPixelFormatR32Uint || f == MTLPixelFormatR32Sint) &&
                   ([dev supportsFamily:MTLGPUFamilyApple6] || [dev supportsFamily:MTLGPUFamilyMac2]);

    caps->msaa = (caps->color_renderable || depth || stencil) && !compressed;
    if (caps->msaa)
    {
        static const int counts[] = { 8, 4, 2 };
        for (int i = 0; i < 3; i++)
        {
            if ([dev supportsTextureSampleCount:counts[i]])
                caps->sample_counts[caps->num_sample_counts++] = counts[i];
        }
        if (caps->num_sample_counts == 0)
            caps->msaa = GL_FALSE;
    }

    return true;
}

static bool texparamTextureInfo(GLMContext glm_ctx, Texture *tex, MGLTexParamMtlInfo *info)
{
    bzero(info, sizeof(*info));

    if (!tex || !tex->mtl_data)
        return false;

    id obj = (__bridge id)tex->mtl_data;
    if (![obj conformsToProtocol:@protocol(MTLTexture)])
        return false;

    id<MTLTexture> t = obj;

    info->sample_count = (GLuint)t.sampleCount;
    info->mipmap_level_count = (GLuint)t.mipmapLevelCount;
    info->array_length = (GLuint)t.arrayLength;
    if (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray)
        info->array_length *= 6;

    if (t.parentTexture)
    {
        info->is_view = GL_TRUE;
        info->parent_level = (GLuint)t.parentRelativeLevel;
        info->parent_slice = (GLuint)t.parentRelativeSlice;
    }

    if (t.buffer)
    {
        info->has_buffer = GL_TRUE;
        info->buffer_offset = (GLsizeiptr)t.bufferOffset;
        info->buffer_length = (GLsizeiptr)(t.bufferBytesPerRow * MAX(t.height, 1));
        info->mtl_buffer = (__bridge void *)t.buffer;
    }

    return true;
}

static void texparamReleaseObj(GLMContext glm_ctx, void *obj)
{
    if (obj)
        CFBridgingRelease(obj);
}

@implementation MGLRenderer (TextureParams)

- (id<MTLDevice>) texparamDevice
{
    return _device;
}

- (void) bindExtTexparamFuncsToGLMContext: (GLMContext) glm_ctx
{
    struct MGLExtMetal_texparam *hooks = glm_ctx->mtl_funcs.ext_texparam;

    if (!hooks)
    {
        hooks = (struct MGLExtMetal_texparam *)calloc(1, sizeof(struct MGLExtMetal_texparam));
        glm_ctx->mtl_funcs.ext_texparam = hooks;
    }

    hooks->formatCaps = texparamFormatCaps;
    hooks->textureInfo = texparamTextureInfo;
    hooks->releaseObj = texparamReleaseObj;
}

#ifndef GL_DEPTH_TEXTURE_MODE
#define GL_DEPTH_TEXTURE_MODE 0x884B
#endif
#ifndef GL_LUMINANCE
#define GL_LUMINANCE 0x1909
#endif
#ifndef GL_INTENSITY
#define GL_INTENSITY 0x8049
#endif

#pragma mark samplers

- (id<MTLSamplerState>) texparamCreateSampler:(TextureParameter *)p target:(GLenum)target
{
    MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
    bool rect = (target == GL_TEXTURE_RECTANGLE);

    switch (p->min_filter)
    {
        case GL_NEAREST:
            desc.minFilter = MTLSamplerMinMagFilterNearest; desc.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
        case GL_LINEAR:
            desc.minFilter = MTLSamplerMinMagFilterLinear; desc.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
        case GL_NEAREST_MIPMAP_NEAREST:
            desc.minFilter = MTLSamplerMinMagFilterNearest; desc.mipFilter = MTLSamplerMipFilterNearest; break;
        case GL_LINEAR_MIPMAP_NEAREST:
            desc.minFilter = MTLSamplerMinMagFilterLinear; desc.mipFilter = MTLSamplerMipFilterNearest; break;
        case GL_NEAREST_MIPMAP_LINEAR:
            desc.minFilter = MTLSamplerMinMagFilterNearest; desc.mipFilter = MTLSamplerMipFilterLinear; break;
        default: /* GL_LINEAR_MIPMAP_LINEAR (values are validated at set time) */
            desc.minFilter = MTLSamplerMinMagFilterLinear; desc.mipFilter = MTLSamplerMipFilterLinear; break;
    }

    desc.magFilter = (p->mag_filter == GL_NEAREST) ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;

    GLenum wraps[3] = { p->wrap_s, p->wrap_t, p->wrap_r };
    MTLSamplerAddressMode modes[3];
    for (int i = 0; i < 3; i++)
    {
        switch (wraps[i])
        {
            case GL_CLAMP_TO_EDGE:        modes[i] = MTLSamplerAddressModeClampToEdge; break;
            case GL_CLAMP_TO_BORDER:      modes[i] = MTLSamplerAddressModeClampToBorderColor; break;
            case GL_MIRRORED_REPEAT:      modes[i] = MTLSamplerAddressModeMirrorRepeat; break;
            case GL_MIRROR_CLAMP_TO_EDGE: modes[i] = MTLSamplerAddressModeMirrorClampToEdge; break;
            default:                      modes[i] = MTLSamplerAddressModeRepeat; break;
        }
    }
    desc.sAddressMode = modes[0];
    desc.tAddressMode = modes[1];
    desc.rAddressMode = modes[2];

    /* Metal has three fixed border colors: pick the nearest */
    {
        float a = p->border_color[3];
        float l = (p->border_color[0] + p->border_color[1] + p->border_color[2]) / 3.0f;
        if (a < 0.5f)
            desc.borderColor = MTLSamplerBorderColorTransparentBlack;
        else if (l < 0.5f)
            desc.borderColor = MTLSamplerBorderColorOpaqueBlack;
        else
            desc.borderColor = MTLSamplerBorderColorOpaqueWhite;
    }

    /* LOD clamps: GL levels are relative to TEXTURE_BASE_LEVEL, which is level 0 of the bound view */
    float min_lod = p->min_lod < 0.0f ? 0.0f : p->min_lod;
    float max_lod = p->max_lod;
    if (max_lod < min_lod) max_lod = min_lod;
    desc.lodMinClamp = min_lod;
    desc.lodMaxClamp = max_lod;

    float aniso = p->max_anisotropy;
    if (aniso < 1.0f) aniso = 1.0f;
    if (aniso > 16.0f) aniso = 16.0f;
    desc.maxAnisotropy = (NSUInteger)(aniso + 0.5f);

    if (p->compare_mode == GL_COMPARE_REF_TO_TEXTURE)
    {
        switch (p->compare_func)
        {
            case GL_LEQUAL:   desc.compareFunction = MTLCompareFunctionLessEqual; break;
            case GL_GEQUAL:   desc.compareFunction = MTLCompareFunctionGreaterEqual; break;
            case GL_LESS:     desc.compareFunction = MTLCompareFunctionLess; break;
            case GL_GREATER:  desc.compareFunction = MTLCompareFunctionGreater; break;
            case GL_EQUAL:    desc.compareFunction = MTLCompareFunctionEqual; break;
            case GL_NOTEQUAL: desc.compareFunction = MTLCompareFunctionNotEqual; break;
            case GL_ALWAYS:   desc.compareFunction = MTLCompareFunctionAlways; break;
            default:          desc.compareFunction = MTLCompareFunctionNever; break;
        }
    }

    if (rect)
    {
        /* unnormalized coordinates: Metal requires clamp-to-edge/zero, min == mag, no mips, no anisotropy */
        desc.normalizedCoordinates = NO;
        desc.minFilter = desc.magFilter;
        desc.mipFilter = MTLSamplerMipFilterNotMipmapped;
        desc.maxAnisotropy = 1;
        MTLSamplerAddressMode clamp = (desc.borderColor == MTLSamplerBorderColorTransparentBlack) ?
                                      MTLSamplerAddressModeClampToZero : MTLSamplerAddressModeClampToEdge;
        desc.sAddressMode = (modes[0] == MTLSamplerAddressModeClampToBorderColor) ? clamp : MTLSamplerAddressModeClampToEdge;
        desc.tAddressMode = (modes[1] == MTLSamplerAddressModeClampToBorderColor) ? clamp : MTLSamplerAddressModeClampToEdge;
        desc.rAddressMode = MTLSamplerAddressModeClampToEdge;
        desc.lodMinClamp = 0.0f;
        desc.lodMaxClamp = 0.0f;
    }

    id<MTLSamplerState> sampler = [_device newSamplerStateWithDescriptor:desc];
    if (!sampler)
        sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];

    return sampler;
}

- (MGLTexParamCache *) texparamCacheFor:(TextureParameter *)p
{
    id obj = p->mtl_data ? (__bridge id)p->mtl_data : nil;

    if (obj && [obj isKindOfClass:[MGLTexParamCache class]])
        return (MGLTexParamCache *)obj;

    /* replace whatever was cached there (e.g. a bare sampler state) */
    if (p->mtl_data)
    {
        CFBridgingRelease(p->mtl_data);
        p->mtl_data = NULL;
    }

    MGLTexParamCache *cache = [MGLTexParamCache new];
    p->mtl_data = (void *)CFBridgingRetain(cache);
    return cache;
}

/* the MTLSamplerState cached in a TextureParameter's mtl_data: after
 * texparamCacheFor runs, mtl_data holds an MGLTexParamCache instead of a bare
 * sampler state, so callers must not cast it blindly (type confusion crash —
 * seen in texextRefreshStage) */
- (id<MTLSamplerState>) texparamCachedSampler:(TextureParameter *)p
{
    if (!p || !p->mtl_data) return nil;
    id obj = (__bridge id)p->mtl_data;
    if ([obj isKindOfClass:[MGLTexParamCache class]])
        return ((MGLTexParamCache *)obj).sampler;
    return (id<MTLSamplerState>)obj;
}

/* sampler for a texture unit: a bound sampler object overrides the texture's own sampler state */
- (id<MTLSamplerState>) texparamSamplerForUnit:(GLuint)unit texture:(Texture *)tex
{
    Sampler *gl_sampler = (unit < TEXTURE_UNITS) ? STATE(texture_samplers[unit]) : NULL;

    if (gl_sampler)
    {
        if (tex->target == GL_TEXTURE_RECTANGLE)
            return [self texparamCreateSampler:&gl_sampler->params target:tex->target];

        if (gl_sampler->dirty_bits && gl_sampler->mtl_data)
        {
            CFBridgingRelease(gl_sampler->mtl_data);
            gl_sampler->mtl_data = NULL;
        }

        if (gl_sampler->mtl_data == NULL)
        {
            gl_sampler->mtl_data = (void *)CFBridgingRetain([self texparamCreateSampler:&gl_sampler->params target:tex->target]);
            gl_sampler->dirty_bits = 0;
        }

        return (__bridge id<MTLSamplerState>)(gl_sampler->mtl_data);
    }

    MGLTexParamCache *cache = [self texparamCacheFor:&tex->params];
    if (!cache.sampler)
        cache.sampler = [self texparamCreateSampler:&tex->params target:tex->target];

    return cache.sampler;
}

/* texture to bind for sampling: a view when swizzle / level range / stencil texturing apply */
- (id<MTLTexture>) texparamTextureForTex:(Texture *)tex base:(id<MTLTexture>)base
{
    TextureParameter *p = &tex->params;

    if (!base || base.textureType == MTLTextureTypeTextureBuffer)
        return base;

    NSUInteger levels = base.mipmapLevelCount;
    NSUInteger base_level = MIN((NSUInteger)p->base_level, levels - 1);
    NSUInteger max_level = MIN((NSUInteger)p->max_level, levels - 1);
    if (max_level < base_level)
        max_level = base_level;

    MTLPixelFormat format = base.pixelFormat;
    if (p->depth_stencil_mode == GL_STENCIL_INDEX)
    {
        MTLPixelFormat stencil_format = MTLPixelFormatInvalid;

        if (format == MTLPixelFormatDepth32Float_Stencil8) stencil_format = MTLPixelFormatX32_Stencil8;
        else if (format == MTLPixelFormatDepth24Unorm_Stencil8) stencil_format = MTLPixelFormatX24_Stencil8;

        if (stencil_format != MTLPixelFormatInvalid)
        {
            if (base.usage & MTLTextureUsagePixelFormatView)
                format = stencil_format;
            else
                NSLog(@"MGL WARNING: stencil texturing needs MTLTextureUsagePixelFormatView on texture %u", tex->name);
        }
    }

    MTLTextureSwizzleChannels swz = MTLTextureSwizzleChannelsMake(
        texparamSwizzle(p->swizzle_r, MTLTextureSwizzleRed),
        texparamSwizzle(p->swizzle_g, MTLTextureSwizzleGreen),
        texparamSwizzle(p->swizzle_b, MTLTextureSwizzleBlue),
        texparamSwizzle(p->swizzle_a, MTLTextureSwizzleAlpha));

    bool swizzled = swz.red != MTLTextureSwizzleRed || swz.green != MTLTextureSwizzleGreen ||
                    swz.blue != MTLTextureSwizzleBlue || swz.alpha != MTLTextureSwizzleAlpha;

    /* GL_DEPTH_TEXTURE_MODE (GL 4.6 8.26): a non-comparison depth texture
     * splats its depth as (d,0,0,1)/(d,d,d,1)/(d,d,d,d)/(0,0,0,d); Metal's
     * depth2d splat is (d,d,d,d), so the mode becomes a view swizzle. The
     * TEXTURE_SWIZZLE selectors are then applied to that splatted vector
     * (geom r58: texture_swizzle.functional composes swizzle [A,B,G,R] with
     * mode RED and requires green <- (d,0,0,1).b == 0), so compose the user
     * selectors through the mode selector instead of overriding it. */
    {
        bool is_depth_fmt = false;
        switch (base.pixelFormat)
        {
            case MTLPixelFormatDepth16Unorm:
            case MTLPixelFormatDepth32Float:
            case MTLPixelFormatDepth24Unorm_Stencil8:
            case MTLPixelFormatDepth32Float_Stencil8:
                is_depth_fmt = true;
                break;
            default: break;
        }
        if (is_depth_fmt)
        {
            /* mode selector: how each of (r,g,b,a) of the splatted vector maps
             * to Metal's raw (d,d,d,d) or a constant */
            MTLTextureSwizzle msel[4];
            switch (p->depth_texture_mode)
            {
                case GL_RED:
                    msel[0] = MTLTextureSwizzleRed;  msel[1] = MTLTextureSwizzleZero; msel[2] = MTLTextureSwizzleZero; msel[3] = MTLTextureSwizzleOne;
                    break;
                case GL_INTENSITY:
                    msel[0] = MTLTextureSwizzleRed;  msel[1] = MTLTextureSwizzleRed;  msel[2] = MTLTextureSwizzleRed;  msel[3] = MTLTextureSwizzleRed;
                    break;
                case GL_ALPHA:
                    msel[0] = MTLTextureSwizzleZero; msel[1] = MTLTextureSwizzleZero; msel[2] = MTLTextureSwizzleZero; msel[3] = MTLTextureSwizzleRed;
                    break;
                default: /* LUMINANCE */
                    msel[0] = MTLTextureSwizzleRed;  msel[1] = MTLTextureSwizzleRed;  msel[2] = MTLTextureSwizzleRed;  msel[3] = MTLTextureSwizzleOne;
                    break;
            }
            /* map a selector (MTLTextureSwizzleZero..Alpha) through the mode
             * splat; Zero/One constants pass through unchanged */
            MTLTextureSwizzle mode_vec[6];
            mode_vec[MTLTextureSwizzleRed]   = msel[0];
            mode_vec[MTLTextureSwizzleGreen] = msel[1];
            mode_vec[MTLTextureSwizzleBlue]  = msel[2];
            mode_vec[MTLTextureSwizzleAlpha] = msel[3];
            mode_vec[MTLTextureSwizzleZero]  = MTLTextureSwizzleZero;
            mode_vec[MTLTextureSwizzleOne]   = MTLTextureSwizzleOne;
            if (swizzled)
            {
                const MTLTextureSwizzle user[4] = { swz.red, swz.green, swz.blue, swz.alpha };
                MTLTextureSwizzle net[4];
                for (int c = 0; c < 4; c++)
                    net[c] = mode_vec[user[c]];
                swz = MTLTextureSwizzleChannelsMake(net[0], net[1], net[2], net[3]);
            }
            else
            {
                swz = MTLTextureSwizzleChannelsMake(msel[0], msel[1], msel[2], msel[3]);
                swizzled = true;
            }
        }
    }

    if (base_level == 0 && max_level == levels - 1 && format == base.pixelFormat && !swizzled)
        return base;

    uint64_t key = (uint64_t)base_level | ((uint64_t)max_level << 8) | ((uint64_t)format << 16) |
                   ((uint64_t)swz.red << 32) | ((uint64_t)swz.green << 40) |
                   ((uint64_t)swz.blue << 48) | ((uint64_t)swz.alpha << 56);

    MGLTexParamCache *cache = [self texparamCacheFor:p];
    if (cache.view && cache.viewParent == (uintptr_t)(__bridge void *)base && cache.viewKey == key)
        return cache.view;

    NSUInteger slices = base.arrayLength;
    switch (base.textureType)
    {
        case MTLTextureTypeCube:
        case MTLTextureTypeCubeArray: slices *= 6; break;
        case MTLTextureType3D: slices = 1; break;
        default: break;
    }

    id<MTLTexture> view = nil;
    @try {
        view = [base newTextureViewWithPixelFormat:format
                                       textureType:base.textureType
                                            levels:NSMakeRange(base_level, max_level - base_level + 1)
                                            slices:NSMakeRange(0, slices)
                                           swizzle:swz];
    } @catch (NSException *e) {
        NSLog(@"MGL ERROR: texture view for texture %u failed: %@", tex->name, e);
        view = nil;
    }

    if (!view)
        return base;

    cache.view = view;
    cache.viewParent = (uintptr_t)(__bridge void *)base;
    cache.viewKey = key;

    return view;
}

/* called while building a texture's MTLTextureDescriptor */
- (void) texparamPrepareTextureDescriptor:(MTLTextureDescriptor *)desc forTex:(Texture *)tex
{
    /* DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX samples through an X*_Stencil8 view */
    if (desc.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ||
        desc.pixelFormat == MTLPixelFormatDepth24Unorm_Stencil8)
    {
        desc.usage |= MTLTextureUsagePixelFormatView;
    }
}

- (MTLTextureSwizzleChannels) texparamSwizzleForParams:(TextureParameter *)p
{
    return MTLTextureSwizzleChannelsMake(texparamSwizzle(p->swizzle_r, MTLTextureSwizzleRed),
                                         texparamSwizzle(p->swizzle_g, MTLTextureSwizzleGreen),
                                         texparamSwizzle(p->swizzle_b, MTLTextureSwizzleBlue),
                                         texparamSwizzle(p->swizzle_a, MTLTextureSwizzleAlpha));
}

@end
