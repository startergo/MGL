/*
 * MGLRenderer+TextureData.m — Metal side of the 'texdata' domain (owned by agent 'texdata').
 *
 * Texture realization (GL shape -> MTLTexture, content carried across shape
 * changes), uploads (staging buffer + blit), synchronous downloads, GPU decode of
 * compressed images (compute), copies from the read framebuffer, CopyImageSubData
 * and mipmap generation.  The storage model is described in mgl_ext_texdata.h.
 *
 * Row order: Metal row r holds GL row r for every texture and render target
 * (the vertex stage applies the clip-space fixup), so no transfer flips rows.
 */

#import <objc/runtime.h>

#import "MGLRenderer+TextureData.h"
#include "mgl_ext_texdata.h"
#include "mgl_ext_texparam.h"
#include "mgl_ext_fb.h"
#import "MGLRenderer+TextureExt.h"
#import "MGLRenderer+Framebuffer.h"   /* compat s36: fbQuantizeRGB9E5 (RGB9_E5 rb read-out) */
#import "mgl_ext_texext.h"
extern uint32_t mtlFormatForGLInternalFormat(GLenum internal_format);
#include "texdata_formats.h"

static char kTDDecodePipeline;

static inline MGLRenderer *TDR(GLMContext ctx) { return (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj; }

static size_t td_align4(size_t v) { return (v + 3) & ~(size_t)3; }

static NSUInteger td_mip(NSUInteger v, NSUInteger level) { NSUInteger r = v >> level; return r ? r : 1; }

/* GL internal format describing the contents of a default-framebuffer texture */
static GLenum td_gl_format_for_mtl(MTLPixelFormat pf)
{
    switch (pf)
    {
        case MTLPixelFormatBGRA8Unorm: return GL_RGBA8;
        case MTLPixelFormatBGRA8Unorm_sRGB: return GL_SRGB8_ALPHA8;
        case MTLPixelFormatRGBA8Unorm: return GL_RGBA8;
        case MTLPixelFormatRGBA8Unorm_sRGB: return GL_SRGB8_ALPHA8;
        case MTLPixelFormatRGB10A2Unorm: return GL_RGB10_A2;
        case MTLPixelFormatRGBA16Float: return GL_RGBA16F;
        case MTLPixelFormatRGBA32Float: return GL_RGBA32F;
        case MTLPixelFormatDepth16Unorm: return GL_DEPTH_COMPONENT16;
        case MTLPixelFormatDepth32Float: return GL_DEPTH_COMPONENT32F;
        case MTLPixelFormatDepth32Float_Stencil8: return GL_DEPTH32F_STENCIL8;
        case MTLPixelFormatStencil8: return GL_STENCIL_INDEX8;
        default: return 0;
    }
}

/* Metal slice of (face, z) */
static NSUInteger td_slice(Texture *tex, GLuint face, GLint z)
{
    if (tex->target == GL_TEXTURE_CUBE_MAP)
        return face;
    if (tex->target == GL_TEXTURE_3D)
        return 0;
    return (NSUInteger)z;
}

static void td_level_dims(const TDShape *s, GLuint level, NSUInteger *w, NSUInteger *h, NSUInteger *d)
{
    *w = td_mip(s->width, level);
    *h = td_mip(s->height, level);
    *d = (s->target == GL_TEXTURE_3D) ? td_mip(s->depth, level) : s->depth;
}

@implementation MGLRenderer (TextureData)

#pragma mark - helpers

- (id<MTLBlitCommandEncoder>) texdataBlitEncoder
{
    [self endRenderEncoding];

    if (_currentCommandBuffer == nil || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        [self newCommandBuffer];

    return [_currentCommandBuffer blitCommandEncoder];
}

/* commit the current command buffer, wait for it and start a new one */
- (void) texdataSubmitAndWait
{
    [self endRenderEncoding];

    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    if (cb && cb.status < MTLCommandBufferStatusCommitted)
    {
        [self commitCommandBufferWithAGXRecovery:cb];
        [cb waitUntilCompleted];
    }

    [self newCommandBuffer];
}

- (MTLTextureType) texdataTypeForTarget:(GLenum)target
{
    switch (target)
    {
        case GL_TEXTURE_1D: return MTLTextureType2D;              /* SPIRV-Cross texture_1D_as_2D */
        case GL_TEXTURE_1D_ARRAY: return MTLTextureType2DArray;
        case GL_TEXTURE_2D_ARRAY: return MTLTextureType2DArray;
        case GL_TEXTURE_CUBE_MAP: return MTLTextureTypeCube;
        case GL_TEXTURE_CUBE_MAP_ARRAY: return MTLTextureTypeCubeArray;
        case GL_TEXTURE_3D: return MTLTextureType3D;
        default: return MTLTextureType2D;
    }
}

/* textures this category realizes (buffer / view / multisample textures are texext's,
 * renderbuffer storage is created by the fb domain) */
- (bool) texdataHandlesTexture:(Texture *)tex
{
    if (!tex || [self texextOwnsTexture:tex])
        return false;

    switch (tex->target)
    {
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY:
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            return tex->faces[0].levels != NULL;
        default:
            return false;
    }
}

- (id<MTLTexture>) texdataNewTextureForShape:(const TDShape *)s tex:(Texture *)tex
{
    const TDFormat *f = tdFormatInfo(s->internalformat);
    const TDFormat *sf = tdStorageFormat(f);

    if (!sf || sf->mtl == MTLPixelFormatInvalid)
        return nil;

    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = [self texdataTypeForTarget:s->target];
    d.pixelFormat = (MTLPixelFormat)sf->mtl;
    d.width = s->width;
    d.height = s->height;
    d.depth = 1;
    d.arrayLength = 1;

    switch (s->target)
    {
        case GL_TEXTURE_3D: d.depth = s->depth; break;
        case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY: d.arrayLength = MAX(s->depth, 1); break;
        case GL_TEXTURE_CUBE_MAP_ARRAY: d.arrayLength = MAX(s->depth / 6, 1); break;
        default: break;
    }

    d.mipmapLevelCount = MAX(s->levels, 1);
    d.storageMode = MTLStorageModePrivate;

    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    bool compressed = (sf->flags & TDF_COMPRESSED) != 0;
    if (!compressed && (f->flags & TDF_RENDERABLE))
        usage |= MTLTextureUsageRenderTarget;
    if (!compressed && !(sf->flags & (TDF_DEPTH | TDF_STENCIL | TDF_SRGB)) && sf->kind != TDK_RGB9E5)
        usage |= MTLTextureUsageShaderWrite;
    d.usage = usage;

    [self texparamPrepareTextureDescriptor:d forTex:tex];
    [self texextAdjustTextureDescriptor:d forTexture:tex];

    id<MTLTexture> t = [_device newTextureWithDescriptor:d];
    if (!t)
    {
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView |
                  ((usage & MTLTextureUsageRenderTarget) ? MTLTextureUsageRenderTarget : 0);
        t = [_device newTextureWithDescriptor:d];
    }
    if (!t)
        NSLog(@"MGL texdata: cannot create %lux%lux%lu texture (format %lu, %lu levels)",
              (unsigned long)d.width, (unsigned long)d.height, (unsigned long)d.depth,
              (unsigned long)d.pixelFormat, (unsigned long)d.mipmapLevelCount);
    return t;
}

#pragma mark - upload / download

/* encode a copy of tight storage-layout data into a texture region (no sync) */
- (void) texdataEncodeUpload:(id<MTLTexture>)texture tex:(Texture *)tex format:(const TDFormat *)sf
                      region:(const TDRegion *)r src:(const TDStorage *)src depth:(bool)wd stencil:(bool)ws
{
    bool compressed = (sf->flags & TDF_COMPRESSED) != 0;
    bool is3d = texture.textureType == MTLTextureType3D;
    NSUInteger slices = is3d ? 1 : (NSUInteger)r->d;
    NSUInteger depth = is3d ? (NSUInteger)r->d : 1;
    NSUInteger rows = compressed ? ((NSUInteger)r->h + sf->block_h - 1) / sf->block_h : (NSUInteger)r->h;

    /* planes: color/depth plane and (D32F_S8) stencil plane */
    for (int plane = 0; plane < 2; plane++)
    {
        if (plane == 0 && !wd && sf->kind == TDK_D32FS8) continue;
        if (plane == 1 && (!ws || sf->kind != TDK_D32FS8 || !src->stencil)) continue;

        const uint8_t *data = plane ? (const uint8_t *)src->stencil : (const uint8_t *)src->data;
        size_t srow = plane ? src->s_row_pitch : src->row_pitch;
        size_t simg = plane ? src->s_image_pitch : src->image_pitch;
        size_t rowbytes = plane ? (size_t)r->w : (compressed ? tdStorageRowBytes(sf, r->w) : (size_t)r->w * sf->bpp);
        size_t pitch = compressed ? rowbytes : td_align4(rowbytes);
        size_t img = pitch * rows;
        size_t total = img * slices * depth;

        /* gpuaudit note: a pooled staging ring was measured here and rejected - the
         * staging allocation is cheap (Metal pools small shared allocations) and the
         * per-upload cost is the copy itself, which is bandwidth bound */
        id<MTLBuffer> staging = [_device newBufferWithLength:MAX(total, 16) options:MTLResourceStorageModeShared];
        uint8_t *dst = (uint8_t *)staging.contents;
        NSUInteger stage_off = 0;

        for (NSUInteger z = 0; z < slices * depth; z++)
            for (NSUInteger y = 0; y < rows; y++)
                memcpy(dst + z * img + y * pitch, data + z * simg + y * srow, rowbytes);

        MTLBlitOption opt = MTLBlitOptionNone;
        if (sf->kind == TDK_D32FS8)
            opt = plane ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;

        id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
        for (NSUInteger s = 0; s < slices; s++)
        {
            [blit copyFromBuffer:staging
                    sourceOffset:stage_off + s * img
               sourceBytesPerRow:pitch
             sourceBytesPerImage:img
                      sourceSize:MTLSizeMake(r->w, r->h, depth)
                       toTexture:texture
                destinationSlice:td_slice(tex, r->face, r->z) + (is3d ? 0 : s)
                destinationLevel:r->level
               destinationOrigin:MTLOriginMake(r->x, r->y, is3d ? r->z : 0)
                         options:opt];
        }
        [blit endEncoding];
    }
}

/* synchronous texture -> tight storage-layout copy */
- (bool) texdataDownload:(id<MTLTexture>)texture tex:(Texture *)tex format:(const TDFormat *)sf
                  region:(const TDRegion *)r dst:(const TDStorage *)dst
{
    /* flush a pending 3D image-unit shadow plane before reading the storage */
    [self geom3DImageShadowFlushTexture:tex];

    bool compressed = (sf->flags & TDF_COMPRESSED) != 0;
    bool is3d = texture.textureType == MTLTextureType3D;
    NSUInteger slices = is3d ? 1 : (NSUInteger)r->d;
    NSUInteger depth = is3d ? (NSUInteger)r->d : 1;
    NSUInteger rows = compressed ? ((NSUInteger)r->h + sf->block_h - 1) / sf->block_h : (NSUInteger)r->h;
    id<MTLBuffer> bufs[2] = { nil, nil };
    size_t pitches[2] = { 0, 0 }, imgs[2] = { 0, 0 }, rowbytes_p[2] = { 0, 0 };

    for (int plane = 0; plane < 2; plane++)
    {
        if (plane == 1 && (sf->kind != TDK_D32FS8 || !dst->stencil)) continue;

        size_t rowbytes = plane ? (size_t)r->w : (compressed ? tdStorageRowBytes(sf, r->w) : (size_t)r->w * sf->bpp);
        size_t pitch = compressed ? rowbytes : td_align4(rowbytes);
        size_t img = pitch * rows;
        bufs[plane] = [_device newBufferWithLength:MAX(img * slices * depth, 16) options:MTLResourceStorageModeShared];
        pitches[plane] = pitch; imgs[plane] = img; rowbytes_p[plane] = rowbytes;

        MTLBlitOption opt = MTLBlitOptionNone;
        if (sf->kind == TDK_D32FS8)
            opt = plane ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;

        id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
        for (NSUInteger s = 0; s < slices; s++)
        {
            [blit copyFromTexture:texture
                      sourceSlice:td_slice(tex, r->face, r->z) + (is3d ? 0 : s)
                      sourceLevel:r->level
                     sourceOrigin:MTLOriginMake(r->x, r->y, is3d ? r->z : 0)
                       sourceSize:MTLSizeMake(r->w, r->h, depth)
                         toBuffer:bufs[plane]
                destinationOffset:s * img
           destinationBytesPerRow:pitch
         destinationBytesPerImage:img
                          options:opt];
        }
        [blit endEncoding];
    }

    [self texdataSubmitAndWait];

    for (int plane = 0; plane < 2; plane++)
    {
        if (!bufs[plane]) continue;
        const uint8_t *src = (const uint8_t *)bufs[plane].contents;
        uint8_t *out = plane ? (uint8_t *)dst->stencil : (uint8_t *)dst->data;
        size_t drow = plane ? dst->s_row_pitch : dst->row_pitch;
        size_t dimg = plane ? dst->s_image_pitch : dst->image_pitch;
        for (NSUInteger z = 0; z < slices * depth; z++)
            for (NSUInteger y = 0; y < rows; y++)
                memcpy(out + z * dimg + y * drow, src + z * imgs[plane] + y * pitches[plane], rowbytes_p[plane]);
    }
    return true;
}

#pragma mark - realization

- (bool) texdataRealize:(Texture *)tex
{
    TDShape s;
    GLMContext glm = ctx;

    if (![self texdataHandlesTexture:tex] || !mgl_texdata_shape(glm, tex, &s))
        return false;

    TDTexInfo *info = mgl_texdata_info(glm, tex);
    const TDFormat *sf = mgl_texdata_storage_format(s.internalformat);
    if (!info || !sf)
        return false;

    GLuint faces = mgl_texdata_num_faces(tex);
    id<MTLTexture> old = (info->realized && tex->mtl_data) ? (__bridge id<MTLTexture>)tex->mtl_data : nil;
    bool same = old && memcmp(&info->shape, &s, sizeof(TDShape)) == 0 && old.pixelFormat == (MTLPixelFormat)sf->mtl;

    if (!same)
    {
        id<MTLTexture> nt = [self texdataNewTextureForShape:&s tex:tex];
        if (!nt)
            return false;

        uint32_t kept[6] = { 0 };

        if (old)
        {
            TDShape os = info->shape;
            const TDFormat *osf = mgl_texdata_storage_format(os.internalformat);
            bool same_fmt = old.pixelFormat == nt.pixelFormat;

            for (GLuint f = 0; f < faces; f++)
            {
                for (GLuint l = 0; l < os.levels && l < TD_MAX_LEVELS; l++)
                {
                    if (!(info->on_gpu[f] & (1u << l)))
                        continue;

                    NSUInteger ow, oh, od, nw, nh, nd;
                    td_level_dims(&os, l, &ow, &oh, &od);
                    td_level_dims(&s, l, &nw, &nh, &nd);

                    if (same_fmt && l < s.levels && ow == nw && oh == nh && od == nd)
                    {
                        id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
                        bool is3d = nt.textureType == MTLTextureType3D;
                        NSUInteger slices = (tex->target == GL_TEXTURE_CUBE_MAP) ? 1 : (is3d ? 1 : od);
                        for (NSUInteger sl = 0; sl < slices; sl++)
                        {
                            NSUInteger slice = (tex->target == GL_TEXTURE_CUBE_MAP) ? f : sl;
                            [blit copyFromTexture:old sourceSlice:slice sourceLevel:l sourceOrigin:MTLOriginMake(0, 0, 0)
                                       sourceSize:MTLSizeMake(ow, oh, is3d ? od : 1)
                                        toTexture:nt destinationSlice:slice destinationLevel:l
                                destinationOrigin:MTLOriginMake(0, 0, 0)];
                        }
                        [blit endEncoding];
                        kept[f] |= 1u << l;
                    }
                    else if (osf)
                    {
                        /* the level leaves the realized shape: keep its content on the CPU */
                        TDStorage st;
                        GLuint gw, gh, gd;
                        mgl_texdata_level_geometry(tex, f, l, &gw, &gh, &gd);
                        if (gw == ow && gh == oh && gd == od && mgl_texdata_shadow(glm, tex, f, l, true, &st))
                        {
                            TDRegion r = { f, l, 0, 0, 0, (GLsizei)ow, (GLsizei)oh, (GLsizei)od };
                            [self texdataDownload:old tex:tex format:osf region:&r dst:&st];
                        }
                    }
                }
            }
        }

        if (tex->mtl_data)
        {
            CFBridgingRelease(tex->mtl_data);
            tex->mtl_data = NULL;
        }
        tex->mtl_data = (void *)CFBridgingRetain(nt);
        mglTexParamInvalidate(glm, &tex->params);

        info->shape = s;
        info->realized = GL_TRUE;
        for (GLuint f = 0; f < 6; f++)
            info->on_gpu[f] = kept[f];
    }

    /* upload pending CPU shadows and mark every level of the shape as GPU resident */
    id<MTLTexture> texture = (__bridge id<MTLTexture>)tex->mtl_data;
    for (GLuint f = 0; f < faces; f++)
    {
        for (GLuint l = 0; l < s.levels && l < TD_MAX_LEVELS; l++)
        {
            GLuint gw, gh, gd;
            NSUInteger w, h, d;
            mgl_texdata_level_geometry(tex, f, l, &gw, &gh, &gd);
            td_level_dims(&s, l, &w, &h, &d);
            if (gw != w || gh != h || gd != d)
                continue;   /* level below the base, not defined */

            TDStorage st;
            if (mgl_texdata_shadow(glm, tex, f, l, false, &st))
            {
                TDRegion r = { f, l, 0, 0, 0, (GLsizei)w, (GLsizei)h, (GLsizei)d };
                [self texdataEncodeUpload:texture tex:tex format:sf region:&r src:&st depth:true stencil:true];
                mgl_texdata_free_shadow(glm, tex, f, l);
            }
            info->on_gpu[f] |= 1u << l;
        }
    }

    tex->complete = true;
    tex->mipmapped = s.levels > 1;
    tex->dirty_bits &= ~(GLuint)(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    return true;
}

/* -bindMTLTexture: for textures this domain realizes; returns false if not ours */
- (bool) texdataBindMTLTexture:(Texture *)tex
{
    if (![self texdataHandlesTexture:tex])
        return false;

    if (![self texdataRealize:tex])
    {
        /* undefined base level: sample a fallback so the draw still binds something */
        if (!tex->mtl_data)
            tex->mtl_data = (void *)CFBridgingRetain([self createFallbackMTLTexture: tex]);
    }

    if (tex->dirty_bits & ~(GLuint)(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL))
    {
        /* parameter change: only the sampler depends on it (views come from texparam) */
        if (tex->params.mtl_data)
        {
            CFBridgingRelease(tex->params.mtl_data);
            tex->params.mtl_data = NULL;
        }
        tex->dirty_bits &= (GLuint)(DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_LEVEL);
    }

    if (tex->params.mtl_data == NULL)
    {
        id<MTLSamplerState> sampler = [self createMTLSamplerForTexParam:&tex->params target:tex->target];
        if (!sampler)
            sampler = [_device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
        tex->params.mtl_data = (void *)CFBridgingRetain(sampler);
    }

    return true;
}

#pragma mark - GPU decode of compressed images

- (id<MTLComputePipelineState>) texdataDecodePipeline
{
    id<MTLComputePipelineState> p = objc_getAssociatedObject(self, &kTDDecodePipeline);
    if (p)
        return p;

    static const char *src =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void td_decode(texture2d<float> t [[texture(0)]], device float4 *o [[buffer(0)]],\n"
        "                      constant uint4 &p [[buffer(1)]], uint2 g [[thread_position_in_grid]])\n"
        "{\n"
        "    if (g.x >= p.z || g.y >= p.w) return;\n"
        "    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);\n"
        "    o[g.y * p.z + g.x] = t.sample(s, float2(g + p.xy) + 0.5);\n"
        "}\n";

    NSError *err = nil;
    id<MTLLibrary> lib = [_device newLibraryWithSource:[NSString stringWithUTF8String:src] options:nil error:&err];
    if (!lib)
    {
        NSLog(@"MGL texdata: decode kernel failed: %@", err);
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"td_decode"];
    p = [_device newComputePipelineStateWithFunction:fn error:&err];
    if (p)
        objc_setAssociatedObject(self, &kTDDecodePipeline, p, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return p;
}

- (bool) texdataDownloadDecoded:(Texture *)tex region:(const TDRegion *)r dst:(float *)dst
{
    /* a 3D texture with a pending image-unit shadow plane must be written back
     * before its storage is decoded (geom domain) */
    [self geom3DImageShadowFlushTexture:tex];

    /* A GL view (texext) whose internalformat differs from the root's (BC6H
     * float <-> ufloat, sRGB <-> linear, view classes) must decode with the
     * VIEW's decoder: build the decode view over the ROOT's storage so the
     * region stays root-relative, and take the pixel format from the view. */
    Texture *owner = tex;

    {
        MGLTexExtInfo *xinfo = mgl_texext_info(ctx, tex);
        if (xinfo && xinfo->kind == MGL_TEXEXT_VIEW && xinfo->view_root)
        {
            Texture *root = xinfo->view_root;
            if (![self texdataRealize:root] || !root->mtl_data)
                return false;
            owner = root;
        }
    }

    if (![self texdataRealize:owner] || !owner->mtl_data)
        return false;

    id<MTLTexture> texture = (__bridge id<MTLTexture>)owner->mtl_data;
    if (texture.textureType == MTLTextureType3D)
        return false;

    id<MTLComputePipelineState> pipe = [self texdataDecodePipeline];
    if (!pipe)
        return false;

    /* GetTexImage returns stored values: read sRGB formats through their linear variant */
    MTLPixelFormat pf = texture.pixelFormat;
    if (owner != tex)
        pf = (MTLPixelFormat)mtlFormatForGLInternalFormat(tex->internalformat);
    switch (pf)
    {
        case MTLPixelFormatBC1_RGBA_sRGB: case MTLPixelFormatBC2_RGBA_sRGB: case MTLPixelFormatBC3_RGBA_sRGB:
        case MTLPixelFormatBC7_RGBAUnorm_sRGB: case MTLPixelFormatETC2_RGB8_sRGB: case MTLPixelFormatETC2_RGB8A1_sRGB:
        case MTLPixelFormatEAC_RGBA8_sRGB:
            pf = (MTLPixelFormat)(pf - 1);
            break;
        default:
            break;
    }

    size_t per = (size_t)r->w * r->h;
    id<MTLBuffer> out = [_device newBufferWithLength:MAX(per * r->d * 16, 16) options:MTLResourceStorageModeShared];

    [self endRenderEncoding];
    if (_currentCommandBuffer == nil || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        [self newCommandBuffer];

    id<MTLComputeCommandEncoder> enc = [_currentCommandBuffer computeCommandEncoder];
    [enc setComputePipelineState:pipe];

    for (GLsizei z = 0; z < r->d; z++)
    {
        id<MTLTexture> view = [texture newTextureViewWithPixelFormat:pf textureType:MTLTextureType2D
                                                              levels:NSMakeRange(r->level, 1)
                                                              slices:NSMakeRange(td_slice(owner, r->face, r->z + z), 1)];
        if (!view)
        {
            [enc endEncoding];
            return false;
        }
        uint32_t p[4] = { (uint32_t)r->x, (uint32_t)r->y, (uint32_t)r->w, (uint32_t)r->h };
        [enc setTexture:view atIndex:0];
        [enc setBuffer:out offset:(NSUInteger)z * per * 16 atIndex:0];
        [enc setBytes:p length:sizeof(p) atIndex:1];
        MTLSize tg = MTLSizeMake(8, 8, 1);
        MTLSize groups = MTLSizeMake(((NSUInteger)r->w + 7) / 8, ((NSUInteger)r->h + 7) / 8, 1);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
    }
    [enc endEncoding];

    [self texdataSubmitAndWait];

    memcpy(dst, out.contents, per * r->d * 16);
    return true;
}

#pragma mark - read framebuffer

/* the texture, level and slice holding the current read buffer (fb domain helpers) */
- (id<MTLTexture>) texdataReadSource:(bool)ds level:(NSUInteger *)level slice:(NSUInteger *)slice plane:(NSUInteger *)plane glFormat:(GLenum *)ifmt
{
    *level = 0; *slice = 0; *plane = 0; *ifmt = 0;

    Framebuffer *fbo = STATE(readbuffer);
    MGLFBDrawState *dsel = mgl_fb_draw_state(ctx, fbo);

    if (fbo)
    {
        FBOAttachment *att;
        if (ds)
        {
            att = mgl_fb_attachment(fbo, GL_DEPTH_ATTACHMENT);
            if (!att || !mgl_fb_attachment_texture(ctx, att))
                att = mgl_fb_attachment(fbo, GL_STENCIL_ATTACHMENT);
        }
        else
        {
            if (!dsel || dsel->read_buffer == GL_NONE)
                return nil;
            att = mgl_fb_attachment(fbo, dsel->read_buffer);
        }
        Texture *tex = att ? mgl_fb_attachment_texture(ctx, att) : NULL;
        if (!tex || ![self bindMTLTexture:tex] || !tex->mtl_data)
            return nil;

        GLuint l = 0, s = 0, p = 0, n = 0;
        mgl_fb_attachment_slice(ctx, fbo, att, &l, &s, &p, &n);
        *level = l; *slice = s; *plane = p;
        *ifmt = tex->internalformat;
        /* compat s36: RGB9_E5 renderbuffers are RGBA16F-backed; snap the backing
         * onto the 5-9-9-9 lattice before glCopyTex* consumes it as a source
         * (see fbQuantizeRGB9E5 in MGLRenderer+Framebuffer.m) */
        if (!ds && att->textarget == GL_RENDERBUFFER && tex->internalformat == GL_RGB9_E5)
            [self fbQuantizeRGB9E5:(__bridge id<MTLTexture>)tex->mtl_data];
        return (__bridge id<MTLTexture>)tex->mtl_data;
    }

    if (!ds && dsel && dsel->read_buffer == GL_NONE)
        return nil;

    struct MGLExtMetal_fb *fb = ctx->mtl_funcs.ext_fb;
    if (!fb)
        return nil;
    if (fb->default_size)
    {
        GLuint w, h;
        fb->default_size(ctx, &w, &h);
    }
    void *obj = ds ? fb->default_depth_stencil : fb->default_color;
    if (!obj)
        return nil;
    id<MTLTexture> t = (__bridge id<MTLTexture>)obj;
    *ifmt = td_gl_format_for_mtl(t.pixelFormat);
    return t;
}

- (bool) texdataReadBufferInfo:(bool)ds internalformat:(GLenum *)ifmt width:(GLuint *)w height:(GLuint *)h
{
    NSUInteger level, slice, plane;
    id<MTLTexture> t = [self texdataReadSource:ds level:&level slice:&slice plane:&plane glFormat:ifmt];
    if (!t)
        return false;
    *w = (GLuint)td_mip(t.width, level);
    *h = (GLuint)td_mip(t.height, level);
    return true;
}

- (GLenum) texdataCopyFromReadFB:(Texture *)dst region:(const TDRegion *)r x:(GLint)x y:(GLint)y
                           width:(GLsizei)w height:(GLsizei)h depth:(bool)wd stencil:(bool)ws
{
    NSUInteger slevel, sslice, splane;
    GLenum sfmt;
    bool ds = wd || ws;

    id<MTLTexture> src = [self texdataReadSource:ds level:&slevel slice:&sslice plane:&splane glFormat:&sfmt];
    if (!src)
        return GL_INVALID_OPERATION;

    if (![self texdataRealize:dst] || !dst->mtl_data)
        return GL_OUT_OF_MEMORY;

    id<MTLTexture> dt = (__bridge id<MTLTexture>)dst->mtl_data;
    const TDFormat *df = mgl_texdata_storage_format(dst->faces[r->face].levels[r->level].mtl_format);
    if (!df)
        return GL_INVALID_OPERATION;

    /* texels outside the read buffer are undefined: copy the intersection only (no row flip) */
    GLint sw = (GLint)td_mip(src.width, slevel), sh = (GLint)td_mip(src.height, slevel);
    GLint x0 = MAX(x, 0), y0 = MAX(y, 0), x1 = MIN(x + w, sw), y1 = MIN(y + h, sh);
    if (x1 <= x0 || y1 <= y0)
        return GL_NO_ERROR;

    bool layers = (dst->target == GL_TEXTURE_1D_ARRAY);   /* framebuffer rows go to consecutive layers */
    GLint dx = r->x + (x0 - x);
    GLint dy = layers ? 0 : r->y + (y0 - y);
    GLint dz = layers ? r->z + (y0 - y) : r->z;
    GLsizei cw = x1 - x0, ch = y1 - y0;

    bool src3d = src.textureType == MTLTextureType3D;
    bool dst3d = dt.textureType == MTLTextureType3D;
    TDTexInfo *info = mgl_texdata_find_info(ctx, dst);
    bool level_on_gpu = info && info->realized && (info->on_gpu[r->face] & (1u << r->level));

    if (src.pixelFormat == dt.pixelFormat && src.sampleCount == 1 && level_on_gpu)
    {
        id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
        NSUInteger n = layers ? (NSUInteger)ch : 1;
        for (NSUInteger i = 0; i < n; i++)
        {
            NSUInteger rows = layers ? 1 : (NSUInteger)ch;
            [blit copyFromTexture:src sourceSlice:src3d ? 0 : sslice sourceLevel:slevel
                     sourceOrigin:MTLOriginMake(x0, y0 + i, src3d ? splane : 0)
                       sourceSize:MTLSizeMake(cw, rows, 1)
                        toTexture:dt
                 destinationSlice:dst3d ? 0 : td_slice(dst, r->face, dz + (GLint)i)
                 destinationLevel:r->level
                destinationOrigin:MTLOriginMake(dx, dy, dst3d ? dz : 0)];
        }
        [blit endEncoding];
        return GL_NO_ERROR;
    }

    /* conversion: read through the fb domain (GL row order), convert on the CPU, write */
    struct MGLExtMetal_fb *fb = ctx->mtl_funcs.ext_fb;
    if (!fb || !fb->read)
        return GL_INVALID_OPERATION;

    MGLFBDrawState *dsel = mgl_fb_draw_state(ctx, STATE(readbuffer));
    size_t texels = (size_t)cw * ch;
    void *tmp = calloc(texels, 16);
    size_t drow = (size_t)cw * df->bpp, dimg = drow * ch;
    uint8_t *dbuf = (uint8_t *)calloc(1, dimg + texels + 16);
    if (!tmp || !dbuf) { free(tmp); free(dbuf); return GL_OUT_OF_MEMORY; }
    TDStorage st = { dbuf, drow, dimg, df->kind == TDK_D32FS8 ? dbuf + dimg : NULL, (size_t)cw, texels };

    bool wrote_d = false, wrote_s = false;
    for (int pass = 0; pass < 3; pass++)
    {
        int aspect = pass == 0 ? MGL_FB_ASPECT_COLOR : pass == 1 ? MGL_FB_ASPECT_DEPTH : MGL_FB_ASPECT_STENCIL;
        if (pass == 0 && ds) continue;
        if (pass == 1 && !wd) continue;
        if (pass == 2 && !ws) continue;

        MGLFBReadRequest req = { STATE(readbuffer), dsel ? dsel->read_buffer : GL_BACK, aspect, x0, y0, cw, ch, tmp };
        int kind = fb->read(ctx, &req);
        if (kind < 0) { free(tmp); free(dbuf); return GL_INVALID_OPERATION; }

        GLenum format, type;
        if (pass == 0)
        {
            if ((kind != MGL_FB_KIND_FLOAT) != ((df->flags & TDF_INTEGER) != 0)) { free(tmp); free(dbuf); return GL_INVALID_OPERATION; }
            format = kind == MGL_FB_KIND_FLOAT ? GL_RGBA : GL_RGBA_INTEGER;
            type = kind == MGL_FB_KIND_FLOAT ? GL_FLOAT : kind == MGL_FB_KIND_INT ? GL_INT : GL_UNSIGNED_INT;
        }
        else if (pass == 1) { format = GL_DEPTH_COMPONENT; type = GL_FLOAT; }
        else { format = GL_STENCIL_INDEX; type = GL_UNSIGNED_INT; }

        size_t gs = pass == 0 ? 16 : 4;
        tdUnpack(df, tmp, (size_t)cw * gs, texels * gs, format, type, false, cw, ch, 1, &st, NULL, NULL);
        if (pass == 1) wrote_d = true;
        if (pass == 2) wrote_s = true;
    }
    free(tmp);

    if (layers)
    {
        st.image_pitch = st.row_pitch;
        st.s_image_pitch = st.s_row_pitch;
    }
    TDRegion dr = { r->face, r->level, dx, dy, dz, cw, layers ? 1 : ch, layers ? ch : 1 };
    mgl_texdata_write_region(ctx, dst, &dr, &st, ds ? wrote_d : true, ds ? wrote_s : true);
    free(dbuf);
    return GL_NO_ERROR;
}

#pragma mark - CopyImageSubData / mipmaps

- (bool) texdataCopyImage:(Texture *)st region:(const TDRegion *)sr to:(Texture *)dt region:(const TDRegion *)dr
{
    for (int i = 0; i < 2; i++)
    {
        Texture *t = i ? dt : st;
        const TDRegion *r = i ? dr : sr;
        if (t->target == GL_RENDERBUFFER)
        {
            if (!t->mtl_data)
                return false;
            continue;
        }
        if (![self texdataRealize:t] || !t->mtl_data)
            return false;
        TDTexInfo *info = mgl_texdata_find_info(ctx, t);
        if (!info || !(info->on_gpu[r->face] & (1u << r->level)))
            return false;
    }

    id<MTLTexture> s = (__bridge id<MTLTexture>)st->mtl_data;
    id<MTLTexture> d = (__bridge id<MTLTexture>)dt->mtl_data;
    bool s3d = s.textureType == MTLTextureType3D, d3d = d.textureType == MTLTextureType3D;
    NSUInteger n = (NSUInteger)sr->d;

    if (s.sampleCount != d.sampleCount)
        return false;

    if (s.pixelFormat == d.pixelFormat)
    {
        id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
        if (s3d && d3d)
        {
            [blit copyFromTexture:s sourceSlice:0 sourceLevel:sr->level sourceOrigin:MTLOriginMake(sr->x, sr->y, sr->z)
                       sourceSize:MTLSizeMake(sr->w, sr->h, n) toTexture:d destinationSlice:0 destinationLevel:dr->level
                destinationOrigin:MTLOriginMake(dr->x, dr->y, dr->z)];
        }
        else
        {
            for (NSUInteger i = 0; i < n; i++)
                [blit copyFromTexture:s sourceSlice:s3d ? 0 : td_slice(st, sr->face, sr->z + (GLint)i) sourceLevel:sr->level
                         sourceOrigin:MTLOriginMake(sr->x, sr->y, s3d ? sr->z + i : 0)
                           sourceSize:MTLSizeMake(sr->w, sr->h, 1)
                            toTexture:d destinationSlice:d3d ? 0 : td_slice(dt, dr->face, dr->z + (GLint)i) destinationLevel:dr->level
                    destinationOrigin:MTLOriginMake(dr->x, dr->y, d3d ? dr->z + i : 0)];
        }
        [blit endEncoding];
        return true;
    }

    /* reinterpretation (same texel / block size): bounce through a buffer */
    if (s.sampleCount > 1)
        return false;
    const TDFormat *sf = mgl_texdata_storage_format(st->target == GL_RENDERBUFFER ? st->internalformat
                                                                                  : st->faces[sr->face].levels[sr->level].mtl_format);
    if (!sf || sf->kind == TDK_D32FS8)
        return false;

    bool sc = (sf->flags & TDF_COMPRESSED) != 0;
    NSUInteger rows = sc ? ((NSUInteger)sr->h + sf->block_h - 1) / sf->block_h : (NSUInteger)sr->h;
    NSUInteger rowbytes = sc ? tdStorageRowBytes(sf, sr->w) : (NSUInteger)sr->w * sf->bpp;
    NSUInteger img = rowbytes * rows;
    id<MTLBuffer> buf = [_device newBufferWithLength:MAX(img * n, 16) options:MTLResourceStorageModePrivate];

    id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
    for (NSUInteger i = 0; i < n; i++)
        [blit copyFromTexture:s sourceSlice:s3d ? 0 : td_slice(st, sr->face, sr->z + (GLint)i) sourceLevel:sr->level
                 sourceOrigin:MTLOriginMake(sr->x, sr->y, s3d ? sr->z + i : 0) sourceSize:MTLSizeMake(sr->w, sr->h, 1)
                     toBuffer:buf destinationOffset:i * img destinationBytesPerRow:rowbytes destinationBytesPerImage:img];
    for (NSUInteger i = 0; i < n; i++)
        [blit copyFromBuffer:buf sourceOffset:i * img sourceBytesPerRow:rowbytes sourceBytesPerImage:img
                  sourceSize:MTLSizeMake(dr->w, dr->h, 1) toTexture:d
            destinationSlice:d3d ? 0 : td_slice(dt, dr->face, dr->z + (GLint)i) destinationLevel:dr->level
           destinationOrigin:MTLOriginMake(dr->x, dr->y, d3d ? dr->z + i : 0)];
    [blit endEncoding];
    return true;
}

- (bool) texdataGenerateMipmaps:(Texture *)tex base:(GLuint)base last:(GLuint)last
{
    if (![self texdataRealize:tex] || !tex->mtl_data)
        return false;

    id<MTLTexture> t = (__bridge id<MTLTexture>)tex->mtl_data;
    if (last >= t.mipmapLevelCount)
        last = (GLuint)t.mipmapLevelCount - 1;
    if (last <= base)
        return true;

    id<MTLTexture> target = t;
    if (base != 0 || last != t.mipmapLevelCount - 1)
    {
        NSUInteger slices = t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
        if (t.textureType == MTLTextureType3D)
            slices = 1;
        target = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:t.textureType
                                           levels:NSMakeRange(base, last - base + 1) slices:NSMakeRange(0, slices)];
        if (!target)
            return false;
    }

    id<MTLBlitCommandEncoder> blit = [self texdataBlitEncoder];
    [blit generateMipmapsForTexture:target];
    [blit endEncoding];
    return true;
}

@end

#pragma mark - C hooks

static void td_upload(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *src, bool depth, bool stencil)
{
    const TDFormat *sf = mgl_texdata_storage_format(tex->faces[r->face].levels[r->level].mtl_format);
    if (!sf || !tex->mtl_data)
        return;
    @autoreleasepool {
        [TDR(ctx) texdataEncodeUpload:(__bridge id<MTLTexture>)tex->mtl_data tex:tex format:sf region:r src:src depth:depth stencil:stencil];
    }
}

static bool td_download(GLMContext ctx, Texture *tex, const TDRegion *r, const TDStorage *dst)
{
    const TDFormat *sf = mgl_texdata_storage_format(tex->faces[r->face].levels[r->level].mtl_format);
    if (!sf || !tex->mtl_data)
        return false;
    @autoreleasepool {
        return [TDR(ctx) texdataDownload:(__bridge id<MTLTexture>)tex->mtl_data tex:tex format:sf region:r dst:dst];
    }
}

static bool td_download_decoded(GLMContext ctx, Texture *tex, const TDRegion *r, float *dst)
{
    @autoreleasepool { return [TDR(ctx) texdataDownloadDecoded:tex region:r dst:dst]; }
}

static bool td_realize(GLMContext ctx, Texture *tex)
{
    @autoreleasepool { return [TDR(ctx) texdataRealize:tex]; }
}

static GLenum td_copy_from_read_fb(GLMContext ctx, Texture *dst, const TDRegion *r, GLint x, GLint y, GLsizei w, GLsizei h, bool depth, bool stencil)
{
    @autoreleasepool { return [TDR(ctx) texdataCopyFromReadFB:dst region:r x:x y:y width:w height:h depth:depth stencil:stencil]; }
}

static bool td_copy_image(GLMContext ctx, Texture *src, const TDRegion *sr, Texture *dst, const TDRegion *dr)
{
    @autoreleasepool { return [TDR(ctx) texdataCopyImage:src region:sr to:dst region:dr]; }
}

static bool td_generate_mipmaps(GLMContext ctx, Texture *tex, GLuint base, GLuint last)
{
    @autoreleasepool { return [TDR(ctx) texdataGenerateMipmaps:tex base:base last:last]; }
}

static void td_release(GLMContext ctx, Texture *tex)
{
    if (tex->mtl_data)
    {
        CFBridgingRelease(tex->mtl_data);
        tex->mtl_data = NULL;
    }
    if (tex->params.mtl_data)
    {
        CFBridgingRelease(tex->params.mtl_data);
        tex->params.mtl_data = NULL;
    }
    mglTexParamInvalidate(ctx, &tex->params);
}

static bool td_read_buffer_info(GLMContext ctx, bool depth, bool stencil, GLenum *ifmt, GLuint *w, GLuint *h)
{
    @autoreleasepool { return [TDR(ctx) texdataReadBufferInfo:(depth || stencil) internalformat:ifmt width:w height:h]; }
}

static void *td_buffer_contents(GLMContext ctx, Buffer *buf)
{
    (void)ctx;
    if (buf->data.mtl_data)
        return [(__bridge id<MTLBuffer>)buf->data.mtl_data contents];
    return (void *)buf->data.buffer_data;
}

@implementation MGLRenderer (TextureDataBinding)

- (void) bindExtTexdataFuncsToGLMContext: (GLMContext) glm_ctx
{
    static struct MGLExtMetal_texdata hooks = {
        td_upload, td_download, td_download_decoded, td_realize, td_copy_from_read_fb,
        td_copy_image, td_generate_mipmaps, td_release, td_read_buffer_info, td_buffer_contents
    };

    glm_ctx->mtl_funcs.ext_texdata = &hooks;

    /* the binder also runs at context build, before _device is attached; there it
     * only leaves the safe defaults (etc decode on, bc on, astc off) */
    id<MTLDevice> dev = _device;
    bool etc = false, bc = true;
    if (dev && @available(macOS 11.0, *))
    {
        etc = [dev supportsFamily:MTLGPUFamilyApple1];
        bc = dev.supportsBCTextureCompression;
    }
    tdSetDeviceCaps(etc, bc);
    /* ASTC LDR: Apple-family GPUs (Apple silicon); Intel Macs have none */
    tdSetASTCCap(dev && [dev supportsFamily:MTLGPUFamilyApple2]);
}

- (void) texdataSetDeviceCaps
{
    [self bindExtTexdataFuncsToGLMContext: ctx];
}

@end
