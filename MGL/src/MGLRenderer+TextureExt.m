/*
 * MGLRenderer+TextureExt.m
 * MGL — Metal side of the 'texext' domain: buffer textures, texture views, multisample textures,
 * texture invalidation and TextureBarrier.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import <objc/runtime.h>
#import "MGLRenderer_Private.h"
#import "MGLRenderer+TextureExt.h"

#include "hash_table.h"
#include "mgl_ext_texext.h"

/* pixel_utils.c returns its own 32-bit MTLPixelFormat enum */
extern uint32_t mtlFormatForGLInternalFormat(GLenum internal_format);

static const void *kTexextPendingInvalidateKey = &kTexextPendingInvalidateKey;

static MTLPixelFormat texextBufferPixelFormat(GLenum f)
{
    switch (f)
    {
        case GL_R8: return MTLPixelFormatR8Unorm;
        case GL_R16: return MTLPixelFormatR16Unorm;
        case GL_R16F: return MTLPixelFormatR16Float;
        case GL_R32F: return MTLPixelFormatR32Float;
        case GL_R8I: return MTLPixelFormatR8Sint;
        case GL_R16I: return MTLPixelFormatR16Sint;
        case GL_R32I: return MTLPixelFormatR32Sint;
        case GL_R8UI: return MTLPixelFormatR8Uint;
        case GL_R16UI: return MTLPixelFormatR16Uint;
        case GL_R32UI: return MTLPixelFormatR32Uint;
        case GL_RG8: return MTLPixelFormatRG8Unorm;
        case GL_RG16: return MTLPixelFormatRG16Unorm;
        case GL_RG16F: return MTLPixelFormatRG16Float;
        case GL_RG32F: return MTLPixelFormatRG32Float;
        case GL_RG8I: return MTLPixelFormatRG8Sint;
        case GL_RG16I: return MTLPixelFormatRG16Sint;
        case GL_RG32I: return MTLPixelFormatRG32Sint;
        case GL_RG8UI: return MTLPixelFormatRG8Uint;
        case GL_RG16UI: return MTLPixelFormatRG16Uint;
        case GL_RG32UI: return MTLPixelFormatRG32Uint;
        case GL_RGB32F: return MTLPixelFormatRGBA32Float;   /* expanded */
        case GL_RGB32I: return MTLPixelFormatRGBA32Sint;    /* expanded */
        case GL_RGB32UI: return MTLPixelFormatRGBA32Uint;   /* expanded */
        case GL_RGBA8: return MTLPixelFormatRGBA8Unorm;
        case GL_RGBA16: return MTLPixelFormatRGBA16Unorm;
        case GL_RGBA16F: return MTLPixelFormatRGBA16Float;
        case GL_RGBA32F: return MTLPixelFormatRGBA32Float;
        case GL_RGBA8I: return MTLPixelFormatRGBA8Sint;
        case GL_RGBA16I: return MTLPixelFormatRGBA16Sint;
        case GL_RGBA32I: return MTLPixelFormatRGBA32Sint;
        case GL_RGBA8UI: return MTLPixelFormatRGBA8Uint;
        case GL_RGBA16UI: return MTLPixelFormatRGBA16Uint;
        case GL_RGBA32UI: return MTLPixelFormatRGBA32Uint;
    }
    return MTLPixelFormatInvalid;
}

/* block-compressed formats cannot be render targets */
static BOOL MTLPixelFormatIsCompressedTexext(MTLPixelFormat f)
{
    return (f >= MTLPixelFormatBC1_RGBA && f <= MTLPixelFormatBC7_RGBAUnorm_sRGB) ||
           (f >= MTLPixelFormatEAC_R11Unorm && f <= MTLPixelFormatETC2_RGB8A1_sRGB) ||
           (f >= MTLPixelFormatASTC_4x4_sRGB && f <= MTLPixelFormatASTC_12x12_HDR);
}

static NSUInteger texextAlignUp(NSUInteger v, NSUInteger a)
{
    if (a <= 1) return v;
    return (v + a - 1) / a * a;
}

#pragma mark - C trampolines

static void texextTextureBarrierC(GLMContext glm_ctx);
static void texextInvalidateC(GLMContext glm_ctx, Texture *tex, GLint level);
static GLsizei texextSampleCountForC(GLMContext glm_ctx, GLenum internalformat, GLsizei samples);
static void texextPrepareViewSourceC(GLMContext glm_ctx, Texture *root);

static struct MGLExtMetal_texext s_texext_metal_funcs = {
    texextTextureBarrierC,
    texextInvalidateC,
    texextSampleCountForC,
    texextPrepareViewSourceC,
};

@implementation MGLRenderer (TextureExt)

#pragma mark - binding / limits

- (void) bindExtTexextFuncsToGLMContext: (GLMContext) glm_ctx
{
    glm_ctx->mtl_funcs.ext_texext = &s_texext_metal_funcs;

    /* multisample limits from the Metal device */
    GLuint max_samples = 1;
    for (GLuint n = 2; n <= 8; n <<= 1)
        if ([_device supportsTextureSampleCount: n])
            max_samples = n;

    glm_ctx->state.var.max_color_texture_samples = max_samples;
    glm_ctx->state.var.max_depth_texture_samples = max_samples;
    glm_ctx->state.var.max_integer_samples = max_samples;

    /* TEXTURE_BUFFER_OFFSET_ALIGNMENT: the strictest Metal texture-buffer alignment over all buffer formats,
       so any GL-aligned offset can be used directly with -newTextureWithDescriptor:offset:bytesPerRow: */
    static const GLenum fmts[] = {
        GL_R8, GL_R16, GL_R16F, GL_R32F, GL_RG8, GL_RG16, GL_RG16F, GL_RG32F, GL_RGBA8, GL_RGBA16, GL_RGBA16F,
        GL_RGBA32F, GL_R8UI, GL_R16UI, GL_R32UI, GL_RG32UI, GL_RGBA32UI, GL_RGBA8I, GL_RGBA16I, GL_RGBA32I
    };
    NSUInteger align = 4;
    for (size_t i = 0; i < sizeof(fmts) / sizeof(fmts[0]); i++)
    {
        NSUInteger a = [_device minimumTextureBufferAlignmentForPixelFormat: texextBufferPixelFormat(fmts[i])];
        if (a > align) align = a;
    }
    NSUInteger p2 = 1;
    while (p2 < align) p2 <<= 1;
    if (p2 > 256) p2 = 256;
    glm_ctx->state.var.texture_buffer_offset_alignment = (GLuint)p2;

    /* Metal texture buffers: 2^26 texels is supported by every Metal GPU family */
    GLuint max_tb = 1u << 26;
    if (glm_ctx->state.var.max_texture_buffer_size == 0 || glm_ctx->state.var.max_texture_buffer_size > max_tb)
        glm_ctx->state.var.max_texture_buffer_size = max_tb;
}

#pragma mark - ownership

- (BOOL) texextOwnsTexture: (Texture *) tex
{
    if (!tex)
        return NO;

    switch (tex->target)
    {
        case GL_TEXTURE_BUFFER:
        case GL_TEXTURE_2D_MULTISAMPLE:
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return YES;
    }

    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);
    return info && (info->kind == MGL_TEXEXT_BUFFER || info->kind == MGL_TEXEXT_VIEW || info->kind == MGL_TEXEXT_MULTISAMPLE);
}

- (id<MTLTexture>) texextCreateMTLTexture: (Texture *) tex
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);
    id<MTLTexture> texture = nil;

    if (info && info->kind == MGL_TEXEXT_VIEW)
        texture = [self texextCreateViewTexture: tex info: info];
    else if (tex->target == GL_TEXTURE_BUFFER)
        texture = [self texextCreateBufferTexture: tex info: info];
    else if (tex->target == GL_TEXTURE_2D_MULTISAMPLE || tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
        texture = [self texextCreateMultisampleTexture: tex info: info];

    if (texture)
    {
        tex->dirty_bits = 0;
        tex->complete = true;
    }

    return texture;
}

#pragma mark - buffer textures

/* authoritative bytes of a GL buffer. MGL keeps small buffers (< 4096 bytes) in a CPU copy next to
   the MTLBuffer; glBufferSubData writes the CPU copy for DYNAMIC/CLIENT storage and the MTLBuffer
   otherwise (buffers.c), so follow the same rule. Large buffers live in the MTLBuffer's contents. */
- (const uint8_t *) texextBufferBytes: (Buffer *) buf mtl: (id<MTLBuffer>) mbuf separate: (BOOL *) separate
{
    const uint8_t *cpu = (const uint8_t *)buf->data.buffer_data;
    const uint8_t *gpu = mbuf ? (const uint8_t *)mbuf.contents : NULL;

    if (cpu && cpu != gpu && (!gpu || (buf->storage_flags & (GL_CLIENT_STORAGE_BIT | GL_DYNAMIC_STORAGE_BIT))))
    {
        if (separate) *separate = YES;
        return cpu;
    }

    if (separate) *separate = NO;

    if (buf->size >= 4096 && (buf->data.dirty_bits & DIRTY_BUFFER_DATA))
        [self updateDirtyBuffer: buf];

    return gpu;
}

- (id<MTLTexture>) texextZeroBufferTexture: (MTLPixelFormat) fmt
{
    NSUInteger align = [_device minimumTextureBufferAlignmentForPixelFormat: fmt];
    NSUInteger len = texextAlignUp(16, align);
    id<MTLBuffer> zero = [_device newBufferWithLength: len options: MTLResourceStorageModeShared];
    if (!zero)
        return nil;
    memset(zero.contents, 0, len);

    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat: fmt
                                                                                        width: 1
                                                                              resourceOptions: zero.resourceOptions
                                                                                        usage: MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    return [zero newTextureWithDescriptor: desc offset: 0 bytesPerRow: len];
}

- (id<MTLTexture>) texextCreateBufferTexture: (Texture *) tex info: (MGLTexExtInfo *) info
{
    GLenum glfmt = (info && info->kind == MGL_TEXEXT_BUFFER) ? info->buffer_format : GL_R8;
    MTLPixelFormat fmt = texextBufferPixelFormat(glfmt);
    if (fmt == MTLPixelFormatInvalid)
        fmt = MTLPixelFormatR8Unorm;

    if (!info || info->kind != MGL_TEXEXT_BUFFER || info->buffer_name == 0)
    {
        if (info) { info->mtl_built_from = NULL; info->mtl_built_size = 0; info->mtl_uses_shadow = false; }
        return [self texextZeroBufferTexture: fmt];
    }

    Buffer *buf = (Buffer *)searchHashTable(&ctx->state.buffer_table, info->buffer_name);
    if (!buf || buf->size <= 0)
    {
        info->mtl_built_from = NULL;
        info->mtl_built_size = 0;
        info->mtl_uses_shadow = false;
        tex->width = 0;
        return [self texextZeroBufferTexture: fmt];
    }

    if (buf->data.mtl_data == NULL)
        [self bindMTLBuffer: buf];

    id<MTLBuffer> mbuf = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
    if (!mbuf)
        return [self texextZeroBufferTexture: fmt];

    NSUInteger texel_size = info->buffer_texel_size;
    NSUInteger offset = info->buffer_ranged ? (NSUInteger)info->buffer_offset : 0;
    NSUInteger bytes = info->buffer_ranged ? (NSUInteger)info->buffer_size : (NSUInteger)buf->size;
    if (offset > mbuf.length) offset = mbuf.length;
    if (offset + bytes > mbuf.length) bytes = mbuf.length - offset;

    NSUInteger texels = bytes / texel_size;
    GLuint max_tb = ctx->state.var.max_texture_buffer_size;
    if (max_tb && texels > max_tb)
        texels = max_tb;

    /* TexBuffer (no range) follows the buffer's size */
    tex->width = (GLuint)texels;

    info->mtl_built_from = (__bridge void *)mbuf;
    info->mtl_built_size = buf->size;

    if (texels == 0)
    {
        info->mtl_uses_shadow = false;
        return [self texextZeroBufferTexture: fmt];
    }

    NSUInteger align = [_device minimumTextureBufferAlignmentForPixelFormat: fmt];
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    BOOL separate = NO;
    const uint8_t *src = [self texextBufferBytes: buf mtl: mbuf separate: &separate];

    if (!info->buffer_expand3 && !separate)
    {
        NSUInteger bpr = texextAlignUp(texels * texel_size, align);

        if ((offset % align) == 0 && offset + bpr <= mbuf.length)
        {
            /* zero-copy: the texture aliases the GL buffer's MTLBuffer, so it follows its contents */
            MTLTextureDescriptor *desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat: fmt
                                                                                                width: texels
                                                                                      resourceOptions: mbuf.resourceOptions
                                                                                                usage: usage];
            id<MTLTexture> t = [mbuf newTextureWithDescriptor: desc offset: offset bytesPerRow: bpr];
            if (t)
            {
                info->mtl_uses_shadow = false;
                if (info->shadow_src) { free(info->shadow_src); info->shadow_src = NULL; info->shadow_src_size = 0; }
                return t;
            }
        }
    }

    /* shadow path: RGB32 expansion (Metal has no 96-bit formats), a buffer whose data lives in a
       separate CPU copy, or a range Metal cannot alias (offset/row padding past the end of the
       buffer). The source bytes are snapshotted and the shadow is rebuilt by
       -texextValidateTexture: whenever they change. */
    if (!src)
        return [self texextZeroBufferTexture: fmt];
    src += offset;

    size_t src_bytes = texels * texel_size;
    if (info->shadow_src_size != src_bytes)
    {
        free(info->shadow_src);
        info->shadow_src = malloc(src_bytes);
        info->shadow_src_size = info->shadow_src ? src_bytes : 0;
    }
    if (info->shadow_src)
        memcpy(info->shadow_src, src, src_bytes);

    NSUInteger out_ts = info->buffer_expand3 ? 16 : texel_size;
    NSUInteger bpr = texextAlignUp(texels * out_ts, align);
    id<MTLBuffer> shadow = [_device newBufferWithLength: bpr options: MTLResourceStorageModeShared];
    if (!shadow)
        return [self texextZeroBufferTexture: fmt];

    uint8_t *dst = (uint8_t *)shadow.contents;
    memset(dst, 0, bpr);

    if (info->buffer_expand3)
    {
        uint32_t one = (glfmt == GL_RGB32F) ? 0x3f800000u : 1u; /* alpha = 1 (Table 8.15) */
        for (NSUInteger i = 0; i < texels; i++)
        {
            memcpy(dst + i * 16, src + i * 12, 12);
            memcpy(dst + i * 16 + 12, &one, 4);
        }
    }
    else
    {
        memcpy(dst, src, src_bytes);
    }

    info->mtl_uses_shadow = true;

    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat: fmt
                                                                                        width: texels
                                                                              resourceOptions: shadow.resourceOptions
                                                                                        usage: usage];
    return [shadow newTextureWithDescriptor: desc offset: 0 bytesPerRow: bpr];
}

#pragma mark - multisample textures

- (id<MTLTexture>) texextCreateMultisampleTexture: (Texture *) tex info: (MGLTexExtInfo *) info
{
    if (!info || info->kind != MGL_TEXEXT_MULTISAMPLE || tex->width == 0 || tex->height == 0)
        return nil;

    MTLPixelFormat fmt = (MTLPixelFormat)mtlFormatForGLInternalFormat(tex->internalformat);
    if (fmt == MTLPixelFormatInvalid)
        return nil;

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    BOOL is_array = (tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY);

    desc.textureType = is_array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
    desc.pixelFormat = fmt;
    desc.width = tex->width;
    desc.height = tex->height;
    desc.depth = 1;
    desc.arrayLength = is_array ? MAX(tex->depth, 1) : 1;
    desc.mipmapLevelCount = 1;
    desc.sampleCount = MAX(info->samples, 1);
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    if (info->has_format_views)
        desc.usage |= MTLTextureUsagePixelFormatView;

    return [_device newTextureWithDescriptor: desc];
}

- (GLsizei) texextSampleCountFor: (GLenum) internalformat samples: (GLsizei) samples
{
    /* Metal multisample textures need sampleCount > 1; GL allows more samples than requested */
    for (GLsizei n = MAX(samples, 2); n <= 8; n++)
        if ([_device supportsTextureSampleCount: n])
            return n;
    return 0;
}

#pragma mark - texture views

- (MTLTextureType) texextViewType: (GLenum) target root: (id<MTLTexture>) root
{
    switch (target)
    {
        case GL_TEXTURE_1D:
            return root.textureType == MTLTextureType1DArray || root.textureType == MTLTextureType1D ? MTLTextureType1D : MTLTextureType2D;
        case GL_TEXTURE_1D_ARRAY:
            return root.textureType == MTLTextureType1DArray || root.textureType == MTLTextureType1D ? MTLTextureType1DArray : MTLTextureType2DArray;
        case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: return MTLTextureType2D;
        case GL_TEXTURE_2D_ARRAY: return MTLTextureType2DArray;
        case GL_TEXTURE_3D: return MTLTextureType3D;
        case GL_TEXTURE_CUBE_MAP: return MTLTextureTypeCube;
        case GL_TEXTURE_CUBE_MAP_ARRAY: return MTLTextureTypeCubeArray;
        case GL_TEXTURE_2D_MULTISAMPLE: return MTLTextureType2DMultisample;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return MTLTextureType2DMultisampleArray;
    }
    return root.textureType;
}

- (id<MTLTexture>) texextCreateViewTexture: (Texture *) tex info: (MGLTexExtInfo *) info
{
    Texture *root = info->view_root;
    if (!root)
        return nil;

    [self bindMTLTexture: root];
    id<MTLTexture> rt = (__bridge id<MTLTexture>)(root->mtl_data);
    if (!rt)
        return nil;

    MTLPixelFormat fmt = (MTLPixelFormat)mtlFormatForGLInternalFormat(tex->internalformat);
    if (fmt == MTLPixelFormatInvalid)
        fmt = rt.pixelFormat;

    MTLTextureType type = [self texextViewType: tex->target root: rt];

    NSUInteger root_slices = rt.arrayLength * ((rt.textureType == MTLTextureTypeCube || rt.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    NSRange levels = NSMakeRange(info->view_min_level, info->view_num_levels);
    NSRange slices;

    if (rt.textureType == MTLTextureType3D)
        slices = NSMakeRange(0, 1);
    else
        slices = NSMakeRange(info->view_min_layer, info->view_num_layers);

    /* clamp to what the root Metal texture really has */
    if (levels.location >= rt.mipmapLevelCount)
        levels.location = rt.mipmapLevelCount - 1;
    if (levels.location + levels.length > rt.mipmapLevelCount)
        levels.length = rt.mipmapLevelCount - levels.location;
    if (slices.location >= root_slices)
        slices.location = root_slices - 1;
    if (slices.location + slices.length > root_slices)
        slices.length = root_slices - slices.location;

    id<MTLTexture> view = nil;
    @try {
        view = [rt newTextureViewWithPixelFormat: fmt textureType: type levels: levels slices: slices];
    } @catch (NSException *e) {
        NSLog(@"MGL texext: texture view creation failed: %@", e.reason);
        view = nil;
    }

    if (!view)
        NSLog(@"MGL texext: view of texture %u (type %lu fmt %lu levels %lu+%lu slices %lu+%lu) from root type %lu fmt %lu levels %lu slices %lu failed",
              tex->name, (unsigned long)type, (unsigned long)fmt, (unsigned long)levels.location, (unsigned long)levels.length,
              (unsigned long)slices.location, (unsigned long)slices.length, (unsigned long)rt.textureType,
              (unsigned long)rt.pixelFormat, (unsigned long)rt.mipmapLevelCount, (unsigned long)root_slices);

    info->view_built_from = (__bridge void *)rt;
    return view;
}

/* re-create root with MTLTextureUsagePixelFormatView, copying its contents (called outside draws) */
- (void) texextPrepareViewSource: (Texture *) root
{
    id<MTLTexture> old = (__bridge id<MTLTexture>)(root->mtl_data);
    MGLTexExtInfo *rinfo = mgl_texext_info(ctx, root);
    if (!old || !rinfo)
        return;

    MTLTextureUsage want = old.usage | MTLTextureUsageShaderRead;
    if (rinfo->has_format_views)
        want |= MTLTextureUsagePixelFormatView;
    if (!old.isFramebufferOnly && old.storageMode == MTLStorageModePrivate && !MTLPixelFormatIsCompressedTexext(old.pixelFormat))
        want |= MTLTextureUsageRenderTarget;
    if (want == old.usage)
        return;

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = old.textureType;
    desc.pixelFormat = old.pixelFormat;
    desc.width = old.width;
    desc.height = old.height;
    desc.depth = old.depth;
    desc.mipmapLevelCount = old.mipmapLevelCount;
    desc.arrayLength = old.arrayLength;
    desc.sampleCount = old.sampleCount;
    desc.storageMode = old.storageMode;
    desc.cpuCacheMode = old.cpuCacheMode;
    desc.usage = want;
    desc.swizzle = old.swizzle;

    id<MTLTexture> fresh = [_device newTextureWithDescriptor: desc];
    if (!fresh)
        return;

    BOOL had_encoder = (_currentRenderEncoder != nil);
    [self endRenderEncoding];

    id<MTLCommandBuffer> cb = _currentCommandBuffer;
    BOOL own_cb = NO;
    if (!cb || cb.status >= MTLCommandBufferStatusCommitted)
    {
        cb = [_commandQueue commandBuffer];
        own_cb = YES;
    }

    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture: old toTexture: fresh];
    [blit endEncoding];

    if (own_cb)
        [cb commit];

    CFBridgingRelease(root->mtl_data);
    root->mtl_data = (void *)CFBridgingRetain(fresh);

    if (had_encoder)
        [self newRenderEncoder];

    ctx->state.dirty_bits |= DIRTY_TEX | DIRTY_TEX_BINDING;
}

#pragma mark - hooks called from MGLRenderer.m

- (void) texextAdjustTextureDescriptor: (MTLTextureDescriptor *) desc forTexture: (Texture *) tex
{
    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);

    if (info && info->has_format_views)
        desc.usage |= MTLTextureUsagePixelFormatView;
    if (info && info->has_views)
        desc.usage |= MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
}

- (void) texextValidateTexture: (Texture *) tex
{
    if (!tex || tex->dirty_bits || !tex->mtl_data)
        return;

    MGLTexExtInfo *info = mgl_texext_info(ctx, tex);
    if (!info)
        return;

    if (info->kind == MGL_TEXEXT_BUFFER)
    {
        Buffer *buf = info->buffer_name ? (Buffer *)searchHashTable(&ctx->state.buffer_table, info->buffer_name) : NULL;
        void *cur = buf ? buf->data.mtl_data : NULL;

        if (buf && !cur)
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;               /* buffer storage (re)allocated, no MTLBuffer yet */
        else if (cur != info->mtl_built_from)
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;               /* buffer storage reallocated / deleted */
        else if (buf && buf->size != info->mtl_built_size)
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
        else if (cur && info->mtl_uses_shadow && info->shadow_src)
        {
            id<MTLBuffer> mbuf = (__bridge id<MTLBuffer>)cur;
            NSUInteger offset = info->buffer_ranged ? (NSUInteger)info->buffer_offset : 0;
            const uint8_t *src = [self texextBufferBytes: buf mtl: mbuf separate: NULL];

            if (src && offset + info->shadow_src_size <= (NSUInteger)buf->size &&
                memcmp(src + offset, info->shadow_src, info->shadow_src_size) != 0)
                tex->dirty_bits |= DIRTY_TEXTURE_DATA;           /* contents changed: rebuild shadow */
        }
    }
    else if (info->kind == MGL_TEXEXT_VIEW && info->view_root)
    {
        Texture *root = info->view_root;

        if (root->dirty_bits || root->mtl_data == NULL || root->mtl_data != info->view_built_from)
            tex->dirty_bits |= DIRTY_TEXTURE_DATA;               /* root storage recreated: re-make the view */
    }
}

- (void) texextAdjustPipelineDescriptor: (MTLRenderPipelineDescriptor *) desc
{
    Framebuffer *fbo = ctx->state.framebuffer;
    NSUInteger samples = 1;

    if (!fbo)
        return;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        if (fbo->color_attachments[i].texture && fbo->color_attachments[i].buf.tex)
        {
            Texture *t = [self framebufferAttachmentTexture: &fbo->color_attachments[i]];
            id<MTLTexture> mt = t ? (__bridge id<MTLTexture>)(t->mtl_data) : nil;
            if (mt && mt.sampleCount > samples)
                samples = mt.sampleCount;
        }
    }

    FBOAttachment *ds[2] = { &fbo->depth, &fbo->stencil };
    for (int i = 0; i < 2; i++)
    {
        if (ds[i]->texture && ds[i]->buf.tex)
        {
            Texture *t = [self framebufferAttachmentTexture: ds[i]];
            id<MTLTexture> mt = t ? (__bridge id<MTLTexture>)(t->mtl_data) : nil;
            if (mt && mt.sampleCount > samples)
                samples = mt.sampleCount;
        }
    }

    if (samples > 1)
        desc.rasterSampleCount = samples;
}

#pragma mark - TextureBarrier

- (void) texextTextureBarrier
{
    /* work in separate encoders / command buffers is already ordered by Metal's hazard tracking */
    if (!_currentRenderEncoder)
        return;

    if (![_device supportsFamily: MTLGPUFamilyApple1] &&
        [_currentRenderEncoder respondsToSelector: @selector(memoryBarrierWithScope:afterStages:beforeStages:)])
    {
        /* immediate-mode GPUs: in-pass barrier between render-target writes and texture reads */
        [_currentRenderEncoder memoryBarrierWithScope: MTLBarrierScopeRenderTargets | MTLBarrierScopeTextures
                                          afterStages: MTLRenderStageFragment
                                         beforeStages: MTLRenderStageVertex | MTLRenderStageFragment];
        return;
    }

    /* tile-based (Apple) GPUs: render-target data reaches memory only at the end of the pass, so
       split the pass: store now, continue in a new pass that loads the attachments */
    [self endRenderEncoding];
    [self newRenderEncoder];
}

#pragma mark - invalidation

- (NSMutableArray *) texextPendingInvalidations
{
    NSMutableArray *a = objc_getAssociatedObject(self, kTexextPendingInvalidateKey);
    if (!a)
    {
        a = [NSMutableArray array];
        objc_setAssociatedObject(self, kTexextPendingInvalidateKey, a, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    return a;
}

- (void) texextInvalidate: (Texture *) tex level: (GLint) level
{
    id<MTLTexture> mt = tex->mtl_data ? (__bridge id<MTLTexture>)(tex->mtl_data) : nil;
    if (!mt || !(mt.usage & MTLTextureUsageRenderTarget))
        return;  /* not a render target: there is no load to skip; contents are simply undefined */

    [[self texextPendingInvalidations] addObject: @[ mt, @(level) ]];
}

static BOOL texextTakeInvalidated(NSMutableArray *pending, id<MTLTexture> t, NSUInteger level)
{
    if (!t)
        return NO;

    for (NSUInteger i = 0; i < pending.count; i++)
    {
        NSArray *e = pending[i];
        if (e[0] == t && [e[1] unsignedIntegerValue] == level)
        {
            [pending removeObjectAtIndex: i];
            return YES;
        }
    }
    return NO;
}

/* ext hook: first render pass after an invalidate doesn't need to load the old contents */
- (void) extTexextRenderPassDescriptor: (MTLRenderPassDescriptor *) desc
{
    NSMutableArray *pending = objc_getAssociatedObject(self, kTexextPendingInvalidateKey);
    if (!pending.count)
        return;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
    {
        MTLRenderPassColorAttachmentDescriptor *ca = desc.colorAttachments[i];
        if (ca.texture && ca.loadAction == MTLLoadActionLoad && texextTakeInvalidated(pending, ca.texture, ca.level))
            ca.loadAction = MTLLoadActionDontCare;
    }

    if (desc.depthAttachment.texture && desc.depthAttachment.loadAction == MTLLoadActionLoad &&
        texextTakeInvalidated(pending, desc.depthAttachment.texture, desc.depthAttachment.level))
        desc.depthAttachment.loadAction = MTLLoadActionDontCare;

    if (desc.stencilAttachment.texture && desc.stencilAttachment.loadAction == MTLLoadActionLoad &&
        texextTakeInvalidated(pending, desc.stencilAttachment.texture, desc.stencilAttachment.level))
        desc.stencilAttachment.loadAction = MTLLoadActionDontCare;
}

/* ext hook: pending invalidations only apply to passes recorded in the same command buffer
   (so a later blit/copy into the texture can never be discarded by a stale entry) */
- (void) extTexextWillCommit: (id<MTLCommandBuffer>) cmdbuf
{
    NSMutableArray *pending = objc_getAssociatedObject(self, kTexextPendingInvalidateKey);
    [pending removeAllObjects];
}

#pragma mark - per-draw texture refresh

- (void) texextRefreshStage: (int) stage encoder: (id<MTLRenderCommandEncoder>) enc
{
    int count = [self getProgramBindingCount: stage type: SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];

    for (int i = 0; i < count; i++)
    {
        int binding = [self getProgramBinding: stage type: SPVC_RESOURCE_TYPE_SAMPLED_IMAGE index: i];
        if (binding < 0 || binding >= TEXTURE_UNITS)
            continue;

        Texture *ptr = ctx->state.active_textures[binding];
        if (!ptr)
            continue;

        BOOL rebind = (stage != _FRAGMENT_SHADER);  /* MGL core binds fragment textures only */

        if (mgl_texext_info(ctx, ptr))
        {
            [self texextValidateTexture: ptr];
            if (ptr->dirty_bits)
                rebind = YES;
        }

        if (!rebind)
            continue;

        if (![self bindMTLTexture: ptr] || !ptr->mtl_data)
            continue;

        id<MTLTexture> texture = (__bridge id<MTLTexture>)(ptr->mtl_data);
        id<MTLSamplerState> sampler = nil;
        Sampler *gl_sampler = ctx->state.texture_samplers[binding];
        if (gl_sampler && gl_sampler->mtl_data)
            sampler = (__bridge id<MTLSamplerState>)(gl_sampler->mtl_data);
        else
            sampler = [self texparamCachedSampler:&ptr->params];   /* mtl_data may hold the param cache, not a sampler */

        /* bind at the program's Metal texture/sampler slot, NOT the GL unit:
         * SPIRV-Cross assigns them independently of the units, and a sampler
         * index >= 16 (unit numbers reach 31) crashes the encoder */
        GLuint msl_slot = (GLuint)binding;
        if (STATE(program))
        {
            SpirvResourceList *rl = &STATE(program)->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];
            if ((GLuint)i < rl->count && rl->list[i].msl_binding != 0xffffffffu)
                msl_slot = rl->list[i].msl_binding;
        }

        if (stage == _FRAGMENT_SHADER)
        {
            [enc setFragmentTexture: texture atIndex: msl_slot];
            if (sampler) [enc setFragmentSamplerState: sampler atIndex: msl_slot];
        }
        else
        {
            [enc setVertexTexture: texture atIndex: msl_slot];
            if (sampler) [enc setVertexSamplerState: sampler atIndex: msl_slot];
        }
    }
}

/* ext hook (end of processGLState for draws): buffer textures follow buffer reallocation /
   contents, views follow their root, and vertex-stage textures (e.g. samplerBuffer skinning
   tables) get bound */
- (void) extTexextPreDraw: (id<MTLRenderCommandEncoder>) enc
{
    if (!enc || !ctx->state.program)
        return;

    [self texextRefreshStage: _FRAGMENT_SHADER encoder: enc];
    [self texextRefreshStage: _VERTEX_SHADER encoder: enc];
}

@end

#pragma mark - C trampolines

static void texextTextureBarrierC(GLMContext glm_ctx)
{
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj texextTextureBarrier];
}

static void texextInvalidateC(GLMContext glm_ctx, Texture *tex, GLint level)
{
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj texextInvalidate: tex level: level];
}

static GLsizei texextSampleCountForC(GLMContext glm_ctx, GLenum internalformat, GLsizei samples)
{
    return [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj texextSampleCountFor: internalformat samples: samples];
}

static void texextPrepareViewSourceC(GLMContext glm_ctx, Texture *root)
{
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj texextPrepareViewSource: root];
}
