/*
 * MGLRenderer+Gpuaudit.m — Metal side of the 'gpuaudit' domain (mgl_ext_gpuaudit.h).
 *
 *   - render pipeline state cache: MGLRenderer.m builds an MTLRenderPipelineDescriptor
 *     whenever GL state dirties the pipeline; creating the PSO is by far the most
 *     expensive thing on that path (measured 25-30 us per build).  The cache keys
 *     every field of the descriptor that can affect compilation, so a repeat of the
 *     same pipeline configuration (e.g. toggling blend modes back and forth, or the
 *     DIRTY_ALL that every flush/commit leaves behind) is a dictionary lookup.
 *   - fill_buffer: glClearBufferData / glClearBufferSubData as GPU work
 *     (MTLBlitCommandEncoder fillBuffer for 1/2/4-byte texels, a staged pattern
 *     buffer + GPU copies for 8/12/16-byte texels).  Fully asynchronous: the CPU
 *     never walks the cleared range.
 *   - read_pixels_to_buffer: glReadPixels with a GL_PIXEL_PACK_BUFFER bound is a
 *     single copyFromTexture:toBuffer: blit into the PBO's own MTLBuffer when the
 *     surface layout matches the requested format/type — no CPU pixel loop, no
 *     waitUntilCompleted, and the read stays ordered with the GL command stream.
 *   - index_buffer_new / draw_indexed_mtl: the rewritten / emulated index lists that
 *     draw_buffers.c produces (primitive restart splitting, provoking-vertex rotation,
 *     triangle fan / line loop / adjacency expansion, 8-bit widening) are uploaded
 *     once into a private MTLBuffer and cached; repeated draws encode a Metal indexed
 *     draw directly from it.
 */

#import <objc/runtime.h>

#import "MGLRenderer_Private.h"
#import "MGLRenderer+Gpuaudit.h"
#include "mgl_ext_gpuaudit.h"
#include "mgl_ext_buffers.h"
#include "mgl_ext_fb.h"
#include "mgl_ext_xfb.h"     /* gpuaudit: GPU xfb scatter (MGLXfbProgram layout) */

static inline MGLRenderer *GAR(GLMContext c) { return (__bridge MGLRenderer *)c->mtl_funcs.mtlObj; }

/* ---------------------------------------------------------------- private helpers
 * of other categories that have no public header (identical signatures) */
@interface MGLRenderer (GpuauditForeignPrivate)
- (bool) extBufEnsure:(Buffer *)buf;
/* fb category: FBTarget { id tex; NSUInteger level, slice, plane, layers; } — mirrored
 * below, with a compile-time size check so a layout change fails the build here */
- (bool) fbSourceFor:(Framebuffer *)fbo aspect:(int)aspect attachment:(GLenum)attachment out:(void *)t;
@end

typedef struct {
    __unsafe_unretained id<MTLTexture> tex;
    NSUInteger level, slice, plane, layers;
} MGLGAFBTarget;
static_assert(sizeof(MGLGAFBTarget) == 40, "MGLRenderer+Framebuffer.m FBTarget layout changed");

/* gpuaudit renderer state, kept as an associated object (nothing is stored in ivars) */
/* A cache entry retains the functions the pipeline was built from. The key hashes
 * function *addresses*, so when a function is released and a new one is allocated at
 * the same address (tess recompiles its TES variants), the key alone would return a
 * stale PSO. Comparing the retained objects makes address reuse safe: a different
 * function object is a miss even when the key matches. */
@interface MGLGAPSOEntry : NSObject
@property (nonatomic, strong) id<MTLRenderPipelineState> ps;
@property (nonatomic, strong) id<MTLFunction> vf, ff;
@end

@implementation MGLGAPSOEntry
@end

@interface MGLGPACache : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSData *, MGLGAPSOEntry *> *psos;
@property (nonatomic, strong) NSMutableArray<NSData *> *psoOrder;
@property (nonatomic, strong) id<MTLComputePipelineState> xfbScatter;
@property (nonatomic, strong) id<MTLBuffer> xfbPrims;
@end

@implementation MGLGPACache
@end

static const char kGPAKey;

#pragma mark - pipeline state cache

/* every MTLRenderPipelineDescriptor field that can change the compiled pipeline */
typedef struct __attribute__((packed)) {
    uint64_t vf, ff;                          /* vertex/fragment function objects */
    uint32_t rasterSamples;
    uint32_t inputTopology;
    uint32_t depthPF, stencilPF;
    uint32_t maxVertexAmplificationCount;
    uint8_t  alphaToCoverage, alphaToOne, rasterEnabled;
    uint32_t tess[7];                         /* factor format, winding, partition, max factor,
                                                 step function, factor scale enabled, cp index type */
    struct { uint32_t pf; uint32_t writeMask; uint8_t blending;
             uint32_t sRGB, dRGB, opRGB, sA, dA, opA; } col[8];
    struct { uint32_t format; uint32_t offset; uint32_t bufferIndex; } attr[31];
    struct { uint32_t stride; uint32_t stepFunction; uint32_t stepRate; } lay[31];
} MGLGAPSOKey;

static NSData *gaPSOKey(MTLRenderPipelineDescriptor *d)
{
    MTLVertexDescriptor *vd = d.vertexDescriptor;
    MGLGAPSOKey k;
    memset(&k, 0, sizeof(k));

    k.vf = (uint64_t)(__bridge void *)d.vertexFunction;
    k.ff = (uint64_t)(__bridge void *)d.fragmentFunction;
    k.rasterSamples = (uint32_t)d.rasterSampleCount;
    k.inputTopology = (uint32_t)d.inputPrimitiveTopology;
    k.depthPF = (uint32_t)d.depthAttachmentPixelFormat;
    k.stencilPF = (uint32_t)d.stencilAttachmentPixelFormat;
    k.maxVertexAmplificationCount = (uint32_t)d.maxVertexAmplificationCount;
    k.alphaToCoverage = d.alphaToCoverageEnabled ? 1 : 0;
    k.alphaToOne = d.alphaToOneEnabled ? 1 : 0;
    k.rasterEnabled = d.rasterizationEnabled ? 1 : 0;
    k.tess[0] = (uint32_t)d.tessellationFactorFormat;
    k.tess[1] = (uint32_t)d.tessellationOutputWindingOrder;
    k.tess[2] = (uint32_t)d.tessellationPartitionMode;
    k.tess[3] = (uint32_t)d.maxTessellationFactor;
    k.tess[4] = (uint32_t)d.tessellationFactorStepFunction;
    k.tess[5] = d.tessellationFactorScaleEnabled ? 1 : 0;
    k.tess[6] = (uint32_t)d.tessellationControlPointIndexType;

    for (int i = 0; i < 8; i++)
    {
        MTLRenderPipelineColorAttachmentDescriptor *a = d.colorAttachments[i];
        k.col[i].pf = (uint32_t)a.pixelFormat;
        if (a.pixelFormat == MTLPixelFormatInvalid)
            continue;
        k.col[i].writeMask = (uint32_t)a.writeMask;
        k.col[i].blending = a.blendingEnabled ? 1 : 0;
        k.col[i].sRGB = (uint32_t)a.sourceRGBBlendFactor;
        k.col[i].dRGB = (uint32_t)a.destinationRGBBlendFactor;
        k.col[i].opRGB = (uint32_t)a.rgbBlendOperation;
        k.col[i].sA = (uint32_t)a.sourceAlphaBlendFactor;
        k.col[i].dA = (uint32_t)a.destinationAlphaBlendFactor;
        k.col[i].opA = (uint32_t)a.alphaBlendOperation;
    }

    if (vd)
    {
        for (int i = 0; i < 31; i++)
        {
            k.attr[i].format = (uint32_t)vd.attributes[i].format;
            k.attr[i].offset = (uint32_t)vd.attributes[i].offset;
            k.attr[i].bufferIndex = (uint32_t)vd.attributes[i].bufferIndex;
            k.lay[i].stride = (uint32_t)vd.layouts[i].stride;
            k.lay[i].stepFunction = (uint32_t)vd.layouts[i].stepFunction;
            k.lay[i].stepRate = (uint32_t)vd.layouts[i].stepRate;
        }
    }

    return [NSData dataWithBytes:&k length:sizeof(k)];
}

@implementation MGLRenderer (Gpuaudit)

- (MGLGPACache *) gpuauditCache
{
    MGLGPACache *c = objc_getAssociatedObject(self, &kGPAKey);
    if (!c)
    {
        c = [[MGLGPACache alloc] init];
        c.psos = [NSMutableDictionary dictionary];
        c.psoOrder = [NSMutableArray array];
        objc_setAssociatedObject(self, &kGPAKey, c, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    return c;
}

/* called by processGLState before building a pipeline state */
- (id<MTLRenderPipelineState>) gpuauditPipelineForDescriptor:(MTLRenderPipelineDescriptor *)desc
{
    if (!desc || !ctx->state.ext_gpuaudit)
        return nil;

    MGLGAPSOEntry *e = [self gpuauditCache].psos[gaPSOKey(desc)];
    if (e && (e.vf != desc.vertexFunction || e.ff != desc.fragmentFunction))
        e = nil;       /* same key, different functions: the key address was recycled */
    if (e)
        ctx->state.ext_gpuaudit->pso_hits++;
    else
        ctx->state.ext_gpuaudit->pso_misses++;
    return e.ps;
}

- (void) gpuauditStorePipeline:(id<MTLRenderPipelineState>)ps forDescriptor:(MTLRenderPipelineDescriptor *)desc
{
    if (!ps || !desc)
        return;

    static const NSUInteger kMax = 128;
    MGLGPACache *c = [self gpuauditCache];
    NSData *key = gaPSOKey(desc);

    if (c.psos[key])
    {
        /* refresh the retained functions: a same-key entry may exist with recycled
         * addresses, and the newly built PSO is the one that matches this descriptor */
        MGLGAPSOEntry *old = c.psos[key];
        old.ps = ps;
        old.vf = desc.vertexFunction;
        old.ff = desc.fragmentFunction;
        return;
    }

    if (c.psoOrder.count >= kMax)
    {
        NSData *old = c.psoOrder[0];
        [c.psoOrder removeObjectAtIndex:0];
        [c.psos removeObjectForKey:old];
    }

    MGLGAPSOEntry *e = [[MGLGAPSOEntry alloc] init];
    e.ps = ps;
    e.vf = desc.vertexFunction;
    e.ff = desc.fragmentFunction;
    c.psos[key] = e;
    [c.psoOrder addObject:key];
}

#pragma mark - transform-feedback scatter on the GPU (F9)

static NSString * const kXFBScatterMSL = @"\n\
#include <metal_stdlib>\n\
using namespace metal;\n\
struct MGLXFBSParams {\n\
    uint32_t written;         /* output primitives including instances */\n\
    uint32_t np;              /* primitives assembled per instance */\n\
    uint32_t k;               /* vertices per primitive */\n\
    uint32_t vcount;          /* vertex count of the draw */\n\
    uint32_t record_words;    /* words in one capture record */\n\
    uint32_t record_off;      /* word offset of the first record (the capture header) */\n\
    uint32_t src_word;        /* word offset of this varying in the record */\n\
    uint32_t words;           /* words of this varying */\n\
    uint32_t out_stride_words;/* stride between output primitive-vertices */\n\
    uint32_t out_off_words;   /* byte offset of this varying inside the output stride */\n\
};\n\
kernel void mgl_xfb_scatter(const device uint32_t* record [[buffer(0)]],\n\
                            device uint32_t* dst [[buffer(1)]],\n\
                            constant MGLXFBSParams& p [[buffer(2)]],\n\
                            const device uint32_t* prims [[buffer(3)]],\n\
                            uint32_t vi [[thread_position_in_grid]])\n\
{\n\
    if (vi >= p.written * p.k) return;\n\
\n\
    uint32_t prim = vi / p.k;\n\
    uint32_t v = vi - prim * p.k;\n\
    uint32_t pp = prim % p.np;\n\
    uint32_t inst = prim / p.np;\n\
    uint32_t slot = prims[pp * p.k + v] + inst * p.vcount;\n\
\n\
    const device uint32_t* r = record + p.record_off + (uint64_t)slot * p.record_words + p.src_word;\n\
    device uint32_t* d = dst + (uint64_t)vi * p.out_stride_words + p.out_off_words;\n\
    for (uint32_t w = 0; w < p.words; w++) d[w] = r[w];\n\
}\n";

- (id<MTLComputePipelineState>) gpuauditXFBCatterPipeline
{
    MGLGPACache *c = [self gpuauditCache];
    if (c.xfbScatter)
        return c.xfbScatter;

    NSError *err = nil;
    id<MTLLibrary> lib = [_device newLibraryWithSource: kXFBScatterMSL options: nil error: &err];
    if (!lib)
    {
        NSLog(@"MGL gpuaudit: xfb scatter library failed: %@", err);
        return nil;
    }

    id<MTLFunction> fn = [lib newFunctionWithName: @"mgl_xfb_scatter"];
    if (!fn)
        return nil;

    id<MTLComputePipelineState> ps = [_device newComputePipelineStateWithFunction: fn error: &err];
    if (!ps)
    {
        NSLog(@"MGL gpuaudit: xfb scatter pipeline failed: %@", err);
        return nil;
    }

    c.xfbScatter = ps;
    return ps;
}

- (bool) gpuauditXFBScatter:(void *)capture_cb record:(void *)record_scratch xp:(void *)xp_v
                       bufs:(Buffer **)bufs boff:(const GLintptr *)boff writeOff:(const GLsizeiptr *)write_offset
                      prims:(const uint32_t *)prims np:(uint64_t)np k:(unsigned)k
                    vcount:(uint64_t)vcount written:(uint64_t)written
{
    MGLXfbProgram *xp = (MGLXfbProgram *)xp_v;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)capture_cb;
    id<MTLBuffer> scratch = (__bridge id<MTLBuffer>)record_scratch;

    if (!cb || !scratch || !xp || !prims || !written || !k)
        return false;

    id<MTLComputePipelineState> ps = [self gpuauditXFBCatterPipeline];
    if (!ps)
        return false;

    /* the prims table travels with the kernel; setBytes is limited to 4 KiB */
    size_t prims_bytes = (size_t)np * k * sizeof(uint32_t);
    id<MTLBuffer> prims_buf = nil;
    NSUInteger prims_off = 0;

    if (prims_bytes <= 4096)
    {
        /* setBytes below */
    }
    else
    {
        MGLGPACache *c = [self gpuauditCache];
        if (!c.xfbPrims || c.xfbPrims.length < prims_bytes)
        {
            size_t cap = MAX(prims_bytes, (size_t)c.xfbPrims.length * 2);
            id<MTLBuffer> fresh = [_device newBufferWithLength:cap options:MTLResourceStorageModeShared];
            if (!fresh)
                return false;
            c.xfbPrims = fresh;
        }

        memcpy(c.xfbPrims.contents, prims, prims_bytes);
        prims_buf = c.xfbPrims;
    }

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    enc.label = @"MGL xfb scatter";
    [enc setComputePipelineState: ps];
    [enc setBuffer: scratch offset: 0 atIndex: 0];

    NSUInteger tg = 64;
    uint64_t threads = written * (uint64_t)k;

    for (unsigned b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
    {
        if (!(xp->buffer_mask & (1u << b)) || !xp->stride[b])
            continue;

        id<MTLBuffer> dst = (__bridge id<MTLBuffer>)bufs[b]->data.mtl_data;
        if (!dst)
        {
            [enc endEncoding];
            return false;        /* the CPU path syncs and memcpys instead */
        }

        uint32_t stride_words = (uint32_t)(xp->stride[b] / 4);

        for (unsigned i = 0; i < (unsigned)xp->count; i++)
        {
            const MGLXfbVarying *var = &xp->vars[i];
            if (!var->words || var->buffer != b)
                continue;

            struct { uint32_t written, np, k, vcount, record_words, record_off, src_word, words, out_stride_words, out_off_words; } p;
            p.written = (uint32_t)written;
            p.np = (uint32_t)np;
            p.k = k;
            p.vcount = (uint32_t)vcount;
            p.record_words = (uint32_t)xp->record_words;
            p.record_off = (uint32_t)MGL_XFB_HEADER_WORDS;
            p.src_word = (uint32_t)var->src_word;
            p.words = (uint32_t)var->words;
            p.out_stride_words = stride_words;
            p.out_off_words = (uint32_t)(var->offset / 4);

            NSUInteger dst_off = (NSUInteger)(boff[b] + write_offset[b]);

            [enc setBuffer: dst offset: dst_off atIndex: 1];
            [enc setBytes: &p length: sizeof(p) atIndex: 2];
            if (prims_buf)
                [enc setBuffer: prims_buf offset: 0 atIndex: 3];
            else
                [enc setBytes: prims length: prims_bytes atIndex: 3];


            [enc dispatchThreads: MTLSizeMake((NSUInteger)threads, 1, 1)
            threadsPerThreadgroup: MTLSizeMake(tg, 1, 1)];
        }
    }

    [enc endEncoding];

    /* the XBO ranges are now GPU-written *by the capture command buffer*: make the
     * serial tracking follow it, so a later CPU read of an XBO waits for exactly this
     * buffer (see mgl_buffers_prepare_cpu_read / make_cpu_visible) */
    /* the capture command buffer becomes the renderer's current one, exactly like a
     * normally committed draw buffer: the serial tracking stays consistent (a later CPU
     * read then waits on precisely this command buffer), and the next encoder allocates
     * a fresh buffer because this one is committed */
    _currentCommandBuffer = cb;

    for (unsigned b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
    {
        if (!(xp->buffer_mask & (1u << b)) || !xp->stride[b] || !bufs[b])
            continue;

        if (written * k * xp->stride[b])
            mgl_buffers_mark_gpu_use(ctx, bufs[b], true);
    }

    [cb commit];

    return true;
}

#pragma mark - GPU buffer clears (glClearBuffer{Sub}Data)

- (bool) gpuauditFillBuffer:(Buffer *)buf offset:(GLintptr)offset size:(GLsizeiptr)size
                      texel:(const uint8_t *)texel texelSize:(int)texelSize
{
    if (!buf || !buf->data.buffer_data || size <= 0 || texelSize <= 0 || texelSize > 16)
        return false;
    if ((size_t)offset + (size_t)size > (size_t)buf->size)
        return false;
    if (![self extBufEnsure:buf])
        return false;

    id<MTLBuffer> mb = (__bridge id<MTLBuffer>)buf->data.mtl_data;
    if ((size_t)offset + (size_t)size > mb.length)
        return false;

    [self endRenderEncoding];
    if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
        [self newCommandBuffer];
    if (!_currentCommandBuffer)
        return false;

    id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
    blit.label = @"MGL buffer clear";

    bool byteFill = false;
    if (@available(macOS 10.15, *))
        byteFill = true;        /* fillBuffer repeats a single byte over a range */

    if (texelSize == 1 && byteFill)
    {
        /* 1-byte texel: fillBuffer repeats a single byte over the whole range */
        [blit fillBuffer:mb range:NSMakeRange((NSUInteger)offset, (NSUInteger)size) value:texel[0]];
    }
    else
    {
        /* 2..16-byte texels: Metal's fillBuffer repeats only a single byte, so seed a
         * 64 KiB pattern block and then double it up inside the destination buffer
         * (16 MiB = 8 doubling copies; a straight staged copy needs 256). */
        const size_t seed = 65536 / (size_t)texelSize * (size_t)texelSize;
        size_t span = (size_t)size - ((size_t)size % (size_t)texelSize);
        if (span < seed)
        {
            /* small clear: one staging block is enough (and cheaper than a seed+copy) */
            id<MTLBuffer> tmp = [_device newBufferWithLength:MAX(seed, span) options:MTLResourceStorageModeShared];
            if (!tmp)
            {
                [blit endEncoding];
                return false;
            }
            uint8_t *p = (uint8_t *)tmp.contents;
            for (size_t i = 0; i + (size_t)texelSize <= span; i += (size_t)texelSize)
                memcpy(p + i, texel, (size_t)texelSize);
            [blit copyFromBuffer:tmp sourceOffset:0 toBuffer:mb destinationOffset:(NSUInteger)offset size:span];
        }
        else
        {
            id<MTLBuffer> tmp = [_device newBufferWithLength:seed options:MTLResourceStorageModeShared];
            if (!tmp)
            {
                [blit endEncoding];
                return false;
            }
            uint8_t *p = (uint8_t *)tmp.contents;
            for (size_t i = 0; i + (size_t)texelSize <= seed; i += (size_t)texelSize)
                memcpy(p + i, texel, (size_t)texelSize);

            /* seed the first block, then double: [done, done+n) <- [done-n, done) */
            [blit copyFromBuffer:tmp sourceOffset:0 toBuffer:mb destinationOffset:(NSUInteger)offset size:seed];

            size_t done = seed;
            while (done < span)
            {
                size_t n = MIN(done, span - done);
                n -= n % (size_t)texelSize;
                [blit copyFromBuffer:mb sourceOffset:(NSUInteger)(offset + done - n)
                            toBuffer:mb destinationOffset:(NSUInteger)(offset + done) size:n];
                done += n;
            }
        }
    }

    [blit endEncoding];

    /* the fill is encoded on the current command buffer: stamp its serial *after*
     * encoding so a later CPU read waits for exactly this buffer */
    mgl_buffers_mark_gpu_use(ctx, buf, true);

    if (ctx->state.ext_gpuaudit)
        ctx->state.ext_gpuaudit->gpu_clears++;

    return true;
}

#pragma mark - glReadPixels into a PIXEL_PACK_BUFFER

- (bool) gpuauditReadPixelsToBuffer:(Framebuffer *)fbo aspect:(int)aspect attachment:(GLenum)attachment
                                  x:(GLint)x y:(GLint)y width:(GLsizei)width height:(GLsizei)height
                              format:(GLenum)format type:(GLenum)type pbo:(Buffer *)pbo
                        pboOffset:(size_t)pboOffset rowBytes:(size_t)rowBytes skipBytes:(size_t)skipBytes
{
    if (!pbo || !pbo->data.buffer_data || width <= 0 || height <= 0)
        return false;
    if (![self extBufEnsure:pbo])
        return false;

    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)pbo->data.mtl_data;

    /* only exact-layout copies: GL's per-pixel conversion has no Metal blit equivalent */
    MTLBlitOption opt = MTLBlitOptionNone;

    if (aspect == MGL_FB_ASPECT_COLOR)
    {
        if (type != GL_UNSIGNED_BYTE)
            return false;

        /* the source surface must be resolved first: its pixel format decides */
        MGLGAFBTarget src;
        if (![self fbSourceFor:fbo aspect:aspect attachment:attachment out:&src] || !src.tex)
            return false;
        if (src.tex.sampleCount > 1)
            return false;

        MTLPixelFormat pf = src.tex.pixelFormat;
        bool srcRGBA = (pf == MTLPixelFormatRGBA8Unorm);
        bool srcBGRA = (pf == MTLPixelFormatBGRA8Unorm);
        bool srcRGBAi = (pf == MTLPixelFormatRGBA8Uint);
        bool srcRGBAii = (pf == MTLPixelFormatRGBA8Sint);

        if (!((srcRGBA && format == GL_RGBA && !ctx->state.pack.swap_bytes) ||
              (srcBGRA && format == GL_BGRA && !ctx->state.pack.swap_bytes) ||
              (srcRGBAi && format == GL_RGBA_INTEGER && type == GL_UNSIGNED_BYTE) ||
              (srcRGBAii && format == GL_RGBA_INTEGER && type == GL_UNSIGNED_BYTE)))
            return false;

        if (ctx->state.pack.row_length > 0 && ctx->state.pack.row_length < (GLuint)width)
            return false;

        NSUInteger W = MAX((NSUInteger)1, src.tex.width >> src.level);
        NSUInteger H = MAX((NSUInteger)1, src.tex.height >> src.level);
        if (x < 0 || y < 0 || (NSUInteger)(x + width) > W || (NSUInteger)(y + height) > H)
            return false;                       /* partial rect: the CPU path fills the rest */

        size_t bytes = (size_t)rowBytes * (size_t)height + skipBytes + pboOffset;
        if (bytes > dst.length)
            return false;

        [self endRenderEncoding];
        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
            [self newCommandBuffer];
        if (!_currentCommandBuffer)
            return false;

        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        blit.label = @"MGL readpixels (PBO)";
        [blit copyFromTexture:src.tex sourceSlice:src.slice sourceLevel:src.level
                 sourceOrigin:MTLOriginMake(x, y, src.plane) sourceSize:MTLSizeMake(width, height, 1)
                     toBuffer:dst destinationOffset:pboOffset + skipBytes
              destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * height options:opt];
        [blit endEncoding];

        mgl_buffers_mark_gpu_use(ctx, pbo, true);
        if (ctx->state.ext_gpuaudit)
            ctx->state.ext_gpuaudit->pbo_readbacks++;
        return true;
    }

    if (aspect == MGL_FB_ASPECT_DEPTH && format == GL_DEPTH_COMPONENT && type == GL_FLOAT &&
        !ctx->state.pack.swap_bytes)
    {
        MGLGAFBTarget src;
        if (![self fbSourceFor:fbo aspect:aspect attachment:attachment out:&src] || !src.tex)
            return false;
        if (src.tex.sampleCount > 1 || src.tex.pixelFormat != MTLPixelFormatDepth32Float || src.plane)
            return false;
        if (ctx->state.pack.row_length > 0 && ctx->state.pack.row_length < (GLuint)width)
            return false;

        NSUInteger W = MAX((NSUInteger)1, src.tex.width >> src.level);
        NSUInteger H = MAX((NSUInteger)1, src.tex.height >> src.level);
        if (x < 0 || y < 0 || (NSUInteger)(x + width) > W || (NSUInteger)(y + height) > H)
            return false;

        size_t bytes = (size_t)rowBytes * (size_t)height + skipBytes + pboOffset;
        if (bytes > dst.length)
            return false;

        [self endRenderEncoding];
        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
            [self newCommandBuffer];
        if (!_currentCommandBuffer)
            return false;

        id<MTLBlitCommandEncoder> blit = [_currentCommandBuffer blitCommandEncoder];
        blit.label = @"MGL readpixels depth (PBO)";
        [blit copyFromTexture:src.tex sourceSlice:src.slice sourceLevel:src.level
                 sourceOrigin:MTLOriginMake(x, y, 0) sourceSize:MTLSizeMake(width, height, 1)
                     toBuffer:dst destinationOffset:pboOffset + skipBytes
              destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * height options:MTLBlitOptionNone];
        [blit endEncoding];

        mgl_buffers_mark_gpu_use(ctx, pbo, true);
        if (ctx->state.ext_gpuaudit)
            ctx->state.ext_gpuaudit->pbo_readbacks++;
        return true;
    }

    return false;
}

#pragma mark - cached rewritten index lists

- (void *) gpuauditIndexBufferNew:(const void *)data bytes:(size_t)bytes
{
    if (!data || !bytes)
        return NULL;
    id<MTLBuffer> b = [_device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    return b ? (void *)CFBridgingRetain(b) : NULL;
}

- (void) gpuauditIndexBufferFree:(void *)mtl_buffer
{
    if (mtl_buffer)
        CFBridgingRelease(mtl_buffer);
}

- (void) gpuauditDrawIndexed:(int)prim type:(GLenum)type buffer:(void *)mtl_buffer offset:(size_t)offset
                      count:(GLsizei)count instances:(GLsizei)instances baseVertex:(GLint)base_vertex
                baseInstance:(GLuint)base_instance
{
    if (!_currentRenderEncoder || !mtl_buffer)
        return;

    MTLIndexType itype = (type == GL_UNSIGNED_INT) ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;

    [_currentRenderEncoder drawIndexedPrimitives:(MTLPrimitiveType)prim indexCount:count indexType:itype
                                      indexBuffer:(__bridge id<MTLBuffer>)mtl_buffer indexBufferOffset:offset
                                    instanceCount:instances baseVertex:base_vertex baseInstance:base_instance];
}

#pragma mark - hook table

static void *ga_pso_lookup(GLMContext ctx, void *desc)
{
    @autoreleasepool { return (__bridge void *)[GAR(ctx) gpuauditPipelineForDescriptor:(__bridge MTLRenderPipelineDescriptor *)desc]; }
}

static void ga_pso_store(GLMContext ctx, void *desc, void *ps)
{
    @autoreleasepool { [GAR(ctx) gpuauditStorePipeline:(__bridge id<MTLRenderPipelineState>)ps forDescriptor:(__bridge MTLRenderPipelineDescriptor *)desc]; }
}

static bool ga_fill(GLMContext ctx, Buffer *buf, GLintptr offset, GLsizeiptr size, const uint8_t *texel, int texelSize)
{
    @autoreleasepool { return [GAR(ctx) gpuauditFillBuffer:buf offset:offset size:size texel:texel texelSize:texelSize]; }
}

static bool ga_read_pixels(GLMContext ctx, Framebuffer *fbo, int aspect, GLenum attachment,
                           GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type,
                           Buffer *pbo, size_t pboOffset, size_t rowBytes, size_t skipBytes)
{
    @autoreleasepool
    {
        return [GAR(ctx) gpuauditReadPixelsToBuffer:fbo aspect:aspect attachment:attachment x:x y:y
                                             width:width height:height format:format type:type pbo:pbo
                                        pboOffset:pboOffset rowBytes:rowBytes skipBytes:skipBytes];
    }
}

static bool ga_xfb_scatter(GLMContext ctx, void *capture_cb, void *record_scratch, void *xp,
                           Buffer **bufs, const GLintptr *boff, const GLsizeiptr *write_offset,
                           const uint32_t *prims, uint64_t np, unsigned k, uint64_t vcount,
                           uint64_t written)
{
    @autoreleasepool
    {
        bool ok = [GAR(ctx) gpuauditXFBScatter:capture_cb record:record_scratch xp:xp
                                          bufs:bufs boff:boff writeOff:write_offset
                                         prims:prims np:np k:k vcount:vcount written:written];
        if (capture_cb) CFBridgingRelease(capture_cb);
        if (record_scratch) CFBridgingRelease(record_scratch);
        return ok;
    }
}

static void *ga_index_new(GLMContext ctx, const void *data, size_t bytes)
{
    @autoreleasepool { return [GAR(ctx) gpuauditIndexBufferNew:data bytes:bytes]; }
}

static void ga_index_free(GLMContext ctx, void *mtl_buffer)
{
    (void)ctx;
    @autoreleasepool { [GAR(ctx) gpuauditIndexBufferFree:mtl_buffer]; }
}

static void ga_draw_indexed(GLMContext ctx, int prim, GLenum type, void *mtl_buffer, size_t offset,
                            GLsizei count, GLsizei instances, GLint base_vertex, GLuint base_instance)
{
    @autoreleasepool
    {
        [GAR(ctx) gpuauditDrawIndexed:prim type:type buffer:mtl_buffer offset:offset
                                 count:count instances:instances baseVertex:base_vertex
                          baseInstance:base_instance];
    }
}

- (void) bindExtGpuauditFuncsToGLMContext:(GLMContext)glm_ctx
{
    struct MGLExtMetal_gpuaudit *m = glm_ctx->mtl_funcs.ext_gpuaudit;
    if (!m)
    {
        m = (struct MGLExtMetal_gpuaudit *)calloc(1, sizeof(struct MGLExtMetal_gpuaudit));
        glm_ctx->mtl_funcs.ext_gpuaudit = m;
    }
    m->pso_lookup = ga_pso_lookup;
    m->pso_store = ga_pso_store;
    m->fill_buffer = ga_fill;
    m->read_pixels_to_buffer = ga_read_pixels;
    m->index_buffer_new = ga_index_new;
    m->xfb_scatter = ga_xfb_scatter;
    m->index_buffer_free = ga_index_free;
    m->draw_indexed_mtl = ga_draw_indexed;
}

@end
