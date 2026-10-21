/*
 * MGLRenderer+RenderState.m — Metal side of the 'rstate' domain.
 *
 * Applies GL fixed-function state to Metal:
 *   encoder state  (-rstateApplyEncoderState:)  viewports, scissors, depth/stencil state, cull,
 *                   winding, depth bias, fill mode, depth clip mode, blend color, stencil refs
 *   pipeline state (-rstateConfigurePipelineDescriptor:)  blend per attachment, write masks,
 *                   alpha-to-coverage/one, rasterizer discard, default-FB attachment formats and
 *                   the function-constant specialization of the rstate shader fixups.
 * Orientation convention: Metal texture row r == GL window row r (see mgl_ext_rstate.h).
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import <objc/runtime.h>
#import "MGLRenderer+RenderState.h"
#import "MGLRenderer_Private.h"
#import "mgl_ext_tess.h"
#import "mgl_ext_geom.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

/* per-renderer rstate bookkeeping kept outside the shared ivar block */
@interface MGLRSCache : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSString *, id<MTLDepthStencilState>> *dsStates;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSArray *> *specialized;
@property (nonatomic, weak)   id<MTLRenderCommandEncoder> appliedEncoder;
@property (nonatomic) GLuint appliedGeneration;
@property (nonatomic) GLenum drawMode;
@end

@implementation MGLRSCache
@end

static GLenum s_rs_draw_mode = GL_TRIANGLES;    /* recorded by the pre-draw hook */

@implementation MGLRenderer (RenderState)

#pragma mark - bookkeeping

- (MGLRSCache *) rstateCache
{
    static const char kKey = 0;
    MGLRSCache *c = objc_getAssociatedObject(self, &kKey);
    if (!c) {
        c = [[MGLRSCache alloc] init];
        c.dsStates = [NSMutableDictionary dictionary];
        c.specialized = [NSMutableDictionary dictionary];
        c.appliedGeneration = ~0u;
        objc_setAssociatedObject(self, &kKey, c, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    return c;
}

#pragma mark - render target geometry

- (MTLSize) rstateRenderTargetSize
{
    MTLRenderPassDescriptor *d = _renderPassDescriptor;
    id<MTLTexture> t = nil;

    if (!d) return MTLSizeMake(0, 0, 0);
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS && !t; i++) t = d.colorAttachments[i].texture;
    if (!t) t = d.depthAttachment.texture;
    if (!t) t = d.stencilAttachment.texture;
    if (t) {
        NSUInteger lvl = 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
            if (d.colorAttachments[i].texture == t) { lvl = d.colorAttachments[i].level; break; }
        return MTLSizeMake(MAX(t.width >> lvl, 1), MAX(t.height >> lvl, 1), 1);
    }
    if (d.renderTargetWidth && d.renderTargetHeight)
        return MTLSizeMake(d.renderTargetWidth, d.renderTargetHeight, 1);
    return MTLSizeMake(0, 0, 0);
}

- (NSUInteger) rstateSampleCount
{
    MTLRenderPassDescriptor *d = _renderPassDescriptor;
    for (int i = 0; d && i < MAX_COLOR_ATTACHMENTS; i++)
        if (d.colorAttachments[i].texture) return MAX(d.colorAttachments[i].texture.sampleCount, 1);
    if (d.depthAttachment.texture) return MAX(d.depthAttachment.texture.sampleCount, 1);
    return 1;
}

#pragma mark - GL -> Metal enum mapping

static MTLCompareFunction rs_compare(GLenum f)
{
    switch (f) {
        case GL_NEVER: return MTLCompareFunctionNever;
        case GL_LESS: return MTLCompareFunctionLess;
        case GL_EQUAL: return MTLCompareFunctionEqual;
        case GL_LEQUAL: return MTLCompareFunctionLessEqual;
        case GL_GREATER: return MTLCompareFunctionGreater;
        case GL_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case GL_GEQUAL: return MTLCompareFunctionGreaterEqual;
        default: return MTLCompareFunctionAlways;
    }
}

static MTLStencilOperation rs_stencil_op(GLenum op)
{
    switch (op) {
        case GL_ZERO: return MTLStencilOperationZero;
        case GL_REPLACE: return MTLStencilOperationReplace;
        case GL_INCR: return MTLStencilOperationIncrementClamp;
        case GL_DECR: return MTLStencilOperationDecrementClamp;
        case GL_INVERT: return MTLStencilOperationInvert;
        case GL_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case GL_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        default: return MTLStencilOperationKeep;
    }
}

static MTLBlendFactor rs_blend_factor(GLenum f)
{
    switch (f) {
        case GL_ZERO: return MTLBlendFactorZero;
        case GL_ONE: return MTLBlendFactorOne;
        case GL_SRC_COLOR: return MTLBlendFactorSourceColor;
        case GL_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case GL_DST_COLOR: return MTLBlendFactorDestinationColor;
        case GL_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case GL_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case GL_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case GL_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case GL_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case GL_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
        case GL_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
        case GL_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
        case GL_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
        case GL_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
        case GL_SRC1_COLOR: return MTLBlendFactorSource1Color;
        case GL_ONE_MINUS_SRC1_COLOR: return MTLBlendFactorOneMinusSource1Color;
        case GL_SRC1_ALPHA: return MTLBlendFactorSource1Alpha;
        case GL_ONE_MINUS_SRC1_ALPHA: return MTLBlendFactorOneMinusSource1Alpha;
        default: return MTLBlendFactorOne;
    }
}

static MTLBlendOperation rs_blend_op(GLenum e)
{
    switch (e) {
        case GL_FUNC_SUBTRACT: return MTLBlendOperationSubtract;
        case GL_FUNC_REVERSE_SUBTRACT: return MTLBlendOperationReverseSubtract;
        case GL_MIN: return MTLBlendOperationMin;
        case GL_MAX: return MTLBlendOperationMax;
        default: return MTLBlendOperationAdd;
    }
}

/* integer render targets cannot blend in Metal (GL ignores blending for them as well) */
static bool rs_format_is_integer(MTLPixelFormat f)
{
    switch (f) {
        case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint: case MTLPixelFormatR16Uint:
        case MTLPixelFormatR16Sint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
        case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRG16Sint: case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGB10A2Uint: case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Uint:
        case MTLPixelFormatRGBA32Sint:
            return true;
        default:
            return false;
    }
}

#pragma mark - depth / stencil state

- (id<MTLDepthStencilState>) rstateDepthStencilStateHasDepth:(bool)hasDepth hasStencil:(bool)hasStencil
{
    GLMParams *v = &ctx->state.var;
    bool dt = ctx->state.caps.depth_test && hasDepth;
    bool st = ctx->state.caps.stencil_test && hasStencil;
    NSString *key = [NSString stringWithFormat:@"%d:%x:%d|%d:%x:%x:%x:%x:%x:%x|%x:%x:%x:%x:%x:%x",
                     dt, v->depth_func, v->depth_writemask, st,
                     v->stencil_func, v->stencil_fail, v->stencil_pass_depth_fail, v->stencil_pass_depth_pass,
                     v->stencil_value_mask & 0xff, v->stencil_writemask & 0xff,
                     v->stencil_back_func, v->stencil_back_fail, v->stencil_back_pass_depth_fail,
                     v->stencil_back_pass_depth_pass, v->stencil_back_value_mask & 0xff, v->stencil_back_writemask & 0xff];
    MGLRSCache *c = [self rstateCache];
    id<MTLDepthStencilState> s = c.dsStates[key];
    if (s) return s;

    MTLDepthStencilDescriptor *d = [[MTLDepthStencilDescriptor alloc] init];
    /* GL: with the depth test disabled the depth buffer is neither tested nor written */
    d.depthCompareFunction = dt ? rs_compare(v->depth_func) : MTLCompareFunctionAlways;
    d.depthWriteEnabled = dt && v->depth_writemask;
    if (st) {
        MTLStencilDescriptor *f = [[MTLStencilDescriptor alloc] init];
        f.stencilCompareFunction = rs_compare(v->stencil_func);
        f.stencilFailureOperation = rs_stencil_op(v->stencil_fail);
        f.depthFailureOperation = rs_stencil_op(v->stencil_pass_depth_fail);
        f.depthStencilPassOperation = rs_stencil_op(v->stencil_pass_depth_pass);
        f.readMask = v->stencil_value_mask;
        f.writeMask = v->stencil_writemask;
        MTLStencilDescriptor *b = [[MTLStencilDescriptor alloc] init];
        b.stencilCompareFunction = rs_compare(v->stencil_back_func);
        b.stencilFailureOperation = rs_stencil_op(v->stencil_back_fail);
        b.depthFailureOperation = rs_stencil_op(v->stencil_back_pass_depth_fail);
        b.depthStencilPassOperation = rs_stencil_op(v->stencil_back_pass_depth_pass);
        b.readMask = v->stencil_back_value_mask;
        b.writeMask = v->stencil_back_writemask;
        d.frontFaceStencil = f;
        d.backFaceStencil = b;
    }
    s = [_device newDepthStencilStateWithDescriptor:d];
    if (s) {
        if (c.dsStates.count > 512) [c.dsStates removeAllObjects];
        c.dsStates[key] = s;
    }
    return s;
}

#pragma mark - encoder state

/* GL primitive class of the current draw (for polygon offset / polygon mode) */
static bool rs_mode_is_polygon(GLenum m)
{
    return m == GL_TRIANGLES || m == GL_TRIANGLE_STRIP || m == GL_TRIANGLE_FAN ||
           m == GL_TRIANGLES_ADJACENCY || m == GL_TRIANGLE_STRIP_ADJACENCY || m == GL_PATCHES;
}

- (void) rstateApplyEncoderState:(id<MTLRenderCommandEncoder>) enc
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    GLMParams *v = &ctx->state.var;

    if (!enc || !rs) return;

    MTLSize rt = [self rstateRenderTargetSize];
    NSInteger rtw = rt.width, rth = rt.height;

    /* GL: the initial viewport and scissor box are the drawable size */
    if (!rs->viewport_set_by_app && !ctx->state.framebuffer && rtw > 0 && rth > 0) {
        for (int i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++) {
            rs->viewport[i][0] = 0; rs->viewport[i][1] = 0;
            rs->viewport[i][2] = rtw; rs->viewport[i][3] = rth;
            rs->scissor[i][0] = 0; rs->scissor[i][1] = 0;
            rs->scissor[i][2] = (GLint)rtw; rs->scissor[i][3] = (GLint)rth;
        }
        for (int k = 0; k < 4; k++) {
            ctx->state.viewport[k] = (GLint)rs->viewport[0][k];
            ctx->state.var.scissor_box[k] = rs->scissor[0][k];
        }
        rs->viewport_set_by_app = GL_TRUE;
    }

    /* depth / stencil */
    bool hasDepth = _renderPassDescriptor.depthAttachment.texture != nil;
    bool hasStencil = _renderPassDescriptor.stencilAttachment.texture != nil;
    id<MTLDepthStencilState> ds = [self rstateDepthStencilStateHasDepth:hasDepth hasStencil:hasStencil];
    if (ds) [enc setDepthStencilState:ds];
    [enc setStencilFrontReferenceValue:(uint32_t)v->stencil_ref & 0xff
                    backReferenceValue:(uint32_t)v->stencil_back_ref & 0xff];

    /* viewports: memory row == GL window row, so GL y is Metal originY */
    NSUInteger nvp = MIN((NSUInteger)MGL_RSTATE_MAX_VIEWPORTS, (NSUInteger)MAX(ctx->state.var.max_viewports, 1u));
    MTLViewport vps[MGL_RSTATE_MAX_VIEWPORTS];
    for (NSUInteger i = 0; i < nvp; i++) {
        vps[i].originX = rs->viewport[i][0];
        vps[i].originY = rs->viewport[i][1];
        vps[i].width   = rs->viewport[i][2];
        vps[i].height  = rs->viewport[i][3];
        vps[i].znear   = rs->depth_range[i][0];
        vps[i].zfar    = rs->depth_range[i][1];
    }
    if (nvp > 1) [enc setViewports:vps count:nvp];
    else         [enc setViewport:vps[0]];

    /* scissors: Metal always scissors and requires rects inside the render target */
    if (rtw > 0 && rth > 0) {
        MTLScissorRect sr[MGL_RSTATE_MAX_VIEWPORTS];
        for (NSUInteger i = 0; i < nvp; i++) {
            NSInteger x0 = 0, y0 = 0, x1 = rtw, y1 = rth;
            if (rs->scissor_test[i]) {
                x0 = rs->scissor[i][0]; y0 = rs->scissor[i][1];
                x1 = x0 + rs->scissor[i][2]; y1 = y0 + rs->scissor[i][3];
                x0 = MAX(0, MIN(x0, rtw)); x1 = MAX(x0, MIN(x1, rtw));
                y0 = MAX(0, MIN(y0, rth)); y1 = MAX(y0, MIN(y1, rth));
            }
            sr[i] = (MTLScissorRect){ (NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0) };
        }
        if (nvp > 1) [enc setScissorRects:sr count:nvp];
        else         [enc setScissorRect:sr[0]];
    }

    /* culling.  FRONT_AND_BACK (no Metal mode) is handled by skipping polygon draws in the
     * pre-draw hook. */
    MTLCullMode cull = MTLCullModeNone;
    if (ctx->state.caps.cull_face) {
        if (v->cull_face_mode == GL_FRONT) cull = MTLCullModeFront;
        else if (v->cull_face_mode == GL_BACK) cull = MTLCullModeBack;
    }
    [enc setCullMode:cull];

    /* winding: the LOWER_LEFT y-flip mirrors window space, which reverses orientation */
    bool flipped = rs->clip_origin == GL_LOWER_LEFT;
    bool ccw = v->front_face == GL_CCW;
    if (flipped) ccw = !ccw;
    [enc setFrontFacingWinding: ccw ? MTLWindingCounterClockwise : MTLWindingClockwise];

    /* polygon mode: LINE natively; POINT is emulated by the draw path (index rewrite to
     * GL_POINTS via mgl_rstate_index_rewrite_needed), FILL otherwise */
    [enc setTriangleFillMode: rs->polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];

    /* polygon offset: only polygons, enable selected by the polygon mode.  Metal's depth bias
     * uses the same units as GL (units * minimum resolvable depth difference r of the depth
     * format, factor * max depth slope, clamp 0 = unclamped), so values pass through. */
    bool offset = false;
    if (rs_mode_is_polygon(s_rs_draw_mode)) {
        switch (rs->polygon_mode) {
            case GL_LINE:  offset = ctx->state.caps.polygon_offset_line; break;
            case GL_POINT: offset = ctx->state.caps.polygon_offset_point; break;
            default:       offset = ctx->state.caps.polygon_offset_fill; break;
        }
    }
    if (offset)
        [enc setDepthBias:rs->polygon_offset_units slopeScale:rs->polygon_offset_factor clamp:rs->polygon_offset_clamp];
    else
        [enc setDepthBias:0 slopeScale:0 clamp:0];

    [enc setDepthClipMode: ctx->state.caps.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];

    [enc setBlendColorRed:v->blend_color[0] green:v->blend_color[1] blue:v->blend_color[2] alpha:v->blend_color[3]];

    MGLRSCache *c = [self rstateCache];
    c.appliedEncoder = enc;
    c.appliedGeneration = rs->generation;
    c.drawMode = s_rs_draw_mode;
}

#pragma mark - pipeline state

- (id<MTLFunction>) rstateSpecialize:(Shader *)shader stage:(int)stage
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    Program *program = ctx->state.program;
    const char *stage_msl;

    if (!shader || !shader->mtl_data.library || !shader->mtl_data.function) return nil;
    stage_msl = program ? program->spirv[stage].msl_str : NULL;

    /* which function constants the stage takes is decided by the fixup its MSL carries:
     * the vertex fixup (clip-space flip) marks any *last pre-rasterization* stage —
     * the vertex shader, or the tessellation evaluation / geometry shader replacing it */
    bool is_vertex_like = (stage == _VERTEX_SHADER) ||
                          (stage != _FRAGMENT_SHADER && stage_msl && strstr(stage_msl, "mgl_rs_vertex_fixup"));
    const char *msl = program ? program->spirv[stage].msl_str : NULL;
    if (!mgl_rstate_msl_is_patched(msl)) return (__bridge id<MTLFunction>)shader->mtl_data.function;

    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)shader->mtl_data.library;
    id<MTLFunction> base = (__bridge id<MTLFunction>)shader->mtl_data.function;
    MTLFunctionConstantValues *cv = [[MTLFunctionConstantValues alloc] init];
    NSString *key;

    if (is_vertex_like) {
        bool z01 = rs->clip_depth_mode == GL_ZERO_TO_ONE;
        bool ul  = rs->clip_origin == GL_UPPER_LEFT;
        bool pps = ctx->state.caps.program_point_size;
        float psz = ctx->state.var.point_size;
        if (psz < 1.0f) psz = 1.0f;          /* Metal point size range [1, 511] */
        if (psz > 511.0f) psz = 511.0f;
        [cv setConstantValue:&z01 type:MTLDataTypeBool atIndex:MGL_RS_FC_DEPTH_ZERO_TO_ONE];
        [cv setConstantValue:&ul type:MTLDataTypeBool atIndex:MGL_RS_FC_ORIGIN_UPPER_LEFT];
        [cv setConstantValue:&pps type:MTLDataTypeBool atIndex:MGL_RS_FC_PROGRAM_POINT_SIZE];
        [cv setConstantValue:&psz type:MTLDataTypeFloat atIndex:MGL_RS_FC_POINT_SIZE];
        bool rd = ctx->state.caps.rasterizer_discard;
        [cv setConstantValue:&rd type:MTLDataTypeBool atIndex:MGL_RS_FC_RASTERIZER_DISCARD];
        /* the key must not collide when a program is deleted and a later one's Metal
         * objects land on the same addresses: the specialized entry would hand the dead
         * program's function to the new one and it would render with the wrong shader
         * (seen as flaky CTS failures once per-program recompiles made reuse common).
         * The function pointer plus a checksum of the MSL make that practically
         * impossible. */
        uint64_t h = 1469598103934665603ULL;
        if (msl) { for (const char *q = msl; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ULL; } }
        key = [NSString stringWithFormat:@"%p:%p:%llx:v:%d%d%d%d:%g", lib, base, (unsigned long long)h, z01, ul, pps, rd, psz];
    } else {
        NSUInteger samples = [self rstateSampleCount];
        bool lower = rs->point_sprite_coord_origin == GL_LOWER_LEFT;
        uint32_t mask = 0xFFFFFFFFu;
        bool per_sample = false;
        if (samples > 1 && ctx->state.caps.multisample) {
            uint32_t all = samples >= 32 ? 0xFFFFFFFFu : ((1u << samples) - 1u);
            if (ctx->state.caps.sample_mask) mask &= rs->sample_mask[0] | ~all;
            if (ctx->state.caps.sample_coverage) {
                uint32_t n = (uint32_t)floorf(rs->sample_coverage_value * (float)samples + 0.5f);
                uint32_t cov = n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u);
                if (rs->sample_coverage_invert) cov = ~cov;
                mask &= (cov & all) | ~all;
            }
            per_sample = ctx->state.caps.sample_shading && rs->min_sample_shading * (float)samples > 1.0f;
        }
        [cv setConstantValue:&lower type:MTLDataTypeBool atIndex:MGL_RS_FC_POINT_ORIGIN_LOWER];
        [cv setConstantValue:&mask type:MTLDataTypeUInt atIndex:MGL_RS_FC_SAMPLE_MASK];
        [cv setConstantValue:&per_sample type:MTLDataTypeBool atIndex:MGL_RS_FC_PER_SAMPLE];
        /* same as above: function pointer + MSL checksum, so a reused library address
         * cannot resurrect a deleted program's fragment function */
        {
            uint64_t h = 1469598103934665603ULL;
            if (msl) { for (const char *q = msl; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ULL; } }
            key = [NSString stringWithFormat:@"%p:%p:%llx:f:%d:%x:%d", lib, base, (unsigned long long)h, lower, mask, per_sample];
        }
    }

    MGLRSCache *c = [self rstateCache];
    NSArray *hit = c.specialized[key];
    if (hit) return hit[1];

    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:base.name constantValues:cv error:&err];
    if (!fn) {
        NSLog(@"MGL rstate: function specialization failed for %@: %@", base.name, err);
        return base;
    }
    if (c.specialized.count > 256) [c.specialized removeAllObjects];
    c.specialized[key] = @[lib, fn];
    return fn;
}

- (void) rstateConfigurePipelineDescriptor:(MTLRenderPipelineDescriptor *)desc
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    Program *program = ctx->state.program;

    if (!desc || !rs) return;

    /* default framebuffer: the encoder's depth/stencil attachments exist only if the tests were
     * enabled when it was created; start a new encoder when a test needs a missing attachment,
     * then make the pipeline formats match the encoder exactly */
    if (!ctx->state.framebuffer && _renderPassDescriptor) {
        bool needDepth = ctx->state.caps.depth_test && ctx->depth_format.mtl_pixel_format &&
                         !_renderPassDescriptor.depthAttachment.texture;
        bool needStencil = ctx->state.caps.stencil_test && ctx->stencil_format.mtl_pixel_format &&
                           !_renderPassDescriptor.stencilAttachment.texture;
        if ((needDepth || needStencil) && _currentRenderEncoder)
            [self newRenderEncoder];
        id<MTLTexture> dt = _renderPassDescriptor.depthAttachment.texture;
        id<MTLTexture> st = _renderPassDescriptor.stencilAttachment.texture;
        desc.depthAttachmentPixelFormat = dt ? dt.pixelFormat : MTLPixelFormatInvalid;
        desc.stencilAttachmentPixelFormat = st ? st.pixelFormat : MTLPixelFormatInvalid;
    }

    /* blending + write masks per attachment */
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *a = desc.colorAttachments[i];
        if (a.pixelFormat == MTLPixelFormatInvalid) continue;
        bool blend = rs->blend[i] && !rs_format_is_integer(a.pixelFormat);
        a.blendingEnabled = blend;
        a.sourceRGBBlendFactor = rs_blend_factor(ctx->state.var.blend_src_rgb[i]);
        a.destinationRGBBlendFactor = rs_blend_factor(ctx->state.var.blend_dst_rgb[i]);
        a.sourceAlphaBlendFactor = rs_blend_factor(ctx->state.var.blend_src_alpha[i]);
        a.destinationAlphaBlendFactor = rs_blend_factor(ctx->state.var.blend_dst_alpha[i]);
        a.rgbBlendOperation = rs_blend_op(ctx->state.var.blend_equation_rgb[i]);
        a.alphaBlendOperation = rs_blend_op(ctx->state.var.blend_equation_alpha[i]);
        MTLColorWriteMask m = MTLColorWriteMaskNone;
        if (rs->color_mask[i][0]) m |= MTLColorWriteMaskRed;
        if (rs->color_mask[i][1]) m |= MTLColorWriteMaskGreen;
        if (rs->color_mask[i][2]) m |= MTLColorWriteMaskBlue;
        if (rs->color_mask[i][3]) m |= MTLColorWriteMaskAlpha;
        a.writeMask = m;
    }

    bool ms = ctx->state.caps.multisample;
    desc.alphaToCoverageEnabled = ms && ctx->state.caps.sample_alpha_to_coverage;
    desc.alphaToOneEnabled = ms && ctx->state.caps.sample_alpha_to_one;
    /* RASTERIZER_DISCARD: Metal's rasterizationEnabled=NO requires a void vertex function, so the
     * vertex fixup instead collapses every vertex to one point far outside the clip volume
     * (function constant): the vertex stage still runs (side effects), no fragment is produced */
    desc.rasterizationEnabled = YES;

    /* shader fixup specialization (function constants) */
    if (program) {
        /* geom: while the geometry domain's pass-through vertex function owns the vertex
         * stage (generatePipelineDescriptor installed it), the program's own vertex shader
         * must NOT be specialized over it — the pass-through is the last pre-rasterization
         * stage and is specialized separately (geomSpecializeVertexOverride).  Without this
         * the raster pass built a pipeline from the program's vertex shader, which writes
         * point_size while the geometry stage's output topology is triangles. */
        bool geomOwnsVertex = ctx->state.ext_geom &&
                              ((struct MGLExtState_geom *)ctx->state.ext_geom)->vertex_override != NULL;
        /* tess: the render pipeline's vertex stage is the evaluation shader; the vertex
         * stage runs as a compute kernel before the render pass (MGLRenderer+Tessellation.m).
         * Isolines / point_mode run the emulated plain-vertex variant of the TES. */
        int vstage = [self extTessActive] ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
        MGLTessProgram *tp = mgl_tess_active_program(ctx);
        /* tess: the plain (emulated) TES variant also renders the MGL_TESS_CPU reference
         * path — without this the specialized post-tessellation TES would win here and
         * the pipeline (built with no tessellation vertex descriptor) fails to create.
         * xfb capture renders through the plain TES as well (mgl_tess_plain_render). */
        if (mgl_tess_plain_render(ctx))
            desc.vertexFunction = (__bridge id<MTLFunction>)tp->plain_function;
        else if (!geomOwnsVertex)
        {
            Shader *vsh = program->shader_slots[vstage];
            /* tess hardware: the post-tessellation function must declare the
             * control point count drawPatches gets (Metal validates the pair);
             * mgl_tess_hw_tes_function rebuilds it, cached per N, when
             * GL_PATCH_VERTICES changed after the link — specialize the rebuilt
             * function through a standin Shader (like the geometry override) */
            void *hw = (vstage == _TESS_EVALUATION_SHADER && tp && !tp->plain_function)
                       ? mgl_tess_hw_tes_function(ctx, tp) : NULL;
            if (hw)
            {
                Shader standin;
                memset(&standin, 0, sizeof standin);
                standin.entry_point = vsh->entry_point;
                standin.mtl_data.function = hw;
                standin.mtl_data.library = tp->tes_hw_library;
                id<MTLFunction> vf = [self rstateSpecialize: &standin stage: vstage];
                if (vf) desc.vertexFunction = vf;
                else desc.vertexFunction = (__bridge id<MTLFunction>)hw;
            }
            else
            {
                id<MTLFunction> vf = [self rstateSpecialize: vsh stage: vstage];
                if (vf) desc.vertexFunction = vf;
            }
            /* DEBUG: unspecialized TES */
            if (getenv("MGL_TESS_NOSPEC") && vstage == _TESS_EVALUATION_SHADER)
                desc.vertexFunction = (__bridge id<MTLFunction>)program->shader_slots[vstage]->mtl_data.function;
        }
        id<MTLFunction> ff = [self rstateSpecialize:program->shader_slots[_FRAGMENT_SHADER] stage:_FRAGMENT_SHADER];
        if (ff) desc.fragmentFunction = ff;
    }
}

#pragma mark - program recompilation after the MSL fixup

- (void) rstateRecompileProgram:(Program *)program stages:(unsigned)mask
{
    for (int stage = _VERTEX_SHADER; stage <= _FRAGMENT_SHADER; stage++) {
        if (!(mask & (1u << stage))) continue;
        Shader *shader = program->shader_slots[stage];
        if (!shader || !program->spirv[stage].msl_str) continue;
        id<MTLLibrary> lib = [self compileShader:program->spirv[stage].msl_str];
        if (!lib) continue;
        const char *entry = program->spirv[stage].entry_point ? program->spirv[stage].entry_point : shader->entry_point;
        /* specialized with no values: the fixups take their GL-default branches and the
         * function stays usable by any pipeline builder (xfb, fallbacks) */
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry]
                                       constantValues:[[MTLFunctionConstantValues alloc] init] error:nil];
        if (!fn) continue;
        if (shader->mtl_data.library) CFBridgingRelease(shader->mtl_data.library);
        if (shader->mtl_data.function) CFBridgingRelease(shader->mtl_data.function);
        shader->mtl_data.library = (void *)CFBridgingRetain(lib);
        shader->mtl_data.function = (void *)CFBridgingRetain(fn);
    }
}

static void rs_recompile_program(GLMContext glm_ctx, Program *program, unsigned mask)
{
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj rstateRecompileProgram:program stages:mask];
}

#pragma mark - ext hooks

/* C pre-draw hook (called by every draw entry point before the Metal draw) */
static bool rs_pre_draw(GLMContext glm_ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count)
{
    GLMContext ctx = glm_ctx;

    if (mode != s_rs_draw_mode) {
        /* polygon offset depends on the primitive class */
        if (rs_mode_is_polygon(mode) != rs_mode_is_polygon(s_rs_draw_mode) && ctx->state.ext_rstate)
            ctx->state.ext_rstate->generation++;
        s_rs_draw_mode = mode;
    }

    /* glCullFace(GL_FRONT_AND_BACK): every polygon is culled, points and lines still draw */
    if (ctx->state.caps.cull_face && ctx->state.var.cull_face_mode == GL_FRONT_AND_BACK &&
        rs_mode_is_polygon(mode) && (!ctx->state.ext_rstate || ctx->state.ext_rstate->polygon_mode != GL_POINT))
        return false;

    return true;
}

- (void) bindExtRstateFuncsToGLMContext: (GLMContext) glm_ctx
{
    if (!glm_ctx->mtl_funcs.ext_rstate)
        glm_ctx->mtl_funcs.ext_rstate = calloc(1, sizeof(struct MGLExtMetal_rstate));
    glm_ctx->mtl_funcs.ext_rstate->recompile_program = rs_recompile_program;
    mgl_ext_register_pre_draw(rs_pre_draw);
}

- (void) extRstateRenderEncoderCreated: (id<MTLRenderCommandEncoder>) enc
{
    MGLRSCache *c = [self rstateCache];
    c.appliedEncoder = nil;
    c.appliedGeneration = ~0u;
}

- (void) extRstatePreDraw: (id<MTLRenderCommandEncoder>) enc
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    MGLRSCache *c = [self rstateCache];

    if (!enc || !rs) return;
    if (c.appliedEncoder != enc || c.appliedGeneration != rs->generation || c.drawMode != s_rs_draw_mode)
        [self rstateApplyEncoderState:enc];

    /* the pipeline has been rebuilt by now; DIRTY_ALPHA_STATE was the rstate rebuild request */
    ctx->state.dirty_bits &= ~DIRTY_ALPHA_STATE;
}

@end
