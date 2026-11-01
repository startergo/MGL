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
#import "MGLRenderer+Compat.h"   /* compat perf: mglProgramRegisterMSLIdentity (pipeline fast-path key) */
#import "mgl_ext_tess.h"
#import "mgl_ext_geom.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"
#include "mgl_ext_fb.h"     /* geom r58: mgl_fb_slot_lacks_alpha */
/* compat: GL_KHR_blend_equation_advanced — declared instead of including
 * mgl_ext_compat.h (it redefines ERROR_RETURN with a `return;', which does not
 * fit this file's bool helpers) */
int mgl_advblend_active(GLMContext ctx);   /* compat_advblend.c */
GLboolean mgl_advblend_seen(GLMContext ctx);  /* compat_advblend.c: fast gate (one flag load) */

/* tess r39: GL 4.6 §17.3.4 late-fragment-tests variant for side-effecting
 * fragment shaders (guarded discard + [[depth(any)]] output); mgl_ext_tess.h
 * holds the cache, tessellation.c generates it */
extern char *mgl_tess_late_fragment_variant(GLMContext ctx, struct Program_t *program);

/* per-renderer rstate bookkeeping kept outside the shared ivar block */
@interface MGLRSCache : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSString *, id<MTLDepthStencilState>> *dsStates;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSArray *> *specialized;
@property (nonatomic, weak)   id<MTLRenderCommandEncoder> appliedEncoder;
@property (nonatomic) GLuint appliedGeneration;
@property (nonatomic) GLenum drawMode;
/* one-entry depth/stencil cache: the state vector of the last lookup (integer key,
 * see rstateDepthStencilStateHasDepth) — skips the NSString key + dictionary walk
 * for the common unchanged-state encoder */
@property (nonatomic, strong) id<MTLDepthStencilState> dsLast;
@property (nonatomic) unsigned long long dsLastKey;
@property (nonatomic) BOOL dsLastValid;
/* compat perf (session 9, Apple-driver comparison): last values actually pushed to
 * the encoder.  rstateApplyEncoderState re-ran on every rstate generation bump
 * (any GL state change) and re-set ALL encoder state — the mtlcount A/B showed
 * 19.9 setDepthStencilState/viewport/scissor/stencil-ref calls per glmark2-ideas
 * frame where Apple's translator issues 3.4 (it re-sets only what changed).
 * Skip the Metal call when the value is unchanged since the last apply on the
 * same encoder; a new encoder resets appliedEncoder, which invalidates these. */
@property (nonatomic, weak)   id<MTLRenderCommandEncoder> lastEnc;
@property (nonatomic, strong) id<MTLDepthStencilState> lastDS;
@property (nonatomic) uint32_t lastStencilRefF, lastStencilRefB;
@property (nonatomic) MTLCullMode lastCull;
@property (nonatomic) MTLWinding lastWinding;
@property (nonatomic) MTLTriangleFillMode lastFill;
@property (nonatomic) float lastBiasUnits, lastBiasFactor, lastBiasClamp, lastBiasScale;
@property (nonatomic) BOOL lastBiasOn, lastClipClamp, lastBiasValid;
@property (nonatomic) MTLViewport lastViewport;
@property (nonatomic) MTLScissorRect lastScissor;
@property (nonatomic) GLfloat lastBC0, lastBC1, lastBC2, lastBC3;
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
    MGLRSCache *c = [self rstateCache];

    /* integer key of the whole state vector (the same fields the string key below
     * uses; the stencil masks truncate to 8 bits exactly like that key — stencil
     * buffers are 8-bit).  One-entry cache: every encoder of a frame usually needs
     * the same state, and the string format + dictionary walk costs ~us per encoder
     * creation (10 encoders/frame in a shadow-map ping-pong scene). */
    unsigned long long ik =
        (unsigned long long)dt
        | ((unsigned long long)(v->depth_func & 7) << 1)
        | ((unsigned long long)(v->depth_writemask ? 1 : 0) << 4)
        | ((unsigned long long)st << 5)
        | ((unsigned long long)(v->stencil_func & 7) << 6)
        | ((unsigned long long)(v->stencil_fail & 7) << 9)
        | ((unsigned long long)(v->stencil_pass_depth_fail & 7) << 12)
        | ((unsigned long long)(v->stencil_pass_depth_pass & 7) << 15)
        | ((unsigned long long)(v->stencil_value_mask & 0xff) << 18)
        | ((unsigned long long)(v->stencil_writemask & 0xff) << 26)
        | ((unsigned long long)(v->stencil_back_func & 7) << 34)
        | ((unsigned long long)(v->stencil_back_fail & 7) << 37)
        | ((unsigned long long)(v->stencil_back_pass_depth_fail & 7) << 40)
        | ((unsigned long long)(v->stencil_back_pass_depth_pass & 7) << 43)
        | ((unsigned long long)(v->stencil_back_value_mask & 0xff) << 46)
        | ((unsigned long long)(v->stencil_back_writemask & 0xff) << 54);
    if (c.dsLastValid && c.dsLastKey == ik && c.dsLast) return c.dsLast;

    NSString *key = [NSString stringWithFormat:@"%d:%x:%d|%d:%x:%x:%x:%x:%x:%x|%x:%x:%x:%x:%x:%x",
                     dt, v->depth_func, v->depth_writemask, st,
                     v->stencil_func, v->stencil_fail, v->stencil_pass_depth_fail, v->stencil_pass_depth_pass,
                     v->stencil_value_mask & 0xff, v->stencil_writemask & 0xff,
                     v->stencil_back_func, v->stencil_back_fail, v->stencil_back_pass_depth_fail,
                     v->stencil_back_pass_depth_pass, v->stencil_back_value_mask & 0xff, v->stencil_back_writemask & 0xff];
    id<MTLDepthStencilState> s = c.dsStates[key];
    if (s) {
        c.dsLast = s; c.dsLastKey = ik; c.dsLastValid = YES;
        return s;
    }

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
        c.dsLast = s; c.dsLastKey = ik; c.dsLastValid = YES;
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

    /* session-9 last-applied dedup: locals shadow the MGLRSCache fields, written
     * back at the end (see the property comment above) */
    MGLRSCache *c = [self rstateCache];
    id<MTLRenderCommandEncoder> c_lastEnc = c.lastEnc;
    id<MTLDepthStencilState> c_lastDS = c.lastDS;
    uint32_t c_lastRefF = c.lastStencilRefF, c_lastRefB = c.lastStencilRefB;

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
    bool freshEnc = (c_lastEnc != enc);
    if (freshEnc) c_lastEnc = enc;
    if (ds && (freshEnc || ds != c_lastDS)) {
        [enc setDepthStencilState:ds];
        c_lastDS = ds;
    }
    uint32_t srefF = (uint32_t)v->stencil_ref & 0xff, srefB = (uint32_t)v->stencil_back_ref & 0xff;
    if (freshEnc || srefF != c_lastRefF || srefB != c_lastRefB) {
        [enc setStencilFrontReferenceValue:srefF backReferenceValue:srefB];
        c_lastRefF = srefF; c_lastRefB = srefB;
    }

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
    else if (freshEnc || vps[0].originX != c.lastViewport.originX || vps[0].originY != c.lastViewport.originY ||
             vps[0].width != c.lastViewport.width || vps[0].height != c.lastViewport.height ||
             vps[0].znear != c.lastViewport.znear || vps[0].zfar != c.lastViewport.zfar) {
        [enc setViewport:vps[0]];
        c.lastViewport = vps[0];
    }

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
        else if (freshEnc || sr[0].x != c.lastScissor.x || sr[0].y != c.lastScissor.y ||
                 sr[0].width != c.lastScissor.width || sr[0].height != c.lastScissor.height) {
            [enc setScissorRect:sr[0]];
            c.lastScissor = sr[0];
        }
    }

    /* culling.  FRONT_AND_BACK (no Metal mode) is handled by skipping polygon draws in the
     * pre-draw hook. */
    MTLCullMode cull = MTLCullModeNone;
    if (ctx->state.caps.cull_face) {
        if (v->cull_face_mode == GL_FRONT) cull = MTLCullModeFront;
        else if (v->cull_face_mode == GL_BACK) cull = MTLCullModeBack;
    }
    if (freshEnc || cull != c.lastCull) {
        [enc setCullMode:cull];
        c.lastCull = cull;
    }

    /* winding: the LOWER_LEFT y-flip mirrors window space, which reverses orientation */
    bool flipped = rs->clip_origin == GL_LOWER_LEFT;
    bool ccw = v->front_face == GL_CCW;
    if (flipped) ccw = !ccw;
    MTLWinding winding = ccw ? MTLWindingCounterClockwise : MTLWindingClockwise;
    if (freshEnc || winding != c.lastWinding) {
        [enc setFrontFacingWinding:winding];
        c.lastWinding = winding;
    }

    /* polygon mode: LINE natively; POINT is emulated by the draw path (index rewrite to
     * GL_POINTS via mgl_rstate_index_rewrite_needed), FILL otherwise */
    MTLTriangleFillMode fill = rs->polygon_mode == GL_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
    if (freshEnc || fill != c.lastFill) {
        [enc setTriangleFillMode:fill];
        c.lastFill = fill;
    }

    /* polygon offset: only polygons, enable selected by the polygon mode.  Metal's depth
     * bias has factor * max depth slope (no r on the slope term, like GL) and the
     * ARB_polygon_offset_clamp one-sided clamp (clamp>0: min(o,c), clamp<0: max(o,c),
     * 0: unclamped — measured on M4 Pro), so factor and clamp pass through unchanged.
     * The bias term, however, is always scaled by 2^-24 no matter which depth format is
     * attached, while GL's r is the attachment's resolvable difference (1/(2^n-1) for an
     * n-bit fixed-point depth buffer).  Pre-scale the bias by r_gl / 2^-24 (256x for a
     * 16-bit depth attachment) so the effective offset — and the clamp — bind in GL units. */
    bool offset = false;
    if (rs_mode_is_polygon(s_rs_draw_mode)) {
        switch (rs->polygon_mode) {
            case GL_LINE:  offset = ctx->state.caps.polygon_offset_line; break;
            case GL_POINT: offset = ctx->state.caps.polygon_offset_point; break;
            default:       offset = ctx->state.caps.polygon_offset_fill; break;
        }
    }
    float biasScale = 1.0f;
    if (offset) {
        /* GL's r: 1/(2^n-1) for an n-bit fixed-point depth attachment, 2^-24 otherwise.
         * Metal's r: 2^-16 for a native Depth16Unorm attachment, 2^-24 for float storage
         * (MGL keeps GL_DEPTH_COMPONENT16 textures in Depth32Float, quantized to the
         * 16-bit lattice; renderbuffers use true Depth16Unorm). */
        GLenum dfmt = GL_NONE;
        if (ctx->state.framebuffer) {
            FBOAttachment *da = &ctx->state.framebuffer->depth;
            if (da->textarget == GL_RENDERBUFFER) {
                if (da->buf.rbo) dfmt = ((MGLRenderbufferEx *)da->buf.rbo)->internalformat;
            } else if (da->buf.tex) {
                dfmt = da->buf.tex->internalformat;
            }
        }
        float rGl = 1.0f / 16777216.0f;
        if (dfmt == GL_DEPTH_COMPONENT16)
            rGl = 1.0f / 65535.0f;
        id<MTLTexture> dt = _renderPassDescriptor.depthAttachment.texture;
        float rMtl = (dt && dt.pixelFormat == MTLPixelFormatDepth16Unorm) ?
                        1.0f / 65536.0f : 1.0f / 16777216.0f;
        biasScale = rGl / rMtl;
    }
    bool biasSame = !freshEnc && c.lastBiasValid && c.lastBiasOn == offset &&
        (!offset || (rs->polygon_offset_units == c.lastBiasUnits &&
                     rs->polygon_offset_factor == c.lastBiasFactor &&
                     rs->polygon_offset_clamp == c.lastBiasClamp &&
                     biasScale == c.lastBiasScale));
    if (!biasSame) {
        if (offset)
            [enc setDepthBias:rs->polygon_offset_units * biasScale slopeScale:rs->polygon_offset_factor clamp:rs->polygon_offset_clamp];
        else
            [enc setDepthBias:0 slopeScale:0 clamp:0];
        c.lastBiasOn = offset;
        c.lastBiasUnits = rs->polygon_offset_units;
        c.lastBiasFactor = rs->polygon_offset_factor;
        c.lastBiasClamp = rs->polygon_offset_clamp;
        c.lastBiasScale = biasScale;
        c.lastBiasValid = YES;
    }

    bool clipClamp = ctx->state.caps.depth_clamp;
    if (freshEnc || clipClamp != c.lastClipClamp) {
        [enc setDepthClipMode: clipClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
        c.lastClipClamp = clipClamp;
    }

    if (freshEnc || v->blend_color[0] != c.lastBC0 || v->blend_color[1] != c.lastBC1 ||
        v->blend_color[2] != c.lastBC2 || v->blend_color[3] != c.lastBC3) {
        [enc setBlendColorRed:v->blend_color[0] green:v->blend_color[1] blue:v->blend_color[2] alpha:v->blend_color[3]];
        c.lastBC0 = v->blend_color[0]; c.lastBC1 = v->blend_color[1];
        c.lastBC2 = v->blend_color[2]; c.lastBC3 = v->blend_color[3];
    }

    c.lastEnc = c_lastEnc;
    c.lastDS = c_lastDS;
    c.lastStencilRefF = c_lastRefF;
    c.lastStencilRefB = c_lastRefB;
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
    /* compat: a CLAMP_TO_BORDER border colour Metal cannot express is emulated by
     * an MSL rewrite of the fragment stage (compat_border.c) — even an otherwise
     * unpatched stage must reach the specialization below for it */
    bool border_emu = (stage == _FRAGMENT_SHADER) && mgl_border_wanted(ctx);
    /* compat: GL_KHR_blend_equation_advanced — the advanced equations run in a
     * framebuffer-fetch rewrite of the fragment stage (compat_advblend.c), so an
     * active equation must reach the variant swap below as well */
    bool advblend = (stage == _FRAGMENT_SHADER) &&
                    mgl_advblend_seen(ctx) && mgl_advblend_active(ctx) >= 0;
    if (getenv("MGL_BORDER_DEBUG") && stage == _FRAGMENT_SHADER)
        fprintf(stderr, "MGL border: rstateSpecialize frag, wanted=%d patched=%d\n",
                (int)border_emu, msl ? mgl_rstate_msl_is_patched(msl) : -1);
    /* geom: the vertex stage's shadow-sampling emulation carries its own function
     * constants (32010+slot) that must be specialized even when the rstate fixups
     * did not patch this MSL (e.g. a vertex function without a [[position]] output) */
    /* a STANDIN Shader (geometry vertex override, tessellation plain/hardware TES
     * variant) is compiled from its own MSL, not program->spirv[stage]: this test must
     * not reject it because the program's stage MSL is unpatched — the standin's own
     * fixup (clip flip / PROGRAM_POINT_SIZE) is what needs the constants.  Seen as
     * size-1 points when a GS consumed tessellated points while the program's vertex
     * MSL had no position output to patch.  Specializing a function that declares no
     * constants is a harmless rebuild. */
    bool standinFn = shader && shader->mtl_data.function && program &&
                     program->shader_slots[stage] &&
                     shader->mtl_data.function != program->shader_slots[stage]->mtl_data.function;
    /* tess r39: the late-fragment-tests variant (GL 4.6 §17.3.4) must reach the
     * variant swap below even when this stage carries no rstate fixups */
    bool lateFragVariant = (stage == _FRAGMENT_SHADER && program &&
                            ctx->state.framebuffer && ctx->state.framebuffer->depth.texture &&
                            mgl_tess_late_fragment_variant(ctx, program) != NULL);
    if (!mgl_rstate_msl_is_patched(msl) &&
        !(msl && strstr(msl, "mgl_shadow_op_") != NULL) &&
        !(msl && strstr(msl, "mgl_wrap_op_") != NULL) && !standinFn && !border_emu &&
        !advblend && !lateFragVariant)
        return (__bridge id<MTLFunction>)shader->mtl_data.function;

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
        /* geom: shadow-compare function constants of the vertex stage's emulation
         * (a no-op unless spirv[VS].shadow_slots is set) — the values are part of
         * the cache key below, a compare-function change re-specializes */
        uint64_t shash = 0;
        [self geomShadowFCValues:cv stage:stage hash:&shash];
        /* the key must not collide when a program is deleted and a later one's Metal
         * objects land on the same addresses: the specialized entry would hand the dead
         * program's function to the new one and it would render with the wrong shader
         * (seen as flaky CTS failures once per-program recompiles made reuse common).
         * The function pointer plus a checksum of the MSL make that practically
         * impossible. */
        uint64_t h = 1469598103934665603ULL;
        if (msl) { for (const char *q = msl; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ULL; } }
        key = [NSString stringWithFormat:@"%p:%p:%llx:v:%d%d%d%d:%g:%llx", lib, base, (unsigned long long)h, z01, ul, pps, rd, psz, (unsigned long long)shash];
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
        uint32_t psamples = (uint32_t)MAX(samples, 1);
        [cv setConstantValue:&psamples type:MTLDataTypeUInt atIndex:MGL_RS_FC_PIPELINE_SAMPLES];
        /* per-sample shading needs sample-qualified interpolants: Metal evaluates
         * center-qualified stage inputs once per pixel even when the function
         * reads [[sample_id]], so every sample of a pixel would shade to the same
         * color (GL_SAMPLE_SHADING, KHR-GL46.sample_shading).  Swap in the
         * sample-qualified variant of this MSL, compiled lazily and cached by
         * text — only programs actually drawn with sample shading pay for it. */
        char *pss_msl = NULL;
        if (per_sample)
        {
            pss_msl = mgl_rstate_pss_fragment_msl(msl, [base.name UTF8String]);
            if (pss_msl)
            {
                id<MTLLibrary> plib = [self extProgramLibraryForMSL: pss_msl key: nil error: nil];
                id<MTLFunction> pfn = plib ? [plib newFunctionWithName: base.name] : nil;
                if (pfn)
                {
                    lib = plib;
                    base = pfn;
                    msl = pss_msl;
                }
            }
        }
        /* compat: GL_CLAMP_TO_BORDER with a border colour Metal's samplers cannot
         * express (they have three fixed colours only) — swap in the lazily
         * rewritten MSL variant whose plain sample calls return the exact border
         * colour (compat_border.c).  Cached by source MSL pointer + table
         * signature; the compiled library is cached by text like the PSS one. */
        MGLBorderEmu bemu[MGL_BORDER_MAX];
        int nbemu = mgl_border_scan(ctx, _FRAGMENT_SHADER, bemu);
        if (getenv("MGL_BORDER_DEBUG"))
            fprintf(stderr, "MGL border: stage %d scan -> %d slots%s\n", stage, nbemu,
                    nbemu ? " (emulated)" : "");
        if (nbemu > 0)
        {
            uint64_t bsig = mgl_border_sig(bemu, nbemu);
            if (getenv("MGL_BORDER_DEBUG"))
                fprintf(stderr, "MGL border: sig %llx slot %u axes %u lin %u kind %u bc %s\n",
                        (unsigned long long)bsig, bemu[0].msl_slot, bemu[0].axes, bemu[0].linear,
                        bemu[0].kind, bemu[0].bc);
            if (!rs->border_msl || rs->border_msl_src != (const void *)msl || rs->border_msl_sig != bsig)
            {
                char *v = mgl_border_rewrite(msl, bemu, nbemu);
                if (getenv("MGL_BORDER_DEBUG"))
                {
                    fprintf(stderr, "MGL border: rewrite %s\n", v ? "OK" : "no match");
                    if (v) for (const char *q = v; *q; ) { const char *e = strchr(q, '\n'); size_t l = e ? (size_t)(e - q) + 1 : strlen(q); if (strstr(q, "mgl_bsamp") || strstr(q, "mgl_border:") || strstr(q, "sample")) fwrite(q, 1, l, stderr); q += l; }
                }
                if (v)
                {
                    free(rs->border_msl);
                    rs->border_msl = v;
                    rs->border_msl_src = (const void *)msl;
                    rs->border_msl_sig = bsig;
                }
                else if (rs->border_msl_src == (const void *)msl)
                {
                    /* this source has no rewritable call: remember so we do not
                     * re-walk the text on every pipeline build */
                    free(rs->border_msl);
                    rs->border_msl = NULL;
                    rs->border_msl_src = (const void *)msl;
                    rs->border_msl_sig = bsig;
                }
            }
            if (rs->border_msl && rs->border_msl_src == (const void *)msl)
            {
                char *berr_msg = NULL;
                if (getenv("MGL_BORDER_DUMP"))
                {
                    char p_[256];
                    snprintf(p_, sizeof p_, "%s/variant_stage%d.msl", getenv("MGL_BORDER_DUMP"), stage);
                    FILE *f = fopen(p_, "wb");
                    if (f) { fwrite(rs->border_msl, 1, strlen(rs->border_msl), f); fclose(f); }
                }
                id<MTLLibrary> blib = [self extProgramLibraryForMSL: rs->border_msl key: nil error: &berr_msg];
                if (berr_msg)
                {
                    if (getenv("MGL_BORDER_DEBUG"))
                        fprintf(stderr, "MGL border: variant compile: %.600s\n", berr_msg);
                    free(berr_msg);   /* strdup'd message from the compiler */
                }
                id<MTLFunction> bfn = blib ? [blib newFunctionWithName: base.name] : nil;
                if (getenv("MGL_BORDER_DEBUG") && !bfn && blib)
                    fprintf(stderr, "MGL border: fn '%s' not among [%@] (head: %.200s)\n",
                            [base.name UTF8String], [blib functionNames],
                            rs->border_msl ? rs->border_msl : "");
                if (getenv("MGL_BORDER_DEBUG"))
                    fprintf(stderr, "MGL border: variant fn=%s\n", bfn ? "OK" : "MISSING");
                if (bfn)
                {
                    lib = blib;
                    base = bfn;
                    msl = rs->border_msl;
                }
            }
        }
        /* compat: GL_KHR_blend_equation_advanced — swap in the framebuffer-fetch
         * variant of THIS fragment MSL (or, with the border variant in, of that
         * rewrite); Metal fixed-function blending is off for attachment 0 while
         * this function is in use (rstateConfigurePipelineDescriptor).  A no-op
         * returning false unless an advanced equation is active. */
        if (advblend)
        {
            id<MTLLibrary> alib = nil;
            id<MTLFunction> afn = nil;
            const char *amsl = NULL;
            if ([self compatAdvancedBlendVariant: shader msl: &amsl library: &alib function: &afn] && afn)
            {
                lib = alib;
                base = afn;
                if (amsl) msl = amsl;
                /* MSAA passes only define the [[color(0)]] read per sample, so the
                 * equation must run per sample there (single-sample passes keep
                 * the ordinary per-pixel shading) */
                if (psamples > 1)
                    per_sample = true;
            }
        }
        /* tess r39 (GL 4.6 §17.3.4): a fragment shader that writes to memory
         * (image/SSBO/atomic counter) and does not declare
         * layout(early_fragment_tests) must execute before the depth test —
         * Apple GPUs cull depth-failed fragments before shading (HSR).  When
         * this pipeline has a depth attachment to test against, swap in the
         * guarded-discard + [[depth(any)]] variant built at link time; Metal
         * rejects a depth output on depthless pipelines, and without one no
         * test can defer.  The GL framebuffer state is the truth here: with a
         * geometry shader the pipeline descriptor is built before the render
         * pass descriptor is finalized. */
        if (stage == _FRAGMENT_SHADER && program &&
            ctx->state.framebuffer && ctx->state.framebuffer->depth.texture)
        {
            char *lft = mgl_tess_late_fragment_variant(ctx, program);
            if (getenv("MGL_LFT_DEBUG"))
                fprintf(stderr, "[lft] psoswap: stage=%d prog=%u variant=%s\n",
                        stage, program->name, lft ? "yes" : "no");
            if (lft)
            {
                char lerr[2048];
                char *lerrp = &lerr[0];
                lerr[0] = 0;
                id<MTLLibrary> llib = [self extProgramLibraryForMSL: lft key: nil error: &lerrp];
                id<MTLFunction> lfn = llib ? [llib newFunctionWithName: base.name] : nil;
                if (getenv("MGL_LFT_DEBUG") && lerr[0])
                    fprintf(stderr, "[lft] variant compile: %.400s\n", lerr);
                if (lfn)
                {
                    lib = llib;
                    base = lfn;
                    msl = lft;
                }
            }
        }
        /* same as above: function pointer + MSL checksum, so a reused library address
         * cannot resurrect a deleted program's fragment function */
        {
            uint64_t h = 1469598103934665603ULL;
            if (msl) { for (const char *q = msl; *q; q++) { h ^= (unsigned char)*q; h *= 1099511628211ULL; } }
            key = [NSString stringWithFormat:@"%p:%p:%llx:f:%d:%x:%d:%u", lib, base, (unsigned long long)h, lower, mask, per_sample, psamples];
        }
        free(pss_msl);
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

/* compact content hash of everything rstate bakes into the MTLRenderPipelineState
 * (blend per attachment, write masks, alpha coverage, the default framebuffer's
 * depth/stencil attachment flip, and the shader function constants rstateSpecialize
 * specializes over).  Used as a PSO-cache key component by the compat pipeline
 * fast path (MGLRenderer+Compat.m): a monotonic generation counter there made
 * alternating state (glmark2 toggles blend per object) miss the cache on every
 * change, walking the full descriptor path ~13x per frame.  Hashing the CONTENT
 * instead makes an A/B/A/B round-trip two stable keys.  Extra components beyond
 * generation_pipeline's dirty envelope only add misses, never stale reuse. */
- (uint64_t) rstatePipelineContentHash
{
    struct MGLExtState_rstate *rs = ctx->state.ext_rstate;
    uint64_t h = 1469598103934665603ULL;
    #define RS_MIXV(v) do { h ^= (uint64_t)(v); h *= 1099511628211ULL; } while (0)
    if (rs)
    {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            RS_MIXV(rs->blend[i]);
            RS_MIXV(ctx->state.var.blend_src_rgb[i]);  RS_MIXV(ctx->state.var.blend_dst_rgb[i]);
            RS_MIXV(ctx->state.var.blend_src_alpha[i]); RS_MIXV(ctx->state.var.blend_dst_alpha[i]);
            RS_MIXV(ctx->state.var.blend_equation_rgb[i]); RS_MIXV(ctx->state.var.blend_equation_alpha[i]);
            for (int c = 0; c < 4; c++) RS_MIXV(rs->color_mask[i][c]);
        }
        RS_MIXV(rs->clip_depth_mode);
        RS_MIXV(rs->clip_origin);
        RS_MIXV(rs->point_sprite_coord_origin);
        RS_MIXV(rs->sample_mask[0]);
        RS_MIXV(rs->sample_coverage_value);
        RS_MIXV(rs->sample_coverage_invert);
        RS_MIXV(rs->min_sample_shading);
    }
    RS_MIXV(ctx->state.caps.multisample);
    RS_MIXV(ctx->state.caps.sample_alpha_to_coverage);
    RS_MIXV(ctx->state.caps.sample_alpha_to_one);
    RS_MIXV(ctx->state.caps.sample_mask);
    RS_MIXV(ctx->state.caps.sample_coverage);
    RS_MIXV(ctx->state.caps.sample_shading);
    RS_MIXV(ctx->state.caps.program_point_size);
    RS_MIXV(ctx->state.caps.rasterizer_discard);
    /* the default framebuffer's depth/stencil attachments appear/disappear with
     * the tests (rstateConfigurePipelineDescriptor starts a new encoder for it) */
    RS_MIXV(ctx->state.caps.depth_test);
    RS_MIXV(ctx->state.caps.stencil_test);
    {
        float psz = ctx->state.var.point_size;
        if (psz < 1.0f) psz = 1.0f;
        if (psz > 511.0f) psz = 511.0f;
        uint32_t bits; memcpy(&bits, &psz, 4);
        RS_MIXV(bits);
    }
    #undef RS_MIXV
    return h;
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
        /* compat: GL_KHR_blend_equation_advanced (GL 4.6 §17.3.8) — the advanced
         * equations composite inside the fragment shader (framebuffer fetch,
         * compat_advblend.c), so Metal's fixed-function unit must stay off for
         * attachment 0 while one is selected; plain equations are unchanged */
        if (i == 0 && blend && mgl_advblend_seen(ctx) && mgl_advblend_active(ctx) >= 0)
            blend = false;
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
        /* geom r58: GL formats with no alpha channel (R3_G3_B2, RGB*, RG*, R*)
         * live on an RGBA Metal backing; GL discards writes to the missing
         * channel and reads return 1.0 — mask the alpha write or a fill leaks
         * its alpha into the backing, which a GL_TEXTURE_SWIZZLE sourcing A
         * then reads back (texture_swizzle.functional). */
        if ((m & MTLColorWriteMaskAlpha) && mgl_fb_slot_lacks_alpha(ctx, ctx->state.framebuffer, (GLuint)i))
            m &= ~MTLColorWriteMaskAlpha;
        a.writeMask = m;
    }

    bool ms = ctx->state.caps.multisample;
    desc.alphaToCoverageEnabled = ms && ctx->state.caps.sample_alpha_to_coverage;
    desc.alphaToOneEnabled = ms && ctx->state.caps.sample_alpha_to_one;
    /* RASTERIZER_DISCARD: Metal's rasterizationEnabled=NO requires a void vertex function, so the
     * vertex fixup instead collapses every vertex to one point far outside the clip volume
     * (function constant): the vertex stage still runs (side effects), no fragment is produced.
     * A vertex-only pipeline (fragment shader absent — GL rasterizer-discard / SSBO-writing
     * vertex programs) keeps rasterization disabled: its vertex function is void. */
    desc.rasterizationEnabled = (desc.fragmentFunction != nil);

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
        {
            /* specialize the plain TES like the hardware variant below: it carries the
             * vertex fixup (clip flip + PROGRAM_POINT_SIZE / point size function
             * constants).  Without this the defaults applied and the TES's gl_PointSize
             * was dropped (point_mode.point_rendering failed: size-1 points). */
            Shader standin;
            Shader *tsh = program->shader_slots[_TESS_EVALUATION_SHADER];

            memset(&standin, 0, sizeof standin);
            standin.entry_point = tsh ? tsh->entry_point : NULL;
            standin.mtl_data.function = tp->plain_function;
            standin.mtl_data.library = tp->plain_library;
            id<MTLFunction> vf = [self rstateSpecialize: &standin stage: _TESS_EVALUATION_SHADER];
            desc.vertexFunction = vf ? vf : (__bridge id<MTLFunction>)tp->plain_function;
        }
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
            /* DEBUG: unspecialized TES (knob latched once — this runs per
             * pipeline rebuild, i.e. per draw when the descriptor path is hot) */
            static int sTessNospecInit = 0, sTessNospec = 0;
            if (!sTessNospecInit) { sTessNospec = getenv("MGL_TESS_NOSPEC") != NULL; sTessNospecInit = 1; }
            if (sTessNospec && vstage == _TESS_EVALUATION_SHADER)
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
        /* compat perf (surgical, one call): record the fixed-up function's MSL
         * content identity — the compat pipeline fast path keys on it, and this
         * install path bypasses extProgramCompileMSL (bindMTLProgram then sees
         * a non-NULL library and skips, leaving the function unregistered, so
         * every draw of the program would decline the fast path) */
        mglProgramRegisterMSLIdentity(shader->mtl_data.function,
                                      program->spirv[stage].msl_str, entry);
    }
}

static void rs_recompile_program(GLMContext glm_ctx, Program *program, unsigned mask)
{
    [(__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj rstateRecompileProgram:program stages:mask];
}

#pragma mark - ext hooks

/* C pre-draw hook (called by every draw entry point before the Metal draw) */
static bool rs_pre_draw(GLMContext glm_ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count,
                        bool indexed, GLenum type, size_t offset)
{
    (void)vertex_count; (void)instance_count; (void)indexed; (void)type; (void)offset;
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

    /* compat: CLAMP_TO_BORDER border-colour emulation — the fragment MSL variant
     * with the exact border colour is baked into the pipeline state, so a change
     * of the per-slot table must rebuild it.  Runs before the state walk (this
     * hook precedes processGLState) and stays off unless an active unit actually
     * clamps to border, so the per-draw cost is one scan of the active units. */
    if (ctx->state.program && ctx->state.ext_rstate && mgl_border_wanted(ctx))
    {
        MGLBorderEmu be[MGL_BORDER_MAX];
        int nb = mgl_border_scan(ctx, _FRAGMENT_SHADER, be);
        uint64_t sig = mgl_border_sig(be, nb);
        if (ctx->state.ext_rstate->border_sig != sig)
        {
            ctx->state.ext_rstate->border_sig = sig;
            ctx->state.dirty_bits |= DIRTY_PROGRAM;
        }
    }

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
