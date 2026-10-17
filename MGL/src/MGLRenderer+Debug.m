/*
 * MGLRenderer+Debug.m — Metal side of the 'debug' domain (KHR_debug).
 *
 *  - debug groups (glPushDebugGroup/glPopDebugGroup) are mirrored onto the
 *    current render command encoder; every newly created render encoder gets
 *    the whole GL group stack re-pushed so Xcode GPU captures show the nesting
 *  - DEBUG_TYPE_MARKER messages become encoder debug signposts
 *  - object labels are copied to the Metal objects' .label (MTLBuffer,
 *    MTLTexture, MTLFunction/MTLLibrary) and re-applied before each commit if
 *    the Metal object was (re)created after labelling
 *  - command-buffer failures feed glGetGraphicsResetStatus
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import "MGLRenderer_Private.h"
#include "mgl_ext_debug.h"

static inline MGLRenderer *mglDebugRenderer(GLMContext glm_ctx)
{
    return (__bridge MGLRenderer *)glm_ctx->mtl_funcs.mtlObj;
}

static void setMTLLabel(void *mtl_obj, NSString *label)
{
    if (!mtl_obj)
        return;

    id obj = (__bridge id)mtl_obj;
    if ([obj respondsToSelector:@selector(setLabel:)])
        [obj setLabel:label];
}

static GLenum resetStatusForCommandBufferError(NSError *error)
{
    if (![error.domain isEqualToString:MTLCommandBufferErrorDomain])
        return GL_UNKNOWN_CONTEXT_RESET;

    switch (error.code)
    {
        case MTLCommandBufferErrorTimeout:
        case MTLCommandBufferErrorPageFault:
        case MTLCommandBufferErrorAccessRevoked:      /* "ignored for causing prior/excessive GPU errors" */
        case MTLCommandBufferErrorInvalidResource:
        case MTLCommandBufferErrorStackOverflow:
            return GL_GUILTY_CONTEXT_RESET;           /* our own commands caused it */

        default:
            return GL_UNKNOWN_CONTEXT_RESET;           /* internal, device removed, OOM, not permitted, ... */
    }
}

@implementation MGLRenderer (Debug)

#pragma mark debug groups

- (void) mglDebugSyncGroupsToEncoder: (id<MTLRenderCommandEncoder>) encoder
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;

    st->mtl_group_encoder = (__bridge void *)encoder;
    st->mtl_group_depth = 0;

    if (!encoder)
        return;

    for (int i = 1; i < st->group_depth; i++)
    {
        const char *msg = st->groups[i].message ? st->groups[i].message : "";
        [encoder pushDebugGroup: [NSString stringWithUTF8String: msg] ?: @"(group)"];
        st->mtl_group_depth++;
    }
}

- (void) mglDebugPushGroup: (const char *) message
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;

    if (!_currentRenderEncoder)
        return;

    if ((__bridge void *)_currentRenderEncoder != st->mtl_group_encoder)
    {
        /* first group activity on this encoder: replay the full stack (includes this group) */
        [self mglDebugSyncGroupsToEncoder: _currentRenderEncoder];
        return;
    }

    [_currentRenderEncoder pushDebugGroup: [NSString stringWithUTF8String: message] ?: @"(group)"];
    st->mtl_group_depth++;
}

- (void) mglDebugPopGroup
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;

    if (!_currentRenderEncoder || (__bridge void *)_currentRenderEncoder != st->mtl_group_encoder)
        return;

    if (st->mtl_group_depth > 0)
    {
        [_currentRenderEncoder popDebugGroup];
        st->mtl_group_depth--;
    }
}

- (void) mglDebugSignpost: (const char *) message
{
    NSString *s = [NSString stringWithUTF8String: message] ?: @"(marker)";

    if (_currentRenderEncoder)
        [_currentRenderEncoder insertDebugSignpost: s];
}

/* auto-called by newRenderEncoder for every new render encoder */
- (void) extDebugRenderEncoderCreated: (id<MTLRenderCommandEncoder>) encoder
{
    if (ctx && ctx->state.ext_debug)
        [self mglDebugSyncGroupsToEncoder: encoder];
}

#pragma mark labels

- (void) mglDebugApplyLabel: (MGLDebugLabel *) e
{
    void *primary = NULL;
    NSString *label = (e->label && e->length) ? [NSString stringWithUTF8String: e->label] : nil;

    switch (e->identifier)
    {
        case GL_BUFFER:
            primary = ((Buffer *)e->object)->data.mtl_data;
            break;

        case GL_TEXTURE:
            primary = ((Texture *)e->object)->mtl_data;
            break;

        case GL_RENDERBUFFER: {
            Renderbuffer *rbo = (Renderbuffer *)e->object;
            primary = rbo->tex ? rbo->tex->mtl_data : NULL;
            break;
        }

        case GL_SHADER: {
            Shader *shader = (Shader *)e->object;
            primary = shader->mtl_data.function;
            if (primary && primary != e->applied_mtl)
                setMTLLabel(shader->mtl_data.library, label);
            break;
        }

        default:
            /* VAOs, framebuffers, programs, pipelines, samplers (MTLSamplerState labels are
               descriptor-only), queries, transform feedback, syncs: no labelable Metal object */
            return;
    }

    if (!primary || primary == e->applied_mtl)
        return;

    setMTLLabel(primary, label);
    e->applied_mtl = primary;
}

/* auto-called before each commit: label Metal objects created since glObjectLabel */
- (void) extDebugWillCommit: (id<MTLCommandBuffer>) cmdbuf
{
    if (!ctx || !ctx->state.ext_debug)
        return;

    struct MGLExtState_debug *st = ctx->state.ext_debug;

    mgl_debug_refresh_metal_labels(ctx);

    [cmdbuf addCompletedHandler: ^(id<MTLCommandBuffer> buffer) {
        if (buffer.status == MTLCommandBufferStatusError || buffer.error)
        {
            GLenum expected = GL_NO_ERROR;
            GLenum status = resetStatusForCommandBufferError(buffer.error);
            __atomic_compare_exchange_n(&st->pending_reset, &expected, status, false,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        }
    }];
}

#pragma mark reset status / device

- (GLenum) mglDebugResetStatus
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;

    if (_currentCommandBuffer && _currentCommandBuffer.status == MTLCommandBufferStatusError)
    {
        if (st->reported_gpu_errors == 0)
        {
            st->reported_gpu_errors = 1;
            return resetStatusForCommandBufferError(_currentCommandBuffer.error);
        }
        return GL_NO_ERROR;
    }

    /* the renderer enters recovery mode after repeated command-buffer failures */
    if (_gpuErrorRecoveryMode)
    {
        if (st->reported_gpu_errors == 0)
        {
            st->reported_gpu_errors = 1;
            return GL_UNKNOWN_CONTEXT_RESET;
        }
        return GL_NO_ERROR;
    }

    st->reported_gpu_errors = 0;
    return GL_NO_ERROR;
}

- (const char *) mglDebugDeviceName
{
    id<MTLDevice> device = _device;

    if (!device)
        device = MTLCreateSystemDefaultDevice();

    return device ? device.name.UTF8String : NULL;
}

#pragma mark binding

static void mtlDebugPushGroup(GLMContext glm_ctx, const char *message)
{
    [mglDebugRenderer(glm_ctx) mglDebugPushGroup: message];
}

static void mtlDebugPopGroup(GLMContext glm_ctx)
{
    [mglDebugRenderer(glm_ctx) mglDebugPopGroup];
}

static void mtlDebugSignpost(GLMContext glm_ctx, const char *message)
{
    [mglDebugRenderer(glm_ctx) mglDebugSignpost: message];
}

static void mtlDebugSetObjectLabel(GLMContext glm_ctx, MGLDebugLabel *entry)
{
    [mglDebugRenderer(glm_ctx) mglDebugApplyLabel: entry];
}

static const char *mtlDebugDeviceName(GLMContext glm_ctx)
{
    return [mglDebugRenderer(glm_ctx) mglDebugDeviceName];
}

static GLenum mtlDebugGetResetStatus(GLMContext glm_ctx)
{
    return [mglDebugRenderer(glm_ctx) mglDebugResetStatus];
}

- (void) bindExtDebugFuncsToGLMContext: (GLMContext) glm_ctx
{
    static struct MGLExtMetal_debug funcs = {
        .mtlPushDebugGroup = mtlDebugPushGroup,
        .mtlPopDebugGroup = mtlDebugPopGroup,
        .mtlInsertDebugSignpost = mtlDebugSignpost,
        .mtlSetObjectLabel = mtlDebugSetObjectLabel,
        .mtlDeviceName = mtlDebugDeviceName,
        .mtlGetResetStatus = mtlDebugGetResetStatus,
    };

    glm_ctx->mtl_funcs.ext_debug = &funcs;
}

@end
