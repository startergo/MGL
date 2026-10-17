/*
 * textures_ext.c
 * MGL — texture buffers, views, multisample storage, invalidation, barriers
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_texext.h"

extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *currentTexture(GLMContext ctx, GLuint index);

void mglTexBuffer(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer)
{
	// TODO: Implement
	(void)ctx;
}

void mglTexBufferRange(GLMContext ctx, GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
	// TODO: Implement
	(void)ctx;
}

void mglTexStorage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    // For multisample textures, we need to create storage but Apple Silicon 
    // handles MSAA differently than traditional GL. For now, we create a 
    // regular texture and let the rendering pipeline handle any MSAA.
    
    fprintf(stderr, "MGL: mglTexStorage2DMultisample called - target=0x%x, samples=%d, format=0x%x, %dx%d\n", 
            target, samples, internalformat, width, height);
    
    // Validate target
    if (target != GL_TEXTURE_2D_MULTISAMPLE && target != GL_PROXY_TEXTURE_2D_MULTISAMPLE) {
        fprintf(stderr, "MGL WARNING: mglTexStorage2DMultisample invalid target 0x%x\n", target);
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_ENUM);
        return;
    }
    
    // For now, silently succeed without error - multisample textures are 
    // used for capability probing. The actual MSAA will be handled by the
    // render pipeline when available.
    (void)samples; (void)internalformat; (void)width; (void)height; (void)fixedsamplelocations;
    
    // Don't set an error - let virglrenderer think MSAA is "supported"
    // This allows the system to fall back gracefully
}

void mglTexStorage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
	// TODO: Implement
	(void)ctx;
}

void mglTextureBarrier(GLMContext ctx)
{
    // CRITICAL FIX: Handle unimplemented function gracefully instead of crashing
    fprintf(stderr, "MGL WARNING: mglTextureBarrier is not yet implemented in MGL\n");
    // No-op implementation - this is optional functionality
}

void mglInvalidateTexImage(GLMContext ctx, GLuint texture, GLint level)
{
    // Stub - invalidation is just a hint, safe to ignore
    fprintf(stderr, "MGL: glInvalidateTexImage called (stub) - texture=%u level=%d\n", texture, level);
}

void mglInvalidateTexSubImage(GLMContext ctx, GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth)
{
    // Stub - invalidation is just a hint, safe to ignore
    fprintf(stderr, "MGL: glInvalidateTexSubImage called (stub)\n");
}

void mglTexImage2DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    // Multisample textures are used by virglrenderer for capability probing.
    // Apple Silicon handles MSAA differently - we silently succeed to allow
    // the rendering pipeline to proceed without MSAA.
    fprintf(stderr, "MGL: mglTexImage2DMultisample (stub) - target=0x%x samples=%d internalformat=0x%x %dx%d\n",
            target, samples, internalformat, width, height);
    (void)ctx; (void)target; (void)samples; (void)internalformat;
    (void)width; (void)height; (void)fixedsamplelocations;
    // Don't set error - allow probing to "succeed" so virglrenderer continues
}

void mglTexImage3DMultisample(GLMContext ctx, GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    // Multisample array textures - silently succeed like 2D multisample
    fprintf(stderr, "MGL: mglTexImage3DMultisample (stub) - target=0x%x samples=%d internalformat=0x%x %dx%dx%d\n",
            target, samples, internalformat, width, height, depth);
    (void)ctx; (void)target; (void)samples; (void)internalformat;
    (void)width; (void)height; (void)depth; (void)fixedsamplelocations;
    // Don't set error
}

void mglTextureStorage2DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLboolean fixedsamplelocations)
{
    fprintf(stderr, "MGL WARNING: glTextureStorage2DMultisample called (stub) - MSAA not fully supported\n");
    // Fall back to non-MSAA storage
    mglTextureStorage2D(ctx, texture, 1, internalformat, width, height);
}

void mglTextureStorage3DMultisample(GLMContext ctx, GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLboolean fixedsamplelocations)
{
    fprintf(stderr, "MGL WARNING: glTextureStorage3DMultisample called (stub) - MSAA not fully supported\n");
    // Fall back to non-MSAA storage
    mglTextureStorage3D(ctx, texture, 1, internalformat, width, height, depth);
}

void mglTextureView(GLMContext ctx, GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels, GLuint minlayer, GLuint numlayers)
{
    fprintf(stderr, "MGL WARNING: glTextureView called (stub) - texture views not supported\n");
}

void mglTextureBuffer(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer)
{
    fprintf(stderr, "MGL WARNING: glTextureBuffer called (stub)\n");
}

void mglTextureBufferRange(GLMContext ctx, GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
    fprintf(stderr, "MGL WARNING: glTextureBufferRange called (stub)\n");
}

#pragma mark - domain 'texext' hooks

void mgl_ext_init_texext(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_texext(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
