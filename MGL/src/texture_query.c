/*
 * texture_query.c
 * MGL — texture parameter / level / internalformat queries
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_texparam.h"

extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern Texture *currentTexture(GLMContext ctx, GLuint index);

void mglGetTextureParameterfv(GLMContext ctx, GLuint texture, GLenum pname, GLfloat *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterfv called (stub) pname=0x%x\n", pname);
    *params = 0.0f;
}

void mglGetTextureParameterIiv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterIiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTextureParameterIuiv(GLMContext ctx, GLuint texture, GLenum pname, GLuint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameterIuiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTextureParameteriv(GLMContext ctx, GLuint texture, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTextureParameteriv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTexParameterIiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    fprintf(stderr, "MGL: glGetTexParameterIiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetTexParameterIuiv(GLMContext ctx, GLenum target, GLenum pname, GLuint *params)
{
    fprintf(stderr, "MGL: glGetTexParameterIuiv called (stub) pname=0x%x\n", pname);
    *params = 0;
}

void mglGetInternalformati64v(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64 *params)
{
        assert(0);
}

void mglGetInternalformativ(GLMContext ctx, GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint *params)
{
        assert(0);
}

#pragma mark - domain 'texparam' hooks

void mgl_ext_init_texparam(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_texparam(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
