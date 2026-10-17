/*
 * legacy_compat.c
 * MGL — compatibility-profile entry points (core profile: GL_INVALID_OPERATION)
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"

void mglColorP3ui(GLMContext ctx, GLenum type, GLuint color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColorP4ui(GLMContext ctx, GLenum type, GLuint color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColorP4uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnMapdv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnMapfv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnMapiv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnPixelMapfv(GLMContext ctx, GLenum map, GLsizei bufSize, GLfloat *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnPixelMapuiv(GLMContext ctx, GLenum map, GLsizei bufSize, GLuint *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnPixelMapusv(GLMContext ctx, GLenum map, GLsizei bufSize, GLushort *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormalP3ui(GLMContext ctx, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormalP3uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColorP3ui(GLMContext ctx, GLenum type, GLuint color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP2ui(GLMContext ctx, GLenum type, GLuint value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP2uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP3ui(GLMContext ctx, GLenum type, GLuint value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP3uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP4ui(GLMContext ctx, GLenum type, GLuint value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexP4uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}
