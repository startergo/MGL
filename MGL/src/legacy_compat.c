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
	// Packed color - deprecated, no-op
	(void)type;
	(void)color;
}

void mglColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	// Packed color - deprecated, no-op
	(void)type;
	(void)color;
}

void mglColorP4ui(GLMContext ctx, GLenum type, GLuint color)
{
	// Packed color - deprecated, no-op
	(void)type;
	(void)color;
}

void mglColorP4uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	// Packed color - deprecated, no-op
	(void)type;
	(void)color;
}

void mglGetnMapdv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLdouble *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnMapfv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLfloat *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnMapiv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLint *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnPixelMapfv(GLMContext ctx, GLenum map, GLsizei bufSize, GLfloat *values)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnPixelMapuiv(GLMContext ctx, GLenum map, GLsizei bufSize, GLuint *values)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnPixelMapusv(GLMContext ctx, GLenum map, GLsizei bufSize, GLushort *values)
{
	// TODO: Implement
	(void)ctx;
}

void mglNormalP3ui(GLMContext ctx, GLenum type, GLuint coords)
{
	// TODO: Implement
	(void)ctx;
}

void mglNormalP3uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
	// TODO: Implement
	(void)ctx;
}

void mglSecondaryColorP3ui(GLMContext ctx, GLenum type, GLuint color)
{
	// TODO: Implement
	(void)ctx;
}

void mglSecondaryColorP3uiv(GLMContext ctx, GLenum type, const GLuint *color)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP2ui(GLMContext ctx, GLenum type, GLuint value)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP2uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP3ui(GLMContext ctx, GLenum type, GLuint value)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP3uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP4ui(GLMContext ctx, GLenum type, GLuint value)
{
	// TODO: Implement
	(void)ctx;
}

void mglVertexP4uiv(GLMContext ctx, GLenum type, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{
    // Legacy function for fixed-function pipeline, safe to ignore in modern GL
    fprintf(stderr, "MGL: glClientActiveTexture called (stub) - legacy function\n");
}
