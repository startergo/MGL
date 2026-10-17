/*
 * state_ext.c
 * MGL — indexed viewport/scissor, clip control, sample and raster state
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_rstate.h"

void mglClampColor(GLMContext ctx, GLenum target, GLenum clamp)
{
	// Clamp color - no-op, clamping handled automatically
	(void)target;
	(void)clamp;
}

void mglClipControl(GLMContext ctx, GLenum origin, GLenum depth)
{
	// Clip control - no-op, use default clip control
	(void)origin;
	(void)depth;
}

void mglColorMaski(GLMContext ctx, GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
	// Indexed color mask - use global mask for simplicity
	(void)index;
	mglColorMask(ctx, r, g, b, a);
}

void mglDepthRangeArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLdouble *v)
{
	// Depth range array - use first value for global depth range
	if (count > 0)
		mglDepthRange(ctx, v[0], v[1]);
	(void)first;
}

void mglDepthRangeIndexed(GLMContext ctx, GLuint index, GLdouble n, GLdouble f)
{
	// Indexed depth range - use global depth range
	(void)index;
	mglDepthRange(ctx, n, f);
}

void mglDepthRangef(GLMContext ctx, GLfloat n, GLfloat f)
{
	mglDepthRange(ctx, (GLdouble)n, (GLdouble)f);
}

void mglMinSampleShading(GLMContext ctx, GLfloat value)
{
	// Set minimum sample shading - no-op
	(void)value;
}

void mglPatchParameterfv(GLMContext ctx, GLenum pname, const GLfloat *values)
{
	// TODO: Implement
	(void)ctx;
}

void mglPatchParameteri(GLMContext ctx, GLenum pname, GLint value)
{
	// TODO: Implement
	(void)ctx;
}

void mglPolygonOffsetClamp(GLMContext ctx, GLfloat factor, GLfloat units, GLfloat clamp)
{
	// TODO: Implement
	(void)ctx;
}

void mglPrimitiveRestartIndex(GLMContext ctx, GLuint index)
{
	// Set primitive restart index - no-op
	(void)index;
}

void mglProvokingVertex(GLMContext ctx, GLenum mode)
{
	// Set provoking vertex mode - no-op
	if (mode != GL_FIRST_VERTEX_CONVENTION && mode != GL_LAST_VERTEX_CONVENTION) {
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	// State not tracked
}

void mglSampleMaski(GLMContext ctx, GLuint maskNumber, GLbitfield mask)
{
	// TODO: Implement
	(void)ctx;
}

void mglScissorArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLint *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglScissorIndexed(GLMContext ctx, GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height)
{
	// TODO: Implement
	(void)ctx;
}

void mglScissorIndexedv(GLMContext ctx, GLuint index, const GLint *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglViewportArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLfloat *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglViewportIndexedf(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
	// TODO: Implement
	(void)ctx;
}

void mglViewportIndexedfv(GLMContext ctx, GLuint index, const GLfloat *v)
{
	// TODO: Implement
	(void)ctx;
}

void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert)
{
    // Stub - sample coverage is a hint for multisampling
    fprintf(stderr, "MGL: glSampleCoverage called (stub) value=%f invert=%d\n", value, invert);
}

#pragma mark - domain 'rstate' hooks

void mgl_ext_init_rstate(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_rstate(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
