/*
 * debug.c
 * MGL — KHR_debug output, object labels, robustness status
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_debug.h"

void mglDebugMessageCallback(GLMContext ctx, GLDEBUGPROC callback, const void *userParam)
{
	// Debug callback - no-op if debug infrastructure not available
	(void)callback;
	(void)userParam;
}

void mglDebugMessageControl(GLMContext ctx, GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint *ids, GLboolean enabled)
{
	// Debug message control - no-op
	(void)source;
	(void)type;
	(void)severity;
	(void)count;
	(void)ids;
	(void)enabled;
}

void mglDebugMessageInsert(GLMContext ctx, GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *buf)
{
	// Insert debug message - no-op
	(void)source;
	(void)type;
	(void)id;
	(void)severity;
	(void)length;
	(void)buf;
}

GLuint  mglGetDebugMessageLog(GLMContext ctx, GLuint count, GLsizei bufSize, GLenum *sources, GLenum *types, GLuint *ids, GLenum *severities, GLsizei *lengths, GLchar *messageLog)
{
	// No debug messages stored
	(void)count;
	(void)bufSize;
	(void)sources;
	(void)types;
	(void)ids;
	(void)severities;
	(void)lengths;
	(void)messageLog;
	return 0;
}

GLenum  mglGetGraphicsResetStatus(GLMContext ctx)
{
	// No robust context support - always return no error
	return GL_NO_ERROR;
}

void mglGetObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei bufSize, GLsizei *length, GLchar *label)
{
	// No labels stored
	(void)identifier;
	(void)name;
	if (length) *length = 0;
	if (label && bufSize > 0) label[0] = '\0';
}

void mglGetObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei bufSize, GLsizei *length, GLchar *label)
{
	// No labels stored
	(void)ptr;
	if (length) *length = 0;
	if (label && bufSize > 0) label[0] = '\0';
}

void mglObjectLabel(GLMContext ctx, GLenum identifier, GLuint name, GLsizei length, const GLchar *label)
{
	// Object label - no-op
	(void)identifier;
	(void)name;
	(void)length;
	(void)label;
}

void mglObjectPtrLabel(GLMContext ctx, const void *ptr, GLsizei length, const GLchar *label)
{
	// Object ptr label - no-op
	(void)ptr;
	(void)length;
	(void)label;
}

void mglPopDebugGroup(GLMContext ctx)
{
	// Pop debug group - no-op
}

void mglPushDebugGroup(GLMContext ctx, GLenum source, GLuint id, GLsizei length, const GLchar *message)
{
	// Push debug group - no-op
	(void)source;
	(void)id;
	(void)length;
	(void)message;
}

void mglGetPointerv(GLMContext ctx, GLenum pname, void **params)
{
    // Unimplemented function
    assert(0);
}

#pragma mark - domain 'debug' hooks

void mgl_ext_init_debug(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_debug(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
