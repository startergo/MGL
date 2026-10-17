/*
 * queries.c
 * MGL — query objects, timers, conditional rendering
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_queries.h"

void mglBeginConditionalRender(GLMContext ctx, GLuint id, GLenum mode)
{
	// Conditional render - no-op, always render
	(void)id;
	(void)mode;
}

void mglBeginQuery(GLMContext ctx, GLenum target, GLuint id)
{
	// Query - no-op, basic stub
	(void)target;
	(void)id;
}

void mglBeginQueryIndexed(GLMContext ctx, GLenum target, GLuint index, GLuint id)
{
	// Indexed query - no-op, basic stub
	(void)target;
	(void)index;
	(void)id;
}

void mglEndConditionalRender(GLMContext ctx)
{
	// End conditional render - no-op
	(void)ctx;
}

void mglEndQuery(GLMContext ctx, GLenum target)
{
	// End query - no-op
	(void)target;
}

void mglEndQueryIndexed(GLMContext ctx, GLenum target, GLuint index)
{
	// End indexed query - no-op
	(void)target;
	(void)index;
}

void mglGetQueryBufferObjecti64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryBufferObjectiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryBufferObjectui64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryBufferObjectuiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryIndexediv(GLMContext ctx, GLenum target, GLuint index, GLenum pname, GLint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryObjecti64v(GLMContext ctx, GLuint id, GLenum pname, GLint64 *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryObjectiv(GLMContext ctx, GLuint id, GLenum pname, GLint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryObjectui64v(GLMContext ctx, GLuint id, GLenum pname, GLuint64 *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryObjectuiv(GLMContext ctx, GLuint id, GLenum pname, GLuint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetQueryiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
	// TODO: Implement
	(void)ctx;
}

GLboolean mglIsQuery(GLMContext ctx, GLuint id)
{
	// TODO: Implement
	(void)ctx;
	return GL_FALSE;
}

void mglQueryCounter(GLMContext ctx, GLuint id, GLenum target)
{
	// TODO: Implement
	(void)ctx;
}

void mglCreateQueries(GLMContext ctx, GLenum target, GLsizei n, GLuint *ids)
{
	for (GLsizei i = 0; i < n; i++)
	{
		mglGenQueries(ctx, 1, &ids[i]);
	}
	(void)target;
}

void mglDeleteQueries(GLMContext ctx, GLsizei n, const GLuint *ids)
{
	// Query deletion - no-op, basic stub
	(void)n;
	(void)ids;
}

void mglGenQueries(GLMContext ctx, GLsizei n, GLuint *ids)
{
	// Generate query IDs - simple sequential IDs
	for (GLsizei i = 0; i < n; i++)
		ids[i] = i + 1;
}

#pragma mark - domain 'queries' hooks

void mgl_ext_init_queries(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_queries(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
