/*
 * framebuffer_ext.c
 * MGL — ClearBuffer*, robust reads, multisample queries
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_fb.h"

void mglClearBufferiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLint *value)
{
	// Clear buffer - use standard clear functions
	(void)buffer;
	(void)drawbuffer;
	(void)value;
}

void mglClearBufferuiv(GLMContext ctx, GLenum buffer, GLint drawbuffer, const GLuint *value)
{
	// Clear buffer - use standard clear functions
	(void)buffer;
	(void)drawbuffer;
	(void)value;
}

void mglGetMultisamplefv(GLMContext ctx, GLenum pname, GLuint index, GLfloat *val)
{
	// Get sample positions for multisample rendering
	if (pname != GL_SAMPLE_POSITION) {
		STATE(error) = GL_INVALID_ENUM;
		return;
	}
	
	// Metal uses fixed sample positions based on sample count
	// Return reasonable default positions (center for simplicity)
	val[0] = 0.5f;
	val[1] = 0.5f;
}

void mglReadnPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum format, GLenum type, GLsizei bufSize, void *data)
{
	// TODO: Implement
	(void)ctx;
}

#pragma mark - domain 'fb' hooks

void mgl_ext_init_fb(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_fb(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
