/*
 * draw_ext.c
 * MGL — GL 4.6 indirect-count draws
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_buffers.h"

void mglMultiDrawArraysIndirectCount(GLMContext ctx, GLenum mode, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
	// TODO: Implement
	(void)ctx;
}

void mglMultiDrawElementsIndirectCount(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride)
{
	// TODO: Implement
	(void)ctx;
}

#pragma mark - domain 'buffers' hooks

void mgl_ext_init_buffers(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_buffers(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
