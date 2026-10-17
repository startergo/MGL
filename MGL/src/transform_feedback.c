/*
 * transform_feedback.c
 * MGL — transform feedback objects and draws
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_xfb.h"

TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name);
TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name);

void mglBeginTransformFeedback(GLMContext ctx, GLenum primitiveMode)
{
	if (!STATE(transform_feedback))
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	if (STATE(transform_feedback)->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->active = GL_TRUE;
	STATE(transform_feedback)->paused = GL_FALSE;
	STATE(transform_feedback)->primitive_mode = primitiveMode;
}

void mglBindTransformFeedback(GLMContext ctx, GLenum target, GLuint id)
{
    if (target != GL_TRANSFORM_FEEDBACK)
    {
        STATE(error) = GL_INVALID_ENUM;
        return;
    }

    // Can't bind if current transform feedback is active and not paused
    if (STATE(transform_feedback) && 
        STATE(transform_feedback)->active && 
        !STATE(transform_feedback)->paused)
    {
        STATE(error) = GL_INVALID_OPERATION;
        return;
    }

    if (id == 0)
    {
        // Unbind transform feedback
        STATE(transform_feedback) = NULL;
    }
    else
    {
        TransformFeedback *ptr = getTransformFeedback(ctx, id);
        if (ptr)
        {
            ptr->target = target;
            STATE(transform_feedback) = ptr;
        }
    }
}

void mglCreateTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
	for (GLsizei i = 0; i < n; i++)
	{
		mglGenTransformFeedbacks(ctx, 1, &ids[i]);
	}
}

void mglDeleteTransformFeedbacks(GLMContext ctx, GLsizei n, const GLuint *ids)
{
    for (GLsizei i = 0; i < n; i++)
    {
        if (ids[i] == 0)
            continue;
            
        TransformFeedback *ptr = findTransformFeedback(ctx, ids[i]);
        if (!ptr)
            continue;
            
        // Can't delete if active
        if (ptr->active)
        {
            STATE(error) = GL_INVALID_OPERATION;
            continue;
        }
            
        // If deleting currently bound transform feedback, unbind it
        if (STATE(transform_feedback) && STATE(transform_feedback)->name == ids[i])
        {
            STATE(transform_feedback) = NULL;
        }
        
        // Remove from hash table and free
        deleteHashElement(&STATE(transform_feedback_table), ids[i]);
        free(ptr);
    }
}

void mglDrawTransformFeedback(GLMContext ctx, GLenum mode, GLuint id)
{
	// Draw from transform feedback - no-op for now
	(void)mode;
	(void)id;
}

void mglDrawTransformFeedbackInstanced(GLMContext ctx, GLenum mode, GLuint id, GLsizei instancecount)
{
	// Draw from transform feedback instanced - no-op for now
	(void)mode;
	(void)id;
	(void)instancecount;
}

void mglDrawTransformFeedbackStream(GLMContext ctx, GLenum mode, GLuint id, GLuint stream)
{
	// Draw from transform feedback stream - no-op for now
	(void)mode;
	(void)id;
	(void)stream;
}

void mglDrawTransformFeedbackStreamInstanced(GLMContext ctx, GLenum mode, GLuint id, GLuint stream, GLsizei instancecount)
{
	// Draw from transform feedback stream instanced - no-op for now
	(void)mode;
	(void)id;
	(void)stream;
	(void)instancecount;
}

void mglEndTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->active = GL_FALSE;
	STATE(transform_feedback)->paused = GL_FALSE;
}

void mglGenTransformFeedbacks(GLMContext ctx, GLsizei n, GLuint *ids)
{
    for (GLsizei i = 0; i < n; i++)
    {
        ids[i] = getNewName(&STATE(transform_feedback_table));
        getTransformFeedback(ctx, ids[i]);
    }
}

void mglGetTransformFeedbackVarying(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLsizei *size, GLenum *type, GLchar *name)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetTransformFeedbacki64_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint64 *param)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetTransformFeedbacki_v(GLMContext ctx, GLuint xfb, GLenum pname, GLuint index, GLint *param)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetTransformFeedbackiv(GLMContext ctx, GLuint xfb, GLenum pname, GLint *param)
{
	// TODO: Implement
	(void)ctx;
}

GLboolean mglIsTransformFeedback(GLMContext ctx, GLuint id)
{
	TransformFeedback *ptr = findTransformFeedback(ctx, id);
	return ptr ? GL_TRUE : GL_FALSE;
}

void mglPauseTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active || STATE(transform_feedback)->paused)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->paused = GL_TRUE;
}

void mglResumeTransformFeedback(GLMContext ctx)
{
	if (!STATE(transform_feedback) || !STATE(transform_feedback)->active || !STATE(transform_feedback)->paused)
	{
		STATE(error) = GL_INVALID_OPERATION;
		return;
	}

	STATE(transform_feedback)->paused = GL_FALSE;
}

void mglTransformFeedbackBufferBase(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer)
{
	// TODO: Implement
	(void)ctx;
}

void mglTransformFeedbackBufferRange(GLMContext ctx, GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size)
{
	// TODO: Implement
	(void)ctx;
}

void mglTransformFeedbackVaryings(GLMContext ctx, GLuint program, GLsizei count, const GLchar *const*varyings, GLenum bufferMode)
{
	// TODO: Implement
	(void)ctx;
}

#pragma mark - domain 'xfb' hooks

void mgl_ext_init_xfb(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_xfb(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
