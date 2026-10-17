/*
 * program_ext.c
 * MGL — separable programs, pipelines, binaries, SPIR-V shaders, validation
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_program.h"

void mglActiveShaderProgram(GLMContext ctx, GLuint pipeline, GLuint program)
{
	// Set active program in pipeline - no-op for now
	(void)pipeline;
	(void)program;
}

void mglCreateProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
	for (GLsizei i = 0; i < n; i++)
	{
		mglGenProgramPipelines(ctx, 1, &pipelines[i]);
	}
}

GLuint  mglCreateShaderProgramv(GLMContext ctx, GLenum type, GLsizei count, const GLchar *const*strings)
{
	GLuint shader = mglCreateShader(ctx, type);
	if (!shader)
		return 0;
	
	mglShaderSource(ctx, shader, count, strings, NULL);
	mglCompileShader(ctx, shader);
	
	GLuint program = mglCreateProgram(ctx);
	if (!program) {
		mglDeleteShader(ctx, shader);
		return 0;
	}
	
	mglAttachShader(ctx, program, shader);
	mglLinkProgram(ctx, program);
	mglDeleteShader(ctx, shader);
	
	return program;
}

void mglGetProgramBinary(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary)
{
	// Program binary not supported
	(void)program; (void)bufSize; (void)binary;
	if (length) *length = 0;
	if (binaryFormat) *binaryFormat = 0;
}

void mglGetProgramPipelineInfoLog(GLMContext ctx, GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
	// Pipeline info log - return empty
	(void)pipeline;
	if (length) *length = 0;
	if (infoLog && bufSize > 0) infoLog[0] = '\0';
}

void mglGetProgramPipelineiv(GLMContext ctx, GLuint pipeline, GLenum pname, GLint *params)
{
	// Get program pipeline parameters - return 0
	(void)pipeline; (void)pname;
	if (params) *params = 0;
}

void mglGetShaderPrecisionFormat(GLMContext ctx, GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision)
{
	// Return shader precision format - full precision for all types
	(void)shadertype;
	(void)precisiontype;
	if (range) {
		range[0] = 127;
		range[1] = 127;
	}
	if (precision) {
		*precision = 23;
	}
}

void mglProgramBinary(GLMContext ctx, GLuint program, GLenum binaryFormat, const void *binary, GLsizei length)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramParameteri(GLMContext ctx, GLuint program, GLenum pname, GLint value)
{
	// TODO: Implement
	(void)ctx;
}

void mglReleaseShaderCompiler(GLMContext ctx)
{
	// No-op - shader compiler is always available
}

void mglShaderBinary(GLMContext ctx, GLsizei count, const GLuint *shaders, GLenum binaryFormat, const void *binary, GLsizei length)
{
	// TODO: Implement
	(void)ctx;
}

void mglSpecializeShader(GLMContext ctx, GLuint shader, const GLchar *pEntryPoint, GLuint numSpecializationConstants, const GLuint *pConstantIndex, const GLuint *pConstantValue)
{
	// TODO: Implement
	(void)ctx;
}

void mglValidateProgram(GLMContext ctx, GLuint program)
{
	// Program validation - no-op for now, programs are validated during linking
	(void)program;
}

void mglValidateProgramPipeline(GLMContext ctx, GLuint pipeline)
{
	// TODO: Implement
	(void)ctx;
}

#pragma mark - domain 'program' hooks

void mgl_ext_init_program(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_program(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
