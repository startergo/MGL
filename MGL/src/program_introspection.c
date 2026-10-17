/*
 * program_introspection.c
 * MGL — program interface queries, ProgramUniform*, subroutines
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_introspect.h"

void mglBindFragDataLocation(GLMContext ctx, GLuint program, GLuint color, const GLchar *name)
{
	// Bind fragment output location - no-op, use automatic assignment
	(void)program;
	(void)color;
	(void)name;
}

void mglBindFragDataLocationIndexed(GLMContext ctx, GLuint program, GLuint colorNumber, GLuint index, const GLchar *name)
{
	// Bind indexed fragment output location - no-op
	(void)program;
	(void)colorNumber;
	(void)index;
	(void)name;
}

void mglGetActiveAtomicCounterBufferiv(GLMContext ctx, GLuint program, GLuint bufferIndex, GLenum pname, GLint *params)
{
	// Atomic counter buffers - return 0
	(void)program; (void)bufferIndex; (void)pname;
	if (params) *params = 0;
}

void mglGetActiveSubroutineName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	// Subroutines - return empty string
	(void)program; (void)shadertype; (void)index; (void)bufSize;
	if (length) *length = 0;
	if (name && bufSize > 0) name[0] = '\0';
}

void mglGetActiveSubroutineUniformName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	// Subroutine uniforms - return empty string
	(void)program; (void)shadertype; (void)index; (void)bufSize;
	if (length) *length = 0;
	if (name && bufSize > 0) name[0] = '\0';
}

void mglGetActiveSubroutineUniformiv(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint *values)
{
	// Subroutine uniform parameters - return 0
	(void)program; (void)shadertype; (void)index; (void)pname;
	if (values) *values = 0;
}

GLint  mglGetFragDataIndex(GLMContext ctx, GLuint program, const GLchar *name)
{
	// Return fragment output index
	(void)program;
	(void)name;
	return 0;
}

GLint  mglGetFragDataLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
	// Return fragment output location - default to 0
	(void)program;
	(void)name;
	return 0;
}

void mglGetProgramInterfaceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLenum pname, GLint *params)
{
	// Program interface query - return 0
	(void)program; (void)programInterface; (void)pname;
	if (params) *params = 0;
}

GLuint  mglGetProgramResourceIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	// Program resource index - return 0 (not found)
	(void)program; (void)programInterface; (void)name;
	return 0;
}

GLint  mglGetProgramResourceLocation(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	// Program resource location - return -1 (not found)
	(void)program; (void)programInterface; (void)name;
	return -1;
}

GLint  mglGetProgramResourceLocationIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
	// Program resource location index - return -1 (not found)
	(void)program; (void)programInterface; (void)name;
	return -1;
}

void mglGetProgramResourceName(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetProgramResourceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum *props, GLsizei count, GLsizei *length, GLint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetProgramStageiv(GLMContext ctx, GLuint program, GLenum shadertype, GLenum pname, GLint *values)
{
	// TODO: Implement
	(void)ctx;
}

GLuint mglGetSubroutineIndex(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
	// TODO: Implement
	(void)ctx;
	return 0;
}

GLint mglGetSubroutineUniformLocation(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
	// TODO: Implement
	(void)ctx;
	return 0;
}

void mglGetUniformSubroutineuiv(GLMContext ctx, GLenum shadertype, GLint location, GLuint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetUniformdv(GLMContext ctx, GLuint program, GLint location, GLdouble *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetUniformuiv(GLMContext ctx, GLuint program, GLint location, GLuint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnUniformdv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLdouble *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnUniformfv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLfloat *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnUniformiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglGetnUniformuiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLuint *params)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1d(GLMContext ctx, GLuint program, GLint location, GLdouble v0)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1f(GLMContext ctx, GLuint program, GLint location, GLfloat v0)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1i(GLMContext ctx, GLuint program, GLint location, GLint v0)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1iv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1ui(GLMContext ctx, GLuint program, GLint location, GLuint v0)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform1uiv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2iv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform2uiv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3iv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform3uiv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4iv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniform4uiv(GLMContext ctx, GLuint program, GLint location, GLsizei count, const GLuint *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2x3dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2x3fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2x4dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix2x4fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3x2dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3x2fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3x4dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix3x4fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4x2dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4x2fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4x3dv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLdouble *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglProgramUniformMatrix4x3fv(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	// TODO: Implement
	(void)ctx;
}

void mglShaderStorageBlockBinding(GLMContext ctx, GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding)
{
	// TODO: Implement
	(void)ctx;
}

void mglUniformSubroutinesuiv(GLMContext ctx, GLenum shadertype, GLsizei count, const GLuint *indices)
{
	// TODO: Implement
	(void)ctx;
}

void mglBindAttribLocation(GLMContext ctx, GLuint program, GLuint index, const GLchar *name)
{
    // Unimplemented function
    // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
}

void mglGetActiveAttrib(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    // Unimplemented function
    // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
}

void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    // Unimplemented function
    // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
}

void mglGetAttachedShaders(GLMContext ctx, GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders)
{
    // Unimplemented function
    // CRITICAL FIX: Handle error gracefully instead of crashing
        fprintf(stderr, "MGL ERROR: Critical error in program.c at line %d\n", __LINE__);
        STATE(error) = GL_INVALID_OPERATION;
}

#pragma mark - domain 'introspect' hooks

void mgl_ext_init_introspect(GLMContext ctx)
{
    (void)ctx;
}

bool mgl_ext_get_introspect(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx; (void)pname; (void)type; (void)data;
    return false;
}
