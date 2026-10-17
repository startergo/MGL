/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * non_core_unimplemented.c
 * MGL
 *
 */

#include "mgl.h"
#include "glm_context.h"

void mglArrayElement(GLMContext ctx, GLint i)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglBegin(GLMContext ctx, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNewList(GLMContext ctx, GLuint list, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEndList(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCallList(GLMContext ctx, GLuint list)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCallLists(GLMContext ctx, GLsizei n, GLenum type, const void *lists)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglDeleteLists(GLMContext ctx, GLuint list, GLsizei range)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

GLuint mglGenLists(GLMContext ctx, GLsizei range)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
    return 0;
}

void mglListBase(GLMContext ctx, GLuint base)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglBitmap(GLMContext ctx, GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte *bitmap)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3bv(GLMContext ctx, const GLbyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3i(GLMContext ctx, GLint red, GLint green, GLint blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3s(GLMContext ctx, GLshort red, GLshort green, GLshort blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3ubv(GLMContext ctx, const GLubyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3uiv(GLMContext ctx, const GLuint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3us(GLMContext ctx, GLushort red, GLushort green, GLushort blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor3usv(GLMContext ctx, const GLushort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue, GLbyte alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4bv(GLMContext ctx, const GLbyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue, GLdouble alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4i(GLMContext ctx, GLint red, GLint green, GLint blue, GLint alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4s(GLMContext ctx, GLshort red, GLshort green, GLshort blue, GLshort alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue, GLubyte alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4ubv(GLMContext ctx, const GLubyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue, GLuint alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4uiv(GLMContext ctx, const GLuint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4us(GLMContext ctx, GLushort red, GLushort green, GLushort blue, GLushort alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColor4usv(GLMContext ctx, const GLushort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEnd(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEdgeFlag(GLMContext ctx, GLboolean flag)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEdgeFlagv(GLMContext ctx, const GLboolean *flag)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexd(GLMContext ctx, GLdouble c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexdv(GLMContext ctx, const GLdouble *c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexf(GLMContext ctx, GLfloat c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexfv(GLMContext ctx, const GLfloat *c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexi(GLMContext ctx, GLint c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexiv(GLMContext ctx, const GLint *c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexs(GLMContext ctx, GLshort c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexsv(GLMContext ctx, const GLshort *c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3b(GLMContext ctx, GLbyte nx, GLbyte ny, GLbyte nz)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3bv(GLMContext ctx, const GLbyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3d(GLMContext ctx, GLdouble nx, GLdouble ny, GLdouble nz)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3f(GLMContext ctx, GLfloat nx, GLfloat ny, GLfloat nz)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3i(GLMContext ctx, GLint nx, GLint ny, GLint nz)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3s(GLMContext ctx, GLshort nx, GLshort ny, GLshort nz)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormal3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2i(GLMContext ctx, GLint x, GLint y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2s(GLMContext ctx, GLshort x, GLshort y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos2sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4i(GLMContext ctx, GLint x, GLint y, GLint z, GLint w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4s(GLMContext ctx, GLshort x, GLshort y, GLshort z, GLshort w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRasterPos4sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectd(GLMContext ctx, GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectdv(GLMContext ctx, const GLdouble *v1, const GLdouble *v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectf(GLMContext ctx, GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectfv(GLMContext ctx, const GLfloat *v1, const GLfloat *v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRecti(GLMContext ctx, GLint x1, GLint y1, GLint x2, GLint y2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectiv(GLMContext ctx, const GLint *v1, const GLint *v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRects(GLMContext ctx, GLshort x1, GLshort y1, GLshort x2, GLshort y2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRectsv(GLMContext ctx, const GLshort *v1, const GLshort *v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1d(GLMContext ctx, GLdouble s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1f(GLMContext ctx, GLfloat s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1i(GLMContext ctx, GLint s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1s(GLMContext ctx, GLshort s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord1sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2d(GLMContext ctx, GLdouble s, GLdouble t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2f(GLMContext ctx, GLfloat s, GLfloat t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2i(GLMContext ctx, GLint s, GLint t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2s(GLMContext ctx, GLshort s, GLshort t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord2sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3d(GLMContext ctx, GLdouble s, GLdouble t, GLdouble r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3f(GLMContext ctx, GLfloat s, GLfloat t, GLfloat r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3i(GLMContext ctx, GLint s, GLint t, GLint r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3s(GLMContext ctx, GLshort s, GLshort t, GLshort r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4d(GLMContext ctx, GLdouble s, GLdouble t, GLdouble r, GLdouble q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4f(GLMContext ctx, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4i(GLMContext ctx, GLint s, GLint t, GLint r, GLint q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4s(GLMContext ctx, GLshort s, GLshort t, GLshort r, GLshort q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoord4sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2i(GLMContext ctx, GLint x, GLint y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2s(GLMContext ctx, GLshort x, GLshort y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex2sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4i(GLMContext ctx, GLint x, GLint y, GLint z, GLint w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4s(GLMContext ctx, GLshort x, GLshort y, GLshort z, GLshort w)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertex4sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglClipPlane(GLMContext ctx, GLenum plane, const GLdouble *equation)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColorMaterial(GLMContext ctx, GLenum face, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogf(GLMContext ctx, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogi(GLMContext ctx, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogiv(GLMContext ctx, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightf(GLMContext ctx, GLenum light, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightfv(GLMContext ctx, GLenum light, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLighti(GLMContext ctx, GLenum light, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightiv(GLMContext ctx, GLenum light, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightModelf(GLMContext ctx, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightModelfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightModeli(GLMContext ctx, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLightModeliv(GLMContext ctx, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLineStipple(GLMContext ctx, GLint factor, GLushort pattern)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMaterialf(GLMContext ctx, GLenum face, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMaterialfv(GLMContext ctx, GLenum face, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMateriali(GLMContext ctx, GLenum face, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMaterialiv(GLMContext ctx, GLenum face, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPolygonStipple(GLMContext ctx, const GLubyte *mask)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglShadeModel(GLMContext ctx, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexEnvf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexEnvi(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexEnviv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGend(GLMContext ctx, GLenum coord, GLenum pname, GLdouble param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGendv(GLMContext ctx, GLenum coord, GLenum pname, const GLdouble *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGenf(GLMContext ctx, GLenum coord, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, const GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGeni(GLMContext ctx, GLenum coord, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, const GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFeedbackBuffer(GLMContext ctx, GLsizei size, GLenum type, GLfloat *buffer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSelectBuffer(GLMContext ctx, GLsizei size, GLuint *buffer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

GLint  mglRenderMode(GLMContext ctx, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
    return 0;
}

void mglPixelMapfv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLfloat *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPixelMapuiv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLuint *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPixelMapusv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLushort *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglCopyPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum type)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglDrawPixels(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetClipPlane(GLMContext ctx, GLenum plane, GLdouble *equation)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetLightfv(GLMContext ctx, GLenum light, GLenum pname, GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetLightiv(GLMContext ctx, GLenum light, GLenum pname, GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetMapdv(GLMContext ctx, GLenum target, GLenum query, GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetMapfv(GLMContext ctx, GLenum target, GLenum query, GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetMapiv(GLMContext ctx, GLenum target, GLenum query, GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetMaterialfv(GLMContext ctx, GLenum face, GLenum pname, GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetMaterialiv(GLMContext ctx, GLenum face, GLenum pname, GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetPixelMapfv(GLMContext ctx, GLenum map, GLfloat *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetPixelMapuiv(GLMContext ctx, GLenum map, GLuint *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetPixelMapusv(GLMContext ctx, GLenum map, GLushort *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetPolygonStipple(GLMContext ctx, GLubyte *mask)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTexEnviv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTexGendv(GLMContext ctx, GLenum coord, GLenum pname, GLdouble *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, GLfloat *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, GLint *params)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

GLboolean mglIsList(GLMContext ctx, GLuint list)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
    return GL_FALSE;
}

void mglFrustum(GLMContext ctx, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadIdentity(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadMatrixf(GLMContext ctx, const GLfloat *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadMatrixd(GLMContext ctx, const GLdouble *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMatrixMode(GLMContext ctx, GLenum mode)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultMatrixf(GLMContext ctx, const GLfloat *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultMatrixd(GLMContext ctx, const GLdouble *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglOrtho(GLMContext ctx, GLdouble left, GLdouble right, GLdouble bottom, GLdouble top, GLdouble zNear, GLdouble zFar)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPopMatrix(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPushMatrix(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRotated(GLMContext ctx, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglRotatef(GLMContext ctx, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglScaled(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglScalef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTranslated(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTranslatef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}


void mglInitNames(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadName(GLMContext ctx, GLuint name)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPassThrough(GLMContext ctx, GLfloat token)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPopName(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPushName(GLMContext ctx, GLuint name)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglClearAccum(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglClearIndex(GLMContext ctx, GLfloat c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexMask(GLMContext ctx, GLuint mask)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglAccum(GLMContext ctx, GLenum op, GLfloat value)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPopAttrib(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPushAttrib(GLMContext ctx, GLbitfield mask)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMap1d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMap1f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMap2d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder, GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMap2f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder, GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMapGrid1d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMapGrid1f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMapGrid2d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMapGrid2f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord1d(GLMContext ctx, GLdouble u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord1dv(GLMContext ctx, const GLdouble *u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord1f(GLMContext ctx, GLfloat u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord1fv(GLMContext ctx, const GLfloat *u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord2d(GLMContext ctx, GLdouble u, GLdouble v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord2dv(GLMContext ctx, const GLdouble *u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord2f(GLMContext ctx, GLfloat u, GLfloat v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalCoord2fv(GLMContext ctx, const GLfloat *u)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalMesh1(GLMContext ctx, GLenum mode, GLint i1, GLint i2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalPoint1(GLMContext ctx, GLint i)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalMesh2(GLMContext ctx, GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEvalPoint2(GLMContext ctx, GLint i, GLint j)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglAlphaFunc(GLMContext ctx, GLenum func, GLfloat ref)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPixelZoom(GLMContext ctx, GLfloat xfactor, GLfloat yfactor)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPixelTransferf(GLMContext ctx, GLenum pname, GLfloat param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPixelTransferi(GLMContext ctx, GLenum pname, GLint param)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnPolygonStipple(GLMContext ctx, GLsizei bufSize, GLubyte *pattern)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnColorTable(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *table)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnConvolutionFilter(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei bufSize, void *image)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnSeparableFilter(GLMContext ctx, GLenum target, GLenum format, GLenum type, GLsizei rowBufSize, void *row, GLsizei columnBufSize, void *column, void *span)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnHistogram(GLMContext ctx, GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglGetnMinmax(GLMContext ctx, GLenum target, GLboolean reset, GLenum format, GLenum type, GLsizei bufSize, void *values)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglDisableClientState(GLMContext ctx, GLenum array)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEdgeFlagPointer(GLMContext ctx, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglEnableClientState(GLMContext ctx, GLenum array)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglInterleavedArrays(GLMContext ctx, GLenum format, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglNormalPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglVertexPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogCoordf(GLMContext ctx, GLfloat coord)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogCoordfv(GLMContext ctx, const GLfloat *coord)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogCoordd(GLMContext ctx, GLdouble coord)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogCoorddv(GLMContext ctx, const GLdouble *coord)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglFogCoordPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3b(GLMContext ctx, GLbyte red, GLbyte green, GLbyte blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3bv(GLMContext ctx, const GLbyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3d(GLMContext ctx, GLdouble red, GLdouble green, GLdouble blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3f(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3i(GLMContext ctx, GLint red, GLint green, GLint blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3s(GLMContext ctx, GLshort red, GLshort green, GLshort blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3ub(GLMContext ctx, GLubyte red, GLubyte green, GLubyte blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3ubv(GLMContext ctx, const GLubyte *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3ui(GLMContext ctx, GLuint red, GLuint green, GLuint blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3uiv(GLMContext ctx, const GLuint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3us(GLMContext ctx, GLushort red, GLushort green, GLushort blue)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColor3usv(GLMContext ctx, const GLushort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglSecondaryColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2d(GLMContext ctx, GLdouble x, GLdouble y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2f(GLMContext ctx, GLfloat x, GLfloat y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2i(GLMContext ctx, GLint x, GLint y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2s(GLMContext ctx, GLshort x, GLshort y)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos2sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3dv(GLMContext ctx, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3fv(GLMContext ctx, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3i(GLMContext ctx, GLint x, GLint y, GLint z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3iv(GLMContext ctx, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglWindowPos3sv(GLMContext ctx, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP1ui(GLMContext ctx, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP1uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP2ui(GLMContext ctx, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP2uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP3ui(GLMContext ctx, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP3uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP4ui(GLMContext ctx, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglTexCoordP4uiv(GLMContext ctx, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP1ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP1uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP2ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP2uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP3ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP3uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP4ui(GLMContext ctx, GLenum texture, GLenum type, GLuint coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoordP4uiv(GLMContext ctx, GLenum texture, GLenum type, const GLuint *coords)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1d(GLMContext ctx, GLenum target, GLdouble s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1f(GLMContext ctx, GLenum target, GLfloat s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1i(GLMContext ctx, GLenum target, GLint s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1iv(GLMContext ctx, GLenum target, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1s(GLMContext ctx, GLenum target, GLshort s)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord1sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2i(GLMContext ctx, GLenum target, GLint s, GLint t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2iv(GLMContext ctx, GLenum target, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2s(GLMContext ctx, GLenum target, GLshort s, GLshort t)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord2sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t, GLdouble r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t, GLfloat r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3i(GLMContext ctx, GLenum target, GLint s, GLint t, GLint r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3iv(GLMContext ctx, GLenum target, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3s(GLMContext ctx, GLenum target, GLshort s, GLshort t, GLshort r)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord3sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4d(GLMContext ctx, GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4dv(GLMContext ctx, GLenum target, const GLdouble *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4f(GLMContext ctx, GLenum target, GLfloat s, GLfloat t, GLfloat r, GLfloat q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4fv(GLMContext ctx, GLenum target, const GLfloat *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4i(GLMContext ctx, GLenum target, GLint s, GLint t, GLint r, GLint q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4iv(GLMContext ctx, GLenum target, const GLint *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4s(GLMContext ctx, GLenum target, GLshort s, GLshort t, GLshort r, GLshort q)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultiTexCoord4sv(GLMContext ctx, GLenum target, const GLshort *v)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglLoadTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglMultTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexub(GLMContext ctx, GLubyte c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglIndexubv(GLMContext ctx, const GLubyte *c)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPopClientAttrib(GLMContext ctx)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

void mglPushClientAttrib(GLMContext ctx, GLbitfield mask)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

GLboolean mglAreTexturesResident(GLMContext ctx, GLsizei n, const GLuint *textures, GLboolean *residences)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
    return GL_FALSE;
}

void mglPrioritizeTextures(GLMContext ctx, GLsizei n, const GLuint *textures, const GLfloat *priorities)
{
    /* compatibility-profile only; MGL is a core-profile context */
    ERROR_RETURN(GL_INVALID_OPERATION);
}


