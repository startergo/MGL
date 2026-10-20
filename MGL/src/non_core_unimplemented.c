/*
 * non_core_unimplemented.c
 * MGL — the imaging-subset queries of the compatibility profile
 *
 * The ARB_imaging commands that would define this state (glColorTable,
 * glConvolutionFilter2D, glHistogram, glMinmax, glSeparableFilter2D) are not
 * part of MGL's GL 4.6 API surface, so there is no color table / convolution
 * / histogram state to query: these robustness getters answer the spec error
 * for "no such filter is defined".  Everything else that used to live in this
 * file is implemented by the compat domain (MGL/src/compat_*.c).
 */

#include "mgl.h"
#include "glm_context.h"


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
