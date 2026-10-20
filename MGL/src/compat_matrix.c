/*
 * compat_matrix.c — matrix stacks of the compatibility profile (compat domain).
 *
 * glMatrixMode / glLoadMatrix / glMultMatrix / glPushMatrix / glOrtho /
 * glFrustum / glRotate / glScale / glTranslate and the transpose variants.
 * Column-major GLdouble math; the shader-visible copies are kept in float.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"

#define CS() (ctx->state.ext_compat)

#pragma mark matrix math (column major, GL convention)

void mglc_matrix_identity(GLdouble *m)
{
    memset(m, 0, 16 * sizeof(GLdouble));
    m[0] = m[5] = m[10] = m[15] = 1.0;
}

void mglc_matrix_transpose(GLdouble *out, const GLdouble *m)
{
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            out[c * 4 + r] = m[r * 4 + c];
}

/* out = a * b (GL "current * m") */
void mglc_matrix_mul(GLdouble *out, const GLdouble *a, const GLdouble *b)
{
    GLdouble t[16];

    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
        {
            GLdouble s = 0;
            for (int k = 0; k < 4; k++)
                s += a[k * 4 + r] * b[c * 4 + k];
            t[c * 4 + r] = s;
        }

    memcpy(out, t, 16 * sizeof(GLdouble));
}

/* general 4x4 inverse (Cramer); returns false for singular matrices */
bool mglc_matrix_invert(GLdouble *out, const GLdouble *m)
{
    double a[16], b[16], det;
    int i;

    /* cofactor matrix */
    a[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    a[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    a[2]  =  m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    a[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    a[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    a[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    a[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    a[7]  =  m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    a[8]  =  m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    a[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    a[10]=  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    a[11]= -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    a[12]= -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    a[13]=  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    a[14]= -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    a[15]=  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    det = m[0]*a[0] + m[1]*a[4] + m[2]*a[8] + m[3]*a[12];
    if (det == 0.0 || !isfinite(det))
        return false;

    det = 1.0 / det;
    for (i = 0; i < 16; i++)
        b[i] = a[i] * det;

    /* transpose of cofactor/det */
    mglc_matrix_transpose(out, b);
    return true;
}

void mglc_matrix_frustum(GLdouble *m, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f)
{
    mglc_matrix_identity(m);
    m[0] = 2.0 * n / (r - l);
    m[5] = 2.0 * n / (t - b);
    m[8] = (r + l) / (r - l);
    m[9] = (t + b) / (t - b);
    m[10] = -(f + n) / (f - n);
    m[11] = -1.0;
    m[14] = -2.0 * f * n / (f - n);
    m[15] = 0.0;
}

void mglc_matrix_ortho(GLdouble *m, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f)
{
    mglc_matrix_identity(m);
    m[0] = 2.0 / (r - l);
    m[5] = 2.0 / (t - b);
    m[10] = -2.0 / (f - n);
    m[12] = -(r + l) / (r - l);
    m[13] = -(t + b) / (t - b);
    m[14] = -(f + n) / (f - n);
}

void mglc_matrix_rotate(GLdouble *m, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{
    GLdouble len = sqrt(x * x + y * y + z * z);
    GLdouble c, s, t;

    if (len == 0.0)
    {
        mglc_matrix_identity(m);
        return;
    }

    x /= len; y /= len; z /= len;
    angle = angle * M_PI / 180.0;
    c = cos(angle); s = sin(angle); t = 1.0 - c;

    mglc_matrix_identity(m);
    m[0] = t * x * x + c;
    m[1] = t * x * y + s * z;
    m[2] = t * x * z - s * y;
    m[4] = t * x * y - s * z;
    m[5] = t * y * y + c;
    m[6] = t * y * z + s * x;
    m[8] = t * x * z + s * y;
    m[9] = t * y * z - s * x;
    m[10] = t * z * z + c;
}

#pragma mark stack access

MGLCMatrixStack *mgl_compat_matrix_stack(GLMContext ctx, GLenum mode, int *texunit)
{
    struct MGLExtState_compat *s = CS();

    if (texunit)
        *texunit = -1;

    switch (mode)
    {
        case GL_MODELVIEW: return &s->modelview;
        case GL_PROJECTION: return &s->projection;
        case GL_COLOR: return &s->color;
        case GL_TEXTURE:
        {
            GLuint unit = STATE(active_texture);
            if (unit >= MGLC_MAX_COORDS)
                unit = 0;
            if (texunit) *texunit = (int)unit;
            return &s->texture[unit];
        }
    }

    return NULL;
}

void mgl_compat_get_matrix(GLMContext ctx, GLenum pname, GLdouble *out)
{
    GLenum mode;

    switch (pname)
    {
        case GL_MODELVIEW_MATRIX: mode = GL_MODELVIEW; break;
        case GL_PROJECTION_MATRIX: mode = GL_PROJECTION; break;
        case GL_COLOR_MATRIX: mode = GL_COLOR; break;
        case GL_TEXTURE_MATRIX: mode = GL_TEXTURE; break;
        default: mode = pname; break;
    }

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, mode, NULL);

    if (!st)
    {
        mglc_matrix_identity(out);
        return;
    }

    memcpy(out, st->stack[st->depth].m, 16 * sizeof(GLdouble));
}

/* cache float copies for the shaders (and bump the state serial) */
static void matrix_changed(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    for (int i = 0; i < 16; i++)
    {
        s->modelview_f[i] = (GLfloat)s->modelview.stack[s->modelview.depth].m[i];
        s->projection_f[i] = (GLfloat)s->projection.stack[s->projection.depth].m[i];
    }
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        for (int i = 0; i < 16; i++)
            s->texture_f[u][i] = (GLfloat)s->texture[u].stack[s->texture[u].depth].m[i];

    memcpy(s->modelview_current.m, s->modelview.stack[s->modelview.depth].m, sizeof(MGLCMat));

    mgl_compat_dirty(ctx);
}

#pragma mark entry points

void mglMatrixMode(GLMContext ctx, GLenum mode)
{    MGLC_COMPAT_GUARD(ctx);

    if (!mgl_compat_matrix_stack(ctx, mode, NULL))
        ERROR_RETURN(GL_INVALID_ENUM);

    CS()->matrix_mode = mode;
}

void mglLoadIdentity(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglc_matrix_identity(st->stack[st->depth].m);
    matrix_changed(ctx);
}

void mglLoadMatrixf(GLMContext ctx, const GLfloat *m)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st || !m)
        ERROR_RETURN(GL_INVALID_OPERATION);

    for (int i = 0; i < 16; i++)
        st->stack[st->depth].m[i] = (GLdouble)m[i];

    matrix_changed(ctx);
}

void mglLoadMatrixd(GLMContext ctx, const GLdouble *m)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st || !m)
        ERROR_RETURN(GL_INVALID_OPERATION);

    memcpy(st->stack[st->depth].m, m, 16 * sizeof(GLdouble));
    matrix_changed(ctx);
}

void mglMultMatrixf(GLMContext ctx, const GLfloat *m)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble d[16];

    if (!st || !m)
        ERROR_RETURN(GL_INVALID_OPERATION);

    for (int i = 0; i < 16; i++)
        d[i] = (GLdouble)m[i];

    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, d);
    matrix_changed(ctx);
}

void mglMultMatrixd(GLMContext ctx, const GLdouble *m)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st || !m)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglLoadTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{    MGLC_COMPAT_GUARD(ctx);

    GLdouble d[16], t[16];

    for (int i = 0; i < 16; i++)
        d[i] = (GLdouble)m[i];

    mglc_matrix_transpose(t, d);
    mglLoadMatrixd(ctx, t);
}

void mglLoadTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{    MGLC_COMPAT_GUARD(ctx);

    GLdouble t[16];
    mglc_matrix_transpose(t, m);
    mglLoadMatrixd(ctx, t);
}

void mglMultTransposeMatrixf(GLMContext ctx, const GLfloat *m)
{    MGLC_COMPAT_GUARD(ctx);

    GLdouble d[16], t[16];

    for (int i = 0; i < 16; i++)
        d[i] = (GLdouble)m[i];

    mglc_matrix_transpose(t, d);
    mglMultMatrixd(ctx, t);
}

void mglMultTransposeMatrixd(GLMContext ctx, const GLdouble *m)
{    MGLC_COMPAT_GUARD(ctx);

    GLdouble t[16];
    mglc_matrix_transpose(t, m);
    mglMultMatrixd(ctx, t);
}

void mglPushMatrix(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (st->depth + 1 >= MGLC_MATRIX_DEPTH)
        ERROR_RETURN(GL_STACK_OVERFLOW);

    memcpy(st->stack[st->depth + 1].m, st->stack[st->depth].m, sizeof(MGLCMat));
    st->depth++;
}

void mglPopMatrix(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (st->depth == 0)
        ERROR_RETURN(GL_STACK_UNDERFLOW);

    st->depth--;
    matrix_changed(ctx);
}

void mglOrtho(GLMContext ctx, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble m[16];

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (l == r || b == t || n == f)
        ERROR_RETURN(GL_INVALID_VALUE);

    mglc_matrix_ortho(m, l, r, b, t, n, f);
    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglFrustum(GLMContext ctx, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble m[16];

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (l == r || b == t || n <= 0 || f <= 0 || l == r)
        ERROR_RETURN(GL_INVALID_VALUE);

    mglc_matrix_frustum(m, l, r, b, t, n, f);
    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglTranslated(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble m[16];

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglc_matrix_identity(m);
    m[12] = x; m[13] = y; m[14] = z;
    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglTranslatef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{    MGLC_COMPAT_GUARD(ctx);

    mglTranslated(ctx, x, y, z);
}

void mglScaled(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble m[16];

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglc_matrix_identity(m);
    m[0] = x; m[5] = y; m[10] = z;
    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglScalef(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{    MGLC_COMPAT_GUARD(ctx);

    mglScaled(ctx, x, y, z);
}

void mglRotated(GLMContext ctx, GLdouble angle, GLdouble x, GLdouble y, GLdouble z)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMatrixStack *st = mgl_compat_matrix_stack(ctx, CS()->matrix_mode, NULL);
    GLdouble m[16];

    if (!st)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglc_matrix_rotate(m, angle, x, y, z);
    mglc_matrix_mul(st->stack[st->depth].m, st->stack[st->depth].m, m);
    matrix_changed(ctx);
}

void mglRotatef(GLMContext ctx, GLfloat angle, GLfloat x, GLfloat y, GLfloat z)
{    MGLC_COMPAT_GUARD(ctx);

    mglRotated(ctx, angle, x, y, z);
}
