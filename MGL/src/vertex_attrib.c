/*
 * vertex_attrib.c
 * MGL — generic vertex attribute current values (GL 4.6 §10.2), attribute queries
 * (§10.3.11 GetVertexAttrib*), and the CPU decoders shared with the Metal fetch path.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_vattrib.h"

#define VA_ERROR(_e_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return; } while (0)

static inline struct MGLExtState_vattrib *VS(GLMContext ctx) { return ctx->state.ext_vattrib; }

static inline GLuint maxAttribs(GLMContext ctx)
{
    GLuint m = ctx->state.max_vertex_attribs;
    if (m == 0 || m > MAX_ATTRIBS) m = MAX_ATTRIBS;
    return m;
}

#pragma mark - normalization / packed decoding (GL 4.6 §2.3.5, §10.2.1)

static inline GLfloat snorm(double c, int bits)
{
    double f = c / (double)((1LL << (bits - 1)) - 1);
    return (GLfloat)(f < -1.0 ? -1.0 : f);
}

static inline GLfloat unorm(double c, int bits)
{
    return (GLfloat)(c / (double)((1ULL << bits) - 1));
}

static inline GLint sext(GLuint v, int bits)
{
    GLuint m = 1u << (bits - 1);
    v &= (1u << bits) - 1;
    return (GLint)((v ^ m) - m);
}

float mgl_va_uf11_to_float(GLuint v)
{
    GLuint e = (v >> 6) & 0x1f, m = v & 0x3f;
    if (e == 0) return m ? ldexpf((float)m / 64.0f, -14) : 0.0f;
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf(1.0f + (float)m / 64.0f, (int)e - 15);
}

float mgl_va_uf10_to_float(GLuint v)
{
    GLuint e = (v >> 5) & 0x1f, m = v & 0x1f;
    if (e == 0) return m ? ldexpf((float)m / 32.0f, -14) : 0.0f;
    if (e == 31) return m ? NAN : INFINITY;
    return ldexpf(1.0f + (float)m / 32.0f, (int)e - 15);
}

static float halfToFloat(GLushort h)
{
    GLuint s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float f;
    if (e == 0) f = ldexpf((float)m / 1024.0f, -14);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = ldexpf(1.0f + (float)m / 1024.0f, (int)e - 15);
    return s ? -f : f;
}

/* decode a packed 32 bit value into 4 floats (x,y,z,w) */
static void decodePacked(GLenum type, GLboolean normalized, GLuint v, GLfloat out[4])
{
    switch (type)
    {
        case GL_INT_2_10_10_10_REV:
        {
            GLint c[4] = { sext(v, 10), sext(v >> 10, 10), sext(v >> 20, 10), sext(v >> 30, 2) };
            for (int i = 0; i < 4; i++)
                out[i] = normalized ? snorm(c[i], i == 3 ? 2 : 10) : (GLfloat)c[i];
            break;
        }

        case GL_UNSIGNED_INT_2_10_10_10_REV:
        {
            GLuint c[4] = { v & 0x3ff, (v >> 10) & 0x3ff, (v >> 20) & 0x3ff, (v >> 30) & 0x3 };
            for (int i = 0; i < 4; i++)
                out[i] = normalized ? unorm(c[i], i == 3 ? 2 : 10) : (GLfloat)c[i];
            break;
        }

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            out[0] = mgl_va_uf11_to_float(v & 0x7ff);
            out[1] = mgl_va_uf11_to_float((v >> 11) & 0x7ff);
            out[2] = mgl_va_uf10_to_float((v >> 22) & 0x3ff);
            out[3] = 1.0f;
            break;

        default:
            out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;
            break;
    }
}

/* decode one element of array data (any GL vertex type) to float, defaults (0,0,0,1) */
void mgl_va_fetch_float(const MGLVertexAttribFormat *fmt, const void *src, GLfloat out[4])
{
    GLint comps = (fmt->size == GL_BGRA) ? 4 : fmt->size;
    GLboolean norm = fmt->normalized && !fmt->integer;

    out[0] = out[1] = out[2] = 0.0f; out[3] = 1.0f;

    switch (fmt->type)
    {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        {
            GLuint v; memcpy(&v, src, 4);
            GLfloat t[4];
            decodePacked(fmt->type, norm, v, t);
            if (fmt->size == GL_BGRA) { GLfloat x = t[0]; t[0] = t[2]; t[2] = x; }
            int n = (fmt->type == GL_UNSIGNED_INT_10F_11F_11F_REV) ? 3 : comps;
            for (int i = 0; i < n && i < 4; i++) out[i] = t[i];
            return;
        }
    }

    for (int i = 0; i < comps; i++)
    {
        double c = 0.0;
        const GLubyte *p = (const GLubyte *)src;

        switch (fmt->type)
        {
            case GL_BYTE:           { GLbyte   x; memcpy(&x, p + i * 1, 1); c = x; out[i] = norm ? snorm(c, 8)  : (GLfloat)c; break; }
            case GL_UNSIGNED_BYTE:  { GLubyte  x; memcpy(&x, p + i * 1, 1); c = x; out[i] = norm ? unorm(c, 8)  : (GLfloat)c; break; }
            case GL_SHORT:          { GLshort  x; memcpy(&x, p + i * 2, 2); c = x; out[i] = norm ? snorm(c, 16) : (GLfloat)c; break; }
            case GL_UNSIGNED_SHORT: { GLushort x; memcpy(&x, p + i * 2, 2); c = x; out[i] = norm ? unorm(c, 16) : (GLfloat)c; break; }
            case GL_INT:            { GLint    x; memcpy(&x, p + i * 4, 4); c = x; out[i] = norm ? snorm(c, 32) : (GLfloat)c; break; }
            case GL_UNSIGNED_INT:   { GLuint   x; memcpy(&x, p + i * 4, 4); c = x; out[i] = norm ? unorm(c, 32) : (GLfloat)c; break; }
            case GL_HALF_FLOAT:     { GLushort x; memcpy(&x, p + i * 2, 2); out[i] = halfToFloat(x); break; }
            case GL_FLOAT:          { GLfloat  x; memcpy(&x, p + i * 4, 4); out[i] = x; break; }
            case GL_DOUBLE:         { GLdouble x; memcpy(&x, p + i * 8, 8); out[i] = (GLfloat)x; break; }
            case GL_FIXED:          { GLint    x; memcpy(&x, p + i * 4, 4); out[i] = (GLfloat)((double)x / 65536.0); break; }
            default: break;
        }
    }

    if (fmt->size == GL_BGRA)
    {
        GLfloat x = out[0]; out[0] = out[2]; out[2] = x;
    }
}

/* decode one element of integer array data (no conversion), defaults (0,0,0,1) */
void mgl_va_fetch_int(const MGLVertexAttribFormat *fmt, const void *src, GLint out[4])
{
    const GLubyte *p = (const GLubyte *)src;
    GLint comps = (fmt->size == GL_BGRA) ? 4 : fmt->size;

    out[0] = out[1] = out[2] = 0; out[3] = 1;

    for (int i = 0; i < comps && i < 4; i++)
    {
        switch (fmt->type)
        {
            case GL_BYTE:           { GLbyte   x; memcpy(&x, p + i, 1);     out[i] = x; break; }
            case GL_UNSIGNED_BYTE:  { GLubyte  x; memcpy(&x, p + i, 1);     out[i] = x; break; }
            case GL_SHORT:          { GLshort  x; memcpy(&x, p + i * 2, 2); out[i] = x; break; }
            case GL_UNSIGNED_SHORT: { GLushort x; memcpy(&x, p + i * 2, 2); out[i] = x; break; }
            case GL_INT:
            case GL_UNSIGNED_INT:   { GLint    x; memcpy(&x, p + i * 4, 4); out[i] = x; break; }
            default:
            {
                /* a float array consumed by an integer shader input: convert the value */
                GLfloat f[4];
                mgl_va_fetch_float(fmt, src, f);
                for (int j = 0; j < 4; j++) out[j] = (GLint)f[j];
                return;
            }
        }
    }
}

#pragma mark - current value storage

static void setF(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    MGLCurrentAttrib *a = &VS(ctx)->current[index];
    a->kind = MGL_VA_KIND_FLOAT;
    a->v.f[0] = x; a->v.f[1] = y; a->v.f[2] = z; a->v.f[3] = w;
    a->d[0] = x; a->d[1] = y; a->d[2] = z; a->d[3] = w;
}

static void setI(GLMContext ctx, GLuint index, GLint x, GLint y, GLint z, GLint w)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    MGLCurrentAttrib *a = &VS(ctx)->current[index];
    a->kind = MGL_VA_KIND_INT;
    a->v.i[0] = x; a->v.i[1] = y; a->v.i[2] = z; a->v.i[3] = w;
    a->d[0] = x; a->d[1] = y; a->d[2] = z; a->d[3] = w;
}

static void setU(GLMContext ctx, GLuint index, GLuint x, GLuint y, GLuint z, GLuint w)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    MGLCurrentAttrib *a = &VS(ctx)->current[index];
    a->kind = MGL_VA_KIND_UINT;
    a->v.u[0] = x; a->v.u[1] = y; a->v.u[2] = z; a->v.u[3] = w;
    a->d[0] = x; a->d[1] = y; a->d[2] = z; a->d[3] = w;
}

static void setL(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    MGLCurrentAttrib *a = &VS(ctx)->current[index];
    a->kind = MGL_VA_KIND_DOUBLE;
    a->d[0] = x; a->d[1] = y; a->d[2] = z; a->d[3] = w;
    /* Metal has no fp64 vertex inputs: the value fed to the shader is the float conversion */
    a->v.f[0] = (GLfloat)x; a->v.f[1] = (GLfloat)y; a->v.f[2] = (GLfloat)z; a->v.f[3] = (GLfloat)w;
}

/* VertexAttribP*: comps = 1..4 (the P<n> in the command name) */
static void setP(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value, int comps)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    switch (type)
    {
        case GL_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            break;

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            /* only meaningful for three components (VertexAttribP3ui[v]) */
            if (comps != 3) VA_ERROR(GL_INVALID_ENUM);
            break;

        default:
            VA_ERROR(GL_INVALID_ENUM);
    }

    GLfloat v[4];
    decodePacked(type, normalized, value, v);

    GLfloat out[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    for (int i = 0; i < comps; i++) out[i] = v[i];

    setF(ctx, index, out[0], out[1], out[2], out[3]);
}

#pragma mark - VertexAttrib{1,2,3,4}{s,f,d}[v]

void mglVertexAttrib1d(GLMContext ctx, GLuint index, GLdouble x) { setF(ctx, index, (GLfloat)x, 0, 0, 1); }
void mglVertexAttrib1dv(GLMContext ctx, GLuint index, const GLdouble *v) { setF(ctx, index, (GLfloat)v[0], 0, 0, 1); }
void mglVertexAttrib1f(GLMContext ctx, GLuint index, GLfloat x) { setF(ctx, index, x, 0, 0, 1); }
void mglVertexAttrib1fv(GLMContext ctx, GLuint index, const GLfloat *v) { setF(ctx, index, v[0], 0, 0, 1); }
void mglVertexAttrib1s(GLMContext ctx, GLuint index, GLshort x) { setF(ctx, index, x, 0, 0, 1); }
void mglVertexAttrib1sv(GLMContext ctx, GLuint index, const GLshort *v) { setF(ctx, index, v[0], 0, 0, 1); }

void mglVertexAttrib2d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y) { setF(ctx, index, (GLfloat)x, (GLfloat)y, 0, 1); }
void mglVertexAttrib2dv(GLMContext ctx, GLuint index, const GLdouble *v) { setF(ctx, index, (GLfloat)v[0], (GLfloat)v[1], 0, 1); }
void mglVertexAttrib2f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y) { setF(ctx, index, x, y, 0, 1); }
void mglVertexAttrib2fv(GLMContext ctx, GLuint index, const GLfloat *v) { setF(ctx, index, v[0], v[1], 0, 1); }
void mglVertexAttrib2s(GLMContext ctx, GLuint index, GLshort x, GLshort y) { setF(ctx, index, x, y, 0, 1); }
void mglVertexAttrib2sv(GLMContext ctx, GLuint index, const GLshort *v) { setF(ctx, index, v[0], v[1], 0, 1); }

void mglVertexAttrib3d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z) { setF(ctx, index, (GLfloat)x, (GLfloat)y, (GLfloat)z, 1); }
void mglVertexAttrib3dv(GLMContext ctx, GLuint index, const GLdouble *v) { setF(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1); }
void mglVertexAttrib3f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat z) { setF(ctx, index, x, y, z, 1); }
void mglVertexAttrib3fv(GLMContext ctx, GLuint index, const GLfloat *v) { setF(ctx, index, v[0], v[1], v[2], 1); }
void mglVertexAttrib3s(GLMContext ctx, GLuint index, GLshort x, GLshort y, GLshort z) { setF(ctx, index, x, y, z, 1); }
void mglVertexAttrib3sv(GLMContext ctx, GLuint index, const GLshort *v) { setF(ctx, index, v[0], v[1], v[2], 1); }

void mglVertexAttrib4d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) { setF(ctx, index, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
void mglVertexAttrib4dv(GLMContext ctx, GLuint index, const GLdouble *v) { setF(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
void mglVertexAttrib4f(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) { setF(ctx, index, x, y, z, w); }
void mglVertexAttrib4fv(GLMContext ctx, GLuint index, const GLfloat *v) { setF(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttrib4s(GLMContext ctx, GLuint index, GLshort x, GLshort y, GLshort z, GLshort w) { setF(ctx, index, x, y, z, w); }
void mglVertexAttrib4sv(GLMContext ctx, GLuint index, const GLshort *v) { setF(ctx, index, v[0], v[1], v[2], v[3]); }

#pragma mark - VertexAttrib4{b,i,s,ub,us,ui}v (unnormalized conversion to float)

void mglVertexAttrib4bv(GLMContext ctx, GLuint index, const GLbyte *v) { setF(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttrib4iv(GLMContext ctx, GLuint index, const GLint *v) { setF(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
void mglVertexAttrib4ubv(GLMContext ctx, GLuint index, const GLubyte *v) { setF(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttrib4usv(GLMContext ctx, GLuint index, const GLushort *v) { setF(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttrib4uiv(GLMContext ctx, GLuint index, const GLuint *v) { setF(ctx, index, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }

#pragma mark - VertexAttrib4N* (normalized fixed point, eq. 2.1/2.2)

void mglVertexAttrib4Nbv(GLMContext ctx, GLuint index, const GLbyte *v)
{ setF(ctx, index, snorm(v[0], 8), snorm(v[1], 8), snorm(v[2], 8), snorm(v[3], 8)); }
void mglVertexAttrib4Niv(GLMContext ctx, GLuint index, const GLint *v)
{ setF(ctx, index, snorm(v[0], 32), snorm(v[1], 32), snorm(v[2], 32), snorm(v[3], 32)); }
void mglVertexAttrib4Nsv(GLMContext ctx, GLuint index, const GLshort *v)
{ setF(ctx, index, snorm(v[0], 16), snorm(v[1], 16), snorm(v[2], 16), snorm(v[3], 16)); }
void mglVertexAttrib4Nub(GLMContext ctx, GLuint index, GLubyte x, GLubyte y, GLubyte z, GLubyte w)
{ setF(ctx, index, unorm(x, 8), unorm(y, 8), unorm(z, 8), unorm(w, 8)); }
void mglVertexAttrib4Nubv(GLMContext ctx, GLuint index, const GLubyte *v)
{ setF(ctx, index, unorm(v[0], 8), unorm(v[1], 8), unorm(v[2], 8), unorm(v[3], 8)); }
void mglVertexAttrib4Nuiv(GLMContext ctx, GLuint index, const GLuint *v)
{ setF(ctx, index, unorm(v[0], 32), unorm(v[1], 32), unorm(v[2], 32), unorm(v[3], 32)); }
void mglVertexAttrib4Nusv(GLMContext ctx, GLuint index, const GLushort *v)
{ setF(ctx, index, unorm(v[0], 16), unorm(v[1], 16), unorm(v[2], 16), unorm(v[3], 16)); }

#pragma mark - VertexAttribI* (pure integer)

void mglVertexAttribI1i(GLMContext ctx, GLuint index, GLint x) { setI(ctx, index, x, 0, 0, 1); }
void mglVertexAttribI1iv(GLMContext ctx, GLuint index, const GLint *v) { setI(ctx, index, v[0], 0, 0, 1); }
void mglVertexAttribI1ui(GLMContext ctx, GLuint index, GLuint x) { setU(ctx, index, x, 0, 0, 1); }
void mglVertexAttribI1uiv(GLMContext ctx, GLuint index, const GLuint *v) { setU(ctx, index, v[0], 0, 0, 1); }
void mglVertexAttribI2i(GLMContext ctx, GLuint index, GLint x, GLint y) { setI(ctx, index, x, y, 0, 1); }
void mglVertexAttribI2iv(GLMContext ctx, GLuint index, const GLint *v) { setI(ctx, index, v[0], v[1], 0, 1); }
void mglVertexAttribI2ui(GLMContext ctx, GLuint index, GLuint x, GLuint y) { setU(ctx, index, x, y, 0, 1); }
void mglVertexAttribI2uiv(GLMContext ctx, GLuint index, const GLuint *v) { setU(ctx, index, v[0], v[1], 0, 1); }
void mglVertexAttribI3i(GLMContext ctx, GLuint index, GLint x, GLint y, GLint z) { setI(ctx, index, x, y, z, 1); }
void mglVertexAttribI3iv(GLMContext ctx, GLuint index, const GLint *v) { setI(ctx, index, v[0], v[1], v[2], 1); }
void mglVertexAttribI3ui(GLMContext ctx, GLuint index, GLuint x, GLuint y, GLuint z) { setU(ctx, index, x, y, z, 1); }
void mglVertexAttribI3uiv(GLMContext ctx, GLuint index, const GLuint *v) { setU(ctx, index, v[0], v[1], v[2], 1); }
void mglVertexAttribI4i(GLMContext ctx, GLuint index, GLint x, GLint y, GLint z, GLint w) { setI(ctx, index, x, y, z, w); }
void mglVertexAttribI4iv(GLMContext ctx, GLuint index, const GLint *v) { setI(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttribI4ui(GLMContext ctx, GLuint index, GLuint x, GLuint y, GLuint z, GLuint w) { setU(ctx, index, x, y, z, w); }
void mglVertexAttribI4uiv(GLMContext ctx, GLuint index, const GLuint *v) { setU(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttribI4bv(GLMContext ctx, GLuint index, const GLbyte *v) { setI(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttribI4sv(GLMContext ctx, GLuint index, const GLshort *v) { setI(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttribI4ubv(GLMContext ctx, GLuint index, const GLubyte *v) { setU(ctx, index, v[0], v[1], v[2], v[3]); }
void mglVertexAttribI4usv(GLMContext ctx, GLuint index, const GLushort *v) { setU(ctx, index, v[0], v[1], v[2], v[3]); }

#pragma mark - VertexAttribL* (64 bit)

void mglVertexAttribL1d(GLMContext ctx, GLuint index, GLdouble x) { setL(ctx, index, x, 0, 0, 1); }
void mglVertexAttribL1dv(GLMContext ctx, GLuint index, const GLdouble *v) { setL(ctx, index, v[0], 0, 0, 1); }
void mglVertexAttribL2d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y) { setL(ctx, index, x, y, 0, 1); }
void mglVertexAttribL2dv(GLMContext ctx, GLuint index, const GLdouble *v) { setL(ctx, index, v[0], v[1], 0, 1); }
void mglVertexAttribL3d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z) { setL(ctx, index, x, y, z, 1); }
void mglVertexAttribL3dv(GLMContext ctx, GLuint index, const GLdouble *v) { setL(ctx, index, v[0], v[1], v[2], 1); }
void mglVertexAttribL4d(GLMContext ctx, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) { setL(ctx, index, x, y, z, w); }
void mglVertexAttribL4dv(GLMContext ctx, GLuint index, const GLdouble *v) { setL(ctx, index, v[0], v[1], v[2], v[3]); }

#pragma mark - VertexAttribP* (packed)

void mglVertexAttribP1ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value) { setP(ctx, index, type, normalized, value, 1); }
void mglVertexAttribP1uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value) { setP(ctx, index, type, normalized, value[0], 1); }
void mglVertexAttribP2ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value) { setP(ctx, index, type, normalized, value, 2); }
void mglVertexAttribP2uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value) { setP(ctx, index, type, normalized, value[0], 2); }
void mglVertexAttribP3ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value) { setP(ctx, index, type, normalized, value, 3); }
void mglVertexAttribP3uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value) { setP(ctx, index, type, normalized, value[0], 3); }
void mglVertexAttribP4ui(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, GLuint value) { setP(ctx, index, type, normalized, value, 4); }
void mglVertexAttribP4uiv(GLMContext ctx, GLuint index, GLenum type, GLboolean normalized, const GLuint *value) { setP(ctx, index, type, normalized, value[0], 4); }

#pragma mark - GetVertexAttrib* (§10.3.11)

/* attribute array state of attrib <index> of vao; returns false for an unknown pname */
bool mgl_va_attrib_param(VertexArray *vao, GLuint index, GLenum pname, GLint64 *out)
{
    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVertexAttribFormat *f = &va->fmt[index];
    MGLVertexBinding *b = &va->binding[f->binding];

    switch (pname)
    {
        case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: *out = b->buffer ? b->buffer->name : 0; return true;
        case GL_VERTEX_ATTRIB_ARRAY_ENABLED:        *out = (vao->enabled_attribs >> index) & 1; return true;
        case GL_VERTEX_ATTRIB_ARRAY_SIZE:           *out = f->size; return true;
        case GL_VERTEX_ATTRIB_ARRAY_STRIDE:         *out = f->pointer_stride; return true;
        case GL_VERTEX_ATTRIB_ARRAY_TYPE:           *out = f->type; return true;
        case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED:     *out = f->normalized; return true;
        case GL_VERTEX_ATTRIB_ARRAY_INTEGER:        *out = f->integer; return true;
        case GL_VERTEX_ATTRIB_ARRAY_LONG:           *out = f->is_long; return true;
        case GL_VERTEX_ATTRIB_ARRAY_DIVISOR:        *out = b->divisor; return true;
        case GL_VERTEX_ATTRIB_BINDING:              *out = f->binding; return true;
        case GL_VERTEX_ATTRIB_RELATIVE_OFFSET:      *out = f->relativeoffset; return true;
    }

    return false;
}

enum { Q_DOUBLE, Q_FLOAT, Q_INT, Q_IINT, Q_IUINT, Q_LDOUBLE };

static void getVertexAttrib(GLMContext ctx, GLuint index, GLenum pname, void *params, int qtype)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);

    if (pname == GL_CURRENT_VERTEX_ATTRIB)
    {
        MGLCurrentAttrib *a = &VS(ctx)->current[index];

        for (int i = 0; i < 4; i++)
        {
            double v;

            switch (a->kind)
            {
                case MGL_VA_KIND_INT:    v = a->v.i[i]; break;
                case MGL_VA_KIND_UINT:   v = a->v.u[i]; break;
                case MGL_VA_KIND_DOUBLE: v = a->d[i]; break;
                default:                 v = a->v.f[i]; break;
            }

            switch (qtype)
            {
                case Q_DOUBLE:
                case Q_LDOUBLE: ((GLdouble *)params)[i] = v; break;
                case Q_FLOAT:   ((GLfloat *)params)[i] = (GLfloat)v; break;
                case Q_INT:
                    /* floating point state is rounded to the nearest integer (§22.1.2) */
                    ((GLint *)params)[i] = (a->kind == MGL_VA_KIND_FLOAT || a->kind == MGL_VA_KIND_DOUBLE) ?
                        (GLint)lround(v) : (GLint)v;
                    break;
                case Q_IINT:
                    ((GLint *)params)[i] = (a->kind == MGL_VA_KIND_INT || a->kind == MGL_VA_KIND_UINT) ? a->v.i[i] : (GLint)v;
                    break;
                case Q_IUINT:
                    ((GLuint *)params)[i] = (a->kind == MGL_VA_KIND_INT || a->kind == MGL_VA_KIND_UINT) ? a->v.u[i] : (GLuint)v;
                    break;
            }
        }
        return;
    }

    /* array state lives in the VAO; there is no default VAO in the core profile */
    if (ctx->state.vao == NULL) VA_ERROR(GL_INVALID_OPERATION);

    GLint64 v;
    if (!mgl_va_attrib_param(ctx->state.vao, index, pname, &v)) VA_ERROR(GL_INVALID_ENUM);

    switch (qtype)
    {
        case Q_DOUBLE: case Q_LDOUBLE: *(GLdouble *)params = (GLdouble)v; break;
        case Q_FLOAT:  *(GLfloat *)params = (GLfloat)v; break;
        default:       *(GLint *)params = (GLint)v; break;
    }
}

void mglGetVertexAttribdv(GLMContext ctx, GLuint index, GLenum pname, GLdouble *params) { getVertexAttrib(ctx, index, pname, params, Q_DOUBLE); }
void mglGetVertexAttribfv(GLMContext ctx, GLuint index, GLenum pname, GLfloat *params) { getVertexAttrib(ctx, index, pname, params, Q_FLOAT); }
void mglGetVertexAttribiv(GLMContext ctx, GLuint index, GLenum pname, GLint *params) { getVertexAttrib(ctx, index, pname, params, Q_INT); }
void mglGetVertexAttribIiv(GLMContext ctx, GLuint index, GLenum pname, GLint *params) { getVertexAttrib(ctx, index, pname, params, Q_IINT); }
void mglGetVertexAttribIuiv(GLMContext ctx, GLuint index, GLenum pname, GLuint *params) { getVertexAttrib(ctx, index, pname, params, Q_IUINT); }
void mglGetVertexAttribLdv(GLMContext ctx, GLuint index, GLenum pname, GLdouble *params) { getVertexAttrib(ctx, index, pname, params, Q_LDOUBLE); }

void mglGetVertexAttribPointerv(GLMContext ctx, GLuint index, GLenum pname, void **pointer)
{
    if (index >= maxAttribs(ctx)) VA_ERROR(GL_INVALID_VALUE);
    if (pname != GL_VERTEX_ATTRIB_ARRAY_POINTER) VA_ERROR(GL_INVALID_ENUM);
    if (ctx->state.vao == NULL) VA_ERROR(GL_INVALID_OPERATION);

    *pointer = (void *)mgl_va_full(ctx->state.vao)->fmt[index].pointer;
}

#pragma mark - domain 'vattrib' hooks

/* glGetIntegeri_v / Integer64i_v / Booleani_v ... for vertex buffer binding state (Table 23.5) */
static bool vattribGetIndexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data)
{
    switch (pname)
    {
        case GL_VERTEX_BINDING_BUFFER:
        case GL_VERTEX_BINDING_OFFSET:
        case GL_VERTEX_BINDING_STRIDE:
        case GL_VERTEX_BINDING_DIVISOR:
            break;

        default:
            return false;
    }

    if (index >= MGL_VA_MAX_BINDINGS)
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_VALUE);
        return true;
    }

    double v = 0;

    if (ctx->state.vao)
    {
        MGLVertexBinding *b = &mgl_va_full(ctx->state.vao)->binding[index];

        switch (pname)
        {
            case GL_VERTEX_BINDING_BUFFER:  v = b->buffer ? b->buffer->name : 0; break;
            case GL_VERTEX_BINDING_OFFSET:  v = (double)b->offset; break;
            case GL_VERTEX_BINDING_STRIDE:  v = b->stride; break;
            case GL_VERTEX_BINDING_DIVISOR: v = b->divisor; break;
        }
    }
    else if (pname == GL_VERTEX_BINDING_STRIDE)
    {
        v = 16;
    }

    mgl_get_store(type, data, 0, v);

    return true;
}

void mgl_ext_init_vattrib(GLMContext ctx)
{
    struct MGLExtState_vattrib *s = (struct MGLExtState_vattrib *)calloc(1, sizeof(*s));
    if (!s) return;

    for (int i = 0; i < MAX_ATTRIBS; i++)
    {
        s->current[i].kind = MGL_VA_KIND_FLOAT;
        s->current[i].v.f[3] = 1.0f;
        s->current[i].d[3] = 1.0;
    }

    ctx->state.ext_vattrib = s;

    /* limits reported by glGet (the macOS GL 4.1 defaults don't know these 4.3 limits) */
    ctx->state.var.max_vertex_attrib_bindings = MGL_VA_MAX_BINDINGS;
    ctx->state.var.max_vertex_attrib_relative_offset = MGL_VA_MAX_RELATIVE_OFFSET;
    ctx->state.var.vertex_array_binding = 0;
    ctx->state.var.element_array_buffer_binding = 0;

    static bool registered = false;
    if (!registered)
    {
        mgl_ext_register_get_indexed(vattribGetIndexed);
        registered = true;
    }
}

bool mgl_ext_get_vattrib(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname)
    {
        case GL_MAX_VERTEX_ATTRIB_STRIDE:
            mgl_get_store(type, data, 0, MGL_VA_MAX_STRIDE);
            return true;
    }

    return false;
}
