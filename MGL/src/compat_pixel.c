/*
 * compat_pixel.c — raster position and pixel operations (compat domain).
 *
 * glDrawPixels / glBitmap / glCopyPixels render a textured quad through the
 * internal blit program (compat_ffp.c) at the current raster position scaled
 * by glPixelZoom, exactly like Mesa's st_cb_drawpixels / st_cb_bitmap.
 * glAccum keeps an internal RGBA32F "accumulation buffer" texture: the
 * operations are done on the CPU over a readback of the drawing buffer.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
extern Texture *findTexture(GLMContext ctx, GLuint texture);
#include "vertex_arrays.h"

static void read_pixel_component(GLenum format, GLenum type, const void *pixels, size_t index, int comp, float *out);

#define CS() (ctx->state.ext_compat)

#pragma mark raster position

void mglRasterPos2d(GLMContext ctx, GLdouble x, GLdouble y) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglRasterPos2f(GLMContext ctx, GLfloat x, GLfloat y) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, x, y, 0.0f); }
void mglRasterPos2i(GLMContext ctx, GLint x, GLint y) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglRasterPos2s(GLMContext ctx, GLshort x, GLshort y) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglRasterPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglRasterPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4f(ctx, x, y, z, 1.0f); }
void mglRasterPos3i(GLMContext ctx, GLint x, GLint y, GLint z) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglRasterPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglRasterPos4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
void mglRasterPos4i(GLMContext ctx, GLint x, GLint y, GLint z, GLint w) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
void mglRasterPos4s(GLMContext ctx, GLshort x, GLshort y, GLshort z, GLshort w) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }

void mglRasterPos2dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos2d(ctx, v[0], v[1]); }
void mglRasterPos2fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos2f(ctx, v[0], v[1]); }
void mglRasterPos2iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos2i(ctx, v[0], v[1]); }
void mglRasterPos2sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos2s(ctx, v[0], v[1]); }
void mglRasterPos3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3d(ctx, v[0], v[1], v[2]); }
void mglRasterPos3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3f(ctx, v[0], v[1], v[2]); }
void mglRasterPos3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3i(ctx, v[0], v[1], v[2]); }
void mglRasterPos3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos3s(ctx, v[0], v[1], v[2]); }
void mglRasterPos4dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4d(ctx, v[0], v[1], v[2], v[3]); }
void mglRasterPos4fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4f(ctx, v[0], v[1], v[2], v[3]); }
void mglRasterPos4iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4i(ctx, v[0], v[1], v[2], v[3]); }
void mglRasterPos4sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglRasterPos4s(ctx, v[0], v[1], v[2], v[3]); }

/* the full raster-pos pipeline: eye->clip->window with the current matrices,
 * lighting applied to the current color, all texture coords */
void mgl_compat_raster_pos(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
    struct MGLExtState_compat *s = CS();
    const GLdouble *mv = s->modelview.stack[s->modelview.depth].m;
    const GLdouble *pr = s->projection.stack[s->projection.depth].m;
    double obj[4] = { x, y, z, w }, eye[4], clip[4];
    GLint *vp = STATE(viewport);

    for (int r = 0; r < 4; r++)
    {
        double t = 0;
        for (int k = 0; k < 4; k++)
            t += mv[k * 4 + r] * obj[k];
        eye[r] = t;
    }
    for (int r = 0; r < 4; r++)
    {
        double t = 0;
        for (int k = 0; k < 4; k++)
            t += pr[k * 4 + r] * eye[k];
        clip[r] = t;
    }

    s->raster_pos[0] = x; s->raster_pos[1] = y; s->raster_pos[2] = z; s->raster_pos[3] = w;
    s->raster_color[0] = s->cur_color[0];
    s->raster_color[1] = s->cur_color[1];
    s->raster_color[2] = s->cur_color[2];
    s->raster_color[3] = s->cur_color[3];
    s->raster_index = s->cur_index;
    s->raster_distance = (GLfloat)-eye[2];

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        memcpy(s->raster_texcoord[u], s->cur_texcoord[u], 4 * sizeof(GLfloat));

    /* outside the view volume: raster position becomes invalid */
    if (clip[3] == 0.0 ||
        clip[0] < -clip[3] || clip[0] > clip[3] ||
        clip[1] < -clip[3] || clip[1] > clip[3] ||
        clip[2] < -clip[3] || clip[2] > clip[3])
    {
        s->raster_valid = GL_FALSE;
        return;
    }

    s->raster_window[0] = (GLint)floor(vp[0] + vp[2] * (clip[0] / clip[3] * 0.5 + 0.5));
    s->raster_window[1] = (GLint)floor(vp[1] + vp[3] * (clip[1] / clip[3] * 0.5 + 0.5));
    s->raster_valid = GL_TRUE;
}

void mglRasterPos4f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_RASTERPOS4F);
        if (o)
        {
            o->f[0] = x; o->f[1] = y; o->f[2] = z; o->f[3] = w;
            if (!s->record_exec)
                return;
        }
    }

    mgl_compat_raster_pos(ctx, x, y, z, w);
}

void mglWindowPos2d(GLMContext ctx, GLdouble x, GLdouble y) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglWindowPos2f(GLMContext ctx, GLfloat x, GLfloat y) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, x, y, 0.0f); }
void mglWindowPos2i(GLMContext ctx, GLint x, GLint y) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglWindowPos2s(GLMContext ctx, GLshort x, GLshort y) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, 0.0f); }
void mglWindowPos3d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglWindowPos3i(GLMContext ctx, GLint x, GLint y, GLint z) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglWindowPos3s(GLMContext ctx, GLshort x, GLshort y, GLshort z) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }
void mglWindowPos4d(GLMContext ctx, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)z); }

void mglWindowPos2dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos2d(ctx, v[0], v[1]); }
void mglWindowPos2fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos2f(ctx, v[0], v[1]); }
void mglWindowPos2iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos2i(ctx, v[0], v[1]); }
void mglWindowPos2sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos2s(ctx, v[0], v[1]); }
void mglWindowPos3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3d(ctx, v[0], v[1], v[2]); }
void mglWindowPos3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3f(ctx, v[0], v[1], v[2]); }
void mglWindowPos3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3i(ctx, v[0], v[1], v[2]); }
void mglWindowPos3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos3s(ctx, v[0], v[1], v[2]); }
void mglWindowPos4dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglWindowPos4d(ctx, v[0], v[1], v[2], v[3]); }

void mglWindowPos3f(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_WINDOWPOS3F);
        if (o)
        {
            o->f[0] = x; o->f[1] = y; o->f[2] = z;
            if (!s->record_exec)
                return;
        }
    }

    s->raster_window[0] = (GLint)x;
    s->raster_window[1] = (GLint)y;
    s->raster_pos[0] = x; s->raster_pos[1] = y; s->raster_pos[2] = z; s->raster_pos[3] = 1.0;
    s->raster_valid = GL_TRUE;
    memcpy(s->raster_color, s->cur_color, 4 * sizeof(GLfloat));
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        memcpy(s->raster_texcoord[u], s->cur_texcoord[u], 4 * sizeof(GLfloat));
}

void mglPixelZoom(GLMContext ctx, GLfloat xfactor, GLfloat yfactor)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_PIXEL_ZOOM);
        if (o)
        {
            o->f[0] = xfactor; o->f[1] = yfactor;
            if (!s->record_exec)
                return;
        }
    }

    s->zoom_x = (xfactor == 0.0f) ? 1.0f : xfactor;
    s->zoom_y = (yfactor == 0.0f) ? 1.0f : yfactor;
}

#pragma mark pixel transfer / maps

void mglPixelTransferf(GLMContext ctx, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int idx = -1;

    switch (pname)
    {
        case GL_MAP_COLOR: idx = 0; break;
        case GL_MAP_STENCIL: idx = 1; break;
        case GL_INDEX_SHIFT: idx = 2; break;
        case GL_INDEX_OFFSET: idx = 3; break;
        case GL_RED_SCALE: idx = 4; break;
        case GL_GREEN_SCALE: idx = 5; break;
        case GL_BLUE_SCALE: idx = 6; break;
        case GL_ALPHA_SCALE: idx = 7; break;
        case GL_RED_BIAS: idx = 8; break;
        case GL_GREEN_BIAS: idx = 9; break;
        case GL_BLUE_BIAS: idx = 10; break;
        case GL_ALPHA_BIAS: idx = 11; break;
        case GL_DEPTH_SCALE: idx = 12; break;
        case GL_DEPTH_BIAS: idx = 13; break;
        case GL_POST_COLOR_MATRIX_RED_SCALE: idx = 14; break;
        case GL_POST_COLOR_MATRIX_GREEN_SCALE: idx = 15; break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    s->pixel_transfer[idx] = param;
}

void mglPixelTransferi(GLMContext ctx, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglPixelTransferf(ctx, pname, (GLfloat)param);
}

static int pixel_map_index(GLenum map)
{
    switch (map)
    {
        case GL_PIXEL_MAP_I_TO_I: return 0;
        case GL_PIXEL_MAP_S_TO_S: return 1;
        case GL_PIXEL_MAP_I_TO_R: return 2;
        case GL_PIXEL_MAP_I_TO_G: return 3;
        case GL_PIXEL_MAP_I_TO_B: return 4;
        case GL_PIXEL_MAP_I_TO_A: return 5;
        case GL_PIXEL_MAP_R_TO_R: return 6;
        case GL_PIXEL_MAP_G_TO_G: return 7;
        case GL_PIXEL_MAP_B_TO_B: return 8;
        case GL_PIXEL_MAP_A_TO_A: return 9;
    }
    return -1;
}

void mglPixelMapfv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLfloat *values)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int i = pixel_map_index(map);

    if (i < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (mapsize < 1 || mapsize > 65536)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!values)
        ERROR_RETURN(GL_INVALID_VALUE);

    GLfloat *nm = (GLfloat *)realloc(s->pixel_map[i], (size_t)mapsize * sizeof(GLfloat));
    if (!nm)
        ERROR_RETURN(GL_OUT_OF_MEMORY);

    s->pixel_map[i] = nm;
    s->pixel_map_size[i] = (GLuint)mapsize;
    memcpy(nm, values, (size_t)mapsize * sizeof(GLfloat));
}

void mglPixelMapuiv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLuint *values)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat *tmp = NULL;

    if (mapsize > 0)
    {
        tmp = (GLfloat *)malloc((size_t)mapsize * sizeof(GLfloat));
        if (!tmp)
            return;
        for (GLsizei i = 0; i < mapsize; i++)
            tmp[i] = (GLfloat)(values[i] / 4294967295.0);
    }

    mglPixelMapfv(ctx, map, mapsize, tmp);
    free(tmp);
}

void mglPixelMapusv(GLMContext ctx, GLenum map, GLsizei mapsize, const GLushort *values)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat *tmp = NULL;

    if (mapsize > 0)
    {
        tmp = (GLfloat *)malloc((size_t)mapsize * sizeof(GLfloat));
        if (!tmp)
            return;
        for (GLsizei i = 0; i < mapsize; i++)
            tmp[i] = (GLfloat)(values[i] / 65535.0);
    }

    mglPixelMapfv(ctx, map, mapsize, tmp);
    free(tmp);
}

static void get_pixel_map(GLMContext ctx, GLenum map, double *out, GLuint *n)
{
    int i = pixel_map_index(map);

    *n = 0;
    if (i < 0)
        return;

    for (GLuint k = 0; k < CS()->pixel_map_size[i]; k++)
        out[k] = (double)CS()->pixel_map[i][k];
    *n = CS()->pixel_map_size[i];
}

void mglGetPixelMapfv(GLMContext ctx, GLenum map, GLfloat *values)
{    MGLC_COMPAT_GUARD(ctx);

    double tmp[65536];
    GLuint n = 0;

    if (pixel_map_index(map) < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    get_pixel_map(ctx, map, tmp, &n);
    for (GLuint i = 0; i < n; i++)
        values[i] = (GLfloat)tmp[i];
}

void mglGetPixelMapuiv(GLMContext ctx, GLenum map, GLuint *values)
{    MGLC_COMPAT_GUARD(ctx);

    double tmp[65536];
    GLuint n = 0;

    if (pixel_map_index(map) < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    get_pixel_map(ctx, map, tmp, &n);
    for (GLuint i = 0; i < n; i++)
        values[i] = (GLuint)(tmp[i] * 4294967295.0);
}

void mglGetPixelMapusv(GLMContext ctx, GLenum map, GLushort *values)
{    MGLC_COMPAT_GUARD(ctx);

    double tmp[65536];
    GLuint n = 0;

    if (pixel_map_index(map) < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    get_pixel_map(ctx, map, tmp, &n);
    for (GLuint i = 0; i < n; i++)
        values[i] = (GLushort)(tmp[i] * 65535.0);
}

#pragma mark drawpixels / bitmap / copypixels

/* uploads pixels into the internal blit texture (RGBA8) applying the color
 * pixel-transfer scales/biases and returns the texture name */
/* one GL pixel-map lookup: v in [0,1] -> map[round(v * (n-1))] */
static float pixel_map_lookup(const GLfloat *map, GLuint n, float v)
{
    if (!map || n == 0)
        return v;

    if (n == 1)
        return map[0];

    GLuint i = (GLuint)(v * (float)(n - 1) + 0.5f);

    if (i >= n)
        i = n - 1;

    return map[i];
}

static GLuint blit_upload_pixels(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{
    struct MGLExtState_compat *s = CS();

    if (!s->blit_texture)
    {
        mglGenTextures(ctx, 1, &s->blit_texture);
    }

    /* convert to RGBA8 with the transfer ops (CPU, like Mesa's swrast path) */
    const GLfloat rs = s->pixel_transfer[4], gs = s->pixel_transfer[5],
                    bs = s->pixel_transfer[6], as = s->pixel_transfer[7];
    const GLfloat rb = s->pixel_transfer[8], gb = s->pixel_transfer[9],
                    ab = s->pixel_transfer[10], abias = s->pixel_transfer[11];

    GLubyte *rgba = (GLubyte *)malloc((size_t)width * height * 4);

    if (!rgba)
        return 0;

    for (GLsizei y = 0; y < height; y++)
        for (GLsizei x = 0; x < width; x++)
        {
            size_t si = ((size_t)y * width + x) * 4;
            float r = 0, g = 0, b = 0, a = 1;

            switch (format)
            {
                case GL_RGB:
                case GL_BGR:
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 0, &r);
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 1, &g);
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 2, &b);
                    a = 1.0f;
                    break;
                case GL_RGBA:
                case GL_BGRA:
                case GL_LUMINANCE_ALPHA:
                case GL_ALPHA:
                case GL_LUMINANCE:
                case GL_RED:
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 0, &r);
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 1, &g);
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 2, &b);
                    read_pixel_component(format, type, pixels, (size_t)y * width + x, 3, &a);
                    break;
                default:
                    r = g = b = 0; a = 1;
                    break;
            }

            /* GL_MAP_COLOR: look each component up in the corresponding map */
            if (s->pixel_transfer[0] != 0.0f)
            {
                r = pixel_map_lookup(s->pixel_map[6], s->pixel_map_size[6], r);
                g = pixel_map_lookup(s->pixel_map[7], s->pixel_map_size[7], g);
                b = pixel_map_lookup(s->pixel_map[8], s->pixel_map_size[8], b);
                a = pixel_map_lookup(s->pixel_map[9], s->pixel_map_size[9], a);
            }

            rgba[si + 0] = (GLubyte)fminf(255.0f, fmaxf(0.0f, (r * rs + rb) * 255.0f));
            rgba[si + 1] = (GLubyte)fminf(255.0f, fmaxf(0.0f, (g * gs + gb) * 255.0f));
            rgba[si + 2] = (GLubyte)fminf(255.0f, fmaxf(0.0f, (b * bs + ab) * 255.0f));
            rgba[si + 3] = (GLubyte)fminf(255.0f, fmaxf(0.0f, (a * as + abias) * 255.0f));
        }

    GLuint saved_active = STATE(active_texture);

    mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);

    mglBindTexture(ctx, GL_TEXTURE_2D, s->blit_texture);
    mglTexImage2D(ctx, GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);
    free(rgba);

    return s->blit_texture;
}

/* reads one component of a pixel from the client buffer */
static void read_pixel_component(GLenum format, GLenum type, const void *pixels, size_t index, int comp, float *out)
{
    int ncomp = 4;

    switch (format)
    {
        case GL_RED: case GL_LUMINANCE: case GL_ALPHA: ncomp = 1; break;
        case GL_LUMINANCE_ALPHA: ncomp = 2; break;
        case GL_RGB: case GL_BGR: ncomp = 3; break;
        default: ncomp = 4; break;
    }

    const GLubyte *base = (const GLubyte *)pixels;

    switch (type)
    {
        case GL_UNSIGNED_BYTE:
        {
            const GLubyte *p = base + index * ncomp;
            float v = p[comp < ncomp ? comp : ncomp - 1] / 255.0f;
            if (format == GL_LUMINANCE_ALPHA)
            {
                /* lum,alpha -> rgba mapping done by the caller order */
            }
            *out = v;
            break;
        }
        case GL_BYTE:
        {
            const GLbyte *p = (const GLbyte *)(base + index * ncomp);
            *out = (float)p[comp < ncomp ? comp : ncomp - 1] / 127.0f;
            break;
        }
        case GL_UNSIGNED_INT:
        case GL_INT:
        {
            const GLuint *p = (const GLuint *)(base + index * ncomp * 4);
            *out = (float)(p[comp < ncomp ? comp : ncomp - 1] / 2147483647.0);
            break;
        }
        case GL_FLOAT:
        {
            const GLfloat *p = (const GLfloat *)(base + index * ncomp * 4);
            *out = p[comp < ncomp ? comp : ncomp - 1];
            break;
        }
        default:
            *out = 0.0f;
            break;
    }
}

/* draws a textured quad through the generated fixed-function pipeline (the
 * same path glBegin uses): the quad is given in clip space with identity
 * matrices, |tex| is bound to unit 0 with texenv REPLACE, and |color| becomes
 * the vertex color so MODULATE carries the raster color */
static void blit_quad(GLMContext ctx, GLint x, GLint y, GLsizei w, GLsizei h,
                      GLuint tex, const GLfloat *color, int bitmap_mode)
{
    struct MGLExtState_compat *s = CS();
    GLint *vp = STATE(viewport);

    if (vp[2] <= 0 || vp[3] <= 0)
        return;

    float x0 = (float)(2 * x - vp[0]) / vp[2] - 1.0f;
    float y0 = (float)(2 * y - vp[1]) / vp[3] - 1.0f;
    float x1 = (float)(2 * (x + w) - vp[0]) / vp[2] - 1.0f;
    float y1 = (float)(2 * (y + h) - vp[1]) / vp[3] - 1.0f;

    /* save the fixed-function state the blit borrows */
    GLuint saved_active = STATE(active_texture);
    Texture *saved_tex0 = STATE(texture_units[0]).textures[_TEXTURE_2D];
    uint32_t saved_mask0 = s->unit_target_mask[0];
    GLenum saved_env0 = s->texenv[0].env_mode;
    GLfloat saved_color[4], saved_tc[4];
    memcpy(saved_color, s->cur_color, 4 * sizeof(GLfloat));
    memcpy(saved_tc, s->cur_texcoord[0], 4 * sizeof(GLfloat));
    MGLCMatrixStack saved_mv = s->modelview;
    MGLCMatrixStack saved_pr = s->projection;
    GLenum saved_matrix_mode = s->matrix_mode;
    GLboolean saved_alpha_test = s->alpha_test;
    GLenum saved_alpha_func = s->alpha_func;
    GLfloat saved_alpha_ref = s->alpha_ref;
    GLboolean saved_recording = s->recording;

    /* set up: unit 0 samples the blit texture, REPLACE + optional alpha test */
    s->recording = GL_FALSE;    /* the internal glBegin must not record */

    mglActiveTexture(ctx, GL_TEXTURE0);
    mglBindTexture(ctx, GL_TEXTURE_2D, tex);
    s->unit_target_mask[0] |= 0x2;      /* GL_TEXTURE_2D enabled */
    s->texenv[0].env_mode = GL_MODULATE;

    if (bitmap_mode)
    {
        s->alpha_test = GL_TRUE;
        s->alpha_func = GL_GREATER;
        s->alpha_ref = 0.5f;
    }

    s->matrix_mode = GL_PROJECTION;
    mglLoadIdentity(ctx);
    s->matrix_mode = GL_MODELVIEW;
    mglLoadIdentity(ctx);

    memcpy(s->cur_color, color, 4 * sizeof(GLfloat));

    mglBegin(ctx, GL_QUADS);
    mglTexCoord2f(ctx, 0.0f, 0.0f);
    mglVertex2f(ctx, x0, y0);
    mglTexCoord2f(ctx, 1.0f, 0.0f);
    mglVertex2f(ctx, x1, y0);
    mglTexCoord2f(ctx, 1.0f, 1.0f);
    mglVertex2f(ctx, x1, y1);
    mglTexCoord2f(ctx, 0.0f, 1.0f);
    mglVertex2f(ctx, x0, y1);
    mglEnd(ctx);
    if (getenv("MGLC_DUMP"))
    {
        GLubyte q[4];
        mglFinish(ctx);
        mglReadPixels(ctx, x + w / 2, y + h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, q);
    }

    /* restore */
    s->matrix_mode = GL_PROJECTION;
    s->projection = saved_pr;
    s->matrix_mode = GL_MODELVIEW;
    s->modelview = saved_mv;
    s->matrix_mode = saved_matrix_mode;

    {
        GLdouble m[16];
        mgl_compat_get_matrix(ctx, GL_MODELVIEW, m);
        for (int i = 0; i < 16; i++) s->modelview_f[i] = (GLfloat)m[i];
        mgl_compat_get_matrix(ctx, GL_PROJECTION, m);
        for (int i = 0; i < 16; i++) s->projection_f[i] = (GLfloat)m[i];
    }

    s->alpha_test = saved_alpha_test;
    s->alpha_func = saved_alpha_func;
    s->alpha_ref = saved_alpha_ref;
    memcpy(s->cur_color, saved_color, 4 * sizeof(GLfloat));
    memcpy(s->cur_texcoord[0], saved_tc, 4 * sizeof(GLfloat));
    s->texenv[0].env_mode = saved_env0;
    s->unit_target_mask[0] = saved_mask0;
    mglBindTexture(ctx, GL_TEXTURE_2D, saved_tex0 ? saved_tex0->name : 0);
    mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

    s->recording = saved_recording;

    mgl_compat_dirty(ctx);
}

void mglDrawPixels(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (width < 0 || height < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (s->recording && !s->replaying)
    {
        /* capture a copy of the pixels into the list */
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_DRAW_PIXELS);
        if (o)
        {
            size_t bytes = (size_t)width * height * 4;
            o->i0 = width;
            o->i1 = height;
            o->a = 0;
            o->b = (uint8_t)GL_RGBA;
            o->blob = malloc(bytes ? bytes : 1);
            o->blob_size = bytes;
            /* store as RGBA8 (converted now, replayed later) */
            if (o->blob)
            {
                GLubyte *dst = (GLubyte *)o->blob;
                for (GLsizei y = 0; y < height; y++)
                    for (GLsizei x = 0; x < width; x++)
                    {
                        float r, g, b, a;
                        read_pixel_component(format, type, pixels, (size_t)y * width + x, 0, &r);
                        read_pixel_component(format, type, pixels, (size_t)y * width + x, 1, &g);
                        read_pixel_component(format, type, pixels, (size_t)y * width + x, 2, &b);
                        read_pixel_component(format, type, pixels, (size_t)y * width + x, 3, &a);
                        size_t si = ((size_t)y * width + x) * 4;
                        dst[si] = (GLubyte)(r * 255);
                        dst[si + 1] = (GLubyte)(g * 255);
                        dst[si + 2] = (GLubyte)(b * 255);
                        dst[si + 3] = (GLubyte)(a * 255);
                    }
                if (!s->record_exec)
                    return;
                pixels = o->blob;
                format = GL_RGBA;
                type = GL_UNSIGNED_BYTE;
            }
        }
    }

    if (!s->raster_valid || !pixels)
        return;

    if (!blit_upload_pixels(ctx, width, height, format, type, pixels))
        return;

    GLsizei w = (GLsizei)(width * s->zoom_x);
    GLsizei h = (GLsizei)(height * s->zoom_y);

    blit_quad(ctx, s->raster_window[0], s->raster_window[1], w, h,
              s->blit_texture, s->raster_color, 0);
}

void mglBitmap(GLMContext ctx, GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig,
               GLfloat xmove, GLfloat ymove, const GLubyte *bitmap)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (width < 0 || height < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_BITMAP);
        if (o)
        {
            o->i0 = width;
            o->i1 = height;
            o->f[0] = xorig; o->f[1] = yorig; o->f[2] = xmove; o->f[3] = ymove;
            size_t bytes = (((size_t)width + 7) / 8) * height;
            o->blob = malloc(bytes ? bytes : 1);
            o->blob_size = bytes;
            if (o->blob && bitmap)
                memcpy(o->blob, bitmap, bytes);

            if (!s->record_exec)
                return;
            if (o->blob)
                bitmap = (const GLubyte *)o->blob;
        }
    }

    if (bitmap && s->raster_valid && width > 0 && height > 0)
    {
        /* 1-bit bitmap -> R8 texture (1 = keep, 0 = discard) */
        GLubyte *tex = (GLubyte *)malloc((size_t)width * height);

        if (tex)
        {
            for (GLsizei row = 0; row < height; row++)
            {
                const GLubyte *src = bitmap + (size_t)row * (((size_t)width + 7) / 8);
                /* GL bitmap row 0 is the bottom row; glTexImage2D row 0 is
                 * also v=0 (bottom), so no vertical flip is needed */

                for (GLsizei col = 0; col < width; col++)
                {
                    GLubyte bit = (src[col / 8] >> (7 - (col % 8))) & 1;   /* MSB first */
                    tex[(size_t)row * width + col] = bit ? 255 : 0;
                }
            }

            if (!s->bitmap_texture)
                mglGenTextures(ctx, 1, &s->bitmap_texture);

            GLuint saved_active = STATE(active_texture);
            mglActiveTexture(ctx, GL_TEXTURE0);
            mglBindTexture(ctx, GL_TEXTURE_2D, s->bitmap_texture);
            mglTexImage2D(ctx, GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, tex);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            /* the pattern lives in alpha: modulate keeps the raster color and
             * the alpha test discards the 0 bits */
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_ONE);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ONE);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_ONE);
            mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_RED);
            mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

            free(tex);

            blit_quad(ctx, s->raster_window[0] - (GLint)xorig, s->raster_window[1] - (GLint)yorig,
                      width, height, s->bitmap_texture, s->raster_color, 1);
        }
    }

    /* the raster position moves after the bitmap */
    s->raster_window[0] += (GLint)xmove;
    s->raster_window[1] += (GLint)ymove;
}

void mglCopyPixels(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height, GLenum type)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (width < 0 || height < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (type != GL_COLOR && type != GL_DEPTH && type != GL_STENCIL)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (type != GL_COLOR)
        return;     /* depth/stencil pixel copy is a documented gap */

    if (!s->raster_valid)
        return;

    /* read the source rectangle and push it through the drawpixels path */
    GLsizei rw = STATE(viewport)[2], rh = STATE(viewport)[3];
    GLvoid *tmp = malloc((size_t)width * height * 4);

    if (!tmp)
        return;

    mglReadPixels(ctx, x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, tmp);

    GLuint tex = blit_upload_pixels(ctx, width, height, GL_RGBA, GL_UNSIGNED_BYTE, tmp);

    free(tmp);
    (void)rw; (void)rh;

    if (tex)
        blit_quad(ctx, s->raster_window[0], s->raster_window[1],
                  (GLsizei)(width * s->zoom_x), (GLsizei)(height * s->zoom_y),
                  s->blit_texture, s->raster_color, 0);
}

#pragma mark accumulation buffer

/* internal RGBA32F texture used as the accumulation buffer */
static GLuint accum_texture(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (s->accum_tex)
        return s->accum_tex;

    GLuint tex = 0;
    mglGenTextures(ctx, 1, &tex);

    GLuint saved_active = STATE(active_texture);
    mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
    mglBindTexture(ctx, GL_TEXTURE_2D, tex);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

    s->accum_tex = tex;
    return tex;
}

void mglClearAccum(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    s->accum_clear[0] = red;
    s->accum_clear[1] = green;
    s->accum_clear[2] = blue;
    s->accum_clear[3] = alpha;
}

/* accum ops on the CPU: readback, combine, upload */
void mglAccum(GLMContext ctx, GLenum op, GLfloat value)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    GLint *vp = STATE(viewport);
    GLsizei w = vp[2], h = vp[3];

    switch (op)
    {
        case GL_ACCUM: case GL_LOAD: case GL_ADD:
        case GL_MULT: case GL_RETURN:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    if (w <= 0 || h <= 0)
        return;

    GLuint tex = accum_texture(ctx);
    GLfloat *buf = (GLfloat *)malloc((size_t)w * h * 4 * sizeof(GLfloat));

    if (!buf)
        return;

    bool have_accum = s->accum_valid && s->accum_w == w && s->accum_h == h;

    if (have_accum)
    {
        /* read the accum texture back */
        GLuint saved_active = STATE(active_texture);
        Texture *saved = STATE(texture_units[MGLC_STIPPLE_UNIT]).textures[_TEXTURE_2D];

        mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
        mglBindTexture(ctx, GL_TEXTURE_2D, tex);
        mglGetTexImage(ctx, GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buf);
        mglBindTexture(ctx, GL_TEXTURE_2D, saved ? saved->name : 0);
        mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);
    }

    switch (op)
    {
        case GL_LOAD:
        {
            mglReadPixels(ctx, 0, 0, w, h, GL_RGBA, GL_FLOAT, buf);
            if (value != 1.0f)
                for (GLsizei i = 0; i < w * h * 4; i++)
                    buf[i] *= value;
            break;
        }
        case GL_ACCUM:
        {
            GLfloat *tmp = (GLfloat *)malloc((size_t)w * h * 4 * sizeof(GLfloat));
            if (!tmp) { free(buf); return; }
            mglReadPixels(ctx, 0, 0, w, h, GL_RGBA, GL_FLOAT, tmp);
            for (GLsizei i = 0; i < w * h * 4; i++)
                buf[i] += tmp[i] * value;
            free(tmp);
            break;
        }
        case GL_MULT:
            for (GLsizei i = 0; i < w * h * 4; i++)
                buf[i] *= value;
            break;
        case GL_ADD:
            for (GLsizei i = 0; i < w * h * 4; i++)
                buf[i] += value;
            break;
        case GL_RETURN:
        {
            if (!have_accum)
                break;
            /* draw the accumulation buffer onto the drawing buffer */
            GLuint saved_active = STATE(active_texture);
            Texture *saved = STATE(texture_units[MGLC_STIPPLE_UNIT]).textures[_TEXTURE_2D];

            mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
            mglBindTexture(ctx, GL_TEXTURE_2D, tex);
            mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

            blit_quad(ctx, 0, 0, w, h, tex, (const GLfloat[4]){ value, value, value, value }, 0);

            mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
            mglBindTexture(ctx, GL_TEXTURE_2D, saved ? saved->name : 0);
            mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

            free(buf);
            return;
        }
    }

    /* store back into the accum texture */
    {
        GLuint saved_active = STATE(active_texture);
        Texture *saved = STATE(texture_units[MGLC_STIPPLE_UNIT]).textures[_TEXTURE_2D];

        mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
        mglBindTexture(ctx, GL_TEXTURE_2D, tex);
        mglTexImage2D(ctx, GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, buf);
        mglBindTexture(ctx, GL_TEXTURE_2D, saved ? saved->name : 0);
        mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);
    }

    s->accum_valid = GL_TRUE;
    s->accum_w = w;
    s->accum_h = h;

    free(buf);
}
