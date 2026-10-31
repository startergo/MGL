/*
 * compat_misc.c — evaluators, selection / feedback, and the remaining
 * compatibility entry points (compat domain).
 *
 * Select and feedback use a CPU transform of the primitives (exactly what
 * Mesa's feedback.c / select.c do): vertices go through the modelview and
 * projection stacks and the viewport transform in software; primitives are
 * rasterized with an edge/point-in-triangle test against the pick window.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
extern Texture *findTexture(GLMContext ctx, GLuint texture);

#define CS() (ctx->state.ext_compat)

#pragma mark helpers shared with the other compat files

GLenum mgl_compat_target_index(GLenum target)
{
    switch (target)
    {
        case GL_TEXTURE_1D: return _TEXTURE_1D;
        case GL_TEXTURE_2D: return _TEXTURE_2D;
        case GL_TEXTURE_3D: return _TEXTURE_3D;
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X: case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y: case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z: case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        case GL_TEXTURE_CUBE_MAP: return _TEXTURE_CUBE_MAP;
    }
    return (GLenum)-1;
}

#pragma mark selection / feedback

#pragma mark selection / feedback

static void fb_put(GLMContext ctx, GLfloat v);

/* Mesa-style record write (main/feedback.c write_record): values beyond the
 * end of the buffer are COUNTED but not stored, so glRenderMode can report
 * the overflow as -1. */
static void select_write(GLMContext ctx, GLuint v)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.select_count < s->rmode.select_size)
        s->rmode.select_buf[s->rmode.select_count] = v;
    s->rmode.select_count++;
}

/* a primitive touched the pick volume: mark the pending hit and widen its
 * depth range.  The record itself is deferred (Mesa HitFlag): it is written
 * when the name stack next changes or the render mode is left, so every
 * primitive drawn with the same names merges into ONE hit record whose
 * zmin/zmax span all of them. */
static void select_start_hit(GLMContext ctx, float zmin, float zmax)
{
    struct MGLExtState_compat *s = CS();

    if (!s->rmode.hit_open)
    {
        s->rmode.hit_open = GL_TRUE;
        s->rmode.zmin = zmin;
        s->rmode.zmax = zmax;
    }
    else
    {
        if (zmin < s->rmode.zmin) s->rmode.zmin = zmin;
        if (zmax > s->rmode.zmax) s->rmode.zmax = zmax;
    }
}

/* write the pending hit record: [namelist length, zmin, zmax, names...]
 * with z scaled by 2^32-1 (spec 5.4 / Mesa update_hit_record) */
static void select_close_hit(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->rmode.hit_open)
        return;

    select_write(ctx, (GLuint)s->rmode.name_depth);
    select_write(ctx, (GLuint)fmin(4294967295.0, (double)s->rmode.zmin * 4294967295.0));
    select_write(ctx, (GLuint)fmin(4294967295.0, (double)s->rmode.zmax * 4294967295.0));
    for (GLuint i = 0; i < s->rmode.name_depth; i++)
        select_write(ctx, s->rmode.name_stack[i]);

    s->rmode.hit_open = GL_FALSE;
    s->rmode.hit_count++;
}

void mglPushName(GLMContext ctx, GLuint name)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        extern void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
        mgl_compat_list_record_simple(ctx, MGLC_OP_PUSH_NAME, (int32_t)name);
        if (!s->record_exec)
            return;
    }

    /* name-stack commands are silently ignored outside GL_SELECT (Mesa) */
    if (s->rmode.mode != GL_SELECT)
        return;

    select_close_hit(ctx);

    if (s->rmode.name_depth >= MGLC_NAME_STACK_DEPTH)
        ERROR_RETURN(GL_STACK_OVERFLOW);

    s->rmode.name_stack[s->rmode.name_depth++] = name;
}

void mglPopName(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        extern void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
        mgl_compat_list_record_simple(ctx, MGLC_OP_POP_NAME, 0);
        if (!s->record_exec)
            return;
    }

    if (s->rmode.mode != GL_SELECT)
        return;

    if (s->rmode.name_depth == 0)
        ERROR_RETURN(GL_STACK_UNDERFLOW);

    select_close_hit(ctx);
    s->rmode.name_depth--;
}

void mglLoadName(GLMContext ctx, GLuint name)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        extern void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
        mgl_compat_list_record_simple(ctx, MGLC_OP_LOAD_NAME, (int32_t)name);
        if (!s->record_exec)
            return;
    }

    if (s->rmode.mode != GL_SELECT)
        return;

    if (s->rmode.name_depth == 0)
        ERROR_RETURN(GL_INVALID_OPERATION);

    select_close_hit(ctx);
    s->rmode.name_stack[s->rmode.name_depth - 1] = name;
}

void mglInitNames(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        extern void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
        mgl_compat_list_record_simple(ctx, MGLC_OP_INIT_NAMES, 0);
        if (!s->record_exec)
            return;
    }

    /* glInitNames is ignored outside GL_SELECT (Mesa) */
    if (s->rmode.mode != GL_SELECT)
        return;

    select_close_hit(ctx);
    s->rmode.name_depth = 0;
}

void mglSelectBuffer(GLMContext ctx, GLsizei size, GLuint *buffer)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (size < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (s->rmode.mode == GL_SELECT)
        ERROR_RETURN(GL_INVALID_OPERATION);

    s->rmode.select_buf = buffer;
    s->rmode.select_size = (GLuint)size;
    s->rmode.select_count = 0;
}

void mglFeedbackBuffer(GLMContext ctx, GLsizei size, GLenum type, GLfloat *buffer)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode == GL_FEEDBACK)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (size < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!buffer && size > 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
    }

    if (type != GL_2D && type != GL_3D && type != GL_3D_COLOR && type != GL_3D_COLOR_TEXTURE && type != GL_4D_COLOR_TEXTURE)
        ERROR_RETURN(GL_INVALID_ENUM);

    s->rmode.feedback_buf = buffer;
    s->rmode.feedback_size = (GLuint)size;
    s->rmode.feedback_type = type;
    s->rmode.feedback_count = 0;
}

void mglPassThrough(GLMContext ctx, GLfloat token)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    /* glPassThrough is a no-op outside feedback mode (no error) */
    if (s->rmode.mode != GL_FEEDBACK)
        return;

    fb_put(ctx, (GLfloat)GL_PASS_THROUGH_TOKEN);
    fb_put(ctx, token);
}

GLint mglRenderMode(GLMContext ctx, GLenum mode)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    struct MGLExtState_compat *s = CS();
    GLint ret = 0;

    if (mode != GL_RENDER && mode != GL_SELECT && mode != GL_FEEDBACK)
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, 0);

    /* close out the previous mode */
    switch (s->rmode.mode)
    {
        case GL_SELECT:
            select_close_hit(ctx);
            ret = (s->rmode.select_count > s->rmode.select_size) ? -1 : (GLint)s->rmode.hit_count;
            s->rmode.name_depth = 0;        /* Mesa reset_name_stack_to_empty */
            break;
        case GL_FEEDBACK:
            ret = (s->rmode.feedback_count > s->rmode.feedback_size) ? -1 : (GLint)s->rmode.feedback_count;
            break;
        default:
            break;
    }

    /* entering SELECT / FEEDBACK needs a buffer: report the error but still
     * switch (Mesa _mesa_RenderMode behaviour) */
    if (mode == GL_SELECT && s->rmode.select_size == 0)
    {
        ctx->error_func(ctx, __func__, GL_INVALID_OPERATION);
    }
    else if (mode == GL_FEEDBACK && s->rmode.feedback_size == 0)
    {
        ctx->error_func(ctx, __func__, GL_INVALID_OPERATION);
    }

    s->rmode.mode = (int)mode;
    s->rmode.hit_count = 0;
    s->rmode.hit_open = GL_FALSE;
    s->rmode.select_count = 0;
    s->rmode.feedback_count = 0;

    return ret;
}

#pragma mark CPU geometry processing (select / feedback)

/* &arr[i * 4] or NULL (NULL arrays mean "use the current values") */
static const float *at(const float *arr, GLuint i)
{
    return arr ? &arr[i * 4] : NULL;
}

/* full transform of a vertex: object -> clip -> window (with viewport) */
static void cpu_transform(GLMContext ctx, const float *obj, float *win, float *eye_z, float *clip_w)
{
    struct MGLExtState_compat *s = CS();
    const GLdouble *mv = s->modelview.stack[s->modelview.depth].m;
    const GLdouble *pr = s->projection.stack[s->projection.depth].m;
    double eye[4], clip[4];
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

    if (clip[3] != 0.0)
    {
        win[0] = (float)(vp[0] + vp[2] * (clip[0] / clip[3] * 0.5 + 0.5));
        win[1] = (float)(vp[1] + vp[3] * (clip[1] / clip[3] * 0.5 + 0.5));
        win[2] = (float)((clip[2] / clip[3]) * 0.5 + 0.5);
    }
    else
    {
        win[0] = win[1] = win[2] = 0.0f;
    }

    if (eye_z) *eye_z = (float)eye[2];
    if (clip_w) *clip_w = (float)clip[3];
}

/* point inside triangle (2D, CCW) */
static bool point_in_tri(float px, float py, float ax, float ay, float bx, float by, float cx, float cy)
{
    float d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
    float d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
    float d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}

static bool clip_visible(GLMContext ctx, const float *win)
{
    const GLint *vp = STATE(viewport);
    float rx1 = (float)(vp[0] + vp[2]), ry1 = (float)(vp[1] + vp[3]);

    return win[0] >= (float)vp[0] && win[0] <= rx1 &&
           win[1] >= (float)vp[1] && win[1] <= ry1 &&
           win[2] >= 0.0f && win[2] <= 1.0f;
}

/* ---- exact 2D coverage tests against the viewport rectangle ---- */

/* the window rect being tested against; refreshed by mgl_compat_cpu_draw */
static float vp_rx0, vp_ry0, vp_rx1, vp_ry1;

static void vp_refresh(GLMContext ctx)
{
    const GLint *vp = STATE(viewport);

    vp_rx0 = (float)vp[0];
    vp_ry0 = (float)vp[1];
    vp_rx1 = (float)(vp[0] + vp[2]);
    vp_ry1 = (float)(vp[1] + vp[3]);
}

static bool pt_in_rect(float px, float py)
{
    return px >= vp_rx0 && px <= vp_rx1 && py >= vp_ry0 && py <= vp_ry1;
}

/* true if segments ab and cd share at least one point */
static bool seg_seg_hit(float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy)
{
    float d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    float d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    float d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    float d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
        return true;

    /* collinear / touching cases */
    (void)d1;
    return false;
}

/* triangle vs viewport rect: vertex inside, corner inside, or edge crossing */
static bool tri_in_viewport(float ax, float ay, float bx, float by, float cx, float cy)
{
    float rx0 = vp_rx0, ry0 = vp_ry0, rx1 = vp_rx1, ry1 = vp_ry1;

    if (pt_in_rect(ax, ay) || pt_in_rect(bx, by) || pt_in_rect(cx, cy))
        return true;

    if (point_in_tri(rx0, ry0, ax, ay, bx, by, cx, cy) ||
        point_in_tri(rx1, ry0, ax, ay, bx, by, cx, cy) ||
        point_in_tri(rx1, ry1, ax, ay, bx, by, cx, cy) ||
        point_in_tri(rx0, ry1, ax, ay, bx, by, cx, cy))
        return true;

    return seg_seg_hit(ax, ay, bx, by, rx0, ry0, rx1, ry0) ||
           seg_seg_hit(ax, ay, bx, by, rx1, ry0, rx1, ry1) ||
           seg_seg_hit(ax, ay, bx, by, rx1, ry1, rx0, ry1) ||
           seg_seg_hit(ax, ay, bx, by, rx0, ry1, rx0, ry0) ||
           seg_seg_hit(bx, by, cx, cy, rx0, ry0, rx1, ry0) ||
           seg_seg_hit(bx, by, cx, cy, rx1, ry0, rx1, ry1) ||
           seg_seg_hit(bx, by, cx, cy, rx1, ry1, rx0, ry1) ||
           seg_seg_hit(bx, by, cx, cy, rx0, ry1, rx0, ry0) ||
           seg_seg_hit(cx, cy, ax, ay, rx0, ry0, rx1, ry0) ||
           seg_seg_hit(cx, cy, ax, ay, rx1, ry0, rx1, ry1) ||
           seg_seg_hit(cx, cy, ax, ay, rx1, ry1, rx0, ry1) ||
           seg_seg_hit(cx, cy, ax, ay, rx0, ry1, rx0, ry0);
}

/* segment vs viewport rect */
static bool line_in_viewport(float ax, float ay, float bx, float by)
{
    float rx0 = vp_rx0, ry0 = vp_ry0, rx1 = vp_rx1, ry1 = vp_ry1;

    if (pt_in_rect(ax, ay) || pt_in_rect(bx, by))
        return true;

    return seg_seg_hit(ax, ay, bx, by, rx0, ry0, rx1, ry0) ||
           seg_seg_hit(ax, ay, bx, by, rx1, ry0, rx1, ry1) ||
           seg_seg_hit(ax, ay, bx, by, rx1, ry1, rx0, ry1) ||
           seg_seg_hit(ax, ay, bx, by, rx0, ry1, rx0, ry0);
}

/* one polygon (as a vertex index list) against the viewport: fan test */
static bool poly_in_viewport(const float (*win)[3], const uint32_t *vi, GLuint vn)
{
    if (vn == 0)
        return false;
    if (vn == 1)
        return pt_in_rect(win[vi[0]][0], win[vi[0]][1]);
    if (vn == 2)
        return line_in_viewport(win[vi[0]][0], win[vi[0]][1], win[vi[1]][0], win[vi[1]][1]);

    for (GLuint i = 1; i + 1 < vn; i++)
        if (tri_in_viewport(win[vi[0]][0], win[vi[0]][1],
                            win[vi[i]][0], win[vi[i]][1],
                            win[vi[i + 1]][0], win[vi[i + 1]][1]))
            return true;

    return false;
}

/* ---- feedback ---- */

/* values per vertex for each feedback type (RGBA mode) */
static int fb_coords(GLenum t)
{
    if (t == GL_2D) return 2;
    if (t == GL_4D_COLOR_TEXTURE) return 4;
    return 3;
}

static int fb_vertex_values(GLenum t)
{
    int v = fb_coords(t);
    if (t == GL_3D_COLOR || t == GL_3D_COLOR_TEXTURE || t == GL_4D_COLOR_TEXTURE) v += 4;
    if (t == GL_3D_COLOR_TEXTURE || t == GL_4D_COLOR_TEXTURE) v += 4;
    return v;
}

/* Mesa-style token write (main/feedback.c feedback_token): tokens beyond the
 * end of the buffer are counted but not stored; glRenderMode reports the
 * overflow as -1. */
static void fb_put(GLMContext ctx, GLfloat v)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.feedback_count < s->rmode.feedback_size)
        s->rmode.feedback_buf[s->rmode.feedback_count] = v;
    s->rmode.feedback_count++;
}

/* one feedback vertex: window coords (+w), color, texture as the type asks.
 * col/tex are this vertex's values (from a color / texcoord array or the
 * immediate-mode stream) or NULL for the current values */
static void fb_vertex(GLMContext ctx, const float *win, const float *col, const float *tex)
{
    struct MGLExtState_compat *s = CS();
    GLenum t = (GLenum)s->rmode.feedback_type;

    for (int i = 0; i < fb_coords(t); i++)
        fb_put(ctx, win[i]);

    if (t == GL_3D_COLOR || t == GL_3D_COLOR_TEXTURE || t == GL_4D_COLOR_TEXTURE)
        for (int i = 0; i < 4; i++)
            fb_put(ctx, col ? col[i] : s->cur_color[i]);

    if (t == GL_3D_COLOR_TEXTURE || t == GL_4D_COLOR_TEXTURE)
        for (int i = 0; i < 4; i++)
            fb_put(ctx, tex ? tex[i] : s->cur_texcoord[0][i]);
}

/* emits the feedback stream for one vertex batch (window-space verts); the
 * optional col/tex arrays are float[4] per vertex in the same order */
static void fb_emit_prims(GLMContext ctx, GLenum mode, const float (*win)[3],
                          const float *col, const float *tex, GLuint n)
{
    struct MGLExtState_compat *s = CS();
    int vv = fb_vertex_values((GLenum)s->rmode.feedback_type);

    switch (mode)
    {
        case GL_POINTS:
            for (GLuint i = 0; i < n; i++)
            {
                if (!clip_visible(ctx, win[i])) continue;
                fb_put(ctx, (GLfloat)GL_POINT_TOKEN);
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
            }
            break;

        case GL_LINES:
            for (GLuint i = 0; i + 1 < n; i += 2)
            {
                if (!clip_visible(ctx, win[i]) || !clip_visible(ctx, win[i + 1])) continue;
                fb_put(ctx, (GLfloat)GL_LINE_TOKEN);
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
                fb_vertex(ctx, win[i + 1], at(col, i + 1), at(tex, i + 1));
            }
            break;

        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        {
            GLuint segs = (mode == GL_LINE_STRIP) ? (n ? n - 1 : 0) : n;

            for (GLuint si = 0; si < segs; si++)
            {
                GLuint a = si;
                GLuint b = (si + 1 == n) ? 0 : si + 1;

                if (!clip_visible(ctx, win[a]) || !clip_visible(ctx, win[b])) continue;
                fb_put(ctx, (GLfloat)GL_LINE_TOKEN);
                fb_vertex(ctx, win[a], at(col, a), at(tex, a));
                fb_vertex(ctx, win[b], at(col, b), at(tex, b));
            }
            break;
        }

        case GL_TRIANGLES:
            for (GLuint i = 0; i + 2 < n; i += 3)
            {
                if (!(clip_visible(ctx, win[i]) && clip_visible(ctx, win[i + 1]) && clip_visible(ctx, win[i + 2]))) continue;
                fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                fb_put(ctx, 3.0f);
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
                fb_vertex(ctx, win[i + 1], at(col, i + 1), at(tex, i + 1));
                fb_vertex(ctx, win[i + 2], at(col, i + 2), at(tex, i + 2));
            }
            break;

        case GL_TRIANGLE_STRIP:
            for (GLuint i = 0; i + 2 < n; i++)
            {
                if (!(clip_visible(ctx, win[i]) && clip_visible(ctx, win[i + 1]) && clip_visible(ctx, win[i + 2]))) continue;
                fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                fb_put(ctx, 3.0f);
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
                fb_vertex(ctx, win[i + 1], at(col, i + 1), at(tex, i + 1));
                fb_vertex(ctx, win[i + 2], at(col, i + 2), at(tex, i + 2));
            }
            break;

        case GL_TRIANGLE_FAN:
            for (GLuint i = 1; i + 1 < n; i++)
            {
                if (!(clip_visible(ctx, win[0]) && clip_visible(ctx, win[i]) && clip_visible(ctx, win[i + 1]))) continue;
                fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                fb_put(ctx, 3.0f);
                fb_vertex(ctx, win[0], at(col, 0), at(tex, 0));
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
                fb_vertex(ctx, win[i + 1], at(col, i + 1), at(tex, i + 1));
            }
            break;

        case GL_QUADS:
            for (GLuint i = 0; i + 3 < n; i += 4)
            {
                bool vis = clip_visible(ctx, win[i]) && clip_visible(ctx, win[i + 1]) &&
                           clip_visible(ctx, win[i + 2]) && clip_visible(ctx, win[i + 3]);
                if (!vis) continue;
                fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                fb_put(ctx, 4.0f);
                fb_vertex(ctx, win[i], at(col, i), at(tex, i));
                fb_vertex(ctx, win[i + 1], at(col, i + 1), at(tex, i + 1));
                fb_vertex(ctx, win[i + 2], at(col, i + 2), at(tex, i + 2));
                fb_vertex(ctx, win[i + 3], at(col, i + 3), at(tex, i + 3));
            }
            break;

        default:    /* GL_QUAD_STRIP, GL_POLYGON */
        {
            GLuint step = (mode == GL_QUAD_STRIP) ? 2 : 1;
            GLuint vn = 0;
            uint32_t stack[4];
            const uint32_t *g = stack;
            uint32_t *heap = NULL;

            if (mode == GL_QUAD_STRIP)
            {
                for (GLuint i = 0; i + 3 < n; i += step)
                {
                    stack[0] = i; stack[1] = i + 1; stack[2] = i + 3; stack[3] = i + 2;
                    bool vis = true;
                    for (int k = 0; k < 4; k++)
                        if (!clip_visible(ctx, win[stack[k]])) vis = false;
                    if (!vis) continue;
                    fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                    fb_put(ctx, 4.0f);
                    for (int k = 0; k < 4; k++)
                        fb_vertex(ctx, win[stack[k]], at(col, stack[k]), at(tex, stack[k]));
                }
                break;
            }

            /* GL_POLYGON (or unknown): the whole batch is one polygon */
            heap = (uint32_t *)malloc(n * sizeof(uint32_t));
            if (!heap)
                break;
            g = heap;
            vn = n;
            for (GLuint i = 0; i < n; i++)
                heap[i] = i;

            bool vis = true;
            for (GLuint i = 0; i < vn; i++)
                if (!clip_visible(ctx, win[g[i]])) vis = false;

            if (vis)
            {
                fb_put(ctx, (GLfloat)GL_POLYGON_TOKEN);
                fb_put(ctx, (GLfloat)vn);
                for (GLuint i = 0; i < vn; i++)
                    fb_vertex(ctx, win[g[i]], at(col, g[i]), at(tex, g[i]));
            }
            free(heap);
            break;
        }
    }
}

/* feedback token + raster-position vertex for glBitmap / glDrawPixels /
 * glCopyPixels (Mesa main/drawpix.c): emitted only when the raster position
 * is valid; nothing is rasterized outside GL_RENDER. */
void mgl_compat_feedback_pixel(GLMContext ctx, GLenum token)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode != GL_FEEDBACK || !s->raster_valid)
        return;

    fb_put(ctx, (GLfloat)(GLint)token);
    fb_vertex(ctx, s->raster_window_f, s->raster_color, s->raster_texcoord[0]);
}

/* window-space position of an object-space vertex (for line stipple) */
void mgl_compat_window_pos(GLMContext ctx, const float *obj, float *win)
{
    float eye_z = 0.0f, clip_w = 1.0f;

    cpu_transform(ctx, obj, win, &eye_z, &clip_w);
}

void mgl_compat_cpu_draw(GLMContext ctx, GLenum mode, const float *pos, GLuint n)
{
    mgl_compat_cpu_draw_rgba(ctx, mode, pos, n, NULL, NULL);
}

/* same, with per-vertex color / unit-0 texcoords for feedback (NULL = use the
 * current values); arrays are float[4] per vertex in the same order as pos */
void mgl_compat_cpu_draw_rgba(GLMContext ctx, GLenum mode, const float *pos, GLuint n,
                              const float *col, const float *tex)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode == GL_RENDER || !pos || n == 0)
        return;

    vp_refresh(ctx);

    if (n > (1u << 20))
        n = 1u << 20;

    /* window-space copies of the vertices */
    float (*win)[3] = (float (*)[3])malloc(n * sizeof *win);
    float *z = (float *)malloc(n * sizeof(float));

    if (!win || !z)
    {
        free(win);
        free(z);
        return;
    }

    for (GLuint i = 0; i < n; i++)
    {
        float obj[4] = { pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2], pos[i * 4 + 3] };
        if (obj[3] == 0.0f) obj[3] = 1.0f;
        cpu_transform(ctx, obj, win[i], &z[i], NULL);
    }

    if (s->rmode.mode == GL_FEEDBACK)
    {
        fb_emit_prims(ctx, mode, win, col, tex, n);
    }
    else
    {
        /* select: one hit record per GL primitive that touches the pick
         * volume (the viewport + depth range, i.e. what the rasterizer sees) */
        switch (mode)
        {
            case GL_POINTS:
                for (GLuint i = 0; i < n; i++)
                    if (clip_visible(ctx, win[i]))
                        select_start_hit(ctx, win[i][2], win[i][2]);
                break;

            case GL_LINES:
                for (GLuint i = 0; i + 1 < n; i += 2)
                {
                    uint32_t g[2] = { i, i + 1 };
                    if (poly_in_viewport(win, g, 2))
                        select_start_hit(ctx, fminf(win[i][2], win[i + 1][2]), fmaxf(win[i][2], win[i + 1][2]));
                }
                break;

            case GL_LINE_STRIP:
            case GL_LINE_LOOP:
                for (GLuint i = 0; i + 1 < n; i++)
                {
                    uint32_t g[2] = { i, i + 1 };
                    if (poly_in_viewport(win, g, 2))
                        select_start_hit(ctx, fminf(win[i][2], win[i + 1][2]), fmaxf(win[i][2], win[i + 1][2]));
                }
                if (mode == GL_LINE_LOOP && n > 2)
                {
                    uint32_t g[2] = { n - 1, 0 };
                    if (poly_in_viewport(win, g, 2))
                        select_start_hit(ctx, fminf(win[n - 1][2], win[0][2]), fmaxf(win[n - 1][2], win[0][2]));
                }
                break;

            case GL_TRIANGLES:
                for (GLuint i = 0; i + 2 < n; i += 3)
                {
                    uint32_t g[3] = { i, i + 1, i + 2 };
                    if (poly_in_viewport(win, g, 3))
                        select_start_hit(ctx, fminf(fminf(win[i][2], win[i + 1][2]), win[i + 2][2]), fmaxf(fmaxf(win[i][2], win[i + 1][2]), win[i + 2][2]));
                }
                break;

            case GL_TRIANGLE_STRIP:
                for (GLuint i = 0; i + 2 < n; i++)
                {
                    uint32_t g[3] = { i, i + 1, i + 2 };
                    if (poly_in_viewport(win, g, 3))
                        select_start_hit(ctx, fminf(fminf(win[i][2], win[i + 1][2]), win[i + 2][2]), fmaxf(fmaxf(win[i][2], win[i + 1][2]), win[i + 2][2]));
                }
                break;

            case GL_TRIANGLE_FAN:
                for (GLuint i = 1; i + 1 < n; i++)
                {
                    uint32_t g[3] = { 0, i, i + 1 };
                    if (poly_in_viewport(win, g, 3))
                        select_start_hit(ctx, fminf(fminf(win[0][2], win[i][2]), win[i + 1][2]), fmaxf(fmaxf(win[0][2], win[i][2]), win[i + 1][2]));
                }
                break;

            case GL_QUADS:
                for (GLuint i = 0; i + 3 < n; i += 4)
                {
                    uint32_t g[4] = { i, i + 1, i + 2, i + 3 };
                    if (poly_in_viewport(win, g, 4))
                    {
                        float z = fminf(fminf(win[i][2], win[i + 1][2]), fminf(win[i + 2][2], win[i + 3][2]));
                        float zm = fmaxf(fmaxf(win[i][2], win[i + 1][2]), fmaxf(win[i + 2][2], win[i + 3][2]));
                        select_start_hit(ctx, z, zm);
                    }
                }
                break;

            case GL_QUAD_STRIP:
                for (GLuint i = 0; i + 3 < n; i += 2)
                {
                    uint32_t g[4] = { i, i + 1, i + 3, i + 2 };
                    if (poly_in_viewport(win, g, 4))
                    {
                        float z = fminf(fminf(win[i][2], win[i + 1][2]), fminf(win[i + 2][2], win[i + 3][2]));
                        float zm = fmaxf(fmaxf(win[i][2], win[i + 1][2]), fmaxf(win[i + 2][2], win[i + 3][2]));
                        select_start_hit(ctx, z, zm);
                    }
                }
                break;

            default:    /* GL_POLYGON and anything else: one primitive */
            {
                uint32_t *g = (uint32_t *)malloc(n * sizeof(uint32_t));
                if (g)
                {
                    float zmin = 2.0f, zmax = -2.0f;
                    for (GLuint i = 0; i < n; i++)
                    {
                        g[i] = i;
                        if (win[i][2] < zmin) zmin = win[i][2];
                        if (win[i][2] > zmax) zmax = win[i][2];
                    }
                    if (poly_in_viewport(win, g, n))
                        select_start_hit(ctx, zmin, zmax);
                    free(g);
                }
                break;
            }
        }

        /* the record stays open: it is written when the name stack next
         * changes or the render mode is left, so every primitive drawn with
         * the same names merges into one hit record (Mesa HitFlag) */
    }

    free(win);
    free(z);
}

#pragma mark evaluators

static int eval_target_index(GLenum target)
{
    switch (target)
    {
        case GL_MAP1_INDEX: return 0;
        case GL_MAP1_TEXTURE_COORD_1: return 1;
        case GL_MAP1_TEXTURE_COORD_2: return 2;
        case GL_MAP1_TEXTURE_COORD_3: return 3;
        case GL_MAP1_TEXTURE_COORD_4: return 4;
        case GL_MAP1_VERTEX_3: return 5;
        case GL_MAP1_VERTEX_4: return 6;
        case GL_MAP1_COLOR_4: return 7;
        case GL_MAP1_NORMAL: return 8;
        case GL_MAP2_INDEX: return 0;
        case GL_MAP2_TEXTURE_COORD_1: return 1;
        case GL_MAP2_TEXTURE_COORD_2: return 2;
        case GL_MAP2_TEXTURE_COORD_3: return 3;
        case GL_MAP2_TEXTURE_COORD_4: return 4;
        case GL_MAP2_VERTEX_3: return 5;
        case GL_MAP2_VERTEX_4: return 6;
        case GL_MAP2_COLOR_4: return 7;
        case GL_MAP2_NORMAL: return 8;
    }
    return -1;
}

static int eval_comps(GLenum target)
{
    switch (target)
    {
        case GL_MAP1_INDEX: case GL_MAP2_INDEX: return 1;
        case GL_MAP1_TEXTURE_COORD_1: case GL_MAP2_TEXTURE_COORD_1: return 1;
        case GL_MAP1_TEXTURE_COORD_2: case GL_MAP2_TEXTURE_COORD_2: return 2;
        case GL_MAP1_TEXTURE_COORD_3: case GL_MAP2_TEXTURE_COORD_3: return 3;
        case GL_MAP1_TEXTURE_COORD_4: case GL_MAP2_TEXTURE_COORD_4: return 4;
        case GL_MAP1_VERTEX_3: case GL_MAP2_VERTEX_3: return 3;
        case GL_MAP1_VERTEX_4: case GL_MAP2_VERTEX_4: return 4;
        case GL_MAP1_COLOR_4: case GL_MAP2_COLOR_4: return 4;
        case GL_MAP1_NORMAL: case GL_MAP2_NORMAL: return 3;
    }
    return 0;
}

/* evaluator component count, shared with the attrib-stack snapshot code */
int mgl_compat_eval_comps(GLenum target) { return eval_comps(target); }

/* ---- display-list recording for evaluator commands (Mesa dlist.c compiles
 * Map1/Map2, MapGrid, EvalCoord, EvalMesh and EvalPoint instead of executing
 * them at compile time) ---- */

static void record_map1(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2,
                        GLint stride, GLint order, int comps, const void *pts, int dbl)
{
    struct MGLExtState_compat *s = CS();
    MGLCListOp *o;
    GLfloat *b;
    size_t n = (size_t)order * comps;

    if (!s->recording || s->replaying)
        return;

    o = mgl_compat_list_op(ctx, MGLC_OP_MAP1);
    if (!o) return;
    o->i0 = (int32_t)target;
    o->i1 = order;
    o->f[0] = u1; o->f[1] = u2; o->f[2] = (float)comps;
    o->blob = malloc(n * sizeof(GLfloat));
    if (!o->blob) return;
    o->blob_size = n * sizeof(GLfloat);
    b = (GLfloat *)o->blob;

    if (dbl)
    {
        const GLdouble *p = (const GLdouble *)pts;
        for (GLint i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                b[i * comps + c] = (GLfloat)p[(size_t)i * stride + c];
    }
    else
    {
        const GLfloat *p = (const GLfloat *)pts;
        for (GLint i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                b[i * comps + c] = p[(size_t)i * stride + c];
    }
}

static void record_map2(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder,
                        GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, int comps,
                        const void *pts, int dbl)
{
    struct MGLExtState_compat *s = CS();
    MGLCListOp *o;
    GLfloat *b;
    size_t n = (size_t)uorder * vorder * comps;

    if (!s->recording || s->replaying)
        return;

    o = mgl_compat_list_op(ctx, MGLC_OP_MAP2);
    if (!o) return;
    o->i0 = (int32_t)target;
    o->i1 = uorder;
    o->a = (uint8_t)vorder;
    o->f[0] = u1; o->f[1] = u2; o->f[2] = v1; o->f[3] = v2; o->f[4] = (float)comps;
    o->blob = malloc(n * sizeof(GLfloat));
    if (!o->blob) return;
    o->blob_size = n * sizeof(GLfloat);
    b = (GLfloat *)o->blob;

    if (dbl)
    {
        const GLdouble *p = (const GLdouble *)pts;
        for (GLint i = 0; i < uorder; i++)
            for (GLint j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    b[(i * vorder + j) * comps + c] =
                        (GLfloat)p[(size_t)i * ustride + (size_t)j * vstride + c];
    }
    else
    {
        const GLfloat *p = (const GLfloat *)pts;
        for (GLint i = 0; i < uorder; i++)
            for (GLint j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    b[(i * vorder + j) * comps + c] =
                        p[(size_t)i * ustride + (size_t)j * vstride + c];
    }
}

/* the caller pattern is:  if (recording && !replaying) { record; if (!record_exec) return; } */

/* glMap1/glMap2 error surface, in Mesa's check order (main/eval.c): domain,
 * order, points, target enum, stride, then the "evaluators generate texture
 * coordinates for unit 0 only" INVALID_OPERATION (GL 1.2.1 spec F.2.13, kept
 * by every later compat spec).  Returns the map slot or -1 (error set). */
static int map_validate(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2,
                        GLdouble v1, GLdouble v2, GLint stride, GLint ustride, GLint vstride,
                        GLint order, GLint uorder, GLint vorder, const void *points, int *comps)
{
    int two_d = (uorder != 0);            /* Map2 passes uorder/vorder */

    if (two_d ? (u1 == u2 || v1 == v2) : (u1 == u2))
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    if (two_d ? (uorder < 1 || uorder > MGLC_MAX_EVAL_ORDER || vorder < 1 || vorder > MGLC_MAX_EVAL_ORDER)
              : (order < 1 || order > MGLC_MAX_EVAL_ORDER))
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);
    if (!points)
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);

    int idx = eval_target_index(target);
    int k = eval_comps(target);
    if (idx < 0 || k == 0)
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, -1);

    if (two_d ? (ustride < k || vstride < k) : (stride < k))
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, -1);

    if (STATE(active_texture) != 0)
        ERROR_RETURN_VALUE(GL_INVALID_OPERATION, -1);

    *comps = k;
    return idx;
}


void mglMap1d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int comps = 0;
    int idx = map_validate(ctx, target, u1, u2, 0, 0, stride, 0, 0, order, 0, 0, points, &comps);

    if (idx < 0)
        return;

    if (s->recording && !s->replaying)
    {
        record_map1(ctx, target, (GLfloat)u1, (GLfloat)u2, stride, order, comps, points, 1);
        if (!s->record_exec)
            return;
    }

    /* stride counts ELEMENTS (doubles) between control points, not bytes */
    s->map1[idx].target = target;
    s->map1[idx].stride = stride;
    s->map1[idx].order = order;
    s->map1[idx].u1 = u1;
    s->map1[idx].u2 = u2;
    s->map1[idx].type = GL_DOUBLE;

    free(s->map1[idx].fpoints);
    s->map1[idx].fpoints = (GLfloat *)malloc((size_t)order * comps * sizeof(GLfloat));
    if (s->map1[idx].fpoints)
        for (int i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                s->map1[idx].fpoints[i * comps + c] = (GLfloat)points[i * (size_t)stride + c];
}

void mglMap1f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int comps = 0;
    int idx = map_validate(ctx, target, u1, u2, 0, 0, stride, 0, 0, order, 0, 0, points, &comps);

    if (idx < 0)
        return;

    if (s->recording && !s->replaying)
    {
        record_map1(ctx, target, (GLfloat)u1, (GLfloat)u2, stride, order, comps, points, 0);
        if (!s->record_exec)
            return;
    }

    s->map1[idx].target = target;
    s->map1[idx].stride = stride;
    s->map1[idx].order = order;
    s->map1[idx].u1 = u1;
    s->map1[idx].u2 = u2;
    s->map1[idx].type = GL_FLOAT;

    free(s->map1[idx].fpoints);
    s->map1[idx].fpoints = (GLfloat *)malloc((size_t)order * comps * sizeof(GLfloat));
    if (s->map1[idx].fpoints)
        for (int i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                s->map1[idx].fpoints[i * comps + c] = points[i * (size_t)stride + c];
}

void mglMap2d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder,
              GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int comps = 0;
    int idx = map_validate(ctx, target, u1, u2, v1, v2, 0, ustride, vstride, 0, uorder, vorder, points, &comps);

    if (idx < 0)
        return;

    if (s->recording && !s->replaying)
    {
        record_map2(ctx, target, (GLfloat)u1, (GLfloat)u2, ustride, uorder,
                    (GLfloat)v1, (GLfloat)v2, vstride, vorder, comps, points, 1);
        if (!s->record_exec)
            return;
    }

    s->map2[idx].target = target;
    s->map2[idx].ustride = ustride;
    s->map2[idx].vstride = vstride;
    s->map2[idx].uorder = uorder;
    s->map2[idx].vorder = vorder;
    s->map2[idx].u1 = u1;
    s->map2[idx].u2 = u2;
    s->map2[idx].v1 = v1;
    s->map2[idx].v2 = v2;
    s->map2[idx].type = GL_DOUBLE;

    free(s->map2[idx].fpoints);
    s->map2[idx].fpoints = (GLfloat *)malloc((size_t)uorder * vorder * comps * sizeof(GLfloat));
    if (s->map2[idx].fpoints)
        for (int i = 0; i < uorder; i++)
            for (int j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    s->map2[idx].fpoints[(i * vorder + j) * comps + c] =
                        (GLfloat)points[i * (size_t)ustride + j * (size_t)vstride + c];
}

void mglMap2f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder,
              GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int comps = 0;
    int idx = map_validate(ctx, target, u1, u2, v1, v2, 0, ustride, vstride, 0, uorder, vorder, points, &comps);

    if (idx < 0)
        return;

    if (s->recording && !s->replaying)
    {
        record_map2(ctx, target, (GLfloat)u1, (GLfloat)u2, ustride, uorder,
                    (GLfloat)v1, (GLfloat)v2, vstride, vorder, comps, points, 0);
        if (!s->record_exec)
            return;
    }

    s->map2[idx].target = target;
    s->map2[idx].ustride = ustride;
    s->map2[idx].vstride = vstride;
    s->map2[idx].uorder = uorder;
    s->map2[idx].vorder = vorder;
    s->map2[idx].u1 = u1;
    s->map2[idx].u2 = u2;
    s->map2[idx].v1 = v1;
    s->map2[idx].v2 = v2;
    s->map2[idx].type = GL_FLOAT;

    free(s->map2[idx].fpoints);
    s->map2[idx].fpoints = (GLfloat *)malloc((size_t)uorder * vorder * comps * sizeof(GLfloat));
    if (s->map2[idx].fpoints)
        for (int i = 0; i < uorder; i++)
            for (int j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    s->map2[idx].fpoints[(i * vorder + j) * comps + c] =
                        points[i * (size_t)ustride + j * (size_t)vstride + c];
}

void mglGetMapdv(GLMContext ctx, GLenum target, GLenum query, GLdouble *v)
{    MGLC_COMPAT_GUARD(ctx);

    int idx = eval_target_index(target);
    struct MGLExtState_compat *s = CS();

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    bool m2 = (target >= GL_MAP2_INDEX);
    int comps = eval_comps(target);
    int order = m2 ? s->map2[idx].uorder * s->map2[idx].vorder : s->map1[idx].order;
    GLfloat *pts = m2 ? s->map2[idx].fpoints : s->map1[idx].fpoints;

    switch (query)
    {
        case GL_ORDER:
            v[0] = m2 ? s->map2[idx].uorder : s->map1[idx].order;
            if (m2) v[1] = s->map2[idx].vorder;
            break;
        case GL_COEFF:
            if (pts)
                for (int i = 0; i < order * comps; i++)
                    v[i] = (GLdouble)pts[i];
            break;
        case GL_DOMAIN:
            if (m2) { v[0] = s->map2[idx].u1; v[1] = s->map2[idx].u2; v[2] = s->map2[idx].v1; v[3] = s->map2[idx].v2; }
            else { v[0] = s->map1[idx].u1; v[1] = s->map1[idx].u2; }
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

void mglGetMapfv(GLMContext ctx, GLenum target, GLenum query, GLfloat *v)
{    MGLC_COMPAT_GUARD(ctx);

    int idx = eval_target_index(target);
    struct MGLExtState_compat *s = CS();

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    bool m2 = (target >= GL_MAP2_INDEX);
    int comps = eval_comps(target);
    int order = m2 ? s->map2[idx].uorder * s->map2[idx].vorder : s->map1[idx].order;
    GLfloat *pts = m2 ? s->map2[idx].fpoints : s->map1[idx].fpoints;

    switch (query)
    {
        case GL_ORDER:
            v[0] = m2 ? (GLfloat)s->map2[idx].uorder : (GLfloat)s->map1[idx].order;
            if (m2) v[1] = (GLfloat)s->map2[idx].vorder;
            break;
        case GL_COEFF:
            if (pts)
                for (int i = 0; i < order * comps; i++)
                    v[i] = pts[i];
            break;
        case GL_DOMAIN:
            if (m2) { v[0] = (GLfloat)s->map2[idx].u1; v[1] = (GLfloat)s->map2[idx].u2; v[2] = (GLfloat)s->map2[idx].v1; v[3] = (GLfloat)s->map2[idx].v2; }
            else { v[0] = (GLfloat)s->map1[idx].u1; v[1] = (GLfloat)s->map1[idx].u2; }
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

void mglGetMapiv(GLMContext ctx, GLenum target, GLenum query, GLint *v)
{    MGLC_COMPAT_GUARD(ctx);

    int idx = eval_target_index(target);
    struct MGLExtState_compat *s = CS();

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    bool m2 = (target >= GL_MAP2_INDEX);
    int comps = eval_comps(target);
    int order = m2 ? s->map2[idx].uorder * s->map2[idx].vorder : s->map1[idx].order;
    GLfloat *pts = m2 ? s->map2[idx].fpoints : s->map1[idx].fpoints;

    switch (query)
    {
        case GL_ORDER:
            v[0] = m2 ? s->map2[idx].uorder : s->map1[idx].order;
            if (m2) v[1] = s->map2[idx].vorder;
            break;
        case GL_COEFF:
            if (pts)
                for (int i = 0; i < order * comps; i++)
                    v[i] = (GLint)pts[i];
            break;
        case GL_DOMAIN:
            if (m2) { v[0] = (GLint)s->map2[idx].u1; v[1] = (GLint)s->map2[idx].u2; v[2] = (int)s->map2[idx].v1; v[3] = (int)s->map2[idx].v2; }
            else { v[0] = (int)s->map1[idx].u1; v[1] = (int)s->map1[idx].u2; }
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }
}

/* de Casteljau over one curve of control points (n x comps doubles) */
static void dc_curve(const double *p, int n, int comps, double t, double *out)
{
    double tmp[MGLC_MAX_EVAL_ORDER * 4];

    for (int i = 0; i < n * comps; i++)
        tmp[i] = p[i];

    for (int k = n - 1; k > 0; k--)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < comps; c++)
                tmp[i * comps + c] = (1.0 - t) * tmp[i * comps + c] + t * tmp[(i + 1) * comps + c];

    for (int c = 0; c < comps; c++)
        out[c] = tmp[c];
}

/* Bernstein evaluation of a 1D map.  The parameter is NOT clamped: like
 * Mesa's horner/de Casteljau, glEvalCoord1/2 extrapolate linearly outside
 * [u1,u2] (the GL spec defines the maps as polynomials over the reals). */
static void eval1(GLMContext ctx, int idx, float u, float *out)
{
    struct MGLExtState_compat *s = CS();
    int n = s->map1[idx].order;
    int comps = eval_comps(s->map1[idx].target);
    double du = (s->map1[idx].u2 - s->map1[idx].u1);
    double t = (du != 0.0) ? (u - s->map1[idx].u1) / du : 0.0;

    for (int c = 0; c < comps; c++)
        out[c] = 0.0f;
    out[3] = 1.0f;

    if (!s->map1[idx].fpoints || n < 1)
        return;                            /* defaults already in out */

    {
        double pts[MGLC_MAX_EVAL_ORDER * 4];
        double o[4] = { 0, 0, 0, 1 };

        for (int i = 0; i < n * comps; i++)
            pts[i] = (double)s->map1[idx].fpoints[i];

        dc_curve(pts, n, comps, t, o);

        for (int c = 0; c < comps; c++)
            out[c] = (float)o[c];
        if (comps < 4)
            out[3] = 1.0f;
    }
}

/* 2-D surface point AND exact partial derivatives d/dtu, d/dtv (the GL_AUTO_NORMAL
 * source).  The derivative control nets are the true Bernstein derivatives
 * (order * forward differences), which differ from Mesa's _math_de_casteljau_surf
 * only by the positive scale that normalization removes. */
static void eval2_deriv(GLMContext ctx, int idx, float u, float v,
                        double *out, double *du_out, double *dv_out)
{
    struct MGLExtState_compat *s = CS();
    int un = s->map2[idx].uorder, vn = s->map2[idx].vorder;
    int comps = eval_comps(s->map2[idx].target);
    double du_ = s->map2[idx].u2 - s->map2[idx].u1, dv_ = s->map2[idx].v2 - s->map2[idx].v1;
    double tu = (du_ != 0.0) ? (u - s->map2[idx].u1) / du_ : 0.0;
    double tv = (dv_ != 0.0) ? (v - s->map2[idx].v1) / dv_ : 0.0;

    double col[MGLC_MAX_EVAL_ORDER * 4];
    double dcol[MGLC_MAX_EVAL_ORDER * 4];

    for (int c = 0; c < comps; c++)
    {
        out[c] = 0.0;
        du_out[c] = 0.0;
        dv_out[c] = 0.0;
        if (c == 3) out[c] = 1.0;
    }

    if (!s->map2[idx].fpoints || un < 1 || vn < 1)
        return;                            /* defaults already in out */

    for (int j = 0; j < un; j++)
    {
        double row[MGLC_MAX_EVAL_ORDER * 4];

        /* value curve of this u row */
        for (int i = 0; i < vn; i++)
            for (int c = 0; c < comps; c++)
                row[i * comps + c] = s->map2[idx].fpoints[((size_t)j * vn + i) * comps + c];
        dc_curve(row, vn, comps, tv, &col[j * comps]);

        /* d/dtv of this u row: vorder * forward differences of the row */
        if (vn > 1)
        {
            for (int i = 0; i < vn - 1; i++)
                for (int c = 0; c < comps; c++)
                    row[i * comps + c] = (double)vn *
                        ((double)s->map2[idx].fpoints[((size_t)j * vn + i + 1) * comps + c] -
                         (double)s->map2[idx].fpoints[((size_t)j * vn + i) * comps + c]);
            dc_curve(row, vn - 1, comps, tv, &dcol[j * comps]);
        }
        else
            for (int c = 0; c < comps; c++)
                dcol[j * comps + c] = 0.0;
    }

    dc_curve(col, un, comps, tu, out);

    /* d/dtu: uorder * forward differences of the collapsed rows */
    if (un > 1)
    {
        double dcurve[MGLC_MAX_EVAL_ORDER * 4];

        for (int i = 0; i < un - 1; i++)
            for (int c = 0; c < comps; c++)
                dcurve[i * comps + c] = (double)un * (col[(i + 1) * comps + c] - col[i * comps + c]);
        dc_curve(dcurve, un - 1, comps, tu, du_out);
    }

    dc_curve(dcol, un, comps, tu, dv_out);
}

static void eval2(GLMContext ctx, int idx, float u, float v, float *out)
{
    double p[4] = { 0, 0, 0, 1 }, dz[4] = { 0, 0, 0, 0 };
    int comps;

    eval2_deriv(ctx, idx, u, v, p, dz, dz);
    comps = eval_comps(CS()->map2[idx].target);
    for (int c = 0; c < comps; c++)
        out[c] = (float)p[c];
    if (comps < 4)
        out[3] = 1.0f;
}

void mglEvalCoord1d(GLMContext ctx, GLdouble u) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord1f(ctx, (GLfloat)u); }
void mglEvalCoord1dv(GLMContext ctx, const GLdouble *u) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord1f(ctx, (GLfloat)u[0]); }
void mglEvalCoord1fv(GLMContext ctx, const GLfloat *u) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord1f(ctx, u[0]); }

void mglEvalCoord1f(GLMContext ctx, GLfloat u)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    float v[4];

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALCOORD1F);

        if (o) o->f[0] = u;
        return;                          /* compiled: replayed with the maps
                                          * current AT CALL TIME (Mesa) */
    }

    /* only ENABLED maps are evaluated; glMap1f merely defines them */
    if (s->map1[7].enabled && s->map1[7].target == GL_MAP1_COLOR_4)
    {
        eval1(ctx, 7, u, v);
        mglColor4f(ctx, v[0], v[1], v[2], v[3]);
    }
    if (s->map1[0].enabled && s->map1[0].target == GL_MAP1_INDEX)
    {
        eval1(ctx, 0, u, v);
        mglIndexf(ctx, v[0]);
    }
    if (s->map1[8].enabled && s->map1[8].target == GL_MAP1_NORMAL)
    {
        eval1(ctx, 8, u, v);
        mglNormal3f(ctx, v[0], v[1], v[2]);
    }
    if (s->map1[4].enabled && s->map1[4].target == GL_MAP1_TEXTURE_COORD_4)
    {
        eval1(ctx, 4, u, v);
        mglTexCoord4f(ctx, v[0], v[1], v[2], v[3]);
    }
    else if (s->map1[3].enabled && s->map1[3].target == GL_MAP1_TEXTURE_COORD_3)
    {
        eval1(ctx, 3, u, v);
        mglTexCoord3f(ctx, v[0], v[1], v[2]);
    }
    else if (s->map1[2].enabled && s->map1[2].target == GL_MAP1_TEXTURE_COORD_2)
    {
        eval1(ctx, 2, u, v);
        mglTexCoord2f(ctx, v[0], v[1]);
    }
    else if (s->map1[1].enabled && s->map1[1].target == GL_MAP1_TEXTURE_COORD_1)
    {
        eval1(ctx, 1, u, v);
        mglTexCoord1f(ctx, v[0]);
    }

    if (s->map1[6].enabled && s->map1[6].target == GL_MAP1_VERTEX_4)
    {
        eval1(ctx, 6, u, v);
        mglVertex4f(ctx, v[0], v[1], v[2], v[3]);
    }
    else if (s->map1[5].enabled && s->map1[5].target == GL_MAP1_VERTEX_3)
    {
        eval1(ctx, 5, u, v);
        mglVertex3f(ctx, v[0], v[1], v[2]);
    }
}

/* GL_AUTO_NORMAL: analytic surface normal = cross(du, dv), normalised, from
 * the de Casteljau partial derivatives.  For a homogeneous (VERTEX_4) map the
 * tangents are corrected to the derivative of the projected point (quotient
 * rule; the 1/w^2 factor is dropped since only the direction matters). */
static void eval_autonormal(GLMContext ctx, const double *p, double *du, double *dv, int comps)
{
    if (comps == 4)
    {
        for (int c = 0; c < 3; c++)
        {
            du[c] = du[c] * p[3] - du[3] * p[c];
            dv[c] = dv[c] * p[3] - dv[3] * p[c];
        }
    }

    double nx = du[1] * dv[2] - du[2] * dv[1];
    double ny = du[2] * dv[0] - du[0] * dv[2];
    double nz = du[0] * dv[1] - du[1] * dv[0];
    double len = sqrt(nx * nx + ny * ny + nz * nz);

    if (len > 1e-30)
    {
        nx /= len;
        ny /= len;
        nz /= len;
    }
    else
    {
        nx = ny = 0.0;                     /* degenerate surface */
        nz = 1.0;
    }

    mglNormal3f(ctx, (float)nx, (float)ny, (float)nz);
}

void mglEvalCoord2d(GLMContext ctx, GLdouble u, GLdouble v) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord2f(ctx, (GLfloat)u, (GLfloat)v); }
void mglEvalCoord2dv(GLMContext ctx, const GLdouble *u) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord2f(ctx, (GLfloat)u[0], (GLfloat)u[1]); }
void mglEvalCoord2fv(GLMContext ctx, const GLfloat *u) {    MGLC_COMPAT_GUARD(ctx);
 mglEvalCoord2f(ctx, u[0], u[1]); }

void mglEvalCoord2f(GLMContext ctx, GLfloat u, GLfloat v)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    float out[4];

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALCOORD2F);

        if (o) { o->f[0] = u; o->f[1] = v; }
        return;                          /* compiled, re-evaluated on call */
    }

    if (s->map2[7].enabled && s->map2[7].target == GL_MAP2_COLOR_4)
    {
        eval2(ctx, 7, u, v, out);
        mglColor4f(ctx, out[0], out[1], out[2], out[3]);
    }
    if (s->map2[0].enabled && s->map2[0].target == GL_MAP2_INDEX)
    {
        eval2(ctx, 0, u, v, out);
        mglIndexf(ctx, out[0]);
    }
    if (!s->auto_normal && s->map2[8].enabled && s->map2[8].target == GL_MAP2_NORMAL)
    {
        eval2(ctx, 8, u, v, out);
        mglNormal3f(ctx, out[0], out[1], out[2]);
    }
    if (s->map2[4].enabled && s->map2[4].target == GL_MAP2_TEXTURE_COORD_4)
    {
        eval2(ctx, 4, u, v, out);
        mglTexCoord4f(ctx, out[0], out[1], out[2], out[3]);
    }
    else if (s->map2[3].enabled && s->map2[3].target == GL_MAP2_TEXTURE_COORD_3)
    {
        eval2(ctx, 3, u, v, out);
        mglTexCoord3f(ctx, out[0], out[1], out[2]);
    }
    else if (s->map2[2].enabled && s->map2[2].target == GL_MAP2_TEXTURE_COORD_2)
    {
        eval2(ctx, 2, u, v, out);
        mglTexCoord2f(ctx, out[0], out[1]);
    }
    else if (s->map2[1].enabled && s->map2[1].target == GL_MAP2_TEXTURE_COORD_1)
    {
        eval2(ctx, 1, u, v, out);
        mglTexCoord1f(ctx, out[0]);
    }

    if (s->map2[6].enabled && s->map2[6].target == GL_MAP2_VERTEX_4)
    {
        double p[4] = { 0, 0, 0, 1 }, du[4] = { 0, 0, 0, 0 }, dv[4] = { 0, 0, 0, 0 };

        eval2_deriv(ctx, 6, u, v, p, du, dv);

        if (s->auto_normal)
            eval_autonormal(ctx, p, du, dv, 4);

        mglVertex4f(ctx, (float)p[0], (float)p[1], (float)p[2], (float)p[3]);
    }
    else if (s->map2[5].enabled && s->map2[5].target == GL_MAP2_VERTEX_3)
    {
        if (s->auto_normal)
        {
            double p[4] = { 0, 0, 0, 1 }, du[4] = { 0, 0, 0, 0 }, dv[4] = { 0, 0, 0, 0 };

            eval2_deriv(ctx, 5, u, v, p, du, dv);
            eval_autonormal(ctx, p, du, dv, 3);
            mglVertex3f(ctx, (float)p[0], (float)p[1], (float)p[2]);
        }
        else
        {
            eval2(ctx, 5, u, v, out);
            mglVertex3f(ctx, out[0], out[1], out[2]);
        }
    }
}

void mglMapGrid1d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (un < 1)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_MAPGRID1);

        if (o) { o->i0 = un; o->f[0] = (GLfloat)u1; o->f[1] = (GLfloat)u2; }
        if (!s->record_exec)
            return;
    }

    s->map_grid1_n = un;
    s->map_grid1_u1 = u1;
    s->map_grid1_u2 = u2;
}

void mglMapGrid1f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2)
{    MGLC_COMPAT_GUARD(ctx);

    mglMapGrid1d(ctx, un, u1, u2);
}

void mglMapGrid2d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (un < 1 || vn < 1)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_MAPGRID2);

        if (o) { o->i0 = un; o->i1 = vn; o->f[0] = (GLfloat)u1; o->f[1] = (GLfloat)u2; o->f[2] = (GLfloat)v1; o->f[3] = (GLfloat)v2; }
        if (!s->record_exec)
            return;
    }

    s->map_grid2_nu = un;
    s->map_grid2_u1 = u1;
    s->map_grid2_u2 = u2;
    s->map_grid2_nv = vn;
    s->map_grid2_v1 = v1;
    s->map_grid2_v2 = v2;
}

void mglMapGrid2f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2)
{    MGLC_COMPAT_GUARD(ctx);

    mglMapGrid2d(ctx, un, u1, u2, vn, v1, v2);
}

void mglEvalMesh1(GLMContext ctx, GLenum mode, GLint i1, GLint i2)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (mode != GL_POINT && mode != GL_LINE)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALMESH);

        if (o)
        {
            o->a = (uint8_t)(mode == GL_POINT ? 0 : 1);
            o->b = 1;
            o->i0 = i1;
            o->i1 = i2;
        }
        return;                          /* compiled: re-meshed on call */
    }

    /* no vertex map enabled: the whole mesh is a no-op (not even the colour
     * or normal maps are evaluated) — Mesa main/draw.c _mesa_EvalMesh1 */
    if (!s->map1[6].enabled && !s->map1[5].enabled)
        return;

    if (s->map_grid1_n < 1)
        return;

    if (mode == GL_POINT)
    {
        mglBegin(ctx, GL_POINTS);
        for (GLint i = i1; i <= i2; i++)
        {
            double u = s->map_grid1_u1 + (double)i * (s->map_grid1_u2 - s->map_grid1_u1) / s->map_grid1_n;
            mglEvalCoord1d(ctx, u);
        }
        mglEnd(ctx);
    }
    else
    {
        mglBegin(ctx, GL_LINE_STRIP);
        for (GLint i = i1; i <= i2; i++)
        {
            double u = s->map_grid1_u1 + (double)i * (s->map_grid1_u2 - s->map_grid1_u1) / s->map_grid1_n;
            mglEvalCoord1d(ctx, u);
        }
        mglEnd(ctx);
    }
}

void mglEvalPoint1(GLMContext ctx, GLint i)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALPOINT1);

        if (o) o->i0 = i;
        return;                          /* compiled (Mesa dlist) */
    }

    double u = s->map_grid1_u1 + (double)i * (s->map_grid1_u2 - s->map_grid1_u1) / s->map_grid1_n;
    mglEvalCoord1d(ctx, u);
}

void mglEvalPoint2(GLMContext ctx, GLint i, GLint j)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALPOINT2);

        if (o) { o->i0 = i; o->i1 = j; }
        return;                          /* compiled (Mesa dlist) */
    }

    double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
    double v = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
    mglEvalCoord2d(ctx, u, v);
}

void mglEvalMesh2(GLMContext ctx, GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (mode != GL_POINT && mode != GL_LINE && mode != GL_FILL)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EVALMESH);

        if (o)
        {
            o->a = (uint8_t)(mode == GL_POINT ? 0 : (mode == GL_LINE ? 1 : 2));
            o->b = 2;
            o->i0 = i1;
            o->i1 = i2;
            o->f[2] = (GLfloat)j1;
            o->f[3] = (GLfloat)j2;
        }
        return;                          /* compiled: re-meshed on call */
    }

    /* no vertex map enabled: the whole mesh is a no-op (Mesa _mesa_EvalMesh2) */
    if (!s->map2[6].enabled && !s->map2[5].enabled)
        return;

    if (s->map_grid2_nu < 1 || s->map_grid2_nv < 1)
        return;

    if (mode == GL_POINT)
    {
        mglBegin(ctx, GL_POINTS);
        for (GLint j = j1; j <= j2; j++)
        {
            double v = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
            for (GLint i = i1; i <= i2; i++)
            {
                double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
                mglEvalCoord2d(ctx, u, v);
            }
        }
        mglEnd(ctx);
    }
    else if (mode == GL_LINE)
    {
        for (GLint j = j1; j <= j2; j++)
        {
            double v = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
            mglBegin(ctx, GL_LINE_STRIP);
            for (GLint i = i1; i <= i2; i++)
            {
                double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
                mglEvalCoord2d(ctx, u, v);
            }
            mglEnd(ctx);
        }
        for (GLint i = i1; i <= i2; i++)
        {
            double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
            mglBegin(ctx, GL_LINE_STRIP);
            for (GLint j = j1; j <= j2; j++)
            {
                double v = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
                mglEvalCoord2d(ctx, u, v);
            }
            mglEnd(ctx);
        }
    }
    else
    {
        for (GLint j = j1; j < j2; j++)
        {
            double v0 = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
            double v1 = s->map_grid2_v1 + (double)(j + 1) * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;

            mglBegin(ctx, GL_QUAD_STRIP);
            for (GLint i = i1; i <= i2; i++)
            {
                double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
                mglEvalCoord2d(ctx, u, v0);
                mglEvalCoord2d(ctx, u, v1);
            }
            mglEnd(ctx);
        }
    }
}

#pragma mark misc compatibility entry points

void mglPrioritizeTextures(GLMContext ctx, GLsizei n, const GLuint *textures, const GLclampf *priorities)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (n < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!textures || !priorities)
        return;

    /* priorities are advisory; MGL keeps all textures resident, so the value
     * is stored on the object for queries only */
    for (GLsizei i = 0; i < n; i++)
    {
        Texture *tex = findTexture(ctx, textures[i]);
        if (tex && textures[i] < MGLC_TEX_PRIO_MAX)
            s->texture_priority[textures[i]] = priorities[i];
    }
}

GLboolean mglAreTexturesResident(GLMContext ctx, GLsizei n, const GLuint *textures, GLboolean *residences)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    if (n < 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, 0);
        return GL_FALSE;
    }

    if (!textures || !residences)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, 0);
        return GL_FALSE;
    }

    /* every texture is resident (it lives in the MTLTexture); the zero name
     * is not a texture and must raise GL_INVALID_VALUE */
    for (GLsizei i = 0; i < n; i++)
    {
        if (textures[i] == 0)
        {
            ERROR_RETURN_VALUE(GL_INVALID_VALUE, 0);
            return GL_FALSE;
        }

        residences[i] = GL_TRUE;
    }

    return GL_TRUE;
}

GLboolean mglIsTextureResident(GLMContext ctx, GLuint texture)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    Texture *tex = findTexture(ctx, texture);
    return tex ? GL_TRUE : GL_FALSE;
}

void mglClearIndex(GLMContext ctx, GLfloat c)
{    MGLC_COMPAT_GUARD(ctx);

    /* RGBA-mode context: the color index clear value is tracked for queries */
    STATE(var.index_clear_value) = (GLint)c;
}

void mglIndexMask(GLMContext ctx, GLuint mask)
{    MGLC_COMPAT_GUARD(ctx);

    /* no color-index framebuffer: tracked, no effect (documented gap) */
    STATE(var.index_writemask) = mask;
}
