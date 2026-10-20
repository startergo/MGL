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

static GLint select_write(GLMContext ctx, GLuint v)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.select_ptr - s->rmode.select_buf >= (GLint)s->rmode.select_size)
        return 0;

    *s->rmode.select_ptr++ = v;
    return 1;
}

static void select_start_hit(GLMContext ctx, float z)
{
    struct MGLExtState_compat *s = CS();

    if (!s->rmode.hit_open)
    {
        s->rmode.hit_open = GL_TRUE;
        s->rmode.hit_zmin_slot = 1;
        s->rmode.hit_zmax_slot = 2;
        select_write(ctx, 0);                       /* name count, patched on close */
        s->rmode.hit_zmin_slot = (GLuint)(s->rmode.select_ptr - s->rmode.select_buf);
        select_write(ctx, (GLuint)z);               /* zmin placeholder */
        s->rmode.hit_zmax_slot = (GLuint)(s->rmode.select_ptr - s->rmode.select_buf);
        select_write(ctx, (GLuint)z);               /* zmax placeholder */
        s->rmode.zmin = z;
        s->rmode.zmax = z;
    }
    else
    {
        if (z < s->rmode.zmin) s->rmode.zmin = z;
        if (z > s->rmode.zmax) s->rmode.zmax = z;
    }
}

static void select_close_hit(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->rmode.hit_open)
        return;

    GLuint ncount = s->rmode.select_ptr - s->rmode.select_buf - s->rmode.hit_zmax_slot - 1;
    s->rmode.select_buf[s->rmode.hit_zmin_slot - 2] = ncount;
    s->rmode.select_buf[s->rmode.hit_zmin_slot] = (GLuint)fmin(4294967295.0, (double)s->rmode.zmin * 2147483647.0);
    s->rmode.select_buf[s->rmode.hit_zmax_slot] = (GLuint)fmin(4294967295.0, (double)s->rmode.zmax * 2147483647.0);
    s->rmode.hit_open = GL_FALSE;
    s->rmode.hit_count++;
}

void mglPushName(GLMContext ctx, GLuint name)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode != GL_SELECT)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = NULL;
        extern void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
        mgl_compat_list_record_simple(ctx, MGLC_OP_PUSH_NAME, (int32_t)name);
        (void)o;
        if (!s->record_exec)
            return;
    }

    select_close_hit(ctx);

    if (s->rmode.name_depth >= MGLC_NAME_STACK_DEPTH)
        ERROR_RETURN(GL_STACK_OVERFLOW);

    s->rmode.name_stack[s->rmode.name_depth++] = name;
}

void mglPopName(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode != GL_SELECT)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (s->rmode.name_depth == 0)
        ERROR_RETURN(GL_STACK_UNDERFLOW);

    select_close_hit(ctx);
    s->rmode.name_depth--;
}

void mglLoadName(GLMContext ctx, GLuint name)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode != GL_SELECT)
        ERROR_RETURN(GL_INVALID_OPERATION);

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

    select_close_hit(ctx);
    s->rmode.name_depth = 0;
}

void mglSelectBuffer(GLMContext ctx, GLsizei size, GLuint *buffer)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode == GL_SELECT)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (!buffer && size > 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->rmode.select_buf = buffer;
    s->rmode.select_size = (GLuint)size;
    s->rmode.select_count = 0;
    s->rmode.select_ptr = buffer;
}

void mglFeedbackBuffer(GLMContext ctx, GLsizei size, GLenum type, GLfloat *buffer)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode == GL_FEEDBACK)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (type != GL_2D && type != GL_3D && type != GL_3D_COLOR && type != GL_3D_COLOR_TEXTURE && type != GL_4D_COLOR_TEXTURE)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!buffer && size > 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->rmode.feedback_buf = buffer;
    s->rmode.feedback_size = (GLuint)size;
    s->rmode.feedback_type = type;
    s->rmode.feedback_count = 0;
}

void mglPassThrough(GLMContext ctx, GLfloat token)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode != GL_FEEDBACK)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (s->rmode.feedback_count + 2 > s->rmode.feedback_size)
        return;

    s->rmode.feedback_buf[s->rmode.feedback_count++] = (GLfloat)GL_PASS_THROUGH_TOKEN;
    s->rmode.feedback_buf[s->rmode.feedback_count++] = token;
}

GLint mglRenderMode(GLMContext ctx, GLenum mode)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    struct MGLExtState_compat *s = CS();
    GLint ret = 0;

    if (mode != GL_RENDER && mode != GL_SELECT && mode != GL_FEEDBACK)
        ERROR_RETURN_VALUE(GL_INVALID_ENUM, 0);

    /* close the previous mode */
    switch (s->rmode.mode)
    {
        case GL_SELECT:
            select_close_hit(ctx);
            ret = (GLint)s->rmode.hit_count;
            break;
        case GL_FEEDBACK:
            if (s->rmode.feedback_count < s->rmode.feedback_size)
                s->rmode.feedback_buf[s->rmode.feedback_count] = (GLfloat)GL_3D_COLOR_TEXTURE + 1;   /* sentinel unused */
            ret = (GLint)s->rmode.feedback_count;
            break;
        default:
            break;
    }

    s->rmode.mode = (int)mode;
    s->rmode.hit_count = 0;
    s->rmode.hit_open = GL_FALSE;
    s->rmode.select_ptr = s->rmode.select_buf;
    s->rmode.feedback_count = 0;
    s->rmode.name_depth = 0;

    return ret;
}

#pragma mark CPU geometry processing (select / feedback)

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

static bool clip_visible(float *win)
{
    return win[0] >= 0.0f && win[0] < 65536.0f && win[1] >= 0.0f && win[1] < 65536.0f && win[2] >= 0.0f && win[2] <= 1.0f;
}

void mgl_compat_cpu_draw(GLMContext ctx, GLenum mode, const float *pos, GLuint n)
{
    struct MGLExtState_compat *s = CS();

    if (s->rmode.mode == GL_RENDER || !pos || n == 0)
        return;

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
        switch (mode)
        {
            case GL_POINTS:
                for (GLuint i = 0; i < n; i++)
                {
                    if (!clip_visible(win[i])) continue;
                    if (s->rmode.feedback_count + 5 <= s->rmode.feedback_size)
                    {
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = (GLfloat)GL_POINT_TOKEN;
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[i][0];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[i][1];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[i][2];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = s->cur_color[0];
                    }
                }
                break;

            case GL_LINES:
            case GL_LINE_STRIP:
            case GL_LINE_LOOP:
            {
                GLuint segs = 0;
                if (mode == GL_LINES) segs = n / 2;
                else if (mode == GL_LINE_STRIP) segs = n ? n - 1 : 0;
                else segs = n;

                for (GLuint si = 0; si < segs; si++)
                {
                    GLuint a = (mode == GL_LINES) ? si * 2 : si;
                    GLuint b = (mode == GL_LINES) ? si * 2 + 1 : ((si + 1 == n) ? 0 : si + 1);

                    if (!clip_visible(win[a]) || !clip_visible(win[b]))
                        continue;

                    if (s->rmode.feedback_count + 9 <= s->rmode.feedback_size)
                    {
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = (GLfloat)GL_LINE_TOKEN;
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[a][0];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[a][1];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[a][2];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = s->cur_color[0];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[b][0];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[b][1];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = win[b][2];
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = s->cur_color[0];
                    }
                }
                break;
            }

            default:
            {
                /* triangles / quads / polygons: POLYGON_TOKEN with a vertex count */
                uint32_t *idx = NULL;
                GLuint tn = mgl_compat_build_tri_indices(ctx, mode, n, &idx);

                for (GLuint t = 0; t + 2 < tn; t += 3)
                {
                    bool vis = clip_visible(win[idx[t]]) && clip_visible(win[idx[t + 1]]) && clip_visible(win[idx[t + 2]]);
                    if (!vis) continue;

                    if (s->rmode.feedback_count + 2 + 4 * 3 <= s->rmode.feedback_size)
                    {
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = (GLfloat)GL_POLYGON_TOKEN;
                        s->rmode.feedback_buf[s->rmode.feedback_count++] = 3;
                        for (GLuint k = 0; k < 3; k++)
                        {
                            s->rmode.feedback_buf[s->rmode.feedback_count++] = win[idx[t + k]][0];
                            s->rmode.feedback_buf[s->rmode.feedback_count++] = win[idx[t + k]][1];
                            s->rmode.feedback_buf[s->rmode.feedback_count++] = win[idx[t + k]][2];
                            s->rmode.feedback_buf[s->rmode.feedback_count++] = s->cur_color[0];
                        }
                    }
                }

                free(idx);
                break;
            }
        }
    }
    else
    {
        /* select: record hits for primitives that touch the pick volume
         * (the whole window: the GL default) */
        switch (mode)
        {
            case GL_POINTS:
                for (GLuint i = 0; i < n; i++)
                    if (clip_visible(win[i]))
                        select_start_hit(ctx, win[i][2]);
                break;

            case GL_LINES:
                for (GLuint i = 0; i + 1 < n; i += 2)
                    if (clip_visible(win[i]) || clip_visible(win[i + 1]))
                        select_start_hit(ctx, fminf(win[i][2], win[i + 1][2]));
                break;

            case GL_LINE_STRIP: case GL_LINE_LOOP:
                for (GLuint i = 0; i + 1 < n; i++)
                    if (clip_visible(win[i]) || clip_visible(win[i + 1]))
                        select_start_hit(ctx, fminf(win[i][2], win[i + 1][2]));
                break;

            default:
            {
                GLuint tn = 0;
                uint32_t *idx = NULL;
                extern GLuint mgl_compat_build_tri_indices(GLMContext ctx, GLenum mode, GLuint n, uint32_t **idx);

                tn = mgl_compat_build_tri_indices(ctx, mode, n, &idx);

                for (GLuint t = 0; t + 2 < tn; t += 3)
                {
                    float ax = win[idx[t]][0], ay = win[idx[t]][1];
                    float bx = win[idx[t + 1]][0], by = win[idx[t + 1]][1];
                    float cx = win[idx[t + 2]][0], cy = win[idx[t + 2]][1];
                    bool hit = false;

                    /* coarse: bounding box overlapping the window plus a
                     * sample point test (triangle rasterizer substitute) */
                    float minx = fminf(fminf(ax, bx), cx), maxx = fmaxf(fmaxf(ax, bx), cx);
                    float miny = fminf(fminf(ay, by), cy), maxy = fmaxf(fmaxf(ay, by), cy);

                    if (maxx >= 0.0f && minx <= 4096.0f && maxy >= 0.0f && miny <= 4096.0f)
                    {
                        if (point_in_tri(0.5f, 0.5f, ax, ay, bx, by, cx, cy) ||
                            point_in_tri(minx + 0.5f, miny + 0.5f, ax, ay, bx, by, cx, cy) ||
                            clip_visible(win[idx[t]]) || clip_visible(win[idx[t + 1]]) || clip_visible(win[idx[t + 2]]))
                            hit = true;
                    }

                    if (hit)
                        select_start_hit(ctx, fminf(fminf(win[idx[t]][2], win[idx[t + 1]][2]), win[idx[t + 2]][2]));
                }

                free(idx);
                break;
            }
        }

        if (s->rmode.hit_open)
        {
            /* append the name stack */
            for (GLuint i = 0; i < s->rmode.name_depth; i++)
                if (!select_write(ctx, s->rmode.name_stack[i]))
                    break;
            s->rmode.hit_open = GL_FALSE;
            s->rmode.hit_count++;
        }
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

void mglMap1d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint stride, GLint order, const GLdouble *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int idx = eval_target_index(target);
    int comps = eval_comps(target);

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (order < 1 || order > MGLC_MAX_EVAL_ORDER)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!points)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->map1[idx].target = target;
    s->map1[idx].stride = stride ? stride : comps;
    s->map1[idx].order = order;
    s->map1[idx].u1 = u1;
    s->map1[idx].u2 = u2;
    s->map1[idx].enabled = GL_TRUE;

    free(s->map1[idx].fpoints);
    s->map1[idx].fpoints = (GLfloat *)malloc((size_t)order * comps * sizeof(GLfloat));
    if (s->map1[idx].fpoints)
    {
        int es = stride ? stride / (int)sizeof(GLdouble) : comps;
        for (int i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                s->map1[idx].fpoints[i * comps + c] = (GLfloat)points[i * es + c];
    }
}

void mglMap1f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint stride, GLint order, const GLfloat *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int idx = eval_target_index(target);
    int comps = eval_comps(target);

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (order < 1 || order > MGLC_MAX_EVAL_ORDER)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!points)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->map1[idx].target = target;
    s->map1[idx].stride = stride ? stride : comps;
    s->map1[idx].order = order;
    s->map1[idx].u1 = u1;
    s->map1[idx].u2 = u2;
    s->map1[idx].enabled = GL_TRUE;

    free(s->map1[idx].fpoints);
    s->map1[idx].fpoints = (GLfloat *)malloc((size_t)order * comps * sizeof(GLfloat));
    if (s->map1[idx].fpoints)
    {
        int es = stride ? stride / (int)sizeof(GLfloat) : comps;
        for (int i = 0; i < order; i++)
            for (int c = 0; c < comps; c++)
                s->map1[idx].fpoints[i * comps + c] = points[i * es + c];
    }
}

void mglMap2d(GLMContext ctx, GLenum target, GLdouble u1, GLdouble u2, GLint ustride, GLint uorder,
              GLdouble v1, GLdouble v2, GLint vstride, GLint vorder, const GLdouble *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int idx = eval_target_index(target);
    int comps = eval_comps(target);

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (uorder < 1 || uorder > MGLC_MAX_EVAL_ORDER || vorder < 1 || vorder > MGLC_MAX_EVAL_ORDER)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!points)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->map2[idx].target = target;
    s->map2[idx].ustride = ustride ? ustride : comps * vorder;
    s->map2[idx].vstride = vstride ? vstride : comps;
    s->map2[idx].uorder = uorder;
    s->map2[idx].vorder = vorder;
    s->map2[idx].u1 = u1;
    s->map2[idx].u2 = u2;
    s->map2[idx].v1 = v1;
    s->map2[idx].v2 = v2;
    s->map2[idx].enabled = GL_TRUE;
    s->map2[idx].type = GL_DOUBLE;

    free(s->map2[idx].fpoints);
    s->map2[idx].fpoints = (GLfloat *)malloc((size_t)uorder * vorder * comps * sizeof(GLfloat));
    if (s->map2[idx].fpoints)
    {
        int ue = ustride ? ustride / (int)sizeof(GLdouble) : comps * vorder;
        int ve = vstride ? vstride / (int)sizeof(GLdouble) : comps;
        for (int i = 0; i < uorder; i++)
            for (int j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    s->map2[idx].fpoints[(i * vorder + j) * comps + c] =
                        (GLfloat)points[i * ue + j * ve + c];
    }
}

void mglMap2f(GLMContext ctx, GLenum target, GLfloat u1, GLfloat u2, GLint ustride, GLint uorder,
              GLfloat v1, GLfloat v2, GLint vstride, GLint vorder, const GLfloat *points)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int idx = eval_target_index(target);
    int comps = eval_comps(target);

    if (idx < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (uorder < 1 || uorder > MGLC_MAX_EVAL_ORDER || vorder < 1 || vorder > MGLC_MAX_EVAL_ORDER)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->map2[idx].target = target;
    s->map2[idx].ustride = ustride ? ustride : comps * vorder;
    s->map2[idx].vstride = vstride ? vstride : comps;
    s->map2[idx].uorder = uorder;
    s->map2[idx].vorder = vorder;
    s->map2[idx].u1 = u1;
    s->map2[idx].u2 = u2;
    s->map2[idx].v1 = v1;
    s->map2[idx].v2 = v2;
    s->map2[idx].enabled = GL_TRUE;
    s->map2[idx].type = GL_FLOAT;

    free(s->map2[idx].fpoints);
    s->map2[idx].fpoints = (GLfloat *)malloc((size_t)uorder * vorder * comps * sizeof(GLfloat));
    if (s->map2[idx].fpoints)
    {
        int ue = ustride ? ustride / (int)sizeof(GLfloat) : comps * vorder;
        int ve = vstride ? vstride / (int)sizeof(GLfloat) : comps;
        for (int i = 0; i < uorder; i++)
            for (int j = 0; j < vorder; j++)
                for (int c = 0; c < comps; c++)
                    s->map2[idx].fpoints[(i * vorder + j) * comps + c] =
                        points[i * ue + j * ve + c];
    }
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

/* Bernstein evaluation of a 1D map */
static void eval1(GLMContext ctx, int idx, float u, float *out)
{
    struct MGLExtState_compat *s = CS();
    int n = s->map1[idx].order;
    int comps = eval_comps(s->map1[idx].target);
    double du = (s->map1[idx].u2 - s->map1[idx].u1);
    double t = (du != 0.0) ? (u - s->map1[idx].u1) / du : 0.0;

    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    /* de Casteljau over the control points */
    double tmp[MGLC_MAX_EVAL_ORDER * 4];
    memcpy(tmp, s->map1[idx].fpoints, (size_t)n * comps * sizeof(GLfloat));

    for (int k = n - 1; k > 0; k--)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < comps; c++)
                tmp[i * comps + c] = (1.0 - t) * tmp[i * comps + c] + t * tmp[(i + 1) * comps + c];

    for (int c = 0; c < comps; c++)
        out[c] = (float)tmp[c];
}

static void eval2(GLMContext ctx, int idx, float u, float v, float *out)
{
    struct MGLExtState_compat *s = CS();
    int un = s->map2[idx].uorder, vn = s->map2[idx].vorder;
    int comps = eval_comps(s->map2[idx].target);
    double du = (s->map2[idx].u2 - s->map2[idx].u1), dv = (s->map2[idx].v2 - s->map2[idx].v1);
    double tu = (du != 0.0) ? (u - s->map2[idx].u1) / du : 0.0;
    double tv = (dv != 0.0) ? (v - s->map2[idx].v1) / dv : 0.0;

    if (tu < 0.0) tu = 0.0; if (tu > 1.0) tu = 1.0;
    if (tv < 0.0) tv = 0.0; if (tv > 1.0) tv = 1.0;

    double row[MGLC_MAX_EVAL_ORDER * 4];
    double col[MGLC_MAX_EVAL_ORDER * 4];

    /* evaluate each u-row down to one point, then de Casteljau along v */
    for (int j = 0; j < vn; j++)
    {
        memcpy(row, &s->map2[idx].fpoints[(size_t)j * un * comps], (size_t)un * comps * sizeof(GLfloat));

        for (int k = un - 1; k > 0; k--)
            for (int i = 0; i < k; i++)
                for (int c = 0; c < comps; c++)
                    row[i * comps + c] = (1.0 - tu) * row[i * comps + c] + tu * row[(i + 1) * comps + c];

        for (int c = 0; c < comps; c++)
            col[j * comps + c] = row[c];
    }

    for (int k = vn - 1; k > 0; k--)
        for (int i = 0; i < k; i++)
            for (int c = 0; c < comps; c++)
                col[i * comps + c] = (1.0 - tv) * col[i * comps + c] + tv * col[(i + 1) * comps + c];

    for (int c = 0; c < comps; c++)
        out[c] = (float)col[c];
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

    if (s->map1[7].enabled && s->map1[7].target == GL_MAP1_COLOR_4)
    {
        eval1(ctx, 7, u, v);
        mglColor4f(ctx, v[0], v[1], v[2], v[3]);
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

    if (s->map2[7].enabled && s->map2[7].target == GL_MAP2_COLOR_4)
    {
        eval2(ctx, 7, u, v, out);
        mglColor4f(ctx, out[0], out[1], out[2], out[3]);
    }
    if (s->map2[8].enabled && s->map2[8].target == GL_MAP2_NORMAL)
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
        eval2(ctx, 6, u, v, out);
        mglVertex4f(ctx, out[0], out[1], out[2], out[3]);
    }
    else if (s->map2[5].enabled && s->map2[5].target == GL_MAP2_VERTEX_3)
    {
        eval2(ctx, 5, u, v, out);
        mglVertex3f(ctx, out[0], out[1], out[2]);
    }
}

void mglMapGrid1d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2)
{    MGLC_COMPAT_GUARD(ctx);

    if (un < 1)
        ERROR_RETURN(GL_INVALID_VALUE);

    CS()->map_grid1_n = un;
    CS()->map_grid1_u1 = u1;
    CS()->map_grid1_u2 = u2;
}

void mglMapGrid1f(GLMContext ctx, GLint un, GLfloat u1, GLfloat u2)
{    MGLC_COMPAT_GUARD(ctx);

    mglMapGrid1d(ctx, un, u1, u2);
}

void mglMapGrid2d(GLMContext ctx, GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2)
{    MGLC_COMPAT_GUARD(ctx);

    if (un < 1 || vn < 1)
        ERROR_RETURN(GL_INVALID_VALUE);

    CS()->map_grid2_nu = un;
    CS()->map_grid2_u1 = u1;
    CS()->map_grid2_u2 = u2;
    CS()->map_grid2_nv = vn;
    CS()->map_grid2_v1 = v1;
    CS()->map_grid2_v2 = v2;
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

    double u = s->map_grid1_u1 + (double)i * (s->map_grid1_u2 - s->map_grid1_u1) / s->map_grid1_n;
    mglEvalCoord1d(ctx, u);
}

void mglEvalPoint2(GLMContext ctx, GLint i, GLint j)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    double u = s->map_grid2_u1 + (double)i * (s->map_grid2_u2 - s->map_grid2_u1) / s->map_grid2_nu;
    double v = s->map_grid2_v1 + (double)j * (s->map_grid2_v2 - s->map_grid2_v1) / s->map_grid2_nv;
    mglEvalCoord2d(ctx, u, v);
}

void mglEvalMesh2(GLMContext ctx, GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (mode != GL_POINT && mode != GL_LINE && mode != GL_FILL)
        ERROR_RETURN(GL_INVALID_ENUM);

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

    /* every texture is resident (it lives in the MTLTexture) */
    for (GLsizei i = 0; i < n; i++)
        residences[i] = GL_TRUE;

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
