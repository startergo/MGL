/*
 * compat_state.c — fixed-function state entry points (compat domain):
 * lights, materials, texgen, texenv, fog, alpha test, shade model,
 * stipple, clip planes, polygon/point compat behaviour.
 *
 * State only; the rendering effect is produced by the generated shaders
 * (compat_ffp.c) and, for stipple, by the draw-time index conversion
 * (compat_vb.c).
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

#define CS() (ctx->state.ext_compat)

void mgl_compat_fog_recompute(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (s->fog_end == s->fog_start)
        s->fog_scale = 1.0f;
    else
        s->fog_scale = 1.0f / (s->fog_end - s->fog_start);

    mgl_compat_dirty(ctx);
}

#pragma mark - lights

static int light_no(GLMContext ctx, GLenum light)
{
    if (light >= GL_LIGHT0 && light < GL_LIGHT0 + MGLC_MAX_LIGHTS)
        return (int)(light - GL_LIGHT0);
    return -1;
}

void mglLightf(GLMContext ctx, GLenum light, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    int i = light_no(ctx, light);
    MGLCLight *l;

    if (i < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    l = &CS()->lights[i];

    switch (pname)
    {
        case GL_SPOT_EXPONENT:
            if (param < 0.0f || param > 128.0f) ERROR_RETURN(GL_INVALID_VALUE);
            l->spot_exponent = param;
            break;
        case GL_SPOT_CUTOFF:
            if ((param < 0.0f || param > 90.0f) && param != 180.0f) ERROR_RETURN(GL_INVALID_VALUE);
            l->spot_cutoff = param;
            break;
        case GL_CONSTANT_ATTENUATION:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            l->constant_attenuation = param;
            break;
        case GL_LINEAR_ATTENUATION:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            l->linear_attenuation = param;
            break;
        case GL_QUADRATIC_ATTENUATION:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            l->quadratic_attenuation = param;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_dirty(ctx);
}

void mglLighti(GLMContext ctx, GLenum light, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglLightf(ctx, light, pname, (GLfloat)param);
}

void mglLightfv(GLMContext ctx, GLenum light, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    int i = light_no(ctx, light);
    struct MGLExtState_compat *s = CS();
    MGLCLight *l;
    GLfloat v[4];

    if (i < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    l = &s->lights[i];

    /* POSITION and SPOT_DIRECTION are transformed by the current modelview
     * matrix at call time (GL 1.0 compatibility rule) */
    switch (pname)
    {
        case GL_POSITION:
        case GL_SPOT_DIRECTION:
        {
            GLdouble *m = s->modelview.stack[s->modelview.depth].m;
            v[0] = params[0]; v[1] = params[1]; v[2] = params[2]; v[3] = (pname == GL_POSITION) ? params[3] : 0.0f;
            for (int r = 0; r < 4; r++)
            {
                GLfloat t = 0;
                for (int k = 0; k < 4; k++)
                    t += (GLfloat)m[k * 4 + r] * v[k];
                if (pname == GL_POSITION)
                    ((GLfloat *)l->position)[r] = t;
                else
                    ((GLfloat *)l->spot_direction)[r] = t;
            }
            mgl_compat_dirty(ctx);
            return;
        }
    }

    switch (pname)
    {
        case GL_AMBIENT: memcpy(l->ambient, params, 4 * sizeof(GLfloat)); break;
        case GL_DIFFUSE: memcpy(l->diffuse, params, 4 * sizeof(GLfloat)); break;
        case GL_SPECULAR: memcpy(l->specular, params, 4 * sizeof(GLfloat)); break;
        case GL_SPOT_EXPONENT:
        case GL_SPOT_CUTOFF:
        case GL_CONSTANT_ATTENUATION:
        case GL_LINEAR_ATTENUATION:
        case GL_QUADRATIC_ATTENUATION:
            mglLightf(ctx, light, pname, params[0]);
            return;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_dirty(ctx);
}

void mglLightiv(GLMContext ctx, GLenum light, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4] = { 0, 0, 0, 0 };

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    f[0] = (GLfloat)params[0]; f[1] = (GLfloat)params[1];
    f[2] = (GLfloat)params[2]; f[3] = (GLfloat)params[3];
    mglLightfv(ctx, light, pname, f);
}

static void get_light(GLMContext ctx, GLenum light, GLenum pname, float *out, int *n)
{
    int i = light_no(ctx, light);
    MGLCLight *l;

    *n = 0;
    if (i < 0)
        return;

    l = &CS()->lights[i];

    switch (pname)
    {
        case GL_AMBIENT: memcpy(out, l->ambient, 4 * sizeof(float)); *n = 4; return;
        case GL_DIFFUSE: memcpy(out, l->diffuse, 4 * sizeof(float)); *n = 4; return;
        case GL_SPECULAR: memcpy(out, l->specular, 4 * sizeof(float)); *n = 4; return;
        case GL_POSITION: memcpy(out, l->position, 4 * sizeof(float)); *n = 4; return;
        case GL_SPOT_DIRECTION: memcpy(out, l->spot_direction, 3 * sizeof(float)); *n = 3; return;
        case GL_SPOT_EXPONENT: out[0] = l->spot_exponent; *n = 1; return;
        case GL_SPOT_CUTOFF: out[0] = l->spot_cutoff; *n = 1; return;
        case GL_CONSTANT_ATTENUATION: out[0] = l->constant_attenuation; *n = 1; return;
        case GL_LINEAR_ATTENUATION: out[0] = l->linear_attenuation; *n = 1; return;
        case GL_QUADRATIC_ATTENUATION: out[0] = l->quadratic_attenuation; *n = 1; return;
    }
}

void mglGetLightfv(GLMContext ctx, GLenum light, GLenum pname, GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    float v[4];
    int n = 0;

    if (!params)
        return;

    get_light(ctx, light, pname, v, &n);

    if (!n)
        ERROR_RETURN(GL_INVALID_ENUM);

    for (int i = 0; i < n; i++)
        params[i] = v[i];
}

void mglGetLightiv(GLMContext ctx, GLenum light, GLenum pname, GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    float v[4];
    int n = 0;

    if (!params)
        return;

    get_light(ctx, light, pname, v, &n);

    if (!n)
        ERROR_RETURN(GL_INVALID_ENUM);

    for (int i = 0; i < n; i++)
        params[i] = (GLint)v[i];
}

#pragma mark - light model

void mglLightModelf(GLMContext ctx, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    switch (pname)
    {
        case GL_LIGHT_MODEL_LOCAL_VIEWER:
            s->local_viewer = param != 0.0f;
            break;
        case GL_LIGHT_MODEL_TWO_SIDE:
            s->two_sided = param != 0.0f;
            break;
        case GL_LIGHT_MODEL_COLOR_CONTROL:
            if (param != GL_SINGLE_COLOR && param != GL_SEPARATE_SPECULAR_COLOR)
                ERROR_RETURN(GL_INVALID_ENUM);
            s->separate_specular = param == GL_SEPARATE_SPECULAR_COLOR;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_dirty(ctx);
}

void mglLightModeli(GLMContext ctx, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglLightModelf(ctx, pname, (GLfloat)param);
}

void mglLightModelfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (pname == GL_LIGHT_MODEL_AMBIENT)
    {
        memcpy(s->light_model_ambient, params, 4 * sizeof(GLfloat));
        mgl_compat_dirty(ctx);
        return;
    }

    mglLightModelf(ctx, pname, params[0]);
}

void mglLightModeliv(GLMContext ctx, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    f[0] = (GLfloat)params[0]; f[1] = (GLfloat)params[1];
    f[2] = (GLfloat)params[2]; f[3] = (GLfloat)params[3];
    mglLightModelfv(ctx, pname, f);
}

#pragma mark - materials

static MGLCMaterial *material(GLMContext ctx, GLenum face)
{
    struct MGLExtState_compat *s = CS();

    switch (face)
    {
        case GL_FRONT: return &s->material_front;
        case GL_BACK: return &s->material_back;
        case GL_FRONT_AND_BACK: return &s->material_front;
    }

    return NULL;
}

void mglMaterialf(GLMContext ctx, GLenum face, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMaterial *m = material(ctx, face);
    struct MGLExtState_compat *s = CS();

    if (!m)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (pname != GL_SHININESS)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (param < 0.0f || param > 128.0f)
        ERROR_RETURN(GL_INVALID_VALUE);

    m->shininess = param;
    if (face == GL_FRONT_AND_BACK)
        s->material_back.shininess = param;

    mgl_compat_dirty(ctx);
}

void mglMateriali(GLMContext ctx, GLenum face, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglMaterialf(ctx, face, pname, (GLfloat)param);
}

void mglMaterialfv(GLMContext ctx, GLenum face, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    MGLCMaterial *m = material(ctx, face);
    struct MGLExtState_compat *s = CS();

    if (!m)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    switch (pname)
    {
        case GL_AMBIENT: memcpy(m->ambient, params, 4 * sizeof(GLfloat));
            if (face == GL_FRONT_AND_BACK) memcpy(s->material_back.ambient, params, 4 * sizeof(GLfloat));
            break;
        case GL_DIFFUSE: memcpy(m->diffuse, params, 4 * sizeof(GLfloat));
            if (face == GL_FRONT_AND_BACK) memcpy(s->material_back.diffuse, params, 4 * sizeof(GLfloat));
            break;
        case GL_SPECULAR: memcpy(m->specular, params, 4 * sizeof(GLfloat));
            if (face == GL_FRONT_AND_BACK) memcpy(s->material_back.specular, params, 4 * sizeof(GLfloat));
            break;
        case GL_EMISSION: memcpy(m->emission, params, 4 * sizeof(GLfloat));
            if (face == GL_FRONT_AND_BACK) memcpy(s->material_back.emission, params, 4 * sizeof(GLfloat));
            break;
        case GL_SHININESS:
            mglMaterialf(ctx, face, pname, params[0]);
            return;
        case GL_AMBIENT_AND_DIFFUSE:
            memcpy(m->ambient, params, 4 * sizeof(GLfloat));
            memcpy(m->diffuse, params, 4 * sizeof(GLfloat));
            if (face == GL_FRONT_AND_BACK)
            {
                memcpy(s->material_back.ambient, params, 4 * sizeof(GLfloat));
                memcpy(s->material_back.diffuse, params, 4 * sizeof(GLfloat));
            }
            break;
        case GL_COLOR_INDEXES:
            for (int i = 0; i < 3; i++) m->color_indexes[i] = (GLint)params[i];
            if (face == GL_FRONT_AND_BACK)
                for (int i = 0; i < 3; i++) s->material_back.color_indexes[i] = (GLint)params[i];
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_dirty(ctx);
}

void mglMaterialiv(GLMContext ctx, GLenum face, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (int i = 0; i < 4; i++)
        f[i] = (GLfloat)params[i];

    mglMaterialfv(ctx, face, pname, f);
}

static void get_material(GLMContext ctx, GLenum face, GLenum pname, float *out, int *n)
{
    MGLCMaterial *m = material(ctx, face);

    *n = 0;
    if (!m)
        return;

    switch (pname)
    {
        case GL_AMBIENT: memcpy(out, m->ambient, 4 * sizeof(float)); *n = 4; return;
        case GL_DIFFUSE: memcpy(out, m->diffuse, 4 * sizeof(float)); *n = 4; return;
        case GL_SPECULAR: memcpy(out, m->specular, 4 * sizeof(float)); *n = 4; return;
        case GL_EMISSION: memcpy(out, m->emission, 4 * sizeof(float)); *n = 4; return;
        case GL_SHININESS: out[0] = m->shininess; *n = 1; return;
        case GL_COLOR_INDEXES:
            for (int i = 0; i < 3; i++) out[i] = (float)m->color_indexes[i];
            *n = 3;
            return;
    }
}

void mglGetMaterialfv(GLMContext ctx, GLenum face, GLenum pname, GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    float v[4];
    int n = 0;

    if (face != GL_FRONT && face != GL_BACK)
        ERROR_RETURN(GL_INVALID_ENUM);

    get_material(ctx, face, pname, v, &n);

    if (!n)
        ERROR_RETURN(GL_INVALID_ENUM);

    for (int i = 0; i < n; i++)
        params[i] = v[i];
}

void mglGetMaterialiv(GLMContext ctx, GLenum face, GLenum pname, GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    float v[4];
    int n = 0;

    if (face != GL_FRONT && face != GL_BACK)
        ERROR_RETURN(GL_INVALID_ENUM);

    get_material(ctx, face, pname, v, &n);

    if (!n)
        ERROR_RETURN(GL_INVALID_ENUM);

    for (int i = 0; i < n; i++)
        params[i] = (GLint)v[i];
}

void mglColorMaterial(GLMContext ctx, GLenum face, GLenum mode)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    switch (face)
    {
        case GL_FRONT: case GL_BACK: case GL_FRONT_AND_BACK: break;
        default: ERROR_RETURN(GL_INVALID_ENUM);
    }

    switch (mode)
    {
        case GL_EMISSION: case GL_AMBIENT: case GL_DIFFUSE:
        case GL_SPECULAR: case GL_AMBIENT_AND_DIFFUSE: break;
        default: ERROR_RETURN(GL_INVALID_ENUM);
    }

    s->color_material_face = face;
    s->color_material_mode = mode;

    mgl_compat_dirty(ctx);
}

#pragma mark - texgen

static int texgen_coord(GLenum coord)
{
    switch (coord)
    {
        case GL_S: return 0;
        case GL_T: return 1;
        case GL_R: return 2;
        case GL_Q: return 3;
    }
    return -1;
}

void mglTexGend(GLMContext ctx, GLenum coord, GLenum pname, GLdouble param)
{    MGLC_COMPAT_GUARD(ctx);

    mglTexGenf(ctx, coord, pname, (GLfloat)param);
}

void mglTexGenf(GLMContext ctx, GLenum coord, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    if (texgen_coord(coord) < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (pname != GL_TEXTURE_GEN_MODE)
        ERROR_RETURN(GL_INVALID_ENUM);

    switch ((GLenum)param)
    {
        case GL_OBJECT_LINEAR: case GL_EYE_LINEAR:
        case GL_SPHERE_MAP: case GL_NORMAL_MAP: case GL_REFLECTION_MAP:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
    CS()->texgen[unit].mode[texgen_coord(coord)] = (GLenum)param;
    mgl_compat_dirty(ctx);
}

void mglTexGeni(GLMContext ctx, GLenum coord, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglTexGenf(ctx, coord, pname, (GLfloat)param);
}

void mglTexGendv(GLMContext ctx, GLenum coord, GLenum pname, const GLdouble *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (int i = 0; i < 4; i++)
        f[i] = (GLfloat)params[i];

    mglTexGenfv(ctx, coord, pname, f);
}

void mglTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    int c = texgen_coord(coord);

    if (c < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
    struct MGLExtState_compat *s = CS();

    switch (pname)
    {
        case GL_TEXTURE_GEN_MODE:
            mglTexGenf(ctx, coord, pname, params[0]);
            return;

        case GL_OBJECT_PLANE:
            memcpy(s->texgen[unit].object_plane[c], params, 4 * sizeof(GLfloat));
            break;

        case GL_EYE_PLANE:
        {
            /* stored transformed by the inverse of the current modelview */
            GLdouble inv[16], *m = s->modelview.stack[s->modelview.depth].m;
            if (!mglc_matrix_invert(inv, m))
                mglc_matrix_identity(inv);
            for (int r = 0; r < 4; r++)
            {
                GLfloat t = 0;
                for (int k = 0; k < 4; k++)
                    t += (GLfloat)inv[r * 4 + k] * params[k];   /* row of inverse */
                s->texgen[unit].eye_plane[c][r] = t;
            }
            break;
        }

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_dirty(ctx);
}

void mglTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (int i = 0; i < 4; i++)
        f[i] = (GLfloat)params[i];

    mglTexGenfv(ctx, coord, pname, f);
}

static void get_texgen(GLMContext ctx, GLenum coord, GLenum pname, double *out, int *n)
{
    int c = texgen_coord(coord);
    GLuint unit;

    *n = 0;
    if (c < 0)
        return;

    unit = STATE(active_texture) % MGLC_MAX_COORDS;
    MGLCTexGen *g = &CS()->texgen[unit];

    switch (pname)
    {
        case GL_TEXTURE_GEN_MODE: out[0] = (double)g->mode[c]; *n = 1; return;
        case GL_OBJECT_PLANE:
            for (int i = 0; i < 4; i++) out[i] = g->object_plane[c][i];
            *n = 4;
            return;
        case GL_EYE_PLANE:
            for (int i = 0; i < 4; i++) out[i] = g->eye_plane[c][i];
            *n = 4;
            return;
    }
}

void mglGetTexGendv(GLMContext ctx, GLenum coord, GLenum pname, GLdouble *params)
{    MGLC_COMPAT_GUARD(ctx);

    double v[4];
    int n = 0;

    get_texgen(ctx, coord, pname, v, &n);
    if (!n) ERROR_RETURN(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) params[i] = v[i];
}

void mglGetTexGenfv(GLMContext ctx, GLenum coord, GLenum pname, GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    double v[4];
    int n = 0;

    get_texgen(ctx, coord, pname, v, &n);
    if (!n) ERROR_RETURN(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) params[i] = (GLfloat)v[i];
}

void mglGetTexGeniv(GLMContext ctx, GLenum coord, GLenum pname, GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    double v[4];
    int n = 0;

    get_texgen(ctx, coord, pname, v, &n);
    if (!n) ERROR_RETURN(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) params[i] = (GLint)v[i];
}

#pragma mark - texenv

void mglTexEnvf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
    MGLCTexEnv *e = &s->texenv[unit];

    if (target == GL_POINT_SPRITE)
    {
        if (pname != GL_COORD_REPLACE)
            ERROR_RETURN(GL_INVALID_ENUM);
        s->coord_replace[unit] = param != 0.0f;
        mgl_compat_dirty(ctx);
        return;
    }

    if (target != GL_TEXTURE_ENV)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (pname != GL_TEXTURE_ENV_MODE)
        ERROR_RETURN(GL_INVALID_ENUM);

    switch ((GLenum)param)
    {
        case GL_MODULATE: case GL_DECAL: case GL_BLEND:
        case GL_REPLACE: case GL_ADD: case GL_COMBINE:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    e->env_mode = (GLenum)param;
    mgl_compat_dirty(ctx);
}

void mglTexEnvi(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglTexEnvf(ctx, target, pname, (GLfloat)param);
}

/* src / operand helper: index 0..5 -> src[6] */
static bool set_env_enum(MGLCTexEnv *e, GLenum pname, GLenum val)
{
    switch (pname)
    {
        case GL_COMBINE_RGB: e->combine_rgb = val; return true;
        case GL_COMBINE_ALPHA: e->combine_alpha = val; return true;
        case GL_SRC0_RGB: e->src[MGLC_ENV_SRC_RGB(0)] = val; return true;
        case GL_SRC1_RGB: e->src[MGLC_ENV_SRC_RGB(1)] = val; return true;
        case GL_SRC2_RGB: e->src[MGLC_ENV_SRC_RGB(2)] = val; return true;
        case GL_SRC0_ALPHA: e->src[MGLC_ENV_SRC_A(0)] = val; return true;
        case GL_SRC1_ALPHA: e->src[MGLC_ENV_SRC_A(1)] = val; return true;
        case GL_SRC2_ALPHA: e->src[MGLC_ENV_SRC_A(2)] = val; return true;
        case GL_OPERAND0_RGB: e->op_rgb[0] = val; return true;
        case GL_OPERAND1_RGB: e->op_rgb[1] = val; return true;
        case GL_OPERAND2_RGB: e->op_rgb[2] = val; return true;
        case GL_OPERAND0_ALPHA: e->op_alpha[0] = val; return true;
        case GL_OPERAND1_ALPHA: e->op_alpha[1] = val; return true;
        case GL_OPERAND2_ALPHA: e->op_alpha[2] = val; return true;
    }
    return false;
}

void mglTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
    MGLCTexEnv *e = &s->texenv[unit];

    if (target == GL_POINT_SPRITE)
    {
        if (pname != GL_COORD_REPLACE)
            ERROR_RETURN(GL_INVALID_ENUM);
        if (!params) ERROR_RETURN(GL_INVALID_VALUE);
        s->coord_replace[unit] = params[0] != 0.0f;
        mgl_compat_dirty(ctx);
        return;
    }

    if (target != GL_TEXTURE_ENV)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    switch (pname)
    {
        case GL_TEXTURE_ENV_MODE:
            mglTexEnvf(ctx, target, pname, params[0]);
            return;

        case GL_TEXTURE_ENV_COLOR:
            memcpy(e->env_color, params, 4 * sizeof(GLfloat));
            break;

        case GL_RGB_SCALE:
            if (params[0] != 1.0f && params[0] != 2.0f && params[0] != 4.0f)
                ERROR_RETURN(GL_INVALID_VALUE);
            e->scale_rgb = params[0];
            break;

        case GL_ALPHA_SCALE:
            if (params[0] != 1.0f && params[0] != 2.0f && params[0] != 4.0f)
                ERROR_RETURN(GL_INVALID_VALUE);
            e->scale_alpha = params[0];
            break;

        default:
        {
            static const GLenum valid_src[] = {
                GL_TEXTURE, GL_CONSTANT, GL_PRIMARY_COLOR, GL_PREVIOUS,
                GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3,
                GL_TEXTURE4, GL_TEXTURE5, GL_TEXTURE6, GL_TEXTURE7,
            };
            static const GLenum valid_op_rgb[] = { GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA };
            static const GLenum valid_op_a[] = { GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA };
            static const GLenum valid_combine[] = {
                GL_REPLACE, GL_MODULATE, GL_ADD, GL_ADD_SIGNED,
                GL_INTERPOLATE, GL_SUBTRACT, GL_DOT3_RGB, GL_DOT3_RGBA
            };
            GLenum v = (GLenum)params[0];
            bool ok = false;

            if (set_env_enum(e, pname, v))
            {
                ok = true;
                if (pname >= GL_SRC0_RGB && pname <= GL_SRC2_ALPHA)
                {
                    ok = false;
                    for (size_t i = 0; i < sizeof valid_src / sizeof valid_src[0]; i++)
                        if (v == valid_src[i]) ok = true;
                }
                else if (pname >= GL_OPERAND0_RGB && pname <= GL_OPERAND2_RGB)
                {
                    ok = false;
                    for (size_t i = 0; i < sizeof valid_op_rgb / sizeof valid_op_rgb[0]; i++)
                        if (v == valid_op_rgb[i]) ok = true;
                }
                else if (pname >= GL_OPERAND0_ALPHA && pname <= GL_OPERAND2_ALPHA)
                {
                    ok = false;
                    for (size_t i = 0; i < sizeof valid_op_a / sizeof valid_op_a[0]; i++)
                        if (v == valid_op_a[i]) ok = true;
                }
                else if (pname == GL_COMBINE_RGB || pname == GL_COMBINE_ALPHA)
                {
                    ok = false;
                    for (size_t i = 0; i < sizeof valid_combine / sizeof valid_combine[0]; i++)
                        if (v == valid_combine[i]) ok = true;
                }
            }

            if (!ok)
                ERROR_RETURN(GL_INVALID_ENUM);
            break;
        }
    }

    mgl_compat_dirty(ctx);
}

void mglTexEnviv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (int i = 0; i < 4; i++)
        f[i] = (GLfloat)params[i];

    mglTexEnvfv(ctx, target, pname, f);
}

static void get_texenv(GLMContext ctx, GLenum target, GLenum pname, double *out, int *n)
{
    struct MGLExtState_compat *s = CS();
    GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
    MGLCTexEnv *e = &s->texenv[unit];

    *n = 0;

    if (target == GL_POINT_SPRITE)
    {
        if (pname == GL_COORD_REPLACE)
        {
            out[0] = s->coord_replace[unit] ? 1.0 : 0.0;
            *n = 1;
        }
        return;
    }

    if (target != GL_TEXTURE_ENV)
        return;

    switch (pname)
    {
        case GL_TEXTURE_ENV_MODE: out[0] = (double)e->env_mode; *n = 1; return;
        case GL_TEXTURE_ENV_COLOR:
            for (int i = 0; i < 4; i++) out[i] = e->env_color[i];
            *n = 4;
            return;
        case GL_COMBINE_RGB: out[0] = (double)e->combine_rgb; *n = 1; return;
        case GL_COMBINE_ALPHA: out[0] = (double)e->combine_alpha; *n = 1; return;
        case GL_SRC0_RGB: out[0] = (double)e->src[MGLC_ENV_SRC_RGB(0)]; *n = 1; return;
        case GL_SRC1_RGB: out[0] = (double)e->src[MGLC_ENV_SRC_RGB(1)]; *n = 1; return;
        case GL_SRC2_RGB: out[0] = (double)e->src[MGLC_ENV_SRC_RGB(2)]; *n = 1; return;
        case GL_SRC0_ALPHA: out[0] = (double)e->src[MGLC_ENV_SRC_A(0)]; *n = 1; return;
        case GL_SRC1_ALPHA: out[0] = (double)e->src[MGLC_ENV_SRC_A(1)]; *n = 1; return;
        case GL_SRC2_ALPHA: out[0] = (double)e->src[MGLC_ENV_SRC_A(2)]; *n = 1; return;
        case GL_OPERAND0_RGB: out[0] = (double)e->op_rgb[0]; *n = 1; return;
        case GL_OPERAND1_RGB: out[0] = (double)e->op_rgb[1]; *n = 1; return;
        case GL_OPERAND2_RGB: out[0] = (double)e->op_rgb[2]; *n = 1; return;
        case GL_OPERAND0_ALPHA: out[0] = (double)e->op_alpha[0]; *n = 1; return;
        case GL_OPERAND1_ALPHA: out[0] = (double)e->op_alpha[1]; *n = 1; return;
        case GL_OPERAND2_ALPHA: out[0] = (double)e->op_alpha[2]; *n = 1; return;
        case GL_RGB_SCALE: out[0] = (double)e->scale_rgb; *n = 1; return;
        case GL_ALPHA_SCALE: out[0] = (double)e->scale_alpha; *n = 1; return;
    }
}

void mglGetTexEnvfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    double v[4];
    int n = 0;

    get_texenv(ctx, target, pname, v, &n);
    if (!n) ERROR_RETURN(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) params[i] = (GLfloat)v[i];
}

void mglGetTexEnviv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    double v[4];
    int n = 0;

    get_texenv(ctx, target, pname, v, &n);
    if (!n) ERROR_RETURN(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) params[i] = (GLint)v[i];
}

#pragma mark - fog

void mglFogf(GLMContext ctx, GLenum pname, GLfloat param)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    switch (pname)
    {
        case GL_FOG_MODE:
            if (param != GL_LINEAR && param != GL_EXP && param != GL_EXP2)
                ERROR_RETURN(GL_INVALID_ENUM);
            s->fog_mode = (GLenum)param;
            break;
        case GL_FOG_DENSITY:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            s->fog_density = param;
            break;
        case GL_FOG_START: s->fog_start = param; break;
        case GL_FOG_END: s->fog_end = param; break;
        case GL_FOG_INDEX: s->fog_index = param; break;
        case GL_FOG_COORD_SRC:
            if (param != GL_FRAGMENT_DEPTH && param != GL_FOG_COORD)
                ERROR_RETURN(GL_INVALID_ENUM);
            s->fog_coord_src = (GLenum)param;
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    mgl_compat_fog_recompute(ctx);
}

void mglFogi(GLMContext ctx, GLenum pname, GLint param)
{    MGLC_COMPAT_GUARD(ctx);

    mglFogf(ctx, pname, (GLfloat)param);
}

void mglFogfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (pname == GL_FOG_COLOR)
    {
        memcpy(s->fog_color, params, 4 * sizeof(GLfloat));
        mgl_compat_dirty(ctx);
        return;
    }

    mglFogf(ctx, pname, params[0]);
}

void mglFogiv(GLMContext ctx, GLenum pname, const GLint *params)
{    MGLC_COMPAT_GUARD(ctx);

    GLfloat f[4];

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (int i = 0; i < 4; i++)
        f[i] = (GLfloat)params[i];

    mglFogfv(ctx, pname, f);
}

#pragma mark - alpha test / shade model / stipple / clip planes

void mglAlphaFunc(GLMContext ctx, GLenum func, GLfloat ref)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    switch (func)
    {
        case GL_NEVER: case GL_LESS: case GL_EQUAL: case GL_LEQUAL:
        case GL_GREATER: case GL_NOTEQUAL: case GL_GEQUAL: case GL_ALWAYS:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    if (ref < 0.0f) ref = 0.0f;
    if (ref > 1.0f) ref = 1.0f;

    s->alpha_func = func;
    s->alpha_ref = ref;

    mgl_compat_dirty(ctx);
}

void mglShadeModel(GLMContext ctx, GLenum mode)
{    MGLC_COMPAT_GUARD(ctx);

    if (mode != GL_SMOOTH && mode != GL_FLAT)
        ERROR_RETURN(GL_INVALID_ENUM);

    CS()->shade_model = mode;
    mgl_compat_dirty(ctx);
}

void mglLineStipple(GLMContext ctx, GLint factor, GLushort pattern)
{    MGLC_COMPAT_GUARD(ctx);

    if (factor < 1 || factor > 256)
        ERROR_RETURN(GL_INVALID_VALUE);

    CS()->line_stipple_pattern = pattern;
    CS()->line_stipple_repeat = factor;
    mgl_compat_dirty(ctx);
}

void mglPolygonStipple(GLMContext ctx, const GLubyte *mask)
{    MGLC_COMPAT_GUARD(ctx);

    if (!mask)
        ERROR_RETURN(GL_INVALID_VALUE);

    /* GL rows are LSB-first starting at the bottom row; the pattern is used
     * from the bottom-left of the window (see compat_ffp.c) */
    for (int row = 0; row < 32; row++)
    {
        GLubyte *dst = CS()->polygon_stipple_pattern + (31 - row) * 4;
        const GLubyte *src = mask + row * 4;
        dst[0] = src[3]; dst[1] = src[2]; dst[2] = src[1]; dst[3] = src[0];
    }

    mgl_compat_dirty(ctx);
}

void mglGetPolygonStipple(GLMContext ctx, GLubyte *mask)
{    MGLC_COMPAT_GUARD(ctx);

    if (!mask)
        return;

    for (int row = 0; row < 32; row++)
    {
        const GLubyte *src = CS()->polygon_stipple_pattern + (31 - row) * 4;
        GLubyte *dst = mask + row * 4;
        dst[0] = src[3]; dst[1] = src[2]; dst[2] = src[1]; dst[3] = src[0];
    }
}

void mglGetnPolygonStipple(GLMContext ctx, GLsizei bufSize, GLubyte *mask)
{    MGLC_COMPAT_GUARD(ctx);

    if (!mask || bufSize < 128)
        ERROR_RETURN(GL_INVALID_OPERATION);

    mglGetPolygonStipple(ctx, mask);
}

void mglClipPlane(GLMContext ctx, GLenum plane, const GLdouble *equation)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (plane < GL_CLIP_PLANE0 || plane >= GL_CLIP_PLANE0 + MGLC_MAX_CLIP_PLANES)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!equation)
        ERROR_RETURN(GL_INVALID_VALUE);

    /* transformed by the inverse of the current modelview (eye space plane) */
    GLdouble inv[16], *m = s->modelview.stack[s->modelview.depth].m;
    int i = plane - GL_CLIP_PLANE0;

    if (!mglc_matrix_invert(inv, m))
        memcpy(s->clip_plane[i], equation, 4 * sizeof(GLdouble));
    else
    {
        for (int r = 0; r < 4; r++)
        {
            GLdouble t = 0;
            for (int k = 0; k < 4; k++)
                t += inv[r * 4 + k] * equation[k];
            s->clip_plane[i][r] = t;
        }
    }

    mgl_compat_dirty(ctx);
}

void mglGetClipPlane(GLMContext ctx, GLenum plane, GLdouble *equation)
{    MGLC_COMPAT_GUARD(ctx);

    if (plane < GL_CLIP_PLANE0 || plane >= GL_CLIP_PLANE0 + MGLC_MAX_CLIP_PLANES)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (!equation)
        return;

    memcpy(equation, CS()->clip_plane[plane - GL_CLIP_PLANE0], 4 * sizeof(GLdouble));
}

#pragma mark - compat polygon mode / point size / point parameters

void mgl_compat_polygon_mode(GLMContext ctx, GLenum face, GLenum mode);

void mgl_compat_point_size(GLMContext ctx, GLfloat size);

void mgl_compat_point_parameterf(GLMContext ctx, GLenum pname, GLfloat param)
{
    struct MGLExtState_compat *s = CS();

    switch (pname)
    {
        case GL_POINT_SIZE_MIN:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            s->point_params[0] = param;
            mgl_compat_dirty(ctx);
            return;
        case GL_POINT_SIZE_MAX:
            if (param < 0.0f) ERROR_RETURN(GL_INVALID_VALUE);
            s->point_params[1] = param;
            mgl_compat_dirty(ctx);
            return;
        default:
            mglPointParameterf(ctx, pname, param);
            if (pname == GL_POINT_FADE_THRESHOLD_SIZE)
            {
                s->point_params[2] = param;
                mgl_compat_dirty(ctx);
            }
            return;
    }
}

void mgl_compat_point_parameterfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = CS();

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (pname == GL_POINT_DISTANCE_ATTENUATION)
    {
        for (int i = 0; i < 3; i++)
            s->point_params[3 + i] = params[i];
        s->point_attenuation = (params[0] != 0.0f || params[1] != 0.0f || params[2] != 0.0f);
        mgl_compat_dirty(ctx);
        return;
    }

    mgl_compat_point_parameterf(ctx, pname, params[0]);
}

void mgl_compat_point_parameteri(GLMContext ctx, GLenum pname, GLint param)
{
    mgl_compat_point_parameterf(ctx, pname, (GLfloat)param);
}

void mgl_compat_point_parameteriv(GLMContext ctx, GLenum pname, const GLint *params)
{
    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    mgl_compat_point_parameterf(ctx, pname, (GLfloat)params[0]);
}
