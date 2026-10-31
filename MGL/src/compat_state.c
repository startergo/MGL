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

#ifndef GL_INTENSITY
#define GL_INTENSITY 0x8049
#endif
#ifndef GL_LUMINANCE8
#define GL_LUMINANCE8 0x8040
#endif
#include <string.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

#define CS() (ctx->state.ext_compat)

/* glClearColorIiEXT / glClearColorIuiEXT (EXT_texture_integer §2.6, exported
 * by entry_ext_compat.c).  The integer clear colour is separate state from
 * GL_COLOR_CLEAR_VALUE: integer attachments are cleared with these exact
 * bit patterns (no float rounding), float attachments keep using the float
 * value.  mgl_compat_slot_lazy materialises the compat slot on first use —
 * core contexts never pay for it until the app actually calls the EXT. */
void mglClearColorIi(GLMContext ctx, GLint r, GLint g, GLint b, GLint a)
{
    struct MGLExtState_compat *cm = mgl_compat_slot_lazy(ctx);

    if (!cm)
        return;

    /* keep both interpretations: INT reads .i, UINT reads .u below — the
     * same bits, so one store serves both attachment kinds */
    memcpy(cm->clear_color_i, (GLint[4]){ r, g, b, a }, sizeof cm->clear_color_i);
    cm->clear_color_i_set = GL_TRUE;
    /* no dirty bit needed: mglClear re-packs MGLFBClearRequest from state
     * on every call, so the next clear picks the integer colour up */
}

void mglClearColorIui(GLMContext ctx, GLuint r, GLuint g, GLuint b, GLuint a)
{
    struct MGLExtState_compat *cm = mgl_compat_slot_lazy(ctx);

    if (!cm)
        return;

    memcpy(cm->clear_color_i, (const GLint[4]){ (GLint)r, (GLint)g, (GLint)b, (GLint)a },
           sizeof cm->clear_color_i);
    cm->clear_color_i_set = GL_TRUE;
}

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

    GLfloat v[1] = { param };
    mglTexEnvfv(ctx, target, pname, v);
}

/* kept for the vector path (handles every pname) */
static void mglTexEnvf_scalar(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
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

    if (pname == GL_TEXTURE_LOD_BIAS)
    {
        e->lod_bias = param;
        mgl_compat_dirty(ctx);
        return;
    }

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
        {
            MGLCTexEnv *e = &CS()->texenv[STATE(active_texture) % MGLC_MAX_COORDS];

            switch ((GLenum)params[0])
            {
                case GL_MODULATE: case GL_DECAL: case GL_BLEND:
                case GL_REPLACE: case GL_ADD: case GL_COMBINE:
                    break;
                default:
                    ERROR_RETURN(GL_INVALID_ENUM);
            }

            e->env_mode = (GLenum)params[0];
            mgl_compat_dirty(ctx);
            return;
        }

        case GL_TEXTURE_LOD_BIAS:
            s->texenv[unit].lod_bias = params[0];
            mgl_compat_dirty(ctx);
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
    /* keep the raw GL layout (LSB = x=0, row 0 = bottom): the upload code
     * in compat_ffp.c reads it in this order */
    memcpy(CS()->polygon_stipple_pattern, mask, 32 * 4);

    mgl_compat_dirty(ctx);
}

void mglGetPolygonStipple(GLMContext ctx, GLubyte *mask)
{    MGLC_COMPAT_GUARD(ctx);

    if (!mask)
        return;

    memcpy(mask, CS()->polygon_stipple_pattern, 32 * 4);
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

        /* (1, 0, 0) is the identity: no shader-driven point size */
        s->point_attenuation = (params[0] != 1.0f || params[1] != 0.0f || params[2] != 0.0f);

        /* Metal only honours a shader point size when the pipeline is built
         * for GL_PROGRAM_POINT_SIZE; drive that cap from the attenuation */
        if (s->point_attenuation)
            mglEnable(ctx, GL_PROGRAM_POINT_SIZE);
        else
            mglDisable(ctx, GL_PROGRAM_POINT_SIZE);

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

#pragma mark legacy texture formats (LUMINANCE / ALPHA / INTENSITY)

/* the legacy "kind" of an internal format (0 = none) */
static GLenum legacy_tex_kind(GLenum internalformat)
{
    switch (internalformat)
    {
        case 0x1909: return GL_LUMINANCE;          /* GL_LUMINANCE */
        case 0x8040: case 0x8042: return GL_LUMINANCE;      /* LUMINANCE8/16 */
        case 0x190A: return GL_LUMINANCE_ALPHA;    /* GL_LUMINANCE_ALPHA */
        case 0x8045: case 0x8046: return GL_LUMINANCE_ALPHA;
        case 0x1906: return GL_ALPHA;              /* GL_ALPHA */
        case 0x803C: case 0x803E: return GL_ALPHA; /* ALPHA8/16 */
        case 0x8049: case 0x804B: return GL_INTENSITY;       /* INTENSITY8/16 */
    }
    return 0;
}

static void legacy_tex_record(GLMContext ctx, GLuint name, GLenum kind, GLenum type, GLenum iformat)
{
    struct MGLExtState_compat *s = CS();

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
        {
            s->legacy_tex[i].kind = kind;
            s->legacy_tex[i].type = type;
            s->legacy_tex[i].iformat = iformat;
            return;
        }

    if (s->legacy_tex_count < MGLC_LEGACY_TEX_MAX)
    {
        memset(&s->legacy_tex[s->legacy_tex_count], 0, sizeof s->legacy_tex[0]);
        s->legacy_tex[s->legacy_tex_count].name = name;
        s->legacy_tex[s->legacy_tex_count].kind = kind;
        s->legacy_tex[s->legacy_tex_count].type = type;
        s->legacy_tex[s->legacy_tex_count].iformat = iformat;
        s->legacy_tex_count++;
    }
}

/* per-texture compat record (creating it when unknown) */
static void compat_tex_entry(GLMContext ctx, GLuint name, int *slot)
{
    struct MGLExtState_compat *s = CS();

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
        {
            *slot = i;
            return;
        }

    *slot = -1;

    if (s->legacy_tex_count < MGLC_LEGACY_TEX_MAX)
    {
        memset(&s->legacy_tex[s->legacy_tex_count], 0, sizeof s->legacy_tex[0]);
        s->legacy_tex[s->legacy_tex_count].name = name;
        s->legacy_tex[s->legacy_tex_count].kind = 0;
        s->legacy_tex[s->legacy_tex_count].type = 0;
        s->legacy_tex[s->legacy_tex_count].iformat = 0;
        *slot = s->legacy_tex_count++;
    }
}


static void legacy_tex_forget(GLMContext ctx, GLuint name)
{
    struct MGLExtState_compat *s = CS();

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
        {
            s->legacy_tex[i] = s->legacy_tex[--s->legacy_tex_count];
            return;
        }
}

/* component count of a legacy kind in the *client* data */
static int legacy_client_comps(GLenum kind)
{
    return (kind == GL_LUMINANCE_ALPHA) ? 2 : 1;
}

/* bytes per component of a supported client type (0 = unsupported here) */
static int legacy_type_bpc(GLenum type)
{
    switch (type)
    {
        case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
        case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2;
        case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4;
    }
    return 0;
}

/* Legacy formats are stored at their real precision (R8/RG8/R16/R32F...) with
 * the legacy replication done by the sampler swizzle, instead of being blown
 * up to RGBA32F.  This picks the storage format and client layout for a
 * (kind, type) pair; returns 0 when the type has no supported storage. */
static GLenum legacy_storage_format(GLenum kind, GLenum type, GLenum *client_format, int *cpu_float)
{
    int two = (kind == GL_LUMINANCE_ALPHA);

    *client_format = two ? GL_RG : GL_RED;
    *cpu_float = 0;     /* 1 = convert the client data to floats here first */

    switch (type)
    {
        case GL_UNSIGNED_BYTE:  return two ? GL_RG8 : GL_R8;
        case GL_UNSIGNED_SHORT: return two ? GL_RG16 : GL_R16;
        case GL_BYTE:           *cpu_float = 1; return two ? GL_RG8_SNORM : GL_R8_SNORM;
        case GL_SHORT:          *cpu_float = 1; return two ? GL_RG16_SNORM : GL_R16_SNORM;
        case GL_HALF_FLOAT:     *cpu_float = 1; return two ? GL_RG16F : GL_R16F;
        case GL_FLOAT:          return two ? GL_RG32F : GL_R32F;
        case GL_UNSIGNED_INT:
        case GL_INT:            *cpu_float = 1; return two ? GL_RG32F : GL_R32F;
    }
    return 0;
}

/* the sampler swizzle that turns the stored channels into the legacy format:
 * LUMINANCE (l,l,l,1), ALPHA (0,0,0,a), LUMINANCE_ALPHA (l,l,l,a), INTENSITY (i,i,i,i) */
static void legacy_set_swizzle(GLMContext ctx, GLenum target, GLenum kind)
{
    GLenum r, g, b, a;

    if (kind == GL_LUMINANCE)              { r = GL_RED;  g = GL_RED;  b = GL_RED;  a = GL_ONE;  }
    else if (kind == GL_ALPHA)              { r = GL_ZERO; g = GL_ZERO; b = GL_ZERO; a = GL_RED; }
    else if (kind == GL_LUMINANCE_ALPHA)    { r = GL_RED;  g = GL_RED;  b = GL_RED;  a = GL_GREEN; }
    else                                    { r = GL_RED;  g = GL_RED;  b = GL_RED;  a = GL_RED; }

    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_R, r);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_G, g);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_B, b);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_A, a);
}

/* drops the legacy swizzle (the texture was re-specified with a core format) */
static void legacy_reset_swizzle(GLMContext ctx, GLenum target)
{
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_R, GL_RED);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
    mglTexParameteri(ctx, target, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
}

/* converts client data to normalized floats (1 or 2 components per texel) */
typedef unsigned short GLhalf;

/* GLhalf -> float (IEEE 754 half precision) */
static float mgl_compat_half_to_float(GLhalf h)
{
    unsigned sign = (unsigned)(h >> 15) & 1;
    unsigned exp  = (unsigned)(h >> 10) & 0x1F;
    unsigned man  = (unsigned)h & 0x3FF;
    union { float f; unsigned u; } out;

    if (exp == 0)
    {
        if (man == 0)
        {
            out.u = sign << 31;
            return out.f;
        }
        /* subnormal: normalize */
        unsigned e = 0;
        while ((man & 0x400) == 0) { man <<= 1; e++; }
        man &= 0x3FF;
        exp = (unsigned)(127 - 15 + 1 - e);
        out.u = (sign << 31) | (exp << 23) | (man << 13);
        return out.f;
    }

    if (exp == 0x1F)
    {
        out.u = (sign << 31) | 0x7F800000u | (man << 13);
        return out.f;
    }

    out.u = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    return out.f;
}

static void legacy_to_float(GLenum type, const void *pixels, size_t n, int cc, GLfloat *out)
{
    for (size_t i = 0; i < n * (size_t)cc; i++)
    {
        float v = 0.0f;

        switch (type)
        {
            case GL_BYTE: v = ((const GLbyte *)pixels)[i] / 127.0f; break;
            case GL_SHORT: v = ((const GLshort *)pixels)[i] / 32767.0f; break;
            case GL_HALF_FLOAT: v = mgl_compat_half_to_float(((const GLhalf *)pixels)[i]); break;
            case GL_UNSIGNED_INT: v = (float)((const GLuint *)pixels)[i] / 4294967295.0f; break;
            case GL_INT: v = (float)((const GLint *)pixels)[i] / 2147483647.0f; break;
            default: v = ((const GLfloat *)pixels)[i]; break;
        }
        out[i] = v;
    }
}

/* packs (l,a) texel values back into a legacy client layout: the components
 * come from the stored channels (l = red, a = green or 1), the swizzle is not
 * applied on readback */
static void legacy_pack(GLenum kind, GLenum format, GLenum type,
                        GLsizei w, GLsizei h, const GLfloat *in, void *out)
{
    size_t n = (size_t)w * (size_t)(h ? h : 1);

    for (size_t i = 0; i < n; i++)
    {
        float l = in[i * 4 + 0];
        float a = (kind == GL_LUMINANCE_ALPHA) ? in[i * 4 + 1] :
                  (kind == GL_INTENSITY) ? in[i * 4 + 0] : 1.0f;
        float v[4];
        int comps = 1;

        switch (format)
        {
            case GL_LUMINANCE: case GL_INTENSITY: case GL_RED:
                v[0] = l; comps = 1; break;
            case GL_ALPHA:
                v[0] = (kind == GL_ALPHA) ? in[i * 4 + 0] : a; comps = 1; break;
            case GL_LUMINANCE_ALPHA:
                v[0] = l; v[1] = a; comps = 2; break;
            case GL_RG:
                v[0] = l; v[1] = a; comps = 2; break;
            case GL_RGB:
                v[0] = v[1] = v[2] = l; comps = 3; break;
            default:    /* GL_RGBA and friends */
                v[0] = v[1] = v[2] = l; v[3] = a; comps = 4; break;
        }

        for (int k = 0; k < comps; k++)
        {
            float x = v[k];
            size_t o = i * (size_t)comps + k;

            switch (type)
            {
                case GL_UNSIGNED_BYTE:  ((GLubyte *)out)[o] = (GLubyte)(x * 255.0f + 0.5f); break;
                case GL_BYTE:           ((GLbyte *)out)[o] = (GLbyte)(x * 127.0f + (x >= 0 ? 0.5f : -0.5f)); break;
                case GL_UNSIGNED_SHORT: ((GLushort *)out)[o] = (GLushort)(x * 65535.0f + 0.5f); break;
                case GL_SHORT:          ((GLshort *)out)[o] = (GLshort)(x * 32767.0f + (x >= 0 ? 0.5f : -0.5f)); break;
                case GL_UNSIGNED_INT:   ((GLuint *)out)[o] = (GLuint)(x * 4294967295.0f + 0.5f); break;
                case GL_INT:            ((GLint *)out)[o] = (GLint)(x * 2147483647.0f + (x >= 0 ? 0.5f : -0.5f)); break;
                default:                ((GLfloat *)out)[o] = x; break;
            }
        }
    }
}

/* the texture currently bound to |target| on the active unit */
static GLuint bound_tex_name(GLMContext ctx, GLenum target)
{
    int idx = mgl_compat_target_index(target);

    if (idx < 0)
        return 0;

    Texture *t = STATE(texture_units[STATE(active_texture)]).textures[idx];
    return t ? t->name : 0;
}

/* runs glGenerateMipmap after a level-0 upload when GL_GENERATE_MIPMAP is on */
static void compat_after_upload(GLMContext ctx, GLenum target, GLint level)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);

    /* the GL_CLAMP emulation uploads the level-0 size as a uniform: sizes can
     * change here, so the fixed-function sync must run again */
    mgl_compat_dirty(ctx);

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name && s->legacy_tex[i].gmm)
        {
            if (level == 0)
                mglGenerateMipmap(ctx, target);
            return;
        }
}

/* uploads a legacy image at its real precision (R8/RG8/R16/RG16/...) and
 * installs the sampler swizzle; returns false when the format must fall
 * through to the core path */
static bool legacy_upload(GLMContext ctx, GLenum target, GLint level, GLenum kind,
                          GLsizei w, GLsizei h, GLsizei depth, GLint border,
                          GLenum format, GLenum type, const void *pixels, GLenum req_iformat)
{
    GLenum client_format = 0, iformat;
    const void *data = pixels;
    GLfloat *conv = NULL;
    int cpu_float, cc = legacy_client_comps(kind);
    size_t n = (size_t)w * (size_t)(h ? h : 1) * (size_t)(depth ? depth : 1);

    if (format != kind || !pixels)
        return false;

    iformat = legacy_storage_format(kind, type, &client_format, &cpu_float);
    if (!iformat)
        return false;

    if (cpu_float)
    {
        conv = (GLfloat *)malloc(n * (size_t)cc * sizeof(GLfloat));
        if (!conv)
            return false;
        legacy_to_float(type, pixels, n, cc, conv);
        data = conv;
        type = GL_FLOAT;
    }

    if (target == GL_TEXTURE_1D)
        mglTexImage1D(ctx, target, level, iformat, w, border, client_format, type, data);
    else if (depth)
        mglTexImage3D(ctx, target, level, iformat, w, h, depth, border, client_format, type, data);
    else
        mglTexImage2D(ctx, target, level, iformat, w, h, border, client_format, type, data);

    free(conv);
    legacy_set_swizzle(ctx, target, kind);
    legacy_tex_record(ctx, bound_tex_name(ctx, target), kind, format == GL_INTENSITY ? format : type, req_iformat);
    compat_after_upload(ctx, target, level);
    return true;
}

void compat_tex_image2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat,
                        GLsizei width, GLsizei height, GLint border,
                        GLenum format, GLenum type, const void *pixels)
{
    GLenum kind = legacy_tex_kind((GLenum)internalformat);

    if (kind && legacy_type_bpc(type) &&
        legacy_upload(ctx, target, level, kind, width, height, 0, border, format, type, pixels,
                      (GLenum)internalformat))
        return;

    mglTexImage2D(ctx, target, level, internalformat, width, height, border, format, type, pixels);

    if (kind)
        legacy_tex_record(ctx, bound_tex_name(ctx, target), kind, type, (GLenum)internalformat);
    else
    {
        /* re-specified with a core format: drop the legacy swizzle/record */
        GLuint name = bound_tex_name(ctx, target);
        struct MGLExtState_compat *s = CS();

        for (int i = 0; i < s->legacy_tex_count; i++)
            if (s->legacy_tex[i].name == name && s->legacy_tex[i].kind)
            {
                s->legacy_tex[i].kind = 0;
                s->legacy_tex[i].iformat = 0;
                legacy_reset_swizzle(ctx, target);
            }
    }

    compat_after_upload(ctx, target, level);
}

void compat_tex_image1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat,
                        GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels)
{
    GLenum kind = legacy_tex_kind((GLenum)internalformat);

    if (kind && legacy_type_bpc(type) &&
        legacy_upload(ctx, target, level, kind, width, 1, 0, border, format, type, pixels,
                      (GLenum)internalformat))
        return;

    mglTexImage1D(ctx, target, level, internalformat, width, border, format, type, pixels);

    if (kind)
        legacy_tex_record(ctx, bound_tex_name(ctx, target), kind, type, (GLenum)internalformat);
    else
    {
        /* re-specified with a core format: drop the legacy swizzle/record */
        GLuint name = bound_tex_name(ctx, target);
        struct MGLExtState_compat *s = CS();

        for (int i = 0; i < s->legacy_tex_count; i++)
            if (s->legacy_tex[i].name == name && s->legacy_tex[i].kind)
            {
                s->legacy_tex[i].kind = 0;
                s->legacy_tex[i].iformat = 0;
                legacy_reset_swizzle(ctx, target);
            }
    }

    compat_after_upload(ctx, target, level);
}

void compat_tex_image3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat,
                        GLsizei width, GLsizei height, GLsizei depth, GLint border,
                        GLenum format, GLenum type, const void *pixels)
{
    GLenum kind = legacy_tex_kind((GLenum)internalformat);

    if (kind && legacy_type_bpc(type) &&
        legacy_upload(ctx, target, level, kind, width, height, depth, border, format, type, pixels,
                      (GLenum)internalformat))
        return;

    mglTexImage3D(ctx, target, level, internalformat, width, height, depth, border, format, type, pixels);

    if (kind)
        legacy_tex_record(ctx, bound_tex_name(ctx, target), kind, type, (GLenum)internalformat);

    compat_after_upload(ctx, target, level);
}

void compat_get_tex_image(GLMContext ctx, GLenum target, GLint level,
                          GLenum format, GLenum type, void *pixels)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);
    GLenum kind = 0;

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
            kind = s->legacy_tex[i].kind;

    if (!kind)
    {
        mglGetTexImage(ctx, target, level, format, type, pixels);
        return;
    }

    /* read RGBA back, then re-pack to the legacy layout */
    GLint w = 0, h = 0;
    mglGetTexLevelParameteriv(ctx, target, level, GL_TEXTURE_WIDTH, &w);
    mglGetTexLevelParameteriv(ctx, target, level, GL_TEXTURE_HEIGHT, &h);

    if (w <= 0 || h <= 0)
        return;

    GLfloat *rgba = (GLfloat *)malloc((size_t)w * h * 4 * sizeof(GLfloat));

    if (!rgba)
        return;

    mglGetTexImage(ctx, target, level, GL_RGBA, GL_FLOAT, rgba);
    legacy_pack(kind, format, type, w, h, rgba, pixels);
    free(rgba);
}

void compat_get_tex_level_parameteriv(GLMContext ctx, GLenum target, GLint level,
                                      GLenum pname, GLint *params)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);

    mglGetTexLevelParameteriv(ctx, target, level, pname, params);

    if (pname == GL_TEXTURE_INTERNAL_FORMAT)
    {
        for (int i = 0; i < s->legacy_tex_count; i++)
            if (s->legacy_tex[i].name == name)
            {
                /* the internal format the app asked for; LUMINANCE(8) etc. are
                 * what GL reports even though the storage is R8/RG8 */
                if (s->legacy_tex[i].kind && s->legacy_tex[i].iformat)
                {
                    *params = s->legacy_tex[i].iformat;
                    return;
                }
                break;      /* re-specified with a core format: ask the core */
            }
    }
}

#pragma mark compat texture parameters (GL_CLAMP, GENERATE_MIPMAP)

extern Texture *findTexture(GLMContext ctx, GLuint texture);

/* one of the three border colors Metal samplers can fetch natively? */
static bool standard_border(const Texture *tex)
{
    const float *b = tex->params.border_color;
    bool black = b[0] == 0.0f && b[1] == 0.0f && b[2] == 0.0f;
    bool white = b[0] == 1.0f && b[1] == 1.0f && b[2] == 1.0f;
    if (!black && !white) return false;
    return b[3] == 0.0f || b[3] == 1.0f;
}

/* NEAREST filtering in both directions (mip selection may still blend levels,
 * the within-level border behaviour is what matters here)? */
static bool nearest_filtering(const Texture *tex)
{
    switch (tex->params.min_filter)
    {
        case GL_NEAREST: case GL_NEAREST_MIPMAP_NEAREST: case GL_NEAREST_MIPMAP_LINEAR:
            break;
        default:
            return false;
    }
    return tex->params.mag_filter == GL_NEAREST;
}

/* Per-axis emulation decision for a texture that asked for GL_CLAMP or
 * GL_CLAMP_TO_BORDER (see mgl_ext_compat.h). */
static void clamp_modes_for(GLMContext ctx, GLuint name, Texture *tex, int emu[3])
{
    struct MGLExtState_compat *s = CS();
    int slot = -1;

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
            slot = i;

    emu[0] = emu[1] = emu[2] = MGLC_CLAMP_NONE;
    if (slot < 0 || !tex)
        return;

    bool bstd = standard_border(tex);
    bool near = nearest_filtering(tex);

    for (int a = 0; a < 3; a++)
    {
        GLenum req = s->legacy_tex[slot].wrap[a];

        if (req == GL_CLAMP_TO_BORDER)
            emu[a] = bstd ? MGLC_CLAMP_NONE : MGLC_CLAMP_EMU_BORDER;
        else if (req == GL_CLAMP)
            /* GL_CLAMP saturates at a 50/50 border blend outside [0,1], which
             * the hardware border mode cannot do; with NEAREST the hardware
             * matches GL exactly, so only LINEAR is emulated */
            emu[a] = (bstd && near) ? MGLC_CLAMP_NONE : MGLC_CLAMP_EMU;
    }
}

void mgl_compat_clamp_modes(GLMContext ctx, Texture *tex, int emu[3])
{
    clamp_modes_for(ctx, tex ? tex->name : 0, tex, emu);
}

/* push the decision into the core sampler state (the texture must be bound) */
void mgl_compat_clamp_apply(GLMContext ctx, GLuint name)
{
    struct MGLExtState_compat *s = CS();
    Texture *tex = findTexture(ctx, name);
    int slot = -1;
    int emu[3];

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
            slot = i;
    if (slot < 0 || !tex)
        return;

    clamp_modes_for(ctx, name, tex, emu);

    static const GLenum axis_pname[3] = { GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R };
    for (int a = 0; a < 3; a++)
    {
        GLenum req = s->legacy_tex[slot].wrap[a];
        GLenum want;

        if (!req)
            continue;                       /* axis never asked for a legacy mode */
        else if (emu[a] == MGLC_CLAMP_NONE)
            want = GL_CLAMP_TO_BORDER;      /* hardware border fetch */
        else
            want = GL_CLAMP_TO_EDGE;        /* emulated in the shader */

        if (want != s->legacy_tex[slot].applied[a])
        {
            mglTexParameteri(ctx, tex->target, axis_pname[a], want);
            s->legacy_tex[slot].applied[a] = want;
            mgl_compat_dirty(ctx);
        }
    }
}

void compat_tex_parameteri(GLMContext ctx, GLenum target, GLenum pname, GLint param)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);

    if (pname == GL_GENERATE_MIPMAP)
    {
        int slot;
        compat_tex_entry(ctx, name, &slot);
        if (slot >= 0)
            s->legacy_tex[slot].gmm = param ? GL_TRUE : GL_FALSE;
        return;     /* not a hardware parameter */
    }

    if (pname == GL_TEXTURE_WRAP_S || pname == GL_TEXTURE_WRAP_T || pname == GL_TEXTURE_WRAP_R)
    {
        int axis = (pname == GL_TEXTURE_WRAP_S) ? 0 : (pname == GL_TEXTURE_WRAP_T) ? 1 : 2;
        int slot;

        if (param == GL_CLAMP || param == GL_CLAMP_TO_BORDER)
        {
            /* remembered for queries and the border emulation decision; the
             * wrap handed to the sampler is chosen by mgl_compat_clamp_apply */
            compat_tex_entry(ctx, name, &slot);
            if (slot >= 0)
                s->legacy_tex[slot].wrap[axis] = (GLenum)param;
            if (slot >= 0)
                mgl_compat_clamp_apply(ctx, name);
            return;
        }

        compat_tex_entry(ctx, name, &slot);
        if (slot >= 0)
        {
            s->legacy_tex[slot].wrap[axis] = 0;
            s->legacy_tex[slot].applied[axis] = 0;
        }
    }
    else if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAG_FILTER)
    {
        /* the filter is part of the CLAMP decision (and the border weight) */
        mglTexParameteri(ctx, target, pname, param);
        mgl_compat_dirty(ctx);
        mgl_compat_clamp_apply(ctx, name);
        return;
    }

    mglTexParameteri(ctx, target, pname, param);
}

void compat_tex_parameterf(GLMContext ctx, GLenum target, GLenum pname, GLfloat param)
{
    compat_tex_parameteri(ctx, target, pname, (GLint)param);
}

void compat_tex_parameteriv(GLMContext ctx, GLenum target, GLenum pname, const GLint *params)
{
    compat_tex_parameteri(ctx, target, pname, params ? params[0] : 0);
}

void compat_tex_parameterfv(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    if (pname == GL_TEXTURE_BORDER_COLOR)
    {
        /* the border color is part of the GL_CLAMP decision (only three colors
         * can be fetched by the hardware) and feeds the mglBorderN uniform */
        mglTexParameterfv(ctx, target, pname, params);
        mgl_compat_dirty(ctx);
        mgl_compat_clamp_apply(ctx, bound_tex_name(ctx, target));
        return;
    }

    compat_tex_parameteri(ctx, target, pname, (GLint)(params ? params[0] : 0));
}

void compat_get_tex_parameteriv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);
    int slot = -1;

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
            slot = i;

    if (pname == GL_GENERATE_MIPMAP)
    {
        /* every texture has the parameter; default GL_FALSE (SGIS_generate_mipmap) */
        *params = (slot >= 0) ? s->legacy_tex[slot].gmm : GL_FALSE;
        return;
    }

    if (slot >= 0 && (pname == GL_TEXTURE_WRAP_S || pname == GL_TEXTURE_WRAP_T || pname == GL_TEXTURE_WRAP_R))
    {
        int axis = (pname == GL_TEXTURE_WRAP_S) ? 0 : (pname == GL_TEXTURE_WRAP_T) ? 1 : 2;
        *params = s->legacy_tex[slot].wrap[axis] ? (GLint)s->legacy_tex[slot].wrap[axis] : (GLint)GL_CLAMP_TO_EDGE;
        return;
    }

    mglGetTexParameteriv(ctx, target, pname, params);
}

void compat_get_tex_parameterfv(GLMContext ctx, GLenum target, GLenum pname, GLfloat *params)
{
    struct MGLExtState_compat *s = CS();
    GLuint name = bound_tex_name(ctx, target);
    int slot = -1;

    for (int i = 0; i < s->legacy_tex_count; i++)
        if (s->legacy_tex[i].name == name)
            slot = i;

    if (pname == GL_GENERATE_MIPMAP)
    {
        *params = (slot >= 0 && s->legacy_tex[slot].gmm) ? 1.0f : 0.0f;
        return;
    }

    if (slot >= 0 && (pname == GL_TEXTURE_WRAP_S || pname == GL_TEXTURE_WRAP_T || pname == GL_TEXTURE_WRAP_R))
    {
        int axis = (pname == GL_TEXTURE_WRAP_S) ? 0 : (pname == GL_TEXTURE_WRAP_T) ? 1 : 2;
        *params = s->legacy_tex[slot].wrap[axis] ? (GLfloat)s->legacy_tex[slot].wrap[axis] : (GLfloat)GL_CLAMP_TO_EDGE;
        return;
    }

    mglGetTexParameterfv(ctx, target, pname, params);
}
