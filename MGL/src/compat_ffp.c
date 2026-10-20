/*
 * compat_ffp.c — the generated fixed-function pipeline (compat domain).
 *
 * OpenGL 1.x/2.x fixed function is turned into GLSL 460 core programs the way
 * Mesa's ffvertex_prog.c / ff_fragment_shader.cpp do it on modern hardware:
 * a state key picks a variant; the vertex shader implements the matrix stack,
 * lighting (two sided, color material, spot + attenuation), texgen, user clip
 * planes and point size attenuation; the fragment shader implements texenv
 * (incl. GL_COMBINE with the crossbar), fog, alpha test, separate specular,
 * point sprites, line/polygon stipple and flat shading.
 *
 * Uniforms are default-block uniforms (MGL packs them into the program's
 * "mgl_DefaultUniforms" UBO automatically); textures are ordinary samplers
 * bound to the fixed-function texture units.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"

#define CS() (ctx->state.ext_compat)

#pragma mark string builder

typedef struct {
    char *s;
    size_t len, cap;
} SB;

static void sb_init(SB *b)
{
    b->cap = 8192;
    b->s = (char *)malloc(b->cap);
    b->s[0] = 0;
    b->len = 0;
}

static void sb_add(SB *b, const char *fmt, ...)
{
    va_list ap;
    int n;
    char tmp[1024];

    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);

    if (n < 0)
        return;
    if ((size_t)n >= sizeof tmp)
        n = (int)sizeof tmp - 1;

    if (b->len + (size_t)n + 1 > b->cap)
    {
        while (b->len + (size_t)n + 1 > b->cap)
            b->cap *= 2;
        b->s = (char *)realloc(b->s, b->cap);
    }

    memcpy(b->s + b->len, tmp, (size_t)n);
    b->len += (size_t)n;
    b->s[b->len] = 0;
}

#pragma mark state key

typedef struct {
    uint8_t bytes[256];
} MGLFFPKey;

#define MGLC_FFP_CACHE 32

typedef struct {
    int used;
    int failed;
    int draws;
    int dirty_late;
    GLuint prog;
    uint64_t sync;
    MGLFFPKey key;
} MGLFFPCacheEnt;

/* effective (fixed-function) texture target of a unit: GL_TEXTURE_1D..CUBE or 0 */
int mgl_compat_enabled_texture_target(GLMContext ctx, GLuint unit)
{
    struct MGLExtState_compat *s = CS();

    if (unit >= MGLC_MAX_COORDS)
        return 0;

    uint32_t mask = s->unit_target_mask[unit];
    static const GLenum targets[] = { GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP };
    static const int texindex[] = { _TEXTURE_1D, _TEXTURE_2D, _TEXTURE_3D, _TEXTURE_CUBE_MAP };

    for (int i = 0; i < 4; i++)
    {
        if (!(mask & (1u << i)))
            continue;
        Texture *tex = STATE(texture_units[unit]).textures[texindex[i]];
        if (tex)
            return (int)targets[i];
    }

    return 0;
}

typedef struct {
    int     target[MGLC_MAX_COORDS];
    int     env_mode[MGLC_MAX_COORDS];
    int     combine_rgb[MGLC_MAX_COORDS];
    int     combine_a[MGLC_MAX_COORDS];
    int     src[6][MGLC_MAX_COORDS];
    int     op_rgb[3][MGLC_MAX_COORDS];
    int     op_a[3][MGLC_MAX_COORDS];
    int     coord_replace[MGLC_MAX_COORDS];
    int     need_texcoord[MGLC_MAX_COORDS];
    int     texgen[MGLC_MAX_COORDS][4];
    GLboolean unit_used[MGLC_MAX_COORDS];
} MGLFFPUnit;

typedef struct {
    /* vertex */
    int  lighting, two_sided, color_material_mode, local_viewer;
    int  light_pos[MGLC_MAX_LIGHTS];
    int  light_spot[MGLC_MAX_LIGHTS];
    int  light_att[MGLC_MAX_LIGHTS];
    int  normalize, rescale;
    int  need_normal, need_color, need_secondary, need_fogcoord;
    int  clip_mask;
    int  point_atten;
    /* fragment */
    int  fog, fog_mode, fog_src;
    int  alpha_test, alpha_func;
    int  flat;
    int  separate_specular;
    int  polygon_stipple;
    int  line_stipple;
    MGLFFPUnit unit;
} MGLFFPConfig;

static void ffp_build_config(GLMContext ctx, MGLFFPConfig *c)
{
    struct MGLExtState_compat *s = CS();

    memset(c, 0, sizeof *c);

    c->lighting = s->lighting;
    c->two_sided = s->two_sided;
    c->local_viewer = s->local_viewer;
    c->normalize = s->normalize;
    c->rescale = s->rescale_normal;
    c->flat = s->shade_model == GL_FLAT;
    c->separate_specular = s->separate_specular;

    c->color_material_mode = s->color_material ? (int)s->color_material_mode : 0;

    for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
    {
        MGLCLight *l = &s->lights[i];
        if (!l->enabled) continue;
        c->light_pos[i] = l->position[3] != 0.0f;
        c->light_spot[i] = l->spot_cutoff < 180.0f;
        c->light_att[i] = (l->linear_attenuation != 0.0f || l->quadratic_attenuation != 0.0f ||
                           l->constant_attenuation != 1.0f);
    }

    c->clip_mask = 0;
    for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
        if (s->clip_plane_enabled[i])
            c->clip_mask |= 1 << i;

    c->point_atten = s->point_attenuation;

    c->fog = s->fog;
    c->fog_mode = (int)s->fog_mode;
    c->fog_src = (int)s->fog_coord_src;

    c->alpha_test = s->alpha_test;
    c->alpha_func = (int)s->alpha_func;

    c->polygon_stipple = s->polygon_stipple;
    c->line_stipple = s->line_stipple;

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        MGLCTexGen *g = &s->texgen[u];
        MGLCTexEnv *e = &s->texenv[u];

        c->unit.target[u] = mgl_compat_enabled_texture_target(ctx, u);
        c->unit.env_mode[u] = (int)e->env_mode;
        c->unit.combine_rgb[u] = (int)e->combine_rgb;
        c->unit.combine_a[u] = (int)e->combine_alpha;
        c->unit.coord_replace[u] = s->coord_replace[u] ? 1 : 0;

        for (int k = 0; k < 6; k++)
            c->unit.src[k][u] = (int)e->src[k];
        for (int k = 0; k < 3; k++)
        {
            c->unit.op_rgb[k][u] = (int)e->op_rgb[k];
            c->unit.op_a[k][u] = (int)e->op_alpha[k];
        }

        for (int cc = 0; cc < 4; cc++)
            c->unit.texgen[u][cc] = g->enabled[cc] ? (int)g->mode[cc] : 0;

        c->unit.unit_used[u] = c->unit.target[u] != 0;
    }

    /* what the vertex shader has to read */
    c->need_color = 1;
    c->need_secondary = 1;
    c->need_normal = 0;

    if (c->lighting)
    {
        c->need_normal = 1;
        if (c->color_material_mode == 0)
            c->need_color = 0;         /* current color does not feed the material */
    }

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        int gen_normal = 0, need_tc = 0;

        for (int cc = 0; cc < 4; cc++)
        {
            if (!c->unit.texgen[u][cc] && c->unit.unit_used[u])
                need_tc = 1;

            switch (c->unit.texgen[u][cc])
            {
                case GL_SPHERE_MAP: case GL_NORMAL_MAP: case GL_REFLECTION_MAP:
                    gen_normal = 1;
                    break;
            }
        }

        if (gen_normal)
            c->need_normal = 1;

        c->unit.need_texcoord[u] = need_tc;
    }

    c->need_fogcoord = (c->fog && c->fog_src == GL_FOG_COORD);
}

static void ffp_build_key(const MGLFFPConfig *c, MGLFFPKey *key)
{
    memset(key, 0, sizeof *key);

    int p = 0;
    uint8_t *b = key->bytes;

    /* enum -> small id so distinct values never collide in a byte */
    struct { GLenum v; int id; } env_ids[] = {
        { GL_MODULATE, 1 }, { GL_DECAL, 2 }, { GL_BLEND, 3 },
        { GL_REPLACE, 4 }, { GL_ADD, 5 }, { GL_COMBINE, 6 }, { 0, 0 }
    };
    struct { GLenum v; int id; } fog_ids[] = {
        { GL_EXP, 1 }, { GL_EXP2, 2 }, { GL_LINEAR, 3 }, { 0, 0 }
    };
    struct { GLenum v; int id; } target_ids[] = {
        { GL_TEXTURE_1D, 1 }, { GL_TEXTURE_2D, 2 }, { GL_TEXTURE_3D, 3 },
        { GL_TEXTURE_CUBE_MAP, 4 }, { 0, 0 }
    };
    struct { GLenum v; int id; } combine_ids[] = {
        { GL_REPLACE, 1 }, { GL_MODULATE, 2 }, { GL_ADD, 3 }, { GL_ADD_SIGNED, 4 },
        { GL_INTERPOLATE, 5 }, { GL_SUBTRACT, 6 }, { GL_DOT3_RGB, 7 },
        { GL_DOT3_RGBA, 8 }, { 0, 0 }
    };
    struct { GLenum v; int id; } texgen_ids[] = {
        { GL_OBJECT_LINEAR, 1 }, { GL_EYE_LINEAR, 2 }, { GL_SPHERE_MAP, 3 },
        { GL_NORMAL_MAP, 4 }, { GL_REFLECTION_MAP, 5 }, { 0, 0 }
    };

    b[p++] = (uint8_t)c->lighting;
    b[p++] = (uint8_t)c->two_sided;
    b[p++] = (uint8_t)c->color_material_mode;
    b[p++] = (uint8_t)((c->local_viewer) | (c->normalize << 1) | (c->rescale << 2) |
                       (c->need_normal << 3) | (c->need_color << 4) |
                       (c->need_secondary << 5) | (c->need_fogcoord << 6) |
                       (c->flat << 7));
    b[p++] = (uint8_t)c->clip_mask;
    b[p++] = (uint8_t)c->point_atten;

    for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
        b[p++] = (uint8_t)((c->light_pos[i] ? 1 : 0) | (c->light_spot[i] ? 2 : 0) | (c->light_att[i] ? 4 : 0));

    b[p++] = (uint8_t)c->fog;
    for (int i = 0; fog_ids[i].id; i++)
        if (fog_ids[i].v == (GLenum)c->fog_mode)
            b[p++] = (uint8_t)fog_ids[i].id;
    b[p++] = (uint8_t)c->fog_src;
    b[p++] = (uint8_t)c->alpha_test;
    b[p++] = (uint8_t)c->alpha_func;
    b[p++] = (uint8_t)((c->separate_specular ? 1 : 0) | (c->polygon_stipple ? 2 : 0) |
                       (c->line_stipple ? 4 : 0));

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        int env_id = 0, fog_unused = 0, tgt_id = 0, crgb = 0, cal = 0;

        for (int i = 0; env_ids[i].id; i++)
            if (env_ids[i].v == (GLenum)c->unit.env_mode[u])
                env_id = env_ids[i].id;
        for (int i = 0; target_ids[i].id; i++)
            if (target_ids[i].v == (GLenum)c->unit.target[u])
                tgt_id = target_ids[i].id;
        for (int i = 0; combine_ids[i].id; i++)
        {
            if (combine_ids[i].v == (GLenum)c->unit.combine_rgb[u])
                crgb = combine_ids[i].id;
            if (combine_ids[i].v == (GLenum)c->unit.combine_a[u])
                cal = combine_ids[i].id;
        }

        (void)fog_unused;

        b[p++] = (uint8_t)tgt_id;
        b[p++] = (uint8_t)env_id;
        b[p++] = (uint8_t)crgb;
        b[p++] = (uint8_t)cal;
        b[p++] = (uint8_t)((c->unit.coord_replace[u] ? 1 : 0) | (c->unit.need_texcoord[u] ? 2 : 0));

        for (int k = 0; k < 6; k++)
            b[p++] = (uint8_t)(c->unit.src[k][u] & 0xFF);
        for (int k = 0; k < 3; k++)
            b[p++] = (uint8_t)(c->unit.op_rgb[k][u] & 0xFF);
        for (int k = 0; k < 3; k++)
            b[p++] = (uint8_t)(c->unit.op_a[k][u] & 0xFF);
        for (int cc = 0; cc < 4; cc++)
        {
            int id = 0;
            for (int i = 0; texgen_ids[i].id; i++)
                if (texgen_ids[i].v == (GLenum)c->unit.texgen[u][cc])
                    id = texgen_ids[i].id;
            b[p++] = (uint8_t)id;
        }
    }
}

#pragma mark vertex shader

static void gen_vertex_shader(GLMContext ctx, const MGLFFPConfig *c, SB *b)
{
    sb_add(b, "#version 460 core\n\n");

    sb_add(b, "uniform mat4 mglModelViewMatrix;\n");
    sb_add(b, "uniform mat4 mglProjectionMatrix;\n");
    if (c->need_normal)
        sb_add(b, "uniform mat3 mglNormalMatrix;\n");
    if (c->rescale && !c->normalize)
        sb_add(b, "uniform vec4 mglNormalParams;   /* .x = 1/|modelview scale| */\n");

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        if (c->unit.unit_used[u])
            sb_add(b, "uniform mat4 mglTextureMatrix%d;\n", u);

    if (c->clip_mask)
        for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
            if (c->clip_mask & (1 << i))
                sb_add(b, "uniform vec4 mglClipPlane%d;\n", i);

    if (c->point_atten)
    {
        sb_add(b, "uniform vec4 mglPointAtten;      /* a, b, c, unused */\n");
        sb_add(b, "uniform vec4 mglPointSizeParams; /* size, min, max, unused */\n");
    }

    if (c->lighting)
    {
        sb_add(b, "uniform vec4 mglLightAmbient[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec4 mglLightDiffuse[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec4 mglLightSpecular[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec4 mglLightPosition[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec3 mglLightSpotDir[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform float mglLightSpotExp[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform float mglLightSpotCosCut[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec3 mglLightAtt[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform int mglLightEnabled[%d];\n", MGLC_MAX_LIGHTS);
        sb_add(b, "uniform vec4 mglMatAmbient[2];\n");
        sb_add(b, "uniform vec4 mglMatDiffuse[2];\n");
        sb_add(b, "uniform vec4 mglMatSpecular[2];\n");
        sb_add(b, "uniform vec4 mglMatEmission[2];\n");
        sb_add(b, "uniform float mglMatShininess[2];\n");
        sb_add(b, "uniform vec4 mglLightModelAmbient;\n");
    }

    /* texgen planes */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        for (int cc = 0; cc < 4; cc++)
        {
            if (c->unit.texgen[u][cc] == GL_OBJECT_LINEAR)
                sb_add(b, "uniform vec4 mglObjectPlane%d_%d;\n", u, cc);
            else if (c->unit.texgen[u][cc] == GL_EYE_LINEAR)
                sb_add(b, "uniform vec4 mglEyePlane%d_%d;\n", u, cc);
        }

    /* inputs */
    sb_add(b, "layout(location=%d) in vec4 mglVertex;\n", MGLC_ATTR_VERTEX);
    if (c->need_normal)
        sb_add(b, "layout(location=%d) in vec3 mglNormal;\n", MGLC_ATTR_NORMAL);
    if (c->need_color)
        sb_add(b, "layout(location=%d) in vec4 mglColor;\n", MGLC_ATTR_COLOR);
    if (c->need_secondary)
        sb_add(b, "layout(location=%d) in vec4 mglSecondaryColor;\n", MGLC_ATTR_SECONDARY);
    if (c->need_fogcoord)
        sb_add(b, "layout(location=%d) in float mglFogCoord;\n", MGLC_ATTR_FOG);
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        if (c->unit.need_texcoord[u])
            sb_add(b, "layout(location=%d) in vec4 mglTexCoord%d;\n", MGLC_ATTR_TEXCOORD(u), u);
    if (c->line_stipple)
        sb_add(b, "layout(location=%d) in float mglLineI;   /* distance along the line, pixels */\n", MGLC_ATTR_LINE_I);

    /* outputs */
    sb_add(b, "%sout vec4 mglColorFront;\n", c->flat ? "flat " : "");
    if (c->two_sided)
        sb_add(b, "%sout vec4 mglColorBack;\n", c->flat ? "flat " : "");
    if (!c->lighting || c->separate_specular)
        sb_add(b, "%sout vec4 mglSecondaryOut;\n", c->flat ? "flat " : "");
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        if (c->unit.unit_used[u])
            sb_add(b, "out vec4 mglTexCoordOut%d;\n", u);
    if (c->fog)
        sb_add(b, "out vec4 mglEyePos;\n");
    if (c->line_stipple)
        sb_add(b, "out float mglLineV;\n");

    sb_add(b, "void main()\n{\n");
    sb_add(b, "    vec4 mglEye = mglModelViewMatrix * mglVertex;\n");
    if (c->line_stipple)
        sb_add(b, "    mglLineV = mglLineI;\n");

    if (c->need_normal)
    {
        sb_add(b, "    vec3 mglN = mglNormalMatrix * mglNormal;\n");
        if (c->normalize || c->rescale)
        {
            if (c->normalize)
                sb_add(b, "    mglN = normalize(mglN);\n");
            else
                sb_add(b, "    mglN *= mglNormalParams.x;\n");
        }
        else
            sb_add(b, "    mglN = normalize(mglN);\n");  /* lighting needs unit normals */
    }

    if (!c->lighting)
    {
        sb_add(b, "    mglColorFront = mglColor;\n");
        sb_add(b, "    mglSecondaryOut = mglSecondaryColor;\n");
    }
    else
    {
        /* material with color-material substitution: all four substituted
         * parameters are declared so the shader always compiles; the ones
         * not tracked by glColorMaterial fall back to the material state */
        sb_add(b, "    vec4 mglMatA0 = mglMatAmbient[0], mglMatA1 = mglMatAmbient[1];\n");
        sb_add(b, "    vec4 mglMatD0 = mglMatDiffuse[0], mglMatD1 = mglMatDiffuse[1];\n");
        sb_add(b, "    vec4 mglMatS0 = mglMatSpecular[0], mglMatS1 = mglMatSpecular[1];\n");
        sb_add(b, "    vec4 mglMatE0 = mglMatEmission[0], mglMatE1 = mglMatEmission[1];\n");

        if (c->color_material_mode)
        {
            switch (c->color_material_mode)
            {
                case GL_EMISSION:
                    sb_add(b, "    mglMatE0 = mglColor; mglMatE1 = mglColor;\n");
                    break;
                case GL_AMBIENT:
                    sb_add(b, "    mglMatA0 = mglColor; mglMatA1 = mglColor;\n");
                    break;
                case GL_DIFFUSE:
                    sb_add(b, "    mglMatD0 = mglColor; mglMatD1 = mglColor;\n");
                    break;
                case GL_SPECULAR:
                    sb_add(b, "    mglMatS0 = mglColor; mglMatS1 = mglColor;\n");
                    break;
                default:    /* AMBIENT_AND_DIFFUSE */
                    sb_add(b, "    mglMatA0 = mglColor; mglMatA1 = mglColor;\n");
                    sb_add(b, "    mglMatD0 = mglColor; mglMatD1 = mglColor;\n");
                    break;
            }
        }
        /* no color material: the mglMatAmbient[0] defaults declared above stand */

        sb_add(b, "    float mglSh0 = mglMatShininess[0], mglSh1 = mglMatShininess[1];\n");
        sb_add(b, "    vec4 mglColF = mglMatE0 + mglMatA0 * mglLightModelAmbient;\n");
        sb_add(b, "    vec4 mglSpecF = vec4(0.0);\n");
        if (c->two_sided)
        {
            sb_add(b, "    vec4 mglColB = mglMatE1 + mglMatA1 * mglLightModelAmbient;\n");
            sb_add(b, "    vec4 mglSpecB = vec4(0.0);\n");
        }
        if (c->local_viewer)
            sb_add(b, "    vec3 mglViewer = normalize(-mglEye.xyz);\n");
        else
            sb_add(b, "    vec3 mglViewer = vec3(0.0, 0.0, 1.0);\n");

        /* the "no special state" loop still needed for plain lights: always emit the generic loop */
        sb_add(b, "    for (int mglI = 0; mglI < %d; mglI++) {\n", MGLC_MAX_LIGHTS);
        sb_add(b, "        if (mglLightEnabled[mglI] == 0) continue;\n");
        sb_add(b, "        vec3 mglVP;\n");
        sb_add(b, "        float mglAtti = 1.0;\n");
        sb_add(b, "        float mglSpot = 1.0;\n");
        sb_add(b, "        vec3 mglDp = mglLightPosition[mglI].xyz * sign(mglLightPosition[mglI].w) - mglEye.xyz * step(0.5, mglLightPosition[mglI].w);\n");
        sb_add(b, "        if (mglLightPosition[mglI].w > 0.5) {\n");
        sb_add(b, "            float mglDist = length(mglDp);\n");
        sb_add(b, "            mglVP = mglDp / max(mglDist, 1e-6);\n");
        sb_add(b, "            mglAtti = 1.0 / (mglLightAtt[mglI].x + mglLightAtt[mglI].y * mglDist + mglLightAtt[mglI].z * mglDist * mglDist);\n");
        sb_add(b, "        } else {\n");
        sb_add(b, "            mglVP = normalize(mglLightPosition[mglI].xyz);\n");
        sb_add(b, "        }\n");
        sb_add(b, "        if (mglLightSpotCosCut[mglI] > -2.0) {\n");
        sb_add(b, "            float mglSd = dot(normalize(mglLightSpotDir[mglI]), -mglVP);\n");
        sb_add(b, "            if (mglSd <= mglLightSpotCosCut[mglI]) mglSpot = 0.0;\n");
        sb_add(b, "            else mglSpot = pow(max(mglSd, 0.0), mglLightSpotExp[mglI]);\n");
        sb_add(b, "        }\n");
        sb_add(b, "        float mglAtSp = mglAtti * mglSpot;\n");
        sb_add(b, "        float mglNdV = max(dot(mglN, mglVP), 0.0);\n");
        sb_add(b, "        mglColF += mglAtSp * (mglLightAmbient[mglI] * mglMatA0 + mglLightDiffuse[mglI] * mglMatD0 * mglNdV);\n");
        sb_add(b, "        vec3 mglHv = normalize(mglVP + mglViewer);\n");
        sb_add(b, "        float mglNdH = max(dot(mglN, mglHv), 0.0);\n");
        sb_add(b, "        mglSpecF += mglAtSp * mglLightSpecular[mglI] * mglMatS0 * ((mglNdH > 0.0) ? pow(mglNdH, mglSh0) : 0.0);\n");
        if (c->two_sided)
        {
            sb_add(b, "        float mglNdVb = max(dot(-mglN, mglVP), 0.0);\n");
            sb_add(b, "        mglColB += mglAtSp * (mglLightAmbient[mglI] * mglMatA1 + mglLightDiffuse[mglI] * mglMatD1 * mglNdVb);\n");
            sb_add(b, "        float mglNdHb = max(dot(-mglN, mglHv), 0.0);\n");
            sb_add(b, "        mglSpecB += mglAtSp * mglLightSpecular[mglI] * mglMatS1 * ((mglNdHb > 0.0) ? pow(mglNdHb, mglSh1) : 0.0);\n");
        }
        sb_add(b, "    }\n");

        if (c->separate_specular)
        {
            sb_add(b, "    mglColorFront = mglColF;\n");
            sb_add(b, "    mglSecondaryOut = mglSpecF;\n");
            if (c->two_sided)
                sb_add(b, "    mglColorBack = mglColB;\n");
        }
        else
        {
            sb_add(b, "    mglColorFront = mglColF + mglSpecF;\n");
            if (c->two_sided)
                sb_add(b, "    mglColorBack = mglColB + mglSpecB;\n");
        }
    }

    /* texture coordinate generation + texture matrix */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        if (!c->unit.unit_used[u])
            continue;

        int any_gen = 0;
        for (int cc = 0; cc < 4; cc++)
            if (c->unit.texgen[u][cc])
                any_gen = 1;

        if (!any_gen)
        {
            sb_add(b, "    mglTexCoordOut%d = mglTextureMatrix%d * mglTexCoord%d;\n", u, u, u);
            continue;
        }

        /* sphere/reflection maps need the reflection vector */
        int sphere = (c->unit.texgen[u][0] == GL_SPHERE_MAP || c->unit.texgen[u][1] == GL_SPHERE_MAP ||
                      c->unit.texgen[u][0] == GL_REFLECTION_MAP || c->unit.texgen[u][1] == GL_REFLECTION_MAP ||
                      c->unit.texgen[u][2] == GL_REFLECTION_MAP);
        if (sphere)
        {
            sb_add(b, "    vec3 mglU = normalize(mglEye.xyz);\n");
            sb_add(b, "    vec3 mglR = reflect(mglU, mglN);\n");
            sb_add(b, "    float mglMr = 2.0 * sqrt(mglR.x * mglR.x + mglR.y * mglR.y + (mglR.z + 1.0) * (mglR.z + 1.0));\n");
        }

        const char *comp[4] = { "s", "t", "r", "q" };

        sb_add(b, "    mglTexCoordOut%d = mglTextureMatrix%d * vec4(", u, u);
        for (int cc = 0; cc < 4; cc++)
        {
            const char *sep = cc ? ", " : "";
            int mode = c->unit.texgen[u][cc];

            if (!mode)
            {
                if (c->unit.need_texcoord[u])
                    sb_add(b, "%smglTexCoord%d.%s", sep, u, comp[cc]);
                else
                    sb_add(b, "%s%s", sep, cc == 3 ? "1.0" : "0.0");
                continue;
            }

            switch (mode)
            {
                case GL_OBJECT_LINEAR:
                    sb_add(b, "%sdot(mglObjectPlane%d_%d, mglVertex)", sep, u, cc);
                    break;
                case GL_EYE_LINEAR:
                    sb_add(b, "%sdot(mglEyePlane%d_%d, mglEye)", sep, u, cc);
                    break;
                case GL_SPHERE_MAP:
                    if (cc == 0)
                        sb_add(b, "%smglR.x / mglMr + 0.5", sep);
                    else if (cc == 1)
                        sb_add(b, "%smglR.y / mglMr + 0.5", sep);
                    else if (cc == 3)
                        sb_add(b, "%s1.0", sep);
                    else
                        sb_add(b, "%s0.0", sep);
                    break;
                case GL_NORMAL_MAP:
                    if (cc == 3)
                        sb_add(b, "%s1.0", sep);
                    else
                        sb_add(b, "%smglN.%s", sep, comp[cc]);
                    break;
                case GL_REFLECTION_MAP:
                    if (cc == 3)
                        sb_add(b, "%s1.0", sep);
                    else
                        sb_add(b, "%smglR.%s", sep, comp[cc]);
                    break;
                default:
                    sb_add(b, "%s0.0", sep);
                    break;
            }
        }
        sb_add(b, ");\n");
    }

    if (c->fog)
        sb_add(b, "    mglEyePos = mglEye;\n");

    if (c->clip_mask)
        for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
            if (c->clip_mask & (1 << i))
                sb_add(b, "    gl_ClipDistance[%d] = dot(mglClipPlane%d, mglEye);\n", i, i);

    if (c->point_atten)
    {
        sb_add(b, "    float mglEyeDist = length(mglEye.xyz);\n");
        sb_add(b, "    gl_PointSize = clamp(mglPointSizeParams.x * inversesqrt(dot(mglPointAtten.xyz, vec3(1.0, mglEyeDist, mglEyeDist * mglEyeDist))), mglPointSizeParams.y, mglPointSizeParams.z);\n");
    }

    sb_add(b, "    gl_Position = mglProjectionMatrix * mglEye;\n");
    sb_add(b, "}\n");
}

static void operand_str(char *out, size_t outsz, const char *v, int op, bool rgb);

#pragma mark fragment shader

/* operand of a combine source, written into |out| */
static void operand_str(char *out, size_t outsz, const char *v, int op, bool rgb)
{
    if (rgb)
    {
        switch (op)
        {
            case GL_SRC_COLOR: snprintf(out, outsz, "%s.rgb", v); return;
            case GL_ONE_MINUS_SRC_COLOR: snprintf(out, outsz, "(vec3(1.0) - %s.rgb)", v); return;
            case GL_SRC_ALPHA: snprintf(out, outsz, "vec3(%s.a)", v); return;
            case GL_ONE_MINUS_SRC_ALPHA: snprintf(out, outsz, "vec3(1.0 - %s.a)", v); return;
            default: snprintf(out, outsz, "%s.rgb", v); return;
        }
    }

    switch (op)
    {
        case GL_SRC_ALPHA: snprintf(out, outsz, "%s.a", v); return;
        case GL_ONE_MINUS_SRC_ALPHA: snprintf(out, outsz, "(1.0 - %s.a)", v); return;
        default: snprintf(out, outsz, "%s.a", v); return;
    }
}

/* source variable of a combine argument (writes a vec4 expression) */
static void sb_src_expr(SB *b, int src, int unit, const MGLFFPConfig *c, const char *prev, const char *primary, char *out, size_t outsz)
{
    if (src == GL_TEXTURE + unit || src == GL_TEXTURE)
        snprintf(out, outsz, "mglTex%d", unit);
    else if (src >= GL_TEXTURE0 && src < GL_TEXTURE0 + MGLC_MAX_COORDS)
        snprintf(out, outsz, "mglTex%d", src - GL_TEXTURE0);
    else if (src == GL_PREVIOUS)
        snprintf(out, outsz, "%s", prev);
    else if (src == GL_PRIMARY_COLOR)
        snprintf(out, outsz, "%s", primary);
    else if (src == GL_CONSTANT)
        snprintf(out, outsz, "mglEnvColor%d", unit);
    else if (src == GL_ZERO)
        snprintf(out, outsz, "vec4(0.0)");
    else if (src == GL_ONE)
        snprintf(out, outsz, "vec4(1.0)");
    else
        snprintf(out, outsz, "%s", prev);

    (void)b; (void)c;
}

static void sb_combine(SB *b, int combine, char *args[3], bool rgb)
{
    switch (combine)
    {
        case GL_REPLACE: sb_add(b, "%s", args[0]); break;
        case GL_MODULATE: sb_add(b, "(%s * %s)", args[0], args[1]); break;
        case GL_ADD: sb_add(b, "(%s + %s)", args[0], args[1]); break;
        case GL_ADD_SIGNED: sb_add(b, "(%s + %s - %s)", args[0], args[1], rgb ? "vec3(0.5)" : "0.5"); break;
        case GL_SUBTRACT: sb_add(b, "(%s - %s)", args[0], args[1]); break;
        case GL_INTERPOLATE: sb_add(b, "(%s * %s + %s * (%s - %s))", args[0], args[2], args[1], rgb ? "vec3(1.0)" : "1.0", args[2]); break;
        default: sb_add(b, "%s", args[0]); break;
    }
}

static void gen_fragment_shader(GLMContext ctx, const MGLFFPConfig *c, SB *b)
{
    sb_add(b, "#version 460 core\n\n");

    sb_add(b, "%sin vec4 mglColorFront;\n", c->flat ? "flat " : "");
    if (c->two_sided)
        sb_add(b, "%sin vec4 mglColorBack;\n", c->flat ? "flat " : "");
    if (!c->lighting || c->separate_specular)
        sb_add(b, "%sin vec4 mglSecondaryIn;\n", c->flat ? "flat " : "");
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        if (c->unit.unit_used[u])
            sb_add(b, "in vec4 mglTexCoordIn%d;\n", u);
    if (c->fog)
    {
        sb_add(b, "in vec4 mglEyePos;\n");
        if (c->fog_src == GL_FOG_COORD)
            sb_add(b, "in float mglFogCoordIn;\n");
    }

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        if (c->unit.unit_used[u])
        {
            sb_add(b, "uniform %s mglTex%d;\n",
                   c->unit.target[u] == GL_TEXTURE_3D ? "sampler3D" :
                   c->unit.target[u] == GL_TEXTURE_CUBE_MAP ? "samplerCube" : "sampler2D", u);
            if (c->unit.env_mode[u] == GL_COMBINE)
            {
                sb_add(b, "uniform vec4 mglEnvColor%d;\n", u);
                sb_add(b, "uniform vec4 mglEnvScale%d;     /* rgb scale, alpha scale, 0, 0 */\n", u);
            }
            else if (c->unit.env_mode[u] == GL_BLEND)
                sb_add(b, "uniform vec4 mglEnvColor%d;\n", u);
        }

    if (c->fog)
    {
        sb_add(b, "uniform vec4 mglFogColor;\n");
        sb_add(b, "uniform vec4 mglFogParams;      /* density, start, end, 1/(end-start) */\n");
    }

    if (c->alpha_test)
        sb_add(b, "uniform vec4 mglAlphaParams;    /* .x = reference value */\n");

    if (c->polygon_stipple)
        sb_add(b, "uniform sampler2D mglStipple;\n");
    if (c->line_stipple)
    {
        sb_add(b, "in float mglLineV;\n");
        sb_add(b, "uniform ivec2 mglLineStipple;   /* .x = pattern, .y = repeat factor */\n");
    }

    sb_add(b, "layout(location=0) out vec4 mglFragColor;\n");

    sb_add(b, "void main()\n{\n");

    sb_add(b, "    vec4 mglPri = ");
    if (c->two_sided)
        sb_add(b, "gl_FrontFacing ? mglColorFront : mglColorBack;\n");
    else
        sb_add(b, "mglColorFront;\n");

    /* sample every unit used by the environment chain (incl. crossbar) */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        if (!c->unit.unit_used[u])
            continue;

        if (c->unit.coord_replace[u])
            sb_add(b, "    vec4 mglTexCoordIn%d = vec4(gl_PointCoord, 0.0, 1.0);\n", u);

        sb_add(b, "    vec4 mglTex%d = ", u);
        if (c->unit.target[u] == GL_TEXTURE_1D)
            sb_add(b, "texture(mglTex%d, vec2(mglTexCoordIn%d.s, 0.5));\n", u, u);
        else if (c->unit.target[u] == GL_TEXTURE_3D || c->unit.target[u] == GL_TEXTURE_CUBE_MAP)
            sb_add(b, "texture(mglTex%d, mglTexCoordIn%d.xyz);\n", u, u);
        else
            sb_add(b, "texture(mglTex%d, mglTexCoordIn%d.xy);\n", u, u);
    }

    sb_add(b, "    vec4 c = mglPri;\n");

    /* texture environment chain */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        if (!c->unit.unit_used[u])
            continue;

        switch (c->unit.env_mode[u])
        {
            case GL_REPLACE:
                sb_add(b, "    c = mglTex%d;\n", u);
                break;
            case GL_MODULATE:
                sb_add(b, "    c = c * mglTex%d;\n", u);
                break;
            case GL_DECAL:
                sb_add(b, "    c = vec4(mix(c.rgb, mglTex%d.rgb, mglTex%d.a), c.a);\n", u, u);
                break;
            case GL_BLEND:
                sb_add(b, "    c = vec4(mix(c.rgb, mglEnvColor%d.rgb, mglTex%d.rgb), c.a * mglTex%d.a);\n", u, u, u);
                break;
            case GL_ADD:
                sb_add(b, "    c = vec4(c.rgb + mglTex%d.rgb, c.a * mglTex%d.a);\n", u, u);
                break;
            case GL_COMBINE:
            {
                char a0[64], a1[64], a2[64], e0[64], e1[64], e2[64];
                char *args[3];

                /* rgb */
                sb_src_expr(b, c->unit.src[0][u], u, c, "c", "mglPri", a0, sizeof a0);
                sb_src_expr(b, c->unit.src[1][u], u, c, "c", "mglPri", a1, sizeof a1);
                sb_src_expr(b, c->unit.src[2][u], u, c, "c", "mglPri", a2, sizeof a2);

                if (c->unit.combine_rgb[u] == GL_DOT3_RGB || c->unit.combine_rgb[u] == GL_DOT3_RGBA)
                {
                    operand_str(e0, sizeof e0, a0, c->unit.op_rgb[0][u], true);
                    sb_add(b, "    vec3 mglDot%d = 4.0 * dot(%s - vec3(0.5), ", u, e0);
                    operand_str(e1, sizeof e1, a1, c->unit.op_rgb[1][u], true);
                    sb_add(b, "%s - vec3(0.5));\n", e1);

                    if (c->unit.combine_rgb[u] == GL_DOT3_RGBA)
                        sb_add(b, "    c = vec4(mglDot%d) * mglEnvScale%d.y;\n", u, u);
                    else
                        sb_add(b, "    c.rgb = mglDot%d * mglEnvScale%d.x;\n", u, u);
                }
                else
                {
                    operand_str(e0, sizeof e0, a0, c->unit.op_rgb[0][u], true);
                    operand_str(e1, sizeof e1, a1, c->unit.op_rgb[1][u], true);
                    operand_str(e2, sizeof e2, a2, c->unit.op_rgb[2][u], true);
                    args[0] = e0; args[1] = e1; args[2] = e2;
                    sb_add(b, "    c.rgb = ");
                    sb_combine(b, c->unit.combine_rgb[u], args, true);
                    sb_add(b, " * mglEnvScale%d.x;\n", u);
                }

                if (c->unit.combine_rgb[u] != GL_DOT3_RGBA)
                {
                    /* alpha */
                    sb_src_expr(b, c->unit.src[3][u], u, c, "c", "mglPri", a0, sizeof a0);
                    sb_src_expr(b, c->unit.src[4][u], u, c, "c", "mglPri", a1, sizeof a1);
                    sb_src_expr(b, c->unit.src[5][u], u, c, "c", "mglPri", a2, sizeof a2);
                    operand_str(e0, sizeof e0, a0, c->unit.op_a[0][u], false);
                    operand_str(e1, sizeof e1, a1, c->unit.op_a[1][u], false);
                    operand_str(e2, sizeof e2, a2, c->unit.op_a[2][u], false);
                    args[0] = e0; args[1] = e1; args[2] = e2;
                    sb_add(b, "    c.a = ");
                    sb_combine(b, c->unit.combine_a[u], args, false);
                    sb_add(b, " * mglEnvScale%d.y;\n", u);
                }
                break;
            }
            default:
                break;
        }
    }

    if (c->lighting && c->separate_specular)
        sb_add(b, "    c.rgb += mglSecondaryIn.rgb;\n");

    if (c->fog)
    {
        sb_add(b, "    float mglFogF;\n");
        sb_add(b, "    float mglFogD = ");
        if (c->fog_src == GL_FOG_COORD)
            sb_add(b, "mglFogCoordIn;\n");
        else
            sb_add(b, "length(mglEyePos.xyz);\n");

        switch (c->fog_mode)
        {
            case GL_LINEAR:
                sb_add(b, "    mglFogF = clamp((mglFogParams.z - mglFogD) * mglFogParams.w, 0.0, 1.0);\n");
                break;
            case GL_EXP:
                sb_add(b, "    mglFogF = clamp(exp(-mglFogParams.x * mglFogD), 0.0, 1.0);\n");
                break;
            default:    /* EXP2 */
                sb_add(b, "    mglFogF = clamp(exp(-mglFogParams.x * mglFogParams.x * mglFogD * mglFogD), 0.0, 1.0);\n");
                break;
        }
        sb_add(b, "    c.rgb = mix(c.rgb, mglFogColor.rgb, mglFogF);\n");
    }

    if (c->alpha_test)
    {
        const char *expr = NULL;
        switch (c->alpha_func)
        {
            case GL_NEVER: expr = "false"; break;
            case GL_LESS: expr = "c.a < mglAlphaParams.x"; break;
            case GL_EQUAL: expr = "abs(c.a - mglAlphaParams.x) < 0.0001"; break;
            case GL_LEQUAL: expr = "c.a <= mglAlphaParams.x"; break;
            case GL_GREATER: expr = "c.a > mglAlphaParams.x"; break;
            case GL_NOTEQUAL: expr = "abs(c.a - mglAlphaParams.x) >= 0.0001"; break;
            case GL_GEQUAL: expr = "c.a >= mglAlphaParams.x"; break;
            default: expr = "true"; break;    /* ALWAYS */
        }
        sb_add(b, "    if (!(%s)) discard;\n", expr);
    }

    if (c->polygon_stipple)
        sb_add(b, "    if (texture(mglStipple, gl_FragCoord.xy / 32.0).r < 0.5) discard;\n");

    if (c->line_stipple)
    {
        /* pattern bit 15 (MSB) is the first pixel of the line */
        sb_add(b, "    int mglLIdx = int(floor(mglLineV / float(mglLineStipple.y) + 1e-4));\n");
        sb_add(b, "    int mglLBit = 15 - (mglLIdx %% 16);\n");
        sb_add(b, "    if (((mglLineStipple.x >> mglLBit) & 1) == 0) discard;\n");
    }

    sb_add(b, "    mglFragColor = c;\n");
    sb_add(b, "}\n");
}

#pragma mark program build / cache

static GLuint compile_stage(GLMContext ctx, GLenum type, const char *src)
{
    GLuint shader = mglCreateShader(ctx, type);

    if (!shader)
        return 0;

    mglShaderSource(ctx, shader, 1, (const GLchar *const *)&src, NULL);
    mglCompileShader(ctx, shader);

    GLint ok = 0;
    mglGetShaderiv(ctx, shader, GL_COMPILE_STATUS, &ok);

    if (!ok)
    {
        GLchar log[2048] = { 0 };
        GLsizei len = 0;
        mglGetShaderInfoLog(ctx, shader, sizeof log - 1, &len, log);
        fprintf(stderr, "MGL compat: fixed-function %s shader failed:\n%s\n-- source --\n%s\n",
                type == GL_VERTEX_SHADER ? "vertex" : "fragment", log, src);
        mglDeleteShader(ctx, shader);
        return 0;
    }

    return shader;
}

/* build (or fetch) the program for the current state; 0 on failure */
static GLuint ffp_program(GLMContext ctx, const MGLFFPConfig *c, const MGLFFPKey *key)
{
    struct MGLExtState_compat *s = CS();
    MGLFFPCacheEnt *cache = (MGLFFPCacheEnt *)s->ffp_cache;

    if (!cache)
    {
        cache = (MGLFFPCacheEnt *)calloc(MGLC_FFP_CACHE, sizeof(MGLFFPCacheEnt));
        if (!cache)
            return 0;
        s->ffp_cache = cache;
    }

    int slot = -1;
    for (int i = 0; i < MGLC_FFP_CACHE; i++)
    {
        if (cache[i].used && !memcmp(&cache[i].key, key, sizeof(MGLFFPKey)))
            return cache[i].failed ? 0 : cache[i].prog;
        if (!cache[i].used && slot < 0)
            slot = i;
    }

    if (slot < 0)
    {
        /* recycle: the first entry (LRU-ish; entries are rarely evicted) */
        slot = 0;
        if (cache[0].prog)
            mglDeleteProgram(ctx, cache[0].prog);
        memset(&cache[0], 0, sizeof cache[0]);
    }

    SB vs, fs;
    GLuint prog = 0;

    sb_init(&vs);
    sb_init(&fs);
    gen_vertex_shader(ctx, c, &vs);
    gen_fragment_shader(ctx, c, &fs);

    if (getenv("MGLC_DUMP"))
    {
        FILE *d = fopen("/tmp/mglc_vs.glsl", "w");
        if (d) { fputs(vs.s, d); fclose(d); }
        d = fopen("/tmp/mglc_fs.glsl", "w");
        if (d) { fputs(fs.s, d); fclose(d); }
    }

    GLuint v = compile_stage(ctx, GL_VERTEX_SHADER, vs.s);
    GLuint f = compile_stage(ctx, GL_FRAGMENT_SHADER, fs.s);

    if (getenv("MGLC_DEBUG") && (!v || !f))
    {
        fprintf(stderr, "MGLC: stage compile failed (v=%u f=%u)\n", v, f);
        FILE *fdbg = fopen("/tmp/mglc_vs.glsl", "w");
        if (fdbg) { fputs(vs.s, fdbg); fclose(fdbg); }
        fdbg = fopen("/tmp/mglc_fs.glsl", "w");
        if (fdbg) { fputs(fs.s, fdbg); fclose(fdbg); }
    }

    if (v && f)
    {
        prog = mglCreateProgram(ctx);
        mglAttachShader(ctx, prog, v);
        mglAttachShader(ctx, prog, f);
        mglLinkProgram(ctx, prog);

        GLint ok = 0;
        mglGetProgramiv(ctx, prog, GL_LINK_STATUS, &ok);

        if (!ok)
        {
            GLchar log[2048] = { 0 };
            GLsizei len = 0;
            mglGetProgramInfoLog(ctx, prog, sizeof log - 1, &len, log);
            fprintf(stderr, "MGL compat: fixed-function link failed:\n%s\n", log);
            mglDeleteProgram(ctx, prog);
            prog = 0;
        }
    }

    if (v) mglDeleteShader(ctx, v);
    if (f) mglDeleteShader(ctx, f);
    free(vs.s);
    free(fs.s);

    cache[slot].used = 1;
    cache[slot].failed = (prog == 0);
    cache[slot].prog = prog;
    cache[slot].sync = 0;
    cache[slot].key = *key;

    return prog;
}

#pragma mark stipple texture

/* (re)uploads the 32x32 polygon-stipple pattern to the internal texture on
 * unit MGLC_STIPPLE_UNIT; the fragment shader samples it with gl_FragCoord/32 */
static void sync_stipple_texture(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    /* FNV-1a over the pattern: skip the upload when nothing changed */
    uint64_t crc = 1469598103934665603ULL;
    for (int i = 0; i < 32 * 4; i++)
    {
        crc ^= s->polygon_stipple_pattern[i];
        crc *= 1099511628211ULL;
    }

    if (s->stipple_tex && s->stipple_serial == crc)
        return;

    GLubyte tex[32 * 32];

    /* GL layout: row 0 = bottom, byte x/8 bit x%8 (LSB = x = 0) */
    for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
            tex[y * 32 + x] = ((s->polygon_stipple_pattern[y * 4 + x / 8] >> (x % 8)) & 1) ? 255 : 0;

    if (!s->stipple_tex)
        mglGenTextures(ctx, 1, &s->stipple_tex);

    GLuint saved_active = STATE(active_texture);

    mglActiveTexture(ctx, GL_TEXTURE0 + MGLC_STIPPLE_UNIT);
    mglBindTexture(ctx, GL_TEXTURE_2D, s->stipple_tex);
    mglTexImage2D(ctx, GL_TEXTURE_2D, 0, GL_R8, 32, 32, 0, GL_RED, GL_UNSIGNED_BYTE, tex);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    mglTexParameteri(ctx, GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    mglActiveTexture(ctx, GL_TEXTURE0 + saved_active);

    s->stipple_serial = crc;
}

#pragma mark uniform sync

/* location lookup that tolerates missing uniforms (returns -1, no GL error) */
static GLint uni(GLMContext ctx, GLuint prog, const char *name)
{
    return mglGetUniformLocation(ctx, prog, name);
}

void mgl_compat_ffp_sync(GLMContext ctx, GLuint prog)
{
    struct MGLExtState_compat *s = CS();
    MGLFFPCacheEnt *cache = (MGLFFPCacheEnt *)s->ffp_cache;
    MGLFFPConfig c;
    GLint loc;

    if (!prog)
        return;

    ffp_build_config(ctx, &c);

    for (int i = 0; i < MGLC_FFP_CACHE; i++)
        if (cache && cache[i].used && cache[i].prog == prog)
        {
            /* the uniform table is re-derived after the MSL compile of the
             * first draw, so force a re-sync on the first few uses */
            if (cache[i].sync == s->state_serial && cache[i].draws >= 3)
                return;
            cache[i].draws++;
            if (cache[i].sync == s->state_serial && cache[i].draws < 3)
                cache[i].dirty_late = 1;
            cache[i].sync = s->state_serial;
            break;
        }

    /* matrices */
    loc = uni(ctx, prog, "mglModelViewMatrix");
    if (loc >= 0) mglUniformMatrix4fv(ctx, loc, 1, GL_FALSE, s->modelview_f);
    loc = uni(ctx, prog, "mglProjectionMatrix");
    if (loc >= 0) mglUniformMatrix4fv(ctx, loc, 1, GL_FALSE, s->projection_f);

    if (c.need_normal)
    {
        /* normal matrix = transpose(inverse(modelview)) */
        GLdouble inv[16], f[16];
        GLfloat nm[9];
        const GLdouble *mv = s->modelview.stack[s->modelview.depth].m;

        if (!mglc_matrix_invert(inv, mv))
            mglc_matrix_identity(inv);
        mglc_matrix_transpose(f, inv);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                nm[i * 3 + j] = (GLfloat)f[i * 4 + j];   /* mat3 column major from the 4x4 */

        loc = uni(ctx, prog, "mglNormalMatrix");
        if (loc >= 0) mglUniformMatrix3fv(ctx, loc, 1, GL_FALSE, nm);
    }

    if (c.rescale && !c.normalize)
    {
        GLdouble inv[16], *mv = s->modelview.stack[s->modelview.depth].m;
        GLfloat scale = 1.0f;

        if (mglc_matrix_invert(inv, mv))
        {
            double len = sqrt(mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]);
            if (len > 0)
                scale = (GLfloat)(1.0 / len);
        }

        loc = uni(ctx, prog, "mglNormalParams");
        if (loc >= 0)
        {
            GLfloat np[4] = { scale, 0.0f, 0.0f, 0.0f };
            mglUniform4fv(ctx, loc, 1, np);
        }
    }

    /* texture matrices (name is fixed width: mglTextureMatrix%d) */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        if (!c.unit.unit_used[u])
            continue;

        char name[32];
        snprintf(name, sizeof name, "mglTextureMatrix%d", u);
        loc = uni(ctx, prog, name);
        if (loc >= 0)
            mglUniformMatrix4fv(ctx, loc, 1, GL_FALSE, s->texture_f[u]);
    }

    /* clip planes */
    for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
    {
        if (!(c.clip_mask & (1 << i)))
            continue;

        char name[32];
        snprintf(name, sizeof name, "mglClipPlane%d", i);
        GLfloat v[4] = { (GLfloat)s->clip_plane[i][0], (GLfloat)s->clip_plane[i][1],
                         (GLfloat)s->clip_plane[i][2], (GLfloat)s->clip_plane[i][3] };
        loc = uni(ctx, prog, name);
        if (loc >= 0) mglUniform4fv(ctx, loc, 1, v);
    }

    /* point size attenuation */
    if (c.point_atten)
    {
        loc = uni(ctx, prog, "mglPointSizeParams");
        if (loc >= 0)
        {
            GLfloat sp[4] = { s->point_size, s->point_params[0], s->point_params[1], 0.0f };
            mglUniform4fv(ctx, loc, 1, sp);
        }
        loc = uni(ctx, prog, "mglPointAtten");
        if (loc >= 0)
        {
            GLfloat a[4] = { s->point_params[3], s->point_params[4], s->point_params[5], 0.0f };
            mglUniform4fv(ctx, loc, 1, a);
        }
    }

    /* lighting */
    if (c.lighting)
    {
        GLfloat amb[8][4], dif[8][4], spe[8][4], pos[8][4], sd[8][3];
        GLfloat se[8], sc[8], att[8][3];
        GLint en[8];

        for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
        {
            MGLCLight *l = &s->lights[i];
            memcpy(amb[i], l->ambient, 4 * sizeof(GLfloat));
            memcpy(dif[i], l->diffuse, 4 * sizeof(GLfloat));
            memcpy(spe[i], l->specular, 4 * sizeof(GLfloat));
            memcpy(pos[i], l->position, 4 * sizeof(GLfloat));
            for (int k = 0; k < 3; k++) sd[i][k] = l->spot_direction[k];
            se[i] = l->spot_exponent;
            sc[i] = (l->spot_cutoff >= 180.0f) ? -2.0f : (GLfloat)cos(l->spot_cutoff * (GLfloat)M_PI / 180.0f);
            att[i][0] = l->constant_attenuation;
            att[i][1] = l->linear_attenuation;
            att[i][2] = l->quadratic_attenuation;
            en[i] = l->enabled ? 1 : 0;
        }

        loc = uni(ctx, prog, "mglLightAmbient");
        if (loc >= 0) mglUniform4fv(ctx, loc, MGLC_MAX_LIGHTS, &amb[0][0]);
        loc = uni(ctx, prog, "mglLightDiffuse");
        if (loc >= 0) mglUniform4fv(ctx, loc, MGLC_MAX_LIGHTS, &dif[0][0]);
        loc = uni(ctx, prog, "mglLightSpecular");
        if (loc >= 0) mglUniform4fv(ctx, loc, MGLC_MAX_LIGHTS, &spe[0][0]);
        loc = uni(ctx, prog, "mglLightPosition");
        if (loc >= 0) mglUniform4fv(ctx, loc, MGLC_MAX_LIGHTS, &pos[0][0]);
        loc = uni(ctx, prog, "mglLightSpotDir");
        if (loc >= 0) mglUniform3fv(ctx, loc, MGLC_MAX_LIGHTS, &sd[0][0]);
        loc = uni(ctx, prog, "mglLightSpotExp");
        if (loc >= 0) mglUniform1fv(ctx, loc, MGLC_MAX_LIGHTS, se);
        loc = uni(ctx, prog, "mglLightSpotCosCut");
        if (loc >= 0) mglUniform1fv(ctx, loc, MGLC_MAX_LIGHTS, sc);
        loc = uni(ctx, prog, "mglLightAtt");
        if (loc >= 0) mglUniform3fv(ctx, loc, MGLC_MAX_LIGHTS, &att[0][0]);
        loc = uni(ctx, prog, "mglLightEnabled");
        if (loc >= 0) mglUniform1iv(ctx, loc, MGLC_MAX_LIGHTS, en);

        GLfloat ma[2][4], md[2][4], ms[2][4], me[2][4], sh[2];
        memcpy(ma[0], s->material_front.ambient, 16);
        memcpy(ma[1], s->material_back.ambient, 16);
        memcpy(md[0], s->material_front.diffuse, 16);
        memcpy(md[1], s->material_back.diffuse, 16);
        memcpy(ms[0], s->material_front.specular, 16);
        memcpy(ms[1], s->material_back.specular, 16);
        memcpy(me[0], s->material_front.emission, 16);
        memcpy(me[1], s->material_back.emission, 16);
        sh[0] = s->material_front.shininess;
        sh[1] = s->material_back.shininess;

        loc = uni(ctx, prog, "mglMatAmbient");
        if (loc >= 0) mglUniform4fv(ctx, loc, 2, &ma[0][0]);
        loc = uni(ctx, prog, "mglMatDiffuse");
        if (loc >= 0) mglUniform4fv(ctx, loc, 2, &md[0][0]);
        loc = uni(ctx, prog, "mglMatSpecular");
        if (loc >= 0) mglUniform4fv(ctx, loc, 2, &ms[0][0]);
        loc = uni(ctx, prog, "mglMatEmission");
        if (loc >= 0) mglUniform4fv(ctx, loc, 2, &me[0][0]);
        loc = uni(ctx, prog, "mglMatShininess");
        if (loc >= 0) mglUniform1fv(ctx, loc, 2, sh);
        loc = uni(ctx, prog, "mglLightModelAmbient");
        if (loc >= 0) mglUniform4fv(ctx, loc, 1, s->light_model_ambient);
    }

    /* texgen planes */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        for (int cc = 0; cc < 4; cc++)
        {
            char name[32];

            if (c.unit.texgen[u][cc] == GL_OBJECT_LINEAR)
            {
                snprintf(name, sizeof name, "mglObjectPlane%d_%d", u, cc);
                loc = uni(ctx, prog, name);
                if (loc >= 0)
                    mglUniform4fv(ctx, loc, 1, s->texgen[u].object_plane[cc]);
            }
            else if (c.unit.texgen[u][cc] == GL_EYE_LINEAR)
            {
                snprintf(name, sizeof name, "mglEyePlane%d_%d", u, cc);
                loc = uni(ctx, prog, name);
                if (loc >= 0)
                    mglUniform4fv(ctx, loc, 1, s->texgen[u].eye_plane[cc]);
            }
        }

    /* samplers + environment */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        if (!c.unit.unit_used[u])
            continue;

        char name[32];
        snprintf(name, sizeof name, "mglTex%d", u);
        loc = uni(ctx, prog, name);
        if (loc >= 0) mglUniform1i(ctx, loc, u);

        if (c.unit.env_mode[u] == GL_COMBINE || c.unit.env_mode[u] == GL_BLEND)
        {
            snprintf(name, sizeof name, "mglEnvColor%d", u);
            loc = uni(ctx, prog, name);
            if (loc >= 0) mglUniform4fv(ctx, loc, 1, s->texenv[u].env_color);
        }

        if (c.unit.env_mode[u] == GL_COMBINE)
        {
            snprintf(name, sizeof name, "mglEnvScale%d", u);
            loc = uni(ctx, prog, name);
            if (loc >= 0)
            {
                GLfloat es[4] = { s->texenv[u].scale_rgb, s->texenv[u].scale_alpha, 0.0f, 0.0f };
                mglUniform4fv(ctx, loc, 1, es);
            }
        }
    }

    if (c.fog)
    {
        loc = uni(ctx, prog, "mglFogColor");
        if (loc >= 0) mglUniform4fv(ctx, loc, 1, s->fog_color);

        loc = uni(ctx, prog, "mglFogParams");
        if (loc >= 0)
        {
            GLfloat fp[4] = { s->fog_density, s->fog_start, s->fog_end, s->fog_scale };
            mglUniform4fv(ctx, loc, 1, fp);
        }
    }

    if (c.alpha_test)
    {
        loc = uni(ctx, prog, "mglAlphaParams");
        if (loc >= 0)
        {
            GLfloat ap[4] = { s->alpha_ref, 0.0f, 0.0f, 0.0f };
            mglUniform4fv(ctx, loc, 1, ap);
        }
    }

    if (c.polygon_stipple)
    {
        sync_stipple_texture(ctx);
        loc = uni(ctx, prog, "mglStipple");
        if (loc >= 0) mglUniform1i(ctx, loc, MGLC_STIPPLE_UNIT);
    }

    if (c.line_stipple)
    {
        loc = uni(ctx, prog, "mglLineStipple");
        if (loc >= 0)
        {
            GLint ls[2] = { (GLint)(s->line_stipple_pattern & 0xFFFF), (GLint)s->line_stipple_repeat };
            mglUniform2iv(ctx, loc, 1, ls);
        }
    }
}

GLuint mgl_compat_ffp_program(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();
    MGLFFPConfig c;
    MGLFFPKey key;
    GLuint prog;

    ffp_build_config(ctx, &c);
    ffp_build_key(&c, &key);

    prog = ffp_program(ctx, &c, &key);

    if (!prog)
    {
        /* generation failed (bad state combination): fall back to a minimal
         * transform + color pipeline so the draw never crashes */
        memset(&c, 0, sizeof c);
        c.need_color = 1;
        memset(&c.unit, 0, sizeof c.unit);
        ffp_build_key(&c, &key);
        prog = ffp_program(ctx, &c, &key);
    }

    s->ffp_program = prog;

    return prog;
}

#pragma mark blit program (glDrawPixels / glBitmap / glCopyPixels)

GLuint mgl_compat_blit_program(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (s->blit_program)
        return s->blit_program;

    static const char vs[] =
        "#version 460 core\n"
        "layout(location=0) in vec2 mglPos;\n"
        "layout(location=3) in vec4 mglCol;\n"
        "layout(location=8) in vec2 mglUV;\n"
        "uniform vec4 mglRect;\n"        /* x0, y0, x1, y1 in clip space */
        "out vec2 mglUVOut;\n"
        "flat out vec4 mglColOut;\n"
        "void main() {\n"
        "    vec2 p = mix(mglRect.xy, mglRect.zw, mglPos);\n"
        "    gl_Position = vec4(p, 0.0, 1.0);\n"
        "    mglUVOut = mglUV;\n"
        "    mglColOut = mglCol;\n"
        "}\n";
    static const char fs[] =
        "#version 460 core\n"
        "in vec2 mglUVOut;\n"
        "flat in vec4 mglColOut;\n"
        "uniform sampler2D mglTex;\n"
        "uniform int mglMode;\n"          /* 0 = rgba, 1 = bitmap */
        "uniform float mglAlphaScale;\n"
        "layout(location=0) out vec4 mglFragColor;\n"
        "void main() {\n"
        "    vec4 t = texture(mglTex, mglUVOut);\n"
        "    if (mglMode == 0) mglFragColor = t;\n"
        "    else if (mglMode == 2) mglFragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "    else { if (t.r < 0.5) discard; mglFragColor = mglColOut; }\n"
        "}\n";

    GLuint prog = mglCreateProgram(ctx);
    GLuint v = compile_stage(ctx, GL_VERTEX_SHADER, vs);
    GLuint f = compile_stage(ctx, GL_FRAGMENT_SHADER, fs);

    if (!prog || !v || !f)
        return 0;

    mglAttachShader(ctx, prog, v);
    mglAttachShader(ctx, prog, f);
    mglLinkProgram(ctx, prog);

    GLint ok = 0;
    mglGetProgramiv(ctx, prog, GL_LINK_STATUS, &ok);

    if (!ok)
        return 0;

    mglDeleteShader(ctx, v);
    mglDeleteShader(ctx, f);

    s->blit_program = prog;
    return prog;
}
