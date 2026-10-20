/*
 * compat.c — compatibility-profile plumbing (compat domain).
 *
 *  * context creation with a profile (createGLMContextProfile, glm_context.c)
 *  * per-context dispatch: compat entry points are live only in compat contexts
 *  * profile-visible strings: GL_VERSION, GL_EXTENSIONS (+ GL_ARB_compatibility),
 *    GL_CONTEXT_PROFILE_MASK
 *  * fixed-function enable caps (GL_LIGHTING, GL_FOG, GL_TEXTURE_GEN_*, ...)
 *
 * The fixed-function pipeline itself is generated GLSL + streamed buffers
 * (compat_ffp.c / compat_vb.c), exactly how Mesa emulates it on hardware
 * without fixed function units.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_compat.h"
#include "mgl_compat.h"
#include "vertex_arrays.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

extern GLMContext createGLMContextProfile(GLenum format, GLenum type,
                                          GLenum depth_format, GLenum depth_type,
                                          GLenum stencil_format, GLenum stencil_type,
                                          GLenum profile);
extern void *CppCreateMGLRendererHeadless(void *glm_ctx);
extern GLuint mgl_num_extensions(void);            /* get.c */
extern const char *mgl_extension_name(GLuint i);   /* get.c */
/* (used by tests / debug via mgl_extension_name) */

#define CS() (ctx->state.ext_compat)

bool mgl_compat_is_compat(GLMContext ctx)
{
    return ctx && ctx->state.ext_compat && ctx->state.ext_compat->is_compat;
}

void mgl_compat_dirty(GLMContext ctx)
{
    if (CS()) CS()->state_serial++;
}

#pragma mark - fixed-function caps

/* GL_LIGHT0..GL_LIGHT7 */
static int light_index(GLenum cap)
{
    if (cap >= GL_LIGHT0 && cap < GL_LIGHT0 + MGLC_MAX_LIGHTS)
        return (int)(cap - GL_LIGHT0);
    return -1;
}

bool mgl_compat_valid_cap(GLMContext ctx, GLenum cap)
{
    if (!mgl_compat_is_compat(ctx))
        return false;

    if (light_index(cap) >= 0)
        return true;

    switch (cap)
    {
        case GL_LIGHTING:
        case GL_COLOR_MATERIAL:
        case GL_FOG:
        case GL_ALPHA_TEST:
        case GL_NORMALIZE:
        case GL_RESCALE_NORMAL:
        case GL_AUTO_NORMAL:
        case GL_COLOR_SUM:
        case GL_LINE_STIPPLE:
        case GL_POLYGON_STIPPLE:
        case GL_POINT_SMOOTH:
        case GL_TEXTURE_1D:
        case GL_TEXTURE_2D:
        case GL_TEXTURE_3D:
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_GEN_S:
        case GL_TEXTURE_GEN_T:
        case GL_TEXTURE_GEN_R:
        case GL_TEXTURE_GEN_Q:
        case GL_MAP1_INDEX:
        case GL_MAP1_TEXTURE_COORD_1:
        case GL_MAP1_TEXTURE_COORD_2:
        case GL_MAP1_TEXTURE_COORD_3:
        case GL_MAP1_TEXTURE_COORD_4:
        case GL_MAP1_VERTEX_3:
        case GL_MAP1_VERTEX_4:
        case GL_MAP1_COLOR_4:
        case GL_MAP1_NORMAL:
        case GL_MAP2_INDEX:
        case GL_MAP2_TEXTURE_COORD_1:
        case GL_MAP2_TEXTURE_COORD_2:
        case GL_MAP2_TEXTURE_COORD_3:
        case GL_MAP2_TEXTURE_COORD_4:
        case GL_MAP2_VERTEX_3:
        case GL_MAP2_VERTEX_4:
        case GL_MAP2_COLOR_4:
        case GL_MAP2_NORMAL:
            return true;
    }

    if (cap >= GL_CLIP_PLANE0 && cap < GL_CLIP_PLANE0 + MGLC_MAX_CLIP_PLANES)
        return true;

    return false;
}

/* evaluator map slot for GL_MAP1_x / GL_MAP2_x caps (-1 = not a map cap) */
static int eval_cap_slot(GLenum cap, int *is2d)
{
    switch (cap)
    {
        case GL_MAP1_INDEX: *is2d = 0; return 0;
        case GL_MAP1_TEXTURE_COORD_1: *is2d = 0; return 1;
        case GL_MAP1_TEXTURE_COORD_2: *is2d = 0; return 2;
        case GL_MAP1_TEXTURE_COORD_3: *is2d = 0; return 3;
        case GL_MAP1_TEXTURE_COORD_4: *is2d = 0; return 4;
        case GL_MAP1_VERTEX_3: *is2d = 0; return 5;
        case GL_MAP1_VERTEX_4: *is2d = 0; return 6;
        case GL_MAP1_COLOR_4: *is2d = 0; return 7;
        case GL_MAP1_NORMAL: *is2d = 0; return 8;
        case GL_MAP2_INDEX: *is2d = 1; return 0;
        case GL_MAP2_TEXTURE_COORD_1: *is2d = 1; return 1;
        case GL_MAP2_TEXTURE_COORD_2: *is2d = 1; return 2;
        case GL_MAP2_TEXTURE_COORD_3: *is2d = 1; return 3;
        case GL_MAP2_TEXTURE_COORD_4: *is2d = 1; return 4;
        case GL_MAP2_VERTEX_3: *is2d = 1; return 5;
        case GL_MAP2_VERTEX_4: *is2d = 1; return 6;
        case GL_MAP2_COLOR_4: *is2d = 1; return 7;
        case GL_MAP2_NORMAL: *is2d = 1; return 8;
    }
    return -1;
}

/* per-unit (active texture) texture target enables */
void mgl_compat_set_cap(GLMContext ctx, GLenum cap, GLboolean on)
{
    struct MGLExtState_compat *s = CS();
    int light = light_index(cap);

    if (light >= 0)
    {
        s->lights[light].enabled = on;
        mgl_compat_dirty(ctx);
        return;
    }

    {
        int is2d = 0;
        int slot = eval_cap_slot(cap, &is2d);
        if (slot >= 0)
        {
            if (is2d) s->map2[slot].enabled = on;
            else s->map1[slot].enabled = on;
            mgl_compat_dirty(ctx);
            return;
        }
    }

    switch (cap)
    {
        case GL_LIGHTING: s->lighting = on; break;
        case GL_COLOR_MATERIAL: s->color_material = on; break;
        case GL_FOG: s->fog = on; break;
        case GL_ALPHA_TEST: s->alpha_test = on; mgl_rs_dirty_pipeline(ctx); break;
        case GL_NORMALIZE: s->normalize = on; break;
        case GL_RESCALE_NORMAL: s->rescale_normal = on; break;
        case GL_AUTO_NORMAL: s->auto_normal = on; break;
        case GL_COLOR_SUM: s->separate_specular = on; break;
        case GL_LINE_STIPPLE: s->line_stipple = on; break;
        case GL_POLYGON_STIPPLE: s->polygon_stipple = on; break;
        case GL_POINT_SMOOTH: STATE(caps.point_smooth) = on; break;
        default:
        {
            if (cap >= GL_CLIP_PLANE0 && cap < GL_CLIP_PLANE0 + MGLC_MAX_CLIP_PLANES)
            {
                s->clip_plane_enabled[cap - GL_CLIP_PLANE0] = on;
                break;
            }

            /* texture target enables: per unit */
            GLuint unit = STATE(active_texture);
            if (unit < MGLC_MAX_COORDS)
            {
                switch (cap)
                {
                    case GL_TEXTURE_1D:
                    case GL_TEXTURE_2D:
                    case GL_TEXTURE_3D:
                    case GL_TEXTURE_CUBE_MAP:
                    {
                        /* unit target bitmask: bit 0 = 1D, 1 = 2D, 2 = 3D, 3 = CUBE */
                        int bit;
                        switch (cap)
                        {
                            case GL_TEXTURE_1D: bit = 0; break;
                            case GL_TEXTURE_2D: bit = 1; break;
                            case GL_TEXTURE_3D: bit = 2; break;
                            default: bit = 3; break;
                        }
                        uint32_t mask = s->unit_target_mask[unit];
                        if (on) mask |= (1u << bit);
                        else mask &= ~(1u << bit);
                        s->unit_target_mask[unit] = mask;
                        break;
                    }
                    default:
                        goto texgen;
                }
                break;
            }

        texgen:
            if (unit < MGLC_MAX_COORDS)
            {
                int c = -1;
                switch (cap)
                {
                    case GL_TEXTURE_GEN_S: c = 0; break;
                    case GL_TEXTURE_GEN_T: c = 1; break;
                    case GL_TEXTURE_GEN_R: c = 2; break;
                    case GL_TEXTURE_GEN_Q: c = 3; break;
                }
                if (c >= 0)
                {
                    s->texgen[unit].enabled[c] = on;
                    break;
                }
            }
            return;     /* not ours */
        }
    }

    mgl_compat_dirty(ctx);
}

GLboolean mgl_compat_get_cap(GLMContext ctx, GLenum cap)
{
    struct MGLExtState_compat *s = CS();
    int light = light_index(cap);

    if (light >= 0)
        return s->lights[light].enabled;

    {
        int is2d = 0;
        int slot = eval_cap_slot(cap, &is2d);
        if (slot >= 0)
            return is2d ? s->map2[slot].enabled : s->map1[slot].enabled;
    }

    switch (cap)
    {
        case GL_LIGHTING: return s->lighting;
        case GL_COLOR_MATERIAL: return s->color_material;
        case GL_FOG: return s->fog;
        case GL_ALPHA_TEST: return s->alpha_test;
        case GL_NORMALIZE: return s->normalize;
        case GL_RESCALE_NORMAL: return s->rescale_normal;
        case GL_AUTO_NORMAL: return s->auto_normal;
        case GL_COLOR_SUM: return s->separate_specular;
        case GL_LINE_STIPPLE: return s->line_stipple;
        case GL_POLYGON_STIPPLE: return s->polygon_stipple;
        case GL_POINT_SMOOTH: return STATE(caps.point_smooth);
    }

    if (cap >= GL_CLIP_PLANE0 && cap < GL_CLIP_PLANE0 + MGLC_MAX_CLIP_PLANES)
        return s->clip_plane_enabled[cap - GL_CLIP_PLANE0];

    GLuint unit = STATE(active_texture);
    if (unit < MGLC_MAX_COORDS)
    {
        switch (cap)
        {
            case GL_TEXTURE_1D: return (s->unit_target_mask[unit] & 0x1) != 0;
            case GL_TEXTURE_2D: return (s->unit_target_mask[unit] & 0x2) != 0;
            case GL_TEXTURE_3D: return (s->unit_target_mask[unit] & 0x4) != 0;
            case GL_TEXTURE_CUBE_MAP: return (s->unit_target_mask[unit] & 0x8) != 0;
            case GL_TEXTURE_GEN_S: return s->texgen[unit].enabled[0];
            case GL_TEXTURE_GEN_T: return s->texgen[unit].enabled[1];
            case GL_TEXTURE_GEN_R: return s->texgen[unit].enabled[2];
            case GL_TEXTURE_GEN_Q: return s->texgen[unit].enabled[3];
        }
    }

    return GL_FALSE;
}

#pragma mark - dispatch wrappers (compat behaviour of core entry points)

static void compat_enable(GLMContext ctx, GLenum cap)
{
    if (mgl_compat_valid_cap(ctx, cap))
    {
        mgl_compat_set_cap(ctx, cap, GL_TRUE);
        return;
    }

    mglEnable(ctx, cap);
}

static void compat_disable(GLMContext ctx, GLenum cap)
{
    if (mgl_compat_valid_cap(ctx, cap))
    {
        mgl_compat_set_cap(ctx, cap, GL_FALSE);
        return;
    }

    mglDisable(ctx, cap);
}

static GLboolean compat_is_enabled(GLMContext ctx, GLenum cap)
{
    if (mgl_compat_valid_cap(ctx, cap))
        return mgl_compat_get_cap(ctx, cap);

    return mglIsEnabled(ctx, cap);
}

static const GLubyte *compat_get_string(GLMContext ctx, GLenum name)
{
    struct MGLExtState_compat *s = CS();

    switch (name)
    {
        case GL_VERSION:
            return (const GLubyte *)s->version;
        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)s->glsl_version;
        case GL_EXTENSIONS:
            return (const GLubyte *)"GL_ARB_compatibility";   /* single-token list */
    }

    return mglGetString(ctx, name);
}

static const GLubyte *compat_get_stringi(GLMContext ctx, GLenum name, GLuint index)
{
    if (name == GL_EXTENSIONS)
    {
        if (index == 0)
            return (const GLubyte *)"GL_ARB_compatibility";
        return mglGetStringi(ctx, name, index - 1);
    }

    return mglGetStringi(ctx, name, index);
}

/* VAO 0 is a real vertex array in a compatibility context */
static void compat_bind_vertex_array(GLMContext ctx, GLuint array)
{
    struct MGLExtState_compat *s = CS();

    if (array == 0)
    {
        VertexArray *ptr = s->vao0;

        STATE(var.vertex_array_binding) = 0;
        STATE(var.element_array_buffer_binding) = (ptr->element_array.buffer) ? ptr->element_array.buffer->name : 0;
        STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = ptr->element_array.buffer;

        if (STATE(vao) != ptr)
        {
            STATE(vao) = ptr;
            STATE(dirty_bits) |= DIRTY_VAO;
        }
        return;
    }

    mglBindVertexArray(ctx, array);
}

#pragma mark - init / dispatch install

/* forward decls of the compat entry points (compat_*.c) */
void mgl_compat_init_dispatch(GLMContext ctx);

#include "compat_dispatch.h"

void mgl_compat_init_state(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    /* matrix stacks: identity */
    s->matrix_mode = GL_MODELVIEW;
    for (int i = 0; i < 4; i++)
    {
        mglc_matrix_identity(s->modelview.stack[i].m);
        mglc_matrix_identity(s->projection.stack[i].m);
        mglc_matrix_identity(s->color.stack[i].m);
        for (int u = 0; u < MGLC_MAX_COORDS; u++)
            mglc_matrix_identity(s->texture[u].stack[i].m);
    }
    memcpy(s->modelview_current.m, s->modelview.stack[0].m, sizeof(MGLCMat));

    /* the float caches the generated shaders read from must start at identity
     * too (they are only refreshed by matrix_changed() on matrix calls) */
    for (int i = 0; i < 16; i++)
    {
        s->modelview_f[i] = (GLfloat)s->modelview.stack[0].m[i];
        s->projection_f[i] = (GLfloat)s->projection.stack[0].m[i];
        for (int u = 0; u < MGLC_MAX_COORDS; u++)
            s->texture_f[u][i] = (GLfloat)s->texture[u].stack[0].m[i];
    }

    /* current vertex state */
    s->cur_color[0] = s->cur_color[1] = s->cur_color[2] = s->cur_color[3] = 1.0f;
    s->cur_normal[0] = 0.0f; s->cur_normal[1] = 0.0f; s->cur_normal[2] = 1.0f;
    s->cur_secondary[3] = 1.0f;
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        s->cur_texcoord[u][0] = s->cur_texcoord[u][1] = s->cur_texcoord[u][3] = 0.0f;
        s->cur_texcoord[u][2] = 1.0f;   /* (s,t,r,q) = (0,0,0,1) per spec */
        s->raster_texcoord[u][0] = s->raster_texcoord[u][1] = s->raster_texcoord[u][3] = 0.0f;
        s->raster_texcoord[u][2] = 0.0f;
    }
    s->cur_edgeflag = GL_TRUE;
    s->cur_index = 1.0f;

    /* lighting defaults (GL 1.0 spec tables) */
    s->light_model_ambient[0] = 0.2f; s->light_model_ambient[1] = 0.2f;
    s->light_model_ambient[2] = 0.2f; s->light_model_ambient[3] = 1.0f;
    for (int i = 0; i < MGLC_MAX_LIGHTS; i++)
    {
        MGLCLight *l = &s->lights[i];
        l->ambient[0] = l->ambient[1] = l->ambient[2] = 0.0f; l->ambient[3] = 1.0f;
        if (i == 0)
        {
            l->diffuse[0] = l->diffuse[1] = l->diffuse[2] = l->diffuse[3] = 1.0f;
            l->specular[0] = l->specular[1] = l->specular[2] = l->specular[3] = 1.0f;
        }
        else
        {
            l->diffuse[0] = l->diffuse[1] = l->diffuse[2] = l->diffuse[3] = 0.0f;
            l->specular[0] = l->specular[1] = l->specular[2] = l->specular[3] = 0.0f;
        }
        l->position[0] = 0.0f; l->position[1] = 0.0f; l->position[2] = 1.0f; l->position[3] = 0.0f;
        l->spot_direction[0] = 0.0f; l->spot_direction[1] = 0.0f; l->spot_direction[2] = -1.0f; l->spot_direction[3] = 0.0f;
        l->spot_exponent = 0.0f;
        l->spot_cutoff = 180.0f;
        l->constant_attenuation = 1.0f;
        l->linear_attenuation = 0.0f;
        l->quadratic_attenuation = 0.0f;
    }

    MGLCMaterial *mf = &s->material_front, *mb = &s->material_back;
    for (int k = 0; k < 2; k++)
    {
        MGLCMaterial *m = k ? mb : mf;
        m->ambient[0] = 0.2f; m->ambient[1] = 0.2f; m->ambient[2] = 0.2f; m->ambient[3] = 1.0f;
        m->diffuse[0] = 0.8f; m->diffuse[1] = 0.8f; m->diffuse[2] = 0.8f; m->diffuse[3] = 1.0f;
        m->specular[0] = m->specular[1] = m->specular[2] = 0.0f; m->specular[3] = 1.0f;
        m->emission[0] = m->emission[1] = m->emission[2] = 0.0f; m->emission[3] = 1.0f;
        m->shininess = 0.0f;
    }
    s->color_material_face = GL_FRONT_AND_BACK;
    s->color_material_mode = GL_AMBIENT_AND_DIFFUSE;

    /* fog */
    s->fog_mode = GL_EXP;
    s->fog_density = 1.0f;
    s->fog_start = 0.0f; s->fog_end = 1.0f;
    s->fog_color[3] = 1.0f;
    s->fog_coord_src = GL_FRAGMENT_DEPTH;
    mgl_compat_fog_recompute(ctx);

    /* alpha test */
    s->alpha_func = GL_ALWAYS;
    s->alpha_ref = 0.0f;

    /* texenv */
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        MGLCTexEnv *e = &s->texenv[u];
        e->env_mode = GL_MODULATE;
        e->combine_rgb = GL_MODULATE;
        e->combine_alpha = GL_MODULATE;
        e->src[0] = GL_TEXTURE; e->src[1] = GL_PREVIOUS; e->src[2] = GL_CONSTANT;
        e->src[3] = GL_TEXTURE; e->src[4] = GL_PREVIOUS; e->src[5] = GL_CONSTANT;
        e->op_rgb[0] = GL_SRC_COLOR; e->op_rgb[1] = GL_SRC_COLOR; e->op_rgb[2] = GL_SRC_ALPHA;
        e->op_alpha[0] = GL_SRC_ALPHA; e->op_alpha[1] = GL_SRC_ALPHA; e->op_alpha[2] = GL_SRC_ALPHA;
        e->scale_rgb = 1.0f; e->scale_alpha = 1.0f;
        e->env_color[3] = 1.0f;
        /* texgen planes zero */
        for (int c = 0; c < 4; c++)
        {
            s->texgen[u].mode[c] = GL_EYE_LINEAR;
            s->texgen[u].object_plane[c][0] = (c == 0) ? 1.0f : 0.0f;
            s->texgen[u].eye_plane[c][0] = (c == 0) ? 1.0f : 0.0f;
        }
    }

    /* misc */
    s->shade_model = GL_SMOOTH;
    s->polygon_mode_front = GL_FILL;
    s->polygon_mode_back = GL_FILL;
    s->point_size = 1.0f;
    s->point_params[0] = 0.0f;   /* point size min */
    s->point_params[1] = 1.0f;   /* point size max */
    s->point_params[2] = 1.0f;   /* fade threshold */
    s->point_params[3] = 1.0f;   /* a */
    s->point_params[4] = 0.0f;   /* b */
    s->point_params[5] = 0.0f;   /* c */
    s->line_stipple_pattern = 0xFFFF;
    s->line_stipple_repeat = 1;
    memset(s->polygon_stipple_pattern, 0xFF, sizeof(s->polygon_stipple_pattern));
    for (int i = 0; i < MGLC_MAX_CLIP_PLANES; i++)
        s->clip_plane[i][3] = 0.0;
    s->zoom_x = 1.0f; s->zoom_y = 1.0f;
    s->raster_pos[0] = s->raster_pos[1] = s->raster_pos[2] = 0.0; s->raster_pos[3] = 1.0;
    s->raster_color[0] = s->raster_color[1] = s->raster_color[2] = s->raster_color[3] = 1.0f;
    s->raster_valid = GL_TRUE;

    /* pixel maps: identity (a single entry mapping everything to itself) */
    for (int i = 0; i < 10; i++)
    {
        s->pixel_map_size[i] = 1;
        s->pixel_map[i] = (GLfloat *)calloc(2, sizeof(GLfloat));
        if (s->pixel_map[i]) { s->pixel_map[i][0] = 1.0f; s->pixel_map[i][1] = 1.0f; }
    }
    for (int i = 0; i < 16; i++)
        s->pixel_transfer[i] = 0.0f;
    /* GL defaults: color scales (and depth scale) are 1, not 0 */
    s->pixel_transfer[4] = s->pixel_transfer[5] = s->pixel_transfer[6] =
    s->pixel_transfer[7] = s->pixel_transfer[12] = 1.0f;

    /* evaluators */
    s->map_grid1_n = 1; s->map_grid2_nu = 1; s->map_grid2_nv = 1;

    /* render mode */
    s->rmode.mode = GL_RENDER;

    s->state_serial = 1;
}

void mgl_ext_init_compat(GLMContext ctx)
{
    if (mglNextContextProfile() != GL_CONTEXT_COMPATIBILITY_PROFILE_BIT)
        return;     /* core context: compat entry points stay stubs */

    struct MGLExtState_compat *s = (struct MGLExtState_compat *)calloc(1, sizeof(struct MGLExtState_compat));

    if (!s)
        return;

    ctx->state.ext_compat = s;
    s->is_compat = GL_TRUE;

    s->version[0] = 0;
    snprintf(s->version, sizeof(s->version), "4.6.0 MGL compat (Metal)");
    snprintf(s->glsl_version, sizeof(s->glsl_version), "4.60 compatibility");

    /* profile-visible state */
    STATE(var.context_profile_mask) = GL_CONTEXT_COMPATIBILITY_PROFILE_BIT;
    STATE(var.major_version) = 4;
    STATE(var.minor_version) = 6;

    /* default fixed-function state */
    mgl_compat_init_state(ctx);

    /* VAO 0: a real (internal) vertex array object */
    s->vao0 = newVAO(ctx, 0);
    STATE(vao) = s->vao0;
    STATE(dirty_bits) |= DIRTY_VAO;

    /* compat entry points go live */
    mgl_compat_init_dispatch(ctx);
}

#pragma mark - glGet pnames of the compat domain

bool mgl_ext_get_compat(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    struct MGLExtState_compat *s = CS();

    if (!s || !s->is_compat)
        return false;

    switch (pname)
    {
        case GL_CONTEXT_PROFILE_MASK:
            mgl_get_store(type, data, 0, GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);
            return true;

        case GL_NUM_EXTENSIONS:
            mgl_get_store(type, data, 0, mgl_num_extensions() + 1);
            return true;

        case GL_MAJOR_VERSION: mgl_get_store(type, data, 0, 4); return true;
        case GL_MINOR_VERSION: mgl_get_store(type, data, 0, 6); return true;

        case GL_LIGHTING: case GL_COLOR_MATERIAL: case GL_FOG: case GL_ALPHA_TEST:
        case GL_NORMALIZE: case GL_RESCALE_NORMAL: case GL_AUTO_NORMAL: case GL_COLOR_SUM:
        case GL_LINE_STIPPLE: case GL_POLYGON_STIPPLE: case GL_POINT_SMOOTH:
        case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_GEN_S: case GL_TEXTURE_GEN_T: case GL_TEXTURE_GEN_R: case GL_TEXTURE_GEN_Q:
            mgl_get_store(type, data, 0, mgl_compat_get_cap(ctx, pname));
            return true;

        case GL_LIGHT0: case GL_LIGHT1: case GL_LIGHT2: case GL_LIGHT3:
        case GL_LIGHT4: case GL_LIGHT5: case GL_LIGHT6: case GL_LIGHT7:
            mgl_get_store(type, data, 0, s->lights[pname - GL_LIGHT0].enabled);
            return true;

        case GL_CLIP_PLANE0: case GL_CLIP_PLANE1: case GL_CLIP_PLANE2: case GL_CLIP_PLANE3:
        case GL_CLIP_PLANE4: case GL_CLIP_PLANE5: case GL_CLIP_PLANE6: case GL_CLIP_PLANE7:
            mgl_get_store(type, data, 0, s->clip_plane_enabled[pname - GL_CLIP_PLANE0]);
            return true;

        case GL_SHADE_MODEL: mgl_get_store(type, data, 0, s->shade_model); return true;

        case GL_MATRIX_MODE: mgl_get_store(type, data, 0, s->matrix_mode); return true;

        case GL_MAX_LIGHTS: mgl_get_store(type, data, 0, MGLC_MAX_LIGHTS); return true;
        case GL_MAX_CLIP_PLANES: mgl_get_store(type, data, 0, MGLC_MAX_CLIP_PLANES); return true;
        case GL_MAX_TEXTURE_COORDS: mgl_get_store(type, data, 0, MGLC_MAX_COORDS); return true;
        case GL_MAX_TEXTURE_UNITS: mgl_get_store(type, data, 0, MGLC_MAX_COORDS); return true;

        case GL_MODELVIEW_STACK_DEPTH:
            mgl_get_store(type, data, 0, s->modelview.depth + 1); return true;
        case GL_PROJECTION_STACK_DEPTH:
            mgl_get_store(type, data, 0, s->projection.depth + 1); return true;
        case GL_TEXTURE_STACK_DEPTH:
            mgl_get_store(type, data, 0, s->texture[STATE(active_texture) % MGLC_MAX_COORDS].depth + 1); return true;

        case GL_MAX_MODELVIEW_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_MATRIX_DEPTH); return true;
        case GL_MAX_PROJECTION_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_MATRIX_DEPTH); return true;
        case GL_MAX_TEXTURE_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_MATRIX_DEPTH); return true;
        case GL_MAX_ATTRIB_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_ATTRIB_STACK); return true;
        case GL_MAX_CLIENT_ATTRIB_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_CLIENT_STACK); return true;
        case GL_MAX_NAME_STACK_DEPTH: mgl_get_store(type, data, 0, MGLC_NAME_STACK_DEPTH); return true;
        case GL_MAX_LIST_NESTING: mgl_get_store(type, data, 0, MGLC_LIST_NESTING); return true;
        case GL_MAX_EVAL_ORDER: mgl_get_store(type, data, 0, MGLC_MAX_EVAL_ORDER); return true;
        case GL_MAX_PIXEL_MAP_TABLE: mgl_get_store(type, data, 0, 65536); return true;

        case GL_NAME_STACK_DEPTH: mgl_get_store(type, data, 0, s->rmode.name_depth); return true;

        case GL_LIST_MODE:
            mgl_get_store(type, data, 0, s->recording ? s->record_mode : 0); return true;
        case GL_LIST_INDEX:
            mgl_get_store(type, data, 0, s->recording ? s->record_list : 0); return true;
        case GL_LIST_BASE: mgl_get_store(type, data, 0, s->list_base); return true;

        case GL_RENDER_MODE: mgl_get_store(type, data, 0, s->rmode.mode); return true;

        case GL_CURRENT_COLOR:
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->cur_color[i]);
            return true;
        case GL_CURRENT_NORMAL:
            for (int i = 0; i < 3; i++) mgl_get_store(type, data, i, s->cur_normal[i]);
            return true;
        case GL_CURRENT_TEXTURE_COORDS:
        {
            GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->cur_texcoord[unit][i]);
            return true;
        }
        case GL_CURRENT_FOG_COORD: mgl_get_store(type, data, 0, s->cur_fogcoord); return true;
        case GL_CURRENT_INDEX: mgl_get_store(type, data, 0, s->cur_index); return true;
        case GL_EDGE_FLAG: mgl_get_store(type, data, 0, s->cur_edgeflag); return true;

        case GL_CURRENT_RASTER_POSITION:
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->raster_pos[i]);
            return true;
        case GL_CURRENT_RASTER_POSITION_VALID:
            mgl_get_store(type, data, 0, s->raster_valid); return true;
        case GL_CURRENT_RASTER_COLOR:
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->raster_color[i]);
            return true;
        case GL_CURRENT_RASTER_DISTANCE:
            mgl_get_store(type, data, 0, s->raster_distance); return true;
        case GL_CURRENT_RASTER_INDEX:
            mgl_get_store(type, data, 0, s->raster_index); return true;
        case GL_CURRENT_RASTER_TEXTURE_COORDS:
        {
            GLuint unit = STATE(active_texture) % MGLC_MAX_COORDS;
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->raster_texcoord[unit][i]);
            return true;
        }

        case GL_ZOOM_X: mgl_get_store(type, data, 0, s->zoom_x); return true;
        case GL_ZOOM_Y: mgl_get_store(type, data, 0, s->zoom_y); return true;

        case GL_FOG_COLOR:
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->fog_color[i]);
            return true;
        case GL_FOG_INDEX: mgl_get_store(type, data, 0, s->fog_index); return true;
        case GL_FOG_DENSITY: mgl_get_store(type, data, 0, s->fog_density); return true;
        case GL_FOG_START: mgl_get_store(type, data, 0, s->fog_start); return true;
        case GL_FOG_END: mgl_get_store(type, data, 0, s->fog_end); return true;
        case GL_FOG_MODE: mgl_get_store(type, data, 0, s->fog_mode); return true;
        case GL_FOG_COORD_SRC: mgl_get_store(type, data, 0, s->fog_coord_src); return true;

        case GL_ALPHA_TEST_FUNC: mgl_get_store(type, data, 0, s->alpha_func); return true;
        case GL_ALPHA_TEST_REF: mgl_get_store(type, data, 0, s->alpha_ref); return true;

        case GL_LINE_STIPPLE_PATTERN: mgl_get_store(type, data, 0, s->line_stipple_pattern); return true;
        case GL_LINE_STIPPLE_REPEAT: mgl_get_store(type, data, 0, s->line_stipple_repeat); return true;

        case GL_POINT_SIZE: mgl_get_store(type, data, 0, s->point_size); return true;
        case GL_POINT_SIZE_MIN: mgl_get_store(type, data, 0, s->point_params[0]); return true;
        case GL_POINT_SIZE_MAX: mgl_get_store(type, data, 0, s->point_params[1]); return true;
        case GL_POINT_FADE_THRESHOLD_SIZE: mgl_get_store(type, data, 0, s->point_params[2]); return true;
        case GL_POINT_DISTANCE_ATTENUATION:
            for (int i = 0; i < 3; i++) mgl_get_store(type, data, i, s->point_params[3 + i]);
            return true;

        case GL_COLOR_MATERIAL_FACE: mgl_get_store(type, data, 0, s->color_material_face); return true;
        case GL_COLOR_MATERIAL_PARAMETER: mgl_get_store(type, data, 0, s->color_material_mode); return true;

        case GL_LIGHT_MODEL_LOCAL_VIEWER: mgl_get_store(type, data, 0, s->local_viewer); return true;
        case GL_LIGHT_MODEL_TWO_SIDE: mgl_get_store(type, data, 0, s->two_sided); return true;
        case GL_LIGHT_MODEL_COLOR_CONTROL:
            mgl_get_store(type, data, 0, s->separate_specular ? GL_SEPARATE_SPECULAR_COLOR : GL_SINGLE_COLOR);
            return true;
        case GL_LIGHT_MODEL_AMBIENT:
            for (int i = 0; i < 4; i++) mgl_get_store(type, data, i, s->light_model_ambient[i]);
            return true;

        case GL_RGBA_MODE: mgl_get_store(type, data, 0, GL_TRUE); return true;
        case GL_INDEX_MODE: mgl_get_store(type, data, 0, GL_FALSE); return true;
        case GL_DOUBLEBUFFER: mgl_get_store(type, data, 0, GL_TRUE); return true;
        case GL_STEREO: mgl_get_store(type, data, 0, GL_FALSE); return true;
        case GL_AUX_BUFFERS: mgl_get_store(type, data, 0, 0); return true;
    }

    /* matrices */
    {
        GLdouble m[16];
        switch (pname)
        {
            case GL_MODELVIEW_MATRIX: case GL_PROJECTION_MATRIX:
            case GL_COLOR_MATRIX:
            {
                mgl_compat_get_matrix(ctx, pname, m);
                for (int i = 0; i < 16; i++) mgl_get_store(type, data, i, m[i]);
                return true;
            }
            case GL_TEXTURE_MATRIX:
            {
                mgl_compat_get_matrix(ctx, GL_TEXTURE_MATRIX, m);
                for (int i = 0; i < 16; i++) mgl_get_store(type, data, i, m[i]);
                return true;
            }
        }
    }

    return false;
}

#pragma mark - public API

/* the library's default core-profile context (captured before the switch) */
static GLMContext mgl_core_ctx = NULL;

void *mglCompatCreateContext(void)
{
    GLMContext ctx;

    /* remember the library's default (core) context for mglCoreContext() */
    if (!mgl_core_ctx)
        mgl_core_ctx = MGLgetCurrentContext();

    ctx = createGLMContextProfile(GL_RGBA, GL_UNSIGNED_BYTE,
                                             GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT,
                                             GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE,
                                             GL_CONTEXT_COMPATIBILITY_PROFILE_BIT);

    if (!ctx)
        return NULL;

    if (!ctx->state.ext_compat || !ctx->state.ext_compat->is_compat)
        return NULL;    /* profile request ignored */

    CppCreateMGLRendererHeadless(ctx);
    MGLsetCurrentContext(ctx);

    return ctx;
}

void *mglCoreContext(void)
{
    if (!mgl_core_ctx)
    {
        mgl_lazy_init();
        mgl_core_ctx = MGLgetCurrentContext();
    }

    return (void *)mgl_core_ctx;
}

void mglMakeContextCurrent(void *ctx)
{
    MGLsetCurrentContext((GLMContext)ctx);
}

int mglCurrentContextIsCompat(void)
{
    return mgl_compat_is_compat(MGLgetCurrentContext()) ? 1 : 0;
}
