/*
 * mgl_ext_compat.h — private state and hooks of the 'compat' domain
 * (OpenGL 4.6 compatibility profile, see agents/tasks/compat.md).
 *
 * Everything fixed-function is emulated the way Mesa emulates it on modern
 * hardware: CPU state (matrix stacks, lights, texenv ...) plus generated
 * GLSL programs, streamed into ordinary GL buffers / VAOs through the core
 * code paths.  No Metal-side code is needed, so MGLExtMetal_compat only
 * carries the profile flag for renderer queries.
 */
#ifndef MGL_EXT_COMPAT_H
#define MGL_EXT_COMPAT_H

#include <stdint.h>
#include <stdbool.h>
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_vattrib.h"
#include "mgl_compat_enums.h"

/* compat limits (GL 4.6 compat profile minimums / common hardware values) */
#define MGLC_MAX_LIGHTS        8
#define MGLC_MAX_COORDS        8      /* gl_MultiTexCoord0..7 (MAX_ATTRIBS - 8) */
#define MGLC_MAX_CLIP_PLANES   8
#define MGLC_MATRIX_DEPTH      64     /* modelview; spec minimum 32 */
#define MGLC_MAX_TEXCOORD_ATTR (MAX_ATTRIBS - 8)
#define MGLC_STIPPLE_UNIT      8      /* texture unit holding stipple/blit textures (FFP uses 0..7) */
#define MGLC_LEGACY_TEX_MAX 64
#define MGLC_TEX_PRIO_MAX      65536  /* tracked glPrioritizeTextures values */
#define MGLC_ATTRIB_STACK      16
#define MGLC_CLIENT_STACK      16
#define MGLC_NAME_STACK        128
#define MGLC_LIST_MAX          65536
#define MGLC_LIST_NESTING      64
#define MGLC_LIST_OPS          96
#define MGLC_MAX_EVAL_ORDER    30
#define MGLC_NAME_STACK_DEPTH  128
#define MGLC_SELECT_HITS       4096
#define MGLC_FEEDBACK_FLOATS   (256 * 1024)

/* ---- fixed locations shared by generated and translated shaders ---------- */
#define MGLC_ATTR_VERTEX      0
#define MGLC_ATTR_LINE_I      1      /* line stipple: distance along the line (pixels) */
#define MGLC_ATTR_NORMAL      2
#define MGLC_ATTR_COLOR       3
#define MGLC_ATTR_SECONDARY   4
#define MGLC_ATTR_FOG         5
#define MGLC_ATTR_EDGEFLAG    6
#define MGLC_ATTR_INDEX       7
#define MGLC_ATTR_TEXCOORD(i) (8 + (i))

#pragma mark matrices

typedef struct {
    GLdouble m[16];               /* column major, like GL */
} MGLCMat;

typedef struct {
    MGLCMat  stack[MGLC_MATRIX_DEPTH];
    int      depth;               /* 0 = bottom */
} MGLCMatrixStack;

#pragma mark lighting

typedef struct {
    GLboolean enabled;
    GLfloat   ambient[4];
    GLfloat   diffuse[4];
    GLfloat   specular[4];
    GLfloat   position[4];        /* user value (object space at call time) */
    GLfloat   spot_direction[4];  /* user value, w unused */
    GLfloat   spot_exponent;
    GLfloat   spot_cutoff;        /* degrees, 180 = no spot */
    GLfloat   constant_attenuation;
    GLfloat   linear_attenuation;
    GLfloat   quadratic_attenuation;
} MGLCLight;

typedef struct {
    GLfloat ambient[4];
    GLfloat diffuse[4];
    GLfloat specular[4];
    GLfloat emission[4];
    GLfloat shininess;
    GLint   color_indexes[3];     /* color index mode (RGBA mode: unused) */
} MGLCMaterial;

#pragma mark texgen / texenv

typedef struct {
    GLboolean enabled[4];         /* S T R Q */
    GLenum    mode[4];            /* GL_OBJECT_LINEAR .. */
    GLfloat   object_plane[4][4];
    GLfloat   eye_plane[4][4];    /* stored at set time (current modelview) */
} MGLCTexGen;

typedef struct {
    GLenum  env_mode;             /* MODULATE DECAL BLEND REPLACE COMBINE ADD */
    /* GL_TEXTURE_ENV / point sprite crossbar state */
    GLenum  combine_rgb, combine_alpha;
    GLenum  src[6];                 /* rgb sources in [0..2], alpha sources in [3..5] */
    GLenum  op_rgb[3];
    GLenum  op_alpha[3];
    GLfloat scale_rgb, scale_alpha;
    GLfloat env_color[4];
    GLfloat coord_replace;        /* point sprite coord replace flag */
} MGLCTexEnv;

/* src / operand arrays: rgb sources in [0..2], alpha sources in [3..5] */
#define MGLC_ENV_SRC_RGB(i)   (i)
#define MGLC_ENV_SRC_A(i)     (3 + (i))

#pragma mark legacy client arrays

typedef struct {
    GLboolean enabled;
    GLint     size;
    GLenum    type;
    GLboolean normalized;
    GLsizei   stride;
    const void *pointer;          /* client memory */
} MGLCClientArray;

#pragma mark display lists

typedef enum {
    MGLC_OP_NONE = 0,
    MGLC_OP_CALL_LIST,
    MGLC_OP_MATRIX_MODE, MGLC_OP_LOAD_IDENTITY, MGLC_OP_LOAD_MATRIXF, MGLC_OP_MULT_MATRIXF,
    MGLC_OP_TRANSLATEF, MGLC_OP_SCALEF, MGLC_OP_ROTATEF, MGLC_OP_ORTHO, MGLC_OP_FRUSTUM,
    MGLC_OP_PUSH_MATRIX, MGLC_OP_POP_MATRIX,
    MGLC_OP_ENABLE, MGLC_OP_DISABLE,
    MGLC_OP_BEGIN, MGLC_OP_END,
    MGLC_OP_VERTEX4F, MGLC_OP_NORMAL3F, MGLC_OP_COLOR4F, MGLC_OP_SECONDARY3F,
    MGLC_OP_TEXCOORD4F, MGLC_OP_MULTITEXCOORD4F, MGLC_OP_FOGCOORDF, MGLC_OP_EDGEFLAG,
    MGLC_OP_INDEXF, MGLC_OP_ARRAY_ELEMENT,
    MGLC_OP_RECTF, MGLC_OP_EVALCOORD2F, MGLC_OP_EVALCOORD1F, MGLC_OP_EVALMESH,
    MGLC_OP_RASTERPOS4F, MGLC_OP_WINDOWPOS3F,
    MGLC_OP_LIGHTFV, MGLC_OP_LIGHTMODELFV, MGLC_OP_MATERIALFV, MGLC_OP_COLORMATERIAL,
    MGLC_OP_TEXENVFV, MGLC_OP_TEXGENFV, MGLC_OP_FOGFV,
    MGLC_OP_BIND_TEXTURE, MGLC_OP_ACTIVE_TEXTURE, MGLC_OP_TEXIMAGE2D, MGLC_OP_TEXSUBIMAGE2D,
    MGLC_OP_TEXPARAMETERF,
    MGLC_OP_CLIPPING_PLANE, MGLC_OP_SHADE_MODEL, MGLC_OP_LINE_STIPPLE, MGLC_OP_POLYGON_STIPPLE,
    MGLC_OP_POINT_SIZE, MGLC_OP_LINE_WIDTH, MGLC_OP_CULL_FACE, MGLC_OP_FRONT_FACE,
    MGLC_OP_DEPTH_FUNC, MGLC_OP_DEPTH_MASK, MGLC_OP_BLEND_FUNC, MGLC_OP_LOGIC_OP,
    MGLC_OP_STENCIL_FUNC, MGLC_OP_STENCIL_OP, MGLC_OP_STENCIL_MASK,
    MGLC_OP_DEPTH_RANGE, MGLC_OP_VIEWPORT, MGLC_OP_SCISSOR, MGLC_OP_CLEAR_COLOR,
    MGLC_OP_CLEAR, MGLC_OP_CLEAR_DEPTH, MGLC_OP_CLEAR_STENCIL,
    MGLC_OP_COLOR_MASK, MGLC_OP_PIXEL_ZOOM, MGLC_OP_DRAW_PIXELS, MGLC_OP_BITMAP,
    MGLC_OP_PUSH_ATTRIB, MGLC_OP_POP_ATTRIB,
    MGLC_OP_PUSH_NAME, MGLC_OP_POP_NAME, MGLC_OP_LOAD_NAME, MGLC_OP_INIT_NAMES,
    MGLC_OP_BIND_BUFFER, MGLC_OP_VERTEX_POINTER, MGLC_OP_NORMAL_POINTER, MGLC_OP_COLOR_POINTER,
    MGLC_OP_TEXCOORD_POINTER, MGLC_OP_ENABLE_CLIENT_STATE, MGLC_OP_DISABLE_CLIENT_STATE,
    MGLC_OP_CLIENT_ACTIVE_TEXTURE,
    MGLC_OP_LIST_BASE, MGLC_OP_CALL_LISTS,
    MGLC_OP_FLUSH, MGLC_OP_FINISH,
    MGLC_OP_MAX
} MGLCOp;

typedef struct MGLCListOp_t {
    uint8_t  op;
    uint8_t  a, b, c, d;          /* small ints / enums packed as needed */
    int32_t  i0, i1;              /* generic ints */
    float    f[16];               /* generic floats */
    void    *blob;                /* malloc'd copy of pixel data etc. */
    size_t   blob_size;
} MGLCListOp;

typedef struct MGLCList_t {
    GLboolean used;
    MGLCListOp *ops;
    GLuint     nops, cap;
    /* vertex data captured from glBegin..glEnd inside the list */
    float     *verts;             /* see compat_vb.c immediate layout */
    GLuint    nverts, vert_cap;
    uint8_t   prim_mode;
} MGLCList;

#pragma mark attribute stacks

typedef struct MGLCAttribSnapshot_t {
    uint32_t  mask;               /* which groups were pushed */
    void     *deep;               /* MGLCAttribSnapshotData (compat_list.c) */
} MGLCAttribSnapshot;

#pragma mark CPU rasterization (select / feedback)

typedef struct {
    int      mode;                /* GL_RENDER / GL_SELECT / GL_FEEDBACK */
    /* select */
    GLuint  *select_buf;
    GLuint   select_size, select_count;
    GLuint  *select_ptr;          /* writing position */
    GLuint   hit_count;
    GLuint   name_stack[MGLC_NAME_STACK_DEPTH];
    GLuint   name_depth;
    GLboolean hit_open;
    GLuint   hit_zmin_slot, hit_zmax_slot;
    float    zmin, zmax;
    /* feedback */
    GLfloat *feedback_buf;
    GLuint   feedback_size, feedback_count;
    GLenum   feedback_type;
} MGLCRenderMode;

#pragma mark state

struct MGLExtState_compat {
    GLboolean is_compat;          /* set at context creation */

    /* profile-visible version strings */
    char      version[64];
    char      glsl_version[64];

    /* serial bumped by every compat state change (uniform sync cache key) */
    uint64_t  state_serial;

    /* ---- matrix stacks ---- */
    GLenum    matrix_mode;
    MGLCMatrixStack modelview, projection, color, texture[MGLC_MAX_COORDS];
    MGLCMat   modelview_current;  /* cached float copy for shaders */
    GLfloat   modelview_f[16], projection_f[16], texture_f[MGLC_MAX_COORDS][16];

    /* ---- current vertex state (immediate mode) ---- */
    GLfloat   cur_pos[4];
    GLfloat   cur_color[4];
    GLfloat   cur_normal[3];
    GLfloat   cur_secondary[4];
    GLfloat   cur_fogcoord;
    GLfloat   cur_texcoord[MGLC_MAX_COORDS][4];
    GLfloat   cur_index;
    GLboolean cur_edgeflag;

    /* ---- lighting ---- */
    GLboolean lighting;
    GLboolean two_sided;          /* GL_LIGHT_MODEL_TWO_SIDE */
    GLboolean local_viewer;
    GLboolean separate_specular;
    GLfloat   light_model_ambient[4];
    GLboolean color_material;
    GLenum    color_material_face;
    GLenum    color_material_mode;
    MGLCLight lights[MGLC_MAX_LIGHTS];
    MGLCMaterial material_front, material_back;
    GLboolean normalize, rescale_normal;

    /* ---- fog ---- */
    GLboolean fog;
    GLenum    fog_mode;
    GLfloat   fog_density, fog_start, fog_end, fog_index;
    GLfloat   fog_color[4];
    GLenum    fog_coord_src;      /* GL_FRAGMENT_DEPTH / GL_FOG_COORD */
    GLfloat   fog_scale;          /* computed 1/(end-start) */

    /* ---- alpha test ---- */
    GLboolean alpha_test;
    GLenum    alpha_func;
    GLfloat   alpha_ref;

    /* ---- texgen / texenv / texture enables ---- */
    MGLCTexGen texgen[MGLC_MAX_COORDS];
    MGLCTexEnv texenv[MGLC_MAX_COORDS];
    uint32_t unit_target_mask[MGLC_MAX_COORDS];     /* bit0=1D bit1=2D bit2=3D bit3=CUBE */
    GLboolean coord_replace[MGLC_MAX_COORDS]; /* point sprites */

    /* ---- clip planes ---- */
    GLboolean clip_plane_enabled[MGLC_MAX_CLIP_PLANES];
    GLdouble  clip_plane[MGLC_MAX_CLIP_PLANES][4];   /* eye space */

    /* ---- misc fixed function ---- */
    GLenum    shade_model;
    GLenum    polygon_mode_front, polygon_mode_back;
    GLfloat   point_size;
    GLfloat   point_params[8];    /* attenuation: size_min,size_max,fade,a,b,c */
    GLboolean point_attenuation;
    GLuint    line_stipple_pattern;
    GLuint    line_stipple_repeat;
    GLboolean line_stipple;
    GLboolean polygon_stipple;
    GLubyte   polygon_stipple_pattern[32 * 4];
    GLboolean auto_normal;
    GLdouble  raster_pos[4];      /* clip/eye space raster position */
    GLfloat   raster_color[4];
    GLfloat   raster_texcoord[MGLC_MAX_COORDS][4];
    GLfloat   raster_index, raster_distance;
    GLboolean raster_valid;
    GLint     raster_window[2];   /* window space raster position */
    GLfloat   zoom_x, zoom_y;

    /* pixel transfer / maps */
    GLfloat   pixel_transfer[16]; /* indexed by MGLC_PT_* */
    GLfloat  *pixel_map[10];      /* GL_PIXEL_MAP_I_TO_I .. A_TO_A */
    GLuint    pixel_map_size[10];

    /* ---- legacy client arrays ---- */
    GLuint    client_active_texture;
    MGLCClientArray arrays[MAX_ATTRIBS];      /* indexed by MGLC_ATTR_* */
    GLboolean edge_flag_enabled;

    /* ---- display lists ---- */
    GLboolean recording;
    GLboolean replaying;          /* executing a list (no re-recording) */
    GLboolean record_exec;        /* GL_COMPILE_AND_EXECUTE */
    GLuint    record_list;
    GLenum    record_mode;        /* GL_COMPILE / GL_COMPILE_AND_EXECUTE */
    MGLCList *lists;              /* [MGLC_LIST_MAX], allocated lazily in blocks */
    GLuint    list_base;
    GLuint    list_count;         /* for glGenLists allocation */
    GLboolean *list_used;         /* bitmap of used list ids */
    int       call_depth;

    /* ---- attribute stacks ---- */
    MGLCAttribSnapshot attrib_stack[MGLC_ATTRIB_STACK];
    int       attrib_depth;
    int       client_attrib_depth;
    struct MGLCClientAttrib_t {
        uint32_t mask;
        MGLCClientArray arrays[MAX_ATTRIBS];
        GLuint client_active_texture;
        GLboolean edge_flag_enabled;
        Buffer *array_buffer, *element_buffer;
        unsigned enabled_server;  /* client-state enables mirrored */
    } client_attrib_stack[MGLC_CLIENT_STACK];

    /* texture priorities (glPrioritizeTextures) */
    GLclampf  texture_priority[MGLC_TEX_PRIO_MAX];

    /* legacy internal formats (LUMINANCE/ALPHA/INTENSITY) expanded to RGBA on
     * upload; remembered so readback re-packs them */
    struct { GLuint name; GLenum kind; GLenum type; } legacy_tex[MGLC_LEGACY_TEX_MAX];
    int       legacy_tex_count;

    /* ---- select / feedback ---- */
    MGLCRenderMode rmode;

    /* ---- evaluators ---- */
    struct {
        GLboolean enabled;
        GLenum    target;
        GLint     stride, order;
        GLenum    type;
        GLdouble  u1, u2;
        const void *points;       /* client memory captured on Map call */
        GLfloat   *fpoints;       /* converted floats */
    } map1[10];
    struct {
        GLboolean enabled;
        GLenum    target;
        GLint     ustride, vstride, uorder, vorder;
        GLenum    type;
        GLdouble  u1, u2, v1, v2;
        GLfloat   *fpoints;
    } map2[10];
    GLint     map_grid1_n;  GLdouble map_grid1_u1, map_grid1_u2;
    GLint     map_grid2_nu, map_grid2_nv;
    GLdouble  map_grid2_u1, map_grid2_u2, map_grid2_v1, map_grid2_v2;

    /* ---- internal GL objects (created lazily) ---- */
    VertexArray *vao0;            /* VAO 0 backing store */
    VertexArray *imm_vao;         /* internal VAO for glBegin/glEnd */
    GLuint    scratch_vbo[MAX_ATTRIBS];   /* streaming vertex buffers */
    GLuint    scratch_ibo;
    GLuint    ffp_program;        /* generated fixed-function program */
    uint64_t  ffp_key;            /* key of the generated program */
    uint64_t  ffp_sync_serial;
    void     *ffp_cache;          /* MGLFFPCacheEnt[] (compat_ffp.c) */
    GLuint    stipple_tex;        /* 32x32 R8 pattern, unit MGLC_STIPPLE_UNIT */
    uint64_t  stipple_serial;
    GLuint    blit_program;       /* drawpixels / bitmap / copypixels */
    GLuint    blit_texture;      /* DrawPixels/CopyPixels source */
    GLuint    bitmap_texture;    /* glBitmap mask (A = pattern) */
    GLuint    accum_tex;          /* accumulation buffer (RGBA32F) */
    GLboolean accum_valid;
    GLsizei   accum_w, accum_h;
    GLfloat   accum_clear[4];

    /* immediate mode vertex accumulator (compat_vb.c) */
    struct MGLCImm_t *imm;
    float   *pos_scratch;         /* last positions read for CPU paths */

    /* display-list / draw temporaries */
    uint32_t *index_scratch;
    GLuint    index_scratch_cap;

    /* cached per-program built-in uniform locations (translated compat shaders) */
    struct MGLCBuiltinProg_t {
        Program *prog;
        void    *link_id;         /* program->linked_glsl_program (link identity) */
        GLint    loc[96];         /* MGLC_BUILTIN_* (compat_glsl.c) */
        uint64_t sync_serial;
    } builtin_prog[8];
    int       builtin_prog_count;

    /* per-context compat caps not in GLMCaps */
    GLboolean texture_target_enabled[4];   /* GL_TEXTURE_{1D,2D,3D,CUBE_MAP} global */
};

/* Metal hooks of the compat domain (only the profile flag is needed). */
struct MGLExtMetal_compat {
    int unused;                   /* fixed-function runs entirely through core paths */
};

#pragma mark entry points implemented by the domain (internal use)

/* compat.c */
void mgl_ext_init_compat(GLMContext ctx);
bool mgl_ext_get_compat(GLMContext ctx, GLenum pname, GLuint type, void *data);
bool mgl_compat_is_compat(GLMContext ctx);
void mgl_compat_dirty(GLMContext ctx);                 /* bump state serial */

/* compat entry points are live only in compatibility-profile contexts */
#define MGLC_COMPAT_GUARD(ctx) \
    do { \
        if (!ctx->state.ext_compat || !ctx->state.ext_compat->is_compat) { \
            ERROR_RETURN(GL_INVALID_OPERATION); \
        } \
    } while (0)

/* The shared error macros only record the error and fall through; the compat
 * entry points are written as early-exit, so they get local versions. */
#undef ERROR_RETURN
#undef ERROR_RETURN_VALUE
#undef ERROR_CHECK_RETURN
#undef ERROR_CHECK_RETURN_VALUE
#define ERROR_RETURN(_type_) do { ctx->error_func(ctx, __func__, (_type_)); return; } while (0)
#define ERROR_RETURN_VALUE(_type_, _val_) do { ctx->error_func(ctx, __func__, (_type_)); return (_val_); } while (0)
#define ERROR_CHECK_RETURN(_expr_, _type_) if (!(_expr_)) { ctx->error_func(ctx, __func__, (_type_)); return; } while (0)
#define ERROR_CHECK_RETURN_VALUE(_expr_, _type_, _val_) if (!(_expr_)) { ctx->error_func(ctx, __func__, (_type_)); return (_val_); } while (0)
#define MGLC_COMPAT_GUARD_RET0(ctx) do { if (!ctx->state.ext_compat || !ctx->state.ext_compat->is_compat) { ctx->error_func(ctx, __func__, GL_INVALID_OPERATION); return 0; } } while (0)
void mgl_compat_init_dispatch(GLMContext ctx);         /* compat entry points live */
void mgl_compat_init_state(GLMContext ctx);            /* default fixed-function state */
void mgl_compat_destroy(GLMContext ctx);

/* compat_matrix.c */
void mglc_matrix_identity(GLdouble *m);
void mglc_matrix_mul(GLdouble *out, const GLdouble *a, const GLdouble *b); /* out = a*b */
bool mglc_matrix_invert(GLdouble *out, const GLdouble *m);
void mglc_matrix_transpose(GLdouble *out, const GLdouble *m);
void mglc_matrix_frustum(GLdouble *m, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f);
void mglc_matrix_ortho(GLdouble *m, GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f);
void mglc_matrix_rotate(GLdouble *m, GLdouble a, GLdouble x, GLdouble y, GLdouble z);
MGLCMatrixStack *mgl_compat_matrix_stack(GLMContext ctx, GLenum mode, int *texunit);
void mgl_compat_get_matrix(GLMContext ctx, GLenum mode, GLdouble *out);

/* compat_glsl.c (compat GLSL -> core GLSL translation) */
char *mgl_compat_translate_shader(GLMContext ctx, GLuint type, GLsizei count,
                                  const GLchar *const *string, const GLint *length);

/* compat_ffp.c (generated fixed-function pipeline) */
GLuint mgl_compat_ffp_program(GLMContext ctx);        /* bind-on-demand, returns name */
void mgl_compat_ffp_sync(GLMContext ctx, GLuint prog);  /* upload uniforms */
void mgl_compat_sync_builtin_uniforms(GLMContext ctx, Program *prog); /* user compat shaders */
GLint mgl_compat_get_uniform_location(GLMContext ctx, GLuint program, const GLchar *name);
void compat_tex_image1D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type, const void *pixels);
void compat_tex_image2D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
void compat_tex_image3D(GLMContext ctx, GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels);
void compat_get_tex_image(GLMContext ctx, GLenum target, GLint level, GLenum format, GLenum type, void *pixels);
void compat_get_tex_level_parameteriv(GLMContext ctx, GLenum target, GLint level, GLenum pname, GLint *params);
void mgl_compat_window_pos(GLMContext ctx, const float *obj, float *win);  /* obj -> window xyz */
GLuint mgl_compat_blit_program(GLMContext ctx);       /* drawpixels/bitmap helper */

/* compat_vb.c (immediate mode, client arrays, draws) */
void mgl_compat_flush_imm(GLMContext ctx);            /* glEnd */
void mgl_compat_prepare_draw(GLMContext ctx, GLenum mode, GLsizei count, const void *indices,
                             GLenum type, GLint basevertex, GLsizei instances);
void mgl_compat_draw(GLMContext ctx, GLenum mode, GLint first, GLsizei count);
void mgl_compat_draw_elements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type,
                              const void *indices, GLint basevertex);
void mgl_compat_vertex_attrib_pointer(GLMContext ctx, GLuint index, GLint size, GLenum type,
                                      GLboolean normalized, GLsizei stride, const void *pointer);
void mgl_compat_upload_client_arrays(GLMContext ctx, GLuint min_index, GLuint max_index);
const float *mgl_compat_read_positions(GLMContext ctx, GLuint first, GLuint count);
MGLCListOp *mgl_compat_list_op(GLMContext ctx, uint8_t op);
void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0);
void mgl_compat_replay_set_pointer(GLMContext ctx, int index, GLint size, GLenum type,
                                   GLboolean normalized, GLsizei stride, const void *ptr);

/* compat_list.c */
void mgl_compat_list_record_begin(GLMContext ctx, GLenum mode);
void mgl_compat_list_record_end(GLMContext ctx);
bool mgl_compat_list_recording(GLMContext ctx);
void mgl_compat_replay_op(GLMContext ctx, MGLCListOp *op);
void mgl_compat_free_lists(GLMContext ctx);
void mgl_compat_call_list(GLMContext ctx, GLuint list);
void mgl_compat_push_attrib(GLMContext ctx, GLbitfield mask);
void mgl_compat_pop_attrib(GLMContext ctx);
void mgl_compat_list_record_vertex(GLMContext ctx, int n, float x, float y, float z, float w);
void mgl_compat_list_record_color(GLMContext ctx, float r, float g, float b, float a);
void mgl_compat_list_record_secondary(GLMContext ctx, float r, float g, float b);
void mgl_compat_list_record_normal(GLMContext ctx, float x, float y, float z);
void mgl_compat_list_record_index(GLMContext ctx, float i);
void mgl_compat_list_record_fogcoord(GLMContext ctx, float f);
void mgl_compat_list_record_edgeflag(GLMContext ctx, GLboolean flag);
void mgl_compat_list_record_texcoord(GLMContext ctx, int unit, int n, float x, float y, float z, float w);
void mgl_compat_list_record_array_element(GLMContext ctx, GLint i);
void mgl_compat_list_record_rect(GLMContext ctx, float x1, float y1, float x2, float y2);
void mgl_compat_list_record_pointer(GLMContext ctx, int index, GLint size, GLenum type,
                                    GLboolean normalized, GLsizei stride, const void *ptr);
void mgl_compat_list_record_client_state(GLMContext ctx, GLenum array, GLboolean enable);
void mgl_compat_list_record_client_active_texture(GLMContext ctx, GLenum texture);

/* compat_misc.c */
void mgl_compat_cpu_draw(GLMContext ctx, GLenum mode, const float *pos, GLuint n);
GLuint mgl_compat_build_tri_indices(GLMContext ctx, GLenum mode, GLuint n, uint32_t **idx);
void mgl_compat_cpu_draw_indexed(GLMContext ctx, GLenum mode, const float *pos, const uint32_t *idx, GLuint count);

/* compat_pixel.c */
void mgl_compat_raster_pos(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
void mgl_compat_draw_pixels(GLMContext ctx, GLsizei width, GLsizei height, GLenum format, GLenum type, const void *pixels);
void mgl_compat_bitmap(GLMContext ctx, GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig, GLfloat xmove, GLfloat ymove, const GLubyte *bitmap);
void mgl_compat_copy_pixels(GLMContext ctx, GLint x, GLint y, GLsizei w, GLsizei h, GLenum type);

/* compat_misc.c: shared small helpers */
void mgl_compat_set_cap(GLMContext ctx, GLenum cap, GLboolean on);
GLboolean mgl_compat_get_cap(GLMContext ctx, GLenum cap);
bool mgl_compat_valid_cap(GLMContext ctx, GLenum cap);
void mgl_compat_fog_recompute(GLMContext ctx);
void mgl_compat_point_parameterf(GLMContext ctx, GLenum pname, GLfloat param);
void mgl_compat_point_parameterfv(GLMContext ctx, GLenum pname, const GLfloat *params);
void mgl_compat_point_parameteri(GLMContext ctx, GLenum pname, GLint param);
void mgl_compat_point_parameteriv(GLMContext ctx, GLenum pname, const GLint *params);
GLenum mgl_compat_target_index(GLenum target);   /* _TEXTURE_* or -1 */
int mgl_compat_enabled_texture_target(GLMContext ctx, GLuint unit);

#endif /* MGL_EXT_COMPAT_H */
