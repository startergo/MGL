/*
 * geometry.c
 * MGL — geometry shaders: compute-based emulation, streams, layered output
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * Metal has no geometry stage, so a GL program with a geometry shader is run as
 *
 *      1. a rasterization-disabled render pass running the *vertex* stage in a "capture
 *         variant" (the technique of transform_feedback.c: the original GLSL main() is
 *         wrapped and stores every output per (vertex, instance) into an SSBO), followed by
 *      2. a compute pass running a *kernel* that is the geometry shader rewritten by a
 *         source transform: gl_in[] and the user input arrays are loaded from the captured
 *         records of the assembled input primitive, EmitVertex/EmitStreamVertex append the
 *         current outputs to an output vertex buffer with an atomic counter (honouring
 *         max_vertices) and EndPrimitive/EndStreamPrimitive close the strip and append its
 *         raster indices, and finally
 *      3. the ordinary raster pass with a generated pass-through vertex function that reads
 *         the GS output buffer at [[vertex_id]] (gl_Layer -> render_target_array_index,
 *         gl_ViewportIndex -> viewport_array_index) and feeds the program's fragment shader,
 *         drawn with a DrawIndexedPrimitives indirect argument buffer the kernel filled in.
 *
 * The choice of the *source-level* capture variant (rather than SPIRV-Cross's
 * "capture output to buffer" option) is deliberate: Metal's fixed-function vertex fetch
 * then handles every GL attribute format / stride / divisor / base offset, and the captured
 * coordinates stay in GL clip space — the rstate Metal clip-space fixup must only be applied
 * to the last stage before rasterization (here: the pass-through vertex function).
 */

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv_cross_c.h"
#include "spirv.h"

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_geom.h"
#include "mgl_ext_texdata.h"   /* TD_MAX_LEVELS: clamp of the required mip chain */
#include "mgl_ext_buffers.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_tess.h"     /* the TES-output capture producer (geometry behind tessellation) */
#include "mgl_ext_queries.h"  /* mgl_queries_tess_counts_active: TES-invocation query gate (r102) */
#include "mgl_ext_program.h"
#include "mgl_ext_introspect.h"   /* subroutine source rewrite for the capture variant */
#include "fp64.h"             /* software doubles: lower the generated variants like program.c does */
#include "cts_glsl_compat.h"  /* layout(packed)/layout(shared) block rewrites (as shaders.c does) */
#include "spirv_vote.h"       /* SPV_KHR subgroup opcodes -> core OpGroupNonUniform* (as program.c does) */
#include "shaders.h"
#include "programs.h"
#include "msl_reserved.h"
#include "hash_table.h"
#include "mgl_ext_rstate.h"

/* shaders.c helpers (declared locally, as transform_feedback.c does) */
void initGLSLInput(GLMContext ctx, GLuint type, const char *src, glslang_input_t *input);
glslang_stage_t getGLSLStage(GLuint type);

/* ------------------------------------------------------------------ glslang resource limits
 *
 * KHR-GL46.limits compiles a shader that stores gl_Max* into an SSBO and compares it with
 * the value glGet* reports, so the TBuiltInResource handed to glslang must be seeded from
 * the same state the queries serve.  Every pname below is one get.c (or an ext domain
 * ahead of it) really answers; an unanswered query would raise GL_INVALID_ENUM as a side
 * effect of compiling a shader, so nothing may be added to the table that is not served. */
void mgl_geom_fill_glslang_resource(GLMContext ctx, glslang_resource_t *res)
{
    static const struct { GLenum pname; size_t off; } ints[] = {
        { GL_MAX_CLIP_DISTANCES,                         offsetof(glslang_resource_t, max_clip_distances) },
        { GL_MAX_CULL_DISTANCES,                         offsetof(glslang_resource_t, max_cull_distances) },
        { GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES,       offsetof(glslang_resource_t, max_combined_clip_and_cull_distances) },
        { GL_MAX_VIEWPORTS,                              offsetof(glslang_resource_t, max_viewports) },
        { GL_MAX_VERTEX_ATTRIBS,                         offsetof(glslang_resource_t, max_vertex_attribs) },
        { GL_MAX_VERTEX_UNIFORM_COMPONENTS,              offsetof(glslang_resource_t, max_vertex_uniform_components) },
        { GL_MAX_VERTEX_UNIFORM_VECTORS,                 offsetof(glslang_resource_t, max_vertex_uniform_vectors) },
        { GL_MAX_VERTEX_OUTPUT_COMPONENTS,               offsetof(glslang_resource_t, max_vertex_output_components) },
        { GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,             offsetof(glslang_resource_t, max_vertex_texture_image_units) },
        { GL_MAX_TEXTURE_IMAGE_UNITS,                    offsetof(glslang_resource_t, max_texture_image_units) },
        { GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,            offsetof(glslang_resource_t, max_fragment_uniform_components) },
        { GL_MAX_FRAGMENT_UNIFORM_VECTORS,               offsetof(glslang_resource_t, max_fragment_uniform_vectors) },
        { GL_MAX_FRAGMENT_INPUT_COMPONENTS,              offsetof(glslang_resource_t, max_fragment_input_components) },
        { GL_MAX_DRAW_BUFFERS,                           offsetof(glslang_resource_t, max_draw_buffers) },
        { GL_MIN_PROGRAM_TEXEL_OFFSET,                   offsetof(glslang_resource_t, min_program_texel_offset) },
        { GL_MAX_PROGRAM_TEXEL_OFFSET,                   offsetof(glslang_resource_t, max_program_texel_offset) },
        { GL_MAX_VARYING_COMPONENTS,                     offsetof(glslang_resource_t, max_varying_components) },
        { GL_MAX_VARYING_VECTORS,                        offsetof(glslang_resource_t, max_varying_vectors) },
        { GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,           offsetof(glslang_resource_t, max_combined_texture_image_units) },
        { GL_MAX_IMAGE_UNITS,                            offsetof(glslang_resource_t, max_image_units) },
        /* seeded from get.c (4): MS image loads read texture2d_ms, stores/atomics
         * run through the image-atomic sidecar (mgl_spirv_lower_ms_images) */
        { GL_MAX_IMAGE_SAMPLES,                          offsetof(glslang_resource_t, max_image_samples) },
        { GL_MAX_VERTEX_IMAGE_UNIFORMS,                  offsetof(glslang_resource_t, max_vertex_image_uniforms) },
        { GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS,            offsetof(glslang_resource_t, max_tess_control_image_uniforms) },
        { GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS,         offsetof(glslang_resource_t, max_tess_evaluation_image_uniforms) },
        { GL_MAX_GEOMETRY_IMAGE_UNIFORMS,                offsetof(glslang_resource_t, max_geometry_image_uniforms) },
        { GL_MAX_FRAGMENT_IMAGE_UNIFORMS,                offsetof(glslang_resource_t, max_fragment_image_uniforms) },
        { GL_MAX_COMPUTE_IMAGE_UNIFORMS,                 offsetof(glslang_resource_t, max_compute_image_uniforms) },
        { GL_MAX_COMBINED_IMAGE_UNIFORMS,                offsetof(glslang_resource_t, max_combined_image_uniforms) },
        { GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES,       offsetof(glslang_resource_t, max_combined_shader_output_resources) },
        { GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS,           offsetof(glslang_resource_t, max_geometry_texture_image_units) },
        { GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS,       offsetof(glslang_resource_t, max_tess_control_texture_image_units) },
        { GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS,    offsetof(glslang_resource_t, max_tess_evaluation_texture_image_units) },
        { GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,            offsetof(glslang_resource_t, max_compute_texture_image_units) },
        { GL_MAX_GEOMETRY_OUTPUT_VERTICES,               offsetof(glslang_resource_t, max_geometry_output_vertices) },
        { GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS,       offsetof(glslang_resource_t, max_geometry_total_output_components) },
        { GL_MAX_GEOMETRY_UNIFORM_COMPONENTS,            offsetof(glslang_resource_t, max_geometry_uniform_components) },
        { GL_MAX_GEOMETRY_INPUT_COMPONENTS,              offsetof(glslang_resource_t, max_geometry_input_components) },
        { GL_MAX_GEOMETRY_OUTPUT_COMPONENTS,             offsetof(glslang_resource_t, max_geometry_output_components) },
        { GL_MAX_TESS_CONTROL_INPUT_COMPONENTS,          offsetof(glslang_resource_t, max_tess_control_input_components) },
        { GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS,         offsetof(glslang_resource_t, max_tess_control_output_components) },
        { GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS,   offsetof(glslang_resource_t, max_tess_control_total_output_components) },
        { GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS,       offsetof(glslang_resource_t, max_tess_evaluation_input_components) },
        { GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS,      offsetof(glslang_resource_t, max_tess_evaluation_output_components) },
        { GL_MAX_TESS_PATCH_COMPONENTS,                  offsetof(glslang_resource_t, max_tess_patch_components) },
        { GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS,        offsetof(glslang_resource_t, max_tess_control_uniform_components) },
        { GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS,     offsetof(glslang_resource_t, max_tess_evaluation_uniform_components) },
        { GL_MAX_COMPUTE_UNIFORM_COMPONENTS,             offsetof(glslang_resource_t, max_compute_uniform_components) },
        { GL_MAX_PATCH_VERTICES,                         offsetof(glslang_resource_t, max_patch_vertices) },
        { GL_MAX_TESS_GEN_LEVEL,                         offsetof(glslang_resource_t, max_tess_gen_level) },
        { GL_MAX_VERTEX_ATOMIC_COUNTERS,                 offsetof(glslang_resource_t, max_vertex_atomic_counters) },
        { GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS,           offsetof(glslang_resource_t, max_tess_control_atomic_counters) },
        { GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS,        offsetof(glslang_resource_t, max_tess_evaluation_atomic_counters) },
        { GL_MAX_GEOMETRY_ATOMIC_COUNTERS,               offsetof(glslang_resource_t, max_geometry_atomic_counters) },
        { GL_MAX_FRAGMENT_ATOMIC_COUNTERS,               offsetof(glslang_resource_t, max_fragment_atomic_counters) },
        { GL_MAX_COMPUTE_ATOMIC_COUNTERS,                offsetof(glslang_resource_t, max_compute_atomic_counters) },
        { GL_MAX_COMBINED_ATOMIC_COUNTERS,               offsetof(glslang_resource_t, max_combined_atomic_counters) },
        { GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS,          offsetof(glslang_resource_t, max_vertex_atomic_counter_buffers) },
        { GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS,    offsetof(glslang_resource_t, max_tess_control_atomic_counter_buffers) },
        { GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS, offsetof(glslang_resource_t, max_tess_evaluation_atomic_counter_buffers) },
        { GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS,        offsetof(glslang_resource_t, max_geometry_atomic_counter_buffers) },
        { GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS,        offsetof(glslang_resource_t, max_fragment_atomic_counter_buffers) },
        { GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS,         offsetof(glslang_resource_t, max_compute_atomic_counter_buffers) },
        { GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS,        offsetof(glslang_resource_t, max_combined_atomic_counter_buffers) },
        { GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE,             offsetof(glslang_resource_t, max_atomic_counter_buffer_size) },
        { GL_MAX_TRANSFORM_FEEDBACK_BUFFERS,             offsetof(glslang_resource_t, max_transform_feedback_buffers) },
        { GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS,
                                                         offsetof(glslang_resource_t, max_transform_feedback_interleaved_components) },
    };
    static const struct { GLenum pname; size_t off[3]; } indexed[] = {
        { GL_MAX_COMPUTE_WORK_GROUP_COUNT, {
            offsetof(glslang_resource_t, max_compute_work_group_count_x),
            offsetof(glslang_resource_t, max_compute_work_group_count_y),
            offsetof(glslang_resource_t, max_compute_work_group_count_z) } },
        { GL_MAX_COMPUTE_WORK_GROUP_SIZE, {
            offsetof(glslang_resource_t, max_compute_work_group_size_x),
            offsetof(glslang_resource_t, max_compute_work_group_size_y),
            offsetof(glslang_resource_t, max_compute_work_group_size_z) } },
    };

    if (!ctx || !res)
        return;

    for (size_t i = 0; i < sizeof(ints) / sizeof(ints[0]); i++)
    {
        GLint v = *(int *)((char *)res + ints[i].off);
        mglGetIntegerv(ctx, ints[i].pname, &v);
        *(int *)((char *)res + ints[i].off) = v;
    }
    for (size_t i = 0; i < sizeof(indexed) / sizeof(indexed[0]); i++)
        for (int axis = 0; axis < 3; axis++)
        {
            GLint v = *(int *)((char *)res + indexed[i].off[axis]);
            mglGetIntegeri_v(ctx, indexed[i].pname, axis, &v);
            *(int *)((char *)res + indexed[i].off[axis]) = v;
        }

    /* gl_MaxAtomicCounterBindings has no core GL pname; get.c reports
     * GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS = MGL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS (8),
     * which glslang's default (1) would contradict (KHR-GL46.limits requires >= 4). */
    res->max_atomic_counter_bindings = 8;
}

#define GEOM()   ((struct MGLExtState_geom *)ctx->state.ext_geom)
#define METAL()  (ctx->mtl_funcs.ext_geom)

/* ------------------------------------------------------------------ helpers */

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    assert(p);
    return p;
}

static char *xstrdup(const char *s)
{
    char *d = xmalloc(strlen(s) + 1);
    strcpy(d, s);
    return d;
}

/* growable text buffer */
typedef struct {
    char  *s;
    size_t len, cap;
    bool   oom;
} SB;

static void sb_reserve(SB *b, size_t extra)
{
    if (b->oom) return;
    if (b->len + extra + 1 > b->cap)
    {
        size_t c = b->cap ? b->cap * 2 : 4096;
        while (c < b->len + extra + 1) c *= 2;
        char *n = realloc(b->s, c);
        if (!n) { b->oom = true; return; }
        b->s = n;
        b->cap = c;
    }
}

static void sb_put(SB *b, const char *fmt, ...)
{
    va_list ap;
    int n;
    char tmp[512];

    if (b->oom) return;

    va_start(ap, fmt);
    n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) { b->oom = true; return; }
    if ((size_t)n < sizeof(tmp))
    {
        sb_reserve(b, (size_t)n);
        if (b->oom) return;
        memcpy(b->s + b->len, tmp, (size_t)n + 1);
        b->len += (size_t)n;
        b->s[b->len] = 0;
        return;
    }
    {
        char *big = xmalloc((size_t)n + 1);
        va_start(ap, fmt);
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        va_end(ap);
        sb_reserve(b, (size_t)n);
        if (!b->oom)
        {
            memcpy(b->s + b->len, big, (size_t)n + 1);
            b->len += (size_t)n;
            b->s[b->len] = 0;
        }
        free(big);
    }
}

static char *sb_take(SB *b)
{
    if (b->oom)
    {
        free(b->s);
        b->s = NULL;
        b->len = b->cap = 0;
        return NULL;
    }
    if (!b->s) return xstrdup("");
    return b->s;
}

static bool mgl_append_log(char **log, const char *fmt, ...)
{
    va_list ap;
    char buf[1024];

    if (!log) return false;

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    size_t len = *log ? strlen(*log) : 0;
    char *n = realloc(*log, len + strlen(buf) + 2);
    if (!n) return false;
    *log = n;
    memcpy(*log + len, buf, strlen(buf) + 1);
    return true;
}

/* ------------------------------------------------------------------ SPIR-V walker
 * Minimal reader for the module structure the emulation needs (types, interface
 * variables, decorations, execution modes).  Kept local so the transform does not
 * depend on reflection details of any backend. */

enum { SPVT_VOID, SPVT_BOOL, SPVT_INT, SPVT_FLOAT, SPVT_VEC, SPVT_MAT, SPVT_ARRAY, SPVT_STRUCT, SPVT_PTR };

typedef struct {
    uint32_t id;
    uint8_t  kind;
    uint32_t elem;          /* element type id for vec / matrix / array / pointer */
    uint32_t n;             /* vector size / matrix columns / array length */
    uint32_t bits;
    bool     is_signed;
    bool     len_is_spec;
    /* OpTypeStruct: the member type ids, in declaration order (an interface block's members) */
    uint32_t *members;
    int      n_members;
} SpvType;

typedef struct {
    uint32_t id;            /* the struct type */
    uint32_t member;        /* the member index */
    uint32_t deco;
    uint32_t value;
} SpvMemberDeco;

typedef struct {
    uint32_t id;            /* the struct type */
    uint32_t member;        /* member index */
    char     name[64];
} SpvMemberName;

/* one resource-binding pin for a generated MSL compile (see geom_build_pins) */
typedef struct { int binding; int msl; bool texture; } GeomPin;

typedef struct {
    uint32_t id;            /* target */
    uint32_t deco;
    uint32_t value;
} SpvDeco;

typedef struct {
    uint32_t id, type_id;
    uint32_t storage;
    bool     in_interface;
} SpvVar;

typedef struct {
    uint32_t id;
    char     name[160];
} SpvName;

/* per-member metadata of OpTypeStruct (interface blocks) */

typedef struct {
    const uint32_t *w;      /* words */
    size_t          n;
    SpvType        *types;
    int             n_types;
    SpvDeco        *decos;
    int             n_decos;
    SpvVar         *vars;
    int             n_vars;
    SpvName        *names;
    int             n_names;
    SpvMemberDeco  *mdecos;        /* OpMemberDecorate of struct types (interface blocks) */
    int             n_mdecos;
    SpvMemberName  *mnames;        /* OpMemberName */
    int             n_mnames;
    struct { uint32_t mode, value; } *modes;
    int             n_modes;
    struct { uint32_t id; uint64_t v; } *consts;
    int             n_consts;
    uint32_t        entry;
    bool            ok;
} SpvModule;

static void spv_push_type(SpvModule *m, const SpvType *t)
{
    m->types = realloc(m->types, sizeof(SpvType) * (size_t)(m->n_types + 1));
    m->types[m->n_types++] = *t;
}

static uint32_t spv_read_word(const uint32_t *in, size_t n, size_t pos)
{
    (void)n;
    return in[pos];
}

static void spv_get_string(const uint32_t *in, size_t wc, size_t pos, char *out, size_t out_len)
{
    size_t o = 0;
    out[0] = 0;
    for (size_t k = 0; k < wc; k++)
    {
        const char *p = (const char *)&in[pos + k];
        for (int b = 0; b < 4; b++)
        {
            char c = p[b];
            if (!c) { if (o < out_len) out[o] = 0; return; }
            if (o + 1 < out_len) out[o++] = c;
        }
    }
    out[o < out_len ? o : out_len - 1] = 0;
}

static SpvType *spv_find_type(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_types; i++)
        if (m->types[i].id == id) return &m->types[i];
    return NULL;
}

static uint64_t spv_const_value(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_consts; i++)
        if (m->consts[i].id == id) return m->consts[i].v;
    return 1;
}

static bool spv_deco(SpvModule *m, uint32_t id, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == id && m->decos[i].deco == deco)
        {
            if (value) *value = m->decos[i].value;
            return true;
        }
    return false;
}

static const char *spv_name(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_names; i++)
        if (m->names[i].id == id) return m->names[i].name;
    return NULL;
}

static int spv_binding(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == id && m->decos[i].deco == SpvDecorationBinding)
            return (int)m->decos[i].value;
    return -1;
}

static SpvVar *spv_find_var(SpvModule *m, uint32_t id) __attribute__((unused));
static SpvVar *spv_find_var(SpvModule *m, uint32_t id)
{
    for (int i = 0; i < m->n_vars; i++)
        if (m->vars[i].id == id) return &m->vars[i];
    return NULL;
}

static bool spv_member_deco(SpvModule *m, uint32_t type_id, uint32_t member, uint32_t deco, uint32_t *value)
{
    for (int i = 0; i < m->n_mdecos; i++)
        if (m->mdecos[i].id == type_id && m->mdecos[i].member == member && m->mdecos[i].deco == deco)
        {
            if (value) *value = m->mdecos[i].value;
            return true;
        }
    return false;
}

static const char *spv_member_name(SpvModule *m, uint32_t type_id, uint32_t member)
{
    for (int i = 0; i < m->n_mnames; i++)
        if (m->mnames[i].id == type_id && m->mnames[i].member == member)
            return m->mnames[i].name;
    return NULL;
}

/* parse a SPIR-V module; returns NULL and sets *err on failure */
static SpvModule *spv_parse(const uint32_t *words, size_t nwords, const char **err)
{
    SpvModule *m = xmalloc(sizeof(SpvModule));
    memset(m, 0, sizeof(*m));

    if (nwords < 5 || words[0] != SpvMagicNumber)
    {
        *err = "invalid SPIR-V magic";
        free(m);
        return NULL;
    }
    m->w = words;
    m->n = nwords;

    size_t pos = 5;
    while (pos < nwords)
    {
        uint32_t first = spv_read_word(words, nwords, pos);
        uint32_t wc = first >> 16;
        uint32_t op = first & 0xffff;

        if (wc == 0 || pos + wc > nwords)
        {
            *err = "truncated SPIR-V instruction";
            break;
        }

        switch (op)
        {
            case SpvOpName:
            {
                char name[160];
                spv_get_string(words, wc - 2, pos + 2, name, sizeof(name));
                m->names = realloc(m->names, sizeof(SpvName) * (size_t)(m->n_names + 1));
                m->names[m->n_names].id = words[pos + 1];
                snprintf(m->names[m->n_names].name, sizeof(m->names[m->n_names].name), "%s", name);
                m->n_names++;
                break;
            }
            case SpvOpEntryPoint:
                m->entry = words[pos + 2];
                break;
            case SpvOpExecutionMode:
                m->modes = realloc(m->modes, sizeof(*m->modes) * (size_t)(m->n_modes + 1));
                m->modes[m->n_modes].mode = words[pos + 2];
                m->modes[m->n_modes].value = wc > 3 ? words[pos + 3] : 0;
                m->n_modes++;
                break;
            case SpvOpTypeVoid:
            case SpvOpTypeBool:
            case SpvOpTypeInt:
            case SpvOpTypeFloat:
            {
                SpvType t;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                if (op == SpvOpTypeVoid) t.kind = SPVT_VOID;
                else if (op == SpvOpTypeBool) { t.kind = SPVT_BOOL; t.bits = 32; }
                else if (op == SpvOpTypeInt) { t.kind = SPVT_INT; t.bits = words[pos + 2]; t.is_signed = words[pos + 3] != 0; }
                else { t.kind = SPVT_FLOAT; t.bits = words[pos + 2]; }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpTypeVector:
            case SpvOpTypeMatrix:
            case SpvOpTypeArray:
            case SpvOpTypeRuntimeArray:
            case SpvOpTypePointer:
            {
                SpvType t;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                if (op == SpvOpTypePointer)
                {
                    /* OpTypePointer: result, storage class, pointee type */
                    t.kind = SPVT_PTR;
                    t.elem = words[pos + 3];
                    t.n = words[pos + 2];
                }
                else
                {
                    t.elem = words[pos + 2];
                    if (op == SpvOpTypeVector) t.kind = SPVT_VEC;
                    else if (op == SpvOpTypeMatrix) t.kind = SPVT_MAT;
                    else
                    {
                        t.kind = SPVT_ARRAY;
                        if (op == SpvOpTypeRuntimeArray) { t.n = 1; t.len_is_spec = true; }
                        else { t.n = words[pos + 3]; t.len_is_spec = true; }
                    }
                    if (op == SpvOpTypeVector || op == SpvOpTypeMatrix) t.n = words[pos + 3];
                }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpTypeStruct:
            {
                /* the member type ids, in declaration order (an interface block) */
                SpvType t;
                int nm = (int)wc - 2;
                memset(&t, 0, sizeof(t));
                t.id = words[pos + 1];
                t.kind = SPVT_STRUCT;
                t.n_members = nm > 0 ? nm : 0;
                if (t.n_members)
                {
                    t.members = xmalloc(sizeof(uint32_t) * (size_t)t.n_members);
                    for (int k = 0; k < t.n_members; k++)
                        t.members[k] = words[pos + 2 + k];
                }
                spv_push_type(m, &t);
                break;
            }
            case SpvOpMemberName:
            {
                char name[64];
                spv_get_string(words, wc - 3, pos + 3, name, sizeof(name));
                m->mnames = realloc(m->mnames, sizeof(SpvMemberName) * (size_t)(m->n_mnames + 1));
                m->mnames[m->n_mnames].id = words[pos + 1];
                m->mnames[m->n_mnames].member = words[pos + 2];
                snprintf(m->mnames[m->n_mnames].name, sizeof(m->mnames[m->n_mnames].name), "%s", name);
                m->n_mnames++;
                break;
            }
            case SpvOpMemberDecorate:
                m->mdecos = realloc(m->mdecos, sizeof(SpvMemberDeco) * (size_t)(m->n_mdecos + 1));
                m->mdecos[m->n_mdecos].id = words[pos + 1];
                m->mdecos[m->n_mdecos].member = words[pos + 2];
                m->mdecos[m->n_mdecos].deco = words[pos + 3];
                m->mdecos[m->n_mdecos].value = wc > 4 ? words[pos + 4] : 0;
                m->n_mdecos++;
                break;
            case SpvOpConstant:
                m->consts = realloc(m->consts, sizeof(*m->consts) * (size_t)(m->n_consts + 1));
                m->consts[m->n_consts].id = words[pos + 2];
                m->consts[m->n_consts].v = wc > 4 ? ((uint64_t)words[pos + 4] << 32) | words[pos + 3] : words[pos + 3];
                m->n_consts++;
                break;
            case SpvOpDecorate:
                m->decos = realloc(m->decos, sizeof(SpvDeco) * (size_t)(m->n_decos + 1));
                m->decos[m->n_decos].id = words[pos + 1];
                m->decos[m->n_decos].deco = words[pos + 2];
                m->decos[m->n_decos].value = wc > 3 ? words[pos + 3] : 0;
                m->n_decos++;
                break;
            case SpvOpVariable:
            {
                m->vars = realloc(m->vars, sizeof(SpvVar) * (size_t)(m->n_vars + 1));
                m->vars[m->n_vars].id = words[pos + 2];
                m->vars[m->n_vars].type_id = words[pos + 1];
                m->vars[m->n_vars].storage = wc > 3 ? words[pos + 3] : SpvStorageClassFunction;
                m->vars[m->n_vars].in_interface = false;
                m->n_vars++;
                break;
            }
            default:
                break;
        }

        pos += wc;
    }

    /* resolve array lengths given as OpConstant ids */
    for (int i = 0; i < m->n_types; i++)
        if (m->types[i].kind == SPVT_ARRAY && m->types[i].len_is_spec)
        {
            uint64_t v = spv_const_value(m, m->types[i].n);
            if (v && v < 0x10000) m->types[i].n = (uint32_t)v;
            m->types[i].len_is_spec = false;
        }

    m->ok = true;
    return m;
}

static void spv_free(SpvModule *m)
{
    if (!m) return;
    /* every struct type owns its member-id array: free them while m->types is still
     * valid.  free(m->types) first would leave the members pointers read from freed
     * memory (the allocator writes its free-list links over the first types), and
     * freeing one of those garbage pointers corrupts the heap — an intermittent
     * SIGTRAP in malloc that took a while to trace back to here. */
    for (int i = 0; i < m->n_types; i++) free(m->types[i].members);
    free(m->types);
    free(m->decos);
    free(m->vars);
    free(m->names);
    free(m->mdecos);
    free(m->mnames);
    free(m->modes);
    free(m->consts);
    free(m);
}

/* strip pointer / array layers off a variable type */
static SpvType *spv_value_type(SpvModule *m, uint32_t type_id, uint32_t *array_elems)
{
    SpvType *t = spv_find_type(m, type_id);
    if (array_elems) *array_elems = 1;
    if (!t) return NULL;
    while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
    if (!t) return NULL;
    if (t->kind == SPVT_ARRAY)
    {
        uint32_t elems = t->n ? t->n : 1;
        t = spv_find_type(m, t->elem);
        while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
        if (!t) return NULL;
        if (array_elems) *array_elems = elems;
    }
    return t;
}

/* the scalar component type of a (possibly vector / matrix) type: a plain SPVT_VEC's kind
 * says nothing about float vs int — the component type is its element type.  Returning
 * GL_FLOAT for an ivec4 would make the kernel load it with uintBitsToFloat and the
 * generated assignment fail to compile ("cannot convert vec4 to ivec4"). */
static GLenum spv_scalar_base(SpvModule *m, const SpvType *t)
{
    const SpvType *s = t;
    if (!s) return GL_FLOAT;
    if (s->kind == SPVT_VEC || s->kind == SPVT_MAT) s = spv_find_type(m, s->elem);
    if (s && s->kind == SPVT_VEC) s = spv_find_type(m, s->elem);   /* dmat: element is a dvec */
    if (!s) return GL_FLOAT;
    if (s->kind == SPVT_FLOAT) return s->bits == 64 ? GL_DOUBLE : GL_FLOAT;
    if (s->kind == SPVT_INT) return s->is_signed ? GL_INT : GL_UNSIGNED_INT;
    if (s->kind == SPVT_BOOL) return GL_BOOL;
    return GL_FLOAT;
}

/* ------------------------------------------------------------------ varying layout */

/* ---------------------------------------------------- storage block offset repair */

/* ArrayStride is an OpDecorate on the array type id */
static bool spv_deco_stride(SpvModule *m, uint32_t tid, uint32_t *stride)
{
    for (int i = 0; i < m->n_decos; i++)
        if (m->decos[i].id == tid && m->decos[i].deco == SpvDecorationArrayStride)
        {
            *stride = m->decos[i].value;
            return true;
        }
    return false;
}

/* the byte span a block member occupies, using only the decorations glslang itself
 * emitted: for an array member the declared constant length times the declared
 * ArrayStride.  Every other member returns 0 ("span unknown, leave the offsets
 * alone") so the repair can never second-guess a layout glslang computed right. */
static uint32_t spv_member_span(SpvModule *m, uint32_t member_type)
{
    SpvType *t = spv_find_type(m, member_type);
    if (!t || t->kind != SPVT_ARRAY || t->len_is_spec || t->n == 0)
        return 0;                                   /* runtime array / not an array */
    uint32_t stride = 0;
    if (!spv_deco_stride(m, member_type, &stride) || !stride)
        return 0;
    return t->n * stride;
}

/* glslang sizes an unsized non-final buffer-block member from the offset of the
 * member that follows it, but then lets a constant index into that member grow the
 * array implicitly ("updateImplicitArraySize"): the member type becomes array[2]
 * while the later members keep the offsets computed for array[1].  The block then
 * self-overlaps and the members after it read the wrong bytes
 * (KHR-GL46.shader_storage_buffer_object basic-syntax / basic-syntaxSSO variant 8:
 * the trailing runtime array landed at offset 16 instead of 32).  Push every member
 * that starts inside the previous member's declared array span past that span, and
 * shift the members after it along.  Returns a malloc'd module, or NULL when nothing
 * changed. */
uint32_t *mgl_spirv_fix_block_offsets(const uint32_t *in, size_t n, size_t *out_n)
{
    const char *err = NULL;
    SpvModule *m = spv_parse(in, n, &err);
    if (!m || !m->ok) { spv_free(m); return NULL; }

    uint32_t *mod = NULL;
    bool changed = false;

    /* struct types that are interface blocks (Block / BufferBlock decoration) */
    for (int ti = 0; ti < m->n_types; ti++)
    {
        SpvType *t = &m->types[ti];
        if (t->kind != SPVT_STRUCT) continue;

        bool block = false, buffer_block = false;
        for (int d = 0; d < m->n_decos; d++)
            if (m->decos[d].id == t->id)
            {
                if (m->decos[d].deco == SpvDecorationBlock) block = true;
                else if (m->decos[d].deco == SpvDecorationBufferBlock) buffer_block = true;
            }
        if (block == buffer_block) continue;      /* not an interface block */

        /* running "first byte the next member may start at" */
        uint32_t limit = 0;
        for (int mem = 0; mem < t->n_members; mem++)
        {
            uint32_t have = 0;
            if (!spv_member_deco(m, t->id, (uint32_t)mem, SpvDecorationOffset, &have))
            {
                limit = 0;                          /* packed block: no offsets to repair */
                break;
            }

            uint32_t want = have > limit ? have : limit;
            uint32_t span = spv_member_span(m, t->members[mem]);
            if (want + span > limit) limit = want + span;

            if (getenv("MGL_BLOCK_DEBUG"))
                fprintf(stderr, "MGL block: type %u %s member %d have=%u want=%u span=%u\n",
                        t->id, buffer_block ? "ssbo" : "ubo", mem, have, want, span);
            if (want == have) continue;

            if (!mod)
            {
                mod = xmalloc(n * sizeof(uint32_t));
                memcpy(mod, in, n * sizeof(uint32_t));
            }
            /* patch the OpMemberDecorate Offset word of (type, member): the words are
             * [0]=opcode/len [1]=type [2]=member [3]=decoration [4]=literal.
             * Walk INSTRUCTION-ALIGNED from the header: a byte-granular scan can
             * match string/debug words that merely look like a patch target and
             * then scribble `want` over a real instruction's opcode/length word,
             * corrupting the module ("SPIR-V instruction goes out of bounds",
             * KHR-GL46.texture_gather.offset-gather-float-2drect flake). */
            for (size_t o = 5; o < n; )
            {
                uint32_t op = mod[o];
                uint32_t len = op >> 16;
                if (len == 0 || o + len > n) break;   /* malformed tail: nothing more to patch */
                if ((op & 0xffff) == SpvOpMemberDecorate && len >= 5 &&
                    mod[o + 1] == t->id && mod[o + 2] == (uint32_t)mem &&
                    mod[o + 3] == SpvDecorationOffset && mod[o + 4] != want)
                {
                    mod[o + 4] = want;
                    changed = true;
                    break;   /* one Offset decorate per member */
                }
                o += len;
            }
        }
    }

    spv_free(m);
    if (!changed) { free(mod); return NULL; }
    *out_n = n;
    return mod;
}

/* ---- arrays-of-matrices stage inputs: lower to per-element attributes ----
 * SPIRV-Cross refuses to emit an array-of-matrices stage input ("MSL cannot emit
 * arrays-of-matrices in input and output variables"), which used to fail the
 * whole link for GLSL-legal attributes like `layout(location=7) in mat2x3 d[2]`
 * (KHR-GL46.program_interface_query.input-types / input-layout).  This runs at
 * the MSL boundary only — the GL view of the module keeps the array, so
 * introspection still reports one input named "d" with GL_ARRAY_SIZE 2.
 *
 * The variable becomes N element-typed Input variables at consecutive locations
 * (each matrix consumes one location per column) plus a Private shadow array of
 * the original type whose entry-prologue copy makes every reference — including
 * dynamic indexing and helper functions — work unchanged. */
typedef struct { uint32_t *w; size_t n, cap; } AomBuf;

static void aom_push(AomBuf *b, uint32_t v)
{
    if (b->n == b->cap) { b->cap = b->cap ? b->cap * 2 : 2048; b->w = realloc(b->w, b->cap * sizeof(uint32_t)); }
    b->w[b->n++] = v;
}

static void aom_inst(AomBuf *b, uint32_t op, int wc, const uint32_t *ops)
{
    aom_push(b, ((uint32_t)wc << 16) | op);
    for (int i = 0; i < wc - 1; i++) aom_push(b, ops[i]);
}

uint32_t *mgl_spirv_lower_aom_input(const uint32_t *in, size_t n, size_t *out_n)
{
    const char *err = NULL;
    SpvModule *m = spv_parse(in, n, &err);
    if (!m || !m->ok) { spv_free(m); return NULL; }

    /* Input variables of array-of-matrix type with an explicit Location */
    struct { uint32_t var, arr_t, elem_t, loc, count, locs; } v[16];
    int nv = 0;
    for (int i = 0; i < m->n_vars && nv < 16; i++) {
        SpvVar *gv = &m->vars[i];
        if (gv->storage != SpvStorageClassInput) continue;
        SpvType *pt = spv_find_type(m, gv->type_id);
        if (!pt || pt->kind != SPVT_PTR) continue;
        SpvType *arr = spv_find_type(m, pt->elem);
        if (!arr || arr->kind != SPVT_ARRAY) continue;
        SpvType *mat = spv_find_type(m, arr->elem);
        if (!mat || mat->kind != SPVT_MAT) continue;
        uint32_t loc = 0;
        if (!spv_deco(m, gv->id, SpvDecorationLocation, &loc)) continue;
        /* the parser resolves OpTypeArray lengths to their constant value */
        v[nv].var = gv->id;
        v[nv].arr_t = pt->elem;
        v[nv].elem_t = arr->elem;
        v[nv].loc = loc;
        v[nv].count = arr->n;
        v[nv].locs = mat->n ? mat->n : 1;   /* one location per matrix column */
        if (!v[nv].count || v[nv].count > 64) continue;
        nv++;
    }
    if (!nv) { spv_free(m); return NULL; }

    /* integer result type for the per-element index constants: the type of any
     * existing OpConstant (all lowered arrays need one for their length anyway),
     * else a fresh 32-bit unsigned int */
    uint32_t index_type = 0;
    for (size_t o = 5; o + 4 <= n; ) {
        uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;
        if (op == SpvOpConstant) { index_type = in[o + 1]; break; }
        o += wc;
    }

    AomBuf out; memset(&out, 0, sizeof out);
    for (int i = 0; i < 5; i++) aom_push(&out, in[i]);

    uint32_t next_id = in[3] > 5 ? in[3] : 5;    /* module bound (header word 3) */
    struct { uint32_t shadow, priv_arr_ptr, priv_elem_ptr, in_elem_ptr;
             uint32_t *elems, *consts, *acs, *ld; } g[16];
    memset(g, 0, sizeof g);
    if (!index_type) index_type = next_id++;
    for (int k = 0; k < nv; k++) {
        g[k].shadow = next_id++;
        g[k].priv_arr_ptr = next_id++;
        g[k].priv_elem_ptr = next_id++;
        g[k].in_elem_ptr = next_id++;
        g[k].elems = malloc(v[k].count * sizeof(uint32_t));
        g[k].consts = malloc(v[k].count * sizeof(uint32_t));
        g[k].acs = malloc(v[k].count * sizeof(uint32_t));
        g[k].ld = malloc(v[k].count * sizeof(uint32_t));
        for (uint32_t e = 0; e < v[k].count; e++) {
            g[k].elems[e] = next_id++;
            g[k].consts[e] = next_id++;
            g[k].acs[e] = next_id++;
            g[k].ld[e] = next_id++;
        }
    }

    bool globals_done = false, in_functions = false, entry_seen = false;
    bool prologue_pending = false, prologue_done = true;
    for (size_t o = 5; o < n; ) {
        uint32_t op = in[o] & 0xffff, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;
        const uint32_t *w = in + o;

        if (!globals_done && op == SpvOpFunction) {
            if (index_type >= in[3]) {          /* the fresh uint type we allocated */
                uint32_t t[3] = { index_type, 32, 0 };
                aom_inst(&out, SpvOpTypeInt, 4, t);
            }
            for (int k = 0; k < nv; k++) {
                uint32_t p1[3] = { g[k].priv_arr_ptr, SpvStorageClassPrivate, v[k].arr_t };
                aom_inst(&out, SpvOpTypePointer, 4, p1);
                uint32_t p2[3] = { g[k].priv_elem_ptr, SpvStorageClassPrivate, v[k].elem_t };
                aom_inst(&out, SpvOpTypePointer, 4, p2);
                uint32_t p3[3] = { g[k].in_elem_ptr, SpvStorageClassInput, v[k].elem_t };
                aom_inst(&out, SpvOpTypePointer, 4, p3);
                for (uint32_t e = 0; e < v[k].count; e++) {
                    uint32_t cv[3] = { index_type, g[k].consts[e], e };
                    aom_inst(&out, SpvOpConstant, 4, cv);
                    uint32_t vv[3] = { g[k].in_elem_ptr, g[k].elems[e], SpvStorageClassInput };
                    aom_inst(&out, SpvOpVariable, 4, vv);
                    uint32_t dv[3] = { g[k].elems[e], SpvDecorationLocation, v[k].loc + e * v[k].locs };
                    aom_inst(&out, SpvOpDecorate, 4, dv);
                }
                uint32_t sv[3] = { g[k].priv_arr_ptr, g[k].shadow, SpvStorageClassPrivate };
                aom_inst(&out, SpvOpVariable, 4, sv);
            }
            globals_done = true;
        }
        if (op == SpvOpFunction) {
            in_functions = true;
            if (wc >= 4 && w[2] == m->entry) { entry_seen = true; prologue_done = false; prologue_pending = false; }
        }
        if (op == SpvOpLabel && entry_seen && !prologue_done) prologue_pending = true;

        if (op == SpvOpVariable) {
            bool drop = false;
            for (int k = 0; k < nv; k++) if (w[2] == v[k].var) drop = true;
            if (drop) { o += wc; continue; }
        }
        if ((op == SpvOpName || op == SpvOpDecorate) && wc >= 2) {
            bool drop = false;
            for (int k = 0; k < nv; k++) if (w[1] == v[k].var) drop = true;
            if (drop) { o += wc; continue; }
        }
        if (op == SpvOpEntryPoint && wc >= 3) {
            bool hit = false;
            for (size_t q = 3; q < (size_t)wc; q++)
                for (int k = 0; k < nv; k++) if (w[q] == v[k].var) hit = true;
            if (hit) {
                uint32_t nw[640]; int nn = 0;
                nw[nn++] = w[1]; nw[nn++] = w[2];
                for (size_t q = 3; q < (size_t)wc; q++) {
                    bool sub = false;
                    for (int k = 0; k < nv; k++) if (w[q] == v[k].var) {
                        for (uint32_t e = 0; e < v[k].count && nn < 600; e++) nw[nn++] = g[k].elems[e];
                        sub = true;
                    }
                    if (!sub && nn < 630) nw[nn++] = w[q];
                }
                aom_inst(&out, op, nn + 1, nw);
                o += wc;
                continue;
            }
        }
        /* prologue: first instruction of the entry block that is not a variable/phi */
        if (prologue_pending && !prologue_done && op != SpvOpVariable && op != SpvOpPhi && op != SpvOpLabel) {
            for (int k = 0; k < nv; k++)
                for (uint32_t e = 0; e < v[k].count; e++) {
                    uint32_t ac[4] = { g[k].priv_elem_ptr, g[k].acs[e], g[k].shadow, g[k].consts[e] };
                    aom_inst(&out, SpvOpAccessChain, 5, ac);
                    uint32_t ld[3] = { v[k].elem_t, g[k].ld[e], g[k].elems[e] };
                    aom_inst(&out, SpvOpLoad, 4, ld);
                    uint32_t st[2] = { g[k].acs[e], g[k].ld[e] };
                    aom_inst(&out, SpvOpStore, 3, st);
                }
            prologue_done = true;
        }
        /* copy the instruction, re-pointing references at the shadow arrays */
        aom_push(&out, w[0]);
        if (in_functions)
            for (size_t q = 1; q < (size_t)wc; q++) {
                uint32_t x = w[q];
                for (int k = 0; k < nv; k++) if (x == v[k].var) { x = g[k].shadow; break; }
                aom_push(&out, x);
            }
        else
            for (size_t q = 1; q < (size_t)wc; q++) aom_push(&out, w[q]);
        o += wc;
    }
    spv_free(m);

    for (int k = 0; k < nv; k++) { free(g[k].elems); free(g[k].consts); free(g[k].acs); free(g[k].ld); }
    if (out.n <= 5 || !globals_done || !prologue_done) { free(out.w); return NULL; }
    out.w[3] = next_id;   /* new bound */
    *out_n = out.n;
    return out.w;
}

/* ---- multisample images: loads/stores/atomics through the atomic sidecar ----
 * Metal's texture2d_ms / texture2d_ms_array are READ-ONLY, so a GL image2DMS /
 * image2DMSArray load, store or atomic cannot go through the texture argument.
 * SPIRV-Cross already emulates image ATOMICS with a side `device atomic_*`
 * buffer per image (the renderer binds one per image unit —
 * geomImageAtomicBuffer), but for a multisample image it builds the sidecar
 * index from the 2D coordinate alone, dropping the sample (and the array layer)
 * of OpImageTexelPointer's separate Sample operand — and plain stores have no
 * sidecar at all (SPIRV-Cross emits `tex.write(...)` on a texture2d_ms, which
 * is not legal MSL; mgl_msl_fix_ms_image_writes drops them).
 *
 * This pass reroutes EVERY access of a multisample storage image with a
 * 32-bit-per-component format (r32f / r32i / r32ui, rg32*, rgba32* — the only
 * layouts whose texels fit one sidecar word per component) through the sidecar:
 *
 *   OpImageTexelPointer  the coordinate becomes int4(x, y, layer, sample*4)
 *                        so the sidecar word index the MSL pass computes
 *                        (mgl_msl_fix_image_atomics → spvMglAIMS) covers the
 *                        layer and sample dimensions
 *   OpImageWrite         lowered to one OpAtomicStore per component through a
 *                        folded-coord OpImageTexelPointer
 *   OpImageRead          of an image THIS module also writes or atomics:
 *                        lowered to per-component OpAtomicLoad + composite, so
 *                        the read observes the program's own stores (GL 8.26:
 *                        accesses within one invocation happen in order)
 *
 * Reads of multisample images the module does not write stay as texture2d_ms
 * reads (zero cost, exact).  Writes to other formats (rgba8, rgba16f, …) keep
 * the old drop path; texture_query reports GL_SHADER_IMAGE_STORE = NONE for
 * those on the multisample targets, so the capability is honest.
 *
 * The renderer seeds the sidecar from the texture (a texture2d_ms read is
 * fine) before the first dispatch/draw that uses it, and rasterizes it back at
 * the glMemoryBarrier that flushes image atomics: one 1-pixel point per texel
 * whose fragment outputs the sample's word with [[sample_mask]] = 1<<sample.
 * Metal guarantees primitive order within a render pass, so write ordering
 * between barriers is preserved. */
typedef struct { uint32_t *w; size_t n, cap; } MsBuf;

static void ms_push(MsBuf *b, uint32_t v)
{
    if (b->n == b->cap) { b->cap = b->cap ? b->cap * 2 : 2048; b->w = realloc(b->w, b->cap * sizeof(uint32_t)); }
    b->w[b->n++] = v;
}

static void ms_inst(MsBuf *b, uint32_t op, uint32_t wc, const uint32_t *ops)
{
    ms_push(b, (wc << 16) | op);
    for (uint32_t i = 0; i < wc - 1; i++) ms_push(b, ops[i]);
}

/* sidecar words per (texel, layer, sample): every component gets a word so one
 * layout serves r32*, rg32* and rgba32* images (mgl_msl_fix_image_atomics and
 * geomImageAtomicBuffer must agree) */
#define MGL_MS_WORDS 4

static int ms_format_components(uint32_t fmt)
{
    switch (fmt)
    {
        case SpvImageFormatRgba32f: case SpvImageFormatRgba32i: case SpvImageFormatRgba32ui:
            return 4;
        case SpvImageFormatRg32f: case SpvImageFormatRg32i: case SpvImageFormatRg32ui:
            return 2;
        case SpvImageFormatR32f: case SpvImageFormatR32i: case SpvImageFormatR32ui:
            return 1;
        default:
            return 0;               /* not a 32-bit-per-component layout: keep the drop path */
    }
}

uint32_t *mgl_spirv_lower_ms_images(const uint32_t *in, size_t n, size_t *out_n)
{
    if (!in || n < 5 || in[0] != SpvMagicNumber || !out_n) return NULL;
    uint32_t bound = in[3];
    if (getenv("MGL_MS_DEBUG"))
        fprintf(stderr, "MGL ms-pass: called, %zu words, bound %u\n", n, bound);

    /* per-id definition map: opcode (0 = unused id) + two operand words
     * (OpTypeImage: base type | packed dims<<0 | format<<16; OpTypePointer:
     *  storage, pointee; OpTypeVector: component, count; OpTypeInt: width,
     *  signedness; OpConstant: type, value; OpLoad/AccessChain: base pointer;
     *  OpVariable/OpFunctionParameter: pointer type) */
    uint32_t *defop = calloc(bound, sizeof(uint32_t));
    uint32_t *defa  = calloc(bound, sizeof(uint32_t));
    uint32_t *defb  = calloc(bound, sizeof(uint32_t));
    if (!defop || !defa || !defb) { free(defop); free(defa); free(defb); return NULL; }

    size_t insert_pos = 0;
    bool in_functions = false;

    /* sweep 1: definitions */
    for (size_t o = 5; o < n; )
    {
        uint32_t op = in[o] & 0xFFFFu, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) { free(defop); free(defa); free(defb); return NULL; }

        if (!in_functions)
        {
            if ((op >= SpvOpTypeVoid && op <= SpvOpTypeForwardPointer) ||
                op == SpvOpConstant || op == SpvOpConstantTrue || op == SpvOpConstantFalse ||
                op == SpvOpConstantComposite || op == SpvOpConstantSampler ||
                op == SpvOpConstantNull || op == SpvOpUndef ||
                (op >= SpvOpSpecConstantTrue && op <= SpvOpSpecConstantOp))
                insert_pos = o + wc;
            else if (op == SpvOpFunction)
                in_functions = true;
        }

        switch (op)
        {
            case SpvOpTypeImage:
                /* [1]=id [2]=sampled [3]=dim [4]=depth [5]=arrayed [6]=ms [7]=sampled [8]=format */
                if (wc >= 9)
                    defop[in[o + 1]] = op,
                    defa[in[o + 1]] = in[o + 2],
                    defb[in[o + 1]] = (in[o + 5] ? 2u : 0u) | (in[o + 6] ? 4u : 0u) |
                                      (in[o + 8] << 16);
                break;
            case SpvOpTypePointer: /* [1]=id [2]=storage [3]=pointee */
                defop[in[o + 1]] = op; defa[in[o + 1]] = in[o + 2]; defb[in[o + 1]] = in[o + 3];
                break;
            case SpvOpTypeVector:  /* [1]=id [2]=component [3]=count */
                defop[in[o + 1]] = op; defa[in[o + 1]] = in[o + 2]; defb[in[o + 1]] = in[o + 3];
                break;
            case SpvOpTypeArray:   /* [1]=id [2]=element [3]=length — image2DMS g[N] */
                defop[in[o + 1]] = op; defa[in[o + 1]] = in[o + 2]; defb[in[o + 1]] = in[o + 3];
                break;
            case SpvOpTypeInt:     /* [1]=id [2]=width [3]=signedness */
                defop[in[o + 1]] = op; defa[in[o + 1]] = in[o + 2]; defb[in[o + 1]] = in[o + 3];
                break;
            case SpvOpTypeFloat:   /* [1]=id [2]=width */
                defop[in[o + 1]] = op; defa[in[o + 1]] = in[o + 2];
                break;
            case SpvOpTypeBool:
                defop[in[o + 1]] = op;
                break;
            case SpvOpConstant:
                defop[in[o + 2]] = op; defa[in[o + 2]] = in[o + 1]; defb[in[o + 2]] = in[o + 3];
                break;
            case SpvOpLoad:        /* [1]=type [2]=id [3]=pointer */
                defop[in[o + 2]] = op; defa[in[o + 2]] = in[o + 3];
                break;
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
                defop[in[o + 2]] = op; defa[in[o + 2]] = in[o + 3];
                break;
            case SpvOpVariable:    /* [1]=ptr type [2]=id [3]=storage */
                defop[in[o + 2]] = op; defa[in[o + 2]] = in[o + 1];
                break;
            case SpvOpFunctionParameter:  /* [1]=type [2]=id */
                defop[in[o + 2]] = op; defa[in[o + 2]] = in[o + 1];
                break;
            default:
                break;
        }
        o += wc;
    }

    /* sweep 2: classify the image accesses */
    uint32_t ms_vars[64]; int nms = 0;              /* variables written / atomiced */
    struct { size_t off; int kind; uint32_t imgt, image, coord, texel, sample, restype, resid;
             bool arrayed; int ncomp; uint32_t var; uint32_t base_ptr; } todo[512];
    int ntodo = 0;

    for (size_t o = 5; o < n; )
    {
        uint32_t op = in[o] & 0xFFFFu, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;

        if (op == SpvOpImageWrite || op == SpvOpImageRead || op == SpvOpImageTexelPointer)
        {
            if (getenv("MGL_MS_DEBUG"))
                fprintf(stderr, "MGL ms-pass: op %u at +%zu wc=%u ok-scan\n", op, o, wc);
            uint32_t image = 0, coord = 0, texel = 0, sample = 0, restype = 0, resid = 0;
            bool ok = false;
            if (op == SpvOpImageWrite)              /* [1]=image [2]=coord [3]=texel [4..]=ops */
            {
                image = in[o + 1]; coord = in[o + 2]; texel = in[o + 3];
                if (wc == 6 && in[o + 4] == SpvImageOperandsSampleMask) { sample = in[o + 5]; ok = true; }
            }
            else                                    /* [1]=type [2]=id [3]=image [4]=coord … */
            {
                restype = in[o + 1]; resid = in[o + 2]; image = in[o + 3]; coord = in[o + 4];
                if (op == SpvOpImageTexelPointer)   /* [5]=sample */
                { sample = in[o + 5]; ok = true; }
                else if (wc == 7 && in[o + 5] == SpvImageOperandsSampleMask)
                { sample = in[o + 6]; ok = true; }
            }
            if (ok && ntodo < 512)
            {
                /* the variable the image value came from (OpLoad of the uniform) */
                uint32_t var = image, hops;
                for (hops = 0; var && hops < 8; hops++)
                {
                    uint32_t dop = defop[var];
                    if (dop == SpvOpLoad || dop == SpvOpAccessChain || dop == SpvOpInBoundsAccessChain)
                        var = defa[var];
                    else if (dop == SpvOpVariable || dop == SpvOpFunctionParameter)
                        break;
                    else { var = 0; break; }
                }
                uint32_t imgt = 0;
                if (var && defop[var] == SpvOpVariable && defop[defa[var]] == SpvOpTypePointer)
                {
                    uint32_t pt = defa[var];
                    if (defop[defb[pt]] == SpvOpTypeImage) imgt = defb[pt];
                    /* an ARRAY of images (image2DMS g[N]): the variable's pointee is an
                     * OpTypeArray whose ELEMENT is the OpTypeImage itself (glslang:
                     * %arr = OpTypeArray %imgt %len; the access chain yields a pointer
                     * to ONE image) — resolve through it, so array elements lower
                     * exactly like the scalar uniform.  Without this the write keeps
                     * its texture2d_ms access::write and Metal rejects the whole stage
                     * at compile. */
                    else if (defop[defb[pt]] == SpvOpTypeArray &&
                             defop[defa[defb[pt]]] == SpvOpTypeImage)
                        imgt = defa[defb[pt]];
                }
                /* the pointer the image VALUE was loaded from: the uniform variable for
                 * a scalar image, the OpAccessChain ELEMENT for an array — the rebuilt
                 * OpImageTexelPointer / OpLoad must name it (SPIRV-Cross builds the
                 * `<name>[i]_atomic[...]` subscript from that chain) */
                uint32_t base_ptr = var;
                if (defop[image] == SpvOpLoad && defa[image]) base_ptr = defa[image];
                if (getenv("MGL_MS_DEBUG"))
                    fprintf(stderr, "MGL ms-pass: op %u image %u -> var %u (defop %u), imgt %u (defop %u flags 0x%x)\n",
                            op, image, var, var ? defop[var] : 0, imgt, imgt ? defop[imgt] : 0,
                            imgt ? defb[imgt] : 0);
                if (getenv("MGL_MS_DEBUG"))
                    fprintf(stderr, "MGL ms-pass: classify-check defa[imgt]=%u defop[sampled]=%u\n",
                            imgt ? defa[imgt] : 0, imgt ? defop[defa[imgt]] : 0);
                if (imgt && (defb[imgt] & 4u /* MS */) && defop[defa[imgt]] != 0)
                {
                    if (getenv("MGL_MS_DEBUG") && !nms)
                        fprintf(stderr, "MGL ms-pass: classified op %u var %u imgt %u flags 0x%x fmt %u\n",
                                op, var, imgt, defb[imgt], defb[imgt] >> 16);
                    if (op != SpvOpImageRead)        /* a write or an atomic touches it */
                    {
                        bool known = false;
                        for (int k = 0; k < nms; k++) if (ms_vars[k] == var) known = true;
                        if (!known && nms < 64) ms_vars[nms++] = var;
                    }
                    todo[ntodo].off = o;
                    todo[ntodo].kind = (op == SpvOpImageWrite) ? 1 : (op == SpvOpImageRead ? 2 : 3);
                    todo[ntodo].imgt = imgt; todo[ntodo].image = image; todo[ntodo].coord = coord;
                    todo[ntodo].texel = texel; todo[ntodo].sample = sample;
                    todo[ntodo].restype = restype; todo[ntodo].resid = resid;
                    todo[ntodo].var = var;
                    todo[ntodo].base_ptr = base_ptr;
                    todo[ntodo].arrayed = (defb[imgt] & 2u) != 0;
                    /* GLSL images WITHOUT a format qualifier (legal with writeonly —
                     * KHR-GL46.multi_bind's fillMSTexture) come out of glslang with
                     * ImageFormat Unknown; the binding's format rules at run time.
                     * The sidecar layout is a 32-bit word per component regardless
                     * (4*(sample + S*texel) + channel, seed/flush/fold agree), so when
                     * the format is Unknown derive the width from the image's SAMPLED
                     * TYPE: 32-bit base (uint/int/float) -> 4 word components; anything
                     * else keeps the drop path.  Without this the whole WRITE was kept
                     * on texture2d_ms, mgl_msl_fix_ms_image_writes dropped it, and the
                     * store silently vanished (dispatch_bind_image_textures). */
                    {
                        int nc = ms_format_components(defb[imgt] >> 16);
                        if (nc == 0)
                        {
                            uint32_t st = defa[imgt];
                            uint32_t base = (defop[st] == SpvOpTypeVector) ? defa[st] : st;
                            uint32_t width = (defop[base] == SpvOpTypeInt ||
                                              defop[base] == SpvOpTypeFloat) ? defa[base] : 0;
                            if (width == 32) nc = 4;
                        }
                        todo[ntodo].ncomp = nc;
                    }
                    ntodo++;
                }
            }
        }
        o += wc;
    }

    if (!nms) { free(defop); free(defa); free(defb); return NULL; }   /* reads only */
    if (getenv("MGL_MS_DEBUG"))
        fprintf(stderr, "MGL ms-pass: %d ms vars, %d todo instrs (bound %u)\n", nms, ntodo, bound);

    /* types & constants: int / v2int / v3int / uint / bool, int 0..4, uint 0/1 */
    uint32_t sint_type = 0, v2int_type = 0, v3int_type = 0, uint_type = 0, bool_type = 0;
    for (uint32_t id = 1; id < bound; id++)
    {
        if (defop[id] == SpvOpTypeInt && defa[id] == 32)
        {
            if (defb[id] == 1 && !sint_type) sint_type = id;
            if (defb[id] == 0 && !uint_type) uint_type = id;
        }
        if (defop[id] == SpvOpTypeBool && !bool_type) bool_type = id;
        if (defop[id] == SpvOpTypeVector && defa[id] == sint_type && defb[id] == 2 && !v2int_type) v2int_type = id;
        if (defop[id] == SpvOpTypeVector && defa[id] == sint_type && defb[id] == 3 && !v3int_type) v3int_type = id;
    }
    if (!sint_type) { free(defop); free(defa); free(defb); return NULL; }
    /* the rewritten read/store of an arrayed image needs v3int; non-arrayed v2int
     * (the coordinate of the OpImageTexelPointer).  A pure-float image shader may
     * lack uint entirely (the atomic scope/semantics constants need it), and one
     * without comparisons may lack bool — synthesize whichever is missing. */
    uint32_t need_v2 = 0, need_v3 = 0;
    for (int t = 0; t < ntodo; t++)
        if (todo[t].arrayed) need_v3 = 1; else need_v2 = 1;

    uint32_t next_id = bound;
    if (!v2int_type && need_v2) v2int_type = next_id++;
    if (!v3int_type && need_v3) v3int_type = next_id++;
    if (!uint_type) uint_type = next_id++;
    if (!bool_type) bool_type = next_id++;

    uint32_t kint[5], kuint[2];
    for (int v = 0; v < 5; v++) kint[v] = 0;
    for (int v = 0; v < 2; v++) kuint[v] = 0;
    for (uint32_t id = 1; id < bound; id++)
    {
        if (defop[id] != SpvOpConstant) continue;
        if (defa[id] == sint_type && defb[id] < 5) kint[defb[id]] = id;
        if (defa[id] == uint_type && defb[id] < 2) kuint[defb[id]] = id;
    }
    for (int v = 0; v < 5; v++) if (!kint[v]) kint[v] = next_id++;
    for (int v = 0; v < 2; v++) if (!kuint[v]) kuint[v] = next_id++;

    /* per-base-type: OpTypePointer Image base + a zero constant */
    struct { uint32_t base, ptr, zero; } bases[8];
    int nbases = 0;
    for (int t = 0; t < ntodo; t++)
    {
        uint32_t base = defa[todo[t].imgt];
        int bi = -1;
        for (int k = 0; k < nbases; k++) if (bases[k].base == base) bi = k;
        if (bi < 0)
        {
            if (nbases >= 8) { free(defop); free(defa); free(defb); return NULL; }
            bi = nbases++;
            bases[bi].base = base; bases[bi].ptr = 0; bases[bi].zero = 0;
            for (uint32_t id = 1; id < bound; id++)
            {
                if (defop[id] == SpvOpTypePointer && defa[id] == SpvStorageClassImage &&
                    defb[id] == base && !bases[bi].ptr) bases[bi].ptr = id;
                if (defop[id] == SpvOpConstant && defa[id] == base && defb[id] == 0 &&
                    !bases[bi].zero) bases[bi].zero = id;
            }
            if (!bases[bi].ptr) bases[bi].ptr = next_id++;
            if (!bases[bi].zero) bases[bi].zero = next_id++;
        }
    }

    /* sweep 3: rebuild */
    MsBuf b = {0};
    for (int i = 0; i < 5; i++) ms_push(&b, in[i]);

    /* OpImageQuerySize / OpImageQuerySamples need the ImageQuery capability;
     * glslang omits it unless the shader itself queries — insert it right after
     * the header (capabilities may appear in any order among themselves) */
    {
        uint32_t t_[1] = { SpvCapabilityImageQuery };
        ms_inst(&b, SpvOpCapability, 2, t_);
    }

    for (size_t o = 5; o < n; )
    {
        uint32_t op = in[o] & 0xFFFFu, wc = in[o] >> 16;
        if (wc == 0 || o + wc > n) break;

        if (o == insert_pos && insert_pos)
        {
            /* synthesized types, then constants (a synthesized id is >= bound) */
            if (v2int_type >= bound)
            {
                uint32_t t_[3] = { v2int_type, sint_type, 2 };
                ms_inst(&b, SpvOpTypeVector, 4, t_);
            }
            if (v3int_type >= bound)
            {
                uint32_t t_[3] = { v3int_type, sint_type, 3 };
                ms_inst(&b, SpvOpTypeVector, 4, t_);
            }
            if (uint_type >= bound)
            {
                uint32_t t_[3] = { uint_type, 32, 0 };
                ms_inst(&b, SpvOpTypeInt, 4, t_);
            }
            if (bool_type >= bound)
            {
                uint32_t t_[1] = { bool_type };
                ms_inst(&b, SpvOpTypeBool, 2, t_);
            }
            for (int i = 0; i < nbases; i++) if (bases[i].ptr >= bound)
            {
                uint32_t t_[3] = { bases[i].ptr, SpvStorageClassImage, bases[i].base };
                ms_inst(&b, SpvOpTypePointer, 4, t_);
            }
            for (int v = 4; v >= 0; v--) if (kint[v] >= bound)
            {
                uint32_t t_[3] = { sint_type, kint[v], (uint32_t)v };
                ms_inst(&b, SpvOpConstant, 4, t_);
            }
            for (int v = 1; v >= 0; v--) if (kuint[v] >= bound)
            {
                uint32_t t_[3] = { uint_type, kuint[v], (uint32_t)v };
                ms_inst(&b, SpvOpConstant, 4, t_);
            }
            for (int i = 0; i < nbases; i++) if (bases[i].zero >= bound)
            {
                uint32_t t_[3] = { bases[i].base, bases[i].zero, 0 };
                ms_inst(&b, SpvOpConstant, 4, t_);
            }
        }

        int t = -1;
        for (int k = 0; k < ntodo; k++) if (todo[k].off == o) { t = k; break; }
        bool rewritten = false;

        if (t >= 0)
        {
            /* reads lower only for variables the module writes/atomics (their
             * sidecar exists); writes lower for any word-addressable format */
            bool do_lower = false;
            if (todo[t].kind == 3)
                do_lower = true;                         /* coordinate fold for atomics */
            else if (todo[t].ncomp > 0)
            {
                if (todo[t].kind == 1)
                    do_lower = true;                     /* this very instruction writes */
                else
                {
                    uint32_t var = 0, img2 = todo[t].image;
                    for (int hops = 0; img2 && hops < 8; hops++)
                    {
                        uint32_t dop = defop[img2];
                        if (dop == SpvOpLoad || dop == SpvOpAccessChain || dop == SpvOpInBoundsAccessChain)
                            img2 = defa[img2];
                        else if (dop == SpvOpVariable || dop == SpvOpFunctionParameter) { var = img2; break; }
                        else break;
                    }
                    for (int k = 0; k < nms && !do_lower; k++)
                        if (ms_vars[k] == var) do_lower = true;
                }
            }

            if (do_lower)
            {
                uint32_t base = defa[todo[t].imgt];
                int bi = -1;
                for (int k = 0; k < nbases; k++) if (bases[k].base == base) bi = k;
                uint32_t coord = todo[t].coord, sample = todo[t].sample;
                uint32_t w_[7];

                /* the query instructions need the LOADED image; an OpImageTexelPointer
                 * references the pointer, so load it once here (base_ptr: the uniform
                 * variable, or the array ELEMENT the access chain selected) */
                uint32_t img_val = todo[t].image;
                if (todo[t].kind == 3)
                {
                    img_val = next_id++;
                    uint32_t t5[3] = { todo[t].imgt, img_val, todo[t].base_ptr };
                    ms_inst(&b, SpvOpLoad, 4, t5);
                }

                /* ---- the sidecar word index, computed entirely in SPIR-V ----
                 * idx = comp + 4*sample + samples*(layer + layers*(x + width*y))
                 * (mgl_spirv_fix... geomImageAtomicTexels / geomMsSeedPso /
                 * geomMsFlushPso use the same layout).  The image dimensions come
                 * from OpImageQuerySize / OpImageQuerySamples, so the index is
                 * bounds-checked at runtime: an out-of-range texel or sample reads
                 * or writes the DUMP word past the used words — the same rule the
                 * MSL helpers use for the non-MS sidecar paths.  The coordinate of
                 * the rewritten OpImageTexelPointer is (idx, 0[, 0]) of the image's
                 * own dimensionality, so SPIRV-Cross's spvImage2DAtomicCoord macro
                 * (padded_width * y + x) reduces to idx exactly. */
                uint32_t qs = next_id++;
                w_[0] = todo[t].arrayed ? v3int_type : v2int_type; w_[1] = qs; w_[2] = img_val;
                ms_inst(&b, SpvOpImageQuerySize, 4, w_);
                uint32_t wv = next_id++, hv = next_id++, lvv;
                w_[0] = sint_type; w_[1] = wv; w_[2] = qs; w_[3] = 0;
                ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                w_[0] = sint_type; w_[1] = hv; w_[2] = qs; w_[3] = 1;
                ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                if (todo[t].arrayed)
                {
                    lvv = next_id++;
                    w_[0] = sint_type; w_[1] = lvv; w_[2] = qs; w_[3] = 2;
                    ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                }
                else lvv = kint[1];                       /* L = 1 */
                uint32_t sw = next_id++;                  /* sample count */
                w_[0] = sint_type; w_[1] = sw; w_[2] = img_val;
                ms_inst(&b, SpvOpImageQuerySamples, 4, w_);

                uint32_t xc = next_id++, yc = next_id++, lc = 0, s4 = next_id++;
                w_[0] = sint_type; w_[1] = xc; w_[2] = coord; w_[3] = 0;
                ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                w_[0] = sint_type; w_[1] = yc; w_[2] = coord; w_[3] = 1;
                ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                if (todo[t].arrayed)
                {
                    lc = next_id++;
                    w_[0] = sint_type; w_[1] = lc; w_[2] = coord; w_[3] = 2;
                    ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                }
                else lc = kint[0];
                w_[0] = sint_type; w_[1] = s4; w_[2] = sample; w_[3] = kint[4];
                ms_inst(&b, SpvOpIMul, 5, w_);            /* 4 * sample */

                uint32_t wy = next_id++, xy = next_id++;
                w_[0] = sint_type; w_[1] = wy; w_[2] = wv; w_[3] = yc;
                ms_inst(&b, SpvOpIMul, 5, w_);            /* width * y */
                w_[0] = sint_type; w_[1] = xy; w_[2] = xc; w_[3] = wy;
                ms_inst(&b, SpvOpIAdd, 5, w_);            /* x + width*y */
                uint32_t lxy = xy;
                if (todo[t].arrayed)
                {
                    uint32_t lxy_ = next_id++;
                    w_[0] = sint_type; w_[1] = lxy_; w_[2] = lvv; w_[3] = xy;
                    ms_inst(&b, SpvOpIMul, 5, w_);        /* layers * (x+w*y) */
                    lxy = next_id++;
                    w_[0] = sint_type; w_[1] = lxy; w_[2] = lc; w_[3] = lxy_;
                    ms_inst(&b, SpvOpIAdd, 5, w_);        /* layer + L*(x+w*y) */
                }
                uint32_t slxy = next_id++, idx0 = next_id++;
                w_[0] = sint_type; w_[1] = slxy; w_[2] = sw; w_[3] = lxy;
                ms_inst(&b, SpvOpIMul, 5, w_);            /* S * (layer + L*(x+w*y)) */
                {   /* the sidecar is word-major by (texel, sample): word =
                     * 4*(sample + S*(layer + L*(x+W*y))) + component — the factor 4
                     * on the SAMPLE term alone (4*s + S*t) COLLIDES across texels
                     * (e.g. (s=3,t=33) == (s=0,t=36) on an 8x8x4 image) and two
                     * fragments' stores then stomp each other */
                    uint32_t sl4 = next_id++;
                    w_[0] = sint_type; w_[1] = sl4; w_[2] = slxy; w_[3] = kint[4];
                    ms_inst(&b, SpvOpIMul, 5, w_);        /* 4 * S * (…) */
                    w_[0] = sint_type; w_[1] = idx0; w_[2] = s4; w_[3] = sl4;
                    ms_inst(&b, SpvOpIAdd, 5, w_);        /* 4*sample + 4*S*(…) */
                }

                /* bounds: 0<=x<w && 0<=y<h && 0<=layer<L && 0<=sample<S */
                uint32_t inb = 0, c_, a2;
                #define MS_AND(var_new, l, r)                                            \
                    do {                                                                  \
                        uint32_t an_ = next_id++;                                         \
                        uint32_t t5_[4] = { bool_type, an_, (l), (r) };                   \
                        ms_inst(&b, SpvOpLogicalAnd, 5, t5_);                             \
                        var_new = an_;                                                    \
                    } while (0)
                #define MS_LT(dst, l, r)                                                  \
                    do {                                                                  \
                        dst = next_id++;                                                  \
                        uint32_t t5_[4] = { bool_type, dst, (l), (r) };                   \
                        ms_inst(&b, SpvOpSLessThan, 5, t5_);                              \
                    } while (0)
                #define MS_GE(dst, l, r)                                                  \
                    do {                                                                  \
                        dst = next_id++;                                                  \
                        uint32_t t5_[4] = { bool_type, dst, (l), (r) };                   \
                        ms_inst(&b, SpvOpSGreaterThanEqual, 5, t5_);                      \
                    } while (0)
                MS_LT(inb, xc, wv);
                MS_GE(c_, xc, kint[0]); MS_AND(inb, inb, c_);
                MS_LT(a2, yc, hv); MS_AND(inb, inb, a2);
                MS_GE(c_, yc, kint[0]); MS_AND(inb, inb, c_);
                if (todo[t].arrayed)
                {
                    MS_LT(a2, lc, lvv); MS_AND(inb, inb, a2);
                    MS_GE(c_, lc, kint[0]); MS_AND(inb, inb, c_);
                }
                MS_LT(a2, sample, sw); MS_AND(inb, inb, a2);
                MS_GE(c_, sample, kint[0]); MS_AND(inb, inb, c_);
                #undef MS_AND
                #undef MS_LT
                #undef MS_GE

                /* words total (the DUMP index) = w*h*L*S*4 */
                uint32_t wh = next_id++, tot;
                w_[0] = sint_type; w_[1] = wh; w_[2] = wv; w_[3] = hv;
                ms_inst(&b, SpvOpIMul, 5, w_);
                if (todo[t].arrayed)
                { uint32_t q2 = next_id++; uint32_t t5[4] = { sint_type, q2, wh, lvv };
                  ms_inst(&b, SpvOpIMul, 5, t5); wh = q2; }
                { uint32_t q2 = next_id++; uint32_t t5[4] = { sint_type, q2, wh, sw };
                  ms_inst(&b, SpvOpIMul, 5, t5); wh = q2; }
                { uint32_t q2 = next_id++; uint32_t t5[4] = { sint_type, q2, wh, kint[4] };
                  ms_inst(&b, SpvOpIMul, 5, t5); wh = q2; }
                tot = wh;                              /* wh * 4 */

                /* (the first bounds AND above had operand-order noise; rebuild it) */
                /* per-component index + select */
                #define MS_IDX_FOR(c, out_idx, out_cc)                                        \
                    do {                                                                      \
                        uint32_t q4 = idx0;                                                   \
                        if ((c))                                                              \
                        {                                                                     \
                            q4 = next_id++;                                                   \
                            uint32_t t5[4] = { sint_type, q4, idx0, kint[c] };                \
                            ms_inst(&b, SpvOpIAdd, 5, t5);                                    \
                        }                                                                     \
                        out_idx = next_id++;                                                  \
                        { uint32_t t5[5] = { sint_type, out_idx, inb, q4, tot };              \
                          ms_inst(&b, SpvOpSelect, 6, t5); }                                  \
                        out_cc = next_id++;                                                   \
                        if (todo[t].arrayed)                                                  \
                        { uint32_t t5[5] = { v3int_type, out_cc, out_idx, kint[0], kint[0] }; \
                          ms_inst(&b, SpvOpCompositeConstruct, 6, t5); }                      \
                        else                                                                  \
                        { uint32_t t5[4] = { v2int_type, out_cc, out_idx, kint[0] };          \
                          ms_inst(&b, SpvOpCompositeConstruct, 5, t5); }                      \
                    } while (0)

                if (todo[t].kind == 1)
                {
                    for (int c = 0; c < todo[t].ncomp; c++)
                    {
                        uint32_t idx, cc, pt = next_id++, vc = next_id++;
                        MS_IDX_FOR(c, idx, cc);
                        w_[0] = bases[bi].ptr; w_[1] = pt; w_[2] = todo[t].base_ptr; w_[3] = cc; w_[4] = kint[0];
                        ms_inst(&b, SpvOpImageTexelPointer, 6, w_);
                        w_[0] = base; w_[1] = vc; w_[2] = todo[t].texel; w_[3] = (uint32_t)c;
                        ms_inst(&b, SpvOpCompositeExtract, 5, w_);
                        w_[0] = pt; w_[1] = kuint[1]; w_[2] = kuint[0]; w_[3] = vc;
                        ms_inst(&b, SpvOpAtomicStore, 5, w_);
                    }
                }
                else if (todo[t].kind == 2)
                {
                    int nout = todo[t].ncomp;
                    if (defop[todo[t].restype] == SpvOpTypeVector && defa[todo[t].restype] == base &&
                        defb[todo[t].restype] >= 1 && defb[todo[t].restype] <= 4)
                        nout = (int)defb[todo[t].restype];
                    uint32_t comps[4];
                    for (int c = 0; c < nout; c++)
                    {
                        if (c < todo[t].ncomp)
                        {
                            uint32_t idx, cc, pt = next_id++, lv = next_id++;
                            MS_IDX_FOR(c, idx, cc);
                            w_[0] = bases[bi].ptr; w_[1] = pt; w_[2] = todo[t].base_ptr; w_[3] = cc; w_[4] = kint[0];
                            ms_inst(&b, SpvOpImageTexelPointer, 6, w_);
                            w_[0] = base; w_[1] = lv; w_[2] = pt; w_[3] = kuint[1]; w_[4] = kuint[0];
                            ms_inst(&b, SpvOpAtomicLoad, 6, w_);
                            comps[c] = lv;
                        }
                        else comps[c] = bases[bi].zero;
                    }
                    if (nout == 1 && defop[todo[t].restype] != SpvOpTypeVector)
                    {
                        uint32_t c2[3] = { todo[t].restype, todo[t].resid, comps[0] };
                        ms_inst(&b, SpvOpCopyObject, 3, c2);
                    }
                    else
                    {
                        uint32_t c2[6] = { todo[t].restype, todo[t].resid, comps[0], comps[1], comps[2], comps[3] };
                        ms_inst(&b, SpvOpCompositeConstruct, (uint32_t)(nout + 3), c2);
                    }
                }
                else
                {
                    /* OpImageTexelPointer: keep result/type/image, new coordinate;
                     * the sample operand stays (spvc ignores it) */
                    uint32_t idx, cc;
                    MS_IDX_FOR(0, idx, cc);
                    uint32_t pw[6] = { in[o + 1], in[o + 2], in[o + 3], cc, in[o + 5] };
                    ms_inst(&b, SpvOpImageTexelPointer, 6, pw);
                }
                rewritten = true;
            }
        }

        if (!rewritten)
            for (uint32_t q = 0; q < wc; q++) ms_push(&b, in[o + q]);
        o += wc;
    }

    free(defop); free(defa); free(defb);
    if (b.n <= 5 || !b.w) { free(b.w); return NULL; }
    b.w[3] = next_id;
    if (getenv("MGL_MS_DUMP"))
    {
        char p_[256];
        snprintf(p_, sizeof p_, "%s/ms_lowered_%p.spv", getenv("MGL_MS_DUMP"), (void *)in);
        FILE *f = fopen(p_, "wb");
        if (f) { fwrite(b.w, 4, b.n, f); fclose(f); }
    }
    *out_n = b.n;
    return b.w;
}




static const char *glsl_type_name(GLenum base, GLuint rows, GLuint cols)
{
    static const char *scalar[5] = { "float", "double", "int", "uint", "bool" };
    static char buf[64];

    int k = base == GL_DOUBLE ? 1 : base == GL_INT ? 2 : base == GL_UNSIGNED_INT ? 3 : base == GL_BOOL ? 4 : 0;

    if (cols > 1)
    {
        if (cols == rows) snprintf(buf, sizeof(buf), "mat%d", cols);
        else snprintf(buf, sizeof(buf), "mat%dx%d", cols, rows);
        return buf;
    }
    if (rows == 1) return scalar[k];

    snprintf(buf, sizeof(buf), "%svec%d", base == GL_DOUBLE ? "d" : base == GL_INT ? "i" :
                                        base == GL_UNSIGNED_INT ? "u" : base == GL_BOOL ? "b" : "", rows);
    return buf;
}

static GLuint base_words(GLenum base)
{
    return base == GL_DOUBLE ? 2 : 1;
}

/* GLSL conversion applied when a value is stored into the word record */
static const char *word_store(GLenum base, const char *e)
{
    static char buf[256];
    if (base == GL_FLOAT) snprintf(buf, sizeof(buf), "floatBitsToUint(%s)", e);
    else if (base == GL_DOUBLE) snprintf(buf, sizeof(buf), "unpackDouble2x32(%s)", e);
    else snprintf(buf, sizeof(buf), "uint(%s)", e);
    return buf;
}

/* number of words a component of `base` occupies */
static GLuint comp_words(GLenum base) { return base == GL_DOUBLE ? 2 : 1; }

/* the fp64 interface representation of a double varying (fp64.c: iface_ty):
 * uvec2 for a scalar, uvec4 for dvec2, a flat uvec2 array for arrays of
 * scalars, and a flat uvec4 array of elems * cols * ceil(rows/2) slots
 * otherwise (dvec3/dvec4 members, matrices, and any array element wider than
 * one location, which SPIRV-Cross unrolls into one location each — matrices
 * and wide vectors use whole uvec4 slots even when a column holds a single
 * double).  Returns the slot count and writes the GLSL type (without array
 * suffix) into buf.  The raster-facing side must match fp64.c's slot types
 * bit for bit: Metal rejects a stage pair whose user attributes differ in
 * type. */
static GLuint geom_double_iface(GLuint rows, GLuint cols, GLuint elems, char *buf, size_t bufsz)
{
    if (elems <= 1 && cols <= 1)
    {
        if (rows <= 1) { snprintf(buf, bufsz, "uvec2"); return 1; }
        if (rows == 2) { snprintf(buf, bufsz, "uvec4"); return 1; }
    }
    if (cols <= 1 && rows <= 1)
    {
        /* array of scalar doubles: one uvec2 slot per element */
        snprintf(buf, bufsz, "uvec2");
        return elems;
    }
    snprintf(buf, bufsz, "uvec4");
    return elems * (cols ? cols : 1) * ((rows + 1) / 2);
}

/* emit "name = <value>;" statements reconstructing v from the word buffer.
 * open/close delimit the word access: <open> + " + N" + <close>.  `index` is the vertex
 * index expression for geometry-stage inputs (which are arrays of the record per vertex). */
/* is this var's GLSL expression "name.member" (an interface block member, or a member of
 * a struct-typed interface variable)?  An anonymous block's member is its own name. */
static bool geom_var_dotted(const MGLGeomVar *v)
{
    return v->member[0] && (v->plain_struct || strcmp(v->name, v->member));
}

/* the access path of a varying inside the rewritten shader: a named interface block
 * member is reached through its instance ("b.c"); an anonymous block member (the
 * reflection records the member name itself) and a plain varying use the bare name */
static const char *geom_var_expr(char *buf, size_t cap, const MGLGeomVar *v)
{
    if (geom_var_dotted(v))
    {
        if (v->block_array)
            snprintf(buf, cap, "%s[%u].%s", v->name, v->index, v->member);
        else
            snprintf(buf, cap, "%s.%s", v->name, v->member);
    }
    else
        snprintf(buf, cap, "%s", v->name);
    return buf;
}

static void gen_varying_load(SB *b, const MGLGeomVar *v, const char *open, const char *close,
                             const char *index)
{
    char prefix[192];

    if (index)
        snprintf(prefix, sizeof(prefix), "%s[%s]%s%s", v->name, index,
                 v->member[0] ? "." : "", v->member);
    else
        snprintf(prefix, sizeof(prefix), "%s%s%s", v->name, v->member[0] ? "." : "", v->member);

    for (GLuint e = 0; e < v->elems; e++)
    {
        char elem_prefix[64] = "";
        if (v->elems > 1 || v->arr) snprintf(elem_prefix, sizeof(elem_prefix), "[%u]", e);   /* r103 */

        if (v->cols > 1)
        {
            /* matrix: one vecN per column */
            char mname[64];
            snprintf(mname, sizeof(mname), "%s", glsl_type_name(v->base, v->rows, v->cols));
            sb_put(b, "    %s%s = %s(", prefix, elem_prefix, mname);
            for (GLuint c = 0; c < v->cols; c++)
            {
                if (c) sb_put(b, ", ");
                sb_put(b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r = 0; r < v->rows; r++)
                {
                    GLuint w = v->offset + (e * v->cols * v->rows + c * v->rows + r) * comp_words(v->base);
                    if (r) sb_put(b, ", ");
                    if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                    else if (v->base == GL_DOUBLE) sb_put(b, "packDouble2x32(uvec2(%s + %u%s, %s + %u%s))",
                                                          open, w, close, open, w + 1, close);
                    else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                    else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
                }
                sb_put(b, ")");
            }
            sb_put(b, ");\n");
        }
        else
        {
            sb_put(b, "    %s%s = ", prefix, elem_prefix);
            if (v->rows == 1 && v->base != GL_DOUBLE)
            {
                GLuint w = v->offset + e * v->rows * comp_words(v->base);
                if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
            }
            else
            {
                sb_put(b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r = 0; r < v->rows; r++)
                {
                    GLuint w = v->offset + (e * v->rows + r) * comp_words(v->base);
                    if (r) sb_put(b, ", ");
                    if (v->base == GL_FLOAT) sb_put(b, "uintBitsToFloat(%s + %u%s)", open, w, close);
                    else if (v->base == GL_DOUBLE) sb_put(b, "packDouble2x32(uvec2(%s + %u%s, %s + %u%s))",
                                                          open, w, close, open, w + 1, close);
                    else if (v->base == GL_BOOL) sb_put(b, "(%s + %u%s) != 0u", open, w, close);
                    else sb_put(b, "%s(%s + %u%s)", v->base == GL_INT ? "int" : "uint", open, w, close);
                }
                sb_put(b, ")");
            }
            sb_put(b, ";\n");
        }
    }
}

/* emit "dest[idx + off] = word;" statements storing v from the producer expression */
static void gen_varying_store(SB *b, const MGLGeomVar *v, const char *dest, const char *idx,
                              const char *src)
{
    static const char *comp[4] = { ".x", ".y", ".z", ".w" };

    for (GLuint e = 0; e < v->elems; e++)
    {
        char ep[32] = "";
        if (v->elems > 1 || v->arr) snprintf(ep, sizeof(ep), "[%u]", e);   /* r103: 1-element arrays still index */

        for (GLuint c = 0; c < v->cols; c++)
        {
            char cp[32] = "";
            if (v->cols > 1) snprintf(cp, sizeof(cp), "[%u]", c);

            for (GLuint r = 0; r < v->rows; r++)
            {
                char rp[8] = "";
                if (v->rows > 1) snprintf(rp, sizeof(rp), "%s", comp[r]);

                char expr[256];
                if (!src)
                    snprintf(expr, sizeof(expr), "0u");     /* undefined input: zeros */
                else
                    snprintf(expr, sizeof(expr), "%s%s%s%s", src, ep, cp, rp);

                GLuint w = v->offset + (e * v->cols * v->rows + c * v->rows + r) * comp_words(v->base);
                if (v->base == GL_DOUBLE)
                {
                    sb_put(b, "    { uvec2 mgl_w = unpackDouble2x32(%s); %s[%s + %u] = mgl_w.x; %s[%s + %u] = mgl_w.y; }\n",
                           expr, dest, idx, w, dest, idx, w + 1);
                }
                else
                {
                    sb_put(b, "    %s[%s + %u] = %s;\n", dest, idx, w, word_store(v->base, expr));
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ program table */

static void geom_free_prog(GLMContext ctx, MGLGeomProgram *gp);

/* ---- name -> slot map: open addressing, linear probing, backward-shift deletion ---- */

static int geom_map_find(struct MGLExtState_geom *g, GLuint name)
{
    if (!g || !g->map_cap || !name) return -1;
    GLuint mask = (GLuint)g->map_cap - 1;
    for (GLuint i = name & mask; ; i = (i + 1) & mask)
    {
        if (!g->map_names[i]) return -1;
        if (g->map_names[i] == name) return g->map_slots[i] - 1;
    }
}

static bool geom_map_insert(struct MGLExtState_geom *g, GLuint name, int slot)
{
    if (!g) return false;

    if ((g->map_used + 1) * 4 > g->map_cap * 3)   /* > 75% load: grow */
    {
        int ncap = g->map_cap ? g->map_cap * 2 : 128;
        GLuint *nn = (GLuint *)malloc((size_t)ncap * sizeof(GLuint));
        int   *ns = (int *)malloc((size_t)ncap * sizeof(int));
        if (!nn || !ns) { free(nn); free(ns); return false; }
        memset(nn, 0, (size_t)ncap * sizeof(GLuint));
        memset(ns, 0, (size_t)ncap * sizeof(int));
        GLuint nmask = (GLuint)ncap - 1;
        for (int i = 0; i < g->map_cap; i++)
        {
            if (!g->map_names[i]) continue;
            GLuint j = g->map_names[i] & nmask;
            while (nn[j]) j = (j + 1) & nmask;
            nn[j] = g->map_names[i];
            ns[j] = g->map_slots[i];
        }
        free(g->map_names);
        free(g->map_slots);
        g->map_names = nn;
        g->map_slots = ns;
        g->map_cap = ncap;
    }

    GLuint mask = (GLuint)g->map_cap - 1;
    GLuint i = name & mask;
    while (g->map_names[i]) i = (i + 1) & mask;
    g->map_names[i] = name;
    g->map_slots[i] = slot + 1;
    g->map_used++;
    return true;
}

static void geom_map_remove(struct MGLExtState_geom *g, GLuint name)
{
    if (!g || !g->map_cap) return;
    GLuint mask = (GLuint)g->map_cap - 1;
    GLuint i = name & mask;
    while (g->map_names[i] && g->map_names[i] != name) i = (i + 1) & mask;
    if (g->map_names[i] != name) return;

    /* backward-shift deletion keeps probe chains intact without tombstones */
    g->map_names[i] = 0;
    g->map_slots[i] = 0;
    g->map_used--;
    GLuint j = (i + 1) & mask;
    while (g->map_names[j])
    {
        GLuint k = g->map_names[j] & mask;
        /* can the entry at j live at i?  (circular interval test) */
        if ((j > i && (k <= i || k > j)) || (j < i && (k <= i && k > j)))
        {
            g->map_names[i] = g->map_names[j];
            g->map_slots[i] = g->map_slots[j];
            g->map_names[j] = 0;
            g->map_slots[j] = 0;
            i = j;
        }
        j = (j + 1) & mask;
    }
}

static MGLGeomProgram *geom_prog_slot(GLMContext ctx, Program *p, bool create)
{
    struct MGLExtState_geom *g = GEOM();

    if (!g || !p || !p->name) return NULL;

    int slot = geom_map_find(g, p->name);
    if (slot >= 0)
    {
        /* names are not recycled today, but if they ever are, the slot must follow the
         * live object (the link path sees a scratch Program with the same name) */
        if (g->progs[slot].program != p && g->progs[slot].name == p->name)
            g->progs[slot].program = findProgram(ctx, p->name);
        return &g->progs[slot];
    }
    if (!create) return NULL;

    /* take a slot from the free list (its program died: mgl_ext_delete_program_geom) */
    if (g->free_head >= 0)
    {
        slot = g->free_head;
        g->free_head = g->progs[slot].next_free;
        geom_free_prog(ctx, &g->progs[slot]);
    }
    else
    {
        if (g->n_progs >= g->progs_cap)
        {
            int cap = g->progs_cap ? g->progs_cap * 2 : MGL_GEOM_MAX_PROGRAMS;
            MGLGeomProgram *grown = realloc(g->progs, (size_t)cap * sizeof(*grown));
            if (!grown) return NULL;
            memset(grown + g->n_progs, 0, (size_t)(cap - g->n_progs) * sizeof(*grown));
            g->progs = grown;
            g->progs_cap = cap;
        }
        slot = g->n_progs++;
    }

    MGLGeomProgram *gp = &g->progs[slot];
    gp->next_free = -1;
    gp->name = p->name;
    /* findProgram, NOT getProgram: getProgram would *create* a Program if the name
     * were somehow absent, growing the program table as a side effect */
    gp->program = findProgram(ctx, p->name);
    if (!geom_map_insert(g, p->name, slot))
    {
        memset(gp, 0, sizeof(*gp));
        if (slot == g->n_progs - 1) g->n_progs--;
        return NULL;
    }
    return gp;
}

/* the GL program object is being destroyed: release its geometry emulation (Metal
 * functions / PSO / MSL) and recycle the slot through the free list */
void mgl_ext_delete_program_geom(GLMContext ctx, Program *p)
{
    struct MGLExtState_geom *g = GEOM();

    if (!g || !p || !p->name) return;

    int slot = geom_map_find(g, p->name);
    if (slot < 0) return;

    geom_map_remove(g, p->name);
    geom_free_prog(ctx, &g->progs[slot]);
    g->progs[slot].next_free = g->free_head;
    g->free_head = slot;
}

static void geom_free_prog(GLMContext ctx, MGLGeomProgram *gp)
{
    if (gp->capture_function && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_function);
    if (gp->capture_library && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_library);
    /* the synthetic cull kernel has no gs Shader to live in (tess domain cull emulation) */
    if (gp->kernel_function && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->kernel_function);
    if (gp->kernel_library && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->kernel_library);
    if (gp->draw_function && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_function);
    if (gp->draw_library && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_library);
    if (gp->capture_pso && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_pso);
    /* r100: the capture-pass setup cache */
    if (gp->capture_vd && ctx->mtl_funcs.ext_program)
        ctx->mtl_funcs.ext_program->release_obj(ctx, gp->capture_vd);
    free(gp->kernel_msl);
    free(gp->kernel_entry);
    free(gp->final_entry);
    free(gp->capture_msl);
    free(gp->capture_entry);
    free(gp->draw_msl);
    free(gp->draw_entry);
    free(gp->fail);
    memset(gp, 0, sizeof(*gp));
}

/* remove `layout(location = N) ` prefixes from uniform/buffer BLOCK members of a
 * decompiled GLSL source (SPIRV-Cross emits them; glslang only accepts locations on
 * in/out block members).  Mutates the buffer in place; block-scoped, brace-counted. */
static void geom_strip_block_member_locations(char *src)
{
    char *p = src;
    int in_block = 0;      /* 1 inside a uniform/buffer block */
    int brace_depth = 0;

    while (*p)
    {
        if (in_block && brace_depth == 1 && !strncmp(p, "layout(location", 15))
        {
            /* only strip when the layout starts a member declaration */
            char *q = p;

            while (*q && *q != ')') q++;
            if (*q == ')')
            {
                q++;
                if (*q == ' ') q++;
                memmove(p, q, strlen(q) + 1);
                continue;
            }
        }

        if (!in_block && (!strncmp(p, "uniform", 7) || !strncmp(p, "buffer", 6)))
        {
            /* a block starts at "uniform <name> {" / "buffer <name> {" / "uniform {" */
            char *q = p + ((p[0] == 'u') ? 7 : 6);
            int names = 0;

            while (*q && *q != '{' && *q != ';')
            {
                if (*q != ' ' && *q != '\t' && *q != '\n')
                {
                    names++;
                    if (names > 1) break;
                    while (*q && *q != '{' && *q != ';' && *q != ' ' && *q != '\t' && *q != '\n') q++;
                    continue;
                }
                q++;
            }
            if (*q == '{' && names <= 1)
            {
                in_block = 1;
                brace_depth = 0;
                p = q;             /* point at '{' */
            }
        }

        if (*p == '{') brace_depth++;
        else if (*p == '}')
        {
            brace_depth--;
            if (in_block && brace_depth <= 0)
            {
                in_block = 0;
                brace_depth = 0;
            }
        }
        p++;
    }
}

static void geom_fail(GLMContext ctx, MGLGeomProgram *gp, const char *fmt, ...)
{
    va_list ap;
    char buf[4096];

    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    free(gp->fail);
    gp->fail = xstrdup(buf);
    gp->linked = false;
    if (getenv("MGL_GEOM_DEBUG") || getenv("MGL_GEOM_TRACE_DRAW"))
        fprintf(stderr, "geom: kernel build failed: %s\n", buf);
}

/* ------------------------------------------------------------------ glslang / SPIRV-Cross wrappers */

/* expand the preprocessor conditionals of a shader source.  The geometry emulation rewrites
 * sources line by line, which cannot follow a conditional that opens inside one declaration
 * and closes inside another (the CTS declares an input block whose closing brace sits within
 * an #ifdef/#else pair); the expanded source has no conditionals left, so the rewrite works
 * on straight-line text.  Returns a malloc'd string, or NULL when preprocessing fails - the
 * callers then fall back to the original source (whose own compile will report the error). */
static char *geom_preprocess_source(GLMContext ctx, GLuint glm_type, const char *src)
{
    glslang_input_t input;
    glslang_shader_t *shader;
    const char *code;
    char *out;

    memset(&input, 0, sizeof(input));
    initGLSLInput(ctx, glm_type, src, &input);
    shader = glslang_shader_create(&input);
    if (!shader) return NULL;
    glslang_shader_set_options(shader, GLSLANG_SHADER_VULKAN_RULES_RELAXED);
    if (!glslang_shader_preprocess(shader, &input))
    {
        glslang_shader_delete(shader);
        return NULL;
    }
    code = glslang_shader_get_preprocessed_code(shader);
    out = code ? strdup(code) : NULL;
    glslang_shader_delete(shader);
    return out;
}

static bool geom_glsl_to_spirv(GLMContext ctx, GLuint glm_type, const char *src, int force_version,
                               uint32_t **ir, size_t *words, char **log)
{
    glslang_input_t input;
    glslang_shader_t *shader;
    glslang_program_t *prog;
    const char *msgs;
    int version = 330, options = GLSLANG_SHADER_VULKAN_RULES_RELAXED;

    /* VULKAN_RULES_RELAXED moves 64-bit vertex attributes to Private storage
     * (Vulkan has no Float64 vertex inputs) — and the capture variant embeds
     * the user's vertex source, where `in double x` must stay a real attribute
     * (the vertex descriptor feeds u32 slots for it; the fp64 lowering maps
     * them).  Drop the flag when the source declares one. */
    if (glm_type == GL_VERTEX_SHADER)
    {
        for (const char *p = src; (p = strstr(p, "in ")); p += 3)
        {
            const char *t = p + 3;
            while (*t == ' ' || *t == '\t') t++;
            if (!strncmp(t, "double", 6) && !isalpha((unsigned char)t[6]) && !isdigit((unsigned char)t[6]))
                { options &= ~GLSLANG_SHADER_VULKAN_RULES_RELAXED; break; }
            if (!strncmp(t, "dvec", 4) || !strncmp(t, "dmat", 4))
                { options &= ~GLSLANG_SHADER_VULKAN_RULES_RELAXED; break; }
        }
    }

    memset(&input, 0, sizeof(input));
    /* GL-legal but SPIR-V-hostile constructs (layout(shared)/layout(packed)
     * uniform blocks — the CTS's named_uniform_blocks uses all three layouts)
     * must be rewritten the same way the program's own compile does
     * (shaders.c: mgl_cts_glsl_compat).  glslang copies the source; the string
     * must stay alive until then. */
    char *compat = mgl_cts_glsl_compat_generated(src);
    initGLSLInput(ctx, glm_type, compat ? compat : src, &input);
    if (force_version && input.default_version < force_version)
    {
        input.default_version = force_version;
        input.client_version = GLSLANG_TARGET_OPENGL_450;
    }

    {
        const char *vstr = strstr(src, "#version");
        if (vstr) sscanf(vstr, "#version %d", &version);
    }
    /* match the program's own compile (shaders.c mglCompileShader): GL assigns interface
     * locations and bindings at link time, so a 330+ shader whose outputs/inputs have no
     * layout(location) (legal GL, e.g. the CTS's "out uint vs_gs_index;") must compile here
     * too.  The invented locations never reach the record layout — the capture variant
     * stores outputs explicitly and the kernel reads them back by record offset — and the
     * invented bindings are re-pinned below (geom_spirv_to_msl), so auto-mapping is inert. */
    options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS | GLSLANG_SHADER_AUTO_MAP_BINDINGS;

    shader = glslang_shader_create(&input);
    if (!shader) { mgl_append_log(log, "ERROR: geometry: glslang shader creation failed\n"); free(compat); return false; }
    glslang_shader_set_options(shader, options);

    if (!glslang_shader_preprocess(shader, &input) ||
        !glslang_shader_parse(shader, &input))
    {
        const char *info = glslang_shader_get_info_log(shader);
        const char *debug = glslang_shader_get_info_debug_log(shader);
        mgl_append_log(log, "ERROR: geometry: GLSL → SPIR-V failed\n%s\n%s", info ? info : "", debug ? debug : "");
        glslang_shader_delete(shader);
        free(compat);
        return false;
    }
    free(compat);   /* glslang has copied the source by now */

    prog = glslang_program_create();
    if (!prog) { glslang_shader_delete(shader); return false; }
    glslang_program_add_shader(prog, shader);

    if (!glslang_program_link(prog, GLSLANG_MSG_DEFAULT_BIT))
    {
        const char *info = glslang_program_get_info_log(prog);
        mgl_append_log(log, "ERROR: geometry: glslang program link failed\n%s", info ? info : "");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    /* run the I/O mapper (as program.c's link does): it is what actually assigns the
     * locations AUTO_MAP_LOCATIONS only promises — without it a shader whose in/out have
     * no layout(location) compiles but SPIRV-Cross then emits a stage_in struct whose
     * members carry no [[attribute(n)]], which Metal rejects */
    if (!glslang_program_map_io(prog))
    {
        const char *info = glslang_program_get_info_log(prog);
        mgl_append_log(log, "ERROR: geometry: glslang I/O mapping failed\n%s", info ? info : "");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    glslang_program_SPIRV_generate(prog, getGLSLStage(glm_type));

    msgs = glslang_program_SPIRV_get_messages(prog);
    if (msgs && *msgs) mgl_append_log(log, "%s\n", msgs);

    size_t n = glslang_program_SPIRV_get_size(prog);
    if (!n)
    {
        mgl_append_log(log, "ERROR: geometry: empty SPIR-V module\n");
        glslang_program_delete(prog);
        glslang_shader_delete(shader);
        return false;
    }

    uint32_t *out = xmalloc(n * sizeof(uint32_t));
    glslang_program_SPIRV_get(prog, out);
    glslang_program_delete(prog);
    glslang_shader_delete(shader);

    /* fp64: the generated variant re-declares the stage's uniform blocks and
     * keeps the body's doubles, so the software-double lowering program.c runs
     * on its own stage modules must run here too — otherwise SPIRV-Cross sees
     * double members in a buffer ("double types are not supported in buffers
     * in MSL").  vertex_inputs matches program.c's rule (VS stage = true). */
    {
        bool lowered64 = false;
        size_t lwords = 0;
        char llog[512] = {0};
        uint32_t *lowered = mgl_fp64_lower_module(out, n, glm_type == GL_VERTEX_SHADER,
                                                  &lowered64, &lwords, llog, sizeof llog);
        if (lowered64 && lowered)
        {
            free(out);
            out = lowered;
            n = lwords;
        }
        else if (lowered64)
        {
            mgl_append_log(log, "ERROR: geometry: fp64 lowering of the generated variant failed: %s\n",
                           llog[0] ? llog : "?");
            free(out);
            return false;
        }
    }

    *ir = out;
    *words = n;
    return true;
}

/* decompile SPIR-V back to GLSL (used when no source is available: ShaderBinary / program
 * binary) so that the same source transform can be applied */
static char *geom_spirv_to_glsl(const uint32_t *ir, size_t words, int version, char **log)
{
    spvc_context ctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_compiler_options opt;
    spvc_result r;
    const char *src = NULL;

    r = spvc_context_create(&ctx);
    if (r != SPVC_SUCCESS) return NULL;

    r = spvc_context_parse_spirv(ctx, ir, words, &pir);
    if (r != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V parse failed (%s)\n", spvc_context_get_last_error_string(ctx));
        spvc_context_destroy(ctx);
        return NULL;
    }

    r = spvc_context_create_compiler(ctx, SPVC_BACKEND_GLSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp);
    if (r != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → GLSL compiler creation failed\n");
        spvc_context_destroy(ctx);
        return NULL;
    }

    r = spvc_compiler_create_compiler_options(comp, &opt);
    if (r == SPVC_SUCCESS)
    {
        spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_GLSL_VERSION, (unsigned)version);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_GLSL_VULKAN_SEMANTICS, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_EMIT_LINE_DIRECTIVES, SPVC_FALSE);
        spvc_compiler_install_compiler_options(comp, opt);
    }

    r = spvc_compiler_compile(comp, &src);
    if (r != SPVC_SUCCESS || !src)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → GLSL failed (%s)\n",
                       spvc_context_get_last_error_string(ctx));
        spvc_context_destroy(ctx);
        return NULL;
    }

    char *out = xstrdup(src);
    spvc_context_destroy(ctx);
    return out;
}

/* translate SPIR-V to MSL with the given buffer pins; returns the MSL source */
/* record the geometry stage's sampler/image array sizes: the kernel's Metal argument for
 * array<texture2d, N> spans N slots, so the binder needs the element count per resource */
static void geom_record_tex_arrays(const uint32_t *ir, size_t words, MGLGeomProgram *gp)
{
    spvc_context sctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_resources res;
    const spvc_reflected_resource *list = NULL;
    size_t count = 0;
    int types[2] = { SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, SPVC_RESOURCE_TYPE_STORAGE_IMAGE };

    gp->n_tex_arrays = 0;
    if (spvc_context_create(&sctx) != SPVC_SUCCESS) return;
    if (spvc_context_parse_spirv(sctx, ir, words, &pir) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }
    if (spvc_context_create_compiler(sctx, SPVC_BACKEND_MSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }
    if (spvc_compiler_create_shader_resources(comp, &res) != SPVC_SUCCESS)
    { spvc_context_destroy(sctx); return; }

    for (int t = 0; t < 2; t++)
    {
        const char *tn = NULL;
        size_t n = 0;

        if (types[t] == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE)
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, &list, &n);
        else
            spvc_resources_get_resource_list_for_type(res, SPVC_RESOURCE_TYPE_STORAGE_IMAGE, &list, &n);
        (void)tn; (void)count;
        for (size_t i = 0; i < n && gp->n_tex_arrays < 8; i++)
        {
            spvc_type ty = spvc_compiler_get_type_handle(comp, list[i].type_id);
            unsigned dims = spvc_type_get_num_array_dimensions(ty);

            if (!dims) continue;
            unsigned asz = (unsigned)spvc_type_get_array_dimension(ty, 0);
            if (!asz) continue;
            snprintf(gp->tex_arrays[gp->n_tex_arrays].name, sizeof gp->tex_arrays[0].name,
                     "%s", list[i].name);
            gp->tex_arrays[gp->n_tex_arrays].n = (int)asz;
            gp->n_tex_arrays++;
        }
    }
    spvc_context_destroy(sctx);
}

/* the original OpName spelling of a SPIR-V id (before mgl_msl_mangle_reserved's
 * compiler-side renames), for recording slot-table keys under the name the
 * program's resource list uses */
static bool geom_spv_opname(const uint32_t *ir, size_t words, uint32_t id,
                            char *out, size_t outsz)
{
    size_t pos;

    if (!ir || words < 5 || ir[0] != SpvMagicNumber)
        return false;
    for (pos = 5; pos < words; )
    {
        uint32_t wc = ir[pos] >> 16;
        uint32_t op = ir[pos] & 0xFFFF;

        if (wc == 0 || pos + wc > words)
            break;
        if (op == SpvOpName && wc >= 3 && ir[pos + 1] == id)
        {
            /* literal string from word 2: 4 bytes little-endian per word, NUL-terminated */
            const char *bytes = (const char *)(ir + pos + 2);
            size_t maxb = (size_t)(wc - 2) * 4, k = 0;

            while (k < maxb && k + 1 < outsz && bytes[k])
            {
                out[k] = bytes[k];
                k++;
            }
            out[k] = 0;
            return *out != 0;
        }
        pos += wc;
    }
    return false;
}

static char *geom_spirv_to_msl(GLMContext ctx, GLuint glm_type, const uint32_t *ir, size_t words,
                               const char *entry, bool disable_rasterization, char **log,
                               int pin_stage, const int *pins_glsl, const int *pins_msl, int pin_count,
                               const GeomPin *pins2, int npins2,
                               MGLGeomTexSlot *slots, int *nslots, int max_slots,
                               bool native_arrays)
{
    spvc_context sctx;
    spvc_parsed_ir pir;
    spvc_compiler comp;
    spvc_compiler_options opt;
    spvc_msl_resource_binding binding;
    spvc_result r;
    const char *msl = NULL;

    r = spvc_context_create(&sctx);
    if (r != SPVC_SUCCESS) return NULL;

    /* GL rectangle storage images (image2DRect) have no MSL mapping — SPIRV-Cross emits
     * `unknown_texture_type`, which never compiles.  compile_stage patches the raster
     * stages' modules; the geometry domain's generated kernels need it too (program.c's
     * mgl_program_spirv_rect_to_2d).  The parser copies the words, so the patch buffer
     * can go as soon as it returns. */
    {
        const uint32_t *parse_ir = ir;
        size_t parse_words = words;
        uint32_t *fixed = NULL;
        size_t fixed_words = 0;

        if (mgl_program_spirv_rect_to_2d(ir, words, &fixed, &fixed_words) > 0 && fixed)
        {
            parse_ir = fixed;
            parse_words = fixed_words;
        }
        /* glslang's GL target emits the SPV_KHR_subgroup_vote/ballot opcodes for the
         * group-vote/ballot calls (GL_ARB_shader_ballot in a geometry shader —
         * KHR-GL46.shader_ballot_tests): the MSL backend lowers only the
         * OpGroupNonUniform* core forms, and an un-rewritten module keeps the bare
         * call names ("use of undeclared identifier 'ballotARB'").  The same pass
         * program.c's compile_stage applies (spirv_vote.c). */
        {
            size_t vf_words = 0;
            uint32_t *vf = mgl_spirv_vote_khr_rewrite(parse_ir, parse_words, &vf_words);

            if (vf)
            {
                free(fixed);
                fixed = vf;
                parse_ir = fixed;
                parse_words = vf_words;
            }
        }
        r = spvc_context_parse_spirv(sctx, parse_ir, parse_words, &pir);
        free(fixed);
    }

    if (r != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V parse failed (%s)\n", spvc_context_get_last_error_string(sctx));
        spvc_context_destroy(sctx);
        return NULL;
    }

    if (spvc_context_create_compiler(sctx, SPVC_BACKEND_MSL, pir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &comp) != SPVC_SUCCESS)
    {
        mgl_append_log(log, "ERROR: geometry: MSL compiler creation failed\n");
        spvc_context_destroy(sctx);
        return NULL;
    }

    if (spvc_compiler_create_compiler_options(comp, &opt) == SPVC_SUCCESS)
    {
        spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_MSL_VERSION, 30000);
        if (disable_rasterization)
            spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_DISABLE_RASTERIZATION, SPVC_TRUE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE);
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, SPVC_TRUE);
        /* match the render stages' option (program.c): image1D/image1DArray compile to
         * texture2d/texture2d_array — the image-unit views are built that way (MGL stores
         * 1D textures as 2D single-row), and image-atomic subscripts then reuse the 2D /
         * array linearizations mgl_msl_fix_image_atomics emits.  Without it a GS using
         * image1DArray atomics emits `g_x_atomic[int2(...)]` (subscript not an integer)
         * and the kernel fails to compile (basic-allTargets-atomicGS). */
        spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D, SPVC_TRUE);
        /* .length() on an SSBO runtime array compiles to a `constant uint*
         * spvBufferSizeConstants [[buffer(N)]]` argument.  SPIRV-Cross's default N is 25
         * — the capture variant's VSOUT slot — and the two collide so hard the Metal
         * compiler rejects the variant ("cannot reserve 'buffer' resource location at
         * index 25", the round-45 VSOUT@25 pin): move it into the user slot range, whose
         * last slot MGL_GEOM_KERNEL_SIZE_BUFFER geom_build_pins keeps free for exactly
         * this argument (kernel and capture variants alike) */
        spvc_compiler_options_set_uint(opt, SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX,
                                       MGL_GEOM_KERNEL_SIZE_BUFFER);
        /* geom r102: thread-local arrays as NATIVE arrays.  SPIRV-Cross maps a GLSL
         * array in thread storage to spvUnsafeArray<T,N>, but a uniform-block member
         * array keeps its raw C-array shape; copying one to the other (`goku = uni_goku;`
         * — a dmat array, KHR-Single-GL46 enhanced_layouts.xfb_explicit_location_dmat2_
         * geometry) then lowers to `goku[0] = uni_goku[0]`, spvUnsafeArray<ulong2,2> =
         * ulong2[2], for which no assignment operator exists and Metal rejects the
         * kernel.  With native arrays the copy is a per-component loop and compiles. */
        if (native_arrays)
            spvc_compiler_options_set_bool(opt, SPVC_COMPILER_OPTION_MSL_FORCE_NATIVE_ARRAYS, SPVC_TRUE);
        spvc_compiler_install_compiler_options(comp, opt);
    }

    spvc_compiler_msl_add_discrete_descriptor_set(comp, 3);

    if (pin_stage >= 0)
    {
        for (int i = 0; i < pin_count; i++)
        {
            memset(&binding, 0, sizeof(binding));
            binding.stage = (SpvExecutionModel)pin_stage;
            binding.desc_set = 0;
            binding.binding = (unsigned)pins_glsl[i];
            binding.msl_buffer = (unsigned)pins_msl[i];
            binding.msl_texture = 0;
            binding.msl_sampler = 0;
            if (spvc_compiler_msl_add_resource_binding(comp, &binding) != SPVC_SUCCESS)
            {
                mgl_append_log(log, "ERROR: geometry: could not pin MSL buffer %d\n", pins_msl[i]);
                spvc_context_destroy(sctx);
                return NULL;
            }
        }

        /* user resources of the stage (uniform / storage buffers, samplers): pinned to the
         * flat positions of the renderer's buffer map, so the buffer map built for the
         * program's own stage binds them verbatim for the emulation shaders too (see
         * geom_build_pins) */
        for (int i = 0; i < npins2; i++)
        {
            memset(&binding, 0, sizeof(binding));
            binding.stage = (SpvExecutionModel)pin_stage;
            binding.desc_set = 0;
            binding.binding = (unsigned)pins2[i].binding;
            if (pins2[i].texture)
            {
                binding.msl_texture = (unsigned)pins2[i].msl;
                binding.msl_sampler = (unsigned)pins2[i].msl;
            }
            else
            {
                binding.msl_buffer = (unsigned)pins2[i].msl;
            }
            if (spvc_compiler_msl_add_resource_binding(comp, &binding) != SPVC_SUCCESS)
            {
                mgl_append_log(log, "ERROR: geometry: could not pin MSL resource %d\n", pins2[i].binding);
                spvc_context_destroy(sctx);
                return NULL;
            }
        }
    }

    (void)glm_type;

    /* GLSL identifiers that collide with MSL keywords/types must be renamed (CTS shaders
     * declare e.g. `uniform sampler2D sampler;`); see msl_reserved.h. */
    mgl_msl_mangle_reserved(comp, ir, words);

    if (spvc_compiler_compile(comp, &msl) != SPVC_SUCCESS || !msl)
    {
        mgl_append_log(log, "ERROR: geometry: SPIR-V → MSL failed (%s)\n", spvc_context_get_last_error_string(sctx));
        spvc_context_destroy(sctx);
        return NULL;
    }

    /* uint-in-MSL / int-in-GL index builtins (msl_reserved.c): `min(4, gl_VertexID)'
     * is an ambiguous Metal overload — the GS emulation's gl_PrimitiveID /
     * gl_InvocationID carry the same rule.  The fixed copy is consumed by the strdup
     * right below (the spvc context owns the original string). */
    char *vid_fixed = mgl_msl_fix_int_builtins(msl);
    char *out = xstrdup(vid_fixed ? vid_fixed : msl);
    free(vid_fixed);

    /* the bind walks need the Metal texture indices this variant ACTUALLY uses: the
     * pins above may be unreachable (a user image and the packed default-uniform block
     * can share SPIR-V binding 0 — MSLResourceBinding is keyed by set/binding alone, so
     * one of the two registrations is lost and SPIRV-Cross falls back to its own
     * numbering).  Query the final index per resource instead of guessing. */
    if (slots && nslots)
    {
        spvc_resources res = NULL;
        spvc_reflected_resource *list;
        size_t n = 0;

        if (spvc_compiler_create_shader_resources(comp, &res) == SPVC_SUCCESS)
        {
            static const int tex_types[2] = { SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
                                              SPVC_RESOURCE_TYPE_SAMPLED_IMAGE };

            *nslots = 0;
            for (int t = 0; t < 2 && *nslots < max_slots; t++)
            {
                if (spvc_resources_get_resource_list_for_type(res, tex_types[t], &list, &n) != SPVC_SUCCESS)
                    continue;
                for (size_t i = 0; i < n && *nslots < max_slots; i++)
                {
                    const char *nm = spvc_compiler_get_name(comp, list[i].id);
                    int msl;

                    if (!nm || !*nm) continue;
                    msl = (int)spvc_compiler_msl_get_automatic_resource_binding(comp, list[i].id);
                    snprintf(slots[*nslots].name, sizeof slots[0].name, "%s", nm);
                    slots[(*nslots)++].msl = msl;

                    /* mgl_msl_mangle_reserved renamed MSL keywords/keyword-colliding
                     * identifiers ("texture", "sampler", "writeonly", ...) to name_mgl[N]
                     * IN THE COMPILER, so the recorded name is the mangled one while the
                     * bind walks (geomTexSlot) key on the program's SPIR-V resource name —
                     * the ORIGINAL identifier.  Record that spelling too, or every
                     * sampler the shader happens to name "texture" misses the table and
                     * falls back to the (wrong) stage-list msl binding
                     * (KHR-GL46.texture_view.view_sampling: all stages declare
                     * 'uniform sampler2D texture;'). */
                    {
                        char orig[256];
                        bool have_orig = geom_spv_opname(ir, words, list[i].id, orig, sizeof orig);
                        const char *canon = have_orig ? orig : nm;

                        if (have_orig && strcmp(orig, nm) && *nslots < max_slots)
                        {
                            snprintf(slots[*nslots].name, sizeof slots[0].name, "%s", orig);
                            slots[(*nslots)++].msl = msl;
                        }
                        /* mgl_spirv_rect_to_2d (program.c) renames every RECT sampler to
                         * name+"_mglrect" in the MSL-facing module — and the stage's
                         * RESOURCE LIST is introspected from that module, so a rect
                         * sampler's sres->name carries the suffix while this compile's
                         * SPIR-V still has the original OpName.  Record the alias or the
                         * bind misses the table (KHR-GL46.texture_view.view_sampling,
                         * RECTANGLE iteration: the geometry kernel bound msl 0 instead
                         * of 1 and sampled the unit's leftover GL_TEXTURE_2D binding). */
                        if (*canon && *nslots < max_slots &&
                            strlen(canon) + strlen("_mglrect") < sizeof slots[0].name)
                        {
                            snprintf(slots[*nslots].name, sizeof slots[0].name,
                                     "%s_mglrect", canon);
                            slots[(*nslots)++].msl = msl;
                        }
                    }
                }
            }
        }
    }

    spvc_context_destroy(sctx);

    /* fp64: the soft-float library functions take scalar parameters by
     * `thread const T&`; a `constant`-space buffer member cannot bind to that
     * (program.c / transform_feedback.c run the same fix on their MSL). */
    {
        char *fixed = mgl_fp64_fix_msl_scalar_params(out);
        if (fixed)
        {
            free(out);
            out = fixed;
        }
    }
    return out;
}

/* ------------------------------------------------------------------ source transform
 * Rewrites a geometry shader into the body of a compute shader:
 *   - global `in` / `out` declarations lose their storage qualifier (they become plain
 *     globals), input arrays get the input primitive size, and gl_PerVertex block
 *     declarations are dropped (replaced by the emulation's own `mgl_in[]`)
 *   - reserved identifiers are renamed (gl_Position, gl_in, EmitVertex, ...) so the
 *     transformed source can be wrapped by the emulation prelude
 */

typedef struct {
    const char *from, *to;
} Rename;

static const Rename geom_renames[] = {
    { "gl_PrimitiveIDIn",     "mgl_gs_prim_in" },
    { "gl_InvocationID",      "mgl_gs_invoc" },
    { "gl_ViewportIndex",     "mgl_gs_vp" },
    { "gl_PrimitiveID",       "mgl_gs_prim_out" },
    { "gl_ClipDistance",      "mgl_gs_clip" },
    { "gl_CullDistance",      "mgl_gs_cull" },
    { "gl_PointSize",         "mgl_gs_psize" },
    { "gl_Position",          "mgl_gs_pos" },
    { "gl_Layer",             "mgl_gs_layer" },
    { "gl_in",                "mgl_in" },
    { "EmitStreamVertex",     "mgl_gs_emit_stream_vertex" },
    { "EndStreamPrimitive",   "mgl_gs_end_stream_primitive" },
    { "EmitVertex",           "mgl_gs_emit_vertex" },
    { "EndPrimitive",         "mgl_gs_end_primitive" },
    { NULL, NULL }
};

static bool is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static size_t read_ident(const char *s, char *out, size_t cap)
{
    size_t n = 0;
    while (s[n] && is_ident_char(s[n]))
    {
        if (n + 1 < cap) out[n] = s[n];
        n++;
    }
    out[n < cap ? n : cap - 1] = 0;
    return n;
}

static const char *rename_ident(const char *id)
{
    for (int i = 0; geom_renames[i].from; i++)
        if (!strcmp(id, geom_renames[i].from)) return geom_renames[i].to;
    return NULL;
}

/* qualifiers that may precede in/out in a GLSL interface declaration */
static bool is_interface_qualifier(const char *w)
{
    return !strcmp(w, "layout") || !strcmp(w, "flat") || !strcmp(w, "smooth") ||
           !strcmp(w, "noperspective") || !strcmp(w, "centroid") || !strcmp(w, "sample") ||
           !strcmp(w, "precise") || !strcmp(w, "invariant") || !strcmp(w, "in") || !strcmp(w, "out") ||
           !strcmp(w, "highp") || !strcmp(w, "mediump") || !strcmp(w, "lowp");
}

/* skip whitespace and comments, return the new position */
static size_t skip_blank(const char *s, size_t i, size_t len)
{
    while (i < len)
    {
        if (isspace((unsigned char)s[i])) { i++; continue; }
        if (s[i] == '/' && i + 1 < len && s[i + 1] == '/')
        {
            while (i < len && s[i] != '\n') i++;
            continue;
        }
        if (s[i] == '/' && i + 1 < len && s[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(s[i] == '*' && s[i + 1] == '/')) i++;
            i += 2;
            continue;
        }
        break;
    }
    return i;
}

enum { STMT_OTHER = 0, STMT_IFACE_IN, STMT_IFACE_OUT };

typedef struct {
    int     kind;
    bool    is_struct;      /* leading token is `struct` */
    size_t  qual_end;       /* offset (from the statement start) after the in/out token */
} StmtInfo;

/* classify the top-level statement beginning at s[pos] (only leading tokens are read) */
static void classify_statement(const char *s, size_t pos, size_t len, StmtInfo *info)
{
    size_t i = pos;
    bool iface = false;

    memset(info, 0, sizeof(*info));

    while (i < len)
    {
        i = skip_blank(s, i, len);
        if (i >= len) break;

        if (is_ident_char(s[i]) && !isdigit((unsigned char)s[i]))
        {
            char id[64];
            size_t n = read_ident(s + i, id, sizeof(id));

            if (!strcmp(id, "in") || !strcmp(id, "out"))
            {
                iface = true;
                info->kind = !strcmp(id, "in") ? STMT_IFACE_IN : STMT_IFACE_OUT;
                info->qual_end = i + n - pos;
                i += n;
                continue;
            }
            if (!iface && is_interface_qualifier(id))
            {
                i += n;
                if (!strcmp(id, "layout"))
                {
                    while (i < len && s[i] != '(') i++;
                    if (i < len)
                    {
                        int d = 0;
                        do {
                            if (s[i] == '(') d++;
                            else if (s[i] == ')') d--;
                            i++;
                        } while (i < len && d > 0);
                    }
                }
                continue;
            }
            if (!iface && !strcmp(id, "struct")) info->is_struct = true;
            break;
        }
        break;
    }
}

/* end (one past) of the top-level statement beginning at pos: the ';' that follows any
 * balanced braces */
static size_t statement_end(const char *s, size_t pos, size_t len)
{
    size_t i = pos;
    int brace = 0;

    while (i < len)
    {
        char c = s[i];
        if (c == '/' && i + 1 < len && s[i + 1] == '/')
        {
            while (i < len && s[i] != '\n') i++;
            continue;
        }
        if (c == '/' && i + 1 < len && s[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(s[i] == '*' && s[i + 1] == '/')) i++;
            i += 2;
            continue;
        }
        if (c == '{') brace++;
        else if (c == '}') brace--;
        else if (c == ';' && brace == 0) return i + 1;
        i++;
    }
    return len;
}

/* append the identifier (renamed when it is one of the reserved names) */
static void emit_ident(SB *out, const char *id, bool *used_gl_in)
{
    const char *r = rename_ident(id);
    sb_put(out, "%s", r ? r : id);
    if (used_gl_in && r && !strcmp(id, "gl_in")) *used_gl_in = true;
}

/* copy a run of source text applying the identifier renames; stops (without consuming) at
 * one of stop_chars or at EOF.  Braces inside the copied run are counted into *depth. */
static void copy_renamed(SB *out, const char *s, size_t *pos, size_t len, const char *stop_chars,
                         int *brace_delta, bool *used_gl_in)
{
    while (*pos < len)
    {
        char c = s[*pos];

        if (stop_chars && strchr(stop_chars, c)) return;

        if (c == '/' && *pos + 1 < len && s[*pos + 1] == '/')
        {
            while (*pos < len && s[*pos] != '\n') { sb_put(out, "%c", s[*pos]); (*pos)++; }
            continue;
        }
        if (c == '/' && *pos + 1 < len && s[*pos + 1] == '*')
        {
            sb_put(out, "/*");
            *pos += 2;
            while (*pos + 1 < len && !(s[*pos] == '*' && s[*pos + 1] == '/')) { sb_put(out, "%c", s[*pos]); (*pos)++; }
            if (*pos + 1 < len) { sb_put(out, "*/"); *pos += 2; }
            continue;
        }

        if (is_ident_char(c) && !isdigit((unsigned char)c))
        {
            char id[128];
            size_t n = read_ident(s + *pos, id, sizeof(id));
            emit_ident(out, id, used_gl_in);
            *pos += n;
            continue;
        }

        if (c == '{' && brace_delta) (*brace_delta)++;
        if (c == '}' && brace_delta) (*brace_delta)--;
        sb_put(out, "%c", c);
        (*pos)++;
    }
}

/* emit the rewritten form of an interface declaration [pos, end) */
/* strip interface-only qualifiers from interface-block member declarations: a layout(...)
 * group and the interpolation/storage keywords are legal on an interface member but not on
 * the plain struct member the rewritten shader declares (locations are tracked separately
 * by the reflection, so nothing is lost) */
/* strip interface-only member qualifiers; `storage_too` also drops in/out — needed when
 * the members end up inside a plain struct (GLSL forbids storage qualifiers on struct
 * members), while re-declared-as-ordinary-declaration paths keep them */
static char *strip_member_quals2(const char *s, size_t n, bool storage_too)
{
    static const char *drop[] = { "flat", "smooth", "noperspective", "centroid", "sample",
                                  "invariant", "precise", NULL };
    static const char *storage[] = { "in", "out", NULL };
    SB out;
    size_t i = 0;

    memset(&out, 0, sizeof(out));
    while (i < n)
    {
        if (s[i] == '/' && i + 1 < n && s[i + 1] == '/')
        {
            while (i < n && s[i] != '\n') sb_put(&out, "%c", s[i++]);
            continue;
        }
        if (s[i] == '/' && i + 1 < n && s[i + 1] == '*')
        {
            sb_put(&out, "/*");
            i += 2;
            while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) sb_put(&out, "%c", s[i++]);
            if (i + 1 < n) { sb_put(&out, "*/"); i += 2; }
            continue;
        }

        size_t wl = 0;
        while (i + wl < n && (isalnum((unsigned char)s[i + wl]) || s[i + wl] == '_')) wl++;
        if (wl && i + wl < n && (isspace((unsigned char)s[i + wl]) || s[i + wl] == '('))
        {
            char w[32];
            int layout = wl == 6 && !strncmp(s + i, "layout", 6);
            int kw = 0;
            if ((size_t)wl < sizeof(w))
            {
                memcpy(w, s + i, wl);
                w[wl] = 0;
                for (int k = 0; drop[k]; k++)
                    if (!strcmp(w, drop[k])) kw = 1;
                if (storage_too)
                    for (int k = 0; storage[k]; k++)
                        if (!strcmp(w, storage[k])) kw = 1;
            }
            if (layout || kw)
            {
                i += wl;
                if (layout)
                {
                    while (i < n && isspace((unsigned char)s[i])) i++;
                    if (i < n && s[i] == '(')
                    {
                        int depth = 0;
                        while (i < n)
                        {
                            if (s[i] == '(') depth++;
                            else if (s[i] == ')') { depth--; if (!depth) { i++; break; } }
                            i++;
                        }
                    }
                }
                continue;
            }
        }
        sb_put(&out, "%c", s[i++]);
    }
    return out.s;
}

static char *strip_member_quals(const char *s, size_t n)
{
    return strip_member_quals2(s, n, false);
}

/* rename the reserved builtins of a hoisted global declaration: the source's
 * `out float gl_CullDistance[8];' must become `float mgl_gs_cull[8];' — glslang
 * reserves gl_* identifiers outside an interface redeclaration, and the
 * emulation's own globals (mgl_gs_cull etc.) are what the kernel reads/writes
 * (KHR-GL46.cull_distance.functional's pass-through GS declares cull outputs). */
/* Is this hoisted global a redeclaration of a gl_PerVertex member
 * (`out float gl_CullDistance[8];')?  The kernel's own mgl_gs_clip/mgl_gs_cull
 * globals replace those members, so the declaration itself must be dropped —
 * renaming it would redeclare the kernel's arrays (and keeping gl_* would trip
 * glslang's reserved-identifier rule). */
static bool geom_is_member_redecl(const char *body)
{
    static const char *names[2] = { "gl_ClipDistance", "gl_CullDistance" };
    const char *p = body;

    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    /* interpolation qualifiers */
    for (;;)
    {
        const char *q = p;
        size_t n = 0;
        while (is_ident_char((unsigned char)q[n])) n++;
        if ((n == 4 && !strncmp(q, "flat", 4)) ||
            (n == 13 && !strncmp(q, "noperspective", 13)) ||
            (n == 8 && !strncmp(q, "centroid", 8)) ||
            (n == 6 && !strncmp(q, "sample", 6)))
        { p = q + n; while (*p == ' ' || *p == '\t') p++; continue; }
        break;
    }
    /* the type */
    {
        size_t n = 0;
        while (is_ident_char((unsigned char)p[n])) n++;
        if (!n) return false;
        p += n;
    }
    while (*p == ' ' || *p == '\t') p++;
    for (int k = 0; k < 2; k++)
    {
        size_t n = strlen(names[k]);
        if (!strncmp(p, names[k], n) &&
            !is_ident_char((unsigned char)p[n]))
        {
            const char *r = p + n;
            while (*r == ' ' || *r == '\t') r++;
            return *r == '[' || *r == ';';
        }
    }
    return false;
}

static char *geom_rename_decl_body(const char *body)
{
    SB o;
    size_t p = 0;
    int bd = 0;
    bool gi = false;
    memset(&o, 0, sizeof(o));
    copy_renamed(&o, body, &p, strlen(body), NULL, &bd, &gi);
    return o.s;
}

static void emit_interface(SB *out, const char *s, size_t pos, size_t end, StmtInfo *info,
                           unsigned in_verts, char *gl_in_name, size_t gl_in_cap, bool *input_iface)
{
    char *raw = xmalloc(end - pos + 1);
    size_t n = end - pos;

    memcpy(raw, s + pos, n);
    raw[n] = 0;

    /* everything up to and including in/out (plus its qualifiers) is dropped */
    const char *body = raw + info->qual_end;

    /* 420pack allows any qualifier order, so interface qualifiers may also trail the
     * in/out token ("highp in layout(location=0) smooth invariant vec4 v;"): drop those
     * too — the kernel declares plain variables */
    for (;;)
    {
        while (*body && isspace((unsigned char)*body)) body++;
        char qid[24];
        size_t qn = read_ident(body, qid, sizeof qid);
        if (qn && is_interface_qualifier(qid))
        {
            body += qn;
            if (!strcmp(qid, "layout"))
            {
                while (*body && isspace((unsigned char)*body)) body++;
                if (*body == '(')
                {
                    int qd = 0;
                    while (*body)
                    {
                        if (*body == '(') qd++;
                        else if (*body == ')') { qd--; if (!qd) { body++; break; } }
                        body++;
                    }
                }
            }
            continue;
        }
        break;
    }

    /* bare `layout(...) in;` — nothing to declare */
    {
        const char *p = body;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == ';')
        {
            free(raw);
            return;
        }
    }

    /* gl_PerVertex block: drop the declaration, remember the instance name */
    if (strstr(body, "gl_PerVertex"))
    {
        const char *close = strrchr(body, '}');
        if (close && close[1])
        {
            char id[64];
            if (read_ident(close + 1, id, sizeof(id)) && gl_in_name && !gl_in_name[0])
                snprintf(gl_in_name, gl_in_cap, "%s", id);
        }
        free(raw);
        return;
    }

    /* an interface block without an instance name puts its members into the enclosing scope:
     * emit them as ordinary declarations (with the block's array dimension folded into them
     * for an input), since a plain struct declaration would not bring the names into scope */
    {
        const char *close = strrchr(body, '}');
        if (close)
        {
            const char *after = close + 1;
            while (*after && isspace((unsigned char)*after)) after++;
            if (!*after || *after == ';')
            {
                const char *open = strchr(body, '{');
                if (open && open < close)
                {
                    SB decl;
                    /* members of an anonymous block may carry interface qualifiers
                     * (layout(location=...), xfb_offset, flat, ...) that are illegal on the
                     * ordinary declarations emitted here */
                    char *members = strip_member_quals(open + 1, (size_t)(close - open - 1));
                    memset(&decl, 0, sizeof(decl));
                    for (const char *p = members ? members : (open + 1); *p; )
                    {
                        if (info->kind == STMT_IFACE_IN && p[0] == '[' && p[1] == ']')
                        {
                            sb_put(&decl, "[%u]", in_verts);
                            p += 2;
                            continue;
                        }
                        sb_put(&decl, "%c", *p);
                        p++;
                    }
                    if (input_iface && info->kind == STMT_IFACE_IN) *input_iface = true;
                    /* the member text already carries its terminating ';' */
                    sb_put(out, "%s", decl.s ? decl.s : "");
                    free(decl.s);
                    free(members);
                    free(raw);
                    return;
                }
            }
        }
    }

    /* drop the qualifier of a variable declaration; size input arrays */
    if (info->kind == STMT_IFACE_IN)
    {
        if (input_iface) *input_iface = true;
        SB decl;
        const char *brace = strchr(body, '{');
        memset(&decl, 0, sizeof(decl));
        if (brace)
        {
            /* a named interface block keeps its struct: GLSL needs the "struct" keyword when
             * it is no longer an interface ("Blk { ... } v[3];" does not parse) */
            char bname[64] = "";
            const char *close = strrchr(body, '}');
            const char *bp = brace, *be;
            while (bp > body && isspace((unsigned char)bp[-1])) bp--;
            be = bp;
            while (be > body && (isalnum((unsigned char)be[-1]) || be[-1] == '_')) be--;
            if (bp - be > 0 && (size_t)(bp - be) < sizeof(bname))
            {
                memcpy(bname, be, bp - be);
                bname[bp - be] = 0;
            }
            if (close && bname[0] && strcmp(bname, "gl_PerVertex"))
            {
                char *members = strip_member_quals2(brace + 1, (size_t)(close - brace - 1), true);
                sb_put(&decl, "struct %s { %s }; %s ", bname, members ? members : "", bname);
                free(members);
                body = close + 1;
            }
        }
        char *rb = geom_rename_decl_body(body);
        if (rb) body = rb;
        if (geom_is_member_redecl(body))
        {
            free(rb);
            free(raw);
            return;             /* the kernel's own globals replace the member */
        }
        for (const char *p = body; *p; )
        {
            if (p[0] == '[' && p[1] == ']')
            {
                sb_put(&decl, "[%u]", in_verts);
                p += 2;
                continue;
            }
            sb_put(&decl, "%c", *p);
            p++;
        }
        sb_put(out, "%s", decl.s ? decl.s : body);
        free(decl.s);
        free(rb);
    }
    else
    {
        SB decl;
        const char *brace = strchr(body, '{');
        memset(&decl, 0, sizeof(decl));

        if (brace)
        {
            /* a named output interface block ("out B { ... } b;") becomes a plain struct
             * plus instance: the shader body refers to the members as b.c, which a plain
             * global of that struct type keeps working; interface-only member qualifiers
             * (layout(location/xfb_offset/stream), flat, ...) are dropped here — the
             * reflection keeps the real values.  GLSL matches the block to the next
             * stage's input block by BLOCK name, so the type name is preserved too. */
            char bname[64] = "";
            const char *close = strrchr(body, '}');
            const char *bp = brace, *be;
            while (bp > body && isspace((unsigned char)bp[-1])) bp--;
            be = bp;
            while (be > body && (isalnum((unsigned char)be[-1]) || be[-1] == '_')) be--;
            if (bp - be > 0 && (size_t)(bp - be) < sizeof(bname))
            {
                memcpy(bname, be, bp - be);
                bname[bp - be] = 0;
            }
            if (close && bname[0] && strcmp(bname, "gl_PerVertex"))
            {
                char *members = strip_member_quals2(brace + 1, (size_t)(close - brace - 1), true);
                sb_put(&decl, "struct %s { %s }; %s ", bname, members ? members : "", bname);
                free(members);
                body = close + 1;
            }
        }
        /* a plain variable declaration may still carry interpolation qualifiers after the
         * stripped in/out ("layout(location = 0) out flat uvec3 v;" leaves "flat"): on the
         * ordinary global emitted here they are illegal-ish, and glslang marks the variable
         * so that SPIRV-Cross's helper threading skips it - the generated kernel then
         * references an undeclared identifier (texture_cube_map_array texture_size_*_sh) */
        if (!brace || decl.s == NULL)
        {
            if (!geom_is_member_redecl(body))
            {
                char *rb = geom_rename_decl_body(body);
                char *stripped = strip_member_quals(rb ? rb : body, strlen(rb ? rb : body));
                sb_put(out, "%s", stripped ? stripped : (rb ? rb : body));
                free(stripped);
                free(rb);
            }
            free(raw);
            return;
        }
        for (const char *p = body; *p; p++) sb_put(&decl, "%c", *p);
        sb_put(out, "%s", decl.s);
        free(decl.s);
    }

    free(raw);
}

/* rewrite a geometry shader into compute-shader source.  Returns the transformed text with
 * the `#version` line removed, or NULL on failure.  gl_in_name receives the instance name a
 * gl_PerVertex block declaration used when it is not "gl_in". */
static char *geom_transform_source(const char *src, unsigned in_verts, char *gl_in_name, size_t gl_in_cap,
                                   bool *input_iface, bool *output_iface)
{
    SB out;
    size_t pos = 0, len = strlen(src);
    int depth = 0;          /* >0 inside a function body */

    memset(&out, 0, sizeof(out));
    if (gl_in_name) gl_in_name[0] = 0;
    if (input_iface) *input_iface = false;
    if (output_iface) *output_iface = false;

    while (pos < len)
    {
        if (depth > 0)
        {
            /* copy the body verbatim (with renames); track braces */
            if (src[pos] == '}')
            {
                depth--;
                sb_put(&out, "}");
                pos++;
                continue;
            }
            if (src[pos] == '{')
            {
                depth++;
                sb_put(&out, "{");
                pos++;
                continue;
            }
            copy_renamed(&out, src, &pos, len, "{}", NULL, NULL);
            continue;
        }

        /* preprocessor directives are copied verbatim to the end of the (logical) line */
        if (src[pos] == '#')
        {
            while (pos < len)
            {
                char c = src[pos];
                sb_put(&out, "%c", c);
                pos++;
                if (c == '\n')
                {
                    /* line continuation */
                    size_t q = pos;
                    while (q > 1 && (src[q - 2] == ' ' || src[q - 2] == '\t' || src[q - 2] == '\r')) q--;
                    if (q >= 2 && src[q - 2] == '\\') continue;
                    break;
                }
            }
            continue;
        }

        /* top level: comments and blank text pass through */
        {
            size_t b = skip_blank(src, pos, len);
            while (pos < b)
            {
                /* copy a comment in place: copy_renamed() has no stop characters here and
                 * would run on to the end of the file, copying every declaration after the
                 * comment verbatim (the CTS limits shaders put a comment between the
                 * preamble and their declarations) */
                if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '/')
                {
                    while (pos < len && src[pos] != '\n') { sb_put(&out, "%c", src[pos]); pos++; }
                }
                else if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '*')
                {
                    sb_put(&out, "/*");
                    pos += 2;
                    while (pos + 1 < len && !(src[pos] == '*' && src[pos + 1] == '/'))
                    { sb_put(&out, "%c", src[pos]); pos++; }
                    if (pos + 1 < len) { sb_put(&out, "*/"); pos += 2; }
                }
                else { sb_put(&out, "%c", src[pos]); pos++; }
            }
            if (pos >= len) break;
            /* a preprocessor directive that follows the skipped blanks is handled by the
             * '#' branch above (it must not fall through to the statement copier, which
             * would swallow the following declaration: "#define A (0)\nflat in int x;")
             */
            if (src[pos] == '#') continue;
            /* likewise a comment: skip_blank does not cross it, and the statement copier
             * would run from the comment to the next ';' — swallowing any "#define" and
             * the declarations that follow it (the CTS limits shaders) */
            if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '/')
            {
                while (pos < len && src[pos] != '\n') { sb_put(&out, "%c", src[pos]); pos++; }
                continue;
            }
            if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '*')
            {
                sb_put(&out, "/*");
                pos += 2;
                while (pos + 1 < len && !(src[pos] == '*' && src[pos + 1] == '/'))
                { sb_put(&out, "%c", src[pos]); pos++; }
                if (pos + 1 < len) { sb_put(&out, "*/"); pos += 2; }
                continue;
            }
        }

        StmtInfo info;
        classify_statement(src, pos, len, &info);
        if (getenv("MGL_GEOM_TRACE"))
        {
            char dbg[80];
            size_t dn = len - pos < 60 ? len - pos : 60;
            memcpy(dbg, src + pos, dn); dbg[dn] = 0;
            fprintf(stderr, "MGL geom trace kind=%d qe=%u: [%s]\n", info.kind, (unsigned)info.qual_end, dbg);
        }

        if (info.kind != STMT_OTHER)
        {
            size_t end = statement_end(src, pos, len);
            emit_interface(&out, src, pos, end, &info, in_verts, gl_in_name, gl_in_cap, input_iface);
            if (info.kind == STMT_IFACE_OUT && output_iface) *output_iface = true;
            pos = end;
            continue;
        }

        /* plain statement (or a function signature): copy until ';' (counting statement
         * braces) or until the '{' that opens a function body */
        int brace = 0;
        while (pos < len)
        {
            char c = src[pos];
            if (c == '/' && pos + 1 < len && (src[pos + 1] == '/' || src[pos + 1] == '*'))
            {
                copy_renamed(&out, src, &pos, len, NULL, NULL, NULL);
                continue;
            }
            if (c == '{')
            {
                if (info.is_struct || brace > 0) { brace++; sb_put(&out, "{"); pos++; continue; }
                depth = 1;
                sb_put(&out, "{");
                pos++;
                break;
            }
            if (c == '}')
            {
                if (brace > 0) brace--;
                sb_put(&out, "}");
                pos++;
                continue;
            }
            if (c == ';' && brace == 0)
            {
                sb_put(&out, ";");
                pos++;
                break;
            }
            if (is_ident_char(c) && !isdigit((unsigned char)c))
            {
                char id[128];
                size_t n = read_ident(src + pos, id, sizeof(id));
                emit_ident(&out, id, NULL);
                pos += n;
                continue;
            }
            sb_put(&out, "%c", c);
            pos++;
        }
    }

    if (out.oom)
    {
        free(out.s);
        return NULL;
    }
    return sb_take(&out);
}

/* ------------------------------------------------------------------ reflection */

/* geom r103: the GL base type of a leaf member type (mirrors the member loop's
 * resolution, incl. software-lowered fp64 members whose GL view is GL_DOUBLE) */
static GLenum geom_type_base(SpvModule *m, SpvType *mt)
{
    return mt->kind == SPVT_FLOAT ? (mt->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
           mt->kind == SPVT_INT ? (mt->is_signed ? GL_INT : GL_UNSIGNED_INT) :
           mt->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, mt);
}

/* geom r103: GL 4.6 13.2.2 transform-feedback packing of a structure captured as a
 * whole (`layout(xfb_offset = N) out TestStruct goku;`): every member starts
 * aligned to the SIZE OF ITS SCALAR BASE TYPE only, matrix components and array
 * elements pack contiguously (no std430 column/vec3 padding), and an inner
 * structure's span is aligned to the largest scalar its members use.  (std430 was
 * wrong: the CTS packs `struct { float a; mat2 b; }` as a@0, b@4 — KHR-Single-GL46
 * enhanced_layouts.xfb_struct_explicit_location_*.) */
static GLuint geom_plain_scalar_bytes(GLenum base)
{
    return base_words(base) * 4u;
}

/* largest scalar size used anywhere under struct `sid` (drives nested-struct
 * alignment); 0 if the tree holds a shape we cannot flatten */
static GLuint geom_struct_max_scalar(SpvModule *m, uint32_t sid)
{
    SpvType *st = spv_find_type(m, sid);
    if (!st || st->kind != SPVT_STRUCT) return 0;
    GLuint mx = 0;
    for (int mi = 0; mi < st->n_members; mi++)
    {
        SpvType *mt = spv_find_type(m, st->members[mi]);
        while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
        if (mt && mt->kind == SPVT_ARRAY) mt = spv_find_type(m, mt->elem);
        while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
        if (!mt) return 0;
        if (mt->kind == SPVT_STRUCT)
        {
            GLuint inner = geom_struct_max_scalar(m, mt->id);
            if (!inner) return 0;
            if (inner > mx) mx = inner;
        }
        else if (mt->kind == SPVT_VEC || mt->kind == SPVT_MAT ||
                 mt->kind == SPVT_FLOAT || mt->kind == SPVT_INT || mt->kind == SPVT_BOOL)
        {
            GLuint sb = geom_plain_scalar_bytes(geom_type_base(m, mt));
            if (sb > mx) mx = sb;
        }
        else return 0;
    }
    return mx;
}

/* geom r103: assign the packed xfb offsets to the leaf entries collected for the
 * struct tree rooted at `sid` (the member loop and geom_plain_flatten emit leaves
 * depth-first in declaration order).  Returns the offset just past this struct's
 * packed span, aligned to its largest scalar. */
static unsigned geom_plain_pack(SpvModule *m, uint32_t sid, unsigned off, int *idx,
                                MGLGeomVar *vars, int n)
{
    SpvType *st = spv_find_type(m, sid);
    GLuint mx = geom_struct_max_scalar(m, sid);
    if (!st || !mx) return off;
    for (int mi = 0; mi < st->n_members; mi++)
    {
        SpvType *mt = spv_find_type(m, st->members[mi]);
        while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
        if (mt && mt->kind == SPVT_ARRAY) mt = spv_find_type(m, mt->elem);
        while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
        if (mt && mt->kind == SPVT_STRUCT)
        {
            GLuint inner_max = geom_struct_max_scalar(m, mt->id);
            if (!inner_max) return off;
            unsigned a = (off + inner_max - 1) & ~(inner_max - 1);
            unsigned end = geom_plain_pack(m, mt->id, a, idx, vars, n);
            off = (end + inner_max - 1) & ~(inner_max - 1);
        }
        else if (*idx < n)
        {
            MGLGeomVar *mv = &vars[*idx];
            GLuint sb = geom_plain_scalar_bytes(mv->base);
            unsigned a = (off + sb - 1) & ~(sb - 1);
            unsigned sz = mv->words * 4u;
            if (mv->xfb_offset == 0xffffffffu) mv->xfb_offset = a;
            off = a + sz;
            (*idx)++;
        }
    }
    return (off + mx - 1) & ~(mx - 1);
}

/* geom r103: flatten a NESTED plain-struct member of a whole-captured plain struct
 * output into leaf capture entries with dotted member paths ("goku.inner.a"),
 * depth-first in declaration order — the order geom_plain_pack assigns offsets
 * in.  Array-of-struct members are not flattened (the caller reports them). */
static bool geom_plain_flatten(SpvModule *m, SpvVar *v, uint32_t sid, const char *path,
                               GLuint *next_loc, MGLGeomVar *vars, int *n, char **log)
{
    SpvType *st = spv_find_type(m, sid);
    uint32_t tmp_stream;
    const char *vname = spv_name(m, v->id);
    if (!st || st->kind != SPVT_STRUCT)
    {
        mgl_append_log(log, "ERROR: geometry: unsupported nested member type under %s\n", vname);
        return false;
    }
    for (int mi = 0; mi < st->n_members && *n < MGL_GEOM_MAX_VARYINGS; mi++)
    {
        const char *mn = spv_member_name(m, sid, mi);
        uint32_t mid = st->members[mi];
        SpvType *mt = spv_find_type(m, mid);
        GLuint melems = 1, mrows, mcols;
        GLenum mbase;

        if (!mn) continue;
        while (mt && mt->kind == SPVT_PTR) { mt = spv_find_type(m, mt->elem); mid = mt->id; }
        bool isarr = false;             /* r103: this member carries its own [] dimension */
        if (mt && mt->kind == SPVT_ARRAY)
        {
            isarr = true;
            melems = mt->n;
            mt = spv_find_type(m, mt->elem);
            while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
        }
        if (!mt || melems == 0 || mt->kind == SPVT_ARRAY)
        {
            mgl_append_log(log, "ERROR: geometry: unsupported nested member array under %s\n", vname);
            return false;
        }
        if (mt->kind == SPVT_STRUCT)
        {
            char sub[128];
            snprintf(sub, sizeof sub, "%s%s%s", path, *path ? "." : "", mn);
            if (melems > 1)
            {
                mgl_append_log(log, "ERROR: geometry: array-of-structure member %s.%s is not supported\n",
                               vname, sub);
                return false;
            }
            if (!geom_plain_flatten(m, v, mt->id, sub, next_loc, vars, n, log)) return false;
            continue;
        }
        if (mt->kind != SPVT_VEC && mt->kind != SPVT_MAT &&
            mt->kind != SPVT_FLOAT && mt->kind != SPVT_INT && mt->kind != SPVT_BOOL)
        {
            mgl_append_log(log, "ERROR: geometry: unsupported interface block member type for %s.%s%s\n",
                           vname, path, mn);
            return false;
        }

        MGLGeomVar *out = &vars[(*n)++];
        memset(out, 0, sizeof(*out));
        snprintf(out->name, sizeof(out->name), "%s", vname);
        snprintf(out->member, sizeof(out->member), "%s%s%s", path, *path ? "." : "", mn);
        out->plain_struct = true;
        out->arr = isarr;   /* r103: keep [i] indexing for 1-element array members */
        out->elems = melems;
        out->xfb_buffer = -1;
        out->xfb_offset = 0xffffffffu;
        out->xfb_stride = 0xffffffffu;
        out->location = *next_loc;
        mbase = geom_type_base(m, mt);
        mrows = mt->kind == SPVT_MAT ? spv_find_type(m, mt->elem)->n : mt->kind == SPVT_VEC ? mt->n : 1;
        mcols = mt->kind == SPVT_MAT ? mt->n : 1;
        *next_loc = out->location + melems * mcols * (mbase == GL_DOUBLE ? (mrows + 1) / 2 : 1u);
        out->base = mbase;
        out->rows = mrows;
        out->cols = mcols;
        out->words = melems * mcols * mrows * base_words(mbase);
        out->flat = spv_deco(m, v->id, SpvDecorationFlat, NULL);
        out->noperspective = spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
        if (spv_deco(m, v->id, SpvDecorationStream, &tmp_stream)) out->stream = tmp_stream;
        if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp_stream)) out->xfb_buffer = (int)tmp_stream;
        if (spv_member_deco(m, sid, mi, SpvDecorationXfbBuffer, &tmp_stream)) out->xfb_buffer = (int)tmp_stream;
        /* NOTE: the variable's own Offset decoration ("the struct starts here") is
         * deliberately NOT copied onto leaves — geom_plain_pack walks it; an explicit
         * MEMBER offset (rare) still wins. */
        if (spv_member_deco(m, sid, mi, SpvDecorationOffset, &tmp_stream)) out->xfb_offset = tmp_stream;
    }
    return true;
}

static bool geom_reflect_stage(SpvModule *m, bool inputs, MGLGeomVar *vars, int *count, char **log)
{
    int n = 0;
    uint32_t want = inputs ? SpvStorageClassInput : SpvStorageClassOutput;

    for (int i = 0; i < m->n_vars && n < MGL_GEOM_MAX_VARYINGS; i++)
    {
        SpvVar *v = &m->vars[i];
        uint32_t location = 0, tmp;
        MGLGeomVar *out;
        SpvType *t;
        uint32_t elems = 1;

        if (v->storage != want) continue;
        if (spv_deco(m, v->id, SpvDecorationBuiltIn, NULL)) continue;   /* builtins are handled separately */
        if (!spv_deco(m, v->id, SpvDecorationLocation, &location)) continue;   /* user varyings only */

        t = spv_value_type(m, v->type_id, &elems);
        if (!t) continue;
        if (inputs)
        {
            /* the outermost array of a geometry input is the primitive's vertices */
            uint32_t dummy = 1;
            t = spv_value_type(m, v->type_id, &dummy);
            elems = 1;
        }
        /* an interface block is a struct-typed interface variable: its members consume
         * consecutive locations starting at the block's location (glslang decorates the
         * variable, not the members).  With an instance name the members are reached through
         * the instance ("v[i].a"); without one the members are the arrays ("a[i]") and the
         * block's outer array dimension lives on them. */
        if (t->kind == SPVT_STRUCT)
        {
            const char *inst = spv_name(m, v->id);
            bool anon = !inst || !inst[0];
            /* a plain struct-typed interface variable ("out StructType v;") is collected the
             * same way, but it is NOT a block: the members are named through the VARIABLE
             * ("v.m"), never through the type name, and the generated code keeps the struct
             * declaration, so the store/clear expressions must go through "v.m" even when
             * the variable and the member happen to share a name (dEQP PIQ
             * transform_feedback_varying.*.struct.*) */
            bool plain = !spv_deco(m, t->id, SpvDecorationBlock, NULL) &&
                         !spv_deco(m, t->id, SpvDecorationBufferBlock, NULL);
            GLuint next_loc = location;
            int m0 = 0;   /* compat r96c: index of the first member var below */

            /* geom r88c: the introspection flattening (r87/r88a, introspect_spirv.c
             * flatten_io_matrix_arrays_dir) re-types a stage-in/out ARRAY OF MATRICES
             * into a struct of N members named <var>_0 .. <var>_(N-1), all of ONE
             * matrix type.  GL still sees ONE array resource ("name", elems = N) —
             * reassemble it here so the whole-array tf capture resolves
             * (dEQP-GL45-ES31 transform_feedback_varying.type.separable_geometry.
             * whole_array.mat3x4: without this the per-member entries made xfb
             * resolve "target" as a block instance and the capture matched nothing).
             * Structural detection: every member has the same type id and is named
             * <varname>_<member index>. */
            do {
                SpvType *mt;
                char want[96];
                bool ok = true;
                int mi;

                if (anon || !inst || t->n_members < 1) break;
                mt = spv_find_type(m, t->members[0]);
                while (mt && mt->kind == SPVT_PTR) mt = spv_find_type(m, mt->elem);
                if (!mt || (mt->kind != SPVT_MAT && mt->kind != SPVT_VEC &&
                            mt->kind != SPVT_FLOAT && mt->kind != SPVT_INT && mt->kind != SPVT_BOOL))
                    break;
                for (mi = 0; mi < t->n_members; mi++)
                {
                    const char *mn = spv_member_name(m, t->id, mi);
                    if (t->members[mi] != t->members[0]) { ok = false; break; }
                    snprintf(want, sizeof want, "%s_%d", inst, mi);
                    if (!mn || strcmp(mn, want)) { ok = false; break; }
                }
                if (!ok) break;

                out = &vars[n++];
                memset(out, 0, sizeof(*out));
                snprintf(out->name, sizeof(out->name), "%s", inst);
                out->elems = (GLuint)t->n_members;
                out->arr = true;   /* the source declared an ARRAY of matrices: every
                                    * store/load expression indexes it (see r103) */
                out->xfb_buffer = -1;
                out->xfb_offset = 0xffffffffu;
                out->location = location;
                out->base = mt->kind == SPVT_FLOAT ? (mt->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
                            mt->kind == SPVT_INT ? (mt->is_signed ? GL_INT : GL_UNSIGNED_INT) :
                            mt->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, mt);
                out->rows = mt->kind == SPVT_MAT ? spv_find_type(m, mt->elem)->n :
                            mt->kind == SPVT_VEC ? mt->n : 1;
                out->cols = mt->kind == SPVT_MAT ? mt->n : 1;
                out->words = out->elems * out->cols * out->rows * base_words(out->base);
                out->flat = spv_deco(m, v->id, SpvDecorationFlat, NULL);
                out->noperspective = spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
                if (spv_deco(m, v->id, SpvDecorationStream, &tmp)) out->stream = tmp;
                if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
                if (spv_deco(m, v->id, SpvDecorationOffset, &tmp)) out->xfb_offset = tmp;
                if (spv_deco(m, v->id, SpvDecorationXfbStride, &tmp)) out->xfb_stride = tmp;
                goto flat_merged;    /* one canonical array entry replaces the member list */
            } while (0);
            m0 = n;   /* compat r96c: the member vars this loop emits (whole-struct xfb) */
            for (int mi = 0; mi < t->n_members && n < MGL_GEOM_MAX_VARYINGS; mi++)
            {
                const char *mn = spv_member_name(m, t->id, mi);
                uint32_t mloc = 0, mid = t->members[mi];
                SpvType *mt;
                GLenum mbase;
                GLuint mrows, mcols, consumed = 1;

                if (!mn) continue;

                /* the vertex dimension of an ANONYMOUS block input sits on the member itself
                 * (the variable is the bare struct), so one array layer is stripped there.
                 * With an instance name the variable's outer array was that dimension
                 * already (stripped above), and the member's array is its own ("vs_gs_out[16]"):
                 * stripping it here would collapse the member to one element. */
                mt = spv_find_type(m, mid);
                while (mt && mt->kind == SPVT_PTR) { mt = spv_find_type(m, mt->elem); mid = mt->id; }
                if (inputs && anon && mt && mt->kind == SPVT_ARRAY)
                {
                    mid = mt->elem;
                    mt = spv_find_type(m, mid);
                }
                /* an array member ("out B { vec4 c[2]; } b;"): the element type is stored and
                 * the length becomes elems — one layer only, the rewritten shader declares it
                 * as a struct member with its own [] dimension, which the generators unroll */
                GLuint melems = 1;
                bool was_array = false;   /* r103: the member carries its own [] dimension */
                if (mt && mt->kind == SPVT_ARRAY)
                {
                    was_array = true;
                    melems = mt->n;
                    mid = mt->elem;
                    mt = spv_find_type(m, mid);
                }
                if (mt && mt->kind == SPVT_STRUCT)
                {
                    /* geom r103: a nested structure member of a whole-captured PLAIN
                     * structure output flattens into dotted leaf entries
                     * ("goku.inner_struct_a.a") depth-first in declaration order —
                     * geom_plain_pack then assigns the packed 13.2.2 offsets.
                     * Interface BLOCKS with nested members stay unsupported. */
                    if (!inputs && plain)
                    {
                        if (melems > 1)
                        {
                            mgl_append_log(log, "ERROR: geometry: array-of-structure member %s.%s is not supported\n",
                                           anon ? "?" : inst, mn);
                            return false;
                        }
                        if (!geom_plain_flatten(m, v, mt->id, mn, &next_loc, vars, &n, log)) return false;
                        continue;
                    }
                    mgl_append_log(log, "ERROR: geometry: unsupported interface block member type for %s.%s\n",
                                   anon ? "?" : inst, mn);
                    return false;
                }
                if (!mt || melems == 0 || (mt->kind == SPVT_ARRAY) ||
                    (mt->kind != SPVT_VEC && mt->kind != SPVT_MAT &&
                     mt->kind != SPVT_FLOAT && mt->kind != SPVT_INT && mt->kind != SPVT_BOOL))
                {
                    mgl_append_log(log, "ERROR: geometry: unsupported interface block member type for %s.%s\n",
                                   anon ? "?" : inst, mn);
                    return false;
                }

                mbase = mt->kind == SPVT_FLOAT ? (mt->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
                        mt->kind == SPVT_INT ? (mt->is_signed ? GL_INT : GL_UNSIGNED_INT) :
                        mt->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, mt);
                /* rows = components per column, cols = number of columns.  An
                 * OpTypeMatrix' own n is the COLUMN COUNT and its element is the
                 * column vector — non-square matrices were transposed here (rows
                 * and cols coincide for square ones, which hid it). */
                mrows = mt->kind == SPVT_MAT ? spv_find_type(m, mt->elem)->n :
                        mt->kind == SPVT_VEC ? mt->n : 1;
                mcols = mt->kind == SPVT_MAT ? mt->n : 1;
                /* GLSL 4.60 4.4.1: a double element (or matrix column) consumes
                 * ceil(rows/2) 128-bit locations, an array one set per element */
                consumed = melems * mcols * (mbase == GL_DOUBLE ? (mrows + 1) / 2 : 1u);

                out = &vars[n++];
                memset(out, 0, sizeof(*out));
                snprintf(out->name, sizeof(out->name), "%s", anon ? mn : inst);
                snprintf(out->member, sizeof(out->member), "%s", anon ? "" : mn);
                out->arr = was_array;   /* r103: keep [i] indexing for 1-element array members */
                if (plain) out->plain_struct = true;
                /* the block NAME (a non-anonymous interface block's type name) names the
                 * member through the API: glTransformFeedbackVaryings writes "Block.member" */
                if (!anon && !plain)
                {
                    const char *bn = spv_name(m, t->id);
                    if (bn && bn[0]) snprintf(out->block, sizeof(out->block), "%s", bn);
                }
                out->elems = melems;
                out->xfb_buffer = -1;
                out->xfb_offset = 0xffffffffu;
                out->xfb_stride = 0xffffffffu;
                out->location = spv_member_deco(m, t->id, mi, SpvDecorationLocation, &mloc) ?
                                mloc : next_loc;
                next_loc = out->location + consumed;
                out->base = mbase;
                out->rows = mrows;
                out->cols = mcols;
                out->words = melems * mcols * mrows * base_words(mbase);
                out->flat = spv_member_deco(m, t->id, mi, SpvDecorationFlat, NULL) ||
                            spv_deco(m, v->id, SpvDecorationFlat, NULL);
                out->noperspective = spv_member_deco(m, t->id, mi, SpvDecorationNoPerspective, NULL) ||
                                     spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
                if (spv_member_deco(m, t->id, mi, SpvDecorationStream, &tmp)) out->stream = tmp;
                /* shader-declared transform feedback capture (xfb_buffer/xfb_offset): the
                 * member inherits the block's buffer, the offset is its own */
                if (spv_member_deco(m, t->id, mi, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
                else if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
                if (spv_member_deco(m, t->id, mi, SpvDecorationOffset, &tmp)) out->xfb_offset = tmp;
                if (spv_member_deco(m, t->id, mi, SpvDecorationXfbStride, &tmp) ||
                    spv_deco(m, v->id, SpvDecorationXfbStride, &tmp)) out->xfb_stride = tmp;
            }
            /* compat r96c / geom r103: a PLAIN structure output captured as a whole
             * (`layout(xfb_offset = N) out Goku goku;` — ARB_enhanced_layouts)
             * captures every member in declaration order.  SPIR-V decorates only
             * the variable, so the members' xfb offsets are derived here.  r96c
             * used the std430 layout; GL 4.6 13.2.2 actually packs captured
             * structures with SCALAR alignment only (matrices/array elements
             * contiguous, inner structs aligned to their largest scalar) — which
             * is what geom_plain_pack walks, nested struct members included
             * (flattened to dotted leaves above). */
            if (plain && spv_deco(m, v->id, SpvDecorationOffset, &tmp))
            {
                int pidx = m0;
                geom_plain_pack(m, t->id, tmp, &pidx, vars, n);
                /* the members inherit the buffer of the variable */
                if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp))
                    for (int mi2 = m0; mi2 < n; mi2++)
                        if (vars[mi2].xfb_buffer < 0) vars[mi2].xfb_buffer = (int)tmp;
            }
            /* geom r102: a named block (or plain struct) INSTANCE with an array dimension
             * ("out Goku { vec4 gohan; } goku[2];") — the entries collected above are
             * ELEMENT 0.  GL 4.6 11.1.2.5 (ARB_enhanced_layouts): a captured block follows
             * the std430 rules unless members carry explicit offsets, and an array of
             * blocks is captured element-consecutively, each element aligned to the
             * block's alignment.  So element e's members sit e * <element size> past
             * element 0's, consume a fresh set of locations, and the kernel's
             * store/clear expressions must carry the element index ("goku[e].gohan") —
             * the element-0 entry alone would name "goku.gohan", which is not
             * well-formed for an array instance at all (KHR-Single-GL46
             * enhanced_layouts.xfb_block_stride_geometry). */
            {
                SpvType *raw = spv_find_type(m, v->type_id);
                while (raw && raw->kind == SPVT_PTR) raw = spv_find_type(m, raw->elem);
                bool inst_is_array = raw && raw->kind == SPVT_ARRAY &&
                                     spv_find_type(m, raw->elem) == t;
                if (!inputs && !anon && inst_is_array && n > m0)
                {
                    int nm = n - m0;
                    GLuint e, i2;
                    unsigned stride = 0, maxa = 1, walk = 0, expl_end = 0;
                    /* per-member std430 size/alignment (the walk the plain-struct offset
                     * derivation above uses), and the end an explicit member offset implies */
                    for (i2 = 0; i2 < (GLuint)nm; i2++)
                    {
                        MGLGeomVar *mv = &vars[m0 + i2];
                        unsigned sb = base_words(mv->base) * 4u;
                        unsigned ma = (mv->rows == 3 ? 4u : mv->rows) * sb;
                        unsigned ms = mv->rows * sb;
                        uint32_t moff = 0;
                        if (spv_member_deco(m, t->id, (int)i2, SpvDecorationOffset, &moff))
                            expl_end = (moff + ms > expl_end) ? moff + ms : expl_end;
                        if (mv->cols > 1)
                            ms = mv->cols * ((ms + ma - 1) & ~(ma - 1));
                        if (mv->elems > 1)
                            ms = mv->elems * ((ms + ma - 1) & ~(ma - 1));
                        walk = (walk + ma - 1) & ~(ma - 1);
                        walk += ms;
                        if (ma > maxa) maxa = ma;
                    }
                    stride = (expl_end > walk ? expl_end : walk);
                    stride = (stride + maxa - 1) & ~(maxa - 1);
                    /* geom r103: a PLAIN struct instance array packs per 13.2.2 (not
                     * std430) — the element stride is the packed span the offset walk
                     * above assigned, aligned to the struct's largest scalar. */
                    if (plain)
                    {
                        int scratch = m0;
                        unsigned span = geom_plain_pack(m, t->id, 0, &scratch, vars, n);
                        GLuint ms = geom_struct_max_scalar(m, t->id);
                        if (ms < 1) ms = 1;
                        unsigned pstride = (span + ms - 1) & ~(ms - 1);
                        stride = pstride;
                    }
                    /* the location span one element's members consume (explicit member
                     * locations are honored in order; the span is what followed them) */
                    GLuint span = (next_loc > vars[m0].location) ? next_loc - vars[m0].location : 0;
                    for (i2 = 0; i2 < (GLuint)nm; i2++)
                    {
                        vars[m0 + i2].block_array = true;
                        vars[m0 + i2].index = 0;
                        vars[m0 + i2].block_elems = elems ? elems : 1;
                    }
                    for (e = 1; e < (elems ? elems : 1) && n < MGL_GEOM_MAX_VARYINGS; e++)
                    {
                        for (i2 = 0; i2 < (GLuint)nm && n < MGL_GEOM_MAX_VARYINGS; i2++)
                        {
                            MGLGeomVar *mv = &vars[m0 + i2], *nv = &vars[n];
                            *nv = *mv;          /* element 0's entry: type, flatness, stride */
                            nv->index = e;
                            nv->location = mv->location + span * e;
                            if (mv->xfb_offset != 0xffffffffu)
                                nv->xfb_offset = mv->xfb_offset + stride * e;
                            n++;
                        }
                    }
                }
            }
            continue;
        flat_merged: ;
            continue;
        }

        if (t->kind != SPVT_VEC && t->kind != SPVT_MAT &&
            t->kind != SPVT_FLOAT && t->kind != SPVT_INT && t->kind != SPVT_BOOL)
        {
            mgl_append_log(log, "ERROR: geometry: unsupported interface type for %s\n",
                           spv_name(m, v->id) ? spv_name(m, v->id) : "?");
            return false;
        }

        out = &vars[n++];
        memset(out, 0, sizeof(*out));
        snprintf(out->name, sizeof(out->name), "%s", spv_name(m, v->id) ? spv_name(m, v->id) : "");
        out->location = location;
        out->xfb_buffer = -1;
        out->xfb_offset = 0xffffffffu;
        out->xfb_stride = 0xffffffffu;
        out->base = t->kind == SPVT_FLOAT ? (t->bits == 64 ? GL_DOUBLE : GL_FLOAT) :
                    t->kind == SPVT_INT ? (t->is_signed ? GL_INT : GL_UNSIGNED_INT) :
                    t->kind == SPVT_BOOL ? GL_BOOL : spv_scalar_base(m, t);
        /* rows = components per column, cols = number of columns (see the block
         * member path above: OpTypeMatrix' n is the column count) */
        out->rows = t->kind == SPVT_MAT ? spv_find_type(m, t->elem)->n :
                    t->kind == SPVT_VEC ? t->n : 1;
        out->cols = t->kind == SPVT_MAT ? t->n : 1;
        out->elems = elems;
        /* the source declared an ARRAY (even of just one element — `out mat2 x[1];`):
         * the rewritten shader keeps that declaration, so every store/load expression
         * must index it ("x[0][c].r"), exactly like a 1-element array member (r103).
         * The input side carries the primitive's vertex dimension as its outermost
         * array layer, which the reflection strips above — test one level below it. */
        {
            SpvType *raw = spv_find_type(m, v->type_id);

            while (raw && raw->kind == SPVT_PTR) raw = spv_find_type(m, raw->elem);
            if (inputs && raw && raw->kind == SPVT_ARRAY)
                raw = spv_find_type(m, raw->elem);
            out->arr = raw && raw->kind == SPVT_ARRAY;
        }
        out->words = elems * out->cols * out->rows * base_words(out->base);
        out->flat = spv_deco(m, v->id, SpvDecorationFlat, NULL);
        out->noperspective = spv_deco(m, v->id, SpvDecorationNoPerspective, NULL);
        out->centroid = spv_deco(m, v->id, SpvDecorationCentroid, NULL);
        out->sample = spv_deco(m, v->id, SpvDecorationSample, NULL);
        if (spv_deco(m, v->id, SpvDecorationStream, &tmp)) out->stream = tmp;
        if (spv_deco(m, v->id, SpvDecorationXfbBuffer, &tmp)) out->xfb_buffer = (int)tmp;
        if (spv_deco(m, v->id, SpvDecorationOffset, &tmp)) out->xfb_offset = tmp;
        if (spv_deco(m, v->id, SpvDecorationXfbStride, &tmp)) out->xfb_stride = tmp;
        (void)location;
    }

    *count = n;
    return true;
}

static int geom_find_location(const MGLGeomVar *vars, int count, GLuint location)
{
    for (int i = 0; i < count; i++)
        if (vars[i].location == location) return i;
    return -1;
}

/* match an interface variable by NAME and type.  Programs linked separately (a pipeline's
 * stages) assign automatic varying locations by their own declaration order, so the same
 * interface can come out with different locations in different programs; conformant
 * implementations interface-match such stages by name when the locations were automatic.
 * The capture expression follows the name+type match, falling back from the location
 * match geom_derive_exprs tries first. */
static int geom_find_name(const MGLGeomVar *vars, int count, const char *name)
{
    if (!name || !name[0]) return -1;
    for (int i = 0; i < count; i++)
        if (vars[i].name[0] && !strcmp(vars[i].name, name)) return i;
    return -1;
}

static bool geom_var_types_match(const MGLGeomVar *a, const MGLGeomVar *b)
{
    return a->rows == b->rows && a->cols == b->cols && a->base == b->base && a->elems == b->elems;
}

/* ------------------------------------------------------------------ code generation */

/* remove the "#version ..." line of a source so a new one can be prepended */
static char *strip_version(const char *src)
{
    const char *v = strstr(src, "#version");
    if (!v) return xstrdup(src);
    const char *e = strchr(v, '\n');
    if (!e) return xstrdup(v && v > src ? src : "");
    size_t head = (size_t)(v - src);
    char *out = xmalloc(strlen(src) + 1);
    memcpy(out, src, head);
    strcpy(out + head, e + 1);
    return out;
}

/* dump helper (MGL_GEOM_DUMP=<dir>): writes the generated shaders for inspection */
/* the trace / behaviour switches are read once (this file is on the draw path) */

/* GLSL `coherent` / `volatile` qualifiers on SSBO blocks become a `volatile`
 * address-space qualifier in SPIRV-Cross's MSL parameter declarations, which MSL
 * rejects (`volatile device Block& x`) and which strips member matrices of their
 * operator[]; `restrict` comes out as the C99 keyword, which MSL reserves but only
 * accepts spelled `__restrict`.  Neither qualifier has a Metal equivalent —
 * cross-invocation ordering is provided by MGL's encoder serialization — so this pass
 * rewrites both in device/constant address-space declarations (called from program.c's
 * stage MSL finalization).  Returns a malloc'd replacement, or NULL when nothing
 * changed. */
char *mgl_geom_strip_msl_volatile(const char *msl)
{
    static const struct { const char *bad, *good; } pats[] = {
        { "volatile device ",  "device " },
        { "volatile constant ", "constant " },
        { "device volatile ",  "device " },
        { "constant volatile ", "constant " },
        /* GLSL `restrict` becomes the C99 keyword, which MSL reserves but only
         * accepts in the compiler's double-underscore spelling */
        { "& restrict ", "& __restrict " },
        { "* restrict ", "* __restrict " },
    };

    if (!msl) return NULL;

    bool changed = false;
    for (size_t p = 0; p < sizeof pats / sizeof pats[0] && !changed; p++)
        changed = strstr(msl, pats[p].bad) != NULL;
    if (!changed) return NULL;

    /* the replacements do not all shrink (`& restrict ` -> `& __restrict ` grows by
     * 2 per occurrence), so the result is rebuilt into a growing buffer: rewriting
     * a strdup in place overflowed the heap whenever an MSL carried `restrict` */
    char *cur = xstrdup(msl);
    if (!cur) return NULL;

    for (size_t p = 0; p < sizeof pats / sizeof pats[0]; p++)
    {
        if (!strstr(cur, pats[p].bad))
            continue;
        SB b = { 0, 0, 0, false };
        size_t bl = strlen(pats[p].bad), gl = strlen(pats[p].good);
        char *q = cur;
        for (char *m; (m = strstr(q, pats[p].bad)) != NULL; q = m + bl)
        {
            sb_reserve(&b, (size_t)(m - q) + gl);
            if (b.oom) break;
            memcpy(b.s + b.len, q, (size_t)(m - q));
            b.len += (size_t)(m - q);
            memcpy(b.s + b.len, pats[p].good, gl);
            b.len += gl;
            b.s[b.len] = 0;
        }
        if (!b.oom)
        {
            sb_reserve(&b, strlen(q));
            if (!b.oom)
            {
                memcpy(b.s + b.len, q, strlen(q) + 1);
                b.len += strlen(q);
            }
        }
        char *next = sb_take(&b);
        free(cur);
        if (!next)
            return NULL;
        cur = next;
    }
    return cur;
}

/* Does the MSL entry function return void?  Metal requires rasterization to be
 * disabled for void vertex functions and enabled otherwise (an Apple-GPU pipeline
 * validation rule both ways), and a GL vertex-only program can legitimately be
 * either: a shader writing only SSBOs/images has no rasterization outputs and
 * SPIRV-Cross emits `vertex void f(...)`, while one that still assigns
 * gl_Position gets `vertex F_out f(...)`.  Finds `<entry>(` with an identifier
 * boundary and inspects the return-type token in front of it. */
bool mgl_msl_vertex_returns_void(const char *msl, const char *entry)
{
    size_t elen;

    if (!msl || !entry || !(elen = strlen(entry)))
        return false;

    for (const char *p = msl; (p = strstr(p, entry)) != NULL; p += 1)
    {
        if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
            continue;
        const char *after = p + elen;
        if (*after != '(')
            continue;

        /* the token ending right before the entry name is the return type */
        const char *e = p;
        while (e > msl && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\n'))
            e--;
        const char *s = e;
        while (s > msl && (isalnum((unsigned char)s[-1]) || s[-1] == '_'))
            s--;
        return (e - s == 4) && strncmp(s, "void", 4) == 0;
    }
    return false;
}

/* ---------------------------------------------------------------- atomic counters
 * SPIRV-Cross's MSL backend does not declare GL atomic-counter (atomic_uint) variables
 * as entry-point arguments: the emitted MSL references `g_counter` without any
 * declaration (`atomic_load_explicit((thread atomic_uint*)&g_counter, ...)`) and every
 * stage using an atomic counter fails to compile.  mgl_msl_inject_atomic_counters()
 * adds the declaration MGL itself needs:
 *   * one `volatile device atomic_uint* <name> [[buffer(N)]]` argument is appended to
 *     the entry function's parameter list.  N is the ordinal the renderer's buffer map
 *     binds the atomic-counter group at: uniform blocks, the default uniform block and
 *     SSBOs are emitted (and bound) first — those are exactly the `[[buffer(i)]]`
 *     arguments with an index below 24 (24..30 are pinned by the tessellation /
 *     transform-feedback / vertex-attribute domains).
 *   * every use `(thread atomic_uint*)&<name>` is rewritten to a `volatile device`
 * pointer into the argument buffer so counters placed by the GLSL `offset`
 * layout qualifier address the right word (the GL atomic counter buffer is bound
 * at base offset 0): a scalar becomes `(volatile device atomic_uint*)&n[w]`
 * (w = offset/4), an array element `(thread atomic_uint*)&n[expr]` becomes
 * `(volatile device atomic_uint*)&n[(w)+(expr)]` — the bracketed index is kept
 * inside the address expression, not left outside the cast
 * (KHR-GL46.compute_shader.pipeline-compute-chain indexes ac[i]).
 * Returns a malloc'd replacement MSL, or NULL on failure (entry not found / out of
 * buffer slots).  *indices_out (if not NULL) receives the chosen Metal buffer index
 * per variable.  `max_metal_index` is the first Metal buffer index the stage must
 * not use: 24 for the render stages (24..30 are pinned by the tessellation /
 * transform-feedback / geometry emulation and SPIRV-Cross's buffer-size slot), 30
 * for the compute stage, whose only reserved slot is the buffer-size slot 30. */

/* Linearise one or more consecutive `[expr]` groups of an atomic-counter reference
 * (`&n[i0]`, `&n[i0][i1]`, ...) into the flat device-buffer index of the ONE pointer
 * argument the injection declares.  GL lays atomic-counter arrays out as consecutive
 * 4-byte words: element (i0..ik) of `n[D0][D1]..[Dk]` sits at word
 * ((i0)*D1*D2.. + (i1)*D2.. + ..), plus the declaration's `offset` base `w`.  `after`
 * points at the first '['; the index text is appended to `idx` (which the caller has
 * already opened with `[`), and `*end_out` receives the position one past the last
 * ']'.  Missing trailing groups (a whole inner array addressed) keep the stride
 * product of the un-indexed dimensions, which is the SPIR-V addressing.  Returns
 * false when the brackets are unbalanced, more groups than dimensions are present,
 * or the flattened text would overflow — the caller then gives up on this set. */
static bool ac_linear_index(const MGLAtomicCounterVar *v, const char *after, unsigned w,
                            char *idx, size_t idx_sz, const char **end_out)
{
    size_t o = strlen(idx);
    int ng = 0;
    const char *g = after;
    const char *ends[5] = { NULL };

    while (*g == '[' && ng < 5)
    {
        const char *e = g + 1;
        int depth = 1;
        while (*e)
        {
            if (*e == '[') depth++;
            else if (*e == ']' && --depth == 0) { e++; break; }
            e++;
        }
        if (depth != 0 || (size_t)(e - g) > 200)
            return false;
        ends[ng++] = e;
        g = e;
    }
    if (!ng || ng > 5)
        return false;
    if (ng > (v ? v->ndims : 0))
        return false;   /* deeper indexing than the declaration has dimensions */

    /* [(w)+((i0))*S0+((i1))*S1+...]: Sk is the element product behind dimension k */
    if (w)
        o += (size_t)snprintf(idx + o, idx_sz - o, "(%u)+(", w);
    else
        idx[o++] = '(';
    for (int k = 0; k < ng; k++)
    {
        uint32_t stride = 1;
        for (int d = k + 1; d < (v ? v->ndims : 0) && d < 4; d++) stride *= v->dims[d];
        const char *body = (k ? ends[k - 1] : after) + 1;
        size_t el = (size_t)(ends[k] - (k ? ends[k - 1] : after)) - 2;   /* inside [ ] */
        if (k)
            idx[o++] = '+';
        o += (size_t)snprintf(idx + o, idx_sz - o, "(%.*s)", (int)el, body);
        if (stride != 1)
            o += (size_t)snprintf(idx + o, idx_sz - o, "*%u", stride);
        if (o + 16 >= idx_sz)
            return false;
    }
    snprintf(idx + o, idx_sz - o, ")");
    *end_out = ends[ng - 1];
    return true;
}

static char *inject_atomic_counters_core(const char *msl, const char *entry,
                                         const MGLAtomicCounterVar *vars, int nvars,
                                         int max_metal_index, int forced_base, int *indices_out);
static char *msl_msl_counters_to_helper_params(const char *msl, const char *entry,
                                               const MGLAtomicCounterVar *vars, int nvars);

char *mgl_msl_inject_atomic_counters(const char *msl, const char *entry,
                                     const MGLAtomicCounterVar *vars, int nvars,
                                     int max_metal_index, int forced_base, int *indices_out)
{
    /* counters referenced inside non-entry functions first become trailing
     * parameters of those functions and of their callers (see
     * msl_msl_counters_to_helper_params) — the entry arguments below then feed
     * the chain from the top */
    char *xformed = msl_msl_counters_to_helper_params(msl, entry, vars, nvars);
    if (!xformed)
        return NULL;

    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL geom: atomic-counter injection, pre-MSL:\n%s\n", msl);
    {
        char *out = inject_atomic_counters_core(xformed, entry, vars, nvars,
                                                max_metal_index, forced_base, indices_out);
        if (out && getenv("MGL_GEOM_DEBUG"))
            fprintf(stderr, "MGL geom: atomic-counter injection, post-MSL:\n%s\n", out);
        free(xformed);
        return out;
    }
}

/* Apple GPUs do not make a write through a `texture2d<..., access::read_write>`
 * (GL image2D with read+write access) visible to a later read of the same texel
 * in the same shader invocation: the load hits the texture cache.  Metal's
 * `tex.fence()` is the required synchronization.  SPIRV-Cross does not emit it,
 * so this pass inserts `<base>.fence();` after every write statement on a
 * read_write texture (KHR-GL46.compute_shader.resource-image stores to a
 * coherent image2D and immediately loads it back; the draw then compares the
 * two images).  Returns a malloc'd copy, or NULL on allocation failure. */
char *mgl_msl_insert_image_fences(const char *msl)
{
    if (!msl)
        return NULL;

    /* 1. names of read_write textures (also inside array<...> declarations) */
    char names[64][256];
    int nnames = 0;
    for (const char *p = msl; (p = strstr(p, "access::read_write")) != NULL; )
    {
        const char *q = p + strlen("access::read_write");
        while (*q == ' ' || *q == '\t' || *q == '>') q++;   /* `> name` / `, N> name` */
        const char *e = q;
        while (*e && (isalnum((unsigned char)*e) || *e == '_')) e++;
        if (e > q && nnames < 64 && (size_t)(e - q) < sizeof names[0])
        {
            memcpy(names[nnames], q, (size_t)(e - q));
            names[nnames][e - q] = 0;
            for (int i = 0; i < nnames; i++)
                if (!strcmp(names[i], names[nnames])) goto next;
            nnames++;
        }
    next:
        p = q;
    }
    if (!nnames)
        return strdup(msl);

    /* 2. every `NAME[expr].write(...)` / `NAME.write(...)` statement */
    struct { size_t at; char *text; } edits[512];
    int nedits = 0;
    size_t len = strlen(msl);

    for (int i = 0; i < nnames; i++)
    {
        size_t nl = strlen(names[i]);
        for (const char *p = msl; (p = strstr(p, names[i])) != NULL; p++)
        {
            if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
                continue;                       /* longer identifier */
            const char *q = p + nl;
            /* optional [index] groups (array-of-texture arguments) */
            while (*q == '[')
            {
                int d = 0;
                const char *r = q;
                while (*r)
                {
                    if (*r == '[') d++;
                    else if (*r == ']' && --d == 0) { r++; break; }
                    r++;
                }
                if (d) { q = NULL; break; }
                q = r;
            }
            if (!q || strncmp(q, ".write(", 7) != 0)
                continue;

            /* statement end: the matching ')' of write(, then ';' */
            int d = 0;
            const char *r = q + 6;              /* at '(' */
            while (*r)
            {
                if (*r == '(') d++;
                else if (*r == ')') { if (--d == 0) { r++; break; } }
                r++;
            }
            if (d) continue;
            while (*r == ' ' || *r == '\t' || *r == '\n' || *r == '\r') r++;
            if (*r != ';')
                continue;                       /* not a simple statement */

            char *ins = malloc((size_t)(q - p) + 32);
            if (!ins) continue;
            memcpy(ins, " ", 1);
            memcpy(ins + 1, p, (size_t)(q - p));       /* NAME or NAME[index] */
            strcpy(ins + 1 + (q - p), ".fence();");
            if (nedits < 512)
            {
                edits[nedits].at = (size_t)(r + 1 - msl);
                edits[nedits].text = ins;
                nedits++;
            }
            else
                free(ins);
            p = r;
        }
    }
    if (!nedits)
        return strdup(msl);

    /* 3. apply in ascending order (offsets refer to the original string) */
    for (int a = 0; a < nedits; a++)
        for (int b = a + 1; b < nedits; b++)
            if (edits[b].at < edits[a].at)
            {
                size_t ta = edits[a].at; char *tt = edits[a].text;
                edits[a].at = edits[b].at; edits[a].text = edits[b].text;
                edits[b].at = ta; edits[b].text = tt;
            }

    /* the inserts must fit: plain len+1 overflowed the heap (same class as the
     * atomic-counter helper-param insert) */
    size_t ins_len = 0;
    for (int a = 0; a < nedits; a++)
        ins_len += strlen(edits[a].text);
    char *out = malloc(len + 1 + ins_len);
    if (out)
    {
        out[0] = 0;
        size_t src = 0;
        for (int a = 0; a < nedits; a++)
        {
            strncat(out, msl + src, edits[a].at - src);
            strcat(out, edits[a].text);
            src = edits[a].at;
        }
        strcat(out, msl + src);
    }
    for (int a = 0; a < nedits; a++) free(edits[a].text);
    return out ? out : NULL;
}

/* ------------------------------------------------------------------ image atomics
 * SPIRV-Cross emulates GL image atomics (OpImageTexelPointer + OpAtomic*) with a side
 * `device atomic_<T>* <name>_atomic [[buffer(N)]]` argument per storage image, next to
 * the texture argument itself: only OpImageRead/OpImageWrite go through the texture,
 * every atomic operation reads/writes the buffer.  Of the GL image targets it
 * linearizes the coordinate correctly for Dim2D alone (the spvImage2DAtomicCoord
 * macro, 16-texel row pitch) — and for image2DArray it applies that same macro, which
 * silently drops the layer.  1D and buffer images use a scalar subscript (valid), but
 * 3D / 1DArray / 2DArray / Cube / CubeArray end up with an int2/int3 subscript, which
 * is not even compilable MSL.
 *
 * This pass rewrites every `<name>_atomic[<coord>]` subscript of those targets into a
 * call of a generated helper that linearizes the coordinate with the row-major layout
 * the renderer's sidecar buffer uses (geomImageAtomicBuffer in MGLRenderer+Geometry.m);
 * plain image2D / image2DRect keep SPIRV-Cross's own macro (and its pitch), so the
 * renderer sizes their sidecar with the same 16-texel alignment.
 *
 * The `<name>_atomic` argument indices are SPIRV-Cross's automatic buffer ordinals:
 * they share the counter with the stage's own auto-assigned UBO/SSBO arguments, so they
 * cannot collide with those — but explicitly pinned resources do not advance the
 * counter, and the geometry-emulation kernels pin the user's blocks.  There the indices
 * are rebased (forced_base >= 0) onto the free slots past the pinned user range.
 *
 * kind 0 = plain 2D (or 1D/buffer/unknown): the subscript is left as written.
 */
enum { MGL_AI_2D = 0, MGL_AI_3D, MGL_AI_ARR2, MGL_AI_CUBE, MGL_AI_CUBEARR, MGL_AI_MS2D, MGL_AI_MSARR };
/* render-stage sidecar (image-atomic buffer) base slot: [16, 24) — below the
 * emulation pins (24) and clear of the default-uniform block's slot 0 */
#define MGL_GEOM_SIDE_BASE 16
#define MGL_GEOM_SIDE_BASE_STR "16"

/* GL 8.17 texture completeness for SAMPLING: with a mipmap min filter, every level
 * base+1..top must halve the previous level's dimensions; with NEAREST/LINEAR only
 * the base level must exist.  MGL validated level EXISTENCE when the texture was
 * created (createMTLTextureFromGLTexture:), but a level can be REPLACED with the
 * wrong size afterwards (glTexImage2D on level 1 with 1x1 under a 32x32 level 0),
 * which makes the texture incomplete — sampling must then return (0,0,0,1) instead
 * of the data (KHR-GL46.texture_gather.incomplete-texture / -last-comp).
 * Used from bindTexturesToCurrentRenderEncoder (MGLRenderer.m); declared in
 * mgl_ext_geom.h. */
/* completeness under one explicit parameter set (texture's own, or a bound
 * sampler object's — see mgl_texture_incomplete_for_unit) */
static bool mgl_texture_incomplete_p(const Texture *tex, const TextureParameter *sp);

bool mgl_texture_incomplete_for_sampling(const Texture *tex)
{
    return mgl_texture_incomplete_p(tex, &tex->params);
}

/* GL 4.6 9.2.3: a sampler object bound to the unit OVERRIDES every sampler
 * state of the texture (filters, base/max level) — completeness must use the
 * EFFECTIVE state or a sampler with MIN_FILTER=NEAREST over a single-level
 * texture with the default NEAREST_MIPMAP_LINEAR is wrongly swapped for the
 * (0,0,0,1) fallback (headless test_compat_bordemu int_border_array_3d). */
bool mgl_texture_incomplete_for_unit(GLMContext ctx, const Texture *tex, GLuint unit)
{
    if (ctx && unit < TEXTURE_UNITS)
    {
        Sampler *smp = STATE(texture_samplers[unit]);
        if (smp)
            return mgl_texture_incomplete_p(tex, &smp->params);
    }
    return mgl_texture_incomplete_p(tex, &tex->params);
}

/* completeness under one explicit parameter set (texture's own, or a bound
 * sampler object's — see mgl_texture_incomplete_for_unit) */
static bool mgl_texture_incomplete_p(const Texture *tex, const TextureParameter *sp)
{
    GLint base, top, i, faces = 1;

    if (!tex || !tex->faces || tex->num_levels < 1)
        return true;

    /* CUBE_MAP (only!) stores one face entry per cube face; a CUBE_MAP_ARRAY
     * keeps the whole array in faces[0] (mgl_texdata_num_faces) — iterating
     * faces[1..5] there read zeroed entries and rejected every complete cube
     * ARRAY chain (plain-gather-*-cube-array / texture_cube_map_array.sampling
     * sampled the 1x1 black fallback). */
    if (tex->target == GL_TEXTURE_CUBE_MAP)
        faces = 6;

    base = (GLint)sp->base_level;
    if (base < 0 || (GLuint)base >= tex->num_levels)
        return true;                      /* base level has no storage */
    if ((GLint)sp->max_level < base)
        return true;                      /* GL 8.17: max level below base */

    /* GL 4.6 8.17.2: with a mip min filter the chain must reach
     * q = min(p, TEXTURE_MAX_LEVEL), p = base + floor(log2(max(w,h,d))) of the
     * LEVEL BASE image — stopping at the highest DEFINED level (the old rule)
     * sampled real data from a level-0-only texture under NEAREST_MIPMAP_*,
     * where GL requires the incomplete color (0,0,0,1)
     * (dEQP-GL45-ES31 texture.gather.* incomplete.mipmap_incomplete).
     * Immutable storage cannot define levels beyond its allocation, so the
     * chain requirement stops there (Mesa caps _MaxLevel at NumLevels-1). */
    top = base;
    /* textures that are never mipmap-filtered: MULTISAMPLE / MULTISAMPLE_ARRAY
     * (GL 3.9.1: min/mag filter state is ignored; only texelFetch reaches them)
     * and TEXTURE_BUFFER (texel fetch, no filtering, one level) must not take
     * the mip-chain branch — their default NEAREST_MIPMAP_LINEAR min filter
     * would mark every single-level image incomplete and swap in the
     * (0,0,0,1) fallback (headless test_geom_msimg sampler2DMS readback,
     * test_texswizzle alphaless_attachment, test_texext buffer_texture_*). */
    bool never_mipmapped = tex->target == GL_TEXTURE_2D_MULTISAMPLE ||
                           tex->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
                           tex->target == GL_TEXTURE_BUFFER;
    if (!never_mipmapped &&
        sp->min_filter != GL_NEAREST && sp->min_filter != GL_LINEAR)
    {
        GLuint w = tex->faces[0].levels[base].width;
        GLuint h = tex->faces[0].levels[base].height;
        GLuint d = tex->faces[0].levels[base].depth;
        /* the mip chain halves w, and h/d for every target but 1D/1D_ARRAY (h)
         * and non-3D (d); RECTANGLE has no mips at all */
        GLuint m = w;
        if (tex->target != GL_TEXTURE_1D && tex->target != GL_TEXTURE_1D_ARRAY && h > m) m = h;
        if (tex->target == GL_TEXTURE_3D && d > m) m = d;
        while (m > 1) { m >>= 1; top++; }
        if ((GLint)sp->max_level < top)
            top = (GLint)sp->max_level;
        if (tex->immutable_storage && top > (GLint)tex->num_levels - 1)
            top = (GLint)tex->num_levels - 1;
        if (top >= TD_MAX_LEVELS)
            top = TD_MAX_LEVELS - 1;
        /* a RECTANGLE texture has a single level: any mipmap min filter makes
         * it incomplete (GL 8.17.2) */
        if (tex->target == GL_TEXTURE_RECTANGLE)
            return true;
    }

    for (int f = 0; f < faces; f++)
        if (!tex->faces[f].levels[base].complete)
            return true;

    for (i = base + 1; i <= top; i++)
    {
        GLuint w = tex->faces[0].levels[i - 1].width;
        GLuint h = tex->faces[0].levels[i - 1].height;
        GLuint d = tex->faces[0].levels[i - 1].depth;
        GLuint ew = (w > 1) ? (w >> 1) : 1;
        GLuint eh = (h > 1) ? (h >> 1) : 1;
        GLuint ed = (d > 1) ? (d >> 1) : 1;

        if (tex->target == GL_TEXTURE_1D || tex->target == GL_TEXTURE_1D_ARRAY)
            eh = h;
        /* GL 8.17.1: only GL_TEXTURE_3D halves the DEPTH of each level — for
         * 2D_ARRAY / CUBE_MAP_ARRAY the layer count stays CONSTANT across the
         * chain (halving it here rejected every complete array mip chain when
         * the min filter consults mipmaps, so the shared 1x1 black fallback
         * was bound: KHR-GL46.texture_gather *-2darray / *-cube-array read
         * back zeros). */
        if (tex->target != GL_TEXTURE_3D)
            ed = d;
        for (int f = 0; f < faces; f++)
        {
            if (tex->faces[f].levels[i].width != ew ||
                tex->faces[f].levels[i].height != eh ||
                tex->faces[f].levels[i].depth != ed)
                return true;              /* inconsistent chain: incomplete */
        }
    }
    return false;
}

char *mgl_msl_fix_image_atomics(const char *msl, int max_index, int forced_base,
                                MGLGeomTexSlot *slots_out, int *nslots_out, int max_slots,
                                int *dub_out)
{
    struct { char name[128]; int auto_idx, index, kind; int scalar; int is2d; char elem[12]; } imgs[32];
    memset(imgs, 0, sizeof imgs);   /* r63: same latent-uninitialized risk as arrs[] below */
    struct { char elem[96]; unsigned count; char hname[152]; } arrs[32]; /* name-indexed */
    memset(arrs, 0, sizeof arrs);   /* r63: an UNinitialized entry with garbage count and
                                     * an empty hname made the emit probe "(" match any
                                     * parenthesised edit text and splice a garbage helper
                                     * into the MSL (broke the Metal build as soon as the
                                     * stack layout shifted) */
    int nimgs = 0;
    struct { size_t at, len; char *text; } edits[256];
    int nedits = 0;
    static const char *dubnames[2] = { "mgl_DefaultUniforms_", "gl_DefaultUniformBlock" };

    if (!msl) return NULL;
    if (nslots_out) *nslots_out = 0;
    if (dub_out) *dub_out = -1;

    /* 0. SPIRV-Cross's spvImage2DAtomicCoord linearizes with NO bounds check, so an
     * imageAtomic* at a coordinate outside the image (GL discards those; the CTS does
     * them from full-window quads into 64x64 images — advanced-cast, allStages-oneImage)
     * indexes past the sidecar buffer and undefined-writes GPU memory.  Re-point the
     * macro's out-of-range branch at the DUMP word the allocation carries (pitch*h,
     * geomImageAtomicBuffer's +1 word): in-range indices are byte-identical.
     *
     * compat r99: the row pitch is a LITERAL 16 TEXELS.  The macro's own expression uses
     * `spvLinearTextureAlignment / 4`, but that constant is a FUNCTION CONSTANT whose
     * default is 4 (bytes) — never specialized here — so the macro linearized DENSELY
     * (pitch = width) while the sidecar allocation (geomImageAtomicTexels), the seed and
     * the resolve kernels all use the 16-texel padded row.  For any image whose width is
     * not a multiple of 16 (the CTS's 11x11 advanced-cast images) the atomics landed in
     * the first w*h words and the resolve read them back transposed/truncated — rows
     * beyond w*h/16 came back zero (test_compat_imgcast.c, KHR-GL46.es_31_compatibility
     * shader_image_load_store.advanced-cast-fs/-cs). */
    {
        static const char def[] = "#define spvImage2DAtomicCoord(tc, tex)";
        const char *d = strstr(msl, def);
        if (d)
        {
            const char *body = d + sizeof def - 1;
            while (*body == ' ' || *body == '\t') body++;
            const char *e = strchr(body, '\n');       /* a #define ends at the newline */
            if (e && nedits < 256)
            {
                char nb[768];
                snprintf(nb, sizeof nb,
                    "((((tc).x >= 0 && (tc).y >= 0 && (tc).x < (int)(tex).get_width() && "
                    "(tc).y < (int)(tex).get_height()) ? "
                    "((((((tex).get_width() + 16u - 1) & ~(16u - 1))) * (uint)(tc).y) + (uint)(tc).x) : "
                    "(((((tex).get_width() + 16u - 1) & ~(16u - 1))) * (uint)(tex).get_height())))");
                edits[nedits].at = (size_t)(body - msl);
                edits[nedits].len = (size_t)(e - body);
                edits[nedits].text = strdup(nb);
                if (edits[nedits].text) nedits++;
            }
        }
    }

    if (dub_out)
    {
        /* the packed default-uniform block argument: `mgl_DefaultUniforms_1 [[buffer(N)]]`
         * (SPIRV-Cross numbers it after the image-atomic buffers, which shifts it off the
         * renderer's map ordinal — the renderer binds it at the index reported here) */
        for (int k = 0; k < 2; k++)
        {
            const char *q = msl;
            while ((q = strstr(q, dubnames[k])) != NULL)
            {
                const char *b = strstr(q, "[[buffer(");
                if (b && (size_t)(b - q) < 48)
                {
                    *dub_out = atoi(b + 9);
                    break;
                }
                q += strlen(dubnames[k]);
            }
            if (*dub_out >= 0) break;
        }
    }

    /* 1. the `device atomic_<T>* <name>_atomic [[buffer(N)]]` arguments */
    for (const char *p = msl; (p = strstr(p, "_atomic [[buffer(")) != NULL; )
    {
        const char *nm_end = p;
        const char *nm = nm_end;
        while (nm > msl && (isalnum((unsigned char)nm[-1]) || nm[-1] == '_')) nm--;
        int idx = atoi(p + strlen("_atomic [[buffer("));
        if (nm_end > nm && nimgs < 32 && (size_t)(nm_end - nm) < sizeof imgs[0].name)
        {
            int dup = 0;
            for (int i = 0; i < nimgs; i++)
                if (!strncmp(imgs[i].name, nm, (size_t)(nm_end - nm))) dup = 1;
            if (!dup)
            {
                memcpy(imgs[nimgs].name, nm, (size_t)(nm_end - nm));
                imgs[nimgs].name[nm_end - nm] = 0;
                imgs[nimgs].auto_idx = idx;
                imgs[nimgs].kind = MGL_AI_2D;
                imgs[nimgs].scalar = 0;
                imgs[nimgs].is2d = 0;
                imgs[nimgs].elem[0] = 0;
                arrs[nimgs].count = 0;              /* set when an array<> type is found */
                arrs[nimgs].hname[0] = 0;
                /* the element type of the side argument (`device atomic_uint*` etc.) —
                 * the imageLoad/imageStore redirect rebuilds the GLSL gvec4 from it */
                {
                    const char *a = nm;
                    const char *e = NULL;

                    while (a > msl && (a[-1] == ' ' || a[-1] == '*')) a--;
                    e = a;
                    while (e > msl && (isalnum((unsigned char)e[-1]) || e[-1] == '_')) e--;
                    if (!strncmp(e, "atomic_", 7) && (size_t)(a - e - 7) < sizeof imgs[0].elem)
                        memcpy(imgs[nimgs].elem, e + 7, (size_t)(a - e - 7)), imgs[nimgs].elem[a - e - 7] = 0;
                }
                nimgs++;
            }
        }
        p += strlen("_atomic [[buffer(");
    }
    if (!nimgs)
        return strdup(msl);

    /* 2. the texture argument of each name: its MSL type names the target.  Walk back
     * over the parameter list to the previous ',' or '(' at angle-bracket depth zero
     * (texture2d<uint, access::read_write> has a comma inside the template). */
    for (const char *p = msl; (p = strstr(p, "[[texture(")) != NULL; )
    {
        const char *attr = p;
        const char *nm_end = p;
        while (nm_end > msl && (nm_end[-1] == ' ' || nm_end[-1] == '\t')) nm_end--;
        const char *nm = nm_end;
        while (nm > msl && (isalnum((unsigned char)nm[-1]) || nm[-1] == '_')) nm--;
        if (nm_end == nm) { p += 10; continue; }

        int angle = 0;
        const char *t = nm;
        while (t > msl)
        {
            char c = t[-1];
            if (c == '>') angle++;
            else if (c == '<') { if (angle) angle--; }
            else if (!angle && (c == ',' || c == '(')) break;
            t--;
        }
        while (t < nm && (*t == ' ' || *t == '\t')) t++;
        size_t tlen = (size_t)(nm - t);

        for (int i = 0; i < nimgs; i++)
        {
            size_t nl = strlen(imgs[i].name);
            if (nl != (size_t)(nm_end - nm) || strncmp(imgs[i].name, nm, nl)) continue;
            int kind = MGL_AI_2D;
            if (tlen >= 13 && !strncmp(t, "texture2d_ms<", 13)) kind = MGL_AI_MS2D;
            else if (tlen >= 19 && !strncmp(t, "texture2d_ms_array<", 19)) kind = MGL_AI_MSARR;
            else if (tlen >= 10 && !strncmp(t, "texture3d<", 10)) kind = MGL_AI_3D;
            else if (tlen >= 16 && !strncmp(t, "texture2d_array<", 16)) kind = MGL_AI_ARR2;
            else if (tlen >= 18 && !strncmp(t, "texturecube_array<", 18)) kind = MGL_AI_CUBEARR;
            else if (tlen >= 12 && !strncmp(t, "texturecube<", 12)) kind = MGL_AI_CUBE;
            imgs[i].kind = kind;
            /* imageLoad/imageStore redirect support: the scalar-subscript targets
             * (imageBuffer / image1D) and plain image2D (through SPIRV-Cross's own
             * spvImage2DAtomicCoord macro, always defined when a 2D atomic exists) */
            imgs[i].scalar = (kind == MGL_AI_2D) &&
                             ((tlen >= 15 && !strncmp(t, "texture_buffer<", 15)) ||
                              (tlen >= 10 && !strncmp(t, "texture1d<", 10)));
            imgs[i].is2d = (kind == MGL_AI_2D) && (tlen >= 10 && !strncmp(t, "texture2d<", 10));
            /* an ARRAY of images: `array<texture2d<int>, 8> u_image [[texture(0)]]`.
             * SPIRV-Cross keeps ONE _atomic side buffer for the whole array but emits
             * every access as `<name>[<i>]_atomic[...]` — an identifier that does not
             * exist.  Record the element type and length: the rewrite below flattens
             * the access into the single buffer with a per-element padded stride, the
             * same layout geomImageAtomicBufferArray builds. */
            if (tlen > 7 && !strncmp(t, "array<", 6))
            {
                /* `array<texture2d<int>, 8>`: split at the TOP-LEVEL comma (the
                 * element type itself contains <>), count runs to the final `>` */
                const char *inner = t + 6, *split = NULL, *cnt, *e;
                int ang = 1;
                for (e = inner; e < t + tlen; e++)
                {
                    if (*e == '<') ang++;
                    else if (*e == '>') ang--;
                    else if (*e == ',' && ang == 1) { split = e; break; }
                }
                if (split)
                {
                    const char *cend = t + tlen - 1;   /* at the array's closing '>' */
                    for (cnt = split + 1; cnt < cend && (*cnt == ' ' || *cnt == '\t'); cnt++) {}
                    unsigned cv = 0;
                    while (cnt < cend && *cnt >= '0' && *cnt <= '9')
                        cv = cv * 10u + (unsigned)(*cnt++ - '0');
                    if (cv > 0 && cv < 256 && (size_t)(split - inner) < sizeof arrs[i].elem)
                    {
                        bool ms_elem = !strncmp(inner, "texture2d_ms_array<", 19);
                        bool ms2_elem = !strncmp(inner, "texture2d_ms<", 13);
                        snprintf(arrs[i].elem, sizeof arrs[i].elem, "%.*s",
                                 (int)(split - inner), inner);
                        arrs[i].count = cv;
                        /* an ARRAY of multisample images (image2DMS g[N]): the ELEMENT
                         * type names the target, and the flattened sidecar offsets need
                         * the MS word layout (4 words per texel-sample, no row padding),
                         * so the image kind follows the element and the helper gets its
                         * own name (r82) */
                        if (ms_elem) imgs[i].kind = MGL_AI_MSARR;
                        else if (ms2_elem) imgs[i].kind = MGL_AI_MS2D;
                        snprintf(arrs[i].hname, sizeof arrs[i].hname, "%s_%s",
                                 (ms_elem || ms2_elem) ? "spvMglAIArrMS" : "spvMglAIArr",
                                 imgs[i].name);
                    }
                }
            }
        }
        p = attr + 10;
    }

    /* 3. final buffer indices */
    for (int i = 0; i < nimgs; i++)
    {
        imgs[i].index = (forced_base >= 0) ? forced_base + i : imgs[i].auto_idx;
        if (imgs[i].index < 0 || imgs[i].index >= max_index)
            return NULL;            /* the caller reports the failure */
        if (slots_out && nslots_out && *nslots_out < max_slots)
        {
            /* multisample sidecars: flag the slot (MGL_GEOM_MS_SLOT) — the renderer's
             * bind paths mask it off and key the MS seed/flush machinery on it;
             * imgs[i].index itself stays the plain buffer index (used below) */
            int idx = imgs[i].index;
            if (imgs[i].kind == MGL_AI_MS2D || imgs[i].kind == MGL_AI_MSARR)
                idx |= MGL_GEOM_MS_SLOT;
            snprintf(slots_out[*nslots_out].name, sizeof slots_out[0].name, "%s", imgs[i].name);
            slots_out[(*nslots_out)++].msl = idx;
        }
    }

    /* 4. re-base the argument declarations */
    if (forced_base >= 0)
    {
        /* the packed default-uniform block may sit ABOVE a sidecar we just moved
         * (SPIRV-Cross numbered the MS sidecar buffer(0), dub buffer(1)); the
         * per-draw uniform streaming still targets the block's historical slot 0,
         * so renumber the dub DOWN into the vacated sidecar slot — otherwise the
         * block binds once at its new index and every later glUniform is lost
         * (verified: two draws with different uniform values both read draw 1's) */
        if (dub_out && *dub_out > 0)
        {
            int freed = -1;
            for (int i = 0; i < nimgs; i++)
                if (imgs[i].auto_idx >= 0 && (freed < 0 || imgs[i].auto_idx < freed))
                    freed = imgs[i].auto_idx;
            if (freed >= 0 && freed < *dub_out)
            {
                char num[12];
                int nl = snprintf(num, sizeof num, "%d", freed);

                for (int k = 0; k < 2; k++)
                {
                    const char *q = msl;
                    while ((q = strstr(q, dubnames[k])) != NULL)
                    {
                        const char *b = strstr(q, "[[buffer(");
                        if (b && (size_t)(b - q) < 48)
                        {
                            int dl = 0;
                            while (b[9 + dl] >= '0' && b[9 + dl] <= '9') dl++;
                            if (dl > 0 && nedits < 256)
                            {
                                char *rep = malloc((size_t)nl + 1);
                                if (rep) { snprintf(rep, (size_t)nl + 1, "%s", num);
                                           edits[nedits].at = (size_t)(b + 9 - msl);
                                           edits[nedits].len = (size_t)dl;
                                           edits[nedits].text = rep; nedits++; }
                            }
                            q = b + 9;
                        }
                        else q += strlen(dubnames[k]);
                    }
                }
                *dub_out = freed;
            }
        }

        for (int i = 0; i < nimgs; i++)
        {
            if (imgs[i].index == imgs[i].auto_idx) continue;
            char pat[192], *rep;
            snprintf(pat, sizeof pat, "%s_atomic [[buffer(%d)]]", imgs[i].name, imgs[i].auto_idx);
            const char *q = strstr(msl, pat);
            if (!q) continue;
            rep = malloc(192);
            if (!rep) continue;
            snprintf(rep, 192, "%s_atomic [[buffer(%d)]]", imgs[i].name, imgs[i].index);
            if (nedits < 256)
            {
                edits[nedits].at = (size_t)(q - msl);
                edits[nedits].len = strlen(pat);
                edits[nedits].text = rep;
                nedits++;
            }
            else free(rep);
        }
    }

    /* 5. rewrite the subscripts whose coordinate needs linearizing */
    for (int i = 0; i < nimgs; i++)
    {
        char sub[160];
        const char *helper = NULL;

        /* 5a-MS: an ARRAY of MULTISAMPLE images (image2DMS g[N]): the access is
         * `<name>[<i>]_atomic[spvImage2DAtomicCoord(int2(<idx>, 0), <name>[<i>])]`
         * where <idx> is the FOLDED within-element word index the SPIR-V pass
         * computed (4*sample + S*(layer + L*(x+W*y)) + comp).  Flatten into the
         * single array side buffer with the MS element layout:
         * `<name>_atomic[spvMglAIArrMS_<name>(<name>, <i>, <idx>)]` — the helper
         * adds element i's region offset (w*h*S*4 words per element, exactly what
         * geomImageAtomicArrayLayout sized, no row padding: the MS layout is
         * folded) and clamps OOB indices onto the +1 dump word. */
        if (arrs[i].count && (imgs[i].kind == MGL_AI_MS2D || imgs[i].kind == MGL_AI_MSARR) &&
            arrs[i].hname[0])
        {
            snprintf(sub, sizeof sub, "%s[", imgs[i].name);
            for (const char *p = msl; (p = strstr(p, sub)) != NULL; )
            {
                if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
                { p += strlen(sub); continue; }
                if (p - msl >= 1 && !strncmp(p - 1, arrs[i].hname, strlen(arrs[i].hname)))
                { p += strlen(sub); continue; }         /* our own helper's parameter use */

                const char *is = p + strlen(sub);       /* element index start */
                const char *ie = is;                    /* ... end (']') */
                int d = 1;
                while (*ie)
                {
                    if (*ie == '[' || *ie == '(') d++;
                    else if (*ie == ']' || *ie == ')') { if (--d == 0) { ie++; break; } }
                    ie++;
                }
                if (d || ie[-1] != ']' || strncmp(ie, "_atomic[", 8))
                { p += strlen(sub); continue; }
                /* the element index expression must be a plain literal or identifier:
                 * digits, letters, '_' (a run-time array index of a helper call is
                 * legal MSL here; nested brackets are not) */
                {
                    bool plain = (ie - 1 > is);
                    for (const char *q = is; q < ie - 1 && plain; q++)
                        if (!isalnum((unsigned char)*q) && *q != '_') plain = false;
                    if (!plain) { p += strlen(sub); continue; }
                }

                const char *cs = ie + 8;                /* subscript start */
                const char *ce = cs;
                d = 1;
                while (*ce)
                {
                    if (*ce == '[' || *ce == '(') d++;
                    else if (*ce == ']' || *ce == ')') { if (--d == 0) { ce++; break; } }
                    ce++;
                }
                if (d) { p = cs; continue; }
                size_t clen = (size_t)(ce - cs - 1);    /* inside [ ... ] */

                /* `spvImage2DAtomicCoord(int2(<idx>, 0), <name>[<i>])`: strip the
                 * wrapper, then the int2/int3(...) to the folded <idx> */
                const char *w = NULL;
                size_t wlen = 0;
                if (clen > 22 && !strncmp(cs, "spvImage2DAtomicCoord(", 22))
                {
                    const char *seg0 = cs + 22;
                    const char *c1 = NULL;
                    int depth = 1;                      /* inside the macro '(' */
                    const char *r = seg0;
                    while (r < cs + clen)
                    {
                        if (*r == '(' || *r == '[') depth++;
                        else if (*r == ')' || *r == ']') depth--;
                        else if (*r == ',' && depth == 1 && !c1) c1 = r;
                        r++;
                    }
                    if (c1 && (!strncmp(seg0, "int2(", 5) || !strncmp(seg0, "int3(", 5)))
                    {
                        r = seg0 + 5;                   /* inside intN( */
                        const char *ci = NULL;
                        depth = 1;
                        while (r < c1 - 1)
                        {
                            if (*r == '(' || *r == '[') depth++;
                            else if (*r == ')' || *r == ']') depth--;
                            else if (*r == ',' && depth == 1 && !ci) ci = r;
                            r++;
                        }
                        if (ci)
                        {
                            const char *z = ci + 1;
                            while (z < c1 - 1 && (*z == ' ' || *z == '\t')) z++;
                            size_t rest = (size_t)(c1 - 1 - z);
                            if ((rest == 1 && z[0] == '0') ||
                                (rest == 4 && z[0] == '0' && z[1] == ',' &&
                                 (z[2] == ' ' || z[2] == '\t') && z[3] == '0'))
                            { w = seg0 + 5; wlen = (size_t)(ci - w); }
                        }
                    }
                }
                if (w && wlen)
                {
                    size_t ilen = (size_t)(ie - 1 - is);
                    char *rep = malloc(strlen(imgs[i].name) * 2 + strlen(arrs[i].hname) +
                                       ilen + wlen + 32);
                    if (rep)
                    {
                        snprintf(rep, strlen(imgs[i].name) * 2 + strlen(arrs[i].hname) +
                                 ilen + wlen + 32,
                                 "%s_atomic[%s(%s, %.*s, %.*s)]", imgs[i].name, arrs[i].hname,
                                 imgs[i].name, (int)ilen, is, (int)wlen, w);
                        if (nedits < 256)
                        {
                            edits[nedits].at = (size_t)(p - msl);
                            edits[nedits].len = (size_t)(ce - p);
                            edits[nedits].text = rep;
                            nedits++;
                        }
                        else free(rep);
                    }
                }
                p = ce;
            }
            continue;                                   /* MS arrays: no other rewrite */
        }

        /* 5a. an ARRAY image: every access is `<name>[<i>]_atomic[<sub>]` (the element
         * index ends up INSIDE the identifier SPIRV-Cross builds).  Flatten into the
         * single side buffer: `<name>_atomic[spvMglAIArr_<name>(<name>, <i>, <coord>)]`
         * where <coord> is SPIRV-Cross's macro's first argument (a plain 2D coord). */
        if (arrs[i].count && imgs[i].kind == MGL_AI_2D &&
            !strncmp(arrs[i].elem, "texture2d<", 10) && arrs[i].hname[0])
        {
            snprintf(sub, sizeof sub, "%s[", imgs[i].name);
            for (const char *p = msl; (p = strstr(p, sub)) != NULL; )
            {
                if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
                { p += strlen(sub); continue; }
                if (arrs[i].hname[0] && p - msl >= 1 &&
                    !strncmp(p - 1, arrs[i].hname, strlen(arrs[i].hname)))
                { p += strlen(sub); continue; }         /* our own helper's parameter use */

                const char *is = p + strlen(sub);       /* index start */
                const char *ie = is;                    /* ... and end (']') */
                int d = 1;
                while (*ie)
                {
                    if (*ie == '[' || *ie == '(') d++;
                    else if (*ie == ']' || *ie == ')') { if (--d == 0) { ie++; break; } }
                    ie++;
                }
                if (d || ie[-1] != ']' || strncmp(ie, "_atomic[", 8))
                { p += strlen(sub); continue; }

                size_t ilen = (size_t)(ie - 1 - is);    /* the element index expression */
                const char *cs = ie + 8;                /* coordinate start */
                const char *ce = cs;
                d = 1;
                while (*ce)
                {
                    if (*ce == '[' || *ce == '(') d++;
                    else if (*ce == ']' || *ce == ')') { if (--d == 0) { ce++; break; } }
                    ce++;
                }
                if (d) { p = cs; continue; }
                size_t clen = (size_t)(ce - cs - 1);    /* inside [ ... ] */

                /* strip spvImage2DAtomicCoord(<coord>, <tex>): keep <coord> (the last
                 * top-level comma inside the macro splits the two arguments) */
                const char *w = NULL;
                if (clen > 24 && !strncmp(cs, "spvImage2DAtomicCoord(", 22))
                {
                    const char *r = cs + 21, *c2 = NULL;
                    int dd = 0, depth = 0;
                    while (r < cs + clen)
                    {
                        if (*r == '(' || *r == '[') depth++;
                        else if (*r == ')' || *r == ']') depth--;
                        else if (*r == ',' && depth == 1) c2 = r;
                        r++;
                    }
                    (void)dd;
                    if (c2) { w = cs + 22; clen = (size_t)(c2 - cs - 22); }
                }
                if (!w)
                {
                    /* a plain coordinate (no macro): accept int2(...) shaped text */
                    w = cs;
                }

                char *rep = malloc(strlen(arrs[i].hname) + 2 * strlen(imgs[i].name) + ilen + clen + 24);
                if (rep)
                {
                    snprintf(rep, strlen(arrs[i].hname) + 2 * strlen(imgs[i].name) + ilen + clen + 24,
                             "%s_atomic[%s(%s, %.*s, %.*s)]", imgs[i].name, arrs[i].hname,
                             imgs[i].name, (int)ilen, is, (int)clen, w);
                    if (nedits < 256)
                    {
                        edits[nedits].at = (size_t)(p - msl);
                        edits[nedits].len = (size_t)(ce - p);
                        edits[nedits].text = rep;
                        nedits++;
                    }
                    else free(rep);
                }
                p = ce;
            }
            continue;                                   /* arrays use no other rewrite */
        }

        snprintf(sub, sizeof sub, "%s_atomic[", imgs[i].name);

        for (const char *p = msl; (p = strstr(p, sub)) != NULL; )
        {
            if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
            { p += strlen(sub); continue; }

            const char *cs = p + strlen(sub);            /* coordinate start */
            const char *ce = cs;
            size_t span;                                 /* <name>_atomic[<coord>] overall */
            {
                int d = 1;                               /* inside the subscript's '[' */
                while (*ce)
                {
                    if (*ce == '[' || *ce == '(') d++;
                    else if (*ce == ']' || *ce == ')') { if (--d == 0) { ce++; break; } }
                    ce++;
                }
                if (d) { p = cs; continue; }
            }
            span = (size_t)(ce - p);
            size_t clen = span - strlen(sub) - 1;        /* the subscript, without ']' */

            /* 5-MS: a MULTISAMPLE image's subscript is SPIRV-Cross's macro wrapper
             * `spvImage2DAtomicCoord(int2(<idx>, 0), <name>)`, where <idx> is the
             * FOLDED linear sidecar word index the SPIR-V pass
             * (mgl_spirv_lower_ms_images) computes: 4*sample + S*(layer + L*(x+W*y))
             * + component, bounds-selected by its own ImageQuery terms.  The macro
             * itself cannot stay: its out-of-range guard (edit 0 above) sends every
             * tc.x >= get_width() — every index past the first ROW of the image — to
             * the dump row pitch*height, so essentially every MS access collapsed
             * onto one word (verified: an 8x8x4 image stored everything at word 64).
             * Strip the wrapper and clamp in an MS-aware helper instead: in-range
             * indices pass through, the pass's own bounds-select already points OOB
             * accesses at tot, and the helper's clamp keeps them on the +1 DUMP word
             * (geomImageAtomicBuffer) no matter which path picked the index. */
            if (imgs[i].kind == MGL_AI_MS2D || imgs[i].kind == MGL_AI_MSARR)
            {
                const char *w = NULL;
                size_t wlen = 0;

                if (clen > 22 && !strncmp(cs, "spvImage2DAtomicCoord(", 22))
                {
                    /* `int2(<idx>, 0), <name>`: the first top-level comma of the
                     * macro call ends the int2(...) argument */
                    const char *seg0 = cs + 22;
                    const char *r = seg0, *c1 = NULL, *ci = NULL;
                    int depth = 1;                          /* inside the macro '(' */

                    while (r < cs + clen)
                    {
                        if (*r == '(' || *r == '[') depth++;
                        else if (*r == ')' || *r == ']') depth--;
                        else if (*r == ',' && depth == 1 && !c1) c1 = r;
                        r++;
                    }
                    if (c1 && (!strncmp(seg0, "int2(", 5) || !strncmp(seg0, "int3(", 5)))
                    {
                        /* inside int2/int3(...): the LAST top-level comma splits
                         * <idx> from the constant 0 (, 0) tail */
                        r = seg0 + 5;
                        depth = 1;
                        while (r < c1 - 1)
                        {
                            if (*r == '(' || *r == '[') depth++;
                            else if (*r == ')' || *r == ']') depth--;
                            else if (*r == ',' && depth == 1 && !ci) ci = r;
                            r++;
                        }
                        if (ci)
                        {
                            const char *z = ci + 1;
                            size_t rest;

                            while (z < c1 - 1 && (*z == ' ' || *z == '\t')) z++;
                            rest = (size_t)(c1 - 1 - z);        /* before the ')' */
                            /* the tail between <idx> and the closing ')' is `0`
                             * for int2 and `0, 0` for int3 (arrayed MS) */
                            if ((rest == 1 && z[0] == '0') ||
                                (rest == 4 && z[0] == '0' && z[1] == ',' && (z[2] == ' ' || z[2] == '\t') && z[3] == '0'))
                            { w = seg0 + 5; wlen = (size_t)(ci - w); }
                        }
                    }
                }
                if (w && wlen)
                {
                    const char *hn = (imgs[i].kind == MGL_AI_MSARR) ? "spvMglAIMSA" : "spvMglAIMS";
                    char *rep = malloc(strlen(hn) + strlen(imgs[i].name) + wlen + 8);

                    if (rep)
                    {
                        snprintf(rep, strlen(hn) + strlen(imgs[i].name) + wlen + 8,
                                 "%s(%s, %.*s)", hn, imgs[i].name, (int)wlen, w);
                        if (nedits < 256)
                        {
                            edits[nedits].at = (size_t)(cs - msl);
                            edits[nedits].len = clen;      /* the whole subscript */
                            edits[nedits].text = rep;
                            nedits++;
                        }
                        else free(rep);
                    }
                }
                p = ce;
                continue;                       /* MS images: no other rewrite */
            }

            /* ARR2 (image1DArray / image2DArray): strip the macro wrapper, keep the
             * plain coordinate as the helper's argument */
            const char *w = NULL;
            if (imgs[i].kind == MGL_AI_ARR2 && clen > 24 &&
                !strncmp(cs, "spvImage2DAtomicCoord(", 22))
            {
                int dd = 0;
                const char *r = cs + 21;
                while (*r)
                {
                    if (*r == '(') dd++;
                    else if (*r == ')') { if (--dd == 0) { r++; break; } }
                    r++;
                }
                if (!dd && (size_t)(r - cs) == clen)
                {
                    const char *c2 = cs + clen - 1;      /* the macro's ')' */
                    while (c2 > cs && *c2 != ',') c2--;  /* the macro's <tex> argument */
                    if (c2 > cs) { w = cs + 22; clen = (size_t)(c2 - cs - 22); }
                }
            }

            switch (imgs[i].kind)
            {
                case MGL_AI_3D:     helper = "spvMglAI3D"; break;
                /* cube and cube ARRAY get distinct helpers (not an overload): MSL has
                 * no texture_type query (get_texture_type does not exist — probed), so
                 * the face count is a compile-time constant per kind: 6 faces, or
                 * array_length*6, exactly what geomImageAtomicTexels sized the sidecar
                 * to on the CPU side */
                case MGL_AI_CUBE:    helper = "spvMglAICB"; break;
                case MGL_AI_CUBEARR: helper = "spvMglAICBA"; break;
                case MGL_AI_ARR2:
                    /* overloads pick the layout: int2 is a 1D array (the layer in y),
                     * int3 a 2D array (the layer in z) */
                    helper = "spvMglAIA";
                    break;
                default:
                    helper = NULL;      /* MGL_AI_MS* never gets here (see above) */
                    break;
            }
            if (!helper) { p = ce; continue; }

            /* keep the `&<name>_atomic[...]` lvalue (SPIRV-Cross takes its address):
             * only the subscript expression is replaced — the full interior, including
             * SPIRV-Cross's macro wrapper when there is one */
            char *rep = malloc(strlen(helper) + strlen(imgs[i].name) + clen + 16);
            if (rep)
            {
                snprintf(rep, strlen(helper) + strlen(imgs[i].name) + clen + 16,
                         "%s(%s, %.*s)", helper, imgs[i].name, (int)clen, w ? w : cs);
                if (nedits < 256)
                {
                    edits[nedits].at = (size_t)(cs - msl);
                    edits[nedits].len = span - strlen(sub) - 1;
                    edits[nedits].text = rep;
                    nedits++;
                }
                else free(rep);
            }
            p = ce;
        }
    }

    /* 5b. coherent loads/stores of an atomics-carrying image must go through the side
     * buffer too: the sidecar is the ONLY storage imageAtomic* sees, so an imageLoad
     * of the same texel through the texture argument would read the pre-dispatch value
     * (KHR-GL46.texture_buffer texture_buffer_atomic_functions: add into [0], read [0]
     * back, CompSwap it into every lane's slot — all within one kernel).  Redirect
     * `<name>.read(c)` / `<name>.write(v, c)` of the scalar targets (imageBuffer /
     * image1D) and plain image2D to the sidecar; the 3D / array / cube targets keep
     * their texture accesses (coherence gap, none of the CTS mixes them). */
    int used_load_helper = 0;
    for (int i = 0; i < nimgs; i++)
    {
        const char *reads[2] = { ".read(", ".write(" };

        if (!imgs[i].elem[0] || (!imgs[i].scalar && !imgs[i].is2d))
            continue;

        for (int which = 0; which < 2; which++)
        {
            char pat[160];
            snprintf(pat, sizeof pat, "%s%s", imgs[i].name, reads[which]);

            for (const char *p = msl; (p = strstr(p, pat)) != NULL; )
            {
                const char *arg;                       /* first argument */
                const char *ae;                        /* its end (matching ')') */
                const char *cs = NULL;                 /* top-level comma (write) */
                int d;

                if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
                { p += strlen(pat); continue; }
                arg = p + strlen(pat);
                d = 1; ae = arg;
                while (*ae)
                {
                    if (*ae == '(' || *ae == '[') d++;
                    else if (*ae == ')' || *ae == ']') { if (--d == 0) break; }
                    ae++;
                }
                if (d) { p = arg; continue; }
                if (which == 1)
                {
                    /* .write(<value>, <coord>): find the top-level comma, then the coord */
                    d = 1; cs = arg;
                    while (*cs)
                    {
                        if (*cs == '(' || *cs == '[') d++;
                        else if (*cs == ')' || *cs == ']') { if (--d == 0) break; }
                        else if (*cs == ',' && d == 1) break;
                        cs++;
                    }
                    if (*cs != ',') { p = ae; continue; }
                    arg = cs + 1;
                    while (*arg == ' ' || *arg == '\t') arg++;
                    d = 1; ae = arg;
                    while (*ae)
                    {
                        if (*ae == '(' || *ae == '[') d++;
                        else if (*ae == ')' || *ae == ']') { if (--d == 0) break; }
                        ae++;
                    }
                    if (d) { p = arg; continue; }
                }

                /* the linear sidecar subscript: scalar targets use the coordinate as
                 * written; image2D strips SPIRV-Cross's uint2(...) cast and reuses its
                 * own spvImage2DAtomicCoord macro (same pitch the atomics use) */
                {
                    const char *c0 = arg, *c1 = ae;
                    char texpr[512];
                    char *rep;

                    if (imgs[i].is2d)
                    {
                        size_t cl = (size_t)(c1 - c0);
                        if (cl > 7 && !strncmp(c0, "uint2(", 6) && c1[-1] == ')')
                        { c0 += 6; c1 -= 1; cl = (size_t)(c1 - c0); }
                        if (cl + 64 >= sizeof texpr) { p = ae; continue; }
                        snprintf(texpr, sizeof texpr, "spvImage2DAtomicCoord(%.*s, %s)",
                                 (int)cl, c0, imgs[i].name);
                    }
                    else
                    {
                        size_t cl = (size_t)(c1 - c0);
                        if (cl + 1 >= sizeof texpr) { p = ae; continue; }
                        snprintf(texpr, sizeof texpr, "%.*s", (int)cl, c0);
                    }

                    if (which == 0)
                    {
                        rep = malloc(strlen(texpr) + strlen(imgs[i].name) + strlen(imgs[i].elem) + 64);
                        if (!rep) { p = ae; continue; }
                        snprintf(rep, strlen(texpr) + strlen(imgs[i].name) + strlen(imgs[i].elem) + 64,
                                 "spvMglAILoad<%s>(%s_atomic, %s)", imgs[i].elem, imgs[i].name, texpr);
                        used_load_helper = 1;
                    }
                    else
                    {
                        /* the value argument spans from just after ".write(" to the comma */
                        const char *v0 = p + strlen(pat);
                        const char *v1 = cs;

                        rep = malloc((size_t)(v1 - v0) + strlen(texpr) + strlen(imgs[i].name) + 80);
                        if (!rep) { p = ae; continue; }
                        snprintf(rep, (size_t)(v1 - v0) + strlen(texpr) + strlen(imgs[i].name) + 80,
                                 "atomic_store_explicit(&%s_atomic[%s], (%.*s).x, memory_order_relaxed)",
                                 imgs[i].name, texpr, (int)(v1 - v0), v0);
                    }
                    if (nedits < 256)
                    {
                        edits[nedits].at = (size_t)(p - msl);
                        edits[nedits].len = (size_t)(ae + 1 - p);   /* through the ')' */
                        edits[nedits].text = rep;
                        nedits++;
                    }
                    else free(rep);
                }
                p = ae;
            }
        }
    }

    /* the helper definitions this MSL actually uses (plus one per ARRAY image —
     * each specialized to the exact `array<texture2d<T,access>, N>` type declared) */
    size_t helpers_cap = 2400 + 720 * (size_t)nimgs;
    char *helpers = malloc(helpers_cap);
    char *arrhelpers = malloc(helpers_cap);
    if (!helpers || !arrhelpers)
    {
        for (int a = 0; a < nedits; a++) free(edits[a].text);
        free(helpers);
        free(arrhelpers);
        return NULL;
    }
    helpers[0] = 0;
    arrhelpers[0] = 0;
    {
        static const struct { const char *name; const char *body; } h[7] = {
            /* The index helpers BOUNDS-CHECK the coordinate: GL discards image accesses
             * outside the image ("If the texel coordinates ... are outside the range of
             * the image, the result is undefined" — but a GPU-side write must not land
             * anywhere visible), and SPIRV-Cross's raw linearization would otherwise
             * address past the sidecar buffer (KHR-GL46.shader_image_load_store.
             * advanced-cast atomics at ivec2(gl_FragCoord.xy) of a window-sized quad
             * into a 64x64 image faulted/undefined-wrote).  An OOB access indexes the
             * one extra DUMP word the allocation carries (geomImageAtomicBuffer):
             * in-range results are unaffected, OOB ones all pile up on that word. */
            { "spvMglAI3D(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAI3D(T tex, int3 tc) { uint w = tex.get_width(), h = tex.get_height(), "
              "d = tex.get_depth(); if (any(tc < int3(0)) || uint(tc.x) >= w || uint(tc.y) >= h || "
              "uint(tc.z) >= d) return w * h * d; "
              "return uint(tc.x) + w * (uint(tc.y) + h * uint(tc.z)); }\n" },
            /* texture2d_array carries both image1DArray (an int2 coordinate: the layer in
             * y) and image2DArray (int3: the layer in z) — the overloads pick the layout */
            { "spvMglAIA(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAIA(T tex, int2 tc) { uint w = tex.get_width(), h = tex.get_height(); "
              "uint n = h; if (any(tc < int2(0)) || uint(tc.x) >= w || uint(tc.y) >= n) return w * h * tex.get_array_size(); "
              "return uint(tc.x) + w * uint(tc.y); }\n"
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAIA(T tex, int3 tc) { uint w = tex.get_width(), h = tex.get_height(); "
              "uint n = tex.get_array_size(); if (any(tc < int3(0)) || uint(tc.x) >= w || uint(tc.y) >= h || "
              "uint(tc.z) >= n) return w * h * n; "
              "return uint(tc.x) + w * (uint(tc.y) + h * uint(tc.z)); }\n" },
            /* a cube face — and a cube-array slice: glslang packs layer*6+face into z */
            { "spvMglAICB(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAICB(T tex, int3 tc) { uint w = tex.get_width(); uint n = 6u; "
              "if (any(tc < int3(0)) || uint(tc.x) >= w || uint(tc.y) >= w || uint(tc.z) >= n) "
              "return w * w * n; "
              "return uint(tc.z) * w * w + uint(tc.y) * w + uint(tc.x); }\n" },
            /* a cube ARRAY: glslang packs layer*6+face into z */
            { "spvMglAICBA(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAICBA(T tex, int3 tc) { uint w = tex.get_width(); uint n = "
              "(uint)tex.get_array_size() * 6u; "
              "if (any(tc < int3(0)) || uint(tc.x) >= w || uint(tc.y) >= w || uint(tc.z) >= n) "
              "return w * w * n; "
              "return uint(tc.z) * w * w + uint(tc.y) * w + uint(tc.x); }\n" },
            /* imageLoad of an atomics-carrying image, through its side buffer: the GLSL
             * gvec4 of a single-channel format is (v, 0, 0, 1) */
            { "spvMglAILoad<",
              "template <typename E, typename A> static inline __attribute__((always_inline)) "
              "vec<E,4> spvMglAILoad(device A* side, uint t) "
              "{ return vec<E,4>(E(atomic_load_explicit(&side[t], memory_order_relaxed)), E(0), E(0), E(1)); }\n" },
            /* a MULTISAMPLE image's sidecar word index: the SPIR-V pass folds
             * 4*sample + S*(layer + L*(x+W*y)) + component into one int; clamp it
             * into the sidecar (w*h*L*S*4 words; the +1 dump word is the last).
             * texture2d_ms_array needs its own helper: MSL has no texture-type
             * query, so the layer count comes from get_array_size() */
            { "spvMglAIMS(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAIMS(T tex, int idx) { uint w = tex.get_width(), h = tex.get_height(), "
              "n = tex.get_num_samples(); uint tot = w * h * n * 4u; "
              "if (idx < 0 || (uint)idx >= tot) return tot; return (uint)idx; }\n" },
            { "spvMglAIMSA(",
              "template <typename T> static inline __attribute__((always_inline)) "
              "uint spvMglAIMSA(T tex, int idx) { uint w = tex.get_width(), h = tex.get_height(), "
              "n = tex.get_num_samples(), l = tex.get_array_size(); uint tot = w * h * n * l * 4u; "
              "if (idx < 0 || (uint)idx >= tot) return tot; return (uint)idx; }\n" }
        };
        for (int k = 0; k < 7; k++)   /* 7 helpers: 3D / array / cube / cube-array + load + the two MS ones */
        {
            int used = (k == 3) ? used_load_helper : 0;
            for (int a = 0; a < nedits && !used; a++)
                used = !strncmp(edits[a].text, h[k].name, strlen(h[k].name));
            if (used && strlen(helpers) + strlen(h[k].body) + 1 < helpers_cap)
                strcat(helpers, h[k].body);
        }
        /* one helper per ARRAY image: the element offset uses the same 16-texel row
         * pitch the sidecar layout (geomImageAtomicBufferArray) and SPIRV-Cross's
         * spvImage2DAtomicCoord macro use */
        for (int i = 0; i < nimgs; i++)
        {
            char probe[160];
            int used = 0;

            if (!arrs[i].count || !arrs[i].hname[0]) continue;
            snprintf(probe, sizeof probe, "%s(", arrs[i].hname);
            for (int a = 0; a < nedits && !used; a++)
                used = (strstr(edits[a].text, probe) != NULL);
            if (!used) continue;

            /* a MULTISAMPLE element array: each element's region is its folded MS
             * layout (w*h*S*4 words; *layers for image2DMSArray elements) with NO row
             * padding — the same footprint geomImageAtomicArrayLayout sizes on the
             * CPU side.  The helper adds element i's offset to the folded index and
             * sends out-of-range accesses (also the pass's own bounds-select) to the
             * total-words DUMP word. */
            if (imgs[i].kind == MGL_AI_MS2D || imgs[i].kind == MGL_AI_MSARR)
            {
                bool arrayed_elem = (imgs[i].kind == MGL_AI_MSARR);
                char body[640], elemt[128];
                /* the parameter type must be a READABLE texture (get_num_samples is
                 * query-access-only): sanitize the element type's access qualifier —
                 * msl_msimg flips the ARGUMENT's write to read only later, and
                 * instantiating metal::array<texture2d_ms<..., access::write>, N>
                 * alone trips texture2d_ms's access static_assert */
                snprintf(elemt, sizeof elemt, "%s", arrs[i].elem);
                for (char *r = elemt; (r = strstr(r, "access::read_write")) != NULL; )
                {
                    memcpy(r, "access::read     ", 17);
                    r += 17;
                }
                for (char *r = elemt; (r = strstr(r, "access::write")) != NULL; )
                {
                    memcpy(r, "access::read ", 13);
                    r += 13;
                }
                snprintf(body, sizeof body,
                    "static inline __attribute__((always_inline)) uint %s"
                    "(thread const array<%s, %u>& a, uint i, int idx) "
                    "{ uint off = 0, tot = 0; "
                    "for (uint j = 0; j < %u; j++) { uint t = a[j].get_width() * a[j].get_height() "
                    "* a[j].get_num_samples() * 4u%s; if (j < i) off += t; tot += t; } "
                    "uint ti = a[i].get_width() * a[i].get_height() * a[i].get_num_samples() * 4u%s; "
                    "if (idx < 0 || (uint)idx >= ti) return tot; "
                    "return off + (uint)idx; }\n",
                    arrs[i].hname, elemt, arrs[i].count, arrs[i].count,
                    arrayed_elem ? " * a[j].get_array_size()" : "",
                    arrayed_elem ? " * a[i].get_array_size()" : "");
                if (strlen(arrhelpers) + strlen(body) + 1 < helpers_cap)
                    strcat(arrhelpers, body);
                continue;
            }

            /* compat r99: ARRAYED sidecars are laid out DENSELY per element
             * (geomImageAtomicBufferArray: w*h words each, no row padding) — only
             * the single-image 2D sidecar pads rows to 16 texels.  With 16u here the
             * arrayed writes landed past their element window (test_geom_imgarr
             * fragment_array_atomic_lands / compute_array_atomic_remapped_units). */
            const char *al = "1u";
            char body[640];
            snprintf(body, sizeof body,
                "static inline __attribute__((always_inline)) uint %s"
                "(thread const array<%s, %u>& a, uint i, int2 tc) "
                "{ uint off = 0, tot = 0; const uint al = (uint)(%s); "
                "for (uint j = 0; j < %u; j++) { uint p = (a[j].get_width() + al - 1) & ~(al - 1); "
                "uint t = p * a[j].get_height(); if (j < i) off += t; tot += t; } "
                "uint p = (a[i].get_width() + al - 1) & ~(al - 1); "
                "if (any(tc < int2(0)) || uint(tc.x) >= a[i].get_width() || uint(tc.y) >= a[i].get_height()) "
                "return tot; "
                "return off + p * uint(tc.y) + uint(tc.x); }\n",
                arrs[i].hname, arrs[i].elem, arrs[i].count, al, arrs[i].count);
            if (strlen(arrhelpers) + strlen(body) + 1 < helpers_cap)
                strcat(arrhelpers, body);
        }
    }
    if (!helpers[0] && !arrhelpers[0] && !nedits)
    {
        free(helpers);
        free(arrhelpers);
        return strdup(msl);
    }

    /* 6. splice: edits at ascending offsets of the ORIGINAL string; the helpers go
     * right after `using namespace metal;` (SPIRV-Cross always emits it) */
    for (int a = 0; a < nedits; a++)
        for (int b = a + 1; b < nedits; b++)
            if (edits[b].at < edits[a].at)
            {
                size_t ta = edits[a].at, tl = edits[a].len;
                char *tt = edits[a].text;
                edits[a] = edits[b]; edits[b].at = ta; edits[b].len = tl; edits[b].text = tt;
            }

    size_t helper_at = 0;
    {
        const char *u = strstr(msl, "using namespace metal;");
        if (u)
            helper_at = (size_t)(u - msl) + strlen("using namespace metal;");
        else
        {
            const char *inc = msl;
            while ((inc = strstr(inc, "#include")) != NULL)
            {
                const char *e = strchr(inc, '\n');
                if (e) inc = e + 1;
            }
            if (inc > msl) helper_at = (size_t)(inc - msl);
        }
    }
    /* the ARRAY helpers reference `spvLinearTextureAlignment` (SPIRV-Cross's
     * function-constant-controlled row pitch), which is declared LATER than the
     * generic injection point: splice them after its declaration instead */
    size_t arrhelper_at = helper_at;
    if (arrhelpers[0])
    {
        const char *lta = strstr(msl, "constant uint spvLinearTextureAlignment ");
        if (!lta)
            lta = strstr(msl, "constant uint spvLinearTextureAlignment=");
        if (lta)
        {
            const char *semi = strchr(lta, ';');
            if (semi) arrhelper_at = (size_t)(semi - msl) + 1;
        }
        else
        {
            const char *def = strstr(msl, "#define spvLinearTextureAlignment");
            if (def)
            {
                const char *e = strchr(def, '\n');
                if (e) arrhelper_at = (size_t)(e - msl) + 1;
            }
        }
    }

    size_t out_len = strlen(msl) + strlen(helpers) + strlen(arrhelpers) + 32;
    for (int a = 0; a < nedits; a++) out_len += strlen(edits[a].text) + 16;
    char *out = malloc(out_len);
    if (!out)
    {
        for (int a = 0; a < nedits; a++) free(edits[a].text);
        free(helpers);
        free(arrhelpers);
        return NULL;
    }

    size_t src = 0, dst = 0;
    for (int a = 0; a <= nedits; a++)
    {
        size_t upto = (a < nedits) ? edits[a].at : strlen(msl);
        if (helper_at > src && helper_at <= upto)
        {
            memcpy(out + dst, msl + src, helper_at - src); dst += helper_at - src;
            memcpy(out + dst, helpers, strlen(helpers)); dst += strlen(helpers);
            if (arrhelper_at == helper_at)
            {
                memcpy(out + dst, arrhelpers, strlen(arrhelpers)); dst += strlen(arrhelpers);
            }
            src = helper_at;
        }
        if (arrhelper_at > src && arrhelper_at <= upto && arrhelper_at != helper_at)
        {
            memcpy(out + dst, msl + src, arrhelper_at - src); dst += arrhelper_at - src;
            memcpy(out + dst, arrhelpers, strlen(arrhelpers)); dst += strlen(arrhelpers);
            src = arrhelper_at;
        }
        memcpy(out + dst, msl + src, upto - src); dst += upto - src;
        if (a < nedits)
        {
            memcpy(out + dst, edits[a].text, strlen(edits[a].text));
            dst += strlen(edits[a].text);
            src = edits[a].at + edits[a].len;
        }
        else src = upto;
    }
    memcpy(out + dst, msl + src, strlen(msl) - src); dst += strlen(msl) - src;
    out[dst] = 0;

    for (int a = 0; a < nedits; a++) free(edits[a].text);
    free(helpers);
    free(arrhelpers);
    return out;
}

/* MSL has no mutable program-scope variables (only `constant` globals), so a GL
 * atomic counter referenced inside a non-entry function cannot see the entry
 * function's injected [[buffer(N)]] argument.  Give every function that uses a
 * counter — directly, or by calling one that does — the counter pointers as
 * trailing parameters, and pass them at every call site.  The entry function is
 * left alone (its arguments come from inject_atomic_counters_core); its callers
 * don't exist.  Returns a malloc'd copy of the MSL, unchanged when no non-entry
 * function references a counter (the common case: counters used in main only). */

typedef struct {
    size_t id_off;        /* offset of the function-name identifier */
    size_t id_len;
    size_t params_close;  /* offset of the ')' closing the parameter list */
    size_t body_open;     /* offset of '{' */
    size_t body_close;    /* offset of the matching '}' */
    bool is_entry;
    bool need[MGL_MAX_ATOMIC_VARS];      /* direct counter uses in the body */
    bool need_all[MGL_MAX_ATOMIC_VARS];  /* transitive, through calls */
} MSLFunction;

static bool msl_ac_ident(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static char *msl_msl_counters_to_helper_params(const char *msl, const char *entry,
                                               const MGLAtomicCounterVar *vars, int nvars)
{
    if (!msl || !entry || !vars || nvars <= 0)
        return NULL;

    size_t len = strlen(msl);
    MSLFunction fns[256];
    int nfns = 0;

    /* 1. top-level function definitions: <ret> name(<params>) { <body> } */
    int depth = 0;
    for (size_t i = 0; i < len && nfns < 256; i++)
    {
        if (msl[i] == '{') { depth++; continue; }
        if (msl[i] == '}' && depth > 0) { depth--; continue; }
        if (msl[i] != '(' || depth != 0)
            continue;

        /* identifier before the '(' (whitespace allowed) */
        size_t j = i;
        while (j > 0 && strchr(" \t\r\n", msl[j - 1])) j--;
        size_t e = j;
        while (j > 0 && msl_ac_ident(msl[j - 1])) j--;
        if (e == j)
            continue;    /* not preceded by an identifier */

        /* control keywords are not functions */
        static const char *const kw[] = { "if", "for", "while", "switch", "return", "sizeof", "alignas", NULL };
        bool is_kw = false;
        for (int k = 0; kw[k]; k++)
            if (e - j == strlen(kw[k]) && !strncmp(msl + j, kw[k], e - j)) { is_kw = true; break; }
        if (is_kw)
            continue;

        /* matching ')' of the parameter list */
        int d2 = 1;
        size_t k = i + 1;
        while (k < len && d2)
        {
            if (msl[k] == '(') d2++;
            else if (msl[k] == ')') d2--;
            k++;
        }
        if (d2)
            continue;
        size_t pc = k - 1;

        /* a definition is followed (after whitespace) by '{' */
        size_t b = k;
        while (b < len && strchr(" \t\r\n", msl[b])) b++;
        if (b >= len || msl[b] != '{')
            continue;

        int d3 = 0;
        size_t bc;
        for (bc = b; bc < len; bc++)
        {
            if (msl[bc] == '{') d3++;
            else if (msl[bc] == '}' && --d3 == 0) break;
        }
        if (d3 != 0)
            continue;

        MSLFunction *fn = &fns[nfns];
        memset(fn, 0, sizeof *fn);
        fn->id_off = j;
        fn->id_len = e - j;
        fn->params_close = pc;
        fn->body_open = b;
        fn->body_close = bc;
        fn->is_entry = (fn->id_len == strlen(entry) && !strncmp(msl + j, entry, fn->id_len));
        nfns++;
        i = bc;    /* continue scanning after the body */
    }

    /* 2. direct counter uses per function body */
    for (int f = 0; f < nfns; f++)
        for (int v = 0; v < nvars; v++)
        {
            char pat[320];
            size_t pl = (size_t)snprintf(pat, sizeof pat, "(thread atomic_uint*)&%s", vars[v].name);
            for (const char *h = msl + fns[f].body_open + 1;
                 (h = strstr(h, pat)) != NULL && h < msl + fns[f].body_close; )
            {
                if (!msl_ac_ident(h[pl]))
                {
                    fns[f].need[v] = fns[f].need_all[v] = true;
                    break;
                }
                h++;
            }
        }

    /* 3. propagate through calls (no recursion in GLSL; iterate to a fixed point) */
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int f = 0; f < nfns; f++)
        {
            if (fns[f].is_entry) continue;
            for (int g = 0; g < nfns; g++)
            {
                if (g == f) continue;
                bool g_needs = false;
                for (int v = 0; v < nvars; v++) g_needs |= fns[g].need_all[v];
                if (!g_needs) continue;

                /* does f's body call g?  \b<name>\s*\( */
                for (size_t p = fns[f].body_open + 1; p + fns[g].id_len < fns[f].body_close; p++)
                {
                    if (strncmp(msl + p, msl + fns[g].id_off, fns[g].id_len) != 0) continue;
                    if (p > 0 && msl_ac_ident(msl[p - 1])) continue;
                    size_t q = p + fns[g].id_len;
                    while (q < fns[f].body_close && strchr(" \t\r\n", msl[q])) q++;
                    if (q >= fns[f].body_close || msl[q] != '(') continue;
                    /* the definition's own identifier is not a call */
                    if (p == fns[g].id_off) continue;

                    for (int v = 0; v < nvars; v++)
                        if (fns[g].need_all[v] && !fns[f].need_all[v]) { fns[f].need_all[v] = true; changed = true; }
                    break;
                }
            }
        }
    }

    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL geom: ac-helper scan: %d functions, entry=%d\n", nfns,
                nfns ? fns[0].is_entry : -1);
    /* 4. anything to do?  (entry functions get [[buffer]] arguments in the core) */
    bool any = false;
    for (int f = 0; f < nfns && !any; f++)
        if (!fns[f].is_entry)
            for (int v = 0; v < nvars; v++) any |= fns[f].need_all[v];
    if (!any)
        return strdup(msl);

    /* overloads share a name: same signature change, so union their sets */
    for (int f = 0; f < nfns; f++)
        for (int g = 0; g < nfns; g++)
            if (g != f && fns[f].id_len == fns[g].id_len &&
                !strncmp(msl + fns[f].id_off, msl + fns[g].id_off, fns[f].id_len))
                for (int v = 0; v < nvars; v++)
                    if (fns[g].need_all[v]) fns[f].need_all[v] = true;

    /* 5. collect the edits (all positions refer to the original string):
     *    a. helper definitions gain the trailing parameters
     *    b. every call site of a changed function passes them along */
    struct { size_t at; char *text; } edits[1024];
    int nedits = 0;

    for (int f = 0; f < nfns; f++)
    {
        bool needs[MGL_MAX_ATOMIC_VARS] = { false };
        bool any_need = false;
        for (int v = 0; v < nvars; v++) { needs[v] = fns[f].need_all[v]; any_need |= needs[v]; }
        if (!any_need) continue;

        /* a. the definition (entry ones keep their [[buffer]] form for the core) */
        if (!fns[f].is_entry)
        {
            char ins[4096];
            size_t off = 0;
            bool empty = true;
            for (size_t q = fns[f].params_close; q > 0; q--)
                if (!strchr(" \t\r\n", msl[q - 1])) { empty = msl[q - 1] == '('; break; }
            for (int v = 0; v < nvars && off + 128 < sizeof ins; v++)
            {
                if (!needs[v]) continue;
                off += (size_t)snprintf(ins + off, sizeof ins - off, "%svolatile device atomic_uint* %s",
                                        (empty && off == 0) ? "" : ", ", vars[v].name);
            }
            if (off && nedits < 1024)
            {
                edits[nedits].at = fns[f].params_close;
                edits[nedits].text = strdup(ins);
                nedits++;
            }
        }

        /* b. call sites of this function (anywhere, including inside the entry) */
        for (size_t p = 0; p + fns[f].id_len < len; p++)
        {
            if (strncmp(msl + p, msl + fns[f].id_off, fns[f].id_len) != 0) continue;
            if (p > 0 && msl_ac_ident(msl[p - 1])) continue;
            size_t q = p + fns[f].id_len;
            while (q < len && strchr(" \t\r\n", msl[q])) q++;
            if (q >= len || msl[q] != '(') continue;
            if (p == fns[f].id_off) continue;   /* the definition */

            /* matching ')' of the argument list */
            int d4 = 1;
            size_t r = q + 1;
            while (r < len && d4)
            {
                if (msl[r] == '(') d4++;
                else if (msl[r] == ')') d4--;
                r++;
            }
            if (d4) continue;
            size_t close_at = r - 1;

            bool empty_args = true;
            for (size_t s = q + 1; s < close_at; s++)
                if (!strchr(" \t\r\n", msl[s])) { empty_args = false; break; }

            char ins[4096];
            size_t off = 0;
            for (int v = 0; v < nvars && off + 128 < sizeof ins; v++)
            {
                if (!needs[v]) continue;
                off += (size_t)snprintf(ins + off, sizeof ins - off, "%s%s",
                                        (empty_args && off == 0) ? "" : ", ", vars[v].name);
            }
            if (off && nedits < 1024)
            {
                edits[nedits].at = close_at;
                edits[nedits].text = strdup(ins);
                nedits++;
            }
            p = r;   /* past the call */
        }
    }

    if (!nedits)
        return strdup(msl);

    /* 6. apply the edits in ascending position order — all offsets refer to the
     * original string, so no shifting is needed */
    for (int i = 0; i < nedits; i++)
        for (int k = i + 1; k < nedits; k++)
            if (edits[k].at < edits[i].at)
            {
                size_t ta = edits[i].at; char *tt = edits[i].text;
                edits[i].at = edits[k].at; edits[i].text = edits[k].text;
                edits[k].at = ta; edits[k].text = tt;
            }

    /* the edits INSERT text: the allocation must cover their total length (a plain
     * len+1 buffer overflowed the heap whenever a helper function touched a counter,
     * which surfaced as a later free() brk in test_atomic_counters) */
    size_t ins_total = 0;
    for (int i = 0; i < nedits; i++)
        ins_total += strlen(edits[i].text);

    char *out = malloc(len + 1 + ins_total);
    if (out)
    {
        out[0] = 0;
        size_t src = 0;
        for (int i = 0; i < nedits; i++)
        {
            strncat(out, msl + src, edits[i].at - src);
            strcat(out, edits[i].text);
            src = edits[i].at;
        }
        strcat(out, msl + src);
    }

    for (int i = 0; i < nedits; i++) free(edits[i].text);
    return out ? out : NULL;
}

static char *inject_atomic_counters_core(const char *msl, const char *entry,
                                         const MGLAtomicCounterVar *vars, int nvars,
                                         int max_metal_index, int forced_base, int *indices_out)
{
    if (!msl || !entry || !vars || nvars <= 0)
        return NULL;
    if (getenv("MGL_GEOM_DEBUG"))
        fprintf(stderr, "MGL geom: atomic-counter injection, pre-MSL:\n%s\n", msl);

    /* the entry function definition: <ret> <entry>(<params>) — require an identifier
     * boundary so a name that prefixes another identifier is not matched */
    size_t elen = strlen(entry);
    const char *sig = NULL;

    for (const char *p = msl; (p = strstr(p, entry)) != NULL; p += 1)
    {
        if (p > msl && (isalnum((unsigned char)p[-1]) || p[-1] == '_'))
            continue;
        const char *after = p + elen;
        while (*after == ' ' || *after == '\t')
            after++;
        if (*after == '(')
        {
            sig = after;
            break;
        }
    }
    if (!sig)
        return NULL;

    /* balanced-paren scan to the closing paren of the parameter list */
    const char *close = sig;
    int depth = 0;
    while (*close)
    {
        if (*close == '(') depth++;
        else if (*close == ')') { if (--depth == 0) break; }
        close++;
    }
    if (depth != 0 || *close != ')')
        return NULL;

    /* the first Metal buffer index of the atomic-counter group.  The renderer's
     * buffer map enumerates uniform blocks, default-uniform (uniform-constant)
     * buffers, shader-storage buffers and then the atomic-counter group in the
     * SPIRV-Cross reflection order, and binds entry i at Metal buffer index i —
     * that total is the group's index.  SPIRV-Cross's MSL signature keeps the
     * declared ordinals but may leave gaps where a declared-but-unused block was
     * never emitted, so counting the signature's arguments (the fallback when no
     * forced base is given) can drift below the map's ordinal
     * (KHR-GL46.compute_shader.pipeline-compute-chain declares g_buffer3/4 that
     * p0 never touches). */
    int base = forced_base;
    if (base < 0)
    {
        base = 0;
        for (const char *q = sig; (q = strstr(q, "[[buffer(")) != NULL && q < close; )
        {
            int idx = atoi(q + 9);
            if (idx >= 0 && idx < max_metal_index)
                base++;
            q += 9;
        }
    }
    if (base + nvars > max_metal_index)
        return NULL;   /* would overlap the emulation's pinned / reserved slots */

    /* build the parameter declarations to insert before the closing paren */
    size_t add_len = 2;
    char add[4096];
    add[0] = 0;
    for (int i = 0; i < nvars; i++)
    {
        int idx = base + i;
        char one[256];
        snprintf(one, sizeof one, "%svolatile device atomic_uint* %s [[buffer(%d)]]",
                 (i || sig + 1 < close) ? ", " : "", vars[i].name, idx);
        if (strlen(add) + strlen(one) + 1 > sizeof add)
            return NULL;
        strcat(add, one);
        if (indices_out)
            indices_out[i] = idx;
        add_len += strlen(one) + 32;
    }

    /* the expression rewrites below can LENGTHEN the text (`(thread ...` is shorter
     * than `(volatile device ...`, and the offset/index forms add characters): count
     * their growth first so the allocation covers it.  Rewriting in place on a buffer
     * sized for the parameter insert alone overflowed the heap (surfaced as a free()
     * brk in test_atomic_counters). */
    size_t grow = 0;
    for (int i = 0; i < nvars; i++)
    {
        char bad[300];
        unsigned w = vars[i].offset / 4;

        snprintf(bad, sizeof bad, "(thread atomic_uint*)&%s", vars[i].name);
        for (const char *q = msl; (q = strstr(q, bad)) != NULL; )
        {
            const char *after = q + strlen(bad);
            size_t bl = strlen(bad);
            char good[768];

            if (isalnum((unsigned char)*after) || *after == '_')
            {
                q += bl;
                continue;
            }
            if (*after == '[')
            {
                char idx[640];
                const char *e = NULL;
                idx[0] = '['; idx[1] = 0;
                if (!ac_linear_index(&vars[i], after, w, idx, sizeof idx, &e))
                    return NULL;    /* the rewrite loop below would reject it too */
                snprintf(good, sizeof good, "(volatile device atomic_uint*)&%s%s]",
                         vars[i].name, idx);
                bl += (size_t)(e - after);
            }
            else if (w)
                snprintf(good, sizeof good, "(volatile device atomic_uint*)&%s[%u]",
                         vars[i].name, w);
            else
                snprintf(good, sizeof good, "(volatile device atomic_uint*)%s",
                         vars[i].name);
            if (strlen(good) > bl)
                grow += strlen(good) - bl;
            q += bl;
        }
    }

    size_t len = strlen(msl);
    size_t insert_at = (size_t)(close - msl);
    char *out = (char *)malloc(len + add_len + grow);
    if (!out)
        return NULL;
    memcpy(out, msl, insert_at);
    memcpy(out + insert_at, add, strlen(add));
    memcpy(out + insert_at + strlen(add), msl + insert_at, len - insert_at + 1);

    /* re-point the atomic expressions at the declared arguments.  Two shapes:
     *   (thread atomic_uint*)&n           scalar counter
     *   (thread atomic_uint*)&n[expr]     element of an atomic_uint array (the index
     *                                     expression is arbitrary — keep it verbatim)
     * the GLSL `offset` qualifier is the base offset of element 0 in bytes.
     * A counter that is declared but never used was stripped from the MSL by
     * SPIRV-Cross: its argument is still declared (the renderer's buffer map
     * enumerates it — the ordinals must stay aligned) but there is nothing to
     * rewrite (KHR-GL46.compute_shader.pipeline-compute-chain p1). */
    for (int i = 0; i < nvars; i++)
    {
        char bad[300];
        unsigned w = vars[i].offset / 4;

        snprintf(bad, sizeof bad, "(thread atomic_uint*)&%s", vars[i].name);
        if (!strstr(out, bad))
            continue;
        for (char *q = out; (q = strstr(q, bad)) != NULL; )
        {
            const char *after = q + strlen(bad);
            char good[768];
            size_t bl = strlen(bad), gl;

            /* a longer variable name sharing this prefix (a / ac): not a match */
            if (isalnum((unsigned char)*after) || *after == '_')
            {
                q += bl;
                continue;
            }

            if (*after == '[')
            {
                /* copy ALL the bracket groups: &n[i] -> &n[(w)+(i)] and, for an
                 * array of arrays, &n[i][j] -> &n[(w)+((i))*D+(j)] (D = inner
                 * size) — the flattened index addresses the one flat pointer the
                 * argument declaration gives (dEQP-GL45-ES31 PIQ atomic-counter
                 * array.var_array_array: `uniform atomic_uint var_array_array[2][3]`
                 * is indexed with two subscripts, which a `device atomic_uint*`
                 * does not accept) */
                char idx[640];
                const char *e = NULL;
                idx[0] = '['; idx[1] = 0;
                if (!ac_linear_index(&vars[i], after, w, idx, sizeof idx, &e))
                {
                    free(out);
                    return NULL;
                }
                snprintf(good, sizeof good, "(volatile device atomic_uint*)&%s%s]",
                         vars[i].name, idx);
                bl += (size_t)(e - after);
            }
            else if (w)
                snprintf(good, sizeof good, "(volatile device atomic_uint*)&%s[%u]",
                         vars[i].name, w);
            else
                snprintf(good, sizeof good, "(volatile device atomic_uint*)%s",
                         vars[i].name);

            gl = strlen(good);
            memmove(q + gl, q + bl, strlen(q + bl) + 1);
            memcpy(q, good, gl);
            q += gl;
        }
    }

    return out;
}

static struct { int read; bool draw, xfb, cpu_xfb, cpu_assembly; } gEnv = { 0, false, false, false, false };

static bool geom_env(int which)
{
    if (!gEnv.read)
    {
        gEnv.read = 1;
        gEnv.draw = getenv("MGL_GEOM_TRACE_DRAW") != NULL;
        gEnv.xfb = getenv("MGL_GEOM_XFB_TRACE") != NULL;
        gEnv.cpu_xfb = getenv("MGL_GEOM_CPU_XFB") != NULL;
        gEnv.cpu_assembly = getenv("MGL_GEOM_CPU_ASSEMBLY") != NULL;
    }
    return which == 0 ? gEnv.draw : which == 1 ? gEnv.xfb : which == 2 ? gEnv.cpu_xfb : gEnv.cpu_assembly;
}

static void geom_dump(const char *tag, const char *text)
{
    static int seq;
    const char *dir = getenv("MGL_GEOM_DUMP");
    char path[512];
    FILE *f;
    if (!dir || !text) return;
    snprintf(path, sizeof(path), "%s/%03d_%s", dir, ++seq, tag);
    f = fopen(path, "w");
    if (!f) return;
    fwrite(text, 1, strlen(text), f);
    fclose(f);
}

/* rename the entry point `main` of a shader source */
static char *rename_main(const char *src, const char *new_name)
{
    size_t len = strlen(src);
    char *out = xmalloc(len + strlen(new_name) + 8);
    size_t o = 0, i = 0;
    bool renamed = false;
    char last_ident[64] = "";

    while (i < len)
    {
        if (is_ident_char(src[i]) && !isdigit((unsigned char)src[i]))
        {
            char id[128];
            size_t n = read_ident(src + i, id, sizeof(id));
            if (!strcmp(id, "main") && !strcmp(last_ident, "void"))
            {
                memcpy(out + o, new_name, strlen(new_name));
                o += strlen(new_name);
                renamed = true;
            }
            else
            {
                memcpy(out + o, id, n);
                o += n;
            }
            snprintf(last_ident, sizeof(last_ident), "%s", id);
            i += n;
            continue;
        }
        /* whitespace between `void` and `main` does not break the sequence */
        if (!is_ident_char(src[i]) && !isspace((unsigned char)src[i])) last_ident[0] = 0;
        out[o++] = src[i++];
    }
    out[o] = 0;

    if (!renamed)
    {
        free(out);
        return NULL;
    }
    return out;
}

/* ---------------- 1. vertex capture variant ---------------- */

/* rewrite the vertex shader's "out" interface blocks into ordinary declarations: an
 * anonymous block's members become variables, a named block's instance becomes a struct.
 * The builtin gl_PerVertex block is left alone. */
static char *geom_plain_outputs(const char *src)
{
    SB out;
    size_t i = 0, len = strlen(src);
    int depth = 0;

    memset(&out, 0, sizeof(out));
    while (i < len)
    {
        /* comments pass through untouched */
        if (src[i] == '/' && i + 1 < len && (src[i + 1] == '/' || src[i + 1] == '*'))
        {
            if (src[i + 1] == '/')
            {
                while (i < len && src[i] != '\n') { sb_put(&out, "%c", src[i]); i++; }
            }
            else
            {
                sb_put(&out, "/*");
                i += 2;
                while (i + 1 < len && !(src[i] == '*' && src[i + 1] == '/')) { sb_put(&out, "%c", src[i]); i++; }
                if (i + 1 < len) { sb_put(&out, "*/"); i += 2; }
            }
            continue;
        }
        if (depth == 0 && src[i] == '#')
        {
            /* a preprocessor directive is line-scoped: it must not be swallowed into the
             * next declaration's head (a '#define X 64' feeding a later declaration's
             * array size would otherwise be dropped when that declaration is rewritten) */
            while (i < len && src[i] != '\n') { sb_put(&out, "%c", src[i]); i++; }
            continue;
        }
        if (depth == 0)
        {
            /* a global-scope declaration: collect the head (up to ';' or '{') and check
             * whether its last storage qualifier is "out" */
            size_t save = i, q = i;
            bool is_out = false, other_storage = false, done_head = false;
            bool line_start = true;
            bool head_has_directive = false;
            (void)save;

            while (q < len && src[q] != ';' && src[q] != '{')
            {
                if (src[q] == '\n') line_start = true;
                else if (src[q] != '#' && !isspace((unsigned char)src[q])) line_start = false;
                if (src[q] == '#' && line_start)
                {
                    /* a preprocessor directive between declarations: it cannot belong to
                     * the head (a '#define X 16u' feeding a later array size must survive
                     * the head being rewritten), so the head ends before it.  The scan is
                     * observational only: nothing is emitted here, because a head that
                     * turns out not to need rewriting leaves the outer loop re-scanning
                     * the same text, which would re-emit the directive once per
                     * character.  Both outcomes emit the region exactly once below. */
                    head_has_directive = true;
                    break;
                }
                if (src[q] == '(')
                {
                    q++;
                    while (q < len && src[q] != ')') q++;
                    if (q < len) q++;
                    continue;
                }
                if (isalnum((unsigned char)src[q]) || src[q] == '_')
                {
                    char word[64];
                    int n2 = 0;
                    while (q < len && (isalnum((unsigned char)src[q]) || src[q] == '_'))
                    {
                        if (n2 < 63) word[n2++] = src[q];
                        q++;
                    }
                    word[n2] = 0;
                    if (!strcmp(word, "out")) { is_out = true; other_storage = false; }
                    else if (!strcmp(word, "in") || !strcmp(word, "uniform") || !strcmp(word, "buffer") ||
                             !strcmp(word, "attribute") || !strcmp(word, "varying") ||
                             !strcmp(word, "struct") || !strcmp(word, "const"))
                        other_storage = true;
                    continue;
                }
                q++;
            }
            if (q < len && src[q] == '{' && is_out && !other_storage) done_head = true;
            (void)save;
            if (done_head)
            {
                size_t brace = q, close = brace;
                int d = 0;
                while (close < len)
                {
                    if (src[close] == '{') d++;
                    else if (src[close] == '}') { d--; if (!d) break; }
                    close++;
                }
                if (close < len)
                {
                    /* the block name is the identifier before '{' */
                    char bname[64] = "";
                    size_t bp = brace, be;
                    while (bp > 0 && isspace((unsigned char)src[bp - 1])) bp--;
                    be = bp;
                    while (be > 0 && (isalnum((unsigned char)src[be - 1]) || src[be - 1] == '_')) be--;
                    if (bp - be < sizeof(bname))
                    {
                        memcpy(bname, src + be, bp - be);
                        bname[bp - be] = 0;
                    }
                    if (strcmp(bname, "gl_PerVertex"))
                    {
                        char inst[64] = "";
                        /* the rewritten block drops the scanned head (it holds the
                         * storage qualifiers), but any preprocessor directive inside it
                         * (a '#define' between the qualifiers and the name) must survive:
                         * copy the head's directive lines, and only those */
                        {
                            size_t k = i;
                            while (k < brace)
                            {
                                size_t ls = k, le = k;
                                while (ls < brace && (src[ls] == ' ' || src[ls] == '\t')) ls++;
                                while (le < brace && src[le] != '\n') le++;
                                if (ls < brace && src[ls] == '#')
                                    sb_put(&out, "%.*s\n", (int)(le - ls), src + ls);
                                k = (le < brace) ? le + 1 : brace;
                            }
                        }
                        size_t p = close + 1;
                        while (p < len && isspace((unsigned char)src[p])) p++;
                        while (p < len && (isalnum((unsigned char)src[p]) || src[p] == '_'))
                        {
                            size_t n2 = strlen(inst);
                            if (n2 + 1 < sizeof(inst)) { inst[n2] = src[p]; inst[n2 + 1] = 0; }
                            p++;
                        }
                        if (inst[0])
                        {
                            /* the block becomes a plain struct + instance; interface-only
                             * member qualifiers (location/xfb_offset/flat) and the storage
                             * qualifiers (in/out) must go: GLSL forbids them on struct
                             * members */
                            char *members = strip_member_quals2(src + brace + 1,
                                                               (size_t)(close - brace - 1), true);
                            sb_put(&out, "\nstruct %s {%s}; %s %s;", bname,
                                   members ? members : "", bname, inst);
                            free(members);
                        }
                        else
                        {
                            /* anonymous block: the members become ordinary outputs of their
                             * own, so the same qualifier stripping applies */
                            char *members = strip_member_quals(src + brace + 1,
                                                               (size_t)(close - brace - 1));
                            sb_put(&out, "\n%s;", members ? members : "");
                            free(members);
                        }
                        i = p;
                        while (i < len && src[i] != ';') i++;
                        if (i < len) i++;
                        continue;
                    }
                }
            }
            /* not rewritten: the head passes through verbatim, exactly once.  A scan that
             * stopped at a preprocessor directive emits only up to the '#', leaving the
             * directive to the outer loop's own branch (a rewrite would otherwise have
             * swallowed it, and emitting a character at a time here would rescan the head
             * for every character of it). */
            if (!done_head)
            {
                if (head_has_directive)
                {
                    for (size_t k = i; k < q && k < len; k++) sb_put(&out, "%c", src[k]);
                    i = q;
                }
                else
                {
                    for (size_t k = i; k <= q && k < len; k++) sb_put(&out, "%c", src[k]);
                    if (q < len && src[q] == '{') depth++;
                    i = (q < len) ? q + 1 : len;
                }
                continue;
            }
        }
        if (src[i] == '{') depth++;
        else if (src[i] == '}') depth--;
        sb_put(&out, "%c", src[i]);
        i++;
    }
    return out.s;
}

/* ---------------- plain (default-block) uniforms of the emulated stages ----------------
 *
 * The vertex capture variant and the geometry kernel are compiled from generated GLSL as
 * their own SPIR-V modules.  A plain "uniform T x;" there would become one
 * [[buffer(n)]] argument per variable (SPIRV-Cross has no notion of a default block),
 * laid out nothing like the program's default-uniform buffer, which introspect injects
 * as the single block "mgl_DefaultUniforms" for the raster stages.  We strip the plain
 * declarations from the transformed source and re-declare them as one std430 block whose
 * member offsets are the ones introspect baked into the program's SPIR-V, so the very
 * buffer that the renderer already builds for the stage can be bound verbatim for the
 * capture pass and the kernel (the renderer's buffer map puts the default block right
 * after the stage's uniform blocks — see mapGLBuffersToMTLBufferMap, which moves the
 * block to the head of the UNIFORM_CONSTANT list — and geomBuildPins below pins the
 * generated MSL to exactly those positions). */

typedef struct {
    char name[64];
    char type[96];      /* GLSL type as written, e.g. "ivec2", "float", "Foo" */
    char arr[192];     /* array suffix as written, "[4]" or "[gl_MaxTFInterleavedComponents / 8]"
                        * (r59: a 48-byte buffer rejected real-world constant-expression sizes
                        * like the enhanced_layouts glsl_constant_integral_expression suffix
                        * (49 chars), the declaration was then kept as a free-standing
                        * uniform, and SPIRV-Cross buffered it at an auto index that
                        * collided with the pinned default-uniforms buffer) */
} GeomPlainUni;

typedef struct {
    char     name[64];
    uint32_t offset;
} GeomDubMember;

static bool is_plain_uniform_type(const char *t)
{
    static const char *words[] = {
        "float", "int", "uint", "bool", "double",
        "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
        "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4",
        "dvec2", "dvec3", "dvec4",
        "mat2", "mat3", "mat4", "mat2x3", "mat2x4", "mat3x2", "mat3x4", "mat4x2", "mat4x3",
        "dmat2", "dmat3", "dmat4",
        "const", "highp", "mediump", "lowp",
        NULL
    };
    char tok[32];
    size_t n = 0;

    while (t[n] && !isspace((unsigned char)t[n]) && t[n] != '[' && n + 1 < sizeof(tok)) { tok[n] = t[n]; n++; }
    tok[n] = 0;
    if (!tok[0]) return false;
    for (int i = 0; words[i]; i++)
        if (!strcmp(tok, words[i])) return true;
    /* opaque types are NOT buffer-storable: they must stay where they were declared
     * (re-declaring a sampler inside the default-uniform block is illegal) */
    if (!strncmp(tok, "sampler", 7) || !strncmp(tok, "isampler", 8) ||
        !strncmp(tok, "usampler", 8) || !strncmp(tok, "image", 5) ||
        !strncmp(tok, "iimage", 6) || !strncmp(tok, "uimage", 6) ||
        !strncmp(tok, "atomic_uint", 11) || !strncmp(tok, "texture", 7))
        return false;
    /* anything else is a user-declared struct type (an identifier): also plain storage */
    for (size_t i = 0; i < n; i++)
        if (!is_ident_char((unsigned char)tok[i])) return false;
    return true;
}

/* Remove top-level plain (default-block) uniform declarations from src, recording them
 * (and the raw declarations, for the fallback) .  Declarations of blocks, samplers, images
 * and atomic counters are kept verbatim. */
/* replace C-style and line comments with spaces (newlines kept), so the plain-uniform
 * scanner's statement split is not confused: a comment has no ';' and would glue a
 * #define and the next uniform declaration into one "statement" */
static char *geom_blank_comments(const char *src)
{
    size_t n = strlen(src);
    char *out = xmalloc(n + 1);
    size_t i = 0, o = 0;

    while (i < n)
    {
        if (src[i] == '/' && i + 1 < n && src[i + 1] == '/')
        {
            while (i < n && src[i] != '\n') { out[o++] = ' '; i++; }
        }
        else if (src[i] == '/' && i + 1 < n && src[i + 1] == '*')
        {
            out[o++] = ' ';
            out[o++] = ' ';
            i += 2;
            while (i < n)
            {
                if (src[i] == '*' && i + 1 < n && src[i + 1] == '/')
                {
                    out[o++] = ' ';
                    out[o++] = ' ';
                    i += 2;
                    break;
                }
                out[o++] = (src[i] == '\n') ? '\n' : ' ';
                i++;
            }
        }
        else out[o++] = src[i++];
    }
    out[o] = 0;
    return out;
}

static char *geom_strip_plain_uniforms_at(const char *src, GeomPlainUni *uni, int *nuni, int max, char **removed, size_t *at);

static char *geom_strip_plain_uniforms(const char *src, GeomPlainUni *uni, int *nuni, int max, char **removed)
{
    return geom_strip_plain_uniforms_at(src, uni, nuni, max, removed, NULL);
}

/* strip the plain (default-block) uniforms of src; *at receives the offset in the
 * returned text where the FIRST declaration was removed — the re-declared
 * default-uniform block must go there (a member's struct type is lexically
 * declared before the uniform that used it, but not necessarily before the top
 * of the source, where the block used to be emitted) */
static char *geom_strip_plain_uniforms_at(const char *src, GeomPlainUni *uni, int *nuni, int max, char **removed, size_t *at)
{
    SB out;
    SB rm;
    size_t pos = 0, len = strlen(src);
    int depth = 0;

    memset(&out, 0, sizeof(out));
    memset(&rm, 0, sizeof(rm));
    *nuni = 0;
    if (at) *at = (size_t)-1;

    while (pos < len)
    {
        if (depth > 0)
        {
            while (pos < len && depth > 0)
            {
                if (src[pos] == '{') depth++;
                else if (src[pos] == '}') depth--;
                sb_put(&out, "%c", src[pos]);
                pos++;
            }
            continue;
        }

        while (pos < len && isspace((unsigned char)src[pos])) { sb_put(&out, "%c", src[pos]); pos++; }
        if (pos >= len) break;
        if (src[pos] == '{')                    /* a function body starts here */
        {
            depth++;
            sb_put(&out, "{");
            pos++;
            continue;
        }
        if (src[pos] == '#')                 /* preprocessor line: verbatim (after the
                                              * whitespace skip — a '#' found at the raw
                                              * scan start made the statement scanner eat
                                              * the NEXT uniform declaration with the #define) */
        {
            size_t end = pos;
            while (end < len && src[end] != '\n') end++;
            /* honour a trailing backslash (line continuation): the directive continues
             * on the next line */
            for (;;)
            {
                size_t back = end;
                while (back > pos && (src[back - 1] == ' ' || src[back - 1] == '\t')) back--;
                if (back > pos && src[back - 1] == '\\' && end < len)
                {
                    end++;
                    while (end < len && src[end] != '\n') end++;
                }
                else break;
            }
            sb_put(&out, "%.*s", (int)(end - pos), src + pos);
            pos = end;
            continue;
        }
        /* comments are pass-through statements of their own: without this, a '// ...'
         * line directly before a declaration becomes the head of that declaration's
         * statement, and the declaration (e.g. a plain uniform to strip) is never
         * recognized */
        if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '/')
        {
            size_t end = pos;
            while (end < len && src[end] != '\n') end++;
            sb_put(&out, "%.*s", (int)(end - pos), src + pos);
            pos = end;
            continue;
        }
        if (src[pos] == '/' && pos + 1 < len && src[pos + 1] == '*')
        {
            size_t end = pos + 2;
            while (end + 1 < len && !(src[end] == '*' && src[end + 1] == '/')) end++;
            end = (end + 1 < len) ? end + 2 : len;
            sb_put(&out, "%.*s", (int)(end - pos), src + pos);
            pos = end;
            continue;
        }

        /* the end of the statement: ';' at depth 0, or a '{' (a block declaration) */
        {
            size_t end = pos;
            int d = 0;
            while (end < len)
            {
                if (src[end] == '{') break;
                if (src[end] == ';' && d == 0) { end++; break; }
                if (src[end] == '(' || src[end] == '[') d++;
                else if (src[end] == ')' || src[end] == ']') d--;
                end++;
            }

            {
                char id[32];
                size_t stmt = pos;      /* statement start (may hold leading qualifiers) */
                size_t upos = pos;      /* where the storage qualifier sits */
                size_t n;

                /* "highp uniform vec4 x;" — precision (and precise/invariant) qualifiers
                 * may precede the storage qualifier.  The CTS spells its unused-uniform
                 * placeholder exactly this way (es31fProgramInterfaceQueryTests), and a
                 * declaration the scanner does not recognize stays a plain uniform in the
                 * capture variant, which SPIRV-Cross buffers at an auto index that
                 * collides with the pinned user blocks ("cannot reserve 'buffer'
                 * resource location at index N") and fails the whole program link. */
                for (;;)
                {
                    size_t qn = read_ident(src + upos, id, sizeof(id));
                    size_t k;

                    if (!qn || (strcmp(id, "highp") && strcmp(id, "mediump") &&
                                strcmp(id, "lowp") && strcmp(id, "precise") &&
                                strcmp(id, "invariant")))
                        break;
                    k = upos + qn;
                    while (k < len && isspace((unsigned char)src[k])) k++;
                    if (k >= len) break;
                    upos = k;
                }
                n = read_ident(src + upos, id, sizeof(id));

                if (n && !strcmp(id, "uniform") && end < len && src[end - 1] == ';')
                {
                    const char *body = src + upos + n;   /* between "uniform" and ';' */
                    const char *bend = src + end - 1;
                    char type[96] = "";
                    bool keep = false;

                    while (body < bend && isspace((unsigned char)*body)) body++;
                    while (bend > body && isspace((unsigned char)bend[-1])) bend--;

                    /* "uniform [const] T a[4], b" — split declarators on ',' */
                    {
                        const char *p = body;
                        while (p < bend && !keep)
                        {
                            const char *c = p, *e, *full;
                            int pd = 0;   /* commas inside an initializer's parens do not
                                          * separate declarators ("= uint[4](1u, 2u, ...)") */
                            while (c < bend && isspace((unsigned char)*c)) c++;
                            e = c;
                            while (e < bend)
                            {
                                if (*e == '(' || *e == '[') pd++;
                                else if (*e == ')' || *e == ']') pd--;
                                else if (*e == ',' && pd <= 0) break;
                                e++;
                            }
                            full = e;   /* uncropped end: the advance below needs it */
                            /* a default-value initializer ("= ...") is dropped instead of
                             * forcing the whole statement to be kept: the value lives in
                             * the default-uniform block, seeded at link from the
                             * initializer (the program's uniform shadow), so re-declaring
                             * the member in the block preserves the value.  Keeping the
                             * declaration would leave a default-block uniform in the
                             * variant that SPIRV-Cross buffers at an auto index,
                             * colliding with the pinned storage buffers
                             * (basic-atomic-case1). */
                            {
                                const char *eq = memchr(c, '=', (size_t)(e - c));
                                if (eq) e = eq;   /* declarator ends before "= ..." */
                            }
                            {
                                const char *nb = e;   /* end of the declarator */
                                const char *ne;
                                char name[64];
                                char arr[192] = "";   /* r59: constant-expression sizes are long */
                                size_t nn;

                                while (nb > c && isspace((unsigned char)nb[-1])) nb--;
                                if (nb > c && nb[-1] == ']')
                                {
                                    /* trailing [N]: keep the suffix text, the name ends before '[' */
                                    const char *ob = nb - 1;   /* at the ']' */

                                    while (ob > c && ob[-1] != '[') ob--;
                                    if (ob == c) keep = true;   /* ']' with no '[' */
                                    else
                                    {
                                        size_t alen = (size_t)(nb - (ob - 1));

                                        if (alen >= sizeof(arr)) keep = true;
                                        else
                                        {
                                            memcpy(arr, ob - 1, alen);
                                            arr[alen] = 0;
                                            nb = ob - 1;         /* the name ends here */
                                        }
                                    }
                                }
                                ne = nb;
                                while (ne > c && is_ident_char((unsigned char)ne[-1])) ne--;
                                nn = (size_t)(nb - ne);
                                if (!nn || ne == c) keep = true;   /* covers "a[2][3]" too */
                                else
                                {
                                    memcpy(name, ne, nn);
                                    name[nn] = 0;
                                    if (!type[0])
                                    {
                                        size_t t = (size_t)(ne - c);
                                        while (t > 0 && isspace((unsigned char)c[t-1])) t--;
                                        if (t == 0 || t >= sizeof(type)) keep = true;
                                        else
                                        {
                                            memcpy(type, c, t);
                                            type[t] = 0;
                                            /* drop precision/storage qualifiers so the
                                             * type name is what is_plain_uniform_type sees
                                             * ("lowp isampler2D" -> "isampler2D") */
                                            for (;;)
                                            {
                                                bool again = false;
                                                static const char *quals[] = { "lowp", "mediump", "highp", "precise", "const", NULL };

                                                for (int q = 0; quals[q]; q++)
                                                {
                                                    size_t l = strlen(quals[q]);
                                                    if (!strncmp(type, quals[q], l) &&
                                                        (type[l] == ' ' || type[l] == '\t'))
                                                    {
                                                        memmove(type, type + l + 1, strlen(type + l + 1) + 1);
                                                        again = true;
                                                        break;
                                                    }
                                                }
                                                if (!again) break;
                                            }
                                            if (!type[0]) keep = true;
                                        }
                                    }
                                    if (!keep)
                                    {
                                        if (*nuni >= max || !is_plain_uniform_type(type)) keep = true;
                                        else
                                        {
                                            snprintf(uni[*nuni].name, sizeof(uni[*nuni].name), "%s", name);
                                            snprintf(uni[*nuni].type, sizeof(uni[*nuni].type), "%s", type);
                                            snprintf(uni[*nuni].arr, sizeof(uni[*nuni].arr), "%s", arr);
                                            (*nuni)++;
                                        }
                                    }
                                }
                            }
                            /* advance from the uncropped declarator end: from a cropped
                             * initializer the scan would land inside "= ..." */
                            p = full + 1;
                        }
                    }

                    if (keep)
                    {
                        /* opaque or unsupported: keep this declaration verbatim (the
                         * already-removed ones stay removed — they are re-declared from
                         * the default-uniform block, so they must NOT be reset here);
                         * from the statement start, so a leading precision qualifier
                         * survives */
                        sb_put(&out, "%.*s", (int)(end - stmt), src + stmt);
                    }
                    else
                    {
                        if (removed) sb_put(&rm, "%.*s\n", (int)(end - stmt), src + stmt);
                        /* the re-declared block replaces this spot: every type it
                         * mentions is lexically declared by here */
                        if (at && *at == (size_t)-1) *at = out.len;
                    }
                    pos = end;
                    continue;
                }
            }

            /* anything else: copy, tracking braces */
            {
                size_t p = pos;
                while (p < end)
                {
                    if (src[p] == '{') depth++;
                    sb_put(&out, "%c", src[p]);
                    p++;
                }
                if (end == pos && pos < len) { pos++; }     /* always make progress */
                else pos = end;
            }
        }
    }

    if (removed) *removed = sb_take(&rm);
    else free(rm.s);
    return sb_take(&out);
}

/* The layout introspect baked into the program's SPIR-V for a stage's default-uniform
 * block: find the injected "mgl_DefaultUniforms" STRUCT and read its member offsets.
 * The name is on both the struct and its instance variable (introspect names them the
 * same): the struct is the id that carries the OpMemberName entries. */
static bool geom_dub_layout(SpvModule *m, GeomDubMember *mem, int *nmem, int max)
{
    uint32_t sid = 0;
    int n = 0;

    *nmem = 0;
    if (!m) return false;
    for (int i = 0; i < m->n_names; i++)
    {
        if (strcmp(m->names[i].name, "mgl_DefaultUniforms")) continue;
        for (int k = 0; k < m->n_mnames; k++)
            if (m->mnames[k].id == m->names[i].id) { sid = m->names[i].id; break; }
        if (sid) break;
    }
    if (getenv("MGL_SSBO_DEBUG"))
        fprintf(stderr, "MGL geom: dub_layout: m=%p n_names=%d n_mnames=%d sid=%u\n",
                (void *)m, m->n_names, m->n_mnames, sid);
    if (!sid) return false;

    for (int k = 0; n < max; k++)
    {
        bool found = false;

        for (int i = 0; i < m->n_mnames && n < max; i++)
        {
            if (m->mnames[i].id != sid || m->mnames[i].member != (uint32_t)k) continue;
            snprintf(mem[n].name, sizeof(mem[n].name), "%s", m->mnames[i].name);
            mem[n].offset = 0;
            found = true;
            break;
        }
        if (!found) break;
        for (int i = 0; i < m->n_mdecos; i++)
            if (m->mdecos[i].id == sid && m->mdecos[i].member == (uint32_t)k &&
                m->mdecos[i].deco == SpvDecorationOffset)
                mem[n].offset = m->mdecos[i].value;
        n++;
    }
    *nmem = n;
    return n > 0;
}

/* re-declare the stripped plain uniforms as one block matching that layout */
static void geom_emit_dub_block(SB *b, const GeomPlainUni *uni, int nuni,
                                const GeomDubMember *mem, int nmem)
{
    sb_put(b, "layout(std430, binding = %d) readonly buffer mgl_DefaultUniforms {\n", MGL_GEOM_SSBO_DUB);
    for (int i = 0; i < nuni; i++)
    {
        uint32_t off = 0;
        bool have = false;

        for (int k = 0; k < nmem; k++)
            if (!strcmp(mem[k].name, uni[i].name)) { off = mem[k].offset; have = true; break; }
        if (!have) continue;      /* introspect did not keep it (unused): skip */
        /* the array suffix is re-declared: std430 gives a 4-component-vector array a
         * 16-byte stride, the same layout introspect's msl_layout used for the offsets */
        sb_put(b, "    layout(offset = %u) %s %s%s;\n", off, uni[i].type, uni[i].name, uni[i].arr);
    }
    sb_put(b, "};\n");
}

/* pin the generated module's user resources to the flat buffer positions the renderer's
 * buffer map gives the same resources of the program's own stage:
 * mapGLBuffersToMTLBufferMap enumerates [uniform buffers, uniform constants (the default
 * block first), storage buffers, atomics] and binds each entry at its list position, which
 * is also the MSL buffer index in the program's stage module (SPIRV-Cross numbers uniform
 * buffers, then plain-uniform / uniform-constant args, then storage buffers).  prog is the
 * program's patched SPIR-V of the stage (it owns the ordering); mine is the generated
 * module (it owns the binding values that key the pins). */

/* Build the kernel-pinning map.  Returns the number of Metal argument slots the stage's
 * program resources span (0 on nothing to pin); a block/SSBO declared as an instance ARRAY
 * spans array_size consecutive slots, so the per-type counters advance by that — the bind
 * walk (geomBindStageBuffers / the capture pass) lays the buffers out the same way. */
/* the Block / BufferBlock decoration sits on the block's OpTypeStruct, not on the
 * variable: resolve through the pointer (and any instance array) */
static bool spv_var_is_buffer_block(SpvModule *m, SpvVar *v)
{
    SpvType *t = spv_value_type(m, v->type_id, NULL);

    return t && t->kind == SPVT_STRUCT && spv_deco(m, t->id, SpvDecorationBufferBlock, NULL);
}

/* does the module read .length() on a runtime-sized SSBO array (OpArrayLength)?  The
 * MSL variants then need SPIRV-Cross's `spvBufferSizeConstants` argument, which
 * geom_spirv_to_msl pins at MGL_GEOM_KERNEL_SIZE_BUFFER — the slot is kept out of the
 * user pin range (see the span checks at the geom_build_pins call sites). */
static bool spv_uses_array_length(const uint32_t *ir, size_t words)
{
    size_t i;

    if (!ir || words < 5)
        return false;
    for (i = 5; i < words; )                 /* skip the 5-word header */
    {
        uint32_t w = ir[i];

        if ((w & 0xffffu) == 117)            /* OpArrayLength */
            return true;
        i += (w >> 16) ? (w >> 16) : 1;
    }
    return false;
}

static int geom_build_pins(SpvModule *prog, SpvModule *mine,
                           GeomPin *pins, int *npins, int max,
                           MGLGeomTexSlot *slots, int *nslots, int max_slots)
{
    uint32_t ub[32], uc[32], sb[32];
    int nub = 0, nuc = 0, nsb = 0, total = 0, ub_span = 0;
    int dub_flat = -1;
    bool has_dub = false;

    *npins = 0;
    if (slots && nslots) *nslots = 0;

    /* Classify exactly like SPIRV-Cross's reflection (and hence like the renderer's
     * mapGLBuffersToMTLBufferMap enumerates the resource types): a Uniform-storage
     * variable with the Block decoration is a UBO — including the program's
     * default-uniform block mgl_DefaultUniforms, at its module position — while one
     * with the Buffer decoration (or StorageBuffer storage) is an SSBO.  glslang in
     * this toolchain emits SSBOs with Uniform storage + Buffer decoration, so going
     * by the storage class alone mixes both classes into one counter and the flats
     * stop matching the renderer's list whenever a stage has both a DUB and SSBOs
     * (KHR-GL46.shader_storage_buffer_object basic-atomic-case1). */
    for (int i = 0; i < prog->n_vars; i++)
    {
        SpvVar *v = &prog->vars[i];

        if (v->storage == SpvStorageClassUniform)
        {
            const char *nm = spv_name(prog, v->id);
            bool is_dub = nm && !strcmp(nm, "mgl_DefaultUniforms");

            if (!is_dub && spv_var_is_buffer_block(prog, v))
            {
                if (nsb < 32) sb[nsb++] = v->id;
                continue;
            }
            if (nub < 32)
            {
                uint32_t asz = 1;

                if (is_dub) { has_dub = true; dub_flat = ub_span; }
                ub[nub++] = v->id;
                spv_value_type(prog, v->type_id, &asz);   /* an instance array spans asz slots */
                ub_span += (int)asz;
            }
        }
        else if (v->storage == SpvStorageClassUniformConstant) { if (nuc < 32) uc[nuc++] = v->id; }
        else if (v->storage == SpvStorageClassStorageBuffer) { if (nsb < 32) sb[nsb++] = v->id; }
    }

    /* flat positions: ub j -> j (the DUB included at its position); uc k -> ub_span + k;
     * sb j -> ub_span + nuc + j — the order mapGLBuffersToMTLBufferMap builds the list */
    {
        int mub = 0, muc = 0, msb = 0;

        for (int i = 0; i < mine->n_vars; i++)
        {
            SpvVar *v = &mine->vars[i];
            int flat;
            uint32_t asz = 1;
            const char *nm = spv_name(mine, v->id);

            /* block-instance arrays span asz consecutive slots (GL binding points base..base+asz-1) */
            spv_value_type(mine, v->type_id, &asz);

            /* the generated variants pin their own pool blocks (MGLGs* / MGLGeom*) at
             * bindings above the DUB slot; user bindings stay below it (MGL reports at
             * most 14 uniform-buffer / 8 storage-buffer binding points), so the binding
             * cleanly separates them */
            {
                uint32_t bnd = (uint32_t)spv_binding(mine, v->id);

                if (bnd == MGL_GEOM_SSBO_DUB)
                {
                    if (!has_dub) continue;   /* no default-uniform block to mirror */
                    flat = dub_flat;
                    goto pin;
                }
                if (bnd > MGL_GEOM_SSBO_DUB) continue;   /* kernel pool resource */
            }

            if (v->storage == SpvStorageClassUniform)
            {
                if (spv_var_is_buffer_block(mine, v))
                {
                    if (msb >= nsb) continue;
                    flat = ub_span + nuc + msb;
                    msb += asz;
                    total += asz;
                }
                else
                {
                    if (mub >= nub) continue;
                    flat = mub;
                    mub += asz;
                    total += asz;
                }
            }
            else if (v->storage == SpvStorageClassUniformConstant)
            {
                if (muc >= nuc) continue;
                flat = ub_span + muc;
                muc++;
                total++;
            }
            else if (v->storage == SpvStorageClassStorageBuffer)
            {
                if (msb >= nsb) continue;
                flat = ub_span + nuc + msb;
                msb += asz;
                total += asz;
            }
            else continue;

pin:
            if (*npins < max)
            {
                pins[*npins].binding = spv_binding(mine, v->id);
                pins[*npins].msl = flat;
                pins[*npins].texture = (v->storage == SpvStorageClassUniformConstant);
                if (getenv("MGL_SSBO_DEBUG"))
                    fprintf(stderr, "MGL geom:   pin var %u storage %d asz %u -> binding %u msl %d name='%s'%s\n",
                            v->id, (int)v->storage, asz, pins[*npins].binding, flat,
                            nm ? nm : "(null)",
                            (flat == dub_flat && has_dub) ? " (DUB)" : "");
                (*npins)++;
            }

            /* texture (sampler / image) argument indices are NOT recorded here: the
             * MSLResourceBinding pins can be unreachable (binding collisions with the
             * packed uniform block), so geom_spirv_to_msl records the ACTUAL index of
             * every image/sampler resource into the slot table after compiling */
        }
    }

    if (getenv("MGL_SSBO_DEBUG"))
        fprintf(stderr, "MGL geom: pins: nub=%d ub_span=%d has_dub=%d dub_flat=%d nuc=%d nsb=%d total=%d npins=%d\n",
                nub, ub_span, (int)has_dub, dub_flat, nuc, nsb, total, *npins);
    return total;
}

/* Vertex-stage input location of a program module variable, by name.  The link
 * pass rewrites the SPIR-V Location decorations to the program's internal
 * attribute slots (introspect_spirv.c), so this is the index the renderer's
 * vertex descriptor binds at — the regenerated capture source must use exactly
 * these, not glslang's declaration-order assignment. */
static uint32_t geom_input_slot(const SpvModule *m, const char *name)
{
    for (int i = 0; i < m->n_vars; i++)
    {
        uint32_t loc;
        SpvVar *v = &m->vars[i];
        const char *nm;
        if (v->storage != SpvStorageClassInput) continue;   /* module scope: function locals use the Function class */
        nm = spv_name((SpvModule *)m, v->id);
        if (!nm || strcmp(nm, name)) continue;
        if (spv_deco((SpvModule *)m, v->id, SpvDecorationLocation, &loc)) return loc;
    }
    return 0xffffffffu;
}

/* Pin every top-level input declaration of the regenerated capture source to
 * the program module's slot: `layout(location = N) ...'.  Without this an
 * UNUSED input declared before a used one shifts glslang's automatic
 * numbering (KHR-GL46.cull_distance.functional declares
 * `in float clipdistance_data[1]' it never reads: culldistance_data moved to
 * attributes 1..8 and position to 9, so the capture kernel read every vertex
 * attribute one buffer off).  A declaration that already carries
 * layout(location=) is rewritten to the program slot for the same reason. */
static char *geom_pin_capture_inputs(const char *glsl, const SpvModule *vs_m)
{
    SB b;
    size_t i = 0, len = strlen(glsl);
    int depth = 0, changed = 0;

    memset(&b, 0, sizeof(b));
    while (i < len)
    {
        /* one source line (declarations may span lines; the storage keyword is
         * on the first) */
        size_t ls = i, e = i;
        while (e < len && glsl[e] != '\n') e++;
        i = e + (e < len);

        /* brace depth at line start (function bodies are skipped entirely) */
        {
            size_t k = ls;
            while (k < e)
            {
                if (glsl[k] == '{') depth++;
                else if (glsl[k] == '}') depth--;
                k++;
            }
        }
        if (depth > 0)
        {
            sb_put(&b, "%.*s", (int)(e - ls), glsl + ls);
            sb_put(&b, "\n");
            continue;
        }

        /* find an input declaration on this line: `in ...' / `attribute ...'
         * possibly behind layout(...) / interpolation qualifiers */
        size_t p = ls;
        while (p < e && (glsl[p] == ' ' || glsl[p] == '\t')) p++;
        size_t kw = (size_t)-1;
        int kwlen = 0;
        /* scan qualifier words until the storage keyword */
        for (;;)
        {
            size_t q = p;
            while (q < e && (is_ident_char((unsigned char)glsl[q]))) q++;
            if (q == p) break;
            if ((q - p == 2 && !strncmp(glsl + p, "in", 2)) ||
                (q - p == 9 && !strncmp(glsl + p, "attribute", 9)))
            {
                kw = p;
                kwlen = (int)(q - p);
                break;
            }
            p = q;
            /* skip separator chars; '(' is deliberately NOT here — it must reach the
             * balanced-paren skip below, which consumes a whole layout(...) group.
             * (r99 fix: with '(' in this list the open paren was eaten here, the
             * paren skip then started INSIDE the group at `location', the '=' stopped
             * the scan and `layout(location=N) in ...' — the most common declaration
             * form — was never pinned; the capture variant kept glslang's numbering
             * while the raster pipeline binds at the introspection slots, so any
             * program whose slots are renumbered (dvec3/dvec4/double-matrix inputs,
             * whose word pairs are packed wide-first) read the wrong attributes) */
            while (p < e && (glsl[p] == ' ' || glsl[p] == '\t' ||
                             glsl[p] == ',' || glsl[p] == ')')) p++;
            /* layout(...) qualifier contents are skipped as non-ident chars above */
            if (p < e && glsl[p] == '(')
            {
                int d2 = 1;
                p++;
                while (p < e && d2)
                {
                    if (glsl[p] == '(') d2++;
                    else if (glsl[p] == ')') d2--;
                    p++;
                }
                while (p < e && (glsl[p] == ' ' || glsl[p] == '\t')) p++;
            }
        }

        if (kw == (size_t)-1)
        {
            sb_put(&b, "%.*s", (int)(e - ls), glsl + ls);
            sb_put(&b, "\n");
            continue;
        }

        /* declared name: the identifier immediately before the first '[' (array
         * size), '=' or ';' of the declaration (which may continue past this
         * line) */
        char name[160];
        name[0] = 0;
        {
            size_t d = kw + kwlen;
            size_t dend = d;
            while (dend < len && glsl[dend] != '[' && glsl[dend] != '=' && glsl[dend] != ';')
            {
                if (glsl[dend] == '{') break;   /* not a simple declaration */
                dend++;
            }
            size_t q = dend;
            while (q > d && !is_ident_char((unsigned char)glsl[q - 1])) q--;
            size_t nend = q;
            while (q > d && is_ident_char((unsigned char)glsl[q - 1])) q--;
            if (nend > q && nend - q < sizeof name)
            {
                memcpy(name, glsl + q, nend - q);
                name[nend - q] = 0;
            }
        }

        uint32_t slot = name[0] ? geom_input_slot(vs_m, name) : 0xffffffffu;
        if (slot == 0xffffffffu || !name[0])
        {
            /* not a program input (or no slot): keep verbatim */
            sb_put(&b, "%.*s", (int)(e - ls), glsl + ls);
            sb_put(&b, "\n");
            continue;
        }

        /* replace an existing location= qualifier in the preserved prefix (a
         * second layout(location) would be a syntax error): keep any other
         * qualifier of the same layout(), then emit ours before the keyword */
        {
            char prefix[512], fixed[600];
            size_t pl = kw - ls;
            if (pl >= sizeof prefix) pl = sizeof prefix - 1;
            memcpy(prefix, glsl + ls, pl);
            prefix[pl] = 0;
            /* find the layout qualifier closest to the storage keyword */
            char *lay = NULL;
            for (char *q = prefix; (q = strstr(q, "layout")); q += 6) lay = q;
            fixed[0] = 0;
            if (lay)
            {
                char *op = strchr(lay, '(');
                char *cp = op ? strchr(op, ')') : NULL;
                if (op && cp)
                {
                    /* qualifiers minus location */
                    char keep[256];
                    size_t kl = 0;
                    char *s2 = op + 1;
                    *cp = 0;
                    while (s2 < cp && kl + 1 < sizeof keep)
                    {
                        char *c2 = strchr(s2, ',');
                        char *e2 = c2 ? c2 : cp;
                        while (s2 < e2 && (*s2 == ' ' || *s2 == '\t')) s2++;
                        while (e2 > s2 && (e2[-1] == ' ' || e2[-1] == '\t')) e2--;
                        if (e2 > s2 && strncmp(s2, "location", 8))
                        {
                            if (kl) keep[kl++] = ',';
                            if (kl + (size_t)(e2 - s2) >= sizeof keep) break;
                            memcpy(keep + kl, s2, (size_t)(e2 - s2));
                            kl += (size_t)(e2 - s2);
                        }
                        s2 = c2 ? c2 + 1 : cp;
                    }
                    keep[kl] = 0;
                    *cp = ')';
                    /* rebuild the line with the kept qualifiers (or the whole
                     * layout() dropped when none remain) */
                    size_t head = (size_t)(lay - prefix);
                    if (kl)
                        snprintf(fixed, sizeof fixed, "%.*slayout(%.*s)%s",
                                 (int)head, prefix, (int)kl, keep, cp + 1);
                    else
                        snprintf(fixed, sizeof fixed, "%.*s%s",
                                 (int)head, prefix, cp + 1);
                }
            }
            if (!fixed[0]) snprintf(fixed, sizeof fixed, "%s", prefix);
            sb_put(&b, "%s", fixed);
        }
        sb_put(&b, "layout(location = %u) ", slot);
        sb_put(&b, "%.*s", kwlen, glsl + kw);
        sb_put(&b, "%.*s", (int)(e - (kw + kwlen)), glsl + kw + kwlen);
        sb_put(&b, "\n");
        changed = 1;
    }
    if (!changed) { free(b.s); return NULL; }
    return b.s;
}

/* Rewrite the declared sizes of gl_ClipDistance / gl_CullDistance in the
 * regenerated capture source to the capture RECORD's word count: the record
 * copies gp->in_clip_size entries of each (the cull/clip evaluation reads a
 * fixed layout), but the shader may declare smaller arrays — glslang rejects
 * `gl_ClipDistance[1]' indexing beyond its declared size, which used to fail
 * the whole synthetic build whenever clip and cull sizes differed
 * (KHR-GL46.cull_distance.functional iteration 1: clip=1 + cull=7).  Only
 * declarations are rewritten (the statement up to the name mentions `out');
 * reads inside the function bodies are left alone. */
static char *geom_pin_builtins(const char *glsl, GLuint clip_words)
{
    static const char *names[2] = { "gl_ClipDistance", "gl_CullDistance" };
    SB b;
    size_t i = 0, len = strlen(glsl);
    int changed = 0;

    if (!clip_words) return NULL;
    memset(&b, 0, sizeof(b));
    while (i < len)
    {
        /* copy through to the next occurrence of either name */
        size_t hit = (size_t)-1;
        int which = 0;
        for (int k = 0; k < 2; k++)
        {
            const char *f = strstr(glsl + i, names[k]);
            if (f)
            {
                size_t off = (size_t)(f - glsl);
                if (!is_ident_char((unsigned char)f[-1]) &&
                    !is_ident_char((unsigned char)f[strlen(names[k])]) &&
                    off < hit) { hit = off; which = k; }
            }
        }
        if (hit == (size_t)-1)
        {
            sb_put(&b, "%s", glsl + i);
            break;
        }
        /* statement start: previous ';' or '{' or '}' */
        size_t st = hit;
        while (st > 0 && glsl[st - 1] != ';' && glsl[st - 1] != '{' && glsl[st - 1] != '}') st--;
        int is_out = 0;
        for (size_t q = st; q + 3 < hit; q++)
            if (!strncmp(glsl + q, "out", 3) &&
                !is_ident_char((unsigned char)glsl[q - 1]) &&
                !is_ident_char((unsigned char)glsl[q + 3]))
            { is_out = 1; break; }
        /* the declaration's size bracket follows the name */
        size_t nb = hit + strlen(names[which]);
        while (nb < len && (glsl[nb] == ' ' || glsl[nb] == '\t')) nb++;
        if (!is_out || nb >= len || glsl[nb] != '[')
        {
            /* not a declaration: copy this occurrence and continue after it */
            sb_put(&b, "%.*s", (int)(nb - i), glsl + i);
            i = nb;
            continue;
        }
        size_t ne = nb + 1;
        while (ne < len && glsl[ne] != ']' && glsl[ne] != ';') ne++;
        if (ne >= len || glsl[ne] != ']')
        {
            sb_put(&b, "%.*s", (int)(nb - i), glsl + i);
            i = nb;
            continue;
        }
        sb_put(&b, "%.*s[%u]", (int)(nb - i), glsl + i, clip_words);
        i = ne + 1;
        changed = 1;
    }
    if (!changed) { free(b.s); return NULL; }
    return b.s;
}

static char *geom_gen_capture_glsl_inner(GLMContext ctx, MGLGeomProgram *gp, const char *vs_src,
                                        const SpvModule *vs_m, char **log);

static char *geom_gen_capture_glsl(GLMContext ctx, MGLGeomProgram *gp, const char *vs_src,
                                   const SpvModule *vs_m, char **log)
{
    /* the program's own compile hoisted the sampler/image members of default-block struct
     * uniforms out of the structs (they cannot live in an MSL buffer); the capture variant
     * must do the same, or its re-declared default-uniform block carries a struct that
     * still has the sampler inside ("member of block cannot be or contain a sampler") */
    char *hoisted = mgl_hoist_struct_samplers(vs_src);
    char *out;
    if (hoisted) vs_src = hoisted;
    out = geom_gen_capture_glsl_inner(ctx, gp, vs_src, vs_m, log);
    free(hoisted);
    return out;
}

static char *geom_gen_capture_glsl_inner(GLMContext ctx, MGLGeomProgram *gp, const char *vs_src,
                                   const SpvModule *vs_m, char **log)
{
    char *renamed = rename_main(vs_src, "mgl_geom_vs_main");
    char *nover = NULL;
    char *noblk = NULL;
    char *removed_src = NULL;
    GeomPlainUni uni[MGL_GEOM_MAX_VARYINGS];
    GeomDubMember dub[MGL_GEOM_MAX_VARYINGS];
    int nuni_ = 0, ndub = 0;
    size_t dub_at = (size_t)-1;
    bool dub_spliced = false;
    SB b;

    if (!renamed)
    {
        mgl_append_log(log, "ERROR: geometry: vertex shader has no main()\n");
        return NULL;
    }
    nover = strip_version(renamed);
    free(renamed);
    /* plain (default-block) uniforms are re-declared below as one block that matches the
     * program's default-uniform buffer (see geom_emit_dub_block) */
    nuni_ = 0;
    dub_at = (size_t)-1;
    {
        char *stripped = geom_strip_plain_uniforms_at(nover, uni, &nuni_, MGL_GEOM_MAX_VARYINGS, &removed_src, &dub_at);
        if (!stripped) { free(nover); return NULL; }
        if (nuni_ && geom_dub_layout((SpvModule *)vs_m, dub, &ndub, MGL_GEOM_MAX_VARYINGS))
        {
            /* splice the re-declared block in at the first removed declaration, now: the
             * later source rewrites (plain outputs, pinned inputs, widened builtins)
             * shift the text and would invalidate the offset — and a member's struct
             * type is only declared by that point, not at the top of the source */
            size_t at = (dub_at == (size_t)-1 || dub_at > strlen(stripped)) ? strlen(stripped) : dub_at;
            SB t;
            memset(&t, 0, sizeof(t));
            sb_put(&t, "%.*s", (int)at, stripped);
            geom_emit_dub_block(&t, uni, nuni_, dub, ndub);
            sb_put(&t, "%s", stripped + at);
            free(stripped);
            stripped = t.s ? t.s : xstrdup("");
            dub_spliced = true;
        }
        if (nuni_) { free(nover); nover = stripped; }
        else free(stripped);
    }
    /* the captured outputs must be ordinary variables: an "out" interface block would be
     * mapped to the vertex function's output struct (and SPIRV-Cross's copy-back of an
     * anonymous block reads a temporary), while the store below needs the members in scope */
    noblk = geom_plain_outputs(nover);
    free(nover);
    nover = noblk;
    /* pin the input locations to the program's attribute slots (see
     * geom_pin_capture_inputs) — the raster pipeline binds at those, and a
     * recompile through glslang would renumber them in declaration order */
    {
        char *pinned = geom_pin_capture_inputs(nover, vs_m);
        if (pinned) { free(nover); nover = pinned; }
    }
    /* and widen the clip/cull builtin arrays to the record's word count (see
     * geom_pin_builtins) */
    {
        char *widened = geom_pin_builtins(nover, gp->in_clip_size);
        if (widened) { free(nover); nover = widened; }
    }

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version 460 core\n");

    /* re-declare the stripped plain uniforms (or put them back if the program's
     * default-uniform block could not be read) — the block was already spliced into the
     * source at the first removed declaration (dub_spliced), so only the fallback is
     * left here */
    if (!dub_spliced && removed_src)
    {
        sb_put(&b, "%s", removed_src);
        nuni_ = 0;
    }
    free(removed_src);
    removed_src = NULL;

    sb_put(&b, "%s\n", nover);
    free(nover);

    sb_put(&b, "layout(std430, binding = %d) buffer MGLGeomVsOutBlock { uint mgl_hdr[8]; uint mgl_rec[]; };\n",
           MGL_GEOM_SSBO_VSOUT);
    sb_put(&b, "void main()\n{\n    mgl_geom_vs_main();\n");
    sb_put(&b, "    uint mgl_o = ((uint(gl_VertexID) - mgl_hdr[0]) + (uint(gl_InstanceID) - mgl_hdr[2]) * mgl_hdr[1]) * mgl_hdr[3];\n");
    /* r63: mgl_hdr[4] (written fresh per draw CPU-side, after the slice sizing) is the
     * record capacity in words: for an indexed capture pass gl_VertexID is the fetched
     * index VALUE, so mgl_o spans the drawn VERTEX RANGE — which a direct draw's pool
     * now covers exactly (geom_route_gpu scans the element range), but an indirect
     * draw's CPU-side bound cannot.  Without the clamp the capture wrote past the
     * slice and paged faulted intermittently (test_geom_xfbperf); with it, records
     * that cannot fit are dropped instead (documented gap for indirect draws whose
     * vertex range exceeds the pool bound). */
    sb_put(&b, "    if (mgl_o + %uu > mgl_hdr[4]) { return; }\n", gp->in_words);

    if (gp->in_has_pos)
    {
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.x);\n", gp->in_pos_off + 0);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.y);\n", gp->in_pos_off + 1);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.z);\n", gp->in_pos_off + 2);
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_Position.w);\n", gp->in_pos_off + 3);
    }
    if (gp->in_has_psize)
        sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_PointSize);\n", gp->in_psize_off);
    if (gp->in_has_clip)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_ClipDistance[%u]);\n", gp->in_clip_off + i, i);
    if (gp->in_has_cull)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "    mgl_rec[mgl_o + %u] = floatBitsToUint(gl_CullDistance[%u]);\n", gp->in_cull_off + i, i);

    for (int i = 0; i < gp->n_in; i++)
    {
        MGLGeomVar *v = &gp->in_v[i];
        char open[64], close[8];
        snprintf(open, sizeof(open), "mgl_rec[mgl_o");
        snprintf(close, sizeof(close), "]");
        /* an input the vertex stage never writes stays undefined in GL: record zeros */
        gen_varying_store(&b, v, "mgl_rec", "mgl_o", v->expr[0] ? v->expr : NULL);
    }

    sb_put(&b, "}\n");
    return sb_take(&b);
}

/* Remove duplicate "struct NAME { ... }; NAME <instance>..." declarations: a shader's input
 * and output interface blocks may share the block NAME, and the per-statement emitter of
 * the transformed kernel source writes one "struct" per block.  Returns a malloc'd deduped
 * source, or NULL when nothing needed removing. */
static char *geom_dedupe_iface_structs(const char *src)
{
    if (!src)
        return NULL;

    struct { char name[64]; char body[512]; } seen[32];
    int nseen = 0, edits = 0;
    char *out = NULL;
    size_t out_len = 0;
    FILE *mem = open_memstream(&out, &out_len);
    size_t i = 0;

    if (!mem)
        return NULL;
    while (src[i])
    {
        const char *st = strstr(src + i, "struct ");
        if (!st)
        {
            fputs(src + i, mem);
            break;
        }
        size_t at = (size_t)(st - src);
        fwrite(src + i, 1, at - i, mem);
        /* parse "struct NAME {" */
        size_t p = at + strlen("struct ");
        char name[64] = "";
        size_t k = 0;
        while (isalnum((unsigned char)src[p]) || src[p] == '_' && k + 1 < sizeof(name))
            name[k++] = src[p++];
        name[k] = 0;
        while (src[p] == ' ' || src[p] == '\t') p++;
        if (!name[0] || src[p] != '{')
        {
            /* not a struct definition: copy the keyword and continue after it */
            fwrite(src + at, 1, p - at, mem);
            i = p;
            continue;
        }
        /* find the matching close brace */
        size_t b = p + 1;
        int depth = 1;
        while (src[b] && depth)
        {
            if (src[b] == '{') depth++;
            else if (src[b] == '}') depth--;
            if (src[b]) b++;
        }
        /* the emitter writes "}; NAME inst..." right after: the declaration ends at the
         * next ';' after '}' */
        size_t e = b;
        while (src[e] && src[e] != ';') e++;
        char body[512] = "";
        {
            size_t n = b - (p + 1);
            if (n >= sizeof(body)) n = sizeof(body) - 1;
            memcpy(body, src + p + 1, n);
            body[n] = 0;
        }
        int dup = -1;
        for (int s = 0; s < nseen; s++)
            if (!strcmp(seen[s].name, name))
            {
                dup = strcmp(seen[s].body, body) ? 0x100 /* differs: keep both, see below */
                                                  : s;
                break;
            }
        if (dup >= 0 && dup != 0x100)
        {
            /* identical redefinition: drop the whole "struct NAME { ... };" — anything
             * after the struct's own ';' (the instance declaration) flows through */
            size_t semi = b;                      /* first char after '}' */
            while (src[semi] && isspace((unsigned char)src[semi])) semi++;
            if (src[semi] == ';')
            {
                i = semi + 1;                     /* skip "struct ... };" plus its ';' */
            }
            else
            {
                /* "struct N { ... } inst;" (instance attached): keep the type reference */
                fwrite(name, 1, strlen(name), mem);    /* "NAME" in place of the struct */
                i = b + 1;                             /* continue after '}' */
            }
            edits++;
            continue;
        }
        if (dup == 0x100)
        {
            /* same NAME, different members: rename the second struct (and its declaration
             * on this line) so both can coexist */
            char renamed[64], line[1024];
            snprintf(renamed, sizeof(renamed), "%s_2", name);
            size_t n = e + 1 - at;
            if (n >= sizeof(line)) n = sizeof(line) - 1;
            memcpy(line, src + at, n);
            line[n] = 0;
            for (char *q = line; (q = strstr(q, name)); )
            {
                size_t pos = q - line;
                if ((pos == 0 || !isalnum((unsigned char)q[-1]) && q[-1] != '_') &&
                    !isalnum((unsigned char)q[strlen(name)]) && q[strlen(name)] != '_')
                {
                    memmove(q + strlen(renamed), q + strlen(name), strlen(q + strlen(name)) + 1);
                    memcpy(q, renamed, strlen(renamed));
                    q += strlen(renamed);
                }
                else q += strlen(name);
            }
            fputs(line, mem);
            i = e + 1;
            edits++;
            continue;
        }
        if (nseen < (int)(sizeof(seen) / sizeof(seen[0])))
        {
            snprintf(seen[nseen].name, sizeof(seen[0].name), "%s", name);
            snprintf(seen[nseen].body, sizeof(seen[0].body), "%s", body);
            nseen++;
        }
        fwrite(src + at, 1, e + 1 - at, mem);
        i = e + 1;
    }
    fclose(mem);
    if (!edits)
    {
        free(out);
        return NULL;
    }
    return out;
}

static char *geom_gen_kernel_glsl_inner(GLMContext ctx, MGLGeomProgram *gp, const char *gs_src,
                                       const SpvModule *gs_m, char **log);

static char *geom_gen_kernel_glsl(GLMContext ctx, MGLGeomProgram *gp, const char *gs_src,
                                  const SpvModule *gs_m, char **log)
{
    /* see geom_gen_capture_glsl: the kernel is generated from the un-hoisted source and
     * would re-declare a sampler-bearing struct inside the default-uniform block */
    char *hoisted = mgl_hoist_struct_samplers(gs_src);
    char *out;
    if (hoisted) gs_src = hoisted;
    out = geom_gen_kernel_glsl_inner(ctx, gp, gs_src, gs_m, log);
    free(hoisted);
    return out;
}

static char *geom_gen_kernel_glsl_inner(GLMContext ctx, MGLGeomProgram *gp, const char *gs_src,
                                  const SpvModule *gs_m, char **log)
{
    char gl_in_name[64];
    char *gs_expanded = geom_preprocess_source(ctx, GL_GEOMETRY_SHADER, gs_src);
    const char *gs_base = gs_expanded ? gs_expanded : gs_src;
    geom_dump("gs.src.glsl", gs_src);
    char *main_renamed = rename_main(gs_base, "mgl_gs_main");
    char *stripped_src = NULL;
    char *removed_src = NULL;
    GeomPlainUni uni[MGL_GEOM_MAX_VARYINGS];
    GeomDubMember dub[MGL_GEOM_MAX_VARYINGS];
    int nuni_ = 0, ndub = 0;
    size_t dub_at = (size_t)-1;
    bool dub_spliced = false;
    char *transformed = geom_transform_source(main_renamed ? main_renamed : gs_base, gp->in_verts,
                                              gl_in_name, sizeof(gl_in_name), NULL, NULL);
    free(main_renamed);
    free(gs_expanded);
    char *nover = NULL;
    SB b;
    GLuint out_type_code = gp->out_type == GL_POINTS ? 0 : gp->out_type == GL_LINE_STRIP ? 1 : 2;
    int version = 460;
    const char *vstr = strstr(gs_src, "#version");

    if (vstr) sscanf(vstr, "#version %d", &version);
    if (version < 450) version = 450;

    if (!transformed)
    {
        mgl_append_log(log, "ERROR: geometry: source transform failed\n");
        return NULL;
    }
    geom_dump("gs.transformed.glsl", transformed);
    {
        /* a shader's input and output interface blocks may share a block NAME ("in B {..}
         * user_in[]; out B {..} user_out;"): the per-statement emitter writes a "struct B"
         * for each, and GLSL rejects the redefinition.  Drop the duplicates (keeping the
         * first definition; the bodies are member-identical when the stages match). */
        char *dedup = geom_dedupe_iface_structs(transformed);
        if (dedup)
        {
            free(transformed);
            transformed = dedup;
        }
    }
    nover = strip_version(transformed);
    free(transformed);
    {
        char *blanked = geom_blank_comments(nover);
        free(nover);
        nover = blanked;
    }
    /* plain (default-block) uniforms are re-declared below as one block that matches the
     * program's default-uniform buffer (see geom_emit_dub_block) */
    stripped_src = geom_strip_plain_uniforms_at(nover, uni, &nuni_, MGL_GEOM_MAX_VARYINGS, &removed_src, &dub_at);
    if (!stripped_src) { free(nover); return NULL; }
    if (nuni_ && geom_dub_layout((SpvModule *)gs_m, dub, &ndub, MGL_GEOM_MAX_VARYINGS))
    {
        /* splice the re-declared block in at the first removed declaration, now: the
         * leading-preprocessor hoist below shifts the text and would invalidate the
         * offset.  A member's struct type is lexically declared by here, and any macro
         * its array size uses is #defined before the uniform as well. */
        size_t at = (dub_at == (size_t)-1 || dub_at > strlen(stripped_src)) ? strlen(stripped_src) : dub_at;
        SB t;
        memset(&t, 0, sizeof(t));
        sb_put(&t, "%.*s", (int)at, stripped_src);
        geom_emit_dub_block(&t, uni, nuni_, dub, ndub);
        sb_put(&t, "%s", stripped_src + at);
        free(stripped_src);
        stripped_src = t.s ? t.s : xstrdup("");
        dub_spliced = true;
    }
    if (nuni_) { free(nover); nover = stripped_src; }
    else free(stripped_src);

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version %d core\n", version);

    /* the source's LEADING preprocessor lines must come first: a re-declared uniform's
     * array size may be a macro (uniform ivec4 a[NUMBER];) that would otherwise not be
     * defined yet where the default-uniform block is emitted */
    {
        size_t nlen = strlen(nover), h = 0, last_pp = 0;

        while (h < nlen)
        {
            size_t line = h, s = h;

            while (line < nlen && nover[line] != '\n') line++;
            while (s < line && (nover[s] == ' ' || nover[s] == '\t')) s++;
            if (s < line && nover[s] == '#' && s + 1 < line)
            {
                /* only self-contained directives may be hoisted: a conditional block
                 * (#ifdef/#else/#endif) must stay intact around the code it guards */
                char d[16];
                size_t t = s + 1, n = 0;

                while (t < line && n < sizeof(d) - 1 &&
                       (isalnum((unsigned char)nover[t]) || nover[t] == '_'))
                    d[n++] = nover[t++];
                d[n] = 0;
                if (!strcmp(d, "define") || !strcmp(d, "pragma") || !strcmp(d, "extension"))
                    last_pp = line + 1;
                else break;
            }
            else if (s < line)
            {
                /* a one-line global precision statement is also order-independent and
                 * commonly sits between the #defines and the uniforms */
                if (!strncmp(nover + s, "precision", 9) && line > s + 2 &&
                    nover[line - 1] == ';')
                    last_pp = line + 1;
                else break;              /* first real statement */
            }
            h = line + 1;                          /* blank lines continue the scan */
        }
        h = last_pp;
        if (h)
        {
            sb_put(&b, "%.*s", (int)h, nover);
            char *rest = xstrdup(nover + h);
            free(nover);
            nover = rest;
        }
    }

    /* re-declare the stripped plain uniforms (or put them back if the program's
     * default-uniform block could not be read) — the block was already spliced in at the
     * first removed declaration (dub_spliced), so only the fallback is left here */
    if (!dub_spliced && removed_src)
    {
        sb_put(&b, "%s", removed_src);
        nuni_ = 0;
    }
    free(removed_src);
    removed_src = NULL;
    sb_put(&b, "layout(local_size_x = 32) in;\n\n");
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsPrimBlock { uint mgl_prim[]; };\n", MGL_GEOM_SSBO_PRIM);
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsVsOutBlock { uint mgl_vsout[]; };\n", MGL_GEOM_SSBO_VSOUT);
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsHdrBlock { uint mgl_hdr[]; };\n", MGL_GEOM_SSBO_HDR);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsOutBlock { uint mgl_out[]; };\n", MGL_GEOM_SSBO_OUTV);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsIdxBlock { uint mgl_idx[]; };\n", MGL_GEOM_SSBO_IDX);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsCtlBlock { uint mgl_ctl[]; };\n", MGL_GEOM_SSBO_CTL);
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsCntBlock { uint mgl_cnt[]; };\n", MGL_GEOM_SSBO_CNT);
    /* the emitted strips land in a per-thread scratch segment; a scan + copy pass (see
     * mgl_geom_lscan / mgl_geom_lcopy) moves them into the raster list in primitive
     * order, so the rasterization order is the emission order the GL spec requires
     * instead of the order of a pack of racing atomicAdds.  The binding starts at the
     * scratch: [scratch hdr[4] words | counts | scanned bases] */
    sb_put(&b, "layout(std430, binding = %d) buffer MGLGsLscrBlock { uint mgl_lscr[]; };\n\n", MGL_GEOM_SSBO_LSCR);

    sb_put(&b, "struct MGLGsInVert\n{\n");
    sb_put(&b, "    vec4 mgl_gs_pos;\n");
    if (gp->in_has_psize) sb_put(&b, "    float mgl_gs_psize;\n");
    if (gp->in_has_clip) sb_put(&b, "    float mgl_gs_clip[%u];\n", gp->in_clip_size);
    if (gp->in_has_cull) sb_put(&b, "    float mgl_gs_cull[%u];\n", gp->in_clip_size);
    sb_put(&b, "};\n");
    sb_put(&b, "MGLGsInVert mgl_in[%u];\n", gp->in_verts);
    if (gl_in_name[0] && strcmp(gl_in_name, "gl_in"))
        sb_put(&b, "#define %s mgl_in\n", gl_in_name);
    sb_put(&b, "\n");

    sb_put(&b, "vec4 mgl_gs_pos;\n");
    if (gp->out_has_psize) sb_put(&b, "float mgl_gs_psize;\n");
    if (gp->out_has_layer) sb_put(&b, "int mgl_gs_layer;\n");
    if (gp->out_has_vp) sb_put(&b, "int mgl_gs_vp;\n");
    if (gp->out_has_prim) sb_put(&b, "int mgl_gs_prim_out;\n");
    if (gp->out_has_clip) sb_put(&b, "float mgl_gs_clip[%u];\n", gp->out_clip_size);
    if (gp->out_has_cull) sb_put(&b, "float mgl_gs_cull[%u];\n", gp->out_clip_size);
    sb_put(&b, "int mgl_gs_prim_in;\nint mgl_gs_invoc;\n");
    sb_put(&b, "uint mgl_written, mgl_strip_n, mgl_seq, mgl_tag, mgl_stream, mgl_slot_base;\n");
    /* per-stream captured-primitive ordinals: transform feedback captures every stream, and a
     * record's ordinal (header bits 9..23) must count the primitives of ITS OWN stream that
     * were closed before it (KHR-GL46.transform_feedback.draw_xfb_stream*) */
    sb_put(&b, "uint mgl_pt[4];\nuint mgl_cl[4];\n");
    sb_put(&b, "uint mgl_t = 0u;          /* thread (primitive/invocation) index - the scratch segment key */\n");
    sb_put(&b, "uint mgl_lcnt = 0u;\n");
    /* the vertices of the strip in emit order: threads interleave (one per primitive per
     * invocation), so the reserved slots are not consecutive */
    sb_put(&b, "uint mgl_strip_v[%u];\n\n", gp->max_vertices ? gp->max_vertices : 1);


    /* the rewritten geometry shader may call the emit helpers: declare them first */
    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s);\n");
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s);\n");
    sb_put(&b, "void mgl_gs_emit_vertex();\n");
    sb_put(&b, "void mgl_gs_end_primitive();\n\n");

    sb_put(&b, "%s\n\n", nover);
    free(nover);

    /* the rewritten geometry shader declares its inputs and outputs and defines the entry
     * point; the emulation helpers must be visible to it */
    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s);\n");
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s);\n");
    sb_put(&b, "void mgl_gs_emit_vertex();\n");
    sb_put(&b, "void mgl_gs_end_primitive();\n\n");
    /* load one input primitive */
    sb_put(&b, "void mgl_load_inputs(uint mgl_pi, uint mgl_inst)\n{\n");
    sb_put(&b, "    for (uint mgl_k = 0u; mgl_k < %uu; mgl_k++) {\n", gp->in_verts);
    sb_put(&b, "        /* an affine direct draw (hdr[21]) derives the window's vertex ids\n"
               "         * arithmetically instead of loading the assembly's primitive slots.\n"
               "         * hdr[20] codes the window pattern: 1 = plain stride-1 window\n"
               "         * [p, p+1, ..]; 2/3 = triangle strip whose ODD windows swap the leading\n"
               "         * vertices (the GL strip winding / provoking-vertex rule: even p ->\n"
               "         * [p, p+1, p+2]; odd p, last-vertex convention -> [p+1, p, p+2]; odd p,\n"
               "         * first-vertex -> [p+2, p, p+1]); >= 4 = disjoint windows of (code - 3)\n"
               "         * vertices. */\n");
    sb_put(&b, "        uint mgl_s = mgl_hdr[20];\n");
    sb_put(&b, "        uint mgl_w = mgl_hdr[19] + mgl_pi * ((mgl_s <= 3u) ? 1u : (mgl_s - 3u));\n");
    sb_put(&b, "        uint mgl_vid;\n");
    sb_put(&b, "        if (mgl_hdr[21] == 0u) mgl_vid = mgl_prim[mgl_pi * %uu + mgl_k];\n", gp->in_verts);
    sb_put(&b, "        else if (mgl_s == 2u && (mgl_pi & 1u) != 0u)\n"
               "            mgl_vid = (mgl_k == 0u) ? (mgl_w + 1u) : ((mgl_k == 1u) ? mgl_w : (mgl_w + 2u));\n");
    sb_put(&b, "        else if (mgl_s == 3u && (mgl_pi & 1u) != 0u)\n"
               "            mgl_vid = (mgl_k == 0u) ? (mgl_w + 2u) : (mgl_w + mgl_k - 1u);\n");
    sb_put(&b, "        else mgl_vid = mgl_w + mgl_k;\n");
    sb_put(&b, "        uint mgl_r = (mgl_vid - mgl_hdr[12] + mgl_inst * mgl_hdr[6]) * %uu;\n",
           gp->in_words);
    if (gp->in_has_pos)
        sb_put(&b, "        mgl_in[mgl_k].mgl_gs_pos = vec4(uintBitsToFloat(mgl_vsout[mgl_r + %u]), uintBitsToFloat(mgl_vsout[mgl_r + %u]),"
                   " uintBitsToFloat(mgl_vsout[mgl_r + %u]), uintBitsToFloat(mgl_vsout[mgl_r + %u]));\n",
               gp->in_pos_off + 0, gp->in_pos_off + 1, gp->in_pos_off + 2, gp->in_pos_off + 3);
    if (gp->in_has_psize)
        sb_put(&b, "        mgl_in[mgl_k].mgl_gs_psize = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", gp->in_psize_off);
    if (gp->in_has_clip)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "        mgl_in[mgl_k].mgl_gs_clip[%u] = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", i, gp->in_clip_off + i);
    if (gp->in_has_cull)
        for (GLuint i = 0; i < gp->in_clip_size; i++)
            sb_put(&b, "        mgl_in[mgl_k].mgl_gs_cull[%u] = uintBitsToFloat(mgl_vsout[mgl_r + %u]);\n", i, gp->in_cull_off + i);
    for (int i = 0; i < gp->n_in; i++)
    {
        sb_put(&b, "        ");
        gen_varying_load(&b, &gp->in_v[i], "mgl_vsout[mgl_r", "]", "mgl_k");
    }
    sb_put(&b, "    }\n}\n\n");

    /* store one output vertex */
    sb_put(&b, "void mgl_store_output(uint slot)\n{\n    uint o = slot * %uu;\n", gp->out_words);
    /* the tag word: [0..7] the strip length (stamped at the close), [8] the close flag,
     * [9..23] the primitives the thread closed before this strip, [24..31] the stream.
     * The pack (prefix + scatter) is the only reader. */
    sb_put(&b, "    mgl_out[o + 0u] = mgl_tag;\n    mgl_out[o + 1u] = (mgl_stream << 24u);\n");
    if (gp->out_has_pos)
    {
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.x);\n", gp->out_pos_off + 0);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.y);\n", gp->out_pos_off + 1);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.z);\n", gp->out_pos_off + 2);
        sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_pos.w);\n", gp->out_pos_off + 3);
    }
    if (gp->out_has_psize) sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_psize);\n", gp->out_psize_off);
    if (gp->out_has_layer) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_layer);\n", gp->out_layer_off);
    if (gp->out_has_vp) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_vp);\n", gp->out_vp_off);
    if (gp->out_has_prim) sb_put(&b, "    mgl_out[o + %uu] = uint(mgl_gs_prim_out);\n", gp->out_prim_off);
    if (gp->out_has_clip)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_clip[%u]);\n", gp->out_clip_off + i, i);
    if (gp->out_has_cull)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    mgl_out[o + %uu] = floatBitsToUint(mgl_gs_cull[%u]);\n", gp->out_cull_off + i, i);
    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char src[192];
        gen_varying_store(&b, v, "mgl_out", "o", geom_var_expr(src, sizeof(src), v));
    }
    sb_put(&b, "}\n\n");

    sb_put(&b, "void mgl_gs_emit_stream_vertex(uint s)\n{\n");
    sb_put(&b, "    mgl_stream = s;\n");
    sb_put(&b, "    if (mgl_written >= mgl_hdr[5]) { mgl_ctl[3] = 1u; return; }\n");
    sb_put(&b, "    uint slot = mgl_slot_base + mgl_seq;\n");
    sb_put(&b, "    if (slot >= mgl_hdr[3]) { mgl_ctl[3] = 1u; return; }\n");
    sb_put(&b, "    mgl_store_output(slot);\n");
    sb_put(&b, "    mgl_strip_v[mgl_strip_n] = slot;\n");
    if (out_type_code == 0)
    {
        sb_put(&b, "    /* a point output makes every vertex its own primitive: flag the record as\n");
        sb_put(&b, "     * closing one (bit 8), stamp its length (1, bits 0..7) and the number of\n");
        sb_put(&b, "     * primitives the thread closed before it ON THE SAME STREAM (bits 9..23),\n");
        sb_put(&b, "     * by which the transform feedback pack indexes the captured primitives */\n");
        sb_put(&b, "    mgl_out[slot * %uu + 1u] |= (1u << 8u) | (mgl_pt[s] << 9u) | 1u;\n", gp->out_words);
        sb_put(&b, "    mgl_pt[s]++;\n");
    }
    sb_put(&b, "    mgl_strip_n++; mgl_written++; mgl_seq++;\n");
    if (out_type_code == 0)
    {
        sb_put(&b, "    atomicAdd(mgl_ctl[13u + s], 1u);\n");
        sb_put(&b, "    atomicAdd(mgl_ctl[17u + s], 1u);\n");
    }
    sb_put(&b, "}\n\n");
    if (gp->out_has_cull)
    {
        /* the whole-primitive cull-distance test on emitted records (see the caller):
         * a, b, c are record slots; b == a / c == a mark unused corners so the same
         * helper serves points, lines and triangles */
        sb_put(&b, "bool mgl_prim_culled(uint a, uint b, uint c)\n{\n");
        sb_put(&b, "    for (uint p = 0u; p < %uu; p++) {\n", gp->out_clip_size);
        sb_put(&b, "        if (uintBitsToFloat(mgl_out[a * %uu + %uu + p]) < 0.0 &&\n",
               gp->out_words, gp->out_cull_off);
        sb_put(&b, "            (b == a || uintBitsToFloat(mgl_out[b * %uu + %uu + p]) < 0.0) &&\n",
               gp->out_words, gp->out_cull_off);
        sb_put(&b, "            (c == a || uintBitsToFloat(mgl_out[c * %uu + %uu + p]) < 0.0)) return true;\n",
               gp->out_words, gp->out_cull_off);
        sb_put(&b, "    }\n    return false;\n}\n\n");
    }
    sb_put(&b, "void mgl_gs_end_stream_primitive(uint s)\n{\n");
    sb_put(&b, "    if (mgl_strip_n == 0u) return;\n");
    sb_put(&b, "    uint n = mgl_strip_n;\n");
    sb_put(&b, "    uint cnt = 0u;\n");
    if (out_type_code == 0)
        sb_put(&b, "    cnt = n;\n");
    else if (out_type_code == 1)
        sb_put(&b, "    if (n >= 2u) cnt = (n - 1u) * 2u;\n");
    else
        sb_put(&b, "    if (n >= 3u) cnt = (n - 2u) * 3u;\n");
    sb_put(&b, "    if (s == 0u && cnt > 0u) {\n");
    /* the strip's list entries go to this thread's scratch segment (mgl_t * lc + the
     * entries written so far); mgl_geom_lscan / mgl_geom_lcopy compact the segments into
     * mgl_idx in primitive order afterwards. Writing them straight into mgl_idx through
     * an atomicAdd reservation leaves the strip order - and so the rasterization order,
     * which the GL spec fixes as the emission order - to a thread race. */
    sb_put(&b, "        uint lc = 3u * mgl_hdr[5];\n");
    sb_put(&b, "        uint base = mgl_t * lc + mgl_lcnt;\n");
    sb_put(&b, "        if (base + cnt > mgl_t * lc + lc || base + cnt > mgl_hdr[4]) { mgl_ctl[3] = 1u; }\n");
    sb_put(&b, "        else {\n");
    if (gp->out_has_cull && !getenv("MGL_NO_KCULL"))
    {
        /* whole-primitive cull distance (GL 4.6 13.5.2): an emitted primitive is
         * dropped when EVERY vertex has gl_CullDistance[p] < 0 for the same p.  The
         * test runs per output primitive (a strip decomposes), on the records the
         * kernel itself wrote; unwritten entries are 0.0 (reset at every
         * mgl_gs_emit_stream_vertex) and never cull.  Only the raster index list is
         * affected - the records (and so transform feedback) keep every vertex. */
        sb_put(&b, "            uint w = 0u;\n");
        if (out_type_code == 0)
        {
            sb_put(&b, "            for (uint i = 0u; i < n; i++)\n");
            sb_put(&b, "                if (!mgl_prim_culled(mgl_strip_v[i], mgl_strip_v[i], mgl_strip_v[i]))\n"
                       "                    mgl_lscr[base + w++] = mgl_strip_v[i];\n");
        }
        else if (out_type_code == 1)
        {
            sb_put(&b, "            for (uint i = 0u; i + 1u < n; i++)\n");
            sb_put(&b, "                if (!mgl_prim_culled(mgl_strip_v[i], mgl_strip_v[i + 1u], mgl_strip_v[i])) {\n"
                       "                    mgl_lscr[base + w] = mgl_strip_v[i];"
                       " mgl_lscr[base + w + 1u] = mgl_strip_v[i + 1u]; w += 2u;\n");
            sb_put(&b, "                }\n");
        }
        else
        {
            sb_put(&b, "            for (uint i = 0u; i + 2u < n; i++) {\n");
            sb_put(&b, "                uint a = mgl_strip_v[i], b2 = mgl_strip_v[i + 1u], c = mgl_strip_v[i + 2u];\n");
            sb_put(&b, "                if ((i & 1u) != 0u) { uint t = a; a = b2; b2 = t; }\n");
            sb_put(&b, "                if (!mgl_prim_culled(a, b2, c)) {\n");
            sb_put(&b, "                    mgl_lscr[base + w] = a; mgl_lscr[base + w + 1u] = b2;"
                       " mgl_lscr[base + w + 2u] = c; w += 3u;\n");
            sb_put(&b, "                }\n");
            sb_put(&b, "            }\n");
        }
        sb_put(&b, "            cnt = w;\n");
    }
    else if (out_type_code == 0)
    {
        sb_put(&b, "            for (uint i = 0u; i < n; i++) mgl_lscr[base + i] = mgl_strip_v[i];\n");
    }
    else if (out_type_code == 1)
    {
        sb_put(&b, "            for (uint i = 0u; i + 1u < n; i++) { mgl_lscr[base + i * 2u] = mgl_strip_v[i];"
                   " mgl_lscr[base + i * 2u + 1u] = mgl_strip_v[i + 1u]; }\n");
    }
    else
    {
        sb_put(&b, "            for (uint i = 0u; i + 2u < n; i++) {\n");
        sb_put(&b, "                uint a = mgl_strip_v[i], b2 = mgl_strip_v[i + 1u], c = mgl_strip_v[i + 2u];\n");
        sb_put(&b, "                if ((i & 1u) == 0u) { mgl_lscr[base + i * 3u] = a; mgl_lscr[base + i * 3u + 1u] = b2; mgl_lscr[base + i * 3u + 2u] = c; }\n");
        sb_put(&b, "                else { mgl_lscr[base + i * 3u] = b2; mgl_lscr[base + i * 3u + 1u] = a; mgl_lscr[base + i * 3u + 2u] = c; }\n");
        sb_put(&b, "            }\n");
    }
    sb_put(&b, "            mgl_lcnt += cnt;\n");
    sb_put(&b, "        }\n    }\n");
    if (out_type_code != 0)
    {
        sb_put(&b, "    /* close the strip: flag its last record for the transform feedback pack\n");
        sb_put(&b, "     * (bit 8), stamp the strip's length (bits 0..7) and the number of primitives\n");
        sb_put(&b, "     * the thread closed before it ON THIS STREAM (bits 9..23) - a strip of n\n");
        sb_put(&b, "     * vertices decomposes into n-1 line / n-2 triangle captured primitives, so\n");
        sb_put(&b, "     * one close can stand for several.  A strip shorter than one output primitive\n");
        sb_put(&b, "     * is degenerate: nothing can be captured from it, so it gets no flag and\n");
        sb_put(&b, "     * consumes no ordinal, exactly what the pack counts.  Every stream is\n");
        sb_put(&b, "     * captured, so the flag is not limited to the rasterization stream. */\n");
        if (out_type_code == 1)
            sb_put(&b, "    if (mgl_strip_n >= 2u) {\n");
        else
            sb_put(&b, "    if (mgl_strip_n >= 3u) {\n");
        sb_put(&b, "        mgl_out[mgl_strip_v[mgl_strip_n - 1u] * %uu + 1u] |=\n", gp->out_words);
        sb_put(&b, "            (1u << 8u) | (mgl_cl[s] << 9u) | mgl_strip_n;\n");
        sb_put(&b, "        mgl_cl[s] += mgl_strip_n - %uu + 1u;\n", (unsigned)(out_type_code == 1 ? 2 : 3));
        sb_put(&b, "    }\n");
        sb_put(&b, "    if (mgl_strip_n > 0u) atomicAdd(mgl_ctl[13u + s], 1u);\n");
        /* PRIMITIVES_GENERATED counts what the rasterizer / transform feedback consume:
         * a strip output decomposes into n-1 line / n-2 triangle primitives */
        if (out_type_code == 1)
            sb_put(&b, "    if (mgl_strip_n >= 2u) atomicAdd(mgl_ctl[17u + s], mgl_strip_n - 1u);\n");
        else
            sb_put(&b, "    if (mgl_strip_n >= 3u) atomicAdd(mgl_ctl[17u + s], mgl_strip_n - 2u);\n");
    }
    else
        sb_put(&b, "    /* points: every emitted vertex was counted in the emit */\n");
    sb_put(&b, "    mgl_strip_n = 0u;\n");
    sb_put(&b, "}\n\n");
    sb_put(&b, "void mgl_gs_emit_vertex() { mgl_gs_emit_stream_vertex(0u); }\n");
    sb_put(&b, "void mgl_gs_end_primitive() { mgl_gs_end_stream_primitive(0u); }\n\n");

    sb_put(&b, "void main()\n{\n");
    sb_put(&b, "    mgl_t = gl_GlobalInvocationID.x;\n");
    /* the per-thread counters are reset up front: a thread that bails out below must
     * still publish a zero so the scan/copy passes and the xfb prefix never read a stale
     * count from an earlier draw that used the same scratch slot */
    sb_put(&b, "    mgl_cnt[mgl_t] = 0u;\n");
    sb_put(&b, "    mgl_lscr[mgl_hdr[4] + mgl_t] = 0u;\n");
    sb_put(&b, "    uint mgl_invocations = mgl_hdr[2];\n");
    sb_put(&b, "    if (mgl_invocations == 0u) return;\n");
    sb_put(&b, "    uint mgl_p = mgl_t / mgl_invocations;\n");
    sb_put(&b, "    uint mgl_inv = mgl_t - mgl_p * mgl_invocations;\n");
    sb_put(&b, "    if (mgl_p >= mgl_hdr[0] * mgl_hdr[1]) return;\n");
    /* GEOMETRY_SHADER_INVOCATIONS: one thread = one (input primitive, invocation)
     * pair — the exact count for indirect draws too, folded into the query
     * accumulator word 8 by mgl_geom_qacc (queries.c) */
    sb_put(&b, "    atomicAdd(mgl_ctl[21u], 1u);\n");
    sb_put(&b, "    uint mgl_inst = mgl_p / mgl_hdr[0];\n");
    sb_put(&b, "    uint mgl_pi = mgl_p - mgl_inst * mgl_hdr[0];\n");
    sb_put(&b, "    mgl_load_inputs(mgl_pi, mgl_inst);\n");
    sb_put(&b, "    mgl_gs_prim_in = int(mgl_pi);\n    mgl_gs_invoc = int(mgl_inv);\n");
    sb_put(&b, "    mgl_tag = mgl_p; mgl_seq = 0u; mgl_written = 0u; mgl_strip_n = 0u; mgl_stream = 0u;\n");
    sb_put(&b, "    mgl_slot_base = mgl_t * mgl_hdr[5];\n");
    /* thread-local arrays are NOT zero-initialised in MSL: the per-stream ordinals must
     * start at 0, or the captured primitive indices of every stream but the first are garbage */
    sb_put(&b, "    mgl_pt[0]=0u; mgl_pt[1]=0u; mgl_pt[2]=0u; mgl_pt[3]=0u;\n");
    sb_put(&b, "    mgl_cl[0]=0u; mgl_cl[1]=0u; mgl_cl[2]=0u; mgl_cl[3]=0u;\n");
    sb_put(&b, "    mgl_gs_pos = vec4(0.0);\n");
    if (gp->out_has_psize) sb_put(&b, "    mgl_gs_psize = 1.0;\n");
    if (gp->out_has_layer) sb_put(&b, "    mgl_gs_layer = 0;\n");
    if (gp->out_has_vp) sb_put(&b, "    mgl_gs_vp = 0;\n");
    if (gp->out_has_prim) sb_put(&b, "    mgl_gs_prim_out = int(mgl_pi);\n");
    if (gp->out_has_clip) sb_put(&b, "    for (int i = 0; i < %u; i++) mgl_gs_clip[i] = 0.0;\n", gp->out_clip_size);
    if (gp->out_has_cull) sb_put(&b, "    for (int i = 0; i < %u; i++) mgl_gs_cull[i] = 0.0;\n", gp->out_clip_size);
    for (int i = 0; i < gp->n_out; i++)
    {
        /* a named output block member is zero-initialized through the instance ("b.c");
         * a global instance in GLSL is zero-initialized anyway, this is belt and braces.
         * r102: an array-of-blocks instance is initialized element-wise
         * ("goku[e].c") — the store/clear expression is shared with mgl_store_output */
        char src[192];
        geom_var_expr(src, sizeof(src), &gp->out_v[i]);
        sb_put(&b, "    %s%s = %s(0);\n", src, (gp->out_v[i].elems > 1 || gp->out_v[i].arr) ? "[0]" : "",
               glsl_type_name(gp->out_v[i].base, gp->out_v[i].rows, gp->out_v[i].cols));
    }
    sb_put(&b, "    mgl_gs_main();\n");
    sb_put(&b, "    mgl_gs_end_stream_primitive(mgl_stream);\n");
    sb_put(&b, "    mgl_cnt[mgl_t] = mgl_written;\n");
    sb_put(&b, "    mgl_lscr[mgl_hdr[4] + mgl_t] = mgl_lcnt;\n");
    sb_put(&b, "    if (mgl_written != 0u) atomicAdd(mgl_ctl[0], mgl_written);\n");
    sb_put(&b, "}\n\n");


    return sb_take(&b);
}

/* ---------------- 3. pass-through raster vertex function ---------------- */

static char *geom_gen_draw_glsl(GLMContext ctx, MGLGeomProgram *gp, char **log)
{
    SB b;
    GLuint next_loc = 0;

    memset(&b, 0, sizeof(b));
    sb_put(&b, "#version 460 core\n\n");
    /* the record is read as a flat word array: the geometry kernel writes it with exactly
     * this layout, so no std430 struct padding can get in the way */
    sb_put(&b, "layout(std430, binding = %d) readonly buffer MGLGsDrawBlock { uint mgl_draw[]; };\n\n",
           MGL_GEOM_SSBO_DRAW);

    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char arr[32] = "";
        char name[160];
        if (v->elems > 1) snprintf(arr, sizeof(arr), "[%u]", v->elems);
        /* a named output block member is emitted as a plain varying of its own: the
         * raster side links to the fragment stage's input block by location, so the
         * member's name is irrelevant here (and the instance name would collide) */
        if (geom_var_dotted(v))
            snprintf(name, sizeof(name), "mgl_ob%d", i);
        else
            snprintf(name, sizeof(name), "%s", v->name);
        if (v->base == GL_DOUBLE)
        {
            /* write the u32-slot representation the fragment stage was lowered
             * to (real double varyings cannot appear in the generated MSL) */
            char ty[32];
            GLuint slots = geom_double_iface(v->rows, v->cols, v->elems > 0 ? v->elems : 1, ty, sizeof(ty));
            snprintf(arr, sizeof(arr), "[%u]", slots);
            if (slots == 1) arr[0] = 0;
            sb_put(&b, "layout(location = %u) %s%s%s out %s %s%s;\n", v->location,
                   v->flat ? "flat " : "", v->noperspective ? "noperspective " : "",
                   v->centroid ? "centroid " : "", ty, name, arr);
            if (v->location + slots > next_loc) next_loc = v->location + slots;
            continue;
        }
        if (v->base != GL_DOUBLE && v->cols > 1 && v->elems > 1)
        {
            /* a non-double ARRAY OF MATRICES cannot be a stage output ("MSL cannot
             * emit arrays-of-matrices in input and output variables"): emit one
             * matrix per element at the element's own locations — exactly the shape
             * the fragment stage's module was flattened to by flatten_io_matrix_arrays
             * (member <name>_<e> at location L + e*cols), and the link is by location
             * anyway.  dEQP-GL45-ES31 transform_feedback_varying.type.vertex_geo_
             * fragment.{array,whole_array}.mat3x4 (xfb of a GS-through matrix array). */
            for (GLuint e = 0; e < v->elems; e++)
                sb_put(&b, "layout(location = %u) %s%s%s out %s mgl_vm%d_%u;\n",
                       v->location + e * v->cols,
                       v->flat ? "flat " : "", v->noperspective ? "noperspective " : "",
                       v->centroid ? "centroid " : "",
                       glsl_type_name(v->base, v->rows, v->cols), i, e);
            if (v->location + v->elems * v->cols > next_loc)
                next_loc = v->location + v->elems * v->cols;
            continue;
        }
        sb_put(&b, "layout(location = %u) %s%s%s out %s %s%s;\n", v->location,
               v->flat ? "flat " : "", v->noperspective ? "noperspective " : "",
               v->centroid ? "centroid " : "", glsl_type_name(v->base, v->rows, v->cols), name, arr);
        if (v->location + v->elems * (v->cols ? v->cols : 1) > next_loc)
            next_loc = v->location + v->elems * (v->cols ? v->cols : 1);
    }
    if (gp->out_has_layer)
    {
        gp->layer_loc = next_loc++;
        sb_put(&b, "layout(location = %u) flat out uint mgl_out_layer;\n", gp->layer_loc);
    }
    if (gp->out_has_vp)
    {
        gp->vp_loc = next_loc++;
        sb_put(&b, "layout(location = %u) flat out uint mgl_out_vp;\n", gp->vp_loc);
    }
    if (gp->out_has_prim)
    {
        gp->prim_loc = next_loc++;
        sb_put(&b, "layout(location = %u) flat out uint mgl_out_prim;\n", gp->prim_loc);
    }

    /* re-declare gl_PerVertex with the distance arrays the main() below assigns
     * (glslang rejects assigning them without a redeclaration that carries them) */
    if ((gp->out_has_clip && gp->out_clip_size) || gp->out_has_cull)
    {
        sb_put(&b, "out gl_PerVertex\n{\n    vec4 gl_Position;\n");
        if (gp->out_has_psize) sb_put(&b, "    float gl_PointSize;\n");
        if (gp->out_has_clip && gp->out_clip_size)
            sb_put(&b, "    float gl_ClipDistance[%u];\n", gp->out_clip_size);
        if (gp->out_has_cull)
            sb_put(&b, "    float gl_CullDistance[%u];\n", gp->out_clip_size);
        sb_put(&b, "};\n");
    }
    sb_put(&b, "\nvoid main()\n{\n");
    sb_put(&b, "    uint r = uint(gl_VertexID) * %uu;\n", gp->out_words);
    sb_put(&b, "    gl_Position = vec4(uintBitsToFloat(mgl_draw[r + %u]), uintBitsToFloat(mgl_draw[r + %u]),"
               " uintBitsToFloat(mgl_draw[r + %u]), uintBitsToFloat(mgl_draw[r + %u]));\n",
           gp->out_pos_off + 0, gp->out_pos_off + 1, gp->out_pos_off + 2, gp->out_pos_off + 3);
    if (gp->out_has_psize)
        sb_put(&b, "    gl_PointSize = uintBitsToFloat(mgl_draw[r + %u]);\n", gp->out_psize_off);
    /* gl_ClipDistance re-exports the distances the stage produced: SPIRV-Cross turns the
     * builtin into [[clip_distance]] members (hardware clipping, exactly GL 13.5.1) and
     * flattened [[user(clipN)]] copies that a fragment stage reading the builtin matches
     * by attribute name; gl_CullDistance has no hardware form and only feeds the
     * fragment stage the same way (whole-primitive culling happened in the kernel) */
    if (gp->out_has_clip && gp->out_clip_size)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    gl_ClipDistance[%u] = %s ? uintBitsToFloat(mgl_draw[r + %u]) : 1.0e30;\n",
                   i, (gp->clip_planes >> i) & 1u ? "true" : "false", gp->out_clip_off + i);
    if (gp->out_has_cull)
        for (GLuint i = 0; i < gp->out_clip_size; i++)
            sb_put(&b, "    gl_CullDistance[%u] = uintBitsToFloat(mgl_draw[r + %u]);\n", i, gp->out_cull_off + i);
    if (gp->out_has_layer)
        sb_put(&b, "    mgl_out_layer = mgl_draw[r + %u];\n", gp->out_layer_off);
    if (gp->out_has_vp)
        sb_put(&b, "    mgl_out_vp = mgl_draw[r + %u];\n", gp->out_vp_off);
    if (gp->out_has_prim)
        sb_put(&b, "    mgl_out_prim = mgl_draw[r + %u];\n", gp->out_prim_off);
    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        char name[160];
        if (geom_var_dotted(v))
            snprintf(name, sizeof(name), "mgl_ob%d", i);
        else
            snprintf(name, sizeof(name), "%s", v->name);
        if (v->base == GL_DOUBLE)
        {
            /* fill the u32-slot representation the fp64 lowering reads: one uint4
             * slot per (element, column, row pair), as_type<ulong2>(slot).x = double
             * 2p, .y = double 2p+1 — the record itself stores doubles densely as
             * (lo,hi) u32 pairs */
            char ty[32];
            GLuint slots = geom_double_iface(v->rows, v->cols, v->elems > 0 ? v->elems : 1, ty, sizeof(ty));
            GLuint cs = (v->cols ? v->cols : 1) * ((v->rows + 1) / 2);   /* slots per element */
            GLuint ew = (v->cols ? v->cols : 1) * v->rows * 2;           /* record words per element */
            for (GLuint e = 0; e < (v->elems > 0 ? v->elems : 1u); e++)
            for (GLuint c = 0; c < (v->cols ? v->cols : 1u); c++)
            for (GLuint p = 0; p < ((v->rows + 1) / 2); p++)
            {
                GLuint base = v->offset + e * ew;
                GLuint w = base + (c * v->rows + 2 * p) * 2;
                char dst[192], one[96];
                if (v->rows == 1)
                {
                    /* scalar double: the uint2 slot the fp64 interface uses */
                    if (slots == 1)
                        snprintf(dst, sizeof(dst), "%s = uvec2(mgl_draw[r+%u], mgl_draw[r+%u]);", name, w, w + 1);
                    else
                        snprintf(dst, sizeof(dst), "%s[%u] = uvec2(mgl_draw[r+%u], mgl_draw[r+%u]);", name, e * cs + c, w, w + 1);
                    sb_put(&b, "    %s\n", dst);
                    continue;
                }
                if (2 * p + 1 < v->rows)
                    snprintf(one, sizeof(one), "uvec4(mgl_draw[r+%u], mgl_draw[r+%u], mgl_draw[r+%u], mgl_draw[r+%u])", w, w + 1, w + 2, w + 3);
                else
                    snprintf(one, sizeof(one), "uvec4(mgl_draw[r+%u], mgl_draw[r+%u], 0u, 0u)", w, w + 1);
                if (slots == 1)
                    snprintf(dst, sizeof(dst), "%s = %s;", name, one);
                else
                    snprintf(dst, sizeof(dst), "%s[%u] = %s;", name, e * cs + c * ((v->rows + 1) / 2) + p, one);
                sb_put(&b, "    %s\n", dst);
            }
            continue;
        }
        for (GLuint e = 0; e < (v->elems > 0 ? v->elems : 1u); e++)
        {
            char arr[32] = "";
            char dst[192];
            /* an array element's words are contiguous: cols * rows * base words each */
            GLuint base = v->offset + e * v->cols * v->rows * comp_words(v->base);
            /* the flattened matrix array writes its per-element output mgl_vm<i>_<e> */
            if (v->base != GL_DOUBLE && v->cols > 1 && v->elems > 1)
                snprintf(dst, sizeof(dst), "mgl_vm%d_%u", i, e);
            else
            {
                if (v->elems > 1) snprintf(arr, sizeof(arr), "[%u]", e);
                snprintf(dst, sizeof(dst), "%s%s", name, arr);
            }
            sb_put(&b, "    %s = ", dst);
            if (v->cols > 1)
            {
                char mname[64];
                snprintf(mname, sizeof(mname), "%s", glsl_type_name(v->base, v->rows, v->cols));
                sb_put(&b, "%s(", mname);
                for (GLuint c = 0; c < v->cols; c++)
                {
                    if (c) sb_put(&b, ", ");
                    sb_put(&b, "%s(", glsl_type_name(v->base, v->rows, 1));
                    for (GLuint r2 = 0; r2 < v->rows; r2++)
                    {
                        GLuint w = base + (c * v->rows + r2) * comp_words(v->base);
                        if (r2) sb_put(&b, ", ");
                        if (v->base == GL_FLOAT) sb_put(&b, "uintBitsToFloat(mgl_draw[r + %u])", w);
                        else if (v->base == GL_DOUBLE) sb_put(&b, "packDouble2x32(uvec2(mgl_draw[r + %u], mgl_draw[r + %u]))", w, w + 1);
                        else sb_put(&b, "%s(mgl_draw[r + %u])", v->base == GL_INT ? "int" : "uint", w);
                    }
                    sb_put(&b, ")");
                }
                sb_put(&b, ");\n");
            }
            else if (v->rows == 1 && v->base != GL_DOUBLE)
            {
                GLuint w = base;
                sb_put(&b, "%s(mgl_draw[r + %u]);\n", v->base == GL_FLOAT ? "uintBitsToFloat" :
                           v->base == GL_BOOL ? "uint" : (v->base == GL_INT ? "int" : "uint"), w);
            }
            else
            {
                sb_put(&b, "%s(", glsl_type_name(v->base, v->rows, 1));
                for (GLuint r2 = 0; r2 < v->rows; r2++)
                {
                    GLuint w = base + r2 * comp_words(v->base);
                    if (r2) sb_put(&b, ", ");
                    if (v->base == GL_FLOAT) sb_put(&b, "uintBitsToFloat(mgl_draw[r + %u])", w);
                    else if (v->base == GL_DOUBLE) sb_put(&b, "packDouble2x32(uvec2(mgl_draw[r + %u], mgl_draw[r + %u]))", w, w + 1);
                    else sb_put(&b, "%s(mgl_draw[r + %u])", v->base == GL_INT ? "int" : "uint", w);
                }
                sb_put(&b, ");\n");
            }
        }
    }
    sb_put(&b, "}\n");

    return sb_take(&b);
}

/* ------------------------------------------------------------------ link */

/* assign the word offsets of the emulation records (shared by the capture variant, the
 * geometry kernel and the pass-through vertex function) */
static void geom_layout(MGLGeomProgram *gp)
{
    int i;

    /* input record layout */
    {
        GLuint off = 0;
        gp->in_pos_off = off; off += 4;
        if (gp->in_has_psize) { gp->in_psize_off = off; off += 1; }
        if (gp->in_has_clip) { gp->in_clip_off = off; off += gp->in_clip_size; }
        if (gp->in_has_cull) { gp->in_cull_off = off; off += gp->in_clip_size; }
        for (i = 0; i < gp->n_in; i++) { gp->in_v[i].offset = off; off += gp->in_v[i].words; }
        gp->in_words = off;
    }
    /* output record layout */
    {
        GLuint off = 2;     /* tag + sequence */
        gp->out_pos_off = off; off += 4;
        if (gp->out_has_psize) { gp->out_psize_off = off; off += 1; }
        if (gp->out_has_layer) { gp->out_layer_off = off; off += 1; }
        if (gp->out_has_vp) { gp->out_vp_off = off; off += 1; }
        if (gp->out_has_prim) { gp->out_prim_off = off; off += 1; }
        if (gp->out_has_clip) { gp->out_clip_off = off; off += gp->out_clip_size; }
        if (gp->out_has_cull) { gp->out_cull_off = off; off += gp->out_clip_size; }
        for (i = 0; i < gp->n_out; i++) { gp->out_v[i].offset = off; off += gp->out_v[i].words; }
        gp->out_words = off;
    }
}

static const char *geom_gs_source(GLMContext ctx, MGLProgramObj *obj)
{
    for (GLuint i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (s && s->glm_type == _GEOMETRY_SHADER && s->src && s->src[0])
            /* introspect: feed the kernel the subroutine-rewritten source (glslang
             * cannot compile GLSL subroutines), as the program's own compile does */
            return mgl_introspect_shader_source(ctx, s);
    }
    return NULL;
}

static const char *geom_vs_source(GLMContext ctx, MGLProgramObj *obj)
{
    for (GLuint i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (s && s->glm_type == _VERTEX_SHADER && s->src && s->src[0])
            /* introspect: glslang cannot compile GLSL subroutines / the
             * #extension GL_ARB_shader_subroutine directive — feed the capture
             * variant the rewritten source, like the program's own compile */
            return mgl_introspect_shader_source(ctx, s);
    }
    return NULL;
}

/* the same for any attached stage (the TES source of a tessellation program: the
 * cull emulation reflects the evaluation stage's declared distance-array sizes) */
static const char *geom_stage_source(GLMContext ctx, MGLProgramObj *obj, int stage)
{
    if (stage == _VERTEX_SHADER) return geom_vs_source(ctx, obj);
    for (GLuint i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (s && s->glm_type == stage && s->src && s->src[0])
            return mgl_introspect_shader_source(ctx, s);
    }
    return NULL;
}

static GLuint geom_declared_clip_words(const char *src, const char *name)
{    /* the largest explicitly declared array size of gl_ClipDistance / gl_CullDistance */
    const char *p = src;
    GLuint best = 0;
    while ((p = strstr(p, name)))
    {
        const char *b = p + strlen(name);
        while (*b && isspace((unsigned char)*b)) b++;
        if (*b == '[')
        {
            unsigned v = 0;
            if (sscanf(b + 1, "%u", &v) == 1 && v > best) best = v;
        }
        p += strlen(name);
    }
    return best;
}

/* build the emulation state for one program's geometry stage; a failure here fails the link */
/* match every geometry-stage input varying against the vertex stage outputs of `vs_program`
 * and record the producing expression (the name the capture variant stores).  Returns false
 * with a message in `log` when an input is not written or its type differs. */
static bool geom_derive_exprs(MGLGeomProgram *gp, Program *vs_program, char **log)
{
    MGLGeomVar vs_out[MGL_GEOM_MAX_VARYINGS];
    int n_vs_out = 0;
    SpvModule *vm = NULL;
    const char *e2 = NULL;
    bool ok = false;

    if (!vs_program->spirv[_VERTEX_SHADER].ir)
    {
        mgl_append_log(log, "ERROR: geometry: no vertex stage to reflect the inputs from\n");
        return false;
    }
    vm = spv_parse(vs_program->spirv[_VERTEX_SHADER].ir, vs_program->spirv[_VERTEX_SHADER].size, &e2);
    if (!vm || !geom_reflect_stage(vm, false, vs_out, &n_vs_out, log))
    {
        mgl_append_log(log, "ERROR: geometry: could not reflect the vertex stage outputs\n");
        goto done;
    }
    for (int i = 0; i < gp->n_in; i++)
    {
        int j = geom_find_location(vs_out, n_vs_out, gp->in_v[i].location);
        if (j >= 0 && !geom_var_types_match(&vs_out[j], &gp->in_v[i]))
        {
            /* separately-linked stages (pipeline) assign automatic locations by their own
             * declaration order: try the canonical name+type pairing before failing */
            int k = geom_find_name(vs_out, n_vs_out, gp->in_v[i].name);
            if (k >= 0 && geom_var_types_match(&vs_out[k], &gp->in_v[i])) j = k;
        }
        if (j < 0)
        {
            int k = geom_find_name(vs_out, n_vs_out, gp->in_v[i].name);
            if (k >= 0) j = k;      /* same automatic-location caveat, unseen slot */
        }
        if (j < 0)
        {
            /* GL leaves such an input undefined (its value may be anything): keep the
             * link valid — the record slot exists, the capture pass just never writes it */
            mgl_append_log(log, "WARNING: geometry: input '%s' at location %u is not written by the vertex stage\n",
                           gp->in_v[i].name, gp->in_v[i].location);
            gp->in_v[i].expr[0] = 0;
            continue;
        }
        if (vs_out[j].rows != gp->in_v[i].rows || vs_out[j].cols != gp->in_v[i].cols ||
            vs_out[j].base != gp->in_v[i].base || vs_out[j].elems != gp->in_v[i].elems)
        {
            mgl_append_log(log, "ERROR: geometry: type mismatch for '%s' (location %u)\n",
                           gp->in_v[i].name, gp->in_v[i].location);
            if (getenv("MGL_GEOM_TRACE"))
                fprintf(stderr, "MGL geom trace: mismatch at loc %u: GS(%s base=%u r=%u c=%u e=%u) "
                                "VS(%s base=%u r=%u c=%u e=%u)\n",
                        gp->in_v[i].location, gp->in_v[i].name, gp->in_v[i].base, gp->in_v[i].rows,
                        gp->in_v[i].cols, gp->in_v[i].elems,
                        vs_out[j].name, vs_out[j].base, vs_out[j].rows, vs_out[j].cols, vs_out[j].elems);
            goto done;
        }
        snprintf(gp->in_v[i].expr, sizeof(gp->in_v[i].expr), "%s%s%s", vs_out[j].name,
                 vs_out[j].member[0] ? "." : "", vs_out[j].member);
    }
    ok = true;

done:
    spv_free(vm);
    return ok;
}

/* enumerate a stage module's atomic-counter declarations: names, SPIR-V offsets and GL
 * buffer binding points (glslang emits ONE AtomicCounter variable per declaration, an
 * array declaration is one variable of OpTypeArray).  Returns the total number of
 * declarations in the module (filling at most `max`); *n_counters / *n_buffers report
 * the counters used (array elements count individually) and the distinct buffers. */
/* per-dimension sizes of an atomic-counter (or any) declaration, OUTERMOST first:
 * walks the OpTypeArray chain through pointer types */
static int spv_array_dims(SpvModule *m, uint32_t type_id, unsigned *dims, int maxd)
{
    SpvType *t = spv_find_type(m, type_id);
    int n = 0;
    while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
    while (t && t->kind == SPVT_ARRAY && n < maxd)
    {
        dims[n++] = t->n ? t->n : 1;
        t = spv_find_type(m, t->elem);
        while (t && t->kind == SPVT_PTR) t = spv_find_type(m, t->elem);
    }
    return n;
}

static int geom_module_ac_vars(SpvModule *m, MGLAtomicCounterVar *vars, GLuint *bindings,
                               int max, int *n_counters, int *n_buffers)
{
    uint32_t dbind[MGL_MAX_ATOMIC_VARS];
    int nac = 0, nacb = 0, nvars = 0;

    for (int i = 0; i < m->n_vars; i++)
    {
        SpvVar *v = &m->vars[i];
        uint32_t bind = 0, asz = 1;
        bool seen = false;

        if (v->storage != SpvStorageClassAtomicCounter) continue;
        spv_value_type(m, v->type_id, &asz);
        spv_deco(m, v->id, SpvDecorationBinding, &bind);
        {
            /* spv_value_type reports only the OUTERMOST length: an array of arrays
             * (atomic_uint a[2][3]) must count every element (6 counters) */
            unsigned dm[4];
            int nd = spv_array_dims(m, v->type_id, dm, 4);
            if (nd > 1)
            {
                uint32_t total = 1;
                for (int k = 0; k < nd; k++) total *= dm[k];
                asz = total;
            }
        }
        nac += (int)asz;
        for (int k = 0; k < nacb; k++)
            if (dbind[k] == bind) { seen = true; break; }
        if (!seen && nacb < MGL_MAX_ATOMIC_VARS)
            dbind[nacb++] = bind;
        if (nvars < max)
        {
            const char *nm = spv_name(m, v->id);

            snprintf(vars[nvars].name_buf, sizeof vars[nvars].name_buf, "%s", nm ? nm : "");
            vars[nvars].name = vars[nvars].name_buf;
            vars[nvars].offset = 0;
            spv_deco(m, v->id, SpvDecorationOffset, &vars[nvars].offset);
            vars[nvars].ndims = spv_array_dims(m, v->type_id, vars[nvars].dims, 4);
            bindings[nvars] = bind;
            nvars++;
        }
    }
    if (n_counters) *n_counters = nac;
    if (n_buffers) *n_buffers = nacb;
    return nvars;
}

/* forward: the cull-distance emulation's generated pass-through (tess domain, tail of
 * this file) — used by the link/draw machinery above */
static char *geom_gen_cull_gs_glsl(MGLGeomProgram *gp);
static int geom_cull_family(GLenum mode);
static bool geom_cull_prepare(GLMContext ctx, MGLGeomProgram *gp, Program *program, GLenum mode);
static bool geom_cull_writes_msl(Program *p);
static bool geom_cull_writes_msl_tes(Program *p);   /* same, on the tess-eval stage */
static int  geom_cull_tess_family(GLMContext ctx, Program *program);
static int  geom_cull_want_family(GLMContext ctx, MGLGeomProgram *gp, GLenum mode);
static uint32_t geom_cull_store_mask(const uint32_t *ir, size_t nwords, uint32_t builtin, bool *writes);
static GLuint geom_out_decl_size(const uint32_t *ir, size_t nwords, uint32_t builtin);
static const char *geom_tess_clip_check(const Program *program, const MGLGeomProgram *gp);

bool mgl_geom_compile_stage(GLMContext ctx, Program *program, void *obj_p, int stage,                            const uint32_t *words, size_t nwords, const char *entry, char **log)
{
    MGLProgramObj *obj = (MGLProgramObj *)obj_p;
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, true);
    SpvModule *m = NULL;
    const char *err = NULL;
    bool ok = false;
    int i;

    if (!gp)
    {
        mgl_append_log(log, "ERROR: geometry: program table exhausted\n");
        return false;
    }
    geom_free_prog(ctx, gp);
    gp->name = program->name;
    gp->program = getProgram(ctx, program->name);
    /* program.c links into a scratch object (attached=0) and moves the result
     * into the real program only on success, so the GLSL sources of the stages
     * are NOT on `obj` — resolve the real program object (same name) for
     * source lookups.  Without this the VS had to be decompiled from SPIR-V,
     * which the GLSL backend rejects for std140 double blocks. */
    {
        MGLProgramObj *real = gp->program ? MGL_PROGRAM_OBJ(gp->program) : NULL;
        if (real && real->num_attached) obj = real;
    }

    m = spv_parse(words, nwords, &err);
    if (!m)
    {
        mgl_append_log(log, "ERROR: geometry: %s\n", err ? err : "SPIR-V parse failed");
        return false;
    }

    gp->in_type = 0;
    gp->out_type = 0;
    gp->in_verts = 0;
    gp->max_vertices = 0;
    gp->invocations = 1;
    bool have_in = false, have_out = false;
    for (i = 0; i < m->n_modes; i++)
    {
        switch (m->modes[i].mode)
        {
            case SpvExecutionModeInputPoints:           gp->in_type = GL_POINTS; gp->in_verts = 1; have_in = true; break;
            case SpvExecutionModeInputLines:            gp->in_type = GL_LINES; gp->in_verts = 2; have_in = true; break;
            case SpvExecutionModeInputLinesAdjacency:   gp->in_type = GL_LINES_ADJACENCY; gp->in_verts = 4; have_in = true; break;
            case SpvExecutionModeTriangles:             gp->in_type = GL_TRIANGLES; gp->in_verts = 3; have_in = true; break;
            case SpvExecutionModeInputTrianglesAdjacency: gp->in_type = GL_TRIANGLES_ADJACENCY; gp->in_verts = 6; have_in = true; break;
            case SpvExecutionModeOutputPoints:          gp->out_type = GL_POINTS; have_out = true; break;
            case SpvExecutionModeOutputLineStrip:       gp->out_type = GL_LINE_STRIP; have_out = true; break;
            case SpvExecutionModeOutputTriangleStrip:   gp->out_type = GL_TRIANGLE_STRIP; have_out = true; break;
            case SpvExecutionModeOutputVertices:        gp->max_vertices = m->modes[i].value; break;
            case SpvExecutionModeInvocations:           gp->invocations = m->modes[i].value ? m->modes[i].value : 1; break;
            default: break;
        }
    }

    if (!have_in || !have_out)     /* GL_POINTS is 0, so test the flags, not the type */
    {
        mgl_append_log(log, "ERROR: geometry: missing input or output primitive type\n");
        goto done;
    }
    if (gp->max_vertices > (GLuint)256)
    {
        mgl_append_log(log, "ERROR: geometry: max_vertices %u exceeds the limit %u\n",
                       gp->max_vertices, (unsigned)256);
        goto done;
    }
    if (gp->invocations > (GLuint)32)
    {
        mgl_append_log(log, "ERROR: geometry: invocations %u exceeds the limit %u\n",
                       gp->invocations, (unsigned)32);
        goto done;
    }

    /* link-time limit check + enumeration: the geometry kernel takes the atomic_uint
     * arguments mgl_msl_inject_atomic_counters declares (see geom_build), so the stage
     * supports the counters the queries report; a shader that exceeds them must fail the
     * link, exactly as GL 4.6 7.11 requires when a stage's resources exceed its limits
     * (the CTS linking.more_ACs_in_GS / more_ACBs_in_GS_than_supported shape). */
    {
        int nac = 0, nacb = 0, nvars;
        GLint max_ac = 0, max_acb = 0;

        nvars = geom_module_ac_vars(m, gp->ac_vars, gp->ac_binding, MGL_MAX_ATOMIC_VARS,
                                    &nac, &nacb);
        mglGetIntegerv(ctx, GL_MAX_GEOMETRY_ATOMIC_COUNTERS, &max_ac);
        mglGetIntegerv(ctx, GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS, &max_acb);
        if (nac > max_ac)
        {
            mgl_append_log(log, "ERROR: geometry: the geometry stage uses %d atomic counter(s), "
                                "but the limit is %d\n", nac, max_ac);
            goto done;
        }
        if (nacb > max_acb)
        {
            mgl_append_log(log, "ERROR: geometry: the geometry stage uses %d atomic counter "
                                "buffer(s), but the limit is %d\n", nacb, max_acb);
            goto done;
        }
        if (nvars > MGL_MAX_ATOMIC_VARS)
        {
            mgl_append_log(log, "ERROR: geometry: the geometry stage declares %d atomic "
                                "counter variables; the Metal emulation supports %d\n",
                           nvars, (int)MGL_MAX_ATOMIC_VARS);
            goto done;
        }
        /* recorded for the kernel build (geom_build injects the arguments;
         * geomBindStageBuffers binds the GL atomic counter buffers there) */
        gp->n_ac_vars = nvars;
        gp->ac_base = 0;
        for (int i = 0; i < nvars; i++) gp->ac_msl[i] = -1;
    }

    /* fp64: `words` is already software-double lowered (u32 words).  The
     * interface tables must reflect the GL view — real double types — which
     * compile_stage keeps in program->spirv[stage].ir; geom_derive_exprs
     * reflects the vertex outputs from exactly that view, so reflecting the
     * lowered module here makes every double varying a type mismatch.  The
     * kernel generation below still uses the lowered module. */
    {
        SpvModule *vm = NULL;
        const char *e2 = NULL;
        if (program && program->spirv[stage].ir)
            vm = spv_parse(program->spirv[stage].ir, program->spirv[stage].size, &e2);
        if (!geom_reflect_stage(vm ? vm : m, true, gp->in_v, &gp->n_in, log)) { if (vm) spv_free(vm); goto done; }
        if (!geom_reflect_stage(vm ? vm : m, false, gp->out_v, &gp->n_out, log)) { if (vm) spv_free(vm); goto done; }
        if (vm) spv_free(vm);
    }

    /* geom r102: GL 4.6 11.1.2.4 — every output assigned to a transform feedback buffer
     * must be written in the same vertex stream; streams have no relative order, so two
     * streams in one buffer is a link error (KHR-Single-GL46 enhanced_layouts.
     * xfb_multiple_vertex_streams).  Vertex streams exist only in a geometry stage. */
    for (int a = 0; a < gp->n_out; a++)
    {
        char na[160];
        if (gp->out_v[a].xfb_buffer < 0) continue;
        if (geom_var_dotted(&gp->out_v[a]))
        {
            if (gp->out_v[a].block_array)
                snprintf(na, sizeof(na), "%s[%u].%s", gp->out_v[a].name, gp->out_v[a].index,
                         gp->out_v[a].member);
            else
                snprintf(na, sizeof(na), "%s.%s", gp->out_v[a].name, gp->out_v[a].member);
        }
        else
            snprintf(na, sizeof(na), "%s", gp->out_v[a].name);
        for (int b = a + 1; b < gp->n_out; b++)
        {
            char nb[160];
            if (gp->out_v[b].xfb_buffer != gp->out_v[a].xfb_buffer) continue;
            if (gp->out_v[b].stream == gp->out_v[a].stream) continue;
            if (geom_var_dotted(&gp->out_v[b]))
            {
                if (gp->out_v[b].block_array)
                    snprintf(nb, sizeof(nb), "%s[%u].%s", gp->out_v[b].name, gp->out_v[b].index,
                             gp->out_v[b].member);
                else
                    snprintf(nb, sizeof(nb), "%s.%s", gp->out_v[b].name, gp->out_v[b].member);
            }
            else
                snprintf(nb, sizeof(nb), "%s", gp->out_v[b].name);
            mgl_append_log(log, "ERROR: geometry: outputs '%s' (stream %u) and '%s' (stream %u) are "
                                "assigned to the same transform feedback buffer %d from different "
                                "vertex streams\n",
                           na, gp->out_v[a].stream, nb, gp->out_v[b].stream,
                           gp->out_v[a].xfb_buffer);
            goto done;
        }
    }

    /* a named output interface block ("out B { vec4 c; } b;") is rewritten to a plain
     * struct + instance in the generated kernel (the body keeps writing b.c) and its
     * members are re-emitted as ordinary location-addressed varyings on the raster side,
     * where the fragment stage's matching input block links by location */

    /* the entry names are needed even when the emulation is deferred */
    gp->kernel_entry = xstrdup("mgl_geom_gs");
    gp->capture_entry = xstrdup("main0");
    gp->draw_entry = xstrdup("main0");

    /* a separable geometry-only program has no vertex stage of its own: the reflection and
     * the record layout are kept (so the program queries work), the emulation is built at
     * the first draw, when the pipeline contributes the vertex stage.  The layout uses the
     * default clip distance size here; geom_build() refines it once the vertex stage is
     * known.  The builtin scan below still runs for such a program: the input record always
     * carries gl_Position, and the capture variant only stores what in_has_* says. */
    geom_layout(gp);
    gp->deferred = !program->shader_slots[_VERTEX_SHADER];

    /* builtins: driven by what the shader source mentions */
    {
        const char *src = geom_gs_source(ctx, obj);
        char *decompiled = NULL;
        if (!src)
        {
            decompiled = geom_spirv_to_glsl(words, nwords, 460, log);
            src = decompiled;
            if (!src)
            {
                mgl_append_log(log, "ERROR: geometry: no geometry shader source available\n");
                goto done;
            }
        }

        gp->in_has_pos = true;
        gp->in_has_psize = strstr(src, "gl_PointSize") != NULL;
        gp->in_has_clip = strstr(src, "gl_ClipDistance") != NULL;
        gp->in_has_cull = strstr(src, "gl_CullDistance") != NULL;
        gp->out_has_pos = true;
        gp->out_has_psize = strstr(src, "gl_PointSize") != NULL;
        gp->out_has_layer = strstr(src, "gl_Layer") != NULL;
        gp->out_has_vp = strstr(src, "gl_ViewportIndex") != NULL;
        gp->out_has_prim = strstr(src, "gl_PrimitiveID") != NULL;
        gp->out_has_clip = strstr(src, "gl_ClipDistance") != NULL;
        gp->out_has_cull = strstr(src, "gl_CullDistance") != NULL;

        {
            const char *vs_src = geom_vs_source(ctx, obj);
            char *vs_dec = NULL;
            GLuint n;
            if (!vs_src && program->spirv[_VERTEX_SHADER].ir)
            {
                vs_dec = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                            program->spirv[_VERTEX_SHADER].size, 460, log);
                vs_src = vs_dec;
            }
            if (!vs_src)
            {
                /* a separable geometry-only program has no vertex stage of its own: the clip
                 * sizes default to the maximum now and geom_build() refines them from the
                 * pipeline's vertex stage at the first draw */
                if (!gp->deferred)
                {
                    mgl_append_log(log, "ERROR: geometry: no vertex shader to capture from\n");
                    free(vs_dec);
                    free(decompiled);
                    goto done;
                }
                gp->in_clip_size = gp->in_has_clip ? MGL_GEOM_MAX_CLIP : 0;
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ? MGL_GEOM_MAX_CLIP : 0;
            }
            else
            {
                n = geom_declared_clip_words(vs_src, "gl_ClipDistance");
                gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
                n = geom_declared_clip_words(vs_src, "gl_CullDistance");
                if (gp->in_has_cull)
                {
                    /* the record carries the clip AND the cull array at one shared
                     * width (geom_layout), so the wider of the two declared sizes
                     * wins - a smaller cull array must not shrink the clip words
                     * (clip=1 + cull=7 kept 1 word and dropped cull[1..6]) */
                    GLuint nc2 = n ? n : MGL_GEOM_MAX_CLIP;
                    if (nc2 > gp->in_clip_size) gp->in_clip_size = nc2;
                }
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ?
                                    (gp->in_clip_size ? gp->in_clip_size : MGL_GEOM_MAX_CLIP) : 0;
            }
            free(vs_dec);
        /* the clip size of the real vertex stage is only known now */
        geom_layout(gp);
        }
        free(decompiled);
    }

    /* which clip planes this geometry stage writes (tess-domain store scan, geometry.c
     * tail): the raster pass-through neutralizes the unwritten ones, whose values are
     * undefined in GL and would otherwise clip spuriously */
    if (gp->out_has_clip)
    {
        bool w = false;
        gp->clip_planes = geom_cull_store_mask(words, nwords, (uint32_t)SpvBuiltInClipDistance, &w);
    }

    if (gp->deferred)
    {
        /* the emulation is built at the first draw, when the pipeline contributes the vertex
         * stage; everything above is reflection only */
        ok = true;
        goto done;
    }

    /* total output component limit — the GL rule (GL 4.6 11.1.3.4 / dEQP's
     * componentsPerVertex gate in es31fGeometryShaderTests): gl_Position (4) +
     * the USER output varyings, times max_vertices.  The emulation record also
     * carries point size / layer / viewport index / primitive id / clip+cull
     * words, but those built-ins do not count toward
     * MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS and the GPU budget is guarded by
     * MGL_GEOM_BUDGET_BYTES (256MB) instead. */
    {
        GLuint comps = 4 * gp->max_vertices;
        for (i = 0; i < gp->n_out; i++)
            comps += gp->out_v[i].words * gp->max_vertices;
        if (comps > (GLuint)2048)
        {
            mgl_append_log(log, "ERROR: geometry: %u total output components exceeds the limit %u\n",
                           comps, (unsigned)2048);
            goto done;
        }
    }

    gp->streams = 1;
    for (i = 0; i < gp->n_out; i++)
        if (gp->out_v[i].stream + 1 > gp->streams) gp->streams = gp->out_v[i].stream + 1;

    /* a tessellation stage in front of the geometry shader: the tess domain's TES-output
     * capture supplies the records, so there is no vertex stage to derive the inputs'
     * producing expressions from (the draw-time import repacks the capture instead) */
    gp->from_tess = (program->shader_slots[_TESS_EVALUATION_SHADER] != NULL);
    if (gp->from_tess)
    {
        const char *cc = geom_tess_clip_check(program, gp);

        if (cc)
        {
            mgl_append_log(log, "ERROR: %s\n", cc);
            goto done;
        }
    }

    /* writer side: the vertex outputs feeding each input location.  A separable
     * geometry-only program has no vertex stage to reflect here; geom_ensure() does it from
     * the pipeline's composite once that is available. */
    if (!gp->deferred && !gp->from_tess)
    {
        if (!geom_derive_exprs(gp, program, log))
            goto done;
    }

    ok = true;

done:
    spv_free(m);
    if (!ok)
    {
        geom_fail(ctx, gp, "geometry shader emulation could not be built");
        return false;
    }
    return true;
}

/* patch a `[[user(locnN)]]` attribute of an MSL struct member into a Metal builtin
 * attribute (used to forward gl_Layer / gl_ViewportIndex).  Returns a newly allocated
 * string (the caller frees the original), or NULL if the member was not found. */
static char *msl_promote_attribute(const char *msl, const char *member, const char *attr)
{
    const char *p = strstr(msl, member);
    const char *u, *close;
    char *out;
    size_t head, tail;

    if (!p) return NULL;
    u = strstr(p, "user(locn");
    if (!u || (size_t)(u - p) > 96) return NULL;
    close = strchr(u, ')');
    if (!close) return NULL;
    close++;                        /* include the ')' */

    head = (size_t)(u - msl);
    tail = strlen(close);
    out = xmalloc(head + strlen(attr) + tail + 1);
    memcpy(out, msl, head);
    memcpy(out + head, attr, strlen(attr));
    memcpy(out + head + strlen(attr), close, tail + 1);
    return out;
}

/* post link: translate the three emulation shaders to MSL, replace the program's vertex
 * stage with the pass-through function */
/* SPIRV-Cross passes loose (explicit-location) uniforms of the default block as
 * `constant T&` arguments of the entry point, but a user function (our renamed shader
 * main) takes them as `thread const T&` — which Metal rejects.  Turn those parameters
 * into value parameters. */
static bool msl_value_parameters(char *msl, const char *entry)
{
    char *def = strstr(msl, entry);
    char *p, *close;
    size_t removed = 0;

    if (!def) return false;
    p = strchr(def, '(');
    if (!p) return false;

    /* find the end of the parameter list */
    close = p + 1;
    while (*close && *close != ')') close++;
    if (!*close) return false;

    /* rewrite "thread const T& name" as "T name" inside the list */
    p = p + 1;
    while (p < close)
    {
        if (!strncmp(p, "thread const ", 13))
        {
            char *amp = p + 13;
            int depth = 0;   /* a template argument list has its own commas
                              * ("spvUnsafeArray<uint, 8>&"): only an '&' at depth 0
                              * separates the type from the parameter name */
            while (amp < close)
            {
                if (*amp == '<') depth++;
                else if (*amp == '>') depth--;
                else if (depth == 0 && (*amp == '&' || *amp == ',')) break;
                amp++;
            }
            if (amp < close && *amp == '&')
            {
                size_t tail = strlen(amp) + 1;
                /* drop the "thread const " qualifier */
                memmove(p, p + 13, strlen(p + 13) + 1);
                close -= 13;
                amp -= 13;
                /* drop the '&' */
                memmove(amp, amp + 1, tail);
                close -= 1;
                removed += 14;
                continue;
            }
        }
        p++;
    }
    return removed > 0;
}

/* the Metal buffer index the renderer's buffer map (mapGLBuffersToMTLBufferMap) gives the
 * geometry stage's atomic-counter group: after the uniform-block group (one entry per
 * block, instance arrays expanded), the uniform-constant group (one entry per resource —
 * the map never expands those: a sampler array is ONE array<texture> argument — including
 * the default-uniform block wherever the introspect domain's link hook moved it: moving it
 * between the two groups keeps the total) and the storage-buffer group (instance arrays
 * expanded).  This mirrors the map's enumeration loop entry for entry, including its
 * skip-ahead after an expanded instance array. */
static int geom_stage_ac_base(Program *program, int stage)
{
    static const int types[3] = {
        SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
        SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT,
        SPVC_RESOURCE_TYPE_STORAGE_BUFFER
    };
    int base = 0;

    for (int t = 0; t < 3; t++)
    {
        SpirvResourceList *rl = &program->spirv_resources_list[stage][types[t]];

        for (GLuint i = 0; i < rl->count; i++)
        {
            GLuint elems = (types[t] == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT) ? 1 :
                           (rl->list[i].array_size ? rl->list[i].array_size : 1);

            base += (int)elems;
            if (elems > 1)
            {
                /* the map skips the flattened instance entries an expanded array covers */
                GLuint skip = elems - 1;

                if (skip > rl->count - i - 1) skip = rl->count - i - 1;
                i += skip;
            }
        }
    }
    return base;
}

/* the highest [[buffer(n)]] argument index below the kernel pool slots in the kernel
 * entry's signature — the user resources geom_build_pins placed.  The atomic-counter
 * arguments are appended above it, so they can not collide with them (the buffer map's
 * own ordinals — which the renderer binds at, one entry per user resource — do not
 * count sampler arrays, which the pins pin as TEXTURE arguments in the same numbering,
 * so the two can differ; the atomic-counter binding is by the indices recorded here). */
static int geom_msl_kernel_buffer_top(const char *msl, const char *entry)
{
    char pat[128];
    const char *p;
    int top = -1;

    snprintf(pat, sizeof pat, "void %s(", entry);
    p = strstr(msl, pat);
    if (!p) return -1;
    p += strlen(pat);
    while (*p && *p != '{' && *p != ';')
    {
        if (p[0] == '[' && p[1] == '[' && !strncmp(p + 2, "buffer(", 7))
        {
            char *end;
            long n = strtol(p + 9, &end, 10);

            if (end != p + 9 && n >= 0 && n < MGL_GEOM_KERNEL_USER_SLOTS && (int)n > top)
                top = (int)n;
            p = end;
            continue;
        }
        p++;
    }
    return top;
}

int mgl_geom_stage_ac_arg_base(Program *program, int stage)
{
    SpirvResourceList *rl;

    if (!program || stage < 0 || stage >= _MAX_SHADER_TYPES) return -1;
    rl = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
    if (rl->count == 0 || !rl->list) return -1;
    if (rl->list[0].msl_binding == 0xffffffffu) return -1;
    /* program.c refined msl_binding[i] to the index mgl_msl_inject_atomic_counters gave
     * declaration i (ac_base + i), so the first entry carries the group's base */
    return (int)rl->list[0].msl_binding;
}

/* build (or rebuild) the emulation for one geometry stage; `program` is the program that
 * owns the vertex + geometry stages (a linked program, or a pipeline's composite) */
/* forward: defined below mgl_geom_post_link's helpers */
static void geom_patch_fs_primitive_id(GLMContext ctx, MGLGeomProgram *gp, Program *program);

static void geom_build(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);
    char *gs_src = NULL, *vs_src = NULL, *glsl, *msl;
    uint32_t *ir = NULL;
    size_t words = 0;
    char *log = NULL;
    int pins_gl[8], pins_msl[8];
    GeomPin pins2[8];
    int npins2 = 0;

#define GEOM_FAIL(...)  do { geom_fail(ctx, gp, __VA_ARGS__); \
                             if (!gp->deferred) obj->link_status = false; goto done; } while (0)

    if (!gp || !obj) return;
    if (!program->shader_slots[_GEOMETRY_SHADER] && !gp->synthetic) return;

    gp->gs_exec = program->shader_slots[_GEOMETRY_SHADER];

    /* a tessellation stage in front of the geometry shader: the tess domain's TES-output
     * capture producer (mgl_ext_tess.h) feeds the geometry kernel instead of a vertex capture
     * variant — the records are repacked into this program's input record at draw time
     * (mgl_geom_route / the import kernel in MGLRenderer+Geometry.m) */
    gp->from_tess = (program->shader_slots[_TESS_EVALUATION_SHADER] != NULL);
    if (gp->from_tess)
    {
        const char *cc = geom_tess_clip_check(program, gp);

        if (cc)
            GEOM_FAIL("%s", cc);
    }
    if (!gp->from_tess &&
        (program->shader_slots[_TESS_CONTROL_SHADER] || program->shader_slots[_TESS_EVALUATION_SHADER]))
        GEOM_FAIL("a tessellation control shader without an evaluation shader cannot feed "
                  "the geometry stage");

    {
        const char *src = geom_gs_source(ctx, obj);
        if (gp->synthetic)
        {
            /* the cull-distance emulation (tess domain, see the tail of this file): the
             * "geometry" stage is a generated pass-through that drops culled primitives */
            gs_src = geom_gen_cull_gs_glsl(gp);
            if (!gs_src) GEOM_FAIL("cull pass-through generation failed");
        }
        else
        {
        gs_src = xstrdup(src ? src : "");
        if (!gs_src[0])
        {
            free(gs_src);
            gs_src = geom_spirv_to_glsl(program->spirv[_GEOMETRY_SHADER].ir,
                                        program->spirv[_GEOMETRY_SHADER].size, 460, &log);
            if (!gs_src) GEOM_FAIL("no geometry shader source: %s", log ? log : "?");
            /* SPIRV-Cross's GLSL output re-declares the introspect domain's packed
             * default-uniform block with a layout(location) per member — legal for
             * in/out blocks only, so re-compiling the decompiled source fails with
             * "location on block member" (the subroutine selectors carry 39xx).  The
             * kernel re-emits the block with explicit std430 offsets anyway, so the
             * member locations are meaningless: strip them. */
            geom_strip_block_member_locations(gs_src);
        }
        }
    }

    /* Metal maps a fragment shader's gl_PrimitiveID to the rasterizer's primitive index,
     * which knows nothing of the geometry stage (an emitted strip decomposes into several
     * raster primitives, and the FS must see the GS's value instead).  When the linked
     * fragment stage reads it, the output record always carries the primitive id and the
     * pass-through vertex function forwards it as a varying (geom_patch_fs_primitive_id
     * redirects the fragment function's attribute onto that varying below).  The kernel
     * keeps the input primitive id in the word when the geometry shader never writes
     * gl_PrimitiveID, which is the value Mesa feeds the fragment stage then. */
    gp->fs_uses_prim = false;
    if (program->spirv[_FRAGMENT_SHADER].msl_str &&
        strstr(program->spirv[_FRAGMENT_SHADER].msl_str, "[[primitive_id]]"))
        gp->fs_uses_prim = true;
    if (gp->fs_uses_prim)
    {
        gp->out_has_prim = true;
        /* the flag was not known when the stage was reflected: recompute the record
         * layout so the output carries the primitive id word */
        geom_layout(gp);
    }
    /* a relink may drop the atomic counters the previous build recorded (the stages are
     * recompiled before this runs); mgl_geom_compile_stage fills gp->ac_* again */
    gp->n_vs_ac_vars = 0;
    gp->vs_ac_map_base = 0;

    if (gp->from_tess)
    {
        /* no vertex stage is captured: the tess domain publishes the records (with its own
         * member layout) and the import kernel repacks them into gp's input record */
        gp->vid_absolute = false;
        gp->iid_absolute = false;
        /* the record's clip/cull words are sized by the evaluation stage's declared
         * OUTPUT sizes (the capture carries its gl_ClipDistance/gl_CullDistance as
         * flattened per-element members; the import maps them onto in_clip_off /
         * in_cull_off) — not by the vertex stage the reflection used, the TES is
         * the producer.  Mirrors the vertex-stage rule one-to-one (see above).
         * The synthetic cull variant keeps the reflected size: it repacks the tess
         * capture's own gl_CullDistance_q / gl_ClipDistance_q members instead. */
        if (gp->synthetic)
            ;
        else
        {
            const Spirv *tes = &program->spirv[_TESS_EVALUATION_SHADER];
            GLuint n  = geom_out_decl_size(tes->ir, tes->size, (uint32_t)SpvBuiltInClipDistance);
            GLuint nc = geom_out_decl_size(tes->ir, tes->size, (uint32_t)SpvBuiltInCullDistance);

            gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
            if (gp->in_has_cull && (nc ? nc : MGL_GEOM_MAX_CLIP) > gp->in_clip_size)
                gp->in_clip_size = nc ? nc : MGL_GEOM_MAX_CLIP;
            gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ?
                                (gp->in_clip_size ? gp->in_clip_size : MGL_GEOM_MAX_CLIP) : 0;
        }
        gp->capture_entry = xstrdup("main0");
        geom_layout(gp);
        goto build;          /* skip the capture variant, the draw variant is built below */
    }

    {
        const char *src = geom_vs_source(ctx, obj);
        vs_src = xstrdup(src ? src : "");
        if (!vs_src[0])
        {
            free(vs_src);
            vs_src = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                        program->spirv[_VERTEX_SHADER].size, 460, &log);
            if (!vs_src) GEOM_FAIL("no vertex shader source: %s", log ? log : "?");
        }
    }

    /* ---- 1. vertex capture variant ---- */
    {
        const char *e2 = NULL;
        SpvModule *p_vs = spv_parse(program->spirv[_VERTEX_SHADER].ir,
                                    program->spirv[_VERTEX_SHADER].size, &e2);

        glsl = geom_gen_capture_glsl(ctx, gp, vs_src, p_vs, &log);
        if (!glsl) GEOM_FAIL("vertex capture variant generation failed");
        geom_dump("capture.glsl", glsl);
        if (!geom_glsl_to_spirv(ctx, GL_VERTEX_SHADER, glsl, 430, &ir, &words, &log))
            GEOM_FAIL("vertex capture variant compile failed: %s", log ? log : "?");
        free(glsl);
        {
            SpvModule *c_vs = spv_parse(ir, words, &e2);
            int span;

            npins2 = 0;
            span = geom_build_pins(p_vs, c_vs, pins2, &npins2, 8,
                                   gp->tex_slots_vs, &gp->n_tex_slots_vs, 64);
            if (span > MGL_GEOM_KERNEL_USER_SLOTS ||
                (span > MGL_GEOM_KERNEL_SIZE_BUFFER &&
                 spv_uses_array_length(ir, words)))
            {
                spv_free(c_vs);
                spv_free(p_vs);
                GEOM_FAIL("vertex stage resources span %d argument slots; the compute "
                          "emulation supports %d (MGL_GEOM_KERNEL_USER_SLOTS%s)",
                          span, (int)MGL_GEOM_KERNEL_USER_SLOTS,
                          spv_uses_array_length(ir, words) ?
                          ", and MGL_GEOM_KERNEL_SIZE_BUFFER is reserved for SSBO "
                          ".length() sizes" : "");
            }
            spv_free(c_vs);
        }
        /* the vertex stage's atomic counters: the capture variant below is the only
         * vertex function of a geometry program (the normal vertex MSL is not used), so
         * it needs the injected atomic_uint arguments — enumerate the declarations here
         * (the module is the program's vertex stage) */
        gp->n_vs_ac_vars = geom_module_ac_vars(p_vs, gp->vs_ac_vars, gp->vs_ac_binding,
                                               MGL_MAX_ATOMIC_VARS, NULL, NULL);
        gp->vs_ac_map_base = geom_stage_ac_base(program, _VERTEX_SHADER);
        if (gp->n_vs_ac_vars > MGL_MAX_ATOMIC_VARS)
        {
            spv_free(p_vs);
            GEOM_FAIL("vertex stage declares %d atomic counter variables; the Metal "
                      "emulation supports %d", gp->n_vs_ac_vars, (int)MGL_MAX_ATOMIC_VARS);
        }
        for (int i = 0; i < gp->n_vs_ac_vars; i++) gp->vs_ac_msl[i] = -1;
        spv_free(p_vs);
    }
    pins_gl[0] = MGL_GEOM_SSBO_VSOUT;
    pins_msl[0] = MGL_GEOM_MSL_VSOUT;
    msl = geom_spirv_to_msl(ctx, GL_VERTEX_SHADER, ir, words, gp->capture_entry, true, &log,
                            SpvExecutionModelVertex, pins_gl, pins_msl, 1, pins2, npins2,
                            gp->tex_slots_vs, &gp->n_tex_slots_vs, 64, true);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("vertex capture variant MSL failed: %s", log ? log : "?");
    /* a vertex stage writing gl_CullDistance leaves BuiltInCullDistance stores pointing
     * at a non-existent array member in the capture MSL too (program.c applies the same
     * repair to the plain vertex MSL) */
    {
        char *fixed_cull = mgl_msl_fix_cull_distance(msl);
        if (fixed_cull) { free(msl); msl = fixed_cull; }
    }
    /* the renamed vertex main is called from the generated entry point */
    msl_value_parameters(msl, "mgl_geom_vs_main(");

    /* atomic counters of the vertex stage: the capture variant's MSL gets the injected
     * atomic_uint arguments (mgl_msl_inject_atomic_counters, geometry.c), one per GLSL
     * declaration, placed above the variant's own user buffer arguments;
     * geomCapturePass binds the GL atomic counter buffers at the recorded indices
     * (NOT at the vertex buffer map's own atomic-counter ordinals) */
    if (gp->n_vs_ac_vars > 0)
    {
        int idx[MGL_MAX_ATOMIC_VARS];
        int base = geom_msl_kernel_buffer_top(msl, gp->capture_entry) + 1;
        int maxidx = strstr(msl, "spvBufferSizeConstants") ? MGL_GEOM_KERNEL_SIZE_BUFFER :
                                                            MGL_GEOM_KERNEL_USER_SLOTS;
        char *injected = mgl_msl_inject_atomic_counters(msl, gp->capture_entry,
                                                        gp->vs_ac_vars, gp->n_vs_ac_vars,
                                                        maxidx, base, idx);

        if (!injected)
            GEOM_FAIL("vertex stage atomic counter buffer arguments could not be "
                      "declared: %d declarations at Metal buffer %d",
                      gp->n_vs_ac_vars, base);
        free(msl);
        msl = injected;
        for (int i = 0; i < gp->n_vs_ac_vars; i++) gp->vs_ac_msl[i] = idx[i];
    }
    gp->capture_msl = msl;
    geom_dump("capture.msl", msl);
    gp->vid_absolute = strstr(msl, "[[base_vertex]]") == NULL;
    gp->iid_absolute = strstr(msl, "[[base_instance]]") == NULL;

    /* the capture function is not part of the program's stages: compile it now */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
    {
        void *lib = NULL, *fn = NULL;
        char *err = NULL;
        if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, gp->capture_entry, &lib, &fn, &err))
        {
            GEOM_FAIL("vertex capture variant Metal compile failed: %s", err ? err : "?");
        }
        gp->capture_library = lib;
        gp->capture_function = fn;
    }
    else
        GEOM_FAIL("no Metal compile hook available");

    /* ---- 2. geometry kernel ---- */
build:
    /* the program's own geometry-stage module owns the resource order the renderer's
     * buffer map enumerates; pin this kernel's user resources to those positions */
    int gs_span = 0;                /* the user pins' slot span: atomics sit above it */
    {
        const char *e2 = NULL;
        /* the synthetic cull pass-through (tess domain) has no geometry stage of its
         * own and no user resources: the kernel's arguments are the eight emulation
         * buffers alone, so there is nothing to reflect or pin */
        SpvModule *p_gs = gp->synthetic ? NULL :
            spv_parse(program->spirv[_GEOMETRY_SHADER].ir,
                      program->spirv[_GEOMETRY_SHADER].size, &e2);
        SpvModule *k_gs = NULL;
        glsl = geom_gen_kernel_glsl(ctx, gp, gs_src, p_gs, &log);
        if (getenv("MGL_GEOM_DUMP"))
            for (int i = 0; i < gp->n_out; i++)
                fprintf(stderr, "geomdbg: out_v[%d] '%s' elems=%u cols=%u rows=%u arr=%d words=%u off=%u\n",
                        i, gp->out_v[i].name, gp->out_v[i].elems, gp->out_v[i].cols,
                        gp->out_v[i].rows, (int)gp->out_v[i].arr, gp->out_v[i].words, gp->out_v[i].offset);
        if (!glsl) GEOM_FAIL("kernel generation failed");
        geom_dump("kernel.glsl", glsl);
        if (!geom_glsl_to_spirv(ctx, GL_COMPUTE_SHADER, glsl, 450, &ir, &words, &log))
            GEOM_FAIL("kernel compile failed: %s", log ? log : "?");
        geom_record_tex_arrays(ir, words, gp);
        free(glsl);
        k_gs = spv_parse(ir, words, &e2);
        {
            /* the kernel's own buffers live at fixed Metal slots (MGL_GEOM_MSL_*): the
             * stage's program resources must fit in the slots below them */
            if (!gp->synthetic)
            {
            int span = geom_build_pins(p_gs, k_gs, pins2, &npins2, 8,
                                       gp->tex_slots_gs, &gp->n_tex_slots_gs, 64);
            bool array_len = spv_uses_array_length(ir, words) != 0;
            gs_span = span;

            gp->array_len = array_len;
            gp->ac_base = geom_stage_ac_base(program, _GEOMETRY_SHADER);
            if (span > MGL_GEOM_KERNEL_USER_SLOTS ||
                (span > MGL_GEOM_KERNEL_SIZE_BUFFER && array_len))
                GEOM_FAIL("geometry stage resources span %d argument slots; the compute "
                          "emulation supports %d (MGL_GEOM_KERNEL_USER_SLOTS%s)",
                          span, (int)MGL_GEOM_KERNEL_USER_SLOTS,
                          array_len ?
                          ", and MGL_GEOM_KERNEL_SIZE_BUFFER is reserved for SSBO "
                          ".length() sizes" : "");
            }
        }
        spv_free(k_gs);
        spv_free(p_gs);
    }
    pins_gl[0] = MGL_GEOM_SSBO_PRIM;     pins_msl[0] = MGL_GEOM_MSL_PRIM;
    pins_gl[1] = MGL_GEOM_SSBO_VSOUT;    pins_msl[1] = MGL_GEOM_MSL_VSOUT;
    pins_gl[2] = MGL_GEOM_SSBO_HDR;      pins_msl[2] = MGL_GEOM_MSL_HDR;
    pins_gl[3] = MGL_GEOM_SSBO_OUTV;     pins_msl[3] = MGL_GEOM_MSL_OUTV;
    pins_gl[4] = MGL_GEOM_SSBO_IDX;      pins_msl[4] = MGL_GEOM_MSL_IDX;
    pins_gl[5] = MGL_GEOM_SSBO_CTL;      pins_msl[5] = MGL_GEOM_MSL_CTL;
    pins_gl[6] = MGL_GEOM_SSBO_CNT;      pins_msl[6] = MGL_GEOM_MSL_CNT;
    pins_gl[7] = MGL_GEOM_SSBO_LSCR;     pins_msl[7] = MGL_GEOM_MSL_LSCR;
    msl = geom_spirv_to_msl(ctx, GL_COMPUTE_SHADER, ir, words, gp->kernel_entry, false, &log,
                            SpvExecutionModelGLCompute, pins_gl, pins_msl, 8, pins2, npins2,
                            gp->tex_slots_gs, &gp->n_tex_slots_gs, 64, true);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("kernel MSL failed: %s", log ? log : "?");
    /* image atomics: the kernel MSL carries SPIRV-Cross's `<name>_atomic [[buffer(0)]]`
     * sidecar arguments — every one at slot 0 (Metal: "cannot reserve 'buffer' resource
     * location at index 0"), and non-2D targets subscript the buffer with the raw
     * coordinate vector.  Rebase them into the span above the user pins and linearize
     * the coordinates exactly like the render stages (mgl_msl_fix_image_atomics) —
     * geomBindStageBuffers binds the per-unit sidecar through gp->atom_slots_gs. */
    {
        char *fixed = mgl_msl_fix_image_atomics(msl, MGL_GEOM_KERNEL_SIZE_BUFFER,
                                                gs_span > 0 ? gs_span : 0,
                                                gp->atom_slots_gs, &gp->n_atom_slots_gs, 32,
                                                NULL);
        if (!fixed)
            GEOM_FAIL("geometry stage image-atomic buffers exceed the kernel's argument "
                      "budget (MGL_GEOM_KERNEL_USER_SLOTS)");
        free(msl);
        msl = fixed;
    }

    /* geom r102: force_native_arrays makes SPIRV-Cross copy an array out of a uniform
     * block with its spvArrayCopyFromStackToStack1 helper, but for a DEVICE source
     * (uniform-block member arrays keep the raw C-array shape) it emits only the call —
     * its helper template is defined for thread-to-thread copies, so Metal sees "no
     * matching function" (`goku = uni_goku;` on a dmat array, KHR-Single-GL46
     * enhanced_layouts.xfb_explicit_location_dmat*_geometry).  Supply the device-source
     * overload ourselves, unless SPIRV-Cross already emitted it. */
    if (strstr(msl, "spvArrayCopyFromStackToStack1") &&
        !strstr(msl, "spvArrayCopyFromStackToStack1(thread T (&dst)[N], constant T (&src)[N])") &&
        !strstr(msl, "spvArrayCopyFromStackToStack1(thread T (&dst)[N], const device T (&src)[N])"))
    {
        static const char arrcopy[] =
            "template<typename T, size_t N>\n"
            "static inline __attribute__((always_inline))\n"
            "void spvArrayCopyFromStackToStack1(thread T (&dst)[N], constant T (&src)[N])\n"
            "{\n"
            "    for (size_t i = 0; i < N; ++i) dst[i] = src[i];\n"
            "}\n"
            "template<typename T, size_t N>\n"
            "static inline __attribute__((always_inline))\n"
            "void spvArrayCopyFromStackToStack1(thread T (&dst)[N], const device T (&src)[N])\n"
            "{\n"
            "    for (size_t i = 0; i < N; ++i) dst[i] = src[i];\n"
            "}\n";
        const char *anchor = "#include <metal_stdlib>\n";
        const char *at = strstr(msl, anchor);
        char *out;
        size_t pre = at ? (size_t)(at - msl) + strlen(anchor) : 0;
        out = xmalloc(strlen(msl) + sizeof arrcopy + 1);
        memcpy(out, msl, pre);
        memcpy(out + pre, arrcopy, sizeof arrcopy - 1);
        memcpy(out + pre + sizeof arrcopy - 1, msl + pre, strlen(msl) - pre + 1);
        free(msl);
        msl = out;
    }

    /* the kernel must have a stable name: SPIRV-Cross always emits "kernel void main0(".
     * The replacement is longer than the match, so the rename goes through a fresh
     * buffer: moving the tail in place writes past the end of the MSL string (it is a
     * strdup of the exact length) and corrupts the heap. */
    {
        const char *want = "kernel void main0(";
        const char *repl = "kernel void mgl_geom_gs(";
        char *e = strstr(msl, want);
        if (!e) GEOM_FAIL("kernel MSL has no main0 entry point");
        {
            size_t pre = (size_t)(e - msl);
            size_t rest = strlen(e + strlen(want));
            char *out = xmalloc(pre + strlen(repl) + rest + 1);

            memcpy(out, msl, pre);
            memcpy(out + pre, repl, strlen(repl));
            memcpy(out + pre + strlen(repl), e + strlen(want), rest + 1);
            free(msl);
            msl = out;
        }
    }
    msl_value_parameters(msl, "mgl_gs_main(");

    /* atomic counters: the kernel's MSL gets the atomic_uint arguments the standard
     * stages declare (mgl_msl_inject_atomic_counters, geometry.c), one per GLSL
     * declaration, placed above the kernel's own user buffer arguments (the MSL
     * compile happens after the entry rename, so the stable kernel name is the entry
     * the counter uses are threaded into — the GS body is the mgl_gs_main helper the
     * entry calls, and msl_msl_counters_to_helper_params adds the trailing parameters).
     * The renderer binds the GL atomic-counter buffers at the recorded indices
     * (geomBindStageBuffers), NOT at the buffer map's own atomic-counter ordinals. */
    if (gp->n_ac_vars > 0)
    {
        int idx[MGL_MAX_ATOMIC_VARS];
        int base = geom_msl_kernel_buffer_top(msl, gp->kernel_entry) + 1;
        int maxidx = gp->array_len ? MGL_GEOM_KERNEL_SIZE_BUFFER :
                                     MGL_GEOM_KERNEL_USER_SLOTS;
        char *injected = mgl_msl_inject_atomic_counters(msl, gp->kernel_entry, gp->ac_vars,
                                                        gp->n_ac_vars, maxidx, base, idx);

        if (!injected)
            GEOM_FAIL("geometry stage atomic counter buffer arguments could not be "
                      "declared: %d declarations at Metal buffer %d (the kernel's user "
                      "arguments span %d slots)", gp->n_ac_vars, base, base);
        free(msl);
        msl = injected;
        for (int i = 0; i < gp->n_ac_vars; i++)
            gp->ac_msl[i] = idx[i];
    }
    gp->kernel_msl = msl;
    geom_dump("kernel.msl", msl);

    /* the geometry stage executable is the kernel; the synthetic cull pass-through has
     * no geometry stage slot to publish it in (it rides in gp->kernel_msl) */
    if (!gp->synthetic)
    {
        free(program->spirv[_GEOMETRY_SHADER].msl_str);
        program->spirv[_GEOMETRY_SHADER].msl_str = xstrdup(gp->kernel_msl);
        free(program->spirv[_GEOMETRY_SHADER].entry_point);
        program->spirv[_GEOMETRY_SHADER].entry_point = xstrdup(gp->kernel_entry);
        if (program->shader_slots[_GEOMETRY_SHADER])
            program->shader_slots[_GEOMETRY_SHADER]->entry_point = xstrdup(gp->kernel_entry);
    }

    /* ---- 3. pass-through raster vertex function ---- */
    glsl = geom_gen_draw_glsl(ctx, gp, &log);
    if (!glsl) GEOM_FAIL("pass-through vertex function generation failed");
    geom_dump("draw.glsl", glsl);
    if (!geom_glsl_to_spirv(ctx, GL_VERTEX_SHADER, glsl, 460, &ir, &words, &log))
        GEOM_FAIL("pass-through vertex function compile failed: %s", log ? log : "?");
    free(glsl);
    pins_gl[0] = MGL_GEOM_SSBO_DRAW;     pins_msl[0] = MGL_GEOM_MSL_DRAW;
    msl = geom_spirv_to_msl(ctx, GL_VERTEX_SHADER, ir, words, gp->draw_entry, false, &log,
                            SpvExecutionModelVertex, pins_gl, pins_msl, 1, NULL, 0, NULL, NULL, 0, true);
    free(ir); ir = NULL; words = 0;
    if (!msl) GEOM_FAIL("pass-through vertex function MSL failed: %s", log ? log : "?");

    /* gl_Layer / gl_ViewportIndex become Metal rasterization controls */
    if (gp->out_has_layer)
    {
        char *n = msl_promote_attribute(msl, "mgl_out_layer", "render_target_array_index");
        if (!n) { free(msl); GEOM_FAIL("could not forward gl_Layer to render_target_array_index"); }
        free(msl);
        msl = n;
    }
    if (gp->out_has_vp)
    {
        char *n = msl_promote_attribute(msl, "mgl_out_vp", "viewport_array_index");
        if (!n) { free(msl); GEOM_FAIL("could not forward gl_ViewportIndex to viewport_array_index"); }
        free(msl);
        msl = n;
    }

    /* the rstate clip-space fixup belongs to the last stage before rasterization */
    {
        char *patched = mgl_rstate_patch_vertex_msl(msl, gp->draw_entry);
        if (patched)
        {
            free(msl);
            msl = patched;
        }
    }

    /* the pass-through re-exports gl_CullDistance for a fragment stage reading it: the
     * flattened members need the same repair as the plain vertex MSL (msl_reserved.c) */
    {
        char *fixed_cull = mgl_msl_fix_cull_distance(msl);
        if (fixed_cull) { free(msl); msl = fixed_cull; }
    }

    /* a point size is only meaningful for point output; some Metal implementations reject a
     * pipeline whose vertex function writes point_size together with an explicit
     * inputPrimitiveTopology (which layered / viewport-indexed output needs) */
    if (gp->out_type != GL_POINTS)
    {
        char *psz = strstr(msl, "[[point_size]]");
        if (psz)
        {
            size_t len = strlen("[[point_size]]");
            memmove(psz, psz + len, strlen(psz + len) + 1);
        }
    }

    gp->draw_msl = msl;
    geom_dump("draw.msl", msl);
    gp->linked = true;

    /* compile the pass-through function: it stands in for the vertex stage during the
     * raster pass.  A pipeline's composite is rebuilt from the stage programs on every
     * draw, so the substitution is made per draw rather than once at link time */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
    {
        void *lib = NULL, *fn = NULL;
        char *err = NULL;

        if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, gp->draw_entry, &lib, &fn, &err))
        {
            char *msg = err ? xstrdup(err) : NULL;
            free(err);
            GEOM_FAIL("pass-through vertex function Metal compile failed: %s", msg ? msg : "?");
        }
        if (gp->draw_library) ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_library);
        if (gp->draw_function) ctx->mtl_funcs.ext_program->release_obj(ctx, gp->draw_function);
        gp->draw_library = lib;
        gp->draw_function = fn;

        /* with a geometry stage the fragment shader's gl_PrimitiveID comes from the
         * record the pass-through function feeds, not from the rasterizer */
        if (gp->fs_uses_prim && gp->out_has_prim && !getenv("MGL_GEOM_NO_FSPRIM"))
            geom_patch_fs_primitive_id(ctx, gp, program);
    }
    else
        GEOM_FAIL("no Metal compile hook available");

#undef GEOM_FAIL

done:
    free(gs_src);
    free(vs_src);
    free(ir);
    free(log);
}

/* whole-word identifier search: `name` must not be part of a longer identifier */
static const char *geom_ident_find(const char *hay, const char *name, size_t nlen)
{
    const char *p = hay;

    if (!nlen || !name) return NULL;
    while ((p = strstr(p, name)) != NULL)
    {
        if ((p == hay || (!isalnum((unsigned char)p[-1]) && p[-1] != '_')) &&
            (!p[nlen] || (!isalnum((unsigned char)p[nlen]) && p[nlen] != '_')))
            return p;
        p += nlen;
    }
    return NULL;
}

/* small string-splice helpers for the MSL surgery below (one-shot, link time) */
static char *geom_str_cut(const char *base, size_t from, size_t to)
{
    size_t n = strlen(base);
    char *out;

    if (to > n) to = n;
    if (from > to) from = to;
    out = xmalloc(n - (to - from) + 1);
    memcpy(out, base, from);
    memcpy(out + from, base + to, n - to + 1);
    return out;
}

static char *geom_str_insert(const char *base, size_t at, const char *ins)
{
    size_t n = strlen(base), m = strlen(ins);
    char *out = xmalloc(n + m + 1);

    if (at > n) at = n;
    memcpy(out, base, at);
    memcpy(out + at, ins, m);
    memcpy(out + at + m, base + at, n - at + 1);
    return out;
}

/* Redirect the fragment function's gl_PrimitiveID attribute onto the varying the
 * pass-through vertex function writes: Metal's [[primitive_id]] is the rasterizer's
 * primitive index, which ignores the geometry stage entirely (an emitted strip becomes
 * several raster primitives, all of which must see the value the GS wrote).
 *
 * The AGX device layer rejects integer fragment inputs as bare entry parameters (it
 * misreads them as color attachment inputs), so the id moves into the entry's
 * [[stage_in]] struct — exactly where SPIRV-Cross puts every other varying — and the
 * body's references are renamed onto the struct member. */
static void geom_patch_fs_primitive_id(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    const char *fmsl = program->spirv[_FRAGMENT_SHADER].msl_str;
    const char *attr = fmsl ? strstr(fmsl, "[[primitive_id]]") : NULL;
    const char *name_end, *pb, *pe, *pl, *p;
    char entry[96], entry_open[104], in_struct[112], member[160], name[80];
    char *cur = NULL, *tmp;
    size_t nlen, from, to, at;
    const char *body, *hit;
    char stage_in_name[32];
    void *lib = NULL, *fn = NULL;
    char *err = NULL;
    const char *fs_entry;
    Shader *fs = program->shader_slots[_FRAGMENT_SHADER];

    if (!attr || !fs) return;

    /* the parameter holding the attribute is "uint <name> [[primitive_id]]" */
    pb = attr;
    while (pb > fmsl && *pb != ',' && *pb != '(') pb--;
    if (pb == fmsl || (*pb != ',' && *pb != '(')) return;
    pe = attr + strlen("[[primitive_id]]");
    while (*pe == ' ') pe++;
    if (*pe != ',' && *pe != ')') return;               /* trailing attributes: not handled */

    name_end = attr;
    while (name_end > pb && name_end[-1] == ' ') name_end--;
    p = name_end;
    while (p > pb && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) p--;
    nlen = (size_t)(name_end - p);
    if (!nlen || nlen >= sizeof name) return;
    memcpy(name, name_end - nlen, nlen);
    name[nlen] = 0;

    /* entry name: the identifier ending at the parameter list's '(' */
    pl = pb;
    while (pl > fmsl && *pl != '(') pl--;
    if (pl == fmsl) return;
    {
        const char *e2 = pl;
        while (e2 > fmsl && e2[-1] == ' ') e2--;
        p = e2;
        while (p > fmsl && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) p--;
        if (p == e2 || (size_t)(e2 - p) >= sizeof entry - 8) return;
        memcpy(entry, p, (size_t)(e2 - p));
        entry[e2 - p] = 0;
    }
    snprintf(entry_open, sizeof entry_open, "%s(", entry);
    snprintf(in_struct, sizeof in_struct, "struct %s_in", entry);
    snprintf(member, sizeof member, "    uint %.*s [[user(locn%u)]];\n", (int)nlen, name_end - nlen, gp->prim_loc);

    /* 1. drop the parameter (keeping the list a valid parameter list) */
    from = (size_t)(pb - fmsl) + 1;
    if (*pe == ',')
    {
        to = (size_t)(pe - fmsl) + 1;
    }
    else
    {
        to = (size_t)(pe - fmsl);
        if (*pb == ',') from = (size_t)(pb - fmsl);
    }
    cur = geom_str_cut(fmsl, from, to);

    /* the stage_in parameter's name (SPIRV-Cross uses "in"; created below otherwise) */
    snprintf(stage_in_name, sizeof stage_in_name, "in");

    /* 2. add the id as a member of the entry's input struct (created if the shader has
     *    no other varyings) */
    if ((hit = strstr(cur, in_struct)) != NULL)
    {
        const char *close = strstr(hit, "};");
        if (!close) { free(cur); return; }
        /* existing [[stage_in]] parameter: keep its name */
        {
            const char *si = strstr(cur, "[[stage_in]]");
            if (si)
            {
                const char *e2 = si;
                while (e2 > cur && e2[-1] == ' ') e2--;
                p = e2;
                while (p > cur && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) p--;
                if (e2 - p > 0 && (size_t)(e2 - p) < sizeof stage_in_name)
                {
                    memcpy(stage_in_name, p, (size_t)(e2 - p));
                    stage_in_name[e2 - p] = 0;
                }
            }
        }
        tmp = geom_str_insert(cur, (size_t)(close - cur), member);
        free(cur);
        cur = tmp;
    }
    else
    {
        const char *esig = strstr(cur, entry_open);
        const char *line;
        char decl[256];

        if (!esig) { free(cur); return; }
        line = esig;
        while (line > cur && line[-1] != '\n') line--;
        snprintf(decl, sizeof decl, "%s\n{\n%s};\n\n", in_struct, member);
        tmp = geom_str_insert(cur, (size_t)(line - cur), decl);
        free(cur);
        cur = tmp;

        esig = strstr(cur, entry_open);
        at = (size_t)(esig - cur) + strlen(entry_open);
        {
            char param[160];
            snprintf(param, sizeof param, " %s_in %s [[stage_in]]%s", entry, stage_in_name,
                     (cur[at] == ')') ? "" : ",");
            tmp = geom_str_insert(cur, at, param);
            free(cur);
            cur = tmp;
        }
    }

    /* 3. rename the body's references onto the struct member */
    {
        const char *esig = strstr(cur, entry_open);
        const char *brace;

        if (!esig) { free(cur); return; }
        brace = strchr(esig, '{');
        if (!brace) { free(cur); return; }
        body = brace + 1;

        {
            size_t body_at = (size_t)(body - cur);
            SB b;
            const char *s = body;
            const char *nm = name;

            memset(&b, 0, sizeof b);
            sb_put(&b, "%.*s", (int)body_at, cur);
            while ((hit = geom_ident_find(s, nm, nlen)) != NULL)
            {
                sb_put(&b, "%.*s%s.%.*s", (int)(hit - s), s, stage_in_name, (int)nlen, nm);
                s = hit + nlen;
            }
            sb_put(&b, "%s", s);
            free(cur);
            cur = b.s;      /* ownership moves */
        }
    }

    /* 4. compile and install (a failure keeps the original function: gl_PrimitiveID
     *    then reports the raster primitive index, which is wrong for a GS but keeps
     *    the program drawing) */
    fs_entry = program->spirv[_FRAGMENT_SHADER].entry_point ?
               program->spirv[_FRAGMENT_SHADER].entry_point : fs->entry_point;
    if (!ctx->mtl_funcs.ext_program || !ctx->mtl_funcs.ext_program->compile_msl ||
        !ctx->mtl_funcs.ext_program->compile_msl(ctx, cur, fs_entry, &lib, &fn, &err))
    {
        free(cur);
        free(err);
        return;
    }

    free(program->spirv[_FRAGMENT_SHADER].msl_str);
    program->spirv[_FRAGMENT_SHADER].msl_str = cur;
    if (fs->mtl_data.library) ctx->mtl_funcs.ext_program->release_obj(ctx, fs->mtl_data.library);
    if (fs->mtl_data.function) ctx->mtl_funcs.ext_program->release_obj(ctx, fs->mtl_data.function);
    fs->mtl_data.library = lib;
    fs->mtl_data.function = fn;
    geom_dump("fsprim.msl", cur);
}
/* post-link hook: the emulation is built here unless the program has no vertex stage of
 * its own (a separable geometry-only program gets it from the pipeline at draw time) */
void mgl_geom_post_link(GLMContext ctx, Program *program)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);

    /* link-time fact for the per-draw mgl_geom_cull_active: scanning the VS MSL
     * for "[[user(cull" on every draw cost glmark2 ideas 42%.  A tessellation
     * program's evaluation stage owns the writes instead of the vertex stage. */
    program->writes_cull_distance =
        (!program->shader_slots[_GEOMETRY_SHADER] &&
         ((program->shader_slots[_TESS_EVALUATION_SHADER] &&
           geom_cull_writes_msl_tes(program)) ||
          (!program->shader_slots[_TESS_EVALUATION_SHADER] &&
           geom_cull_writes_msl(program)))) ? 1 : 0;

    /* no geometry stage: the program has no geometry emulation, and creating a slot for
     * it (the old behaviour) made EVERY link walk/grow the program table — a VS+FS
     * program per sub-case decayed quadratically (texture_swizzle smoke) */
    if (!program->shader_slots[_GEOMETRY_SHADER])
    {
        /* cull-distance emulation (tess domain, tail of this file): without a geometry
         * stage, a vertex stage — or a tessellation program's evaluation stage — that
         * writes gl_CullDistance needs the synthetic kernel slot; the kernel itself is
         * built at the first draw (its input layout depends on the primitive family) */
        if (program->writes_cull_distance)
        {
            MGLGeomProgram *gp = geom_prog_slot(ctx, program, true);
            if (gp && !gp->linked && !gp->synthetic)
            {
                memset(gp, 0, sizeof *gp);
                gp->program = program;
                gp->name = program->name;
                gp->synthetic = true;    /* linked stays false: built at first draw */
            }
        }
        else
        {
            MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
            if (gp && gp->synthetic && (gp->linked || gp->cull_family == -1 || gp->cull_family != 0))
                geom_free_prog(ctx, gp);  /* stale: rebuild lazily after this relink */
        }
        return;
    }

    MGLGeomProgram *gp = geom_prog_slot(ctx, program, true);
    if (!gp || !obj) return;
    if (gp->synthetic)
        geom_free_prog(ctx, gp);          /* a real geometry stage supersedes the cull emulation */
    else if (gp->capture_function || gp->kernel_msl)
    {
        /* r100: a RE-LINK rebuilds every generated string/function; the old ones were
         * simply overwritten before (leaking the retained MTL functions and the MSL
         * text on every relink), and the capture-pass setup cache went stale.  Free
         * the previous build first — geom_free_prog zeroes the slot, so name/program
         * are restored right after (the pattern of the two other slot (re)users). */
        geom_free_prog(ctx, gp);
        gp->name = program->name;
        gp->program = program;
    }

    gp->gs_exec = program->shader_slots[_GEOMETRY_SHADER];
    gp->deferred = !program->shader_slots[_VERTEX_SHADER];
    if (gp->deferred) return;               /* separable: built at first draw */

    geom_build(ctx, gp, program);
}

/* make sure the emulation exists for the program that is about to draw */
static bool geom_ensure(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    if (!gp || !program) { if (getenv("MGL_GEOM_TRACE_DRAW")) fprintf(stderr, "geom_ensure: null gp/program\n"); return false; }
    if (gp->linked) return true;
    if (!program->shader_slots[_VERTEX_SHADER]) { if (getenv("MGL_GEOM_TRACE_DRAW")) fprintf(stderr, "geom_ensure: prog %u no VS\n", program->name); return false; }

    /* a relink of the geometry program (glUseProgramStages relinks its stage programs)
     * leaves the slot reflected-empty: redo it from the program's own geometry stage */
    if ((!gp->out_type || !gp->out_words) && ctx->state.program_pipeline)
    {
        Program *gs = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_GEOMETRY_SHADER];
        if (gs && gs->spirv[_GEOMETRY_SHADER].ir)
            mgl_geom_compile_stage(ctx, gs, MGL_PROGRAM_OBJ(gs), _GEOMETRY_SHADER,
                                   gs->spirv[_GEOMETRY_SHADER].ir, gs->spirv[_GEOMETRY_SHADER].size,
                                   "main", NULL);
    }

    /* a separable geometry-only program reflected its own stage only: the inputs' producing
     * vertex outputs and the real clip sizes come from the composite the pipeline builds,
     * which carries the vertex stage.  geom_build() reads them from gp->in_v[].expr. */
    if (gp->deferred && program->spirv[_VERTEX_SHADER].ir && gp->n_in)
    {
        char *llog = NULL;

        if (!geom_derive_exprs(gp, program, &llog))
        {
            geom_fail(ctx, gp, "%s", llog ? llog : "geometry inputs could not be matched");
            free(llog);
            return false;
        }
        /* llog stays owned by this frame: geom_derive_exprs() only fills it on
         * warning/error paths, and the single free is after the clip-reflection
         * block below (removing a second free(llog) here — it double-freed and
         * aborted the process whenever a warning had allocated the log). */
        {
            const char *vs_src = geom_vs_source(ctx, MGL_PROGRAM_OBJ(program));
            char *vs_dec = NULL;
            GLuint n;

            if (!vs_src)
            {
                vs_dec = geom_spirv_to_glsl(program->spirv[_VERTEX_SHADER].ir,
                                            program->spirv[_VERTEX_SHADER].size, 460, &llog);
                vs_src = vs_dec;
            }
            if (vs_src)
            {
                n = geom_declared_clip_words(vs_src, "gl_ClipDistance");
                gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
                n = geom_declared_clip_words(vs_src, "gl_CullDistance");
                if (gp->in_has_cull)
                {
                    /* the record carries the clip AND the cull array at one shared
                     * width (geom_layout), so the wider of the two declared sizes
                     * wins - a smaller cull array must not shrink the clip words
                     * (clip=1 + cull=7 kept 1 word and dropped cull[1..6]) */
                    GLuint nc2 = n ? n : MGL_GEOM_MAX_CLIP;
                    if (nc2 > gp->in_clip_size) gp->in_clip_size = nc2;
                }
                gp->out_clip_size = (gp->out_has_clip || gp->out_has_cull) ?
                                    (gp->in_clip_size ? gp->in_clip_size : MGL_GEOM_MAX_CLIP) : 0;
                geom_layout(gp);
            }
            free(vs_dec);
            if (getenv("MGL_GEOM_TRACE_DRAW"))
                fprintf(stderr, "geom_ensure: free(llog=%p)\n", (void *)llog);
            free(llog);
        }
    }

    gp->deferred = false;
    geom_build(ctx, gp, program);
    return gp->linked;
}

bool mgl_geom_link_failed(GLMContext ctx, Program *program, const char **reason)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    if (!program->shader_slots[_GEOMETRY_SHADER]) return false;
    if (!gp || gp->linked || gp->deferred) return false;
    if (reason) *reason = gp->fail ? gp->fail : "geometry stage failed";
    return true;
}

/* like mgl_program_check_xfb_link: the post-link hook builds the emulation; a failure there
 * fails the link and is reported through the program info log */
bool mgl_program_check_geom_link(GLMContext ctx, Program *program)
{
    const char *glog = NULL;
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);

    if (!mgl_geom_link_failed(ctx, program, &glog)) return true;

    obj->link_status = false;
    if (getenv("MGL_SHADER_DUMP"))
        fprintf(stderr, "MGL: program %u geom link failed: %s\n", obj->base.name, glog ? glog : "?");
    if (glog && glog[0])
    {
        size_t len = strlen(glog) + (obj->info_log ? strlen(obj->info_log) : 0) + 8;
        char *old_log = obj->info_log;
        obj->info_log = malloc(len);
        if (obj->info_log)
            snprintf(obj->info_log, len, "%s%s%s", old_log ? old_log : "",
                     old_log ? "\n" : "", glog);
        free(old_log);
    }
    return false;
}

/* ------------------------------------------------------------------ program queries */

bool mgl_geom_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    /* the four geometry queries are answered from the geometry stage's OWN reflection (the
     * SPIR-V execution modes), which is complete as soon as the stage compiled — including a
     * separable geometry-only program whose emulation is only built at first draw
     * (KHR-GL46.geometry_shader.api.getProgramiv3 queries a linked GS-only SPO that never
     * draws).  gp->kernel_entry is set by mgl_geom_compile_stage, deferred or not. */
    if (!gp || !(gp->linked || gp->kernel_entry)) return false;

    switch (pname)
    {
        case GL_GEOMETRY_VERTICES_OUT:  *params = (GLint)gp->max_vertices; return true;
        case GL_GEOMETRY_INPUT_TYPE:    *params = (GLint)gp->in_type; return true;
        case GL_GEOMETRY_OUTPUT_TYPE:   *params = (GLint)gp->out_type; return true;
        case GL_GEOMETRY_SHADER_INVOCATIONS: *params = (GLint)gp->invocations; return true;
        default: return false;
    }
}

/* ------------------------------------------------------------------ draw */

static MGLGeomProgram *geom_for_program(GLMContext ctx, Program *p)
{
    if (!p || !p->shader_slots[_GEOMETRY_SHADER]) return NULL;
    return geom_prog_slot(ctx, p, false);
}

MGLGeomProgram *mgl_geom_current(GLMContext ctx)
{
    MGLGeomProgram *gp;

    if (ctx->state.program)
    {
        gp = geom_for_program(ctx, ctx->state.program);
        if (gp) return gp;
        /* the cull-distance emulation (tess domain, tail of this file): a program whose
         * vertex stage — or, for a tessellation program, its evaluation stage — writes
         * gl_CullDistance draws through a synthetic pass-through */
        if (!ctx->state.program->shader_slots[_GEOMETRY_SHADER])
        {
            gp = geom_prog_slot(ctx, ctx->state.program, false);
            if (gp && gp->synthetic) return gp;
        }
    }
    if (ctx->state.program_pipeline)
    {
        /* a pipeline resolves through the program that owns the geometry stage (the
         * composite it builds for the renderer is a fresh object every time) */
        Program *gs = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_GEOMETRY_SHADER];
        if (gs)
        {
            gp = geom_for_program(ctx, gs);
            if (gp) return gp;
        }
        else
        {
            /* cull-distance emulation: the vertex stage program owns the slot (a
             * pipeline tessellation stage routes through the same slot: the capture
             * path keys on the TES program's slot just like a plain program) */
            Program *vs = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_VERTEX_SHADER];
            Program *tes = MGL_PIPELINE_OBJ(STATE(program_pipeline))->base.stage_programs[_TESS_EVALUATION_SHADER];
            if (vs && !tes)
            {
                gp = geom_prog_slot(ctx, vs, false);
                if (gp && gp->synthetic) return gp;
            }
            if (tes)
            {
                gp = geom_prog_slot(ctx, tes, false);
                if (gp && gp->synthetic) return gp;
            }
        }
    }
    return NULL;
}

bool mgl_geom_present(GLMContext ctx)
{
    if (!ctx->state.ext_geom) return false;
    if (ctx->state.program && ctx->state.program->shader_slots[_GEOMETRY_SHADER]) return true;
    if (ctx->state.program_pipeline)
    {
        /* r83: no composite build — mgl_geom_route asks this on EVERY draw, and the
         * pipeline composite (a fresh Program: memset + per-stage resource-list
         * copies) was built and thrown away once per draw for this one boolean.
         * The composite's slot is stage_programs[GS]->shader_slots[GS] by
         * construction (mgl_program_pipeline_composite copies it only when the
         * stage program has the stage attached), so reading the stage program
         * directly answers the identical question. */
        Program *p = MGL_PIPELINE_OBJ(ctx->state.program_pipeline)->base.stage_programs[_GEOMETRY_SHADER];
        return p && p->shader_slots[_GEOMETRY_SHADER] != NULL;
    }
    return false;
}

/* does the draw mode feed the geometry stage's input primitive type? */
static bool geom_mode_matches(GLenum in_type, GLenum mode)
{
    switch (in_type)
    {
        case GL_POINTS:                 return mode == GL_POINTS;
        case GL_LINES:                  return mode == GL_LINES || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP;
        case GL_LINES_ADJACENCY:        return mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY;
        case GL_TRIANGLES:              return mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN;
        case GL_TRIANGLES_ADJACENCY:    return mode == GL_TRIANGLES_ADJACENCY || mode == GL_TRIANGLE_STRIP_ADJACENCY;
        default: return false;
    }
}

/* number of input primitives of `count` vertices (per instance) */
static uint64_t geom_prim_count(GLenum gs_in, GLenum mode, uint64_t n)
{
    switch (mode)
    {
        case GL_POINTS:                    return n;
        case GL_LINES:                     return n / 2;
        case GL_LINE_STRIP:                return n >= 2 ? n - 1 : 0;
        case GL_LINE_LOOP:                 return n >= 2 ? n : 0;
        case GL_LINES_ADJACENCY:           return n / 4;
        case GL_LINE_STRIP_ADJACENCY:      return n >= 4 ? n - 3 : 0;
        case GL_TRIANGLES:                 return n / 3;
        case GL_TRIANGLE_STRIP:            return n >= 3 ? n - 2 : 0;
        case GL_TRIANGLE_FAN:              return n >= 3 ? n - 2 : 0;
        case GL_TRIANGLES_ADJACENCY:       return n / 6;
        case GL_TRIANGLE_STRIP_ADJACENCY:  return n >= 6 ? (n - 4) / 2 : 0;
        default: return 0;
    }
    (void)gs_in;
}

/* assemble one sequence of `n` vertex slots into input primitives */
static uint64_t geom_assemble(GLMContext ctx, MGLGeomProgram *gp, GLenum mode, const uint32_t *idx,
                              uint64_t n, uint32_t base, uint32_t *out, uint64_t cap)
{
    uint64_t np = 0;
    bool provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;
    GLuint k = gp->in_verts;

#define EMIT1(a)      do { if (np < cap) { out[np * k + 0] = idx[a] + base; np++; } } while (0)
#define EMIT2(a, b)   do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; np++; } } while (0)
#define EMIT3(a, b, c) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                out[np * k + 2] = idx[c] + base; np++; } } while (0)
#define EMIT4(a, b, c, d) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                  out[np * k + 2] = idx[c] + base; out[np * k + 3] = idx[d] + base; np++; } } while (0)
#define EMIT6(a, b, c, d, e, f) do { if (np < cap) { out[np * k + 0] = idx[a] + base; out[np * k + 1] = idx[b] + base; \
                                                        out[np * k + 2] = idx[c] + base; out[np * k + 3] = idx[d] + base; \
                                                        out[np * k + 4] = idx[e] + base; out[np * k + 5] = idx[f] + base; np++; } } while (0)

    for (uint64_t i = 0; i + k <= n; i += (mode == GL_LINES || mode == GL_TRIANGLES ||
                                           mode == GL_LINES_ADJACENCY || mode == GL_TRIANGLES_ADJACENCY) ? k : 1)
    {
        switch (mode)
        {
            case GL_POINTS:
                EMIT1(i);
                break;
            case GL_LINES:
            case GL_LINE_STRIP:
                EMIT2(i, i + 1);
                break;
            case GL_LINE_LOOP:
                /* the loop's closing segment (n-1, 0): the loop bound i + k <= n stops at
                 * i = n - 2, so the wrap-around segment is emitted after the loop (the
                 * dead `else` arm above never fired for the same reason) */
                if (i + 1 < n) EMIT2(i, i + 1);
                break;
            case GL_LINES_ADJACENCY:
            case GL_LINE_STRIP_ADJACENCY:
                EMIT4(i, i + 1, i + 2, i + 3);
                break;
            case GL_TRIANGLES:
            case GL_TRIANGLE_FAN:
                if (mode == GL_TRIANGLE_FAN)
                {
                    if (provoking_first) EMIT3(i + 1, i + 2, 0);
                    else EMIT3(0, i + 1, i + 2);
                }
                else EMIT3(i, i + 1, i + 2);
                break;
            case GL_TRIANGLE_STRIP:
                if ((i & 1) == 0) EMIT3(i, i + 1, i + 2);
                else if (provoking_first) EMIT3(i + 2, i, i + 1);
                else EMIT3(i + 1, i, i + 2);
                break;
            case GL_TRIANGLES_ADJACENCY:
                EMIT6(i, i + 1, i + 2, i + 3, i + 4, i + 5);
                break;
            case GL_TRIANGLE_STRIP_ADJACENCY:
            {
                /* GL 4.6 table 10.4 (as the CTS encodes it): even slots are triangle
                 * vertices, odd ones adjacency; odd triangles flip winding; the first and
                 * last triangles have no preceding/following adjacency vertex. The strip
                 * advances two stream vertices per triangle. */
                uint64_t T = n >= 4 ? (n - 4) / 2 : 0;

                if (np < T)
                {
                    uint64_t b = 2 * np, e = (np & 1) == 0;
                    uint64_t v0 = e ? b : b + 2, v1 = e ? b + 2 : b, v2 = b + 4;
                    uint64_t a0 = np == 0 ? 1 : b - 2;
                    uint64_t a1 = np == T - 1 ? (e ? b + 5 : b + 3) : (e ? b + 6 : b + 3);
                    uint64_t a2 = np == T - 1 ? (e ? b + 3 : b + 5) : (e ? b + 3 : b + 6);

                    EMIT6(v0, a0, v1, a1, v2, a2);
                }
                break;
            }
            default:
                break;
        }
    }
    if (mode == GL_LINE_LOOP && n >= 2 && np < cap)
    {
        /* close the loop: (n-1, 0) */
        out[np * k + 0] = idx[n - 1] + base;
        out[np * k + 1] = idx[0] + base;
        np++;
    }

#undef EMIT1
#undef EMIT2
#undef EMIT3
#undef EMIT4
#undef EMIT6

    return np;
}

/* the largest number of input primitives `n` vertices can form (the prep kernel's regions
 * are sized by this: the primitive count of the mode ignoring primitive restart) */
uint64_t mgl_geom_worst_prims(GLenum mode, uint64_t n)
{
    return geom_prim_count(GL_TRIANGLES, mode, n);
}

/* read the index list of an indexed draw as uint32 (handles restart by skipping runs) */
static uint32_t *geom_index_list(GLMContext ctx, GLenum type, size_t offset, GLsizei count, uint32_t *restart_out)
{
    Buffer *eb = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
    uint32_t *out = xmalloc(sizeof(uint32_t) * (size_t)(count > 0 ? count : 1));
    const uint8_t *p;
    /* the type's largest representable value, reported for the callers' min/max
     * window EXCLUSION: when restart is disabled it still keeps a stray sentinel
     * out of the window, and when it is enabled the CPU assembly route overrides
     * this with the draw's real sentinel (fixed-index mode or a custom
     * glPrimitiveRestartIndex) after the call — geom_index_list does NOT know the
     * restart state. */
    GLuint restart = (type == GL_UNSIGNED_BYTE) ? 0xff : (type == GL_UNSIGNED_SHORT) ? 0xffff : 0xffffffffu;

    if (restart_out) *restart_out = restart;

    mgl_buffers_prepare_cpu_read(ctx, eb);
    if (!eb->data.buffer_data)
    {
        free(out);
        return NULL;
    }
    p = (const uint8_t *)eb->data.buffer_data + offset;

    for (GLsizei i = 0; i < count; i++)
    {
        if (type == GL_UNSIGNED_BYTE) out[i] = p[i];
        else if (type == GL_UNSIGNED_SHORT) out[i] = ((const uint16_t *)p)[i];
        else out[i] = ((const uint32_t *)p)[i];
    }
    return out;
}

/* the GL draw mode as the prep kernel's code (see kGeomPrepMSL) */
int mgl_geom_mode_code(GLenum mode)
{
    switch (mode)
    {
        case GL_POINTS:                   return 0;
        case GL_LINES:                    return 1;
        case GL_LINE_STRIP:               return 2;
        case GL_LINE_LOOP:                return 3;
        case GL_LINES_ADJACENCY:          return 4;
        case GL_LINE_STRIP_ADJACENCY:     return 5;
        case GL_TRIANGLES:                return 6;
        case GL_TRIANGLE_STRIP:           return 7;
        case GL_TRIANGLE_FAN:             return 8;
        case GL_TRIANGLES_ADJACENCY:      return 9;
        case GL_TRIANGLE_STRIP_ADJACENCY: return 10;
        default: return -1;
    }
}

/* is the primitive restart index enabled and what is it? */
/* affine primitive windows for a direct draw: the window-derivation code the geometry
 * kernel applies (see MGLGeomDraw.implicit_stride and the kernel's mgl_load_inputs).
 * LINE_LOOP, TRIANGLE_FAN, POLYGON and TRIANGLE_STRIP_ADJACENCY do not follow a fixed
 * pattern (the closing segment / shared apex / GL 4.6 table 10.4 adjacency layout), so
 * they return 0 = assemble.  A triangle strip's odd windows swap their leading vertices
 * (the GL strip winding / provoking-vertex rule, matching the prep kernel's case 8) -
 * encodable as a per-parity permutation, so strips use codes 2 (last-vertex convention)
 * and 3 (first-vertex); plain stride-1 line-strip windows use 1; disjoint list windows
 * encode their size + 3 (never colliding with the strip codes). */
static uint32_t geom_affine_stride(GLenum mode, uint32_t in_verts, bool provoking_first)
{
    switch (mode)
    {
        case GL_POINTS:
        case GL_LINES:
        case GL_TRIANGLES:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
            return in_verts + 3;        /* list-like: disjoint windows of in_verts */
        case GL_LINE_STRIP:
        case GL_LINE_STRIP_ADJACENCY:
            return 1;                   /* strip-like: windows advance by one vertex */
        case GL_TRIANGLE_STRIP:
            return provoking_first ? 3 : 2;
        default:
            return 0;
    }
}

static uint32_t geom_restart_index(GLMContext ctx, GLenum type)
{
    /* PRIMITIVE_RESTART_FIXED_INDEX (which wins when both modes are enabled, the
     * same precedence the CPU assembly route's split uses) restarts at the largest
     * value the index TYPE can represent; plain PRIMITIVE_RESTART uses
     * glPrimitiveRestartIndex.  0xffffffff doubles as the "no restart" sentinel:
     * the prep kernel's restart_en (and the CPU route's rst) gate that case. */
    if (ctx->state.caps.primitive_restart_fixed_index)
        return (type == GL_UNSIGNED_BYTE) ? 0xffu : (type == GL_UNSIGNED_SHORT) ? 0xffffu : 0xffffffffu;
    if (!ctx->state.caps.primitive_restart) return 0xffffffffu;
    return (uint32_t)ctx->state.var.primitive_restart_index;
}

static bool geom_route_cpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d, GLenum gs_in);
static bool geom_route_gpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d);

/* the last context that ran a geometry draw (the trace is a debugging aid) */
static GLMContext gTraceCtx;
void *mgl_geom_ctx(void) { return gTraceCtx; }

/* ---- r82 diagnostics: monotonic sidecar counters (tests/headless/mgl_geom_diag.h).
 * The headless suite asserts on these to prove the MS/atomic sidecar costs NOTHING
 * when a program declares no image atomics (seeds 0, flushes 0, resolves 0) and
 * that the passes run exactly once when one does — structural, no timing.
 * which: 0 seeds, 1 flushes, 2 compute resolves, 3 sidecar allocations. */
uint64_t mgl_geom_ms_stat(int which)
{
    GLMContext c = (GLMContext)gTraceCtx;
    struct MGLExtState_geom *g;

    if (!c) return 0;
    g = (struct MGLExtState_geom *)c->state.ext_geom;
    if (!g) return 0;
    switch (which)
    {
        case 0: return g->stat_seeds;
        case 1: return g->stat_flushes;
        case 2: return g->stat_resolves;
        case 3: return g->stat_sidecar_allocs;
        case 4: return g->stat_gate_misses;   /* r83: sidecar-gate memo misses (once per program link) */
        default: return 0;
    }
}

/* ---- tessellation in front of the geometry stage ---- */

/* the tess domain tessellates this draw with its TES-output capture enabled (the records and
 * the output index list are GPU-visible), then the geometry emulation consumes them: the
 * import kernel (MGLRenderer+Geometry.m) repacks the capture records into this program's
 * input record and builds the primitive slots, exactly what the prep kernel produces for a
 * plain draw. */
static bool geom_route_tess(GLMContext ctx, MGLGeomProgram *gp, bool indexed, GLenum type,
                            size_t offset, GLint first, GLsizei count, GLsizei instances,
                            GLint basevertex, GLuint baseinstance)
{
    Program *program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                   : ctx->state.program;
    const MGLTessCapture *c;
    MGLGeomDraw d; memset(&d, 0, sizeof d);   /* route fields read by the renderer must never hold stack garbage */
    GLuint saved_discard;
    uint32_t want = gp->in_type;      /* the GS input primitive the capture must produce */
    static const uint32_t kof[3] = { 1u, 2u, 3u };     /* GL_POINTS, GL_LINES, GL_TRIANGLES */

    if (!program || !program->shader_slots[_TESS_EVALUATION_SHADER]) return false;

    /* the tess stage must not rasterize: the geometry stage owns rasterization, and the
     * fragment shader would otherwise see the tessellation outputs.  rstate discards in the
     * last pre-rasterization stage's vertex function, so the tess pipeline is rebuilt for it
     * (the geometry pass restores the value afterwards, before its own raster pass). */
    mgl_tess_set_capture(ctx, program, true);
    saved_discard = ctx->state.caps.rasterizer_discard;
    ctx->state.caps.rasterizer_discard = GL_TRUE;
    ctx->state.dirty_bits |= DIRTY_PROGRAM;
    mgl_tess_draw_patches(ctx, indexed, type, offset, first, count, instances,
                          basevertex, baseinstance);
    ctx->state.caps.rasterizer_discard = saved_discard;
    ctx->state.dirty_bits |= DIRTY_PROGRAM;

    c = mgl_tess_get_capture(ctx);
    if (geom_env(0))
        fprintf(stderr, "geom: trace tess capture c=%p valid=%d prim=%u want=%u members=%d\n",
                (const void *)c, c ? (int)c->valid : -1, c ? c->prim : 0u, want, c ? c->members : -1);
    /* These are emulation-internal conditions, not GL spec violations: a linked
     * tess+GS program drawn with GL_PATCHES has no error case at draw time, and
     * surfacing GL_INVALID_OPERATION made the CTS's post-test glGetError() check
     * fail (shader_image_load_store *GeometryStages: a store-only TES has no
     * outputs, so no capture record to parse — see mgl_tess_capture_msl's void
     * handling).  GL leaves the rasterized results undefined instead. */
    if (!c || !c->valid || !c->buffer_object || !c->buffer)
    {
        if (geom_env(0))
            fprintf(stderr, "geom: tess capture missing after draw - geometry stage dropped\n");
        return false;
    }
    if (c->prim != want)
    {
        /* triangles -> triangles, isolines -> lines, point_mode -> points;
         * a mismatch is a link-time validation case, undefined at draw time */
        if (geom_env(0))
            fprintf(stderr, "geom: tess capture prim %u != GS input %u - geometry stage dropped\n",
                    c->prim, want);
        return false;
    }

    /* repack: for every geometry input location, the tess capture member at the same location
     * supplies the words (gl_Position is words 0..3 in both layouts) */
    memset(&d, 0, sizeof(d));
    d.tess_ring = c->buffer_object;     /* the MTLBuffer (buffer is its CPU address) */
    d.tess_recs_off = c->records_offset;
    d.tess_idx_off = c->index_offset;
    d.tess_cnt_off = c->count_offset;
    d.tess_stride = c->stride;
    d.tess_prim = c->prim;
    /* r102: while a GL_TESS_EVALUATION_SHADER_INVOCATIONS query is open (with a geometry
     * stage bound, mgl_queries_tess_counts_active answers exactly that), fold the GPU
     * tessellator's invocation count into the query accumulator (word 9): the GPU-emulated
     * tess path never runs the CPU reference counter, so without this the query reads 0.
     * tes_invocations_offset is 0 on the CPU reference path, whose invocations the CPU
     * already counted (mgl_tess_note_draw) — folding there would double-count. */
    if (c->tes_invocations_offset && mgl_queries_tess_counts_active(ctx))
        d.tess_tesinv_off = c->tes_invocations_offset;
    d.tess_pairs[d.tess_npairs++] = 0; d.tess_pairs[d.tess_npairs++] = 0;
    d.tess_pairs[d.tess_npairs++] = 1; d.tess_pairs[d.tess_npairs++] = 1;
    d.tess_pairs[d.tess_npairs++] = 2; d.tess_pairs[d.tess_npairs++] = 2;
    d.tess_pairs[d.tess_npairs++] = 3; d.tess_pairs[d.tess_npairs++] = 3;
    /* gl_in[].gl_PointSize: the capture record carries the evaluation stage's
     * gl_PointSize output as a one-word member (mgl_tess_capture_msl); map it onto
     * the input record's point-size word, which geom_layout placed at in_psize_off.
     * Without it a geometry shader reading gl_in[].gl_PointSize always saw 0 (dEQP
     * ES31 point_size vertex_set_control_pass_eval_add_geometry_add).  A capture
     * whose TES never writes gl_PointSize has no such member: gl_in[].gl_PointSize
     * is then undefined per GL, the record word stays 0. */
    if (gp->in_has_psize)
    {
        for (int m2 = 0; m2 < (int)c->members; m2++)
            if (!strcmp(c->member[m2].name, "gl_PointSize"))
            {
                d.tess_pairs[d.tess_npairs++] = c->member[m2].offset;
                d.tess_pairs[d.tess_npairs++] = gp->in_psize_off;
                break;
            }
    }
    for (int i = 0; i < gp->n_in; i++)
    {
        int j = -1, jspan = 1;
        uint32_t jw = 0;     /* words of the matched span (>= one member when the
                              * capture flattened an array into "name_q" members) */

        /* the consumer matches the tess capture's members by name: a linkable program has the
         * same names for the evaluation stage's outputs and the geometry stage's inputs.
         * A block member arrives flattened by SPIRV-Cross as "<instance>_<member>" (the
         * instance names of the two stages may differ: blocks match by block name), an
         * ARRAY member further as "<...>_<member>_<q>", one member per element */
        for (int m = 0; m < c->members; m++)
        {
            const char *want = gp->in_v[i].member[0] ? gp->in_v[i].member : gp->in_v[i].name;
            size_t wl = strlen(want), cl = strlen(c->member[m].name);
            int hit = !strcmp(c->member[m].name, want);
            int flat = 0;

            if (!hit && gp->in_v[i].member[0] && cl > wl + 1 &&
                c->member[m].name[cl - wl - 1] == '_' &&
                !strcmp(c->member[m].name + cl - wl, want))
                hit = 1;               /* "<instance>_<member>" */
            if (!hit && cl > wl + 2)
            {
                /* "<...>_<member>_<digits>": strip the element index, then the
                 * instance prefix (either or both may be present) */
                const char *e = c->member[m].name + cl;
                const char *d = e;

                while (d > c->member[m].name && d[-1] >= '0' && d[-1] <= '9') d--;
                if (d < e && d > c->member[m].name + 1 && d[-1] == '_' &&
                    (size_t)(d - c->member[m].name) - 1 >= wl)
                {
                    char base[160];
                    size_t bl = (size_t)(d - c->member[m].name) - 1;   /* without the trailing '_' */

                    if (bl >= sizeof base) bl = sizeof base - 1;
                    snprintf(base, sizeof base, "%.*s", (int)bl, c->member[m].name);
                    if (!strcmp(base, want) ||
                        (bl > wl + 1 && base[bl - wl - 1] == '_' && !strcmp(base + bl - wl, want)))
                    {
                        hit = 1;
                        flat = 1;
                    }
                }
            }
            if (hit)
            {
                uint32_t mw = c->member[m].words;
                int span = 1;

                if (flat)
                {
                    /* merge the sibling elements' words (SPIRV-Cross emits the
                     * flattened members consecutively, "name_0", "name_1", ...) */
                    size_t blen = (size_t)(strrchr(c->member[m].name, '_') - c->member[m].name);

                    for (int q = m + 1; q < c->members; q++)
                    {
                        const char *qn = c->member[q].name;
                        size_t ql = strlen(qn), i2;
                        int digits = 0;

                        if (ql <= blen || strncmp(qn, c->member[m].name, blen) || qn[blen] != '_')
                            break;
                        for (i2 = blen + 1; i2 < ql; i2++)
                            if (qn[i2] < '0' || qn[i2] > '9') { digits = 0; break; }
                            else digits = 1;
                        if (!digits)
                            break;
                        mw += c->member[q].words;
                        span++;
                    }
                }
                if (mw >= gp->in_v[i].words)
                {
                    j = m;
                    jw = mw;
                    jspan = span;
                    break;
                }
            }
        }
        if (j < 0)
        {
            if (geom_env(0))
                fprintf(stderr, "geom: tess import: no capture member for %s/%s (want %u words)\n",
                        gp->in_v[i].name, gp->in_v[i].member, gp->in_v[i].words);
            return false;   /* internal: an interface the link should have rejected */
        }
        if (jw < gp->in_v[i].words ||
            d.tess_npairs + 2 * gp->in_v[i].words > MGL_GEOM_TESS_MAX_WORDS)
        {
            if (geom_env(0))
                fprintf(stderr, "geom: tess import: %s words mismatch (cap %u vs want %u)\n",
                        gp->in_v[i].name, jw, gp->in_v[i].words);
            return false;   /* internal: an interface the link should have rejected */
        }
        if (gp->in_v[i].base == GL_DOUBLE)
        {
            /* the doubles sit two per uint4 capture slot (one per uint2 slot),
             * column-major, with a padded tail in the last slot of an odd-width
             * dvec3-style column: gather them compactly — double d of element
             * e / column c / row r is in slot e*cols*ceil(rows/2) + c*ceil + r/2,
             * half r&1 of it.  Flattened captures (one member per slot) are
             * addressed member-wise. */
            uint32_t D = gp->in_v[i].elems * gp->in_v[i].cols * gp->in_v[i].rows;
            uint32_t rows = gp->in_v[i].rows ? gp->in_v[i].rows : 1;
            uint32_t cols = gp->in_v[i].cols ? gp->in_v[i].cols : 1;
            uint32_t cslots = (rows + 1) / 2;           /* slots per column */
            uint32_t stride = (cols <= 1 && rows <= 1) ? 2u : 4u;   /* uint2 / uint4 slot */

            if (d.tess_npairs + 2 * gp->in_v[i].words > MGL_GEOM_TESS_MAX_WORDS ||
                (jspan <= 1 && jw < gp->in_v[i].words))
            {
                return false;   /* internal: import table exhausted */
            }
            for (uint32_t dd = 0; dd < D; dd++)
            {
                uint32_t e = dd / (cols * rows), rem = dd % (cols * rows);
                uint32_t cc = rem / rows, r = rem % rows;
                uint32_t slot = e * cols * cslots + cc * cslots + r / 2;
                uint32_t pr = r & 1;
                uint32_t src = (jspan > 1 && (int)slot < jspan)
                                   ? c->member[j + (int)slot].offset
                                   : c->member[j].offset + slot * stride;

                if (jspan > 1 && (int)slot >= jspan)
                    src = c->member[j + jspan - 1].offset;   /* defensive: padded tail */
                for (uint32_t half = 0; half < 2; half++)
                {
                    d.tess_pairs[d.tess_npairs++] = src + pr * 2 + half;
                    d.tess_pairs[d.tess_npairs++] = gp->in_v[i].offset + dd * 2 + half;
                }
            }
        }
        else
        {
            for (uint32_t w = 0; w < gp->in_v[i].words; w++)
            {
                d.tess_pairs[d.tess_npairs++] = c->member[j].offset + w;   /* word offset */
                d.tess_pairs[d.tess_npairs++] = gp->in_v[i].offset + w;
            }
        }
    }

    /* gl_in[].gl_ClipDistance / gl_CullDistance: the capture record carries the
     * evaluation stage's outputs as flattened per-element members
     * ("gl_ClipDistance_<i>", "gl_CullDistance_<i>" — see mgl_tess_capture_msl);
     * map every element word onto the input record's clip/cull words, which
     * geom_layout placed at in_clip_off / in_cull_off.  The synthetic cull kernel's
     * record consumes the same members (its record carries the builtin word sections
     * the emulation evaluates) */
    if (gp->in_has_clip || gp->in_has_cull)
    {
        static const char kPref[2][20] = { "gl_ClipDistance_", "gl_CullDistance_" };

        for (int b = 0; b < 2; b++)
        {
            if (b == 0 && !gp->in_has_clip)
                continue;
            if (b == 1 && !gp->in_has_cull)
                continue;
            for (GLuint i = 0; i < gp->in_clip_size; i++)
            {
                char nm[64];
                int j = -1;

                snprintf(nm, sizeof nm, "%s%u", kPref[b], i);
                for (int m2 = 0; m2 < (int)c->members; m2++)
                    if (!strcmp(c->member[m2].name, nm)) { j = m2; break; }
                if (j < 0)
                {
                    if (gp->synthetic)
                        break;  /* unwritten plane: the record word stays as the kernel left it */
                    if (geom_env(0))
                        fprintf(stderr, "geom: tess import: no capture member %s\n", nm);
                    return false;   /* internal: the link validated the declared sizes */
                }
                if (d.tess_npairs + 2 > MGL_GEOM_TESS_MAX_WORDS)
                    return false;   /* internal: import table exhausted */
                d.tess_pairs[d.tess_npairs++] = c->member[j].offset;
                d.tess_pairs[d.tess_npairs++] = (b ? gp->in_cull_off : gp->in_clip_off) + i;
            }
        }
    }

    /* sizing bounds: the capture ring block's record and index-list capacities (the tess
     * domain sizes them for this draw's patches).  The tess producer tessellates
     * patches * instances patches, so every instance is already in the capture and the
     * geometry stage sees a single instance of the assembled primitive stream. */
    d.gp = gp;
    d.program = program;
    d.mode = c->prim;                 /* the assembled primitives are of the capture's type */
    d.first = first;
    d.count = c->indices ? (GLsizei)c->indices : count;
    d.instances = 1;
    d.basevertex = basevertex;
    d.baseinstance = baseinstance;
    d.tess_rec_cap = c->records ? c->records : 1;
    d.tess_idx_cap = c->indices ? c->indices : 1;
    /* the GPU tess path permutes the capture records into emission order (one record
     * per emitted vertex): primitive slot t IS record t; the published index list is
     * the tessellator's coordinate-index list and must NOT be used as a record map.
     * The CPU reference path writes one record per unique coordinate — there the list
     * IS the map.  (Quads/isolines tess domains emit vertices in an order where the
     * two differ: KHR-GL46.texture_view.view_sampling draws a unit quad — 4 corners,
     * indices 0 1 3 1 2 3 — and the geometry stage consumed records 0,1,3,1,2,3
     * instead of the emitted 0..5, collapsing the quad into degenerate triangles.) */
    d.tess_emission = c->emission_order ? 1u : 0u;

    (void)kof;
    return METAL()->draw(ctx, &d);
}

void mgl_geom_route(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                    GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    MGLGeomProgram *gp;
    MGLGeomDraw d; memset(&d, 0, sizeof d);   /* route fields read by the renderer must never hold stack garbage */
    GLenum gs_in;

    if (!mgl_geom_present(ctx) && !mgl_geom_cull_active(ctx)) return;

    gTraceCtx = ctx;
    gp = mgl_geom_current(ctx);
    if (!gp) return;

    /* the cull-distance emulation's kernel is generated per input-primitive family (a
     * draw-time property): (re)build it before the mode check below applies */
    if (gp->synthetic && (!gp->linked || gp->cull_family != geom_cull_want_family(ctx, gp, mode)))
    {
        if (!geom_cull_prepare(ctx, gp, gp->program, mode)) return;  /* draws unculled */
    }

    if (geom_env(0))
        fprintf(stderr, "geom: trace route mode=%u indexed=%d first=%u count=%u inst=%u gp=%p linked=%d\n",
                mode, (int)indexed, first, count, instances, (void *)gp, (int)gp->linked);

    if (!gp->linked && !geom_ensure(ctx, gp,
                                    ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                                : ctx->state.program))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* a tessellation stage in front of the geometry shader: the tess domain tessellates (and
     * publishes its TES-output capture), then the geometry emulation consumes it.  The GL
     * draw mode is GL_PATCHES, so the usual input-type check does not apply — the capture's
     * output primitive type is validated against it instead. */
    if (gp->from_tess)
    {
        if (mode != GL_PATCHES)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        if (geom_route_tess(ctx, gp, indexed, type, offset, first, count, instances,
                            basevertex, baseinstance))
            return;
        /* the tess capture failed to publish (an emulation-internal condition): the
         * geometry stage is dropped and the draw's results are undefined — NOT a
         * GL error, the program and the draw are both valid (CTS post-test
         * glGetError checks) */
        return;
    }

    if (!geom_mode_matches(gp->in_type, mode))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    gs_in = gp->in_type;

    memset(&d, 0, sizeof(d));
    d.gp = gp;
    d.program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx) : ctx->state.program;
    d.mode = mode;
    d.indexed = indexed;
    d.type = type;
    d.offset = offset;
    d.first = first;
    d.count = count;
    d.instances = instances;
    d.basevertex = basevertex;
    d.baseinstance = baseinstance;
    d.restart = geom_restart_index(ctx, type);
    d.provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;
    /* affine direct draws (glDrawArrays*): the primitive windows are a fixed stride apart,
     * so the geometry kernel can derive them arithmetically and the whole prep stage
     * (one to six GPU launches per draw) is skipped.  Non-indexed draws never restart (the
     * prep ignores restarts for them, and GL only restarts non-indexed draws through
     * GL_PRIMITIVE_RESTART_FOR_PRIMITIVES, which this path does not take when enabled). */
    if (!indexed && !ctx->state.caps.primitive_restart_fixed_index)
    {
        d.implicit_stride = geom_affine_stride(mode, gp->in_verts, d.provoking_first);
        if (d.implicit_stride)
        {
            /* the capture pass draws the range directly (no prep kernel): it needs the
             * same vstart/vcount the CPU route would have supplied */
            d.vstart = (uint32_t)first;
            d.vcount = (uint32_t)(count > 0 ? count : 1);
        }
    }

    if (!METAL() || !METAL()->draw)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    /* the primitive assembly runs on the GPU whenever the element buffer is device-visible
     * and nothing forces the CPU path */
    if (!geom_env(3) &&
        METAL()->can_assemble_gpu && METAL()->can_assemble_gpu(ctx) &&
        geom_route_gpu(ctx, gp, &d))
        return;

    if (count <= 0 || instances <= 0) return;
    if (geom_route_cpu(ctx, gp, &d, gs_in)) return;

    METAL()->draw(ctx, &d);
}

/* CPU assembly: the fallback (MGL_GEOM_CPU_ASSEMBLY=1, or an element buffer the GPU cannot
 * read).  Returns true when it filled d with the lists it built. */
static bool geom_route_cpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d, GLenum gs_in)
{
    uint32_t *seq = NULL, *indices = NULL;
    uint64_t nseq = 0, nprims;
    uint32_t *prims = NULL;
    uint32_t restart = d->restart;
    uint32_t vmin = 0xffffffffu, vmax = 0;
    GLsizei index_count = 0;
    GLuint i;

    /* build the vertex sequence and (for indexed draws) the capture index list.
     * A RESTART only splits a run when primitive restart is actually ENABLED —
     * geom_restart_index returns the 0xffffffff sentinel when it is not, and a
     * WILD index with that value must survive assembly untouched (test_geom
     * wild_index_survives_cpu_assembly).  Restart entries are removed from the
     * sequence; split[i] marks that a restart precedes sequence position i.
     * (dEQP geometry_shading.basic.primitive_id_in_restarted: POINTS
     * [3, 2, restart, 1] must yield 3 points with gl_PrimitiveIDIn 0,1,2.) */
    uint8_t *split = (uint8_t *)calloc((size_t)(d->count > 0 ? d->count : 1) + 1, 1);
    if (!split) return false;
    if (d->indexed)
    {
        uint32_t *raw = geom_index_list(ctx, d->type, d->offset, d->count, &restart);
        if (!raw) { free(split); return false; }
        seq = xmalloc(sizeof(uint32_t) * (size_t)(d->count > 0 ? d->count : 1));
        indices = xmalloc(sizeof(uint32_t) * (size_t)(d->count > 0 ? d->count : 1));
        bool rst = ctx->state.caps.primitive_restart || ctx->state.caps.primitive_restart_fixed_index;
        /* the sentinel that opens a new run: geom_restart_index already carries the
         * right value for both modes (the fixed-index mode wins when both are on,
         * returning the type max; a custom glPrimitiveRestartIndex otherwise) */
        uint32_t rstv = d->restart;
        (void)restart;
        GLuint k = 0;
        for (GLsizei j = 0; j < d->count; j++)
        {
            if (rst && raw[j] == rstv)
                split[k] = 1;              /* the next kept entry opens a new run */
            else
            {
                seq[k] = raw[j];
                indices[index_count++] = raw[j];
                k++;
            }
        }
        free(raw);
        nseq = k;    /* restart entries are not part of the sequence */
    }
    else
    {
        nseq = (uint64_t)d->count;
        index_count = 0;
        seq = xmalloc(sizeof(uint32_t) * (size_t)nseq);
        for (i = 0; i < (GLuint)nseq; i++) seq[i] = (uint32_t)(d->first + (GLint)i);
    }

    /* the vertex range the capture pass must cover.  The restart sentinel is not a vertex:
     * counting it would make vmax 0xffffffff and the capture record buffer would be sized by
     * 4 billion vertices (a >100 GB allocation, which is what stalled primitive restart) */
    for (i = 0; i < (GLuint)nseq; i++)
    {
        if (d->indexed && seq[i] == restart) continue;
        if (seq[i] < vmin) vmin = seq[i];
        if (seq[i] > vmax) vmax = seq[i];
    }
    if (vmin == 0xffffffffu)
    {
        free(seq);
        free(indices);
        free(split);
        return true;              /* nothing to draw */
    }
    if (d->indexed) vmin += (uint32_t)d->basevertex, vmax += (uint32_t)d->basevertex;

    /* split on primitive restart and assemble.  split[i] marks that a restart precedes
     * sequence position i, i.e. a NEW RUN OPENS at i — the boundary does not consume i
     * (the restart entries themselves were dropped while building seq), so the walk
     * advances s = e, never s = e + 1: that swallowed the run after a trailing restart
     * and the last primitive of [3, 2, restart, 1] never reached the GS. */
    {
        uint64_t s = 0;

        nprims = 0;
        while (s < nseq)
        {
            uint64_t e = s + 1;
            while (e < nseq && !split[e]) e++;
            nprims += geom_prim_count(gs_in, d->mode, e - s);
            s = e;
        }
        if (!nprims)
        {
            free(seq);
            free(indices);
            free(split);
            return true;
        }

        prims = xmalloc(sizeof(uint32_t) * (size_t)(nprims * gp->in_verts));
        if (!prims)
        {
            free(seq);
            free(indices);
            free(split);
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return true;
        }

        {
            uint64_t np = 0;
            s = 0;
            while (s < nseq)
            {
                uint64_t e = s + 1;
                while (e < nseq && !split[e]) e++;
                if (e > s)
                {
                    uint32_t *sub = xmalloc(sizeof(uint32_t) * (size_t)(e - s));
                    uint64_t len = e - s;
                    for (uint64_t j = 0; j < len; j++) sub[j] = seq[s + j];
                    np += geom_assemble(ctx, gp, d->mode, sub, len, 0,
                                        prims + np * gp->in_verts, nprims - np);
                    free(sub);
                }
                s = e;
            }
            (void)np;
        }
    }
    free(seq);
    free(split);

    d->indices = indices;
    d->index_count = index_count;
    d->prims = prims;
    d->prims_count = nprims;
    d->vstart = vmin;
    d->vcount = vmax - vmin + 1;

    METAL()->draw(ctx, d);

    free(prims);
    free(indices);
    return true;
}

/* ------------------------------------------------------------------ init / queries */

void mgl_ext_init_geom(GLMContext ctx)
{
    if (!ctx->state.ext_geom)
        ctx->state.ext_geom = calloc(1, sizeof(struct MGLExtState_geom));
    /* calloc zeroes free_head; the free list is empty at -1 */
    if (ctx->state.ext_geom)
        ((struct MGLExtState_geom *)ctx->state.ext_geom)->free_head = -1;
    gTraceCtx = ctx;                    /* the diag accessors need a context */
    mgl_ext_register_post_link(mgl_geom_post_link);
}

bool mgl_ext_get_geom(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    return false;
}

/* GPU assembly: the prep kernel derives everything from the draw command and the element
 * buffer, so the CPU never reads the element buffer or assembles the slot list */
static bool geom_route_gpu(GLMContext ctx, MGLGeomProgram *gp, MGLGeomDraw *d)
{
    if (mgl_geom_mode_code(d->mode) < 0) return false;
    d->count_bound = (uint32_t)(d->count > 0 ? d->count : 1);
    if (d->count <= 0 || d->instances <= 0) return false;
    if (d->indexed && d->type == GL_UNSIGNED_INT && ((d->offset & 3u) != 0)) return false;
    if (d->indexed && d->type == GL_UNSIGNED_SHORT && ((d->offset & 1u) != 0)) return false;

    /* the prep kernel reads the command itself when the draw is indirect */
    d->cmd_layout = 0;
    d->cmd_buffer = NULL;
    d->cmd_offset = 0;

    /* r63 (test_geom_xfbperf page-fault root cause): the capture pass draws the prep
     * kernel's capture list as an INDEX BUFFER, so the capture variant's gl_VertexID
     * is the FETCHED INDEX VALUE and it stores its record at
     *     mgl_o = (gl_VertexID - vmin + (instanceID - baseInstance) * vcount) * in_words
     * — the record slot spans the drawn VERTEX RANGE (vmax-vmin+1), NOT the index
     * count.  Sizing the record pool by the index count ("the vertex range is only
     * known on the GPU") let a sparse indexed draw — the perf test's 30k indices
     * over a 100k-vertex buffer — write 3x past the pool slice: mostly silently into
     * the rest of the pool, and into unmapped pages whenever the slice landed near
     * the pool buffer's end (an intermittent GPU page fault that needed several
     * draws in flight for the pool to be in that state).  Read the range on the CPU
     * (direct draws only — for indirect ones the prep kernel clamps instead) and
     * hand it to the renderer through d->vstart/d->vcount, the same fields the CPU
     * route supplies. */
    if (d->indexed && d->type && d->count > 0)
    {
        Buffer *eb = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;

        if (eb && eb->data.mtl_data)
        {
            uint32_t restart = 0, vmin = 0xffffffffu, vmax = 0;
            uint32_t *ix = geom_index_list(ctx, d->type, d->offset, d->count, &restart);

            if (ix)
            {
                /* with a custom glPrimitiveRestartIndex the sentinel to keep out of
                 * the window is THAT value (d->restart), not the type max; when
                 * restart is disabled d->restart is the no-restart sentinel and the
                 * type max still keeps a stray 0xffffffff from sizing a >4GB window */
                uint32_t excl = (ctx->state.caps.primitive_restart ||
                                 ctx->state.caps.primitive_restart_fixed_index) ? d->restart : restart;
                for (GLsizei i = 0; i < d->count; i++)
                    if (ix[i] != excl)
                    {
                        if (ix[i] < vmin) vmin = ix[i];
                        if (ix[i] > vmax) vmax = ix[i];
                    }
                free(ix);
                if (vmin != 0xffffffffu)
                {
                    d->vstart = vmin;
                    d->vcount = vmax - vmin + 1;
                }
            }
        }
    }

    METAL()->draw(ctx, d);
    return true;
}

/* an indirect draw: the prep kernel reads the command from the user's buffer, so the CPU
 * never sees the counts.  Returns false when the buffer sizes cannot be bounded without
 * reading the command (then the caller falls back to the CPU route). */
bool mgl_geom_route_indirect(GLMContext ctx, GLenum mode, GLenum type, Buffer *ibuf, size_t offset,
                             const uint32_t *cmd)
{
    MGLGeomProgram *gp;
    MGLGeomDraw d; memset(&d, 0, sizeof d);   /* route fields read by the renderer must never hold stack garbage */
    uint32_t bound = 0;

    if (!mgl_geom_present(ctx)) return false;
    if (mgl_geom_mode_code(mode) < 0) return false;

    gp = mgl_geom_current(ctx);
    if (!gp) return false;
    if (!gp->linked && !geom_ensure(ctx, gp,
                                    ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx)
                                                                : ctx->state.program))
        return false;
    if (!geom_mode_matches(gp->in_type, mode)) return false;

    /* bound the vertex count without reading the command: an indexed draw cannot use more
     * indices than the element buffer holds */
    if (type)
    {
        Buffer *eb = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
        if (!eb || !eb->data.mtl_data) return false;
        bound = (uint32_t)((eb->size - offset) / (type == GL_UNSIGNED_BYTE ? 1 :
                                                  type == GL_UNSIGNED_SHORT ? 2 : 4));
    }
    else
    {
        /* arrays: no more vertices than the smallest attribute buffer can supply */
        VertexArray *vao = ctx->state.vao;
        bound = 1u << 20;
        if (vao)
        {
            for (GLuint loc = 0; loc < MAX_ATTRIBS; loc++)
            {
                VertexAttrib *a = &vao->attrib[loc];
                Buffer *ab = a->buffer;
                uint32_t n;

                if (!(vao->enabled_attribs & (1u << loc))) continue;
                if (!ab || ab->size <= 0) continue;
                n = (uint32_t)((size_t)ab->size / (a->stride ? a->stride : 16));
                if (n && n < bound) bound = n;
            }
        }
        if (bound == (1u << 20)) return false;      /* unbounded: read the command instead */
    }
    if (!bound) return false;

    if (!METAL() || !METAL()->draw || !METAL()->can_assemble_gpu || !METAL()->can_assemble_gpu(ctx))
        return false;

    memset(&d, 0, sizeof(d));
    d.gp = gp;
    d.program = ctx->state.program_pipeline ? mgl_program_pipeline_composite(ctx) : ctx->state.program;
    d.mode = mode;
    d.indexed = type != 0;
    d.type = type;
    d.offset = 0;
    d.count = 0;
    d.instances = 1;                 /* the kernel reads the real values from the command */
    d.cmd_buffer = ibuf ? ibuf->data.mtl_data : NULL;
    d.cmd_offset = offset;
    d.cmd_layout = type ? 2 : 1;
    d.restart = geom_restart_index(ctx, type);
    d.provoking_first = ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION;
    d.count_bound = bound;

    /* A CPU-known command (read for an active pipeline-statistics query, or the
     * buffer still carries its CPU copy) gives the EXACT counts: pass them along so
     * the sizing is exact instead of worst-case-with-one-instance.  Sizing for a
     * single instance truncated the output vertex buffer (and with it the emitted-
     * primitives query and the raster list) to a fraction whenever an indirect draw
     * carried primcount > 1 — KHR-GL46.pipeline_statistics_query_tests_ARB.
     * functional_geometry_shader_queries saw 6 of 12 (lines) / 3 of 12 (triangles). */
    if (cmd)
    {
        uint64_t vc = cmd[0], prims;

        if (vc > bound) vc = bound;              /* never trust count past the index/data bound */
        prims = geom_prim_count(GL_TRIANGLES, mode, vc);
        if (prims > bound) prims = bound;        /* restart runs only shrink it */
        d.known_prims     = (uint32_t)prims;
        d.known_instances = cmd[1] ? cmd[1] : 1;
        d.known_vcount    = (uint32_t)vc;
    }

    if (!d.cmd_buffer) return false;

    METAL()->draw(ctx, &d);
    return true;
}

/* ------------------------------------------------------------------ transform feedback of the geometry output
 * With a geometry stage active, GL captures the vertices the geometry shader emits to the
 * rasterization stream.  The emulation has them in the output vertex buffer already, so the
 * capture is a CPU pack of those records into the transform feedback buffers. */

static MGLXfbProgram *geom_xfb_prog(GLMContext ctx, Program *p, bool create)
{
    struct MGLExtState_xfb *x = ctx->state.ext_xfb;

    if (!x || !p) return NULL;
    for (GLuint i = 0; i < x->prog_count; i++)
    {
        MGLXfbProgram *xp = x->progs[i];
        if (xp->name == p->name)
        {
            if (xp->program != p) xp->program = p;
            return xp;
        }
    }
    if (!create) return NULL;
    if (x->prog_count == x->prog_cap)
    {
        GLuint cap = x->prog_cap ? x->prog_cap * 2 : 16;
        MGLXfbProgram **n = realloc(x->progs, cap * sizeof(*n));
        if (!n) return NULL;
        x->progs = n;
        x->prog_cap = cap;
    }
    MGLXfbProgram *xp = calloc(1, sizeof(MGLXfbProgram));
    if (!xp) return NULL;
    xp->name = p->name;
    xp->program = p;
    xp->pending_mode = GL_INTERLEAVED_ATTRIBS;
    xp->mode = GL_INTERLEAVED_ATTRIBS;
    x->progs[x->prog_count++] = xp;
    return xp;
}

static void geom_xfb_reset(GLMContext ctx, MGLXfbProgram *xp)
{
    if (xp->vars)
    {
        for (GLuint i = 0; i < xp->count; i++)
        {
            free(xp->vars[i].name);
            free(xp->vars[i].expr);
        }
        free(xp->vars);
    }
    xp->vars = NULL;
    xp->count = 0;
    xp->buffer_mask = 0;
    memset(xp->stride, 0, sizeof(xp->stride));
    xp->record_words = 0;
    free(xp->link_log);
    xp->link_log = NULL;
    xp->link_failed = false;
    xp->linked = true;
}

static GLenum geom_var_type(const MGLGeomVar *v)
{
    /* the type constant names columns x rows (GL_FLOAT_MAT2x3 is a mat2x3, i.e. 2 columns
     * of vec3).  MGLGeomVar.rows is the components per column and .cols the number of
     * columns (see the collector and glsl_type_name), so the tables are indexed
     * [v->cols][v->rows] — indexing them the other way round reported every non-square
     * matrix type transposed (dEQP PIQ transform_feedback_varying.type.*.mat3x4). */
    if (v->cols > 1)
    {
        static const GLenum fq[5][5] = {
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, GL_FLOAT_MAT2, GL_FLOAT_MAT2x3, GL_FLOAT_MAT2x4},
            {0, 0, GL_FLOAT_MAT3x2, GL_FLOAT_MAT3, GL_FLOAT_MAT3x4},
            {0, 0, GL_FLOAT_MAT4x2, GL_FLOAT_MAT4x3, GL_FLOAT_MAT4},
        };
        static const GLenum dq[5][5] = {
            {0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, GL_DOUBLE_MAT2, GL_DOUBLE_MAT2x3, GL_DOUBLE_MAT2x4},
            {0, 0, GL_DOUBLE_MAT3x2, GL_DOUBLE_MAT3, GL_DOUBLE_MAT3x4},
            {0, 0, GL_DOUBLE_MAT4x2, GL_DOUBLE_MAT4x3, GL_DOUBLE_MAT4},
        };
        if (v->rows >= 1 && v->rows <= 4 && v->cols >= 1 && v->cols <= 4)
            return (v->base == GL_DOUBLE) ? dq[v->cols][v->rows] : fq[v->cols][v->rows];
        return v->base == GL_INT ? GL_INT : GL_FLOAT;
    }
    switch (v->base)
    {
        case GL_FLOAT:         return v->rows == 1 ? GL_FLOAT : GL_FLOAT_VEC2 + (v->rows - 2);
        case GL_INT:           return v->rows == 1 ? GL_INT : GL_INT_VEC2 + (v->rows - 2);
        case GL_UNSIGNED_INT:  return v->rows == 1 ? GL_UNSIGNED_INT : GL_UNSIGNED_INT_VEC2 + (v->rows - 2);
        case GL_DOUBLE:        return v->rows == 1 ? GL_DOUBLE : GL_DOUBLE_VEC2 + (v->rows - 2);
        default:               return GL_FLOAT;
    }
}

/* resolve the transform feedback varyings of a program with a geometry stage against the
 * geometry stage's outputs (called from the xfb domain's post-link hook) */
/* resolve the transform feedback varyings of one program against the outputs of its LAST
 * pre-rasterization stage: the geometry stage's out table, or (without a geometry shader)
 * the tessellation evaluation stage's SPIR-V.  The GL layout rules (gl_NextBuffer /
 * gl_SkipComponents, interleaved vs separate) are applied to the caller's pending list. */
/* record an xfb layout error as a link failure with an info-log message */
static void geom_xfb_fail(GLMContext ctx, MGLXfbProgram *xp, const char *msg)
{
    char *log = NULL;

    (void)ctx;
    mgl_append_info_log(&log, "ERROR: transform feedback: %s\n", msg);
    free(xp->link_log);
    xp->link_log = log;
    xp->link_failed = true;
}

static void geom_xfb_resolve(GLMContext ctx, MGLXfbProgram *xp,
                             const MGLGeomVar *vars, int nvars,
                             GLuint pos_word, GLuint psize_word,
                             const char *stage, bool from_shader)
{
    GLuint dbuffer[64], doffset[64];
    GLsizei dn = 0;
    GLuint dbuf_stride[MGL_XFB_MAX_BUFFERS];
    bool dbuf_dbl[MGL_XFB_MAX_BUFFERS];
    bool dbuf_seen[MGL_XFB_MAX_BUFFERS];
    bool dbuf_have_stride[MGL_XFB_MAX_BUFFERS];
    GLuint buffer = 0, offset = 0, n = 0, xn = 0;
    MGLXfbVarying *v;
    char dnames[64][192];
    char xnames[128][192];
    GLuint xsrc[128];                 /* SEPARATE: the pending name each expansion came from */
    GLuint sepo[128];                 /* SEPARATE: running byte offset per pending name */
    bool separate = xp->mode == GL_SEPARATE_ATTRIBS;

    memset(xsrc, 0, sizeof xsrc);
    memset(sepo, 0, sizeof sepo);

    geom_xfb_reset(ctx, xp);
    /* ARB_enhanced_layouts (GL 4.6 11.1.2.1 + issue 18): when the recording stage
     * declares any xfb_* layout qualifiers, the glTransformFeedbackVaryings list is
     * IGNORED — the captured set is exactly the outputs with an xfb_offset.  A
     * variable with only xfb_buffer/xfb_stride (no offset) is NOT captured. */
    if (from_shader) xp->mode = GL_INTERLEAVED_ATTRIBS;
    else xp->mode = xp->pending_mode;
    if (from_shader || !xp->pending_count)
    {
        /* shader-declared capture (xfb_buffer / xfb_offset on outputs, including interface
         * block members): GL 4.6 11.1.2.1 — only outputs that declare an xfb_offset are
         * captured (a bare xfb_buffer / xfb_stride selects the buffer and its stride but
         * captures nothing); an explicit xfb_stride sizes the buffer's per-vertex record,
         * otherwise it is derived from the captures (aligned to 8 when the buffer holds
         * doubles, ARB_enhanced_layouts) */
        memset(dbuf_stride, 0xff, sizeof dbuf_stride);
        memset(dbuf_dbl, 0, sizeof dbuf_dbl);
        memset(dbuf_seen, 0, sizeof dbuf_seen);
        memset(dbuf_have_stride, 0, sizeof dbuf_have_stride);
        for (int k = 0; k < nvars && dn < 64; k++)
        {
            if (vars[k].xfb_buffer < 0) continue;
            GLuint b = (GLuint)vars[k].xfb_buffer;
            if (b < MGL_XFB_MAX_BUFFERS)
            {
                dbuf_seen[b] = true;
                if (vars[k].xfb_stride != 0xffffffffu && vars[k].xfb_stride < dbuf_stride[b])
                    dbuf_stride[b] = vars[k].xfb_stride;
                if (vars[k].base == GL_DOUBLE) dbuf_dbl[b] = true;
            }
            if (vars[k].xfb_offset == 0xffffffffu) continue;   /* no xfb_offset: not captured */
            if (geom_var_dotted(&vars[k]))
            {
                if (vars[k].block_array)
                    snprintf(dnames[dn], sizeof(dnames[dn]), "%s[%u].%s",
                             vars[k].name, vars[k].index, vars[k].member);
                else
                    snprintf(dnames[dn], sizeof(dnames[dn]), "%s.%s", vars[k].name, vars[k].member);
            }
            else
                snprintf(dnames[dn], sizeof(dnames[dn]), "%s", vars[k].name);
            dbuffer[dn] = b;
            doffset[dn] = vars[k].xfb_offset;
            dn++;
        }
        /* declared buffers with an explicit stride publish it even when nothing is
         * captured into them (the stride query must not report the derived 0) */
        for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
            if (dbuf_seen[b] && dbuf_stride[b] != 0xffffffffu)
            { xp->stride[b] = dbuf_stride[b]; dbuf_have_stride[b] = true; }
        if (!dn) return;             /* nothing captured for this program */
        /* capture in ascending (buffer, offset) order */
        for (GLsizei a = 1; a < dn; a++)
            for (GLsizei b = a; b > 0 && (dbuffer[b - 1] > dbuffer[b] ||
                 (dbuffer[b - 1] == dbuffer[b] && doffset[b - 1] > doffset[b])); b--)
            {
                char tn[192];
                GLuint tb = dbuffer[b], to = doffset[b];
                memcpy(tn, dnames[b], sizeof(tn));
                memcpy(dnames[b], dnames[b - 1], sizeof(dnames[b]));
                memcpy(dnames[b - 1], tn, sizeof(dnames[b - 1]));
                dbuffer[b] = dbuffer[b - 1]; doffset[b] = doffset[b - 1];
                dbuffer[b - 1] = tb; doffset[b - 1] = to;
            }
        n = (GLuint)dn;
        xp->from_shader = true;   /* compat r96c: offsets/strides come from layout(xfb_*) qualifiers —
                                   * the draw-time pack must not re-derive a tight interleaved layout */
    }
    else
    {
        /* API-specified varyings: an interface block instance name ("b") selects every
         * member of the block (GL 4.6 13.3.1), so expand the names before resolving */
        n = (GLuint)xp->pending_count;
        for (GLuint q = 0; q < n && xn < 128; q++)
        {
            const char *nm = xp->pending[q];
            int first = -1, members = 0;

            if (!strcmp(nm, "gl_NextBuffer") || !strncmp(nm, "gl_SkipComponents", 17))
            {
                xsrc[xn] = q;
                snprintf(xnames[xn++], sizeof(xnames[0]), "%s", nm);
                continue;
            }
            for (int k = 0; k < nvars; k++)
                if (vars[k].member[0] && !strcmp(vars[k].name, nm))
                { if (first < 0) first = k; members++; }
            if (members > 0)
            {
                for (int k = 0; k < nvars && xn < 128; k++)
                    if (vars[k].member[0] && !strcmp(vars[k].name, nm))
                    {
                        xsrc[xn] = q;
                        /* r102: an ARRAY-OF-BLOCKS instance ("out B{..} b[2];") has one
                         * collected entry per (element, member) — name each element's
                         * member so the resolver below picks the element's own entry
                         * (offset advanced by the std430 block size) */
                        if (vars[k].block_array)
                            snprintf(xnames[xn++], sizeof(xnames[0]), "%s[%u].%s",
                                     vars[k].name, vars[k].index, vars[k].member);
                        else
                            snprintf(xnames[xn++], sizeof(xnames[0]), "%s.%s",
                                     vars[k].name, vars[k].member);
                    }
            }
            else
            {
                xsrc[xn] = q;
                snprintf(xnames[xn++], sizeof(xnames[0]), "%s", nm);
            }
        }
        n = xn;
    }

    xp->vars = calloc(n, sizeof(MGLXfbVarying));
    if (!xp->vars) return;
    xp->count = n;
    if (getenv("MGL_TESS_DUMP_CAP"))
        for (int k = 0; k < nvars; k++)
            fprintf(stderr, "tessxfb-var: name='%s' block='%s' member='%s' elems=%u words=%u off=%u\n",
                    vars[k].name, vars[k].block, vars[k].member, vars[k].elems, vars[k].words, vars[k].offset);

    for (GLuint i = 0; i < n; i++)
    {
        /* from_shader: the API list was ignored, the names/buffers come from the
         * shader-declared table even when a pending API list exists */
        bool use_shader = from_shader || !xp->pending_count;
        const char *name = use_shader ? dnames[i] : xnames[i];

        if (use_shader)
        {
            buffer = dbuffer[i];
            offset = doffset[i];
        }
        v = &xp->vars[i];
        v->name = xstrdup(name);
        xp->max_name_length = (GLuint)(strlen(name) + 1 > (size_t)xp->max_name_length ?
                                        strlen(name) + 1 : (size_t)xp->max_name_length);

        if (!strcmp(name, "gl_NextBuffer"))
        {
            v->type = GL_NONE;
            v->size = 0;
            if (separate)
            {   /* GL 4.6 13.3.1: gl_NextBuffer is only defined for interleaved capture */
                geom_xfb_fail(ctx, xp, "gl_NextBuffer used with GL_SEPARATE_ATTRIBS");
                return;
            }
            xp->stride[buffer] = offset;
            buffer++;
            offset = 0;
            if (buffer >= MGL_XFB_MAX_BUFFERS) { xp->link_failed = true; return; }
            continue;
        }
        if (!strncmp(name, "gl_SkipComponents", 17))
        {
            GLuint n = (GLuint)atoi(name + 17);
            v->type = GL_NONE;
            v->size = (GLint)n;
            /* words MUST stay 0: a marker is not a capture.  Every consumer (the GPU
             * scatter's `!vr.words`, the CPU pack's `!var->words`, the VS capture path)
             * treats words == 0 as "skip / next-buffer marker" and skips the var; leaving
             * it at n made the marker overwrite the FIRST captured varying of the buffer
             * (dst_off 0) with record word 0 - zeros, exactly what
             * KHR-GL46.transform_feedback3.multiple_streams saw. */
            v->words = 0;
            if (separate)
            {
                geom_xfb_fail(ctx, xp, "gl_SkipComponents used with GL_SEPARATE_ATTRIBS");
                return;
            }
            offset += n * 4;
            continue;
        }

        /* builtins */
        if (!strcmp(name, "gl_Position"))
        {
            v->type = GL_FLOAT_VEC4;
            v->size = 1;          /* ARRAY_SIZE of a builtin is always 1 (dEQP PIQ
                                  * array_size.*.builtin_gl_position) */
            v->comp_kind = GL_FLOAT;
            v->rows = 4;
            v->columns = 1;
            v->comp_per_elem = 1;
            v->words = 4;
            v->src_word = pos_word;
            if (separate) { v->buffer = xsrc[i]; v->offset = 0; xp->stride[xsrc[i]] = 16; }
            else          { v->buffer = buffer;  v->offset = offset; offset += 16; }
            xp->buffer_mask |= (1u << v->buffer);
            continue;
        }
        if (!strcmp(name, "gl_PointSize"))
        {
            v->type = GL_FLOAT;
            v->size = 1;      /* ARRAY_SIZE of a scalar builtin is 1 */
            v->comp_kind = GL_FLOAT;
            v->rows = 1;
            v->columns = 1;
            v->comp_per_elem = 1;
            v->words = 1;
            v->src_word = psize_word;
            if (separate) { v->buffer = xsrc[i]; v->offset = 0; xp->stride[xsrc[i]] = 4; }
            else          { v->buffer = buffer;  v->offset = offset; offset += 4; }
            xp->buffer_mask |= (1u << v->buffer);
            continue;
        }

        /* a user output of the stage: either a plain varying, or an interface block member
         * written as instance.member ("b.c"); an interface block instance with a single
         * member also names that member */
        /* r102: an ARRAY-OF-BLOCKS instance is named "inst[e].member" — the element index
         * sits on the instance name, before the member.  Normalize to "inst.member" and
         * remember the element; the collected table holds one entry per (element, member). */
        char belem_nrm[192];
        GLuint belem = 0;
        bool bhave_elem = false;
        {
            const char *bd = strchr(name, '.');
            const char *bb = name ? strchr(name, '[') : NULL;
            if (bd && bb && bb < bd)
            {
                char *bxend;
                long be = strtol(bb + 1, &bxend, 10);
                if (be >= 0 && bxend && bxend[0] == ']' && bxend[1] == '.')
                {
                    snprintf(belem_nrm, sizeof(belem_nrm), "%.*s%s",
                             (int)(bb - name), name, bxend + 1);
                    name = belem_nrm;
                    belem = (GLuint)be;
                    bhave_elem = true;
                }
            }
        }
        /* a trailing "[N]" selects ONE array element (GL 4.6 13.3.1: "the name may include
         * an array index"); without it the whole array is captured.  Strip it before the
         * name/member match and remember the element for the offset below. */
        const char *dot = strchr(name, '.');
        char xname[192];
        GLuint xelem = 0;
        bool xhave_elem = false;
        const char *xbrk = strrchr(name, '[');
        if (xbrk && (!dot || xbrk > dot))
        {
            char *xend;
            long e = strtol(xbrk + 1, &xend, 10);

            if (e >= 0 && xend && *xend == ']' && xend[1] == 0)
            {
                snprintf(xname, sizeof xname, "%.*s", (int)(xbrk - name), name);
                xelem = (GLuint)e;
                xhave_elem = true;
            }
        }
        const char *match = xhave_elem ? xname : name;
        if (xhave_elem && dot)
            dot = strchr(match, '.');      /* member part also lost the "[N]" */
        int j = -1;
        /* pass 1: the name must be an output of the stage — a plain varying, an
         * interface-block instance (whole block), or "<block|instance>.<member>".
         * pass 2: GL 4.6 13.3.1 — a member of an output interface block may be named
         * bare ("tc_position" of the TES's OUT_TE block; the tessellation CTS captures
         * exactly this way), so fall back to matching a block member by name. */
        for (int pass = 0; pass < 2 && j < 0; pass++)
        for (int k = 0; k < nvars; k++)
        {
            if (dot)
            {
                if (pass)
                    continue;
                size_t ilen = (size_t)(dot - match);
                /* a member is named "<instance>.<member>" in shader sources but "<block>.<member>"
                 * through the API (glTransformFeedbackVaryings names interface-block members by
                 * their BLOCK name), so accept either prefix */
                if ((!strncmp(vars[k].name, match, ilen) && vars[k].name[ilen] == 0 ||
                     vars[k].block[0] && !strncmp(vars[k].block, match, ilen) && vars[k].block[ilen] == 0) &&
                    !strcmp(vars[k].member, dot + 1))
                { j = k; break; }
            }
            else if (!strcmp(vars[k].name, match))
            { j = k; break; }
            else if (pass && vars[k].block[0] && !strcmp(vars[k].member, match))
            { j = k; break; }
        }
        if (j < 0 || (xhave_elem && xelem >= vars[j].elems))
        {
            char *log = NULL;
            mgl_append_info_log(&log, "ERROR: transform feedback varying '%s' is not an output of the "
                                      "%s\n", name, stage);
            free(xp->link_log);
            xp->link_log = log;
            xp->link_failed = true;
            return;
        }
        /* r102: the matched entry of an array-of-blocks instance is one ELEMENT — select
         * the named element's own entry (xfb_offset already advanced by the element's
         * std430 block size); without an index the pending name expanded to one entry per
         * element above, so this only narrows an explicit "inst[e].member" */
        if (vars[j].block_array)
        {
            if (geom_env(1))
                fprintf(stderr, "geom: xfbcap block-array narrow '%s' belem=%u have=%d\n",
                        name, belem, (int)bhave_elem);
            if (bhave_elem && belem >= vars[j].block_elems)
            {
                char *log = NULL;
                mgl_append_info_log(&log, "ERROR: transform feedback varying '%s' indexes past the "
                                          "end of the array-of-blocks instance '%s'\n", name, vars[j].name);
                free(xp->link_log);
                xp->link_log = log;
                xp->link_failed = true;
                return;
            }
            if (bhave_elem)
            {
                int k;
                for (k = 0; k < nvars; k++)
                    if (vars[k].block_array && vars[k].index == belem &&
                        !strcmp(vars[k].name, vars[j].name) &&
                        !strcmp(vars[k].member, vars[j].member))
                        break;
                if (k < nvars) j = k;
            }
        }

        const MGLGeomVar *g = &vars[j];
        v->type = geom_var_type(g);
        v->comp_kind = g->base == GL_INT ? GL_INT : g->base == GL_UNSIGNED_INT ? GL_UNSIGNED_INT : GL_FLOAT;
        v->rows = g->rows;
        v->columns = g->cols;
        v->comp_per_elem = g->cols * g->rows * base_words(g->base);
        if (xhave_elem && g->elems > 1)
        {
            /* one element of an arrayed output: a single element's words at its own
             * word offset inside the record */
            GLuint per = g->words / g->elems;

            v->size = 1;
            v->words = per;
            v->src_word = g->offset + xelem * per;
        }
        else
        {
            v->size = (GLint)g->elems;
            v->words = g->words;
            v->src_word = g->offset;
        }
        v->stream = g->stream;   /* only stream 0 rasterizes, but GL captures every stream */
        if (separate)
        {
            /* each API-selected varying is its own binding point (GL 4.6 13.3.1); a name
             * that expanded to several block members captures them into that one buffer */
            v->buffer = xsrc[i];
            v->offset = sepo[xsrc[i]];
            sepo[xsrc[i]] += v->words * 4;
            xp->stride[xsrc[i]] = sepo[xsrc[i]];
        }
        else
        {
            v->buffer = buffer;
            v->offset = offset;
            if (xp->pending_count)
                offset += v->words * 4;
            else
            {
                /* declared capture: the stride of a buffer is the extent of its captures,
                 * unless an xfb_stride was declared for it (already set above); a buffer
                 * holding doubles aligns its derived stride to 8 basic machine units */
                GLuint end = offset + v->words * 4;
                if (dbuf_dbl[buffer]) end = (end + 7u) & ~7u;
                if (!dbuf_have_stride[buffer] && end > xp->stride[buffer]) xp->stride[buffer] = end;
            }
        }
        xp->buffer_mask |= (1u << v->buffer);
    }

    if (xp->pending_count && !separate)
        xp->stride[buffer] = offset;

    if (getenv("MGL_GEOM_XFB_TRACE"))
        for (GLuint i = 0; i < xp->count; i++)
            fprintf(stderr, "geom: xvar[%u] '%s' size=%d rows=%d cols=%d words=%u src_word=%u buf=%d off=%u stride=%u\n",
                    i, xp->vars[i].name, xp->vars[i].size, xp->vars[i].rows,
                    xp->vars[i].columns, xp->vars[i].words, xp->vars[i].src_word,
                    xp->vars[i].buffer, xp->vars[i].offset, xp->stride[xp->vars[i].buffer]);
}

void mgl_geom_xfb_link(GLMContext ctx, Program *program, bool from_shader)
{
    MGLGeomProgram *gp = geom_prog_slot(ctx, program, false);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, true);

    if (!gp || !xp) return;
    geom_xfb_resolve(ctx, xp, gp->out_v, gp->n_out, gp->out_pos_off, gp->out_psize_off,
                     "geometry stage", from_shader);
}

/* transform feedback from a tessellation evaluation stage (no geometry shader): the TES's
 * SPIR-V outputs are the recorded stream.  The per-record word offsets of the tess output
 * capture differ from this table, so the pack refreshes src_word from the published member
 * table on every draw. */
void mgl_geom_tess_xfb_link(GLMContext ctx, Program *program, bool from_shader)
{
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, true);
    MGLGeomVar vars[MGL_GEOM_MAX_VARYINGS];
    SpvModule *m = NULL;
    char *err = NULL;
    int n = 0;

    if (!xp || !program || !program->spirv[_TESS_EVALUATION_SHADER].ir) return;
    m = spv_parse(program->spirv[_TESS_EVALUATION_SHADER].ir,
                  program->spirv[_TESS_EVALUATION_SHADER].size, (const char **)&err);
    if (!m)
    {
        char *log = NULL;
        mgl_append_info_log(&log, "ERROR: transform feedback: TES reflection failed (%s)\n",
                            err ? err : "SPIR-V parse failed");
        free(xp->link_log);
        xp->link_log = log;
        xp->link_failed = true;
        return;
    }
    if (!geom_reflect_stage(m, false, vars, &n, NULL))
        n = 0;
    spv_free(m);
    /* gl_Position is the first word of every capture record; the tess output capture does not
     * record gl_PointSize (a TES writing it would be dropped) */
    geom_xfb_resolve(ctx, xp, vars, n, 0, 0, "tessellation evaluation stage", from_shader);
}

/* transform feedback varyings on a SEPARABLE TCS-only program (no vs/tes/gs): the names
 * resolve against the control stage's outputs, which are per-vertex arrays (and per-patch
 * outs).  Such a program can never feed a draw by itself (GL_PATCHES needs a TES), so the
 * link only fills the varying table — no capture is built. */
void mgl_geom_tcs_xfb_link(GLMContext ctx, Program *program, bool from_shader)
{
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, true);
    MGLGeomVar vars[MGL_GEOM_MAX_VARYINGS];
    SpvModule *m = NULL;
    const char *err = NULL;
    int n = 0;

    if (!xp || !program || !program->spirv[_TESS_CONTROL_SHADER].ir) return;
    m = spv_parse(program->spirv[_TESS_CONTROL_SHADER].ir,
                  program->spirv[_TESS_CONTROL_SHADER].size, &err);
    if (!m)
    {
        char *log = NULL;
        mgl_append_info_log(&log, "ERROR: transform feedback: TCS reflection failed (%s)\n",
                            err ? err : "SPIR-V parse failed");
        free(xp->link_log);
        xp->link_log = log;
        xp->link_failed = true;
        return;
    }
    if (!geom_reflect_stage(m, false, vars, &n, NULL))
        n = 0;
    spv_free(m);
    geom_xfb_resolve(ctx, xp, vars, n, 0, 0, "tessellation control stage", from_shader);
}

/* the active transform feedback object (the xfb domain's current-object rule) */
static MGLXfbObject *geom_xfb_object(GLMContext ctx)
{
    struct MGLExtState_xfb *x = ctx->state.ext_xfb;
    GLuint name;

    if (!x) return NULL;
    name = ctx->state.transform_feedback ? ctx->state.transform_feedback->name : 0;
    if (name && name < x->obj_cap && x->objs[name]) return x->objs[name];
    return &x->default_obj;
}


/* ---------------- GPU pack of the geometry output ---------------- */

/* the test harness reaches the tracking state through here */
extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);

void *mgl_geom_ctx(void);

/* trace helper: the buffer-tracking state of a transform feedback buffer at a CPU read */
void mgl_geom_xfb_log_read_name(void *ctxp, unsigned name, const char *what)
{
    GLMContext ctx = (GLMContext)ctxp;
    Buffer *buf = ctx ? (ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf) : NULL;
    struct MGLExtMetal_buffers *b = ctx ? ctx->mtl_funcs.ext_buffers : NULL;
    MGLBufferTrack *t = ctx ? mgl_buffers_track(ctx, buf, false) : NULL;
    uint64_t completed = (b && b->completed_serial) ? b->completed_serial(ctx) : 0;

    fprintf(stderr, "geom: read %-9s name=%u buf=%p ctx=%p write_serial=%llu completed=%llu (%s)\n",
            what, name, (void *)buf, (void *)ctx,
            (unsigned long long)(t ? t->write_serial : 0),
            (unsigned long long)completed,
            (!t || t->write_serial <= completed) ? "no wait" : "MUST wait");
}

/* the Metal side of the pack (MGLRenderer+Geometry.m) */
extern bool geom_xfb_gpu_encode(GLMContext ctx, MGLGeomProgram *gp, MGLXfbObject *o, MGLXfbProgram *xp,
                                BufferBaseTarget *tfb, uint32_t threads, uint32_t max_vertices,
                                uint32_t out_words, uint32_t vertex_cap);
extern bool geom_xfb_tess_encode(GLMContext ctx, MGLXfbObject *o, MGLXfbProgram *xp,
                                 BufferBaseTarget *tfb, const MGLGeomTessXfbSource *src);
extern void geom_xfb_gpu_read(GLMContext ctx, MGLXfbObject *o, uint32_t *dvert, uint32_t *dprim);
extern void geom_xfb_gpu_reset(GLMContext ctx);
extern bool geom_xfb_gpu_pending(void);
extern bool geom_xfb_gpu_valid(GLMContext ctx, GLuint name);
extern bool geom_xfb_gpu_ensure(GLMContext ctx, Buffer *buf);
extern void geom_xfb_gpu_wait(GLMContext ctx);

/* try to encode the GPU pack for one draw.  Returns false when the CPU pack must be used. */
bool mgl_geom_xfb_pack(GLMContext ctx, MGLGeomProgram *gp, Program *program,
                       uint32_t threads, uint32_t max_vertices, uint32_t out_words,
                       uint32_t vertex_cap)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, gp->program, false);
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;
    GLuint b;

    if (!o || !o->active || !xp || !xp->count) return false;
    /* the GPU pack is the default (the counters and write offsets stay on the GPU and are
     * only folded in where the application observes them); MGL_GEOM_CPU_XFB=1 asks for the
     * CPU pack, which advances the object's offsets at draw time */
    if (geom_env(2)) return false;
    if (mgl_geom_xfb_pending(ctx) && !geom_xfb_gpu_valid(ctx, o->name)) mgl_geom_xfb_flush(ctx);

    /* the varying table and the buffer strides/sizes must fit the fixed blocks; a transform
     * feedback buffer without Metal backing yet gets one from the renderer, so a block never
     * mixes the CPU pack (which advances the object's offsets on the CPU) with the GPU pack
     * (whose counter block holds its own) */
    for (b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
        if (xp->buffer_mask & (1u << b))
        {
            Buffer *buf = tfb[b].buf;
            if (!buf) return false;
            if (!buf->data.mtl_data && !geom_xfb_gpu_ensure(ctx, buf)) return false;
        }
    if (xp->count > MGL_GEOM_XFB_MAX_VARS) return false;

    return geom_xfb_gpu_encode(ctx, gp, o, xp, tfb, threads, max_vertices, out_words, vertex_cap);
}

/* fold the pending GPU counters into the transform feedback object.  Called at the points
 * where the application can observe them, so the command buffer has usually completed. */
void mgl_geom_xfb_flush(GLMContext ctx)
{
    extern void geom_xfb_gpu_read_streams(GLMContext ctx, uint32_t dvert[MGL_XFB_MAX_STREAMS],
                                          uint32_t dprim[2 * MGL_XFB_MAX_STREAMS]);
    MGLXfbObject *o = geom_xfb_object(ctx);
    uint32_t dvert, dprim;
    uint32_t sv[MGL_XFB_MAX_STREAMS];
    uint32_t sp[2 * MGL_XFB_MAX_STREAMS];

    if (!mgl_geom_xfb_pending(ctx)) return;

    /* make the pack's results visible: wait for the command buffer it was encoded into */
    geom_xfb_gpu_wait(ctx);

    if (geom_env(1))
    {
        Buffer *b0 = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf;
        fprintf(stderr, "geom:   fold reads TF backing %p\n", b0 ? b0->data.mtl_data : NULL);
    }
    { extern void geom_xfb_dump(GLMContext ctx); geom_xfb_dump(ctx); }
    /* geom_xfb_gpu_read also refreshes o->write_offset[] from the block */
    geom_xfb_gpu_read(ctx, o, &dvert, &dprim);
    /* the per-stream counters of the geometry pack are authoritative when they were used
     * (a geometry stage captures every stream); the tessellation / vertex-stage paths only
     * fill stream 0's slots (their finish kernels mirror the totals in) */
    geom_xfb_gpu_read_streams(ctx, sv, sp);
    if (geom_env(1))
    {
        Buffer *b0 = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[0].buf;
        float *f = b0 ? (float *)(void *)(vm_address_t)b0->data.buffer_data : NULL;
        fprintf(stderr, "geom: flush name=%u total dvert=%u dprim=%u streams v=[%u %u %u %u] w=[%u %u %u %u] n=[%u %u %u %u] tf=%p words=[%f %f %f %f]\n",
                o ? o->name : 0, dvert, dprim,
                sv[0], sv[1], sv[2], sv[3],
                sp[0], sp[1], sp[2], sp[3],
                sp[4], sp[5], sp[6], sp[7],
                (void *)f,
                f ? f[0] : 0.0f, f ? f[1] : 0.0f, f ? f[2] : 0.0f, f ? f[3] : 0.0f);
    }

    for (GLuint s = 0; s < MGL_XFB_MAX_STREAMS; s++)
    {
        if (o)
        {
            o->vertices[s] += sv[s];
            o->prims_written[s] += sp[s];
            o->prims_needed[s] += sp[s + MGL_XFB_MAX_STREAMS];
        }
        if (ctx->state.ext_xfb)
        {
            struct MGLExtState_xfb *x = ctx->state.ext_xfb;
            x->total_written[s] += sp[s];
            x->total_needed[s] += sp[s + MGL_XFB_MAX_STREAMS];
        }
    }

    /* the transform feedback primitive queries are counted by the queries domain */
    {
        typedef void (*geom_note_queries_fn)(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
        static geom_note_queries_fn note;
        static int looked;
        if (!looked)
        {
            note = (geom_note_queries_fn)dlsym(RTLD_DEFAULT, "mgl_queries_note_xfb_primitives");
            looked = 1;
        }
        /* sp[s] = written, sp[s + MAX] = emitted (needed): needed > written is the overflow
         * condition of GL_TRANSFORM_FEEDBACK_OVERFLOW_ARB / _STREAM_OVERFLOW_ARB */
        if (note)
            for (GLuint s = 0; s < MGL_XFB_MAX_STREAMS; s++)
                note(ctx, s, sp[s], sp[s + MGL_XFB_MAX_STREAMS]);
    }

    geom_xfb_gpu_reset(ctx);
}

/* the GPU pack's own view of "paused": true when the bound transform feedback object is
 * active but paused (nothing is captured, the GPU counter block is left untouched) */
bool mgl_geom_xfb_paused(GLMContext ctx)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    return o && o->active && o->paused;
}

bool mgl_geom_xfb_pending(GLMContext ctx)
{
    (void)ctx;
    return geom_xfb_gpu_pending();
}

/* pack the emitted geometry output into the bound transform feedback buffers.  Called after
 * the geometry kernel has been synchronized, from the Metal side of the emulation.  The
 * kernel assigns every thread a deterministic slot range (thread * max_vertices), so the
 * emission order is simply the slot order. */
void mgl_geom_xfb_capture(GLMContext ctx, MGLGeomProgram *gp, const uint32_t *records, size_t words,
                          const uint32_t *ctl, const uint32_t *cnt, size_t cnt_words)
{
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, gp->program, false);
    uint32_t count = ctl ? ctl[0] : 0;
    uint32_t prims = ctl ? ctl[13] : 0;
    uint32_t written = 0;    /* primitives whose last vertex actually landed */
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;

    if (!o || !o->active || !xp || !xp->count || !count) return;
    if (!records || !cnt) return;
    uint32_t close_base = 0;   /* closed stream-0 primitives in the threads before t */
    uint32_t extent[4] = { 0, 0, 0, 0 };

    /* the per-buffer captured extent: a primitive fits when its last vertex's largest
     * varying fits (GL records whole primitives, never a partial one) */
    for (GLuint v = 0; v < xp->count; v++)
        if (xp->vars[v].words)
        {
            uint32_t e = (uint32_t)(xp->vars[v].offset + xp->vars[v].words * 4);
            if (xp->vars[v].buffer < 4 && e > extent[xp->vars[v].buffer]) extent[xp->vars[v].buffer] = e;
        }

    /* vertices per decomposed output primitive (a strip is captured decomposed, with
     * the odd triangles transposed - see the GPU scatter) */
    GLuint per = (gp->out_type == GL_LINE_STRIP) ? 2 :
                 (gp->out_type == GL_TRIANGLE_STRIP) ? 3 : 1;

    for (uint32_t t = 0; t < cnt_words; t++)
    {
        uint32_t n = cnt[t];
        uint32_t base_slot = t * gp->max_vertices;
        uint32_t thread_prims = 0;   /* captured primitives this thread closed */

        if (base_slot + n > (uint32_t)(words / gp->out_words)) break;

        for (uint32_t s = 0; s < n; s++)
        {
            const uint32_t *rec = records + (size_t)(base_slot + s) * gp->out_words;
            GLuint stream = (rec[1] >> 24) & 0xff;

            /* GL captures the rasterization stream; MGL has no glRasterizationStream yet, so
             * it is stream 0 (the default) */
            if (stream != 0) continue;

            /* only the vertex closing a strip captures anything: the geometry kernel
             * flags exactly the non-degenerate closes, and the tag carries the strip's
             * length (bits 0..7) plus the primitives the thread closed before it
             * (bits 9..23).  The strip decomposes into n-per+1 line / triangle
             * primitives, every one captured with all of its vertices. */
            if (!((rec[1] >> 8) & 1u)) continue;
            {
                uint32_t n = rec[1] & 0xffu;
                uint32_t before = (rec[1] >> 9) & 0x7fffu;

                if (n >= per) thread_prims += n - per + 1u;
                if (n < per) continue;          /* defensive: a degenerate close */

                for (uint32_t p = 0; p + per <= n; p++)
                {
                    uint32_t prim = close_base + before + p;
                    bool fits = true;

                    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
                    {
                        if (!(xp->buffer_mask & (1u << b))) continue;
                        Buffer *buf = tfb[b].buf;
                        size_t endb = (size_t)o->write_offset[b]
                                      + (size_t)(prim * per + per - 1u) * xp->stride[b] + extent[b];
                        if (!buf || endb > (size_t)buf->size) { fits = false; break; }
                    }
                    if (!fits) break;           /* full: the primitive is dropped, as GL does */

                    for (uint32_t j = 0; j < per; j++)
                    {
                        /* mirror the GPU scatter: GL records the strip's triangles with
                         * the odd ones transposed (the GL strip-decomposition rule), so a 4-vertex strip is
                         * captured as v0,v1,v2 | v2,v1,v3 (KHR-GL46.transform_feedback.geometry) */
                        uint32_t sj = (per == 3 && (p & 1u) && j < 2) ? (p + 1u - j) : (p + j);
                        const uint32_t *rj = records + (size_t)(base_slot + s - n + 1u + sj) * gp->out_words;

                        for (GLuint v = 0; v < xp->count; v++)
                        {
                            MGLXfbVarying *var = &xp->vars[v];
                            Buffer *buf;
                            uint32_t *dst;
                            size_t pos;

                            if (!var->words) continue;            /* skip / next-buffer markers */
                            if (!(xp->buffer_mask & (1u << var->buffer))) continue;
                            buf = tfb[var->buffer].buf;
                            if (!buf || !buf->data.buffer_data) continue;

                            pos = (size_t)o->write_offset[var->buffer]
                                  + (size_t)(prim * per + j) * xp->stride[var->buffer] + var->offset;
                            dst = (uint32_t *)((uint8_t *)buf->data.buffer_data + pos);
                            for (GLuint w = 0; w < var->words; w++)
                                dst[w] = rj[var->src_word + w];
                        }
                    }

                    /* the whole primitive landed: advance every buffer by its per vertices */
                    for (GLuint b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
                        if (xp->buffer_mask & (1u << b))
                            o->write_offset[b] += (GLsizeiptr)(per * xp->stride[b]);
                    o->vertices[0] += per;
                    written++;
                }
            }
        }

        close_base += thread_prims;
    }

    o->prims_written[0] += written;
    o->prims_needed[0] += prims;
    if (ctx->state.ext_xfb)
    {
        struct MGLExtState_xfb *x = ctx->state.ext_xfb;
        x->total_written[0] += written;
        x->total_needed[0] += prims;
    }

    /* the transform feedback primitive queries are counted by the queries domain */
    {
        typedef void (*geom_note_queries_fn)(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed);
        static geom_note_queries_fn note;
        static int looked;
        if (!looked)
        {
            note = (geom_note_queries_fn)dlsym(RTLD_DEFAULT, "mgl_queries_note_xfb_primitives");
            looked = 1;
        }
        if (note) note(ctx, 0, written, prims);
    }
}

/* ------------------------------------------------------------------ xfb from tessellation
 * Without a geometry shader the tessellation evaluation stage is the LAST
 * pre-rasterization stage, so transform feedback records its outputs.  The tess domain's
 * output capture (the same one the geometry stage consumes) is the recorded stream: the
 * draw runs with the capture enabled and still rasterizes normally, then this pack writes
 * the captured records into the transform feedback buffers. */

/* SPIRV-Cross cannot emit an ARRAY OF MATRICES (or an array of structs of matrices) as a
 * Metal stage output, so it splits such a TES output into one synthesized variable per
 * column ("m<id>_<i>") and every connection to the GL name is lost — the name matches in
 * mgl_geom_tess_xfb_route find nothing and the varying captures zero words
 * (enhanced_layouts xfb_explicit_location_mat*_tessellation_evaluation).
 *
 * The capture record's structure is still deterministic: every USER output contributes one
 * record member per flattened column-vector slot, and the members sit in the record in
 * SPIRV-Cross's interface order, i.e. (location, component) order of the original outputs
 * (gl_Position is hoisted to word 0; builtin clip/cull/point-size members trail the user
 * members at higher offsets).  So map the varying by SLOT ORDINAL: reflect the TES's own
 * SPIR-V (the same geom_reflect_stage the link-time resolver used), count the slots of the
 * outputs that precede this one by location, then take that many consecutive members of
 * the published capture table, filtered to user members and sorted by offset.
 * `vidx` selects one element of an arrayed output (consume just its columns).
 * Returns false when the shape cannot be reconciled (caller keeps the no-match rule). */
static bool geom_tess_xfb_slot_map(GLMContext ctx, Program *program, MGLXfbVarying *v,
                                   uint32_t vindex, const MGLTessCapture *c, const char *nm,
                                   uint32_t vidx, MGLGeomXfbSplit *splits, uint32_t *splits_n,
                                   uint32_t cap){
    MGLGeomVar vars[MGL_GEOM_MAX_VARYINGS];
    SpvModule *m = NULL;
    const MGLTessCaptureMember *user[MGL_TESS_CAPTURE_MAX];
    uint32_t nuser = 0, slot0 = 0, slots;
    int n = 0, j = -1;
    const char *dot;

    if (!program || !program->spirv[_TESS_EVALUATION_SHADER].ir || !c)
        return false;
    m = spv_parse(program->spirv[_TESS_EVALUATION_SHADER].ir,
                  program->spirv[_TESS_EVALUATION_SHADER].size, NULL);
    if (!m)
        return false;
    if (!geom_reflect_stage(m, false, vars, &n, NULL))
        n = 0;
    spv_free(m);
    if (n <= 0)
        return false;

    /* the varying's own output entry: a plain name, "<instance>.<member>" /
     * "<block>.<member>", a bare block member, or an array-of-blocks element already
     * flattened to "<instance>_<element>_<member>" by the caller */
    dot = strchr(nm, '.');
    for (int k = 0; k < n && j < 0; k++)
    {
        if (dot)
        {
            size_t ilen = (size_t)(dot - nm);

            if ((!strncmp(vars[k].name, nm, ilen) && vars[k].name[ilen] == 0 ||
                 vars[k].block[0] && !strncmp(vars[k].block, nm, ilen) && vars[k].block[ilen] == 0) &&
                !strcmp(vars[k].member, dot + 1))
                j = k;
        }
        else
        {
            if (!strcmp(vars[k].name, nm) ||
                (vars[k].block[0] && !strcmp(vars[k].member, nm)))
                j = k;
            if (j < 0 && vars[k].block_array)
            {
                char flat[192];

                snprintf(flat, sizeof flat, "%s_%u_%s", vars[k].name, vars[k].index, vars[k].member);
                if (!strcmp(flat, nm))
                    j = k;
            }
        }
    }
    if (j < 0)
        return false;

    /* the member stream SPIRV-Cross flattens a user output to (in record order):
     * one member per (array element, matrix column) — except a DOUBLE column wider
     * than two components, which becomes double2 + double<N-2> parts in two whole
     * uint4 slots (the second slot's tail is padding for odd N) */
    {
        bool dbl = vars[j].base == GL_DOUBLE;

        slots = vars[j].elems * vars[j].cols * (dbl && vars[j].rows > 2 ? 2 : 1);
    }
    if (vidx)
    {
        if (vidx >= vars[j].elems)
            return false;
        {
            bool dbl = vars[j].base == GL_DOUBLE;
            uint32_t per = vars[j].cols * (dbl && vars[j].rows > 2 ? 2 : 1);

            slot0 = vidx * per;
            slots = per;
        }
    }
    for (int k = 0; k < n; k++)
        if (vars[k].location < vars[j].location)
        {
            bool dbl = vars[k].base == GL_DOUBLE;

            slot0 += vars[k].elems * vars[k].cols * (dbl && vars[k].rows > 2 ? 2 : 1);
        }

    /* the capture table's user members, in record order (gl_Position sits at word 0;
     * gl_ClipDistance_N / gl_CullDistance_N / gl_PointSize are the trailing builtins) */
    for (uint32_t k = 0; k < c->members && nuser < MGL_TESS_CAPTURE_MAX; k++)
    {
        const char *cn = c->member[k].name;

        if (!strncmp(cn, "gl_", 3) || !strcmp(cn, "mgl_rs_PointSize"))
            continue;
        user[nuser++] = &c->member[k];
    }
    for (uint32_t a = 1; a < nuser; a++)          /* offset-ascending = location order */
        for (uint32_t b = a; b > 0 && user[b - 1]->offset > user[b]->offset; b--)
        {
            const MGLTessCaptureMember *t = user[b];

            user[b] = user[b - 1];
            user[b - 1] = t;
        }
    if (nuser < 1 || slot0 >= nuser || slot0 + slots > nuser)
    {
        if (geom_env(1))
            fprintf(stderr, "geom: tessxfb slot map '%s': slot0=%u+%u > %u user members\n",
                    nm, slot0, slots, nuser);
        return false;
    }

    /* validate the span and derive the copy spec: the published words must cover the
     * varying's words; a DOUBLE matrix with 3-component columns pads its second slot
     * (valid 6 words in 8) and copies per-column parts instead of one range */
    {
        uint32_t tw = 0;

        for (uint32_t k = 0; k < slots; k++)
            tw += user[slot0 + k]->words;
        if (tw != v->words)
        {
            bool dbl = vars[j].base == GL_DOUBLE;
            uint32_t percol = dbl && vars[j].rows > 2 ? 2 : 1, ncols = vars[j].elems * vars[j].cols;

            if (dbl && vars[j].rows == 3 && slots == ncols * 2 && tw == ncols * 8 && v->words == ncols * 6)
            {
                /* dvec3 columns: double2 slot (4 words) + double1 slot (2 valid of 4) */
                if (splits && *splits_n < cap)
                {
                    splits[*splits_n].var = vindex;
                    splits[*splits_n].col_w0 = 4;
                    splits[*splits_n].col_w1 = 2;
                    splits[*splits_n].col_sstride = 8;
                    splits[*splits_n].col_dstride = 6;
                    (*splits_n)++;
                }
            }
            else
            {
                if (geom_env(1))
                    fprintf(stderr, "geom: tessxfb slot map '%s': span words %u != %u\n",
                            nm, tw, v->words);
                return false;
            }
        }
    }
    v->src_word = user[slot0]->offset;
    if (geom_env(1))
        fprintf(stderr, "geom: tessxfb slot map '%s' -> user slot %u src_word=%u (%u slots)\n",
                nm, slot0, v->src_word, slots);
    return true;
}

bool mgl_geom_tess_xfb_route(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                             GLint first, GLsizei count, GLsizei instances,
                             GLint basevertex, GLuint baseinstance)
{
    /* the captured stage is the tessellation evaluation program: with a pipeline bound
     * (useProgram(0)) that is the pipeline's TES stage program — the varyings live there,
     * not on any single ctx->state.program */
    Program *program = ctx->state.program;
    if (!program && ctx->state.program_pipeline)
        program = ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER];
    MGLXfbObject *o = geom_xfb_object(ctx);
    MGLXfbProgram *xp = geom_xfb_prog(ctx, program, false);
    BufferBaseTarget *tfb = ctx->state.buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers;
    BufferBaseTarget live[4];           /* base binds re-sized after binding (below) */
    const MGLTessCapture *c;
    MGLGeomTessXfbSource src;
    MGLGeomXfbSplit splits[8];      /* geom r105d: padded dvec3-column copy specs */
    uint32_t splits_n = 0;
    GLuint b;
    bool ok;

    src.splits = splits;
    src.splits_n = 0;

    if (!program || !o || !o->active || !xp || !xp->count)
    {
        if (getenv("MGL_TESS_DUMP_CAP"))
            fprintf(stderr, "tessxfb-skip: program=%p obj=%p active=%d xp=%p count=%d\n",
                    (void *)program, (void *)o, o ? (int)o->active : -1,
                    (void *)xp, xp ? (int)xp->count : -1);
        return false;
    }
    /* A *Base binding refers to the buffer's whole CURRENT store: glBufferData re-defines
     * it, and the context's base table only snapshots the size at bind time.  The CTS
     * re-sizes the transform feedback buffer between draws (bind once, bufferData per
     * draw), so the pack's fill test must see the live size or every draw after a resize
     * loses records beyond the stale bound. */
    for (b = 0; b < 4; b++)
    {
        live[b] = tfb[b];
        if (b < MGL_XFB_MAX_BUFFERS && tfb[b].buf && o->is_base[b])
        {
            GLsizeiptr total = (GLsizeiptr)tfb[b].buf->size;
            if (tfb[b].buf->data.buffer_size &&
                (GLsizeiptr)tfb[b].buf->data.buffer_size < total)
                total = (GLsizeiptr)tfb[b].buf->data.buffer_size;
            live[b].size = (tfb[b].offset < total) ? (total - tfb[b].offset) : 0;
        }
    }
    tfb = live;
    if (geom_env(2)) return false;
    if (mgl_geom_xfb_pending(ctx) && !geom_xfb_gpu_valid(ctx, o->name)) mgl_geom_xfb_flush(ctx);
    if (xp->count > MGL_GEOM_XFB_MAX_VARS) return false;
    for (b = 0; b < MGL_XFB_MAX_BUFFERS; b++)
        if (xp->buffer_mask & (1u << b))
        {
            Buffer *buf = tfb[b].buf;
            if (!buf) return false;
            if (!buf->data.mtl_data && !geom_xfb_gpu_ensure(ctx, buf)) return false;
        }

    /* the tess domain records its output for us and rasterizes the patch itself */
    mgl_tess_set_capture(ctx, program, true);
    ok = mgl_tess_draw_patches(ctx, indexed, type, offset, first, count, instances,
                               basevertex, baseinstance);
    if (geom_env(1))
        fprintf(stderr, "geom: tessxfb route tess_draw ok=%d\n", (int)ok);
    if (!ok) return false;

    c = mgl_tess_get_capture(ctx);
    if (!c || !c->valid || !c->buffer_object || !c->stride)
    {
        if (getenv("MGL_TESS_DUMP_CAP"))
            fprintf(stderr, "tessxfb-nocap: c=%p valid=%d buf=%p stride=%u\n",
                    (const void *)c, c ? (int)c->valid : -1,
                    c ? (void *)c->buffer_object : NULL, c ? c->stride : 0u);
        return true;   /* nothing captured */
    }
    if (geom_env(1))
        for (uint32_t k = 0; k < c->members; k++)
            fprintf(stderr, "geom: tess capture member[%u] '%s' off=%u words=%u\n",
                    k, c->member[k].name, c->member[k].offset, c->member[k].words);

    /* the capture record's word offsets come from the published member table: they follow the
     * tess output capture's layout, not the geometry stage's table the link-time resolution
     * used */
    for (GLsizei i = 0; i < xp->count; i++)
    {
        MGLXfbVarying *v = &xp->vars[i];
        char gname[192];               /* "inst[e].member" flattened to "inst_e_member" */
        const char *nm = v->name ? v->name : "";
        uint32_t w = v->words;
        bool found = false;
        /* an indexed block member ("Vertex.value[3]") captures ELEMENT 3 of the
         * member's array: match on the base name, then advance whole elements */
        char vbase[256];
        uint32_t vidx = 0;
        bool indexed = false;
        bool slot_mapped = false;

        if (nm[0])
        {
            size_t l = strlen(nm);
            if (l > 3 && l < sizeof vbase && nm[l - 1] == ']')
            {
                const char *ob = strrchr(nm, '[');
                if (ob && ob > nm)
                {
                    memcpy(vbase, nm, ob - nm);
                    vbase[ob - nm] = 0;
                    vidx = (uint32_t)strtoul(ob + 1, NULL, 10);
                    nm = vbase;
                    indexed = true;
                }
            }
        }

        if (geom_env(1))
            fprintf(stderr, "geom: tessxfb var[%d] name='%s' words=%u type=%04x buf=%u off=%u\n",
                    i, nm, v->words, v->type, v->buffer, v->offset);

        if (!strcmp(nm, "gl_Position")) { v->src_word = 0; continue; }   /* always first */
        /* r102: an array-of-blocks element is named "inst[e].member" in GL but the
         * capture record flattens it the way SPIRV-Cross does, "inst_e_member" —
         * normalize once so the exact/suffix matches below see the flattened form */
        {
            const char *gd = strchr(nm, '.');
            const char *gb = strchr(nm, '[');
            if (gd && gb && gb < gd)
            {
                char *gxend;
                long ge = strtol(gb + 1, &gxend, 10);
                if (ge >= 0 && gxend && gxend[0] == ']' && gxend[1] == '.')
                {
                    snprintf(gname, sizeof(gname), "%.*s_%ld_%s",
                             (int)(gb - nm), nm, ge, gxend + 2);
                    nm = gname;
                }
            }
        }
        for (uint32_t k = 0; k < c->members; k++)
        {
            const char *cn = c->member[k].name;
            bool hit = !strcmp(cn, nm);
            if (!hit)
            {
                /* the TF varying may name the member through the BLOCK name while the
                 * capture records the MSL record field ("BLOCK_INOUT.value" vs
                 * "user_out_value"): match the member suffix after the final separator */
                const char *cd = strrchr(cn, '.'), *nd = strrchr(nm, '.');
                if (cd && nd && !strcmp(cd, nd))
                    hit = true;
                if (!hit)
                {
                    const char *cu = strrchr(cn, '_'), *nd2 = strrchr(nm, '.');
                    if (cu && nd2 && !strcmp(cu + 1, nd2 + 1))
                        hit = true;
                }
                /* a BARE member name (GL 4.6 13.3.1: interface-block members are named
                 * without their block) against SPIRV-Cross's flattened "<instance>_<member>"
                 * capture field ("tc_position" vs "te_data_tc_position") */
                if (!hit && !strchr(nm, '.') && !strchr(nm, '['))
                {
                    size_t nl = strlen(nm);
                    if (strlen(cn) > nl + 1 && cn[strlen(cn) - nl - 1] == '_' &&
                        !strcmp(cn + strlen(cn) - nl, nm))
                        hit = true;
                }
            }
            if (hit)
            {
                v->src_word = c->member[k].offset;
                if (!w) w = c->member[k].words;
                found = true;
                if (geom_env(1))
                    fprintf(stderr, "geom: tessxfb hit var '%s' -> member '%s' src_word=%u\n",
                            nm, cn, c->member[k].offset);
                break;
            }
        }
        if (!found)
        {
            /* matrices (and flattened array elements) are recorded as split members
             * "<name>_<i>": consume all of them, starting at the lowest offset.
             * The varying may still carry its BLOCK prefix ("Vertex.value") — the
             * flattened fields are "<instance>_<member>_<i>", so match the bare
             * member name between the underscores. */
            const char *bare = strrchr(nm, '.') ? strrchr(nm, '.') + 1 : nm;
            size_t nl = strlen(nm), bl = strlen(bare);
            uint32_t lo = ~0u, tw = 0;

            for (uint32_t k = 0; k < c->members; k++)
            {
                const char *cn = c->member[k].name;
                size_t cl = strlen(cn);
                bool hit = false;

                if (cl > nl + 1 && !strncmp(cn, nm, nl) &&
                    cn[nl] == '_' && cn[nl + 1] >= '0' && cn[nl + 1] <= '9')
                    hit = true;                      /* "<name>_<i>" */
                if (!hit && bl && cl > bl + 2)
                {
                    /* "<instance>_<baremember>_<i>": digits after the last '_',
                     * the bare member right before it */
                    const char *last = strrchr(cn, '_');
                    if (last && last > cn && last[1] >= '0' && last[1] <= '9' &&
                        (size_t)(last - cn) > bl && !strncmp(last - bl, bare, bl) &&
                        last[-bl - 1] == '_')
                        hit = true;
                }
                if (hit)
                {
                    if (c->member[k].offset < lo) lo = c->member[k].offset;
                    tw += c->member[k].words;
                }
            }
            if (tw)
            {
                v->src_word = lo;
                if (!w) w = tw;
                found = true;
            }
        }
        if (!found && v->type != GL_NONE)
            found = slot_mapped = geom_tess_xfb_slot_map(ctx, program, v, (uint32_t)i, c, nm,
                                                         indexed ? vidx : 0, splits, &splits_n, 8);
        if (indexed && found && v->words && !slot_mapped)
        {
            /* element `vidx` of the array member: advance by whole elements
             * (the GL type of "Block.member[i]" is the ELEMENT type) */
            v->src_word += vidx * v->words;
        }
        if (!found && v->type != GL_NONE)
        {
            /* a varying the capture does not record (e.g. gl_PointSize): record nothing for it
             * rather than capturing the wrong words */
            v->words = 0;
            v->src_word = 0;
        }
    }

    /* words may have been filled in from the published capture member table above —
     * the interleaved record layout (per-buffer strides, var offsets) must be
     * recomputed from the final word counts, exactly as the link-time resolver
     * would have done with these numbers.  Without this a double varying whose
     * words only became known now packs into a zero-stride record and every
     * write is dropped ("full: drop" in the encode kernel).
     * NOT for a shader-declared layout (compat r96c): its offsets/strides are
     * explicit qualifiers (xfb_offset/xfb_stride), authoritative over the word
     * counts — e.g. `layout(xfb_offset = 16) out Goku goku;` records members at
     * 16/32/48 with stride 64. */
    if (!xp->from_shader)
    {
        GLuint off[4] = { 0, 0, 0, 0 };

        for (GLuint v = 0; v < xp->count; v++)
        {
            MGLXfbVarying *xv = &xp->vars[v];

            if (xv->type == GL_NONE)
                continue;   /* gl_NextBuffer advances the stream only */
            xv->offset = off[xv->buffer];
            off[xv->buffer] += xv->words * 4;
        }
        for (GLuint b = 0; b < 4; b++)
            xp->stride[b] = off[b];
    }

    memset(&src, 0, sizeof src);
    src.splits = splits;
    src.splits_n = splits_n;      /* r105d: the per-column copy specs from the slot map */
    src.ring = c->buffer_object;
    src.records = c->records_offset;
    src.counts = c->count_offset;
    src.stride = c->stride;
    src.capacity = c->records ? c->records : 1;
    /* GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN counts PRIMITIVES: k captured
     * vertices each.  k comes from the transform feedback primitive mode the
     * app selected (the GL capture rules: POINTS captures one record per vertex from
     * ANY tessellated output — KHR-GL46.tessellation_shader.vertex.vertex_spacing
     * captures POINTS from a triangle domain), not from the TES's domain. */
    {
        GLenum tfm = o->active ? o->primitive_mode : c->prim;
        src.per_prim = (tfm == GL_TRIANGLES || tfm == GL_TRIANGLE_STRIP || tfm == GL_TRIANGLE_FAN ||
                        tfm == GL_TRIANGLES_ADJACENCY) ? 3 :
                       (tfm == GL_LINES || tfm == GL_LINE_STRIP || tfm == GL_LINE_LOOP ||
                        tfm == GL_LINES_ADJACENCY) ? 2 : 1;
    }

    return geom_xfb_tess_encode(ctx, o, xp, tfb, &src);
}

/* SPIRV-Cross emits its array-copy helpers for native C arrays only
 * (`template<T, A> void spvArrayCopyFromConstantToThreadGroup1(threadgroup T (&dst)[A], ...)`)
 * but struct members it lays out as spvUnsafeArray<T,A> — assigning a whole array
 * to a member of a shared variable (GL `g_shared[id].v = vec4[2](...)`,
 * KHR-GL46.compute_shader.shared-max) then calls the helper with an spvUnsafeArray
 * destination, which cannot bind to a C-array reference; the runtime Metal compiler
 * rejects the module (the command-line `metal` accepts it — do not rely on that).
 * This pass appends element-wise overloads for spvUnsafeArray destinations/sources
 * right after the struct's definition.  Returns a malloc'd copy (or NULL). */
char *mgl_msl_add_unsafearray_copies(const char *msl)
{
    /* spvUnsafeArray's size parameter is size_t (unsigned long): the overloads
     * must take size_t A or template deduction fails with "deduced non-type
     * template argument does not have the same type" */
    static const char overloads[] =
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromConstantToThreadGroup1(threadgroup spvUnsafeArray<T, A>& dst, constant T (&src)[A])\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n"
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromStackToThreadGroup1(threadgroup spvUnsafeArray<T, A>& dst, thread const T (&src)[A])\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n"
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromThreadGroupToThreadGroup1(threadgroup spvUnsafeArray<T, A>& dst, threadgroup const spvUnsafeArray<T, A>& src)\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n"
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromConstantToStack1(thread spvUnsafeArray<T, A>& dst, constant T (&src)[A])\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n"
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromStackToStack1(thread spvUnsafeArray<T, A>& dst, thread const spvUnsafeArray<T, A>& src)\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n"
        "template<typename T, size_t A>\n"
        "inline void spvArrayCopyFromThreadGroupToStack1(thread spvUnsafeArray<T, A>& dst, threadgroup const spvUnsafeArray<T, A>& src)\n"
        "{\n    for (size_t i = 0; i < A; ++i) dst[i] = src[i];\n}\n";

    if (!msl || !strstr(msl, "struct spvUnsafeArray") || !strstr(msl, "spvArrayCopyFrom"))
        return msl ? strdup(msl) : NULL;

    /* insert after the `};` that closes struct spvUnsafeArray */
    const char *s = strstr(msl, "struct spvUnsafeArray");
    const char *e = s ? strstr(s, "};") : NULL;
    if (!e)
        return strdup(msl);
    size_t at = (size_t)(e + 2 - msl);

    char *out = malloc(strlen(msl) + sizeof overloads);
    if (!out)
        return NULL;
    memcpy(out, msl, at);
    memcpy(out + at, overloads, sizeof overloads);
    strcpy(out + at + sizeof overloads - 1, msl + at);
    return out;
}

/* ==========================================================================
 * cull-distance whole-primitive emulation (tess domain)
 * ==========================================================================
 *
 * Metal raster stages have no whole-primitive rejection: SPIRV-Cross exports
 * gl_CullDistance as [[user(cullN)]] attributes, which reach only the fragment stage.
 * GL 4.6 13.5.2 requires a primitive to be dropped ENTIRELY when the cull distance of
 * all its vertices is negative for the same cull plane — a fragment-stage discard
 * cannot express that (an interpolated negative distance also occurs inside primitives
 * that must stay visible, and sibling vertices are invisible to a fragment).
 *
 * A draw whose last pre-rasterization stage writes gl_CullDistance therefore runs
 * through the geometry-emulation path with a SYNTHETIC pass-through "geometry" stage:
 * the vertex capture variant records the real vertex stage's outputs (position, cull
 * and clip distances, varyings), a generated kernel evaluates the cull rule over each
 * input primitive's vertices and emits the primitive only when it survives, and the
 * existing pass-through raster function draws the survivor list.  Everything stays on
 * the GPU (no readback, no CPU stall): the same capture -> kernel -> raster pipeline a
 * user geometry shader takes, with the cull check standing in for the GS body.
 *
 * The kernel is generated per input-primitive family (points / lines / triangles /
 * line-adjacency / triangle-adjacency) because the input layout qualifier must match
 * the GL draw mode; the family is a draw-time property, so the route (re)builds the
 * kernel lazily whenever it changes.
 * ========================================================================== */

/* the input-primitive family a GL draw mode maps to (-1: cannot run here).  NOTE:
 * GL_POINTS == 0 is a valid family, so 0 must not double as the failure value */
static int geom_cull_family(GLenum mode)
{
    switch (mode)
    {
        case GL_POINTS:                   return GL_POINTS;
        case GL_LINES:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:                return GL_LINES;
        case GL_LINES_ADJACENCY:
        case GL_LINE_STRIP_ADJACENCY:     return GL_LINES_ADJACENCY;
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:            return GL_TRIANGLES;
        case GL_TRIANGLES_ADJACENCY:
        case GL_TRIANGLE_STRIP_ADJACENCY: return GL_TRIANGLES_ADJACENCY;
        default:                          return -1;
    }
}

/* the family this draw's cull kernel must cover: for a tessellation program the TES
 * layout decides (the draw mode is GL_PATCHES and the tessellated output is what the
 * synthetic kernel receives); otherwise the GL draw mode's family */
static int geom_cull_want_family(GLMContext ctx, MGLGeomProgram *gp, GLenum mode)
{
    if (gp->program && gp->program->shader_slots[_TESS_EVALUATION_SHADER])
    {
        int fam = geom_cull_tess_family(ctx, gp->program);
        if (fam >= 0) return fam;    /* GL_POINTS is 0: any valid family wins */
    }
    return geom_cull_family(mode);
}

/* the vertices that form the primitive (adjacency vertices never do — GL 4.6 13.5
 * keeps them out of clipping and culling) and the order they are emitted in: the
 * primitive's winding is preserved (a cyclic rotation) while the GL provoking vertex
 * ends up emitted last, which is the geometry stage's provoking rule. */
static void geom_cull_emit_list(GLenum in_type, bool provoking_first,
                                int emit[8], int *n_emit, int check[8], int *n_check)
{
    switch (in_type)
    {
        case GL_POINTS:
            emit[0] = check[0] = 0; *n_emit = *n_check = 1;
            return;
        case GL_LINES:
            /* line winding is irrelevant to rasterization */
            if (provoking_first) { emit[0] = 1; emit[1] = 0; }
            else                 { emit[0] = 0; emit[1] = 1; }
            *n_emit = 2;
            check[0] = 0; check[1] = 1; *n_check = 2;
            return;
        case GL_LINES_ADJACENCY:
            emit[0] = check[0] = 1; emit[1] = check[1] = 2;
            *n_emit = *n_check = 2;
            return;
        case GL_TRIANGLES_ADJACENCY:
            check[0] = 0; check[1] = 2; check[2] = 4; *n_check = 3;
            if (provoking_first) { emit[0] = 2; emit[1] = 4; emit[2] = 0; }
            else                 { emit[0] = 0; emit[1] = 2; emit[2] = 4; }
            *n_emit = 3;
            return;
        default:    /* GL_TRIANGLES: strips/fans arrive pre-assembled as windows */
            check[0] = 0; check[1] = 1; check[2] = 2; *n_check = 3;
            if (provoking_first) { emit[0] = 1; emit[1] = 2; emit[2] = 0; }
            else                 { emit[0] = 0; emit[1] = 1; emit[2] = 2; }
            *n_emit = 3;
            return;
    }
}

/* SPIR-V word scan: which gl_CullDistance entries does the stage actually STORE?
 *
 * glslang exports gl_CullDistance as the BuiltIn CullDistance(4) member of the
 * gl_PerVertex struct.  Entries the shader never writes are undefined in GL and must
 * not cull (Mesa leaves them zero), so the generated kernel only evaluates the stored
 * entries: a constant-index store names its plane, any store through an index the scan
 * cannot resolve turns every plane on.  Returns the plane bitmask and sets *writes when
 * a store reaches the array at all. */
static uint32_t geom_cull_store_mask(const uint32_t *ir, size_t nwords, uint32_t builtin, bool *writes)
{
    uint32_t max_id = 0;
    uint32_t *const_val = NULL, *ptr_pointee = NULL, *arr_elem = NULL;
    uint32_t *chain_idx_val = NULL, *cull_member = NULL, *chain_base = NULL;
    uint8_t  *var_storage = NULL, *chain_idx_const = NULL, *chain_cull = NULL;
    uint8_t  *chain_kind = NULL;      /* 1: base is the builtin variable, 2: base is another cull chain */
    uint32_t cull_vars[8];
    int n_cull_vars = 0;
    uint32_t member = 0xffffffff, mask = 0;
    bool all = false, any = false;
    size_t i = 5;

    *writes = false;
    if (!ir || nwords < 5) return 0;
    /* the header's Bound bounds every result id */
    max_id = ir[3];
    if (max_id < 2 || max_id > (1u << 22)) return 0;

    const_val      = malloc((max_id + 1) * sizeof *const_val);
    ptr_pointee    = calloc(max_id + 1, sizeof *ptr_pointee);
    arr_elem       = calloc(max_id + 1, sizeof *arr_elem);
    var_storage    = calloc(max_id + 1, 1);
    chain_base     = calloc(max_id + 1, sizeof *chain_base);
    chain_idx_const = calloc(max_id + 1, 1);
    chain_idx_val  = calloc(max_id + 1, sizeof *chain_idx_val);
    chain_cull     = calloc(max_id + 1, 1);
    chain_kind     = calloc(max_id + 1, 1);
    cull_member    = calloc(max_id + 1, sizeof *cull_member);
    if (!const_val || !ptr_pointee || !arr_elem || !var_storage || !chain_base ||
        !chain_idx_const || !chain_idx_val || !chain_cull || !cull_member || !chain_kind)
        goto done;
    for (i = 0; i <= max_id; i++) const_val[i] = 0xffffffff;

    /* pass 1: pointer/array types, int constants, output variables, the gl_PerVertex
     * member carrying the CullDistance builtin */
    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
        if (!wc) break;

        switch (op)
        {
            case SpvOpTypePointer:
                if (i + 4 <= nwords) ptr_pointee[ir[i+1]] = ir[i+3];
                break;
            case SpvOpTypeArray:
                if (i + 3 <= nwords) arr_elem[ir[i+1]] = ir[i+2];
                break;
            case SpvOpConstant:
                if (i + 4 <= nwords && ir[i+2] <= max_id) const_val[ir[i+2]] = ir[i+3];
                break;
            case SpvOpVariable:
                /* [op, result type, result, storage class] */
                if (i + 4 <= nwords && ir[i+2] <= max_id) var_storage[ir[i+2]] = (uint8_t)ir[i+3];
                break;
            case SpvOpMemberDecorate:
                /* [op, struct type, member index, decoration, literal...] */
                if (i + 5 <= nwords && ir[i+1] <= max_id &&
                    ir[i+3] == (uint32_t)SpvDecorationBuiltIn &&
                    ir[i+4] == builtin)
                    cull_member[ir[i+1]] = ir[i+2] + 1;   /* member index + 1 (0 = none) */
                break;
            default:
                break;
        }
        i += wc;
    }

    /* the output variables that address that struct (possibly through arrays) */
    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
        if (!wc) break;
        if (op == SpvOpVariable && i + 4 <= nwords)
        {
            uint32_t rid = ir[i+2], ty = ir[i+1], sc = ir[i+3];   /* [op, type, result, storage] */
            if (sc == SpvStorageClassOutput && ty <= max_id)
            {
                uint32_t pt = ptr_pointee[ty];
                int guard = 0;
                while (pt && pt <= max_id && arr_elem[pt] && ++guard < 4) pt = arr_elem[pt];
                if (pt && pt <= max_id && cull_member[pt] && n_cull_vars < 8)
                {
                    cull_vars[n_cull_vars++] = rid;
                    member = cull_member[pt] - 1;
                }
            }
        }
        i += wc;
    }
    if (!n_cull_vars) goto done;

    /* pass 2: the access chains into the array and the stores through them.  SPIR-V is
     * dominated, so a chain's base is always defined earlier in the stream. */
    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
        if (!wc) break;

        if (op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain)
        {
            uint32_t rid = ir[i+2], base = ir[i+3];   /* [op, type, result, base, idx...] */
            int nidx = (int)wc - 4;
            uint8_t into = 0;

            for (int k = 0; k < n_cull_vars; k++) if (base == cull_vars[k]) into = 1;
            if (!into && base <= max_id && chain_cull[base]) into = 2;    /* chain of a chain */
            if (rid <= max_id) chain_base[rid] = base;
            if (into && rid <= max_id)
            {
                chain_kind[rid] = into;
                for (int k = 0; k < nidx && k < 2; k++)
                {
                    uint32_t ix = ir[i+4+k];
                    if (ix <= max_id && const_val[ix] != 0xffffffff)
                    {
                        chain_idx_const[rid] |= (uint8_t)(1u << k);
                        if (k == 0) chain_idx_val[rid] = (chain_idx_val[rid] & 0xffff0000u) | (const_val[ix] & 0xffffu);
                        else        chain_idx_val[rid] = (chain_idx_val[rid] & 0x0000ffffu) | (const_val[ix] << 16);
                    }
                }
                /* gl_Position and the varyings live in the same gl_PerVertex variable:
                 * only a chain that selects the cull-distance member addresses the array */
                if (into == 1)
                    chain_cull[rid] = ((chain_idx_const[rid] & 1u) &&
                                       (chain_idx_val[rid] & 0xffffu) == member) ? 1 : 0;
                else
                    chain_cull[rid] = 1;
            }
        }
        else if (op == SpvOpStore && i + 3 <= nwords)
        {
            uint32_t ptr = ir[i+1];
            if (ptr > max_id) { i += wc; continue; }

            bool direct = false;
            for (int k = 0; k < n_cull_vars; k++) if (ptr == cull_vars[k]) direct = true;

            if (direct) all = any = true;             /* whole-array store */
            else if (chain_cull[ptr])
            {
                uint32_t plane;
                bool plane_known = false;

                any = true;
                if (member == 0xffffffff)
                {
                    /* a bare BuiltIn variable: the first index is the plane */
                    if (chain_idx_const[ptr] & 1u)
                    {
                        plane = chain_idx_val[ptr] & 0xffffu;
                        plane_known = true;
                    }
                }
                else if (chain_kind[ptr] == 1)
                {
                    /* [member, plane]: this chain selects the gl_PerVertex member first */
                    if ((chain_idx_const[ptr] & 1u) && (chain_idx_val[ptr] & 0xffffu) == member &&
                        (chain_idx_const[ptr] & 2u))
                    {
                        plane = (chain_idx_val[ptr] >> 16) & 0xffffu;
                        plane_known = true;
                    }
                }
                else
                {
                    /* chain of a chain: the inner chain selected the member (its only
                     * index), this chain's first index is the plane */
                    uint32_t inner = chain_base[ptr];
                    if (inner <= max_id && chain_cull[inner] && chain_kind[inner] == 1 &&
                        (chain_idx_const[inner] & 1u) && !(chain_idx_const[inner] & 2u) &&
                        (chain_idx_val[inner] & 0xffffu) == member && (chain_idx_const[ptr] & 1u))
                    {
                        plane = chain_idx_val[ptr] & 0xffffu;
                        plane_known = true;
                    }
                }
                if (plane_known)
                {
                    if (plane < MGL_GEOM_MAX_CLIP) mask |= 1u << plane;
                    else all = true;                 /* out of range: be conservative */
                }
                else all = true;                     /* not statically known */
            }
        }
        i += wc;
    }

done:
    free(const_val); free(ptr_pointee); free(arr_elem); free(var_storage);
    free(chain_base); free(chain_idx_const); free(chain_idx_val); free(chain_cull); free(cull_member);
    free(chain_kind);
    if (all) mask = (1u << MGL_GEOM_MAX_CLIP) - 1u;
    *writes = any;
    return mask;
}

/* the program the synthetic cull slot keys on (mirrors mgl_geom_current's resolution):
 * the last pre-rasterization stage is the vertex stage and no geometry stage may be
 * present (a geometry stage owns the path and evaluates cull itself) */
static Program *geom_cull_program(GLMContext ctx)
{
    if (!ctx->state.ext_geom) return NULL;
    if (ctx->state.program)
    {
        Program *p = ctx->state.program;
        /* a geometry stage is the geometry domain's route; a tessellation evaluation
         * stage routes the cull through the tess capture (geom_route_tess) */
        if (p->shader_slots[_GEOMETRY_SHADER]) return NULL;
        return p;
    }
    if (ctx->state.program_pipeline)
    {
        /* a pipeline resolves through the program owning the stage (mgl_geom_current):
         * the vertex stage program here; a pipeline geometry stage wins instead, and a
         * pipeline tessellation stage keys on the TES program's slot (tess path) */
        MGLPipelineObj *pl = MGL_PIPELINE_OBJ(STATE(program_pipeline));
        if (pl->base.stage_programs[_GEOMETRY_SHADER]) return NULL;
        if (pl->base.stage_programs[_TESS_EVALUATION_SHADER])
            return pl->base.stage_programs[_TESS_EVALUATION_SHADER];
        return pl->base.stage_programs[_VERTEX_SHADER];
    }
    return NULL;
}

/* fast presence check: the MSL substring hit is only paid until the link-time slot
 * exists (mgl_geom_post_link creates it) */
static bool geom_cull_writes_msl(Program *p)
{
    const char *msl = p ? p->spirv[_VERTEX_SHADER].msl_str : NULL;
    return msl && strstr(msl, "[[user(cull") != NULL;
}

/* same test for a tessellation evaluation stage: its output is what must be culled */
static bool geom_cull_writes_msl_tes(Program *p)
{
    const char *msl = p ? p->spirv[_TESS_EVALUATION_SHADER].msl_str : NULL;
    return msl && strstr(msl, "[[user(cull") != NULL;
}

/* the primitive family a tessellation program's output takes (TES layout): points for
 * point_mode, lines for isolines, triangles otherwise. -1 when the program has no TES. */
static int geom_cull_tess_family(GLMContext ctx, Program *program)
{
    MGLTessProgram *tp;

    if (!program->shader_slots[_TESS_EVALUATION_SHADER]) return -1;
    tp = mgl_tess_program(ctx, program);
    if (!tp) return -1;
    if (tp->point_mode) return GL_POINTS;
    if (tp->domain == MGL_TESS_DOMAIN_ISOLINES) return GL_LINES;
    return GL_TRIANGLES;
}

bool mgl_geom_cull_active(GLMContext ctx)
{
    Program *p = geom_cull_program(ctx);
    MGLGeomProgram *gp;

    if (!p) return false;
    /* link-time flag first (mgl_geom_post_link, glmark2 perf r29): the geometry
     * program-table lookup is only paid by programs that actually write
     * gl_CullDistance — rescanning the VS MSL here cost glmark2 ideas 42% */
    if (!p->writes_cull_distance) return false;
    gp = geom_prog_slot(ctx, p, false);
    if (gp) return gp->synthetic;
    return true;
}

/* the synthetic pass-through "geometry" stage source: cull evaluation + emit */
static char *geom_gen_cull_gs_glsl(MGLGeomProgram *gp)
{
    SB b;
    const char *in_kw, *out_kw;
    int emit[8], check[8], n_emit, n_check;

    memset(&b, 0, sizeof b);
    switch (gp->in_type)
    {
        case GL_POINTS:              in_kw = "points";              out_kw = "points, max_vertices = 1"; break;
        case GL_LINES:               in_kw = "lines";               out_kw = "line_strip, max_vertices = 2"; break;
        case GL_LINES_ADJACENCY:     in_kw = "lines_adjacency";     out_kw = "line_strip, max_vertices = 2"; break;
        case GL_TRIANGLES_ADJACENCY: in_kw = "triangles_adjacency"; out_kw = "triangle_strip, max_vertices = 3"; break;
        default:                     in_kw = "triangles";           out_kw = "triangle_strip, max_vertices = 3"; break;
    }

    sb_put(&b, "#version 460 core\n");
    sb_put(&b, "layout(%s) in;\n", in_kw);
    sb_put(&b, "layout(%s) out;\n\n", out_kw);

    /* only the members the capture records exist on gl_in */
    sb_put(&b, "in gl_PerVertex\n{\n    vec4 gl_Position;\n");
    if (gp->in_has_psize) sb_put(&b, "    float gl_PointSize;\n");
    if (gp->in_has_clip && gp->in_clip_size) sb_put(&b, "    float gl_ClipDistance[%u];\n", gp->in_clip_size);
    if (gp->in_clip_size) sb_put(&b, "    float gl_CullDistance[%u];\n", gp->in_clip_size);
    sb_put(&b, "} gl_in[];\n\n");

    for (int i = 0; i < gp->n_in; i++)
    {
        MGLGeomVar *v = &gp->in_v[i];
        const char *ty = glsl_type_name(v->base, v->rows, v->cols);
        char arr[24];
        if (!ty) { free(b.s); return NULL; }
        if (v->elems > 1) snprintf(arr, sizeof arr, "[%u]", v->elems); else arr[0] = 0;
        sb_put(&b, "%s%s%sin %s mgl_ci%d%s[];\n", v->flat ? "flat " : "",
               v->noperspective ? "noperspective " : "", v->centroid ? "centroid " : "",
               ty, i, arr);
    }
    sb_put(&b, "\n");
    for (int i = 0; i < gp->n_out; i++)
    {
        MGLGeomVar *v = &gp->out_v[i];
        const char *ty = glsl_type_name(v->base, v->rows, v->cols);
        char arr[24];
        if (!ty) { free(b.s); return NULL; }
        if (v->elems > 1) snprintf(arr, sizeof arr, "[%u]", v->elems); else arr[0] = 0;
        sb_put(&b, "%s%s%sout %s mgl_co%d%s;\n", v->flat ? "flat " : "",
               v->noperspective ? "noperspective " : "", v->centroid ? "centroid " : "",
               ty, i, arr);
    }

    sb_put(&b, "\nvoid main()\n{\n");
    sb_put(&b, "    bool mgl_culled = false;\n");
    geom_cull_emit_list(gp->in_type, false, emit, &n_emit, check, &n_check);
    for (int i = 0; i < (int)gp->in_clip_size; i++)
    {
        if (!(gp->cull_planes & (1u << i))) continue;
        sb_put(&b, "    {\n        bool mgl_neg = true;\n");
        for (int v = 0; v < n_check; v++)
            sb_put(&b, "        if (gl_in[%d].gl_CullDistance[%d] >= 0.0) mgl_neg = false;\n",
                   check[v], i);
        sb_put(&b, "        if (mgl_neg) mgl_culled = true;\n    }\n");
    }
    sb_put(&b, "    if (mgl_culled) return;\n");
    geom_cull_emit_list(gp->in_type, gp->provoking_first_emit, emit, &n_emit, check, &n_check);
    for (int e_i = 0; e_i < n_emit; e_i++)
    {
        int e = emit[e_i];
        sb_put(&b, "    gl_Position = gl_in[%d].gl_Position;\n", e);
        if (gp->in_has_psize) sb_put(&b, "    gl_PointSize = gl_in[%d].gl_PointSize;\n", e);
        if (gp->in_has_clip && gp->in_clip_size)
            for (int i = 0; i < (int)gp->in_clip_size; i++)
                sb_put(&b, "    gl_ClipDistance[%d] = gl_in[%d].gl_ClipDistance[%d];\n", i, e, i);
        if (gp->in_clip_size)
            for (int i = 0; i < (int)gp->in_clip_size; i++)
                sb_put(&b, "    gl_CullDistance[%d] = gl_in[%d].gl_CullDistance[%d];\n", i, e, i);
        for (int i = 0; i < gp->n_out; i++)
            sb_put(&b, "    mgl_co%d = mgl_ci%d[%d];\n", i, i, e);
        sb_put(&b, "    EmitVertex();\n");
    }
    sb_put(&b, "    EndPrimitive();\n}\n");
    return b.s;
}

/* module-side gl_ClipDistance / gl_CullDistance declaration: the word width the
 * vertex stage's gl_PerVertex block carries for `builtin` (the member's array
 * length, or a bare BuiltIn variable's array length; 0 when not declared).
 * GLSL-linked programs answer this from the source text (geom_declared_clip_
 * words), but a program linked from SPIR-V binaries has no source — its clip
 * distances must survive the synthetic cull pass-through too, or every plane is
 * dropped and nothing clips (KHR-GL46.gl_spirv.spirv_validation_builtin_
 * variable_decorations, clipNegative on the SPIR-V iteration). */
static GLuint geom_builtin_decl_size(const uint32_t *ir, size_t nwords, uint32_t builtin)
{
    uint32_t max_id, member_of = 0, member_idx = 0, bare_var = 0;
    uint32_t *const_val = NULL, *ptr_pointee = NULL, *arr_len = NULL, *struct_mbr = NULL;
    GLuint words = 0;
    size_t i;

    if (!ir || nwords < 5) return 0;
    max_id = ir[3];
    if (max_id < 2 || max_id > (1u << 22)) return 0;

    const_val   = malloc((max_id + 1) * sizeof *const_val);
    ptr_pointee = calloc(max_id + 1, sizeof *ptr_pointee);
    arr_len     = calloc(max_id + 1, sizeof *arr_len);
    struct_mbr  = calloc(max_id + 1, sizeof *struct_mbr);
    if (!const_val || !ptr_pointee || !arr_len || !struct_mbr) goto done;
    for (i = 0; i <= max_id; i++) const_val[i] = 0xffffffff;

    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
        if (!wc) break;

        switch (op)
        {
            case SpvOpConstant:      /* [op, type, result, value] */
                if (i + 4 <= nwords && ir[i+2] <= max_id) const_val[ir[i+2]] = ir[i+3];
                break;
            case SpvOpTypePointer:
                if (i + 4 <= nwords && ir[i+1] <= max_id) ptr_pointee[ir[i+1]] = ir[i+3];
                break;
            case SpvOpTypeArray:     /* [op, result, elem, len] */
                if (i + 4 <= nwords && ir[i+1] <= max_id)
                    arr_len[ir[i+1]] = (ir[i+3] <= max_id && const_val[ir[i+3]] != 0xffffffff)
                                       ? const_val[ir[i+3]] : 0xffffffff;
                break;
            case SpvOpTypeStruct:    /* [op, result, member types...] */
                if (ir[i+1] <= max_id) struct_mbr[ir[i+1]] = (uint32_t)(i + 2); /* stream index of member 0's type */
                break;
            case SpvOpMemberDecorate:/* [op, struct, member, decoration, literal...] */
                if (i + 5 <= nwords && ir[i+1] <= max_id &&
                    ir[i+3] == (uint32_t)SpvDecorationBuiltIn && ir[i+4] == builtin)
                {
                    member_of = ir[i+1];
                    member_idx = ir[i+2];
                }
                break;
            case SpvOpDecorate:      /* a bare BuiltIn variable (no gl_PerVertex block) */
                if (i + 4 <= nwords && ir[i+3] == (uint32_t)SpvDecorationBuiltIn &&
                    ir[i+4] == builtin)
                    bare_var = ir[i+1];
                break;
            default:
                break;
        }
        i += wc;
    }

    {
        uint32_t ty = 0;
        if (member_of && member_of <= max_id && struct_mbr[member_of])
        {
            /* member m's type is the (m+1)-th operand after the struct id */
            size_t base = struct_mbr[member_of];
            if (base + member_idx < nwords) ty = ir[base + member_idx];
        }
        else if (bare_var && bare_var <= max_id)
        {
            /* OpVariable [op, ptr type, result, storage] -> pointee */
            /* the variable's own instruction is needed for its pointer type; scan again */
            i = 5;
            while (i < nwords)
            {
                uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
                if (!wc) break;
                if (op == SpvOpVariable && i + 4 <= nwords && ir[i+2] == bare_var)
                {
                    if (ir[i+1] <= max_id) ty = ptr_pointee[ir[i+1]];
                    break;
                }
                i += wc;
            }
        }
        if (ty && ty <= max_id && arr_len[ty] && arr_len[ty] != 0xffffffff)
            words = arr_len[ty];
    }

done:
    free(const_val);
    free(ptr_pointee);
    free(arr_len);
    free(struct_mbr);
    return words;
}

/* The declared OUTPUT size of a gl_PerVertex builtin (gl_ClipDistance /
 * gl_CullDistance) of a stage module, preferring the OUTPUT interface: an
 * evaluation-stage module declares gl_in[] as well, and the input and output
 * sizes may differ.  Resolves the member of the gl_PerVertex block that an
 * Output variable points at (unwrapping a TCS-style array of blocks), or a bare
 * Output builtin variable.  Returns 0 when the stage does not declare it. */
static GLuint geom_out_decl_size(const uint32_t *ir, size_t nwords, uint32_t builtin)
{
    uint32_t max_id;
    uint32_t *const_val = NULL, *ptr_pointee = NULL, *arr_elem = NULL, *arr_len = NULL;
    uint32_t *struct_mbr = NULL, *member_of = NULL, *member_idx = NULL, *bare_var = NULL;
    GLuint words = 0;
    size_t i;

    if (!ir || nwords < 5) return 0;
    max_id = ir[3];
    if (max_id < 2 || max_id > (1u << 22)) return 0;
    const_val   = malloc((max_id + 1) * sizeof *const_val);
    ptr_pointee = calloc(max_id + 1, sizeof *ptr_pointee);
    arr_elem    = calloc(max_id + 1, sizeof *arr_elem);
    arr_len     = calloc(max_id + 1, sizeof *arr_len);
    struct_mbr  = calloc(max_id + 1, sizeof *struct_mbr);
    member_of   = calloc(max_id + 1, sizeof *member_of);   /* per builtin-decorated member */
    member_idx  = calloc(max_id + 1, sizeof *member_idx);
    bare_var    = calloc(max_id + 1, sizeof *bare_var);
    if (!const_val || !ptr_pointee || !arr_elem || !arr_len || !struct_mbr ||
        !member_of || !member_idx || !bare_var)
        goto done;
    for (i = 0; i <= max_id; i++) const_val[i] = 0xffffffff;

    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;

        if (!wc) break;
        switch (op)
        {
            case SpvOpConstant:
                if (i + 4 <= nwords && ir[i+2] <= max_id) const_val[ir[i+2]] = ir[i+3];
                break;
            case SpvOpTypePointer:
                if (i + 4 <= nwords && ir[i+1] <= max_id) ptr_pointee[ir[i+1]] = ir[i+3];
                break;
            case SpvOpTypeArray:
                if (i + 4 <= nwords && ir[i+1] <= max_id)
                {
                    arr_elem[ir[i+1]] = ir[i+2];
                    arr_len[ir[i+1]] = (ir[i+3] <= max_id && const_val[ir[i+3]] != 0xffffffff)
                                       ? const_val[ir[i+3]] : 0xffffffff;
                }
                break;
            case SpvOpTypeStruct:
                if (ir[i+1] <= max_id) struct_mbr[ir[i+1]] = (uint32_t)(i + 2);
                break;
            case SpvOpMemberDecorate:
                if (i + 5 <= nwords && ir[i+1] <= max_id &&
                    ir[i+3] == (uint32_t)SpvDecorationBuiltIn && ir[i+4] == builtin)
                {
                    member_of[ir[i+1]] = 1;
                    member_idx[ir[i+1]] = ir[i+2];
                }
                break;
            case SpvOpDecorate:
                if (i + 4 <= nwords && ir[i+1] <= max_id &&
                    ir[i+3] == (uint32_t)SpvDecorationBuiltIn && ir[i+4] == builtin)
                    bare_var[ir[i+1]] = 1;
                break;
            default:
                break;
        }
        i += wc;
    }

    /* an Output variable whose pointee (through one array level) is a decorated
     * gl_PerVertex block, or a bare decorated Output builtin variable */
    i = 5;
    while (i < nwords)
    {
        uint32_t inst = ir[i], wc = inst >> 16, op = inst & 0xffff;
        uint32_t ty = 0;

        if (!wc) break;
        if (op == SpvOpVariable && i + 4 <= nwords && ir[i+3] == (uint32_t)SpvStorageClassOutput)
        {
            uint32_t vid = ir[i+2];      /* the variable's result id */
            uint32_t pty;                /* the pointee before any array unwrap */

            if (ir[i+1] <= max_id) pty = ty = ptr_pointee[ir[i+1]];
            else pty = ty = 0;
            if (ty && ty <= max_id && arr_len[ty] && arr_elem[ty] <= max_id)
                ty = arr_elem[ty];                       /* array of blocks -> block */
            if (ty && ty <= max_id && member_of[ty] && struct_mbr[ty])
            {
                size_t base = struct_mbr[ty];

                if (base + member_idx[ty] < nwords)
                {
                    uint32_t mt = ir[base + member_idx[ty]];

                    if (mt <= max_id && arr_len[mt] && arr_len[mt] != 0xffffffff)
                        words = arr_len[mt];
                }
            }
            else if (vid <= max_id && bare_var[vid] && pty && pty <= max_id &&
                     arr_len[pty] && arr_len[pty] != 0xffffffff)
                words = arr_len[pty];
            if (words)
                break;
        }
        i += wc;
    }

done:
    free(const_val);
    free(ptr_pointee);
    free(arr_elem);
    free(arr_len);
    free(struct_mbr);
    free(member_of);
    free(member_idx);
    free(bare_var);
    return words;
}

/* Validate a tess-fed geometry stage's gl_in[].gl_ClipDistance / gl_CullDistance reads
 * against the evaluation stage's declared outputs: the TES-output capture carries the
 * builtins as flattened per-element members (mgl_tess_capture_msl), so the reads are
 * legal exactly when the evaluation stage declares the outputs.  Returns a static
 * message to fail the link with, or NULL when the capture can supply the words. */
static const char *geom_tess_clip_check(const Program *program, const MGLGeomProgram *gp)
{
    const Spirv *tes;
    GLuint tc, tcu;

    if (!gp->in_has_clip && !gp->in_has_cull)
        return NULL;
    if (!program || !program->shader_slots[_TESS_EVALUATION_SHADER])
        return NULL;
    tes = &program->spirv[_TESS_EVALUATION_SHADER];
    tc  = geom_out_decl_size(tes->ir, tes->size, (uint32_t)SpvBuiltInClipDistance);
    tcu = geom_out_decl_size(tes->ir, tes->size, (uint32_t)SpvBuiltInCullDistance);
    if (gp->in_has_clip && !tc)
        return "geometry: the geometry stage reads gl_in[].gl_ClipDistance but the "
               "evaluation stage does not declare the gl_ClipDistance output";
    if (gp->in_has_cull && !tcu)
        return "geometry: the geometry stage reads gl_in[].gl_CullDistance but the "
               "evaluation stage does not declare the gl_CullDistance output";
    return NULL;
}

/* fill the emulation state for a cull program: everything mgl_geom_compile_stage derives
 * from a real geometry stage's interface comes from the vertex stage here (the synthetic
 * stage is a pass-through, so its inputs and outputs are the vertex stage's outputs) */
static bool geom_cull_reflect(GLMContext ctx, MGLGeomProgram *gp, Program *program)
{
    MGLGeomVar vs_out[MGL_GEOM_MAX_VARYINGS];
    int n_vs_out = 0;
    SpvModule *vm = NULL;
    const char *e2 = NULL, *src;
    char *vs_src = NULL;
    GLuint n;
    bool ok = false;
    /* the interface the synthetic kernel consumes: the TES output for a tessellation
     * program (the tess capture supplies the records), else the vertex stage output */
    int stage = program->shader_slots[_TESS_EVALUATION_SHADER] ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;

    if (!program->shader_slots[stage] || !program->spirv[stage].ir)
        return false;

    vm = spv_parse(program->spirv[stage].ir, program->spirv[stage].size, &e2);
    if (!vm) goto done;
    if (!geom_reflect_stage(vm, false, vs_out, &n_vs_out, NULL)) goto done;
    if (n_vs_out > MGL_GEOM_MAX_VARYINGS) n_vs_out = MGL_GEOM_MAX_VARYINGS;

    gp->cull_planes = geom_cull_store_mask(program->spirv[stage].ir,
                                           program->spirv[stage].size,
                                           (uint32_t)SpvBuiltInCullDistance, &gp->writes_cull);
    if (!gp->writes_cull) goto done;    /* nothing is stored: nothing can cull */

    /* which clip planes the vertex stage really writes: the rest are undefined in GL
     * and must not clip (the raster pass-through neutralizes them) */
    {
        bool w = false;
        gp->clip_planes = geom_cull_store_mask(program->spirv[stage].ir,
                                               program->spirv[stage].size,
                                               (uint32_t)SpvBuiltInClipDistance, &w);
    }

    if (stage == _TESS_EVALUATION_SHADER)
    {
        /* the TES's GLSL (the declared gl_ClipDistance/gl_CullDistance sizes drive the
         * record layout); geom_vs_source resolves only the vertex stage */
        src = geom_stage_source(ctx, MGL_PROGRAM_OBJ(program), stage);
    }
    else
    {
        src = geom_vs_source(ctx, MGL_PROGRAM_OBJ(program));
    }
    vs_src = xstrdup(src ? src : "");

    /* clip/cull word count: the capture wrapper must not read past either declaration.
     * A declaration carries its own size; without one the largest STORED entry decides
     * (the source's highest gl_CullDistance[7] literal is an INDEX, not a size — the
     * store scan above knows the real extent).  Presence on the module side comes from
     * the STORE scan as well — glslang always emits the whole gl_PerVertex block, so a
     * bare OpMemberDecorate BuiltIn says nothing; a SPIR-V-binary program (no source)
     * that stores gl_ClipDistance must still clip, and one that does not store it must
     * not grow a clip region (the record offsets would shift and cull would break:
     * test_cull lines_dynamic_index_writes). */
    {
        GLuint mod_clip = geom_builtin_decl_size(program->spirv[_VERTEX_SHADER].ir,
                                                 program->spirv[_VERTEX_SHADER].size,
                                                 (uint32_t)SpvBuiltInClipDistance);
        bool writes_clip = false;
        geom_cull_store_mask(program->spirv[_VERTEX_SHADER].ir,
                             program->spirv[_VERTEX_SHADER].size,
                             (uint32_t)SpvBuiltInClipDistance, &writes_clip);
        gp->in_has_clip = strstr(vs_src, "gl_ClipDistance") != NULL || writes_clip;
        n = geom_declared_clip_words(vs_src, "gl_ClipDistance");
        if (gp->in_has_clip && mod_clip > n) n = mod_clip;
        gp->in_clip_size = gp->in_has_clip ? (n ? n : MGL_GEOM_MAX_CLIP) : 0;
    }
    n = geom_declared_clip_words(vs_src, "gl_CullDistance");
    if (n && n > gp->in_clip_size) gp->in_clip_size = n;   /* shared record width: the wider array wins */
    if (!gp->in_clip_size) gp->in_clip_size = n;
    /* the record must carry every entry a store reaches */
    {
        uint32_t both = gp->cull_planes | gp->clip_planes;
        GLuint hb = 0;
        for (GLuint b = 0; b < MGL_GEOM_MAX_CLIP; b++)
            if (both & (1u << b)) hb = b + 1;
        if (hb > gp->in_clip_size) gp->in_clip_size = hb;
        if (gp->in_clip_size > MGL_GEOM_MAX_CLIP) gp->in_clip_size = MGL_GEOM_MAX_CLIP;
    }
    /* only planes the declaration carries can cull */
    gp->cull_planes &= (gp->in_clip_size >= 32) ? 0xffffffffu : ((1u << gp->in_clip_size) - 1u);
    if (!gp->cull_planes) goto done;
    gp->in_has_cull = true;

    gp->in_has_pos = true;
    /* the source scan covers GLSL-linked programs; a program linked from SPIR-V
     * binaries has no source, so also detect the store in the module (same rule as
     * the source text: only a stage that really writes gl_PointSize re-exports it —
     * glslang leaves unwritten builtin members out of the emitted gl_PerVertex). */
    {
        bool writes_psize = false;
        geom_cull_store_mask(program->spirv[_VERTEX_SHADER].ir,
                             program->spirv[_VERTEX_SHADER].size,
                             (uint32_t)SpvBuiltInPointSize, &writes_psize);
        gp->in_has_psize = strstr(vs_src, "gl_PointSize") != NULL || writes_psize;
    }
    gp->out_has_pos = true;
    gp->out_has_psize = gp->in_has_psize;
    gp->out_has_clip = gp->in_has_clip;
    gp->out_has_cull = true;
    gp->out_has_layer = false;
    gp->out_has_vp = false;
    gp->out_has_prim = false;
    gp->out_clip_size = gp->in_clip_size;
    gp->streams = 1;
    gp->invocations = 1;

    gp->n_in = gp->n_out = n_vs_out;
    for (int i = 0; i < n_vs_out; i++)
    {
        MGLGeomVar *iv = &gp->in_v[i], *ov = &gp->out_v[i];

        snprintf(iv->name, sizeof iv->name, "mgl_ci%d", i);
        snprintf(ov->name, sizeof ov->name, "mgl_co%d", i);
        snprintf(iv->expr, sizeof iv->expr, "%s%s%s", vs_out[i].name,
                 vs_out[i].member[0] ? "." : "", vs_out[i].member);
        snprintf(iv->block, sizeof iv->block, "%s", vs_out[i].block);
        snprintf(ov->block, sizeof ov->block, "%s", vs_out[i].block);
        iv->member[0] = ov->member[0] = 0;
        iv->location = ov->location = vs_out[i].location;
        iv->base = ov->base = vs_out[i].base;
        iv->rows = ov->rows = vs_out[i].rows;
        iv->cols = ov->cols = vs_out[i].cols;
        iv->elems = ov->elems = vs_out[i].elems;
        iv->words = ov->words = vs_out[i].words;
        iv->flat = ov->flat = vs_out[i].flat;
        iv->noperspective = ov->noperspective = vs_out[i].noperspective;
        iv->centroid = ov->centroid = vs_out[i].centroid;
        iv->sample = ov->sample = vs_out[i].sample;
        iv->stream = ov->stream = 0;
        iv->xfb_buffer = ov->xfb_buffer = -1;
        iv->xfb_offset = ov->xfb_offset = 0;
    }
    geom_layout(gp);
    ok = true;

done:
    spv_free(vm);
    free(vs_src);
    return ok;
}

/* (re)build the synthetic emulation for a GL draw mode.  Returns false when the program
 * turns out not to cull (the caller falls back to the normal draw path). */
static bool geom_cull_prepare(GLMContext ctx, MGLGeomProgram *gp, Program *program, GLenum mode)
{
    int fam = geom_cull_want_family(ctx, gp, mode);

    if (getenv("MGL_CULL_TRACE"))
        fprintf(stderr, "cull prepare: mode=%u fam=%d gp=%p linked=%d family=%d\n",
                mode, fam, (void *)gp, gp ? gp->linked : -1, gp ? gp->cull_family : -1);
    if (fam < 0) return false;
    if (gp->linked && gp->synthetic && gp->cull_family == fam) return true;
    if (gp->synthetic && gp->cull_family == -1) return false;   /* build failed once */

    geom_free_prog(ctx, gp);
    memset(gp, 0, sizeof *gp);
    gp->program = program;
    gp->name = program->name;
    gp->synthetic = true;
    gp->cull_family = fam;
    /* a failed build must not poison the program's own link status (GEOM_FAIL) */
    gp->deferred = true;
    gp->in_type = fam;
    gp->out_type = (fam == GL_POINTS) ? GL_POINTS :
                   ((fam == GL_LINES || fam == GL_LINES_ADJACENCY) ? GL_LINE_STRIP : GL_TRIANGLE_STRIP);
    gp->in_verts = (fam == GL_POINTS) ? 1 : (fam == GL_LINES) ? 2 :
                   (fam == GL_LINES_ADJACENCY) ? 4 : (fam == GL_TRIANGLES_ADJACENCY) ? 6 : 3;
    gp->max_vertices = (fam == GL_POINTS) ? 1 : (fam == GL_LINES_ADJACENCY) ? 2 :
                       (fam == GL_TRIANGLES_ADJACENCY) ? 3 : gp->in_verts;
    gp->provoking_first_emit = (ctx->state.var.provoking_vertex == GL_FIRST_VERTEX_CONVENTION);
    gp->kernel_entry = xstrdup("mgl_geom_gs");
    gp->capture_entry = xstrdup("main0");
    gp->draw_entry = xstrdup("main0");

    if (!geom_cull_reflect(ctx, gp, program))
    {
        geom_free_prog(ctx, gp);        /* release anything reflect left behind */
        memset(gp, 0, sizeof *gp);
        gp->program = program;
        gp->name = program->name;
        gp->synthetic = true;           /* stays inert (linked false): draws pass through */
        gp->cull_family = -1;
        return false;
    }
    geom_build(ctx, gp, program);
    if (!gp->linked)
    {
        geom_free_prog(ctx, gp);
        memset(gp, 0, sizeof *gp);
        gp->program = program;
        gp->name = program->name;
        gp->synthetic = true;
        gp->cull_family = -1;           /* do not retry: draws pass through unculled */
        return false;
    }
    return true;
}

/* ensure the synthetic kernel exists for this draw's primitive family; false leaves the
 * caller to draw through the normal (unculled) path */
bool mgl_geom_cull_ready(GLMContext ctx, GLenum mode)
{
    MGLGeomProgram *gp = mgl_geom_current(ctx);

    if (!gp || !gp->synthetic) return false;
    if (gp->linked && gp->cull_family == geom_cull_want_family(ctx, gp, mode)) return true;
    return geom_cull_prepare(ctx, gp, gp->program, mode);
}
