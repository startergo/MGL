/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * get.c
 * MGL
 *
 * glGet* family. Lookup order for every pname:
 *   1. per-domain hooks (mgl_ext_get / mgl_ext_get_indexed) — a domain that
 *      keeps its own state for a pname always wins, so values stay live;
 *   2. live MGL state (bindings, render state, pixel store, caps, ...);
 *   3. implementation limits (macOS defaults from getMacOSDefaults, with
 *      GL 4.6 values for limits a 4.1 host context cannot report).
 * Unknown pnames raise GL_INVALID_ENUM; out-of-range indices GL_INVALID_VALUE.
 * Conversions follow the GL 4.6 spec (2.2.2 / 22.1): normalized/colour state
 * maps [-1,1] linearly onto the integer range, other floats round to nearest,
 * anything non-zero is GL_TRUE.
 */

#include <math.h>
#include <string.h>
#include <stdint.h>
#include <mach/mach_time.h>

#include "glm_context.h"
#include "mgl.h"
#include "mgl_ext.h"
#include "mgl_ext_debug.h"
#include "mgl_ext_program.h"

#define MAX_GET_VALUES 32

typedef struct {
    int     count;
    bool    normalized;     /* colour / normalized value: linear map for integer queries */
    double  v[MAX_GET_VALUES];
} GetValues;

#pragma mark - conversions

static void store_value(GLuint type, void *data, int i, double v, bool normalized)
{
    switch (type)
    {
        case MGL_GET_BOOL:
            ((GLboolean *)data)[i] = (v != 0.0) ? GL_TRUE : GL_FALSE;
            break;

        case MGL_GET_INT: {
            GLint r;
            if (normalized) {
                double t = v * 2147483647.0;
                r = t >= 2147483647.0 ? INT32_MAX : t <= -2147483648.0 ? INT32_MIN : (GLint)llround(t);
            } else if (v > 2147483647.0 && v <= 4294967295.0) {
                r = (GLint)(GLuint)v;           /* unsigned masks (e.g. 0xFFFFFFFF) */
            } else if (v > 2147483647.0) {
                r = INT32_MAX;
            } else if (v < -2147483648.0) {
                r = INT32_MIN;
            } else {
                r = (GLint)lround(v);
            }
            ((GLint *)data)[i] = r;
            break;
        }

        case MGL_GET_INT64: {
            GLint64 r;
            if (normalized) {
                double t = v * 9223372036854775807.0;
                r = t >= 9223372036854775807.0 ? INT64_MAX : t <= -9223372036854775808.0 ? INT64_MIN : (GLint64)llround(t);
            } else if (v >= 9223372036854775807.0) {
                r = INT64_MAX;
            } else if (v <= -9223372036854775808.0) {
                r = INT64_MIN;
            } else {
                r = (GLint64)llround(v);
            }
            ((GLint64 *)data)[i] = r;
            break;
        }

        case MGL_GET_FLOAT:
            ((GLfloat *)data)[i] = (GLfloat)v;
            break;

        case MGL_GET_DOUBLE:
            ((GLdouble *)data)[i] = v;
            break;
    }
}

static void store_values(GLuint type, void *data, const GetValues *gv)
{
    for (int i = 0; i < gv->count; i++)
        store_value(type, data, i, gv->v[i], gv->normalized);
}

#define PUT1(a)          do { gv->count = 1; gv->v[0] = (a); return true; } while (0)
#define PUT2(a, b)       do { gv->count = 2; gv->v[0] = (a); gv->v[1] = (b); return true; } while (0)
#define PUT4(a, b, c, d) do { gv->count = 4; gv->v[0] = (a); gv->v[1] = (b); gv->v[2] = (c); gv->v[3] = (d); return true; } while (0)
#define NAME(p)          ((p) ? (double)(p)->name : 0.0)
#define LIMIT(var, def)  ((double)((var) ? (var) : (def)))
#define MIN_D(a, b)      ((double)(a) < (double)(b) ? (double)(a) : (double)(b))

#pragma mark - limits

/* GL 4.6 values used when the 4.1 host context has no answer */
static GLuint max_draw_buffers(GLMContext ctx)
{
    GLuint n = STATE_VAR(max_draw_buffers) ? STATE_VAR(max_draw_buffers) : MAX_COLOR_ATTACHMENTS;
    return n > MAX_COLOR_ATTACHMENTS ? MAX_COLOR_ATTACHMENTS : n;
}

static GLuint max_uniform_buffer_bindings(GLMContext ctx)
{
    GLuint n = STATE_VAR(max_uniform_buffer_bindings) ? STATE_VAR(max_uniform_buffer_bindings) : 14;
    return n > MAX_BINDABLE_BUFFERS ? MAX_BINDABLE_BUFFERS : n;
}

#define MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS  8
#define MGL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS  8
#define MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS      4
#define MGL_MAX_IMAGE_UNITS                     8
#define MGL_MAX_VERTEX_ATTRIB_BINDINGS          MAX_BINDABLE_BUFFERS
#define MGL_MAX_VIEWPORTS                       16

/* the 18 compressed formats required by GL 4.6 (RGTC, BPTC, ETC2/EAC), all mapped to Metal formats */
static const GLenum compressed_formats[] = {
    GL_COMPRESSED_RED_RGTC1, GL_COMPRESSED_SIGNED_RED_RGTC1,
    GL_COMPRESSED_RG_RGTC2, GL_COMPRESSED_SIGNED_RG_RGTC2,
    GL_COMPRESSED_RGBA_BPTC_UNORM, GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
    GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
    GL_COMPRESSED_RGB8_ETC2, GL_COMPRESSED_SRGB8_ETC2,
    GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
    GL_COMPRESSED_RGBA8_ETC2_EAC, GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
    GL_COMPRESSED_R11_EAC, GL_COMPRESSED_SIGNED_R11_EAC,
    GL_COMPRESSED_RG11_EAC, GL_COMPRESSED_SIGNED_RG11_EAC,
};
#define NUM_COMPRESSED_FORMATS ((int)(sizeof(compressed_formats) / sizeof(compressed_formats[0])))

#pragma mark - extensions / strings

/* Extensions whose functionality MGL implements as GL 4.6 core. Deliberately
 * not advertised: fp64 (no doubles in MSL), gl_spirv / get_program_binary
 * (no binary formats exposed), pipeline statistics, sparse, bindless. */
static const char *mgl_extensions[] = {
    "GL_ARB_ES2_compatibility",
    "GL_ARB_ES3_compatibility",
    "GL_ARB_arrays_of_arrays",
    "GL_ARB_base_instance",
    "GL_ARB_blend_func_extended",
    "GL_ARB_buffer_storage",
    "GL_ARB_clear_buffer_object",
    "GL_ARB_clear_texture",
    "GL_ARB_clip_control",
    "GL_ARB_compute_shader",
    "GL_ARB_conservative_depth",
    "GL_ARB_copy_buffer",
    "GL_ARB_copy_image",
    "GL_ARB_depth_buffer_float",
    "GL_ARB_depth_clamp",
    "GL_ARB_direct_state_access",
    "GL_ARB_draw_buffers_blend",
    "GL_ARB_draw_elements_base_vertex",
    "GL_ARB_draw_indirect",
    "GL_ARB_draw_instanced",
    "GL_ARB_explicit_attrib_location",
    "GL_ARB_explicit_uniform_location",
    "GL_ARB_fragment_coord_conventions",
    "GL_ARB_fragment_layer_viewport",
    "GL_ARB_framebuffer_no_attachments",
    "GL_ARB_framebuffer_object",
    "GL_ARB_framebuffer_sRGB",
    "GL_ARB_get_texture_sub_image",
    "GL_ARB_gl_spirv",                 /* shader binary format: SPIR-V (get.c) */
    "GL_ARB_gpu_shader_fp64",          /* software fp64 (fp64.c), GLSL doubles + fp64 lib */
    "GL_ARB_half_float_pixel",
    "GL_ARB_half_float_vertex",
    "GL_ARB_instanced_arrays",
    "GL_ARB_internalformat_query",
    "GL_ARB_internalformat_query2",
    "GL_ARB_invalidate_subdata",
    "GL_ARB_map_buffer_alignment",
    "GL_ARB_map_buffer_range",
    "GL_ARB_multi_bind",
    "GL_ARB_multi_draw_indirect",
    "GL_ARB_spirv_extensions",         /* glGetStringi(GL_SPIR_V_EXTENSIONS) */
    "GL_ARB_occlusion_query",
    "GL_ARB_occlusion_query2",
    "GL_ARB_pixel_buffer_object",
    "GL_ARB_polygon_offset_clamp",
    "GL_ARB_program_interface_query",
    "GL_ARB_provoking_vertex",
    "GL_ARB_robustness",
    "GL_ARB_sampler_objects",
    "GL_ARB_seamless_cube_map",
    "GL_ARB_separate_shader_objects",
    "GL_ARB_shader_atomic_counters",
    "GL_ARB_shader_bit_encoding",
    "GL_ARB_shader_image_load_store",
    "GL_ARB_shader_image_size",
    "GL_ARB_shader_storage_buffer_object",
    "GL_ARB_shader_texture_lod",
    "GL_ARB_shading_language_420pack",
    "GL_ARB_shading_language_packing",
    "GL_ARB_stencil_texturing",
    "GL_ARB_sync",
    "GL_ARB_texture_buffer_object",
    "GL_ARB_texture_buffer_object_rgb32",
    "GL_ARB_texture_buffer_range",
    "GL_ARB_texture_compression_bptc",
    "GL_ARB_texture_compression_rgtc",
    "GL_ARB_texture_cube_map_array",
    "GL_ARB_texture_filter_anisotropic",
    "GL_ARB_texture_float",
    "GL_ARB_texture_gather",
    "GL_ARB_texture_mirror_clamp_to_edge",
    "GL_ARB_texture_multisample",
    "GL_ARB_texture_query_levels",
    "GL_ARB_texture_rectangle",
    "GL_ARB_texture_rg",
    "GL_ARB_texture_rgb10_a2ui",
    "GL_ARB_texture_stencil8",
    "GL_ARB_texture_storage",
    "GL_ARB_texture_storage_multisample",
    "GL_ARB_texture_swizzle",
    "GL_ARB_texture_view",
    "GL_ARB_timer_query",
    "GL_ARB_transform_feedback2",
    "GL_ARB_transform_feedback3",
    "GL_ARB_uniform_buffer_object",
    "GL_ARB_vertex_array_bgra",
    "GL_ARB_vertex_array_object",
    "GL_ARB_vertex_attrib_64bit",      /* dvecN attributes + glVertexAttribL* (vertex_attrib.c) */
    "GL_ARB_vertex_attrib_binding",
    "GL_ARB_vertex_type_10f_11f_11f_rev",
    "GL_ARB_vertex_type_2_10_10_10_rev",
    "GL_ARB_viewport_array",
    "GL_EXT_texture_filter_anisotropic",
    "GL_KHR_debug",
};
#define NUM_MGL_EXTENSIONS ((GLuint)(sizeof(mgl_extensions) / sizeof(mgl_extensions[0])))

/* compat domain: size / element access of the core extension list */
GLuint mgl_num_extensions(void) { return NUM_MGL_EXTENSIONS; }
const char *mgl_extension_name(GLuint i) { return (i < NUM_MGL_EXTENSIONS) ? mgl_extensions[i] : NULL; }

/* GLSL versions accepted by the glslang front end in a core context */
static const char *mgl_glsl_versions[] = {
    "460 core", "450 core", "440 core", "430 core", "420 core", "410 core",
    "400 core", "330 core", "150 core", "140",
};
#define NUM_GLSL_VERSIONS ((GLuint)(sizeof(mgl_glsl_versions) / sizeof(mgl_glsl_versions[0])))

#pragma mark - live state

static GLuint tex_binding(GLMContext ctx, int target_index)
{
    GLuint unit = STATE(active_texture);

    if (unit >= TEXTURE_UNITS)
        return 0;

    Texture *tex = STATE(texture_units[unit].textures[target_index]);
    return tex ? tex->name : 0;
}

static GLboolean cap_value(GLMContext ctx, GLenum pname, bool *found)
{
    *found = true;

    switch (pname)
    {
        case GL_BLEND: return STATE(caps.blend);
        case GL_COLOR_LOGIC_OP: return STATE(caps.color_logic_op);
        case GL_CULL_FACE: return STATE(caps.cull_face);
        case GL_DEPTH_CLAMP: return STATE(caps.depth_clamp);
        case GL_DEPTH_TEST: return STATE(caps.depth_test);
        case GL_DITHER: return STATE(caps.dither);
        case GL_FRAMEBUFFER_SRGB: return STATE(caps.framebuffer_srgb);
        case GL_LINE_SMOOTH: return STATE(caps.line_smooth);
        case GL_MULTISAMPLE: return STATE(caps.multisample);
        case GL_POLYGON_OFFSET_FILL: return STATE(caps.polygon_offset_fill);
        case GL_POLYGON_OFFSET_LINE: return STATE(caps.polygon_offset_line);
        case GL_POLYGON_OFFSET_POINT: return STATE(caps.polygon_offset_point);
        case GL_POLYGON_SMOOTH: return STATE(caps.polygon_smooth);
        case GL_PRIMITIVE_RESTART: return STATE(caps.primitive_restart);
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: return STATE(caps.primitive_restart_fixed_index);
        case GL_PROGRAM_POINT_SIZE: return STATE(caps.program_point_size);
        case GL_RASTERIZER_DISCARD: return STATE(caps.rasterizer_discard);
        case GL_SAMPLE_ALPHA_TO_COVERAGE: return STATE(caps.sample_alpha_to_coverage);
        case GL_SAMPLE_ALPHA_TO_ONE: return STATE(caps.sample_alpha_to_one);
        case GL_SAMPLE_COVERAGE: return STATE(caps.sample_coverage);
        case GL_SAMPLE_MASK: return STATE(caps.sample_mask);
        case GL_SAMPLE_SHADING: return STATE(caps.sample_shading);
        case GL_SCISSOR_TEST: return STATE(caps.scissor_test);
        case GL_STENCIL_TEST: return STATE(caps.stencil_test);
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: return STATE(caps.texture_cube_map_seamless);
        case GL_DEBUG_OUTPUT: return STATE(caps.debug_output);
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: return STATE(caps.debug_output_synchronous);
    }

    if (pname >= GL_CLIP_DISTANCE0 && pname < GL_CLIP_DISTANCE0 + MAX_CLIP_DISTANCES)
        return STATE(caps.clip_distances[pname - GL_CLIP_DISTANCE0]);

    *found = false;
    return GL_FALSE;
}

static double buffer_binding(GLMContext ctx, int index)
{
    return NAME(STATE(buffers[index]));
}

static bool get_state(GLMContext ctx, GLenum pname, GetValues *gv)
{
    bool found;
    GLboolean cap = cap_value(ctx, pname, &found);

    if (found)
        PUT1(cap);

    gv->normalized = false;

    switch (pname)
    {
        /* ---- buffer / object bindings ---- */
        case GL_ARRAY_BUFFER_BINDING: PUT1(buffer_binding(ctx, _ARRAY_BUFFER));
        case GL_ELEMENT_ARRAY_BUFFER_BINDING: PUT1(buffer_binding(ctx, _ELEMENT_ARRAY_BUFFER));
        case GL_COPY_READ_BUFFER_BINDING: PUT1(buffer_binding(ctx, _COPY_READ_BUFFER));
        case GL_COPY_WRITE_BUFFER_BINDING: PUT1(buffer_binding(ctx, _COPY_WRITE_BUFFER));
        case GL_DISPATCH_INDIRECT_BUFFER_BINDING: PUT1(buffer_binding(ctx, _DISPATCH_INDIRECT_BUFFER));
        case GL_DRAW_INDIRECT_BUFFER_BINDING: PUT1(buffer_binding(ctx, _DRAW_INDIRECT_BUFFER));
        case GL_PIXEL_PACK_BUFFER_BINDING: PUT1(buffer_binding(ctx, _PIXEL_PACK_BUFFER));
        case GL_PIXEL_UNPACK_BUFFER_BINDING: PUT1(buffer_binding(ctx, _PIXEL_UNPACK_BUFFER));
        case GL_QUERY_BUFFER_BINDING: PUT1(buffer_binding(ctx, _QUERY_BUFFER));
        case GL_TEXTURE_BUFFER_BINDING: PUT1(buffer_binding(ctx, _TEXTURE_BUFFER));
        case GL_UNIFORM_BUFFER_BINDING: PUT1(buffer_binding(ctx, _UNIFORM_BUFFER));
        case GL_SHADER_STORAGE_BUFFER_BINDING: PUT1(buffer_binding(ctx, _SHADER_STORAGE_BUFFER));
        case GL_ATOMIC_COUNTER_BUFFER_BINDING: PUT1(buffer_binding(ctx, _ATOMIC_COUNTER_BUFFER));
        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: PUT1(buffer_binding(ctx, _TRANSFORM_FEEDBACK_BUFFER));
        case GL_PARAMETER_BUFFER_BINDING: PUT1(0);  /* no MGL binding point; the buffers domain hooks it */

        case GL_VERTEX_ARRAY_BINDING: PUT1(NAME(STATE(vao)));
        case GL_CURRENT_PROGRAM: PUT1(NAME(STATE(program)));
        case GL_PROGRAM_PIPELINE_BINDING: PUT1(NAME(STATE(program_pipeline)));
        case GL_TRANSFORM_FEEDBACK_BINDING: PUT1(NAME(STATE(transform_feedback)));
        case GL_TRANSFORM_FEEDBACK_ACTIVE: PUT1(STATE(transform_feedback) ? STATE(transform_feedback)->active : 0);
        case GL_TRANSFORM_FEEDBACK_PAUSED: PUT1(STATE(transform_feedback) ? STATE(transform_feedback)->paused : 0);
        case GL_DRAW_FRAMEBUFFER_BINDING: PUT1(NAME(STATE(framebuffer)));
        case GL_READ_FRAMEBUFFER_BINDING: PUT1(NAME(STATE(readbuffer)));
        case GL_RENDERBUFFER_BINDING: PUT1(NAME(STATE(renderbuffer)));

        case GL_ACTIVE_TEXTURE: PUT1(GL_TEXTURE0 + STATE(active_texture));
        case GL_TEXTURE_BINDING_1D: PUT1(tex_binding(ctx, _TEXTURE_1D));
        case GL_TEXTURE_BINDING_2D: PUT1(tex_binding(ctx, _TEXTURE_2D));
        case GL_TEXTURE_BINDING_3D: PUT1(tex_binding(ctx, _TEXTURE_3D));
        case GL_TEXTURE_BINDING_1D_ARRAY: PUT1(tex_binding(ctx, _TEXTURE_1D_ARRAY));
        case GL_TEXTURE_BINDING_2D_ARRAY: PUT1(tex_binding(ctx, _TEXTURE_2D_ARRAY));
        case GL_TEXTURE_BINDING_RECTANGLE: PUT1(tex_binding(ctx, _TEXTURE_RECTANGLE));
        case GL_TEXTURE_BINDING_CUBE_MAP: PUT1(tex_binding(ctx, _TEXTURE_CUBE_MAP));
        case GL_TEXTURE_BINDING_CUBE_MAP_ARRAY: PUT1(tex_binding(ctx, _TEXTURE_CUBE_MAP_ARRAY));
        case GL_TEXTURE_BINDING_BUFFER: PUT1(tex_binding(ctx, _TEXTURE_BUFFER));
        case GL_TEXTURE_BINDING_2D_MULTISAMPLE: PUT1(tex_binding(ctx, _TEXTURE_2D_MULTISAMPLE));
        case GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY: PUT1(tex_binding(ctx, _TEXTURE_2D_MULTISAMPLE_ARRAY));
        case GL_SAMPLER_BINDING:
            PUT1(STATE(active_texture) < TEXTURE_UNITS ? NAME(STATE(texture_samplers[STATE(active_texture)])) : 0);

        /* ---- framebuffer control ---- */
        case GL_DRAW_BUFFER: PUT1(STATE(draw_buffer));
        case GL_READ_BUFFER: PUT1(STATE(read_buffer));
        case GL_COLOR_WRITEMASK:
            PUT4(STATE_VAR(color_writemask[0][0]), STATE_VAR(color_writemask[0][1]),
                 STATE_VAR(color_writemask[0][2]), STATE_VAR(color_writemask[0][3]));
        case GL_DEPTH_WRITEMASK: PUT1(STATE_VAR(depth_writemask));
        case GL_STENCIL_WRITEMASK: PUT1(STATE_VAR(stencil_writemask));
        case GL_STENCIL_BACK_WRITEMASK: PUT1(STATE_VAR(stencil_back_writemask));
        case GL_COLOR_CLEAR_VALUE:
            gv->normalized = true;
            PUT4(STATE(color_clear_value[0]), STATE(color_clear_value[1]),
                 STATE(color_clear_value[2]), STATE(color_clear_value[3]));
        case GL_DEPTH_CLEAR_VALUE: gv->normalized = true; PUT1(STATE_VAR(depth_clear_value));
        case GL_STENCIL_CLEAR_VALUE: PUT1(STATE_VAR(stencil_clear_value));

        /* ---- viewport / rasterization ---- */
        case GL_VIEWPORT: PUT4(STATE(viewport[0]), STATE(viewport[1]), STATE(viewport[2]), STATE(viewport[3]));
        case GL_DEPTH_RANGE: gv->normalized = true; PUT2(STATE_VAR(depth_range[0]), STATE_VAR(depth_range[1]));
        case GL_SCISSOR_BOX:
            PUT4((GLint)STATE_VAR(scissor_box[0]), (GLint)STATE_VAR(scissor_box[1]),
                 (GLint)STATE_VAR(scissor_box[2]), (GLint)STATE_VAR(scissor_box[3]));
        case GL_CLIP_ORIGIN: PUT1(GL_LOWER_LEFT);
        case GL_CLIP_DEPTH_MODE: PUT1(GL_NEGATIVE_ONE_TO_ONE);
        case GL_POINT_SIZE: PUT1(STATE_VAR(point_size));
        case GL_POINT_FADE_THRESHOLD_SIZE: PUT1(1.0);
        case GL_POINT_SPRITE_COORD_ORIGIN: PUT1(GL_UPPER_LEFT);
        case GL_LINE_WIDTH: PUT1(STATE_VAR(line_width));
        case GL_CULL_FACE_MODE: PUT1(STATE_VAR(cull_face_mode));
        case GL_FRONT_FACE: PUT1(STATE_VAR(front_face));
        case GL_POLYGON_MODE: PUT2(STATE_VAR(polygon_mode), STATE_VAR(polygon_mode));
        case GL_POLYGON_OFFSET_FACTOR: PUT1(0.0);   /* glPolygonOffset state lives in the rstate domain */
        case GL_POLYGON_OFFSET_UNITS: PUT1(0.0);
        case GL_POLYGON_OFFSET_CLAMP: PUT1(0.0);
        case GL_PROVOKING_VERTEX: PUT1(GL_LAST_VERTEX_CONVENTION);
        case GL_PRIMITIVE_RESTART_INDEX: PUT1(0);
        case GL_SAMPLE_COVERAGE_VALUE: PUT1(1.0);
        case GL_SAMPLE_COVERAGE_INVERT: PUT1(GL_FALSE);
        case GL_MIN_SAMPLE_SHADING_VALUE: PUT1(0.0);
        case GL_PATCH_VERTICES: PUT1(3);
        case GL_PATCH_DEFAULT_INNER_LEVEL: PUT2(1.0, 1.0);
        case GL_PATCH_DEFAULT_OUTER_LEVEL: PUT4(1.0, 1.0, 1.0, 1.0);

        /* ---- per-fragment ---- */
        case GL_DEPTH_FUNC: PUT1(STATE_VAR(depth_func));
        case GL_STENCIL_FUNC: PUT1(STATE_VAR(stencil_func));
        case GL_STENCIL_VALUE_MASK: PUT1(STATE_VAR(stencil_value_mask));
        case GL_STENCIL_REF: PUT1((GLint)STATE_VAR(stencil_ref));
        case GL_STENCIL_FAIL: PUT1(STATE_VAR(stencil_fail));
        case GL_STENCIL_PASS_DEPTH_FAIL: PUT1(STATE_VAR(stencil_pass_depth_fail));
        case GL_STENCIL_PASS_DEPTH_PASS: PUT1(STATE_VAR(stencil_pass_depth_pass));
        case GL_STENCIL_BACK_FUNC: PUT1(STATE_VAR(stencil_back_func));
        case GL_STENCIL_BACK_VALUE_MASK: PUT1(STATE_VAR(stencil_back_value_mask));
        case GL_STENCIL_BACK_REF: PUT1((GLint)STATE_VAR(stencil_back_ref));
        case GL_STENCIL_BACK_FAIL: PUT1(STATE_VAR(stencil_back_fail));
        case GL_STENCIL_BACK_PASS_DEPTH_FAIL: PUT1(STATE_VAR(stencil_back_pass_depth_fail));
        case GL_STENCIL_BACK_PASS_DEPTH_PASS: PUT1(STATE_VAR(stencil_back_pass_depth_pass));
        case GL_BLEND_SRC: /* = GL_BLEND_SRC_RGB semantics for attachment 0 */
        case GL_BLEND_SRC_RGB: PUT1(STATE_VAR(blend_src_rgb[0]));
        case GL_BLEND_DST:
        case GL_BLEND_DST_RGB: PUT1(STATE_VAR(blend_dst_rgb[0]));
        case GL_BLEND_SRC_ALPHA: PUT1(STATE_VAR(blend_src_alpha[0]));
        case GL_BLEND_DST_ALPHA: PUT1(STATE_VAR(blend_dst_alpha[0]));
        case GL_BLEND_EQUATION_RGB: PUT1(STATE_VAR(blend_equation_rgb[0]));
        case GL_BLEND_EQUATION_ALPHA: PUT1(STATE_VAR(blend_equation_alpha[0]));
        case GL_BLEND_COLOR:
            gv->normalized = true;
            PUT4(STATE_VAR(blend_color[0]), STATE_VAR(blend_color[1]), STATE_VAR(blend_color[2]), STATE_VAR(blend_color[3]));
        case GL_LOGIC_OP_MODE: PUT1(STATE_VAR(logic_op) ? STATE_VAR(logic_op) : GL_COPY);

        /* ---- hints ---- */
        case GL_LINE_SMOOTH_HINT: PUT1(STATE(hints.line_smooth_hint));
        case GL_POLYGON_SMOOTH_HINT: PUT1(STATE(hints.polygon_smooth_hint));
        case GL_TEXTURE_COMPRESSION_HINT: PUT1(STATE(hints.texture_compression_hint));
        case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: PUT1(STATE(hints.fragment_shader_derivative_hint));

        /* ---- pixel store ---- */
        case GL_PACK_SWAP_BYTES: PUT1(STATE(pack.swap_bytes));
        case GL_PACK_LSB_FIRST: PUT1(STATE(pack.lsb_first));
        case GL_PACK_ROW_LENGTH: PUT1(STATE(pack.row_length));
        case GL_PACK_IMAGE_HEIGHT: PUT1(STATE(pack.image_height));
        case GL_PACK_SKIP_ROWS: PUT1(STATE(pack.skip_rows));
        case GL_PACK_SKIP_PIXELS: PUT1(STATE(pack.skip_pixels));
        case GL_PACK_SKIP_IMAGES: PUT1(STATE(pack.skip_images));
        case GL_PACK_ALIGNMENT: PUT1(STATE(pack.alignment));
        case GL_UNPACK_SWAP_BYTES: PUT1(STATE(unpack.swap_bytes));
        case GL_UNPACK_LSB_FIRST: PUT1(STATE(unpack.lsb_first));
        case GL_UNPACK_ROW_LENGTH: PUT1(STATE(unpack.row_length));
        case GL_UNPACK_IMAGE_HEIGHT: PUT1(STATE(unpack.image_height));
        case GL_UNPACK_SKIP_ROWS: PUT1(STATE(unpack.skip_rows));
        case GL_UNPACK_SKIP_PIXELS: PUT1(STATE(unpack.skip_pixels));
        case GL_UNPACK_SKIP_IMAGES: PUT1(STATE(unpack.skip_images));
        case GL_UNPACK_ALIGNMENT: PUT1(STATE(unpack.alignment));
        case GL_PACK_COMPRESSED_BLOCK_WIDTH:
        case GL_PACK_COMPRESSED_BLOCK_HEIGHT:
        case GL_PACK_COMPRESSED_BLOCK_DEPTH:
        case GL_PACK_COMPRESSED_BLOCK_SIZE:
        case GL_UNPACK_COMPRESSED_BLOCK_WIDTH:
        case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT:
        case GL_UNPACK_COMPRESSED_BLOCK_DEPTH:
        case GL_UNPACK_COMPRESSED_BLOCK_SIZE:
            PUT1(0);    /* PixelStore has no compressed-block fields; texdata domain hooks them when stored */

        /* ---- context ---- */
        case GL_MAJOR_VERSION: PUT1(4);
        case GL_MINOR_VERSION: PUT1(6);
        case GL_NUM_EXTENSIONS: PUT1(NUM_MGL_EXTENSIONS);
        case GL_NUM_SHADING_LANGUAGE_VERSIONS: PUT1(NUM_GLSL_VERSIONS);
        case GL_NUM_SPIR_V_EXTENSIONS: PUT1(0);
        case GL_CONTEXT_FLAGS:
            PUT1(STATE_VAR(context_flags) & (GL_CONTEXT_FLAG_FORWARD_COMPATIBLE_BIT | GL_CONTEXT_FLAG_DEBUG_BIT |
                                              GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT | GL_CONTEXT_FLAG_NO_ERROR_BIT));
        case GL_CONTEXT_PROFILE_MASK: PUT1(GL_CONTEXT_CORE_PROFILE_BIT);
        case GL_CONTEXT_RELEASE_BEHAVIOR: PUT1(GL_CONTEXT_RELEASE_BEHAVIOR_FLUSH);
        case GL_RESET_NOTIFICATION_STRATEGY: PUT1(GL_NO_RESET_NOTIFICATION);
        case GL_DOUBLEBUFFER: PUT1(GL_TRUE);
        case GL_STEREO: PUT1(GL_FALSE);
        case GL_SAMPLE_BUFFERS: PUT1(0);    /* default framebuffer is single-sampled; fb domain hooks FBOs */
        case GL_SAMPLES: PUT1(0);
        case GL_IMPLEMENTATION_COLOR_READ_FORMAT: PUT1(GL_RGBA);
        case GL_IMPLEMENTATION_COLOR_READ_TYPE: PUT1(GL_UNSIGNED_BYTE);
        case GL_TIMESTAMP: {
            static mach_timebase_info_data_t tb;
            if (tb.denom == 0) mach_timebase_info(&tb);
            PUT1((double)mach_absolute_time() * tb.numer / tb.denom);
        }

        case GL_NUM_COMPRESSED_TEXTURE_FORMATS: PUT1(NUM_COMPRESSED_FORMATS);
        case GL_COMPRESSED_TEXTURE_FORMATS:
            gv->count = NUM_COMPRESSED_FORMATS;
            for (int i = 0; i < NUM_COMPRESSED_FORMATS; i++) gv->v[i] = compressed_formats[i];
            return true;
        case GL_NUM_SHADER_BINARY_FORMATS: PUT1(0);
        case GL_SHADER_BINARY_FORMATS: gv->count = 0; return true;
        case GL_NUM_PROGRAM_BINARY_FORMATS: PUT1(0);
        case GL_PROGRAM_BINARY_FORMATS: gv->count = 0; return true;
        case GL_SHADER_COMPILER: PUT1(GL_TRUE);
    }

    return false;
}

static bool get_limit(GLMContext ctx, GLenum pname, GetValues *gv)
{
    switch (pname)
    {
        /* textures / framebuffers */
        case GL_MAX_TEXTURE_SIZE: PUT1(LIMIT(STATE_VAR(max_texture_size), 16384));
        case GL_MAX_3D_TEXTURE_SIZE: PUT1(LIMIT(STATE_VAR(max_3d_texture_size), 2048));
        case GL_MAX_CUBE_MAP_TEXTURE_SIZE: PUT1(LIMIT(STATE_VAR(max_cube_map_texture_size), 16384));
        case GL_MAX_RECTANGLE_TEXTURE_SIZE: PUT1(LIMIT(STATE_VAR(max_rectangle_texture_size), 16384));
        case GL_MAX_ARRAY_TEXTURE_LAYERS: PUT1(LIMIT(STATE_VAR(max_array_texture_layers), 2048));
        case GL_MAX_TEXTURE_BUFFER_SIZE: PUT1(LIMIT(STATE_VAR(max_texture_buffer_size), 1 << 26));
        case GL_MAX_TEXTURE_LOD_BIAS: PUT1(LIMIT(STATE_VAR(max_texture_lod_bias), 16));
        case GL_MAX_TEXTURE_MAX_ANISOTROPY: PUT1(16.0);
        case GL_MAX_RENDERBUFFER_SIZE: PUT1(LIMIT(STATE_VAR(max_renderbuffer_size), 16384));
        case GL_MAX_VIEWPORT_DIMS: PUT2(16384, 16384);
        case GL_MAX_FRAMEBUFFER_WIDTH: PUT1(LIMIT(STATE_VAR(max_framebuffer_width), 16384));
        case GL_MAX_FRAMEBUFFER_HEIGHT: PUT1(LIMIT(STATE_VAR(max_framebuffer_height), 16384));
        case GL_MAX_FRAMEBUFFER_LAYERS: PUT1(LIMIT(STATE_VAR(max_framebuffer_layers), 2048));
        case GL_MAX_FRAMEBUFFER_SAMPLES: PUT1(LIMIT(STATE_VAR(max_framebuffer_samples), 4));
        case GL_MAX_SAMPLES: PUT1(4);
        case GL_MAX_COLOR_TEXTURE_SAMPLES: PUT1(LIMIT(STATE_VAR(max_color_texture_samples), 4));
        case GL_MAX_DEPTH_TEXTURE_SAMPLES: PUT1(LIMIT(STATE_VAR(max_depth_texture_samples), 4));
        case GL_MAX_INTEGER_SAMPLES: PUT1(LIMIT(STATE_VAR(max_integer_samples), 4));
        case GL_MAX_SAMPLE_MASK_WORDS: PUT1(LIMIT(STATE_VAR(max_sample_mask_words), 1));
        case GL_MAX_COLOR_ATTACHMENTS: PUT1(MIN_D(LIMIT(STATE(max_color_attachments), MAX_COLOR_ATTACHMENTS), MAX_COLOR_ATTACHMENTS));
        case GL_MAX_DRAW_BUFFERS: PUT1(max_draw_buffers(ctx));
        case GL_MAX_DUAL_SOURCE_DRAW_BUFFERS: PUT1(1);
        case GL_MAX_TEXTURE_IMAGE_UNITS: PUT1(LIMIT(STATE_VAR(max_texture_image_units), 16));
        case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: PUT1(LIMIT(STATE_VAR(max_vertex_texture_image_units), 16));
        case GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS: PUT1(LIMIT(STATE_VAR(max_geometry_texture_image_units), 16));
        case GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS: PUT1(16);
        case GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS: PUT1(16);
        case GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS: PUT1(LIMIT(STATE_VAR(max_compute_texture_image_units), 16));
        case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: PUT1(LIMIT(STATE_VAR(max_combined_texture_image_units), 80));
        case GL_MIN_PROGRAM_TEXEL_OFFSET: PUT1(STATE_VAR(min_program_texel_offset) ? (GLint)STATE_VAR(min_program_texel_offset) : -8);
        case GL_MAX_PROGRAM_TEXEL_OFFSET: PUT1(LIMIT(STATE_VAR(max_program_texel_offset), 7));
        case GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET: PUT1(-8);
        case GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET: PUT1(7);
        case GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT: PUT1(LIMIT(STATE_VAR(texture_buffer_offset_alignment), 256));
        case GL_MAX_IMAGE_UNITS: PUT1(MGL_MAX_IMAGE_UNITS);
        case GL_MAX_IMAGE_SAMPLES: PUT1(0);
        /* the geometry kernel is a compute pass and binds image units like the compute path
         * (MGLRenderer+Geometry.m's resource walk), so it supports the full set of image units */
        case GL_MAX_GEOMETRY_IMAGE_UNIFORMS: PUT1(MGL_MAX_IMAGE_UNITS);
        case GL_MAX_VERTEX_IMAGE_UNIFORMS:
        case GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS:
        case GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS: PUT1(0);
        case GL_MAX_FRAGMENT_IMAGE_UNIFORMS:
        case GL_MAX_COMPUTE_IMAGE_UNIFORMS:
        case GL_MAX_COMBINED_IMAGE_UNIFORMS: PUT1(MGL_MAX_IMAGE_UNITS);
        case GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES: PUT1(MGL_MAX_IMAGE_UNITS + MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS + MAX_COLOR_ATTACHMENTS);

        /* points / lines (Metal rasterizes 1-pixel lines and points up to 511 px) */
        case GL_POINT_SIZE_RANGE: PUT2(1.0, 511.0);
        case GL_POINT_SIZE_GRANULARITY: PUT1(1.0);
        case GL_ALIASED_LINE_WIDTH_RANGE: PUT2(1.0, 1.0);
        case GL_SMOOTH_LINE_WIDTH_RANGE: PUT2(1.0, 1.0);
        case GL_SMOOTH_LINE_WIDTH_GRANULARITY: PUT1(1.0);
        case GL_SUBPIXEL_BITS: PUT1(8);

        /* viewports */
        case GL_MAX_VIEWPORTS: PUT1(MGL_MAX_VIEWPORTS);
        case GL_VIEWPORT_SUBPIXEL_BITS: PUT1(0);
        case GL_VIEWPORT_BOUNDS_RANGE: PUT2(-32768.0, 32767.0);
        case GL_LAYER_PROVOKING_VERTEX: PUT1(GL_UNDEFINED_VERTEX);
        case GL_VIEWPORT_INDEX_PROVOKING_VERTEX: PUT1(GL_UNDEFINED_VERTEX);

        /* vertex input */
        case GL_MAX_VERTEX_ATTRIBS: PUT1(MIN_D(LIMIT(STATE(max_vertex_attribs), MAX_ATTRIBS), MAX_ATTRIBS));
        case GL_MAX_VERTEX_ATTRIB_BINDINGS: PUT1(MGL_MAX_VERTEX_ATTRIB_BINDINGS);
        case GL_MAX_VERTEX_ATTRIB_RELATIVE_OFFSET: PUT1(LIMIT(STATE_VAR(max_vertex_attrib_relative_offset), 2047));
        case GL_MAX_VERTEX_ATTRIB_STRIDE: PUT1(2048);
        case GL_MAX_ELEMENTS_VERTICES: PUT1(LIMIT(STATE_VAR(max_elements_vertices), 1048575));
        case GL_MAX_ELEMENTS_INDICES: PUT1(LIMIT(STATE_VAR(max_elements_indices), 150000));
        case GL_MAX_ELEMENT_INDEX: PUT1(4294967295.0);
        case GL_PRIMITIVE_RESTART_FOR_PATCHES_SUPPORTED: PUT1(GL_FALSE);

        /* uniforms / varyings */
        case GL_MAX_VERTEX_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_vertex_uniform_components), 4096));
        case GL_MAX_VERTEX_UNIFORM_VECTORS: PUT1(LIMIT(STATE_VAR(max_vertex_uniform_vectors), 1024));
        case GL_MAX_FRAGMENT_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_fragment_uniform_components), 4096));
        case GL_MAX_FRAGMENT_UNIFORM_VECTORS: PUT1(LIMIT(STATE_VAR(max_fragment_uniform_vectors), 1024));
        /* the geometry constants must match what glslang bakes into shaders as
         * gl_MaxGeometry* (its default resources: 1024 uniform components, no atomic
         * counters) — KHR-GL46.geometry_shader.constant_variables compares the two */
        case GL_MAX_GEOMETRY_UNIFORM_COMPONENTS: PUT1(1024);
        case GL_MAX_TESS_CONTROL_UNIFORM_COMPONENTS: PUT1(1024);
        case GL_MAX_TESS_EVALUATION_UNIFORM_COMPONENTS: PUT1(1024);
        case GL_MAX_COMPUTE_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_compute_uniform_components), 1024));
        case GL_MAX_UNIFORM_LOCATIONS: PUT1(LIMIT(STATE_VAR(max_uniform_locations), 1024));
        case GL_MAX_VARYING_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_varying_components), 124));
        case GL_MAX_VARYING_VECTORS: PUT1(LIMIT(STATE_VAR(max_varying_vectors), 31));
        case GL_MAX_VERTEX_OUTPUT_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_vertex_output_components), 128));
        case GL_MAX_GEOMETRY_INPUT_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_geometry_input_components), 128));
        case GL_MAX_GEOMETRY_OUTPUT_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_geometry_output_components), 128));
        case GL_MAX_FRAGMENT_INPUT_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_fragment_input_components), 128));
        case GL_MAX_GEOMETRY_OUTPUT_VERTICES: PUT1(256);
        case GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS: PUT1(2048);
        case GL_MAX_GEOMETRY_SHADER_INVOCATIONS: PUT1(32);
        case GL_MAX_VERTEX_STREAMS: PUT1(4);
        case GL_MAX_CLIP_DISTANCES: PUT1(MAX_CLIP_DISTANCES);
        case GL_MAX_CULL_DISTANCES: PUT1(MAX_CLIP_DISTANCES);
        case GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES: PUT1(MAX_CLIP_DISTANCES);
        case GL_MAX_SUBROUTINES: PUT1(256);
        case GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS: PUT1(1024);
        case GL_MIN_FRAGMENT_INTERPOLATION_OFFSET: PUT1(-0.5);
        case GL_MAX_FRAGMENT_INTERPOLATION_OFFSET: PUT1(0.5);
        case GL_FRAGMENT_INTERPOLATION_OFFSET_BITS: PUT1(4);

        /* tessellation */
        case GL_MAX_PATCH_VERTICES: PUT1(32);
        case GL_MAX_TESS_GEN_LEVEL: PUT1(64);
        case GL_MAX_TESS_CONTROL_OUTPUT_COMPONENTS: PUT1(128);
        case GL_MAX_TESS_PATCH_COMPONENTS: PUT1(120);
        case GL_MAX_TESS_CONTROL_TOTAL_OUTPUT_COMPONENTS: PUT1(4096);
        case GL_MAX_TESS_EVALUATION_OUTPUT_COMPONENTS: PUT1(128);
        case GL_MAX_TESS_CONTROL_INPUT_COMPONENTS: PUT1(128);
        case GL_MAX_TESS_EVALUATION_INPUT_COMPONENTS: PUT1(128);

        /* uniform blocks */
        case GL_MAX_VERTEX_UNIFORM_BLOCKS: PUT1(MIN_D(LIMIT(STATE_VAR(max_vertex_uniform_blocks), 14), max_uniform_buffer_bindings(ctx)));
        /* the geometry stage runs as a COMPUTE kernel (compute emulation): its user
         * UBOs share the kernel's buffer argument table (slots 0..17; 18..27 are the
         * emulation's own — MGL_GEOM_SSBO_*), so the geometry-stage block limit is
         * capped at the GL 4.6 minimum the kernel can actually bind, not at what the
         * render stages report.  Reporting more makes conformant apps build pipelines
         * the kernel cannot host (KHR-GL46.geometry_shader.limits.max_uniform_blocks
         * builds MAX_GEOMETRY_UNIFORM_BLOCKS blocks and renders). */
        case GL_MAX_GEOMETRY_UNIFORM_BLOCKS:
            PUT1(MIN_D(MIN_D(LIMIT(STATE_VAR(max_geometry_uniform_blocks), 14),
                             max_uniform_buffer_bindings(ctx)), 14));
        case GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS: PUT1(MIN_D(LIMIT(STATE_VAR(max_tess_control_uniform_blocks), 14), max_uniform_buffer_bindings(ctx)));
        case GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS: PUT1(MIN_D(LIMIT(STATE_VAR(max_tess_evaluation_uniform_blocks), 14), max_uniform_buffer_bindings(ctx)));
        case GL_MAX_FRAGMENT_UNIFORM_BLOCKS: PUT1(MIN_D(LIMIT(STATE_VAR(max_fragment_uniform_blocks), 14), max_uniform_buffer_bindings(ctx)));
        case GL_MAX_COMPUTE_UNIFORM_BLOCKS: PUT1(MIN_D(LIMIT(STATE_VAR(max_compute_uniform_blocks), 14), max_uniform_buffer_bindings(ctx)));
        case GL_MAX_COMBINED_UNIFORM_BLOCKS: PUT1(LIMIT(STATE_VAR(max_combined_uniform_blocks), 70));
        case GL_MAX_UNIFORM_BUFFER_BINDINGS: PUT1(max_uniform_buffer_bindings(ctx));
        case GL_MAX_UNIFORM_BLOCK_SIZE: PUT1(LIMIT(STATE_VAR(max_uniform_block_size), 65536));
        case GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT: PUT1(LIMIT(STATE_VAR(uniform_buffer_offset_alignment), 256));
        case GL_MAX_COMBINED_VERTEX_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_combined_vertex_uniform_components), 233472));
        case GL_MAX_COMBINED_GEOMETRY_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_combined_geometry_uniform_components), 230400));
        case GL_MAX_COMBINED_TESS_CONTROL_UNIFORM_COMPONENTS: PUT1(230400);
        case GL_MAX_COMBINED_TESS_EVALUATION_UNIFORM_COMPONENTS: PUT1(230400);
        case GL_MAX_COMBINED_FRAGMENT_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_combined_fragment_uniform_components), 233472));
        case GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS: PUT1(LIMIT(STATE_VAR(max_combined_compute_uniform_components), 230400));

        /* shader storage */
        case GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS:
        case GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS:
        case GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS:
        case GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS:
        case GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS:
        case GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS:
        case GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS:
        case GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS: PUT1(MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS);
        case GL_MAX_SHADER_STORAGE_BLOCK_SIZE: PUT1(1 << 27);
        case GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT: PUT1(LIMIT(STATE_VAR(shader_storage_buffer_offset_alignment), 16));

        /* atomic counters */
        case GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_TESS_CONTROL_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_GEOMETRY_ATOMIC_COUNTER_BUFFERS: PUT1(0);   /* see above */
        case GL_MAX_COMBINED_ATOMIC_COUNTER_BUFFERS:
        case GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS: PUT1(MGL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS);
        case GL_MAX_VERTEX_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_vertex_atomic_counters), 1024));
        case GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_tess_control_atomic_counters), 1024));
        case GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_tess_evaluation_atomic_counters), 1024));
        case GL_MAX_GEOMETRY_ATOMIC_COUNTERS: PUT1(0);
        case GL_MAX_FRAGMENT_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_fragment_atomic_counters), 1024));
        case GL_MAX_COMPUTE_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_compute_atomic_counters), 1024));
        case GL_MAX_COMBINED_ATOMIC_COUNTERS: PUT1(LIMIT(STATE_VAR(max_combined_atomic_counters), 1024));
        case GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE: PUT1(16384);

        /* compute */
        case GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: PUT1(LIMIT(STATE_VAR(max_compute_work_group_invocations), 1024));
        case GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: PUT1(32768);

        /* transform feedback */
        case GL_MAX_TRANSFORM_FEEDBACK_BUFFERS: PUT1(MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS);
        case GL_MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS: PUT1(64);
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_ATTRIBS: PUT1(4);
        case GL_MAX_TRANSFORM_FEEDBACK_SEPARATE_COMPONENTS: PUT1(4);

        /* misc */
        case GL_MAX_SERVER_WAIT_TIMEOUT: PUT1(9.2233720368547748e18);   /* effectively unbounded */
        case GL_MIN_MAP_BUFFER_ALIGNMENT: PUT1(LIMIT(STATE_VAR(min_map_buffer_alignment), 64));
    }

    return false;
}

#pragma mark - glGet

/* entry-point name for error / KHR_debug messages */
static const char *get_func_name(GLuint type, bool indexed)
{
    static const char *plain[] = { "mglGetBooleanv", "mglGetIntegerv", "mglGetFloatv", "mglGetDoublev", "mglGetInteger64v" };
    static const char *idx[] = { "mglGetBooleani_v", "mglGetIntegeri_v", "mglGetFloati_v", "mglGetDoublei_v", "mglGetInteger64i_v" };
    return type <= MGL_GET_INT64 ? (indexed ? idx[type] : plain[type]) : "mglGet";
}

static void mglGet(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    GetValues gv;

    if (!data)
        return;

    /* 1. domains with their own state for this pname */
    if (mgl_ext_get(ctx, pname, type, data))
        return;

    gv.count = 0;
    gv.normalized = false;

    /* 2./3. live state, then implementation limits */
    if (get_state(ctx, pname, &gv) || get_limit(ctx, pname, &gv))
    {
        store_values(type, data, &gv);
        return;
    }

    ctx->error_func(ctx, get_func_name(type, false), GL_INVALID_ENUM);
}

void mglGetBooleanv(GLMContext ctx, GLenum pname, GLboolean *data)
{
    mglGet(ctx, pname, MGL_GET_BOOL, (void *)data);
}

void mglGetDoublev(GLMContext ctx, GLenum pname, GLdouble *data)
{
    mglGet(ctx, pname, MGL_GET_DOUBLE, (void *)data);
}

void mglGetFloatv(GLMContext ctx, GLenum pname, GLfloat *data)
{
    mglGet(ctx, pname, MGL_GET_FLOAT, (void *)data);
}

void mglGetIntegerv(GLMContext ctx, GLenum pname, GLint *data)
{
    mglGet(ctx, pname, MGL_GET_INT, (void *)data);
}

void mglGetInteger64v(GLMContext ctx, GLenum pname, GLint64 *data)
{
    mglGet(ctx, pname, MGL_GET_INT64, (void *)data);
}

#pragma mark - indexed glGet

/* returns 1 if served, 0 if pname unknown, -1 if index out of range */
static int get_indexed_state(GLMContext ctx, GLenum pname, GLuint index, GetValues *gv)
{
    BufferBaseTarget *bb = NULL;
    GLuint limit = 0;
    int field = 0;   /* 0 binding, 1 start, 2 size */

    gv->normalized = false;

    switch (pname)
    {
        case GL_UNIFORM_BUFFER_BINDING: field = 0; goto ubo;
        case GL_UNIFORM_BUFFER_START: field = 1; goto ubo;
        case GL_UNIFORM_BUFFER_SIZE: field = 2;
        ubo:
            limit = max_uniform_buffer_bindings(ctx);
            if (index >= limit) return -1;
            bb = &STATE(buffer_base[_UNIFORM_BUFFER].buffers[index]);
            break;

        case GL_SHADER_STORAGE_BUFFER_BINDING: field = 0; goto ssbo;
        case GL_SHADER_STORAGE_BUFFER_START: field = 1; goto ssbo;
        case GL_SHADER_STORAGE_BUFFER_SIZE: field = 2;
        ssbo:
            if (index >= MGL_MAX_SHADER_STORAGE_BUFFER_BINDINGS) return -1;
            bb = &STATE(buffer_base[_SHADER_STORAGE_BUFFER].buffers[index]);
            break;

        case GL_ATOMIC_COUNTER_BUFFER_BINDING: field = 0; goto acb;
        case GL_ATOMIC_COUNTER_BUFFER_START: field = 1; goto acb;
        case GL_ATOMIC_COUNTER_BUFFER_SIZE: field = 2;
        acb:
            if (index >= MGL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS) return -1;
            bb = &STATE(buffer_base[_ATOMIC_COUNTER_BUFFER].buffers[index]);
            break;

        case GL_TRANSFORM_FEEDBACK_BUFFER_BINDING: field = 0; goto xfb;
        case GL_TRANSFORM_FEEDBACK_BUFFER_START: field = 1; goto xfb;
        case GL_TRANSFORM_FEEDBACK_BUFFER_SIZE: field = 2;
        xfb:
            if (index >= MGL_MAX_TRANSFORM_FEEDBACK_BUFFERS) return -1;
            bb = &STATE(buffer_base[_TRANSFORM_FEEDBACK_BUFFER].buffers[index]);
            break;

        /* per-draw-buffer blend / mask state */
        case GL_BLEND_SRC_RGB: case GL_BLEND_SRC_ALPHA: case GL_BLEND_DST_RGB: case GL_BLEND_DST_ALPHA:
        case GL_BLEND_EQUATION_RGB: case GL_BLEND_EQUATION_ALPHA: case GL_COLOR_WRITEMASK: case GL_BLEND:
            if (index >= max_draw_buffers(ctx)) return -1;
            switch (pname) {
                case GL_BLEND_SRC_RGB: gv->count = 1; gv->v[0] = STATE_VAR(blend_src_rgb[index]); return 1;
                case GL_BLEND_SRC_ALPHA: gv->count = 1; gv->v[0] = STATE_VAR(blend_src_alpha[index]); return 1;
                case GL_BLEND_DST_RGB: gv->count = 1; gv->v[0] = STATE_VAR(blend_dst_rgb[index]); return 1;
                case GL_BLEND_DST_ALPHA: gv->count = 1; gv->v[0] = STATE_VAR(blend_dst_alpha[index]); return 1;
                case GL_BLEND_EQUATION_RGB: gv->count = 1; gv->v[0] = STATE_VAR(blend_equation_rgb[index]); return 1;
                case GL_BLEND_EQUATION_ALPHA: gv->count = 1; gv->v[0] = STATE_VAR(blend_equation_alpha[index]); return 1;
                case GL_BLEND: gv->count = 1; gv->v[0] = STATE(caps.blend); return 1;
                default:
                    gv->count = 4;
                    for (int i = 0; i < 4; i++) gv->v[i] = STATE_VAR(color_writemask[index][i]);
                    return 1;
            }

        /* image units */
        case GL_IMAGE_BINDING_NAME: case GL_IMAGE_BINDING_LEVEL: case GL_IMAGE_BINDING_LAYERED:
        case GL_IMAGE_BINDING_LAYER: case GL_IMAGE_BINDING_ACCESS: case GL_IMAGE_BINDING_FORMAT: {
            if (index >= MGL_MAX_IMAGE_UNITS) return -1;
            ImageUnit *iu = &STATE(image_units[index]);
            gv->count = 1;
            switch (pname) {
                case GL_IMAGE_BINDING_NAME: gv->v[0] = iu->texture; break;
                case GL_IMAGE_BINDING_LEVEL: gv->v[0] = iu->level; break;
                case GL_IMAGE_BINDING_LAYERED: gv->v[0] = iu->layered; break;
                case GL_IMAGE_BINDING_LAYER: gv->v[0] = iu->layer; break;
                case GL_IMAGE_BINDING_ACCESS: gv->v[0] = iu->texture ? iu->access : GL_READ_ONLY; break;
                default: gv->v[0] = iu->texture ? iu->internalformat : GL_R8; break;
            }
            return 1;
        }

        /* vertex buffer bindings of the bound VAO (derived from the attribs using the binding) */
        case GL_VERTEX_BINDING_BUFFER: case GL_VERTEX_BINDING_OFFSET:
        case GL_VERTEX_BINDING_STRIDE: case GL_VERTEX_BINDING_DIVISOR: {
            if (index >= MGL_MAX_VERTEX_ATTRIB_BINDINGS) return -1;
            VertexAttrib *a = NULL;
            if (STATE(vao))
                for (int i = 0; i < MAX_ATTRIBS && i < (int)STATE(max_vertex_attribs); i++)
                    if (STATE(vao)->attrib[i].buffer_bindingindex == index && STATE(vao)->attrib[i].buffer)
                        { a = &STATE(vao)->attrib[i]; break; }
            gv->count = 1;
            switch (pname) {
                case GL_VERTEX_BINDING_BUFFER: gv->v[0] = a ? a->buffer->name : 0; break;
                case GL_VERTEX_BINDING_OFFSET: gv->v[0] = 0; break;
                case GL_VERTEX_BINDING_STRIDE: gv->v[0] = a ? a->stride : 16; break;
                default: gv->v[0] = a ? a->divisor : 0; break;
            }
            return 1;
        }

        case GL_MAX_COMPUTE_WORK_GROUP_COUNT:
            if (index >= 3) return -1;
            gv->count = 1; gv->v[0] = STATE_VAR(max_compute_work_group_count[index]) ? STATE_VAR(max_compute_work_group_count[index]) : 65535;
            return 1;

        case GL_MAX_COMPUTE_WORK_GROUP_SIZE:
            if (index >= 3) return -1;
            gv->count = 1; gv->v[0] = STATE_VAR(max_compute_work_group_size[index]) ? STATE_VAR(max_compute_work_group_size[index]) : (index == 2 ? 64 : 1024);
            return 1;

        /* viewport arrays: until the rstate domain hooks per-index state, every
           viewport/scissor equals the global one (glViewport/glScissor set all) */
        case GL_VIEWPORT:
            if (index >= MGL_MAX_VIEWPORTS) return -1;
            gv->count = 4;
            for (int i = 0; i < 4; i++) gv->v[i] = STATE(viewport[i]);
            return 1;
        case GL_SCISSOR_BOX:
            if (index >= MGL_MAX_VIEWPORTS) return -1;
            gv->count = 4;
            for (int i = 0; i < 4; i++) gv->v[i] = (GLint)STATE_VAR(scissor_box[i]);
            return 1;
        case GL_DEPTH_RANGE:
            if (index >= MGL_MAX_VIEWPORTS) return -1;
            gv->normalized = true;
            gv->count = 2; gv->v[0] = STATE_VAR(depth_range[0]); gv->v[1] = STATE_VAR(depth_range[1]);
            return 1;
        case GL_SCISSOR_TEST:
            if (index >= MGL_MAX_VIEWPORTS) return -1;
            gv->count = 1; gv->v[0] = STATE(caps.scissor_test);
            return 1;
        case GL_SAMPLE_MASK_VALUE:
            if (index >= LIMIT(STATE_VAR(max_sample_mask_words), 1)) return -1;
            gv->count = 1; gv->v[0] = 4294967295.0;
            return 1;

        default:
            return 0;
    }

    gv->count = 1;
    gv->v[0] = field == 0 ? (double)bb->buffer : field == 1 ? (double)bb->offset : (double)bb->size;
    return 1;
}

static void mglGetIndexed(GLMContext ctx, GLenum target, GLuint index, GLuint type, void *data)
{
    GetValues gv;
    int r;

    if (!data)
        return;

    if (mgl_ext_get_indexed(ctx, target, index, type, data))
        return;

    gv.count = 0;
    gv.normalized = false;

    r = get_indexed_state(ctx, target, index, &gv);
    if (r > 0)
    {
        store_values(type, data, &gv);
        return;
    }

    ctx->error_func(ctx, get_func_name(type, true), r < 0 ? GL_INVALID_VALUE : GL_INVALID_ENUM);
}

void mglGetIntegeri_v(GLMContext ctx, GLenum target, GLuint index, GLint *data)
{
    mglGetIndexed(ctx, target, index, MGL_GET_INT, data);
}

void mglGetInteger64i_v(GLMContext ctx, GLenum target, GLuint index, GLint64 *data)
{
    mglGetIndexed(ctx, target, index, MGL_GET_INT64, data);
}

void mglGetBooleani_v(GLMContext ctx, GLenum target, GLuint index, GLboolean *data)
{
    mglGetIndexed(ctx, target, index, MGL_GET_BOOL, data);
}

void mglGetDoublei_v(GLMContext ctx, GLenum target, GLuint index, GLdouble *data)
{
    mglGetIndexed(ctx, target, index, MGL_GET_DOUBLE, data);
}

void mglGetFloati_v(GLMContext ctx, GLenum target, GLuint index, GLfloat *data)
{
    mglGetIndexed(ctx, target, index, MGL_GET_FLOAT, data);
}

#pragma mark - strings

static void build_strings(GLMContext ctx)
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;
    const char *device = NULL;

    if (st->renderer[0])
        return;

    if (ctx->mtl_funcs.ext_debug && ctx->mtl_funcs.ext_debug->mtlDeviceName && ctx->mtl_funcs.mtlObj)
        device = ctx->mtl_funcs.ext_debug->mtlDeviceName(ctx);

    if (!device || !device[0])
    {
        /* renderer not bound yet: answer generically, retry next time */
        snprintf(st->vendor, sizeof(st->vendor), "Apple");
        snprintf(st->version, sizeof(st->version), "4.6.0 MGL (Metal)");
        return;
    }

    if (strstr(device, "AMD") || strstr(device, "Radeon"))
        snprintf(st->vendor, sizeof(st->vendor), "AMD");
    else if (strstr(device, "Intel"))
        snprintf(st->vendor, sizeof(st->vendor), "Intel");
    else if (strstr(device, "NVIDIA"))
        snprintf(st->vendor, sizeof(st->vendor), "NVIDIA");
    else
        snprintf(st->vendor, sizeof(st->vendor), "Apple");

    snprintf(st->renderer, sizeof(st->renderer), "MGL on Metal (%s)", device);
    snprintf(st->version, sizeof(st->version), "4.6.0 MGL (Metal)");
}

const GLubyte *mglGetString(GLMContext ctx, GLenum name)
{
    struct MGLExtState_debug *st = ctx->state.ext_debug;

    build_strings(ctx);

    switch (name)
    {
        case GL_VENDOR:
            return (const GLubyte *)st->vendor;

        case GL_RENDERER:
            return (const GLubyte *)(st->renderer[0] ? st->renderer : "MGL on Metal");

        case GL_VERSION:
            return (const GLubyte *)st->version;

        case GL_SHADING_LANGUAGE_VERSION:
            return (const GLubyte *)"4.60";
    }

    ERROR_RETURN(GL_INVALID_ENUM);
    return NULL;
}

const GLubyte *mglGetStringi(GLMContext ctx, GLenum name, GLuint index)
{
    switch (name)
    {
        case GL_EXTENSIONS:
            if (index >= NUM_MGL_EXTENSIONS) break;
            return (const GLubyte *)mgl_extensions[index];

        case GL_SHADING_LANGUAGE_VERSION:
            if (index >= NUM_GLSL_VERSIONS) break;
            return (const GLubyte *)mgl_glsl_versions[index];

        case GL_SPIR_V_EXTENSIONS:
            return mgl_program_spirv_extension(ctx, index);    /* program domain */

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return NULL;
    }

    ERROR_RETURN(GL_INVALID_VALUE);
    return NULL;
}
