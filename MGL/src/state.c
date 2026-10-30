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
 * state.c
 * MGL
 *
 * Fixed-function render state (enable caps, blend, depth/stencil, raster, viewport).
 * GL state lives in ctx->state.{caps,var,hints} (read by get.c) and ctx->state.ext_rstate
 * (indexed state); MGLRenderer+RenderState.m applies it to Metal.
 */

#include "mgl.h"
#include "glm_context.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"
/* compat: GL_KHR_blend_equation_advanced — declared directly because
 * mgl_ext_compat.h redefines ERROR_RETURN with a `return;' that does not fit
 * this file's bool helpers (see compat_advblend.c) */
int mgl_advblend_mode(GLenum eq);
void mgl_advblend_armed(GLMContext ctx);   /* compat_advblend.c: seen flag + lazy pre-draw hook */
void mgl_compat_set_blend_advanced_coherent(GLMContext ctx, GLboolean on);   /* compat.c: ext slot */
GLboolean mgl_compat_get_blend_advanced_coherent(GLMContext ctx);

void mgl_rstate_set_all_viewports(GLMContext ctx, GLfloat x, GLfloat y, GLfloat w, GLfloat h);
void mgl_rstate_set_all_scissors(GLMContext ctx, GLint x, GLint y, GLint w, GLint h);
void mgl_rstate_set_all_depth_ranges(GLMContext ctx, GLdouble n, GLdouble f);

#pragma mark - enable / disable

/* which Metal object a capability lives in */
enum { CAP_NONE, CAP_ENCODER, CAP_PIPELINE };

static int rs_cap_kind(GLenum cap)
{
    switch (cap) {
        case GL_BLEND: case GL_SAMPLE_ALPHA_TO_COVERAGE: case GL_SAMPLE_ALPHA_TO_ONE:
        case GL_SAMPLE_COVERAGE: case GL_SAMPLE_MASK: case GL_SAMPLE_SHADING:
        case GL_RASTERIZER_DISCARD: case GL_PROGRAM_POINT_SIZE: case GL_MULTISAMPLE:
        case GL_FRAMEBUFFER_SRGB:
        case GL_DEPTH_TEST: case GL_STENCIL_TEST:   /* default-FB attachments / pipeline formats */
            return CAP_PIPELINE;
        case GL_CULL_FACE: case GL_SCISSOR_TEST: case GL_DEPTH_CLAMP:
        case GL_POLYGON_OFFSET_FILL: case GL_POLYGON_OFFSET_LINE: case GL_POLYGON_OFFSET_POINT:
            return CAP_ENCODER;
        default:
            return CAP_NONE;
    }
}

/* returns false for an unknown cap */
static bool rs_set_cap(GLMContext ctx, GLenum cap, bool on)
{
    GLMCaps *c = &ctx->state.caps;

    switch (cap) {
        case GL_BLEND:
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) RS()->blend[i] = on;
            c->blend = on;
            break;
        case GL_SCISSOR_TEST:
            for (int i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++) RS()->scissor_test[i] = on;
            c->scissor_test = on;
            break;
        case GL_LINE_SMOOTH: c->line_smooth = on; break;
        case GL_POLYGON_SMOOTH: c->polygon_smooth = on; break;
        case GL_CULL_FACE: c->cull_face = on; break;
        case GL_DEPTH_TEST: c->depth_test = on; break;
        case GL_STENCIL_TEST: c->stencil_test = on; break;
        case GL_DITHER: c->dither = on; break;
        case GL_BLEND_ADVANCED_COHERENT_KHR:
            /* compat (GL_KHR_blend_equation_advanced_coherent): query-only state —
             * Metal framebuffer-fetch reads are coherent inside a render pass,
             * so no barrier is ever required (compat_advblend.c).  Lives in the
             * compat ext slot, NOT GLMCaps (layout stability, perf s30). */
            mgl_compat_set_blend_advanced_coherent(ctx, on);
            break;
        case GL_COLOR_LOGIC_OP: c->color_logic_op = on; break;
        case GL_POLYGON_OFFSET_POINT: c->polygon_offset_point = on; break;
        case GL_POLYGON_OFFSET_LINE: c->polygon_offset_line = on; break;
        case GL_POLYGON_OFFSET_FILL: c->polygon_offset_fill = on; break;
        case GL_MULTISAMPLE: c->multisample = on; break;
        case GL_SAMPLE_ALPHA_TO_COVERAGE: c->sample_alpha_to_coverage = on; break;
        case GL_SAMPLE_ALPHA_TO_ONE: c->sample_alpha_to_one = on; break;
        case GL_SAMPLE_COVERAGE: c->sample_coverage = on; break;
        case GL_RASTERIZER_DISCARD: c->rasterizer_discard = on; break;
        case GL_FRAMEBUFFER_SRGB: c->framebuffer_srgb = on; break;
        case GL_PRIMITIVE_RESTART: c->primitive_restart = on; break;
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: c->primitive_restart_fixed_index = on; break;
        case GL_DEPTH_CLAMP: c->depth_clamp = on; break;
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: c->texture_cube_map_seamless = on; break;
        case GL_SAMPLE_MASK: c->sample_mask = on; break;
        case GL_SAMPLE_SHADING: c->sample_shading = on; break;
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: c->debug_output_synchronous = on; break;
        case GL_DEBUG_OUTPUT: c->debug_output = on; break;
        case GL_PROGRAM_POINT_SIZE: c->program_point_size = on; break;
        default:
            if (cap >= GL_CLIP_DISTANCE0 && cap < GL_CLIP_DISTANCE0 + MAX_CLIP_DISTANCES) {
                c->clip_distances[cap - GL_CLIP_DISTANCE0] = on;
                break;
            }
            return false;
    }

    switch (rs_cap_kind(cap)) {
        case CAP_PIPELINE: mgl_rs_dirty_pipeline(ctx); break;
        case CAP_ENCODER:  mgl_rs_dirty_encoder(ctx); break;
        default:           ctx->state.dirty_bits |= DIRTY_STATE; break;
    }
    return true;
}

/* legacy fixed-function caps tolerated (silently) for virglrenderer compatibility */
static bool rs_legacy_cap(GLenum cap)
{
    return cap == GL_TEXTURE_1D || cap == GL_TEXTURE_2D || cap == GL_TEXTURE_3D || cap == GL_TEXTURE_CUBE_MAP;
}

void mglDisable(GLMContext ctx, GLenum cap)
{
    if (rs_set_cap(ctx, cap, false)) return;
    if (rs_legacy_cap(cap)) return;
    ERROR_RETURN(GL_INVALID_ENUM);
}

void mglEnable(GLMContext ctx, GLenum cap)
{
    if (rs_set_cap(ctx, cap, true)) return;
    if (rs_legacy_cap(cap)) return;
    ERROR_RETURN(GL_INVALID_ENUM);
}

#define RET_CAP(_CAP_)  return ctx->state.caps._CAP_

GLboolean mglIsEnabled(GLMContext ctx, GLenum cap)
{
    switch(cap)
    {
        case GL_BLEND: return RS()->blend[0];
        case GL_SCISSOR_TEST: return RS()->scissor_test[0];
        case GL_COLOR_LOGIC_OP: RET_CAP(color_logic_op);
        case GL_CULL_FACE: RET_CAP(cull_face);
        case GL_DEPTH_CLAMP: RET_CAP(depth_clamp);
        case GL_DEBUG_OUTPUT: RET_CAP(debug_output);
        case GL_DEBUG_OUTPUT_SYNCHRONOUS: RET_CAP(debug_output_synchronous);
        case GL_DEPTH_TEST: RET_CAP(depth_test);
        case GL_DITHER: RET_CAP(dither);
        case GL_BLEND_ADVANCED_COHERENT_KHR:
            /* ext-slot state (see rs_set_cap): coherent default even if the
             * slot could not be allocated */
            return mgl_compat_get_blend_advanced_coherent(ctx);
        case GL_FRAMEBUFFER_SRGB: RET_CAP(framebuffer_srgb);
        case GL_LINE_SMOOTH: RET_CAP(line_smooth);
        case GL_MULTISAMPLE: RET_CAP(multisample);
        case GL_POLYGON_SMOOTH: RET_CAP(polygon_smooth);
        case GL_POLYGON_OFFSET_FILL: RET_CAP(polygon_offset_fill);
        case GL_POLYGON_OFFSET_LINE: RET_CAP(polygon_offset_line);
        case GL_POLYGON_OFFSET_POINT: RET_CAP(polygon_offset_point);
        case GL_PROGRAM_POINT_SIZE: RET_CAP(program_point_size);
        case GL_PRIMITIVE_RESTART: RET_CAP(primitive_restart);
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: RET_CAP(primitive_restart_fixed_index);
        case GL_RASTERIZER_DISCARD: RET_CAP(rasterizer_discard);
        case GL_SAMPLE_ALPHA_TO_COVERAGE: RET_CAP(sample_alpha_to_coverage);
        case GL_SAMPLE_ALPHA_TO_ONE: RET_CAP(sample_alpha_to_one);
        case GL_SAMPLE_COVERAGE: RET_CAP(sample_coverage);
        case GL_SAMPLE_MASK: RET_CAP(sample_mask);
        case GL_SAMPLE_SHADING: RET_CAP(sample_shading);
        case GL_STENCIL_TEST: RET_CAP(stencil_test);
        case GL_TEXTURE_CUBE_MAP_SEAMLESS: RET_CAP(texture_cube_map_seamless);

        default:
            if (cap >= GL_CLIP_DISTANCE0 && cap < GL_CLIP_DISTANCE0 + MAX_CLIP_DISTANCES)
                return ctx->state.caps.clip_distances[cap - GL_CLIP_DISTANCE0];
            ERROR_RETURN_VALUE(GL_INVALID_ENUM, GL_FALSE);
    }

    return GL_FALSE;
}

/* indexed capabilities: BLEND (per draw buffer) and SCISSOR_TEST (per viewport) */
static bool rs_indexed_cap(GLMContext ctx, GLenum target, GLuint index, int set, GLboolean *out)
{
    switch (target) {
        case GL_BLEND:
            if (index >= MAX_COLOR_ATTACHMENTS) { ERROR_RETURN(GL_INVALID_VALUE); return false; }
            if (set >= 0) {
                RS()->blend[index] = set;
                ctx->state.caps.blend = RS()->blend[0];
                mgl_rs_dirty_pipeline(ctx);
            }
            if (out) *out = RS()->blend[index];
            return true;

        case GL_SCISSOR_TEST:
            if (index >= MGL_RSTATE_MAX_VIEWPORTS || index >= ctx->state.var.max_viewports) {
                ERROR_RETURN(GL_INVALID_VALUE); return false;
            }
            if (set >= 0) {
                RS()->scissor_test[index] = set;
                ctx->state.caps.scissor_test = RS()->scissor_test[0];
                mgl_rs_dirty_encoder(ctx);
            }
            if (out) *out = RS()->scissor_test[index];
            return true;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return false;
    }
}

void mglEnablei(GLMContext ctx, GLenum target, GLuint index)
{
    rs_indexed_cap(ctx, target, index, 1, NULL);
}

void mglDisablei(GLMContext ctx, GLenum target, GLuint index)
{
    rs_indexed_cap(ctx, target, index, 0, NULL);
}

GLboolean mglIsEnabledi(GLMContext ctx, GLenum target, GLuint index)
{
    GLboolean v = GL_FALSE;

    if (!rs_indexed_cap(ctx, target, index, -1, &v))
        return GL_FALSE;

    return v;
}

#pragma mark - raster state

void mglCullFace(GLMContext ctx, GLenum mode)
{
    switch(mode)
    {
        case GL_FRONT:
        case GL_BACK:
        case GL_FRONT_AND_BACK:
            ctx->state.var.cull_face_mode = mode;
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglFrontFace(GLMContext ctx, GLenum mode)
{
    switch(mode)
    {
        case GL_CW:
        case GL_CCW:
            ctx->state.var.front_face = mode;
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglHint(GLMContext ctx, GLenum target, GLenum mode)
{
    RS_CHECK(mode == GL_FASTEST || mode == GL_NICEST || mode == GL_DONT_CARE, GL_INVALID_ENUM);

    /* hints are recorded and queryable; Metal has no equivalent controls (derivative /
     * smoothing / compression quality are fixed by the driver) */
    switch(target)
    {
        case GL_LINE_SMOOTH_HINT: ctx->state.hints.line_smooth_hint = mode; break;
        case GL_POLYGON_SMOOTH_HINT: ctx->state.hints.polygon_smooth_hint = mode; break;
        case GL_TEXTURE_COMPRESSION_HINT: ctx->state.hints.texture_compression_hint = mode; break;
        case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: ctx->state.hints.fragment_shader_derivative_hint = mode; break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }
}

void mglLineWidth(GLMContext ctx, GLfloat width)
{
    RS_CHECK(width > 0.0f, GL_INVALID_VALUE);

    /* Metal rasterises lines 1 pixel wide only; the width is tracked and queryable
     * (ALIASED_LINE_WIDTH_RANGE reports [1,1]). */
    ctx->state.var.line_width = width;
}

void mglPointSize(GLMContext ctx, GLfloat size)
{
    RS_CHECK(size > 0.0f, GL_INVALID_VALUE);

    ctx->state.var.point_size = size;

    /* point size is a vertex function constant (used unless PROGRAM_POINT_SIZE) */
    mgl_rs_dirty_pipeline(ctx);
}

void mglPolygonMode(GLMContext ctx, GLenum face, GLenum mode)
{
    RS_CHECK(face == GL_FRONT_AND_BACK, GL_INVALID_ENUM);

    switch(mode)
    {
        case GL_POINT:
        case GL_LINE:
        case GL_FILL:
            ctx->state.var.polygon_mode = mode;
            RS()->polygon_mode = mode;
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglPolygonOffset(GLMContext ctx, GLfloat factor, GLfloat units)
{
    RS()->polygon_offset_factor = factor;
    RS()->polygon_offset_units = units;
    RS()->polygon_offset_clamp = 0.0f;
    ctx->state.var.polygon_offset_factor = factor;
    ctx->state.var.polygon_offset_units = units;

    mgl_rs_dirty_encoder(ctx);
}

#pragma mark - points

void mglPointParameterf(GLMContext ctx, GLenum pname, GLfloat param)
{
    switch (pname) {
        case GL_POINT_FADE_THRESHOLD_SIZE:
            RS_CHECK(param >= 0.0f, GL_INVALID_VALUE);
            /* only affects multisampled point rendering of points smaller than the threshold:
             * Metal points are not multisample-faded, the value is tracked/queryable */
            RS()->point_fade_threshold = param;
            break;

        case GL_POINT_SPRITE_COORD_ORIGIN:
            RS_CHECK(param == (GLfloat)GL_LOWER_LEFT || param == (GLfloat)GL_UPPER_LEFT, GL_INVALID_VALUE);
            RS()->point_sprite_coord_origin = (GLenum)param;
            mgl_rs_dirty_pipeline(ctx);     /* fragment function constant */
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }
}

void mglPointParameterfv(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    if (!params) return;
    mglPointParameterf(ctx, pname, params[0]);
}

void mglPointParameteri(GLMContext ctx, GLenum pname, GLint param)
{
    mglPointParameterf(ctx, pname, (GLfloat)param);
}

void mglPointParameteriv(GLMContext ctx, GLenum pname, const GLint *params)
{
    if (!params) return;
    mglPointParameterf(ctx, pname, (GLfloat)params[0]);
}

#pragma mark - scissor / viewport / depth range

void mglScissor(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height)
{
    RS_CHECK(width >= 0 && height >= 0, GL_INVALID_VALUE);

    mgl_rstate_set_all_scissors(ctx, x, y, width, height);
}

void mglViewport(GLMContext ctx, GLint x, GLint y, GLsizei width, GLsizei height)
{
    RS_CHECK(width >= 0 && height >= 0, GL_INVALID_VALUE);

    mgl_rstate_set_all_viewports(ctx, (GLfloat)x, (GLfloat)y, (GLfloat)width, (GLfloat)height);
}

void mglDepthRange(GLMContext ctx, GLdouble n, GLdouble f)
{
    mgl_rstate_set_all_depth_ranges(ctx, n, f);
}

void mglClearDepthf(GLMContext ctx, GLfloat d)
{
    if (d < 0.0f) d = 0.0f;
    if (d > 1.0f) d = 1.0f;
    ctx->state.var.depth_clear_value = d;
    ctx->state.dirty_bits |= DIRTY_STATE;
}

#pragma mark - logic op

void mglLogicOp(GLMContext ctx, GLenum opcode)
{
    switch(opcode)
    {
        case GL_CLEAR:
        case GL_SET:
        case GL_COPY:
        case GL_COPY_INVERTED:
        case GL_NOOP:
        case GL_INVERT:
        case GL_AND:
        case GL_NAND:
        case GL_OR:
        case GL_NOR:
        case GL_XOR:
        case GL_EQUIV:
        case GL_AND_REVERSE:
        case GL_AND_INVERTED:
        case GL_OR_REVERSE:
        case GL_OR_INVERTED:
            /* Metal has no framebuffer logic op: state is tracked and queryable, rendering
             * behaves as GL_COPY (see report) */
            ctx->state.var.logic_op = opcode;
            ctx->state.var.logic_op_mode = opcode;
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }
}

#pragma mark - stencil

static bool rs_valid_func(GLenum func)
{
    switch(func)
    {
        case GL_LEQUAL: case GL_GEQUAL: case GL_LESS: case GL_GREATER:
        case GL_EQUAL: case GL_NOTEQUAL: case GL_ALWAYS: case GL_NEVER:
            return true;
    }
    return false;
}

static bool validStencilOpSeparate(GLMContext ctx, GLenum op)
{
    switch(op)
    {
        case GL_KEEP:
        case GL_ZERO:
        case GL_REPLACE:
        case GL_INCR:
        case GL_INCR_WRAP:
        case GL_DECR:
        case GL_DECR_WRAP:
        case GL_INVERT:
            return true;
    }

    return false;
}

void mglStencilFuncSeparate(GLMContext ctx, GLenum face, GLenum func, GLint ref, GLuint mask)
{
    RS_CHECK(face == GL_FRONT || face == GL_BACK || face == GL_FRONT_AND_BACK, GL_INVALID_ENUM);
    RS_CHECK(rs_valid_func(func), GL_INVALID_ENUM);

    if (face != GL_BACK) {
        ctx->state.var.stencil_func = func;
        ctx->state.var.stencil_ref = ref;
        ctx->state.var.stencil_value_mask = mask;
    }
    if (face != GL_FRONT) {
        ctx->state.var.stencil_back_func = func;
        ctx->state.var.stencil_back_ref = ref;
        ctx->state.var.stencil_back_value_mask = mask;
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglStencilFunc(GLMContext ctx, GLenum func, GLint ref, GLuint mask)
{
    mglStencilFuncSeparate(ctx, GL_FRONT_AND_BACK, func, ref, mask);
}

void mglStencilOpSeparate(GLMContext ctx, GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass)
{
    RS_CHECK(face == GL_FRONT || face == GL_BACK || face == GL_FRONT_AND_BACK, GL_INVALID_ENUM);
    RS_CHECK(validStencilOpSeparate(ctx, sfail), GL_INVALID_ENUM);
    RS_CHECK(validStencilOpSeparate(ctx, dpfail), GL_INVALID_ENUM);
    RS_CHECK(validStencilOpSeparate(ctx, dppass), GL_INVALID_ENUM);

    if (face != GL_BACK) {
        ctx->state.var.stencil_fail = sfail;
        ctx->state.var.stencil_pass_depth_fail = dpfail;
        ctx->state.var.stencil_pass_depth_pass = dppass;
    }
    if (face != GL_FRONT) {
        ctx->state.var.stencil_back_fail = sfail;
        ctx->state.var.stencil_back_pass_depth_fail = dpfail;
        ctx->state.var.stencil_back_pass_depth_pass = dppass;
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglStencilOp(GLMContext ctx, GLenum fail, GLenum zfail, GLenum zpass)
{
    mglStencilOpSeparate(ctx, GL_FRONT_AND_BACK, fail, zfail, zpass);
}

void mglStencilMaskSeparate(GLMContext ctx, GLenum face, GLuint mask)
{
    RS_CHECK(face == GL_FRONT || face == GL_BACK || face == GL_FRONT_AND_BACK, GL_INVALID_ENUM);

    if (face != GL_BACK)  ctx->state.var.stencil_writemask = mask;
    if (face != GL_FRONT) ctx->state.var.stencil_back_writemask = mask;

    mgl_rs_dirty_encoder(ctx);
}

void mglStencilMask(GLMContext ctx, GLuint mask)
{
    mglStencilMaskSeparate(ctx, GL_FRONT_AND_BACK, mask);
}

#pragma mark - color / depth masks, depth func

void mglColorMask(GLMContext ctx, GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha)
{
    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++)
        mglColorMaski(ctx, i, red, green, blue, alpha);
}

void mglDepthMask(GLMContext ctx, GLboolean flag)
{
    ctx->state.var.depth_writemask = flag ? GL_TRUE : GL_FALSE;

    mgl_rs_dirty_encoder(ctx);
}

void mglDepthFunc(GLMContext ctx, GLenum func)
{
    RS_CHECK(rs_valid_func(func), GL_INVALID_ENUM);

    ctx->state.var.depth_func = func;

    mgl_rs_dirty_encoder(ctx);
}

#pragma mark - blending

void mglBlendColor(GLMContext ctx, GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha)
{
    ctx->state.var.blend_color[0] = red;
    ctx->state.var.blend_color[1] = green;
    ctx->state.var.blend_color[2] = blue;
    ctx->state.var.blend_color[3] = alpha;

    mgl_rs_dirty_encoder(ctx);     /* -setBlendColorRed:green:blue:alpha: */
}

static bool rs_valid_equation(GLenum mode)
{
    switch(mode)
    {
        case GL_FUNC_ADD: case GL_FUNC_SUBTRACT: case GL_FUNC_REVERSE_SUBTRACT:
        case GL_MIN: case GL_MAX:
            return true;
    }
    return false;
}

static bool rs_valid_factor(GLenum f)
{
    switch(f)
    {
        case GL_ZERO: case GL_ONE:
        case GL_SRC_COLOR: case GL_ONE_MINUS_SRC_COLOR:
        case GL_DST_COLOR: case GL_ONE_MINUS_DST_COLOR:
        case GL_SRC_ALPHA: case GL_ONE_MINUS_SRC_ALPHA:
        case GL_DST_ALPHA: case GL_ONE_MINUS_DST_ALPHA:
        case GL_CONSTANT_COLOR: case GL_ONE_MINUS_CONSTANT_COLOR:
        case GL_CONSTANT_ALPHA: case GL_ONE_MINUS_CONSTANT_ALPHA:
        case GL_SRC_ALPHA_SATURATE:
        case GL_SRC1_COLOR: case GL_ONE_MINUS_SRC1_COLOR:
        case GL_SRC1_ALPHA: case GL_ONE_MINUS_SRC1_ALPHA:
            return true;
    }
    return false;
}

static void rs_blend_equation(GLMContext ctx, GLuint first, GLuint last, GLenum rgb, GLenum alpha)
{
    for (GLuint i = first; i <= last; i++) {
        ctx->state.var.blend_equation_rgb[i] = rgb;
        ctx->state.var.blend_equation_alpha[i] = alpha;
    }
    mgl_rs_dirty_pipeline(ctx);
}

static void rs_blend_func(GLMContext ctx, GLuint first, GLuint last, GLenum srgb, GLenum drgb, GLenum sa, GLenum da)
{
    for (GLuint i = first; i <= last; i++) {
        ctx->state.var.blend_src_rgb[i] = srgb;
        ctx->state.var.blend_dst_rgb[i] = drgb;
        ctx->state.var.blend_src_alpha[i] = sa;
        ctx->state.var.blend_dst_alpha[i] = da;
    }
    mgl_rs_dirty_pipeline(ctx);
}

void mglBlendEquationSeparate(GLMContext ctx, GLenum modeRGB, GLenum modeAlpha)
{
    RS_CHECK(rs_valid_equation(modeRGB) && rs_valid_equation(modeAlpha), GL_INVALID_ENUM);
    rs_blend_equation(ctx, 0, MAX_COLOR_ATTACHMENTS - 1, modeRGB, modeAlpha);
}

void mglBlendEquation(GLMContext ctx, GLenum mode)
{
    /* compat (GL_KHR_blend_equation_advanced, GL 4.6 §17.3.8): the 15 advanced
     * equations are accepted by glBlendEquation (and glBlendEquationi) ONLY —
     * the Separate variants reject them with GL_INVALID_ENUM.  The pair is
     * installed as RGB+A together; the compositing itself runs in a rewritten
     * fragment shader (mgl_advblend.c), draw buffer 0 */
    if (mgl_advblend_mode(mode) >= 0)
    {
        rs_blend_equation(ctx, 0, MAX_COLOR_ATTACHMENTS - 1, mode, mode);
        mgl_advblend_armed(ctx);      /* lazy pre-draw hook + seen flag */
        return;
    }
    mglBlendEquationSeparate(ctx, mode, mode);
}

void mglBlendEquationSeparatei(GLMContext ctx, GLuint buf, GLenum modeRGB, GLenum modeAlpha)
{
    RS_CHECK(buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);
    RS_CHECK(rs_valid_equation(modeRGB) && rs_valid_equation(modeAlpha), GL_INVALID_ENUM);
    rs_blend_equation(ctx, buf, buf, modeRGB, modeAlpha);
}

void mglBlendEquationi(GLMContext ctx, GLuint buf, GLenum mode)
{
    RS_CHECK(buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);
    if (mgl_advblend_mode(mode) >= 0)   /* advanced equations: this path only */
    {
        rs_blend_equation(ctx, buf, buf, mode, mode);
        mgl_advblend_armed(ctx);      /* lazy pre-draw hook + seen flag */
        return;
    }
    mglBlendEquationSeparatei(ctx, buf, mode, mode);
}

void mglBlendFuncSeparate(GLMContext ctx, GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha)
{
    RS_CHECK(rs_valid_factor(sfactorRGB) && rs_valid_factor(dfactorRGB) &&
             rs_valid_factor(sfactorAlpha) && rs_valid_factor(dfactorAlpha), GL_INVALID_ENUM);
    rs_blend_func(ctx, 0, MAX_COLOR_ATTACHMENTS - 1, sfactorRGB, dfactorRGB, sfactorAlpha, dfactorAlpha);
}

void mglBlendFunc(GLMContext ctx, GLenum sfactor, GLenum dfactor)
{
    mglBlendFuncSeparate(ctx, sfactor, dfactor, sfactor, dfactor);
}

void mglBlendFuncSeparatei(GLMContext ctx, GLuint buf, GLenum srcRGB, GLenum dstRGB, GLenum srcAlpha, GLenum dstAlpha)
{
    RS_CHECK(buf < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);
    RS_CHECK(rs_valid_factor(srcRGB) && rs_valid_factor(dstRGB) &&
             rs_valid_factor(srcAlpha) && rs_valid_factor(dstAlpha), GL_INVALID_ENUM);
    rs_blend_func(ctx, buf, buf, srcRGB, dstRGB, srcAlpha, dstAlpha);
}

void mglBlendFunci(GLMContext ctx, GLuint buf, GLenum sfactor, GLenum dfactor)
{
    mglBlendFuncSeparatei(ctx, buf, sfactor, dfactor, sfactor, dfactor);
}
