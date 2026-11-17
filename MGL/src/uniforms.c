/*
 * Copyright (C) Michael Larson on on 1/6/25.
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
 * uniforms.c
 * MGL
 *
 * Default-block uniform storage and the uniform / uniform-block queries.
 *
 * Every linked program owns a CPU shadow of each default-block uniform (MGLIUniform.shadow,
 * GL-natural column-major, doubles kept as doubles) and, per shader stage, one Buffer holding
 * the stage's packed "mgl_DefaultUniforms" block (see introspect_spirv.c).  glUniform* and
 * glProgramUniform* share mgli_set_uniform(), which validates per GL 4.6 section 7.6.1,
 * updates the shadow and writes the converted values into every stage's block.  Opaque
 * uniforms (samplers / images) store the unit and re-point the program's resource-list binding
 * so the renderer binds the selected unit.
 *
 * Double precision: Metal has no fp64.  dvec/dmat values are stored and returned as doubles;
 * the GPU-side copy uses the 8-byte layout of the SPIR-V type (programs using fp64 only run if
 * the shader translation lowers doubles).
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "spirv_cross_c.h"

#include "shaders.h"
#include "programs.h"
#include "mgl_ext_buffers.h"
#include "cts_glsl_compat.h"
#include "buffers.h"
#include "glm_context.h"
#include "mgl_ext_introspect.h"
#include "mgl_ext_program.h"

#pragma mark - storage

static int comp_bytes(const MGLIUniform *u) { return u->kind == MGLI_K_DOUBLE ? 8 : 4; }

/* r170: getenv scans the environment; it ran once per glUniform* and once per copied
 * component */
static int fp64_dump_on(void)
{
    static int on = -1;
    if (on < 0) on = getenv("MGL_FP64_DUMP") != NULL;
    return on;
}

void mgli_upload_uniform(GLMContext ctx, MGLIProgram *ip, MGLIUniform *u)
{
    if (fp64_dump_on()) { FILE*d=fopen("/tmp/fp64uni.txt","a"); if (d) { fprintf(d, "[uni] %s present[f]=%d scalar=%d off=%d\n", u->name?u->name:"?", u->st[4].present, u->st[4].scalar_size, u->st[4].offset); fclose(d);} }
    if (!u->shadow) return;
    u->uploaded = 1;   /* r170: the shadow has reached the program's storage at least once */
    Program *p = ip->prog;
    for (int s = 0; s < MGLI_STAGES; s++) {
        if (!u->st[s].present) continue;
        if (u->opaque_class) {
            if (!p) continue;
            SpirvResourceList *l = &p->spirv_resources_list[s][u->st[s].res_type];
            for (GLuint k = 0; k < l->count; k++)
                if (l->list[k]._id == u->st[s].spv_id)
                {
                    if (u->st[s].elem_base == 0)
                        l->list[k].binding = ((uint32_t *)u->shadow)[0];

                    /* geom: image/sampler ARRAY elements are independent uniforms —
                     * remember every element's unit (base+e is only the default) so the
                     * renderers' array walks can honour strided units
                     * (shader_image_load_store.advanced-sso-simple binds g_image[0..3]
                     * to units 0,2,4,6).  The pipeline composite shares this array.
                     * The unit table spans the WHOLE Metal array — an opaque
                     * arrays-of-arrays / hoisted struct member flattens to one array of
                     * the total element count (geom_opaque_flatten.c) while the GL view
                     * reports one resource per outer row (two uniforms "u.i[0].a[0]"
                     * and "u.i[1].a[0]" share one 4-slot array), so each GL element
                     * lands at its FLAT index (st[].elem_base + e).  array_size stays
                     * the REFLECTED MSL array length (0 = scalar resource: the image
                     * binders key on that) — never grown here. */
                    {
                        GLuint need = (GLuint)(u->st[s].elem_base + u->size);
                        if (!l->list[k].unit_vals)
                        {
                            GLuint n = l->list[k].array_size > need ? l->list[k].array_size : need;
                            l->list[k].unit_vals = (GLuint *)calloc((size_t)n, sizeof(GLuint));
                        }
                        else if (l->list[k].array_size && need > l->list[k].array_size)
                        {
                            /* a GL view wider than the reflected array (cannot happen
                             * for a flattened module, kept for safety): grow, keeping
                             * the elements already recorded */
                            GLuint *nv = (GLuint *)calloc((size_t)need, sizeof(GLuint));
                            if (nv)
                            {
                                memcpy(nv, l->list[k].unit_vals,
                                       (size_t)l->list[k].array_size * sizeof(GLuint));
                                free(l->list[k].unit_vals);
                                l->list[k].unit_vals = nv;
                            }
                        }
                        if (l->list[k].unit_vals)
                            for (int e = 0; e < u->size; e++)
                                l->list[k].unit_vals[u->st[s].elem_base + e] = ((uint32_t *)u->shadow)[e];
                    }
                }
            ctx->state.dirty_bits |= DIRTY_TEX_BINDING | DIRTY_IMAGE_UNIT_STATE;
            ctx->state.sampler_map_gen++;      /* r158: see glm_context.h */
            continue;
        }
        Buffer *b = ip->dub[s];
        if (!b || !b->data.buffer_data) continue;
        /* buffer_data aliases a managed MTLBuffer the last draw may still be
         * reading: order this CPU write against it (rename or wait) per the
         * 'buffers' domain memory model */
        mgl_buffers_prepare_cpu_write(ctx, b, true);
        uint8_t *dst = (uint8_t *)b->data.buffer_data;
        const uint8_t *src = u->shadow;
        for (int e = 0; e < u->size; e++)
            for (int c = 0; c < u->cols; c++)
                for (int r = 0; r < u->rows; r++) {
                    size_t si = ((size_t)e * u->cols * u->rows + (size_t)c * u->rows + r) * comp_bytes(u);
                    size_t di = u->st[s].offset + (size_t)e * u->st[s].array_stride + (size_t)c * u->st[s].matrix_stride +
                                (size_t)r * u->st[s].scalar_size;
                    if (di + u->st[s].scalar_size > ip->dub_size[s]) continue;
                    if (fp64_dump_on()) { FILE*d=fopen("/tmp/fp64uni.txt","a"); if (d) { fprintf(d, "[cpy] name=%s off=%d scalar=%d\n", u->name, u->st[s].offset, u->st[s].scalar_size); fclose(d);} }
                    if (u->kind == MGLI_K_DOUBLE && u->st[s].scalar_size == 4) {
                        /* GPU stores fp32 (Metal has no fp64): convert from the double shadow */
                        float f = (float)*(const double *)(src + si);
                        memcpy(dst + di, &f, 4);
                    } else if (u->st[s].scalar_size == 1) {
                        dst[di] = (*(const uint32_t *)(src + si)) ? 1 : 0;
                    } else {
                        memcpy(dst + di, src + si, u->st[s].scalar_size);
                    }
                }
        b->data.dirty_bits |= DIRTY_BUFFER_DATA | DIRTY_BUFFER_UNSYNCED;   /* r172 */
    }
}

static bool type_accepts(const MGLIUniform *u, int set_kind, int cols, int rows, bool matrix_call)
{
    if (matrix_call) {
        if (u->cols != cols || u->rows != rows || u->cols < 2) return false;
        return set_kind == MGLI_SET_D ? u->kind == MGLI_K_DOUBLE : u->kind == MGLI_K_FLOAT;
    }
    if (u->cols != 1 || u->rows != rows) return false;
    if (u->opaque_class) return set_kind == MGLI_SET_I && rows == 1;
    switch (u->kind) {
        case MGLI_K_FLOAT:  return set_kind == MGLI_SET_F;
        case MGLI_K_DOUBLE: return set_kind == MGLI_SET_D;
        case MGLI_K_INT:    return set_kind == MGLI_SET_I;
        case MGLI_K_UINT:   return set_kind == MGLI_SET_UI;
        case MGLI_K_BOOL:   return set_kind == MGLI_SET_F || set_kind == MGLI_SET_I || set_kind == MGLI_SET_UI;
        default: return false;
    }
}

void mgli_set_uniform(GLMContext ctx, const char *func, MGLIProgram *ip, GLint location, GLsizei count,
                      int set_kind, int cols, int rows, GLboolean transpose, const void *data)
{
    bool matrix_call = cols > 1;
    if (count < 0) { mgli_set_error(ctx, func, GL_INVALID_VALUE); return; }
    if (location == -1) return;
    if (!ip || !ip->linked) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    if (location < -1 || location >= ip->nlocs || ip->locs[location].uniform < 0) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    MGLIUniform *u = &ip->uniforms[ip->locs[location].uniform];
    int elem = ip->locs[location].element;
    if (!type_accepts(u, set_kind, cols, rows, matrix_call)) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    if (count > 1 && u->size == 1) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    if (count == 0 || !data) return;
    if (count > u->size - elem) count = u->size - elem;
    int comps = cols * rows;
    /* r170: the value range this call writes, to tell a redundant write.  The
     * upload re-wrote the program's default-uniform block (a CPU write that may
     * rename storage the GPU still reads) and, for a sampler / image uniform,
     * bumped sampler_map_gen, which forces the full texture bind walk.  A D3D9
     * front end re-sends unchanged uniforms (sampler units after every program
     * switch).  MGL_UNIFORM_SKIP_SAME=0: always upload. */
    static int skip_same = -1;
    if (skip_same < 0) { const char *e = getenv("MGL_UNIFORM_SKIP_SAME"); skip_same = (e && e[0] == '0') ? 0 : 1; }
    size_t cb_ = (size_t)comp_bytes(u);
    size_t r_off = (size_t)elem * (size_t)comps * cb_;
    size_t r_len = (size_t)count * (size_t)comps * cb_;
    uint8_t before_buf[256];
    uint8_t *before = NULL;
    if (skip_same && u->shadow)
    {
        before = r_len <= sizeof before_buf ? before_buf : (uint8_t *)malloc(r_len);
        if (before) memcpy(before, u->shadow + r_off, r_len);
    }
    if (u->opaque_class) {
        const GLint *v = data;
        /* image-unit values are bounded by GL_MAX_IMAGE_UNITS (16 — get.c
         * MGL_MAX_IMAGE_UNITS / glm_limits.h), not the sampler-unit count */
        GLint max_units = u->opaque_class == 2 ? MGL_IMAGE_UNITS_MAX : TEXTURE_UNITS;
        if (ctx->state.var.max_combined_texture_image_units > 0 && u->opaque_class == 1) max_units = ctx->state.var.max_combined_texture_image_units;
        for (GLsizei i = 0; i < count; i++) if (v[i] < 0 || v[i] >= max_units) { mgli_set_error(ctx, func, GL_INVALID_VALUE); if (before && before != before_buf) free(before); return; }
    }
    for (GLsizei i = 0; i < count; i++) {
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++) {
                int src_index = (int)i * comps + (transpose && matrix_call ? r * cols + c : c * rows + r);
                size_t dst = ((size_t)(elem + i) * comps + (size_t)c * rows + r);
                double d = 0; int64_t iv = 0; bool is_int = false;
                switch (set_kind) {
                    case MGLI_SET_F: d = ((const GLfloat *)data)[src_index]; break;
                    case MGLI_SET_D: d = ((const GLdouble *)data)[src_index]; break;
                    case MGLI_SET_I: iv = ((const GLint *)data)[src_index]; is_int = true; break;
                    case MGLI_SET_UI: iv = ((const GLuint *)data)[src_index]; is_int = true; break;
                }
                switch (u->kind) {
                    case MGLI_K_FLOAT: { GLfloat f = (GLfloat)d; memcpy(u->shadow + dst * 4, &f, 4); break; }
                    case MGLI_K_DOUBLE: memcpy(u->shadow + dst * 8, &d, 8); break;
                    case MGLI_K_INT: case MGLI_K_UINT: { uint32_t x = (uint32_t)iv; memcpy(u->shadow + dst * 4, &x, 4); break; }
                    case MGLI_K_BOOL: { uint32_t x = is_int ? (iv != 0) : (d != 0.0); memcpy(u->shadow + dst * 4, &x, 4); break; }
                    default: break;
                }
            }
    }
    if (before)
    {
        bool same = memcmp(before, u->shadow + r_off, r_len) == 0;
        if (before != before_buf) free(before);
        if (same && u->uploaded)
            return;                     /* r170: nothing changed — no upload, no dirt */
    }
    mgli_upload_uniform(ctx, ip, u);
}

void mgli_get_uniform(GLMContext ctx, const char *func, GLuint program, GLint location, GLsizei bufSize,
                      int out_kind, void *params)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, true);
    if (!ip) return;
    if (location < 0 || location >= ip->nlocs || ip->locs[location].uniform < 0) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    MGLIUniform *u = &ip->uniforms[ip->locs[location].uniform];
    int elem = ip->locs[location].element;
    int comps = u->cols * u->rows;
    size_t osz = out_kind == MGLI_SET_D ? 8 : 4;
    if (bufSize >= 0 && (size_t)bufSize < comps * osz) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return; }
    if (!params || !u->shadow) return;
    for (int k = 0; k < comps; k++) {
        size_t si = (size_t)elem * comps + k;
        double d = 0; int64_t iv = 0; bool is_int = true;
        switch (u->kind) {
            case MGLI_K_FLOAT: { GLfloat f; memcpy(&f, u->shadow + si * 4, 4); d = f; is_int = false; break; }
            case MGLI_K_DOUBLE: memcpy(&d, u->shadow + si * 8, 8); is_int = false; break;
            case MGLI_K_INT: { int32_t x; memcpy(&x, u->shadow + si * 4, 4); iv = x; break; }
            case MGLI_K_UINT: case MGLI_K_BOOL: { uint32_t x; memcpy(&x, u->shadow + si * 4, 4); iv = x; break; }
            default: break;
        }
        switch (out_kind) {
            case MGLI_SET_F: ((GLfloat *)params)[k] = is_int ? (GLfloat)iv : (GLfloat)d; break;
            case MGLI_SET_D: ((GLdouble *)params)[k] = is_int ? (GLdouble)iv : d; break;
            case MGLI_SET_I: ((GLint *)params)[k] = is_int ? (GLint)iv : (GLint)lround(d); break;
            /* a negative float/double converts to 0, not the wrapped two's-complement of
             * the rounded signed value (the CTS state_query expectations pin this) */
            case MGLI_SET_UI: ((GLuint *)params)[k] = is_int ? (GLuint)iv : (d > 0.0 ? (GLuint)llround(d) : 0u); break;
        }
    }
}

static MGLIProgram *current_program(GLMContext ctx, const char *func)
{
    Program *p = mgl_program_uniform_target(ctx); /* current program, else the pipeline's active program */
    if (!p) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return NULL; }
    MGLIProgram *ip = NULL;
    for (MGLIProgram *it = ctx->state.ext_introspect ? ctx->state.ext_introspect->programs : NULL; it; it = it->next)
        if (it->prog == p) { ip = it; break; }
    if (!ip || !ip->linked) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return NULL; }
    return ip;
}

#pragma mark uniforms

int mgli_uniform_index(MGLIProgram *ip, const char *name)
{
    if (!ip || !name) return -1;
    for (int i = 0; i < ip->nuniforms; i++) {
        MGLIUniform *u = &ip->uniforms[i];
        if (u->hidden) continue;
        if (!strcmp(u->name, name)) return i;
        size_t ln = strlen(name);
        /* "arr" matches "arr[0]" */
        if (!strncmp(u->name, name, ln) && !strcmp(u->name + ln, "[0]")) return i;
    }
    /* hoisted sampler/image struct members (see mgli_uniform_location);
     * a miss here must NOT stop the search: dotted NON-sampler paths
     * ("u1.m1[1]" of a struct array member) still resolve below */
    if (strchr(name, '.')) {
        char hoisted[256];
        if (mgl_hoist_lookup_name(name, hoisted, sizeof hoisted)) {
            int hi = mgli_uniform_index(ip, hoisted);
            if (hi >= 0) return hi;
        }
    }
    return -1;
}

/* GL sampler/image ARRAY elements are independent uniforms: element e samples
 * the unit the app stored on THAT element (glUniform1i on arr[e]), which need not
 * be base+e.  The renderer's texture binder (MGLRenderer.m) consults this for
 * array elements > 0; `def` (the resource's base+e convention) is returned when
 * the program's uniform table has no entry for that element. */
GLuint mgli_sampler_elem_unit(GLMContext ctx, Program *p, int stage, uint32_t spv_id,
                              GLuint e, GLuint def)
{
    if (!ctx || !p) return def;
    struct MGLExtState_introspect *ist = ctx->state.ext_introspect;
    if (!ist) return def;
    for (MGLIProgram *ip = ist->programs; ip; ip = ip->next) {
        if (ip->prog != p) continue;
        for (int i = 0; i < ip->nuniforms; i++) {
            MGLIUniform *u = &ip->uniforms[i];
            if (!u->opaque_class || u->opaque_class == 3) continue;
            if ((GLuint)u->st[stage].spv_id != spv_id) continue;
            /* the whole array is one uniform: elem_base is its first element and
             * shadow[] holds each element's unit in order */
            if (e < (GLuint)u->st[stage].elem_base ||
                e >= (GLuint)u->st[stage].elem_base + (GLuint)u->size) continue;
            if (!u->shadow) continue;
            GLint v;
            memcpy(&v, u->shadow + (e - (GLuint)u->st[stage].elem_base) * 4, sizeof v);
            return (GLuint)v;
        }
        return def;
    }
        return def;
}

GLint mgli_uniform_location(MGLIProgram *ip, const char *name)
{
    if (!ip || !name || !strncmp(name, "gl_", 3)) return -1;
    int i = mgli_uniform_index(ip, name);
    if (i >= 0) return ip->uniforms[i].location;
    /* sampler/image members hoisted out of default-block structs
     * (mgl_hoist_struct_samplers) live under the joined name: "s.c" ->
     * "s_mglhoist_c", "s[0].c" -> "s_mglhoist_c[0]".  Only a hit short-circuits:
     * dotted struct-array element lookups ("u1.m1[1]") continue to the
     * array-index fallback below. */
    if (strchr(name, '.')) {
        char hoisted[256];
        if (mgl_hoist_lookup_name(name, hoisted, sizeof hoisted)) {
            GLint hl = mgli_uniform_location(ip, hoisted);
            if (hl >= 0) return hl;
        }
    }
    /* "arr[N]" (also "a[i][N]") -> base + N.  The index must be a canonical decimal
     * integer: no whitespace, no sign, no leading zeros (CTS program_interface_query
     * array-names: "a[ 0]", "a[01]", "a[0+0]" are NOT valid uniform names) */
    size_t ln = strlen(name);
    if (ln > 3 && name[ln - 1] == ']') {
        const char *lb = strrchr(name, '[');
        if (!lb || lb == name) return -1;
        size_t il = (size_t)(&name[ln - 1] - (lb + 1));
        long idx = -1;
        if (il == 0 || il > 9) return -1;
        for (size_t q = 0; q < il; q++) {
            if (lb[1 + q] < '0' || lb[1 + q] > '9') return -1;
            if (q == 0 && lb[1] == '0' && il > 1) return -1;   /* "00", "01" */
        }
        idx = lb[1] - '0';
        for (size_t q = 1; q < il; q++) idx = idx * 10 + (lb[1 + q] - '0');
        char base[512];
        size_t bl = (size_t)(lb - name);
        if (bl + 4 > sizeof base) return -1;
        memcpy(base, name, bl); strcpy(base + bl, "[0]");
        for (int k = 0; k < ip->nuniforms; k++) {
            MGLIUniform *u = &ip->uniforms[k];
            if (u->hidden || strcmp(u->name, base)) continue;
            if (u->location < 0 || idx >= u->size) return -1;
            return u->location + (GLint)idx;
        }
    }
    return -1;
}

GLint mglGetUniformLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, true);
    if (!ip) return -1;
    return mgli_uniform_location(ip, name);
}

void mglGetUniformfv(GLMContext ctx, GLuint program, GLint location, GLfloat *params)
{
    mgli_get_uniform(ctx, __FUNCTION__, program, location, -1, MGLI_SET_F, params);
}

void mglGetUniformiv(GLMContext ctx, GLuint program, GLint location, GLint *params)
{
    mgli_get_uniform(ctx, __FUNCTION__, program, location, -1, MGLI_SET_I, params);
}

void mglGetUniformIndices(GLMContext ctx, GLuint program, GLsizei uniformCount, const GLchar *const*uniformNames, GLuint *uniformIndices)
{
    if (uniformCount < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    for (GLsizei i = 0; i < uniformCount; i++) {
        int idx = (ip->linked && uniformNames && uniformNames[i]) ? mgli_uniform_index(ip, uniformNames[i]) : -1;
        if (uniformIndices) uniformIndices[i] = idx >= 0 ? (GLuint)idx : GL_INVALID_INDEX;
    }
}

/* public uniform index -> internal (hidden uniforms are skipped from the public list) */
static MGLIUniform *public_uniform(MGLIProgram *ip, GLuint index)
{
    GLuint n = 0;
    for (int i = 0; i < ip->nuniforms; i++) {
        if (ip->uniforms[i].hidden) continue;
        if (n == index) return &ip->uniforms[i];
        n++;
    }
    return NULL;
}

static GLuint public_uniform_count(MGLIProgram *ip)
{
    GLuint n = 0;
    for (int i = 0; i < ip->nuniforms; i++) if (!ip->uniforms[i].hidden) n++;
    return n;
}

void mglGetActiveUniformsiv(GLMContext ctx, GLuint program, GLsizei uniformCount, const GLuint *uniformIndices, GLenum pname, GLint *params)
{
    if (uniformCount < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    switch (pname) {
        case GL_UNIFORM_TYPE: case GL_UNIFORM_SIZE: case GL_UNIFORM_NAME_LENGTH: case GL_UNIFORM_BLOCK_INDEX:
        case GL_UNIFORM_OFFSET: case GL_UNIFORM_ARRAY_STRIDE: case GL_UNIFORM_MATRIX_STRIDE: case GL_UNIFORM_IS_ROW_MAJOR:
        case GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX: break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return;
    }
    GLuint n = public_uniform_count(ip);
    for (GLsizei i = 0; i < uniformCount; i++)
        if (!uniformIndices || uniformIndices[i] >= n) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    for (GLsizei i = 0; i < uniformCount && params; i++) {
        MGLIUniform *u = public_uniform(ip, uniformIndices[i]);
        switch (pname) {
            case GL_UNIFORM_TYPE: params[i] = (GLint)u->type; break;
            case GL_UNIFORM_SIZE: params[i] = u->size; break;
            case GL_UNIFORM_NAME_LENGTH: params[i] = (GLint)strlen(u->name) + 1; break;
            case GL_UNIFORM_BLOCK_INDEX: params[i] = u->block_index; break;
            case GL_UNIFORM_OFFSET: params[i] = u->offset; break;
            case GL_UNIFORM_ARRAY_STRIDE: params[i] = u->array_stride; break;
            case GL_UNIFORM_MATRIX_STRIDE: params[i] = u->matrix_stride; break;
            case GL_UNIFORM_IS_ROW_MAJOR: params[i] = u->is_row_major; break;
            case GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX: params[i] = u->atomic_buffer_index; break;
        }
    }
}

static void copy_name(const char *src, GLsizei bufSize, GLsizei *length, GLchar *dst)
{
    GLsizei n = 0;
    if (dst && bufSize > 0) {
        n = (GLsizei)strlen(src);
        if (n > bufSize - 1) n = bufSize - 1;
        memcpy(dst, src, (size_t)n);
        dst[n] = 0;
    }
    if (length) *length = n;
}

void mglGetActiveUniformName(GLMContext ctx, GLuint program, GLuint uniformIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformName)
{
    if (bufSize < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    MGLIUniform *u = public_uniform(ip, uniformIndex);
    if (!u) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    copy_name(u->name, bufSize, length, uniformName);
}

void mgli_get_active_uniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (bufSize < 0) { mgli_set_error(ctx, "mglGetActiveUniform", GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    MGLIUniform *u = public_uniform(ip, index);
    if (!u) { mgli_set_error(ctx, "mglGetActiveUniform", GL_INVALID_VALUE); return; }
    copy_name(u->name, bufSize, length, name);
    if (size) *size = u->size;
    if (type) *type = u->type;
}

#pragma mark uniform blocks

GLuint  mglGetUniformBlockIndex(GLMContext ctx, GLuint program, const GLchar *uniformBlockName)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip || !ip->linked || !uniformBlockName) return GL_INVALID_INDEX;
    for (int i = 0; i < ip->nubos; i++) {
        if (!strcmp(ip->ubos[i].name, uniformBlockName)) return (GLuint)i;
        /* "Block" names element 0 of a block array */
        size_t ln = strlen(uniformBlockName);
        if (ip->ubos[i].array_element == 0 && !strncmp(ip->ubos[i].name, uniformBlockName, ln) && !strcmp(ip->ubos[i].name + ln, "[0]"))
            return (GLuint)i;
    }
    return GL_INVALID_INDEX;
}

void mglGetActiveUniformBlockiv(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLenum pname, GLint *params)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (uniformBlockIndex >= (GLuint)ip->nubos) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIBlock *b = &ip->ubos[uniformBlockIndex];
    if (!params) return;
    switch (pname) {
        case GL_UNIFORM_BLOCK_BINDING: *params = b->binding; break;
        case GL_UNIFORM_BLOCK_DATA_SIZE: *params = b->data_size; break;
        case GL_UNIFORM_BLOCK_NAME_LENGTH: *params = (GLint)strlen(b->name) + 1; break;
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: *params = b->member_count; break;
        case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES:
            for (int i = 0; i < b->member_count; i++) {
                /* internal -> public index */
                int pub = 0;
                for (int k = 0; k < b->members[i]; k++) if (!ip->uniforms[k].hidden) pub++;
                params[i] = pub;
            }
            break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *params = (b->stages >> _VERTEX_SHADER) & 1; break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER: *params = (b->stages >> _TESS_CONTROL_SHADER) & 1; break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER: *params = (b->stages >> _TESS_EVALUATION_SHADER) & 1; break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER: *params = (b->stages >> _GEOMETRY_SHADER) & 1; break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *params = (b->stages >> _FRAGMENT_SHADER) & 1; break;
        case GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER: *params = (b->stages >> _COMPUTE_SHADER) & 1; break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); break;
    }
}

void mglGetActiveUniformBlockName(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLsizei bufSize, GLsizei *length, GLchar *uniformBlockName)
{
    if (bufSize < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (uniformBlockIndex >= (GLuint)ip->nubos) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    copy_name(ip->ubos[uniformBlockIndex].name, bufSize, length, uniformBlockName);
}

/* re-point the block's SPIR-V resources so the renderer's buffer map pulls the new binding point */
void mgli_apply_block_binding(GLMContext ctx, MGLIProgram *ip, MGLIBlock *b, int res_type)
{
    Program *p = ip->prog;
    if (!p) return;
    for (int s = 0; s < MGLI_STAGES; s++) {
        if (!b->spv_id[s]) continue;
        SpirvResourceList *l = &p->spirv_resources_list[s][res_type];
        for (GLuint k = 0; k < l->count; k++)
        {
            /* instance arrays carry one resource entry per instance; match it by the
             * instance index recorded in `location` (see compile_stage in program.c) */
            if (l->list[k]._id != b->spv_id[s] ||
                l->list[k].location != (GLuint)(b->array_element > 0 ? b->array_element : 0))
                continue;
            l->list[k].binding = (GLuint)b->binding;
        }
    }
    ctx->state.dirty_bits |= DIRTY_BUFFER_BASE_STATE; ctx->state.buffer_base_gen++;   /* r169 */
}

void mglUniformBlockBinding(GLMContext ctx, GLuint program, GLuint uniformBlockIndex, GLuint uniformBlockBinding)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (uniformBlockIndex >= (GLuint)ip->nubos) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    /* GL_MAX_UNIFORM_BUFFER_BINDINGS is reported as >= 84 and the renderer resolves
     * every point (buffer_base rows for < MAX_BINDABLE_BUFFERS, the overflow table
     * above), so no renderer-table clamp here: KHR-GL46.layout_binding block
     * binding_api_update re-points blocks at sparse points up to 83 and draws. */
    /* r159: the reported GL_MAX_UNIFORM_BUFFER_BINDINGS (compat r111: 512), as in
     * get.c / buffers.c — the cached macOS default (84) rejected points the GL
     * reports as valid (KHR-GL46.layout_binding block_layout_binding_block_*:
     * glUniformBlockBinding(..., 511)) */
    GLuint max = ctx->state.var.max_uniform_buffer_bindings > MGL_MAX_UNIFORM_BUFFER_BINDINGS
                     ? (GLuint)ctx->state.var.max_uniform_buffer_bindings : MGL_MAX_UNIFORM_BUFFER_BINDINGS;
    if (uniformBlockBinding >= max) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIBlock *b = &ip->ubos[uniformBlockIndex];
    b->binding = (GLint)uniformBlockBinding;
    mgli_apply_block_binding(ctx, ip, b, SPVC_RESOURCE_TYPE_UNIFORM_BUFFER);
}

bool checkUniformParams(GLMContext ctx, GLint location)
{
    Program* ptr = mgl_program_uniform_target(ctx);    // current program, else pipeline's active program
    
    ERROR_CHECK_RETURN_VALUE(ptr, GL_INVALID_OPERATION, false)

    ERROR_CHECK_RETURN_VALUE(location >= 0, GL_INVALID_OPERATION, false)
        
    ERROR_CHECK_RETURN_VALUE(location < MAX_BINDABLE_BUFFERS, GL_INVALID_OPERATION, false)

    return true;
}

#pragma mark glUniform* (current program)

/* legacy raw entry: write bytes into the uniform at location (component kind of the uniform) */
void mglUniform(GLMContext ctx, GLint location, void *ptr, GLsizei size)
{
    MGLIProgram *ip = current_program(ctx, __FUNCTION__);
    if (!ip || location == -1) return;
    if (location < 0 || location >= ip->nlocs || ip->locs[location].uniform < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
    MGLIUniform *u = &ip->uniforms[ip->locs[location].uniform];
    size_t esz = (size_t)u->cols * u->rows * comp_bytes(u);
    size_t off = (size_t)ip->locs[location].element * esz, total = (size_t)u->size * esz;
    if (size < 0 || !ptr || !u->shadow) return;
    if (off + (size_t)size > total) size = (GLsizei)(total - off);
    memcpy(u->shadow + off, ptr, (size_t)size);
    mgli_upload_uniform(ctx, ip, u);
}

#define UNI_VEC(N, SUF, CT, K) \
void mglUniform##N##SUF##v(GLMContext ctx, GLint location, GLsizei count, const CT *value) \
{ MGLIProgram *ip = current_program(ctx, __FUNCTION__); if (!ip) return; \
  mgli_set_uniform(ctx, __FUNCTION__, ip, location, count, K, 1, N, GL_FALSE, value); }

#define UNI_MAT(NAME, C, R, CT, K) \
void mglUniformMatrix##NAME(GLMContext ctx, GLint location, GLsizei count, GLboolean transpose, const CT *value) \
{ MGLIProgram *ip = current_program(ctx, __FUNCTION__); if (!ip) return; \
  mgli_set_uniform(ctx, __FUNCTION__, ip, location, count, K, C, R, transpose, value); }

UNI_VEC(1, f, GLfloat, MGLI_SET_F) UNI_VEC(2, f, GLfloat, MGLI_SET_F) UNI_VEC(3, f, GLfloat, MGLI_SET_F) UNI_VEC(4, f, GLfloat, MGLI_SET_F)
UNI_VEC(1, i, GLint, MGLI_SET_I)   UNI_VEC(2, i, GLint, MGLI_SET_I)   UNI_VEC(3, i, GLint, MGLI_SET_I)   UNI_VEC(4, i, GLint, MGLI_SET_I)
UNI_VEC(1, ui, GLuint, MGLI_SET_UI) UNI_VEC(2, ui, GLuint, MGLI_SET_UI) UNI_VEC(3, ui, GLuint, MGLI_SET_UI) UNI_VEC(4, ui, GLuint, MGLI_SET_UI)
UNI_VEC(1, d, GLdouble, MGLI_SET_D) UNI_VEC(2, d, GLdouble, MGLI_SET_D) UNI_VEC(3, d, GLdouble, MGLI_SET_D) UNI_VEC(4, d, GLdouble, MGLI_SET_D)

void mglUniform1f(GLMContext ctx, GLint location, GLfloat v0) { GLfloat v[1] = { v0 }; mglUniform1fv(ctx, location, 1, v); }
void mglUniform2f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1) { GLfloat v[2] = { v0, v1 }; mglUniform2fv(ctx, location, 1, v); }
void mglUniform3f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1, GLfloat v2) { GLfloat v[3] = { v0, v1, v2 }; mglUniform3fv(ctx, location, 1, v); }
void mglUniform4f(GLMContext ctx, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { GLfloat v[4] = { v0, v1, v2, v3 }; mglUniform4fv(ctx, location, 1, v); }
void mglUniform1i(GLMContext ctx, GLint location, GLint v0) { GLint v[1] = { v0 }; mglUniform1iv(ctx, location, 1, v); }
void mglUniform2i(GLMContext ctx, GLint location, GLint v0, GLint v1) { GLint v[2] = { v0, v1 }; mglUniform2iv(ctx, location, 1, v); }
void mglUniform3i(GLMContext ctx, GLint location, GLint v0, GLint v1, GLint v2) { GLint v[3] = { v0, v1, v2 }; mglUniform3iv(ctx, location, 1, v); }
void mglUniform4i(GLMContext ctx, GLint location, GLint v0, GLint v1, GLint v2, GLint v3) { GLint v[4] = { v0, v1, v2, v3 }; mglUniform4iv(ctx, location, 1, v); }
void mglUniform1ui(GLMContext ctx, GLint location, GLuint v0) { GLuint v[1] = { v0 }; mglUniform1uiv(ctx, location, 1, v); }
void mglUniform2ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1) { GLuint v[2] = { v0, v1 }; mglUniform2uiv(ctx, location, 1, v); }
void mglUniform3ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1, GLuint v2) { GLuint v[3] = { v0, v1, v2 }; mglUniform3uiv(ctx, location, 1, v); }
void mglUniform4ui(GLMContext ctx, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3) { GLuint v[4] = { v0, v1, v2, v3 }; mglUniform4uiv(ctx, location, 1, v); }
void mglUniform1d(GLMContext ctx, GLint location, GLdouble x) { GLdouble v[1] = { x }; mglUniform1dv(ctx, location, 1, v); }
void mglUniform2d(GLMContext ctx, GLint location, GLdouble x, GLdouble y) { GLdouble v[2] = { x, y }; mglUniform2dv(ctx, location, 1, v); }
void mglUniform3d(GLMContext ctx, GLint location, GLdouble x, GLdouble y, GLdouble z) { GLdouble v[3] = { x, y, z }; mglUniform3dv(ctx, location, 1, v); }
void mglUniform4d(GLMContext ctx, GLint location, GLdouble x, GLdouble y, GLdouble z, GLdouble w) { GLdouble v[4] = { x, y, z, w }; mglUniform4dv(ctx, location, 1, v); }

UNI_MAT(2fv, 2, 2, GLfloat, MGLI_SET_F)   UNI_MAT(3fv, 3, 3, GLfloat, MGLI_SET_F)   UNI_MAT(4fv, 4, 4, GLfloat, MGLI_SET_F)
UNI_MAT(2x3fv, 2, 3, GLfloat, MGLI_SET_F) UNI_MAT(3x2fv, 3, 2, GLfloat, MGLI_SET_F) UNI_MAT(2x4fv, 2, 4, GLfloat, MGLI_SET_F)
UNI_MAT(4x2fv, 4, 2, GLfloat, MGLI_SET_F) UNI_MAT(3x4fv, 3, 4, GLfloat, MGLI_SET_F) UNI_MAT(4x3fv, 4, 3, GLfloat, MGLI_SET_F)
UNI_MAT(2dv, 2, 2, GLdouble, MGLI_SET_D)   UNI_MAT(3dv, 3, 3, GLdouble, MGLI_SET_D)   UNI_MAT(4dv, 4, 4, GLdouble, MGLI_SET_D)
UNI_MAT(2x3dv, 2, 3, GLdouble, MGLI_SET_D) UNI_MAT(3x2dv, 3, 2, GLdouble, MGLI_SET_D) UNI_MAT(2x4dv, 2, 4, GLdouble, MGLI_SET_D)
UNI_MAT(4x2dv, 4, 2, GLdouble, MGLI_SET_D) UNI_MAT(3x4dv, 3, 4, GLdouble, MGLI_SET_D) UNI_MAT(4x3dv, 4, 3, GLdouble, MGLI_SET_D)
