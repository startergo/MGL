/*
 * program_introspection.c
 * MGL — program interface queries, ProgramUniform*, subroutines, location bindings
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * Owned by the 'introspect' domain.  Program reflection tables (MGLIProgram) are built by the
 * post-link hook registered here (mgl_ext_register_post_link) from the patched SPIR-V, see
 * introspect_spirv.c.  This file holds:
 *   - per-program records (bindings requested before link, reflection after link)
 *   - GL 4.3 program interface queries (GetProgramInterfaceiv / GetProgramResource*)
 *   - legacy attribute / fragment output / atomic counter / storage block queries
 *   - glProgramUniform* (shares mgli_set_uniform with glUniform*, uniforms.c)
 *   - subroutines (emulated: source rewrite to an int selector + switch dispatch,
 *     introspect_subroutine.c)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spirv_cross_c.h"
#include "mgl.h"
#include "glm_context.h"
#include "shaders.h"
#include "programs.h"
#include "mgl_ext_introspect.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_program.h"

Program *findProgram(GLMContext ctx, GLuint program);
Shader *findShader(GLMContext ctx, GLuint shader);
void mgli_get_active_uniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name);
void mgli_apply_block_binding(GLMContext ctx, MGLIProgram *ip, MGLIBlock *b, int res_type);

#pragma mark - records

void mgli_set_error(GLMContext ctx, const char *func, GLenum err)
{
    ctx->error_func(ctx, func, err);
}

static struct MGLExtState_introspect *ext(GLMContext ctx)
{
    if (!ctx->state.ext_introspect) ctx->state.ext_introspect = calloc(1, sizeof(struct MGLExtState_introspect));
    return ctx->state.ext_introspect;
}

static void free_link_tables(MGLIProgram *ip)
{
    for (int i = 0; i < ip->nuniforms; i++) { free(ip->uniforms[i].name); free(ip->uniforms[i].shadow); }
    free(ip->uniforms); ip->uniforms = NULL; ip->nuniforms = 0;
    free(ip->locs); ip->locs = NULL; ip->nlocs = 0;
    for (int i = 0; i < ip->nubos; i++) { free(ip->ubos[i].name); free(ip->ubos[i].members); }
    free(ip->ubos); ip->ubos = NULL; ip->nubos = 0;
    for (int i = 0; i < ip->nssbos; i++) { free(ip->ssbos[i].name); free(ip->ssbos[i].members); }
    free(ip->ssbos); ip->ssbos = NULL; ip->nssbos = 0;
    for (int i = 0; i < ip->nbvars; i++) free(ip->bvars[i].name);
    free(ip->bvars); ip->bvars = NULL; ip->nbvars = 0;
    for (int i = 0; i < ip->ninputs; i++) free(ip->inputs[i].name);
    free(ip->inputs); ip->inputs = NULL; ip->ninputs = 0;
    for (int i = 0; i < ip->noutputs; i++) free(ip->outputs[i].name);
    free(ip->outputs); ip->outputs = NULL; ip->noutputs = 0;
    for (int i = 0; i < ip->nvtx_outputs; i++) free(ip->vtx_outputs[i].name);
    free(ip->vtx_outputs); ip->vtx_outputs = NULL; ip->nvtx_outputs = 0;
    for (int i = 0; i < ip->nabufs; i++) free(ip->abufs[i].members);
    free(ip->abufs); ip->abufs = NULL; ip->nabufs = 0;
    for (int s = 0; s < MGLI_STAGES; s++) mgli_free_stage_sub(&ip->sub[s]);
    ip->nxfb = 0;
    for (int s = 0; s < MGLI_STAGES; s++) ip->dub_res_index[s] = -1;
    ip->linked = 0;
}

static void reset_record(MGLIProgram *ip)
{
    free_link_tables(ip);
    for (int i = 0; i < ip->nattrib_binds; i++) free(ip->attrib_binds[i].name);
    free(ip->attrib_binds); ip->attrib_binds = NULL; ip->nattrib_binds = 0;
    for (int i = 0; i < ip->nfrag_binds; i++) free(ip->frag_binds[i].name);
    free(ip->frag_binds); ip->frag_binds = NULL; ip->nfrag_binds = 0;
    /* the default-block Buffers may still be referenced by a renderer buffer map: keep them for reuse */
}

MGLIProgram *mgli_find(GLMContext ctx, GLuint program)
{
    Program *p = findProgram(ctx, program);
    if (!p) return NULL;
    for (MGLIProgram *it = ext(ctx)->programs; it; it = it->next)
        if (it->name == program) {
            if (it->prog != p) { reset_record(it); it->prog = p; }
            return it;
        }
    return NULL;
}

MGLIProgram *mgli_get(GLMContext ctx, GLuint program)
{
    MGLIProgram *ip = mgli_find(ctx, program);
    if (ip) return ip;
    Program *p = findProgram(ctx, program);
    if (!p) return NULL;
    ip = calloc(1, sizeof(MGLIProgram));
    ip->name = program;
    ip->prog = p;
    for (int s = 0; s < MGLI_STAGES; s++) ip->dub_res_index[s] = -1;
    ip->next = ext(ctx)->programs;
    ext(ctx)->programs = ip;
    return ip;
}

static MGLIProgram *find_by_ptr(GLMContext ctx, Program *p)
{
    if (!p) return NULL;
    for (MGLIProgram *it = ext(ctx)->programs; it; it = it->next) if (it->prog == p) return it;
    return NULL;
}

MGLIProgram *mgli_program_checked(GLMContext ctx, GLuint program, bool need_linked)
{
    if (!findProgram(ctx, program)) {
        mgli_set_error(ctx, "program", (program && findShader(ctx, program)) ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
        return NULL;
    }
    MGLIProgram *ip = mgli_get(ctx, program);
    if (need_linked && !ip->linked) { mgli_set_error(ctx, "program", GL_INVALID_OPERATION); return NULL; }
    return ip;
}

#pragma mark - link hooks

static void introspect_pre_link(GLMContext ctx, Program *ptr)
{
    MGLIProgram *ip = mgli_get(ctx, ptr->name);
    if (!ip) return;
    ip->prog = ptr;
    mgli_free_inits(ip);
}

static MGLIShaderRec *find_shader_rec(GLMContext ctx, Shader *sh)
{
    for (MGLIShaderRec *r = ext(ctx)->shaders; r; r = r->next) if (r->ptr == sh && r->shader == sh->name) return r;
    return NULL;
}

static void build_subroutines(GLMContext ctx, MGLIProgram *ip, Program *ptr)
{
    for (int s = 0; s < MGLI_STAGES; s++) {
        MGLIStageSub *d = &ip->sub[s];
        mgli_free_stage_sub(d);
        if (!ptr->shader_slots[s]) continue;
        MGLIShaderRec *r = NULL;
        MGLProgramObj *po = MGL_PROGRAM_OBJ(ptr);
        for (int a = 0; a < po->num_attached && !r; a++)
            if (po->attached[a] && (int)po->attached[a]->glm_type == s) r = find_shader_rec(ctx, po->attached[a]);
        if (!r || (!r->sub.nfuncs && !r->sub.nunis)) continue;
        d->nfuncs = r->sub.nfuncs;
        d->funcs = calloc((size_t)d->nfuncs + 1, sizeof(MGLISubFunc));
        for (int i = 0; i < d->nfuncs; i++) { d->funcs[i].name = strdup(r->sub.funcs[i].name); d->funcs[i].index = r->sub.funcs[i].index; }
        d->nunis = r->sub.nunis;
        d->unis = calloc((size_t)d->nunis + 1, sizeof(MGLISubUniform));
        d->nlocs = 0;
        for (int i = 0; i < d->nunis; i++) {
            MGLISubUniform *su = &r->sub.unis[i], *du = &d->unis[i];
            du->name = strdup(su->name); du->type_name = strdup(su->type_name); du->selector = strdup(su->selector);
            du->location = su->location; du->size = su->size;
            du->ncompat = su->ncompat; du->compat = calloc((size_t)su->ncompat + 1, sizeof(int));
            memcpy(du->compat, su->compat, (size_t)su->ncompat * sizeof(int));
            if (du->location + du->size > d->nlocs) d->nlocs = du->location + du->size;
        }
        d->selection = calloc((size_t)d->nlocs + 1, sizeof(GLuint));
        for (int i = 0; i < d->nunis; i++)
            for (int e = 0; e < d->unis[i].size; e++)
                d->selection[d->unis[i].location + e] = d->unis[i].ncompat ? (GLuint)d->unis[i].compat[0] : 0;
    }
}

static void write_selectors(GLMContext ctx, MGLIProgram *ip, int stage)
{
    MGLIStageSub *d = &ip->sub[stage];
    for (int i = 0; i < d->nunis; i++) {
        MGLIUniform *u = NULL;
        size_t ln = strlen(d->unis[i].selector);
        for (int k = 0; k < ip->nuniforms; k++) {
            if (!ip->uniforms[k].st[stage].present) continue;
            if (!strcmp(ip->uniforms[k].name, d->unis[i].selector) ||
                (!strncmp(ip->uniforms[k].name, d->unis[i].selector, ln) && !strcmp(ip->uniforms[k].name + ln, "[0]"))) u = &ip->uniforms[k];
        }
        if (!u || !u->shadow) continue;
        for (int e = 0; e < d->unis[i].size && e < u->size; e++) ((uint32_t *)u->shadow)[e] = d->selection[d->unis[i].location + e];
        mgli_upload_uniform(ctx, ip, u);
    }
}

static void introspect_post_link(GLMContext ctx, Program *ptr)
{
    MGLIProgram *ip = mgli_get(ctx, ptr->name);
    if (!ip) return;
    ip->prog = ptr;
    free_link_tables(ip);
    /* a stage whose SPIR-V is missing failed to link */
    for (int s = 0; s < MGLI_STAGES; s++) if (ptr->shader_slots[s] && !ptr->spirv[s].ir) return;
    if (!mgli_process_link(ctx, ptr, ip)) return;
    build_subroutines(ctx, ip, ptr);
    ip->linked = 1;
    for (int s = 0; s < MGLI_STAGES; s++) if (ip->sub[s].nunis) write_selectors(ctx, ip, s);

    /* the stage MSL was generated from the patched SPIR-V (mgl_introspect_patch_stage) */
    ctx->state.dirty_bits |= DIRTY_PROGRAM | DIRTY_BUFFER_BASE_STATE | DIRTY_TEX_BINDING;
}

Buffer *mgl_introspect_default_uniform_buffer(GLMContext ctx, Program *program, int stage, int index)
{
    if (!program || stage < 0 || stage >= MGLI_STAGES) return NULL;
    MGLIProgram *ip = find_by_ptr(ctx, program);
    if (!ip && ctx->state.program_pipeline && program == mgl_program_pipeline_composite(ctx))
        ip = find_by_ptr(ctx, ctx->state.program_pipeline->stage_programs[stage]); /* separable pipeline */
    if (!ip || !ip->linked || ip->dub_res_index[stage] != index) return NULL;
    return ip->dub[stage];
}

const char *mgl_introspect_shader_source(GLMContext ctx, Shader *shader)
{
    if (!shader || !shader->src) return shader ? shader->src : NULL;
    struct MGLExtState_introspect *e = ext(ctx);
    MGLIShaderRec *r = NULL;
    for (MGLIShaderRec *it = e->shaders; it; it = it->next) if (it->shader == shader->name) { r = it; break; }
    MGLIStageSub sub; memset(&sub, 0, sizeof sub);
    char *rewritten = mgli_rewrite_subroutines(shader->src, &sub);
    if (!rewritten) {
        if (r) { free(r->src); r->src = NULL; mgli_free_stage_sub(&r->sub); r->ptr = shader; }
        return shader->src;
    }
    if (!r) { r = calloc(1, sizeof *r); r->shader = shader->name; r->next = e->shaders; e->shaders = r; }
    free(r->src); mgli_free_stage_sub(&r->sub);
    r->ptr = shader; r->src = rewritten; r->sub = sub;
    return r->src;
}

#pragma mark - domain 'introspect' hooks

void mgl_ext_init_introspect(GLMContext ctx)
{
    ext(ctx);
    mgl_ext_register_pre_link(introspect_pre_link);
    mgl_ext_register_post_link(introspect_post_link);
    mgl_program_register_programiv_hook(mgl_introspect_get_programiv);
}

bool mgl_ext_get_introspect(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    (void)ctx;
    switch (pname) {
        case GL_MAX_SUBROUTINES: mgl_get_store(type, data, 0, 256); return true;
        case GL_MAX_SUBROUTINE_UNIFORM_LOCATIONS: mgl_get_store(type, data, 0, 1024); return true;
        default: return false;
    }
}

static GLint max_len_uniforms(MGLIProgram *ip)
{
    GLint m = 0;
    for (int i = 0; i < ip->nuniforms; i++) if (!ip->uniforms[i].hidden) { GLint l = (GLint)strlen(ip->uniforms[i].name) + 1; if (l > m) m = l; }
    return m;
}

static bool vertex_inputs(MGLIProgram *ip) { return ip->first_stage == _VERTEX_SHADER; }

bool mgl_introspect_get_programiv(GLMContext ctx, Program *program, GLenum pname, GLint *params)
{
    if (!program || !params) return false;
    MGLIProgram *ip = mgli_get(ctx, program->name);
    if (!ip || !ip->linked) return false;
    switch (pname) {
        case GL_ACTIVE_UNIFORMS: { GLint n = 0; for (int i = 0; i < ip->nuniforms; i++) n += !ip->uniforms[i].hidden; *params = n; return true; }
        case GL_ACTIVE_UNIFORM_MAX_LENGTH: *params = max_len_uniforms(ip); return true;
        case GL_ACTIVE_ATTRIBUTES: *params = vertex_inputs(ip) ? ip->ninputs : 0; return true;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: {
            GLint m = 0;
            if (vertex_inputs(ip)) for (int i = 0; i < ip->ninputs; i++) { GLint l = (GLint)strlen(ip->inputs[i].name) + 1; if (l > m) m = l; }
            *params = m; return true;
        }
        case GL_ACTIVE_UNIFORM_BLOCKS: *params = ip->nubos; return true;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: {
            GLint m = 0; for (int i = 0; i < ip->nubos; i++) { GLint l = (GLint)strlen(ip->ubos[i].name) + 1; if (l > m) m = l; }
            *params = m; return true;
        }
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS: *params = ip->nabufs; return true;
        default: return false;
    }
}

#pragma mark - attributes / fragment outputs

static bool name_matches_var(const char *res, const char *name)
{
    if (!strcmp(res, name)) return true;
    size_t ln = strlen(name);
    return !strncmp(res, name, ln) && !strcmp(res + ln, "[0]");
}

/* "a[N]" -> location of "a[0]" + N (N < size) */
static GLint var_location(const MGLIVar *vars, int n, const char *name, GLint *index_out)
{
    for (int i = 0; i < n; i++)
        if (name_matches_var(vars[i].name, name)) { if (index_out) *index_out = vars[i].index; return vars[i].location; }
    size_t ln = strlen(name);
    if (ln > 3 && name[ln - 1] == ']') {
        const char *lb = strrchr(name, '[');
        char *end; long idx = strtol(lb + 1, &end, 10);
        if (end != lb + 1 && *end == ']' && idx >= 0) {
            char base[512]; size_t bl = (size_t)(lb - name);
            if (bl + 4 < sizeof base) {
                memcpy(base, name, bl); strcpy(base + bl, "[0]");
                for (int i = 0; i < n; i++)
                    if (!strcmp(vars[i].name, base) && idx < vars[i].size && vars[i].location >= 0) {
                        if (index_out) *index_out = vars[i].index;
                        return vars[i].location + (GLint)idx;
                    }
            }
        }
    }
    return -1;
}

void mglBindAttribLocation(GLMContext ctx, GLuint program, GLuint index, const GLchar *name)
{
    if (index >= (ctx->state.max_vertex_attribs ? ctx->state.max_vertex_attribs : MAX_ATTRIBS)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (!name) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    if (!strncmp(name, "gl_", 3)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
    for (int i = 0; i < ip->nattrib_binds; i++)
        if (!strcmp(ip->attrib_binds[i].name, name)) { ip->attrib_binds[i].index = index; return; }
    ip->attrib_binds = realloc(ip->attrib_binds, (size_t)(ip->nattrib_binds + 1) * sizeof(*ip->attrib_binds));
    ip->attrib_binds[ip->nattrib_binds].name = strdup(name);
    ip->attrib_binds[ip->nattrib_binds].index = index;
    ip->nattrib_binds++;
}

GLint mgl_introspect_get_attrib_location(GLMContext ctx, GLuint program, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, true);
    if (!ip || !name || !strncmp(name, "gl_", 3) || !vertex_inputs(ip)) return -1;
    return var_location(ip->inputs, ip->ninputs, name, NULL);
}

void mglGetActiveAttrib(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    if (bufSize < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (!vertex_inputs(ip) || index >= (GLuint)ip->ninputs) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIVar *v = &ip->inputs[index];
    GLsizei n = 0;
    if (name && bufSize > 0) { n = (GLsizei)strlen(v->name); if (n > bufSize - 1) n = bufSize - 1; memcpy(name, v->name, (size_t)n); name[n] = 0; }
    if (length) *length = n;
    if (size) *size = v->size;
    if (type) *type = v->type;
}

void mglGetActiveUniform(GLMContext ctx, GLuint program, GLuint index, GLsizei bufSize, GLsizei *length, GLint *size, GLenum *type, GLchar *name)
{
    mgli_get_active_uniform(ctx, program, index, bufSize, length, size, type, name);
}

void mglGetAttachedShaders(GLMContext ctx, GLuint program, GLsizei maxCount, GLsizei *count, GLuint *shaders)
{
    if (maxCount < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    if (!mgli_program_checked(ctx, program, false)) return;
    MGLProgramObj *po = MGL_PROGRAM_OBJ(findProgram(ctx, program));
    GLsizei n = 0;
    for (int i = 0; i < po->num_attached && n < maxCount; i++) { if (shaders) shaders[n] = po->attached[i]->name; n++; }
    if (count) *count = n;
}

void mglBindFragDataLocationIndexed(GLMContext ctx, GLuint program, GLuint colorNumber, GLuint index, const GLchar *name)
{
    GLuint max_db = ctx->state.var.max_draw_buffers > 0 ? (GLuint)ctx->state.var.max_draw_buffers : 8;
    GLuint max_ds = ctx->state.var.max_dual_source_draw_buffers > 0 ? (GLuint)ctx->state.var.max_dual_source_draw_buffers : 1;
    if (index > 1 || colorNumber >= max_db || (index == 1 && colorNumber >= max_ds)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (!name) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    if (!strncmp(name, "gl_", 3)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
    for (int i = 0; i < ip->nfrag_binds; i++)
        if (!strcmp(ip->frag_binds[i].name, name)) { ip->frag_binds[i].color = colorNumber; ip->frag_binds[i].index = index; return; }
    ip->frag_binds = realloc(ip->frag_binds, (size_t)(ip->nfrag_binds + 1) * sizeof(*ip->frag_binds));
    ip->frag_binds[ip->nfrag_binds].name = strdup(name);
    ip->frag_binds[ip->nfrag_binds].color = colorNumber;
    ip->frag_binds[ip->nfrag_binds].index = index;
    ip->nfrag_binds++;
}

void mglBindFragDataLocation(GLMContext ctx, GLuint program, GLuint color, const GLchar *name)
{
    mglBindFragDataLocationIndexed(ctx, program, color, 0, name);
}

static bool frag_outputs(MGLIProgram *ip) { return ip->last_stage == _FRAGMENT_SHADER; }

GLint  mglGetFragDataLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, true);
    if (!ip || !name || !frag_outputs(ip) || !strncmp(name, "gl_", 3)) return -1;
    return var_location(ip->outputs, ip->noutputs, name, NULL);
}

GLint  mglGetFragDataIndex(GLMContext ctx, GLuint program, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, true);
    if (!ip || !name || !frag_outputs(ip) || !strncmp(name, "gl_", 3)) return -1;
    GLint idx = -1;
    if (var_location(ip->outputs, ip->noutputs, name, &idx) < 0) return -1;
    return idx < 0 ? 0 : idx;
}

#pragma mark - atomic counters / storage blocks

void mglGetActiveAtomicCounterBufferiv(GLMContext ctx, GLuint program, GLuint bufferIndex, GLenum pname, GLint *params)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (bufferIndex >= (GLuint)ip->nabufs) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLIAtomicBuffer *a = &ip->abufs[bufferIndex];
    if (!params) return;
    switch (pname) {
        case GL_ATOMIC_COUNTER_BUFFER_BINDING: *params = a->binding; break;
        case GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE: *params = a->data_size; break;
        case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS: *params = a->member_count; break;
        case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES:
            for (int i = 0; i < a->member_count; i++) {
                int pub = 0; for (int k = 0; k < a->members[i]; k++) if (!ip->uniforms[k].hidden) pub++;
                params[i] = pub;
            }
            break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER: *params = (a->stages >> _VERTEX_SHADER) & 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER: *params = (a->stages >> _TESS_CONTROL_SHADER) & 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER: *params = (a->stages >> _TESS_EVALUATION_SHADER) & 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER: *params = (a->stages >> _GEOMETRY_SHADER) & 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER: *params = (a->stages >> _FRAGMENT_SHADER) & 1; break;
        case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER: *params = (a->stages >> _COMPUTE_SHADER) & 1; break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); break;
    }
}

void mglShaderStorageBlockBinding(GLMContext ctx, GLuint program, GLuint storageBlockIndex, GLuint storageBlockBinding)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    if (storageBlockIndex >= (GLuint)ip->nssbos) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    GLuint max = ctx->state.var.max_shader_storage_buffer_bindings > 0 ? (GLuint)ctx->state.var.max_shader_storage_buffer_bindings : MAX_BINDABLE_BUFFERS;
    if (max > MAX_BINDABLE_BUFFERS) max = MAX_BINDABLE_BUFFERS; /* renderer binding table size */
    if (storageBlockBinding >= max) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    ip->ssbos[storageBlockIndex].binding = (GLint)storageBlockBinding;
    mgli_apply_block_binding(ctx, ip, &ip->ssbos[storageBlockIndex], SPVC_RESOURCE_TYPE_STORAGE_BUFFER);
}

#pragma mark - program interface queries (GL 4.3)

static int stage_of_shadertype(GLenum t)
{
    switch (t) {
        case GL_VERTEX_SHADER: return _VERTEX_SHADER;
        case GL_TESS_CONTROL_SHADER: return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SHADER: return _TESS_EVALUATION_SHADER;
        case GL_GEOMETRY_SHADER: return _GEOMETRY_SHADER;
        case GL_FRAGMENT_SHADER: return _FRAGMENT_SHADER;
        case GL_COMPUTE_SHADER: return _COMPUTE_SHADER;
        default: return -1;
    }
}

/* subroutine interfaces: returns stage, sets *is_uniform */
static int subroutine_iface(GLenum iface, int *is_uniform)
{
    *is_uniform = 0;
    switch (iface) {
        case GL_VERTEX_SUBROUTINE: return _VERTEX_SHADER;
        case GL_TESS_CONTROL_SUBROUTINE: return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SUBROUTINE: return _TESS_EVALUATION_SHADER;
        case GL_GEOMETRY_SUBROUTINE: return _GEOMETRY_SHADER;
        case GL_FRAGMENT_SUBROUTINE: return _FRAGMENT_SHADER;
        case GL_COMPUTE_SUBROUTINE: return _COMPUTE_SHADER;
        case GL_VERTEX_SUBROUTINE_UNIFORM: *is_uniform = 1; return _VERTEX_SHADER;
        case GL_TESS_CONTROL_SUBROUTINE_UNIFORM: *is_uniform = 1; return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SUBROUTINE_UNIFORM: *is_uniform = 1; return _TESS_EVALUATION_SHADER;
        case GL_GEOMETRY_SUBROUTINE_UNIFORM: *is_uniform = 1; return _GEOMETRY_SHADER;
        case GL_FRAGMENT_SUBROUTINE_UNIFORM: *is_uniform = 1; return _FRAGMENT_SHADER;
        case GL_COMPUTE_SUBROUTINE_UNIFORM: *is_uniform = 1; return _COMPUTE_SHADER;
        default: return -1;
    }
}

static bool valid_iface(GLenum iface)
{
    int u;
    switch (iface) {
        case GL_UNIFORM: case GL_UNIFORM_BLOCK: case GL_ATOMIC_COUNTER_BUFFER: case GL_PROGRAM_INPUT: case GL_PROGRAM_OUTPUT:
        case GL_TRANSFORM_FEEDBACK_VARYING: case GL_TRANSFORM_FEEDBACK_BUFFER: case GL_BUFFER_VARIABLE: case GL_SHADER_STORAGE_BLOCK:
            return true;
        default: return subroutine_iface(iface, &u) >= 0;
    }
}

/* transform feedback varyings come from the xfb domain (mgl_xfb_program_info) */
static const MGLXfbProgram *g_xfb; /* set by xfb_sync for the query in progress */

static void xfb_sync(GLMContext ctx, MGLIProgram *ip)
{
    const MGLXfbProgram *x = ip->prog ? mgl_xfb_program_info(ctx, ip->prog) : NULL;
    g_xfb = (x && x->linked && !x->link_failed) ? x : NULL;
    ip->nxfb = g_xfb ? (int)g_xfb->count : 0;
}

static int xfb_buffer_count(MGLIProgram *ip)
{
    (void)ip;
    if (!g_xfb) return 0;
    int n = 0; for (int b = 0; b < MGL_XFB_MAX_BUFFERS; b++) if (g_xfb->buffer_mask & (1u << b)) n++;
    return n;
}

/* resource index -> TF buffer binding */
static int xfb_buffer_binding(int index)
{
    int n = 0;
    for (int b = 0; g_xfb && b < MGL_XFB_MAX_BUFFERS; b++) if (g_xfb->buffer_mask & (1u << b)) { if (n == index) return b; n++; }
    return -1;
}

typedef struct { GLenum type; GLint size, offset, buffer; } XfbInfo;

static XfbInfo xfb_info(MGLIProgram *ip, int i)
{
    (void)ip;
    XfbInfo x = { GL_NONE, 0, -1, 0 };
    if (!g_xfb || i < 0 || (GLuint)i >= g_xfb->count) return x;
    const MGLXfbVarying *v = &g_xfb->vars[i];
    x.type = v->type; x.size = v->size; x.buffer = (GLint)v->buffer;
    x.offset = strcmp(v->name, "gl_NextBuffer") ? (GLint)v->offset : -1;
    return x;
}

static const char *xfb_name(int i) { return (g_xfb && (GLuint)i < g_xfb->count) ? g_xfb->vars[i].name : ""; }

static GLint xfb_buffer_stride(MGLIProgram *ip, int index)
{
    (void)ip;
    int b = xfb_buffer_binding(index);
    return b >= 0 ? (GLint)g_xfb->stride[b] : 0;
}

static int resource_count(MGLIProgram *ip, GLenum iface)
{
    int u, st = subroutine_iface(iface, &u);
    if (st >= 0) return u ? ip->sub[st].nunis : ip->sub[st].nfuncs;
    switch (iface) {
        case GL_UNIFORM: { int n = 0; for (int i = 0; i < ip->nuniforms; i++) n += !ip->uniforms[i].hidden; return n; }
        case GL_UNIFORM_BLOCK: return ip->nubos;
        case GL_ATOMIC_COUNTER_BUFFER: return ip->nabufs;
        case GL_PROGRAM_INPUT: return ip->ninputs;
        case GL_PROGRAM_OUTPUT: return ip->noutputs;
        case GL_TRANSFORM_FEEDBACK_VARYING: return ip->nxfb;
        case GL_TRANSFORM_FEEDBACK_BUFFER: return xfb_buffer_count(ip);
        case GL_BUFFER_VARIABLE: return ip->nbvars;
        case GL_SHADER_STORAGE_BLOCK: return ip->nssbos;
        default: return 0;
    }
}

static MGLIUniform *pub_uniform(MGLIProgram *ip, int index)
{
    int n = 0;
    for (int i = 0; i < ip->nuniforms; i++) { if (ip->uniforms[i].hidden) continue; if (n == index) return &ip->uniforms[i]; n++; }
    return NULL;
}

static int pub_index_of(MGLIProgram *ip, int internal)
{
    int n = 0; for (int k = 0; k < internal; k++) if (!ip->uniforms[k].hidden) n++;
    return n;
}

static const char *resource_name(MGLIProgram *ip, GLenum iface, int index)
{
    int u, st = subroutine_iface(iface, &u);
    if (st >= 0) return u ? ip->sub[st].unis[index].name : ip->sub[st].funcs[index].name;
    switch (iface) {
        case GL_UNIFORM: return pub_uniform(ip, index)->name;
        case GL_UNIFORM_BLOCK: return ip->ubos[index].name;
        case GL_PROGRAM_INPUT: return ip->inputs[index].name;
        case GL_PROGRAM_OUTPUT: return ip->outputs[index].name;
        case GL_TRANSFORM_FEEDBACK_VARYING: return xfb_name(index);
        case GL_BUFFER_VARIABLE: return ip->bvars[index].name;
        case GL_SHADER_STORAGE_BLOCK: return ip->ssbos[index].name;
        default: return "";
    }
}

void mglGetProgramInterfaceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLenum pname, GLint *params)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    xfb_sync(ctx, ip);
    if (!valid_iface(programInterface)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return; }
    int u, st = subroutine_iface(programInterface, &u);
    int n = resource_count(ip, programInterface);
    GLint v = 0;
    switch (pname) {
        case GL_ACTIVE_RESOURCES: v = n; break;
        case GL_MAX_NAME_LENGTH:
            if (programInterface == GL_ATOMIC_COUNTER_BUFFER || programInterface == GL_TRANSFORM_FEEDBACK_BUFFER) {
                mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return;
            }
            for (int i = 0; i < n; i++) { GLint l = (GLint)strlen(resource_name(ip, programInterface, i)) + 1; if (l > v) v = l; }
            break;
        case GL_MAX_NUM_ACTIVE_VARIABLES:
            switch (programInterface) {
                case GL_UNIFORM_BLOCK: for (int i = 0; i < ip->nubos; i++) if (ip->ubos[i].member_count > v) v = ip->ubos[i].member_count; break;
                case GL_SHADER_STORAGE_BLOCK: for (int i = 0; i < ip->nssbos; i++) if (ip->ssbos[i].member_count > v) v = ip->ssbos[i].member_count; break;
                case GL_ATOMIC_COUNTER_BUFFER: for (int i = 0; i < ip->nabufs; i++) if (ip->abufs[i].member_count > v) v = ip->abufs[i].member_count; break;
                case GL_TRANSFORM_FEEDBACK_BUFFER:
                    for (int bi = 0; bi < xfb_buffer_count(ip); bi++) {
                        GLint c = 0; int b = xfb_buffer_binding(bi);
                        for (int i = 0; i < ip->nxfb; i++) { XfbInfo x = xfb_info(ip, i); if (x.buffer == b && x.offset >= 0 && strncmp(xfb_name(i), "gl_", 3)) c++; }
                        if (c > v) v = c;
                    }
                    break;
                default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return;
            }
            break;
        case GL_MAX_NUM_COMPATIBLE_SUBROUTINES:
            if (st < 0 || !u) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
            for (int i = 0; i < ip->sub[st].nunis; i++) if (ip->sub[st].unis[i].ncompat > v) v = ip->sub[st].unis[i].ncompat;
            break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return;
    }
    if (params) *params = v;
}

GLuint  mglGetProgramResourceIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return GL_INVALID_INDEX;
    xfb_sync(ctx, ip);
    if (!valid_iface(programInterface) || programInterface == GL_ATOMIC_COUNTER_BUFFER || programInterface == GL_TRANSFORM_FEEDBACK_BUFFER) {
        mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return GL_INVALID_INDEX;
    }
    if (!name) return GL_INVALID_INDEX;
    int n = resource_count(ip, programInterface);
    int is_su, st = subroutine_iface(programInterface, &is_su);
    if (st >= 0 && !is_su) {
        for (int i = 0; i < n; i++) if (!strcmp(ip->sub[st].funcs[i].name, name)) return (GLuint)ip->sub[st].funcs[i].index;
        return GL_INVALID_INDEX;
    }
    for (int i = 0; i < n; i++) if (!strcmp(resource_name(ip, programInterface, i), name)) return (GLuint)i;
    for (int i = 0; i < n; i++) if (name_matches_var(resource_name(ip, programInterface, i), name)) return (GLuint)i;
    return GL_INVALID_INDEX;
}

/* subroutine function resources are indexed by their subroutine index */
static int sub_func_slot(MGLIStageSub *s, GLuint index)
{
    for (int i = 0; i < s->nfuncs; i++) if ((GLuint)s->funcs[i].index == index) return i;
    return -1;
}

void mglGetProgramResourceName(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    xfb_sync(ctx, ip);
    if (!valid_iface(programInterface) || programInterface == GL_ATOMIC_COUNTER_BUFFER || programInterface == GL_TRANSFORM_FEEDBACK_BUFFER) {
        mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return;
    }
    int is_su, st = subroutine_iface(programInterface, &is_su);
    int slot = (st >= 0 && !is_su) ? sub_func_slot(&ip->sub[st], index) : (int)index;
    if (bufSize < 0 || slot < 0 || slot >= resource_count(ip, programInterface)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    const char *rn = resource_name(ip, programInterface, slot);
    GLsizei n = 0;
    if (name && bufSize > 0) { n = (GLsizei)strlen(rn); if (n > bufSize - 1) n = bufSize - 1; memcpy(name, rn, (size_t)n); name[n] = 0; }
    if (length) *length = n;
}

static GLbitfield resource_stages(MGLIProgram *ip, GLenum iface, int index)
{
    switch (iface) {
        case GL_UNIFORM: return pub_uniform(ip, index)->stages;
        case GL_UNIFORM_BLOCK: return ip->ubos[index].stages;
        case GL_ATOMIC_COUNTER_BUFFER: return ip->abufs[index].stages;
        case GL_SHADER_STORAGE_BLOCK: return ip->ssbos[index].stages;
        case GL_BUFFER_VARIABLE: return ip->bvars[index].stages;
        case GL_PROGRAM_INPUT: return ip->inputs[index].stages;
        case GL_PROGRAM_OUTPUT: return ip->outputs[index].stages;
        default: return 0;
    }
}

void mglGetProgramResourceiv(GLMContext ctx, GLuint program, GLenum programInterface, GLuint index, GLsizei propCount, const GLenum *props, GLsizei count, GLsizei *length, GLint *params)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return;
    xfb_sync(ctx, ip);
    if (!valid_iface(programInterface)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return; }
    if (propCount <= 0 || count < 0 || !props) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    GLenum I = programInterface;
    int is_su, st = subroutine_iface(I, &is_su);
    int idx = (st >= 0 && !is_su) ? sub_func_slot(&ip->sub[st], index) : (int)index;
    if (idx < 0 || idx >= resource_count(ip, I)) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    static GLint vals[8192]; GLsizei nv = 0;
    for (GLsizei p = 0; p < propCount; p++) {
        GLenum prop = props[p];
        bool ok = true;
#define PUSH(x) do { if (nv < 8192) vals[nv++] = (GLint)(x); } while (0)
        switch (prop) {
            case GL_NAME_LENGTH:
                if (I == GL_ATOMIC_COUNTER_BUFFER || I == GL_TRANSFORM_FEEDBACK_BUFFER) { ok = false; break; }
                PUSH(strlen(resource_name(ip, I, idx)) + 1); break;
            case GL_TYPE:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->type);
                else if (I == GL_PROGRAM_INPUT) PUSH(ip->inputs[idx].type);
                else if (I == GL_PROGRAM_OUTPUT) PUSH(ip->outputs[idx].type);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].type);
                else if (I == GL_TRANSFORM_FEEDBACK_VARYING) PUSH(xfb_info(ip, idx).type);
                else ok = false;
                break;
            case GL_ARRAY_SIZE:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->size);
                else if (I == GL_PROGRAM_INPUT) PUSH(ip->inputs[idx].size);
                else if (I == GL_PROGRAM_OUTPUT) PUSH(ip->outputs[idx].size);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].size);
                else if (I == GL_TRANSFORM_FEEDBACK_VARYING) PUSH(xfb_info(ip, idx).size);
                else if (st >= 0 && is_su) PUSH(ip->sub[st].unis[idx].size);
                else ok = false;
                break;
            case GL_OFFSET:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->offset);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].offset);
                else if (I == GL_TRANSFORM_FEEDBACK_VARYING) PUSH(xfb_info(ip, idx).offset);
                else ok = false;
                break;
            case GL_BLOCK_INDEX:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->block_index);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].block_index);
                else ok = false;
                break;
            case GL_ARRAY_STRIDE:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->array_stride);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].array_stride);
                else ok = false;
                break;
            case GL_MATRIX_STRIDE:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->matrix_stride);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].matrix_stride);
                else ok = false;
                break;
            case GL_IS_ROW_MAJOR:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->is_row_major);
                else if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].is_row_major);
                else ok = false;
                break;
            case GL_ATOMIC_COUNTER_BUFFER_INDEX:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->atomic_buffer_index); else ok = false;
                break;
            case GL_BUFFER_BINDING:
                if (I == GL_UNIFORM_BLOCK) PUSH(ip->ubos[idx].binding);
                else if (I == GL_SHADER_STORAGE_BLOCK) PUSH(ip->ssbos[idx].binding);
                else if (I == GL_ATOMIC_COUNTER_BUFFER) PUSH(ip->abufs[idx].binding);
                else if (I == GL_TRANSFORM_FEEDBACK_BUFFER) PUSH(xfb_buffer_binding(idx));
                else ok = false;
                break;
            case GL_BUFFER_DATA_SIZE:
                if (I == GL_UNIFORM_BLOCK) PUSH(ip->ubos[idx].data_size);
                else if (I == GL_SHADER_STORAGE_BLOCK) PUSH(ip->ssbos[idx].data_size);
                else if (I == GL_ATOMIC_COUNTER_BUFFER) PUSH(ip->abufs[idx].data_size);
                else ok = false;
                break;
            case GL_NUM_ACTIVE_VARIABLES: case GL_ACTIVE_VARIABLES: {
                int *mem = NULL, nm = 0; bool uniform_idx = false;
                int tmp[1024]; int ntmp = 0;
                if (I == GL_UNIFORM_BLOCK) { mem = ip->ubos[idx].members; nm = ip->ubos[idx].member_count; uniform_idx = true; }
                else if (I == GL_SHADER_STORAGE_BLOCK) { mem = ip->ssbos[idx].members; nm = ip->ssbos[idx].member_count; }
                else if (I == GL_ATOMIC_COUNTER_BUFFER) { mem = ip->abufs[idx].members; nm = ip->abufs[idx].member_count; uniform_idx = true; }
                else if (I == GL_TRANSFORM_FEEDBACK_BUFFER) {
                    for (int i = 0; i < ip->nxfb && ntmp < 1024; i++) {
                        XfbInfo x = xfb_info(ip, i);
                        if (x.buffer == xfb_buffer_binding(idx) && x.offset >= 0 && strncmp(xfb_name(i), "gl_", 3)) tmp[ntmp++] = i;
                    }
                    mem = tmp; nm = ntmp;
                }
                else { ok = false; break; }
                if (prop == GL_NUM_ACTIVE_VARIABLES) PUSH(nm);
                else for (int i = 0; i < nm; i++) PUSH(uniform_idx ? pub_index_of(ip, mem[i]) : mem[i]);
                break;
            }
            case GL_REFERENCED_BY_VERTEX_SHADER: case GL_REFERENCED_BY_TESS_CONTROL_SHADER: case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
            case GL_REFERENCED_BY_GEOMETRY_SHADER: case GL_REFERENCED_BY_FRAGMENT_SHADER: case GL_REFERENCED_BY_COMPUTE_SHADER: {
                if (!(I == GL_UNIFORM || I == GL_UNIFORM_BLOCK || I == GL_ATOMIC_COUNTER_BUFFER || I == GL_SHADER_STORAGE_BLOCK ||
                      I == GL_BUFFER_VARIABLE || I == GL_PROGRAM_INPUT || I == GL_PROGRAM_OUTPUT)) { ok = false; break; }
                int s = prop == GL_REFERENCED_BY_VERTEX_SHADER ? _VERTEX_SHADER : prop == GL_REFERENCED_BY_TESS_CONTROL_SHADER ? _TESS_CONTROL_SHADER :
                        prop == GL_REFERENCED_BY_TESS_EVALUATION_SHADER ? _TESS_EVALUATION_SHADER : prop == GL_REFERENCED_BY_GEOMETRY_SHADER ? _GEOMETRY_SHADER :
                        prop == GL_REFERENCED_BY_FRAGMENT_SHADER ? _FRAGMENT_SHADER : _COMPUTE_SHADER;
                PUSH((resource_stages(ip, I, idx) >> s) & 1);
                break;
            }
            case GL_TOP_LEVEL_ARRAY_SIZE: if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].top_size); else ok = false; break;
            case GL_TOP_LEVEL_ARRAY_STRIDE: if (I == GL_BUFFER_VARIABLE) PUSH(ip->bvars[idx].top_stride); else ok = false; break;
            case GL_LOCATION:
                if (I == GL_UNIFORM) PUSH(pub_uniform(ip, idx)->location);
                else if (I == GL_PROGRAM_INPUT) PUSH(ip->inputs[idx].location);
                else if (I == GL_PROGRAM_OUTPUT) PUSH(ip->outputs[idx].location);
                else if (st >= 0 && is_su) PUSH(ip->sub[st].unis[idx].location);
                else ok = false;
                break;
            case GL_LOCATION_INDEX:
                if (I == GL_PROGRAM_OUTPUT) PUSH(ip->last_stage == _FRAGMENT_SHADER && ip->outputs[idx].location >= 0 ? ip->outputs[idx].index : -1);
                else ok = false;
                break;
            case GL_LOCATION_COMPONENT:
                if (I == GL_PROGRAM_INPUT) PUSH(ip->inputs[idx].component);
                else if (I == GL_PROGRAM_OUTPUT) PUSH(ip->outputs[idx].component);
                else ok = false;
                break;
            case GL_IS_PER_PATCH:
                if (I == GL_PROGRAM_INPUT) PUSH(ip->inputs[idx].is_per_patch);
                else if (I == GL_PROGRAM_OUTPUT) PUSH(ip->outputs[idx].is_per_patch);
                else ok = false;
                break;
            case GL_NUM_COMPATIBLE_SUBROUTINES:
                if (st >= 0 && is_su) PUSH(ip->sub[st].unis[idx].ncompat); else ok = false;
                break;
            case GL_COMPATIBLE_SUBROUTINES:
                if (st >= 0 && is_su) { for (int i = 0; i < ip->sub[st].unis[idx].ncompat; i++) PUSH(ip->sub[st].unis[idx].compat[i]); }
                else ok = false;
                break;
            case GL_TRANSFORM_FEEDBACK_BUFFER_INDEX:
                if (I == GL_TRANSFORM_FEEDBACK_VARYING) PUSH(xfb_info(ip, idx).buffer); else ok = false;
                break;
            case GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE:
                if (I == GL_TRANSFORM_FEEDBACK_BUFFER) PUSH(xfb_buffer_stride(ip, idx)); else ok = false;
                break;
            default:
                mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return;
        }
#undef PUSH
        if (!ok) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return; }
    }
    GLsizei w = nv < count ? nv : count;
    if (params && w > 0) memcpy(params, vals, (size_t)w * sizeof(GLint));
    if (length) *length = w;
}

GLint  mglGetProgramResourceLocation(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return -1;
    int is_su, st = subroutine_iface(programInterface, &is_su);
    if (!(programInterface == GL_UNIFORM || programInterface == GL_PROGRAM_INPUT || programInterface == GL_PROGRAM_OUTPUT ||
          (st >= 0 && is_su))) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return -1; }
    if (!ip->linked) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return -1; }
    if (!name || !strncmp(name, "gl_", 3)) return -1;
    switch (programInterface) {
        case GL_UNIFORM: return mgli_uniform_location(ip, name);
        case GL_PROGRAM_INPUT: return var_location(ip->inputs, ip->ninputs, name, NULL);
        case GL_PROGRAM_OUTPUT: return var_location(ip->outputs, ip->noutputs, name, NULL);
        default:
            for (int i = 0; i < ip->sub[st].nunis; i++) {
                MGLISubUniform *su = &ip->sub[st].unis[i];
                if (!strcmp(su->name, name)) return su->location;
                size_t ln = strlen(su->name);
                if (!strncmp(name, su->name, ln) && name[ln] == '[') { long e = strtol(name + ln + 1, NULL, 10); if (e >= 0 && e < su->size) return su->location + (GLint)e; }
            }
            return -1;
    }
}

GLint  mglGetProgramResourceLocationIndex(GLMContext ctx, GLuint program, GLenum programInterface, const GLchar *name)
{
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return -1;
    if (programInterface != GL_PROGRAM_OUTPUT) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return -1; }
    if (!ip->linked) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return -1; }
    if (!name || !frag_outputs(ip)) return -1;
    GLint idx = -1;
    if (var_location(ip->outputs, ip->noutputs, name, &idx) < 0) return -1;
    return idx < 0 ? 0 : idx;
}

#pragma mark - subroutines

static MGLIStageSub *stage_sub_checked(GLMContext ctx, const char *func, GLuint program, GLenum shadertype, MGLIProgram **ipo)
{
    int st = stage_of_shadertype(shadertype);
    if (st < 0) { mgli_set_error(ctx, func, GL_INVALID_ENUM); return NULL; }
    MGLIProgram *ip = mgli_program_checked(ctx, program, false);
    if (!ip) return NULL;
    if (ipo) *ipo = ip;
    return &ip->sub[st];
}

void mglGetProgramStageiv(GLMContext ctx, GLuint program, GLenum shadertype, GLenum pname, GLint *values)
{
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, NULL);
    if (!s) return;
    GLint v = 0;
    switch (pname) {
        case GL_ACTIVE_SUBROUTINE_UNIFORMS: v = s->nunis; break;
        case GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS: v = s->nlocs; break;
        case GL_ACTIVE_SUBROUTINES: v = s->nfuncs; break;
        case GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH:
            for (int i = 0; i < s->nunis; i++) { GLint l = (GLint)strlen(s->unis[i].name) + 1; if (l > v) v = l; }
            break;
        case GL_ACTIVE_SUBROUTINE_MAX_LENGTH:
            for (int i = 0; i < s->nfuncs; i++) { GLint l = (GLint)strlen(s->funcs[i].name) + 1; if (l > v) v = l; }
            break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return;
    }
    if (values) *values = v;
}

GLuint mglGetSubroutineIndex(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
    MGLIProgram *ip = NULL;
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, &ip);
    if (!s || !name) return GL_INVALID_INDEX;
    if (!ip->linked) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return GL_INVALID_INDEX; }
    for (int i = 0; i < s->nfuncs; i++) if (!strcmp(s->funcs[i].name, name)) return (GLuint)s->funcs[i].index;
    return GL_INVALID_INDEX;
}

GLint mglGetSubroutineUniformLocation(GLMContext ctx, GLuint program, GLenum shadertype, const GLchar *name)
{
    MGLIProgram *ip = NULL;
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, &ip);
    if (!s || !name) return -1;
    if (!ip->linked) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_OPERATION); return -1; }
    for (int i = 0; i < s->nunis; i++) {
        MGLISubUniform *su = &s->unis[i];
        if (!strcmp(su->name, name)) return su->location;
        size_t ln = strlen(su->name);
        if (!strncmp(name, su->name, ln) && name[ln] == '[') { long e = strtol(name + ln + 1, NULL, 10); if (e >= 0 && e < su->size) return su->location + (GLint)e; }
    }
    return -1;
}

static void copy_str(const char *src, GLsizei bufSize, GLsizei *length, GLchar *dst)
{
    GLsizei n = 0;
    if (dst && bufSize > 0) { n = (GLsizei)strlen(src); if (n > bufSize - 1) n = bufSize - 1; memcpy(dst, src, (size_t)n); dst[n] = 0; }
    if (length) *length = n;
}

void mglGetActiveSubroutineName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, NULL);
    if (!s) return;
    if (bufSize < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    int slot = sub_func_slot(s, index);
    if (slot < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    copy_str(s->funcs[slot].name, bufSize, length, name);
}

void mglGetActiveSubroutineUniformName(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei *length, GLchar *name)
{
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, NULL);
    if (!s) return;
    if (bufSize < 0 || index >= (GLuint)s->nunis) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    copy_str(s->unis[index].name, bufSize, length, name);
}

void mglGetActiveSubroutineUniformiv(GLMContext ctx, GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint *values)
{
    MGLIStageSub *s = stage_sub_checked(ctx, __FUNCTION__, program, shadertype, NULL);
    if (!s) return;
    if (index >= (GLuint)s->nunis) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    MGLISubUniform *u = &s->unis[index];
    switch (pname) {
        case GL_NUM_COMPATIBLE_SUBROUTINES: if (values) *values = u->ncompat; break;
        case GL_COMPATIBLE_SUBROUTINES: if (values) for (int i = 0; i < u->ncompat; i++) values[i] = u->compat[i]; break;
        case GL_UNIFORM_SIZE: if (values) *values = u->size; break;
        case GL_UNIFORM_NAME_LENGTH: if (values) *values = (GLint)strlen(u->name) + 1; break;
        default: mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); break;
    }
}

static MGLIProgram *current_linked(GLMContext ctx, const char *func)
{
    MGLIProgram *ip = find_by_ptr(ctx, ctx->state.program);
    if (!ctx->state.program || !ip || !ip->linked) { mgli_set_error(ctx, func, GL_INVALID_OPERATION); return NULL; }
    return ip;
}

void mglUniformSubroutinesuiv(GLMContext ctx, GLenum shadertype, GLsizei count, const GLuint *indices)
{
    int st = stage_of_shadertype(shadertype);
    if (st < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return; }
    MGLIProgram *ip = current_linked(ctx, __FUNCTION__);
    if (!ip) return;
    MGLIStageSub *s = &ip->sub[st];
    if (count != s->nlocs) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    if (!count) return;
    if (!indices) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    /* every index must name a subroutine compatible with the uniform at that location */
    for (int i = 0; i < s->nunis; i++)
        for (int e = 0; e < s->unis[i].size; e++) {
            GLuint v = indices[s->unis[i].location + e];
            bool ok = false;
            for (int k = 0; k < s->unis[i].ncompat; k++) if ((GLuint)s->unis[i].compat[k] == v) ok = true;
            if (!ok) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
        }
    memcpy(s->selection, indices, (size_t)count * sizeof(GLuint));
    write_selectors(ctx, ip, st);
}

void mglGetUniformSubroutineuiv(GLMContext ctx, GLenum shadertype, GLint location, GLuint *params)
{
    int st = stage_of_shadertype(shadertype);
    if (st < 0) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_ENUM); return; }
    MGLIProgram *ip = current_linked(ctx, __FUNCTION__);
    if (!ip) return;
    if (location < 0 || location >= ip->sub[st].nlocs) { mgli_set_error(ctx, __FUNCTION__, GL_INVALID_VALUE); return; }
    if (params) *params = ip->sub[st].selection[location];
}

#pragma mark - glGetUniform* / glGetnUniform*

void mglGetUniformdv(GLMContext ctx, GLuint program, GLint location, GLdouble *params)
{ mgli_get_uniform(ctx, __FUNCTION__, program, location, -1, MGLI_SET_D, params); }
void mglGetUniformuiv(GLMContext ctx, GLuint program, GLint location, GLuint *params)
{ mgli_get_uniform(ctx, __FUNCTION__, program, location, -1, MGLI_SET_UI, params); }

static bool nbuf_ok(GLMContext ctx, const char *f, GLsizei b) { if (b < 0) { mgli_set_error(ctx, f, GL_INVALID_VALUE); return false; } return true; }
void mglGetnUniformdv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLdouble *params)
{ if (nbuf_ok(ctx, __FUNCTION__, bufSize)) mgli_get_uniform(ctx, __FUNCTION__, program, location, bufSize, MGLI_SET_D, params); }
void mglGetnUniformfv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLfloat *params)
{ if (nbuf_ok(ctx, __FUNCTION__, bufSize)) mgli_get_uniform(ctx, __FUNCTION__, program, location, bufSize, MGLI_SET_F, params); }
void mglGetnUniformiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLint *params)
{ if (nbuf_ok(ctx, __FUNCTION__, bufSize)) mgli_get_uniform(ctx, __FUNCTION__, program, location, bufSize, MGLI_SET_I, params); }
void mglGetnUniformuiv(GLMContext ctx, GLuint program, GLint location, GLsizei bufSize, GLuint *params)
{ if (nbuf_ok(ctx, __FUNCTION__, bufSize)) mgli_get_uniform(ctx, __FUNCTION__, program, location, bufSize, MGLI_SET_UI, params); }

#pragma mark - glProgramUniform*

#define PU_VEC(N, SUF, CT, K) \
void mglProgramUniform##N##SUF##v(GLMContext ctx, GLuint program, GLint location, GLsizei count, const CT *value) \
{ MGLIProgram *ip = mgli_program_checked(ctx, program, false); if (!ip) return; \
  mgli_set_uniform(ctx, __FUNCTION__, ip, location, count, K, 1, N, GL_FALSE, value); }

#define PU_MAT(NAME, C, R, CT, K) \
void mglProgramUniformMatrix##NAME(GLMContext ctx, GLuint program, GLint location, GLsizei count, GLboolean transpose, const CT *value) \
{ MGLIProgram *ip = mgli_program_checked(ctx, program, false); if (!ip) return; \
  mgli_set_uniform(ctx, __FUNCTION__, ip, location, count, K, C, R, transpose, value); }

PU_VEC(1, f, GLfloat, MGLI_SET_F) PU_VEC(2, f, GLfloat, MGLI_SET_F) PU_VEC(3, f, GLfloat, MGLI_SET_F) PU_VEC(4, f, GLfloat, MGLI_SET_F)
PU_VEC(1, i, GLint, MGLI_SET_I)   PU_VEC(2, i, GLint, MGLI_SET_I)   PU_VEC(3, i, GLint, MGLI_SET_I)   PU_VEC(4, i, GLint, MGLI_SET_I)
PU_VEC(1, ui, GLuint, MGLI_SET_UI) PU_VEC(2, ui, GLuint, MGLI_SET_UI) PU_VEC(3, ui, GLuint, MGLI_SET_UI) PU_VEC(4, ui, GLuint, MGLI_SET_UI)
PU_VEC(1, d, GLdouble, MGLI_SET_D) PU_VEC(2, d, GLdouble, MGLI_SET_D) PU_VEC(3, d, GLdouble, MGLI_SET_D) PU_VEC(4, d, GLdouble, MGLI_SET_D)

void mglProgramUniform1f(GLMContext ctx, GLuint program, GLint location, GLfloat v0) { GLfloat v[1] = { v0 }; mglProgramUniform1fv(ctx, program, location, 1, v); }
void mglProgramUniform2f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1) { GLfloat v[2] = { v0, v1 }; mglProgramUniform2fv(ctx, program, location, 1, v); }
void mglProgramUniform3f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2) { GLfloat v[3] = { v0, v1, v2 }; mglProgramUniform3fv(ctx, program, location, 1, v); }
void mglProgramUniform4f(GLMContext ctx, GLuint program, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) { GLfloat v[4] = { v0, v1, v2, v3 }; mglProgramUniform4fv(ctx, program, location, 1, v); }
void mglProgramUniform1i(GLMContext ctx, GLuint program, GLint location, GLint v0) { GLint v[1] = { v0 }; mglProgramUniform1iv(ctx, program, location, 1, v); }
void mglProgramUniform2i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1) { GLint v[2] = { v0, v1 }; mglProgramUniform2iv(ctx, program, location, 1, v); }
void mglProgramUniform3i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2) { GLint v[3] = { v0, v1, v2 }; mglProgramUniform3iv(ctx, program, location, 1, v); }
void mglProgramUniform4i(GLMContext ctx, GLuint program, GLint location, GLint v0, GLint v1, GLint v2, GLint v3) { GLint v[4] = { v0, v1, v2, v3 }; mglProgramUniform4iv(ctx, program, location, 1, v); }
void mglProgramUniform1ui(GLMContext ctx, GLuint program, GLint location, GLuint v0) { GLuint v[1] = { v0 }; mglProgramUniform1uiv(ctx, program, location, 1, v); }
void mglProgramUniform2ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1) { GLuint v[2] = { v0, v1 }; mglProgramUniform2uiv(ctx, program, location, 1, v); }
void mglProgramUniform3ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2) { GLuint v[3] = { v0, v1, v2 }; mglProgramUniform3uiv(ctx, program, location, 1, v); }
void mglProgramUniform4ui(GLMContext ctx, GLuint program, GLint location, GLuint v0, GLuint v1, GLuint v2, GLuint v3) { GLuint v[4] = { v0, v1, v2, v3 }; mglProgramUniform4uiv(ctx, program, location, 1, v); }
void mglProgramUniform1d(GLMContext ctx, GLuint program, GLint location, GLdouble v0) { GLdouble v[1] = { v0 }; mglProgramUniform1dv(ctx, program, location, 1, v); }
void mglProgramUniform2d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1) { GLdouble v[2] = { v0, v1 }; mglProgramUniform2dv(ctx, program, location, 1, v); }
void mglProgramUniform3d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2) { GLdouble v[3] = { v0, v1, v2 }; mglProgramUniform3dv(ctx, program, location, 1, v); }
void mglProgramUniform4d(GLMContext ctx, GLuint program, GLint location, GLdouble v0, GLdouble v1, GLdouble v2, GLdouble v3) { GLdouble v[4] = { v0, v1, v2, v3 }; mglProgramUniform4dv(ctx, program, location, 1, v); }

PU_MAT(2fv, 2, 2, GLfloat, MGLI_SET_F)   PU_MAT(3fv, 3, 3, GLfloat, MGLI_SET_F)   PU_MAT(4fv, 4, 4, GLfloat, MGLI_SET_F)
PU_MAT(2x3fv, 2, 3, GLfloat, MGLI_SET_F) PU_MAT(3x2fv, 3, 2, GLfloat, MGLI_SET_F) PU_MAT(2x4fv, 2, 4, GLfloat, MGLI_SET_F)
PU_MAT(4x2fv, 4, 2, GLfloat, MGLI_SET_F) PU_MAT(3x4fv, 3, 4, GLfloat, MGLI_SET_F) PU_MAT(4x3fv, 4, 3, GLfloat, MGLI_SET_F)
PU_MAT(2dv, 2, 2, GLdouble, MGLI_SET_D)   PU_MAT(3dv, 3, 3, GLdouble, MGLI_SET_D)   PU_MAT(4dv, 4, 4, GLdouble, MGLI_SET_D)
PU_MAT(2x3dv, 2, 3, GLdouble, MGLI_SET_D) PU_MAT(3x2dv, 3, 2, GLdouble, MGLI_SET_D) PU_MAT(2x4dv, 2, 4, GLdouble, MGLI_SET_D)
PU_MAT(4x2dv, 4, 2, GLdouble, MGLI_SET_D) PU_MAT(3x4dv, 3, 4, GLdouble, MGLI_SET_D) PU_MAT(4x3dv, 4, 3, GLdouble, MGLI_SET_D)
