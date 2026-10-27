/*
 * queries.c
 * MGL — query objects, timers, pipeline statistics, conditional rendering
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 *
 * GL-side query object management and spec error checking.  GPU-backed query
 * targets (occlusion family, FRAGMENT_SHADER_INVOCATIONS, TIME_ELAPSED, TIMESTAMP)
 * are serviced by ctx->mtl_funcs.ext_queries (MGLRenderer+Queries.m); all other
 * targets are counted CPU-side from draw parameters through the pre-draw hook.
 *
 * Exactness of the CPU-counted statistics (no hardware statistics on Metal):
 *   exact:   VERTICES_SUBMITTED, PRIMITIVES_SUBMITTED, CLIPPING_INPUT_PRIMITIVES,
 *            PRIMITIVES_GENERATED (no geometry/tessellation stage), COMPUTE_SHADER_INVOCATIONS,
 *            TRANSFORM_FEEDBACK_* (as reported by the xfb domain)
 *   upper bound: VERTEX_SHADER_INVOCATIONS (= vertices submitted, no post-transform cache)
 *   approximate: CLIPPING_OUTPUT_PRIMITIVES (= input, clipping not observed),
 *            GEOMETRY_SHADER_INVOCATIONS / _PRIMITIVES_EMITTED (= input primitives),
 *            TESS_CONTROL_SHADER_PATCHES (= patches), TESS_EVALUATION_SHADER_INVOCATIONS
 *            (= patches * 3), PRIMITIVES_GENERATED with GS/tess (= input primitives),
 *            FRAGMENT_SHADER_INVOCATIONS (= samples passing depth/stencil, GPU visibility count)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_queries.h"
#include "mgl_ext_geom_xfb.h"
#include "mgl_ext_buffers.h"   /* CPU-ordered element reads for restart counting */

uint64_t mgl_prof_begin(int id);
void mgl_prof_end(int id, uint64_t t0);

extern Buffer *findBuffer(GLMContext ctx, GLuint buffer);

#define QERR(_e_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return; } while (0)
#define QERR_VAL(_e_, _v_) do { ctx->error_func(ctx, __FUNCTION__, (_e_)); return (_v_); } while (0)

#define QS(ctx) ((ctx)->state.ext_queries)
#define QMTL(ctx) ((ctx)->mtl_funcs.ext_queries)

enum { QK_I32 = 0, QK_U32, QK_I64, QK_U64 };

#pragma mark - target helpers

static bool target_is_indexed(GLenum target)
{
    switch (target) {
        case GL_PRIMITIVES_GENERATED:
        case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN:
        case GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW:
            return true;
    }
    return false;
}

static bool target_is_occlusion(GLenum target)
{
    return target == GL_SAMPLES_PASSED || target == GL_ANY_SAMPLES_PASSED ||
           target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE;
}

static bool target_is_gpu(GLenum target)
{
    return target_is_occlusion(target) || target == GL_FRAGMENT_SHADER_INVOCATIONS ||
           target == GL_TIME_ELAPSED || target == GL_TIMESTAMP;
}

/* binding point for Begin/End/GetQueryiv targets; -1 if target is not a Begin-able target */
static int binding_for(GLenum target, GLuint index)
{
    switch (target) {
        case GL_SAMPLES_PASSED:
        case GL_ANY_SAMPLES_PASSED:
        case GL_ANY_SAMPLES_PASSED_CONSERVATIVE:          return MGL_QB_OCCLUSION;
        case GL_TIME_ELAPSED:                             return MGL_QB_TIME_ELAPSED;
        case GL_PRIMITIVES_GENERATED:                     return MGL_QB_PRIMITIVES_GENERATED + (int)index;
        case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN:    return MGL_QB_XFB_PRIMITIVES_WRITTEN + (int)index;
        case GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW:       return MGL_QB_XFB_STREAM_OVERFLOW + (int)index;
        case GL_TRANSFORM_FEEDBACK_OVERFLOW:              return MGL_QB_XFB_OVERFLOW;
        case GL_VERTICES_SUBMITTED:                       return MGL_QB_VERTICES_SUBMITTED;
        case GL_PRIMITIVES_SUBMITTED:                     return MGL_QB_PRIMITIVES_SUBMITTED;
        case GL_VERTEX_SHADER_INVOCATIONS:                return MGL_QB_VERTEX_SHADER_INVOCATIONS;
        case GL_TESS_CONTROL_SHADER_PATCHES:              return MGL_QB_TESS_CONTROL_SHADER_PATCHES;
        case GL_TESS_EVALUATION_SHADER_INVOCATIONS:       return MGL_QB_TESS_EVALUATION_SHADER_INVOCATIONS;
        case GL_GEOMETRY_SHADER_INVOCATIONS:              return MGL_QB_GEOMETRY_SHADER_INVOCATIONS;
        case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED:       return MGL_QB_GEOMETRY_SHADER_PRIMITIVES_EMITTED;
        case GL_FRAGMENT_SHADER_INVOCATIONS:              return MGL_QB_FRAGMENT_SHADER_INVOCATIONS;
        case GL_COMPUTE_SHADER_INVOCATIONS:               return MGL_QB_COMPUTE_SHADER_INVOCATIONS;
        case GL_CLIPPING_INPUT_PRIMITIVES:                return MGL_QB_CLIPPING_INPUT_PRIMITIVES;
        case GL_CLIPPING_OUTPUT_PRIMITIVES:               return MGL_QB_CLIPPING_OUTPUT_PRIMITIVES;
    }
    return -1;
}

static bool target_valid_for_create(GLenum target)
{
    return target == GL_TIMESTAMP || binding_for(target, 0) >= 0;
}

/* GL_INVALID_VALUE check for the index of an indexed Begin/End/Get */
static bool index_valid(GLenum target, GLuint index)
{
    if (target_is_indexed(target))
        return index < MGL_QUERY_MAX_VERTEX_STREAMS;
    return index == 0;
}

static MGLQuery *lookup_query(GLMContext ctx, GLuint id)
{
    struct MGLExtState_queries *S = QS(ctx);
    if (!S || id == 0 || id >= S->query_table.size)
        return NULL;
    return (MGLQuery *)searchHashTable(&S->query_table, id);
}

static void free_query(GLMContext ctx, MGLQuery *q)
{
    if (q->gpu && QMTL(ctx) && QMTL(ctx)->release)
        QMTL(ctx)->release(ctx, q);
    q->gpu = NULL;
    free(q);
}

static MGLQuery *new_query_name(GLMContext ctx, GLuint name)
{
    MGLQuery *q = (MGLQuery *)calloc(1, sizeof(MGLQuery));
    if (!q) return NULL;
    q->name = name;
    insertHashElement(&QS(ctx)->query_table, name, q);
    return q;
}

/* attach a target to a reserved name (object creation) */
static void create_object(MGLQuery *q, GLenum target, GLuint index)
{
    q->target = target;
    q->index = index;
    q->result = 0;
    q->result_ready = true;     /* a never-run query reports 0, available */
}

/* resolve a pending result; true if available */
static bool query_resolve(GLMContext ctx, MGLQuery *q, bool wait)
{
    if (q->result_ready)
        return true;
    if (q->active)
        return false;

    if (!target_is_gpu(q->target) || !QMTL(ctx) || !QMTL(ctx)->resolve)
    {
        q->result = q->cpu_count;
        q->result_ready = true;
        return true;
    }

    if (!QMTL(ctx)->resolve(ctx, q, wait))
        return false;

    if (q->target == GL_ANY_SAMPLES_PASSED || q->target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE)
        q->result = q->result ? 1 : 0;
    q->result_ready = true;
    return true;
}

static uint64_t clamp_for_kind(uint64_t v, int kind)
{
    switch (kind) {
        case QK_I32: return v > INT32_MAX ? INT32_MAX : v;
        case QK_U32: return v > UINT32_MAX ? UINT32_MAX : v;
        case QK_I64: return v > INT64_MAX ? INT64_MAX : v;
    }
    return v;
}

static size_t kind_size(int kind)
{
    return (kind == QK_I32 || kind == QK_U32) ? 4 : 8;
}

static void store_kind(void *dst, int kind, uint64_t v)
{
    v = clamp_for_kind(v, kind);
    switch (kind) {
        case QK_I32: { GLint x = (GLint)v; memcpy(dst, &x, 4); break; }
        case QK_U32: { GLuint x = (GLuint)v; memcpy(dst, &x, 4); break; }
        case QK_I64: { GLint64 x = (GLint64)v; memcpy(dst, &x, 8); break; }
        default:     { GLuint64 x = (GLuint64)v; memcpy(dst, &x, 8); break; }
    }
}

static bool buffer_range_ok(Buffer *b, GLintptr offset, int kind)
{
    size_t sz = kind_size(kind);
    if (offset < 0) return false;
    if ((uint64_t)offset + sz > (uint64_t)b->size) return false;
    return true;
}

/* write a query value into a buffer object's data store (CPU backing store + Metal copy) */
static void write_query_buffer(GLMContext ctx, Buffer *b, GLintptr offset, int kind, uint64_t v)
{
    uint8_t bytes[8];
    size_t sz = kind_size(kind);

    store_kind(bytes, kind, v);

    if (b->data.buffer_data && (uint64_t)offset + sz <= b->data.buffer_size)
    {
        memcpy((uint8_t *)b->data.buffer_data + offset, bytes, sz);
        b->data.dirty_bits |= DIRTY_BUFFER_DATA;
        ctx->state.dirty_bits |= DIRTY_BUFFER;
    }

    if (b->data.mtl_data && ctx->mtl_funcs.mtlBufferSubData)
        ctx->mtl_funcs.mtlBufferSubData(ctx, b, (size_t)offset, sz, bytes);
}

#pragma mark - begin / end

static void begin_query(GLMContext ctx, GLenum target, GLuint index, GLuint id)
{
    struct MGLExtState_queries *S = QS(ctx);
    int bp;
    MGLQuery *q;

    bp = binding_for(target, 0);
    if (bp < 0) QERR(GL_INVALID_ENUM);
    if (!index_valid(target, index)) QERR(GL_INVALID_VALUE);
    bp = binding_for(target, index);

    if (S->active[bp]) QERR(GL_INVALID_OPERATION);
    if (id == 0) QERR(GL_INVALID_OPERATION);

    q = lookup_query(ctx, id);
    if (!q) QERR(GL_INVALID_OPERATION);         /* not a name returned by Gen/CreateQueries */
    if (q->active) QERR(GL_INVALID_OPERATION);
    if (q->target != 0 && q->target != target) QERR(GL_INVALID_OPERATION);
    if (S->cond_active && S->cond_query == q) QERR(GL_INVALID_OPERATION);

    if (q->target == 0)
        create_object(q, target, index);
    q->index = index;

    if (q->gpu && QMTL(ctx) && QMTL(ctx)->release)
        QMTL(ctx)->release(ctx, q);

    q->active = true;
    q->result_ready = false;
    q->result = 0;
    q->cpu_count = 0;
    S->active[bp] = q;

    /* a geometry stage counts these on the GPU (per-stream counters folded into a
     * persistent accumulator after every draw): the word restarts at the next draw,
     * which opens this query's counting period.  The clipping counters ride on the
     * stream-0 generated word: what the geometry stage emits is what reaches clipping.
     * GEOMETRY_SHADER_INVOCATIONS is accumulator word 8: the geometry kernel bumps
     * ctl[21] once per (input primitive, invocation) thread — the exact count for
     * indirect draws too, where the CPU-side vertex/instance estimate sees zeros
     * (KHR-GL46.pipeline_statistics_query_tests_ARB.*geometry_shader_queries). */
    if (target == GL_PRIMITIVES_GENERATED || target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED ||
        target == GL_CLIPPING_INPUT_PRIMITIVES || target == GL_CLIPPING_OUTPUT_PRIMITIVES ||
        target == GL_GEOMETRY_SHADER_INVOCATIONS)
    {
        unsigned word = (target == GL_PRIMITIVES_GENERATED) ? index :
                        (target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED) ? 4u + index :
                        (target == GL_GEOMETRY_SHADER_INVOCATIONS) ? 8u : 0u;
        if (ctx->mtl_funcs.ext_geom && ctx->mtl_funcs.ext_geom->query_begin)
            ctx->mtl_funcs.ext_geom->query_begin(ctx, 1u << word);
    }

    if (target_is_gpu(target) && QMTL(ctx) && QMTL(ctx)->begin)
        QMTL(ctx)->begin(ctx, q);
}

static void end_query(GLMContext ctx, GLenum target, GLuint index)
{
    struct MGLExtState_queries *S = QS(ctx);
    int bp;
    MGLQuery *q;

    bp = binding_for(target, 0);
    if (bp < 0) QERR(GL_INVALID_ENUM);
    if (!index_valid(target, index)) QERR(GL_INVALID_VALUE);
    bp = binding_for(target, index);

    /* the geometry domain's transform feedback pack keeps its counters on the GPU: fold them
     * in before the query stops, so the draws it covered are counted.  The overflow targets
     * must flush too: their flag is set from the pack's written/needed pair when it lands
     * (transform_feedback_overflow_query_ARB.multiple-streams-*), which never happens if
     * the counters sit in the block until a later flush. */
    if (target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ||
        target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED || target == GL_GEOMETRY_SHADER_INVOCATIONS ||
        target == GL_TRANSFORM_FEEDBACK_OVERFLOW || target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW)
        mgl_geom_xfb_flush(ctx);

    q = S->active[bp];
    if (!q || q->target != target) QERR(GL_INVALID_OPERATION);

    S->active[bp] = NULL;
    q->active = false;

    /* the geometry stage's exact per-stream counters accumulated since the begin fold
     * into the CPU-tracked part here; queries without a geometry-stage draw read 0 */
    if (target == GL_PRIMITIVES_GENERATED || target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED ||
        target == GL_CLIPPING_INPUT_PRIMITIVES || target == GL_CLIPPING_OUTPUT_PRIMITIVES ||
        target == GL_GEOMETRY_SHADER_INVOCATIONS)
    {
        unsigned word = (target == GL_PRIMITIVES_GENERATED) ? index :
                        (target == GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED) ? 4u + index :
                        (target == GL_GEOMETRY_SHADER_INVOCATIONS) ? 8u : 0u;
        if (ctx->mtl_funcs.ext_geom && ctx->mtl_funcs.ext_geom->query_read)
        {
            uint64_t v = 0;
            ctx->mtl_funcs.ext_geom->query_read(ctx, word, &v);
            q->cpu_count += v;
            if (getenv("MGL_TESS_DEBUG") && target == GL_PRIMITIVES_GENERATED)
                fprintf(stderr, "MGL tess: end PG query id=%u cpu=%llu geom_word%u=%llu\n",
                        q->name, (unsigned long long)q->cpu_count, word, (unsigned long long)v);
        }
    }

    if (target_is_gpu(target) && QMTL(ctx) && QMTL(ctx)->end)
    {
        QMTL(ctx)->end(ctx, q);
    }
    else
    {
        q->result = q->cpu_count;
        q->result_ready = true;
    }

    if (q->deleted)
        free_query(ctx, q);
}

void mglBeginQuery(GLMContext ctx, GLenum target, GLuint id)
{
    begin_query(ctx, target, 0, id);
}

void mglBeginQueryIndexed(GLMContext ctx, GLenum target, GLuint index, GLuint id)
{
    begin_query(ctx, target, index, id);
}

void mglEndQuery(GLMContext ctx, GLenum target)
{
    end_query(ctx, target, 0);
}

void mglEndQueryIndexed(GLMContext ctx, GLenum target, GLuint index)
{
    end_query(ctx, target, index);
}

void mglQueryCounter(GLMContext ctx, GLuint id, GLenum target)
{
    MGLQuery *q;

    if (target != GL_TIMESTAMP) QERR(GL_INVALID_ENUM);

    q = lookup_query(ctx, id);
    if (!q) QERR(GL_INVALID_OPERATION);
    if (q->active) QERR(GL_INVALID_OPERATION);
    if (q->target != 0 && q->target != GL_TIMESTAMP) QERR(GL_INVALID_OPERATION);

    if (q->target == 0)
        create_object(q, GL_TIMESTAMP, 0);

    if (q->gpu && QMTL(ctx) && QMTL(ctx)->release)
        QMTL(ctx)->release(ctx, q);

    q->result_ready = false;
    q->result = 0;

    if (QMTL(ctx) && QMTL(ctx)->counter)
    {
        QMTL(ctx)->counter(ctx, q);
    }
    else
    {
        q->result = 0;
        q->result_ready = true;
    }
}

#pragma mark - conditional rendering

static bool cond_mode_valid(GLenum mode)
{
    switch (mode) {
        case GL_QUERY_WAIT:
        case GL_QUERY_NO_WAIT:
        case GL_QUERY_BY_REGION_WAIT:
        case GL_QUERY_BY_REGION_NO_WAIT:
        case GL_QUERY_WAIT_INVERTED:
        case GL_QUERY_NO_WAIT_INVERTED:
        case GL_QUERY_BY_REGION_WAIT_INVERTED:
        case GL_QUERY_BY_REGION_NO_WAIT_INVERTED:
            return true;
    }
    return false;
}

void mglBeginConditionalRender(GLMContext ctx, GLuint id, GLenum mode)
{
    struct MGLExtState_queries *S = QS(ctx);
    MGLQuery *q;

    if (!cond_mode_valid(mode)) QERR(GL_INVALID_ENUM);
    if (S->cond_active) QERR(GL_INVALID_OPERATION);

    q = lookup_query(ctx, id);
    if (!q || q->target == 0) QERR(GL_INVALID_VALUE);
    if (!(target_is_occlusion(q->target) || q->target == GL_TRANSFORM_FEEDBACK_OVERFLOW ||
          q->target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW))
        QERR(GL_INVALID_OPERATION);
    if (q->active) QERR(GL_INVALID_OPERATION);

    S->cond_active = true;
    S->cond_mode = mode;
    S->cond_query = q;
    S->cond_fixed = false;
}

void mglEndConditionalRender(GLMContext ctx)
{
    struct MGLExtState_queries *S = QS(ctx);

    if (!S->cond_active) QERR(GL_INVALID_OPERATION);

    S->cond_active = false;
    S->cond_query = NULL;
    S->cond_fixed = false;
}

/* gpuaudit: is any query active whose CPU counter is fed from the per-draw counts in
 * queries_pre_draw()?  When none is, the draw path does not need the vertex/instance
 * counts of an indirect command and can hand the command buffer straight to Metal
 * without reading it (occlusion / time-elapsed / compute queries are not affected). */
bool mgl_queries_counting_active(GLMContext ctx)
{
    static const int cpu_counted[] = {
        MGL_QB_VERTICES_SUBMITTED,
        MGL_QB_PRIMITIVES_SUBMITTED,
        MGL_QB_VERTEX_SHADER_INVOCATIONS,
        MGL_QB_TESS_CONTROL_SHADER_PATCHES,
        MGL_QB_TESS_EVALUATION_SHADER_INVOCATIONS,
        MGL_QB_GEOMETRY_SHADER_INVOCATIONS,
        MGL_QB_GEOMETRY_SHADER_PRIMITIVES_EMITTED,
        MGL_QB_CLIPPING_INPUT_PRIMITIVES,
        MGL_QB_CLIPPING_OUTPUT_PRIMITIVES,
    };
    struct MGLExtState_queries *S = QS(ctx);
    if (!S)
        return false;

    for (size_t i = 0; i < sizeof(cpu_counted) / sizeof(cpu_counted[0]); i++)
        if (S->active[cpu_counted[i]])
            return true;

    for (int stream = 0; stream < MGL_QUERY_MAX_VERTEX_STREAMS; stream++)
        if (S->active[MGL_QB_PRIMITIVES_GENERATED + stream])
            return true;

    return false;
}

bool mgl_queries_render_allowed(GLMContext ctx)
{
    struct MGLExtState_queries *S = QS(ctx);
    uint64_t r;
    bool wait, inverted, pass;

    if (!S || !S->cond_active)
        return true;

    switch (S->cond_mode) {
        case GL_QUERY_WAIT: case GL_QUERY_BY_REGION_WAIT:
            wait = true; inverted = false; break;
        case GL_QUERY_WAIT_INVERTED: case GL_QUERY_BY_REGION_WAIT_INVERTED:
            wait = true; inverted = true; break;
        case GL_QUERY_NO_WAIT_INVERTED: case GL_QUERY_BY_REGION_NO_WAIT_INVERTED:
            wait = false; inverted = true; break;
        default:
            wait = false; inverted = false; break;
    }

    if (S->cond_fixed)
    {
        r = S->cond_fixed_result;
    }
    else
    {
        if (!S->cond_query || !query_resolve(ctx, S->cond_query, wait))
            return true;    /* NO_WAIT and not available yet: render */
        r = S->cond_query->result;
    }

    pass = (r != 0);
    return inverted ? !pass : pass;
}

#pragma mark - CPU-side counting

static uint64_t primitives_for(GLenum mode, uint64_t n)
{
    switch (mode) {
        case GL_POINTS:                     return n;
        case GL_LINES:                      return n / 2;
        case GL_LINE_STRIP:                 return n >= 2 ? n - 1 : 0;
        case GL_LINE_LOOP:                  return n >= 2 ? n : 0;
        case GL_TRIANGLES:                  return n / 3;
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:               return n >= 3 ? n - 2 : 0;
        case GL_LINES_ADJACENCY:            return n / 4;
        case GL_LINE_STRIP_ADJACENCY:       return n >= 4 ? n - 3 : 0;
        case GL_TRIANGLES_ADJACENCY:        return n / 6;
        case GL_TRIANGLE_STRIP_ADJACENCY:   return n >= 6 ? (n - 4) / 2 : 0;
        case GL_PATCHES:                    return n / 3;   /* default GL_PATCH_VERTICES */
    }
    return 0;
}

static bool program_has_stage(GLMContext ctx, int stage)
{
    Program *p = ctx->state.program;
    if (p)
        return p->shader_slots[stage] != NULL;
    if (ctx->state.program_pipeline)
    {
        Program *sp = ctx->state.program_pipeline->stage_programs[stage];
        return sp && sp->shader_slots[stage] != NULL;
    }
    return false;
}

static inline void count_active(struct MGLExtState_queries *S, int bp, uint64_t v)
{
    if (S->active[bp])
        S->active[bp]->cpu_count += v;
}

/* exact vertex / primitive counts for an indexed draw with primitive restart enabled:
 * the index stream is split at restart indices, the restart indices themselves are not
 * submitted, and incomplete primitives at a restart are discarded.  Returns false when
 * the element range is not CPU-visible right now (caller falls back to the estimate). */
static bool restart_index_counts(GLMContext ctx, GLenum mode, GLenum type, size_t offset,
                                 GLsizei count, uint64_t *vertices_out, uint64_t *prims_out)
{
    uint32_t restart = 0, index;
    uint64_t verts = 0, prims = 0, seg = 0;
    VertexArray *vao = ctx->state.vao;
    Buffer *eb = vao ? vao->element_array.buffer : NULL;
    const uint8_t *base;
    int tsize;

    if (ctx->state.caps.primitive_restart_fixed_index)
        restart = type == GL_UNSIGNED_BYTE ? 0xFFu : type == GL_UNSIGNED_SHORT ? 0xFFFFu : 0xFFFFFFFFu;
    else if (ctx->state.caps.primitive_restart)
        restart = ctx->state.var.primitive_restart_index;
    else
        return false;

    if (count <= 0)
        return true;    /* nothing submitted: both counts stay 0 */

    if (eb)
    {
        /* GPU-written element data must be visible before the CPU scan */
        MGLBufferTrack *t = mgl_buffers_track(ctx, eb, false);
        if (t && ctx->mtl_funcs.ext_buffers && t->write_serial > ctx->mtl_funcs.ext_buffers->completed_serial(ctx))
            mgl_buffers_prepare_cpu_read(ctx, eb);
        base = (const uint8_t *)eb->data.buffer_data;
        if (!base)
            return false;
        if (offset > (uint64_t)eb->size)
            return false;
    }
    else
    {
        base = (const uint8_t *)(uintptr_t)offset;    /* client index pointer */
        offset = 0;
    }

    tsize = type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4;
    if (eb && (uint64_t)offset + (uint64_t)count * tsize > (uint64_t)eb->size)
        return false;

    base += offset;
    for (GLsizei i = 0; i < count; i++)
    {
        switch (type) {
            case GL_UNSIGNED_BYTE:  index = base[i]; break;
            case GL_UNSIGNED_SHORT: index = ((const uint16_t *)base)[i]; break;
            default:                index = ((const uint32_t *)base)[i]; break;
        }

        if (index == restart)
        {
            prims += primitives_for(mode, seg);
            seg = 0;
            continue;
        }

        verts++;
        seg++;
    }
    prims += primitives_for(mode, seg);

    *vertices_out = verts;
    *prims_out = prims;
    return true;
}

static bool queries_pre_draw(GLMContext ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count,
                             bool indexed, GLenum type, size_t offset)
{
    struct MGLExtState_queries *S = QS(ctx);
    uint64_t inst, verts, prims;
    bool has_gs, has_tess;
    uint64_t t26 = mgl_prof_begin(26);

    if (!S)
        return true;

    if (!mgl_queries_render_allowed(ctx))
    {
        mgl_prof_end(26, t26);
        return false;   /* conditional rendering discards the draw */
    }

    if (vertex_count <= 0 || instance_count <= 0)
        return true;

    inst = (uint64_t)instance_count;

    if (indexed && (ctx->state.caps.primitive_restart || ctx->state.caps.primitive_restart_fixed_index))
    {
        /* exact counts: split the index stream at restart indices */
        uint64_t rverts = 0, rprims = 0;
        if (restart_index_counts(ctx, mode, type, offset, vertex_count, &rverts, &rprims))
        {
            verts = rverts * inst;
            prims = rprims * inst;
        }
        else
        {
            verts = (uint64_t)vertex_count * inst;
            prims = primitives_for(mode, (uint64_t)vertex_count) * inst;
        }
    }
    else
    {
        verts = (uint64_t)vertex_count * inst;
        prims = primitives_for(mode, (uint64_t)vertex_count) * inst;
    }

    has_gs = program_has_stage(ctx, _GEOMETRY_SHADER);
    has_tess = program_has_stage(ctx, _TESS_EVALUATION_SHADER) && mode == GL_PATCHES;

    count_active(S, MGL_QB_VERTICES_SUBMITTED, verts);
    count_active(S, MGL_QB_PRIMITIVES_SUBMITTED, prims);
    count_active(S, MGL_QB_VERTEX_SHADER_INVOCATIONS, verts);
    if (has_tess)
    {
        if (program_has_stage(ctx, _TESS_CONTROL_SHADER))
            count_active(S, MGL_QB_TESS_CONTROL_SHADER_PATCHES, prims);
        /* TES invocations and the generated/clipped primitive counts depend on the
           tessellation factors: the tess domain reports the exact numbers after
           tessellating (mgl_queries_note_tess) */
    }
    else if (!has_gs)
    {
        count_active(S, MGL_QB_CLIPPING_INPUT_PRIMITIVES, prims);
        count_active(S, MGL_QB_CLIPPING_OUTPUT_PRIMITIVES, prims);
        /* without a geometry shader every primitive goes to vertex stream 0; with one
         * the counts come from the kernel's per-stream counters (see end_query) */
        count_active(S, MGL_QB_PRIMITIVES_GENERATED + 0, prims);
    }
    if (has_gs)
    {
        /* GEOMETRY_SHADER_PRIMITIVES_EMITTED and PRIMITIVES_GENERATED come from the
         * geometry kernel's per-stream counters, read through the ext_geom query hooks
         * (begin/end query); so does GEOMETRY_SHADER_INVOCATIONS (accumulator word 8,
         * ctl[21]: one bump per (input primitive, invocation) thread).  The CPU-side
         * estimate would double-count — and is zero for indirect draws whose
         * vertex/instance counts never reach the CPU (mgl_geom_route_indirect). */
    }

    mgl_prof_end(26, t26);
    return true;
}

void mgl_queries_note_tess(GLMContext ctx, uint64_t tes_invocations, uint64_t tessellated_primitives)
{
    struct MGLExtState_queries *S = QS(ctx);
    if (!S)
        return;
    count_active(S, MGL_QB_TESS_EVALUATION_SHADER_INVOCATIONS, tes_invocations);
    count_active(S, MGL_QB_CLIPPING_INPUT_PRIMITIVES, tessellated_primitives);
    count_active(S, MGL_QB_CLIPPING_OUTPUT_PRIMITIVES, tessellated_primitives);
    count_active(S, MGL_QB_PRIMITIVES_GENERATED + 0, tessellated_primitives);
}

void mgl_queries_note_dispatch(GLMContext ctx, uint64_t invocations)
{
    struct MGLExtState_queries *S = QS(ctx);
    if (S)
        count_active(S, MGL_QB_COMPUTE_SHADER_INVOCATIONS, invocations);
}

bool mgl_queries_tess_counts_active(GLMContext ctx)
{
    struct MGLExtState_queries *S = QS(ctx);
    if (!S)
        return false;
    return S->active[MGL_QB_TESS_EVALUATION_SHADER_INVOCATIONS] != NULL ||
           S->active[MGL_QB_CLIPPING_INPUT_PRIMITIVES] != NULL ||
           S->active[MGL_QB_CLIPPING_OUTPUT_PRIMITIVES] != NULL ||
           S->active[MGL_QB_PRIMITIVES_GENERATED + 0] != NULL;
}

void mgl_queries_note_xfb_primitives(GLMContext ctx, GLuint stream, uint64_t written, uint64_t needed)
{
    struct MGLExtState_queries *S = QS(ctx);
    if (!S || stream >= MGL_QUERY_MAX_VERTEX_STREAMS)
        return;
    count_active(S, MGL_QB_XFB_PRIMITIVES_WRITTEN + (int)stream, written);
    if (needed > written)
    {
        if (S->active[MGL_QB_XFB_STREAM_OVERFLOW + stream])
            S->active[MGL_QB_XFB_STREAM_OVERFLOW + stream]->cpu_count = 1;
        if (S->active[MGL_QB_XFB_OVERFLOW])
            S->active[MGL_QB_XFB_OVERFLOW]->cpu_count = 1;
    }
}

#pragma mark - object management

void mglGenQueries(GLMContext ctx, GLsizei n, GLuint *ids)
{
    if (n < 0) QERR(GL_INVALID_VALUE);
    if (n > 0 && ids == NULL) return;

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name = getNewName(&QS(ctx)->query_table);
        if (!new_query_name(ctx, name)) QERR(GL_OUT_OF_MEMORY);
        ids[i] = name;
    }
}

void mglCreateQueries(GLMContext ctx, GLenum target, GLsizei n, GLuint *ids)
{
    if (!target_valid_for_create(target)) QERR(GL_INVALID_ENUM);
    if (n < 0) QERR(GL_INVALID_VALUE);
    if (n > 0 && ids == NULL) return;

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name = getNewName(&QS(ctx)->query_table);
        MGLQuery *q = new_query_name(ctx, name);
        if (!q) QERR(GL_OUT_OF_MEMORY);
        create_object(q, target, 0);
        ids[i] = name;
    }
}

void mglDeleteQueries(GLMContext ctx, GLsizei n, const GLuint *ids)
{
    struct MGLExtState_queries *S = QS(ctx);

    if (n < 0) QERR(GL_INVALID_VALUE);
    if (n > 0 && ids == NULL) return;

    for (GLsizei i = 0; i < n; i++)
    {
        MGLQuery *q = lookup_query(ctx, ids[i]);
        if (!q)
            continue;   /* unused names and 0 are silently ignored */

        deleteHashElement(&S->query_table, ids[i]);

        if (S->cond_active && S->cond_query == q)
        {
            /* freeze the condition: the object goes away under an active conditional render */
            bool wait = !(S->cond_mode == GL_QUERY_NO_WAIT || S->cond_mode == GL_QUERY_BY_REGION_NO_WAIT ||
                          S->cond_mode == GL_QUERY_NO_WAIT_INVERTED || S->cond_mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED);
            if (query_resolve(ctx, q, wait))
                S->cond_fixed_result = q->result;
            else
                S->cond_fixed_result = 1;   /* NO_WAIT, unavailable: keep rendering */
            S->cond_fixed = true;
            S->cond_query = NULL;
        }

        if (q->active)
        {
            /* name becomes unused now; the object lives on until EndQuery */
            q->deleted = true;
            q->name = 0;
            continue;
        }

        free_query(ctx, q);
    }
}

GLboolean mglIsQuery(GLMContext ctx, GLuint id)
{
    MGLQuery *q = lookup_query(ctx, id);
    return (q && q->target != 0) ? GL_TRUE : GL_FALSE;
}

#pragma mark - state queries

static void get_query_indexed(GLMContext ctx, GLenum target, GLuint index, GLenum pname, GLint *params)
{
    struct MGLExtState_queries *S = QS(ctx);
    int bp;

    if (target == GL_TIMESTAMP)
    {
        if (index != 0) QERR(GL_INVALID_VALUE);
        if (pname != GL_QUERY_COUNTER_BITS) QERR(GL_INVALID_ENUM);
        if (params) *params = 64;
        return;
    }

    bp = binding_for(target, 0);
    if (bp < 0) QERR(GL_INVALID_ENUM);
    if (!index_valid(target, index)) QERR(GL_INVALID_VALUE);
    bp = binding_for(target, index);

    switch (pname) {
        case GL_CURRENT_QUERY: {
            MGLQuery *q = S->active[bp];
            if (params) *params = (q && q->target == target && !q->deleted) ? (GLint)q->name : 0;
            break;
        }
        case GL_QUERY_COUNTER_BITS:
            if (params) *params = 64;
            break;
        default:
            QERR(GL_INVALID_ENUM);
    }
}

void mglGetQueryiv(GLMContext ctx, GLenum target, GLenum pname, GLint *params)
{
    get_query_indexed(ctx, target, 0, pname, params);
}

void mglGetQueryIndexediv(GLMContext ctx, GLenum target, GLuint index, GLenum pname, GLint *params)
{
    get_query_indexed(ctx, target, index, pname, params);
}

static bool query_pname_valid(GLenum pname)
{
    return pname == GL_QUERY_RESULT || pname == GL_QUERY_RESULT_NO_WAIT ||
           pname == GL_QUERY_RESULT_AVAILABLE || pname == GL_QUERY_TARGET;
}

/* common body for GetQueryObject* and GetQueryBufferObject*; buf != NULL writes to buf at offset */
static void get_query_object(GLMContext ctx, const char *func, GLuint id, GLenum pname, int kind,
                             void *params, Buffer *buf, GLintptr offset)
{
    /* the geometry domain's transform feedback pack keeps its counters on the GPU until the
     * application can observe them */
    mgl_geom_xfb_flush(ctx);
    MGLQuery *q = lookup_query(ctx, id);
    uint64_t v = 0;
    bool write = true;

    if (!q || q->target == 0) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }
    if (!query_pname_valid(pname)) { ctx->error_func(ctx, func, GL_INVALID_ENUM); return; }
    if (q->active) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }
    if (buf && !buffer_range_ok(buf, offset, kind)) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }

    switch (pname) {
        case GL_QUERY_RESULT:
            query_resolve(ctx, q, true);
            v = q->result;
            break;
        case GL_QUERY_RESULT_NO_WAIT:
            write = query_resolve(ctx, q, false);
            v = q->result;
            break;
        case GL_QUERY_RESULT_AVAILABLE:
            v = query_resolve(ctx, q, false) ? GL_TRUE : GL_FALSE;
            break;
        case GL_QUERY_TARGET:
            v = q->target;
            break;
    }

    if (!write)
        return;

    if (buf)
        write_query_buffer(ctx, buf, offset, kind, v);
    else if (params)
        store_kind(params, kind, v);
}

static void get_query_object_client(GLMContext ctx, const char *func, GLuint id, GLenum pname, int kind, void *params)
{
    Buffer *qb = ctx->state.buffers[_QUERY_BUFFER];
    if (qb)
        get_query_object(ctx, func, id, pname, kind, NULL, qb, (GLintptr)params);
    else
        get_query_object(ctx, func, id, pname, kind, params, NULL, 0);
}

void mglGetQueryObjectiv(GLMContext ctx, GLuint id, GLenum pname, GLint *params)
{
    get_query_object_client(ctx, __FUNCTION__, id, pname, QK_I32, params);
}

void mglGetQueryObjectuiv(GLMContext ctx, GLuint id, GLenum pname, GLuint *params)
{
    get_query_object_client(ctx, __FUNCTION__, id, pname, QK_U32, params);
}

void mglGetQueryObjecti64v(GLMContext ctx, GLuint id, GLenum pname, GLint64 *params)
{
    get_query_object_client(ctx, __FUNCTION__, id, pname, QK_I64, params);
}

void mglGetQueryObjectui64v(GLMContext ctx, GLuint id, GLenum pname, GLuint64 *params)
{
    get_query_object_client(ctx, __FUNCTION__, id, pname, QK_U64, params);
}

static void get_query_buffer_object(GLMContext ctx, const char *func, GLuint id, GLuint buffer,
                                    GLenum pname, GLintptr offset, int kind)
{
    Buffer *b;

    if (!lookup_query(ctx, id) || lookup_query(ctx, id)->target == 0)
    { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }

    b = buffer ? findBuffer(ctx, buffer) : NULL;
    if (!b) { ctx->error_func(ctx, func, GL_INVALID_OPERATION); return; }
    if (!query_pname_valid(pname)) { ctx->error_func(ctx, func, GL_INVALID_ENUM); return; }
    if (offset < 0) { ctx->error_func(ctx, func, GL_INVALID_VALUE); return; }

    get_query_object(ctx, func, id, pname, kind, NULL, b, offset);
}

void mglGetQueryBufferObjectiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    get_query_buffer_object(ctx, __FUNCTION__, id, buffer, pname, offset, QK_I32);
}

void mglGetQueryBufferObjectuiv(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    get_query_buffer_object(ctx, __FUNCTION__, id, buffer, pname, offset, QK_U32);
}

void mglGetQueryBufferObjecti64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    get_query_buffer_object(ctx, __FUNCTION__, id, buffer, pname, offset, QK_I64);
}

void mglGetQueryBufferObjectui64v(GLMContext ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset)
{
    get_query_buffer_object(ctx, __FUNCTION__, id, buffer, pname, offset, QK_U64);
}

#pragma mark - domain 'queries' hooks

void mgl_ext_init_queries(GLMContext ctx)
{
    struct MGLExtState_queries *S;

    if (ctx->state.ext_queries)
        return;

    S = (struct MGLExtState_queries *)calloc(1, sizeof(*S));
    if (!S)
        return;
    initHashTable(&S->query_table, 128);
    ctx->state.ext_queries = S;

    mgl_ext_register_pre_draw(queries_pre_draw);
}

bool mgl_ext_get_queries(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname) {
        case GL_QUERY_BUFFER_BINDING: {
            Buffer *qb = ctx->state.buffers[_QUERY_BUFFER];
            mgl_get_store(type, data, 0, qb ? (double)qb->name : 0.0);
            return true;
        }
        case GL_TIMESTAMP: {
            int64_t now = 0;
            if (QMTL(ctx) && QMTL(ctx)->now_ns)
                now = QMTL(ctx)->now_ns(ctx);
            if (type == MGL_GET_INT64)
                ((GLint64 *)data)[0] = now;
            else
                mgl_get_store(type, data, 0, (double)now);
            return true;
        }
    }
    return false;
}
