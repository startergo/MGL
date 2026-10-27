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
 * draw_buffers.c
 * MGL
 *
 * GL draw entry points: validation (GL 4.6 §10.3-10.5), the cross-domain
 * pre-draw hook, and emulation of primitive types Metal lacks:
 *   LINE_LOOP              -> line strip with a closing index
 *   TRIANGLE_FAN           -> triangle list
 *   LINES_ADJACENCY        -> lines (v1,v2) of every 4 vertices
 *   LINE_STRIP_ADJACENCY   -> line strip over v1..vn-2
 *   TRIANGLES_ADJACENCY    -> triangles (v0,v2,v4) of every 6 vertices
 *   TRIANGLE_STRIP_ADJ.    -> triangle strip over the even vertices
 * (without a geometry shader GL ignores adjacency vertices). GL_UNSIGNED_BYTE
 * indices are widened to 16 bit. Emulated draws build a 32-bit index list on
 * the CPU from the (synchronised) element buffer contents.
 */

#include <mach/mach_vm.h>
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#include <string.h>

#include "glm_context.h"
/* MGL_PROFILE=1 per-draw profiler (MGLRenderer.m) */
uint64_t mgl_prof_begin(int id);
void mgl_prof_end(int id, uint64_t t0);
void mgl_profile_dump(void);
void mgl_prof_frame(void);
#include "mgl.h"
#include "mgl_ext.h"
#include "mgl_ext_buffers.h"
#include "mgl_buffers_err.h"
#include "mgl_ext_rstate.h"
#include "mgl_ext_gpuaudit.h"   /* gpuaudit: cached index rewrites / restart scans */
#include "mgl_ext_queries.h"   /* gpuaudit: multi-draw indirect without CPU command reads */
#include "mgl_ext_tess.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom.h"

#define MTLX() (ctx->mtl_funcs.ext_buffers)

#pragma mark validation helpers

bool check_draw_modes(GLenum mode)
{
    switch(mode)
    {
        case GL_POINTS:
        case GL_LINE_STRIP:
        case GL_LINE_LOOP:
        case GL_LINES:
        case GL_LINE_STRIP_ADJACENCY:
        case GL_LINES_ADJACENCY:
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_FAN:
        case GL_TRIANGLES:
        case GL_TRIANGLE_STRIP_ADJACENCY:
        case GL_TRIANGLES_ADJACENCY:
        case GL_PATCHES:
            return true;
    }

    return false;
}

bool check_element_type(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_UNSIGNED_INT:
            return true;
    }

    return false;
}

GLsizei getTypeSize(GLenum type)
{
    switch(type)
    {
        case GL_UNSIGNED_BYTE:  return sizeof(unsigned char);
        case GL_UNSIGNED_SHORT: return sizeof(unsigned short);
        case GL_UNSIGNED_INT:   return sizeof(unsigned int);
    }

    return 0;
}

static bool mappedNonPersistent(Buffer *b)
{
    return b && b->mapped && !(b->access_flags & GL_MAP_PERSISTENT_BIT);
}

bool processVAO(GLMContext ctx)
{
    VertexArray *vao = ctx->state.vao;

    if (!vao)
        return false;

    if (vao->dirty_bits & DIRTY_VAO_BUFFER_BASE)
    {
        for (int i = 0; i < ctx->state.max_vertex_attribs; i++)
        {
            if ((vao->enabled_attribs & (0x1u << i)) && vao->attrib[i].buffer == NULL)
                return false; // no buffer bound to active attrib

            if ((vao->enabled_attribs >> (i + 1)) == 0)
                break;
        }

        vao->dirty_bits &= ~DIRTY_VAO_BUFFER_BASE;
    }

    return true;
}

/* GL_INVALID_OPERATION conditions on the vertex array state */
bool validate_vao(GLMContext ctx, bool uses_elements)
{
    if (!VAO())
        return false;

    if (ctx->state.vao->dirty_bits && !processVAO(ctx))
        return false;

    unsigned enabled = ctx->state.vao->enabled_attribs;

    for (int i = 0; enabled; i++, enabled >>= 1)
    {
        if (enabled & 0x1)
        {
            Buffer *b = VAO_ATTRIB_STATE(i).buffer;

            // core profile: enabled arrays need a buffer; mapped buffers can't be used
            if (b == NULL || mappedNonPersistent(b))
                return false;
        }
    }

    if (uses_elements)
    {
        Buffer *e = ctx->state.vao->element_array.buffer;

        /* no bound element buffer is legal here: GL's <indices> argument is then a pointer
         * into application memory (Mesa / macOS GL accept client index arrays even in the
         * core profile; the CTS transform_feedback suite draws this way).  The pointer is
         * validated in exec_draw_raw. */
        if (!e)
            return true;

        if (mappedNonPersistent(e))
            return false;
    }

    return true;
}

static Shader *activeStageShader(GLMContext ctx, int stage)
{
    if (ctx->state.program)
        return ctx->state.program->shader_slots[stage];

    if (ctx->state.program_pipeline && ctx->state.program_pipeline->stage_programs[stage])
        return ctx->state.program_pipeline->stage_programs[stage]->shader_slots[stage];

    return NULL;
}

bool validate_program(GLMContext ctx)
{
    // a geometry stage is emulated by the geometry domain (compute + capture passes);
    // refuse to draw only when that emulation is not available
    if (activeStageShader(ctx, _GEOMETRY_SHADER) && !mgl_geom_present(ctx))
        return false;

    // NULL program is legal (vertex / fragment results undefined, not an error)
    return true;
}

/* transform feedback primitive mode compatibility (GL 4.6 §13.3) */
static bool xfb_mode_ok(GLMContext ctx, GLenum mode)
{
    TransformFeedback *tf = ctx->state.transform_feedback;

    if (!tf || !tf->active || tf->paused)
        return true;

    // tessellation / geometry output decides the captured primitive type
    if (activeStageShader(ctx, _TESS_EVALUATION_SHADER) || activeStageShader(ctx, _GEOMETRY_SHADER))
        return true;

    switch(tf->primitive_mode)
    {
        case GL_POINTS:
            return mode == GL_POINTS;

        case GL_LINES:
            return mode == GL_LINES || mode == GL_LINE_LOOP || mode == GL_LINE_STRIP ||
                   mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY;

        case GL_TRIANGLES:
            return mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN ||
                   mode == GL_TRIANGLES_ADJACENCY || mode == GL_TRIANGLE_STRIP_ADJACENCY;
    }

    return true;
}

static bool framebuffer_complete(GLMContext ctx)
{
    Framebuffer *fbo = ctx->state.framebuffer;

    if (!fbo)
        return true; // default framebuffer is always complete

    // mglCheckFramebufferStatus dereferences colour attachment 0
    if (fbo->color_attachments[0].buf.tex == NULL)
        return true;

    return mglCheckFramebufferStatus(ctx, GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

/* returns 0 or the GL error for state-dependent draw validation */
static GLenum draw_dbg(GLMContext ctx, GLenum mode, bool uses_elements, int why)
{
    if (getenv("MGL_DRAW_TRACE"))
        fprintf(stderr, "MGL draw: rejected by validation check %d (mode=%04x elem=%d prog=%p vao=%p xfb=%u)\n",
                why, mode, (int)uses_elements, (void *)ctx->state.program,
                (void *)ctx->state.vao, (unsigned)ctx->state.transform_feedback);
    (void)ctx;
    return GL_INVALID_OPERATION;
}

static GLenum validate_draw_state(GLMContext ctx, GLenum mode, bool uses_elements)
{
    if (!validate_vao(ctx, uses_elements))
        return draw_dbg(ctx, mode, uses_elements, 1);

    if (!validate_program(ctx))
        return draw_dbg(ctx, mode, uses_elements, 2);

    bool tess = activeStageShader(ctx, _TESS_EVALUATION_SHADER) != NULL ||
                activeStageShader(ctx, _TESS_CONTROL_SHADER) != NULL;

    // PATCHES iff a tessellation stage is active
    if ((mode == GL_PATCHES) != tess)
        return draw_dbg(ctx, mode, uses_elements, 3);

    if (!xfb_mode_ok(ctx, mode))
        return draw_dbg(ctx, mode, uses_elements, 4);

    if (!framebuffer_complete(ctx))
        return GL_INVALID_FRAMEBUFFER_OPERATION;

    return 0;
}

#define VALIDATE_STATE(_mode_, _elements_) do { GLenum _e = validate_draw_state(ctx, _mode_, _elements_); \
        if (_e) { ctx->error_func(ctx, __FUNCTION__, _e); return; } } while (0)

#pragma mark draw execution

typedef struct {
    bool begun;
    bool failed;
} DrawBatch;

static bool batch_begin(GLMContext ctx, DrawBatch *b)
{
    if (b->failed)
        return false;

    if (!b->begun)
    {
        if (!MTLX() || !MTLX()->begin_draw(ctx))
        {
            b->failed = true;
            return false;
        }

        b->begun = true;
    }

    return true;
}

static void batch_end(GLMContext ctx, DrawBatch *b)
{
    if (b->begun)
        MTLX()->end_draw(ctx);

    b->begun = false;
}

static int native_prim(GLenum mode)
{
    switch(mode)
    {
        case GL_POINTS: return MGL_PRIM_POINT;
        case GL_LINES: return MGL_PRIM_LINE;
        case GL_LINE_STRIP: return MGL_PRIM_LINE_STRIP;
        case GL_TRIANGLES: return MGL_PRIM_TRIANGLE;
        case GL_TRIANGLE_STRIP: return MGL_PRIM_TRIANGLE_STRIP;
    }

    return -1;
}

static bool restart_enabled(GLMContext ctx, GLenum type, uint32_t *restart_index)
{
    if (ctx->state.caps.primitive_restart_fixed_index)
    {
        *restart_index = type == GL_UNSIGNED_BYTE ? 0xFFu : type == GL_UNSIGNED_SHORT ? 0xFFFFu : 0xFFFFFFFFu;
        return true;
    }

    if (ctx->state.caps.primitive_restart)
    {
        *restart_index = ctx->state.var.primitive_restart_index;
        return true;
    }

    return false;
}

/* source vertex numbers of a draw, honouring element type */
typedef struct {
    const uint8_t *indices;   // NULL: arrays (first + k)
    GLenum type;
    GLint first;
    bool restart;
    uint32_t restart_index;
} VertexSource;

static inline uint32_t vs_get(const VertexSource *s, GLsizei k)
{
    if (!s->indices)
        return (uint32_t)(s->first + k);

    switch(s->type)
    {
        case GL_UNSIGNED_BYTE: return s->indices[k];
        case GL_UNSIGNED_SHORT: return ((const uint16_t *)s->indices)[k];
        default: return ((const uint32_t *)s->indices)[k];
    }
}

static inline bool vs_is_restart(const VertexSource *s, GLsizei k)
{
    return s->indices && s->restart && vs_get(s, k) == s->restart_index;
}

typedef struct {
    uint32_t *data;
    size_t count, cap;
} IndexList;

static void il_push(IndexList *l, uint32_t v)
{
    if (l->count == l->cap)
    {
        l->cap = l->cap ? l->cap * 2 : 256;
        l->data = (uint32_t *)realloc(l->data, l->cap * sizeof(uint32_t));
    }

    l->data[l->count++] = v;
}

#define MGL_STRIP_RESTART 0xFFFFFFFFu

/* expand an emulated primitive type into a native index list; returns native prim */
static int build_emulated(GLenum mode, const VertexSource *src, GLsizei count, IndexList *out)
{
    GLsizei seg_start = 0;

    switch(mode)
    {
        case GL_LINES_ADJACENCY:
            for (GLsizei k = 0; k + 3 < count; k += 4)
            {
                il_push(out, vs_get(src, k + 1));
                il_push(out, vs_get(src, k + 2));
            }
            return MGL_PRIM_LINE;

        case GL_TRIANGLES_ADJACENCY:
            for (GLsizei k = 0; k + 5 < count; k += 6)
            {
                il_push(out, vs_get(src, k));
                il_push(out, vs_get(src, k + 2));
                il_push(out, vs_get(src, k + 4));
            }
            return MGL_PRIM_TRIANGLE;

        default:
            break;
    }

    // strip-like modes: process restart-separated segments
    for (GLsizei k = 0; k <= count; k++)
    {
        if (k < count && !vs_is_restart(src, k))
            continue;

        GLsizei n = k - seg_start;
        GLsizei s = seg_start;

        switch(mode)
        {
            case GL_LINE_LOOP:
                if (n >= 2)
                {
                    if (out->count) il_push(out, MGL_STRIP_RESTART);
                    for (GLsizei i = 0; i < n; i++) il_push(out, vs_get(src, s + i));
                    il_push(out, vs_get(src, s));
                }
                break;

            case GL_TRIANGLE_FAN:
                for (GLsizei i = 1; i + 1 < n; i++)
                {
                    il_push(out, vs_get(src, s));
                    il_push(out, vs_get(src, s + i));
                    il_push(out, vs_get(src, s + i + 1));
                }
                break;

            case GL_LINE_STRIP_ADJACENCY:
                if (n >= 4)
                {
                    if (out->count) il_push(out, MGL_STRIP_RESTART);
                    for (GLsizei i = 1; i < n - 1; i++) il_push(out, vs_get(src, s + i));
                }
                break;

            case GL_TRIANGLE_STRIP_ADJACENCY:
                if (n >= 6)
                {
                    GLsizei tris = (n - 4) / 2;
                    if (out->count) il_push(out, MGL_STRIP_RESTART);
                    for (GLsizei i = 0; i < tris + 2; i++) il_push(out, vs_get(src, s + 2 * i));
                }
                break;
        }

        seg_start = k + 1;
    }

    switch(mode)
    {
        case GL_LINE_LOOP:
        case GL_LINE_STRIP_ADJACENCY:
            return MGL_PRIM_LINE_STRIP;

        case GL_TRIANGLE_STRIP_ADJACENCY:
            return MGL_PRIM_TRIANGLE_STRIP;

        default:
            return MGL_PRIM_TRIANGLE;
    }
}


/* CPU pointer to element indices (after making GPU writes visible) */
static const uint8_t *element_pointer(GLMContext ctx, Buffer *ebuf, size_t offset, GLsizei count, GLenum type)
{
    if (!ebuf || !ebuf->data.buffer_data)
        return NULL;

    size_t bytes = (size_t)count * getTypeSize(type);
    if (offset > (size_t)ebuf->size || bytes > (size_t)ebuf->size - offset)
        return NULL;

    mgl_buffers_prepare_cpu_read(ctx, ebuf);

    return (const uint8_t *)ebuf->data.buffer_data + offset;
}

/* gpuaudit: same range validation as element_pointer() but without syncing the CPU
 * view - a fully native draw never reads the bytes, so it must not be stalled by
 * GPU-written element buffers either */
static bool element_range_ok(Buffer *ebuf, size_t offset, GLsizei count, GLenum type)
{
    if (!ebuf || !ebuf->data.buffer_data)
        return false;

    size_t bytes = (size_t)count * getTypeSize(type);
    if (offset > (size_t)ebuf->size || bytes > (size_t)ebuf->size - offset)
        return false;

    return true;
}

/* gpuaudit: snapshot of every GL state that decides whether a rewrite happens or
 * changes the rewritten list: the restart index, the provoking-vertex convention and
 * the bound program (flat-varying detection).  Part of every cache key below. */
static MGLIndexKeyState index_key_state(GLMContext ctx, bool restart, uint32_t restart_index)
{
    MGLIndexKeyState st;
    memset(&st, 0, sizeof(st));
    st.restart_index = restart_index;
    st.restart_on = restart ? 1 : 0;
    st.lastpv = (ctx->state.ext_rstate && ctx->state.ext_rstate->provoking_vertex == GL_LAST_VERTEX_CONVENTION) ? 1 : 0;
    st.program = ctx->state.program;
    return st;
}

/* gpuaudit: does this draw need an index rewrite?  mgl_rstate_index_rewrite_needed()
 * scans the whole index range on the CPU; the answer only depends on the buffer
 * contents, the draw parameters and the state snapshot, so it is cached and only the
 * first draw after a real change pays the scan. */
static bool index_rewrite_needed(GLMContext ctx, GLenum mode, GLenum type, const uint8_t *cpu, GLsizei count,
                                 Buffer *ebuf, size_t offset, MGLIndexKeyState st)
{
    int cached = mgl_gpuaudit_scan_lookup(ctx, ebuf, (uint32_t)offset, mode, type, count, st);
    if (cached >= 0)
        return cached != 0;

    bool r = mgl_rstate_index_rewrite_needed(ctx, mode, type, cpu, count);
    mgl_gpuaudit_scan_store(ctx, ebuf, (uint32_t)offset, mode, type, count, st, r);
    return r;
}

/* gpuaudit: serve an emulated / rewritten / widened draw from the cached index list
 * (an MTLBuffer built once).  Returns false when this draw is not cached. */
static bool draw_cached_index_list(GLMContext ctx, DrawBatch *batch, Buffer *ebuf, uint32_t offset,
                                   GLenum mode, GLenum type, GLsizei count, GLint first,
                                   MGLIndexKeyState st, int adjacency, GLsizei instances,
                                   GLint basevertex, GLuint baseinstance)
{
    struct MGLExtMetal_gpuaudit *gpm = ctx->mtl_funcs.ext_gpuaudit;
    if (!gpm || !gpm->draw_indexed_mtl)
        return false;

    MGLIndexCacheEnt *e = mgl_gpuaudit_index_lookup(ctx, ebuf, offset, mode, type, count, first, st, adjacency);
    if (!e)
        return false;

    int prim = native_prim(e->out_mode);
    if (prim < 0)
        return false;      /* emulated strips with restart sentinels never get here */

    if (batch_begin(ctx, batch))
        gpm->draw_indexed_mtl(ctx, prim, e->out_type, e->mtl_buffer, 0, e->out_count,
                              instances, basevertex, baseinstance);
    return true;
}

/* gpuaudit: draw a just-produced (and freshly cached) index list from its MTLBuffer;
 * returns false when the list was not cached - then draw from `fallback` instead */
static bool draw_stored_index_list(GLMContext ctx, DrawBatch *batch, int prim,
                                   MGLIndexCacheEnt *e, GLsizei instances, GLint basevertex,
                                   GLuint baseinstance)
{
    struct MGLExtMetal_gpuaudit *gpm = ctx->mtl_funcs.ext_gpuaudit;
    if (!e || !gpm || !gpm->draw_indexed_mtl || prim < 0)
        return false;

    if (batch_begin(ctx, batch))
        gpm->draw_indexed_mtl(ctx, prim, e->out_type, e->mtl_buffer, 0, e->out_count,
                              instances, basevertex, baseinstance);
    return true;
}

static bool is_adjacency(GLenum mode)
{
    return mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY ||
           mode == GL_TRIANGLES_ADJACENCY || mode == GL_TRIANGLE_STRIP_ADJACENCY;
}

/* gpuaudit: provenance of a CPU index list handed to draw_index_list; when set, the
 * widened / emulated list is cached as an MTLBuffer so the next identical draw needs
 * no CPU work at all (see mgl_ext_gpuaudit.h) */
typedef struct IndexListOrigin {
    Buffer         *ebuf;
    uint32_t        offset;
    GLenum          src_mode, src_type;
    GLsizei         src_count;
    GLint           first;
    MGLIndexKeyState state;
    uint8_t         adjacency;
} IndexListOrigin;

/* draw a (CPU) index list of a GL mode: native modes directly, the rest emulated */
static void draw_index_list(GLMContext ctx, DrawBatch *batch, GLenum mode, GLenum type, const void *indices, GLsizei count,
                            GLsizei instances, GLint basevertex, GLuint baseinstance, bool restart, uint32_t restart_index,
                            const IndexListOrigin *origin)
{
    int prim = native_prim(mode);

    if (count <= 0)
        return;

    if (prim >= 0 && type != GL_UNSIGNED_BYTE)
    {
        if (batch_begin(ctx, batch))
            MTLX()->draw_indexed(ctx, prim, type, NULL, 0, indices, count, instances, basevertex, baseinstance);
        return;
    }

    VertexSource src = { (const uint8_t *)indices, type, 0, restart, restart_index };

    if (prim >= 0)
    {
        // GL_UNSIGNED_BYTE indices with a native mode: widen to 16 bit
        uint16_t *wide = (uint16_t *)malloc(count * sizeof(uint16_t));
        if (!wide) { ctx->error_func(ctx, __FUNCTION__, GL_OUT_OF_MEMORY); return; }

        for (GLsizei k = 0; k < count; k++)
            wide[k] = vs_is_restart(&src, k) ? 0xFFFF : (uint16_t)vs_get(&src, k);

        MGLIndexCacheEnt *e = origin
            ? mgl_gpuaudit_index_store(ctx, origin->ebuf, origin->offset, origin->src_mode, origin->src_type,
                                       origin->src_count, origin->first, origin->state, origin->adjacency,
                                       wide, mode, GL_UNSIGNED_SHORT, count)
            : NULL;

        if (!draw_stored_index_list(ctx, batch, prim, e, instances, basevertex, baseinstance))
            if (batch_begin(ctx, batch))
                MTLX()->draw_indexed(ctx, prim, GL_UNSIGNED_SHORT, NULL, 0, wide, count, instances, basevertex, baseinstance);

        free(wide);
        return;
    }

    IndexList list = { 0 };
    prim = build_emulated(mode, &src, count, &list);

    if (list.count)
    {
        MGLIndexCacheEnt *e = origin
            ? mgl_gpuaudit_index_store(ctx, origin->ebuf, origin->offset, origin->src_mode, origin->src_type,
                                       origin->src_count, origin->first, origin->state, origin->adjacency,
                                       list.data, mode, GL_UNSIGNED_INT, (GLsizei)list.count)
            : NULL;

        if (!draw_stored_index_list(ctx, batch, prim, e, instances, basevertex, baseinstance))
            if (batch_begin(ctx, batch))
                MTLX()->draw_indexed(ctx, prim, GL_UNSIGNED_INT, NULL, 0, list.data, (GLsizei)list.count,
                                     instances, basevertex, baseinstance);
    }

    free(list.data);
}

/* one GL draw without hooks: primitive restart / provoking vertex / uint8 rewrites
 * (rstate helpers), primitive type emulation, Metal draw. offset: byte offset into the
 * element buffer for indexed draws. */
static void exec_draw_raw(GLMContext ctx, DrawBatch *batch, GLenum mode, bool indexed, GLenum type, size_t offset,
                          GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    if (count <= 0 || instances <= 0)
        return;

    if (mode == GL_PATCHES)
    {
        /* tess domain: GL_PATCHES is executed by the tessellation pipeline
           (MGLRenderer+Tessellation.m); it needs an active tessellation evaluation shader */
        if (!mgl_tess_draw_patches(ctx, indexed, type, offset, first, count, instances,
                                   basevertex, (GLint)baseinstance))
        {
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
        return;
    }

    int prim = native_prim(mode);
    bool adjacency = is_adjacency(mode);
    uint64_t t27 = mgl_prof_begin(27);

    if (!indexed)
    {
        MGLIndexKeyState st = index_key_state(ctx, false, 0);

        if (!adjacency && mgl_rstate_arrays_rewrite_needed(ctx, mode))
        {
            /* gpuaudit: the generated list depends only on (mode, first, count, state);
             * cache it so a repeated draw does not rebuild it */
            if (draw_cached_index_list(ctx, batch, NULL, 0, mode, 0, count, first, st, 0,
                                       instances, 0, baseinstance))
                return;

            GLenum out_mode = mode;
            GLsizei out_count = 0;
            void *idx = mgl_rstate_generate_array_indices(ctx, mode, first, count, &out_mode, &out_count);

            if (idx)
            {
                MGLIndexCacheEnt *e = mgl_gpuaudit_index_store(ctx, NULL, 0, mode, 0, count, first, st, 0,
                                                               idx, out_mode, GL_UNSIGNED_INT, out_count);
                if (!draw_stored_index_list(ctx, batch, native_prim(out_mode), e, instances, 0, baseinstance))
                    draw_index_list(ctx, batch, out_mode, GL_UNSIGNED_INT, idx, out_count, instances, 0, baseinstance, false, 0, NULL);
                free(idx);
                return;
            }
        }

        if (prim >= 0)
        {
            if (batch_begin(ctx, batch))
                MTLX()->draw_arrays(ctx, prim, first, count, instances, baseinstance);
            return;
        }

        /* gpuaudit: line loop / fan / adjacency emulation for array draws is cached the
         * same way (the list only depends on the draw parameters) */
        if (draw_cached_index_list(ctx, batch, NULL, 0, mode, 0, count, first, st, adjacency,
                                   instances, 0, baseinstance))
            return;

        IndexList list = { 0 };
        VertexSource src = { NULL, 0, first, false, 0 };
        prim = build_emulated(mode, &src, count, &list);

        if (list.count)
        {
            MGLIndexCacheEnt *e = mgl_gpuaudit_index_store(ctx, NULL, 0, mode, 0, count, first, st, adjacency,
                                                           list.data, mode, GL_UNSIGNED_INT, (GLsizei)list.count);
            if (!draw_stored_index_list(ctx, batch, prim, e, instances, 0, baseinstance))
                if (batch_begin(ctx, batch))
                    MTLX()->draw_indexed(ctx, prim, GL_UNSIGNED_INT, NULL, 0, list.data, (GLsizei)list.count, instances, 0, baseinstance);
        }

        free(list.data);
        return;
    }

    Buffer *ebuf = VAO_STATE(element_array.buffer);
    /* no element buffer bound: GL's <indices> argument is a client pointer into memory owned
     * by the application (Mesa and macOS GL accept this even in the core profile; the CTS
     * transform_feedback suite draws with client index arrays).  The CPU walk reads the
     * memory directly and the index cache is bypassed for it. */
    const uint8_t *client = ebuf ? NULL : (const uint8_t *)(uintptr_t)offset;

    uint32_t restart_index = 0;
    bool restart = restart_enabled(ctx, type, &restart_index);
    MGLIndexKeyState st = index_key_state(ctx, restart, restart_index);

    /* gpuaudit: a fully native draw (native primitive mode, 16/32-bit indices) never
     * reads the index bytes on the CPU, so it cannot be stalled by GPU-written element
     * buffers; only 8-bit indices, emulated modes, adjacency and restart need the CPU
     * view (mgl_rstate_index_rewrite_needed / draw_index_list below). */
    bool need_cpu = adjacency || prim < 0 || type == GL_UNSIGNED_BYTE || restart || client != NULL;
    const uint8_t *cpu = NULL;

    if (need_cpu)
    {
        cpu = client ? client : element_pointer(ctx, ebuf, offset, count, type);

        if (!cpu)
        {
            // index range outside the element buffer (or a NULL client pointer)
            ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
            return;
        }
    }
    else if (!element_range_ok(ebuf, offset, count, type))
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    if (need_cpu && !adjacency && !client &&
        index_rewrite_needed(ctx, mode, type, cpu, count, ebuf, offset, st))
    {
        /* gpuaudit: repeated draws reuse the rewritten list, cached as an MTLBuffer */
        if (draw_cached_index_list(ctx, batch, ebuf, (uint32_t)offset, mode, type, count, 0, st, 0,
                                   instances, basevertex, baseinstance))
            return;

        GLenum out_mode = mode, out_type = type;
        GLsizei out_count = 0;
        void *idx = mgl_rstate_rewrite_indices(ctx, mode, type, cpu, count, &out_mode, &out_type, &out_count);

        if (idx)
        {
            // restart has been resolved by the rewrite (Metal restart value for strips)
            MGLIndexCacheEnt *e = mgl_gpuaudit_index_store(ctx, ebuf, (uint32_t)offset, mode, type, count, 0,
                                                           st, 0, idx, out_mode, out_type, out_count);

            if (!draw_stored_index_list(ctx, batch, native_prim(out_mode), e, instances, basevertex, baseinstance))
            {
                uint32_t mrv = out_type == GL_UNSIGNED_INT ? 0xFFFFFFFFu : 0xFFFFu;
                draw_index_list(ctx, batch, out_mode, out_type, idx, out_count, instances, basevertex, baseinstance,
                                native_prim(out_mode) < 0 && restart, mrv, NULL);
            }

            free(idx);
            return;
        }
    }

    if (prim >= 0 && type != GL_UNSIGNED_BYTE && !client)
    {
        if (batch_begin(ctx, batch))
            MTLX()->draw_indexed(ctx, prim, type, ebuf, offset, NULL, count, instances, basevertex, baseinstance);
        return;
    }

    /* 8-bit widening / primitive emulation: served from the cache when possible
     * (client-memory indices are never cached — the application owns the memory) */
    if (!client &&
        draw_cached_index_list(ctx, batch, ebuf, (uint32_t)offset, mode, type, count, 0, st, adjacency,
                               instances, basevertex, baseinstance))
        return;

    IndexListOrigin origin;
    memset(&origin, 0, sizeof(origin));
    origin.ebuf = ebuf;
    origin.offset = (uint32_t)offset;
    origin.src_mode = mode;
    origin.src_type = type;
    origin.src_count = count;
    origin.state = st;
    origin.adjacency = adjacency ? 1 : 0;

    draw_index_list(ctx, batch, mode, type, cpu, count, instances, basevertex, baseinstance,
                    restart, restart_index, client ? NULL : &origin);
    mgl_prof_end(27, t27);
}

/* hand a draw to the transform feedback capture wrappers (xfb domain), which capture and
 * then call the raw draw back through mtl_funcs (MGLRenderer+Draw.m -> mgl_buffers_raw_draw) */
static void route_xfb(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                      GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    if (indexed)
        ctx->mtl_funcs.mtlDrawElementsInstancedBaseVertexBaseInstance(ctx, mode, count, type, (const void *)(uintptr_t)offset,
                                                                      instances, basevertex, baseinstance);
    else
        ctx->mtl_funcs.mtlDrawArraysInstancedBaseInstance(ctx, mode, first, count, instances, baseinstance);
}

void mgl_buffers_raw_draw(GLMContext ctx, GLenum mode, bool indexed, GLenum type, size_t offset,
                          GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    DrawBatch batch = { 0 };

    exec_draw_raw(ctx, &batch, mode, indexed, type, offset, first, count, instances, basevertex, baseinstance);
    batch_end(ctx, &batch);
}

/* one GL draw (after validation): pre-draw hook, transform feedback, raw draw */
static void exec_draw(GLMContext ctx, DrawBatch *batch, GLenum mode, bool indexed, GLenum type, size_t offset,
                      GLint first, GLsizei count, GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    uint64_t p7 = mgl_prof_begin(7);
    if (count <= 0 || instances <= 0)
    {
        mgl_prof_end(7, p7);
        return;
    }

    /* tess domain: only GL_PATCHES can be drawn while a tessellation shader is active */
    if (mode != GL_PATCHES && mgl_tess_active_program(ctx))
    {
        ctx->error_func(ctx, __FUNCTION__, GL_INVALID_OPERATION);
        return;
    }

    {
        uint64_t t24 = mgl_prof_begin(24);
        bool ok = mgl_ext_pre_draw(ctx, mode, count, instances, indexed, indexed ? type : 0,
                                   indexed ? offset : 0);
        mgl_prof_end(24, t24);
        if (!ok)
        {
            mgl_prof_end(7, p7);
            return;
        }
    }

    /* a geometry stage runs through its own path: capture pass + kernel + pass-through
     * raster (it also performs the transform feedback capture of the geometry output) */
    if (mgl_geom_present(ctx))
    {
        batch_end(ctx, batch);
        mgl_geom_route(ctx, mode, indexed, type, offset, first, count, instances, basevertex, baseinstance);
        mgl_prof_end(7, p7);
        return;
    }

    if (mgl_xfb_capturing(ctx))
    {
        /* geom: with a tessellation evaluation stage and no geometry shader the TES is the
         * last pre-rasterization stage, so transform feedback records its output - the tess
         * domain's capture is packed by the geometry domain's kernel.  The xfb domain's
         * vertex-stage capture (route_xfb) would record the wrong stage. */
        if (activeStageShader(ctx, _TESS_EVALUATION_SHADER) && !activeStageShader(ctx, _GEOMETRY_SHADER))
        {
            batch_end(ctx, batch);
            mgl_geom_tess_xfb_route(ctx, indexed, type, offset, first, count, instances,
                                    basevertex, baseinstance);
            mgl_prof_end(7, p7);
            return;
        }
        batch_end(ctx, batch);
        route_xfb(ctx, mode, indexed, type, offset, first, count, instances, basevertex, baseinstance);
        mgl_prof_end(7, p7);
        return;
    }

    exec_draw_raw(ctx, batch, mode, indexed, type, offset, first, count, instances, basevertex, baseinstance);

    /* perf instrumentation (MGL_PROFILE): run lengths of consecutive draws with an
     * identical draw identity — the merging potential for instanced-draw batching */
    {
        static int on = -1;
        if (on < 0) on = getenv("MGL_PROFILE") != NULL;
        if (on)
        {
            static uint64_t id = 0, run = 1, tot = 0, merged = 0, hist[33] = {0};
            uint64_t k = (uint64_t)(uintptr_t)ctx->state.program * 31
                       + (uint64_t)(uintptr_t)ctx->state.vao * 131
                       + (uint64_t)(uintptr_t)ctx->state.vao->element_array.buffer * 977
                       + (uint64_t)mode * 7919 + (uint64_t)type * 104729
                       + (uint64_t)count * 1299709 + (uint64_t)instances * 15485863
                       + (uint64_t)(uint32_t)basevertex * 32452843 + (uint64_t)baseinstance
                       + (uint64_t)(indexed ? offset : (size_t)first * 648391);
            if (k == id)
                run++;
            else
            {
                if (run < 32) hist[run]++; else hist[32]++;
                merged += run; tot += run; run = 1; id = k;
                if (++tot % 4000 == 0)
                {
                    fprintf(stderr, "MGL PROFILE draw-run histogram (draws=%llu, batchable=%llu): ",
                            (unsigned long long)tot, (unsigned long long)merged);
                    for (int i = 1; i <= 32; i++)
                        if (hist[i]) fprintf(stderr, "%d:%llu ", i, (unsigned long long)hist[i]);
                    fprintf(stderr, "\n");
                }
            }
        }
    }
    mgl_prof_end(7, p7);
}

static void draw_arrays_common(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instances, GLuint baseinstance)
{
    DrawBatch batch = { 0 };

    exec_draw(ctx, &batch, mode, false, 0, 0, first, count, instances, 0, baseinstance);
    batch_end(ctx, &batch);
}

static void draw_elements_common(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices,
                                 GLsizei instances, GLint basevertex, GLuint baseinstance)
{
    DrawBatch batch = { 0 };
    size_t offset = (size_t)(uintptr_t)indices;

    exec_draw(ctx, &batch, mode, true, type, offset, 0, count, instances, basevertex, baseinstance);
    batch_end(ctx, &batch);
}

#pragma mark GL draw entry points

void mglDrawArrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(first >= 0 && count >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, false);

    draw_arrays_common(ctx, mode, first, count, 1, 0);
}

void mglDrawArraysInstanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(first >= 0 && count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, false);

    draw_arrays_common(ctx, mode, first, count, instancecount, 0);
}

void mglDrawArraysInstancedBaseInstance(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(first >= 0 && count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, false);

    draw_arrays_common(ctx, mode, first, count, instancecount, baseinstance);
}

void mglDrawElements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, 1, 0, 0);
}

void mglDrawRangeElements(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(end >= start && count >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    // [start, end] is only a hint
    draw_elements_common(ctx, mode, count, type, indices, 1, 0, 0);
}

void mglDrawElementsInstanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, instancecount, 0, 0);
}

void mglDrawElementsBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, 1, basevertex, 0);
}

void mglDrawRangeElementsBaseVertex(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(end >= start && count >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, 1, basevertex, 0);
}

void mglDrawElementsInstancedBaseVertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, instancecount, basevertex, 0);
}

void mglDrawElementsInstancedBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, instancecount, 0, baseinstance);
}

void mglDrawElementsInstancedBaseVertexBaseInstance(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(count >= 0 && instancecount >= 0, GL_INVALID_VALUE);
    VALIDATE_STATE(mode, true);

    draw_elements_common(ctx, mode, count, type, indices, instancecount, basevertex, baseinstance);
}

void mglMultiDrawArrays(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);

    for (GLsizei i = 0; i < drawcount; i++)
        ERROR_CHECK_RETURN(count[i] >= 0 && first[i] >= 0, GL_INVALID_VALUE);

    VALIDATE_STATE(mode, false);

    // behaves as a sequence of DrawArrays (one pre-draw hook call each)
    DrawBatch batch = { 0 };
    for (GLsizei i = 0; i < drawcount; i++)
        exec_draw(ctx, &batch, mode, false, 0, 0, first[i], count[i], 1, 0, 0);
    batch_end(ctx, &batch);
}

static void multi_draw_elements(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type,
                                const void *const *indices, GLsizei drawcount, const GLint *basevertex)
{
    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);

    for (GLsizei i = 0; i < drawcount; i++)
        ERROR_CHECK_RETURN(count[i] >= 0, GL_INVALID_VALUE);

    VALIDATE_STATE(mode, true);

    DrawBatch batch = { 0 };
    for (GLsizei i = 0; i < drawcount; i++)
    {
        size_t offset = (size_t)(uintptr_t)indices[i];

        exec_draw(ctx, &batch, mode, true, type, offset, 0, count[i], 1, basevertex ? basevertex[i] : 0, 0);
    }
    batch_end(ctx, &batch);
}

void mglMultiDrawElements(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount)
{
    multi_draw_elements(ctx, mode, count, type, indices, drawcount, NULL);
}

void mglMultiDrawElementsBaseVertex(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex)
{
    multi_draw_elements(ctx, mode, count, type, indices, drawcount, basevertex);
}

#pragma mark indirect draws

/* read `words` uint32 of an indirect command if its contents are CPU-visible now
 * (or wait for them when `must`); returns false when unknown */
static bool read_indirect(GLMContext ctx, Buffer *buf, size_t offset, uint32_t *out, int words, bool must)
{
    if (!buf || !buf->data.buffer_data)
        return false;

    MGLBufferTrack *t = mgl_buffers_track(ctx, buf, false);
    bool gpu_pending = t && MTLX() && t->write_serial > MTLX()->completed_serial(ctx);

    if (gpu_pending)
    {
        if (!must)
            return false;

        mgl_buffers_prepare_cpu_read(ctx, buf);
    }

    memcpy(out, (const uint8_t *)buf->data.buffer_data + offset, words * sizeof(uint32_t));
    return true;
}

/* `need_counts`: only an active CPU-counted pipeline-statistics query wants the
 * vertex/instance counts of the command (gpuaudit).  When it does not and Metal can
 * run the draw straight from the app's buffer, the command is never read on the CPU. */
static void exec_indirect(GLMContext ctx, DrawBatch *batch, GLenum mode, GLenum type, Buffer *ibuf,
                          size_t offset, bool need_counts)
{
    uint32_t cmd[5] = { 0, 0, 0, 0, 0 };
    int words = type ? 5 : 4;
    int prim = native_prim(mode);
    bool capturing = mgl_xfb_capturing(ctx);

    // the GPU can consume the command directly unless the draw needs CPU-side work
    bool geom = mgl_geom_present(ctx);

    bool gpu_ok = prim >= 0 && type != GL_UNSIGNED_BYTE && !capturing && !geom && mode != GL_PATCHES &&
                  !(type ? mgl_rstate_index_rewrite_needed(ctx, mode, type, NULL, 0)
                         : mgl_rstate_arrays_rewrite_needed(ctx, mode));

    /* a geometry stage can take the command directly as well: the prep kernel reads it from
     * the app's buffer, so the CPU never sees the counts.  An active CPU-counted query
     * (pipeline statistics) still needs the counts: read the command, run the pre-draw
     * hook (query counting + conditional rendering), then route. */
    bool counted = false;   /* queries already counted this draw (don't count twice) */

    if (geom && !capturing && prim >= 0 && type != GL_UNSIGNED_BYTE && mode != GL_PATCHES)
    {
        bool route = true;

        if (need_counts)
        {
            if (read_indirect(ctx, ibuf, offset, cmd, words, true))
            {
                counted = true;
                /* an empty or oversized command draws (and counts) nothing */
                if (cmd[0] == 0 || cmd[1] == 0 || cmd[0] > INT32_MAX || cmd[1] > INT32_MAX)
                    return;
                /* false: conditional rendering discards the draw */
                if (!mgl_ext_pre_draw(ctx, mode, (GLsizei)cmd[0], (GLsizei)cmd[1], type != 0, type,
                                      type ? (size_t)cmd[2] * getTypeSize(type) : 0))
                    return;
            }
            else
            {
                /* counts unavailable (buffer not CPU-visible): route anyway rather than
                 * drop the draw — the query result stays approximate, as before */
                route = false;
            }
        }

        if (route)
        {
            /* give the geometry emulation the command when it is already in hand (the
             * query read it) or cheaply readable from the buffer's CPU copy: it sizes
             * the emulation buffers EXACTLY (the GPU-only fallback sizes by the mode's
             * worst case with a single assumed instance) */
            uint32_t cmd5[5] = { 0, 0, 0, 0, 0 };
            const uint32_t *gcmd = counted ? cmd : NULL;

            if (!gcmd && read_indirect(ctx, ibuf, offset, cmd5, words, false))
                gcmd = cmd5;
            batch_end(ctx, batch);
            if (mgl_geom_route_indirect(ctx, mode, type, ibuf, offset, gcmd))
                return;
        }
    }

    // read the command when CPU-visible; when nothing needs the counts and Metal can
    // take the command buffer directly, skip the read entirely (gpuaudit)
    bool known = counted ? true
                         : ((gpu_ok && !need_counts) ? false
                                                     : read_indirect(ctx, ibuf, offset, cmd, words, !gpu_ok || need_counts));

    if (!known && !gpu_ok)
        return;

    if (known)
    {
        if (!counted && !mgl_ext_pre_draw(ctx, mode, (GLsizei)cmd[0], (GLsizei)cmd[1], type != 0, type,
                                          type ? (size_t)cmd[2] * getTypeSize(type) : 0))
            return;

        if (cmd[0] == 0 || cmd[1] == 0 || cmd[0] > INT32_MAX || cmd[1] > INT32_MAX)
            return;
    }
    else if (!mgl_ext_pre_draw(ctx, mode, 0, 0, false, 0, 0))
    {
        // counts are produced on the GPU and not visible yet
        return;
    }

    if (gpu_ok && !known && ctx->state.ext_gpuaudit)
        ctx->state.ext_gpuaudit->indirect_noread++;

    if (gpu_ok)
    {
        if (batch_begin(ctx, batch))
            MTLX()->draw_indirect(ctx, prim, ibuf, offset, type, type ? VAO_STATE(element_array.buffer) : NULL);
        return;
    }

    GLsizei count = (GLsizei)cmd[0], inst = (GLsizei)cmd[1];
    size_t first_byte = type ? (size_t)cmd[2] * getTypeSize(type) : 0;
    GLint first = type ? 0 : (GLint)cmd[2];
    GLint basevertex = type ? (GLint)cmd[3] : 0;
    GLuint baseinstance = type ? cmd[4] : cmd[3];

    if (geom)
    {
        batch_end(ctx, batch);
        mgl_geom_route(ctx, mode, type != 0, type, first_byte, first, count, inst, basevertex, baseinstance);
        return;
    }

    if (capturing)
    {
        batch_end(ctx, batch);
        route_xfb(ctx, mode, type != 0, type, first_byte, first, count, inst, basevertex, baseinstance);
        return;
    }

    exec_draw_raw(ctx, batch, mode, type != 0, type, first_byte, first, count, inst, basevertex, baseinstance);
}

void mgl_buffers_multi_draw_indirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount,
                                     GLsizei stride, bool use_count, GLintptr count_offset)
{
    size_t cmd_size = type ? sizeof(DrawElementsIndirectCommand) : sizeof(DrawArraysIndirectCommand);
    size_t offset = (size_t)(uintptr_t)indirect;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    if (type)
        ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    ERROR_CHECK_RETURN(drawcount >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN((stride % 4) == 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN((offset % 4) == 0, GL_INVALID_VALUE);
    if (use_count)
        ERROR_CHECK_RETURN(count_offset >= 0 && (count_offset % 4) == 0, GL_INVALID_VALUE);

    VALIDATE_STATE(mode, type != 0);

    Buffer *ibuf = STATE(buffers[_DRAW_INDIRECT_BUFFER]);
    ERROR_CHECK_RETURN(ibuf != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!mappedNonPersistent(ibuf), GL_INVALID_OPERATION);

    size_t step = stride ? (size_t)stride : cmd_size;

    if (drawcount > 0)
    {
        uint64_t last = (uint64_t)offset + (uint64_t)step * (uint64_t)(drawcount - 1) + cmd_size;
        ERROR_CHECK_RETURN(last <= (uint64_t)ibuf->size, GL_INVALID_OPERATION);
    }

    if (use_count)
    {
        Buffer *pbuf = ctx->state.ext_buffers ? ctx->state.ext_buffers->parameter_buffer : NULL;

        ERROR_CHECK_RETURN(pbuf != NULL, GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN(!mappedNonPersistent(pbuf), GL_INVALID_OPERATION);
        ERROR_CHECK_RETURN((uint64_t)count_offset + 4 <= (uint64_t)pbuf->size, GL_INVALID_OPERATION);

        uint32_t n = 0;
        // the draw count lives in GPU-visible memory: wait for queued writes
        if (!read_indirect(ctx, pbuf, (size_t)count_offset, &n, 1, true))
            return;

        mgl_buffers_mark_gpu_use(ctx, pbuf, false);

        if ((GLsizei)n < drawcount && n <= (uint32_t)INT32_MAX)
            drawcount = (GLsizei)n;   // clamped to maxdrawcount
    }

    /* gpuaudit: the vertex/instance counts of each command are only needed by a
     * CPU-counted pipeline-statistics query (transform feedback capture forces the CPU
     * path anyway); otherwise the commands are handed to Metal unread. */
    bool need_counts = mgl_queries_counting_active(ctx) || mgl_xfb_capturing(ctx);

    DrawBatch batch = { 0 };
    for (GLsizei i = 0; i < drawcount; i++)
        exec_indirect(ctx, &batch, mode, type, ibuf, offset + step * (size_t)i, need_counts);
    batch_end(ctx, &batch);
}

static void single_indirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    size_t cmd_size = type ? sizeof(DrawElementsIndirectCommand) : sizeof(DrawArraysIndirectCommand);
    size_t offset = (size_t)(uintptr_t)indirect;

    ERROR_CHECK_RETURN(check_draw_modes(mode), GL_INVALID_ENUM);
    if (type)
        ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);
    ERROR_CHECK_RETURN((offset % 4) == 0, GL_INVALID_VALUE);

    VALIDATE_STATE(mode, type != 0);

    Buffer *ibuf = STATE(buffers[_DRAW_INDIRECT_BUFFER]);
    ERROR_CHECK_RETURN(ibuf != NULL, GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN(!mappedNonPersistent(ibuf), GL_INVALID_OPERATION);
    ERROR_CHECK_RETURN((uint64_t)offset + cmd_size <= (uint64_t)ibuf->size, GL_INVALID_OPERATION);

    DrawBatch batch = { 0 };
    exec_indirect(ctx, &batch, mode, type, ibuf, offset, true);
    batch_end(ctx, &batch);
}

void mglDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect)
{
    single_indirect(ctx, mode, 0, indirect);
}

void mglDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect)
{
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    single_indirect(ctx, mode, type, indirect);
}

void mglMultiDrawArraysIndirect(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    mgl_buffers_multi_draw_indirect(ctx, mode, 0, indirect, drawcount, stride, false, 0);
}

void mglMultiDrawElementsIndirect(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride)
{
    ERROR_CHECK_RETURN(check_element_type(type), GL_INVALID_ENUM);

    mgl_buffers_multi_draw_indirect(ctx, mode, type, indirect, drawcount, stride, false, 0);
}
