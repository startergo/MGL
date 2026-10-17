/* gpuaudit.c — C side of the 'gpuaudit' domain (see mgl_ext_gpuaudit.h).
 *
 * Contents:
 *   - the per-buffer content-generation table (every path that changes a
 *     buffer's CPU-visible contents bumps it via mgl_gpuaudit_invalidate()),
 *   - the primitive-restart scan cache and the rewritten-index-list cache used
 *     by draw_buffers.c,
 *   - mgl_ext_init_gpuaudit / mgl_ext_get_gpuaudit.
 *
 * The caches are deliberately small and keyed on the *exact* draw parameters,
 * so a false hit is impossible: two draws share an entry only when every input
 * that can change the produced indices (source bytes, range, mode, index type,
 * restart value, provoking-vertex convention, adjacency handling) is identical.
 */
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "glm_context.h"
#include "mgl_ext.h"
#include "mgl_ext_gpuaudit.h"

#define GA(ctx) ((ctx)->state.ext_gpuaudit)

static inline uint32_t ga_hash_ptr(void *p)
{
    uintptr_t v = (uintptr_t)p >> 4;
    v *= 2654435761u;
    return (uint32_t)(v ^ (v >> 16));
}

/* ------------------------------------------------------------------ content */

static void ga_serial_grow(GLMContext ctx)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s)
        return;

    uint32_t old_cap = s->serial_cap;
    MGLSerialEnt *old = s->serial;

    s->serial_cap = old_cap ? old_cap * 2 : 256;
    s->serial = (MGLSerialEnt *)calloc(s->serial_cap, sizeof(MGLSerialEnt));
    if (!s->serial)
    {
        s->serial = old;
        s->serial_cap = old_cap;
        return;
    }

    for (uint32_t i = 0; i < old_cap; i++)
        if (old[i].buf)
        {
            uint32_t mask = s->serial_cap - 1;
            uint32_t j = ga_hash_ptr(old[i].buf) & mask;
            while (s->serial[j].buf)
                j = (j + 1) & mask;
            s->serial[j] = old[i];
        }

    free(old);
}

static MGLSerialEnt *ga_serial_ent(GLMContext ctx, Buffer *buf, bool create)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s || !buf)
        return NULL;

    if (create && (s->serial_cap == 0 || (s->serial_cap - 1) * 4 <= s->serial_cap * 3))
        ga_serial_grow(ctx);

    if (s->serial_cap == 0)
        return NULL;

    uint32_t mask = s->serial_cap - 1;
    uint32_t i = ga_hash_ptr(buf) & mask;

    for (uint32_t n = 0; n < s->serial_cap; n++, i = (i + 1) & mask)
    {
        if (s->serial[i].buf == buf)
            return &s->serial[i];
        if (s->serial[i].buf == NULL)
        {
            if (!create)
                return NULL;
            s->serial[i].buf = buf;
            s->serial[i].serial = ++s->serial_next;
            return &s->serial[i];
        }
    }

    return NULL;
}

void mgl_gpuaudit_invalidate(GLMContext ctx, Buffer *buf)
{
    MGLSerialEnt *e = ga_serial_ent(ctx, buf, true);
    if (e)
        e->serial = ++GA(ctx)->serial_next;
}

uint64_t mgl_gpuaudit_content(GLMContext ctx, Buffer *buf)
{
    MGLSerialEnt *e = ga_serial_ent(ctx, buf, false);
    return e ? e->serial : 0;
}

void mgl_gpuaudit_invalidate_all(GLMContext ctx)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s)
        return;

    for (int i = 0; i < MGL_GA_SCAN_CACHE; i++)
        s->scan[i].ebuf = NULL;
    for (int i = 0; i < MGL_GA_INDEX_CACHE; i++)
    {
        if (s->index[i].mtl_buffer && ctx->mtl_funcs.ext_gpuaudit && ctx->mtl_funcs.ext_gpuaudit->index_buffer_free)
            ctx->mtl_funcs.ext_gpuaudit->index_buffer_free(ctx, s->index[i].mtl_buffer);
        memset(&s->index[i], 0, sizeof(s->index[i]));
    }
}

/* a buffer's CPU view is cacheable while it is not mapped: a persistent map
 * lets the app rewrite the bytes with no GL call, so no notification exists */
bool mgl_gpuaudit_cacheable(GLMContext ctx, Buffer *ebuf)
{
    (void)ctx;
    return ebuf && !ebuf->mapped && ebuf->data.buffer_data != 0;
}

/* ------------------------------------------------------- restart scan cache */

static bool ga_state_equal(const MGLIndexKeyState *a, const MGLIndexKeyState *b)
{
    return a->restart_index == b->restart_index && a->restart_on == b->restart_on &&
           a->lastpv == b->lastpv && a->program == b->program;
}

int mgl_gpuaudit_scan_lookup(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode, GLenum type,
                             GLsizei count, MGLIndexKeyState state)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s || !mgl_gpuaudit_cacheable(ctx, ebuf))
        return -1;

    uint64_t content = mgl_gpuaudit_content(ctx, ebuf);

    for (int i = 0; i < MGL_GA_SCAN_CACHE; i++)
    {
        MGLScanCacheEnt *e = &s->scan[i];
        if (e->ebuf != ebuf || e->content != content || e->offset != offset ||
            e->mode != mode || e->type != type || e->count != count ||
            !ga_state_equal(&e->state, &state))
            continue;

        e->lru = ++s->lru;
        s->scan_hits++;
        return e->needed ? 1 : 0;
    }

    return -1;
}

void mgl_gpuaudit_scan_store(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode, GLenum type,
                             GLsizei count, MGLIndexKeyState state, int needed)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s || !mgl_gpuaudit_cacheable(ctx, ebuf) || needed < 0)
        return;

    uint64_t content = mgl_gpuaudit_content(ctx, ebuf);
    MGLScanCacheEnt *best = &s->scan[0];

    for (int i = 0; i < MGL_GA_SCAN_CACHE; i++)
    {
        MGLScanCacheEnt *e = &s->scan[i];
        if (e->ebuf == ebuf && e->content == content && e->offset == offset &&
            e->mode == mode && e->type == type && e->count == count &&
            ga_state_equal(&e->state, &state))
        {
            best = e;
            break;
        }
        if (e->lru < best->lru)
            best = e;
    }

    best->ebuf = ebuf;
    best->content = content;
    best->offset = offset;
    best->mode = mode;
    best->type = type;
    best->count = count;
    best->state = state;
    best->needed = needed ? 1 : 0;
    best->lru = ++s->lru;
    s->scan_misses++;
}

/* -------------------------------------------------- rewritten index cache */

static bool ga_index_match(const MGLIndexCacheEnt *e, Buffer *ebuf, uint64_t content, uint32_t offset,
                           GLenum mode, GLenum type, GLsizei count, GLint first,
                           MGLIndexKeyState state, int adjacency)
{
    return e->ebuf == ebuf && e->content == content && e->offset == offset &&
           e->mode == mode && e->type == type && e->count == count && e->first == first &&
           ga_state_equal(&e->state, &state) &&
           e->adjacency == (uint8_t)(adjacency ? 1 : 0);
}

MGLIndexCacheEnt *mgl_gpuaudit_index_lookup(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode,
                                            GLenum type, GLsizei count, GLint first,
                                            MGLIndexKeyState state, int adjacency)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s)
        return NULL;

    /* array-index emulation (ebuf == NULL) depends only on the parameters */
    uint64_t content = 0;
    if (ebuf)
    {
        if (!mgl_gpuaudit_cacheable(ctx, ebuf))
            return NULL;
        content = mgl_gpuaudit_content(ctx, ebuf);
    }

    for (int i = 0; i < MGL_GA_INDEX_CACHE; i++)
    {
        MGLIndexCacheEnt *e = &s->index[i];
        if (!e->mtl_buffer || !ga_index_match(e, ebuf, content, offset, mode, type, count, first,
                                              state, adjacency))
            continue;

        e->lru = ++s->lru;
        s->index_hits++;
        return e;
    }

    return NULL;
}

MGLIndexCacheEnt *mgl_gpuaudit_index_store(GLMContext ctx, Buffer *ebuf, uint32_t offset, GLenum mode,
                                           GLenum type, GLsizei count, GLint first,
                                           MGLIndexKeyState state, int adjacency,
                                           void *data, GLenum out_mode, GLenum out_type, GLsizei out_count)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    struct MGLExtMetal_gpuaudit *m = ctx ? ctx->mtl_funcs.ext_gpuaudit : NULL;

    if (!s || !m || !m->index_buffer_new || out_count <= 0)
        return NULL;      /* caller keeps `data` and draws from it itself */

    uint64_t content = 0;
    if (ebuf)
    {
        if (!mgl_gpuaudit_cacheable(ctx, ebuf))
            return NULL;
        content = mgl_gpuaudit_content(ctx, ebuf);
    }

    size_t bytes = (size_t)out_count * (out_type == GL_UNSIGNED_SHORT ? 2 : 4);
    void *mtl = m->index_buffer_new(ctx, data, bytes);

    if (!mtl)
        return NULL;

    /* evict the least recently used entry */
    MGLIndexCacheEnt *best = &s->index[0];
    for (int i = 0; i < MGL_GA_INDEX_CACHE; i++)
    {
        MGLIndexCacheEnt *e = &s->index[i];
        if (!e->mtl_buffer)
        {
            best = e;
            break;
        }
        if (e->lru < best->lru)
            best = e;
    }

    if (best->mtl_buffer && m->index_buffer_free)
        m->index_buffer_free(ctx, best->mtl_buffer);

    memset(best, 0, sizeof(*best));
    best->ebuf = ebuf;
    best->content = content;
    best->offset = offset;
    best->mode = mode;
    best->type = type;
    best->count = count;
    best->first = first;
    best->state = state;
    best->adjacency = adjacency ? 1 : 0;
    best->out_mode = out_mode;
    best->out_type = out_type;
    best->out_count = out_count;
    best->mtl_buffer = mtl;
    best->lru = ++s->lru;
    s->index_misses++;

    return best;
}

/* ------------------------------------------------ converted attribute streams */

static bool ga_shadow_match(const MGLShadowCacheEnt *e, Buffer *buf, uint64_t content, GLintptr start,
                            GLint stride, GLenum type, GLint size, int kind)
{
    return e->mtl_buffer && e->buf == buf && e->content == content && e->start == start &&
           e->stride == stride && e->type == type && e->size == size && e->kind == kind;
}

void *mgl_gpuaudit_shadow_lookup(GLMContext ctx, Buffer *buf, GLintptr start, GLint stride,
                                 GLenum type, GLint size, int kind)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s || !mgl_gpuaudit_cacheable(ctx, buf))
        return NULL;

    uint64_t content = mgl_gpuaudit_content(ctx, buf);

    for (int i = 0; i < MGL_GA_SHADOW_CACHE; i++)
    {
        MGLShadowCacheEnt *e = &s->shadow[i];
        if (!ga_shadow_match(e, buf, content, start, stride, type, size, kind))
            continue;

        e->lru = ++s->lru;
        s->shadow_hits++;
        return e->mtl_buffer;
    }

    return NULL;
}

void *mgl_gpuaudit_shadow_store(GLMContext ctx, Buffer *buf, GLintptr start, GLint stride,
                                GLenum type, GLint size, int kind, const void *data, size_t bytes)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    struct MGLExtMetal_gpuaudit *m = ctx ? ctx->mtl_funcs.ext_gpuaudit : NULL;

    if (!s || !m || !m->index_buffer_new || !data || bytes == 0)
        return NULL;

    if (!mgl_gpuaudit_cacheable(ctx, buf))
        return NULL;

    uint64_t content = mgl_gpuaudit_content(ctx, buf);
    void *mtl = m->index_buffer_new(ctx, data, bytes);
    if (!mtl)
        return NULL;

    MGLShadowCacheEnt *best = &s->shadow[0];
    for (int i = 0; i < MGL_GA_SHADOW_CACHE; i++)
    {
        MGLShadowCacheEnt *e = &s->shadow[i];
        if (!e->mtl_buffer) { best = e; break; }
        if (e->lru < best->lru) best = e;
    }

    if (best->mtl_buffer && m->index_buffer_free)
        m->index_buffer_free(ctx, best->mtl_buffer);

    memset(best, 0, sizeof(*best));
    best->buf = buf;
    best->content = content;
    best->start = start;
    best->stride = stride;
    best->type = type;
    best->size = size;
    best->kind = kind;
    best->mtl_buffer = mtl;
    best->lru = ++s->lru;
    s->shadow_misses++;

    return mtl;
}

/* ------------------------------------------------------------- ext plumbing */

void mgl_ext_init_gpuaudit(GLMContext ctx)
{
    if (ctx->state.ext_gpuaudit == NULL)
    {
        ctx->state.ext_gpuaudit = (struct MGLExtState_gpuaudit *)calloc(1, sizeof(struct MGLExtState_gpuaudit));
        assert(ctx->state.ext_gpuaudit);
    }
}

bool mgl_ext_get_gpuaudit(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    struct MGLExtState_gpuaudit *s = GA(ctx);
    if (!s || !data)
        return false;

    GLuint v = 0;
    switch (pname)
    {
        case MGL_GPUAUDIT_PSO_HITS:      v = s->pso_hits; break;
        case MGL_GPUAUDIT_PSO_MISSES:    v = s->pso_misses; break;
        case MGL_GPUAUDIT_SCAN_HITS:     v = s->scan_hits; break;
        case MGL_GPUAUDIT_SCAN_MISSES:   v = s->scan_misses; break;
        case MGL_GPUAUDIT_INDEX_HITS:    v = s->index_hits; break;
        case MGL_GPUAUDIT_INDEX_MISSES:  v = s->index_misses; break;
        case MGL_GPUAUDIT_PBO_READBACKS: v = s->pbo_readbacks; break;
        case MGL_GPUAUDIT_GPU_CLEARS:    v = s->gpu_clears; break;
        case MGL_GPUAUDIT_INDIRECT_NOREAD: v = s->indirect_noread; break;
        case MGL_GPUAUDIT_SHADOW_HITS:   v = s->shadow_hits; break;
        case MGL_GPUAUDIT_SHADOW_MISSES: v = s->shadow_misses; break;
        case MGL_GPUAUDIT_XFB_SCATTERS:  v = s->xfb_scatters; break;
        default: return false;
    }

    mgl_get_store(type, data, 0, v);
    return true;
}
