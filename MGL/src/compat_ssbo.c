/*
 * compat_ssbo.c — ALIASED shader-storage-block bindings inside one Metal function.
 *
 * GL 4.6 §7.12 (glBufferStorage / binding points) lets several shader storage
 * block bindings name the same buffer object range: a block binding is a VIEW of
 * the store, so accesses through the two blocks alias, and inside one
 * invocation GL orders them (a store through block A followed by a load through
 * block B of the same memory must see the store).  Metal's compiler is allowed
 * to assume distinct [[buffer(n)]] pointer ARGUMENTS never alias, so it forwards
 * the store to the later load and the shader observes the stale value
 * (dEQP ES31 synchronization.in_invocation.ssbo_alias_overwrite: store 456
 * through sb_store0, store groupNdx through sb_store1, read sb_store0 -> 456;
 * memoryBarrierBuffer() or folding both accesses onto ONE block both pass).
 *
 * The fix is the one Mesa-style trick Metal has: at dispatch time, when two
 * storage blocks of the active compute program are bound to the identical
 * buffer range, compile a VARIANT of the stage whose secondary block argument
 * is renamed and re-bound inside the function body to a reinterpret_cast view of
 * the primary block's pointer — both GL names then hold the same address and
 * the compiler cannot forward across the pair.  The variant is cached per
 * (base function, alias signature) by the renderer (MGLRenderer+Compat.m,
 * compatSSBOAliasComputeFunction:); the MSL rewrite itself is here, pure text:
 *
 *   device Storage1& sb_store1 [[buffer(1)]]
 *     ->
 *   device Storage1& sb_store1__mglAlias [[buffer(1)]]
 *   { device Storage1& sb_store1 = *reinterpret_cast<device Storage1>(&sb_store0); ... }
 *
 * The dead __mglAlias argument keeps its attribute index (Metal strips unused
 * arguments; the encoder still binds the row, which is legal), and the block's
 * runtime-array .length() keeps reading its own spvBufferSizeConstants entry —
 * both ranges are identical, so the length is unchanged.
 *
 * MGL_NO_SSBO_ALIAS=1 switches the variant selection off (A/B).
 */

#include "glm_context.h"
#include "spirv_cross_c.h"
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* MSL entry-point signature parsing                                   */
/* ------------------------------------------------------------------ */

static bool msl_ident(char c)
{
    return isalnum((unsigned char)c) || c == '_';
}

/* offset of the bracket matching the one at <open>, or -1.  Tracks every MSL
 * bracket class so template arguments and attributes nest correctly. */
static long msl_match(const char *msl, size_t open)
{
    int depth = 0;

    for (size_t p = open; msl[p]; p++)
    {
        char c = msl[p];

        if (c == '(' || c == '[' || c == '{' || c == '<')
            depth++;
        else if (c == ')' || c == ']' || c == '}' || c == '>')
        {
            if (--depth == 0) return (long)p;
            if (depth < 0) return -1;
        }
    }
    return -1;
}

struct MslBufParam {
    unsigned    msl_index;  /* the [[buffer(n)]] argument index */
    size_t      name_off;   /* the parameter's identifier inside msl */
    size_t      name_len;
    size_t      type_off;   /* "device [const] Struct&" before the name */
    size_t      type_len;
};

/* parse one parameter text [s, e); true when it is a buffer argument */
static bool msl_parse_param(const char *msl, size_t s, size_t e, struct MslBufParam *out)
{
    while (s < e && isspace((unsigned char)msl[s])) s++;
    while (e > s && isspace((unsigned char)msl[e - 1])) e--;
    if (s + 2 >= e || msl[e - 1] != ']') return false;

    /* the attribute ends the parameter: find the "[[" that opens this LAST
     * attribute group (a same-name CALL would list plain expressions, no
     * "[[buffer(n)]]", and is rejected here) */
    size_t a = e - 2;
    while (a > s && !(msl[a] == '[' && msl[a + 1] == '[')) a--;
    if (!(msl[a] == '[' && msl[a + 1] == '[')) return false;

    if (strncmp(msl + a, "[[buffer(", 9) != 0) return false;
    size_t d = a + 9;
    unsigned idx = 0;
    bool any = false;
    while (d < e && isdigit((unsigned char)msl[d]))
    {
        idx = idx * 10u + (unsigned)(msl[d] - '0');
        any = true;
        d++;
    }
    if (!any || d >= e || msl[d] != ')') return false;

    /* declaration text [s, a): "device [const] Struct& name" */
    size_t de = a;
    while (de > s && isspace((unsigned char)msl[de - 1])) de--;

    size_t ne = de;
    size_t ns = ne;
    while (ns > s && msl_ident(msl[ns - 1])) ns--;
    if (ns == ne) return false;

    size_t ts = ns;
    while (ts > s && isspace((unsigned char)msl[ts - 1])) ts--;
    if (ts == s) return false;

    out->msl_index = idx;
    out->name_off = ns;
    out->name_len = ne - ns;
    out->type_off = s;
    out->type_len = ts - s;
    return true;
}

/* locate <entry>'s DEFINITION and its parameter list; true when found */
static bool msl_entry_params(const char *msl, const char *entry,
                             size_t *open_off, size_t *close_off,
                             struct MslBufParam *params, int max_params, int *nparams)
{
    size_t elen = strlen(entry);
    const char *p = msl;

    *nparams = 0;
    while ((p = strstr(p, entry)) != NULL)
    {
        if (p == msl || !msl_ident(p[-1]))
        {
            const char *q = p + elen;
            while (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n') q++;
            if (*q == '(')
            {
                size_t open = (size_t)(q - msl);
                long close = msl_match(msl, open);
                int n = 0;
                bool has_buffer = false;

                if (close > 0)
                {
                    size_t s = open + 1, e = (size_t)close;

                    /* split top-level parameters */
                    int depth = 0;
                    size_t start = s;
                    for (size_t i = s; i <= e; i++)
                    {
                        char c = msl[i];
                        if (i < e && (c == '(' || c == '[' || c == '{' || c == '<')) depth++;
                        else if (i < e && (c == ')' || c == ']' || c == '}' || c == '>')) depth--;
                        else if ((c == ',' && depth == 0) || i == e)
                        {
                            if (n < max_params)
                            {
                                struct MslBufParam prm;
                                if (msl_parse_param(msl, start, i, &prm))
                                {
                                    params[n++] = prm;
                                    has_buffer = true;
                                }
                            }
                            start = i + 1;
                        }
                    }
                }

                if (has_buffer)
                {
                    *open_off = open;
                    *close_off = (size_t)close;
                    *nparams = n;
                    return true;
                }
            }
        }
        p += elen;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* the rewrite                                                          */
/* ------------------------------------------------------------------ */

struct MslEdit {
    size_t      pos;
    char        *text;      /* malloc'd */
};

char *mgl_compat_msl_alias_ssbo(const char *msl, const char *entry,
                                const unsigned *from_msl, const unsigned *to_msl,
                                int npairs)
{
    struct MslBufParam params[64];
    int nparams = 0;
    size_t open = 0, close = 0;
    struct MslEdit edits[72];
    int nedit = 0;
    char *out = NULL;

    if (!msl || !entry || !from_msl || !to_msl || npairs <= 0 || npairs > 32)
        return NULL;

    if (!msl_entry_params(msl, entry, &open, &close, params, 64, &nparams))
        return NULL;

    /* the function body's opening brace follows the parameter list */
    size_t b = close + 1;
    while (msl[b] == ' ' || msl[b] == '\t' || msl[b] == '\r' || msl[b] == '\n') b++;
    if (msl[b] != '{')
        return NULL;

    /* alias declarations to inject at the top of the body */
    char *body = (char *)malloc(npairs * 512 + 16);
    if (!body) return NULL;
    body[0] = 0;
    size_t body_len = 0;
    int nused = 0;

    for (int k = 0; k < npairs; k++)
    {
        const struct MslBufParam *from = NULL, *to = NULL;
        char type_from[256], type_to[256], name_from[256], name_to[256];
        char decl[1024];

        for (int i = 0; i < nparams; i++)
        {
            if (params[i].msl_index == from_msl[k]) from = &params[i];
            if (params[i].msl_index == to_msl[k]) to = &params[i];
        }
        if (!from || !to || from == to) continue;

        /* storage blocks live in `device` memory; UBO-style `constant`
         * arguments are read-only and keep their own binding */
        snprintf(type_from, sizeof type_from, "%.*s", (int)from->type_len, msl + from->type_off);
        snprintf(type_to, sizeof type_to, "%.*s", (int)to->type_len, msl + to->type_off);
        snprintf(name_from, sizeof name_from, "%.*s", (int)from->name_len, msl + from->name_off);
        snprintf(name_to, sizeof name_to, "%.*s", (int)to->name_len, msl + to->name_off);

        if (strncmp(type_from, "device ", 7) != 0 || strncmp(type_to, "device ", 7) != 0)
            continue;

        /* reinterpret_cast cannot cast constness away: a non-const block may
         * not fold onto a const-qualified one (the reverse is fine) */
        if (strstr(type_to, "const") && !strstr(type_from, "const"))
            continue;

        size_t tfl = strlen(type_from);
        bool is_ref = tfl > 0 && type_from[tfl - 1] == '&';
        bool is_ptr = tfl > 0 && type_from[tfl - 1] == '*';
        if (!is_ref && !is_ptr) continue;

        if (is_ref)
        {
            /* the cast target is the POINTER form: `device Storage1& s =
             * *reinterpret_cast<device Storage1 *>(&primary);` — a bare
             * struct type is not a cast target */
            char type_base[256], type_ptr[258];

            snprintf(type_base, sizeof type_base, "%.*s", (int)strlen(type_from) - 1, type_from);
            snprintf(type_ptr, sizeof type_ptr, "%s*", type_base);
            snprintf(decl, sizeof decl, "%s& %s = *reinterpret_cast<%s>(&%s);\n",
                     type_base, name_from, type_ptr, name_to);
        }
        else
        {
            snprintf(decl, sizeof decl, "%s %s = reinterpret_cast<%s>(%s);\n",
                     type_from, name_from, type_from, name_to);
        }

        /* rename the dead argument, then append the alias declaration */
        char *ren = (char *)malloc(16);
        if (!ren) break;
        memcpy(ren, "__mglAlias", 11);
        edits[nedit].pos = from->name_off + from->name_len;
        edits[nedit].text = ren;
        nedit++;

        body_len += strlen(decl);
        if (body_len + 1 >= (size_t)npairs * 512 + 16) break;
        strcat(body, decl);
        nused++;
    }

    if (nused == 0)
    {
        for (int i = 0; i < nedit; i++) free(edits[i].text);
        free(body);
        return NULL;
    }

    /* the injected declarations open the body */
    {
        char *pre = (char *)malloc(body_len + 2);
        if (!pre) { for (int i = 0; i < nedit; i++) free(edits[i].text); free(body); return NULL; }
        pre[0] = '\n';
        memcpy(pre + 1, body, body_len + 1);
        edits[nedit].pos = b + 1;
        edits[nedit].text = pre;
        nedit++;
    }
    free(body);

    /* apply the edits in address order */
    for (int i = 1; i < nedit; i++)
    {
        struct MslEdit key = edits[i];
        int j = i - 1;
        while (j >= 0 && edits[j].pos > key.pos) { edits[j + 1] = edits[j]; j--; }
        edits[j + 1] = key;
    }

    {
        size_t len = strlen(msl), extra = 0, o = 0;

        for (int i = 0; i < nedit; i++) extra += strlen(edits[i].text);
        out = (char *)malloc(len + extra + 1);
        if (!out)
        {
            for (int i = 0; i < nedit; i++) free(edits[i].text);
            return NULL;
        }

        size_t cursor = 0;
        for (int i = 0; i < nedit; i++)
        {
            if (edits[i].pos > len) { free(out); out = NULL; break; }
            memcpy(out + o, msl + cursor, edits[i].pos - cursor);
            o += edits[i].pos - cursor;
            cursor = edits[i].pos;
            size_t tl = strlen(edits[i].text);
            memcpy(out + o, edits[i].text, tl);
            o += tl;
        }
        if (out)
        {
            memcpy(out + o, msl + cursor, len - cursor + 1);
            for (int i = 0; i < nedit; i++) free(edits[i].text);
        }
    }

    return out;
}

/* ------------------------------------------------------------------ */
/* which storage blocks alias at this moment                            */
/* ------------------------------------------------------------------ */

/* Storage-block bindings of <stage> that name the IDENTICAL buffer range
 * (same Buffer object, same offset, same size — glBindBufferBase of one buffer
 * to several points, or equal glBindBufferRange calls).  Every later member of
 * a group is returned as a pair folding it onto the group's first member.
 * Returns the pair count (0 when nothing aliases, -1 when there are more pairs
 * than <max_pairs> — the caller then keeps the unmodified function). */
int mgl_compat_ssbo_alias_pairs(GLMContext ctx, BufferMapList *map,
                                unsigned *from_msl, unsigned *to_msl, int max_pairs)
{
    enum { MAX_ROWS = 32 };
    struct {
        Buffer      *buf;
        GLintptr    off;
        GLsizeiptr  sz;
        unsigned    msl;
    } rows[MAX_ROWS];
    int n = 0, np = 0;

    (void)ctx;

    if (!from_msl || !to_msl || max_pairs <= 0)
        return 0;
    if (!map || map->count < 2)
        return 0;

    /* r101: walk the renderer's compute BUFFER MAP (built by
     * mapGLBuffersToMTLBufferMap for this very dispatch) instead of re-walking
     * the resource list.  The map is exactly what the encoder binds — it has
     * already resolved instance-array expansion (one row per Metal argument
     * slot), unbound-unit dummy substitution (nullShaderBuffer rows) and the
     * msl_binding fallbacks, and where two rows collide on one Metal index the
     * LATER row is what setComputeBuffer leaves in place.
     *
     * Re-deriving all that from SpirvResource duplicated the rules and got the
     * instance-array case wrong: dEQP ssbo.layout.random.all_shared_buffer.35
     * (base-seed 1) declares `buffer BlockB {..} blockB[2]' whose FIRST
     * instance was stripped (msl_binding = 0xffffffff; the flattened instance
     * 1 WRAPPED to Metal index 0, colliding with BlockC).  The resource walk
     * expanded the array from instance 1's binding and read unit binding+1 —
     * BlockC's range — matched it as an "alias" and rewrote a non-aliasing
     * block to a reinterpret_cast of an unrelated pointer. */

    /* last row per Metal index wins, as the encoder binds them in order */
    for (GLuint i = 0; i < map->count; i++)
    {
        BufferMap *m = &map->buffers[i];
        int slot = -1;

        if (m->attribute_mask)
            continue;               /* vertex-attribute row, not a buffer binding */

        for (int k = 0; k < n; k++)
            if (rows[k].msl == m->mtl_index)
                slot = k;
        if (slot < 0 && n < MAX_ROWS)
        {
            slot = n;
            n++;
        }
        if (slot < 0)
            continue;

        rows[slot].buf = m->buf;
        rows[slot].off = m->offset;
        rows[slot].sz = m->size;
        rows[slot].msl = m->mtl_index;
    }

    for (int i = 0; i < n && np < max_pairs; i++)
    {
        if (!rows[i].buf || rows[i].sz <= 0)
            continue;       /* unbound/dummy row: the shader never sees data through it */
        for (int j = 0; j < i && np < max_pairs; j++)
        {
            if (!rows[j].buf || rows[j].sz <= 0)
                continue;
            if (rows[j].buf != rows[i].buf || rows[j].off != rows[i].off || rows[j].sz != rows[i].sz)
                continue;
            if (rows[j].msl == rows[i].msl)
                continue;
            from_msl[np] = rows[i].msl;
            to_msl[np] = rows[j].msl;
            if (getenv("MGL_SSBO_DEBUG"))
                fprintf(stderr, "MGL compat: ssbo alias pair %u -> %u (buf %p off %ld sz %ld)\n",
                        rows[i].msl, rows[j].msl, (void *)rows[i].buf,
                        (long)rows[i].off, (long)rows[i].sz);
            np++;
            break;      /* one alias target is enough per secondary */
        }
    }
    return np;
}
