/*
 * tessellation.c
 * MGL — tessellation shaders (TCS/TES) and GL_PATCHES: GL side.
 *
 * The Metal side of this domain is MGLRenderer+Tessellation.m; see mgl_ext_tess.h
 * for the overall design. This file owns:
 *   - the MSL rewrites of the tessellation stages (called from program.c during link)
 *   - SPIR-V parsing (tessellation execution modes, control point layouts)
 *   - the CPU tessellator (GL primitive generation: triangles/quads/isolines)
 *   - the GL state, glGet hooks, program queries and the GL_PATCHES draw entry
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>

#include "glm_context.h"
char *tess_shadow_implicit_lod(const char *src);


#include "mgl.h"
#include "mgl_ext.h"
#include "mgl_ext_program.h"      /* mgl_program_register_programiv_hook */
#include "mgl_ext_tess.h"
#include "mgl_ext_rstate.h"
#include "mgl_ext_queries.h"
#include "mgl_ext_program.h"
#include "spirv_cross_c.h"
#include "glcorearb.h"

/* ------------------------------------------------------------------ helpers */

/* Trace flag, evaluated once: getenv() per draw (or per shader rewrite) is not
 * free, and this file sits on the draw path. MGL_TESS_DEBUG=1 enables tracing. */
static int tess_trace_flag = -1;
static bool tess_trace(void)
{
    if (tess_trace_flag < 0)
        tess_trace_flag = (getenv("MGL_TESS_DEBUG") != NULL) ? 1 : 0;
    return tess_trace_flag != 0;
}

/* MGL_TESS_CPU (force the CPU reference path), evaluated once like tess_trace();
 * MGLRenderer.m consults it when it picks the pipeline's vertex function */
static int tess_cpu_flag = -1;
bool mgl_tess_cpu_reference(void)
{
    if (tess_cpu_flag < 0)
        tess_cpu_flag = (getenv("MGL_TESS_CPU") != NULL) ? 1 : 0;
    return tess_cpu_flag != 0;
}

/* failure exit of the TES->plain-MSL rewrite: never silent, but only verbose with
 * MGL_TESS_DEBUG (the step number localises the failing parse in rewrite_tes) */
#define TES_REWRITE_FAIL(step)                                                    \
    do {                                                                          \
        if (tess_trace())                                                         \
            fprintf(stderr, "MGL tess: plain-TES rewrite failed at step %d\n",    \
                    (step));                                                      \
        return NULL;                                                              \
    } while (0)

static void tess_reset_scratch(struct MGLExtState_tess *t)
{
    t->tess_vertex_count = 0;
    t->tess_index_count = 0;
    t->patch_count = 0;
}

/* ------------------------------------------------------------------ GLSL compat
 * GLSL 4.60 §7.1.2: a tessellation shader may redeclare gl_in/gl_out (gl_PerVertex) with a
 * SUBSET of the default members, and the interface stays legal when both sides declare the
 * same subset.  glslang's cross-stage link instead demands the two blocks carry the SAME
 * member set ("vertex block member has no corresponding member in tessellation control
 * block").  Expanding every gl_PerVertex redeclaration of a tessellation stage to the full
 * built-in set is semantically identical (added outputs are never written, added inputs
 * never read), so the rewrite splices the missing members in. */
static const char *tess_pervtx_name[4] = {
    "gl_Position", "gl_PointSize", "gl_ClipDistance", "gl_CullDistance",
};

/* find the matching close brace, skipping // and * comments */
static size_t tess_skip_ws_comments(const char *s, size_t i)
{
    for (;;)
    {
        while (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') i++;
        if (s[i] == '/' && s[i + 1] == '/')
        {
            while (s[i] && s[i] != '\n') i++;
            continue;
        }
        if (s[i] == '/' && s[i + 1] == '*')
        {
            i += 2;
            while (s[i] && !(s[i] == '*' && s[i + 1] == '/')) i++;
            i = s[i] ? i + 2 : i;
            continue;
        }
        return i;
    }
}

static int tess_ident_char(int c)
{
    return isalnum(c) || c == '_';
}
#define is_ident_char tess_ident_char

/* the size bracket of the FIRST `name[N]' occurrence in the source ("8" for an
 * empty `[]'), written into out ("" when the name is not declared with a
 * bracket at all) */
static void tess_declared_size(const char *src, const char *name, char *out, size_t outn)
{
    const char *p = src;
    size_t nl = strlen(name);
    out[0] = 0;
    while ((p = strstr(p, name)))
    {
        if ((p == src || !tess_ident_char((unsigned char)p[-1])) &&
            !tess_ident_char((unsigned char)p[nl]))
        {
            const char *b = p + nl;
            while (*b == ' ' || *b == '\t') b++;
            if (*b == '[')
            {
                size_t n = 0;
                b++;
                while (*b == ' ' || *b == '\t') b++;
                if (*b == ']') { out[0] = 0; return; }   /* `[]' - unsized: the link sizes it */
                while (*b >= '0' && *b <= '9' && n + 1 < outn) out[n++] = *b++;
                if (n && (*b == ']' || *b == ' '))
                {
                    out[n] = 0;
                    return;
                }
                out[0] = 0;   /* non-numeric bracket (a macro parameter): keep looking */
            }
        }
        p += nl;
    }
}

/* GL_ARB_shader_ballot in a TES: the post-tessellation vertex function has no
 * simdgroup attributes on Metal (see mgl_tess_glsl_compat), so this rewrites
 * the ARB subgroup builtins/functions of a TES source to a size-1 subgroup —
 * invocation 0, the lane-0 masks, a one-lane ballot, identity reads.  The
 * subgroup size is implementation-defined per stage, and every other MGL stage
 * keeps the real 32-wide simdgroup, so this is conformant.  Returns NULL when
 * the source does not mention any of the names. */
static char *tess_ballot_one_lane(const char *src)
{
    static const struct { const char *name, *repl; } map[] = {
        /* the cts_glsl_compat pass may already have rewritten the size */
        { "mgl_ballot_subgroup_size()", "1u" },
        { "gl_SubGroupSizeARB",         "1u" },
        { "gl_SubGroupInvocationARB",   "0u" },
        { "gl_SubGroupEqMaskARB",       "mgl_tess_mask_eq()" },
        { "gl_SubGroupGeMaskARB",       "mgl_tess_mask_ge()" },
        { "gl_SubGroupGtMaskARB",       "mgl_tess_mask_gt()" },
        { "gl_SubGroupLeMaskARB",       "mgl_tess_mask_le()" },
        { "gl_SubGroupLtMaskARB",       "mgl_tess_mask_lt()" },
        { "ballotARB",           "mgl_tess_ballot1" },
        { "readInvocationARB",   "mgl_tess_read1" },
        { "readFirstInvocationARB", "mgl_tess_readfirst1" },
    };
    static const char pre[] =
        /* the helpers use uint64_t (ballotARB's type); enable it in case the
         * source declared ballotARB's extension but not int64's */
        "#extension GL_ARB_gpu_shader_int64 : enable\n"
        /* the ARB masks are uint64_t on the desktop profile (glslang declares
         * them so at >= 450), so the emulated lane-0 values are too; a size-1
         * subgroup has only lane 0, so Eq/Ge/Le = {0} = 1 and Gt/Lt = {} = 0 */
        "uint64_t mgl_tess_ballot1(bool p) { return uint64_t(p); }\n"
        "uint64_t mgl_tess_mask_eq() { return uint64_t(1); }\n"
        "uint64_t mgl_tess_mask_ge() { return uint64_t(1); }\n"
        "uint64_t mgl_tess_mask_gt() { return 0ul; }\n"
        "uint64_t mgl_tess_mask_le() { return uint64_t(1); }\n"
        "uint64_t mgl_tess_mask_lt() { return 0ul; }\n"
        "float  mgl_tess_read1(float v, uint i)  { return v; }\n"
        "vec2   mgl_tess_read1(vec2 v, uint i)   { return v; }\n"
        "vec3   mgl_tess_read1(vec3 v, uint i)   { return v; }\n"
        "vec4   mgl_tess_read1(vec4 v, uint i)   { return v; }\n"
        "int    mgl_tess_read1(int v, uint i)    { return v; }\n"
        "ivec2  mgl_tess_read1(ivec2 v, uint i)  { return v; }\n"
        "ivec3  mgl_tess_read1(ivec3 v, uint i)  { return v; }\n"
        "ivec4  mgl_tess_read1(ivec4 v, uint i)  { return v; }\n"
        "uint   mgl_tess_read1(uint v, uint i)   { return v; }\n"
        "uvec2  mgl_tess_read1(uvec2 v, uint i)  { return v; }\n"
        "uvec3  mgl_tess_read1(uvec3 v, uint i)  { return v; }\n"
        "uvec4  mgl_tess_read1(uvec4 v, uint i)  { return v; }\n"
        "bool   mgl_tess_read1(bool v, uint i)   { return v; }\n"
        "bvec2  mgl_tess_read1(bvec2 v, uint i)  { return v; }\n"
        "bvec3  mgl_tess_read1(bvec3 v, uint i)  { return v; }\n"
        "bvec4  mgl_tess_read1(bvec4 v, uint i)  { return v; }\n"
        "double mgl_tess_read1(double v, uint i) { return v; }\n"
        /* readFirstInvocationARB takes no index: it degenerates to the value */
        "float  mgl_tess_readfirst1(float v)  { return v; }\n"
        "vec2   mgl_tess_readfirst1(vec2 v)   { return v; }\n"
        "vec3   mgl_tess_readfirst1(vec3 v)   { return v; }\n"
        "vec4   mgl_tess_readfirst1(vec4 v)   { return v; }\n"
        "int    mgl_tess_readfirst1(int v)    { return v; }\n"
        "ivec2  mgl_tess_readfirst1(ivec2 v)  { return v; }\n"
        "ivec3  mgl_tess_readfirst1(ivec3 v)  { return v; }\n"
        "ivec4  mgl_tess_readfirst1(ivec4 v)  { return v; }\n"
        "uint   mgl_tess_readfirst1(uint v)   { return v; }\n"
        "uvec2  mgl_tess_readfirst1(uvec2 v)  { return v; }\n"
        "uvec3  mgl_tess_readfirst1(uvec3 v)  { return v; }\n"
        "uvec4  mgl_tess_readfirst1(uvec4 v)  { return v; }\n"
        "bool   mgl_tess_readfirst1(bool v)   { return v; }\n"
        "bvec2  mgl_tess_readfirst1(bvec2 v)  { return v; }\n"
        "bvec3  mgl_tess_readfirst1(bvec3 v)  { return v; }\n"
        "bvec4  mgl_tess_readfirst1(bvec4 v)  { return v; }\n"
        "double mgl_tess_readfirst1(double v) { return v; }\n";

    if (!src) return NULL;
    int any = 0;
    for (size_t k = 0; k < sizeof map / sizeof map[0]; k++)
        if (strstr(src, map[k].name)) { any = 1; break; }
    if (!any) return NULL;
    /* idempotence: on a re-entry (the recursion from mgl_tess_glsl_compat) the
     * prelude is already there and only the token renames must continue — in
     * particular gl_SubGroupInvocationARB inside the cts pass's mask helpers */
    int have_prelude = strstr(src, "mgl_tess_ballot1") != NULL;

    size_t len = strlen(src);
    /* inject the helpers after the last #extension (and #version) line */
    size_t insert = 0;
    {
        const char *v = strstr(src, "#version");
        if (v)
        {
            const char *nl = strchr(v, '\n');
            insert = nl ? (size_t)(nl - src) + 1 : len;
        }
        const char *p = src;
        while ((p = strstr(p, "#extension")) != NULL)
        {
            const char *nl = strchr(p, '\n');
            if (!nl) break;
            size_t at = (size_t)(nl - src) + 1;
            if (at > insert) insert = at;
            p = nl;
        }
    }

    char *out = malloc(len * 3 + sizeof pre + 64);
    if (!out) return NULL;
    size_t o = 0;
    int changed = 0;
    for (size_t i = 0; i <= len; i++)
    {
        if (insert && i == insert)
        {
            if (!have_prelude)
            {
                memcpy(out + o, pre, sizeof pre - 1);
                o += sizeof pre - 1;
            }
            insert = 0;
        }
        size_t took = 0;
        if (i < len)
        {
            for (size_t k = 0; k < sizeof map / sizeof map[0]; k++)
            {
                size_t nl = strlen(map[k].name);
                if (i + nl <= len && !strncmp(src + i, map[k].name, nl) &&
                    (i == 0 || !is_ident_char((unsigned char)src[i - 1])) &&
                    (i + nl == len || !is_ident_char((unsigned char)src[i + nl])))
                {
                    /* the cts pass's helper DEFINITION ("uint
                     * mgl_ballot_subgroup_size() { ... }") must survive even
                     * though call sites become 1u: skip a match that is a
                     * function head (followed by '{') or a declaration */
                    size_t j = i + nl;
                    while (j < len && (src[j] == ' ' || src[j] == '\t')) j++;
                    int is_head = (j < len && src[j] == '{') ||
                                  (i >= 5 && !strncmp(src + i - 5, "uint ", 5));
                    if (!is_head)
                    {
                        size_t rl = strlen(map[k].repl);
                        memcpy(out + o, map[k].repl, rl);
                        o += rl;
                        took = nl;
                        changed = 1;
                    }
                    else
                    {
                        memcpy(out + o, src + i, nl);
                        o += nl;
                        took = nl;
                    }
                    break;
                }
            }
        }
        if (!took && i < len)
            out[o++] = src[i];
        i += took ? took - 1 : 0;
    }
    out[o] = 0;
    if (!changed) { free(out); return NULL; }   /* only the definition matched: nothing to rewrite */
    if (getenv("MGL_TESS_BALLOT_DUMP")) fprintf(stderr, "[tess ballot1 source]\n%s\n[/tess ballot1]\n", out);
    return out;
}

/* A VERTEX stage may redeclare the distance members loose (`out float
 * gl_CullDistance[8];') — legal GLSL and what the CTS writes — while the
 * tessellation stages' gl_PerVertex BLOCK redeclarations (expanded to the full
 * member set by the pass below) stay blocks.  glslang's cross-stage machinery
 * cannot pair a loose-member interface with a block one: the link fails
 * ("vertex block member has no corresponding member in tessellation control
 * block") or its io mapper aborts ("Invalid In/Out variable type :
 * gl_PerVertex").  Rewrite the loose members of a VERTEX source into one
 * gl_PerVertex block carrying the full built-in member set (arrays sized as
 * declared, 8 = the implementation maximum when loose/unsized) — the SPIR-V
 * interface is identical. */
static char *tess_vs_member_to_block(const char *src)
{
    size_t i = 0, len = strlen(src);
    int depth = 0, clip_sz = -1, cull_sz = -1, changed = 0, decline = 0;
    char *out = NULL;
    size_t out_len = 0;
    FILE *mem;
    long insert_at = -1;      /* output offset right after the last #version line */

    if (!strstr(src, "gl_ClipDistance") && !strstr(src, "gl_CullDistance")) return NULL;
    /* gl_CullDistance in a block needs GLSL 4.50 or GL_ARB_cull_distance; the
     * block emitter below adds the extension pragma itself when a sub-4.50
     * stage redeclared the member loose, and omits the member otherwise
     * (clip-only #version 130 shaders — KHR-GL46.clip_distance). */

    mem = open_memstream(&out, &out_len);
    if (!mem) return NULL;

    while (i < len)
    {
        size_t ls = i, e = i;
        while (e < len && src[e] != '\n') e++;

        /* brace depth (function bodies are not interface declarations) */
        for (size_t k = ls; k < e; k++)
        {
            if (src[k] == '{') depth++;
            else if (src[k] == '}') depth--;
        }

        if (depth == 0)
        {
            /* a loose member redeclaration: `out float gl_(Clip|Cull)Distance[N];' */
            size_t p = ls;
            int is_decl = 0, which = 0;
            long sz = -1;
            while (p < e && (src[p] == ' ' || src[p] == '\t')) p++;
            if (p + 3 < e && !strncmp(src + p, "out", 3) &&
                !is_ident_char((unsigned char)src[p + 3]))
            {
                size_t q = tess_skip_ws_comments(src, p + 3);
                if (q + 5 < e && !strncmp(src + q, "float", 5) &&
                    !is_ident_char((unsigned char)src[q + 5]))
                {
                    size_t r = tess_skip_ws_comments(src, q + 5);
                    for (int w = 0; w < 2; w++)
                    {
                        const char *nm = w ? "gl_CullDistance" : "gl_ClipDistance";
                        size_t nl = strlen(nm);
                        if (r + nl <= e && !strncmp(src + r, nm, nl) &&
                            !is_ident_char((unsigned char)src[r + nl]))
                        {
                            size_t s = tess_skip_ws_comments(src, r + nl);
                            if (s < e && src[s] == '[')
                            {
                                size_t t = s + 1;
                                int ndigits = 0, bad = 0;
                                sz = 0;
                                while (t < e && src[t] != ']' && src[t] != ';')
                                {
                                    if (src[t] >= '0' && src[t] <= '9')
                                    {
                                        sz = sz * 10 + (src[t] - '0');
                                        ndigits++;
                                    }
                                    else if (src[t] != ' ' && src[t] != '\t')
                                        bad = 1;
                                    t++;
                                }
                                if (t >= e || src[t] != ']') { sz = -2; break; }
                                /* a size EXPRESSION (`gl_MaxClipDistances + 1') or a
                                 * literal outside 1..8 cannot be folded into the
                                 * block without hiding the compile error GL
                                 * requires (KHR-GL46.clip_distance.negative):
                                 * leave the declaration loose and decline the
                                 * rewrite - glslang judges the source itself */
                                if (bad) { sz = -2; break; }
                                if (ndigits == 0) sz = 8;   /* `[]' - implicit maximum */
                                else if (sz < 1 || sz > 8) { sz = -2; break; }
                            }
                            else sz = 8;                 /* unsized member */
                            is_decl = 1;
                            which = w;
                            break;
                        }
                    }
                }
            }

            if (is_decl && sz >= 0)
            {
                if (which == 0) { if (clip_sz < 0) clip_sz = (int)sz; }
                else { if (cull_sz < 0) cull_sz = (int)sz; }
                changed = 1;
                i = e + (e < len);
                continue;               /* drop the loose declaration */
            }
            if (is_decl && sz == -2)
                decline = 1;            /* unfolded size: keep the source intact */
        }

        fwrite(src + ls, 1, e - ls, mem);
        if (e < len) fputc('\n', mem);
        i = e + (e < len);
        if (insert_at < 0 && src[ls] == '#' && !strncmp(src + ls, "#version", 8))
            insert_at = ftell(mem);    /* right after the version line's newline */
    }

    if (!changed || decline)
    {
        fclose(mem);
        free(out);
        return NULL;
    }

    /* splice the block in after #version (or at the head) */
    {
        char block[320], cszs[16] = "", uszs[16] = "";
        int n, cull_member;
        int version = 0;
        char *fin;
        size_t at = insert_at > 0 ? (size_t)insert_at : 0;
        const char *v = strstr(src, "#version");
        fclose(mem);
        if (v) version = atoi(v + strlen("#version"));
        if (clip_sz > 0) snprintf(cszs, sizeof cszs, "%d", clip_sz);
        if (cull_sz > 0) snprintf(uszs, sizeof uszs, "%d", cull_sz);
        /* gl_CullDistance is not a member of the built-in gl_PerVertex before
         * GLSL 4.50 (nor does the type exist): a block redeclaration that adds
         * it fails glslang with "'gl_PerVertex' : block redeclaration has extra
         * members" — KHR-GL46.clip_distance.functional ships a #version 130
         * vertex shader.  Include the member below 4.50 only when this stage
         * itself redeclared it loose, and enable GL_ARB_cull_distance so the
         * block+member is legal at that version. */
        cull_member = (version >= 450) || (cull_sz >= 0);
        if (cull_member)
        {
            n = snprintf(block, sizeof block,
                         "%s"
                         "out gl_PerVertex { vec4 gl_Position; float gl_PointSize;"
                         " float gl_ClipDistance[%s]; float gl_CullDistance[%s]; };\n",
                         (version < 450) ? "#extension GL_ARB_cull_distance : enable\n" : "",
                         cszs, uszs);
        }
        else
        {
            n = snprintf(block, sizeof block,
                         "out gl_PerVertex { vec4 gl_Position; float gl_PointSize;"
                         " float gl_ClipDistance[%s]; };\n",
                         cszs);
        }
        if (n < 0 || (size_t)n >= sizeof block) { free(out); return NULL; }
        fin = malloc(out_len + (size_t)n + 1);
        if (!fin) { free(out); return NULL; }
        memcpy(fin, out, at);
        memcpy(fin + at, block, (size_t)n);
        memcpy(fin + at + n, out + at, out_len - at);
        fin[out_len + n] = 0;
        free(out);
        return fin;
    }
}

/* ---- GLSL 4.6 §7.1: indexing the unsized distance arrays -------------------
 *
 * gl_ClipDistance and gl_CullDistance are predeclared UNSIZED; a shader may
 * index them with a non-constant expression only after redeclaring the array
 * with an explicit size.  glslang silently sizes them to the implementation
 * maximum, so the invalid shader compiles and links (KHR-GL46
 * cull_distance.negative, "does not redeclare ... uses dynamic index-based
 * writes").  The check has to run on the RAW source: the introspect pass pins
 * and folds the declarations, and a #ifdef'd declaration is resolved away in
 * the compiled text — so the scanner tracks the preprocessor itself.
 */
#define TESS_PP_DEFS 512

typedef struct {
    const char *name;
    size_t      name_len;
    const char *val;        /* object-like expansion (points into src) */
    size_t      val_len;
} TessPpDefine;

static int tess_pp_define_find(const TessPpDefine *defs, int ndefs, const char *tok, size_t len)
{
    for (int i = ndefs - 1; i >= 0; i--)              /* the last definition wins */
        if (defs[i].name_len == len && !memcmp(defs[i].name, tok, len))
            return i;
    return -1;
}

/* is the bracket text a constant expression?  literals, parenthesised operator
 * combinations and identifiers the macro table resolves to literals count; any
 * other identifier (a loop counter, a uniform) does not. */
static bool tess_pp_index_is_constant(const char *s, size_t n, const TessPpDefine *defs,
                                      int ndefs, int depth)
{
    size_t i = 0;
    int    tokens = 0, idents = 0;
    if (!n) return false;
    while (i < n)
    {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { i++; continue; }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_')
        {
            size_t j = i;
            while (j < n && (is_ident_char((unsigned char)s[j]))) j++;
            tokens++;
            int d = tess_pp_define_find(defs, ndefs, s + i, j - i);
            if (d < 0) return false;                 /* not a known constant */
            idents++;
            if (depth < 4 && !tess_pp_index_is_constant(defs[d].val, defs[d].val_len,
                                                       defs, ndefs, depth + 1))
                return false;                        /* expands to something dynamic */
            i = j;
            continue;
        }
        if (c == '-' || c == '+' || c == '*' || c == '/' || c == '%' || c == '(' ||
            c == ')' || c == '<' || c == '>' || c == '?' || c == ':' || c == '!')
        {
            tokens++;
            i++;
            continue;
        }
        if (c >= '0' && c <= '9')
        {
            size_t j = i;
            while (j < n && !((s[j] >= 'a' && s[j] <= 'z') || (s[j] >= 'A' && s[j] <= 'Z') ||
                              s[j] == '_' || s[j] == '(')) j++;
            if (j < n) return false;                 /* `2x': not a plain literal */
            tokens++;
            i = j;
            continue;
        }
        return false;                                /* anything else: dynamic */
    }
    return tokens > 0 && idents < tokens;            /* a lone macro name is still constant */
}

/* `#if' with a literal-only condition: 0 / 1 / defined(NAME) */
static int tess_pp_cond_value(const char *line, size_t len, const TessPpDefine *defs, int ndefs)
{
    const char *p = line, *e = line + len;
    while (p < e && (*p == ' ' || *p == '\t')) p++;
    if (p + 7 <= e && !strncmp(p, "defined", 7) && !is_ident_char((unsigned char)p[7]))
    {
        p += 7;
        while (p < e && (*p == ' ' || *p == '\t')) p++;
        int want = 1;
        if (p < e && *p == '!') { want = 0; p++; while (p < e && (*p==' '||*p=='\t')) p++; }
        if (p < e && *p == '(') { p++; while (p < e && (*p==' '||*p=='\t')) p++; }
        size_t n = 0;
        while (p + n < e && is_ident_char((unsigned char)p[n])) n++;
        if (!n) return -1;
        return tess_pp_define_find(defs, ndefs, p, n) >= 0 ? want : !want;
    }
    if (p < e && (*p == '0' || *p == '1') && (p + 1 == e || p[1] == ' ' || p[1] == '\t'))
        return *p - '0';
    return -1;                                       /* not literal: caller assumes true */
}

/* walk the raw source: return the #error source when a non-constant index is
 * written to an unsized gl_ClipDistance / gl_CullDistance, else NULL */
char *mgl_tess_reject_dynamic_distance_index(const char *src)
{
    if (!src) return NULL;
    if (!strstr(src, "gl_ClipDistance") && !strstr(src, "gl_CullDistance")) return NULL;

    static const char *names[2] = { "gl_ClipDistance", "gl_CullDistance" };
    TessPpDefine defs[TESS_PP_DEFS];
    int ndefs = 0;
    bool sized[2] = { false, false };
    bool dynamic_write[2] = { false, false };
    int  cond_active[32], cond_taken[32], cond_depth = 0;   /* -1 = inactive branch */
    size_t i = 0, len = strlen(src);

    while (i < len)
    {
        size_t ls = i, e = i;
        while (e < len && src[e] != '\n') e++;

        /* --- preprocessor directives decide which lines are live --- */
        if (src[ls] == '#')
        {
            size_t p = ls + 1;
            while (p < e && (src[p] == ' ' || src[p] == '\t')) p++;
            struct { const char *w; size_t n; } d;
            d.w = src + p;
            d.n = 0;
            while (p + d.n < e && is_ident_char((unsigned char)src[p + d.n])) d.n++;
            bool live = true;
            for (int k = 0; k < cond_depth; k++) if (cond_active[k] == 0) live = false;

            #define TESS_DIR_IS(s) (d.n == strlen(s) && !strncmp(d.w, s, d.n))
            if (live && TESS_DIR_IS("define"))
            {
                size_t q = p + d.n;
                while (q < e && (src[q] == ' ' || src[q] == '\t')) q++;
                size_t ns = q;
                while (q < e && is_ident_char((unsigned char)src[q])) q++;
                if (q > ns && q < e && src[q] == '(')
                {
                    /* function-like macro: its expansion is not a plain literal */
                    if (ndefs < TESS_PP_DEFS)
                    {
                        defs[ndefs].name = src + ns;
                        defs[ndefs].name_len = q - ns;
                        defs[ndefs].val = src + e;
                        defs[ndefs].val_len = 0;
                        ndefs++;
                    }
                }
                else if (q > ns && ndefs < TESS_PP_DEFS)
                {
                    size_t vs = q;
                    while (vs < e && (src[vs] == ' ' || src[vs] == '\t')) vs++;
                    defs[ndefs].name = src + ns;
                    defs[ndefs].name_len = q - ns;
                    defs[ndefs].val = src + vs;
                    defs[ndefs].val_len = e - vs;
                    ndefs++;
                }
            }
            else if (TESS_DIR_IS("ifdef") || TESS_DIR_IS("ifndef"))
            {
                size_t q = p + d.n;
                while (q < e && (src[q] == ' ' || src[q] == '\t')) q++;
                size_t ns = q;
                while (q < e && is_ident_char((unsigned char)src[q])) q++;
                int defined = 0;
                if (q > ns && live)
                    defined = tess_pp_define_find(defs, ndefs, src + ns, q - ns) >= 0;
                else if (q > ns && !live)
                    defined = -1;                     /* outer level off: stay off */
                if (cond_depth < 32)
                {
                    cond_active[cond_depth] = live ? (TESS_DIR_IS("ifdef") ? (defined == 1)
                                                                          : (defined == 0)) : 0;
                    cond_taken[cond_depth] = cond_active[cond_depth];
                    cond_depth++;
                }
            }
            else if (TESS_DIR_IS("if"))
            {
                int v = tess_pp_cond_value(src + p + d.n, e - (p + d.n), defs, ndefs);
                if (cond_depth < 32)
                {
                    cond_active[cond_depth] = live ? (v != 0) : 0;  /* unknown -> true */
                    cond_taken[cond_depth] = cond_active[cond_depth];
                    cond_depth++;
                }
            }
            else if (TESS_DIR_IS("elif"))
            {
                if (cond_depth)
                {
                    int taken = cond_taken[cond_depth - 1];
                    int v = tess_pp_cond_value(src + p + d.n, e - (p + d.n), defs, ndefs);
                    bool outer = true;
                    for (int k = 0; k < cond_depth - 1; k++) if (cond_active[k] == 0) outer = false;
                    cond_active[cond_depth - 1] = (!taken && outer) ? (v != 0) : 0;
                    cond_taken[cond_depth - 1] = cond_taken[cond_depth - 1] ||
                                                  (cond_active[cond_depth - 1] != 0);
                }
            }
            else if (TESS_DIR_IS("else"))
            {
                if (cond_depth)
                {
                    bool outer = true;
                    for (int k = 0; k < cond_depth - 1; k++) if (cond_active[k] == 0) outer = false;
                    cond_active[cond_depth - 1] = (!cond_taken[cond_depth - 1] && outer) ? 1 : 0;
                    cond_taken[cond_depth - 1] = cond_taken[cond_depth - 1] || 1;
                }
            }
            else if (TESS_DIR_IS("endif"))
            {
                if (cond_depth) cond_depth--;
            }
            i = e + (e < len);
            continue;
        }
        #undef TESS_DIR_IS

        bool live = true;
        for (int k = 0; k < cond_depth; k++) if (cond_active[k] == 0) live = false;
        if (live)
        {
            for (int w = 0; w < 2; w++)
            {
                const char *nm = names[w];
                size_t nl = strlen(nm);
                const char *q = src + ls;
                while ((q = strstr(q, nm)) && q + nl <= src + e)
                {
                    if (is_ident_char((unsigned char)q[nl]) ||
                        (q > src && is_ident_char((unsigned char)q[-1])) ||
                        (q > src + 1 && !strncmp(q - 3, "gl_", 3) && q[-3] != '.'))
                    {
                        /* gl_in[].gl_ClipDistance / a longer identifier */
                    }
                    else if (q[nl] == '[')
                    {
                        const char *close = strchr(q + nl, ']');
                        if (close && close < src + e)
                        {
                            size_t in = close - (q + nl + 1);
                            const char *after = close + 1;
                            while (after < src + e && (*after == ' ' || *after == '\t')) after++;
                            if (*after == ';')
                            {
                                /* `gl_(Clip|Cull)Distance[N];' — a declaration; an
                                 * empty bracket (`[]') stays unsized */
                                if (in > 0) sized[w] = true;
                            }
                            else if ((after[0] == '=' && after[1] != '=') ||
                                     (after + 1 < src + e && after[1] == '=' &&
                                      (after[0] == '+' || after[0] == '-' ||
                                       after[0] == '*' || after[0] == '/')))
                            {
                                /* a write: only a dynamic index needs the size */
                                if (!tess_pp_index_is_constant(q + nl + 1, in, defs, ndefs, 0))
                                    dynamic_write[w] = true;
                            }
                        }
                    }
                    q += nl;
                }
            }
        }
        i = e + (e < len);
    }

    for (int w = 0; w < 2; w++)
        if (dynamic_write[w] && !sized[w])
        {
            const char *msg = w ? "'gl_CullDistance' : array must be redeclared with an "
                                  "explicit size before indexing with a non-constant expression"
                                : "'gl_ClipDistance' : array must be redeclared with an "
                                  "explicit size before indexing with a non-constant expression";
            size_t n = strlen(src);
            char *out = (char *)malloc(n + strlen(msg) + 32);
            if (!out) return NULL;
            snprintf(out, n + strlen(msg) + 32, "#error %s\n%s", msg, src);
            return out;
        }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* shadow samplers with implicit LOD outside the fragment stage        */
/* ------------------------------------------------------------------ */

/* Collect the identifiers declared as `uniform <...>sampler...Shadow NAME[, NAME2];`.
 * Skips comments so a commented-out declaration does not fool the scan. */
#define TESS_MAX_SHADOW_SAMPLERS 64
static int tess_shadow_sampler_names(const char *src, size_t len,
                                     const char *names[TESS_MAX_SHADOW_SAMPLERS])
{
    int n = 0;
    size_t i = 0;
    while (i < len)
    {
        /* skip comments */
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '/')
        {
            while (i < len && src[i] != '\n') i++;
            continue;
        }
        if (i + 1 < len && src[i] == '/' && src[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < len && !(src[i] == '*' && src[i + 1] == '/')) i++;
            i = i + 2 < len ? i + 2 : len;
            continue;
        }
        /* the `uniform' keyword at an identifier boundary */
        if (strncmp(src + i, "uniform", 7) == 0 &&
            (i == 0 || !isalnum((unsigned char)src[i - 1])) &&
            !isalnum((unsigned char)src[i + 7]))
        {
            /* scan the declaration up to the semicolon: the TYPE is the first
             * identifier that is not a precision/interp qualifier; every later
             * identifier before a ',' or '=' is a name of that type */
            size_t j = i + 7, end = j;
            while (end < len && src[end] != ';') end++;
            const char *type = NULL;
            size_t type_len = 0;
            while (j < end)
            {
                if (isspace((unsigned char)src[j]) || src[j] == ',') { j++; continue; }
                size_t e = j;
                while (e < end && (isalnum((unsigned char)src[e]) || src[e] == '_')) e++;
                if (e == j) { j++; continue; }
                const char *w = src + j;
                static const char *skip[] = { "lowp", "mediump", "highp", "precision", NULL };
                int is_skip = 0;
                for (int k = 0; skip[k]; k++)
                    if (strncmp(w, skip[k], e - j) == 0) { is_skip = 1; break; }
                if (!is_skip)
                {
                    if (!type)
                    {
                        type = w;
                        type_len = e - j;
                    }
                    else if (type_len > 6 && strncmp(type + type_len - 6, "Shadow", 6) == 0)
                    {
                        /* a name of a shadow-sampler type */
                        if (n < TESS_MAX_SHADOW_SAMPLERS) names[n++] = w;
                    }
                }
                j = e;      /* continue right after this token */
            }
            i = end < len ? end + 1 : len;
            continue;
        }
        i++;
    }
    return n;
}

static int tess_name_at(const char *src, size_t len, size_t i,
                        const char *names[TESS_MAX_SHADOW_SAMPLERS], int n,
                        size_t *name_len)
{
    size_t e = i;
    while (e < len && (isalnum((unsigned char)src[e]) || src[e] == '_')) e++;
    if (e == i) return 0;
    for (int k = 0; k < n; k++)
    {
        size_t kl = 0;
        while (names[k][kl] && isalnum((unsigned char)names[k][kl])) kl++;
        if (kl == e - i && strncmp(names[k], src + i, kl) == 0)
        {
            *name_len = kl;
            return 1;
        }
    }
    return 0;
}

/* skip a balanced (...) group starting at src[i] == '(' */
static size_t tess_skip_paren(const char *src, size_t len, size_t i)
{
    int depth = 0;
    while (i < len)
    {
        if (src[i] == '(') depth++;
        else if (src[i] == ')')
        {
            depth--;
            if (depth == 0) return i + 1;
        }
        i++;
    }
    return len;
}

/* Rewrite implicit-lod shadow samples in a NON-fragment stage to the explicit
 * base-level form (see the comment at the call site).  Returns a malloc'd copy,
 * or NULL when the source has no matching call (the caller then keeps `src'). */
char *tess_shadow_implicit_lod(const char *src)
{
    size_t len = strlen(src);
    const char *names[TESS_MAX_SHADOW_SAMPLERS];
    int n = tess_shadow_sampler_names(src, len, names);
    if (getenv("MGL_TESS_SLDBG"))
        fprintf(stderr, "sl-re: n=%d first400=[%.400s]\n", n, src ? src : "");

    char *out = NULL;
    size_t out_len = 0, out_cap = 0;
    size_t i = 0, copied = 0;

#define TESS_OUT_PUSH(p, count)                                                \
    do {                                                                       \
        if (out_len + (count) > out_cap)                                       \
        {                                                                      \
            out_cap = (out_len + (count)) * 2 + 256;                           \
            char *t_ = (char *)realloc(out, out_cap);                          \
            if (!t_) { free(out); return NULL; }                               \
            out = t_;                                                          \
        }                                                                      \
        memcpy(out + out_len, (p), (count));                                   \
        out_len += (count);                                                    \
    } while (0)

    while (i < len)
    {
        /* texture( / textureOffset( at an identifier boundary */
        int fn_len = 0;
        if ((i == 0 || !isalnum((unsigned char)src[i - 1])) && src[i] == 't')
        {
            if (strncmp(src + i, "textureOffset(", 14) == 0 &&
                !isalnum((unsigned char)src[i + 14 - 1]))
                fn_len = 13;                       /* "textureOffset" */
            else if (strncmp(src + i, "texture(", 8) == 0)
                fn_len = 7;                        /* "texture" */
        }
        if (fn_len)
        {
            size_t a = i + fn_len + 1;             /* first char after '(' */
            size_t nl;
            if (tess_name_at(src, len, a, names, n, &nl))
            {
                size_t close = tess_skip_paren(src, len, i + fn_len);
                if (close > a && close <= len && src[close - 1] == ')')
                {
                    /* find the first top-level comma of the argument list */
                    int depth = 0;
                    size_t c1 = 0;
                    for (size_t q = a; q < close - 1; q++)
                    {
                        if (src[q] == '(') depth++;
                        else if (src[q] == ')') depth--;
                        else if (src[q] == ',' && depth == 0) { c1 = q; break; }
                    }
                    TESS_OUT_PUSH(src + copied, i - copied);      /* everything before `texture' */
                    if (fn_len == 13)
                    {
                        /* textureOffset(s, P, off) -> textureLodOffset(s, P, 0.0, off):
                         * the lod goes AFTER P, i.e. after the SECOND top-level comma */
                        size_t c2 = 0;
                        depth = 0;
                        int commas = 0;
                        for (size_t q = a; q < close - 1; q++)
                        {
                            if (src[q] == '(') depth++;
                            else if (src[q] == ')') depth--;
                            else if (src[q] == ',' && depth == 0)
                            {
                                commas++;
                                if (commas == 2) { c2 = q; break; }
                            }
                        }
                        if (c2)
                        {
                            TESS_OUT_PUSH("textureLodOffset", 16);
                            TESS_OUT_PUSH(src + i + 13, c2 - (i + 13));            /* "(s, P" */
                            TESS_OUT_PUSH(", 0.0", 5);
                            TESS_OUT_PUSH(src + c2, close - c2);                   /* ", off)" */
                        }
                        else
                        {
                            /* only one argument: leave it untouched */
                            TESS_OUT_PUSH(src + i, close - i);
                        }
                    }
                    else
                    {
                        /* texture(s, P[, compare]) -> textureLod(s, P[, compare], 0.0) */
                        TESS_OUT_PUSH("textureLod", 10);
                        TESS_OUT_PUSH(src + i + 7, close - (i + 7) - 1);       /* "(args" */
                        TESS_OUT_PUSH(", 0.0)", 6);
                    }
                    copied = close;
                    i = close;
                    continue;
                }
            }
        }
        i++;
    }
    if (!out) return NULL;                        /* nothing rewritten */
    TESS_OUT_PUSH(src + copied, len - copied);
#undef TESS_OUT_PUSH
    /* NUL-terminate */
    if (out_len + 1 > out_cap)
    {
        char *t = (char *)realloc(out, out_len + 1);
        if (!t) { free(out); return NULL; }
        out = t;
    }
    out[out_len] = 0;
    return out;
}

char *mgl_tess_glsl_compat(GLenum type, const char *src)
{
    if (!src ||
        (type != GL_TESS_CONTROL_SHADER && type != GL_TESS_EVALUATION_SHADER &&
         type != GL_GEOMETRY_SHADER && type != GL_VERTEX_SHADER && type != GL_FRAGMENT_SHADER))
        return NULL;
    /* GL_EXT_texture_shadow_lod / core GLSL 4.6: implicit-lod texture() on a shadow
     * sampler outside the fragment stage samples the base level (GL 4.6 8.14.1),
     * but glslang gates the overload behind an extension it only enables for ES
     * ("The following extension must be enabled to use this feature" — an error
     * in a vertex shader; KHR-GL46.ext_texture_shadow_lod's *_vertex cases die
     * on it).  The explicit-lod form is semantically identical here, so rewrite
     * texture(s, P[, compare]) -> textureLod(s, P[, compare], 0.0) and
     * textureOffset(s, P, off) -> textureLodOffset(s, P, 0.0, off) for every
     * sampler*Shadow uniform, in every non-fragment stage. */
    if (type != GL_FRAGMENT_SHADER)
    {
        char *r = tess_shadow_implicit_lod(src);
        if (r)
        {
            char *rest = mgl_tess_glsl_compat(type, r);
            if (rest) { free(r); return rest; }
            return r;
        }
    }
    /* GL_ARB_shader_ballot where Metal has no simdgroup attributes: the TES
     * (post-tessellation vertex function — verified: "invalid
     * 'thread_index_in_simdgroup' attribute ... in a post-tessellation vertex
     * function") and the VERTEX stage (a plain vertex function whenever the
     * pipeline has no tessellation; Metal rejects the attribute there the same
     * way, and MGL cannot know at compile time whether the link will turn the
     * VS into a compute kernel).  SPIRV-Cross refuses both ("Subgroup builtins
     * are not available in this type of function").  The subgroup size is
     * implementation-defined per stage, so emulate a size-1 subgroup in those
     * two stages only — every other MGL stage runs compute/fragment kernels
     * with real 32-wide simdgroups: invocation 0, the lane-0 masks, a one-lane
     * ballot, and the reads degenerate to the value itself. */
    if (type == GL_TESS_EVALUATION_SHADER || type == GL_VERTEX_SHADER)
    {
        char *emulated = tess_ballot_one_lane(src);
        if (emulated)
        {
            /* run the rest of the stage's passes on the rewritten source (the
             * recursion re-enters here; tess_ballot_one_lane is idempotent).
             * The body below returns NULL when it has nothing to rewrite —
             * then the emulated copy is the final result. */
            char *rest = mgl_tess_glsl_compat(type, emulated);
            if (rest) { free(emulated); return rest; }
            return emulated;
        }
    }
    /* vertex AND geometry stages: fold loose `out float gl_*Distance[N];' members
     * into one gl_PerVertex block.  A block interface in one stage and a loose
     * member in the other is exactly the mix glslang's link / io-mapper cannot
     * pair ("vertex block member has no corresponding member..." / "Invalid
     * In/Out variable type : gl_PerVertex"): a program with a tessellation stage
     * (block form, expanded below) and a loose-member VS or GS hits it, and the
     * CTS writes the same members loose in every stage - make them uniform. */
    if (type == GL_VERTEX_SHADER || type == GL_GEOMETRY_SHADER)
    {
        char *blocked = tess_vs_member_to_block(src);
        if (blocked)
        {
            /* the blocked source has no loose members left: the recursive call
             * goes straight to the injection/expansion below */
            char *next = mgl_tess_glsl_compat(type, blocked);
            if (next) { free(blocked); return next; }
            return blocked;
        }
    }

    if (!strstr(src, "gl_PerVertex"))
        return NULL;

    /* a stage that INDEXES gl_in[..].gl_ClipDistance / gl_CullDistance with a
     * variable must redeclare its input block with a size first ("array must be
     * redeclared with a size before being indexed with a variable"); inject the
     * full-member block when the source reads the distance members through gl_in */
    if ((type == GL_TESS_CONTROL_SHADER || type == GL_TESS_EVALUATION_SHADER ||
         type == GL_GEOMETRY_SHADER) &&
        strstr(src, "gl_in") && !strstr(src, "in gl_PerVertex") &&
        (strstr(src, "gl_ClipDistance") || strstr(src, "gl_CullDistance")))
    {
        int version = 0;
        const char *v = strstr(src, "#version");
        if (v) version = atoi(v + strlen("#version"));
        if (version < 450 && strstr(src, "gl_CullDistance")) version = 0;
        if (version)
        {
            char *with_in = NULL;
            size_t wl = 0;
            /* the injected members must not pin sizes the producer contradicts
             * (clip=1 + cull=7 programs): take the stage's OWN declared size for
             * each array - programs declare the same size in every stage - and
             * leave it unsized when this stage does not mention it */
            char cs[32], us[32];
            tess_declared_size(src, "gl_ClipDistance", cs, sizeof cs);
            tess_declared_size(src, "gl_CullDistance", us, sizeof us);
            FILE *m2 = open_memstream(&with_in, &wl);
            if (m2)
            {
                const char *nl = v ? strchr(v, '\n') : src;
                size_t head = nl ? (size_t)(nl - src) + 1 : 0;
                fwrite(src, 1, head, m2);
                fprintf(m2, "in gl_PerVertex { vec4 gl_Position; float gl_PointSize;"
                            " float gl_ClipDistance[%s]; float gl_CullDistance[%s]; } gl_in[];\n",
                        cs[0] ? cs : "", us[0] ? us : "");
                fputs(src + head, m2);
                fclose(m2);
                if (with_in)
                {
                    char *next = mgl_tess_glsl_compat(type, with_in);
                    if (next) { free(with_in); return next; }
                    return with_in;   /* the expansion declined: the injection stands */
                }
            }
        }
    }

    /* gl_CullDistance is an ARB extension before GLSL 4.50: expanding a subset
     * redeclaration with it would fail ("block redeclaration has extra members") */
    int version = 0;
    {
        const char *v = strstr(src, "#version");
        if (v)
            version = atoi(v + strlen("#version"));
    }
    const char *pervtx_member[4] = {
        "vec4 gl_Position",
        "float gl_PointSize",
        "float gl_ClipDistance[]",            /* unsized: takes the producer's size at link */
        version >= 450 ? "float gl_CullDistance[]" : NULL,
    };

    char *out = NULL;
    size_t out_len = 0;
    FILE *mem = open_memstream(&out, &out_len);
    char *tokctx = NULL, *tokctx2 = NULL;
    if (!mem)
        return NULL;

    size_t i = 0;
    while (src[i])
    {
        const char *pv = strstr(src + i, "gl_PerVertex");
        if (!pv)
        {
            fputs(src + i, mem);
            break;
        }
        size_t at = (size_t)(pv - src);
        size_t b = tess_skip_ws_comments(src, at + strlen("gl_PerVertex"));
        if (src[b] != '{')
        {
            /* not a redeclaration (a comment, or an unrelated identifier): copy through */
            fwrite(src + i, 1, at + strlen("gl_PerVertex") - i, mem);
            i = at + strlen("gl_PerVertex");
            continue;
        }
        /* find the closing brace of the member list */
        size_t e = b + 1;
        int depth = 1;
        while (src[e] && depth)
        {
            e = tess_skip_ws_comments(src, e);
            if (src[e] == '{') depth++;
            else if (src[e] == '}') depth--;
            else if (src[e]) e++;
        }
        if (!src[e])
        {
            /* unterminated: copy one character and continue */
            fwrite(src + i, 1, at - i, mem);
            i = at;
            fputc(src[i++], mem);
            continue;
        }
        /* collect the declared member names, preserving order */
        char body[1024];
        size_t n = e - (b + 1);
        if (n >= sizeof(body)) n = sizeof(body) - 1;
        memcpy(body, src + b + 1, n);
        body[n] = 0;
        int have[4] = { 0, 0, 0, 0 };
        for (char *decl = strtok_r(body, ";", &tokctx); decl; decl = strtok_r(NULL, ";", &tokctx))
        {
            char name[64] = "";
            char last[64] = "";
            /* the member name is the final identifier of the declaration (an array's
             * "float gl_ClipDistance[]" tokenizes as "gl_ClipDistance[]") */
            for (char *t = strtok_r(decl, " \t\r\n", &tokctx2); t; t = strtok_r(NULL, " \t\r\n", &tokctx2))
                snprintf(last, sizeof(last), "%s", t);
            size_t ln = strlen(last);
            if (ln > 1 && last[ln - 1] == ']')
            {
                /* an array declarator's "[...]"/"[]" suffix is not part of the
                 * member name: strip back to the matching '[' so sized and
                 * unsized redeclarations of gl_ClipDistance / gl_CullDistance
                 * compare equal ("float gl_CullDistance[8]" left the name as
                 * "gl_CullDistance[8" and the member was expanded in twice) */
                size_t o = ln - 1;
                while (o > 1 && last[o - 1] != '[') o--;
                if (last[o - 1] == '[') last[o - 1] = 0;   /* cut at '[' */
                else last[ln - 1] = 0;                     /* no bracket found */
            }
            snprintf(name, sizeof(name), "%s", last);
            for (int k = 0; k < 4; k++)
                if (!strcmp(name, tess_pervtx_name[k]))
                    have[k] = 1;
        }
        int missing = 0;
        for (int k = 0; k < 4; k++) missing += !have[k];
        if (!missing)
        {
            fwrite(src + i, 1, e + 1 - i, mem);
            i = e + 1;
            continue;
        }
        /* splice: original member list + the missing built-ins */
        fwrite(src + i, 1, e - i, mem);          /* up to (not including) '}' */
        for (int k = 0; k < 4; k++)
            if (!have[k] && pervtx_member[k])
                fprintf(mem, "%s    %s;\n", n ? "" : "\n", pervtx_member[k]);
        fputc('}', mem);
        i = e + 1;
    }
    fclose(mem);
    if (!out)
        return NULL;
    if (tess_trace())
    {
        fprintf(stderr, "MGL tess: expanded a subset gl_PerVertex redeclaration (stage %x):\n%s\n",
                (unsigned)type, out);
    }
    return out;
}

/* ------------------------------------------------------------------ SPIR-V
 * Minimal walker: execution modes of the tessellation stages.
 */

#define SPV_OP_ENTRYPOINT      15
#define SPV_OP_EXECUTIONMODE   16
#define SPV_OP_CAPABILITY      17

/* values from spirv.h (SpvExecutionMode*) */
#define SPV_EM_SPACING_EQUAL                1
#define SPV_EM_SPACING_FRACTIONAL_EVEN      2
#define SPV_EM_SPACING_FRACTIONAL_ODD       3
#define SPV_EM_VERTEX_ORDER_CW              4
#define SPV_EM_VERTEX_ORDER_CCW             5
#define SPV_EM_POINT_MODE                   10
#define SPV_EM_TRIANGLES                    22
#define SPV_EM_QUADS                        24
#define SPV_EM_ISOLINES                     25
#define SPV_EM_OUTPUT_VERTICES              26
#define SPV_EM_POST_DEPTH_COVERAGE        4426
#define SPV_EXECUTION_MODEL_FRAGMENT        4

/* collect execution modes of the tessellation entry point (model-filtered) */
static bool spv_tess_modes(const uint32_t *ir, size_t n, unsigned model_wanted,
                           int modes[16], int *count, int *out_vertices)
{
    size_t i = 5;
    uint32_t entry_id = 0;

    *count = 0;
    if (out_vertices) *out_vertices = 0;
    if (!ir || n < 5 || ir[0] != 0x07230203)
        return false;

    while (i < n)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint32_t op = word & 0xffff;

        if (wc == 0 || i + wc > n)
            break;

        if (op == SPV_OP_ENTRYPOINT && wc >= 2 && ir[i + 1] == model_wanted)
            entry_id = ir[i + 2];
        i += wc;
    }

    if (!entry_id)
        return false;

    i = 5;
    while (i < n)
    {
        uint32_t word = ir[i];
        uint32_t wc = word >> 16;
        uint32_t op = word & 0xffff;

        if (wc == 0 || i + wc > n)
            break;

        if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && ir[i + 1] == entry_id)
        {
            uint32_t mode = ir[i + 2];
            if (mode == SPV_EM_OUTPUT_VERTICES && wc >= 4 && out_vertices)
                *out_vertices = (int)ir[i + 3];
            if (*count < 16)
                modes[(*count)++] = (int)mode;
        }
        i += wc;
    }
    return true;
}

static int spv_tess_domain(const uint32_t *ir, size_t n, int *spacing, int *ccw, int *point_mode)
{
    int modes[16], count = 0, domain = -1;

    if (!spv_tess_modes(ir, n, 2 /* SpvExecutionModelTessellationEvaluation */, modes, &count, NULL))
        return -1;

    *spacing = MGL_TESS_SPACING_EQUAL;
    *ccw = 1;
    *point_mode = 0;
    domain = MGL_TESS_DOMAIN_TRIANGLES;      /* SPIR-V default when no domain mode is given */
    for (int i = 0; i < count; i++)
    {
        switch (modes[i])
        {
            case SPV_EM_TRIANGLES: domain = MGL_TESS_DOMAIN_TRIANGLES; break;
            case SPV_EM_QUADS:     domain = MGL_TESS_DOMAIN_QUADS; break;
            case SPV_EM_ISOLINES:  domain = MGL_TESS_DOMAIN_ISOLINES; break;
            case SPV_EM_SPACING_EQUAL:           *spacing = MGL_TESS_SPACING_EQUAL; break;
            case SPV_EM_SPACING_FRACTIONAL_EVEN: *spacing = MGL_TESS_SPACING_FRACTIONAL_EVEN; break;
            case SPV_EM_SPACING_FRACTIONAL_ODD:  *spacing = MGL_TESS_SPACING_FRACTIONAL_ODD; break;
            case SPV_EM_VERTEX_ORDER_CCW: *ccw = 1; break;
            case SPV_EM_VERTEX_ORDER_CW:  *ccw = 0; break;
            case SPV_EM_POINT_MODE: *point_mode = 1; break;
            default: break;
        }
    }
    return domain;
}

/* isolines are emulated on the quad domain: SPIRV-Cross refuses ExecutionModeIsolines.
 * returns a copy of the module with Isolines replaced by Quads (and dynamic interface
 * member-array indices lowered), or NULL when neither applies. */uint32_t *mgl_tess_prepare_spirv(const uint32_t *words, size_t n, size_t *out_n)
{
    bool has_isolines = false;
    uint32_t entry_id = 0;
    struct SpvDissolveMap dmaps[64];
    uint32_t n_dmaps = 0;
    const uint32_t *cur;
    uint32_t *dis, *copy = NULL;
    size_t i = 5, cn;

    *out_n = 0;
    if (!words || n < 5)
        return NULL;

    /* per-vertex INPUT interface blocks first: SPIRV-Cross mis-resolves (and
     * aborts on) access chains into their split members, so dissolve them
     * into one plain per-vertex variable per block member before anything
     * else looks at the module (tess_spv_lower.c) */
    {
        int partial = 0;

        dis = mgl_tess_dissolve_vertex_input_blocks(words, n, &cn, dmaps, 64,
                                                    &n_dmaps, &partial);
        if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess prepare: dissolve -> %s (%zu words, %u maps, partial=%d)\n",
                    dis ? (mgl_tess_module_declined(dis, cn) ? "DECLINED(marker)" : "ok") : "NULL",
                    cn, n_dmaps, partial);
        if (dis && mgl_tess_module_declined(dis, cn))
        {
            /* declined marker (whole struct-typed interface loads): nothing else
             * may rebuild this module — a rebuild would resurrect a module that
             * aborts SPIRV-Cross; return the marker so every compile fails
             * cleanly and the link reports a stage error */
            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL tess prepare: dissolve DECLINED (marker) — module unusable\n");
            *out_n = cn;
            return dis;
        }

        /* the module the lower runs on (and falls back to) */
        {
            const uint32_t *lbase = dis ? dis : words;
            size_t lbase_n = dis ? cn : n;

            cur = lbase;
            cn = lbase_n;
        }
        if (dis)
        {
            /* re-scan execution modes on the dissolved module below */
        }

        i = 5;
        while (i < cn)
        {
            uint32_t wc = cur[i] >> 16, op = cur[i] & 0xffff;
            if (wc == 0 || i + wc > cn) break;
            if (op == SPV_OP_ENTRYPOINT && wc >= 4 && cur[i + 1] == 2 /* TessellationEvaluation */)
                entry_id = cur[i + 2];
            i += wc;
        }

        if (!entry_id)
        {
            free(dis);
            return NULL;
        }

        i = 5;
        while (i < cn)
        {
            uint32_t wc = cur[i] >> 16, op = cur[i] & 0xffff;
            if (wc == 0 || i + wc > cn) break;
            if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && cur[i + 1] == entry_id &&
                cur[i + 2] == SPV_EM_ISOLINES)
                has_isolines = true;
            i += wc;
        }

        if (!has_isolines)
        {
            /* even without isolines, dynamic member-array indexing may need
             * lowering; when that did not fire the dissolve result (if any) is
             * still the module to use.
             *
             * The lowering rewrites the dynamic chains to constant-indexed
             * copies in function-local storage, re-rooted at the DISSOLVED
             * member variables when the dissolve had to leave a chain on the
             * original block (dynamic member indices): the block variable
             * must end up unreferenced, otherwise its attribute span collides
             * with the dissolved variables' on the stage-in.  When the lower
             * declines such a PARTIAL module, fall back to the original
             * words, which SPIRV-Cross rejects cleanly (link error). */
            uint32_t *lowered = mgl_tess_lower_dynamic_io(cur, cn, out_n,
                                                          dis ? dmaps : NULL,
                                                          dis ? n_dmaps : 0);

            if (lowered)
            {
                if (getenv("MGL_TESS_DUMP_PREP"))
                {
                    char path[512];
                    snprintf(path, sizeof path, "%s/prep_lowered.spv", getenv("MGL_TESS_DUMP_PREP"));
                    FILE *f = fopen(path, "wb");
                    if (f) { fwrite(lowered, 4, *out_n, f); fclose(f); }
                }
                if (lowered != dis)
                    free(dis);
                return lowered;
            }
            if (partial && dis)
            {
                /* declined with chains still on the block: the dissolved
                 * module would collide on the attribute slots */
                if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL tess prepare: lower declined a partial dissolve — using the original module\n");
                free(dis);
                if (mgl_tess_module_has_block_leafmember_chains(words, n))
                {
                    /* the original module would ABORT SPIRV-Cross (direct
                     * leaf-member chains on a composite block): decline with
                     * a corrupted-magic marker so every compile fails
                     * cleanly and the link reports a stage error */
                    uint32_t *bad = (uint32_t *)malloc(64);

                    if (bad)
                    {
                        bad[0] = 0x07230203u ^ 1u;
                        *out_n = 16;
                        return bad;
                    }
                }
                *out_n = n;
                cur = malloc(n * 4);
                if (cur)
                {
                    memcpy(cur, words, n * 4);
                    return cur;
                }
                *out_n = 0;
                return NULL;
            }
            if (!dis && mgl_tess_module_has_block_leafmember_chains(words, n))
            {
                /* no dissolve at all (bail) and the lower declined: the
                 * original module's leaf-member chains would abort */
                uint32_t *bad = (uint32_t *)malloc(64);

                if (bad)
                {
                    bad[0] = 0x07230203u ^ 1u;
                    *out_n = 16;
                    return bad;
                }
            }
            *out_n = cn;
            return dis;
        }
    }

    copy = (uint32_t *)malloc(cn * sizeof(uint32_t));
    if (!copy)
    {
        free(dis);
        return NULL;
    }
    memcpy(copy, cur, cn * sizeof(uint32_t));
    free(dis);

    i = 5;
    while (i < cn)
    {
        uint32_t wc = copy[i] >> 16, op = copy[i] & 0xffff;
        if (wc == 0 || i + wc > cn) break;
        if (op == SPV_OP_EXECUTIONMODE && wc >= 3 && copy[i + 1] == entry_id &&
            copy[i + 2] == SPV_EM_ISOLINES)
            copy[i + 2] = SPV_EM_QUADS;     /* isolines -> quad domain emulation */
        i += wc;
    }

    {
        /* SPIRV-Cross cannot codegen dynamic indices into interface member
         * arrays for TES inputs — lower them (tess_spv_lower.c).  The
         * dissolve maps apply VERBATIM to this module: the isolines->quad
         * swap above only rewrites one ExecutionMode word, every id (the
         * block variables, their members, the dissolved variables) is
         * unchanged.  Passing NULL here left the dynamic-member sites
         * re-rooted at the ORIGINAL block: SPIRV-Cross then flattened the
         * block AND the dissolved variables — duplicate [[attribute(N)]]
         * and a link failure (dEQP per_vertex_block isolines). */
        size_t ln = 0;
        uint32_t *lowered = mgl_tess_lower_dynamic_io(copy, cn, &ln,
                                                      dis ? dmaps : NULL,
                                                      dis ? n_dmaps : 0);
        if (lowered)
        {
            free(copy);
            copy = lowered;
            cn = ln;
        }
    }

    *out_n = cn;
    return copy;
}

/* Apple GPUs expose 31 vertex-attribute slots; a GL TES may declare
 * gl_MaxTessEvaluationInputComponents/4 = 32 vec4s.  Count the SPIRV-Cross
 * attribute annotations on stage inputs. */
bool mgl_tess_tes_msl_exceeds_attributes(const char *msl)
{
    if (!msl)
        return false;
    if (!strstr(msl, "patch_control_point<"))
        return false;               /* not a post-tessellation function */
    int n = 0;
    for (const char *p = msl; (p = strstr(p, "[[attribute(")) != NULL; p += 12)
        n++;
    return n > 30;
}

struct MslStruct;
static bool tess_cp_record_name(const char *msl, char *out, size_t out_len);
static void tess_free_pair(GLMContext ctx, MGLTessProgram *pair);
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len);
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len);
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out, char *name, size_t name_len);
static bool msl_tcs_record_names(const char *msl, char *in_name, size_t in_len,
                                 char *out_name, size_t out_len);
static void tess_canonicalize_records(GLMContext ctx, MGLTessProgram *tp);
static char *tcs_inject_float_levels(const char *msl);
static char *tcs_redirect_tri_inner1(char *msl);
static char *tcs_patch_primid(const char *msl);

/* ------------------------------------------------------------------ MSL struct layout
 * Parse an MSL interface struct declaration and compute the byte layout Metal
 * uses for it (C++ rules: natural alignment, arrays padded to element alignment).
 */

struct MslStruct {
    char body[8192];                       /* full declaration text */
    MGLTessPatchMember members[MGL_TESS_MAX_PATCH_MEMBERS];
    int count;
    size_t size;
    int cp_ref_member;                     /* index of the patch_control_point member */
};

/* element alignment/size of a base MSL type name (no array suffix) */
static bool msl_scalar_type(const char *t, size_t *align, size_t *size)
{
    struct { const char *name; size_t a, s; } tab[] = {
        { "float",  4, 4 }, { "half",  2, 2 },
        { "int",    4, 4 }, { "uint",  4, 4 }, { "bool", 1, 1 },
        { "short",  2, 2 }, { "ushort", 2, 2 }, { "char", 1, 1 }, { "uchar", 1, 1 },
        { "float2", 8, 8 }, { "half2",  4, 4 }, { "int2",  8, 8 }, { "uint2", 8, 8 },
        { "float3", 16, 16 }, { "half3",  8, 8 }, { "int3", 16, 16 }, { "uint3", 16, 16 },
        { "float4", 16, 16 }, { "half4",  8, 8 }, { "int4", 16, 16 }, { "uint4", 16, 16 },
        { "double", 8, 8 }, { "long", 8, 8 }, { "ulong", 8, 8 },
        { "double2", 16, 16 }, { "long2", 16, 16 }, { "ulong2", 16, 16 },
        { "double3", 32, 32 }, { "long3", 32, 32 }, { "ulong3", 32, 32 },
        { "double4", 32, 32 }, { "long4", 32, 32 }, { "ulong4", 32, 32 },
        { NULL, 0, 0 }
    };
    for (int i = 0; tab[i].name; i++)
        if (!strcmp(t, tab[i].name)) { *align = tab[i].a; *size = tab[i].s; return true; }
    return false;
}

/* matrix floatCxR: named like float4x4 (4 columns of float4) */
static bool msl_matrix_type(const char *t, size_t *align, size_t *size)
{
    int c, r;
    if (strlen(t) != 8 || t[6] != 'x' || strncmp(t, "float", 5))
        return false;
    c = t[5] - '0'; r = t[7] - '0';
    if (c < 2 || c > 4 || r < 2 || r > 4)
        return false;
    static const size_t colalign[5] = { 0, 4, 8, 16, 16 };
    static const size_t colsize[5]  = { 0, 4, 8, 16, 16 };
    *align = colalign[r];                 /* alignment of the column vector type */
    *size = (size_t)c * colsize[r];       /* C columns, each padded to column size */
    return true;
}

static size_t round_up(size_t v, size_t a) { return a ? ((v + a - 1) / a) * a : v; }

/* alignment/size of a possibly-arrayed element type: "float3", "spvUnsafeArray<float3, 4>", "float4x4" */
/* the comma separating the element type from N in "spvUnsafeArray<T, N>" — the
 * FIRST comma at angle-bracket depth 0 (T may itself be an array: commas inside) */
static const char *msl_array_split(const char *t)
{
    int depth = 0;

    if (!t)
        return NULL;
    for (const char *p = t + 15; *p; p++)
    {
        if (*p == '<')
            depth++;
        else if (*p == '>')
        {
            if (depth == 0)
                return NULL;
            depth--;
        }
        else if (*p == ',' && depth == 0)
            return p;
    }
    return NULL;
}

static bool msl_element_type(const char *t, size_t *align, size_t *size)
{
    if (!strncmp(t, "spvUnsafeArray<", 15))
    {
        char inner[96];
        long n = 0;
        /* spvUnsafeArray<T, N> */
        const char *comma = msl_array_split(t);
        const char *close = comma ? strchr(comma, '>') : NULL;
        if (!comma || !close || (size_t)(comma - (t + 15)) >= sizeof(inner))
            return false;
        memcpy(inner, t + 15, comma - (t + 15));
        inner[comma - (t + 15)] = 0;
        n = strtol(comma + 1, NULL, 10);
        if (n <= 0)
            return false;
        size_t ia, is;
        if (!msl_element_type(inner, &ia, &is))
            return false;
        *align = ia;
        *size = round_up((size_t)n * round_up(is, ia), ia);
        return true;
    }
    if (msl_scalar_type(t, align, size))
        return true;
    if (msl_matrix_type(t, align, size))
        return true;
    return false;
}

/* "spvUnsafeArray<T, N>" -> T (nesting-aware) and N; the element STRIDE is the
 * T copy rounded to T's own alignment (Metal's C layout of T elements[N]) */
static bool msl_array_inner(const char *t, char *inner, size_t cap, long *n)
{
    int depth = 0;
    const char *p;

    if (!t || strncmp(t, "spvUnsafeArray<", 15))
        return false;
    for (p = t + 15; *p; p++)
    {
        if (*p == '<')
            depth++;
        else if (*p == '>')
        {
            if (depth == 0)
                break;
            depth--;
        }
        else if (*p == ',' && depth == 0)
            break;
    }
    if (*p != ',' || p == t + 15)
        return false;
    if ((size_t)(p - (t + 15)) + 1 > cap)
        return false;
    memcpy(inner, t + 15, (size_t)(p - (t + 15)));
    inner[p - (t + 15)] = 0;
    {
        const char *gt = strchr(p, '>');

        if (!gt)
            return false;
        *n = strtol(p + 1, NULL, 10);
    }
    return *n > 0;
}

static size_t msl_array_stride(const char *t)
{
    char inner[96];
    long n = 0;
    size_t a = 1, s = 0;

    if (!msl_array_inner(t, inner, sizeof inner, &n))
        return 0;
    if (!msl_element_type(inner, &a, &s))
        return 0;
    return round_up(s, a);
}

/* find "struct NAME { ... };" in msl; parse members and layout */
static bool msl_parse_struct(const char *msl, const char *name, struct MslStruct *out){
    char head[256];
    const char *p, *open, *close;
    size_t off = 0;
    size_t cursor = 0;

    snprintf(head, sizeof head, "struct %s\n{", name);
    p = strstr(msl, head);
    if (!p)
    {
        /* tolerate single-line form */
        snprintf(head, sizeof head, "struct %s {", name);
        p = strstr(msl, head);
        if (!p)
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: struct %s not found\n", name);
            return false;
        }
    }
    open = strchr(p, '{');
    close = strchr(open, '}');
    if (!open || !close || (size_t)(close - open) >= sizeof(out->body))
        return false;

    memcpy(out->body, open + 1, close - open - 1);
    out->body[close - open - 1] = 0;
    out->count = 0;
    out->cp_ref_member = -1;
    out->size = 0;

    /* split members on ';' */
    while (off < strlen(out->body))
    {
        char member[512], type[256], namebuf[128];
        const char *semi = strchr(out->body + off, ';');
        const char *attr;
        long len;
        char *last;
        int i;

        if (!semi)
            break;
        len = semi - (out->body + off);
        if (len <= 0 || len >= (long)sizeof(member))
        {
            off = (semi - out->body) + 1;
            continue;
        }
        memcpy(member, out->body + off, (size_t)len);
        member[len] = 0;
        off = (semi - out->body) + 1;

        /* strip [[...]] attributes, recording an [[attribute(N)]] number */
        int attr_num = -1;
        attr = strstr(member, "[[");
        if (attr)
        {
            const char *aend = strstr(attr, "]]");
            if (!aend) continue;
            {
                const char *adec = strstr(attr, "attribute(");
                if (adec && adec < aend)
                    attr_num = (int)strtol(adec + 10, NULL, 10);
            }
            memmove((void *)attr, (void *)(aend + 2), strlen(aend + 2) + 1);
        }

        /* trim whitespace on both ends: "type name" or "type name[N]" */
        {
            char *end = member + strlen(member);
            while (end > member && (end[-1] == ' ' || end[-1] == '\n' || end[-1] == '\t'))
                *--end = 0;
            while (*member == ' ' || *member == '\n' || *member == '\t')
                memmove(member, member + 1, strlen(member));
        }
        last = strrchr(member, ' ');
        if (!last) continue;
        snprintf(type, sizeof type, "%.*s", (int)(last - member), member);
        snprintf(namebuf, sizeof namebuf, "%s", last + 1);

        /* trailing array "name[4]" */
        char *brk = strchr(namebuf, '[');
        long arr = 1;
        if (brk)
        {
            arr = strtol(brk + 1, NULL, 10);
            if (arr <= 0) arr = 1;
            *brk = 0;
            /* move the count into the element type: "T[N]" */
            char witharr[300];
            snprintf(witharr, sizeof witharr, "spvUnsafeArray<%s, %ld>", type, arr);
            snprintf(type, sizeof type, "%s", witharr);
        }

        if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS)
            break;

        if (!strncmp(type, "patch_control_point<", 20))
        {
            out->cp_ref_member = out->count;
            out->members[out->count].attr = -1;
            out->members[out->count].is_tess_level = 0;
            snprintf(out->members[out->count].type, sizeof(out->members[out->count].type), "%s", type);
            snprintf(out->members[out->count].name, sizeof(out->members[out->count].name), "%s", namebuf);
            out->members[out->count].offset = 0;
            out->members[out->count].size = 0;      /* not part of the blob */
            out->count++;
            continue;
        }

        size_t align = 0, size = 0;
        size_t malign = 0, msize = 0;
        if (msl_matrix_type(type, &malign, &msize))
        {
            /* the stages whose MSL flattens interfaces split a matrix into its columns
             * ("<name>_<i>"): split it here too so both sides of the record agree */
            int c = type[5] - '0', r = type[7] - '0';
            char coltype[16];
            size_t ca = 0, cs = 0;

            snprintf(coltype, sizeof coltype, "float%d", r);
            msl_element_type(coltype, &ca, &cs);
            for (int col = 0; col < c; col++)
            {
                if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS) break;
                cursor = round_up(cursor, ca);
                out->members[out->count].offset = cursor;
                out->members[out->count].size = cs;
                out->members[out->count].attr = attr_num;
                out->members[out->count].array_idx = 0;     /* set by tess_annotate_keys */
                out->members[out->count].array_idx2 = 0;
                    out->members[out->count].array_idx3 = 0;
                out->members[out->count].flat_dims = 0;
                out->members[out->count].is_tess_level = 0;
                snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", coltype);
                snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s_%d", namebuf, col);
                cursor += cs;
                out->count++;
            }
            continue;
        }
        if (!msl_element_type(type, &align, &size))
        {
            /* an array of structs ("spvUnsafeArray<S, N> inst"): the canonical record
             * keeps ONE ARRAY PER FIELD ("inst_<field>" of N elements) so a dynamic
             * element index stays a real array index in the rewritten stage body
             * (".inst[i].x" -> ".inst_x[i]"; dEQP ES31 user_defined_io blockSa) */
            char elem[128] = "";
            long es = 1;
            if (!strncmp(type, "spvUnsafeArray<", 15))
            {
                const char *cm = msl_array_split(type);

                if (cm && (size_t)(cm - (type + 15)) < sizeof elem)
                {
                    const char *cnt = cm + 1;
                    memcpy(elem, type + 15, (size_t)(cm - (type + 15)));
                    elem[cm - (type + 15)] = 0;
                    while (*cnt == ' ') cnt++;
                    es = strtol(cnt, NULL, 10);
                    if (es <= 0 || es > 64) es = 1;
                }
            }

            if (elem[0])
            {
                struct MslStruct sub;
                size_t run_end = 0;

                if (!msl_parse_struct(msl, elem, &sub))
                {
                    /* arrays-of-arrays — a dissolved patch-block ARRAY member
                     * ("spvUnsafeArray<spvUnsafeArray<S, Ni>, Ne>"): flatten the
                     * struct per field with BOTH dimensions wrapped around the
                     * field type (Blk[2] with S{..z[2]..}: "<n>_z" : float[e][i][j]) */
                    long es2 = 1;
                    char in2[128] = "";

                    if (strncmp(elem, "spvUnsafeArray<", 15) == 0)
                    {
                        const char *cm2 = msl_array_split(elem);
                        const char *cnt2 = cm2 ? cm2 + 1 : NULL;

                        if (cm2 && (size_t)(cm2 - (elem + 15)) < sizeof in2)
                        {
                            memcpy(in2, elem + 15, (size_t)(cm2 - (elem + 15)));
                            in2[cm2 - (elem + 15)] = 0;
                            while (*cnt2 == ' ') cnt2++;
                            es2 = strtol(cnt2, NULL, 10);
                            if (es2 <= 0 || es2 > 64) es2 = 1;
                        }
                    }
                    if (in2[0] && msl_parse_struct(msl, in2, &sub))
                    {
                        for (int f = 0; f < sub.count; f++)
                        {
                            char wrap[256], wrap2[320];
                            size_t fa = 0, fs = 0;

                            if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS) break;
                            if (sub.members[f].size == 0) continue;
                            if (!msl_element_type(sub.members[f].type, &fa, &fs))
                                continue;
                            snprintf(wrap, sizeof wrap, "spvUnsafeArray<%s, %ld>",
                                     sub.members[f].type, es2);
                            snprintf(wrap2, sizeof wrap2, "spvUnsafeArray<%s, %ld>", wrap, es);
                            cursor = round_up(cursor, fa);
                            out->members[out->count].offset = cursor;
                            out->members[out->count].size =
                                (size_t)es * (size_t)es2 * round_up(fs, fa);

                            out->members[out->count].attr = 0;
                            out->members[out->count].array_idx = 0;
                            out->members[out->count].array_idx2 = 0;
                    out->members[out->count].array_idx3 = 0;
                            out->members[out->count].array_idx3 = 0;
                            out->members[out->count].flat_dims = 0;
                            out->members[out->count].is_tess_level = 0;
                            snprintf(out->members[out->count].type,
                                     sizeof out->members[out->count].type, "%s", wrap2);
                            snprintf(out->members[out->count].name,
                                     sizeof out->members[out->count].name, "%s_%s",
                                     namebuf, sub.members[f].name);
                            cursor += out->members[out->count].size;
                            if (cursor > run_end) run_end = cursor;
                            out->count++;
                        }
                        cursor = run_end;
                        continue;
                    }
                    continue;   /* unsupported member type: skip */
                }

                for (int f = 0; f < sub.count; f++)
                {
                    char ftype[160];
                    size_t fa = 0, fs = 0;

                    if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS) break;
                    if (sub.members[f].size == 0) continue;
                    if (!msl_element_type(sub.members[f].type, &fa, &fs))
                        continue;   /* a struct-in-struct field: unsupported */
                    /* the field array: N copies of the field; an array field nests
                     * ("spvUnsafeArray<spvUnsafeArray<float, 2>, N>") */
                    snprintf(ftype, sizeof ftype, "spvUnsafeArray<%s, %ld>",
                             sub.members[f].type, es);
                    cursor = round_up(cursor, fa);
                    out->members[out->count].offset = cursor;
                    out->members[out->count].size = (size_t)es * round_up(fs, fa);
                    out->members[out->count].attr = attr_num;
                    out->members[out->count].array_idx = 0;  /* set by tess_annotate_keys */
                    out->members[out->count].array_idx2 = 0;
                    out->members[out->count].array_idx3 = 0;
                    out->members[out->count].flat_dims = 0;
                    out->members[out->count].is_tess_level = 0;
                    snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", ftype);
                    snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s_%s", namebuf, sub.members[f].name);
                    cursor += out->members[out->count].size;
                    if (cursor > run_end) run_end = cursor;
                    out->count++;
                }
                cursor = run_end;
                continue;
            }

            /* a struct-typed member ("<Struct> inst;"): inline the struct's own fields
             * ("<inst>_<field>"), matching the flattened form the other stages use */
            struct MslStruct sub;

            if (msl_parse_struct(msl, type, &sub))
            {
                size_t salign = 1;

                for (int f = 0; f < sub.count; f++)
                {
                    size_t fa = 0, fs = 0;

                    if (sub.members[f].size == 0) continue;
                    if (msl_element_type(sub.members[f].type, &fa, &fs) && fa > salign)
                        salign = fa;
                }
                cursor = round_up(cursor, salign);
                for (int f = 0; f < sub.count; f++)
                {
                    if (out->count >= MGL_TESS_MAX_PATCH_MEMBERS) break;
                    if (sub.members[f].size == 0) continue;
                    out->members[out->count].offset = cursor + sub.members[f].offset;
                    out->members[out->count].size = sub.members[f].size;
                    out->members[out->count].attr = attr_num;
                    out->members[out->count].array_idx = 0; /* set by tess_annotate_keys */
                    out->members[out->count].array_idx2 = 0;
                    out->members[out->count].array_idx3 = 0;
                    out->members[out->count].flat_dims = 0;
                    out->members[out->count].is_tess_level = 0;
                    snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", sub.members[f].type);
                    snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s_%s", namebuf, sub.members[f].name);
                    out->count++;
                }
                cursor = round_up(cursor + sub.size, salign);
                continue;
            }
            continue;   /* unsupported member type: skip (shader did not use it in a layout we know) */
        }

        cursor = round_up(cursor, align);
        out->members[out->count].offset = cursor;
        out->members[out->count].size = size;
        out->members[out->count].attr = attr_num;
        out->members[out->count].array_idx = 0;     /* set by tess_annotate_keys */
        out->members[out->count].array_idx2 = 0;
                    out->members[out->count].array_idx3 = 0;
        out->members[out->count].flat_dims = 0;
        out->members[out->count].is_tess_level =
            !strcmp(namebuf, "gl_TessLevel") || !strcmp(namebuf, "gl_TessLevelOuter") ? 1 :
            !strcmp(namebuf, "gl_TessLevelInner") ? 2 : 0;
        snprintf(out->members[out->count].type, sizeof out->members[out->count].type, "%s", type);
        snprintf(out->members[out->count].name, sizeof out->members[out->count].name, "%s", namebuf);
        cursor += size;
        out->count++;
        (void)i;
    }

    out->size = cursor;
    return true;        /* the struct was found; count may be 0 (no data members) */
}

/* ------------------------------------------------------------------ canonical records
 * The per-vertex control point record and the per-patch record are written by one
 * stage (the VS kernel / the TCS kernel) and read by another (the TCS kernel / the
 * TES). SPIRV-Cross emits a struct per stage holding exactly the members THAT stage
 * statically uses, in declaration order — so a consumer that declares-but-does-not-
 * use a varying, or declares its inputs in a different order, would read every
 * following member at a shifted offset. The fix is one canonical record layout per
 * interface, derived from the UNION of the stages' members keyed by GL varying
 * location (never by name or by what SPIRV-Cross happens to keep):
 *
 *   - user varyings      key = Location decoration (from the program's reflected
 *                        SPIR-V resources of that stage)
 *   - builtins           key = fixed slots (gl_Position, gl_PointSize, ...) so every
 *                        stage agrees regardless of declaration order
 *   - unreflectable      key = stable hash of the member name (interface blocks whose
 *                        members SPIRV-Cross flattens but whose locations the resource
 *                        list does not carry)
 *
 * Every stage's MSL struct is then rewritten to the canonical member sequence (the
 * stage's own member names for members it has, never-referenced pads otherwise), and
 * the record the stages exchange is byte-identical for producer and consumer.
 */

/* fixed builtin slots: after user locations, before name hashes, in a stable order */
static uint32_t tess_builtin_key(const char *name)
{
    static const struct { const char *name; uint32_t key; } tab[] = {
        { "gl_Position",     0x80010000u },
        { "gl_PointSize",    0x80010001u },
        { "gl_ClipDistance", 0x80010002u },
        { "gl_CullDistance", 0x80010003u },
        { "gl_TessLevel",      0x80020000u },
        { "gl_TessLevelOuter", 0x80020000u },   /* packed half4 alias of the same slot */
        { "gl_TessLevelInner", 0x80020001u },
        { NULL, 0 }
    };
    for (int i = 0; tab[i].name; i++)
        if (!strcmp(name, tab[i].name))
            return tab[i].key;
    return 0;
}

static uint32_t tess_name_hash(const char *s)
{
    uint32_t h = 2166136261u;
    while (*s)
    {
        h ^= (uint8_t)*s++;
        h *= 16777619u;
    }
    return 0x40000000u | (h & 0x3fffffffu);
}

/* attach the canonical key to every member of a parsed stage struct, using the
 * stage's reflected SPIR-V interface (name -> location). Location 0 is valid, so
 * "not found" is tracked separately from the key value. Arrayed varyings are
 * flattened by SPIRV-Cross into "name_0", "name_1" members: they resolve to the
 * base varying's location with their element index recorded separately */
/* does the stage's GL-view module declare a variable or interface-block
 * member of this name?  (The reflection lists carry the block INSTANCE name
 * only, so members are resolved through OpMemberName/OpName.  Block members
 * carry no Location decorations in the module, so the canonical key for them
 * is a stable hash of the resolved name — identical on every side of a
 * linkable program.) */
static bool tess_stage_var_named(GLMContext ctx, Program *prog, int stage,
                                  const char *name)
{
    uint32_t *ir;
    size_t n;

    if (!ctx || !prog || !name || stage < 0 || stage >= _MAX_SHADER_TYPES)
        return false;
    ir = prog->spirv[stage].ir;
    n = prog->spirv[stage].size;
    if (!ir || n < 8)
        return false;

    /* 1) find the name: OpMemberName -> (struct type, member), OpName -> id */
    {
        size_t m = 5;
        uint32_t found_type = 0, found_id = 0;

        while (m + 3 < n)
        {
            uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
            if (wc == 0) break;
            if (op == 5 /* OpName */ || op == 6 /* OpMemberName */)
            {
                char buf[128];
                size_t o = 0;
                const uint32_t *w = ir + m + (op == 5 ? 2 : 3);
                const uint32_t *we = ir + m + wc;

                for (; w < we && o + 4 < sizeof buf; w++)
                    for (int b = 0; b < 4; b++)
                    {
                        char c = (char)((*w >> (8 * b)) & 255);
                        if (!c) { w = we; break; }
                        buf[o++] = c;
                    }
                buf[o] = 0;
                if (!strcmp(buf, name))
                {
                    if (op == 6)
                    {
                        found_type = ir[m + 1];
                    }
                    else
                        found_id = ir[m + 1];
                    break;
                }
            }
            m += wc;
        }
        if (!found_type && !found_id)
            return false;
        if (found_type)
        {
            /* an OpMemberName hit only names a VARYING when its struct is an
             * interface BLOCK: a plain value struct's field names (e.g. S.x)
             * are not varyings and must not resolve a flattened base */
            size_t m2 = 5;
            bool block = false;

            while (m2 + 3 < n)
            {
                uint32_t wc2 = ir[m2] >> 16, op2 = ir[m2] & 0xffff;

                if (wc2 == 0) break;
                if (op2 == 71 /* OpDecorate */ && ir[m2 + 1] == found_type &&
                    ir[m2 + 2] == 2 /* Block */)
                {
                    block = true;
                    break;
                }
                m2 += wc2;
            }
            if (!block)
                return false;
        }
    }
    return true;
}

/* Instance names of the stage module's interface blocks (patch or per-vertex,
 * input or output).  SPIRV-Cross flattens a block into the stage struct with
 * member names "<instance>_<member>..."; the instance prefix is what
 * distinguishes WHICH block a flattened member belongs to, so keying by the
 * block-relative name needs the prefix list (tess_annotate_keys). */
static int tess_block_instance_names(GLMContext ctx, Program *prog, int stage,
                                     char out[][128], int max)
{
    uint32_t *ir;
    size_t n;
    int cnt = 0;

    if (!ctx || !prog || !out || max <= 0 || stage < 0 || stage >= _MAX_SHADER_TYPES)
        return 0;
    ir = prog->spirv[stage].ir;
    n = prog->spirv[stage].size;
    if (!ir || n < 8)
        return 0;
    {
        uint32_t blocks[32];
        int n_blocks = 0;
        size_t m = 5;

        while (m + 3 < n && n_blocks < 32)
        {
            uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
            if (wc == 0) break;
            if (op == 71 /* OpDecorate */ && wc >= 3 && ir[m + 2] == 2 /* Block */)
                blocks[n_blocks++] = ir[m + 1];
            m += wc;
        }
        if (!n_blocks)
            return 0;

        /* 2) pointer types onto a block (directly or through an array — a
         * per-vertex block is "array<TheBlock>") */
        uint32_t ptrs[64];
        int n_ptrs = 0;
        m = 5;
        while (m + 3 < n && n_ptrs < 64)
        {
            uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;

            if (wc == 0) break;
            if (op == 32 /* OpTypePointer */ && wc == 4)
            {
                uint32_t pointee = ir[m + 3];
                bool is_block = false;
                size_t k = 5;

                while (k + 3 < n)
                {
                    uint32_t w2 = ir[k] >> 16, o2 = ir[k] & 0xffff;
                    if (w2 == 0) break;
                    /* only a type DEFINITION has ir[k+1] as a result id —
                     * OpName/OpDecorate reference the array in an operand slot
                     * and would end the strip early */
                    if (o2 >= 19 && o2 <= 33 && ir[k + 1] == pointee)
                    {
                        if (o2 == 28 /* OpTypeArray */ && w2 >= 4)
                            pointee = ir[k + 2];
                        break;
                    }
                    k += w2;
                }
                    for (int b = 0; b < n_blocks; b++)
                        if (pointee == blocks[b]) { is_block = true; break; }
                if (tess_trace() && getenv("MGL_TESS_DEBUG3"))
                    fprintf(stderr, "   ptr m=%zu pointee=%u resolved=%u is_block=%d (blocks:",
                            m, ir[m + 3], pointee, is_block);
                if (tess_trace() && getenv("MGL_TESS_DEBUG3"))
                {
                    for (int b = 0; b < n_blocks; b++) fprintf(stderr, " %u", blocks[b]);
                    fprintf(stderr, ")\n");
                }
                if (is_block)
                    ptrs[n_ptrs++] = ir[m + 1];
            }
            m += wc;
        }
        if (!n_ptrs)
            return 0;

        /* 3) interface variables of those pointer types: OpName of the variable */
        m = 5;
        while (m + 3 < n && cnt < max)
        {
            uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
            if (wc == 0) break;
            if (op == 59 /* OpVariable */ && wc >= 4)
            {
                uint32_t rtype = ir[m + 1], vid = ir[m + 2];
                bool hit = false;

                for (int p = 0; p < n_ptrs; p++)
                    if (ptrs[p] == rtype) { hit = true; break; }
                if (hit)
                {
                    /* the variable's OpName */
                    size_t k = 5;

                    while (k + 3 < n)
                    {
                        uint32_t w2 = ir[k] >> 16, o2 = ir[k] & 0xffff;
                        if (w2 == 0) break;
                        if (o2 == 5 /* OpName */ && ir[k + 1] == vid && w2 >= 3)
                        {
                            char buf[128];
                            size_t o = 0;
                            const uint32_t *w = ir + k + 2;
                            const uint32_t *we = ir + k + w2;

                            for (; w < we && o + 4 < sizeof buf; w++)
                                for (int b = 0; b < 4; b++)
                                {
                                    char c = (char)((*w >> (8 * b)) & 255);
                                    if (!c) { w = we; break; }
                                    buf[o++] = c;
                                }
                            buf[o] = 0;
                            if (o)
                            {
                                bool dup = false;

                                for (int q = 0; q < cnt; q++)
                                    if (!strcmp(out[q], buf)) { dup = true; break; }
                                if (!dup)
                                    snprintf(out[cnt++], 128, "%s", buf);
                            }
                            break;
                        }
                        k += w2;
                    }
                }
            }
            m += wc;
        }
    }
    return cnt;
}

/* Parse a dissolved patch-block-ARRAY leaf name "mgl_ds_<member>_<e>_<path>"
 * (<path>: "" | "<k>" | "<field>" | "<i>_<field>" | "<i>_<field>_<j>") into the
 * block element (array_idx), the struct-array/plain-field element (array_idx2),
 * the innermost element (array_idx3) and the index-free path the slot is keyed
 * by ("blockSa_z").  Returns 0 = not a ds leaf. */
static int tess_ds_leaf_key(const char *name, uint32_t *idx, uint32_t *idx2,
                            uint32_t *idx3, uint32_t *dims, char *base, size_t base_sz)
{
    char rest[128], field[80] = "";
    char *r, *u;
    const char *mbr = name + 7;
    char *sep;
    uint32_t e = 0, i2 = 0, j3 = 0;

    if (strncmp(name, "mgl_ds_", 7) || !name[7])
        return 0;
    sep = strchr(mbr, '_');
    snprintf(rest, sizeof rest, "%s", sep ? sep + 1 : "");
    if (sep && (size_t)(sep - mbr) >= base_sz)
        return 0;

    r = rest;
    while (*r == ' ') r++;
    if (*r >= '0' && *r <= '9')
    {
        e = (uint32_t)strtoul(r, &u, 10);
        if (*u == '_') r = u + 1;
        else r = u;                            /* a bare element index: the
                                                 * FIRST (and only) dimension */
    }
    if (*r >= '0' && *r <= '9')
    {
        uint32_t v = (uint32_t)strtoul(r, &u, 10);

        if (*u == '_' && u[1])
        {
            i2 = v;                          /* "<i>_<field>[_<j>]" */
            r = u + 1;
        }
        else
        {
            i2 = v;                          /* plain field element "<k>" */
            r = u;
        }
    }
    {
        size_t fl = 0;

        while (r[fl] && (isalnum((unsigned char)r[fl]) || r[fl] == '_')) fl++;
        if (fl && fl < sizeof field) snprintf(field, sizeof field, "%s", r);
    }
    {
        char *lb = field[0] ? strrchr(field, '_') : NULL;

        if (lb && lb[1] >= '0' && lb[1] <= '9' && lb != field)
        {
            char *ue;

            j3 = (uint32_t)strtoul(lb + 1, &ue, 10);
            if (*ue) j3 = 0;
            else *lb = 0;
        }
    }
    snprintf(base, base_sz, "%.*s%s%s", (int)(sep ? (size_t)(sep - mbr) : strlen(mbr)),
             mbr, field[0] ? "_" : "", field);
    *idx = e;
    *idx2 = i2;
    *idx3 = j3;
    *dims = (j3 ? 3u : 0u) | (i2 ? 2u : 0u) | (e ? 1u : 0u);
    return 1;
}

static void tess_annotate_keys(GLMContext ctx, Program *names_prog, int stage,
                               struct MslStruct *s, const SpirvResourceList *rl)
{
    char prefixes[8][128];
    /* the producer's block instance names (for whole-block members) plus the
     * CONSUMER's (the TES's dissolved variables carry the consumer's
     * instance name: "teBlock_mgl_ds_...").  GL matches blocks by the block
     * TYPE, so producer and consumer instance names legitimately differ
     * (dEQP: tcBlock / teBlock). */
    int n_prefixes = tess_block_instance_names(ctx, names_prog, stage, prefixes, 8);

    if (stage != _TESS_EVALUATION_SHADER && names_prog->spirv[_TESS_EVALUATION_SHADER].ir)
    {
        char extra[8][128];
        int n_extra = tess_block_instance_names(ctx, names_prog, _TESS_EVALUATION_SHADER,
                                                extra, 8);

        for (int q = 0; q < n_extra && n_prefixes < 8; q++)
        {
            int dup = 0;

            for (int w = 0; w < n_prefixes; w++)
                if (!strcmp(prefixes[w], extra[q]))
                {
                    dup = 1;
                    break;
                }
            if (!dup)
                snprintf(prefixes[n_prefixes++], 128, "%s", extra[q]);
        }
    }

    if (tess_trace() && getenv("MGL_TESS_DEBUG2"))
        fprintf(stderr, "MGL tess: stage %d block instance prefixes (%d): %s%s%s%s\n", stage,
                n_prefixes, n_prefixes > 0 ? prefixes[0] : "-", n_prefixes > 1 ? " " : "",
                n_prefixes > 1 ? prefixes[1] : "", n_prefixes > 2 ? " ..." : "");

    for (int i = 0; i < s->count; i++)
    {
        MGLTessPatchMember *m = &s->members[i];
        uint32_t key = 0;
        bool found = false;
        char base[128];                 /* the bare GL name of the varying */

        if (strstr(m->name, "mgl_ds_") && getenv("MGL_TESS_DEBUG2"))
            fprintf(stderr, "MGL tess annotate[key]: member '%s' (prefixes %d: %s)\n",
                    m->name, n_prefixes, n_prefixes ? prefixes[0] : "-");

        /* builtins first: gl_Position & co must key to their fixed slots on every
         * side of the record.  The generic path below would otherwise name-hash
         * them whenever the stage module declares the variable at top level
         * (the vertex stage's gl_Position output), splitting the record into a
         * producer-written and a consumer-read copy that never alias. */
        key = tess_builtin_key(m->name);
        if (key) { m->key = key; continue; }

        /* a FLATTENED builtin array member ("gl_CullDistance_3"): key it to the
         * builtin slot with the suffix as the element index, so the consumer's
         * flattened struct aliases the producer's whole-array member at the same
         * canonical offsets (the generic path below resolves names through the
         * stage's reflected variables, which never list the builtins) */
        {
            char b2[128];
            size_t n = strlen(m->name);
            while (n > 1 && m->name[n - 1] >= '0' && m->name[n - 1] <= '9') n--;
            if (n > 1 && m->name[n - 1] == '_')
            {
                snprintf(b2, sizeof b2, "%.*s", (int)(n - 1), m->name);
                key = tess_builtin_key(b2);
                if (key)
                {
                    m->key = key;
                    m->array_idx = (uint32_t)atoi(m->name + n);
                    m->flat_dims = 1;
                    continue;
                }
            }
        }

        base[0] = 0;
        /* a dissolved patch-block-ARRAY leaf carries the marker FIRST (it is not
         * prefixed by any block instance): key it by the index-free path now —
         * the prefix scan below would never fire for it */
        if (!strncmp(m->name, "mgl_ds_", 7) && m->name[7])
        {
            char ds_base_l[128];

            if (tess_ds_leaf_key(m->name, &m->array_idx, &m->array_idx2,
                                 &m->array_idx3, &m->flat_dims,
                                 ds_base_l, sizeof ds_base_l))
            {
                m->key = tess_name_hash(ds_base_l);
                if (getenv("MGL_TESS_DEBUG2"))
                    fprintf(stderr, "MGL tess annotate[ds0]: '%s' -> base '%s' idx=%u idx2=%u idx3=%u dims=%u\n",
                            m->name, ds_base_l, m->array_idx, m->array_idx2,
                            m->array_idx3, m->flat_dims);
                continue;
            }
        }
        /* a DISSOLVED member variable (mgl_tess_dissolve_vertex_input_blocks)
         * reflects under its own name with a SYNTHETIC location: never key it
         * by that — the producer keeps the whole block and keys by the hashed
         * block-relative path, so both sides must agree (the ds branch below) */
        if (rl && !strstr(m->name, "mgl_ds_"))
            for (GLuint j = 0; j < rl->count; j++)
                if (rl->list && !strcmp(rl->list[j].name, m->name))
                {
                    key = (uint32_t)rl->list[j].location;
                    found = true;
                    break;
                }
        if (!found)
        {
            /* interface-block member reached with the INSTANCE prefix
             * ("<instance>_<member>..."): strip the whole instance name and key
             * by the block-relative path.  The generic split below would key by
             * a struct FIELD name ("x"), which every block member containing
             * that field hashes identically to (blockS.x and blockSa[0].x
             * collided) — the record then dropped one of them.  Two flattened
             * dimensions are recognised along the way: "<m>_<i>_<field>" (i =
             * the struct-array element) and a trailing "_<j>" (an array-field
             * element). */
            for (int q = 0; q < n_prefixes && !found; q++)
            {
                size_t pl = strlen(prefixes[q]);

                if (!strncmp(m->name, prefixes[q], pl) && m->name[pl] == '_' && m->name[pl + 1])
                {
                    char rest[128], b2[128];
                    uint32_t fidx = m->array_idx, fidx2 = 0, fidx3 = 0;
                    bool mid = false, mid2 = false, ds = false;
                    char ds_mbr[80];
                    size_t rn;

                    snprintf(rest, sizeof rest, "%s", m->name + pl + 1);

                    /* a DISSOLVED member variable of this block (created by
                     * mgl_tess_dissolve_vertex_input_blocks in the module
                     * handed to SPIRV-Cross; the GL-view module keeps the
                     * block, so detect it by the "mgl_ds_<member>" marker):
                     * the leaf path is "<member>_<leaf...>" with a struct-array
                     * element index FIRST and an array-field element LAST */
                    if (!strncmp(rest, "mgl_ds_", 7) && rest[7])
                    {
                        char *sep = strchr(rest + 7, '_');

                        if (sep && (size_t)(sep - (rest + 7)) < sizeof ds_mbr)
                        {
                            memcpy(ds_mbr, rest + 7, sep - (rest + 7));
                            ds_mbr[sep - (rest + 7)] = 0;
                            snprintf(rest, sizeof rest, "%s", sep + 1);
                            ds = true;
                        }
                        else if (!sep && strlen(rest + 7) < sizeof ds_mbr)
                        {
                            /* a leaf member without further path: the whole
                             * remainder IS the member name */
                            snprintf(ds_mbr, sizeof ds_mbr, "%s", rest + 7);
                            rest[0] = 0;
                            ds = true;
                        }
                    }

                    /* a dissolved patch-block-ARRAY leaf: the name is
                     * "mgl_ds_<member>_<e>_<path>" with <path> one of
                     * "" | "<k>" | "<field>" | "<i>_<field>" | "<i>_<field>_<j>"
                     * (the element index e selects the block-array element,
                     * i the struct-array element, j/k the innermost array
                     * element).  Keyed by the path WITHOUT indices so the
                     * producer's field members ("mgl_ds_<member>_<field>",
                     * typed array[e][i][j]) hash identically. */
                    if (ds)
                    {
                        char ds_base_l[128];

                        if (tess_ds_leaf_key(m->name, &m->array_idx, &m->array_idx2,
                                             &m->array_idx3, &m->flat_dims,
                                             ds_base_l, sizeof ds_base_l))
                        {
                            key = tess_name_hash(ds_base_l);
                            if (getenv("MGL_TESS_DEBUG2"))
                                fprintf(stderr, "MGL tess annotate[ds]: '%s' -> base '%s' idx=%u idx2=%u idx3=%u dims=%u\n",
                                        m->name, ds_base_l, m->array_idx, m->array_idx2,
                                        m->array_idx3, m->flat_dims);
                            found = true;
                            break;
                        }
                    }
                    if (ds)
                    {
                        size_t dn = 0;

                        while (rest[dn] >= '0' && rest[dn] <= '9') dn++;
                        if (dn && (rest[dn] == '_' || !rest[dn]))
                        {
                            fidx = (uint32_t)atoi(rest);
                            mid = true;
                            if (rest[dn])
                                memmove(rest, rest + dn + 1, strlen(rest + dn + 1) + 1);
                            else
                                rest[0] = 0;
                        }
                    }
                    else for (char *b = rest; (b = strchr(b, '_')) != NULL; b++)
                    {
                        char *d = b + 1;
                        size_t dn = 0, hn, tn;

                        while (d[dn] >= '0' && d[dn] <= '9') dn++;
                        if (!dn || d[dn] != '_') continue;
                        hn = (size_t)(b - rest);
                        if (hn >= sizeof b2) break;
                        memcpy(b2, rest, hn);
                        b2[hn] = 0;
                        if (!tess_stage_var_named(ctx, names_prog, stage, b2))
                        {
                            if (getenv("MGL_TESS_DEBUG2"))
                                fprintf(stderr, "MGL tess annotate[mid]: '%s' head '%s' not a stage var\n", m->name, b2);
                            continue;
                        }
                        tn = strlen(d + dn + 1);
                        if (hn + 1 + tn + 1 > sizeof rest) break;
                        fidx = (uint32_t)atoi(d);     /* d points into rest: read before the rebase */
                        mid = true;
                        {
                            char tail[96];

                            snprintf(tail, sizeof tail, "%s", d + dn + 1);
                            snprintf(rest, sizeof rest, "%s_%s", b2, tail);
                        }
                        break;
                    }

                    /* a dissolved patch-block ARRAY element adds a SECOND mid
                     * index ("blockSa_<e>_<i>_<field>_<j>"): array_idx keeps
                     * the block element (e), array_idx2 takes the struct-array
                     * element (i) and the trailing index (j) moves to
                     * array_idx3 — the canonical member's type is three
                     * dimensions deep ("float[e][i][j]") */
                    if (mid && !ds)
                    for (char *b = rest; (b = strchr(b, '_')) != NULL; b++)
                    {
                        char *d = b + 1;
                        size_t dn = 0, hn, tn;

                        while (d[dn] >= '0' && d[dn] <= '9') dn++;
                        if (!dn || d[dn] != '_') continue;
                        hn = (size_t)(b - rest);
                        if (hn >= sizeof b2) break;
                        memcpy(b2, rest, hn);
                        b2[hn] = 0;
                        if (!tess_stage_var_named(ctx, names_prog, stage, b2))
                            continue;
                        tn = strlen(d + dn + 1);
                        if (hn + 1 + tn + 1 > sizeof rest) break;
                        fidx3 = (uint32_t)atoi(d);
                        mid2 = true;
                        {
                            char tail[96];

                            snprintf(tail, sizeof tail, "%s", d + dn + 1);
                            snprintf(rest, sizeof rest, "%s_%s", b2, tail);
                        }
                        break;
                    }

                    /* a trailing "_<digits>" group: the array-FIELD element —
                     * the second canonical dimension when a mid index was found
                     * above, the plain array element otherwise */
                    rn = strlen(rest);
                    {
                        size_t dn = 0;

                        while (dn < rn && rest[rn - 1 - dn] >= '0' && rest[rn - 1 - dn] <= '9') dn++;
                        if (dn && rn - dn > 1 && rest[rn - 1 - dn] == '_')
                        {
                            if (mid)
                                fidx2 = (uint32_t)atoi(rest + rn - dn);
                            else
                                fidx = (uint32_t)atoi(rest + rn - dn);
                            rest[rn - 1 - dn] = 0;
                        }
                    }

                    /* the block-relative path keys the slot: a reflected
                     * location when the member is one (rare for blocks), the
                     * stable name hash otherwise; a dissolved member's leaf
                     * keys by "<member>_<leaf path>" — the same string the
                     * producer's whole-block member derives */
                    bool matched = false;
                    char ds_base[160];

                    if (ds)
                    {
                        if (rest[0])
                            snprintf(ds_base, sizeof ds_base, "%s_%s", ds_mbr, rest);
                        else
                            snprintf(ds_base, sizeof ds_base, "%s", ds_mbr);
                        if (getenv("MGL_TESS_DEBUG2"))
                            fprintf(stderr, "MGL tess annotate[ds]: '%s' -> mbr '%s' rest '%s' mid=%d fidx=%u fidx2=%u base '%s'\n",
                                    m->name, ds_mbr, rest, (int)mid, fidx, fidx2, ds_base);
                    }
                    else
                        snprintf(ds_base, sizeof ds_base, "%s", rest);

                    if (!ds && rl)
                        for (GLuint j = 0; j < rl->count; j++)
                        if (rl->list)
                        {
                            const char *nm = rl->list[j].name;
                            size_t nl = nm ? strlen(nm) : 0, bl = strlen(ds_base);

                            if (nm && (!strcmp(nm, ds_base) ||
                                (nl == bl + 3 && !strncmp(nm, ds_base, bl) && !strcmp(nm + bl, "[0]"))))
                            {
                                key = (uint32_t)rl->list[j].location;
                                matched = true;
                                break;
                            }
                        }
                    if (!matched) key = tess_name_hash(ds_base);
                    m->array_idx = fidx;
                    m->array_idx2 = mid2 ? fidx3 : fidx2;   /* second dim: struct-array element */
                    m->array_idx3 = mid2 ? fidx2 : 0;       /* third dim: the innermost element */
                    m->flat_dims = (m->array_idx3 ? 3u : 0u) |
                                   (fidx2 ? 2u : (fidx ? 1u : 0u));
                    if (getenv("MGL_TESS_DEBUG2"))
                        fprintf(stderr, "MGL tess annotate[blk]: '%s' -> base '%s' idx=%u idx2=%u idx3=%u dims=%u\n",
                                m->name, ds_base, fidx, m->array_idx2, m->array_idx3, m->flat_dims);
                    found = true;
                }
            }
        }
        if (!found && !strstr(m->name, "mgl_ds_"))
        {
            /* flattened array element ("<name>_<q>") and/or interface-block
             * member reached with the instance prefix ("<instance>_<member>",
             * anonymous blocks as "m_<id>_<member>").  GL varying names
             * commonly end in digits themselves ("variable0"), so both the
             * plain and the element-stripped reading are tried, validated
             * against the stage module's declared names — the same varying
             * then annotates to the same key on every side of the record and
             * the array/flattened merge in tess_canon_union can fire */
            uint32_t fidx = m->array_idx;

            base[0] = 0;
            for (const char *b = m->name; !base[0] && (b = strchr(b, '_')); b++)
            {
                const char *suf = b + 1;
                size_t n = strlen(suf);
                char b2[128];

                if (tess_stage_var_named(ctx, names_prog, stage, suf))
                {
                    snprintf(base, sizeof base, "%s", suf);
                    break;
                }
                while (n > 1 && suf[n - 1] >= '0' && suf[n - 1] <= '9')
                    n--;
                if (n > 1 && suf[n - 1] == '_')
                {
                    snprintf(b2, sizeof b2, "%.*s", (int)(n - 1), suf);
                    if (tess_stage_var_named(ctx, names_prog, stage, b2))
                    {
                        snprintf(base, sizeof base, "%s", b2);
                        fidx = (uint32_t)atoi(suf + n);
                    }
                }
            }
            if (tess_trace() && getenv("MGL_TESS_DEBUG2"))
                fprintf(stderr, "MGL tess annotate[gen]: stage %d '%s' base='%s' fidx=%u\n",
                        stage, m->name, base, fidx);
            if (!base[0])
            {
                char b2[128];
                size_t n = strlen(m->name);

                if (tess_stage_var_named(ctx, names_prog, stage, m->name))
                    snprintf(base, sizeof base, "%s", m->name);
                else
                {
                    while (n > 1 && m->name[n - 1] >= '0' && m->name[n - 1] <= '9')
                        n--;
                    if (n > 1 && m->name[n - 1] == '_')
                    {
                        snprintf(b2, sizeof b2, "%.*s", (int)(n - 1), m->name);
                        if (tess_stage_var_named(ctx, names_prog, stage, b2))
                        {
                            snprintf(base, sizeof base, "%s", b2);
                            fidx = (uint32_t)atoi(m->name + n);
                        }
                    }
                }
            }
            if (base[0])
            {
                /* the split base may itself be a reflected varying: key by its
                 * location so the producer's whole-array member (keyed by the
                 * same location) aliases these flattened elements */
                bool matched = false;
                if (rl)
                    for (GLuint j = 0; j < rl->count; j++)
                    if (rl->list)
                    {
                        const char *n = rl->list[j].name;
                        size_t nl = n ? strlen(n) : 0, bl = strlen(base);
                        if (!n || !strcmp(n, base) ||
                            (nl == bl + 3 && !strncmp(n, base, bl) && !strcmp(n + bl, "[0]")))
                        {
                            key = (uint32_t)rl->list[j].location;   /* 0 is a valid location */
                            matched = true;
                            break;
                        }
                    }
                if (!matched) key = tess_name_hash(base);
                m->array_idx = fidx;
                m->flat_dims = fidx ? 1u : 0u;
                found = true;
            }
        }
        if (!found)
            key = tess_builtin_key(m->name);
        if (!found && !key)
        {
            /* a dissolved variable no block prefix claimed: key by the
             * block-relative path after the marker, mirroring the ds branch
             * of the prefix path so the producer pairs with it */
            const char *ds = strstr(m->name, "mgl_ds_");
            const char *n = ds ? ds + 7 : m->name;
            size_t len = strlen(n);
            char b2[128];

            /* an element ("<slot>_<q>") of an ARRAY slot that itself could not
             * be resolved to a declared variable (a STRUCT FIELD split, e.g.
             * S.z[2] naming "vb_s_z_0" — or the same leaf under a dissolved
             * variable, "vb_mgl_ds_s_z_0"): key by the stripped slot name so
             * it pairs with the producer's whole "vb_s_z" slot, and record
             * the element index for the stride fixup */
            while (len > 1 && n[len - 1] >= '0' && n[len - 1] <= '9')
                len--;
            if (len > 1 && n[len - 1] == '_' && len - 1 < sizeof b2)
            {
                snprintf(b2, sizeof b2, "%.*s", (int)(len - 1), n);
                key = tess_name_hash(b2);
                m->array_idx = (uint32_t)atoi(n + len);
                m->flat_dims = 1;
            }
            else
                key = tess_name_hash(n);   /* unreflected (block member / gl_ extra) */
        }
        m->key = key;
    }
}

/* Interface blocks reach the MSL stage structs as flattened members whose names * SPIRV-Cross prefixes differently on the two sides of an interface
 * ("<instance>_<member>" on the consumer, "m_<id>_<member>" for an anonymous
 * block on the producer), so neither the reflected names nor the name hash
 * line up: the producer's and the consumer's copy of the same GL member
 * become two different record slots and the value never crosses the stage.
 * Pair such leftover members positionally — SPIRV-Cross emits block members in
 * declaration order and both stages declare the same block — requiring the
 * same MSL type, the same array element, and a common suffix that starts at a
 * '_' boundary in both names.  GL requires matching blocks across stages, so
 * the order (and types) agree. */
static void tess_pair_iface_members(struct MslStruct *a, struct MslStruct *b)
{
    static uint32_t next_pair_key = 0x60000000u;
    char used[MGL_TESS_MAX_PATCH_MEMBERS];

    if (!a || !b || a == b) return;
    memset(used, 0, sizeof used);
    for (int i = 0; i < a->count; i++)
    {
        MGLTessPatchMember *ma = &a->members[i];
        size_t na;

        if ((ma->key & 0xc0000000u) != 0x40000000u) continue;   /* only unreflected members */
        na = strlen(ma->name);
        for (int j = 0; j < b->count; j++)
        {
            MGLTessPatchMember *mb = &b->members[j];
            size_t nb, k;

            if (used[j] || (mb->key & 0xc0000000u) != 0x40000000u) continue;
            if (strcmp(ma->type, mb->type) != 0) continue;
            if (ma->array_idx != mb->array_idx) continue;
            nb = strlen(mb->name);
            k = 0;
            while (k < na && k < nb && ma->name[na - 1 - k] == mb->name[nb - 1 - k]) k++;
            if (!k) continue;
            /* the shared suffix must be a whole component of both names: it
             * consumes a name entirely, starts at a '_' separator, or the
             * character before it is the separator */
            if (!((k == na || ma->name[na - k] == '_' || ma->name[na - k - 1] == '_') &&
                  (k == nb || mb->name[nb - k] == '_' || mb->name[nb - k - 1] == '_'))) continue;
            ma->key = mb->key = next_pair_key++;
            used[j] = 1;
            if (tess_trace())
                fprintf(stderr, "MGL tess: paired block member %s <-> %s\n", ma->name, mb->name);
            break;
        }
    }
}

struct CanonRec {
    MGLTessPatchMember m[MGL_TESS_MAX_PATCH_MEMBERS];
    int    n;
    size_t size;        /* record stride, tail-padded to the struct alignment */
};

/* canonical record = union of the stages' members by key, sorted by key; each key's
 * slot takes the largest member any stage declares there (the producer's data must
 * fit even if a consumer declares a narrower type it never reads). A member whose
 * type is an arrayed varying covers the same slots as another stage's flattened
 * "name_i" elements of that varying: the two merge into the wider (array) entry */
static bool tess_canon_union(struct CanonRec *out, struct MslStruct *const *in, int nin)
{
    size_t dummy_align = 0, dummy_size = 0;

    memset(out, 0, sizeof *out);
    for (int s = 0; s < nin; s++)
    {
        if (!in[s])
            continue;
        for (int i = 0; i < in[s]->count; i++)
        {
            MGLTessPatchMember *m = &in[s]->members[i];
            bool m_arr = strstr(m->type, "spvUnsafeArray") != NULL;
            int found = -1;
            bool absorbed = false;

            if (m->size == 0)
                continue;               /* patch_control_point reference and friends */
            for (int j = 0; j < out->n; j++)
            {
                bool o_arr = strstr(out->m[j].type, "spvUnsafeArray") != NULL;

                if (out->m[j].key != m->key)
                    continue;
                /* a dissolved patch-block-ARRAY leaf: its indices select a
                 * position INSIDE the array slot (stride math at bind time),
                 * never a slot (or a merge) of its own */
                if (o_arr && !m_arr && !strncmp(m->name, "mgl_ds_", 7))
                {
                    absorbed = true;
                    break;
                }
                /* two ARRAY members of the same key but DIFFERENT element
                 * index ("pb_ps_0_z" / "pb_ps_1_z": the z field of block
                 * element 0 and 1 of a dissolved patch block array) are two
                 * SLOTS, not one varying — merging them drops element 1 and
                 * its reads land inside element 0's array */
                if (o_arr && m_arr && out->m[j].array_idx != m->array_idx)
                    continue;
                if (o_arr || m_arr ||
                    (!o_arr && !m_arr && out->m[j].array_idx == m->array_idx &&
                     out->m[j].array_idx2 == m->array_idx2))
                {
                    found = j;          /* same varying (whole array or an element) */
                    if (!o_arr && m_arr)
                    {
                        /* the array covers the elements already registered —
                         * including the other flattened elements of the same
                         * varying registered as their own slots */
                        out->m[j] = *m;
                        out->m[j].offset = 0;
                        for (int q = 0; q < out->n; q++)
                        {
                            if (q == j || out->m[q].key != m->key) continue;
                            if (strstr(out->m[q].type, "spvUnsafeArray") != NULL) continue;
                            memmove(&out->m[q], &out->m[q + 1],
                                    (out->n - q - 1) * sizeof(out->m[0]));
                            out->n--;
                            if (q < j) j--;
                            q--;
                        }
                    }
                    break;
                }
                /* a flattened element of an array slot already present: any
                 * element, including the [0][0] of a two-dimensional slot
                 * (flat_dims marks the member as flattened even then) */
                if (o_arr && !m_arr && (m->flat_dims > 0 ||
                                        !strncmp(m->name, "mgl_ds_", 7)))
                    absorbed = true;   /* a dissolved patch-block-ARRAY leaf:
                                        * its indices select a position INSIDE
                                        * the array slot (stride math at bind
                                        * time), never a slot of its own */
                if (!o_arr && !m_arr && m->array_idx != 0 &&
                    msl_matrix_type(out->m[j].type, &dummy_align, &dummy_size))
                    absorbed = true;   /* flattened MATRIX COLUMN of a matrix slot already
                                         * present (the TES side sees one [[attribute]] per
                                         * column; the producer's whole matrix covers it) */
            }
            if (absorbed)
                continue;
            if (found < 0)
            {
                if (out->n >= MGL_TESS_MAX_PATCH_MEMBERS)
                    return false;
                out->m[out->n] = *m;
                out->m[out->n].offset = 0;
                out->n++;
            }
            else if (m->size > out->m[found].size)
            {
                /* keep the wider type, remember the member that provides it */
                int whole_matrix = msl_matrix_type(m->type, &dummy_align, &dummy_size) != 0;

                snprintf(out->m[found].type, sizeof out->m[found].type, "%s", m->type);
                snprintf(out->m[found].name, sizeof out->m[found].name, "%s", m->name);
                out->m[found].size = m->size;
                out->m[found].array_idx = m->array_idx;
                out->m[found].array_idx2 = m->array_idx2;
                out->m[found].flat_dims = m->flat_dims;
                if (whole_matrix)
                {
                    /* a WHOLE matrix replaces its column slots: drop the other
                     * flattened columns of the same member */
                    for (int d = 0; d < out->n; d++)
                        if (d != found && out->m[d].key == m->key)
                        {
                            memmove(&out->m[d], &out->m[d + 1],
                                    (out->n - d - 1) * sizeof out->m[d]);
                            out->n--;
                            if (d < found)
                                found--;
                            d--;
                        }
                }
            }
        }
    }

    /* sort by key (insertion sort: n is bounded by MGL_TESS_MAX_PATCH_MEMBERS) */
    for (int i = 1; i < out->n; i++)
    {
        MGLTessPatchMember tmp = out->m[i];
        int j = i - 1;
        while (j >= 0 && out->m[j].key > tmp.key)
        {
            out->m[j + 1] = out->m[j];
            j--;
        }
        out->m[j + 1] = tmp;
    }

    /* byte layout: C++ natural alignment, struct tail-padded to its alignment */
    size_t cursor = 0, maxalign = 1;
    for (int i = 0; i < out->n; i++)
    {
        size_t align = 1, size = 0;
        if (!msl_element_type(out->m[i].type, &align, &size))
            return false;               /* unknown member type: keep the stage structs */
        out->m[i].offset = round_up(cursor, align);
        cursor = out->m[i].offset + size;
        out->m[i].size = size;
        if (align > maxalign)
            maxalign = align;
    }
    out->size = round_up(cursor, maxalign);
    return true;
}

/* element count of an MSL arrayed varying type ("spvUnsafeArray<T, N>"), 1 otherwise */
static uint32_t msl_array_len(const char *type)
{
    const char *at = strstr(type, "spvUnsafeArray<");

    if (!at)
        return 1;
    at = strrchr(at, ',');
    while (at && *at && (*at < '0' || *at > '9'))
        at++;
    return at ? (uint32_t)strtoul(at, NULL, 10) : 1;
}

/* does the stage's own struct already match the canonical layout? */
static bool tess_canon_matches(const struct MslStruct *own, const struct CanonRec *canon)
{
    if (!own || own->count != canon->n)
        return false;
    for (int i = 0; i < canon->n; i++)
        if (own->members[i].key != canon->m[i].key ||
            strcmp(own->members[i].type, canon->m[i].type))
            return false;
    return true;
}

/* element type of "spvUnsafeArray<T, N>" (text between '<' and the last ','), with the
 * inner template name already renamed when rename_arr; false when type is not an array */
static bool tess_canon_element_type(const char *type, bool rename_arr, char *out, size_t cap)
{
    const char *at = strstr(type, "spvUnsafeArray<");
    const char *end;
    size_t n;

    if (!at)
        return false;
    at += strlen("spvUnsafeArray<");
    end = strrchr(type, ',');
    if (!end || end < at)
        return false;
    n = (size_t)(end - at);
    while (n > 0 && at[n - 1] == ' ')
        n--;
    if (n >= cap)
        return false;
    memcpy(out, at, n);
    out[n] = 0;
    if (rename_arr)
        for (char *p = out; (p = strstr(p, "spvUnsafeArray")) != NULL; )
        {
            memcpy(p, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
            p += strlen("spvUnsafeArray");
        }
    return true;
}

/* copy of a member type with spvUnsafeArray renamed to our own template (the plain-TES
 * MSL declares mgl_spv_arr in the prelude, SPIRV-Cross's template only exists in the
 * kernel forms) */
static void tess_type_renamed(const char *type, bool rename_arr, char *out, size_t cap)
{
    snprintf(out, cap, "%s", type);
    if (!rename_arr)
        return;
    for (char *p = out; (p = strstr(p, "spvUnsafeArray")) != NULL; )
    {
        memcpy(p, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
        p += strlen("spvUnsafeArray");
    }
}

/* one canonical member as struct-body lines for a stage. stage/nstage: the members the
 * stage's own struct declares for this key. A canonical slot that is an ARRAY while the
 * stage flattened it into per-element members ("name_i", array_idx = element index) is
 * re-expanded to those members — they tile the slot exactly, and the stage's code refers
 * to them by their own names and (scalar/vector) types. rename_arr: see above. */
static void str_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...);

static void tess_canon_member_lines(char **buf, size_t *len, size_t *cap,
                                    const MGLTessPatchMember *cm,
                                    const MGLTessPatchMember *stage, int nstage,
                                    bool rename_arr, bool *first)
{
    char type[160];
    bool canon_arr = strstr(cm->type, "spvUnsafeArray") != NULL;
    int nflat = 0;

    tess_type_renamed(cm->type, rename_arr, type, sizeof type);

    if (stage)
        for (int j = 0; j < nstage; j++)
            if (stage[j].key == cm->key && stage[j].size > 0 &&
                strstr(stage[j].type, "spvUnsafeArray") == NULL)
                nflat++;
    if (canon_arr && nflat > 0)
    {
        /* the canonical slot is the whole array, the stage uses flattened elements */
        char elem[120];
        uint32_t n = msl_array_len(cm->type);
        bool two_dim = false;
        char inner_type[120];
        long n2 = 0;

        if (stage)
            for (int j = 0; j < nstage; j++)
                if (stage[j].key == cm->key && stage[j].size > 0 &&
                    strstr(stage[j].type, "spvUnsafeArray") == NULL && stage[j].flat_dims == 2)
                    two_dim = true;

        if (two_dim && !msl_array_inner(cm->type, inner_type, sizeof inner_type, &n2))
            two_dim = false;

        if (two_dim)
        {
            /* a struct-array FIELD slot ("name_<field>" of spvUnsafeArray<T, N>
             * with T itself an array): the stage flattens BOTH dimensions — one
             * element line per (outer, inner) pair */
            char e2[120];

            if (n2 > 0 && tess_canon_element_type(inner_type, rename_arr, e2, sizeof e2))
            {
                for (uint32_t a = 0; a < n; a++)
                    for (long b = 0; b < n2; b++)
                    {
                        const char *nm = NULL;
                        char pad[64];

                        if (stage)
                            for (int j = 0; j < nstage; j++)
                                if (stage[j].key == cm->key && stage[j].size > 0 &&
                                    stage[j].array_idx == a &&
                                    stage[j].array_idx2 == (uint32_t)b &&
                                    strstr(stage[j].type, "spvUnsafeArray") == NULL)
                                {
                                    nm = stage[j].name;
                                    break;
                                }
                        if (!nm)
                        {
                            snprintf(pad, sizeof pad, "mgl_k%08x_%u_%ld", cm->key, a, b);
                            nm = pad;
                        }
                        str_append(buf, len, cap, "%s    %s %s;\n", *first ? "\n" : "", e2, nm);
                        *first = false;
                    }
                return;
            }
        }
        if (tess_canon_element_type(cm->type, rename_arr, elem, sizeof elem))
            for (uint32_t e = 0; e < n; e++)
            {
                const char *nm = NULL;
                char pad[64];

                if (stage)
                    for (int j = 0; j < nstage; j++)
                        if (stage[j].key == cm->key && stage[j].size > 0 &&
                            stage[j].array_idx == e && stage[j].flat_dims != 2 &&
                            strstr(stage[j].type, "spvUnsafeArray") == NULL)
                        {
                            nm = stage[j].name;
                            break;
                        }
                if (!nm)
                {
                    snprintf(pad, sizeof pad, "mgl_k%08x_%u", cm->key, e);
                    nm = pad;
                }
                str_append(buf, len, cap, "%s    %s %s;\n", *first ? "\n" : "", elem, nm);
                *first = false;
            }
        return;
    }

    {
        const char *nm = NULL;
        char pad[48];

        if (stage)
            for (int j = 0; j < nstage; j++)
                if (stage[j].key == cm->key && stage[j].size > 0 &&
                    (cm->array_idx == 0 || stage[j].array_idx == cm->array_idx))
                {
                    nm = stage[j].name;
                    break;
                }
        if (!nm)
        {
            snprintf(pad, sizeof pad, "mgl_k%08x", cm->key);
            nm = pad;
        }
        str_append(buf, len, cap, "%s    %s %s;\n", *first ? "\n" : "", type, nm);
        *first = false;
    }
}

/* canonical struct body text for one stage: the stage's member names for members it
 * declares, syntactically-valid pads for the rest (never referenced by its code) */
/* A scalar-double VS output renders as uint2 in the VS's own MSL but occupies
 * the full uint4 slot in the canonical control-point record (the consumer
 * side's repr, "one uint4 = one 128-bit GL location").  Widen the whole-
 * variable stores `out.<name> = <expr>;` so the canonicalized VS kernel still
 * compiles.  (Reads of such members always go through `.xy` and work either
 * way.) */
static void tess_widen_double_stores(char **msl_p, const struct MslStruct *stage,
                                     const struct CanonRec *canon)
{
    char pat[160];

    for (int i = 0; i < canon->n; i++)
    {
        const MGLTessPatchMember *cm = &canon->m[i];
        const MGLTessPatchMember *sm = NULL;

        for (int j = 0; j < stage->count; j++)
            if (stage->members[j].key == cm->key) { sm = &stage->members[j]; break; }
        if (!sm || !sm->name[0])
            continue;
        if (!strstr(cm->type, "uint4") || !strstr(sm->type, "uint2"))
            continue;
        if (strstr(sm->type, "spvUnsafeArray") || strstr(cm->type, "spvUnsafeArray"))
            continue;   /* per-element stores only happen on the TCS side */

        snprintf(pat, sizeof pat, "out.%s = ", sm->name);
        {
            char *hit = strstr(*msl_p, pat);

            while (hit)
            {
                size_t e0 = (size_t)(hit - *msl_p) + strlen(pat);
                const char *sc = strchr(*msl_p + e0, ';');
                size_t elen, add, olen;
                char *grown;

                if (!sc)
                    break;
                elen = (size_t)(sc - (*msl_p + e0));
                add = strlen("uint4(") + strlen(", 0u, 0u)");
                olen = strlen(*msl_p);
                grown = (char *)realloc(*msl_p, olen + add + 1);
                if (!grown)
                    return;
                *msl_p = grown;
                /* layout: "uint4(" expr ", 0u, 0u)" <tail from ';'> — the tail
                 * moves first (it is the only region beyond the expr), then the
                 * expr itself shifts right past "uint4(" */
                memmove(*msl_p + e0 + 6 + elen + 9, *msl_p + e0 + elen,
                        olen - (e0 + elen) + 1);
                memmove(*msl_p + e0 + 6, *msl_p + e0, elen);
                memcpy(*msl_p + e0, "uint4(", 6);
                memcpy(*msl_p + e0 + 6 + elen, ", 0u, 0u)", 9);
                hit = strstr(*msl_p + e0 + 6 + elen + 9, pat);
            }
        }
    }
}

/* When the canonical record flattens a struct member (SPIRV-Cross must flatten
 * struct-typed stage-in attributes of the post-tessellation TES, so a member
 * "S in_te_s" becomes "int in_te_s_x; float4 in_te_s_y;"), a producer stage whose
 * OWN struct nests the struct still writes "<expr>.in_te_s.x" — rewrite those
 * accesses to the flattened member names ("<expr>.in_te_s_x") or the rewritten
 * struct body does not compile ("no member named 'in_te_s'"; dEQP-GL45-ES31
 * tessellation.user_defined_io.* struct/array varyings). */
static bool tess_flatten_body_access(char *msl, const struct MslStruct *own, const struct CanonRec *canon)
{
    bool changed = false;

    /* the struct body still declares the NESTED members ("S in_te_s;") while the
     * parsed member list is already flattened (in_te_s_x, in_te_s_y) — take the
     * nested names from the body */
    const char *body = own->body;
    while (body && *body)
    {
        const char *semi = strchr(body, ';');
        if (!semi) break;
        const char *decl = body;
        body = semi + 1;
        /* strip an [[attribute...]] qualifier */
        const char *attr = memchr(decl, '[', (size_t)(semi - decl));
        if (attr) semi = attr;
        /* the member name is the last identifier of the declaration */
        const char *e = semi;
        while (e > decl && !isalnum((unsigned char)e[-1]) && e[-1] != '_') e--;
        const char *s = e;
        while (s > decl && (isalnum((unsigned char)s[-1]) || s[-1] == '_')) s--;
        if (s == e) continue;

        char m[128];
        size_t ml = (size_t)(e - s);
        if (ml == 0 || ml >= sizeof m) continue;
        memcpy(m, s, ml); m[ml] = 0;

        bool present = false, expanded = false;
        for (int k = 0; k < own->count; k++)
            if (strcmp(own->members[k].name, m) == 0) present = true;
        for (int k = 0; k < canon->n; k++)
        {
            if (strcmp(canon->m[k].name, m) == 0) present = true;
            if (strncmp(canon->m[k].name, m, ml) == 0 && canon->m[k].name[ml] == '_')
                expanded = true;         /* canon has m_<field> members */
        }
        if (present || !expanded) continue;
        if (tess_trace() && getenv("MGL_TESS_DEBUG2"))
            fprintf(stderr, "MGL tess flatten: body member '%s' (present=%d expanded=%d)\n", m, present, expanded);

        char pat[224];
        if (ml + 3 >= sizeof pat) continue;
        snprintf(pat, sizeof pat, ".%s", m);
        for (char *w = msl; (w = strstr(w, pat)) != NULL; )
        {
            bool skip_w = false;
            /* a member access continues with ".field" or "[index].field" */
            if (w[ml + 1] != '.' && w[ml + 1] != '[')
            { w += ml + 1; continue; }
            /* longest chain ".f1[.f2...] whose joined name m_f1[_f2...] is a canon
             * member; the replacement always SHRINKS the text (each '.' consumed) */
            char joined[256];
            char idx_txt[32];
            snprintf(joined, sizeof joined, "%s", m);
            const char *p = w + ml + 1;               /* first '.' of ".f1" */
            bool any = false;
            idx_txt[0] = 0;
            /* an array-of-struct member first appears as ".m[<i>].<f>...": the
             * canonical record stores one array per field ("m_<f>", N elements),
             * so hoist the field out of the index — ".m[i].f[j]" becomes
             * ".m_f[i][j]" and a DYNAMIC i stays a real array index.  The fold
             * only applies when canon carries the field-array member. */
            if (*p == '[')
            {
                const char *cl;
                char fld[96];
                size_t il, fl;
                char idx_all[64];

                /* collect EVERY consecutive index group (a dissolved
                 * patch-block ARRAY member reads ".m[e][i].<field>[j]": the
                 * canonical field member is "m_<field>" with dimensions
                 * [e][i][j]) */
                idx_all[0] = 0;
                cl = p;
                for (;;)
                {
                    const char *e2 = strchr(cl + 1, ']');
                    bool bad = false;

                    if (!e2) bad = true;
                    else
                    {
                        if (idx_all[0]) strncat(idx_all, "][", 2);
                        if ((size_t)(e2 - (cl + 1)) + strlen(idx_all) + 2 >= sizeof idx_all)
                            bad = true;
                        else
                        {
                            strncat(idx_all, cl + 1, (size_t)(e2 - (cl + 1)));
                            cl = e2 + 1;
                        }
                    }
                    if (bad) { skip_w = true; break; }
                    if (*cl != '[') break;
                }
                if (skip_w) { w += ml + 1; continue; }
                if (*cl != '.') { w += ml + 1; continue; }
                il = strlen(idx_all);
                fl = 0;
                while (cl[1 + fl] && (isalnum((unsigned char)cl[1 + fl]) || cl[1 + fl] == '_')) fl++;
                if (!fl || fl >= sizeof fld) { w += ml + 1; continue; }
                {
                    char cand[300];

                    if (snprintf(cand, sizeof cand, "%s_%.*s", joined, (int)fl, cl + 1) >= (int)sizeof cand)
                    { w += ml + 1; continue; }
                    bool hit = false;
                    for (int k = 0; k < canon->n; k++)
                        if (strcmp(canon->m[k].name, cand) == 0) { hit = true; break; }
                    if (!hit) { w += ml + 1; continue; }
                    snprintf(joined, sizeof joined, "%s", cand);
                    memcpy(idx_txt, idx_all, il); idx_txt[il] = 0;
                    p = cl + 1 + fl;               /* right after the field name */
                    any = true;
                }
            }
            while (*p == '.')
            {
                size_t fl = 0;
                while (p[1 + fl] && (isalnum((unsigned char)p[1 + fl]) || p[1 + fl] == '_')) fl++;
                if (!fl) break;
                char cand[300];
                if (snprintf(cand, sizeof cand, "%s_%.*s", joined, (int)fl, p + 1) >= (int)sizeof cand)
                    break;
                bool hit = false;
                for (int k = 0; k < canon->n; k++)
                    if (strcmp(canon->m[k].name, cand) == 0) { hit = true; break; }
                if (!hit) break;
                snprintf(joined, sizeof joined, "%s", cand);
                p += 1 + fl;
                any = true;
            }
            if (!any) { w += ml + 1; continue; }
            /* replacement: "." + joined [+ "[" + idx_txt + "]"]; the folded array
             * index goes AFTER the hoisted field names.  The array fold SHRINKS
             * the span (one '.' consumed); a plain struct field fold is
             * length-neutral ("vb_s.x" -> "vb_s_x") and must splice too */
            {
                char rep[320];
                size_t consumed = 1 + ml + (size_t)(p - (w + ml + 1));
                size_t rl;

                if (snprintf(rep, sizeof rep, ".%s%s%s%s", joined, idx_txt[0] ? "[" : "",
                             idx_txt[0] ? idx_txt : "", idx_txt[0] ? "]" : "") >= (int)sizeof rep)
                { w += ml + 1; continue; }
                rl = strlen(rep);
                if (rl > consumed) { w += consumed; continue; }
                if (rl < consumed)
                    memmove(w + rl, w + consumed, strlen(w + consumed) + 1);
                memcpy(w, rep, rl);
                changed = true;
                w += rl;
            }
        }
    }
    return changed;
}

/* the matrix-column merge below may only fire for a varying that THIS stage
 * writes as a whole matrix: the parse splits matrix members into columns, so
 * the only witness left is the struct BODY's declaration ("float2x2 m;") —
 * the per-element leaves of a dissolved patch-block ARRAY ("pb_v_0"/"pb_v_1")
 * are also adjacent same-key float4s and would merge into a bogus matrix */
static bool canon_body_matrix(const struct MslStruct *own, const char *base0, int run, int rows)
{
    const char *body = own ? own->body : NULL;
    char want[32];

    if (!body || !base0[0])
        return false;
    snprintf(want, sizeof want, "float%dx%d", run, rows);
    while (*body)
    {
        const char *semi = strchr(body, ';'), *decl = body;

        if (!semi)
            break;
        body = semi + 1;
        {
            const char *attr = memchr(decl, '[', (size_t)(semi - decl));

            if (attr)
                semi = attr;
        }
        /* identifier: last token of the declaration */
        {
            const char *e = semi, *s, *ts, *te;
            size_t ml, tl;
            char tmp[192];
            const char *ds;

            while (e > decl && !isalnum((unsigned char)e[-1]) && e[-1] != '_') e--;
            s = e;
            while (s > decl && (isalnum((unsigned char)s[-1]) || s[-1] == '_')) s--;
            if (s == e)
                continue;
            ml = (size_t)(e - s);
            if (ml >= sizeof tmp)
                continue;
            memcpy(tmp, s, ml);
            tmp[ml] = 0;
            /* type: the token right before the identifier */
            te = s;
            while (te > decl && (te[-1] == ' ' || te[-1] == '\t' || te[-1] == '\n')) te--;
            ts = te;
            while (ts > decl && (isalnum((unsigned char)ts[-1]) || ts[-1] == '_')) ts--;
            tl = (size_t)(te - ts);
            if (tl != strlen(want) || strncmp(ts, want, tl))
                continue;
            if (!strcmp(tmp, base0))
                return true;
            /* the dissolved spelling "<inst>_mgl_ds_<member>" keys the same
             * varying as the canon column base "<inst>_<member>" */
            ds = strstr(tmp, "mgl_ds_");
            if (ds)
            {
                memmove((void *)ds, ds + 7, strlen(ds + 7) + 1);
                if (!strcmp(tmp, base0))
                    return true;
            }
        }
    }
    return false;
}

static void tess_canon_body(char *out, size_t cap, const struct MslStruct *own,                            const struct CanonRec *canon)
{
    char *buf = NULL;
    size_t len = 0, bc = 0;
    bool first = true;
    char base0[112];

    out[0] = 0;
    for (int i = 0; i < canon->n; i++)
    {
        /* flattened MATRIX columns merge back into one matrix member: the stages
         * whose MSL keeps the interface whole write `m = float2x2(...)' — a body
         * of loose float2 columns would leave those stores nameless.  (Layout is
         * unchanged: a float<C>x<R> occupies the same C*R*4 bytes as the C
         * columns it replaces.)  Only for a stage that really has the whole
         * matrix (canon_own_matrix): a dissolved patch-block ARRAY's per-element
         * vec4 leaves ("pb_ps_0_y"/"pb_ps_1_y", "pb_v_0"/"pb_v_1") are adjacent
         * same-key float4s too and must keep their own slots. */
        size_t bl0 = strlen(canon->m[i].name);

        if (!strncmp(canon->m[i].type, "float", 5) && strlen(canon->m[i].type) == 6 &&
            bl0 >= 2 && !strcmp(canon->m[i].name + bl0 - 2, "_0"))
        {
            int run = 1;
            size_t bl = bl0;

            while (i + run < canon->n && canon->m[i + run].key == canon->m[i].key &&
                   !strcmp(canon->m[i + run].type, canon->m[i].type))
            {
                char suf[8];
                size_t rl = strlen(canon->m[i + run].name);

                snprintf(suf, sizeof suf, "_%d", run);
                /* the run member must END in the column suffix and share the
                 * base (matrix columns are "<base>_0..C-1" — a dissolved
                 * element leaf ends in its own field path) */
                if (rl <= (size_t)run ||
                    strcmp(canon->m[i + run].name + rl - strlen(suf), suf))
                    break;
                if (strncmp(canon->m[i + run].name, canon->m[i].name, bl - 2) ||
                    strcmp(canon->m[i + run].name + bl - 2, suf))
                    break;
                run++;
            }
            if (run >= 2 && run <= 4 && bl > 2 && bl - 2 < sizeof base0)
            {
                memcpy(base0, canon->m[i].name, bl - 2);
                base0[bl - 2] = 0;                  /* strip the "_0" column suffix */
                if (!canon_body_matrix(own, base0, run, canon->m[i].type[5] - '0'))
                    run = 1;                        /* no whole-matrix writer: keep
                                                 * the separate column slots */
            }
            if (run >= 2 && run <= 4 && bl > 2 && bl - 2 < sizeof base0)
            {
                MGLTessPatchMember whole = canon->m[i];

                snprintf(whole.type, sizeof whole.type, "float%dx%d", run,
                         canon->m[i].type[5] - '0');
                whole.size = canon->m[i].size * run;
                whole.array_idx = 0;
                /* emit directly (the ORIGINAL column-stripped name): the member table
                 * lookup in tess_canon_member_lines would rename it back to the
                 * stage's split "_0" column */
                str_append(&buf, &len, &bc, "%s    %s %s;\n",
                           first ? "\n" : "", whole.type, base0);
                if (!buf)
                    return;
                first = false;
                i += run - 1;                       /* the whole run is emitted */
                continue;
            }
        }
        /* not renamed here: msl_rewrite_struct_body renames (and injects the template)
         * only when the original MSL does not define spvUnsafeArray before the struct */
        tess_canon_member_lines(&buf, &len, &bc, &canon->m[i],
                                own ? own->members : NULL, own ? own->count : 0,
                                false, &first);
        if (!buf)
            return;
    }
    snprintf(out, cap, "%s", buf);
    free(buf);
}

/* replace the body of "struct NAME { ... };" in msl with new_body; returns a malloc'd
 * copy, or NULL when the struct is absent (nothing to rewrite). The spvUnsafeArray
 * template is injected when the canonical body needs it and the original declares it
 * later (SPIRV-Cross emits it after the stage's own structs in kernel form) */
static const char kSpvUnsafeArray[] =
    "template <typename T, size_t Num>\n"
    "struct mgl_spv_arr\n"
    "{\n"
    "    T elements[Num ? Num : 1];\n"
    "    \n"
    "    thread T& operator [] (size_t pos) thread\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const thread T& operator [] (size_t pos) const thread\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    device T& operator [] (size_t pos) device\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const device T& operator [] (size_t pos) const device\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    constexpr const constant T& operator [] (size_t pos) const constant\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    \n"
    "    threadgroup T& operator [] (size_t pos) threadgroup\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup\n"
    "    {\n"
    "        return elements[pos];\n"
    "    }\n"
    "};\n";

static char *msl_rewrite_struct_body(const char *msl, const char *name, const char *body)
{
    char head[256];
    const char *p, *open, *close;
    const char *tmpl = NULL;
    char *out, *new_body;
    size_t prefix, tail;

    snprintf(head, sizeof head, "struct %s\n{", name);
    p = strstr(msl, head);
    if (!p)
    {
        snprintf(head, sizeof head, "struct %s {", name);
        p = strstr(msl, head);
    }
    if (!p)
        return NULL;
    open = strchr(p, '{');
    close = open ? strchr(open, '}') : NULL;
    if (!open || !close)
        return NULL;

    /* array members reference spvUnsafeArray: if its template is not already defined
     * ahead of this struct (SPIRV-Cross emits it after the stage structs in kernel
     * form), use our own name and declare it in front — the two never collide */
    new_body = strdup(body);
    if (new_body && strstr(new_body, "spvUnsafeArray"))
    {
        bool before_us = false, before_orig = false;

        for (const char *q = msl; q < p; q++)
        {
            if (!before_orig && !strncmp(q, "struct spvUnsafeArray", 21))
                before_orig = true;
            if (!before_us && !strncmp(q, "struct mgl_spv_arr", 18))
                before_us = true;
        }
        if (!before_orig)
        {
            for (char *at = new_body; (at = strstr(at, "spvUnsafeArray")) != NULL; )
            {
                memcpy(at, "mgl_spv_arr   ", strlen("spvUnsafeArray"));
                at += strlen("spvUnsafeArray");
            }
            if (!before_us)
                tmpl = kSpvUnsafeArray;
        }
    }
    if (!new_body)
        return NULL;

    prefix = (size_t)(open + 1 - msl);
    tail = strlen(close);                 /* "};\n..." to the end */
    out = (char *)malloc(prefix + strlen(new_body) + tail + 1 + (tmpl ? strlen(tmpl) : 0));
    if (!out)
    {
        free(new_body);
        return NULL;
    }
    if (tmpl)
    {
        /* the template goes in front of the whole struct declaration */
        size_t decl = (size_t)(p - msl);

        memcpy(out, msl, decl);
        memcpy(out + decl, tmpl, strlen(tmpl));
        memcpy(out + decl + strlen(tmpl), msl + decl, prefix - decl);
        memcpy(out + prefix + strlen(tmpl), new_body, strlen(new_body));
        memcpy(out + prefix + strlen(tmpl) + strlen(new_body), close, tail + 1);
    }
    else
    {
        memcpy(out, msl, prefix);
        memcpy(out + prefix, new_body, strlen(new_body));
        memcpy(out + prefix + strlen(new_body), close, tail + 1);
    }
    free(new_body);
    return out;
}

/* ------------------------------------------------------------------ MSL rewrites
 * Called from program.c's compile_stage right after SPIRV-Cross produced the MSL.
 */

static void str_append(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap, ap2;
    int need;
    char tmp[2048];
    char *heap = NULL;
    const char *src;
    char *nb;

    va_start(ap, fmt);
    va_copy(ap2, ap);
    need = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (need < 0)
    {
        va_end(ap2);
        return;
    }
    if ((size_t)need >= sizeof tmp)
    {
        /* long appends (whole shader bodies) do not fit the stack buffer: format
         * into the heap so they are not truncated */
        heap = (char *)malloc((size_t)need + 1);
        if (!heap)
        {
            va_end(ap2);
            return;
        }
        vsnprintf(heap, (size_t)need + 1, fmt, ap2);
    }
    va_end(ap2);
    src = heap ? heap : tmp;
    if (*len + (size_t)need + 1 > *cap)
    {
        size_t nc = (*len + (size_t)need + 1) * 2;

        nb = (char *)realloc(*buf, nc);
        if (!nb)
        {
            free(heap);
            return;
        }
        *buf = nb;
        *cap = nc;
    }
    memcpy(*buf + *len, src, (size_t)need + 1);
    *len += (size_t)need;
    free(heap);
}

/* replace an exact occurrence of old with new inside a growable buffer */
static void buf_replace(char **buf, size_t *len, size_t *cap, const char *old, const char *new)
{
    char *at = strstr(*buf, old);
    size_t oldlen = strlen(old), newlen = strlen(new);

    if (!at)
        return;
    if (newlen > oldlen)
    {
        size_t prefix = (size_t)(at - *buf);
        *cap = *len + newlen - oldlen + 1;
        *buf = (char *)realloc(*buf, *cap);
        at = *buf + prefix;
    }
    memmove(at + newlen, at + oldlen, *len - (size_t)(at - *buf) - oldlen + 1);
    memcpy(at, new, newlen);
    *len += newlen - oldlen;
}

/* VS kernel (SPIRV-Cross VERTEX_FOR_TESSELLATION):
 *  - route the base vertex/instance through our params buffer (GL: gl_VertexID includes
 *    base vertex, gl_InstanceID excludes base instance)
 *  - drop the [[grid_size]] / [[grid_origin]] parameters and provide them as locals from
 *    the params buffer: fetching stage-in attributes silently returns zeros when a kernel
 *    also takes [[grid_size]] (AGX driver bug, verified with a bisected probe)
 */
/* gl_DrawID (GL_ARB_shader_draw_parameters / GLSL 4.60): BuiltIn DrawIndex is
 * demoted to a Private variable by mgl_spirv_lower_draw_index(), so
 * SPIRV-Cross emits a plain function-local declaration
 * ("int _RESERVED_IDENTIFIER_FIXUP_gl_DrawID;").  Give the function a
 * draw-parameters buffer parameter and initialize the variable from it; the
 * renderer stamps the buffer with the single draw's id at draw time
 * (draw_buffers.c exec_draw -> MGLRenderer+Draw.m / +Tessellation.m). */
/* locate the vertex function (or the tess vertex kernel) in stage MSL: its
 * keyword, '(' and the matching ')'.  Returns nonzero when none is found. */
static int stage_function(const char *msl, const char **kind, const char **open, const char **close)
{
    const char *k = strstr(msl, "\nvertex ");
    if (!k && strncmp(msl, "vertex ", 7) == 0)
        k = msl;
    if (!k)
    {
        k = strstr(msl, "\nkernel ");
        if (!k && strncmp(msl, "kernel ", 7) == 0)
            k = msl;
    }
    if (!k)
        return 1;
    const char *o = strchr(k, '(');
    if (!o)
        return 1;
    const char *c = o + 1;
    while (*c && *c != ')')
    {
        if (*c == '(')
        {
            c++;
            int depth = 1;                  /* nested parens in a parameter type */
            while (*c && depth)
            {
                depth += (*c == '(') - (*c == ')');
                c++;
            }
            continue;
        }
        c++;
    }
    if (!*c)
        return 1;
    *kind = k; *open = o; *close = c;
    return 0;
}

char *mgl_draw_params_rewrite_vertex_msl(const char *msl, int buffer_index)
{
    if (!msl)
        return NULL;
    bool draw_id = strstr(msl, "gl_DrawID;") != NULL;
    bool base_vertex = strstr(msl, "gl_BaseVertex [[base_vertex]]") != NULL;
    if (!draw_id && !base_vertex)
        return NULL;                       /* neither built-in in this stage */

    char *out = NULL;
    size_t len = 0, cap = 0;
    str_append(&out, &len, &cap, "%s", msl);

    /* 1. the parameter, appended to the vertex function / kernel signature */
    {
        const char *kind = NULL, *open = NULL, *close = NULL;
        if (stage_function(out, &kind, &open, &close))
        {
            free(out);
            return NULL;
        }
        while (close && *close && *close != ')')
        {
            if (*close == '(')
            {
                close++;                   /* nested parens in a parameter type */
                int depth = 1;
                while (*close && depth)
                {
                    depth += (*close == '(') - (*close == ')');
                    close++;
                }
                continue;
            }
            close++;
        }
        if (!close || !*close)
        {
            free(out);
            return NULL;
        }
        char add[96];
        snprintf(add, sizeof add, "%sconstant uint2* mgl_dp [[buffer(%d)]]",
                 (close[-1] == '(') ? "" : ", ", buffer_index);
        size_t addlen = strlen(add), off = (size_t)(close - out);
        char *grown = (char *)realloc(out, len + addlen + 1);
        if (!grown) { free(out); return NULL; }
        out = grown; cap = len + addlen + 1;
        memmove(out + off + addlen, out + off, len - off + 1);
        memcpy(out + off, add, addlen);
        len += addlen;
    }

    /* 2a. gl_DrawID declaration initializes from the buffer:
     *     "TYPE IDENT;" -> "TYPE IDENT = TYPE(mgl_dp[0]);" */
    if (draw_id)
    {
        const char *d = strstr(out, "_RESERVED_IDENTIFIER_FIXUP_gl_DrawID;");
        const char *d2 = !d ? strstr(out, "gl_DrawID;") : NULL;
        d = d ? d : d2;
        if (!d)
        {
            free(out);
            return NULL;
        }
        const char *istart = d;            /* identifier start (back over the fixup) */
        while (istart > out && !isspace((unsigned char)istart[-1])) istart--;
        const char *iend = istart;
        while (*iend && !isspace((unsigned char)*iend) && *iend != ';') iend++;
        const char *te = istart;           /* type token end */
        while (te > out && isspace((unsigned char)te[-1])) te--;
        const char *ts = te;
        while (ts > out && !isspace((unsigned char)ts[-1]) && ts[-1] != ';') ts--;
        char target[96], init[128];
        snprintf(target, sizeof target, "%.*s;", (int)(iend - istart), istart);
        snprintf(init, sizeof init, "%.*s = %.*s(mgl_dp[0].x);",
                 (int)(iend - istart), istart, (int)(te - ts), ts);
        buf_replace(&out, &len, &cap, target, init);
    }

    /* 2b. gl_BaseVertex: Metal fills [[base_vertex]] with vertexStart, which for
     *     arrays draws is GL's `first', not GL's base vertex (0).  Rename the
     *     attribute parameter and define gl_BaseVertex from the buffer. */
    if (base_vertex)
    {
        static const char param[] = "gl_BaseVertex [[base_vertex]]";
        buf_replace(&out, &len, &cap, param, "mgl_base_vertex_mtl [[base_vertex]]");
        /* first statement of the function body */
        const char *kind = NULL, *open = NULL, *close = NULL;
        const char *brace = NULL;
        if (!stage_function(out, &kind, &open, &close))
            brace = close ? strchr(close, '{') : NULL;
        if (brace)
        {
            static const char decl[] = "\n    int gl_BaseVertex = int(mgl_dp[0].y);";
            size_t dl = sizeof decl - 1, off = (size_t)(brace - out) + 1;
            char *grown = (char *)realloc(out, len + dl + 1);
            if (!grown) { free(out); return NULL; }
            out = grown; cap = len + dl + 1;
            memmove(out + off + dl, out + off, len - off + 1);
            memcpy(out + off, decl, dl);
            len += dl;
        }
        else
        {
            free(out);
            return NULL;
        }
    }

    return out;
}

static char *rewrite_vs_kernel(const char *msl)
{
    char *out = NULL;
    size_t len = 0, cap = 0;

    str_append(&out, &len, &cap, "%s", msl);

    /* parameters -> locals (indices must not collide with the vertex descriptor) */
    buf_replace(&out, &len, &cap, ", uint3 spvStageInputSize [[grid_size]]", "");
    buf_replace(&out, &len, &cap, ", uint3 spvDispatchBase [[grid_origin]]", "");

    /* insert the locals right after the opening brace.  The buffer is grown with realloc,
     * so the insertion point is kept as an offset — a moved allocation invalidated the
     * pointer before, the prologue landed in freed memory and the kernel compiled with
     * undeclared spvStageInputSize / mgl_vs_params (KHR-GL46 tessellation clusters) */
    {
        char *open = strstr(out, "kernel void ");
        open = open ? strchr(open, '{') : NULL;
        if (!open)
        {
            free(out);
            return NULL;
        }

        char prologue[512];
        int n = snprintf(prologue, sizeof prologue,
                         "\n    uint3 spvStageInputSize = uint3(mgl_vs_params[1], mgl_vs_params[2], 1u);\n"
                         "    uint3 spvDispatchBase = uint3(mgl_vs_params[0], mgl_vs_params[3], 0u);\n");
        (void)n;
        size_t plen = strlen(prologue), open_off = (size_t)(open + 1 - out);
        size_t taillen = len - open_off;
        char *grown = (char *)realloc(out, len + plen + 1);
        if (!grown) { free(out); return NULL; }
        out = grown; cap = len + plen + 1;
        char *ins = out + open_off;
        memmove(ins + plen, ins, taillen + 1);
        memcpy(ins, prologue, plen);
        len += plen;
    }

    /* GL id semantics (replaces the uses of the removed locals).  The
     * replacements must stay valid MSL: `as_uint' is an OpenCL/HLSL builtin
     * (Metal spells it as_type<uint>()), and no cast is needed anyway —
     * gl_GlobalInvocationID and mgl_vs_params are already uint, so a plain
     * expression is type-correct (KHR-GL46.draw_elements_base_vertex_tests.
     * AEP_shader_stages links a tess VS that reads gl_InstanceID and failed
     * with "use of undeclared identifier 'as_uint'"). */
    buf_replace(&out, &len, &cap,
                "uint gl_VertexID = as_uint(gl_GlobalInvocationID.x + spvDispatchBase.x);",
                "uint gl_VertexID = gl_GlobalInvocationID.x + mgl_vs_params[0];");
    buf_replace(&out, &len, &cap,
                "uint gl_VertexID = gl_GlobalInvocationID.x + spvDispatchBase.x;",
                "uint gl_VertexID = gl_GlobalInvocationID.x + mgl_vs_params[0];");
    buf_replace(&out, &len, &cap,
                "uint gl_InstanceID = as_uint(gl_GlobalInvocationID.y + spvDispatchBase.y);",
                "uint gl_InstanceID = gl_GlobalInvocationID.y;");
    buf_replace(&out, &len, &cap,
                "uint gl_InstanceID = gl_GlobalInvocationID.y + spvDispatchBase.y;",
                "uint gl_InstanceID = gl_GlobalInvocationID.y;");

    /* append the params parameter to the kernel signature */
    {
        char *p = strstr(out, "kernel void ");
        char *paren = p ? strchr(p, '(') : NULL;
        int depth = 0;
        char *close = NULL;

        if (!paren)
        {
            free(out);
            return NULL;
        }
        for (char *q = paren; *q; q++)
        {
            if (q[0] == '[' && q[1] == '[') { const char *e = strstr(q, "]]"); if (e) { q = (char *)e; continue; } }
            if (*q == '(') depth++;
            else if (*q == ')') { depth--; if (depth == 0) { close = q; break; } }
        }
        if (!close)
        {
            free(out);
            return NULL;
        }

        char add[128];
        size_t addlen, taillen, close_off;
        snprintf(add, sizeof add, ", constant uint* mgl_vs_params [[buffer(%d)]]", MGL_TESS_VS_PARAMS);
        addlen = strlen(add);
        taillen = strlen(close);
        close_off = (size_t)(close - out);          /* offset: realloc may move the buffer */
        char *grown = (char *)realloc(out, len + addlen + 1);
        if (!grown) { free(out); return NULL; }
        out = grown; cap = len + addlen + 1;
        close = out + close_off;
        memmove(close + addlen, close, taillen + 1);
        memcpy(close, add, addlen);
        len += addlen;
    }

    return out;
}

/* ------------------------------------------------------------------ TES rewrite
 * SPIRV-Cross emits the TES as a post-tessellation vertex function
 *   [[patch(triangle, N)]] vertex out_t entry(in_patch_t patchIn [[stage_in]],
 *                                            uint gl_PrimitiveID [[patch_id]],
 *                                            float3 gl_TessCoord [[position_in_patch]])
 * This rewrite turns it into a plain vertex function reading the tessellation
 * coordinates / control points / patch data from buffers indexed by [[vertex_id]],
 * which works on GPUs without hardware tessellation:
 *   vertex out_t entry(device const float4* coords [[buffer(27)]],
 *                      device const in_t* cp [[buffer(28)]],
 *                      device const uchar* patch [[buffer(29)]],
 *                      constant uint* params [[buffer(26)]],
 *                      uint mgl_vid [[vertex_id]])
 */
static const char *kTessPrelude =
    "template <typename T>\n"
    "struct mgl_cp_ref\n"
    "{\n"
    "    device const T *mgl_p;\n"
    "    uint mgl_n;\n"
    "    device const T & operator[] (size_t i) const { return mgl_p[i]; }\n"
    "    uint size() const { return mgl_n; }\n"
    "};\n\n";

static char *rewrite_tes(const char *msl, const MGLTessProgram *tp)
{
    const char *sig = strstr(msl, "[[ patch(");
    if (!sig)
        sig = strstr(msl, "[[patch(");
    if (!sig)
        { TES_REWRITE_FAIL(9); }

    const char *vertex_kw = strstr(sig, "vertex ");
    if (!vertex_kw || (size_t)(vertex_kw - sig) > 64)
        { TES_REWRITE_FAIL(13); }
    const char *sig_open = strchr(vertex_kw, '(');
    if (!sig_open)
        { TES_REWRITE_FAIL(16); }

    /* matching close paren of the parameter list */
    const char *sig_close = NULL;
    int depth = 0;
    for (const char *q = sig_open; *q; q++)
    {
        if (q[0] == '[' && q[1] == '[')
        {
            const char *e = strstr(q, "]]");
            if (e) { q = e; continue; }
        }
        if (*q == '(') depth++;
        else if (*q == ')') { if (--depth == 0) { sig_close = q; break; } }
    }
    if (!sig_close)
        { TES_REWRITE_FAIL(32); }

    /* the opening brace of the function body */
    const char *body_open = strchr(sig_close, '{');
    if (!body_open)
        { TES_REWRITE_FAIL(37); }

    /* return type and entry name: the entry name is the last identifier before '(' */
    char entry_ret[128], entry_name[128];
    {
        const char *name_start = sig_open;
        while (name_start > vertex_kw && *(name_start - 1) != ' ' && *(name_start - 1) != '\n' && *(name_start - 1) != '\t')
            name_start--;
        if (name_start <= vertex_kw + 7 || name_start == sig_open)
            return NULL;
        snprintf(entry_name, sizeof entry_name, "%.*s", (int)(sig_open - name_start), name_start);
        snprintf(entry_ret, sizeof entry_ret, "%.*s", (int)(name_start - vertex_kw - 8), vertex_kw + 7);
    }

    /* patchIn struct type name: first word of the parameter list. A TES that reads
     * neither its control points nor any per-patch varying gets no stage_in struct at
     * all (the first parameter is then gl_TessCoord or patch_id) — flagged and every
     * patchIn-related rewrite is skipped */
    char patchin_name[128];
    bool no_patchin = false;
    {
        const char *p = sig_open + 1;
        while (*p == ' ' || *p == '\n') p++;
        const char *e = p;
        while (*e && *e != ' ' && *e != '\n') e++;
        if (e == p || (size_t)(e - p) >= sizeof patchin_name)
            { TES_REWRITE_FAIL(62); }
        snprintf(patchin_name, sizeof patchin_name, "%.*s", (int)(e - p), p);

        char head[160];
        snprintf(head, sizeof head, "struct %s\n{", patchin_name);
        if (!strstr(msl, head))
        {
            snprintf(head, sizeof head, "struct %s {", patchin_name);
            if (!strstr(msl, head))
                no_patchin = true;      /* no stage_in struct: no inputs to rebuild */
        }
    }

    /* control point struct type: from patch_control_point<T> in the patchIn struct */
    char cp_type[128];
    {
        char head[160];
        const char *p = NULL;
        const char *pcp, *t0, *t1;

        if (!no_patchin)
        {
            snprintf(head, sizeof head, "struct %s\n{", patchin_name);
            p = strstr(msl, head);
            if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
        }
        pcp = p ? strstr(p, "patch_control_point<") : NULL;
        if (pcp && pcp < strchr(p, '}'))
        {
            t0 = pcp + strlen("patch_control_point<");
            t1 = strchr(t0, '>');
            if (!t1 || (size_t)(t1 - t0) >= sizeof cp_type)
                { TES_REWRITE_FAIL(82); }
            memcpy(cp_type, t0, (size_t)(t1 - t0));
            cp_type[t1 - t0] = 0;
        }
        else
        {
            /* no control points referenced: the type is unused, but must name a type */
            snprintf(cp_type, sizeof cp_type, "mgl_cp_rec");
        }
    }

    struct MslStruct patch_in;
    memset(&patch_in, 0, sizeof patch_in);
    if (!no_patchin && !msl_parse_struct(msl, patchin_name, &patch_in))
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: rewrite_tes: patchIn struct parse failed (%s)\n", patchin_name);
        { TES_REWRITE_FAIL(92); }
    }
    if (tess_trace())
        fprintf(stderr, "MGL tess: rewrite_tes: sig ok, cp=%s patchin=%s members=%d cpref=%d\n",
                cp_type, patchin_name, patch_in.count, patch_in.cp_ref_member);

    /* ---- assemble the rewritten source ---- */
    char *out = NULL;
    size_t len = 0, cap = 0;

    const char *ins = strstr(msl, "using namespace metal;");
    if (!ins)
        { TES_REWRITE_FAIL(104); }
    ins += strlen("using namespace metal;");
    str_append(&out, &len, &cap, "%.*s%s", (int)(ins - msl), msl, kTessPrelude);

    /* everything between the prelude and "[[ patch(", with the canonical control point
     * record type declared immediately after the prelude, BEFORE the stage's own
     * structs (the patched patchIn struct references it) */
    if (tp && tp->num_cp_record > 0)
    {
        bool needs_arr = false;

        for (int j = 0; j < tp->num_cp_record && !needs_arr; j++)
            if (strstr(tp->cp_record[j].type, "spvUnsafeArray"))
                needs_arr = true;
        if (needs_arr)
            str_append(&out, &len, &cap, "%s", kSpvUnsafeArray);
        str_append(&out, &len, &cap, "\nstruct mgl_cp_rec\n{\n");
        {
            bool first = true;

            for (int j = 0; j < tp->num_cp_record; j++)
            {
                /* renamed: the template above is ours; array slots the TES flattened
                 * ("tc_arr_0", "tc_arr_1") are re-expanded so the record's members have
                 * the names and element types the TES code accesses them by */
                tess_canon_member_lines(&out, &len, &cap, &tp->cp_record[j],
                                        tp->tes_cp, tp->num_tes_cp, true, &first);
            }
        }
        str_append(&out, &len, &cap, "};\n");
        if (getenv("MGL_TESS_DEBUG2"))
            fprintf(stderr, "MGL tess TES mgl_cp_rec (%u members):\n%s\n", tp->num_cp_record,
                    out ? strstr(out, "struct mgl_cp_rec") : "(nil)");
    }
    else
    {
        /* no canonical record (TES reads neither control points nor per-patch varyings,
         * or pairing is off): the entry signature still names the control point buffer,
         * so the fallback type must be declared here — a TES with no cp inputs gets
         * cp_type == "mgl_cp_rec" and nothing would declare it ("unknown type name
         * 'mgl_cp_rec'" from newLibraryWithSource) */
        str_append(&out, &len, &cap, "\nstruct mgl_cp_rec\n{\n    float4 gl_Position;\n};\n");
    }
    str_append(&out, &len, &cap, "%.*s", (int)(sig - ins), ins);

    /* new signature.  The original entry point's buffer/texture/sampler
     * parameters must survive: everything except the patch stage_in struct,
     * gl_TessCoord / patch_id, which the prologue rebuilds from the coordinate
     * stream (a TES reading a uniform block would otherwise lose it —
     * "use of undeclared identifier" in the generated body). */
    char *kept = NULL;
    size_t klen = 0, kcap = 0;
    {
        const char *ps = sig_open + 1;
        while (ps < sig_close)
        {
            const char *e = ps;
            int depth = 0;

            while (e < sig_close && (depth || *e != ','))
            {
                if (*e == '(' || *e == '<') depth++;
                else if (*e == ')' || *e == '>') depth--;
                e++;
            }
            {
                size_t n = (size_t)(e - ps);
                char ent[512];

                if (n >= sizeof ent) n = sizeof ent - 1;
                memcpy(ent, ps, n);
                ent[n] = 0;
                if (!strstr(ent, "[[stage_in]]") && !strstr(ent, "position_in_patch") &&
                    !strstr(ent, "[[patch_id]]"))
                    str_append(&kept, &klen, &kcap, "%s%s", klen ? ", " : "", ent);
            }
            ps = e < sig_close ? e + 1 : e;
            while (ps < sig_close && (*ps == ' ' || *ps == '\n' || *ps == '\t')) ps++;
        }
    }
    str_append(&out, &len, &cap,
               "vertex %s %s(device const float4* mgl_tess_coords [[buffer(%d)]], "
               "device const %s* mgl_tess_cp [[buffer(%d)]], "
               "device const uchar* mgl_tess_patch [[buffer(%d)]], "
               "constant uint* mgl_tess_params [[buffer(%d)]], "
               "uint mgl_vid [[vertex_id]]%s%s)",
               entry_ret, entry_name, MGL_TESS_TES_COORDS,
               (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type, MGL_TESS_TES_CP,
               MGL_TESS_TES_PATCH, MGL_TESS_TES_PARAMS,
               kept && klen ? ", " : "", kept ? kept : "");
    free(kept);

    /* prologue: rebuild patchIn from the buffers. Offsets and the record stride come
     * from the canonical per-patch record (tp->tes_patch_in), so the blob layout is
     * identical no matter which members the TES happens to declare */
    size_t patch_stride = patch_in.size;
    {
        const MGLTessProgram *ctp = tp;
        if (ctp && ctp->num_tes_patch_in)
            patch_stride = ctp->tes_patch_in_size;
    }
    if (!no_patchin)
    {
        str_append(&out, &len, &cap, "\n{\n    %s patchIn;\n", patchin_name);
        for (int i = 0; i < patch_in.count; i++)
        {
            size_t moff = patch_in.members[i].offset;

            if (i == patch_in.cp_ref_member)
                continue;
            if (tp)
                for (int j = 0; j < tp->num_tes_patch_in; j++)
                    if (!strcmp(tp->tes_patch_in[j].name, patch_in.members[i].name) &&
                        !strcmp(tp->tes_patch_in[j].type, patch_in.members[i].type))
                    {
                        moff = tp->tes_patch_in[j].offset;   /* canonical offset */
                        break;
                    }
            /* per-patch record: the blob holds total_patches records of the canonical
             * patch size, indexed by the patch id the tessellator stored in the
             * coordinate stream */
            str_append(&out, &len, &cap,
                       "    patchIn.%s = *(device const %s*)(mgl_tess_patch + "
                       "(size_t)mgl_tess_coords[mgl_vid].w * %zu + %zu);\n",
                       patch_in.members[i].name, patch_in.members[i].type, patch_stride, moff);
        }
        if (patch_in.cp_ref_member >= 0)
            str_append(&out, &len, &cap,
                       "    patchIn.%s = mgl_cp_ref<%s>{ &mgl_tess_cp[(uint)mgl_tess_coords[mgl_vid].w * mgl_tess_params[0]], mgl_tess_params[0] };\n",
                       patch_in.members[patch_in.cp_ref_member].name,
                       (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type);
    }
    else
        str_append(&out, &len, &cap, "\n{\n");

    /* debug hook: dump control point N as the vertex position (MGL_TESS_CPDBG=N) */
    if (tp && tp->num_cp_record > 0)
    {
        const char *n = getenv("MGL_TESS_CPDBG");
        if (n && *n)
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(mgl_tess_cp[%s].gl_Position.xy, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret, n);
        else if (getenv("MGL_TESS_PARAMDBG"))
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(float(mgl_tess_params[0]) * 0.2 - 0.9, float(mgl_tess_params[1]) * 0.2 - 0.9, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret);
        else if (getenv("MGL_TESS_COORDDBG"))
            str_append(&out, &len, &cap,
                       "\n    { %s dbg = {}; dbg.gl_Position = float4(mgl_tess_coords[mgl_vid].xy * 1.6 - 0.8, 0.0, 1.0); dbg.mgl_rs_PointSize = 1.0; return dbg; }\n",
                       entry_ret);
    }

    /* direct builtin parameters of the original entry point */
    {
        char params[768];
        snprintf(params, sizeof params, "%.*s", (int)(sig_close - sig_open - 1), sig_open + 1);
        if (strstr(params, "[[patch_id]]"))
            str_append(&out, &len, &cap,
                       "    uint gl_PrimitiveID = (uint)mgl_tess_coords[mgl_vid].w %% max(mgl_tess_params[1], 1u);\n");
        if (strstr(params, "float2 gl_TessCoordIn [[position_in_patch]]"))
            str_append(&out, &len, &cap, "    float2 gl_TessCoordIn = mgl_tess_coords[mgl_vid].xy;\n");
        else if (strstr(params, "float3 gl_TessCoord [[position_in_patch]]"))
            str_append(&out, &len, &cap, "    float3 gl_TessCoord = mgl_tess_coords[mgl_vid].xyz;\n");
    }

    /* original body */
    str_append(&out, &len, &cap, "%s", body_open + 1);
    if (getenv("MGL_TESS_STEPS")) { FILE *fq = fopen("/tmp/tessstep0.txt", "w"); if (fq) { fputs(out, fq); fclose(fq); } }

    /* strip [[attribute(N)]] from the two struct declarations and replace
       patch_control_point<T> by mgl_cp_ref<mgl_cp_rec> so the types work outside
       stage_in and the control point stride is the canonical record's */
    const char *decl_names[2] = { no_patchin ? "" : patchin_name, cp_type };
    for (int d = 0; d < 2; d++)
    {
        char head[160];
        const char *p;

        if (!decl_names[d][0])
            continue;                       /* no patchIn struct at all */
        snprintf(head, sizeof head, "struct %s\n{", decl_names[d]);
        p = strstr(out, head);
        if (!p) { snprintf(head, sizeof head, "struct %s {", decl_names[d]); p = strstr(out, head); }
        if (!p)
            continue;
        const char *open = strchr(p, '{');
        const char *close = strchr(open, '}');
        if (!open || !close)
            continue;

        char canon[4096];
        char *canon_buf = NULL;
        const char *newbody = NULL;

        if (d == 1 && tp && tp->num_cp_record > 0)
        {
            /* every TES member must be present in the canonical record (it is, unless
             * the union bailed out) — otherwise keep the stage's own layout */
            bool complete = true;

            for (int k = 0; k < tp->num_tes_cp && complete; k++)
            {
                bool found = false;
                for (int j = 0; j < tp->num_cp_record; j++)
                    if (tp->cp_record[j].key == tp->tes_cp[k].key)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                    complete = false;
            }
            if (complete)
            {
                bool first = true;
                size_t blen = 0, bcap = 0;

                canon[0] = 0;
                for (int j = 0; j < tp->num_cp_record; j++)
                {
                    /* renamed: the plain TES declares our own template in the prelude
                     * (kSpvUnsafeArray), SPIRV-Cross's may not exist in vertex form */
                    tess_canon_member_lines(&canon_buf, &blen, &bcap, &tp->cp_record[j],
                                            tp->tes_cp, tp->num_tes_cp, true, &first);
                    if (!canon_buf)
                        break;
                }
                newbody = canon_buf;
            }
        }

        if (!newbody)
        {
            size_t bodylen = (size_t)(close - open - 1);
            char *body = (char *)malloc(bodylen + 1);
            if (!body)
                continue;
            memcpy(body, open + 1, bodylen);
            body[bodylen] = 0;

            for (char *at = body; (at = strstr(at, "[[")) != NULL; )
            {
                char *e = strstr(at, "]]");
                if (!e) break;
                memmove(at, e + 2, strlen(e + 2) + 1);
            }
            for (char *at = body; (at = strstr(at, "patch_control_point<")) != NULL; )
            {
                size_t plen = strlen("patch_control_point<");
                char *close_ = strchr(at + plen, '>');
                char repl[192];

                snprintf(repl, sizeof repl, "mgl_cp_ref<%s>",
                         (tp && tp->num_cp_record > 0) ? "mgl_cp_rec" : cp_type);
                if (!close_)
                    break;
                /* drop the original template argument, keep the member name after '>' */
                memmove(at + strlen(repl), close_ + 1, strlen(close_ + 1) + 1);
                memcpy(at, repl, strlen(repl));
                at += strlen(repl);
            }
            newbody = body;
        }

        {
            /* splice "struct X { OLD }" -> "struct X { NEW }".  Offsets are
             * recomputed from strlen(out) (ground truth) because the tracked
             * `len` drifts when the fallback path above already shrank its
             * copied body.  The buffer is only ever GROWN: a shrinking realloc
             * would truncate below `slen`, and the memmove source tail
             * [off+1+bodylen, slen) would read past the allocation. */
            size_t bodylen = (size_t)(close - open - 1);
            size_t slen = strlen(out);
            size_t off = (size_t)(open - out);
            size_t newlen = slen - bodylen + strlen(newbody);
            const char *hold = newbody;

            if (cap < slen + 1 || cap < newlen + 1)
            {
                size_t want = (slen > newlen ? slen : newlen) + 1;
                char *grown = (char *)realloc(out, want);

                if (!grown)
                    return NULL;
                out = grown;
                cap = want;
            }
            memmove((void *)(out + off + 1 + strlen(newbody)),
                    out + off + 1 + bodylen, slen - (off + 1 + bodylen));
            memcpy((void *)(out + off + 1), newbody, strlen(newbody));
            out[newlen] = 0;   /* explicit terminator: never read out[slen] */
            len = newlen;
            if (getenv("MGL_TESS_STEPS")) { char pth[64]; snprintf(pth, sizeof pth, "/tmp/tessstep%d.txt", d + 1); FILE *fq = fopen(pth, "w"); if (fq) { fputs(out, fq); fclose(fq); }
                const char *z = memchr(out, 0, len + 1);
                fprintf(stderr, "SPLICE-POST d=%d newlen=%zu len=%zu strlen=%zu cap=%zu nul_at=%ld ctx=[%.40s]\n",
                        d, newlen, len, strlen(out), cap, z ? (long)(z - out) : -1L, z ? z - 20 : "");
                if (z) { fprintf(stderr, "  hex@nul-8:"); for (int q = -8; q <= 2; q++) fprintf(stderr, " %02x", (unsigned char)z[q]); fprintf(stderr, "\n"); } }
            if (hold != canon)
                free((void *)hold);
        }
    }

    return out;
}

/* hardware TES: SPIRV-Cross's post-tessellation vertex function, with the control point
 * count of [[patch(domain, 0)]] fixed up from the TCS (Metal ignores a mismatch with
 * drawPatches, but gl_PatchVerticesIn reads it) */
static char *rewrite_tes_hardware(const char *msl, int cp_count)
{
    const char *sig = strstr(msl, "[[ patch(");
    char *out = NULL;
    size_t len = 0, cap = 0;

    if (!sig)
        sig = strstr(msl, "[[patch(");
    if (!sig)
        return NULL;

    /* "[[ patch(triangle, 0)]] vertex" -> "[[ patch(triangle, N)]] vertex" */
    {
        const char *comma = strchr(sig, ',');
        const char *digits, *rest;
        if (!comma)
            return NULL;
        digits = comma + 1;
        while (*digits == ' ')
            digits++;
        rest = digits;
        while (*rest >= '0' && *rest <= '9')
            rest++;

        str_append(&out, &len, &cap, "%.*s%d%s", (int)(digits - msl), msl,
                   cp_count > 0 ? cp_count : 3, rest);
    }
    return out;
}

/* public re-substitution used when GL_PATCH_VERTICES changes after link */
char *mgl_tess_rewrite_tes_hw(const char *msl, int n)
{
    return rewrite_tes_hardware(msl, n);
}

/* entry from program.c: rewrite the MSL of a tessellation-pipeline stage */
char *mgl_tess_rewrite_stage_msl(GLMContext ctx, int stage, bool tess_pipeline, const char *msl)
{
    if (!msl)
        return NULL;

    if (stage == _TESS_EVALUATION_SHADER)
    {
        /* keep the native post-tessellation vertex function (hardware path), only
         * fixing the baked control point count; the plain-vertex variant used for
         * isolines / point_mode is generated from it on demand (mgl_tess_plain_msl) */
        int cp_count = 3;
        if (ctx->state.program && ctx->state.program->spirv[_TESS_CONTROL_SHADER].ir)
        {
            int modes[16], count = 0, out_vertices = 0;
            if (spv_tess_modes(ctx->state.program->spirv[_TESS_CONTROL_SHADER].ir,
                               ctx->state.program->spirv[_TESS_CONTROL_SHADER].size,
                               1 /* TessellationControl */, modes, &count, &out_vertices) &&
                out_vertices > 0)
                cp_count = out_vertices;
        }
        return rewrite_tes_hardware(msl, cp_count);
    }

    if (stage == _VERTEX_SHADER && tess_pipeline)
        return rewrite_vs_kernel(msl);

    return NULL;
}

/* plain-vertex variant of the TES (isolines / point_mode): generated from the stored
 * hardware MSL and cached. The caller gets a copy and frees it. */
char *mgl_tess_plain_msl(GLMContext ctx, MGLTessProgram *tp)
{
    if (!tp || !tp->program)
        return NULL;
    if (!tp->plain_msl)
    {
        const char *msl = tp->program->spirv[_TESS_EVALUATION_SHADER].msl_str;
        char *plain = rewrite_tes(msl, tp);
        if (getenv("MGL_DUMP_MSL") && plain)
            fprintf(stderr, "==== plain TES MSL:\n%s\n==== end\n", plain);
        if (!plain)
            return NULL;
        tp->plain_msl = plain;
    }
    return strdup(tp->plain_msl);
}

/* ------------------------------------------------------------------ CPU tessellator
 * GL 4.6 §11.3.4 "Tessellation": generate the tessellation coordinates and the
 * connectivity of the tessellated patch. Vertex arrays are laid out per patch:
 *   coordinates: float4 (u, v, w, patch id) per tessellated vertex
 *   indices:     triangle/line/point lists in the same order
 */

struct TessMesh {
    float    *coords;      /* 4 per vertex */
    unsigned ncoords, coord_cap;
    uint32_t *indices;
    unsigned nindices, index_cap;
};

static bool mesh_reserve(struct TessMesh *m, unsigned more_coords, unsigned more_indices)
{
    if (m->ncoords + more_coords > m->coord_cap)
    {
        unsigned cap = (m->ncoords + more_coords) * 2 + 64;
        float *c = (float *)realloc(m->coords, cap * sizeof(float) * 4);
        if (!c) return false;
        m->coords = c;
        m->coord_cap = cap;
    }
    if (m->nindices + more_indices > m->index_cap)
    {
        unsigned cap = (m->nindices + more_indices) * 2 + 64;
        uint32_t *i = (uint32_t *)realloc(m->indices, cap * sizeof(uint32_t));
        if (!i) return false;
        m->indices = i;
        m->index_cap = cap;
    }
    return true;
}

static unsigned mesh_vertex(struct TessMesh *m, float u, float v, float w, float pid)
{
    if (!mesh_reserve(m, 1, 0))
        return 0;
    m->coords[m->ncoords * 4 + 0] = u;
    m->coords[m->ncoords * 4 + 1] = v;
    m->coords[m->ncoords * 4 + 2] = w;
    m->coords[m->ncoords * 4 + 3] = pid;
    return m->ncoords++;
}

static void mesh_index(struct TessMesh *m, unsigned a, unsigned b, unsigned c)
{
    if (!mesh_reserve(m, 0, 3))
        return;
    m->indices[m->nindices++] = a;
    m->indices[m->nindices++] = b;
    m->indices[m->nindices++] = c;
}

/* ------------------------------------------------------------------ D3D11 tessellator
 * Port of Microsoft's D3D11 fixed-function tessellator reference (May 7, 2012,
 * amar.patel@microsoft.com) as adapted for GPU execution by Mesa's "poly"
 * library (.agent/ref/poly/cl/tessellator.h; Copyright (c) Microsoft Corporation
 * and Copyright 2023-2024 Valve Corporation, MIT licensed).  Mesa runs it on
 * Apple GPUs to pass GL 4.6 tessellation conformance; MGL runs the same
 * algorithm as the CPU reference (this file) and, transcribed 1:1, as the GPU
 * compute tessellator of MGLTessKernels.h (isolines + point_mode + capture).
 *
 * Differences from the reference, all of them API-facing rather than algorithmic:
 *   - GL domains: the point writer emits barycentric (u, v, w) for triangles and
 *     (u, v) for quads/isolines as floats, instead of packed fixed-point pairs.
 *   - GL winding: reference triangles are counter-clockwise in the (u, v) chart
 *     (the reference is "clockwise" only under D3D's V-down convention), so the
 *     last two indices are swapped when the TES declares cw.
 *   - GL culling: a patch with any relevant outer level <= 0 (or NaN) generates
 *     nothing, matching the reference and Mesa's GL driver.
 * The traversal order, the ring layout (outer edges subdivided ONLY by their own
 * outer level, stitched to the inner rings with the ruler-function order) and the
 * exact vertex/index count formulas are unchanged, so the output is bit-identical
 * to the reference up to the float conversion of the n.16 coordinates.
 */

typedef unsigned D3D_FXP;

#define D3D_FXP_FRACTION_BITS 16
#define D3D_FXP_FRACTION_MASK 0x0000ffffu
#define D3D_FXP_INTEGER_MASK  0x7fff0000u
#define D3D_FXP_ONE           (1u << D3D_FXP_FRACTION_BITS)
#define D3D_FXP_ONE_THIRD     0x00005555u
#define D3D_FXP_TWO_THIRDS    0x0000aaaau
#define D3D_FXP_ONE_HALF      0x00008000u

#define D3D_MIN_ODD  1.0f
#define D3D_MAX_ODD  63.0f
#define D3D_MIN_EVEN 2.0f
#define D3D_MAX_EVEN 64.0f
#define D3D_MIN_ODD_PLUS_HALF_EPS (D3D_MIN_ODD + 0.0000152587890625f / 2.0f)

/* spacing -> reference partitioning */
#define D3D_PARTITIONING_INTEGER          0
#define D3D_PARTITIONING_FRACTIONAL_EVEN  1
#define D3D_PARTITIONING_FRACTIONAL_ODD   2

struct D3DFactorCtx {
    D3D_FXP fraction;    /* fxpHalfTessFactorFraction */
    int     num_half;    /* numHalfTessFactorPoints */
    int     split_point; /* splitPointOnFloorHalfTessFactor */
    D3D_FXP inv_floor;   /* fxpInvNumSegmentsOnFloorTessFactor */
    D3D_FXP inv_ceil;    /* fxpInvNumSegmentsOnCeilTessFactor */
};

/* per-patch generator state (count mode fills ncoord/nindex only) */
struct D3DTess {
    struct TessMesh *mesh;
    float  pid;
    unsigned base;        /* index of this patch's first vertex in the stream */
    bool    ccw;
    bool    tri;          /* triangles: emit w = 1 - u - v */
    bool    count_only;
    unsigned ncoord, nindex;
    /* index patching (see the ring stitching below) */
    bool patched, patched2;
    int ipc_inside_delta, ipc_inside_bad, ipc_inside_repl;
    int ipc_outside_base, ipc_outside_delta, ipc_outside_bad, ipc_outside_repl;
    int ipc2_base_invert, ipc2_inv_end, ipc2_corner_bad, ipc2_corner_repl;
};

static D3D_FXP d3d_float_to_fixed(float input)
{
    return (D3D_FXP)(input * 65536.0f + 0.5f);
}

static bool d3d_is_odd(float input)
{
    return ((int)input) & 1;
}

static D3D_FXP d3d_ceil(D3D_FXP input)
{
    if (input & D3D_FXP_FRACTION_MASK)
        return (input & D3D_FXP_INTEGER_MASK) + D3D_FXP_ONE;
    return input;
}

static D3D_FXP d3d_floor(D3D_FXP input)
{
    return input & D3D_FXP_INTEGER_MASK;
}

static unsigned d3d_remove_msb(unsigned val)
{
    unsigned bit = val ? (1u << (31 - __builtin_clz(val))) : 0;
    return val & ~bit;
}

static float d3d_clamp_factor(float factor, int partitioning, float maxf)
{
    float lower = (partitioning == D3D_PARTITIONING_FRACTIONAL_EVEN) ? D3D_MIN_EVEN : D3D_MIN_ODD;
    float upper = (partitioning == D3D_PARTITIONING_FRACTIONAL_ODD) ? D3D_MAX_ODD : D3D_MAX_EVEN;

    /* if any factor ends up > 1, force the inside factors > 1 so a picture
     * frame is generated (fractional-odd only) */
    if (partitioning == D3D_PARTITIONING_FRACTIONAL_ODD && maxf > D3D_MIN_ODD_PLUS_HALF_EPS)
        lower = D3D_MIN_ODD + 0.0000152587890625f;

    factor = factor < lower ? lower : (factor > upper ? upper : factor);
    if (partitioning == D3D_PARTITIONING_INTEGER)
        factor = ceilf(factor);
    return factor;
}

static int d3d_num_points_for_factor(bool odd, D3D_FXP fxp)
{
    D3D_FXP f = fxp + (odd ? (D3D_FXP_ONE + 1) : 1);
    int r = (int)(d3d_ceil(f / 2) >> (D3D_FXP_FRACTION_BITS - 1));
    return odd ? r : r + 1;
}

static void d3d_compute_factor_ctx(bool odd, D3D_FXP fxp, struct D3DFactorCtx *c)
{
    D3D_FXP half = (fxp + 1) / 2;
    D3D_FXP fl, ce;
    int nfloor, nceil;

    /* half == 1/2 if the factor is 1, but we are pretending we are even */
    if (odd || half == D3D_FXP_ONE_HALF)
        half += D3D_FXP_ONE_HALF;
    fl = d3d_floor(half);
    ce = d3d_ceil(half);
    c->fraction = half - fl;
    c->num_half = (int)(ce >> D3D_FXP_FRACTION_BITS);
    if (ce == fl)
        c->split_point = c->num_half + 1;               /* ignored by the test below */
    else if (odd)
        c->split_point = (fl == D3D_FXP_ONE) ? 0 : (int)((d3d_remove_msb((unsigned)(fl >> D3D_FXP_FRACTION_BITS) - 1u) << 1) + 1);
    else
        c->split_point = (int)((d3d_remove_msb((unsigned)(fl >> D3D_FXP_FRACTION_BITS)) << 1) + 1);
    nfloor = (int)((fl * 2) >> D3D_FXP_FRACTION_BITS);
    nceil = (int)((ce * 2) >> D3D_FXP_FRACTION_BITS);
    if (odd) { nfloor -= 1; nceil -= 1; }
    c->inv_floor = d3d_float_to_fixed(1.0f / (float)nfloor);
    c->inv_ceil = d3d_float_to_fixed(1.0f / (float)nceil);
}

static D3D_FXP d3d_place_point_in_1d(const struct D3DFactorCtx *c, bool odd, int point)
{
    bool flip = point >= c->num_half;
    D3D_FXP loc;
    unsigned idx_ceil, idx_floor;

    if (flip)
        point = (c->num_half << 1) - point - odd;
    /* the middle point cannot be reproduced by the 16-bit fixed math below */
    if (point == c->num_half)
        return D3D_FXP_ONE_HALF;

    idx_ceil = (unsigned)point;
    idx_floor = idx_ceil;
    if (point > c->split_point)
        idx_floor -= 1;
    loc = (idx_floor * c->inv_floor) * (D3D_FXP_ONE - c->fraction) +
          (idx_ceil * c->inv_ceil) * c->fraction;
    loc = (loc + D3D_FXP_ONE_HALF) >> D3D_FXP_FRACTION_BITS;
    if (flip)
        loc = D3D_FXP_ONE - loc;
    return loc;
}

/* ---- output: one domain point / one triangle, with GL winding ---- */

static unsigned d3d_point(struct D3DTess *t, D3D_FXP u, D3D_FXP v)
{
    float uf = (float)u / 65536.0f;
    float vf = (float)v / 65536.0f;

    if (t->count_only)
        return t->ncoord++;
    return mesh_vertex(t->mesh, uf, vf, t->tri ? 1.0f - uf - vf : 0.0f, t->pid) - t->base;
}

static int d3d_patch_index(const struct D3DTess *t, int index)
{
    if (t->patched) {
        if (index >= t->ipc_outside_base) {
            if (index == t->ipc_outside_bad)
                return t->ipc_outside_repl;
            return index + t->ipc_outside_delta;
        }
        if (index == t->ipc_inside_bad)
            return t->ipc_inside_repl;
        return index + t->ipc_inside_delta;
    }
    if (t->patched2) {
        if (index == t->ipc2_corner_bad)
            return t->ipc2_corner_repl;
        if (index >= t->ipc2_base_invert)
            return t->ipc2_inv_end - index;
    }
    return index;
}

static void d3d_triangle(struct D3DTess *t, int i0, int i1, int i2)
{
    if (t->count_only) { t->nindex += 3; return; }
    i0 = (int)d3d_patch_index(t, i0) + (int)t->base;
    i1 = (int)d3d_patch_index(t, i1) + (int)t->base;
    i2 = (int)d3d_patch_index(t, i2) + (int)t->base;
    if (t->ccw)
        mesh_index(t->mesh, (unsigned)i0, (unsigned)i1, (unsigned)i2);
    else
        mesh_index(t->mesh, (unsigned)i0, (unsigned)i2, (unsigned)i1);
}

/* ---- ring stitching (identical to the reference) ---- */

static void d3d_line(struct D3DTess *t, int i0, int i1)
{
    if (t->count_only) { t->nindex += 2; return; }
    if (!mesh_reserve(t->mesh, 0, 2))
        return;
    t->mesh->indices[t->mesh->nindices++] =
        (unsigned)((int)d3d_patch_index(t, i0) + (int)t->base);
    t->mesh->indices[t->mesh->nindices++] =
        (unsigned)((int)d3d_patch_index(t, i1) + (int)t->base);
}

#define D3D_DIAG_INSIDE_TO_OUTSIDE              0
#define D3D_DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MID   1
#define D3D_DIAG_MIRRORED                       2

static void d3d_stitch_regular(struct D3DTess *t, bool trapezoid, int diagonals,
                               int base_index_offset, int num_inside_edge_points,
                               int inside_base, int outside_base)
{
    int inside = inside_base, outside = outside_base, p;

    if (trapezoid) {
        d3d_triangle(t, outside, outside + 1, inside);
        base_index_offset += 3;
        outside++;
    }
    switch (diagonals) {
    case D3D_DIAG_INSIDE_TO_OUTSIDE:
        for (p = 0; p < num_inside_edge_points - 1; p++) {
            d3d_triangle(t, inside, outside, outside + 1);
            d3d_triangle(t, inside, outside + 1, inside + 1);
            inside++; outside++;
        }
        break;
    case D3D_DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MID:
        for (p = 0; p < num_inside_edge_points / 2 - 1; p++) {
            d3d_triangle(t, outside, outside + 1, inside);
            d3d_triangle(t, inside, outside + 1, inside + 1);
            inside++; outside++;
        }
        d3d_triangle(t, outside, inside + 1, inside);
        d3d_triangle(t, outside, outside + 1, inside + 1);
        inside++; outside++;
        p += 2;
        for (; p < num_inside_edge_points; p++) {
            d3d_triangle(t, outside, outside + 1, inside);
            d3d_triangle(t, inside, outside + 1, inside + 1);
            inside++; outside++;
        }
        break;
    default: /* mirrored */
        for (p = 0; p < num_inside_edge_points / 2; p++) {
            d3d_triangle(t, outside, inside + 1, inside);
            d3d_triangle(t, outside, outside + 1, inside + 1);
            inside++; outside++;
        }
        for (; p < num_inside_edge_points - 1; p++) {
            d3d_triangle(t, inside, outside, outside + 1);
            d3d_triangle(t, inside, outside + 1, inside + 1);
            inside++; outside++;
        }
        break;
    }
    if (trapezoid) {
        d3d_triangle(t, outside, outside + 1, inside);
        base_index_offset += 3;
    }
}

static int d3d_loop_start(int n)
{
    if (n < 2) return 1;
    else if (n == 2) return 17;
    else if (n < 5) return 9;
    else if (n < 9) return 5;
    else if (n < 17) return 3;
    else return 2;
}

static int d3d_loop_end(int n)
{
    if (n < 2) return 0;
    else if (n < 4) return 17;
    else if (n < 8) return 25;
    else if (n < 16) return 29;
    else if (n < 32) return 31;
    else return 32;
}

/* where vertex i [0..33] ends up on the half edge at the maximum tessellation
 * amount, given the ruler-function split order (odd factors up to 65, even up
 * to 64) */
static unsigned d3d_final_point_position(unsigned x)
{
    unsigned shift, items, shifted, bias;

    if (x == 0) return 0;
    if (x == 1) return 0x20;
    if ((x & 1) == 0) shift = 1;
    else if ((x & 3) == 3) shift = 2;
    else if ((x & 7) == 5) shift = 3;
    else if (x != 17) shift = 4;
    else shift = 5;

    items = x | (0x20u << 16);
    shifted = items >> shift;
    bias = (shifted >> 16) < 0xfu ? (shifted >> 16) : 0xfu;
    return bias + (shifted & 0xffffu);
}

static void d3d_stitch_transition(struct D3DTess *t, int inside_base,
                                  int inside_num_half_points, bool inside_odd,
                                  int outside_base, int outside_num_half_points,
                                  bool outside_odd)
{
    int outside, inside, i, i_start, i_end;

    if (inside_odd) inside_num_half_points -= 1;
    if (outside_odd) outside_num_half_points -= 1;

    outside = outside_base;
    inside = inside_base;
    i_start = d3d_loop_start(inside_num_half_points) < d3d_loop_start(outside_num_half_points) ?
              d3d_loop_start(inside_num_half_points) : d3d_loop_start(outside_num_half_points);
    i_end = d3d_loop_end(inside_num_half_points > outside_num_half_points ?
                         inside_num_half_points : outside_num_half_points);

    if (0 < outside_num_half_points) {          /* advance outside */
        d3d_triangle(t, outside, outside + 1, inside);
        outside++;
    }
    for (i = i_start; i <= i_end; i++) {
        int bound = (int)d3d_final_point_position((unsigned)i);
        if (bound < inside_num_half_points) {   /* advance inside */
            d3d_triangle(t, inside, outside, inside + 1);
            inside++;
        }
        if (bound < outside_num_half_points) {  /* advance outside */
            d3d_triangle(t, outside, outside + 1, inside);
            outside++;
        }
    }
    if ((inside_odd != outside_odd) || inside_odd) {
        if (inside_odd == outside_odd) {        /* quad in the middle */
            d3d_triangle(t, inside, outside, inside + 1);
            d3d_triangle(t, inside + 1, outside, outside + 1);
            inside++; outside++;
        } else if (!inside_odd) {               /* triangle pointing inside */
            d3d_triangle(t, inside, outside, outside + 1);
            outside++;
        } else {                                /* triangle pointing outside */
            d3d_triangle(t, inside, outside, inside + 1);
            inside++;
        }
    }
    for (i = i_end; i >= i_start; i--) {
        int bound = (int)d3d_final_point_position((unsigned)i);
        if (bound < outside_num_half_points) {  /* advance outside */
            d3d_triangle(t, outside, outside + 1, inside);
            outside++;
        }
        if (bound < inside_num_half_points) {   /* advance inside */
            d3d_triangle(t, inside, outside, inside + 1);
            inside++;
        }
    }
    if (0 < outside_num_half_points) {
        d3d_triangle(t, outside, outside + 1, inside);
        outside++;
    }
}

/* ---- isolines: levels [density, detail], lines run along u ---- */

static void d3d_isoline_process(struct D3DTess *t, int partitioning, const float *factors)
{
    float density = factors[0], detail = factors[1];
    bool density_odd, detail_odd;
    struct D3DFactorCtx density_ctx, detail_ctx;
    D3D_FXP fxp_detail, fxp_density;
    int num_points_per_line, num_lines, num_points, line, point;

    if (!(density > 0.0f) || !(detail > 0.0f)) {
        t->ncoord = 0; t->nindex = 0;                     /* culled */
        return;
    }
    density = density < 1.0f ? 1.0f : (density > 64.0f ? 64.0f : density);
    detail = d3d_clamp_factor(detail, partitioning, 0.0f);

    if (partitioning == D3D_PARTITIONING_INTEGER)
        detail_odd = d3d_is_odd(detail);
    else
        detail_odd = (partitioning == D3D_PARTITIONING_FRACTIONAL_ODD);

    fxp_detail = d3d_float_to_fixed(detail);
    d3d_compute_factor_ctx(detail_odd, fxp_detail, &detail_ctx);
    num_points_per_line = d3d_num_points_for_factor(detail_odd, fxp_detail);

    density = ceilf(density);
    density_odd = d3d_is_odd(density);
    fxp_density = d3d_float_to_fixed(density);
    d3d_compute_factor_ctx(density_odd, fxp_density, &density_ctx);
    num_lines = d3d_num_points_for_factor(density_odd, fxp_density) - 1;

    num_points = num_points_per_line * num_lines;
    if (t->count_only) {
        t->ncoord = (unsigned)num_points;
        t->nindex = (unsigned)(num_lines * (num_points_per_line - 1) * 2);
        return;
    }

    for (line = 0; line < num_lines; line++) {
        D3D_FXP v = d3d_place_point_in_1d(&density_ctx, density_odd, line);
        for (point = 0; point < num_points_per_line; point++)
            d3d_point(t, d3d_place_point_in_1d(&detail_ctx, detail_odd, point), v);
    }
    /* connectivity: a strip per line, in vertex order */
    for (line = 0; line < num_lines; line++) {
        int base = line * num_points_per_line;
        for (point = 1; point < num_points_per_line; point++)
            d3d_line(t, base + point - 1, base + point);
    }
}

/* ---- triangles: factors [outer_u0, outer_v0, outer_w0, -, inner, -] ---- */

static void d3d_tri_process(struct D3DTess *t, int partitioning, const float *factors)
{
    float outer_f[3], inner_f;
    D3D_FXP outer[3], inner;
    bool outer_odd[3], inner_odd;
    struct D3DFactorCtx outer_ctx[3], inner_ctx;
    int num_outer[3], num_inner, inside_base, num_points, i;
    int num_rings;
    int point_offset, inside_offset, outside_base, ring, edge;

    for (i = 0; i < 3; i++) {
        outer_f[i] = factors[i];
        if (!(outer_f[i] > 0.0f)) {                /* culled (NaN passes) */
            t->ncoord = 0; t->nindex = 0;
            return;
        }
    }
    inner_f = factors[4];

    for (i = 0; i < 3; i++)
        outer_f[i] = d3d_clamp_factor(outer_f[i], partitioning, 0.0f);
    {
        float maxf = outer_f[0] > outer_f[1] ? outer_f[0] : outer_f[1];
        maxf = maxf > outer_f[2] ? maxf : outer_f[2];
        inner_f = d3d_clamp_factor(inner_f, partitioning, maxf);
    }

    if (partitioning == D3D_PARTITIONING_INTEGER) {
        for (i = 0; i < 3; i++)
            outer_odd[i] = d3d_is_odd(outer_f[i]);
        inner_odd = d3d_is_odd(inner_f) && (1.0f != inner_f);
    } else {
        bool odd = (partitioning == D3D_PARTITIONING_FRACTIONAL_ODD);
        for (i = 0; i < 3; i++)
            outer_odd[i] = odd;
        inner_odd = odd;
    }
    for (i = 0; i < 3; i++)
        outer[i] = d3d_float_to_fixed(outer_f[i]);
    inner = d3d_float_to_fixed(inner_f);

    /* minimum tessellation: a single triangle */
    if (partitioning != D3D_PARTITIONING_FRACTIONAL_EVEN &&
        inner == D3D_FXP_ONE && outer[0] == D3D_FXP_ONE &&
        outer[1] == D3D_FXP_ONE && outer[2] == D3D_FXP_ONE) {
        if (t->count_only) { t->ncoord = 3; t->nindex = 3; return; }
        /* V=1 (start of the Ueq0 edge VW), W=1 (start of Veq0 edge WU),
         * U=1 (start of the Weq0 edge UV) */
        d3d_point(t, 0, D3D_FXP_ONE);
        d3d_point(t, 0, 0);
        d3d_point(t, D3D_FXP_ONE, 0);
        d3d_triangle(t, 0, 1, 2);
        return;
    }

    for (i = 0; i < 3; i++)
        d3d_compute_factor_ctx(outer_odd[i], outer[i], &outer_ctx[i]);
    d3d_compute_factor_ctx(inner_odd, inner, &inner_ctx);

    num_points = 0;
    for (i = 0; i < 3; i++) {
        num_outer[i] = d3d_num_points_for_factor(outer_odd[i], outer[i]);
        num_points += num_outer[i];
    }
    num_points -= 3;

    num_inner = d3d_num_points_for_factor(inner_odd, inner);
    {
        int min_points = inner_odd ? 4 : 3;
        /* max() allows degenerate transition regions when the inner factor == 1 */
        num_inner = min_points > num_inner ? min_points : num_inner;
    }
    inside_base = num_points;
    {
        int interior_rings = (num_inner >> 1) - 1;
        int even = inner_odd ? 0 : 1;
        num_points += 3 * (interior_rings * (interior_rings + even)) + even;
    }

    if (t->count_only) {
        int outer_pts = num_outer[0] + num_outer[1] + num_outer[2];
        int rings18, num_indices;
        num_rings = (num_inner + 1) >> 1;
        rings18 = num_rings * 18;
        num_indices = ((rings18 - 27) * num_inner) + (3 * outer_pts) -
                      (rings18 * (num_rings - 1)) + (inner_odd ? 3 : 0);
        t->ncoord = (unsigned)num_points;
        t->nindex = (unsigned)num_indices;
        return;
    }

    /* exterior ring: clockwise starting from point V, i.e. edge0 = VW (u == 0) */
    point_offset = 0;
    for (edge = 0; edge < 3; edge++) {
        int odd = edge & 1;
        int end_point = num_outer[edge] - 1;   /* next edge starts with the last */
        int p;
        for (p = 0; p < end_point; p++, point_offset++) {
            /* edge0 VW has v decreasing -> reverse; edge1 WU has u increasing;
             * edge2 UV has u decreasing -> reverse */
            int q = odd ? p : end_point - p;
            D3D_FXP param = d3d_place_point_in_1d(&outer_ctx[edge], outer_odd[edge], q);
            d3d_point(t, edge == 0 ? 0 : param,
                         edge == 0 ? param : (edge == 2 ? D3D_FXP_ONE - param : 0));
        }
    }

    /* interior rings, clockwise spiralling in */
    num_rings = num_inner >> 1;
    for (ring = 1; ring < num_rings; ring++) {
        int start = ring, end = num_inner - 1 - start;
        int perp = start;
        D3D_FXP fxp_perp = d3d_place_point_in_1d(&inner_ctx, inner_odd, perp);
        fxp_perp *= D3D_FXP_TWO_THIRDS;
        fxp_perp = (fxp_perp + D3D_FXP_ONE_HALF) >> D3D_FXP_FRACTION_BITS;

        for (edge = 0; edge < 3; edge++) {
            int odd = edge & 1;
            int p;
            for (p = start; p < end; p++, point_offset++) {
                int q = odd ? p : end - (p - start);
                D3D_FXP param = d3d_place_point_in_1d(&inner_ctx, inner_odd, q);
                D3D_FXP tmp = param - (fxp_perp + 1) / 2;
                d3d_point(t, edge > 0 ? tmp : fxp_perp,
                             edge == 0 ? tmp : (edge == 1 ? fxp_perp : D3D_FXP_ONE - tmp - fxp_perp));
            }
        }
    }
    if (!inner_odd)
        d3d_point(t, D3D_FXP_ONE_THIRD, D3D_FXP_ONE_THIRD);   /* the centre */

    /* connectivity: concentric rings, one side at a time */
    num_rings = (num_inner + 1) >> 1;
    inside_offset = inside_base;
    outside_base = 0;
    for (ring = 1; ring < num_rings; ring++) {
        int num_inside_edge = num_inner - 2 * ring;
        int edge0_inside_base = inside_offset;
        int edge0_outside_base = outside_base;
        for (edge = 0; edge < 3; edge++) {
            int outside_points = (ring == 1) ? num_outer[edge] : (num_inside_edge + 2);
            int inside_base, obase;

            if (edge == 2) {
                /* wrap-around: patch the indices so the stitch sees two
                 * sequentially increasing rows */
                t->patched = true;
                t->ipc_inside_delta = inside_offset;
                t->ipc_inside_bad = num_inside_edge - 1;
                t->ipc_inside_repl = edge0_inside_base;
                t->ipc_outside_base = t->ipc_inside_bad + 1;
                t->ipc_outside_delta = outside_base - t->ipc_outside_base;
                t->ipc_outside_bad = t->ipc_outside_base + outside_points - 1;
                t->ipc_outside_repl = edge0_outside_base;
                inside_base = 0;
                obase = t->ipc_outside_base;
            } else {
                inside_base = inside_offset;
                obase = outside_base;
            }
            if (ring == 1)
                d3d_stitch_transition(t, inside_base, inner_ctx.num_half, inner_odd,
                                      obase, outer_ctx[edge].num_half, outer_odd[edge]);
            else
                d3d_stitch_regular(t, true, D3D_DIAG_MIRRORED, 0, num_inside_edge,
                                   inside_base, obase);
            t->patched = false;
            outside_base += outside_points - 1;
            inside_offset += num_inside_edge - 1;
        }
    }
    if (inner_odd) {                       /* the centre is a single triangle */
        d3d_triangle(t, outside_base, outside_base + 1, outside_base + 2);
    }
}

/* ---- quads: factors [outer_u0, outer_v0, outer_u1, outer_v1, inner_u, inner_v] ---- */

static void d3d_quad_process(struct D3DTess *t, int partitioning, const float *factors)
{
    float outer_f[4], inner_f[2], maxf;
    D3D_FXP outer[4], inner[2];
    bool outer_odd[4], inner_odd[2];
    struct D3DFactorCtx outer_ctx[4], inner_ctx[2];
    int num_outer[4], num_inner[2], inside_base, num_points, i;
    int num_rings, num_rows_center[2], ring, edge, point_offset;
    int inside_offset, outside_base, degenerate_ring[2];

    for (i = 0; i < 4; i++) {
        outer_f[i] = factors[i];
        if (!(outer_f[i] > 0.0f)) {                /* culled (NaN passes) */
            t->ncoord = 0; t->nindex = 0;
            return;
        }
    }
    inner_f[0] = factors[4];
    inner_f[1] = factors[5];

    for (i = 0; i < 4; i++)
        outer_f[i] = d3d_clamp_factor(outer_f[i], partitioning, 0.0f);
    maxf = outer_f[0];
    for (i = 1; i < 4; i++) maxf = maxf > outer_f[i] ? maxf : outer_f[i];
    for (i = 0; i < 2; i++) maxf = maxf > inner_f[i] ? maxf : inner_f[i];
    inner_f[0] = d3d_clamp_factor(inner_f[0], partitioning, maxf);
    inner_f[1] = d3d_clamp_factor(inner_f[1], partitioning, maxf);

    if (partitioning == D3D_PARTITIONING_INTEGER) {
        for (i = 0; i < 4; i++)
            outer_odd[i] = d3d_is_odd(outer_f[i]);
        for (i = 0; i < 2; i++)
            inner_odd[i] = d3d_is_odd(inner_f[i]) && (1.0f != inner_f[i]);
    } else {
        bool odd = (partitioning == D3D_PARTITIONING_FRACTIONAL_ODD);
        for (i = 0; i < 4; i++)
            outer_odd[i] = odd;
        inner_odd[0] = inner_odd[1] = odd;
    }
    for (i = 0; i < 4; i++)
        outer[i] = d3d_float_to_fixed(outer_f[i]);
    for (i = 0; i < 2; i++)
        inner[i] = d3d_float_to_fixed(inner_f[i]);

    /* minimum tessellation: a single quad */
    if (partitioning != D3D_PARTITIONING_FRACTIONAL_EVEN &&
        inner[0] == D3D_FXP_ONE && inner[1] == D3D_FXP_ONE &&
        outer[0] == D3D_FXP_ONE && outer[1] == D3D_FXP_ONE &&
        outer[2] == D3D_FXP_ONE && outer[3] == D3D_FXP_ONE) {
        if (t->count_only) { t->ncoord = 4; t->nindex = 6; return; }
        d3d_point(t, 0, 0);
        d3d_point(t, D3D_FXP_ONE, 0);
        d3d_point(t, D3D_FXP_ONE, D3D_FXP_ONE);
        d3d_point(t, 0, D3D_FXP_ONE);
        d3d_triangle(t, 0, 1, 3);
        d3d_triangle(t, 1, 2, 3);
        return;
    }

    for (i = 0; i < 4; i++)
        d3d_compute_factor_ctx(outer_odd[i], outer[i], &outer_ctx[i]);
    for (i = 0; i < 2; i++)
        d3d_compute_factor_ctx(inner_odd[i], inner[i], &inner_ctx[i]);

    num_points = 0;
    for (i = 0; i < 4; i++) {
        num_outer[i] = d3d_num_points_for_factor(outer_odd[i], outer[i]);
        num_points += num_outer[i];
    }
    num_points -= 4;

    for (i = 0; i < 2; i++) {
        int min_points;
        num_inner[i] = d3d_num_points_for_factor(inner_odd[i], inner[i]);
        min_points = inner_odd[i] ? 4 : 3;
        num_inner[i] = min_points > num_inner[i] ? min_points : num_inner[i];
    }
    inside_base = num_points;
    num_points += (num_inner[0] - 2) * (num_inner[1] - 2);

    if (t->count_only) {
        int outer_pts = num_outer[0] + num_outer[1] + num_outer[2] + num_outer[3];
        int inner_pts = num_inner[0] + num_inner[1];
        int num_indices;
        num_rows_center[0] = (num_inner[0] + 1) >> 1;
        num_rows_center[1] = (num_inner[1] + 1) >> 1;
        num_rings = num_rows_center[0] < num_rows_center[1] ? num_rows_center[0] : num_rows_center[1];
        num_indices = (outer_pts * 3) + (12 * num_rings * inner_pts) -
                      (inner_pts * 18) - (24 * num_rings * (num_rings - 1));
        {
            int M, m;                       /* major / minor axis */
            M = num_inner[0] > num_inner[1] ? 0 : 1;
            m = 1 - M;
            if (inner_odd[m]) {
                num_indices += 12 * ((num_inner[M] >> 1) - (num_inner[m] >> 1));
                num_indices += inner_odd[M] ? 6 : 12;
            }
        }
        t->ncoord = (unsigned)num_points;
        t->nindex = (unsigned)num_indices;
        return;
    }

    /* exterior ring: clockwise from top-left (u == 0, v == 1) */
    point_offset = 0;
    for (edge = 0; edge < 4; edge++) {
        int odd = edge & 1;
        int end_point = num_outer[edge] - 1;
        int p;
        for (p = 0; p < end_point; p++, point_offset++) {
            int q = (edge == 1 || edge == 2) ? p : end_point - p;
            D3D_FXP param = d3d_place_point_in_1d(&outer_ctx[edge], outer_odd[edge], q);
            d3d_point(t, odd ? param : (edge == 2 ? D3D_FXP_ONE : 0),
                         odd ? (edge == 3 ? D3D_FXP_ONE : 0) : param);
        }
    }

    /* interior rings, clockwise from (u == 0, v == 1) spiralling in */
    {
        int min_inner = num_inner[0] < num_inner[1] ? num_inner[0] : num_inner[1];
        num_rings = min_inner >> 1;
    }
    for (ring = 1; ring < num_rings; ring++) {
        int start = ring;
        int end[2];
        end[0] = num_inner[0] - 1 - start;
        end[1] = num_inner[1] - 1 - start;
        for (edge = 0; edge < 4; edge++) {
            int odd0 = edge & 1, odd1 = (edge + 1) & 1;
            int perp = (edge < 2) ? start : end[odd0];
            D3D_FXP fxp_perp = d3d_place_point_in_1d(&inner_ctx[odd0], inner_odd[odd0], perp);
            int p;
            for (p = start; p < end[odd1]; p++, point_offset++) {
                int q = (edge == 1 || edge == 2) ? p : end[odd1] - (p - start);
                D3D_FXP param = d3d_place_point_in_1d(&inner_ctx[odd1], inner_odd[odd1], q);
                d3d_point(t, odd1 ? fxp_perp : param, odd1 ? param : fxp_perp);
            }
        }
    }
    /* for even tessellation the innermost "ring" degenerates to a row of points */
    if (num_inner[0] > num_inner[1] && !inner_odd[1]) {
        int start = num_rings, end = num_inner[0] - 1 - start, p;
        for (p = start; p <= end; p++)
            d3d_point(t, d3d_place_point_in_1d(&inner_ctx[0], inner_odd[0], p), D3D_FXP_ONE_HALF);
    } else if (num_inner[1] >= num_inner[0] && !inner_odd[0]) {
        int start = num_rings, end = num_inner[1] - 1 - start, p;
        for (p = end; p >= start; p--)
            d3d_point(t, D3D_FXP_ONE_HALF, d3d_place_point_in_1d(&inner_ctx[1], inner_odd[1], p));
    }

    /* connectivity */
    num_rows_center[0] = (num_inner[0] + 1) >> 1;
    num_rows_center[1] = (num_inner[1] + 1) >> 1;
    num_rings = num_rows_center[0] < num_rows_center[1] ? num_rows_center[0] : num_rows_center[1];
    degenerate_ring[0] = !inner_odd[1] ? num_rows_center[1] - 1 : -1;
    degenerate_ring[1] = !inner_odd[0] ? num_rows_center[0] - 1 : -1;

    inside_offset = inside_base;
    outside_base = 0;
    for (ring = 1; ring < num_rings; ring++) {
        int num_inside_edge[2];
        int edge0_inside_base = inside_offset;
        int edge0_outside_base = outside_base;

        num_inside_edge[0] = num_inner[0] - 2 * ring;
        num_inside_edge[1] = num_inner[1] - 2 * ring;

        for (edge = 0; edge < 4; edge++) {
            int odd = (edge + 1) & 1;
            int num_outside = num_outer[edge];
            int inside_base, obase;

            if (edge == 3) {
                if (ring == degenerate_ring[odd]) {
                    t->patched2 = true;
                    t->ipc2_base_invert = inside_offset + 1;
                    t->ipc2_corner_bad = outside_base + num_outside - 1;
                    t->ipc2_corner_repl = edge0_outside_base;
                    t->ipc2_inv_end = (t->ipc2_base_invert << 1) - 1;
                    inside_base = t->ipc2_base_invert;
                    obase = outside_base;
                } else {
                    t->patched = true;
                    t->ipc_inside_delta = inside_offset;
                    t->ipc_inside_bad = num_inside_edge[odd] - 1;
                    t->ipc_inside_repl = edge0_inside_base;
                    t->ipc_outside_base = t->ipc_inside_bad + 1;
                    t->ipc_outside_delta = outside_base - t->ipc_outside_base;
                    t->ipc_outside_bad = t->ipc_outside_base + num_outside - 1;
                    t->ipc_outside_repl = edge0_outside_base;
                    inside_base = 0;
                    obase = t->ipc_outside_base;
                }
            } else if (edge == 2 && ring == degenerate_ring[odd]) {
                t->patched2 = true;
                t->ipc2_base_invert = inside_offset;
                t->ipc2_corner_bad = -1;
                t->ipc2_corner_repl = -1;
                t->ipc2_inv_end = t->ipc2_base_invert << 1;
                inside_base = t->ipc2_base_invert;
                obase = outside_base;
            } else {
                inside_base = inside_offset;
                obase = outside_base;
            }
            if (ring == 1)
                d3d_stitch_transition(t, inside_base, inner_ctx[odd].num_half, inner_odd[odd],
                                      obase, outer_ctx[edge].num_half, outer_odd[edge]);
            else
                d3d_stitch_regular(t, true, D3D_DIAG_MIRRORED, 0, num_inside_edge[odd],
                                   inside_base, obase);
            t->patched = false;
            t->patched2 = false;
            outside_base += num_outside - 1;
            if (edge == 2 && ring == degenerate_ring[odd])
                inside_offset -= num_inside_edge[odd] - 1;
            else
                inside_offset += num_inside_edge[odd] - 1;
            num_outer[edge] = num_inside_edge[odd];
        }
    }
    /* the centre: a row of quads if odd */
    if (num_inner[0] > num_inner[1] && inner_odd[1]) {
        int strip_quads = (((num_inner[0] >> 1) - (num_inner[1] >> 1)) << 1) + (inner_odd[0] ? 1 : 2);
        t->patched2 = true;
        t->ipc2_base_invert = outside_base + strip_quads + 2;
        t->ipc2_corner_bad = t->ipc2_base_invert;
        t->ipc2_corner_repl = outside_base;
        t->ipc2_inv_end = t->ipc2_base_invert + t->ipc2_base_invert + strip_quads;
        d3d_stitch_regular(t, false, D3D_DIAG_INSIDE_TO_OUTSIDE, 0, strip_quads + 1,
                           t->ipc2_base_invert, outside_base + 1);
        t->patched2 = false;
    } else if (num_inner[1] >= num_inner[0] && inner_odd[0]) {
        int strip_quads = (((num_inner[1] >> 1) - (num_inner[0] >> 1)) << 1) + (inner_odd[1] ? 1 : 2);
        int diag = inner_odd[1] ? D3D_DIAG_INSIDE_TO_OUTSIDE_EXCEPT_MID : D3D_DIAG_INSIDE_TO_OUTSIDE;
        t->patched2 = true;
        t->ipc2_base_invert = outside_base + strip_quads + 1;
        t->ipc2_corner_bad = -1;
        t->ipc2_inv_end = t->ipc2_base_invert + t->ipc2_base_invert + strip_quads;
        d3d_stitch_regular(t, false, diag, 0, strip_quads + 1, t->ipc2_base_invert, outside_base);
        t->patched2 = false;
    }
}

static bool tess_generate(GLMContext ctx, MGLTessProgram *tp, const float *factors,
                          unsigned patches)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    struct TessMesh mesh = { NULL, 0, 0, NULL, 0, 0 };

    tess_reset_scratch(S);
    if (!patches)
        return true;

    for (unsigned p = 0; p < patches; p++)
    {
        const float *f = factors + p * 6;
        struct D3DTess t;

        memset(&t, 0, sizeof t);
        t.mesh = &mesh;
        t.pid = (float)p;
        t.ccw = tp->vertex_order_ccw != 0;
        t.tri = (tp->domain == MGL_TESS_DOMAIN_TRIANGLES);
        t.base = mesh.ncoords;
        /* spacing and partitioning use the same values (equal=integer, even, odd) */
        switch (tp->domain)
        {
            case MGL_TESS_DOMAIN_TRIANGLES:
                d3d_tri_process(&t, tp->spacing, f);
                break;
            case MGL_TESS_DOMAIN_ISOLINES:
                d3d_isoline_process(&t, tp->spacing, f);
                break;
            default:
                d3d_quad_process(&t, tp->spacing, f);
                break;
        }
    }

    if (tp->point_mode && mesh.ncoords)
    {
        /* point mode: one point per tessellated vertex */
        free(mesh.indices);
        mesh.indices = NULL;
        mesh.nindices = 0;
        mesh.index_cap = 0;
        if (!mesh_reserve(&mesh, 0, mesh.ncoords))
        {
            free(mesh.coords);
            return false;
        }
        for (unsigned i = 0; i < mesh.ncoords; i++)
            mesh.indices[mesh.nindices++] = i;
    }

    free(S->coord_buf);
    free(S->index_buf);
    S->coord_buf = mesh.coords;
    S->index_buf = mesh.indices;
    S->tess_vertex_count = (GLsizei)mesh.ncoords;
    S->tess_index_count = (GLsizei)mesh.nindices;
    S->patch_count = (GLsizei)patches;

    /* query bookkeeping (GL_PRIMITIVES_GENERATED etc. see the exact tessellated output) */
    if (mesh.nindices)
    {
        GLuint prims;
        if (tp->point_mode || tp->domain == MGL_TESS_DOMAIN_ISOLINES)
            prims = (GLuint)(tp->point_mode ? mesh.ncoords : mesh.nindices / 2);
        else
            prims = (GLuint)(mesh.nindices / 3);
        mgl_tess_note_draw(ctx, patches, (GLuint)mesh.ncoords, prims);
    }
    return true;
}

/* producer/consumer patch-member match: the canonical key (varying location) is
 * the primary match, but when BOTH sides are flattened per-element members —
 * tess r48 dissolves a per-patch block ARRAY into "name_0..N" plain typed
 * members — the element indices must match too, or every consumer element copies
 * producer element 0 (patch_block_array_pipeline: targetInstance_target_1/2 read
 * back targetInstance_target_0).  A source kept as one spvUnsafeArray member
 * (flat_dims 0) still lets any consumer element match; the caller's stride
 * compensation steps to the element then. */
static bool tess_patch_member_matches(const MGLTessPatchMember *p, const MGLTessPatchMember *m)
{
    if (p->key != m->key) return false;
    if (!p->flat_dims || !m->flat_dims) return true;
    if (p->array_idx != m->array_idx) return false;
    if (p->flat_dims > 1 && m->flat_dims > 1 && p->array_idx2 != m->array_idx2) return false;
    return true;
}

/* build the per-patch data blob consumed by the rewritten TES (TES patchIn layout):
 * - members written by the TCS are copied from the read-back patch output struct
 * - the tessellation level members are filled from the factors
 * - anything else stays zero (undefined per GL when no TCS writes it)
 *
 * CPU form: used by the MGL_TESS_CPU reference path and the GPU-tessellator
 * fallback only — the GPU paths assemble the blob with the mgl_tess_blob kernel
 * (see MGLTessKernels.h) from the same map (mgl_tess_build_blob_map). */
static unsigned tess_cpu_blob_builds;   /* diagnostic: CPU blob builds since load */
static unsigned tess_cpu_generate_runs; /* diagnostic: CPU tessellator runs since load */
static unsigned tess_scan_waits;        /* diagnostic: GPU index-scan waits since load */
static unsigned tess_cpu_index_reads;   /* diagnostic: CPU element-buffer reads since load */
static unsigned tess_restart_checks;    /* diagnostic: CPU restart-presence scans since load */

unsigned mgl_tess_cpu_blob_builds(void)   { return tess_cpu_blob_builds; }
unsigned mgl_tess_cpu_generate_runs(void) { return tess_cpu_generate_runs; }
unsigned mgl_tess_scan_waits(void)        { return tess_scan_waits; }
unsigned mgl_tess_cpu_index_reads(void)   { return tess_cpu_index_reads; }
unsigned mgl_tess_restart_checks(void)    { return tess_restart_checks; }
void mgl_tess_note_scan_wait(void)      { tess_scan_waits++; }
void mgl_tess_note_cpu_index_read(void) { tess_cpu_index_reads++; }
void mgl_tess_note_restart_check(void)  { tess_restart_checks++; }

static void tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                                  const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                                  const float *factors, unsigned patches)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    size_t need = (size_t)tp->tes_patch_in_size * patches;

    tess_cpu_blob_builds++;
    if (!need)
        return;
    if (S->patch_cap < need)
    {
        unsigned char *p = (unsigned char *)realloc(S->patch_data, need);
        if (!p)
            return;
        S->patch_data = p;
        S->patch_cap = need;
    }
    memset(S->patch_data, 0, need);

    for (unsigned pi = 0; pi < patches; pi++)
    {
        unsigned char *dst = S->patch_data + (size_t)pi * tp->tes_patch_in_size;
        const unsigned char *src = tcs_patch_out ? tcs_patch_out + (size_t)pi * tcs_patch_stride : NULL;
        const float *f = factors ? factors + pi * 6 : NULL;

        for (int i = 0; i < tp->num_tes_patch_in; i++)
        {
            const MGLTessPatchMember *m = &tp->tes_patch_in[i];

            if (!strcmp(m->name, "gl_TessLevel") || !strcmp(m->name, "gl_TessLevelOuter"))
            {
                if (!f) continue;
                float lv[4] = { f[0], f[1], f[2], f[3] };
                if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
                    lv[3] = f[4];                    /* triangles pack inner[0] into .w */
                memcpy(dst + m->offset, lv, sizeof lv);
            }
            else if (!strcmp(m->name, "gl_TessLevelInner"))
            {
                if (!f) continue;
                float lv[2] = { f[4], f[5] };
                memcpy(dst + m->offset, lv, sizeof lv);
            }
            else if (src)
            {
                /* copy from the TCS patch output by varying location (canonical key).
                 * key 0 is a REAL key (GL location 0 — dEQP's primitive_discard
                 * TCS writes 'patch out vec2 in_te_positionScale' at location 0):
                 * dropping it left the blob member zero, and the CPU reference
                 * draw rasterized garbage positions.  The GPU blob map
                 * (mgl_tess_build_blob_map) matches the same keys without the
                 * exclusion and was always correct. */
                for (int j = 0; j < tp->num_tcs_patch_out; j++)
                {
                    if (tess_patch_member_matches(&tp->tcs_patch_out[j], m))
                    {
                        size_t so = tp->tcs_patch_out[j].offset;

                        /* flattened element ("name_N", array_idx N) of a canonical
                         * spvUnsafeArray patch output: advance the source by the element
                         * stride (the GPU blob map needed the same fix); a struct-array
                         * field element steps a second dimension inside the field */
                        if (m->flat_dims && strstr(tp->tcs_patch_out[j].type, "spvUnsafeArray"))
                        {
                            size_t stride = msl_array_stride(tp->tcs_patch_out[j].type);

                            if (stride)
                                so += m->array_idx * stride;
                            if (stride && m->array_idx2)
                            {
                                char inner[96];
                                long n2 = 0;

                                if (msl_array_inner(tp->tcs_patch_out[j].type, inner, sizeof inner, &n2) &&
                                    msl_array_stride(inner))
                                    so += m->array_idx2 * msl_array_stride(inner);
                                if (m->array_idx3)
                                {
                                    char inner2[96];
                                    long n3 = 0;

                                    if (msl_array_inner(inner, inner2, sizeof inner2, &n3) &&
                                        msl_array_stride(inner2))
                                        so += m->array_idx3 * msl_array_stride(inner2);
                                }
                            }
                        }
                        memcpy(dst + m->offset, src + so, m->size);
                        break;
                    }
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ per-program state */

static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out,
                                   char *struct_name, size_t struct_len);
MGLTessProgram *mgl_tess_program(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return NULL;
    for (tp = S->programs[(((uintptr_t)program) >> 4) % MGL_TESS_PROGRAM_HASH]; tp; tp = tp->next)
        if (tp->program == program)
            return tp;
    return NULL;
}

static void tess_free_program(GLMContext ctx, MGLTessProgram *tp)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram **head = &S->programs[(((uintptr_t)tp->program) >> 4) % MGL_TESS_PROGRAM_HASH];

    while (*head && *head != tp)
        head = &(*head)->next;
    if (*head)
        *head = tp->next;
    /* a derived pipeline pair dies with its base record */
    if (tp->pair)
    {
        tess_free_pair(ctx, tp->pair);
        tp->pair = NULL;
    }
    /* release the bridged Metal objects of the compile-once variants */
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
    {
        void **owned[8] = { &tp->capture_function, &tp->capture_library, &tp->capture_pso,
                            &tp->plain_function, &tp->tcs_function, &tp->tcs_library,
                            &tp->vs_function, &tp->vs_library };
        void *extra[2] = { tp->plain_library, NULL };

        for (int i = 0; i < 8; i++)
            if (*owned[i])
            {
                ctx->mtl_funcs.ext_program->release_obj(ctx, *owned[i]);
                *owned[i] = NULL;
            }
        if (extra[0])
            ctx->mtl_funcs.ext_program->release_obj(ctx, extra[0]);
    }
    free(tp->plain_msl);
    free(tp);
}

/* drop a program's tess RECORD (and records derived from it): used both on
 * program deletion and on a successful relink that no longer carries a TES
 * (the record's layouts must not survive the new executable).  Deliberately
 * leaves the vs_kernel cache alone: the program is very much alive at the
 * post-link call site and its cached separable vertex kernel must SURVIVE a
 * relink — the kernel is compiled during the link itself (compile_stage), so
 * purging it here deleted the entry the link just made and every later
 * pipeline draw of a separable vertex stage failed with "tess VS stage
 * function is not a kernel" (KHR-GL46.separable_programs_tf.geometry_active);
 * a relink of the SAME program refreshes the kernel there (mgl_tess_cache_vs_kernel
 * replaces a changed MSL). */
static void tess_drop_program_record(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return;
    while ((tp = mgl_tess_program(ctx, program)) != NULL)
        tess_free_program(ctx, tp);
    /* pairs DERIVED from this program (it played the tc-only side of a pipeline
     * pairing): drop them, the tc program they were canonicalized against is gone */
    for (unsigned h = 0; h < MGL_TESS_PROGRAM_HASH; h++)
        for (tp = S->programs[h]; tp; tp = tp->next)
            if (tp->pair && tp->pair_program == program)
            {
                tess_free_pair(ctx, tp->pair);
                tp->pair = NULL;
                tp->pair_program = NULL;
                tp->pair_key = NULL;
            }
}

void mgl_ext_delete_program_tess(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    if (!S)
        return;
    tess_drop_program_record(ctx, program);
    /* drop the program's cached separable vertex kernel (name reuse must not find it) */
    for (unsigned slot = 0; slot < S->vs_kernel_n; )
        if (S->vs_kernel[slot].name == program->name)
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: sep-vs cache entry dropped (program %u deleted)\n",
                        program->name);
            if (S->vs_kernel[slot].library && ctx->mtl_funcs.ext_program &&
                ctx->mtl_funcs.ext_program->release_obj)
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[slot].library);
            if (S->vs_kernel[slot].function && ctx->mtl_funcs.ext_program &&
                ctx->mtl_funcs.ext_program->release_obj)
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[slot].function);
            free(S->vs_kernel[slot].msl);
            free(S->vs_kernel[slot].words);
            /* compact: deleted programs must not waste cache capacity (a CTS case
             * links hundreds of short-lived single-stage programs) */
            if (slot + 1 < S->vs_kernel_n)
                S->vs_kernel[slot] = S->vs_kernel[S->vs_kernel_n - 1];
            memset(&S->vs_kernel[--S->vs_kernel_n], 0, sizeof S->vs_kernel[0]);
        }
        else
            slot++;
}

/* ---------------- GL 4.6 §17.3.4: late per-fragment tests ----------------
 *
 * "When there is no active program ... or the active program was linked with
 * early fragment tests disabled, these [depth/stencil per-fragment] operations
 * are performed only after fragment program execution" (ARB_shader_image_load_
 * store.txt §"Interactions with early fragment tests").  Apple GPUs always run
 * the depth test before the fragment shader (HSR), so a GL fragment shader that
 * writes memory and does NOT declare layout(early_fragment_tests) still gets
 * culled before its imageStore/SSBO side effects run — verified on M4 Pro
 * (.agent/probe_latedepth.m):
 *
 *   [[depth(any)]] output + guarded discard_fragment()  ->  shader runs, test late   ✓
 *   [[depth(any)]] output + unconditional discard       ->  culled before the shader ✗
 *   no depth output (any discard)                       ->  culled before the shader ✗
 *
 * The fix (all GPU-side, no readback): when a linked fragment stage writes to
 * memory and has no EarlyFragmentTests mode, patch its MSL to
 *  (a) add a `float [[depth(any)]]` output written from [[position]].z unless
 *      the shader already writes gl_FragDepth — that makes Metal defer the test
 *      until after the fragment function — and
 *  (b) guard every `discard_fragment();` with `if (as_type<uint>(pos.z) !=
 *      0xffffffffu)` (opaque to the compiler, always true at runtime: a valid
 *      window-space z never has that bit pattern).  An unguarded discard makes
 *      Metal classify the function as "kill-only late state" and revert to the
 *      early test.  GL semantics are unchanged: the guard always fires exactly
 *      where the user's discard was.
 * Shaders with side effects but no discard are covered by (a) alone (probe:
 * depth=1 rows above with no discard also ran); gl_FragDepth writers already
 * carry the output, so they get (b) only. */
#define MGL_LFT_MARKER "/* mgl-late-fragment-tests */"

/* SPIR-V opcodes the scanner below needs (unified1 grammar) */
enum {
    SPV_OP_TYPE_POINTER        = 32,
    SPV_OP_FUNCTION_PARAMETER  = 55,
    SPV_OP_VARIABLE            = 59,
    SPV_OP_IMAGE_TEXEL_POINTER = 60,
    SPV_OP_STORE               = 62,
    SPV_OP_COPY_MEMORY         = 63,
    SPV_OP_COPY_MEMORY_SIZED   = 64,
    SPV_OP_ACCESS_CHAIN        = 65,
    SPV_OP_IN_BOUNDS_AC        = 66,
    SPV_OP_PTR_ACCESS_CHAIN    = 67,
    SPV_OP_IMAGE_WRITE         = 99,
};

/* atomics that write (everything except OpAtomicLoad=227); OpAtomicStore=228
 * has no result id, the rest take their pointer at operand 1 */
static bool spv_op_is_writing_atomic(unsigned op)
{
    return (op >= 228 && op <= 242) ||
           op == 318 /*FlagTestAndSet*/ || op == 319 /*FlagClear*/ ||
           op == 5614 || op == 5615 || op == 6035;   /* float atomics */
}

/* does this module (one entry point's SPIR-V) write to memory through images,
 * storage buffers or atomic counters?  Walks the flat word stream: pointer ids
 * rooted in Uniform/StorageBuffer/Image storage are followed through access
 * chains to a fixpoint; OpStore/OpCopyMemory into them, any OpImageWrite and
 * every writing atomic count as side effects. */
static bool spv_module_writes_memory(const unsigned int *ir, size_t size)
{
    unsigned int max_id;
    unsigned char *is_gptr, *is_texel;
    size_t n_inst;
    bool changed, writes = false;
    const unsigned int *p, *end;

    if (!ir || size < 20) return false;
    max_id = ir[3];
    if (max_id < 2) return false;
    is_gptr  = (unsigned char *)calloc(max_id + 1, 1);
    is_texel = (unsigned char *)calloc(max_id + 1, 1);
    if (!is_gptr || !is_texel) { free(is_gptr); free(is_texel); return false; }

    p = ir + 5;
    end = ir + size;                /* size is in 32-bit words */

    /* pass 1: global pointer-typed types, global variables, texel pointers,
     * global pointer function parameters (helper functions receive them).
     * Storage classes: Uniform=2, AtomicCounter=10, Image=11, StorageBuffer=12 */
#define LFT_GLOBAL_SC(sc) ((sc) == 2 || (sc) == 10 || (sc) == 11 || (sc) == 12)
    for (n_inst = 0; p < end; ) {
        unsigned int len = *p >> 16, op = *p & 0xffff;
        if (!len) break;
        if (op == SPV_OP_TYPE_POINTER) {
            unsigned int sc = p[3];              /* operand 1: storage class */
            if (LFT_GLOBAL_SC(sc))
                is_gptr[p[1]] = 1;               /* result type is such a pointer */
        } else if (op == SPV_OP_VARIABLE) {
            unsigned int sc = p[3];              /* operand 1: storage class */
            if (LFT_GLOBAL_SC(sc))
                is_gptr[p[2]] = 1;               /* the variable id itself */
        } else if (op == SPV_OP_FUNCTION_PARAMETER) {
            if (p[1] <= max_id && is_gptr[p[1]]) /* result type is a global pointer */
                is_gptr[p[2]] = 1;
        } else if (op == SPV_OP_IMAGE_TEXEL_POINTER) {
            is_texel[p[2]] = 1;
        }
        p += len;
        n_inst++;
    }
#undef LFT_GLOBAL_SC

    /* pass 2: propagate through access chains (base is operand 3 for the
     * access-chain family) */
    do {
        changed = false;
        p = ir + 5;
        for (n_inst = 0; p < end; ) {
            unsigned int len = *p >> 16, op = *p & 0xffff;
            if (!len) break;
            if ((op == SPV_OP_ACCESS_CHAIN || op == SPV_OP_IN_BOUNDS_AC ||
                 op == SPV_OP_PTR_ACCESS_CHAIN) &&
                p[3] <= max_id && is_gptr[p[3]] && !is_gptr[p[2]]) {
                is_gptr[p[2]] = 1;
                changed = true;
            }
            p += len;
            n_inst++;
        }
    } while (changed);

    /* pass 3: the actual stores */
    p = ir + 5;
    for (n_inst = 0; p < end && !writes; ) {
        unsigned int len = *p >> 16, op = *p & 0xffff;
        if (!len) break;
        if (op == SPV_OP_IMAGE_WRITE)
            writes = true;
        else if (op == SPV_OP_STORE || op == SPV_OP_COPY_MEMORY || op == SPV_OP_COPY_MEMORY_SIZED) {
            unsigned int ptr = p[1];             /* pointer is operand 1 */
            if (ptr <= max_id && (is_gptr[ptr] || is_texel[ptr]))
                writes = true;
        } else if (spv_op_is_writing_atomic(op)) {
            unsigned int ptr = (op == 228) ? p[1] : p[3];
            if (ptr <= max_id && (is_gptr[ptr] || is_texel[ptr]))
                writes = true;
        }
        p += len;
        n_inst++;
    }

    free(is_gptr);
    free(is_texel);
    return writes;
}

/* find `fragment <ret> <entry>(` and return the index just past '(' (params
 * start), filling ret/retlen with the return type token.  The entry can also be
 * preceded by attributes (SPIRV-Cross puts none there for fragments we patch,
 * the early-fragment-test form is rejected before). */
static long mgl_lft_find_entry(const char *msl, const char *entry, const char **ret, size_t *retlen)
{
    char pat[256];
    const char *p;

    snprintf(pat, sizeof pat, "fragment ");
    p = strstr(msl, pat);
    while (p) {
        const char *e = p + strlen(pat);
        const char *s = e;
        /* return type token */
        while (*e && *e != '(' && !isspace((unsigned char)*e)) e++;
        if (*e == '(' || isspace((unsigned char)*e)) {
            size_t rl = (size_t)(e - s);
            const char *n = e;
            while (*n == ' ' || *n == '\n' || *n == '\t') n++;
            if (!strncmp(n, entry, strlen(entry)) && n[strlen(entry)] == '(') {
                if (ret) { *ret = s; *retlen = rl; }
                return (long)(n + strlen(entry) + 1 - msl);
            }
        }
        p = strstr(e, pat);
    }
    return -1;
}

/* the MSL text patch.  Returns a malloc'ed new MSL or NULL if nothing to do. */
static char *mgl_late_fragment_tests_msl(const char *msl, const char *entry,
                                         bool needs_depth_out)
{
    const char *ret;
    size_t retlen;
    long params_at, close_paren, body_at, guard_to;
    char posname[64];
    bool have_pos = false, is_void, have_depth, add_depth;
    char *out;
    size_t out_len = 0, cap;
    const char *body, *p, *seg, *tail;
    size_t body_len;

    if (!msl || !entry) return NULL;
    if (strstr(msl, MGL_LFT_MARKER)) return NULL;                    /* idempotent */

    params_at = mgl_lft_find_entry(msl, entry, &ret, &retlen);
    if (params_at < 0) return NULL;

    {   /* parameter list / body scope: the closing paren of the parameter list
         * is at paren depth 0 — attributes such as [[texture(0)]] nest parens */
        const char *cp = NULL;
        const char *bo;
        int depth = 0;
        const char *q;
        for (q = msl + params_at; *q; q++) {
            if (*q == '(') depth++;
            else if (*q == ')') {
                if (depth == 0) { cp = q; break; }
                depth--;
            }
        }
        if (!cp) return NULL;
        close_paren = (long)(cp - msl);
        bo = strchr(cp, '{');
        if (!bo) return NULL;
        body_at = (long)(bo - msl);
    }
    {   /* entry end: matching brace */
        int depth = 0;
        const char *q;
        for (q = msl + body_at; *q; q++) {
            if (*q == '{') depth++;
            else if (*q == '}') { if (--depth == 0) break; }
        }
        if (*q != '}') return NULL;
        guard_to = (long)(q - msl) + 1;
    }

    /* an existing [[position]] argument? */
    {
        const char *pp = msl + params_at;
        const char *pl = msl + close_paren;
        const char *hit;
        while ((hit = strstr(pp, "[[position]]")) && hit < pl) {
            const char *b = hit, *n;
            while (b > pp && (b[-1] == ' ' || b[-1] == '\n' || b[-1] == '\t')) b--;
            n = b;
            while (n > pp && (isalnum((unsigned char)n[-1]) || n[-1] == '_')) n--;
            if (n < b && (size_t)(b - n) < sizeof posname) {
                memcpy(posname, n, (size_t)(b - n));
                posname[b - n] = 0;
                have_pos = true;
            }
            pp = hit + 1;
        }
    }

    is_void = (retlen == 4 && !strncmp(ret, "void", 4));
    have_depth = strstr(msl, "[[depth") != NULL;   /* gl_FragDepth writer */
    add_depth = needs_depth_out && !have_depth;

    if (!add_depth && !strstr(msl + body_at, "discard_fragment")) return NULL;

    cap = strlen(msl) + 8192;
    out = (char *)malloc(cap);
    if (!out) return NULL;

#define LFT_APPEND(...) do { out_len += (size_t)snprintf(out + out_len, cap - out_len, __VA_ARGS__); } while (0)

    LFT_APPEND("%s\n", MGL_LFT_MARKER);

    /* helper struct for void-returning functions that gain a depth output */
    if (is_void && add_depth)
        LFT_APPEND("struct mgl_lft_out { float mgl_lft_z [[depth(any)]]; };\n");

    /* prefix: everything up to the parameters, with an extra member added to
     * the return-type struct when the function keeps its struct return.  The
     * return type token is re-emitted between its two halves. */
    {
        long ret_at = (long)(ret - msl);
        long after_ret = ret_at + (long)retlen;
        if (!is_void && add_depth) {
            char structpat[128];
            snprintf(structpat, sizeof structpat, "struct %.*s\n{", (int)retlen, ret);
            const char *sdecl = strstr(msl, structpat);
            bool done = false;
            if (sdecl) {
                const char *b = strchr(sdecl, '{');
                const char *q = b;
                int depth = 0;
                for (; *q; q++) {
                    if (*q == '{') depth++;
                    else if (*q == '}') { if (--depth == 0) break; }
                }
                if (*q == '}' && (long)(q - msl) < ret_at) {
                    LFT_APPEND("%.*s", (int)(q - msl), msl);
                    LFT_APPEND("    float mgl_lft_z [[depth(any)]];\n");
                    LFT_APPEND("%.*s", (int)(ret_at - (q - msl)), q);
                    done = true;
                }
            }
            if (!done) { free(out); return NULL; }  /* unexpected shape, bail out */
        } else {
            LFT_APPEND("%.*s", (int)ret_at, msl);
        }
        /* signature: return type, then up to (and including) the '(' */
        if (is_void && add_depth)
            LFT_APPEND("mgl_lft_out ");
        else
            LFT_APPEND("%.*s ", (int)retlen, ret);
        LFT_APPEND("%.*s", (int)(params_at - after_ret), msl + after_ret);
    }

    /* parameters: original list + a position argument when missing */
    LFT_APPEND("%.*s", (int)(close_paren - params_at), msl + params_at);
    if (!have_pos) {
        snprintf(posname, sizeof posname, "mgl_lft_pos");
        LFT_APPEND(", float4 %s [[position]]", posname);
    }
    LFT_APPEND(")%.*s{", (int)(body_at - close_paren - 1), msl + close_paren + 1);

    /* body: guard discards, write the depth member on every return */
    body = msl + body_at + 1;
    body_len = (size_t)(guard_to - body_at - 2);   /* drop the closing '}' */
    p = body;
    seg = body;
    while (p < body + body_len) {
        const char *d = strstr(p, "discard_fragment();");
        const char *r = strstr(p, "return ");
        const char *semi;
        if (d && (!r || d < r) && d + 19 <= body + body_len) {
            LFT_APPEND("%.*s", (int)(d - seg), seg);
            LFT_APPEND("{ if (as_type<uint>(%s.z) != 0xffffffffu) discard_fragment(); }", posname);
            p = d + 19;
            seg = p;
        } else if (r && r < body + body_len) {
            semi = strchr(r, ';');
            if (!semi || semi > body + body_len) break;
            LFT_APPEND("%.*s", (int)(r - seg), seg);
            if (!strncmp(r, "return out;", 11)) {
                if (add_depth && !is_void)
                    LFT_APPEND("{ out.mgl_lft_z = %s.z; return out; }", posname);
                else
                    LFT_APPEND("return out;");
            } else if (!strncmp(r, "return;", 7)) {
                if (add_depth && is_void)
                    LFT_APPEND("return mgl_lft_out{ %s.z };", posname);
                else
                    LFT_APPEND("return;");
            } else {
                LFT_APPEND("%.*s", (int)(semi - r + 1), r);
            }
            p = semi + 1;
            seg = p;
        } else {
            break;
        }
    }
    LFT_APPEND("%.*s", (int)(body + body_len - seg), seg);

    /* a void body that falls off the end still returns the depth value */
    if (is_void && add_depth)
        LFT_APPEND("return mgl_lft_out{ %s.z };\n", posname);
    LFT_APPEND("}");

    tail = msl + guard_to;
    LFT_APPEND("%s", tail);
#undef LFT_APPEND

    /* structural sanity: braces balanced */
    {
        int depth = 0;
        size_t i;
        for (i = 0; i < out_len; i++) {
            if (out[i] == '{') depth++;
            else if (out[i] == '}') depth--;
            if (depth < 0) { free(out); return NULL; }
        }
        if (depth != 0) { free(out); return NULL; }
    }
    return out;
}

/* the variant MSL for `program', or NULL when its fragment stage needs no
 * late-fragment-tests emulation.  Called from MGLRenderer+RenderState.m at
 * pipeline build time; the text is generated once at link time (cached). */
char *mgl_tess_late_fragment_variant(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    Spirv *sp;
    int i, slot = -1;
    char *v;

    if (!S || !program) return NULL;
    sp = &program->spirv[_FRAGMENT_SHADER];
    for (i = 0; i < MGL_TESS_LATE_FRAG_CACHE; i++)
        /* msl_src must match: relinking (and recycled Program addresses) hand
         * out a fresh msl_str, so a stale variant can never be selected */
        if (S->late_frag[i].program == program &&
            S->late_frag[i].msl_src == sp->msl_str)
            return S->late_frag[i].msl;          /* may be NULL: known-unneeded */

    if (!sp->msl_str || !sp->ir || strstr(sp->msl_str, "early_fragment_tests"))
        goto store_none;
    if (!spv_module_writes_memory(sp->ir, sp->size))
        goto store_none;

    v = mgl_late_fragment_tests_msl(sp->msl_str, sp->entry_point,
                                    !strstr(sp->msl_str, "[[depth"));
    if (!v)
        goto store_none;

    slot = (int)(program->name % MGL_TESS_LATE_FRAG_CACHE);
    if (S->late_frag[slot].msl)
        free(S->late_frag[slot].msl);
    S->late_frag[slot].program = program;
    S->late_frag[slot].msl_src = sp->msl_str;
    S->late_frag[slot].msl = v;
    return v;

store_none:
    slot = (int)(program->name % MGL_TESS_LATE_FRAG_CACHE);
    if (S->late_frag[slot].msl)
        free(S->late_frag[slot].msl);
    S->late_frag[slot].program = program;
    S->late_frag[slot].msl_src = sp->msl_str;
    S->late_frag[slot].msl = NULL;
    return NULL;
}

/* post-link: parse the tessellation modes and the patch data layouts */
/* True if the GLSL source uses the post_depth_coverage layout qualifier — as
 * opposed to merely naming the extension in a #extension line, which glslang
 * accepts with no behavioural effect.  Comments are skipped, preprocessor
 * lines (#extension/#ifdef/#define) are ignored (the extension NAME lives
 * there), and the token is matched with identifier boundaries so
 * my_post_depth_coverage / post_depth_coverage2 do not match. */
static bool mgl_src_uses_post_depth_coverage(const char *src)
{
    static const char tok[] = "post_depth_coverage";
    const size_t tlen = sizeof tok - 1;
    const char *p = src;
    bool in_line_comment = false, in_block_comment = false, line_is_pp = false;

    if (!src) return false;
    while (*p)
    {
        if (*p == '\n')
        {
            in_line_comment = false;
            line_is_pp = false;
            p++;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '#') line_is_pp = true;
            continue;
        }
        if (in_line_comment) { p++; continue; }
        if (in_block_comment)
        {
            if (p[0] == '*' && p[1] == '/') { in_block_comment = false; p += 2; }
            else p++;
            continue;
        }
        if (p[0] == '/' && p[1] == '/') { in_line_comment = true; p += 2; continue; }
        if (p[0] == '/' && p[1] == '*') { in_block_comment = true; p += 2; continue; }
        if (!line_is_pp && strncmp(p, tok, tlen) == 0)
        {
            bool before_ok = (p == src) ||
                             !(isalnum((unsigned char)p[-1]) || p[-1] == '_');
            bool after_ok = !(isalnum((unsigned char)p[tlen]) || p[tlen] == '_');
            if (before_ok && after_ok)
                return true;
            p += tlen;
            continue;
        }
        p++;
    }
    return false;
}

/* True if a linked program uses layout(post_depth_coverage).  Two layers:
 * the GLSL qualifier (glslang's OpenGL client compiles it away — SPIR-V's
 * PostDepthCoverage execution mode is Vulkan-only, the ARB spelling only
 * emits EarlyFragmentTests, probed with external glslang -G) and, for
 * completeness, the SPIR-V mode itself (were it ever to appear, SPIRV-Cross
 * would emit Metal's [[post_depth_coverage]], whose stencil semantics MGL
 * cannot honor). */
static bool mgl_program_uses_post_depth_coverage(GLMContext ctx, Program *program)
{
    (void)ctx;
    Shader *fs = program->shader_slots[_FRAGMENT_SHADER];
    if (fs && mgl_src_uses_post_depth_coverage(fs->src))
        return true;

    {
        int modes[16], nmodes = 0;
        if (spv_tess_modes(program->spirv[_FRAGMENT_SHADER].ir,
                           program->spirv[_FRAGMENT_SHADER].size,
                           SPV_EXECUTION_MODEL_FRAGMENT, modes, &nmodes, NULL))
            for (int m = 0; m < nmodes; m++)
                if (modes[m] == SPV_EM_POST_DEPTH_COVERAGE)
                    return true;
    }
    return false;
}

static void mgl_ext_post_link_tess(GLMContext ctx, Program *program)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;
    int domain, spacing, ccw, point_mode, out_vertices = 0;
    bool has_tcs, has_tes;

    if (!S || !program)
        return;

    /* GL_ARB_post_depth_coverage is not advertised (get.c, r41): Metal's
     * [[sample_mask, post_depth_coverage]] misses stencil-failed samples
     * (.agent/gap_post_depth_coverage.md), so no shader may depend on it.
     * glslang still accepts the GLSL extension name, so the feature use is
     * rejected here, at link: a fragment shader carrying the
     * post_depth_coverage layout qualifier fails the link (the CTS
     * post_depth_coverage_tests group is extension-gated and stays
     * NotSupported).  layout(early_fragment_tests) — GL 4.2 core — is a
     * different qualifier and unaffected. */
    if (mgl_program_uses_post_depth_coverage(ctx, program))
    {
        MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);
        obj->link_status = false;
        mgl_set_info_log(&obj->info_log,
                         "ERROR: GL_ARB_post_depth_coverage is not supported "
                         "(layout(post_depth_coverage) is not available)\n");
        tess_drop_program_record(ctx, program);
        return;
    }

    /* GL 4.6 §17.3.4: late per-fragment tests for side-effecting fragment
     * shaders.  mgl_tess_late_fragment_variant() generates the patched variant
     * lazily (at pipeline build time, when a depth attachment is present);
     * nothing to do here beyond pre-warming the cache.  Runs before the
     * tessellation early-out below — it applies to non-tessellation programs. */
    mgl_tess_late_fragment_variant(ctx, program);

    has_tcs = program->spirv[_TESS_CONTROL_SHADER].ir != NULL;
    has_tes = program->spirv[_TESS_EVALUATION_SHADER].ir != NULL;
    if (!has_tes)
    {
        /* a successful relink that dropped the TES must not leave this program's
         * old record installed: mgl_tess_active_program keys on the Program
         * pointer and would keep routing draws through the stale layout (the
         * O(1) mgl_tess_stage_bound gate and the record then disagree).  The
         * vs_kernel cache is NOT dropped — the program is alive and a relink
         * of the same program refreshes its kernel (a changed MSL replaces the
         * cached one). */
        tess_drop_program_record(ctx, program);
        return;                         /* not (a usable) tessellation program */
    }

    domain = spv_tess_domain(program->spirv[_TESS_EVALUATION_SHADER].ir,
                             program->spirv[_TESS_EVALUATION_SHADER].size,
                             &spacing, &ccw, &point_mode);
    if (domain < 0)
        return;

    if (has_tcs)
    {
        int modes[16], count = 0;
        if (!spv_tess_modes(program->spirv[_TESS_CONTROL_SHADER].ir,
                            program->spirv[_TESS_CONTROL_SHADER].size,
                            1 /* SpvExecutionModelTessellationControl */, modes, &count, &out_vertices) ||
            out_vertices < 1 || out_vertices > MGL_TESS_MAX_PATCH_VERTICES)
            out_vertices = 3;
    }

    /* replace any stale entry (relink) — records only, keep the separable vs
     * kernel: the program is live and both link paths (compile_stage and the
     * whole-link cache hit) re-cache the kernel, whose replace-on-changed-MSL
     * rule makes a purge unnecessary and would drop the fresh entry */
    tess_drop_program_record(ctx, program);

    tp = (MGLTessProgram *)calloc(1, sizeof(MGLTessProgram));
    if (!tp)
        return;
    tp->program = program;
    tp->program_name = program->name;
    tp->has_tcs = has_tcs;
    tp->domain = domain;
    tp->spacing = spacing;
    tp->vertex_order_ccw = ccw;
    tp->point_mode = point_mode;
    tp->tcs_out_vertices = has_tcs ? out_vertices : 0;
    tp->tessellation_active = 1;

    {
        MGLTessProgram **head = &S->programs[(((uintptr_t)program) >> 4) % MGL_TESS_PROGRAM_HASH];
        tp->next = *head;
        *head = tp;

    }

    /* patch data layouts, parsed from the generated MSL (raw per-stage structs; the
     * canonical records below give the offsets actually used at draw time) */
    if (has_tcs)
    {
        struct MslStruct po;
        char po_name[160];
        if (msl_parse_struct(program->spirv[_TESS_CONTROL_SHADER].msl_str, "main0_patchOut", &po) ||
            msl_parse_struct_entry(program->spirv[_TESS_CONTROL_SHADER].msl_str, &po, po_name, sizeof po_name))
        {
            tp->num_tcs_patch_out = po.count;
            tp->tcs_patch_out_size = po.size;
            for (int i = 0; i < po.count && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
                tp->tcs_patch_out[i] = po.members[i];
        }
    }

    /* TES interface layouts (from the stored hardware-form MSL): control point members
     * and patch-in members with their [[attribute(N)]] indices build the render
     * pipeline's vertex descriptor at draw time */
    {
        struct MslStruct pi;
        const char *msl = program->spirv[_TESS_EVALUATION_SHADER].msl_str;
        char name[160];

        if (msl_patch_in_name(msl, name, sizeof name) && msl_parse_struct(msl, name, &pi))
        {
            int m = 0;
            for (int i = 0; i < pi.count && m < MGL_TESS_MAX_PATCH_MEMBERS; i++)
            {
                if (i == pi.cp_ref_member)
                    continue;
                tp->tes_patch_in[m++] = pi.members[i];
            }
            tp->num_tes_patch_in = m;
            tp->tes_patch_in_size = pi.size;

            struct MslStruct cp;
            char cpname[128];
            if (msl_cp_struct_name(msl, name, cpname, sizeof cpname) &&
                msl_parse_struct(msl, cpname, &cp))
            {
                tp->cp_record_size = cp.size;
                tp->num_tes_cp = cp.count < MGL_TESS_MAX_PATCH_MEMBERS ? cp.count : MGL_TESS_MAX_PATCH_MEMBERS;
                for (int i = 0; i < tp->num_tes_cp; i++)
                    tp->tes_cp[i] = cp.members[i];
            }
        }

        /* the baked control point count of [[patch(domain, N)]] */
        tp->hw_cp_count = tp->has_tcs ? tp->tcs_out_vertices : 3;

        /* one canonical record layout per interface (control points / per-patch),
         * shared by producer and consumer stages; recompiles the VS / TCS kernels
         * when their SPIRV-Cross struct differs (see "canonical records") */
        tess_canonicalize_records(ctx, tp);
    }

    /* the GPU blob kernel's member map (levels + TCS patch-output copies) */
    mgl_tess_build_blob_map(ctx, tp);
}

/* ------------------------------------------------------------------ GL hooks */

static bool mgl_tess_program_completion_status(GLMContext ctx, Program *program, GLenum pname, GLint *params);

void mgl_ext_init_tess(GLMContext ctx)
{
    struct MGLExtState_tess *S;

    if (ctx->state.ext_tess)
        return;
    S = (struct MGLExtState_tess *)calloc(1, sizeof(struct MGLExtState_tess));
    if (!S)
        return;
    ctx->state.ext_tess = S;

    /* ARB/KHR_parallel_shader_compile: the spec's initial "no limit" state */
    S->max_shader_compiler_threads = (GLint)0xFFFFFFFFu;

    mgl_ext_register_post_link(mgl_ext_post_link_tess);
    mgl_program_register_programiv_hook(mgl_tess_program_completion_status);
}

bool mgl_ext_get_tess(GLMContext ctx, GLenum pname, GLenum type, void *data)
{
    /* limits are served by get.c; GL_PATCH_VERTICES / default levels by rstate */
    if (pname == GL_MAX_SHADER_COMPILER_THREADS_ARB /* == ..._KHR, 0x91B0 */)
    {
        /* the glMaxShaderCompilerThreads hint (always answered in every glGet
         * type; the ARB/KHR enum values are identical) */
        GLint v = ctx->state.ext_tess ? ctx->state.ext_tess->max_shader_compiler_threads
                                      : (GLint)0xFFFFFFFFu;
        mgl_get_store(type, data, 0, (double)v);
        return true;
    }
    return false;
}

/* glGetProgramiv(GL_COMPLETION_STATUS_ARB/KHR): MGL links synchronously, which
 * ARB_parallel_shader_compile explicitly permits, so a program's compile/link
 * is always complete by the time glGetProgramiv observes it. */
static bool mgl_tess_program_completion_status(GLMContext ctx, Program *program, GLenum pname, GLint *params)
{
    (void)ctx;
    (void)program;
    if (pname == GL_COMPLETION_STATUS_ARB)      /* == GL_COMPLETION_STATUS_KHR */
    {
        *params = GL_TRUE;
        return true;
    }
    return false;
}

/* glMaxShaderCompilerThreads[ARB|KHR]: a *hint* only.  MGL compiles and links
 * synchronously, which the ARB_parallel_shader_compile spec explicitly allows
 * ("the implementation may ignore the limit entirely"), so this just stores
 * the value glGet(GL_MAX_SHADER_COMPILER_THREADS_ARB) reports back. */
void mglMaxShaderCompilerThreads(GLMContext ctx, GLuint count)
{
    if (!ctx->state.ext_tess)
        mgl_ext_init_tess(ctx);
    if (ctx->state.ext_tess)
        ctx->state.ext_tess->max_shader_compiler_threads = (GLint)count;
}

/* GL 4.6 §11.1.3.1/§7.3 link validation: the tessellation control shader's
 * layout(vertices=N) output patch size must lie in [1, GL_MAX_TESS_CONTROL_OUTPUT_
 * VERTICES]; outside, the program must FAIL to link (glslang accepts any N, the
 * limit is implementation-dependent — this is where MGL enforces it). */
/* ---------------- gl_PerVertex interface validation (link time) ----------------
 *
 * GL 4.6 §11.1.2.5 / GLSL 4.60 §7.1: a shader stage may redeclare the built-in
 * gl_PerVertex block with a SUBSET of its members; linking then requires every
 * member the consumer stage declares to be present in the producer stage's output
 * block (a stage without a redeclaration exports the full built-in set for its
 * #version).  mgl_tess_glsl_compat expands all redeclarations to the full set
 * before glslang sees them (glslang demands equal member sets), so the mismatch
 * is invisible to the compiler - parse the ORIGINAL shader sources here instead
 * (KHR-GL46.CommonBugs.CommonBug_PerVertexValidation). */

enum { PVM_POSITION = 1u << 0, PVM_POINT_SIZE = 1u << 1, PVM_CLIP = 1u << 2, PVM_CULL = 1u << 3 };

static const char *tess_perv_member_name(unsigned m)
{
    switch (m)
    {
        case PVM_POSITION:   return "gl_Position";
        case PVM_POINT_SIZE: return "gl_PointSize";
        case PVM_CLIP:       return "gl_ClipDistance";
        case PVM_CULL:       return "gl_CullDistance";
    }
    return "?";
}

static const char *tess_stage_name(int stage)
{
    switch (stage)
    {
        case _VERTEX_SHADER:            return "vertex";
        case _TESS_CONTROL_SHADER:      return "tessellation control";
        case _TESS_EVALUATION_SHADER:   return "tessellation evaluation";
        case _GEOMETRY_SHADER:          return "geometry";
        case _FRAGMENT_SHADER:          return "fragment";
    }
    return "?";
}

/* members of the built-in gl_PerVertex block for the source's #version */
static unsigned tess_perv_full_mask(const char *src)
{
    int version = 110;
    const char *v = strstr(src, "#version");
    if (v)
        version = atoi(v + 8);
    unsigned m = PVM_POSITION | PVM_POINT_SIZE;
    if (version >= 130) m |= PVM_CLIP;
    if (version >= 450) m |= PVM_CULL;
    return m;
}

/* bitmask of the members an 'in'/'out gl_PerVertex {...}' redeclaration declares.
 * *found is 0 when the source carries no such redeclaration (full built-in set).
 * One declaration per direction is assumed (GL requires it). */
static unsigned tess_perv_declared(const char *src, int want_out, int *found)
{
    size_t len = strlen(src);
    *found = 0;
    const char *p = src;
    while ((p = strstr(p, "gl_PerVertex")) != NULL)
    {
        size_t at = (size_t)(p - src);
        /* word boundaries */
        if ((at == 0 || (isalnum((unsigned char)src[at - 1]) || src[at - 1] == '_')) ||
            (at + 12 < len && (isalnum((unsigned char)p[12]) || p[12] == '_')))
        {
            p += 12;
            continue;
        }
        /* the qualifier directly before the block name */
        size_t q = at;
        while (q > 0 && (src[q - 1] == ' ' || src[q - 1] == '\t' || src[q - 1] == '\r' || src[q - 1] == '\n'))
            q--;
        int is_out = (q >= 3 && !strncmp(src + q - 3, "out", 3) &&
                      (q - 3 == 0 || !isalnum((unsigned char)src[q - 4]) && src[q - 4] != '_'));
        int is_in  = (q >= 2 && !strncmp(src + q - 2, "in", 2) &&
                      (q - 2 == 0 || !isalnum((unsigned char)src[q - 3]) && src[q - 3] != '_'));
        if ((want_out && !is_out) || (!want_out && !is_in))
        {
            p += 12;
            continue;
        }
        /* the block body */
        size_t b = at + 12;
        while (src[b] == ' ' || src[b] == '\t' || src[b] == '\r' || src[b] == '\n') b++;
        if (src[b] != '{')
        {
            p += 12;
            continue;
        }
        int depth = 1;
        size_t e = b + 1;
        while (src[e] && depth)
        {
            if (src[e] == '{') depth++;
            else if (src[e] == '}') depth--;
            if (depth) e++;
        }
        if (!src[e])
            break;
        char body[1024];
        size_t n = e - (b + 1);
        if (n >= sizeof(body)) n = sizeof(body) - 1;
        memcpy(body, src + b + 1, n);
        body[n] = 0;

        unsigned mask = 0;
        char *tokctx = NULL, *tokctx2 = NULL;
        for (char *decl = strtok_r(body, ";", &tokctx); decl; decl = strtok_r(NULL, ";", &tokctx))
        {
            char last[64] = "";
            for (char *w = strtok_r(decl, " \t\r\n", &tokctx2); w; w = strtok_r(NULL, " \t\r\n", &tokctx2))
                snprintf(last, sizeof last, "%s", w);
            size_t ln = strlen(last);
            if (ln && last[ln - 1] == ']')
            {
                last[ln - 1] = 0;
                ln--;
                while (ln > 1 && (last[ln - 1] == ' ' || last[ln - 1] == '\t')) { last[ln - 1] = 0; ln--; }
                if (ln && last[ln - 1] == '[')
                    last[ln - 1] = 0;
            }
            for (int k = 0; k < 4; k++)
                if (!strcmp(last, tess_pervtx_name[k]))
                    mask |= 1u << k;
        }
        *found = 1;
        return mask;
    }
    return 0;
}

/* append a message to the program's link log and fail the link */
static void tess_link_log_fail(MGLProgramObj *obj, const char *msg)
{
    size_t len = strlen(msg) + (obj->info_log ? strlen(obj->info_log) : 0) + 8;
    char *old_log = obj->info_log;

    obj->info_log = (char *)malloc(len);
    if (obj->info_log)
        snprintf(obj->info_log, len, "%s%s%s", old_log ? old_log : "",
                 old_log ? "\n" : "", msg);
    free(old_log);
    obj->link_status = false;
}

/* check every active producer->consumer gl_PerVertex interface of the program */
static void tess_check_perv_interface(GLMContext ctx, Program *program)
{
    (void)ctx;
    static const int consumers[] = { _TESS_CONTROL_SHADER, _TESS_EVALUATION_SHADER, _GEOMETRY_SHADER, _FRAGMENT_SHADER };

    for (size_t c = 0; c < sizeof(consumers) / sizeof(consumers[0]); c++)
    {
        int cs = consumers[c];
        Shader *csh = program->shader_slots[cs];
        if (!csh || !csh->src)
            continue;
        /* the producer is the highest active stage below the consumer */
        Shader *psh = NULL;
        int ps = -1;
        for (int s = cs - 1; s >= _VERTEX_SHADER; s--)
            if (program->shader_slots[s] && program->shader_slots[s]->src)
            {
                psh = program->shader_slots[s];
                ps = s;
                break;
            }
        if (!psh)
            continue;

        int cfound = 0, pfound = 0;
        unsigned cmask = tess_perv_declared(csh->src, 0, &cfound);
        unsigned pmask = tess_perv_declared(psh->src, 1, &pfound);
        if (!pfound)
            pmask = tess_perv_full_mask(psh->src);
        /* Only a consumer that EXPLICITLY redeclares gl_PerVertex with a subset
         * demands those members from the producer; without a redeclaration the
         * stage reads at most what the built-in declaration provides and the
         * producer's own subset is fine (a VS exporting only gl_Position links
         * against any un-redeclared consumer - test_compat_glsl
         * glsl_150_gl_per_vertex_redeclaration). */
        if (!cfound)
            continue;
        unsigned missing = cmask & ~pmask;
        if (!missing)
            continue;

        char msg[512];
        snprintf(msg, sizeof msg,
                 "ERROR: gl_PerVertex interface mismatch: %s shader declares %s but the "
                 "%s shader does not export it (link failed)\n",
                 tess_stage_name(cs), tess_perv_member_name(missing & -missing),
                 tess_stage_name(ps));
        tess_link_log_fail(MGL_PROGRAM_OBJ(program), msg);
    }
}


bool mgl_program_check_tess_link(GLMContext ctx, Program *program)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);
    const uint32_t *ir;
    size_t ir_size;
    int modes[16], count = 0, out_vertices = 0;

    /* gl_PerVertex interface validation runs for every program (GL 4.6 §11.1.2.5:
     * a consumer stage may redeclare gl_PerVertex with a subset, but every member it
     * declares must be exported by the producer stage).  glslang cannot check this:
     * mgl_tess_glsl_compat normalizes all redeclarations to the full member set so the
     * cross-stage blocks always match, so the ORIGINAL sources are parsed here. */
    tess_check_perv_interface(ctx, program);

    if (!program->spirv[_TESS_CONTROL_SHADER].ir)
        return true;

    ir = program->spirv[_TESS_CONTROL_SHADER].ir;
    ir_size = program->spirv[_TESS_CONTROL_SHADER].size;
    if (!spv_tess_modes(ir, ir_size, 1 /* SpvExecutionModelTessellationControl */,
                        modes, &count, &out_vertices))
        return true;            /* no execution modes parsed: nothing to enforce */

    if (out_vertices >= 1 && out_vertices <= MGL_TESS_MAX_PATCH_VERTICES)
        return true;

    obj->link_status = false;
    {
        char log[256];

        snprintf(log, sizeof log,
                 "ERROR: tessellation control shader output patch vertex count %d is outside "
                 "the supported range [1, %d]\n", out_vertices, MGL_TESS_MAX_PATCH_VERTICES);
        {
            size_t len = strlen(log) + (obj->info_log ? strlen(obj->info_log) : 0) + 8;
            char *old_log = obj->info_log;

            obj->info_log = (char *)malloc(len);
            if (obj->info_log)
                snprintf(obj->info_log, len, "%s%s%s", old_log ? old_log : "",
                         old_log ? "\n" : "", log);
            free(old_log);
        }
    }
    return false;
}

/* note exact tessellation statistics to the query domain */
/* query bookkeeping for GPU-path draws (no CPU tessellation): the tessellated output
 * counts, computed from the levels with exactly the arithmetic of the GPU tessellator's
 * mgl_patch_counts (MGLTessKernels.h) and the CPU tessellator. */
void mgl_tess_note_counts(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches)
{
    GLuint ncoords = 0, nindex = 0;

    for (unsigned p = 0; p < patches; p++)
    {
        const float *f = factors + p * 6;
        struct D3DTess t;

        memset(&t, 0, sizeof t);
        t.mesh = NULL;
        t.pid = (float)p;
        t.ccw = tp->vertex_order_ccw != 0;
        t.tri = (tp->domain == MGL_TESS_DOMAIN_TRIANGLES);
        t.base = 0;
        t.count_only = true;

        /* count-only run of the same generator the GPU tessellator uses (a culled
         * patch reports 0/0 and adds nothing) */
        switch (tp->domain)
        {
            case MGL_TESS_DOMAIN_ISOLINES:
                d3d_isoline_process(&t, tp->spacing, f);
                break;
            case MGL_TESS_DOMAIN_TRIANGLES:
                d3d_tri_process(&t, tp->spacing, f);
                break;
            default:
                d3d_quad_process(&t, tp->spacing, f);
                break;
        }

        ncoords += t.ncoord;
        nindex += (tp->point_mode) ? t.ncoord : t.nindex;
    }

    if (ncoords || nindex)
    {
        GLuint prims;

        if (tp->point_mode)
            prims = ncoords;
        else if (tp->domain == MGL_TESS_DOMAIN_ISOLINES)
            prims = nindex / 2u;
        else
            prims = nindex / 3u;
        mgl_tess_note_draw(ctx, patches, ncoords, prims);
    }
}

/* words a packed MSL type occupies (geometry.c's rule: elems*cols*rows*base) */
static bool tess_msl_type_words(const char *type, uint32_t *words)
{
    uint32_t base = 1, cols = 1, rows = 1;
    const char *d;

    if (!strncmp(type, "spvUnsafeArray<", 15) && (d = strchr(type + 15, ',')) != NULL)
    {
        /* a flat slot array: N elements of the element type */
        char elem[32];
        uint32_t n = 0, w = 0;
        int k = (int)(d - (type + 15));

        if (k < 0 || k >= (int)sizeof elem) return false;
        snprintf(elem, sizeof elem, "%.*s", k, type + 15);
        if (sscanf(d + 1, "%u", &n) != 1) return false;
        if (!tess_msl_type_words(elem, &w)) return false;
        *words = w * n;
        return true;
    }
    if (!strncmp(type, "float", 5) || !strncmp(type, "int", 3) || !strncmp(type, "uint", 4) ||
        !strncmp(type, "bool", 4) || !strncmp(type, "short", 5) || !strncmp(type, "half", 4))
        base = 1;
    else if (!strncmp(type, "double", 6))
        base = 2;
    else
        return false;               /* arrays / structs are not captured */

    d = type;
    while (*d && !(*d >= '0' && *d <= '9'))
        d++;
    if (*d)
    {
        cols = (uint32_t)strtoul(d, NULL, 10);
        while (*d && *d != 'x')
            d++;
        if (*d == 'x')
            rows = (uint32_t)strtoul(d + 1, NULL, 10);
    }
    *words = base * cols * rows;
    return true;
}

/* the capture-kernel MSL: the plain TES vertex function rewritten into a compute
 * kernel storing the packed output record (see the MGLTessCapture contract).
 * Fills tp->cap_members[] / tp->cap_stride; NULL when the output interface has
 * something unsupported. Caller frees. */
/* number of doubles behind a TES output member, from the GL-view module: the
 * capture record's MSL repr pads each dvecR column to whole uint4 slots, but
 * every consumer of the capture (the geometry domain's import and the xfb
 * publish) reads the doubles compactly, two words each.  0 = not a double
 * varying (u32/float members keep their natural size). */
static uint32_t tess_out_doubles(GLMContext ctx, MGLTessProgram *tp, const char *name)
{
    Program *prog = tp->program;
    uint32_t *ir;
    size_t n, m;

    if (!prog || !name || !*name)
        return 0;
    ir = prog->spirv[_TESS_EVALUATION_SHADER].ir;
    n = prog->spirv[_TESS_EVALUATION_SHADER].size;
    if (!ir || n < 8 || !ctx)
        return 0;

    /* find the member/variable by name: OpName (variables) or OpMemberName
     * (interface block members) */
    m = 5;
    while (m + 3 < n)
    {
        uint32_t wc = ir[m] >> 16, op = ir[m] & 0xffff;
        uint32_t tid = 0;

        if (wc == 0) break;
        if (op == 5 /* OpName */)
            tid = ir[m + 1];
        else if (op == 6 /* OpMemberName */)
            tid = 0;                 /* handled below: need the member index */
        if (op == 5 || op == 6)
        {
            char buf[128];
            size_t o = 0;
            const uint32_t *w = ir + m + (op == 5 ? 2 : 3);
            const uint32_t *we = ir + m + wc;

            for (; w < we && o + 4 < sizeof buf; w++)
            {
                for (int b = 0; b < 4; b++)
                {
                    char c = (char)((*w >> (8 * b)) & 255);
                    if (!c) { w = we; break; }
                    buf[o++] = c;
                }
            }
            buf[o] = 0;
            if (!strcmp(buf, name))
            {
                /* resolve the type: the variable itself, or the struct member */
                uint32_t ty = 0;

                if (op == 5)
                {
                    /* OpName names the variable; find its OpVariable */
                    size_t v = 5;
                    while (v + 4 < n)
                    {
                        uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                        if (wv == 0) break;
                        if (ov == 59 && ir[v + 2] == tid && ir[v + 3] == SpvStorageClassOutput)
                        {
                            ty = ir[v + 1];   /* the pointer type */
                            break;
                        }
                        v += wv;
                    }
                }
                else
                {
                    /* OpMemberName: (struct type, member index) -> member type */
                    size_t v = 5;
                    uint32_t midx = ir[m + 2];

                    while (v + 3 < n)
                    {
                        uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                        if (wv == 0) break;
                        if (ov == 30 /* OpTypeStruct */ && ir[v + 1] == tid)
                        {
                            if (midx + 2 < wv)
                                ty = ir[v + 2 + midx];
                            break;
                        }
                        v += wv;
                    }
                }
                /* walk the (pointer-stripped) type: array * matrix columns *
                 * vector rows doubles */
                {
                    uint32_t elems = 1, cols = 1, rows = 1, isdbl = 0, pass = 0;

                    while (ty && pass++ < 8)
                    {
                        size_t v = 5;

                        while (v + 3 < n)
                        {
                            uint32_t wv = ir[v] >> 16, ov = ir[v] & 0xffff;
                            if (wv == 0) break;
                            if (ov >= 19 && ov <= 33 && ir[v + 1] == ty)
                            {
                                if (ov == 32) { ty = ir[v + 3]; break; }             /* pointer */
                                if (ov == 28)                                        /* array */
                                {
                                    size_t c = 5;
                                    uint32_t alen = 1;
                                    while (c + 3 < n)
                                    {
                                        uint32_t wcc = ir[c] >> 16, oc = ir[c] & 0xffff;
                                        if (wcc == 0) break;
                                        if (oc == 43 && ir[c + 2] == ir[v + 3])
                                        {
                                            alen = ir[c + 3];
                                            break;
                                        }
                                        c += wcc;
                                    }
                                    elems *= alen;
                                    ty = ir[v + 2];
                                    break;
                                }
                                if (ov == 24) { cols = ir[v + 3]; ty = ir[v + 2]; break; }   /* matrix */
                                if (ov == 23) { rows = ir[v + 3]; ty = ir[v + 2]; break; }   /* vector */
                                if (ov == 22) { isdbl = (ir[v + 2] == 64); ty = 0; break; }  /* float */
                                ty = 0;
                                break;
                            }
                            v += wv;
                        }
                    }
                    if (isdbl)
                        return elems * cols * rows;
                    return 0;
                }
            }
        }
        m += wc;
    }
    return 0;
}

char *mgl_tess_capture_msl(GLMContext ctx, MGLTessProgram *tp)
{
    char *plain = mgl_tess_plain_msl(ctx, tp);
    char *out = NULL, *k = NULL;    const char *sig, *name, *open, *vid, *body, *p;
    char types[MGL_TESS_CAPTURE_MAX][48];
    uint32_t dbls[MGL_TESS_CAPTURE_MAX];    /* per member: double count (0 = plain u32/float member) */
    uint32_t slotsz[MGL_TESS_CAPTURE_MAX];  /* ... words per fp64 slot: 2 (uint2) or 4 (uint4) */
    uint32_t slotn[MGL_TESS_CAPTURE_MAX];   /* ... slots in the MSL repr (1 for a bare uint2/uint4) */
    char cap_retname[96];
    bool void_ret = false;                 /* the entry returns void (store-only TES) */
    struct MslStruct rec;
    const char *cursor;
    uint32_t off = 4;
    size_t cap, len = 0;

    if (!plain)
        return NULL;

    sig = strstr(plain, "vertex ");
    name = open = vid = body = NULL;
    if (sig)
    {
        /* the return type is the first token after "vertex " */
        const char *rt = sig + 7 + strspn(sig + 7, " \t\n");
        const char *rte = rt + strcspn(rt, " \t\n");

        name = rte + strspn(rte, " \t\n");
        open = name ? strchr(name, '(') : NULL;
        vid = plain ? strstr(plain, "uint mgl_vid [[vertex_id]]") : NULL;
        body = vid ? strchr(vid, '{') : NULL;
        if (rt >= rte || (size_t)(rte - rt) >= sizeof cap_retname)
        {
            rt = NULL;
            name = NULL;
        }
        else
            snprintf(cap_retname, sizeof cap_retname, "%.*s", (int)(rte - rt), rt);
    }
    if (!sig || !name || !open || !vid || !body || open > vid)
        { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 1 sig=%p name=%p open=%p vid=%p body=%p\n", (const void*)sig, (const void*)name, (const void*)open, (const void*)vid, (const void*)body); goto done; }
    void_ret = !strcmp(cap_retname, "void");

    /* "vertex <Ret> name(...)": the record type is between 'vertex ' and the name */
    {
        if (!void_ret)
        {
            if (!msl_parse_struct(plain, cap_retname, &rec))
                { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 3 (%s)\n", cap_retname); goto done; }
        }
        else
        {
            /* a store-only TES (image/SSBO side effects, no gl_Position, no varyings)
             * returns void: there is no record struct to parse.  The capture record
             * degenerates to a zero gl_Position — gl_in[].gl_Position is undefined
             * per GL when the TES never wrote it, and the geometry stage's user
             * inputs match nothing (a linkable GS then reads no varyings either).
             * The kernel still runs the body: its image/SSBO writes must happen. */
            memset(&rec, 0, sizeof rec);
            snprintf(rec.members[0].type, sizeof rec.members[0].type, "float4");
            snprintf(rec.members[0].name, sizeof rec.members[0].name, "gl_Position");
            rec.count = 1;
        }
    }

    tp->cap_members_n = 0;
    for (int i = 0; i < rec.count && tp->cap_members_n < MGL_TESS_CAPTURE_MAX; i++)
    {
        const char *m = rec.members[i].name;

        /* the rstate fixup injects a synthetic point-size member into the last
         * pre-raster stage's output record; it is not a shader output, skip it */
        if (!strcmp(m, "mgl_rs_PointSize"))
            continue;
        uint32_t w;

        dbls[tp->cap_members_n] = 0;
        slotsz[tp->cap_members_n] = 0;
        slotn[tp->cap_members_n] = 1;
        if (!strcmp(m, "gl_Position"))
        {
            w = 4;
            snprintf(types[tp->cap_members_n], 48, "float4");
            tp->cap_members[tp->cap_members_n].offset = 0;
        }
        else if (!strcmp(m, "gl_ClipDistance") || !strcmp(m, "gl_CullDistance"))
            continue;               /* the array-attribute twin of the flattened members */
        else if (!strncmp(m, "gl_ClipDistance_", 16) || !strncmp(m, "gl_CullDistance_", 16))
        {
            /* flattened gl_PerVertex builtin array element (SPIRV-Cross [[user(clipN)]] /
             * [[user(cullN)]]): the tess lowering copies the evaluation stage's
             * gl_ClipDistance/gl_CullDistance outputs into per-element user attributes
             * ("gl_ClipDistance_0", ...).  Capture them — a following geometry stage
             * reads gl_in[].gl_ClipDistance/gl_CullDistance from the record, the GLSL/
             * gl_spirv validation programs (KHR-GL46.gl_spirv
             * spirv_validation_builtin_variable_decorations tessGeom) thread them through
             * every stage, and the cull-distance emulation (the synthetic geometry
             * kernel) evaluates whole-primitive culling on the TES output from them (the
             * geometry import maps the element members onto its input record's clip/cull
             * words by name).  One float word per element. */
            const char *e = m + 16;    /* both prefixes are 16 bytes */
            int digits = 0;

            for (const char *q = e; *q; q++)
                if (*q < '0' || *q > '9') { digits = 0; break; }
                else digits++;
            if (!digits || strcmp(rec.members[i].type, "float"))
            {
                if (tess_trace())
                    fprintf(stderr, "MGL tess: capture: unsupported builtin element %s of type %s\n",
                            m, rec.members[i].type);
                { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 4b\n"); goto done; }
            }
            w = 1;
            snprintf(types[tp->cap_members_n], 48, "float");
            tp->cap_members[tp->cap_members_n].offset = off;
        }
        else if (!strcmp(m, "gl_PointSize"))
        {
            /* one float word: a following geometry stage reads gl_in[].gl_PointSize
             * from the capture record (dEQP ES31 point_size
             * vertex_set_control_pass_eval_add_geometry_add: the VS's size + the
             * evaluation stage's + the geometry stage's), and a transform feedback
             * session capturing the evaluation stage records it by name */
            w = 1;
            snprintf(types[tp->cap_members_n], 48, "float");
            tp->cap_members[tp->cap_members_n].offset = off;
        }
        else if (!strncmp(m, "gl_", 3))
            continue;               /* other builtins are not captured */
        else if (!tess_msl_type_words(rec.members[i].type, &w))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: capture: unsupported varying %s of type %s\n",
                        m, rec.members[i].type);
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 4\n"); goto done; }
        }
        else
        {
            const char *t = rec.members[i].type;
            uint32_t d = tess_out_doubles(ctx, tp, m);

            snprintf(types[tp->cap_members_n], 48, "%s", t);
            tp->cap_members[tp->cap_members_n].offset = off;
            if (d)
            {
                /* fp64 repr: the doubles sit two per uint4 slot (one per uint2
                 * slot), possibly with a padded tail — the capture record
                 * carries them COMPACTLY, matching the geometry import and the
                 * xfb publish */
                uint32_t wsz = strstr(t, "uint2") ? 2 : 4;

                dbls[tp->cap_members_n] = d;
                slotsz[tp->cap_members_n] = wsz;
                {
                    const char *nm = strstr(t, "spvUnsafeArray<");
                    uint32_t cnt = 1;

                    if (nm && sscanf(strchr(nm, ',') + 1, "%u", &cnt) != 1)
                        cnt = 1;
                    slotn[tp->cap_members_n] = cnt;
                }
                w = 2 * d;
            }
        }

        snprintf(tp->cap_members[tp->cap_members_n].name, 64, "%s", m);
        tp->cap_members[tp->cap_members_n].words = w;
        if (strcmp(m, "gl_Position"))
            off += w;
        tp->cap_members_n++;
    }
    {
        bool have_pos = false;

        for (int i = 0; i < tp->cap_members_n; i++)
            if (!strcmp(tp->cap_members[i].name, "gl_Position"))
                have_pos = true;
        if (!have_pos)
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 5\n"); goto done; }               /* gl_Position must be present */
    }
    /* Pad the record to a multiple of 4 words (16 bytes): every record starts with the
     * gl_Position float4 store, and a device float4 store at a non-16-byte-aligned
     * address is undefined on Metal - the tessellated vertices past the first would
     * come back as garbage.  The consumers (the geometry domain's import kernel and the
     * diagnostic capture query) read tp->cap_stride, so they follow the padding. */
    tp->cap_stride = (off + 3u) & ~3u;

    if (tess_trace())
        fprintf(stderr, "MGL tess: capture: %s (stride %u, %d members)\n",
                "layout built", tp->cap_stride, tp->cap_members_n);
    cap = strlen(plain) + 4096;
    k = (char *)malloc(cap);
    if (!k)
        { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 6\n"); goto done; }

    str_append(&k, &len, &cap, "%.*skernel void ", (int)(sig - plain), plain);
    str_append(&k, &len, &cap, "%.*s_cap(", (int)(open - name), name);
    str_append(&k, &len, &cap, "%.*s", (int)(vid - (open + 1)), open + 1);
    /* the plain signature's tail after the mgl_vid parameter: the original
     * entry's buffers (uniform blocks etc.) — they must stay in the kernel's
     * signature, the body reads them */
    {
        const char *vp = vid + strlen("uint mgl_vid [[vertex_id]]");
        const char *close = NULL;
        int depth = 0;

        for (const char *q = open; *q; q++)
        {
            if (q[0] == '[' && q[1] == '[')
            {
                const char *e = strstr(q, "]]");
                if (e) { q = e; continue; }
            }
            if (*q == '(') depth++;
            else if (*q == ')' && --depth == 0) { close = q; break; }
        }
        while (*vp == ' ') vp++;
        if (*vp == ',' && close)
        {
            const char *t = vp + 1;

            while (*t == ' ' || *t == '\n') t++;
            str_append(&k, &len, &cap, "%.*s, ", (int)(close - t), t);
        }
    }
    str_append(&k, &len, &cap, "uint mgl_vid [[thread_position_in_grid]], "
                               "device const uint* mgl_cap_count [[buffer(%d)]], "
                               "device uint* mgl_cap [[buffer(%d)]])\n",
               MGL_TESS_TES_CAP_COUNT, MGL_TESS_TES_CAP_RECS);
    /* the original signature's tail ("uint mgl_vid [[vertex_id]])") is dropped: the
     * record store replaces the return value, and thread_position_in_grid is mgl_vid */
    str_append(&k, &len, &cap, "{");
    str_append(&k, &len, &cap, "\n    if (mgl_vid >= mgl_cap_count[0]) return;\n");

    /* replace every "return <var>;" with the packed record store: `cursor` walks the
     * original body starting just after its '{' (the '{' was emitted above) */
    cursor = body + 1;
    while ((p = strstr(cursor, "return ")) != NULL)
    {
        const char *semi = strchr(p, ';');
        char varname[32], store[2048];
        size_t vn = semi ? (size_t)(semi - (p + 7)) : 0;
        int slen = 0;

        if (!semi)
            break;
        if (vn >= sizeof varname)
            vn = sizeof varname - 1;
        snprintf(varname, sizeof varname, "%.*s", (int)vn, p + 7);

        str_append(&k, &len, &cap, "%.*s", (int)(p - cursor), cursor);
        if (vn == 0)
        {
            /* a bare "return;" inside the body is legal in the kernel too */
            str_append(&k, &len, &cap, "return;");
            cursor = semi + 1;
            continue;
        }
        for (int i = 0; i < tp->cap_members_n; i++)
        {
            const MGLTessCaptureMember *cm = &tp->cap_members[i];
            const char *t = types[i];
            const char *arr = strstr(t, "spvUnsafeArray<");

            str_append(&k, &len, &cap, "%s", i == 0 ? "{ " : "");
            if (dbls[i])
            {
                /* the doubles sit two per uint4 slot (one per uint2 slot) with
                 * a possibly padded tail: copy them out compactly, two words
                 * per double */
                uint32_t sw = slotsz[i];     /* words per slot: 2 or 4 */
                uint32_t spdbl = sw / 2;     /* doubles per slot: 1 or 2 */

                str_append(&k, &len, &cap,
                           "device uint* mgl_w = (device uint*)((device uchar*)mgl_cap + "
                           "(size_t)mgl_vid * %uu + %uu); (void)mgl_w;\n",
                           tp->cap_stride * 4u, cm->offset * 4u);
                for (uint32_t d = 0; d < dbls[i]; d++)
                {
                    uint32_t slot = d / spdbl;
                    const char *lo = (spdbl == 2 && (d & 1)) ? ".z" : ".x";
                    const char *hi = (spdbl == 2 && (d & 1)) ? ".w" : ".y";
                    char ix[16];

                    ix[0] = 0;
                    if (arr) snprintf(ix, sizeof ix, "[%u]", slot);
                    str_append(&k, &len, &cap, "mgl_w[%u] = %s.%s%s%s; mgl_w[%u] = %s.%s%s%s;\n",
                               2 * d, varname, cm->name, ix, lo,
                               2 * d + 1, varname, cm->name, ix, hi);
                }
            }
            else if (arr)
            {
                /* plain (non-double) slot array: element-wise copy — a struct
                 * copy across address spaces would not compile */
                str_append(&k, &len, &cap,
                           "device uint* mgl_w = (device uint*)((device uchar*)mgl_cap + "
                           "(size_t)mgl_vid * %uu + %uu);\n",
                           tp->cap_stride * 4u, cm->offset * 4u);
                for (uint32_t q = 0; q < slotn[i] && q < 32; q++)
                    str_append(&k, &len, &cap, "mgl_w[%u] = %s.%s[%u];\n", q, varname, cm->name, q);
            }
            else
                str_append(&k, &len, &cap,
                           "*(device %s*)((device uchar*)mgl_cap + (size_t)mgl_vid * %uu + %uu) = %s.%s;\n",
                           t, tp->cap_stride * 4u, cm->offset * 4u, varname, cm->name);
        }
        str_append(&k, &len, &cap, "}\nreturn;\n");
        cursor = semi + 1;
    }
    if (void_ret)
    {
        /* no return statement carries a record: emit the body up to its closing
         * brace and store the zero gl_Position record at its end */
        const char *bclose = NULL;
        int bdepth = 0;

        for (const char *q = body; *q; q++)
        {
            if (*q == '{') bdepth++;
            else if (*q == '}')
            {
                if (--bdepth == 0) { bclose = q; break; }
            }
        }
        if (!bclose || bclose < cursor)
            { if (tess_trace()) fprintf(stderr, "MGL tess: capture_msl: fail point 7 (void body)\n"); free(k); k = NULL; goto done; }
        str_append(&k, &len, &cap, "%.*s", (int)(bclose - cursor), cursor);
        str_append(&k, &len, &cap,
                   "*(device float4*)((device uchar*)mgl_cap + (size_t)mgl_vid * %uu) = float4(0.0);\n",
                   tp->cap_stride * 4u);
        cursor = bclose;
    }
    str_append(&k, &len, &cap, "%s", cursor);

    /* the record struct was written for a vertex function: stage attributes such as
     * [[position]] / [[user(locn0)]] are not legal on a kernel local, strip them */
    {
        char head[160];
        const char *at;
        char *decl;

        snprintf(head, sizeof head, "struct %s\n{", cap_retname);
        at = strstr(k, head);
        if (!at) { snprintf(head, sizeof head, "struct %s {", cap_retname); at = strstr(k, head); }
        if (at)
        {
            char *ob = strchr(at, '{');
            char *cb = ob ? strchr(ob, '}') : NULL;

            if (ob && cb)
                for (char *a = ob; (a = strstr(a, "[[")) != NULL && a < cb; )
                {
                    char *e = strstr(a, "]]");
                    if (!e || e > cb)
                        break;
                    memmove(a, e + 2, strlen(e + 2) + 1);
                    cb -= (size_t)(e + 2 - a);
                    len -= (size_t)(e + 2 - a);
                }
            (void)decl;
        }
    }
    out = k;

done:
    if (!out)
    {
        tp->cap_members_n = 0;
        tp->cap_stride = 0;
    }
    else if (getenv("MGL_DUMP_MSL"))
    {
        FILE *df = fopen("/tmp/cap_tes.metal", "w");

        if (df)
        {
            fputs(out, df);
            fclose(df);
        }
        fprintf(stderr, "==== capture TES MSL (%u members, stride %u words) ====\n",
                tp->cap_members_n, tp->cap_stride);
        for (int i = 0; i < tp->cap_members_n && i < 12; i++)
            fprintf(stderr, "  capmember %-24s words=%u@%u\n", tp->cap_members[i].name,
                    tp->cap_members[i].words, tp->cap_members[i].offset);
    }
    free(plain);
    return out;
}

/* True when the current tessellated draw rasterizes through the plain TES
 * (MGLRenderer+Tessellation.m's emulated path: GPU tessellator + indirect
 * draw, or the CPU reference) instead of the hardware post-tessellation
 * pipeline.  Transform-feedback capture forces this path as well — the
 * captured records must be GPU-visible — and so do isolines / point_mode.
 * With a geometry shader the geometry pass-through renders instead, so the
 * plain TES is never the raster vertex function there. */
bool mgl_tess_plain_render(GLMContext ctx)
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    if (!tp || !tp->plain_function)
        return false;
    if (ctx->state.program && ctx->state.program->shader_slots[_GEOMETRY_SHADER])
        return false;
    return tp->domain == MGL_TESS_DOMAIN_ISOLINES || tp->point_mode || tp->capture ||
           S->cpu_render_now || mgl_tess_cpu_reference();
}

void mgl_tess_set_capture(GLMContext ctx, Program *program, bool enable){
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    MGLTessProgram *base;

    if (!tp)
        return;
    (void)program;
    tp->capture = enable;
    /* keep the base record in agreement: an active pair is rebuilt lazily (a tc
     * relink) and must not resurrect a stale capture flag */
    base = mgl_tess_program(ctx, tp->program);
    if (base && base != tp)
        base->capture = enable;
}

/* Compile + cache the compute-kernel variant of a SEPARABLE vs-only program's vertex
 * stage (produced by compile_stage from the same SPIR-V with the tessellation options).
 * The tessellation draw path finds it by program name when a pipeline supplies the
 * vertex stage.  Returns true when the kernel is available (now or cached). */
bool mgl_tess_cache_vs_kernel(GLMContext ctx, GLuint name, const char *msl, const char *entry)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    void *lib = NULL, *fn = NULL;
    char *err = NULL;
    unsigned slot;

    if (!S || !msl || !ctx->mtl_funcs.ext_program || !ctx->mtl_funcs.ext_program->compile_msl)
        return false;
    for (slot = 0; slot < S->vs_kernel_n; slot++)
        if (S->vs_kernel[slot].name == name)
        {
            /* a relink regenerates the stage: only a cached kernel with the SAME
             * MSL is reusable — a different MSL is a stale relink leftover and
             * must be recompiled (the old function is released below) */
            if (S->vs_kernel[slot].function &&
                S->vs_kernel[slot].msl && strcmp(S->vs_kernel[slot].msl, msl) == 0)
                return true;
            if (S->vs_kernel[slot].function && ctx->mtl_funcs.ext_program->release_obj)
            {
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[slot].library);
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[slot].function);
                S->vs_kernel[slot].library = NULL;
                S->vs_kernel[slot].function = NULL;
            }
            break;
        }
    if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, entry ? entry : "main", &lib, &fn, &err))
    {
        if (tess_trace())
        {
            FILE *df = fopen("/tmp/sepvs_fail.metal", "w");
            if (df) { fputs(msl, df); fclose(df); }
            fprintf(stderr, "MGL tess: separable vs-kernel compile failed: %s\n", err ? err : "?");
        }
        free(err);
        return false;
    }
    free(err);
    if (slot >= S->vs_kernel_n)
    {
        if (S->vs_kernel_n >= MGL_TESS_VS_KERNEL_CACHE)
        {
            /* more live vs-only programs than cache slots: evict the oldest (its
             * draw would rebuild a plain-vertex fallback — a degenerate case no
             * real workload hits; deleting programs compacts the cache) */
            if (ctx->mtl_funcs.ext_program->release_obj)
            {
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[0].library);
                ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[0].function);
            }
            free(S->vs_kernel[0].msl);
            free(S->vs_kernel[0].words);
            memmove(&S->vs_kernel[0], &S->vs_kernel[1],
                    (S->vs_kernel_n - 1) * sizeof S->vs_kernel[0]);
            memset(&S->vs_kernel[--S->vs_kernel_n], 0, sizeof S->vs_kernel[0]);
            slot = S->vs_kernel_n;
        }
        else
            slot = S->vs_kernel_n++;
        S->vs_kernel[slot].name = name;
    }
    S->vs_kernel[slot].library = lib;      /* owned (CFBridgingRetain) */
    S->vs_kernel[slot].function = fn;
    free(S->vs_kernel[slot].msl);
    S->vs_kernel[slot].msl = strdup(msl);  /* pair rewrites re-compile from it */
    snprintf(S->vs_kernel[slot].entry, sizeof S->vs_kernel[slot].entry, "%s",
             entry ? entry : "main0");
    free(S->vs_kernel[slot].words);
    S->vs_kernel[slot].words = NULL;
    S->vs_kernel[slot].words_n = 0;
    if (tess_trace())
        fprintf(stderr, "MGL tess: sep-vs kernel cached for program %u (slot %u fn=%p)\n",
                name, slot, fn);
    return true;
}

/* Cache a PREPARED SPIR-V module of a vs-only program for lazy kernel
 * compilation: a NON-separable vs-only program can still be paired with a
 * pipeline's TCS/TES at draw time (GL 4.6 §7.3 stage resolution — the bound
 * program wins for the stages it defines, the pipeline supplies the rest).
 * Compiling the kernel variant at link time for every program would double
 * link cost, so program.c stores the prepared module here and the kernel MSL
 * is generated on the first tessellation draw that actually needs it. */
bool mgl_tess_cache_vs_words(GLMContext ctx, GLuint name, const uint32_t *words, size_t nwords,
                             const char *spirv_entry)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    uint32_t *copy;
    unsigned slot;

    if (!S || !words || !nwords)
        return false;
    copy = (uint32_t *)malloc(nwords * sizeof(uint32_t));
    if (!copy)
        return false;
    memcpy(copy, words, nwords * sizeof(uint32_t));
    for (slot = 0; slot < S->vs_kernel_n; slot++)
        if (S->vs_kernel[slot].name == name)
        {
            free(S->vs_kernel[slot].words);
            S->vs_kernel[slot].words = copy;
            S->vs_kernel[slot].words_n = nwords;
            snprintf(S->vs_kernel[slot].spirv_entry, sizeof S->vs_kernel[slot].spirv_entry,
                     "%s", spirv_entry ? spirv_entry : "main");
            return true;                    /* relink refresh of the module */
        }
    if (S->vs_kernel_n >= MGL_TESS_VS_KERNEL_CACHE)
    {
        /* same eviction policy as the compiled cache above */
        if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
        {
            ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[0].library);
            ctx->mtl_funcs.ext_program->release_obj(ctx, S->vs_kernel[0].function);
        }
        free(S->vs_kernel[0].msl);
        free(S->vs_kernel[0].words);
        memmove(&S->vs_kernel[0], &S->vs_kernel[1],
                (S->vs_kernel_n - 1) * sizeof S->vs_kernel[0]);
        memset(&S->vs_kernel[--S->vs_kernel_n], 0, sizeof S->vs_kernel[0]);
    }
    slot = S->vs_kernel_n++;
    memset(&S->vs_kernel[slot], 0, sizeof S->vs_kernel[slot]);
    S->vs_kernel[slot].name = name;
    S->vs_kernel[slot].words = copy;
    S->vs_kernel[slot].words_n = nwords;
    snprintf(S->vs_kernel[slot].spirv_entry, sizeof S->vs_kernel[slot].spirv_entry,
             "%s", spirv_entry ? spirv_entry : "main");
    if (tess_trace())
        fprintf(stderr, "MGL tess: sep-vs module cached (lazy) for program %u (slot %u)\n",
                name, slot);
    return true;
}

/* read a program's cached prepared module back (program.c's link cache keeps it
 * across program deletion so a cache-hit relink can re-prime this cache —
 * names are reused, the by-name entries are dropped on program free) */
bool mgl_tess_vs_words_get(GLMContext ctx, GLuint name, const uint32_t **words, size_t *nwords,
                           const char **entry)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    if (!S)
        return false;
    for (unsigned slot = 0; slot < S->vs_kernel_n; slot++)
        if (S->vs_kernel[slot].name == name && S->vs_kernel[slot].words)
        {
            if (words)
                *words = S->vs_kernel[slot].words;
            if (nwords)
                *nwords = S->vs_kernel[slot].words_n;
            if (entry)
                *entry = S->vs_kernel[slot].spirv_entry;
            return true;
        }
    return false;
}

const char *mgl_tess_vs_kernel_msl(GLMContext ctx, GLuint name, const char **entry)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    if (!S)
        return NULL;
    for (unsigned slot = 0; slot < S->vs_kernel_n; slot++)
    {
        if (S->vs_kernel[slot].name != name)
            continue;
        if (S->vs_kernel[slot].msl)
        {
            if (entry)
                *entry = S->vs_kernel[slot].entry;
            return S->vs_kernel[slot].msl;
        }
        if (S->vs_kernel[slot].words)
        {
            /* lazily build the kernel MSL of a NON-separable vs-only program on
             * its first tessellation draw (program.c cached the prepared module
             * at link — see mgl_tess_cache_vs_words) */
            char kentry[128];
            char *msl = mgl_program_tess_kernel_msl(ctx, S->vs_kernel[slot].words,
                                                    S->vs_kernel[slot].words_n,
                                                    S->vs_kernel[slot].spirv_entry, kentry);
            if (msl)
            {
                mgl_tess_cache_vs_kernel(ctx, name, msl, kentry);  /* takes over the slot */
                free(msl);
                /* the cache may have compacted (eviction) — re-resolve by name */
                for (unsigned s2 = 0; s2 < S->vs_kernel_n; s2++)
                    if (S->vs_kernel[s2].name == name && S->vs_kernel[s2].msl)
                    {
                        if (entry)
                            *entry = S->vs_kernel[s2].entry;
                        return S->vs_kernel[s2].msl;
                    }
            }
            else if (tess_trace())
                fprintf(stderr, "MGL tess: lazy sep-vs kernel MSL generation failed "
                                "(program %u)\n", name);
        }
        return NULL;
    }
    return NULL;
}

const MGLTessCapture *mgl_tess_get_capture(GLMContext ctx)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;

    return (S && S->capture.valid) ? &S->capture : NULL;
}

void mgl_tess_note_draw(GLMContext ctx, GLuint patches, GLuint tes_invocations, GLuint tessellated_primitives)
{
    if (tess_trace())
        fprintf(stderr, "MGL tess: note_draw patches=%u tes_inv=%u prims=%u\n",
                patches, tes_invocations, tessellated_primitives);
    mgl_queries_note_tess(ctx, tes_invocations, tessellated_primitives);
}

/* the TCS kernel's two record struct names: the types of the spvIn / spvOut
 * parameters of its entry point */
static bool msl_tcs_record_names(const char *msl, char *in_name, size_t in_len,
                                 char *out_name, size_t out_len)
{
    bool got_in = false, got_out = false;
    const char *at = msl;

    while ((at = strstr(at, "device ")) != NULL && !(got_in && got_out))
    {
        const char *star = strchr(at, '*');
        const char *b = at + strlen("device ");
        const char *nl = strchr(at, '\n');

        /* "device const T* spvIn" / "device T* spvOut" (kernel parameter list) */
        if (star && nl && star < nl)
        {
            const char *param = star + 1;
            bool is_in, is_out;
            const char *e = star;

            if (!strncmp(b, "const ", 6))
                b += 6;                 /* device const T* spvIn */
            while (param < at + strlen(at) && *param == ' ')
                param++;
            is_in = !strncmp(param, "spvIn", 5);
            is_out = !strncmp(param, "spvOut", 6);
            while (e > b && *(e - 1) == ' ')
                e--;
            if ((is_in || is_out) && e > b)
            {
                char name[128];
                snprintf(name, sizeof name, "%.*s", (int)(e - b), b);
                if (is_in && !got_in)
                {
                    snprintf(in_name, in_len, "%s", name);
                    got_in = true;
                }
                else if (is_out && !got_out)
                {
                    snprintf(out_name, out_len, "%s", name);
                    got_out = true;
                }
            }
        }
        at += strlen("device ");
    }
    /* a TCS that reads NONE of its control point inputs gets no spvIn parameter at
     * all (SPIRV-Cross drops the unused buffer): the OUTPUT record must still be
     * canonicalized — return success with an empty in-name, the caller then skips
     * the (nonexistent) input rewrite only */
    if (!got_in && got_out)
    {
        in_name[0] = 0;
        return true;
    }
    return got_in && got_out;
}

/* ------------------------------------------------------------------ record canonicalization
 * Parses every stage's interface struct, builds the canonical control-point and
 * per-patch records (union by varying location — see "canonical records" above),
 * rewrites the producer MSL structs to them and recompiles those kernels. Consumers
 * read the records through the canonical layout: the TCS kernel through its rewritten
 * spvIn/spvOut structs, the plain TES through its rewritten control-point struct and
 * blob offsets, the hardware TES through the vertex descriptor offsets (tp->tes_cp /
 * tp->tes_patch_in are already canonical after this call). */
static void tess_canonicalize_records(GLMContext ctx, MGLTessProgram *tp)
{
    Program *program = tp->program;
    const char *vs_msl = program->spirv[_VERTEX_SHADER].msl_str;
    const char *tcs_msl = tp->has_tcs ? program->spirv[_TESS_CONTROL_SHADER].msl_str : NULL;
    if (tcs_msl && getenv("MGL_DUMP_TCS"))
    {
        char fn[128];

        snprintf(fn, sizeof fn, "%s", getenv("MGL_DUMP_TCS"));
        FILE *f = fopen(fn, "w");

        if (f) { fputs(tcs_msl, f); fclose(f); }
    }
    const char *tes_msl = program->spirv[_TESS_EVALUATION_SHADER].msl_str;

    struct MslStruct vs_out, tcs_in, tcs_out, tes_cp, tcs_po, tes_pi;
    char vs_name[160] = "", tcs_in_name[160] = "", tcs_out_name[160] = "";
    char po_name[160] = "", pi_name[160] = "", cp_name[160] = "";
    bool have_vs = false, have_tcs_in = false, have_tcs_out = false;
    bool have_tcs_po = false, have_pi = false, have_cp = false;
    struct CanonRec canon, canon_tcs, canon_patch;

    memset(&vs_out, 0, sizeof vs_out);
    memset(&tcs_in, 0, sizeof tcs_in);
    memset(&tcs_out, 0, sizeof tcs_out);
    memset(&tes_cp, 0, sizeof tes_cp);
    memset(&tcs_po, 0, sizeof tcs_po);
    memset(&tes_pi, 0, sizeof tes_pi);

    if (vs_msl && tess_cp_record_name(vs_msl, vs_name, sizeof vs_name))
        have_vs = msl_parse_struct(vs_msl, vs_name, &vs_out);
    if (tcs_msl)
    {
        if (msl_tcs_record_names(tcs_msl, tcs_in_name, sizeof tcs_in_name,
                                 tcs_out_name, sizeof tcs_out_name))
        {
            have_tcs_in = tcs_in_name[0] &&
                          msl_parse_struct(tcs_msl, tcs_in_name, &tcs_in);
            have_tcs_out = msl_parse_struct(tcs_msl, tcs_out_name, &tcs_out);
        }
        have_tcs_po = msl_parse_struct_entry(tcs_msl, &tcs_po, po_name, sizeof po_name);
    }
    if (tes_msl)
    {
        if (msl_patch_in_name(tes_msl, pi_name, sizeof pi_name))
            have_pi = msl_parse_struct(tes_msl, pi_name, &tes_pi);
        if (have_pi && msl_cp_struct_name(tes_msl, pi_name, cp_name, sizeof cp_name))
            have_cp = msl_parse_struct(tes_msl, cp_name, &tes_cp);
    }

    /* attach varying locations to every member */
    if (getenv("MGL_TESS_DEBUG2") && have_tcs_po)
        for (int i = 0; i < tcs_po.count; i++)
            fprintf(stderr, "MGL tess tcs_po[%d] '%s' type '%s' @%zu (%zu b)\n",
                    i, tcs_po.members[i].name, tcs_po.members[i].type,
                    tcs_po.members[i].offset, tcs_po.members[i].size);
    tess_annotate_keys(ctx, program, _VERTEX_SHADER, &vs_out, have_vs ?
        &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, program, _VERTEX_SHADER, &tcs_in, have_tcs_in ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, program, _TESS_CONTROL_SHADER, &tcs_out, have_tcs_out ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    /* the TES control-point struct is annotated against the interface's PRODUCER
     * module: member names are stripped to the bare GL name by checking the module
     * that declares the block (the TCS normally).  Without a TCS the producer is the
     * vertex stage — passing the (absent) TCS module here left "inVertex_iv"
     * unstripped, so it hashed under its full name while the VS's "outVertex_iv"
     * hashed as "iv": the two never aliased in the canonical record and the TES
     * read zeros (single.max_patch_vertices case 2). */
    tess_annotate_keys(ctx, program, tp->has_tcs ? _TESS_CONTROL_SHADER : _VERTEX_SHADER,
        &tes_cp, have_cp ?
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, program, _TESS_CONTROL_SHADER, &tcs_po, have_tcs_po ?
        &program->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, program, _TESS_EVALUATION_SHADER, &tes_pi, have_pi ?
        &program->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);

    /* pair flattened interface-block members that no reflected name matched.
     * KNOWN-INCOMPLETE (round 5): with the pairing active the canonical-record
     * build of the plain TES still emits invalid MSL (heap garbage in the
     * epilogue — "MSL source is not valid UTF-8"); the values do not cross
     * stages yet, so it is off by default.  Enable with MGL_TESS_PAIR_BLOCKS=1
     * to resume debugging: the pairing itself works (see the trace), the
     * overflow is in the canon path it enables (rewrite_tes epilogue). */
    if (getenv("MGL_TESS_PAIR_BLOCKS"))
    {
        if (have_tcs_in)  tess_pair_iface_members(&vs_out, &tcs_in);
        else if (have_cp) tess_pair_iface_members(&vs_out, &tes_cp);
        if (have_cp)      tess_pair_iface_members(&tcs_out, &tes_cp);
    }

    /* propagate the keys to the member copies stored in tp (filled earlier at link
     * time from the same structs, without annotations) */
    {
        struct MslStruct *parsed[3] = { have_tcs_po ? &tcs_po : NULL, have_pi ? &tes_pi : NULL,
                                        have_cp ? &tes_cp : NULL };
        for (int s = 0; s < 3; s++)
        {
            if (!parsed[s])
                continue;
            for (int i = 0; i < parsed[s]->count; i++)
            {
                MGLTessPatchMember *src = &parsed[s]->members[i];
                MGLTessPatchMember *dstarr[3] = { NULL, NULL, NULL };
                int ndst = 0, ndstmax[3] = { 0, 0, 0 };

                if (parsed[s] == &tcs_po)
                {
                    dstarr[0] = tp->tcs_patch_out;
                    ndstmax[0] = tp->num_tcs_patch_out;
                    ndst = 1;
                }
                else if (parsed[s] == &tes_pi)
                {
                    dstarr[0] = tp->tes_patch_in;
                    ndstmax[0] = tp->num_tes_patch_in;
                    ndst = 1;
                }
                else
                {
                    dstarr[0] = tp->tes_cp;
                    ndstmax[0] = tp->num_tes_cp;
                    ndst = 1;
                }
                for (int d = 0; d < ndst; d++)
                    for (int j = 0; j < ndstmax[d]; j++)
                        if (!strcmp(dstarr[d][j].name, src->name) &&
                            !strcmp(dstarr[d][j].type, src->type))
                        {
                            dstarr[d][j].key = src->key;
                            dstarr[d][j].array_idx = src->array_idx;
                            dstarr[d][j].array_idx2 = src->array_idx2;
                            dstarr[d][j].array_idx3 = src->array_idx3;
                            dstarr[d][j].flat_dims = src->flat_dims;
                            break;
                        }
            }
        }
    }

    if (tess_trace())
    {
        struct { const char *tag; struct MslStruct *s; } dumps[] = {
            { "vs_out", have_vs ? &vs_out : NULL },
            { "tcs_in", have_tcs_in ? &tcs_in : NULL },
            { "tcs_out", have_tcs_out ? &tcs_out : NULL },
            { "tes_cp", have_cp ? &tes_cp : NULL },
            { "tcs_po", have_tcs_po ? &tcs_po : NULL },
            { "tes_pi", have_pi ? &tes_pi : NULL },
        };
        for (size_t d = 0; d < sizeof(dumps) / sizeof(dumps[0]); d++)
        {
            if (!dumps[d].s)
                continue;
            fprintf(stderr, "MGL tess: stage struct %s (%d members):\n", dumps[d].tag, dumps[d].s->count);
            if (dumps[d].s->body)
                fprintf(stderr, "MGL tess:   body: %s\n", dumps[d].s->body);
            for (int i = 0; i < dumps[d].s->count; i++)
                fprintf(stderr, "MGL tess:   %-14s %-24s key=%08x idx=%u off=%zu\n",
                        dumps[d].s->members[i].type, dumps[d].s->members[i].name,
                        dumps[d].s->members[i].key, dumps[d].s->members[i].array_idx,
                        dumps[d].s->members[i].offset);
        }
    }

    /* ---- canonical control point records: one per interface ----
     * vertex -> TCS: union of the VS output struct and the TCS input struct
     * TCS -> TES:    union of the TCS output struct and the TES control point struct
     * (no TCS: the vertex kernel's output feeds the TES directly, one interface)
     * The pair is keyed by varying location, so a consumer that drops or reorders
     * members never shifts the record its producer writes. */
    if (tp->has_tcs)
    {
        struct MslStruct *vs_side[2] = { have_vs ? &vs_out : NULL, have_tcs_in ? &tcs_in : NULL };
        struct MslStruct *tes_side[2] = { have_tcs_out ? &tcs_out : NULL, have_cp ? &tes_cp : NULL };

        if (!tess_canon_union(&canon, vs_side, 2) ||
            !tess_canon_union(&canon_tcs, tes_side, 2))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: canonical record union failed (vs=%d tcs=%d); layouts kept per-stage\n",
                        canon.n, canon_tcs.n);
            return;
        }
    }
    else
    {
        struct MslStruct *stages[2] = { have_vs ? &vs_out : NULL, have_cp ? &tes_cp : NULL };

        if (!tess_canon_union(&canon, stages, 2))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: canonical record union failed; layouts kept per-stage\n");
            return;
        }
        canon_tcs = canon;
    }
    /* ---- canonical per-patch record: union of the TCS patch outputs and TES patch
     * inputs (tessellation levels included: the blob and the vertex descriptor read
     * them from the factor record, but they occupy the same slot here) */
    {
        struct MslStruct *stages[2] = { have_tcs_po ? &tcs_po : NULL, have_pi ? &tes_pi : NULL };
        if (!tess_canon_union(&canon_patch, stages, 2))
            return;
    }

    if (tess_trace())
    {
        fprintf(stderr, "MGL tess: canonical vs->tcs record (%d members, %zu bytes):\n",
                canon.n, canon.size);
        for (int i = 0; i < canon.n; i++)
            fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                    canon.m[i].key, canon.m[i].type, canon.m[i].name,
                    canon.m[i].offset, canon.m[i].size);
        if (tp->has_tcs)
        {
            fprintf(stderr, "MGL tess: canonical tcs->tes record (%d members, %zu bytes):\n",
                    canon_tcs.n, canon_tcs.size);
            for (int i = 0; i < canon_tcs.n; i++)
                fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                        canon_tcs.m[i].key, canon_tcs.m[i].type, canon_tcs.m[i].name,
                        canon_tcs.m[i].offset, canon_tcs.m[i].size);
        }
        fprintf(stderr, "MGL tess: canonical patch record (%d members, %zu bytes):\n",
                canon_patch.n, canon_patch.size);
        for (int i = 0; i < canon_patch.n; i++)
            fprintf(stderr, "MGL tess:   key=%08x %-34s %-22s @%zu (%zu b)\n",
                    canon_patch.m[i].key, canon_patch.m[i].type, canon_patch.m[i].name,
                    canon_patch.m[i].offset, canon_patch.m[i].size);
    }

    /* the control point stream the TES reads: the TCS's output record, or the vertex
     * kernel's output record when no TCS runs */
    {
        const struct CanonRec *rec = tp->has_tcs ? &canon_tcs : &canon;
        size_t malign = 0, msize = 0;

        tp->num_cp_record = rec->n;
        tp->cp_record_full_size = rec->size;
        tp->cp_record_size = rec->size;
        /* the vertex kernel's own record (the VS->TCS interface) — see the header:
         * with a TCS it differs from the TCS-out record the TES reads */
        tp->vs_record_size = canon.size;
        for (int i = 0; i < rec->n && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
            tp->cp_record[i] = rec->m[i];

        /* TES control point members read at the canonical offsets (attribute indices
         * unchanged: they belong to the TES's own struct). Flattened elements of an
         * arrayed varying point into the array member's element slots; a flattened
         * MATRIX COLUMN has its own canonical slot (the parse split the columns) */
        for (int i = 0; i < tp->num_tes_cp; i++)
        {
            for (int j = 0; j < tp->num_cp_record; j++)
                if (tp->tes_cp[i].key == rec->m[j].key &&
                    /* the record splits matrix columns into "_0"/"_1" entries with the
                     * same key: take THIS member's own slot, not the first match; a
                     * FLATTENED ARRAY ELEMENT (flat_dims) takes the whole-array slot —
                     * the first match — and steps inside it by the stride (only when
                     * that slot really is an array: a split matrix column has the same
                     * flat_dims and must keep matching its own column slot) */
                    (tp->tes_cp[i].array_idx == 0 || rec->m[j].array_idx == tp->tes_cp[i].array_idx ||
                     !strcmp(rec->m[j].name, tp->tes_cp[i].name) ||
                     (tp->tes_cp[i].flat_dims && strstr(rec->m[j].type, "spvUnsafeArray"))))
                {
                    tp->tes_cp[i].offset = rec->m[j].offset;
                    if (tp->tes_cp[i].flat_dims && strstr(rec->m[j].type, "spvUnsafeArray"))
                    {
                        size_t stride = msl_array_stride(rec->m[j].type);

                        if (stride)
                            tp->tes_cp[i].offset = rec->m[j].offset +
                                tp->tes_cp[i].array_idx * stride;
                        if (stride && tp->tes_cp[i].array_idx2)
                        {
                            char inner[96];
                            long n2 = 0;

                            if (msl_array_inner(rec->m[j].type, inner, sizeof inner, &n2) &&
                                msl_array_stride(inner))
                                tp->tes_cp[i].offset += tp->tes_cp[i].array_idx2 *
                                    msl_array_stride(inner);
                        }
                    }
                    else if (tp->tes_cp[i].array_idx &&
                             msl_matrix_type(rec->m[j].type, &malign, &msize))
                    {
                        /* a flattened column of a matrix member: float<C>x<R> has C
                         * columns, each msize/C bytes (tes_cp[i].size is the column) */
                        int cols = rec->m[j].type[5] - '0';

                        if (cols >= 2 && cols <= 4)
                            tp->tes_cp[i].offset = rec->m[j].offset +
                                tp->tes_cp[i].array_idx * (msize / (size_t)cols);
                    }
                    else if (tp->tes_cp[i].array_idx == rec->m[j].array_idx)
                    {
                        /* an exact (key, column) slot: the canonical record itself is
                         * split into columns (msl_parse_struct) */
                        tp->tes_cp[i].offset = rec->m[j].offset;
                    }
                    break;
                }
            if (tess_trace())
                fprintf(stderr, "MGL tess: tes_cp[%d] %-22s attr=%d key=%08x idx=%u -> offset %zu\n",
                        i, tp->tes_cp[i].name, tp->tes_cp[i].attr, tp->tes_cp[i].key,
                        tp->tes_cp[i].array_idx, tp->tes_cp[i].offset);
        }
    }

    /* TES patch-in members at canonical offsets (levels too: the blob is laid out
     * this way; the hardware descriptor overrides the level members to the factor
     * buffer at draw time) */
    for (int i = 0; i < tp->num_tes_patch_in; i++)
    {
        int best = -1, exact = 0;

        for (int j = 0; j < canon_patch.n; j++)
            if (tp->tes_patch_in[i].key == canon_patch.m[j].key)
            {
                /* flattened array elements share one key ("name_N" carries
                 * array_idx N): prefer the canonical element of the SAME
                 * index — a dissolved/flattened patch block array has one
                 * canonical member per element, and taking the first would
                 * read element 0 for all of them (mirrors the control-point
                 * fixup above) */
                if (tp->tes_patch_in[i].array_idx == canon_patch.m[j].array_idx)
                {
                    best = j;
                    exact = 1;
                    break;
                }
                if (best < 0)
                    best = j;    /* fallback: array base (spvUnsafeArray) or
                                  * an unmatched element index */
            }
        if (best >= 0)
        {
            int j = best;

            tp->tes_patch_in[i].offset = canon_patch.m[j].offset;
                /* flattened array element ("name_N", array_idx N) of a canonical
                 * spvUnsafeArray member: advance by the element stride (mirrors the
                 * control-point fixup above — without it every element read at the
                 * array's base offset, and 'patch in int r[16]' returned r[0] always).
                 * Only for a BASE match — an exact element match already carries its
                 * own offset.  A struct-array FIELD element carries a second dimension
                 * ("name_<i>_<field>_<j>", array_idx2) stepping inside the field */
                if (tp->tes_patch_in[i].flat_dims && strstr(canon_patch.m[j].type, "spvUnsafeArray"))
                {
                    size_t stride = msl_array_stride(canon_patch.m[j].type);

                    if (getenv("MGL_TESS_DEBUG2"))
                        fprintf(stderr, "MGL tess: stride fixup '%s' canon '%s %s' dims=%u idx=%u idx2=%u idx3=%u stride=%zu exact=%d\n",
                                tp->tes_patch_in[i].name, canon_patch.m[j].type,
                                canon_patch.m[j].name, tp->tes_patch_in[i].flat_dims,
                                tp->tes_patch_in[i].array_idx, tp->tes_patch_in[i].array_idx2,
                                tp->tes_patch_in[i].array_idx3, stride, exact);
                    /* the FIRST dimension steps the whole array member — an
                     * exact element match already carries its own offset */
                    if (!exact && stride)
                        tp->tes_patch_in[i].offset = canon_patch.m[j].offset +
                            tp->tes_patch_in[i].array_idx * stride;
                    /* the SECOND and THIRD dimensions step INSIDE the array
                     * member (a struct-array field element, and the innermost
                     * element of a dissolved patch-block ARRAY field): these
                     * apply for base AND exact matches alike */
                    if (stride)
                    {
                        char inner[96];
                        long n2 = 0;
                        size_t s_in = 0;

                        /* A 1-D slot ("spvUnsafeArray<float, 2>") has a
                         * NON-array inner type, so msl_array_stride(inner) is
                         * 0 — its element stride IS the slot stride
                         * ("pb_ps_0_z_1" must step 4 bytes, not 0) */
                        if (msl_array_inner(canon_patch.m[j].type, inner, sizeof inner, &n2) &&
                            msl_array_stride(inner))
                            s_in = msl_array_stride(inner);
                        else
                            s_in = stride;
                        if (tp->tes_patch_in[i].array_idx2)
                            tp->tes_patch_in[i].offset += tp->tes_patch_in[i].array_idx2 * s_in;
                        if (tp->tes_patch_in[i].array_idx3)
                        {
                            char inner2[96];
                            long n3 = 0;
                            size_t s_in2 = 0;

                            if (s_in != stride &&
                                msl_array_inner(inner, inner2, sizeof inner2, &n3) &&
                                msl_array_stride(inner2))
                                s_in2 = msl_array_stride(inner2);
                            else
                                s_in2 = s_in;
                            tp->tes_patch_in[i].offset += tp->tes_patch_in[i].array_idx3 * s_in2;
                        }
                    }
                }
        }
        if (tess_trace())
            fprintf(stderr, "MGL tess: tes_patch_in[%d] '%s' key=%x idx=%u off=%zu (stride %zu)\n",
                    i, tp->tes_patch_in[i].name, tp->tes_patch_in[i].key,
                    tp->tes_patch_in[i].array_idx, tp->tes_patch_in[i].offset, canon_patch.size);
    }
    tp->tes_patch_in_size = canon_patch.size;
    if (canon_patch.size)
        tp->tcs_patch_out_size = canon_patch.size;  /* the TCS patchOut is rewritten below */

    /* the TCS patch output members (the rewritten kernel's canonical patchOut struct)
     * sit at the canonical offsets as well — same slot rule as the TES patch-ins
     * above: exact name, else canonical key + flattened element index (a flattened
     * per-element producer must carry ITS OWN offset or every element copies
     * element 0 through the blob map) */
    for (int i = 0; i < tp->num_tcs_patch_out; i++)
    {
        int best = -1;

        for (int j = 0; j < canon_patch.n; j++)
            if (tp->tcs_patch_out[i].key == canon_patch.m[j].key)
            {
                /* same key on both: prefer the canonical slot of the SAME
                 * element index (a dissolved patch-block ARRAY keeps one slot
                 * per element — "pb_ps_0_z"/"pb_ps_1_z" — and the first match
                 * would write element 1 into element 0's slot) */
                if (tp->tcs_patch_out[i].array_idx == canon_patch.m[j].array_idx)
                {
                    best = j;
                    break;
                }
                if (best < 0)
                    best = j;
            }
        if (best >= 0)
            tp->tcs_patch_out[i].offset = canon_patch.m[best].offset;
    }

    /* ---- producer rewrites: whichever kernel writes a record whose SPIRV-Cross
     * struct differs from the canonical layout is recompiled with the canonical
     * struct (member names it uses are kept; pads are never referenced) */
    {
        char body[4096];

        if (have_vs && !tess_canon_matches(&vs_out, &canon))
        {
            char *nw;
            tess_canon_body(body, sizeof body, &vs_out, &canon);
            nw = msl_rewrite_struct_body(vs_msl, vs_name, body);
            /* scalar-double outputs: uint2 store in the VS body, uint4 slot in
             * the canonical record — widen the stores before compiling */
            if (nw)
            {
                tess_widen_double_stores(&nw, &vs_out, &canon);
                /* struct members flattened by the canonical record (the TES's
                 * scalar-only attributes): the kernel body's ".m.f" accesses
                 * follow the rewritten member names */
                tess_flatten_body_access(nw, &vs_out, &canon);
            }
            if (getenv("MGL_DUMP_MSL") && nw)
                fprintf(stderr, "==== tess VS canonical MSL:\n%s\n==== end\n", nw);
            if (nw && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, nw,
                        program->spirv[_VERTEX_SHADER].entry_point, &lib, &fn, &err))
                {
                    tp->vs_function = fn;
                    tp->vs_library = lib;
                    if (tess_trace())
                        fprintf(stderr, "MGL tess: VS kernel canonicalized (record %zu b)\n",
                                canon.size);
                }
                else
                {
                    fprintf(stderr, "MGL ERROR: tess: VS canonical-record recompile failed: %s\n",
                            err ? err : "?");
                    free(err);
                }
            }
            free(nw);
        }

        if (tp->has_tcs && tcs_msl)
        {
            char *msl = strdup(tcs_msl);
            bool changed = false;

            if (have_tcs_in && !tess_canon_matches(&tcs_in, &canon))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_in, &canon);
                nw = msl_rewrite_struct_body(msl, tcs_in_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }
            if (have_tcs_out && !tess_canon_matches(&tcs_out, &canon_tcs))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_out, &canon_tcs);
                nw = msl_rewrite_struct_body(msl, tcs_out_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }
            if (have_tcs_po && !tess_canon_matches(&tcs_po, &canon_patch))
            {
                char *nw;
                tess_canon_body(body, sizeof body, &tcs_po, &canon_patch);
                nw = msl_rewrite_struct_body(msl, po_name, body);
                if (nw) { free(msl); msl = nw; changed = true; }
            }
            /* the struct-body rewrite above flattens struct members the canonical
             * record splits (the TES's scalar-only attributes); the kernel body's
             * "<expr>.member.field" accesses must follow or the rewrite fails to
             * compile and the draw falls back with garbage varyings */
            /* NOT a || chain: it short-circuits after the first pass that
             * changed something and the patchOut accesses would keep the
             * pre-canonical member paths (TCS recompile fails, nothing draws) */
            {
                bool fl = false;

                fl |= tess_flatten_body_access(msl, &tcs_in, &canon);
                fl |= tess_flatten_body_access(msl, &tcs_out, &canon_tcs);
                fl |= tess_flatten_body_access(msl, &tcs_po, &canon_patch);
                if (fl)
                    changed = true;
            }
            /* triangle domains: write MTLTriangleTessellationFactorsHalf (8-byte factor
             * records) instead of the quad struct (see mgl_tess_factor_stride) */
            if (tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
            {
                char *step = strstr(msl, "device MTLQuadTessellationFactorsHalf*");
                if (step)
                {
                    size_t oldlen = strlen("device MTLQuadTessellationFactorsHalf*");
                    size_t newlen = strlen("device MTLTriangleTessellationFactorsHalf*");
                    /* remember the offset before the realloc: realloc may move the
                     * block and `step` would then dangle (use-after-free read, and on a
                     * moved block a corrupted kernel source) */
                    size_t at = (size_t)(step - msl);
                    char *grown = (char *)realloc(msl, strlen(msl) + newlen - oldlen + 1);
                    if (grown)
                    {
                        char *p;
                        msl = grown;
                        p = msl + at;
                        memmove(p + newlen, p + oldlen, strlen(p + oldlen) + 1);
                        memcpy(p, "device MTLTriangleTessellationFactorsHalf*", newlen);
                        for (char *w = msl; (w = strstr(w, ".insideTessellationFactor[0]")) != NULL; )
                        {
                            size_t tail = strlen(w + strlen(".insideTessellationFactor[0]")) + 1;
                            memmove(w + strlen(".insideTessellationFactor"),
                                    w + strlen(".insideTessellationFactor[0]"), tail);
                            w += strlen(".insideTessellationFactor");
                        }
                        /* gl_TessLevelInner[1] is ignored for triangles (GL 4.6 11.2.3):
                         * keep its store off the scalar inside factor */
                        msl = tcs_redirect_tri_inner1(msl);
                        /* gl_TessLevelOuter[3] is unused for triangles and the factor record
                         * only has three edges: drop such stores (keeping the right-hand
                         * side's side effects) */
                        for (char *w = msl; (w = strstr(w, ".edgeTessellationFactor[3]")) != NULL; )
                        {
                            char *ls = w, *semi = strchr(w, ';'), *eq, *rhs;
                            char repl[512];
                            while (ls > msl && ls[-1] != '\n' && ls[-1] != ';' &&
                                   ls[-1] != '{' && ls[-1] != '}')
                                ls--;
                            if (!semi) { w += strlen(".edgeTessellationFactor"); continue; }
                            eq = semi;
                            while (eq > ls && eq[-1] != '=') eq--;
                            if (eq == ls) { w = semi; continue; }   /* not a plain assignment */
                            rhs = eq;
                            while (*rhs == ' ' || *rhs == '=') rhs++;
                            snprintf(repl, sizeof repl, "(void)(%.*s);", (int)(semi - rhs), rhs);
                            size_t span = (size_t)(semi - ls + 1);
                            memset(ls, ' ', span);
                            memcpy(ls, repl, strlen(repl) <= span ? strlen(repl) : span);
                            w = ls + span;
                        }
                        changed = true;
                    }
                }
            }

            {
                /* GL's exact float levels: the half factor record alone quantises them
                 * (2.7 -> 2.69922 in the TES); add the TCS-side float copy.  This runs
                 * even when the record layout needed no rewriting — a levels-only TCS
                 * has no patchOut struct to canonicalize, but still must write floats.
                 * (Runs BEFORE the per-instance gl_PrimitiveID rewrite: it matches the
                 * SPIRV-Cross store spelling `spvTessLevel[gl_PrimitiveID].`.) */
                char *lvl = tcs_inject_float_levels(msl);

                if (lvl) { free(msl); msl = lvl; tp->tcs_levels_float = 1; changed = true; }
            }

            {
                /* per-instance gl_PrimitiveID (a third indirect-params word carries the
                 * patches-per-instance count; see the dispatcher) */
                char *pd = tcs_patch_primid(msl);

                if (pd) { free(msl); msl = pd; changed = true; }
            }
            if (changed && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                if (getenv("MGL_DUMP_MSL"))
                    fprintf(stderr, "==== canonical TCS MSL:\n%s\n==== end\n", msl);
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl,
                        program->spirv[_TESS_CONTROL_SHADER].entry_point, &lib, &fn, &err))
                {
                    tp->tcs_function = fn;
                    tp->tcs_library = lib;
                    if (tess_trace())
                        fprintf(stderr, "MGL tess: TCS kernel canonicalized (cp %zu b, patch %zu b)\n",
                                canon.size, canon_patch.size);
                }
                else
                {
                    fprintf(stderr, "MGL ERROR: tess: TCS canonical-record recompile failed: %s\n",
                            err ? err : "?");
                    free(err);
                }
            }
            free(msl);
        }
    }
}

/* ------------------------------------------------------------------ pipeline pairs */

/* release a derived pipeline-pair record (never in the program hash) */
static void tess_free_pair(GLMContext ctx, MGLTessProgram *pair)
{
    if (!pair)
        return;
    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
    {
        void **owned[8] = { &pair->capture_function, &pair->capture_library, &pair->capture_pso,
                            &pair->plain_function, &pair->tcs_function, &pair->tcs_library,
                            &pair->vs_function, &pair->vs_library };
        void *extra[2] = { pair->plain_library, pair->tes_hw_function };
        void *extra2 = pair->tes_hw_library;

        for (int i = 0; i < 8; i++)
            if (*owned[i])
                ctx->mtl_funcs.ext_program->release_obj(ctx, *owned[i]);
        for (int i = 0; i < 2; i++)
            if (extra[i])
                ctx->mtl_funcs.ext_program->release_obj(ctx, extra[i]);
        if (extra2)
            ctx->mtl_funcs.ext_program->release_obj(ctx, extra2);
    }
    free(pair->plain_msl);
    free(pair);
}

/* Build the derived record of a (TES program, tc-only program) pipeline pair: a copy
 * of the TES program's own record with has_tcs=1 and the interface layouts
 * canonicalized across BOTH programs — the union of the tc program's TCS structs and
 * the TES's patch-in/cp structs, exactly like a single-program link does it
 * (tess_canonicalize_records), but with the structs coming from two SPIR-V modules.
 * The TCS kernel is recompiled against those layouts (plus the domain factor fixups
 * and the GL float-level copy); the VS kernel gets a pair variant when its raw
 * output record differs. */
static MGLTessProgram *tess_build_pair(GLMContext ctx, MGLTessProgram *tp, Program *tc)
{
    MGLTessProgram *pair;
    Program *tes = tp->program;
    Program *vs_prog = NULL;
    const char *vs_msl = NULL, *tcs_msl = NULL, *tes_msl = NULL;
    struct MslStruct vs_out, tcs_in, tcs_out, tes_cp, tcs_po, tes_pi;
    char vs_name[160] = "", tcs_in_name[160] = "", tcs_out_name[160] = "";
    char po_name[160] = "", pi_name[160] = "", cp_name[160] = "";
    bool have_vs = false, have_tcs_in = false, have_tcs_out = false;
    bool have_tcs_po = false, have_pi = false, have_cp = false;
    struct CanonRec canon, canon_tcs, canon_patch;

    memset(&vs_out, 0, sizeof vs_out);
    memset(&tcs_in, 0, sizeof tcs_in);
    memset(&tcs_out, 0, sizeof tcs_out);
    memset(&tes_cp, 0, sizeof tes_cp);
    memset(&tcs_po, 0, sizeof tcs_po);
    memset(&tes_pi, 0, sizeof tes_pi);

    tcs_msl = tc->spirv[_TESS_CONTROL_SHADER].msl_str;
    tes_msl = tes->spirv[_TESS_EVALUATION_SHADER].msl_str;
    if (!tcs_msl || !tes_msl)
        return NULL;

    /* the vertex stage of the same draw (GL 4.6 §7.3 stage resolution, exactly
     * like extTessDrawPatches does it): a program bound with glUseProgram wins
     * for the stages it defines, the pipeline supplies the rest.  KHR-GL46
     * tessellation utilities draw with useProgram() while an earlier run's
     * pipeline is still bound — the pipeline's stale VS slot must not win. */
    {
        Program *cur = (ctx->state.program && ctx->state.program->name != 0)
                           ? ctx->state.program : NULL;

        if (cur && cur->spirv[_VERTEX_SHADER].ir)
            vs_prog = cur;                    /* the bound program defines the VS */
        else if (ctx->state.program_pipeline)
            vs_prog = ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER];
        if (tess_trace())
            fprintf(stderr, "MGL tess: pair: vs_prog %u from %s (obj=%p table=%p sep=%d msl=%d)\n",
                    vs_prog ? vs_prog->name : 0,
                    (cur && cur->spirv[_VERTEX_SHADER].ir && vs_prog == cur) ? "cur" : "pipeline",
                    (const void *)vs_prog,
                    vs_prog ? (const void *)findProgram(ctx, vs_prog->name) : NULL,
                    vs_prog ? (int)MGL_PROGRAM_OBJ(vs_prog)->separable : -1,
                    vs_prog ? (int)(vs_prog->spirv[_VERTEX_SHADER].msl_str != NULL) : -1);
    }
    if (!vs_prog)
        vs_prog = tes;                   /* a vs+tes program in the TES slot */
    if (vs_prog == tes)
        vs_msl = tes->spirv[_VERTEX_SHADER].msl_str;
    else if (vs_prog->spirv[_VERTEX_SHADER].msl_str)
    {
        /* a vs-only program keeps a plain vertex function as its stage MSL — its
         * tessellation kernel form lives in the separable-kernel cache */
        const char *entry = NULL;
        const char *kmsl = mgl_tess_vs_kernel_msl(ctx, vs_prog->name, &entry);

        if (!kmsl && tess_trace())
            fprintf(stderr, "MGL tess: pair: vs-only program %u has no cached kernel "
                            "(separable=%d msl=%d)\n", vs_prog->name,
                    (int)MGL_PROGRAM_OBJ(vs_prog)->separable, vs_prog->spirv[_VERTEX_SHADER].msl_str != NULL);
        vs_msl = kmsl;                   /* owned by the cache (never freed here) */
    }

    if (vs_msl && tess_cp_record_name(vs_msl, vs_name, sizeof vs_name))
        have_vs = msl_parse_struct(vs_msl, vs_name, &vs_out);
    if (msl_tcs_record_names(tcs_msl, tcs_in_name, sizeof tcs_in_name,
                             tcs_out_name, sizeof tcs_out_name))
    {
        have_tcs_in = tcs_in_name[0] &&
                      msl_parse_struct(tcs_msl, tcs_in_name, &tcs_in);
        have_tcs_out = msl_parse_struct(tcs_msl, tcs_out_name, &tcs_out);
    }
    have_tcs_po = msl_parse_struct_entry(tcs_msl, &tcs_po, po_name, sizeof po_name);
    if (msl_patch_in_name(tes_msl, pi_name, sizeof pi_name))
        have_pi = msl_parse_struct(tes_msl, pi_name, &tes_pi);
    if (have_pi && msl_cp_struct_name(tes_msl, pi_name, cp_name, sizeof cp_name))
        have_cp = msl_parse_struct(tes_msl, cp_name, &tes_cp);

    /* interface keys from each module's own reflection.  Names resolve through the
     * module that DECLARES the interface (a te-only program has no TCS module, so
     * its patch members must not be looked up there — the block's member names live
     * in the tc program's TCS module, exactly where a single-program link finds
     * them, which keeps the keys identical on both sides of the pair) */
    tess_annotate_keys(ctx, vs_prog, _VERTEX_SHADER, &vs_out, have_vs ?
        &vs_prog->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tc, _TESS_CONTROL_SHADER, &tcs_in, have_tcs_in ?
        &tc->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, tc, _TESS_CONTROL_SHADER, &tcs_out, have_tcs_out ?
        &tc->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tc, _TESS_CONTROL_SHADER, &tcs_po, have_tcs_po ?
        &tc->spirv_resources_list[_TESS_CONTROL_SHADER][SPVC_RESOURCE_TYPE_STAGE_OUTPUT] : NULL);
    tess_annotate_keys(ctx, tc, _TESS_CONTROL_SHADER, &tes_cp, have_cp ?
        &tes->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);
    tess_annotate_keys(ctx, tes, _TESS_EVALUATION_SHADER, &tes_pi, have_pi ?
        &tes->spirv_resources_list[_TESS_EVALUATION_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT] : NULL);

    {
        struct MslStruct *vs_side[2] = { have_vs ? &vs_out : NULL, have_tcs_in ? &tcs_in : NULL };
        struct MslStruct *tes_side[2] = { have_tcs_out ? &tcs_out : NULL, have_cp ? &tes_cp : NULL };
        struct MslStruct *patch_side[2] = { have_tcs_po ? &tcs_po : NULL, have_pi ? &tes_pi : NULL };

        if (!tess_canon_union(&canon, vs_side, 2) ||
            !tess_canon_union(&canon_tcs, tes_side, 2) ||
            !tess_canon_union(&canon_patch, patch_side, 2))
        {
            if (tess_trace())
                fprintf(stderr, "MGL tess: pair canonical union failed (tc=%p tes=%p)\n",
                        (void *)tc, (void *)tes);
            return NULL;
        }
    }

    pair = (MGLTessProgram *)calloc(1, sizeof *pair);
    if (!pair)
        return NULL;
    *pair = *tp;                         /* program, domain, spacing, capture flag, ... */
    pair->next = NULL;
    pair->pair = NULL;
    pair->pair_program = NULL;
    pair->pair_key = NULL;
    pair->has_tcs = 1;
    pair->tcs_out_vertices = mgl_tess_program_out_vertices(ctx, tc);
    pair->hw_cp_count = pair->tcs_out_vertices;
    /* every compile-once variant of the base record embeds ITS layouts (or a baked
     * control point count) — the pair rebuilds its own on first use */
    pair->tes_hw_function = NULL;
    pair->tes_hw_library = NULL;
    pair->tes_hw_n = 0;
    pair->plain_msl = NULL;
    pair->plain_msl_dumped = false;
    pair->plain_function = NULL;
    pair->plain_library = NULL;
    pair->capture_function = NULL;
    pair->capture_library = NULL;
    pair->capture_pso = NULL;
    pair->capture_fc_fixup = false;
    pair->vs_function = NULL;
    pair->vs_library = NULL;

    /* canonical cp record of the pair: the TES reads its members at these offsets */
    pair->num_cp_record = canon_tcs.n;
    pair->cp_record_full_size = canon_tcs.size;
    pair->cp_record_size = canon_tcs.size;
    pair->vs_record_size = canon.size;   /* the VS->TCS interface record (see header) */
    for (int i = 0; i < canon_tcs.n && i < MGL_TESS_MAX_PATCH_MEMBERS; i++)
        pair->cp_record[i] = canon_tcs.m[i];
    /* the pair's TES member keys must be the ones resolved HERE (against the tc
     * program's modules, like a single-program link): the te-only program's own
     * link had no TCS module to resolve block-member bases through, so its keys
     * name-hash the flattened members and would never alias the producer side */
    for (int i = 0; i < pair->num_tes_cp; i++)
        for (int j = 0; j < tes_cp.count; j++)
            if (!strcmp(pair->tes_cp[i].name, tes_cp.members[j].name))
            {
                pair->tes_cp[i].key = tes_cp.members[j].key;
                break;
            }
    for (int i = 0; i < pair->num_tes_patch_in; i++)
        for (int j = 0; j < tes_pi.count; j++)
            if (!strcmp(pair->tes_patch_in[i].name, tes_pi.members[j].name))
            {
                pair->tes_patch_in[i].key = tes_pi.members[j].key;
                break;
            }
    for (int i = 0; i < pair->num_tes_cp; i++)
    {
        for (int j = 0; j < canon_tcs.n; j++)
            if (pair->tes_cp[i].key == canon_tcs.m[j].key)
            {
                pair->tes_cp[i].offset = canon_tcs.m[j].offset;
                if (pair->tes_cp[i].flat_dims && strstr(canon_tcs.m[j].type, "spvUnsafeArray"))
                {
                    size_t stride = msl_array_stride(canon_tcs.m[j].type);

                    if (stride)
                        pair->tes_cp[i].offset = canon_tcs.m[j].offset +
                            pair->tes_cp[i].array_idx * stride;
                    if (stride && pair->tes_cp[i].array_idx2)
                    {
                        char inner[96];
                        long n2 = 0;

                        if (msl_array_inner(canon_tcs.m[j].type, inner, sizeof inner, &n2) &&
                            msl_array_stride(inner))
                            pair->tes_cp[i].offset += pair->tes_cp[i].array_idx2 *
                                msl_array_stride(inner);
                    }
                }
                break;
            }
        if (tess_trace())
            fprintf(stderr, "MGL tess: pair tes_cp[%d] %-22s key=%08x -> offset %zu\n",
                    i, pair->tes_cp[i].name, pair->tes_cp[i].key, pair->tes_cp[i].offset);
    }

    /* canonical per-patch record: TES patch-in offsets, blob layout and the TCS
     * patchOut members all use it */
    for (int i = 0; i < pair->num_tes_patch_in; i++)
        for (int j = 0; j < canon_patch.n; j++)
            if (pair->tes_patch_in[i].key == canon_patch.m[j].key)
            {
                pair->tes_patch_in[i].offset = canon_patch.m[j].offset;
                break;
            }
    pair->tes_patch_in_size = canon_patch.size;
    pair->tcs_patch_out_size = canon_patch.size;
    pair->num_tcs_patch_out = 0;
    for (int i = 0; i < tcs_po.count && pair->num_tcs_patch_out < MGL_TESS_MAX_PATCH_MEMBERS; i++)
    {
        MGLTessPatchMember *m = &tcs_po.members[i];

        for (int j = 0; j < canon_patch.n; j++)
            if (m->key == canon_patch.m[j].key)
            {
                pair->tcs_patch_out[pair->num_tcs_patch_out++] = *m;
                pair->tcs_patch_out[pair->num_tcs_patch_out - 1].offset = canon_patch.m[j].offset;
                break;
            }
    }

    /* the TCS kernel: recompiled with the pair's canonical structs, the domain's
     * factor-record form and the GL float-level copy (same rewrite sequence a
     * single-program link applies) */
    {
        char *msl = strdup(tcs_msl);
        bool changed = false;
        char body[4096];

        if (have_tcs_in && !tess_canon_matches(&tcs_in, &canon))
        {
            char *nw;
            tess_canon_body(body, sizeof body, &tcs_in, &canon);
            nw = msl_rewrite_struct_body(msl, tcs_in_name, body);
            if (nw) { free(msl); msl = nw; changed = true; }
        }
        if (have_tcs_out && !tess_canon_matches(&tcs_out, &canon_tcs))
        {
            char *nw;
            tess_canon_body(body, sizeof body, &tcs_out, &canon_tcs);
            nw = msl_rewrite_struct_body(msl, tcs_out_name, body);
            if (nw) { free(msl); msl = nw; changed = true; }
        }
        if (have_tcs_po && !tess_canon_matches(&tcs_po, &canon_patch))
        {
            char *nw;
            tess_canon_body(body, sizeof body, &tcs_po, &canon_patch);
            nw = msl_rewrite_struct_body(msl, po_name, body);
            if (nw) { free(msl); msl = nw; changed = true; }
        }

        /* triangle domains: write MTLTriangleTessellationFactorsHalf (8-byte factor
         * records) instead of the quad struct (see mgl_tess_factor_stride) */
        if (pair->domain == MGL_TESS_DOMAIN_TRIANGLES)
        {
            char *step = strstr(msl, "device MTLQuadTessellationFactorsHalf*");
            if (step)
            {
                size_t oldlen = strlen("device MTLQuadTessellationFactorsHalf*");
                size_t newlen = strlen("device MTLTriangleTessellationFactorsHalf*");
                size_t at = (size_t)(step - msl);
                char *grown = (char *)realloc(msl, strlen(msl) + newlen - oldlen + 1);

                if (grown)
                {
                    char *p;
                    msl = grown;
                    p = msl + at;
                    memmove(p + newlen, p + oldlen, strlen(p + oldlen) + 1);
                    memcpy(p, "device MTLTriangleTessellationFactorsHalf*", newlen);
                    for (char *w = msl; (w = strstr(w, ".insideTessellationFactor[0]")) != NULL; )
                    {
                        size_t tail = strlen(w + strlen(".insideTessellationFactor[0]")) + 1;
                        memmove(w + strlen(".insideTessellationFactor"),
                                w + strlen(".insideTessellationFactor[0]"), tail);
                        w += strlen(".insideTessellationFactor");
                    }
                    /* gl_TessLevelInner[1] is ignored for triangles (see
                     * tcs_redirect_tri_inner1) */
                    msl = tcs_redirect_tri_inner1(msl);
                    for (char *w = msl; (w = strstr(w, ".edgeTessellationFactor[3]")) != NULL; )
                    {
                        char *ls = w, *semi = strchr(w, ';'), *eq, *rhs;
                        char repl[512];
                        while (ls > msl && ls[-1] != '\n' && ls[-1] != ';' &&
                               ls[-1] != '{' && ls[-1] != '}')
                            ls--;
                        if (!semi) { w += strlen(".edgeTessellationFactor"); continue; }
                        eq = semi;
                        while (eq > ls && eq[-1] != '=') eq--;
                        if (eq == ls) { w = semi; continue; }
                        rhs = eq;
                        while (*rhs == ' ' || *rhs == '=') rhs++;
                        snprintf(repl, sizeof repl, "(void)(%.*s);", (int)(semi - rhs), rhs);
                        {
                            size_t span = (size_t)(semi - ls + 1);
                            memset(ls, ' ', span);
                            memcpy(ls, repl, strlen(repl) <= span ? strlen(repl) : span);
                            w = ls + span;
                        }
                    }
                    changed = true;
                }
            }
        }

        {
            char *lvl = tcs_inject_float_levels(msl);

            if (lvl) { free(msl); msl = lvl; pair->tcs_levels_float = 1; changed = true; }
        }

        {
            char *pd = tcs_patch_primid(msl);

            if (pd) { free(msl); msl = pd; changed = true; }
        }
        if (getenv("MGL_DUMP_MSL"))
        {
            FILE *df = fopen("/tmp/pair_tcs.metal", "w");

            if (df) { fputs(msl, df); fclose(df); }
        }
        if (changed && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
        {
            void *lib = NULL, *fn = NULL;
            char *err = NULL;
            const char *entry = tc->spirv[_TESS_CONTROL_SHADER].entry_point;

            if (!entry || !*entry)
                entry = "main0";
            if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl, entry, &lib, &fn, &err))
            {
                pair->tcs_function = fn;
                pair->tcs_library = lib;
                if (tess_trace())
                    fprintf(stderr, "MGL tess: pair TCS kernel compiled (cp %zu b, patch %zu b, outv %d)\n",
                            canon_tcs.size, canon_patch.size, pair->tcs_out_vertices);
            }
            else
            {
                fprintf(stderr, "MGL ERROR: tess: pair TCS recompile failed: %s\n", err ? err : "?");
                free(err);
            }
        }
        else if (!changed)
        {
            /* the raw kernel already matches the canonical layouts: still needs the
             * float-level copy writer, which tcs_inject_float_levels added above */
            pair->tcs_levels_float = 1;
        }
        free(msl);
    }

    /* the pair's VS kernel: recompiled when the raw output record differs from the
     * canonical (vs -> TCS) record — the source MSL is the program's kernel form
     * (a vs+tes program) or the separable-kernel cache (a vs-only program) */
    if (have_vs && vs_msl && !tess_canon_matches(&vs_out, &canon))
    {
        char body[4096];
        char *nw;
        struct MslStruct own = vs_out;

        tess_canon_body(body, sizeof body, &own, &canon);
        nw = msl_rewrite_struct_body(vs_msl, vs_name, body);
        if (nw)
        {
            tess_widen_double_stores(&nw, &own, &canon);
            if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                const char *entry = "main0";
                char kentry[128] = "main0";
                const char *kd = strstr(nw, "kernel void ");

                if (kd)
                {
                    const char *ks = kd + strlen("kernel void ");
                    size_t kl = 0;

                    while (ks[kl] && (isalnum((unsigned char)ks[kl]) || ks[kl] == '_'))
                        kl++;
                    if (kl && kl < sizeof kentry)
                    {
                        memcpy(kentry, ks, kl);
                        kentry[kl] = 0;
                    }
                }
                entry = kentry;
                if (vs_prog == tes && tes->spirv[_VERTEX_SHADER].entry_point &&
                    *tes->spirv[_VERTEX_SHADER].entry_point && !strcmp(entry, "main0"))
                    entry = tes->spirv[_VERTEX_SHADER].entry_point;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, nw, entry, &lib, &fn, &err))
                {
                    pair->vs_function = fn;
                    pair->vs_library = lib;
                    pair->pair_vs = vs_prog;
                    if (tess_trace())
                        fprintf(stderr, "MGL tess: pair VS kernel compiled (record %zu b)\n",
                                canon.size);
                }
                else
                {
                    fprintf(stderr, "MGL ERROR: tess: pair VS recompile failed: %s\n",
                            err ? err : "?");
                    free(err);
                }
            }
            free(nw);
        }
    }

    /* the GPU blob map of the pair: TES patch-in layout, sources from the tc
     * program's patchOut members at the canonical offsets */
    mgl_tess_build_blob_map(ctx, pair);

    if (tess_trace())
    {
        fprintf(stderr, "MGL tess: pair built: tes prog %u + tc prog %u: canon vs->tcs %zu b, "
                "tcs->tes %zu b, patch %zu b, outv=%d tcs_fn=%p\n",
                tp->program_name, tc->name, canon.size, canon_tcs.size, canon_patch.size,
                pair->tcs_out_vertices, pair->tcs_function);
    }
    return pair;
}

/* name of the TES patch input struct: the type of the "patchIn" parameter (hardware
 * form: "<patchIn> patchIn [[stage_in]]") or of the prologue local (plain form) */
static bool msl_patch_in_name(const char *msl, char *out, size_t out_len)
{
    const char *decl = strstr(msl, " patchIn [[stage_in]]");

    if (!decl)
        decl = strstr(msl, " patchIn;");
    if (!decl)
        return false;

    const char *e = decl;
    while (e > msl && *(e - 1) == ' ')
        e--;
    const char *b = e;
    while (b > msl && *(b - 1) != ' ' && *(b - 1) != '\n' && *(b - 1) != '(' && *(b - 1) != ',')
        b--;
    if (b == e)
        return false;
    snprintf(out, out_len, "%.*s", (int)(e - b), b);
    return true;
}

/* control point struct name of the TES ("patch_control_point<T> gl_in;" in the
 * hardware form, "mgl_cp_ref<T> gl_in;" in the plain form) */
static bool msl_cp_struct_name(const char *msl, const char *patchin_name, char *out, size_t out_len)
{
    char head[160];
    const char *p;

    snprintf(head, sizeof head, "struct %s\n{", patchin_name);
    p = strstr(msl, head);
    if (!p) { snprintf(head, sizeof head, "struct %s {", patchin_name); p = strstr(msl, head); }
    if (!p)
        return false;
    const char *ref = strstr(p, "patch_control_point<");
    if (!ref)
        ref = strstr(p, "mgl_cp_ref<");
    if (!ref || ref > strchr(p, '}'))
        return false;
    const char *t0 = ref + (strstr(ref, "<") == ref + strlen("patch_control_point<") - 1
                            ? strlen("patch_control_point<") : strlen("mgl_cp_ref<"));
    /* t0 currently points at the start of the template argument either way */
    {
        const char *lt = strchr(p, '<');
        if (!lt) return false;
        t0 = lt + 1;
    }
    const char *t1 = strchr(t0, '>');
    if (!t1 || (size_t)(t1 - t0) >= out_len)
        return false;
    memcpy(out, t0, (size_t)(t1 - t0));
    out[t1 - t0] = 0;
    return true;
}


/* name of the control point record struct: the TCS's spvIn element type, or the VS
 * kernel's spvOut element type */
static bool tess_cp_record_name(const char *msl, char *out, size_t out_len)
{
    const char *mark = strstr(msl, "device ");
    const char *star = NULL;
    char word[128];
    const char *b, *e;

    while (mark)
    {
        /* device TYPE* spvIn / device TYPE* spvOut */
        if (!strncmp(mark + 7, "const ", 6))
            mark = strstr(mark + 1, "device ");
        star = strchr(mark, '*');
        if (star && star - mark < 120)
        {
            const char *after = star + 1;
            while (*after == ' ') after++;
            if (!strncmp(after, "spvIn", 5) || !strncmp(after, "spvOut", 6))
            {
                b = mark + 7;
                if (!strncmp(b, "const ", 6)) b += 6;
                e = star;
                while (e > b && *(e - 1) == ' ') e--;
                if (e > b && (size_t)(e - b) < out_len)
                {
                    snprintf(out, out_len, "%.*s", (int)(e - b), b);
                    return true;
                }
            }
        }
        mark = strstr(mark + 1, "device ");
        (void)word;
    }
    return false;
}


/* find the "<name>_patchOut" struct of a TCS (any name) */
static bool msl_parse_struct_entry(const char *msl, struct MslStruct *out,
                                   char *struct_name, size_t struct_len)
{
    const char *at = msl;

    if (struct_name && struct_len)
        struct_name[0] = 0;
    while ((at = strstr(at, "_patchOut")) != NULL)
    {
        const char *b = at;
        char name[160];

        while (b > msl && *(b - 1) != ' ' && *(b - 1) != '\n')
            b--;
        if (b == at || b == msl || strncmp(b - 7, "struct ", 7))
        {
            at += 9;
            continue;
        }
        snprintf(name, sizeof name, "%.*s", (int)(at + 9 - b), b);
        if (msl_parse_struct(msl, name, out))
        {
            if (struct_name && struct_len)
                snprintf(struct_name, struct_len, "%s", name);
            return true;
        }
        at += 9;
    }
    return false;
}

/* ------------------------------------------------------------------ GL_PATCHES draw
 * Called from draw_buffers.c once the draw is validated: hands the draw to the
 * Metal side (see MGLRenderer+Tessellation.m).
 */
bool mgl_tess_draw_patches(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                           GLint first, GLsizei count, GLsizei instances,
                           GLint basevertex, GLint baseinstance)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLTessProgram *tp;

    if (!S)
        return false;

    tp = mgl_tess_active_program(ctx);
    if (!tp)
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: draw rejected: no active tess program (prog=%p)\n",
                    (void *)ctx->state.program);
        return false;
    }

    if (!ctx->mtl_funcs.ext_tess || !ctx->mtl_funcs.ext_tess->draw_patches)
    {
        if (tess_trace())
            fprintf(stderr, "MGL tess: draw rejected: no draw hook\n");
        return false;
    }

    return ctx->mtl_funcs.ext_tess->draw_patches(ctx, indexed, type, offset, first, count,
                                                 instances ? instances : 1, basevertex, baseinstance);
}

/* GL_TESS_CONTROL_OUTPUT_VERTICES for a linked program: the tess link record's value
 * when one exists (programs with a TES); a separable tc-only program has no record, so
 * read the declaration straight from the TCS SPIR-V (matches the link-time parse) */
GLint mgl_tess_program_out_vertices(GLMContext ctx, Program *program)
{
    MGLTessProgram *tp = mgl_tess_program(ctx, program);

    if (tp)
        return tp->tcs_out_vertices;
    if (ctx && program && program->spirv[_TESS_CONTROL_SHADER].ir)
    {
        int modes[16], count = 0, out_vertices = 0;

        if (spv_tess_modes(program->spirv[_TESS_CONTROL_SHADER].ir,
                           program->spirv[_TESS_CONTROL_SHADER].size,
                           1 /* SpvExecutionModelTessellationControl */,
                           modes, &count, &out_vertices) &&
            out_vertices >= 1 && out_vertices <= MGL_TESS_MAX_PATCH_VERTICES)
            return out_vertices;
    }
    return 3;    /* the link-time fallback for an unparseable declaration */
}

/* the tessellation program of the current GL state (program object or separable pipeline) */
MGLTessProgram *mgl_tess_active_program(GLMContext ctx)
{
    Program *tes = NULL;
    MGLTessProgram *tp;

    /* geom: a bound program pipeline installs its COMPOSITE as state.program during
     * processGLState (MGL_PIPELINE_PROGRAM_SCOPE); the composite (name 0) has no
     * tess record of its own — resolve through the pipeline's stage programs below
     * or the TES render pass builds its pipeline with the plain vertex function */
    if (ctx->state.program && ctx->state.program->name != 0)
    {
        tes = ctx->state.program;
    }
    else if (ctx->state.program_pipeline)
    {
        if (ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER])
            tes = ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER];
        else if (ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER])
            tes = ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER];
    }

    if (!tes)
        return NULL;
    tp = mgl_tess_program(ctx, tes);
    if (!tp)
        return NULL;

    /* a pipeline can pair this TES with a tc-only program: the TES program's own
     * record was linked without a TCS (has_tcs=0), but such a draw must run the
     * pipeline's TCS. Hand out the DERIVED pair record instead — built on first
     * use with has_tcs=1 and the two programs' canonical interface layouts (see
     * tess_build_pair); rebuilt when the tc program relinks. */
    if (ctx->state.program_pipeline && !tp->has_tcs)
    {
        Program *tc = ctx->state.program_pipeline->stage_programs[_TESS_CONTROL_SHADER];

        if (tc && tc->spirv[_TESS_CONTROL_SHADER].ir && tc != tp->program)
        {
            const void *key = (const void *)tc->spirv[_TESS_CONTROL_SHADER].msl_str;

            if (tp->pair && (tp->pair_program != tc || tp->pair_key != key))
            {
                tess_free_pair(ctx, tp->pair);
                tp->pair = NULL;
                tp->pair_program = NULL;
                tp->pair_key = NULL;
            }
            if (!tp->pair)
            {
                tp->pair = tess_build_pair(ctx, tp, tc);
                if (tp->pair)
                {
                    tp->pair_program = tc;
                    tp->pair_key = key;
                }
            }
            if (tp->pair)
                return tp->pair;
        }
    }
    return tp;
}

/* record size of the TES control-point struct (rewritten TES MSL), so the Metal side
 * can size its scratch buffers without re-parsing the shader source */
bool mgl_tess_cp_record_size(const char *msl, char *patchin_name, size_t patchin_len,
                             char *cp_name, size_t cp_len, size_t *record_size)
{
    struct MslStruct cp;
    char pin[160], cpn[128];

    if (!msl || !msl_patch_in_name(msl, pin, sizeof pin))
        return false;
    if (!msl_cp_struct_name(msl, pin, cpn, sizeof cpn))
        return false;
    if (!msl_parse_struct(msl, cpn, &cp))
        return false;
    if (patchin_name && patchin_len)
        snprintf(patchin_name, patchin_len, "%s", pin);
    if (cp_name && cp_len)
        snprintf(cp_name, cp_len, "%s", cpn);
    *record_size = cp.size;
    return cp.size > 0;
}

/* exported for MGLRenderer+Tessellation.m */
bool mgl_tess_generate(GLMContext ctx, MGLTessProgram *tp, const float *factors, unsigned patches)
{
    tess_cpu_generate_runs++;
    return tess_generate(ctx, tp, factors, patches);
}

/* ------------------------------------------------------------------ per-instance gl_PrimitiveID
 * GL restarts gl_PrimitiveID at 0 for every instance (GL 4.6 table 11.1 "PrimitiveID
 * restarts for each instance"); SPIRV-Cross's multi-patch TCS kernel derives it from
 * the global thread index (thread = patch*out_vertices + invocation), which counts
 * instances too.  Rename that derivation to mgl_pid_g — the record writes below it
 * (tess factors, float levels, patch outputs) must keep the GLOBAL patch index, one
 * record per (instance, patch) — and define the builtin as the per-instance index,
 * taken from a third indirect-params word (patches per instance, == total when the
 * draw is not instanced).  Returns a fresh string (caller frees), NULL if the
 * derivation pattern was not found. */
static char *tcs_patch_primid(const char *msl)
{
    static const char head[] = "uint gl_PrimitiveID = min(gl_GlobalInvocationID.x / ";
    const char *at = strstr(msl, head);
    char *out;
    char po_name[96];
    const char *po;
    size_t cap, len = 0, nlen;

    if (!at)
        return NULL;
    {
        /* the divisor: out-vertex count baked into the kernel */
        const char *d = at + sizeof head - 1;
        size_t dn = strspn(d, "0123456789");

        if (!dn || strncmp(d + dn, ", spvIndirectParams[1] - 1);", 28) != 0)
            return NULL;
        nlen = dn;
    }

    /* the patch-output buffer's parameter name (buffer 26): its [gl_PrimitiveID]
     * stores must keep the global index */
    po_name[0] = 0;
    for (po = strstr(msl, "[[buffer(26)]]"); po; po = strstr(po + 1, "[[buffer(26)]]"))
    {
        const char *star = po, *nm = po;
        size_t nn = 0;

        while (star > msl && star[-1] != ',' && star[-1] != '\n')
            star--;
        if (strncmp(star, "device ", 7) != 0)
            continue;
        nm = star + 7;
        nm += strspn(nm, " \t");
        nm = strchr(nm, '*');
        if (!nm)
            continue;
        nm = nm + 1 + strspn(nm + 1, " \t");
        nn = strcspn(nm, " ,[];");
        if (nn && nn < sizeof po_name)
        {
            snprintf(po_name, sizeof po_name, "%.*s", (int)nn, nm);
            break;
        }
    }

    cap = strlen(msl) + 1024;
    out = (char *)malloc(cap);
    if (!out)
        return NULL;

    /* everything before the derivation */
    str_append(&out, &len, &cap, "%.*s", (int)(at - msl), msl);
    str_append(&out, &len, &cap,
               "uint mgl_pid_g = min(gl_GlobalInvocationID.x / %.*s, spvIndirectParams[1] - 1);\n"
               "    uint gl_PrimitiveID = mgl_pid_g %% max(spvIndirectParams[2], 1u);",
               (int)nlen, at + sizeof head - 1);
    str_append(&out, &len, &cap, "%s", at + (sizeof head - 1) + nlen + 28);

    /* record stores keep the global patch index */
    {
        const char *names[3];
        int n = 0;

        names[n++] = "spvTessLevel";
        names[n++] = "mgl_levels";
        if (po_name[0])
            names[n++] = po_name;
        for (int i = 0; i < n; i++)
        {
            char *nw, *w;
            char pat[160];

            snprintf(pat, sizeof pat, "%s[gl_PrimitiveID]", names[i]);
            while ((w = strstr(out, pat)) != NULL)
            {
                char repl[160];
                size_t off = (size_t)(w - out);
                size_t tail_len = len - off - strlen(pat);   /* bytes behind the pattern */

                snprintf(repl, sizeof repl, "%s[mgl_pid_g]", names[i]);
                nw = (char *)malloc(off + strlen(repl) + tail_len + 1);
                if (!nw)
                    break;
                memcpy(nw, out, off);
                memcpy(nw + off, repl, strlen(repl));
                memcpy(nw + off + strlen(repl), w + strlen(pat), tail_len + 1);  /* + NUL */
                free(out);
                out = nw;
                len = off + strlen(repl) + tail_len;
                cap = len + 1;
                w = out;
            }
        }
    }
    return out;
}

/* ------------------------------------------------------------------ float tessellation levels
 * GL promises the TES reads back the EXACT float levels the TCS wrote (GL 4.6
 * 11.1.3.2); Metal's factor record stores halfs, which loses e.g. 2.7 to 2.69922.
 * The canonicalized TCS kernel additionally writes a float copy per patch: every
 * `spvTessLevel[gl_PrimitiveID].edgeTessellationFactor[i] = half(EXPR);` gains
 * ` mgl_levels[gl_PrimitiveID].o[i] = float(EXPR);` (inside -> .in[i]), and the
 * kernel signature gains `device mgl_tess_levels_f* mgl_levels [[buffer(30)]]`.
 * Returns a fresh string (caller frees) or NULL when nothing could be injected. */
/* Triangle patches: gl_TessLevelInner[1] is IGNORED by GL (only [0] is the inside
 * factor), but SPIRV-Cross lowers the array onto the scalar insideTessellationFactor
 * of MTLTriangleTessellationFactorsHalf — a later [1] store would clobber the [0]
 * value both the hardware tessellator and the emulated paths need (KHR-GL46
 * point_mode.points_verification: inner written as (33,-1) collapsed to one ring).
 * Route every [1] access to the float-levels record's in[1] slot, which is unused
 * for triangles and keeps the expression (and side effects) evaluating.  Requires
 * another tess-level store so tcs_inject_float_levels adds the mgl_levels buffer
 * (a TCS writing only inner[1] is degenerate; keep the old behaviour there). */
static char *tcs_redirect_tri_inner1(char *msl)
{
    const char *needle = "spvTessLevel";
    bool touched = false;
    char *out, *w;
    size_t cap, len = 0;

    if (!msl || !strstr(msl, ".insideTessellationFactor[1]"))
        return msl;
    if (!strstr(msl, ".insideTessellationFactor[0]") && !strstr(msl, ".edgeTessellationFactor"))
        return msl;

    cap = strlen(msl) + 4096;
    out = (char *)malloc(cap);
    if (!out)
        return msl;
    w = msl;
    for (;;)
    {
        char *hit = strstr(w, needle);
        char *at;
        if (!hit)
            break;
        at = hit + strlen(needle);
        if (*at != '[')
        {
            str_append(&out, &len, &cap, "%.*s", (int)(at - w), w);
            w = at;
            continue;
        }
        {
            int depth = 0;
            for (; *at; at++)
            {
                if (*at == '[')
                    depth++;
                else if (*at == ']')
                {
                    depth--;
                    if (!depth)
                    {
                        at++;
                        break;
                    }
                }
            }
        }
        if (strncmp(at, ".insideTessellationFactor[1]", 28) != 0)
        {
            str_append(&out, &len, &cap, "%.*s", (int)(at - w), w);
            w = at;
            continue;
        }
        str_append(&out, &len, &cap, "%.*s", (int)(hit - w), w);
        str_append(&out, &len, &cap, "%s",
                   "mgl_levels[gl_PrimitiveID].in[1]");
        touched = true;
        w = at + 28;
    }
    if (!touched)
    {
        free(out);
        return msl;
    }
    str_append(&out, &len, &cap, "%s", w);
    free(msl);
    return out;
}

static char *tcs_inject_float_levels(const char *msl)
{
    const char *sig = strstr(msl, "spvTessLevel [[buffer(27)]]");
    const char *kdecl = strstr(msl, "kernel void ");
    size_t cap, len = 0;
    char *out;

    if (!sig || !kdecl || !strstr(msl, "TessellationFactor"))
        return NULL;

    cap = strlen(msl) + 4096;
    out = (char *)malloc(cap);
    if (!out)
        return NULL;

    for (const char *p = msl; *p; )
    {
        const char *edge = strstr(p, ".edgeTessellationFactor");
        const char *ins  = strstr(p, ".insideTessellationFactor");
        const char *hit  = NULL;
        int is_edge = 0;

        if (edge && (!ins || edge < ins)) { hit = edge; is_edge = 1; }
        else if (ins)                      { hit = ins; }
        if (!hit)
        {
            str_append(&out, &len, &cap, "%s", p);
            break;
        }

        /* copy up to the member */
        str_append(&out, &len, &cap, "%.*s", (int)(hit - p), p);

        /* index: [N] after the member (absent for the scalar triangle inside level) */
        const char *q = hit + strlen(is_edge ? ".edgeTessellationFactor" : ".insideTessellationFactor");
        char idx[32] = "0";
        const char *assign = q;

        if (*q == '[')
        {
            const char *cl = strchr(q, ']');
            if (cl && (size_t)(cl - q - 1) < sizeof idx)
            {
                memcpy(idx, q + 1, (size_t)(cl - q - 1));
                idx[cl - q - 1] = 0;
                assign = cl + 1;
            }
        }
        while (*assign == ' ')
            assign++;

        /* the statement must be `= half(EXPR);` — extract EXPR to keep the float copy */
        const char *semi = strchr(assign, ';');
        if (!semi || strncmp(assign, "= half(", 7) != 0)
        {
            /* not a recognised half store: copy the member text verbatim and continue */
            str_append(&out, &len, &cap, "%.*s", (int)(q - hit), hit);
            p = q;
            continue;
        }
        {
            int depth = 0;
            const char *hp = assign + 6;      /* at "half(" open paren */
            const char *e = NULL;

            for (const char *r = hp; *r && r < semi + 1; r++)
            {
                if (*r == '(') depth++;
                else if (*r == ')') { if (--depth == 0) { e = r; break; } }
            }
            if (!e || e > semi)
            {
                str_append(&out, &len, &cap, "%.*s", (int)(q - hit), hit);
                p = q;
                continue;
            }
            /* write the member + the original `= half(EXPR)` (its `;` follows below),
             * then the appended float store */
            str_append(&out, &len, &cap, "%.*s%.*s; mgl_levels[gl_PrimitiveID].%s[%s] = float(%.*s);",
                       (int)(q - hit), hit,
                       (int)(semi - q), q,
                       is_edge ? "o" : "in", idx,
                       (int)(e - (hp + 1)), hp + 1);
            p = semi + 1;
        }
    }

    /* the signature parameter + the record struct.  NOTE: "kernel void" precedes the
     * parameter list, so the two insert points can be in either order — splice by
     * ascending offset */
    {
        char *sig_at = strstr(out, "spvTessLevel [[buffer(27)]]");
        char *k_at = strstr(out, "kernel void ");

        if (sig_at && k_at)
        {
            size_t sig_off = (size_t)(sig_at - out) + strlen("spvTessLevel [[buffer(27)]]");
            size_t k_off = (size_t)(k_at - out);
            const char *ins_sig = ", device mgl_tess_levels_f* mgl_levels [[buffer(30)]]";
            const char *ins_str = "struct mgl_tess_levels_f { float o[4]; float in[2]; };\n";
            size_t L1 = strlen(ins_sig), L2 = strlen(ins_str);
            size_t nlen = len + L1 + L2 + 16;
            char *n = (char *)malloc(nlen);
            char *w;
            const char *r = out;
            size_t rem = len + 1;      /* including the NUL */

            if (!n) { free(out); return NULL; }
            w = n;
            /* struct before the kernel declaration ... */
            if (k_off < sig_off)
            {
                memcpy(w, r, k_off); w += k_off;
                memcpy(w, ins_str, L2); w += L2;
                r += k_off; rem -= k_off;
                memcpy(w, r, sig_off - k_off); w += sig_off - k_off;
                memcpy(w, ins_sig, L1); w += L1;
                r += sig_off - k_off; rem -= sig_off - k_off;
            }
            else                        /* ... or after it, if the signature came first */
            {
                memcpy(w, r, sig_off); w += sig_off;
                memcpy(w, ins_sig, L1); w += L1;
                r += sig_off; rem -= sig_off;
                memcpy(w, r, k_off - sig_off); w += k_off - sig_off;
                memcpy(w, ins_str, L2); w += L2;
                r += k_off - sig_off; rem -= k_off - sig_off;
            }
            memcpy(w, r, rem);           /* the rest, including the NUL */
            free(out);
            return n;
        }
    }
    return out;
}

/* precompute, once per program, how the GPU blob kernel must assemble each patch's
 * TES patch-in record: for every TES patch-in member, either the tessellation
 * levels (from the half factor record) or a byte range of the TCS patch output,
 * matched by varying name exactly like tess_build_patch_blob does on the CPU. */
/* patch members match through tess_patch_member_matches() (defined above
 * tess_build_patch_blob): canonical key first, flattened element index second. */

void mgl_tess_build_blob_map(GLMContext ctx, MGLTessProgram *tp)
{
    int n = 0;

    (void)ctx;
    tp->blob_map_n = 0;
    for (int i = 0; i < tp->num_tes_patch_in && n < MGL_TESS_BLOB_MAX; i++)
    {
        const MGLTessPatchMember *m = &tp->tes_patch_in[i];
        uint32_t kind = 0, src = 0, size = 0;

        if (!strcmp(m->name, "gl_TessLevel") || !strcmp(m->name, "gl_TessLevelOuter"))
            kind = MGL_TESS_BLOB_OUTER;
        else if (!strcmp(m->name, "gl_TessLevelInner"))
            kind = MGL_TESS_BLOB_INNER;
        else
        {
            /* copy from the TCS patch output record, matched by varying location
             * (the canonical key — names may differ between the stages; flattened
             * per-element members additionally match their element index) */
            bool found = false;
            for (int j = 0; j < tp->num_tcs_patch_out && !found; j++)
            {
                if (tess_patch_member_matches(&tp->tcs_patch_out[j], m))
                {
                    size_t elem = 0;

                    src = (uint32_t)tp->tcs_patch_out[j].offset;
                    /* flattened element ("name_N", array_idx N) of a canonical
                     * spvUnsafeArray patch output: advance the source by the element
                     * stride — without it every element copied element 0's bytes into
                     * its slot ('patch out int r[16]' read back as r[0] everywhere);
                     * a struct-array field element steps the inner dimension too */
                    if (m->flat_dims && strstr(tp->tcs_patch_out[j].type, "spvUnsafeArray"))
                    {
                        size_t stride = msl_array_stride(tp->tcs_patch_out[j].type);

                        if (stride)
                            elem = stride;
                        if (m->array_idx2)
                        {
                            char inner[96];
                            long n2 = 0;

                            /* A 1-D slot ("spvUnsafeArray<float, 2>") has a NON-array
                             * inner type, so msl_array_stride(inner) is 0 — the element
                             * stride is then the member size ("pb_ps_0_z_1" must copy
                             * 4 bytes further than z_0, not 0) */
                            if (msl_array_inner(tp->tcs_patch_out[j].type, inner, sizeof inner, &n2) &&
                                msl_array_stride(inner))
                                src += (uint32_t)(m->array_idx2 * msl_array_stride(inner));
                            else
                                src += (uint32_t)(m->array_idx2 * m->size);
                            if (m->array_idx3)
                            {
                                char inner2[96];
                                long n3 = 0;

                                if (msl_array_inner(inner, inner2, sizeof inner2, &n3) &&
                                    msl_array_stride(inner2))
                                    src += (uint32_t)(m->array_idx3 * msl_array_stride(inner2));
                                else
                                    src += (uint32_t)(m->array_idx3 * m->size);
                            }
                        }
                        else if (m->array_idx3)
                            src += (uint32_t)(m->array_idx3 * m->size);
                    }
                    src += (uint32_t)(m->array_idx * elem);
                    size = (uint32_t)m->size;
                    kind = MGL_TESS_BLOB_COPY;
                    found = true;
                }
            }
            if (!found)
                continue;               /* no producer: stays zero (undefined per GL) */
        }

        tp->blob_kind[n] = kind;
        tp->blob_dst[n] = (uint32_t)m->offset;
        tp->blob_src[n] = src;
        tp->blob_size[n] = size;
        n++;
    }
    tp->blob_map_n = n;
}

void mgl_tess_build_patch_blob(GLMContext ctx, MGLTessProgram *tp,
                               const unsigned char *tcs_patch_out, size_t tcs_patch_stride,
                               const float *factors, unsigned patches)
{
    tess_build_patch_blob(ctx, tp, tcs_patch_out, tcs_patch_stride, factors, patches);
}

GLsizei mgl_tess_patch_vertices_in(GLMContext ctx, MGLTessProgram *tp)
{
    if (tp->has_tcs)
        return tp->tcs_out_vertices;
    if (ctx->state.ext_rstate)
        return ((struct MGLExtState_rstate *)ctx->state.ext_rstate)->patch_vertices;
    return 3;
}

/* the tessellator's factor record stride: 8 bytes for triangles (3 edge + 1 inside
 * halfs), 12 for quads/isolines (4 edge + 2 inside) */
size_t mgl_tess_factor_stride(GLMContext ctx, MGLTessProgram *tp)
{
    if (tp && tp->domain == MGL_TESS_DOMAIN_TRIANGLES)
        return 8;
    return 12;
}

/* post-link: fix the TCS factor-struct type for triangle programs and recompile the
 * kernel. SPIRV-Cross types the factor buffer as MTLQuadTessellationFactorsHalf (12-
 * byte records) because the TCS alone doesn't know the domain, and the domain is only
 * known once the TES linked. The corrected function lives on the MGLTessProgram and
 * is preferred by the draw path. */

/* ---- tests-only capture access (mgl_tess_diag.h) ---- */
extern GLMContext _ctx;

void mgl_tess_diag_capture(GLuint program, int enable)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)(_ctx ? _ctx->state.ext_tess : NULL);

    (void)program;
    if (S && _ctx)
    {
        MGLTessProgram *tp = mgl_tess_active_program(_ctx);
        if (tess_trace())
            fprintf(stderr, "MGL tess: diag_capture(enable=%d): _ctx=%p tp=%p\n",
                    enable, (void *)_ctx, (void *)tp);
        if (tp)
            tp->capture = enable != 0;
    }
}

bool mgl_tess_diag_capture_info(void **buffer, uint32_t *records_offset, uint32_t *stride_words,
                                uint32_t *prim, uint32_t *count)
{
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)(_ctx ? _ctx->state.ext_tess : NULL);

    if (!S || !S->capture.valid || !buffer)
        return false;
    *buffer = S->capture.buffer;
    if (records_offset) *records_offset = (uint32_t)S->capture.records_offset;
    if (stride_words)   *stride_words = S->capture.stride;
    if (prim)           *prim = S->capture.prim;
    if (count)
    {
        const uint32_t *w = (const uint32_t *)((const uint8_t *)S->capture.buffer +
                                               S->capture.count_offset);
        *count = *w;
        if (tess_trace())
            fprintf(stderr, "MGL tess: diag capture_info: buf=%p count_off=%zu val=%u "
                            "rec_off=%zu idx_off=%zu\n",
                    S->capture.buffer, (size_t)S->capture.count_offset, *w,
                    (size_t)S->capture.records_offset, (size_t)S->capture.index_offset);
    }
    return true;
}

/* test-only: the CPU (D3D11 reference) tessellator's counts for one patch */
bool mgl_tess_diag_cpu_counts(int domain, int spacing, int ccw, int point_mode,
                              const float levels[6], uint32_t *nvertices, uint32_t *nindices)
{
    struct D3DTess t;

    if (domain < 0 || domain > 2 || spacing < 0 || spacing > 2 || !levels)
        return false;
    memset(&t, 0, sizeof t);
    t.mesh = NULL;
    t.pid = 0.0f;
    t.ccw = ccw != 0;
    t.tri = (domain == MGL_TESS_DOMAIN_TRIANGLES);
    t.base = 0;
    t.count_only = true;
    switch (domain)
    {
        case MGL_TESS_DOMAIN_ISOLINES:  d3d_isoline_process(&t, spacing, levels); break;
        case MGL_TESS_DOMAIN_TRIANGLES: d3d_tri_process(&t, spacing, levels); break;
        default:                        d3d_quad_process(&t, spacing, levels); break;
    }
    if (nvertices) *nvertices = t.ncoord;
    if (nindices) *nindices = point_mode ? t.ncoord : t.nindex;
    return true;
}

/* test-only: the CPU (D3D11 reference) tessellator's full stream for ONE patch.
 * coords gets 4 floats per vertex (u, v, w, 0), indices the triangle/line list;
 * returns the vertex count (or -1 when the buffers are too small). */
int mgl_tess_diag_cpu_mesh(int domain, int spacing, int ccw, int point_mode,
                           const float levels[6], float *coords, int max_coords,
                           uint32_t *indices, int max_indices, int *out_index_count)
{
    struct TessMesh mesh = { NULL, 0, 0, NULL, 0, 0 };
    struct D3DTess t;
    unsigned nv, ni;
    int r = -1;

    memset(&t, 0, sizeof t);
    t.mesh = &mesh;
    t.pid = 0.0f;
    t.ccw = ccw != 0;
    t.tri = (domain == MGL_TESS_DOMAIN_TRIANGLES);
    t.base = 0;
    t.count_only = true;
    switch (domain)
    {
        case MGL_TESS_DOMAIN_ISOLINES:  d3d_isoline_process(&t, spacing, levels); break;
        case MGL_TESS_DOMAIN_TRIANGLES: d3d_tri_process(&t, spacing, levels); break;
        default:                        d3d_quad_process(&t, spacing, levels); break;
    }
    nv = t.ncoord;
    ni = point_mode ? t.ncoord : t.nindex;
    if ((int)nv > max_coords || (int)ni > max_indices)
    {
        if (out_index_count) *out_index_count = -1;
        return -1;
    }
    t.count_only = false;
    switch (domain)
    {
        case MGL_TESS_DOMAIN_ISOLINES:  d3d_isoline_process(&t, spacing, levels); break;
        case MGL_TESS_DOMAIN_TRIANGLES: d3d_tri_process(&t, spacing, levels); break;
        default:                        d3d_quad_process(&t, spacing, levels); break;
    }
    if (point_mode)
    {
        if (!mesh_reserve(&mesh, 0, nv))
            goto out;
        for (unsigned i = 0; i < nv; i++)
            mesh.indices[mesh.nindices++] = i;
    }
    if (mesh.ncoords != nv || mesh.nindices != ni)
        goto out;                         /* internal inconsistency */
    if (coords)
        memcpy(coords, mesh.coords, (size_t)nv * 4 * sizeof(float));
    if (indices)
        memcpy(indices, mesh.indices, (size_t)ni * sizeof(uint32_t));
    if (out_index_count)
        *out_index_count = (int)ni;
    r = (int)nv;
out:
    free(mesh.coords);
    free(mesh.indices);
    return r;
}

/* gl_DrawID: the program whose vertex stage was rewritten to read the draw id
 * from the draw-parameters buffer (program.c sets uses_draw_params).
 *
 * The two gates below are the O(1) per-draw form (glmark2 perf r29): the flags
 * are computed once at link, so a plain VS+FS program pays two loads instead of
 * a pipeline-composite rebuild (mgl_program_pipeline_composite memsets and
 * re-copies the whole Program on EVERY call — and never copied uses_draw_params,
 * so the pipeline arm of this lookup always missed). */
bool mgl_tess_draw_params_bound(GLMContext ctx)
{
    Program *p = ctx->state.program;

    if (p && p->name != 0)
        return p->uses_draw_params != 0;
    if (ctx->state.program_pipeline)
    {
        Program *vs = MGL_PIPELINE_OBJ(ctx->state.program_pipeline)->base.stage_programs[_VERTEX_SHADER];
        return vs != NULL && vs->uses_draw_params != 0;
    }
    return false;
}

/* O(1): a tessellation evaluation stage is bound (program object or separable
 * pipeline) — mirrors mgl_tess_active_program's stage resolution without the
 * per-program tess-record hash walk (a record exists exactly for programs whose
 * link had spirv[_TESS_EVALUATION_SHADER].ir, see mgl_ext_post_link_tess). */
bool mgl_tess_stage_bound(GLMContext ctx)
{
    Program *p = ctx->state.program;

    if (p && p->name != 0)
        return p->spirv[_TESS_EVALUATION_SHADER].ir != NULL;
    if (ctx->state.program_pipeline)
        return MGL_PIPELINE_OBJ(ctx->state.program_pipeline)->base.stage_programs[_TESS_EVALUATION_SHADER] != NULL;
    return false;
}
