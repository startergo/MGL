/*
 * state_ext.c
 * MGL — indexed viewport/scissor, clip control, sample and raster state,
 *       shader fixups (clip space / point size / point coord / sample mask),
 *       primitive restart + provoking vertex index rewriting, rstate glGet hooks.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#define MGL_RSTATE_INTERNAL 1
#include "mgl_ext_rstate.h"

GLboolean mglIsEnabled(GLMContext ctx, GLenum cap);
void mgl_rstate_sync_viewport0(GLMContext ctx);
void mgl_rstate_sync_scissor0(GLMContext ctx);

#pragma mark - helpers

static GLdouble rs_clamp01(GLdouble a)
{
    if (a < 0.0) return 0.0;
    if (a > 1.0) return 1.0;
    return a;
}

static GLuint rs_max_viewports(GLMContext ctx)
{
    GLuint n = ctx->state.var.max_viewports;
    if (n == 0 || n > MGL_RSTATE_MAX_VIEWPORTS) n = MGL_RSTATE_MAX_VIEWPORTS;
    return n;
}

static GLfloat rs_clamp_viewport_coord(GLfloat v)
{
    if (v < -MGL_RSTATE_VIEWPORT_BOUNDS) return -MGL_RSTATE_VIEWPORT_BOUNDS;
    if (v > MGL_RSTATE_VIEWPORT_BOUNDS - 1.0f) return MGL_RSTATE_VIEWPORT_BOUNDS - 1.0f;
    return v;
}

static GLfloat rs_clamp_viewport_dim(GLMContext ctx, GLfloat v)
{
    GLfloat max = ctx->state.var.max_viewport_dims ? (GLfloat)ctx->state.var.max_viewport_dims : 16384.0f;
    return v > max ? max : v;
}

/* store viewport i without error checking (callers validate) */
static void rs_set_viewport(GLMContext ctx, GLuint i, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
    RS()->viewport[i][0] = rs_clamp_viewport_coord(x);
    RS()->viewport[i][1] = rs_clamp_viewport_coord(y);
    RS()->viewport[i][2] = rs_clamp_viewport_dim(ctx, w);
    RS()->viewport[i][3] = rs_clamp_viewport_dim(ctx, h);
    RS()->viewport_set_by_app = GL_TRUE;
    if (i == 0) mgl_rstate_sync_viewport0(ctx);
}

static void rs_set_scissor(GLMContext ctx, GLuint i, GLint x, GLint y, GLint w, GLint h)
{
    RS()->scissor[i][0] = x;
    RS()->scissor[i][1] = y;
    RS()->scissor[i][2] = w;
    RS()->scissor[i][3] = h;
    RS()->viewport_set_by_app = GL_TRUE;
    if (i == 0) mgl_rstate_sync_scissor0(ctx);
}

/* index 0 is mirrored into the legacy state fields read by get.c and MGLRenderer */
void mgl_rstate_sync_viewport0(GLMContext ctx)
{
    ctx->state.viewport[0] = (GLint)RS()->viewport[0][0];
    ctx->state.viewport[1] = (GLint)RS()->viewport[0][1];
    ctx->state.viewport[2] = (GLint)RS()->viewport[0][2];
    ctx->state.viewport[3] = (GLint)RS()->viewport[0][3];
}

void mgl_rstate_sync_scissor0(GLMContext ctx)
{
    for (int k = 0; k < 4; k++) ctx->state.var.scissor_box[k] = RS()->scissor[0][k];
}

#pragma mark - glClampColor / glClipControl

void mglClampColor(GLMContext ctx, GLenum target, GLenum clamp)
{
    RS_CHECK(target == GL_CLAMP_READ_COLOR, GL_INVALID_ENUM);
    RS_CHECK(clamp == GL_TRUE || clamp == GL_FALSE || clamp == GL_FIXED_ONLY, GL_INVALID_ENUM);

    RS()->clamp_read_color = clamp;
}

bool mgl_rstate_clamp_read_color(GLMContext ctx, bool read_fixed_point)
{
    if (!ctx || !RS()) return read_fixed_point;
    switch (RS()->clamp_read_color) {
        case GL_TRUE:  return true;
        case GL_FALSE: return false;
        default:       return read_fixed_point;   /* GL_FIXED_ONLY */
    }
}

void mglClipControl(GLMContext ctx, GLenum origin, GLenum depth)
{
    RS_CHECK(origin == GL_LOWER_LEFT || origin == GL_UPPER_LEFT, GL_INVALID_ENUM);
    RS_CHECK(depth == GL_NEGATIVE_ONE_TO_ONE || depth == GL_ZERO_TO_ONE, GL_INVALID_ENUM);

    if (RS()->clip_origin == origin && RS()->clip_depth_mode == depth)
        return;

    RS()->clip_origin = origin;
    RS()->clip_depth_mode = depth;

    /* selects the vertex function specialization (function constants) and the
     * front-face winding handed to Metal */
    mgl_rs_dirty_pipeline(ctx);
}

#pragma mark - indexed color mask

void mglColorMaski(GLMContext ctx, GLuint index, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
    RS_CHECK(index < MAX_COLOR_ATTACHMENTS, GL_INVALID_VALUE);

    RS()->color_mask[index][0] = r ? GL_TRUE : GL_FALSE;
    RS()->color_mask[index][1] = g ? GL_TRUE : GL_FALSE;
    RS()->color_mask[index][2] = b ? GL_TRUE : GL_FALSE;
    RS()->color_mask[index][3] = a ? GL_TRUE : GL_FALSE;

    for (int k = 0; k < 4; k++)
        ctx->state.var.color_writemask[index][k] = RS()->color_mask[index][k];
    ctx->state.caps.use_color_mask[index] = !(r && g && b && a);

    mgl_rs_dirty_pipeline(ctx);
}

#pragma mark - depth range

void mglDepthRangeArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLdouble *v)
{
    RS_CHECK(count >= 0, GL_INVALID_VALUE);
    RS_CHECK((GLuint64)first + (GLuint64)count <= rs_max_viewports(ctx), GL_INVALID_VALUE);
    if (count && !v) return;

    for (GLsizei i = 0; i < count; i++) {
        RS()->depth_range[first + i][0] = rs_clamp01(v[2 * i + 0]);
        RS()->depth_range[first + i][1] = rs_clamp01(v[2 * i + 1]);
        if (first + i == 0) {
            ctx->state.var.depth_range[0] = RS()->depth_range[0][0];
            ctx->state.var.depth_range[1] = RS()->depth_range[0][1];
        }
    }

    mgl_rs_dirty_encoder(ctx);
}

void mglDepthRangeIndexed(GLMContext ctx, GLuint index, GLdouble n, GLdouble f)
{
    GLdouble v[2] = { n, f };

    RS_CHECK(index < rs_max_viewports(ctx), GL_INVALID_VALUE);
    mglDepthRangeArrayv(ctx, index, 1, v);
}

void mglDepthRangef(GLMContext ctx, GLfloat n, GLfloat f)
{
    mglDepthRange(ctx, (GLdouble)n, (GLdouble)f);
}

#pragma mark - sample state

void mglMinSampleShading(GLMContext ctx, GLfloat value)
{
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;

    RS()->min_sample_shading = value;

    /* selects per-sample fragment execution (fragment [[sample_id]] variant) */
    mgl_rs_dirty_pipeline(ctx);
}

void mglSampleCoverage(GLMContext ctx, GLfloat value, GLboolean invert)
{
    if (value < 0.0f) value = 0.0f;
    if (value > 1.0f) value = 1.0f;

    RS()->sample_coverage_value = value;
    RS()->sample_coverage_invert = invert ? GL_TRUE : GL_FALSE;
    ctx->state.var.sample_coverage_value = value;
    ctx->state.var.sample_coverage_invert = RS()->sample_coverage_invert;

    /* folded into the fragment [[sample_mask]] function constant */
    mgl_rs_dirty_pipeline(ctx);
}

void mglSampleMaski(GLMContext ctx, GLuint maskNumber, GLbitfield mask)
{
    RS_CHECK(maskNumber < MGL_RSTATE_MAX_SAMPLE_WORDS, GL_INVALID_VALUE);

    RS()->sample_mask[maskNumber] = mask;

    mgl_rs_dirty_pipeline(ctx);
}

#pragma mark - tessellation patch parameters

void mglPatchParameteri(GLMContext ctx, GLenum pname, GLint value)
{
    RS_CHECK(pname == GL_PATCH_VERTICES, GL_INVALID_ENUM);
    RS_CHECK(value > 0 && value <= 32, GL_INVALID_VALUE);   /* GL_MAX_PATCH_VERTICES */

    RS()->patch_vertices = value;
}

void mglPatchParameterfv(GLMContext ctx, GLenum pname, const GLfloat *values)
{
    switch (pname) {
        case GL_PATCH_DEFAULT_OUTER_LEVEL:
            if (!values) return;
            for (int i = 0; i < 4; i++) RS()->patch_default_outer[i] = values[i];
            break;

        case GL_PATCH_DEFAULT_INNER_LEVEL:
            if (!values) return;
            for (int i = 0; i < 2; i++) RS()->patch_default_inner[i] = values[i];
            break;

        default:
            RS_ERROR(GL_INVALID_ENUM);
    }
}

#pragma mark - polygon offset clamp / restart index / provoking vertex

void mglPolygonOffsetClamp(GLMContext ctx, GLfloat factor, GLfloat units, GLfloat clamp)
{
    RS()->polygon_offset_factor = factor;
    RS()->polygon_offset_units = units;
    RS()->polygon_offset_clamp = clamp;
    ctx->state.var.polygon_offset_factor = factor;
    ctx->state.var.polygon_offset_units = units;

    mgl_rs_dirty_encoder(ctx);
}

void mglPrimitiveRestartIndex(GLMContext ctx, GLuint index)
{
    RS()->primitive_restart_index = index;
    ctx->state.var.primitive_restart_index = index;
}

void mglProvokingVertex(GLMContext ctx, GLenum mode)
{
    RS_CHECK(mode == GL_FIRST_VERTEX_CONVENTION || mode == GL_LAST_VERTEX_CONVENTION, GL_INVALID_ENUM);

    RS()->provoking_vertex = mode;
    ctx->state.var.provoking_vertex = mode;
}

#pragma mark - indexed scissor

void mglScissorArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLint *v)
{
    RS_CHECK(count >= 0, GL_INVALID_VALUE);
    RS_CHECK((GLuint64)first + (GLuint64)count <= rs_max_viewports(ctx), GL_INVALID_VALUE);
    if (count && !v) return;
    for (GLsizei i = 0; i < count; i++)
        RS_CHECK(v[4 * i + 2] >= 0 && v[4 * i + 3] >= 0, GL_INVALID_VALUE);

    for (GLsizei i = 0; i < count; i++)
        rs_set_scissor(ctx, first + i, v[4 * i + 0], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);

    mgl_rs_dirty_encoder(ctx);
}

void mglScissorIndexed(GLMContext ctx, GLuint index, GLint left, GLint bottom, GLsizei width, GLsizei height)
{
    GLint v[4] = { left, bottom, width, height };

    RS_CHECK(index < rs_max_viewports(ctx), GL_INVALID_VALUE);
    mglScissorArrayv(ctx, index, 1, v);
}

void mglScissorIndexedv(GLMContext ctx, GLuint index, const GLint *v)
{
    RS_CHECK(index < rs_max_viewports(ctx), GL_INVALID_VALUE);
    mglScissorArrayv(ctx, index, 1, v);
}

#pragma mark - indexed viewport

void mglViewportArrayv(GLMContext ctx, GLuint first, GLsizei count, const GLfloat *v)
{
    RS_CHECK(count >= 0, GL_INVALID_VALUE);
    RS_CHECK((GLuint64)first + (GLuint64)count <= rs_max_viewports(ctx), GL_INVALID_VALUE);
    if (count && !v) return;
    for (GLsizei i = 0; i < count; i++)
        RS_CHECK(v[4 * i + 2] >= 0.0f && v[4 * i + 3] >= 0.0f, GL_INVALID_VALUE);

    for (GLsizei i = 0; i < count; i++)
        rs_set_viewport(ctx, first + i, v[4 * i + 0], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);

    mgl_rs_dirty_encoder(ctx);
}

void mglViewportIndexedf(GLMContext ctx, GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
    GLfloat v[4] = { x, y, w, h };

    RS_CHECK(index < rs_max_viewports(ctx), GL_INVALID_VALUE);
    mglViewportArrayv(ctx, index, 1, v);
}

void mglViewportIndexedfv(GLMContext ctx, GLuint index, const GLfloat *v)
{
    RS_CHECK(index < rs_max_viewports(ctx), GL_INVALID_VALUE);
    mglViewportArrayv(ctx, index, 1, v);
}

/* used by mglViewport / mglScissor / mglDepthRange in state.c (which apply to all indices) */
void mgl_rstate_set_all_viewports(GLMContext ctx, GLfloat x, GLfloat y, GLfloat w, GLfloat h)
{
    for (GLuint i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++)
        rs_set_viewport(ctx, i, x, y, w, h);
    mgl_rs_dirty_encoder(ctx);
}

void mgl_rstate_set_all_scissors(GLMContext ctx, GLint x, GLint y, GLint w, GLint h)
{
    for (GLuint i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++)
        rs_set_scissor(ctx, i, x, y, w, h);
    mgl_rs_dirty_encoder(ctx);
}

void mgl_rstate_set_all_depth_ranges(GLMContext ctx, GLdouble n, GLdouble f)
{
    for (GLuint i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++) {
        RS()->depth_range[i][0] = rs_clamp01(n);
        RS()->depth_range[i][1] = rs_clamp01(f);
    }
    ctx->state.var.depth_range[0] = RS()->depth_range[0][0];
    ctx->state.var.depth_range[1] = RS()->depth_range[0][1];
    mgl_rs_dirty_encoder(ctx);
}

#pragma mark - MSL fixups (clip space, point size, point coord, sample mask)
/*
 * GL -> Metal clip space differences handled in the vertex shader:
 *   - y: Metal NDC +y maps to texture row 0; GL window y=0 (row 0) is NDC -y.  With
 *     CLIP_ORIGIN = LOWER_LEFT gl_Position.y is negated, so memory row r == GL window row r.
 *     UPPER_LEFT (ARB_clip_control) is exactly the un-negated Metal mapping.
 *   - z: Metal clips 0 <= z <= w.  NEGATIVE_ONE_TO_ONE: z' = (z + w) / 2, so GL's [-1,1] clip
 *     volume is kept and window z = n + (zd+1)/2 (f-n) as GL specifies.  ZERO_TO_ONE: unchanged.
 *   - point size: GL uses glPointSize unless PROGRAM_POINT_SIZE is enabled (Metal requires a
 *     [[point_size]] output for point topologies, so one is added when the shader has none).
 * The choices are Metal function constants with GL-default fallbacks, so the plain
 * newFunctionWithName: function implements the GL defaults and other state picks a cached
 * specialization (no recompile per draw).
 */

#define RS_MARKER "MGL_RSTATE_FIXUP"

typedef struct { char *s; size_t len, cap; } rs_str;

static void rs_str_init(rs_str *b, const char *src)
{
    b->len = strlen(src);
    b->cap = b->len + 4096;
    b->s = malloc(b->cap);
    memcpy(b->s, src, b->len + 1);
}

static void rs_str_insert(rs_str *b, size_t pos, const char *txt)
{
    size_t n = strlen(txt);
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->len + n + 1) * 2;
        b->s = realloc(b->s, b->cap);
    }
    memmove(b->s + pos + n, b->s + pos, b->len - pos + 1);
    memcpy(b->s + pos, txt, n);
    b->len += n;
}

static void rs_str_replace_at(rs_str *b, size_t pos, size_t oldlen, const char *txt)
{
    memmove(b->s + pos, b->s + pos + oldlen, b->len - pos - oldlen + 1);
    b->len -= oldlen;
    rs_str_insert(b, pos, txt);
}

static bool rs_ident_char(char c) { return isalnum((unsigned char)c) || c == '_'; }

/* find "<entry>(" as a whole identifier; returns offset of the identifier or -1 */
static long rs_find_ident_call(const char *s, const char *ident)
{
    size_t n = strlen(ident);
    const char *p = s;
    while ((p = strstr(p, ident))) {
        if ((p == s || !rs_ident_char(p[-1])) && p[n] == '(')
            return p - s;
        p += n;
    }
    return -1;
}

/* given the offset of '(' find matching ')' */
static long rs_match(const char *s, long open, char o, char c)
{
    int depth = 0;
    for (long i = open; s[i]; i++) {
        if (s[i] == o) depth++;
        else if (s[i] == c && --depth == 0) return i;
    }
    return -1;
}

/* identifier that ends right before position end (skipping spaces); copies into out */
static bool rs_ident_before(const char *s, long end, char *out, size_t outsz)
{
    long e = end;
    while (e > 0 && isspace((unsigned char)s[e - 1])) e--;
    long b = e;
    while (b > 0 && rs_ident_char(s[b - 1])) b--;
    if (b == e || (size_t)(e - b) >= outsz) return false;
    memcpy(out, s + b, e - b);
    out[e - b] = 0;
    return true;
}

typedef struct {
    long sig_line;      /* start of the line holding the entry signature */
    long name;          /* offset of entry identifier */
    long params_open, params_close;
    long body_open, body_close;
    char ret_type[256];
} rs_entry;

static bool rs_find_entry(const char *s, const char *entry, rs_entry *e)
{
    e->name = rs_find_ident_call(s, entry);
    if (e->name < 0) return false;
    e->params_open = e->name + (long)strlen(entry);
    e->params_close = rs_match(s, e->params_open, '(', ')');
    if (e->params_close < 0) return false;
    const char *b = strchr(s + e->params_close, '{');
    if (!b) return false;
    e->body_open = b - s;
    e->body_close = rs_match(s, e->body_open, '{', '}');
    if (e->body_close < 0) return false;
    if (!rs_ident_before(s, e->name, e->ret_type, sizeof e->ret_type)) e->ret_type[0] = 0;
    long l = e->name;
    while (l > 0 && s[l - 1] != '\n') l--;
    e->sig_line = l;
    return true;
}

/* member of struct <type> carrying attribute attr (e.g. "[[position") -> name; struct range out */
static bool rs_struct_member_with_attr(const char *s, const char *type, const char *attr,
                                       char *name, size_t namesz, long *struct_open)
{
    char pat[300];
    snprintf(pat, sizeof pat, "struct %s", type);
    const char *p = s;
    while ((p = strstr(p, pat))) {
        const char *q = p + strlen(pat);
        if (!rs_ident_char(*q)) break;
        p = q;
    }
    if (!p) return false;
    const char *open = strchr(p, '{');
    if (!open) return false;
    long o = open - s;
    long c = rs_match(s, o, '{', '}');
    if (c < 0) return false;
    if (struct_open) *struct_open = o;
    if (!attr) return true;
    const char *a = strstr(open, attr);
    if (!a || a - s > c) return false;
    return rs_ident_before(s, a - s, name, namesz);
}

/* replace every "return out;" in [from, to) by "{ <pre> return out; }" */
static void rs_patch_returns(rs_str *b, long from, long *to, const char *pre)
{
    const char *ret = "return out;";
    char repl[1024];
    snprintf(repl, sizeof repl, "{ %s return out; }", pre);
    long pos = from;
    while (1) {
        char *hit = strstr(b->s + pos, ret);
        if (!hit || hit - b->s >= *to) break;
        long at = hit - b->s;
        size_t before = b->len;
        rs_str_replace_at(b, at, strlen(ret), repl);
        *to += (long)(b->len - before);
        pos = at + (long)strlen(repl);
    }
}

bool mgl_rstate_msl_is_patched(const char *msl)
{
    return msl && strstr(msl, RS_MARKER) != NULL;
}

char *mgl_rstate_patch_vertex_msl(const char *msl, const char *entry)
{
    rs_entry e;
    char pos[128], psz[128];
    long struct_open;

    if (!msl || !entry || mgl_rstate_msl_is_patched(msl)) return NULL;
    if (!rs_find_entry(msl, entry, &e) || !e.ret_type[0]) return NULL;
    if (!rs_struct_member_with_attr(msl, e.ret_type, "[[position", pos, sizeof pos, &struct_open))
        return NULL;    /* no gl_Position output: nothing to fix (e.g. rasterizer-discard only) */

    bool has_psz = rs_struct_member_with_attr(msl, e.ret_type, "[[point_size", psz, sizeof psz, NULL);

    rs_str b;
    rs_str_init(&b, msl);

    /* 1) body: fix up before each return (do this first: later edits are before the body) */
    char pre[512];
    snprintf(pre, sizeof pre, "mgl_rs_vertex_fixup(out.%s, out.%s);", pos,
             has_psz ? psz : "mgl_rs_PointSize");
    long body_close = e.body_close;
    rs_patch_returns(&b, e.body_open, &body_close, pre);

    /* 2) prelude before the entry signature line */
    char prelude[2048];
    snprintf(prelude, sizeof prelude,
        "// " RS_MARKER " (rstate: GL clip space / point size on Metal)\n"
        "constant bool mgl_rs_fc_z01_ [[function_constant(%d)]];\n"
        "constant bool mgl_rs_z01 = is_function_constant_defined(mgl_rs_fc_z01_) && mgl_rs_fc_z01_;\n"
        "constant bool mgl_rs_fc_ul_ [[function_constant(%d)]];\n"
        "constant bool mgl_rs_ul = is_function_constant_defined(mgl_rs_fc_ul_) && mgl_rs_fc_ul_;\n"
        "constant bool mgl_rs_fc_pps_ [[function_constant(%d)]];\n"
        "constant bool mgl_rs_pps = is_function_constant_defined(mgl_rs_fc_pps_) && mgl_rs_fc_pps_;\n"
        "constant float mgl_rs_fc_psz_ [[function_constant(%d)]];\n"
        "constant float mgl_rs_psz = is_function_constant_defined(mgl_rs_fc_psz_) ? mgl_rs_fc_psz_ : 1.0;\n"
        "static inline __attribute__((always_inline)) void mgl_rs_vertex_fixup(thread float4 &p, thread float &ps)\n"
        "{\n"
        "    if (!mgl_rs_ul) p.y = -p.y;\n"
        "    if (!mgl_rs_z01) p.z = (p.z + p.w) * 0.5;\n"
        "    if (!mgl_rs_pps%s) ps = mgl_rs_psz;\n"
        "}\n\n",
        MGL_RS_FC_DEPTH_ZERO_TO_ONE, MGL_RS_FC_ORIGIN_UPPER_LEFT,
        MGL_RS_FC_PROGRAM_POINT_SIZE, MGL_RS_FC_POINT_SIZE,
        has_psz ? "" : " || true");
    rs_str_insert(&b, e.sig_line, prelude);

    /* 3) add a [[point_size]] member when the shader has none (struct precedes the entry) */
    if (!has_psz && struct_open < e.sig_line)
        rs_str_insert(&b, struct_open + 1, "\n    float mgl_rs_PointSize [[point_size]];");

    return b.s;
}

char *mgl_rstate_patch_fragment_msl(const char *msl, const char *entry)
{
    rs_entry e;
    char pc[128], sm[128];
    long struct_open = -1;

    if (!msl || !entry || mgl_rstate_msl_is_patched(msl)) return NULL;
    if (!rs_find_entry(msl, entry, &e)) return NULL;

    bool returns_struct = e.ret_type[0] && strcmp(e.ret_type, "void") &&
                          rs_struct_member_with_attr(msl, e.ret_type, NULL, NULL, 0, &struct_open) &&
                          struct_open < e.sig_line;
    bool has_sm = returns_struct &&
                  rs_struct_member_with_attr(msl, e.ret_type, "[[sample_mask", sm, sizeof sm, NULL);

    /* point coord parameter in the signature */
    bool has_pc = false;
    long pc_attr = -1;
    {
        const char *a = strstr(msl + e.params_open, "[[point_coord]]");
        if (a && a - msl < e.params_close && rs_ident_before(msl, a - msl, pc, sizeof pc)) {
            has_pc = true;
            pc_attr = a - msl;
        }
    }

    if (!returns_struct && !has_pc) return NULL;

    rs_str b;
    rs_str_init(&b, msl);
    long body_open = e.body_open, body_close = e.body_close, params_close = e.params_close;

    /* body edits first (they are after the signature) */
    if (returns_struct) {
        char pre[256];
        if (has_sm)
            snprintf(pre, sizeof pre, "out.%s &= mgl_rs_sample_mask(mgl_rs_sid_arg);", sm);
        else
            snprintf(pre, sizeof pre, "out.mgl_rs_SampleMask = mgl_rs_sample_mask(mgl_rs_sid_arg);");
        rs_patch_returns(&b, body_open, &body_close, pre);
    }
    if (has_pc) {
        char init[512];
        snprintf(init, sizeof init,
                 "\n    float2 %s = float2(mgl_rs_pc_in_.x, mgl_rs_pc_lower ? mgl_rs_pc_in_.y : (1.0 - mgl_rs_pc_in_.y));",
                 pc);
        rs_str_insert(&b, body_open + 1, init);
    }

    /* signature edits: per-sample shading argument, renamed point coord */
    {
        bool empty = true;
        for (long i = e.params_open + 1; i < params_close; i++)
            if (!isspace((unsigned char)b.s[i])) { empty = false; break; }
        char arg[256];
        snprintf(arg, sizeof arg, "%suint mgl_rs_sid_ [[sample_id, function_constant(mgl_rs_fc_pss_)]]",
                 empty ? "" : ", ");
        if (returns_struct) rs_str_insert(&b, params_close, arg);
    }
    if (has_pc) {
        /* rename "<pc> [[point_coord]]" -> "mgl_rs_pc_in_ [[point_coord]]" */
        long e2 = pc_attr;
        while (e2 > 0 && isspace((unsigned char)b.s[e2 - 1])) e2--;
        long s2 = e2 - (long)strlen(pc);
        rs_str_replace_at(&b, s2, strlen(pc), "mgl_rs_pc_in_");
    }

    /* prelude */
    char prelude[2048];
    snprintf(prelude, sizeof prelude,
        "// " RS_MARKER " (rstate: point coord origin / sample mask / sample shading on Metal)\n"
        "constant bool mgl_rs_fc_pcl_ [[function_constant(%d)]];\n"
        "constant bool mgl_rs_pc_lower = is_function_constant_defined(mgl_rs_fc_pcl_) && mgl_rs_fc_pcl_;\n"
        "constant uint mgl_rs_fc_smask_ [[function_constant(%d)]];\n"
        "constant uint mgl_rs_smask = is_function_constant_defined(mgl_rs_fc_smask_) ? mgl_rs_fc_smask_ : 0xffffffffu;\n"
        "constant bool mgl_rs_fc_pss_ [[function_constant(%d)]];\n"
        "#define mgl_rs_sid_arg (is_function_constant_defined(mgl_rs_fc_pss_) && mgl_rs_fc_pss_ ? (1u << mgl_rs_sid_) : 0xffffffffu)\n"
        "static inline __attribute__((always_inline)) uint mgl_rs_sample_mask(uint per_sample) { return mgl_rs_smask & per_sample; }\n\n",
        MGL_RS_FC_POINT_ORIGIN_LOWER, MGL_RS_FC_SAMPLE_MASK, MGL_RS_FC_PER_SAMPLE);
    rs_str_insert(&b, e.sig_line, prelude);

    if (returns_struct && !has_sm)
        rs_str_insert(&b, struct_open + 1, "\n    uint mgl_rs_SampleMask [[sample_mask]];");

    return b.s;
}

/* post-link: patch the vertex / fragment MSL of the program and rebuild their Metal functions */
static void rs_post_link(GLMContext ctx, struct Program_t *program)
{
    unsigned mask = 0;

    if (!program) return;

    for (int stage = _VERTEX_SHADER; stage <= _FRAGMENT_SHADER; stage++) {
        if (stage != _VERTEX_SHADER && stage != _FRAGMENT_SHADER) continue;
        Spirv *sp = &program->spirv[stage];
        Shader *sh = program->shader_slots[stage];
        if (!sh || !sp->msl_str) continue;
        const char *entry = sp->entry_point ? sp->entry_point : sh->entry_point;
        char *patched = stage == _VERTEX_SHADER ? mgl_rstate_patch_vertex_msl(sp->msl_str, entry)
                                                : mgl_rstate_patch_fragment_msl(sp->msl_str, entry);
        if (!patched) continue;
        if (getenv("MGL_RSTATE_DUMP_MSL")) fprintf(stderr, "MGL rstate patched MSL:\n%s\n", patched);
        free(sp->msl_str);
        sp->msl_str = patched;
        mask |= 1u << stage;
    }

    if (mask && ctx->mtl_funcs.ext_rstate && ctx->mtl_funcs.ext_rstate->recompile_program)
        ctx->mtl_funcs.ext_rstate->recompile_program(ctx, program, mask);
    else if (mask)
        program->dirty_bits |= DIRTY_PROGRAM;   /* Metal side not up yet: rebuilt on first bind */

    if (mask && ctx->state.program == program)
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
}

#pragma mark - primitive restart / provoking vertex index rewriting

static GLuint rs_index_max(GLenum type)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:  return 0xFFu;
        case GL_UNSIGNED_SHORT: return 0xFFFFu;
        default:                return 0xFFFFFFFFu;
    }
}

static GLuint rs_read_index(const void *p, GLenum type, GLsizei i)
{
    switch (type) {
        case GL_UNSIGNED_BYTE:  return ((const GLubyte *)p)[i];
        case GL_UNSIGNED_SHORT: return ((const GLushort *)p)[i];
        default:                return ((const GLuint *)p)[i];
    }
}

/* is restart active, and with which index value */
static bool rs_restart(GLMContext ctx, GLenum type, GLuint *value)
{
    if (ctx->state.caps.primitive_restart_fixed_index) { *value = rs_index_max(type); return true; }
    if (ctx->state.caps.primitive_restart)             { *value = RS()->primitive_restart_index; return true; }
    return false;
}

static bool rs_is_strip(GLenum mode)
{
    return mode == GL_LINE_STRIP || mode == GL_TRIANGLE_STRIP ||
           mode == GL_LINE_STRIP_ADJACENCY || mode == GL_TRIANGLE_STRIP_ADJACENCY;
}

/* Metal's provoking vertex for flat varyings is the first vertex of each primitive */
static bool rs_program_has_flat(GLMContext ctx)
{
    Program *p = ctx->state.program;
    if (!p || !p->spirv[_FRAGMENT_SHADER].msl_str) return false;
    return strstr(p->spirv[_FRAGMENT_SHADER].msl_str, "[[flat]]") != NULL;
}

static bool rs_provoking_rewrite(GLMContext ctx, GLenum mode)
{
    if (RS()->provoking_vertex != GL_LAST_VERTEX_CONVENTION) return false;
    switch (mode) {
        case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
        case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN:
            return rs_program_has_flat(ctx);
        default:
            return false;
    }
}

bool mgl_rstate_arrays_rewrite_needed(GLMContext ctx, GLenum mode)
{
    if (!ctx || !RS()) return false;
    return rs_provoking_rewrite(ctx, mode);
}

bool mgl_rstate_index_rewrite_needed(GLMContext ctx, GLenum mode, GLenum type,
                                     const void *indices, GLsizei count)
{
    GLuint rv;

    if (!ctx || !RS()) return false;
    if (type == GL_UNSIGNED_BYTE) return true;            /* Metal has no 8-bit index type */
    if (rs_provoking_rewrite(ctx, mode)) return true;
    if (mode == GL_LINE_LOOP || mode == GL_TRIANGLE_FAN) return rs_restart(ctx, type, &rv);

    GLuint metal_max = rs_index_max(type);
    bool restart = rs_restart(ctx, type, &rv);

    if (restart && !rs_is_strip(mode) && mode != GL_POINTS) {
        /* lists: Metal ignores restart; split only if the restart value occurs */
        if (!indices) return true;
        for (GLsizei i = 0; i < count; i++) if (rs_read_index(indices, type, i) == rv) return true;
        return false;
    }
    if (restart && mode == GL_POINTS) {
        if (!indices) return true;
        for (GLsizei i = 0; i < count; i++) if (rs_read_index(indices, type, i) == rv) return true;
        return false;
    }
    if (restart && rv == metal_max) return false;         /* native Metal behaviour */
    if (!rs_is_strip(mode)) return false;                 /* lists never restart in Metal */

    /* strips: Metal restarts at metal_max (must not when GL doesn't) / GL restarts at rv */
    if (!indices) return true;
    for (GLsizei i = 0; i < count; i++) {
        GLuint v = rs_read_index(indices, type, i);
        if (v == metal_max || (restart && v == rv)) return true;
    }
    return false;
}

/* vertices per primitive for list topologies */
static int rs_list_verts(GLenum mode)
{
    switch (mode) {
        case GL_POINTS: return 1;
        case GL_LINES: return 2;
        case GL_TRIANGLES: return 3;
        case GL_LINES_ADJACENCY: return 4;
        case GL_TRIANGLES_ADJACENCY: return 6;
        default: return 0;
    }
}

typedef struct { GLuint *v; GLsizei n, cap; } rs_ilist;
static bool rs_push(rs_ilist *l, GLuint v)
{
    if (l->n == l->cap) {
        GLsizei nc = l->cap ? l->cap * 2 : 64;
        GLuint *nv = realloc(l->v, sizeof(GLuint) * nc);
        if (!nv) return false;
        l->v = nv; l->cap = nc;
    }
    l->v[l->n++] = v;
    return true;
}

/* emit one GL primitive (verts in GL order) as Metal list primitive, rotating so the GL
 * provoking vertex becomes Metal's (first) when LAST_VERTEX_CONVENTION is active */
static bool rs_emit(rs_ilist *l, const GLuint *v, int n, bool last_provoking)
{
    if (!last_provoking || n == 1) {
        for (int i = 0; i < n; i++) if (!rs_push(l, v[i])) return false;
        return true;
    }
    /* rotate (a,b,c)->(c,a,b) keeps the winding; (a,b)->(b,a) for lines */
    if (!rs_push(l, v[n - 1])) return false;
    for (int i = 0; i < n - 1; i++) if (!rs_push(l, v[i])) return false;
    return true;
}

/* process one restart-free run of GL indices */
static bool rs_run(rs_ilist *out, GLenum mode, const GLuint *r, GLsizei n, bool lastpv,
                   bool keep_strip, GLuint metal_restart)
{
    GLuint t[3];

    switch (mode) {
        case GL_POINTS:
            for (GLsizei i = 0; i < n; i++) if (!rs_push(out, r[i])) return false;
            return true;

        case GL_LINES: case GL_TRIANGLES: case GL_LINES_ADJACENCY: case GL_TRIANGLES_ADJACENCY: {
            int k = rs_list_verts(mode);
            bool rot = lastpv && (mode == GL_LINES || mode == GL_TRIANGLES);
            for (GLsizei i = 0; i + k <= n; i += k) if (!rs_emit(out, r + i, k, rot)) return false;
            return true;
        }

        case GL_LINE_STRIP:
            if (keep_strip) {
                if (out->n && !rs_push(out, metal_restart)) return false;
                for (GLsizei i = 0; i < n; i++) if (!rs_push(out, r[i])) return false;
                return true;
            }
            for (GLsizei i = 0; i + 1 < n; i++) { t[0] = r[i]; t[1] = r[i + 1]; if (!rs_emit(out, t, 2, lastpv)) return false; }
            return true;

        case GL_LINE_LOOP:
            for (GLsizei i = 0; i + 1 < n; i++) { t[0] = r[i]; t[1] = r[i + 1]; if (!rs_emit(out, t, 2, lastpv)) return false; }
            if (n > 1) { t[0] = r[n - 1]; t[1] = r[0]; if (!rs_emit(out, t, 2, lastpv)) return false; }
            return true;

        case GL_TRIANGLE_STRIP:
            if (keep_strip) {
                if (out->n && !rs_push(out, metal_restart)) return false;
                for (GLsizei i = 0; i < n; i++) if (!rs_push(out, r[i])) return false;
                return true;
            }
            for (GLsizei i = 0; i + 2 < n; i++) {
                /* GL strip triangle i: (i, i+1, i+2) for even i, (i+1, i, i+2) for odd i;
                 * provoking (LAST) vertex is i+2 in both cases */
                if (i & 1) { t[0] = r[i + 1]; t[1] = r[i]; t[2] = r[i + 2]; }
                else       { t[0] = r[i]; t[1] = r[i + 1]; t[2] = r[i + 2]; }
                if (!rs_emit(out, t, 3, lastpv)) return false;
            }
            return true;

        case GL_TRIANGLE_FAN:
            for (GLsizei i = 1; i + 1 < n; i++) {
                /* GL fan triangle: (0, i, i+1), LAST provoking = i+1 */
                t[0] = r[0]; t[1] = r[i]; t[2] = r[i + 1];
                if (!rs_emit(out, t, 3, lastpv)) return false;
            }
            return true;

        default:   /* adjacency strips: keep as strip */
            if (out->n && !rs_push(out, metal_restart)) return false;
            for (GLsizei i = 0; i < n; i++) if (!rs_push(out, r[i])) return false;
            return true;
    }
}

static GLenum rs_out_mode(GLenum mode, bool keep_strip)
{
    switch (mode) {
        case GL_LINE_STRIP:     return keep_strip ? GL_LINE_STRIP : GL_LINES;
        case GL_LINE_LOOP:      return GL_LINES;
        case GL_TRIANGLE_STRIP: return keep_strip ? GL_TRIANGLE_STRIP : GL_TRIANGLES;
        case GL_TRIANGLE_FAN:   return GL_TRIANGLES;
        default:                return mode;
    }
}

static void *rs_rewrite(GLMContext ctx, GLenum mode, GLenum type, const void *indices, GLint first,
                        GLsizei count, GLenum *out_mode, GLenum *out_type, GLsizei *out_count)
{
    GLuint rv = 0;
    bool restart = indices ? rs_restart(ctx, type, &rv) : false;
    bool lastpv = rs_provoking_rewrite(ctx, mode);
    bool keep_strip = rs_is_strip(mode) && !lastpv;
    rs_ilist out = { 0 }, run = { 0 };
    GLuint maxv = 0;

    for (GLsizei i = 0; i <= count; i++) {
        bool end = (i == count);
        GLuint v = 0;
        if (!end) v = indices ? rs_read_index(indices, type, i) : (GLuint)(first + i);
        if (end || (restart && v == rv)) {
            if (!rs_run(&out, mode, run.v, run.n, lastpv, keep_strip, 0xFFFFFFFFu)) goto oom;
            run.n = 0;
            continue;
        }
        if (!rs_push(&run, v)) goto oom;
        if (v > maxv) maxv = v;
    }
    free(run.v);

    /* choose the narrowest Metal index type that never collides with Metal's restart value */
    bool use16 = maxv < 0xFFFFu && (type != GL_UNSIGNED_INT || maxv < 0xFFFFu);
    *out_mode = rs_out_mode(mode, keep_strip);
    *out_count = out.n;
    if (use16) {
        GLushort *s = malloc(sizeof(GLushort) * (out.n ? out.n : 1));
        if (!s) goto oom2;
        for (GLsizei i = 0; i < out.n; i++) s[i] = out.v[i] == 0xFFFFFFFFu ? 0xFFFF : (GLushort)out.v[i];
        free(out.v);
        *out_type = GL_UNSIGNED_SHORT;
        return s;
    }
    *out_type = GL_UNSIGNED_INT;
    if (!out.v) out.v = malloc(sizeof(GLuint));
    return out.v;

oom:
    free(run.v);
oom2:
    free(out.v);
    return NULL;
}

void *mgl_rstate_rewrite_indices(GLMContext ctx, GLenum mode, GLenum type, const void *indices,
                                 GLsizei count, GLenum *out_mode, GLenum *out_type, GLsizei *out_count)
{
    if (!ctx || !RS() || !indices || count < 0 || !out_mode || !out_type || !out_count) return NULL;
    return rs_rewrite(ctx, mode, type, indices, 0, count, out_mode, out_type, out_count);
}

void *mgl_rstate_generate_array_indices(GLMContext ctx, GLenum mode, GLint first, GLsizei count,
                                        GLenum *out_mode, GLsizei *out_count)
{
    GLenum t;
    void *r;

    if (!ctx || !RS() || count < 0 || !out_mode || !out_count) return NULL;
    r = rs_rewrite(ctx, mode, GL_UNSIGNED_INT, NULL, first, count, out_mode, &t, out_count);
    if (r && t == GL_UNSIGNED_SHORT) {
        /* widen: callers asked for GL_UNSIGNED_INT */
        GLuint *w = malloc(sizeof(GLuint) * (*out_count ? *out_count : 1));
        if (!w) { free(r); return NULL; }
        for (GLsizei i = 0; i < *out_count; i++) w[i] = ((GLushort *)r)[i];
        free(r);
        r = w;
    }
    return r;
}

#pragma mark - domain 'rstate' hooks

static bool rs_get_indexed(GLMContext ctx, GLenum pname, GLuint index, GLuint type, void *data)
{
    switch (pname) {
        case GL_VIEWPORT:
            if (index >= rs_max_viewports(ctx)) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            for (int k = 0; k < 4; k++) mgl_get_store(type, data, k, RS()->viewport[index][k]);
            return true;
        case GL_SCISSOR_BOX:
            if (index >= rs_max_viewports(ctx)) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            for (int k = 0; k < 4; k++) mgl_get_store(type, data, k, RS()->scissor[index][k]);
            return true;
        case GL_DEPTH_RANGE:
            if (index >= rs_max_viewports(ctx)) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            for (int k = 0; k < 2; k++) mgl_get_store(type, data, k, RS()->depth_range[index][k]);
            return true;
        case GL_SCISSOR_TEST:
            if (index >= rs_max_viewports(ctx)) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            mgl_get_store(type, data, 0, RS()->scissor_test[index]);
            return true;
        case GL_BLEND:
            if (index >= MAX_COLOR_ATTACHMENTS) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            mgl_get_store(type, data, 0, RS()->blend[index]);
            return true;
        case GL_COLOR_WRITEMASK:
            if (index >= MAX_COLOR_ATTACHMENTS) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            for (int k = 0; k < 4; k++) mgl_get_store(type, data, k, RS()->color_mask[index][k]);
            return true;
        case GL_BLEND_SRC_RGB: case GL_BLEND_SRC_ALPHA: case GL_BLEND_DST_RGB: case GL_BLEND_DST_ALPHA:
        case GL_BLEND_EQUATION_RGB: case GL_BLEND_EQUATION_ALPHA: {
            if (index >= MAX_COLOR_ATTACHMENTS) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            GLuint v = 0;
            switch (pname) {
                case GL_BLEND_SRC_RGB: v = ctx->state.var.blend_src_rgb[index]; break;
                case GL_BLEND_SRC_ALPHA: v = ctx->state.var.blend_src_alpha[index]; break;
                case GL_BLEND_DST_RGB: v = ctx->state.var.blend_dst_rgb[index]; break;
                case GL_BLEND_DST_ALPHA: v = ctx->state.var.blend_dst_alpha[index]; break;
                case GL_BLEND_EQUATION_RGB: v = ctx->state.var.blend_equation_rgb[index]; break;
                default: v = ctx->state.var.blend_equation_alpha[index]; break;
            }
            mgl_get_store(type, data, 0, v);
            return true;
        }
        case GL_SAMPLE_MASK_VALUE:
            if (index >= MGL_RSTATE_MAX_SAMPLE_WORDS) { ERROR_RETURN(GL_INVALID_VALUE); return true; }
            mgl_get_store(type, data, 0, RS()->sample_mask[index]);
            return true;
        default:
            return false;
    }
}

void mgl_ext_init_rstate(GLMContext ctx)
{
    struct MGLExtState_rstate *rs = calloc(1, sizeof(*rs));
    assert(rs);
    ctx->state.ext_rstate = rs;

    if (ctx->state.var.max_viewports == 0 || ctx->state.var.max_viewports > MGL_RSTATE_MAX_VIEWPORTS)
        ctx->state.var.max_viewports = MGL_RSTATE_MAX_VIEWPORTS;
    if (ctx->state.var.max_sample_mask_words == 0)
        ctx->state.var.max_sample_mask_words = MGL_RSTATE_MAX_SAMPLE_WORDS;
    if (ctx->state.var.max_clip_distances == 0)
        ctx->state.var.max_clip_distances = MAX_CLIP_DISTANCES;
    if (ctx->state.var.max_viewport_dims == 0)
        ctx->state.var.max_viewport_dims = 16384;
    ctx->state.var.viewport_subpixel_bits = 8;
    ctx->state.var.viewport_index_provoking_vertex = GL_PROVOKING_VERTEX;
    ctx->state.var.layer_provoking_vertex = GL_PROVOKING_VERTEX;

    for (int i = 0; i < MGL_RSTATE_MAX_VIEWPORTS; i++) {
        for (int k = 0; k < 4; k++) rs->viewport[i][k] = (GLfloat)ctx->state.viewport[k];
        rs->depth_range[i][0] = 0.0;
        rs->depth_range[i][1] = 1.0;
        for (int k = 0; k < 4; k++) rs->scissor[i][k] = (GLint)ctx->state.var.scissor_box[k];
        rs->scissor_test[i] = GL_FALSE;
    }
    ctx->state.var.depth_range[0] = 0.0;
    ctx->state.var.depth_range[1] = 1.0;
    ctx->state.var.depth_writemask = GL_TRUE;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        rs->blend[i] = GL_FALSE;
        for (int k = 0; k < 4; k++) rs->color_mask[i][k] = GL_TRUE;
    }

    rs->clip_origin = GL_LOWER_LEFT;
    rs->clip_depth_mode = GL_NEGATIVE_ONE_TO_ONE;
    rs->provoking_vertex = GL_LAST_VERTEX_CONVENTION;
    ctx->state.var.provoking_vertex = GL_LAST_VERTEX_CONVENTION;
    rs->polygon_mode = GL_FILL;
    rs->patch_vertices = 3;
    for (int i = 0; i < 4; i++) rs->patch_default_outer[i] = 1.0f;
    for (int i = 0; i < 2; i++) rs->patch_default_inner[i] = 1.0f;
    rs->primitive_restart_index = 0;
    rs->sample_coverage_value = 1.0f;
    ctx->state.var.sample_coverage_value = 1.0f;
    rs->sample_coverage_invert = GL_FALSE;
    for (int i = 0; i < MGL_RSTATE_MAX_SAMPLE_WORDS; i++) rs->sample_mask[i] = 0xFFFFFFFFu;
    rs->min_sample_shading = 0.0f;
    rs->clamp_read_color = GL_FIXED_ONLY;
    rs->point_fade_threshold = 1.0f;
    rs->point_sprite_coord_origin = GL_UPPER_LEFT;
    ctx->state.var.stencil_value_mask = 0xFFFFFFFFu;
    ctx->state.var.stencil_back_value_mask = 0xFFFFFFFFu;
    ctx->state.var.logic_op_mode = GL_COPY;
    ctx->state.hints.line_smooth_hint = GL_DONT_CARE;
    ctx->state.hints.polygon_smooth_hint = GL_DONT_CARE;
    ctx->state.hints.texture_compression_hint = GL_DONT_CARE;
    ctx->state.hints.fragment_shader_derivative_hint = GL_DONT_CARE;

    mgl_ext_register_get_indexed(rs_get_indexed);
    mgl_ext_register_post_link(rs_post_link);
}

/* glGet pnames not covered by get.c */
bool mgl_ext_get_rstate(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname) {
        /* capabilities are queryable with glGet* too */
        case GL_BLEND: case GL_CULL_FACE: case GL_DEPTH_TEST: case GL_STENCIL_TEST: case GL_DITHER:
        case GL_SCISSOR_TEST: case GL_COLOR_LOGIC_OP: case GL_POLYGON_OFFSET_POINT:
        case GL_POLYGON_OFFSET_LINE: case GL_POLYGON_OFFSET_FILL: case GL_MULTISAMPLE:
        case GL_SAMPLE_ALPHA_TO_COVERAGE: case GL_SAMPLE_ALPHA_TO_ONE: case GL_SAMPLE_COVERAGE:
        case GL_RASTERIZER_DISCARD: case GL_FRAMEBUFFER_SRGB: case GL_PRIMITIVE_RESTART:
        case GL_PRIMITIVE_RESTART_FIXED_INDEX: case GL_DEPTH_CLAMP: case GL_TEXTURE_CUBE_MAP_SEAMLESS:
        case GL_SAMPLE_MASK: case GL_SAMPLE_SHADING: case GL_PROGRAM_POINT_SIZE: case GL_LINE_SMOOTH:
        case GL_POLYGON_SMOOTH: case GL_DEBUG_OUTPUT: case GL_DEBUG_OUTPUT_SYNCHRONOUS:
            mgl_get_store(type, data, 0, mglIsEnabled(ctx, pname));
            return true;

        case GL_POLYGON_OFFSET_CLAMP:      mgl_get_store(type, data, 0, RS()->polygon_offset_clamp); return true;
        case GL_CLIP_ORIGIN:               mgl_get_store(type, data, 0, RS()->clip_origin); return true;
        case GL_CLIP_DEPTH_MODE:           mgl_get_store(type, data, 0, RS()->clip_depth_mode); return true;
        case GL_PATCH_VERTICES:            mgl_get_store(type, data, 0, RS()->patch_vertices); return true;
        case GL_PATCH_DEFAULT_OUTER_LEVEL: for (int k = 0; k < 4; k++) mgl_get_store(type, data, k, RS()->patch_default_outer[k]); return true;
        case GL_PATCH_DEFAULT_INNER_LEVEL: for (int k = 0; k < 2; k++) mgl_get_store(type, data, k, RS()->patch_default_inner[k]); return true;
        case GL_MAX_PATCH_VERTICES:        mgl_get_store(type, data, 0, 32); return true;
        case GL_SAMPLE_MASK_VALUE:         mgl_get_store(type, data, 0, RS()->sample_mask[0]); return true;
        case GL_MIN_SAMPLE_SHADING_VALUE:  mgl_get_store(type, data, 0, RS()->min_sample_shading); return true;
        case GL_CLAMP_READ_COLOR:          mgl_get_store(type, data, 0, RS()->clamp_read_color); return true;
        case GL_POINT_FADE_THRESHOLD_SIZE: mgl_get_store(type, data, 0, RS()->point_fade_threshold); return true;
        case GL_POINT_SPRITE_COORD_ORIGIN: mgl_get_store(type, data, 0, RS()->point_sprite_coord_origin); return true;
        case GL_BLEND_EQUATION_RGB:        mgl_get_store(type, data, 0, ctx->state.var.blend_equation_rgb[0]); return true;
        case GL_LINE_SMOOTH_HINT:          mgl_get_store(type, data, 0, ctx->state.hints.line_smooth_hint); return true;
        case GL_POLYGON_SMOOTH_HINT:       mgl_get_store(type, data, 0, ctx->state.hints.polygon_smooth_hint); return true;
        case GL_TEXTURE_COMPRESSION_HINT:  mgl_get_store(type, data, 0, ctx->state.hints.texture_compression_hint); return true;
        case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: mgl_get_store(type, data, 0, ctx->state.hints.fragment_shader_derivative_hint); return true;
        case GL_STENCIL_BACK_REF:          mgl_get_store(type, data, 0, ctx->state.var.stencil_back_ref); return true;
        default:
            return false;
    }
}
