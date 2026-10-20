/*
 * compat_vb.c — immediate mode, legacy client arrays and compat draws (compat domain).
 *
 * Immediate-mode vertices accumulate in CPU float streams ("vbo_save" style);
 * at glEnd the streams are uploaded into streaming GL buffers, an internal
 * VAO is set up with the fixed locations, the fixed-function program is bound
 * (unless a user program is active) and the core draw path renders it.
 * GL_QUADS / GL_QUAD_STRIP / GL_POLYGON become triangle index lists with the
 * provoking-vertex (last vertex) preserved for flat shading.
 *
 * Client vertex arrays and VAO 0 are supported by uploading the referenced
 * range of client memory into streaming buffers before each draw.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"
#include "vertex_arrays.h"

#define CS() (ctx->state.ext_compat)

#pragma mark immediate mode accumulator

typedef struct MGLCImm_t {
    GLboolean active;
    GLboolean inside_vertex;        /* glBegin..glEnd validity */
    GLenum    mode;
    GLuint    count, cap;

    float *pos;                     /* 4 floats per vertex */
    float *normal;                  /* 3 */
    float *color;                   /* 4 */
    float *secondary;               /* 4 */
    float *fog;                     /* 1 */
    float *edge;                    /* 1 */
    float *index;                   /* 1 */
    float *texcoord[MGLC_MAX_COORDS];   /* 4 */
} MGLCImm;

static MGLCImm *imm(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->imm)
    {
        s->imm = (MGLCImm *)calloc(1, sizeof(MGLCImm));
        if (!s->imm)
            return NULL;

        s->imm->cap = 256;
        s->imm->pos = (float *)malloc(256 * 4 * sizeof(float));
        s->imm->normal = (float *)malloc(256 * 3 * sizeof(float));
        s->imm->color = (float *)malloc(256 * 4 * sizeof(float));
        s->imm->secondary = (float *)malloc(256 * 4 * sizeof(float));
        s->imm->fog = (float *)malloc(256 * sizeof(float));
        s->imm->edge = (float *)malloc(256 * sizeof(float));
        s->imm->index = (float *)malloc(256 * sizeof(float));
        for (int u = 0; u < MGLC_MAX_COORDS; u++)
            s->imm->texcoord[u] = (float *)malloc(256 * 4 * sizeof(float));

        if (!s->imm->pos || !s->imm->normal || !s->imm->color || !s->imm->secondary ||
            !s->imm->fog || !s->imm->edge || !s->imm->index)
            return NULL;
    }

    return (MGLCImm *)s->imm;
}

static bool imm_grow(MGLCImm *im)
{
    GLuint cap = im->cap * 2;

    float *p = (float *)realloc(im->pos, cap * 4 * sizeof(float));
    if (!p) return false;
    im->pos = p;
    float *n = (float *)realloc(im->normal, cap * 3 * sizeof(float));
    if (!n) return false;
    im->normal = n;
    float *c = (float *)realloc(im->color, cap * 4 * sizeof(float));
    if (!c) return false;
    im->color = c;
    float *sc = (float *)realloc(im->secondary, cap * 4 * sizeof(float));
    if (!sc) return false;
    im->secondary = sc;
    float *f = (float *)realloc(im->fog, cap * sizeof(float));
    if (!f) return false;
    im->fog = f;
    float *e = (float *)realloc(im->edge, cap * sizeof(float));
    if (!e) return false;
    im->edge = e;
    float *ix = (float *)realloc(im->index, cap * sizeof(float));
    if (!ix) return false;
    im->index = ix;
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
    {
        float *t = (float *)realloc(im->texcoord[u], cap * 4 * sizeof(float));
        if (!t) return false;
        im->texcoord[u] = t;
    }

    im->cap = cap;
    return true;
}

/* one vertex worth of current state is appended to every stream */
static void imm_emit(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();
    MGLCImm *im = imm(ctx);

    if (!im || !im->active || im->inside_vertex)
        return;

    if (im->count >= im->cap && !imm_grow(im))
        return;

    GLuint i = im->count;

    memcpy(&im->pos[i * 4], s->cur_pos, 4 * sizeof(float));
    memcpy(&im->normal[i * 3], s->cur_normal, 3 * sizeof(float));
    memcpy(&im->color[i * 4], s->cur_color, 4 * sizeof(float));
    memcpy(&im->secondary[i * 4], s->cur_secondary, 4 * sizeof(float));
    im->fog[i] = s->cur_fogcoord;
    im->edge[i] = s->cur_edgeflag ? 1.0f : 0.0f;
    im->index[i] = s->cur_index;
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        memcpy(&im->texcoord[u][i * 4], s->cur_texcoord[u], 4 * sizeof(float));

    im->count++;
}

#pragma mark streaming buffer helpers

/* reads one component of a legacy client array element as float */
static float readf(const void *p, GLenum type, int comp, GLboolean normalized)
{
    float v = 0.0f;

    switch (type)
    {
        case GL_BYTE: v = (float)((const GLbyte *)p)[comp]; if (normalized) v /= 127.0f; break;
        case GL_SHORT: v = (float)((const GLshort *)p)[comp]; if (normalized) v /= 32767.0f; break;
        case GL_INT: v = (float)((const GLint *)p)[comp]; if (normalized) v /= 2147483647.0f; break;
        case GL_UNSIGNED_BYTE: v = (float)((const GLubyte *)p)[comp]; if (normalized) v /= 255.0f; break;
        case GL_UNSIGNED_SHORT: v = (float)((const GLushort *)p)[comp]; if (normalized) v /= 65535.0f; break;
        case GL_UNSIGNED_INT: v = (float)((const GLuint *)p)[comp]; if (normalized) v /= 4294967295.0f; break;
        case GL_FLOAT: v = ((const GLfloat *)p)[comp]; break;
        case GL_DOUBLE: v = (float)((const GLdouble *)p)[comp]; break;
        case GL_HALF_FLOAT: { uint16_t h = ((const uint16_t *)p)[comp];
            /* half -> float (IEEE 754) */
            uint32_t sign = (uint32_t)(h >> 15) << 31;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t man = h & 0x3FF;
            uint32_t bits;
            if (exp == 0) bits = sign;
            else if (exp == 31) bits = sign | 0x7F800000 | (man << 13);
            else { exp += 127 - 15; bits = sign | (exp << 23) | (man << 13); }
            float f; memcpy(&f, &bits, 4); v = f; break; }
        default: v = 0.0f; break;
    }

    return v;
}

static GLuint scratch_buffer(GLMContext ctx, int index)
{
    struct MGLExtState_compat *s = CS();

    if (index >= 0)
    {
        if (!s->scratch_vbo[index])
        {
            GLuint name = 0;
            mglGenBuffers(ctx, 1, &name);
            s->scratch_vbo[index] = name;
        }
        return s->scratch_vbo[index];
    }

    if (!s->scratch_ibo)
    {
        GLuint name = 0;
        mglGenBuffers(ctx, 1, &name);
        s->scratch_ibo = name;
    }
    return s->scratch_ibo;
}

/* upload |count| elements of a client array stream into its scratch buffer and
 * attach it to |vao| at |attrib|; returns the buffer name (0 on failure) */
static void upload_stream(GLMContext ctx, VertexArray *vao, GLuint attrib,
                          GLint size, GLenum type, GLboolean normalized,
                          GLsizei stride, const void *pointer, GLuint count)
{
    struct MGLExtState_compat *s = CS();
    size_t esz = mgl_va_element_size(type, size);
    size_t st = stride ? (size_t)stride : esz;
    const char *src = (const char *)pointer;
    Buffer *saved = STATE(buffers[_ARRAY_BUFFER]);

    if (!count || !src)
        return;

    GLuint buf = scratch_buffer(ctx, (int)attrib);

    mglBindBuffer(ctx, GL_ARRAY_BUFFER, buf);
    mglBufferData(ctx, GL_ARRAY_BUFFER, (GLsizeiptr)(st * count), src, GL_STREAM_DRAW);
    mglVertexAttribPointer(ctx, attrib, size, type, normalized, stride, NULL);

    if (saved)
        mglBindBuffer(ctx, GL_ARRAY_BUFFER, saved->name);
    else
        mglBindBuffer(ctx, GL_ARRAY_BUFFER, 0);

    mglEnableVertexAttribArray(ctx, attrib);

    (void)vao; (void)s;
}

/* save / restore of the vertex array binding around internal draws */
typedef struct {
    VertexArray *vao;
    GLuint name;
} VaoSave;

static void compat_bind_imm_vao(GLMContext ctx, VaoSave *save)
{
    struct MGLExtState_compat *s = CS();

    save->vao = STATE(vao);
    save->name = STATE(var.vertex_array_binding);

    if (!s->imm_vao)
        s->imm_vao = newVAO(ctx, 0xfeed0000);

    if (STATE(vao) != s->imm_vao)
    {
        STATE(vao) = s->imm_vao;
        STATE(var.vertex_array_binding) = 0;
        STATE(dirty_bits) |= DIRTY_VAO;
    }
}

static void compat_restore_vao(GLMContext ctx, const VaoSave *save)
{
    if (STATE(vao) != save->vao)
    {
        STATE(vao) = save->vao;
        STATE(var.vertex_array_binding) = save->name;
        STATE(buffers[_ELEMENT_ARRAY_BUFFER]) = save->vao ? save->vao->element_array.buffer : NULL;
        STATE(dirty_bits) |= DIRTY_VAO;
    }
}

#pragma mark element / index helpers

static void push_index(GLMContext ctx, uint32_t **list, GLuint *count, GLuint *cap, uint32_t v)
{
    if (*count >= *cap)
    {
        *cap = *cap ? *cap * 2 : 256;
        *list = (uint32_t *)realloc(*list, *cap * sizeof(uint32_t));
    }
    if (*list)
        (*list)[(*count)++] = v;
}

/* converts QUADS / QUAD_STRIP / POLYGON to triangles keeping the last vertex
 * of each quad/polygon as the provoking vertex (flat shading); plain modes
 * return 0 (draw them with glDrawArrays) */
static GLuint convert_prim(GLMContext ctx, GLenum mode, GLuint n, uint32_t **out);

GLuint mgl_compat_build_tri_indices(GLMContext ctx, GLenum mode, GLuint n, uint32_t **out)
{
    return convert_prim(ctx, mode, n, out);
}

static GLuint convert_prim(GLMContext ctx, GLenum mode, GLuint n, uint32_t **out)
{
    uint32_t *list = NULL;
    GLuint count = 0, cap = 0;

    /* triangles are emitted rotated so the provoking vertex (the LAST vertex
     * of the quad/polygon) is first: Metal takes flat-shaded values from the
     * first vertex of a triangle, OpenGL from the last */
    switch (mode)
    {
        case GL_QUADS:
            for (GLuint q = 0; q + 3 < n; q += 4)
            {
                push_index(ctx, &list, &count, &cap, q + 3);
                push_index(ctx, &list, &count, &cap, q + 0);
                push_index(ctx, &list, &count, &cap, q + 1);
                push_index(ctx, &list, &count, &cap, q + 3);
                push_index(ctx, &list, &count, &cap, q + 1);
                push_index(ctx, &list, &count, &cap, q + 2);
            }
            break;

        case GL_QUAD_STRIP:
            for (GLuint q = 0; q + 3 < n; q += 2)
            {
                push_index(ctx, &list, &count, &cap, q + 3);
                push_index(ctx, &list, &count, &cap, q + 0);
                push_index(ctx, &list, &count, &cap, q + 1);
                push_index(ctx, &list, &count, &cap, q + 3);
                push_index(ctx, &list, &count, &cap, q + 1);
                push_index(ctx, &list, &count, &cap, q + 2);
            }
            break;

        case GL_POLYGON:
            if (n >= 3)
                for (GLuint i = 0; i + 2 < n; i++)
                {
                    push_index(ctx, &list, &count, &cap, n - 1);
                    push_index(ctx, &list, &count, &cap, i + 0);
                    push_index(ctx, &list, &count, &cap, i + 1);
                }
            break;

        default:
            break;
    }

    *out = list;
    return count;
}

/* draws an index list through the scratch element buffer */
static void draw_index_list(GLMContext ctx, GLenum mode, const uint32_t *idx, GLuint count,
                            GLsizei instances, GLint basevertex)
{
    GLuint buf = scratch_buffer(ctx, -1);
    VertexArray *vao = STATE(vao);
    Buffer *saved = vao ? vao->element_array.buffer : NULL;

    mglBindBuffer(ctx, GL_ELEMENT_ARRAY_BUFFER, buf);
    mglBufferData(ctx, GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(count * sizeof(uint32_t)), idx, GL_STREAM_DRAW);

    if (instances > 1)
        mglDrawElementsInstancedBaseVertex(ctx, mode, (GLsizei)count, GL_UNSIGNED_INT, NULL, instances, basevertex);
    else if (basevertex)
        mglDrawElementsBaseVertex(ctx, mode, (GLsizei)count, GL_UNSIGNED_INT, NULL, basevertex);
    else
        mglDrawElements(ctx, mode, (GLsizei)count, GL_UNSIGNED_INT, NULL);

    /* restore the caller's element binding */
    if (saved)
        mglBindBuffer(ctx, GL_ELEMENT_ARRAY_BUFFER, saved->name);
    else
        mglBindBuffer(ctx, GL_ELEMENT_ARRAY_BUFFER, 0);
}

#pragma mark fixed-function program binding

/* fixed-function default (current) attribute values for every input the
 * generated program reads but the app did not bind (GL: unbound attributes
 * use the current value; MGL implements that as constant attribute fetch) */
/* upload per-vertex line-stipple distances (pixels from the start of the line);
 * GL_LINES resets per line, strips/loops continue through the batch */
static void upload_line_stipple(GLMContext ctx, GLenum mode, const float *pos, GLuint n)
{
    struct MGLExtState_compat *s = CS();

    if (!s->line_stipple)
        return;

    switch (mode)
    {
        case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
            break;
        default:
            return;
    }

    float *li = (float *)malloc((size_t)n * sizeof(float));
    if (!li)
        return;

    float run = 0.0f;

    for (GLuint i = 0; i < n; i++)
    {
        if (mode == GL_LINES && (i & 1) == 0)
            run = 0.0f;     /* each pair is its own primitive */

        if (i > 0)
        {
            float a[3], b[3];

            mgl_compat_window_pos(ctx, &pos[(i - 1) * 4], a);
            mgl_compat_window_pos(ctx, &pos[i * 4], b);
            float dx = a[0] - b[0], dy = a[1] - b[1];
            run += sqrtf(dx * dx + dy * dy);
        }

        li[i] = run;
    }

    upload_stream(ctx, NULL, MGLC_ATTR_LINE_I, 1, GL_FLOAT, GL_FALSE, 0, li, n);
    free(li);
}

static void set_current_attribs(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    mglVertexAttrib4f(ctx, MGLC_ATTR_VERTEX, 0.0f, 0.0f, 0.0f, 1.0f);
    mglVertexAttrib3f(ctx, MGLC_ATTR_NORMAL, s->cur_normal[0], s->cur_normal[1], s->cur_normal[2]);
    mglVertexAttrib4f(ctx, MGLC_ATTR_COLOR, s->cur_color[0], s->cur_color[1], s->cur_color[2], s->cur_color[3]);
    mglVertexAttrib4f(ctx, MGLC_ATTR_SECONDARY, s->cur_secondary[0], s->cur_secondary[1],
                      s->cur_secondary[2], s->cur_secondary[3]);
    mglVertexAttrib1f(ctx, MGLC_ATTR_FOG, s->cur_fogcoord);
    mglVertexAttrib1f(ctx, MGLC_ATTR_EDGEFLAG, s->cur_edgeflag ? 1.0f : 0.0f);
    mglVertexAttrib1f(ctx, MGLC_ATTR_INDEX, s->cur_index);

    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        mglVertexAttrib4f(ctx, MGLC_ATTR_TEXCOORD(u), s->cur_texcoord[u][0], s->cur_texcoord[u][1],
                          s->cur_texcoord[u][2], s->cur_texcoord[u][3]);
}

/* bind the generated fixed-function program (or sync the active user program);
 * returns the program that must be restored afterwards (NULL if none) */
static Program *bind_ffp(GLMContext ctx)
{
    Program *user = STATE(program);

    if (user)
    {
        mgl_compat_sync_builtin_uniforms(ctx, user);
        return NULL;
    }

    GLuint prog = mgl_compat_ffp_program(ctx);

    if (prog)
    {
        mglUseProgram(ctx, prog);
        set_current_attribs(ctx);
        mgl_compat_ffp_sync(ctx, prog);
        return user;    /* NULL */
    }

    return NULL;
}

static void unbind_ffp(GLMContext ctx, Program *user)
{
    if (!user && !STATE(program))
        return;

    if (user)
        mglUseProgram(ctx, user->name);
    else
        mglUseProgram(ctx, 0);
}

#pragma mark polygon mode (point / line rasterisation of polygons)

/* state from compat_state.c */
void mgl_compat_polygon_mode(GLMContext ctx, GLenum face, GLenum mode);

static GLenum polygon_mode_effective(GLMContext ctx, GLenum *other)
{
    struct MGLExtState_compat *s = CS();

    GLenum front = s->polygon_mode_front, back = s->polygon_mode_back;

    if (other)
        *other = (front == GL_FILL) ? back : front;

    return (front != GL_FILL) ? front : back;
}

/* emit GL_LINES / GL_POINTS for a triangle index list */
static void draw_polygon_edges(GLMContext ctx, GLenum mode, const uint32_t *tri, GLuint count,
                               GLsizei instances, GLint basevertex)
{
    uint32_t *list = NULL;
    GLuint n = 0, cap = 0;

    for (GLuint i = 0; i + 2 < count; i += 3)
    {
        uint32_t a = tri[i], b = tri[i + 1], c = tri[i + 2];

        if (mode == GL_LINE)
        {
            push_index(ctx, &list, &n, &cap, a);
            push_index(ctx, &list, &n, &cap, b);
            push_index(ctx, &list, &n, &cap, b);
            push_index(ctx, &list, &n, &cap, c);
            push_index(ctx, &list, &n, &cap, c);
            push_index(ctx, &list, &n, &cap, a);
        }
        else
        {
            push_index(ctx, &list, &n, &cap, a);
            push_index(ctx, &list, &n, &cap, b);
            push_index(ctx, &list, &n, &cap, c);
        }
    }

    if (n)
        draw_index_list(ctx, mode == GL_LINE ? GL_LINES : GL_POINTS, list, n, instances, basevertex);

    free(list);
}

#pragma mark flush immediate mode

void mgl_compat_flush_imm(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();
    MGLCImm *im = (MGLCImm *)s->imm;

    if (!im || !im->count)
        return;

    VaoSave save;
    compat_bind_imm_vao(ctx, &save);

    GLuint n = im->count;

    upload_stream(ctx, s->imm_vao, MGLC_ATTR_VERTEX, 4, GL_FLOAT, GL_FALSE, 0, im->pos, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, im->normal, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_COLOR, 4, GL_FLOAT, GL_FALSE, 0, im->color, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_SECONDARY, 4, GL_FLOAT, GL_FALSE, 0, im->secondary, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_FOG, 1, GL_FLOAT, GL_FALSE, 0, im->fog, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_EDGEFLAG, 1, GL_FLOAT, GL_FALSE, 0, im->edge, n);
    upload_stream(ctx, s->imm_vao, MGLC_ATTR_INDEX, 1, GL_FLOAT, GL_FALSE, 0, im->index, n);
    for (int u = 0; u < MGLC_MAX_COORDS; u++)
        upload_stream(ctx, s->imm_vao, MGLC_ATTR_TEXCOORD(u), 4, GL_FLOAT, GL_FALSE, 0, im->texcoord[u], n);

    upload_line_stipple(ctx, im->mode, im->pos, n);

    Program *user = bind_ffp(ctx);

    /* select / feedback mode handles geometry on the CPU */
    if (s->rmode.mode != GL_RENDER)
    {
        /* feedback records per-vertex color / unit-0 texcoords of the batch */
        mgl_compat_cpu_draw_rgba(ctx, im->mode, im->pos, n, im->color, im->texcoord[0]);
    }
    else
    {
        GLenum mode = im->mode;
        uint32_t *idx = NULL;
        GLuint idxcount = 0;

        switch (mode)
        {
            case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON:
                idxcount = convert_prim(ctx, mode, n, &idx);
                if (idxcount)
                {
                    draw_index_list(ctx, GL_TRIANGLES, idx, idxcount, 1, 0);

                    GLenum other;
                    GLenum pm = polygon_mode_effective(ctx, &other);
                    if (pm != GL_FILL)
                        draw_polygon_edges(ctx, pm, idx, idxcount, 1, 0);

                    free(idx);
                }
                break;

            case GL_TRIANGLES:
                if (CS()->shade_model == GL_FLAT)
                {
                    /* GL flat shading uses the last vertex; rotate so the
                     * last vertex leads (Metal takes the first) */
                    uint32_t *flat = NULL;
                    GLuint fc = 0, fcap = 0;

                    for (GLuint i = 0; i + 2 < n; i += 3)
                    {
                        push_index(ctx, &flat, &fc, &fcap, i + 2);
                        push_index(ctx, &flat, &fc, &fcap, i + 0);
                        push_index(ctx, &flat, &fc, &fcap, i + 1);
                    }

                    if (fc)
                        draw_index_list(ctx, GL_TRIANGLES, flat, fc, 1, 0);

                    free(flat);
                    break;
                }
                /* FALLTHROUGH */
            default:
                mglDrawArrays(ctx, mode, 0, (GLsizei)n);
                break;
        }
    }

    unbind_ffp(ctx, user);
    compat_restore_vao(ctx, &save);

    im->count = 0;
}

#pragma mark glBegin / glEnd

static bool valid_begin_mode(GLenum mode)
{
    switch (mode)
    {
        case GL_POINTS: case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP:
        case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN:
        case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON:
            return true;
    }
    return false;
}

void mglBegin(GLMContext ctx, GLenum mode)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCImm *im;

    if (!valid_begin_mode(mode))
        ERROR_RETURN(GL_INVALID_ENUM);

    im = imm(ctx);

    if (!im)
        ERROR_RETURN(GL_OUT_OF_MEMORY);

    if (im->active)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (s->recording && !s->replaying)
    {
        mgl_compat_list_record_begin(ctx, mode);
        im->active = GL_TRUE;
        im->mode = mode;
        im->count = 0;
        return;
    }

    im->active = GL_TRUE;
    im->mode = mode;
    im->count = 0;
}

void mglEnd(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCImm *im = (MGLCImm *)s->imm;

    if (!im || !im->active)
        ERROR_RETURN(GL_INVALID_OPERATION);

    im->active = GL_FALSE;

    if (s->recording && !s->replaying)
    {
        mgl_compat_list_record_end(ctx);
        return;
    }

    mgl_compat_flush_imm(ctx);
}

#pragma mark vertex attribute entry points

/* glVertex* : completes the current vertex */
#define VXD(n, t, CT) \
void mglVertex##n##t(GLMContext ctx, CT x, CT y) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); MGLCImm *im = imm(ctx); \
  if (!im) return; if (!im->active) ERROR_RETURN(GL_INVALID_OPERATION); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_vertex(ctx, n, (float)x, (float)y, 0.0f, 1.0f); return; } \
  s->cur_pos[0] = (float)x; s->cur_pos[1] = (float)y; s->cur_pos[2] = 0.0f; s->cur_pos[3] = 1.0f; \
  imm_emit(ctx); }

#define VXD3(n, t, CT) \
void mglVertex##n##t(GLMContext ctx, CT x, CT y, CT z) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); MGLCImm *im = imm(ctx); \
  if (!im) return; if (!im->active) ERROR_RETURN(GL_INVALID_OPERATION); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_vertex(ctx, n, (float)x, (float)y, (float)z, 1.0f); return; } \
  s->cur_pos[0] = (float)x; s->cur_pos[1] = (float)y; s->cur_pos[2] = (float)z; s->cur_pos[3] = 1.0f; \
  imm_emit(ctx); }

#define VXD4(n, t, CT) \
void mglVertex##n##t(GLMContext ctx, CT x, CT y, CT z, CT w) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); MGLCImm *im = imm(ctx); \
  if (!im) return; if (!im->active) ERROR_RETURN(GL_INVALID_OPERATION); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_vertex(ctx, n, (float)x, (float)y, (float)z, (float)w); return; } \
  s->cur_pos[0] = (float)x; s->cur_pos[1] = (float)y; s->cur_pos[2] = (float)z; s->cur_pos[3] = (float)w; \
  imm_emit(ctx); }

VXD(2, s, GLshort) VXD(2, i, GLint) VXD(2, f, GLfloat) VXD(2, d, GLdouble)
VXD3(3, s, GLshort) VXD3(3, i, GLint) VXD3(3, f, GLfloat) VXD3(3, d, GLdouble)
VXD4(4, s, GLshort) VXD4(4, i, GLint) VXD4(4, f, GLfloat) VXD4(4, d, GLdouble)

void mglVertex2sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex2s(ctx, v[0], v[1]); }
void mglVertex2iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex2i(ctx, v[0], v[1]); }
void mglVertex2fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex2f(ctx, v[0], v[1]); }
void mglVertex2dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex2d(ctx, v[0], v[1]); }
void mglVertex3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex3s(ctx, v[0], v[1], v[2]); }
void mglVertex3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex3i(ctx, v[0], v[1], v[2]); }
void mglVertex3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex3f(ctx, v[0], v[1], v[2]); }
void mglVertex3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex3d(ctx, v[0], v[1], v[2]); }
void mglVertex4sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex4s(ctx, v[0], v[1], v[2], v[3]); }
void mglVertex4iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex4i(ctx, v[0], v[1], v[2], v[3]); }
void mglVertex4fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex4f(ctx, v[0], v[1], v[2], v[3]); }
void mglVertex4dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex4d(ctx, v[0], v[1], v[2], v[3]); }

/* packed 2101010-rev forms (GL 4.2 "P" variants) */
void mglVertexP2ui(GLMContext ctx, GLenum type, GLuint value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex2f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f); }
void mglVertexP2uiv(GLMContext ctx, GLenum type, const GLuint *value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertexP2ui(ctx, type, value[0]); }
void mglVertexP3ui(GLMContext ctx, GLenum type, GLuint value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex3f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f, ((value >> 20) & 0x3FF) / 1023.0f); }
void mglVertexP3uiv(GLMContext ctx, GLenum type, const GLuint *value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertexP3ui(ctx, type, value[0]); }
void mglVertexP4ui(GLMContext ctx, GLenum type, GLuint value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertex4f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f, ((value >> 20) & 0x3FF) / 1023.0f, ((value >> 30) & 0x3) / 3.0f); }
void mglVertexP4uiv(GLMContext ctx, GLenum type, const GLuint *value) {    MGLC_COMPAT_GUARD(ctx);
 mglVertexP4ui(ctx, type, value[0]); }

/* color */
#define CLR(n, t, CT, DIV, A) \
void mglColor##n##t(GLMContext ctx, CT r, CT g, CT b) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_color(ctx, (float)r / (DIV), (float)g / (DIV), (float)b / (DIV), A); return; } \
  s->cur_color[0] = (float)r / (DIV); s->cur_color[1] = (float)g / (DIV); s->cur_color[2] = (float)b / (DIV); s->cur_color[3] = A; \
  mgl_compat_dirty(ctx); }

#define CLR4(n, t, CT, DIV) \
void mglColor##n##t(GLMContext ctx, CT r, CT g, CT b, CT a) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_color(ctx, (float)r / (DIV), (float)g / (DIV), (float)b / (DIV), (float)a / (DIV)); return; } \
  s->cur_color[0] = (float)r / (DIV); s->cur_color[1] = (float)g / (DIV); s->cur_color[2] = (float)b / (DIV); s->cur_color[3] = (float)a / (DIV); \
  mgl_compat_dirty(ctx); }

CLR(3, b, GLbyte, 127.0f, 1.0f)
CLR(3, s, GLshort, 32767.0f, 1.0f)
CLR(3, i, GLint, 2147483647.0f, 1.0f)
CLR(3, f, GLfloat, 1.0f, 1.0f)
CLR(3, d, GLdouble, 1.0, 1.0f)
CLR(3, ub, GLubyte, 255.0f, 1.0f)
CLR(3, us, GLushort, 65535.0f, 1.0f)
CLR(3, ui, GLuint, 4294967295.0f, 1.0f)
CLR4(4, b, GLbyte, 127.0f)
CLR4(4, s, GLshort, 32767.0f)
CLR4(4, i, GLint, 2147483647.0f)
CLR4(4, f, GLfloat, 1.0f)
CLR4(4, d, GLdouble, 1.0)
CLR4(4, ub, GLubyte, 255.0f)
CLR4(4, us, GLushort, 65535.0f)
CLR4(4, ui, GLuint, 4294967295.0f)

void mglColor3bv(GLMContext ctx, const GLbyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3b(ctx, v[0], v[1], v[2]); }
void mglColor3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3s(ctx, v[0], v[1], v[2]); }
void mglColor3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3i(ctx, v[0], v[1], v[2]); }
void mglColor3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3f(ctx, v[0], v[1], v[2]); }
void mglColor3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3d(ctx, v[0], v[1], v[2]); }
void mglColor3ubv(GLMContext ctx, const GLubyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3ub(ctx, v[0], v[1], v[2]); }
void mglColor3usv(GLMContext ctx, const GLushort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3us(ctx, v[0], v[1], v[2]); }
void mglColor3uiv(GLMContext ctx, const GLuint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor3ui(ctx, v[0], v[1], v[2]); }
void mglColor4bv(GLMContext ctx, const GLbyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4b(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4s(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4i(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4f(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4d(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4ubv(GLMContext ctx, const GLubyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4ub(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4usv(GLMContext ctx, const GLushort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4us(ctx, v[0], v[1], v[2], v[3]); }
void mglColor4uiv(GLMContext ctx, const GLuint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglColor4ui(ctx, v[0], v[1], v[2], v[3]); }

void mglColorP3ui(GLMContext ctx, GLenum type, GLuint value)
{    MGLC_COMPAT_GUARD(ctx);

    mglColor3f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f, ((value >> 20) & 0x3FF) / 1023.0f);
}

void mglColorP3uiv(GLMContext ctx, GLenum type, const GLuint *value) {    MGLC_COMPAT_GUARD(ctx);
 mglColorP3ui(ctx, type, value[0]); }

void mglColorP4ui(GLMContext ctx, GLenum type, GLuint value)
{    MGLC_COMPAT_GUARD(ctx);

    mglColor4f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f, ((value >> 20) & 0x3FF) / 1023.0f, ((value >> 30) & 0x3) / 3.0f);
}

void mglColorP4uiv(GLMContext ctx, GLenum type, const GLuint *value) {    MGLC_COMPAT_GUARD(ctx);
 mglColorP4ui(ctx, type, value[0]); }

/* secondary color */
#define SCLR(n, t, CT, DIV) \
void mglSecondaryColor##n##t(GLMContext ctx, CT r, CT g, CT b) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_secondary(ctx, (float)r / (DIV), (float)g / (DIV), (float)b / (DIV)); return; } \
  s->cur_secondary[0] = (float)r / (DIV); s->cur_secondary[1] = (float)g / (DIV); s->cur_secondary[2] = (float)b / (DIV); s->cur_secondary[3] = 1.0f; \
  mgl_compat_dirty(ctx); }

SCLR(3, b, GLbyte, 127.0f)
SCLR(3, s, GLshort, 32767.0f)
SCLR(3, i, GLint, 2147483647.0f)
SCLR(3, f, GLfloat, 1.0f)
SCLR(3, d, GLdouble, 1.0)
SCLR(3, ub, GLubyte, 255.0f)
SCLR(3, us, GLushort, 65535.0f)
SCLR(3, ui, GLuint, 4294967295.0f)

void mglSecondaryColor3bv(GLMContext ctx, const GLbyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3b(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3s(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3i(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3f(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3d(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3ubv(GLMContext ctx, const GLubyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3ub(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3usv(GLMContext ctx, const GLushort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3us(ctx, v[0], v[1], v[2]); }
void mglSecondaryColor3uiv(GLMContext ctx, const GLuint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglSecondaryColor3ui(ctx, v[0], v[1], v[2]); }

/* normal */
#define NRM(n, t, CT) \
void mglNormal##n##t(GLMContext ctx, CT x, CT y, CT z) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_normal(ctx, (float)x, (float)y, (float)z); return; } \
  s->cur_normal[0] = (float)x; s->cur_normal[1] = (float)y; s->cur_normal[2] = (float)z; }

NRM(3, b, GLbyte) NRM(3, s, GLshort) NRM(3, i, GLint) NRM(3, f, GLfloat) NRM(3, d, GLdouble)

void mglNormal3bv(GLMContext ctx, const GLbyte *v) {    MGLC_COMPAT_GUARD(ctx);
 mglNormal3b(ctx, v[0], v[1], v[2]); }
void mglNormal3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglNormal3s(ctx, v[0], v[1], v[2]); }
void mglNormal3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglNormal3i(ctx, v[0], v[1], v[2]); }
void mglNormal3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglNormal3f(ctx, v[0], v[1], v[2]); }
void mglNormal3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglNormal3d(ctx, v[0], v[1], v[2]); }

/* index (color index mode: tracked, no effect in RGBA mode) */
#define IDX(t, CT) \
void mglIndex##t(GLMContext ctx, CT c) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_index(ctx, (float)c); return; } \
  s->cur_index = (float)c; }

IDX(s, GLshort) IDX(i, GLint) IDX(f, GLfloat) IDX(d, GLdouble)

void mglIndexsv(GLMContext ctx, const GLshort *c) {    MGLC_COMPAT_GUARD(ctx);
 mglIndexs(ctx, c[0]); }
void mglIndexiv(GLMContext ctx, const GLint *c) {    MGLC_COMPAT_GUARD(ctx);
 mglIndexi(ctx, c[0]); }
void mglIndexfv(GLMContext ctx, const GLfloat *c) {    MGLC_COMPAT_GUARD(ctx);
 mglIndexf(ctx, c[0]); }
void mglIndexdv(GLMContext ctx, const GLdouble *c) {    MGLC_COMPAT_GUARD(ctx);
 mglIndexd(ctx, c[0]); }

/* fog coordinate */
#define FOG(t, CT) \
void mglFogCoord##t(GLMContext ctx, CT c) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_fogcoord(ctx, (float)c); return; } \
  s->cur_fogcoord = (float)c; mgl_compat_dirty(ctx); }

FOG(f, GLfloat) FOG(d, GLdouble)

void mglFogCoordfv(GLMContext ctx, const GLfloat *c) {    MGLC_COMPAT_GUARD(ctx);
 mglFogCoordf(ctx, c[0]); }
void mglFogCoorddv(GLMContext ctx, const GLdouble *c) {    MGLC_COMPAT_GUARD(ctx);
 mglFogCoordd(ctx, c[0]); }

/* edge flag */
void mglEdgeFlag(GLMContext ctx, GLboolean flag)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->recording && !s->replaying)
    {
        mgl_compat_list_record_edgeflag(ctx, flag);
        return;
    }

    s->cur_edgeflag = flag;
}

void mglEdgeFlagv(GLMContext ctx, const GLboolean *flag)
{    MGLC_COMPAT_GUARD(ctx);

    if (!flag)
        return;
    mglEdgeFlag(ctx, flag[0]);
}

/* texcoord (active texture unit) */
#define TXD1(t, CT) \
void mglTexCoord1##t(GLMContext ctx, CT x) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = s->client_active_texture % MGLC_MAX_COORDS; \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, 0, 1, (float)x, 0.0f, 0.0f, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = 0.0f; s->cur_texcoord[u][2] = 0.0f; s->cur_texcoord[u][3] = 1.0f; }

#define TXD(n, t, CT) \
void mglTexCoord##n##t(GLMContext ctx, CT x, CT y) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = s->client_active_texture % MGLC_MAX_COORDS; \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, 0, n, (float)x, (float)y, 0.0f, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = 0.0f; s->cur_texcoord[u][3] = 1.0f; }

#define TXD3(n, t, CT) \
void mglTexCoord##n##t(GLMContext ctx, CT x, CT y, CT z) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = s->client_active_texture % MGLC_MAX_COORDS; \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, 0, n, (float)x, (float)y, (float)z, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = (float)z; s->cur_texcoord[u][3] = 1.0f; }

#define TXD4(n, t, CT) \
void mglTexCoord##n##t(GLMContext ctx, CT x, CT y, CT z, CT w) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = s->client_active_texture % MGLC_MAX_COORDS; \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, 0, n, (float)x, (float)y, (float)z, (float)w); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = (float)z; s->cur_texcoord[u][3] = (float)w; }

TXD1(s, GLshort) TXD1(i, GLint) TXD1(f, GLfloat) TXD1(d, GLdouble)
TXD(2, s, GLshort) TXD(2, i, GLint) TXD(2, f, GLfloat) TXD(2, d, GLdouble)
TXD3(3, s, GLshort) TXD3(3, i, GLint) TXD3(3, f, GLfloat) TXD3(3, d, GLdouble)
TXD4(4, s, GLshort) TXD4(4, i, GLint) TXD4(4, f, GLfloat) TXD4(4, d, GLdouble)

void mglTexCoord1sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord1s(ctx, v[0]); }
void mglTexCoord1iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord1i(ctx, v[0]); }
void mglTexCoord1fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord1f(ctx, v[0]); }
void mglTexCoord1dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord1d(ctx, v[0]); }
void mglTexCoord2sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord2s(ctx, v[0], v[1]); }
void mglTexCoord2iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord2i(ctx, v[0], v[1]); }
void mglTexCoord2fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord2f(ctx, v[0], v[1]); }
void mglTexCoord2dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord2d(ctx, v[0], v[1]); }
void mglTexCoord3sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord3s(ctx, v[0], v[1], v[2]); }
void mglTexCoord3iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord3i(ctx, v[0], v[1], v[2]); }
void mglTexCoord3fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord3f(ctx, v[0], v[1], v[2]); }
void mglTexCoord3dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord3d(ctx, v[0], v[1], v[2]); }
void mglTexCoord4sv(GLMContext ctx, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord4s(ctx, v[0], v[1], v[2], v[3]); }
void mglTexCoord4iv(GLMContext ctx, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord4i(ctx, v[0], v[1], v[2], v[3]); }
void mglTexCoord4fv(GLMContext ctx, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord4f(ctx, v[0], v[1], v[2], v[3]); }
void mglTexCoord4dv(GLMContext ctx, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglTexCoord4d(ctx, v[0], v[1], v[2], v[3]); }

/* multi texcoord */
#define MTX1(t, CT) \
void mglMultiTexCoord1##t(GLMContext ctx, GLenum target, CT x) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = target - GL_TEXTURE0; \
  if (target < GL_TEXTURE0 || target >= GL_TEXTURE0 + MGLC_MAX_COORDS) ERROR_RETURN(GL_INVALID_ENUM); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, (int)u, 1, (float)x, 0.0f, 0.0f, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = 0.0f; s->cur_texcoord[u][2] = 0.0f; s->cur_texcoord[u][3] = 1.0f; }

#define MTX(n, t, CT) \
void mglMultiTexCoord##n##t(GLMContext ctx, GLenum target, CT x, CT y) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = target - GL_TEXTURE0; \
  if (target < GL_TEXTURE0 || target >= GL_TEXTURE0 + MGLC_MAX_COORDS) ERROR_RETURN(GL_INVALID_ENUM); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, (int)u, n, (float)x, (float)y, 0.0f, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = 0.0f; s->cur_texcoord[u][3] = 1.0f; }

#define MTX3(n, t, CT) \
void mglMultiTexCoord##n##t(GLMContext ctx, GLenum target, CT x, CT y, CT z) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = target - GL_TEXTURE0; \
  if (target < GL_TEXTURE0 || target >= GL_TEXTURE0 + MGLC_MAX_COORDS) ERROR_RETURN(GL_INVALID_ENUM); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, (int)u, n, (float)x, (float)y, (float)z, 1.0f); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = (float)z; s->cur_texcoord[u][3] = 1.0f; }

#define MTX4(n, t, CT) \
void mglMultiTexCoord##n##t(GLMContext ctx, GLenum target, CT x, CT y, CT z, CT w) \
{ MGLC_COMPAT_GUARD(ctx); struct MGLExtState_compat *s = CS(); GLuint u = target - GL_TEXTURE0; \
  if (target < GL_TEXTURE0 || target >= GL_TEXTURE0 + MGLC_MAX_COORDS) ERROR_RETURN(GL_INVALID_ENUM); \
  if (s->recording && !s->replaying) { mgl_compat_list_record_texcoord(ctx, (int)u, n, (float)x, (float)y, (float)z, (float)w); return; } \
  s->cur_texcoord[u][0] = (float)x; s->cur_texcoord[u][1] = (float)y; s->cur_texcoord[u][2] = (float)z; s->cur_texcoord[u][3] = (float)w; }

MTX1(s, GLshort) MTX1(i, GLint) MTX1(f, GLfloat) MTX1(d, GLdouble)
MTX(2, s, GLshort) MTX(2, i, GLint) MTX(2, f, GLfloat) MTX(2, d, GLdouble)
MTX3(3, s, GLshort) MTX3(3, i, GLint) MTX3(3, f, GLfloat) MTX3(3, d, GLdouble)
MTX4(4, s, GLshort) MTX4(4, i, GLint) MTX4(4, f, GLfloat) MTX4(4, d, GLdouble)

void mglMultiTexCoord1sv(GLMContext ctx, GLenum t, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord1s(ctx, t, v[0]); }
void mglMultiTexCoord1iv(GLMContext ctx, GLenum t, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord1i(ctx, t, v[0]); }
void mglMultiTexCoord1fv(GLMContext ctx, GLenum t, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord1f(ctx, t, v[0]); }
void mglMultiTexCoord1dv(GLMContext ctx, GLenum t, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord1d(ctx, t, v[0]); }
void mglMultiTexCoord2sv(GLMContext ctx, GLenum t, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord2s(ctx, t, v[0], v[1]); }
void mglMultiTexCoord2iv(GLMContext ctx, GLenum t, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord2i(ctx, t, v[0], v[1]); }
void mglMultiTexCoord2fv(GLMContext ctx, GLenum t, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord2f(ctx, t, v[0], v[1]); }
void mglMultiTexCoord2dv(GLMContext ctx, GLenum t, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord2d(ctx, t, v[0], v[1]); }
void mglMultiTexCoord3sv(GLMContext ctx, GLenum t, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord3s(ctx, t, v[0], v[1], v[2]); }
void mglMultiTexCoord3iv(GLMContext ctx, GLenum t, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord3i(ctx, t, v[0], v[1], v[2]); }
void mglMultiTexCoord3fv(GLMContext ctx, GLenum t, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord3f(ctx, t, v[0], v[1], v[2]); }
void mglMultiTexCoord3dv(GLMContext ctx, GLenum t, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord3d(ctx, t, v[0], v[1], v[2]); }
void mglMultiTexCoord4sv(GLMContext ctx, GLenum t, const GLshort *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord4s(ctx, t, v[0], v[1], v[2], v[3]); }
void mglMultiTexCoord4iv(GLMContext ctx, GLenum t, const GLint *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord4i(ctx, t, v[0], v[1], v[2], v[3]); }
void mglMultiTexCoord4fv(GLMContext ctx, GLenum t, const GLfloat *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord4f(ctx, t, v[0], v[1], v[2], v[3]); }
void mglMultiTexCoord4dv(GLMContext ctx, GLenum t, const GLdouble *v) {    MGLC_COMPAT_GUARD(ctx);
 mglMultiTexCoord4d(ctx, t, v[0], v[1], v[2], v[3]); }

#pragma mark glRect*

#define RECT(t, CT) \
static void rect_##t(GLMContext ctx, CT x1, CT y1, CT x2, CT y2) \
{ \
    if (CS()->recording && !CS()->replaying) \
    { \
        mgl_compat_list_record_rect(ctx, (float)x1, (float)y1, (float)x2, (float)y2); \
        return; \
    } \
    mglBegin(ctx, GL_QUADS); \
    mglVertex2##t(ctx, x1, y1); \
    mglVertex2##t(ctx, x2, y1); \
    mglVertex2##t(ctx, x2, y2); \
    mglVertex2##t(ctx, x1, y2); \
    mglEnd(ctx); \
} \
void mglRect##t(GLMContext ctx, CT x1, CT y1, CT x2, CT y2) { rect_##t(ctx, x1, y1, x2, y2); } \
void mglRect##t##v(GLMContext ctx, const CT *v1, const CT *v2) { rect_##t(ctx, v1[0], v1[1], v2[0], v2[1]); }

RECT(s, GLshort)
RECT(i, GLint)
RECT(f, GLfloat)
RECT(d, GLdouble)

#pragma mark legacy client arrays

static int legacy_attrib_index(GLMContext ctx, GLenum array)
{
    switch (array)
    {
        case GL_VERTEX_ARRAY: return MGLC_ATTR_VERTEX;
        case GL_NORMAL_ARRAY: return MGLC_ATTR_NORMAL;
        case GL_COLOR_ARRAY: return MGLC_ATTR_COLOR;
        case GL_SECONDARY_COLOR_ARRAY: return MGLC_ATTR_SECONDARY;
        case GL_FOG_COORD_ARRAY: return MGLC_ATTR_FOG;
        case GL_INDEX_ARRAY: return MGLC_ATTR_INDEX;
        case GL_EDGE_FLAG_ARRAY: return MGLC_ATTR_EDGEFLAG;
        case GL_TEXTURE_COORD_ARRAY:
        {
            int unit = (int)(CS()->client_active_texture % MGLC_MAX_COORDS);
            return MGLC_ATTR_TEXCOORD(unit);
        }
    }
    return -1;
}

static void set_pointer(GLMContext ctx, int index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *ptr)
{
    struct MGLExtState_compat *s = CS();

    if (index < 0 || index >= MAX_ATTRIBS)
        return;

    s->arrays[index].size = size;
    s->arrays[index].type = type;
    s->arrays[index].normalized = normalized;
    s->arrays[index].stride = stride;
    s->arrays[index].pointer = ptr;

    if (s->recording && !s->replaying)
        mgl_compat_list_record_pointer(ctx, index, size, type, normalized, stride, ptr);

    if (s->arrays[index].enabled && VAO())
        mglEnableVertexAttribArray(ctx, index);
}

void mglVertexPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(size >= 2 && size <= 4, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_VERTEX, size, type, GL_FALSE, stride, pointer);
}

void mglNormalPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_NORMAL, 3, type, GL_TRUE, stride, pointer);
}

void mglColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(size >= 3 && size <= 4, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_COLOR, size, type, GL_TRUE, stride, pointer);
}

void mglIndexPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_INDEX, 1, type, GL_FALSE, stride, pointer);
}

void mglTexCoordPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(size >= 1 && size <= 4, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_TEXCOORD(CS()->client_active_texture % MGLC_MAX_COORDS),
                size, type, GL_FALSE, stride, pointer);
}

void mglEdgeFlagPointer(GLMContext ctx, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_EDGEFLAG, 1, GL_UNSIGNED_BYTE, GL_TRUE, stride, pointer);
}

void mglSecondaryColorPointer(GLMContext ctx, GLint size, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(size == 3, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_SECONDARY, size, type, GL_TRUE, stride, pointer);
}

void mglFogCoordPointer(GLMContext ctx, GLenum type, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);
    set_pointer(ctx, MGLC_ATTR_FOG, 1, type, GL_FALSE, stride, pointer);
}

/* glVertexAttribPointer with a client pointer (compat only path) */
void mgl_compat_vertex_attrib_pointer(GLMContext ctx, GLuint index, GLint size, GLenum type,
                                      GLboolean normalized, GLsizei stride, const void *pointer)
{
    struct MGLExtState_compat *s = CS();

    if (STATE(buffers[_ARRAY_BUFFER]))
    {
        /* buffer-backed: ordinary core behaviour */
        mglVertexAttribPointer(ctx, index, size, type, normalized, stride, pointer);
        return;
    }

    set_pointer(ctx, (int)index, size, type, normalized, stride, pointer);

    if (s->arrays[index].enabled)
        mglEnableVertexAttribArray(ctx, (GLuint)index);
}

void mglEnableClientState(GLMContext ctx, GLenum array)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int index = legacy_attrib_index(ctx, array);

    if (index < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
        mgl_compat_list_record_client_state(ctx, array, GL_TRUE);

    s->arrays[index].enabled = GL_TRUE;

    if (s->arrays[index].pointer && VAO())
        mglEnableVertexAttribArray(ctx, index);
}

void mglDisableClientState(GLMContext ctx, GLenum array)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int index = legacy_attrib_index(ctx, array);

    if (index < 0)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
        mgl_compat_list_record_client_state(ctx, array, GL_FALSE);

    s->arrays[index].enabled = GL_FALSE;

    if (VAO())
        mglDisableVertexAttribArray(ctx, index);
}

void mglClientActiveTexture(GLMContext ctx, GLenum texture)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (texture < GL_TEXTURE0 || texture >= GL_TEXTURE0 + TEXTURE_UNITS)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
        mgl_compat_list_record_client_active_texture(ctx, texture);

    s->client_active_texture = texture - GL_TEXTURE0;
}

/* glInterleavedArrays: set the legacy pointers of an interleaved block */
void mglInterleavedArrays(GLMContext ctx, GLenum format, GLsizei stride, const void *pointer)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    int tex = 0, color = 0, normal = 0;
    GLsizei element = 4;

    switch (format)
    {
        case GL_V2F: element = 8; break;
        case GL_V3F: element = 12; break;
        case GL_C4UB_V2F: color = 4; element = 12; break;
        case GL_C4UB_V3F: color = 4; element = 16; break;
        case GL_C3F_V3F: color = 3; element = 24; break;
        case GL_N3F_V3F: normal = 3; element = 24; break;
        case GL_C4F_N3F_V3F: color = 4; normal = 3; element = 40; break;
        case GL_T2F_V3F: tex = 2; element = 20; break;
        case GL_T4F_V4F: tex = 4; element = 32; break;
        case GL_T2F_C4UB_V3F: tex = 2; color = 4; element = 24; break;
        case GL_T2F_C3F_V3F: tex = 2; color = 3; element = 32; break;
        case GL_T2F_N3F_V3F: tex = 2; normal = 3; element = 32; break;
        case GL_T2F_C4F_N3F_V3F: tex = 2; color = 4; normal = 3; element = 48; break;
        case GL_T4F_C4F_N3F_V4F: tex = 4; color = 4; normal = 3; element = 64; break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
    }

    ERROR_CHECK_RETURN(stride >= 0, GL_INVALID_VALUE);

    if (stride == 0)
        stride = element;

    const char *p = (const char *)pointer;
    int unit = (int)(s->client_active_texture % MGLC_MAX_COORDS);
    int vsize = 3;

    switch (format)
    {
        case GL_V2F: case GL_C4UB_V2F: vsize = 2; break;
        case GL_T4F_V4F: case GL_T4F_C4F_N3F_V4F: vsize = 4; break;
        default: vsize = 3; break;
    }

    if (tex)
    {
        set_pointer(ctx, MGLC_ATTR_TEXCOORD(unit), tex, GL_FLOAT, GL_FALSE, stride, p);
        p += tex * 4;
    }
    if (color)
    {
        GLenum type = (color == 4 && (format == GL_C4UB_V2F || format == GL_C4UB_V3F ||
                     format == GL_T2F_C4UB_V3F)) ? GL_UNSIGNED_BYTE : GL_FLOAT;
        set_pointer(ctx, MGLC_ATTR_COLOR, color, type, GL_TRUE, stride, p);
        p += (type == GL_UNSIGNED_BYTE) ? 4 : color * 4;
    }
    if (normal)
    {
        set_pointer(ctx, MGLC_ATTR_NORMAL, 3, GL_FLOAT, GL_TRUE, stride, p);
        p += 3 * 4;
    }

    set_pointer(ctx, MGLC_ATTR_VERTEX, vsize, GL_FLOAT, GL_FALSE, stride, p);

    /* enable exactly these arrays */
    mglEnableClientState(ctx, GL_VERTEX_ARRAY);
    if (tex) mglEnableClientState(ctx, GL_TEXTURE_COORD_ARRAY);
    if (color) mglEnableClientState(ctx, GL_COLOR_ARRAY);
    if (normal) mglEnableClientState(ctx, GL_NORMAL_ARRAY);

    mglDisableClientState(ctx, GL_SECONDARY_COLOR_ARRAY);
    mglDisableClientState(ctx, GL_INDEX_ARRAY);
    mglDisableClientState(ctx, GL_EDGE_FLAG_ARRAY);
    mglDisableClientState(ctx, GL_FOG_COORD_ARRAY);
}

/* glArrayElement: pull one element of every enabled array into the current vertex */
void mglArrayElement(GLMContext ctx, GLint i)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCImm *im = imm(ctx);

    if (!im || !im->active)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (s->recording && !s->replaying)
    {
        mgl_compat_list_record_array_element(ctx, i);
        return;
    }

    for (int index = 0; index < MAX_ATTRIBS; index++)
    {
        MGLCClientArray *a = &s->arrays[index];

        if (!a->enabled || !a->pointer)
            continue;

        GLsizei esz = (GLsizei)mgl_va_element_size(a->type, a->size);
        GLsizei st = a->stride ? a->stride : esz;
        const char *src = (const char *)a->pointer + (size_t)i * st;

        switch (index)
        {
            case MGLC_ATTR_VERTEX:
                for (int k = 0; k < a->size; k++)
                    s->cur_pos[k] = readf(src, a->type, k, a->normalized);
                for (int k = a->size; k < 4; k++)
                    s->cur_pos[k] = (k == 3) ? 1.0f : 0.0f;
                break;
            case MGLC_ATTR_NORMAL:
                for (int k = 0; k < 3; k++)
                    s->cur_normal[k] = readf(src, a->type, k, GL_TRUE);
                break;
            case MGLC_ATTR_COLOR:
                for (int k = 0; k < a->size; k++)
                    s->cur_color[k] = readf(src, a->type, k, GL_TRUE);
                for (int k = a->size; k < 4; k++)
                    s->cur_color[k] = 1.0f;
                break;
            case MGLC_ATTR_SECONDARY:
                for (int k = 0; k < 3; k++)
                    s->cur_secondary[k] = readf(src, a->type, k, GL_TRUE);
                s->cur_secondary[3] = 1.0f;
                break;
            case MGLC_ATTR_FOG:
                s->cur_fogcoord = readf(src, a->type, 0, GL_FALSE);
                break;
            case MGLC_ATTR_EDGEFLAG:
                s->cur_edgeflag = readf(src, a->type, 0, GL_TRUE) != 0.0f;
                break;
            case MGLC_ATTR_INDEX:
                s->cur_index = readf(src, a->type, 0, GL_FALSE);
                break;
            default:
            {
                int unit = index - 8;
                if (unit >= 0 && unit < MGLC_MAX_COORDS)
                {
                    for (int k = 0; k < a->size; k++)
                        s->cur_texcoord[unit][k] = readf(src, a->type, k, GL_FALSE);
                    for (int k = a->size; k < 4; k++)
                        s->cur_texcoord[unit][k] = (k == 3) ? 1.0f : 0.0f;
                }
                break;
            }
        }
    }
}

#pragma mark compat draw entry points (client arrays, VAO 0, quads, polygon mode)

/* uploads every enabled legacy/generic client array covering [min,max] */
void mgl_compat_upload_client_arrays(GLMContext ctx, GLuint min_index, GLuint max_index)
{
    struct MGLExtState_compat *s = CS();
    VertexArray *vao = VAO();

    if (!vao)
        return;

    for (int index = 0; index < MAX_ATTRIBS; index++)
    {
        MGLCClientArray *a = &s->arrays[index];

        if (!a->enabled || !a->pointer)
            continue;

        GLsizei esz = (GLsizei)mgl_va_element_size(a->type, a->size);
        GLsizei st = a->stride ? a->stride : esz;
        const char *src = (const char *)a->pointer + (size_t)min_index * st;

        upload_stream(ctx, vao, index, a->size, a->type, a->normalized, st, src, max_index - min_index + 1);
    }
}

/* reads |count| positions of the current VAO's vertex attribute starting at
 * element |first| (the upload above made element 0 == the lowest index used) */
/* reads `count` elements of an uploaded client array as float[4] per vertex
 * (malloc'd, caller frees); NULL when the array is not readable */
float *mgl_compat_read_array(GLMContext ctx, GLuint attrib, GLuint first, GLuint count)
{
    VertexArray *vao = VAO();

    if (!vao || count == 0)
        return NULL;

    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVertexAttribFormat *f = &va->fmt[attrib];
    MGLVertexBinding *b = &va->binding[f->binding];

    if (!b || !b->buffer || !b->buffer->data.buffer_data)
        return NULL;

    {
        GLsizei stride = b->stride ? b->stride : (GLsizei)mgl_va_element_size(f->type, (GLint)f->size);
        const char *base = (const char *)(uintptr_t)b->buffer->data.buffer_data + b->offset;
        float *out = (float *)calloc((size_t)count * 4, sizeof(float));

        if (!out)
            return NULL;

        for (GLuint i = 0; i < count; i++)
        {
            const void *el = base + (size_t)(first + i) * stride + f->relativeoffset;
            mgl_va_fetch_float(f, el, &out[i * 4]);
        }

        return out;
    }
}

const float *mgl_compat_read_positions(GLMContext ctx, GLuint first, GLuint count)
{
    struct MGLExtState_compat *s = CS();

    free(s->pos_scratch);
    s->pos_scratch = mgl_compat_read_array(ctx, MGLC_ATTR_VERTEX, first, count);
    return s->pos_scratch;
}

/* per-vertex data for feedback: the color and unit-0 texture-coordinate
 * arrays when enabled (NULL = the current values are used) */
void mgl_compat_read_feedback_attribs(GLMContext ctx, GLuint first, GLuint count,
                                      float **col, float **tex)
{
    struct MGLExtState_compat *s = CS();

    *col = NULL;
    *tex = NULL;

    if (s->arrays[MGLC_ATTR_COLOR].enabled)
        *col = mgl_compat_read_array(ctx, MGLC_ATTR_COLOR, first, count);
    if (s->arrays[MGLC_ATTR_TEXCOORD(0)].enabled)
        *tex = mgl_compat_read_array(ctx, MGLC_ATTR_TEXCOORD(0), first, count);
}

/* index range scan */
static void index_range(const uint32_t *idx, GLsizei count, GLuint *mn, GLuint *mx)
{
    *mn = ~0u;
    *mx = 0;

    for (GLsizei i = 0; i < count; i++)
    {
        if (idx[i] < *mn) *mn = idx[i];
        if (idx[i] > *mx) *mx = idx[i];
    }

    if (*mn == ~0u)
        *mn = 0;
}

void mgl_compat_draw(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    struct MGLExtState_compat *s = CS();
    MGLCImm *im = (MGLCImm *)s->imm;

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);
    ERROR_CHECK_RETURN(first >= 0, GL_INVALID_VALUE);

    if (im && im->active)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (count == 0)
        return;

    mgl_compat_upload_client_arrays(ctx, first, first + count - 1);

    const float *pos = mgl_compat_read_positions(ctx, 0, count);

    if (s->rmode.mode != GL_RENDER)
    {
        if (pos)
        {
            float *col = NULL, *tex = NULL;

            mgl_compat_read_feedback_attribs(ctx, 0, count, &col, &tex);
            mgl_compat_cpu_draw_rgba(ctx, mode, pos, count, col, tex);
            free(col);
            free(tex);
        }
        return;
    }

    Program *user = bind_ffp(ctx);

    if (mode == GL_QUADS || mode == GL_QUAD_STRIP || mode == GL_POLYGON)
    {
        uint32_t *idx = NULL;
        GLuint idxcount = mgl_compat_build_tri_indices(ctx, mode, count, &idx);

        if (idxcount)
        {
            draw_index_list(ctx, GL_TRIANGLES, idx, idxcount, 1, 0);

            GLenum other;
            GLenum pm = polygon_mode_effective(ctx, &other);
            if (pm != GL_FILL)
                draw_polygon_edges(ctx, pm, idx, idxcount, 1, 0);

            free(idx);
        }
    }
    else if (mode == GL_TRIANGLES)
    {
        GLenum other;
        GLenum pm = polygon_mode_effective(ctx, &other);

        if (s->shade_model == GL_FLAT)
        {
            uint32_t *flat = NULL;
            GLuint fc = 0, fcap = 0;

            for (GLint i = 0; i + 2 < count; i += 3)
            {
                push_index(ctx, &flat, &fc, &fcap, (uint32_t)(first + i + 2));
                push_index(ctx, &flat, &fc, &fcap, (uint32_t)(first + i + 0));
                push_index(ctx, &flat, &fc, &fcap, (uint32_t)(first + i + 1));
            }

            if (fc)
                draw_index_list(ctx, GL_TRIANGLES, flat, fc, 1, 0);

            free(flat);
            unbind_ffp(ctx, user);
            return;
        }

        mglDrawArrays(ctx, mode, first, count);

        if (pm != GL_FILL)
        {
            uint32_t *idx = NULL;
            GLuint cap = 0, n = 0;

            for (GLint i = 0; i + 2 < count; i += 3)
            {
                push_index(ctx, &idx, &n, &cap, i);
                push_index(ctx, &idx, &n, &cap, i + 1);
                push_index(ctx, &idx, &n, &cap, i + 2);
            }

            if (n)
                draw_polygon_edges(ctx, pm, idx, n, 1, 0);

            free(idx);
        }
    }
    else
    {
        if (s->line_stipple)
        {
            const float *pos = mgl_compat_read_positions(ctx, first, count);
            if (pos)
                upload_line_stipple(ctx, mode, pos, (GLuint)count);
        }

        mglDrawArrays(ctx, mode, first, count);
    }

    unbind_ffp(ctx, user);
}

/* glDrawElements + line stipple: the distance from the start of the strip is
 * a per-vertex attribute, but a vertex shared by two positions of the index
 * stream has two distances.  The draw is de-indexed: every enabled client
 * array is gathered in draw order (element k of the new stream is index[k]),
 * the distances are uploaded for that order, and the draw becomes a plain
 * array draw.  Returns false when it could not (no readable array memory). */
static bool draw_elements_stippled_lines(GLMContext ctx, GLenum mode, const uint32_t *idx,
                                         GLsizei count, const float *pos)
{
    struct MGLExtState_compat *s = CS();
    VertexArray *vao = VAO();
    float *stream = NULL;
    bool ok = true;

    if (!vao || !pos)
        return false;

    /* positions in draw order */
    stream = (float *)malloc((size_t)count * 4 * sizeof(float));
    if (!stream)
        return false;

    for (GLsizei i = 0; i < count; i++)
    {
        stream[i * 4 + 0] = pos[idx[i] * 4 + 0];
        stream[i * 4 + 1] = pos[idx[i] * 4 + 1];
        stream[i * 4 + 2] = pos[idx[i] * 4 + 2];
        stream[i * 4 + 3] = pos[idx[i] * 4 + 3];
    }

    upload_line_stipple(ctx, mode, stream, count);

    /* gather every enabled array in the same order */
    for (int index = 0; index < MAX_ATTRIBS && ok; index++)
    {
        MGLCClientArray *a = &s->arrays[index];

        if (!a->enabled || !a->pointer)
            continue;

        GLsizei esz = (GLsizei)mgl_va_element_size(a->type, a->size);
        GLsizei st = a->stride ? a->stride : esz;
        char *gathered = (char *)malloc((size_t)count * (size_t)esz);

        if (!gathered)
        {
            ok = false;
            break;
        }

        for (GLsizei i = 0; i < count; i++)
            memcpy(gathered + (size_t)i * esz,
                   (const char *)a->pointer + (size_t)idx[i] * st, (size_t)esz);

        upload_stream(ctx, vao, index, a->size, a->type, a->normalized, 0, gathered, count);
        free(gathered);
    }

    if (ok)
        mglDrawArrays(ctx, mode, 0, count);

    free(stream);
    return ok;
}

void mgl_compat_draw_elements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type,
                              const void *indices, GLint basevertex)
{
    struct MGLExtState_compat *s = CS();
    MGLCImm *im = (MGLCImm *)s->imm;

    ERROR_CHECK_RETURN(count >= 0, GL_INVALID_VALUE);

    if (im && im->active)
        ERROR_RETURN(GL_INVALID_OPERATION);

    if (count == 0)
        return;

    VertexArray *vao = VAO();
    const void *src = indices;
    uint32_t *idx32;

    /* resolve the index list (client pointer or element buffer) */
    if (vao && vao->element_array.buffer)
    {
        Buffer *b = vao->element_array.buffer;

        /* a NULL/offset "pointer" indexes into the bound element buffer */
        if (b->data.buffer_data)
            src = (const char *)(uintptr_t)b->data.buffer_data + (size_t)(uintptr_t)indices;
    }

    if (!src)
        return;

    idx32 = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));
    if (!idx32)
        return;

    switch (type)
    {
        case GL_UNSIGNED_BYTE:
            for (GLsizei i = 0; i < count; i++) idx32[i] = ((const GLubyte *)src)[i];
            break;
        case GL_UNSIGNED_SHORT:
            for (GLsizei i = 0; i < count; i++) idx32[i] = ((const GLushort *)src)[i];
            break;
        case GL_UNSIGNED_INT:
            memcpy(idx32, src, (size_t)count * sizeof(uint32_t));
            break;
        default:
            free(idx32);
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    GLuint mn, mx;
    index_range(idx32, count, &mn, &mx);

    mgl_compat_upload_client_arrays(ctx, mn, mx);

    const float *pos = mgl_compat_read_positions(ctx, 0, mx - mn + 1);

    if (s->rmode.mode != GL_RENDER)
    {
        if (pos)
        {
            float *col = NULL, *tex = NULL;
            uint32_t *reb = (uint32_t *)malloc((size_t)count * sizeof(uint32_t));

            /* pos/col/tex element 0 is vertex mn: gather in draw order */
            if (reb)
            {
                for (GLsizei i = 0; i < count; i++)
                    reb[i] = (idx32[i] >= mn) ? idx32[i] - mn : 0;
                mgl_compat_read_feedback_attribs(ctx, 0, mx - mn + 1, &col, &tex);
                mgl_compat_cpu_draw_indexed_rgba(ctx, mode, pos, reb, count, col, tex);
                free(reb);
                free(col);
                free(tex);
            }
        }
        free(idx32);
        return;
    }

    /* rebase to the uploaded range */
    for (GLsizei i = 0; i < count; i++)
        if (idx32[i] >= mn)
            idx32[i] -= mn;

    Program *user = bind_ffp(ctx);

    /* line stipple counts pixels from the start of each strip, which a shared
     * vertex cannot carry as an attribute when a vertex is reused at several
     * places of the index stream: de-index the draw (gather every enabled
     * array in draw order) and feed the distances as a plain array */
    if (s->line_stipple &&
        (mode == GL_LINES || mode == GL_LINE_STRIP || mode == GL_LINE_LOOP) && pos)
    {
        bool ok = draw_elements_stippled_lines(ctx, mode, idx32, count, pos);

        if (ok)
        {
            free(idx32);
            unbind_ffp(ctx, user);
            return;
        }
    }

    GLenum draw_mode = mode;
    uint32_t *draw_idx = idx32;
    GLuint draw_count = count;
    uint32_t *converted = NULL;

    if (mode == GL_QUADS || mode == GL_QUAD_STRIP || mode == GL_POLYGON)
    {
        GLuint n = mgl_compat_build_tri_indices(ctx, mode, count, &converted);

        if (n)
        {
            for (GLuint i = 0; i < n; i++)
                if (converted[i] < count)
                    converted[i] = idx32[converted[i]];

            draw_mode = GL_TRIANGLES;
            draw_idx = converted;
            draw_count = n;
        }
    }

    draw_index_list(ctx, draw_mode, draw_idx, draw_count, 1, basevertex);

    {
        GLenum other;
        GLenum pm = polygon_mode_effective(ctx, &other);

        if (pm != GL_FILL && draw_count)
            draw_polygon_edges(ctx, pm, draw_idx, draw_count, 1, basevertex);
    }

    free(converted);
    free(idx32);

    unbind_ffp(ctx, user);
}

void mgl_compat_cpu_draw_indexed(GLMContext ctx, GLenum mode, const float *pos, const uint32_t *idx, GLuint count)
{
    mgl_compat_cpu_draw_indexed_rgba(ctx, mode, pos, idx, count, NULL, NULL);
}

/* with per-vertex color / texcoords gathered through the indices */
void mgl_compat_cpu_draw_indexed_rgba(GLMContext ctx, GLenum mode, const float *pos, const uint32_t *idx,
                                      GLuint count, const float *col, const float *tex)
{
    if (!pos || !idx || !count)
        return;

    float *stream = (float *)malloc((size_t)count * 4 * sizeof(float));
    float *col_s = col ? (float *)malloc((size_t)count * 4 * sizeof(float)) : NULL;
    float *tex_s = tex ? (float *)malloc((size_t)count * 4 * sizeof(float)) : NULL;

    if (!stream || (col && !col_s) || (tex && !tex_s))
    {
        free(stream);
        free(col_s);
        free(tex_s);
        return;
    }

    for (GLuint i = 0; i < count; i++)
    {
        stream[i * 4 + 0] = pos[idx[i] * 4 + 0];
        stream[i * 4 + 1] = pos[idx[i] * 4 + 1];
        stream[i * 4 + 2] = pos[idx[i] * 4 + 2];
        stream[i * 4 + 3] = pos[idx[i] * 4 + 3];

        if (col_s)
            memcpy(&col_s[i * 4], &col[idx[i] * 4], 4 * sizeof(float));
        if (tex_s)
            memcpy(&tex_s[i * 4], &tex[idx[i] * 4], 4 * sizeof(float));
    }

    mgl_compat_cpu_draw_rgba(ctx, mode, stream, count, col_s, tex_s);
    free(stream);
    free(col_s);
    free(tex_s);
}

/* display-list replay helper: the recorded pointer call */
void mgl_compat_replay_set_pointer(GLMContext ctx, int index, GLint size, GLenum type,
                                   GLboolean normalized, GLsizei stride, const void *ptr)
{
    struct MGLExtState_compat *s = CS();

    if (index < 0 || index >= MAX_ATTRIBS)
        return;

    s->arrays[index].size = size;
    s->arrays[index].type = type;
    s->arrays[index].normalized = normalized;
    s->arrays[index].stride = stride;
    s->arrays[index].pointer = ptr;
}

void mgl_compat_list_record_simple(GLMContext ctx, uint8_t op, int32_t i0)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, op);
    if (o)
        o->i0 = i0;
}

/* packed 2101010-rev texcoords (GL 4.2 "P" variants) */
#define P2101010(name, N) \
static void name##_packed(GLMContext ctx, GLenum type, GLuint v) \
{ \
    float x = (v & 0x3FF) / 1023.0f, y = ((v >> 10) & 0x3FF) / 1023.0f, z = ((v >> 20) & 0x3FF) / 1023.0f; \
    float w = ((v >> 30) & 0x3) / 3.0f; \
    switch (N) { \
        case 1: mglTexCoord1f(ctx, x); break; \
        case 2: mglTexCoord2f(ctx, x, y); break; \
        case 3: mglTexCoord3f(ctx, x, y, z); break; \
        default: mglTexCoord4f(ctx, x, y, z, w); break; \
    } \
    (void)type; \
} \
void name(GLMContext ctx, GLenum type, GLuint value) { name##_packed(ctx, type, value); } \
void name##v(GLMContext ctx, GLenum type, const GLuint *value) { if (value) name##_packed(ctx, type, value[0]); }

P2101010(mglTexCoordP1ui, 1)
P2101010(mglTexCoordP2ui, 2)
P2101010(mglTexCoordP3ui, 3)
P2101010(mglTexCoordP4ui, 4)

#define MP2101010(name, N) \
static void name##_packed(GLMContext ctx, GLenum target, GLenum type, GLuint v) \
{ \
    float x = (v & 0x3FF) / 1023.0f, y = ((v >> 10) & 0x3FF) / 1023.0f, z = ((v >> 20) & 0x3FF) / 1023.0f; \
    float w = ((v >> 30) & 0x3) / 3.0f; \
    switch (N) { \
        case 1: mglMultiTexCoord1f(ctx, target, x); break; \
        case 2: mglMultiTexCoord2f(ctx, target, x, y); break; \
        case 3: mglMultiTexCoord3f(ctx, target, x, y, z); break; \
        default: mglMultiTexCoord4f(ctx, target, x, y, z, w); break; \
    } \
    (void)type; \
} \
void name(GLMContext ctx, GLenum target, GLenum type, GLuint value) { name##_packed(ctx, target, type, value); } \
void name##v(GLMContext ctx, GLenum target, GLenum type, const GLuint *value) { if (value) name##_packed(ctx, target, type, value[0]); }

MP2101010(mglMultiTexCoordP1ui, 1)
MP2101010(mglMultiTexCoordP2ui, 2)
MP2101010(mglMultiTexCoordP3ui, 3)
MP2101010(mglMultiTexCoordP4ui, 4)

void mglNormalP3ui(GLMContext ctx, GLenum type, GLuint value)
{    MGLC_COMPAT_GUARD(ctx);

    mglNormal3f(ctx, (value & 0x3FF) / 511.0f, ((value >> 10) & 0x3FF) / 511.0f, ((value >> 20) & 0x3FF) / 511.0f);
}

void mglNormalP3uiv(GLMContext ctx, GLenum type, const GLuint *value)
{    MGLC_COMPAT_GUARD(ctx);

    if (value) mglNormalP3ui(ctx, type, value[0]);
}

void mglSecondaryColorP3ui(GLMContext ctx, GLenum type, GLuint value)
{    MGLC_COMPAT_GUARD(ctx);

    mglSecondaryColor3f(ctx, (value & 0x3FF) / 1023.0f, ((value >> 10) & 0x3FF) / 1023.0f, ((value >> 20) & 0x3FF) / 1023.0f);
}

void mglSecondaryColorP3uiv(GLMContext ctx, GLenum type, const GLuint *value)
{    MGLC_COMPAT_GUARD(ctx);

    if (value) mglSecondaryColorP3ui(ctx, type, value[0]);
}

/* color index variants */
void mglIndexub(GLMContext ctx, GLubyte c) {    MGLC_COMPAT_GUARD(ctx);
 mglIndexf(ctx, (GLfloat)(c / 255.0f)); }
void mglIndexubv(GLMContext ctx, const GLubyte *c) {    MGLC_COMPAT_GUARD(ctx);
 if (c) mglIndexf(ctx, (GLfloat)(c[0] / 255.0f)); }

#pragma mark robustness getters (GL_robust_buffer_access_behavior variants)

void mglGetnMapdv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLdouble *v)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetMapdv(ctx, target, query, v);
}

void mglGetnMapfv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLfloat *v)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetMapfv(ctx, target, query, v);
}

void mglGetnMapiv(GLMContext ctx, GLenum target, GLenum query, GLsizei bufSize, GLint *v)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetMapiv(ctx, target, query, v);
}

void mglGetnPixelMapfv(GLMContext ctx, GLenum map, GLsizei bufSize, GLfloat *values)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetPixelMapfv(ctx, map, values);
}

void mglGetnPixelMapuiv(GLMContext ctx, GLenum map, GLsizei bufSize, GLuint *values)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetPixelMapuiv(ctx, map, values);
}

void mglGetnPixelMapusv(GLMContext ctx, GLenum map, GLsizei bufSize, GLushort *values)
{    MGLC_COMPAT_GUARD(ctx);

    if (bufSize < 0)
        ERROR_RETURN(GL_INVALID_VALUE);
    mglGetPixelMapusv(ctx, map, values);
}
