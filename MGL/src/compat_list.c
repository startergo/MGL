/*
 * compat_list.c — display lists and attribute stacks (compat domain).
 *
 * Display lists record compact op records (MGLCListOp).  On replay the same
 * compat entry points are called with the "replaying" flag set so they run
 * their immediate path instead of re-recording.  Vertex data inside a list is
 * captured as glVertex-style ops (Mesa's vbo_save captures whole primitives;
 * the op stream here is equivalent for the sizes we care about).
 *
 * glPushAttrib snapshots GLMCaps + GLMParams + texture unit bindings + the
 * compat scalar state; glPopAttrib restores them and marks everything dirty.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_compat.h"

#define CS() (ctx->state.ext_compat)

#pragma mark list storage

static MGLCList *list_storage(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->lists)
    {
        s->lists = (MGLCList *)calloc(MGLC_LIST_MAX, sizeof(MGLCList));
        s->list_used = (GLboolean *)calloc(MGLC_LIST_MAX, 1);
        if (!s->lists || !s->list_used)
            return NULL;
    }

    return s->lists;
}

static bool s_list_used(GLMContext ctx, GLuint list)
{
    return CS()->list_used && CS()->list_used[list];
}

static MGLCList *find_list(GLMContext ctx, GLuint list)
{
    MGLCList *lists;

    if (!list || list >= MGLC_LIST_MAX)
        return NULL;

    lists = list_storage(ctx);
    if (!lists || !s_list_used(ctx, list))
        return NULL;

    return &lists[list];
}

static void list_free(GLMContext ctx, MGLCList *l)
{
    for (GLuint i = 0; i < l->nops; i++)
        free(l->ops[i].blob);
    free(l->ops);
    memset(l, 0, sizeof *l);
}

void mgl_compat_free_lists(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->lists)
        return;

    for (GLuint i = 0; i < MGLC_LIST_MAX; i++)
        if (s->list_used[i])
            list_free(ctx, &s->lists[i]);

    free(s->lists);
    free(s->list_used);
    s->lists = NULL;
    s->list_used = NULL;
}

MGLCListOp *mgl_compat_list_op(GLMContext ctx, uint8_t op)
{
    struct MGLExtState_compat *s = CS();
    MGLCList *lists = list_storage(ctx);
    MGLCList *l;

    if (!lists || !s->record_list || s->record_list >= MGLC_LIST_MAX)
        return NULL;

    l = &lists[s->record_list];

    if (l->nops >= l->cap)
    {
        GLuint cap = l->cap ? l->cap * 2 : 128;
        MGLCListOp *ops = (MGLCListOp *)realloc(l->ops, cap * sizeof(MGLCListOp));
        if (!ops)
            return NULL;
        memset(ops + l->nops, 0, (cap - l->nops) * sizeof(MGLCListOp));
        l->ops = ops;
        l->cap = cap;
    }

    MGLCListOp *o = &l->ops[l->nops++];
    memset(o, 0, sizeof *o);
    o->op = op;
    s->list_used[s->record_list] = GL_TRUE;
    return o;
}

/* records the op and returns whether the caller should still execute it now */
static bool record(GLMContext ctx, uint8_t op)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return true;

    mgl_compat_list_op(ctx, op);
    return s->record_mode == GL_COMPILE_AND_EXECUTE;
}

#pragma mark recording hooks called from the compat entry points

void mgl_compat_list_record_begin(GLMContext ctx, GLenum mode)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_BEGIN);
    if (o) o->i0 = (int32_t)mode;
}

void mgl_compat_list_record_end(GLMContext ctx)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    mgl_compat_list_op(ctx, MGLC_OP_END);
}

void mgl_compat_list_record_vertex(GLMContext ctx, int n, float x, float y, float z, float w)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_VERTEX4F);
    if (!o) return;
    o->a = (uint8_t)n;
    o->f[0] = x; o->f[1] = y; o->f[2] = z; o->f[3] = w;
}

void mgl_compat_list_record_color(GLMContext ctx, float r, float g, float b, float a)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_COLOR4F);
    if (!o) return;
    o->f[0] = r; o->f[1] = g; o->f[2] = b; o->f[3] = a;
}

void mgl_compat_list_record_secondary(GLMContext ctx, float r, float g, float b)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_SECONDARY3F);
    if (!o) return;
    o->f[0] = r; o->f[1] = g; o->f[2] = b;
}

void mgl_compat_list_record_normal(GLMContext ctx, float x, float y, float z)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_NORMAL3F);
    if (!o) return;
    o->f[0] = x; o->f[1] = y; o->f[2] = z;
}

void mgl_compat_list_record_index(GLMContext ctx, float i)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_INDEXF);
    if (o) o->f[0] = i;
}

void mgl_compat_list_record_fogcoord(GLMContext ctx, float f)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_FOGCOORDF);
    if (o) o->f[0] = f;
}

void mgl_compat_list_record_edgeflag(GLMContext ctx, GLboolean flag)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_EDGEFLAG);
    if (o) o->a = flag ? 1 : 0;
}

void mgl_compat_list_record_texcoord(GLMContext ctx, int unit, int n, float x, float y, float z, float w)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_TEXCOORD4F);
    if (!o) return;
    o->a = (uint8_t)unit;
    o->b = (uint8_t)n;
    o->f[0] = x; o->f[1] = y; o->f[2] = z; o->f[3] = w;
}

void mgl_compat_list_record_array_element(GLMContext ctx, GLint i)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_ARRAY_ELEMENT);
    if (o) o->i0 = i;
}

void mgl_compat_list_record_rect(GLMContext ctx, float x1, float y1, float x2, float y2)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_RECTF);
    if (!o) return;
    o->f[0] = x1; o->f[1] = y1; o->f[2] = x2; o->f[3] = y2;
}

void mgl_compat_list_record_pointer(GLMContext ctx, int index, GLint size, GLenum type,
                                    GLboolean normalized, GLsizei stride, const void *ptr)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_VERTEX_POINTER + index);
    if (!o) return;
    o->i0 = size;
    o->i1 = stride;
    o->a = (uint8_t)(index & 0xFF);
    o->b = (uint8_t)normalized;
    *(int64_t *)(void *)&o->f[12] = (int64_t)(uintptr_t)ptr;
    o->f[0] = (float)type;    /* enums fit in float bits losslessly up to 2^24 */
    {
        uint32_t t = (uint32_t)type;
        memcpy(&o->f[1], &t, 4);
    }
}

void mgl_compat_list_record_client_state(GLMContext ctx, GLenum array, GLboolean enable)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, enable ? MGLC_OP_ENABLE_CLIENT_STATE : MGLC_OP_DISABLE_CLIENT_STATE);
    if (o)
    {
        uint32_t t = (uint32_t)array;
        memcpy(&o->f[0], &t, 4);
    }
}

void mgl_compat_list_record_client_active_texture(GLMContext ctx, GLenum texture)
{
    struct MGLExtState_compat *s = CS();

    if (!s->recording || s->replaying)
        return;

    MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_CLIENT_ACTIVE_TEXTURE);
    if (o)
    {
        uint32_t t = (uint32_t)texture;
        memcpy(&o->f[0], &t, 4);
    }
}

bool mgl_compat_list_recording(GLMContext ctx)
{
    return CS()->recording && !CS()->replaying;
}

#pragma mark glNewList / glEndList / glCallList / glGenLists / ...

void mglNewList(GLMContext ctx, GLuint list, GLenum mode)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCList *lists;

    if (list == 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (mode != GL_COMPILE && mode != GL_COMPILE_AND_EXECUTE)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording)
        ERROR_RETURN(GL_INVALID_OPERATION);

    lists = list_storage(ctx);
    if (!lists)
        ERROR_RETURN(GL_OUT_OF_MEMORY);

    if (s->list_used[list])
        list_free(ctx, &lists[list]);

    s->recording = GL_TRUE;
    s->replaying = GL_FALSE;
    s->record_exec = (mode == GL_COMPILE_AND_EXECUTE);
    s->record_list = list;
    s->record_mode = mode;

    /* a list nested inside another list is executed, not recorded */
    if (s->call_depth > 0)
    {
        s->recording = GL_FALSE;
        s->record_exec = GL_TRUE;
    }
}

void mglEndList(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (!s->recording)
        ERROR_RETURN(GL_INVALID_OPERATION);

    MGLCList *lists = list_storage(ctx);

    if (lists && s->record_list && s->record_list < MGLC_LIST_MAX)
        s->list_used[s->record_list] = GL_TRUE;

    s->recording = GL_FALSE;
    s->record_list = 0;
}

GLuint mglGenLists(GLMContext ctx, GLsizei range)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCList *lists;

    if (range < 0)
    {
        ERROR_RETURN_VALUE(GL_INVALID_VALUE, 0);
        return 0;
    }

    if (range == 0)
        return 0;   /* spec: no lists generated */

    lists = list_storage(ctx);
    if (!lists)
        return 0;

    /* first free block of |range| consecutive ids */
    for (GLuint base = 1; base + range <= MGLC_LIST_MAX; base++)
    {
        bool ok = true;

        for (GLsizei i = 0; i < range; i++)
            if (s->list_used[base + i])
                ok = false;

        if (ok)
        {
            for (GLsizei i = 0; i < range; i++)
            {
                s->list_used[base + i] = GL_TRUE;
                memset(&lists[base + i], 0, sizeof(MGLCList));
            }
            return base;
        }
    }

    return 0;
}

void mglDeleteLists(GLMContext ctx, GLuint list, GLsizei range)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCList *lists;

    if (range < 0)
        ERROR_RETURN(GL_INVALID_VALUE);

    if (!s->lists)
        return;

    lists = s->lists;

    for (GLsizei i = 0; i < range; i++)
    {
        GLuint id = list + i;
        if (id == 0 || id >= MGLC_LIST_MAX)
            continue;
        if (s->list_used[id])
            list_free(ctx, &lists[id]);
        s->list_used[id] = GL_FALSE;
    }
}

GLboolean mglIsList(GLMContext ctx, GLuint list)
{    MGLC_COMPAT_GUARD_RET0(ctx);

    if (!list || list >= MGLC_LIST_MAX)
        return GL_FALSE;

    return CS()->list_used ? CS()->list_used[list] : GL_FALSE;
}

void mglListBase(GLMContext ctx, GLuint base)
{    MGLC_COMPAT_GUARD(ctx);

    if (record(ctx, MGLC_OP_LIST_BASE))
        CS()->list_base = base;
}

void mglCallList(GLMContext ctx, GLuint list)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    MGLCList *l = find_list(ctx, list);

    if (!l)
        return;     /* calling an undefined list is a no-op */

    if (s->recording && !s->replaying)
    {
        /* record a call: nested lists replay the callee */
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_CALL_LIST);
        if (o) o->i0 = (int32_t)list;
        if (!s->record_exec)
            return;
    }

    if (s->call_depth >= MGLC_LIST_NESTING)
        return;

    s->call_depth++;
    s->replaying = GL_TRUE;

    for (GLuint i = 0; i < l->nops; i++)
        mgl_compat_replay_op(ctx, &l->ops[i]);

    s->replaying = GL_FALSE;
    s->call_depth--;
}

void mglCallLists(GLMContext ctx, GLsizei n, GLenum type, const void *lists)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();
    GLuint base = s->list_base;

    if (n < 0 || !lists)
        ERROR_RETURN(GL_INVALID_VALUE);

    for (GLsizei i = 0; i < n; i++)
    {
        GLuint name = 0;

        switch (type)
        {
            case GL_BYTE: name = (GLuint)(base + ((const GLbyte *)lists)[i]); break;
            case GL_UNSIGNED_BYTE: name = base + ((const GLubyte *)lists)[i]; break;
            case GL_SHORT: name = (GLuint)(base + ((const GLshort *)lists)[i]); break;
            case GL_UNSIGNED_SHORT: name = base + ((const GLushort *)lists)[i]; break;
            case GL_INT: name = base + (GLuint)((const GLint *)lists)[i]; break;
            case GL_UNSIGNED_INT: name = base + ((const GLuint *)lists)[i]; break;
            case GL_FLOAT: name = base + (GLuint)((const GLfloat *)lists)[i]; break;
            case GL_2_BYTES: { const GLubyte *b = (const GLubyte *)lists + i * 2; name = base + (GLuint)((b[0] << 8) | b[1]); } break;
            case GL_3_BYTES: { const GLubyte *b = (const GLubyte *)lists + i * 3; name = base + (GLuint)((b[0] << 16) | (b[1] << 8) | b[2]); } break;
            case GL_4_BYTES: { const GLubyte *b = (const GLubyte *)lists + i * 4; name = base + (GLuint)(((GLuint)b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]); } break;
            default:
                ERROR_RETURN(GL_INVALID_ENUM);
                return;
        }

        mglCallList(ctx, name);
    }
}

#pragma mark replay

/* defined in compat_vb.c: they dispatch straight to the immediate path */
void mgl_compat_replay_set_pointer(GLMContext ctx, int index, GLint size, GLenum type,
                                   GLboolean normalized, GLsizei stride, const void *ptr);

void mgl_compat_replay_op(GLMContext ctx, MGLCListOp *o)
{
    struct MGLExtState_compat *s = CS();

    switch (o->op)
    {
        case MGLC_OP_BEGIN: mglBegin(ctx, (GLenum)o->i0); break;
        case MGLC_OP_END: mglEnd(ctx); break;
        case MGLC_OP_VERTEX4F:
        {
            float x = o->f[0], y = o->f[1], z = o->f[2], w = o->f[3];
            switch (o->a)
            {
                case 2: mglVertex2f(ctx, x, y); break;
                case 3: mglVertex3f(ctx, x, y, z); break;
                default: mglVertex4f(ctx, x, y, z, w); break;
            }
            break;
        }
        case MGLC_OP_COLOR4F: mglColor4f(ctx, o->f[0], o->f[1], o->f[2], o->f[3]); break;
        case MGLC_OP_SECONDARY3F: mglSecondaryColor3f(ctx, o->f[0], o->f[1], o->f[2]); break;
        case MGLC_OP_NORMAL3F: mglNormal3f(ctx, o->f[0], o->f[1], o->f[2]); break;
        case MGLC_OP_INDEXF: mglIndexf(ctx, o->f[0]); break;
        case MGLC_OP_FOGCOORDF: mglFogCoordf(ctx, o->f[0]); break;
        case MGLC_OP_EDGEFLAG: mglEdgeFlag(ctx, o->a); break;
        case MGLC_OP_TEXCOORD4F:
        {
            if (o->a == 0)
            {
                switch (o->b)
                {
                    case 1: mglTexCoord1f(ctx, o->f[0]); break;
                    case 2: mglTexCoord2f(ctx, o->f[0], o->f[1]); break;
                    case 3: mglTexCoord3f(ctx, o->f[0], o->f[1], o->f[2]); break;
                    default: mglTexCoord4f(ctx, o->f[0], o->f[1], o->f[2], o->f[3]); break;
                }
            }
            else
            {
                GLenum t = GL_TEXTURE0 + o->a;
                switch (o->b)
                {
                    case 1: mglMultiTexCoord1f(ctx, t, o->f[0]); break;
                    case 2: mglMultiTexCoord2f(ctx, t, o->f[0], o->f[1]); break;
                    case 3: mglMultiTexCoord3f(ctx, t, o->f[0], o->f[1], o->f[2]); break;
                    default: mglMultiTexCoord4f(ctx, t, o->f[0], o->f[1], o->f[2], o->f[3]); break;
                }
            }
            break;
        }
        case MGLC_OP_ARRAY_ELEMENT: mglArrayElement(ctx, o->i0); break;
        case MGLC_OP_EVALCOORD1F: mglEvalCoord1f(ctx, o->f[0]); break;
        case MGLC_OP_EVALCOORD2F: mglEvalCoord2f(ctx, o->f[0], o->f[1]); break;
        case MGLC_OP_EVALPOINT1: mglEvalPoint1(ctx, o->i0); break;
        case MGLC_OP_EVALPOINT2: mglEvalPoint2(ctx, o->i0, o->i1); break;
        case MGLC_OP_EVALMESH:
        {
            static const GLenum mesh_mode[3] = { GL_POINT, GL_LINE, GL_FILL };

            if (o->b == 2)
                mglEvalMesh2(ctx, mesh_mode[o->a % 3], o->i0, o->i1, (GLint)o->f[2], (GLint)o->f[3]);
            else
                mglEvalMesh1(ctx, mesh_mode[o->a % 3], o->i0, o->i1);
            break;
        }
        case MGLC_OP_MAP1:
        {
            int comps = (int)o->f[2];

            mglMap1f(ctx, (GLenum)o->i0, o->f[0], o->f[1], comps, o->i1,
                     (const GLfloat *)o->blob);
            break;
        }
        case MGLC_OP_MAP2:
        {
            int comps = (int)o->f[4];
            int vorder = o->a;

            mglMap2f(ctx, (GLenum)o->i0, o->f[0], o->f[1], comps * vorder, o->i1,
                     o->f[2], o->f[3], comps, vorder, (const GLfloat *)o->blob);
            break;
        }
        case MGLC_OP_MAPGRID1: mglMapGrid1f(ctx, o->i0, o->f[0], o->f[1]); break;
        case MGLC_OP_MAPGRID2: mglMapGrid2f(ctx, o->i0, o->f[0], o->f[1], o->i1, o->f[2], o->f[3]); break;
        case MGLC_OP_RECTF:
        {
            float x1 = o->f[0], y1 = o->f[1], x2 = o->f[2], y2 = o->f[3];
            mglRectf(ctx, x1, y1, x2, y2);
            break;
        }
        case MGLC_OP_MATRIX_MODE: mglMatrixMode(ctx, (GLenum)o->i0); break;
        case MGLC_OP_LOAD_IDENTITY: mglLoadIdentity(ctx); break;
        case MGLC_OP_LOAD_MATRIXF: mglLoadMatrixf(ctx, o->f); break;
        case MGLC_OP_MULT_MATRIXF: mglMultMatrixf(ctx, o->f); break;
        case MGLC_OP_TRANSLATEF: mglTranslatef(ctx, o->f[0], o->f[1], o->f[2]); break;
        case MGLC_OP_SCALEF: mglScalef(ctx, o->f[0], o->f[1], o->f[2]); break;
        case MGLC_OP_ROTATEF: mglRotatef(ctx, o->f[0], o->f[1], o->f[2], o->f[3]); break;
        case MGLC_OP_ORTHO: mglOrtho(ctx, o->f[0], o->f[1], o->f[2], o->f[3], o->f[4], o->f[5]); break;
        case MGLC_OP_FRUSTUM: mglFrustum(ctx, o->f[0], o->f[1], o->f[2], o->f[3], o->f[4], o->f[5]); break;
        case MGLC_OP_PUSH_MATRIX: mglPushMatrix(ctx); break;
        case MGLC_OP_POP_MATRIX: mglPopMatrix(ctx); break;
        case MGLC_OP_ENABLE: mglEnable(ctx, (GLenum)o->i0); break;
        case MGLC_OP_DISABLE: mglDisable(ctx, (GLenum)o->i0); break;
        case MGLC_OP_BIND_TEXTURE: mglBindTexture(ctx, (GLenum)o->i0, (GLuint)o->i1); break;
        case MGLC_OP_ACTIVE_TEXTURE: mglActiveTexture(ctx, (GLenum)o->i0); break;
        case MGLC_OP_SHADE_MODEL: mglShadeModel(ctx, (GLenum)o->i0); break;
        case MGLC_OP_POINT_SIZE: mglPointSize(ctx, o->f[0]); break;
        case MGLC_OP_LINE_WIDTH: mglLineWidth(ctx, o->f[0]); break;
        case MGLC_OP_CULL_FACE: mglCullFace(ctx, (GLenum)o->i0); break;
        case MGLC_OP_FRONT_FACE: mglFrontFace(ctx, (GLenum)o->i0); break;
        case MGLC_OP_DEPTH_FUNC: mglDepthFunc(ctx, (GLenum)o->i0); break;
        case MGLC_OP_DEPTH_MASK: mglDepthMask(ctx, o->a); break;
        case MGLC_OP_BLEND_FUNC: mglBlendFunc(ctx, (GLenum)o->i0, (GLenum)o->i1); break;
        case MGLC_OP_LOGIC_OP: mglLogicOp(ctx, (GLenum)o->i0); break;
        case MGLC_OP_STENCIL_FUNC: mglStencilFunc(ctx, (GLenum)o->i0, (GLint)o->i1, (GLuint)o->f[0]); break;
        case MGLC_OP_STENCIL_OP: mglStencilOp(ctx, (GLenum)o->i0, (GLenum)o->i1, (GLenum)o->a); break;
        case MGLC_OP_STENCIL_MASK: mglStencilMask(ctx, (GLuint)o->i0); break;
        case MGLC_OP_DEPTH_RANGE: mglDepthRange(ctx, o->f[0], o->f[1]); break;
        case MGLC_OP_VIEWPORT: mglViewport(ctx, o->i0, o->i1, (GLsizei)o->a, (GLsizei)o->b); break;
        case MGLC_OP_SCISSOR: mglScissor(ctx, o->i0, o->i1, (GLsizei)o->a, (GLsizei)o->b); break;
        case MGLC_OP_CLEAR_COLOR: mglClearColor(ctx, o->f[0], o->f[1], o->f[2], o->f[3]); break;
        case MGLC_OP_CLEAR_DEPTH: mglClearDepth(ctx, o->f[0]); break;
        case MGLC_OP_CLEAR_STENCIL: mglClearStencil(ctx, (GLint)o->i0); break;
        case MGLC_OP_CLEAR: mglClear(ctx, (GLbitfield)o->i0); break;
        case MGLC_OP_COLOR_MASK: mglColorMask(ctx, o->a, o->b, o->c, o->d); break;
        case MGLC_OP_LIGHTFV:
        {
            /* i0 = light<<16|pname, f[] = up to 4 params */
            mglLightfv(ctx, (GLenum)(o->i0 >> 16), (GLenum)(o->i0 & 0xFFFF), o->f);
            break;
        }
        case MGLC_OP_LIGHTMODELFV: mglLightModelfv(ctx, (GLenum)o->i0, o->f); break;
        case MGLC_OP_MATERIALFV: mglMaterialfv(ctx, (GLenum)(o->i0 >> 16), (GLenum)(o->i0 & 0xFFFF), o->f); break;
        case MGLC_OP_COLORMATERIAL: mglColorMaterial(ctx, (GLenum)o->i0, (GLenum)o->i1); break;
        case MGLC_OP_TEXENVFV:
            mglTexEnvfv(ctx, (GLenum)o->i0, (GLenum)o->i1, o->f);
            break;
        case MGLC_OP_TEXGENFV: mglTexGenfv(ctx, (GLenum)o->i0, (GLenum)o->i1, o->f); break;
        case MGLC_OP_FOGFV: mglFogfv(ctx, (GLenum)o->i0, o->f); break;
        case MGLC_OP_CLIPPING_PLANE:
        {
            GLdouble e[4] = { o->f[0], o->f[1], o->f[2], o->f[3] };
            mglClipPlane(ctx, (GLenum)o->i0, e);
            break;
        }
        case MGLC_OP_LINE_STIPPLE: mglLineStipple(ctx, o->i0, (GLushort)o->i1); break;
        case MGLC_OP_POLYGON_STIPPLE:
        {
            if (o->blob && o->blob_size >= 128)
                mglPolygonStipple(ctx, (const GLubyte *)o->blob);
            break;
        }
        case MGLC_OP_RASTERPOS4F: mglRasterPos4f(ctx, o->f[0], o->f[1], o->f[2], o->f[3]); break;
        case MGLC_OP_WINDOWPOS3F: mglWindowPos3f(ctx, o->f[0], o->f[1], o->f[2]); break;
        case MGLC_OP_PIXEL_ZOOM: mglPixelZoom(ctx, o->f[0], o->f[1]); break;
        case MGLC_OP_DRAW_PIXELS:
        {
            if (o->blob)
                mglDrawPixels(ctx, o->i0, o->i1, (GLenum)o->a | 0, (GLenum)o->b, o->blob);
            break;
        }
        case MGLC_OP_BITMAP:
        {
            if (o->blob)
                mglBitmap(ctx, o->i0, o->i1, o->f[0], o->f[1], o->f[2], o->f[3], (const GLubyte *)o->blob);
            break;
        }
        case MGLC_OP_PUSH_ATTRIB: mglPushAttrib(ctx, (GLbitfield)o->i0); break;
        case MGLC_OP_POP_ATTRIB: mglPopAttrib(ctx); break;
        case MGLC_OP_PUSH_NAME: mglPushName(ctx, (GLuint)o->i0); break;
        case MGLC_OP_POP_NAME: mglPopName(ctx); break;
        case MGLC_OP_LOAD_NAME: mglLoadName(ctx, (GLuint)o->i0); break;
        case MGLC_OP_INIT_NAMES: mglInitNames(ctx); break;
        case MGLC_OP_LIST_BASE: mglListBase(ctx, (GLuint)o->i0); break;
        case MGLC_OP_CALL_LIST: mglCallList(ctx, (GLuint)o->i0); break;
        case MGLC_OP_ENABLE_CLIENT_STATE:
        {
            uint32_t t;
            memcpy(&t, &o->f[0], 4);
            mglEnableClientState(ctx, (GLenum)t);
            break;
        }
        case MGLC_OP_DISABLE_CLIENT_STATE:
        {
            uint32_t t;
            memcpy(&t, &o->f[0], 4);
            mglDisableClientState(ctx, (GLenum)t);
            break;
        }
        case MGLC_OP_CLIENT_ACTIVE_TEXTURE:
        {
            uint32_t t;
            memcpy(&t, &o->f[0], 4);
            mglClientActiveTexture(ctx, (GLenum)t);
            break;
        }
        default:
            /* MGLC_OP_VERTEX_POINTER + index */
            if (o->op >= MGLC_OP_VERTEX_POINTER && o->op < MGLC_OP_VERTEX_POINTER + MAX_ATTRIBS)
            {
                uint32_t type;
                memcpy(&type, &o->f[1], 4);
                const void *ptr;
                int64_t p;
                memcpy(&p, &o->f[12], 8);
                ptr = (const void *)(uintptr_t)p;
                mgl_compat_replay_set_pointer(ctx, o->a, o->i0, (GLenum)type, o->b, o->i1, ptr);
            }
            break;
    }

    (void)s;
}

#pragma mark attribute stacks

/* snapshot of the fixed-function + GL value state (deep-copied scalars) */
typedef struct MGLCAttribSnapshotData_t {
    uint32_t mask;
    GLMCaps caps;
    GLMParams var;
    TextureUnit texture_units[TEXTURE_UNITS];
    struct MGLExtState_compat compat;
    GLubyte *pixel_map_copy;
    size_t pixel_map_bytes[10];
    /* evaluator control points are heap arrays: deep-copy them, because a
     * glMap1/glMap2 between push and pop frees the snapshot pointers (and a
     * struct restore would resurrect them as dangling pointers) */
    GLubyte *eval_map_copy;
    size_t eval_map_bytes[18];      /* map1[0..8], map2[0..8] */
} MGLCAttribSnapshotData;

/* number of control-point floats in evaluator map slot i (map1 / map2) */
static int eval_points_count(GLMContext ctx, int two_d, int i)
{
    struct MGLExtState_compat *s = CS();
    int comps = two_d ? mgl_compat_eval_comps(s->map2[i].target)
                      : mgl_compat_eval_comps(s->map1[i].target);

    if (two_d)
        return s->map2[i].uorder * s->map2[i].vorder * comps;
    return s->map1[i].order * comps;
}

void mglPushAttrib(GLMContext ctx, GLbitfield mask)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (!record(ctx, MGLC_OP_PUSH_ATTRIB))
        return;

    if (s->attrib_depth >= MGLC_ATTRIB_STACK)
    {
        ERROR_RETURN(GL_STACK_OVERFLOW);
        return;
    }

    MGLCAttribSnapshotData *snap = (MGLCAttribSnapshotData *)calloc(1, sizeof *snap);

    if (!snap)
        return;

    snap->mask = mask;
    snap->caps = ctx->state.caps;
    snap->var = ctx->state.var;
    memcpy(snap->texture_units, ctx->state.texture_units, sizeof snap->texture_units);
    snap->compat = *s;

    /* pixel maps are small tables: copy their contents */
    {
        size_t total = 0;
        for (int i = 0; i < 10; i++)
        {
            snap->pixel_map_bytes[i] = (size_t)s->pixel_map_size[i] * sizeof(GLfloat);
            total += snap->pixel_map_bytes[i];
        }
        snap->pixel_map_copy = (GLubyte *)malloc(total ? total : 1);
        if (snap->pixel_map_copy)
        {
            size_t off = 0;
            for (int i = 0; i < 10; i++)
            {
                memcpy(snap->pixel_map_copy + off, s->pixel_map[i], snap->pixel_map_bytes[i]);
                off += snap->pixel_map_bytes[i];
            }
        }
    }

    /* evaluator maps: deep-copy the control points (9 x map1 + 9 x map2) */
    {
        size_t total = 0;
        for (int i = 0; i < 9; i++)
        {
            int k1 = (int)eval_points_count(ctx, 0, i), k2 = (int)eval_points_count(ctx, 1, i);

            snap->eval_map_bytes[i] = (size_t)(k1 > 0 ? k1 : 0) * sizeof(GLfloat);
            snap->eval_map_bytes[9 + i] = (size_t)(k2 > 0 ? k2 : 0) * sizeof(GLfloat);
            total += snap->eval_map_bytes[i] + snap->eval_map_bytes[9 + i];
        }
        snap->eval_map_copy = (GLubyte *)malloc(total ? total : 1);
        if (snap->eval_map_copy)
        {
            size_t off = 0;
            for (int i = 0; i < 9; i++)
            {
                if (snap->eval_map_bytes[i])
                {
                    memcpy(snap->eval_map_copy + off, s->map1[i].fpoints, snap->eval_map_bytes[i]);
                    off += snap->eval_map_bytes[i];
                }
                if (snap->eval_map_bytes[9 + i])
                {
                    memcpy(snap->eval_map_copy + off, s->map2[i].fpoints, snap->eval_map_bytes[9 + i]);
                    off += snap->eval_map_bytes[9 + i];
                }
            }
        }
    }

    s->attrib_stack[s->attrib_depth].mask = mask;
    s->attrib_stack[s->attrib_depth].deep = (struct MGLExtState_compat *)snap;
    s->attrib_depth++;
}

void mglPopAttrib(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (!record(ctx, MGLC_OP_POP_ATTRIB))
        return;

    if (s->attrib_depth == 0)
    {
        ERROR_RETURN(GL_STACK_UNDERFLOW);
        return;
    }

    MGLCAttribSnapshot *slot = &s->attrib_stack[--s->attrib_depth];
    MGLCAttribSnapshotData *snap = (MGLCAttribSnapshotData *)slot->deep;

    if (!snap)
        return;

    ctx->state.caps = snap->caps;
    ctx->state.var = snap->var;
    memcpy(ctx->state.texture_units, snap->texture_units, sizeof snap->texture_units);

    /* scalars come back; live objects / recording state stay */
    {
        struct MGLExtState_compat keep = *s;

        *s = snap->compat;

        /* the struct restore brought back the map control-point pointers
         * that were live at PUSH time: they may have been freed by a
         * glMap1/glMap2 in between, so free the CURRENT live arrays (captured
         * in keep) and install deep copies from the snapshot below — never
         * freeing the restored values themselves */
        for (int i = 0; i < 9; i++)
        {
            free(keep.map1[i].fpoints);
            free(keep.map2[i].fpoints);
        }

        s->lists = keep.lists;
        s->list_used = keep.list_used;
        s->imm = keep.imm;
        s->ffp_cache = keep.ffp_cache;
        s->vao0 = keep.vao0;
        s->imm_vao = keep.imm_vao;
        s->index_scratch = keep.index_scratch;
        s->scratch_ibo = keep.scratch_ibo;
        for (int i = 0; i < MAX_ATTRIBS; i++)
            s->scratch_vbo[i] = keep.scratch_vbo[i];
        /* compat perf session 37: the streaming rings / caches describe the
         * LIVE scratch buffers above — a popped snapshot would replay stale
         * offsets (overwriting in-flight draws) and stale Buffer* identities */
        memcpy(s->scratch_ring, keep.scratch_ring, sizeof keep.scratch_ring);
        memcpy(s->scratch_obj, keep.scratch_obj, sizeof keep.scratch_obj);
        s->scratch_used = keep.scratch_used;
        s->attr_shadow = keep.attr_shadow;
        s->attr_shadow_set = keep.attr_shadow_set;
        s->uni_sync_cache = keep.uni_sync_cache;
        s->ffp_sync_cache = keep.ffp_sync_cache;
        s->stipple_tex = keep.stipple_tex;
        s->stipple_serial = keep.stipple_serial;
        s->blit_program = keep.blit_program;
        s->blit_texture = keep.blit_texture;
        s->accum_tex = keep.accum_tex;
        s->accum_valid = keep.accum_valid;
        s->accum_w = keep.accum_w;
        s->accum_h = keep.accum_h;
        s->rmode = keep.rmode;
        s->recording = keep.recording;
        s->replaying = keep.replaying;
        s->record_exec = keep.record_exec;
        s->record_list = keep.record_list;
        s->record_mode = keep.record_mode;
        s->call_depth = keep.call_depth;
        s->attrib_depth = keep.attrib_depth;
        s->client_attrib_depth = keep.client_attrib_depth;
        memcpy(s->attrib_stack, keep.attrib_stack, sizeof keep.attrib_stack);
        memcpy(s->client_attrib_stack, keep.client_attrib_stack, sizeof keep.client_attrib_stack);
        memcpy(s->builtin_prog, keep.builtin_prog, sizeof keep.builtin_prog);
        s->builtin_prog_count = keep.builtin_prog_count;
        s->ffp_program = keep.ffp_program;

        /* pixel map *contents* restored into the live tables */
        if (snap->pixel_map_copy)
        {
            size_t off = 0;
            for (int i = 0; i < 10; i++)
            {
                if (s->pixel_map[i])
                    memcpy(s->pixel_map[i], snap->pixel_map_copy + off, snap->pixel_map_bytes[i]);
                off += snap->pixel_map_bytes[i];
            }
        }

        /* evaluator control points: fresh copies of the snapshot contents */
        if (snap->eval_map_copy)
        {
            size_t off = 0;

            for (int i = 0; i < 9; i++)
            {
                s->map1[i].fpoints = (GLfloat *)malloc(snap->eval_map_bytes[i] ? snap->eval_map_bytes[i] : 1);
                if (s->map1[i].fpoints && snap->eval_map_bytes[i])
                    memcpy(s->map1[i].fpoints, snap->eval_map_copy + off, snap->eval_map_bytes[i]);
                off += snap->eval_map_bytes[i];

                s->map2[i].fpoints = (GLfloat *)malloc(snap->eval_map_bytes[9 + i] ? snap->eval_map_bytes[9 + i] : 1);
                if (s->map2[i].fpoints && snap->eval_map_bytes[9 + i])
                    memcpy(s->map2[i].fpoints, snap->eval_map_copy + off, snap->eval_map_bytes[9 + i]);
                off += snap->eval_map_bytes[9 + i];
            }
        }
        else
        {
            /* no snapshot copy (allocation failure at push): NULL them out so
             * no dangling pointer is left behind */
            for (int i = 0; i < 9; i++)
            {
                s->map1[i].fpoints = NULL;
                s->map2[i].fpoints = NULL;
            }
        }
    }

    /* recompute the float matrix caches from the restored stacks */
    {
        GLdouble m[16];

        mgl_compat_get_matrix(ctx, GL_MODELVIEW, m);
        for (int i = 0; i < 16; i++)
            s->modelview_f[i] = (GLfloat)m[i];
        mgl_compat_get_matrix(ctx, GL_PROJECTION, m);
        for (int i = 0; i < 16; i++)
            s->projection_f[i] = (GLfloat)m[i];

        for (int u = 0; u < MGLC_MAX_COORDS; u++)
        {
            for (int i = 0; i < 16; i++)
            {
                GLdouble *tm = s->texture[u].stack[s->texture[u].depth].m;
                s->texture_f[u][i] = (GLfloat)tm[i];
            }
        }
    }

    free(snap->pixel_map_copy);
    free(snap->eval_map_copy);
    free(snap);
    slot->deep = NULL;

    STATE(dirty_bits) |= DIRTY_ALL;
    mgl_compat_dirty(ctx);
}

#pragma mark client attribute stack

void mglPushClientAttrib(GLMContext ctx, GLbitfield mask)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->client_attrib_depth >= MGLC_CLIENT_STACK)
    {
        ERROR_RETURN(GL_STACK_OVERFLOW);
        return;
    }

    struct MGLCClientAttrib_t *slot = &s->client_attrib_stack[s->client_attrib_depth++];

    slot->mask = mask;
    memcpy(slot->arrays, s->arrays, sizeof slot->arrays);
    slot->client_active_texture = s->client_active_texture;
    slot->edge_flag_enabled = s->edge_flag_enabled;
    slot->array_buffer = STATE(buffers[_ARRAY_BUFFER]);
    slot->element_buffer = STATE(buffers[_ELEMENT_ARRAY_BUFFER]);
}

void mglPopClientAttrib(GLMContext ctx)
{    MGLC_COMPAT_GUARD(ctx);

    struct MGLExtState_compat *s = CS();

    if (s->client_attrib_depth == 0)
    {
        ERROR_RETURN(GL_STACK_UNDERFLOW);
        return;
    }

    struct MGLCClientAttrib_t *slot = &s->client_attrib_stack[--s->client_attrib_depth];

    memcpy(s->arrays, slot->arrays, sizeof s->arrays);
    s->client_active_texture = slot->client_active_texture;
    s->edge_flag_enabled = slot->edge_flag_enabled;

    mglBindBuffer(ctx, GL_ARRAY_BUFFER, slot->array_buffer ? slot->array_buffer->name : 0);
    mglBindBuffer(ctx, GL_ELEMENT_ARRAY_BUFFER, slot->element_buffer ? slot->element_buffer->name : 0);
}
