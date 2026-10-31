/*
 * compat_dispatch.h — per-context dispatch overrides of the compat domain.
 *
 * mgl_compat_init_dispatch() is called from mgl_ext_init_compat() for
 * compatibility-profile contexts only: the fixed-function entry points become
 * live and a handful of core entry points get compat behaviour (VAO 0,
 * legacy caps, compat GLSL translation, client arrays, quads).
 * A core context keeps the core implementation / stubs untouched.
 */
#ifndef COMPAT_DISPATCH_H
#define COMPAT_DISPATCH_H

/* ---- compat behaviour of core entry points (record into display lists,
 *      then run the compat path) ---- */

/* compat GLSL: translate legacy shaders to core GLSL before glslang */
static void compat_shader_source2(GLMContext ctx, GLuint shader, GLsizei count,
                                  const GLchar *const *string, const GLint *length)
{
    extern Shader *findShader(GLMContext ctx, GLuint shader);
    GLenum type = GL_NONE;

    if (findShader(ctx, shader))
        type = findShader(ctx, shader)->type;

    if (type == GL_VERTEX_SHADER || type == GL_FRAGMENT_SHADER ||
        type == GL_GEOMETRY_SHADER || type == GL_TESS_CONTROL_SHADER ||
        type == GL_TESS_EVALUATION_SHADER)
    {
        char *translated = mgl_compat_translate_shader(ctx, type, count, string, length);

        if (translated)
        {
            mglShaderSource(ctx, shader, 1, (const GLchar *const *)&translated, NULL);
            free(translated);
            return;
        }
    }

    mglShaderSource(ctx, shader, count, string, length);
}

static void compat_draw_arrays(GLMContext ctx, GLenum mode, GLint first, GLsizei count)
{
    mgl_compat_draw(ctx, mode, first, count);
}

static void compat_draw_elements(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices)
{
    mgl_compat_draw_elements(ctx, mode, count, type, indices, 0);
}

static void compat_draw_elements_base_vertex(GLMContext ctx, GLenum mode, GLsizei count, GLenum type,
                                             const void *indices, GLint basevertex)
{
    mgl_compat_draw_elements(ctx, mode, count, type, indices, basevertex);
}

static void compat_draw_arrays_instanced(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount)
{
    mgl_compat_draw(ctx, mode, first, count * (instancecount ? 1 : 1));
    for (GLsizei i = 1; i < instancecount; i++)
        mgl_compat_draw(ctx, mode, first, count);
}

static void compat_draw_elements_instanced(GLMContext ctx, GLenum mode, GLsizei count, GLenum type,
                                           const void *indices, GLsizei instancecount)
{
    for (GLsizei i = 0; i < instancecount; i++)
        mgl_compat_draw_elements(ctx, mode, count, type, indices, 0);
}

static void compat_polygon_mode(GLMContext ctx, GLenum face, GLenum mode)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (face != GL_FRONT && face != GL_BACK && face != GL_FRONT_AND_BACK)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (mode != GL_POINT && mode != GL_LINE && mode != GL_FILL)
        ERROR_RETURN(GL_INVALID_ENUM);

    if (s->recording && !s->replaying)
    {
        /* polygon mode is recorded as part of the list state */
    }

    if (face == GL_FRONT || face == GL_FRONT_AND_BACK)
        s->polygon_mode_front = mode;
    if (face == GL_BACK || face == GL_FRONT_AND_BACK)
        s->polygon_mode_back = mode;

    /* Metal rasterises triangles filled; point/line polygons are converted at
     * draw time (compat_vb.c), so keep the raster state at fill */
    mglPolygonMode(ctx, GL_FRONT_AND_BACK, GL_FILL);
    mgl_compat_dirty(ctx);
}

static void compat_point_size(GLMContext ctx, GLfloat size)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_POINT_SIZE);
        if (o)
        {
            o->f[0] = size;
            if (!s->record_exec)
                return;
        }
    }

    if (size <= 0.0f)
        ERROR_RETURN(GL_INVALID_VALUE);

    s->point_size = size;
    mglPointSize(ctx, size);
    mgl_compat_dirty(ctx);
}

static void compat_get_pointerv(GLMContext ctx, GLenum pname, void **params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;
    int index = -1;

    if (!params)
        ERROR_RETURN(GL_INVALID_VALUE);

    switch (pname)
    {
        case GL_VERTEX_ARRAY_POINTER: index = MGLC_ATTR_VERTEX; break;
        case GL_NORMAL_ARRAY_POINTER: index = MGLC_ATTR_NORMAL; break;
        case GL_COLOR_ARRAY_POINTER: index = MGLC_ATTR_COLOR; break;
        case GL_SECONDARY_COLOR_ARRAY_POINTER: index = MGLC_ATTR_SECONDARY; break;
        case GL_FOG_COORD_ARRAY_POINTER: index = MGLC_ATTR_FOG; break;
        case GL_INDEX_ARRAY_POINTER: index = MGLC_ATTR_INDEX; break;
        case GL_EDGE_FLAG_ARRAY_POINTER: index = MGLC_ATTR_EDGEFLAG; break;
        case GL_TEXTURE_COORD_ARRAY_POINTER: index = MGLC_ATTR_TEXCOORD(s->client_active_texture % MGLC_MAX_COORDS); break;
        case GL_FEEDBACK_BUFFER_POINTER:
            *params = (void *)s->rmode.feedback_buf;
            return;
        case GL_SELECTION_BUFFER_POINTER:
            *params = (void *)s->rmode.select_buf;
            return;
        default:
            mglGetPointerv(ctx, pname, params);
            return;
    }

    *params = (void *)s->arrays[index].pointer;
}

/* compat perf session 37 — program-epoch guard for the per-program uniform
 * location caches: any (re)link or delete of any program invalidates every
 * cached table wholesale (locations are only stable for a linked, live
 * program; heap-address reuse of Program objects made pointer-identity checks
 * insufficient — caught as intermittent black draws in test_compat_glsl). */
static void compat_link_program(GLMContext ctx, GLuint program)
{
    ctx->state.ext_compat->prog_epoch++;
    mglLinkProgram(ctx, program);
}

static void compat_delete_program(GLMContext ctx, GLuint program)
{
    ctx->state.ext_compat->prog_epoch++;
    mglDeleteProgram(ctx, program);
}

/* wrappers that record into a display list before executing */
#define REC0V(OP, CALL) \
    static void rec_##OP(GLMContext ctx) { \
        struct MGLExtState_compat *st = ctx->state.ext_compat; \
        if (st->recording && !st->replaying) { \
            mgl_compat_list_op(ctx, MGLC_OP_##OP); \
            if (!st->record_exec) return; \
        } \
        CALL; \
    }

#define REC1V(OP, T0, A0, CALL) \
    static void rec_##OP(GLMContext ctx, T0 A0) { \
        struct MGLExtState_compat *st = ctx->state.ext_compat; \
        if (st->recording && !st->replaying) { \
            MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_##OP); \
            if (o) o->i0 = (int32_t)A0; \
            if (!st->record_exec) return; \
        } \
        CALL; \
    }

#define REC2V(OP, T0, A0, T1, A1, CALL) \
    static void rec_##OP(GLMContext ctx, T0 A0, T1 A1) { \
        struct MGLExtState_compat *st = ctx->state.ext_compat; \
        if (st->recording && !st->replaying) { \
            MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_##OP); \
            if (o) { o->i0 = (int32_t)A0; o->i1 = (int32_t)A1; } \
            if (!st->record_exec) return; \
        } \
        CALL; \
    }

#define REC2FV(OP, F0, F1, CALL) \
    static void rec_##OP(GLMContext ctx, GLfloat F0, GLfloat F1) { \
        struct MGLExtState_compat *st = ctx->state.ext_compat; \
        if (st->recording && !st->replaying) { \
            MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_##OP); \
            if (o) { o->f[0] = F0; o->f[1] = F1; } \
            if (!st->record_exec) return; \
        } \
        CALL; \
    }

REC0V(PUSH_MATRIX, mglPushMatrix(ctx))
REC0V(POP_MATRIX, mglPopMatrix(ctx))
REC0V(LOAD_IDENTITY, mglLoadIdentity(ctx))
REC1V(MATRIX_MODE, GLenum, mode, mglMatrixMode(ctx, mode))
REC0V(FLUSH, mglFlush(ctx))
REC0V(FINISH, mglFinish(ctx))

static void rec_LOAD_MATRIXF(GLMContext ctx, const GLfloat *m)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_LOAD_MATRIXF);
        if (o)
        {
            memcpy(o->f, m, 16 * sizeof(GLfloat));
            if (!s->record_exec)
                return;
        }
    }

    mglLoadMatrixf(ctx, m);
}

static void rec_MULT_MATRIXF(GLMContext ctx, const GLfloat *m)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_MULT_MATRIXF);
        if (o)
        {
            memcpy(o->f, m, 16 * sizeof(GLfloat));
            if (!s->record_exec)
                return;
        }
    }

    mglMultMatrixf(ctx, m);
}

static void rec_TRANSLATEF(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    struct MGLExtState_compat *st = ctx->state.ext_compat;

    if (st->recording && !st->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_TRANSLATEF);
        if (o)
        {
            o->f[0] = x; o->f[1] = y; o->f[2] = z;
            if (!st->record_exec)
                return;
        }
    }

    mglTranslatef(ctx, x, y, z);
}

static void rec_SCALEF(GLMContext ctx, GLfloat x, GLfloat y, GLfloat z)
{
    struct MGLExtState_compat *st = ctx->state.ext_compat;

    if (st->recording && !st->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_SCALEF);
        if (o)
        {
            o->f[0] = x; o->f[1] = y; o->f[2] = z;
            if (!st->record_exec)
                return;
        }
    }

    mglScalef(ctx, x, y, z);
}

static void rec_ROTATEF(GLMContext ctx, GLfloat a, GLfloat x, GLfloat y, GLfloat z)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_ROTATEF);
        if (o)
        {
            o->f[0] = a; o->f[1] = x; o->f[2] = y; o->f[3] = z;
            if (!s->record_exec)
                return;
        }
    }

    mglRotatef(ctx, a, x, y, z);
}

REC1V(CULL_FACE, GLenum, mode, (mglCullFace(ctx, mode)))
REC1V(FRONT_FACE, GLenum, mode, (mglFrontFace(ctx, mode)))
REC1V(DEPTH_FUNC, GLenum, func, (mglDepthFunc(ctx, func)))
REC2V(BLEND_FUNC, GLenum, sfactor, GLenum, dfactor, (mglBlendFunc(ctx, sfactor, dfactor)))
REC1V(LOGIC_OP, GLenum, opcode, (mglLogicOp(ctx, opcode)))
REC1V(DEPTH_MASK, GLboolean, flag, (mglDepthMask(ctx, flag)))
REC1V(SHADE_MODEL, GLenum, mode, (mglShadeModel(ctx, mode)))
REC1V(LINE_WIDTH, GLfloat, width, (mglLineWidth(ctx, width)))
REC1V(STENCIL_MASK, GLuint, mask, (mglStencilMask(ctx, mask)))
REC1V(CLEAR_DEPTH, GLclampd, depth, (mglClearDepth(ctx, depth)))
REC1V(CLEAR_STENCIL, GLint, s, (mglClearStencil(ctx, s)))
REC1V(CLEAR, GLbitfield, mask, (mglClear(ctx, mask)))

static void rec_ENABLE(GLMContext ctx, GLenum cap)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_ENABLE);
        if (o)
        {
            o->i0 = (int32_t)cap;
            if (!s->record_exec)
                return;
        }
    }

    compat_enable(ctx, cap);
}

static void rec_DISABLE(GLMContext ctx, GLenum cap)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_DISABLE);
        if (o)
        {
            o->i0 = (int32_t)cap;
            if (!s->record_exec)
                return;
        }
    }

    compat_disable(ctx, cap);
}

static void rec_VIEWPORT(GLMContext ctx, GLint x, GLint y, GLsizei w, GLsizei h)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_VIEWPORT);
        if (o)
        {
            o->i0 = x; o->i1 = y; o->a = (uint8_t)w; o->b = (uint8_t)h;
            /* sizes beyond 255 are stored in f[] */
            o->f[2] = (float)w; o->f[3] = (float)h;
            if (!s->record_exec)
                return;
        }
    }

    mglViewport(ctx, x, y, w, h);
}

static void rec_SCISSOR(GLMContext ctx, GLint x, GLint y, GLsizei w, GLsizei h)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_SCISSOR);
        if (o)
        {
            o->i0 = x; o->i1 = y; o->f[2] = (float)w; o->f[3] = (float)h;
            if (!s->record_exec)
                return;
        }
    }

    mglScissor(ctx, x, y, w, h);
}

static void rec_CLEAR_COLOR(GLMContext ctx, GLfloat r, GLfloat g, GLfloat b, GLfloat a)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_CLEAR_COLOR);
        if (o)
        {
            o->f[0] = r; o->f[1] = g; o->f[2] = b; o->f[3] = a;
            if (!s->record_exec)
                return;
        }
    }

    mglClearColor(ctx, r, g, b, a);
}

static void rec_DEPTH_RANGE(GLMContext ctx, GLclampd n, GLclampd f)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_DEPTH_RANGE);
        if (o)
        {
            o->f[0] = (float)n; o->f[1] = (float)f;
            if (!s->record_exec)
                return;
        }
    }

    mglDepthRange(ctx, n, f);
}

static void rec_COLOR_MASK(GLMContext ctx, GLboolean r, GLboolean g, GLboolean b, GLboolean a)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_COLOR_MASK);
        if (o)
        {
            o->a = r ? 1 : 0; o->b = g ? 1 : 0; o->c = b ? 1 : 0; o->d = a ? 1 : 0;
            if (!s->record_exec)
                return;
        }
    }

    mglColorMask(ctx, r, g, b, a);
}

static void rec_STENCIL_FUNC(GLMContext ctx, GLenum func, GLint ref, GLuint mask)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_STENCIL_FUNC);
        if (o)
        {
            o->i0 = (int32_t)func; o->i1 = ref; o->f[0] = (float)mask;
            if (!s->record_exec)
                return;
        }
    }

    mglStencilFunc(ctx, func, ref, mask);
}

static void rec_STENCIL_OP(GLMContext ctx, GLenum fail, GLenum zfail, GLenum zpass)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_STENCIL_OP);
        if (o)
        {
            o->i0 = (int32_t)fail; o->i1 = (int32_t)zfail; o->a = (uint8_t)zpass;
            if (!s->record_exec)
                return;
        }
    }

    mglStencilOp(ctx, fail, zfail, zpass);
}

static void rec_BIND_TEXTURE(GLMContext ctx, GLenum target, GLuint texture)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_BIND_TEXTURE);
        if (o)
        {
            o->i0 = (int32_t)target; o->i1 = (int32_t)texture;
            if (!s->record_exec)
                return;
        }
    }

    mglBindTexture(ctx, target, texture);
    mgl_compat_dirty(ctx);
}

static void rec_ACTIVE_TEXTURE(GLMContext ctx, GLenum texture)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_ACTIVE_TEXTURE);
        if (o)
        {
            o->i0 = (int32_t)texture;
            if (!s->record_exec)
                return;
        }
    }

    mglActiveTexture(ctx, texture);
}

static void rec_LIGHTFV(GLMContext ctx, GLenum light, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_LIGHTFV);
        if (o)
        {
            o->i0 = ((int32_t)light << 16) | ((int32_t)pname & 0xFFFF);
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglLightfv(ctx, light, pname, params);
}

static void rec_LIGHTMODELFV(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_LIGHTMODELFV);
        if (o)
        {
            o->i0 = (int32_t)pname;
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglLightModelfv(ctx, pname, params);
}

static void rec_MATERIALFV(GLMContext ctx, GLenum face, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_MATERIALFV);
        if (o)
        {
            o->i0 = ((int32_t)face << 16) | ((int32_t)pname & 0xFFFF);
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglMaterialfv(ctx, face, pname, params);
}

static void rec_COLORMATERIAL(GLMContext ctx, GLenum face, GLenum mode)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_COLORMATERIAL);
        if (o)
        {
            o->i0 = (int32_t)face; o->i1 = (int32_t)mode;
            if (!s->record_exec)
                return;
        }
    }

    mglColorMaterial(ctx, face, mode);
}

static void rec_TEXENVFV(GLMContext ctx, GLenum target, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_TEXENVFV);
        if (o)
        {
            o->i0 = (int32_t)target; o->i1 = (int32_t)pname;
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglTexEnvfv(ctx, target, pname, params);
}

static void rec_TEXGENFV(GLMContext ctx, GLenum coord, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_TEXGENFV);
        if (o)
        {
            o->i0 = (int32_t)coord; o->i1 = (int32_t)pname;
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglTexGenfv(ctx, coord, pname, params);
}

static void rec_FOGFV(GLMContext ctx, GLenum pname, const GLfloat *params)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_FOGFV);
        if (o)
        {
            o->i0 = (int32_t)pname;
            if (params)
            {
                o->f[0] = params[0]; o->f[1] = params[1]; o->f[2] = params[2]; o->f[3] = params[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglFogfv(ctx, pname, params);
}

static void rec_CLIPPING_PLANE(GLMContext ctx, GLenum plane, const GLdouble *equation)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_CLIPPING_PLANE);
        if (o)
        {
            o->i0 = (int32_t)plane;
            if (equation)
            {
                o->f[0] = (float)equation[0]; o->f[1] = (float)equation[1];
                o->f[2] = (float)equation[2]; o->f[3] = (float)equation[3];
            }
            if (!s->record_exec)
                return;
        }
    }

    mglClipPlane(ctx, plane, equation);
}

static void rec_LINE_STIPPLE(GLMContext ctx, GLint factor, GLushort pattern)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_LINE_STIPPLE);
        if (o)
        {
            o->i0 = factor; o->i1 = pattern;
            if (!s->record_exec)
                return;
        }
    }

    mglLineStipple(ctx, factor, pattern);
}

static void rec_POLYGON_STIPPLE(GLMContext ctx, const GLubyte *mask)
{
    struct MGLExtState_compat *s = ctx->state.ext_compat;

    if (s->recording && !s->replaying)
    {
        MGLCListOp *o = mgl_compat_list_op(ctx, MGLC_OP_POLYGON_STIPPLE);
        if (o)
        {
            o->blob = malloc(128);
            o->blob_size = 128;
            if (o->blob && mask)
                memcpy(o->blob, mask, 128);
            if (!s->record_exec)
                return;
        }
    }

    mglPolygonStipple(ctx, mask);
}

static void compat_vertex_attrib_pointer(GLMContext ctx, GLuint index, GLint size, GLenum type,
                                         GLboolean normalized, GLsizei stride, const void *pointer)
{
    mgl_compat_vertex_attrib_pointer(ctx, index, size, type, normalized, stride, pointer);
}

static void compat_enable_client_state(GLMContext ctx, GLenum array)
{
    mglEnableClientState(ctx, array);
}

static void compat_disable_client_state(GLMContext ctx, GLenum array)
{
    mglDisableClientState(ctx, array);
}

static void compat_client_active_texture(GLMContext ctx, GLenum texture)
{
    mglClientActiveTexture(ctx, texture);
}

void mgl_compat_init_dispatch(GLMContext ctx)
{
    /* ---- compat behaviour of core entry points ---- */
    ctx->dispatch.enable = rec_ENABLE;
    ctx->dispatch.disable = rec_DISABLE;
    ctx->dispatch.is_enabled = compat_is_enabled;
    ctx->dispatch.get_string = compat_get_string;
    ctx->dispatch.get_stringi = compat_get_stringi;
    ctx->dispatch.bind_vertex_array = compat_bind_vertex_array;
    ctx->dispatch.get_pointerv = compat_get_pointerv;
    ctx->dispatch.link_program = compat_link_program;
    ctx->dispatch.delete_program = compat_delete_program;
    ctx->dispatch.polygon_mode = compat_polygon_mode;
    ctx->dispatch.point_size = compat_point_size;
    ctx->dispatch.point_parameterf = mgl_compat_point_parameterf;
    ctx->dispatch.point_parameterfv = mgl_compat_point_parameterfv;
    ctx->dispatch.point_parameteri = mgl_compat_point_parameteri;
    ctx->dispatch.point_parameteriv = mgl_compat_point_parameteriv;
    ctx->dispatch.vertex_attrib_pointer = compat_vertex_attrib_pointer;
    ctx->dispatch.enable_client_state = compat_enable_client_state;
    ctx->dispatch.disable_client_state = compat_disable_client_state;
    ctx->dispatch.client_active_texture = compat_client_active_texture;
    ctx->dispatch.draw_arrays = compat_draw_arrays;
    ctx->dispatch.draw_elements = compat_draw_elements;
    ctx->dispatch.draw_elements_base_vertex = compat_draw_elements_base_vertex;
    ctx->dispatch.draw_arrays_instanced = compat_draw_arrays_instanced;
    ctx->dispatch.draw_elements_instanced = compat_draw_elements_instanced;

    /* ---- matrix stack ---- */
    ctx->dispatch.matrix_mode = rec_MATRIX_MODE;
    ctx->dispatch.load_identity = rec_LOAD_IDENTITY;
    ctx->dispatch.load_matrixf = rec_LOAD_MATRIXF;
    ctx->dispatch.load_matrixd = mglLoadMatrixd;
    ctx->dispatch.mult_matrixf = rec_MULT_MATRIXF;
    ctx->dispatch.mult_matrixd = mglMultMatrixd;
    ctx->dispatch.load_transpose_matrixf = mglLoadTransposeMatrixf;
    ctx->dispatch.load_transpose_matrixd = mglLoadTransposeMatrixd;
    ctx->dispatch.mult_transpose_matrixf = mglMultTransposeMatrixf;
    ctx->dispatch.mult_transpose_matrixd = mglMultTransposeMatrixd;
    ctx->dispatch.push_matrix = rec_PUSH_MATRIX;
    ctx->dispatch.pop_matrix = rec_POP_MATRIX;
    ctx->dispatch.ortho = mglOrtho;
    ctx->dispatch.frustum = mglFrustum;
    ctx->dispatch.translatef = mglTranslatef;
    ctx->dispatch.translated = mglTranslated;
    ctx->dispatch.scalef = mglScalef;
    ctx->dispatch.scaled = mglScaled;
    ctx->dispatch.translatef = rec_TRANSLATEF;
    ctx->dispatch.scalef = rec_SCALEF;
    ctx->dispatch.rotatef = rec_ROTATEF;
    ctx->dispatch.rotated = mglRotated;

    /* ---- compiled state commands ---- */
    ctx->dispatch.cull_face = rec_CULL_FACE;
    ctx->dispatch.front_face = rec_FRONT_FACE;
    ctx->dispatch.depth_func = rec_DEPTH_FUNC;
    ctx->dispatch.depth_mask = rec_DEPTH_MASK;
    ctx->dispatch.blend_func = rec_BLEND_FUNC;
    ctx->dispatch.logic_op = rec_LOGIC_OP;
    ctx->dispatch.stencil_func = rec_STENCIL_FUNC;
    ctx->dispatch.stencil_op = rec_STENCIL_OP;
    ctx->dispatch.stencil_mask = rec_STENCIL_MASK;
    ctx->dispatch.depth_range = rec_DEPTH_RANGE;
    ctx->dispatch.viewport = rec_VIEWPORT;
    ctx->dispatch.scissor = rec_SCISSOR;
    ctx->dispatch.clear_color = rec_CLEAR_COLOR;
    ctx->dispatch.clear_depthf = mglClearDepthf;
    ctx->dispatch.clear_depth = rec_CLEAR_DEPTH;
    ctx->dispatch.clear_stencil = rec_CLEAR_STENCIL;
    ctx->dispatch.clear = rec_CLEAR;
    ctx->dispatch.color_mask = rec_COLOR_MASK;
    ctx->dispatch.bind_texture = rec_BIND_TEXTURE;
    ctx->dispatch.active_texture = rec_ACTIVE_TEXTURE;
    ctx->dispatch.lightfv = rec_LIGHTFV;
    ctx->dispatch.lightiv = mglLightiv;
    ctx->dispatch.lightf = mglLightf;
    ctx->dispatch.lighti = mglLighti;
    ctx->dispatch.light_modelfv = rec_LIGHTMODELFV;
    ctx->dispatch.light_modeliv = mglLightModeliv;
    ctx->dispatch.light_modelf = mglLightModelf;
    ctx->dispatch.light_modeli = mglLightModeli;
    ctx->dispatch.materialfv = rec_MATERIALFV;
    ctx->dispatch.materialiv = mglMaterialiv;
    ctx->dispatch.materialf = mglMaterialf;
    ctx->dispatch.materiali = mglMateriali;
    ctx->dispatch.color_material = rec_COLORMATERIAL;
    ctx->dispatch.tex_envfv = rec_TEXENVFV;
    ctx->dispatch.tex_enviv = mglTexEnviv;
    ctx->dispatch.tex_envf = mglTexEnvf;
    ctx->dispatch.tex_envi = mglTexEnvi;
    ctx->dispatch.tex_genfv = rec_TEXGENFV;
    ctx->dispatch.tex_geniv = mglTexGeniv;
    ctx->dispatch.tex_genf = mglTexGenf;
    ctx->dispatch.tex_gend = mglTexGend;
    ctx->dispatch.tex_geni = mglTexGeni;
    ctx->dispatch.tex_gendv = mglTexGendv;
    ctx->dispatch.fogfv = rec_FOGFV;
    ctx->dispatch.fogiv = mglFogiv;
    ctx->dispatch.fogf = mglFogf;
    ctx->dispatch.fogi = mglFogi;
    ctx->dispatch.clip_plane = rec_CLIPPING_PLANE;
    ctx->dispatch.shade_model = rec_SHADE_MODEL;
    ctx->dispatch.line_stipple = rec_LINE_STIPPLE;
    ctx->dispatch.polygon_stipple = rec_POLYGON_STIPPLE;
    ctx->dispatch.line_width = rec_LINE_WIDTH;
    ctx->dispatch.flush = rec_FLUSH;
    ctx->dispatch.finish = rec_FINISH;

    /* ---- compat GLSL ---- */
    ctx->dispatch.shader_source = compat_shader_source2;
    ctx->dispatch.get_uniform_location = mgl_compat_get_uniform_location;

    /* ---- legacy texture formats (LUMINANCE / ALPHA / INTENSITY) ---- */
    ctx->dispatch.tex_image1D = compat_tex_image1D;
    ctx->dispatch.tex_image2D = compat_tex_image2D;
    ctx->dispatch.tex_image3D = compat_tex_image3D;
    ctx->dispatch.get_tex_image = compat_get_tex_image;
    ctx->dispatch.get_tex_level_parameteriv = compat_get_tex_level_parameteriv;

    /* ---- compat texture parameters (GL_CLAMP, GENERATE_MIPMAP) ---- */
    ctx->dispatch.tex_parameteri = compat_tex_parameteri;
    ctx->dispatch.tex_parameterf = compat_tex_parameterf;
    ctx->dispatch.tex_parameteriv = compat_tex_parameteriv;
    ctx->dispatch.tex_parameterfv = compat_tex_parameterfv;
    ctx->dispatch.get_tex_parameteriv = compat_get_tex_parameteriv;
    ctx->dispatch.get_tex_parameterfv = compat_get_tex_parameterfv;
}

#endif /* COMPAT_DISPATCH_H */
