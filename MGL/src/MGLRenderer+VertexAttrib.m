/*
 * MGLRenderer+VertexAttrib.m — Metal side of the 'vattrib' domain.
 *
 * GL vertex array state -> MTLVertexDescriptor + vertex buffer bindings.
 *
 *  - Each GL vertex buffer binding used by an enabled attribute gets one Metal vertex buffer
 *    index (allocated after MGL's uniform/storage buffers in the vertex BufferMapList) and one
 *    MTLVertexBufferLayoutDescriptor (stride, PerVertex / PerInstance stepRate = divisor,
 *    Constant for stride 0).  Attributes fetch from it with their relative offset.
 *  - Shader inputs whose array is disabled (or has no buffer) read the current generic value:
 *    a MTLVertexStepFunctionConstant attribute into a 256 byte setVertexBytes block at
 *    MGL_VA_CONST_BUFFER_INDEX, formatted as Float4 / Int4 / UInt4 to match the shader input.
 *  - Arrays Metal cannot fetch directly (GL_FIXED, GL_DOUBLE / L*, non-normalized packed
 *    2_10_10_10, BGRA packed, 32 bit normalized ints, unnormalized integers read by float inputs,
 *    signed/unsigned mismatches, strides or offsets not a multiple of 4, 10F_11F_11F before
 *    macOS 14) are converted on the CPU at draw time into a tightly packed float4/int4 shadow
 *    stream bound at MGL_VA_SHADOW_BUFFER_TOP and below.  Metal has no fp64 vertex inputs, so
 *    double attributes are always delivered as float (the shader input must be float — MSL
 *    cannot express dvec inputs).
 */

#import "MGLRenderer_Private.h"
// gpuaudit: cached shadow attribute streams (CPU format conversion off the draw path)
#include "mgl_ext_gpuaudit.h"
#import "mgl_ext_vattrib.h"
#import "mgl_ext_introspect.h"

#define F4(_a_, _b_, _c_, _d_) switch (n) { case 1: return _a_; case 2: return _b_; case 3: return _c_; case 4: return _d_; } break

/* Metal vertex format reading GL array data of (type, size, normalized) directly, or Invalid */
MTLVertexFormat mglVertexFormatForAttrib(GLenum type, GLint size, bool normalized, bool integer)
{
    int n = (size == GL_BGRA) ? 4 : size;

    if (integer)
        normalized = false;

    if (size == GL_BGRA)
    {
        if (type == GL_UNSIGNED_BYTE && normalized)
            return MTLVertexFormatUChar4Normalized_BGRA;

        return MTLVertexFormatInvalid;
    }

    switch (type)
    {
        case GL_UNSIGNED_BYTE:
            if (normalized) { F4(MTLVertexFormatUCharNormalized, MTLVertexFormatUChar2Normalized, MTLVertexFormatUChar3Normalized, MTLVertexFormatUChar4Normalized); }
            else            { F4(MTLVertexFormatUChar, MTLVertexFormatUChar2, MTLVertexFormatUChar3, MTLVertexFormatUChar4); }
            break;

        case GL_BYTE:
            if (normalized) { F4(MTLVertexFormatCharNormalized, MTLVertexFormatChar2Normalized, MTLVertexFormatChar3Normalized, MTLVertexFormatChar4Normalized); }
            else            { F4(MTLVertexFormatChar, MTLVertexFormatChar2, MTLVertexFormatChar3, MTLVertexFormatChar4); }
            break;

        case GL_UNSIGNED_SHORT:
            if (normalized) { F4(MTLVertexFormatUShortNormalized, MTLVertexFormatUShort2Normalized, MTLVertexFormatUShort3Normalized, MTLVertexFormatUShort4Normalized); }
            else            { F4(MTLVertexFormatUShort, MTLVertexFormatUShort2, MTLVertexFormatUShort3, MTLVertexFormatUShort4); }
            break;

        case GL_SHORT:
            if (normalized) { F4(MTLVertexFormatShortNormalized, MTLVertexFormatShort2Normalized, MTLVertexFormatShort3Normalized, MTLVertexFormatShort4Normalized); }
            else            { F4(MTLVertexFormatShort, MTLVertexFormatShort2, MTLVertexFormatShort3, MTLVertexFormatShort4); }
            break;

        case GL_HALF_FLOAT:
            F4(MTLVertexFormatHalf, MTLVertexFormatHalf2, MTLVertexFormatHalf3, MTLVertexFormatHalf4);
            break;

        case GL_FLOAT:
            F4(MTLVertexFormatFloat, MTLVertexFormatFloat2, MTLVertexFormatFloat3, MTLVertexFormatFloat4);
            break;

        case GL_INT:
            if (normalized) return MTLVertexFormatInvalid;  // no 32 bit normalized formats
            F4(MTLVertexFormatInt, MTLVertexFormatInt2, MTLVertexFormatInt3, MTLVertexFormatInt4);
            break;

        case GL_UNSIGNED_INT:
            if (normalized) return MTLVertexFormatInvalid;
            F4(MTLVertexFormatUInt, MTLVertexFormatUInt2, MTLVertexFormatUInt3, MTLVertexFormatUInt4);
            break;

        case GL_INT_2_10_10_10_REV:
            return (normalized && size == 4) ? MTLVertexFormatInt1010102Normalized : MTLVertexFormatInvalid;

        case GL_UNSIGNED_INT_2_10_10_10_REV:
            return (normalized && size == 4) ? MTLVertexFormatUInt1010102Normalized : MTLVertexFormatInvalid;

        case GL_UNSIGNED_INT_10F_11F_11F_REV:
            if (@available(macOS 14.0, *))
                return MTLVertexFormatFloatRG11B10;
            return MTLVertexFormatInvalid;

        default:
            break;  // GL_FIXED, GL_DOUBLE: converted on the CPU
    }

    return MTLVertexFormatInvalid;
}

static int shaderKindForDataType(MTLDataType t, int *comps)
{
    switch (t)
    {
        case MTLDataTypeFloat:  case MTLDataTypeHalf:  *comps = 1; return MGL_VA_KIND_FLOAT;
        case MTLDataTypeFloat2: case MTLDataTypeHalf2: *comps = 2; return MGL_VA_KIND_FLOAT;
        case MTLDataTypeFloat3: case MTLDataTypeHalf3: *comps = 3; return MGL_VA_KIND_FLOAT;
        case MTLDataTypeFloat4: case MTLDataTypeHalf4: *comps = 4; return MGL_VA_KIND_FLOAT;

        case MTLDataTypeInt:  case MTLDataTypeShort:  case MTLDataTypeChar:  case MTLDataTypeBool:  *comps = 1; return MGL_VA_KIND_INT;
        case MTLDataTypeInt2: case MTLDataTypeShort2: case MTLDataTypeChar2: case MTLDataTypeBool2: *comps = 2; return MGL_VA_KIND_INT;
        case MTLDataTypeInt3: case MTLDataTypeShort3: case MTLDataTypeChar3: case MTLDataTypeBool3: *comps = 3; return MGL_VA_KIND_INT;
        case MTLDataTypeInt4: case MTLDataTypeShort4: case MTLDataTypeChar4: case MTLDataTypeBool4: *comps = 4; return MGL_VA_KIND_INT;

        case MTLDataTypeUInt:  case MTLDataTypeUShort:  case MTLDataTypeUChar:  *comps = 1; return MGL_VA_KIND_UINT;
        case MTLDataTypeUInt2: case MTLDataTypeUShort2: case MTLDataTypeUChar2: *comps = 2; return MGL_VA_KIND_UINT;
        case MTLDataTypeUInt3: case MTLDataTypeUShort3: case MTLDataTypeUChar3: *comps = 3; return MGL_VA_KIND_UINT;
        case MTLDataTypeUInt4: case MTLDataTypeUShort4: case MTLDataTypeUChar4: *comps = 4; return MGL_VA_KIND_UINT;

        default:
            *comps = 4;
            return MGL_VA_KIND_FLOAT;
    }
}

static bool isSignedIntType(GLenum t)   { return t == GL_BYTE || t == GL_SHORT || t == GL_INT; }
static bool isUnsignedIntType(GLenum t) { return t == GL_UNSIGNED_BYTE || t == GL_UNSIGNED_SHORT || t == GL_UNSIGNED_INT; }

@implementation MGLRenderer (VertexAttrib)

- (struct MGLExtState_vattrib *) vattribState
{
    return ctx->state.ext_vattrib;
}

#pragma mark buffer map

/* called from mapGLBuffersToMTLBufferMap for the vertex stage after uniforms / storage buffers */
- (bool) vattribMapVertexBuffers: (BufferMapList *) buffer_map
{
    struct MGLExtState_vattrib *s = [self vattribState];
    VertexArray *vao = ctx->state.vao;

    if (!s || !vao)
        return true;

    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVattribDrawPlan *plan = &s->plan;
    GLuint max_attribs = ctx->state.max_vertex_attribs;

    if (max_attribs == 0 || max_attribs > MAX_ATTRIBS)
        max_attribs = MAX_ATTRIBS;

    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
        plan->binding_index[b] = -1;

    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
    {
        GLuint mask = 0;

        for (GLuint a = 0; a < max_attribs; a++)
        {
            if ((vao->enabled_attribs & (1u << a)) && va->fmt[a].binding == (GLuint)b)
                mask |= (1u << a);
        }

        if (mask == 0 || va->binding[b].buffer == NULL)
            continue;

        if (buffer_map->count >= MAX_MAPPED_BUFFERS || buffer_map->count >= MGL_VA_SHADOW_BUFFER_TOP)
        {
            NSLog(@"MGL ERROR: vattrib: too many vertex buffers (%u)", buffer_map->count);
            return false;
        }

        BufferMap *m = &buffer_map->buffers[buffer_map->count];
        m->attribute_mask = mask;
        m->buffer_base_index = 0;
        m->buf = va->binding[b].buffer;
        m->offset = va->binding[b].offset;

        plan->binding_index[b] = (int)buffer_map->count;
        buffer_map->count++;
    }

    return true;
}

#pragma mark vertex descriptor

- (MTLVertexDescriptor *) vattribGenerateVertexDescriptor
{
    struct MGLExtState_vattrib *s = [self vattribState];
    VertexArray *vao = ctx->state.vao;
    Program *program = ctx->state.program;

    MTLVertexDescriptor *desc = [[MTLVertexDescriptor alloc] init];

    if (!s || !vao || !program || !program->shader_slots[_VERTEX_SHADER])
        return desc;

    id<MTLFunction> vfunc = (__bridge id<MTLFunction>)(program->shader_slots[_VERTEX_SHADER]->mtl_data.function);
    if (vfunc == nil)
        return desc;

    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVattribDrawPlan *plan = &s->plan;
    int next_shadow = MGL_VA_SHADOW_BUFFER_TOP;
    int max_binding_index = -1;
    bool layout_done[MGL_VA_MAX_BINDINGS] = { false };

    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
        if (plan->binding_index[b] > max_binding_index)
            max_binding_index = plan->binding_index[b];

    plan->const_mask = 0;
    for (int a = 0; a < MAX_ATTRIBS; a++)
    {
        plan->fetch[a] = MGL_VA_FETCH_NONE;
        plan->shadow_index[a] = -1;
        plan->owner_gl[a] = -1;
    }

    /* pass 1: record what the shader consumes at every attribute index (the
       double-slot resolution in the main loop needs the kind of location-1) */
    for (MTLVertexAttribute *attr in vfunc.vertexAttributes)
    {
        if (!attr.active) continue;
        NSUInteger pl = attr.attributeIndex;
        if (pl >= MAX_ATTRIBS) continue;
        int pcomps = 0;
        plan->shader_kind[pl] = shaderKindForDataType(attr.attributeType, &pcomps);
        plan->shader_comps[pl] = pcomps;
    }

    for (MTLVertexAttribute *attr in vfunc.vertexAttributes)
    {
        if (!attr.active)
            continue;

        NSUInteger loc = attr.attributeIndex;
        if (loc >= MAX_ATTRIBS)
        {
            NSLog(@"MGL ERROR: vattrib: shader attribute location %lu out of range", (unsigned long)loc);
            return nil;
        }

        int comps = plan->shader_comps[loc];
        int kind = plan->shader_kind[loc];

        /* fp64 double slots: the lowering fetches double/dvecN inputs as raw
           u32 words (two per component); dvec3/dvec4 span locations k and k+1,
           both fed from GL attribute k with 8 bytes per component.  The GL view
           of the program tells which slot is a double input (a uvec4 input at
           the same index is NOT). */
        GLint owner_loc = (GLint)loc, cbase = 0;
        GLenum ity = mgl_introspect_slot_owner(ctx, program->name, (GLint)loc, &owner_loc, &cbase);
        plan->owner_gl[loc] = owner_loc;   /* GL location feeding this internal slot */
        bool dbl_slot = (kind == MGL_VA_KIND_UINT) &&
                        (ity == GL_DOUBLE || ity == GL_DOUBLE_VEC2 || ity == GL_DOUBLE_VEC3 || ity == GL_DOUBLE_VEC4);
        int owner = dbl_slot ? (int)owner_loc : (int)loc;

        MGLVertexAttribFormat *f = &va->fmt[owner];
        MGLVertexBinding *b = &va->binding[f->binding];
        int bidx = plan->binding_index[f->binding];

        if (dbl_slot)
        {
            /* double-word slot: dvec1/dvec2 use location k only, dvec3/dvec4
               use k and k+1 (this slot carries components [cbase, cbase+n)) */
            int words = f->size - cbase;
            if (words <= 0) words = 1;
            if (words > 2) words = 2;
            plan->dbl[loc] = cbase ? 2 : 1;
            plan->dbl_words[loc] = words;
            MTLVertexFormat wfmt = (words >= 2) ? MTLVertexFormatUInt4 : MTLVertexFormatUInt2;
            GLsizei elem = f->size * 8;

            if (!(vao->enabled_attribs & (1u << (GLuint)owner)) || b->buffer == NULL || bidx < 0)
            {
                // disabled array: feed the current generic double value bits
                desc.attributes[loc].format = wfmt;
                desc.attributes[loc].offset = loc * 16;
                desc.attributes[loc].bufferIndex = MGL_VA_CONST_BUFFER_INDEX;
                plan->fetch[loc] = MGL_VA_FETCH_CONST;
                plan->const_mask |= (1u << loc);
                continue;
            }

            bool aligned = (b->stride % 4) == 0 && (b->offset % 4) == 0 &&
                           (f->relativeoffset % 4) == 0 &&
                           (b->stride == 0 || f->relativeoffset + elem <= (GLuint)b->stride);

            if (f->is_long && aligned)
            {
                // glVertexAttribLPointer: the raw words in the buffer ARE the value
                desc.attributes[loc].format = wfmt;
                desc.attributes[loc].offset = f->relativeoffset + cbase * 8;
                desc.attributes[loc].bufferIndex = bidx;
                plan->fetch[loc] = MGL_VA_FETCH_DIRECT;
                if (!layout_done[bidx])
                {
                    desc.layouts[bidx].stride = b->stride ? b->stride : elem;
                    desc.layouts[bidx].stepFunction = (b->divisor == 0) ? MTLVertexStepFunctionPerVertex
                                                                      : MTLVertexStepFunctionPerInstance;
                    desc.layouts[bidx].stepRate = 1;
                    layout_done[bidx] = true;
                }
            }
            else
            {
                // converted shadow stream of the attribute's double words (16-32
                // bytes per element), shared by both slots of a dvec3/dvec4
                if (next_shadow <= max_binding_index)
                {
                    NSLog(@"MGL ERROR: vattrib: out of Metal vertex buffer indices for converted attributes");
                    return nil;
                }
                if (plan->dbl[loc] == 1)
                {
                    plan->shadow_index[loc] = --next_shadow;
                    plan->shadow_stride[loc] = (elem + 15) & ~15;
                }
                else
                {
                    plan->shadow_index[loc] = plan->shadow_index[owner];   // same stream
                }
                int sidx = plan->shadow_index[loc];
                desc.attributes[loc].format = wfmt;
                desc.attributes[loc].offset = cbase * 8;
                desc.attributes[loc].bufferIndex = sidx;
                desc.layouts[sidx].stride = plan->shadow_stride[owner];
                desc.layouts[sidx].stepFunction = (b->divisor == 0) ? MTLVertexStepFunctionPerVertex
                                                                   : MTLVertexStepFunctionPerInstance;
                desc.layouts[sidx].stepRate = 1;
                plan->fetch[loc] = MGL_VA_FETCH_SHADOW;
            }
            continue;
        }

        if (!(vao->enabled_attribs & (1u << loc)) || b->buffer == NULL || bidx < 0)
        {
            // disabled array: feed the current generic value
            desc.attributes[loc].format = (kind == MGL_VA_KIND_INT) ? MTLVertexFormatInt4 :
                                          (kind == MGL_VA_KIND_UINT) ? MTLVertexFormatUInt4 : MTLVertexFormatFloat4;
            desc.attributes[loc].offset = loc * 16;
            desc.attributes[loc].bufferIndex = MGL_VA_CONST_BUFFER_INDEX;
            plan->fetch[loc] = MGL_VA_FETCH_CONST;
            plan->const_mask |= (1u << loc);
            continue;
        }

        // can Metal fetch the GL array directly?
        MTLVertexFormat format = MTLVertexFormatInvalid;
        GLsizei elem = mgl_va_element_size(f->type, f->size);
        bool aligned = (b->stride % 4) == 0 && (b->offset % 4) == 0 && (f->relativeoffset % 4) == 0 &&
                       (b->stride == 0 || f->relativeoffset + elem <= (GLuint)b->stride);

        if (aligned && !f->is_long)
        {
            if (kind == MGL_VA_KIND_FLOAT)
            {
                bool unnormalized_int = !f->normalized && (isSignedIntType(f->type) || isUnsignedIntType(f->type));

                if (!f->integer && !unnormalized_int)
                    format = mglVertexFormatForAttrib(f->type, f->size, f->normalized, false);
            }
            else
            {
                bool sign_ok = (kind == MGL_VA_KIND_INT) ? isSignedIntType(f->type) : isUnsignedIntType(f->type);

                if (sign_ok && (f->integer || !f->normalized) && f->size != GL_BGRA)
                    format = mglVertexFormatForAttrib(f->type, f->size, false, true);
            }
        }

        if (format != MTLVertexFormatInvalid)
        {
            desc.attributes[loc].format = format;
            desc.attributes[loc].offset = f->relativeoffset;
            desc.attributes[loc].bufferIndex = bidx;
            plan->fetch[loc] = MGL_VA_FETCH_DIRECT;

            if (!layout_done[f->binding])
            {
                MTLVertexBufferLayoutDescriptor *layout = desc.layouts[bidx];

                if (b->stride == 0)
                {
                    // every vertex reads the same element
                    layout.stride = 16 * ((elem + 15) / 16);
                    layout.stepFunction = MTLVertexStepFunctionConstant;
                    layout.stepRate = 0;
                }
                else
                {
                    layout.stride = b->stride;
                    layout.stepFunction = b->divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
                    layout.stepRate = b->divisor ? b->divisor : 1;
                }

                layout_done[f->binding] = true;
            }
            continue;
        }

        // CPU converted shadow stream
        if (next_shadow <= max_binding_index)
        {
            NSLog(@"MGL ERROR: vattrib: out of Metal vertex buffer indices for converted attributes");
            return nil;
        }

        int sidx = next_shadow--;

        desc.attributes[loc].format = (kind == MGL_VA_KIND_INT) ? MTLVertexFormatInt4 :
                                      (kind == MGL_VA_KIND_UINT) ? MTLVertexFormatUInt4 : MTLVertexFormatFloat4;
        desc.attributes[loc].offset = 0;
        desc.attributes[loc].bufferIndex = sidx;

        MTLVertexBufferLayoutDescriptor *layout = desc.layouts[sidx];
        layout.stride = 16;
        if (b->stride == 0)
        {
            layout.stepFunction = MTLVertexStepFunctionConstant;
            layout.stepRate = 0;
        }
        else
        {
            layout.stepFunction = b->divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            layout.stepRate = b->divisor ? b->divisor : 1;
        }

        plan->fetch[loc] = MGL_VA_FETCH_SHADOW;
        plan->shadow_index[loc] = sidx;
    }

    if (plan->const_mask)
    {
        if (max_binding_index >= MGL_VA_CONST_BUFFER_INDEX)
        {
            NSLog(@"MGL ERROR: vattrib: constant attribute buffer index collides with vertex buffers");
            return nil;
        }

        MTLVertexBufferLayoutDescriptor *layout = desc.layouts[MGL_VA_CONST_BUFFER_INDEX];
        layout.stride = MAX_ATTRIBS * 16;
        layout.stepFunction = MTLVertexStepFunctionConstant;
        layout.stepRate = 0;
    }

    plan->valid = 1;

    if (getenv("MGL_VATTRIB_DEBUG"))
        for (int i = 0; i < 31; i++)
            if (desc.layouts[i].stride && (i < 8 || i > 26))
                NSLog(@"vattrib: layout %d stride %lu step %lu rate %lu", i, (unsigned long)desc.layouts[i].stride,
                      (unsigned long)desc.layouts[i].stepFunction, (unsigned long)desc.layouts[i].stepRate);

    return desc;
}

#pragma mark draw time binding

- (void) vattribBindBuffer: (Buffer *) buf offset: (GLintptr) offset index: (int) index encoder: (id<MTLRenderCommandEncoder>) enc
{
    static const float zeros[4] = { 0, 0, 0, 0 };

    if (buf->data.mtl_data == NULL && buf->size >= 4096)
        [self bindMTLBuffer: buf];

    if (buf->data.mtl_data)
    {
        id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)(buf->data.mtl_data);

        if ((NSUInteger)offset < mtl.length)
        {
            [enc setVertexBuffer: mtl offset: offset atIndex: index];
            return;
        }
    }
    else if (buf->data.buffer_data && offset < buf->size)
    {
        NSUInteger len = (NSUInteger)(buf->size - offset);
        if (len > 4096) len = 4096;
        [enc setVertexBytes: (const void *)(buf->data.buffer_data + offset) length: len atIndex: index];
        return;
    }

    [enc setVertexBytes: zeros length: sizeof(zeros) atIndex: index];
}

- (const GLubyte *) vattribCPUData: (Buffer *) buf
{
    if (buf->data.mtl_data)
        return (const GLubyte *)[(__bridge id<MTLBuffer>)(buf->data.mtl_data) contents];

    return (const GLubyte *)buf->data.buffer_data;
}

- (void) vattribBindShadow: (int) attrib index: (int) index encoder: (id<MTLRenderCommandEncoder>) enc
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    MGLVertexAttribFormat *f = &va->fmt[attrib];
    MGLVertexBinding *b = &va->binding[f->binding];
    struct MGLExtState_vattrib *s = [self vattribState];
    int kind = s->plan.shader_kind[attrib];
    bool dbl = s->plan.dbl[attrib] != 0;   /* the plan marked this slot as a double word slot */

    const GLubyte *base = [self vattribCPUData: b->buffer];
    GLsizeiptr size = b->buffer->size;
    GLsizei elem = dbl ? (GLsizei)f->size * 8 : mgl_va_element_size(f->type, f->size);
    size_t stride_out = dbl ? (size_t)s->plan.shadow_stride[attrib] : 16;
    GLsizeiptr start = b->offset + f->relativeoffset;
    size_t n = 0;

    if (base && start >= 0 && start + elem <= size)
        n = (b->stride == 0) ? 1 : (size_t)((size - start - elem) / b->stride) + 1;

    /* gpuaudit: the converted stream only depends on the source bytes and the layout,
     * so it is cached as an MTLBuffer keyed on the buffer's content generation instead
     * of being rebuilt (calloc + per-element conversion + newBufferWithBytes) per draw */
    void *cached = mgl_gpuaudit_shadow_lookup(ctx, b->buffer, start, b->stride, f->type, f->size,
                                              dbl ? MGL_VA_KIND_DOUBLE : kind);
    if (cached)
    {
        [enc setVertexBuffer: (__bridge id<MTLBuffer>)cached offset: 0 atIndex: index];
        return;
    }

    size_t count = n ? n : 1;
    GLuint *out = (GLuint *)calloc(count, stride_out);
    if (!out)
        return;

    for (size_t i = 0; i < n; i++)
    {
        const GLubyte *src = base + start + i * (size_t)b->stride;
        GLuint *o = &out[i * (stride_out / 4)];

        if (dbl)
            mgl_va_fetch_dbl_words(f, src, o);
        else if (kind == MGL_VA_KIND_FLOAT)
            mgl_va_fetch_float(f, src, (GLfloat *)o);
        else
            mgl_va_fetch_int(f, src, (GLint *)o);
    }

    size_t len = count * stride_out;

    void *stored = mgl_gpuaudit_shadow_store(ctx, b->buffer, start, b->stride, f->type, f->size,
                                             dbl ? MGL_VA_KIND_DOUBLE : kind, out, len);
    if (stored)
    {
        [enc setVertexBuffer: (__bridge id<MTLBuffer>)stored offset: 0 atIndex: index];
        free(out);
        return;
    }

    if (len <= 4096)
    {
        [enc setVertexBytes: out length: len atIndex: index];
    }
    else
    {
        id<MTLBuffer> tmp = [_device newBufferWithBytes: out length: len options: MTLResourceStorageModeShared];
        [enc setVertexBuffer: tmp offset: 0 atIndex: index];
    }

    free(out);
}

- (void) vattribBindConstants: (id<MTLRenderCommandEncoder>) enc
{
    struct MGLExtState_vattrib *s = [self vattribState];
    MGLVattribDrawPlan *plan = &s->plan;
    union { GLfloat f[4]; GLint i[4]; GLuint u[4]; } data[MAX_ATTRIBS];

    memset(data, 0, sizeof(data));

    for (int a = 0; a < MAX_ATTRIBS; a++)
    {
        if (!(plan->const_mask & (1u << a)))
            continue;

        MGLCurrentAttrib *cur = &s->current[a];

        /* fp64 word slot: the current double value's raw u32 words (component
           c occupies words 2c..2c+1; the high slot carries components 2,3) */
        if (plan->dbl[a])
        {
            int own = plan->owner_gl[a] >= 0 ? plan->owner_gl[a] : a;   /* owning GL location */
            MGLCurrentAttrib *c2 = &s->current[own];
            int c0 = (plan->dbl[a] == 2) ? 2 : 0;
            int nw = (plan->dbl_words[a] >= 2) ? 2 : 1;
            for (int c = c0; c < c0 + nw; c++)
            {
                GLdouble v;
                switch (c2->kind)
                {
                    case MGL_VA_KIND_INT:   v = (GLdouble)c2->v.i[c]; break;
                    case MGL_VA_KIND_UINT:  v = (GLdouble)c2->v.u[c]; break;
                    case MGL_VA_KIND_DOUBLE: v = c2->d[c]; break;
                    default:                v = (GLdouble)c2->v.f[c]; break;
                }
                memcpy(&data[a].u[(c - c0) * 2], &v, 8);
            }
            continue;
        }

        for (int c = 0; c < 4; c++)
        {
            switch (plan->shader_kind[a])
            {
                case MGL_VA_KIND_FLOAT:
                    data[a].f[c] = (cur->kind == MGL_VA_KIND_INT) ? (GLfloat)cur->v.i[c] :
                                   (cur->kind == MGL_VA_KIND_UINT) ? (GLfloat)cur->v.u[c] : cur->v.f[c];
                    break;

                default:
                    data[a].i[c] = (cur->kind == MGL_VA_KIND_INT || cur->kind == MGL_VA_KIND_UINT) ? cur->v.i[c] :
                                   (GLint)cur->v.f[c];
                    break;
            }
        }
    }

    [enc setVertexBytes: data length: sizeof(data) atIndex: MGL_VA_CONST_BUFFER_INDEX];
}

/* auto-called at the end of processGLState(draw_command = true) */
- (void) extVattribPreDraw: (id<MTLRenderCommandEncoder>) enc
{
    struct MGLExtState_vattrib *s = [self vattribState];
    VertexArray *vao = ctx->state.vao;

    if (!s || !vao || !s->plan.valid || enc == nil)
        return;

    MGLVertexArrayFull *va = mgl_va_full(vao);
    MGLVattribDrawPlan *plan = &s->plan;

    if (getenv("MGL_VATTRIB_DEBUG"))
    {
        for (int a = 0; a < MAX_ATTRIBS; a++)
            if (plan->fetch[a])
                NSLog(@"vattrib: attrib %d fetch %d kind %d binding %u mtl %d shadow %d", a, plan->fetch[a], plan->shader_kind[a],
                      va->fmt[a].binding, plan->binding_index[va->fmt[a].binding], plan->shadow_index[a]);
        for (unsigned i = 0; i < ctx->state.vertex_buffer_map_list.count; i++)
            NSLog(@"vattrib: map[%u] buf %u size %ld mtl %p mask %x", i, ctx->state.vertex_buffer_map_list.buffers[i].buf ? ctx->state.vertex_buffer_map_list.buffers[i].buf->name : 0,
                  ctx->state.vertex_buffer_map_list.buffers[i].buf ? (long)ctx->state.vertex_buffer_map_list.buffers[i].buf->size : 0L,
                  ctx->state.vertex_buffer_map_list.buffers[i].buf ? ctx->state.vertex_buffer_map_list.buffers[i].buf->data.mtl_data : NULL,
                  ctx->state.vertex_buffer_map_list.buffers[i].attribute_mask);
    }

    // uniform / storage buffers of the vertex stage: the map is rebuilt on a buffer base change
    // (DIRTY_BUFFER_BASE_STATE) without a new render encoder, so (re)bind its current contents
    BufferMapList *list = &ctx->state.vertex_buffer_map_list;
    for (GLuint i = 0; i < list->count; i++)
    {
        if (list->buffers[i].attribute_mask == 0 && list->buffers[i].buf)
            [self vattribBindBuffer: list->buffers[i].buf offset: list->buffers[i].offset index: (int)i encoder: enc];
    }

    // vertex buffer bindings, with their binding offsets
    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
    {
        if (plan->binding_index[b] >= 0 && va->binding[b].buffer)
            [self vattribBindBuffer: va->binding[b].buffer offset: va->binding[b].offset index: plan->binding_index[b] encoder: enc];
    }

    for (int a = 0; a < MAX_ATTRIBS; a++)
    {
        if (plan->fetch[a] != MGL_VA_FETCH_SHADOW || !va->binding[va->fmt[a].binding].buffer)
            continue;
        /* dvec3/dvec4: both word slots share one converted stream, owned by
           the attribute's low slot (adjacent internal slots) */
        int owner = (plan->dbl[a] == 2) ? a - 1 : a;
        if (owner != a)
            continue;   /* bound when the low slot is reached */
        [self vattribBindShadow: a index: plan->shadow_index[a] encoder: enc];
    }

    if (plan->const_mask)
        [self vattribBindConstants: enc];
}

@end
