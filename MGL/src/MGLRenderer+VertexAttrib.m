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
#import "MGLRenderer+Compat.h"   /* compat: bind shadow + upload ring (perf session 3) */
// gpuaudit: cached shadow attribute streams (CPU format conversion off the draw path)
#include "mgl_ext_gpuaudit.h"
#import "mgl_ext_vattrib.h"
#import "mgl_ext_introspect.h"
#import "mgl_ext_geom.h"      /* MGL_SSBO_SIZE_BUFFER_INDEX: attribute ordinals must skip it */
#import "mgl_ext_tess.h"      /* tess: the render vertex function is the TES, not the VS */
/* memoized spvBufferSizeConstants probe (MGLRenderer+Geometry.m) — the per-draw
 * probe is program-keyed (Spirv.sizes_scan); the raw-text form is uncached */
extern bool mgl_program_has_size_constants(Program *p, int stage);
#import "mgl_ext_program.h"   /* MGL_PIPELINE_OBJ: a bound pipeline's composite has no name */

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

/* does ANY slot of the double attribute whose first internal slot is <lo> read components
   the array does not supply (a size 3/2/1 format feeding a dvec3/dvec4 element)?  The
   decision between the raw-word DIRECT fetch and the converted shadow stream must be made
   for the attribute as a whole: its low slot may be fully array-supplied while its high
   slot needs the current generic value — the shadow element is padded for both (GL 4.6
   §10.5, KHR-GL46.vertex_attrib_binding basic-inputL). */
static bool vattribDblNeedsPad(MGLVattribDrawPlan *plan, int lo, int size)
{
    for (int s = lo; s <= lo + 1 && s < MAX_ATTRIBS; s++)
    {
        if (s != lo && plan->owner_gl[s] != plan->owner_gl[lo]) break;
        {
            int cbase = (s - lo) * 2;        /* each internal slot covers TWO components */
            int needed = plan->shader_comps[s] / 2;
            int words = size - cbase;

            if (needed < 1) needed = 1;
            if (needed > 2) needed = 2;
            if (words < 0) words = 0;
            if (words > 2) words = 2;
            if (needed - words > 0) return true;
        }
    }
    return false;
}

/* the padded shadow-stream element of the double attribute whose first internal slot is
   <lo>: it must span every component any of the attribute's slots reads — the array's own
   components and the ones the current generic value pads (GL 4.6 §10.5).  Deterministic
   from either slot, so the first-enumerated one can allocate the shared stream. */
static GLsizei vattribDblShadowSpan(MGLVattribDrawPlan *plan, int lo, int size)
{
    int span = size;

    for (int s = lo; s <= lo + 1 && s < MAX_ATTRIBS; s++)
    {
        if (s != lo && plan->owner_gl[s] != plan->owner_gl[lo]) break;
        {
            int needed = plan->shader_comps[s] / 2;
            int cbase = (s - lo) * 2;        /* each internal slot covers TWO components */

            if (needed < 1) needed = 1;
            if (needed > 2) needed = 2;
            if (cbase + needed > span) span = cbase + needed;
        }
    }
    if (span < 2) span = 2;
    return (GLsizei)span * 8;
}

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

        /* ordinal MGL_SSBO_SIZE_BUFFER_INDEX carries SPIRV-Cross's
         * spvBufferSizeConstants argument when the vertex stage reads .length() on
         * an SSBO runtime array (geomBindSSBOSizes binds it on the render encoder):
         * never hand that slot to an attribute — pad the map with an unused entry
         * (no buffer, so every binder skips it) and allocate past it.  Round-44
         * buffer-slot audit; same class as the fixed slot-30 const/xfb collision. */
        if (buffer_map->count == MGL_SSBO_SIZE_BUFFER_INDEX &&
            ctx->state.program &&
            mgl_program_has_size_constants(ctx->state.program, _VERTEX_SHADER))
        {
            BufferMap *pad = &buffer_map->buffers[buffer_map->count];
            pad->from_ubo = 0;   /* r166 */
            pad->attribute_mask = 0;
            pad->buffer_base_index = 0;
            pad->buf = NULL;
            pad->offset = 0;
            pad->size = 0;
            buffer_map->count++;
        }

        BufferMap *m = &buffer_map->buffers[buffer_map->count];
        m->from_ubo = 0;   /* r166 */
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

/* the program whose introspection maps internal slots to GL attribute locations.
 * With a bound program pipeline and no current program, MGL_PIPELINE_PROGRAM_SCOPE
 * substitutes the pipeline's COMPOSITE for ctx->state.program, and that composite is
 * anonymous (name 0) — mgl_introspect_slot_owner would then resolve nothing and the
 * draw plan would feed every Metal slot the GL attribute of the same index (the fp64
 * word slots are renumbered, so e.g. a vec4 position at MSL slot 5 read disabled
 * attribute 5's constant).  The vertex STAGE program owns the inputs: use its name. */
static GLuint vattribIntrospectProgram(GLMContext ctx, Program *program)
{
    if (program && program->name == 0 && ctx->state.program_pipeline)
    {
        Program *vs = MGL_PIPELINE_OBJ(ctx->state.program_pipeline)->base.stage_programs[_VERTEX_SHADER];
        if (vs)
            return vs->name;
    }
    return program ? program->name : 0;
}

/* a vertex input whose internal slots carry raw u32 double words — doubles,
   dvecN and the COLUMN vectors of dmatNxM attributes (the column of a matrix
   has its own GL location, so the per-column slot logic below applies to it
   exactly like a dvecN input of the same width) */
static bool vattribIsDoubleType(GLenum t)
{
    switch (t)
    {
        case GL_DOUBLE: case GL_DOUBLE_VEC2: case GL_DOUBLE_VEC3: case GL_DOUBLE_VEC4:
        case GL_DOUBLE_MAT2: case GL_DOUBLE_MAT3: case GL_DOUBLE_MAT4:
        case GL_DOUBLE_MAT2x3: case GL_DOUBLE_MAT2x4:
        case GL_DOUBLE_MAT3x2: case GL_DOUBLE_MAT3x4:
        case GL_DOUBLE_MAT4x2: case GL_DOUBLE_MAT4x3:
            return true;
        default:
            return false;
    }
}

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
    GLuint iprog = vattribIntrospectProgram(ctx, program);
    int next_shadow = MGL_VA_SHADOW_BUFFER_TOP;
    int max_binding_index = -1;
    bool layout_done[MGL_VA_MAX_BINDINGS] = { false };
    /* the SSBO size-constant slot (see vattribMapVertexBuffers) is off limits for a
     * converted shadow stream as well */
    bool skip_size_slot = mgl_program_has_size_constants(program, _VERTEX_SHADER);

    for (int b = 0; b < MGL_VA_MAX_BINDINGS; b++)
        if (plan->binding_index[b] > max_binding_index)
            max_binding_index = plan->binding_index[b];

    plan->const_mask = 0;
    for (int a = 0; a < MAX_ATTRIBS; a++)
    {
        plan->fetch[a] = MGL_VA_FETCH_NONE;
        plan->shadow_index[a] = -1;
        plan->owner_gl[a] = -1;
        /* dbl/dbl_words/shadow_stride are per-plan too: a stale dbl[] from a
           previous program made the bind path convert a plain int/float shadow
           stream as double words (24-32 byte writes into 16-byte elements,
           heap overflow) and read a stale shadow_stride (calloc of 0 bytes).
           Round-40 crash hunt: heap corruption behind the flaky CTS aborts. */
        plan->dbl[a] = 0;
        plan->dbl_words[a] = 0;
        plan->shadow_stride[a] = 0;
    }

    /* pass 1: record what the shader consumes at every attribute index (the
       double-slot resolution in the main loop needs the kind of location-1).
       owner_gl is filled HERE, not in the main loop: vfunc.vertexAttributes is
       an unordered set, and the whole-attribute pad/shadow-span decisions below
       read owner_gl[] of the OTHER slot of a dvec3/dvec4 — with the fill left to
       the main loop, a low slot enumerated before its high slot saw owner_gl[]
       == -1 and split the attribute across a raw fetch and a converted stream
       (KHR-GL46.vertex_attrib_binding basic-inputL: dvec2@5 fed a dvec4 element,
       the low slot went DIRECT, the high slot's shadow stream was never bound) */
    for (MTLVertexAttribute *attr in vfunc.vertexAttributes)
    {
        if (!attr.active) continue;
        NSUInteger pl = attr.attributeIndex;
        if (pl >= MAX_ATTRIBS) continue;
        int pcomps = 0;
        plan->shader_kind[pl] = shaderKindForDataType(attr.attributeType, &pcomps);
        plan->shader_comps[pl] = pcomps;
        GLint ol = (GLint)pl, cb = 0;
        plan->owner_gl[pl] = mgl_introspect_slot_owner(ctx, iprog, (GLint)pl, &ol, &cb) ? ol : (GLint)pl;
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
        GLenum ity = mgl_introspect_slot_owner(ctx, iprog, (GLint)loc, &owner_loc, &cbase);
        plan->owner_gl[loc] = owner_loc;   /* GL location feeding this internal slot */
        bool dbl_slot = (kind == MGL_VA_KIND_UINT) && vattribIsDoubleType(ity);
        /* the owning GL attribute: the internal slot index is NOT the GL location when
           the fp64 word slots / the interface-location pass renumbered the inputs
           (advanced-bindingUpdate: dvec3@3 takes slots 0-1, the vec4 position ends up at
           MSL slot 5 — reading va->fmt[5] fed it the disabled attribute 5's constant) */
        int owner = (owner_loc >= 0 && owner_loc < MAX_ATTRIBS) ? (int)owner_loc : (int)loc;
        MGLVertexAttribFormat *f = &va->fmt[owner];
        MGLVertexBinding *b = &va->binding[f->binding];
        int bidx = plan->binding_index[f->binding];

        if (dbl_slot)
        {
            /* double-word slot: dvec1/dvec2 use location k only, dvec3/dvec4
               use k and k+1 (this slot carries components [cbase, cbase+n)) */
            /* the doubles this slot's MSL attribute expects (uint4 slot -> 2, uint2 -> 1)
               vs the doubles the array supplies (f->size - cbase): GL 4.6 §10.5 fills the
               missing components from the CURRENT generic vertex attribute value
               (KHR-GL46.vertex_attrib_binding basic-inputL feeds dvec4-declared elements
               with size 3/2/1 formats and expects glVertexAttribL4d for the rest) */
            int needed = plan->shader_comps[loc] / 2;
            int words = f->size - cbase;

            if (words > 2) words = 2;
            if (words < 0) words = 0;
            if (needed < 1) needed = 1;
            if (needed > 2) needed = 2;
            int pad = needed - words;                    /* doubles from the current value */
            bool attr_pad = vattribDblNeedsPad(plan, cbase ? loc - 1 : loc, f->size);

            plan->dbl[loc] = cbase ? 2 : 1;
            plan->dbl_words[loc] = needed;
            MTLVertexFormat wfmt = (needed >= 2) ? MTLVertexFormatUInt4 : MTLVertexFormatUInt2;
            GLsizei elem = (GLsizei)f->size * 8;         /* bytes the array supplies */

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

            if (f->is_long && aligned && !attr_pad)
            {
                // glVertexAttribLPointer: the raw words in the buffer ARE the value.
                // A padded shadow stream cannot read its missing components from the
                // user's buffer — fall through to the converted stream, which appends
                // them from the current generic value (both slots of the attribute
                // together: the low slot alone would split the attribute across a raw
                // and a converted stream)
                desc.attributes[loc].format = wfmt;
                desc.attributes[loc].offset = f->relativeoffset + cbase * 8;
                desc.attributes[loc].bufferIndex = bidx;
                plan->fetch[loc] = MGL_VA_FETCH_DIRECT;
                if (!layout_done[bidx])
                {
                    desc.layouts[bidx].stride = b->stride ? b->stride : elem;
                    desc.layouts[bidx].stepFunction = (b->divisor == 0) ? MTLVertexStepFunctionPerVertex
                                                                      : MTLVertexStepFunctionPerInstance;
                    desc.layouts[bidx].stepRate = b->divisor ? b->divisor : 1;
                    layout_done[bidx] = true;
                }
            }
            else
            {
                // converted shadow stream of the attribute's double words, shared by both
                // slots of a dvec3/dvec4.  When the array supplies fewer components than
                // the shader reads, the stream element is padded to every component the
                // slots cover and the tail words come from the current generic value
                if (next_shadow <= max_binding_index)
                {
                    NSLog(@"MGL ERROR: vattrib: out of Metal vertex buffer indices for converted attributes");
                    return nil;
                }
                /* the stream belongs to the attribute's FIRST internal slot; either slot
                   may be enumerated first (vfunc.vertexAttributes is an unordered set),
                   so allocate it on demand and derive the padded element stride from the
                   whole attribute — never from the (possibly not yet visited) other slot.
                   For an ARRAY attribute that first slot is loc-1 for the high slot:
                   owner_loc is the GL location, which is not an internal slot at all */
                int lo = cbase ? loc - 1 : loc;

                if (lo < 0) lo = loc;
                if (plan->shadow_index[lo] < 0)
                {
                    if (skip_size_slot && next_shadow == MGL_SSBO_SIZE_BUFFER_INDEX)
                        next_shadow--;
                    /* allocate exactly like the non-double converted path below
                       (sidx = next_shadow--, i.e. from the TOP down): the pre-decrement
                       used here first shared index TOP-1 with the first converted float
                       stream of the same draw and one of the two attributes read the
                       other's bytes (advanced-bindingUpdate draw 2: the dvec3 shadow
                       stream was bound over by color_b's) */
                    plan->shadow_index[lo] = next_shadow--;
                    plan->shadow_stride[lo] = (vattribDblShadowSpan(plan, lo, f->size) + 15) & ~15;
                }
                int sidx = plan->shadow_index[lo];
                if (sidx < 0)
                    NSLog(@"MGL vattrib: slot %d (owner %d cbase %d size %d) has no shadow stream",
                          (int)loc, owner, cbase, f->size);
                desc.attributes[loc].format = wfmt;
                desc.attributes[loc].offset = cbase * 8;
                desc.attributes[loc].bufferIndex = sidx;
                desc.layouts[sidx].stride = plan->shadow_stride[lo];
                if (b->stride == 0)
                {
                    // every vertex reads the same element (the stream holds one)
                    desc.layouts[sidx].stepFunction = MTLVertexStepFunctionConstant;
                    desc.layouts[sidx].stepRate = 0;
                }
                else
                {
                    /* an instanced double attribute advances every `divisor`
                       instances — the converted stream holds one element per
                       SOURCE element, so Metal's stepRate must carry the GL
                       divisor (KHR-GL46.vertex_attrib_64bit.vao, divisor 2) */
                    desc.layouts[sidx].stepFunction = (b->divisor == 0) ? MTLVertexStepFunctionPerVertex
                                                                       : MTLVertexStepFunctionPerInstance;
                    desc.layouts[sidx].stepRate = b->divisor ? b->divisor : 1;
                }
                plan->fetch[loc] = MGL_VA_FETCH_SHADOW;
            }
            continue;
        }

        if (!(vao->enabled_attribs & (1u << (GLuint)owner)) || b->buffer == NULL || bidx < 0)
        {
            // disabled array: feed the current generic value of the OWNING attribute
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

        if (skip_size_slot && next_shadow == MGL_SSBO_SIZE_BUFFER_INDEX)
            next_shadow--;
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

    /* perf: latch the knob once — plan rebuilds run per draw in glmark2
     * (attribs are re-specified between draws), and getenv takes the env lock */
    static int sVattrDbgInit = 0, sVattrDbg = 0;
    if (!sVattrDbgInit) { sVattrDbg = getenv("MGL_VATTRIB_DEBUG") != NULL; sVattrDbgInit = 1; }
    if (sVattrDbg)
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
            /* compat perf: shadow-filtered (skip when this buffer+offset is
             * already bound at this index of the current encoder) */
            [self compatBindVertexBuffer:mtl offset:(NSUInteger)offset atIndex:(NSUInteger)index];
            return;
        }
    }
    else if (buf->data.buffer_data && offset < buf->size)
    {
        NSUInteger len = (NSUInteger)(buf->size - offset);
        if (len > 4096) len = 4096;
        /* compat perf: client-array payload through the argument ring */
        [self compatRingVertexBytes:(const void *)(buf->data.buffer_data + offset) length:len atIndex:(NSUInteger)index];
        return;
    }

    [self compatRingVertexBytes:zeros length:sizeof(zeros) atIndex:(NSUInteger)index];
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
    struct MGLExtState_vattrib *s = [self vattribState];
    /* a double word slot reads the GL attribute that feeds it (owner_gl), which is
       the slot itself except for dvec3/dvec4 whose high slot shares the low one */
    int owner = s->plan.owner_gl[attrib];
    if (owner < 0 || owner >= MAX_ATTRIBS) owner = attrib;
    MGLVertexAttribFormat *f = &va->fmt[owner];
    MGLVertexBinding *b = &va->binding[f->binding];
    int kind = s->plan.shader_kind[attrib];
    bool dbl = s->plan.dbl[attrib] != 0;   /* the plan marked this slot as a double word slot */

    const GLubyte *base = [self vattribCPUData: b->buffer];
    GLsizeiptr size = b->buffer->size;
    GLsizei elem = dbl ? (GLsizei)f->size * 8 : mgl_va_element_size(f->type, f->size);
    /* dbl: the element stride must cover the PADDED tail the plan reserved for components
       the array does not supply (GL 4.6 §10.5: they come from the current generic value —
       basic-inputL's size 3/2/1 formats feeding dvec4 elements).  Derive it from the
       plan's own stride (the lo slot records it), with the format-derived value as the
       floor — a stale/zero entry must never make this calloc a 0-byte buffer and then
       write 8*size bytes into it (the heap overflow this used to be). */
    size_t stride_out = dbl ? (size_t)(((size_t)f->size * 8 + 15) & ~(size_t)15) : 16;
    if (dbl)
    {
        int lo = (s->plan.dbl[attrib] == 2) ? attrib - 1 : attrib;
        size_t plan_stride = (lo >= 0 && lo < MAX_ATTRIBS) ? (size_t)s->plan.shadow_stride[lo] : 0;

        if (plan_stride > stride_out) stride_out = plan_stride;
        if (stride_out < 16) stride_out = 16;
    }
    /* components the array does not supply (the plan's pad): filled from the current
       generic value below — never the content-addressed cache (it depends on the current
       value too, which can change between draws) */
    int pad_from = dbl ? (int)f->size : 0;
    int pad_to = dbl ? (int)(stride_out / 8) : 0;
    static int sVattrDbg2Init = 0, sVattrDbg2 = 0;
    if (!sVattrDbg2Init) { sVattrDbg2 = getenv("MGL_VATTRIB_DEBUG") != NULL; sVattrDbg2Init = 1; }
    if (sVattrDbg2)
        NSLog(@"vattrib: shadow bind a=%d dbl=%d owner=%d kind=%d size=%d type=0x%x long=%d stride=%d reloff=%d "
              "shadow_stride=%d elem=%d bufoff=%ld bufsize=%ld", attrib, dbl, s->plan.owner_gl[attrib], kind,
              f->size, f->type, f->is_long, b->stride, f->relativeoffset, s->plan.shadow_stride[attrib], elem,
              (long)b->offset, (long)size);
    GLsizeiptr start = b->offset + f->relativeoffset;
    size_t n = 0;

    if (base && start >= 0 && start + elem <= size)
        n = (b->stride == 0) ? 1 : (size_t)((size - start - elem) / b->stride) + 1;

    /* geom r99: this method is also called with the geometry capture pass's PRIVATE
     * render encoder (MGLRenderer+Geometry.m), which is never _currentRenderEncoder —
     * the compat fast paths below would then encode on the wrong encoder (nil during
     * that pass: a silent no-op, so the stream was never bound).  `own` switches the
     * three bind sites to the encoder this method was handed. */
    bool own = (enc != nil && enc != _currentRenderEncoder);

    /* gpuaudit: the converted stream only depends on the source bytes and the layout,
     * so it is cached as an MTLBuffer keyed on the buffer's content generation instead
     * of being rebuilt (calloc + per-element conversion + newBufferWithBytes) per draw */
    void *cached = (pad_from >= pad_to) ? mgl_gpuaudit_shadow_lookup(ctx, b->buffer, start, b->stride, f->type, f->size,
                                              dbl ? MGL_VA_KIND_DOUBLE : kind) : NULL;
    if (cached)
    {
        if (own)
            [enc setVertexBuffer: (__bridge id<MTLBuffer>)cached offset: 0 atIndex: (NSUInteger)index];
        else
            [self compatBindVertexBuffer:(__bridge id<MTLBuffer>)cached offset:0 atIndex:(NSUInteger)index];
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
        {
            mgl_va_fetch_dbl_words(f, src, o);
            if (pad_to > pad_from)
            {
                /* GL 4.6 §10.5: the components the array does not supply (the plan's
                   pad — a size 3/2/1 format feeding a dvec4 element) come from the
                   CURRENT generic vertex attribute value */
                int own = s->plan.owner_gl[attrib] >= 0 ? s->plan.owner_gl[attrib] : attrib;
                MGLCurrentAttrib *c2 = &s->current[own];

                for (int c = pad_from; c < pad_to && c < 4; c++)
                {
                    GLdouble v;
                    union { double d; GLuint w[2]; } u;

                    switch (c2->kind)
                    {
                        case MGL_VA_KIND_INT:    v = (GLdouble)c2->v.i[c]; break;
                        case MGL_VA_KIND_UINT:   v = (GLdouble)c2->v.u[c]; break;
                        case MGL_VA_KIND_DOUBLE: v = c2->d[c]; break;
                        default:                 v = (GLdouble)c2->v.f[c]; break;
                    }
                    u.d = v;
                    o[c * 2] = u.w[0];
                    o[c * 2 + 1] = u.w[1];
                }
            }
        }
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
        if (own)
            [enc setVertexBytes: out length: len atIndex: (NSUInteger)index];   /* geom capture encoder */
        else
            [self compatRingVertexBytes:out length:len atIndex:(NSUInteger)index];
    }
    else
    {
        id<MTLBuffer> tmp = [_device newBufferWithBytes: out length: len options: MTLResourceStorageModeShared];
        if (own)
            [enc setVertexBuffer: tmp offset: 0 atIndex: (NSUInteger)index];    /* geom capture encoder */
        else
            [self compatBindVertexBuffer:tmp offset:0 atIndex:(NSUInteger)index];
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

        /* the current generic value belongs to the OWNING GL attribute, not to the
           internal slot index (fp64 word slots / renumbered inputs shift it) */
        int own_gl = (plan->owner_gl[a] >= 0 && plan->owner_gl[a] < MAX_ATTRIBS) ? plan->owner_gl[a] : a;
        MGLCurrentAttrib *cur = &s->current[own_gl];

        /* fp64 word slot: the current double value's raw u32 words (component
           c occupies words 2c..2c+1; the high slot carries components 2,3) */
        if (plan->dbl[a])
        {
            MGLCurrentAttrib *c2 = cur;   /* owning GL location */
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

    /* compat perf: constant attribs stream through the argument ring; the
     * memcmp in compatRingVertexBytes skips the encode call whenever the
     * current-attribute block is unchanged between draws.
     * geom r99: the geometry capture pass hands its private encoder here — the
     * ring would encode on the wrong (nil) encoder, so bind directly then. */
    if (enc != nil && enc != _currentRenderEncoder)
        [enc setVertexBytes: data length: sizeof(data) atIndex: (NSUInteger)MGL_VA_CONST_BUFFER_INDEX];
    else
        [self compatRingVertexBytes:data length:sizeof(data) atIndex:(NSUInteger)MGL_VA_CONST_BUFFER_INDEX];
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

    /* perf: latched once (per draw) */
    static int sVattrDbg3Init = 0, sVattrDbg3 = 0;
    if (!sVattrDbg3Init) { sVattrDbg3 = getenv("MGL_VATTRIB_DEBUG") != NULL; sVattrDbg3Init = 1; }
    if (sVattrDbg3)
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
    // tess: the VS runs as a compute kernel, so the render pipeline's vertex
    // function is the (rewritten) evaluation shader — it has no [[stage_in]]
    // inputs, and its uniform/storage buffer slots start at 0.  Binding the
    // vertex array's attribute streams to the render encoder would clobber the
    // evaluation shader's SSBO/UBO arguments (KHR-GL46.shader_storage_buffer_object
    // basic-atomic-case2: the TES's first storage buffer at slot 0 was overwritten
    // by the vertex buffer of attribute 0).  The kernel consumes the attributes
    // through the compute encoder instead.
    bool tess_render_stage = (ctx->mtl_funcs.ext_tess && mgl_tess_active_program(ctx) != NULL);
    for (int b = 0; b < MGL_VA_MAX_BINDINGS && !tess_render_stage; b++)
    {
        if (plan->binding_index[b] >= 0 && va->binding[b].buffer)
            [self vattribBindBuffer: va->binding[b].buffer offset: va->binding[b].offset index: plan->binding_index[b] encoder: enc];
    }

    for (int a = 0; a < MAX_ATTRIBS && !tess_render_stage; a++)
    {
        if (plan->fetch[a] != MGL_VA_FETCH_SHADOW)
            continue;
        /* the slot index is NOT the GL attribute location (the fp64 word slots of a
           dvec3/dvec4/array attribute are renumbered past the plain inputs), so the
           source binding comes from the OWNING attribute — va->fmt[a] would look up
           an unrelated GL location whose binding has no buffer and silently skip the
           bind (basic-inputL's double@7: slot 14 never got its converted stream) */
        int own = plan->owner_gl[a];
        if (own < 0 || own >= MAX_ATTRIBS) own = a;
        if (va->binding[va->fmt[own].binding].buffer == NULL)
            continue;
        /* dvec3/dvec4: both word slots share one converted stream, owned by
           the attribute's low slot (adjacent internal slots) */
        int owner = (plan->dbl[a] == 2) ? a - 1 : a;
        if (owner != a)
            continue;   /* bound when the low slot is reached */
        [self vattribBindShadow: a index: plan->shadow_index[a] encoder: enc];
    }

    if (plan->const_mask && !tess_render_stage)   /* slot 29 is the tess patch blob (MGL_TESS_TES_PATCH) */
        [self vattribBindConstants: enc];
}

@end
