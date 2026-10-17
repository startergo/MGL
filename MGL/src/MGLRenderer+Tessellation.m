/*
 * MGLRenderer+Tessellation.m — Metal side of the 'tess' domain.
 *
 * Executes a GL_PATCHES draw on GPUs without hardware tessellation (all Apple-silicon
 * GPUs: drawPatches is silently dropped). See mgl_ext_tess.h for the design:
 *
 *   compute:  VS kernel (SPIRV-Cross VERTEX_FOR_TESSELLATION) over (vertices, instances)
 *             -> optional gather kernel (indexed draws) -> TCS kernel (multi-patch)
 *   CPU:      tessellator (GL §11.3.4) from the read-back tessellation factors
 *   render:   the rewritten TES is the render pipeline's vertex function; it reads
 *             (tess coord, patch id) / control points / patch data from buffers and the
 *             tessellated primitives are drawn with drawIndexedPrimitives.
 *
 * The default-levels case (no TCS) needs no GPU readback: the factors are the
 * glPatchParameterfv defaults and control points pass through from the VS kernel.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#import "MGLRenderer_Private.h"
#import "MGLTessKernels.h"
#import "glm_context.h"
#import "mgl.h"
#import "mgl_ext.h"
#import "mgl_ext_tess.h"
#import "mgl_ext_rstate.h"
#import "mgl_ext_vattrib.h"
#import "mgl_ext_rstate.h"
#import "mgl_ext_buffers.h"
#import "mgl_ext_program.h"
#import "glcorearb.h"

#define MAX_ATTRIB_SLOTS 16

@implementation MGLRenderer (Tessellation)

/* ------------------------------------------------------------------ scratch state
 * Categories cannot add ivars; the scratch buffers are file-static and grow on demand.
 * They hold no GL objects, so lifetime is "until program exit".
 */
static id<MTLBuffer> tVSOut;        /* VS kernel output records            */
static id<MTLBuffer> tTcsCp;        /* gathered control-point stream       */
static id<MTLBuffer> tTcsOut;       /* TCS kernel control point records    */
static id<MTLBuffer> tTcsPatch;     /* TCS per-patch output records        */
static id<MTLBuffer> tFactors;      /* MTL*Half factors, one per patch     */
static id<MTLBuffer> tCoords;       /* (u,v,w,patch id) per tess vertex    */
static id<MTLBuffer> tIndices;      /* uint32 index list                   */
static id<MTLBuffer> tPatchData;    /* TES patch-in blob                   */
static id<MTLBuffer> tGatherIdx;    /* widened (+ base vertex) index list  */
static id<MTLBuffer> tVsParams;     /* base vertex / base instance         */
static id<MTLBuffer> tTcsParams;    /* spvIndirectParams                   */
static id<MTLBuffer> tTesParams;    /* cp-per-patch                        */
static id<MTLBuffer> tConstValues;  /* current generic attribute values    */
static id<MTLBuffer> tShadow;       /* CPU-converted attribute stream      */
static id<MTLBuffer> tSmallAttr;    /* small attribute buffers             */
static id<MTLBuffer> tGpuEmu;       /* allocs|totals|args|coords|indices   */
static id<MTLLibrary> tKernelLib;
static id<MTLComputePipelineState> tCountPso, tEmitPso, tArgsPso;

static NSMutableDictionary *tVsPsoCache;    /* VS kernel PSO, per program+VAO */
static NSMutableDictionary *tTcsPsoCache;   /* TCS kernel PSO, per program    */
static id<MTLComputePipelineState> tGatherPso;

/* grown-out scratch buffers stay alive for the lifetime of the process: a command
 * buffer still in flight may reference them, and Metal requires a resource to outlive
 * every encoder that used it */
static id<MTLBuffer> tRetired[16];
static int tNumRetired;

static bool tessBufferEnsure(id<MTLDevice> device, id<MTLBuffer> __strong *buf, size_t bytes)
{
    if (*buf && (*buf).length >= bytes)
        return true;
    /* metal requires length != 0 */
    if (bytes == 0)
        bytes = 16;
    id<MTLBuffer> nb = [device newBufferWithLength: bytes options: MTLResourceStorageModeShared];
    if (!nb)
        return false;
    if (*buf && tNumRetired < 16)
        tRetired[tNumRetired++] = *buf;
    *buf = nb;
    return true;
}

static id<MTLFunction> tessStageFunction(Program *program, int stage)
{
    Shader *sh = program ? program->shader_slots[stage] : NULL;
    if (!sh || !sh->mtl_data.function)
        return nil;
    return (__bridge id<MTLFunction>)(sh->mtl_data.function);
}

/* ------------------------------------------------------------------ VS attribute plan
 * The VS of a tessellation pipeline is a compute kernel: its attributes are fed
 * through a compute-pipeline stage input descriptor (MTLStepFunctionThreadPositionInGridX/Y),
 * so the "can Metal fetch the array directly" decisions of the vattrib domain are
 * re-made here from the same VAO state (see vattribGenerateVertexDescriptor).
 */
enum { TESS_FETCH_NONE = 0, TESS_FETCH_DIRECT, TESS_FETCH_SHADOW, TESS_FETCH_CONST };

typedef struct TessAttribSlot {
    int     loc;
    int     fetch;              /* TESS_FETCH_* */
    int     kind;               /* MGL_VA_KIND_* */
    int     mtl_format;         /* MTLVertexFormat for DIRECT */
    size_t  offset;             /* attribute offset inside its binding */
    int     binding;            /* GL vertex binding 0..15 */
    int     buffer_index;       /* Metal buffer index in the compute pipeline */
    int     shadow_buffer;      /* Metal buffer index of the converted stream */
    int     shadow_words;       /* words per element of the converted stream */
    int     binding_step;       /* divisor of the slot's GL binding */
} TessAttribSlot;

typedef struct TessVSPlan {
    TessAttribSlot slot[MAX_ATTRIB_SLOTS];
    int            count;
    int            binding_buffer_index[16];    /* GL binding -> Metal index (-1 none) */
    size_t         binding_stride[16];
    int            binding_step[16];            /* 0 per-vertex, >0 per-instance divisor, -1 constant */
    size_t         binding_offset[16];          /* extra byte offset (first / baseinstance) */
    int            num_buffers;                /* highest Metal index used + 1 */
    int            const_buffer;               /* Metal index of the generic-value buffer */
} TessVSPlan;

/* VS input locations and their base type kinds, from the VS stage reflection list */
static void tessVsInputKinds(GLMContext ctx, Program *program, bool has[MAX_ATTRIBS], int kind[MAX_ATTRIBS])
{
    SpirvResourceList *list = &program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT];

    memset(has, 0, MAX_ATTRIBS * sizeof(bool));
    memset(kind, 0, MAX_ATTRIBS * sizeof(int));

    if (getenv("MGL_TESS_DEBUG"))
        NSLog(@"MGL tess: VS stage inputs: %d", list->count);

    for (int i = 0; i < list->count; i++)
    {
        SpirvResource *r = &list->list[i];

        /* the location decoration lives on the variable */
        int loc = -1;
        uint32_t *ir = program->spirv[_VERTEX_SHADER].ir;
        size_t n = program->spirv[_VERTEX_SHADER].size;
        size_t k = 5;
        while (k + 3 < n)
        {
            uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;
            if (wc == 0) break;
            if (op == 71 /* OpDecorate */ && ir[k + 1] == r->_id && ir[k + 2] == 30 /* Location */)
            {
                loc = (int)ir[k + 3];
                break;
            }
            k += wc;
        }
        if (loc < 0 || loc >= MAX_ATTRIBS)
        {
            if (getenv("MGL_TESS_DEBUG"))
                NSLog(@"MGL tess: VS input %@ has no location (id %u)", [NSString stringWithUTF8String:r->name ? r->name : "?"], r->_id);
            continue;
        }

        has[loc] = true;

        /* resolve the base type: OpTypePointer -> (OpTypeVector)? -> OpTypeFloat / OpTypeInt */
        uint32_t type_id = r->base_type_id;
        int vec = 1;
        for (int pass = 0; pass < 2 && type_id; pass++)
        {
            k = 5;
            while (k + 3 < n)
            {
                uint32_t wc = ir[k] >> 16, op = ir[k] & 0xffff;
                if (wc == 0) break;
                if (ir[k + 1] == type_id)
                {
                    if (op == 23 /* OpTypeVector */)
                    {
                        type_id = ir[k + 2];
                        vec = ir[k + 3];
                        break;
                    }
                    if (op == 32 /* OpTypePointer */)
                    {
                        type_id = ir[k + 3];
                        break;
                    }
                    if (op == 4 /* OpTypeFloat */)
                    {
                        kind[loc] = ir[k + 3] == 64 ? MGL_VA_KIND_DOUBLE : MGL_VA_KIND_FLOAT;
                        type_id = 0;
                        break;
                    }
                    if (op == 5 /* OpTypeInt */)
                    {
                        kind[loc] = ir[k + 3] ? MGL_VA_KIND_INT : MGL_VA_KIND_UINT;
                        type_id = 0;
                        break;
                    }
                    break;
                }
                k += wc;
            }
        }
        (void)vec;
    }
}

/* forward declarations (helpers used by the draw path) */
static bool tessReadIndices(GLMContext ctx, GLenum type, size_t offset, GLsizei count, GLint basevertex,
                            uint32_t *dst, uint32_t *minv, uint32_t *maxv);
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count);
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count);
static uint16_t tessFloatToHalf(float f);
static float tessHalfToFloat(uint16_t h);
static bool tessKernels(GLMContext ctx);

/* format decision of the vattrib domain (MGLRenderer+VertexAttrib.m) */
extern MTLVertexFormat mglVertexFormatForAttrib(GLenum type, GLint size, bool normalized, bool integer);

static bool tessSignedInt(GLenum t)   { return t == GL_BYTE || t == GL_SHORT || t == GL_INT; }
static bool tessUnsignedInt(GLenum t) { return t == GL_UNSIGNED_BYTE || t == GL_UNSIGNED_SHORT || t == GL_UNSIGNED_INT; }

static bool tessBuildVSPlan(GLMContext ctx, Program *program, TessVSPlan *plan,
                            GLint first, GLint baseinstance)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    bool has[MAX_ATTRIBS];
    int kind[MAX_ATTRIBS];
    int next_index;

    memset(plan, 0, sizeof *plan);
    for (int i = 0; i < 16; i++)
        plan->binding_buffer_index[i] = -1;
    plan->const_buffer = -1;
    if (!va)
        return false;

    tessVsInputKinds(ctx, program, has, kind);

    /* uniform/storage buffers occupy 0..k-1 of the compute pipeline; attribute
       bindings continue after them; the VS kernel specials live at 30/31 */
    {
        int k = program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_UNIFORM_BUFFER].count +
                program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STORAGE_BUFFER].count +
                program->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER].count;
        next_index = k > 8 ? k : 8;
    }

    for (int loc = 0; loc < MAX_ATTRIB_SLOTS; loc++)
    {
        TessAttribSlot *s = &plan->slot[plan->count];
        MGLVertexAttribFormat *f;
        MGLVertexBinding *b;
        MTLVertexFormat format = MTLVertexFormatInvalid;
        bool aligned;
        GLsizei elem;

        if (!has[loc])
            continue;
        f = &va->fmt[loc];
        b = &va->binding[f->binding];

        s->loc = loc;
        s->kind = kind[loc];
        s->binding_step = b->divisor;

        if (!(ctx->state.vao->enabled_attribs & (1u << loc)) || b->buffer == NULL)
        {
            /* disabled array: the current generic value is a constant */
            if (plan->const_buffer < 0)
            {
                plan->const_buffer = next_index++;
                if (next_index >= MGL_TESS_VS_PARAMS)
                {
                    NSLog(@"MGL ERROR: tess: vertex compute pipeline ran out of buffer indices");
                    return false;
                }
            }
            s->fetch = TESS_FETCH_CONST;
            plan->count++;
            continue;
        }

        elem = mgl_va_element_size(f->type, f->size);
        aligned = (b->stride % 4) == 0 && (b->offset % 4) == 0 && (f->relativeoffset % 4) == 0 &&
                  (b->stride == 0 || f->relativeoffset + elem <= (GLuint)b->stride);

        if (aligned && !f->is_long)
        {
            if (s->kind == MGL_VA_KIND_FLOAT)
            {
                bool unnormalized_int = !f->normalized && (tessSignedInt(f->type) || tessUnsignedInt(f->type));
                if (!f->integer && !unnormalized_int)
                    format = mglVertexFormatForAttrib(f->type, f->size, f->normalized, false);
            }
            else if (s->kind != MGL_VA_KIND_DOUBLE)
            {
                bool sign_ok = (s->kind == MGL_VA_KIND_INT) ? tessSignedInt(f->type) : tessUnsignedInt(f->type);
                if (sign_ok && (f->integer || !f->normalized) && f->size != GL_BGRA)
                    format = mglVertexFormatForAttrib(f->type, f->size, false, true);
            }
        }

        if (format != MTLVertexFormatInvalid)
        {
            s->fetch = TESS_FETCH_DIRECT;
            s->mtl_format = (int)format;
        }
        else
        {
            s->fetch = TESS_FETCH_SHADOW;
            s->shadow_words = 4;     /* float4 / int4 / uint4 converted stream */
        }

        s->binding = f->binding;
        s->offset = (size_t)b->offset + (size_t)f->relativeoffset;

        if (plan->binding_buffer_index[s->binding] < 0)
        {
            plan->binding_buffer_index[s->binding] = next_index++;
            plan->binding_stride[s->binding] = b->stride ? (size_t)b->stride : (size_t)elem;
            plan->binding_step[s->binding] = b->divisor;
        }
        if (s->fetch == TESS_FETCH_SHADOW)
            s->shadow_buffer = next_index++;

        if (next_index >= MGL_TESS_VS_PARAMS)
        {
            NSLog(@"MGL ERROR: tess: vertex compute pipeline ran out of buffer indices");
            return false;
        }
        plan->count++;
    }

    /* fetch offsets: per-vertex bindings start at 'first', per-instance at 'baseinstance'
       (per GL, per-instance attribute indexing includes the base instance) */
    for (int b = 0; b < 16; b++)
    {
        if (plan->binding_buffer_index[b] < 0)
            continue;
        if (plan->binding_step[b] > 0)
            plan->binding_offset[b] = (size_t)baseinstance * plan->binding_stride[b];
        else
            plan->binding_offset[b] = (size_t)first * plan->binding_stride[b];
    }
    plan->num_buffers = next_index;
    return true;
}


/* ------------------------------------------------------------------ PSO caches */

/* the compute PSO of the VS kernel depends on the program (kernel function) and on the
   current vertex-array layout; the key mirrors xfbPipelineKey */
static NSString *tessVSKey(GLMContext ctx, Program *program, TessVSPlan *plan)
{
    NSMutableString *key = [NSMutableString stringWithFormat: @"%p-", (void *)program];
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        MGLVertexAttribFormat *f = &va->fmt[s->loc];
        MGLVertexBinding *b = &va->binding[f->binding];
        [key appendFormat: @"%d:%d:%x:%d:%u:%d:%u;", s->loc, s->fetch, f->type, f->size,
                         f->normalized, b->stride, b->divisor];
    }
    return key;
}

static id<MTLComputePipelineState> tessVsPSO(GLMContext ctx, id<MTLDevice> device, Program *program, TessVSPlan *plan)
{
    id<MTLFunction> fn = tessStageFunction(program, _VERTEX_SHADER);
    NSString *key;
    id<MTLComputePipelineState> pso;
    MTLComputePipelineDescriptor *desc;
    MTLStageInputOutputDescriptor *sd;

    if (!fn || fn.functionType != MTLFunctionTypeKernel)
    {
        NSLog(@"MGL ERROR: tess VS stage function is not a kernel (fn=%@ type=%ld)",
              fn, (long)(fn ? fn.functionType : -1));
        return nil;
    }

    key = tessVSKey(ctx, program, plan);
    if (tVsPsoCache)
    {
        pso = tVsPsoCache[key];
        if (pso)
            return pso;
    }

    desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;

    sd = [MTLStageInputOutputDescriptor new];
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        int bidx = s->binding;

        if (s->fetch == TESS_FETCH_CONST)
        {
            /* current generic value: 16-byte slot per location, constant step */
            sd.attributes[s->loc].format = (s->kind == MGL_VA_KIND_INT) ? MTLAttributeFormatInt4 :
                                           (s->kind == MGL_VA_KIND_UINT) ? MTLAttributeFormatUInt4 :
                                           MTLAttributeFormatFloat4;
            sd.attributes[s->loc].offset = (NSUInteger)(s->loc * 16);
            sd.attributes[s->loc].bufferIndex = plan->const_buffer;
            continue;
        }
        if (s->fetch == TESS_FETCH_SHADOW)
        {
            sd.attributes[s->loc].format = (s->kind == MGL_VA_KIND_INT) ? MTLAttributeFormatInt4 :
                                           (s->kind == MGL_VA_KIND_UINT) ? MTLAttributeFormatUInt4 :
                                           MTLAttributeFormatFloat4;
            sd.attributes[s->loc].offset = 0;
            sd.attributes[s->loc].bufferIndex = s->shadow_buffer;
            continue;
        }

        sd.attributes[s->loc].format = (MTLAttributeFormat)s->mtl_format;
        sd.attributes[s->loc].offset = s->offset;
        sd.attributes[s->loc].bufferIndex = plan->binding_buffer_index[bidx];
    }

    for (int b = 0; b < 16; b++)
    {
        int idx = plan->binding_buffer_index[b];
        if (idx < 0)
            continue;
        sd.layouts[idx].stride = plan->binding_stride[b];
        if (plan->binding_step[b] > 0)
        {
            sd.layouts[idx].stepFunction = MTLStepFunctionThreadPositionInGridY;
            sd.layouts[idx].stepRate = (NSUInteger)plan->binding_step[b];
        }
        else
        {
            sd.layouts[idx].stepFunction = MTLStepFunctionThreadPositionInGridX;
            sd.layouts[idx].stepRate = 1;
        }
    }

    /* constant + converted streams */
    if (plan->const_buffer >= 0)
    {
        sd.layouts[plan->const_buffer].stride = 16;
        sd.layouts[plan->const_buffer].stepFunction = MTLStepFunctionConstant;
        sd.layouts[plan->const_buffer].stepRate = 0;
    }
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        if (s->fetch != TESS_FETCH_SHADOW)
            continue;
        sd.layouts[s->shadow_buffer].stride = 16;
        if (s->binding_step > 0)
        {
            sd.layouts[s->shadow_buffer].stepFunction = MTLStepFunctionThreadPositionInGridY;
            sd.layouts[s->shadow_buffer].stepRate = (NSUInteger)s->binding_step;
        }
        else
        {
            sd.layouts[s->shadow_buffer].stepFunction = MTLStepFunctionThreadPositionInGridX;
            sd.layouts[s->shadow_buffer].stepRate = 1;
        }
    }
    desc.stageInputDescriptor = sd;

    if (getenv("MGL_TESS_DEBUG"))
    {
        NSMutableString *dbg = [NSMutableString stringWithString: @"MGL tess: VS stage-input descriptor:"];
        for (int i = 0; i < MAX_ATTRIB_SLOTS; i++)
        {
            MTLVertexAttributeDescriptor *a = sd.attributes[i];
            if (a.format != MTLAttributeFormatInvalid)
                [dbg appendFormat: @" attr[%d]{fmt=%ld,off=%lu,buf=%lu}", i, (long)a.format,
                                    (unsigned long)a.offset, (unsigned long)a.bufferIndex];
        }
        for (int i = 0; i < 31; i++)
        {
            MTLBufferLayoutDescriptor *l = sd.layouts[i];
            if (l.stride)
                [dbg appendFormat: @" layout[%d]{stride=%lu,fn=%ld,rate=%lu}", i, (unsigned long)l.stride,
                                    (long)l.stepFunction, (unsigned long)l.stepRate];
        }
        NSLog(@"%@", dbg);
    }

    NSError *err = nil;
    pso = [device newComputePipelineStateWithDescriptor: desc
                                                      options: MTLPipelineOptionNone
                                                   reflection: nil
                                                        error: &err];
    if (!pso)
    {
        NSLog(@"MGL ERROR: tess vertex kernel PSO failed: %@", err.localizedDescription);
        return nil;
    }
    if (!tVsPsoCache)
        tVsPsoCache = [[NSMutableDictionary alloc] init];
    tVsPsoCache[key] = pso;
    return pso;
}

static id<MTLComputePipelineState> tessTcsPSO(GLMContext ctx, id<MTLDevice> device, Program *program)
{
    id<MTLFunction> fn = tessStageFunction(program, _TESS_CONTROL_SHADER);
    id<MTLComputePipelineState> pso;

    if (!fn || fn.functionType != MTLFunctionTypeKernel)
        return nil;
    if (tTcsPsoCache)
    {
        pso = tTcsPsoCache[[NSString stringWithFormat: @"%p", (void *)program]];
        if (pso)
            return pso;
    }

    NSError *err = nil;
    pso = [device newComputePipelineStateWithFunction: fn error: &err];
    if (!pso)
    {
        NSLog(@"MGL ERROR: tess control kernel PSO failed: %@", err.localizedDescription);
        return nil;
    }
    if (!tTcsPsoCache)
        tTcsPsoCache = [[NSMutableDictionary alloc] init];
    tTcsPsoCache[[NSString stringWithFormat: @"%p", (void *)program]] = pso;
    return pso;
}

/* tiny kernel used to expand an indexed draw's vertex list into the linear
   control-point stream the TCS consumes */
static id<MTLComputePipelineState> tessGatherPSO(id<MTLDevice> device)
{
    static NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void mgl_tess_gather(device const uint* src [[buffer(0)]],\n"
        "                            device const uint* idx [[buffer(1)]],\n"
        "                            device uint* dst [[buffer(2)]],\n"
        "                            constant uint& words [[buffer(3)]],\n"
        "                            constant uint& base  [[buffer(4)]],\n"
        "                            constant uint& total [[buffer(5)]],\n"
        "                            uint3 gid [[thread_position_in_grid]]) {\n"
        "    if (gid.x >= total) return;\n"
        "    uint v = idx[gid.x] + base;\n"
        "    for (uint w = 0; w < words; w++)\n"
        "        dst[gid.x * words + w] = src[v * words + w];\n"
        "}\n";

    if (tGatherPso)
        return tGatherPso;
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource: src options: nil error: &err];
    if (!lib)
    {
        NSLog(@"MGL ERROR: tess gather library failed: %@", err.localizedDescription);
        return nil;
    }
    tGatherPso = [device newComputePipelineStateWithFunction: [lib newFunctionWithName: @"mgl_tess_gather"] error: &err];
    if (!tGatherPso)
        NSLog(@"MGL ERROR: tess gather PSO failed: %@", err.localizedDescription);
    return tGatherPso;
}

/* ------------------------------------------------------------------ per-stage resource binding
 * The kernels read the uniform/storage buffers and textures of their own stage: bind
 * them at the indices SPIRV-Cross assigned (mirrors mapGLBuffersToMTLBufferMap /
 * bindTexturesToComputeEncoder but for an arbitrary stage).
 */
static void tessBindStageUniforms(GLMContext ctx, id<MTLComputeCommandEncoder> enc, Program *program, int stage)
{
    static const struct { int spvc; int glt; } types[4] = {
        { SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,    GL_UNIFORM_BUFFER },
        { SPVC_RESOURCE_TYPE_STORAGE_BUFFER,    GL_SHADER_STORAGE_BUFFER },
        { SPVC_RESOURCE_TYPE_ATOMIC_COUNTER,    GL_ATOMIC_COUNTER_BUFFER },
    };

    if (!program)
        return;

    for (int t = 0; t < 3; t++)
    {
        SpirvResourceList *list = &program->spirv_resources_list[stage][types[t].spvc];
        int index = 0;

        for (int r = 0; r < list->count && r < 24; r++)
        {
            Buffer *buf = ctx->state.buffer_base[types[t].glt].buffers[list->list[r].binding].buf;
            if (!buf || !buf->data.buffer_data)
            {
                index++;
                continue;
            }
            [enc setBuffer: (__bridge id<MTLBuffer>)buf->data.mtl_data
                     offset: (NSUInteger)ctx->state.buffer_base[types[t].glt].buffers[list->list[r].binding].offset
                    atIndex: (NSUInteger)index];
            index++;
        }
    }
}

static void tessBindStageTextures(GLMContext ctx, id<MTLComputeCommandEncoder> enc, Program *program, int stage)
{
    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    SpirvResourceList *list = &program->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_SAMPLED_IMAGE];

    if (!program)
        return;

    for (int r = 0; r < list->count && r < 16; r++)
    {
        Texture *tex = STATE(active_textures[list->list[r].binding]);
        id<MTLTexture> mtl = [self2 texparamTextureForTex: tex base: nil];

        [enc setTexture: mtl atIndex: (NSUInteger)r];
    }
}

/* ------------------------------------------------------------------ the draw */

- (bool) extTessActive
{
    return mgl_tess_active_program(ctx) != NULL;
}

- (bool) extTessDrawPatches: (bool) indexed type: (GLenum) type offset: (size_t) offset
                      first: (GLint) first count: (GLsizei) count instances: (GLsizei) instances
                basevertex: (GLint) basevertex baseinstance: (GLint) baseinstance
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    Program *prog_vs = ctx->state.program, *prog_tcs = NULL, *prog_tes = NULL;
    struct MGLExtState_tess *S = (struct MGLExtState_tess *)ctx->state.ext_tess;
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    GLsizei patch_verts_in = mgl_tess_patch_vertices_in(ctx, tp);
    GLsizei patches, total_patches;
    size_t cp_record = tp->cp_record_size;
    size_t patch_record = tp->tcs_patch_out_size ? tp->tcs_patch_out_size : 16;
    /* the control point record stride: the FULL TCS/VS output record (the TES may
     * only read a subset of it; its offsets were remapped at link time) */
    size_t cp_stride = tp->cp_record_full_size ? tp->cp_record_full_size : cp_record;
    uint32_t minv = 0, maxv = 0;
    TessVSPlan plan;
    id<MTLCommandBuffer> compute_cb;
    id<MTLComputeCommandEncoder> cenc;
    id<MTLComputePipelineState> vs_pso, tcs_pso = nil, gather_pso = nil;
    bool need_readback = tp->has_tcs != 0;
    bool use_cpu_reference = getenv("MGL_TESS_CPU") != NULL;
    bool use_hardware = (tp->domain != MGL_TESS_DOMAIN_ISOLINES) && !tp->point_mode && !use_cpu_reference;
    bool gpu_emulated = false;
    GLuint vcount;
    GLsizei vfirst = first;
    id<MTLBuffer> cp_source;

    if (!tp || !va || cp_stride == 0)
        return false;

    if (patch_verts_in < 1)
        patch_verts_in = 1;
    patches = count / patch_verts_in;             /* trailing incomplete patch is dropped */
    if (patches <= 0)
        return true;
    total_patches = patches * instances;

    if (ctx->state.program_pipeline)
    {
        prog_tes = ctx->state.program_pipeline->stage_programs[_TESS_EVALUATION_SHADER] ?: ctx->state.program;
        prog_tcs = ctx->state.program_pipeline->stage_programs[_TESS_CONTROL_SHADER];
        prog_vs = ctx->state.program_pipeline->stage_programs[_VERTEX_SHADER] ?: prog_tes;
    }
    else
    {
        prog_tcs = prog_tes = ctx->state.program;
    }

    /* ---- 1. vertex stage: compute kernel over (vertices, instances) ---- */
    if (!tessBuildVSPlan(ctx, prog_vs, &plan, vfirst, baseinstance))
        return false;
    vs_pso = tessVsPSO(ctx, _device, prog_vs, &plan);
    if (!vs_pso)
        return false;

    /* indexed draws: read the indices (CPU visible storage), bake the base vertex
       into a widened list and gather the referenced vertex records with a kernel */
    if (indexed)
    {
        if (!tessBufferEnsure(_device, &tGatherIdx, (size_t)count * instances * sizeof(uint32_t) + 16))
            return false;
        if (!tessReadIndices(ctx, type, offset, count, basevertex,
                             (uint32_t *)tGatherIdx.contents, &minv, &maxv))
            return false;
        gather_pso = tessGatherPSO(_device);
        if (!gather_pso)
            return false;
        for (GLsizei i = 0; i < count * instances; i++)
            ((uint32_t *)tGatherIdx.contents)[i] -= minv;   /* index into the [minv..maxv] range */
        vfirst = (GLsizei)minv;
        vcount = maxv - minv + 1;
    }
    else
    {
        vcount = (GLuint)count;
    }

    /* ---- 2. compute phase (VS kernel, optional gather, optional TCS kernel) ---- */
    [self endRenderEncoding];

    bool debug_wait = getenv("MGL_TESS_DEBUG") && use_hardware;
    if ((need_readback && !use_hardware) || debug_wait)
        compute_cb = [_commandQueue commandBuffer];
    else
    {
        /* kernels and the render pass share one command buffer: Metal executes encoders
           in order, so the TCS output is complete before the render pass reads it */
        if (!_currentCommandBuffer || _currentCommandBuffer.status >= MTLCommandBufferStatusCommitted)
            _currentCommandBuffer = [_commandQueue commandBuffer];
        compute_cb = _currentCommandBuffer;
    }

    cenc = [compute_cb computeCommandEncoder];

    if (!tessBufferEnsure(_device, &tVSOut, (size_t)vcount * instances * cp_record + 16) ||
        !tessBufferEnsure(_device, &tVsParams, 16))
    {
        [cenc endEncoding];
        return false;
    }

    /* VS kernel */
    {
        uint32_t params[4] = { (uint32_t)vfirst, (uint32_t)vcount, (uint32_t)instances,
                               (uint32_t)baseinstance };

        if (getenv("MGL_TESS_DEBUG"))
            NSLog(@"MGL tess: VS kernel dispatch %u x %d threads", vcount, instances);
        [cenc setComputePipelineState: vs_pso];
        tessBindStageUniforms(ctx, cenc, prog_vs, _VERTEX_SHADER);
        tessBindStageTextures(ctx, cenc, prog_vs, _VERTEX_SHADER);
        if (!tessBindVSAttributes(ctx, cenc, &plan, (GLuint)vcount))
        {
            [cenc endEncoding];
            return false;
        }
        [cenc setBuffer: tVSOut offset: 0 atIndex: MGL_TESS_VS_OUT_BUF];
        memcpy(tVsParams.contents, params, sizeof params);
        [cenc setBuffer: tVsParams offset: 0 atIndex: MGL_TESS_VS_PARAMS];
        [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)vcount, (NSUInteger)instances, 1)
              threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];
    }

    cp_source = tVSOut;

    /* gather kernel (indexed): linear control-point stream per index occurrence */
    if (indexed)
    {
        uint32_t gparams[3] = { (uint32_t)(cp_stride / 4), 0, (uint32_t)(count * instances) };
        NSUInteger groups = ((NSUInteger)(count * instances) + 63) / 64;

        if (!tessBufferEnsure(_device, &tTcsCp, (size_t)count * instances * cp_stride + 16))
        {
            [cenc endEncoding];
            return false;
        }
        [cenc setComputePipelineState: gather_pso];
        [cenc setBuffer: tVSOut offset: 0 atIndex: 0];
        [cenc setBuffer: tGatherIdx offset: 0 atIndex: 1];
        [cenc setBuffer: tTcsCp offset: 0 atIndex: 2];
        [cenc setBytes: &gparams[0] length: sizeof(uint32_t) atIndex: 3];
        [cenc setBytes: &gparams[1] length: sizeof(uint32_t) atIndex: 4];
        [cenc setBytes: &gparams[2] length: sizeof(uint32_t) atIndex: 5];
        [cenc dispatchThreadgroups: MTLSizeMake(groups, 1, 1) threadsPerThreadgroup: MTLSizeMake(64, 1, 1)];

        cp_source = tTcsCp;
    }

    /* TCS kernel */
    if (tp->has_tcs)
    {
        uint32_t iparams[2] = { (uint32_t)patch_verts_in, (uint32_t)total_patches };
        NSUInteger threads = (NSUInteger)total_patches * (NSUInteger)tp->tcs_out_vertices;

        tcs_pso = tessTcsPSO(ctx, _device, prog_tcs);
        if (!tcs_pso)
        {
            [cenc endEncoding];
            return false;
        }
        if (!tessBufferEnsure(_device, &tTcsOut,
                              (size_t)total_patches * tp->tcs_out_vertices * cp_stride + 16) ||
            !tessBufferEnsure(_device, &tTcsPatch, (size_t)total_patches * patch_record + 16) ||
            !tessBufferEnsure(_device, &tFactors, (size_t)total_patches * 12 + 16) ||
            !tessBufferEnsure(_device, &tTcsParams, 16))
        {
            [cenc endEncoding];
            return false;
        }

        [cenc setComputePipelineState: tcs_pso];
        tessBindStageUniforms(ctx, cenc, prog_tcs, _TESS_CONTROL_SHADER);
        tessBindStageTextures(ctx, cenc, prog_tcs, _TESS_CONTROL_SHADER);
        [cenc setBuffer: cp_source offset: 0 atIndex: MGL_TESS_TCS_IN];
        [cenc setBuffer: tTcsOut offset: 0 atIndex: MGL_TESS_TCS_OUT];
        [cenc setBuffer: tTcsPatch offset: 0 atIndex: MGL_TESS_TCS_PATCH_OUT];
        [cenc setBuffer: tFactors offset: 0 atIndex: MGL_TESS_TCS_FACTORS];
        memcpy(tTcsParams.contents, iparams, sizeof iparams);
        [cenc setBuffer: tTcsParams offset: 0 atIndex: MGL_TESS_TCS_PARAMS];
        [cenc dispatchThreadgroups: MTLSizeMake(threads, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

        cp_source = tTcsOut;
    }
    else
    {
        /* no TCS: the default tessellation levels are CPU-known constants; the control
           points pass through from the vertex kernel output (patch p control point i
           lives at [instance * count + p * patch_verts_in + i] == [p * patch_verts_in + i]) */
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));

        for (int p = 0; p < total_patches; p++)
        {
            factors[p * 6 + 0] = rs->patch_default_outer[0];
            factors[p * 6 + 1] = rs->patch_default_outer[1];
            factors[p * 6 + 2] = rs->patch_default_outer[2];
            factors[p * 6 + 3] = rs->patch_default_outer[3];
            factors[p * 6 + 4] = rs->patch_default_inner[0];
            factors[p * 6 + 5] = rs->patch_default_inner[1];
        }
        mgl_tess_build_patch_blob(ctx, tp, NULL, 0, factors, (unsigned)total_patches);
        mgl_tess_generate(ctx, tp, factors, (unsigned)total_patches);
        free(factors);
    }

    [cenc endEncoding];

    if (debug_wait)
    {
        [compute_cb commit];
        [compute_cb waitUntilCompleted];
        if (compute_cb.error)
            NSLog(@"MGL tess: compute cb error %@", compute_cb.error);
        {
            uint16_t *hf = (uint16_t *)tFactors.contents;
            for (int p = 0; p < total_patches && p < 6; p++)
                NSLog(@"MGL tess: dbg factors[%d]=%d %d %d / %d", p, hf[p*6], hf[p*6+1], hf[p*6+2], hf[p*6+4]);
            float *cp = (float *)cp_source.contents;
            for (int i = 0; i < total_patches * (tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in) && i < 12; i++)
                NSLog(@"MGL tess: dbg cp[%d] col=(%g,%g,%g,%g) pos=(%g,%g,%g,%g)", i,
                      cp[i*8], cp[i*8+1], cp[i*8+2], cp[i*8+3], cp[i*8+4], cp[i*8+5], cp[i*8+6], cp[i*8+7]);
            {
                int n = indexed ? (int)vcount : count;
                float *vo = (float *)tVSOut.contents;
                for (int i = 0; i < n && i < 12; i++)
                    NSLog(@"MGL tess: dbg vsout[%d]=(%g,%g,%g,%g)", i, vo[i*8], vo[i*8+1], vo[i*8+2], vo[i*8+3]);
            }
            uint32_t *ip = (uint32_t *)tTcsParams.contents;
            NSLog(@"MGL tess: dbg tcs params = %u %u (total_patches %d)", ip[0], ip[1], total_patches);
        }
        _currentCommandBuffer = [_commandQueue commandBuffer];
    }

    /* ---- 3a. hardware tessellation path (triangles / quads): the TCS kernel wrote the
     * factors on the GPU; the render pass feeds them to drawPatches directly. No
     * readback, no CPU tessellation. ---- */
    if (use_hardware)
    {
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;

        if (!tp->has_tcs)
        {
            /* no TCS: the default levels are CPU-known constants; control points pass
               through from the vertex kernel output */
            if (!tessBufferEnsure(_device, &tFactors, (size_t)total_patches * 12 + 16) ||
                !tessBufferEnsure(_device, &tPatchData,
                                  (size_t)total_patches * tp->tes_patch_in_size + 16))
                return false;
            uint16_t *hf = (uint16_t *)tFactors.contents;
            for (int p = 0; p < total_patches; p++)
            {
                for (int i = 0; i < 4; i++)
                    hf[p * 6 + i] = tessFloatToHalf(rs->patch_default_outer[i]);
                for (int i = 0; i < 2; i++)
                    hf[p * 6 + 4 + i] = tessFloatToHalf(rs->patch_default_inner[i]);
            }

            /* patch data blob in the TES patch-in layout (zeros: undefined per GL) */
            {
                float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));
                for (int p = 0; p < total_patches; p++)
                {
                    factors[p * 6 + 0] = rs->patch_default_outer[0];
                    factors[p * 6 + 1] = rs->patch_default_outer[1];
                    factors[p * 6 + 2] = rs->patch_default_outer[2];
                    factors[p * 6 + 3] = rs->patch_default_outer[3];
                    factors[p * 6 + 4] = rs->patch_default_inner[0];
                    factors[p * 6 + 5] = rs->patch_default_inner[1];
                }
                mgl_tess_build_patch_blob(ctx, tp, NULL, 0, factors, (unsigned)total_patches);
                free(factors);
            }
            if (!tessBufferEnsure(_device, &tPatchData, (size_t)total_patches * tp->tes_patch_in_size + 16))
                return false;
            memcpy(tPatchData.contents, S->patch_data, (size_t)total_patches * tp->tes_patch_in_size);
        }

        /* ---- 4a. render: drawPatches with the post-tessellation TES ---- */
        {
            id<MTLRenderCommandEncoder> enc;
            GLsizei cp_count = tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in;
            id<MTLBuffer> patch_buf = tp->has_tcs ? tTcsPatch : tPatchData;

            if (![self processGLState: true])
                return false;
            enc = _currentRenderEncoder;
            if (!enc)
                return false;

            [enc setVertexBuffer: cp_source offset: 0 atIndex: MGL_TESS_VD_CP];
            [enc setVertexBuffer: patch_buf offset: 0 atIndex: MGL_TESS_VD_PATCH];
            [enc setVertexBuffer: tFactors offset: 0 atIndex: MGL_TESS_VD_FACTORS];
            [enc setTessellationFactorBuffer: tFactors offset: 0 instanceStride: 0];

            if (getenv("MGL_TESS_DEBUG"))
            {
                NSLog(@"MGL tess: drawPatches cp=%d patches=%d (hw) factors=%lu bytes cp=%lu bytes",
                      cp_count, total_patches, (unsigned long)tFactors.length, (unsigned long)cp_source.length);
            }

            /* One drawPatches per patch: a single drawPatches with patchCount = N (and
             * patchStart > 0) rasterises only patch 0 through the full MGL pipeline
             * (VS/TCS compute kernels -> render encoder) on this AGX driver, although
             * isolated Metal probes render every patch. Rebinding the control point /
             * patch data / factor buffers at each patch's offset works for every patch.
             * [[patch_id]] then reads 0, so gl_PrimitiveID in the TES is 0 for every
             * patch (documented gap). */
            for (int p = 0; p < total_patches; p++)
            {
                [enc setVertexBuffer: cp_source
                              offset: (size_t)p * (size_t)cp_count * cp_stride
                             atIndex: MGL_TESS_VD_CP];
                [enc setVertexBuffer: patch_buf
                              offset: (size_t)p * (tp->has_tcs ? patch_record : tp->tes_patch_in_size)
                             atIndex: MGL_TESS_VD_PATCH];
                [enc setTessellationFactorBuffer: tFactors
                                          offset: (size_t)p * 12
                                  instanceStride: 0];
                [enc drawPatches: (NSUInteger)cp_count
                      patchStart: 0
                       patchCount: 1
                 patchIndexBuffer: nil
           patchIndexBufferOffset: 0
                    instanceCount: 1
                     baseInstance: 0];
            }
            ctx->state.dirty_bits = DIRTY_ALL;
        }
        return true;
    }

    /* ---- 3b. emulated path (isolines / point_mode): GPU tessellator ----
     * count -> alloc -> emit -> indirect-args kernels run right after the TCS in the
     * same compute encoder; the plain-vertex TES then draws from the coordinate and
     * index buffers through an indirect draw. No CPU readback. (MGL_TESS_CPU=1 keeps
     * the CPU tessellator of tessellation.c as a reference path.) */
    if (!use_cpu_reference && tessKernels(ctx))
    {
        struct TessKParams { uint32_t patches, domain, spacing, point_mode, ccw; } kp;
        id<MTLComputeCommandEncoder> kenc;

        /* the VS/TCS encoder ended above; the tessellator kernels get their own encoder
           in the same command buffer (encoder order = execution order) */
        kenc = [compute_cb computeCommandEncoder];
        cenc = kenc;
        size_t coords_off = (((size_t)total_patches * 2 + 2 + 5 + 3) & ~(size_t)3) * 4;
        size_t coords_cap = (size_t)total_patches * (tp->domain == MGL_TESS_DOMAIN_QUADS ? 4225 :
                            tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 4160 : 2145);
        size_t indices_cap = (size_t)total_patches * (tp->point_mode ? 4225 :
                             tp->domain == MGL_TESS_DOMAIN_QUADS ? 24576 :
                             tp->domain == MGL_TESS_DOMAIN_ISOLINES ? 8192 : 12288);
        size_t need = coords_off + coords_cap * 16 + indices_cap * 4;

        if (need <= (size_t)(64u << 20) && tessBufferEnsure(_device, &tGpuEmu, need))
        {
            uint32_t *base = (uint32_t *)tGpuEmu.contents;
            kp.patches = (uint32_t)total_patches;
            kp.domain = (uint32_t)tp->domain;
            kp.spacing = (uint32_t)tp->spacing;
            kp.point_mode = (uint32_t)(tp->point_mode ? 1 : 0);
            kp.ccw = (uint32_t)(tp->vertex_order_ccw ? 1 : 0);
            base[total_patches * 2 + 0] = 0;    /* coord total */
            base[total_patches * 2 + 1] = 0;    /* index total */

            [cenc setComputePipelineState: tCountPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: tFactors offset: 0 atIndex: 1];
            [cenc setBuffer: tGpuEmu offset: 0 atIndex: 2];
            [cenc setBuffer: tGpuEmu offset: (size_t)(total_patches * 2) * 4 atIndex: 3];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [cenc setComputePipelineState: tEmitPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: tFactors offset: 0 atIndex: 1];
            [cenc setBuffer: tGpuEmu offset: 0 atIndex: 2];
            [cenc setBuffer: tGpuEmu offset: coords_off atIndex: 4];
            [cenc setBuffer: tGpuEmu offset: coords_off + coords_cap * 16 atIndex: 5];
            [cenc dispatchThreadgroups: MTLSizeMake((NSUInteger)total_patches, 1, 1)
                  threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            [cenc setComputePipelineState: tArgsPso];
            [cenc setBytes: &kp length: sizeof kp atIndex: 0];
            [cenc setBuffer: tGpuEmu offset: (size_t)(total_patches * 2) * 4 atIndex: 3];
            [cenc setBuffer: tGpuEmu offset: (size_t)(total_patches * 2 + 2) * 4 atIndex: 6];
            [cenc dispatchThreadgroups: MTLSizeMake(1, 1, 1) threadsPerThreadgroup: MTLSizeMake(1, 1, 1)];

            /* expose the GPU tessellated output (geometry / xfb consumers) */
            S->gpu_scratch = (__bridge void *)tGpuEmu;
            S->gpu_coords_offset = coords_off;
            S->gpu_indices_offset = coords_off + coords_cap * 16;
            S->gpu_args_offset = (size_t)(total_patches * 2 + 2) * 4;

            gpu_emulated = true;
        }
        else if (getenv("MGL_TESS_DEBUG"))
            NSLog(@"MGL tess: GPU tessellator scratch too large (%zu bytes), using CPU path", need);
    }
    if (gpu_emulated)
        [cenc endEncoding];     /* the tessellator encoder (kenc); the CPU path's encoder ended above */

    /* GPU emulated path: patch data blob + render with an indirect draw */
    if (gpu_emulated)
    {
        struct MGLExtState_rstate *rs = (struct MGLExtState_rstate *)ctx->state.ext_rstate;
        float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));
        struct MGLExtState_tess *SS = (struct MGLExtState_tess *)ctx->state.ext_tess;

        if (!tp->has_tcs)
        {
            for (int p = 0; p < total_patches; p++)
            {
                factors[p * 6 + 0] = rs->patch_default_outer[0];
                factors[p * 6 + 1] = rs->patch_default_outer[1];
                factors[p * 6 + 2] = rs->patch_default_outer[2];
                factors[p * 6 + 3] = rs->patch_default_outer[3];
                factors[p * 6 + 4] = rs->patch_default_inner[0];
                factors[p * 6 + 5] = rs->patch_default_inner[1];
            }
            mgl_tess_build_patch_blob(ctx, tp, NULL, 0, factors, (unsigned)total_patches);
        }
        free(factors);
        (void)SS;

        /* plain-vertex variant of the TES */
        if (!tp->plain_function)
        {
            char *msl = mgl_tess_plain_msl(ctx, tp);
            if (!getenv("MGL_TESS_NO_FIXUP"))
            {
                char *patched = mgl_rstate_patch_vertex_msl(msl, prog_tes->spirv[_TESS_EVALUATION_SHADER].entry_point);
                if (patched) { free(msl); msl = patched; }
            }
            if (msl && ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
            {
                void *lib = NULL, *fn = NULL;
                char *err = NULL;
                if (ctx->mtl_funcs.ext_program->compile_msl(ctx, msl,
                        prog_tes->spirv[_TESS_EVALUATION_SHADER].entry_point, &lib, &fn, &err))
                {
                    tp->plain_function = fn;
                    tp->plain_library = lib;
                    ctx->state.dirty_bits |= DIRTY_PROGRAM;
                }
                else
                    NSLog(@"MGL ERROR: tess: plain TES variant failed to compile: %s", err ? err : "?");
                free(err);
            }
            free(msl);
        }
        if (!tessBufferEnsure(_device, &tPatchData, (size_t)total_patches * tp->tes_patch_in_size + 16))
            return false;
        if (tp->has_tcs)
        {
            /* TCS patch records are read back through shared storage to build the blob
             * in the TES layout (the render pass waits for the compute phase anyway) */
            [compute_cb commit];
            [compute_cb waitUntilCompleted];
            if (compute_cb.error)
                return false;
            mgl_tess_build_patch_blob(ctx, tp, (const unsigned char *)tTcsPatch.contents,
                                      patch_record, NULL, (unsigned)total_patches);
            _currentCommandBuffer = [_commandQueue commandBuffer];   /* fresh cb for the render */
        }
        memcpy(tPatchData.contents, S->patch_data, (size_t)total_patches * tp->tes_patch_in_size);

        {
            id<MTLRenderCommandEncoder> enc;

            if (![self processGLState: true])
                return false;
            enc = _currentRenderEncoder;
            if (!enc)
                return false;

            [enc setVertexBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch offset: S->gpu_coords_offset atIndex: MGL_TESS_TES_COORDS];
            [enc setVertexBuffer: cp_source offset: 0 atIndex: MGL_TESS_TES_CP];
            [enc setVertexBuffer: tPatchData offset: 0 atIndex: MGL_TESS_TES_PATCH];
            {
                uint32_t tes_params[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), 0, 0, 0 };
                if (!tessBufferEnsure(_device, &tTesParams, 16))
                    return false;
                memcpy(tTesParams.contents, tes_params, sizeof tes_params);
                [enc setVertexBuffer: tTesParams offset: 0 atIndex: MGL_TESS_TES_PARAMS];
            }

            if (tp->point_mode)
                [enc drawPrimitives: MTLPrimitiveTypePoint
                       indirectBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                 indirectBufferOffset: S->gpu_args_offset];
            else
                [enc drawIndexedPrimitives: (tp->domain == MGL_TESS_DOMAIN_ISOLINES ?
                                            MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle)
                                  indexType: MTLIndexTypeUInt32
                                indexBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                          indexBufferOffset: S->gpu_indices_offset
                             indirectBuffer: (__bridge id<MTLBuffer>)S->gpu_scratch
                       indirectBufferOffset: S->gpu_args_offset];

            ctx->state.dirty_bits = DIRTY_ALL;
            return true;
        }
    }

    /* CPU reference path (MGL_TESS_CPU=1): read the factors back and tessellate on the CPU */
    {
    if (getenv("MGL_TESS_DEBUG") && need_readback)
    {
        [compute_cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
            if (b.error) NSLog(@"MGL tess: compute cb error: %@", b.error);
        }];
    }
    if (need_readback)
    {
        [compute_cb commit];
        [compute_cb waitUntilCompleted];
        if (compute_cb.error)
            return false;
        _currentCommandBuffer = [_commandQueue commandBuffer];   /* fresh cb for the render */

        {
            const uint16_t *hf = (const uint16_t *)tFactors.contents;
            float *factors = (float *)calloc((size_t)total_patches, 6 * sizeof(float));

            for (int p = 0; p < total_patches; p++)
            {
                const uint16_t *f = hf + p * 6;    /* MTL*FactorsHalf = 12 bytes per patch */
                factors[p * 6 + 0] = tessHalfToFloat(f[0]);
                factors[p * 6 + 1] = tessHalfToFloat(f[1]);
                factors[p * 6 + 2] = tessHalfToFloat(f[2]);
                factors[p * 6 + 3] = tessHalfToFloat(f[3]);
                factors[p * 6 + 4] = tessHalfToFloat(f[4]);
                factors[p * 6 + 5] = tessHalfToFloat(f[5]);
            }
            mgl_tess_build_patch_blob(ctx, tp, (const unsigned char *)tTcsPatch.contents,
                                      patch_record, factors, (unsigned)total_patches);
            mgl_tess_generate(ctx, tp, factors, (unsigned)total_patches);
            if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL tess: cpu path patch 0 factors %g %g %g %g / %g %g -> %d verts %d idx\n",
                        factors[0], factors[1], factors[2], factors[3], factors[4], factors[5],
                        S->tess_vertex_count, S->tess_index_count);
            free(factors);
        }
    }

    if (!S->tess_index_count)
        return true;                                /* degenerate: nothing to rasterize */

    /* ---- 4. render phase (CPU reference path): TES is the pipeline's vertex function ---- */
    {
        size_t coord_bytes = (size_t)S->tess_vertex_count * 16;
        size_t index_bytes = (size_t)S->tess_index_count * sizeof(uint32_t);
        uint32_t tes_params[4] = { (uint32_t)(tp->has_tcs ? tp->tcs_out_vertices : patch_verts_in), 0, 0, 0 };
        id<MTLRenderCommandEncoder> enc;
        MTLPrimitiveType prim = tp->point_mode ? MTLPrimitiveTypePoint :
                                (tp->domain == MGL_TESS_DOMAIN_ISOLINES && !tp->point_mode) ?
                                    MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;

        if (!tessBufferEnsure(_device, &tCoords, coord_bytes + 16) ||
            !tessBufferEnsure(_device, &tIndices, index_bytes + 16) ||
            !tessBufferEnsure(_device, &tPatchData, (size_t)S->patch_count * tp->tes_patch_in_size + 16) ||
            !tessBufferEnsure(_device, &tTesParams, 16))
            return false;

        memcpy(tCoords.contents, S->coord_buf, coord_bytes);
        memcpy(tIndices.contents, S->index_buf, index_bytes);
        memcpy(tPatchData.contents, S->patch_data, (size_t)S->patch_count * tp->tes_patch_in_size);
        memcpy(tTesParams.contents, tes_params, sizeof tes_params);

        /* processGLState builds the render encoder and the (TES-based) pipeline */
        if (![self processGLState: true])
            return false;
        enc = _currentRenderEncoder;
        if (!enc)
            return false;

        [enc setVertexBuffer: tCoords offset: 0 atIndex: MGL_TESS_TES_COORDS];
        [enc setVertexBuffer: cp_source offset: 0 atIndex: MGL_TESS_TES_CP];
        [enc setVertexBuffer: tPatchData offset: 0 atIndex: MGL_TESS_TES_PATCH];
        [enc setVertexBuffer: tTesParams offset: 0 atIndex: MGL_TESS_TES_PARAMS];

        if (getenv("MGL_TESS_DEBUG"))
            NSLog(@"MGL tess: CPU reference path drawing %d idx", S->tess_index_count);
        if (tp->point_mode)
            [enc drawPrimitives: MTLPrimitiveTypePoint vertexStart: 0
                     vertexCount: (NSUInteger)S->tess_vertex_count];
        else
            [enc drawIndexedPrimitives: prim indexCount: (NSUInteger)S->tess_index_count
                             indexType: MTLIndexTypeUInt32 indexBuffer: tIndices indexBufferOffset: 0];

        ctx->state.dirty_bits = DIRTY_ALL;
    }
    }   /* CPU reference path */

    return true;
}

/* forward declarations (used by the draw above) */
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count);
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count);
static bool tessReadIndices(GLMContext ctx, GLenum type, size_t offset, GLsizei count, GLint basevertex,
                            uint32_t *dst, uint32_t *minv, uint32_t *maxv);
static bool tessDrawPatchesHook(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                                GLint first, GLsizei count, GLsizei instances,
                                GLint basevertex, GLint baseinstance);

/* bind the attribute/const/shadow buffers of the VS kernel compute pipeline */
static bool tessBindVSAttributes(GLMContext ctx, id<MTLComputeCommandEncoder> enc, TessVSPlan *plan, GLuint vertex_count)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);

    /* per-binding source buffers */
    for (int b = 0; b < 16; b++)
    {
        int idx = plan->binding_buffer_index[b];
        Buffer *buf = idx < 0 ? NULL : va->binding[b].buffer;

        if (idx < 0)
            continue;
        if (getenv("MGL_TESS_DEBUG"))
            NSLog(@"MGL tess: VS binding %d -> mtl index %d (buf=%p mtl=%p fetch...)", b, idx,
                  (void *)buf, buf ? buf->data.mtl_data : NULL);
        if (!buf || !buf->data.buffer_data)
        {
            [enc setBuffer: nil offset: 0 atIndex: (NSUInteger)idx];
            continue;
        }
        {
            MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
            [self2 bindMTLBuffer: buf];
            if (getenv("MGL_TESS_DEBUG"))
                NSLog(@"MGL tess: VS binding %d idx %d: mtl_data=%p", b, idx, buf->data.mtl_data);
        }
        if (buf->data.mtl_data)
        {
            id<MTLBuffer> mtl = (__bridge id<MTLBuffer>)(buf->data.mtl_data);
            [enc setBuffer: mtl offset: plan->binding_offset[b] atIndex: (NSUInteger)idx];
        }
        else
        {
            /* small buffer without a Metal object: compute stage-in needs a real buffer */
            if (!tessBufferEnsure(((__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj)->_device, &tSmallAttr, 8192))
                return false;
            memcpy(tSmallAttr.contents, (const void *)(uintptr_t)buf->data.buffer_data,
                   (size_t)buf->size < 8192 ? (size_t)buf->size : 8192);
            [enc setBuffer: tSmallAttr offset: plan->binding_offset[b] atIndex: (NSUInteger)idx];
        }
    }

    /* current generic values (disabled arrays): one 16-byte slot per location */
    if (plan->const_buffer >= 0)
    {
        struct MGLExtState_vattrib *vs = (struct MGLExtState_vattrib *)ctx->state.ext_vattrib;
        uint8_t data[MAX_ATTRIB_SLOTS * 16];

        memset(data, 0, sizeof data);
        if (vs)
        {
            for (int i = 0; i < plan->count; i++)
            {
                TessAttribSlot *s = &plan->slot[i];
                if (s->fetch == TESS_FETCH_CONST)
                    memcpy(data + s->loc * 16, vs->current[s->loc].v.f, 16);
            }
        }
        [enc setBytes: data length: sizeof(data) atIndex: (NSUInteger)plan->const_buffer];
    }

    /* CPU-converted streams */
    for (int i = 0; i < plan->count; i++)
    {
        TessAttribSlot *s = &plan->slot[i];
        if (s->fetch != TESS_FETCH_SHADOW)
            continue;
        if (!tessConvertShadowStream(ctx, s, vertex_count))
            return false;
        [enc setBuffer: tShadow offset: 0 atIndex: (NSUInteger)s->shadow_buffer];
    }
    return true;
}

/* read a draw's index list (CPU-visible shared storage) and widen it to uint32,
   reporting the referenced vertex range */
static bool tessReadIndices(GLMContext ctx, GLenum type, size_t offset, GLsizei count, GLint basevertex,
                            uint32_t *dst, uint32_t *minv, uint32_t *maxv)
{
    Buffer *ebuf = ctx->state.vao ? ctx->state.vao->element_array.buffer : NULL;
    const uint8_t *ptr;
    size_t esize = (type == GL_UNSIGNED_SHORT) ? 2 : (type == GL_UNSIGNED_INT) ? 4 : 1;
    uint32_t lo = 0xffffffff, hi = 0;

    if (!ebuf || !ebuf->data.buffer_data)
        return false;
    if (offset > (size_t)ebuf->size || (size_t)count * esize > (size_t)ebuf->size - offset)
        return false;

    mgl_buffers_prepare_cpu_read(ctx, ebuf);
    ptr = (const uint8_t *)ebuf->data.buffer_data + offset;

    for (GLsizei i = 0; i < count; i++)
    {
        uint32_t v;
        switch (type)
        {
            case GL_UNSIGNED_SHORT: v = ((const uint16_t *)ptr)[i]; break;
            case GL_UNSIGNED_INT:   v = ((const uint32_t *)ptr)[i]; break;
            default:                v = ptr[i]; break;
        }
        dst[i] = (uint32_t)basevertex + v;
        if (dst[i] < lo) lo = dst[i];
        if (dst[i] > hi) hi = dst[i];
    }
    if (lo == 0xffffffff) { lo = 0; hi = 0; }
    *minv = lo;
    *maxv = hi;
    return true;
}

/* CPU conversion of a non-Metal-fetchable attribute stream (same decisions as the
   vattrib domain's shadow streams: GL_FIXED / doubles / BGRA / misaligned) */
static bool tessConvertShadowStream(GLMContext ctx, TessAttribSlot *s, GLuint vertex_count)
{
    MGLVertexArrayFull *va = mgl_va_full(ctx->state.vao);
    MGLVertexAttribFormat *f = &va->fmt[s->loc];
    MGLVertexBinding *b = &va->binding[f->binding];
    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    const uint8_t *src = b->buffer && b->buffer->data.buffer_data ?
                         (const uint8_t *)b->buffer->data.buffer_data : NULL;
    size_t stride = b->stride ? (size_t)b->stride : (size_t)mgl_va_element_size(f->type, f->size);
    size_t need = (size_t)vertex_count * 16 + 16;

    if (!tessBufferEnsure(self2->_device, &tShadow, need))
        return false;

    if (src)
        mgl_buffers_prepare_cpu_read(ctx, b->buffer);

    {
        uint8_t *dst = (uint8_t *)tShadow.contents;
        for (GLuint i = 0; i < vertex_count; i++)
        {
            const uint8_t *el = src + (size_t)b->offset + (size_t)f->relativeoffset + (size_t)i * stride;
            float *out = (float *)(dst + (size_t)i * 16);

            if (!src)
            {
                memset(out, 0, 16);
                continue;
            }

            if (s->kind == MGL_VA_KIND_INT || s->kind == MGL_VA_KIND_UINT)
            {
                GLint iv[4];
                mgl_va_fetch_int(f, el, iv);
                memcpy(out, iv, sizeof iv);
            }
            else
            {
                GLfloat fv[4];
                mgl_va_fetch_float(f, el, fv);
                memcpy(out, fv, sizeof fv);
            }
        }
    }
    return true;
}

static bool tessDrawPatchesHook(GLMContext ctx, bool indexed, GLenum type, size_t offset,
                                GLint first, GLsizei count, GLsizei instances,
                                GLint basevertex, GLint baseinstance)
{
    return [(__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj
            extTessDrawPatches: indexed type: type offset: offset
            first: first count: count instances: instances
            basevertex: basevertex baseinstance: baseinstance];
}

#pragma mark ext hooks

- (void) bindExtTessFuncsToGLMContext: (GLMContext) glm_ctx
{
    if (!glm_ctx->mtl_funcs.ext_tess)
        glm_ctx->mtl_funcs.ext_tess = (struct MGLExtMetal_tess *)calloc(1, sizeof(struct MGLExtMetal_tess));
    if (!glm_ctx->mtl_funcs.ext_tess)
        return;
    glm_ctx->mtl_funcs.ext_tess->draw_patches = tessDrawPatchesHook;
}


/* compile the GPU tessellator kernels (once) */
static bool tessKernels(GLMContext ctx)
{
    if (tCountPso && tEmitPso && tArgsPso)
        return true;

    MGLRenderer *self2 = (__bridge MGLRenderer *)ctx->mtl_funcs.mtlObj;
    NSError *err = nil;

    if (!tKernelLib)
    {
        tKernelLib = [self2->_device newLibraryWithSource: kTessKernelSource options: nil error: &err];
        if (!tKernelLib)
        {
            NSLog(@"MGL ERROR: tess: tessellator kernels failed to compile: %@", err.localizedDescription);
            return false;
        }
    }
    if (!tCountPso) tCountPso = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_count"] error: &err];
    if (!tEmitPso)  tEmitPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_emit"]  error: &err];
    if (!tArgsPso)  tArgsPso  = [self2->_device newComputePipelineStateWithFunction: [tKernelLib newFunctionWithName: @"mgl_tess_args"]  error: &err];
    if (!tCountPso || !tEmitPso || !tArgsPso)
    {
        NSLog(@"MGL ERROR: tess: tessellator kernel PSO failed: %@", err.localizedDescription);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ hardware pipeline */

/* half -> float (factor readback on the emulated path) */
static float tessHalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;

    if (exp == 0)
    {
        if (man == 0)
            bits = sign;
        else
        {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    }
    else if (exp == 0x1f)
        bits = sign | 0x7f800000 | (man << 13);
    else
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);

    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* float -> half (default factor levels on the hardware path) */
static uint16_t tessFloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t man = bits & 0x7fffff;

    if (((bits >> 23) & 0xff) == 0xff)
        return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));   /* inf/nan */
    if (((bits >> 23) & 0xff) == 0)
    {
        if (man == 0)
            return (uint16_t)sign;
        /* subnormal float -> half subnormal (rare for levels) */
        int e = -1;
        while (!(man & 0x400000)) { man <<= 1; e--; }
        man >>= 13;
        return (uint16_t)(sign | ((uint32_t)(e + 1) << 10) | man);
    }
    if (exp >= 0x1f)
        return (uint16_t)(sign | 0x7c00);                        /* overflow -> inf */
    if (exp <= 0)
        return (uint16_t)(sign | 0x3ff);                         /* underflow -> max subnormal */

    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

/* MSL member type -> Metal vertex format of the TES interface attributes */
static MTLVertexFormat tessMemberFormat(const char *type)
{
    if (!strcmp(type, "float"))  return MTLVertexFormatFloat;
    if (!strcmp(type, "float2")) return MTLVertexFormatFloat2;
    if (!strcmp(type, "float3")) return MTLVertexFormatFloat3;
    if (!strcmp(type, "float4")) return MTLVertexFormatFloat4;
    if (!strcmp(type, "int"))    return MTLVertexFormatInt;
    if (!strcmp(type, "int2"))   return MTLVertexFormatInt2;
    if (!strcmp(type, "int3"))   return MTLVertexFormatInt3;
    if (!strcmp(type, "int4"))   return MTLVertexFormatInt4;
    if (!strcmp(type, "uint"))   return MTLVertexFormatUInt;
    if (!strcmp(type, "uint2"))  return MTLVertexFormatUInt2;
    if (!strcmp(type, "uint3"))  return MTLVertexFormatUInt3;
    if (!strcmp(type, "uint4"))  return MTLVertexFormatUInt4;
    if (!strcmp(type, "half"))   return MTLVertexFormatHalf;
    if (!strcmp(type, "half2"))  return MTLVertexFormatHalf2;
    if (!strcmp(type, "half3"))  return MTLVertexFormatHalf3;
    if (!strcmp(type, "half4"))  return MTLVertexFormatHalf4;
    return MTLVertexFormatInvalid;
}

/* vertex descriptor of the post-tessellation TES:
 *   buffer MGL_TESS_VD_CP      control point records  (PerPatchControlPoint)
 *   buffer MGL_TESS_VD_PATCH   per-patch data          (PerPatch)
 *   buffer MGL_TESS_VD_FACTORS half tess levels        (PerPatch, stride 12)
 */
- (MTLVertexDescriptor *) extTessVertexDescriptor
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);
    MTLVertexDescriptor *vd;

    if (!tp)
        return nil;
    vd = [MTLVertexDescriptor vertexDescriptor];

    for (int i = 0; i < tp->num_tes_cp; i++)
    {
        MGLTessPatchMember *m = &tp->tes_cp[i];
        MTLVertexFormat fmt = tessMemberFormat(m->type);
        size_t offset = m->offset;

        if (fmt == MTLVertexFormatInvalid || m->attr < 0)
            continue;

        /* offsets were remapped to the full record layout (by GL location) at link time */
        vd.attributes[m->attr].format = fmt;
        vd.attributes[m->attr].offset = (NSUInteger)offset;
        vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_CP;
    }
    /* stride: the FULL control point record (TCS/VS output), even if the TES reads a subset */
    vd.layouts[MGL_TESS_VD_CP].stride = (NSUInteger)(tp->cp_record_full_size ?
        tp->cp_record_full_size : (tp->cp_record_size ? tp->cp_record_size : 16));
    vd.layouts[MGL_TESS_VD_CP].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
    vd.layouts[MGL_TESS_VD_CP].stepRate = 1;

    bool any_patch = false, any_factors = false;
    for (int i = 0; i < tp->num_tes_patch_in; i++)
    {
        MGLTessPatchMember *m = &tp->tes_patch_in[i];
        MTLVertexFormat fmt;
        NSUInteger offset = (NSUInteger)m->offset;

        if (m->attr < 0)
            continue;

        if (m->is_tess_level == 1)
        {
            /* gl_TessLevelOuter / packed gl_TessLevel: half4 at the head of the factor record */
            vd.attributes[m->attr].format = MTLVertexFormatHalf4;
            vd.attributes[m->attr].offset = 0;
            vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_FACTORS;
            any_factors = true;
            continue;
        }
        if (m->is_tess_level == 2)
        {
            /* gl_TessLevelInner: half2 at byte 8 (after the four half edge factors) */
            vd.attributes[m->attr].format = MTLVertexFormatHalf2;
            vd.attributes[m->attr].offset = 8;
            vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_FACTORS;
            any_factors = true;
            continue;
        }

        fmt = tessMemberFormat(m->type);
        if (fmt == MTLVertexFormatInvalid)
            continue;
        if (tp->has_tcs)
        {
            /* the TCS kernel writes its own patchOut struct: use its member offsets */
            offset = 0;
            for (int j = 0; j < tp->num_tcs_patch_out; j++)
            {
                if (!strcmp(tp->tcs_patch_out[j].name, m->name) &&
                    tp->tcs_patch_out[j].size == m->size)
                {
                    offset = (NSUInteger)tp->tcs_patch_out[j].offset;
                    break;
                }
            }
        }
        vd.attributes[m->attr].format = fmt;
        vd.attributes[m->attr].offset = offset;
        vd.attributes[m->attr].bufferIndex = MGL_TESS_VD_PATCH;
        any_patch = true;
    }
    if (any_patch)
    {
        vd.layouts[MGL_TESS_VD_PATCH].stride = (NSUInteger)(tp->has_tcs ?
            (tp->tcs_patch_out_size ? tp->tcs_patch_out_size : 16) :
            (tp->tes_patch_in_size ? tp->tes_patch_in_size : 16));
        vd.layouts[MGL_TESS_VD_PATCH].stepFunction = MTLVertexStepFunctionPerPatch;
        vd.layouts[MGL_TESS_VD_PATCH].stepRate = 1;
    }
    if (any_factors)
    {
        vd.layouts[MGL_TESS_VD_FACTORS].stride = 12;   /* MTL{Triangle,Quad}TessellationFactorsHalf */
        vd.layouts[MGL_TESS_VD_FACTORS].stepFunction = MTLVertexStepFunctionPerPatch;
        vd.layouts[MGL_TESS_VD_FACTORS].stepRate = 1;
    }

    return vd;
}

/* tessellation stage of the render pipeline descriptor */
- (void) extTessConfigurePipeline: (MTLRenderPipelineDescriptor *) desc
{
    MGLTessProgram *tp = mgl_tess_active_program(ctx);

    if (!tp || !desc)
        return;

    switch (tp->spacing)
    {
        case MGL_TESS_SPACING_FRACTIONAL_EVEN: desc.tessellationPartitionMode = MTLTessellationPartitionModeFractionalEven; break;
        case MGL_TESS_SPACING_FRACTIONAL_ODD:  desc.tessellationPartitionMode = MTLTessellationPartitionModeFractionalOdd; break;
        default:                               desc.tessellationPartitionMode = MTLTessellationPartitionModeInteger; break;
    }
    desc.maxTessellationFactor = MGL_TESS_MAX_LEVEL;
    desc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
    desc.tessellationOutputWindingOrder = tp->vertex_order_ccw ? MTLWindingCounterClockwise : MTLWindingClockwise;
}
@end
