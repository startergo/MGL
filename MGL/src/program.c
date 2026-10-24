/*
 * Copyright (C) Michael Larson on 1/6/2022
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * program.c
 * MGL
 *
 * Program objects: attachments, linking (GLSL via glslang, or SPIR-V modules from
 * glShaderBinary/glSpecializeShader, or a program binary) into per-stage SPIR-V →
 * SPIRV-Cross MSL → Metal functions, plus reflection and program queries.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <CoreFoundation/CoreFoundation.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>
#include "spirv-tools/libspirv.h"
#include "spirv_cross_c.h"
#include "spirv.h"

#include "glm_context.h"
#include "mgl.h"
#include "mgl_ext.h"
#include "mgl_ext_program.h"
#include "mgl_ext_introspect.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_geom.h"
#include "mgl_ext_tess.h"
#include "shaders.h"
#include "fp64.h"
#include "buffers.h"
#include "msl_reserved.h"

void mgl_program_ref(Program *ptr);
void mgl_program_unref(GLMContext ctx, Program *ptr);
GLint mgl_program_binary_length(GLMContext ctx, Program *ptr);

#pragma mark - program pipeline objects (storage; API in program_ext.c)

ProgramPipeline *newProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    MGLPipelineObj *ptr;

    ptr = (MGLPipelineObj *)calloc(1, sizeof(MGLPipelineObj));
    if (!ptr)
        return NULL;

    ptr->base.name = pipeline;

    return &ptr->base;
}

ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    if (pipeline == 0 || STATE(program_pipeline_table).keys == NULL)
        return NULL;

    return (ProgramPipeline *)searchHashTable(&STATE(program_pipeline_table), pipeline);
}

ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);

    if (!ptr)
    {
        ptr = newProgramPipeline(ctx, pipeline);
        if (ptr)
            insertHashElement(&STATE(program_pipeline_table), pipeline, ptr);
    }

    return ptr;
}

#pragma mark - transform feedback objects (storage)

TransformFeedback *newTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr;

    ptr = (TransformFeedback *)malloc(sizeof(TransformFeedback));
    assert(ptr);

    bzero(ptr, sizeof(TransformFeedback));
    ptr->name = name;
    ptr->target = GL_TRANSFORM_FEEDBACK;
    ptr->active = GL_FALSE;
    ptr->paused = GL_FALSE;
    ptr->primitive_mode = GL_NONE;

    return ptr;
}

TransformFeedback *findTransformFeedback(GLMContext ctx, GLuint name)
{
    return (TransformFeedback *)searchHashTable(&STATE(transform_feedback_table), name);
}

TransformFeedback *getTransformFeedback(GLMContext ctx, GLuint name)
{
    TransformFeedback *ptr = findTransformFeedback(ctx, name);

    if (!ptr)
    {
        ptr = newTransformFeedback(ctx, name);
        insertHashElement(&STATE(transform_feedback_table), name, ptr);
    }

    return ptr;
}

#pragma mark - program objects

Program *newProgram(GLMContext ctx, GLuint program)
{
    MGLProgramObj *ptr;

    ptr = (MGLProgramObj *)calloc(1, sizeof(MGLProgramObj));
    if (!ptr)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return NULL;
    }

    ptr->base.name = program;

    return &ptr->base;
}

Program *getProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    ptr = (Program *)searchHashTable(&STATE(program_table), program);

    if (!ptr)
    {
        ptr = newProgram(ctx, program);
        if (!ptr)
            return NULL;

        MGL_PROGRAM_OBJ(ptr)->in_table = GL_TRUE;
        insertHashElement(&STATE(program_table), program, ptr);
    }

    return ptr;
}

int isProgram(GLMContext ctx, GLuint program)
{
    if (program == 0)
        return 0;

    return searchHashTable(&STATE(program_table), program) != NULL;
}

Program *findProgram(GLMContext ctx, GLuint program)
{
    if (program == 0)
        return NULL;

    return (Program *)searchHashTable(&STATE(program_table), program);
}

bool mgl_program_is_linked(Program *program)
{
    return program && MGL_PROGRAM_OBJ(program)->link_status;
}

/* program-name lookup with the GL error rules for "program" parameters */
static Program *lookupProgram(GLMContext ctx, GLuint program)
{
    Program *ptr = findProgram(ctx, program);

    if (!ptr)
    {
        if (program && findShader(ctx, program))
            ERROR_RETURN(GL_INVALID_OPERATION);
        else
            ERROR_RETURN(GL_INVALID_VALUE);
    }

    return ptr;
}

GLuint mglCreateProgram(GLMContext ctx)
{
    GLuint program;

    /* shaders and programs share one GL namespace */
    program = getNewName(&STATE(program_table));
    if (program < STATE(shader_table).current_name)
        program = STATE(shader_table).current_name;
    STATE(program_table).current_name = program + 1;
    STATE(shader_table).current_name = program + 1;

    if (!getProgram(ctx, program))
        return 0;

    return program;
}

#pragma mark - linked executable storage

static void release_mtl(GLMContext ctx, void *obj)
{
    if (!obj)
        return;

    if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->release_obj)
        ctx->mtl_funcs.ext_program->release_obj(ctx, obj);
    else
        CFRelease(obj);
}

/* free the linked executable of a program (stages, SPIR-V, MSL, reflection) */
static void exec_free(GLMContext ctx, MGLProgramObj *obj)
{
    Program *ptr = &obj->base;

    for (int i = 0; i < _MAX_SHADER_TYPES; i++)
    {
        free(ptr->spirv[i].ir);
        free(ptr->spirv[i].msl_str);
        free(ptr->spirv[i].entry_point);
        release_mtl(ctx, ptr->spirv[i].mtl_function);
        release_mtl(ctx, ptr->spirv[i].mtl_library);
        memset(&ptr->spirv[i], 0, sizeof(ptr->spirv[i]));

        for (int j = 0; j < _MAX_SPIRV_RES; j++)
        {
            SpirvResourceList *rl = &ptr->spirv_resources_list[i][j];
            for (GLuint k = 0; rl->list && k < rl->count; k++)
                free((void *)rl->list[k].name);
            free(rl->list);
            rl->list = NULL;
            rl->count = 0;
        }

        if (ptr->shader_slots[i])
        {
            /* executable stages are internal shader objects owned by the program */
            if (MGL_SHADER_OBJ(ptr->shader_slots[i])->internal)
                mglFreeShader(ctx, ptr->shader_slots[i]);
            ptr->shader_slots[i] = NULL;
        }

        free(obj->stage_entry[i]);
        obj->stage_entry[i] = NULL;
    }

    for (int i = 0; i < obj->num_samplers; i++)
        free(obj->samplers[i].name);
    free(obj->samplers);
    obj->samplers = NULL;
    obj->num_samplers = 0;

    if (ptr->linked_glsl_program)
    {
        glslang_program_delete(ptr->linked_glsl_program);
        ptr->linked_glsl_program = NULL;
    }

    ptr->local_workgroup_size.x = ptr->local_workgroup_size.y = ptr->local_workgroup_size.z = 0;
    obj->linked_mask = 0;
    obj->active_attributes = obj->active_attribute_max_length = 0;
    obj->active_uniforms = obj->active_uniform_max_length = 0;
    obj->active_uniform_blocks = obj->active_uniform_block_max_name_length = 0;
    obj->active_atomic_counter_buffers = 0;
}

/* move a freshly linked executable from src into dst (dst's must be freed) */
static void exec_move(MGLProgramObj *dst, MGLProgramObj *src)
{
    memcpy(dst->base.shader_slots, src->base.shader_slots, sizeof(dst->base.shader_slots));
    memcpy(dst->base.spirv, src->base.spirv, sizeof(dst->base.spirv));
    memcpy(dst->base.spirv_resources_list, src->base.spirv_resources_list, sizeof(dst->base.spirv_resources_list));
    memcpy(dst->stage_entry, src->stage_entry, sizeof(dst->stage_entry));
    dst->base.local_workgroup_size.x = src->base.local_workgroup_size.x;
    dst->base.local_workgroup_size.y = src->base.local_workgroup_size.y;
    dst->base.local_workgroup_size.z = src->base.local_workgroup_size.z;
    dst->base.linked_glsl_program = src->base.linked_glsl_program;
    dst->linked_mask = src->linked_mask;
    dst->samplers = src->samplers;
    dst->num_samplers = src->num_samplers;
    dst->active_attributes = src->active_attributes;
    dst->active_attribute_max_length = src->active_attribute_max_length;
    dst->active_uniforms = src->active_uniforms;
    dst->active_uniform_max_length = src->active_uniform_max_length;
    dst->active_uniform_blocks = src->active_uniform_blocks;
    dst->active_uniform_block_max_name_length = src->active_uniform_block_max_name_length;
    dst->active_atomic_counter_buffers = src->active_atomic_counter_buffers;

    memset(src->base.shader_slots, 0, sizeof(src->base.shader_slots));
    memset(src->base.spirv, 0, sizeof(src->base.spirv));
    memset(src->base.spirv_resources_list, 0, sizeof(src->base.spirv_resources_list));
    memset(src->stage_entry, 0, sizeof(src->stage_entry));
    src->base.linked_glsl_program = NULL;
    src->samplers = NULL;
    src->num_samplers = 0;
}

static void detach_all(GLMContext ctx, MGLProgramObj *obj)
{
    for (int i = 0; i < obj->num_attached; i++)
        mgl_shader_unref(ctx, obj->attached[i]);

    free(obj->attached);
    obj->attached = NULL;
    obj->num_attached = obj->cap_attached = 0;
}

void mglFreeProgram(GLMContext ctx, Program *ptr)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(ptr);

    exec_free(ctx, obj);

    if (ptr->mtl_data)
    {
        release_mtl(ctx, ptr->mtl_data);
        ptr->mtl_data = NULL;
    }

    detach_all(ctx, obj);

    free(obj->info_log);

    if (obj->in_table)
        deleteHashElement(&STATE(program_table), ptr->name);

    /* tess: drop per-program tessellation metadata */
    mgl_ext_delete_program_tess(ctx, ptr);

    free(obj);
}

void mgl_program_ref(Program *ptr)
{
    if (ptr)
        ptr->refcount++;
}

/* drop a use reference (current program / pipeline stage); frees a program flagged for deletion */
void mgl_program_unref(GLMContext ctx, Program *ptr)
{
    if (!ptr)
        return;

    if (ptr->refcount > 0)
        ptr->refcount--;

    if (ptr->refcount == 0 && ptr->delete_status)
        mglFreeProgram(ctx, ptr);
}

void mglDeleteProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;

    /* a value of 0 is silently ignored */
    if (program == 0)
        return;

    ptr = findProgram(ctx, program);
    if (!ptr)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (ptr->delete_status)
        return;

    ptr->delete_status = GL_TRUE;

    /* in use (current program / pipeline stage): flagged, freed when released */
    if (ptr->refcount == 0)
        mglFreeProgram(ctx, ptr);
}

GLboolean mglIsProgram(GLMContext ctx, GLuint program)
{
    return isProgram(ctx, program) ? GL_TRUE : GL_FALSE;
}

static Shader *lookupShaderForProgram(GLMContext ctx, GLuint shader)
{
    Shader *sptr = findShader(ctx, shader);

    if (!sptr)
    {
        if (shader && findProgram(ctx, shader))
            ERROR_RETURN(GL_INVALID_OPERATION);
        else
            ERROR_RETURN(GL_INVALID_VALUE);
    }

    return sptr;
}

void mglAttachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    MGLProgramObj *obj;

    pptr = lookupProgram(ctx, program);
    if (!pptr)
        return;

    sptr = lookupShaderForProgram(ctx, shader);
    if (!sptr)
        return;

    obj = MGL_PROGRAM_OBJ(pptr);

    for (int i = 0; i < obj->num_attached; i++)
    {
        if (obj->attached[i] == sptr)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    if (obj->num_attached == obj->cap_attached)
    {
        int cap = obj->cap_attached ? obj->cap_attached * 2 : 4;
        Shader **list = (Shader **)realloc(obj->attached, cap * sizeof(Shader *));
        if (!list)
        {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
        obj->attached = list;
        obj->cap_attached = cap;
    }

    obj->attached[obj->num_attached++] = sptr;
    sptr->refcount++;
}

void mglDetachShader(GLMContext ctx, GLuint program, GLuint shader)
{
    Program *pptr;
    Shader *sptr;
    MGLProgramObj *obj;

    pptr = lookupProgram(ctx, program);
    if (!pptr)
        return;

    sptr = lookupShaderForProgram(ctx, shader);
    if (!sptr)
        return;

    obj = MGL_PROGRAM_OBJ(pptr);

    for (int i = 0; i < obj->num_attached; i++)
    {
        if (obj->attached[i] == sptr)
        {
            memmove(&obj->attached[i], &obj->attached[i + 1], (obj->num_attached - i - 1) * sizeof(Shader *));
            obj->num_attached--;
            mgl_shader_unref(ctx, sptr);
            return;
        }
    }

    /* not attached */
    ERROR_RETURN(GL_INVALID_OPERATION);
}

#pragma mark - SPIR-V → MSL → Metal for one stage

static_assert(_VERTEX_SHADER == GLSLANG_STAGE_VERTEX, "_VERTEX_SHADER == GLSLANG_STAGE_VERTEX failed");
static_assert(_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL, "_TESS_CONTROL_SHADER == GLSLANG_STAGE_TESSCONTROL failed");
static_assert(_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION, "_TESS_EVALUATION_SHADER == GLSLANG_STAGE_TESSEVALUATION failed");
static_assert(_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY, "_GEOMETRY_SHADER == GLSLANG_STAGE_GEOMETRY failed");
static_assert(_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT, "_FRAGMENT_SHADER == GLSLANG_STAGE_FRAGMENT failed");
static_assert(_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE, "_COMPUTE_SHADER == GLSLANG_STAGE_COMPUTE failed");

static const char *stage_label(int stage)
{
    static const char *names[] = {"vertex", "tessellation control", "tessellation evaluation",
        "geometry", "fragment", "compute"};
    return (stage >= 0 && stage < _MAX_SHADER_TYPES) ? names[stage] : "unknown";
}

static SpvExecutionModel stage_model(int stage)
{
    switch (stage)
    {
        case _VERTEX_SHADER: return SpvExecutionModelVertex;
        case _TESS_CONTROL_SHADER: return SpvExecutionModelTessellationControl;
        case _TESS_EVALUATION_SHADER: return SpvExecutionModelTessellationEvaluation;
        case _GEOMETRY_SHADER: return SpvExecutionModelGeometry;
        case _FRAGMENT_SHADER: return SpvExecutionModelFragment;
        default: return SpvExecutionModelGLCompute;
    }
}

static void spvc_log_cb(void *userdata, const char *error)
{
    mgl_append_info_log((char **)userdata, "ERROR: SPIRV-Cross: %s\n", error ? error : "(null)");
}

/* reflection accumulator: names deduplicated across stages */
typedef struct {
    char  **uniforms;  int num_uniforms, cap_uniforms;
    char  **blocks;    int num_blocks, cap_blocks;
    GLuint  atomic_bindings[64]; int num_atomic;
} ReflectAcc;

static bool name_list_add(char ***list, int *count, int *cap, const char *name)
{
    for (int i = 0; i < *count; i++)
        if (!strcmp((*list)[i], name))
            return false;

    if (*count == *cap)
    {
        int ncap = *cap ? *cap * 2 : 16;
        char **nl = (char **)realloc(*list, ncap * sizeof(char *));
        if (!nl)
            return false;
        *list = nl;
        *cap = ncap;
    }

    (*list)[(*count)++] = strdup(name);
    return true;
}

static void reflect_acc_free(ReflectAcc *acc)
{
    for (int i = 0; i < acc->num_uniforms; i++) free(acc->uniforms[i]);
    for (int i = 0; i < acc->num_blocks; i++) free(acc->blocks[i]);
    free(acc->uniforms);
    free(acc->blocks);
}

static unsigned array_size(spvc_type t)
{
    unsigned dims = spvc_type_get_num_array_dimensions(t);
    if (dims == 0)
        return 0;
    if (!spvc_type_array_dimension_is_literal(t, 0))
        return 1;
    unsigned n = (unsigned)spvc_type_get_array_dimension(t, 0);
    return n ? n : 1;   /* runtime-sized arrays count as one */
}

/* enumerate the GL-visible uniform names of a block member tree */
static void reflect_members(spvc_compiler c, spvc_type_id type_id, const char *prefix, ReflectAcc *acc, int depth)
{
    spvc_type t = spvc_compiler_get_type_handle(c, type_id);
    unsigned members = spvc_type_get_num_member_types(t);

    if (depth > 8)
        return;

    for (unsigned m = 0; m < members; m++)
    {
        spvc_type_id mt_id = spvc_type_get_member_type(t, m);
        spvc_type mt = spvc_compiler_get_type_handle(c, mt_id);
        const char *mname = spvc_compiler_get_member_name(c, type_id, m);
        char name[512];
        unsigned n = array_size(mt);

        if (!mname || !*mname)
            continue;

        if (spvc_type_get_basetype(mt) == SPVC_BASETYPE_STRUCT)
        {
            if (n == 0)
            {
                snprintf(name, sizeof(name), "%s%s.", prefix, mname);
                reflect_members(c, mt_id, name, acc, depth + 1);
            }
            else
            {
                for (unsigned e = 0; e < n && e < 64; e++)
                {
                    snprintf(name, sizeof(name), "%s%s[%u].", prefix, mname, e);
                    reflect_members(c, mt_id, name, acc, depth + 1);
                }
            }
        }
        else
        {
            snprintf(name, sizeof(name), n ? "%s%s[0]" : "%s%s", prefix, mname);
            name_list_add(&acc->uniforms, &acc->num_uniforms, &acc->cap_uniforms, name);
        }
    }
}

static GLuint sampler_type_key(spvc_compiler c, spvc_type_id type_id)
{
    spvc_type t = spvc_compiler_get_type_handle(c, type_id);
    spvc_type sampled = spvc_compiler_get_type_handle(c, spvc_type_get_image_sampled_type(t));
    GLuint key;

    key  = (GLuint)spvc_type_get_image_dimension(t) & 0xF;
    key |= spvc_type_get_image_arrayed(t) ? 0x10 : 0;
    key |= spvc_type_get_image_is_depth(t) ? 0x20 : 0;
    key |= spvc_type_get_image_multisampled(t) ? 0x40 : 0;
    key |= ((GLuint)spvc_type_get_basetype(sampled) & 0xFF) << 8;

    return key;
}

/* ------------------------------------------------------------------ */
/* Gather emulation for the non-fragment stages (AGX: the hardware gather
 * returns raw values / zeros in vertex and compute functions; fragment
 * functions are unaffected).
 *
 * Plain gathers become mgl_gath_vs: four explicit texel reads of the 2x2 tap
 * block, channel-selected.  A gather selects its own taps (no filtering), so
 * reads are exact; the block base is `floor(uv * size - 0.5 + off)' for
 * normalized samplers and `floor(uv - 0.5 + off)' for rectangle (unnormalized)
 * ones — mgl_spirv_rect_to_2d marks those variables with the _mglrect suffix.
 * Tap order = the GL/CTS convention measured from hardware gathers:
 * (x0,y1) (x1,y1) (x1,y0) (x0,y0) of the 2x2 block.
 *
 * Shadow (depth) gathers additionally need the comparison.  In VERTEX
 * functions sample_compare itself is broken (the sampler's compare function is
 * ignored and the raw depth is returned) AND sampling a depth texture through a
 * plain texture2d with a compare sampler is unreliable there, so the vertex
 * stage compares in the shader: mgl_gcmp_vs does four depth2d.read() fetches
 * plus a manual `ref OP d', where OP is GL_TEXTURE_COMPARE_FUNC of the unit
 * that feeds the texture slot, passed as a function constant 32010+slot
 * (mgl_shadow_op_N, see mgl_ext_geom.h).  Function constants work at any scope
 * (the CTS calls textureGather from a nested function) and rstateSpecialize
 * fills them from the GL sampler state when the pipeline is built.  In COMPUTE
 * functions sample_compare works, so mgl_gcmp_vs there keeps hardware
 * comparison taps at exact texel centres (bilinear puts all weight on one tap,
 * so any filter setting gives the same answer as the hardware gather_compare).
 *
 * depthcube / depthcube_array gathers keep the (broken) hardware path: their
 * tap quads are face-layout dependent and the CTS cube cases are documented as
 * a gap.  Plain `.sample_compare' in a vertex function also stays hardware:
 * it cannot be emulated with texel reads when the min filter is LINEAR. */

static const char *k_gath_helpers =
"template <typename T>\n"
"static inline __attribute__((always_inline)) T mgl_gsel(vec<T,4> v, component c)\n"
"{\n"
"    if (c == component::x) return v.x;\n"
"    if (c == component::y) return v.y;\n"
"    if (c == component::z) return v.z;\n"
"    return v.w;\n"
"}\n"
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gath_vs(texture2d<T> t, sampler s, float2 uv, int2 off, component c, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = mgl_gsel(t.read(uint2(b + int2(0, 1)), 0), c);\n"
"    r[1] = mgl_gsel(t.read(uint2(b + int2(1, 1)), 0), c);\n"
"    r[2] = mgl_gsel(t.read(uint2(b + int2(1, 0)), 0), c);\n"
"    r[3] = mgl_gsel(t.read(uint2(b + int2(0, 0)), 0), c);\n"
"    return r;\n"
"}\n"
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gath_vs(texture2d_array<T> t, sampler s, float2 uv, uint layer, int2 off, component c, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = mgl_gsel(t.read(uint2(b + int2(0, 1)), layer, 0), c);\n"
"    r[1] = mgl_gsel(t.read(uint2(b + int2(1, 1)), layer, 0), c);\n"
"    r[2] = mgl_gsel(t.read(uint2(b + int2(1, 0)), layer, 0), c);\n"
"    r[3] = mgl_gsel(t.read(uint2(b + int2(0, 0)), layer, 0), c);\n"
"    return r;\n"
"}\n";

/* shadow gathers in the VERTEX stage: texel reads + the emulated comparison */
static const char *k_gcmp_read_helpers =
"static inline float mgl_cmp_do(float ref, float d, int op)\n"
"{\n"
"    if (op == 1) return ref >= d;\n"    /* GL_GEQUAL */
"    if (op == 2) return ref < d;\n"     /* GL_LESS */
"    if (op == 3) return ref > d;\n"     /* GL_GREATER */
"    if (op == 4) return ref == d;\n"    /* GL_EQUAL */
"    if (op == 5) return ref != d;\n"    /* GL_NOTEQUAL */
"    if (op == 6) return 1.0;\n"         /* GL_ALWAYS */
"    if (op == 7) return 0.0;\n"         /* GL_NEVER */
"    return ref <= d;\n"                 /* 0 = GL_LEQUAL (the GL default) */
"}\n"
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gcmp_vs(depth2d<T> t, sampler s, float2 uv, T ref, int2 off, int op, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = mgl_cmp_do(ref, t.read(uint2(b + int2(0, 1)), 0), op);\n"
"    r[1] = mgl_cmp_do(ref, t.read(uint2(b + int2(1, 1)), 0), op);\n"
"    r[2] = mgl_cmp_do(ref, t.read(uint2(b + int2(1, 0)), 0), op);\n"
"    r[3] = mgl_cmp_do(ref, t.read(uint2(b + int2(0, 0)), 0), op);\n"
"    return r;\n"
"}\n"
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gcmp_vs(depth2d_array<T> t, sampler s, float2 uv, uint layer, T ref, int2 off, int op, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = mgl_cmp_do(ref, t.read(uint2(b + int2(0, 1)), layer, 0), op);\n"
"    r[1] = mgl_cmp_do(ref, t.read(uint2(b + int2(1, 1)), layer, 0), op);\n"
"    r[2] = mgl_cmp_do(ref, t.read(uint2(b + int2(1, 0)), layer, 0), op);\n"
"    r[3] = mgl_cmp_do(ref, t.read(uint2(b + int2(0, 0)), layer, 0), op);\n"
"    return r;\n"
"}\n";

/* shadow gathers in the COMPUTE stage: hardware sample_compare taps */
static const char *k_gcmp_sc_helpers =
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gcmp_vs(depth2d<T> t, sampler s, float2 uv, T ref, int2 off, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = t.sample_compare(s, unorm ? (float2(b + int2(0, 1)) + float2(0.5)) : (float2(b + int2(0, 1)) + float2(0.5)) / sz, ref, level(0.0));\n"
"    r[1] = t.sample_compare(s, unorm ? (float2(b + int2(1, 1)) + float2(0.5)) : (float2(b + int2(1, 1)) + float2(0.5)) / sz, ref, level(0.0));\n"
"    r[2] = t.sample_compare(s, unorm ? (float2(b + int2(1, 0)) + float2(0.5)) : (float2(b + int2(1, 0)) + float2(0.5)) / sz, ref, level(0.0));\n"
"    r[3] = t.sample_compare(s, unorm ? (float2(b + int2(0, 0)) + float2(0.5)) : (float2(b + int2(0, 0)) + float2(0.5)) / sz, ref, level(0.0));\n"
"    return r;\n"
"}\n"
"template <typename T>\n"
"static inline __attribute__((always_inline)) vec<T,4> mgl_gcmp_vs(depth2d_array<T> t, sampler s, float2 uv, uint layer, T ref, int2 off, bool unorm)\n"
"{\n"
"    float2 sz = float2(float(t.get_width()), float(t.get_height()));\n"
"    int2 b = int2(floor((unorm ? uv : uv * sz) - float2(0.5)) + float2(off));\n"
"    vec<T,4> r;\n"
"    r[0] = t.sample_compare(s, unorm ? (float2(b + int2(0, 1)) + float2(0.5)) : (float2(b + int2(0, 1)) + float2(0.5)) / sz, layer, ref, level(0.0));\n"
"    r[1] = t.sample_compare(s, unorm ? (float2(b + int2(1, 1)) + float2(0.5)) : (float2(b + int2(1, 1)) + float2(0.5)) / sz, layer, ref, level(0.0));\n"
"    r[2] = t.sample_compare(s, unorm ? (float2(b + int2(1, 0)) + float2(0.5)) : (float2(b + int2(1, 0)) + float2(0.5)) / sz, layer, ref, level(0.0));\n"
"    r[3] = t.sample_compare(s, unorm ? (float2(b + int2(0, 0)) + float2(0.5)) : (float2(b + int2(0, 0)) + float2(0.5)) / sz, layer, ref, level(0.0));\n"
"    return r;\n"
"}\n";

static bool mgl_is_idchar(int c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c == '_' || c == '.' || c == '[' || c == ']';
}

/* ------------------------------------------------------------------ */
/* length of the balanced-paren argument list starting at open (the '(') */
static size_t mgl_arglist_len(const char *open)
{
    size_t depth = 0, i = 0;
    do
    {
        if (open[i] == '(') depth++;
        else if (open[i] == ')') depth--;
        else if (open[i] == '\0') return 0;
        i++;
    } while (depth > 0);
    return i;    /* includes both parens */
}

/* split the top-level comma-separated args of "(a, b, c)" into out[4] */
static int mgl_split_args(const char *args, size_t len, char (*out)[256])
{
    int n = 0;
    size_t depth = 0, start = 1;    /* skip '(' */
    for (size_t i = 1; i < len; i++)
    {
        char c = args[i];
        if (c == '(' || c == '[') depth++;
        else if (c == ')' || c == ']') depth--;
        if ((c == ',' && depth == 0) || i == len - 1)
        {
            size_t end = (c == ',' && depth == 0) ? i : i;
            while (end > start && (args[end - 1] == ' ' || args[end - 1] == '\n' || args[end - 1] == '\t')) end--;
            size_t a = start;
            while (a < end && (args[a] == ' ' || args[a] == '\n' || args[a] == '\t')) a++;
            if (n < 5 && end > a && end - a < 256)
            {
                memcpy(out[n], args + a, end - a);
                out[n][end - a] = 0;
                n++;
            }


            start = i + 1;
        }
    }
    return n;
}


/* one gather object of the stage: name, its Metal texture slot and shape
 * (depth => a shadow gather that also needs the compare-op function constant) */
typedef struct MGLShadowObj {
    char    name[128];
    int     slot;
    bool    array;
    bool    depth;
} MGLShadowObj;

/* find `depth2d[_array]<float> NAME [[texture(N)]]' or the plain
 * `texture2d[_array]<float> NAME [[texture(N)]]' in the MSL and return the slot
 * (and whether the object is an array texture); -1 when the object is not
 * declared that way at the entry point (cube / cube_array / depthcube stay on
 * the hardware path — there is no read()-based emulation for them) */
static int mgl_shadow_slot_of(const char *msl, const char *name, bool *is_array, bool *depth)
{
    char pat[192];
    static const char *k_types[4][2] = {
        { "depth2d_array", "depth2d" },
        { "texture2d_array", "texture2d" },
    };
    /* float/int/uint channels (GLSL sampler2D/Rect/2DArray and the shadow
     * depth forms; half-float gathers are not exercised by the CTS) */
    static const char *k_channels[3] = { "float", "int", "uint" };
    for (int t = 0; t < 2; t++)
    for (int ch = 0; ch < 3; ch++)
    for (int arr = 1; arr >= 0; arr--)
    {
        snprintf(pat, sizeof pat, "%s<%s> %s", arr ? k_types[t][0] : k_types[t][1], k_channels[ch], name);
        const char *p = msl;
        while ((p = strstr(p, pat)) != NULL)
        {
            const char *q = p + strlen(pat);
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (strncmp(q, "[[texture(", 10) == 0)
            {
                int slot = atoi(q + 10);
                if (slot >= 0 && slot < 32)
                {
                    *is_array = arr != 0;
                    *depth = t == 0;
                    return slot;
                }
            }
            p += strlen(pat);
        }
    }
    return -1;
}

static char *mgl_fix_gathers_vs(const char *msl, uint32_t *slots_out, bool compute)
{
    if (!msl || (!strstr(msl, ".gather_compare(") && !strstr(msl, ".gather(")))
        return NULL;

    /* 1) collect the objects used by gather calls and their slots */
    MGLShadowObj objs[32];
    int nobjs = 0;
    const char *p = msl;
    while (nobjs < 32)
    {
        const char *gc = strstr(p, ".gather_compare(");
        const char *g = strstr(p, ".gather(");
        const char *hit;
        bool shadow;
        if (gc && (!g || gc < g)) { hit = gc; shadow = true; }
        else if (g)               { hit = g; shadow = false; }
        else break;
        if (gc && g && g < gc) { hit = g; shadow = false; }   /* plain call first */

        const char *obj = hit;
        while (obj > msl && mgl_is_idchar((unsigned char)obj[-1])) obj--;
        char name[128];
        size_t nl = (size_t)(hit - obj);
        if (nl == 0 || nl >= sizeof name) { p = hit + 10; continue; }
        memcpy(name, obj, nl);
        name[nl] = 0;

        bool known = false;
        for (int i = 0; i < nobjs; i++)
            if (strcmp(objs[i].name, name) == 0) { known = true; break; }
        if (!known)
        {
            bool arr = false, dep = false;
            int slot = mgl_shadow_slot_of(msl, name, &arr, &dep);
            if (slot < 0) { p = hit + 10; continue; }   /* not a 2D/array [[texture(N)]] argument: keep hardware path */
            snprintf(objs[nobjs].name, sizeof objs[nobjs].name, "%s", name);
            objs[nobjs].slot = slot;
            objs[nobjs].array = arr;
            objs[nobjs].depth = dep;   /* declared depth2d[_array]: shadow gathers */
            nobjs++;
        }
        else if (shadow)
        {
            /* the same object may gather both ways; a shadow use of a plain
             * texture cannot happen (GL types are distinct), but a depth object
             * that was first seen as `depth2d' keeps its depth flag either way */
            for (int i = 0; i < nobjs; i++)
                if (strcmp(objs[i].name, name) == 0 && objs[i].depth) break;
        }
        p = hit + 10;
    }
    if (!nobjs) return NULL;

    /* 2) rewrite the calls into a fresh buffer */
    const char *helpers = compute ? k_gcmp_sc_helpers : k_gcmp_read_helpers;
    size_t need = strlen(msl) * 2 + strlen(k_gath_helpers) + strlen(helpers) + nobjs * 192 + 4096;
    char *out = (char *)malloc(need);
    if (!out) return NULL;
    out[0] = 0;

    const char *src = msl;
    while (*src)
    {
        const char *gc = strstr(src, ".gather_compare(");
        const char *g = strstr(src, ".gather(");
        const char *hit;
        bool shadow;
        if (gc && (!g || gc < g)) { hit = gc; shadow = true; }
        else if (g)               { hit = g; shadow = false; }
        else { strcat(out, src); break; }

        const char *obj = hit;
        while (obj > src && mgl_is_idchar((unsigned char)obj[-1])) obj--;

        size_t open_at = shadow ? 15 : 7;   /* the '(' of .gather_compare( / .gather( */
        size_t alen = mgl_arglist_len(hit + open_at);
        if (alen == 0 || obj == hit)
        {
            /* malformed or not a simple OBJ.method(args) call: copy up to (but not
             * including) the '(' so the list itself is copied verbatim below and the
             * scan cannot re-match the same call */
            strncat(out, src, (size_t)(hit - src) + open_at);
            src = hit + open_at;
            continue;
        }

        char name[128];
        size_t nl = (size_t)(hit - obj);
        if (nl >= sizeof name) nl = sizeof name - 1;
        memcpy(name, obj, nl);
        name[nl] = 0;

        MGLShadowObj *o = NULL;
        for (int i = 0; i < nobjs; i++)
            if (strcmp(objs[i].name, name) == 0) { o = &objs[i]; break; }

        strncat(out, src, (size_t)(obj - src));
        if (o)
        {
            char args[6][256];
            int nargs = mgl_split_args(hit + open_at, alen, args);
            /* (s, xy[, uint layer], ref[, off]) for shadow calls and
             * (s, xy[, uint layer][, off][, component::x]) for plain gathers */
            int base = o->array ? 3 : 2;
            /* gather_compare only ever happens on depth2d[_array], plain gather
             * only on texture2d[_array] (GL sampler types are distinct) */
            bool ok = shadow == o->depth && nargs >= (shadow ? base + 1 : base);
            if (ok)
            {
                char buf[2048];
                size_t at = (size_t)snprintf(buf, sizeof buf, "%s(%s",
                                             shadow ? "mgl_gcmp_vs" : "mgl_gath_vs", name);
                for (int a = 0; a < base; a++)
                {
                    at += (size_t)snprintf(buf + at, sizeof buf - at, ", %s", args[a]);
                    if (at >= sizeof buf - 1) at = sizeof buf - 1;   /* snprintf returns the untruncated length */
                }

                const char *off = "int2(0)";
                const char *comp = "component::x";
                for (int a = base; a < nargs; a++)
                {
                    if (strncmp(args[a], "component::", 11) == 0) comp = args[a];
                    else off = args[a];
                }
                size_t rl = strlen(o->name);
                const char *rect = (rl > 8 && strcmp(o->name + rl - 8, "_mglrect") == 0) ? "true" : "false";
                if (shadow && !compute)
                {
                    at += (size_t)snprintf(buf + at, sizeof buf - at, ", %s, %s, mgl_shadow_op_%d, %s)",
                                           args[base], off, o->slot, rect);
                    if (slots_out) *slots_out |= 1u << o->slot;
                }
                else if (shadow)
                    at += (size_t)snprintf(buf + at, sizeof buf - at, ", %s, %s, %s)",
                                           args[base], off, rect);
                else
                    at += (size_t)snprintf(buf + at, sizeof buf - at, ", %s, %s, %s)", off, comp, rect);
                if (at >= sizeof buf) at = sizeof buf - 1;
                (void)at;
                strcat(out, buf);
            }
            else
                strncat(out, obj, (size_t)(hit - obj) + open_at + alen);
        }
        else
            strncat(out, obj, (size_t)(hit - obj) + open_at + alen);
        src = hit + open_at + alen;
    }

    /* 3) prepend the helper preamble (+ per-slot compare-op function constants
     * in the vertex stage) at file scope, right after `using namespace metal;' */
    {
        char *pre = (char *)malloc(strlen(k_gath_helpers) + strlen(helpers) + nobjs * 192 + 32);
        if (!pre) { free(out); return NULL; }
        pre[0] = 0;
        for (int i = 0; i < nobjs && !compute; i++)
        {
            if (!objs[i].depth) continue;
            char line[256];
            int dup = 0;
            for (int j = 0; j < i; j++) if (objs[j].slot == objs[i].slot) dup = 1;
            if (dup) continue;
            snprintf(line, sizeof line,
                     "constant int mgl_shadow_op_%d_ [[function_constant(%d)]];\n"
                     "constant int mgl_shadow_op_%d = is_function_constant_defined(mgl_shadow_op_%d_) ? mgl_shadow_op_%d_ : 0;\n",
                     objs[i].slot, MGL_GEOM_FC_SHADOW_OP_BASE + objs[i].slot,
                     objs[i].slot, objs[i].slot, objs[i].slot);
            strcat(pre, line);
        }
        strcat(pre, k_gath_helpers);
        strcat(pre, helpers);

        const char *anchor = strstr(out, "using namespace metal;");
        char *full;
        if (anchor)
        {
            size_t at = (size_t)(anchor - out) + strlen("using namespace metal;");
            full = (char *)malloc(strlen(out) + strlen(pre) + 4);
            if (!full) { free(pre); free(out); return NULL; }
            memcpy(full, out, at);
            full[at] = '\n';
            strcpy(full + at + 1, pre);
            strcat(full, out + at);
        }
        else
        {
            full = (char *)malloc(strlen(out) + strlen(pre) + 4);
            if (!full) { free(pre); free(out); return NULL; }
            strcpy(full, pre);
            strcat(full, out);
        }
        free(pre);
        free(out);
        return full;
    }
}

/* rewrite the image Dim of every OpTypeImage from Rect to 2D in a SPIR-V
 * module (in place); returns how many image types were patched.  OpTypeImage
 * packs Dim(3) | Depth(2) | Arrayed(1) | MS(1) | Sampled(2) | Format(5) into
 * its 4th word (bits from the LSB up).  See the caller for why.
 *
 * Every sampled-image variable built on such a type is additionally marked by
 * extending its OpName with the suffix "_mglrect": SPIRV-Cross uses OpName for
 * the MSL argument name, so the MSL-side gather rewrites (mgl_fix_shadow_vs)
 * can tell an unnormalized rectangle sampler from a normalized one — the
 * texel-block base for a gather is `floor(uv * size - 0.5)' for normalized
 * coordinates but `floor(uv - 0.5)' for raw rectangle coordinates.  The module
 * is rebuilt word by word because the longer string needs more words. */
static size_t mgl_spirv_rect_to_2d(const uint32_t *w, size_t n, uint32_t **out_words, size_t *out_n)
{
    enum { kSpvOpName = 5, kSpvOpTypeImage = 25, kSpvOpTypeSampledImage = 27,
           kSpvOpTypePointer = 32, kSpvOpVariable = 59 };
    enum { kSpvDim2D = 1, kSpvDimRect = 4 };
    uint32_t *ids = NULL;
    size_t patched = 0;

    /* 1) walk the module once: find the rect image types and everything
     *    downstream of them, and count how many words the rebuild needs */
    size_t need = 5;
    if (n < 5) { *out_words = NULL; *out_n = 0; return 0; }
    ids = (uint32_t *)calloc(n, sizeof(uint32_t));
    if (!ids) { *out_words = NULL; *out_n = 0; return 0; }
    /* ids[id] tags: 1 = rect image type, 2 = rect sampled-image type,
     * 3 = pointer to it, 4 = variable of it */
    size_t i = 5;
    while (i < n)
    {
        uint32_t wc = w[i] >> 16;
        uint32_t op = w[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        switch (op)
        {
            case kSpvOpTypeImage:
                if (wc >= 4 && (w[i + 3] & 0x7u) == kSpvDimRect && w[i + 1] < n)
                    ids[w[i + 1]] = 1;
                break;
            case kSpvOpTypeSampledImage:
                if (wc >= 3 && w[i + 2] < n && ids[w[i + 2]] == 1 && w[i + 1] < n)
                    ids[w[i + 1]] = 2;
                break;
            case kSpvOpTypePointer:
                if (wc >= 4 && w[i + 3] < n && ids[w[i + 3]] == 2 && w[i + 1] < n)
                    ids[w[i + 1]] = 3;
                break;
            case kSpvOpVariable:
                if (wc >= 4 && w[i + 1] < n && ids[w[i + 1]] == 3 && w[i + 2] < n)
                    ids[w[i + 2]] = 4;
                break;
            default:
                break;
        }
        if (op == kSpvOpName && wc >= 2 && w[i + 1] < n && ids[w[i + 1]] == 4)
            need += 2;   /* "_mglrect" = 8 chars = 2 words */
        else
            need += wc;
        i += wc;
    }

    uint32_t *o = (uint32_t *)malloc(need * sizeof(uint32_t));
    if (!o) { free(ids); *out_words = NULL; *out_n = 0; return 0; }
    size_t j = 0;
    i = 5;
    memcpy(o, w, 5 * sizeof(uint32_t));
    j = 5;
    while (i < n)
    {
        uint32_t wc = w[i] >> 16;
        uint32_t op = w[i] & 0xffff;
        if (wc == 0 || i + wc > n) break;
        if (op == kSpvOpTypeImage && wc >= 4 && (w[i + 3] & 0x7u) == kSpvDimRect)
        {
            memcpy(o + j, w + i, wc * sizeof(uint32_t));
            o[j + 3] = (w[i + 3] & ~0x7u) | kSpvDim2D;
            j += wc;
            patched++;
        }
        else if (op == kSpvOpName && wc >= 2 && w[i + 1] < n && ids[w[i + 1]] == 4)
        {
            /* rebuild the literal with the suffix, keeping the padding rules
             * (NUL-terminated, zero-padded to a word multiple) */
            const char *str = (const char *)(w + i + 2);
            size_t slen = strnlen(str, (wc - 2) * 4);
            size_t newlen = slen + strlen("_mglrect");
            size_t newwords = 2 + (newlen / 4 + 1);
            if (j + newwords <= need)
            {
                o[j] = (uint32_t)((newwords << 16) | kSpvOpName);
                o[j + 1] = w[i + 1];
                memset(o + j + 2, 0, (newwords - 2) * sizeof(uint32_t));
                memcpy(o + j + 2, str, slen);
                memcpy((char *)(o + j + 2) + slen, "_mglrect", strlen("_mglrect"));
                j += newwords;
            }
            else
            {
                memcpy(o + j, w + i, wc * sizeof(uint32_t));
                j += wc;
            }
        }
        else
        {
            memcpy(o + j, w + i, wc * sizeof(uint32_t));
            j += wc;
        }
        i += wc;
    }
    free(ids);
    *out_words = o;
    *out_n = j;
    return patched;
}

/* the same rect→2D patch for modules compiled outside compile_stage (the geometry
 * domain's generated kernels share program.c's DimRect limitation) */
size_t mgl_program_spirv_rect_to_2d(const uint32_t *w, size_t n, uint32_t **out_words, size_t *out_n)
{
    return mgl_spirv_rect_to_2d(w, n, out_words, out_n);
}

/* GL texture target a sampler/image uniform of this SPIR-V type samples: the
 * GLSL sampler dimension picks WHICH of the unit's per-target bindings feeds it
 * (sampler2DMS -> GL_TEXTURE_2D_MULTISAMPLE, sampler2DArray -> GL_TEXTURE_2D_ARRAY,
 * ...).  0 when the type is not an image. */
static GLuint sampler_gl_target(spvc_compiler c, spvc_type_id type_id)
{
    spvc_type t = spvc_compiler_get_type_handle(c, type_id);
    spvc_basetype bt = spvc_type_get_basetype(t);
    if (bt != SPVC_BASETYPE_IMAGE && bt != SPVC_BASETYPE_SAMPLED_IMAGE)
        return 0;

    bool arrayed = spvc_type_get_image_arrayed(t);
    bool ms = spvc_type_get_image_multisampled(t);

    switch (spvc_type_get_image_dimension(t))
    {
        case SpvDim1D:  return GL_TEXTURE_1D;
        case SpvDim2D:  return ms ? (arrayed ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_MULTISAMPLE)
                                   : (arrayed ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D);
        case SpvDim3D:  return GL_TEXTURE_3D;
        case SpvDimCube:return arrayed ? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP;
        case SpvDimRect:return GL_TEXTURE_RECTANGLE;
        case SpvDimBuffer: return GL_TEXTURE_BUFFER;
        default: break;
    }
    return 0;
}

static bool add_sampler(MGLProgramObj *obj, GLuint unit, GLuint key, int stage, const char *name)
{
    MGLSamplerUse *list = (MGLSamplerUse *)realloc(obj->samplers, (obj->num_samplers + 1) * sizeof(MGLSamplerUse));
    if (!list)
        return false;
    obj->samplers = list;
    list[obj->num_samplers].unit = unit;
    list[obj->num_samplers].type_key = key;
    list[obj->num_samplers].stage = stage;
    list[obj->num_samplers].name = strdup(name ? name : "");
    obj->num_samplers++;
    return true;
}

static void msl_entry_name(int stage, GLuint program, char *out, size_t len)
{
    switch (stage)
    {
        case _VERTEX_SHADER: snprintf(out, len, "vertex_%u_main", program); break;
        case _TESS_CONTROL_SHADER: snprintf(out, len, "tess_control_%u_main", program); break;
        case _TESS_EVALUATION_SHADER: snprintf(out, len, "tess_evaluation_%u_main", program); break;
        case _GEOMETRY_SHADER: snprintf(out, len, "geometry_%u", program); break;
        case _FRAGMENT_SHADER: snprintf(out, len, "fragment_%u", program); break;
        default: snprintf(out, len, "compute_%u", program); break;
    }
}

/* growable SPIR-V word buffer */
typedef struct { uint32_t *w; size_t n, cap; bool oom; } WordBuf;

static void wb_push(WordBuf *b, uint32_t v)
{
    if (b->n == b->cap)
    {
        size_t ncap = b->cap ? b->cap * 2 : 1024;
        uint32_t *nw = (uint32_t *)realloc(b->w, ncap * sizeof(uint32_t));
        if (!nw) { b->oom = true; return; }
        b->w = nw;
        b->cap = ncap;
    }
    b->w[b->n++] = v;
}

static void wb_push_n(WordBuf *b, const uint32_t *w, size_t n)
{
    for (size_t i = 0; i < n; i++)
        wb_push(b, w[i]);
}

/*
 * Prepare a GL SPIR-V module for SPIRV-Cross MSL:
 *  - freeze specialization constants (SpecializeShader / constant_id defaults are already in
 *    the module): OpSpecConstant* → OpConstant*, SpecId decorations dropped, so MSL gets plain
 *    constants instead of Metal function constants that would need MTLFunctionConstantValues;
 *  - vertex stage: GL's gl_InstanceID excludes the base instance but Metal's [[instance_id]]
 *    includes it, so every load of the InstanceId builtin becomes (InstanceId - BaseInstance).
 *    gl_VertexID (VertexId) is left alone: GL and Metal [[vertex_id]] both include base vertex.
 * Returns a malloc'd module, or NULL on malformed input / OOM.
 */
static uint32_t *spirv_prepare_for_msl(const uint32_t *in, size_t n, int stage, size_t *out_n)
{    static const char draw_params_ext[] = "SPV_KHR_shader_draw_parameters";
    uint32_t bound, inst_var = 0, base_var = 0, inst_ptr_type = 0, inst_type = 0;
    bool has_cap = false, has_ext = false, add_base = false, cap_section_done = false;
    WordBuf b = {0};
    size_t pos;

    if (n < 5 || in[0] != SpvMagicNumber)
        return NULL;
    bound = in[3];

    /* pass 1: find the InstanceId / BaseInstance builtins */
    for (pos = 5; pos < n; )
    {
        uint32_t op = in[pos] & 0xFFFF, wc = in[pos] >> 16;
        if (wc == 0 || pos + wc > n)
            return NULL;
        if (op == SpvOpDecorate && wc >= 4 && in[pos + 2] == SpvDecorationBuiltIn)
        {
            if (in[pos + 3] == SpvBuiltInInstanceId) inst_var = in[pos + 1];
            if (in[pos + 3] == SpvBuiltInBaseInstance) base_var = in[pos + 1];
        }
        else if (op == SpvOpCapability && wc == 2 && in[pos + 1] == SpvCapabilityDrawParameters)
            has_cap = true;
        else if (op == SpvOpExtension && wc >= 2 && !strncmp((const char *)&in[pos + 1], draw_params_ext, (wc - 1) * 4))
            has_ext = true;
        pos += wc;
    }

    if (stage == _VERTEX_SHADER && inst_var)
    {
        for (pos = 5; pos < n; pos += in[pos] >> 16)
            if ((in[pos] & 0xFFFF) == SpvOpVariable && (in[pos] >> 16) >= 4 && in[pos + 2] == inst_var)
                inst_ptr_type = in[pos + 1];
        for (pos = 5; pos < n; pos += in[pos] >> 16)
            if ((in[pos] & 0xFFFF) == SpvOpTypePointer && (in[pos] >> 16) == 4 && in[pos + 1] == inst_ptr_type)
                inst_type = in[pos + 3];
        if (inst_ptr_type && inst_type)
        {
            if (!base_var) { base_var = bound++; add_base = true; }
        }
        else
            inst_var = 0;
    }
    else
        inst_var = 0;

    wb_push_n(&b, in, 5);

    /* pass 2: rewrite */
    for (pos = 5; pos < n; )
    {
        uint32_t op = in[pos] & 0xFFFF, wc = in[pos] >> 16;
        const uint32_t *ins = &in[pos];

        if (!cap_section_done && op != SpvOpCapability)
        {
            cap_section_done = true;
            if (inst_var && !has_cap)
            {
                wb_push(&b, (2u << 16) | SpvOpCapability);
                wb_push(&b, SpvCapabilityDrawParameters);
            }
            if (inst_var && !has_ext)
            {
                uint32_t sw[8] = {0};
                memcpy(sw, draw_params_ext, sizeof(draw_params_ext));
                wb_push(&b, ((1u + 8u) << 16) | SpvOpExtension);
                wb_push_n(&b, sw, 8);
            }
        }

        switch (op)
        {
            case SpvOpDecorate:
                if (wc >= 4 && ins[2] == SpvDecorationSpecId)
                    break;                                      /* frozen: drop SpecId */
                wb_push_n(&b, ins, wc);
                if (add_base && wc >= 4 && ins[1] == inst_var && ins[2] == SpvDecorationBuiltIn)
                {
                    wb_push(&b, (4u << 16) | SpvOpDecorate);
                    wb_push(&b, base_var);
                    wb_push(&b, SpvDecorationBuiltIn);
                    wb_push(&b, SpvBuiltInBaseInstance);
                }
                break;

            case SpvOpSpecConstantTrue:
                wb_push(&b, (wc << 16) | SpvOpConstantTrue); wb_push_n(&b, ins + 1, wc - 1); break;
            case SpvOpSpecConstantFalse:
                wb_push(&b, (wc << 16) | SpvOpConstantFalse); wb_push_n(&b, ins + 1, wc - 1); break;
            case SpvOpSpecConstant:
                wb_push(&b, (wc << 16) | SpvOpConstant); wb_push_n(&b, ins + 1, wc - 1); break;
            case SpvOpSpecConstantComposite:
                wb_push(&b, (wc << 16) | SpvOpConstantComposite); wb_push_n(&b, ins + 1, wc - 1); break;

            case SpvOpEntryPoint:
                if (add_base && wc >= 3 && ins[1] == SpvExecutionModelVertex)
                {
                    wb_push(&b, ((wc + 1) << 16) | SpvOpEntryPoint);
                    wb_push_n(&b, ins + 1, wc - 1);
                    wb_push(&b, base_var);                       /* interface list */
                }
                else
                    wb_push_n(&b, ins, wc);
                break;

            case SpvOpVariable:
                wb_push_n(&b, ins, wc);
                if (add_base && wc >= 4 && ins[2] == inst_var)
                {
                    wb_push(&b, (4u << 16) | SpvOpVariable);
                    wb_push(&b, inst_ptr_type);
                    wb_push(&b, base_var);
                    wb_push(&b, SpvStorageClassInput);
                }
                break;

            case SpvOpLoad:
                if (inst_var && wc >= 4 && ins[3] == inst_var)
                {
                    uint32_t raw = bound++, basev = bound++;
                    wb_push(&b, ins[0]); wb_push(&b, ins[1]); wb_push(&b, raw); wb_push_n(&b, ins + 3, wc - 3);
                    wb_push(&b, (4u << 16) | SpvOpLoad); wb_push(&b, inst_type); wb_push(&b, basev); wb_push(&b, base_var);
                    wb_push(&b, (5u << 16) | SpvOpISub); wb_push(&b, inst_type); wb_push(&b, ins[2]); wb_push(&b, raw); wb_push(&b, basev);
                }
                else
                    wb_push_n(&b, ins, wc);
                break;

            default:
                wb_push_n(&b, ins, wc);
                break;
        }

        pos += wc;
    }

    if (b.oom)
    {
        free(b.w);
        return NULL;
    }

    b.w[3] = bound;
    *out_n = b.n;
    return b.w;
}

/* transform_feedback.c compiles its capture variant as a lone vertex stage;
 * route it through the same default-uniform-block packing as every other stage
 * so loose uniforms do not become constant address-space entry-point arguments
 * (soft-float64 calls take `thread` references and cannot bind those). */
uint32_t *mgl_program_pack_stage_for_msl(GLMContext ctx, Program *program,
                                         const uint32_t *in, size_t n, size_t *out_n)
{
    size_t pn = 0;
    uint32_t *patched = mgl_introspect_patch_stage(ctx, program, _VERTEX_SHADER, in, n, &pn);
    if (!patched) return NULL;
    uint32_t *prep = spirv_prepare_for_msl(patched, pn, _VERTEX_SHADER, out_n);
    free(patched);
    return prep;
}

/*
 * Translate one stage's SPIR-V module into the executable of dst:
 * SPIRV-Cross reflection + MSL, then a Metal library/function (errors go to *log).
 */
static bool compile_stage(GLMContext ctx, MGLProgramObj *dst, int stage, const uint32_t *words,
                          size_t nwords, const char *entry, ReflectAcc *acc, char **log,
                          bool tess_pipeline)
{
    Program *ptr = &dst->base;
    spvc_context context = NULL;
    spvc_parsed_ir ir = NULL;
    spvc_compiler compiler = NULL;
    spvc_compiler_options options = NULL;
    spvc_resources resources = NULL;
    SpvExecutionModel model = stage_model(stage);
    const char *result = NULL;
    char *tess_owned = NULL;
    char msl_entry[128];
    uint32_t *prepared = NULL;
    size_t prepared_words = 0;
    bool ok = false;

    if (!words || nwords < 5 || words[0] != SpvMagicNumber)
    {
        mgl_append_info_log(log, "ERROR: %s stage: invalid SPIR-V module\n", stage_label(stage));
        return false;
    }

    /* introspect: default uniform block, attribute / fragment-output locations, buffer order */
    uint32_t *patched = mgl_introspect_patch_stage(ctx, ptr, stage, words, nwords, &prepared_words);
    if (patched)
    {
        words = patched;
        nwords = prepared_words;
    }

    ptr->spirv[stage].stage = stage;
    ptr->spirv[stage].size = nwords;
    ptr->spirv[stage].dub_msl = 0xffffffffu;   /* geom: set by mgl_msl_fix_image_atomics */
    ptr->spirv[stage].ir = (unsigned int *)malloc(nwords * sizeof(uint32_t));
    if (!ptr->spirv[stage].ir)
    {
        mgl_append_info_log(log, "ERROR: out of memory\n");
        return false;
    }
    memcpy(ptr->spirv[stage].ir, words, nwords * sizeof(uint32_t));
    dst->stage_entry[stage] = strdup(entry);

    prepared = spirv_prepare_for_msl(words, nwords, stage, &prepared_words);
    if (!prepared)
    {
        mgl_append_info_log(log, "ERROR: %s stage: malformed SPIR-V instruction stream\n", stage_label(stage));
        return false;
    }

    if (stage == _VERTEX_SHADER)
    {
        /* a vertex input location overlap is a link error (GL 4.6, 11.1.1); the
           fp64 lowering would otherwise emit two Metal attributes at one index */
        MGLIProgram *ip = mgli_find(ctx, ptr->name);
        if (ip && ip->attrib_conflict)
        {
            mgl_append_info_log(log, "ERROR: vertex stage: conflicting vertex attribute locations\n");
            free(prepared);
            return false;
        }
        /* more attribute vectors than MAX_VERTEX_ATTRIBS (dvec3/dvec4 count two,
           ARB_vertex_attrib_64bit) — the slots exceed the Metal attribute limit */
        if (ip && ip->attrib_overflow)
        {
            mgl_append_info_log(log, "ERROR: vertex stage: too many vertex attribute components\n");
            free(prepared);
            return false;
        }
    }

    bool lowered64 = false;

    /* fp64: lower software doubles (u64 bit patterns + soft-float calls).
     * ptr->spirv[stage].ir above keeps the GL view (real double types) for
     * introspection; everything MSL-facing uses the lowered module. */
    {
        char fp64log[512] = {0};
        size_t lower_words = 0;
        uint32_t *lowered = mgl_fp64_lower_module(prepared, prepared_words, stage == _VERTEX_SHADER,
                                                  &lowered64, &lower_words,
                                                  fp64log, sizeof fp64log);
        if (lowered64 && lowered)
        {
            free(prepared);
            prepared = lowered;
            prepared_words = lower_words;
        }
        else if (lowered64)
        {
            mgl_append_info_log(log, "ERROR: %s stage: %s\n", stage_label(stage),
                                fp64log[0] ? fp64log : "fp64 lowering failed");
            free(prepared);
            return false;
        }
    }

    /* tess: isoline tessellation is emulated on the quad domain (see mgl_ext_tess.h) */
    if (stage == _TESS_EVALUATION_SHADER)
    {
        size_t tess_words = 0;
        uint32_t *tess_ir = mgl_tess_prepare_spirv(prepared, prepared_words, &tess_words);
        if (tess_ir)
        {
            free(prepared);
            prepared = tess_ir;
            prepared_words = tess_words;
        }
    }

    /* GL rectangle textures: SPIRV-Cross's MSL backend has no DimRect mapping
     * (it emits `unknown_texture_type`, which never compiles), so the MSL-facing
     * module rewrites the image Dim of every OpTypeImage from Rect to 2D.  The
     * unnormalized coordinates come from the sampler (texparamCreateSampler
     * builds it from the RECTANGLE target) and a Metal texture2d with an
     * unnormalized sampler is exactly rectangle semantics.  The GL view in
     * ptr->spirv[stage].ir keeps DimRect, so introspection still reports
     * GL_TEXTURE_RECTANGLE and the unit lookup finds the right binding. */
    {
        uint32_t *rect_fixed = NULL;
        size_t rect_words = 0;
        if (mgl_spirv_rect_to_2d(prepared, prepared_words, &rect_fixed, &rect_words) > 0 && rect_fixed)
        {
            free(prepared);
            prepared = rect_fixed;
            prepared_words = rect_words;
        }
        else
        {
            free(rect_fixed);
        }
    }

    if (spvc_context_create(&context) != SPVC_SUCCESS)
    {
        mgl_append_info_log(log, "ERROR: SPIRV-Cross context creation failed\n");
        free(prepared);
        return false;
    }
    spvc_context_set_error_callback(context, spvc_log_cb, log);

    if (spvc_context_parse_spirv(context, prepared, prepared_words, &ir) != SPVC_SUCCESS ||
        spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) != SPVC_SUCCESS)
    {
        mgl_append_info_log(log, "ERROR: %s stage: SPIR-V module could not be parsed\n", stage_label(stage));
        goto done;
    }

    if (spvc_compiler_msl_add_discrete_descriptor_set(compiler, 3) != SPVC_SUCCESS ||
        spvc_compiler_create_compiler_options(compiler, &options) != SPVC_SUCCESS ||
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) != SPVC_SUCCESS ||
        spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3,1,0)) != SPVC_SUCCESS ||
        /* texext: samplerBuffer/imageBuffer -> MSL texture_buffer (bound as MTLTextureTypeTextureBuffer) */
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, SPVC_TRUE) != SPVC_SUCCESS ||
        /* texdata: 1D textures are realized as 2D Metal textures (Metal 1D textures cannot be mipmapped) */
        spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D, SPVC_TRUE) != SPVC_SUCCESS ||
        spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS)
    {
        mgl_append_info_log(log, "ERROR: %s stage: SPIRV-Cross MSL options rejected\n", stage_label(stage));
        goto done;
    }

    if (spvc_compiler_set_entry_point(compiler, entry, model) != SPVC_SUCCESS)
    {
        mgl_append_info_log(log, "ERROR: %s stage: entry point \"%s\" not found in SPIR-V module\n", stage_label(stage), entry);
        goto done;
    }

    msl_entry_name(stage, ptr->name, msl_entry, sizeof(msl_entry));
    {
        const char *cleansed = spvc_compiler_get_cleansed_entry_point_name(compiler, entry, model);
        if (spvc_compiler_rename_entry_point(compiler, cleansed ? cleansed : entry, msl_entry, model) != SPVC_SUCCESS)
        {
            mgl_append_info_log(log, "ERROR: %s stage: cannot rename entry point \"%s\"\n", stage_label(stage), entry);
            goto done;
        }
    }

    /* tess: the vertex stage of a tessellation pipeline becomes a compute kernel writing
       its outputs to a buffer; the TCS becomes a multi-patch compute kernel */
    if (stage == _VERTEX_SHADER && tess_pipeline)
    {
        if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_VERTEX_FOR_TESSELLATION, SPVC_TRUE) != SPVC_SUCCESS ||
            spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER, SPVC_TRUE) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX, MGL_TESS_VS_OUT_BUF) != SPVC_SUCCESS ||
            spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS)
        {
            mgl_append_info_log(log, "ERROR: %s stage: tessellation MSL options rejected\n", stage_label(stage));
            goto done;
        }
    }
    else if (stage == _TESS_CONTROL_SHADER)
    {
        if (spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_MULTI_PATCH_WORKGROUP, SPVC_TRUE) != SPVC_SUCCESS ||
            spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER, SPVC_TRUE) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_SHADER_INPUT_BUFFER_INDEX, MGL_TESS_TCS_IN) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX, MGL_TESS_TCS_OUT) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_SHADER_PATCH_OUTPUT_BUFFER_INDEX, MGL_TESS_TCS_PATCH_OUT) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_SHADER_TESS_FACTOR_OUTPUT_BUFFER_INDEX, MGL_TESS_TCS_FACTORS) != SPVC_SUCCESS ||
            spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_INDIRECT_PARAMS_BUFFER_INDEX, MGL_TESS_TCS_PARAMS) != SPVC_SUCCESS ||
            spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS)
        {
            mgl_append_info_log(log, "ERROR: %s stage: tessellation MSL options rejected\n", stage_label(stage));
            goto done;
        }
    }

    if (stage == _COMPUTE_SHADER)
    {
        ptr->local_workgroup_size.x = spvc_compiler_get_execution_mode_argument_by_index(compiler, SpvExecutionModeLocalSize, 0);
        ptr->local_workgroup_size.y = spvc_compiler_get_execution_mode_argument_by_index(compiler, SpvExecutionModeLocalSize, 1);
        ptr->local_workgroup_size.z = spvc_compiler_get_execution_mode_argument_by_index(compiler, SpvExecutionModeLocalSize, 2);

        /* geom: the compute stage owns the whole buffer table except slot 30, so
         * move SPIRV-Cross's buffer-size constants (default 25, shared with the
         * render stages) out of the program's argument range — up to 29 program
         * buffer arguments must fit (KHR-GL46.compute_shader.resources-max) */
        if (spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_BUFFER_SIZE_BUFFER_INDEX,
                                           MGL_CS_SSBO_SIZE_BUFFER_INDEX) != SPVC_SUCCESS ||
            spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS)
        {
            mgl_append_info_log(log, "ERROR: %s stage: SPIRV-Cross MSL options rejected\n", stage_label(stage));
            goto done;
        }
    }

    /* reflection */
    if (spvc_compiler_create_shader_resources(compiler, &resources) != SPVC_SUCCESS)
    {
        mgl_append_info_log(log, "ERROR: %s stage: SPIR-V reflection failed\n", stage_label(stage));
        goto done;
    }

    /* atomic counters: names + SPIR-V offsets, for the MSL argument injection below
     * (SPIRV-Cross declares no atomic_uint arguments at all) */
    MGLAtomicCounterVar atvars[MGL_MAX_ATOMIC_VARS];
    int natvars = 0;

    for (int res_type = SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; res_type < SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE && res_type < _MAX_SPIRV_RES; res_type++)
    {
        const spvc_reflected_resource *list = NULL;
        size_t count = 0;

        spvc_resources_get_resource_list_for_type(resources, res_type, &list, &count);

        ptr->spirv_resources_list[stage][res_type].count = (GLuint)count;
        if (count == 0)
            continue;

        ptr->spirv_resources_list[stage][res_type].list = (SpirvResource *)calloc(count, sizeof(SpirvResource));
        if (!ptr->spirv_resources_list[stage][res_type].list)
        {
            ptr->spirv_resources_list[stage][res_type].count = 0;
            mgl_append_info_log(log, "ERROR: out of memory\n");
            goto done;
        }

        for (size_t i = 0; i < count; i++)
        {
            SpirvResource *r = &ptr->spirv_resources_list[stage][res_type].list[i];
            const char *rname = list[i].name ? list[i].name : "";
            spvc_type rtype = spvc_compiler_get_type_handle(compiler, list[i].type_id);
            unsigned n = array_size(rtype);
            char name[512];

            r->_id = list[i].id;
            r->base_type_id = list[i].base_type_id;
            r->type_id = list[i].type_id;
            r->name = strdup(rname);
            r->set = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationDescriptorSet);
            r->binding = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationBinding);
            r->msl_binding = r->binding; /* refined with the real Metal index after MSL emission */
            r->array_size = n;           /* sampler/image arrays: consecutive units + argument slots */
            r->location = spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationLocation);
            r->msl_atomic = 0xffffffffu;  /* geom: filled for storage images after MSL emission */
            if (res_type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE || res_type == SPVC_RESOURCE_TYPE_SEPARATE_IMAGE ||
                res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE)
            {
                r->target = sampler_gl_target(compiler, list[i].type_id);
                /* the image's sampled type (float / int / uint) picks the Metal texture
                 * type SPIRV-Cross declares; geomImageUnitView:elem_class: reinterprets
                 * the bound texture when the image unit's format disagrees with it */
                if (res_type == SPVC_RESOURCE_TYPE_STORAGE_IMAGE && rtype)
                {
                    spvc_basetype bt = spvc_type_get_basetype(rtype);

                    if (bt == SPVC_BASETYPE_IMAGE)
                    {
                        /* an image's type is OpTypeImage: the element kind is the SAMPLED
                         * type's.  spvc_type_get_image_sampled_type returns the sampled
                         * type's SPIR-V <id>, NOT a basetype — comparing it against
                         * SPVC_BASETYPE_* misreads whatever type happens to carry that id
                         * (KHR-GL46.shader_image_load_store multiple-uniforms: the float
                         * type's id 8 == SPVC_BASETYPE_UINT32 turned every float image
                         * into a uint one and the unit-format views off) */
                        spvc_type_id stid = spvc_type_get_image_sampled_type(rtype);
                        spvc_type stype = spvc_compiler_get_type_handle(compiler, stid);

                        bt = stype ? spvc_type_get_basetype(stype) : SPVC_BASETYPE_UNKNOWN;
                    }
                    r->elem_class = (bt == SPVC_BASETYPE_INT32 || bt == SPVC_BASETYPE_INT64) ? 1u :
                                    (bt == SPVC_BASETYPE_UINT32 || bt == SPVC_BASETYPE_UINT64) ? 2u : 0u;
                }
            }

            switch (res_type)
            {
                case SPVC_RESOURCE_TYPE_STAGE_INPUT:
                    if (stage == _VERTEX_SHADER && strncmp(rname, "gl_", 3))
                    {
                        dst->active_attributes++;
                        if ((GLint)strlen(rname) + 1 > dst->active_attribute_max_length)
                            dst->active_attribute_max_length = (GLint)strlen(rname) + 1;
                    }
                    break;

                case SPVC_RESOURCE_TYPE_UNIFORM_BUFFER:
                {
                    const char *block = spvc_compiler_get_name(compiler, list[i].base_type_id);
                    if (!block || !*block)
                        block = rname;

                    if (!strcmp(block, "gl_DefaultUniformBlock"))
                    {
                        reflect_members(compiler, list[i].base_type_id, "", acc, 0);
                    }
                    else
                    {
                        snprintf(name, sizeof(name), "%s.", block);
                        reflect_members(compiler, list[i].base_type_id, name, acc, 0);
                        if (n == 0)
                            name_list_add(&acc->blocks, &acc->num_blocks, &acc->cap_blocks, block);
                        for (unsigned e = 0; e < n && e < 64; e++)
                        {
                            snprintf(name, sizeof(name), "%s[%u]", block, e);
                            name_list_add(&acc->blocks, &acc->num_blocks, &acc->cap_blocks, name);
                        }
                    }
                    break;
                }

                case SPVC_RESOURCE_TYPE_SAMPLED_IMAGE:
                case SPVC_RESOURCE_TYPE_SEPARATE_IMAGE:
                    for (unsigned e = 0; e < (n ? n : 1); e++)
                        add_sampler(dst, r->binding + e, sampler_type_key(compiler, list[i].type_id), stage, rname);
                    /* fall through: samplers are uniforms */
                case SPVC_RESOURCE_TYPE_STORAGE_IMAGE:
                case SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT:
                    snprintf(name, sizeof(name), n ? "%s[0]" : "%s", rname);
                    name_list_add(&acc->uniforms, &acc->num_uniforms, &acc->cap_uniforms, name);
                    break;

                case SPVC_RESOURCE_TYPE_ATOMIC_COUNTER:
                {
                    int known = 0;
                    snprintf(name, sizeof(name), n ? "%s[0]" : "%s", rname);
                    name_list_add(&acc->uniforms, &acc->num_uniforms, &acc->cap_uniforms, name);
                    for (int k = 0; k < acc->num_atomic; k++)
                        if (acc->atomic_bindings[k] == r->binding) known = 1;
                    if (!known && acc->num_atomic < 64)
                        acc->atomic_bindings[acc->num_atomic++] = r->binding;
                    /* remember name + offset for the MSL argument injection */
                    if (natvars < MGL_MAX_ATOMIC_VARS && strlen(rname) < sizeof atvars[0].name_buf)
                    {
                        MGLAtomicCounterVar *v = &atvars[natvars++];
                        memset(v, 0, sizeof *v);
                        v->name = v->name_buf;
                        snprintf(v->name_buf, sizeof v->name_buf, "%s", rname);
                        v->offset = (unsigned)spvc_compiler_get_decoration(compiler, list[i].id, SpvDecorationOffset);
                    }
                    break;
                }

                default:
                    break;
            }
        }
    }

    /* KHR-GL46.shader_image_load_store uniform-limits: glslang does not enforce
     * max_*_image_uniforms for every stage (a 9th geometry image uniform links fine),
     * so reject a stage that declares more image uniforms than the query it serves —
     * GL 4.6 §11.1.3.11 "link will fail if ... the number of active image uniforms
     * referenced by any shader stage exceeds the limit for that stage".  The combined
     * limit is checked once all stages are linked (mgl_link_program). */
    {
        static const GLenum stage_img_limit[_MAX_SPIRV_RES] = {
            [_VERTEX_SHADER]           = GL_MAX_VERTEX_IMAGE_UNIFORMS,
            [_TESS_CONTROL_SHADER]     = GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS,
            [_TESS_EVALUATION_SHADER]  = GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS,
            [_GEOMETRY_SHADER]         = GL_MAX_GEOMETRY_IMAGE_UNIFORMS,
            [_FRAGMENT_SHADER]         = GL_MAX_FRAGMENT_IMAGE_UNIFORMS,
            [_COMPUTE_SHADER]          = GL_MAX_COMPUTE_IMAGE_UNIFORMS,
        };
        GLuint imgs = 0;
        SpirvResourceList *il = &ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];

        for (GLuint i = 0; i < il->count; i++)
            imgs += il->list[i].array_size ? il->list[i].array_size : 1;
        if (stage_img_limit[stage])
        {
            GLint lim = 0;

            ctx->dispatch.get_integerv(ctx, stage_img_limit[stage], &lim);
            if (lim > 0 && imgs > (GLuint)lim)
            {
                mgl_append_info_log(log,
                    "ERROR: %s stage: %u image uniforms exceed the limit of %d\n",
                    stage_label(stage), imgs, lim);
                goto done;
            }
        }
    }

    /* GLSL identifiers that are MSL keywords/types (e.g. `uniform sampler2D sampler;`)
     * must be renamed before MSL emission; introspection above already read the GLSL
     * names, which must stay the ones the GL API reports. */
    mgl_msl_mangle_reserved(compiler, prepared, prepared_words);

    if (stage == _GEOMETRY_SHADER)
    {
        /* SPIRV-Cross has no geometry → MSL backend.  The reflection above is kept (so
         * uniform / texture / atomic introspection is unchanged); the geometry domain
         * builds the emulation for this stage: the geometry kernel (which becomes this
         * stage's executable), a vertex capture variant and the pass-through raster
         * function that replaces the vertex stage at post-link. */
        if (!mgl_geom_compile_stage(ctx, ptr, dst, stage, prepared, prepared_words, entry, log))
            goto done;

        {
            Shader *exec = mgl_shader_new_internal(ctx, stage);
            if (!exec)
            {
                mgl_append_info_log(log, "ERROR: out of memory\n");
                goto done;
            }
            exec->entry_point = strdup(msl_entry);
            ptr->shader_slots[stage] = exec;
        }
        dst->linked_mask |= SHADER_MASK_BIT(stage);
        ok = true;
        goto done;
    }

    if (spvc_compiler_compile(compiler, &result) != SPVC_SUCCESS || !result)
    {
        mgl_append_info_log(log, "ERROR: %s stage: SPIR-V → MSL translation failed\n", stage_label(stage));
        goto done;
    }

    /* tess: rewrite the generated MSL of the tessellation stages (tessellation.c) */
    if (stage == _TESS_CONTROL_SHADER || stage == _TESS_EVALUATION_SHADER ||
        (stage == _VERTEX_SHADER && tess_pipeline))
    {
        char *tess_msl = mgl_tess_rewrite_stage_msl(ctx, stage, tess_pipeline, result);
        if (tess_msl)
        {
            result = tess_msl;     /* the spvc context owns the original string */
            tess_owned = tess_msl;
        }
    }

    /* geom: SPIRV-Cross maps GLSL `coherent`/`volatile` SSBO qualifiers to a `volatile`
     * address-space qualifier in the MSL parameter declarations (`volatile device
     * Block& ...`).  MSL cannot parse that qualifier order (basic-syntax's coherent
     * restrict volatile block), and a volatile block reference makes member matrices
     * lose their operator[] (advanced-matrix).  Neither qualifier is expressible on
     * Metal anyway - cross-invocation ordering is provided by MGL's encoder
     * serialization (glMemoryBarrier) - so the qualifier is dropped (geometry.c). */
    {
        char *stripped = mgl_geom_strip_msl_volatile(result);
        if (stripped)
        {
            free(tess_owned);
            tess_owned = stripped;
            result = stripped;
        }
    }

    /* atomic counters: declare the atomic_uint arguments (geometry.c — see the
     * buffer-index scheme there) at the ordinals the renderer's buffer map binds
     * the atomic-counter group at: the map's uniform-block + default-uniform +
     * shader-storage total (declared blocks count, even when SPIRV-Cross never
     * emitted an argument for an unused one).  Render stages must stay below 24
     * (the tessellation / xfb / geometry emulation pins); the compute stage may
     * span to 29 — its only reserved slot is the buffer-size slot at 30 */
    int atidx[MGL_MAX_ATOMIC_VARS];
    if (natvars > 0)
    {
        /* the map's ordinal for the atomic-counter group: one entry per element
         * of every uniform-block / default-uniform / shader-storage resource —
         * block instance arrays (uniform UB {..} ub[12]) expand to one entry
         * each, see mapGLBuffersToMTLBufferMap */
        int ac_base = 0;
        for (int t = 0; t < 3; t++)
        {
            int type = t == 0 ? SPVC_RESOURCE_TYPE_UNIFORM_BUFFER
                    : t == 1 ? SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT
                             : SPVC_RESOURCE_TYPE_STORAGE_BUFFER;
            for (GLuint i = 0; i < ptr->spirv_resources_list[stage][type].count; i++)
            {
                unsigned as = ptr->spirv_resources_list[stage][type].list[i].array_size;
                ac_base += as ? (int)as : 1;
            }
        }
        char *injected = mgl_msl_inject_atomic_counters(result, msl_entry, atvars, natvars,
                                                        stage == _COMPUTE_SHADER ?
                                                        MGL_CS_SSBO_SIZE_BUFFER_INDEX : 24,
                                                        ac_base, atidx);
        if (!injected)
        {
            mgl_append_info_log(log, "ERROR: %s stage: atomic counter buffer arguments could not be "
                                     "declared (out of Metal buffer indices?)\n", stage_label(stage));
            goto done;
        }
        if (tess_owned != result) free(tess_owned);
        tess_owned = injected;
        result = injected;
    }

    /* AGX: gather_compare AND plain gather are broken in vertex and compute
     * functions (raw values / zeros; fragment functions are fine), and in the
     * vertex stage sample_compare itself is broken too.  mgl_fix_gathers_vs
     * rewrites both gather kinds to explicit-tap helpers: plain gathers become
     * texel reads, shadow gathers compare in the shader in the vertex stage
     * (the compare op rides in as a function constant) and keep hardware
     * sample_compare taps in compute, where it works. */
    if (stage == _COMPUTE_SHADER || (stage == _VERTEX_SHADER && tess_pipeline))
    {
        char *fixed_gc = mgl_fix_gathers_vs(result, NULL, true);
        if (fixed_gc)
        {
            if (tess_owned != result) free(tess_owned);
            tess_owned = fixed_gc;
            result = fixed_gc;
        }
    }
    else if (stage == _VERTEX_SHADER)
    {
        uint32_t shadow_slots = 0;
        char *fixed_sh = mgl_fix_gathers_vs(result, &shadow_slots, false);
        if (fixed_sh)
        {
            if (tess_owned != result) free(tess_owned);
            tess_owned = fixed_sh;
            result = fixed_sh;
            ptr->spirv[stage].shadow_slots = shadow_slots;
        }
    }
    else if (stage == _TESS_CONTROL_SHADER || stage == _TESS_EVALUATION_SHADER)
    {
        /* the tessellation stages run as post-tessellation vertex functions /
         * kernels, where the hardware gather is equally broken: same texel-read
         * rewrite (the shadow-compare function constants are not filled for
         * these stages, so only plain gathers are fully covered) */
        char *fixed_gc = mgl_fix_gathers_vs(result, NULL, false);
        if (fixed_gc)
        {
            if (tess_owned != result) free(tess_owned);
            tess_owned = fixed_gc;
            result = fixed_gc;
        }
    }

    /* Apple GPUs: a write through an access::read_write texture (GL image2D) is
     * not visible to a later read of the same texel in the same invocation
     * without tex.fence() — insert it after every such write (geometry.c) */
    {
        char *fenced = mgl_msl_insert_image_fences(result);
        if (fenced)
        {
            if (tess_owned != result) free(tess_owned);
            tess_owned = fenced;
            result = fenced;
        }
    }

    /* image atomics: SPIRV-Cross emulates them with a side `device atomic_*` buffer
     * argument per storage image, whose subscript only image2D linearizes correctly —
     * mgl_msl_fix_image_atomics (geometry.c) fixes the other targets' coordinates and
     * reports the argument slots, which become the resource list's msl_atomic (the
     * renderer binds a zero-initialized sidecar MTLBuffer per image unit there).
     * Render stages must keep the slot below the emulation pins (24), the compute
     * stage below its buffer-size slot (30). */
    {
        MGLGeomTexSlot aslots[32];
        int naslots = 0, dub_msl = -1;
        char *fixed = mgl_msl_fix_image_atomics(result,
                                                stage == _COMPUTE_SHADER ?
                                                    MGL_CS_SSBO_SIZE_BUFFER_INDEX : 24,
                                                -1, aslots, &naslots, 32, &dub_msl);
        if (!fixed)
        {
            mgl_append_info_log(log, "ERROR: %s stage: image-atomic buffer argument slots exceed the "
                                     "Metal buffer index budget\n", stage_label(stage));
            goto done;
        }
        if (naslots > 0)
        {
            SpirvResourceList *srl = &ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];
            for (GLuint i = 0; i < srl->count; i++)
            {
                srl->list[i].msl_atomic = 0xffffffffu;
                for (int k = 0; k < naslots; k++)
                    if (srl->list[i].name && !strcmp(srl->list[i].name, aslots[k].name))
                    { srl->list[i].msl_atomic = (GLuint)aslots[k].msl; break; }
                if (srl->list[i].msl_atomic == 0xffffffffu && getenv("MGL_GEOM_DEBUG") && srl->list[i].name)
                    fprintf(stderr, "MGL geom: %s stage image %s: no _atomic argument matched\n",
                            stage_label(stage), srl->list[i].name);
            }
        }
        /* the packed default-uniform block's real argument index (also meaningful when
         * the stage has no image atomics — 0xffffffff keeps the map's ordinal) */
        if (dub_msl >= 0)
            ptr->spirv[stage].dub_msl = (uint32_t)dub_msl;
        if (tess_owned != result) free(tess_owned);
        tess_owned = fixed;
        result = fixed;
    }

    /* SPIRV-Cross's array-copy helpers cannot bind spvUnsafeArray struct members
     * (whole-array assignment to a shared variable's member, shared-max) */
    {
        char *copied = mgl_msl_add_unsafearray_copies(result);
        if (copied)
        {
            if (tess_owned != result) free(tess_owned);
            tess_owned = copied;
            result = copied;
        }
    }

    /* SPIRV-Cross gives soft-float64 library functions `thread const T&`
     * scalar parameters; a `constant`-space buffer member cannot bind to
     * those.  mgl_fp64_fix_msl_scalar_params() takes them by value instead —
     * functions that assign to their parameters (mgl_fdiv64 rescales its
     * operands) also get plain `thread T&` refs — by-value is the GLSL
     * semantics either way. */
    if (lowered64)
    {
        char *fixed = mgl_fp64_fix_msl_scalar_params(result);
        if (fixed)
            ptr->spirv[stage].msl_str = fixed;
        else
            ptr->spirv[stage].msl_str = strdup(result);
    }
    else
    {
        ptr->spirv[stage].msl_str = strdup(result);
    }
    if (getenv("MGL_FP64_DUMP") && ptr->spirv[stage].msl_str) {
        char path[256]; snprintf(path, sizeof path, "/tmp/fp64msl%d.metal", stage);
        FILE *f_ = fopen(path, "w");
        if (f_) { fputs(ptr->spirv[stage].msl_str, f_); fclose(f_); }
    }
    if (getenv("MGL_FP64_DUMP") && lowered64 && prepared) {   /* lowered SPIR-V, for spirv-dis */
        char path[256]; snprintf(path, sizeof path, "/tmp/fp64lower%d.spv", stage);
        FILE *f_ = fopen(path, "wb");
        if (f_) { fwrite(prepared, 4, prepared_words, f_); fclose(f_); }
    }
    if (getenv("MGL_DUMP_MSL"))
        fprintf(stderr, "MGL: program %u %s stage MSL:\n%s\n", ptr->name, stage_label(stage), result);
    ptr->spirv[stage].entry_point = strdup(msl_entry);


    /* Metal argument indices are only known after the MSL is emitted: patch the recorded
     * resources with SPIRV-Cross's automatic bindings (texture slots differ from GL units).
     * `resources` is the same (context-owned) reflection object created above. */
    for (int res_type = SPVC_RESOURCE_TYPE_UNIFORM_BUFFER; res_type < SPVC_RESOURCE_TYPE_ACCELERATION_STRUCTURE && res_type < _MAX_SPIRV_RES; res_type++)
    {
        SpirvResourceList *rl = &ptr->spirv_resources_list[stage][res_type];
        const spvc_reflected_resource *list = NULL;
        size_t count = 0;
        spvc_resources_get_resource_list_for_type(resources, res_type, &list, &count);
        for (size_t i = 0; i < count && i < rl->count; i++)
        {
            if (list[i].id != rl->list[i]._id) continue;
            unsigned msl_b = spvc_compiler_msl_get_automatic_resource_binding(compiler, list[i].id);
            /* the automatic binding is the resource's real Metal argument index.
             * UINT32_MAX (returned for a variable SPIRV-Cross stripped from the
             * MSL — declared but unused, e.g. the readonly g_input_image
             * KHR-GL46.compute_shader.pipeline-post-fs pass 1 never reads) is the
             * established "no Metal binding" sentinel (0xffffffffu, see
             * MGLRenderer+TextureExt.m): buffers keep indices < 31, textures < 128 */
            {
                bool is_buffer = res_type == SPVC_RESOURCE_TYPE_UNIFORM_BUFFER ||
                                 res_type == SPVC_RESOURCE_TYPE_UNIFORM_CONSTANT ||
                                 res_type == SPVC_RESOURCE_TYPE_STORAGE_BUFFER ||
                                 res_type == SPVC_RESOURCE_TYPE_ATOMIC_COUNTER;
                unsigned cap = is_buffer ? 31u : 128u;
                rl->list[i].msl_binding = (msl_b < cap) ? msl_b : 0xffffffffu;
            }
            if (getenv("MGL_GEOM_DEBUG"))
                fprintf(stderr, "MGL geom: %s stage resource %s (id %u) auto msl binding %u\n",
                        stage_label(stage), rl->list[i].name, list[i].id, msl_b);
        }
    }

    /* atomic counters: SPIRV-Cross assigned no argument index (it emitted none); the
     * indices chosen by the MSL injection above are the real ones */
    if (natvars > 0)
    {
        SpirvResourceList *arl = &ptr->spirv_resources_list[stage][SPVC_RESOURCE_TYPE_ATOMIC_COUNTER];
        for (GLuint i = 0; i < arl->count && (int)i < natvars; i++)
            arl->list[i].msl_binding = (GLuint)atidx[i];
    }

    /*
     * UBO / SSBO instance arrays (GL: `uniform Block { ... } inst[3];`) are flattened by
     * SPIRV-Cross into one Metal buffer argument per instance (blk_0 [[buffer(b)]],
     * blk_1 [[buffer(b+1)]], ...).  GL addresses every instance separately: one block index
     * per instance, each with its own binding point (the CTS sets binding = block index).
     * The resource list therefore carries one entry per instance, in MSL buffer order —
     * which is the order the renderer's buffer map binds at.  `location` (unused for
     * buffers) carries the instance index so mgli_apply_block_binding() can re-point each
     * entry from the matching MGLIBlock.
     */
    static const int buffer_res_types[2] = { SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, SPVC_RESOURCE_TYPE_STORAGE_BUFFER };
    for (size_t t = 0; t < 2; t++)
    {
        int res_type = buffer_res_types[t];
        SpirvResourceList *rl = &ptr->spirv_resources_list[stage][res_type];
        if (!rl->list) continue;
        for (GLuint i = 0; i < rl->count; i++)
        {
            spvc_type rtype = spvc_compiler_get_type_handle(compiler, rl->list[i].type_id);
            if (!rtype) continue;
            unsigned n = array_size(rtype);
            if (n <= 1) { rl->list[i].location = 0; continue; }
            SpirvResource *grown = (SpirvResource *)realloc(rl->list, (size_t)(rl->count + n - 1) * sizeof(SpirvResource));
            if (!grown) continue;
            rl->list = grown;
            memmove(&rl->list[i + n], &rl->list[i + 1], (size_t)(rl->count - i - 1) * sizeof(SpirvResource));
            rl->count += n - 1;
            SpirvResource base = rl->list[i];
            for (unsigned j = 0; j < n; j++)
            {
                SpirvResource *e = &rl->list[i + j];
                if (j) *e = base;
                e->binding += j;
                e->msl_binding += j;
                e->location = j;
                if (j) e->name = strdup(base.name);   /* each entry owns its name */
            }
            i += n - 1;   /* the copies are already instance entries */
        }
    }

    /* executable stage object read by the renderer */
    {
        Shader *exec = mgl_shader_new_internal(ctx, stage);
        if (!exec)
        {
            mgl_append_info_log(log, "ERROR: out of memory\n");
            goto done;
        }
        exec->entry_point = strdup(msl_entry);
        ptr->shader_slots[stage] = exec;

        /* compile to Metal now so Metal compile errors land in the link log.
         * A TES with more stage-input attributes than the device's slots cannot
         * compile in its raw form — the tess canonical rewrite (which reads the
         * control points from buffers) is the real pipeline and compiles there. */
        if (ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl &&
            !(stage == _TESS_EVALUATION_SHADER &&
              mgl_tess_tes_msl_exceeds_attributes(ptr->spirv[stage].msl_str)))
        {
            char *mtl_err = NULL;
            if (!ctx->mtl_funcs.ext_program->compile_msl(ctx, ptr->spirv[stage].msl_str, msl_entry,
                                                        &exec->mtl_data.library, &exec->mtl_data.function, &mtl_err))
            {
                mgl_append_info_log(log, "ERROR: %s stage: Metal shader compilation failed:\n%s\n",
                                    stage_label(stage), mtl_err ? mtl_err : "(no details)");
                free(mtl_err);
                goto done;
            }
            free(mtl_err);
        }
    }

    dst->linked_mask |= SHADER_MASK_BIT(stage);
    ok = true;

    /* A SEPARABLE vs-only program can feed a tessellation PIPELINE (the TES lives in
     * another program).  The tessellation draw needs this stage as a COMPUTE kernel,
     * which only SPIRV-Cross can produce (vertex-for-tessellation options at MSL
     * generation) — compile that variant now, from the same SPIR-V, and hand it to the
     * tess domain's cache.  The plain vertex function stays the program's own (normal
     * draws of the same program use it). */
    if (getenv("MGL_TESS_DEBUG") && stage == _VERTEX_SHADER)
        fprintf(stderr, "MGL tess: sep-vs? stage=%d tess_pipe=%d sep=%d has_ep=%d\n",
                stage, (int)tess_pipeline, (int)dst->separable,
                (int)(ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl));
    if (stage == _VERTEX_SHADER && !tess_pipeline && dst->separable == GL_TRUE &&
        ctx->mtl_funcs.ext_program && ctx->mtl_funcs.ext_program->compile_msl)
    {
        /* a FRESH compiler: re-installing options on the used one mixes the plain and
         * captured entry-point lowerings (redefinition of the out local) */
        spvc_compiler c2 = NULL;
        spvc_compiler_options o2 = NULL;
        spvc_parsed_ir ir2 = NULL;
        const char *cleansed = NULL;

        /* re-parse: the first compiler owns (and lowers) the parsed IR, so a second
         * compiler must start from the module again */
        if (spvc_context_parse_spirv(context, prepared, prepared_words, &ir2) == SPVC_SUCCESS &&
            spvc_context_create_compiler(context, SPVC_BACKEND_MSL, ir2, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &c2) == SPVC_SUCCESS &&
            spvc_compiler_set_entry_point(c2, entry, model) == SPVC_SUCCESS &&
            (cleansed = spvc_compiler_get_cleansed_entry_point_name(c2, entry, model)) != NULL &&
            spvc_compiler_create_compiler_options(c2, &o2) == SPVC_SUCCESS &&
            spvc_compiler_options_set_bool(o2, SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE) == SPVC_SUCCESS &&
            spvc_compiler_options_set_uint(o2, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(3, 1, 0)) == SPVC_SUCCESS &&
            spvc_compiler_options_set_bool(o2, SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, SPVC_TRUE) == SPVC_SUCCESS &&
            spvc_compiler_options_set_bool(o2, SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D, SPVC_TRUE) == SPVC_SUCCESS &&
            spvc_compiler_msl_add_discrete_descriptor_set(c2, 3) == SPVC_SUCCESS &&
            spvc_compiler_options_set_bool(o2, SPVC_COMPILER_OPTION_MSL_VERTEX_FOR_TESSELLATION, SPVC_TRUE) == SPVC_SUCCESS &&
            spvc_compiler_options_set_bool(o2, SPVC_COMPILER_OPTION_MSL_CAPTURE_OUTPUT_TO_BUFFER, SPVC_TRUE) == SPVC_SUCCESS &&
            spvc_compiler_options_set_uint(o2, SPVC_COMPILER_OPTION_MSL_SHADER_OUTPUT_BUFFER_INDEX, MGL_TESS_VS_OUT_BUF) == SPVC_SUCCESS &&
            spvc_compiler_install_compiler_options(c2, o2) == SPVC_SUCCESS)
        {
            const char *tmsl = NULL;

            if (spvc_compiler_compile(c2, &tmsl) == SPVC_SUCCESS && tmsl)
            {
                char *k = mgl_tess_rewrite_stage_msl(ctx, _VERTEX_SHADER, true, tmsl);

                if (k)
                {
                    /* SPIRV-Cross names the kernel after the entry (e.g. "main0") when
                     * it is not renamed: take the actual function name from the MSL */
                    char kentry[128] = "main";
                    const char *kd = strstr(k, "kernel void ");

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
                    mgl_tess_cache_vs_kernel(ctx, ptr->name, k, kentry);
                    free(k);
                }
                else if (getenv("MGL_TESS_DEBUG"))
                    fprintf(stderr, "MGL tess: sep-vs kernel: rewrite failed (head: %.100s)\n", tmsl);
            }
            else if (getenv("MGL_TESS_DEBUG"))
                fprintf(stderr, "MGL tess: sep-vs kernel: second SPIRV-Cross compile failed\n");
        }
        else if (getenv("MGL_TESS_DEBUG"))
            fprintf(stderr, "MGL tess: sep-vs kernel: fresh compiler/options rejected\n");
    }

done:
    spvc_context_destroy(context);
    free(prepared);
    free(patched);
    free(tess_owned);
    return ok;
}

static void finish_reflection(MGLProgramObj *dst, ReflectAcc *acc)
{
    dst->active_uniforms = acc->num_uniforms;
    for (int i = 0; i < acc->num_uniforms; i++)
        if ((GLint)strlen(acc->uniforms[i]) + 1 > dst->active_uniform_max_length)
            dst->active_uniform_max_length = (GLint)strlen(acc->uniforms[i]) + 1;

    dst->active_uniform_blocks = acc->num_blocks;
    for (int i = 0; i < acc->num_blocks; i++)
        if ((GLint)strlen(acc->blocks[i]) + 1 > dst->active_uniform_block_max_name_length)
            dst->active_uniform_block_max_name_length = (GLint)strlen(acc->blocks[i]) + 1;

    dst->active_atomic_counter_buffers = acc->num_atomic;
}

/* stage combinations this backend can execute */
static bool check_stage_mask(GLuint mask, char **log, bool separable)
{
    if (mask == 0)
    {
        mgl_append_info_log(log, "ERROR: no shader objects attached to the program\n");
        return false;
    }

    if ((mask & SHADER_MASK_BIT(_COMPUTE_SHADER)) && (mask & ~SHADER_MASK_BIT(_COMPUTE_SHADER)))
    {
        mgl_append_info_log(log, "ERROR: a program containing a compute shader cannot contain other shader stages\n");
        return false;
    }

    if (mask & SHADER_MASK_BIT(_GEOMETRY_SHADER))
    {
        /* emulated by the geometry domain: the stage becomes a compute kernel.  A geometry
         * shader after tessellation stages consumes the tess domain's TES-output capture
         * (mgl_ext_tess.h) instead of the vertex stage output. */
        if (!(mask & SHADER_MASK_BIT(_VERTEX_SHADER)) && !separable)
        {
            mgl_append_info_log(log, "ERROR: a program containing a geometry shader must contain a vertex shader\n");
            return false;
        }
    }

    if ((mask & SHADER_MASK_BIT(_TESS_CONTROL_SHADER)) && !(mask & SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)))
    {
        /* GL 4.6 §7.3: a program fails to link if it contains a tessellation control
           shader but no tessellation evaluation shader — EXCEPT when it is separable:
           a pipeline object then supplies the TES from another program at draw time
           (KHR-GL46.tessellation_shader links tc-only separable programs and expects
           success; glCreateShaderProgramv implies separable too) */
        if (!separable)
        {
            mgl_append_info_log(log, "ERROR: program contains a tessellation control shader "
                                "but no tessellation evaluation shader\n");
            return false;
        }
    }

    return true;
}

/* glslang's tessellation cross-stage link has a false positive on sized 'patch' arrays:
 * isIoResizeArray() (linkValidate.cpp) exempts patch variables on the control side but
 * not on the evaluation side, so a TES 'patch in T name[N]' loses its only dimension to
 * the (nonexistent) control-point dimension and every pair with EQUAL sizes is reported
 * as "Array sizes must be compatible" — printing two identical types.  Detect exactly
 * that pattern (all errors are tesc/tese array-size mismatches whose printed sizes
 * match) so link_glsl can continue; genuine mismatches still fail the link. */
static bool mgl_link_log_is_tess_patch_false_positive(const char *log)
{
    const char *p = log;
    bool saw_block = false;

    while (*p)
    {
        const char *eol;
        const char *next;
        const char *q1, *q1e, *q2, *q2e, *s1, *s2;
        long n1, n2;

        if (strncmp(p, "ERROR:", 6) != 0)
            break;                       /* unexpected line shape: not the pattern */

        if (!strstr(p, "Array sizes must be compatible:"))
            return false;                /* some other link error */

        eol = strstr(p, "\n");
        if (!eol)
            break;

        /* the two stage lines that follow carry quoted type strings */
        q1 = strchr(eol, '"');
        if (!q1)
            return false;
        q1e = strchr(q1 + 1, '"');
        if (!q1e)
            return false;
        q2 = strchr(q1e + 1, '"');
        if (!q2)
            return false;
        q2e = strchr(q2 + 1, '"');
        if (!q2e)
            return false;

        s1 = q1e;
        while (s1 > q1 && s1[-1] != '[') s1--;   /* last '[N' inside the quoted type */
        s1 = (s1 > q1) ? s1 - 1 : NULL;
        s2 = q2e;
        while (s2 > q2 && s2[-1] != '[') s2--;
        s2 = (s2 > q2) ? s2 - 1 : NULL;
        if (!s1 || !s2)
            return false;
        n1 = strtol(s1 + 1, NULL, 10);
        n2 = strtol(s2 + 1, NULL, 10);
        if (n1 != n2 || n1 <= 0)
            return false;                /* sizes differ: a genuine mismatch */

        saw_block = true;

        /* advance past the second stage line */
        next = strchr(q2e, '\n');
        if (!next)
            break;
        p = next + 1;
        while (*p == ' ' || *p == '\t' || *p == '\n')
            p++;                         /* log lines are indented; skip separators */
    }

    return saw_block;
}

static bool link_glsl(GLMContext ctx, MGLProgramObj *obj, MGLProgramObj *dst, ReflectAcc *acc, char **log)
{
    glslang_program_t *glsl_program;
    GLuint mask = 0;
    bool ok = true;

    for (int i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        if (!MGL_SHADER_OBJ(s)->compile_status || !s->compiled_glsl_shader)
        {
            mgl_append_info_log(log, "ERROR: %s shader %u is not successfully compiled\n", stage_label(s->glm_type), s->name);
            ok = false;
        }
        mask |= SHADER_MASK_BIT(s->glm_type);
    }

    if (!ok || !check_stage_mask(mask, log, obj ? obj->separable == GL_TRUE : false))
        return false;

    glsl_program = glslang_program_create();
    if (!glsl_program)
    {
        mgl_append_info_log(log, "ERROR: glslang program creation failed\n");
        return false;
    }

    for (int i = 0; i < obj->num_attached; i++)
        glslang_program_add_shader(glsl_program, obj->attached[i]->compiled_glsl_shader);

    if (!glslang_program_link(glsl_program, GLSLANG_MSG_DEFAULT_BIT))
    {
        const char *info = glslang_program_get_info_log(glsl_program);
        if (info && mgl_link_log_is_tess_patch_false_positive(info))
        {
            /* glslang's cross-stage check mis-handles a TES 'patch in' ARRAY:
             * linkValidate.cpp's isIoResizeArray() drops the patch exemption on the
             * evaluation side, so the patch array's single dimension is consumed as
             * the (nonexistent) control-point dimension and every sized TCS patch
             * out / TES patch in pair with EQUAL sizes is rejected ("Array sizes
             * must be compatible" printing two identical types).  The stages were
             * each parsed successfully, per-stage SPIR-V generation below does not
             * depend on the failed merge, and GL's own patch-in/out size rule is still
             * enforced later by the interface checks that consume the SPIR-V — continue
             * with a note instead of failing the link (tessellation_shader_tc_barriers
             * barrier_guarded_read_write_calls links a 'patch out int r[16]' pair). */
            mgl_append_info_log(log, "WARNING: %s", info);
        }
        else
        {
            mgl_append_info_log(log, "%s", (info && *info) ? info : "ERROR: GLSL link failed\n");
            glslang_program_delete(glsl_program);
            return false;
        }
    }

    /* assigns locations for legacy (< 330) shaders compiled with AUTO_MAP_LOCATIONS.
     * glslang's default resolver assigns them per stage ("does not do proper cross-stage
     * lining up"), but GL matches interface variables by NAME across linked stages — the
     * GLSL I/O mapper/resolver does that (a VS "out vec4 a; out vec4 b;" must feed an FS
     * that declares "in vec4 b; in vec4 a;" the same two slots).  Failing to share the
     * slots by name also breaks any consumer that reflects producer and consumer
     * separately (the geometry emulation's VS→GS record layout, transform feedback). */
    {
        glslang_mapper_t *mapper = glslang_glsl_mapper_create();
        glslang_resolver_t *resolver = NULL;
        int mapped = 0;

        if (mapper)
        {
            int first = -1;
            for (int i = 0; i < obj->num_attached; i++)
            {
                int t = (int)obj->attached[i]->glm_type;
                if (first < 0 || t < first) first = t;
            }
            if (first >= 0)
                resolver = glslang_glsl_resolver_create(glsl_program, (glslang_stage_t)first);
            if (resolver)
                mapped = glslang_program_map_io_with_resolver_and_mapper(glsl_program, resolver, mapper);
        }
        if (resolver) glslang_glsl_resolver_delete(resolver);
        if (mapper) glslang_glsl_mapper_delete(mapper);

        if (!mapped)
        {
            const char *info = glslang_program_get_info_log(glsl_program);
            mgl_append_info_log(log, "%s", (info && *info) ? info : "ERROR: GLSL I/O mapping failed\n");
            glslang_program_delete(glsl_program);
            return false;
        }
    }

    /* Generate every stage's SPIR-V before any stage is compiled to MSL: MGL's
     * per-stage work (introspection rewrite, fp64 lowering, SPIRV-Cross, Metal
     * library build) must not interleave with glslang's pool-based generator.
     * Interleaving them corrupted glslang's TPoolAllocator state on the next
     * glslang_program_SPIRV_generate (garbage pool sizes -> std::bad_alloc),
     * so glslang work is finished first and each module is handed over as a
     * private copy. */
    {
        const uint32_t *mods[_MAX_SHADER_TYPES] = {0};
        size_t sizes[_MAX_SHADER_TYPES] = {0};

        for (int stage = 0; stage < _MAX_SHADER_TYPES && ok; stage++)
        {
            const char *msgs;
            size_t size;

            if (!(mask & SHADER_MASK_BIT(stage)))
                continue;

            /* generate without glslang's built-in SPIRV-Tools validation: it
             * applies Vulkan rules to GLSL that GL allows.  Implicit-derivative
             * textureProjOffset() on a sampler2DRect in a non-fragment stage
             * (legal GL, undefined LOD) is emitted as an explicit Lod operand,
             * which the validator rejects for rectangular textures
             * (KHR-GL46.compute_shader.resource-texture).  SPIRV-Cross still
             * rejects modules it cannot translate, so malformed output is not
             * silently accepted */
            {
                glslang_spv_options_t spv_options = { 0 };
                glslang_program_SPIRV_generate_with_options(glsl_program, (glslang_stage_t)stage, &spv_options);
            }

            msgs = glslang_program_SPIRV_get_messages(glsl_program);
            if (msgs && *msgs)
            {
                mgl_append_info_log(log, "%s", msgs);
                if (strstr(msgs, "error") || strstr(msgs, "ERROR"))
                {
                    ok = false;
                    break;
                }
            }

            size = glslang_program_SPIRV_get_size(glsl_program);
            if (size == 0)
            {
                mgl_append_info_log(log, "ERROR: %s stage: SPIR-V generation produced no code\n", stage_label(stage));
                ok = false;
                break;
            }
            mods[stage] = (const uint32_t *)malloc(size * 4);
            if (!mods[stage])
            {
                mgl_append_info_log(log, "ERROR: out of memory\n");
                ok = false;
                break;
            }
            memcpy((void *)mods[stage], glslang_program_SPIRV_get_ptr(glsl_program), size * 4);
            sizes[stage] = size;
        }

        for (int stage = 0; stage < _MAX_SHADER_TYPES && ok; stage++)
        {
            if (!(mask & SHADER_MASK_BIT(stage)) || !mods[stage])
                continue;

            ok = compile_stage(ctx, dst, stage, mods[stage], sizes[stage], "main", acc, log,
                               (mask & SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)) != 0);
            free((void *)mods[stage]);

            /* executable stages keep the GLSL source / type of the shader they came from
             * (the xfb domain re-translates the vertex stage from source for capture) */
            if (ok && dst->base.shader_slots[stage])
            {
                Shader *exec = dst->base.shader_slots[stage];
                for (int i = 0; i < obj->num_attached; i++)
                {
                    Shader *a = obj->attached[i];
                    if ((int)a->glm_type == stage && a->src)
                    {
                        exec->src = strdup(a->src);
                        exec->src_len = a->src_len;
                        break;
                    }
                }
            }
        }

        /* the glslang program refers to the shaders' intermediates; never keep it past the link */
        glslang_program_delete(glsl_program);

        return ok;
    }
}

static bool link_spirv_attached(GLMContext ctx, MGLProgramObj *obj, MGLProgramObj *dst, ReflectAcc *acc, char **log)
{
    GLuint mask = 0;
    bool ok = true;

    for (int i = 0; i < obj->num_attached; i++)
    {
        Shader *s = obj->attached[i];
        MGLShaderObj *so = MGL_SHADER_OBJ(s);

        if (!so->specialized || !so->compile_status)
        {
            mgl_append_info_log(log, "ERROR: SPIR-V %s shader %u has not been successfully specialized (glSpecializeShader)\n",
                                stage_label(s->glm_type), s->name);
            ok = false;
        }
        if (mask & SHADER_MASK_BIT(s->glm_type))
        {
            mgl_append_info_log(log, "ERROR: more than one SPIR-V shader attached for the %s stage\n", stage_label(s->glm_type));
            ok = false;
        }
        mask |= SHADER_MASK_BIT(s->glm_type);
    }

    if (!ok || !check_stage_mask(mask, log, obj ? obj->separable == GL_TRUE : false))
        return false;

    for (int i = 0; i < obj->num_attached && ok; i++)
    {
        Shader *s = obj->attached[i];
        MGLShaderObj *so = MGL_SHADER_OBJ(s);
        ok = compile_stage(ctx, dst, s->glm_type, so->spirv, so->spirv_words, so->spirv_entry, acc, log,
                           (mask & SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)) != 0);
    }

    return ok;
}

enum { LINK_FROM_ATTACHED = 0, LINK_FROM_EXECUTABLES = 1 };

/*
 * Common link driver. Builds the new executable in a scratch object and swaps it in on
 * success. On failure LINK_STATUS is FALSE; for glLinkProgram the previous executable stays
 * installed (GL 4.6 7.3: a current program keeps its executable), for glProgramBinary it is lost.
 */
static bool run_link(GLMContext ctx, MGLProgramObj *obj, int source,
                     const uint32_t *const *words, const size_t *counts, const char *const *entries, GLuint mask)
{
    MGLProgramObj *scratch;
    ReflectAcc acc;
    char *log = NULL;
    bool ok = false;

    memset(&acc, 0, sizeof(acc));

    scratch = (MGLProgramObj *)calloc(1, sizeof(MGLProgramObj));
    if (!scratch)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return false;
    }
    scratch->base.name = obj->base.name;
    /* GL_PROGRAM_SEPARABLE semantics must be visible inside the link (the tess domain
     * compiles the separable vertex stage's compute-kernel variant for pipelines) */
    scratch->separable = obj->separable;

    if (source == LINK_FROM_ATTACHED)
    {
        bool any_spirv = false, any_glsl = false;

        for (int i = 0; i < obj->num_attached; i++)
        {
            if (MGL_SHADER_OBJ(obj->attached[i])->spirv_binary) any_spirv = true;
            else any_glsl = true;
        }

        if (any_spirv && any_glsl)
            mgl_append_info_log(&log, "ERROR: cannot link SPIR-V shaders together with GLSL shaders\n");
        else if (any_spirv)
            ok = link_spirv_attached(ctx, obj, scratch, &acc, &log);
        else
            ok = link_glsl(ctx, obj, scratch, &acc, &log);
    }
    else
    {
        ok = check_stage_mask(mask, &log, obj ? obj->separable == GL_TRUE : false);
        for (int stage = 0; stage < _MAX_SHADER_TYPES && ok; stage++)
            if (mask & SHADER_MASK_BIT(stage))
                ok = compile_stage(ctx, scratch, stage, words[stage], counts[stage], entries[stage], &acc, &log,
                              (mask & SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)) != 0);
    }

    /* GL link rule: a program with a compute shader must have a work group size.
     * Multiple compute shaders link as one unit — none of them declaring a
     * local size (KHR-GL46.compute_shader.glsl-link-time-errors) fails the link,
     * as does a size given only through spec constants' LocalSizeId mode. */
    /* GL link rule: a program with a compute shader must have a work group size
     * (KHR-GL46.compute_shader.glsl-link-time-errors links two compute shaders,
     * neither declaring one).  glslang defaults the linked intermediate's size
     * to (1,1,1), so the SPIR-V cannot tell "declared 1,1,1" from "not declared"
     * — check whether any attached compute shader's source declares a
     * local_size qualifier (also covers local_size_*_id) */

    /* GL 4.6 §11.1.3.11: the program as a whole may not declare more image uniforms
     * than GL_MAX_COMBINED_IMAGE_UNIFORMS; the per-stage limits are enforced in
     * compile_stage (glslang does not check them for every stage) */
    if (ok)
    {
        GLuint total = 0;

        for (int stage = 0; stage < _MAX_SHADER_TYPES; stage++)
        {
            SpirvResourceList *il = &scratch->base.spirv_resources_list[stage][SPVC_RESOURCE_TYPE_STORAGE_IMAGE];

            for (GLuint i = 0; i < il->count; i++)
                total += il->list[i].array_size ? il->list[i].array_size : 1;
        }
        if (total)
        {
            GLint lim = 0;

            ctx->dispatch.get_integerv(ctx, GL_MAX_COMBINED_IMAGE_UNIFORMS, &lim);
            if (lim > 0 && total > (GLuint)lim)
            {
                mgl_append_info_log(&log,
                    "ERROR: %u image uniforms exceed GL_MAX_COMBINED_IMAGE_UNIFORMS (%d)\n",
                    total, lim);
                ok = false;
            }
        }
    }

    if (ok && (mask & SHADER_MASK_BIT(_COMPUTE_SHADER) || scratch->base.shader_slots[_COMPUTE_SHADER]))
    {
        bool declared = false, any_spirv_cs = false;
        for (int i = 0; i < obj->num_attached && !declared; i++)
        {
            Shader *s = obj->attached[i];
            if ((int)s->glm_type != _COMPUTE_SHADER)
                continue;
            if (MGL_SHADER_OBJ(s)->spirv_binary)
            {
                /* glShaderBinary SPIR-V: the module's own execution modes are
                 * authoritative (LocalSize / LocalSizeId).  The binary is parsed
                 * by the SPIR-V link driver; the linked stage module ends up in
                 * the scratch, checked below */
                any_spirv_cs = true;
                continue;
            }
            if (!s->src)
                continue;
            /* a `local_size' identifier in the source: outside comments this only
             * occurs in the in/out layout qualifier */
            for (const char *q = s->src; (q = strstr(q, "local_size")) != NULL; q++)
                if (q == s->src || (!isalnum((unsigned char)q[-1]) && q[-1] != '_'))
                    declared = true;   /* local_size, local_size_x/_y/_z(+_id) */
        }
        if (!declared && (!obj->num_attached || any_spirv_cs))
        {
            /* glProgramBinary / glShaderBinary SPIR-V: no GLSL sources — the
             * linked compute SPIR-V's execution modes are authoritative */
            unsigned *ir = scratch->base.spirv[_COMPUTE_SHADER].ir;
            size_t n = scratch->base.spirv[_COMPUTE_SHADER].size;
            for (size_t w = 5; w + 3 < n; w++)
                if ((ir[w] == 16 && ir[w + 3] == 17) ||   /* OpExecutionMode LocalSize */
                    (ir[w] == 17 && ir[w + 3] == 18))     /* OpExecutionModeId LocalSizeId */
                    declared = true;
        }
        if (!declared)
        {
            mgl_append_info_log(&log, "ERROR: no local work group size was specified for the compute shader\n");
            ok = false;
        }
    }

    if (ok)
    {
        finish_reflection(scratch, &acc);

        exec_free(ctx, obj);
        exec_move(obj, scratch);

        /* non-NULL linked_glsl_program == "linked" for code that predates link_status;
         * an empty glslang program object, never linked, owned by this program */
        if (STATE(ext_program) && STATE(ext_program)->compiler_released)
        {
            glslang_initialize_process();
            STATE(ext_program)->compiler_released = GL_FALSE;
        }
        obj->base.linked_glsl_program = glslang_program_create();

        obj->link_status = GL_TRUE;
        /* Metal functions are fresh: the renderer must not release/recompile them */
        obj->base.dirty_bits &= ~DIRTY_PROGRAM;
    }
    else
    {
        exec_free(ctx, scratch);
        obj->link_status = GL_FALSE;
        if (source == LINK_FROM_EXECUTABLES)
            exec_free(ctx, obj);
        if (getenv("MGL_SHADER_DUMP") && log)
            fprintf(stderr, "MGL: program %u link failed: %s\n", obj->base.name, log);
    }

    obj->validate_status = GL_FALSE;
    free(obj->info_log);
    obj->info_log = log;

    reflect_acc_free(&acc);
    free(scratch);

    STATE(dirty_bits) |= DIRTY_PROGRAM;

    return ok;
}

/* the xfb domain resolves captured varyings in its post-link hook; a failure there fails the link */
bool mgl_program_check_xfb_link(GLMContext ctx, Program *program)
{
    const char *xlog = NULL;
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(program);

    if (!mgl_xfb_link_failed(ctx, program, &xlog))
        return true;

    obj->link_status = GL_FALSE;
    mgl_append_info_log(&obj->info_log, "%s%s", xlog ? xlog : "ERROR: transform feedback varyings could not be resolved",
                        (xlog && xlog[0] && xlog[strlen(xlog) - 1] == '\n') ? "" : "\n");
    return false;
}

void mglLinkProgram(GLMContext ctx, GLuint program)
{
    Program *pptr;

    pptr = lookupProgram(ctx, program);
    if (!pptr)
        return;

    /* relinking the current program while transform feedback captures; the rule also covers a
     * paused session of another transform feedback object that still holds this program
     * (GL 4.6 §7.3.1) */
    if (ctx->state.program == pptr && mgl_xfb_capturing(ctx))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }
    if (mgl_xfb_program_in_use(ctx, pptr))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    mgl_ext_pre_link(ctx, pptr);

    if (run_link(ctx, MGL_PROGRAM_OBJ(pptr), LINK_FROM_ATTACHED, NULL, NULL, NULL, 0))
    {
        mgl_ext_post_link(ctx, pptr);
        mgl_program_check_xfb_link(ctx, pptr);
        mgl_program_check_geom_link(ctx, pptr);
        mgl_program_check_tess_link(ctx, pptr);
    }
}

/* link from final per-stage SPIR-V (program binaries); runs the link hooks */
bool mgl_program_link_spirv_stages(GLMContext ctx, Program *program, const uint32_t *const *words,
                                   const size_t *word_counts, const char *const *entries, GLuint stage_mask)
{
    bool ok;

    mgl_ext_pre_link(ctx, program);

    ok = run_link(ctx, MGL_PROGRAM_OBJ(program), LINK_FROM_EXECUTABLES, words, word_counts, entries, stage_mask);

    if (ok)
    {
        mgl_ext_post_link(ctx, program);
        ok = mgl_program_check_xfb_link(ctx, program) && mgl_program_check_geom_link(ctx, program) &&
             mgl_program_check_tess_link(ctx, program);
    }

    return ok;
}

void mglUseProgram(GLMContext ctx, GLuint program)
{
    Program *pptr = NULL;
    Program *old;

    if (program)
    {
        pptr = lookupProgram(ctx, program);
        if (!pptr)
            return;

        if (!MGL_PROGRAM_OBJ(pptr)->link_status)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    /* the current program cannot change while transform feedback captures */
    if (mgl_xfb_capturing(ctx) && ctx->state.program != pptr)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    old = ctx->state.program;
    if (old == pptr)
        return;

    ctx->state.program = pptr;
    mgl_program_ref(pptr);
    STATE_VAR(current_program) = program;

    if (pptr || STATE(program_pipeline))
    {
        /* binding a program, or falling back to the bound program pipeline */
        ctx->state.dirty_bits |= DIRTY_PROGRAM;
    }
    /* unbinding with no pipeline: keep the existing Metal pipeline state */

    mgl_program_unref(ctx, old);
}

GLint  mglGetAttribLocation(GLMContext ctx, GLuint program, const GLchar *name)
{
    Program *ptr;

    ptr = lookupProgram(ctx, program);
    if (!ptr)
        return -1;

    if (!MGL_PROGRAM_OBJ(ptr)->link_status)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return -1;
    }

    if (!name || !strncmp(name, "gl_", 3))
        return -1;

    /* the link-time introspection table carries the GL locations: dvec3/dvec4
       inputs (and their array elements / matrix columns) take one GL location
       each while the SPIR-V Location decoration (the resource list below) is
       the internal slot the fp64 word-pair interface fetches */
    {
        GLint iloc = mgl_introspect_get_attrib_location(ctx, program, name);
        if (iloc >= 0)
            return iloc;
    }

    {
        int count = ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].count;

        for (int i = 0; i < count; i++)
        {
            const char *str = ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].list[i].name;

            if (str && !strcmp(str, name))
                return (GLint)ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].list[i].location;
        }
        /* GLSL 4.60 §11.1.1 / glGetActiveAttrib name form: an array (or matrix
         * column / matrix itself) is queried as "name[0]" — the exact form
         * reflection reports — while the resource list carries the bare name
         * (or "name[0]"); accept both directions of that one suffix. */
        {
            size_t len = name ? strlen(name) : 0;

            if (len > 3 && !strcmp(name + len - 3, "[0]"))
            {
                for (int i = 0; i < count; i++)
                {
                    const char *str = ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].list[i].name;
                    size_t slen = str ? strlen(str) : 0;

                    if (!str)
                        continue;
                    if (slen == len - 3 && !strncmp(str, name, slen))
                        return (GLint)ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].list[i].location;
                    if (!strcmp(str, name))
                        return (GLint)ptr->spirv_resources_list[_VERTEX_SHADER][SPVC_RESOURCE_TYPE_STAGE_INPUT].list[i].location;
                }
            }
        }
    }

    return -1;
}

#pragma mark - program queries

#define MAX_PROGRAMIV_HOOKS 8
static mgl_programiv_hook programiv_hooks[MAX_PROGRAMIV_HOOKS];
static int num_programiv_hooks;

void mgl_program_register_programiv_hook(mgl_programiv_hook fn)
{
    for (int i = 0; i < num_programiv_hooks; i++)
        if (programiv_hooks[i] == fn)
            return;
    if (num_programiv_hooks < MAX_PROGRAMIV_HOOKS)
        programiv_hooks[num_programiv_hooks++] = fn;
}

void mglGetProgramiv(GLMContext ctx, GLuint program, GLenum pname, GLint *params)
{
    Program *pptr;
    MGLProgramObj *obj;

    pptr = lookupProgram(ctx, program);
    if (!pptr)
        return;
    obj = MGL_PROGRAM_OBJ(pptr);

    if (!params)
        return;

    for (int i = 0; i < num_programiv_hooks; i++)
        if (programiv_hooks[i](ctx, pptr, pname, params))
            return;

    switch (pname) {
        case GL_DELETE_STATUS:
            *params = pptr->delete_status ? GL_TRUE : GL_FALSE;
            break;
        case GL_LINK_STATUS:
            /* xfb: varying resolution can fail the link after the stages compiled */
            *params = (obj->link_status && !mgl_xfb_link_failed(ctx, pptr, NULL)) ? GL_TRUE : GL_FALSE;
            break;
        case GL_VALIDATE_STATUS:
            *params = obj->validate_status ? GL_TRUE : GL_FALSE;
            break;
        case GL_INFO_LOG_LENGTH:
            *params = (obj->info_log && *obj->info_log) ? (GLint)strlen(obj->info_log) + 1 : 0;
            break;
        case GL_ATTACHED_SHADERS:
            *params = obj->num_attached;
            break;
        case GL_ACTIVE_ATTRIBUTES:
            *params = obj->active_attributes;
            break;
        case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH:
            *params = obj->active_attribute_max_length;
            break;
        case GL_ACTIVE_UNIFORMS:
            *params = obj->active_uniforms;
            break;
        case GL_ACTIVE_UNIFORM_MAX_LENGTH:
            *params = obj->active_uniform_max_length;
            break;
        case GL_ACTIVE_UNIFORM_BLOCKS:
            *params = obj->active_uniform_blocks;
            break;
        case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH:
            *params = obj->active_uniform_block_max_name_length;
            break;
        case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS:
            *params = obj->active_atomic_counter_buffers;
            break;
        case GL_COMPUTE_WORK_GROUP_SIZE:
            if (!obj->link_status || !(obj->linked_mask & SHADER_MASK_BIT(_COMPUTE_SHADER)))
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            params[0] = pptr->local_workgroup_size.x;
            params[1] = pptr->local_workgroup_size.y;
            params[2] = pptr->local_workgroup_size.z;
            break;
        case GL_PROGRAM_SEPARABLE:
            *params = obj->separable ? GL_TRUE : GL_FALSE;
            break;
        case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
            *params = obj->binary_retrievable_hint ? GL_TRUE : GL_FALSE;
            break;
        case GL_PROGRAM_BINARY_LENGTH:
            *params = mgl_program_binary_length(ctx, pptr);
            break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_MODE:
        case GL_TRANSFORM_FEEDBACK_VARYINGS:
        case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH:
            if (mgl_xfb_get_programiv(ctx, pptr, pname, params))    /* xfb domain */
                break;
            *params = (pname == GL_TRANSFORM_FEEDBACK_BUFFER_MODE) ? GL_INTERLEAVED_ATTRIBS : 0;
            break;
        /* geometry programs: the geom domain answers from its reflection */
        case GL_GEOMETRY_VERTICES_OUT:
        case GL_GEOMETRY_INPUT_TYPE:
        case GL_GEOMETRY_OUTPUT_TYPE:
        case GL_GEOMETRY_SHADER_INVOCATIONS:
            if (!obj->link_status || !(obj->linked_mask & SHADER_MASK_BIT(_GEOMETRY_SHADER)))
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            if (!mgl_geom_get_programiv(ctx, pptr, pname, params))
                *params = 0;
            break;
        case GL_TESS_CONTROL_OUTPUT_VERTICES:
            if (!obj->link_status || !(obj->linked_mask & SHADER_MASK_BIT(_TESS_CONTROL_SHADER)))
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
        {
            /* tess domain: the link record's value, or the TCS's own declaration for a
             * separable tc-only program (no tess record — the TES is in the pipeline) */
            *params = mgl_tess_program_out_vertices(ctx, pptr);
        }
            break;
        case GL_TESS_GEN_MODE:
        case GL_TESS_GEN_SPACING:
        case GL_TESS_GEN_VERTEX_ORDER:
        case GL_TESS_GEN_POINT_MODE:
            if (!obj->link_status || !(obj->linked_mask & SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)))
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
        {
            MGLTessProgram *tp = mgl_tess_program(ctx, pptr);
            if (!tp)
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
            switch (pname)
            {
                case GL_TESS_GEN_MODE:
                    *params = tp->domain == MGL_TESS_DOMAIN_TRIANGLES ? GL_TRIANGLES :
                              (tp->domain == MGL_TESS_DOMAIN_QUADS ? GL_QUADS : GL_ISOLINES);
                    break;
                case GL_TESS_GEN_SPACING:
                    *params = tp->spacing == MGL_TESS_SPACING_EQUAL ? GL_EQUAL :
                              (tp->spacing == MGL_TESS_SPACING_FRACTIONAL_EVEN ? GL_FRACTIONAL_EVEN :
                               GL_FRACTIONAL_ODD);
                    break;
                case GL_TESS_GEN_VERTEX_ORDER:
                    *params = tp->vertex_order_ccw ? GL_CCW : GL_CW;
                    break;
                default:
                    *params = tp->point_mode ? GL_TRUE : GL_FALSE;
                    break;
            }
        }
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetProgramInfoLog(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Program *pptr;

    pptr = lookupProgram(ctx, program);
    if (!pptr)
        return;

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mgl_copy_info_log(MGL_PROGRAM_OBJ(pptr)->info_log, bufSize, length, infoLog);
}
