/*
 * program_ext.c
 * MGL — separable programs, program pipelines, program binaries, SPIR-V shaders
 * (ARB_gl_spirv / GL 4.6), validation, compiler release and precision queries.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glslang_c_interface.h>

#include "mgl.h"
#include "glm_context.h"
#include "mgl_ext_program.h"
#include "mgl_ext_xfb.h"
#include "mgl_ext_introspect.h"   /* subroutine selection reset on UseProgramStages/BindProgramPipeline */
#include "shaders.h"
#include "spirv.h"

void mgl_program_ref(Program *ptr);
void mgl_program_unref(GLMContext ctx, Program *ptr);
void mglFreeProgram(GLMContext ctx, Program *ptr);
ProgramPipeline *newProgramPipeline(GLMContext ctx, GLuint pipeline);
ProgramPipeline *findProgramPipeline(GLMContext ctx, GLuint pipeline);
ProgramPipeline *getProgramPipeline(GLMContext ctx, GLuint pipeline);

#pragma mark - helpers

/* program-name lookup with the GL error rules for "program" parameters */
static Program *lookup_program(GLMContext ctx, GLuint program)
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

static const GLbitfield stage_bits[_MAX_SHADER_TYPES] = {
    [_VERTEX_SHADER] = GL_VERTEX_SHADER_BIT,
    [_TESS_CONTROL_SHADER] = GL_TESS_CONTROL_SHADER_BIT,
    [_TESS_EVALUATION_SHADER] = GL_TESS_EVALUATION_SHADER_BIT,
    [_GEOMETRY_SHADER] = GL_GEOMETRY_SHADER_BIT,
    [_FRAGMENT_SHADER] = GL_FRAGMENT_SHADER_BIT,
    [_COMPUTE_SHADER] = GL_COMPUTE_SHADER_BIT,
};

static const char *stage_name(int stage)
{
    static const char *names[] = {"vertex", "tessellation control", "tessellation evaluation",
        "geometry", "fragment", "compute"};
    return (stage >= 0 && stage < _MAX_SHADER_TYPES) ? names[stage] : "unknown";
}

#pragma mark - sampler / program validation

static const char *sampler_dim_name(GLuint key)
{
    switch (key & 0xF)
    {
        case SpvDim1D: return "1D";
        case SpvDim2D: return "2D";
        case SpvDim3D: return "3D";
        case SpvDimCube: return "Cube";
        case SpvDimRect: return "Rect";
        case SpvDimBuffer: return "Buffer";
        default: return "?";
    }
}

/* GL 4.6 11.1.3.11: samplers of different types must not use the same texture image unit */
bool mgl_program_validate_samplers(GLMContext ctx, Program *const *programs, int count, char **log)
{
    bool ok = true;
    GLuint max_units = STATE_VAR(max_combined_texture_image_units);
    GLuint used_units = 0;
    GLuint seen_units[256];

    for (int a = 0; a < count; a++)
    {
        MGLProgramObj *pa = programs[a] ? MGL_PROGRAM_OBJ(programs[a]) : NULL;
        if (!pa) continue;

        for (int i = 0; i < pa->num_samplers; i++)
        {
            MGLSamplerUse *sa = &pa->samplers[i];
            bool counted = false;

            for (GLuint u = 0; u < used_units; u++)
                if (seen_units[u] == sa->unit) counted = true;
            if (!counted && used_units < 256)
                seen_units[used_units++] = sa->unit;

            for (int b = a; b < count; b++)
            {
                MGLProgramObj *pb = programs[b] ? MGL_PROGRAM_OBJ(programs[b]) : NULL;
                if (!pb) continue;

                for (int j = (b == a) ? i + 1 : 0; j < pb->num_samplers; j++)
                {
                    MGLSamplerUse *sb = &pb->samplers[j];

                    if (sa->unit == sb->unit && sa->type_key != sb->type_key)
                    {
                        mgl_append_info_log(log, "ERROR: samplers of different types use texture image unit %u: "
                                            "\"%s\" (%s sampler%s%s) and \"%s\" (%s sampler%s%s)\n", sa->unit,
                                            sa->name, sampler_dim_name(sa->type_key),
                                            (sa->type_key & 0x10) ? " array" : "", (sa->type_key & 0x20) ? " shadow" : "",
                                            sb->name, sampler_dim_name(sb->type_key),
                                            (sb->type_key & 0x10) ? " array" : "", (sb->type_key & 0x20) ? " shadow" : "");
                        ok = false;
                    }
                }
            }
        }
    }

    if (max_units && used_units > max_units)
    {
        mgl_append_info_log(log, "ERROR: %u texture image units used, more than GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS (%u)\n",
                            used_units, max_units);
        ok = false;
    }

    return ok;
}

void mglValidateProgram(GLMContext ctx, GLuint program)
{
    Program *ptr;
    MGLProgramObj *obj;
    char *log = NULL;
    bool ok = true;

    ptr = lookup_program(ctx, program);
    if (!ptr)
        return;
    obj = MGL_PROGRAM_OBJ(ptr);

    if (!obj->link_status)
    {
        mgl_append_info_log(&log, "ERROR: program %u is not successfully linked\n", program);
        ok = false;
    }
    else
    {
        ok = mgl_program_validate_samplers(ctx, &ptr, 1, &log);

        /* A program with a vertex stage and no fragment stage is link-valid and
         * validate-valid in GL (separable programs: the fragment stage usually
         * comes from another program of the pipeline).  MGL rasterizes such a
         * program with the trampoline fragment function (see bindMTLProgram:),
         * so it is not a validation error.  (KHR-GL46.direct_state_access
         * program_pipelines_functional validates a vertex-only program.) */
    }

    if (ok)
        mgl_append_info_log(&log, "Program %u validated against the current state.\n", program);

    obj->validate_status = ok ? GL_TRUE : GL_FALSE;
    free(obj->info_log);
    obj->info_log = log;
}

#pragma mark - program pipeline objects

static MGLPipelineObj *pipeline_for_use(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *pp = findProgramPipeline(ctx, pipeline);

    if (!pp)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return NULL;
    }

    /* a generated but never bound name gets its state vector on first use */
    MGL_PIPELINE_OBJ(pp)->created = GL_TRUE;

    return MGL_PIPELINE_OBJ(pp);
}

void mglGenProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++)
    {
        pipelines[i] = getNewName(&STATE(program_pipeline_table));
        if (!getProgramPipeline(ctx, pipelines[i]))
        {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
    }
}

void mglCreateProgramPipelines(GLMContext ctx, GLsizei n, GLuint *pipelines)
{
    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++)
    {
        ProgramPipeline *pp;

        pipelines[i] = getNewName(&STATE(program_pipeline_table));
        pp = getProgramPipeline(ctx, pipelines[i]);
        if (!pp)
        {
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
        MGL_PIPELINE_OBJ(pp)->created = GL_TRUE;
    }
}

GLboolean mglIsProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = findProgramPipeline(ctx, pipeline);

    return (ptr && MGL_PIPELINE_OBJ(ptr)->created) ? GL_TRUE : GL_FALSE;
}

static void pipeline_set_stage(GLMContext ctx, MGLPipelineObj *pipe, int stage, Program *prog)
{
    Program *old = pipe->base.stage_programs[stage];

    if (old == prog)
        return;

    mgl_program_ref(prog);
    pipe->base.stage_programs[stage] = prog;
    mgl_program_unref(ctx, old);
}

static void pipeline_set_active(GLMContext ctx, MGLPipelineObj *pipe, Program *prog)
{
    Program *old = pipe->active_program;

    if (old == prog)
        return;

    mgl_program_ref(prog);
    pipe->active_program = prog;
    mgl_program_unref(ctx, old);
}

void mglDeleteProgramPipelines(GLMContext ctx, GLsizei n, const GLuint *pipelines)
{
    if (n < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    for (GLsizei i = 0; i < n; i++)
    {
        ProgramPipeline *ptr;
        MGLPipelineObj *pipe;

        if (pipelines[i] == 0)
            continue;

        ptr = findProgramPipeline(ctx, pipelines[i]);
        if (!ptr)
            continue;
        pipe = MGL_PIPELINE_OBJ(ptr);

        /* deleting the bound pipeline reverts the binding to zero */
        if (STATE(program_pipeline) == ptr)
        {
            STATE(program_pipeline) = NULL;
            STATE_VAR(program_pipeline_binding) = 0;
            STATE(dirty_bits) |= DIRTY_PROGRAM;
        }

        for (int s = 0; s < _MAX_SHADER_TYPES; s++)
            pipeline_set_stage(ctx, pipe, s, NULL);
        pipeline_set_active(ctx, pipe, NULL);

        deleteHashElement(&STATE(program_pipeline_table), pipelines[i]);
        free(pipe->info_log);
        free(pipe);
    }
}

void mglBindProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    ProgramPipeline *ptr = NULL;

    if (pipeline)
    {
        ptr = findProgramPipeline(ctx, pipeline);
        if (!ptr)
        {
            /* not a name returned by GenProgramPipelines */
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
        MGL_PIPELINE_OBJ(ptr)->created = GL_TRUE;
    }

    if (STATE(program_pipeline) == ptr)
        return;

    STATE(program_pipeline) = ptr;
    STATE_VAR(program_pipeline_binding) = pipeline;
    STATE(dirty_bits) |= DIRTY_PROGRAM;
    /* GL 4.6 §7.4.2: the subroutine uniform selection of every program the
     * pipeline will use is reset when the pipeline is bound */
    if (ptr)
    {
        MGLPipelineObj *po = MGL_PIPELINE_OBJ(ptr);
        for (int s = 0; s < _MAX_SHADER_TYPES; s++)
            if (po->base.stage_programs[s]) mgl_introspect_reset_subroutines(ctx, po->base.stage_programs[s]);
    }
}

void mglUseProgramStages(GLMContext ctx, GLuint pipeline, GLbitfield stages, GLuint program)
{
    MGLPipelineObj *pipe;
    Program *prog = NULL;
    MGLProgramObj *pobj = NULL;
    const GLbitfield valid = GL_VERTEX_SHADER_BIT | GL_FRAGMENT_SHADER_BIT | GL_GEOMETRY_SHADER_BIT |
                             GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT | GL_COMPUTE_SHADER_BIT;

    if (stages != GL_ALL_SHADER_BITS && (stages & ~valid))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (program)
    {
        prog = lookup_program(ctx, program);
        if (!prog)
            return;
        pobj = MGL_PROGRAM_OBJ(prog);

        if (!pobj->separable_latched || !pobj->link_status)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    pipe = pipeline_for_use(ctx, pipeline);
    if (!pipe)
        return;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        if (!(stages & stage_bits[s]))
            continue;

        /* stages the program has no executable for become empty */
        pipeline_set_stage(ctx, pipe, s, (pobj && (pobj->linked_mask & SHADER_MASK_BIT(s))) ? prog : NULL);
    }

    /* GL 4.6 §7.4.2: UseProgramStages resets the subroutine selection of the
     * program(s) it installs */
    if (prog) mgl_introspect_reset_subroutines(ctx, prog);

    pipe->validate_status = GL_FALSE;
    if (STATE(program_pipeline) == &pipe->base)
        STATE(dirty_bits) |= DIRTY_PROGRAM;
}

void mglActiveShaderProgram(GLMContext ctx, GLuint pipeline, GLuint program)
{
    MGLPipelineObj *pipe;
    Program *prog = NULL;

    if (program)
    {
        prog = lookup_program(ctx, program);
        if (!prog)
            return;

        if (!MGL_PROGRAM_OBJ(prog)->link_status)
        {
            ERROR_RETURN(GL_INVALID_OPERATION);
            return;
        }
    }

    pipe = pipeline_for_use(ctx, pipeline);
    if (!pipe)
        return;

    pipeline_set_active(ctx, pipe, prog);
}

void mglGetProgramPipelineiv(GLMContext ctx, GLuint pipeline, GLenum pname, GLint *params)
{
    MGLPipelineObj *pipe;
    int stage = -1;

    switch (pname)
    {
        case GL_ACTIVE_PROGRAM:
        case GL_VALIDATE_STATUS:
        case GL_INFO_LOG_LENGTH:
            break;
        case GL_VERTEX_SHADER: stage = _VERTEX_SHADER; break;
        case GL_FRAGMENT_SHADER: stage = _FRAGMENT_SHADER; break;
        case GL_GEOMETRY_SHADER: stage = _GEOMETRY_SHADER; break;
        case GL_TESS_CONTROL_SHADER: stage = _TESS_CONTROL_SHADER; break;
        case GL_TESS_EVALUATION_SHADER: stage = _TESS_EVALUATION_SHADER; break;
        case GL_COMPUTE_SHADER: stage = _COMPUTE_SHADER; break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    pipe = pipeline_for_use(ctx, pipeline);
    if (!pipe || !params)
        return;

    if (stage >= 0)
    {
        *params = pipe->base.stage_programs[stage] ? (GLint)pipe->base.stage_programs[stage]->name : 0;
        return;
    }

    switch (pname)
    {
        case GL_ACTIVE_PROGRAM:
            *params = pipe->active_program ? (GLint)pipe->active_program->name : 0;
            break;
        case GL_VALIDATE_STATUS:
            *params = pipe->validate_status ? GL_TRUE : GL_FALSE;
            break;
        case GL_INFO_LOG_LENGTH:
            *params = (pipe->info_log && *pipe->info_log) ? (GLint)strlen(pipe->info_log) + 1 : 0;
            break;
    }
}

/* GL 4.6 11.1.3.11 program pipeline validation */
static bool validate_pipeline(GLMContext ctx, MGLPipelineObj *pipe, char **log)
{
    static const int order[] = {_VERTEX_SHADER, _TESS_CONTROL_SHADER, _TESS_EVALUATION_SHADER,
                                _GEOMETRY_SHADER, _FRAGMENT_SHADER};
    Program *progs[_MAX_SHADER_TYPES];
    int nprogs = 0;
    bool ok = true;
    bool any = false;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        Program *p = pipe->base.stage_programs[s];
        MGLProgramObj *po;
        bool dup = false;

        if (!p)
            continue;
        any = true;
        po = MGL_PROGRAM_OBJ(p);

        for (int k = 0; k < nprogs; k++)
            if (progs[k] == p) dup = true;
        if (!dup)
            progs[nprogs++] = p;

        if (!po->link_status)
        {
            mgl_append_info_log(log, "ERROR: program %u bound to the %s stage is not successfully linked\n", p->name, stage_name(s));
            ok = false;
        }
        else if (!(po->linked_mask & SHADER_MASK_BIT(s)))
        {
            mgl_append_info_log(log, "ERROR: program %u bound to the %s stage has no %s executable\n", p->name, stage_name(s), stage_name(s));
            ok = false;
        }

        if (!po->separable_latched)
        {
            mgl_append_info_log(log, "ERROR: program %u bound to the %s stage is not separable\n", p->name, stage_name(s));
            ok = false;
        }
    }

    if (!any)
    {
        mgl_append_info_log(log, "ERROR: program pipeline %u has no program bound to any stage\n", pipe->base.name);
        return false;
    }

    /* a program must be active for all of the stages it was linked with */
    for (int k = 0; k < nprogs; k++)
    {
        MGLProgramObj *po = MGL_PROGRAM_OBJ(progs[k]);
        for (int s = 0; s < _MAX_SHADER_TYPES; s++)
        {
            if ((po->linked_mask & SHADER_MASK_BIT(s)) && pipe->base.stage_programs[s] != progs[k])
            {
                mgl_append_info_log(log, "ERROR: program %u is active for some but not all of its linked stages "
                                    "(%s stage is not bound from it)\n", progs[k]->name, stage_name(s));
                ok = false;
            }
        }
    }

    /* no other program may sit between two stages of one program */
    for (int a = 0; a < 5; a++)
    {
        Program *pa = pipe->base.stage_programs[order[a]];
        if (!pa) continue;
        for (int c = a + 2; c < 5; c++)
        {
            if (pipe->base.stage_programs[order[c]] != pa) continue;
            for (int b = a + 1; b < c; b++)
            {
                Program *pb = pipe->base.stage_programs[order[b]];
                if (pb && pb != pa)
                {
                    mgl_append_info_log(log, "ERROR: program %u is active for the %s stage between two stages of program %u\n",
                                        pb->name, stage_name(order[b]), pa->name);
                    ok = false;
                }
            }
        }
    }

    if ((pipe->base.stage_programs[_TESS_CONTROL_SHADER] || pipe->base.stage_programs[_TESS_EVALUATION_SHADER] ||
         pipe->base.stage_programs[_GEOMETRY_SHADER]) && !pipe->base.stage_programs[_VERTEX_SHADER])
    {
        mgl_append_info_log(log, "ERROR: a tessellation or geometry stage is active without a vertex stage\n");
        ok = false;
    }

    if (pipe->base.stage_programs[_VERTEX_SHADER] && !pipe->base.stage_programs[_FRAGMENT_SHADER] &&
        !ctx->state.caps.rasterizer_discard)
    {
        /* geom (reported): a pipeline without a fragment program is legal GL - the stage
         * is optional and results are undefined; with transform feedback active (the CTS
         * api.program_pipeline_vs_gs_capture pass-2 shape draws with discard re-enabled)
         * nothing rasterizes anyway.  Metal renders such a pipeline with a nil fragment
         * function, so it is not an error; kept as an info-log note only. */
        mgl_append_info_log(log,
                            "WARNING: no fragment stage; rasterization output is undefined\n");
    }

    if (!mgl_program_validate_samplers(ctx, progs, nprogs, log))
        ok = false;

    return ok;
}

void mglValidateProgramPipeline(GLMContext ctx, GLuint pipeline)
{
    MGLPipelineObj *pipe;
    char *log = NULL;
    bool ok;

    pipe = pipeline_for_use(ctx, pipeline);
    if (!pipe)
        return;

    ok = validate_pipeline(ctx, pipe, &log);
    if (ok)
        mgl_append_info_log(&log, "Program pipeline %u validated.\n", pipeline);

    pipe->validate_status = ok ? GL_TRUE : GL_FALSE;
    free(pipe->info_log);
    pipe->info_log = log;
}

void mglGetProgramPipelineInfoLog(GLMContext ctx, GLuint pipeline, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    MGLPipelineObj *pipe;

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    /* a nonexistent pipeline name is INVALID_VALUE here (not INVALID_OPERATION
     * like the other pipeline entry points) - KHR-GL46.sepshaderobjs.PipelineApi
     * checks it; the spec text is silent, see the CTS comment "GL_INVALID_VALUE
     * error (the error is still missing from the spec)" */
    pipe = MGL_PIPELINE_OBJ(findProgramPipeline(ctx, pipeline));
    if (!pipe)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    pipe->created = GL_TRUE;

    mgl_copy_info_log(pipe->info_log, bufSize, length, infoLog);
}

/* the bound pipeline's stages merged into one Program the renderer can consume */
Program *mgl_program_pipeline_composite(GLMContext ctx)
{
    MGLPipelineObj *pipe;
    Program *c;
    bool any = false;

    if (!STATE(program_pipeline))
        return NULL;

    pipe = MGL_PIPELINE_OBJ(STATE(program_pipeline));
    c = &pipe->composite;

    memset(c, 0, sizeof(*c));
    c->name = 0;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        Program *p = pipe->base.stage_programs[s];

        if (!p || !MGL_PROGRAM_OBJ(p)->link_status || !p->shader_slots[s])
            continue;

        c->shader_slots[s] = p->shader_slots[s];
        c->spirv[s] = p->spirv[s];
        memcpy(c->spirv_resources_list[s], p->spirv_resources_list[s], sizeof(c->spirv_resources_list[s]));
        if (s == _COMPUTE_SHADER)
            c->local_workgroup_size = p->local_workgroup_size;
        c->linked_glsl_program = p->linked_glsl_program;
        any = true;
    }

    /* stage programs keep ownership; composite dirty bits stay clear so the renderer
     * never releases the stage programs' Metal functions through it */
    c->dirty_bits = 0;

    return any ? c : NULL;
}

Program *mgl_program_uniform_target(GLMContext ctx)
{
    if (ctx->state.program)
        return ctx->state.program;

    if (STATE(program_pipeline))
        return MGL_PIPELINE_OBJ(STATE(program_pipeline))->active_program;

    return NULL;
}

void mgl_program_invalidate_pipelines(GLMContext ctx)
{
    STATE(dirty_bits) |= DIRTY_PROGRAM;
}

#pragma mark - separable programs / parameters

void mglProgramParameteri(GLMContext ctx, GLuint program, GLenum pname, GLint value)
{
    Program *ptr;

    ptr = lookup_program(ctx, program);
    if (!ptr)
        return;

    switch (pname)
    {
        case GL_PROGRAM_SEPARABLE:
            if (value != GL_TRUE && value != GL_FALSE)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            /* takes effect at the next link */
            MGL_PROGRAM_OBJ(ptr)->separable = (GLboolean)value;
            break;

        case GL_PROGRAM_BINARY_RETRIEVABLE_HINT:
            if (value != GL_TRUE && value != GL_FALSE)
            {
                ERROR_RETURN(GL_INVALID_VALUE);
                return;
            }
            MGL_PROGRAM_OBJ(ptr)->binary_retrievable_hint = (GLboolean)value;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

GLuint  mglCreateShaderProgramv(GLMContext ctx, GLenum type, GLsizei count, const GLchar *const*strings)
{
    GLuint shader, program;
    Shader *sptr;
    Program *pptr;

    switch (type)
    {
        case GL_VERTEX_SHADER: case GL_FRAGMENT_SHADER: case GL_GEOMETRY_SHADER:
        case GL_TESS_CONTROL_SHADER: case GL_TESS_EVALUATION_SHADER: case GL_COMPUTE_SHADER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return 0;
    }

    if (count < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return 0;
    }

    shader = mglCreateShader(ctx, type);
    if (!shader)
        return 0;

    mglShaderSource(ctx, shader, count, strings, NULL);
    mglCompileShader(ctx, shader);

    program = mglCreateProgram(ctx);
    if (!program)
    {
        mglDeleteShader(ctx, shader);
        return 0;
    }

    pptr = findProgram(ctx, program);
    sptr = findShader(ctx, shader);
    MGL_PROGRAM_OBJ(pptr)->separable = GL_TRUE;

    if (MGL_SHADER_OBJ(sptr)->compile_status)
    {
        mglAttachShader(ctx, program, shader);
        mglLinkProgram(ctx, program);
        mglDetachShader(ctx, program, shader);
    }
    else
    {
        MGL_PROGRAM_OBJ(pptr)->link_status = GL_FALSE;
    }

    /* the shader's info log is appended to the program's */
    if (sptr->log && *sptr->log)
        mgl_append_info_log(&MGL_PROGRAM_OBJ(pptr)->info_log, "%s", sptr->log);

    mglDeleteShader(ctx, shader);

    return program;
}

#pragma mark - program binaries

/*
 * Binary layout (little endian uint32 words):
 *   [0] 'MGLB' [1] version [2] total bytes [3] FNV-1a of bytes 16..end
 *   [4] flags (bit0 separable, bit1 retrievable hint) [5] stage mask
 *   per stage in stage order: stage, entry length (bytes, NUL padded to 4), entry, word count, words
 * The per-stage SPIR-V is the final (linked / specialized) module, so ProgramBinary relinks
 * without GLSL through SPIRV-Cross → MSL → Metal and recomputes all reflection.
 */
#define BIN_HEADER_WORDS 6

static uint32_t fnv1a(const uint8_t *p, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static size_t binary_size(MGLProgramObj *obj)
{
    size_t bytes = BIN_HEADER_WORDS * 4;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        if (!(obj->linked_mask & SHADER_MASK_BIT(s)))
            continue;
        size_t elen = obj->stage_entry[s] ? strlen(obj->stage_entry[s]) + 1 : 1;
        bytes += 8 + ((elen + 3) & ~(size_t)3) + 4 + obj->base.spirv[s].size * 4;
    }

    return bytes;
}

GLint mgl_program_binary_length(GLMContext ctx, Program *ptr)
{
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(ptr);

    if (!obj->link_status)
        return 0;

    return (GLint)binary_size(obj);
}

void mglGetProgramBinary(GLMContext ctx, GLuint program, GLsizei bufSize, GLsizei *length, GLenum *binaryFormat, void *binary)
{
    Program *ptr;
    MGLProgramObj *obj;
    size_t size;
    uint32_t *w;
    size_t pos;

    ptr = lookup_program(ctx, program);
    if (!ptr)
        return;
    obj = MGL_PROGRAM_OBJ(ptr);

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!obj->link_status)
    {
        if (length) *length = 0;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    size = binary_size(obj);
    if ((size_t)bufSize < size || !binary)
    {
        if (length) *length = 0;
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    w = (uint32_t *)calloc(1, size);
    if (!w)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    w[0] = MGL_PROGRAM_BINARY_FORMAT;
    w[1] = MGL_PROGRAM_BINARY_VERSION;
    w[2] = (uint32_t)size;
    w[4] = (obj->separable ? 1u : 0u) | (obj->binary_retrievable_hint ? 2u : 0u);
    w[5] = obj->linked_mask;
    pos = BIN_HEADER_WORDS;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        const char *entry;
        size_t elen, epad;

        if (!(obj->linked_mask & SHADER_MASK_BIT(s)))
            continue;

        entry = obj->stage_entry[s] ? obj->stage_entry[s] : "";
        elen = strlen(entry) + 1;
        epad = (elen + 3) & ~(size_t)3;

        w[pos++] = (uint32_t)s;
        w[pos++] = (uint32_t)epad;
        memcpy(&w[pos], entry, elen);
        pos += epad / 4;
        w[pos++] = (uint32_t)ptr->spirv[s].size;
        memcpy(&w[pos], ptr->spirv[s].ir, ptr->spirv[s].size * 4);
        pos += ptr->spirv[s].size;
    }

    w[3] = fnv1a((const uint8_t *)&w[4], size - 16);

    memcpy(binary, w, size);
    free(w);

    if (length) *length = (GLsizei)size;
    if (binaryFormat) *binaryFormat = MGL_PROGRAM_BINARY_FORMAT;
}

/* load-time failure: LINK_STATUS FALSE, executable and previous link information lost */
static void binary_reject(GLMContext ctx, Program *ptr, const char *why)
{
    static const uint32_t *no_words[_MAX_SHADER_TYPES];
    static const size_t no_counts[_MAX_SHADER_TYPES];
    static const char *no_entries[_MAX_SHADER_TYPES];
    MGLProgramObj *obj = MGL_PROGRAM_OBJ(ptr);

    /* an empty relink tears down the old executable and sets LINK_STATUS FALSE */
    mgl_program_link_spirv_stages(ctx, ptr, no_words, no_counts, no_entries, 0);
    free(obj->info_log);
    obj->info_log = NULL;
    mgl_set_info_log(&obj->info_log, "ERROR: program binary rejected: %s\n", why);
}

void mglProgramBinary(GLMContext ctx, GLuint program, GLenum binaryFormat, const void *binary, GLsizei length)
{
    Program *ptr;
    const uint32_t *w;
    const uint32_t *words[_MAX_SHADER_TYPES] = {0};
    size_t counts[_MAX_SHADER_TYPES] = {0};
    const char *entries[_MAX_SHADER_TYPES] = {0};
    char entry_buf[_MAX_SHADER_TYPES][256];
    size_t nwords, pos;
    GLuint mask, seen = 0;

    ptr = lookup_program(ctx, program);
    if (!ptr)
        return;

    if (binaryFormat != MGL_PROGRAM_BINARY_FORMAT)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    /* reloading the current program while transform feedback captures */
    if (ctx->state.program == ptr && mgl_xfb_capturing(ctx))
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (length < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (!binary || length < BIN_HEADER_WORDS * 4 || (length & 3))
    {
        binary_reject(ctx, ptr, "truncated or misaligned data");
        return;
    }

    w = (const uint32_t *)binary;
    nwords = (size_t)length / 4;

    if (w[0] != MGL_PROGRAM_BINARY_FORMAT) { binary_reject(ctx, ptr, "bad magic"); return; }
    if (w[1] != MGL_PROGRAM_BINARY_VERSION) { binary_reject(ctx, ptr, "produced by a different MGL version"); return; }
    if (w[2] != (uint32_t)length) { binary_reject(ctx, ptr, "length mismatch"); return; }
    if (w[3] != fnv1a((const uint8_t *)&w[4], (size_t)length - 16)) { binary_reject(ctx, ptr, "checksum mismatch (corrupt data)"); return; }

    mask = w[5];
    pos = BIN_HEADER_WORDS;

    for (int s = 0; s < _MAX_SHADER_TYPES; s++)
    {
        uint32_t stage, epad, count;

        if (!(mask & SHADER_MASK_BIT(s)))
            continue;

        if (pos + 2 > nwords) { binary_reject(ctx, ptr, "truncated stage header"); return; }
        stage = w[pos++];
        epad = w[pos++];
        if (stage != (uint32_t)s || (epad & 3) || epad == 0 || epad > sizeof(entry_buf[0]) || pos + epad / 4 + 1 > nwords)
        {
            binary_reject(ctx, ptr, "malformed stage record");
            return;
        }
        memcpy(entry_buf[s], &w[pos], epad);
        entry_buf[s][epad - 1] = 0;
        entries[s] = entry_buf[s];
        pos += epad / 4;

        count = w[pos++];
        if (count < 5 || pos + count > nwords || w[pos] != SpvMagicNumber)
        {
            binary_reject(ctx, ptr, "malformed SPIR-V payload");
            return;
        }
        words[s] = &w[pos];
        counts[s] = count;
        pos += count;
        seen |= SHADER_MASK_BIT(s);
    }

    if (pos != nwords || seen != mask || mask == 0)
    {
        binary_reject(ctx, ptr, "inconsistent stage table");
        return;
    }

    MGL_PROGRAM_OBJ(ptr)->separable = (w[4] & 1) ? GL_TRUE : GL_FALSE;
    MGL_PROGRAM_OBJ(ptr)->separable_latched = (w[4] & 1) ? GL_TRUE : GL_FALSE;   /* a loaded binary counts as linked */
    MGL_PROGRAM_OBJ(ptr)->binary_retrievable_hint = (w[4] & 2) ? GL_TRUE : GL_FALSE;

    mgl_program_link_spirv_stages(ctx, ptr, words, counts, entries, mask);
}

#pragma mark - SPIR-V shader binaries (GL 4.6 / ARB_gl_spirv)

static const char *spirv_extensions[] = {
    "SPV_KHR_storage_buffer_storage_class",
    "SPV_KHR_shader_draw_parameters",
    "SPV_KHR_16bit_storage",
    "SPV_KHR_8bit_storage",
};
#define NUM_SPIRV_EXTENSIONS (sizeof(spirv_extensions) / sizeof(spirv_extensions[0]))

const GLubyte *mgl_program_spirv_extension(GLMContext ctx, GLuint index)
{
    if (index >= NUM_SPIRV_EXTENSIONS)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return NULL;
    }

    return (const GLubyte *)spirv_extensions[index];
}

static bool spirv_string_eq(const uint32_t *words, size_t avail_words, const char *s)
{
    size_t len = strlen(s) + 1;
    if (len > avail_words * 4)
        return false;
    return memcmp(words, s, len) == 0;
}

static bool spirv_string_ok(const uint32_t *words, size_t avail_words)
{
    return memchr(words, 0, avail_words * 4) != NULL;
}

void mglShaderBinary(GLMContext ctx, GLsizei count, const GLuint *shaders, GLenum binaryFormat, const void *binary, GLsizei length)
{
    const uint32_t *w = (const uint32_t *)binary;
    size_t nwords;

    if (count < 0 || length < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (binaryFormat != GL_SHADER_BINARY_FORMAT_SPIR_V)
    {
        ERROR_RETURN(GL_INVALID_ENUM);
        return;
    }

    /* every name must be a shader object */
    for (GLsizei i = 0; i < count; i++)
    {
        if (!findShader(ctx, shaders[i]))
        {
            if (shaders[i] && findProgram(ctx, shaders[i]))
                ERROR_RETURN(GL_INVALID_OPERATION);
            else
                ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
        /* GL 4.6 7.2: more than one shader of the same type → INVALID_OPERATION */
        for (GLsizei j = 0; j < i; j++)
        {
            if (findShader(ctx, shaders[j])->glm_type == findShader(ctx, shaders[i])->glm_type)
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return;
            }
        }
    }

    if (!binary || (length & 3) || length < 20 || w[0] != SpvMagicNumber)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    nwords = (size_t)length / 4;

    for (GLsizei i = 0; i < count; i++)
    {
        Shader *s = findShader(ctx, shaders[i]);
        MGLShaderObj *so = MGL_SHADER_OBJ(s);

        free(so->spirv);
        free(so->spirv_entry);
        so->spirv_entry = NULL;
        so->spirv = (uint32_t *)malloc(nwords * 4);
        if (!so->spirv)
        {
            so->spirv_words = 0;
            ERROR_RETURN(GL_OUT_OF_MEMORY);
            return;
        }
        memcpy(so->spirv, w, nwords * 4);
        so->spirv_words = nwords;
        so->spirv_binary = GL_TRUE;
        so->specialized = GL_FALSE;
        so->compile_status = GL_FALSE;

        /* a SPIR-V module replaces any compiled GLSL */
        if (s->compiled_glsl_shader)
        {
            glslang_shader_delete(s->compiled_glsl_shader);
            s->compiled_glsl_shader = NULL;
            if (STATE(ext_program) && STATE(ext_program)->live_glslang_shaders > 0)
                STATE(ext_program)->live_glslang_shaders--;
        }
        free(s->log);
        s->log = NULL;
    }
}

static SpvExecutionModel model_for_stage(GLuint glm_type)
{
    switch (glm_type)
    {
        case _VERTEX_SHADER: return SpvExecutionModelVertex;
        case _TESS_CONTROL_SHADER: return SpvExecutionModelTessellationControl;
        case _TESS_EVALUATION_SHADER: return SpvExecutionModelTessellationEvaluation;
        case _GEOMETRY_SHADER: return SpvExecutionModelGeometry;
        case _FRAGMENT_SHADER: return SpvExecutionModelFragment;
        default: return SpvExecutionModelGLCompute;
    }
}

void mglSpecializeShader(GLMContext ctx, GLuint shader, const GLchar *pEntryPoint, GLuint numSpecializationConstants, const GLuint *pConstantIndex, const GLuint *pConstantValue)
{
    Shader *s;
    MGLShaderObj *so;
    uint32_t *w;
    size_t n, pos;
    const char *entry = pEntryPoint ? pEntryPoint : "main";
    bool entry_found = false;
    char *log = NULL;
    /* SpecId → result id map */
    uint32_t *spec_ids = NULL, *spec_targets = NULL;
    size_t num_spec = 0, cap_spec = 0;
    bool ok = true;

    s = findShader(ctx, shader);
    if (!s)
    {
        if (shader && findProgram(ctx, shader))
            ERROR_RETURN(GL_INVALID_OPERATION);
        else
            ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }
    so = MGL_SHADER_OBJ(s);

    /* must hold a SPIR-V module that has not been specialized yet */
    if (!so->spirv_binary || so->specialized)
    {
        ERROR_RETURN(GL_INVALID_OPERATION);
        return;
    }

    if (numSpecializationConstants && (!pConstantIndex || !pConstantValue))
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    w = so->spirv;
    n = so->spirv_words;

    /* scan the module: entry points, SpecId decorations, extensions */
    for (pos = 5; pos < n; )
    {
        uint32_t op = w[pos] & 0xFFFF, wc = w[pos] >> 16;

        if (wc == 0 || pos + wc > n)
        {
            mgl_append_info_log(&log, "ERROR: malformed SPIR-V instruction stream at word %zu\n", pos);
            ok = false;
            break;
        }

        if (op == SpvOpEntryPoint && wc >= 4)
        {
            if (w[pos + 1] == (uint32_t)model_for_stage(s->glm_type) && spirv_string_eq(&w[pos + 3], wc - 3, entry))
                entry_found = true;
        }
        else if (op == SpvOpDecorate && wc >= 4 && w[pos + 2] == SpvDecorationSpecId)
        {
            if (num_spec == cap_spec)
            {
                cap_spec = cap_spec ? cap_spec * 2 : 16;
                spec_ids = (uint32_t *)realloc(spec_ids, cap_spec * 4);
                spec_targets = (uint32_t *)realloc(spec_targets, cap_spec * 4);
            }
            spec_ids[num_spec] = w[pos + 3];
            spec_targets[num_spec] = w[pos + 1];
            num_spec++;
        }
        else if (op == SpvOpExtension && wc >= 2)
        {
            bool known = false;
            if (!spirv_string_ok(&w[pos + 1], wc - 1))
            {
                mgl_append_info_log(&log, "ERROR: malformed OpExtension\n");
                ok = false;
            }
            else
            {
                for (size_t k = 0; k < NUM_SPIRV_EXTENSIONS; k++)
                    if (!strcmp((const char *)&w[pos + 1], spirv_extensions[k])) known = true;
                if (!known)
                {
                    mgl_append_info_log(&log, "ERROR: SPIR-V extension %s is not supported (see GL_SPIR_V_EXTENSIONS)\n",
                                        (const char *)&w[pos + 1]);
                    ok = false;
                }
            }
        }

        pos += wc;
    }

    if (ok && !entry_found)
    {
        free(spec_ids); free(spec_targets); free(log);
        ERROR_RETURN(GL_INVALID_VALUE);    /* entry point not found for this stage */
        return;
    }

    /* every requested constant id must exist */
    for (GLuint i = 0; ok && i < numSpecializationConstants; i++)
    {
        bool found = false;
        for (size_t k = 0; k < num_spec; k++)
            if (spec_ids[k] == pConstantIndex[i]) found = true;
        if (!found)
        {
            free(spec_ids); free(spec_targets); free(log);
            ERROR_RETURN(GL_INVALID_VALUE);
            return;
        }
    }

    /* patch OpSpecConstant / OpSpecConstantTrue / OpSpecConstantFalse defaults in place */
    for (pos = 5; ok && pos < n; )
    {
        uint32_t op = w[pos] & 0xFFFF, wc = w[pos] >> 16;

        if ((op == SpvOpSpecConstant && wc >= 4) || ((op == SpvOpSpecConstantTrue || op == SpvOpSpecConstantFalse) && wc == 3))
        {
            uint32_t result = w[pos + 2];

            for (size_t k = 0; k < num_spec; k++)
            {
                if (spec_targets[k] != result)
                    continue;
                for (GLuint i = 0; i < numSpecializationConstants; i++)
                {
                    if (pConstantIndex[i] != spec_ids[k])
                        continue;
                    if (op == SpvOpSpecConstant)
                    {
                        w[pos + 3] = pConstantValue[i];
                        /* 64-bit constants: GL supplies 32-bit values; clear the high word */
                        if (wc >= 5)
                            w[pos + 4] = 0;
                    }
                    else
                    {
                        w[pos] = (wc << 16) | (pConstantValue[i] ? SpvOpSpecConstantTrue : SpvOpSpecConstantFalse);
                    }
                }
            }
        }

        pos += wc;
    }

    free(spec_ids);
    free(spec_targets);

    so->specialized = GL_TRUE;      /* a module can be specialized once, even if it fails */
    so->compile_status = ok ? GL_TRUE : GL_FALSE;
    free(so->spirv_entry);
    so->spirv_entry = strdup(entry);
    free(s->log);
    s->log = log;
}

#pragma mark - compiler / precision

void mglReleaseShaderCompiler(GLMContext ctx)
{
    /* frees compiler scratch memory and glslang's process state (deferred while compiled
     * shader objects are alive); later compiles transparently re-initialize the compiler */
    mgl_shader_release_compiler_resources(ctx);
}

void mglGetShaderPrecisionFormat(GLMContext ctx, GLenum shadertype, GLenum precisiontype, GLint *range, GLint *precision)
{
    switch (shadertype)
    {
        case GL_VERTEX_SHADER:
        case GL_FRAGMENT_SHADER:
            break;
        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }

    switch (precisiontype)
    {
        /* Metal evaluates every GLSL precision as IEEE-754 binary32 / 32-bit two's complement */
        case GL_LOW_FLOAT:
        case GL_MEDIUM_FLOAT:
        case GL_HIGH_FLOAT:
            if (range) { range[0] = 127; range[1] = 127; }
            if (precision) *precision = 23;
            break;

        case GL_LOW_INT:
        case GL_MEDIUM_INT:
        case GL_HIGH_INT:
            if (range) { range[0] = 31; range[1] = 30; }
            if (precision) *precision = 0;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return;
    }
}

#pragma mark - draw-time validation (pre_draw hook)

static bool program_pre_draw(GLMContext ctx, GLenum mode, GLsizei vertex_count, GLsizei instance_count)
{
    (void)mode; (void)vertex_count; (void)instance_count;

    if (ctx->state.program)
    {
        Program *p = ctx->state.program;
        if (MGL_PROGRAM_OBJ(p)->num_samplers > 1)
        {
            char *log = NULL;
            bool ok = mgl_program_validate_samplers(ctx, &p, 1, &log);
            free(log);
            if (!ok)
            {
                ERROR_RETURN(GL_INVALID_OPERATION);
                return false;
            }
        }
        return true;
    }

    if (STATE(program_pipeline))
    {
        char *log = NULL;
        bool ok = validate_pipeline(ctx, MGL_PIPELINE_OBJ(STATE(program_pipeline)), &log);
        if (!ok)
        {
            if (getenv("MGL_DEBUG") && log)
                fprintf(stderr, "MGL: pipeline validation: %s\n", log);
            free(log);
            ERROR_RETURN(GL_INVALID_OPERATION);
            return false;
        }
        free(log);
    }

    return true;
}

#pragma mark - domain 'program' hooks

void mgl_ext_init_program(GLMContext ctx)
{
    if (!STATE(ext_program))
        STATE(ext_program) = (struct MGLExtState_program *)calloc(1, sizeof(struct MGLExtState_program));

    if (STATE(program_pipeline_table).keys == NULL)
        initHashTable(&STATE(program_pipeline_table), 32);

    STATE_VAR(num_program_binary_formats) = 1;
    STATE_VAR(program_binary_formats) = MGL_PROGRAM_BINARY_FORMAT;
    STATE_VAR(num_shader_binary_formats) = 1;
    STATE_VAR(shader_binary_formats) = GL_SHADER_BINARY_FORMAT_SPIR_V;
    STATE_VAR(current_program) = 0;
    STATE_VAR(program_pipeline_binding) = 0;

    mgl_ext_register_pre_draw(program_pre_draw);
}

bool mgl_ext_get_program(GLMContext ctx, GLenum pname, GLuint type, void *data)
{
    switch (pname)
    {
        case GL_NUM_SPIR_V_EXTENSIONS:
            mgl_get_store(type, data, 0, (double)NUM_SPIRV_EXTENSIONS);
            return true;
        case GL_NUM_PROGRAM_BINARY_FORMATS:
        case GL_NUM_SHADER_BINARY_FORMATS:
            mgl_get_store(type, data, 0, 1.0);
            return true;
        case GL_PROGRAM_BINARY_FORMATS:
            mgl_get_store(type, data, 0, (double)MGL_PROGRAM_BINARY_FORMAT);
            return true;
        case GL_SHADER_BINARY_FORMATS:
            mgl_get_store(type, data, 0, (double)GL_SHADER_BINARY_FORMAT_SPIR_V);
            return true;
        case GL_CURRENT_PROGRAM:
            mgl_get_store(type, data, 0, ctx->state.program ? (double)ctx->state.program->name : 0.0);
            return true;
        case GL_PROGRAM_PIPELINE_BINDING:
            mgl_get_store(type, data, 0, STATE(program_pipeline) ? (double)STATE(program_pipeline)->name : 0.0);
            return true;
    }

    return false;
}
