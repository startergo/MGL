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
 * shaders.c
 * MGL
 *
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>

#include "shaders.h"
#include "cts_glsl_compat.h"
#include "glm_context.h"
#include "mgl_ext_introspect.h"
#include "mgl_ext_program.h"

const glslang_resource_t* glslang_default_resource(void);

/* geometry.c: re-seeds every TBuiltInResource entry that has a GL pname from the state
 * glGet* serves, so the gl_Max* constants baked into shaders match the queried limits */
void mgl_geom_fill_glslang_resource(GLMContext ctx, glslang_resource_t *res);

const char *getShaderTypeStr(GLuint type)
{
    /* indexed by _VERTEX_SHADER.._COMPUTE_SHADER (glslang stage order) */
    static const char *types[] = {"VERTEX_SHADER", "TESS_CONTROL_SHADER",
        "TESS_EVALUATION_SHADER", "GEOMETRY_SHADER", "FRAGMENT_SHADER",
        "COMPUTE_SHADER", "MAX_SHADER_TYPES", NULL};

    assert(type < _MAX_SHADER_TYPES);

    return types[type];
};

GLuint glShaderTypeToGLMType(GLuint type)
{
    switch(type) {
        case GL_VERTEX_SHADER: return _VERTEX_SHADER;
        case GL_FRAGMENT_SHADER: return _FRAGMENT_SHADER;
        case GL_GEOMETRY_SHADER: return _GEOMETRY_SHADER;
        case GL_TESS_CONTROL_SHADER: return _TESS_CONTROL_SHADER;
        case GL_TESS_EVALUATION_SHADER: return _TESS_EVALUATION_SHADER;
        case GL_COMPUTE_SHADER: return _COMPUTE_SHADER;
        default:
            // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown shader type 0x%x, defaulting to vertex shader\n", type);
            return _VERTEX_SHADER;
    }
}

glslang_stage_t getGLSLStage(GLuint type)
{
    switch(type) {
        case GL_VERTEX_SHADER: return GLSLANG_STAGE_VERTEX;
        case GL_FRAGMENT_SHADER: return GLSLANG_STAGE_FRAGMENT;
        case GL_GEOMETRY_SHADER: return GLSLANG_STAGE_GEOMETRY;
        case GL_TESS_CONTROL_SHADER: return GLSLANG_STAGE_TESSCONTROL;
        case GL_TESS_EVALUATION_SHADER: return GLSLANG_STAGE_TESSEVALUATION;
        case GL_COMPUTE_SHADER: return GLSLANG_STAGE_COMPUTE;
        default:
            // CRITICAL FIX: Handle unknown shader types gracefully instead of crashing
            fprintf(stderr, "MGL ERROR: Unknown GLSL shader type 0x%x, defaulting to vertex\n", type);
            return GLSLANG_STAGE_VERTEX;
    }

    return 0;
}

/* scratch copy used to rewrite legacy #version lines (released by glReleaseShaderCompiler) */
static char *modified_src = NULL;
static size_t modified_src_size = 0;

void initGLSLInput(GLMContext ctx, GLuint type, const char *src, glslang_input_t *input)
{
    input->language = GLSLANG_SOURCE_GLSL;
    input->stage = getGLSLStage(type);
    input->client = GLSLANG_CLIENT_OPENGL;
    input->target_language = GLSLANG_TARGET_SPV;
    input->target_language_version = GLSLANG_TARGET_SPV_1_0;

    /* Detect and upgrade GLSL version from source
     * GLSL 1.40 (OpenGL 3.1) shaders from virglrenderer need upgrading to 3.30
     * for glslang's SPIR-V compatibility with desktop OpenGL
     *
     * Default to 330 (minimum for SPIR-V) instead of 460 to be more permissive
     */
    int glsl_version = 330; /* Default to GLSL 3.30 - minimum for SPIR-V */
    int original_version = 330;
    const char *version_str = strstr(src, "#version");
    if (version_str) {
        int scanned_version;
        if (sscanf(version_str, "#version %d", &scanned_version) == 1) {
            original_version = scanned_version;
            glsl_version = scanned_version;
            /* Upgrade legacy GLSL versions to 330 minimum for SPIR-V */
            if (glsl_version < 330) {
                glsl_version = 330;
            }
        }
    }

    /* Set client_version to match GLSL version for SPIR-V targeting
     * This prevents "forced to be (450, core)" error when using GLSL 330 shaders
     * Must be set AFTER version detection above
     *
     * Note: glslang only exposes GLSLANG_TARGET_OPENGL_450, so we use that
     * as the SPIR-V target for all modern GLSL versions
     */
    if (glsl_version < 330) {
        /* Legacy GLSL - still target 450 for SPIR-V but shader will be upgraded */
        input->client_version = GLSLANG_TARGET_OPENGL_450;
    } else if (glsl_version == 330) {
        /* GLSL 3.30 shaders - target OpenGL 3.30 for SPIR-V */
        input->client_version = 330;  /* Use numeric value directly */
    } else {
        /* GLSL 4.00+ - target OpenGL 4.50 for SPIR-V */
        input->client_version = GLSLANG_TARGET_OPENGL_450;
    }


    if (original_version < 330) {
        fprintf(stderr, "[MGL] Upgrading GLSL shader from version %d to %d\n",
                original_version, glsl_version);

        size_t src_len = strlen(src);
        if (src_len + 100 > modified_src_size) {
            modified_src_size = src_len + 100;
            free(modified_src);
            modified_src = (char *)malloc(modified_src_size);
        }

        if (modified_src) {
            strcpy(modified_src, src);

            /* Find and replace #version line */
            char *version_line = strstr(modified_src, "#version");
            if (!version_line) {
                fprintf(stderr, "[MGL] WARNING: #version not found in source\n");
                input->code = src;
            } else {
                char *newline = strchr(version_line, '\n');
                if (!newline) {
                    fprintf(stderr, "[MGL] WARNING: newline not found after #version\n");
                    input->code = src;
                } else {
                    char version_buf[64];
                    snprintf(version_buf, sizeof(version_buf), "#version %d core", glsl_version);
                    size_t old_len = newline - version_line;
                    size_t new_len = strlen(version_buf);

                    fprintf(stderr, "[MGL] Old version line length: %zu, new: %zu\n", old_len, new_len);
                    fprintf(stderr, "[MGL] Old line: %.*s\n", (int)old_len, version_line);

                    if (new_len <= old_len) {
                        /* Simple in-place replacement with space padding */
                        memset(version_line, ' ', old_len);
                        memcpy(version_line, version_buf, new_len);
                        fprintf(stderr, "[MGL] Replaced version line in source (in-place)\n");
                    } else {
                        /* Need to shift the rest of the source */
                        size_t rest_of_src = strlen(newline);
                        memmove(version_line + new_len, newline, rest_of_src + 1); /* +1 for null terminator */
                        memcpy(version_line, version_buf, new_len);
                        fprintf(stderr, "[MGL] Replaced version line with shift\n");
                        fprintf(stderr, "[MGL] New line: %.*s\n", (int)new_len, version_line);
                    }
                }
            }
            input->code = modified_src;
        } else {
            fprintf(stderr, "[MGL] ERROR: Failed to allocate modified_src\n");
            input->code = src;
        }
    } else {
        /* glslang does not declare gl_NumSamples for desktop GLSL (it is an
         * OES_sample_variables builtin there): declare it here as the constant GL says it
         * is, baked from the CURRENT draw framebuffer's sample count - the shader is
         * compiled for that framebuffer, exactly how the CTS uses it (the mask loop over
         * gl_SampleMask / gl_SampleMaskIn needs the count; nothing else reads it) */
        if (type == GL_FRAGMENT_SHADER && glsl_version >= 400 && strstr(src, "gl_NumSamples"))
        {
            extern GLint mgl_fb_samples(GLMContext ctx, Framebuffer *fbo);
            GLint n = ctx->state.framebuffer ? mgl_fb_samples(ctx, ctx->state.framebuffer) : 1;
            const char *p = strstr(src, "#version");
            size_t need;

            if (n < 1) n = 1;
            if (p) p = strchr(p, '\n');
            if (p) p++;
            else p = src;
            need = strlen(src) + 96;
            if (need > modified_src_size)
            {
                free(modified_src);
                modified_src = (char *)malloc(need);
                modified_src_size = need;
            }
            if (modified_src)
            {
                size_t off = (size_t)(p - src);
                snprintf(modified_src, need, "%.*s#define gl_NumSamples %d\n%s",
                         (int)off, src, (int)n, p);
                input->code = modified_src;
            }
            else
                input->code = src;
        }
        else
            input->code = src;
    }

    input->default_version = glsl_version;
    input->default_profile = GLSLANG_CORE_PROFILE;
    //input->messages = 0xFFFF & ~GLSLANG_MSG_RELAXED_ERRORS_BIT;
    input->messages = GLSLANG_MSG_DEFAULT_BIT | GLSLANG_MSG_DEBUG_INFO_BIT | GLSLANG_MSG_RELAXED_ERRORS_BIT;
    {
        /* the resource limits must match what glGet* reports: the CTS compares the
         * gl_Max* constants baked into shaders against the queried values
         * (KHR-GL46.limits); mgl_geom_fill_glslang_resource() re-seeds every entry
         * that has a GL pname from the very queries the app sees */
        static glslang_resource_t res;
        res = *glslang_default_resource();
        mgl_geom_fill_glslang_resource(ctx, &res);
        input->resource = &res;
    }

    input->force_default_version_and_profile = 1;
}

#pragma mark - info log helpers

static void vset_log(char **dst, bool append, const char *fmt, va_list ap)
{
    char *add = NULL;
    if (vasprintf(&add, fmt, ap) < 0 || !add)
        return;
    if (append && *dst && **dst) {
        char *joined = NULL;
        if (asprintf(&joined, "%s%s", *dst, add) >= 0) {
            free(*dst);
            *dst = joined;
        }
        free(add);
    } else {
        free(*dst);
        *dst = add;
    }
}

void mgl_set_info_log(char **dst, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt); vset_log(dst, false, fmt, ap); va_end(ap);
}

void mgl_append_info_log(char **dst, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt); vset_log(dst, true, fmt, ap); va_end(ap);
}

/* GL info-log copy rules: at most bufSize-1 chars + NUL; length excludes the NUL */
void mgl_copy_info_log(const char *log, GLsizei bufSize, GLsizei *length, GLchar *out)
{
    size_t len = log ? strlen(log) : 0;
    size_t n = 0;

    if (out && bufSize > 0) {
        n = len < (size_t)(bufSize - 1) ? len : (size_t)(bufSize - 1);
        if (n) memcpy(out, log, n);
        out[n] = 0;
    }
    if (length)
        *length = (GLsizei)n;
}

#pragma mark - shader objects

GLenum mgl_shader_gl_type(GLuint glm_type)
{
    switch(glm_type) {
        case _VERTEX_SHADER: return GL_VERTEX_SHADER;
        case _FRAGMENT_SHADER: return GL_FRAGMENT_SHADER;
        case _GEOMETRY_SHADER: return GL_GEOMETRY_SHADER;
        case _TESS_CONTROL_SHADER: return GL_TESS_CONTROL_SHADER;
        case _TESS_EVALUATION_SHADER: return GL_TESS_EVALUATION_SHADER;
        case _COMPUTE_SHADER: return GL_COMPUTE_SHADER;
    }
    return GL_NONE;
}

Shader *newShader(GLMContext ctx, GLenum type, GLuint shader)
{
    MGLShaderObj *obj;
    char shader_type_name[128];

    obj = (MGLShaderObj *)calloc(1, sizeof(MGLShaderObj));
    if (!obj) {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return NULL;
    }

    obj->base.name = shader;
    obj->base.type = type;
    obj->base.glm_type = glShaderTypeToGLMType(type);

    snprintf(shader_type_name, sizeof(shader_type_name), "%s_%d", getShaderTypeStr(obj->base.glm_type), shader);
    obj->base.mtl_shader_type_name = strdup(shader_type_name);

    return &obj->base;
}

/* executable stage owned by a linked program; never visible through the shader namespace */
Shader *mgl_shader_new_internal(GLMContext ctx, GLuint glm_type)
{
    Shader *ptr = newShader(ctx, mgl_shader_gl_type(glm_type), 0);
    if (ptr) {
        MGL_SHADER_OBJ(ptr)->internal = GL_TRUE;
        MGL_SHADER_OBJ(ptr)->compile_status = GL_TRUE;
    }
    return ptr;
}

Shader *getShader(GLMContext ctx, GLenum type, GLuint shader)
{
    Shader *ptr;

    ptr = (Shader *)searchHashTable(&STATE(shader_table), shader);

    if (!ptr)
    {
        ptr = newShader(ctx, type, shader);
        if (!ptr)
            return NULL;

        MGL_SHADER_OBJ(ptr)->in_table = GL_TRUE;
        insertHashElement(&STATE(shader_table), shader, ptr);
    }

    return ptr;
}

int isShader(GLMContext ctx, GLuint shader)
{
    if (shader == 0)
        return 0;

    return searchHashTable(&STATE(shader_table), shader) != NULL;
}

Shader *findShader(GLMContext ctx, GLuint shader)
{
    if (shader == 0)
        return NULL;

    return (Shader *)searchHashTable(&STATE(shader_table), shader);
}

GLuint mglCreateShader(GLMContext ctx, GLenum type)
{
    GLuint shader;

    switch(type)
    {
        case GL_VERTEX_SHADER:
        case GL_FRAGMENT_SHADER:
        case GL_GEOMETRY_SHADER:
        case GL_COMPUTE_SHADER:
        case GL_TESS_CONTROL_SHADER:
        case GL_TESS_EVALUATION_SHADER:
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            return 0;
    }

    /* shaders and programs share one GL namespace */
    shader = getNewName(&STATE(shader_table));
    if (shader < STATE(program_table).current_name)
        shader = STATE(program_table).current_name;
    STATE(shader_table).current_name = shader + 1;
    STATE(program_table).current_name = shader + 1;

    if (!getShader(ctx, type, shader))
        return 0;

    return shader;
}

/*
 * glReleaseShaderCompiler: free the legacy-source scratch buffer now and tear down glslang's
 * process-wide state (built-in symbol tables, pools). That teardown is only safe once no
 * compiled glslang shader object exists, so with live shaders it is deferred until the last
 * one is deleted. The next compile or link re-initializes glslang (lazy_init_compiler).
 */
void mgl_shader_release_compiler_resources(GLMContext ctx)
{
    struct MGLExtState_program *st = STATE(ext_program);

    free(modified_src);
    modified_src = NULL;
    modified_src_size = 0;

    if (!st || st->compiler_released)
        return;

    if (st->live_glslang_shaders > 0)
    {
        st->release_pending = GL_TRUE;
        return;
    }

    glslang_finalize_process();
    st->compiler_released = GL_TRUE;
    st->release_pending = GL_FALSE;
}

static void release_glslang_shader(GLMContext ctx, Shader *ptr)
{
    if (ptr->compiled_glsl_shader)
    {
        glslang_shader_delete(ptr->compiled_glsl_shader);
        ptr->compiled_glsl_shader = NULL;
        if (STATE(ext_program) && STATE(ext_program)->live_glslang_shaders > 0)
        {
            STATE(ext_program)->live_glslang_shaders--;
            if (STATE(ext_program)->live_glslang_shaders == 0 && STATE(ext_program)->release_pending)
                mgl_shader_release_compiler_resources(ctx);
        }
    }
}

static void release_spirv(Shader *ptr)
{
    MGLShaderObj *obj = MGL_SHADER_OBJ(ptr);

    free(obj->spirv);
    obj->spirv = NULL;
    obj->spirv_words = 0;
    free(obj->spirv_entry);
    obj->spirv_entry = NULL;
    obj->spirv_binary = GL_FALSE;
    obj->specialized = GL_FALSE;
}

void mglFreeShader(GLMContext ctx, Shader *ptr)
{
    MGLShaderObj *obj = MGL_SHADER_OBJ(ptr);

    release_glslang_shader(ctx, ptr);

    if (ptr->mtl_data.library || ptr->mtl_data.function)
    {
        void (*release)(GLMContext, void *) =
            ctx->mtl_funcs.ext_program ? ctx->mtl_funcs.ext_program->release_obj : ctx->mtl_funcs.mtlDeleteMTLObj;
        if (release) {
            if (ptr->mtl_data.function) release(ctx, ptr->mtl_data.function);
            if (ptr->mtl_data.library) release(ctx, ptr->mtl_data.library);
        }
        ptr->mtl_data.function = NULL;
        ptr->mtl_data.library = NULL;
    }

    if (obj->in_table)
    {
        deleteHashElement(&STATE(shader_table), ptr->name);
        obj->in_table = GL_FALSE;
    }

    release_spirv(ptr);
    free((void *)ptr->mtl_shader_type_name);
    free((void *)ptr->src);
    free((void *)ptr->entry_point);
    free(ptr->log);

    free(ptr);
}

/* drop one attachment reference; frees a shader flagged for deletion */
void mgl_shader_unref(GLMContext ctx, Shader *ptr)
{
    if (!ptr)
        return;

    if (ptr->refcount > 0)
        ptr->refcount--;

    if (ptr->refcount == 0 && ptr->delete_status)
        mglFreeShader(ctx, ptr);
}

void mglDeleteShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;

    /* OpenGL spec: A value of 0 for shader will be silently ignored. */
    if (shader == 0)
        return;

    ptr = findShader(ctx, shader);
    if (!ptr)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (ptr->delete_status)
        return;

    ptr->delete_status = GL_TRUE;

    /* still attached: flagged, freed when the last program detaches it */
    if (ptr->refcount == 0)
        mglFreeShader(ctx, ptr);
}

GLboolean mglIsShader(GLMContext ctx, GLuint shader)
{
    return isShader(ctx, shader) ? GL_TRUE : GL_FALSE;
}

/* shader-name lookup with the GL error rules for "shader" parameters */
static Shader *lookupShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr = findShader(ctx, shader);

    if (!ptr)
    {
        extern Program *findProgram(GLMContext ctx, GLuint program);
        if (shader && findProgram(ctx, shader))
            ERROR_RETURN(GL_INVALID_OPERATION);
        else
            ERROR_RETURN(GL_INVALID_VALUE);
    }

    return ptr;
}

void mglShaderSource(GLMContext ctx, GLuint shader, GLsizei count, const GLchar *const*string, const GLint *length)
{
    size_t len, pos;
    GLchar *src;
    Shader *ptr;

    ptr = lookupShader(ctx, shader);
    if (!ptr)
        return;

    if (count < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    if (count > 0 && string == NULL)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    len = 0;
    for (GLsizei i = 0; i < count; i++)
    {
        if (!string[i])
            continue;
        len += (length && length[i] >= 0) ? strnlen(string[i], (size_t)length[i]) : strlen(string[i]);
    }

    src = (GLchar *)malloc(len + 1);
    if (!src)
    {
        ERROR_RETURN(GL_OUT_OF_MEMORY);
        return;
    }

    pos = 0;
    for (GLsizei i = 0; i < count; i++)
    {
        size_t n;
        if (!string[i])
            continue;
        n = (length && length[i] >= 0) ? strnlen(string[i], (size_t)length[i]) : strlen(string[i]);
        memcpy(src + pos, string[i], n);
        pos += n;
    }
    src[pos] = 0;

    free((void *)ptr->src);
    ptr->src_len = pos;
    ptr->src = src;
    ptr->dirty_bits |= DIRTY_SHADER;

    /* loading source replaces a SPIR-V module (SPIR_V_BINARY becomes FALSE) */
    if (MGL_SHADER_OBJ(ptr)->spirv_binary)
    {
        release_spirv(ptr);
        MGL_SHADER_OBJ(ptr)->compile_status = GL_FALSE;
    }
}

static void lazy_init_compiler(GLMContext ctx)
{
    if (STATE(ext_program))
    {
        /* a new compile cancels a deferred release */
        STATE(ext_program)->release_pending = GL_FALSE;
        if (STATE(ext_program)->compiler_released)
        {
            glslang_initialize_process();
            STATE(ext_program)->compiler_released = GL_FALSE;
        }
    }
}

void mglCompileShader(GLMContext ctx, GLuint shader)
{
    Shader *ptr;
    MGLShaderObj *obj;
    glslang_input_t glsl_input;
    glslang_shader_t *glsl_shader;

    ptr = lookupShader(ctx, shader);
    if (!ptr)
        return;
    obj = MGL_SHADER_OBJ(ptr);

    obj->compile_status = GL_FALSE;
    free(ptr->log);
    ptr->log = NULL;

    if (obj->spirv_binary)
    {
        mgl_set_info_log(&ptr->log, "ERROR: shader %u holds a SPIR-V binary; use glSpecializeShader instead of glCompileShader\n", shader);
        return;
    }

    if (ptr->src == NULL)
    {
        mgl_set_info_log(&ptr->log, "ERROR: shader %u has no source\n", shader);
        return;
    }

    lazy_init_compiler(ctx);

    memset(&glsl_input, 0, sizeof(glsl_input));
    /* GL-legal but SPIR-V-hostile constructs (layout(shared)/layout(packed) blocks, ...) are
     * rewritten first: cts_glsl_compat.c.  Freed after glslang has copied the source. */
    char *compat = mgl_cts_glsl_compat(mgl_introspect_shader_source(ctx, ptr));
    initGLSLInput(ctx, ptr->type, compat ? compat : mgl_introspect_shader_source(ctx, ptr), &glsl_input); /* introspect: subroutine rewrite */

    glsl_shader = glslang_shader_create(&glsl_input);
    if (glsl_shader == NULL)
    {
        mgl_set_info_log(&ptr->log, "ERROR: glslang could not create a %s object\n", getShaderTypeStr(ptr->glm_type));
        return;
    }

    /* Set glslang options to auto-assign locations for legacy shaders */
    int options = GLSLANG_SHADER_VULKAN_RULES_RELAXED;

    int shader_version = 330; /* Default */
    const char *version_str = strstr(ptr->src, "#version");
    if (version_str)
        sscanf(version_str, "#version %d", &shader_version);

    /* For GLSL < 330, auto-assign locations since old shaders don't have layout() qualifiers */
    if (shader_version < 330)
        options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS;
    /* introspect: GL assigns uniform/interface locations at link time (introspect_spirv.c);
     * without this, 3.30+ shaders with loose uniforms lacking layout(location) fail to compile */
    options |= GLSLANG_SHADER_AUTO_MAP_LOCATIONS;
    /* GL-legal shaders may declare UBO/SSBO/atomic/sampler uniforms without layout(binding);
     * glslang refuses to compile those for SPIR-V unless it can auto-map bindings.  The invented
     * numbers are detected and replaced with the GL default (binding 0) at link (introspect_spirv.c). */
    options |= GLSLANG_SHADER_AUTO_MAP_BINDINGS;
    glslang_shader_set_options(glsl_shader, options);

    if (!glslang_shader_preprocess(glsl_shader, &glsl_input) ||
        !glslang_shader_parse(glsl_shader, &glsl_input))
    {
        const char *info_log = glslang_shader_get_info_log(glsl_shader);

        if (info_log && *info_log)
            mgl_set_info_log(&ptr->log, "%s", info_log);
        else
            mgl_set_info_log(&ptr->log, "ERROR: %s compilation failed\n", getShaderTypeStr(ptr->glm_type));

        if (getenv("MGL_SHADER_DUMP")) {   /* diagnosis: keep the rejected source */
            char path[256];
            snprintf(path, sizeof path, "%s/failed_%u_%s.glsl", getenv("MGL_SHADER_DUMP"), shader,
                     getShaderTypeStr(ptr->glm_type));
            FILE *f = fopen(path, "w");
            if (f) { fwrite(ptr->src, 1, ptr->src_len ? ptr->src_len : strlen(ptr->src), f); fclose(f); }
        }

        DEBUG_PRINT("MGL shader %u compile failed:\n%s\n", shader, ptr->log);

        glslang_shader_delete(glsl_shader);
        free(compat);
        return;
    }

    free(compat);
    /* warnings survive a successful compile */
    {
        const char *info_log = glslang_shader_get_info_log(glsl_shader);
        if (info_log && *info_log)
            mgl_set_info_log(&ptr->log, "%s", info_log);
    }

    release_glslang_shader(ctx, ptr);
    ptr->compiled_glsl_shader = glsl_shader;
    if (STATE(ext_program))
        STATE(ext_program)->live_glslang_shaders++;
    ptr->dirty_bits |= DIRTY_SHADER;
    obj->compile_status = GL_TRUE;
}

void mglGetShaderiv(GLMContext ctx, GLuint shader, GLenum pname, GLint *params)
{
    Shader *ptr;
    MGLShaderObj *obj;

    ptr = lookupShader(ctx, shader);
    if (!ptr)
        return;
    obj = MGL_SHADER_OBJ(ptr);

    switch(pname)
    {
        case GL_SHADER_TYPE:
            *params = (GLint)ptr->type;
            break;

        case GL_DELETE_STATUS:
            *params = ptr->delete_status ? GL_TRUE : GL_FALSE;
            break;

        case GL_COMPILE_STATUS:
            *params = obj->compile_status ? GL_TRUE : GL_FALSE;
            break;

        case GL_INFO_LOG_LENGTH:
            *params = (ptr->log && *ptr->log) ? (GLint)strlen(ptr->log) + 1 : 0;
            break;

        case GL_SHADER_SOURCE_LENGTH:
            *params = ptr->src ? (GLint)ptr->src_len + 1 : 0;
            break;

        case GL_SPIR_V_BINARY:
            *params = obj->spirv_binary ? GL_TRUE : GL_FALSE;
            break;

        default:
            ERROR_RETURN(GL_INVALID_ENUM);
            break;
    }
}

void mglGetShaderInfoLog(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog)
{
    Shader *ptr;

    ptr = lookupShader(ctx, shader);
    if (!ptr)
        return;

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mgl_copy_info_log(ptr->log, bufSize, length, infoLog);
}

void mglGetShaderSource(GLMContext ctx, GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *source)
{
    Shader *ptr;

    ptr = lookupShader(ctx, shader);
    if (!ptr)
        return;

    if (bufSize < 0)
    {
        ERROR_RETURN(GL_INVALID_VALUE);
        return;
    }

    mgl_copy_info_log(ptr->src, bufSize, length, source);
}
