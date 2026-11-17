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
 * glm_context.h
 * MGL
 *
 */

#ifndef glm_context_h
#define glm_context_h

#include <stdio.h>
#include <assert.h>

#include <mach/vm_types.h>
#include <glslang_c_interface.h>
#include <glslang_c_shader_types.h>

#include "glm_dispatch.h"

#include "hash_table.h"
#include "mgl_ext.h"

// defines above set sizes in glm_params
#include "glm_params.h"

#ifdef DEBUG
#define DEBUG_LEVEL 3
#endif

#if defined(DEBUG_LEVEL) && DEBUG_LEVEL > 0
 #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d: " fmt, \
    __func__, __LINE__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif


// macros because I get tired of write if this and that then return
#define RETURN_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return; }
#define RETURN_FALSE_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_FALSE_ON_NULL(_expr_) if (_expr_ == NULL) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return false; }
#define RETURN_NULL_ON_FAILURE(_expr_) if (_expr_ == false) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return NULL; }
#define RETURN_ON_NULL(_expr_) if (_expr_ == NULL) { printf("failure %s:%d\n",__FUNCTION__,__LINE__); return; }

#define STATE(_VAR_)     ctx->state._VAR_
#define STATE_VAR(_VAR_) ctx->state.var._VAR_

#define VAO()   ctx->state.vao
#define VAO_STATE(_val_)   ctx->state.vao->_val_
#define VAO_ATTRIB_STATE(_index_) ctx->state.vao->attrib[_index_]

#define ERROR_RETURN(_type_) ctx->error_func(ctx, __FUNCTION__, _type_)
#define ERROR_RETURN_VALUE(_type_, _val_) ctx->error_func(ctx, __FUNCTION__, _type_); return _val_
#define ERROR_CHECK_RETURN(_expr_, _type_) if ((_expr_) == false) {ctx->error_func(ctx, __FUNCTION__, _type_);}
#define ERROR_CHECK_RETURN_VALUE(_expr_, _type_, _val_) if ((_expr_) == false) {ctx->error_func(ctx, __FUNCTION__, _type_); return _val_;}

enum {
    _TEXTURE_BUFFER = 0, // duplicate of _TEXTURE_BUFFER_TARGET
    _ARRAY_BUFFER,
    _ELEMENT_ARRAY_BUFFER,
    _UNIFORM_BUFFER,
    _UNIFORM_CONSTANT,
    _SHADER_STORAGE_BUFFER,
    _TRANSFORM_FEEDBACK_BUFFER,
    _QUERY_BUFFER,
    _PIXEL_PACK_BUFFER,
    _PIXEL_UNPACK_BUFFER,
    _ATOMIC_COUNTER_BUFFER,
    _COPY_READ_BUFFER,
    _COPY_WRITE_BUFFER,
    _DISPATCH_INDIRECT_BUFFER,
    _DRAW_INDIRECT_BUFFER,
    _MAX_BUFFER_TYPES
};

enum {
    _TEXTURE_BUFFER_TARGET = 0, // duplicate of _TEXTURE_BUFFER
    _TEXTURE_1D,
    _TEXTURE_2D,
    _TEXTURE_3D,
    _TEXTURE_RECTANGLE,
    _TEXTURE_1D_ARRAY,
    _TEXTURE_2D_ARRAY,
    _TEXTURE_CUBE_MAP,
    _TEXTURE_CUBE_MAP_ARRAY,
    _TEXTURE_2D_MULTISAMPLE,
    _TEXTURE_2D_MULTISAMPLE_ARRAY,
    _RENDERBUFFER,
    _MAX_TEXTURE_TYPES
};

static_assert(_TEXTURE_BUFFER == _TEXTURE_BUFFER_TARGET, "_TEXTURE_BUFFER != _TEXTURE_BUFFER_TARGET");

enum {
    _VERTEX_SHADER = 0,
    _TESS_CONTROL_SHADER,
    _TESS_EVALUATION_SHADER,
    _GEOMETRY_SHADER,
    _FRAGMENT_SHADER,
    _COMPUTE_SHADER,
    _MAX_SHADER_TYPES
};

enum {
    _UNIFORM_BASE = 0,
    _TRANSFORM_FEEDBACK_BASE,
    _SHADER_STORAGE_BASE,
    _ATOMIC_COUNTER_BASE,
    _MAX_BASE_TARGET
};

enum {
    _UNKNOWN_RES = 0,
    _UNIFORM_BUFFER_RES,
    _UNIFORM_CONSTANT_RES,
    _STORAGE_BUFFER_RES,
    _STAGE_INPUT_RES,
    _STAGE_OUTPUT_RES,
    _STORAGE_OUTPUT_RES,
    _ATOMIC_COUNTER_RES,
    _PUSH_CONSTANT_RES,
    _SEPARATE_IMAGE_RES,
    _SEPARATE_SAMPLERS_RES,
    _ACCEL_STRUCT_RES,
    _RAY_QUERY,
    _MAX_SPIRV_RES
};

#define SHADER_MASK_BIT(_TYPE_)    (0x1 << _TYPE_)
#define VERTEX_SHADER_MASK_BIT  SHADER_MASK_BIT(_VERTEX_SHADER)
#define FRAGMENT_SHADER_MASK_BIT  SHADER_MASK_BIT(_FRAGMENT_SHADER)
#define GEOMETRY_SHADER_MASK_BIT  SHADER_MASK_BIT(_GEOMETRY_SHADER)
#define TESS_CONTROL_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_CONTROL_SHADER)
#define TESS_EVALUATION_SHADER_MASK_BIT  SHADER_MASK_BIT(_TESS_EVALUATION_SHADER)
#define COMPUTE_SHADER_MASK_BIT  SHADER_MASK_BIT(_COMPUTE_SHADER)

#define DIRTY_BUFFER_DATA   0x1
#define DIRTY_BUFFER_ADDR   (DIRTY_BUFFER_DATA << 1)

#define DIRTY_TEXTURE_LEVEL 0x1
#define DIRTY_TEXTURE_DATA  (DIRTY_TEXTURE_LEVEL << 1)
#define DIRTY_TEXTURE_PARAM (DIRTY_TEXTURE_DATA << 1)
#define DIRTY_TEXTURE_ACCESS (DIRTY_TEXTURE_PARAM << 1)

#define DIRTY_FBO_BINDING   0x1
#define DIRTY_FBO_TEX      (DIRTY_FBO_BINDING << 1)

#define DIRTY_RENDBUF       0x1
#define DIRTY_RENDBUF_TEX   (DIRTY_RENDBUF << 1)

#define DIRTY_VAO_BUFFER_BASE  0x1
#define DIRTY_VAO_ATTRIB       (DIRTY_VAO_BUFFER_BASE << 1)

typedef struct {
    unsigned int  count;
    unsigned int  instanceCount;
    unsigned int  first;
    unsigned int  baseInstance;
} DrawArraysIndirectCommand;

typedef struct {
    unsigned int  count;
    unsigned int  instanceCount;
    unsigned int  first;
    int  baseVertex;
    unsigned int  baseInstance;
} DrawElementsIndirectCommand;

typedef struct BufferData_t {
    GLuint          dirty_bits;
    size_t          buffer_size;
    vm_address_t    buffer_data;
    void            *mtl_data;
    void            *pool;      /* r153: retired Metal storages of THIS buffer, reused once
                                 * their command buffer completed (MGLRenderer+Draw.m) */
} BufferData;

/* r153: release a buffer's pool of retired Metal storages (buffer deletion) */
void mgl_buffer_pool_free(BufferData *d);

#define BUFFER_IMMUTABLE_STORAGE_FLAG   0x1
#define BUFFER_MAP_PERSISTENT_BIT       (BUFFER_IMMUTABLE_STORAGE_FLAG << 1)

typedef struct Buffer_t {
    GLuint name;
    GLenum target;
    GLuint index;
    GLsizeiptr size;
    GLenum usage;
    GLenum access;
    GLbitfield access_flags;
    GLboolean immutable_storage; // GL_BUFFER_IMMUTABLE_STORAGE
    GLboolean mapped;
    GLuint storage_flags; // GL_BUFFER_STORAGE_FLAGS
    GLsizeiptr mapped_offset;
    GLsizeiptr mapped_length;
    BufferData data;
} Buffer;

typedef struct BufferBaseTarget_t {
    GLuint      buffer;
    GLsizeiptr  offset;
    GLsizeiptr  size;
    Buffer      *buf;
} BufferBaseTarget;

/* GL 4.6 minimum for GL_MAX_UNIFORM_BUFFER_BINDINGS is 84; the CTS multi_bind
 * functional cases bind a buffer on every reported point and read them back, so the
 * indexed binding array must really hold that many rows (the overflow table in
 * buffers.c only covered 8 of them). */
#define MAX_BINDABLE_BUFFERS    84
/* compat r111: GL uniform-buffer BINDING POINTS are pure indirection — the
 * Metal buffer slot is assigned per linked program, so supporting more GL
 * binding points than the STATE(buffer_base) rows costs only the small sparse
 * overflow table (buffers.c: points >= MAX_BINDABLE_BUFFERS live there and
 * the draw path reads UBO blocks through it).  rsvga's D3D9 translation
 * places its fragment constant block at binding 256 (the D3D9 PS constant
 * space starts after the VS's), so MGL advertises 512 — comfortably above the
 * GL 4.6 minimum of 84 and above the whole D3D9 VS+PS constant space. */
#define MGL_MAX_UNIFORM_BUFFER_BINDINGS 512
typedef struct BufferBase_t {
    BufferBaseTarget    buffers[MAX_BINDABLE_BUFFERS];
} BufferBase;

typedef struct TextureParameter_t {
    GLenum  depth_stencil_mode;
    /* GL_DEPTH_TEXTURE_MODE: how a non-comparison depth texture splats its
     * depth into the sampled vec4 (RED/LUMINANCE/INTENSITY/ALPHA) */
    GLenum  depth_texture_mode;
    GLuint  base_level;
    GLfloat border_color[4];
    GLint   border_color_i[4];
    GLuint   border_color_ui[4];
    GLenum  compare_func;
    GLenum  compare_mode;
    GLfloat lod_bias;
    GLenum  min_filter;
    GLenum  mag_filter;
    GLfloat max_anisotropy;
    GLfloat min_lod;
    GLfloat max_lod;
    GLuint  max_level;
    GLboolean swizzled;
    GLenum  swizzle_r;
    GLenum  swizzle_g;
    GLenum  swizzle_b;
    GLenum  swizzle_a;
    GLenum  wrap_s;
    GLenum  wrap_t;
    GLenum  wrap_r;
    /* GL_EXT_texture_sRGB_decode (compat): GL_DECODE_EXT / GL_SKIP_DECODE_EXT.
     * Sampler-state semantics: the sampler object bound to a unit replaces
     * the texture's value. Metal has no sampler flag for this — SKIP_DECODE
     * sampling binds a plain (non-sRGB) view of the texture's storage
     * (compatSRGBDecodeViewForTex, MGLRenderer+Compat.m). */
    GLenum  srgb_decode;
    void *mtl_data;
    /* r154: bumped (from a process-wide counter) every time mtl_data is
     * released by mglTexParamInvalidate.  The released cache object's address
     * is routinely reused by its successor, so identity caches keyed on
     * mtl_data alone (the draw-fast gate, the texture-resolution cache) must
     * also compare this — a glSamplerParameter on a bound sampler otherwise
     * kept sampling with the previous sampler state. */
    uint32_t mtl_serial;
} TextureParameter;

typedef struct TextureLevel_t {
    GLboolean complete;
    GLuint width;
    GLuint height;
    GLuint depth;
    size_t pitch;
    GLuint mtl_format;
    GLboolean gen_compressed;   /* created via a GENERIC compressed internal format
                                 * (GL_COMPRESSED_RGBA etc.): stored uncompressed
                                 * (GL 4.6 8.5.3 lets the GL choose), but ClearTexImage/
                                 * ClearTexSubImage must still fail (KHR-GL46
                                 * clear_tex_image.negative_*_compressed_texture) */
    size_t  data_size;
    vm_address_t data;
} TextureLevel;

enum {
    _CUBE_MAP_POSITIVE_X = 0,
    _CUBE_MAP_NEGATIVE_X,
    _CUBE_MAP_POSITIVE_Y,
    _CUBE_MAP_NEGATIVE_Y,
    _CUBE_MAP_POSITIVE_Z,
    _CUBE_MAP_NEGATIVE_Z,
    _CUBE_MAP_MAX_FACE
};

typedef struct TextureFace_t {
    TextureLevel    *levels;
} TextureFace;

#define DIRTY_SAMPLER_PARAM   0x1
typedef struct Sampler_t {
    GLuint dirty_bits;
    GLuint name;
    TextureParameter params;
    void *mtl_data;
} Sampler;

typedef struct Texture_t {
    GLuint dirty_bits;
    GLuint dirty_on_gpu;
    GLboolean is_render_target;
    GLenum access;
    GLboolean immutable_storage;
    GLuint name;
    GLuint target;
    GLuint index;
    GLuint mipmapped;
    GLboolean genmipmaps;
    GLboolean mtl_requires_private_storage; // depth, multi sample
    TextureParameter params;

    // base level params
    GLenum internalformat;
    GLuint width;
    GLuint height;
    GLuint depth;
    GLboolean is_array;
    GLboolean complete;
    GLuint num_levels;
    GLuint mipmap_levels;
    TextureFace faces[6];
    void    *mtl_data;
} Texture;

typedef struct TextureUnit_t {
    Texture *textures[_MAX_TEXTURE_TYPES];
} TextureUnit;

typedef struct ImageUnit_t {
    GLuint unit;
    GLuint texture;
    GLuint level;
    GLboolean layered;
    GLint layer;
    GLenum access;
    GLenum internalformat;
    Texture *tex;
} ImageUnit;

typedef struct BufferBinding_t {
    Buffer  *buffer;
    GLintptr offset;
    GLsizei stride;
    GLuint divisor;
} BufferBinding;

typedef struct VertexAttrib_t {
    Buffer  *buffer;
    GLuint  size;
    GLenum  type;
    GLuint  normalized;
    GLuint  stride;
    GLuint  divisor;
    GLintptr  relativeoffset;
    GLuint  buffer_bindingindex;
} VertexAttrib;

typedef struct VertexElementArray_t {
    Buffer  *buffer;
    GLenum  type;
    GLuint  size;
    const void *ptr;
} VertexElementArray;

typedef struct VertexArray_t {
    GLuint dirty_bits;
    unsigned name;
    unsigned enabled_attribs;
    VertexAttrib attrib[MAX_ATTRIBS];
    VertexElementArray element_array;
    void *mtl_data;
} VertexArray;

typedef struct Shader_t {
    GLuint dirty_bits;
    GLuint name;
    GLuint type;
    GLuint glm_type;
    const char *mtl_shader_type_name;
    size_t src_len;
    const char *src;
    glslang_shader_t *compiled_glsl_shader;
    const char *entry_point;
    char *log;
    int refcount;
    GLboolean delete_status;
    struct {
        void *function;
        void *library;
    } mtl_data;
} Shader;

typedef struct Spirv_t {
    GLuint stage;
    size_t size;
    unsigned int *ir;
    char *msl_str;
    char *entry_point;
    void *mtl_function;
    void *mtl_library;
    uint32_t shadow_slots;  /* geom: bitmask of Metal texture slots whose VERTEX MSL emulates
                             * shadow sampling (AGX returns raw values for sample_compare in
                             * vertex functions) — those slots bind a NON-compare sampler and
                             * take the unit's compare function as a function constant */
    uint32_t dub_msl;      /* geom: Metal buffer index of the packed default-uniform block
                             * argument (`mgl_DefaultUniforms`); SPIRV-Cross numbers it after
                             * the image-atomic emulation buffers, which shifts it off the
                             * renderer's buffer-map ordinal — mgl_msl_fix_image_atomics
                             * records it and the map binds it here (0xffffffff = none) */
    uint32_t fbfetch;     /* compat (GL_EXT_shader_framebuffer_fetch), FRAGMENT stage only:
                             * bits 0..7 = colour locations fetched through `inout' outputs,
                           * bits 8..11 = gl_LastFragData range (compat_fbfetch.c), recomputed
                           * from the shader source at every link (see run_link) */
    uint32_t sizes_scan;  /* geom: lazy memo of "does this stage's MSL read
                           * spvBufferSizeConstants" (0 unknown, 1 no, 2 yes).
                           * Stored ON the program: the previous static cache keyed
                           * on (Program*, msl_str*) + a 32-byte head memcmp aliased
                           * across recycled program/string addresses — every
                           * generated MSL starts with the same ~60 bytes, so the
                           * head guard never distinguishes two shaders and a fresh
                           * program inherited a dead one's answer (geomBindSSBOSizes
                           * then skipped the size-constant bind and .length() read
                           * zeros; KHR-GL46 advanced-unsizedArrayLength-fs flakes) */
} Spirv;
/* compat (GL_KHR_blend_equation_advanced): the per-stage blend_support_* mask is
 * NOT cached here any more — a field inside Spirv_t grew every Program and
 * shifted the hot struct layout (perf s30, glmark2 jellyfish -5%).  The mask is
 * re-scanned from Shader->src on the advanced-blend paths only (cold). */

typedef struct SpirvResource_t {
    GLuint  _id;
    GLuint  base_type_id;
    GLuint  type_id;
    const char *name;
    GLuint  set;
    GLuint  binding;
    GLuint  location;
    GLuint  msl_binding; /* Metal argument index SPIRV-Cross chose (differs from the GL unit for textures) */
    GLuint  array_size;  /* elements for arrayed resources (sampler/image arrays span that many argument slots and units) */
    GLuint  target;      /* GL texture target this sampler/image type reads (GL_TEXTURE_2D, ..._2D_MULTISAMPLE, ...); 0 = none */
    GLuint  msl_atomic;  /* geom: storage images only — Metal buffer index of SPIRV-Cross's
                          * `<name>_atomic` emulation argument (0xffffffff when the image has
                          * no atomics / was stripped); the renderer binds a sidecar buffer */
    GLuint  elem_class;  /* geom: storage images only — element kind of the SPIR-V image's
                          * sampled type: 0 float, 1 int, 2 uint (the unit-format view) */
    GLuint  *unit_vals;  /* geom: sampler/image ARRAY elements are independent GL uniforms
                          * (glUniform1i per element): element e's unit lives here when the app
                          * set it away from the default base+e (mgli_upload_uniform latches it;
                          * the pipeline composite's resource copy shares this array). NULL =
                          * all elements at the default binding+e */
} SpirvResource;

/* geom: element e's GL unit for an arrayed opaque resource (sampler/image array),
 * honouring the per-element uniforms the app set; see SpirvResource.unit_vals */
static inline GLuint mgl_res_elem_unit(const SpirvResource *r, GLuint e)
{
    return (r->unit_vals && e < r->array_size) ? r->unit_vals[e] : r->binding + e;
}

typedef struct SpirvResourceList_t {
    GLuint  count;
    SpirvResource   *list;
} SpirvResourceList;

typedef struct BufferMap_t {
    GLuint      buffer_base_index;
    GLuint      attribute_mask;
    Buffer      *buf;
    GLintptr    offset;
    /* byte size of the bound range (whole buffer for glBindBufferBase, the range
     * size for glBindBufferRange): SPIRV-Cross's spvBufferSizeConstants — the
     * runtime-array .length() implementation — reads it per bound SSBO */
    GLsizeiptr  size;
    GLuint      mtl_index;  /* geom: the Metal argument index the stage MSL gives this
                             * buffer (SpirvResource.msl_binding + array element) — the
                             * map's ordinal only coincides with it while no image-atomic
                             * emulation buffer interleaves in the MSL signature; the
                             * encoder binds here (falls back to the ordinal) */
    GLuint      from_ubo;   /* r166: 1 = enumerated from a uniform-buffer binding point
                             * (buffer_base_index is that point); the draw gate patches
                             * a same-buffer offset / size change into such entries */
} BufferMap;

typedef struct BufferMapList_t {
    GLuint      count;
    BufferMap   buffers[MAX_MAPPED_BUFFERS];
} BufferMapList;

typedef struct Program_t {
    GLuint dirty_bits;
    GLuint name;
    int refcount;
    GLboolean delete_status;
    Shader *shader_slots[_MAX_SHADER_TYPES];
    glslang_program_t *linked_glsl_program;
    Spirv spirv[_MAX_SHADER_TYPES];
    SpirvResourceList spirv_resources_list[_MAX_SHADER_TYPES][_MAX_SPIRV_RES];
    struct {
        unsigned x, y, z;
    } local_workgroup_size;
    int uses_draw_params;   /* tess: gl_DrawID read from the draw-params buffer (mgl_ext_tess.h) */
    int writes_cull_distance; /* tess: VS writes gl_CullDistance (mgl_geom_post_link computes;
                               * the per-draw mgl_geom_cull_active must not rescan the MSL —
                               * glmark2 ideas lost 42% to that strstr per draw) */
    void *mtl_data;
} Program;

typedef struct ProgramPipeline_t {
    GLuint name;
    GLboolean validated;
    Program *stage_programs[_MAX_SHADER_TYPES];  // Programs attached to each stage
} ProgramPipeline;

typedef struct TransformFeedback_t {
    GLuint name;
    GLenum target;
    GLboolean active;
    GLboolean paused;
    GLenum primitive_mode;
} TransformFeedback;

typedef struct Renderbuffer_t {
    GLuint dirty_bits;
    GLuint  name;
    GLboolean is_draw_buffer;
    Texture *tex;
} Renderbuffer;

typedef struct FBOAttachment_t {
    GLuint dirty_bits;
    GLuint textarget;   // GL_RENDERBUFFER for renderbuffers
    GLuint texture;
    GLuint level;
    GLuint layer;
    GLbitfield clear_bitmask;
    GLfloat clear_color[4];
    union {
        Texture *tex;
        Renderbuffer *rbo;
    } buf;
    void *seen_mtl;     /* r152: the attached texture's realized MTLTexture at attach time
                         * (framebuffers.c setAttachment: a re-attach of the identical,
                         * unchanged image is a no-op) */
} FBOAttachment;

typedef struct Framebuffer_t {
    GLuint dirty_bits;
    GLuint  name;
    GLbitfield color_attachment_bitfield;
    FBOAttachment color_attachments[MAX_COLOR_ATTACHMENTS];
    FBOAttachment depth;
    FBOAttachment stencil;
    // Default framebuffer parameters (for FBOs with no attachments)
    GLint default_width;
    GLint default_height;
    GLint default_layers;
    GLint default_samples;
    GLboolean default_fixed_sample_locations;
} Framebuffer;

typedef struct __GLsync {
    GLsizei name;
    void *mtl_event;
#ifdef __cplusplus
} Sync;
#else
} Sync, *__GLsync;
#endif

typedef struct PixelFormat_t {
    GLuint  format;
    GLuint  type;
    GLuint  mtl_pixel_format;
} PixelFormat;

typedef struct GLSLState_t {
    glslang_resource_t  resrc;
    glslang_limits_t    limits;
} GLSLState;

typedef struct PixelStore_t {
    GLboolean   swap_bytes;
    GLboolean   lsb_first;
    GLint row_length;
    GLint image_height;
    GLint skip_rows;
    GLint skip_pixels;
    GLint skip_images;
    GLint alignment;
} PixelStore;


enum {
    dirtyVAO = 0,
    dirtyState,
    dirtyBuffer,
    dirtyTexture,
    dirtyTexParam,
    dirtyTexBinding,
    dirtySampler,
    dirtyShader,
    dirtyProgram,
    dirtyFBO,
    dirtyDrawable,
    dirtyRenderState,
    dirtyAlphaState,
    dirtyImageUnit,
    dirtyBufferBase,
    maxDirtyState,
    dirtyAllBit = 31
};

#define DIRTY_VAO       (0x1 << dirtyVAO)
#define DIRTY_STATE     (0x1 << dirtyState)
#define DIRTY_BUFFER    (0x1 << dirtyBuffer)
#define DIRTY_TEX       (0x1 << dirtyTexture)
#define DIRTY_TEX_PARAM   (0x1 << dirtyTexParam)
#define DIRTY_TEX_BINDING (0x1 << dirtyTexBinding)
#define DIRTY_SAMPLER (0x1 << dirtySampler)
#define DIRTY_SHADER    (0x1 << dirtyShader)
#define DIRTY_PROGRAM   (0x1 << dirtyProgram)
#define DIRTY_FBO       (0x1 << dirtyFBO)
#define DIRTY_DRAWABLE      (0x1 << dirtyDrawable)
#define DIRTY_RENDER_STATE  (0x1 << dirtyRenderState)
#define DIRTY_ALPHA_STATE   (0x1 << dirtyAlphaState)
#define DIRTY_IMAGE_UNIT_STATE   (0x1 << dirtyImageUnit)
#define DIRTY_BUFFER_BASE_STATE   (0x1 << dirtyBufferBase)
#define DIRTY_ALL_BIT   ((unsigned)0x1 << dirtyAllBit)    // so we know the dirty all was set.
#define DIRTY_ALL       (0xFFFFFFFF)
/* r152: what a plain encoder / command-buffer break (flush, commit, CPU-visibility
 * sync, buffer copy, barrier, fence — MGLRenderer+Draw.m) must re-arm: all state
 * re-applies, but no shader and no drawable changed, so the two bits that make
 * the pipeline fast path (compatPipelineFastPath) decline stay out.  With
 * DIRTY_ALL there, the next draw after every break rebuilt the whole pipeline
 * descriptor (~170 us).  The cache key re-hashes the attachment formats for every
 * new encoder.  The tessellation / geometry emulation paths keep DIRTY_ALL: their
 * vertex-stage overrides (wide-line replay, pass-through) are not in the key and
 * rely on the descriptor path. */
#define DIRTY_REAPPLY   (DIRTY_ALL & ~(DIRTY_SHADER | DIRTY_DRAWABLE))

typedef struct {
    GLuint dirty_bits;

    // clear request clear_bitmask from glClear to Metal
    GLbitfield  clear_bitmask;

    // opengl state

    // keep these out of the var struct for debugging and access

    GLenum error;   // glGetError

    GLuint draw_buffer; // GL_DRAW_BUFFER
    GLuint read_buffer; // GL_READ_BUFFER
    GLuint max_color_attachments; // GL_MAX_COLOR_ATTACHMENTS
    GLuint max_vertex_attribs; // GL_MAX_VERTEX_ATTRIBS
    GLint viewport[4]; // GL_VIEWPORT
    GLfloat color_clear_value[4]; // GL_COLOR_CLEAR_VALUE

    Buffer *buffers[MAX_BINDABLE_BUFFERS];

    VertexArray *vao;
    Texture     *tex;
    Renderbuffer *renderbuffer;
    Framebuffer *framebuffer;
    Framebuffer *readbuffer;

    GLuint      active_texture; // GL_ACTIVE_TEXTURE
    unsigned    active_texture_mask[4];
    /* r159: uniform-buffer binding points MAX_BINDABLE_BUFFERS ..
     * MGL_MAX_UNIFORM_BUFFER_BINDINGS-1 (compat r111 advertises 512; rsvga's D3D9
     * fragment constants sit at 256).  They were an 8-entry table shared by every
     * context that dropped the lowest point when full, so glBindBuffersBase over
     * the whole range lost bindings (KHR-GL46.multi_bind functional_bind_buffers_*).
     * ubo_high_gen is bumped on every change: the draw-fast gate's indexed-binding
     * check covers only the buffer_base rows. */
    BufferBaseTarget ubo_high[MGL_MAX_UNIFORM_BUFFER_BINDINGS - MAX_BINDABLE_BUFFERS];
    GLuint      ubo_high_gen;
    /* r169: bumped wherever the indexed buffer tables (buffer_base) or the image
     * units change — exactly the sites that arm DIRTY_BUFFER_BASE_STATE /
     * DIRTY_IMAGE_UNIT_STATE.  The draw-fast snapshot compares these instead of
     * memcmp-ing ~11 KB / 5 KB per draw; unlike the dirty bits they are never
     * cleared by an unrelated path. */
    GLuint      buffer_base_gen;
    GLuint      image_units_gen;
    uint32_t    ubo_high_range[(MGL_MAX_UNIFORM_BUFFER_BINDINGS - MAX_BINDABLE_BUFFERS + 31) / 32];
    /* r158: bumped by every sampler / image uniform write (glUniform1i on an
     * opaque uniform re-points its slot at another unit).  The draw-fast gate
     * latches it: the per-unit texture checks cannot see a slot moving between
     * two unchanged units. */
    GLuint      sampler_map_gen;
    Texture     *active_textures[TEXTURE_UNITS];
    TextureUnit texture_units[TEXTURE_UNITS];
    Sampler     *texture_samplers[TEXTURE_UNITS];
    ImageUnit   image_units[TEXTURE_UNITS];

    GLsizei sync_name;

    HashTable vao_table;
    HashTable buffer_table;
    HashTable texture_table;
    HashTable shader_table;
    HashTable program_table;
    HashTable program_pipeline_table;
    HashTable transform_feedback_table;
    HashTable renderbuffer_table;
    HashTable framebuffer_table;
    HashTable sampler_table;

    Shader      *shaders[_MAX_SHADER_TYPES];
    Program     *program;
    ProgramPipeline *program_pipeline;
    TransformFeedback *transform_feedback;

    BufferBase  buffer_base[_MAX_BUFFER_TYPES];

    // glsl info
    GLSLState   glsl;

    // pixel pack unpack
    PixelStore  pack;
    PixelStore  unpack;
    
    // metal buffer mappings
    BufferMapList vertex_buffer_map_list;
    BufferMapList fragment_buffer_map_list;
    BufferMapList compute_buffer_map_list;

    // enable / disable caps
    GLMCaps     caps;

    // hints
    GLMHints    hints;
    
    // per-domain extension state (mgl_ext.h) — do not reorder
    struct MGLExtState_fb *ext_fb;
    struct MGLExtState_queries *ext_queries;
    struct MGLExtState_debug *ext_debug;
    struct MGLExtState_introspect *ext_introspect;
    struct MGLExtState_program *ext_program;
    struct MGLExtState_xfb *ext_xfb;
    struct MGLExtState_vattrib *ext_vattrib;
    struct MGLExtState_rstate *ext_rstate;
    struct MGLExtState_texparam *ext_texparam;
    struct MGLExtState_texdata *ext_texdata;
    struct MGLExtState_texext *ext_texext;
    struct MGLExtState_buffers *ext_buffers;
    struct MGLExtState_geom *ext_geom;
    struct MGLExtState_tess *ext_tess;
    struct MGLExtState_gpuaudit *ext_gpuaudit;
    struct MGLExtState_compat *ext_compat;

    // put at end, big chunk of yuck
    GLMParams   var;
} GLMState;

static_assert(TEXTURE_UNITS == 128, "active_texture_mask relies on this");

typedef struct GLMContextRec_t *GLMContext;

struct GLMMetalFuncs {
    void *mtlObj;
    void *mtlView;

    void (*mtlBindBuffer)(GLMContext glm_ctx, Buffer *ptr);
    void (*mtlBindTexture)(GLMContext glm_ctx, Texture *ptr);
    void (*mtlBindProgram)(GLMContext glm_ctx, Program *ptr);

    void (*mtlDeleteMTLObj)(GLMContext glm_ctx, void *obj);

    void (*mtlGetSync)(GLMContext glm_ctx, Sync *sync);
    void (*mtlWaitForSync)(GLMContext glm_ctx, Sync *sync);

    void (*mtlFlush)(GLMContext glm_ctx, bool finish);
    void (*mtlSwapBuffers)(GLMContext glm_ctx);
    
    void (*mtlClearBuffer)(GLMContext glm_ctx, GLuint type, GLbitfield mask);
    void (*mtlBlitFramebuffer)(GLMContext ctx, GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);


    void (*mtlBufferSubData)(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, const void *ptr);
    void *(*mtlMapUnmapBuffer)(GLMContext glm_ctx, Buffer *buf, size_t offset, size_t size, GLenum access, bool map);
    void (*mtlFlushBufferRange)(GLMContext glm_ctx, Buffer *buf, GLintptr offset, GLsizeiptr length);

    void (*mtlReadDrawable)(GLMContext glm_ctx, void *pixelBytes, GLuint bytesPerRow, GLuint bytesPerImage, GLint x, GLint y, GLsizei width, GLsizei height);

    void (*mtlGenerateMipmaps)(GLMContext glm_ctx, Texture *tex);
    void (*mtlTexSubImage)(GLMContext glm_ctx, Texture *tex, Buffer *buf, size_t src_offset, size_t src_pitch, size_t src_image_size, size_t src_size, GLuint slice, GLuint level, size_t width, size_t height, size_t depth, size_t xoffset, size_t yoffset, size_t zoffset);
    void (*mtlCopyTexSubImage)(GLMContext glm_ctx, Texture *tex, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width, GLsizei height);
    void (*mtlCopyImageSubData)(GLMContext glm_ctx, Texture *srcTex, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, Texture *dstTex, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);

    // draw arrays / elements
    void (*mtlDrawArrays)(GLMContext ctx, GLenum mode, GLint first, GLsizei count);
    void (*mtlDrawElements)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices);
    void (*mtlDrawRangeElements)(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices);

    // draw arrays / elements instanced
    void (*mtlDrawArraysInstanced)(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount);
    void (*mtlDrawElementsInstanced)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount);

    // draw arrays / elements base vertex
    void (*mtlDrawElementsBaseVertex)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLint basevertex);
    void (*mtlDrawRangeElementsBaseVertex)(GLMContext ctx, GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint basevertex);
    void (*mtlDrawElementsInstancedBaseVertex)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex);

    // draw arrays / elements intanced base vertex
    void (*mtlDrawArraysIndirect)(GLMContext ctx, GLenum mode, const void *indirect);
    void (*mtlDrawElementsIndirect)(GLMContext ctx, GLenum mode, GLenum type, const void *indirect);

    void (*mtlDrawArraysInstancedBaseInstance)(GLMContext ctx, GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance);
    void (*mtlDrawElementsInstancedBaseInstance)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLuint baseinstance);
    // ?? running out of names here.
    void (*mtlDrawElementsInstancedBaseVertexBaseInstance)(GLMContext ctx, GLenum mode, GLsizei count, GLenum type, const void *indices, GLsizei instancecount, GLint basevertex, GLuint baseinstance);

    // multi calls of many of the above
    void (*mtlMultiDrawArrays)(GLMContext ctx, GLenum mode, const GLint *first, const GLsizei *count, GLsizei drawcount);
    void (*mtlMultiDrawElements)(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount);
    void (*mtlMultiDrawElementsBaseVertex)(GLMContext ctx, GLenum mode, const GLsizei *count, GLenum type, const void *const*indices, GLsizei drawcount, const GLint *basevertex);

    void (*mtlMultiDrawArraysIndirect)(GLMContext ctx, GLenum mode, const void *indirect, GLsizei drawcount, GLsizei stride);
    void (*mtlMultiDrawElementsIndirect)(GLMContext ctx, GLenum mode, GLenum type, const void *indirect, GLsizei drawcount, GLsizei stride);


    void (*mtlDispatchCompute)(GLMContext ctx, GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z);
    void (*mtlDispatchComputeIndirect)(GLMContext ctx, GLintptr indirect);

    // per-domain Metal hook tables (mgl_ext.h) — do not reorder
    struct MGLExtMetal_fb *ext_fb;
    struct MGLExtMetal_queries *ext_queries;
    struct MGLExtMetal_debug *ext_debug;
    struct MGLExtMetal_introspect *ext_introspect;
    struct MGLExtMetal_program *ext_program;
    struct MGLExtMetal_xfb *ext_xfb;
    struct MGLExtMetal_vattrib *ext_vattrib;
    struct MGLExtMetal_rstate *ext_rstate;
    struct MGLExtMetal_texparam *ext_texparam;
    struct MGLExtMetal_texdata *ext_texdata;
    struct MGLExtMetal_texext *ext_texext;
    struct MGLExtMetal_buffers *ext_buffers;
    struct MGLExtMetal_geom *ext_geom;
    struct MGLExtMetal_tess *ext_tess;
    struct MGLExtMetal_gpuaudit *ext_gpuaudit;
    struct MGLExtMetal_compat *ext_compat;
} ;

typedef struct GLMContextRec_t {
    GLuint      context_flags;

    struct GLMDispatchTable dispatch;
    struct GLMMetalFuncs mtl_funcs;

    GLMState    state;
    GLboolean   assert_on_error;

    PixelFormat pixel_format;
    PixelFormat depth_format;
    PixelFormat stencil_format;

    BufferData  *temp_element_buffer;

    void (* error_func)(GLMContext ctx, const char *func, GLenum type);
} GLMContextRec;


GLMContext createGLMContext(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type);
GLMContext createGLMContextProfile(GLenum format, GLenum type,
                            GLenum depth_format, GLenum depth_type,
                            GLenum stencil_format, GLenum stencil_type,
                            GLenum profile);
void mglSetNextContextProfile(GLenum profile);
GLenum mglNextContextProfile(void);

void mgl_lazy_init(void);

void MGLsetCurrentContext(GLMContext ctx);

enum {
    MGL_PIXEL_FORMAT,
    MGL_PIXEL_TYPE,
    MGL_DEPTH_FORMAT,
    MGL_DEPTH_TYPE,
    MGL_STENCIL_FORMAT,
    MGL_STENCIL_TYPE,
    MGL_CONTEXT_FLAGS
};

#ifdef __cplusplus
extern "C" {
#endif

GLuint sizeForFormatType(GLenum format, GLenum type);
GLMContext MGLgetCurrentContext(void);
void MGLget(GLMContext ctx, GLenum param, GLuint *data);
bool pixelConvertToInternalFormat(GLMContext ctx, GLenum internalformat, GLenum format, GLenum type, const void *src, void *dst, size_t len);

bool createTextureLevel(GLMContext ctx, Texture *tex, GLuint face, GLint level, GLboolean is_array, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, void *pixels, GLboolean proxy);


#ifdef __cplusplus
};
#endif

#endif /* glm_context_h */
