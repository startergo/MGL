/* mgl_ext_rstate.h — private state and Metal hooks of the 'rstate' domain. Owned by agent 'rstate'.
 *
 * Orientation convention (agreed project-wide): Metal texture memory row r == GL window row y=r
 * for every render target.  rstate realises it by a vertex-shader clip fixup injected into the
 * MSL at link time (gl_Position.y = -y for CLIP_ORIGIN LOWER_LEFT, z = (z+w)/2 for
 * CLIP_DEPTH_MODE NEGATIVE_ONE_TO_ONE), selected per draw through Metal function constants.
 * Consequently viewport / scissor y are passed to Metal unchanged, readback and uploads need no
 * row flip, and the front-face winding handed to Metal is inverted while the y-flip is active.
 */
#ifndef MGL_EXT_RSTATE_H
#define MGL_EXT_RSTATE_H

#include "glm_context.h"
#include "mgl_ext.h"

#define MGL_RSTATE_MAX_VIEWPORTS    16
#define MGL_RSTATE_MAX_SAMPLE_WORDS 1
#define MGL_RSTATE_VIEWPORT_BOUNDS  32768.0f

/* Metal function constant ids used by the injected shader fixups */
#define MGL_RS_FC_DEPTH_ZERO_TO_ONE   32000
#define MGL_RS_FC_ORIGIN_UPPER_LEFT   32001
#define MGL_RS_FC_PROGRAM_POINT_SIZE  32002
#define MGL_RS_FC_POINT_SIZE          32003
#define MGL_RS_FC_POINT_ORIGIN_LOWER  32004
#define MGL_RS_FC_SAMPLE_MASK         32005
#define MGL_RS_FC_PER_SAMPLE           32006
#define MGL_RS_FC_RASTERIZER_DISCARD   32007
#define MGL_RS_FC_PIPELINE_SAMPLES     32008   /* the render pipeline's sample count; a
                                                * single-sample pipeline IGNORES a fragment
                                                * shader's gl_SampleMask write (GL 4.6 13.3,
                                                * KHR-GL46.sample_variables mask.*) */

/* GL-side state, reachable as ctx->state.ext_rstate; allocated in mgl_ext_init_rstate(). */
struct MGLExtState_rstate {
    /* indexed viewport / depth range / scissor (index 0 mirrors state.viewport etc.) */
    GLfloat   viewport[MGL_RSTATE_MAX_VIEWPORTS][4];
    GLdouble  depth_range[MGL_RSTATE_MAX_VIEWPORTS][2];
    GLint     scissor[MGL_RSTATE_MAX_VIEWPORTS][4];
    GLboolean scissor_test[MGL_RSTATE_MAX_VIEWPORTS];
    GLboolean viewport_set_by_app;      /* GL: initial viewport/scissor = drawable size */

    /* per draw buffer blend enable + color mask */
    GLboolean blend[MAX_COLOR_ATTACHMENTS];
    GLboolean color_mask[MAX_COLOR_ATTACHMENTS][4];

    GLfloat   polygon_offset_factor;
    GLfloat   polygon_offset_units;
    GLfloat   polygon_offset_clamp;

    GLenum    clip_origin;              /* GL_LOWER_LEFT / GL_UPPER_LEFT */
    GLenum    clip_depth_mode;          /* GL_NEGATIVE_ONE_TO_ONE / GL_ZERO_TO_ONE */
    GLenum    provoking_vertex;
    GLenum    polygon_mode;

    GLint     patch_vertices;
    GLfloat   patch_default_outer[4];
    GLfloat   patch_default_inner[2];

    GLuint    primitive_restart_index;

    GLfloat   sample_coverage_value;
    GLboolean sample_coverage_invert;
    GLbitfield sample_mask[MGL_RSTATE_MAX_SAMPLE_WORDS];
    GLfloat   min_sample_shading;

    GLenum    clamp_read_color;         /* GL_FIXED_ONLY / GL_TRUE / GL_FALSE */

    GLfloat   point_fade_threshold;
    GLenum    point_sprite_coord_origin;

    GLuint    generation;               /* bumped on every encoder-state change */
    GLuint    generation_pipeline;      /* bumped only on pipeline-baked state (blend, masks,
                                           function constants) — not viewport/scissor/raster, which
                                           never change MTLRenderPipelineState */

    /* compat: GL_CLAMP_TO_BORDER border-colour emulation (compat_border.c).
     * The per-draw signature of the emulated-slot table (0 = none) rides the
     * pipeline fast-path key, and the rewritten MSL variant is cached against
     * the source MSL pointer it was built from. */
    uint64_t  border_sig;
    const void *border_msl_src;
    char      *border_msl;
    uint64_t  border_msl_sig;

    /* geom r87: "this program's fragment stage declares a flat interpolant",
     * refreshed at every successful link by the post-link hook
     * (mgl_rstate_post_link) and read per draw by rs_program_has_flat().
     * Name-keyed, direct-mapped: a program cannot draw before its first link
     * refreshes the entry, so a recycled name never serves another program's
     * answer — the previous pointer-keyed memo did exactly that and flipped
     * dEQP separate_shader.random.* between identical runs (16 stale hits per
     * group run: a freed msl_str address recycled by a different program whose
     * SPIRV-Cross boilerplate header matched the fingerprint). */
    struct { GLuint name; uint8_t flat; } fs_flat[64];
};

/* one MSL texture slot needing GL_CLAMP_TO_BORDER colour emulation */
#define MGL_BORDER_MAX 16
typedef struct MGLBorderEmu {
    GLuint  msl_slot;    /* Metal [[texture(N)]] argument index */
    uint8_t axes;        /* bit0 = S, bit1 = T, bit2 = R clamp-to-border */
    uint8_t linear;      /* within-level LINEAR weights (else NEAREST) */
    uint8_t kind;        /* 0 = float4, 1 = half4, 2 = int4, 3 = uint4 colour */
    char    bc[128];     /* MSL literal of the exact border colour */
    char    bh[128];     /* MSL literal of the colour the Metal sampler returns */
} MGLBorderEmu;

/* Metal hooks, reachable as ctx->mtl_funcs.ext_rstate; filled by MGLRenderer (RenderState). */
struct MGLExtMetal_rstate {
    /* recompile the MTLLibrary/MTLFunction of the given stages from program->spirv[].msl_str */
    void (*recompile_program)(GLMContext ctx, struct Program_t *program, unsigned stage_mask);
};

#ifdef MGL_RSTATE_INTERNAL
/* internal to the rstate files (state.c, state_ext.c, MGLRenderer+RenderState.m) */
#define RS()                 (ctx->state.ext_rstate)
#define RS_ERROR(e)          do { ERROR_RETURN(e); return; } while (0)
#define RS_CHECK(cond, e)    do { if (!(cond)) { ERROR_RETURN(e); return; } } while (0)
/* encoder-only state (viewport, scissor, depth/stencil, raster) */
static inline void mgl_rs_dirty_encoder(GLMContext ctx)
{
    ctx->state.dirty_bits |= DIRTY_RENDER_STATE;
    if (ctx->state.ext_rstate) ctx->state.ext_rstate->generation++;
}
/* state baked into the MTLRenderPipelineState (blend, masks, shader variant constants) */
static inline void mgl_rs_dirty_pipeline(GLMContext ctx)
{
    ctx->state.dirty_bits |= DIRTY_ALPHA_STATE | DIRTY_RENDER_STATE;
    if (ctx->state.ext_rstate)
    {
        ctx->state.ext_rstate->generation++;
        ctx->state.ext_rstate->generation_pipeline++;
    }
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* arrays draws that need an index list to emulate GL (provoking vertex); see below */
bool mgl_rstate_arrays_rewrite_needed(GLMContext ctx, GLenum mode);

/* ---- helpers exported for other domains ---------------------------------------------------- */

/* GL_CLAMP_TO_BORDER border-colour emulation (compat_border.c, fragment stage) */
bool mgl_border_wanted(GLMContext ctx);                      /* cheap per-draw gate */
int  mgl_border_scan(GLMContext ctx, int stage, MGLBorderEmu out[MGL_BORDER_MAX]);
uint64_t mgl_border_sig(const MGLBorderEmu *e, int n);
char *mgl_border_rewrite(const char *msl, const MGLBorderEmu *emu, int n);   /* malloc'd MSL or NULL */

/* GL_CLAMP_READ_COLOR for glReadPixels (owner of ReadPixels: 'fb').
 * Returns true if colors read back must be clamped to [0,1].  read_fixed_point is true when the
 * read framebuffer's color buffer is fixed-point (normalized), which matters for GL_FIXED_ONLY. */
bool mgl_rstate_clamp_read_color(GLMContext ctx, bool read_fixed_point);

/* Primitive restart emulation for the draw path (owner: 'buffers').
 * Metal restarts only strip topologies, only at 0xFFFF / 0xFFFFFFFF, always.  GL restarts any
 * topology at PRIMITIVE_RESTART_INDEX (or the type's max value with PRIMITIVE_RESTART_FIXED_INDEX)
 * only when enabled.  When this returns true, call mgl_rstate_rewrite_indices() and draw the
 * returned indices instead. */
bool mgl_rstate_index_rewrite_needed(GLMContext ctx, GLenum mode, GLenum type,
                                     const void *indices, GLsizei count);

/* Rewrites count indices of GL type (UNSIGNED_BYTE/SHORT/INT) into a malloc'ed buffer that Metal
 * draws with the same result.  Outputs the Metal-drawable mode (fans/loops become lists, lists keep
 * their mode with restart-split primitives removed), the index type (GL_UNSIGNED_SHORT or
 * GL_UNSIGNED_INT) and the index count.  Returns NULL on allocation failure.  Caller frees. */
void *mgl_rstate_rewrite_indices(GLMContext ctx, GLenum mode, GLenum type, const void *indices,
                                 GLsizei count, GLenum *out_mode, GLenum *out_type, GLsizei *out_count);

/* Inject the rstate clip-space / point-size fixup into vertex MSL, and the point-coord fixup into
 * fragment MSL.  Idempotent (marker "MGL_RSTATE_FIXUP").  Returns a malloc'ed string or NULL if
 * nothing had to change.  The 'xfb' capture variant must not call this (GL captures unmodified
 * gl_Position). */
char *mgl_rstate_patch_vertex_msl(const char *msl, const char *entry_point);
char *mgl_rstate_patch_fragment_msl(const char *msl, const char *entry_point);
/* per-sample-shading MSL variant (sample-qualified interpolants); NULL = nothing to patch */
char *mgl_rstate_pss_fragment_msl(const char *msl, const char *entry_point);

/* For glDrawArrays-style draws when mgl_rstate_arrays_rewrite_needed(): returns a malloc'ed
 * index list (GL_UNSIGNED_INT) for vertices first..first+count-1 in the order Metal must draw them. */
void *mgl_rstate_generate_array_indices(GLMContext ctx, GLenum mode, GLint first, GLsizei count,
                                        GLenum *out_mode, GLsizei *out_count);

/* true if program stage msl was patched by rstate */
bool mgl_rstate_msl_is_patched(const char *msl);

#ifdef __cplusplus
}
#endif

#endif
