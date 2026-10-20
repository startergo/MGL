# compat domain — progress log

Session log for the GL 4.6 compatibility-profile work (task: agents/tasks/compat.md).
Update this file at every commit so a fresh session can continue.

## How to build/test fast

* `./build_r.sh` — rebuilds the lib forcing every compat TU to recompile (dylib caches).
* `tests/headless/run.sh test_compat_pipeline test_compat_profile test_compat_render2` — the three
  compat suites (pipeline = pixel tests of the FFP; profile = state/queries/errors; render2 = the
  remaining rendering features).
* `MGLC_DUMP=1 <test>` dumps the generated FFP shaders to /tmp/mglc_vs.glsl / /tmp/mglc_fs.glsl.
* Probes: tests/headless/test_compat_*probe*.c are scratch, delete before commit.

## Architecture recap (see commits b1c6e60, 797057e)

* Context: `mglCompatCreateContext()` (tests) → compat profile flag in `MGLExtState_compat`.
  Core contexts keep rejecting compat calls (`MGLC_COMPAT_GUARD`).
* All compat entry points live in MGL/src/compat_*.c (dispatch in glm_dispatch.c is shared but the
  mgl* bodies are compat-only files).
* Fixed function pipeline: `compat_ffp.c` generates GLSL VS/FS from an `MGLFFPKey` (cached per key
  in `s->ffp_cache`), binds it at draw time (`bind_ffp`), syncs uniforms (`mgl_compat_ffp_sync`)
  guarded by `s->state_serial` (bumped by `mgl_compat_dirty`).
* Immediate mode: compat_vb.c `MGLCImm` batches into stream buffers on an internal VAO
  (s->imm_vao), GL_QUADS/QUAD_STRIP/POLYGON converted to triangles (`convert_prim`).
* Client arrays + VAO 0: compat_vb.c uploads ranges at draw.
* Display lists: compat_list.c records ops into a list; replay dispatches them.
* Raster ops: compat_pixel.c — RasterPos/WindowPos (CPU transform), DrawPixels/Bitmap/CopyPixels
  via an internal texture + a quad drawn through the FFP itself (`blit_quad`), accum buffer as a
  float attachment (see `#pragma mark accumulation buffer`).
* Compat GLSL (#version 110..150 / compatibility): compat_glsl.c rewrites to core GLSL.

## Former "core bugs" — resolved (they were never core bugs)

The two workarounds from earlier sessions were removed after the geom worker proved core uniform
packing + glProgramUniform* correct (tests/headless/test_uniforms.c, 297 checks):
* FFP scalar uniforms are now plain scalars again: mglNormalScale (float), mglAlphaRef (float),
  mglEnvScaleN (vec2), mglPointSizeParams/mglPointAtten (vec3). No vec4 packing.
* The "re-sync the first 3 draws because the uniform table is re-derived after the MSL compile"
  hack in mgl_compat_ffp_sync is gone; one sync per state_serial is enough.
Regression tests: test_compat_glsl.c glsl_scalar_uniform_sandwich / glsl_program_uniform_compat
(readback + pixels on a compat context), test_compat_render2.c specular_shininess (the float
uniform mglMatShininess[0] verified pixel-exact against pow(NdH, exp) for exp 128/16/2).

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| 1 | profile plumbing, compat GLSL | done (b1c6e60) |
| 2 | immediate mode, client arrays, quads/polygons, glRect | done (797057e) |
| 3 | fixed-function vertex (matrices, lighting, texgen, clip, fog coord, point size) | done (797057e), light probe tests added |
| 4 | fixed-function fragment (texenv+combine, fog, alpha test, stipple, point sprites) | done incl. polygon stipple texture |
| 5 | display lists | done (797057e), replay test passes |
| 6 | attrib stacks, raster ops (DrawPixels/Bitmap/CopyPixels/RasterPos/WindowPos, pixel maps, accum) | done, pixel-tested |
| 7 | legacy texture formats/state, glGet* of compat state | mostly done (profile suite) |
| 8 | select/feedback, evaluators, edge flags | done, tested (select per-primitive hits, feedback tokens) |

## Pixel-tested (test_compat_pipeline, 60 checks green)

immediate mode quads/rect, current color, glShadeModel flat/smooth (front/back), textures
(REPLACE/MODULATE + filtering), texgen OBJECT_LINEAR/EYE_LINEAR, clip planes, lighting
directional (ambient/diffuse/emission paths), fog linear/exp/exp2, alpha test funcs, display-list
replay, push/pop attrib, client arrays + VAO 0, glInterleavedArrays, glArrayElement.

## Pixel-tested now (test_compat_render2, 65 checks green)

matrix transforms (rotate/scale/push-pop), positional light + color material (per-vertex CPU ref),
texenv REPLACE/DECAL/BLEND/ADD/COMBINE (modulate/replace), RasterPos/DrawPixels + zoom,
glBitmap pattern (MSB-first, no flip, alpha-masked), accumulation buffer load/accum/return,
polygon stipple 32x32 (GPU texture sample), evaluators map1/map2 + EvalMesh1/2,
select mode (per-primitive hit records, viewport-exact culling), feedback mode
(GL_2D/3D/3D_COLOR/3D_COLOR_TEXTURE/4D_COLOR_TEXTURE token sizes).

All seven suites green: pipeline 60, profile 61, render2 65, glsl 18, render3 16, render4 35,
render5 17. Full headless suite: 4560 checks, 0 failures. stub_scanner clean for compat files.

## Fixed this session (committed per area)

* compat_ffp.c: color-material mode redeclared `vec4 mglMatA0...` twice → GLSL compile error →
  all *material* (non-color-material) lighting drew black. Now declared once.
* compat_ffp.c: FFP key now maps enums to small ids so distinct enum values can't collide in the
  key bytes (fog mode, env mode, combine modes, texgen modes, texture targets).
* compat_pixel.c: DrawPixels/Bitmap/CopyPixels blit path rewritten to draw through the FFP itself
  (glBegin quad + MODULATE + alpha test) instead of a separate blit program; saves/restores FF
  state. Fix: `mglActiveTexture(ctx, GL_TEXTURE0 + saved_active)` (state stores a unit *index*).
* compat.c: pixel transfer defaults — color scales default to 1.0, not 0 (DrawPixels was black).


## Session 8 (commit "default matrices, diamond-rule line rasterization")

* REAL BUG FIXED: the float matrix caches the generated FFP/compat-GLSL shaders read
  (s->modelview_f / projection_f / texture_f) were only refreshed by matrix_changed(),
  so a compat context that never calls a matrix entry point (glLoadIdentity etc.) drew
  with all-zero matrices — nothing appeared. mgl_compat_init_state now initialises the
  float caches to identity.
* Line rasterization "one pixel left/down" was NOT a Metal-vs-GL gap: a probe drawing
  the same segments on a CORE context (VBO + glDrawArrays) and on the compat context
  (glBegin) produces byte-identical coverage. A vertical line at integer x=16 lights
  column 15 (and not 16) because x=16 is the RIGHT corner of column 15's diamonds:
  GL's diamond-exit rule counts right/top corners and not left/bottom. Same for
  horizontal lines (row 15 for y=16). Test test_compat_render2.c line_diamond_rule pins
  this (endpoints, half-pixel cases), and the evaluators test now asserts column 15
  exactly instead of accepting both.
* Probe note: glReadPixels of a 1-pixel-wide GL_RGB row writes PACK_ALIGNMENT-padded
  rows (4 bytes per 3-byte row) — read GL_RGBA in probes.

## Session 7 (commit "GL_CLAMP / GENERATE_MIPMAP / texenv LOD bias")

* glTexParameteri(GL_TEXTURE_WRAP_*, GL_CLAMP) is accepted: the hardware gets CLAMP_TO_EDGE
  (Apple GPUs have no border fetch - documented approximation) and the requested GL_CLAMP is
  reported back by glGetTexParameter* (compat wrappers compat_tex_parameter*, compat_get_tex_parameter*).
* GL_TEXTURE_BORDER_COLOR round-trips through the core parameter storage (queries return it);
  sampling ignores it (gap).
* GL_GENERATE_MIPMAP is tracked per texture in the compat texture table; compat_tex_image1D/2D/3D
  call glGenerateMipmap after a level-0 upload when it is on (levels verified 32->4->1).
* glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_LOD_BIAS, x) is stored per unit, enters the FFP key and is
  passed to texture() as the bias argument (uniform mglLodBias, sync from the active unit).
* glPushAttrib(GL_TEXTURE_BIT)/PopAttrib no longer errors (it was re-applying the invalid params).

## Session 6 (commit "point sprites/attenuation, legacy texture formats, residency")

* Point attenuation now reaches Metal: mgl_compat_point_parameterfv enables/disables
  GL_PROGRAM_POINT_SIZE (the Metal pipeline flag that honours a shader point size) depending on
  whether the attenuation is the identity. Sizes verified pixel-exact: 16*sqrt(1/d) for d=4/9/16.
* Point sprites verified: gl_PointCoord (0,0) maps to the top-left texel of the point (spec).
* Legacy internal formats (GL_LUMINANCE/_ALPHA/ALPHA/INTENSITY + the *8/*16 spellings) are expanded
  to RGBA on upload by compat dispatch wrappers (compat_tex_image1D/2D/3D), stored in
  s->legacy_tex[], re-packed by compat_get_tex_image, and reported as LUMINANCE8/... by
  compat_get_tex_level_parameteriv. FLOAT/half/16-bit inputs expand through float; storage is
  RGBA32F.
* glAreTexturesResident raises GL_INVALID_VALUE for name 0 (spec) and still reports resident.

## Session 5 (commit "line stipple, texgen reflection fix, pixel maps")

* Line stipple is now really implemented: a per-vertex "distance along the line" attribute
  (MGLC_ATTR_LINE_I = location 1) is computed on the CPU in window space and streamed for
  immediate mode and glDrawArrays; the FFP vertex shader passes it to the fragment shader, which
  masks with pattern/factor uniforms. (Indexed lines with stipple still stream no distances - gap.)
* REFLECTION_MAP texgen referenced the reflection vector without declaring it (SPHERE_MAP only)
  -> shader compile error -> crash when drawing with no program bound. Both maps declare mglR now,
  and mgl_compat_ffp_program falls back to a minimal transform+color program when generation
  fails, so a bad state combination can never crash the draw.
* DrawPixels applies GL_MAP_COLOR via the R/G/B/A pixel maps (v -> map[round(v*(n-1))]); the
  default single-entry maps are identity (1.0).
* fixed-function texgen/lighting are per-vertex (as GL 1.x says), tests compute the
  interpolated CPU reference. (The "one pixel left/down" line note here was wrong — see
  session 8: that IS the GL diamond rule.)

## Session 4 (commit "compat GLSL fixes")

* gl_FogFragCoord was only substituted in fragment shaders - now also in vertex shaders (declared
  as `out float mglFogFragCoord`, already declared in the preamble).
* ftransform() wrapper referenced mglModelViewProjectionMatrix even when only U_MVM was used -
  the uniform is now declared in that case too.
* mgl_compat_sync_builtin_uniforms uploaded `spotDirection` (a vec3) with glUniform4fv ->
  GL_INVALID_OPERATION at every draw of a shader using gl_LightSource; now matches the type.
* glGetUniformLocation in compat contexts accepts built-in names (gl_ModelViewMatrix ->
  mglModelViewMatrix, gl_LightModel.ambient -> mglLightModelAmbient) via a compat dispatch
  wrapper (`mgl_compat_get_uniform_location`); unused builtins correctly return -1.
* gl_LightModel is now a proper struct in translated shaders (mgl_LightModelParameters) and is
  synced (`mglLightModel.ambient`).

## Session 3 (commit "render2 fixes")

* glBitmap: bits are MSB-first and row 0 is the bottom (no vertical flip); the pattern now lives in
  a dedicated `s->bitmap_texture` (GL_R8 swizzled A=RED) so MODULATE keeps the raster color and the
  alpha test discards 0-bits.
* Polygon stipple: the 32x32 pattern is uploaded (FNV-1a guarded) to `s->stipple_tex` on unit
  MGLC_STIPPLE_UNIT by `sync_stipple_texture()` at FFP sync time; `glPolygonStipple` stores the raw
  GL layout and `glGetPolygonStipple` returns it unchanged.
* Evaluator maps: GL_MAP1_x/GL_MAP2_x are now real caps (enable/disable/IsEnabled); de Casteljau
  used to memcpy GLfloat control points into a double array (garbage) and assumed a [v][u] layout -
  fixed to convert element-wise with the documented [u][v] layout; strides are bytes (0 = tight).
* Select mode: clip tests use the actual viewport; hits are recorded per GL primitive (QUADS = one
  record), coverage is an exact triangle/segment-vs-rect test (vertex/corner/edge cases); the
  name-count patch wrote to slot hit_zmin_slot-2 (out of bounds) and the close path never patched
  the header - both fixed.
* Feedback mode: rewritten (`fb_emit_prims`) with per-type vertex layouts (GL_2D/3D/3D_COLOR/
  3D_COLOR_TEXTURE/4D_COLOR_TEXTURE), per-primitive tokens; TRIANGLE_STRIP/FAN/TRIANGLES now emit
  (the old code relied on `mgl_compat_build_tri_indices` which only converts QUADS/POLYGON).
* (removed: the line-on-boundary note above was a misdiagnosis, see session 8)

## Remaining work

* More GLSL-compat coverage: test_compat_glsl.c now covers #version 110/120/150-compat,
  ftransform, gl_FragData, gl_TextureMatrix, gl_LightSource/gl_FrontMaterial/gl_Fog uniforms,
  builtin uniform locations.
* GL_CLAMP border sampling: CLAMP maps to CLAMP_TO_EDGE (Apple GPUs lack border fetch); the border
  color is stored/queried but not sampled.
* GENERATE_MIPMAP on glTexImage (compat flag); color-index (glIndex) beyond state tracking.
* Line stipple pixel test (fragment shader pattern) - implemented, untested.
* Indexed glDrawElements + line stipple (no per-vertex distances yet).
