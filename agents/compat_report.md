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

## Known core bugs (worked around, owned by geom worker — do NOT fix here)

1. Default-block `float` uniforms next to `vec4`s get clobbered by UBO-style packing.
   Workaround in FFP shaders: keep scalar uniforms in separate decl groups; verify any new scalar
   uniform actually round-trips (use EXPECT on a query or a pixel test).
2. `glProgramUniform1f` is a no-op — use `glUniform1f` on the bound program.

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

All three suites green: pipeline 60, profile 61, render2 65.

## Fixed this session (committed per area)

* compat_ffp.c: color-material mode redeclared `vec4 mglMatA0...` twice → GLSL compile error →
  all *material* (non-color-material) lighting drew black. Now declared once.
* compat_ffp.c: FFP key now maps enums to small ids so distinct enum values can't collide in the
  key bytes (fog mode, env mode, combine modes, texgen modes, texture targets).
* compat_pixel.c: DrawPixels/Bitmap/CopyPixels blit path rewritten to draw through the FFP itself
  (glBegin quad + MODULATE + alpha test) instead of a separate blit program; saves/restores FF
  state. Fix: `mglActiveTexture(ctx, GL_TEXTURE0 + saved_active)` (state stores a unit *index*).
* compat.c: pixel transfer defaults — color scales default to 1.0, not 0 (DrawPixels was black).


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
* Known gap: a vertical/horizontal line exactly on a pixel boundary rasterizes to the left/bottom
  pixel on Metal (D3D rule) instead of the GL diamond-rule pixel; the evaluator test accepts both.

## Remaining work

* More GLSL-compat coverage (#version 120 + ftransform test exists in pipeline; test edge cases:
  gl_FragData, gl_MultiTexCoord, gl_ModelViewMatrix uniforms).
* glTexGen SPHERE_MAP/REFLECTION_MAP/NORMAL_MAP pixel tests; point sprites/point size attenuation.
* CopyPixels test; glPixelMap effect on DrawPixels/readback; color-index behaviour.
* Line stipple pixel test (fragment shader pattern) - implemented, untested.
