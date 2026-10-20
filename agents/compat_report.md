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
| 4 | fixed-function fragment (texenv+combine, fog, alpha test, stipple, point sprites) | texenv/fog/alpha done; stipple/point-sprite = in progress (render2) |
| 5 | display lists | done (797057e), replay test passes |
| 6 | attrib stacks, raster ops (DrawPixels/Bitmap/CopyPixels/RasterPos/WindowPos, pixel maps, accum) | DrawPixels now pixel-exact; Bitmap/accum = in progress |
| 7 | legacy texture formats/state, glGet* of compat state | mostly done (profile suite) |
| 8 | select/feedback, evaluators, edge flags | implemented, render2 tests failing (below) |

## Pixel-tested (test_compat_pipeline, 60 checks green)

immediate mode quads/rect, current color, glShadeModel flat/smooth (front/back), textures
(REPLACE/MODULATE + filtering), texgen OBJECT_LINEAR/EYE_LINEAR, clip planes, lighting
directional (ambient/diffuse/emission paths), fog linear/exp/exp2, alpha test funcs, display-list
replay, push/pop attrib, client arrays + VAO 0, glInterleavedArrays, glArrayElement.

## Current failures (test_compat_render2) — work order

1. `positional_light_and_colormaterial` — attenuation math (pixel 68 vs 255) + NdotL range.
2. `texenv_modes` — DECAL/BLEND/ADD alpha/green wrong (0,128,0 vs 0,255,0).
3. `bitmap` — glBitmap offset/origin wrong (pattern shifted, colors wrong).
4. `polygon_stipple` — ~28-36 pixels expected, count out of range.
5. `evaluators` — glMap1/2 + glEvalMesh produce nothing.
6. `select_mode` — hit records wrong (first name 0, extra hit).
7. `feedback_mode` — no feedback tokens written.

## Fixed this session (uncommitted until noted)

* compat_ffp.c: color-material mode redeclared `vec4 mglMatA0...` twice → GLSL compile error →
  all *material* (non-color-material) lighting drew black. Now declared once.
* compat_ffp.c: FFP key now maps enums to small ids so distinct enum values can't collide in the
  key bytes (fog mode, env mode, combine modes, texgen modes, texture targets).
* compat_pixel.c: DrawPixels/Bitmap/CopyPixels blit path rewritten to draw through the FFP itself
  (glBegin quad + MODULATE + alpha test) instead of a separate blit program; saves/restores FF
  state. Fix: `mglActiveTexture(ctx, GL_TEXTURE0 + saved_active)` (state stores a unit *index*).
* compat.c: pixel transfer defaults — color scales default to 1.0, not 0 (DrawPixels was black).
