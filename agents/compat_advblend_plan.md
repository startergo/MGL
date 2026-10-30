# Plan: KHR_blend_equation_advanced (GL 4.6 core §17.3.8) — r28 sweep, 88 NotSupported cases

Supervisor task (Oct 29): implement `KHR-GL46.blend_equation_advanced.*`
(88 cases, currently NotSupported because the extension is not advertised).

## What it is
15 blend equations (MULTIPLY/SCREEN/OVERLAY/DARKEN/LIGHTEN/COLORDODGE/COLORBURN/
HARDLIGHT/SOFTLIGHT/DIFFERENCE/EXCLUSION/HSL_HUE/HSL_SATURATION/HSL_COLOR/
HSL_LUMINOSITY, all `GL_*_KHR`), promoted into GL 4.6 core from
GL_KHR_blend_equation_advanced. GLSL gains `layout(blend_support_*) out;`
qualifiers and the `GL_KHR_blend_equation_advanced` macro; `glBlendEquation`
(and `glBlendEquationi`) accept the new enums while the `Separate` variants
reject them; draws are INVALID_OPERATION when a non-NONE draw buffer other
than 0 is active or the FS lacks the matching qualifier.

## Approach (same idea as Mesa gl_nir_lower_blend_equation_advanced.c and ANGLE)
Metal has no fixed-function equivalent, but Apple GPUs support programmable
blending through framebuffer fetch: a fragment entry-point parameter
`float4 dst [[color(0)]]` reads the current attachment-0 value from tile
memory (raster-order, coherent — this is why the `_coherent` extension holds).

1. GL state (`compat_advblend.c`, compat domain):
   * `mgl_advblend_mode(enum)` → 0..14; `mgl_advblend_active(ctx)` = blend
     enabled on draw buffer 0 + both equations of buffer 0 are the same
     advanced enum + an FS stage is bound.
   * `glBlendEquation`/`glBlendEquationi` accept the advanced enums (edit in
     state.c, 2 function bodies); Separate variants keep rejecting them.
   * `GL_BLEND_ADVANCED_COHERENT_KHR` enable/disable/isEnabled/get, default
     TRUE (query-only: Metal tile reads are coherent, no barrier needed).
   * Extension strings in get.c: GL_KHR_blend_equation_advanced +
     GL_KHR_blend_equation_advanced_coherent.
2. Qualifier mask: glslang validates `blend_support_*` but emits nothing in
   SPIR-V, so the mask is scanned from the FS source at link time
   (post-link hook) into `Program.spirv[FS].adv_blend` (one new uint32_t).
3. Draw-time validation (compat pre-draw hook, registered for every context):
   advanced active + >1 non-NONE draw buffer or missing qualifier bit →
   GL_INVALID_OPERATION, draw skipped.
4. Shader (compat_advblend.c `mgl_advblend_rewrite`): textual MSL rewrite of
   the fragment stage — append a `float4 mglAdvDst [[color(0)]]` argument to
   the entry signature, and rewrite every `return out;` to blend
   `out.<color(0) member>` first. All 15 formulas transliterated from Mesa's
   gl_nir_lower_blend_equation_advanced.c (spec / dEQP equations: unpremultiply
   src+dst, f(Cs,Cd), then RGB = f*p0 + Cs*p1 + Cd*p2, A = p0+p1+p2).
   Per-(fragment MSL, equation) variant, compiled through the existing
   MSL→library cache (extProgramLibraryForMSL) and specialized by the existing
   rstate machinery (function constants unchanged).
5. Pipeline (2 surgical edits in MGLRenderer+RenderState.m, rstate domain):
   * rstateConfigurePipelineDescriptor: disable Metal fixed-function blending
     for attachment 0 when `mgl_advblend_active()` (the shader composites).
     The rstate content hash already mixes the equation, so the compat PSO
     fast path and the gpuaudit PSO cache distinguish the variants for free.
   * rstateSpecialize: when advanced blending is active, swap in the patched
     fragment function (mirrors the existing CLAMP_TO_BORDER variant swap).
   * MSAA render passes force per-sample shading (fetch must read per-sample).

## Risks / costs
* Cost when NOT used: zero Metal work. The pre-draw hook is one table lookup
  (enum → mode) and the pipeline paths are one bool; the rewrite only runs
  on pipeline builds while an advanced equation is active.
* One extra MTLLibrary compile per (program FS, equation) pair, only when
  that combination is first drawn (test_coherency cycles 11 equations on one
  program ≈ 11 compiles, one-time each).
* MSAA advanced blending forces per-sample shading (slower but correct);
  single-sample is unaffected.
* Integer attachment-0 formats fall back to overwrite (blending is a no-op
  for integer targets in GL anyway).
* glBlendBarrierKHR is not exported: MGL advertises the coherent variant, so
  no barrier is ever required (Metal tile reads within a pass are coherent).
  Documented as a gap.
* Textual MSL rewriting assumes the SPIRV-Cross entry signature shape
  (struct return with `[[color(0)]]` float4 member, `return out;` forms);
  anything else falls back to overwrite with a one-time diagnostic.

## Test
tests/headless/test_compat_advblend.c — all 15 equations pixel-exact vs the
CTS reference math, qualifier/MRT error paths, coherent-cap roundtrip,
Separate-rejects-advanced, equation queries; plus a core-context check.
Then the KHR-GL46.blend_equation_advanced CTS group locally.
