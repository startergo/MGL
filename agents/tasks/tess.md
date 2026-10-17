# Domain: tess — tessellation shaders (TCS/TES, GL_PATCHES) on Metal

Worktree `wt/tess`, branch `gl46/tess`. Your files: `MGL/src/tessellation.c`, `MGL/src/mgl_ext_tess.h`,
new `MGL/src/MGLRenderer+Tessellation.m` (category `MGLRenderer (Tessellation)`, binder
`-bindExtTessFuncsToGLMContext:`), `tests/headless/test_tess.c`. First run `git merge gl46-port`.

## Current state
`program.c`'s link driver compiles TCS/TES but fails the link with an info log ("no tessellation") —
find and replace that refusal. The draw path (`draw_buffers.c` exec_draw_raw → `MGLRenderer+Draw.m`)
validates GL_PATCHES and refuses to draw. You may edit `program.c`, `draw_buffers.c`,
`MGLRenderer+Draw.m`, `MGLRenderer.m` surgically (list each edit). A parallel `geom` agent edits the
same link/draw spots — keep your logic in your files behind small hook calls.

## Design (MoltenVK's approach; SPIRV-Cross MSL options are already linked)
- **Link** for programs with TCS and/or TES:
  (a) vertex shader → compute kernel writing stage outputs to a buffer
      (`SPVC_COMPILER_OPTION_MSL_VERTEX_FOR_TESSELLATION`, capture output to buffer, output buffer index);
  (b) TCS → compute kernel reading VS outputs, writing per-vertex + per-patch outputs and
      `MTLTriangle/QuadTessellationFactorsHalf` (input/output/patch-output/tess-factor buffer indices,
      multi-patch workgroup, raw buffer input);
  (c) TES → post-tessellation vertex function (`[[patch(triangle|quad, N)]]`).
  No TCS → built-in kernel writing `GL_PATCH_DEFAULT_OUTER/INNER_LEVEL` factors, control points pass through.
- **Draw (GL_PATCHES):** VS kernel over all vertices/instances → TCS kernel per patch
  (`GL_PATCH_VERTICES`: see `mgl_ext_rstate.h`) → render pass with `drawPatches` /
  `drawIndexedPatches` + post-tess vertex function + FS + `setTessellationFactorBuffer`. Partitioning
  from the TES layout (equal_spacing→Integer, fractional_even/odd), winding (see y-convention), max
  factor 64. Instancing, base vertex/instance, indirect (read counts on CPU), multi-draw. No per-draw
  CPU stalls; order compute and render work correctly (see buffers' serial tracking in MGLRenderer+Draw.m).
- **Metal gaps to emulate:** isolines (quad domain with suitable levels, or compute tessellation drawn as
  lines), `point_mode` (draw unique tessellated vertices as points), clamp levels to 64.
- **Y-convention:** MGL stores render targets in GL row order; the `rstate` domain injects a clip-space
  fixup (y=-y, z=(z+w)/2, point size; function constants 32000+) into the VERTEX stage MSL at post_link
  (`MGLRenderer+RenderState.m`, `mgl_ext_rstate.h`). With tessellation the fixup must go to the TES (last
  pre-raster stage), never the VS-as-compute. Adjust rstate's patch target surgically and keep the rule
  general ("last pre-rasterization stage" — the geom agent uses the same rule). The y-flip mirrors
  winding: choose Metal winding accordingly and verify with front-face culling.
- **Interactions:** transform feedback (`mgl_ext_xfb.h`) captures TES output when tessellation is active
  (e.g. a post-tess variant with rasterization disabled writing to a buffer); queries
  (`mgl_ext_queries.h`) TESS_CONTROL_SHADER_PATCHES, TESS_EVALUATION_SHADER_INVOCATIONS,
  PRIMITIVES_GENERATED from tessellated output; GetProgramiv TESS_CONTROL_OUTPUT_VERTICES,
  TESS_GEN_MODE/SPACING/VERTEX_ORDER/POINT_MODE via `mgl_program_register_programiv_hook`; glGet limits
  MAX_PATCH_VERTICES (32), MAX_TESS_GEN_LEVEL (64), MAX_TESS_* per GL 4.6 minimums; separable pipelines.

## Tests (pixel-verified with glReadPixels)
Triangle-domain patch with constant levels covering the viewport + interpolated TES colour; quad domain;
default levels (no TCS); fractional spacing changing vertex count (PRIMITIVES_GENERATED or xfb);
isolines; point_mode; instanced + indexed patches; glPatchParameteri errors; link/validate errors (TES
without TCS valid, TCS without TES link error); separable pipeline; xfb of tessellated output;
orientation + culling with tessellation.

Errors: `ERROR_RETURN(GL_X); return;` (the macro does not return by itself).

## CORRECTION (supervisor, verified on the M4 Pro): hardware tessellation WORKS on Apple silicon
`drawPatches` is NOT dropped. A minimal probe (`.agent/probe_drawpatches.m`: post-tessellation vertex
function `[[patch(triangle,3)]]`, `MTLVertexStepFunctionPerPatchControlPoint`, factor buffer set with
`setTessellationFactorBuffer`, `drawPatches:`) renders a red pixel at the centre. Build/run it:
`clang -fobjc-arc -framework Metal -framework Foundation .agent/probe_drawpatches.m -o /tmp/p && /tmp/p`.
An earlier "dropped" observation came from a setup bug (missing factor buffer, wrong pipeline
tessellation fields, wrong step function, or the vertex function lacking `[[patch]]`).

Required design — everything on the GPU, no `waitUntilCompleted` / CPU readback per draw:
- VS kernel → TCS kernel (SPIRV-Cross TESC→MSL kernel) writes control points, patch data and
  `MTLTriangle/QuadTessellationFactorsHalf` directly into the factor buffer (GPU → GPU).
- No TCS: a tiny kernel (or a constant buffer) fills the factor buffer from GL_PATCH_DEFAULT_*_LEVEL.
- TES → SPIRV-Cross TESE→MSL **post-tessellation vertex function** (native `[[patch(...)]]` output,
  not a "plain vertex function" rewrite); render with `drawPatches` / `drawIndexedPatches`
  (patch index buffer for indexed draws), instancing via instanceCount/baseInstance.
- Pipeline: `tessellationPartitionMode` from TES spacing (Integer/FractionalEven/FractionalOdd — Metal
  pow2 is unused), `tessellationOutputWindingOrder` from vertex_order mirrored by the y-flip,
  `maxTessellationFactor` 64, factor step function PerPatch (PerPatchAndPerInstance when instanced).
- Order: compute encoder then render encoder in the same command buffer (Metal tracks the hazard for
  untracked resources only if you use fences/barriers — use the buffers domain's serial tracking).
- Keep the existing CPU tessellator ONLY for what Metal cannot do natively: `isolines` and `point_mode`
  (or replace it with a compute-shader tessellator writing an index/coord buffer + indirect args, which
  is preferred: still no CPU readback). Document which path each mode takes.
- Re-verify every test in test_tess.c pixel-exactly on the hardware path; report the before/after.

## Reference implementations (read-only copies in `.agent/ref/`, Mesa commit 335d569, MIT licensed)
Mesa already solves this on Apple GPUs; study before designing, borrow ideas freely, port code only
with its MIT/Microsoft copyright header kept and a line in your report naming the source file.
- `.agent/ref/poly/` — the shared GPU emulation library used by BOTH Asahi (AGX GL 4.6, conformant) and
  KosmicKrisp: `tessellator.h`, `geometry.h`, `prim.h`, `cl/*.cl` (compute kernels: the D3D11 reference
  tessellator ported to GPU, GS/primitive-restart/index unrolling), `nir/` (the lowering passes).
- `.agent/ref/kosmickrisp/` — **Mesa's Vulkan-on-Metal driver** (macOS, MSL output: `compiler/nir_to_msl.c`,
  `vulkan/kk_cmd_draw.c`, `libkk/kk_tessellator.h`). It is the closest existing design to MGL: how it
  runs tessellation and geometry emulation through Metal compute + MSL is the model to follow.
- `.agent/ref/asahi/` (`libagx/*.cl`: tessellation, geometry, draws, query, copy) and
  `.agent/ref/gallium-asahi/` (the GL driver: agx_state.c draw path, xfb, queries).
MGL uses glslang → SPIR-V → SPIRV-Cross → MSL, not NIR: translate the algorithms (OpenCL C → MSL kernels
is mostly mechanical), not the NIR plumbing.
