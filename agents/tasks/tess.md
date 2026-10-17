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
