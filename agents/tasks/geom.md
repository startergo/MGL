# Domain: geom — geometry shaders on Metal (compute emulation)

Worktree `wt/geom`, branch `gl46/geom`. Your files: `MGL/src/geometry.c`, `MGL/src/mgl_ext_geom.h`,
new `MGL/src/MGLRenderer+Geometry.m` (category `MGLRenderer (Geometry)`, binder
`-bindExtGeomFuncsToGLMContext:`), `tests/headless/test_geom.c`, optional new `MGL/src/*.cpp` for a
SPIR-V pass. First run `git merge gl46-port`.

## Current state
`program.c`'s link driver compiles geometry shaders but fails the link with an info log — find and
replace that refusal. Draw path: `draw_buffers.c` exec_draw_raw → `MGLRenderer+Draw.m`. You may edit
`program.c`, `draw_buffers.c`, `MGLRenderer+Draw.m`, `MGLRenderer.m` surgically (list each edit). A
parallel `tess` agent edits the same spots — keep logic in your files behind small hook calls. Adding a
`.cpp` needs a minimal Makefile rule for `MGL/src/*.cpp` (allowed; report it).

## Design — compute emulation (works on every Metal GPU; mesh shaders are a later optimization)
1. **Vertex stage → buffer:** run the VS as a compute kernel writing stage outputs per vertex/instance
   (SPIRV-Cross MSL vertex-for-tessellation / capture-output-to-buffer options, or the xfb domain's
   capture-variant technique in `transform_feedback.c` / `MGLRenderer+TransformFeedback.m`). Choose the
   more robust and say why.
2. **Geometry stage → compute kernel** (SPIRV-Cross cannot translate GS). Preferred: a SPIR-V-level
   transform (also handles SPIR-V GS from glShaderBinary): execution model → GLCompute; `gl_in[]` and
   input arrays → reads from the VS output buffer indexed by the assembled primitive's vertex indices
   (points/lines/triangles + adjacency, from draw mode and index buffer); gl_PrimitiveIDIn,
   gl_InvocationID (`invocations = N` → extra dispatch dimension); EmitVertex/EmitStreamVertex → append
   current outputs (gl_Position, gl_Layer, gl_ViewportIndex, gl_PrimitiveID, user outputs) to an output
   vertex buffer via an atomic counter, honouring max_vertices; EndPrimitive/EndStreamPrimitive → close
   the strip; build list indices from strips. Implement in C over SPIR-V words or with SPIRV-Tools' C++
   API. A GLSL source rewrite is acceptable only if robust (state its limits).
3. **Draw:** the compute pass writes a DrawIndirect / DrawIndexedIndirect argument buffer from the atomic
   counter (no CPU readback); draw with a generated pass-through vertex function reading the GS output
   buffer + the program's fragment shader. Order compute/render correctly (buffers' serial tracking).
4. **Layered/viewport:** gl_Layer → `[[render_target_array_index]]` (fb sets renderTargetArrayLength for
   layered attachments); gl_ViewportIndex → `[[viewport_array_index]]` (rstate sets 16 viewports).
5. **Y-convention:** the `rstate` domain injects a clip-space fixup (y=-y, z=(z+w)/2, point size;
   function constants 32000+) into the VERTEX stage MSL at post_link (`MGLRenderer+RenderState.m`,
   `mgl_ext_rstate.h`). With a GS, apply it to the LAST pre-rasterization stage — your pass-through
   vertex function — never the VS-as-compute or the GS kernel. Keep the rule general so the tess agent's
   TES rule is compatible.
6. **Interactions:** transform feedback captures GS output, including streams 1–3 (only stream 0
   rasterizes); queries GEOMETRY_SHADER_INVOCATIONS, GEOMETRY_SHADER_PRIMITIVES_EMITTED,
   PRIMITIVES_GENERATED per stream (from counters, read at resolve); GetProgramiv GEOMETRY_VERTICES_OUT,
   GEOMETRY_INPUT_TYPE, GEOMETRY_OUTPUT_TYPE, GEOMETRY_SHADER_INVOCATIONS via
   `mgl_program_register_programiv_hook`; limits MAX_GEOMETRY_OUTPUT_VERTICES (256),
   MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS (1024), MAX_GEOMETRY_SHADER_INVOCATIONS (32), MAX_VERTEX_STREAMS
   (4); draw-mode vs GS input validation (INVALID_OPERATION) including adjacency (with a GS, adjacency
   vertices must reach gl_in); separable pipelines. Tessellation + GS: consume a "TES to buffer" capture
   if the tess domain exposes one; otherwise fail with a clear link/validation error and document.

## Tests (pixel-verified with glReadPixels)
Pass-through triangles; point → quad (triangle_strip); line_strip output; amplification + max_vertices;
EndPrimitive splitting; gl_PrimitiveIDIn / gl_InvocationID with invocations; gl_Layer into a 2D-array FBO
(read per layer); gl_ViewportIndex; lines_adjacency / triangles_adjacency input; instanced, indexed,
indirect draws; xfb of GS output incl. stream 1; primitive queries; link errors and draw-mode mismatch;
separable pipeline with a GS.

Errors: `ERROR_RETURN(GL_X); return;` (the macro does not return by itself).

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
