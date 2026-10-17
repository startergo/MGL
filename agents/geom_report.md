# geom — geometry shaders on Metal (compute emulation)

Branch `gl46/geom`. Files owned: `MGL/src/geometry.c`, `MGL/src/mgl_ext_geom.h`,
`MGL/src/MGLRenderer+Geometry.m`, `tests/headless/test_geom.c`, `tests/headless/test_geom_spirv.c`,
`tests/headless/test_geom_perf.c`.

## How it works

Metal has no geometry stage, so a program with a geometry shader draws in four GPU passes:

0. **prep kernel** — assembles the draw's primitives on the GPU (no CPU read of the element
   buffer, no CPU assembly). One workgroup scans the index buffer in tiles, splits at
   primitive restart and emits, per restart-free run: the capture index list (restart
   removed), the input primitive slot list for the geometry kernel (all 10 GL modes incl.
   adjacency and the provoking-vertex conventions), the vertex range of the capture pass
   (workgroup-reduced min/max), the capture pass's indirect draw arguments and the kernel's
   thread count. The draw command itself may come from a device buffer, so *indirect* draws
   need no CPU read either. Algorithm after Mesa's `poly_unroll_geometry`
   (`.agent/ref/poly/cl/restart.h`).
1. **vertex capture pass** — a rasterization-disabled render pass running a *capture variant*
   of the vertex shader (the `transform_feedback.c` technique: the original GLSL `main()` is
   renamed and wrapped so every output is stored per (vertex, instance) as raw 32-bit words in
   an SSBO). Metal's fixed-function vertex fetch handles every GL attribute format / stride /
   divisor / base offset, and the captured coordinates stay in GL clip space.
2. **geometry kernel** — the geometry shader rewritten by a GLSL source transform into a
   compute kernel: `gl_in[]` / user inputs load from the captured records indexed by the slot
   list, `EmitVertex`/`EmitStreamVertex` append the current outputs to an output vertex buffer
   (deterministic slot = thread * max_vertices + seq, honouring `max_vertices` and recording
   the emission stream), `EndPrimitive`/`EndStreamPrimitive` append the strip's raster indices
   and per-stream primitive counts, and a finalize kernel writes
   `MTLDrawIndexedPrimitivesIndirectArguments` from the counters — no CPU readback. If the
   stage only exists as SPIR-V (program binary, `glShaderBinary`) it is decompiled to GLSL
   with SPIRV-Cross first, so the same transform applies.
3. **raster pass** — a generated pass-through vertex function reads the output buffer at
   `[[vertex_id]]` and feeds the program's fragment shader (`gl_Layer` →
   `render_target_array_index`, `gl_ViewportIndex` → `viewport_array_index`, point size, rstate
   clip fixup), drawn with the GPU-built index list via an indirect draw.

Transform feedback of the geometry output: `glTransformFeedbackVaryings` resolves against the
geometry outputs (`gl_Position`, `gl_PointSize`, user outputs, `gl_NextBuffer`,
`gl_SkipComponents`), and after the kernel the emitted records are packed into the bound TF
buffers in emission order, with the object / context / query counters advanced.

## CPU work moved to the GPU (this round)

1. **Primitive assembly** — `geom_assemble` + the element-buffer read moved into the prep
   kernel (above). The CPU path remains as a fallback (`MGL_GEOM_CPU_ASSEMBLY=1`, or a
   client-side element pointer).
2. **Indirect draws** — `mgl_geom_route_indirect` hands the app's indirect buffer to the prep
   kernel, which reads the command itself; the CPU bounds the buffer sizes from the element
   buffer (indexed) or the VAO attribute buffers (arrays). `exec_indirect` no longer forces a
   CPU command read for geometry draws.
   *Caveat*: the geometry kernel is dispatched by the mode's worst case rather than
   `dispatchThreadgroupsWithIndirectBuffer:`, because that entry point is a **no-op** on this
   Metal implementation — verified with a standalone Metal program (`args = {2,0,0}`, 0
   threads executed). The kernel returns early past the real primitive count.
3. **Transform feedback of GS output** — the records now have a deterministic order (no
   atomics, no CPU sort), and the pack walks them directly. The pack itself is still CPU-side
   with a sync, because the xfb object's counters and the TF queries are CPU state in the xfb /
   queries domains; moving them to a GPU accumulator per object is the remaining work.

## Timings (`tests/headless/test_geom_perf.c`)

300k scattered indices, 40 frames, geometry pass-through shader:

| path | before | after |
|------|--------|-------|
| CPU assembly | 804-843 us/frame | — |
| GPU assembly (prep kernel) | — | 505-549 us/frame |

**1.2x-1.7x faster**; indirect draws with a geometry shader: 553 us/frame (no CPU command
read).

## Shared-file edits

* `MGL/src/program.c` — geometry stage routed through `mgl_geom_compile_stage`; stage mask
  allows vertex+geometry (and separable geometry-only); `mgl_program_check_geom_link`;
  `GetProgramiv` `GEOMETRY_*`.
* `MGL/src/draw_buffers.c` — `validate_program` no longer refuses geometry; draws route
  through `mgl_geom_route` / `mgl_geom_route_indirect`.
* `MGL/src/MGLRenderer.m` — `generatePipelineDescriptor` calls `-geomConfigureTopology:`
  (Metal needs `inputPrimitiveTopology` for layered output).
* `MGL/src/MGLRenderer_Private.h` — declares `-geomConfigureTopology:`.
* `MGL/src/transform_feedback.c` — geometry programs delegate their capture to the geometry
  domain instead of failing the link.
* `tests/headless/test_program.c` — `unsupported_stages_fail_link` updated for the new
  behaviour.

## Known gaps

* Tessellation before a geometry shader is rejected with a clear link error.
* `glRasterizationStream` is not tracked by MGL, so the captured stream is always 0; stream
  1-3 vertices are correctly excluded from the capture (tested).
* `gl_ClipDistance`/`gl_CullDistance` written by a GS are carried in the records but not
  applied to rasterizer clipping.
* The output vertex buffer is sized by the mode's worst case (clamped by a 256 MB budget);
  beyond that the kernel drops vertices and sets an overflow flag.
* A VS that redeclares `gl_PerVertex` with fewer members than the GS reads trips glslang's
  cross-stage check (pre-existing MGL/glslang behaviour).
* xfb of GS output still syncs the CPU for the pack (see above).
