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
   The geometry kernel is dispatched with `dispatchThreadgroupsWithIndirectBuffer:` from the
   threadgroup count the prep kernel wrote (`{groups, 1, 1}` — Y and Z must be 1, a count of 0
   dispatches nothing). An earlier measurement that called this entry point a no-op was wrong:
   the probe passed `{2,0,0}`.
3. **Transform feedback of GS output** — the records have a deterministic order (no atomics,
   no CPU sort) and the pack walks them directly. The *data* copy now has a GPU
   implementation (three kernels after the geometry kernel: a prefix sum over the per-thread
   emitted counts gives every output vertex its compacted transform-feedback index, a scatter
   writes the varyings, a finish kernel advances the write offsets and folds the stream-0
   primitive count), with the write offsets and "since the last resolve" counters in a
   per-object GPU block that the CPU folds in lazily — hooked into `glEndTransformFeedback`,
   `glPauseTransformFeedback`, `glResumeTransformFeedback`, `glEndQuery(…)` (the four
   primitive targets) and `glGetQueryObject*`. The per-draw inputs and the varying table live
   in a 4-slot ring so the CPU never overwrites a block the GPU is still reading.
   Two real bugs were found and fixed this round (both suggested by the lead's checklist):
   * the finish kernel advanced the write offsets by the **cumulative** vertex counter, which
     is never reset inside a block, so draw N advanced by every vertex so far — there is now
     a separate per-draw counter (`xfb[8]`) that the prefix kernel zeroes;
   * the counter block is now initialised by a **begin kernel on the command stream** (the CPU
     wrote it at encode time, racing the previous draw's finish kernel), and the per-draw
     inputs live in a 4-slot ring with a dedicated seed buffer — the ring slot is restored
     before the kernel runs, so reusing it for the seed read garbage.
   `memoryBarrierWithScope:MTLBarrierScopeBuffers` is issued between the pack's kernels.

   **Still not the default** (`MGL_GEOM_GPU_XFB=1` enables it): with it on, single-draw
   blocks are byte-exact (`xfb_of_geometry_output`, `xfb_streams` pass), but in a
   multi-draw block the *first* draws' scatter writes are lost — the reference block of
   `xfb_gpu_pack_multiple_draws_and_pause` ends up with only the last draw's vertices. The
   write offsets are now correct (the object folds 9 vertices / 72 bytes for three 3-vertex
   draws), so the remaining suspect is the scatter reading a per-draw input (stride/size)
   that makes it `continue` — i.e. the ring slot still being clobbered, or the record buffer
   (`gOutv`) already reused. The CPU pack stays the default path.

## Timings (`tests/headless/test_geom_perf.c`)

300k scattered indices, 40 frames, geometry pass-through shader:

| path | us/frame |
|------|----------|
| CPU assembly (fallback) | 630-840 |
| GPU assembly, worst-case dispatch | 490-550 |
| GPU assembly, indirect dispatch (exact count) | 485-540 |
| indirect draw, prep kernel reads the command | 533 |
| primitive restart every 6th index, worst-case dispatch | 49 800 |
| primitive restart every 6th index, indirect dispatch | 46 000 |

**1.2x-1.7x faster** for the normal (restart-free) case; the indirect dispatch wins ~8% when
primitive restart reduces the primitive count below the mode's worst case. Restart-heavy
draws are dominated by the prep kernel's single-workgroup scan (one tile per restart run); a
multi-workgroup prefix sum is the fix if that matters.

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
* xfb of GS output uses the CPU pack by default (a sync per draw); the GPU pack is implemented
  but gated off — see above.
* `MGLXfbObject.write_offset` is `GLsizeiptr` (64-bit) while the GPU block holds 32-bit words;
  the resolve widens each entry individually.
* `test_texparam` aborts intermittently at teardown (~20-40% of direct runs) — verified to
  happen with the unmodified base build as well (4/10 vs 2/10); its checks always report 0
  failures.
