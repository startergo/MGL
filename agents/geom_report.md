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

   All per-draw inputs (the input block, the varying table, the begin kernel's seed) are now
   passed with `setBytes:` — Metal copies them into the command buffer at encode time, so no
   later CPU write can change what a pending draw reads; the input ring and its seed buffer
   are gone.

   The geometry category also **enqueues its command buffer when it starts encoding a draw**
   (`[cb enqueue]`, guarded against Metal's "enqueue an already enqueued command buffer"
   assertion), so the queue order equals the creation order even when a later command buffer
   is committed earlier — the commit-order hazard the lead identified.

   Each draw now also gets **its own range of the record buffer** (`gOutv` is sub-allocated
   per draw, 256-byte aligned, growing on demand; the geometry kernel, the pack's scatter and
   the raster pass all bind it at that offset), so a later draw's kernel can no longer
   overwrite the records an earlier draw's pack still has to read.

   **Opt-in** (`MGL_GEOM_GPU_XFB=1`; the CPU pack is the default). The pause handling is now
   as the spec wants it: `glPauseTransformFeedback`/`glResumeTransformFeedback` only set the
   CPU flag, draws while paused skip the pack entirely (`mgl_geom_xfb_paused`, so neither pack
   runs and the CPU pack can no longer advance the object's write offsets behind the GPU
   block's back), the begin kernel runs only at `glBeginTransformFeedback`, and the CPU folds
   the block only where GL makes the values observable — `glEndTransformFeedback`,
   `glEndQuery` of the four primitive targets and `glGetQueryObject*`, each waiting for the
   command buffer that last wrote it.

   `gVsout` (the VS capture records) and `gPrim` (the header, capture index list and input
   primitive slots) are sub-allocated per draw too now, exactly like `gOutv`/`gCnt`/`gPrep`,
   and the 8-slot serial ring that used to guard `gPrim` (with its per-draw `wait_serial`
   stall) is gone. A transform feedback buffer without Metal backing gets one from the
   renderer (`geom_xfb_gpu_ensure`), so a block cannot start on the CPU pack (which advances
   the object's offsets on the CPU) and continue on the GPU one.

   The counter block is now fenced (`waitForFence:` before the prefix sum, `updateFence:`
   after the finish), because it is `MTLResourceStorageModeShared`, which Metal does not
   hazard-track across command buffers — draw N+1's scatter could read the write offsets
   before draw N's finish advanced them.  The fence wait is skipped when the pack shares its
   command buffer with the previous pack (Metal forbids waiting on a fence updated in the
   same buffer, and the encoders inside one buffer are already ordered).  The fold waits for
   the specific command buffer the pack was encoded into.

   The transform feedback buffers are marked GPU-written only *after* the scatter is encoded,
   and the serial is queried first so it belongs to the command buffer that carries it; the
   multi-draw test reads the buffer back with no glFinish (`immediate read` line).

   Lead followed and *excluded*: the trace now logs the MTLBuffer bound as the scatter's
   destination (with its serial and command buffer) and the backing the fold reads.  For the
   multi-draw test all three draws of a block bind the **same** backing the CPU later reads,
   in three separate command buffers — no backing is created, replaced or renamed between the
   scatter and the read.

   Correction of the earlier reading: `(0,1)` is **not** stale data from another test — it is
   the third vertex of the triangles most of these tests draw, and more importantly the same
   `(0,1)` corruption shows up with the **CPU pack** as well: on the multi-draw test the CPU
   path captures draw 1 correctly and draws 2-3 as `(0,1)` repeated (the varying comes out as
   `gl_Position.zw`).  So the remaining defect is *upstream of both packs* — in the capture
   pass / geometry-kernel input record for the second and later draws of a multi-draw block —
   and the multi-draw test passes on the CPU path only because both blocks are wrong in the
   same way.  The per-draw record sub-allocator no longer wraps silently (it starts a fresh,
   zero-filled buffer instead), so leftover records cannot be read; that was real but not this
   bug.  The GPU pack additionally mis-captures draw 1, which is the only remaining
   GPU-pack-specific difference.  The CPU pack stays the default.

   Timing (`tests/headless/test_geom_xfbperf.c`, 200 draws of 30k indices in one transform
   feedback block, 8 MB buffer): **CPU pack 759-1295 us/frame vs GPU pack 548-645 us/frame
   (~2x)** — the CPU path synchronises after every draw, the GPU path resolves the counters
   once at the end of the block.

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
* xfb of GS output uses the GPU pack by default; the CPU pack (sync + memcpy per draw) stays
  as a fallback for debugging (`MGL_GEOM_CPU_XFB=1`). In `test_geom_xfbperf` (200 draws of
  30k indices in one block) the two are within noise of each other on this virtualised GPU
  (GPU 593-613 vs CPU 582-649 us/frame) because the block is one long stream; the GPU pack's
  win is that it never stalls the CPU mid-block.
* Observation for follow-up (identical on both pack paths, so not a GPU-pack issue): in the
  multi-draw test the second and third captured triangles come out as (0,1) repeated — the
  geometry kernel appears to read the varying at the wrong record word offset for those
  draws, or the capture pass writes them shifted. `xfb_of_geometry_output` (single draw,
  value assertions) is exact.
* `MGLXfbObject.write_offset` is `GLsizeiptr` (64-bit) while the GPU block holds 32-bit words;
  the resolve widens each entry individually.
* `test_texparam` aborts intermittently at teardown (~20-40% of direct runs) — verified to
  happen with the unmodified base build as well (4/10 vs 2/10); its checks always report 0
  failures.

### Bisect state (commit 469bd84 round)

With the trace on (a commit+wait after every draw's pack) the multi-draw test is
**byte-exact and correct on both blocks**: the captured VS records, the prep header
(prims/instances/vcount/vmin), the counters and the write offsets are all verified right for
every draw, so the prep kernel and the capture pass are both exonerated.  `MGL_GEOM_COMMIT=1`
(commit per draw, no wait) still fails; `MGL_GEOM_COMMIT=1 MGL_GEOM_WAIT=1` passes.  The
remaining defect is therefore a CPU/GPU race that only a wait closes — but every buffer the
CPU touches between draws is now per-draw sub-allocated (records, counts, prep outputs,
primitive slots, VS capture records, inputs by value), which leaves MGL's own serial/buffer
tracking as the only un-instrumented shared state between draws.  Next step: log
`t->write_serial` / `completed_serial` for the transform feedback buffer at the final read
(`mgl_buffers_prepare_cpu_read`) with and without the per-draw wait.

### Round f5aa6f0+: readback confirmed as the race, wiper not yet identified

Correlated trace: the pack marks the transform feedback buffers with serials 2854/2855/2856
(`scatter dst[0] ... serial NNNN`), and the very next CPU read sees `write_serial=0 completed=0
(no wait)`.  So the tracking entry is created by the pack and wiped (or its table slot reused
and zeroed) before the read.  `mgl_buffers_track` zeroes entries when a slot is created or
reused (draw_ext.c:111/119/133) and `mgl_buffers_prepare_cpu_write` zeroes them again after an
orphan/rename (draw_ext.c:214) - one of those runs between the pack and the read, most likely
from the xfb domain's own `glEndTransformFeedback` bookkeeping.  Next step: stamp the buffers
again at the *end* of `mglEndTransformFeedback` (after that bookkeeping, i.e. a hook there or
re-stamp in the geometry domain's fold, which runs first and is therefore too early today) with
the pack's serial `gXfbSerial`, so `prepare_cpu_read` waits for exactly that command buffer.

### Round bca9e4c: the tracking theory was an artifact; the bug is a stale GS output range

Two corrections and one decisive narrowing:

* `mgl_geom_ctx` was only set at the end of `mgl_geom_route`, which the GPU-assembly path
  never reaches (it returns early), so the readback trace ran with `ctx == NULL` and printed
  `write_serial=0` for a **non-existent lookup**.  With that fixed the real state at the
  reference read is `write_serial=2854 completed=2854 (no wait)` — the tracking is **correct**
  and the fold has already waited.  The "two entries / wiped entry" theory is dead.
* `MGL_GEOM_XFB_FLUSH=1` now opts into the per-draw commit+wait separately from
  `MGL_GEOM_XFB_TRACE=1`, and `geom_xfb_dump` prints the GS output records at the fold:
  they are **correct for every draw of both blocks** (verified: tri1/tri2/tri3 positions
  *and* varyings, at the draw's own range).

* The failing values are the tell: with the CPU pack (which "passes" because both blocks are
  identically wrong) draws 2-3 capture `(0,1) (0,1) (0,1)` and `(-1.6,-1.2), (-1.6,0.8), (0,0)`
  — i.e. **doubled** vertices.  The doubling factor belongs to `xfb_of_geometry_output`'s
  geometry shader (`g_pos = v_pos * 2.0`), which runs earlier in the same process.  So the
  pack is reading GS output records left in `gOutv` by *another program's earlier draw*, while
  this draw's own records went elsewhere.  That points at the geometry kernel binding a stale
  `gOutvOff` for the second and later draws of a block (write and read disagree on the range),
  which the per-draw flush hides by forcing each draw into its own command buffer with fresh
  bindings.  Next step: log `gOutvOff` in the GS binding and in the scatter binding for draw 2
  of the multi-draw test; they should be identical — one of them is not.

### Isolation round: the separable failure is NOT caused by the allocator

`test_gen2.c` (only `control_no_geometry_shader` + `separable_pipeline_with_gs`) fails at BOTH 94116a0 (no allocator fix) and 25bdf79 (allocator fix). So the pipeline-bound geometry path is broken in isolation regardless of the pool, and my earlier attribution of the regression to the allocator was wrong â what changed with the allocator is only *where the test draws land* in the shared scratch buffer, which changes whether the broken pipeline path is hit at an offset that happens to be 0 (works) or not (fails).  Root cause still open; the next step is supervisor item (b): dump which vertex function the render pipeline for the pipeline composite uses and whether creation succeeds, since that is the one code path only `separable_pipeline_with_gs` exercises.

# Round 18 (this session)

1. `3dc0c94` — the pass-through vertex override survives rstate's specialization.
   `rstateConfigurePipelineDescriptor` (called directly and again through
   `bindBlendStateToPipelineStateDescriptor`) replaced the descriptor's vertex function with a
   specialized copy of the *program's* vertex shader, discarding the override; amplified
   geometry drew nothing. It is now re-applied after those calls as a specialized copy of the
   pass-through function (`geomSpecializeVertexOverride`, a stand-in Shader so rstate's own
   constants and cache are used). The vertexFunction is part of the gpuaudit PSO key, so an
   overridden and a plain build never alias.
   Also fixed the real reason the override was needed: `mgl_geom_compile_stage` returned early
   for a deferred (separable, GS-only) program *before* the builtin scan, so `in_has_pos` was
   never set and the capture variant stored nothing (gl_Position included) — the geometry
   kernel read zero inputs. The scan now runs for deferred programs too (clip sizes default to
   the maximum), and `geom_ensure` derives the input varyings' producing vertex outputs from
   the pipeline composite (`geom_derive_exprs`). `separable_pipeline_with_gs` now also draws
   pv alone after the geometry draw and the geometry pipeline again. `MGL_GEOM_TRACE_DRAW`
   dumps a draw's emulation buffers.
2. `f2e9f68` — the GPU transform-feedback pack is the default (`MGL_GEOM_CPU_XFB=1` keeps the
   CPU pack). 10/10 isolated runs of `xfb_gpu_pack_multiple_draws_and_pause` on a clean build
   with either path.
3. `b137afc` — arrays of interface blocks with xfb_offset are unrolled in `xfb_layout_shader`
   (transform_feedback.c, shared: two surgical edits — the array unrolling and AUTO_MAP
   locations/bindings on the recompile, so shaders relying on link-time locations compile).
   New `xfb_arrays_of_blocks_xfb_offset` test. `mgl_geom_ctx` /
   `mgl_geom_xfb_log_read_name` are declared for the harness in tests/headless/mgl_geom_diag.h.

## Still open
* A geometry stage cannot use interface blocks / struct-typed interface variables: the record
  layout has no room for the members, so xfb would silently capture zeros. A source-level
  check in `mgl_geom_compile_stage` that refuses the link was written but reverted: it broke
  every geometry link in a way I could not explain (the mode loop reported
  have_in/have_out = 1/1 yet its failure branch fired — stale/inconsistent build state I ran
  out of time to chase). The compute-source half of the work (emit_block_as_struct: a block
  becomes `struct T {...}; T inst[n];` with the member qualifiers dropped) was correct and is
  worth re-landing with the reflection unrolling.
* GS after tessellation / xfb from the tessellation stages: tess exposes no TES-output
  buffer for the hardware path (`mgl_tess_draw_patches` rasterizes directly); the emulated
  path does expose GPU scratch. Needs a TES capture variant feeding the geometry kernel.
* GS + interface blocks in the *raster* path (the pass-through vertex function would have to
  emit a matching block declaration for the fragment shader).
