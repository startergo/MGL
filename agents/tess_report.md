# tess domain — round 42 report (priority: extension audit vs CTS; then robustness/perf; merges e8ff34b → 38057b4 → 1121e66)

READY 10db5c4 (gl46-port 1121e66 [geom r78-81] merged as bbcd43f; every
number below is on exactly this tree unless noted).

## Task 1 — audit every advertised extension against its KHR-GL46 group

Method: I re-ran the extension-mapped KHR-GL46 groups **fresh on this round's
trees** (results dirs `~/MGL-cts/tess-r42-audit` / `-spot` / `-spot2` /
`-postmerge` / `-final` / `-final2`, 3827 cases across 70 groups), skipping only
the groups the supervisor assigned to compat (direct_state_access,
buffer_storage, copy_image, texture_border_clamp, blend_equation_advanced,
sample_shading, shader_multisample_interpolation, texture_buffer, gl_spirv,
packed_pixels) and geom's multisample-image work; compat's own full-package run
(9493 cases on e8ff34b+fix, 0 Fail — supervisor attestation) covers the rest.

**Findings.** Two real problems surfaced; both resolved this round:

| extension | CTS group | result |
|---|---|---|
| GL_ARB_polygon_offset_clamp | polygon_offset_clamp | **Fail=1 found by my fresh audit** (PolygonOffsetClampMinMax: GL-depth offset applied in Metal depth space, r*units stopped short of the clamp) → fixed by compat s35 in gl46-port 38057b4; re-verified **3/3 Pass on my merged tree** |
| GL_ARB_texture_swizzle | texture_swizzle | 4/4 **Pass**, but `smoke` needs ~466 s on this M4 Pro (measured; it builds ~52k programs, link-cache hit ≈1.7 ms each + 104k draws/readbacks) — flagged as a perf item for the texture/readback domain; my audit-runner timeouts (25-40 s) are the only thing that marked it Timeout |
| GL_ARB_post_depth_coverage | post_depth_coverage_tests | not advertised (r41); group gates to **2 NotSupported** (re-verified) — correct |
| everything else I ran | (table below) | **100% Pass**, or Pass + documented CTS-gated NotSupported |

Fresh per-group evidence (mine, this round; "NS" = CTS-gated NotSupported with
the documented reason):

```
100% Pass (fresh): CommonBugs 8, aggressive_optimizations 12, clip_control 16,
clip_control_ARB 8, clip_distance 6, compute_shader 42, conditional_render_inverted 2,
constant_expressions 830, cull_distance 6, draw_indirect 210, draw_indirect_43 4,
explicit_uniform_location 48, exposed_extensions 1, ext_texture_shadow_lod 16,
geometry_shader 544, get_texture_sub_image 2, glsl_noperspective 2, gpu_shader5 22,
gpu_shader5_gl 3, incomplete_texture_access 1, indirect_parameters_tests 3, info 6,
layout_binding 21, layout_location 35, limits 140, map_buffer_alignment 2, multi_bind 17,
nearest_edge 4, packed_depth_stencil 25, parallel_shader_compile 3,
pipeline_statistics_query_tests_ARB 10, program_interface_query 43, separable_programs_tf 6,
sepshaderobjs 8, shader_atomic_counter_ops_tests 4, shader_atomic_counters 30,
shader_ballot_tests 4, shader_group_vote 5, shader_storage_buffer_object 124,
shader_subroutine 28, shader_viewport_layer_array 3, shaders42 2, shaders44 12, shaders45 21,
shading_language_420pack 43, spirv_extensions 1, stencil_texturing 1, sync 1,
tessellation_shader 232, texture_barrier 4, texture_barrier_ARB 4,
texture_filter_anisotropic 2, texture_gather 75, texture_repeat_mode 162,
texture_size_promotion 1, texture_view 21, transform_feedback 80,
transform_feedback_overflow_query_ARB 13, vertex_attrib_64bit 4,
vertex_attrib_binding 30, viewport_array 17
Pass + gated NS: draw_elements_base_vertex_tests 11+1NS (ES-only case by CTS design),
internalformat 51+1NS (rgb9_e5 readback), sample_variables 456+114NS (no 8x MSAA on
Apple silicon), shader_image_load_store 49+2NS / shader_image_size 38+36NS /
shader_texture_image_samples_tests 3+1NS (GL_MAX_IMAGE_SAMPLES gate; geom documented),
texture_cube_map_array 25+2NS (ES-only by design)
Not advertised (verified gating): post_depth_coverage_tests 2NS,
texture_filter_minmax_tests 5NS (r41), sparse_* NS
```

No extension had to be de-advertised this round (r41's post_depth_coverage
removal remains in place).

## Task 3 — robustness/perf in the tess domain: GL_PATCHES draw-path rework

`.agent/ab_patches.c` (kept in the worktree, uncommitted) measures the
submit cost of GL_PATCHES draws. Baseline (gl46-port 1121e66): every tess draw
(1) signalled an MTLEvent and **committed its own command buffer**, then
(2) opened a fresh render buffer with an event wait and (3) set `DIRTY_ALL`,
so the next draw rebuilt the whole `MTLRenderPipelineDescriptor`
(~126 µs of NSString-keyed descriptor per draw) — ~268 µs of CPU per
GL_PATCHES draw vs ~5 µs for a plain drawArrays.

What I changed (`MGLRenderer+Tessellation.m`, three edits):

1. **One shared command buffer.** The VS/TCS/blob compute encoders and the
   drawPatches render pass now go into the SAME command buffer (Metal runs the
   encoders of one buffer strictly in encoding order — the ordering the old
   per-draw signal/wait event pair used to buy). No event, no per-draw commit,
   no second buffer. The readback / CPU-reference / debug paths keep the old
   committed-buffer + event flow.
2. **Narrow dirty bits.** After drawPatches the code sets
   `DIRTY_VAO|DIRTY_RENDER_STATE` instead of `DIRTY_ALL`: the VAO bit re-runs
   the vertex-buffer map list (the draw bound the tess rings at
   MGL_TESS_VD_CP/PATCH/FACTORS) and keeps compat's pipeline fast path
   admissible (it declines on DIRTY_FBO); pipeline-descriptor builds dropped
   from 1/draw to ~1/120 draws (`pipeHit` in MGL_PROFILE; the descriptor bucket
   went 184 ms → 3.2 ms per 600 draws).
3. **AGX keep-alive flush (the interesting bug).** Metal batches command-buffer
   submissions on a timer (seen as `-_MTLCommandQueue
   _submitAvailableCommandBuffers` on a dispatch source in `sample`). A shared
   buffer that carries several tess draws and stays uncommitted lets that
   machinery go idle; when the buffer finally commits — at ring exhaustion —
   the completion notification (and with it every ring-slot retire handler)
   can take **~250 ms under load / ~6.7 ms quiet**, all slept through inside
   `tessRingAcquire`. Mitigation: commit the shared buffer every **2nd** tess
   draw, and if the ring still goes dry, tighten to **every draw for the next
   64 draws** (that is exactly the tight-loop shape where submission latency
   dominates). The ring spin itself now force-flushes when all 8 slots sit on
   the open buffer (it can end the render encoding and commit, instead of
   spinning).

Numbers (quiet machine, min-of-3, 600-1000 draws, submit-only):
GL_PATCHES **noTCS 262 → 153 µs/draw, TCS+TES 190 → 92 µs/draw**; hook
p50 2 µs (p99 162 µs); per-draw max 9.5 ms vs 281 ms before the keep-alive.
Honest caveats: (a) in tight submit loops the p99 is set by Metal's batch
window (~5-7 ms) where the old per-draw-commit design could stay < 1 ms on a
quiet box — under load the old design also showed 7-32 ms tails, and mine is
bounded (~7 ms) since the 250 ms case is gone; (b) wall-clock A/B on this
machine is contaminated whenever other agents' CTS runs are active (I measured
both directions); the CPU-side accounting (PROFILE buckets, hook percentiles)
is load-independent and is what I quote.

## Gates on this tree (10db5c4)

* `tests/headless/run.sh` (full): **135 test summaries, 0 FAIL / 0 BUILD-FAIL,
  682 PASS checks** (run after every step; final run after the last edit).
  test_tess.c: 684 checks, 0 failures.
* KHR-GL46 fresh on the final tree: **tessellation_shader 58/58,
  geometry_shader 136/136, transform_feedback 20/20 Pass** (plus the 490-case
  sweep in tess-r42-final across sample_variables/shaders45/draw_indirect/
  texture_view: 100% Pass except the documented 38 no-8x-MSAA NS).
* `python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT`: clean for
  `MGL/src/MGLRenderer+Tessellation.m`, `MGL/src/tessellation.c`,
  `MGL/src/mgl_ext_tess.h`.
* ab_bench min-of-7 (ab_old = baseline-1121e66, ab_new = 10db5c4): every mode
  within 0.81x-1.21x — machine noise (re-ran the worst mode 4x: ranges fully
  overlap, e.g. old 49-63 ms vs new 39-70 ms); none of these modes execute the
  GL_PATCHES path.
* Locked glmark2 (interleaved 6+6, scenes ideas/jellyfish/gl41-geometry,
  DYLD_INSERT of nsgl dylibs, logs `~/MGL-cts/glmark2-bench/ab42[PC]*.log`):
  baseline 2246 2476 2606 2747 2773 2816 (median 2677) vs
  tip 1285 2617 2657 2693 2805 2837 (median 2675) — **parity** (the 1285 is a
  cold first run under load; max is the tip's).

## Shared-file edits

None this round (draw_buffers.c was used for temporary instrumentation only
and is byte-identical to HEAD). All work is in
`MGL/src/MGLRenderer+Tessellation.m` (my file).

## Known gaps / notes for other domains

* `KHR-GL46.texture_swizzle.smoke/functional` take ~8 minutes each (~52k
  program links at ~1.7 ms cache-hit each + 104k draws and depth readbacks);
  correctness is fine — perf item for the texture/readback/program domains.
* Metal's command-buffer submission batch window (~6.7 ms quiet, worse loaded)
  is a general latency floor for any MGL subsystem that parks completions on a
  shared, uncommitted command buffer; the keep-alive pattern (periodic commit
  + tighten on exhaustion) is reusable.
