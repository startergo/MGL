# tess — round 9 handoff: tessellation_shader conformance (in progress)

Branch `gl46/tess`, HEAD **952a232** (= gl46-port 769277d merged + tess work).
**Full headless suite: 438 PASS / 0 FAIL.**  `MGL_FP64_FAST=1`: 433 PASS (only
the three bit-exactness fp64 tests differ — documented, merged by supervisor).

**Mergeable now: `READY 952a232`** — tess query-count fix + command-buffer
guards + trace dumps; no regressions (suite green), CTS improves 25→29 Pass.

## KHR-GL46.tessellation_shader current state (58 cases)

29 Pass / 25 Fail / 2 "TIMEOUT" (they RUN, just slower than the 180 s harness
limit — the counting-query fix makes the CTS vertex-counter draws take the
exact-count CPU route) / 1 NotSupported (`max_in_out_attributes`, needs
MAX_..._ATTRIBUTES limit work) / 1 crash (`single.primitive_coverage`).

### Done this round
* **Counting queries on tessellated output**: `mgl_queries_tess_counts_active()`
  (queries.c); while PRIMITIVES_GENERATED(stream 0) / TESS_EVALUATION_-
  INVOCATIONS / clipping counters are active the tess draw takes the exact-count
  CPU reference path (hardware tess factors are GPU-computed; CPU cannot know
  the counts).  This fixed the whole CTS `glMapBufferRange() failed:
  GL_INVALID_OPERATION` cluster: the CTS sizes its TF buffers from a
  PRIMITIVES_GENERATED query (with RASTERIZER_DISCARD), got 0, mapped length 0
  (spec: INVALID_OPERATION).  Now `tessellation_shader_tessellation.{TES,TCS_TES}`,
  `gl_InvocationID...`, `vertex_spacing/ordering` run their real assertions.
* **Committed command buffer guards**: CPU readbacks commit the command buffer;
  the geom tess-xfb route must not ask a committed buffer for an encoder
  (Metal asserts).  A nil encoder is handled as "no work".  (Do NOT call
  newCommandBuffer there — its event/serial re-arm deadlocks the GPU,
  verified.)
* xfb-trace dumps (geometry.c): `MGL_GEOM_XFB_TRACE=1` prints the tess capture
  member table and the TF varying mapping (run headless tests DIRECTLY —
  `tests/headless/run.sh` swallows stderr!).

### Where the next session should pick up (active investigation)
**Multi-varying TF capture corrupts data** — the root of
`tcs_to_tes.gl_in/gl_tessLevel/data_pass_through/...` ("Invalid component at
index 0 (found:0 expected:22)").  Repro: `tests/headless/zz31_glin.c` (untracked
scratch; run `./build/tests/zz31_glin` directly, env `MGL_GEOM_XFB_TRACE=1`).
Findings so far:
* ONE TF varying → whole chain (VS block/plain → TCS gl_in/patches → TES →
  capture → TF) is CORRECT.
* TWO SEPARATE_ATTRIBS varyings (vec4 + ivec4) → second buffer all zeros AND
  the FIRST varying's y component becomes 0.
* Routing/matching is correct (trace: members result0@4, result1@8,
  gl_Position@0, stride 12; vars buf/off sane; `mgl_geom_xfb_tess` pack kernel
  in MGLRenderer+Geometry.m:900 looks word-correct).
* Next suspects: the post-link stride/offset recompute in
  `mgl_geom_tess_xfb_route` (geometry.c ~6425, "words may have been filled in
  from the published capture member table") and the CAPTURE kernel's record
  packing for ivec4 members (tessellation.c `mgl_tess_capture_layout`), or the
  vars[] table field order vs the MSL `struct mgl_geom_xfb_var`.

### Remaining failures, grouped with hints
* `vertex.vertex_ordering` — real assertion now: "Clockwise ordering was
  expected but retrieved tessellation coordinates are laid out in
  counter-clockwise order" → tessellationOutputWindingOrder mirroring
  (y-flip) in the CPU + GPU tessellator paths (MGLRenderer+Tessellation.m
  `vertex_order_ccw`, tessellation.c CPU tessellator).
* `single.primitive_coverage` — CRASHES (assert or hang under 180 s; reproduce
  with `bash agents/conformance/one.sh` + longer window).
* `quads.inner_tessellation_level_rounding`, `triangles.degenerate_triangle` —
  "Zero vertices were generated": degenerate/negative levels must still
  generate vertices (negative → clamp to 0 per-edge rules; inner 1/2 with
  equal spacing still generates the interior).
* `tc_barriers.*` (3) — TCS barrier semantics (reads of other invocations'
  outputs).
* `point_mode.*` (2) — point rendering verification.
* `invariance_rule1..7` — tessellation invariance (needs deterministic
  per-vertex output; likely follows the plumbing fixes).
* `compilation_and_linking_errors.tc_invalid_output_patch_vertex_count` —
  missing link error (TCS output vertices > MAX_PATCH_VERTICES).
* `single.xfb_captures_data_from_correct_stage` — "Separable program should
  have succeeded": separable TES program + glCreateShaderProgramv link failure
  (esextcTessellationShaderXFB.cpp:228).
* `single.max_patch_vertices` — 32-vertex patches: captured vec4 wrong scale
  (explicit-length gl_in arrays; patch vertex count wiring).
* `single.isolines_tessellation` — isolines rendering check.
* `max_in_out_attributes` NotSupported — MAX_GEOMETRY.../limit queries; decide
  real limits vs genuine gap.

### Harness notes
* Case list: `build/tess_list.txt`; per-case: `bash agents/conformance/one.sh
  '<case>'` (180 s timeout, prints `== name: status` on stdout, glcts output
  on stderr).
* `MGL_TESS_DEBUG=1`, `MGL_GEOM_XFB_TRACE=1` for traces; CTS runs print them
  (one.sh passes stderr through).
* The CPU reference route is slow (per-draw readback); the two "TIMEOUT" cases
  need a longer window or the counter program sped up later (the count query
  is what forces the CPU path).
