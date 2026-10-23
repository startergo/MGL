# tess — round 13 handoff: KHR-GL46.tessellation_shader (in progress)

Branch `gl46/tess`, HEAD 77965e8 (= merge of gl46-port 305b968 + ec6adc3 + 5 commits).
**Headless suite 61 files / 0 failures; test_tess 499/0; test_geom 593/0.**
CTS last chunk run: 39 Pass / 17 Fail of the 56 quick cases (vertex_spacing + vertex_ordering are
long, see below); round-12 baseline was 41/56 on the same subset. Some cases are FLAKY
(invariance_rule5/6, gl_MaxPatchVertices_Position_PointSize) — see "flakiness" below.

## This round's fixes (all committed)
1. **Per-instance gl_PrimitiveID** (`tcs_patch_primid()` in tessellation.c): SPIRV-Cross's
   multi-patch TCS kernel derives it from the global thread id; the global index is kept in
   `mgl_pid_g` (factor/float-level/patchOut writes stay per-(instance,patch)), the builtin becomes
   `mgl_pid_g % max(spvIndirectParams[2],1)`; the third indirect word = patches per instance
   (dispatcher passes 3 words). Plain/capture TES: `mgl_tess_params[1]` now carries patches per
   instance (`gl_PrimitiveID = coords.w % max(params[1],1)`). **Bug caution: run the float-level
   injection BEFORE this rewrite** (it matches `spvTessLevel[gl_PrimitiveID].`). 72-run matrix
   repro kept in `build/test_zz61.c` (untracked).
2. **Deterministic patch order** (MGLTessKernels.h): `mgl_tess_count` writes per-patch counts (no
   atomics), new `mgl_tess_scan` kernel does the exclusive prefix sum + totals; coords/indices now
   stream in patch order (xfb records, queries, gl_PrimitiveID depend on it).
3. **Link-fail on layout(vertices=N) outside [1,32]**: `mgl_program_check_tess_link()` called from
   the two link paths in program.c (shared-file edit, listed below).
4. **GL_MAX_VERTEX_OUTPUT_COMPONENTS = 128** (glm_params.c shared edit): max_in_out_attributes now
   runs but its TES indexes a 31x vec4 interface array with a loop variable — SPIRV-Cross cannot
   lower that (flattened composites + dynamic index). REMAINING GAP.
5. **GL patch-discard rule** (all outer levels <= 0 -> no vertices/primitives): `mgl_patch_discarded()`
   in the count helper + emit kernel + CPU tessellator + `mgl_tess_note_counts`.
6. **Ring-slot robustness** (MGLRenderer+Tessellation.m): `tessRingAcquire` waits (2 s bound) for
   in-flight slots, commits a pending command buffer when nothing forces a flush, releases directly
   when the release site has no command buffer, and retires STALE busy flags (a draw exits without
   registering its release — still a small leak, see TODO 3).

## CTS status after this round (quick cases)
FAIL: single.max_patch_vertices (case 2: plain-TES capture reads beyond 32 control points),
quads.inner_tessellation_level_rounding + single.primitive_coverage + invariance_rule2/3/4/7
(+ flaky 5/6) — ALL need the D3D11 tessellator layout (outer edges tessellated ONLY by their outer
level + ring stitching; MGL currently derives boundary vertices from the inner grid),
point_mode.point_rendering + points_verification (fractional-spacing vertex-count formula),
tcte.data_pass_through (repro `build/test_zz62.c` passes all 90 shapes standalone — needs the
CTS's exact config; suspect batch-state interaction), tc_barriers x3 (TCS barriers need
threadgroup shared memory — MGL runs the TCS as a multi-patch workgroup with 1 thread per
threadgroup, so gl_barrier() is a no-op), gl_MaxPatchVertices_Position_PointSize (NEW REGRESSION:
`glMapBufferRange` GL_INVALID_OPERATION at TCTE.cpp:2909 — knock-on of the ring wait? bisect),
degenerate_triangle (flaky in batches, passes alone), max_in_out_attributes (see 4).

## Flakiness to chase first (round 14)
invariance_rule5/6 and degenerate_triangle flip between runs in a 56-case batch but pass/fail
randomly; gl_MaxPatchVertices_Position_PointSize failed in the last batch only. Suspects: the
ring safety valve (2 s wait retires stale slots) and the batch's shared context. Reproduce with
`python3 build/chunk.py <caselist> 470` (runner kept in build/, untracked) — one batch at a time.

## Remaining work list (priority)
1. Port the D3D11 tessellator layout for quads+triangles from `.agent/ref/poly/cl/tessellator.h`
   (Mesa MIT): outer-edge vertices from the OUTER level only + ring stitching + the matching
   vertex-count formulas. Fixes invariance_rule2/3/4/7(+5/6), inner_tessellation_level_rounding,
   primitive_coverage and probably point_mode counts. This is the big one (5-8 tests).
2. max_in_out_attributes: needs unrolling constant-bound loops in the TES GLSL (or spirv-opt
   loop-unroll; SPIRV-Tools-opt is already linked but MGL doesn't use its C++ optimizer yet).
3. Ring leak: find the draw that exits without registering tessRingRelease (safety valve masks it).
4. tc_barriers x3: implement real TCS barriers (threadgroup shared memory, multi-invocation
   workgroups) — see `.agent/ref/poly/cl/tessellator.h` patterns.
5. vertex.vertex_ordering / vertex_spacing are LONG (>450 s each); run them individually with a
   bigger budget; vertex_ordering failed round-12 on triangles outer (1,33,1,-1) ccw orientation.

## CTS recipe
`bash agents/conformance/one.sh '<full case name>'` for one case;
`python3 build/chunk.py build/t13_c1.txt 470` for a batch (foreground, <=480 s, one at a time).
Debug envs: MGL_TESS_DEBUG, MGL_DUMP_MSL, MGL_TESS_DUMP_CAP, MGL_TESS_CPU=1.
