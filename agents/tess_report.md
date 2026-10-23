# tess — round 14 handoff: D3D11 tessellator layout (ring stitching, per-edge outer levels)

Branch `gl46/tess`, HEAD 8d3f852 (= merge of gl46-port b812d2a (GLX/NSGL) + the round-14
commits).  Note: gl46-port now tracks `external/glfw/*`; the worktree's `external` is a
symlink to the main checkout, so merge with the link moved aside and restore it after
(the round-14 sequence: `cp -P external .bck && unlink external && mkdir external &&
 git merge gl46-port && rm -rf external && mv .bck external`).
**Headless suite 61 files / 616 checks / 0 failures; test_tess 499/0; test_tessemu 117/0; test_geom 593/0.**

## What landed

1. **D3D11 reference tessellator** (the item left by round 13), ported into the CPU
   reference (`tessellation.c`) and transcribed 1:1 into the GPU emulated tessellator
   (`MGLTessKernels.h`), from Mesa's poly library (`.agent/ref/poly/cl/tessellator.h`,
   Microsoft/Valve, MIT — copyright headers kept, source named in the file comments):
   * outer edges are subdivided **only** by their own outer level; the outer and inner
     rings are stitched with the reference's ruler-function order (`stitchTransition` /
     `stitchRegular`, `finalPointPosition`), so boundary vertex counts/positions no
     longer follow `max(inner, outer)`;
   * exact `NumPoints` / `NumIndices` formulas in count mode = the same generator code
     that emits, so the GPU count pass and emit pass can never disagree;
   * the kernels now read GL's **exact float tess levels** (the TCS-written / default
     copy at `off_levels_f`) instead of Metal's half factor record, so generator,
     `mgl_tess_note_counts` (queries) and the CPU reference agree bit-exactly;
   * GL culling: a patch with any *relevant* outer level <= 0 (NaN passes) generates
     nothing — reference + Mesa semantics (round 13's "all <= 0" rule replaced);
     irrelevant levels (outer[3] for triangles, inner for isolines) are ignored.
   * isolines now use the reference placement `v = i/n0` (also the CTS reference
     layout, `vktTessellationUtil.cpp::generateReferenceIsolineTessCoords`), not
     `(i+0.5)/n0`.  Test expectations updated accordingly.

   Verified bit-exact against Mesa's reference over 3 domains x 3 partitionings x
   ~21k level combinations + the extreme sets (all 63/64, mixed), comparing the
   fixed-point point list, the counts and the index list (`.agent/d3d11/compare.c`,
   untracked).  Also: every emitted index < NumPoints.

2. **Two real bugs found by that verification**
   * the emit kernel added the patch base twice (pointer already pointed into the
     patch's range), so every patch but the first landed past its range;
   * the MSL transcription of the quad edge-3 degenerate-ring index patch used
     `ipc2_inv_end = base<<1` instead of `(base<<1)-1`, emitting indices one vertex
     past the patch for even-inside quads — the CTS invariance_rule7 "garbage
     triangle" (denormal gl_TessCoord records).
   * capture-record / permute reservations now use the D3D11 worst case per patch
     (quads 4225 v / 24576 i, triangles 3200 v / 18432 i, isolines 4160 v / 8192 i)
     instead of the old grid caps (triangles were 2145/12288 -> under-sized).

3. **New test-only diag API** (`tests/headless/mgl_tess_diag.h`):
   `mgl_tess_diag_cpu_counts()` and `mgl_tess_diag_cpu_mesh()` run the CPU reference
   tessellator.  `TessEmuStreamEqualsCpu` (test_tessemu.c) uses it to prove the
   emulated GPU stream equals the CPU stream **exactly** — one capture record per
   emitted vertex, in generation order (`record[i] == cpu_coords[cpu_indices[i]]`) —
   over 27 level sets x 3 spacings, including invariance_rule7's 61/60/59/58 +
   63/62 quads (23400 records).  This is the regression net for the dual
   C/MSL implementation; extend the level table if a new layout bug is suspected.
   Also added `TessCaptureCounts` (per-spacing counts + PRIMITIVES_GENERATED must
   match the CPU generator) and `TessCaptureHigh` (no stale/garbage records at high
   levels; ccw/cw streams hold identical triangles with the last two indices swapped).

## CTS (KHR-GL46.tessellation_shader) — before -> after

| case | round 13 | now |
|---|---|---|
| quads.inner_tessellation_level_rounding | Fail | **Pass** |
| triangles.inner_tessellation_level_rounding | Fail | **Pass** |
| single.primitive_coverage | Fail | **Pass** |
| tessellation_invariance.invariance_rule2..7 | Fail/flaky (2,3,4,7) | **Pass (all 7)** |
| quads/triangles degenerate_case, identical_triangles | Pass | Pass |

Run with (foreground, one glcts at a time, no alarm — let the runner kill its own batch):
```
R=$(pwd)/.agent/cts-tess2
python3 agents/conformance/run.py --pattern 'KHR-GL46.tessellation_shader.<group>.*' \
    --results-dir "$R" --refresh-caselist --max-seconds 300 --per-case-timeout 150 --batch 2
```
NOTE: `--results-dir` must be **absolute** (glcts's cwd is the CTS build dir) and each
pattern regenerates the caselist, so use one pattern per invocation.  Kill any leftover
`glcts` before starting the next batch.

## Open items

* **`single.isolines_tessellation` no longer finishes in 480 s** (no QPA progress after
  500 s).  Not reproduced in MGL: a 12k-configuration sweep (no-TCS default levels,
  and TCS levels with 1 control point + RASTERIZER_DISCARD + XFB capture + counter
  query run + buffer map, density/detail 1..64 x 3 spacings) is clean and fast, and
  PRIMITIVES_GENERATED == captured-records/2 everywhere.  The test is one of the LONG
  ones (a huge level matrix, two draws + a map per iteration); it may simply exceed the
  480 s budget now that the reference layout emits more vertices for fractional
  spacing.  Next step: give it a single-case run with a >480 s budget (needs the
  supervisor's allowance) or instrument the iteration count; if it is a real hang it
  is in the XFB/TF-buffer interaction (transform_feedback.c / MGLRenderer+Geometry.m),
  not the tessellator.
* `point_mode.point_rendering` / `points_verification` (round-13 leftovers) still fail:
  they need the *fractional-spacing vertex-count formula* in point mode — the D3D11
  count formula is now in place (`d3d_*` in count mode), so re-check them first.
* `tcte.data_pass_through` — untouched this round.

## Notes for the next agent

* The algorithm exists twice (C in `tessellation.c`, MSL string in `MGLTessKernels.h`).
  Keep them in lockstep: any change to one must be mirrored in the other, then extend
  the level table in `TessEmuStreamEqualsCpu` (and optionally re-run
  `.agent/d3d11/compare.c` against Mesa's reference, still untracked in `.agent/d3d11`).
* Scratch: `.agent/d3d11/` holds the oracle harness (untracked, keep it).
* Shared-file edits this round: none outside my own files (only `MGLTessKernels.h`,
  `MGLRenderer+Tessellation.m`, `tessellation.c`, `mgl_tess_diag.h`, `test_tessemu.c`).
