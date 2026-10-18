# tess domain — round 7 final report (branch gl46/tess, HEAD = 0e87d51)

Commits this round: `9b3ac72` (single drawPatches + 8-byte factor records + ring — in the
history), `754fc7b` + `0e87d51` (reports/close-out). Working tree clean; branch HEAD carries
the round-6 green code (per-patch loop).

## Root cause: CONFIRMED (supervisor's stride diagnosis)

`sizeof(MTLTriangleTessellationFactorsHalf) == 8`, `MTLQuad... == 12` (verified with
`.agent/probe_factor_sizes.m`). MGL wrote 12-byte factor records for triangles; a multi-patch
drawPatches reads patch N's factors at `8*N` — patch 1 reads patch 0's padding (zeros →
dropped), patch 2 reads patch 1's edge factors. This reproduces every recorded misrender, and
the standalone replay (`.agent/scratch/replay.m`, `F8` switch) renders all four patches
correctly once the records are repacked to 8 bytes. Two measurement artefacts compounded the
"live-only" appearance: the harness framebuffer is **200×200** (tests sampled 128-based
coordinates straddling patch boundaries), and a full-fb glReadPixels sweep forces enough
synchronisation to mask the underlying scratch race.

## `9b3ac72` — the fix, in the history

- `mgl_tess_factor_stride` (8 tri / 12 quad) used by every factor sizing/fill/readback and the
  TES tess-level attribute layout; triangle TCS kernels recompiled at link with the triangle
  factor struct (incl. the scalar inside-factor fixup).
- ONE `drawPatches` with `patchCount = N`; four-patch colour test with per-patch
  `gl_PrimitiveID`, viewport-relative sampling; indexed+instanced patches — all pixel-verified.
- Per-draw scratch ring (exclusive blocks retired in `addCompletedHandler`, hazard-tracked,
  never resized in place), `setBytes` for small params, GPU-event compute→render ordering.

## Why the branch is not on `9b3ac72`

A **clean rebuild** (`make clean`) exposes a deterministic failure from the third tess draw
onward (first two pixel-correct, then black, no errors) — the "green" runs that validated the
commit were built with stale objects. This is a ring slot-lifetime / event-ordering bug.
The round-6 code kept as HEAD has the complementary defect: its singleton scratch buffers
race under load (standalone 5/5 pass after a clean rebuild; consecutive suite runs flake to
117 failures). Both defects are documented with the exact next steps (release the ring slot on
both command buffers; busy-acquire waits on the oldest slot; re-verify with clean rebuilds and
repeated full suites).

## Test state at HEAD (per-file, standalone)

- `test_tess` 294 checks, `test_tessemu` 4 checks — pass after a clean rebuild standalone;
  flaky under repeated/concurrent load (the singleton race).
- `test_geom` / `test_geom_spirv`: 12 failures at the merge base (pre-existing).
- Full suite: 21 files; tess green standalone; the above flakiness documented.

## Remaining (next round)

1. Re-apply `9b3ac72`, fix the draw-3 bug (ring release on both command buffers, bounded
   busy-wait), verify with clean rebuild + repeated full suites.
2. Steps 3–5 of `tess_round3.md` (TCS blob on GPU, GPU index reads, isolines/point_mode pixels).
