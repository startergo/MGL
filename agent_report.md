# tess domain — round 7 final report (branch gl46/tess)

Commits this round: `9b3ac72` (single drawPatches + 8-byte factor records + ring — in the
history, currently reverted), `754fc7b` (report). Final branch state: **the round-6 green
code (per-patch loop)** — see "Why the single-draw work was reverted" below.

## Root cause: CONFIRMED (the supervisor's stride diagnosis was right)

`sizeof(MTLTriangleTessellationFactorsHalf) == 8`, `MTLQuad... == 12` (verified with
`.agent/probe_factor_sizes.m`). MGL wrote 12-byte factor records for triangles; a
multi-patch drawPatches reads patch N's factors at `8*N` — patch 1 reads patch 0's padding
(zeros → dropped), patch 2 reads patch 1's edge factors. This exactly reproduces every
recorded misrender pattern, and the standalone replay (`.agent/scratch/replay.m`, `F8`
switch) renders all four patches correctly once the records are repacked to 8 bytes. The
"live-only" appearance was compounded by two measurement artefacts I corrected along the
way: the harness framebuffer is **200×200** (the tests sampled with 128-based coordinates
that straddled patch boundaries), and a full-framebuffer glReadPixels sweep forces enough
synchronisation to mask the underlying scratch race.

## What `9b3ac72` contains (the fix, in the history)

- `mgl_tess_factor_stride` (8 tri / 12 quad) used by every factor sizing/fill/readback and
  the TES tess-level attribute layout; triangle-domain TCS kernels recompiled at link with
  the triangle factor struct (incl. the scalar inside-factor fixup).
- ONE `drawPatches` with `patchCount = N`; the four-patch colour test (`gl_PrimitiveID`
  per patch, viewport-relative sampling) passes; indexed+instanced patches pass.
- Per-draw scratch ring (exclusive blocks, retired in `addCompletedHandler`, hazard-tracked,
  never resized in place), `setBytes` for all small params, GPU-event compute→render ordering.

## Why it was reverted

After a **clean rebuild** (`make clean`) the single-draw state deterministically misrenders
from the third tess draw onward (the first two draws are pixel-correct); the earlier "green"
runs were built with stale objects that masked it. Symptom: draws 1–2 correct, draw 3+ black,
no error logs — a slot-lifetime/ordering bug in the ring or the event wait that only shows
under the exact timing of a fresh binary. The round-6 code has the same *class* of problem
differently: its singleton scratch buffers race (standalone runs pass 5/5; under full-suite
load, later draws go black), which is exactly the supervisor's original diagnosis.

## Branch state (committed)

The round-6 green code (per-patch loop, 12-byte records): `test_tess` 294 checks /
`test_tessemu` 4 checks — both pass standalone (5/5 consecutive runs after a clean rebuild);
`test_geom`/`test_geom_spirv` fail at the merge base as well (pre-existing). Under full-suite
load the tess tests are **flaky** (the singleton race).

## Next steps

1. Re-apply `9b3ac72` and fix the draw-3 bug: the ring release must be gated on BOTH the
   compute and render command buffers (the release currently rides the render buffer only),
   and a busy ring should wait on the oldest slot instead of failing; verify with a clean
   rebuild + `for i in 1..10 run` and the full suite.
2. Steps 3–5 of `tess_round3.md` (TCS blob on GPU, GPU index reads, isolines/point_mode
   pixels) — untouched.
