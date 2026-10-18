# tess domain — round 5 report (scratch ring; branch gl46/tess)

Merged gl46-port (geom, gpuaudit). Commits this round: none merged to the branch tip beyond
the previous state — the scratch-ring work is fully characterised below but reverted per the
round rules (green suite required; the single-draw multi-patch fix is not complete).
Suite at HEAD: **20 test files, 0 failures** (`test_tess` 294 checks).

## What was built (then reverted): per-draw scratch ring + ordering

- **Per-draw scratch ring** (`MGL_TESS_RING` blocks, retired in `addCompletedHandler`): every
  tess scratch range (VS-out, gathered indices, TCS control points, TCS patch outputs,
  factors, patch blob, emulated coords/indices/args) sub-allocated from one exclusive block
  per draw; small params via `setBytes`/`setVertexBytes` (inline, cannot be overwritten);
  never resized in place; the block is released only when the last command buffer encoding
  the draw completes. Replaces the file-static singletons the supervisor identified.
- **GPU-side compute→render ordering**: the tessellation kernels run in their own command
  buffer, `encodeSignalEvent` at its end, `encodeWaitForEvent` at the head of the render
  command buffer (no CPU wait), the in-flight buffer committed in between. Without this the
  results were non-deterministic; with it they are stable run-to-run.
- **Hazard-tracked ring storage** (`MTLResourceHazardTrackingModeTracked`): shared storage is
  untracked by default, and for untracked resources an event does not order *memory*. This
  made the draw deterministic.
- The 8-byte triangle factor records were re-applied (TCS kernel recompiled post-link with
  `MTLTriangleTessellationFactorsHalf`, all sizes/fills/readbacks/TES-attribute layout at the
  domain stride) and the per-patch loop removed (one `drawPatches` for all patches).

Result: single-patch draws all pass; multi-patch draws are deterministic but consistently
misrender — so the ring is correct infrastructure, but one more live-only defect remains.

## The exact unsolved difference (measured, reproducible)

With the trace (`MGL_TESS_TRACE`) capturing the live pipeline descriptor (tess fields +
vertex descriptor), the TES/FS MSL, the draw arguments, every bind offset, and the control
point / factor buffer *bytes* after the compute buffer completes (all verified correct:
records {pos, tc_col} per patch, factors `0x4000` at the 8-byte stride), `.agent/scratch/
replay.m` re-issues those exact calls and **renders all four patches correctly** — including
with the live framebuffer formats (BGRA8 + Depth32Float_Stencil8), depth test, viewport, both
clip-constant specialisations, and the ring-style single-buffer binds. The live MGL encoder
with the same inputs (verified by debug prints of the descriptor and binds at draw time, and
by re-issuing the draw with a freshly built pipeline on the same encoder) renders patches
with control points fetched from the wrong records: the rendered geometry matches control
points `{rec[3i], rec[3i+3], rec[3i+6]}` — i.e. control point *j* of patch *i* is fetched
from record `3(i+j)` (a 96-byte stride per control point) instead of `3i+j` (32 bytes), which
makes patch 1 degenerate (all three cps at y=+0.5) and stretches the others. The fetch
stride the hardware uses does not follow the vertex descriptor's 32-byte layout stride in the
live encoder, while the same descriptor in a standalone encoder does. This is the smallest
reproducible difference; everything upstream of it (buffers, descriptor, MSL, draw args,
ordering) is now byte-verified identical.

## Step 1 (PSO cache) — carried from the previous round, still green

## Remaining gaps
- Multi-patch hardware draws keep the per-patch loop (12-byte records, per-patch rebind) —
  the committed state; `gl_PrimitiveID` reads 0 for patches > 0; the 4-patch colour test
  stays behind `MGL_TESS_UNSTABLE=1`.
- Steps 3–5 of the round-3 brief (TCS blob without CPU wait, GPU index gather, isolines /
  point_mode pixels) not started.
