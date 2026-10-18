# tess domain — round 3 report (branch gl46/tess)

Merged gl46-port `ea5a3b8` (incl. geom, gpuaudit). Committed: `b430013` (step 1).
Suite after every step: **19 test files, 0 failures** (`test_tess` 294 checks, `test_tessemu` 4).

## Step 1 — PSO cache key: DONE (commit b430013)

- **Failing test first**: `TessPSOCacheStress` — 50 link/draw/delete cycles of tess programs
  with distinct TES colours, cache on. Failed intermittently (stale PSO colours), exactly the
  address-reuse race.
- **Root cause**: caches keyed on raw object addresses. Two separate instances:
  (a) `gaPSOKey` (gpuaudit): missing `tessellationFactorScaleEnabled` /
  `tessellationControlPointIndexType`; `vf`/`ff` are `MTLFunction` pointers, so a freed
  function reallocated at the same address matched a stale entry.
  (b) **the tess kernel PSO caches were the actual black-quad culprit**: `tTcsPsoCache` keyed on
  the *program* pointer — a new program at a recycled address got the previous program's
  3-control-point TCS kernel for its 4-control-point draw (verified with a standalone
  triangle-then-quad binary: 1–3/20 black before, 0/200 after).
- **Fix**: cache entries retain the functions they were built from; lookups verify the retained
  objects (address reuse can never match); key gained the two missing fields; the tess
  gpuaudit-cache bypass was removed; VS/TCS/TES params moved to `setBytes` (inline) since the
  shared CPU-rewritten param buffers also raced in-flight draws.
- Suite: 19 files, 0 failures, 8 consecutive `test_tess` runs green.

## Step 2 — one drawPatches for all patches: STOPPED at the turn budget with a precise finding

Method per instructions: `MGL_TESS_TRACE=1` dumps the exact encoder calls, the vertex
descriptor, the tessellation pipeline fields, the entry MSL (TES+FS), and — from a
`addCompletedHandler` so the TCS kernel has actually run — the control point, patch and factor
buffer bytes (`.agent/scratch/trace/`). `.agent/scratch/replay.m` re-issues exactly those
calls; `.agent/scratch/mp5.m` is the same with a synthetic-data/factor-stride/mutation
harness.

**The exact smallest difference found** (all measurements reproducible on demand):

1. The traced 4-patch draw replays with **patches {0,3} rendered and {1,2} dropped** — patch 0
   red, patch 3 white, quarters 1 and 2 black — with MGL's exact MSL, descriptor, bindings and
   buffer bytes. So the failure is fully captured by the trace; no MGL-side hidden state.
2. Mutating only the control-point **data** (swap patch 0's and patch 1's records inside the
   buffer) moves the rendering to `{1,3}`: slot selection depends on the buffer *content*, not
   on slot index, buffer length (400 vs 768 bytes — tested with padding), MSL (MGL's exact
   TES/FS vs a minimal shader — both reproduce), factor values (2 vs 4 — both reproduce),
   instanceStride (0/12/16 — same), or buffer index (0 vs 8).
3. Rewriting only the **factor records** at controlled strides (`FSTRIDE`): 12 B → `{0,3}`,
   16 B → `{0,2}` (quarter 2 shows *slot 2's* colour), 24 B → `{0,3}`, 32 B → `{0}`.
   The tessellator's effective factor-record stride and its coupling to control-point slot
   selection do not follow the documented `sizeof(MTLTriangleTessellationFactorsHalf)` = 12 B
   semantics on this AGX driver.
4. A minimal probe with different buffer *contents* renders all four quarters "lit", but its
   centroid colours are wrong/black in the same pattern — i.e. **no configuration of a single
   multi-patch `drawPatches` was found that renders every patch with per-slot-correct data**;
   the per-patch rebind workaround renders all four correctly every time.

Conclusion: multi-patch `drawPatches` on this AGX (M4 Pro, macOS 26) driver does not honour the
documented factor-buffer/control-point layout; the per-patch loop stays (documented in
`MGLRenderer+Tessellation.m`). `gl_PrimitiveID` in a hardware-path TES therefore reads 0 for
patches > 0, and the 4-patch colour test stays behind `MGL_TESS_UNSTABLE=1`.

## Steps 3–5: not started (step 2 budget exhausted)

Remaining known gaps from previous rounds (unchanged): TCS patch-record blob still built with a
commit+wait on the emulated path (step 3), indexed patches still read the element buffer on the
CPU (step 4), isolines/point_mode still render the fallback image (step 5, draw-only tests).
