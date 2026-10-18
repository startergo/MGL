# tess domain — round 6 report (bind-trace bisection; branch gl46/tess)

Merged gl46-port (`61eb64c` → current, 21 test files). Nothing merged to the branch tip —
the round's work is fully characterised below and reverted per the rules (green suite).
Suite at HEAD: **21 test files, 0 failures** (`test_tess` 294 checks, `test_tessemu` 4).

## Step 1 — full encoder bind trace: DONE, hypothesis eliminated

Added `MGL_TESS_TRACE`-gated `fprintf` bind tracing (`__FILE__:__LINE__`, index, encoder) to
every `setVertexBuffer`/`setVertexBytes` call site that can touch the render encoder
(`MGLRenderer.m` uniform map binder, `MGLRenderer+VertexAttrib.m` all 8 sites, `MGLRenderer+Draw.m`,
`MGLRenderer+Tessellation.m` 9 sites). For the four-patch draw, in encode order:

```
VertexAttrib.m:383 setVertexBuffer idx=0   (GL position array — vattrib map slot 0)
VertexAttrib.m:383 setVertexBuffer idx=0   (GL colour array    — vattrib map slot 1, index reused)
Tessellation.m     setVertexBuffer idx=8  (control points)
Tessellation.m     setVertexBuffer idx=9  (patch data)
Tessellation.m     setVertexBuffer idx=10 (factors) + setTessellationFactorBuffer
drawPatches
```

**Nothing binds 8/9/10 after the tess binds** — the uniform-map binder never fires (no TES
UBOs in the test) and the vattrib binder uses the low indices. The supervisor's hypothesis
(another binder overwriting the cp slot) is therefore eliminated for this state.

## Steps taken beyond the trace (all measured, then reverted)

- Re-applied the full round-5 stack (per-draw scratch ring + `setBytes` params + dedicated
  compute command buffer + `encodeSignalEvent`/`encodeWaitForEvent` + hazard-tracked ring)
  and the 8-byte triangle factor records with ONE `drawPatches` for all patches.
  Single-patch draws pass; the four-patch draw is now **deterministic** but still misrenders.
- Reproduced the misrender **standalone** with the exact traced bytes: `.agent/scratch/replay.m`
  loads the traced pipeline descriptor (tess fields + vertex descriptor), the TES/FS MSL, the
  cp/factor buffer bytes dumped after the compute buffer completes, the traced bind offsets,
  the same specialisation constants (both clip variants), the same framebuffer formats
  (BGRA8 + Depth32Float_Stencil8 + depth test + viewport) — and with **8-byte factor records
  it renders all four patches correctly**; with the committed 12-byte records only patches
  {0,3} render. (Probe switches: F8 = repack records to 8 bytes — the fix; FSEP/FOFF/F8 +
  same-buffer offsets, TRK = hazard-tracked buffers, EXTRA0 = extra low-index binds, NOF10,
  CP0, BGRA/D32S8/CULL/SCISSOR — none affect the correct result.)
- Live-side eliminations this round: fresh PSO on the same encoder (same corruption), draw on
  a brand-new render encoder with only the three tess binds (same corruption — so prior
  encoder binds are not it), cp buffer additionally bound at index 0 (same), bind trace
  (above). The live pipeline descriptor debug print matches the traced descriptor field for
  field (partition=1 maxFactor=64 stepFn=PerPatch winding=CCW, layout[8] stride=32 step
  PerPatchControlPoint, attr0@0/attr1@16 both buffer 8).

**Net finding (supersedes round 5's):** with the factor records at the correct 8-byte
triangle stride, a single multi-patch `drawPatches` renders correctly from CPU-written
buffers in a standalone encoder, and still fetches control points as if cp *j* of patch *i*
lived at record `cpCount*(i+j)` when issued from the MGL renderer — with byte-identical
inputs, an identical pipeline descriptor, and no other binder touching the tess slots. The
remaining live-only variable is the exact `MTLFunction`/`MTLLibrary` object pair (live:
compiled by `MGLRenderer+Program.m`'s `compile_msl` at link; replay: recompiled from the
dumped MSL text at draw time) — i.e. a compiler/driver interaction on the AGX tessellator's
control-point addressing that depends on which library built the function. That is the next
thing to bisect (rebuild the live PSO from a replay-compiled function, or vice versa).

## Green state kept

`test_tess` 294 checks / `test_tessemu` 4 / full suite 21 files, 0 failures at HEAD
(`cda3b3c` + merge). All multi-patch hardware draws still use the committed per-patch loop
with 12-byte records; the ring/event/tracking work and the probes are preserved under
`.agent/scratch/` and this report for the next round.
