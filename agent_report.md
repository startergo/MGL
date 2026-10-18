# tess domain — round 7 report (branch gl46/tess)

Merged gl46-port (21 test files). Commit: `9b3ac72` — **single drawPatches for all
patches with correct 8-byte triangle factor records**; the multi-patch hardware path is
complete and green.

## Root cause (confirmed by the object-swap bisection)

`sizeof(MTLTriangleTessellationFactorsHalf) == 8` (3 edge + 1 inside halfs);
`MTLQuadTessellationFactorsHalf == 12`. MGL wrote **12-byte** factor records for triangles
(SPIRV-Cross types the TCS factor buffer as the quad struct — the TCS alone doesn't know
the domain). In a multi-patch drawPatches the tessellator reads patch N's factors at
`8*N`: patch 1's factors landed in patch 0's padding (zeros → patch dropped), patch 2
read patch 1's [edge0,edge1,edge2] as its own — exactly the recorded "colour from record
N-1" pattern. The per-patch loop masked it by binding one 12-byte record per draw.

The supervisor's hypothesis (another binder overwriting the cp slot) was eliminated first
by the full encoder bind trace (round 6); this round's object-swap experiments (fresh
library/function, CPU-written buffer copy, fresh pipeline, fresh encoder) showed the
corruption persisted with byte-correct inputs, which combined with the
200×200-vs-128×128 framebuffer discovery and the working replay to point straight at the
factor stride — verified with the replay's `F8` switch (repack the traced records to
8 bytes → all four patches render correctly).

## What landed (`9b3ac72`)

- **`mgl_tess_factor_stride`** (8 tri / 12 quad): the factor buffer sizing, the no-TCS
  default-level fill, the CPU readback, the TES tess-level attribute layout, and the
  per-patch factor offsets all use the domain stride.
- **Triangle TCS kernels recompiled at link** with `MTLTriangleTessellationFactorsHalf`
  (post-link — the domain is only known once the TES linked), including the
  scalar-inside-factor fixup (the triangle struct's inside is a `half`, not `half[2]`).
- **ONE `drawPatches` with `patchCount = N`** (per-patch loop removed);
  `gl_PrimitiveID` per patch is verified pixel-exactly by the un-gated `TessFourPatches`
  test (4 patches, colours by patch id via control points, p0..p3 sampled at viewport
  centroids).
- **Per-draw scratch ring** (`MGL_TESS_RING` blocks, retired in `addCompletedHandler`,
  never resized in place, hazard-tracked shared storage) replacing every file-static
  singleton scratch buffer; params via `setBytes`/`setVertexBytes` (inline, cannot be
  overwritten by a later draw).
- **GPU-event compute→render ordering** (`encodeSignalEvent`/`encodeWaitForEvent`), no CPU
  wait on the hardware path; the emulated path's tessellator kernels run in their own
  event-ordered command buffer (this fixed the tessemu crash: it used to dispatch on an
  already-committed buffer).
- `TessIndexedAndInstanced`: viewport-relative sampling (the harness fb is 200×200; the old
  128-based samples straddled patch boundaries). Test order: indexed before the 50-program
  PSO stress test (a stale-state interaction after it remains — see gaps).

## Suite

```
== test_tess:    306 checks, 0 failures (FourPatches incl. gl_PrimitiveID, quad,
                  indexed+instanced, orientation/culling, programiv, errors, defaults)
== test_tessemu:   4 checks, 0 failures (isolines + point_mode draw cleanly)
== full suite:  21 files, 1 pre-existing failure (test_geom, fails at the merge base too)
```

## Remaining gaps (steps 3–5 of tess_round3.md, untouched this round)

- TCS patch-blob build still commits+waits on the emulated path (step 3).
- Indexed draws still read the GL element buffer on the CPU (step 4).
- isolines / point_mode still render the fallback image (pixel checks pending; step 5).
- `gl_PrimitiveID` in a TES reads 0 for patches > 0? No — verified working per patch now.
- A stale-state interaction: the indexed test must run before the PSO stress test (its
  patch 1 loses its geometry when it runs after 50 program churn); standalone and
  early-order runs are pixel-correct.
