# tess domain — round 3/4 report (branch gl46/tess)

Merged gl46-port `ea5a3b8`. Commits: `b430013` (step 1, PSO caches), `c6738b7` (round-3
report). Suite at HEAD: **20 test files, 0 failures** (`test_tess` 294 checks).

## Step 1 — PSO cache key: DONE (b430013, previous report)

## Step 2 — factor-record stride (8 tri / 12 quad): root cause CONFIRMED, MGL integration NOT solved

The supervisor's diagnosis is correct and I verified it end to end with probes:

- `sizeof(MTLTriangleTessellationFactorsHalf)==8`, `sizeof(MTLQuad…)==12` on this Mac.
- MGL wrote 12-byte factor records for triangles; in a multi-patch drawPatches Metal reads
  patch i's factors at `8*i`, crossing into the next record / the zero padding → zero factor →
  dropped patch. That reproduced my earlier content-dependent `{0,3}`/`{0,2}`/`{0}` patterns.
- Probe `.agent/scratch/mp5.m` (MGL's exact traced TES/FS MSL + vertex descriptor + traced cp
  bytes): with factor records rewritten at an 8-byte stride **a single 4-patch drawPatches
  renders every patch with correct colours** (`p0=red p1=green p2=blue p3=white`, full
  coverage top+bottom rows). With 12-byte records only patches {0,3} render.

I implemented the fix in MGL (triangle-domain TCS kernel recompiled post-link with the
triangle factor struct + scalar `inside` fixup; factor buffer sized/filled/read at the domain
stride; TES factor-attribute layout stride 8; single multi-patch drawPatches; compute phase
in its own command buffer). Result in MGL, reproducibly:

- Single-patch draws (Triangle/Quad/DefaultLevels/Orientation) pass — the factor records are
  now read correctly.
- Multi-patch draws MISRENDER in a way none of my probes reproduce: patches tessellate across
  control-point slots (colours bleed past the patch's convex hull), the domain's lower half is
  dropped, and the last patch is overwritten. The cp and factor buffers were dumped after
  command-buffer completion and are byte-correct, and the isolated replay of the SAME
  descriptor + MSL + buffer bytes renders correctly.
- Bisected against the isolated probes: BGRA8 vs RGBA8 colour attachment (the "colour shift"
  was my probe reading BGRA bytes as RGBA — not a driver issue), Depth32Float vs
  Depth32Float_Stencil8 (no difference), CPU-written vs compute-written buffers in the same
  command buffer (`.agent/scratch/mp9.m`: compute-filled buffers drop the LAST-written patch
  — a real hazard, worked around by running the tessellation kernels in a dedicated command
  buffer, which Metal orders before the render pass on the same queue; the misrender above
  remained), factor values ([2,2,2,2] vs [2,2,2,0,2,0] overlap — both correct in isolation).
- Per the round rules (green suite required, no workarounds), I reverted the integration to
  the last committed state (12-byte records + per-patch loop, which rebinds the factor offset
  per patch). Everything below is preserved in `.agent/scratch/mp5.m`, `mp9.m`, `mp10.m` and
  this report.

**Exact smallest unsolved difference**: identical vertex descriptor, MSL (specialized with
the same constants), buffer bytes, draw arguments and framebuffer formats render all patches
correctly in isolation, while MGL's render pass — whose compute phase ran in a prior command
buffer — fetches control points across patch records. The one thing the replay cannot copy is
the live encoder's full state (blend, the TES's uniform bindings at indices 0..k, textures),
so the suspicion is a state interaction on the tessellator's control-point fetch path that
only occurs in the full pipeline.

## Steps 3–5: not started (step 2 budget exhausted first)

Remaining gaps unchanged: TCS patch-blob commit+wait (step 3), CPU index reads (step 4),
isolines/point_mode fallback image (step 5), `gl_PrimitiveID`=0 for patches>0 on the hardware
path (consequence of the per-patch loop), 4-patch colour test gated behind
`MGL_TESS_UNSTABLE=1`.
