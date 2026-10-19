# tess — final domain summary (round 17)

GL 4.6 tessellation shaders (TCS/TES, GL_PATCHES) on Metal, branch `gl46/tess`.
Files: `MGL/src/tessellation.c` (GL side: program introspection, MSL rewrites,
CPU reference), `MGL/src/MGLRenderer+Tessellation.m` (draw execution),
`MGL/src/MGLTessKernels.h` (GPU kernel source), `MGL/src/mgl_ext_tess.h`
(state/design), tests `test_tess.c`, `test_tessemu.c`, `test_tesscap.c` +
`tests/headless/mgl_tess_diag.h` (test-only diagnostics).

## What runs on the GPU now

**Hardware path** (triangles/quads without point_mode — Apple GPUs *do*
tessellate; verified by probe in round 10/11):

1. VS as a compute kernel (SPIRV-Cross `VERTEX_FOR_TESSELLATION`), outputs to a
   buffer in a per-draw ring block.
2. Indexed draws are fully GPU-driven (round 14): a scan kernel reduces
   min/max/restart-count with atomics in the ring, a `mgl_tess_scan_setup`
   kernel writes the VS parameter block (vfirst=0, vcount=max+1 clamped to the
   attribute capacity), and the VS is dispatched at the CPU-known capacity with
   threads past vcount returning early. `dispatchThreadgroupsWithIndirectBuffer`
   was rejected: Metal's stage-in fetch does not follow an indirect grid.
3. Gather kernel: expands the element buffer into the linear control-point
   stream (type conversion, base vertex, restart skip; a `modulo` mode indexes a
   cached gapless list for restart ranges).
4. TCS kernel (multi-patch workgroup): control points, patch outputs, and
   `MTLTriangle/QuadTessellationFactorsHalf` written directly into the ring —
   GPU to GPU.
5. `mgl_tess_blob` kernel (round 12): assembles each patch's TES patch-in
   record (levels from the half factors, members copied from the TCS patch
   record by a link-time map).
6. Render pass: post-tessellation TES (`[[patch(...)]]`) with
   `drawPatches`/`drawIndexedPatches` + `setTessellationFactorBuffer`;
   partitioning from the TES spacing, winding mirrored for the y-flip.

**Emulated path** (isolines and point_mode — Metal has no isoline domain and
cannot emit points): the same VS/TCS/blob compute chain, then the D3D11-style
GPU tessellator kernels (count → emit → indirect draw args) writing a
coordinate stream, and a plain-vertex TES variant (reading tess coord + patch
id from buffers) drawn through an indirect draw. No CPU readback on any path.

**Default levels** (no TCS): CPU-written half factor records (constants) +
control points passing through from the VS kernel.

## What CPU work remains, and why

* **No-TCS half-factor fill** — 12 bytes/patch of constants written to the ring
  before dispatch; no GPU dependency, and a kernel dispatch would cost more.
* **`mgl_tess_note_counts`** — arithmetic-only query bookkeeping
  (PRIMITIVES_GENERATED / TES invocations) for TCS-less draws; O(patches), no
  memory traffic. GPU query counters would be the only alternative.
* **Restart handling** — presence check + gapless list build, at most **once per
  element-buffer content** (cached, keyed on the gpuaudit content serial; rounds
  13/15). A restart re-forms patches, which fixed dispatch grids cannot express
  without fully indirect draws; the build reads CPU-visible memory only — no GPU
  wait.
* Link-time work (MSL rewrites, introspection, PSO compilation) — cached per
  program/descriptor.
* `MGL_TESS_CPU=1` reference path — opt-in debug only.

Zero `waitUntilCompleted` on the default path (rounds 13–15; proven by
`mgl_tess_scan_waits()` staying 0 through first draws, redraws, buffer-content
changes and restart ranges), zero CPU blob builds / CPU tessellations
(`mgl_tess_cpu_blob_builds` / `_cpu_generate_runs`), and `glFinish` was verified
to wait for all committed tess command buffers (round 16).

## Trace/diagnostic hygiene (round 17)

All `getenv` uses are once-only cached-flag initializers (`tess_trace()`,
`tTrace()`, `tCpuReference()`, `tNoFixup()`); no per-draw `getenv`. Every
`fprintf`/`NSLog` is gated by those flags or is an `MGL ERROR`/warning on a
failure path. Test-only counters and the opt-in GPU-time instrumentation live in
`tests/headless/mgl_tess_diag.h` (`mgl_tess_diag_enable()` attaches the
completion handlers; off = no per-draw cost).

## Known limits

* **Capacity clamp** (round 14): a draw whose absolute max index+1 exceeds the
  attribute capacity (massively sparse range out of a much larger vertex
  buffer) is clamped by the GPU — vertices past the capacity stay untransformed
  and those patches degrade. Measured cost of the capacity-spanning dispatch:
  GPU time 16–22 µs/frame for a 1M-thread dispatch vs 28–44 µs/frame for the
  6-thread one (the small case pays the vattrib domain's shadow-copy path
  instead) — no capping needed.
* **Half-precision tess factors**: the TCS's levels are consumed (and seen by
  the TES) in Metal's half factor format — same precision as MoltenVK.
* **PRIMITIVES_GENERATED with a TCS** is not noted (levels live on the GPU).
* **Isolines/point_mode are emulated** (no Metal support) — pixel-verified
  against the analytic tessellation and, for points, against plain GL_POINTS.
* **Pixel-edge tie-break** for points/lines goes to the lower/left pixel —
  identical to Apple's native GL (verified in round 11).
* **MGL_TESS_CPU reference path** renders incorrect pixels (crash fixed in
  round 12; geometry and counters correct) — debug-only, unfixed.

## Tests

* `test_tess.c` (468 checks) — hardware path (triangle/quad domains, four
  patches, indexed + instanced + base vertex, restart, consecutive draws, PSO
  cache stress, orientation/culling, program queries, errors, default levels),
  `TessIndexedNoRescan` (zero scan waits, cache invalidation, minv>0 pixels),
  `TessRestartCheckCached` (one presence check per content).
* `test_tessemu.c` (53 checks) — isolines/point_mode pixel-exact (incl. vs.
  plain GL_POINTS), emulated-path patch-blob tests with zero CPU blob builds /
  tessellations, default levels without a TCS.
* `test_tesscap.c` (4 checks) — glFinish-waits verification, GPU-time
  measurement of the capacity dispatch with a pixel assertion.

Full suite: 22/22 files, 0 failures.

---

# Round history (rounds 11–16; records kept for reference)

