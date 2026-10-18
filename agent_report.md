# tess domain — round 9 report (branch gl46/tess)

Merged gl46-port (`a59031f` → clean, no geom-file conflicts). Commits: `3c7d726` (merge),
`0024220` (step 4), `87b214e` (cleanup).

## Step 4 — indexed patches without CPU index reads: DONE (`0024220`)

The gather path now reads the GL element buffer as a device buffer on the GPU:

- **`mgl_tess_scan` kernel**: atomically min/max's the vertex indices directly from the
  element buffer's raw bytes (ubyte/ushort/uint, +base vertex), skipping and counting
  primitive-restart indices. Runs in its own tiny command buffer; the VS grid dimension is
  sized from its 16-byte result (the only CPU wait in an indexed draw, and it's an atomic
  result, not a staging copy of the index list).
- **`mgl_tess_gather` kernel**: reads the element buffer directly (`device const uchar*`,
  type-dispatched) and expands the indices into the control-point stream with base vertex
  baked in — **no CPU index list at all** on the no-restart path.
- **Primitive restart with GL_PATCHES**: the scan detects restarts; the draw falls back to
  a compacted CPU read (`tessReadIndicesInto` drops restarts and re-sizes the patch count)
  and the plain gapless gather — restart is rare, and correct beats fast.
- New **`TessIndexedRestart`** test: a 7-index ushort list with a `0xFFFF` restart splitting
  two patches; both triangles pixel-checked.

## Step 5 — isolines/point_mode pixels: partially done (not committed separately)

The emulated path still builds the TES patch-in blob on the CPU (its `emu_cb waitUntilCompleted`)
because the plain-vertex TES reads patch data at its own layout offsets, which requires the
name-mapped copy. Making the emulated path bind the TCS patchOut buffer directly (like the
hardware path's vertex descriptor does) removes it — this is the remaining work.

## Suite

```
test_tess:    429 checks, 0 failures (11 tests: FourPatches/gl_PrimitiveID, Indexed,
               IndexedRestart, ConsecutiveDraws, PSOCacheStress, Triangle, Quad,
               Orientation/culling, programiv, errors, default levels)
test_tessemu:    4 checks, 0 failures (isolines + point_mode draw cleanly)
full suite:  20 files, 0 failures
```
