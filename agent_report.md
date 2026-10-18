# tess domain — round 8 report (branch gl46/tess)

Merged gl46-port (`022861f`), restored the single-drawPatches code/tests from `9b3ac72`
(`807c25c`), added the consecutive-draw regression test (`ccb053a`), completed step 3
(`175b7ed`).

## Verified green after clean rebuilds (the supervisor's procedure)

`rm -rf build/MGL build/arc build/libmgl.dylib && make lib && run.sh test_tess`:
`test_tess` **427 checks, 0 failures, 3/3 consecutive runs** (FourPatches with per-patch
`gl_PrimitiveID`, ConsecutiveDraws — 30 same-program draws, every corner pixel-checked —
PSOCacheStress, Triangle, Quad, Indexed+instanced, Orientation/culling, programiv, errors,
default levels). `test_tessemu` 4 checks 0 failures. Full suite 21 files; `test_texparam`
flakes (rc=134) intermittently under load — pre-existing, passes 3/3 standalone.

The earlier "third draw goes black after a clean rebuild" was **not reproducible on the
merged state**; the new `TessConsecutiveDraws` test pins the scenario as a permanent
regression test. (My original version of that test failed 30/30 — a sampling bug: 2 of my
3 corner samples landed outside the tessellated triangle. With correct corner samples the
30 draws are pixel-exact every iteration.)

## Step 3 (no CPU wait with a TCS) — DONE (`175b7ed`)

The hardware path (triangles/quads) now runs **entirely GPU-ordered**: the compute
command buffer signals an `MTLEvent`, the render command buffer waits on it before any
encoder; params are inline (`setBytes`/`setVertexBytes`); every scratch range is a
per-draw ring block. **Zero CPU waits.** Remaining waits, all intentional:
`MGL_TESS_CPU=1` (reference path) and the emulated (isolines/point_mode) path's patch-blob
build — the blob is CPU-built from the TCS patch records; a copy kernel would remove it
(not done; isolines/point_mode are the remaining gap).

## Steps 4–5 — not started (budget)

Step 4 (indexed draws: bind the GL element buffer to the gather kernel instead of
`tessReadIndices`' CPU read) and step 5 (isolines/point_mode pixel verification — the
GPU tessellator kernels produce verified-correct coords/indices/args but the rendered
image still needs debugging) remain for the next round.
