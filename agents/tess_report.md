# tess — round 11 report (point_mode + isolines GPU verification)

Worktree `wt/tess`, branch `gl46/tess`. Commits (one per supervisor step):

1. `f6b955c` step 2 — point_mode GPU tests verified pixel-exact against GL_POINTS
2. `fc03370`       — cache the MGL_TESS_* env flags; no getenv on the draw path
3. `829bc58` step 3 — isolines GPU test tightened to pixel-exact positions

## Step 1 — apply the round-10 test patch and report failures

`git apply --include='tests/*' .agent/round10_wip.patch` applied cleanly
(only `tests/headless/test_tessemu.c`). Result with the unmodified library:

```
PASS IsolinesGpu
FAIL QuadPointModeGpu
  [QuadPointModeGpu] test_tessemu.c:210: expected true: corners == 4
```

Pixels behind the failure (dumped with MGL_DUMP_POINTS): exactly 4 lit pixels
at **(49,49) (149,49) (49,149) (149,149)** in the 200x200 test viewport.
`IsolinesGpu` passed its (loose) assertions; a probe dump showed 3 lit rows
(49/99/149), 180 px each, x=[10..189], red→green left to right.

## Step 2 — point_mode positions: they were already correct

The quad corners are at NDC (±0.5, ±0.5) → window (50.0, 150.0) — **exact pixel
edges**. Two probes (kept in `.agent/`, outside the tree) pinned down the rule:

* `.agent/probe_apple_gl_points.m` — Apple's own (Metal-backed) OpenGL on this
  M4 Pro rasterizes points as: window 49.5→px49, 49.75→px49, **50.0→px49**,
  50.25→px50, 50.5→px50, 50.75→px50, **51.0→px50**, 51.25→px51.
* MGL matches that table **exactly**, for a plain `glDrawArrays(GL_POINTS)` and
  for the tessellated point_mode output alike.

So the hardware rule is "pixel whose centre is closest, ties → the LOWER/left
pixel", and both MGL and Apple GL agree; the test had assumed ties go to the
upper/right pixel. No source change was needed — instead the test now *proves*
correctness:

* `QuadPointModeGpu`: full-framebuffer readback; 4 points counted; each corner
  accepted within ±1 px of the projected position; and a **pixel-exact diff**
  against a plain `GL_POINTS` draw of the same 4 NDC positions
  (`both=4 tess-only=0 plain-only=0`).
* New `TrianglePointModeGpu`: a level-1 triangle patch emits exactly its 3
  control points as points, verified pixel-exact against GL_POINTS the same way
  (`both=3 tess-only=0 plain-only=0`).

## Step 3 — isolines: verified numerically

`IsolinesGpu` tightened from "3–24 lit rows + 2 colour samples" to:

* exactly 3 lines at rows 49/99/149 = window y h/4, h/2, 3h/4 ∓ tie (v=(i+0.5)/3
  → NDC y=-0.5/0/+0.5), each within ±1 px;
* full horizontal extent x=[10..189] (NDC ±0.9 → window 10..190, right end
  exclusive — correct line rasterization);
* the linear red→green gradient sampled at x=0.1w/0.5w/0.9w
  (r=240/127/13, g=15/128/242), which validates the u parametrisation of the
  8-segment lines and not just the line count.

## Source hygiene (supervisor requirement)

* `tessellation.c`: `tess_trace()` caches `MGL_TESS_DEBUG` once; the 11
  unconditional round-10 `rewrite_tes NULL at rel-line N` prints became
  `TES_REWRITE_FAIL(step)`, silent unless the flag is set.
* `MGLRenderer+Tessellation.m`: `tTrace()/tCpuReference()/tNoFixup()` cache
  `MGL_TESS_DEBUG/MGL_TESS_CPU/MGL_TESS_NO_FIXUP`; 14 `getenv()` call sites
  were on the GL_PATCHES draw path. Stale header comment ("drawPatches is
  dropped on Apple silicon") replaced with the real hardware/emulated split.
* `tests/headless/test_tess.c`: repaired a truncated `printf` in
  `TessConsecutiveDraws` (dangling argument list) that made the file fail to
  build — pre-existing damage, not from this round.
* The round-10 patch's `draw_buffers.c`/`MGLRenderer+Tessellation.m` debug
  printfs were **not** applied, as instructed.

## Test results

```
tests/headless/run.sh test_tess test_tessemu
== wt/tess/tests/headless/test_tess.c:   429 checks, 0 failures
== wt/tess/tests/headless/test_tessemu.c: 19 checks, 0 failures
```
Full suite: 22/22 files, 0 failures (`grep -c '^FAILED|^BUILD-FAIL'` → 0).
`stub_scanner.py --list EMPTY|NOOP|ASSERT` clean for both tess source files.

## Remaining gaps

None for point_mode/isolines positions. Known platform-wide characteristic
(not tess-specific): the point/line pixel-edge tie-break (exact integer window
coordinates) goes to the lower/left pixel; identical to Apple's native GL, so
no emulation is attempted.
