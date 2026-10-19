# tess — round 11 report (point_mode + isolines GPU verification)

> Round 12 (GPU patch-blob assembly for the emulated path) is documented at the
> bottom; this section is the round-11 record.

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

---

# tess — round 13 report (audit: remaining CPU sync points, then removal)

> Round 14 (fully GPU-driven indexed draws — no scan wait at all) is documented
> at the very bottom; this section is the round-13 record.

## Step 1 — the audit (state as of 9e0a676, before the fixes)

Greps: `waitUntilCompleted|waitUntilScheduled|waitUntilSignaled`,
`.contents`, and per-draw CPU loops over vertices/patches in
`tessellation.c` and `MGLRenderer+Tessellation.m`.

### GPU-wait sites (`waitUntilCompleted`)

| site | path | default? |
|---|---|---|
| `MGLRenderer+Tessellation.m:931` | index scan (`scb`) for indexed GL_PATCHES draws: the VS grid and the ring layout need `minv/maxv/restart-count` on the CPU | **YES — every indexed patch draw waits once** |
| `MGLRenderer+Tessellation.m:1255` | `cpu_reference \|\| debug_wait` (MGL_TESS_CPU / MGL_TESS_DEBUG) | no |
| `MGLRenderer+Tessellation.m:1322-1334` | `tTrace()` dump in the hardware `drawPatches` block | no |
| `MGLRenderer+Tessellation.m:1497` | `tTrace()` dump in the emulated render block | no |

### `.contents` reads of GPU-written results

| site | what | default? |
|---|---|---|
| `MGLRenderer+Tessellation.m:933-935` | `scanout[0..2]` — the scan kernel's min/max/restart count | **YES — indexed patch draws** (paired with :931) |
| `MGLRenderer+Tessellation.m:1326-1332` | trace dump of cp/factors | no (tTrace) |
| `MGLRenderer+Tessellation.m:1498-1502` | trace dump of tcs_patch/blob/factors/coords | no (tTrace) |
| `MGLRenderer+Tessellation.m:1550,1563` | CPU reference path: factor read-back + patch records | no (MGL_TESS_CPU / GPU-tessellator fallback) |

CPU→GPU **writes** through `.contents` (no GPU dependency, not read-backs;
listed for completeness): :917 scanbuf init, :981 widened index upload,
:1193-1195 no-TCS half factors, :1386 emulated kernel params,
:1621-1623 CPU-path uploads, :1745/:1838 attribute shadow uploads.

### Per-draw CPU loops over vertices/patches (outside MGL_TESS_CPU)

| site | what | default? |
|---|---|---|
| `MGLRenderer+Tessellation.m:943` `tessReadIndicesInto` | reads the element buffer on the CPU and loops over `count` indices (drops restarts) | **YES when the range contains primitive-restart values** |
| `MGLRenderer+Tessellation.m:980-981` | widened gapless index list: loop over `tess_idx_used × instances` | same restart case |
| `MGLRenderer+Tessellation.m:1190-1196` | no-TCS half-factor fill: writes `total_patches × 6` constants | yes, but a pure CPU→GPU constant write (no GPU dependency, 12 B/patch); a kernel dispatch would cost more — kept |
| `tessellation.c:1475` `mgl_tess_note_counts` | arithmetic-only loop over `total_patches` for query bookkeeping on TCS-less GPU draws | yes; no memory traffic, O(patches) — kept (GL needs the counts; the alternative is GPU query counters) |

Also found while auditing: the scan path allocates a fresh 16-byte
`MTLBuffer` **per indexed draw** (:914).

## Step 2 — removals

* **commit `68fb738`** — the index-scan stall, the per-draw scan-buffer allocation and
  the restart-range CPU index loop are gone for repeat draws: the scan results (and,
  for restart ranges, the gapless widened list as an `MTLBuffer`) are cached in an
  8-entry LRU in the renderer, keyed on (context, element buffer object, its
  **gpuaudit content serial** — bumped by every CPU-visible modification —, Metal
  storage pointer and size, offset, count, index type, restart value/enabled). A hit
  skips the scan kernel, the wait and the allocation; the gather kernel indexes the
  cached list with a new `modulo` parameter instead of a per-instance ring copy.
  Proven by `mgl_tess_scan_waits()` / `mgl_tess_cpu_index_reads()`:
  `TessIndexedNoRescan` draws each range three times, asserts exactly one scan per
  range (one CPU index read for the restart case), one more scan after a
  `glBufferSubData` invalidates the entry, and checks pixels every time.

  Three real bugs surfaced by the new test were fixed in the same commit:
  indexed patch draws whose indices do not start at 0 rendered the wrong vertices
  (`tessBuildVSPlan` computed the vertex-rate attribute fetch offsets from `first`
  before the scan produced the real base vertex — 10 of the test's checks fail
  against the round-12 library); the widened (primitive-restart) path never actually
  dropped the restart values (`tessReadIndicesInto` returns them; the old 'gapless'
  list still contained 0xFFFF entries and used a stale `tess_idx_used`); and the raw
  gather dispatch left `itype/minv/restart` unbound (undefined values) while the
  widened dispatch subtracted minv twice. Every kernel parameter is now set
  explicitly.

Post-fix state of the wait sites (file `MGLRenderer+Tessellation.m`):

| line | path |
|---|---|
| 1060 | the one-time scan of a *first* draw of an element range (cache miss); repeat draws skip it entirely |
| 1469 | `cpu_reference \|\| debug_wait` (MGL_TESS_CPU / MGL_TESS_DEBUG) |
| 1539 | `tTrace()` dump in the hardware `drawPatches` block |
| 1711 | `tTrace()` dump in the emulated render block |

All remaining CPU work on the default path is the two documented, justified items:
the no-TCS half-factor constant fill (12 B/patch, no GPU dependency) and
`mgl_tess_note_counts` (arithmetic-only query bookkeeping, O(patches), no memory
traffic) — plus the first-draw-of-a-range scan miss above.

---

# tess — round 14 report (indexed draws fully GPU-driven: the scan wait is gone)

Commits: `0fd3a48` (step 1, test red) and `2d06bb6` (step 2).

## Design

The vertex kernel now indexes its output **absolutely** (vertex id, `vfirst = 0`),
so the scan's min is not needed at all and the ring's vertex-output region is
sized by the attribute capacity (`tessVSCapacity`: the smallest vertex-rate
binding's `(bytes - offset) / stride` — always ≥ maxv+1 for a valid draw)
instead of the scan range. Only `maxv + 1`, clamped to that capacity, matters:

* the scan kernel (same math, atomics into the ring now) runs in its own command
  buffer, followed by a new `mgl_tess_scan_setup` kernel that writes the vertex
  kernel's parameter block `{vfirst=0, vcount, instances, baseinstance}` beside
  it — both committed before the compute buffer and consumed GPU-side through
  FIFO ordering; **never read back, never waited on**;
* the vertex kernel is dispatched at the CPU-known capacity × instances and
  every thread past the GPU-decided vcount returns before writing; its
  parameter block is bound from the ring instead of `setBytes`;
* an indirect dispatch (`dispatchThreadgroupsWithIndirectBuffer`) was
  implemented first and **rejected**: Metal's stage-in attribute fetch does not
  follow an indirect grid — every thread beyond the first fetched zeros. The
  capacity-bound direct dispatch with a GPU-written vcount guard has the same
  effect with no such limitation;
* the gather shifts by base only (`minv` unused), reading the absolute records.

The one decision still made on the CPU is whether the range contains a
primitive-restart value (patches re-form around one, which no kernel-side trick
can express without variable patch counts): `tessRangeHasRestart` scans the
element buffer's CPU-visible storage — **no GPU sync** — and restart ranges keep
the round-13 gapless-list path, which also never waits (the CPU reads the
indices directly from CPU-visible memory). The scan cache now serves only that
path, where it still saves the CPU index read on repeat draws — exactly the
"keep only if it saves work" rule.

## Test

`TessIndexedNoRescan` (strengthened in step 1, red until step 2):
`mgl_tess_scan_waits()` stays at its initial value through the first draw of a
fresh element range, three redraws, a `glBufferSubData` that changes the
contents, and a primitive-restart range — with pixel checks after every draw.

```
scan waits 0 -> 0, cpu index reads 1 -> 1
after glBufferSubData: scan waits 0 -> 0
restart: scan waits 0 -> 0, cpu index reads 1 -> 2
== wt/tess/tests/headless/test_tess.c:   455 checks, 0 failures
== wt/tess/tests/headless/test_tessemu.c: 53 checks, 0 failures
```
Full suite: 21/21 files, 0 failures. The three remaining `waitUntilCompleted`
sites are behind MGL_TESS_CPU / MGL_TESS_DEBUG / the cached trace flag only.

## Known limits

* A draw whose absolute max index +1 exceeds the attribute capacity (a
  massively sparse range drawn from a much larger vertex buffer) is clamped by
  the GPU: vertices past the capacity stay untransformed and those patches
  degrade. Sizing by capacity instead of the scan range trades that (pathological)
  case for never needing min/max on the CPU.
* The restart presence check is a CPU scan of `count` indices when primitive
  restart is enabled and the range is not cache-hot; it reads CPU-visible
  memory only (no GPU interaction) and is counted by
  `mgl_tess_cpu_index_reads()`.


---

# tess — round 12 report (emulated path: GPU patch-blob assembly)

Commits (one per step), after merging `gl46-port` (66de6c9):

1. `ed1196b` step 1 — tests + CPU-work counters (red)
2. `6346028` step 2 — the GPU blob kernel, five real bugs fixed

## Step 1 — the test and the counters

`mgl_tess_cpu_blob_builds()` / `mgl_tess_cpu_generate_runs()` are exported from
libmgl and incremented by the CPU blob build / CPU tessellator. Two tests in
`test_tessemu.c` force the emulated path (isolines never take the hardware
tessellator):

* `EmuPatchDataGpu` — a TCS with a per-patch varying (`layout(location=1) patch
  out vec4 p_flag`) and an isolines TES reading it: the value reaches the TES
  only through the per-patch data blob. Checks the 3 lines at v=(i+0.5)/3 in
  the patch colour, full extent, then a **two-patch** variant where each patch
  carries its own colour and level count (per-patch record indexing).
* `EmuDefaultLevelsGpu` — TES without TCS + `glPatchParameterfv` defaults
  {3,8}: 3 lines, per-vertex colour interpolation.

Both check pixels *and* that neither counter moved across the draw. Against the
pre-round-12 library both failed, and for real reasons:

* has-TCS emulated draws stalled (`[emu_cb waitUntilCompleted]`), built the blob
  on the CPU and memcpy'd it to **ring offset 0** (off_patch_data was never
  allocated with a TCS), clobbering the control-point records the TES read: the
  patch-var program's blob was 64 B/patch and overwrote cp1's gl_Position —
  lines ended at window x=125 instead of 189. Measured: blob builds 3→4/draw.
* no-TCS emulated draws never wrote the half tessellation-factor records the
  GPU tessellator reads (only the hardware path filled them) — it tessellated
  from zeros: one line at v=0.5 instead of 3. Full CPU blob + tessellator also
  still ran per draw (builds 4→6, runs 0→1).

## Step 2 — the GPU replacement

New compute kernel `mgl_tess_blob` (`MGLTessKernels.h`), one thread per patch,
dispatched in the VS/TCS compute encoder right after the TCS kernel: zero the
patch's TES patch-in record, fill the level members from the half factor record
(half→float; triangles pack the inside level into `gl_TessLevelOuter.w`), and
copy the TCS-written members from the TCS patch-output record, using a
per-program map (`mgl_tess_build_blob_map`) computed once at link time with the
same name/size matching as the CPU builder. Ordering is FIFO command-buffer
order plus the existing event wait — **no CPU read-back, no wait** on the
emulated or the TCS-less hardware path.

Additional fixes the new tests flushed out:

* the plain-TES rewrite read patch data at a fixed record offset — every patch
  saw patch 0's values (`mgl_tess_patch + member_off`, no patch id). Now
  indexed by the patch id the tessellator stores in the coordinate stream;
  verified by the two-patch case (left half red-dominant, right half
  blue-dominant, per-patch level counts).
* the emulated and CPU paths bound the control-point buffer at ring offset 0
  (the VS output) instead of `cp_source_off` — only pass-through TCSs worked.
* the no-TCS GPU draws ran the full CPU tessellator per draw just to feed
  `glEndQuery` bookkeeping; replaced by `mgl_tess_note_counts` (same arithmetic
  as the GPU tessellator's `mgl_patch_counts`), so queries keep working with no
  CPU tessellation.
* the CPU reference path (MGL_TESS_CPU) committed `compute_cb` twice and
  aborted with "commit an already committed command buffer" (it had been
  unrunnable); also its factor read-back used a 6-half stride for triangles
  (8-byte records). Both fixed; the CPU path completes again.

Kept on the CPU only behind `MGL_TESS_CPU` (or the GPU-tessellator fallback):
`mgl_tess_build_patch_blob` + `mgl_tess_generate`. A trace facility
(MGL_TESS_DEBUG, cached flag) can dump the TCS patch records / blob / factors /
first coordinates after the compute stages — it does wait once, and only when
tracing.

## Test results (clean rebuild, `rm -rf build && make -j8 lib`)

```
== wt/tess/tests/headless/test_tess.c:   429 checks, 0 failures
== wt/tess/tests/headless/test_tessemu.c: 53 checks, 0 failures
```
Full `tests/headless/run.sh`: 21/21 files, 0 failures (no FAILED/BUILD-FAIL).
`stub_scanner.py --list EMPTY|NOOP|ASSERT` clean for `tessellation.c` and
`MGLRenderer+Tessellation.m`.

## Known gaps

* The MGL_TESS_CPU reference path renders incorrect pixels (colour/geometry
  scrambled) although it no longer crashes; its geometry and counters are
  correct. It is a debug reference only — the GPU paths are the contract. Left
  as a follow-up.
* TCS-written tessellation levels are consumed by the GPU tessellator in half
  precision (Metal's factor-buffer format), so a TES reading gl_TessLevel*
  sees half-rounded values — same precision as the hardware path (MoltenVK
  behaves identically).
* Query counters for TCS-level-dependent GPU draws (PRIMITIVES_GENERATED with
  a TCS) are not noted — the levels live on the GPU; unchanged from round 11.

