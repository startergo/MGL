READY 26ab48f (all r45 code; the report commit on top is the branch tip)

# tess domain — round 45 report (supervisor-assigned: glCopyImageSubData completeness fix, dEQP-GL45-ES31 copy_image 7680->0 fails)

Task from the 14:35 note: the official-equivalent CTS (opengl-cts-4.6.8.1, gl45-es31-main)
failed ALL of dEQP-GL45-ES31.functional.copy_image.non_compressed.* with
"glCopyImageSubData failed.: glGetError() returned GL_INVALID_OPERATION" on every
valid copy involving a texture, while KHR-GL46.copy_image passed.  This round finds
the bug, fixes it, and finishes the verification runs the previous run left undone.

## Root cause and fix (381b73e)
`copyTexComplete()` in MGL/src/textures.c (the texdata domain's file — surgical edit,
see "Shared-file edits") computed the top of the mipmap chain a copy source/dest must
have as `TEXTURE_MAX_LEVEL` (default 1000).  A texture that defines fewer than 1000
levels was therefore treated as incomplete whenever its MIN_FILTER was a mip filter,
and glCopyImageSubData returned GL_INVALID_OPERATION.  dEQP's copy_image tests render
their source images through a sampler with a mip min filter and never restore
GL_NEAREST, so every texture case failed; renderbuffer-only pairs (155 cases) passed.

Per GL 4.6 8.17 ("Sampler Completeness") the chain a mip min filter needs ends at
`p = min(max_level, q)` with `q = base + floor(log2(max(w,h,d) of the level-base
image))`.  The fix computes exactly that (`fullChainLevels` already existed for
texture completeness elsewhere in the file) and additionally requires the level-base
image itself to be defined.  A texture whose base..p levels all exist is now
copyable even with the default MAX_LEVEL 1000; a texture actually missing a level in
base..p stays incomplete (KHR-GL46 copy_image.incomplete_tex: 4x4, level 0 only,
needs 0..2, still fails with GL_INVALID_OPERATION).

## Verification (this round's runs)
* **dEQP-GL45-ES31.functional.copy_image.\* (full group, new glcts at
  ~/MGL-cts-new, -j4, nice'd, 256x256, seed -1): 7835 Pass / 0 Fail / 0 NotSupported.**
  Baseline r965d88f (same case set): **7680 Fail / 155 Pass** — every previously
  failing case now passes.  Per viewclass: 8-bit 361/361, 16-bit 1089/1089,
  24-bit 441/441, 32-bit 4708/4708, 48-bit 144/144, 64-bit 752/752, 96-bit 144/144,
  128-bit 196/196.  Repro: .agent/cts45_ci.sh (log .agent/cts45_ci.log, per-run
  qpa in .agent/cts45_ci/).
* **KHR-GL46.copy_image.\*: 324/324 Pass, 0 Fail, 0 NotSupported** (same glcts,
  .agent/cts45_khr.log) — no regression, including the incomplete_tex cases.
* Full headless suite: **144 test files, 0 failures** (139 check-summaries, all
  "0 failures"; no BUILD-FAIL/crash), including the new regression test
  `copy_image_sub_data_mip_complete` in test_texdata.c (26ab48f): 65x63 texture,
  levels 0..6, NEAREST_MIPMAP_NEAREST, MAX_LEVEL default -> both copies succeed and
  read back pixel-exact; 4x4 texture with only level 0 -> still GL_INVALID_OPERATION.
* stub_scanner EMPTY/NOOP/ASSERT: clean for MGL/src/tessellation.c and
  MGL/src/textures.c.  `make lib` clean (the one -Wpointer-bool-conversion warning at
  textures.c:3053 is pre-existing on the 965d88f base, untouched by this round).

## Pre-READY ab_bench (.agent/abbench_r45.sh, ab_patches, MGL_AB_CPU=1, min-of-7,
alternating, background CTS load 5-8)
| bench (ms/1000)      | A = 965d88f | C = r45 |
|----------------------|-------------|---------|
| drawArrays-tris      | 0.433       | 0.372   |
| patches-noTCS        | 89.7        | 85.4    |
| patches-TCS-TES      | 131.7       | 94.1    |
| patches-TCS-TES-64   | 89.4        | 92.3    |
CPU-time parity within the run's load noise (as expected: the change only touches a
texture-completeness predicate on the copy path, no draw-path code).

## Shared-file edits
* **MGL/src/textures.c** (+15/-2, commit 381b73e): `copyTexComplete()` only — the
  mip-chain top is now min(MAX_LEVEL, 8.17 q) instead of MAX_LEVEL, plus the
  level-base-defined check.  Nothing else in the file changed; the texdata domain's
  own tests all still pass (test_texdata 257 checks, 0 failures in the suite run).
* **tests/headless/test_texdata.c** (+49, commit 26ab48f): one new TEST, no edits to
  existing tests.

## Notes
* GL_KHR_shader_subgroup left unexposed, per the user's 15:44 decision — no work
  started, no API/state touched.
* Tess code itself is untouched this round (r44 ring work is merged and unchanged);
  the round was entirely the supervisor-assigned copy_image fix plus verification.

---

# (round 44 history below)
# tess domain — round 44 report (ring retire latency + draw-size robustness)

Builds clean (`make -j8 lib`, no new warnings in my files), full headless suite
**139 test files, 0 failures**; stub scanner clean for
MGLRenderer+Tessellation.m / tessellation.c / tess_spv_lower.c; CTS
KHR-GL46.tessellation_shader **58/58 Pass** on the merged tree.

## What this round fixed

### 1. Ring slots waited on Metal's completion-handler jitter (per-draw CPU, the dominant cost of every GPU-contended GL_PATCHES loop)
A slot retired only when Metal fired the command buffer's completion handler —
0.5-2 ms **after the GPU finished** the buffer, several ms under load.  With
draws encoding in ~100 us, that latency is worth 5-20 in-flight draws: tight
GL_PATCHES loops went dry with 20+ slots sitting on *finished* buffers and
burned 48-150 us/draw sleeping in tessRingAcquire (phase counters, load 5-8).

**Fix (all in my file):** every retire armed through a handler is now ALSO
recorded on an in-flight list polled from the GL thread (every acquire that
needs a slot), so a slot frees as soon as the GPU has *actually* finished the
buffer.  The handler stays armed as the exact wake for workloads that stop
drawing.  Measured interleaved a/c/a/c (same load, matched 200-draw windows):
**ring phase 78-150 -> 0.36 us/draw**, draw TOTAL 120-209 -> 40-71 us/draw.

A retire can now arrive twice, so the slot state word packs
`generation | RETIRING | BUSY`: a retire CASes BUSY->BUSY|RETIRING at the
generation it was armed for, mutates exclusively in that state (pos reset,
big-block recycle), then publishes FREE with the generation bumped.  The twin
(handler after poll, poll after handler — and the pre-existing parked-after-
escape case) is an exact no-op; `mgl_tess_ring_dup_retires()` counts them.
The fast path also now claims **before** reading pos/cap (a second GL context
running a whole claim->retire cycle in between could otherwise hand out a
nil'd big block through stale capacity — same class as the r43b hole).

### 2. Draw-size overflow guards (robustness)
`patches * instances` and `count * instances` were GLsizei x GLsizei — signed
overflow (UB) above 2^31, and the wrapped value sized the ring block: the
kernels would dispatch garbage grids and Metal validation would abort (a
silent memory-corruption class bug, worse than a failed draw).  Both products
are computed in 64 bit now; a draw whose totals exceed 4M patches / 2^31
gathered control points is dropped before any GPU work (single log, no GL
error — running out of drawable resources is not a GL error).  The ring's grow
path also refuses blocks above `device.maxBufferLength`.

### 3. Keep-alive cadence knob
`MGL_TESS_KAMAX` (1..4) caps the AGX keep-alive commit cadence — an A/B knob
for the poll-based retirement (measured indistinguishable between 4 and 2 under
load; the r43 hysteresis stands).

## Tests added (test_tess.c: 689 -> 807 checks, 0 failures)
* **TessRingGenerationHammer** — 600 mixed-shape draws (TCS+TES / 64-patch /
  noTCS-default-levels), 36 pixel-exact frames on a prime finish stride, ring
  must never wedge (worst acquire poll < 50 ms, vs the 2 s escape).
* **TessHugePatchCountDropped** — 2^32 total patches (int32-overflow product)
  is dropped cleanly (frame untouched, no GL error, single log) and the ring
  stays healthy for everything after it.

## Gates (supervisor protocol)
* Full headless suite after the f8be669 merge: **139 files, 0 failures**.
* CTS per-group: `KHR-GL46.tessellation_shader*` → **58/58 Pass** (0 Fail, 0
  NotSupported) on the merged tree.
* ab_bench (ab_patches, MGL_AB_CPU=1, min-of-7, alternating) vs
  **baseline-f8be669**: drawArrays 0.71/0.81, noTCS 134.9/137.8,
  TCS+TES 141.2/152.1, 64-patch 138.0/137.8 ms/1000 — CPU-time parity within
  the load noise of the run (load 6-8; r43 recorded single-run swings larger
  than these deltas).  The r44 win is wall-clock latency, which CPU-time
  accounting deliberately excludes: the interleaved phase windows above show
  where the sleeping went.
* Locked glmark2 (lockf, 8 rounds, alternating, same 3 scenes as r43):
  P(f8be669) median 2951, C(r44) median 2709 — **treat as contaminated**: C's
  two lowest rounds (1588, 1813) coincide with other agents' GPU bursts
  (machine load 5-8 through the window), and r44 executes **no code** on these
  scenes (no GL_PATCHES draws in ideas/jellyfish/gl41-geometry; the ring is
  entered only from mgl_tess_draw_patches).  The merged-tree deltas that could
  matter here (geom r82/r83, compat s38) are identical on both sides.

## Shared-file edits
None this round (all work in MGL/src/MGLRenderer+Tessellation.m,
tests/headless/test_tess.c, tests/headless/mgl_tess_diag.h).  The f8be669
merge took gl46-port's side everywhere except MGLRenderer+Tessellation.m,
where my r44 ring rework supersedes (upstream f492c47 is byte-identical to my
c83f507, so the resolution loses nothing).

## Gaps / follow-ups
* The ~20-40 us/draw Metal render-encoder rebuild remains structural for the
  shared-CB compute/render interleaving; the clean fix is batching the tess
  compute encoders on a second command buffer chained by events so the render
  encoder can stay open across draws — noted as future work, not correctness.
* GPU-bound submit loops still backpressure by design (ring = pipeline depth,
  24 draws); the poll only removes *handler* latency, not GPU execution time.
* Still recommended for the queries domain (their file):
  extQueriesRenderPassDescriptor allocates per render-pass build even with no
  query targets active; the tess path builds one render pass per draw.

---

# (round 43 history below)

# tess domain — round 43 report (robustness + per-draw CPU in the GL_PATCHES path / shared tess command buffer)

READY c83f507 (gl46-port d11d184 [r43 upstream] merged; r43b on top).  Builds
clean (`make -j8 lib`, no new warnings in my files), full headless suite
**137 test files, 0 failures**; stub scanner clean for
MGLRenderer+Tessellation.m / tessellation.c / tess_spv_lower.c.

## What this round found and fixed

### 0. The r42 "TCS+TES" benchmark was measuring the wrong program (bench bug)
`.agent/ab_patches.c` built its TCS program with a VS `out vec2 vpos[3]` against
a TCS `in vec2 vpos[]`.  That GLSL is **illegal** (a VS array output makes the
TCS input an array-of-arrays; element types `vec2[3]` vs `vec2` genuinely
mismatch — MGL's link error is correct, verified with `.agent/linktest.c`), so
the link failed, `glUseProgram` errored, and every "TCS+TES" number r42 quoted
was actually the TES-only program.  The bench now checks GL_LINK_STATUS (and
uses conventional `out vec2 vpos;` / `in vec2 vpos[]`).  Consequence: the real
TCS shape had never been measured before — and it was the one that exhausted
the ring.
uses conventional `out vec2 vpos;` / `in vec2 vpos[]`).  Consequence: the real
TCS shape had never been measured before — and it was the one that exhausted
the ring.

### 1. Ring exhaustion — the dominant per-draw CPU/wall cost (fixed)
Diagnosis via a new opt-in CPU-phase breakdown (`mgl_tess_phase_*`, armed by
`mgl_tess_phase_reset`, zero cost when off): in a tight GL_PATCHES loop the
**ring** phase (tessRingAcquire) cost ~330 us/draw of sleeping — 327 spinning
acquires per 1000-draw bench.  A slot retires only when its command buffer
**completes**, Metal fires completion handlers with 0.5-2 ms jitter, and with
draws encoding in ~100 us the old 8-slot ring went dry in every burst (the
r42 keep-alive committed every 2nd draw = 2 slots per in-flight buffer; the
window holds far more).

Fixes (all in MGLRenderer+Tessellation.m):
* **depth 8 -> 24** (slots still allocate lazily and only grow to the largest
  block handed out; an idle ring costs 24 pointers);
* **keep-alive cadence driven by ring occupancy** with hysteresis: 4/2/1 draws
  at >2/3 free / moderate / <=1/6 free (a plain low/high threshold oscillated
  around the pressure point and cost ~6 us/draw on the noTCS shape); an actual
  exhaustion still forces per-draw commits for 64 draws (r42);
* **memory bound for big blocks**: retired blocks >4 MB (capture/emulated
  scratch, tens of MB) are freed beyond two warm ones — a burst of 40 x 5.3 MB
  emulated draws pins 2 slots (`big_slots=2`), not 40 (TessRingBigBlockRecycle);
* **efficient dry-ring wait**: after ~1 ms of polling, block on a COMMITTED
  parked buffer's completion (exact wake) instead of more usleeps; verified
  not to be a CPU cost itself (wait vs MGL_TESS_NOWAIT identical).

Result on the quiet machine (load <0.5): spinning acquires 327 -> 19 per
1000-draw bench, ring phase 338 -> 0.25 us/draw in healthy windows.

### 2. Per-draw CPU on the GL_PATCHES path itself
* **VS-kernel PSO lookup**: the NSString key (program pointer + the whole VAO
  layout, ~7 appendFormat per attribute) plus the dictionary hash cost ~5 us
  per draw.  The stage-input descriptor a VS PSO wraps is fully determined by
  the freshly built plan + function object, so a direct-mapped
  (context, program, function, 64-bit plan signature) entry fronts the cache.
  `setup` phase 14.3 -> 3.3 us/draw.
* **noTCS path**: default tessellation levels now written straight into the
  ring block (no per-draw calloc of the float factor array), and
  **mgl_tess_note_counts is gone from the GPU paths** — it ran the full CPU
  tessellator count pass per patch per draw, but its only consumer is a count
  query, and an active tess count query forces the CPU reference path which
  notes its own counts (locked by TessCountQueriesExact: levels-1 patch
  PRIMITIVES_GENERATED=1/INVOCATIONS=3, culled patch 0/0).
* Remaining and **structural**: the render-encoder rebuild per tess draw
  (~30 us, ~all of it Metal's own renderCommandEncoderWithDescriptor) — the
  compute encoders must be ordered before their own drawPatches, so a render
  encoder per tess draw is inherent to the shared-command-buffer design.
  Batched encoders across draws would break GL's state-change interleaving
  semantics; documented, not attempted.

### 3. A/B (CPU-time, min of 7, alternating order; `ab/abbench_r43.sh`)
Baseline = ~/MGL-cts/baseline-288e724 (r42 upstream) when r43 was developed,
re-run against ~/MGL-cts/baseline-a3c493b (compat s37) after the final merge;
same corrected bench both sides.  Quiet-machine runs (load <0.5):

```
[drawArrays-tris submit]     A=0.479 ms/1000  C=0.544   (no tess: unchanged)
[patches-noTCS submit]       A=128.7 ms/1000  C=76.8    (-40%)
[patches-TCS-TES submit]     A=143.7 ms/1000  C=136.6   (-5%)
[patches-TCS-TES-64]         A=119.6 ms/1000  C=138.4   (GPU-bound; see below)
```

Honest caveat: the GPU-bound shapes (TCS+TES 64-patch: 368 us GPU per command
buffer vs ~100 us CPU) have high run-to-run variance — **the baseline binary
itself swings 84 <-> 142 ms/1000 between runs** with background load (other
agents), and Metal's render-encoder creation — ~30 us of every tess draw in
every tree — swings 31 <-> 58 us *within one binary*.  Under heavy external
load (5-8) both trees are statistically indistinguishable (84-154); the
noTCS quiet delta and the deterministic per-phase numbers are the signal:

```
noTCS phase breakdown (us/draw, my code only, same binary 3 runs):
setup 3.3-9.6 (was 14.3 pre-r43)   notcs 0.17 (was 0.52)
ring 0.2-2.8 in healthy windows (was 338)   post 1.8-3.3 (was 4.8)
spinning acquires per 1000-draw bench: 19 (was 327)
```

**Locked glmark2** (8 rounds each, alternating; `.agent/ab_glmark_r43.sh`,
overall score medians): A(baseline-288e724) = **2623**, C(r43) = **2621** —
unchanged, as expected (no shared draw-path files touched this round).

### 4. Verification
* full `tests/headless/run.sh`: 136 files, **0 failures** (test_tess 689/689,
  test_tessemu 117/117)
* KHR-GL46 per-group (this tree): **tessellation_shader 58/58 Pass**,
  separable_programs_tf 2/2, transform_feedback_overflow_query_ARB 13/13
* `python3 agents/stub_scanner.py` EMPTY/NOOP/ASSERT: clean for my files

## Files
* MGL/src/MGLRenderer+Tessellation.m — everything above
* tests/headless/test_tess.c — TessCountQueriesExact, TessRingBigBlockRecycle
* tests/headless/mgl_tess_diag.h — mgl_tess_ring_spins/_freed_big/_big_slots,
  mgl_tess_phase_reset/us/cnt declarations
* ab/abbench_r43.sh, ab/ab_patches.c — committed A/B bench (corrected shaders)
* agents/tess_report.md — this report

**Shared-file edits: none this round** (only my category file + tests + ab/;
the two gl46-port merges touched MGL/src/MGLRenderer+Tessellation.m as
"ours" — the same r42 content I authored upstream, plus my r43 work).

### 5. r43b — airtight claim/publish protocol for the ring (merged upstream r43, then hardened)
Auditing the r43 recycle for this round found a real hazard in it: the
big-block free wrote `tRing=nil`/`tRingCap=0` from the completion-handler
thread **after** publishing the slot free (`tRingBusy=false`), so a GL-thread
claim in that window could observe the stale capacity and hand a nil'd buffer
to the draw.  Fixed by protocol, not luck (all in my file):
* **release mutates, then publishes**: every mutation (pos reset, big-block
  recycle) happens before the release store;
* **tRingBusy is an atomic_bool** — acquire loads in the acquire scan, spin,
  and diagnostics; release store in the releaser — so a claim that observes a
  free slot sees its final `tRing`/`tRingCap`/`tRingPos` by construction;
* **claim-then-mutate** in acquire (fast path and growth path claim with an
  atomic exchange before writing anything), so a slot is only ever mutated by
  its owner.
Perf-neutral-or-better (A/B vs baseline-d11d184, the direct predecessor:
noTCS 124->102, TCS 138->132, 64-patch 135->133, drawArrays 0.41->0.37
ms/1000, min-of-7 at load <0.7); full suite and tessellation_shader 58/58
re-verified after.

## Gaps / follow-ups
* The ~30 us/draw Metal render-encoder rebuild is structural for the
  shared-CB compute/render interleaving; a safe batching would need
  render-state-change tracking across draws (same rule the geom agent's stages
  use) — noted as future work, not correctness.
* GPU-bound submit loops still hit ring backpressure by design (the GPU is the
  bottleneck); the acquire now blocks efficiently instead of polling.
* Recommendation for the queries domain (their file, not mine):
  extQueriesRenderPassDescriptor allocates an MTLRenderPassDescriptor +
  MGLQPass per render-pass build even with no query targets active; the tess
  path builds one render pass per draw, so a zero-active early-out there would
  shave every tess draw.
* The per-draw bind walks (tessBindStageUniforms/Textures/VSAttributes) are
  already flat array walks; the remaining cost is Metal's own
  setBuffer/setTexture and the per-draw compute/render encoder creation —
  nothing left to win in my file without encoder batching.
