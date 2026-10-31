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
