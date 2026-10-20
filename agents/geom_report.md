# geom domain — working handoff (round 27, checkpoint)

Session context: branch gl46/geom; everything through commit a376bfb (adjacency table
10.4, SEPARATE_ATTRIBS xfb, GS/VS stage texture arrays + pre-realize, source-transform
#define/block fixes, MAX_GEOMETRY_UNIFORM_BLOCKS capped at 14) is committed and green
(full headless suite 5344 checks / 0 failures, CTS geometry_shader 90+/136, adjacency
8/8, limits 7/9).  The uncommitted delta below is the uniform-block-ARRAY work.

## What I am fixing right now (uncommitted checkpoint)

Uniform-block instance arrays (`layout(binding=0) uniform UB { int entry; }
ub[N];`) in the GEOMETRY kernel: `array<UB,N>` in the generated MSL spans N consecutive
buffer argument slots, but the kernel pinned ONE slot per resource and the bind walk
bound one buffer per resource.  Implemented in the working tree:

* geometry.c `geom_build_pins`: per-type counters advance by the array size
  (spv_value_type), UBO span (`ub_span`) becomes the base for UC/DUB/SB flats, returns
  the total slot span; both callers (capture kernel + GS kernel) fail the LINK with a
  clear info-log message when the span exceeds MGL_GEOM_KERNEL_USER_SLOTS (22, new in
  mgl_ext_geom.h — the kernel's own SSBOs are pinned at fixed slots 22..29).
* MGLRenderer.m `mapGLBuffersToMTLBufferMap` (shared): a UB/SB resource with
  array_size>1 now appends one list entry PER ELEMENT (binding base..base+N-1, offset
  per element), so every stage's enumeration matches the expanded argument slots
  (FS/VS UBO arrays were silently broken before, same as textures were).

Results: `layout(binding=0) uniform UB { int entry; } ub[8]` all-1.0 works; CTS
limits.max_input_components/combined/max_texture_units now PASS (7/9 limits pass).

## The open bug (next step)

`tests/headless/test_gsubo.c` (NOT RUN, main() disabled): NB=14 blocks with DISTINCT
values (i+1, sum must be 105 — same shape as KHR-GL46.geometry_shader.limits.
max_uniform_blocks, which fails "Expected: 105 Extracted: 104").  The kernel's sum
reads 104: exactly ONE element contributes 0 — almost certainly element 0 (value 1) or
the last.  CTS cluster: limits.max_uniform_blocks + max_uniform_components (the latter
still Expected 524800 / Extracted 0: a default-uniform array of ~131200 vec4s — likely
DUB-size or glslang scale; separate issue).

Debug plan:
1. Re-enable RUN(GsUBOArray) in tests/headless/test_gsubo.c; make the GS emit one
   vertex per element with ub[i].c.x as g_s and read all 14 back (which element is 0?).
2. Suspects: (a) the DUB entry — mapGLBuffers puts the DUB (UNIFORM_CONSTANT) AFTER the
   UBO group, but geom_build_pins pins DUB at ub_span (=14) while the bind list has it
   at index 14 ✓ should match; check with MGL_GEOM_DUMP=/tmp/x + MGL_DUMP_MSL=1 that
   the kernel MSL really has array<UB,14> [[buffer(0)]] and DUB [[buffer(14)]];
   (b) std140 stride: MGL array<UB,14> element stride must be 16 (int padded) — check
   the MSL struct; (c) the FIRST slot colliding with something bound earlier on the
   same encoder (the prep kernel? gCtl at MSL 29...).
3. After the fix: re-enable the test, run CTS geometry_shader.limits (want 9/9),
   then the whole group + full suite, commit.

## CTS snapshot (current tree)

geometry_shader 90+ Pass / 136 total; biggest remaining clusters:
rendering.* (~30 pixel-diff failures), linking.more_ACs_in_GS_than_supported (must FAIL
link when atomic counters/blocks exceed limits), primitive_queries (3), api singles
(program_pipeline_vs_gs_capture, fb_texture_invalid_level_number, getProgramiv3).

## Rules reminder (supervisor)

Never `git stash` (shared stash list); never `git add -A` at repo root during merges
(external/ symlink); max -j4; one CTS batch at a time via agents/conformance/run.py.
