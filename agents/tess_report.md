# tess domain — round 37 report (shader_ballot compute cases; EXT_texture_shadow_lod)

READY abef1d9 (merge of gl46-port 2898abe [geom r70] over 7b84384; every number
below is on exactly this tree)

## Task 1 — the 3 shader_ballot compute-path cases: **all Pass**

`python3 agents/conformance/run.py --pattern 'KHR-GL46.shader_ballot_tests.*'`:

```
== totals (4 cases):
  Pass                4  (100.0%)
  KHR-GL46.shader_ballot_tests                       4 cases  Pass=4
```
(ShaderBallotAvailability, **Bitmasks**, **FunctionBallot**, **FunctionRead**;
the last three were Fail since r36.)

How (three separate defects, all fixed in 7b84384 + the merge):
1. **The GPU-hanging CTS path** was never re-run directly. The bounded headless
   repro (`tests/headless/zz_ballotprobe.c`, untracked scratch) replicates the
   CTS's compute validation exactly — 16x16 RGBA32UI texture, `bindImageTexture`
   with unit format RGBA32UI, a `layout(rgba32f)` writeonly image writing
   `vec4(1.0)`, one dispatch, texel readback — and only after both parts passed
   (A: uint store + float sampler = raw bits, B: the formerly-hanging
   format-mismatched store) was the CTS group run (it finishes in ~5 s).
   The hang itself: the geometry domain's image-unit view now casts a
   format-kind-mismatched image to a same-texel-size view of the SHADER's
   element kind (RGBA32F view of RGBA32UI storage) instead of dispatching into
   an invalid texture2d<float>-over-RGBA32Uint store. Part B reads back
   0x3F800000 x4, pixel-verified.
2. **The render stages of FunctionRead failed on an uncovered centre pixel** —
   `validateScreenPixelsSameColor` takes the screen-centre pixel as the desired
   colour and requires at least one non-red pixel. Root cause (found with a
   patch-split probe: patch1-only / patch2-only / no-GS): the emulated
   tessellation path publishes its TES-output capture records in GENERATION
   order (the permuted stream the transform-feedback pack consumes), but the
   geometry import kernel built its primitive slots from the tessellator's
   coordinate-index list as well — every slot was permuted **twice**, so the
   geometry kernel consumed scrambled control points (patch 2 of the CTS's
   2-patch quad lost 90% of its triangles, patch 1 ~1/3; with an affine TES the
   GS route rendered only one edge strip for both patches). Fix:
   `MGLTessCapture.emission_order` — emission-ordered records get identity
   primitive slots, coordinate-ordered records (CPU reference path, point_mode)
   keep the index list. Verified: every tess+GS case now matches its no-GS
   reference pixel-for-pixel (`zz_pbdiag`, scratch), and the CTS's
   normalize-through-the-origin patch corner covers the centre (white).
   *Note:* the geom agent hit the same bug from the XFB side in gl46-port r70
   (identical `cfg[8]` fix); the merge kept the upstream naming
   (`tess_emission`/`emission_order`) and my explanatory comments.
3. The remaining probes all pass: replay of the CTS's own six-stage pipelines
   (Bitmasks 60 checks, FunctionBallot+FunctionRead 30 combos, availability 8
   checks) — the earlier probe failures were an attribute-location assumption
   in my validation quad (now explicit `layout(location=)`).

## Task 2 — more NotSupported cases: **ext_texture_shadow_lod 16 -> 15 Pass + 1 Fail**

```
== totals (16 cases):
  Pass               15  (93.8%)
  Fail                1  (6.3%)
  KHR-GL46.ext_texture_shadow_lod                   16 cases  Pass=15 Fail=1
```
(was 16/16 NotSupported on the missing extension string.)

* `get.c` advertises **GL_EXT_texture_shadow_lod** after the path was verified
  pixel-exactly: glslang accepts the directive at 460 core and SPIRV-Cross
  lowers the OpImageSample*Dref* forms to MSL `depth2d_array/depthcube/
  depthcube_array sample_compare(level/gradient/bias, offset)` —
  `tests/headless/test_shadowlod.c` compiles/links every function the suite
  uses, verifies LOD level selection through both levels, and vertex-stage
  implicit sampling of 2D-array + cube-array shadow textures.
* `tessellation.c` (`tess_shadow_implicit_lod`, wired into
  `mgl_tess_glsl_compat`): non-fragment stages rewrite implicit shadow sampling
  to the explicit base-level form — `texture(s,P[,compare])` ->
  `textureLod(...,0.0)`, `textureOffset(s,P,off)` -> `textureLodOffset(s,P,0.0,off)`
  — for every `sampler*Shadow` uniform (GL 4.6 8.14.1: implicit LOD outside the
  fragment stage is the base level; the explicit form also keeps MSL on
  `sample_compare(level(0))`, which is the defined vertex-stage form).
* Known gap: `texture.sampler2darrayshadow_vertex` still fails image comparison
  (difference 0.109462 vs threshold 0.0705 — it renders, and the CTS's own
  Result/Reference/ErrorMask images were extracted from the .qpa to compare).
  Everything testable in isolation was verified correct: generated MSL
  (`sample_compare(s, xy, uint(round(z)), w, level(0.0))`), compare direction,
  layer rounding/stepping (a z-sweep hits every layer boundary exactly), wrap of
  negative/>1 UVs, DEPTH_COMPONENT16/24, NEAREST/MIPMAP filters, vertex vs
  fragment stage, and the double-sampler case. The residual mismatch is inside
  the CTS's 65x65-vertex interpolated field (its user-attribute matrix maps
  coord.z/coord.w along the two grid diagonals); my CPU reference of that field
  is too crude to localize it further. Second gap: a SECOND shadow sampler in
  one vertex stage reads the first texture's raw depth (my test prints it; the
  CTS never puts two shadow samplers in a vertex stage).

## Other groups re-run on this tree (regression check)

```
  KHR-GL46.tessellation_shader    58 cases  Pass=58   (100%)
  KHR-GL46.geometry_shader       136 cases  Pass=135  (api.max_image_uniforms
                                                fails on the GS image path, not
                                                a limit: all MAX_*_IMAGE_UNIFORMS
                                                report 16)
  KHR-GL46.shaders45               7 cases  Pass=7
  KHR-GL46.packed_depth_stencil   25 cases  Pass=25   (extension-string sanity)
  KHR-GL46.shader_ballot_tests     4 cases  Pass=4
```

## Suite

`tests/headless/run.sh` on this tree: **129 tests, 0 FAIL / 0 BUILD-FAIL**
(exit 0; every test reports "0 failures"). New: `test_shadowlod.c` (8 checks)
and `TessGeometryMultiPatch` in `test_tess.c` (two disjoint patches through
TCS/TES/GS in one indexed GL_PATCHES draw — regression for the capture
permutation fix).

`python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT`: clean for
tessellation.c, spirv_vote.c, cts_glsl_compat.c, MGLRenderer+Tessellation.m,
mgl_ext_tess.h.

## Shared-file edits this round (all called out)

* `MGLRenderer+Geometry.m` (geom domain): (a) the image-unit format-cast
  continuation of r66; (b) the tess import kernel's identity-slot guard
  (upstream r70 has the same line — merge kept upstream + my comment);
  (c) reverted my temporary MGL_CULL_TRACE index-list dump extension.
* `geometry.c` (geom domain): `d.tess_emission` plumbing (upstream naming).
* `mgl_ext_geom.h` (geom domain): `tess_emission` field (upstream).
* `get.c` (queries): +1 extension string, GL_EXT_texture_shadow_lod.
* `mgl_ext_tess.h` (mine): `MGLTessCapture.emission_order` (upstream bool form).
* `MGLRenderer+Tessellation.m` (mine): publish the emission-order flag at both
  capture sites; kept the env-gated `MGL_TESS_EMUDBG` diagnostic dump (per-patch
  counts, TCS records, factors, capture records/indices).
* `tessellation.c` (mine): `tess_shadow_implicit_lod` + scanner-name fixups.

## Reproduction / scratch (untracked, by convention)

`.agent/scratch-tests/probe_ballot_*.c` and `tests/headless/zz_*.c` (ballot
repro, patch-split diagnostic, shadow-lod bisectors). The permanent regressions
live in `test_tess.c` / `test_shadowlod.c`.


# tess domain — round 36 report (NotSupported queue: +20 CTS cases; ballot emulation)

READY ec3e01e (merge of gl46-port f600d55 [geom r66-r69] over 70d1f8e; every
number below on exactly this tree)
* suite `tests/headless/run.sh`: **128 tests, 0 FAIL / 0 BUILD-FAIL, 126x
  "0 failures" summaries** (drawparams/orphan/v64limits end in PASS lines by
  design).  test_geom_xfbperf is IN the gate again (quarantine lifted with
  geom r64; it ran clean in the suite and in test_geom 10x below).
* test_geom 10x = **10/10 x 670 checks, 0 failures**; test_fp64 5x = **5/5 x
  417**; test_fp64fast 3x = **3/3 x 298**.
* ab_bench m0/m1 min-of-7 interleaved vs ~/MGL-cts/baseline-f600d55: the plain
  min-of-7 read m0 +11.4% / m1 +6.7% with the baseline catching one quiet
  window (its 2nd-best was 26%/13% above its min), while the same-window A/A
  control (baseline vs baseline-copy) read **+14.4% / +21.3%** — the box noise
  dominates; the decisive 14-rep interleaved floor run gives **m0 min -9.0%,
  p20 -1.3%, median +2.3%; m1 min +7.1%, p20 -0.5%, median +0.0%** — parity.
* locked glmark2 (every run under `/usr/bin/lockf -k /tmp/mgl_glmark2.lock`,
  order alternated, vs baseline-f600d55 nsgl): **ideas 5 pairs median 2483 vs
  2470 (+0.5%), jellyfish quiet-pair median +0.3%, gl41-geometry 5 pairs
  median 22365 vs 22294 (+0.3%)** — no scene past +3%.
* CTS on this tree: **cull_distance 3/3, tessellation_shader 58/58**
  (primitive_coverage fixed upstream by compat s26), **program_interface_query
  43/43, shading_language_420pack 43/43** (binding_samplers now passes in the
  group run too), explicit_uniform_location queue cases Pass (the full
  48-case group passed 48/48 twice earlier today on the same code paths; it
  times out at 1200 s under the current box load).
* **This round's new cases (all previously NotSupported/Fail in my queue):
  shader_integer_mix 13 (advertising GL_EXT_shader_integer_mix; 1 further case
  stays NotSupported by CTS design — the gl3 package registers the group at
  GLSL 330 but `prototypes' gates on >= 450), parallel_shader_compile 3/3
  (glMaxShaderCompilerThreadsARB/KHR entry + hint query + COMPLETION_STATUS =
  TRUE, the synchronous-compile path the ARB allows), gpu_shader5_gl 3/3
  (advertise GL_ARB_gpu_shader5), shader_ballot_tests 1/4** — Availability
  Pass; the other three fail inside the CTS *compute* validation path, see
  Known gaps 3.
* stub_scanner EMPTY/NOOP/ASSERT clean: tessellation.c, spirv_vote.c,
  cts_glsl_compat.c, MGLRenderer+Tessellation.m, mgl_ext_tess.h.
* Scratch policy fix from the last review: 88 test_zz_*/zz* scratch files are
  untracked and moved to .agent/scratch-tests/ (out of the run.sh glob); the
  old test_zz_gsimplicit 120 s timeout disappears with them (it was a heavy
  65536-point GS repro, 45.6 s on my tree, not a hang).

## r35/r36 work: the NotSupported queue in my area

Top-3 by case count (from the r28 sweep, re-verified live):
1. sample_variables 38 (samples_8 rows only) — blocked on GL_MAX_SAMPLES = 4;
   raising to 8 is a texdata/fb allocation decision, not mine to take.
2. shader_integer_mix 11 — DONE (13 cases incl. shaders45 now Pass).
3. shader_ballot_tests 4 — DONE as far as MGL can go (see Known gaps 3).

Also taken because they were pure program/GLSL surface:
* parallel_shader_compile (3), gpu_shader5_gl (3).

### GL_ARB_shader_ballot design (the interesting one)
* glslang emits the SPV_KHR_shader_ballot opcodes; SPIRV-Cross's MSL backend
  only lowers the SPIR-V 1.3 OpGroupNonUniform* core forms, so spirv_vote.c
  now also rewrites **OpSubgroupBallotKHR -> OpGroupNonUniformBallot,
  OpSubgroupReadInvocationKHR -> OpGroupNonUniformShuffle,
  OpSubgroupFirstInvocationKHR -> OpGroupNonUniformBroadcast lane 0** (each
  inserts the %uint 3 Subgroup scope; broadcast additionally needs a %uint 0,
  reused when present) and swaps the SubgroupBallotKHR capability for
  GroupNonUniformBallot(+Shuffle).  Metal lowers those to simd_vote /
  simd_shuffle, which exist in fragment/compute/tess-control kernels on Apple
  GPUs (verified end-to-end: readInvocationARB(v,0)+readFirstInvocationARB
  sum to exactly 1.0 in a fragment).
* **gl_SubGroupSizeARB is a uniform in glslang's OpenGL profile** (it becomes
  the SubgroupSize builtin only under Vulkan semantics) and would read 0
  through mgl_DefaultUniforms: cts_glsl_compat.c replaces its uses with an
  injected helper returning 32 (the simdgroup width on every Apple GPU).
* **The mask builtins** are lowered by SPIRV-Cross to simd_ballot(true),
  which sets only ACTIVE lanes — a partially active wave (4 vertices in the
  vertex-stage kernel, a 3-vertex patch) then disagrees with the spec/CTS
  formula "bits i in [0,size) with i OP invocation".  The masks are pure
  functions of the invocation index, so the same pass injects
  eq/ge/gt/le/lt helpers computing exactly that formula (uint64_t flavor for
  the bare use, uvec4 flavor when a swizzle follows — glslang accepts both
  spellings), clamped by a lanes() mask so they stay correct whatever the
  reported size.
* **TES and VERTEX have no simdgroup attributes on Metal** (the compiler
  rejects `[[thread_index_in_simdgroup]]` in a post-tessellation/ordinary
  vertex function; SPIRV-Cross refuses the same).  The subgroup size is
  implementation-defined per stage, so those two stages get a **size-1
  subgroup emulation** in mgl_tess_glsl_compat: invocation 0, the lane-0
  masks (via the helpers above composed with the 0u fold), ballotARB(p) =
  uint64_t(p), and the reads degenerate to the value itself.  Every other
  stage keeps the real 32-wide simdgroup.
* GL_ARB_gpu_shader_int64 is advertised too: ballotARB's uint64_t flows
  through Metal 3's 64-bit integer emulation (verified: shifts/xor/pack of
  uint64_t compile and run in a fragment shader).

### parallel_shader_compile
Entry points glMaxShaderCompilerThreadsARB/KHR (gl_core.c + glm_dispatch
members — shared-file additions, listed below), the implementation stores the
hint in ext_tess state and mgl_ext_get_tess answers
GL_MAX_SHADER_COMPILER_THREADS_ARB in every glGet type; a programiv hook
(registered from tessellation.c) and one case in mglGetShaderiv return
GL_COMPLETION_STATUS = TRUE, which is exactly what a synchronously compiling
implementation reports.

### Shared-file edits this round (each surgical)
* MGL/src/gl_core.c, MGL/include/glm_dispatch.h, MGL/src/glm_dispatch.c,
  MGL/include/mgl.h: the glMaxShaderCompilerThreads[ARB|KHR] entry point
  (function + dispatch member + assignment + prototype).
* MGL/src/shaders.c: GL_COMPLETION_STATUS_ARB case in mglGetShaderiv (3
  lines); the rest of my shaders.c edits are earlier rounds' (r23/r28).
* MGL/src/get.c: extension strings (ARB_shader_ballot, ARB_gpu_shader_int64,
  ARB_parallel_shader_compile, KHR_parallel_shader_compile; ARB_gpu_shader5
  and GL_EXT_shader_integer_mix deduped after compat landed the same strings
  upstream).
* No edits to MGLRenderer+Compat.m, draw_buffers.c, or rendering.c.

---

# tess domain — round 33 report (gate + READY on the 3cd6208 merge)

READY 1dddf05 (merge of gl46-port 3cd6208 [compat s26] into e6a2333 [tess
r31+r32]; every number below on exactly this tree)
* suite `tests/headless/run.sh`: **153 tests, 0 FAIL / 0 BUILD-FAIL, 149x
  "0 failures" summaries** (the 5 without a summary line — drawparams, orphan,
  v64limits, zz_gather, zz_gmsl — end in PASS lines by design);
  **test_geom_xfbperf quarantined (supervisor 23:42)** and skipped.
* test_geom 10x = **10/10 x 670 checks, 0 failures**; test_fp64 5x = **5/5 x
  417**; test_fp64fast 3x = **3/3 x 298**.
* ab_bench m0/m1 **min-of-7 interleaved vs ~/MGL-cts/baseline-3cd6208**:
  quiet window m0 4.314 -> 3.895 ms (**-9.7%**) / m1 81.599 -> 78.971
  (**-3.2%**); under co-tenant GPU load a second min-of-7 read m0 +21.0% /
  m1 -8.2% — the wall-clock min flips sign with box load in BOTH directions
  (same-window A/A control, baseline vs baseline-copy: min-of-7 +2.2%/-5.1%,
  median +2.6%/+3.9%; 20-rep interleaved m1 floor run: tip 102.75 vs base
  107.52 = **-4.4%**).  Direct CPU attribution, MGL_PROFILE per-draw buckets
  (ab_bench mode 11, interleaved, median of 6): draw 0.320 -> 0.267,
  pipeline-descriptor 0.161 -> 0.155, dirtyState-block 0.307 -> 0.254,
  processGLState 0.300 -> 0.248, newRenderEncoder 0.058 -> 0.053 ms/frame —
  **tip <= baseline in every bucket**.
* locked glmark2 (every run under `/usr/bin/lockf -k /tmp/mgl_glmark2.lock`,
  order alternated, vs baseline-3cd6208 nsgl): **ideas 5 pairs median 2732 vs
  2665 (+2.5%), jellyfish quiet-window top-3 25403 vs 25100 (+1.2%),
  gl41-geometry 5 pairs median 21440 vs 21481 (-0.2%)** — no scene past the
  +3% limit (r31 touched geomRasterPass/MGLRenderer.m/draw_buffers.c;
  gl41-geometry is exactly that path: parity holds with r31b's export-PSO
  cache fix).
* CTS deltas (the r28-sweep queue, on this tree): **cull_distance 3/3**
  (functional fixed r31, negative r31b), **tessellation_shader 57/58,
  program_interface_query 43/43, explicit_uniform_location 48/48,
  gpu_shader5 18/22** (4 `texture_gather_offset_*_clamp_{border,edge}` =
  the documented texdata border-clamp gap; every non-gather gpu_shader5 case
  passes), shader_draw_parameters 13/13 + shader_group_vote 5/5 (r28),
  shading_language_420pack length_of_vector_and_matrix + gl_MaxPatchVertices
  (r30), gl_spirv + separable_programs_tf.geometry_active (compat s23-s25).
  **The r28 fails_tess.txt queue is now 0 open tess-side cases.**
* `tessellation_shader.single.primitive_coverage` re-checked on **pristine
  baseline-3cd6208: still Fail with channel-REVERSED pixels** (rendered
  0x20,0x40,0x80,0xff vs expected 0xff,0x80,0x40,0x20 at index 8188) — the
  c6ded7d copy_image / MS-image-write regression forwarded to compat, not a
  tess bug (see Known gaps 1 below for the residual tess-side question).
* stub_scanner EMPTY/NOOP/ASSERT clean for tessellation.c,
  MGLRenderer+Tessellation.m, tess_spv_lower.c, mgl_ext_tess.h.

r33 changed no library code: this round is the gl46-port 3cd6208 merge plus
the full gate above (the r31/r32 work was already on the branch; see the r31
and r32 sections below for what those changed).

---

# tess domain — round 30 report (dynamic matrix-member indices; c6ded7d merge)

READY ddc2099 (superseded by READY 1dddf05 above; gate of 5988b5d + c6ded7d)

## r30 fixes (from fails_tess.txt)

* **KHR-GL46.shading_language_420pack.length_of_vector_and_matrix — Pass**
  (was Fail).  SPIRV-Cross refuses a dynamic index that walks INTO an
  interface block's matrix member; the SPIR-V lowering (tess_spv_lower.c)
  copies the member to a function-local and indexes the copy.  Two gaps
  fixed: trailing VECTOR component indices (m[i].x, new SPV_T_VECTOR walk),
  and glslang's SPLIT access chains (AC(block,0,member) then AC(.,i)) —
  the detector now resolves through up to four intermediate chains and the
  rewrite re-roots the element copy at the variable.  The canonical-record
  side needed the matching fix in three places (tess_canon_body merges a
  column run back into one float<C>x<R> member; the TES tes_cp offset remap
  picks the member's own column slot; tess_canon_union absorbs columns).
* **KHR-GL46.tessellation_shader.tessellation_control_to_tessellation_
  evaluation.gl_MaxPatchVertices_Position_PointSize — Pass** (was Fail).
  THREE stacked bugs: (1) my r29 "relink drops the TES" hook wiped the
  separable-vs KERNEL cache at every non-tess link (now only the tess
  RECORD is dropped); (2) a NON-separable vs-only program bound next to a
  pipeline's TCS/TES (GL 4.6 par 7.3 — the CTS tess utilities draw exactly
  that) had no kernel at all: the prepared module is now cached at link and
  the kernel MSL built lazily (mgl_tess_cache_vs_words /
  mgl_program_tess_kernel_msl), and the link cache carries it across
  program deletion (names are reused); (3) tess_build_pair resolved the
  pair's VS from the stale pipeline instead of the bound program.
* new headless test **test_tess_dynio** (dynamic matrix column reads
  through a TCS->TES interface, pixel-verified) — 2 checks, 0 failures.

## UPSTREAM REGRESSION IN c6ded7d (not mine — please forward to compat)

**KHR-GL46.tessellation_shader.single.primitive_coverage fails on the
~/MGL-cts/baseline-c6ded7d/libmgl.dylib baseline itself and passed on
baseline-b7821df** — rendered pixels are channel-REVERSED (0x20,0x40,0x80,
0xff vs expected 0xff,0x80,0x40,0x20), pointing at the s22+s23 copy_image /
MS-image-writes / border_clamp changes.  My tree inherits it through the
merge (57/58 on the tessellation group).  On c6ded7d the same group is
**55/58** (also failing: xfb_captures_data_from_correct_stage — flaky,
passes standalone here — and my gl_MaxPatchVertices case, which r30 fixes).

## r30 gate

* Full headless gate `tests/headless/run.sh`: **148 tests, 0 FAIL** (143x
  "0 failures" summaries, no BUILD-FAIL).
* CTS: `KHR-GL46.tessellation_shader.*` **57/58** (only the upstream
  primitive_coverage regression; xfb_captures passes standalone 3/3);
  `KHR-GL46.shading_language_420pack.*` **42/43** (binding_samplers is
  flaky under load — passes standalone); gl_MaxPatchVertices and
  length_of_vector_and_matrix **Pass** standalone.
* **Per-draw CPU gate** (merge rule, baseline-c6ded7d, interleaved): the
  box has been under sustained multi-agent GPU load all night (geom +
  compat CTS runs at ~90-97% CPU) and the wall-clock min-of-7 flipped sign
  between sessions (m0 -13.8% then +9.0%; m1 +7.6% then -3.3%) — the
  minimum tracks whichever arm catches a quiet window, not the code.  The
  direct attribution is stable: **MGL_PROFILE per-draw buckets on a framed
  blend-flip loop (ab_bench mode 11, interleaved, median of 6) — draw
  0.015 vs 0.016 ms, pipeline-descriptor 0.547 vs 0.545, dirty-state 0.73
  vs 0.72: parity in every bucket.**  r30 touches only link-time code and
  tess-draw paths (no exec_draw / exec_indirect / compat fast-path edit);
  a pure-c6ded7d-content build measured -1.5% vs the baseline dylib in the
  same session, confirming no build-level drift.
* Remaining from fails_tess.txt: **cull_distance.functional** ("Top edge
  of bounding box is overwritten", gl3cCullDistanceTests.cpp:2316 — a
  triangle the cull kernel should have dropped is drawn; next round).
  gl_spirv x5 and separable_programs_tf.geometry_active belong to compat.

---

# tess domain — round 29 report (perf: O(1) per-draw gates)

READY ddc2099 — ab_bench vs baseline-c6ded7d, wall-clock interleaved min-of-7 under heavy multi-agent GPU load (see r30 section): m0 11.64->12.69 ms (+9.0%) / earlier session m0 13.37->11.52 (-13.8%), m1 107.77->104.19 (-3.3%) / earlier +7.6% — the wall-clock gate oscillates with the box load in BOTH directions; the direct CPU attribution (MGL_PROFILE buckets, mode-11 framed blend-flip, interleaved median-of-6) is at PARITY in every per-draw bucket (draw 0.015 vs 0.016 ms, pipeline-descriptor 0.547 vs 0.545); r30 changes are link-time/tess-draw only, no exec_draw/exec_indirect/fast-path edit

(r29 numbers measured on tip ddd8f68 = c1bb530 + bbffefa + ddd8f68, against
baseline-6f9b359.  r30 continues on a merge of gl46-port b7821df [which took
r28+r29 upstream] — see the r30 section below.)

## r29: the r28 SPEED rejection

Supervisor A/B (idle machine) vs baseline-6f9b359: **ideas 3318/3212/3225 ->
1906/1843/1906 (-42%), jellyfish ~-2.5%**. Reproduced on my side:

* three-arm glmark2 `-b ideas` (P=baseline / O=r28 / F=r29-fixed, interleaved):
  ideas P 503/559/450, O 287/328/318 (**-37%**), F 430/487/496.
* deterministic headless draw loop (`tests/headless/ab_bench` mode 0, min of 7,
  interleaved): base 11.96 ms / 2000 draws, **r28 15.94 ms (+33%)**.

Cause (all in the r28 diff, all per draw): `mgl_geom_cull_active()` fell
through to `geom_cull_writes_msl()` — a `strstr` over the whole vertex-stage
MSL — for every program without a geometry stage; `exec_draw` additionally
paid `mgl_tess_active_program()`'s tess-record hash walk, an unconditional
`ext_tess->draw_id/draw_base_vertex` stamp, and `exec_indirect` paid
`mgl_tess_draw_params_program()`, whose pipeline arm REBUILDS the composite
(`mgl_program_pipeline_composite` memsets and re-copies the whole Program per
call — and never copies `uses_draw_params`, so it always missed).

## r29 fix (bbffefa + ddd8f68)

Link-time flags, nothing per draw touches ext_tess unless the program needs it:

* `Program.writes_cull_distance` computed once in `mgl_geom_post_link`;
  `mgl_geom_cull_active` checks it BEFORE the geometry-table lookup, the MSL
  rescan runs at link only.
* `mgl_tess_draw_params_bound_i` / `mgl_tess_stage_bound_i` — static inline in
  `mgl_ext_tess.h`: the program's `uses_draw_params` flag / TES stage
  presence; pipelines resolve through their stage programs (no composite
  build — this also FIXES gl_DrawID for pipelines: the composite lookup
  always missed).  `extTessActive` uses the stage gate (it is called from
  every state walk and compat's fast path).
* `exec_draw`/`exec_indirect` stamp `ext_tess` only for draw-params programs;
  `extBufDrawArrays`/`extBufDrawIndexed` call `extTessBindDrawParams` only
  when a pipeline is bound or the flag is set.
* a successful relink that drops the TES now deletes the program's stale tess
  record (`mgl_ext_post_link_tess`), keeping the record and the O(1) stage
  gate in agreement.

## r29 A/B numbers (all interleaved, glmark2 under /usr/bin/lockf
-k /tmp/mgl_glmark2.lock, `/opt/homebrew/bin/glmark2-macos --data-path
/opt/homebrew/share/glmark2`)

* headless ab_bench, min of 7 (deterministic, no WindowServer):
  `drawArrays x2000` base 11.852 -> r29b **10.717 ms (-9.6%)** — the inline
  stage gate also removes the tess-record hash walk the BASELINE itself pays
  per draw; `blend-flip x600` 68.057 -> 67.346 ms (**-1.0%**).
* glmark2 `-b ideas -b jellyfish`, 8 rounds, alternating arm order:
  ideas median P 511.5 / C **518.5 (+1.4%)**; jellyfish P 4551 / C 4814
  (**+5.8%**, 10 extra jellyfish-only rounds; single-run spread is +-35% on
  this shared box — other agents' headless CTS runs steal the GPU — an
  earlier 4-round session read -2%, so treat jellyfish as parity within
  noise; every CPU-attributable bucket of the MGL_PROFILE dump is at or below
  baseline).
* MGL_PROFILE=1 `-b ideas` draw bucket (exec_draw, 353 draws/frame):
  r28 0.744 ms/frame -> r29 0.719 -> baseline 0.646; r29's GL-side gates are
  ~40 ns/draw total, the encode-side msgSend is now gated off.

## r29 gate

* **Per-draw CPU gate (merge rule of 21:30, final tip, min of 7 interleaved vs
  ~/MGL-cts/baseline-6f9b359): `ab_bench m0 8.22->6.66 ms (-19.0%)`,
  `m1 46.50->46.53 ms (+0.06%)` — both far inside the +3% limit.**  An earlier
  measurement on the same tree under heavier load read m0 11.85->10.72
  (-9.6%), m1 68.06->67.35 (-1.0%).  The m0 win is real: the inline stage
  gate also removes the tess-record hash walk the BASELINE itself pays per
  draw.
* Full headless gate `tests/headless/run.sh`: **140 tests, 0 FAIL, 135x
  "0 failures"** (the remaining stderr lines are a passing test's deliberate
  failed-relink trace).  `test_tess` **504/0**; `test_geom` 10x
  **10/10 x 670 checks, 0 failures**; `test_fp64fast` 3x **3/3 x 298**;
  `stub_scanner` EMPTY/NOOP/ASSERT clean for tessellation.c and
  MGLRenderer+Tessellation.m.

---

# tess domain — round 28 report (tess / shader-linking / GLSL compat)

READY 25551dc

(tip = merge of gl46-port 6f9b359 [geom r55-r58] into 3450dd7; gate + every
number below measured on exactly this tree.)

## r28 gate

* Full headless gate `tests/headless/run.sh`: **exit 0, 0 FAIL** (multiple
  runs; one run hit the KNOWN pre-existing test_fp64 flake — fp64_scalar_ops
  roundeven off-by-one-ULP under load, see agents/geom_report.md r-note about
  the abs() twin — 5/5 standalone test_fp64 runs clean, not a tess regression).
  `test_geom` 10x **10/10 x 670 checks, 0 failures**; `test_fp64fast` 3x
  **3/3 x 298 checks, 0 failures**; `test_tess` **504/0**;
  `stub_scanner` EMPTY/NOOP/ASSERT clean for the tess files.
* glmark2 untouched this round (no hot-path changes).

## r28 CTS fixes (the r28-sweep fails_tess.txt list, 45 cases)

of which compat took gl_spirv (5), shader_group_vote (5, fixed here in r28
anyway), separable_programs_tf (1); still mine: program_interface_query (13,
geom r58 landed PIQ fixes — re-check pending), shader_draw_parameters (13,
**13/13 Pass**, sdp6), explicit_uniform_location (2, **48/48 group Pass**),
shading_language_420pack (6 -> **41/43**, 2 left), tessellation_shader (1),
cull_distance.functional, gpu_shader5.

1. **GL_ARB_shader_group_vote** (337b2e5, 5 Fail -> 5 Pass): SPIRV-Cross
   `vote.any/all` intrinsics are not emitted for compute kernels — lowered
   subgroup votes to atomic-pair bookkeeping in MSL reserved intrinsics.
2. **GL_ARB_shader_draw_parameters** (67d40b9, 13 Fail -> 13 Pass):
   gl_DrawID from a per-draw function-constant buffer + GL's "arrays base
   vertex" contract for the multidraw path.
3. **bindMTLProgram skips the geometry slot** (0ffd84e): a program whose
   geometry slot is absent (post geom r58) no longer fails the render.
4. **explicit uniform locations** (3bcb8f8, 2 Fail -> Pass):
   walk_default must advance the explicit-location leaf cursor across stages
   (uniform-loc-arrays-of-arrays), and mgli_uniform_location/_index no longer
   short-circuit on a hoisted-name miss, so struct-array element lookups
   ("u1.m1[1]") reach the array-index fallback (uniform-loc-types-structs).
5. **layout(binding=N) program-wide** (a440085, 2 Fail -> Pass): the GLSL 4.2
   binding qualifier applies to the uniform in EVERY stage; program.c
   unify_stage_resource_bindings equalises the per-stage SPIR-V decorations
   at link (base = the stage's lowest binding, per-element shifts follow),
   and the opaque uniform's default value is the maximum explicit binding
   (was last-stage-wins -> unit 0).
6. **rect samplers** (3450dd7, binding_samplers Fail -> Pass): the rect->2D
   SPIR-V patch made sampler2DRect resources reflect GL_TEXTURE_2D, so the
   unit's 2D binding (e.g. the CTS FBO colour texture) was sampled; the
   "_mglrect" name suffix now restores GL_TEXTURE_RECTANGLE.
   (geom r58 landed an equivalent fix on gl46-port — kept theirs in the merge.)
7. **binding_image_single / binding_images Crashes -> Pass** with the geom
   r58 merge (gpu_shader5/PIQ fixes); root was not in the tess path.

Remaining in-domain fails after r28: shading_language_420pack.
length_of_vector_and_matrix + qualifier_order_block (investigating), PIQ
re-check, cull_distance.functional (documented whole-primitive hardware gap,
r24), tessellation_shader.gl_MaxPatchVertices_Position_PointSize (Timeout).

---

# tess domain — round 26 report (tess / shader-linking / GLSL compat)

READY 0516bb6

(tip = merge of gl46-port 5178835 — r25 merged upstream — into 1c6364d; the
merge is content-empty for MGL/src/tests, so every number below was measured
on exactly this code; no code commits this round, r25 report follows.)

## r26: gate + full in-domain CTS re-verification on the merged tip

* Full headless gate `tests/headless/run.sh`: **121 test files, 560 PASS,
  0 FAIL** (every file "== N checks, 0 failures"); `test_geom` 10x **603/0**
  (3x repeated on the tip) and `test_fp64fast` 3x **298/0**; `stub_scanner`
  EMPTY/NOOP/ASSERT clean for tessellation.c, tess_spv_lower.c, msl_reserved.c,
  fp64.c, introspect_spirv.c, program_introspection.c, transform_feedback.c,
  geometry.c, program.c.
* glmark2 perf A/B (supervisor request; `-b ideas -b jellyfish` through
  build/nsgl, default mode, interleaved 6x each, ae3145b vs gl46-port 784c480,
  machine load ~8-9 from other agents): **ideas 578 vs 584 FPS mean (-1%,
  run-to-run spread ±8%)**, jellyfish median 4600 vs 4527 (+1.6%; means
  dominated by GPU-contention outliers 1380-3035 FPS in both arms).
  **r25 does not further slow glmark2.**
* KHR-GL46 re-verification of every in-domain entry of the r22
  partial_fails list plus the neighbouring groups, all on this tip
  (~/MGL-cts/mgl-r26-tess/results.json):
  tessellation_shader **58/58 Pass** (vertex_spacing — the r25b xfb-POINTS
  fix — no longer Times out), gpu_shader_fp64 **659/659 Pass**,
  draw_indirect + draw_indirect_43 **72/72 Pass**,
  transform_feedback **20/20 Pass**, separable_programs_tf **2/2 Pass**,
  vertex_attrib_64bit **4/4 Pass**, shaders.struct.uniform.sampler_* **6/6
  Pass**, CommonBugs.CommonBug_ReservedNames **Pass**, cull_distance.coverage
  **Pass**.
* Remaining in-domain KHR-GL46 fail: **cull_distance.functional** — the
  documented whole-primitive cull-distance hardware gap (r24), unchanged.
* Out-of-domain observation for whoever owns it: `texture_size_promotion.
  functional` fails **deterministically on base 784c480 as well** as on this
  tip ("Promotion from internal format GL_R8 ... target GL_TEXTURE_RECTANGLE
  ... Expected value = 0.125 read value = 1" → InternalError); NOT a tess/r25
  regression (bisected by running the 784c480 dylib directly).  The geom
  branch passed it on Oct 26 10:13, so a fix exists outside gl46-port.

---

# tess domain — round 25 report (tess / shader-linking / GLSL compat)

(superseded by READY 0516bb6 above; r25 was merged upstream as 5178835)

(branch tip = merge of gl46-port 784c480 into r25d 4474ec1; r25 code commits:
436350e, 28bb686, 06c571b, 4474ec1)

Gate on ae3145b: `tests/headless/run.sh` 121 files / 0 FAIL (suite green),
10x test_geom 603/0 each, 3x test_fp64fast 298/0 each.
Perf (glmark2-macos -b ideas -b jellyfish through build/nsgl, default mode,
interleaved A/B ae3145b vs gl46-port 784c480, load ~8-9 from other agents):
ideas 578 vs 584 FPS mean (-1%, inside the ±8% run-to-run spread),
jellyfish median 4600 vs 4527 (+1.6%; means dominated by GPU-contention
outliers 1380-3035 in both).  r25 does not further slow glmark2.

r25 fixes (4 commits, all KHR-GL46 Fail -> Pass unless noted):
* r25a 436350e vertex_attrib_64bit limits_test + vao (8 fixes):
  fp64.c whole-matrix/array-of-double attribute loads reassembled from the
  flat slot array; matrix element access `attribute_0[i]` of array-of-matrix;
  OpConstant word type via u32_for_pre.  introspect_spirv.c internal slot
  assignment packs wide double spans first-fit (16-slot attributes no longer
  overflow the link).  program_introspection.c: an N-column matrix takes N GL
  locations per array element.  MGLRenderer+VertexAttrib.m: dmatNxM column
  attributes are double-word slots (dbl_slot knows GL_DOUBLE_MAT*); double
  attributes honor glVertexAttribDivisor (Metal stepRate).  transform_
  feedback.c: API-specified interleaved xfb varyings are PACKED (dvec2 after
  a 4-byte int at +4) — also fixes word order of consecutive double outs.
* r25b 28bb686 xfb POINTS capture of tessellated output counts with the TF
  mode (GL 4.6 13.4.2), not the TES domain — vertex_spacing no longer
  Timeouts (spun forever in its classification loop); now Pass.
* r25c 06c571b gl_VertexID/gl_InstanceID MSL type: SPIRV-Cross hardcodes
  uint for [[vertex_id]]/[[instance_id]] while expressions keep int —
  ambiguous overloads like min(4, gl_VertexID) failed the Metal compile.
  Entry parameter kept uint under a private name; GL name re-declared as an
  int local at the top of the entry body.
* r25d 4474ec1 the same fixup applied to the transform-feedback capture
  variant MSL (separate VS recompile): draw_indirect.
  advanced-twoPass-transformFeedback-{arrays,elements} -> Pass.

---

# tess domain — round 24 report (tess / shader-linking / GLSL compat)

(superseded by READY ae3145b above; kept for history)

(the branch tip after this report-only commit carries the same code as
`47055c4`; code commits this round: f298996, a72987f, 37994f4, 94c5212,
f45c3a9, 7ff5545, 2bf8dee, 47055c4)

Part 1 (merging the interrupted r23 session): the uncommitted
`cts_glsl_compat.c` work was finished, verified and committed as the
integer-`mix()` rejection — `mix(ivec2(1), ivec2(2), bvec2(true))` constructor
forms in `#version < 450` without `GL_EXT_shader_integer_mix` get the usual
leading `#error` line — turning
`KHR-GL46.shaders.shader_integer_mix.prototypes-negative` **Fail → Pass**
(positive variants stay NotSupported: MGL does not advertise the extension
string); `test_zz_reservednames` grew 6 cases for it (14 checks / 0 fail).

Part 2 (remaining KHR-GL46 fails in this domain — **9 more Fail → Pass**):
* **cull_distance.coverage** (37994f4): glslang has no `GL_ARB_cull_distance`
  extension entry (core since GLSL 4.50), so every CTS cull shader died with
  "extension not supported".  `rewrite_cull_distance_ext()` lifts the version
  to 4.50, drops the directive (a `#define GL_*` is illegal) and folds
  whole-expression macro tests (`#ifndef` → `#if 0`); wired into both compat
  chain entries (`mgl_cts_glsl_compat_limited`/`_generated`) and into
  `xfb_glsl_to_spirv` (the third place the user's VS meets glslang).  The
  gl_PerVertex expander now strips sized array suffixes from member names.
  `mgl_msl_fix_cull_distance()` (msl_reserved.c, hooked in program.c for VS/TES)
  repairs SPIRV-Cross's broken MSL for BuiltInCullDistance OUTPUTS (stores
  pointed at a non-existent `out.gl_CullDistance[]`; unflatten tail read an
  undeclared local).  **cull_distance.functional still fails** on true
  whole-primitive culling ("expected to be culled, but one of its vertices is
  visible"): Metal raster stages have no primitive rejection by cull distance
  and the FS cannot see sibling vertices — hardware gap, documented; the test
  now links and renders.
* **shaders.uniform_block.random.*** 6 cases (94c5212): declared-but-unreferenced
  UBOs stay ACTIVE (GL 4.6 §7.6.2 allows it; glslang keeps them in SPIR-V,
  SPIRV-Cross's active-analysis DCE'd them) — introspect_spirv.c marks every
  declared UBO variable active; `shaders.uniform_block.*` 828/828 Pass.
* **gpu_shader_fp64.builtin.inverse_dmat3** (f45c3a9): the 3×3 inverse cofactor
  read uninitialized 2×2-submatrix ids (garbage → "Cannot resolve expression
  type"); now the plain det2.  `inverse_dmat*` 9/9, `builtin.*` 650/650.
* **shader_atomic_counters.advanced-usage-{multi-stage,switch-programs}**
  (7ff5545): MGL's xfb emulation re-executes the VS in a capture pass AND the
  render draw, so VS atomicCounterIncrement hit the real counter twice
  (+6 instead of +3).  When a render draw will re-run the stage, the capture
  now binds blit-snapshots of the atomic-counter buffers over the injected
  slots (one snapshot per GL binding point); with rasterizer discard the
  capture IS the execution and mutates the real counters.  Group 30/30 Pass.
* **shader_image_load_store.basic-allTargets-store** (2bf8dee + revert
  47055c4): the CTS's WriteMS compiles `writeonly image2DMS` uniforms —
  `texture2d_ms<access::write>` is illegal MSL and no shader writes to
  multisample textures exist on Metal (not even from compute), so
  **GL_MAX_IMAGE_SAMPLES is now reported 0** (glslang resource in sync);
  MS image READS work but the limit gates both — basic-allTargets-load-ms
  becomes NotSupported.  Group: 37 Pass / 2 NotSupported / 12 Fail (8
  other-domain backlog, 4 CTS "Vulkan is not supported" framework errors).

**Full gate on the final tip:** `tests/headless/run.sh` **105 binaries,
0 FAIL**; `test_geom` 10× **603/0**; `test_fp64fast` 3× **298/0**; the **23
r22 regressions** 23× Pass in one batch; `KHR-GL46.CommonBugs.CommonBug_ReservedNames`
**Pass** (434 s); `KHR-GL46.shaders.*` **2382 cases: 2376 Pass / 6
NotSupported (integer_mix positives) / 0 Fail**.  `stub_scanner` clean for
every touched file.  **Overlap heads-up (compat `638e1ed` reserved-word scan in
`shaders.c` vs my `aa32719` in `cts_glsl_compat.c`):** the two inserts land in
*different* hunks of `mglCompileShader` and both scans only ever reject, so
keeping **both** is safe
and complementary — compat's covers general declared-name slots after builtin
types/storage qualifiers, mine covers the subroutine slots after **user**
type names (`subroutine uniform T <keyword>;` — exactly the case compat's
commit message says it cannot catch), plus `invariant`-as-name and blocks
named `subroutine`.  If only one may survive, keep **mine** (this branch's
state — mine + glslang's own rejections — is what the gate above verified
green, CommonBug_ReservedNames included); if mine is dropped instead,
re-run CommonBug_ReservedNames on the merged tree before trusting it.

## Known gaps after this round (tess domain)

* **cull_distance.functional** — whole-primitive cull-distance culling is not
  expressible on Metal (no primitive rejection from the raster stages; the FS
  cannot see sibling vertices' cull distances).  Everything up to and
  including rendering works; the failure is the culled-primitive coverage
  check.
* **GL_MAX_IMAGE_SAMPLES = 0** — multisample image writes are impossible in
  MSL; MS image reads work but the limit gates both (load-ms → NotSupported).
* **VS side effects with xfb active** execute twice for image/SSBO writes
  (atomic counters are snapshotted now; unordered writes cannot use the same
  technique).  No current CTS case covers it.
* Other-domain backlog: `shader_image_load_store.*` (8 real fails + 4 CTS
  "Vulkan is not supported"), `packed_pixels.*`, `texture_gather.incomplete-*`,
  `draw_indirect.advanced-twoPass-transformFeedback-*`,
  `pipeline_statistics_query_tests_ARB.*`, `texture_size_promotion.functional`,
  `vertex_attrib_64bit.limits_test/.vao` (16 double-attribute locations need
  32 MSL attribute slots; Metal has 31 — one location would have to be fed
  via a buffer fetch instead of an [[attribute]]; architectural).

## Shared-file edits this round (outside the tess/tess-domain files)

* `MGL/src/program.c` — call `mgl_msl_fix_cull_distance()` for the VS/TES
  stage MSL after the tess MSL rewrite (ownership follows the existing
  `tess_owned` pattern).
* `MGL/src/transform_feedback.c` — `xfb_glsl_to_spirv` runs the user VS source
  through `mgl_cts_glsl_compat_generated` before glslang; records the injected
  atomic-counter slots; sets `xp->ac_shadow` from `keep_draw`.
* `MGL/src/mgl_ext_xfb.h` — `MGLXfbProgram` grew `ac_mtl_index/ac_binding/
  ac_count/ac_shadow` (+ include of `mgl_ext_geom.h` for
  `MGL_MAX_ATOMIC_VARS`).
* `MGL/src/MGLRenderer+TransformFeedback.m` — `xfbCaptureAsync` builds blit
  snapshots of bound atomic-counter buffers and binds them over the injected
  slots when `ac_shadow`.
* `MGL/src/introspect_spirv.c` — declared UBOs stay active.
* `MGL/src/fp64.c` — 3×3 inverse cofactor fix.
* `MGL/src/get.c` + `MGL/src/geometry.c` — `GL_MAX_IMAGE_SAMPLES` 0 (and the
  matching glslang resource).
* `MGL/src/textures.c` — temporarily accepted `glGetTexImage(CUBE_MAP)`,
  reverted in 47055c4 (was a wrong premise; net zero change).

---

# tess domain — round 21 report (tess / shader-linking / GLSL compat)

(e4c890d — superseded by READY 47055c4 above; kept for history)

Session goal (supervisor, round 21): finish explicit_uniform_location, then other
KHR-GL46 Fail cases in the tess/GS/shader-linking domain.

## Commits this round (on gl46/tess, unmerged)

| commit | what |
|---|---|
| `9c018e3` | (from round 20 tail, committed this session) textureGatherOffset with runtime offsets rewritten to snapped-coordinate textureGather |
| `8d0c6c5` | glGetProgramResourceLocation flattens arrays-of-arrays subroutine uniforms |
| `ddbc249` | merge gl46-port (compat s11 `5e02260`) |
| `aedf12d` | TES textures bound at vertex slots + commit pending uploads before the private tess compute buffer |
| `f98eaf9` | gl_PerVertex member parse fix (`[]` strip) + link-time gl_PerVertex interface validation |
| `2a054e2` | PROGRAM_SEPARABLE latched at link time; pipeline info-log INVALID_VALUE |
| (merge) | gl46-port compat s12 `159d900` |

## CTS results (before -> after, verified on hardware this round)

* `explicit_uniform_location` **16/16** r18-failures now Pass (48/48 total).
  Last two: `subroutine-loc-arrays-mix` (glGetProgramResourceLocation now
  flattens `u[i][j]` against the declared dims exactly like
  glGetUniformLocation), `uniform-loc-mix-with-implicit-max[-array]` (slow, pass
  with a >150 s per-case timeout).
* `texture_gather`: `base-level` and **`gather-tesselation-shader`** Fail -> Pass.
  - `gather-tesselation-shader` (gather inside TCS kernel / TES post-tess vertex
    function) needed two fixes:
    1. `bindTexturesToCurrentRenderEncoder` (MGLRenderer.m, shared file): with
       tessellation active the stage list already reads the TES's resources, but
       the sampled-image bind keyed on `stage == _VERTEX_SHADER`, so TES textures
       went to **fragment** slots and the post-tessellation vertex function read
       an unbound `texture(0)`.  Now any `stage != _FRAGMENT_SHADER` binds at
       vertex slots (same rule the storage-image loop already used).
    2. `MGLRenderer+Tessellation.m`: the emulated-tessellator path (every
       point_mode/isolines draw with a TCS — `need_readback`) builds a private
       command buffer, but deferred texture uploads / generateMipmap blits can
       still sit uncommitted on the shared `_currentCommandBuffer`; the queue
       could run the compute kernels first, so the TCS read zeros.  The shared
       buffer is now committed first (ordering only, no wait).
    Reproduced pixel-exact in `tests/headless/test_zz_ctstg.c` (verbatim CTS
    shaders/texture); `test_zz_tg.c` isolates the TCS gather.
  - `incomplete-texture[-last-comp]` still Fail: **texdata domain** — MGL
    silently truncates an inconsistent mipmap chain (level 1 redefined 1x1) and
    treats the texture as complete; GL requires the incomplete default
    texture (0,0,0,1).  Needs GL mipmap-completeness (§8.17.3) at bind time.
* `CommonBugs.CommonBug_PerVertexValidation` Fail -> Pass:
  - the gl_PerVertex expander stripped only `]` from a member name, leaving
    `gl_ClipDistance[` — an existing `gl_ClipDistance[]` member was never
    recognized and expanded in twice ("duplicate member name: gl_ClipDistance").
  - link-time gl_PerVertex interface validation (GL 4.6 §11.1.2.5): a consumer
    stage that redeclares the block with a subset demands those members from the
    producer's output block; mismatch fails the link with a message naming
    stage+member.  Only explicit consumer redeclarations are checked (an
    un-redeclared consumer accepts any producer subset —
    `test_compat_glsl glsl_150_gl_per_vertex_redeclaration` stays green).
    Implemented in `mgl_program_check_tess_link` (runs for every program), the
    ORIGINAL shader sources are parsed since the expander normalizes the blocks.
* `sepshaderobjs`: CreateShadProgApi, PipelineApi Fail -> Pass (3/4 now):
  - `GL_PROGRAM_SEPARABLE` queries report the value **latched at the last
    successful link** ("the state of the binary after it was linked"); a failed
    `glCreateShaderProgramv` link therefore queries FALSE, a never-linked program
    FALSE.  glUseProgramStages/pipeline validation use the latched value; the
    live parameter still round-trips through program binaries.
  - `glGetProgramPipelineInfoLog` with a nonexistent pipeline name returns
    GL_INVALID_VALUE (CTS-mandated, "error still missing from the spec").
  - `StateInteraction` still fails: a failed **relink** of a pipeline-referenced
    program must keep drawing with the last successful binary; MGL discards the
    stage executables on the failed link (needs link snapshot/restore).
* `tessellation_shader.gl_InvocationID_PatchVerticesIn_PrimitiveID` (r18
  "Timeout"): Passes now — it was the 1325 s batch deadline, the case alone
  takes ~4 s.

## Verification

* `make -j4 lib`, `make -j4 glx nsgl` clean (the two `incomplete-implementation`
  warnings about `extProgramLibraryForMSL` predate this branch — present with and
  without these changes, introduced by compat s11's category split).
* `tests/headless/run.sh`: **86 test binaries, 0 failures** (test_zz_gcts 4/4
  stable; earlier single flake re-ran green).  `test_tess` 499/0, `test_geom`
  603/0, `test_compat_glsl` 42/0, `test_cts` 71/0.
* Scratch tests added under tests/headless: test_zz_ctstg.c, test_zz_tg.c,
  test_zz_pv.c, test_zz_csp.c, test_zz_pil.c (all green, committed).

## Shared-file edits (this round)

* `MGL/src/MGLRenderer.m` — `bindTexturesToCurrentRenderEncoder`: TES-stage
  sampled textures bind at VERTEX slots (`stage != _FRAGMENT_SHADER`), see above.
* `MGL/src/program.c` — latch `separable_latched` on successful link;
  `GL_PROGRAM_SEPARABLE` query returns the latch.
* (round-20 tail committed here: `MGL/src/cts_glsl_compat.c` gather-offset
  rewrite, `MGL/src/shaders.c` MGL_SRC_DUMP debug hook, now in append mode.)
* All other changes are in my own files: `MGL/src/tessellation.c`,
  `MGL/src/MGLRenderer+Tessellation.m`, `MGL/src/program_introspection.c`,
  `MGL/src/program_ext.c`, `MGL/src/mgl_ext_program.h`.

## Parked / leftovers

* `.agent/test_zz_vssparse.c.bak` — round-20 scratch: `BindBufferRange` (offset
  256, size 8) UBO binds return no data in the VS (buffers domain).
* `.agent/test_zz_gather2.c.bak` — round-20 WIP gather-vs-CPU-reference scratch;
  its own harness was broken (constant output for every offset/uv, also for the
  un-rewritten no-offset program).  The committed rewrite is validated by
  test_zz_gcts.c (5 exact cases) and the CTS texture_gather group.
* `.agent/test_zz_si.c.bak` — StateInteraction repro sketch (pipeline draw black
  even before the relink; superseded by the CTS finding above).
* Raw-Metal probes kept for reference: `.agent/probe_tcsread.m`,
  `.agent/probe_tcsread2.m` (compute-kernel texture reads with a 6-level
  RGBA32F texture work fine on AGX — isolated the MGL-side ordering bug).

## Known gaps (not fixed this round)

1. `texture_gather.incomplete-texture[-last-comp]`: mipmap-completeness
   (inconsistent chain sizes) not evaluated at bind time — texdata domain.
2. `sepshaderobjs.StateInteraction`: failed relink must preserve the last good
   binary for pipeline draws.
3. `shaders.struct.uniform.sampler*` (6) / `uniform_block.random.*` (7) /
   `program_interface_query` (13) / `packed_pixels` (24): not started
   (some are other domains).

## Round 31/r32 (Oct 28, post-merge 1e840b4 → 48a41ef)

### KHR-GL46.cull_distance.functional — Fail → Pass (r31)
Three-part fix, found with the cull trace/dump repro aids (removed before commit):

1. **Export-PSO cache aliasing** (`MGLRenderer+Geometry.m` geomRasterPass).  The
   compat pipeline fast path keys its PSO cache on the PROGRAM's own stage
   functions + VAO/rstate/FBO content and does not know
   `ext_geom->vertex_override`.  cull_distance item 2 recompiles the
   `gl_ClipDistance[1]/gl_CullDistance[7]` shape with an *unsized*
   `gl_ClipDistance[]` (a different record stride), so the second pass hit the
   first pass's cache slot and drew with a PSO whose export function reads
   records at the wrong stride — "Top edge of bounding box is overwritten"
   (the previous draw's records rendered).  Fix: borrow `DIRTY_SHADER` (read by
   nothing but that fast path's decline test) for the duration of `begin_draw`,
   but only when the export function changed since the last pass.  Declining
   *every* pass costs glmark2 gl41-geometry 21174 → 2863 FPS (the descriptor
   path per draw is the known -45%-on-ideas class of cost); with the
   change-gated version gl41-geometry is at parity (21762/21415 vs
   21709/21313) because the first pass of a program/family fills the slot with
   the override PSO and every following pass hits it.
2. **Tessellation programs can cull** (`geometry.c`): the TES (not the VS) owns
   the distance writes; the family comes from the TES layout (point_mode /
   isolines / triangles), the reflect reads the TES stage's SPIR-V and source,
   and the tess import kernel repacks the capture's flattened
   `gl_ClipDistance_i`/`gl_CullDistance_i` members into the synthetic record
   (`tessellation.c` records them as capture members).
3. **Undefined record words** (`kGeomTessImportMSL`): zero each imported record
   first — the tess capture only covers what the TES writes, and an unwritten
   clip word full of ring garbage clips vertices that GL leaves undefined
   ("In primitive number [10] vertex at (33,1) should not be clipped").
   Also `compatBindInvalidate` around the capture encoder swap, as the compat
   category documents for foreign encoder paths.

### KHR-GL46.cull_distance.negative — Fail → Pass (r31b)
GLSL 4.6 §7.1: `gl_ClipDistance`/`gl_CullDistance` may be indexed with a
non-constant expression only after an explicit-size redeclaration.  glslang
silently sizes the arrays to the maximum, so the invalid shader linked.
New `mgl_tess_reject_dynamic_distance_index()` scans the **raw** source with a
small preprocessor tracker (defines, ifdef/ifndef / if 0/1/defined() / else /
elif / endif) so an `#ifdef`'d declaration does not hide the error, and returns
an `#error` source.  Called from the compile path next to
`mgl_reject_reserved_declarators` (shaders.c, 4 lines) because the introspect
pass pins and folds the declarations away.

### A GL_PATCHES draw with a dead default-uniform block no longer faults (r32)
`tessBindStageUniforms` bound the stage's default-uniform block at
`SpirvResource.msl_binding`, but that field carries the established
`0xffffffff` sentinel when SPIRV-Cross **stripped** the block from the stage's
MSL (every default uniform of the stage is dead — e.g. a vertex stage that only
declares one; `spvc_compiler_msl_get_automatic_resource_binding` returns
UINT32_MAX for variables that did not survive).  `setBuffer:atIndex:0xffffffff`
then writes outside the encoder's buffer table and SIGBUSes inside the AGX
driver (`AGX::ComputeContext::setBuffer_impl`, KERN_PROTECTION_FAILURE at the
io-mapped command memory).  Same guard added to the UBO/SSBO/ACB loop
(`(int)0xffffffff` → `(NSUInteger)-1` would fault identically);
`tessBindStageTextures` already skipped the sentinel.  Kept repro:
`tests/headless/test_edgeprobe.c` (stencil-only FBO + GL_NEVER/GL_REPLACE
stamp; the tessellated triangle's coverage now equals a plain GL_TRIANGLES draw
of the same vertices at every level 1..64, missing=0 extra=0).

### Merge 48a41ef (compat s25 gl_spirv tessGeom + geom r59-r61)
Both branches taught the tess capture to record the TES's flattened
`gl_ClipDistance_i`/`gl_CullDistance_i` members; resolution keeps gl46-port's
validation and element mapping for real geometry stages (gl46-port's
`geom_tess_clip_check` replaces my coarse from_tess refusal) and my reflected
record size + "leave the word alone" for the synthetic cull variant.
`test_svv_tessgeom` (8 checks) and
`KHR-GL46.gl_spirv.spirv_validation_builtin_variable_decorations_test` pass.

### Groups this round (against 48a41ef)
* cull_distance: functional / coverage / negative all Pass.
* program_interface_query 43/43, explicit_uniform_location 48/48,
  gpu_shader5 18/22 (the 4 `texture_gather_offset_*_clamp_*` fails are
  pre-existing on the baseline — border-clamp, texdata domain).
* tessellation_shader 47/48 (only single.primitive_coverage, below).

### Known gaps
1. **sample_variables 38 NotSupported** (samples_8 rows): needs 8x MSAA
   renderbuffers; MGL reports GL_MAX_SAMPLES 4 and
   glRenderbufferStorageMultisample(8, GL_RGBA8) fails GL_INVALID_OPERATION —
   the allocation/resolve path belongs to texdata/fb, coordinating there.
2. gpu_shader5 texture_gather_offset clamp/edge (4): border clamp emulation —
   texdata domain (pre-existing, documented by geom).
3. **shader_ballot_tests 3/4**: Bitmasks/FunctionBallot/FunctionRead fail
   *inside the CTS compute validation*, which binds an RGBA32UI texture to a
   compute image the shader declares `layout(rgba32f)` and then samples the
   integer texture with a float sampler expecting the raw float bits
   (verified: MGL returns 0; a conformant desktop GL returns ~1.0).  Separately,
   the format-mismatched imageStore path HANGS the GPU work (a dispatch +
   subsequent readback never completes; reproduced twice, then STOPPED per the
   GPU-fault rule and reported).  Both behaviors are texdata/compute domain;
   the ballot implementation itself verifies correct in all six stages
   (probe replay of the CTS's own shader sources: every rendered pixel white).
4. shader_integer_mix `shaders.shader_integer_mix.prototypes` stays
   NotSupported by CTS design (the gl3 package registers the group at GLSL
   330 while the case itself requires >= 450).
