# tess domain — round 29 report (perf: O(1) per-draw gates)

READY ddd8f68 — ab_bench m0 8.22->6.66 ms (-19.0%), m1 46.50->46.53 ms (+0.06%); glmark2 ideas median 511.5->518.5 (+1.4%), jellyfish 4551->4814 (+5.8%, noise-dominated)

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
