# tess domain — round 24 report (tess / shader-linking / GLSL compat)

READY 47055c4

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
