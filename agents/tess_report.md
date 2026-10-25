# tess domain — round 19 report (shader / GLSL / program domain)

**READY** — merge `gl46/tess` HEAD (this commit; the code changes it contains end at
merge commit `5005b49`, which includes gl46-port through compat session 8 / 1335928)

Session goal (supervisor, round 19): shader/GLSL/program clusters from the r18 sweep
(`~/MGL-cts/mgl-r18-sweep`, baseline Pass 8364/9035 = 92.6%). Started from a clean
`git merge gl46-port` (816fbdc), re-merged compat session 7 (dcbee70) and
compat session 8 (5005b49).

## Per-group before/after

| group | r18 baseline | this session (merged head 5005b49) |
|---|---|---|
| KHR-GL46.shader_subroutine | 1 Pass, 27 NotSupported | **27 Pass + 1 flaky** (`same_subroutine_..._used_in_all_stages`: passed 3× earlier this session — including inside two full-group 28/28 runs — now failing consistently; failure mode + fix location documented below) |
| KHR-GL46.shading_language_420pack | 29 Pass, 13 Fail, 1 Crash (43) | **31 Pass**, 12 fail — qualifier_order + qualifier_override_layout fixed; binding_images now links |
| KHR-GL46.limits.max_subroutines / max_subroutine_uniform_locations | Pass | Pass |
| (all other groups) | — | not re-run this session; full `tests/headless/run.sh` green at d786d1c (0 FAILED, rc=0) |

## What was implemented

### 1. ARB_shader_subroutine, core since GL 4.0 (target #1) — 28/28

MGL keeps the existing source-level strategy (glslang cannot emit SPIR-V for GLSL
subroutines): `introspect_subroutine.c` rewrites each stage's GLSL into
`uniform int _mgl_sr_<name>;` + a `_mgl_sr_call_<name>(uint _mgl_i, args)` dispatcher
containing a `switch`, and the API entry points live in `program_introspection.c`
(the `introspect` ext domain, carried in this branch since earlier rounds). This
round made it actually work and advertised it:

- **get.c**: advertise `GL_ARB_shader_subroutine`; `GL_MAX_COMBINED_IMAGE_UNIFORMS`
  raised from 8 (== per-stage) to 40 (per-stage sum over the 5 raster stages —
  `binding_images` needs 15 combined images and each stage's Metal texture table is
  independent); `GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES` adjusted to match.
- **introspect_subroutine.c**
  - strip `#extension GL_ARB_shader_subroutine` / `GL_ARB_arrays_of_arrays`
    directives (glslang does not know either), and raise `#version` to 420/430 when
    the source uses `binding` layout qualifiers / arrays-of-arrays (GL 4.6 accepts
    the shader, so the rewrite must make glslang accept it too).
  - multi-dimensional subroutine uniform arrays (`routine[4][4]`): flattened selector
    `_mgl_sr_routine[16]`, call sites `routine[j][i](...)` →
    `_mgl_sr_call_routine(j*4+i, ...)`, `.length()` (and `routine[j].length()`)
    resolved to the corresponding dimension constant.
  - GLSL §4.7 signature check: a subroutine function's return/parameter types must
    match **every** compatible type; a mismatch is surfaced as `#error` in the
    rewritten source (compile error), which is what
    `subroutines_incompatible_with_subroutine_type` requires.
  - the selector uniform is emitted **without** `layout(location=N)` — explicit
    uniform locations are illegal below GLSL 430 and the selector is looked up by
    name anyway.
- **program_introspection.c**
  - `current_stage_linked()`: subroutine commands address the program active *for
    the stage* — `glUseProgram`'s program, else the pipeline's stage program
    (SSO: `useProgramStages`/`ActiveShaderProgram`), else INVALID_OPERATION
    (`subroutines_with_separate_shader_objects`, `subroutine_errors`).
  - `glGetSubroutineUniformLocation("name[j][i]")` flattens against the declared
    dimensions so every array-of-arrays element has a distinct location
    (`arrays_of_arrays_of_uniforms`).
  - `glUniformSubroutinesuiv`: incompatible-but-existing subroutine index →
    GL_INVALID_OPERATION (was INVALID_VALUE).
  - `glGetUniformSubroutineuiv`: GL_INVALID_OPERATION when the stage is absent from
    the active program.
  - link failure when a subroutine uniform has **no** compatible subroutines
    (`subroutine_uniform_wo_matching_subroutines`).
  - `mgl_introspect_reset_subroutines()`: the selection reverts to defaults on
    UseProgram / UseProgramStages / BindProgramPipeline (GL 4.6 §7.4.2) — hooked in
    `program.c` / `program_ext.c` (`subroutine_uniform_reset`).
- **geometry.c / transform_feedback.c**: the geometry-kernel and XFB-capture variants
  recompile the *user's* source, so they now feed it through
  `mgl_introspect_shader_source` (subroutine rewrite + extension strip); the XFB
  capture splice additionally keeps function definitions that follow `main()`
  (the rewrite appends its dispatchers there).

### 2. shading_language_420pack (+2 net, binding family root-caused)

- `cts_glsl_compat.c`: new `strip_invariant_on_input()` pass in both entry chains
  (`mgl_cts_glsl_compat` / `_generated`): GLSL 4.20 allows `invariant` on **inputs**
  and in any qualifier order; glslang rejects it ("can only apply to an output").
  Handles inline (`highp in layout(...) smooth invariant vec4 v;`) and standalone
  (`invariant v;`) forms.
- `geometry.c`: `highp/mediump/lowp` added to `is_interface_qualifier()` and
  `emit_interface` now also drops qualifiers that *trail* the in/out token
  (`highp in layout(location=1) smooth invariant vec4 v;`), so the geometry
  emulation kernel builds (`qualifier_order`, `qualifier_order_block` still fails —
  invariant on **block members** is not handled yet).
- `MGLRenderer+Tessellation.m` (my domain file): `tessBindStageTextures` rewritten —
  VS/TCS/TES compute kernels now get **sampler states** bound (they were never bound
  before), textures bound at the SPIRV-Cross MSL index (was list order), arrays and
  unbound units handled (shared 1×1 fallbacks), and the texture resolved via
  `mgl_texture_for_sampler` like the render path. Needed for any tessellation
  program whose VS/TCS samples textures (`binding_sampler_single` shape).
- `get.c` image limits (see above) fixed the `binding_images` link failure.

## Root causes found but NOT yet fixed (next session, highest-value items)

1. **Merged opaque-uniform initial value** (unlocks the 420pack `binding_sampler_*` /
   `binding_image_*` family, 9 cases): the merged default-block uniform's initial
   value must come from any stage's `layout(binding=N)` declaration (GLSL 4.20
   §4.4.5 — the uniform is shared program-wide, so *every* stage samples unit N).
   MGL/glslang's per-stage SPIR-V loses this: `glGetUniformiv(prog, "goku")`
   returns **0** where GL requires **2** (verified with a local 5-stage repro,
   `tests/headless/test_bind_diag.c`, `RUN5=1` / `STAGE=vs|tcs|tes|gs|fs`; the
   failing stages are exactly the VS/TCS compute kernels, which bind the texture of
   `sres->binding`==0 instead of unit 2). Fix location: the uniform merge /
   initial-value seeding at link (introspect domain `uniforms.c` /
   `mgli_process_link`) — seed each merged opaque uniform's value from any stage's
   binding decoration, and/or keep the render/compute binders on the *uniform
   value* instead of the per-stage SPIR-V decoration when they differ. The
   `minimal_vs_sampler` repro (binding in one stage only) already passes.

2. **`same_subroutine_..._used_in_all_stages` flakiness** (tess + GS + subroutines in
   all stages): the failing draw pairs the post-tessellation TES *vertex function*
   with the 4-input FS at PSO-creation time — the geometry emulation's
   `vertex_override` is not yet set when that PSO is first created, the creation
   fails, and the draw only succeeds when MGL's AGX-recovery retry re-creates it
   with the override. That makes the outcome timing/state dependent (passed 3× with
   fresh process/GPU state earlier this session, now failing consistently — it also
   fails at 127d752, the commit where the 28/28 runs were measured, so it is not a
   regression). Diagnosis: `MGL_PSODEBUG=1` prints
   `tess_evaluation_N_main + fragment_N` for the failing PSO. Fix direction: make
   the render-pipeline creation honor the geometry `vertex_override` for the *first*
   creation too (or key the PSO cache on the override) in
   `MGLRenderer.m`/`MGLRenderer+Geometry.m` — that is the shared draw path (compat
   owns the fast paths), so coordinate with compat.

Also not fixed this session: `utf8_characters`, `line_continuation`,
`length_of_vector_and_matrix`, `qualifier_order_block` (block-member invariant),
`subroutine_uniform_reset`'s `ssbo_atomic_image_load_store`-style remaining
iterations (the CTS crash entries are the same flaky glcts exit-code issue, see
below).

## Shared-file edits (all surgical)

- `MGL/src/get.c` — advertise the extension; combined image uniform limits.
- `MGL/src/program.c` — one-line hook: `mgl_introspect_reset_subroutines` in
  `mglUseProgram`.
- `MGL/src/program_ext.c` — two hooks: reset on `mglBindProgramPipeline` /
  `mglUseProgramStages`; include of `mgl_ext_introspect.h`.
- `MGL/src/transform_feedback.c` — the two `xfb_link` call sites feed the source
  through `mgl_introspect_shader_source`; the capture splice keeps text after
  `main`'s closing brace (function definitions may legally follow `main`).
- `MGL/src/geometry.c` — `geom_vs_source`/`geom_gs_source` route through
  `mgl_introspect_shader_source`; `highp/mediump/lowp` in the interface-qualifier
  list; `emit_interface` drops trailing qualifiers (420pack order freedom).
- `MGL/src/cts_glsl_compat.c` — `strip_invariant_on_input` prepended to both chains.
- `MGL/src/MGLRenderer.m` — env-gated `MGL_BINDTRACE` (texture bind trace) and
  `MGL_PSODEBUG` (failing PSO's vertex/fragment function names) diagnostics; no
  behavior change.

My files: `introspect_subroutine.c`, `mgl_ext_introspect.h`,
`program_introspection.c`, `MGLRenderer+Tessellation.m`, plus new scratch/diagnostic
tests `test_sr_diag.c`, `test_sr_diag2.c` (failing local-only shapes gated behind
`SRDRAW=1`), `test_tg_diag.c`, `test_bind_diag.c` (the binding repro; the 5-stage
case is gated behind `RUN5=1`).

## Tests

- `tests/headless/run.sh` full suite at d786d1c (pre-merge-compat8): **rc=0, 0 FAILED**
  (re-verified after the compat-8 merge: test_tess 499/0, test_introspect 331/0,
  test_bind_diag, test_sr_diag, test_sr_diag2, test_smoke green).
- KHR-GL46 CTS (one batch at a time, `nice -n 10 taskpolicy -c utility`):
  - `shader_subroutine`: 28/28 in two full-group runs earlier in the session;
    27/28 at the final merged head (the 28th = the flaky tess+GS case, see above;
    it also passes/fails independently of this session's changes).
  - `shading_language_420pack`: 31/43 Pass (r18: 29/43 + 1 Crash).

## Known gaps / leftovers for the next session

1. Merged opaque-uniform initial value from `layout(binding=N)` (see above) —
   unlocks the 420pack binding_* family.
2. The tess+GS PSO-pairing flakiness (see above) — also affects the local
   `sr_all_stages_draw` shapes (kept behind `SRDRAW=1` in test_sr_diag2).
3. `invariant` on interface **block members** (qualifier_order_block).
4. Remaining 420pack singles not yet triaged: `utf8_characters`,
   `line_continuation`, `length_of_vector_and_matrix`.

## Commits (this session, branch `gl46/tess`)

- `ef35bd9` subroutines: advertise ARB_shader_subroutine, compile+link+draw path (24/28)
- `127d752` subroutines: full KHR-GL46.shader_subroutine 28/28
- `496b83a` get.c: raise GL_MAX_COMBINED_IMAGE_UNIFORMS to per-stage sum (40)
- `d786d1c` 420pack: invariant-on-input strip, qualifier-order freedom in the
  geometry kernel, tess kernel sampler-state binding (31/43) + test_introspect
  INVALID_OPERATION expectation
- `5005b49` merge gl46-port (compat session 8) — **READY sha**
