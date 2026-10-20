# MGL conformance harness — first pass against the Khronos GL CTS

Status: **harness running, first pass complete, no conformance fixes started** (by
instruction; the work list below is for splitting). One MGL hang had to be fixed
just to let the CTS run past its first case — see "MGL bug fixed during bring-up".

## 1. Suite choice: VK-GL-CTS (KHR-GL46) — why not piglit

* **VK-GL-CTS** (`https://github.com/KhronosGroup/VK-GL-CTS`, branch
  `opengl-cts-4.6.1`, commit `642964df5ee1319fe3330ebf54465f4c7367f3ab`) is the
  *official* OpenGL conformance suite; `KHR-GL46.*` is exactly the GL 4.6 core
  conformance target MGL claims. A custom `tcu::Platform` port is a small,
  well-defined piece of glue (~250 lines) because the framework already provides
  everything MGL needs: an FBO-backed "default framebuffer"
  (`--deqp-surface-type=fbo` → `glu::FboRenderContext`), GL entry-point
  resolution by symbol name from a shared library (`glw::initFunctions`), and a
  plain-text `.qpa` log with one `<Result StatusCode=..>` per case.
* **piglit** would have needed waffle with a CGL/MGL backend plus its own result
  parser, and covers GL 4.6 far less systematically than the CTS (no
  per-extension KHR-GL46 grouping, no official conformance meaning).
* The GTF half of the GL CTS (`GTF-GL46.*`, Khronos-confidential, fetched from a
  members-only repo into `external/kc-cts`) is **not available** on this machine;
  the first pass therefore covers `KHR-GL46.*` (9035 cases).

## 2. What was built and where (nothing in the repo tree except committed glue)

* CTS tree: `~/MGL-cts` (clone + `external/fetch_sources.py` downloads; cmake
  `-DDEQP_TARGET=mgl -DCMAKE_POLICY_VERSION_MINIMUM=3.5`, ninja `glcts`).
  Binary: `~/MGL-cts/build/external/openglcts/modules/glcts`, run with that
  directory as cwd (the `gl_cts/` data dir lives next to it).
* Committed glue (this repo):
  * `agents/conformance/glue/framework/platform/mgl/tcuMGLPlatform.{hpp,cpp}` —
    `tcu::Platform` + `glu::ContextFactory` ("mgl"): dlopens `libmgl.dylib`
    (path from `$MGL_LIBRARY`, default `<repo>/build/libmgl.dylib`), resolves the
    whole `glw::Functions` table from it, reports one process-global core-4.6
    context (shared-context requests alias it and are warned about).
  * `agents/conformance/glue/targets/mgl/mgl.cmake` — the `DEQP_TARGET=mgl`
    definition.
  * `agents/conformance/setup.sh` — clone/fetch/configure/build, idempotent.
  * `agents/conformance/run.py` — runner (below).
* Run recipe:
  ```
  make lib
  bash agents/conformance/setup.sh
  python3 agents/conformance/run.py --resume --batch 16 --per-case-timeout 25
  python3 agents/conformance/run.py --summarize-only
  ```
  Single case: `python3 run.py` flags or directly
  `glcts --deqp-case='...' --deqp-surface-type=fbo --deqp-gl-context-type=mgl
  --deqp-visibility=hidden --deqp-surface-width=256 --deqp-surface-height=256`.

## 3. Runner design (per-case timeout + crash recovery)

* Case list exported once (`--deqp-runmode=stdout-caselist`), cases run in
  batches of 16 through `--deqp-caselist-file=<trie>`; a batch is killed at
  `45s + 25s·n` deadline. Every case that began but produced no
  `<Result>` is the "victim": recorded as Timeout/Crash, the *rest of its batch
  is re-queued* in the next batch. Results checkpoint (`results.json`) is
  rewritten per batch, so the pass resumes across invocations.
* `--deqp-log-images/shader-sources/decompiled-spirv/empty-loginfo=disable` keep
  logs small; per-batch `.qpa/.out/.err` are kept under `mgl-results/logs/`.

## 4. First-pass results (MGL @ gl46/geom 78478ec, M4 Pro, 2025-10-20)

Totals (9035 listed, 9031 recorded; the 4 lost to batch bookkeeping were re-run
individually below):

```
  Pass             4558  (50.5%)
  Fail             3228  (35.7%)
  NotSupported     1167  (12.9%)
  Crash              60  (0.7%)
  InternalError      16
  CompatibilityWarning 1
  ResourceError       1   (geometry_shader.max_components, expected)
```

Per-group table (complete): `~/MGL-cts/mgl-results/summary.txt`; worst groups:

```
  pixelstoragemodes      896  Fail=736 NotSupported=160
  constant_expressions  1392  Pass=696 Fail=695 Crash=1
  shaders               2382  Pass=1687 Fail=689 NotSupported=6
  packed_pixels          250  Fail=249 Pass=1
  sample_variables       190  Fail=150 NotSupported=38 Pass=2
  geometry_shader        132  Fail=91 Pass=38 Crash=1 ResourceError=1
  direct_state_access    371  Pass=290 Fail=76 InternalError=4
  shader_storage_buffer_object 124 Fail=50 Pass=44 Crash=30
  texture_gather          75  Fail=38 Pass=37
  tessellation_shader     58  Fail=35 Pass=20 Crash=2
  layout_location         35  Fail=31 Pass=4
  shading_language_420pack 43 Fail=29 Pass=13 Crash=1
  limits                 140  Pass=112 Fail=28
```

NotSupported (1167) is dominated by `gpu_shader_fp64` (658; fp64 not
implemented), `shader_image_size` (60), `blend_equation_advanced` (88),
`shader_subroutine` (27) — i.e. extensions MGL does not advertise; that part is
healthy signalling, not failures.

## 5. Ranked work list (top failure clusters, one representative each)

Root causes marked **[rc]** were reproduced/verified locally; others are
first-guess from the CTS logs.

| # | cases | symptom (representative) | first-guess root cause | likely domain |
|---|-------|--------------------------|------------------------|--------------|
| 1 | 736 | `pixelstoragemodes.teximage2d.r11g11b10f.0_0` — "Shader creation failed" | **[rc]** GLSL uniform named `sampler` (and any MSL type name: `texture`, `depth2d`…) collides in SPIRV-Cross MSL output: `sampler samplerSmplr` → Metal: "must use 'struct' tag to refer to type 'sampler'". Need reserved-name remap of user symbols when emitting MSL. Repro: 20-line program with `uniform sampler2D sampler;` | shaders/program (MSL emission) |
| 2 | 645 | `buffer_storage.map_persistent_draw` — "vertex capture variant compile failed: 'location': SPIR-V requires location for user input/output" | **[rc]** the XFB *capture-variant* compile path does not set `GLSLANG_SHADER_AUTO_MAP_LOCATIONS` (or inject locations) for `#version <330`-style shaders without layout() — plain compile of the same shader works | transform_feedback |
| 3 | 581 | `shaders.uniform_block.instance_array_basic_type.std140.bool` — "Uniform type mismatch: expected: bool got: uint" | **[rc]** `glGetActiveUniform(siv)` reports bool uniforms as `GL_UNSIGNED_INT`; SPIR-V type→GL type mapping for OpTypeBool | program_introspection |
| 4 | 398 | `constant_expressions.array_abs_int_tess_control` — "glMapBufferRange(): glGetError() returned GL_INVALID_OPERATION" | MGL rejects a legal map configuration (READ flag on a buffer the CTS just wrote with a compute/vertex shader; possibly MAP_PERSISTENT/COHERENT or unsynchronized read map rules) | buffers |
| 5 | 297 | `constant_expressions.array_abs_int_geometry` — "Compilation failed at glcShaderConstExprTests.cpp:159" | same as #2 (capture variant of the const-expr geometry tests) | transform_feedback |
| 6 | 169 | `compute_shader.conditional-dispatching` — "Test Run() failed" | various; needs triage (compute results wrong) | compute |
| 7 | 148 (+83 +18) | `packed_pixels.pbo_rectangle.compressed_red` — "Invalid format used but glTexImage2D succeeded"; `GL_UNPACK_ROW_LENGTH` cases; "Valid format used but failed" | missing/incorrect GL_INVALID_OPERATION / GL_INVALID_ENUM validation of format/type/fetch combinations in glTexImage/glReadPixels paths; UNPACK_ROW_LENGTH not honoured for some uploads | pixel transfer / textures |
| 8 | 144 | `sample_variables.mask.rgba32f.samples_0.mask_0` — "gl_NumSamples: undeclared identifier" | glslang front-end lacks the GL 3.3-era GL_ARB_sample_variables spelling for 330 shaders (or MGL rewrites the #version and drops the extension); needs shader-source preprocessing | shaders |
| 9 | 115 | `direct_state_access.renderbuffers_storage` — Fail (no detail) | needs triage (DSA renderbuffer storage paths) | framebuffers |
| 10 | 60 | `glcts exited rc=-N` — e.g. `compute_shader.copy-image` (**[rc]** begins, process dies, no result), `shader_storage_buffer_object.*` (30), `gl_spirv.spirv_glsl_to_spirv_builtin_functions_test` (8) | real in-process crashes (null function pointers or segfaults); each needs a stack sample | several |
| 11 | 48 | `shaders.uniform_block.*.column_major_mat2` — "Image comparison failed, got 16384 non-white pixels" | rendering with matrix-in-UBO instance arrays wrong (possibly the same bool/uint report bug confusing the *test's* reference, or genuine UBO packing/rendering issue) — needs triage | program introspection |
| 12 | 36 | `draw_indirect.advanced-twoPass-transformFeedback-arrays` — "After test execution glGetError() returned: GL_INVALID_OPERATION" | leftover INVALID_OPERATION from an xfb/indirect path that should succeed | transform_feedback / draw |
| 13 | 31 | `layout_location.isampler_2d` — "Creation of program failed" | needs triage (location assignment for samplers?) | program link |
| 14 | 30 | `texture_gather.base-level` — Fail | textureGather result wrong at base level | textures |
| 15 | 22+19+6+3+2 | `internalformat.copy_tex_image.rgb10_a2`, … — "Compile failed at glcInternalformatTests.cpp" | shader compile of glsl es-style sources (same family as #1/#8?) — needs triage | shaders |

Geometry-shader-specific (this agent's domain; separate note for the work split):

* `geometry_shader` overall: 91 Fail / 38 Pass / 1 Crash / 1 ResourceError.
  * `adjacency.adjacency_indiced_line_strip` (**[rc]**): "At [4] position
    adjacency buffer position Reference value is different than the rendered
    data (1) vs (0)" — adjacency vertex delivery/assembly for indexed
    `line_strip_adjacency` is wrong (geom compute-emulation input assembly).
  * `rendering.*` (23) and `primitive_counter` (19): "ERROR: N compilation
    errors. No code generated" — GS variants fail to compile in the CTS
    pre-pass (likely the #1 `sampler`/MSL collision or #8 preprocessing).
  * `layered_fbo.fb_texture_invalid_level_number` Fail (3 sibling cases Pass).
* The 4 cases lost by the runner were re-run singly: `fb_texture_invalid_attachment`,
  `fb_texture_invalid_value`, `fb_texture_no_fbo_bound_to_target` **Pass**;
  `fb_texture_invalid_level_number` **Fail** (missing GL_INVALID_VALUE for an
  invalid FBO texture level).

## 6. MGL bug fixed during bring-up (harness blocker)

`glTexBuffer(GL_TEXTURE_BUFFER, GL_R8, 0)` on the **default** (name-0) texture
buffer object — which `glu::resetStateGLCore` issues after *every* CTS case —
hung MGL forever: the texparam-record array growth `while (n <= name) n *= 2`
wrapped a GLuint for the internal sentinel name `TEX_OBJ_RES_NAME`
(0xcafebeef) and spun. `texture_query.c` now refuses records for sentinel
names and caps the array (`MGL_TEXPARAM_MAX_RECORDS`, 1<<22). Full MGL headless
suite after the fix: 4647 checks, 0 failures.

## 7. Harness limitations / caveats

* `GTF-GL46.*` not run (Khronos-confidential sources unavailable).
* Single global GL context: CTS cases that need genuinely separate (shared or
  unshared) contexts alias the same context; results for those are unreliable.
  The platform prints a warning when it happens.
* Per-case deadline is enforced at batch granularity (25 s/case): cases slower
  than that die as batch-kill and are recorded as Timeout — none occurred in
  this pass; slow-but-working cases would need a rerun with a bigger budget.
* Batch logs from the first invocation were overwritten by later invocations
  with the same batch index; `results.json` keeps status+reason for every case,
  and any case can be re-run singly for a full log (recipe in §2).
* 4 cases were dropped by the runner's bookkeeping and re-run manually (§5).
* fp64 (gpu_shader_fp64, 658 cases) is correctly NotSupported, not a failure.
