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

## 7. Fix log (cts agent)

| date | cluster | fix | before → after (group) | overall |
|------|---------|-----|------------------------|---------|
| 2025-10-21 | #4 (464) constant_expressions.*_tess_control/_tess_eval (glMapBufferRange GL_INVALID_OPERATION / wrong values) | Five bugs in the tessellation path, none actually in glMapBufferRange (whose flag matrix was already spec-correct - verified against GL 4.6 6.3 in tests/headless/test_mapbuf.c): (a) rewrite_vs_kernel (tessellation.c) grew its MSL buffer with realloc and kept using the stale pointers, so the injected spvStageInputSize/mgl_vs_params locals landed in freed memory -> 'use of undeclared identifier spvStageInputSize'; same class of bug in the params-parameter append. (b) the triangle-domain TCS rewrite left .insideTessellationFactor[1] (a scalar in MTLTriangleTessellationFactorsHalf) and .edgeTessellationFactor[3] (array of 3) subscripted -> Metal compile error; now mapped onto the scalar / dropped with side effects kept. (c) a TCS writing only tess levels (no per-vertex outputs) has a 0-byte control-point record and the draw bailed with GL_INVALID_OPERATION (stride clamped to 16). (d) tessBindStageUniforms (MGLRenderer+Tessellation.m) indexed ctx->state.buffer_base with the GLenum (0x8F11) instead of the internal buffer-type index, so UBOs/SSBOs of tess stages were read from garbage memory and never bound; it also bound at a running counter instead of the resource's Metal buffer index. (e) compile_stage treated Metal buffer index 0 as 'no binding' (msl_b > 0), so resources at [[buffer(0)]] were bound at their GL binding point instead. Plus: NSLog with a C-struct pointer argument crashed objc_msgSend in generatePipelineDescriptor's error path (now %p), and SSBO/atomic buffer bindings now mark the buffer GPU-written so later CPU reads synchronise. | constant_expressions Pass 928 -> 1160 (all 464 tess cases pass; the 232 left are geometry stage = cluster #5, geom worker) | 65.9% -> 71.1% |
| 2025-10-21 | #1 (736) pixelstoragemodes "Shader creation failed" | GLSL identifiers that are MSL keywords/types (`sampler`, `texture`, `kernel`, `float4`, ...) are now renamed (`<name>_mgl`, collision-checked against every identifier of the module) before SPIRV-Cross emits MSL: `MGL/src/msl_reserved.{c,h}` + `mgl_msl_mangle_reserved()` hooked after introspection in `program.c:compile_stage`, `geometry.c:geom_spirv_to_msl`, `transform_feedback.c:xfb_spirv_to_msl`. Introspection keeps the GLSL names (rename runs after reflection). | pixelstoragemodes Fail=736 → **0** (Pass=736, NotSupported=160) | 50.5% → **58.6%** |

### geom worker fix log

| date | cluster | fix | before → after (group) | notes |
|------|---------|-----|------------------------|-------|
| 2026-10-20 (round 26) | **suite regression** "test_tess 121/499, test_geom 22/415, test_compat_gears 4/10 from 48bb3bd on" | Two separate bugs found, neither what the bisect suggested.  (1) **Real pool hazard in MY code (48bb3bd)**: the slot-reuse shortcut "the current, uncommitted command buffer cannot have executed" is wrong in the other direction — its already-encoded kernels read the pool ranges when it is finally committed, so a 5th batched draw reused a slot whose ranges draws 1-4 of the SAME command buffer still read (92/96 quads lost; the old pool_rotation_hazard test never caught it because its interleaved glBufferSubData flushed between draws).  Rewritten: pool slots hand out **disjoint bump ranges** (any number outstanding), reclaim (bump reset) only after the slot's recorded consumer completed, the recorded consumer is the one that completes LAST (uncommitted beats committed; never displaced by another mark), consumer pointers now **retained** (the old raw `__bridge void*` stores dangled → objc_msgSend on freed objects), and when every slot is full of live ranges the oldest pending consumer is waited for (committing the still-empty current buffer first — the only wait).  Knobs MGL_GEOM_POOL_CAP/MGL_GEOM_POOL_SLOTS; new pool_wrap_stress_tiny runs FIRST in test_geom and forces 16 KiB x 2 slots for the whole binary (96 batched GS draws with nothing between them + a tess->GS draw every 8th).  (2) **The visible test_tess/test_compat_gears failures were the headless default framebuffer**, not the pools: identical 121/499 with dc36fe0's Geometry.m and with 48bb3bd's, load-independent that day, and every failing pixel was outside a 100x100 readback; the windowless dummy NSView's layer bounds became 100x100 "on current AppKit" and the layer-creation races make it flaky per launch (it passed 4x in a row mid-session).  Fixed by cherry-picking the cts agent's 4a32acc (fbDrawableSize ignores the layer for a windowless view, sizes the default fb to the union of GL viewports) + d561e01's test viewport hygiene; d561e01's 200x200 in the three interface-block tests is a thinko (px(128,128) is the centre of 256x256 — with 200x200 it lands outside the drawn triangle) → corrected to 256x256; 4a32acc/d561e01's test_cts/test_pixelformats riders were dropped (need the unmerged cts gl_NumSamples rewrite / textures' stricter texdata validation) | test_tess 121→0 failures (stable across runs AND under GPU load), test_geom 432/0 (whole binary under tiny pools), full suite 5343 checks 0 failures | commits 5670020/0c1509c/5e937b5 |
| 2026-10-20 | #2+#5 (942) "SPIR-V requires location for user input/output" / "vertex capture variant compile failed" | glslang's map_io is what actually assigns the AUTO_MAP locations/bindings: geom_glsl_to_spirv() and xfb_glsl_to_spirv() now call it between link and SPIRV_generate; program.c links through the GLSL I/O mapper+resolver (glslang_glsl_mapper_create) so interface variables are matched by NAME across stages (GL link semantics) instead of the default resolver, which "does not do proper cross-stage lining up" | constant_expressions.*_geometry 297 → all pass; buffer_storage/geometry groups improved | the cross-stage change needed the compat varying fix below |
| 2026-10-20 | geometry_shader 91 Fail | (a) geometry program table grows on demand (was fixed 64 slots — "program table exhausted"); (b) a '#define' after blanks no longer swallows the next declaration; (c) vector/matrix varyings derive their base type from the SPIR-V component type (ivec4 inputs were loaded as vec4); (d) top-level comments were copied with copy_renamed() which has no stop chars there and consumed the rest of the file ("flat out x;" survived into the compute kernel — the limits cluster); (e) plain (default-block) uniforms of the capture variant/kernel are re-declared as one std430 block with the offsets introspect baked into mgl_DefaultUniforms, pinned to the renderer's flat buffer positions; (f) the capture pass builds the vertex-attrib plan (buffer indices) under the program before generating its descriptor — a draw-stale plan made the FIRST draw after a link read the uniform buffer as vertex data; (g) GS inputs not written by the VS link (undefined) and record zeros; (h) glFramebufferTexture level > texture levels → GL_INVALID_VALUE; (i) geometry limit constants aligned with glslang's resources (1024 uniform components, 0 atomic counters/buffers, TOTAL_OUTPUT 2048) | geometry_shader 38 Pass → 65 Pass / 64 Fail (131 cases; remaining: adjacency assembly 8, rendering drawcall pixel diffs ~29, primitive_queries 3, atomics) | test_geom 427/0, test_geom_cts 7/0 |
| 2026-10-20 | compat regression from the above | compat_ffp.c/compat_glsl.c varyings now use ONE name per varying under fixed locations (mglSecondary/mglTexCoordV%d/mglFogCoordV, MGLC_VAR_* in mgl_ext_compat.h); the FFP fog-coordinate attribute is actually forwarded through a varying now | test_compat_pipeline/render2 green again | gl46/compat's newer clamp_axes line kept with the shared names |

| 2026-10-21 | KHR-GL46.gpu_shader_fp64.fp64 group tally (round 6) | **8/9 Pass** (331 s): errors, max_uniform_components, named_uniform_blocks, state_query, conversions, illegal_conversions, valid_constructors, operators.  Only **varyings** fails (14/26 variable batches "failed to link").  Fixes this round: glGetUniformuiv of a negative double now converts to 0 (uniforms.c); TES-output transform-feedback capture also runs on the CPU-reference tessellator path when the GPU tessellator's worst-case scratch (64 MiB budget) does not fit — 1024-patch draws (MGLRenderer+Tessellation.m tessEnsureCapturePso/tessPublishCapture); the capture kernel's fallback `mgl_cp_rec` type is declared (it compiled to "unknown type name"); xfb "name[N]" element subscripts resolve in the geometry capture path (geometry.c); double arrays/matrices as vertex ATTRIBUTES fold to raw u32 slot arrays (fp64.c attr_slotarr_ty/attrsgeom/attrorig — constant (element,column,row) chains); `uint64BitsToDouble(uvec2(..))` → `packDouble2x32(uvec2(..))` in the geometry kernel's generated GLSL.  Remaining varyings gap: dmat*/dvec3-4 VARYINGS — the fp64 lowering now emits a named-member slot struct per member, but the tess canonical-record MSL splice (msl_rewrite_struct_body + member renames) does not flatten nested slot structs, so any batch containing a matrix/dvec3+/dvec4 varying fails the TES MSL compile.  Scalar/dvec2-only batches link.  Plan: expand fp64 block members into N flat uint4 members (like the vertex-attribute slots) so no nested structs reach the splice. | test_tess 499/0, test_fp64{,b,c} 417+54+48 checks 0 fail, full headless suite rc=0 (425 PASS lines), fp64 group 8/9 |

**Cross-branch note (RESOLVED 2026-10-20):** the headless default framebuffer
size was the *real* cause of the test_tess 121 / test_compat_gears 4 failures
(see the round-26 row above): the windowless dummy NSView's layer bounds became
100x100 on current AppKit and the layer-creation race makes the size flaky per
launch.  The cts agent's renderer fix (4a32acc: fbDrawableSize sizes the
default framebuffer to the union of GL viewports for a windowless view) plus
d561e01's test viewport hygiene are now on this branch (cherry-picked);
gl46-port still needs them merged from either branch.



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
* fp64 (gpu_shader_fp64) is SUPPORTED since the soft-float port; the fp64.fp64 group now runs (see the 2026-10-21 row above).
