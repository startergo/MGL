READY 63343cc (geometry_shader 133/136 + transform_feedback 18/20 under the 480 s cap; headless suite green; test_geom 541 checks / 0 failures)

# geom domain — round 31 handoff

Branch `gl46/geom`, code tip `63343cc` (after merging gl46-port `0d4ed40`).
Baselines: `test_geom` 541 checks / 0 failures, whole `tests/headless/run.sh` green (57 binaries).

## What this round did (all committed)

geometry_shader group: **133 Pass / 2 NotSupported / 1 Fail** of 136 (`~/MGL-cts/geom31c-results`).

1. `mgl_geom_get_programiv` (geometry.c) answers GEOMETRY_INPUT_TYPE / OUTPUT_TYPE /
   VERTICES_OUT / SHADER_INVOCATIONS from the stage's own SPIR-V reflection as soon as the
   stage compiled — a linked geometry-only *separable* program whose emulation is only built at
   the first draw (CTS `api.getProgramiv3`).
2. `geomBindStageBuffers` / the capture-pass bind walk (MGLRenderer+Geometry.m) honour the
   `glBindBufferRange` offset of each UBO/SSBO entry instead of binding offset 0 (one buffer
   bound 8 times at consecutive offsets all aliased element 0 → CTS
   `api.max_shader_storage_blocks`).
3. `rstateConfigurePipelineDescriptor` (MGLRenderer+RenderState.m, shared) no longer replaces
   the vertex function while the geometry domain's pass-through owns the vertex stage: the
   raster pass built a PSO from the program's own vertex shader, which writes `[[point_size]]`
   while `inputPrimitiveTopology` is Triangle → newRenderPipelineState failed → no blending
   into layered attachments (CTS `layered_framebuffer.blending_support`).
4. `bindTexturesToCurrentRenderEncoder` (MGLRenderer.m, shared) binds the fragment/vertex
   stage's **storage images** (`glBindImageTexture`) as `[[texture(n)]]` arguments, including a
   slice view for non-layered bindings of array/3D textures — the render path only bound
   sampled textures, so a fragment `imageStore` never landed (CTS
   `layered_rendering_fbo_no_attachment`).
5. `glFramebufferTexture` (framebuffers.c, shared) rejects `level` above the texture's maximum
   mip level (`tex->num_levels`, exact for texStorage textures) — CTS
   `layered_fbo.fb_texture_invalid_level_number`.
6. `mglDeleteTextures` (textures.c + textures_ext.c + mgl_ext_texext.h, shared/texext) frees the
   texext side-table entry: it is keyed by the `Texture` pointer, so a stale entry made texdata
   refuse a Texture that reuses the address → `glGenerateMipmap` failed with GL_OUT_OF_MEMORY
   only when a previous case in the same context had deleted a texture view (CTS
   `layered_fbo.fb_texture_invalid_target`, order-dependent in a batch).
7. `GL_MAX_GEOMETRY_IMAGE_UNIFORMS = 8` (get.c, shared): the geometry kernel binds image units
   like the compute path → CTS `api.max_image_uniforms` Pass.

transform_feedback group: **18 Pass / 2 Fail** of 20 (single-case runs via one.sh).

8. `xfb_capture_glsl` (transform_feedback.c) re-versions *any* pre-430 source to 460 when it
   contains no pre-140 keywords (new `xfb_has_legacy_keywords`), so the injected std430 capture
   block compiles for the CTS's `#version 130` shaders (6 InternalError cases).
9. `exec_draw_raw` / `validate_vao` (draw_buffers.c, shared) accept **client-memory index
   arrays** (`glDrawElements` with no ELEMENT_ARRAY_BUFFER bound, like Mesa/macOS GL) and never
   cache them; `xfb_process_draw` reads the client pointer too. The CTS transform_feedback
   suite draws with client indices (this was the deinit-crash / InternalError cluster).
10. `mglLinkProgram` (program.c, shared) generates INVALID_OPERATION while any transform
    feedback object — paused or not — holds the program (`mgl_xfb_program_in_use`).
11. `mglTransformFeedbackVaryings` rejects `gl_NextBuffer` / `gl_SkipComponentsN` in
    SEPARATE_ATTRIBS mode and more `gl_NextBuffer` entries than MAX_TRANSFORM_FEEDBACK_BUFFERS.
12. `xfb_draw_from` uses the `glIsTransformFeedback` validity rule (`o->created`), so a merely
    generated name is INVALID_VALUE, not the INVALID_OPERATION of a never-ended object.
13. `geom_var_type` (geometry.c) reports the real matrix type constant (mat2/mat3/mat2x3/
    mat2x4/…/dmat*) instead of always GL_FLOAT_MAT4 — CTS `get_xfb_varying` (introspection of
    geometry-stage varyings).

Regression tests for 1/2/3/4 + geometry image reads are in `tests/headless/test_geom.c`
(`r31_*`). `tests/headless/test_xfb.c`'s error_cases now expects the SEPARATE_ATTRIBS
special-name API error instead of a link failure (spec-correct).

## Known remaining gaps

* `KHR-GL46.geometry_shader.api.max_atomic_counter_buffers` / `api.max_atomic_counters` are
  **NotSupported**: MGL reports 0 atomic-counter buffers for every stage (a program-domain
  feature), so the CTS bails out before running.
* `KHR-GL46.geometry_shader.constant_variables.constant_variables` **fails by choice**: MGL now
  reports MAX_GEOMETRY_IMAGE_UNIFORMS = 8 (the kernel really binds image units; 0 was a spec
  violation for a 4.6 context, it only matched the CTS's ES constant table). Swap the two
  cases by reverting the value in get.c if you prefer the ES-table agreement.
* `KHR-GL46.transform_feedback.draw_xfb_stream_test` / `draw_xfb_stream_instanced_test` fail:
  xfb capture of *streams other than 0* is still not implemented. The multi-stream pack design
  is fully worked out and was reverted only because two unrelated headless tests
  (`xfb_of_geometry_output`, `xfb_of_tes_output`) regressed with it; the complete patch is
  saved in this worktree at `tests/headless/scratch_dts2.c` (the CTS replica test) — see git history of this round for the reverted patch (per-stream record ordinals
  `mgl_pt[4]`/`mgl_cl[4]`, per-stream pack passes driven by `in[20]`/`in[21]`, per-stream
  counter-block slots xfb[9..20], per-stream fold + query note). Re-apply, fix the regression,
  and both CTS cases pass (verified during the round).
* Overflow queries (`GL_TRANSFORM_FEEDBACK_OVERFLOW_ARB`) are single-stream: the multi-stream
  cases of `transform_feedback_overflow_query_ARB` fail with it (2 of 13; the same patch adds
  the needed-vs-written per-stream counters).

## Shared-file edits this round (all surgical, all listed above)

`MGLRenderer.m`, `MGLRenderer+RenderState.m`, `framebuffers.c`, `textures.c`,
`textures_ext.c`, `mgl_ext_texext.h`, `get.c`, `program.c`, `draw_buffers.c`, `mgl_ext_xfb.h`,
`tests/headless/test_xfb.c`. My own files: `geometry.c`, `MGLRenderer+Geometry.m`,
`mgl_ext_geom.h`, `mgl_ext_geom_xfb.h`, `test_geom.c`.
