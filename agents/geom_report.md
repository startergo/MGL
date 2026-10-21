READY edbec60 (packed_pixels 250/250, was 1/250; headless suite green; test_geom 580 checks / 0 failures)

# geom domain — round 33 handoff (packed_pixels cluster)

Branch `gl46/geom`, code tip `edbec60`. Baselines: `test_geom` 580 checks / 0 failures, whole
`tests/headless/run.sh` green (57 binaries), CTS `KHR-GL46.packed_pixels` **250/250 Pass**
(results dirs `~/MGL-cts/geom32-pp10`), and the neighbouring groups re-checked unchanged
(pixelstoragemodes 736 Pass / 160 NotSupported, texture_repeat_mode 162/162, texture_gather
37/75, texture_border_clamp 26/32, texture_buffer 7/16, texture_view 1/7, copy_image 7/13,
texture_swizzle 2/4 — all exactly the round-32 full-pass numbers).

## What this round did

Multi-stream transform feedback (round 32) is merged; the supervisor then assigned the
top CTS failure clusters in size order. **Cluster 1: KHR-GL46.packed_pixels (249 fails → 0).**
Four independent bugs, all in the pixel-transfer validation/conversion (`edbec60`):

1. **table 8.3 exact pairs** (`texdata_formats.c: tdValidateFormatType`): MGL accepted
   RGB/BGR with the 4-component packed types (4_4_4_4, 5_5_5_1, 1_5_5_5_REV, 8_8_8_8(_REV),
   10_10_10_2), BGR with the float-packed types, and RGB_INTEGER/BGR_INTEGER with 8_8_8_8 —
   418 (format,type) combos the CTS proves invalid ("Invalid format used but glTexImage
   succeeded", 148 of the 249 case failures).
2. **operation-aware internal-format check** (`tdCheckFormatCompat`): glTexImage must
   reject STENCIL_INDEX client data outright and refuse to create a stencil internal
   format (GL 4.6 table 8.11 - those are TexStorage-only), SubImage/the clears still allow
   stencil data, and DEPTH_STENCIL client data must also feed a DEPTH_COMPONENT texture
   (the 18 "Valid format used but glTexImage failed").
3. **readPixels pair table** (`framebuffer_ext.c: checkFormatType`): only the component
   *count* was checked for packed types, so (BGR, 5_6_5) etc. were accepted; now the same
   exact table 8.3 sets apply, while the colour branch deliberately stays permissive about
   component counts (RED/GREEN/BGR reads of an RGBA buffer are legal on desktop GL - the
   CTS reads back every non-packed colour pair and expects success).
4. **SNORM conversions** (`f_to_sn`, the readback remap): -1.0 now maps to the most
   negative value (2.3.5.2), so the snorm round-trip of the minimum is exact, and a snorm
   storage read back through an unsigned-normalised client clamps to [0,1] instead of
   biasing by 0.5 (8.5.4 converts through floats) - both were wrong by up to 128/255.
5. **RGTC with uncompressed client data**: glTexImage(internalformat = COMPRESSED_*_RGTC1/2,
   uncompressed data) now stores the format's uncompressed equivalent (GL 4.6 8.5
   "compressed if possible"). I first wrote a real BC4/BC5 block encoder
   (`td_rgtc.c`, self-consistent unit test) but the CTS compares gradients at 8-bit
   precision with a 1/127 epsilon while BC4 on a ~145-unit-per-block ramp loses ~10/255 -
   mathematically impossible to pass with real compression, so the encoder was dropped for
   the storage substitution (what drivers without a compressor do). BPTC/ETC2/S3TC still
   reject uncompressed uploads (documented gap; needs those encoders).

`tests/headless/test_pixelformats.c` (the texdata agent's sweep) was updated to the same
rules: the D4 sweep now skips the table-8.3-invalid combos and expects the clamped (not
biased) snorm readback; 1292 combos, 0 failures.

# geom domain — round 32 handoff (multi-stream transform feedback)

Branch `gl46/geom`, code tip `f22b93a` (merge of gl46-port `64a3837` + the multi-stream xfb patch).
Baselines: `test_geom` 580 checks / 0 failures, whole `tests/headless/run.sh` green (57 binaries),
CTS `KHR-GL46.transform_feedback` **20/20 Pass** (`~/MGL-cts/mgl-results`).
(Merged to gl46-port as 16e8498; round 33 continues above.)

## What this round did (all committed)

1. **Streams 1–3 are captured** (`f22b93a`, re-applies the reverted round-31 design, fixed):
   the geometry-output pack runs one prefix + scatter pass per captured stream. The tags the
   geometry kernel stamps in its records now carry *per-stream* captured-primitive ordinals
   (`mgl_pt[4]` / `mgl_cl[4]` replacing the single `mgl_written` / `mgl_closes` counter), so
   each stream's captures are numbered independently and land in the buffers its varyings
   name. Changes vs. the reverted `.ref` (these are what fixed the two regressions):
   * the scatter's capacity test checks only **this pass's** buffers (`in[22]` mask) — the
     `.ref` tested all four, so a stream-0 pass could drop primitives because a stream-1
     buffer was full under stream-0 numbering;
   * the passes share one compute encoder held in a **strong local** `enc` (the `.ref` parked
     it in `static void *gXfbEnc` with a plain `__bridge` cast — a use-after-release across
     loop iterations);
   * the counter block grew **per-stream block totals** (vertices [21..24], primitives written
     [13..16], primitives emitted ("needed") [17..20]); the per-draw slots [9..12] are cleared
     by the finish kernel of the draw that produced them, so multi-draw blocks no longer
     double-count (the `.ref` folded only the last draw's per-draw value);
   * the prefix pass atomically adds its scan total (the stream's emitted primitives) into
     `xfb[17+st]`, which the resolve feeds to the queries domain as "needed" — this makes the
     multi-stream `transform_feedback_overflow_query_ARB` cases count correctly too.
2. **Tess-path finish kernel per-draw fix**: `xfb[8]` (its per-draw vertex counter) is cleared
   after the offsets advance; two tess draws in one transform feedback block used to
   double-advance the write offsets. It also mirrors its counts into the stream-0 block-total
   slots the resolve reads.
3. `MGLXfbVarying` (mgl_ext_xfb.h) gained `stream`; `geom_xfb_resolve` copies it from the
   geometry stage's `MGLGeomVar.stream` (SPIR-V `Stream` decoration), and `mgl_geom_xfb_flush`
   folds per-stream vertices / written / needed into the object, the context totals, and the
   per-stream query buckets (`note(ctx, s, written, needed)` for all four streams).
4. New tests `xfb_multi_stream_capture` and `xfb_multi_stream_instanced` (the CTS
   `draw_xfb_stream{,_instanced}_test` shape: per-stream indexed queries, captured contents in
   emission order, `glDrawTransformFeedbackStream[Instanced]` rasterizing each stream), plus a
   `TEST_ONLY=<name>` filter in test_geom's `main` for debugging state cascades.

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
* Streams whose captures **interleave emission** (EmitStreamVertex(0) / EmitStreamVertex(1)
  alternating inside one open strip) share one strip buffer in the kernel, so the two strips
  are conflated — the CTS never interleaves, and GL only rasterizes stream 0, so only the
  captured contents of such a shader would be wrong.
* The CPU-pack fallback (`MGL_GEOM_CPU_XFB=1`) still captures stream 0 only; the GPU pack is
  the default path.

## Shared-file edits this round (all surgical, all listed above)

Round 32: `mgl_ext_xfb.h` (the `stream` field of `MGLXfbVarying`) — my files otherwise.
Round 31: `MGLRenderer.m`, `MGLRenderer+RenderState.m`, `framebuffers.c`, `textures.c`,
`textures_ext.c`, `mgl_ext_texext.h`, `get.c`, `program.c`, `draw_buffers.c`, `mgl_ext_xfb.h`,
`tests/headless/test_xfb.c`. My own files: `geometry.c`, `MGLRenderer+Geometry.m`,
`mgl_ext_geom.h`, `mgl_ext_geom_xfb.h`, `test_geom.c`.
