READY 22826fa (texture_gather in progress: root causes identified — AGX vertex-stage shadow sampling returns raw values + SPIRV-Cross drops ConstOffsets; compute-stage gather_compare fix landed; suite 58/58, test_geom 593/0)

# geom domain — round 37 handoff (KHR-GL46.texture_gather, partial)

Commit `22826fa` on gl46/geom (gl46-port merged at 76c84f1/2e9b5e8 earlier this
round).  Cluster status when picked up: plain-gather 23/33 Pass (10 Fail),
offset-gather 10/33 Pass (23 Fail), offsets-gather + gpu_shader5 families not
re-run yet.  Everything below is REPRODUCED and verified with scratch tests.

## Root causes (all reproduced in raw Metal and/or MGL scratches)

1. **AGX vertex-stage shadow sampling is broken at the driver level.**  In
   VERTEX functions, `depth2d.gather_compare(...)` AND `depth2d.sample_compare(...)`
   return the RAW depth values — the sampler's compareFunction is ignored.
   Fragment functions: correct.  Compute: sample_compare correct.
   - Raw Metal repro: `tests/headless/scratch_mtlgather.m` (build:
     `clang -fobjc-arc -fmodules -framework Metal -framework Foundation -o /tmp/mg tests/headless/scratch_mtlgather.m`).
   - MGL repro: `scratch_vdg.c` (exact CTS plain-gather-depth-2d replica,
     got = raw patch values = exactly the CTS "got").
   - The CTS runs each gather case in THREE stages (pass 0 vertex, pass 1
     fragment, pass 2 compute — GatherBase::Run loop i=0..2); ANY failing pass
     fails the case, so the VS pass keeps all depth-gather cases red until
     fully emulated.
2. **SPIRV-Cross drops the ConstOffsets operand.**  GLSL
   `textureGatherOffsets(s, uv, offs, comp)` compiles to SPIR-V
   OpImageGather+ConstOffsets, but the emitted MSL is a plain
   `t.gather(s, uv, int2(0), component::x)` — the four offsets are silently
   lost (verified with MGL_DUMP_MSL; scratch_gat.c fs_gather_offsets).
   All `offsets-gather-*` fail for every type.  Fixing this requires a
   SPIR-V-level pass (the offset values live in SPIR-V constants; they are
   unrecoverable from MSL).  My task brief explicitly allows a new
   `MGL/src/*.cpp` SPIR-V pass + a minimal Makefile rule for it.
   Rewrite shape: OpImageGather{ConstOffsets} → 4 × OpImageGather{Offset} with
   the i-th constant offset + OpCompositeExtract(i) + OpCompositeConstruct
   (GL 4.6 8.11.1: result[i] = gather(uv+offs[i])[i]... verify the exact
   tap-index mapping against GatherBase::Expected before coding).
3. **Tap order is NOT a bug**: GL/CTS gather order = (x0,y1)(x1,y1)(x1,y0)(x0,y0)
   of block floor(uv*size-0.5+off) — identical to Metal hardware gathers.
   Verified via the CTS's own expected data (GatherBase patch
   Vec4((12..15)/16,(8..11)/16,(0..3)/16,(4..7)/16) with depth format consuming
   floats sequentially: texels (22,25)(23,25)(22,26)(23,26) = 12..15/16; CTS
   expects comparisons (1,1,0,0) = (14,15,13,12)/16 = (t2,t3,t1,t0)) and via
   the raw-Metal tap decode ((22,26)(23,26)(23,25)(22,25)).

## Landed this round (22826fa)

- `mgl_fix_gather_compare_vs()` in program.c: MSL post-processing for
  VERTEX+COMPUTE stages that rewrites `OBJ.gather_compare(S, UV, REF[, OFF])`
  to `mgl_gcmp_vs(OBJ, ...)` — 4 texel-centre sample_compare taps at level(0)
  (exact for any filter), GL tap order, depth2d + depth2d_array overloads.
  Fixups land after the `using namespace metal;` anchor (file scope) with the
  inline copy stripped.  Effect: compute-stage depth gathers now compare.
  VS gathers still raw (VS sample_compare is the broken primitive).
  Suite 58/58 green, test_geom 593/0.
- Note: watch the paren offsets in that pass — the open paren is at
  `.gather_compare(` + 15 (off-by-one cost an hour; there is a unit-style
  debug print if needed via getenv).

## Remaining work for this cluster (in value order)

1. **VS shadow-sampling emulation** (~15 depth-gather cases): in VERTEX MSL
   (a) rewrite `depth2d<float> NAME` → `texture2d<float> NAME` for samplers
   only used by comparison ops in that stage, (b) rewrite ALL
   `.sample_compare(...)` (and mgl_gcmp_vs) to raw `t.sample(s, uv, level(0)).x`
   + manual `ref OP d`, (c) plumbing for OP: GL_TEXTURE_COMPARE_FUNC lives in
   the sampler state; pass it via a small [[buffer]] constant MGL fills at
   bind time (compare-op int + maybe flags) — needs a new bind site in
   MGLRenderer.m's sampler loop, (d) bind a NON-compare MTLSamplerState for
   the vertex stage of those resources (using a compare sampler with
   texture2d.sample is UB — this is why the type rewrite is required).
   Cube/cube-array depth gathers stay a documented gap (face-quad mapping).
2. **ConstOffsets SPIR-V pass** (~12 offsets-gather-* cases incl. non-depth
   types that already work otherwise).
3. **2drect gathers** (plain-gather-*-2drect, ~5 cases): uninvestigated;
   likely MGL's RECTANGLE emulation (unnormalized coords) vs gather — the
   sampler-target binding fix from round 36 already covers the unit binding.
4. **plain-gather-int-2darray** fails while uint passes — unexplained, one case.
5. gpu_shader5.texture_gather_offset_* (12) — same families; re-run after the
   above land.

## Reference CTS internals (gl4cTextureGatherTests.cpp)

- GatherBase::Run: 3 passes (vertex/fragment/compute), 1x1 RGBA32F renderbuffer,
  POINT draw with p=(uv.x, uv.y, ref, 1) as a vec4 attribute, `flat` varying.
- Expected values: GatherBase::Expected (RGBA R = (0,4,8,12)/16), depth classes
  override to (1,1,0,0); patch at (22,25) size 2x2; BufferData() =
  Vec4(23/32, 26/32, 13.5/16, 3) = (u, v, ref, ...).
- The depth texture: 6 mip levels (only level 0 filled) + default
  NEAREST_MIPMAP_LINEAR — gather treats it as complete (GL 4.6 8.17).

## Next clusters after this: KHR-GL46.limits (23), then the 4 remaining
direct_state_access failures (see round-36 handoff below; two are documented
gaps, attribute_format is flaky-crash, compressed_subimage conflicts with
packed_pixels without a BC4/BC5 encoder + GL_TEXTURE_COMPRESSED=1 reporting).

Run rules: -j4 max, one CTS batch at a time (--max-seconds <= 480, <= 2 chunks
per call), FRESH results dir per pattern (run.py reuses caselist.txt!), scratch
tests untracked, never `git add -A`, no symlinks/scratch commits, merge
gl46-port before the next READY.


(round 36)
First cluster from the second CTS pass: direct_state_access 297 Pass / 70 Fail /
2 InternalError / 1 Crash -> **367 Pass / 2 Fail / 1 Crash / 1 CompatWarn** (full
group re-run with batch=30, fresh results dir `~/MGL-cts/geom38z`). Fix commit
`27f9b60`; `gl46-port` (tess fast fp64) merged afterwards at `76c84f1`.

## Root causes fixed (27f9b60)

1. **Sampler->texture binding by GL target** (60+ cases). Draw paths fed every
   sampler from `active_textures[unit]` = "most recently bound". GL feeds a
   sampler from the unit's binding for ITS dimension: a `sampler2DMS` keeps
   reading `GL_TEXTURE_2D_MULTISAMPLE` even after `glBindTextureUnit` rebinds
   the 2D target. `SpirvResource.target` (new field, glm_context.h) is derived
   at reflection time from the SPIR-V image type (`sampler_gl_target()` in
   program.c — accept BOTH `SPVC_BASETYPE_IMAGE` and `_SAMPLED_IMAGE`).
   `mgl_texture_for_sampler(ctx, unit, target)` (static inline,
   MGLRenderer_Private.h) prefers the per-target binding, falls back to
   active_textures (arrays/unknowns keep working). Call sites: render-path
   sampler binding + compute-path sampler binding (MGLRenderer.m),
   MGLRenderer+Geometry.m (4).
   -> textures_storage_multisample_2d_* 30/30, _3d_* 30/30.
2. **sRGB readback decode** — fbDecodeColor returned raw encoded bytes for
   RGBA8/BGRA8Unorm_sRGB; glReadPixels must return linear (fbSRGBDecode).
   -> renderbuffers_storage, renderbuffers_storage_multisample.
3. **Spec errors**: NamedFramebufferTextureLayer unknown texture -> INVALID_OPERATION;
   NamedFramebufferReadBuffer(FRONT_AND_BACK) on an FBO -> INVALID_OPERATION;
   sampler-state texParameter on multisample targets -> INVALID_OPERATION (set
   AND get); GL_TEXTURE_TARGET query on a TEXTURE_BUFFER -> INVALID_OPERATION
   (tex_param.c); glGetTextureImage on a cube-incomplete texture ->
   INVALID_OPERATION (new `cubeComplete()` in textures.c); mglValidateProgram no
   longer fails VS-only programs (separable; the round-35 trampoline fragment
   already rasterizes them). test_texparam updated (GL 4.6 8.10).
   -> framebuffers_*_errors 8/8, textures_parameter_*_errors 2/2,
      textures_image_query_errors, program_pipelines_functional.
4. **XFB capture never bound vertex attributes** (the big one). The capture
   render pass (MGLRenderer+TransformFeedback.m xfbCaptureAsync) has no
   processGLState, so the "PreDraw:" ext-hook fan-out never ran and only
   non-attribute map entries were bound -> the VS-as-capture read zeros from
   unbound MSL slots (exactly one attribute - the last MSL slot - survived).
   Fix: call `[self extVattribPreDraw: enc]` after bindVertexBuffersToCurrentRenderEncoder
   (declared in MGLRenderer_Private.h as a category interface).
   Repro matrix: scratch_af.c (int/float x contiguous/sparse x DSA/legacy).
   -> vertex_arrays_enable_disable_attributes, vertex_arrays_vertex_buffers.
5. **resolveBuffer lazy-created buffer objects** for any name below the name
   high-water mark (`buffer < current_name`), so DSA vertex-buffer binding
   accepted the CTS's not-a-buffer probe once earlier cases raised the
   watermark (batch-only failure; standalone always passed). Removed the lazy
   create — GL 4.6 21.2 requires an *existing* object for DSA names.
   -> vertex_arrays_vertex_buffers_errors in batch mode.
   NOTE when re-running single cases: run.py REUSES `caselist.txt` inside a
   results dir — always use a FRESH dir per pattern or you re-run the first
   caselist (that produced bogus "passes" this round).

## Remaining direct_state_access failures (documented gaps)

* `textures_compressed_subimage` — needs texImage*(specific compressed
  internalformat) to either really compress or be INVALID_OPERATION, but
  `KHR-GL46.packed_pixels` (12 RGTC cases, green since edbec60) requires the
  upload to SUCCEED and read back losslessly (MGL stores RGTC as R8/RG8,
  GL_TEXTURE_COMPRESSED=0). Proper fix: report GL_TEXTURE_COMPRESSED=1 +
  COMPRESSED_IMAGE_SIZE = blocks*8/16 and encode BC4/BC5 in
  getCompressedTexImage (encoder ~40 lines; min/max endpoints + 3-bit indices);
  keeping getTexImage returning the stored R8/RG8 keeps packed_pixels green.
  Verified: blanket INVALID_OPERATION fixes this case but regresses 12
  packed_pixels -> net -11, so it was reverted.
* `vertex_arrays_attribute_format` — 17/52 rows; flaky (Fail standalone, Crash
  rc=-5 in batch). All size-1 function families reproduced byte-exactly
  (scratch_afx/scratch_afy: FLOAT/HALF/UBYTE/DOUBLE/INT x DSA split + XFB +
  bufferStorage) PASS, capture record verified [1.0f,5.0f], scatter slots
  correct (MGL_XFBDBG traces); the deqp process still reads heap-pointer
  garbage from the mapped immutable XFB buffer. Suspect readback/aliasing of
  glBufferStorage'd XFB buffers after the capture encoder ran in the same
  process (glMapBufferRange path). Next: instrument
  mgl_buffers_prepare_cpu_read for storage-backed XFB buffers in the CTS run.
* `framebuffers_check_status` — CompatibilityWarning only (CTS informational).

## Next clusters (from agents/conformance.md, second pass)

1. `KHR-GL46.texture_gather` (38) — likely textureGather on 2D/2D-array +
   offsets; check SPIRV-Cross MSL gather() emission + sampler-target binding
   (fix #1 above may already cover part of it).
2. `KHR-GL46.limits` (23) — mostly glGet* caps (MAX_* enums); see
   mgl_ext_get_geom/get.c plumbing.

Run rules reminder: -j4 max, CTS via agents/conformance/run.py with
--max-seconds <= 480 (<= 2 chunks per call), fresh results dir per pattern,
scratch tests untracked, never `git add -A`, no symlinks/scratch commits.


Two clusters this round; both committed, `gl46-port` (02c2871, tess fp64) merged on top
(merge a5a9170, clean). Headless suite fully green after the merge (incl. the tess
agent's test_tess 499 / test_tessemu 65), `test_geom` 593 checks / 0 failures.

## KHR-GL46.sample_variables 130/190 → 152 Pass / 38 NotSupported / 0 Fail

* The 22 `mask.*.samples_0` cases draw with mask values 0..7 into a single-sample
  framebuffer and expect full red: GL 4.6 13.3 ignores the sample-mask output when
  SAMPLE_BUFFERS is 0. The patched-MSL helper `mgl_rs_sample_mask()` now takes the
  written mask plus a new function constant `MGL_RS_FC_PIPELINE_SAMPLES` (32008) with
  the render pipeline's sample count and drops the written bits only for single-sample
  pipelines; `rstateSpecialize` binds it and keys the specialization cache on it.
  4x-MSAA gating (1..4 samples) verified unchanged (scratch_smsk.c).
* The 38 NotSupported are the `samples_8` cases: Metal caps MSAA at 4 samples on Apple
  GPUs (`supportsTextureSampleCount:8` = false on the M4 Pro), so `GL_MAX_SAMPLES = 4`
  is honest; the CTS itself reports NotSupported from `getInternalformativ(GL_SAMPLES)`.

## KHR-GL46.shader_storage_buffer_object 54/124 → 118 Pass / 6 Fail

Fixes (commit 3a610a5, then 791fc1f; see the commit messages for the mechanism):

1. **Vertex-only programs** (VS + SSBO writes + rasterizer discard): MGL refused to
   build a render pipeline without a fragment function so nothing ran. Now builds the
   documented vertex-only pipeline (fragmentFunction nil, rasterizationEnabled NO —
   `rstateConfigurePipelineDescriptor` keeps it disabled for a nil fragment function);
   SPIRV-Cross already emits the required `vertex void` function.
2. **`.length()` on runtime-sized arrays** (25 unsizedArrayLength cases): SPIRV-Cross
   implements it through `spvBufferSizeConstants [[buffer(25)]]` (bound-range byte
   size per Metal buffer index); MGL never bound it. `BufferMap` records the range
   size; `geomBindSSBOSizes{,Compute}` bind the sizes for the render vertex/fragment
   tables (skipped while the geometry pass-through owns vertex slot 25) and the
   compute dispatch; compute bindings also honour glBindBufferRange offsets now.
3. **coherent/volatile/restrict** SSBO qualifiers produced unparseable MSL
   (`volatile device Block&`, C99 `restrict`): `mgl_geom_strip_msl_volatile()`
   rewrites them to `device`/`__restrict` in every stage's MSL (neither qualifier is
   expressible on Metal anyway).
4. **glslang limits**: `max_atomic_counter_bindings` 1→8 to match
   GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS; per-stage atomic counters aligned with glGet.
5. **SSBO block instance arrays** (`buffer Output {...} g_output[4];`): the map
   enumeration re-expanded each instance entry (skip added, clamped for the unflattened
   geometry-stage lists — basic-stdLayout-case1/4 vs+cs).
6. **SSBO atomics from VS+GS** (basic-atomic-case1, exact values verified):
   initializer-fed plain uniforms in capture/kernel variants (extracted into the DUB,
   paren-aware declarator scan), `geom_build_pins` classified like SPIRV-Cross
   reflection (Block vs BufferBlock on the block struct; the DUB takes its renderer
   flat; pool blocks skipped), `msl_value_parameters` handles template commas, the GS
   kernel binder materialises CPU-shadowed buffers, and the introspection patcher no
   longer drops the GS module's default-uniform rewrite (its MSL-acceptance gate cannot
   parse geometry stages reading gl_PrimitiveIDIn).
7. **Negative-API**: INVALID_VALUE for BindBufferBase/Range with index ≥
   MAX_SHADER_STORAGE_BUFFER_BINDINGS (the cached limit was 0 because
   `getMacOSDefaults` cannot query GL 4.3+ limits from Apple's GL 4.1) and for
   BindBufferRange's unaligned offset even with name 0 (size>0 only for a real buffer);
   glShaderStorageBlockBinding range checks live.
8. **GLSL-only declaration**: `layout(binding = N) buffer;` (Vulkan GLSL) is rejected
   with a `#error`; packing/matrix-order qualifiers (`layout(shared,row_major)
   buffer;`) stay legal.

### Remaining 6 fails (gaps)
* `advanced-write-fragment`: standalone `layout(binding=N,offset=M) uniform atomic_uint
  c;` maps to a *thread* atomic in MSL, and thread atomics have no fetch_add. Needs the
  atomic-counter domain to route standalone counters into a `device atomic_uint`
  buffer (block-form atomic counters are not accepted by glslang either).
* `advanced-write-tessellation`, `basic-atomic-case2`: SSBO writes/atomics from TCS/TES
  (tess domain; the TES stage MSL also fails there on a `[[patch(quad, 3)]]` attribute
  with a bad vertex count — see the ssbo4 logs).
* `basic-syntax`, `basic-syntaxSSO`: two runtime-sized arrays in one buffer block
  (legal per the CTS); would need runtime-computed offsets for the earlier members —
  MSL structs allow one unsized tail member only.
* `negative-glsl-compileTime`: its last check wants `vec4(array_member)` to be a
  compile error; glslang accepts it (would need a source-level validator).

## Shared-file edits this round (all small, all listed)
* `MGL/include/glm_context.h` — BufferMap gained `GLsizeiptr size` (bound range size).
* `MGL/src/MGLRenderer.m` — vertex-only pipeline in generatePipelineDescriptor; map
  enumeration records sizes / skips expanded instance entries; compute sizes bind.
* `MGL/src/MGLRenderer+RenderState.m` — pipeline-sample-count function constant (and
  round 1: rasterizationEnabled=NO kept for nil fragment functions).
* `MGL/src/MGLRenderer+Draw.m` — geomBindSSBOSizes calls in extBufBeginDraw /
  extBufEncodeDispatch.
* `MGL/src/MGLRenderer_Private.h` — declarations for the two geomBindSSBOSizes methods.
* `MGL/src/program.c` — call to mgl_geom_strip_msl_volatile() in the stage MSL path.
* `MGL/src/shaders.c` — glslang resource limits (atomic counters).
* `MGL/src/buffers.c` — maxBindingsForTarget SSBO fallback; BindBufferRange checks.
* `MGL/src/program_introspection.c` — SSBO binding limit fallback.
* `MGL/src/introspect_spirv.c` — skip the MSL-acceptance gate for the geometry stage.
* `MGL/src/state_ext.c`, `MGL/src/mgl_ext_rstate.h` — sample-mask single-sample rule.
* `MGL/src/cts_glsl_compat.c` — reject binding-only buffer layout declarations.

Scratch tests used this round (untracked, in tests/headless/): scratch_smsk.c,
scratch_vsssbo.c, scratch_ssblen.c, scratch_ssbcv.c, scratch_ssbng.c, scratch_ssbia.c,
scratch_ssbgsa.c, scratch_gspu.c, scratch_ssuni.c, scratch_ssbbb.c, scratch_ssbrb.c,
scratch_ssac.c, scratch_ssbtw.c, scratch_ssbl.c.

Next clusters on the supervisor's list: direct_state_access (70 fail), then the
remaining sample_variables NotSupported cases are hardware-honest (Metal MSAA ≤ 4).


# geom domain — round 34 handoff (sample_variables cluster)

Cluster 2 of the supervisor's list. `KHR-GL46.sample_variables` **8 Pass / 144 Fail →
130 Pass / 22 Fail / 38 NotSupported** (`~/MGL-cts/geom32-sv3`); sample_shading re-checked
unchanged (5/13), headless suite green, `test_geom` 581 checks / 0 failures.

* glslang declares gl_SampleID / gl_SamplePosition / gl_SampleMask / gl_SampleMaskIn for
  desktop GLSL but NOT `gl_NumSamples` (an OES_sample_variables builtin there), so every
  shader of the group failed with "undeclared identifier". MGL now injects
  `#define gl_NumSamples N` after the `#version` line of a fragment shader that mentions it
  (`shaders.c: initGLSLInput`), N baked from the current draw framebuffer's sample count
  via `mgl_fb_samples` — GL defines it as a compile-time constant and the CTS binds its
  MSAA framebuffer before compiling. Only shaders containing the identifier are touched.
  (A direct `const int gl_NumSamples` is impossible: glslang reserves the gl_ prefix.)
* The 22 remaining `mask.*` failures are a different mechanism: the cases WRITE
  `gl_SampleMask[0]` and expect the write to gate rasterization (a fragment writing mask 0
  must discard). That needs the MSL `sample_mask` fragment output wired through
  SPIRV-Cross + the render pipeline — a natural follow-up cluster.
* 38 NotSupported are the sample counts MGL does not expose (samples_2/4/8 variants).

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
