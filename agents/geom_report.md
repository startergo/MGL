READY 1206ece (gl46-port merged at 76e075e + the four r48 commits; clean rebuild, full headless suite **495 PASS / 0 FAIL**, test_geom **603/0 x8**; r48 CTS groups re-verified with before/after table below; no zz*/scratch_* committed).

## Round 49 — fresh session: merge, clean rebuild, WIP triage, r48 CTS re-verification

Round 48 hit the transcript limit before READY; this round landed the leftover work and
verified everything against the r18 sweep baseline (`~/MGL-cts/mgl-r18-sweep/summary.txt`).

### Uncommitted round-48 WIP triage (kept, finished, or reverted)
* `MGL/src/shaders.c` (kept): glslang compile-cache teardown in
  `mgl_shader_release_compiler_resources` — cache-owned `glslang_shader_t` handles must not
  outlive `glslang_finalize_process` (exit-order crash in test_debug teardown); plus an
  `MGL_NO_GLSL_CACHE` env kill-switch for debugging.
* `MGL/src/get.c` (kept + fixed properly): the round-48 `GL_COMPRESSED_TEXTURE_FORMATS`
  patch truncated the list to the 32-slot `GetValues` array while
  `GL_NUM_COMPRESSED_TEXTURE_FORMATS` still reported the true 52 — an app asking for
  NUM and then the list would have read 20 uninitialised values (and the unpatched code
  wrote 52 doubles through a 32-slot struct = stack corruption). `MAX_GET_VALUES` is now
  64 so the full 52-entry list is delivered and both queries agree; the list query still
  clamps defensively to the slot count.
* `tests/headless/test_fb.c` (kept + fixed): the RGB9_E5-storage expectation matches the
  committed r48b behaviour (sized non-renderable formats are legal storage; CTS
  internalformat.renderbuffer.rgb9_e5 expects no error, completeness fails later), but the
  storage call was made on the **attached** renderbuffer, which re-specified its storage to
  RGB9_E5 and left the FBO incomplete for the readback error checks further down. Now done
  on a spare renderbuffer.
* `tests/headless/test_texdata.c` (not part of the WIP, but the suite was red because of
  r48d): `copy_image_sub_data` still assumed the pre-r48d CopyImageSubData semantics.
  The single-level textures now clamp `GL_TEXTURE_MAX_LEVEL` to 0 (r48d rejects incomplete
  textures per GL 8.17, matching KHR-GL46 copy_image.incomplete_tex which passes), and the
  target-mismatch expectation is `GL_INVALID_ENUM` (KHR-GL46 copy_image.target_miss_match).
* `MGL/src/MGLRenderer+Program.m` (reverted): whitespace-only `{`/`@autoreleasepool`
  reflow, no content.

### CTS before/after (r18 sweep -> now, one batch at a time, nice/taskpolicy)
| group | r18 sweep | round 48 | round 49 (now) |
|---|---|---|---|
| KHR-GL46.internalformat | 25 Pass / 27 NotSupported | 48 Pass / 4 Fail (48d) -> 51 Pass + 1 NS (48e/f) | **51 Pass / 1 NotSupported** |
| KHR-GL46.pixelstoragemodes | 736 Pass / 160 NotSupported | 896 Pass (48c) | **896 Pass / 896** |
| KHR-GL46.packed_pixels | 226 Pass / 24 Fail | 234 Pass / 16 Fail (48j) | **234 Pass / 16 Fail** |
| KHR-GL46.packed_depth_stencil | 21 Pass / 4 Fail | 25 Pass (48p) | **25 Pass / 25** |
| KHR-GL46.copy_image | 7 Pass / 6 Fail | 10 Pass / 2 Fail / 1 Timeout (48u) | **12 Pass / 1 Fail** |
| KHR-GL46.texture_swizzle | 2 Pass / 1 Crash / 1 Fail | 2 Pass / 2 Timeout (48a) | 2 Pass / 2 Timeout |
| KHR-GL46.texture_view | 1 Pass / 6 Fail | 3 Pass (r47) | **3 Pass / 4 Fail** (unchanged) |
| KHR-GL46.texture_buffer | 14 Pass / 2 Fail | 15 Pass (r47) | **15 Pass / 1 Fail** (unchanged) |

Notes on the remaining failures:
* **copy_image.exceeding_boundaries** (the 1 Fail): CTS sub-case for `GL_TEXTURE_1D_ARRAY`
  — the test allocates a 16-**layer** 1D array (`texImage2D(16,16)`) and then expects
  `GL_INVALID_VALUE` for a copy at `srcY=14, srcHeight=1`, i.e. layer 14 of 16. Per GL
  4.6 (y is the array layer, region fits) that copy is legal; the CTS models the 1D-array
  layer axis as 6 deep (the `depth` constant it uses for 2D_ARRAY/3D). Rejecting it would
  break legal copies, so MGL keeps the spec behaviour and the sub-case stays red.
  The two r48d failures (incompatible_formats_compression, invalid_alignment) now pass and
  `functional` passes with a longer per-case timeout (it was a 305 s deadline kill, not a hang).
* **texture_swizzle.smoke/.functional** (2 Timeout): not a driver hang — the smoke case
  walks ~131k sub-cases (8^4 swizzle combos x 4 channels x access routines), each with a
  fresh output texture + render + readback, and needs far more than the 20 s default
  per-case budget (killed at 240 s with the test still making progress, ~1370 textures in).
  A per-sub-case cost reduction (shader/sampler reuse across the swizzle loop) would be the
  textures-domain follow-up.
* **internalformat.renderbuffer.rgb9_e5** (1 NotSupported): correct per CTS semantics —
  storage succeeds, RGB9_E5 is not colour-renderable, so the FBO is incomplete and the
  case reports NotSupported (allowed outcome).
* packed_pixels 16 Fails: 8 "Gradient comparison failed ... GL_BGRA / UNSIGNED_INT_10_10_10_2"
  readback conversions, 4 UNPACK_ROW_LENGTH + 4 UNPACK_IMAGE_HEIGHT on sRGB/RGTC uploads
  (same clusters as r48; no regression).

### Verification
* clean rebuild (`rm -rf build`, `make -j4 lib`), no new warnings in the touched files.
* full headless suite **495 PASS / 0 FAIL** (rc=0; 485 -> 495 tests with the merged
  compat s9 + tess r19 test additions), run twice.
* `test_geom` **603 checks / 0 failures x 8 runs**; `test_fb` 293/0 x4, `test_texdata` 240/0 x4.
* `agents/stub_scanner.py --list EMPTY|NOOP|ASSERT` clean for geometry.c / shaders.c / get.c.

### Shared-file edits this round
* `MGL/src/get.c` (get domain): MAX_GET_VALUES 32 -> 64 + list clamp (stack-corruption fix,
  described above).
* `tests/headless/test_fb.c` (fb), `tests/headless/test_texdata.c` (texdata): expectation
  updates required to match the already-committed r48b/r48d semantics; both called out above.


## Round 47 — SSBO 119 -> 124/124, texture_view +2, texture_buffer +1

### KHR-GL46.shader_storage_buffer_object 124/124 (was 119)
Fixed: basic-atomic-case1/2/3, advanced-write-tessellation, basic-syntax, basic-syntaxSSO,
negative-glsl-compileTime. Four root causes:

1. **TES SSBO slots were clobbered by the vertex array.** With tessellation active the
   render pipeline's vertex function is the (rewritten) evaluation shader, whose UBO/SSBO
   arguments start at Metal buffer slot 0; `extVattribPreDraw` still bound the vertex
   array's attribute streams (and the const-attrib ring at slot 29 = MGL_TESS_TES_PATCH)
   onto that encoder. The evaluation shader's storage bindings were overwritten before the
   draw — with an SSBO at slot 0 its writes went into the vertex buffer of attribute 0 and
   `glGetBufferSubData` read the untouched GL buffer. Fix: `MGLRenderer+VertexAttrib.m`
   skips the vattrib render binds (streams + shadow + constants) when
   `mgl_tess_active_program(ctx)` is set — the VS kernel already consumes the attributes
   through `tessBindVSAttributes` on the compute encoder (that path is unchanged).
2. **Small SSBOs were streamed through the argument ring.** A 4-byte SSBO bound at the
   vertex stage took the `compatRingVertexBytes` fast path (read-only ring), so shader
   writes landed in ring memory. Fix: `bindVertexBuffersToCurrentRenderEncoder` /
   `bindFragmentBuffersToCurrentRenderEncoder` force a real aliasing MTLBuffer for
   bindings whose Buffer is on a GL_SHADER_STORAGE_BUFFER / atomic-counter binding point
   (new local helper `mgl_writable_binding`, same predicate as extBufStorageBinding).
3. **glslang's implicit array growth breaks buffer-block offsets** (basic-syntax /
   basic-syntaxSSO): an unsized non-final buffer-block member is sized by the largest
   constant index that touches it (`updateImplicitArraySize`) but the offsets of the
   members that follow it are not recomputed, so the block self-overlaps. New
   `mgl_spirv_fix_block_offsets()` (geometry.c, SPIR-V walk over the interface-block
   structs) pushes every member that starts inside the previous member's declared array
   span (constant length x declared ArrayStride) past that span; anything glslang laid
   out consistently is reproduced exactly (arrays only — no second-guessing of matrix or
   struct layouts). Hooked in program.c after the Rect->2D pass, MSL path only.
4. **Compile-time layout rules glslang does not check for OpenGL clients**
   (negative-glsl-compileTime): `layout(binding >= MAX_SHADER_STORAGE_BUFFER_BINDINGS)
   buffer ...` / `binding + block-array size > max` and `layout(std430) uniform Block`
   are rejected with a `#error` line in cts_glsl_compat.c (new `reject_ssbo_binding_range`
   and `reject_std430_uniform_block`). MGL-authored capture variants (geometry.c
   `geom_glsl_to_spirv`) call the new `mgl_cts_glsl_compat_generated()` which skips the
   two rejections — their blocks sit at MGL's own renumbered binding points (21+).

Also added `MGL_SPIRV_DUMP` (per-stage SPIR-V dump) and `MGL_BINDMAP_DEBUG` /
`MGL_VIEW_DEBUG` env-gated diagnostics.

### KHR-GL46.texture_view 3/7 (was 1) and texture_buffer 15/16 (was 14)
* `errors`: a name never handed out by GenTextures is GL_INVALID_OPERATION (was
  INVALID_VALUE), and the `<numlayers>` target rules are validated on the caller's value
  *before* the view is clamped to the parent's layer count (a 2D view must pass 1).
* `gettexparameter`: a texture view reports the ORIGINAL's GL_TEXTURE_IMMUTABLE_LEVELS —
  `mgl_texparam_record_view()` gained the parent's level count (new record field
  `view_immutable_levels`, used by both query paths).
* `view_classes`: texel-buffer / view level descriptors now carry their internal format
  (`texext_alloc_levels` grew a `fmt` argument), so `levelFormat()` is no longer 0 and
  TexSubImage* into a view / texel buffer is accepted.
* texture_buffer `parameters`: GL_TEXTURE_BUFFER_SIZE follows the data store of the bound
  buffer (a resize after glTexBuffer shows up in the query without rebinding); a ranged
  binding keeps its range while the store still matches it.
* image-atomic sidecars are now **seeded** from the image's contents: `geomPrepareImageAtomicsCompute`
  creates the dispatch's sidecars and runs the new seed kernel (geomAtomSeedPso, the
  resolve kernel's mirror) in its own compute pass before the dispatch's encoder is
  opened, so imageAtomicCompSwap compares the real old texel. (texture_buffer
  `atomic_functions` still fails: imageLoad/imageStore read the texture while atomics read
  the sidecar — a mixed-use image is not coherent yet; the test's final compSwap leg needs
  that. Seeding alone made the atomicAdd leg correct.)

### Verification
* full headless suite: **485 PASS / 0 failures** (test_texext updated to the spec-correct
  glTextureView errors: never-generated name -> INVALID_OPERATION, 2D view numlayers!=1 ->
  INVALID_VALUE).
* test_geom: **600 checks / 0 failures x 10 runs** (593 -> 600: r48_ssbo_two_runtime_arrays,
  r48_ssbo_layout_compile_errors, r48_tes_ssbo_atomics).
* CTS after the final merge: SSBO 124/124, texture_view 3/7 (coherency, reference_counting,
  view_classes, view_sampling remain), texture_buffer 15/16, tessellation_shader 58/58
  (one Timeout was load-induced: the case passes in 186 s with a longer timeout),
  geometry_shader 133 Pass + 2 NotSupported (GS atomics) + 1 load-induced kill, and
  image_load_store unchanged at 38/51.


## Round 46 — image_load_store 35->38, texture_* baseline 296->302, GS image atomics end-to-end

### Commits (gl46/geom)
* `4398846` untrack ~166 tests/headless/scratch_* / zz_* (kept on disk; explicit adds from now on)
* `97ff55b` store-only TES capture + deferred-upload clobber + no GL error on internal tess-capture misses
* `7e48fbf` geometry-kernel image atomics: sidecar slot rebase + bind + sidecar->texture resolve
* `62dec89` xfb capture binds the vertex stage's textures (realize before the capture encoder)
* `0102202` strip leftover interpolation qualifiers on output globals in the GS kernel GLSL
* `dd0a7de`/`cf6e7b5` multisample texParameter error codes (target vs DSA entry points)
* merges: `8346754` (r45), `da577cd` (compat perf), `1dd1785` (fence listener), `0b826a8` (tess r17) + clean rebuild

### Fixes and mechanisms
1. **basic-allFormats-{load,store}GeometryStages** (were Crash): three stacked bugs.
   * A TES with no outputs (image side effects only) compiles to `vertex void` — `mgl_tess_capture_msl`
     could not parse a record, no capture published, and `geom_route_tess` raised a spurious
     GL_INVALID_OPERATION.  Now a void TES synthesizes a zero gl_Position record (gl_in[].gl_Position
     is undefined per GL) and the record store is appended at the end of the body.  All internal
     tess-capture misses in geometry.c now drop the geometry stage silently (GL error only for
     spec-level misuse; the CTS post-test glGetError checks were failing on it).
   * The geometry-kernel texture realize ran INSIDE the tess import encoder (stays open across the
     whole pass): a deferred upload opened a second encoder -> Metal assertion crash.  Hoisted above
     the import encoder (geomRealizeKernelStageTextures).
   * The deferred upload blit (initial texel data) executed AFTER the tess VS/TCS kernels' image
     stores (they run on an earlier-committed compute_cb) and clobbered them.  geomRealizeStageTextures
     now commits+waits when it uploaded pending CPU data; the tess file realizes the stages BEFORE
     compute_cb is picked (would leave a committed buffer).
2. **basic-allTargets-atomicGS** (Test Run() failed, [0,0,0,0]): image atomics never worked from the
   geometry kernel.  SPIRV-Cross puts every `<name>_atomic` sidecar at `[[buffer(0)]]` (Metal: cannot
   reserve buffer 0) and leaves non-2D subscripts un-linearized; the kernel failed to compile and the
   draw silently dropped.  `mgl_msl_fix_image_atomics` is now applied to the kernel MSL (rebased into
   the span above the user pins, `gp->atom_slots_gs`), the sidecar buffer is bound in
   geomBindStageBuffers, and `geomResolveImageAtomics` copies dirty sidecars back into their textures
   at glMemoryBarrier time (one generated kernel per texture type / 32-bit single-channel format;
   layouts mirror geomImageAtomicTexels: 2D pitch 16, `spvMglAI3D`, array/cube helpers).
   Also `SPVC_COMPILER_OPTION_MSL_TEXTURE_1D_AS_2D` for the kernel compile (matches program.c; without
   it image1DArray atomics emit `g_x_atomic[int2(..)]`).
   basic-allTargets-atomic{,CS,GS,VS,FS} 5/5.
3. **texture_cube_map_array texture_size_{vertex,geometry}_sh**: (a) the XFB capture never bound the
   vertex stage's textures (bindVertexBuffers... only does buffers) -> textureSize returned 1,1,1;
   bindTexturesToCurrentRenderEncoder is called on the capture encoder now, with a realize pre-pass
   (geomRealizeStageTextures) so a deferred upload cannot end the capture encoder twice (crash).
   (b) `layout(location = N) out flat uvec3 v;` keeps `flat` after the in/out strip; glslang then marks
   the global so SPIRV-Cross's helper-parameter threading skips it and the GS kernel references an
   undeclared identifier -> kernel compile failed, draw dropped, textureSize [0][0][0].
   strip_member_quals on the plain-variable emission.
4. **texture_border_clamp border_color_errors / texparameteri_errors**: sampler-state pnames on
   multisample targets — target-based entries generate GL_INVALID_ENUM (GL 4.6 8.21), DSA
   glTextureParameter* GL_INVALID_OPERATION (both CTS-checked!); `dsa` flag threaded through
   mglTexParamSet.

### Baselines / CTS final numbers (with tess-17 in)
* KHR-GL46.shader_image_load_store **38/51** Pass (r45: 35): +store/loadGeometryStages, +atomicGS,
  +allStages-oneImage values.  Remaining: 5 platform artifacts (see gaps), 2 MS-store link gap,
  advanced-cast / sync-drawIndirect / sync-bufferUpdate / early-fragment-tests / non-layered_binding,
  memory-order (crash), multiple-uniforms (crash at VS+FS link, images domain).
* KHR-GL46.texture_* **302/339** Pass +11 legit NotSupported (baseline was 296): border_clamp 26/32
  (+2 error cases), cube_map_array 20/27 (+texture_size vertex/geometry; 2 NS legit), gather 8/12,
  view 10/16, buffer 8/11, swizzle 4/6 (1 crash), filtering 24/24.
* KHR-GL46.shader_storage_buffer_object **119/124**: advanced-write-tessellation, basic-atomic-case2,
  basic-syntax{,SSO}, negative-glsl-compileTime.
* Full headless suite: 482 PASS / 0 FAILED (clean rebuild); test_geom 593/0 x10.

### Shared-file edits (this round)
* `MGL/src/tessellation.c` (tess): void-TES record synthesis in mgl_tess_capture_msl (+ trace).
* `MGL/src/MGLRenderer+Tessellation.m` (tess): geomRealizeStageTextures moved above compute_cb pick.
* `MGL/src/MGLRenderer+Draw.m`: extBufBarrier calls geomResolveImageAtomics first (no-op when clean).
* `MGL/src/MGLRenderer+TransformFeedback.m` (xfb): capture binds textures + realize pre-pass.
* `MGL/src/MGLRenderer_Private.h`: declared geomAtomResolvePso / geomResolveImageAtomics.
* `MGL/src/tex_param.c` + `tex_param_internal.h` + `samplers.c`: dsa flag (signature change of
  mglTexParamSet; internal header).
* `MGL/src/mgl_ext_geom.h`: gp->atom_slots_gs, img_atom[].dirty.
* `tests/headless/test_texparam.c`: expectations updated to the two CTS-checked error codes.

### Known gaps / next
* **image2DMS / image2DMSArray stores**: Metal cannot write MS textures from shaders; the store
  program fails to link (basic-allTargets-store, advanced-allStages-oneImage stale GL_INVALID_OPERATION
  from the failed glUseProgram).  Full emulation needs sidecar storage + sampler-side redirect of
  sampler2DMS reads — documented gap.
* **5 image cases are CTS-platform artifacts**: the test itself calls
  `getPlatform().getVulkanPlatform().getMemoryLimits()` (gl4cShaderImageLoadStoreTests.cpp:194,
  scaleDimensionsToMemory) and the surfaceless platform throws "Vulkan is not supported" — zero GL
  executed.  Harness-side fix (outside the worktree) would be to give the surfaceless platform memory
  limits (copyImage, sync-imageAccess2, sso-simple/subroutine/perSample).
* Image-atomic sidecar: zero-initialized, only atomic ops see it — imageStore + imageAtomic on the same
  texel do not alias, and a non-cleared texture's content is not the atomic's base.  MS targets skip
  the resolve (unwritable).
* texture_cube_map_array: texture_size_tessellation_{con,ev}_sh = the tess capture binds textures the
  same way the VS capture did ([1,1,1] fallback, MUTABLE storage iteration) — tess domain, same recipe
  as 62dec89.  image_op_tessellation_* (imageCubeArray in TCS/TES), sampling (glGetActiveUniform
  invalid type for samplerCubeArrayShadow), color_depth_attachments (FBO cube-array completeness).
* texture_border_clamp: 6 integer border-color sampling cases (R32I/R32UI/RG32I/RG32UI/RGBA32I/
  RGBA32UI) — Metal sampler border colors do not cover arbitrary integer borders; needs shader-side
  coordinate clamp / border texel emulation (textures domain).
* texture_view view_classes (glTexSubImageND INVALID_OPERATION on a view), coherency (vertex-shader
  sampling of a view), reference_counting; texture_swizzle 1 crash; texture_gather base-level tesselation.

## Round 45 — finish the uncommitted fp64/vattrib work: vertex_attrib_binding 27/30 -> 30/30 (DONE)

The two dirty files at session start (`MGL/src/fp64.c`, `MGL/src/MGLRenderer+VertexAttrib.m`) were
the round-44 leftovers: whole dvec-array-element access in the fp64 lowering + the padded
size 3/2/1 double formats.  Finished, root-caused three further bugs, closed the group.

### Fixes (all in the vattrib/fp64 domain files I own; no shared-file edits this round)

1. `fp64.c`: whole dvec3/dvec4 ARRAY elements (`vs_in_attrib[i]`, constant index) load through the
   flattened uint4 slot array — element k reads slots 2k/2k+1 and reassembles the vector (the
   dvec3 tail slot only carries .z).  Limits: constant indices only (a dynamic index still errors
   with "copy it to a local first"); the CTS unrolls.
2. `MGLRenderer+VertexAttrib.m` **order dependence**: the whole-attribute pad decision
   (`vattribDblNeedsPad`/`vattribDblShadowSpan`) reads `owner_gl[]` of BOTH word slots of a
   dvec3/dvec4, but `vfunc.vertexAttributes` is an unordered set — a low slot enumerated before its
   high slot saw `owner_gl[] == -1`, split the attribute across a raw-word slot and a converted
   stream, and the stream was never bound.  `owner_gl[]` is now filled in pass 1 (order-free).
3. Same file **internal-slot -> owning-GL-attribute mapping** (4 sites): the interface-location pass
   renumbers inputs (a dvec3 takes the two LOWEST free slots; a vec4 position can land on MSL slot
   5).  The enabled-array test, the format lookup, the shadow-stream bind and the
   const-current-value fill all used the SLOT index as a GL location — slot 5 read disabled
   attribute 5's constant and nothing rasterized.  Also: a stride-0 binding's converted stream is
   `MTLVertexStepFunctionConstant` (the stream holds ONE element; PerVertex read past it).
4. Same file **pipeline composite**: with a bound program pipeline, `MGL_PIPELINE_PROGRAM_SCOPE`
   substitutes an ANONYMOUS composite (name 0) for ctx->state.program, so the introspection
   resolved nothing.  `vattribIntrospectProgram()` resolves the pipeline's vertex STAGE program.
5. Same file **converted-stream index collision**: the double path allocated with `--next_shadow`
   and the float/int path with `next_shadow--`, so the first converted stream of each kind in one
   draw SHARED a Metal vertex buffer index (dvec3 read color_b's float bytes = zeros).  Both now
   allocate identically from `MGL_VA_SHADOW_BUFFER_TOP` down, SSBO-size skip included.

`tests/headless/test_vab_slots.c` now 5 tests / **157 checks**: case3 basic-inputL verbatim
(padded components from the current generic value, stride-0 double), case4 dvec3@3+ivec2@4 draws
green, case5 unaligned stride 62 (every attribute converted; dvec3 and float streams must keep
their own buffer indices).  `tests/headless/scratch_inputL{,2,3}.c` committed as CTS repros.

### CTS counts (fresh single-batch runs, `.agent/cts-vab-r45`, `.agent/cts-img-r45`)

* `KHR-GL46.vertex_attrib_binding` — **30 cases Pass=30 (100%)** (was 27/30; advanced-bindingUpdate
  + basic-inputL-case1/2 fixed).
* `KHR-GL46.shader_image_load_store` — 51 cases **Pass=35 Fail=16** (was 27/23/1 Crash; the
  round-45 image-atomics commits of the main branch fixed 8, **0 new failures**; 6 of the 16 are
  the CTS's own "Vulkan is not supported" platform line).

### Verification

* `make -j8 lib` clean, no new warnings.  Stub scanner clean (0 EMPTY/NOOP/ASSERT) for
  geometry.c, MGLRenderer+Geometry.m, MGLRenderer+VertexAttrib.m, fp64.c.
* Full `tests/headless/run.sh`: **0 failures, 63 files, 6822 checks**.  `test_geom` x10:
  **593/0 every run**.  `test_vab_slots` 157/0.

### Handoff

1. shader_image_load_store 16 Fail: 4 x stale GL_INVALID_OPERATION "after test execution" in the
   *GeometryStages* variants (allFormats-load/store, allTargets-store/atomic) — smells like the GS
   capture/draw path leaving an error behind (query glGetError after the draw in a GS program);
   6 x "Test Run() failed" (negative-bind, uniform-limits now pass; early-fragment-tests,
   non-layered_binding, advanced-sync-drawIndirect/bufferUpdate, basic-allTargets-atomicGS);
   6 x CTS-platform "Vulkan is not supported" (ignore).
2. 420pack backlog unchanged (see round 44 handoff): invariant-on-inputs strip (5), raster-stage
   `layout(binding=N)` (7).
3. fp64 remaining: dynamic-index whole-element dvec array access; dvec3 slot renumbering now
   handled via owner_gl, so nothing known open there.



## Round 44 — vertex_attrib_binding slot-30 collision (15 Fail -> 3), buffer-slot audit, image/420pack baselines

### 1. KHR-GL46.vertex_attrib_binding: 15 Fail -> 3 Fail (27/30 Pass) — DONE

Root cause (from r43) fixed: Metal vertex buffer slot 30 was claimed by BOTH
`MGL_VA_CONST_BUFFER_INDEX` (disabled-attribute currents, bound by extVattribPreDraw) and
`MGL_XFB_MSL_BUFFER_INDEX` (xfb capture scratch, bound on the SAME render encoder after it in
xfbCaptureAsync).  The capture draw overwrote the const binding -> every disabled attribute read
the scratch bytes = stale vertex data of an earlier capture in the same process (cases passed
standalone on a fresh zeroed scratch).

* Fix: `MGL/src/mgl_ext_vattrib.h` (vattrib domain header, 2 defines, called out): const
  30 -> **29**, shadow-buffer top 29 -> **28**.  Slot 30 stays xfb/tess-VS-out (no tess draw has
  vertex attributes).
* `tests/headless/test_vab_slots.c` (committed, 2 tests / 132 checks): two captures in one
  process; case2's disabled slot 2 has current (7,7,7,7) — distinct from both zero and the stale
  poison (1,2,3,1), so neither luck nor poison passes.  Verified the test FAILS with the old
  defines (attrib[2] == (1,2,3,1) reproduced exactly).
* Remaining 3 fails (triaged, NOT slot bugs):
  * `basic-inputL-case1/2` — fp64 lowering (fp64.c rw_access) rejects whole-element access into a
    dvec array attribute: `vs_out.attrib[0] = vs_in_attrib[0]` -> "element access into a double
    vertex attribute is not supported".  Needs the fp64 pass to accept a const element index
    landing on a dvecN (assemble the whole element from its word slots) — fp64 domain.
  * `advanced-bindingUpdate` — nothing rasterizes (clear color) when a program mixes
    `layout(location=3) in dvec3` + `layout(location=4) in ivec2`; with vec3 instead it draws.
    The fp64 lowering renumbers the dvec3's uvec4 word slots onto MSL [[attribute(0..1)]]
    (plan debug: kind[0]=kind[1]=UINT, position float4 lands on slot 2) and the plan feeds
    degenerate positions.  Repro: `tests/headless/zz_adv.c` (scratch, untracked).  fp64/vattrib
    domain.

### 2. Render-encoder vertex-stage buffer-slot audit (task 1b) — DONE

Full map after the fix (31 slots, 0..30):
* 0..R-1 program resource ordinals (buffer-map ordinals; geom kernel guard keeps render stages
  < 24, r43's compute work raised the kernel table to 29),
* R..27 GL attribute bindings (cap `count < MGL_VA_SHADOW_BUFFER_TOP`=28),
* 28 down shadow/converted streams, **skipping 25** (new, below),
* 29 disabled-attribute const, 30 xfb capture scratch (tess: VS-out buffer),
* 22..30 are ALSO pinned by the geometry KERNEL path — compute encoder only, no overlap.

Second overlap of the same class, fixed defensively: when the vertex MSL uses `.length()` on an
SSBO runtime array, SPIRV-Cross emits `spvBufferSizeConstants [[buffer(25)]]` and
geomBindSSBOSizes binds slot 25 — attribute/shadow ordinals could land on 25 (needs >= 26
occupied slots + a VS SSBO; unreachable in the current failing set).  Fix:
* `MGL/src/MGLRenderer+VertexAttrib.m` (vattrib domain file, surgical, called out): the
  attribute allocator pads the buffer map with an unused entry at 25 when the VS MSL contains
  spvBufferSizeConstants, and the shadow allocator steps over 25; + `#import "mgl_ext_geom.h"`.
* Still open (documented, unreachable in tests): during a GEOMETRY capture draw the geom encoder
  pins VSOUT@25 unconditionally — a GS program with >= 26 occupied vertex slots would collide;
  would need the same skip keyed on "GS active" instead of "VS has SSBOs".

### 3. KHR-GL46.shader_image_load_store — fresh full-group baseline 27/51 Pass / 23 Fail / 1 Crash (task 2: NOT cheap)

r43's "40/42" was a partial resume (r42 entries left in place); this is the first full fresh run
(results: `wt/geom/.agent/cts-img`).  Triage:
* **Image atomics on non-2D images are not wired**: SPIRV-Cross MSL emits a `device atomic_*`
  emulation array named `<image>_atomic` for cube/3D/array/rect/buffer images and VERTEX/GS/TCS
  stages — MGL never binds any `_atomic` buffer (grep: no handler).  For image3D the emitted MSL
  is even invalid: `g_image_3d_atomic[int3(coord, 0)]` -> "array subscript is not an integer"
  (needs coordinate linearization, e.g. a small MSL pass).  basic-allTargets-atomic{,CS,GS,TCS,VS},
  advanced-{cast,copyImage,memory-order,sso-*,sync-*}, early-fragment-tests...
* **8 x stale GL_INVALID_OPERATION** "after test execution" — r41/42 leftovers
  (basic-allFormats-{load,store}GeometryStages, store/atomic variants).
* The reason extractor prints "Vulkan is not supported at tcuPlatform.cpp:54" for 6 cases — that
  is a platform LOG line, not the real failure reason (runner artifact, harmless).
* `multiple-uniforms` crashes (rc=-6, likely the uniform-count validation path).

### 4. KHR-GL46.shading_language_420pack — fresh tally 22/43 Pass / 20 Fail / 1 Timeout (task 3, triage before starting; NOT started)

* 4-5 qualifier-order cases: glslang rejects `invariant` on INPUT declarations ("'invariant' :
  can only apply to an output") — GLSL 4.20 allows it on inputs.  Likely fix: GLSL source
  pre-pass stripping `invariant` from input declarations (semantically safe).
* 7 binding_sampler/image cases ("binding_samplers" etc.): red texel — `layout(binding=N)`
  sampler/image bindings not honoured in RASTER stages (r43 fixed compute).  One root cause for 7.
* Scattered semantics: initializer_list, vector `.length()` (length_of_vector_and_matrix,
  length_of_compute_result), implicit_conversions, built_in_values, scalar_swizzlers,
  line_continuation, const_dynamic_value — each red texel at index 0, several root causes.
* utf8_characters: timeout (compiler hang on UTF-8 identifiers).

### 5. Infra notes for the next session

* gl46-port merges now touch `external/` (compat's glfw change).  This worktree SHARES external/
  with the main checkout via a masked symlink; git cannot `add` "beyond a symbolic link" and the
  merge wrapper auto-stashes dirty trees (which fails on that path).  Working recipe (used for
  98a20cf and f2bbd51): `git rev-parse gl46-port:<path>` -> `git update-index --cacheinfo
  100644,<blob>,<path>` -> commit ("adopt gl46-port's ...") -> `git update-index --skip-worktree
  <path>` (clears the phantom "D" that trips the wrapper) -> `git merge gl46-port` -> clean
  rebuild.  No writes into the shared directory, no stash.
* CTS runs: always poll `ps` for another agent's glcts FIRST (I once started a batch during
  tess's run — my tool timeout killed the runner and left an orphan glcts; killed by PID from the
  `--deqp-caselist-file=wt/geom/...` match, then re-ran --resume cleanly).
* Direct single-case runs: `cd $CTS_BIN` is blocked by the shell pattern `cd /|~|..`; use
  `CTS=$(python3 -c "import os;print(os.path.abspath('../../..'+'/MGL-cts/build/external/openglcts/modules'))")`
  then a subshell `(cd "$CTS" && MGL_LIBRARY=$PWD/build/libmgl.dylib ./glcts --deqp-case=...
  --deqp-log-filename=<abs .qpa> ...)`.  Absolute --results-dir for run.py (relative paths break
  glcts's cwd).

### Verification (this round's READY state)

* `make -j4 lib` clean rebuild after the f2bbd51 merge: no errors/warnings.
* `tests/headless/run.sh` (full, 63 files): **0 failures**, 6797 checks.
* `test_geom`: **593 checks, 0 failures** (10 tests).  `test_vab_slots`: 132/0.  `test_tessemu`
  green after the tess merge.
* CTS: vertex_attrib_binding 27/30 Pass (was 15 Fail); image/420pack baselines above.
* Commits on gl46/geom: d87ac3a (slot-30 fix + test_vab_slots), e988be0 (external blob adopt),
  e616c0e (merge 98a20cf), e3bd815 (merge f2bbd51), 65d6862 (slot-25 audit fix).  Scratch
  (zz_*.c, .agent/*) untracked as required.

### Handoff — pick next (ranked by value/cost)

(r44 handoff; see the round-45 handoff above for the current ranking)

1. 420pack `invariant`-on-inputs strip (5 cases, ~1 pre-pass edit) + `layout(binding=N)` for
   raster-stage samplers/images (7 cases, mirror r43's compute fix) -> up to 12 cases.
2. image atomics emulation buffers (`<name>_atomic` binding + image3D coordinate linearization)
   — biggest image_load_store block (10+ cases), needs a binding mechanism like the SSBO-size one.
3. fp64: whole-element dvec array access + the dvec3/ivec4 slot renumbering (vab inputL x2 +
   advanced-bindingUpdate + more in other groups).  [DONE in round 45 — see above.]
