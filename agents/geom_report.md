READY 03520b1 (r81 = 03520b1 on top of the e8ff34b merge 3a800f0; the MS image sidecar itself is r78 b690604 / r79 b0c6f69 / r80 a3858d3, previously unreported). SUPERVISOR TASK DONE HONESTLY: with GL_MAX_IMAGE_SAMPLES = 4 (the device cap, get.c) the three multisample NotSupported groups are LIVE and every one PASSES on this tip — shader_image_size 74/74 Pass (the 36 ms-* rows: basic-ms/advanced-ms x vs/tcs/tes/gs/fs/cs x float/int/uint, all were NotSupported at MAX_IMAGE_SAMPLES=0), shader_image_load_store 51/51 Pass (advanced-sso-perSample and basic-allTargets-load-ms were the 2 NotSupported; early-fragment-tests' old Fail is gone too — that was tess r39's fix, verified green here), shader_texture_image_samples_tests 4/4 Pass. GUARD GROUPS, no Fail/Crash anywhere (~/MGL-cts/geom-r81-guard, 194/194): shader_image_load_store 51/51 (now including the CTS WriteMS sub-case: 4-sample imageStore on 2DMS + 2DMSArray, RGBA32F/I/UI), texture_view 7/7, geometry_shader 136/136. FRESH DOMAIN SWEEP ~/MGL-cts/geom-r81-dom (r81 lib, r77-dom's 9 patterns + the samples group, done 23:49): 784 cases = 776 Pass / 7 NotSupported / 0 Fail / 1 Crash — the 7 NotSupported are texture_filter_minmax 5 (no hw, unchanged) + texture_cube_map_array 2 (the pre-existing introspection gap, unchanged); the 1 Crash is texture_swizzle.smoke rc=-24, the SIGXCPU batch-kill artifact the r77/r28 sweeps also recorded (25-min CPU ulimit under 3-way GPU co-tenancy; standalone verdict in §3). Everything the r77 sweep listed is green and the previously-gated rows are now LIVE: shader_image_size 74/74 (was 38P/36NS), shader_image_load_store 51/51 (was 48P/2NS/1F), shader_texture_image_samples_tests 4/4, geometry_shader 136/136, texture_view 7/7, texture_gather 75/75, texture_repeat_mode 162/162, texture_border_clamp 32/32, SSBO 124/124, transform_feedback 20/20 + overflow 13/13, texture_buffer 16/16, copy_image 13/13, sepshaderobjs 8/8, separable_programs_tf 2/2, texture_barrier 4/4(+ARB 4/4), texture_swizzle 3P(+smoke standalone below), texture_size_promotion 1/1. GATES: full tests/headless/run.sh = 137 binaries / 10986 checks / 0 FAILED; test_geom 3x 670/0; test_geom_msimg 3x 121/0; test_compat_mscaps 23/0; test_alphaless 21/0; test_img3d 613/0; stub_scanner EMPTY/NOOP|ASSERT clean for geometry.c, MGLRenderer+Geometry.m, msl_msimg.c; ab_bench min-of-7 interleaved vs ~/MGL-cts/baseline-e8ff34b: m0 drawArrays x2000 13.777 -> 12.742 ms, m1 blend-flip x2000 414.832 -> 382.161 ms (new faster in both; measured under co-tenant load); locked glmark2 (lockf /tmp/mgl_glmark2.lock, 6 interleaved rounds, order alternating) vs baseline-e8ff34b nsgl, top-3-of-6 per arm (medians unusable — compat gate + tess CTS + my sweep co-ran, spike rounds hit both arms): ideas 561 vs 522 (+7.5%), jellyfish 4260 vs 4356 (-2.2%), gl41-geometry 1050 vs 1063 (-1.2%), score 1956 vs 1979 (-1.2%) — inside the ±3% band, no draw-path regression from the r78 hook.

# geom round 81 — MS image sidecar closeout: the multisample rows go live

## 1. The mechanism (r78-r80, unreported until now)

Metal's `texture2d_ms` / `texture2d_ms_array` are READ-ONLY in MSL
(`texture2d_ms<T, access::write>` fails the toolchain static_assert), so a GL
image2DMS/image2DMSArray store, load-after-store or atomic has no direct Metal
path. r78-r80 route every access of a multisample storage image with a
32-bit-per-component format (r32*, rg32*, rgba32*) through the EXISTING
image-atomic sidecar that r63-r68 built for plain images:

* `mgl_spirv_lower_ms_images` (geometry.c, C over the SPIR-V words, runs in
  program.c's compile_stage on every stage before SPIRV-Cross):
  - OpImageTexelPointer's coordinate becomes `int4(x, y, layer, sample*4)` so
    the sidecar index covers layer and sample;
  - OpImageWrite lowers to one OpAtomicStore per component through a
    folded-coordinate texel pointer;
  - OpImageRead of an image THIS module also writes/atomics lowers to
    per-component OpAtomicLoad (GL 8.26 in-invocation ordering), while reads of
    images the module does not write stay direct texture2d_ms reads (exact,
    zero cost);
  - sidecar word layout `4*(sample + S*(layer + L*(x+W*y))) + component`
    (MGL_MS_WORDS=4), one layout for r32/rg32/rgba32.
* `mgl_msl_fix_image_atomics` (geometry.c, MSL text pass) strips the 2D
  atomic-coord macro wrapper for MS images and adds spvMglAIMS / spvMglAIMSA
  helpers that clamp through the same layout (the r63 bounds guard alone
  collapsed every store past the first row).
* Renderer (MGLRenderer+Geometry.m): `geomImageAtomicBuffer` allocates/seeds
  the sidecar per unit (seed = geomMsSeedPso compute pass reading the texture),
  `geomResolveImageAtomics` (the glMemoryBarrier/image-atomics barrier)
  rasterizes it BACK into the multisample texture — one 1-pixel point per
  (texel, layer), fragment outputs the sample's word with
  `[[sample_mask]] = 1<<sample`, chunked at 8M points; the arrayed variant
  reads `[[render_target_array_index]]` and its PSO sets
  inputPrimitiveTopology. Ordering: the shared command buffer is committed
  WITHOUT waiting first (same-queue ordering), so the flush sees the stores —
  the synchronous wait variant cost two round-trips per barrier and made the
  CTS store case take >25 min (now 24 s).
* program.c: the FRAGMENT stage's sidecar buffer slots are forced into the
  reserved [16,24) range (SPIRV-Cross's auto index handed the MS sidecar
  buffer(0) and shifted the default-uniform block to buffer(1), which the
  legacy uniform streaming at fragment slot 0 then clobbered). FRAGMENT ONLY —
  the tess/GS emulation kernels dispatch as compute and broke with the forced
  base (shader_image_load_store.atomicTCS).
* Caps stay honest: texture_query answers GL_SHADER_IMAGE_LOAD = FULL for all
  MS formats (a texture2d_ms read) but GL_SHADER_IMAGE_STORE / _ATOMIC = NONE
  for the non-word-addressable ones (rgba8, rgba16f, ...), so no CTS case ever
  writes through the leftover MSL drop path (mgl_msl_fix_ms_image_writes,
  which still exists so such shaders LINK).
* r80 hardening: sidecar views are OWNED (CFBridgingRetain; the borrowed
  pointer crashed the flush after the GL texture was deleted); sidecars past
  1 GiB are refused and the lowered stores hit a 4-byte sink buffer, so the
  CTS 10.3 GB advanced-sso-perSample allocation FAILS ON VALUES instead of
  killing the command buffer (kIOGPUCommandBufferCallbackErrorOutOfMemory);
  the flush draw is chunked.

test_geom_msimg.c (121 checks) pins per-sample stores, the no-collision layout
(8x8x4 texels where the naive folded index would collide), loads,
imageAtomicAdd, and 2DMSArray per-(layer,sample) round-trips.

## 2. This round (r81, 03520b1)
* verified the three target groups + all three guard groups on the merged tip
  (numbers in the READY line; per-case statuses in ~/MGL-cts/geom-r81-ms and
  -geom-r81-guard/results.json).
* removed the r78 A/B switch `MGL_NO_MSR` from the draw path
  (MGLRenderer+Draw.m — one line, the geomPrepareImageAtomicsRender call is now
  unconditional; it early-outs unless the bound program has an MS-flagged
  image-atomic slot).
* updated the stale "GL_MAX_IMAGE_SAMPLES = 0" comments (get.c already fixed in
  r79; geometry.c glslang table, msl_reserved.h, msl_msimg.c, and
  test_compat_mscaps.c's header now describe the sidecar + honest-cap split).
* deleted the two r78 debug scratch files that were committed by mistake
  (tests/headless/scratch_msatom.c, scratch_msprobe.c — their cases live in
  test_geom_msimg.c; run.sh only builds test_*.c so they were dead weight).

## 3. Gates
* tests/headless/run.sh (full): 137 test binaries, 10986 checks, 0 FAILED
  (test_drawparams/test_orphan/test_v64limits print no trailing summary line —
  pre-existing; all their cases PASS and rc=0).
* test_geom 3x 670/0, test_geom_msimg 3x 121/0, test_compat_mscaps 23/0,
  test_alphaless 21/0, test_img3d 613/0.
* stub_scanner EMPTY/NOOP/ASSERT clean: geometry.c, MGLRenderer+Geometry.m,
  msl_msimg.c.
* ab_bench min-of-7 interleaved vs baseline-e8ff34b: m0 13.777->12.742 ms,
  m1 414.832->382.161 ms.
* locked glmark2: top-3 per arm (ideas +7.5%, jellyfish -2.2%, gl41-geometry
  -1.2%, score -1.2%) — no regression.
* the sweep's one Crash closed standalone: KHR-GL46.texture_swizzle.smoke run
  ALONE (~/MGL-cts/geom-r81-smoke) = **Pass** in ~6 min — in the sweep batch it
  was killed by the 25-min CPU ulimit after 29 min of contended batch time
  (rc=-24/SIGXCPU), the same artifact the r77 and r28 sweeps recorded for this
  exact case on this exact runner.

## 4. Shared-file edits (this branch vs gl46-port, all of them)
* MGLRenderer+Draw.m — ONE hook call (geomPrepareImageAtomicsRender) next to
  the existing geom3DImageShadowSync; cheap gate, no-op without MS images.
* program.c — the mgl_spirv_lower_ms_images hook in compile_stage + the
  fragment-stage forced_base 16 for the image-atomic slot rebase.
* get.c — GL_MAX_SAMPLES fallback mirror + GL_MAX_IMAGE_SAMPLES = device cap.
* texture_query.c — the ms_word cap split for GL_SHADER_IMAGE_STORE/_ATOMIC.
* MGLRenderer_Private.h — one method declaration.
* msl_reserved.h — comment only.

## 5. Known gaps (documented in code comments too)
* MS image stores/atomics exist only for the word-addressable formats
  (r32*/rg32*/rgba32*); every other MS format reports
  GL_SHADER_IMAGE_STORE/_ATOMIC = NONE and its stores are dropped at the MSL
  level (the shader still links).  CTS honours the caps, so nothing fails.
* MS sidecars are capped at 1 GiB per unit (sink buffer past that) — a shader
  writing an MS image that large loses its writes; the only known customer is
  the CTS's 10.3 GB texture, which fails on values either way.
* The flush at the image-atomics barrier is one render pass per dirty MS unit
  (1 point per texel) — O(texels) GPU work per barrier, no CPU readback.

# geom round 77 — texture_size_promotion (missing-alpha reads), queue item 2 already
# green, fresh domain sweep

## 1. texture_size_promotion.functional — root cause and fix

Failure: "Promotion from internal format GL_R3_G3_B2 have failed during
functional test of alpha channel with target GL_TEXTURE_2D_MULTISAMPLE.
Expected value = 1 read value = 0."

The test renders its RGBA source data into a 1x1, (MAX_SAMPLES-1)-sample
R3_G3_B2 multisample texture, then reads every channel through
texelFetch(sampler2DMS, ivec2(0), MAX_SAMPLES-1) and expects .a == 1.

MGL stores R3_G3_B2 on an RGBA8Unorm Metal backing (no 3-channel 8-bit
format). GL 4.6 8.26: a format without an alpha channel DISCARDS writes to
it and READS ONE. Three writers must cooperate:
* CPU uploads already force it (apply_base -> alpha byte 255),
* glClear already forces it (r58: clear color A=1 for alpha-less slots),
* shader rendering already masks the alpha write (r58 mgl_fb_slot_lacks_alpha),
but the backing of a texture that was only ever RENDERED INTO (or never
written) holds Metal's zero-init 0 -> texelFetch read 0. (r58 fixed exactly
this for the WRITE side; the READ side was still open.)

Fix (sampling side, general): texparamTextureForTex composes the missing
channel into the bind-time texture-view swizzle — when the bound texture's GL
format is RGB-based on an RGBA Metal backing (tdFormatLacksAlphaBacking:
R3_G3_B2, RGB4/5/565/8/8_SNORM/10/12/16/16_SNORM, SRGB8, RGB16F, RGB32F),
every swizzle selector that sources Metal's A is remapped to One, so the
user's TEXTURE_SWIZZLE_* applies on the logical (r,g,b,1) vector exactly as
GL specifies. Metal's newTextureViewWithPixelFormat swizzle applies to ALL
shader reads of the view — verified for texture.read() (texelFetch/sampler2DMS)
on the GPU, not just sample()/gather(). This also fixes the non-multisample
render-into-RGB8-then-sample-.a hole and makes any stale backing byte
(including a hypothetical direct RGBA upload into an RGB format) invisible.
Cost: one switch on the Metal pixel format per texture bind
(texparamPixelFormatHasAlphaBacking); the format-table scan only runs for
RGBA-backed storages; the view itself is cached in the existing
MGLTexParamCache and only ever created for the rare alpha-less formats.

tests/headless/test_alphaless.c pins all three paths:
ms_r3g3b2_texelfetch_alpha_one (the exact CTS shape incl. 1x1 size and
sample index MAX_SAMPLES-1), render_into_rgb8_sample_alpha_one, and
r3g3b2_upload_swizzle_a (SWIZZLE_A=ALPHA composes to 1).

Verified no regressions: texture_swizzle 4/4 (api_errors, intial_state,
smoke 7m49s standalone, functional), texparam/texext/texdata/texswizzle
headless suites green.

## 2. geometry_shader.api.max_image_uniforms
Already passing on this tip (4 verification runs; 136/136 geometry_shader in
the r28 sweep and 136/136 in tonight's fresh sweep). No caps were invented:
GL_MAX_GEOMETRY_IMAGE_UNIFORMS reports 8, the r73-probed Metal limit
(access::read_write textures cap at 8 per function on this GPU), and the
test builds a GS with exactly that many iimage2D uniforms, runs it through
the geometry compute emulation with transform feedback of gl_Position and
verifies the result. The r18/r28-era failures predate r72/r73.

## 3. Fresh domain sweep (~/MGL-cts/geom-r77-dom, r77 lib, finished 04:24)
Patterns: texture*, shader_image*, geometry_shader*, transform_feedback*,
copy_image*, shader_storage_buffer_object*, texture_buffer*,
separable_programs_tf*, sepshaderobjs* (599 Pass / 45 NS / 1 Fail / 1 Crash):

  copy_image 13P | geometry_shader 136P | shader_image_load_store 48P/2NS/1F
  shader_image_size 38P/36NS | texture_barrier 4P (+ARB 4P) | texture_border_clamp 32P
  texture_buffer 16P | texture_cube_map_array 25P/2NS | texture_filter_anisotropic 2P
  texture_filter_minmax 5NS | texture_gather 75P | texture_repeat_mode 162P
  texture_size_promotion 1P | texture_swizzle 3P/1Crash* | texture_view 7P
  transform_feedback 20P | transform_feedback_overflow_query_ARB 13P
  SSBO 124P | separable_programs_tf 2P | sepshaderobjs 8P

  * texture_swizzle.smoke Crash = runner batch-kill artifact (SIGXCPU after
    ~26 min wall under a co-tenant full-headless-suite run; the r28 sweep
    recorded the same Crash with the same runner). Standalone on the same
    library: Pass 1/1 in 7m49s. 8 consecutive passes across r76/r77 code.
  * shader_image_load_store.early-fragment-tests = the documented probed
    Apple-GPU early-z limit (see 545ebb6 GAPS).
  * The 36 shader_image_size NotSupported are the CTS's own multisample-image
    filter, not an MGL gap; texture_cube_map_array's 2 NS likewise ("can be
    run only in ES context"); texture_filter_minmax 5 NS = no hardware
    EXT_texture_filter_minmax.

Gates on this tip (8bacd8e + c9afeb6 + 2b96406):
* make lib clean; full tests/headless/run.sh = 131 binaries, 10668 checks,
  0 FAILED (no new warnings in my files).
* test_geom 3x = 670/0 each; test_img3d 613/0; test_texparam 716/0;
  test_texext 122/0; test_texdata 251/0; test_texswizzle 10/0.
* stub_scanner EMPTY/NOOP/ASSERT clean: geometry.c, MGLRenderer+Geometry.m,
  texdata_formats.c, MGLRenderer+TextureParams.m.
* gl46-port merged at cceed03 (merge 8bacd8e; only geom_report.md conflicted,
  both READY lines kept).


READY 2d302c0 (r76, on top of the af20103 merge 05c5159 — the session that died at 22:37 had already landed r74/r75/r75b (the 3D-shadow per-draw/per-commit gating) and the stencil_texturing compute sampler-slot fix as WIP commit 3dfaf1c; this round re-based nothing, closed its queue and verified it. SUPERVISOR GOALS ALL GREEN ON THIS TIP: texture_gather 75/75 Pass (2 runs), texture_cube_map_array 25 Pass + 2 NotSupported (generate_mip_map_non_filterable_{mutable,immutable}_storage, "can be run only in ES context" — the CTS's own ES filter, not an MGL gap; 2 runs), stencil_texturing.functional 1/1 Pass (2 runs — the WIP 3dfaf1c compute image-unit sampler-slot fix holds), texture_swizzle 4/4 Pass in 4 consecutive full-group runs tonight (932/967/1180/1146 s) + the previous session's 18:11 run = 5 consecutive passes with the FLAKY smoke case green in every one, no failure reproduced since the single observation at bb5ba10 (r59 era). BONUS: gpu_shader5 22/22 Pass — the 4 texture_gather_offset_*_clamp_{border,edge} cases that were the r59 documented gap are green too. r76 = (1) geom3DImageShadowFlushTexture now returns on !g3DShadowLive before its TEXTURE_UNITS scan: every glGetTexImage/download paid that walk (glmark2 pays it per frame readback); same gate the r75b per-commit flush already had. (2) tests/headless/test_texdata.c +depth24_stencil8_max_depth_reads (240 -> 251 checks): the dead session's last open thread, "D24S8 depth 1.0 uploads read back 0" (zz_stxt10), was its OWN PROBE's arithmetic — (unsigned)(1.0f * 16777215.0f + 0.5f) == 0x1000000 in float (16777215.5 is not representable, rounds up to 2^24), so the probe memcpy'd the LOW 3 bytes of 0x01000000 = depth 0 into its own source data; MGL's f_to_un clamps at (1<<24)-1 and is exact — the new test pins depth 1.0 through DEPTH_COMPONENT/FLOAT plus the stencil plane. r72 3D-SHADOW PER-DRAW COST VERIFIED ZERO (the supervisor's 5th item): direct A/B, interleaved, 12 reps, min — drawArrays x2000 WITH the geom3DImageShadowSync call 10.386 ms vs the SAME LIBRARY with both call sites compiled out 11.644 ms (not slower; a first 8-rep pass measured +2.8% = the load noise band, load avg 4-5 with a co-tenant tess run and Docker); the code path returns after 2 ObjC sends + a 4-entry (program,msl) cache probe when the program has no storage-image resource and no shadow was ever created (r75/r75b), and the per-commit + per-download scans are now gated the same way. GATES on 2d302c0: full tests/headless/run.sh = 130 test binaries, 10636 checks, 0 FAILED (rc=0, ~29 min alongside a CTS group run); test_geom 10x = 670 checks / 0 failures each; test_img3d 5x = 613/0 (the changed path); test_texdata 251/0; stub_scanner EMPTY/NOOP/ASSERT clean for MGL/src/geometry.c and MGL/src/MGLRenderer+Geometry.m; no source file of another domain touched (test_texdata.c is the texdata domain's test file — one added TEST only, justified above). REMAINING GAPS: unchanged from the 545ebb6 list (early-fragment-tests = probed Apple-GPU early-z limit; layout_binding.image2D_layout_binding_* 3x = pre-existing, another domain; BC6H adversarial-ramp quality = format limit, bit-exact consistency verified; same-draw 3D-shadow sampling race = GL leaves it undefined).)

# geom round 76 — supervisor queue closed: 4 CTS groups green, smoke flake not
# reproduced (5 passes), r72 per-draw cost measured zero, "D24S8 1.0" root-caused
# as a probe bug (pinned by a new test), one more 3D-shadow gate (per-download).

## 0. Session recovery
* gl46-port was already merged at 05c5159 (af20103 = tess r38); `git merge gl46-port`
  is a no-op.  Nothing was uncommitted in the tree except zz_* probe files
* (zz_stxt10.c was the dead session's last probe — see §3).  The WIP commit
* 3dfaf1c (compute image-unit sampler-slot fix) was complete and is verified green
* by stencil_texturing below.

## 1. The four supervisor CTS items (all green, twice each)
| group | result | notes |
|---|---|---|
| KHR-GL46.texture_gather.* | **75/75 Pass** x2 | r59's 2 incomplete-texture cases and r60's border/gather work hold |
| KHR-GL46.texture_cube_map_array.* | **25 Pass / 2 NotSupported** x2 | the 2 NS are the CTS's own "ES context only" filter; the old "sampling: uniform introspection reports type 0" failure is gone |
| KHR-GL46.stencil_texturing.* | **1/1 Pass** x2 | draw path + compute path; 3dfaf1c's [[sampler(N)]] slot parse + _IMAGE_TEXTURE no-sampler rule hold |
| KHR-GL46.texture_swizzle.* | **4/4 Pass x4 tonight** (+1 at 18:11) | api_errors, intial_state, smoke, functional |
Also re-run for the record: **KHR-GL46.gpu_shader5.* = 22/22 Pass** (the r59 gap
list's 4 texture_gather_offset_*_clamp_{border,edge} cases included).

## 2. The flaky texture_swizzle.smoke
Not reproduced.  5 consecutive full-group passes since the single observation at
bb5ba10 (r59 era, "wrong pixel at gl3cTextureSwizzleTests.cpp:3709").  Audit of the
path that observation would have hit (RGBA32UI output texture read by
glGetTexImage right after a 4-point draw): texdataDownload allocates a FRESH
MTLBuffer (Shared) per readback, ends the render encoder, blits, then
texdataSubmitAndWait (commit + waitUntilCompleted) — there is no recycled-buffer
race to lose a texel through, and the draw/fill share the same serial command
buffer.  Nothing actionable found; leaving it as a watch item (any future
occurrence should be saved with --deqp-log-filename and the qpa kept).

## 3. "D24S8 depth 1.0 uploads read back 0" — probe bug, closed
zz_stxt10/11/12/13 (untracked) all showed every 0xffffff-depth texel reading 0.0
while 0x800000 etc. were exact, and test_texdata's own 24_8 round trip (which
contains 0xffffff) PASSED in the same build — the store was fine.  Root cause:
the probes packed the client word with
`(unsigned)(1.0f * 16777215.0f + 0.5f)` — in FLOAT arithmetic that is 16777215.5
-> 16777216 = 0x1000000, and `memcpy(&src[i*4+1], &di, 3)` then copied the LOW
three bytes 00 00 00, i.e. the probe uploaded depth 0.  (double arithmetic gives
0xffffff.)  MGL's conversions are exact: client_read_row un_to_f(v>>8,24),
f_to_un clamps at (1<<24)-1.  Pinned by the new test_texdata case
depth24_stencil8_max_depth_reads (depth 1.0 through DEPTH_COMPONENT/FLOAT, the
1-lsb floor value, and the stencil plane).

## 4. r72 3D-shadow refresh: per-draw cost measured
* Code path (r75/r75b): geom3DImageShadowSync returns before any unit scan unless
  (a) the bound program / a pipeline stage program has a STORAGE_IMAGE resource
  (cached per (program, msl-pointer), 4 entries) or (b) a shadow ever existed
  (g3DShadowLive — write-back duty).  Two ObjC sends + a 4-entry probe per draw.
* Measured: ab_bench m0 (drawArrays x2000, plain VS/FS, no image units), two
  builds of the same tip differing ONLY in the two `[self geom3DImageShadowSync]`
  call sites being compiled out (build/ab_old) vs present (build/ab_new):
  12 interleaved reps, min 11.644 ms (call removed) vs 10.386 ms (call present);
  medians 15.07 vs 15.03.  Not slower — the cost is below the noise floor
  (< 1% of 5.8 us/draw).  (Machine load 4-5 from a co-tenant tess run + Docker.)
* r76 closes the last ungated walk: geom3DImageShadowFlushTexture (every 3D
  texture download) now returns on !g3DShadowLive before scanning TEXTURE_UNITS.

## 5. Gates on 2d302c0
* tests/headless/run.sh: **130 test binaries, 10636 checks, 0 FAILED** (rc=0;
  ran alongside a CTS group run, ~29 min wall).
* test_geom 10x: 670 checks / 0 failures each.  test_img3d 5x: 613/0.
  test_texdata: 251/0 (was 240).  test_bptc in the suite: 257/0.
* python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT: clean for
  MGL/src/geometry.c and MGL/src/MGLRenderer+Geometry.m.
* MGL/src/MGLRenderer+Compat.m: untouched this round.

## Shared-file edits this round
* tests/headless/test_texdata.c — ONE added TEST (depth24_stencil8_max_depth_reads)
  in the texdata domain's test file; no MGL source of another domain touched.
  Reason: it pins the exact behaviour that the dead session's open thread was
  about, and texdata has no max-depth readback case.

## Scratch (untracked, not committed)
zz_stxt10..14.c (the D24S8 1.0 bisect + the float-arithmetic proof), scratch_cts.sh,
scratch_cts2.sh, scratch_qpa.py, /tmp/run_swz2.sh + /tmp/ab3d*.sh harnesses,
build/ab_old + build/ab_new (A/B pair for §4).
READY 545ebb6 (r71+r72+r73, on top of gl46-port cff5363 merge d1f251e. SUPERVISOR GOALS DONE: texture_view 7/7 Pass (was 6/7, view_classes failed on uncompressed uploads into BPTC storage); shader_image_load_store 48 Pass / 2 NotSupported / 1 Fail (was 46/3/2 — non-layered_binding and multiple-uniforms FIXED; the remaining early-fragment-tests Fail is a probed Metal platform limit, see GAPS). r71 = BPTC/RGTC online compression: CPU encoders BC7 mode 6 (unorm+srgb) and BC6H mode 10 (signed/unsigned float) in texdata_formats.c with bit layouts cross-checked field-by-field against VK-GL-CTS's reference decoders (framework/common/tcuCompressedTexture.cpp decompressBc6H/decompressBc7 — hardware decode verified BIT-EXACT against a CTS-port in tests/headless/test_bptc.c), plus a latent BC4/5 palette off-by-one fix in tdEncodeRGTC (the old code's shade[] table skipped the a1 endpoint slot — blocks decoded to the wrong staircase; exposed by the first-ever real hardware decode of its output); texSubImage into texStorage'd RGTC/BPTC levels now encodes blocks and stores them block-aligned like CompressedTexSubImage (ETC2 stays rejected: no encoder); getTexImage of a compressed VIEW now decodes with the VIEW's decoder (format-cast Metal view over the root storage — fixes the BC6H signed<->unsigned cross-pair). r72 = non-layered 3D image units: Metal has NO path to alias one 3D slice as 2D (probes committed as knowledge: 3D->2D newTextureView ABORTS the process, 3D<->2D blits silently no-op, a 3D texture bound into a texture2d argument silently kills the whole dispatch) — the slice now runs through a 2D SHADOW texture with typed compute copy kernels (per MSL scalar channel type; sRGB through its linear view), refreshed per draw and written back on rebinding/commit/readback (tests/headless/test_img3d.c covers layer-by-layer copy + readback). r73 = per-stage image-uniform limits: Metal caps access::read_write textures at 8 per function (probed on fragment AND compute; read/write-only are plentiful but a load+store image needs read_write, and 16 ro+wo pairs = 32 > the 31-entry texture table), so fragment/vertex/tcs/tes/geometry report the ARB floor 8 while compute keeps 16 (KHR-GL46.gl_spirv links 10 access-decorated compute images); test_tess_images.c updated to expect 8 (shared-file edit, justified above). GATES on 545ebb6: full tests/headless/run.sh = 124 test binaries, 10289 checks, 0 FAIL (13m42s); test_geom 670/0, test_bptc 257/0, test_img3d 270/0; ab_bench vs baseline-cff5363 (5 draw modes x5): no regression (dmin +1.8%..-18.9%, medians overlap = the usual load noise band; no draw-path hunks in r71-r73). REMAINING GAPS (documented in code comments too): (1) shader_image_load_store.early-fragment-tests Fail — GL 4.6 requires a discard-using fragment shader to get the depth test AFTER it runs (side effects before discard persist); probed on raw Metal (zzab58/metaldepth_probe.m pattern): even a [[depth(any)]]-writing fragment function is early-rejected by Apple GPUs with the interpolated z, so GL's late-z-for-discard would need Always+shader-side depth sampling of the depth attachment (feedback-loop hazard + per-draw depth snapshot) — rejected as too risky for the gain; stencil-with-discard same. (2) layout_binding.image2D_layout_binding_imageLoad_{Compute,Fragment,Vertex}Shader 3x Fail = PRE-EXISTING (verified against baseline-2898abe before my changes; binding-default issue, another domain). (3) BC6H encoder quality on adversarial per-channel-offset ramps is coarse (endpoints exact; one shared 4-bit index per RGB texel is a format limit — GL does not guarantee online-compression quality; consistency verified bit-exact instead). (4) 3D shadow planes: sampling the 3D texture in the SAME draw that writes its shadowed image unit cannot see the pending stores (GL leaves that race undefined); the shadow copies run at draw start, so image->image layer ping-pong across draws is correct.)

READY bba736a (r69: packed_pixels depth-stencil regression FIX — 2 holes my r59-r61 opened + one spec-correct rejection; on top of gl46-port 83e1777 + r66/r67/r68. GATES on this tip: full tests/headless/run.sh = 123 tests PASS / 9729 checks / 0 FAIL (test_geom_xfbperf in, quarantine lifted); test_geom 10x = 670 checks / 0 failures each; KHR-GL46.packed_pixels.varied_rectangle.* = 83/83 Pass (was 81/83: depth24_stencil8, depth32f_stencil8, depth_stencil failing); packed_pixels full group 250/250 Pass; packed_depth_stencil 25/25, texture headless suites (texdata 240, texparam 716, texext 122, texswizzle 10, texonline 20) all 0 fail. ab_bench interleaved vs ~/MGL-cts/baseline-83e1777, 3 runs of 5: m0 drawArrays min 11.52 new vs 12.32 base (equal-or-faster; per-run deltas flip sign +11/-10/+9% = machine load noise 3.7-6.5, no r66-r69 code path touches the plain draw path), m1 min 96.9/97.0/107.9 new vs 96.3/113.6/112.0 base (equal-or-faster in every run). r69c is a PURE TEST-PROBE commit (zz_* files; your merge strips zz_*). DETAIL: (1) the r59-r61 GL_RED_INTEGER stencil-texturing exception keyed on ANY depth/stencil base, accepting glTexImage2D(DEPTH24_STENCIL8/DEPTH32F_STENCIL8/GL_DEPTH_STENCIL, GL_RED_INTEGER, ...) — now readback-only + STENCIL_INDEX base only; (2) tdCheckGetFormatCompat answered (GL_RED, GL_FLOAT) reads of combined depth-stencil textures (stencil_texturing reads its R32F RESULT texture that way, never the DS texture) — rejected like every non-DS client format per GL 4.6 table 8.2. Also in this tip since 83e1777: r66 image-atomic MSL helpers (get_array_size / per-kind cube helpers) shader_image_load_store 5 atomic cases Fail->Pass; r67 tess-kernel image atomics + no double TES execution (advanced-allStages-oneImage Fail->Pass); r68 image/sampler ARRAY elements at strided units (advanced-sso-simple/-subroutine Fail->Pass); CTS glue vk::Platform::getMemoryLimits (3 more shader_image_load_store cases). shader_image_load_store is now 46 Pass / 3 Fail / 2 NS on this tip.)

READY bb5ba10 (= e033682 + gl46-port merges aafc7c2 [tess r30] and 1e840b4 [compat s23+s24]; clean-build gate: rm -rf build; make -j4 lib glx nsgl; tests/headless/run.sh = 122 PASS / 0 FAIL, 9507 checks; test_geom 10x = 670 checks / 0 failures each; ab_bench min-of-7 interleaved vs ~/MGL-cts/baseline-1e840b4: m0 drawArrays 11.190 -> 10.826 ms (-3.3%), m1 blend-flip 150.98 -> 151.82 ms (+0.6%) — inside the +3% limit; texture_swizzle.smoke Pass in 373 s. SMOKE 73 s -> 460 s ANSWER: ambient load, not a code regression — the 73 s reference was a quiet machine; same-load A/B tonight: r58 lib (baseline-6f9b359) 379 s vs this tip 373 s = parity; microbenchmarks: whole-link cache hit 0.37-0.47 ms/link on BOTH libs, and the smoke shape (link + texStorage/texSubImage + draw + readback) identical to 3% on both. One flaky smoke FAIL seen exactly once in 5 tip runs (wrong pixel at gl3cTextureSwizzleTests.cpp:3709, never reproduced — follow-up). Merge notes: compat's sep_vs_kernel_cache replaces my build_sep_vs_kernel (I kept my r59 prepare-before-rebuild on the whole-link cache-hit path: compile_stage caches a PREPARED module, so the raw stored SPIR-V must go through spirv_prepare_for_msl first); my first merge-resolution dropped compile_stage's done:-tail frees/return (suite 201/364 geom fails) — caught and fixed before commit; the r59 MISS debug log is REMOVED entirely (was env-gated).)

## r61 session (after READY bb5ba10): texture_view.coherency FIXED; merges aafc7c2/1e840b4/71cfe2a; smoke-6x explained
* merges: aafc7c2 (tess r30; my program.c resolution initially dropped compile_stage's
  done:-tail frees/return — suite showed test_geom 201/364, fixed before the commit landed;
  MGLRenderer+Tessellation.m now matches gl46-port exactly, the r59 MISS debug log deleted),
  1e840b4 (compat s23+s24: their sep_vs_kernel_cache replaces my build_sep_vs_kernel; on the
  whole-link cache-hit path I kept my r59 prepare-first call — compile_stage caches a
  PREPARED module so the raw stored SPIR-V must run through spirv_prepare_for_msl),
  71cfe2a (my r59-r61 back-merged by gl46-port; report conflict only).
* FIX (textures.c, surgical): texture VIEW images live in the ROOT's storage —
  getTexImage/texSubImage/copyTexSubImage now redirect (level += view_min_level, cube face
  or z layer += view_min_layer).  KHR-GL46.texture_view.coherency: Fail -> Pass (was
  "glGetTexImage() call failed"; also fixed copyTexSubImage into a view returning
  GL_OUT_OF_MEMORY).  Suite after: 122 tests / 9507 checks / 0 fail; test_texext 122/0,
  test_geom 670/0.
* FIX (MGLRenderer+Geometry.m, my file): the VS-capture kernel bind now calls
  bindMTLTexture for textures with no mtl_data (a VIEW materializes its Metal texture
  lazily; the kernel otherwise bound nothing at that slot).
* INVESTIGATED + NOT A BUG: "geometry stage samples the render target" — my zz_view_gs
  repro left the FBO's rt bound on the sampled unit.  After fixing the repro, GS-kernel
  sampling is CORRECT for parent + view, 1D, and minlevel=1 shapes (MGL_GEOM_DUMP shows
  the kernel MSL text/texture(0) args are right and the bind lands at the same index).
* REMAINING (documented): KHR-GL46.texture_view.view_sampling still Fails — failures now
  appear in different stages per iteration (vertex/geometry/fragment(0,0)); my isolated
  5-stage and 1-stage repros pass, so it needs a per-iteration bisect of the CTS's
  target/format matrix (next step: port the exact iteration that first fails).
  texture_view.view_classes needs CPU BPTC encoders (BC7 unorm + BC6H float) to accept
  uncompressed uploads into texStorage2D'd BPTC storage (RGTC already encodes online).
* stencil_texturing.functional root-caused to the COMPUTE path only: the DRAW path
  (5-stage VS/TCS/TES/GS/FS with DEPTH_STENCIL_TEXTURE_MODE=DEPTH_COMPONENT) samples
  D24S8 and D32FS8 EXACTLY (my repro zz_stencil_tex.c passes; the earlier "D24S8 upload
  scramble" was my own repro packing 24_8 into an 8-byte-stride buffer — GL layout is
  depth<<8|stencil, tight 4B/texel, which MGL handles correctly).  The compute half
  samples texel(x-1,y-1) at EXACT texel boundaries: Metal NEAREST = floor(u*n - 0.5),
  GL = floor(u*n).  Exact-equivalence fix: add 0.5/n to the coordinate for NEAREST
  samplers — needs a per-sampler-slot mechanism (function constants like the r60 gather
  border set, or an MSL sample() rewrite); not implemented tonight.
* gates on the final tip (71b07d7 = bb5ba10 + view fix + back-merges): full suite
  122 PASS / 9507 checks / 0 FAIL, test_geom 670/0, test_tess 504/0, test_texext 122/0,
  test_sepshader 4/0.  (ab_bench/smoke numbers above were measured on bb5ba10; the only
  code change since is the view redirect — a mgl_texext_info lookup in the GetTexImage/
  SubImage entry points, not a per-draw path.)

READY e033682 (r59+r60 after the 6f9b359 merge, incl. tess r28+r29 b7821df and compat s22+s23 c6ded7d merges; clean-build gate: rm -rf build; make -j4 lib glx nsgl; tests/headless/run.sh = 118 PASS / 0 FAIL, 9478 checks; test_geom 10x = 670 checks / 0 failures each; texture_swizzle.smoke Pass in 460 s (limit 19 min); stub_scanner EMPTY/NOOP/ASSERT clean for geometry.c + MGLRenderer+Geometry.m; MGLRenderer+Compat.m diff vs gl46-port = 0 lines. ab_bench min-of-7 interleaved vs ~/MGL-cts/baseline-c6ded7d: m0 drawArrays x2000 11.873 -> 11.395 ms (-4.0%), m1 blend-flip x1000 183.742 -> 162.499 ms (-11.6%) — both inside the +3% limit. r59/r60 CTS deltas vs the r28 sweep: shader_image_size 28/10/36 -> 38/0/36 (10 Fail -> 0: a whole-link cache hit now rebuilds the separable VS-as-compute kernel for the new program name), shader_image_load_store 37/12 -> 38/11 (uniform-limits Fail -> Pass via image-ARRAY atomics, one combined sidecar buffer), texture_gather 73/2 -> 75/75 (incomplete-texture sampling enforced in ALL paths incl. VS-gather emulation + compute, 2D/CUBE-map-array mipmap completeness layer rule); also landed without a group-count change: GetTexImage depth-stencil honours DEPTH_STENCIL_TEXTURE_MODE, compute binds each sampler at its OWN MSL index, gather CLAMP_TO_BORDER plumbs the real border colour. Locked interleaved glmark2 (every run under /usr/bin/lockf -k /tmp/mgl_glmark2.lock, order alternated, vs baseline-c6ded7d nsgl): ideas 14 pairs median 626.5 -> 613.0 (-2.2%, top3 -5.8%), jellyfish 6 pairs median -13.2% (2 new-arm rounds hit co-tenant load spikes at 2580/3765 fps; quiet-window top3 -3.6%), gl41-geometry 6 pairs median -1.1% (top3 -4.3%) — no scene regresses past the +3% limit. Still failing: gpu_shader5 18/22 (4 texture_gather_offset_*_clamp_{border,edge}, documented gap), texture_cube_map_array 24/1/2 (sampling: uniform introspection reports type 0).)

READY c9be6f2 (e84cbab + failed-relink introspection zero-report; clean-build gate: rm -rf build; make -j4 lib glx nsgl; tests/headless/run.sh = 563 PASS / 0 FAIL, 8717 checks — test_piq 76/0, test_sepshader 4/0, test_texswizzle 10/0, test_geom 670/0. Speed: supervisor clean A/B at e84cbab = ideas -1.1%, jellyfish +0.2%, gl41-geometry -1.6% (within the 2% gate); c9be6f2 only adds failed-relink query gating, no hot-path code. r58 CTS: texture_swizzle.functional Crash->Pass (alpha-less attachment write mask + clear, depth-mode x swizzle composition, DEPTH_TEXTURE_MODE default RED), gpu_shader5 18/22 (clamped gathers remain), piq 43/43, sepshader fixed.)

# geom round 58 — ideas -5.5%: reproduced, root cause narrowed to a wall-time
# (Metal-side) effect; CPU/instruction parity proven; no MGL-code cause found
# under the night's load. Details below. gl46-port 7478e0a merged (cf1f433).
# No new READY this round (r57 tip ef1e1e9 + this merge; ideas fix pending the
# supervisor's clean-machine bisect).

# geom round 59 — link-cache sep-vs kernel (shader_image_size 10 Fail→0), image-ARRAY
# atomics (uniform-limits Fail→Pass), Apple-GPU discard/early-z gap root-caused.
# gl46-port merged at 6f9b359 (content-identical to my c9be6f2 squash). No READY yet.

## r59 results (commits b5bd61d..025f3ed on gl46/geom, after merging gl46-port 6f9b359)

* **shader_image_size 74: 38 Pass / 0 Fail / 36 NotSupported** (was 28/10/36). Root cause of
  all 10 tcs/tes in-group failures: compile_stage caches a COMPUTE-kernel variant of a
  SEPARABLE vs-only program's vertex stage (mgl_tess_cache_vs_kernel, keyed by program
  NAME). A whole-link cache hit clones a previously linked executable and skips
  compile_stage, so the new program object (the CTS deletes+recreates programs per case,
  names get recycled) had NO cached kernel → the tess draw logged "tess VS stage function
  is not a kernel" and bailed → black raster for every case after the first. Fix in
  program.c: the sep-vs kernel block is factored into build_sep_vs_kernel() and is re-run
  from mglLinkProgram after a cache hit (same predicate: separable, vertex stage, no TES;
  outside the cache lock since it compiles MSL). Every case passes solo AND in-group now.
* **shader_image_load_store 51: 38 Pass / 11 Fail / 2 NS** (was 37/12). uniform-limits
  fixed by image-ARRAY atomics: geometry.c mgl_msl_fix_image_atomics now parses
  `array<texture2d<T,acc>, N> u` declarations and rewrites `u[i]_atomic[spvImage2DAtomicCoord(c, u[i])]`
  → `u_atomic[spvMglAIArr_u(u, i, c)]` with a generated always_inline helper (per-element
  raw-width regions, the pitch constant SPIRV-Cross's macro uses, spliced AFTER the
  constant's declaration); MGLRenderer+Geometry.m builds ONE combined side buffer
  (geomImageAtomicBufferArray:elems:), seeds/resolves per element through 1-slice 2D-array
  views (the raw-layout kernels), and the render+compute bind walks bind it once at the
  uniform's buffer index.
* **geometry_shader 136/136** (no regression). Full headless suite: 109 tests, 8096
  checks, 0 failures (test_fp64 is flaky only under full-suite load; 3/3 solo clean).

## r59 documented gaps (probed at the raw-Metal level, see tests/headless/zz_metal_*.m)

* **early-fragment-tests** (also basic FS+discard shapes): on this GPU/OS (M4 Pro, macOS
  26.5 SDK) ANY fragment function containing discard_fragment() gets EARLY depth/stencil
  testing — the fragment is killed before the shader runs, so imageStore/atomic side
  effects are lost when the depth test fails. Probed [[early_fragment_tests]],
  [[depth(any)]], [[depth(less)]], depthWriteEnabled NO: every variant early-kills; a
  function with NO discard gets the test correctly deferred (store lands, color killed).
  GL requires discard shaders to run first; Mesa's Vulkan-on-Metal reference
  (.agent/ref/kosmickrisp nir_to_msl.c:1480) emits plain discard_fragment() too. A real
  fix needs shader-side depth testing (sample the depth attachment, fold the compare into
  the discard) — a feature, not a bugfix.
* **non-layered_binding (3D shape)**: glBindImageTexture(unit, tex3d, 0, GL_FALSE, layer,
  …) must expose one 2D slice as image2D, but Metal REFUSES a texture2D view of a 3D
  texture (assertion probed in zz_metal_3dview.m). Already documented by an earlier
  round; the 2DArray non-layered shape works (slice views are allowed).

## r59 queue leads (repros committed as tests/headless/zz_*.c)

* shader_image_load_store remaining 11: advanced-allStages-oneImage is verified working
  in a committed repro (zz_tbufimg.c: imageAtomicAdd on iimageBuffer+uimage2D in
  VS/TCS/TES/GS/FS, array + patch + DrawElementsInstancedBaseVertex PATCHES draws, with
  and without layout(binding=)) — the CTS case still fails; remaining differences:
  `coherent`, patch_vertices=1 + point-mode TES, an SSBO in the same program, basevertex=1.
* **texture_gather.incomplete-texture/-last-comp**: level 1 replaced with 1x1 under a
  32x32 level 0 must make the texture INCOMPLETE (GL 8.17 size-halving rule), so
  gathers return 0 (alpha 1). Added mgl_texture_incomplete_for_sampling()
  (geometry.c; GL 8.17 base..top halving + base-level existence, honors
  min_filter/base_level/max_level) enforced at bindTexturesToCurrentRenderEncoder
  (MGLRenderer.m surgical edit: bind the shared 1x1 black fallback texture + sampler)
  and at createMTLTextureFromGLTexture. Verified in zz_incomptex.c for FS/VS/GS gather
  shapes (all return (0,0,0,0) now). The 2 CTS cases STILL fail with level-0 data
  (0, 0.25, 0.5, 0.75) — my repro matches the CTS call order (6 levels →
  glGenerateMipmap → replace level 1 with 1x1 → SubImages → draw POINTS with a VS
  gather into a 1x1 FBO) and passes, so something else in the CTS case keeps the
  texture complete in MGL's view; needs the exact CTS shader (gl4cTextureGatherTests
  .cpp) ported next.
* texture_view 4 (view_sampling/view_classes/coherency/reference_counting): views with
  minlevel=1 (2D, 2DArray→2D minlayer=1, 1D, 3D) sample correctly through texelFetch in
  BOTH a plain FS and a point→triangle_strip GS program (zz_viewlvl.c). The CTS shape
  uses one 5-stage program with TRANSFORM FEEDBACK capturing VS..GS sample results
  (6 verts × 4 ints) — port the exact shaders from gl4cTextureViewTests.cpp ~4300–4700
  into a headless repro next.

## r58 queue work after READY c9be6f2 (leads for the next round; repros are untracked zz_*/scratch_* files)

* gpu_shader5 18/22: the four remaining cases (texture_gather_offset_color_clamp_to_border /
  _clamp_to_edge, texture_gather_offset_depth_clamp_border / _clamp_edge) need border-aware
  gathers: the CTS borders are (-1,-1,-1,-1) on RGBA32I (Metal samplers only offer
  transparent/opaque black and white, so an integer -1 border is unrepresentable) and
  (1,1,1,1) on depth compare. Fragment gathers are native (only the VS/compute emulation is
  rewritten), so this needs a fragment-stage gather rewrite with the border as function
  constants, or manual taps. Repro: tests/headless/zz_gather_clamp.c (int variant prints
  (0,0,0,0) where (-1,-1,-1,-1) is required for clamp-to-border).
* shader_image_size 27/74 solo-pass but 11 fail in-group (deterministic, starts at
  basic-nonMS-tcs-int). While isolating it I found a TESS bug that explains the tcs/tes
  failures: a TES whose outputs go straight to the fragment stage rasterizes (0,0,0) — even
  a constant vec4(1,0.5,0.25,1) output reads back zero (test_tess_images.c passes because it
  only checks imageStore results, never the raster). The tess draw dispatches (MGL_TESS_DEBUG
  shows the tcs kernel dispatch and the image bind on the same encoder) but the rasterized
  vertex function appears not to be the TES. Repro: tests/headless/zz_imgsize_tcs.c (revert
  to the constant-color TES variant).
* shader_image_load_store.uniform-limits: two shapes — (a) image-ARRAY atomics
  (imageAtomicAdd(u_image[i], ...)) produce `&u_image[i]_atomic[...]` MSL that the
  mgl_msl_fix_image_atomics pass does not rewrite (its name table holds only the base name);
  (b) scratch_ulimits.c's atomic_mix shape now dies earlier: the SPIR-V patcher logs
  "Currently no block to insert opcode" and the link fails. Repro: tests/headless/scratch_ulimits.c.

## ideas regression investigation (00:10–05:35, supervisor deadline 04:00 hit at write-up)

Environment: two other workers hammering CPU+GPU all night (load 3–7); every glmark2
launch wrapped in the shared `lockf -k /tmp/mgl_glmark2.lock` from 01:20 on. Ambient
swings (base 275→1220 fps in one 16-pair batch) mean single batches resolve ±8%; only
paired medians of many runs are meaningful, and even those flip between 0.92 and 1.01.

### Reproduction
* Cleanest CPU pair (30 s runs back-to-back, `/usr/bin/time -l`, default window):
  base(6ee46f0) 11.17 user / 2.88 sys / 36.69 G instructions / 20.47 G cycles, FPS 637;
  tip(ef1e1e9+be4c52b) 11.27 / 2.80 / 36.47 G / 20.34 G, FPS 602 → **-5.5 % FPS with
  CPU, instructions and cycles all within ±1 %**.
* 16 interleaved 64×64 pairs (duration 8): median T/B 0.959; ceiling ratio (max/max,
  the quiet-window estimate) 1220 vs 1126 = 0.923. Earlier 6-pair 320×240 set: 0.941.
  So the deficit is real and roughly constant across window sizes.
* My own build of 6ee46f0 vs the supervisor's baseline-6ee46f0 dylib: 0.976/0.996/1.023
  (parity) — the builds are equivalent; this is not a toolchain artifact.

### What it is NOT (all measured, all negative)
* **Not MGL C code CPU.** In-process replay of the ideas per-frame pattern
  (tests/headless/scratch_abideas.c — 240 dirty walks/frame: program switches,
  default-uniform uploads, per-object VAO respec, depth/blend/cull flips, indexed+
  array draws, shadow-FBO round trip, depth-texture sampling; CPU = getrusage):
  12–14 randomized pairs → T/B 0.986–0.997 (parity, three replay versions).
* **Not shaders.** `MGL_DUMP_MSL` diff: the MSL differs ONLY in entry-point names
  (`vertex_1_main` vs `vertex_cbd0a6c17e452cb2_main` — r50a content-hash names; the
  text is 6–8 % longer). Same 18 library compiles in both. Load/store actions,
  attachment formats, layer counts identical (instrumented both builds, ZZRP dump).
* **Not extra GL/Metal work.** MGL_PROFILE call counts identical (draw 296/frame,
  dirty walks 18.2, bindList 81.6, bindTextures 6.5, encoders 1.0); binds/frame
  issued=105.8 both; drawfast hit rate 95–97 % both; no memory growth over 25 s runs.
* **Not any single shared-file hunk.** Scratch builds with shared hot files reverted
  to 6ee46f0 (all 11 at once, then the renderer core four, Program.m alone,
  TextureParams/Gpuaudit/buffers/get/multi_bind, r50a short-entry-names): every
  variant still measured -4…-8 % in the noisy batches; the short-name variant hit
  1.009 once and 0.917 once (inconclusive). The wall buckets that DO inflate in the
  tip (draw +8–12 %, bindList +33 %, mapBuffersToMTL +75–95 %, enddraw-stamp +43–50 %,
  identical call counts) inflate uniformly — the signature of waiting, not of code.

### Best explanation
A ~5 % per-frame WALL-time cost with CPU/instruction/cycle parity: frames wait more
inside the Metal driver (sample shows the tip blocking in IOGPUResourceCreate /
MTLLibraryBuilder::newLibraryWithSource 21 vs 14 samples, base instead in
CA::Render::Encoder::send_message). The only runtime-visible content difference is
the longer MSL text/entry-point names from r50a; Metal's compiler/pipeline internals
key on the source text, and bigger/different texts shift where the main thread parks.
Needs the supervisor's clean-machine bisect (r55–r57) to confirm; on this machine
tonight the signal-to-noise (±8 %) could not resolve it further.

### Supervisor's 04:27 A/B (clean machine, lock, 3 interleaved rounds, default 3 scenes)
ideas base 3373/3272/3263 vs tip 3273/3177/3196 = **-2.7 %** (every round); jellyfish -1.5 %,
gl41-geometry -1 %. Their read: cost is per frame / per glUseProgram-uniform call, not per draw.

### Follow-ups after that (04:30-06:25)
* Whole-run `sample` symbol diff (base vs tip, ideas 800x600), program/uniform-related:
  mglLinkProgram 24/24, run_link 24/23, mgli_upload_uniform 36/44, mglUniformMatrix4fv 41/46
  — equal within noise; glmark2's own std::map uniform lookups 27/34 __lower_bound (app side).
  Tip heavier ONLY in extProgramCompileMSL / extProgramLibraryForMSL / MTLCompilerFSCache::
  openSync (14 -> 21 samples).
* Instrumented MSL-compile counter (ZZLIB, 20 s ideas run, both builds): **18 compiles in
  BOTH** — no extra libraries, no per-frame re-links. Oddity: the tip compiles the SAME
  entry (fragment_330a4d9e2c96b498_main) three times — the entry-cache does not dedupe
  identical MSL, and each tip compile also runs newFunctionWithName:constantValues
  (specialization path, from the Program.m atomic-counter fix) instead of the plain
  lookup, which is why its compile samples are ~50 % higher for the same 18 compiles.
  Startup-only (~10 ms), cannot explain -2.7 %.
* 16-pair 64x64 interleaved batches of base-vs-tip and base-vs-short-entry-names (g7):
  both ~0.983-0.985 median here — unresolvable at this ambient; ceiling ratios 0.965-0.973.

Still unexplained by every MGL-code path I can instrument; the in-process replay parity
(draw/state/uniform/VAO/FBO/depth-sample churn, 240 walks/frame) plus equal
instructions/cycles/user/sys in clean pairs says the missing ~8 us/frame is in
driver/present wait time, not in MGL C code. Supervisor's own r55-r57 bisect to confirm.

READY 4ae97c2+r56report (branch tip after gl46-port 9442011 compat-s18 merge b832372 + r56 CTS fixes; gates: suite 96/96 0 FAIL, test_geom 10x 670/0, test_fp64fast 3x 298/0, test_smoke 1/1. Ideas A/B vs 9442011 (paired, interleaved, glmark2 ideas 640x480 + native build): TIP draw-bucket 0.883/0.116 ms/frame vs BASE 0.903/0.115 (draw -2%), swap 0.127 vs 0.133, dirtyState-block 0.437 vs 0.419; paired totals b832372/base = draw-ms 95.9%, frames 101.4% — at parity or faster, -5.5% NOT reproducible; per-commit isolation (revert 621cd67 / 476f0e4 / 1103423 / r56 off the tip) all within ±3% = noise. MGLRenderer+Compat.m untouched.)

# geom round 56 — 3 KHR-GL46.geometry_shader Fail->Pass + ideas -5.5% investigation

Branch: gl46/geom @ 4ae97c2 (cd967ee + merge b832372 of gl46-port 9442011 = compat s18 + r56 fixes).

## Task 1: real code failures in my domain (geometry / indirect / primitive emulation)

All KHR-GL46 draw_indirect.* (70/70) and pipeline_statistics_query_tests_ARB.* (10/10) now pass —
the two `advanced-twoPass-transformFeedback-*` r22 fails were fixed by r55's indirect work. Running
the full geometry_shader group found 3 real failures (all ABSENT in the r22 sweep, i.e. never ran):

1. `geometry_shader.api.fs_gs_draw_call` and `.api.pipeline_program_without_active_vs`
   (batch-only: identical dummy GS source linked twice in one process):
   root cause — `link_cache_get` (program.c) served a link from the cache but never latched
   `separable_latched` (run_link's tail does). A FIRST link served from the cache therefore reports
   GL_PROGRAM_SEPARABLE = 0 and glUseProgramStages() rejects the separable, linked program with
   GL_INVALID_OPERATION. Fix: `obj->separable_latched = obj->separable;` in link_cache_get's
   success path (the cache key already includes obj->separable, so the clone is the separable build).
2. `geometry_shader.program_resource.program_resource` (standalone):
   root cause — REFERENCED_BY_*_SHADER for GL_UNIFORM_BLOCK / GL_SHADER_STORAGE_BLOCK /
   GL_BUFFER_VARIABLE / block-member uniforms was DECLARATION-based (a declared-but-untouched block
   reported 1 for every stage that declares it). GL 4.6 §9.2: 1 only if the block "contains at least
   one variable referenced by" that stage. Fix in introspect_spirv.c: compute_active() now records,
   per UBO/SSBO variable, a bitmask of top-level members actually accessed in that stage's module
   (SPIRV-Cross spvc_compiler_get_active_buffer_ranges, bit i = top-level member containing the
   access; >64 members / call failure → MGL_BUSED_UNKNOWN = declaration-based fallback), plus a
   function-body reference scan (OpAccessChain/OpLoad/OpStore/OpCopyMemory on the variable) for
   whole-block access; reflect_stage gates `b->stages` and the member `u->stages`/`bvars[].stages`
   bits on it. Declared-but-unused UBOs STAY listed (required by shaders.uniform_block.random, which
   still passes 130/130) — only the stage bits become usage-based.

Results: geometry_shader 136/136 (was 133/136). Regression checks around the introspection change:
program_interface_query 30/43 (13 fails are pre-existing — identical list on the 9442011 baseline
lib), uniform_block.random 130/130, sepshaderobjs 8/8, shader_storage_buffer_object 122/124
(2 pre-existing on base too: advanced-switchBuffers, advanced-indirectAddressing-case2; one
unsizedArrayLength flake did not reproduce in 2 further full-group runs).

## Task 2: ideas -5.5% per-draw CPU investigation (supervisor 18:22)

Environment notes: `nice` + DYLD_INSERT_LIBRARIES dies (arm64e system binaries) — no nice for A/Bs;
homebrew glmark2-macos works once the nsgl dylib is codesigned (install_name_tool invalidates the
linker signature). Built gl46-port 9442011 as the baseline (scratch, not committed).

Findings:
- 9442011+cd967ee merges CLEAN (git merge-tree: 0 conflicts) → the supervisor's merged tree is
  content-identical to my b832372; `git diff 9442011 HEAD -- MGL/src/MGLRenderer+Compat.m` empty.
- Paired, interleaved A/Bs (ideas:duration=10, both /tmp/glmark2-native and homebrew -s 640x480,
  MGL_PROFILE=1 draw-bucket sum over all 120-frame windows + FPS):
  * tip 4ae97c2 vs 9442011: draw-ms 92-97% of base, FPS 98-105% (tip faster or equal)
  * b832372 (= the supervisor's merged content) vs 9442011: draw-ms 95.9%, frames 101.4%
  * single-commit isolation off the tip (revert 621cd67 / 476f0e4 / 1103423, and tip vs b832372
    for r56): all 97-104% = noise (A/A control shows ±5%)
  * drawfast hit/miss identical (87% hit, same dirty patterns); swap bucket keeps compat s18's
    single-pass present (0.127 vs 0.133 ms/frame)
- Early bisect runs that showed +13-30% at 476f0e4/621cd67 were branch-divergence + ambient-load
  artifacts (those intermediates predate compat s18's per-draw cuts; verify2 was building during
  the measurements — window counts swung 46..96 per 10 s run).
Conclusion: no r55/r56 commit adds measurable per-draw CPU on ideas; my tree is within ~1% (in fact
~2-4% faster on the draw bucket). The -5.5% is most plausibly the A/B having run while the
from-scratch verify2 build loaded the machine (their FPS-style numbers; compat's own scripts note
"FPS is present/GPU bound"). If it reappears: reproduce with the paired draw-bucket protocol above
(load-independent), checking that no other build is running.

## Shared-file edits this round (both called out in task 1)
- MGL/src/program.c: link_cache_get success path latches separable_latched (1 line + comment).
- MGL/src/introspect_spirv.c: usage-based REFERENCED_BY_*_SHADER (StageCtx.bused + BlockWalk
  used_mask/used_all/cur_used/block_type + compute_active ranges collection + reflect_stage gating).


Work on gl46/geom (not merged yet): dc23e89, 1103423, 476f0e4, 621cd67, cd967ee (+ merge dfecf27 + a161c5f).

## 1. Reviewed + committed the uncommitted r54 tail (1103423)
GEOMETRY_SHADER_INVOCATIONS counted exactly by the geometry kernel: each valid
(primitive, invocation) thread bumps ctl[21]; the qacc accumulator folds it into
word 8 at every draw; queries.c reads word 8 (the old CPU estimate
count_active(...) removed — it was 0 for indirect draws). Endless-query reset mask
bit 8 wired. test_geom 660/0 → CTS (see §4).

## 2. Flaky SIGTRAP in test_atomic_counters: 3 heap overflows (476f0e4)
Symptom: rc=133 (mfm_free brk #0xc473) in 2/5 runs, always at
free(xformed) ← mgl_msl_inject_atomic_counters. Root cause: three MSL rewrite
passes in geometry.c wrote past their allocation:
* msl_msl_counters_to_helper_params step 6: `malloc(len + 1)` while strcat()ing the
  edit texts — overflowed by their full length (helper functions touching a counter).
* inject_atomic_counters_core: the expression rewrites LENGTHEN the text
  (`(thread atomic_uint*)&n[i]` → `(volatile device atomic_uint*)&n[wu+(i)]`) but the
  allocation only covered the parameter insert. Fixed by a pre-pass that counts the
  exact growth with the same snprintf formats and adds it to the allocation.
* mgl_msl_insert_image_fences: same class (inserts without room). Fixed with
  ins_len = Σ strlen(edit texts).
* mgl_geom_strip_msl_volatile: `& restrict ` → `& __restrict ` GROWS the text; the
  strdup-in-place rewrite now rebuilds through an SB (growing) buffer.
Result: test_atomic_counters 8/8 clean runs (was 2/5 trapping), suite green.

## 3. Indirect draws through the geometry emulation (621cd67) — 2 real bugs
Reproduced as KHR-GL46.pipeline_statistics_query_tests_ARB.functional_geometry_shader_queries
Fail (PRIMITIVES_EMITTED found 6 of 12 / 3 of 12), then with a CTS-side debug patch
(local, reverted) + MGL_GEOM_DUMP_CTL (kept, env-gated):
* firstIndex ignored: for glDrawElementsIndirect the element buffer binds at its
  BASE and the command's firstIndex carries the offset; the prep/tiles/emit kernels
  read indices[i] directly, i.e. vertex bytes as indices (vmin=vmax=0xBDA89C8D =
  the -0.1f vertex data). MGL_IDX now reads indices[first + i] in all three kernels
  (first = 0 for direct draws whose offset is in the binding).
* primcount sizing: the GPU-assembly path sized pools/dispatch by
  worst_prims(mode, bound) × ONE instance (the real counts are GPU-side), so any
  indirect draw with primcount > 1 had an output-vertex budget for a single instance
  and truncated both the raster list and the EMITTED query. mgl_geom_route_indirect
  now takes the command (new param) when the CPU has it — the pipeline-statistics
  readback already read it, or a non-forcing read from the buffer's CPU copy (no GPU
  sync; the GPU-only fallback keeps the old worst-case sizing) — and passes
  known_prims / known_instances / known_vcount for exact sizing in
  MGLRenderer+Geometry.m (bound_threads, vsout_bytes).
  draw_buffers.c: 8-line edit passing `gcmd` into mgl_geom_route_indirect.
Regression test: r55_indirect_firstindex_and_primcount (test_geom 670/0) — one
shared buffer holding vertices + indices + both indirect commands (CTS layout),
5 GS input modes × EMITTED=12 / INVOCATIONS=4 exact.
Note: glBeginQuery with a query object whose immutable target differs →
GL_INVALID_OPERATION is MGL being spec-correct (the test first drafted against it).

## 4. CTS results (this build, pristine CTS tree — the r54 G54 instrumentation was
   left in the tree by r54; reverted + glcts rebuilt first)
* texture_swizzle.smoke: **Pass**, 284 s (quiet) / 413 s (loaded) — target < 19 min.
* pipeline_statistics_query_tests_ARB.functional_geometry_shader_queries: **Pass**
  (was Fail r22; GEOMETRY_SHADER_INVOCATIONS + PRIMITIVES_EMITTED now exact).
  Also Pass: functional_compute_shader_invocations,
  functional_primitives_vertices_submitted_and_clipping_input_output_primitives,
  api_coverage_unsupported_calls, draw_indirect.advanced-twoPass-transformFeedback-
  {arrays,elements}, shader_atomic_counters.advanced-usage-{multi-stage,switch-programs},
  CommonBugs.CommonBug_ReservedNames (was Crash r22),
  shader_image_load_store.basic-allTargets-store, texture_swizzle.api_errors,
  texture_swizzle.intial_state.
* texture_swizzle.functional: STILL FAILS (as in r22) — one sub-case of ~12k:
  GL_TEXTURE_3D + RGTC1 + swizzle [A,B,G,R] + texelFetch, expected 255 got 0. The
  CTS's own error logger then SIGSEGVs (strlen(NULL) from glu helpers for this
  format combo — upstream CTS bug; crashes the process before logging). MGL side:
  a 3D/2D_ARRAY texture whose GL internal format is RGTC1 ends up stored as
  RGBA8Unorm (debugged via a temporary MGL_TEXPARAM_DEBUG in texparamTextureForTex:
  "VIEW NEW tex=1300 tgt=37120 ifmt=0x2a10 fmt=70"), so the A that GL defines as 1
  for a RED format reads as the stored 0. Isolated repros (2D RGTC1 via compressed
  upload, R8, and the CTS's RGB/565-into-RGTC1 path) all PASS — only the 3D/2D_ARRAY
  storage choice is wrong (needs the channel-correct decoded equivalent, R8).
  Open question for the textures/packed_pixels owner: textures.c rejects
  glTexImage3D(RGTC1) for byte-exact glGetCompressedTexImage (r54, packed_pixels),
  while this CTS case requires it to succeed — a shared decision is needed; left
  as the one known remaining swizzle failure. All MGL/CTS debug patches reverted.

## 5. gl46-port merge (tess r25, 5178835) — dfecf27 + a161c5f
Conflicts: msl_reserved.c (kept BOTH int-builtin fixers: geom's
mgl_msl_fix_int_builtins — wraps every use — and tess r25's
mgl_msl_fix_vertex_id_builtins; the resolution dropped a closing brace, fixed in
a161c5f) and program.c (kept both the r54 texelFetch-srgb block and tess's
vertex-id-builtin call). tess r25 also de-duplicated a block in my
mgl_geom_tess_xfb_route (kept). Build clean (the one new warning, MGLRenderer.m
'extProgramLibraryForMSL:key:error:' incomplete-implementation, comes from tess's
MGLRenderer_Private.h declaration, not from this branch's files).

## 6. glmark2 (Terminal launch, default mode 800x600, on-screen)
Visuals (window captures, 3 per scene): ideas (67,66,65) neutral grey — no leak
band, no logo; jellyfish (62-65, 87-90, 151-163) bell blue — no flashes; gl41-
geometry (62-77, 82-96, 131-151) the expected grid. Magenta: 0.000-0.006%
(antialiased edge pixels, same on gl46-port).
FPS, interleaved A/B at equal load (load 3.4-8; the machine is shared and ran a
macOS contactsd storm at times): gl41-geometry geom 1811/1799 vs gl46-port
1832/1853 (−2%); jellyfish geom 5454/5433 vs 5367/5353 (+1.5%); ideas geom
514-552 vs 543-586 (within the ±6% round-to-round noise). No regression from this
branch's changes. Quiet-machine absolute numbers (2800/12450/3225) were not
reproducible today — the machine never went below load ~3.4.

## 7. Gates (all on cd967ee)
* tests/headless/run.sh: 96 binaries, **0 FAILED** (8m52s).
* test_geom 10×: 10/10 × 670 checks, 0 failures.
* test_fp64fast 3×: 3/3 × 298 checks, 0 failures.
  (An earlier flaky test_fp64 failure — fp64_scalar_ops abs() returning the input —
  reproduced identically on the pre-heap-fix build, i.e. pre-existing/latent, and
  did not reappear after the tess r25 merge: 12/12 clean + 3/3 in the suite.)
* stub_scanner: EMPTY/NOOP/ASSERT clean for MGL/src/geometry.c and
  MGL/src/MGLRenderer+Geometry.m.
* Compat ownership: diff of MGL/src/MGLRenderer+Compat.m vs gl46-port is empty.

## Shared-file edits this round
* draw_buffers.c — pass the CPU-known indirect command to mgl_geom_route_indirect
  (8 lines; the GPU-only path is unchanged).
* mgl_ext_geom.h — mgl_geom_route_indirect signature + MGLGeomDraw.known_* fields.
* msl_reserved.c / program.c — tess r25 conflict resolutions (both sides kept).
* (temporarily instrumented, all reverted: MGLRenderer+TextureParams.m debug prints,
  the CTS tree's gl3cTextureSwizzleTests.cpp + glcts rebuild.)

## Scratch (untracked, not committed)
tests/headless/scratch_{swz55,gm55,gm55v,gsind,gates55,tf55,lldb55,lldb55b,g54sub,g54tr,rgtc,gm55fps,gm55ab}.*,
build-asan/, Makefile.asan, Makefile.e, build-e/, TestResults.qpa, /tmp logs.
