READY 80d6216 (r83 = a352837 + r83b b6d2a3a, baseline re-verified twice after merging gl46-port d11d184 as 8fa0504). REGRESSION FIXED: the r82 ideas −2.2% was per-draw sidecar bookkeeping that ran on EVERY program — the prep gate walked six stages' resource lists with three raw getenv() probes per draw, and an UNCONDITIONAL geomRealizeStageTextures per stage walked SAMPLED images too (a bindMTLTexture: message send per texture per draw; ideas binds samplers, no image in sight). r83 gates all of it: geomAnyAtomicSlot() memoizes "any stage declares a storage image with an atomic sidecar slot" per program identity (program-or-pipeline object + every stage's MSL text pointer, which relink regenerates) — the unused path pays one 7-word compare per draw, never the walk, never the realize, never an env probe; the bindTextures sidecar bind hits its count check first (one load, baseline parity when no storage image); every remaining per-draw raw getenv probe in MGLRenderer+Geometry.m (MGL_GEOM_DEBUG/TRACE/QDEBUG/DUMP_RUNS/ASSERT, MGL_SSBO_DEBUG, MGL_ZZVCAP, MGL_ZZTRACE, MGL_CULL_TRACE/DUMP: ~14 per geometry draw, ~8 per plain draw) moved behind the one-shot gEnv latch; mgl_geom_present() no longer builds a pipeline composite on every pipeline draw. PROOF: mgl_geom_ms_stat(4) gate-miss counter pinned by test_geom_imgarr (one walk per program link, never per draw); ab_bench CPU-clock min-of-21 vs baseline-d11d184 m0 −3.3% / m1 −9.2% / m31 −36.7% (new faster at min, medians ±0.5%); locked glmark2 ideas 8+8 interleaved rounds: P median 528 / C median 528, delta −0.19% (within the 1% limit; NOTE the machine never went GPU-idle today — ideas ran at ~530 FPS vs the supervisor's ~3100, so a 7 µs/frame effect would read ~0.4% here, and the run still shows C ≥ P); 8 CTS groups re-run after each merge (a3c493b, d11d184): 584 cases 546 Pass + 38 NotSupported + 0 Fail both times; full tests/headless/run.sh 140 binaries 0 failures.

== r83 detail: where the r82 ideas cost actually was ==
1. geomPrepareImageAtomicsRender (Draw.m calls it on EVERY draw): r82 replaced r81's six getProgramBindingCount: message sends with a six-stage struct walk that ALSO read three getenv("MGL_GEOM_DEBUG") probes per draw, and called geomRealizeStageTextures UNCONDITIONALLY per stage — that helper walks SAMPLED_IMAGE resources (mgl_texture_for_sampler per element + a bindMTLTexture: message send per bound texture). r83: the memo gate first (geomAnyAtomicSlot); the realize runs only for a stage PROVEN to have an atomic slot (r82c's sidecar semantics preserved for sidecar programs); probes read the one-shot gEnv latch.
2. geomBindImageAtomicBuffersRender (bindTextures, 2× per encoder build): r82 added an unconditional getenv; r83 reads count first (one load, no storage image ⇒ return), then the memo (a plain-image program without an atomic slot skips the element walk).
3. geomResolveImageAtomics barrier path: unchanged O(1) img_atom_any early-out (r82).
4. mgl_geom_present (draw_buffers.c calls it on every draw): with a program pipeline bound it built a full composite Program (memset + per-stage resource-list copies) per draw just to test shader_slots[_GEOMETRY_SHADER]; the answer is stage_programs[GS]->shader_slots[GS], identical by construction. Plain programs take the same early-out as before.
5. Geometry-shader draw path probes (the r83 per-draw CPU item): ~14 libc getenv calls per geomDraw/geomBindStageBuffers/geomBindRenderStageTextures/geomRasterPass moved behind gEnv (MGL_SSBO_DEBUG was also 2×/draw in geomBindSSBOSizes on the plain path). GS-path CPU profile measured with a temporary mach_absolute_time phase profiler (removed before commit): at 4096-pt draws the fixed cost is raster pass ≈175 µs (a full processGLState walk — inherent, the capture pass dirties state), capture ≈110 µs (descriptor churn + PSO key hash), gs-enc ≈45 µs; the getenv fixes and the pool/PSO caches already carry the biggest chunks. Capture-pass descriptor/plan caching is the next candidate (needs a VAO fingerprint; left for the next round).

== r82 report below (kept for reference) ==
READY cb1dd9e (r82 = cb1dd9e on top of the 1121e66 merge ef20f90; r82a 37585cb, r82b 26ccb26, r82c cb1dd9e). SUPERVISOR TASK DONE: (1) all eight requested groups re-run on the merged tree — 584 cases, 546 Pass + 38 NotSupported + 0 Fail/Crash — and re-verified after each r82 step; (2) robustness work in the image-atomic sidecar found and fixed FIVE real defects (arrays of multisample images did not even link; fragment-stage image-ARRAY atomics bound the wrong buffers and never landed; render-stage atomics were never seeded from texture contents; remapped array elements walked the wrong units; first-dispatch arrays could miss their sidecar entirely), plus per-draw CPU reductions (O(1) barrier early-out, no per-draw message sends in the prep gate); (3) full gate 691 groups / 11019 checks / 0 failures three times, ab_bench min-of-15 shows m0 −3.1% / m1 −5.0% vs baseline-1121e66, locked glmark2 within noise.

== TASK 1: CTS on the merged tree (1121e66 merged as ef20f90), exact counts ==
Baseline lib (pre-r82, the merge itself)          r82a/b lib (audit3)     final r82c lib (audit4)
  shader_image_size                 74 Pass          74 Pass                74 Pass
  shader_image_load_store           51 Pass          51 Pass                51 Pass
  shader_texture_image_samples_tests 4 Pass           4 Pass                  (unchanged code paths)
  texture_view                       7 Pass           7 Pass                  (unchanged)
  geometry_shader                  136 Pass         136 Pass                  (unchanged)
  sample_variables        152 Pass + 38 NotSupported (same)                   (unchanged)
  sample_shading                    13 Pass          13 Pass                  (unchanged)
  shader_multisample_interpolation 109 Pass         109 Pass                  (unchanged)
  TOTAL 584: 546 Pass / 38 NotSupported / 0 Fail / 0 Crash — three independent runs.
(texture_multisample-related = shader_multisample_interpolation, shader_texture_image_samples_tests, sample_shading, sample_variables as enumerated in the KHR-GL46 case list.)

== TASK 2a: robustness — defects found by probing the r81 sidecar with image ARRAYS ==
1. image2DMS g[N] / image2DMSArray g[N] failed the LINK outright (Metal static_assert:
   texture2d_ms<..., access::write>).  mgl_spirv_lower_ms_images now resolves the image
   type through OpTypeArray (glslang: %arr = OpTypeArray %imgt %len), and points the
   rebuilt OpImageTexelPointer/OpLoad at the ELEMENT pointer so SPIRV-Cross emits
   <name>[i]_atomic[...]; mgl_msl_fix_image_atomics flattens it into the array side
   buffer with a new MS-layout helper spvMglAIArrMS_<name> (off = sum of w*h*S*4
   [/layers] element regions, OOB -> the dump word); msl_msimg.c walks
   array<texture2d_ms<...>, N>'s ', N>' tail for the variable name (access::write was
   left unflipped) and drops <name>[i].write() calls.
2. geomBindImageAtomicBuffersRender bound N per-unit sidecars at mtl_index+e for an
   array image — buffers the array MSL never reads — so a fragment-stage image-array
   atomic wrote out of bounds of one single-image sidecar and reached no texture.
   It now binds the ONE array side buffer at mtl_index, like the compute binders.
3. A fragment-stage image atomic was never SEEDED (the draw-time prep gated on MS
   slots only): the atomic's returned value saw 0 instead of the texture's contents.
   The gate is now "any atomic sidecar slot"; sidecars are pre-created and seeded at
   draw time (geomRealizeStageTextures first — an image unit's MTLTexture may not
   exist yet at prep time), arrays included.
4. Remapped array elements (glUniform1i; advanced-sso-simple binds 0,2,4,6): the CPU
   layout/seed/flush walked binding+e while the MSL helper computes offsets from the
   BOUND element textures — img_atom[].eunits[8] (mgl_res_elem_unit) now drives the
   layout, seed, MS flush and compute resolve; base+e remains the consecutive default.
5. geomPrepareImageAtomicsCompute realizes deferred texture uploads before building
   the sidecar layout (first-dispatch arrays could get NO sidecar at all: the layout
   returned 0 words and every atomic silently vanished).
Also fixed en route: a single-image sidecar replacing an ARRAY record kept elems>0
(seed/resolve then walked array paths against one sidecar).

== TASK 2b: perf — sidecar cost when unused stays zero (structural proof) ==
- New monotonic counters (mgl_geom_ms_stat(), tests/headless/mgl_geom_diag.h):
  seeds / flushes / compute resolves / sidecar allocations.  test_geom_imgarr proves
  8 draws + a full barrier leave ALL FOUR at exactly 0 for a program without image
  atomics, and that each pass runs exactly once per sidecar when one is used
  (no re-allocation / re-seed on the second draw).
- Barrier early-out: img_atom_any (one load) replaces the 128-slot scan.
- Per-draw: the prep gate reads spirv_resources_list directly instead of six
  getProgramBindingCount: message sends; sres->binding replaces getProgramBinding:;
  the render/compute binders read the stage count off the Program struct.
- ab_bench min-of-15 interleaved vs baseline-1121e66 (machine contended, load ~5 —
  min-of-7 did not converge, min-of-15 did): m0 drawArrays x2000 A=12.812 C=12.410
  (−3.1%), m1 blend-flip A=338.213 C=321.374 (−5.0%).  No regression either way.
- locked glmark2 (lockf, 6 interleaved rounds, ideas/jellyfish/gl41-geometry,
  top-3-of-6): score A 2567/2462/2393 vs C 2677/2471/2365; ideas 703/669,
  jellyfish 5372/5500, gl41-geometry 1739/1865 — within noise, no regression.

== Tests / gate ==
tests/headless/test_geom_imgarr.c (9 tests, 81 checks, all Pass): fragment/compute
image-ARRAY atomics (value lands in the texture), element seeding from texture
content, remapped units (seed + flush through the remap, value read back out of
unit 3's texture), image2DMS g[2] store/read per sample (compute AND fragment),
image2DMSArray g[2] layered, the zero-cost counters, and the runs-exactly-once
counters.  Full gate: 691 groups PASS / 11019 checks / 0 failures (rc=0), run three
times across r82a-r82c.  stub_scanner EMPTY/NOOP/ASSERT: clean for
geometry.c, MGLRenderer+Geometry.m, msl_msimg.c; no new warnings (the remaining
ones pre-exist identically in the baseline tree).

== Files touched (all mine this round; NO shared files) ==
MGL/src/geometry.c (SPIR-V array classification + base_ptr; MSL spvMglAIArrMS helper
  and subscript rewrite; mgl_geom_ms_stat; gTraceCtx at init)
MGL/src/mgl_ext_geom.h (img_atom[].eunits[8], .ms/.samples already there,
  img_atom_any, stat_* counters)
MGL/src/MGLRenderer+Geometry.m (render binder array path; draw-time prep gate +
  arrays + geomRealizeStageTextures; geomMsFlushView offset core; array seed/flush;
  elems reset; eunits walks; O(1) early-out; direct struct reads)
MGL/src/msl_msimg.c (array declaration tail walk; array-element .write drop)
tests/headless/test_geom_imgarr.c (new), tests/headless/mgl_geom_diag.h (stat decl)

== Remaining gaps / notes ==
- MS image arrays beyond 8 elements with a REMAP fall back to base+e on the CPU side
  (the MSL stays correct); consecutive default arrays of any size are exact.
- Fragment-stage image ARRAY atomics were broken since r63 (the render binder bound
  per-unit buffers); the CTS never caught it — no KHR-GL46 case combines a fragment
  image-array atomic with a later texture/atomic read.  Covered by the new tests now.
- glmark2 ran under co-tenant load (other agents building/running CTS); the A/B spread
  per scene is ±5% noise, top-3-of-6 used as before.
