# compat domain — progress log
## session 24 (Oct 28, 01:57-02:4x) — merge gl46-port aafc7c2 (tess r30), conflict resolved, gate re-run

### after READY: fails_compat3 finished (47d7336, on top of the READY)

tessGeom sub-case of KHR-GL46.gl_spirv.spirv_validation_builtin_variable_
decorations_test Fail->Pass, and with it the whole fails_compat3 queue
(gl_spirv 9/9, separable_programs_tf 2/2, shader_group_vote 5/5).

A geometry stage fed by tessellation could not read gl_in[].gl_ClipDistance /
gl_CullDistance AT ALL: the link refused (honest gap error — the TES-output
capture skipped every non-Position builtin), so the tessGeom validation failed
on BOTH iterations (GLSL and glShaderBinary; reproduced standalone in
tests/headless/test_svv_tessgeom.c before fixing). Fix, three pieces:
1. tessellation.c `mgl_tess_capture_msl`: the tess lowering already copies the
   evaluation stage's gl_ClipDistance/gl_CullDistance outputs into per-element
   user attributes (gl_ClipDistance_0, ...) — capture those members (one float
   word each, offsets after the varyings) instead of skipping them.
   mgl_ext_tess.h capture contract updated.
2. geometry.c `geom_out_decl_size()`: reads a stage module's declared OUTPUT
   gl_PerVertex builtin sizes (prefers the Output interface — an evaluation
   module declares gl_in[] too and the sizes may differ); `geom_tess_clip_check()`
   replaces the blanket link error with "legal iff the evaluation stage declares
   the outputs"; the from_tess link path now sizes in_clip_size/out_clip_size
   from the TES (mirroring the vertex-stage rule) instead of zeroing them, so
   geom_layout gives the input record real clip/cull words.
3. geom_route_tess: the import kernel table gains (capture element word ->
   in_clip_off/in_cull_off) pairs, found by the flattened member names.
Post-fix verification: CTS gl_spirv 9/9, geometry_shader 135/136 (the 1 fail =
api.max_image_uniforms, fails identically on pristine aafc7c2 AND the
pre-merge tip — the documented MS-image caps gap), tessellation_shader 57/58
(unchanged), separable_programs_tf 2/2, full headless suite 617 PASS / 0 FAIL,
stub scanner clean; short locked glmark2 sanity (3 pairs, alternating):
ideas median 2803 vs 2740 (+2.3%, one load dip each side), jellyfish 25352 vs
25947 (-2.3%, B dip 21376) — parity within noise, and the change is off the
glmark2 path entirely (no tess+GS program there).

READY s24 08b19dc (report commits on top touch only agents/compat_report.md) —
merge gl46-port aafc7c2 into 52dcd0d, conflict resolved, gate re-run. The
merge-tree was NOT clean (supervisor correction): MGL/src/tessellation.c conflicted
because BOTH sides had independently landed the records-only post-link cleanup —
upstream r29 named it `tess_drop_program_record`, mine `tess_forget_records`
(identical bodies modulo my defensive `!S` guard). Resolution keeps BOTH behaviours:
* upstream's name/structure + r30's content (gl_MaxPatchVertices handling, lazy
  sep-vs `words` caching, the `free(vs_kernel.words)` fix, the drop trace);
* my `!S` guard, and — the behaviour that must survive — RECORDS-ONLY cleanup on
  the TES-carrying relink path (`mgl_ext_post_link_tess`), where upstream r29 had
  gone back to the full `mgl_ext_delete_program_tess` (purging the sep-vs kernel).
  Safe because `sep_vs_kernel_cache` runs on BOTH link paths (compile_stage and the
  whole-link cache hit, program.c 3355/4175) and `mgl_tess_cache_vs_kernel` replaces
  a changed MSL in place; a purge would drop the fresh entry
  (KHR-GL46.separable_programs_tf.geometry_active).

Gate (clean `rm -rf build; make -j4 lib glx nsgl`, rc=0, no new warnings — the one
`-Wincomplete-implementation` for extProgramLibraryForMSL in MGLRenderer.m is
pre-existing upstream: none of MGLRenderer{,_Private.h,+Program.m} changed in the
merge, the method lives in the +Program category):
* test_tess{,_dynio,_images,_buf,_cap,_emu}: all PASS (test_tessemu 117 checks).
* KHR-GL46.separable_programs_tf: **2/2 Pass** (the conflict-resolution group).
* KHR-GL46.tessellation_shader: **57/58** — the 1 Fail
  (`single.primitive_coverage`, pixel mismatch idx 8188) reproduces IDENTICALLY on
  pristine gl46-port aafc7c2 (/tmp/gp30b build) AND on my pre-merge tip c66f3dc
  (/tmp/premerge build), so it is a pre-existing upstream regression in the r27-r29
  window (r26 reported 58/58), NOT from this merge — for the tess agent. My
  resolution cannot affect it (no relink in that case).
* Full headless suite: **615 PASS / 0 FAIL**, 119 files "== N checks, 0 failures",
  9507 checks, rc=0 (up from 614/609: r30's new tests).
* test_fp64 + test_fp64c 10x: 417/0 + 48/0 every run, 0 flakes.
* ab_bench vs ~/MGL-cts/baseline-aafc7c2 (min of 7, interleaved, order alternating):
  **m0 13.026 -> 12.506 ms (-4.0%), m1 164.618 -> 133.719 ms (-18.8%)** (medians
  15.0/15.2 and 182.0/157.1; r5 old-m0 142.9 ms is a load spike, min discards it).
* glmark2 locked (`/usr/bin/lockf -k /tmp/mgl_glmark2.lock`), interleaved 6 pairs,
  order alternating: ideas median 2502.5 vs 2558.5 (**-2.2%**, single-run spread
  +-3%, B max 2645 > A max 2583 — parity within noise), jellyfish 24970 vs 24935.5
  (**+0.1%**, parity).

Remaining (supervisor queue): none of fails_compat3 left open — see the
"after READY" section above for the tessGeom fix that closed it.

---

READY s23 52dcd0d (bf253e9 + merge gl46-port c6ded7d). fails_compat3 remainder done:
- KHR-GL46.separable_programs_tf.geometry_active Fail->Pass (group 2/2). Four bugs found
  with a single headless repro (tests/headless/test_sptf_geom.c, VS+TCS+TES+GS+FS
  separable programs in one pipeline, TF PRIMITIVES_WRITTEN + captured data):
  1) transform_feedback xfb_active_program preferred the TES program over the GS (GL 4.6
     13.4.1: geometry > evaluation > vertex) so TF captured o_tess instead of o_geom;
  2) the tess post-link hook ran the full mgl_ext_delete_program_tess for LIVE programs,
     purging the separable-vertex KERNEL the link itself had just cached -> every pipeline
     draw failed with "tess VS stage function is not a kernel"; records-only cleanup is now
     split out (tess_forget_records), real deletes keep the purge;
  3) the whole-link cache hit clones the executable without compile_stage, so a second
     program with identical sources never cached the kernel under its own name (the CTS
     runs tessellation_active then geometry_active with the same sources) — sep-vs caching
     extracted to sep_vs_kernel_cache() and called on both paths;
  4) geometry VS-cull clip presence from the module must use the STORE scan, not the
     declaration: glslang always emits the full gl_PerVertex block, so every program
     "declared" clip, the record grew a clip region, cull offsets shifted and culling broke
     (test_cull lines_dynamic_index_writes caught this; fixed in the same commit).
- KHR-GL46.gl_spirv.spirv_validation_builtin_variable_decorations: validations 1
  (clipNegative) and 3 (point) Fail->Pass on the SPIR-V iteration — geom_cull_reflect
  derived in_has_clip/in_has_psize from the GLSL SOURCE only, and a program linked from
  SPIR-V binaries has none: the synthetic cull pass-through dropped every gl_ClipDistance
  (nothing clipped) and gl_PointSize (1px points). geom_builtin_decl_size() reads the
  declared member width from the module; presence uses the store scan (see 4).
  gl_spirv group 8/9 Pass. Remaining: the tessGeom sub-case needs the tess capture record
  to carry gl_ClipDistance/gl_CullDistance words for a following geometry stage (today the
  capture skips other gl_* builtins: tessellation.c "other builtins are not captured") —
  that is a tess*geom interface change, left for the next session.
Gate (after the c6ded7d merge, clean -j4 build): full headless suite 614 PASS / 0 FAIL
rc=0; test_fp64+test_fp64c 10x = 0 fails; gl_spirv 8/9, separable_programs_tf 2/2. A/B vs
~/MGL-cts/baseline-c6ded7d: ab_bench m0 11.829->11.661 ms (-1.4%), m1 100.687->101.977 ms
(+1.3%) (min of 7, two interleaved runs); locked glmark2 medians ideas 2436->2510 (+3.0%
faster), jellyfish 25345->24998 (-1.4%) — r7 jellyfish A was killed by the supervisor's
01:35 glmark2 restart; medians computed from 7 clean pairs.
READY s23 ef59367 (= ead3d39 + gl46-port b7821df merge + test fixes). Merged tess r28+r29
(3d03a6b): only conflict tests/headless/test_vote.c (add/add) — kept BOTH sets (the CTS
extension-less spellings + my plain #version 460 core ones under vote_core_*; 9/9 PASS).
4d74caa had committed test_capszoo.c with a dropped `if (!ok) {` (BUILD-FAIL in run.sh);
restored (ef59367), caps_zoo_compile_link PASS. Clean gate: rm -rf build; make -j4 lib glx
nsgl; full suite 609 PASS / 0 FAIL / rc=0; test_fp64+test_fp64c 10x = 417/0 + 48/0, 0 flake.
A/B vs ~/MGL-cts/baseline-b7821df (interleaved): ab_bench min-of-42 m0 base 11.418 -> new
11.116 ms (-2.6%), m1 102.455 -> 102.053 (-0.4%) — first 7-round window read +10.7% but the
distributions overlap (standalone new samples 9.7-15.4 ms); per-draw MGL_PROFILE buckets
(abb_prof.c, min of 5) show NO bucket differing > 0.15 ms / 2000 draws (<75 ns/draw), so the
wall-clock spread is machine load, not code. pipeHit 19.0/frame ideas, 0 declines (= baseline,
100% fast-path accept survives the merge). Locked interleaved glmark2 (run_ab_s23b.sh, 6 ideas
+ 8 jellyfish, order alternates): ideas median A 2549.5 vs B 2617.5 (-2.6%, single-run spread
+-3.5%), jellyfish A 25294.5 vs B 25221.5 (+0.3%) — parity within noise.
READY s23 ead3d39 (supersedes the rejected 1605445). Registration fix + two correctness
fixes on top of 944befc (gl_spirv fp64/spirv fixes): 1605445's fnid-registry fast-path key
declined EVERY draw on programs whose functions were installed outside extProgramCompileMSL
(bindMTLProgram's library-cache path, rstate's fixup recompile) — mglProgramRegisterMSLIdentity
now registers all install paths (once per install), MGL_FNID_CAP 32768. Gate: ab_bench m0
10.861->10.933 ms (+0.66%), m1 176.1->104.0 ms (-41%); pipeHit 19+2/frame = 100% accept
(baseline parity; 1605445 was 100% decline); locked interleaved glmark2 medians ideas +35%
/ jellyfish +5% (B>=A, shared-GPU noise, single-run spread +-35%); rm -rf build gate:
584 PASS / 0 FAIL / 114 zero-fail files; test_fp64 10x 417/0 (0 flake); test_fp64c 48/0
(fp64 frexp/modf *Struct lowering duplicated the definition — link fail, now fixed);
test_capszoo PASS (gl_spirv capabilities image zoo: writable-MS-image MSL emulation,
msl_msimg.c; shader transcribed verbatim from the CTS source).
READY s23 1605445 (gated code; report commits on top touch only agents/compat_report.md). Contains
gl46-port 6f9b359 (geom r55-r58) via merge d7b5cce, conflicts
resolved keeping both border-emu and wrap-op guards in rstateSpecialize. CRITICAL FIX for the
merged tree: gl46-port 6f9b359 made test_fp64 fp64_scalar_ops flaky (3-4/10 runs, own-tree
repro /tmp trees; 7478e0a clean 10/10) — the compat pipeline fast path keyed PSOs on a sampled
MSL key (string pointer + first 192 + last 128 bytes), which ALIASES for programs whose MSL
differs only in the middle (fp64-lowered operators) once program/string addresses recycle in
a create->draw->delete loop: one program then drew with another program's PSO (abs rendered
-ua, roundeven rendered ceil). key.a is now the fnid registry's full-MSL content hash per
raster stage (registered at link in extProgramCompileMSL); unregistered/uncompiled stages
decline the fast path. After the fix: test_fp64 22/22 clean (12 idle + 10 under concurrent
GPU load). Gate: full suite 112 files 9432 checks 0 FAIL, test_geom 10x 603/0, test_fp64fast
3x 298/0, flash_check PASS (10:49, median 114.1; the 12:1x reruns were INVALID — glmark2's
window occluded by other agents' windows, present path untouched by 1605445; rerun when the
display is free). Carries the s22 fails_compat2 queue 13/13 -> 0 Fail (buffer_storage 9/9,
copy_image 13/13, direct_state_access 371/371, texture_border_clamp 32/32) plus fails_compat3
progress (committed, all green): shader_group_vote 5/5 root cause + SPIR-V opcode rewrite,
gl_spirv LocalSize + glCompileShader error path (3 of the 5 gl_spirv cases fixed at the source
level). Details in the session 23 section.
READY s22 1f74da1 — CTS queue 13/13 -> 0 Fail (fails_compat2): buffer_storage 9/9,
copy_image 13/13 (GL_TEXTURE_1D_ARRAY layer-in-Z semantics), direct_state_access 371/371
(RGTC online compression, GPU-side decoded level + CPU re-encode for downloads; Guard
Malloc caught a 2x plane underallocation that showed up as random malloc traps),
texture_border_clamp 32/32 (already green from s20, re-verified). Gate: full suite
9233 checks 0 FAIL, test_geom 10x 603/0, test_fp64fast 3x 298/0, flash_check PASS
(n=120 median 115.9, 16 foreign-window frames only). Details in the session 22 section.
READY s20 100d132 — CTS queue 21/21 -> 0 Fail: draw_elements_base_vertex (as_uint MSL fix
+tess VS ids), sample_shading 13/13, shader_multisample_interpolation 109/109; flat-varying
provoking rotation was dead (probe missed [[user(locnN), flat]]); PSO-failure path no longer
wedges glFinish. Gate: full suite 0 FAIL, test_geom 10x 603/0, test_fp64fast 3x 298/0,
flash_check 3x PASS (glmark2 now under /tmp/mgl_glmark2.lock). Geom-emulation gaps found and
documented below (GS raster provoking/basevertex/vcount, gl_VertexID=0 through GS).
READY s19 9ed54ae — per-draw CPU: compact VAO layout key + MGL_DF_REBIND respec tier + bindList
skip. Gate: full suite 7880 checks 0 FAIL (3 runs), test_geom 10x 603/0, test_fp64fast 3x 298/0,
flash_check PASS (7/7 no flash), no logo/magenta (terrain+bump+jellyfish+grid+shadow+idea visual).
ideas CPU/frame A/B (quiet, 3 interleaved rounds): draw 1.713 -> 1.350 ms (-21%), processGLState
0.610 -> 0.186 ms (-70%); FPS 265 -> 285 (+8%).  Details in the session 19 section below.

## session 20 (Oct 26, 23:26-...) — s20: conformance queue part 2 (basevertex, interpolateAtOffset)

## session 23b (Oct 27, 16:00-22:15) — rejected-READY root cause: fnid registration on every MSL-install path
Supervisor REJECTED 1605445 on speed (interleaved vs baseline-6f9b359: ideas 3182->1215,
jellyfish 13586->6073). Root cause measured, not guessed:
* MGL_PROFILE counters (mgl_prof_cnt 16/17, dumped as pipeHit / pipeMiss-decline):
  baseline 6f9b359 = pipeHit 19.0/frame [ideas] + 2.0 [jellyfish], 0 declines;
  1605445 = pipeMiss-decline 19.0 + 2.0 — **100% decline, every draw took the descriptor
  path**. The geom link-cache suspicion was close but the actual holes: (a) bindMTLProgram
  installs VS/FS straight from the MSL library cache (fresh MTLFunction per link,
  compileShader:), never registered; (b) rstateRecompileProgram installs the FIXED-UP
  functions (rs_post_link rewrote spirv[stage].msl_str, so bindMTLProgram later sees a
  non-NULL library and skips) — also unregistered. glmark2's programs go through exactly
  these two paths.
* Fix: mglProgramRegisterMSLIdentity(fn, msl, entry) in MGLRenderer+Program.m (next to the
  registry: fnidRegister + fnidOf(full MSL + entry)); called at both installs. Hash computed
  ONCE per install (link/fixup), never per draw. MGL_FNID_CAP 8192->32768 (ids are never
  removed; >2k program relinks saturated it and registration degraded to decline-all).
* Verified: fixed build pipeHit 19.0 + 2.0/frame, 0 declines (= baseline). NOTE /tmp was
  wiped by the 15:45 reboot — glmark2 data restored from /opt/homebrew/share/glmark2 (now
  used directly, per supervisor 16:40).
* Two pre-existing suite failures found by the fresh gate (both in my domain, NOT caused by
  the WIP; fp64_out_param_fns failed at e3af5fe/944befc too, both fp64 lib modes):
  - test_fp64c fp64_out_param_fns: rw_extinst_val's GLSLstd450FrexpStruct/ModfStruct emitted
    the lowered OpCompositeConstruct with rid AND returned 0; 0 = "copy unchanged" since
    944befc, so the raw OpExtInst stayed beside it -> two defs of one id -> MSL "redefinition
    of _NN" + ambiguous frexp/modf(double) -> link fail. Now constructs into a fresh id and
    returns through emit_result like every other case. 6/6 (48 checks).
  - test_capszoo (untracked stuck-session draft): shader had calls the real CTS source does
    not (texelFetch on sampler2DRect — no glslang overload; gl_FragDepth in a compute
    shader). Transcribed verbatim from gl4cGlSpirvTests.cpp SpirvValidationCapabilitiesTest.
    Then the real failure: `texture2d_ms<float, access::write>' is illegal MSL. New
    MGL/src/msl_msimg.c mgl_msl_fix_ms_image_writes(): _ms declarations access::write/
    read_write -> read (same-length in-place) and their `<name>.write(...)' calls -> (void)0
    (store dropped; documented gap, MS image reads unaffected). Called from program.c's MSL
    fixup chain (shared file, +9 lines).
* Gate (rm -rf build; make -j4 lib glx nsgl): 584 PASS / 0 FAIL (114 files "0 failures");
  test_fp64 10x 417/0; ab_bench (new supervisor rule) m0 10.861->10.933 ms (+0.66%, limit
  +3%), m1 176.1->104.0 ms (-41%, the s21/s22 fast-path win restored); locked interleaved
  glmark2 medians (8 rounds jellyfish, 6 rounds ideas, concurrent CTS GPU load, single-run
  spread +-35%): ideas B +35%, jellyfish B +5% (B>=A); MGL_PROFILE steady state
  (2nd dump): 214.8 vs 212.3 ms/120 frames (+1.2%), draw bucket 0.628 vs 0.606 — the
  residual is ~31ns/draw of fnid lock+probe per raster stage in the fast path.
* Known residuals: (a) that ~31ns/draw lookup (could be made lock-free read; left as-is,
  inside the +3% budget); (b) jellyfish FPS on this shared box is noise-dominated — the
  supervisor's own A/B decides; (c) caps zoo still stores nothing to MS images (gap
  documented in get.c GL_MAX_IMAGE_SAMPLES=0).
## session 23 (Oct 27, 08:40-...) — s23: merge gl46-port 6f9b359 + PSO-alias fix, then fails_compat3 queue

Queue: ~/MGL-cts/mgl-r28-sweep/fails_compat3.txt (11 cases: gl_spirv 5, shader_group_vote 5,
separable_programs_tf 1). Rules: -j4 build, one glcts batch at a time, nice.

### 0. merge + gate (supervisor requests 10:25/11:26)
- `git merge gl46-port` (6f9b359 = geom r55-r58): one conflict, MGLRenderer+RenderState.m
  rstateSpecialize MSL-rebuild guard — kept BOTH the compat border-emu skip and geom's
  mgl_wrap_op_ skip. Merge = d7b5cce.
- test_fp64 fp64_scalar_ops came up flaky on the merged tree (24/5/3 failures, bursty).
  Isolated: same on clean gl46-port 6f9b359 trees (git archive to /tmp/gl46port +
  /tmp/g59, perl-chdir so the relative install name resolves); 7478e0a 10/10 clean;
  1f74da1 8/8 clean; my merged tip 4/10 idle, worse under concurrent GPU load.
  => regression is IN geom r55-r58 content, not in the merge or my queue commits.
- Root cause (see MGLRenderer+Compat.m compatPipelineFastPath comment): the compat
  pipeline fast path keyed PSOs by compatMSLKeyCached = hash(MSL string POINTER + first
  192 + last 128 bytes). The fp64-lowered per-operator programs differ ONLY in the middle
  (one mgl_f*64 helper call inside a ~200 KB common prelude) and test_fp64's
  create->draw->delete loop recycles both the Program and the MSL string addresses, so
  operator k+1 produced operator k's key: its draws rendered with the previous program's
  PSO (abs -> -ua values, roundeven -> ceil values — the "previous" answers are exactly
  what the readback returned). Why r55-r58 triggered it: the allocation pattern (program
  link/free order through the new link-cache/relink work) now recycles the msl_str
  address; the key design was unsound all along (head/tail sampling).
  Fix (1605445): key.a = fnid registry value of each raster stage's compiled MTLFunction
  (mglProgramFunctionId; extProgramCompileMSL registers fnidOf(msl, entry) — a hash of the
  FULL MSL text — at link time). Missing/unregistered stage -> decline the fast path
  (one descriptor build per program lifecycle; no cached-PSO reuse). After: 22/22 clean.
  NOTE for geom agent: b5bd61d (r59 link-cache program-name fix) does NOT cover this.
- Gate: suite 112 files 9432/0, test_geom 10x 603/0, test_fp64fast 3x 298/0,
  flash_check PASS at 10:49 (median 114.1); 12:0x-12:2x reruns INVALID (occluded window,
  31/59/89 covered frames — other agents' GUI work; retry when free).
- stub_scanner EMPTY/NOOP/ASSERT: 0 for compat + spirv_vote files.

### 1. shader_group_vote (5/5 root-caused; CTS run pending display/GPU quiet)
- symptom: "Shader compilation failed" — glslang compiles allInvocations/anyInvocation/
  allInvocationsEqual (GLSL 4.60 core, unsuffixed; the CTS uses #version 460 from
  getContextTypeGLSLVersion) to the LEGACY SPV_KHR_subgroup_vote opcodes 4428-4430
  (OpSubgroupAll/Any/AllEqualKHR). SPIRV-Cross's MSL backend has no cases for them and
  falls through to the GLSL backend, which emits allInvocationsARB()/anyInvocationARB()/
  allInvocationsEqualARB() INTO THE MSL — identifiers that do not exist in Metal.
- fix: MGL/src/spirv_vote.c mgl_spirv_vote_khr_rewrite() (new pass, called from
  program.c right after spirv_prepare_for_msl): rewrites the three opcodes to core
  OpGroupNonUniformAll/Any/AllEqual with an ExecutionScope of Subgroup (synthesizes
  %uint 3 — or an OpTypeInt too if absent — before the first OpFunction, skipping the
  glslang-interleaved global OpVariables; rewrites capability SubgroupVoteKHR ->
  GroupNonUniformVote). SPIRV-Cross MSL then emits simd_all()/simd_any()/
  spvSubgroupAllEqual() — real Metal 2 votes. Per ARB_shader_group_vote the SIMD
  invocation group is implementation-defined (Metal's 32-lane simdgroup is legal and
  satisfies the only guarantees: anyInvocation true when <value> true, allInvocations
  false when <value> false, one answer per group). Workgroup-wide reduction would need
  injected threadgroup memory; documented as an emulation limit.
  Two bugs hit while writing it: new_bound was latched before the bound++ bumps (header
  advertised a bound that excluded the synthesized id -> SPIRV-Cross wrote past its ids
  vector -> SEGV in spvc parse), and the constant was inserted before glslang's
  interleaved globals referencing a later-defined type.
- tests/headless/test_vote.c: compile/link/dispatch/pixel checks for all three builtins
  (460 core unsuffixed), incl. anyInvocation(false)=black, allInvocations(true)=white.

### 2. gl_spirv (3 of 5 root-caused+fixed; state_queries + capabilities pending CTS verify)
- spirv_glsl_to_spirv_builtin_functions + spirv_validation_builtin_variable_decorations:
  link of SPIR-V-binary compute programs failed "no local work group size was specified"
  although the module has OpExecutionMode LocalSize — the scan in program.c compared the
  PACKED first word (wc<<16|op) to the bare opcode and read the mode literal at w+3
  instead of w+2; it never fired. Fixed to walk instructions by word count.
- spirv_modules_error_verification: glCompileShader on a shader with SPIR_V_BINARY_ARB
  TRUE must generate GL_INVALID_OPERATION (ARB_gl_spirv 7.1) — shaders.c now sets the
  error (info log was already there). NOTE glShaderSource on such a shader is LEGAL and
  clears SPIR_V_BINARY_ARB (spec 7.1 ShaderSource addition) — do not copy my first
  (wrong) test variant.
- spirv_validation_capabilities: needs 10 active image uniforms in ONE compute stage to
  link; MGL_MAX_IMAGE_UNITS was 8. Raised to 16 (Metal texture argument tables hold 128
  entries on Apple silicon; image_units[] is sized TEXTURE_UNITS=128). Shared-file edit:
  tests/headless/test_tess_images.c tess_image_uniforms asserted the old value 8 —
  updated to 16 (TESS AGENT: your branch will conflict there; the value follows
  GL_MAX_IMAGE_UNITS which the gl_spirv capabilities CTS case requires >= 10).
- spirv_modules_state_queries (NOT yet fixed): TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH
  must be 1 for programs linked from name-less SPIR-V ("no name reflection information
  available -> one"); MGL returns 0 because xfb_post_link only builds the shader-declared
  xfb layout when vs->src (GLSL) exists — SPIR-V-binary shaders have no source.
- tests/headless/test_spirvbin2.c + spv_builtinfn.h: embeds the CTS builtin-functions
  module (assembled from the CTS qpa disassembly with spirv-as --target-env opengl4.5):
  binary compile status, LocalSize link, dispatch, and the CompileShader error path.

## session 22 (Oct 27, 05:53-...) — s22: conformance queue part 3 (fails_compat2, 13 cases)

Task (supervisor): commit the ~280-line uncommitted s21 tail (texdata_formats.c/.h, textures.c,
test_buffers.c, 2 deleted debug tests) after building and running the affected CTS groups; then
drive the 13-case queue at ~/MGL-cts/mgl-r28-sweep/fails_compat2.txt to 0 Fail. Rules: -j4, one
glcts batch at a time, glmark2 under /tmp/mgl_glmark2.lock.

### 1. the s21 tail was corrupting the heap — found with Guard Malloc
Build + full suite were green (9233/0), but `direct_state_access.textures_compressed_subimage`
crashed nondeterministically (SIGSEGV / Trace-BPT in malloc, sometimes Pass under lldb).
Guard Malloc (`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib` through tests/cts_one.sh)
pinned it on the FIRST faulting access (crash report: fetchOnlineRGTC <- fetchCompressed <-
mglGetCompressedTexImage): the RG8 de-interleave plane was allocated `row*depth = 2*w*depth`
bytes but indexed `w*h*depth` — a 2x underallocation for every h>2 texture; plain 2D CTS levels
(h=4..16) scribbled the heap and a later malloc trapped. Fixed (img/2*w*h*depth-byte allocation;
also bsize now uses the requested d, not the level depth, on both copy-out paths). The 20-min
supervisor note about libgmalloc never returning: it had in fact already produced the 06:44 crash
report used here; full CTS runs under libgmalloc stay impractical (~100x slowdown), so the fixed
paths were re-validated with MallocCheckHeapEach=50 + MallocScribble on the headless repros.

### 2. what the s21 tail implements (3 commits)
* **9fa567b copy_image 1D-array semantics**: GL addressing for GL_TEXTURE_1D_ARRAY puts the
  1-texel-high image row in Y and the LAYER in Z; MGL's level stores the layer count in height,
  so mglCopyImageSubData's bounds check and region swap y/z + height/depth for 1D arrays
  (srcY+srcHeight<=1 else GL_INVALID_VALUE, layer count checked via the height axis).
  KHR-GL46.copy_image.exceeding_boundaries Pass; test_copy1da.c (11 checks: valid copy lands,
  y>0 / z>=layers / x overrun rejected).
* **30178b4 RGTC online compression** (uncompressed data uploaded to a compressed
  internalformat, GL 4.6 8.5.3): new tdDecodeRGTC CPU decoder (BC4/BC5, unorm+snorm); texImage
  accepts the request for the RGTC family, the level stores the DECODED R8/RG8 (Metal has no
  BC4/BC5-on-GPU compression path for this), glCompressedTexSubImage into such a level decodes
  in place and keeps the original blocks, glGetCompressedTexImage re-encodes with the same
  deterministic tdEncodeRGTC (upload->download round trip byte-exact), GL_TEXTURE_COMPRESSED
  reports TRUE and GL_TEXTURE_COMPRESSED_IMAGE_SIZE the block size. Families with no CPU
  encoder (BPTC, ETC2/EAC, generic GL_COMPRESSED_RGB) now answer GL_INVALID_OPERATION — a
  legal "no online compression" response the CTS handles by uploading its own precompressed
  blocks — instead of storing a decoded level that claimed GL_TEXTURE_COMPRESSED=TRUE while
  glGetCompressedTexImage failed. KHR-GL46.direct_state_access.textures_compressed_subimage
  Pass (3/3 runs); test_texonline.c (20 checks).
* **1f74da1 test expectations**: glBufferSubData on a range DISJOINT from a persistent mapping
  is legal per ARB_buffer_storage (only overlapping ranges raise GL_INVALID_OPERATION — the
  implementation from 7478e0a was right, the test asserted the old blanket rejection); dropped
  the scratch debug tests dbg_bordemu.c / test_intsampprobe.c.

### 3. queue + gate results (r28 sweep, this build)
* buffer_storage 9/9 Pass (incl. map_persistent_buffer_sub_data, map_persistent_read_pixels)
* copy_image 13/13 Pass (incl. exceeding_boundaries, incompatible_formats_compression)
* direct_state_access 371/371 Pass (incl. renderbuffers_storage, renderbuffers_storage_multisample,
  textures_compressed_subimage) — the whole group, not just the 3 queued cases
* texture_border_clamp 32/32 Pass (6 queued R32I/R32UI cases were already fixed by s20's border
  emulation; re-verified on this build)
* gate: full headless suite 9233 checks 0 FAIL; test_geom 10x 603/0; test_fp64fast 3x 298/0;
  flash_check PASS (n=120 median 115.9 max 239.6, all 16 over-band frames foreign-window).
* stub_scanner EMPTY/NOOP/ASSERT: clean on textures.c + texdata_formats.c. Build -j4 clean.

### shared-file edits this session
* none beyond my own files (textures.c, texdata_formats.c/.h are the compat/texture-domain
  files this worktree owns for texture-format paths; no renderer or context-header changes).


Task (supervisor): review+commit the uncommitted s19 tail (msl_reserved.c/.h, program.c, mgl_test.h,
report), check the test_compat_basevertex 0%-CPU stall, then drive the 21-case CTS queue
(fails_compat.txt) to 0 Fail.  Rules: -j4, merge gl46-port before READY.

### 1. the uncommitted s19 tail — verified and committed
* `msl_reserved.c/.h` `mgl_msl_fix_interpolate_at_offset` + the `program.c` hook (12 lines):
  GLSL's interpolateAtOffset takes the offset from the pixel **center**, Metal's
  `interpolate_at_offset` from the pixel's **upper-left corner**; SPIRV-Cross emits a
  `+ 0.4375` bias (0.5 - 1/16, its closest stand-in for the center), a systematic 1/16-pixel
  error that the CTS check (offset = gl_SamplePosition - 0.5, tol 1/255) rejects. The rewrite
  bumps the bias to the exact `+ 0.5` in every call tail. test_compat_ipolmap probes 5 offsets
  and matches GL's exact values, test_compat_interpatoffset is the CTS-shaped regression
  (sample-qualified varying == interpolateAtOffset(gl_SamplePosition-0.5)) at 1/2/4 samples.
* `mgl_test.h` `mgl_make_program3` (vs+gs+fs helper), tests test_compat_ipolconv/ipolmap promoted
  from print-probes to assertions (ipolmap now asserts the exact GL value per offset).

### 2. the test_compat_basevertex "stall": three stacked bugs
Symptom: the test sat at 0% CPU until killed. `sample` showed the main thread in
`waitUntilCompleted` from `c_mtl_flush` (glFinish) — a command buffer that can never complete.
Chain, each layer fixed:
1. **test bug** (mine, from s19): the FS consumed `f_color` while the VS wrote `v_color` (no match
   by name/location), and the "triangle" positions were 2D pairs read as a vec4/stride-16 attribute
   (degenerate). GL makes the unwritten FS input *undefined* (the link is legal), so MGL's PSO
   creation fails for it.
2. **MGL wedge** (MGLRenderer+Draw.m, `extBufWaitSerial`): after a PSO failure, MGLRenderer.m's
   `resetMetalState` recreates the command queue and *abandons the current command buffer
   uncommitted*; glFinish then waited forever on it. Fix: if the wait target is still uncommitted
   after the flush attempt, commit it (error-recovery already ended its encoders); if even that
   fails, advance `completed` to its serial instead of blocking. Regression case
   `unwritten_fs_input_no_hang` in test_compat_basevertex (was a 100% repro hang, now finishes and
   skips the draw).
3. **real GL bug** (state_ext.c `rs_program_has_flat`): the provoking-vertex index rotation probe
   looked for `[[flat]]`, but SPIRV-Cross writes located flat inputs as `[[user(locn0), flat]]` —
   so *no* flat-varying program ever rotated its indices and flat inputs took Metal's FIRST vertex
   instead of GL's LAST. Probe now matches `[[flat`, ` flat]]`, `flat, `. With that,
   glDrawElementsBaseVertex + `flat out vec4` = provoking id exactly (73/109/219 r-channel checks).
   This also silently fixed gl_VertexID-vs-flat ordering for every existing flat program.

### 3. the CTS case: KHR-GL46.draw_elements_base_vertex_tests.AEP_shader_stages
Reproduced ("Program linking failed"): the AEP tessellation combos link a TES-stage program whose
VS reads gl_InstanceID. `rewrite_vs_kernel` (tessellation.c) rewrote the vertex kernel ids with
`as_uint(...)` — an OpenCL/HLSL builtin, MSL spells it `as_type<uint>` and neither is needed: the
operands are already uint. Any tess VS touching gl_InstanceID/gl_VertexID failed MSL compilation
and the link. Fixed to plain expressions (tess domain file — surgical, flagged in the report).
Result: **draw_elements_base_vertex_tests 12 cases: 11 Pass / 1 NotSupported (ES-only) / 0 Fail.**

### 4. found while verifying (geometry-emulation gaps — NOT fixed here, geom domain)
Probe /tmp/bv7 (GS pass-through triangle, 64x64 FBO), all through the GS path:
* gl_VertexID is 0 for every vertex the capture pass re-runs (colors constant at id 0) — the CTS
  AEP GS combos still pass because they only compare base-vs-reference textures, which come out
  equal when both use id 0.
* CPU assembly drops draws with a nonzero vstart: glDrawElementsBaseVertex(ushort, bv=4) renders
  nothing (vstart=4 vcount=3 traced, no raster output).
* GPU assembly (GL_UNSIGNED_INT element buffer) renders nothing for a 3-index draw (vcount=0 in the
  trace); UBYTE/SHORT fall back to CPU assembly (U8 has no Metal index type; U16 untested by CTS).
* the raster pass does not apply GL's last-vertex provoking rule to the GS's *outputs* (flat inputs
  take Metal's first emitted vertex).
Documented in test_compat_basevertex.c; the VS+GS+FS link is asserted there, the GS draws are not.

### 5. gate (s20)
* full headless suite: **0 FAIL** (every test "== N checks, 0 failures").
* test_geom 10x: 603/0 every run.  test_fp64fast 3x: 298/0 every run.
* flash_check: 3x PASS (median 113.9, max 119.3 with the lock held). Hardened first
  (a) the whole check re-execs under `lockf -k /tmp/mgl_glmark2.lock` — two glmark2 runs at
  once were skewing both agents' numbers (supervisor 01:18), (b) a >med+15 frame now counts
  as a flash only if it correlates with the scene it flashes over; flat uncorrelated bright
  frames are other agents' windows (measured: corr 0.007 for the 245.4 frames that failed
  the first runs), (c) the first 12 captures are skipped (scene fade-in legitimately rides
  the band top, 131 -> 115). Earlier FAILs were all foreign-window occlusion; under the lock
  the band is tight (min 107.7 max 119.3).
* CTS re-runs (current build, agents/conformance/run.py, nice, one batch at a time):
  - draw_elements_base_vertex_tests: 12 cases = 11 Pass + 1 NotSupported (ES-only) — 0 Fail
  - sample_shading: 13 cases = 13 Pass
  - shader_multisample_interpolation: 109 cases = 109 Pass
  fails_compat.txt queue (21 cases) therefore at 0 Fail.
* stub_scanner EMPTY/NOOP/ASSERT on compat*.c: clean.  gl46-port (6ee46f0) merged (already
  contained).  Build -j4, no new warnings.

### shared-file edits this session (call-outs)
* MGL/src/program.c (+12): call mgl_msl_fix_interpolate_at_offset on fragment MSL (s19 tail).
* MGL/src/msl_reserved.c/.h (+83/+4): the new interpolateAtOffset bias rewrite (s19 tail).
* MGL/src/MGLRenderer+Draw.m (~+12, extBufWaitSerial): never wait on an uncommittable
  command buffer (PSO-failure recovery abandons it) — was a hard 0%-CPU hang.
* MGL/src/tessellation.c (4 lines, rewrite_vs_kernel): as_uint() is not MSL; plain uint
  expressions.  Any tess VS reading gl_InstanceID/gl_VertexID failed to link.
* ab/flash_check.sh: glmark2 lock + flash discrimination (above).


## session 19 (Oct 26, 17:10-...) — per-draw CPU: the respec walk

Task: cut [ideas] per-draw CPU (targets processGLState 0.58 ms, draw 1.20 ms at load ~5).  Merged
gl46-port 9442011 first (s18 was merged there).

### What the [ideas] walks actually are

Recovered the real [ideas] bucket breakdown from the s16 A/B logs (ab16_*.log): 89 of 353 draws
re-specify vertex arrays; per walk: tail-pipeBind+PreDraw 0.14-0.22 ms/frame (biggest), pipeline-
block 0.055-0.082, mapBuffersToMTL 0.060-0.092, plus gate 0.15-0.26 and encode 0.11-0.19 across
all draws.  glmark2's SceneIdeas/SGILogo re-specify **identical layouts against a different VBO per
mesh** (logo.cc: bufferObjects_[0] + per-mesh glVertexAttribPointer with per-piece offsets) — the
gate misses at check 13 (vao-fmt/binding, the Buffer identity is part of the memcmp), so every mesh
pays a full processGLState walk although the only real work is re-pointing the encoder's
vertex-buffer bindings.  (pipeHit=89/frame at one slot in [ideas]: the PSO key never saw the
buffer identity — it hashes `(buffer != NULL)`, and the offsets there are constant.)

Built two headless benches to iterate without glmark2 (no window, no present):
- test_compat_ideaspec: the exact [ideas] shape on a compat context — one VAO, per-mesh VBO respec
  + uniform update, 133 walks + 267 skips/frame, pixel check + WALL ms/frame.  Gotcha found on the
  way: small (<ring threshold) buffers and element buffers stream through the ring and inject ~25
  µs/draw GPU-coupled stalls — the bench's VBOs/EBO are padded to 32 KB so the Metal path matches
  [ideas] (real MTL buffers; the Metal call itself measures 1.6-2.4 µs).
- test_compat_rebind: 9 adversarial respec patterns, pixel-exact, verified identical on the
  pre-change build (patterns that must full-walk: stride, divisor, alignment-class, misaligned
  offset with data really at byte 13; patterns that may rebind: buffer swap, offset move, split
  bindings, alternating L/R; plus uniform-updates-on-the-same-draw).

### Changes (compat files + 2 surgical edits)

1. **Compact VAO layout key** (a72280c).  compatVAOHashCached packed all 16*3 attrib + 16*3
   binding words (776 B stack pack + memcmp per walk) and hashed the raw byte offset, so
   per-mesh-offset shapes fanned one layout into one PSO cache entry per mesh.  Now it packs the
   ENABLED attribs' formats + the bindings they reference — exactly the set vattribMapVertex-
   Buffers maps (mask==0 or bufferless bindings never reach the encoder), so the descriptor and the
   shared plan's binding_index[] depend on nothing outside it — and a binding's offset enters as
   its 4-byte alignment class (the aligned-vs-shadow decision); the base offset reaches the encoder
   per draw.  Disabled attribs feed fixed-format const streams (double value bits), so their fmt
   words are not key material.  Snapshot buffer is fixed-capacity now (the packed length varies;
   the first version malloc'd for the first pack size and a later larger pack overran it — caught
   by test_vattrib crashing 1-in-3 runs in malloc, fixed before commit).  Measured on the bench:
   rstate+vao part of the pipeline block 1157 -> 690 ns/walk (the pack) and one PSO for all
   per-mesh-offset layouts.

2. **MGL_DF_REBIND: buffer-identity-only respec tier** (974fb3a).  compatDrawGate classifies a
   failed check-13 content compare (compatRebindClassify: layout words equal, alignment class
   equal, buffer presence equal, changed buffers MTL-backed, no SHADOW fetch, plan valid) and
   returns MGL_DF_REBIND instead of MGL_DF_FULL.  compatDrawFastRebind then: re-points the vertex
   map list's attribute suffix **in place** (the entry set/order, mtl indices and the SSBO pad
   position are unchanged — no bindmap-suffix rebuild), flushes dirty listed buffers
   (updateDirtyBaseBufferList), rebinds through extBufBindList, clears the redundant dirty bits
   and re-latches only the changed groups (bindings, vabuf ids, map entries).  Skipped as provable
   no-ops: the PSO path (the compact key ignores buffer identity and the gate proved program+pipe
   identity), updateEncoder, the PreDraw hook fan-out (rstate/texext gate-proven unchanged; the
   vattrib hook's binds are the bindList rebinds), geomBindSSBOSizes.  Everything the gate checks
   still runs — the tier only replaces the *processing* of gate-proven-equal state.
   Bench: processGLState 0.99 -> 0.24-0.33 ms/frame, the respec draw's compat CPU ~4x cheaper.

3. **extBufBindList skips already-bound slots in C** (6a3f64f): one call to
   compatVbufAlreadyBound (exact mirror of compatBindVertexBuffer's shadow test: same encoder
   generation, same buffer object, same offset at the same index) replaces the storage/ring
   dispatch + ObjC bind chain for unchanged entries.  Rebind path 1.85 -> 1.5 µs.

Shared-file edits: MGLRenderer+Draw.m +23 lines (the MGL_DF_REBIND branch in extBufBeginDraw, the
C pre-check in extBufBindList), MGLRenderer+VertexAttrib.m 0 (the temporary probes were removed),
MGL/src/compat.c -6 (debug print removed).  New state lives in MGLCompatDrawFastSnap (mine).

### Results (numbers)

(pending the quiet-machine glmark2 A/B — see ab/run_ideas_cpu_s19.sh; headless bench numbers above.)

## session 18 (Oct 26, 15:35-17:00 EDT) — s18 task 1: single-pass present copy (be2a894)

Supervisor verdict from the quiet A/B: no ~16% code regression (159d900 itself ~2960 today); the
only real hit is jellyfish -3% matching e4142bb's per-present compute dispatch.  Tasks: (1) drop
that dispatch without losing the caustics/flash fix, (2) measure ideas per-draw CPU, (3) gate.

- e4142bb present order was [compatCSDrain + compatForceOpaque kernel over the source (read_write
  over the whole default framebuffer) + compatCSDrain + blit copy] per present = 2 encoders,
  2 drains, 2 full-surface GPU passes.  Replaced by compatPresentCopyOpaque:to: (new, in
  MGLRenderer+Compat.m): a PSO cached per dst pixel format, one render pass, fullscreen triangle
  from vertex_id (no vertex buffer), fragment does src.read(uint2(pos.xy)) — no sampler state —
  and writes float4(rgb, 1.0).  Requires same size + same format + float-kind source (BGRA8/RGBA8
  ±sRGB, RGBA16Float); everything else falls back to fbShaderBlitFrom + post-blit kernel (rare
  resize-race path, unchanged), and the env-gated direct path keeps its in-place kernel.
  Bonus: the source's alpha channel is no longer rewritten, so glReadPixels of the default
  framebuffer after a present returns GL's blended alpha again.
- Correctness: MGL_PRESENT_DUMP=25 on jellyfish — 81 presented-surface dumps, alpha=255 in every
  pixel, bell (79,100,161) with blue spread 231 (caustic web).  ab/flash_check.sh PASS (hardened:
  re-frontmost every 10 captures + INVALID guard when >25% frames are ~240 = window occluded by
  another agent — first run FAILed on exactly that, frames 70+ constant 240.6, visible frames
  108-119 clean).
- Perf (quiet, interleaved, ab/run_quiet_ab_s18.sh; ab/prev=a036e29, ab/gl46port=5178835 built
  with git archive + external symlink): jellyfish new 24100/24672 vs gl46port 22834/23535 (+5.2%);
  vs my own old tip a036e29 4176-4745 vs 3916-4298 (+6-10%) in a visible window.  swap bucket CPU
  0.0326-0.0342 vs 0.0388-0.0392 ms/frame.  ideas neutral as expected (draw-bound).
- Task 2 (ideas per-draw CPU, ab/run_ideas_cpu_s18.sh + a 3-round bucket diff): tip vs gl46-port —
  draw 1.204 vs 1.473 (-18.3%), processGLState 0.577 vs 0.709 (-18.6%), encode-draw 0.152 vs
  0.177 (-14%), newRenderEncoder 0.098 vs 0.126 (-22%), updateEncoder 0.037 vs 0.040, vao-buflists
  0.009 vs 0.011.  /usr/bin/time user CPU per 10 s scene was NOT usable (load spikes 9-16 from
  parallel agents swamp user+sys); the MGL_PROFILE buckets are the load-tolerant metric.  All four
  s17 per-draw commits keep (CPU measurably down ~18% on the draw path).
- Gate: suite 96 files 0 FAIL, test_geom 10x 603/0, test_fp64fast 3x 298/0, flash_check PASS,
  window captures: jellyfish (82,111,193) blue-std 66, ideas (68,68,66), 0 magenta px, no logo
  bleed.  nsgl_smoke: quadrants 4/4, readback byte-identical, present throughput 9418-12592 Hz;
  its GL_VERSION check FAILs because the honest-strings change (s14) prints "4.6 Metal - 90.5"
  while the test greps for "compat" — pre-existing, not touched (compat.c:616 / get.c:1158).
- MGLRenderer+Framebuffer.m: one comment updated (the ShaderWrite flag's reason changed from
  "force-opaque kernel on the source" to "direct-present kernel"; flags unchanged).

READY be2a894 (supersedes e4142bb; s18 task 1+2 done): the per-present compute pass is GONE —
compatPresentCopyOpaque:to: is ONE cached fullscreen-triangle render pass that copies the frame
and writes alpha=1 straight into the pool IOSurface (same e4142bb flash property — the surface's
only write per frame is already opaque, so CA can never catch a half-alpha frame — and the
source's GL alpha is no longer clobbered, default-fb readback keeps blended alpha).  Quiet A/B
(load<1.5, interleaved, 2 rounds): jellyfish 24100/24672 vs gl46-port 5178835 22834/23535 (+5.2%);
ideas 2471/2456 vs 2484/2427 (neutral, not present-bound).  MGL_PROFILE=1 swap bucket 0.033 vs
0.039 ms/frame (-15% CPU in the present path).  Ideas per-draw CPU (s18 task 2, tip vs gl46-port):
draw 1.204 vs 1.473 ms/frame (-18%), processGLState 0.577 vs 0.709 (-19%), encode-draw -14%,
newRenderEncoder -22%, updateEncoder -8%, vao-buflists -18% — the four s17 per-draw commits STAY
(CPU measurably down; FPS is GPU/present-bound near 2900 in a visible window).  Gate: suite 96
files 0 FAIL, test_geom 10x 603/0, test_fp64fast 3x 298/0, ab/flash_check.sh PASS (120 captures
median 113.6 max 119.4, 0 flash frames), MGL_PRESENT_DUMP 81 frames alpha=255 in every pixel,
jellyfish window mean (82,111,193) blue-std 66 (caustic web like native), ideas (68,68,66) neutral
gray, 0 magenta-ish px both scenes, no logo bleed.  NOTE: my A/B windows run occluded (~24k FPS
scale vs the supervisor's ~12.5k visible-window scale) — the interleaved same-condition RATIO is
the comparable number, not absolute FPS.
READY e4142bb (supersedes 30f7bb8/fa2a242; adds the [jellyfish] present-flash fix the user
reported at 07:00 + everything from 30f7bb8: merge d31c9ba, ONE reserved-word implementation,
[ideas] layout-keyed PSO cache, DIRTY_FBO gate rule, drawpix sRGB/RGTC; suite 0 FAIL (95
files), test_geom 10x 603/0, test_fp64fast 3x 298/0, jellyfish 4498 FPS caustics intact no
flashes (ab/flash_check.sh PASS, old order FAILs it), gl41-geometry 2008 FPS no magenta, ideas
392-422 FPS under load 5.9-7.7 (quiet-machine A/B 485->578; user idle-machine 2887 on fa2a242 —
the flash fix touches only the present path, not the draw path)).
NOT-PASSING (pre-existing, not compat): KHR-GL46.texture_swizzle.smoke HANGS before any
subcase — sampled the stuck glcts: glslang ProcessDeferred/initializeExtensionBehavior
(external/glslang internal), also NoResult in the r22 sweep.  The compat part of geom's finding
is in (fast path declines DIRTY_FBO, 30f7bb8); the hang itself is texture/glslang domain.
BENCH-REQUEST e4142bb: user visual + same-hour A/B as before; watch [jellyfish] for flashes
(fixed) and confirm ideas/jellyfish/gl41 FPS unchanged vs the fa2a242 run (present-path only).
BENCH-REQUEST 66f0f1c: same bench as mglR16 (full default list, same-hour A/B).  Diff vs
c6b8124 = tess merge + reserved-word unification + the [ideas] PSO-layout-key fast path
(layout-keyed pipeline cache + plan-restore skip; expect [ideas] up, everything else neutral
— jellyfish/ideas/gl41-geometry visuals verified unchanged this session).
READY c6b8124 (on 852eee2 + 638e1ed; jellyfish caustics FIXED — see session 15; suite 0 FAIL
(85 files), test_geom 603/0, test_fp64fast 298/0; glmark2 [jellyfish] FPS 4580 (was 3987 at
852eee2), bell now dark rich blue with caustic web like native: window-capture bell mean
(79-92, 90-121, 128-195) grad 5.2-6.2 vs native (81-89, 93-114, 132-180) grad 2.6-8.1; before
the fix MGL was pale cream (193-230, 176-213, 154-186)).
BENCH-REQUEST c6b8124: same bench as mglR16 (full default list, same-hour A/B).  The present
path gained one tiny compute dispatch per present (force-alpha-opaque kernel, 800x600, ~0.03 ms,
~12-15 presents/s) — expected neutral; verify jellyfish + ideas visuals.

## session 14 (Oct 25, 17:30-20:39 EDT) — CAMetalLayer stacking/opaque, honest GL strings (from the supervisor's summary; commits f5c3256, 852eee2 + uncommitted GLSL scanner work)
- CAMetalLayer stacking fix (ideas logo no longer drawn into the jellyfish), hash-table delete
  leak fix, compat-profile GL_EXTENSIONS, silent default build, xctx tests, test_xctx9-12 drafts.
- CAMetalLayer.opaque=YES (washed colours fixed then), honest GL strings (Vendor Apple, Renderer
  MTLDevice.name, Version '4.6 Metal - 90.5', GLSL 4.60, env overrides), xctx3-8, test_gsindirect*.
- Left uncommitted: the GLSL §3.7 reserved-word scanner (false-positived on jellyfish.vert) —
  finished and committed as 638e1ed in session 15.

## session 16 (Oct 26, 01:15-04:15 EDT) — merge gl46-port (tess r23/r24); ONE reserved-word impl; [ideas] per-draw CPU -20%
- `git merge gl46-port` (9485327: tess r23/r24) auto-resolved cleanly (e62b50b) but left BOTH
  reserved-word scanners live: compat's `scan_reserved_names` (638e1ed, shaders.c) and tess's
  `mgl_reject_reserved_declarators` (aa32719, cts_glsl_compat.c, called on the raw source in
  mglCompileShader AND from mgl_cts_glsl_compat_limited).
- Unification decision, driven by measurement not preference: new `tests/headless/test_reserved.c`
  battery (1463 compiles — 38 reserved words + 95 keywords × 11 declaration slots + positive
  controls).  With compat's scanner disabled, glslang + tess's reject everything EXCEPT
  `struct invariant {...}` (struct TYPE name).  So tess's is the survivor (targeted, no heuristic
  lexer; tess's own report asks for exactly that if only one may live) and the struct-name case
  was added to it (+10 lines in cts_glsl_compat.c; MGL's compat preamble struct names are all
  mgl_*, unaffected).  compat's 270-line scanner deleted from shaders.c.
- test_reserved.c also pins the integer-mix version rule (tess r24's headless test was not on
  the branch): `mix(ivec2(1), ivec2(2), bvec2(true))` rejected below #version 450, legal at 460,
  float mix always legal; plus legal-keyword positives (constructor after an identifier operand —
  the old jellyfish.vert false positive — flow control, qualifier positions, correct subroutine
  declarations, #version 120 attribute/varying on a compat context).
- Known gap (documented in the test, pre-existing): a legacy identifier named sample/patch/
  centroid breaks on the compat path because the translator re-emits the body at #version 460
  where those words became keywords.  glmark2 doesn't use them; fix would need symbol renaming.
- **[ideas] per-draw CPU (a6431f2, USER PRIORITY)**: the pipeline-cache key mixed the VAO
  pointer, the per-attrib buffer pointers and the element-array buffer into key.b.  [ideas]
  draws ~89 VAOs/frame with identical layouts but different VBOs → 89 PSO entries → every hit
  was a move-to-front memmove of up to ~44×~512 B slots + a 940 B re-hash/snapshot copy per
  walk.  Now `compatVAOHashCached` packs exactly the LAYOUT words the descriptor reads (fmt:
  size/type/normalized/integer/is_long/relativeoffset/binding; binding: buffer-bound/offset/
  stride/divisor) and validates the single-entry cache by memcmp of the packed words (an EXACT
  validator — two VAOs with identical layouts validate each other, collapsing 89 draws to ONE
  PSO entry; pipeHit=89/frame at slot 0).  Plus a plan-restore skip (plan_live_msl/plan_live_vao:
  the fetch/shadow/const analysis is a pure function of program content + layout; only another
  key's hit or the descriptor path rewrites it).  test_vattrib float_conversions caught the first
  cut that hashed the legacy attrib[] mirror instead of fmt[]/binding[] (integer/is_long/
  binding-offset were missing) — the full-struct pack fixed it.
- Also fixed in passing: `misswhy[24]` was written at indices 24/25 (OOB counter writes, session
  13 bug); stale `compatDrawFast` declaration (session 12) removed.
- **A/B numbers**: draw bucket (MGL_PROFILE tail, identical instrumentation) 1.436 → 1.111
  ms/frame (-23%); median-of-dumps sequential A/B (base=f608578 built in /tmp, 4 interleaved
  rounds) draw 0.185 → 0.148 ms/frame (-20%); final-round FPS 485 → 578 (+19%).  Quiet-machine
  steady state (938 FPS): draw 0.534 ms/frame = 47% state walks (2.8 µs × 89; pipeline-block
  0.6 µs/walk after the fix, was ~2.0), 15% gate (0.23 µs/draw), 13% encode (0.2), 16% draw
  prologue (0.25 µs; execDrawRaw = 1 call/frame — the batch path appends without re-entering
  exec_draw_raw).  Remaining fat if someone continues: the 89 walks' PreDraw-hooks fan-out
  (0.74 µs/walk) and bindList (0.35 µs/call).
- Gate (all on the final tip 66f0f1c): suite **93 files 0 FAIL**; test_geom 10× 603/0;
  test_fp64fast 3× 298/0; KHR-GL46.CommonBugs.CommonBug_ReservedNames **Pass**; KHR-GL46.shaders.*
  (one glcts batch, tests/cts_group.sh, run on 07cd1f5 before the draw-path-only perf change)
  **2382 cases: 2370 Pass / 6 Fail (pre-existing uniform_block.random.*, identical set to
  r22/r24) / 6 NotSupported (integer_mix positives)**.
- glmark2 visuals after the perf change (ab/xctx_capture.sh): jellyfish bell (74,107,196) ≈
  before (74,106,194), caustic web, FPS 4606; ideas (74,73,71) ≈ before (80,79,78), no leak;
  gl41-geometry periodic grid unchanged.  No table-growth messages.
- Next: remaining r22 partial_fails are all OTHER domains (see the 16b re-sweep below); more
  [ideas] CPU if the bench still shows headroom (PreDraw-hooks fan-out 0.74 µs/walk, bindList
  0.35 µs/call).

## session 16b (Oct 26, 04:20-05:20 EDT) — r22 partial_fails re-sweep + glDrawPixels sRGB/RGTC
- Re-swept all 71 remaining entries of ~/MGL-cts/mgl-r22-sweep/partial_fails.txt on the current
  build (tests/cts_many.sh): ReservedNames + integer_mix.prototypes-negative already fixed by
  this session's merge.  Still failing, by owner: cull_distance.functional/coverage (GLSL
  gl_CullDistance compile/link — shader/tess), draw_indirect twoPass-tf ×2 (draw), fp64
  inverse_dmat3 (fp64), pipeline_statistics functional_geometry_shader (queries), atomic_
  counters advanced ×2 (program), image_load_store basic-allTargets (texture), uniform_block.
  random ×6 (pre-existing, tess), texture_gather incomplete ×2 + texture_swizzle (texture,
  swizzle NoResult at 75 s), vertex_attrib_64bit limits/vao (vattrib+fp64), viewport_array.
  scissor_clear (draw).  **packed_pixels ×16 (sRGB/RGTC via glTexImage2D/getTexImage/
  readback — TEXTURE domain; glctsPackedPixelsTests.cpp never calls glDrawPixels)** — left
  for the texture agent, noted here for the supervisor.
- Closed the matching compat gaps anyway (fa2a242): glDrawPixels writes its payload VERBATIM
  into an sRGB-encoded framebuffer (GL 4.6 §16.4 — the pixel path is a copy; sRGB encoding is
  for shader-written fragments).  Implementation: blit_upload_pixels uploads through an
  sRGB8_ALPHA8 view of the blit texture when FBO attachment 0 is sRGB (blit_target_is_srgb —
  sRGB-ness from the Texture internalformat via FBOAttachment.textarget): sampling decodes,
  the sRGB attachment re-encodes, decode∘encode is the 8-bit identity.  Plus GL_SRGB8 /
  GL_COMPRESSED_SRGB input-format grouping (3 components) and CPU block-decode of RGTC1/
  RGTC2 (signed + unsigned) drawpix payloads (BC4/BC5, code bits at block bytes 2..7 — the
  first cut read them at offset 8; test_compat_drawpix's pixel-exact RGTC case caught it).
- New test tests/headless/test_compat_drawpix.c (16 checks, 0 fail; suite now 94 files 0 FAIL).

## session 16d (Oct 26, 07:10-10:05 EDT) — [jellyfish] present-flash fix (user report 07:00)
- User on fa2a242 (idle machine): ideas 2887, jellyfish 12326, gl41-geometry 2934 -> 6048,
  logo gone, log silent — but "[jellyfish] has some flashes like from a flash on a camera".
- Root cause (found with a causal probe, MGL_PRESENT_PROBE_US): layer.contents on a raw
  IOSurface is composited ASYNCHRONOUSLY — CA does not wait for our swap command buffer.  The
  s15 present order (blit frame -> surface, THEN force alpha opaque on the surface) left a
  window in which one display refresh could composite the new frame while its alpha still held
  GL's blended value: the whole translucent frame over the white window backdrop = one bright
  flash.  Probe evidence: old order + 50 ms window -> luminance 107 -> 130.7 (flash state);
  new order with the SAME window -> steady 107.4-109.3.  Real world: old order 10 s video
  (1661 frames): 7 whole-frame spikes (median 98.8 -> 116); new order: 0 in 150 captures.
- Fix (e4142bb): force the copy SOURCE's alpha opaque BEFORE the blit — the presented surface
  transitions old -> opaque-new in one blit; the half-alpha state never reaches an IOSurface
  CA can see.  Two corrections found while verifying: (a) the private default framebuffer
  lacked MTLTextureUsageShaderWrite so the read_write kernel on it was UB — the surface showed
  magenta garbage (FF 00 FF pattern); fbEnsureDefaultFramebuffer adds ShaderWrite and
  compatSurfacePresent only pre-fixes sources that carry it (else the old post-blit kernel on
  the pool surface); (b) compatCSDrain before the kernel dispatch (no concurrent encoders).
- ab/flash_check.sh (new regression check): 150 window captures, per-frame mean must stay
  within median+15 WITH THE SCENE VISIBLE (flat ~240 median = info page/covered window =
  INVALID, a vacuous-pass guard added after it passed on the wrong content once).  Old order:
  FAIL (3 spikes 134/128/127); fixed order: PASS (median 114.3, max 119.1).
- Gate: suite 95 files 0 FAIL; test_geom 10x 603/0; test_fp64fast 3x 298/0; visuals:
  jellyfish bell (69,104,195) caustics intact, gl41-geometry (72,98,164), ideas dark-phase
  matches the old build at the same scene time; no magenta, no flashes.  FPS under load
  5.9-7.7: jellyfish 4498, gl41-geometry 2008, ideas 392-422.
- texture_swizzle.smoke: still hangs BEFORE any subcase verdict (sampled glcts: glslang
  ProcessDeferred / initializeExtensionBehavior — external dependency, also NoResult in r22).
  The compat part (fast path declines DIRTY_FBO so the descriptor path retargets the encoder
  to a newly attached FBO texture) is in since 30f7bb8; the hang is texture/glslang domain.
- Supervisor ownership note (08:42) acknowledged: compat owns MGLRenderer+Compat.m and the
  pipeline fast path.  The gate rule stays narrowed to DIRTY_FBO-always-declines
  (ALPHA/RENDER_STATE content is hashed fresh in the fast path; the full mask cost ideas 4
  descriptor builds/frame — FPS 328 vs 459-578, plus geom's magenta frames were this same
  ShaderWrite UB class, fixed here).

## session 16c (Oct 26, 05:50-06:40 EDT) — merge gl46-port d31c9ba + geom's fast-path gate rule
- `git merge gl46-port` (d31c9ba: tess r24 final — xfb capture in the draw path, cull_distance
  GLSL, msl_reserved table, fp64 matrix_inverse, get.c) auto-merged clean (a0825a5); the compat
  reserved-word struct-name case survived; test_reserved 21/0.
- Supervisor relayed geom e767e6a: the compat pipeline fast-path gate must not take DIRTY_FBO/
  ALPHA/RENDER_STATE triggers (a newly attached FBO texture is never bound → texture_swizzle.
  smoke wrong pixel).  My gate had the same hole (declined only on DIRTY_SHADER|DIRTY_DRAWABLE).
  Fixed 30f7bb8 — DIRTY_FBO always declines.  ALPHA/RENDER_STATE kept admissible (see the
  GATE-DEVIATION NOTE above for the reasoning + the measured cost of the full rule + the
  one-line re-adoption).
- KHR-GL46.texture_swizzle.smoke on this tree: HANGS (NoResult) — pre-existing, texture domain
  (same in the r22 sweep), so geom's exact repro could not be re-run here; the mask change is
  driven by the rule + the invisible-swap analysis, and test_compat_pipefbo.c pins the
  observable class (re-attach same-format T1→T2; yellow must land in T2, T1 stays green).
- Gate on 30f7bb8: suite **95 files 0 FAIL**; test_geom 10× 603/0; test_fp64fast 3× 298/0;
  CommonBug_ReservedNames Pass; glmark2 jellyfish 4404 FPS bell (73,106,193) caustics intact,
  ideas neutral no-leak (103,102,97 — animated phase; structure thumbnail unchanged),
  gl41-geometry periodic grid (74,99,160); [ideas] profile after the gate change: pipeHit=89/
  frame, 0 descriptor builds.

## session 15 (Oct 25, 20:45-23:30 EDT) — jellyfish caustics: presented-surface alpha

PRIORITY-1 bug (user-visible, blocked the merge): glmark2 [jellyfish] bell pale/flat, no caustic
web, through the NSOpenGL interposer.  Root cause found and fixed.

Investigation trail (tools committed under .agent/):
- per-draw texture binding verified live (MGL_BINDTRACE on a real run): unit 1 gets the rotating
  caustic textures (names cycle, each a distinct 256x256 MTLTexture), unit 0 gets jellyfish256.
- the linked MSL is correct (MGL_DUMP_MSL): uSampler1 = texture(0) sampled with the world-uv
  caustic term, uSampler = texture(1) with vTextureCoord.  No swap.
- headless repro with the real shaders/PNGs/blend on the compat path works (test_xctx13).
- MGL_PRESENT_DUMP (new, post-commit readback of the presented IOSurface): every presented frame
  HAS the caustic web — bell mean (82-87, 100-110, 143-170) local-contrast grad 8-10.  The render
  was never wrong.
- the WINDOW (screen) showed a washed-out bell (193-230, 176-213, 154-186) while the presented
  surfaces were rich, and the opaque gradient background matched the screen bit-exactly ->
  difference only where alpha < 255.  Alpha dump of the presented IOSurface: the jellyfish region
  has alpha 85..254 — GL leaves dst.a = a*a + (1-a) after SRC_ALPHA/ONE_MINUS_SRC_ALPHA blending.
  CoreAnimation alpha-composites layer.contents over the (white) window backdrop, so every
  translucent pixel washed out.  (GL window semantics discard the default framebuffer's alpha at
  present; Apple's own GL surfaces are opaque.)

Fix (MGLRenderer+Compat.m):
- mgl_force_opaque compute kernel (built once per pool from MSL source, cached in the pool):
  rewrites the presented IOSurface's alpha to 1.0 before the commit that shows it.  Dispatched in
  both present paths (direct render-into-slot and blit).  ~15 dispatches/s of 800x600 threads —
  glmark2 [jellyfish] FPS 4580 vs 3987 before (noise).
- pool.layer.opaque = YES (the presenting sublayer; _layer.opaque from 852eee2 only covered the
  never-presented CAMetalLayer itself).
- MGL_PRESENT_DUMP=<n> debug: dumps the presented surface (+ alpha channel) after the commit,
  /tmp/mgl_present_<k>.ppm + mgl_present_alpha_<k>.pgm — this is how the alpha bug was proven.
- MGL_BINDTRACE extended with the GL name and MTLTexture size (MGLRenderer.m, debug-only).

Also fixed here (blocked the s14 GLSL reserved-word work): the scanner draft rejected legal
constructor calls after an identifier operand (real jellyfish.vert 'uWorldInvTranspose * vec4(..)'
-> "'vec4' reserved word").  Now keywords are only rejected in a declared-name slot after a
type/storage token, always-reserved words anywhere, and the compat translator's output is marked
/*mgl-compat:<user version>*/ so only the user's body is scanned, and only for #version >= 400.
Committed as 638e1ed with tests xctx9-12; suite 0 FAIL, test_geom 603/0, test_fp64fast 298/0.

Tests: tests/headless/test_xctx13.c — full jellyfish pipeline on a compat context (real
jellyfish.vert/frag, real jellyfish256 RGBA + caustics RGB upload, 34 texture names past the
initial hash-table size, LINEAR + REPEAT set after upload like the scene, SRC_ALPHA blending):
sanity coverage, caustic contribution (web vs solid black must differ), caustic spread across the
quad, REPEAT wrap at u=10.0 -> LINEAR 2x2 average 72 (clamp would be 136).  15 checks 0 failures.

## session 13 (Oct 25, 12:30-16:00 EDT) — bindmap prefix cache + PSO cache C slots + plan restore

Machine mostly loaded (sweep/other agents); all numbers interleaved same-instant vs ab/base13
(= ec592db = session-12 code + 159d900), profile via MGL_PROFILE=1.
* mapBuffersToMTL (bucket 11, 89 walks/frame in ideas) re-enumerated the program's SPIR-V buffer
  resources per walk although the UBO/SSBO/atomic/dub PREFIX depends only on (program, link ->
  DIRTY_PROGRAM, buffer_base content, dub identity).  Now served from an exact cache
  (struct MGLCompatBindmapCache in MGLRenderer+Compat.m, state slot in mgl_compat_perf.h):
  validation = program ptr + !DIRTY_PROGRAM + vstage + 4x buffer_base memcmp + nullbuf + per-stage
  dub pointer revalidation; hit path memcpy-prefix + mgl_buffers_mark_gpu_use for SSBO/ACB entries
  + vattrib suffix re-runs.  ideas: 85% hit (82/f, misses = the 14 program switches/f).  Kill
  switch MGL_BINDMAP=0.
* PSO cache (compatPipelineFastPath/Store): NSMutableArray<CompatPipeBox> -> plain C slots
  (key a/b + CFBridgingRetained pipe), no ObjC on the hit path; MTF via memmove.
* REAL BUG found+fixed while instrumenting: on a PSO cache hit, s->plan kept the LAST descriptor
  build's fetch/shadow/const analysis, but a cache hit only proves the key — scenes alternating
  two cached layouts (ideas: 2-vs-3 component inputs; "plan diff at +196 (2->3)") drew each mesh
  with the other layout's plan (wrong const-mask/shadow-slot/dbl handling — same class as the
  heap-overflow crash fixed earlier in vattrib).  Each slot now stores the MGLVattribDrawPlan it
  was built with; the hit path memcmps the static part and restores the slot copy (this walk's
  binding_index preserved).  First attempt (re-run the generator on digest mismatch) refreshed
  ~every walk in ideas (oscillating digests, +0.08 ms/f) — the memcpy restore costs ~0.1 us.
* Profiling buckets renamed/added (MGLRenderer.m names[]): 15 texture-bind-walk, 20
  pipeline-block, 21 tail-pipeBind+PreDraw, 23 PreDraw-hooks; ssboSizes (geomSSBOSizesNeeded)
  was already cached per program — supervisor item 3 needs nothing.
* Draw merging (item 5): still not applicable — the run histogram shows zero consecutive draws
  with equal (PSO, count, first, instance, offset, stride, divisor); Apple's 353->266 must come
  from elsewhere (their text glyphs / instanced paths).
* ideas decomposition after this session (quiet-ish hour): draw 0.59 ms/f = gate 0.10 (353),
  processGLState 0.27 (89; pipeline-block 0.09, tail 0.10 incl PreDraw-hooks 0.08, mapBuffers
  0.04), bindList 0.06 (134), encode-draw 0.08 (352) — the tail is now dominated by real Metal
  binds + the vattrib hook's ring write of the changed dub (per-object matrices).
* Shared-file edits: MGLRenderer.m (mapBuffersToMTL 4-line hook + 3 prof buckets + names[]),
  MGLRenderer+VertexAttrib.m none this session (plan fix is in my file), glm_context.h none.

BENCH-RESULT 28386d2 (head 15f5af0) 12:15 EDT Oct 25: native13 6492; mglR16 13629 / mglR16b 13684 vs same-hour baseline 5e02260 (mglBase5e) 13711 = -0.4% noise (machine ~5% slower than this morning). ideas 3404/3427 vs baseline 2804 (+22%) vs native 3576 (96%). Not pinned. MERGED. table_R16.txt. (native12/mglR15 at 10:20 = invalid, system degraded.)

READY 28386d2 (gl46-port merged as f5607b5 + 3 commits; suite 496/0, test_geom 603/0,
run twice + test_atomic_counters 3x; default path only — encoder thread untouched, still OFF).
BENCH-REQUEST 28386d2: full default glmark2 list (must stay >= 14385; ideas expected to rise
toward native 3635).  Ideas draw-path CPU is down 38-52% vs f5607b5 in interleaved
MGL_PROFILE rounds (see session 12); under the 4-6 load of this session absolute FPS deltas
were unmeasurable, so the supervisor bench is the arbiter.

## session 12 (Oct 25, 06:15-09:30 EDT) — per-draw state-walk fast path (CPU/draw)

Goal (supervisor): ideas is per-draw-CPU bound; cut processGLState / bindList / per-draw
bookkeeping.  Baseline (f5607b5, steady state, interleaved runs): draw bucket 2.75-3.56 us/draw
(353 draws/frame); processGLState 1.7-2.3 us avg/draw (0.6-0.8 ms/f), bindList 0.33 us/draw,
enddraw-stamp 0.2-0.25 us/draw, ssboSizes 0.07.

1. FOUND+FIXED: DIRTY_IMAGE_UNIT_STATE was set by glBindImageTexture and by every sampler-
   uniform write (uniforms.c) but only the COMPUTE path ever cleared it — after the first
   such write every draw took the dirty-state walk forever (ideas: dirty=0x2000 on all draws).
   The render path's encoder refresh binds image units, so it now owns and clears the bit
   (MGLRenderer.m texture branch mask+clear).  This alone turned "clean" draws into clean
   draws for the first time.
2. compatDrawGate/compatDrawLatch (MGLRenderer+Compat.m, ext slot draw_fast): exact-content
   snapshots of every group the per-draw walk reads — buffer maps + per-buffer Metal identity,
   VAO fmt/binding + vattrib plan + generic attribs, texture units + realized tex/view/sampler
   ids + image units, rstate generations, encoder/PSO/cmdbuf/ring identity.  Verdicts:
   SKIP (whole walk skipped; ctx bits DIRTY_VAO|TEX_BINDING|BUFFER_BASE|IMAGE_UNIT proven
   redundant and cleared), BUFFERS (only listed-buffer content dirty: uniform upload -> ring
   re-stream + dirty-bit clear, walk skipped), RSTATE (only rstate encoder generation moved:
   extRstatePreDraw + skip), FULL.  extBufEndDraw gates the serial stamps the same way
   (stamps 353 -> 19/frame).  Content-based, so no missed invalidation is possible; mapped /
   coherent buffers, client-array streams, buffer textures, extra PreDraw hooks all decline.
3. texext PreDraw skip epoch: the gate proves texture state clean and stamps an epoch;
   extTexextPreDraw early-outs while epoch+program+encoder-gen are unchanged; any
   texture-dirty walk zeroes it (incl. tess/geom/flush paths that bypass the gate).
   ideas' 89 VAO-changed draws no longer re-walk both stages' sampler bindings.
4. Gate slimming: plan/fmt/binding compares behind DIRTY_VAO; tex compares behind the tex
   dirty bits; profiler buckets 22 (drawFastGate+partial) .. 24 added.

Results (interleaved prof_ab.sh A/B, clean rounds): draw bucket 0.667-0.806 ms/f =
1.89-2.28 us/draw (vs 2.75-3.56 base); processGLState 89 calls/f (was 353), 3.2-3.9 us each
(the genuinely-changed draws only); bindList 134/f x 0.4-0.6 us; stamps 19/f; drawfast 86%
hit; gate 0.22-0.30 us/draw.  drawfast-miss left: 82/f vao-fmt/binding (real per-mesh
re-specification), 7/f program switches.  Score A/B under load-4..6: 2544 vs 2462 (+3.3%)
in the one 6-round window where load was comparable; single-run deltas drowned in load noise.
Draw merging (Apple 353->266): the draw-run histogram says ideas has ZERO consecutive
identical draws (all runs length 1) — no mergeable runs with a per-draw identity key; skipped.

Remaining known costs: 82 full walks/frame for real VAO re-specification (mapBuffersToMTL
0.42us + plan rebuild + vattrib hook rebinding — that work is real, Apple pays it too), the
gate at 0.3us/draw, encode-draw 0.28us/draw.  Files touched this session: MGLRenderer+Compat.{h,m},
MGLRenderer+Draw.m, MGLRenderer+TextureExt.m, MGLRenderer.m (texture branch bit ownership +
epoch zeroing + mglExtHookCacheCount + prof buckets), mgl_compat_perf.h, buffers.c (temp debug
removed).  ab/base12 (f5607b5 build) + ab/run_ab_ideas.sh + ab/prof_ab.sh for repeatable A/B.


## session 11 (Oct 25, 01:55-06:15 EDT) — encoder thread: measured, pipelined, reverted

Merged gl46-port (611a380 = my 9c3c52d squashed; merge 85451b0 was content-identical).

### 1. Where the -30% comes from (MGL_CS_STATS=1, added this session)
`build` scene, 1 draw/frame, 19 packets/frame, flag ON, load 6-15:
- kicks/f=19.0 but wakes/f=2.0: the worker re-parks after every catch-up, so EVERY
  packet pays a mutex+signal kick (0.3-1.6us) on the GL thread (cs_kick cost 6-31us/f).
- worker busy=0.011-0.062 ms/f, GL-thread drain wait 0.001-0.015 ms/f: the Metal encode
  of a light frame is ~50us — far too small to pay for handshakes.
- worker wake latency (kick->executing): 90-692us quiet-ish, 0.9-1.6ms under load.  At
  2040fps (0.49ms frames) one parked-worker wake on the critical path = +20-30% frame
  time — matches the 20405->13740 regression.
=> per-packet hand-off can never win on light scenes; fixed the park churn (below) but the
fundamental "nothing to offload" stands.  ideas MGL_PROFILE=1 (steady, 353 draws/f):
processGLState 1.2-1.6 ms/f, draw bucket 1.9-2.6, encode-draw only 0.21-0.31 ms/f —
even a zero-cost thread caps the win at ~0.3ms/f.

### 2. Fixes kept (default OFF, inert unless MGL_ENCODER_THREAD=1)
- worker spins MGL_CS_SPIN_US (default 50us) before parking; drain spin is time-based
  MGL_CS_DRAIN_SPIN_US (default 30us; sched_yield removed — it hands the quantum away
  under load).
- MGL_CS_STATS=1 + compatCSFrameStats (hooked from compatSurfacePresent): per-120-frame
  packets/kicks/wakes/batches/drains, GL-thread drain+park+kick time, worker busy/idle/
  wake-latency.  test_csdbg.c added (block_vs_fs_control clone) — passes with and without
  the flag.

### 3. Full pipelining (Apple model) — built, half-working, REVERTED
Idea: zero hand-offs per frame.  CREATE packet (worker builds render encoders from a
deep-copied pass descriptor — the shared mutable ivar races otherwise), BLIT_COPY+COMMIT
packets for the present tail, layer.contents deferred one present gated on
`executed >= commit_seq`, endRenderEncoding/newCommandBuffer drains dropped, the drain
moved to endRenderEncoding (the shared funnel) for rare direct-encoder paths.
Correctness status: simple scenes + the 250-suite render right; test_geom fails
deterministically — the worker drops packets while cur_enc==NULL (fixed), two-open-encoder
asserts on readback blits (fixed), then a deterministic mtlcompilerd-connection abort in
r31_layered_fbo_no_attachment + 13-17min livelocks under load (NOT fixed).
Preserved on branch gl46/compat-cs-pipe (commit e77c606); reverted from this branch
(014f08c).  If anyone ever retries: the abort smells like heap corruption from the packet
path — check arena/ring handling of the obj3 field and the deep descriptor copy first.

### 4. Task 2 data (next session): per-draw CPU cuts, no thread
ideas steady-state per frame (353 draws): dirtyState-block 1.22-1.65 ms (processGLState
1.19-1.59 of it — ~3.5-4.5us/draw), draw bucket 1.93-2.62, enddraw-stamp 0.14-0.28,
bindList 0.21-0.28 (353 calls/f), ssboSizes 0.075 (353/f), encode-draw 0.21-0.31,
updateEncoder 0.04-0.06, vao-buflists 0.02-0.07, swap 0.17-0.32, newRenderEncoder 0.10-0.18.
Targets ranked: (1) processGLState per-draw work, (2) bindList construction per draw,
(3) enddraw-stamp, (4) ssboSizes caching (per program+binding, not per draw).
Apple-ACC-style: record state deltas per draw, validate/encode once per draw batch.

BENCH-RESULT 9c3c52d (code 5d3ad2b) flag OFF score=14385 (best so far) / MGL_ENCODER_THREAD=1 score=10031 (-30%) native=6574 ratio=2.19 load=4.5-7 — NOT pinned; native10/mglR13/mglR13t 01:40 EDT Oct 25. With the thread ON: light scenes 20405->13740 (build), ideas 2962->1357, shadow 13504->8553, jellyfish 15555->10470; slightly up: gl41-mrt 7341->7708, texarray 11430->12747, terrain 2165->2305, refract 3830->3956. Per-frame hand-off latency dominates — keep default OFF.
READY 4dfce4d (CS encoder-thread prototype, default OFF; gl46-port merged through 74185cd; suite 250/0, test_geom 603/0, both with and without MGL_ENCODER_THREAD=1)
BENCH-REQUEST: two runs of the full default list (same mode as mglR12), one with
MGL_ENCODER_THREAD=1 exported.  The flag adds one worker thread that replays recorded
render-encoder packets; under load 6-8 it is score-neutral but cuts GL-thread user CPU
~10% on ideas (9.5s vs 10.6s per run, 45us/frame).  At low load (3-4) that CPU saving
should convert to +8-12% on draw-heavy scenes (ideas, shadow, gl41-mrt, conditionals).
Also worth reading: MGL_PROFILE=1 with the flag prints "MGL CS: packets=… drains=… sleeps=…".
Recheck note (done this session): shadow 13029->11018 and terrain 2220->2045 in mglR12 are
LOAD, not code — simultaneous A/B vs bf0074e: shadow 0.966/1.005/0.977, terrain
0.997/1.023/0.997 (alternating launch order).
BENCH-RESULT beb4a3a (built at HEAD 25dcabe) score=14249 native=6575 ratio=2.17 load=6-7 (user Docker) — NOT pinned; mglR12 vs native9 21:34 EDT. Flat vs mglR11 14255 (noise). ideas 2371->2900 (native 3640), gl41-mrt 7754, gl41-texarray 10435; down vs mglR11: shadow 13029->11018, terrain 2220->2045 (load 6-7 vs 3.5 — recheck).
READY beb4a3a (all code + study; this report line is the only later commit — bench the branch tip of gl46/compat; gl46-port 1335928 + 2657647 merged; session-9 per-draw CPU work)
BENCH-REQUEST: glmark2-macos full default list, background launch, window visible, same mode
as mglR11.  Changes since 09e272a are CPU-side per-draw savings (DIRTY_PROGRAM consumption on
the PSO fast path + encoder-state re-set dedup): expect +2.5-4.5% on draw-heavy scenes (ideas,
shadow, gl41-mrt), noise-level elsewhere.  Matched-phase interleaved native-vs-MGL ideas is now
0.91-1.04x (was 0.91-0.94x); the bench-table ideas 2371/3623 was machine-phase skew.
BENCH-RESULT 09e272a (built at HEAD 4106e66) score=14255 native=6550 ratio=2.18 load=3.5-9 — NOT pinned; mglR11 vs native8 19:14 EDT. desktop-shadow 4804 > native 4642 (was 4345); gl41-mrt 7337; terrain 2220 ~ native 2274; still below native: ideas 2371/3623, refract 3599/3660 (~equal).

## session 10 (Oct 24, 22:00-01:00 EDT) — shadow/terrain recheck (load, not code) + CS encoder-thread prototype

Branch: 8e86b40 (gl46-port merged) -> 5d3ad2b; gl46-port 74185cd (geom r48-49) merged after.

### 1. shadow/terrain regression recheck (mglR12 table)
Simultaneous A/B current-vs-ab/bf0074e, alternating launch order (ab/run_ab_s10.sh):
shadow 0.966 / 1.005 / 0.977; terrain 0.997 / 1.023 / 0.997.  The 13029->11018 and
2220->2045 deltas were machine load (6-8 vs 3.5), not code.  Closed.

### 2. MGL_ENCODER_THREAD=1: one encoder worker thread (DXVK CS-thread, Apple ACC)
New file MGL/src/MGLRenderer+CompatCS.m (+~700 lines), category MGLRenderer (CompatCS),
declaration in MGLRenderer+Compat.h.  Default OFF (env MGL_ENCODER_THREAD or MGL_CS_THREAD).

Design:
* newRenderEncoder() wraps the real encoder in MGLCSEncoderProxy and stores it in
  _currentRenderEncoder.  All shared code keeps calling encoder methods unchanged; the
  proxy records: PSO, vertex/fragment buffer(+offset/bytes), textures, samplers,
  viewport(s)/scissor(s) (arrays snapshotted into the arena), cull/winding/fill/clip,
  depth bias, depth-stencil + stencil refs, blend color, visibility result, all 6
  drawPrimitives/drawIndexedPrimitives variants incl. baseVertex/baseInstance and both
  indirect variants, memoryBarrier, useResource, tess factor buffer/scale.  Rare
  protocol methods (debug groups etc.) drain + call through (NSObject forwarding).
* One worker (pthread, per renderer via associated object) replays packets against the
  real encoder in order => Metal stream == GL stream.  Objects are CFRetained into
  packets; worker consumes with __bridge_transfer.  Ring: 16k packets, 512KB byte arena.
* Drains (GL thread must see all encoding done): proxy endEncoding (pass boundary —
  every compute/blit/encoder-creation path goes through it), newRenderEncoder,
  newCommandBuffer, extBuffersWillCommit (every commit), geomDraw:, extTessDrawPatches:,
  compatSurfacePresent blit, renderer dealloc, mgl_buffers_prepare_cpu_write (buffer
  write-after-draw ordering; the GPU serial stamps are taken at record time so the
  existing rename/wait tracking then handles the rest).  No buffers.c edits needed.
* Shared-file edits (all small): MGLRenderer.m (newRenderEncoder drain+proxy,
  newCommandBuffer drain, dealloc stop), MGLRenderer+Draw.m (extBuffersWillCommit drain),
  MGLRenderer+Geometry.m (geomDraw: drain), MGLRenderer+Tessellation.m (extTessDrawPatches
  drain), MGL/src/draw_ext.c (mgl_buffers_prepare_cpu_write drain), MGLRenderer+Compat.m
  (surface-present blit drain), MGLRenderer+Compat.h (declarations).

Bugs found and fixed during bring-up (worth remembering):
* drawIndexedPrimitives …baseVertex:baseInstance: is THE hot indexed path; leaving it on
  the forwarding fallback made every draw a condvar handshake (5x crawl, looked like a hang).
* Two deadlock holes in the handshake: (a) the drain must signal cv_work under the same
  mutex the worker parks on (a kick before the park is lost); (b) the worker must
  broadcast cv_done under its mutex after EVERY batch — a lock-free waiters check reads a
  stale 0 and parks the producer forever (lldb: written==executed==head with both parked).
* Ring-full / arena-full / any worker exception => recorder faults permanently; the proxy
  then drains + calls through = previous synchronous behaviour.

Results (load 6-8, user Docker):
* Correctness: suite 250/0 and test_geom 603/0 with the flag ON and OFF; ideas stress x5,
  jellyfish/terrain/desktop single runs: no hangs, no Metal exceptions.
* ideas user CPU: 9.5s (on) vs 10.6s (off) per fixed run = ~45us/frame off the GL thread.
* ideas score at load 8: neutral to -4% (the ~0.5 frame-end sleeps/frame + worker
  scheduling under load eat the saving; frame rate not GL-CPU-bound there).
* Profile counters (MGL_PROFILE=1): ~482 packets/frame, ~2.2 drains/frame, 21% of drains
  need a cv handshake — the hot path records without syncing, as designed.
* Next lever if we want more: also defer processGLState decisions (needs GL state
  snapshots per draw, Apple-ACC style), or drain less at frame end (encode the present
  path as packets too).

## session 9 (Oct 24, 19:34-21:00 EDT) — Apple-driver study (mtlcount), DXVK study, DIRTY_PROGRAM + encoder-state wins



Merged gl46-port 1335928 (start) and 2657647 (tess r19) before READY. Suite 250/0, test_geom 603/0.

### Apple GL-on-Metal reference study (supervisor 19:46)
`sample` of native glmark2 ideas (GLEngine -> AppleMetalOpenGLRenderer -> AGXMetalG16X):
draws are 19% of main-thread samples, CGLFlushDrawable/present is 65%. Hot GL-path symbols:
glDrawElements_ACC_GL3Exec / glDrawArrays_ACC_GL3Exec ("ACC" accumulator records, does NOT
validate/encode), gleDoDrawDispatchCoreGL3, GLDContextRec::beginRenderPass/buildPipelineState/
setRenderUniformBuffers/setRenderSamplersAndTextures/addRenderPassResources — i.e. Apple defers
validation + Metal encoding to flush time and re-sets only what a state group changed.
xctrace 'Metal System Trace' export failed on this box (Document Missing Template Error,
383MB trace) — replaced by ab/mtlcount.m: a counting interposer that swizzles a fixed list of
Metal framework methods (imp_implementationWithBlock passes (self,args...) WITHOUT the SEL —
capturing the selector per trampoline is required; also every new*PSO/depth-stencil/sampler
creator takes a trailing NSError** that must be forwarded).

Metal calls per frame, glmark2 ideas steady state (interval counts / measured frames):

| call                        | Apple (391fps, 2346f) | MGL 1335928 (288fps, 1728f) | MGL session-9 (358fps, 2148f) |
|-----------------------------|----------------------|-----------------------------|-------------------------------|
| drawIndexedPrimitives (all) | 266.5                | 273.8                       | 312.5                         |
| drawPrimitives instanced    | (in idx*)            | 32.0                        | 36.7                          |
| setVertexBytes              | 84.5                 | 0                           | 0                             |
| setFragmentBytes            | 83.5                 | 0                           | 0                             |
| setVertexBuffer:offset:     | 104.1                | 4.0                         | 4.5                           |
| setVertexBufferOffset:      | (n/a***)             | (not counted)               | 207.0                         |
| setFragmentBuffer(/Offset)  | 0                    | 1.2                         | 0.98+1.4                      |
| setFragmentTexture/Sampler  | 16.7                 | 0.87                        | 0.98                          |
| setRenderPipelineState      | 7.5                  | 6.1                         | 6.9                           |
| setDepthStencilState        | 3.4                  | 19.9                        | 2.2                           |
| setTriangleFillMode         | 20.9                 | 19.9                        | 0.99                          |
| setCullMode                 | 20.9                 | 19.9                        | 2.96                          |
| setFrontFacingWinding       | 20.9                 | 19.9                        | 0.99                          |
| render encoders / commits   | 2 / 2                | 1.7 / 0.87                  | 1.98 / 0.99                   |
| PSO/DS/sampler creation     | 0 (steady)           | 0                           | 0                             |
* Apple also batches: 353 GL draws -> ~266 Metal draws. ** not swizzled in the first run.
After session 9 MGL makes ~570 Metal calls/frame vs Apple's ~900 at the same draw count.

### Fixes shipped (both validated by simultaneous A/B vs ab/1335928 = session-8 baseline)
1. MGLRenderer.m compat fast path now consumes DIRTY_PROGRAM like the descriptor path does.
   The bit was set by glUseProgram and NEVER cleared on a fast-path hit, so every draw after
   the first program switch re-ran mapBuffersToMTL + bindActiveTexturesToMTL +
   bindTexturesToCurrentRenderEncoder. ideas MGL_PROFILE (same load): draw 2.52 -> 0.95
   ms/frame, processGLState 1.85 -> 0.55, bindTextures 0.227 -> 0.005 (7 calls/frame),
   mapGLBuffers walk 0.254 -> 0.045 (89 real VAO changes). A/B ideas +3.5-4.5%, shadow
   +0.5-2.6%, gl41-mrt +2.3-3.9%, jellyfish/terrain/refract noise (-2%..+2%).
2. MGLRenderer+RenderState.m rstateApplyEncoderState dedups re-sets against the last values
   pushed to the same encoder (depth-stencil object, stencil refs, viewport, scissor, cull,
   winding, fill mode, depth bias, clip mode, blend color; extRstateRenderEncoderCreated
   already invalidates per encoder). setDepthStencilState/cull/winding/fill 19.9 -> 1-3/frame.

### DXVK study + CS-thread evaluation (supervisor 20:36)
Read /tmp/dxvk-master/src/dxvk/dxvk_cs.{h,cpp} + dxvk_context/dxvk_graphics: app thread records
compact typed commands into 16KB DxvkCsChunks (data copied in), ONE DxvkCsThread executes
chunks against the context, synchronize(seq)/lastSequenceNumber() for flush points; state is
dirty-flag tracked per object class; graphics pipelines are content-hashed; constant data
streams through per-frame staging (discard) rings.

Evaluation for MGL (from today's numbers): MGL's residual per-draw CPU is ~2.7us/draw
(under load) of which the Metal emission is only ~0.5us (1 drawIndexedPrimitives + 1
setVertexBufferOffset + stamp); the rest is app-side compute (dirty walk, buffer-list diff +
memcmp, ring memcpy, serial bookkeeping). A CS thread can therefore hide at most ~0.2ms/frame
under load (~+8-12% ideas at 2400fps) UNLESS it also owns state translation — which requires
recording GL state snapshots per draw (DXVK-style chunk copies) because the app thread mutates
ctx->state for the next draw. Ordering also requires that every app-side encoder mutation
(newRenderEncoder, rstateApplyEncoderState, bindTextures, bind*BuffersToCurrentRenderEncoder,
clears, blits, readbacks, queries, xfb, tess/geom passes, commit) drains first — those sites
are spread over MGLRenderer.m/Framebuffer.m/RenderState.m. Concrete design that keeps it
contained: record ops only in MGLRenderer+Draw.m (extBufDrawArrays/Indexed/Indirect +
extBufBindList emission tail), CFRetain the referenced MTL objects, one worker with
os_unfair_lock+condvar and a seq counter, drain = wait until consumed, eligibility gate
(no xfb/query/tess/geom active, encoder exists, no client index lists), env MGL_CS_THREAD=1.
Estimated 400-600 lines across 3 shared files + Draw.m; NOT prototyped this session (bench +
suite green were the priority) — recommend a dedicated session with the gpuaudit CPU-state
dumper as the debugging aid.


READY 09e272a (gl46/compat; gl46-port bd31b86 merged as aad7d91 + session-8 per-draw CPU work)
BENCH-REQUEST: glmark2-macos full default list, background launch, window visible, same mode
as mglR10.  Changes since bd31b86 are CPU-side per-draw savings (texture-resolution cache,
7 latched getenv, memoized per-draw MSL strstr): expect small gains on draw-heavy scenes
(ideas, desktop, jellyfish, terrain) and no regressions; the mrt/texarray numbers should
return to their load-3-4 values (proven load, not code — see session 8).

## session 8 (Oct 24, 17:00-18:40 EDT) — mrt/texarray proven load; per-draw CPU work; machine oscillation

TASK 1 (gl41-mrt 6049 vs mglR9 7780, gl41-texarray 10569 vs 12356): LOAD, NOT CODE.
Three independent measurements on bf0074e vs bd31b86 (single-scene):
1. Interleaved back-to-back (3 rounds/scene): deltas inside round-to-round noise,
   sign flips (mrt 21192/21074, 21151/21222, 2976/2626; texarray 24449/26171,
   27328/27395, 21705/23440).
2. SIMULTANEOUS (identical machine state, launch order alternated, 3 rounds/scene):
   texarray ratios bd31b86/bf0074e = 0.966/0.999/0.952, mrt = 0.966/1.016/1.033 —
   no consistent delta beyond ~3%, and the 3-5% first-launched-window penalty is
   compositing order, not code.
3. Code audit bf0074e..bd31b86: neither scene enters the geometry-emulation path
   (MGL_GEOM_TRACE_DRAW=1: zero lines).  Hot-path deltas are a 2xMAX_BINDABLE_BUFFERS
   pointer scan per non-attribute buffer map (~50 ns), a tess-active guard per draw,
   minus the removed per-map getenv (environ lock).  Net < 1 us/frame vs the 25-40 us/frame
   the bench deltas imply.

TASK 2 (below-native scenes) — the machine itself is the dominant variable today:
the SAME build/scene swings 400 <-> 2900 FPS within minutes (user's VM/Docker
time-slices the virtualized AGX; I saw ideas 141/211/448/2350/2420 across the session,
and one simultaneous pair of 21195 vs 2350).  Interleaved native-vs-MGL (alternating
order, same state) is the only measurement that means anything here:
  ideas:speed=duration  native 2421/2460/2429/2433  mgl 2302/2435/2366/2362  = 0.95-0.99x
  terrain               native 2306/2290/2277      mgl 2732/2556/2891      = 1.12-1.27x
  desktop shadow        native 2446/2388/2484      mgl 2747/2624/2632      = 1.06-1.12x
  gl41-geometry         native 2247/2377/2350      mgl 2673/2963/21195*    = 1.19-1.25x
(*one side caught a fast phase).  So the bench's ideas 2177/3789 (0.57x), terrain
1919/2261 (0.85x), desktop-shadow 4345/4650 (0.93x) are NOT reproducible as code gaps —
they are sequential-bench artifacts of the machine's oscillation.

CPU-side work done anyway (ideas profile: 353 draws/frame, per-draw ~1.1 us CPU,
processGLState = 70% of it; sample attribution showed ~40% of draw CPU in the texture
resolution chain, plus hot getenvs and a per-draw strstr) — commit 09e272a:
- MGLCompatTexResolve (compat slot, 2x32 entries): skips texextValidate +
  texdataBindMTLTexture + texparamSamplerForUnit + texparamTextureForTex when the unit
  still feeds the same realized texture.  Keyed on tex ptr, tex->mtl_data,
  tex->params.mtl_data (ANY glTexParameter releases it -> mglTexParamInvalidate), the
  unit's sampler object + its params cache, and encoder generation.  tex->dirty_bits
  nonzero always takes the full path (the upload lives there).  VS geom-shadow slots
  bypass the cache (geomShadowCheckStale must run).  Replays through the shadow-filtered
  compatSet*, so a changed resolved id still rebinds.  [ideas] profiled bindTextures
  slot 0.331 -> 0.125-0.199 ms/frame.
- 7 hot-path getenv latched once: MGL_FP64_DUMP (per default-uniform binding per stage
  per draw), MGL_GEOM_DEBUG x3, MGL_VATTRIB_DEBUG x3, MGL_TESS_NOSPEC.
- state_ext.c rs_program_has_flat: per-draw strstr over the whole fragment MSL
  (provoking=LAST default + any LINES/TRIANGLES mode) now memoized content-keyed
  (mgl_msl_has_size_constants pattern).
- compat_misc.c clip_visible const fix (clears 10 warnings).
Measured vs session-8 base (simultaneous A/B): ideas +0-2%, terrain +0-1.6% — real but
small; today's GPU time-slicing hides CPU wins.  Profiled [ideas] per-frame draw slot
2.94 -> 1.38-2.58 ms (machine-dependent), mapBuffers slot 0.368 -> 0.157-0.281.

Verification: suite 65 files rc=0, 8184 checks / 0 failures; test_geom 603/0;
glmark2 --validate 0 Fail/Error (11-12 pre-existing "Validation: Unknown" scenes,
identical on the old build); stub_scanner EMPTY/NOOP/ASSERT clean for compat files;
build has no new warnings.

Method for future sessions on this machine: ALWAYS interleave/simultaneous when
comparing; single sequential numbers are meaningless under load swings of 6x.
ab/ trees (git-archive copies + external symlink): ab/run_ab_sim.sh (two builds),
ab/run_ab2.sh (two builds, this one vs ab/aad7d91), ab/run_nat.sh (native vs MGL).

BENCH-RESULT 5efa031 (built at HEAD 244a311) score=14125 native=6591 ratio=2.14 load=5-8 (user Docker/VM busy) — NOT pinned; mglR10 vs native7 16:20 EDT. gl41-geometry 3114 vs native 2637 (was 1114). Still below mglR9: gl41-mrt 6049 (7780), gl41-texarray 10569 (12356), terrain 1919 (2250) — load was higher, recheck on a quiet machine. Below native: ideas 2177/3789, desktop-shadow 4345/4650, terrain 1919/2261.
READY 5efa031 (gl46/compat; gl46-port 9cc5052 merged as 5614538 + the getenv hot-path fix)
— geometry-shader emulation throughput work + the mglP10 mrt/texarray investigation.
BENCH-REQUEST: glmark2-macos full default list, background launch, window visible (same
mode as before).  Focus scenes: gl41-geometry (expect well above the 1114 of mglR9; local
occluded run went 706->1285 FPS, GPU/draw 1138->901us free-running, serialized latency
3.3->1.5ms).  mglR9 vs mglP10 gl41-mrt/gl41-texarray: NOT a code regression — see the
session-7 note below; expect those two scenes back at their load-3-4 numbers.

## session 7 (Oct 24) — gl41-geometry throughput; mrt/texarray regression triage

gl41-geometry (the biggest gap, 1114 vs native 2605) is one 65536-point glDrawArrays per
frame whose GS expands each point to a 4-vertex strip.  Ground truth came from a new
opt-in hook, MGL_GEOM_GPUTIME=1 (MGLRenderer+Geometry.m, geom-owned): a completed handler
per command buffer the geometry path encodes into, printing the running average GPU busy
time per command buffer and per draw.  Stage ablation knobs MGL_GEOM_SKIP_
{CAPTURE,KERNEL,ORDER,GS,SCAN,COPY,RASTER} + MGL_GEOM_NO_{BARRIER,IMPLICIT} were added for
attribution (skipping a stage leaves the consumers reading the same data volume, so run
differences measure the stage; NO_BARRIER is a perf probe only, results may be wrong).

Findings on this machine (virtualized AGX): (a) the order pass's mgl_geom_lscan was a
single 32-lane simdgroup walking threads/32 chunks in a dependent load->scan->store chain
(~1.6ms serialized for 65536 threads); (b) kernel LAUNCH overhead is ~100us per dispatch
(the whole prep chain is 6 launches); (c) the raster pass is cheap in throughput terms
(~120us) and overlaps well.

Fixes (commit 3dc2b69 "geom perf: parallel chunk scan + affine implicit-window fast
path", merged gl46-port 9cc5052 as 5614538):
- lscan -> mgl_geom_pscan (1024-thread groups: simd prefixes + one threadgroup scan of
  the 32 simd totals, chunk-local bases) + mgl_geom_pbase (scan of the <= threads/1024
  chunk totals, writes ctl[1]); lcopy adds the chunk base.  gIdx grows by the chunk-sum
  words.  Serialized latency 3.3 -> 1.5ms; test_geom_perf assembly wall -38%.
- affine implicit windows: direct non-indexed draws (glDrawArrays*, no
  PRIMITIVE_RESTART_FIXED_INDEX) of POINTS/LINES/TRIANGLES(+adjacency) lists,
  LINE(_STRIP)_ADJACENCY and TRIANGLE_STRIP now skip the ENTIRE prep stage — the kernel
  derives each primitive window's vertex ids from hdr[19..21] (list/strip/strip-permute
  codes; the strip's odd windows apply the GL winding/provoking-vertex permutation,
  matching the prep kernel's case 8), the capture pass draws the vertex range directly
  (non-indexed), the GS dispatch is exact+direct, and the gPrim pool suballoc shrinks to
  the header.  geom_affine_stride() in geometry.c classifies; MGLGeomDraw.implicit_stride
  carries the code.  gl41-geometry: 6 launches fewer per draw.
- gl41-geometry local (occluded, load ~5): FPS 706 -> 1285; GPU free-running 1138 -> 901
  us/draw; CPU draw slot 0.9ms with zero pool waits (the geomSubAlloc waitUntilCompleted
  the earlier sample showed is gone now that the latency halved).

mglP10 gl41-mrt 7780->6552 / gl41-texarray 12356->10168: NOT reproducible as a code
regression.  Built bf0074e and 9cc5052 side by side (ab/ trees, -j4) and ran them
SIMULTANEOUSLY (same load, same throttle state): texarray 5781 vs 5729, mrt 1181 vs 1250
(sign flips between scenes, +-1-6%).  Per-command-buffer GPU time is identical (94.6 vs
94.4us).  MGL_PROFILE CPU slots are within noise (the machine's load swings 3-5 dominate;
the P10 bench ran at load ~5 vs 3.1 for mglR9, and mrt/texarray are the two
highest-FPS/most load-sensitive scenes).  geom r47's only hot-path additions for non-SSBO
scenes are a 32-compare scan per non-attribute buffer map and — the one real waste — a
getenv("MGL_BINDMAP_DEBUG") PER BUFFER MAP PER DRAW (environ lock in the hot path).  That
getenv is now read once into a static (MGLRenderer.m, shared file, surgical; debug print
behaviour unchanged).  SSBO correctness untouched by it (debug-print guard only); SSBO
behaviour itself is r47's and was not modified.

Remaining gl41-geometry gap vs native (1285 vs ~2605-equivalent locally): the frame is
still ~14 launches worth of GPU work + ~0.6ms CPU submit.  Next levers, in order: merge
mgl_geom_init into mgl_geom_lzero and mgl_geom_finalize into mgl_geom_pbase (-2
launches); extend the implicit path to drawArraysIndirect and to the tess producer; cut
CPU submit (the capture pass's descriptor walk + processGLState full-dirty re-walk per
draw — geomDraw marks DIRTY_VAO|DIRTY_PROGRAM|DIRTY_STATE after the capture pass, so the
raster pass re-translates everything every draw).

BENCH-RESULT c57f712 (built at HEAD 67074d2) score=14066 native=6608 ratio=2.13 load=3.1 — NOT pinned (0 scenes at 119-121); mglR9 vs native6, 12:08 EDT. Slower than native: ideas 2179/3514, desktop-shadow 4316/4677, gl41-geometry 1114/2605; ~equal: terrain, refract, buffer.
READY c57f712 (gl46/compat, fixes on top of 1984249 which already contains gl46-port a2f16f4
via merge 04d3fbd; `git merge gl46-port` confirms "Already up to date").
BENCH-REQUEST: glmark2-macos full default list, background launch
(nohup sh -c "DYLD_INSERT_LIBRARIES=<wt>/build/nsgl/libmgl_nsgl.dylib glmark2-macos
--data-path ..."), window visible — the exact mode that pinned before.  Expected: no scene at
120-121 FPS; local full run Score 5075 with every scene free-running ([build] 6813-7980 vs
6367572 reference 7081 back to back; [buffer] 38-99 is CPU-bound and the reference build reads
49 in the same conditions — absolute scale tracks machine load, the pin did not).
Verification on the final build: suite 7557 checks / 0 failures / 65 files / rc=0
(test_geom 59 checks, 3/3 — was SIGSEGV since bcc0629); nsgl_smoke PASS (present == readback
byte-identical, CALayer flip calibration ok, present throughput 6305 Hz > 240 Hz floor);
stub_scanner EMPTY/NOOP/ASSERT clean for compat+nsgl files.

ROOT CAUSE FOUND + FIXED (perf session 6, 10:00-11:00 EDT) — the 120 FPS pin was NEVER the
direct-present path.  fa528c9 added a compatSurfacePresentEnabled() call to
fbEnsureDefaultFramebuffer(); the NSOpenGL interposer constructor called
MGLgetCurrentContext() (libmgl auto-context init -> default-fb ensure, renderer already owns an
offscreen CAMetalLayer so _layer != nil) BEFORE its setenv("MGL_SURFACE_PRESENT","1") at the end
of the constructor -> the static env cache in compatSurfacePresentEnabled latched "disabled"
for the whole process -> EVERY swap fell through to -[CAMetalLayer nextDrawable], which pins to
the display refresh (120-121 FPS visible window, 27-40 FPS occluded).  That is why bcc0629's
direct-off default did not help and why single-scene foreground runs looked fine (they also ran
the drawable path; the pin visibility depends on window order/compositing).
Evidence (all full default list, background nohup launch, back to back):
- HEAD build: 27-40 FPS every scene (sampler: 175/193 samples in nextDrawable usleep)
- 6367572 (gl46-port = beab36b+geom r46, i.e. pre-fa528c9 present path): 5663-7097 FPS
- HEAD with MGL_SURFACE_PRESENT=1 pre-set in the env: 7633-7980 FPS (same binary!)
- variants: -D1 (direct-present code removed) free; -D2/-D3 alone still pinned (the D2/D3
  "revert" builds were no-ops at first — GNU make 3.81 same-second mtime trap; deleted the
  .o files to force rebuilds before trusting those results)
FIX (two layers):
- MGL/nsgl/mgl_nsgl.m: setenv moved to the TOP of the constructor (before MGLgetCurrentContext)
- MGL/src/MGLRenderer+Compat.m compatSurfacePresentEnabled: only a POSITIVE env result is
  cached (early "unset" reads no longer latch disabled forever)
RESULT: full default run, same launch: Score 5075, every scene 38-7980 FPS free-running
([buffer] 38-99 is CPU-bound under load 4.7 — the 6367572 reference build gets 49 in the same
conditions; [build] 6813-7980 vs reference 7081).  Direct present (MGL_DIRECT_PRESENT=1) is
exonerated but stays OFF by default pending a dedicated bench.

SECOND BUG (same session): the bcc0629 texture-walk inlining in bindTexturesToCurrentRenderEncoder
dropped the accessors' NULL-program guard -> SIGSEGV in test_geom (transform-feedback capture
encodes with no program bound; crashed at 30/59 checks since bcc0629, "suite 6964/0" was really
6964/0 + test_geom rc=139).  Fixed by keeping the direct struct read but guarding
STATE(program) == NULL (accessor contract preserved: count 0, loop skipped).
Suite now: 7557 checks, 0 failures, 65 files, rc=0 (test_geom 59/59, 3/3 runs).
LESSON: reverting a default in source is meaningless unless the binary is rebuilt — the 09:19
nsgl relink reused the 08:34 libmgl (which already had the fix; mtime analysis + disassembly of
the getenv default proved the binary matched the tree, so the pin was real code, not staleness).
Also: incremental variant builds with same-second mtimes are no-ops under GNU make 3.81 —
delete the .o or touch the source.

BENCH-RESULT 50dc9b1 (built at HEAD 1984249) score=NOT-RUN — STILL vsync-PINNED: full default run (mglR8.log/mglR8b.log) = 120-121 FPS on every scene; gl46-port build (beab36b+geom) in the same way = 20340 FPS; supervisor 09:20-09:45 EDT

REGRESSION+REVERT bcc0629 (08:45 EDT): BENCH 04d3fbd = 119 — the direct-present path
(fa528c9) pinned EVERY scene at 120-121 FPS = display vsync in a visible window
(cross-process IOSurface sync at commit when the render pass writes the CA-displayed
surface).  Could NOT be reproduced locally: the nsgl_smoke window is not composited the
way the bench window is (3.1 kHz both paths; occluded/background windows take no fence).
Fix: blit path default again (a4f8027 present behaviour); MGL_DIRECT_PRESENT=1 keeps the
experimental path available.  nsgl_smoke now FAILS if present throughput at swap
interval 0 is under 240 Hz (2x refresh) — catches gross caps when the window is actually
composited; note the local repro limitation.  LESSON recorded: local stability runs must
check FPS magnitudes, not just completion.
Still in from this session: geom capture-PSO cache (gl41-geometry local +27%), MSL-key
cache, per-draw strlen removal, depth/stencil one-entry cache, pass-descriptor reuse,
texture-walk accessor inlining.  gl41-geometry remains the big scene target (MGL ~1000
vs native 2609; the capture-PSO cache landed after the last clean geometry number).
READY 50dc9b1 (branch head ad6251d = gl46-port a2f16f4 merged + geom capture-PSO cache + direct-present
regression reverted to the a4f8027 blit path + texture-walk accessor inlining +
nsgl_smoke present-throughput check).  Clean rebuild; full suite 6964/0 (65 test files,
several have timing-dependent check counts); nsgl_smoke PASS with 14 kHz presents;
--validate Success on texture/shading/desktop-shadow/bump/effect2d/pulsar.
Expected vs the 10918 run: flat (present path identical to a4f8027), gl41-geometry up
from the capture-PSO cache.  BENCH when convenient.
BENCH-RESULT 04d3fbd (built at HEAD bfc5f92) score=119 native=6599 ratio=0.02 load=2.9-5.0 — REGRESSION: every scene pinned at 120-121 FPS (vsync) with direct present; supervisor run mglR7 08:22 EDT

PROGRESS fa528c9 + 04d3fbd (perf session 5, 07:35 EDT): DIRECT PRESENT — the default
framebuffer renders straight into the IOSurface the swap shows (fbEnsureDefaultFramebuffer
hands out the pool write slot; compatSurfacePresent then only sets layer.contents — the
blit encoder + fullscreen copy per frame is gone; it measured ~2.5x the command-buffer
commit cost locally).  Write-slot rotation is lazy (after a present, at the next ensure),
so readback-after-present still sees the shown surface; a rotated back buffer loads
DontCare (GL: post-swap back-buffer content undefined).  fbSourceFor (readback) resolves
the default FB color WITHOUT the rotating ensure.  MGL_DIRECT_PRESENT=0 is the fallback
to the blit path.  nsgl_smoke PASS (present == readback), --validate Success (texture,
shading, desktop:shadow), suite 7557/0 after merging gl46-port a2f16f4 (geom round 46).
Also landed earlier this session (79e782d): geom capture-PSO cache — geomDraw rebuilt the
capture-pass pipeline on EVERY geometry draw (4-6% of the gl41-geometry frame; cached on
MGLGeomProgram keyed on capture-function + vertex-layout hash); local gl41-geometry
773-799 -> 1010-1015 FPS.
BENCH-REQUEST 04d3fbd (expect every windowed scene up by the saved blit+encoder, biggest
on the 1-draw scenes; gl41-geometry up from the PSO cache; watch gl41-instancing/texarray/
mrt — they read 15-23% below the previous run with no causal change, likely variance)
STABILITY 08:05 EDT: full local glmark2 run through all scenes with the direct-present
path — no GPU errors, no crashes, 37 scene results (killed during the last scene);
--validate Success on every scene that has a reference checksum locally.
BENCH-RESULT df38931 (built at HEAD a4f8027) score=13458 load=4.8-7.8 validate=27/0 (supervisor run mglR5 04:18-04:25 EDT; an earlier run mglR4 = 11012 was invalid: macOS AddressBook/remindd storm, load 85)
BENCH-RESULT 2e20f64 (built at HEAD 109bf4f) score=13415 load=4.4-4.7 validate=27/0 (supervisor run 03:20-03:27 EDT; log ~/MGL-cts/glmark2-bench/mglR3.log)

PROGRESS 2e20f64 (perf session 4, 2026-10-24 03:10): four per-draw/per-encoder CPU costs removed.
1. PSO fast-path key caches (MGLRenderer+Compat.m + mgl_compat_perf.h, compat-owned):
   compatMSLKeyCached (direct-mapped on msl_str ptr, 2x128-byte head/tail memcmp validate;
   index hash is multiplicative — a >>5 mask aliased 4 KB-strided malloc addresses, the
   desktop:shadow sample showed it re-hashing every draw) and compatVAOHashCached (single
   entry, exact memcmp of enabled_attribs..element_array; mtl_data excluded).  Was ~320 B of
   FNV per raster stage + ~130 dependent multiplies per draw.
2. geom pool exhaustion doubling (MGLRenderer+Geometry.m, geom-owned, flagged): slots grew
   to need*2 (~2 draws), so gl41-geometry's 9.8 MB/frame of GS scratch exhausted the 4-slot
   rotation within ~3 draws and every following geomSubAlloc blocked in waitUntilCompleted
   (~2.7 ms/frame in the sample; 60/66 samples in psynch_cvwait).  A slot reclaimed while it
   cannot fit the current draw now doubles (per-slot 64 MB cap, global 512 MB growth budget,
   MGL_GEOM_POOL_BUDGET override).  Local trace: waits/draw 68% -> 46% and falling as the
   pools reach steady state; test_geom 593/0, forced-tiny-pool tests unchanged.
3. Per-draw strlen of generated MSL gone: geomSSBOSizesNeeded + mgl_msl_has_size_constants
   (MGLRenderer+Geometry.m, geom-owned, flagged) validated their memos with strlen(msl) —
   strlen(30 KB) x2 per draw from geomBindSSBOSizes (ideas profiler: ssboSizes 0.28->~0);
   fingerprint is now a 32-byte head memcmp.
4. Depth/stencil one-entry cache (MGLRenderer+RenderState.m, rstate-owned, flagged): the
   16-arg NSString key + dictionary lookup per render encoder (10/frame in desktop:shadow)
   is now an integer compare on the unchanged-state fast path.
Also: profiler slot 19 = fbBuildPassDesc timing (4.4 us/encoder at load 5 — the Metal
renderCommandEncoderWithDescriptor dominates the ~30-48 us encoder creation, and the 10
encoders/frame of desktop:shadow are intrinsic: all FBO switches happen by re-attaching
textures to ONE fbo object, every attach must end the pass).
Suite 7556/0 after every step; nsgl_smoke renders (present == readback).
PROGRESS 109bf4f (perf session 4, cont.): fbBuildRenderPassDescriptor now reuses the last
descriptor per (ctx, fbo, collected targets, pre-pass invalid-mask) — the encoder copies it,
so reuse is legal; ~4 us/encoder saved on the ~50% hit rate of desktop:shadow (MGL_DEBUG
prints 'pass-descriptor cache HIT'; targetsChanged counter under MGL_PROFILE showed the 10
encoder teardowns/frame are all genuine pass switches: 1 command buffer/frame, the scene
re-attaches textures to one fbo object).  Suite 7556/0 again; gl41-streaming-sync shows no
ClientWaitSync warnings with the merged fence listener.
BENCH-RESULT 109bf4f = 13415 (supervisor 03:28 EDT, validate 27/0): ideas 1319 -> 2189,
desktop shadow 3369 -> 4243, desktop blur 3939 -> 4399, most simple scenes +5-10%.
gl41-geometry 1000 -> 678: geomSlotSize's need*2 fallback SHRANK the 64 MB slots on every
wrap after the budget declined the doubling (allocation thrash; the same change was 2x
FASTER under the local occlusion throttle, where the wait dominated).  FIXED in 36b6031:
geomSlotSize returns 0 (keep the current buffer) when the doubling is declined and the slot
still fits a draw, and never returns a smaller size than the buffer it replaces.  Local:
grows 45/1396 draws (was 48/386), waits 21% of draws, 773-799 FPS locally (was ~500).
gl41-streaming-sync: no ClientWaitSync warnings locally; the bench's single warning is the
first-frame PSO-compile stall (fence signalled late), not the listener path.
MERGED gl46-port 4b71da1 (tess round 17); full suite 7556/0 after merge.
BENCH-RESULT df38931 = 13458 (supervisor 04:26 EDT, built at a4f8027): gl41-geometry 678 -> 1151
(the slot-shrink fix), gl41-streaming-sync 849 -> 956 (above native 867), desktop shadow
4354 (native 4592).  gl41-instancing/texarray/mrt read 15-23% lower than the previous run
with no causal change of ours (instancing/texarray/mrt use neither GS pools nor the
compat fast-path caches; the only non-tess delta was the geom slot fix, which cannot touch
them) — likely variance at 6-19k FPS; worth watching in the next bench.
READY a4f8027 (= gl46-port 4b71da1 merged in df38931 + 36b6031 geom slot fix + a4f8027
MSL-key cache/draw-run histogram).  Clean rebuild, full suite 7556/0, nsgl smoke renders,
stub scanner clean for compat files.

BENCH-REQUEST e9a376a (code = f8b9c47 after the FBO/alpha PSO fast-path fix; expect
desktop blur + shadow ~2-3x; ideas/geometry unchanged)
BENCH-RESULT e9a376a score=12344 load=2.9-7.0 validate=27/0 (supervisor run 00:48-00:55 EDT; log ~/MGL-cts/glmark2-bench/mglR2.log)

READY f8b9c47 (= gl46-port 34381c6 + compat perf through the FBO/alpha PSO fast path;
clean rebuild, headless suite 478 PASS / 0 FAIL, nsgl interposer smoke renders,
glx shim builds [no X server on this Mac for a live run]).
Supervisor BENCH-RESULT for aa2adea: score 11932 @ load 3.8-4.0 (native 6558, bar 6000
PASSED).  Since then: desktop blur/shadow PSO fast path (see below) — desktop shadow
128 -> 325 FPS with the profiler running; expect a good chunk of the score gap closed.
BENCH-READY aa2adea recorded below (score 11932, supervisor run).

NEXT (vs native): desktop shadow 1032|4592 (PSO fix landed, re-measure), ideas 1631|3582,
desktop blur 1745|2979 (same PSO fix applies), gl41-geometry 982|2668,
gl41-streaming-sync 846|867 (+ ClientWaitSync GL_TIMEOUT_EXPIRED warning),
verify light scenes ~2x native are real (--validate already 27 Success / 0 Failure).
NOTES (analysis for the next session):
- light scenes verified REAL: build/texture/shading --validate Success locally
  (conditionals/function have no bundled reference -> Unknown).
- ideas is CPU-bound when the window is visible; per-draw floor is now
  ~1 drawIndexedPrimitives + ~1-2 offset binds (uniform data changes per draw by
  design).  Going materially faster needs draw batching (per-instance UBO indexing
  via shader rewrite) — architectural.
- gl41-geometry runs the 3-pass emulation (capture render pass + import kernel +
  pass-through raster, GPU-built indirect args, no CPU readback): the remaining
  982-vs-2668 gap is per-draw Metal pass overhead; fusing capture+import would need
  geom-domain rework.
- ClientWaitSync: fence.c busy-polls extBufEvent.signaledValue; the glmark2 warning
  means one fence never signaled within its timeout — likely a fence whose command
  buffer was dropped/replaced before commit (AGX recovery path); scene is at parity
  (846 vs 867) so low priority.
- local FPS runs in this session were occlusion-throttled (background window, ~161
  FPS); only supervisor-run numbers and MGL_PROFILE counters are trustworthy.

PROGRESS e511d81/f8b9c47 (perf session 3 round 2, 2026-10-24 00:30): PSO fast path now
covers DIRTY_FBO and DIRTY_ALPHA_STATE draws (desktop blur/shadow: 26 descriptor walks
per frame, 4.5 ms/frame, EVERY draw).  Key additions: attachment formats + sample counts
of the current encoder, hashed once per encoder generation (attachment changes always end
the encoder: framebuffers.c targetsChanged).  desktop shadow @ load 4 WITH profiler:
128 -> 325 FPS, pipeline-descriptor 4.5 -> 0.03 ms/frame, pipeHit 26/26, processGLState
5.8 -> 1.0 ms/frame.  Merged gl46-port 34381c6 (geom round 45: mtl_index buffer slots
routed through the shadow helpers).

BENCH-READY aa2adea (score 11932 @ load 3.8-4.0, supervisor run 23:40-23:47; native 6558,
bar 6000, Parallels ~14000).  --validate 27 Success / 0 Failure.
Session-3 deltas that got here (from 4349): encoder bind shadow + streaming argument ring
(88% of per-draw argument binds skipped), PSO fast path keyed on rstate content hash
(17 -> 4 descriptor walks/frame), ffp user-program-unbind correctness fix.

BENCH-REQUEST 2aa2adea (full default glmark2 list, windowed 800x600, swap interval 0 as in
sessions 1+2; env must NOT contain MGL_PROFILE — note the empty string enables the profiler)
BENCH-RESULT aa2adea score=11932 load=3.8-4.0 (supervisor run 23:40-23:47 EDT, full default list, swap 0; --validate: 27 Success, 0 Failure, rest Unknown=no reference; log ~/MGL-cts/glmark2-bench/mglR1.log)

PROGRESS d3cbfd6 (perf session 3 continued, 2026-10-23 23:00): still no clean benchmark
(load 9-12 all day: user's UTM/Parallels VM; measure only when load < 3).  Landed:
1. b2595ec encoder bind shadow + streaming argument ring: 88% of per-draw argument binds
   skipped (1592/frame), ring carries 270 KB/frame with offset-only updates; per-draw
   Metal encode calls now ~1 draw + ~0.6 binds.  Kill switch MGL_BIND_SHADOW=0.
2. 4f3885d PSO fast path keyed on rstate CONTENT hash (was the monotonic generation):
   pipeline-descriptor walks 17 -> 4/frame (the 4 unavoidable declines), 175 hits/frame.
3. d3cbfd6 CORRECTNESS FIX found by the new tests: bind_ffp returned NULL for a bound
   user program, so unbind_ffp issued glUseProgram(0) after EVERY compat draw — the
   app's program was silently unbound, next draw fell back to fixed-function (white)
   and glUniform* failed 0x502.  Apps that rebind per draw (glmark2) never noticed.
   Read-verify loops did (tests/headless/test_compat_perf.c: 127 checks — ring wrap
   over 2 MB, client arrays per draw, CPU indices, program alternation, texture
   content changes).  Suite fully green.
Design borrowed from ANGLE's Metal backend (src/libANGLE/renderer/metal/:
mtl_command_buffer.mm state cache keyed by content, mtl_buffer_pool.mm streaming
pools fenced by command-buffer serial, ProgramMtl default-uniform pool).
CAUTION: the bind shadow must see EVERY bind to the shared encoder; the xfb/geometry
capture paths swap _currentRenderEncoder around their own encoder (compatBindInvalidate
guards the swap) — test_xfb catches violations.
Baseline to beat when quiet: 4349 @ load 4.5 (b5df6c6); bar 6000; native 6558; Parallels ~14000.
Sanity (not a benchmark): ideas+desktop @ load 10 renders correctly, 217 FPS windowed.

PROGRESS 4f3885d (perf session 3, 2026-10-23 20:55): **no clean benchmark yet — machine load
10-12 all session (user's UTM/Parallels VM active); measure only when load < 3.** Two Metal-call
reductions landed (both profiler-verified on ideas @ load 8, counts are load-independent):
1. b2595ec encoder bind shadow + streaming argument ring: 88% of per-draw argument binds
   skipped (1592/frame), ring carries 270 KB/frame (uniforms/client arrays/const attribs/
   copied indices/SSBO sizes) with offset-only updates; per-draw Metal encode calls are now
   ~1 draw + ~0.6 binds (was ~4.5 binds).  Kill switch MGL_BIND_SHADOW=0.
2. 4f3885d PSO fast path keyed on rstate CONTENT hash (was monotonic generation_pipeline ->
   every blend/mask change minted a new key): pipeline-descriptor walks 17 -> 4/frame (the 4
   unavoidable declines), 175 fast-path hits/frame.
Design borrowed from ANGLE's Metal backend (src/libANGLE/renderer/metal/: mtl_command_buffer.mm
state cache, mtl_buffer_pool.mm ring with serial fencing, ProgramMtl default-uniform pool).
CAUTION for later sessions: the shadow must see EVERY bind to the shared encoder — the xfb /
geometry capture paths swap _currentRenderEncoder around their own encoder and reuse the
category's bind code (compatBindInvalidate guards the swap); test_xfb catches violations.
Baseline to beat when quiet: 4349 @ load 4.5 (b5df6c6); bar 6000; native 6558; Parallels ~14000.

PROGRESS b5df6c6 (perf session 1+2, 2026-10-23 18:30): glmark2 full default list (swap interval 0)
**4349 @ machine load 4.5 / 3709 @ load 8** — the Mac is shared with other agents' verify loops
(earlier clean-machine measurement of older code: 4160 @ load ~1; expect >= that when quiet).
Score path this session: 37 (vsync bug) -> 1929 -> 3865 -> 4160 -> 4349.  Native 6558, Parallels
~14000, user bar 6000 (not met yet — do NOT write BENCH-READY below 6000).
CPU profile (user-reported 180% CPU on classic scenes): after this session's fixes the classic
scenes run at ~102-111% CPU (ideas/desktop; single GL thread, CPU-bound) and gl41 scenes at 185%
(Metal driver threads at 6000+ FPS) — the inversion is fixed.  xctrace Time Profiler on ideas
(7.6 s, 480k draws): per-draw MGL code is now ~100-300 ns per frame item; the remaining mass is
unsymbolized AGX driver encode internals (setVertexBytes/drawIndexedPrimitives) — the per-draw
Metal API cost is the ideas/desktop/buffer bottleneck (ideas 353 draws/frame).
Done this session (each commit suite-green, 473 PASS): NSOpenGL swap-interval 222/196 +
getValues:; msl_binding sentinel guard (terrain SIGSEGV in AGX setTexture, regression test);
IOSurface present path (CGL-like async swap: blit-copy via MTLBlitCommandEncoder + one-time
CALayer y-flip transform, validated by a CALayer-flip calibration in nsgl_smoke) — killed the
nextDrawable block (36 -> 3518 FPS on build); compat pipeline-key cache (content-keyed MSL hash
+ VAO hash + rstate generation_pipeline + FBO ptr; separate pipeline generation counter added to
mgl_ext_rstate.h); no encoder rebuild for dirty VAO; memoized spvBufferSizeConstants strstr
(geom round 44 re-added two full-MSL scans per draw); skip redundant setRenderPipelineState;
vaoChanged no-op for identical re-specified layouts; points-vs-pixels fix
(wantsBestResolutionOpenGLSurface swizzle + per-bind headless flag — the user-visible
"quarter of the window" bug); MGL_PROFILE=1 per-draw profiler (noinline+barriers — microsecond
attribution is unreliable on this virtualized Mac, use A/B FPS).
Top remaining per-frame costs: (1) pipeline-descriptor slow path on real program/blend/FBO
switches (~80-150 us each, 17-21/frame ideas); (2) per-draw AGX encode (~5-10 us/draw, needs
denser batching to beat); (3) the buffer scene's single big draw at 10x native (GPU-side,
vertex-throughput bench in tests/headless/test_compat_thru.c reproduces ~20M tri/s).
Next: bench clean (score target 6000+), buffer/desktop GPU-side mystery, reduce descriptor-path
cost (rstateSpecialize/gaPSOKey), gl41-streaming-sync waits.

BENCH-READY c74797c DYLD_INSERT_LIBRARIES=$HOME/MGL/wt/compat/build/nsgl/libmgl_nsgl.dylib /tmp/glmark2-native/opt/homebrew/bin/glmark2-macos --data-path /tmp/glmark2-native/opt/homebrew/share/glmark2

(zero-copy present: flushBuffer -> MGLswapBuffers -> GPU blit + presentDrawable; stderr is quiet by default: ~27 KB/min vs ~5 MB/min before; gl41-pipeline now runs; gl46-port merged incl. geom round 98a20cf)

READY c74797c (gl46/compat: NSOpenGL interposer zero-copy present + quiet logging + gl_PerVertex fix; GLX shim fence swap; headless suite 6667 checks / 0 failures; test_geom 593/0; glmark2-macos --validate green)

Session log for the GL 4.6 compatibility-profile work (task: agents/tasks/compat.md).
Update this file at every commit so a fresh session can continue.

## Session 9b (2026-10-23 12:15-13:00) — supervisor follow-ups: zero-copy present, dylib paths, logging, gl41-pipeline
1. ZERO-COPY PRESENT (MGL/nsgl/mgl_nsgl.m): -setView: now binds a real MGLRenderer to the view
   through the entry GLFW uses (objc MGLRenderer -createMGLRendererAndBindToContext:view:,
   declared via a protocol; the renderer is retained as an associated object): CAMetalLayer on
   the view, default framebuffer sized from the drawable, -flushBuffer = MGLswapBuffers() =
   fbSwapBuffers -> GPU blit + presentDrawable (zero CPU copies; the CGImage readback path
   stays as a fallback for view-less contexts).  setValues:forParameter: SwapInterval toggles
   layer.displaySyncEnabled (0 = no display sync); makeCurrentContext binds the headless
   renderer lazily so GL-before-setView still works.  glmark2 --validate still green.
   NOTE: on this Mac the display is asleep/locked, so nextDrawable stalls ~30 ms/frame (sample
   shows the block inside nextDrawable — nothing consumes the drawables); on a visible display
   the path is drawable-recycle bound and the old 7.6 ms readback floor is gone.
2. DYLIB PATHS: both shims re-export libmgl as @rpath/libmgl.dylib with LC_RPATH @loader_path
   and a libmgl.dylib symlink beside them (install_name_tool -change; NSGL got it now) — they
   load from any cwd.
3. LOGGING (shared MGLRenderer.m, logging-only): per-frame INFO/DEBUG NSLogs gated behind
   MGL_DEBUG (render-encoder create/end pair, tess-pipeline creation cluster incl. the
   vertex-descriptor dump).  Root cause of the "already gated" cmdbuf lines still firing:
   `static BOOL mgl_log_cmdbuf = -1;` — BOOL is C99 bool on modern SDKs so -1 becomes true and
   the `< 0` sentinel never triggers; changed to `static int`.  glmark2 stderr ~5 MB/min ->
   ~27 KB/min (what remains is one-time init logs and spec-correct
   glGetString(GL_EXTENSIONS)=GL_INVALID_ENUM on core contexts).
4. gl41-pipeline "Set up failed" FIXED: its #version 150 vertex shader redeclares
   `out gl_PerVertex { vec4 gl_Position; };` which glslang rejects below GLSL 4.10 without
   GL_ARB_separate_shader_objects (Mesa and Apple's 4.1 core accept it).  shaders.c (shared,
   compiler-only edit): if the source redeclares gl_PerVertex at < 4.10 and does not already
   request the extension, inject `#extension GL_ARB_separate_shader_objects : enable` after
   the #version line.  Regression tests: glsl_150_gl_per_vertex_redeclaration
   (test_compat_glsl.c, 38 checks); the scene now runs (Validation: Unknown = no reference
   checksum, not a failure).
5. CTS compat-superset sampling (started): KHR-GL46.draw_indirect.* core 68/70 vs compat
   66/70 — the two extra failures are negative-noVAO-* which EXPECT GL_INVALID_OPERATION for
   VAO 0; correct for core but legal in a compat context (the default vertex array exists), so
   not an MGL bug.  buffer_storage 6/9 == 6/9 (same three core failures).  The ~/MGL-cts MGL
   platform gained MGL_CTS_FORCE_COMPAT=1 to force a compat context for core packages
   (glu::ApiType::compatibility rewrite in MglContextFactory::createContext, ninja rebuilt).

## Session 2026-10-23 (fresh session, phase 2)

* **CTS compatibility contexts + KHR-GL42-COMPAT.shaders42.*: 14/14 PASS.**
  - glcts platform glue (`agents/conformance/glue/framework/platform/mgl/tcuMGLPlatform.cpp`,
    installed into ~/MGL-cts by setup.sh; rebuild: `ninja -C ~/MGL-cts/build glcts`):
    PROFILE_COMPATIBILITY requests are no longer NotSupported; the render context calls
    `mglCompatCreateContext()` (exported by libmgl) and remembers it; makeCurrent() re-asserts it.
    Run: `bash tests/cts_one.sh 'KHR-GL42-COMPAT.shaders42.*' /tmp/cts_compat`
    (`--deqp-gl-context-type=mgl`; the COMPAT profile comes from the test package, not the flag).
  - Fixes the CTS found in MGL compat:
    1. `glTexImage1D` in a compat context raised GL_INVALID_ENUM (compat_tex_image1D forwarded
       to glTexImage2D with target GL_TEXTURE_1D) — killed every CTS case at the gluStateReset
       (it does texImage1D). Now a real 1D path (mglTexImage1D + legacy_upload 1D branch).
    2. Core-profile shaders (#version > 140 without "compatibility") were translated, so the
       negative cases (gl_FragColor under `#version 420` must fail) wrongly compiled. A
       version/profile pre-scan (scan_glsl_version) now leaves core shaders untouched.
    3. Translator gaps: texture{1,2,3}DProj{,Lod}, textureCubeLod, gl_MaxVaryingFloats →
       gl_MaxVaryingComponents, the full shadow1D/2D family with bias overloads; shadow
       implicit-LOD helpers are emitted for fragment shaders only (defining
       texture(sampler2DShadow,...) in a vertex shader is illegal core GLSL).
  - gl46-master.txt has no compatibility-profile cases (its 7 "compat" name hits are unrelated
    core tests owned by other domains). gl30..gl43-compat master lists = remaining CTS work.

* Merged gl46-port twice (fp64, geom rounds 32-41, tess rounds 9-13). **IMPORTANT pitfall: after
  `git merge gl46-port`, `rm -rf build/MGL && make -j2 lib`** — a plain incremental build left
  ~113 stale objects compiled against the old `glm_context.h` layout and produced an
  ABI-inconsistent dylib. Symptoms looked like GPU races: test_xfb scatter lost varyings,
  test_debug SIGSEGV'd in mglDebugResetStatus reading `ctx->state.ext_debug` as NULL
  (offset 0x99e0 vs the value actually initialized), lldb DWARF offsets disagreed with the asm.
  Clean rebuild: all green. If a "flake" survives a relink, suspect stale objects first.
* Fixed the real bug the immediate-mode stress test found (commit ac835b8):
  `mgli_upload_uniform` wrote the per-stage default-uniform MTLBuffer contents without the
  `mgl_buffers_prepare_cpu_write` hazard step required by mgl_ext_buffers.h (rename/wait against a
  still-queued draw). Shared-file edit: MGL/src/uniforms.c (+include, +1 call) — flagged to lead.
  (Merged into gl46-port as f9984b7.)
* test_compat_render2.c gained `immediate_mode_stress` (60 frames x 1024 translated quads,
  verified per frame; ~35 s, keeps the binary under the run.sh timeout). test_compat_gears.c now
  renders into an explicit FBO so the reference is display-scale independent.
* Full suite on the clean build: 59 binaries, 442 PASS / 0 FAIL, 6506 checks, 0 failures;
  test_geom 59/59; stress repro (80 frames) OK twice.
* render3 additions: line-stipple GL_LINES per-line restart fix (the second line of a batch
  started mid-pattern because the reset leaked the distance from the previous vertex), stipple
  factor/strip/array/indexed-order coverage, GENERATE_MIPMAP default query (was GL_INVALID_ENUM
  with untouched output for textures never touched by compat params) and pixel checks.

## How to build/test fast

* `./build_r.sh` — rebuilds the lib forcing every compat TU to recompile (dylib caches).
* `tests/headless/run.sh test_compat_pipeline test_compat_profile test_compat_render2` — the three
  compat suites (pipeline = pixel tests of the FFP; profile = state/queries/errors; render2 = the
  remaining rendering features).
* `MGLC_DUMP=1 <test>` dumps the generated FFP shaders to /tmp/mglc_vs.glsl / /tmp/mglc_fs.glsl.
* Probes: tests/headless/test_compat_*probe*.c are scratch, delete before commit.

## Architecture recap (see commits b1c6e60, 797057e)

* Context: `mglCompatCreateContext()` (tests) → compat profile flag in `MGLExtState_compat`.
  Core contexts keep rejecting compat calls (`MGLC_COMPAT_GUARD`).
* All compat entry points live in MGL/src/compat_*.c (dispatch in glm_dispatch.c is shared but the
  mgl* bodies are compat-only files).
* Fixed function pipeline: `compat_ffp.c` generates GLSL VS/FS from an `MGLFFPKey` (cached per key
  in `s->ffp_cache`), binds it at draw time (`bind_ffp`), syncs uniforms (`mgl_compat_ffp_sync`)
  guarded by `s->state_serial` (bumped by `mgl_compat_dirty`).
* Immediate mode: compat_vb.c `MGLCImm` batches into stream buffers on an internal VAO
  (s->imm_vao), GL_QUADS/QUAD_STRIP/POLYGON converted to triangles (`convert_prim`).
* Client arrays + VAO 0: compat_vb.c uploads ranges at draw.
* Display lists: compat_list.c records ops into a list; replay dispatches them.
* Raster ops: compat_pixel.c — RasterPos/WindowPos (CPU transform), DrawPixels/Bitmap/CopyPixels
  via an internal texture + a quad drawn through the FFP itself (`blit_quad`), accum buffer as a
  float attachment (see `#pragma mark accumulation buffer`).
* Compat GLSL (#version 110..150 / compatibility): compat_glsl.c rewrites to core GLSL.

## Former "core bugs" — resolved (they were never core bugs)

The two workarounds from earlier sessions were removed after the geom worker proved core uniform
packing + glProgramUniform* correct (tests/headless/test_uniforms.c, 297 checks):
* FFP scalar uniforms are now plain scalars again: mglNormalScale (float), mglAlphaRef (float),
  mglEnvScaleN (vec2), mglPointSizeParams/mglPointAtten (vec3). No vec4 packing.
* The "re-sync the first 3 draws because the uniform table is re-derived after the MSL compile"
  hack in mgl_compat_ffp_sync is gone; one sync per state_serial is enough.
Regression tests: test_compat_glsl.c glsl_scalar_uniform_sandwich / glsl_program_uniform_compat
(readback + pixels on a compat context), test_compat_render2.c specular_shininess (the float
uniform mglMatShininess[0] verified pixel-exact against pow(NdH, exp) for exp 128/16/2).

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| 1 | profile plumbing, compat GLSL | done (b1c6e60) |
| 2 | immediate mode, client arrays, quads/polygons, glRect | done (797057e) |
| 3 | fixed-function vertex (matrices, lighting, texgen, clip, fog coord, point size) | done (797057e), light probe tests added |
| 4 | fixed-function fragment (texenv+combine, fog, alpha test, stipple, point sprites) | done incl. polygon stipple texture |
| 5 | display lists | done (797057e), replay test passes |
| 6 | attrib stacks, raster ops (DrawPixels/Bitmap/CopyPixels/RasterPos/WindowPos, pixel maps, accum) | done, pixel-tested |
| 7 | legacy texture formats/state, glGet* of compat state | mostly done (profile suite) |
| 8 | select/feedback, evaluators, edge flags | done, tested (select per-primitive hits, feedback tokens) |

## Pixel-tested (test_compat_pipeline, 60 checks green)

immediate mode quads/rect, current color, glShadeModel flat/smooth (front/back), textures
(REPLACE/MODULATE + filtering), texgen OBJECT_LINEAR/EYE_LINEAR, clip planes, lighting
directional (ambient/diffuse/emission paths), fog linear/exp/exp2, alpha test funcs, display-list
replay, push/pop attrib, client arrays + VAO 0, glInterleavedArrays, glArrayElement.

## Pixel-tested now (test_compat_render2, 65 checks green)

matrix transforms (rotate/scale/push-pop), positional light + color material (per-vertex CPU ref),
texenv REPLACE/DECAL/BLEND/ADD/COMBINE (modulate/replace), RasterPos/DrawPixels + zoom,
glBitmap pattern (MSB-first, no flip, alpha-masked), accumulation buffer load/accum/return,
polygon stipple 32x32 (GPU texture sample), evaluators map1/map2 + EvalMesh1/2,
select mode (per-primitive hit records, viewport-exact culling), feedback mode
(GL_2D/3D/3D_COLOR/3D_COLOR_TEXTURE/4D_COLOR_TEXTURE token sizes).

All seven suites green: pipeline 60, profile 61, render2 65, glsl 18, render3 16, render4 35,
render5 17. Full headless suite: 4560 checks, 0 failures. stub_scanner clean for compat files.

## Fixed this session (committed per area)

* compat_ffp.c: color-material mode redeclared `vec4 mglMatA0...` twice → GLSL compile error →
  all *material* (non-color-material) lighting drew black. Now declared once.
* compat_ffp.c: FFP key now maps enums to small ids so distinct enum values can't collide in the
  key bytes (fog mode, env mode, combine modes, texgen modes, texture targets).
* compat_pixel.c: DrawPixels/Bitmap/CopyPixels blit path rewritten to draw through the FFP itself
  (glBegin quad + MODULATE + alpha test) instead of a separate blit program; saves/restores FF
  state. Fix: `mglActiveTexture(ctx, GL_TEXTURE0 + saved_active)` (state stores a unit *index*).
* compat.c: pixel transfer defaults — color scales default to 1.0, not 0 (DrawPixels was black).


## Session 8 (commit "default matrices, diamond-rule line rasterization")

* REAL BUG FIXED: the float matrix caches the generated FFP/compat-GLSL shaders read
  (s->modelview_f / projection_f / texture_f) were only refreshed by matrix_changed(),
  so a compat context that never calls a matrix entry point (glLoadIdentity etc.) drew
  with all-zero matrices — nothing appeared. mgl_compat_init_state now initialises the
  float caches to identity.
* Line rasterization "one pixel left/down" was NOT a Metal-vs-GL gap: a probe drawing
  the same segments on a CORE context (VBO + glDrawArrays) and on the compat context
  (glBegin) produces byte-identical coverage. A vertical line at integer x=16 lights
  column 15 (and not 16) because x=16 is the RIGHT corner of column 15's diamonds:
  GL's diamond-exit rule counts right/top corners and not left/bottom. Same for
  horizontal lines (row 15 for y=16). Test test_compat_render2.c line_diamond_rule pins
  this (endpoints, half-pixel cases), and the evaluators test now asserts column 15
  exactly instead of accepting both.
* Probe note: glReadPixels of a 1-pixel-wide GL_RGB row writes PACK_ALIGNMENT-padded
  rows (4 bytes per 3-byte row) — read GL_RGBA in probes.

## Session 8e (commit "glxgears-style end-to-end test")

* tests/headless/test_compat_gears.c: a classic-app frame — three gears built in
  immediate mode (triangle-fan faces with per-vertex normals + flat-shaded tooth
  flanks, design after the MIT glxgears gear()), recorded into display lists,
  drawn through the matrix stacks with a perspective frustum, two directional
  lights, materials with specular, GL_NORMALIZE and depth testing; the frame is
  compared against tests/headless/compat_gears_ref.h (10x10-block RGB averages,
  tolerance 4/255, generated with `test_compat_gears --dump`) plus structural
  checks (clear colour, coverage, all three material hues).  Deterministic
  across runs on this GPU.
* NOTE on the full-suite runs: with another agent building/testing in a parallel
  worktree on this Mac, whole-suite runs occasionally fail a *random* test
  (test_geom / test_introspect / even an untouched compat test) — every such
  test passes standalone and the next suite run is clean.  GPU contention, not a
  regression: verify suspected failures in isolation before digging.
* OPERATION NOTE: this repo is a shared checkout with worktrees — NEVER use
  `git stash` here (the stash list is shared; a `git stash pop` applied another
  agent's entry into this worktree once and had to be undone with
  `git checkout HEAD -- <file>`).

## Session 8d (commits "indexed line stipple ...", "feedback ... per-vertex data")

* Indexed (glDrawElements/glDrawRangeElements) line stipple works now: the distance
  from the strip start is a per-vertex attribute, but a vertex reused at two places
  of the index stream has two distances, so stippled indexed line draws are
  de-indexed (draw_elements_stippled_lines() in compat_vb.c): every enabled client
  array is gathered in draw order, upload_line_stipple() feeds the distances for
  that order, then a plain glDrawArrays draws it.  Falls back to the indexed path
  when the array memory cannot be read.
* REAL BUG FIXED while testing that: the generated shader tested the stipple
  pattern MSB-first ("bit 15 is the first pixel").  GL numbers the pattern
  LSB-first (proof: Mesa draw_pipe_stipple.c stipple_test() tests
  (1 << ((counter / factor) & 15)) & pattern, counter starting at 0).  The shader
  now tests bit (idx % 16) and tests/test_compat_render3.c line_stipple pins the
  exact pixel set of 0x00FF (first 8 of each 16 lit).  tests/test_compat_render8.c
  covers indexed strips with vertex reuse, per-segment restarts, and the
  unstippled indexed paths.
* Feedback now records per-vertex data: the color array (unit 0 texcoord array)
  is read per vertex for glDrawArrays/glDrawElements and from the immediate-mode
  batch (mgl_compat_cpu_draw_rgba / _indexed_rgba, mgl_compat_read_feedback_attribs
  / mgl_compat_read_array in compat_vb.c); when the arrays are disabled the
  current color/texcoord is recorded as before.  Gap (documented): lighting and
  texgen are not applied to feedback colors/coords (GL records the *lit* vertex
  color); feedback is a picking-style feature and no test relies on lit colors.
  Also fixed in passing: the glDrawElements feedback path used absolute indices
  into the [mn,mx]-rebased position array.
* Tests: test_compat_render8.c (72 checks), feedback tests in render2 (+46).

## Session 8c (commit "legacy texture formats at real precision")

* LUMINANCE / ALPHA / INTENSITY / LUMINANCE_ALPHA textures are no longer expanded
  to RGBA32F.  compat_state.c legacy_upload() stores them at the data's own
  precision — R8/RG8 (UNSIGNED_BYTE), R16/RG16 (UNSIGNED_SHORT), R8_SNORM/RG8_SNORM
  (BYTE), R16_SNORM (SHORT), R16F (HALF_FLOAT), R32F (FLOAT), 32F+CPU-normalize
  (INT/UINT) — with client format GL_RED/GL_RG, and sets the sampler swizzle to
  the legacy replication ((r,r,r,1), (0,0,0,r), (r,r,r,g), (r,r,r,r)).
* glTexSubImage with legacy client formats now works against the same storage
  (the core texdata path knows GL_LUMINANCE/ALPHA client layouts).
* glGetTexImage reads the stored channels and legacy_pack() rebuilds any legacy
  client layout (LUMINANCE/ALPHA/INTENSITY/LA/RGB/RGBA, byte/short/int/float).
* GL_TEXTURE_INTERNAL_FORMAT returns the format the app asked for (recorded);
  after re-specifying the texture with a core format the record and the swizzle
  are dropped (core query takes over; unsized GL_RGBA is reported as GL_RGBA8,
  which the GL spec allows).
* Test: tests/headless/test_compat_render7.c (82 checks).

## Session 8b (commit "GL_CLAMP / CLAMP_TO_BORDER: hardware border colours + shader emulation")

* GL_CLAMP/GL_CLAMP_TO_BORDER are no longer downgraded to CLAMP_TO_EDGE.
  Per texture axis, compat_state.c decides (mgl_compat_clamp_modes/apply):
  - border colour is one of Metal's three (transparent/opaque black, opaque white)
    AND (GL_CLAMP with NEAREST filtering, or GL_CLAMP_TO_BORDER): the sampler gets
    GL_CLAMP_TO_BORDER -> MTLSamplerAddressModeClampToBorderColor (hardware).
  - otherwise: sampler stays CLAMP_TO_EDGE and the generated FFP fragment shader
    emulates it (uniform mglBorderN + mglTexSizeN + a mglSampleN() helper):
    NEAREST -> border outside [0,1]; LINEAR -> the GL weight function
    w = clamp(0.5 + min(max(u,0), max(1-u,0)) * N, 0.5, 1) for GL_CLAMP (50/50 at
    the edges, saturated outside), w = clamp(0.5 + min(u, 1-u) * N, 0, 1) for
    CLAMP_TO_BORDER (full border half a texel out).  GL_CLAMP_TO_BORDER with an
    arbitrary border colour is emulated too (the core path keeps the nearest
    standard colour).
  * The decision is re-taken whenever WRAP_*/MIN/MAG_FILTER/BORDER_COLOR changes
    and after every texture upload (size uniform).  Wrap queries return the
    requested mode.  FFP key gained 2 bytes/unit (clamp axes+nearest). Cube-map
    textures keep CLAMP_TO_EDGE (border emulation is 2D/1D/3D only).
* Test: tests/headless/test_compat_render6.c (30 checks) — arbitrary border,
  hardware colours, runtime switching between the two paths, the 50/50 edge
  blend and its saturation vs CLAMP_TO_BORDER's full border.

## Session 7 (commit "GL_CLAMP / GENERATE_MIPMAP / texenv LOD bias")

* glTexParameteri(GL_TEXTURE_WRAP_*, GL_CLAMP) is accepted: the hardware gets CLAMP_TO_EDGE
  (Apple GPUs have no border fetch - documented approximation) and the requested GL_CLAMP is
  reported back by glGetTexParameter* (compat wrappers compat_tex_parameter*, compat_get_tex_parameter*).
* GL_TEXTURE_BORDER_COLOR round-trips through the core parameter storage (queries return it);
  sampling ignores it (gap).
* GL_GENERATE_MIPMAP is tracked per texture in the compat texture table; compat_tex_image1D/2D/3D
  call glGenerateMipmap after a level-0 upload when it is on (levels verified 32->4->1).
* glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_LOD_BIAS, x) is stored per unit, enters the FFP key and is
  passed to texture() as the bias argument (uniform mglLodBias, sync from the active unit).
* glPushAttrib(GL_TEXTURE_BIT)/PopAttrib no longer errors (it was re-applying the invalid params).

## Session 6 (commit "point sprites/attenuation, legacy texture formats, residency")

* Point attenuation now reaches Metal: mgl_compat_point_parameterfv enables/disables
  GL_PROGRAM_POINT_SIZE (the Metal pipeline flag that honours a shader point size) depending on
  whether the attenuation is the identity. Sizes verified pixel-exact: 16*sqrt(1/d) for d=4/9/16.
* Point sprites verified: gl_PointCoord (0,0) maps to the top-left texel of the point (spec).
* Legacy internal formats (GL_LUMINANCE/_ALPHA/ALPHA/INTENSITY + the *8/*16 spellings) are expanded
  to RGBA on upload by compat dispatch wrappers (compat_tex_image1D/2D/3D), stored in
  s->legacy_tex[], re-packed by compat_get_tex_image, and reported as LUMINANCE8/... by
  compat_get_tex_level_parameteriv. FLOAT/half/16-bit inputs expand through float; storage is
  RGBA32F.
* glAreTexturesResident raises GL_INVALID_VALUE for name 0 (spec) and still reports resident.

## Session 5 (commit "line stipple, texgen reflection fix, pixel maps")

* Line stipple is now really implemented: a per-vertex "distance along the line" attribute
  (MGLC_ATTR_LINE_I = location 1) is computed on the CPU in window space and streamed for
  immediate mode and glDrawArrays; the FFP vertex shader passes it to the fragment shader, which
  masks with pattern/factor uniforms. (Indexed lines with stipple still stream no distances - gap.)
* REFLECTION_MAP texgen referenced the reflection vector without declaring it (SPHERE_MAP only)
  -> shader compile error -> crash when drawing with no program bound. Both maps declare mglR now,
  and mgl_compat_ffp_program falls back to a minimal transform+color program when generation
  fails, so a bad state combination can never crash the draw.
* DrawPixels applies GL_MAP_COLOR via the R/G/B/A pixel maps (v -> map[round(v*(n-1))]); the
  default single-entry maps are identity (1.0).
* fixed-function texgen/lighting are per-vertex (as GL 1.x says), tests compute the
  interpolated CPU reference. (The "one pixel left/down" line note here was wrong — see
  session 8: that IS the GL diamond rule.)

## Session 4 (commit "compat GLSL fixes")

* gl_FogFragCoord was only substituted in fragment shaders - now also in vertex shaders (declared
  as `out float mglFogFragCoord`, already declared in the preamble).
* ftransform() wrapper referenced mglModelViewProjectionMatrix even when only U_MVM was used -
  the uniform is now declared in that case too.
* mgl_compat_sync_builtin_uniforms uploaded `spotDirection` (a vec3) with glUniform4fv ->
  GL_INVALID_OPERATION at every draw of a shader using gl_LightSource; now matches the type.
* glGetUniformLocation in compat contexts accepts built-in names (gl_ModelViewMatrix ->
  mglModelViewMatrix, gl_LightModel.ambient -> mglLightModelAmbient) via a compat dispatch
  wrapper (`mgl_compat_get_uniform_location`); unused builtins correctly return -1.
* gl_LightModel is now a proper struct in translated shaders (mgl_LightModelParameters) and is
  synced (`mglLightModel.ambient`).

## Session 3 (commit "render2 fixes")

* glBitmap: bits are MSB-first and row 0 is the bottom (no vertical flip); the pattern now lives in
  a dedicated `s->bitmap_texture` (GL_R8 swizzled A=RED) so MODULATE keeps the raster color and the
  alpha test discards 0-bits.
* Polygon stipple: the 32x32 pattern is uploaded (FNV-1a guarded) to `s->stipple_tex` on unit
  MGLC_STIPPLE_UNIT by `sync_stipple_texture()` at FFP sync time; `glPolygonStipple` stores the raw
  GL layout and `glGetPolygonStipple` returns it unchanged.
* Evaluator maps: GL_MAP1_x/GL_MAP2_x are now real caps (enable/disable/IsEnabled); de Casteljau
  used to memcpy GLfloat control points into a double array (garbage) and assumed a [v][u] layout -
  fixed to convert element-wise with the documented [u][v] layout; strides are bytes (0 = tight).
* Select mode: clip tests use the actual viewport; hits are recorded per GL primitive (QUADS = one
  record), coverage is an exact triangle/segment-vs-rect test (vertex/corner/edge cases); the
  name-count patch wrote to slot hit_zmin_slot-2 (out of bounds) and the close path never patched
  the header - both fixed.
* Feedback mode: rewritten (`fb_emit_prims`) with per-type vertex layouts (GL_2D/3D/3D_COLOR/
  3D_COLOR_TEXTURE/4D_COLOR_TEXTURE), per-primitive tokens; TRIANGLE_STRIP/FAN/TRIANGLES now emit
  (the old code relied on `mgl_compat_build_tri_indices` which only converts QUADS/POLYGON).
* (removed: the line-on-boundary note above was a misdiagnosis, see session 8)

## Legacy-GL app smoke (phase 2 step 4)

* `tests/headless/test_compat_gears.c` is a port of the real glxgears (gears geometry, FFP
  lighting/materials, display lists, projection/reshape) running against an MGL compat context:
  10/10 checks green every run (gears_frame verifies the rendered pixels of the three gears).
* The actual `/opt/local/bin/glxgears` cannot run on MGL: it links `/opt/local/lib/libGL.1.dylib`
  and needs GLX (glXChooseVisual/glXMakeCurrent...); libmgl exports no glX* symbols, so a GLX
  shim over XQuartz would be needed — platform integration work outside the compat domain.
* GLFW apps cannot create compat contexts either: `external/glfw/src/mgl_context.m` always calls
  `createGLMContext` (core) and ignores GLFW_OPENGL_COMPAT_PROFILE. external/ is off-limits for
  edits in this task; the fix would be one call to mglCompatCreateContext() when the compat
  profile hint is set (flagged to the lead).

## Remaining work (refreshed session 9, 2026-10-23)

Done since the last refresh (sessions 7-9): all eight milestones, CTS compat contexts
(KHR-GL42-COMPAT.shaders42 14/14), GLSL translator gaps (texture{1,2,3}DProj{,Lod},
textureCubeLod, gl_MaxVaryingFloats, the shadow1D/2D family incl. bias; sampler1D now
pixel-verified end to end, test_compat_glsl.c 36 checks), GLFW compat-profile contexts
(external/glfw), the GLX shim (glxgears 42 FPS on XQuartz) and the NSOpenGL interposer
(glmark2-macos full default list green, 27/27 reference-validated scenes).

Still open, in priority order:
* Compat-as-superset audit: run a sample of KHR-GL46 core groups through glcts with a
  COMPATIBILITY context (one batch each of shaders, texture_*, buffer_storage,
  draw_indirect, vertex_attrib_binding, transform_feedback, sample_variables) and compare
  against core-context results; fix the top regressions.  (Plan only so far.)
* GL_CLAMP border sampling: CLAMP maps to CLAMP_TO_EDGE (Apple GPUs lack border fetch);
  the border colour is stored/queried but not sampled.  Documented emulation gap.
* glIndex*/color-index rendering: state + queries only (spec-required minimum in RGBA mode).
* glRenderMode(GL_SELECT) name-stack depth and GL_FEEDBACK token stream corner cases.
* Evaluator (glMap1/2) normalization + glMapGrid subdivision limits beyond the pixel tests.
* GLU: no libGLU on top of MGL yet (X11 apps like glxgears only need gl*; apps needing GLU
  would require a GLU port or Mesa's libGLU linking against the GLX shim).
* Interposer/glx polish: vsync is a no-op (interval recorded), contexts are never destroyed
  (MGL has no teardown path), share lists are per-context.

---

## Session 17 (Oct 26, 12:00-13:35 EDT) — texture_swizzle diagnosis, ideas perf, 4 fast-path commits

Branch: gl46/compat. Tip after this session: `b80fabf` (merge of gl46-port 5178835/tess r25)
on top of `16179e8`. Full headless suite after everything: **7831 checks, 0 failures**.

### TASK 1 — KHR-GL46.texture_swizzle.smoke "hangs": DIAGNOSED, not a hang, fix is geom's link cache

Causal chain (all runs today, cap times in wall seconds, single case, --deqp-log-images=disable):

| build | result |
|---|---|
| my f6a2a46 (no link cache)          | killed at 1032 s, no verdict (CPU 6:44 and rising, progress verified) |
| my f6a2a46 + geom's geometry.c/mgl_ext_geom.h only (slot map + eager free) | **TIMEOUT at 900 s** — the slot map alone does NOT fix it |
| geom e767e6a (whole-link executable cache) | **Pass 478 s** |
| geom e767e6a + MGL_NO_LINK_CACHE=1   | **TIMEOUT at 720 s** — cache off, the same build collapses |

Sampling (30 s / 2.5 m / 5 m / 6 m): main thread walks a continuously CHANGING stack —
glslang preprocess/parse (mglCompileShader) -> mglLinkProgram -> run_link -> compile_stage ->
mgl_introspect_patch_stage (SPIRV-Cross) -> extProgramLibraryForMSL (MSL->AIR, waits on the
Metal compiler service); CPU time monotonically accumulating, no cycle. So it is per-link
throughput decay, not a deadlock. Why the workload is heavy: the smoke iterates 6^4 = 1296
swizzle values x n_texture_access x 4 channels x 2 stages — tens of thousands of
build+link cycles, each of which (without a link cache) redoes glslang, SPIRV-Cross,
MSL->AIR and a PSO.

Conclusion: the fix that matters is geom's **whole-link executable cache family** —
bc52d8b (content-keyed link cache, ~20x relinks), 7d0a47e (MSL entry-name content hash ->
MTLLibrary cache), 0410220 (hit skips the geom record rebuild), efb646c (keeps the glslang
compile cache hot) — NOT the geometry.c slot map by itself (proven above). All of those are
in gl46/geom r50-r53, not yet in gl46-port. geometry.c/mgl_ext_geom.h slot map still helps
(the table walk was quadratic + resurrected zombie programs per dead slot); take both when
geom's branch merges. No edits made by me to geometry.c.

### FPS task — where ideas lost ~16% (3404 @159d900 -> ~2874 @d31c9ba): INCONCLUSIVE today, environment could not be excluded

What I did:
* Bisect builds at 159d900 (p0), 64fa3f1 (p1=tess r21), 9485327 (p2), d31c9ba (p3) in /tmp/s17.
* The machine ran at load 4.9-17 all session (other agents' builds/tests, Docker, WebKit).
  Every wall-clock method (FPS, /usr/bin/time CPU, sample CPU-shares, MGL_PROFILE bucket
  medians, native-GL calibration) produced ratios with +-15% spread — the expected signal
  (~16% for p0->p3, ~3% for mine vs d31c9ba) is inside the noise envelope. One early clean
  round (load ~8) did show draw=0.791 p0 vs 0.906 p3 ms/frame (+14.5%) — direction matches,
  but later rounds under higher load reversed. CALL COUNTS ARE IDENTICAL across p0/p3
  (353 draws, 89 walks, 134 bindList, 352 encode) — if real, the regression is per-call
  cost, not extra work; r21's only per-walk code changes (mapBuffers UBO rows via
  mgl_buffers_indexed_binding, sampler-array element loop) are ~tens of ns, which cannot
  explain 16%. Cross-day comparison risk: yesterday's 3404 was measured with native at
  3576; today native GL itself was unmeasurable under load.
* NEEDS: a quiet-machine rerun of /tmp/s17-style A/B (p0 vs p3 vs tip) before blaming tess;
  scripts preserved in /tmp/s17/{bisect.sh,ratio.py,profmed.py}; the four /tmp builds are
  159d900/64fa3f1/9485327/d31c9ba + oldT (=7a50188 pre-session).

### Per-draw CPU: 4 commits (suite green after each, 7831/0 at the end)

1. `95a99c1` PSO cache hit path O(1): the hit path paid a linear scan of up to 512 slot
   structs + a move-to-front memmove of everything above the hit (several KB per
   first-of-group walk; ideas does 89 program-switch walks/frame). Hits now keep their slot
   position (stable indices), and a 16-entry direct-mapped key->slot memo answers repeated
   keys with one compare; the scan runs only for unknown keys; misses/eviction clear the memo.
2. `993e17b` differential snapshot latch: compatDrawLatch copied the whole ~4 KB snapshot
   after every walk; groups whose live content already equals the snapshot (ideas: identical
   fmt/binding/plan/current/image-units/textures across its 89 same-layout walks) now latch
   as memcmp+skip; the buffer-map list latch compares count+entries+backing ids first.
3. `0683180` gate drops two per-draw calls: [self extTessActive] (msgSend + tess table hash
   walk) is latched — it can only flip via link (arms DIRTY_PROGRAM -> dirty-mask miss) or
   program/pipeline rebind (identity miss), both checked earlier; mglExtHookCacheCount
   ("PreDraw:") is counted once per encoder generation instead of per draw.
4. `16179e8` current-attrib serial: the gate memcmp'd 16x56 B of generic attribs on every
   draw (glVertexAttrib arms no dirty bit). All four current-value writers in
   vertex_attrib.c bump a new MGLExtState_vattrib.current_serial; the gate compares the
   serial and only memcmps when it moved (memoizes the verified serial).
   SHARED-FILE EDITS (called out): mgl_ext_vattrib.h +1 field, vertex_attrib.c +1 line in
   set{F,I,U,L} (additive; setP delegates to setF). No behaviour change outside the gate.

Expected effect is a few percent of per-draw CPU each; a quiet-machine A/B (oldT vs tip)
must confirm — under today's load the median ratio was 0.955-1.031 with +-25% spread.
jellyfish/goal numbers (>=3400 / >=12800) NOT verified today — no quiet window existed.

### Also this session
* Merged gl46-port twice: 784c480 (my s13-s16 squashed back) as 7a50188, and 5178835
  (tess r25) as b80fabf; suite green after both merges.

## Session 25 (Oct 28, 04:2x-05:4x EDT) — merge gl46-port 1e840b4, READY re-verified

Merge 37254a7 (gl46-port 1e840b4 into gl46/compat @ 58d55ae). Conflicts: geometry.c
(gl46-port predates my 47d7336 geom_out_decl_size/geom_tess_clip_check — kept HEAD,
both call sites intact) and compat_report.md (kept HEAD). Gate after merge:
**617 PASS / 0 FAIL** headless.

fails_compat3 re-verified post-merge (tests/cts_group.sh):
- KHR-GL46.gl_spirv **9/9 Pass**
- KHR-GL46.separable_programs_tf **2/2 Pass**
- KHR-GL46.shader_group_vote **5/5 Pass**

READY 37254a7 (= 47d7336 + merge of gl46-port 1e840b4), A/B vs baseline-1e840b4:
- ab_bench (ab/abbench_s25.sh, min of 7 interleaved): m0 11.648 -> 11.157 (**-4.2%**),
  m1 101.061 -> 102.409 (**+1.3%**, within the +3% limit).
- locked glmark2 (ab/run_ab_s25.sh, best-of interleaved, load 0.0-0.9):
  ideas 2953 vs 2966 (**-0.4%**), jellyfish 25883 vs 25831 (**+0.2%**),
  gl41-geometry 21894 vs 22062 (**-0.8%**) — tess->geom capture path at parity.

## Session 25 addendum — the test_geom_xfbperf GPU page fault (NOT READY, root cause not fully closed)

Supervisor report: test_geom_xfbperf (built from this tree) caused GPU Address Faults +
a WindowServer GPU restart. Investigation summary:

* **Pre-existing at the READY**: a pristine `git archive 37254a7` build (built outside
  the worktree, no local changes) faults the same way — this is NOT from my post-READY
  commits (efbce1b's MSL pass is inert for this program: its shaders contain no
  `_atomic [[buffer(` args; the draw_ext.c change is CPU-side sync only).
* **Where**: the faulting CB is the draw's own (label = xfb-finish, the last set before
  commit: capture + prefix + scatter + finish + the raster of that draw).  CPU-pack
  mode (`MGL_GEOM_CPU_XFB=1`) never faults; `MTL_SHADER_VALIDATION=1` never faults
  (serialized/slow) => racing/stale data dependent, needs >= ~8 frames (several draws
  in flight).  Fault kills one CB and the primitive count loses ~2 frames.
* **CPU-side bounds now proven at dispatch** (new MGL_GEOM_DEBUG assert block in
  geomXfbEncode): recs/cnt/xbase/ctl slices, xfb dst capacities, nil bindings —
  all PASS on the failing config, so the bindings and slice sizes are not the fault.
  (One assert bug found+fixed on the way: cnt needs `threads` words, not threads*4.)
* **Three unguarded index computations found and bounded** (defence in depth):
  1. scatter `recs + (slot - n + 1 + p + j) * out_words`: uint underflow when a stale/
     half-written flagged record claims a strip longer than its position (n > s+1)
     => wild read.  Guarded (drop the record); the CPU pack had the same arithmetic,
     guarded too.
  2. scatter capacity test re-read `xfb[3+b]` after testing it (TOCTOU vs a concurrent
     draw's finish bumping the write offset): now a per-primitive `base[4]` snapshot,
     so each thread writes inside the capacity it tested.
  3. prefix scan `for (s = 0; s < cnt[i]; ...)` unclamped against the tile's record
     slots: clamped to maxv (same in the CPU pack).
* **Still faulting** after those guards (one sanctioned MTL_DEBUG_LAYER+debug run):
  the remaining suspect set is the CAPTURE/ASSEMBLY side (the indirect draw argument
  for the raster derives from live GPU counters; a stale/large count would walk the
  raster past the records buffer) or a consumer-tracking hole in the record-pool
  reclaim; both are geometry-domain code.  NOT root-caused to the end; needs a
  per-dispatch scoping run (errorOptions encoder status) or a geometry-domain review.
  **No more faulting runs were made** after the sanctioned one.

Gate with test_geom_xfbperf.c parked (pre-existing fault): **619 PASS / 0 FAIL**.

Commits this session: b5a262e (merge gl46-port 1e840b4 + READY re-verify), 5c57d94
(merge gl46-port 71cfe2a, arrs zero-init), efbce1b (texture_buffer.atomic_functions
fix: 15/16 -> 16/16 CTS, gl_spirv 9/9, sptf 2/2, vote 5/5, ab parity) + this addendum.

## Session 25 resume (22:32 EDT, after the user pause) — merge gl46-port 48a41ef

Merge d9abb22 (48a41ef = geom r59-r61 + my gl_spirv 9/9 landed by the lead) — clean.
Gate with test_geom_xfbperf.c parked (the pre-existing GPU fault; hard rule):
**619 PASS / 0 FAIL**.  (test_gather_xfb had "timed out" in the aborted 07:16 run —
that was the user pause freezing the process mid-suite; single run now: PASS in 4.2 s.)

A/B vs baseline-48a41ef (quiet machine):
- ab_bench m0/m1 min-of-7 interleaved: 12.746 -> 11.802 (**-7.4%**), 109.822 -> 100.286
  (**-8.7%**) — faster.
- locked glmark2 best-of interleaved (ab/run_ab_s48.sh): ideas 2788 vs 2752 (+1.3%),
  jellyfish 25596 vs 25663 (-0.3%), gl41-geometry 21929 vs 22123 (-0.9%) — parity
  (an external load spike 9-11 hit the jellyfish/geometry rounds; best-of held).
- (Earlier run vs baseline-1e840b4 had also been at parity; a one-off +12% reading
  against 71cfe2a under load 0.9 was noise, not reproduced.)

READY accounting: efbce1b (texture_buffer.atomic_functions: coherent image atomics
on buffer textures; texture_buffer 16/16 CTS, gl_spirv 9/9, sptf 2/2, vote 5/5,
gate 619/0) is READY-able with the numbers above.  c0cfdc8 (the xfb-pack bounds
hardening) touches the faulting path and is NOT claimed READY — root cause open.

Queue after texture_buffer: shader_storage_buffer_object.advanced-switchBuffers and
.advanced-indirectAddressing-case2 fail identically on baseline-48a41ef (pre-existing,
deep multi-draw cases); sparse_buffer_tests 3x NotSupported is correct (GL_ARB_sparse_buffer
not advertised, documented gap).  Otherwise my-area queue is empty.

## Session 26 (Oct 28, 23:18 EDT) — revert c0cfdc8, un-park test_geom_xfbperf

Supervisor direction: the branch tip must contain exactly the claimed READY work, and
no test may be parked out of the gate.  c0cfdc8 (xfb-pack bounds hardening +
MGL_GEOM_DEBUG asserts) has behaviour clamps, not asserts only, so it does not qualify
as "safe to include" — reverted on top of 6cd2083 (229c1b4).  Its investigation
narrative above is kept for the record; the guards are gone from the code.
test_geom_xfbperf.c was never removed from the tree (the "park" was a run-time
move-aside).

Sequence and results this session:

* Revert 229c1b4 applied clean (only agents/compat_report.md conflicted, resolved
  keeping both narratives).  Code delta vs gl46-port 48a41ef is exactly efbce1b
  (texture_buffer.atomic_functions) + the two merge resolutions.
* Clean build: `rm -rf build; make -j4 lib glx nsgl` rc=0, only the pre-existing
  upstream `-Wincomplete-implementation` for extProgramLibraryForMSL.
* test_geom_xfbperf ran ONCE at 229c1b4 (23:21, then inside the full gate 23:26):
  the TEST passed both times (200 frames, 2 checks, 0 failures) — but per the
  supervisor's 23:42 quarantine message those two runs GPU-page-faulted at machine
  level.  A test-level PASS therefore does NOT rule out the intermittent fault; my
  earlier "no fault" statement was wrong at the machine level, and the 06:52-07:16
  "pristine 37254a7 archive faults" observation stands as one data point in what is
  now clearly a latent, timing-dependent gl46-port bug seen from three worker trees.
  Not run again after 23:42.
* Full gate (23:26, xfbperf still in): 619 PASS / 0 FAIL, 9717 checks, rc=0.
* Quarantined gate per supervisor 23:42 (test moved aside for the run only, file
  restored, nothing committed): **619 PASS / 0 FAIL**, 9715 checks, rc=0.
  test_geom_xfbperf quarantined (supervisor 23:42).
* stub_scanner EMPTY/NOOP/ASSERT clean on compat.c/compat_ffp.c/geometry.c/
  draw_ext.c/MGLRenderer+Geometry.m.

### A/B harness defects found this session (important for all domains' numbers)

1. abbench_s2x.sh (m0/m1) was measuring MY BUILD AGAINST ITSELF.  libmgl.dylib's
   install name is the RELATIVE path `build/libmgl.dylib` (no @rpath), so `-Wl,
   -rpath,<baseline>` never applies (rpaths only substitute into @rpath install
   names) and dyld resolves the relative name against the LAUNCH CWD.  Run from the
   worktree root, BOTH /tmp/abb_base and /tmp/abb_new load wt/compat/build/
   libmgl.dylib — proven with DYLD_PRINT_LIBRARIES (identical UUIDs).  Every
   m0/m1 "delta" those scripts printed was noise (plus a base-first warm-up
   asymmetry), including this session's earlier -7.4%/-8.7% and the +23.7%/+41.7%
   first attempt tonight.  ab/abbench_abs.sh fixes it: copy each side's dylib to a
   private dir, `install_name_tool -id` to an absolute path, and PROVE the two
   sides load different files before printing numbers.
   (Also: this harness exports HOME='~' literally, so `$HOME/MGL-cts/...` paths
   silently break; resolve the real home directory first.)
2. glmark2 A/B (DYLD_INSERT_LIBRARIES) is structurally sound — nsgl carries an
   `@loader_path` rpath so the inserted baseline nsgl reexports the baseline
   libmgl — but only when the insert path is absolute.  Tonight's $HOME='~' would
   have silently compared system-GL vs system-GL; verified insert works by
   `mgl_nsgl: NSOpenGL -> MGL interposer active` in stderr.

### A/B vs baseline-48a41ef on 229c1b4 (fixed harness)

* ab_bench m0/m1 (abbench_abs.sh, min-of-7 and min-of-11, interleaved, load 2.3-5.4):
  m0 +4.0% then -13.0%, m1 -8.6% then -1.3% — the sign flips between runs with
  heavily overlapping distributions => parity within noise; consistent with the
  code delta (nothing in efbce1b/revert is on the m0/m1 paths —
  bound_for_foreign_gpu_write runs only in the CPU-read sync path, and ab_bench
  modes 0/1 never CPU-read a buffer).
* locked glmark2 (interleaved, alternating order, 3 rounds/scene, load ~3.5;
  NOTE absolute levels are 3-4x below last session's — plain system-GL ideas
  scores 550 tonight, so the machine is contended; only the interleaved deltas
  carry information): ideas 793 vs 786 (+0.9%), jellyfish 5606 vs 5550 (+1.0%),
  gl41-geometry 2105 vs 2153 (-2.2%) — parity.

READY (this session): 229c1b4 + the report/harness commit on top of it (no MGL
code changes in the latter) — gate 619 PASS / 0 FAIL, 9715 checks,
test_geom_xfbperf quarantined (supervisor 23:42); ab_bench + glmark2 parity vs
baseline-48a41ef on the fixed harness.  Claimed work in the tip: efbce1b
(texture_buffer.atomic.functions) + the gl_spirv tessGeom fix (48a41ef, landed by
the lead) + merge resolutions; c0cfdc8 reverted.

### Next (supervisor queue)
shader_storage_buffer_object.advanced-switchBuffers and
[HEAD
.advanced-indirectAddressing-case2 (fail identically on baseline-48a41ef).

## Session 26 addendum (00:05-00:35 EDT) — primitive_coverage root cause: PSO key
ignored the tessellation stages; FIXED (tessellation_shader 58/58)

Supervisor forward (tess's r30 report): KHR-GL46.tessellation_shader.single.
primitive_coverage "renders channel-reversed pixels".  Investigation:

* The two colours are exact RGBA reverses BY DESIGN: the test stencil-marks the
  tessellated patch's coverage (GL_NEVER+REPLACE), then draws reference triangles
  with GL_NOTEQUAL — any pixel left at stencil_pass_color (32,64,128,255) is a
  HOLE in tessellated coverage vs the reference rasterization.  One pixel
  (index 8188 of 2048x2048), 4 sub-iterations.  Not a swizzle bug.
* Reproduced on baselines: b7821df Pass, c6ded7d Fail (both via MGL_LIBRARY).
  c6ded7d is a single commit on b7821df, so bisected by content in a detached
  worktree build: reverting only the two mglProgramRegisterMSLIdentity call
  sites (MGLRenderer.m bindMTLProgram, MGLRenderer+RenderState.m specialize
  install) to no-ops makes c6ded7d PASS — the fnid registration is the trigger.
* Root cause (my s23 change): the registration made these programs ELIGIBLE for
  the compat pipeline fast path, whose key.a hashes ONLY the VS/FS/GEOM stage
  function identities.  For a tessellated draw the PSO's vertex function IS the
  TES (MGLRenderer.m swaps the kernel-typed vertex function for
  shader_slots[_TESS_EVALUATION_SHADER]->mtl_data.function) — the quad-tess and
  triangle-tess programs in this test share IDENTICAL VS+FS MSL, so their keys
  collided and the second program drew with the first program's PSO; its
  coverage differed from the reference draw by one pixel.  (b7821df passed only
  because the unregistered functions made the fast path decline.)
* Fix (MGLRenderer+Compat.m): key.a now mixes the TCS and TES function
  identities as well (id[5], two more odd multipliers; attached-but-unregistered
  tess stages still decline the fast path).  Non-tess programs: unchanged key,
  cost = two NULL slot tests.

Verification on the fixed tip:
* primitive_coverage: Pass x3 (c6ded7d+fix detached, branch tip x2).
* KHR-GL46 tessellation_shader **58/58 Pass** (was 57/58; 4m35s).
* separable_programs_tf 2/2, shader_group_vote 5/5, gl_spirv 9/9,
  texture_buffer 16/16 — no regressions.
* Full headless gate (test_geom_xfbperf quarantined (supervisor 23:42)):
  **619 PASS / 0 FAIL**, 9715 checks, rc=0.
* Perf: no new measurement — machine load 5.9-9.9 (other worker trees active;
  a locked glmark2 sanity under that load swung 4x and is meaningless).  The
  fix adds two NULL tests for non-tess programs, so the 23:44 parity numbers
  (fixed harness) still describe this tip for everything glmark2/ab_bench run.
---- (below: compat-worker session 26 continues)
.advanced-indirectAddressing-case2 (fail identically on baseline-48a41ef).


## Session 26 final (01:00-01:20 EDT) — SSBO advanced-switchBuffers / indirectAddressing FIXED; merge gl46-port 3cd6208

Both queued pre-existing fails were ONE bug, in my drawfast gate (MGLRenderer+Compat.m):

* DIRTY_BUFFER_BASE_STATE was listed in MGL_DF_REDUNDANT ("processing proven
  redundant by the content gates below") — but no content gate covered the four
  INDEXED binding tables (UBO / uniform-constant / SSBO / ACB).  A
  glBindBufferBase / glBindBufferRange between draws of one program therefore
  skipped mapGLBuffersToMTLBufferMap, the fast path re-bound the LATCHED map
  lists, and every following draw kept reading the first draw's buffer object.
  Repro (tests/headless/test_ssbo_switch.c): 4 quads from 4 SSBOs — only the
  first drew; with glFinish between draws the stale binding lagged visibly.
* Fix: gate 26 — when DIRTY_BUFFER_BASE_STATE stands, memcmp the four tables
  against the snapshot and MISS(26) to the full walk when any content moved;
  the latch snapshots them (compare-first).  Same-buffer rebinds compare equal
  and still skip; steady-state draws have the bit clear and pay nothing.
  MGL_BINDMAP=0 and MGL_BIND_SHADOW=0 both still failed, which is how the gate
  (not the bindmap prefix cache / bind shadow) was isolated.
* CTS: shader_storage_buffer_object.advanced-switchBuffers,
  .advanced-indirectAddressing-case1 and -case2 all Pass;
  shader_storage_buffer_object.* 124/124.
* tests/headless/test_ssbo_switch.c: bindBase / bindRange / finish loops, 12
  checks.  (Data layout note: std430 puts vec3 color at +16 of the 32-byte
  VertexData record — a probe writing it at +8 makes every quad render its
  red channel only, which cost one debug round.)

### merge gl46-port 3cd6208 (supervisor 01:03)

Merged as 4f136c3 (report conflict only, both narratives kept).  On the merged
tip, clean build rc=0 and re-verified: primitive_coverage 1/1, SSBO switch /
indirectAddressing-case1/-case2 Pass, separable_programs_tf 2/2,
shader_group_vote 5/5, gl_spirv 9/9, texture_buffer 16/16, tessellation_shader
58/58 (from the PSO-key fix below).  Full headless gate:
**620 PASS / 0 FAIL**, 9727 checks, rc=0
(test_geom_xfbperf quarantined (supervisor 23:42) — file moved aside for the run
only, restored after, nothing committed).

### A/B vs baseline-3cd6208 (fixed harness; machine load 4-6 from parallel trees)

* ab_bench (min-of-9, two runs): m0 +3.1% then +16.9%, m1 +2.2% then -10.4% —
  the noise floor between runs (base m0 min 9.4 vs 11.7) is larger than any
  delta; nothing in the delta (revert, PSO-key fix, gate-26) executes on the
  m0/m1 paths (no indexed respec between draws).  Parity within noise.
* locked glmark2 (interleaved, alternating order, load ~6): ideas 760/837 vs
  707/807, jellyfish 5523/5291 vs 5407/5271, gl41-geometry 1872/1851 vs
  1951/1871 — parity (mine >= base on ideas/jellyfish, -1..-4% on
  gl41-geometry).

READY (session 26 final): 4f136c3 + report commit on top (no code change) —
merge of gl46-port 3cd6208; PSO fast-path key includes TCS/TES
(tessellation_shader 57/58 -> 58/58); drawfast gate re-enumerates on indexed
respec (SSBO group 124/124); c0cfdc8 reverted; A/B harness defect found and
fixed (abbench had been loading the same dylib on both sides).  Gate 620 PASS /
0 FAIL (9717 checks + test_ssbo_switch 12 = 9727), test_geom_xfbperf
quarantined (supervisor 23:42).  ab_bench + glmark2 parity vs baseline-3cd6208.

## session 27 (supervisor 01:24 EDT) — r28 sweep NotSupported triage

Task: from ~/MGL-cts/mgl-r28-sweep/results.json (239 NotSupported / 116 Fail /
7 Crash / 1 InternalError), pick the NotSupported groups in my area (buffers,
DSA, copy/blit, spirv, NSOpenGL/GLX, compat) that a real implementation can
flip, report top 3, start the biggest.

NotSupported triage by owner (case counts):
* not mine / other domains: KHR_blend_equation_advanced 88 (renderstate),
  sample_variables 38 (samples_8 — Metal max 4 samples, hardware limit),
  shader_image_size 36 ("Required 4 image samples but only 0 available" —
  image2DMS as image, shaders), ext_texture_shadow_lod 16 (textures),
  texture_filter_minmax 5, shader_ballot 4, shader_viewport_layer_array 3,
  post_depth_coverage 2, parallel_shader_compile 3 (program domain),
  texture_barrier_ARB 4 (renderstate; needs GL_ARB_texture_barrier string),
  shader_draw_parameters_extension 1 (string IS exposed — needs a look, draw
  domain), sparse textures 21 (hardware gap).
* my area, real implementation possible:
  1. **GL_EXT_shader_integer_mix 11 cases** (shaders.shader_integer_mix 6 +
     shaders45 5): pure GLSL enable extension; glslang already implements the
     select-style integer mix() overloads for desktop GLSL 1.50-4.50
     (Initialize.cpp:3639 builtin gate, ParseHelper:3467 requireExtensions at
     <450) and they are core GLSL 4.50+ (shaders45.prototypes already Passes).
     Only the extension string was missing.
  2. GL_ARB_gpu_shader5 3 cases (gpu_shader5_gl.*): features core since
     GL 4.0; glslang supports the extension (#version 150 tests get upgraded
     to 330 core by shaders.c, extension stays).  Only the string missing.
  3. GL_ARB_sparse_buffer 3 cases: documented gap (sparse buffers) — skipped
     per supervisor.

### GL_EXT_shader_integer_mix (11 cases)

* tests/headless/test_compat_intmix.c (failing-first): extension in the
  glGetStringi list; `#ifdef GL_EXT_shader_integer_mix == 1` at #version 330;
  ivec4/uvec4/int/uint/bool/bvec4 mix() renders select semantics (a ? y : x);
  same code REJECTED at #version 330 without the extension (glslang hard
  error, matches CTS prototypes-negative which already passed); works at
  #version 450 without extension (core).
* Shared-file edit: MGL/src/get.c — added "GL_EXT_shader_integer_mix" to
  mgl_extensions[] (one string; mgl_num_extensions/mgl_extension_name pick it
  up).  No Metal change: glslang lowers integer mix to OpSelect, SPIRV-Cross
  emits MSL select().
* Result: test_compat_intmix 5/5 (9 checks).  Neighbours clean
  (test_compat_profile, test_debug, test_program, test_spirvbin,
  test_compat_glsl).

### GL_ARB_gpu_shader5 (3 cases) + CTS verification

* tests/headless/test_compat_gpus5.c (failing-first): extension in the
  glGetStringi list; the three CTS scenarios at #version 150 + `#extension
  GL_ARB_gpu_shader5 : require` (which shaders.c upgrades to 330 core, keeping
  the extension line): implicit conversions (int->uint, ivec4->uvec4,
  uvec4->vec4 with white-iff-exact render check), function overloading
  (f(vec4,vec4)/f(uvec4,uvec4) resolved through an ivec4->uvec4 implicit
  conversion), float bit encoding (floatBitsToInt/Uint, int/uintBitsToFloat
  round-trip); negative: `uint v = 1u - u1` (implicit int->uint) must NOT
  compile at #version 330 core without the extension.
* Shared-file edit: MGL/src/get.c — added "GL_ARB_gpu_shader5" to
  mgl_extensions[] (features are GL 4.0 core, already exercised by the
  gpu_shader5 group; glslang implements the extension itself).
* CTS against this worktree's build/libmgl.dylib:
  - KHR-GL46.*integer_mix* (14 cases): **13 Pass, 1 NotSupported** — the
    remaining `shaders.shader_integer_mix.prototypes` is a CTS-side gate (the
    "shaders" package registers the group at GLSL 3.30 and the case returns
    NotSupported for any version < 4.50; not reachable for us).
  - KHR-GL46.gpu_shader5_gl (3 cases): **3 Pass**.
  - KHR-GL46.exposed_extensions: Pass (extension list still consistent).
* Full headless gate (test_geom_xfbperf moved aside for the run only, restored
  after — supervisor 23:42 quarantine): **126 test binaries, 630 PASS / 0 FAIL**,
  0 build failures.  stub_scanner EMPTY/NOOP/ASSERT clean on get.c (only file
  touched besides tests).

Net effect on the r28 sweep: 16 NotSupported cases flip to Pass (11 integer_mix
minus the 1 CTS-gated case, plus 3 gpu_shader5_gl, plus both
prototypes-negative/45-prototypes which now also run — 13 + 3 counted above).

### merge gl46-port 539f74f (supervisor 02:36) + gate

* 1c8d936: merge of gl46-port 539f74f (only agents/compat_report.md conflicted,
  both narratives kept).  My s27 work re-verified on the merged tip:
  test_compat_intmix 5/5 (9 checks), test_compat_gpus5 5/5 (16 checks);
  CTS: integer_mix 13 Pass + 1 CTS-gated NotSupported, gpu_shader5_gl 3/3,
  copy_image 13/13 (one earlier Timeout was my own concurrent load; clean when
  run alone), packed_pixels.*rgtc* 12/12.
* Full gate (xfbperf still quarantined, moved aside for the run only):
  **126 test binaries, 0 FAIL**.
* Shared-file edit beyond s27's get.c: tests/headless/test_geom.c
  r54_compressed_no_3d — its TEXTURE_2D_ARRAY expectations encoded the old
  over-broad rejection; updated to the spec/CTS matrix (RGTC family succeeds
  on 2D arrays — online compression, GL_RED client data is legal for an RG
  base format per packed_pixels isFormatValid; S3TC/BPTC still error with
  uncompressed data = no CPU encoder; TEXTURE_3D still rejects all six).
  test_geom 670 checks, 0 failures.

### HANDED-OFF: packed_pixels depth-stencil regression on gl46-port

KHR-GL46.packed_pixels.varied_rectangle.{depth_stencil, depth24_stencil8,
depth32f_stencil8} fail on **baseline-539f74f** (A/B: 3 Fail with
~/MGL-cts/baseline-539f74f/libmgl.dylib, 3 Fail on my tip, 0 Fail in the r28
sweep = geom 3cac39f): "Invalid format used but glTexImage2D/3D succeeded" —
MGL accepts (format,type) combos for DEPTH24/32F_STENCIL8 internal formats
that the CTS isFormatValid(INPUT_TEXIMAGE) rejects (depth/stencil client
formats only with depth base formats).  Regression window 3cac39f..539f74f;
texdata/textures domain, not compat.  Repro: pattern
'KHR-GL46.packed_pixels.varied_rectangle.depth*' (8 cases, 3 Fail).
