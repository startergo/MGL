# MGL GL 4.6 port: status (2026-11-13 14:05 EST, gl46-port @ 59d72636)

## Merged compat r108 + tess r75: 40 of the 42 gl46-main regressions fixed (2026-11-13 14:05 EST)
The full run on 1cbdd1ce found 42 KHR-GL46 cases that passed on the Nov 1 build (7edc431) and failed on 2f95cadc. Each was bisected over the saved baselines and routed:

| cases | first bad | fixed by |
|---|---|---|
| packed_pixels float formats x27 | bd9ae9a5 (compat r103-r105) | compat r108 (e6b02647) |
| compute_shader.shared-struct, gl_spirv.spirv_validation_capabilities_test | a095741 (compat r97a-r98c) | compat r108 |
| direct_state_access.queries_functional (InternalError) | 4c3bb48 (compat r86-r95) | compat r108 |
| packed_pixels *_format_stencil_index x9 | 8554b98 (tess r46/r47) | tess r75 (59d72636) |
| draw_elements_base_vertex_tests.AEP_shader_stages | d2df080b (tess r66) | tess r75 |
| compute_shader.resources-max | dd77727 (geom r88/r89) | open (geom) |
| gl_spirv.spirv_modules_state_queries_test | 323c0cb4 (geom r130-r134) | open (geom) |

Verification (each stage separately on 2bccda0c): headless suite 0 failures; official-args caselist of the 42: each worker's cases Pass, the rest unchanged; KHR-GL46.es_31_compatibility unchanged. shader_storage_buffer_object.basic-binding is order-flaky on both builds.

New regression found, open: compat r107 (2f95cadc) broke 38 KHR-GL46.es_31_compatibility.sample_variables *samples_0* cases (mask x rgba8/8i/8ui/32f, position fixed/non-fixed): 317 Pass on 1cbdd1ce and on b37b3ceb (tess r74 only), 279 Pass with r107. Routed to compat.

Upstream-candidate bug #5 (from rsvga), open: an SM1-3-translated vertex shader draws nothing whenever a depth attachment is bound, even with the depth test off; VGPU10-style shaders are unaffected. SPIR-V repro pair saved in ~/MGL-cts/sup/rsvga-bug5/. Routed to compat.

## Full official-equivalent CTS run finished 2026-11-13 ~05:30 EST (baseline-1cbdd1ce, started 06:20 EST Nov 12)
Totals vs the Nov 1 run (r7edc431): Pass 97,999 (was 95,428), Fail 661 (was 2,339), NotSupported 24,114 (was 25,017), InternalError 4 (4), QualityWarning 0 (10), Crash 21 (was 1).

| group | Nov 1 (7edc431) | Nov 12-13 (1cbdd1ce) |
|---|---|---|
| gl45-es3 | 64 Fail / 1261 Pass | 26 Fail / 1299 Pass |
| gl45-es31 | 1120 Fail / 28771 Pass / 0 Crash | 334 Fail / 30453 Pass / 10 Crash |
| gl46-khr-single | 991 Fail / 4064 Pass / 1 Crash | 97 Fail / 4958 Pass / 1 Crash |
| gl46-main cfg-0 runs 0-3 | 38/43/37/38 Fail, 0 Crash | 47/48/49/49 Fail, 4/4/1/1 Crash |
| gl30/40/43/45-khr | small | small (gl43 5 Fail / 7 Pass) |

All 21 crashes, re-run singly on baseline-2f95cadc (gl46-port with c5cee355 + tess r74 + compat r107), now Pass:
- 6 KHR-GL46.framebuffer_blit SEGVs (cfg-0 runs 0-1): fixed by c5cee355.
- 4 KHR-GL46.texture_swizzle.smoke_access 90-min alarms and the 7 primitive_bounding_box vertex_tessellation_geometry_fragment hangs: fixed by r74 + r107 (default-texture 0x0x0 fallback loop).
- 3 dEQP-GL45-ES31 copy_image 90-min alarms: Pass in 2 s each. The ES31 part logs were 10-15 GB each of the same "emergency fallback" flood, so the alarms were the same slowdown, not a copy_image bug.
- KHR-Single-GL46.enhanced_layouts.uniform_block_member_invalid_offset_alignment_mat4_vertex 90-min alarm: Pass in 16 s.
gl46-main Fail rose by about 10 per run vs Nov 1 (not yet triaged per case). InternalError (direct_state_access.queries_functional) unchanged and still unrouted.

Results: ~/MGL-cts-new/official/r1cbdd1ce (65 GB, mostly flood logs).

## Merged 2026-11-13 03:55 EST: texture_swizzle flood + tess/geometry bounding-box hangs (b37b3ceb tess r74, 2f95cadc compat r107)
The user pre-approved both merges at 22:53 EST Nov 12 ("merge all patches after fixes"). No GL strings change; the ES-only EXT alias hunk carried on gl46/tess (GL_EXT_gpu_shader5 / GL_EXT_tessellation_shader / GL_EXT_texture_cube_map_array) was excluded and stays held.

Found by the full official CTS run (baseline-1cbdd1ce): KHR-GL46.texture_swizzle.smoke_access_* hit the 90-minute alarm (a 5.3 GB log of "Primary texture creation returned NULL ... emergency fallback texture (0x0x0)") and 7 dEQP-GL45-ES31.functional.primitive_bounding_box.*.vertex_tessellation_geometry_fragment.* cases hung. Both pass alone; they fail only after earlier cases in the same process. Bisect over baselines: first bad bd9ae9a5 (compat r103..r105), which made glBindTexture(target, 0) bind a real per-target default texture object.

Root cause: the CTS between-case state reset issues glTexImage{2,3}DMultisample(..., 0, 0, ...) into those default objects. texext_alloc_levels marked the zero-extent levels complete, so every later draw tried to create a 0x0x0 Metal texture, failed, and fell back; bindMTLTexture's fallback path never cleared dirty_bits, so the fallback was released and re-created on every draw (every tess draw re-walks the texture units).
- compat r107 (textures_ext.c, textures.c): zero-extent levels are empty/incomplete (GL 4.6 8.6.1) and the usable-level summary is recomputed (updateTexSummary now shared). Zero-extent multisample texImage stays legal (the CTS reset expects no error). Test test_compat_default_texture_degenerate.
- tess r74 (MGLRenderer.m): the fallback path clears dirty_bits like the primary path. Test test_vtg_hang (1200 fallback creations before, 1 after).

Verify: each GREEN alone and combined (0 FAILED, 817/0, 298/0, Tess 85 PASS, rc 0). Combined build: repro pair (api_errors then smoke_access) 2/2 Pass with 0 fallback lines in 11 s; the 7 vtg cases at the official 256x256 surface all Pass in 4-8 s; run.py texture_swizzle.* 501 Pass / 192 NotSupported / 4 Timeout / 0 Fail (baseline c5cee355: 497 / 192 / 8 / 0; the smoke Timeouts are load noise on the 25-s per-case limit); framebuffer_fetch 44/44; primitive_bounding_box.* 221 Pass / 6 Timeout, same as baseline. compat also reports KHR-GL46.clip_distance.coverage Pass (same cause).

New baseline: ~/MGL-cts/baseline-2f95cadc. Local main moved to gl46-port (not pushed).

## Merged 2026-11-12 20:27 EST: blit segfault on the default framebuffer (c5cee355)
The user approved at 19:38 EST (fix it directly) and 20:27 EST (merge). Files: framebuffer_ext.c, tests/headless/test_blit_neg.c. No GL strings change.

Found by the full official CTS run: KHR-GL46.framebuffer_blit.scissor_blit and multisampled_to_singlesampled_blit_{color,depth}_config_test segfaulted (gl46-main, every config). compat r99 (8504721d, Nov 5) added the GL 4.6 §18.3.1 depth/stencil format-match check to `mgl_fb_blit` and read `&readfbo->depth` / `&drawfbo->depth` unconditionally; both are NULL for the default (window) framebuffer, so any DEPTH/STENCIL blit to or from the window crashed (EXC_BAD_ACCESS at 0x194, framebuffer_ext.c:948). The check now runs only between two framebuffer objects, and a renderbuffer attachment without a renderbuffer object counts as a missing aspect. It reproduced only with the official window surface (64x64), not with agents/conformance/run.py's surface: segfault on 1cbdd1ce and 0929b5df, Pass on d00c78b.

Test: test_blit_neg `d_default_framebuffer_ds_blit_no_crash` (aborted before the fix). Verify: v-blitfix GREEN (0 FAILED, 817/0, 298/0, Tess 84 PASS, Xfb PASS, rc 0). Official-surface repro 3/3 Pass; ES31 negative_coverage blit_framebuffer 6/6 unchanged.

Still open from the same CTS run: texture_swizzle.smoke_access_idx_11_channel_idx_2 loops on failed texture creation (0x0x0 fallback) until the 90-min alarm (passed Nov 1); 7 primitive_bounding_box vertex_tessellation_geometry_fragment hangs (tess+GS); 3 ES31 copy_image and 1 enhanced_layouts process-alarm timeouts, to re-run singly.

New baseline: ~/MGL-cts/baseline-c5cee355.

## Merged 2026-11-12 19:18 EST: geom r146+r147, SPIR-V name independence (0929b5df)
The user approved at 10:43 EST (r146 brief, from the rsvga OpName finding) and at 19:18 EST (merge). Files: geometry.c, transform_feedback.c, mgl_ext_geom.h, tests/headless/test_geom_nonames.c, tests/headless/spv_nonames.h. No GL strings change.

OpName/OpMemberName are non-semantic, but the geometry generators built their GLSL/MSL from them, so a name-less ARB_gl_spirv module silently miscompiled (for example `floatBitsToUint(.x)` from an empty name). `spv_parse` now synthesizes stable internal names for unnamed variables, struct types and members (from %id or Location). The SPIRV-Cross decompiler and the transform-feedback resolver use the same names. They never reach the application: program-interface and transform-feedback varying queries report the original module's names (empty when it had none), and TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH follows them. r146 leaked the synthesized names into the xfb varying queries; r147 fixed that.

Test: test_geom_nonames (58 checks) — name-less modules render and capture byte-identically to their named twins, every reported name is empty, no `_mgl_` name leaks, and the named twin still reports its real names. Verify: v-r147 GREEN (0 FAILED, 817/0 x6, 298/0, Tess 84 PASS, Xfb PASS, rc 0). CTS vs b9ef44f1 over 9 sets (geometry_shader 136/136, transform_feedback 21/21, overflow_query 13/13, gl_spirv 7/9, draw_indirect 70/70, draw_indirect_43 2/2, draw_elements_base_vertex 10/1/1NS, shader_draw_parameters 13/13, indirect_parameters 3/3): no change. glmark2 swap: base 3742, cand 3756 (+0.4%, noise).

Follow-ups (not in this merge): geometry.c blanks reported names by prefix (`_mgl_v/_mgl_in/_mgl_out/_mgl_t`) while the vertex path uses the exact synthesized list, so an app name that happens to start with one of those prefixes would report empty; and deleting a program can leave its GL name on a stale transform-feedback table slot (reproduced standalone by geom).

New baseline: ~/MGL-cts/baseline-0929b5df.

## Merged 2026-11-12 11:33 EST: patch 0004, GS xfb CPU-pack offsets and CPU readback sync (b9ef44f1)
The user approved at 11:09 EST. Source: rsvga's patches/0004-geometry-xfb-pack-offsets-and-readback.fixed.patch, plus one supervisor refinement and a regression test. Files: geometry.c, draw_ext.c, mgl_ext_buffers.h, tests/headless/test_geom_cpuxfb.c. No GL strings change.

- geometry.c, CPU pack of geometry-shader transform feedback (used with MGL_GEOM_CPU_XFB or when the GPU pack can't run): every vertex already lands at write_offset + (prim*per + j)*stride, yet write_offset also advanced by per*stride after each primitive, so every primitive after the first was written at twice its offset and the next draw started past a doubled end. Offsets now advance once per draw. Refinement: advance past the highest primitive index that landed rather than by the landed count, so an index gap can never let the next draw overwrite the tail.
- draw_ext.c `mgl_buffers_prepare_cpu_read`: a GPU-written managed buffer was not synchronized for CPU reads once its write serial had completed. A per-buffer cpu_serial now tracks what the CPU has synced. On Apple Silicon this only waits on an already-completed serial; it fixes managed buffers on Intel/AMD hosts.

Test: test_geom_cpuxfb (strip and points GS, several primitives per draw, two draws into one buffer, PRIMITIVES_WRITTEN, untouched tail) fails on 110ade8d with the doubled offsets and passes 132/0 with the change. Verify: v-p0004 GREEN (0 FAILED, 817/0 x6, 298/0, Tess 84 PASS, Xfb PASS, headless rc 0).

New baseline: ~/MGL-cts/baseline-b9ef44f1.

## Merged 2026-11-12 11:07 EST: xfb queries hook without dlsym(RTLD_DEFAULT) (110ade8d)
The user approved at 10:43 EST, after the rsvga analysis. Files: geometry.c, transform_feedback.c. No GL strings change.

The geometry-shader xfb capture (both sites in geometry.c) and transform_feedback.c `xfb_queries_note()` looked up `mgl_queries_note_xfb_primitives` with `dlsym(RTLD_DEFAULT, ...)`. That only resolves when libmgl is loaded RTLD_GLOBAL; a host that dlopens it RTLD_LOCAL (rsvga's default) silently lost GL_PRIMITIVES_GENERATED / GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN counts. Both now reference the symbol directly (same library). rsvga can drop its RTLD_GLOBAL workaround.

Verify: v-dlsym GREEN (0 FAILED, 817/0 x6, 298/0, Tess 84 PASS, Xfb PASS, headless rc 0).

New baseline: ~/MGL-cts/baseline-110ade8d.

## Merged 2026-11-12 08:53 EST: x86_64 build fix, portable spin-wait hint (ce9263cd)
The user approved at 08:31 EST, after reviewing the rsvga agent's universal-build notes. File: MGLRenderer+CompatCS.m only. No GL strings change.

MGLRenderer+CompatCS.m called `__builtin_arm_yield()` unguarded in its two spin loops, so gl46-port did not compile for x86_64 ("use of unknown builtin"; the rsvga build-mgl.sh worked around it with a sed swap). A static inline `cs_cpu_relax()` now emits YIELD on arm64, PAUSE on x86_64 and nothing elsewhere; arm64 code is unchanged.

Verify: x86_64 `-fsyntax-only` over all 84 MGL/src .c/.m files is clean (the ARC files with `-fobjc-arc`, as the Makefile builds them), so this was the only architecture-specific construct in MGL's own sources; the bundled externals (glslang, SPIRV-Cross, SPIRV-Tools) still need a native x86_64 build. arm64 full headless gate green on 451d35d0 (0 FAIL, 817/0, 298/0, Tess/Xfb PASS).

Notes passed to the rsvga side (08:30 EST): the IOSurface path is tested on AGX (test_iosurface_tex 44/0 on the M4 Pro) but not on the Intel/AMD side or with BGRA8; patches 0001–0003 are already in gl46-port (45309315, 9800bc2a), so build-mgl.sh must not re-apply them, and the original 0003 is buggy; match the uv Python path on the host arch rather than a glob.

New baseline: ~/MGL-cts/baseline-ce9263cd.

## Merged 2026-11-12 08:10 EST: GL_ARB_indirect_parameters (0cb9214b)
The user asked at 07:50 EST whether the instanced/indirect entry points are exported (all core ones are) and approved adding the two missing pieces at 07:52 EST. Files: get.c, gl_core.c, tests test_buffers.c and test_entry_ext.c.

- **New GL string:** GL_ARB_indirect_parameters. Everything it covers was already implemented as GL 4.6 core: glMultiDraw{Arrays,Elements}IndirectCount, the GL_PARAMETER_BUFFER target and the GL_PARAMETER_BUFFER_BINDING query.
- **New exports:** glMultiDrawArraysIndirectCountARB and glMultiDrawElementsIndirectCountARB, forwarding to the core entry points (loaders such as glcts fetch the ARB names once the extension is listed).

Verify: green from scratch on 14eb7d68 (0 FAIL, 817/0, 298/0, Tess/Xfb PASS; test_entry_ext and test_buffers include the new checks). CTS on the verify build with the extension advertised: indirect_parameters_tests 3/3, draw_indirect 70/70, shader_draw_parameters_tests 13/13, the same as base 45309315.

New baseline: ~/MGL-cts/baseline-0cb9214b.

## Merged 2026-11-12 07:48 EST: patch 0003, IOSurface-backed textures, corrected (9800bc2a)
Source: patches/0003-iosurface-backed-textures.patch (rsvga local build). The user asked at 07:28 EST to inspect it, verify its claim and merge it with any fixes. Corrected version: patches/0003-iosurface-backed-textures.fixed.patch. Files: MGLRenderer+TextureData.m, mgl_ext_texdata.h, textures.c, new tests/headless/test_iosurface_tex.c. No GL strings change.

What it adds: `void *mglTextureIOSurface2D(GLuint texture, GLenum internalformat, GLsizei w, GLsizei h)` gives a GL_TEXTURE_2D single-level immutable storage on a fresh IOSurface. The realized MTLTexture wraps the surface (shared storage), so a CoreAnimation layer can show the texture as the GL draws it, with no readback; the caller releases the returned IOSurfaceRef and should glFlush/glFinish before compositing. Storage RGBA8/BGRA8 unorm only. The content copy that texdataRealize does on a shape change is factored into texdataInstallTexture.

The submitted patch's claim did not hold:
- "Content carries across the storage change" was false: the levels were redefined first, and defineLevel clears the GPU-resident bit and frees the CPU shadow, so the surface came up zeroed. Now level 0 is kept when it already has the requested format and size.
- Refused formats were refused after the texture had been redefined (GL_RGBA16F left it immutable RGBA16F with the content gone). The format is now checked first.
- Realization is finished like texdataRealize; the surface texture gets the ordinary usage flags (views, image stores) and the texparam/texext descriptor hooks.

Verify: test_iosurface_tex 44/0 on the corrected patch and 9 failures on the submitted one; full headless gate green on 85479f97 (0 FAIL, 817/0, 298/0, Tess/Xfb PASS). CTS and glmark2 do not call the new entry point, and ordinary texture realization is the same code moved into a helper, so they were not re-run (a full official-equivalent CTS run on 1cbdd1ce is in progress).

New baseline: ~/MGL-cts/baseline-9800bc2a.

## Merged 2026-11-12 05:14 EST: compat r106 (1cbdd1ce)
This squashes gl46/compat 86c11c43 (READY r106) over 87cb342f: compat_glsl.c (+12/−2), compat_vb.c (+9) and two new headless tests. The report commit and test_scratch_xfbcompat.c (a scratch repro) are excluded. No GL strings change. The user approved the merge at 04:54 EST.

What merged: two regressions found by compat's compat-as-superset audit, which runs CTS on a compatibility-profile context (MGL_CTS_COMPAT=1, an env-gated hook compat added to the shared CTS build in ~/MGL-cts-new; default behaviour unchanged).
- A shader with no compatibility-only tokens is plain GLSL whatever its #version. It used to go through the fixed-function translator just for a #version below 330, which upgraded it to 460 and injected mgl* varyings and gl_PerVertex differences, breaking multi-stage links on a compat context. KHR-GL46.transform_feedback on a compat context: 8/21 → 21/21.
- bind_ffp yields to a bound program pipeline (GL 4.6 §7.3.1): glUseProgram(0) + glBindProgramPipeline rendered through the fixed-function program instead of the pipeline's stages (direct_state_access.program_pipelines_functional on a compat context).

Verify: green on 45309315 and again on 87cb342f (0 FAIL, 817/0, 298/0, Tess/Xfb PASS). compat's own gates on its tip: headless 1034 PASS / 0 FAIL, multi_bind 17/17, framebuffer_fetch 44/44, transform_feedback 21/21, synchronization 104/104, es_31_compatibility 317 Pass + 38 NotSupported.

glmark2 order swap (full default run, 04:22 EST): base 4477, cand 4514, cand 4526, base 4477 (+1.0%); no MGL errors.

Open (compat's cross-domain finding): KHR-GL46.direct_state_access.queries_functional ends InternalError with no log text on core and compat contexts, on gl46-port 323c0cb4 and 45309315 too — pre-existing, likely a missing query-buffer entry point.

New baseline: ~/MGL-cts/baseline-1cbdd1ce.

## Merged 2026-11-12 04:53 EST: geom r143–r145 (9b8587b2)
This squashes gl46/geom 7bb5aab9 (READY r145) over 45309315: MGLRenderer+Geometry.m (+66/−20 net) and tests/headless/test_geom_stageio.c (+156). The report commits are excluded. No GL strings change. The user approved the merge at 04:52 EST.

What merged: the baseInstance > 0 capture fix for repeated element draws through a geometry shader.
- The prep kernel's staging-command binding ignored the pool offset (gPrepOff). Only the first pool range worked; once the pool rotated, the kernel read a stale command (baseInstance 0). Latent since r139.
- **Correction to the r142 entry below:** Metal's MTLDrawIndexedPrimitivesIndirectArguments is five words — {indexCount, instanceCount, indexStart, baseVertex, baseInstance} = prep[7..11]. r142 wrote the GL baseInstance into prep[10], the baseVertex slot, so the capture drew indices shifted by baseInstance and captured nothing. Both prep-kernel paths now write indexStart 0, baseVertex 0 (the capture list is absolute) and baseInstance in word 11. geom's r144 had attributed this to "Metal's baseInstance being unreliable in indirect arguments" and worked around it with a second explicit-argument draw path; the supervisor sent it back, and r145 removed the workaround and the claim.
- Tests: consecutive baseInstance = 2 draws on the CPU and GPU assembly paths, glDrawElementsIndirect with baseInstance = 2 four times, and baseVertex ≠ 0.

Verify: green from scratch on 45309315 (0 FAIL, 817/0, 298/0, Tess/Xfb PASS, test_geom ×5 clean).

CTS on the verify build against base45309315 (supervisor runner, 8 geometry groups), identical case by case: geometry_shader 136/136, draw_indirect 70/70, draw_indirect_43 2/2, draw_elements_base_vertex 10 Pass / 1 Fail / 1 NotSupported (same on both sides), shader_draw_parameters 13/13, indirect_parameters 3/3, transform_feedback 21/21, transform_feedback_overflow_query_ARB 13/13.

glmark2 order swap (full default run, 03:39 EST, load about 5–6): base 4062, cand 4291, cand 4603, base 4231 (means 4146 vs 4447, +7.2%); no MGL errors. Both sides are below the 6000 floor, as was the baseline.

Ops: the Mac reboot on Nov 11 wiped /tmp, so the supervisor helpers were rebuilt in ~/MGL-cts/sup/ (symlinked to the old /tmp names). Also on Nov 12: tess r73 (verification-only: CTS re-sweep all Pass, nothing to merge besides the held EXT aliases); compat r106 READY (compat-context transform_feedback and program-pipeline fixes) is in verification. 01:52–02:23 EST compat rebuilt the shared CTS with unbounded ninja (load 22); it now uses -j4.

New baseline: ~/MGL-cts/baseline-9b8587b2.

## Merged 2026-11-11 08:59 EST: patches 0001 + 0002, corrected (45309315)
Source: patches/0001-x86_64-pointer-plausibility-bound.patch and patches/0002-render-encoder-bind-separate-images-and-samplers.patch (rsvga local build). The user asked to check, verify, correct and apply them at 08:49 EST. Files: MGLRenderer.m, plus the new tests/headless/test_sep_imgsamp.c and spv_sepimg.h. No GL strings change.

0001: on x86_64 macOS, heap and Metal pointers sit around 0x7f.., above the 16 TB plausibility bound, so every draw was treated as state corruption and draws ended up disabled. Correction: x86_64 now uses 0x800000000000 (top of the 47-bit user space) rather than the 52-bit ASan value. ASan and arm64 are unchanged; macro selection was checked with the x86_64 preprocessor.

0002: Vulkan-style separate images and samplers (SEPARATE_IMAGE / SEPARATE_SAMPLERS) were never bound, so samples read zeros. A shared bindSeparateImagesSamplersStage now binds them for the vertex/TES, fragment and compute stages. The submitted patch was render-only, and it said compute already handled them; compute did not. Other corrections: images are realized (bindMTLTexture) rather than skipped; incomplete textures get the typed fallback; the sRGB-decode view is applied; per-element units come from mgl_res_elem_unit. Samplers: no hard-coded 2D target; no NULL texture passed to texparamSamplerForUnit; pairing with the stage's first separate image when the sampler's unit has no texture; array_size honoured; slots ≥16 skipped. ARB_gl_spirv requires combined image-samplers, so this serves translated Vulkan shaders (rsvga) only.

Verify: fresh worktree on 7bea5e94, make lib glx nsgl, headless suite 201 tests, 0 failures (817/0 and 298/0 present). New test_sep_imgsamp: 20/0 here, 6 failures (all zeros) against the 7bea5e94 library. Geometry CTS and glmark2 were not run: the Mac restart at about 00:00 EST cleared the /tmp runners. GL CTS and glmark2 use only combined samplers, so neither exercises the new path. Baseline: ~/MGL-cts/baseline-45309315.

## Merged 2026-11-10 23:44 EST: geom r142 (7bea5e94)
This squashes gl46/geom 3f78574b over ac6eb1f7: MGLRenderer+Geometry.m only, +6/−2. The report commits are excluded. No GL strings change. The user approved the merge at 23:43 EST.

What merged: Metal's indexed indirect-draw arguments are {count, instances, baseVertex, baseInstance} = prep[7..10]. The word labels had been off by one since r133, so word 10, the real baseInstance, was always 0. It now carries the GL baseInstance. Draws with baseInstance = 0 are byte-identical to before.

Verify: green from scratch on ac6eb1f7 (0 FAIL, test_geom 817/0 ×11, 298/0 ×4, TessCtsVerbatim/Xfb PASS).

CTS on the verify build against baseline-ac6eb1f7, absolute counts, identical on both sides: transform_feedback 20, geometry_shader 136, ES31 geometry_shading 203, tessellation_geometry_interaction 28, ES31 tessellation 395 (all Pass); primitive_bounding_box.wide_lines 19 Pass / 20 NotSupported / 1 Timeout (the same fbo_bbox_smaller case on both sides).

glmark2 order swap (23:23 EST, load about 5–7): scores 686 / 710 / 693 / 685. No scene is lower in both orders; no MGL errors in any log.

New baseline: ~/MGL-cts/baseline-7bea5e94; the 12 ab_*.sh scripts point at it.

## Merged 2026-11-10 20:29 EST: geom r139 (ac6eb1f7)
This squashes gl46/geom dc775187 over 5ef89e38: MGLRenderer+Geometry.m only, +18/−7. The report commits are excluded. No GL strings change. The user approved the merge at 20:28 EST.

What merged: the plumbing half of geom's baseInstance fix plan. The staging command for direct draws gains a fifth word carrying baseinstance (the 16-word staging region had word 4 free). The serial prep kernel now parses it from every command layout and publishes it as prep[11], the capture draw's baseInstance, and the prepc stage threads it through. The remaining half of the fix is still in progress on gl46/geom.

Verify: green from scratch on 5ef89e38 (0 FAIL, test_geom 817/0 ×11, 298/0 ×4, TessCtsVerbatim/Xfb PASS).

CTS on the verify build against baseline-5ef89e38, absolute counts, identical on both sides: transform_feedback 20, geometry_shader 136, ES31 geometry_shading 203, tessellation_geometry_interaction 28, ES31 tessellation 395 (all Pass); primitive_bounding_box.wide_lines 19 Pass / 20 NotSupported / 1 Timeout (the same fbo_bbox_smaller case on both sides).

glmark2 order swap (20:23 EST, load about 5–6): scores 681 / 674 / 680 / 673. Every scene is within about 3% in both orders; no MGL errors in any log.

New baseline: ~/MGL-cts/baseline-ac6eb1f7; the 12 ab_*.sh scripts point at it.

## Merged 2026-11-10 17:10 EST: compat r103–r105 (bd9ae9a5) + geom r135–r137 tests (5ef89e38)
The user said "merge" at 17:08 EST, approving both: compat r105 merges on its earlier CTS results, without a CTS re-run on 323c0cb4, and geom's tests-only commits merge with it.

**compat r103–r105 (bd9ae9a5)** is the verified staged diff of READY r105 (32bceaf8), which is r104 plus the r104b glmark2 fix, merged with gl46-port 323c0cb4. It touches 33 files.
- New GL strings: GL_EXT_shader_framebuffer_fetch and GL_EXT_texture_sRGB_decode, both CTS-proven. GL_EXT_color_buffer_float and GL_EXT_color_buffer_half_float are also exposed; the GL mustpass has no cases for them, and the user approved them at 11:26.
- Fixes: framebuffer fetch, sRGB-decode texture views (SKIP_DECODE), blits honour GL_FRAMEBUFFER_SRGB, readPixels float/integer type rules, multi_bind, SSBO aliasing, ES31 negative coverage and synchronization.
- r104b: a texture unit made active but never bound (or bound to name 0) held the DEFAULT texture object, which has zero levels, and MGL tried to realize it as a Metal texture (the glmark2 shadow/jellyfish "MGLRenderer.m:1237" assert). Such units are now skipped, and sampler-unit completeness is checked before binding; an incomplete texture binds the black fallback.
- Verify on 323c0cb4: green (0 FAIL, 817/0 ×11, 298/0 ×5).
- CTS (r104 against base b5940dfe): 0 regressions, +116 (negative_coverage +39, synchronization +5, framebuffer_fetch 44 NotSupported → Pass, srgb_decode state_query 28 Fail → Pass).
- glmark2 (r104b against 13aea814): 398 / 394 / 403 / 402, every scene within 5%, 0 "MGL ERROR" lines.

**geom r135–r137 tests (5ef89e38):** test_geom_stageio.c +238 lines, no driver changes. They cover baseVertex on direct element draws through a geometry shader (absolute gl_VertexID in raster and xfb capture) and baseInstance on arrays-indirect and element-indirect draws (absolute gl_InstanceID).

Combined-tree verify of 5ef89e38: green (0 FAIL, test_geom 817/0 ×11, 298/0 ×4, TessCtsVerbatim/Xfb PASS, test_geom_stageio all PASS).

New baseline: ~/MGL-cts/baseline-5ef89e38; the 12 ab_*.sh scripts point at it. compat and tess are idle with their work merged; geom continues.

## Merged 2026-11-10 14:08 EST: geom r130–r134 (323c0cb4)
This squashes gl46/geom over 13aea814 as READY r134 (96104830), which is the r133 code merged with gl46-port 13aea814. The report commits are excluded. geom's later r135 commit (2deddae0, 145 lines of tests only) is not included; it comes with geom's next READY. No GL strings change. The user approved the merge at 14:07 EST.

What merged:
- r133: baseVertex on element-indirect draws with a geometry shader bound. The prep kernels' primitive and capture lists are absolute, so the indirect draw arguments now publish baseVertex 0; before, baseVertex was added twice in the raster record fetch.
- Stage-IO, transform-feedback and program-interface fixes in geometry.c, program.c, program_ext.c, transform_feedback.c, mgl_ext_geom.h and MGLRenderer+Geometry.m.
- Tests: test_geom_stageio.c (+3550 lines) and test_geom_aoa64.c.

Verify: green from scratch on 13aea814 (0 FAIL, test_geom 817/0 ×11, 298/0 ×5, TessCtsVerbatim/Xfb PASS).

CTS on the verify build against baseline-13aea814, absolute counts, identical on both sides: transform_feedback 20, geometry_shader 136, ES31 geometry_shading 203, tessellation_geometry_interaction 28, ES31 tessellation 395 (all Pass); primitive_bounding_box.wide_lines 19 Pass / 20 NotSupported / 1 Timeout. The timeout is the same fbo_bbox_smaller case on both sides; it arrived with the tess r70–r72 merge and passes when run alone.

glmark2 order swap (base, cand, cand, base; 13:40 EST, load about 4): scores 420 / 397 / 397 / 396. In the second pair every scene is within 2.5%; no MGL errors in any log.

Also on Nov 10: **compat r104b** (825ceec1, on 13aea814) fixes the r104 hold. The assert came from the DEFAULT texture object, which has zero levels: a unit made active but never bound held it, and MGL tried to realize it as a Metal texture. r104b skips such units and checks sampler-unit completeness before binding (an incomplete texture binds the black fallback). Verify green on 13aea814; glmark2 swap clean: 0 "MGL ERROR" lines in all four runs, scores 398 / 394 / 403 / 402. It must be re-verified on 323c0cb4 before it can merge.

New baseline: ~/MGL-cts/baseline-323c0cb4; the 12 ab_*.sh scripts point at it.

## Merged 2026-11-10 11:28 EST: tess r70–r72 (13aea814)
This squashes gl46/tess b5940dfe..b0f98f15 (r70–r72) onto b5940dfe. The report commit (56bbf8be) is excluded, and so is the [DROPPABLE] ES-only EXT-alias commit (7aad6ab2, GL_EXT_gpu_shader5 / GL_EXT_tessellation_shader / GL_EXT_texture_cube_map_array), which stays held. No GL strings change. The user approved the merge at 11:26 EST.

What merged:
- r70: aliased wide lines. Vertex-only pipelines get a GPU quad expansion of the VS kernel's output records, tessellated isolines get a forced TES capture with a generated TES replay, and geometry-stage line output replays the expansion kernel's records through the geom raster pass. ALIASED_LINE_WIDTH_RANGE is now [1,64] (was [1,1]).
- r71: geometry-stage wide lines un-gated (ring reserve/emit split), multi-segment layout fix, instanced wide isolines, culling and query invariance checks.
- r72: the vtg CTS black-viewport triage resolved; TES transform feedback of matrices and the remaining enhanced_layouts TES-xfb shapes verified and guarded.
- Tests: test_tess.c (+1528 lines), new test_wlgeom_stress.c.

Verify: green from scratch on b5940dfe (0 FAIL, test_geom 817/0 ×11, 298/0 ×4, TessCtsVerbatim/Xfb PASS).

CTS on the verify build against baseline-b5940dfe, absolute counts: transform_feedback 20, geometry_shader 136, ES31 geometry_shading 203, tessellation_geometry_interaction 28, ES31 tessellation 395, gpu_shader_fp64 659, enhanced_layouts 1890, ES31 PIQ 4599, KHR PIQ/SSO/subroutine 71, ES31 separate_shader 183 and program_binary 7 are identical on both sides, except fp64 +1 (a base-side runner timeout). ES31 primitive_bounding_box.wide_lines (40): base 40 NotSupported → 19 Pass / 20 NotSupported / 1 Timeout. The timeout (tessellation_set_per_draw.vertex_tessellation_geometry_fragment.fbo_bbox_smaller) passed 2/2 run alone, so it was load. 0 Pass→notPass.

glmark2 order swap (base, cand, cand, base; 11:03 EST, load ~7): scores 457 / 404 / 408 / 402. Every scene is within 5% of the adjacent base run; no MGL errors in the logs. (Absolute scores are low because the user's non-MGL ptolemy-engine processes kept the load at 7–10 all morning.)

Other candidates in the same round (Nov 10):
- **compat r104 (28c87762) HELD.** CTS: 0 regressions, +116 (negative_coverage +39, synchronization +5, framebuffer_fetch 44 NotSupported → Pass, srgb_decode state_query 28 Fail → Pass). It adds GL_EXT_texture_sRGB_decode and GL_EXT_shader_framebuffer_fetch (CTS-proven) plus GL_EXT_color_buffer_float and GL_EXT_color_buffer_half_float, which have no GL CTS cases; the user approved exposing those two at 11:26. glmark2 performance was flat (404 vs 415), but both candidate runs logged `MGL ERROR: Assertion hit in MGLRenderer.m at line 1237` (texture creation returned NULL, 0x0x0 fallback) in shadow and jellyfish, never on base. compat was relaunched to root-cause it; the likely trigger is the new compatSRGBDecodeViewForTex calls.
- **geom r129 (2c4a8e85):** CTS 0 regressions (fp64 +1). glmark2 inconclusive: run at load 9–10, with base-to-base noise as large as the gap (score 412 vs 443). Superseded by r133 (5c7ec823), which fixes baseVertex on element-indirect draws with a GS bound. geom is merging 13aea814; its next READY gets a verify, the geometry CTS sets (user OK 11:26) and a glmark2 swap on the new base.

New baseline: ~/MGL-cts/baseline-13aea814; the 12 ab_*.sh scripts point at it.

## Merged 2026-11-06 07:00 EST: geom r108 + r109 (b5940dfe)
This squashes seven commits from gl46/geom onto 9b049175: 6c5701e3, 79bd7608, 1ea61c25, 94561a3d, 4f4481f4, 17544c02 and 8ad6d9c2. The report commits are excluded. The user approved the merge at 06:59 EST.

What merged:
- r108: arrays-of-arrays stage IO. Variables such as `flat out vec4 v[2][3]` and `out Inner o[2][3]`, including the GS/TES variants, previously failed the whole link. They are now split per leaf with a private shadow, for both render and transform feedback. This also covers the geometry-emulation record, GS in/out, TES output, and SEPARATE_ATTRIBS capture of degenerate leaves.
- r109: arrays-of-arrays with f64 leaves (VS/FS render, GS and xfb). It also adds 32-bit-int-typed index constants, dynamic-index fragment-shader reads, and three nesting levels.

Verify: green from scratch on 9b049175 (0 FAIL, test_geom 817/0 ×10, 298/0 ×3).

CTS on the 9b049175 + r108/r109 verify build against baseline-9b049175, absolute counts:

| Group | base Pass | r108/r109 Pass |
|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 | 20 |
| KHR-GL46 geometry_shader | 136 | 136 |
| ES31 geometry_shading | 203 | 203 |
| ES31 tessellation_geometry_interaction | 28 | 28 |
| ES31 tessellation | 395 | 395 |
| KHR-GL46 gpu_shader_fp64 | 659 | 659 |
| KHR-Single-GL46 enhanced_layouts xfb_* + ssb_member_overlapping | 1890 | 1890 |
| ES31 program_interface_query | 4599 | 4599 |
| KHR-GL46 program_interface_query + separate_shader_objects + shader_subroutine | 71 | 71 |
| ES31 separate_shader | 183 | 183 |

There were no failures on either side.

glmark2 order swap, run twice. Scores are base / candidate:

| Run | base first | candidate first |
|---|---|---|
| 1 | 8507 / 8435 | 7974 / 8075 |
| 2 | 8515 / 8483 | 8358 / 8428 |

The result is flat. In both runs the third slot scored low on shadow (9819 and 10057, against about 11000 elsewhere). The same dip appeared in every swap this session, whichever library ran in that slot, so it is a position effect.

Still held: compat r102 (1a5a37bb). It advertises GL_EXT_shader_framebuffer_fetch, and dEQP-GL45-ES31.functional.shaders.framebuffer_fetch.basic.texel_fetch fails deterministically on it. The rest of r102 is equal or better: negcov 1453P/39NS → 1492P, synchronization 99P/5F → 104P, and the ES31 srgb_decode state_query cases go from 28F to 28P.

## Merged 2026-11-06 00:58 EST: tess r67–r69 (9b049175)
This squashes five commits from gl46/tess onto 24911baa: 7b928b8f, 412ce0f9, e881396c, e9a9085f and b98c3c84. The [DROPPABLE] ES-only EXT-alias commit (4208fd6c) is **not** included and stays held. The user approved this merge at 20:17 EST on Nov 5.

What merged:
- r67: link-time stage-interface matching, per GL 4.6 §7.4.1.
- r68: glslang cache tombstones, and geometry-import matrix pairing.
- r69: test_tess_dyncp carries its TessCtsVerbatim sources in-tree, so it no longer reads /tmp.

Verify: green from scratch on 24911baa. 0 FAIL, TessCtsVerbatim/Xfb PASS, test_geom 817/0 ×10, 298/0 ×3.

CTS on the 24911baa + r67–r69 verify build against baseline-24911baa, absolute counts:

| Group | base Pass / Fail | r67–r69 Pass / Fail |
|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 / 0 | 20 / 0 |
| KHR-GL46 geometry_shader | 136 / 0 | 136 / 0 |
| ES31 geometry_shading | 203 / 0 | 203 / 0 |
| ES31 tessellation_geometry_interaction | 28 / 0 | 28 / 0 |
| ES31 tessellation | 395 / 0 | 394 / 0 + 1 runner timeout* |
| KHR-GL46 gpu_shader_fp64 | 659 / 0 | 659 / 0 |
| KHR-Single-GL46 enhanced_layouts xfb_* + ssb_member_overlapping | 1890 / 0 | 1890 / 0 |
| ES31 program_interface_query | 4599 / 0 | 4599 / 0 |
| KHR-GL46 program_interface_query + separate_shader_objects + shader_subroutine | 71 / 0 | 71 / 0 |
| ES31 separate_shader | 183 / 0 | 183 / 0 |

\* `user_defined_io.per_patch_block_array.vertex_io_array_size_implicit.quads` was cut off by the runner's 900 s per-part alarm. Run in isolation it passes twice on each library.

glmark2 order swap at 00:47 EST, with the machine quiet:

| Scene | base→cand: base | cand | cand→base: cand | base |
|---|---|---|---|---|
| build | 14961 | 14101 | 14007 | 13901 |
| phong | 14488 | 13901 | 13743 | 13521 |
| blur | 4254 | 4262 | 4248 | 3695 |
| desktop-shadow | 3779 | 3829 | 3838 | 3761 |
| shadow | 10531 | 10616 | 10781 | 10498 |
| ideas | 2813 | 2746 | 2809 | 2828 |
| jellyfish | 11926 | 12527 | 11834 | 11803 |
| gl41-geometry | 2960 | 2928 | 2937 | 2882 |
| **score** | 8213 | 8112 | 8023 | 7860 |

Only ideas is lower in both orders, by 2.4% and 0.7%, which is within noise.

Pending: compat r101 (2e8445ac) fixes the r100 ssbo blocker and is being verified on 9b049175. geom r108 has a READY header, but its worker has not finished yet.

## Merged 2026-11-06 00:10 EST: geom r106 + r107 (24911baa)
This squashes four code commits from gl46/geom onto 8504721d: cc2ec0d5, 55293d04, 3d163e96 and c51f9241. The report commits were left out. The user approved it at 20:17 EST on Nov 5, conditional on the checks passing.

What merged:
- r106: the fp64 rw_access step re-walks access chains it had suppressed. Arrays of matrices in stage-IO structs are lowered. test_clipdist_spv regenerates its SPIR-V inputs, since the reboot wiped /tmp.
- r107a: stage-IO struct members written or read through functions now work (callee access chains are rewritten).
- r107b: arrays of structs in stage IO now work for transform-feedback capture and geometry-shader consistency.

Verify: green from scratch on d2df080b and again on 8504721d (0 FAIL, test_geom 817/0 ×10, 298/0 ×3). The only failures were TessCtsVerbatim/Xfb, which read reboot-wiped /tmp sources; tess r69 fixes that.

CTS on the 8504721d + r106/r107 verify build against baseline-8504721d, absolute counts:

| Group | cases | base Pass / Fail | r106/r107 Pass / Fail |
|---|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 | 20 / 0 | 20 / 0 |
| KHR-GL46 geometry_shader | 136 | 136 / 0 | 136 / 0 |
| ES31 geometry_shading | 203 | 203 / 0 | 203 / 0 |
| ES31 tessellation_geometry_interaction | 28 | 28 / 0 | 28 / 0 |
| ES31 tessellation | 395 | 395 / 0 | 395 / 0 |
| KHR-GL46 gpu_shader_fp64 | 659 | 659 / 0 | 659 / 0 |
| KHR-Single-GL46 enhanced_layouts xfb_* + ssb_member_overlapping | 1890 | 1888 / 2 | 1890 / 0 |

The two enhanced_layouts cases that went from Fail to Pass are xfb_struct_explicit_location_vertex and xfb_struct_explicit_location_tessellation_evaluation.

glmark2 order swap, run on a quiet machine at 00:02 EST:

| Scene | base→cand: base | cand | cand→base: cand | base |
|---|---|---|---|---|
| build | 15364 | 15526 | 14551 | 13474 |
| phong | 15188 | 15316 | 15188 | 12857 |
| blur | 4400 | 4291 | 4385 | 4200 |
| desktop-shadow | 3710 | 3861 | 3877 | 3869 |
| shadow | 11281 | 11138 | 11069 | 10938 |
| ideas | 2903 | 2876 | 2805 | 2497 |
| jellyfish | 12766 | 12817 | 11650 | 12673 |
| gl41-geometry | 2936 | 2937 | 2345 | 2573 |
| **score** | 8567 | 8594 | 8232 | 7884 |

No scene is lower in both orders.

Also on Nov 6: compat's r100 READY (eeb9f9ae) is **held**. dEQP-GL45-ES31.functional.ssbo.layout.random.all_shared_buffer.35 goes from Pass to Fail every time: isolated runs gave base Pass ×2 and r100 Fail ×2. The rest of r100 improves things, including synchronization 99P/5F → 104P. Next: tess r67–r69, which is being re-verified on 24911baa.

## Merged 2026-11-05 20:20 EST: compat r99 (8504721d)
Squash of compat r99 onto d2df080b (18 files; report and helper files excluded). The user approved it at 20:17 EST.

What merged: ES 3.1 image-atomic sidecar sync; SSBO and compute buffer-map fixes; blit format-match fixes.

Verify: green from scratch on d2df080b (0 FAIL, test_geom 817/0 ×10, 298/0 ×3).

CTS on the d2df080b + r99 verify build against baseline-d2df080b, absolute counts:

| Group | base Pass / Fail / NS / other | r99 Pass / Fail / NS / other |
|---|---|---|
| negative coverage | 1450 / 1 / 39 / 2 QW | 1453 / 0 / 39 / 0 |
| ssbo | 2028 / 32 / 1 / 0 | 2060 / 0 / 1 / 0 |
| compute | 191 / 4 / 0 / 0 | 195 / 0 / 0 / 0 |
| sync | 93 / 11 / 0 / 0 | 99 / 5 / 0 / 0 |
| es_31_compatibility | 314 / 3 / 38 / 0 | 317 / 0 / 38 / 0 |
| ES31 geometry_shading | 195 / 0 / 0 / 0 | 195 / 0 / 0 / 0 |

glmark2 order swap on a quiet machine: flat. Score: base→r99 8687 / 8488; r99→base 8574 / 8501. gl41-geometry: 3093→3174 and 3164 vs 3152.

Next in the approved sequence: geom r106+r107, then tess r67–r69. Each is re-verified on the new head before it merges.

## Merged 2026-11-05 14:54 EST: tess r66 (d2df080b)
This squashes two code commits from gl46/tess, eefc67c2 and 210132b3, onto c29e72e0. The report commits (4e651a6c, 9eca9654, which also carried tess-local ab/ helper scripts) and the [DROPPABLE] EXT-alias commit (now 26b39a11) were excluded. The aliases are still **not merged**.

What merged: instanced tessellation draws were broken on the GPU path.
- tessellation.c: the TCS per-patch record redirect to the global patch index matched only `spvTessLevel[gl_PrimitiveID]`; SPIRV-Cross also emits `spvTessLevel[int(gl_PrimitiveID)]`, which stayed on the local index.
- MGLRenderer+Tessellation.m: the tess gather kernel indexed source records by vertex alone, so with instances > 1 every instance received instance 0's records. It now enumerates (instance, index occurrence) and reads record [instance × vstride + vertex].
- New GPU/CPU parity regression tests in test_tess.c (self-exec with MGL_TESS_CPU=1). tess reports all single-instance cases were already pixel-identical between the two paths.

Verify: green on becb2fa4 and again on c29e72e0 after the geom merge (both touch tessellation.c): 0 FAIL, test_geom 817/0 ×11, 298/0 ×4.

CTS on the c29e72e0 + r66 verify build against baseline-c29e72e0, absolute counts:

| Group | cases | base Pass / Fail / NS / other | r66 Pass / Fail / NS / other |
|---|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 | 20 / 0 / 0 / 0 | 20 / 0 / 0 / 0 |
| ES31 tessellation_geometry_interaction | 28 | 28 / 0 / 0 / 0 | 28 / 0 / 0 / 0 |
| ES31 tessellation | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| ES31 geometry_shading | 203 | 203 / 0 / 0 / 0 | 203 / 0 / 0 / 0 |
| KHR-GL46 (11 groups) | 857 | 819 / 0 / 38 / 0 | 819 / 0 / 38 / 0 |

glmark2: a supervisor swap run under load (9.5→20, UTM ~116%, a worker CTS started mid-run) had r66 lower in both orders (8862/7378, 5495/7117). The user then ran the order swap by hand on a quiet machine after the 14:07 reboot:

| Scene | base→r66: base | r66 | r66→base: r66 | base |
|---|---|---|---|---|
| build | 15508 | 15302 | 16201 | 14461 |
| phong | 15162 | 15036 | 15931 | 13717 |
| blur | 4077 | 4351 | 4367 | 4320 |
| desktop-shadow | 3812 | 3791 | 3904 | 3694 |
| shadow | 10839 | 11195 | 10620 | 11183 |
| ideas | 2878 | 2919 | 3032 | 3029 |
| jellyfish | 12382 | 13048 | 12773 | 13175 |
| gl41-geometry | 2858 | 2890 | 2550 | 2515 |
| Score | 8438 | 8565 | 8671 | 8260 |

r66 is at or above the baseline in both orders (+1.5% / +5.0% overall). The user approved the merge at 14:52 EST.

Ops note: the Mac rebooted at 14:06–14:07 EST (shutdown-stall report, no panic). All three workers were relaunched with --resume at 14:55 at the user's request.

## Merged 2026-11-05 13:17 EST: geom r105a–m (c29e72e0)
This squashes gl46/geom r105a–r105m (tip 3caf338f) onto becb2fa4 as the same net diff that was verified on b25d975. The round report and ab/probe_r101.sh were excluded. The patch touches only geom-owned paths plus tess_spv_lower.c / tessellation.c (no overlap with r65).

What merged: the enhanced_layouts transform-feedback fixes. geom reports KHR-Single-GL46.enhanced_layouts xfb at 732/734 (was 696) and ssb at 1395/1395. enhanced_layouts itself was not re-run by the supervisor (user decision 10:22 Nov 4: SKIP).
- Array-of-matrices output reassembly keeps the "declared as array" bit.
- The xfb var encoding is reconciled with upstream r97c's chunk/gap words; the chunk/gap walk applies only to the slot map's padded dvec3-column specs.
- gl_NextBuffer resets the per-buffer interleaved component count (GL 4.6 §13.2.2).
- Layout validation skips cross-stage location caps for separable programs.
- VS-path capture variants load struct-member leaves into locals (the fp64 pass).

The first r105 READY (tip 5814c654) was **blocked**: test_compat_xfbdmat failed deterministically (dmat2x3 / dmat3 / dmat4x3 TES captures skipped the 4th double). geom fixed it in r105m. The re-verify on b25d975 was green: 0 FAIL, test_geom 817/0 ×11, 298/0 ×4, test_compat_xfbdmat PASS.

CTS on the r105m verify build against baseline-b25d975, absolute counts:

| Group | cases | base Pass / Fail / NS / other | r105m Pass / Fail / NS / other |
|---|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 | 20 / 0 / 0 / 0 | 20 / 0 / 0 / 0 |
| ES31 geometry_shading | 203 | 203 / 0 / 0 / 0 | 203 / 0 / 0 / 0 |
| ES31 tessellation_geometry_interaction | 28 | 28 / 0 / 0 / 0 | 28 / 0 / 0 / 0 |
| ES31 tessellation | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| KHR-GL46 (11 groups) | 857 | 818 / 0 / 38 / 1 NR | 819 / 0 / 38 / 0 |

glmark2 (user's method from 12:28 EST: one run per side in each order, all 8 scenes), against baseline-becb2fa4, load 16–20:

| Scene | base→r105m: base | r105m | r105m→base: r105m | base |
|---|---|---|---|---|
| build | 8782 | 12643 | 10939 | 9573 |
| phong | 7475 | 12672 | 9160 | 7966 |
| blur | 1884 | 3536 | 2297 | 1967 |
| desktop-shadow | 2980 | 3440 | 2529 | 1995 |
| shadow | 8313 | 7317 | 8335 | 6646 |
| ideas | 2866 | 2509 | 2851 | 2070 |
| jellyfish | 9495 | 9696 | 12883 | 7095 |
| gl41-geometry | 2037 | 2136 | 2289 | 2010 |
| Score | 5478 | 6742 | 6409 | 4914 |

r105m is at or above the baseline in both orders; the absolute numbers are depressed by load. The user approved the merge at 13:15 EST.

## Merged 2026-11-05 11:46 EST: tess r65 (becb2fa4)
This squashes one code commit from gl46/tess, 78856b32. The report commit ac8fa7a2 and the [DROPPABLE] ES-only EXT-alias commit (now 6c95c472) were excluded. The aliases are still **not merged** and wait on the user's decision.

What merged: exact tessellation counting queries while a TES-output transform-feedback capture owns the draw. With xfb capturing TES output and no geometry stage, every tess counting query (TES_INVOCATIONS, PRIMITIVES_GENERATED, CLIPPING_INPUT/OUTPUT) read 0. A stale r43 guard (`!tp->capture`) kept the draw on the GPU-emulated path, whose count words are folded only by a geometry draw. The exact-count CPU reference path is now forced when a counting query is active, unless a geometry stage consumes the capture. The regression test TessCountQueriesWithXfbCapture asserts all six query targets.

The verify build on b25d975 was green: 0 FAIL, test_geom 817/0 ×11 and the 298-check suite 298/0 ×4.

CTS on the verify build against baseline-b25d975, absolute counts:

| Group | cases | base Pass / Fail / NS / other | r65 Pass / Fail / NS / other |
|---|---|---|---|
| KHR-GL46 transform_feedback (subset) | 20 | 20 / 0 / 0 / 0 | 20 / 0 / 0 / 0 |
| ES31 tessellation_geometry_interaction | 28 | 28 / 0 / 0 / 0 | 28 / 0 / 0 / 0 |
| ES31 tessellation | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| ES31 geometry_shading | 203 | 203 / 0 / 0 / 0 | 203 / 0 / 0 / 0 |
| KHR-GL46 (11 groups) | 857 | 818 / 0 / 38 / 1 NR | 819 / 0 / 38 / 0 |

- The 1 NR on the baseline side (shader_image_load_store.basic-allTargets-atomic) was a runner artefact; it passes on r65.
- ES31 primitive_bounding_box (227): the full runs on **both** libraries stalled at identical positions on the tessellation+geometry cases and hit the 900 s runner timeout repeatedly. There were no GPU-fault lines. The two stalled cases passed quickly when run on their own on both libraries, so this is a runner/sequence artefact, not r65. The reference result for this group is the t64 run on identical b25d975 code: 187 P / 40 NS (all 40 NS are wide_lines, which need a line width of 5; MGL reports ALIASED_LINE_WIDTH_RANGE [1,1], which is legal).

glmark2: the interleaved 8+8 A/B ran three times on a heavily loaded machine (load 12–14; the user's Virtualization.framework VM at 140–400% CPU, Docker at up to 260%). Both sides swung between about 2100 and 4200 within a run. The blur, desktop-shadow and ideas medians missed the −2% gate (blur −9.4 / −9.4 / −6.6%). The user then ran single back-to-back passes of those three scenes in both orders: baseline-first gave r65 −5.7% overall, r65-first gave +0.5% overall (desktop-shadow flipped from −10.9% to +5.0%), so the gap tracks run order rather than the library. The user approved the merge at 11:44 EST.

## Merged 2026-11-05 05:11 EST: tess r61–r64 (b25d975)
This squashes four commits from gl46/tess — cf71223 (r61), 591c38d (r62), 0739f13 (r63) and 6a854c6 (r64) — re-staged onto a095741. The report and the [DROPPABLE] commit with the ES-only EXT aliases (751f0a6e, which would add GL_EXT_gpu_shader5, GL_EXT_tessellation_shader and GL_EXT_texture_cube_map_array) were excluded. The aliases are still **not merged** and wait on the user's decision. The only change to the GL extension string is the addition of GL_ARB_ES3_2_compatibility.

What merged:
- **r61:** GL_ARB_ES3_2_compatibility is implemented:
  - glPrimitiveBoundingBoxARB, exported for dlsym. It is state only; the spec treats the box as a hint.
  - The GL_PRIMITIVE_BOUNDING_BOX_ARB query, with defaults (−1,−1,−1,1, 1,1,1,1).
  - GL_MULTISAMPLE_LINE_WIDTH_RANGE_ARB = 1,1 and GL_MULTISAMPLE_LINE_WIDTH_GRANULARITY_ARB = 1.
  - An ESSL 3.20 shader audit.
- **r62:** fixes for the CTS cases the extension string un-gates:
  - gl_BoundingBox no longer counts against the per-vertex output budget.
  - Stale compile-cache stage sets.
  - An FBO zombie attachment.
  - A CPU-capture count pair, where an empty vertex range had drawn zero primitives.
  - glGet rounding.
- **r63:** fixes the 20 opaque_type_indexing failures, which were in atomic-counter arrays and shadow-sampler shapes.
- **r64:** adds the extension string.

The verify build on a095741 was green: 0 FAIL, test_geom 817/0 ×11 and the 298-check suite 298/0 ×4.

CTS on the verify build (made on 524c293) against baseline-524c293, absolute counts:

| Group | cases | base Pass / Fail / NS / other | r61–r64 Pass / Fail / NS / other |
|---|---|---|---|
| ES31 shaders.opaque_type_indexing | 516 | 100 / 8 / 408 / 0 | 516 / 0 / 0 / 0 |
| ES31 primitive_bounding_box | 227 | 0 / 0 / 227 / 0 | 187 / 0 / 40 / 0 |
| ES31 tessellation | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| ES31 tessellation_geometry_interaction | 28 | 28 / 0 / 0 / 0 | 28 / 0 / 0 / 0 |
| ES31 geometry_shading | 203 | 203 / 0 / 0 / 0 | 203 / 0 / 0 / 0 |
| ES31 program_interface_query | 4,599 | 4,599 / 0 / 0 / 0 | 4,598 / 0 / 0 / 1 NR (runner timeout) |
| KHR-GL46 set (draw_indirect, geometry_shader, PIQ, sample_shading/variables, image load-store/size, multisample_interpolation, tessellation_shader, texture_view, xfb) | 857 | 819 / 0 / 38 / 0 | 819 / 0 / 38 / 0 |
| KHR-GL46 transform_feedback | 20 | 20 / 0 / 0 / 0 | 20 / 0 / 0 / 0 |

There were no regressions. Four primitive_bounding_box cases came back NoResult in the batch run. These were 900 s alarm timeouts while the glmark2 A/B had glcts paused with SIGSTOP. Run on their own, each passed in about a second. The PIQ NoResult (program_input separable_fragment mat3) was not a test failure. The runner's 900 s per-part timer killed part 0 while that case was in progress, after 1,937 of 4,599 cases. The case passed 2 of 2 when run on its own against each library, so the true count is 4,599/4,599 on both. The 38 NotSupported in the KHR-GL46 set are all in sample_variables (36 .mask, 2 .position). Each one requests a sample count above MGL's reported MAX_SAMPLES or MAX_INTEGER_SAMPLES, or above what the format supports.

glmark2, interleaved 8+8 runs against baseline-a095741, change in the r61–r64 median: build +1.8%, phong +9.4%, blur +4.1%, desktop shadow −0.3%, shadow +0.3%, ideas 0.0%, jellyfish −0.3%. gl41-geometry came in at −6.9% on the first run, where scores were bimodal. Its re-run measured 3,221 → 3,161 (−1.9%).

## Merged 2026-11-05 03:43 EST: compat r97a–r98c (a095741)
This squashes the ten commits on gl46/compat from 73bc476 to fe820d55. Three scratch tests, ab/cts98.sh (which contained an absolute path) and the report were excluded. The rejected r97 merge was blocked because 12 ES31 program_interface_query cases went from Pass to NotSupported. r98 fixes that: GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS is now the sum of the per-stage limits (40 = 5×8, the Mesa and dEQP model) instead of the binding-point count. GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES rises from 96 to 128 to match.

The squash also carries:
- ES31 negative-API and negative-shader error fixes.
- Advanced-blend and sampler validation now runs for draws with a count of zero.
- An explicit layout(offset) on an atomic_uint is treated as the atomic-counter offset.
- The base alignment of a row_major matrix is now the row vector's alignment.
- The scatter for transform feedback from the TES is dense again, which fixes dmat2x3, dmat3 and dmat4x3 capture.
- A zero-length MapBufferRange now raises INVALID_OPERATION; the tessellation and transform-feedback groups were unchanged by this.

The verify build on 524c293 was green: 0 FAIL, test_geom 817/0 ×11 and the 298-check suite 298/0 ×4.

CTS on the verify build against baseline-524c293 (absolute counts):

| Group | cases | base Pass / Fail / NS / other | r98 Pass / Fail / NS / other |
|---|---|---|---|
| ES31 program_interface_query | 4,599 | 4,599 / 0 / 0 / 0 | 4,599 / 0 / 0 / 0 |
| ES31 compute + debug.negative_coverage | 1,687 | 1,589 / 47 / 39 / 12 QW | 1,640 / 5 / 39 / 2 QW, 1 NR |
| ES31 ssbo | 2,061 | 2,020 / 40 / 1 / 0 | 2,028 / 32 / 1 / 0 |
| KHR-GL46 es_31_compatibility | 355 | 310 / 7 / 38 / 0 | 314 / 3 / 38 / 0 |
| ES31 tessellation | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| KHR-Single enhanced_layouts.xfb* | 734 | 696 / 38 / 0 / 0 | 696 / 38 / 0 / 0 |
| KHR-GL46 transform_feedback | 20 | 20 / 0 / 0 / 0 | 20 / 0 / 0 / 0 |

There were no regressions. One case flagged on r98 only, debug.negative_coverage.callbacks.shader_image_exchange.buffer, but it passed 2 of 2 when run on its own against each library.

glmark2, interleaved 8+8 runs, change in the r98 median against base: build −0.4%, phong +5.2%, blur −0.5%, shadow −0.9%, ideas +1.4%, jellyfish +1.4%, gl41-geometry +0.9%. Desktop shadow came in at −10.5% on the first run, where scores were bimodal on both libraries. Its re-run measured 3,252 → 3,622 (+11.4%).

## Merged 2026-11-05 01:26 EST: geom r104 (524c293)
This is the net diff of gl46/geom (tip 2101f79) over 73bc476, limited to four files: MGL/src/fp64.c, MGL/src/geometry.c, MGL/src/mgl_ext_geom.h and tests/headless/test_geom.c (830 lines added, 41 removed). It carries the r103 work, and also fixes the regression that held r103 back: KHR-GL46.gpu_shader_fp64.fp64.varyings had started failing to link. The fix splices deferred-clone tails out of `pre`.

What merged:
- Transform feedback can now capture a whole struct, packed, as GL 4.6 §13.2.2 requires.
- Nested structs are flattened for capture.
- Array members with a single element work.
- Several fp64 SPIR-V layout fixes.

The from-scratch verify build on 73bc476 was green: 0 FAIL, test_geom 817/0 ×11 and the 298-check suite 298/0 ×4.

CTS, run on the verify build and on baseline-73bc476 (absolute counts):

| Group | cases | base Pass / Fail / NS / other | r104 Pass / Fail / NS / other |
|---|---|---|---|
| I | 28 | 28 / 0 / 0 / 0 | 28 / 0 / 0 / 0 |
| T | 395 | 395 / 0 / 0 / 0 | 395 / 0 / 0 / 0 |
| G | 203 | 203 / 0 / 0 / 0 | 203 / 0 / 0 / 0 |
| C | 387 | 387 / 0 / 0 / 0 | 387 / 0 / 0 / 0 |
| K | 857 | 819 / 0 / 38 / 0 | 819 / 0 / 38 / 0 |
| X (transform_feedback) | 20 | 19 / 0 / 0 / 1 NoResult | 20 / 0 / 0 / 0 |
| F (gpu_shader_fp64) | 659 | 659 / 0 / 0 / 0 | 659 / 0 / 0 / 0 |
| E (KHR-Single enhanced_layouts) | 4,250 | 4,143 / 106 / 0 / 1 NoResult | 4,144 / 105 / 0 / 1 NoResult |
| P | 4,599 | 4,599 / 0 / 0 / 0 | 4,599 / 0 / 0 / 0 |

There were no regressions. fp64.varyings passes. One case, enhanced_layouts.ssb_member_align_non_power_of_2_int, failed only on r104 in the batch run, but it passed 2 of 2 when run on its own against each library, so it was a batch flake.

glmark2, interleaved 8+8 runs, medians shown as base / r104:

| Scene | Run | base | r104 | Change |
|---|---|---|---|---|
| build | run 1 | 15,252 | 15,540 | +1.9% |
| shading (phong) | re-run | 15,013 | 15,405 | +2.6% |
| blur | 3rd run | 4,190 | 4,184 | −0.1% |
| desktop shadow | re-run | 3,376 | 3,577 | +6.0% |
| shadow | run 1 | 11,044 | 11,017 | −0.2% |
| ideas | run 1 | 3,092 | 3,058 | −1.1% |
| jellyfish | run 1 | 12,555 | 12,633 | +0.6% |
| gl41-geometry | re-run | 3,108 | 3,113 | +0.2% |

Run 1 happened while tess and compat were running. Four scenes missed in it: phong −14.6%, blur −3.2%, desktop shadow −4.3% and gl41-geometry −5.4%. All four had bimodal scores on both libraries. They passed on re-run, except blur, which needed a third run because the re-run came in at −2.6%.

## Official cts-runner pass (gl46), library d00c78b — completed 2026-11-04 11:45 EST
This was the Khronos `cts-runner --type=gl46` run on the d00c78b library (baseline from Nov 2), started at 18:20 EST on Nov 2. The runner hung at 21:50 EST on Nov 2 on KHR-GL46.gpu_shader_fp64.fp64.max_uniform_components, in the third gl46-main pass (64×-1). On the user's instruction it was killed on Nov 4 at 10:36 EST. The hung case passed when re-run alone. The rest of the third pass and the whole fourth pass were then run directly with glcts, using the runner's exact arguments and a 50-minute no-progress watchdog that never fired.

| Configuration | cases | Pass | Fail | NotSupported | other |
|---|---|---|---|---|---|
| ES3 aosp (256×256) | 1,325 | 1,277 | 48 | 0 | — |
| ES31 aosp (256×256) | 31,248 | 29,359 | 530 | 1,347 | 12 QualityWarning |
| gl30 khr noctx | 1 | 0 | 0 | 1 | — |
| gl40 khr noctx | 1 | 0 | 1 | 0 | — |
| gl43 khr noctx | 12 | 4 | 3 | 5 | — |
| gl45 khr noctx | 8 | 0 | 4 | 4 | — |
| gl46 khr-single | 11,348 | 4,907 | 149 | 6,292 | — |
| gl46-main pass 0 (64×64, seed 1) | 19,714 | 15,335 | 35 | 4,342 | 2 InternalError |
| gl46-main pass 1 (113×47, seed 2) | 19,714 | 15,332 | 38 | 4,342 | 2 InternalError |
| gl46-main pass 2 (64×-1, seed 3) | 19,714 | 15,339 | 31 | 4,342 | 2 InternalError |
| gl46-main pass 3 (-1×64, seed 3) | 19,714 | 15,339 | 31 | 4,342 | 2 InternalError |
| **Total** | **122,799** | **96,892** | **870** | **25,017** | 12 QW, 8 IE |

This is the d00c78b library. Merges since then (geom r98–r102, tess r56–r60, compat r96) are not reflected. Logs are in ~/MGL-cts/official-d00c78b/; the qpa files are config-*.qpa plus rest-*.qpa for the completed part.

## Merged 2026-11-04 07:42 EST: tess r60 (73bc476)
This squashes gl46/tess e4043f4..d67e171 (r59 and r60 part 1). GL_ARB_ES3_2_compatibility is no longer advertised, because MGL does not implement glPrimitiveBoundingBoxARB or its state. The ES-only aliases GL_EXT_gpu_shader5, GL_EXT_tessellation_shader and GL_EXT_texture_cube_map_array sit in a separate commit, 3d4d0af, which is **not merged** pending the user's decision.

What merged:
- Outputs the next stage never consumes no longer count toward the link-time location cap.
- In ES-profile non-fragment stages, an explicit textureLod on a shadow sampler compiles through GL_EXT_texture_shadow_lod.
- Transform feedback from the TES now packs fp64 values that split across slots correctly.

Files: geometry.c, mgl_layout_validate.c, program.c, tessellation.c, plus the new headless tests test_tess_esext.c and test_xfb_unconsumed.c.

| CTS group | cases | baseline e4043f4 | tess r60 (on e4043f4) |
|---|---|---|---|
| ES31 tessellation_geometry_interaction (256×256) | 28 | 28 Pass | 28 Pass |
| ES31 tessellation (256×256) | 395 | 395 Pass | 395 Pass |
| ES31 opaque_type_indexing, tessellation stages (256×256) | 172 | 172 NotSupported | 172 NotSupported |
| KHR tessellation_shader group | 387 | 387 Pass | 387 Pass |
| KHR-GL46 transform_feedback (minus get_xfb_varying) | 20 | 20 Pass | 20 Pass |
| ES31 geometry_shading (256×256) | 203 | 203 Pass | 203 Pass |
| KHR geometry_shader + arrays_of_arrays / opaque / PIQ | 857 | 819 Pass / 38 NotSupported | 819 Pass / 38 NotSupported |
| KHR-Single enhanced_layouts | 4250 | 4125 Pass / 124 Fail / 1 NoResult | **4132** Pass / 117 Fail / 1 NoResult |
| ES31 program_interface_query | 4599 | 4599 Pass | 4599 Pass |

The 172 opaque_type_indexing tessellation cases stay NotSupported on both sides: dEQP only runs them when GL_EXT_tessellation_shader is advertised, so they depend on the held alias commit.

A recheck on top of e1ce59a matched baseline-e1ce59a exactly: tessellation_geometry_interaction 28/28, geometry_shading 203/203 and transform_feedback 20/20, all Pass on both sides.

0 regressions, 0 crashes. One batch-run Pass→Fail (enhanced_layouts ssb_member_align_non_power_of_2_ivec2) passed 2/2 run alone on both libraries. The headless suite is green on a from-scratch build on e1ce59a (0 FAIL, test_geom 782/0 ×11, 298/0 ×4).

glmark2 interleaved 8+8 against baseline-e1ce59a, candidate vs baseline median:

| Scene | baseline | tess r60 | change |
|---|---|---|---|
| build | 14,327 | 14,491 | +1.1% |
| phong | 11,719 | 11,767 | +0.4% |
| desktop shadow | 3,713 | 3,686 | −0.7% |
| shadow | 9,897 | 10,417 | +5.3% |
| ideas | 3,081 | 3,103 | +0.7% |
| jellyfish | 12,560 | 12,554 | −0.1% |
| gl41-geometry | 2,800 | 2,760 | −1.4% |
| desktop blur, 8+8 re-run | 4,323 | 4,267 | −1.3% |

Desktop blur first read −2.2% (3,643 vs 3,725) in a noisy pass, with baseline runs ranging from 2,320 to 4,347; the 8+8 re-run above replaced it.

## Merged 2026-11-04 06:28 EST: geom r102 (e1ce59a)
With a geometry shader bound behind tessellation, the TES-invocation query now reports the exact count. Transform feedback now captures arrays of blocks and dmat arrays correctly. Files: MGLRenderer+Geometry.m, MGLRenderer+Tessellation.m, geometry.c, mgl_ext_geom.h, mgl_ext_tess.h, queries.c. test_geom grows from 718 to 782 checks.

| CTS group | cases | baseline e4043f4 | e1ce59a | change |
|---|---|---|---|---|
| ES31 tessellation_geometry_interaction (256×256) | 28 | 28 Pass | 28 Pass | — |
| ES31 tessellation (256×256) | 395 | 395 Pass | 395 Pass | — |
| ES31 geometry_shading (256×256) | 203 | 203 Pass | 203 Pass | — |
| KHR tessellation_shader group | 387 | 387 Pass | 387 Pass | — |
| KHR geometry_shader + arrays_of_arrays / opaque / PIQ | 857 | 819 Pass / 38 NotSupported | 819 Pass / 38 NotSupported | — |
| KHR-Single enhanced_layouts | 4250 | 4125 Pass / 124 Fail / 1 NoResult | **4136** Pass / 113 Fail / 1 NoResult | +11 |
| ES31 program_interface_query | 4599 | 4599 Pass | 4599 Pass | — |

0 regressions, 0 crashes. One batch-run Pass→Fail (enhanced_layouts ssb_member_align_non_power_of_2_vec3) passed 2/2 run alone on both libraries. The headless suite is green on a from-scratch build (846 PASS / 0 FAIL, test_geom 782/0 ×11, 298/0 ×4). glmark2 interleaved 8+8, candidate vs baseline median: build 13,869 vs 13,523 (+2.6%), phong 11,943 vs 10,951 (+9.1%), desktop blur 3,682 vs 3,641 (+1.1%), desktop shadow 3,640 vs 3,494 (+4.2%), shadow 9,566 vs 9,120 (+4.9%), ideas 2,720 vs 2,728 (−0.3%), jellyfish 9,303 vs 9,174 (+1.4%), gl41-geometry 2,188 vs 2,196 (−0.3%). The A/B waited from 01:44 to 05:58 for the host QEMU to go idle.


## Merged 2026-11-03 19:03 EST: geom r101/r101b (e4043f4)
A tessellation evaluation shader feeding a geometry shader now works with transform feedback, the PRIMITIVES_GENERATED query and gl_PointSize. An open PRIMITIVES_GENERATED query no longer forces the tessellation CPU-reference path when a geometry shader is attached, which had left the geometry stage reading an empty capture. The counters are no longer double counted. The indirect geometry dispatch zeroes its per-thread emit counts, so stale records from the previous draw are no longer packed into the feedback buffer. gl_PointSize is now carried through the tessellation capture into gl_in[]. test_geom grows from 702 to 718 checks.

| CTS group | cases | baseline 411a33e | e4043f4 | change |
|---|---|---|---|---|
| ES31 tessellation_geometry_interaction (256×256) | 28 | 15 Pass / 6 Fail / 7 NotSupported | **28** Pass | +13 |
| ES31 geometry_shading (256×256) | 203 | 202 Pass / 1 NotSupported | **203** Pass | +1 |
| ES31 tessellation (256×256) | 395 | 395 Pass | 395 Pass | — |
| KHR tessellation_shader group | 387 | 387 Pass | 387 Pass | — |
| KHR geometry_shader + arrays_of_arrays / opaque / PIQ | 857 | 819 Pass / 38 NotSupported | 819 Pass / 38 NotSupported | — |
| ES31 program_interface_query | 4599 | 4599 Pass | 4599 Pass | — |

0 regressions, 0 crashes. The headless suite is green on a from-scratch build (test_geom 718/0 ×11, 298/0 ×4, 842 PASS / 0 FAIL). glmark2 interleaved 8+8, candidate median vs baseline: build −0.1%, desktop blur +3.8%, desktop shadow −1.2%, shadow +1.0%, ideas +3.2%, jellyfish −0.1%, gl41-geometry −0.7%. Phong first read −5.6% in a noisy pass where the baseline side was also low; the 8+8 re-run measured −0.8% (14,396 vs 14,506).

## Merged 2026-11-03 15:25 EST: tess r57 (411a33e)
`GL_BUFFER_DATA_SIZE` for a shader storage block now counts one element of a trailing unsized (runtime-sized) array, as GL 4.6 and ES 3.1 require (introspect_spirv.c). The round adds the headless test test_ssbo_minsize.c.

| CTS group | cases | baseline c823022 | 411a33e | change |
|---|---|---|---|---|
| ES31 program_interface_query | 4599 | 4596 Pass / 3 Fail | **4599** Pass / 0 Fail | +3 |
| ES31 ssbo | 2061 | 2020 Pass / 40 Fail / 1 NotSupported | 2020 Pass / 40 Fail / 1 NotSupported | — |
| ES31 tessellation (256×256) | 395 | 395 Pass | 395 Pass | — |
| KHR tessellation_shader group | 387 | 387 Pass | 387 Pass | — |
| KHR-GL44.tessellation_shader | 140 | 140 Pass | 140 Pass | — |

0 regressions, 0 crashes. The headless suite is green on a from-scratch build (test_geom 702/0 ×11, 298/0 ×4, 839 PASS / 0 FAIL). glmark2 interleaved 8+8, candidate median vs baseline: build +0.2%, phong +0.7%, desktop blur +0.3%, desktop shadow +3.2%, shadow +0.3%, ideas −0.2%, jellyfish +0.2%, gl41-geometry 0.0%.

**Held: compat r97a–r97h.** It improves ES31 negative coverage + compute (1687 cases: 1589 → 1641 Pass), es_31_compatibility (355: 309 → 312 Pass) and ES31 ssbo (2061: 2020 → 2028 Pass), but loses 12 ES31 program_interface_query cases (Pass → NotSupported, "resource limits exceeded"; storage-block arrays shared across the tessellation and geometry stages). It was sent back for a fix.

**Official `cts-runner --type=gl46`** is still hung on `KHR-GL46.gpu_shader_fp64.fp64.max_uniform_components` (since 2026-11-02 21:50), with 9 configurations done.

## Merged 2026-11-03 09:35 EST: tess r56 (c823022)
GL_KHR_robustness is now advertised (get.c). On the baseline the KHR-GL44.tessellation_shader group crashed, so all 140 cases reported no result; with the change all 140 pass. A tessellation debug print is now behind an environment gate. The round also adds the headless test test_tess_esext.c and the CTS A/B helper ab/cts_suite_ab.sh.

| CTS group | cases | baseline 9cc45a7 | c823022 | change |
|---|---|---|---|---|
| KHR-GL44.tessellation_shader | 140 | 0 Pass / 140 no result (crash) | **140** Pass | +140 |
| ES31 tessellation (256×256) | 395 | 395 Pass | 395 Pass | — |
| KHR tessellation_shader group | 387 | 387 Pass | 387 Pass | — |
| ES31 program_interface_query | 4599 | 4589 Pass / 10 Fail | 4589 Pass / 10 Fail | — |

0 regressions, 0 crashes. The headless suite is green on a from-scratch build on top of e0927e4 (test_geom 702/0 ×11, 298/0 ×4, 836 PASS / 0 FAIL). glmark2 interleaved 8+8, candidate median vs baseline: build −1.1%, phong +4.2%, desktop blur +0.2%, desktop shadow +0.3%, jellyfish −0.9%, gl41-geometry +0.4%. Shadow and ideas were re-measured 8+8 after host-VM contamination: shadow 10,948 → 10,941 (−0.1%), ideas 3,068 → 3,056 (−0.4%).

## Merged 2026-11-03 07:38 EST: geom r98–r100 (e0927e4)
Opaque arrays-of-arrays now work on every stage. The round also fixes the geometry-shader capture pass and adds a capture-pass setup cache. test_geom grows from 680 to 702 checks.

| CTS group | cases | baseline 9cc45a7 | e0927e4 | change |
|---|---|---|---|---|
| ES31 program_interface_query | 4599 | 4589 Pass / 10 Fail | **4596** Pass / 3 Fail | +7 |
| ES31 geometry_shading (256×256) | 203 | 202 Pass / 1 NotSupported | 202 Pass / 1 NotSupported | — |
| KHR geometry_shader + arrays_of_arrays / opaque / PIQ | 857 | 819 Pass / 38 NotSupported | 819 Pass / 38 NotSupported | — |

0 regressions, 0 crashes. Headless suite green (test_geom 702/0 ×10+, 298/0 ×3). glmark2 interleaved 8+8, candidate median vs baseline: build −1.7%, phong −1.1%, desktop blur −0.4%, desktop shadow +0.7%, shadow 0.0%, ideas −0.1%, jellyfish +0.4%, gl41-geometry +0.9%.

**Official `cts-runner --type=gl46`** is still hung on `KHR-GL46.gpu_shader_fp64.fp64.max_uniform_components` (since 2026-11-02 21:50), with 9 configurations done. It runs against baseline-d00c78b, which predates tess r56's GL_KHR_robustness change.

## Merged 2026-11-02 22:58 EST: tess r48b–r54 (9cc45a7)
Per-patch interface block arrays now work end to end. Array-form per-patch blocks are dissolved for SPIRV-Cross, with each element matched by array index in the canonical patch record. TES inputs get per-element lowering, and mixed per-vertex blocks plus patch block arrays link. Capture and isolines lowering close all of r52's leftovers, and dynamic array-member index lowering is now limited to dissolved modules. r54 makes the TES patch-block-array local module-scope Private instead of entry-block Function, which fixes the ES31 PIQ `patch_in.block_array.array.mat3x2` regression that held the earlier READY.

| CTS group | cases | baseline d00c78b | 9cc45a7 | change |
|---|---|---|---|---|
| ES31 tessellation (256×256) | 395 | 368 Pass / 27 Fail | **395** Pass / 0 Fail | +27 |
| ES31 program_interface_query | 4599 | 4588 Pass / 11 Fail | **4589** Pass / 10 Fail | +1 |
| KHR tessellation_shader group | 387 | 386 Pass / 1 Fail | **387** Pass / 0 Fail | +1 |
| KHR-Single enhanced_layouts | 4250 | 4125 Pass / 124 Fail / 1 no result | 4125 Pass / 124 Fail / 1 no result | — |

0 regressions, 0 crashes; headless suite green (test_geom 680/0 ×11, 298/0 ×4). The one enhanced_layouts "no result" (`ssb_member_align_non_power_of_2_mat3` on the candidate) was a batch artefact; it passed twice in isolation on both libraries. glmark2 interleaved 8+8, candidate median vs baseline: build +0.3%, phong +5.8%, desktop blur −1.0%, desktop shadow +6.7%, shadow +9.7%, ideas +5.3%, jellyfish +0.8%, gl41-geometry +0.2%. The first A/B attempt was discarded because the host VM was using 200–300% CPU.

**Official `cts-runner --type=gl46`** (baseline-d00c78b, started 18:20): 9 configurations finished, then it hung at 21:50 on `KHR-GL46.gpu_shader_fp64.fp64.max_uniform_components` in the third gl46-main configuration (width 64, height −1, seed 3). The same case passes in the first gl46-main configuration. It has been left running pending a decision.

## Merged 2026-11-02 17:48 EST: geom r97–r97f (d00c78b)
Atomic-counter arrays-of-arrays are flattened (MSL argument, SPIR-V opaque flatten, GL resource names per row). The geometry-shader link now enforces the total output-component limit, and anonymous and inline-struct interface blocks link. **Layered MSAA-array textures no longer trip the macOS IOGPU assertion** (fresh buffer in geomDefineMultisampleTexture). Primitive restart now works on the CPU assembly route and on the geometry path, including PRIMITIVE_RESTART_FIXED_INDEX with a type-aware sentinel. r97e fixed three regressions that r97/r97b had introduced against 6a3cb31 before the READY.

| CTS group | cases | baseline 6a3cb31 | d00c78b | change |
|---|---|---|---|---|
| ES31 geometry_shading minus the 8 msaa-array cases (256×256) | 195 | 185 Pass / 9 Fail / 1 NotSupported | **194** Pass / 0 Fail / 1 NotSupported | +9 |
| ES31 geometry_shading `*_2d_multisample_array` (256×256) | 8 | not run (IOGPU assertion) | **8** Pass | +8 |
| ES31 tessellation + program_interface_query (256×256) | 4994 | 4954 Pass / 40 Fail | **4956** Pass / 38 Fail | +2 |
| KHR-Single enhanced_layouts | 4250 | 4125 Pass / 124 Fail / 1 no result | 4125 Pass / 124 Fail / 1 no result | — |
| KHR geometry_shader family (geom's 814-case list) | 814 | 775 Pass / 1 Fail / 38 NotSupported | 775 Pass / 1 Fail / 38 NotSupported | — |

0 regressions, 0 crashes; headless suite green (test_geom 680/0 ×11, 298/0 ×4). glmark2 was measured as an interleaved 8+8 per scene because Docker bursts to ~145% CPU. Candidate medians vs baseline: build −1.4%, desktop blur +0.5%, desktop shadow +7.7%, shadow +0.8%, ideas +0.3%, jellyfish +0.1%, gl41-geometry +1.6%. Phong first read −4.9% in a noisy pass; the 8+8 re-run measured +1.7% (15,925 vs 15,663).

The official `cts-runner --type=gl46` was started against baseline-d00c78b (the IOGPU blocker is fixed); see the 9cc45a7 section for its status.

## Merged 2026-11-02 11:52 EST: compat r96 (6a3cb31)
Interface-matching and transform-feedback round: cross-stage location renumbering with TES attribute compression (user-declared locations are never renumbered); transform-feedback capture of arrays-of-matrices and whole struct instances (std430 flattening), TES-declared capture with a GS attached; pipeline-bound programs get their buffer bindings; framebuffer completeness honours no-attachment FBOs; vertex-input aliasing is legal while component aliasing of other stages' inputs fails the link. Also a geom_ensure double-free fix.

| CTS group | cases | baseline f2a8e50 | 6a3cb31 | change |
|---|---|---|---|---|
| KHR-Single enhanced_layouts | 4250 | 4101 Pass / 148 Fail / 1 no result | **4126** Pass / 123 Fail / 1 no result | +25 |
| ES31 program_interface_query | 4599 | 4585 Pass / 14 Fail | **4586** Pass / 13 Fail | +1 |
| ES31 tessellation + debug.negative_coverage (256×256) | 1887 | 1752 Pass / 84 Fail / 39 NotSupported / 12 QualityWarning | **1770** Pass / 66 Fail / 39 NotSupported / 12 QualityWarning | +18 |
| KHR geometry_shader, multi_bind, layout_binding, draw_indirect, tessellation_shader, transform_feedback (minus get_xfb_varying), vertex_attrib_binding, direct_state_access | 805 | 801 Pass / 3 Fail / 1 InternalError | 801 Pass / 3 Fail / 1 InternalError | — |

0 regressions, 0 crashes; headless suite green. glmark2: 3-round desktop blur −2.2% re-judged 8+8 at +1.7%; shadow +5.4%, gl41-geometry +7.5%, other scenes within ±1.5%; scores 9942 (5-scene) and 6242 (3-scene).

The two enhanced_layouts "crashers" reported earlier do not exist: the names mixed two test families and match no case in any CTS list (compat checked the lists and the CTS source). They do not block the official cts-runner run. That run waited on geom's fix for the 8 layered `*_2d_multisample_array` cases; the fix landed in d00c78b.

## Merged 2026-11-02 07:00 EST: geom r90–r95c (f2a8e50)
Link-time and transform-feedback correctness round. Overlapping explicit block-member offsets and per-stage in/out component budgets (GL 4.6 §11.1.1) now fail the link; member offset/align outside an explicit std140/std430 block is a compile error. Transform feedback: fp64 capture through widened std140 double arrays, in-shader `xfb_*` qualifiers override an API varying list on GS/TES/TCS, and only outputs with an explicit `xfb_offset` are captured. Flattened per-patch block arrays now match each element to its own slot (the long-running `test_piq88 patch_block_array_pipeline` failure was this bug, not a cache race). Geometry-shader array-of-matrix outputs now link (one matrix per element in the draw pass).

| CTS group | cases | baseline 4c3bb48 | f2a8e50 | change |
|---|---|---|---|---|
| KHR-Single enhanced_layouts (full list) | 4250 | 4078 Pass / 171 Fail / 1 no result | **4101** Pass / 148 Fail / 1 no result | +23 |
| ES31 program_interface_query | 4599 | 4581 Pass / 18 Fail | **4585** Pass / 14 Fail | +4 |
| KHR geometry_shader, multi_bind, layout_binding, draw_indirect, tessellation_shader | 384 | 383 Pass / 1 Fail | 383 Pass / 1 Fail | — |
| ES31 tessellation (64×64 surface) | 395 | 244 Pass / 151 NotSupported | 244 Pass / 151 NotSupported | — |

No NotSupported, crash or QualityWarning results except as listed. "No result" = the run's 900 s batch timer expired on that case (a different case on each side); both pass in isolation in about 4 s. The one KHR Fail is `tessellation_shader_tessellation.max_in_out_attributes` on both sides. The ES31 tessellation NotSupported are all "render target must be at least 256" — this comparison ran at 64×64, so it only exercised 244 of the 395 cases (tess measures that group at 256×256: 368/395 at r48).

0 regressions, 0 crashes; headless suite fully green for the first time today (test_piq88 included). glmark2: first 3-round pass showed blur −2.3%, ideas −2.4%, gl41-geometry −2.5%; the 8+8 re-judge put them at +0.3%, −0.4%, +0.8%. Scores ~10,000 (5-scene) and ~6,300 (ideas/jellyfish/gl41). Not run: ES31 geometry_shading, whose 8 layered `*_2d_multisample_array` cases trip a macOS IOGPU assertion on the baseline too.

## Merged 2026-11-02 00:22 EST: compat r86–r95 (4c3bb48)
Compiler/validation round. A new link-time validator (`mgl_layout_validate.c`) checks GLSL 4.60 block-member offset overlap and alignment and per-stage varying location limits; shader-declared transform-feedback capture (XfbStride) is honoured; `#version N es` shaders get strict ES semantics; shared/packed blocks keep their packing when a member declares offset/align; array-of-matrix outputs are lowered for MSL; qualifiers on plain struct members are rejected. The validator skips MGL's own injected blocks, per-vertex interfaces, xfb blocks and glslang auto-mapped block instances.

| CTS group (vs baseline-1f4c16b) | cases | change |
|---|---|---|
| KHR-Single enhanced_layouts (compat's sample) | 1581 | **+156** Fail→Pass |
| ES31 debug.negative_coverage | 1492 | **+24** (1360 → 1384 Pass) |
| KHR shaders.negative / integer_mix / uniform_block sample / PIQ | 158 | +1 |
| KHR-Single arrays_of_arrays (sample) | 81 | unchanged |

0 regressions, 0 crashes. Two enhanced_layouts cases (`uniform_block_member_overlapping_offsets_vec4_vertex`, `ssb_member_overlapping_offsets_ivec4_vertex`) crash glcts on the baseline as well; compat is asked to fix them next. glmark2: all scenes within 2% (gl41-geometry −5.9% on 6 bimodal rounds, +0.5% on the 8+8 re-judge); scores ~10,000 (5-scene) and ~6,100–6,400 (ideas/jellyfish/gl41). Headless `test_piq88 patch_block_array_pipeline` now fails when run alone on the 1f4c16b baseline too, so it is pre-existing, not from this merge; geom is root-causing it.

## Merged 2026-11-01 20:21 EST: tess r48 (1f4c16b)
Struct and array tessellation interface blocks between TCS and TES (array-of-struct members dissolved to one variable per leaf, because SPIRV-Cross mis-binds chains into per-vertex array<struct>); whole-struct loads that used to abort SPIRV-Cross now fail the link cleanly; texture-buffer-backed buffers track reads; sampler rebinds are no longer skipped when Metal reuses a sampler's address.

| CTS group (vs baseline-dd77727) | before | after |
|---|---|---|
| ES31 tessellation | 359 | **368** / 395 |
| ES31 texture.texture_buffer | 346 | **351** / 351 |
| ES31 gather / border_clamp / image_load_store | 831 / 595 / 747 | unchanged |
| KHR tessellation_shader / texture_view | 139/140, 7/7 | unchanged |

0 Pass→Fail, 0 crashes. glmark2: the 3-round run was noisy (baseline itself swung up to 30% on gl41-geometry), so build, phong, ideas and gl41-geometry were re-judged with 8+8 at low load: +2.1%, +1.3%, −0.3%, +1.4%.

**Known parallel-only headless flakes:** test_capszoo caps_zoo_compile_link and test_piq88 patch_block_array_pipeline each failed once in the parallel `run.sh` and pass every time alone; geom is checking the GLSL/MSL disk cache for a write race between concurrent processes.

## Merged 2026-11-01 18:58 EST: geom r88/r89 (dd77727)
Fixes the 6 gl46-main regressions the 7edc431 re-measure found (atomic_uint arrays now share one binding at consecutive offsets; TRIANGLES_ADJACENCY uses vertices 0/2/4), multisample image handling for multi_bind, flattening of arrays of patch interface blocks, and — the reason the first READY (dbdf449) was held at 15:00 — an in-process link-cache bug: a cache hit introspected the already-flattened module, so 7 ES31 program_interface_query patch-block-array cases failed only when an earlier case in the same glcts process linked the same source. Plus transform-feedback capture of flattened whole matrix arrays.

| CTS group (vs baseline-8554b98) | before | after |
|---|---|---|
| ES31 program_interface_query | 4,541 | **4,576** / 4,599 |
| KHR-GL46 multi_bind | 15 | **17** / 17 |
| KHR-GL46 layout_binding | 18 | **21** / 21 |
| KHR-GL46 draw_indirect | 68 | **70** / 70 |
| KHR-GL46 geometry_shader | 135 | **136** / 136 |

0 Pass→Fail, 0 crashes (separate_shader 183/183, KHR PIQ 43/43, transform_feedback 21/21, primitive_restart 1/1 unchanged). glmark2 within 2% on every scene (gl41-geometry re-judged with 8+8: +0.5%; desktop shadow −1.96% in the 3-round run).

**Load incident 13:48 EST:** compat's before/after CTS ran 12 glcts at once (load 22/18/15). Compat was paused by the load rule; the cap is now 4 glcts machine-wide (budgets: compat 2, tess 1, geom 1). Compat resumed by the user at 16:22.

## Merged 2026-11-01 07:08 EST: tess r46/r47 (8554b98)
textureGatherOffsets and GL_CLAMP_TO_BORDER emulation for gather, image atomics on 3D/cube/buffer images, tessellation interface-block varyings and primitive_discard/state_query, the gl46-main q_tess batch (texture_query_levels, clear_tex_image, STENCIL_INDEX8, LOD bias), glBlitFramebuffer format rules, plus the two blockers that held r46.1: a SPIRV-Cross abort in a tessellation case and order-dependent wrong-image writes from stale pointer-keyed caches.

| dEQP group (vs baseline-f3e99eb) | before | after |
|---|---|---|
| gl46-main q_tess list | 1/21 | **19/21** |
| ES31 tessellation | 293/395 | **359/395** |
| ES31 image_load_store | 663/747 | **747/747** |
| ES31 texture.gather | 688/831 | **831/831** |
| ES31 texture.border_clamp | 494/603 | **595/603** |
| ES3 negative blit_framebuffer | 0/2 | 1/2 |

0 Pass→Fail flips and 0 crashes in any group. glmark2 within 2% (desktop shadow re-judged with an 8+8 run: 3,904 vs 3,891; ideas −1.2%, jellyfish −1.1%, gl41-geometry −0.9%). New baseline ~/MGL-cts/baseline-8554b98.

**Regressions found by the 7edc431 re-measure, routed to geom:** six gl46-main cases fail 3/3 on f3e99eb and passed on 001679a: draw_indirect basic-mode triangles_adjacency (drawArrays, drawElements), geometry_shader.api.max_atomic_counters, layout_binding atomic_uint (compute, fragment, vertex). The seventh (ssbo advanced-unsizedArrayLength-vs-std140-matC) passes 3/3, so it was a one-off.

## Merged 2026-11-01 05:25 EST: geom r87 (f3e99eb)
- **separate_shader "flakiness" root-caused and fixed:** rs_program_has_flat() memoized its answer keyed on the MSL string POINTER, so a freed-and-reused buffer returned another program's answer — run-to-run nondeterminism on every earlier build. dEQP-GL45-ES31 separate_shader **183/183 on two consecutive runs** (baseline 147 and 153).
- Arrays of matrices in stage I/O flattened at the SPIR-V level: program_interface_query 4,493 → **4,541**/4,599 (+48, 0 regressions).
- glmark2 within 2% (shadow and gl41-geometry confirmed with 8+8 runs). New baseline ~/MGL-cts/baseline-f3e99eb.

**Official-equivalent re-measure on 7edc431 — complete (Nov 1 03:28–06:40 EST):**

| Package | 001679a Pass / Fail / NS / Crash | 7edc431 Pass / Fail / NS / Crash |
|---|---|---|
| ES3 (1,325) | 1,261 / 64 / 0 / 0 | 1,261 / 64 / 0 / 0 |
| ES31 (31,248) | 28,179 / 1,638 / 1,350 / 71 | **28,771 / 1,120 / 1,347 / 0** |
| no-context gl30/40/43/45 | 4 / 8 / 10 / 0 | 4 / 8 / 10 / 0 |
| KHR-Single (11,348) | 4,064 / 991 / 6,292 / 1 | 4,064 / 991 / 6,292 / 1 |
| gl46-main 64×64 | 15,331 / 40 | 15,333 / 38 |
| gl46-main 113×47 | 15,333 / 38 | 15,328 / 43 |
| gl46-main 64×−1 | 15,337 / 34 | 15,334 / 37 |
| gl46-main −1×64 | 15,336 / 35 | 15,333 / 38 |
| **Total** | 94,845 / 2,848 / 25,020 / 72 | **95,428 / 2,339 / 25,017 / 1** |

gl46-main: fixed since 001679a — es_31_compatibility ×3, primitive_restart, transform_feedback, transform_feedback3. Newly failing in at least one config (7 cases; 6 confirmed on f3e99eb, see above): draw_indirect basic-mode triangles_adjacency ×2, geometry_shader.api.max_atomic_counters, layout_binding atomic_uint ×3, ssbo advanced-unsizedArrayLength-vs-std140-matC. The only crash left is KHR-Single enhanced_layouts.xfb_stride_of_empty_list_and_api. 7edc431 predates geom r87, so separate_shader is still nondeterministic in this run.

**Tess r46.1 (243734e) held:** big gains (gather 831/831, border clamp 595/603, image load/store +83, 16/21 queued gl46-main cases) but 1 new crash (SPIRV-Cross assertion in a tessellation case) and 26 3D-image single-layer regressions; returned to tess.

## Merged 2026-11-01 02:45 EST: glmark2 regression FIXED (7edc431)
Root cause: 784c480 declined the compat pipeline cache on every draw after an FBO switch, rebuilding the full Metal pipeline descriptor (~1 ms/frame in [desktop]). Fix: removed the decline; the cache key now hashes the attachment formats the descriptor path would bake plus each attachment's GL internal format/target; DIRTY_FBO cleared on a cache hit; ShaderWrite dropped from the default framebuffer texture.

| glmark2 scene | 7edc431 (fix) | e26f143 (before) | d31c9ba (pre-regression) |
|---|---|---|---|
| desktop blur | **4,350** | 1,425 | ~4,020 |
| desktop shadow | **3,773** | 803 | ~3,650 |
| shadow | **11,287** | 6,725 | ~12,100 |
| phong | 14,953 | 14,668 | — |
| build | 15,039 | 14,843 | — |

Checks: verify green; flash_check PASS; captures correct; CTS replay (ES3/ES31/KHR-Single/gl46-main) shows no regression, gl46-main 15,336 Pass. New baseline ~/MGL-cts/baseline-7edc431.

## Merged 2026-11-01 02:25 EST: geom r86h/i (e26f143)
Transform feedback of geometry-shader output (strips captured decomposed into primitives) and primitive restart with the adjacency list modes. KHR-GL46.transform_feedback* 36 → **38/38**, primitive_restart 0 → **1/1**, geom's queued gl46-main list 2 → 5/7; no regressions; glmark2 within 1.5% (gl41-geometry 8+8: 3,410 → 3,409). New baseline ~/MGL-cts/baseline-e26f143.

**Tess r46 (221366d) rejected:** a from-scratch build fails 7 headless test files (tess_dynio crash, geom_msimg, texext buffer textures, bordemu, alphaless, texswizzle, buffers) on tess's own branch too; returned to tess.
**Compat paused** since 00:51 by the load rule (load 27.9/18/12.7), until the user resumes it.

**glmark2 fix v4 CTS regression check — complete and clean** (on 9f0c6f0): ES3 identical; ES31 only known-flaky separate_shader.random flips; KHR-Single 0 regressions; gl46-main 15,336 Pass / 35 Fail, texture_size_promotion fixed, +6 vs r001679a (one flip, vertex_attrib_binding case11, passes 3/3 in isolation and 30/30 in-group ×2). Remaining: verify on e26f143, A/B vs baseline-e26f143, flash_check, captures, then merge.

## Merged 2026-11-01 01:24: geom r86 (7833246)
Introspection + link correctness for the mixed-stage shapes the CTS generates (hoisted sampler-struct names, nested SSBO members, cross-stage uniform-location relink, binding-array validation) and flat varyings through program pipelines (provoking vertex). Cherry-pick of 127a9f7..5660feb; four ab/ helper scripts with absolute paths left out.
- dEQP-GL45-ES31 vs baseline-9f0c6f0: program_interface_query 4,287 → **4,493**/4,599; program_uniform 957 → **1,002**/1,014; separate_shader 134 → **151**/183; KHR-GL46.sepshaderobjs 8/8. No real regressions (9 separate_shader flips are order-dependent: pass 3/3 in isolation on r86, baseline fails 6 of them).
- glmark2 within 2% on every scene (gl41-geometry 8+8 at low load: 3,186 → 3,132, −1.7%). New baseline ~/MGL-cts/baseline-7833246.

## glmark2 regression: root cause found (fix in final CTS check)
The ~60% loss on desktop blur/shadow and ~40% on shadow since Oct 26 came from 784c480 (compat s13–s16), NOT the present copy (d31c9ba copied every swap too). 784c480 added a decline at the top of the compat pipeline fast path — any draw with DIRTY_FBO (or without VAO/program/alpha dirt) rebuilt the full Metal pipeline descriptor. FBO-switching scenes paid that on all 26 draws/frame (MGL_PROFILE: ~1 ms/frame on 9f0c6f0; d31c9ba hit the cache 26/26).
- Same-session: d31c9ba desktop blur ~4,020 / desktop shadow ~3,650 / shadow ~12,100; 784c480 ~1,590 / ~1,050 / ~7,740; 9f0c6f0 ~1,430 / ~800 / ~6,900.
- Fix v4: remove the decline; key the attachment formats from the same inputs the descriptor path bakes PLUS each attachment's GL internal format/target (re-hashed on every FBO change), so emulated formats sharing a Metal format (GL_RGBA2 in RGBA8) can't alias — v1 without that broke KHR-GL46.texture_size_promotion; drop ShaderWrite from the default framebuffer texture.
- v4 vs 9f0c6f0: desktop blur 1,428 → 4,327, desktop shadow 798 → 3,807, shadow 6,719 → 11,128, build/phong +2–3%. texture_swizzle 505/0, texture_size_promotion pass, flash_check PASS. CTS regression run: ES3 identical, ES31 only known-flaky separate_shader.random flips; KHR-Single + gl46-main in progress. Rebased onto 7833246 unchanged.

Known flaky / order-dependent on every build (not counted as regressions): dEQP-GL45-ES31 separate_shader.random.* and separate_shader.interface.*, draw_indirect base_vertex index_(neg_)byte, KHR-GL46 direct_state_access textures_buffer_rgba32{i,ui,f}.

## Re-measure on 001679a (complete, Oct 31 ~17:30–21:20)
The same 12 official-equivalent runs, replayed with the crash-resuming driver. 001679a predates geom r85, so the 71 ES31 PIQ crashes below are already fixed in 9f0c6f0.

| Package | 965d88f Pass / Fail / NS / Crash | 001679a Pass / Fail / NS / Crash | ps5-opengl Pass / NS |
|---|---|---|---|
| ES3 (1,325) | 1,251 / 64 / 0 / 10 | 1,261 / 64 / 0 / 0 | 1,325 / 0 |
| ES31 (31,248) | 16,351 / 13,498 / 1,312 / 71 | **28,179 / 1,638** / 1,350 / 71 | 30,866 / 382 |
| no-context gl30/40/43/45 | 4 / 8 / 10 / 0 | 4 / 8 / 10 / 0 | 3 / 19 |
| KHR-Single (11,348) | 4,064 / 991 / 6,292 / 1 | 4,064 / 991 / 6,292 / 1 | 5,056 / 6,292 |
| gl46-main 64×64 | 15,043 / 328 / 4,342 | **15,331 / 40** / 4,342 | 15,335 / 4,378 |
| gl46-main 113×47 | 15,042 / 329 | 15,333 / 38 | same |
| gl46-main 64×−1 | 15,044 / 327 | 15,337 / 34 | same |
| gl46-main −1×64 | 15,044 / 327 | 15,336 / 35 | same |
| **Total** | 81,843 / 15,872 / 24,982 / 82 | **94,845 / 2,848** / 25,020 / 72 | |

gl46-main: 42 distinct failing cases (most fail in all four configs). Routed as "after the current round" items: tess 21 (texture_query_levels 13, framebuffer_blit, clear_tex_image, texture_stencil8, texture_lod_bias, direct_state_access); geom 7 (multi_bind, vertex_attrib_binding, transform_feedback/3, primitive_restart, buffer_objects); compat 14 (es_31_compatibility image load/store + SSBO unsizedArrayLength, shaders.negative, get_uniform, atomic_counters negative-large-offset, shader_ballot read, shader_image_load_store advanced-cast). Lists: ~/MGL-cts-new/official/q_{tess,geom,compat}.txt.

Workers since 20:08–21:12: tess on ES31 texture/image + tessellation; geom on remaining PIQ, separate_shader, program_uniform, layout_binding; compat on the compiler backlog (KHR-Single enhanced_layouts/arrays_of_arrays ~990, debug.negative_coverage, ssbo.layout, sample_variables).

**Perf (parked):** compat s41 found no recovery for the 784c480 glmark2 regression (parity with 965d88f; its rate-limit option, default off, not merged). Its explanation (mandatory copy pass; the old drawable path is OS-throttled at ~230 FPS) conflicts with today's same-machine run of d31c9ba (shadow ~11,100, desktop shadow ~3,700), so the root cause is still open. Revisit later with a direct d31c9ba vs 784c480 present-path diff.

**References:** ps5-opengl's numbers are self-reported (Mesa/Gallium on PS5), used because it is the only published per-list result for this exact suite. WineMetalGL is the closer peer (Metal, same Mac) but only its gl46 cfg1 is known (15,156 / 3 / 4,554); a full-suite run of WineMetalGL 3.7.0 here was offered to the user.

## Current state at 965d88f

**New since f8be669 (three merges, each verified from scratch on f8be669 and A/B'd against baseline-f8be669; the merged head 965d88f rebuilt and verified green; new baseline ~/MGL-cts/baseline-965d88f):**
- **compat s39 (c8afd32):** evaluators (glMap/glEvalCoord/glEvalMesh), selection and feedback modes, deep-copy push/pop attrib, display-list evaluator commands. A/B (6 rounds, medians baseline/new): ideas 3118/3121, jellyfish 11996/11990, gl41-geometry 3436/3612.
- **tess r44 (91a291a):** tessellation ring retire-latency fix and draw-size guards. A/B: ideas 3065/3120, jellyfish 11935/12043, gl41-geometry 3309/3637.
- **geom r84 (965d88f):** indexed geometry-shader draws no longer stall (element buffer marked read, not written); memoized MSL size probes. A/B: ideas 2973/3114, jellyfish 11968/11898; gl41-geometry was noisy in the 6-round run, an 8-round rerun gave 3279/3268 (−0.4%).
- All three workers have stopped and wait for the official CTS results; those failures become their next work.

**Merged after the 965d88f CTS run (gl46-port now 9f0c6f0; baseline ~/MGL-cts/baseline-9f0c6f0):**
- **tess r45 (90e51ba):** glCopyImageSubData mip-chain completeness per GL 4.6 §8.17. The code used TEXTURE_MAX_LEVEL (default 1000) as the required chain top, so most textures counted as incomplete. dEQP-GL45-ES31 copy_image 7,680 Fail → **7,835/7,835 Pass**; KHR-GL46.copy_image 324/324. glmark2 within 2% (build needed an 8+8 rerun: −0.4%).
- **compat s40 (001679a):** glBlendBarrierKHR added, plus an audit making every advertised extension's entry points resolve (this fixes the cts-runner SIGSEGV). `#version 310/320 es` shaders now keep the ES profile. On the verify build: ES3 draw_buffers_indexed 10 Crash → 76/76 Pass; ES31 compute 191/195; ssbo.layout.single_basic_type 484/484; KHR-GL46.es_31_compatibility ~296 Fail → 312 Pass / 5 Fail / 38 NotSupported; KHR-GL46.shaders unchanged. glmark2 within 2% on all eight checked scenes.
- **geom r85 (9f0c6f0):** spec-correct GL_PROGRAM_INPUT/OUTPUT enumeration for struct members of named interface blocks. dEQP-GL45-ES31 program_interface_query 4,003 Pass / 522 Fail / **71 Crash** → **4,287 / 309 / 0**; KHR-GL46.program_interface_query 43/43. glmark2 within 2%. (Cherry-picked only geom's two new commits; its branch still carried already-merged history.)
- A fresh official-equivalent re-measure on 001679a is running (it predates geom r85, so it still shows the 71 PIQ crashes) (~/MGL-cts-new/official/r001679a).

**Reference — ps5-opengl** (darkxex/ps5-opengl: Mesa/Gallium on the PS5 GPU, GPL-3.0, core profile only, "not Khronos-certified"). It claims a full `cts-runner --type=gl46` 4.6.8.1 run with 0 Fail, not submitted. Per list (Pass / NotSupported): ES3 1,325 / 0; ES31 30,866 / 382; no-context 3 / 19; KHR-Single 5,056 / **6,292**; gl46-main ×4 15,335 / 4,378 (+1 warning).
- Their KHR-Single NotSupported equals ours exactly (they don't expose subgroups or mesh shaders either), so leaving subgroups unexposed does not prevent a clean result.
- Our ES31 NotSupported (1,312) is ~930 higher than theirs — review after the re-measure.
- Our 8 no-context Fails are NotSupported for them — check.
- Their two −1 surface configs are FBOs of 64×16384 / 16384×64 — confirm MGL's headless platform matches.

**Post-port design note:** an SDL2 → MGL video driver so QEMU (`-display sdl,gl=on`) and virglrenderer can use GL 4.6 on Metal, modelled on ps5-opengl's SDL2 bridge. See project doc `claude/sdl2-mgl-bridge.md` (also ~/MGL/docs/sdl2-mgl-bridge.md). Not started: QEMU/UTM work waits until the port is done.

**Official-equivalent CTS run on 965d88f (opengl-cts-4.6.8.1, Oct 31 14:14–17:05):**
The real `cts-runner --type=gl46` died at its 14th case: MGL advertised GL_KHR_blend_equation_advanced but did not export glBlendBarrierKHR, the test called a NULL pointer, and one crash ends the whole runner. So the same 12 test runs (exact cts-runner command lines) were replayed through glcts with a driver that records a crash and continues with the next case. This is complete data, not a submittable run; submission needs one clean cts-runner run.

| Package (cases) | Pass | Fail | NotSupported | Crash | Other |
|---|---|---|---|---|---|
| **KHR-GL46 gl46-main, 64×64** (19,714) | 15,043 | 328 | 4,342 | 0 | 1 InternalError |
| KHR-GL46 gl46-main, 113×47 | 15,042 | 329 | 4,342 | 0 | 1 IE |
| KHR-GL46 gl46-main, rgba8888d24s8 64×−1 | 15,044 | 327 | 4,342 | 0 | 1 IE |
| KHR-GL46 gl46-main, rgba8888d24s8 −1×64 | 15,044 | 327 | 4,342 | 0 | 1 IE |
| *WineMetalGL 3.0.0, gl46-main cfg1 (their evidence)* | *15,156* | *3* | *4,554* | — | *1 CompatibilityWarning* |
| KHR-Single-GL46 (11,348) | 4,064 | 991 | 6,292 | 1 | — |
| dEQP-GL45-ES31 (31,248) | 16,351 | 13,498 | 1,312 | 71 | 6 IE, 10 QualityWarning |
| dEQP-GL45-ES3 (1,325) | 1,251 | 64 | 0 | 10 | — |
| KHR-NoContext gl30/40/43/45 | 4 | 8 | 10 | 0 | — |

What the failures are:
- **gl46-main (the package comparable to the old 0-Fail list):** of 328 Fails, 296 are KHR-GL46.es_31_compatibility, 13 texture_query_levels, ~19 scattered. Only 5 are cases from the old 4.6.1 list (9,034 cases), so the old "0 Fail" held for that list; the rest are tests new since 4.6.1. The earlier "0 Fail" claim applied only to the old KHR-GL46 list, never to the whole package set.
- **ES-profile compiler bug (compat, fixed on its branch c1346cb, pending merge):** shaders.c forced every shader to desktop core profile, so `#version 310 es` compiled as desktop 3.10. About 4,300 ES31 Fails (2,550 "compute shaders require … 420", 1,269 "not supported for this version", ~270 samplerCubeArray reserved word, 80 gl_HelperInvocation) plus the 296 es_31_compatibility.
- **copy_image (tess r45, READY, being verified):** glCopyImageSubData treated any texture with fewer than 1000 mip levels as incomplete → 7,680 ES31 Fails. Fixed in textures.c per GL 4.6 §8.17.
- **Crashes:** 10 ES3 = missing glBlendBarrierKHR (compat a7fa149: entry point + audit of every advertised extension's entry points); 71 ES31 = program_interface_query program_input/output struct members of interface blocks (geom 3bd6003); 1 KHR-Single enhanced_layouts.xfb_stride_of_empty_list_and_api.
- **KHR-Single Fails:** almost all enhanced_layouts (e.g. uniform_block_member_overlapping_offsets_* — compiler validation) and arrays_of_arrays_gl (compiler).
- **ES31 remainder** (after the two big bugs): "could not find target resource" 185, "Rendering failed" 141, "Image verification failed" 117, advanced-blend runtime check 90, SPIRV-Cross arrays-of-matrices in stage I/O 60, others small. To be routed after the ES-profile and copy_image merges, then re-measured.
- **NotSupported, deliberate:** KHR-Single subgroups ~6,000 (MGL does not expose GL_KHR_shader_subgroup; user decision 15:44 — leave unexposed until the Fail count is down; ~4,000 would be implementable via Metal SIMD-group ops, partitioned has no Metal equivalent) and meshShader 218.
- Results: ~/MGL-cts-new/official/r965d88f (tally: `python3 cts_tally.py r965d88f [--list Fail|Crash|NotSupported]`).

**Conformance — CTS version correction (Oct 31):**
- The CTS tree used so far (~/MGL-cts) is the old **opengl-cts-4.6.1** branch: KHR-GL46 mustpass 9,034 cases plus GTF-GL46 (1,742 cases, Khronos-member-only source, never built here).
- Current releases (latest tag **opengl-cts-4.6.8.1**) do **not** need GTF (required only before 4.6.6). The gl46 mustpass there is gl46-main, about 19.7k cases per surface config, run by `cts-runner --type=gl46` over 4 surface configs plus single-config and no-context lists.
- New tree ~/MGL-cts-new (4.6.8.1) with MGL's headless platform ported to the new framework; glcts + cts-runner build in progress. The official run on 965d88f follows.
- Old-tree KHR-Single-GL46 (6,198 cases) was started and stopped after 100 cases: 88 Pass / **12 Fail** (arrays_of_arrays_gl ×7, enhanced_layouts ×5, all GLSL compiler area). So "0 Fail" holds for KHR-GL46 only, not yet for the whole official package.

**WineMetalGL reference (metalsharp, LGPL-3.0):** release 3.7.0 (current) and 3.0.0 core libraries downloaded and checksum-verified. Their published native result (3.0.0, gl46 config 1): 19,714 cases, 15,156 Pass / 3 Fail / 1 CompatibilityWarning / 4,554 NotSupported. A CTS side-by-side is planned after MGL's official run.

### Previous state at f8be669

**New since d11d184 (three merges, each verified from scratch and A/B'd against d11d184; the merged head rebuilt and verified green):**
- **geom r82+r83 (7afedb8):** correctness fixes for the multisample image sidecar on image arrays. r82 had been rejected for an ideas −2.2% regression; r83 made the sidecar cost nothing on draws that bind no MS image (memoized gate), and ideas is now −0.4%. geom's 8 CTS groups: 546 Pass / 38 NotSupported / 0 Fail.
- **tess r43b (f492c47):** tighter claim/publish protocol for the shared tessellation ring.
- **compat s38 (f8be669):** the GLX shim had rejected every legacy compatibility context since s14 (it looked for "compat" in GL_VERSION), so glxgears could not start; it now checks GL_CONTEXT_PROFILE_MASK, and glxgears runs again on XQuartz (70–73 FPS). Also: complete glX symbol surface, registry-exact legacy enums with regression tests, client-array queries, fixed-function state round-trips.

Conformance: still 0 Fail as last measured group-by-group on 38057b4. Merges since then were checked on their own groups plus the headless suite; a whole-package run on the current tree awaits the user's go-ahead. NotSupported inventory: 67.

**New — tess r43, d11d184:** the shared tessellation command buffer's ring no longer runs out on real TCS+TES workloads (24 slots, pressure-driven keep-alive, waits on a parked buffer instead of failing), with regression tests. tess also found that its r42 TCS+TES benchmark had been measuring the TES-only program (the bench shader failed to link); fixed. No conformance change.

**New — compat s37, a3c493b:** per-draw CPU work in the compatibility path (streaming scratch rings with a 1 MB floor, lazy position read-back, uniform-location caches). compat also re-checked every remaining NotSupported case that is not a hardware limit (shader_integer_mix.prototypes, sparse ×18, texture filter_minmax ×5): all are honest, no code change. Inventory stays at 67.

**New — tess r42, 288e724:** the shared tessellation command buffer is kept alive adaptively after a ring exhaustion. No conformance change; glmark2 within noise or better.

**Not merged — geom r82** (multisample image-array sidecar correctness fixes): verified green, but two locked glmark2 runs showed a consistent ideas −2.2% (a scene that uses no multisample images), so the sidecar bookkeeping is costing the plain draw path. Sent back to geom to gate it off when no MS image is bound.

**New — compat s36, f386b99: RGB9_E5 renderbuffers are live.** Backed by RGBA16F and snapped exactly onto the 5-9-9-9 shared-exponent lattice wherever the contents are read or copied out, so readback is bit-exact. internalformat.renderbuffer.rgb9_e5 goes NotSupported → Pass; RGB9_E5 stays non-renderable as a texture attachment (honest caps). That leaves **67 NotSupported**. s36 also stops rebuilding the Metal pipeline on every blend-state change (−18% CPU on blend-flip draws).

**New since the audit below — geom r78–r81, 1121e66: multisample images are live.** Metal's texture2d_ms is read-only in MSL, so GL multisample image stores, load-after-store and atomics (32-bit-per-component formats) now go through a sidecar buffer. GL_MAX_IMAGE_SAMPLES honestly reports 4 instead of 0. geom reports the 39 multisample-image cases that were NotSupported now Pass (shader_image_size ms rows ×36, image_load_store load-ms/perSample ×2, texture_image_samples ×1). These come from geom's per-group runs; compat's whole-package audit below predates this merge. That leaves **68 NotSupported**, all hardware or design limits.

### Measured: every KHR-GL46 group, run group by group (compat s35, Oct 30)
compat ran **every group of the KHR-GL46 package** (9493 cases, more than the 9035-case mustpass list) on e8ff34b plus its polygon_offset_clamp fix, which is exactly what 38057b4 contains. Result: **0 Fail, 0 Crash.** Every non-Pass is an explained NotSupported:

| NotSupported | Cases | Reason |
|---|---|---|
| sample_variables samples_8 | 38 | Apple GPUs top out at 4× MSAA (GL_MAX_SAMPLES = 4) |
| shader_image_size ms + image_load_store load-ms + texture_image_samples | 39 | GL_MAX_IMAGE_SAMPLES was 0 at audit time — **now live and passing** after 1121e66 (sidecar) |
| ES-only by CTS design | 3 | cube_map_array non-filterable mipmaps ×2, draw_elements_base_vertex valid_active_tf |
| shader_integer_mix.prototypes | 1 | version-gated below GLSL 450 by design (the GLSL 450 instance passes) |
| internalformat.renderbuffer.rgb9_e5 | 1 | was: RGB9E5 not renderable on Apple GPUs — **now Pass** via exact emulation (f386b99) |
| post_depth_coverage_tests | 2 | extension de-advertised (like Mesa/Asahi) |

- The audit found one real Fail the estimate below had missed: **polygon_offset_clamp (3 cases)**, fixed in 38057b4.
- This was run by a worker group by group, not as one supervisor-run mustpass sweep. A single full sweep (about 1.5–2 h) would be the formal confirmation; it needs the user's go-ahead.

### Estimate (mustpass list, 9035 cases), kept for comparison
The Oct 26 sweep (r28) plus every fix merged since, each confirmed by re-running its own group.

| Result | Estimated now | Measured Oct 26 (r28) |
|---|---|---|
| **Pass** | **≈ 8928 (98.8%)** | 8672 (96.0%) |
| NotSupported | ≈ 107 | 239 |
| Fail | 0 | 116 |
| Crash | 0 | 7 |
| InternalError | 0 | 1 |

How the estimate is built:
- All 124 r28 Fail/Crash/InternalError cases are fixed (per-group CTS runs on the workers' trees, then the supervisor's full headless gate on each merge).
- 132 of the 239 NotSupported cases are now live and pass.
- GL_ARB_post_depth_coverage is no longer advertised (e8ff34b, user-approved): Metal could not do its stencil sub-case, and Mesa's asahi driver does not advertise it either. Its 2 cases are NotSupported again, so the last known Fail is gone.
- The ≈107 still NotSupported are honest hardware or design limits, or optional extensions not advertised; they are listed under "Remaining NotSupported" below.
- Every implemented case is expected to pass. Only a full sweep can confirm that.

## History

### Full sweep r28 (Oct 26, 20:34–22:15 EDT) — the last measured sweep
Library: gl46-port 9442011 + geom rounds 55–56. Results are in ~/MGL-cts/mgl-r28-sweep.

| Result | Cases |
|---|---|
| **Pass** | **8672 (96.0%)** |
| NotSupported | 239 |
| Fail | 116 |
| Crash | 7 |
| InternalError | 1 |

- The previous full sweep (r18, Oct 24) passed 8364 (92.6%), so this is +308 passes.
- 98.6% of the implemented cases pass.

### Fixed and merged since the sweep
- **compat s20, 7478e0a:** queue 21 → 0 Fail.
  - draw_elements_base_vertex
  - sample_shading 13/13
  - shader_multisample_interpolation 109/109
- **geom r55–r58, 6f9b359:**
  - All 7 sweep crashes fixed: 6 gpu_shader5 gather depth cases and texture_swizzle.functional.
  - texture_swizzle.smoke passes.
  - geometry_shader 136/136.
  - Several program_interface_query cases.
- **tess r28+r29, b7821df:**
  - shader_draw_parameters: gl_DrawID and gl_BaseVertex.
  - cull_distance.
  - gl_PerVertex normalization.
  - Struct-array uniform lookups.
  - Separable geometry-only programs.
  - Invariant on input blocks.
- **compat s22+s23, c6ded7d:** queue 13 → 0 Fail.
  - buffer_storage 9/9
  - copy_image 13/13
  - direct_state_access 371/371, which includes the RGTC InternalError.
  - texture_border_clamp 32/32
  - Also: shader_group_vote 5/5, 3 of 5 gl_spirv cases, writable multisample-image emulation, and the PSO-key aliasing fix that made test_fp64 flaky.
- **tess r30, aafc7c2:** shading_language_420pack.length_of_vector_and_matrix and tessellation_shader gl_MaxPatchVertices_Position_PointSize.
- **compat s23+s24, 1e840b4:** separable_programs_tf 2/2 (geometry_active) and gl_spirv validations 1+3.
- **geom r59–r61, 71cfe2a:** shader_image_size 10 Fail → 0 (38/0/36) and shader_image_load_store uniform-limits.
- **compat s25, 48a41ef:** gl_spirv 9/9 (tessGeom built-in decorations).
- **compat s26a, 3cd6208:** texture_buffer 16/16 (atomic_functions).
- **compat s26b, 539f74f:**
  - tessellation_shader 58/58: primitive_coverage rendered channel-reversed because the PSO fast-path key ignored TCS/TES (a regression since c6ded7d).
  - shader_storage_buffer_object 124/124: the drawfast gate now re-enumerates on an indexed respec.
- **geom r61–r65, 984b686:**
  - texture_view coherency.
  - gpu_shader5 texture_gather_offset clamp_to_edge/border 4 Fail → Pass.
  - Image-atomic OOB safety and buffer-texture atomic visibility.
  - The xfb capture GPU-page-fault fix (see Ops).
- **tess r31–r33, 83e1777:**
  - cull_distance 3/3 (functional + negative).
  - A GL_PATCHES draw with a dead default-uniform block no longer faults.
  - program_interface_query 43/43 and explicit_uniform_location 48/48 on that tree.
  - **The fails_tess queue is now empty.**
- **compat s27+s28, 7031b52** (NotSupported → live):
  - GL_EXT_shader_integer_mix: 13 Pass, plus 7/7 in shaders45.
  - GL_ARB_gpu_shader5 advertised: gpu_shader5_gl 3/3.
  - copy_image incompatible_formats_compression.
  - packed_pixels compressed_* 10/10 (RGTC/S3TC on 2D arrays).
- **geom r66–r69, f600d55:**
  - Image-atomic MSL helpers: 5 shader_image_load_store Fail → Pass (46 Pass / 3 Fail / 2 NS at that point).
  - Image atomics in the tess compute kernels; image/sampler array elements; CTS glue getMemoryLimits.
  - The packed_pixels varied_rectangle depth-stencil regression is fixed (83/83).
- **tess r34–r36, d71ce14** (NotSupported → live):
  - parallel_shader_compile 3/3.
  - shader_integer_mix 13 and gpu_shader5_gl 3/3 on the tess side (deduplicated against compat's 7031b52 strings).
  - GL_ARB_shader_ballot + GL_ARB_gpu_shader_int64 availability.
- **geom r70, 2898abe:** the test_vs_gather heap corruption is fixed.
  - The gather rewrite in program.c had a heap overflow.
  - The xfb capture now uses the specialized mgl_xfb_vertex function.
  - test_vs_gather passes 10/10 under MallocScribble=1.
- **tess r37, cff5363:**
  - shader_ballot_tests 4/4. Bitmasks, FunctionBallot and FunctionRead now Pass.
    - A format-mismatched imageStore (rgba32f image over RGBA32UI storage) used to hang the GPU; it now writes through a same-texel-size cast view.
    - The tess→GS capture import permuted primitive slots twice; that is fixed.
    - tess reproduced the hang with a small bounded headless repro before re-running the CTS group.
  - GL_EXT_texture_shadow_lod: 16 NotSupported → 15 Pass / 1 Fail.
- **geom r71–r73, eb9eaeb:**
  - **texture_view 7/7.** view_classes failed on uncompressed uploads into BPTC storage.
    - New CPU encoders for BC7 (mode 6, unorm/sRGB) and BC6H (mode 10, signed/unsigned float).
    - They were cross-checked against the VK-GL-CTS reference decoders; test_bptc is bit-exact.
  - A latent BC4/5 (RGTC) palette off-by-one in the encoder is fixed.
  - texSubImage into compressed texStorage levels encodes block-aligned.
  - getTexImage of a compressed view decodes with the view's own format.
  - Non-layered 3D image units go through a 2D shadow texture, because Metal cannot alias a 3D slice as 2D.
  - **shader_image_load_store 48 Pass / 1 Fail / 2 NS.** The remaining Fail is early-fragment-tests, a Metal platform limit.
- **tess r38, af20103:**
  - **ext_texture_shadow_lod 16/16.**
    - Apple GPUs compare a Depth16Unorm reference in fixed point, while GL compares in float.
    - DEPTH_COMPONENT16 is now stored as Depth32Float, quantized to the 16-bit lattice on upload; readback is still bit-exact.
  - GL_ARB_shader_viewport_layer_array (gl_Layer / gl_ViewportIndex from the VS/TES).
  - About 7 more NotSupported cases now Pass.
- **geom r74–r76, cceed03:** this closes the fails_geom2 queue.
  - The r72 3D-shadow refresh is gated so it costs nothing per draw, per commit or per readback when no 3D image unit is bound.
  - The stencil_texturing compute sampler-slot fix.
  - On geom's tree: texture_gather 75/75, texture_cube_map_array 25 Pass + 2 NotSupported (ES-only), stencil_texturing 1/1, texture_swizzle 4/4 in 5 consecutive runs (the smoke flake is gone), gpu_shader5 22/22.
- **geom r77, 81f5f90:**
  - texture_size_promotion fixed: alpha-less internal formats get a view swizzle that forces alpha to 1 (new test_alphaless.c).
  - max_image_uniforms was already green.
  - Fresh geom-domain sweep: 599 Pass / 45 NotSupported / 1 Fail (early-fragment-tests, Metal limit) / 1 Crash (a batch-runner artifact; the swizzle smoke passes standalone). **geom's Fail queue is effectively done**; geom has moved to the multisample NotSupported group.
- **tess r39, f973a81:**
  - **shader_image_load_store early-fragment-tests Fail → Pass** (it was thought to be a Metal limit). A fragment shader that writes memory without declaring early tests gets a late-fragment-tests PSO variant, chosen at pipeline build time and gated on a depth attachment; the cache is keyed by program and fragment MSL, so relinks are safe. **shader_image_load_store is now 49 Pass / 0 Fail / 2 NS.**
  - GL_ARB_post_depth_coverage advertised and implemented natively: post_depth_coverage_tests 2 NotSupported → 1 Pass / 1 Fail (the stencil sub-case needs shader-readable stencil, which Metal lacks).
  - GL_MAX_IMAGE_SAMPLES stays 0 (honest).
- **compat s29–s33, fb8915e — KHR_blend_equation_advanced (88 NotSupported → 88/88 on compat's tree):**
  - Framebuffer-fetch implementation of all advanced equations (compat_advblend.c), coherent and non-coherent.
  - Zero cost when unused: the ext slot is lazy (no allocation for core contexts) and the pre-draw hook is armed only by an advanced-equation setter. This fixed the per-draw regression that rejected s29 and s30 (ideas −10/−11%).
  - Coexists with tess r39's late-fragment-tests variant.
- **tess r40, 52e6276:** regression fixes.
  - clip_distance 3/3 again: the gl_PerVertex block rewrite is version-aware (fixes tess's own r28 regression).
  - The link-cache exec_clone now carries uses_draw_params, so gl_DrawID/gl_BaseVertex no longer work only on the first link of a cached program.
- **compat s34, 8c97af2 — the last 2 Fails of the r28 sweep list:**
  - direct_state_access.renderbuffers_storage{,_multisample} Fail → Pass. Renderbuffer attachments were handed to Metal in their _sRGB format, so clears and writes with GL_FRAMEBUFFER_SRGB off were sRGB-encoded. They now use the same plain-UNORM view as texture attachments, and blits are sRGB-neutral.
  - Alpha-less integer formats read the missing channel as 1 (r77 view swizzle).
- **Every Fail and Crash on the r28 sweep list is now fixed**, and the only known remaining Fail is post_depth_coverage PostDepthSampleMask (a Metal gap, below). About 136 NotSupported cases are live (advanced blend adds 88). A new full sweep would confirm; it is offered to the user, not started.

- **tess r41, e8ff34b:** GL_ARB_post_depth_coverage is no longer advertised (user-approved). Metal's post-depth coverage mask ignores the stencil test and stencil is not shader-readable, so PostDepthSampleMask could never pass; Mesa's asahi driver does not advertise the extension either. shader_image_load_store stays 49/0/2.

### Known open issues
- None known in the CTS. A full sweep would confirm.

### Remaining NotSupported (tess r40/r41 audit)
- post_depth_coverage_tests (2): GL_ARB_post_depth_coverage is de-advertised, like Mesa/Asahi (Metal gap on the stencil sub-case).
- sample_variables samples_8 rows (38): no 8x MSAA render targets on Apple-silicon Metal (GL_MAX_SAMPLES=4).
- shader_image_size ms rows (36) + image_load_store load-ms/perSample (2) + texture_image_samples (1): GL_MAX_IMAGE_SAMPLES=0 is the honest answer (MSL has no texture2d_ms write); geom is working on an MS image sidecar (r78).
- sparse_texture / sparse_texture2 / sparse_texture_clamp (15), sparse_buffer (3): no Metal equivalent.
- texture_filter_minmax (5): tess r41 verified it cannot be exact on Metal.
- internalformat.renderbuffer.rgb9_e5 (1): MTLRGB9E5Float is not renderable on Apple GPUs; **resolved in f386b99** (compat s36: RGBA16F backing quantised exactly to the RGB9_E5 lattice on readback/copy).
- ES-only by CTS design (3): texture_cube_map_array generate_mip_map_non_filterable ×2, draw_elements_base_vertex valid_active_tf.

### On worker branches
- **compat:** advanced blend is merged (fb8915e). It is now on GL_MAX_SAMPLES (24bc87e) or other compat-domain cases. History of the advanced-blend gate — **rejected twice for per-draw cost before s33:**
  - s29 (78e1fba), Oct 29 11:37: ideas −11%, jellyfish −6% vs d71ce14.
  - s30 (b310d50), Oct 29 19:15: on the combined tree eb9eaeb + s30, ideas −10% (3250 → 2915) and jellyfish −6.4% vs eb9eaeb. Correctness was green both times.
  - compat's own A/B claimed +1.1% / +2.0%. Its method differed: one scene per process, a /tmp data copy, and a tree without eb9eaeb. It was told to reproduce with the supervisor's method.
  - compat was paused Oct 30 01:35–04:57 under the load rule and resumed by the user. Its GL_MAX_SAMPLES change (24bc87e) may come as a separate READY.
- **geom:** the multisample NotSupported group (taken over from compat).
- **tess:** more NotSupported cases.

NotSupported 239 (r28):
- Optional extensions: blend_equation_advanced 88 (now live, fb8915e), ext_texture_shadow_lod 16 (now 16/16).
- Multisample sample-variable and image-size limits (about 74; geom).
- Sparse textures and sparse buffers, which Metal has no equivalent for.
- integer_mix, gpu_shader5_gl, parallel_shader_compile, shader_ballot and shader_viewport_layer_array are now live.

## Performance: glmark2-macos through the NSOpenGL interposer
- **MGL vs WineMetalGL (Oct 31, full default glmark2, same machine, under the lock).** WineMetalGL was driven natively (no Wine) through a small NSOpenGL interposer onto its wmgl_* driver interface (~/MGL-cts/wmgl/shim, not in the repo). Both got a core/compat 4.6 context and the same 800x600 depth-32 surface.
  - Scores: WineMetalGL 3.0.0 **7467**, WineMetalGL 3.7.0 **7298** (noisy run), MGL **10902**.
  - MGL ahead: simple scenes about 2× (vs 3.0.0), jellyfish 11954 vs ~7100, ideas 3058 vs 2283–2709, gl41-geometry 3434 vs ~1300, conditionals/function/loop and gl41 instancing/pipeline/texarray 12–29%.
  - MGL behind (isolated reruns, MGL vs WineMetalGL 3.7.0): desktop blur 1447 vs 5997, desktop shadow 785 vs 6969, shadow 6336 vs 10897, gl41-mrt 7311 vs 9374. WineMetalGL's gl41-streaming-sync hits GL_TIMEOUT_EXPIRED (its ~1000 FPS is not real).
- **glmark2 regression found (Oct 31, after the user noticed MGL "used to score ~14000"):**
  - Today's full score is 10902 vs 13.6–14.4k on Oct 25.
  - Code part, bisected to **784c480** (compat s13–s16: per-present force-opaque compute pass, opaque per-context present layers, jellyfish flash fix). Same session, d31c9ba → 784c480: desktop shadow 3735/3661 → 1060/879 (−72%), shadow 11114/10874 → 7279/7278 (−34%), build −5..−15%, jellyfish −5%. Every later build is flat with 784c480. Owner: compat (present path). The goal is to keep the opaque/no-flash correctness and cut the per-frame cost; timing is the user's call.
  - The other ~20% is environment: 159d900 (the Oct 25 head) measures build 14.9k today (was ~19–20k on Oct 25). An external 1920×1080 @ 60 Hz display is attached now.
  - The merge A/B covered only ideas, jellyfish and gl41-geometry, which is why this slipped through. From now on the A/B also covers build, desktop blur, desktop shadow and shadow, and a full default score is part of judging a merge.
- **MGL vs native Apple GL 4.1** (Oct 27–28, same machine, back-to-back runs under the lock):
  - ideas: native 3274 vs MGL 3287 (Oct 28: 3180 vs 3196), even.
  - jellyfish: native about 5250 vs MGL about 12760–15100 (2.4–2.9×).
  - gl41-geometry: native about 520 vs MGL about 3360–3650 (6.5–7×).
  - 3-scene score: MGL about 6600–7300, native about 2700–2900.
- **Floor:** a clean full default glmark2 score below 6000 is a failure.
- **Merge gate:**
  - An interleaved A/B against ~/MGL-cts/baseline-<sha>. No scene may be more than 2% below.
  - Close calls get a single-scene 8-round rerun (ab_g55_one.sh for gl41-geometry, ab_g55_jf.sh for jellyfish), or a second full 6-round run.
  - Every glmark2 run takes /tmp/mgl_glmark2.lock. The ab scripts have no lock of their own, so launch them under lockf. Workers' own glmark2 runs take the same lock, so an A/B may wait behind them.
  - Each A/B log must show "mgl_nsgl: NSOpenGL -> MGL interposer active" and GL 4.6; otherwise the dylib didn't load.
  - The binary is /opt/homebrew/bin/glmark2-macos with --data-path /opt/homebrew/share/glmark2. The scene order in ab_g55.sh is ideas, jellyfish, gl41-geometry, in one process. A "[scene]" log line appears when that scene starts.
  - Visual check: capture the glmark2 WINDOW only (~/MGL-cts/vscripts/winid + screencapture -l), never the full screen. The user is often working on the Mac. A capture slows the run it lands in, so discount that run.
- **Pre-READY rule (Oct 27 21:30, approved by the user):**
  - Each worker runs tests/headless/ab_bench mode 0 (2000 draws) and mode 1 (blend-flip ×600) against the baseline libmgl, minimum of 7, interleaved.
  - The limit is +3%, and the numbers go in the READY line.
  - Draw-path changes also need a locked glmark2 ideas/jellyfish check.
  - Reason: on Oct 27 two READYs that passed every correctness test were rejected for per-draw regressions (tess −42%, compat −63% on ideas). On Oct 29 compat's advanced blend was rejected twice the same way (ideas −11%, then −10%).
  - ab_bench wall-clock numbers are noisy under multi-agent load. MGL_PROFILE bucket parity is accepted as supporting evidence, and the supervisor's glmark2 A/B decides.
- **b7821df vs 6f9b359:** ideas −1.3%, jellyfish −1.5%, gl41-geometry +0.9%; score 6705.
- **c6ded7d vs b7821df:** ideas +0.3%, jellyfish −1.3%, gl41-geometry +0.6%; score 6688.
- **aafc7c2 vs c6ded7d:** ideas 3290 → 3290, jellyfish +0.7%, gl41-geometry even; score 6675 → 6703.
- **1e840b4 vs aafc7c2:** ideas 3232 → 3258, jellyfish −1.7%, gl41-geometry 3350 → 3354 (8-round rerun).
- **71cfe2a vs 1e840b4:** ideas +0.9%, jellyfish −0.8%, gl41-geometry −1.3%; score 6894 → 6841.
- **48a41ef vs 71cfe2a:** ideas 3192 → 3275, jellyfish −0.3%, gl41-geometry +2.1% (rounds 1–5); full score 6936.
- **3cd6208 vs 48a41ef:** ideas 3294 → 3292, jellyfish +1.7%, gl41-geometry +1.2%; score 6824 → 6917.
- **539f74f vs 3cd6208:** ideas −1.65% (about 1.5% lower in two runs; inside the gate), jellyfish −0.1%, gl41-geometry +1.4%; score even.
- **984b686 vs 539f74f:** ideas +1.6%, gl41-geometry +0.9%, jellyfish 13286 → 13454 (+1.3%, 8-round rerun after a noisy −2.7%).
- **83e1777 vs 984b686:** ideas +0.8%, jellyfish −1.6%, gl41-geometry 3368 → 3360 (−0.2%, 8-round rerun after one outlier round).
- **7031b52 vs 83e1777:** ideas −0.6%, jellyfish −0.7%, gl41-geometry +0.9%; score 6894 → 6866.
- **f600d55 vs 7031b52:** ideas −0.2%, jellyfish −0.5%, gl41-geometry −0.6% (6-round medians); scores 7053–7334.
- **d71ce14 vs f600d55:** ideas −0.4%, jellyfish −0.7%, gl41-geometry +1.2% (6-round medians); scores 6975–7148.
- **compat s29 (rejected) vs d71ce14:** ideas −11%, jellyfish −6%, gl41-geometry even; scores 6653–6881 vs 7223–7533.
- **2898abe vs d71ce14:** ideas −0.4%, jellyfish +1.7%, gl41-geometry +1.3% (6-round medians); scores 7196–7306.
- **cff5363 vs 2898abe:**
  - ideas +1.0%, jellyfish +0.3%.
  - gl41-geometry read −3.3% in the noisy 6-round set; the 8-round rerun showed 3557 vs 3566 (−0.2%).
  - Scores 7117–7338.
- **eb9eaeb vs cff5363 (two 6-round runs):**
  - ideas −1.2% / −0.9%, jellyfish −0.8% / −1.3%, gl41-geometry +0.2% / +0.7%.
  - One C run was hit by outside load (jellyfish 11053, score 5770). Every other score was 7118–7306.
- **compat s30 (rejected) vs eb9eaeb:** ideas −10%, jellyfish −6.4%, gl41-geometry even; scores 6751–6876 vs 7174–7372.
- **af20103 vs eb9eaeb:** ideas +0.4%, jellyfish +0.4%, gl41-geometry −0.8% (medians); scores 7241–7320.
- **cceed03 vs af20103:**
  - ideas +0.6%, jellyfish +1.2%, gl41-geometry even (5-round medians).
  - Both sides ran about 4% lower in absolute terms because all 3 workers were busy; the same conditions applied to P and C.
- **81f5f90 vs cceed03:**
  - ideas +0.3%, jellyfish −0.6% (6-round medians).
  - gl41-geometry read −2.2% in the 6-round set (one C run overlapped the window capture); the 8-round rerun showed 3554 → 3531 (−0.65%).
  - Scores 6343–6841 (all 3 workers busy).
- **f973a81 vs 81f5f90:**
  - ideas +1.5%, jellyfish +0.7%, gl41-geometry +1.1% (6-round medians).
  - Rounds 1–3 were hit by outside load on both sides (P3 5786, C2 5817); the clean rounds 4–6 alone are even to +1.3%, scores 6708–6840.
  - A window capture taken about 4 s into a fresh glmark2 process showed a stale grid frame; a recapture during the next C run showed ideas correctly. Capture at least 3 s into a scene.

- **fb8915e vs f973a81 (compat s33 advanced blend):**
  - ideas −1.2% (−0.6% excluding the window-capture round), jellyfish −0.35%, gl41-geometry −0.4% (6-round medians); scores 6712–7380.
  - The s29/s30 attempts had cost ideas −10/−11%.

- **52e6276 vs fb8915e (tess r40):** ideas −1.0%, jellyfish −0.9%, gl41-geometry −0.3% (6-round medians); scores 6468–7221.

- **8c97af2 vs 52e6276 (compat s34):**
  - The first A/B (19:49) was spoiled by outside load (worker CTS + ab_bench, WebKit, ReportCrash): the baseline itself dropped to 5800–5910 in 4 of 6 rounds. Not judged.
  - Clean rerun (20:11): ideas −1.25%, jellyfish +0.2%, gl41-geometry +0.35% (6-round medians); scores 6512–6994.

- **e8ff34b vs 8c97af2 (tess r41):** ideas +0.3%, jellyfish −0.8%, gl41-geometry −0.8% (6-round medians); scores 6144–6799.

- **38057b4 vs e8ff34b (compat s35):** ideas −0.3%, gl41-geometry +1.0%, jellyfish −0.9% over rounds 2–6 (−1.5% including the window-capture round); scores 6142–6639.

- **compat s38 vs d11d184:** ideas +2.2% (run 2) / +1.9% pooled, jellyfish −0.3% / −0.2%, gl41-geometry −0.5% (8-round rerun; 6-round runs showed −3.2% / −2.6% pooled with outlier rounds while worker GPU tests ran). Jellyfish renders correctly.
- **tess r43b vs d11d184:** ideas +0.8%, jellyfish +1.6%, gl41-geometry +6.8% (baseline side slowed by a concurrent build; no scene below). Jellyfish renders correctly.
- **geom r83 vs d11d184:** ideas −0.4%, jellyfish −0.3%, gl41-geometry −0.1% (r82 alone was ideas −2.2%, rejected).
- **d11d184 vs a3c493b (tess r43):** ideas −0.3%, jellyfish −0.2% (rounds 2–6 medians); gl41-geometry +0.2% (8-round rerun; the 5-round run showed −2.4% with one outlier round). Jellyfish renders correctly.
- **a3c493b vs 288e724 (compat s37):** ideas +0.3%, gl41-geometry +0.3% (5-round medians); jellyfish +0.2% (8-round rerun — a 5-round run under outside load showed −2.4%). Jellyfish renders correctly.
- **288e724 vs f386b99 (tess r42):** ideas +1.2%, jellyfish +4.1%, gl41-geometry +1.1% (rounds 2–6 medians); jellyfish renders correctly.
- **geom r82 vs f386b99 (rejected):** ideas −2.4% / −2.2% in two runs (pooled −2.2%), jellyfish −1.5% pooled, gl41 +1.2% pooled.
- **f386b99 vs 1121e66 (compat s36):** ideas +1.2%, jellyfish −0.4% (6-round medians, rounds 2–6); gl41-geometry −0.3% (8-round rerun; a single 6-round run showed −5.9%, which was noise). Blend-flip draws −18% CPU (compat ab_bench). jellyfish and gl41-geometry render correctly.
- **1121e66 vs 38057b4 (geom r78–r81, MS image sidecar):** ideas +0.2%, jellyfish +1.6%, gl41-geometry −0.8% (6-round medians); scores 6153–6500; all three scenes render correctly.

## Merged
- **Oct 31, 9f0c6f0 — geom r85:** program_interface_query in/out enumeration for interface-block struct members (71 ES31 crashes → 0)
- **Oct 31, 001679a — compat s40:** glBlendBarrierKHR + advertised-extension entry-point audit (cts-runner segfault); ES-profile shaders keep ES semantics
- **Oct 31, 90e51ba — tess r45:** glCopyImageSubData mip-chain completeness (dEQP-GL45-ES31 copy_image 7,680 Fail → Pass)
- **Oct 31, 965d88f — geom r84:** indexed GS draw stall fix, memoized MSL size probes
- **Oct 31, 91a291a — tess r44:** ring retire latency fix, draw-size guards
- **Oct 31, c8afd32 — compat s39:** evaluators, select/feedback, push/pop-attrib deep copy, display-list evaluator commands
- **Oct 31, f8be669 — compat s38:** GLX shim compat-context fix (glxgears works again), glX symbol surface, registry-exact legacy enums + tests, client-array queries, FFP state round-trips
- **Oct 31, f492c47 — tess r43b:** tessellation ring claim/publish protocol
- **Oct 31, 7afedb8 — geom r82+r83:** MS image-array sidecar fixes, zero cost when unused
- **Oct 31, d11d184 — tess r43:** tessellation ring-exhaustion fix + regression tests
- **Oct 31, a3c493b — compat s37:** per-draw CPU in the compatibility path; NotSupported triage (all remaining non-hardware NotSupported honest)
- **Oct 31, 288e724 — tess r42:** adaptive AGX keep-alive for the shared tess command buffer
- **Oct 31, f386b99 — compat s36:** exact RGB9_E5 renderbuffer emulation (1 NotSupported → Pass), pipeline fast-path covers blend/alpha state
- **Oct 31, 1121e66 — geom r78–r81:** multisample image sidecar, GL_MAX_IMAGE_SAMPLES 0 → 4 (39 NotSupported → Pass)
- **Oct 31, 38057b4 — compat s35:** polygon_offset_clamp 3/3 (per-format depth offset, native one-sided clamp); full KHR-GL46 group audit, 9493 cases, 0 Fail
- **Oct 30, e8ff34b — tess r41:** GL_ARB_post_depth_coverage no longer advertised (removes the last known Fail)
- **Oct 30, 8c97af2 — compat s34:** sRGB renderbuffer attachments (the last 2 r28-sweep Fails), alpha-less integer formats
- **Oct 30, 52e6276 — tess r40:** clip_distance 3/3 again (version-aware gl_PerVertex rewrite), link-cache keeps draw-parameter use
- **Oct 30, fb8915e — compat s29–s33:** KHR_blend_equation_advanced (88 NotSupported → live), zero-cost when unused
- **Oct 30, f973a81 — tess r39:** early-fragment-tests via late-tests PSO variant (shader_image_load_store 49/0/2), ARB_post_depth_coverage
- **Oct 30, 81f5f90 — geom r77:** texture_size_promotion (alpha-less view swizzle), test_alphaless; geom-domain sweep 599/45/1
- **Oct 30, cceed03 — geom r74–r76:** zero-cost 3D-shadow gating, stencil_texturing compute sampler-slot fix; closes fails_geom2
- **Oct 29, af20103 — tess r38:** ext_texture_shadow_lod 16/16 (D16 as Depth32Float), ARB_shader_viewport_layer_array, more NotSupported cases live
- **Oct 29, eb9eaeb — geom r71–r73:** texture_view 7/7, shader_image_load_store 48/1/2, BPTC/RGTC CPU encoders, non-layered 3D image units
- **Oct 29, cff5363 — tess r37:** shader_ballot 4/4, GL_EXT_texture_shadow_lod 15/16, tess→GS capture order fix
- **Oct 29, 2898abe — geom r70:** xfb capture hardening, gather-rewrite heap overflow fix (test_vs_gather)
- **Oct 29, d71ce14 — tess r34–r36:** integer_mix, parallel_shader_compile 3/3, gpu_shader5_gl, ballot/int64 availability
- **Oct 29, f600d55 — geom r66–r69:** image-atomic helpers, tess-kernel image atomics, image/sampler array elements, getMemoryLimits, packed_pixels depth-stencil fix
- **Oct 29, 7031b52 — compat s27+s28:** GL_EXT_shader_integer_mix, GL_ARB_gpu_shader5, compressed 2D-array copy/upload
- **Oct 29, 83e1777 — tess r31–r33:** cull_distance 3/3, GL_PATCHES dead-uniform fault fix
- **Oct 29, 984b686 — geom r61–r65:** xfb capture GPU-fault fix, texture_view coherency, gather_offset clamp, image-atomic safety
- **Oct 29, 539f74f — compat s26b:** PSO fast-path key includes TCS/TES (tessellation_shader 58/58), drawfast re-enumeration (SSBO 124/124)
- **Oct 29, 3cd6208 — compat s26a:** texture_buffer.atomic_functions, fixed ab harness
- **Oct 28, 48a41ef — compat s25:** gl_spirv 9/9, TES clip/cull capture for a following GS
- **Oct 28, 71cfe2a — geom r59–r61:** shader_image_size 10 → 0, image load/store uniform limits, sep-vs kernel rebuild on link-cache hits
- **Oct 28, 1e840b4 — compat s23+s24:** separable_programs_tf geometry_active, gl_spirv 1+3, xfb GS-stage priority, sep-vs kernel caching on link-cache hits
- **Oct 28, aafc7c2 — tess r30:** 420pack length_of_vector_and_matrix, gl_MaxPatchVertices_Position_PointSize
- **Oct 27, c6ded7d — compat s22+s23:** buffer_storage/copy_image/DSA/border_clamp, shader_group_vote, gl_spirv, MS-image writes, fnid-registry PSO key
- **Oct 27, b7821df — tess r28+r29:**
  - draw parameters, cull distance, gl_PerVertex, GLSL compat
  - O(1) link-time per-draw gates
- **Oct 27, 6f9b359 — geom r55–r58:**
  - link cache
  - texture_swizzle.smoke
  - 7 crashes fixed
  - geometry_shader 136/136
  - PIQ, indirect draws, GS invocations
- **Oct 27, 7478e0a — compat s20:** CTS queue 21 → 0 Fail
- **Oct 26, 6ee46f0 — compat s19:** ideas +16.6%
- **Oct 26, 9442011 — compat s18:** single-pass present copy plus per-draw CPU cuts
- **Oct 26, 5178835 — tess r25:** vertex_attrib_64bit, xfb POINTS, gl_VertexID
- **Oct 26, 784c480 — compat s13–s16:** jellyfish caustics and flash fix, per-context layers, honest GL strings
- **Oct 26, d31c9ba / 9485327 — tess r23/r24:** 23 regressions fixed, plus 9 more Fail→Pass

## Ops
- 3 workers, -j4, nice + background QoS. Load limit 12; compat pauses first.
  - Oct 28 03:21–03:37: compat was paused under the load rule. The load came from macOS contactsd/WebKit/VM/Docker, not MGL. The user resumed it.
  - Oct 28 07:16–22:31: all work was paused at the user's request.
  - Oct 29 07:23–08:36: compat was paused under the load rule (5/15-min load 16.8/13.3, from MGL builds and tests), then resumed by the user.
  - Oct 30 01:35–04:57: compat was paused under the load rule (5/15-min load 14.2/12.5: a glmark2 A/B, a geom test and a CTS run), then resumed by the user.
  - Oct 30 05:04: the user lifted the load limit until 07:30 EDT.
  - Oct 30 ~08:40–16:57: the Mac was offline. On wake the load hit 67/78/53 (contactsd, AddressBookSourceSync); compat was paused under the load rule and resumed by the user at 16:58 (relaunched fresh; its session had died at 08:31).
- **Mac offline, Oct 29 22:24–Oct 30 00:44:** the Mac slept or lost its network. All 3 workers crashed with APIConnectionError, and the supervisor link was down from 22:41.
  - On wake, the load spiked to 49 from macOS catch-up (contactsd, knowledge-agent).
  - The user said "resume" at 00:46, and all 3 workers were relaunched with fresh sessions.
- **Hung workers on dead API connections (Oct 29):**
  - Worker sessions stalled with a half-closed socket (CLOSE_WAIT), no child process and no transcript progress. Restarts:
    - compat at 08:54, 13:07 and 15:44.
    - tess at 12:46, 13:20 and 21:16.
    - geom at 13:29 and 20:54.
  - Fix 37ca24b: agents/llm_client.py sets a per-request timeout (MGL_LLM_TIMEOUT). 332c47b lowered the default from 420 s to 180 s.
  - The timeout does not always catch it: geom stalled 18 min at 20:35 with the 180 s timeout active, and the SDK's internal retries are silent.
  - Rule: restart a worker when its transcript is more than 20 min old, it has a CLOSE_WAIT socket, and it has NO child processes. A worker running its own long gate (run.sh) is not hung.
- Workers also end on "max turns" or "done" with no process left; check pgrep and relaunch.
- READY detection: the READY line is near the top of agents/<domain>_report.md (tess puts it on line 3); grep the first ~400 bytes.
- Every READY is verified from scratch: build, full headless suite 0 FAIL, test_geom 11× 0 failures, fp64 4× 298/0.
- When two READYs land back to back, or when the branch lacks the newest gl46-port commits, the combined tree is verified from a temporary commit (git merge-tree + commit-tree) before it becomes a baseline.
- Verify scripts live in ~/MGL-cts/vscripts with symlinks in /tmp. Recreate the symlinks after a reboot.
- Squash merges only, with scratch tests excluded (zz*, scratch_*, test_zz_*, *probe*, bench_scratch.c). Workers must not commit scratch files. Strip counts: f600d55 about 300, d71ce14 52, 2898abe 317, cff5363 57, eb9eaeb 317, af20103 63, cceed03 325, 81f5f90 337 (5 real files kept), f973a81 67.
- Workers must not attach debuggers (lldb/debugserver). That raises a "Developer Tools Access" password prompt on the user's screen and hangs.
- **GPU page faults: test_geom_xfbperf (Oct 28–29), FIXED in 984b686.**
  - About 12 faults between Oct 28 06:08 and Oct 29 00:30, from compat probes, then from the test in every tree (including the supervisor's own verify). Every fault reset the GPU for the whole machine.
  - **Root cause (geom r64):** the xfb capture pass draws the prep kernel's capture list as an index buffer. Each record lands at (index − vmin) × in_words, but the record pool was sized by the index COUNT. A sparse indexed draw wrote up to 3× past its pool slice.
  - **Fix:** size the pool by max(index count, vertex range); add a per-draw capacity clamp in the capture variant; add MGL_GEOM_ASSERT=1 CPU-side sizing proofs.
  - The quarantine ran from Oct 28 23:42 to Oct 29 03:33. It was lifted after a full-suite verify that included the test with no fault; every verify since has been clean.
  - Rule: never re-run a binary that faults the GPU; check `log show` for GPU Address Fault / GPU Hang before blaming a candidate for collateral failures.
- Oct 27 interruptions:
  - The Mac rebooted at 15:45, after about 1.5 h of API connection errors.
  - The network was out from about 16:45 to 20:25. The workers survived it.
