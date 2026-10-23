READY 1fe21a6 (branch gl46/geom tip: merge of gl46-port 305b968 + image-unit work 8787fdf/58c91fc/bb9cc5c).

## Round 41 — image units (shader_image_size cluster) + encoder-conflict crashes

### Tally (results dirs on the Mac: ~/MGL-cts/geom41-*)
| cluster | round-40 | round-41 |
|---|---|---|
| KHR-GL46.shader_image_size (74) | Fail=20 NotSupported=54 Pass=0 | **Pass=50, Fail=0, NotSupported=24** (ims3 + ms-gs fix) |
| KHR-GL46.shader_image_load_store (51) | Pass=12 Fail=19 NotSupported=18 Crash=2 | **Pass=23 Fail=22 NotSupported=6 Crash=0** (ils2) |

(shader_image_size re-tally after the last fix: advanced-ms-gs x3 pass in one.sh spot checks; the 50/24 split above is ims3 + those 3 — NotSupported = 24 tcs/tes cases, see gaps.)

Headless suite green after merging gl46-port 305b968: test_geom 593/593, test_tess 499/499, test_smoke, full run 442 PASS / 0 FAIL.
stub_scanner EMPTY/NOOP/ASSERT: clean for MGL/src/geometry.c, MGL/src/MGLRenderer+Geometry.m.

### What was fixed (all commits mine unless noted)
1. **Image-unit level/layer views** — `geomImageUnitView` (MGLRenderer+Geometry.m, declared in
   MGLRenderer_Private.h): glBindImageTexture binds ONE mip level (and one layer when
   non-layered on an arrayed target); render path (MGLRenderer.m), compute path
   (bindTexturesToComputeEncoder), geometry kernel + capture walks now hand the MSL a
   properly sliced view. 3D layered views use the LEVEL's depth (base depth overflows Metal
   validation). Fixes every imageSize/imageStore at level>0.
2. **Compute dispatch encoder crash (9 CTS aborts)** — extBufEncodeDispatch (MGLRenderer+Draw.m)
   realizes the compute stage's textures BEFORE the compute encoder exists (deferred
   texdataEncodeUpload opened a blit encoder mid-encoding). Same class as the geom one below.
3. **Geometry kernel encoder crash** — the texture pre-realize walk ran with STATE(program)==NULL
   on pipeline draws; the first deferred upload then fired from geomBindStageBuffers INTO the
   open compute encoder. STATE(program) is switched around the walk now.
4. **Geometry kernel MSL fixes**: rect (image2DRect) → 2D SPIR-V patch applied (program.c's
   mgl_program_spirv_rect_to_2d, now exported); native texture_buffer MSL type; decompiled GS
   source (SPIRV-Cross GLSL backend) re-declares the packed default-uniform block with
   layout(location) members — illegal for uniform blocks, stripped (subroutine selectors).
5. **Kernel texture slots** — the generated kernels' image/sampler Metal indices come from
   SPIRV-Cross AFTER compilation (get_automatic_resource_binding, recorded into
   gp->tex_slots_gs / gp->tex_slots_vs in geom_spirv_to_msl): the flat pin numbering can be
   unreachable (user image at SPIR-V binding 0 collides with the DUB block's
   MSLResourceBinding key). The bind walks look the slot up by resource name.
6. **Compute path indices** — bindTexturesToComputeEncoder bound image/sampler units at the GL
   unit index; SPIRV-Cross numbers MSL args by declaration order. Now binds at the resource's
   refined msl_binding (and expands arrayed resources over consecutive slots/units).
7. **Raster path image arrays** — image2D g[2] spans array_size slots fed by units base+N-1
   (like sampler arrays).
8. **Rasterizer discard** — the GS kernel now runs under discard when anything can observe it
   (xfb, active pre-raster query, storage image/SSBO/atomic in the GS: geom_discard_skippable);
   the raster pass is also skipped when the program has no FS (Metal cannot build that PSO:
   rasterizationEnabled=NO needs a void vertex function).
9. **Limits** — GL_MAX_VERTEX_IMAGE_UNIFORMS 0→8 (VS images verified end-to-end),
   GL_MAX_IMAGE_SAMPLES 0→4 (image2DMS/image2DMSArray + imageSamples verified). TCS/TES stay 0.
10. **negative-compileTime** — glslang's "extension ... is being used" feature warnings become
    compile errors (shaders.c, only for the exact GL_ARB_shader_image_size feature string so
    the GL_EXT_spec_constant_composites noise stays a warning).

### Shared-file edits (surgical, all in my commits)
* MGL/src/MGLRenderer.m — render-path storage-image loop: level/layer view + array expansion;
  compute path: pre-realize n/a here, msl_binding indices + image-unit views + array expansion.
* MGL/src/MGLRenderer_Private.h — +1 declaration (geomImageUnitView).
* MGL/src/MGLRenderer+Draw.m — pre-realize compute stage textures before the compute encoder.
* MGL/src/get.c — limits (8) and (9).
* MGL/src/program.c — exported mgl_program_spirv_rect_to_2d wrapper.
* MGL/src/shaders.c — (10).
* MGLRenderer+Geometry.m, geometry.c, mgl_ext_geom.h — my files (slot tables, views, strip).

### Known gaps / next round
* **shader_image_load_store 22 Fails**: distinct causes; "After test execution glGetError()
  returned GL_INVALID_OPERATION" (basic-allTargets-store/atomic/atomicCS/atomicVS/store,
  advanced-sso-atomicCounters — trace shows mglUseProgram(0x502) then glProgramUniform* failing
  at the END of the test, after texture cleanup — likely name-reuse or a stale error flag);
  "Test Run() failed" (advanced-cast, basic-glsl-misc, early-fragment-tests,
  incomplete_textures, multiple-uniforms, negative-bind, uniform-limits, basic-allTargets-
  atomicGS, advanced-sync-*). Scratch repros for image paths: tests/headless/zz_ms.c (compute
  MS images — PASSES), zz_msgs.c (GS+subroutine+MS — PASSES), zz_adv.c (advanced bindings).
* **non-layered 3D image bindings** (imageLoadStore non-layered_binding, image_size
  advanced): a 3D texture can only alias as 3D (Metal view rules) — MGL hands the whole
  texture, so an image2D binding of a 3D slice reads the wrong plane (only remaining fail of
  that case). Proper fix = bind a per-draw 2D proxy + blit, or a SPIR-V coordinate rewrite.
* **TCS/TES image uniforms (24 NotSupported)**: the tess kernels (MGLRenderer+Tessellation.m
  tessBindStageTextures) bind only sampled images; adding STORAGE_IMAGE binding there
  (image_units + geomImageUnitView + sres->msl_binding) would flip the limit to 8. Local repro
  tests/headless/zz_ti.c: shaders link, kernel runs, but units are never bound (sizes read 0).
  Also: glTexStorage2D after a tess program link sets a stale GL_INVALID_OPERATION (the
  "not sure how we got here" path at MGLRenderer.m:~880 fires twice for tess-internal
  textures) — worth fixing before flipping the limit.
* **Clusters not started this round**: KHR-GL46.compute_shader (42: Fail=14 Crash=4 at round
  32 — re-tally first), KHR-GL46.vertex_attrib_binding (30: Fail=18; may share the vattrib
  plan fixed in round 40).
* glslang noise: "GL_EXT_spec_constant_composites is being used for spec constant aggregate
  constructor" warnings appear in many CTS shader logs (harmless, kept as warnings).

### Test artifacts
* Committed: tests/headless/test_geom.c unchanged this round (593 checks green).
* Untracked scratch (kept out of commits per rules): tests/headless/zz_gis.c zz_gis2.c
  zz_adv.c zz_vis.c zz_neg.c zz_tb.c zz_ctb.c zz_ms.c zz_msgs.c zz_ti.c zz_dc.c — the useful
  ones to promote next round: zz_ms.c, zz_msgs.c, zz_adv.c, zz_ti.c.
