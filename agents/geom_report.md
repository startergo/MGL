READY bb97080 (branch gl46/geom tip: merge of gl46-port 63acd8e + round-42 work 7ef58a3/76752ca/bb97080; headless suite 450 PASS / 0 fail, test_geom 10x 593/593).

## Round 42 — atomic counters, TCS/TES storage images, vertex-only pipeline rule

### Done
1. **GL atomic counters (atomic_uint) work in the vertex / fragment / compute stages**
   (commit 7ef58a3).  SPIRV-Cross's MSL backend declares no argument for GL
   atomic-counter variables — every stage using one failed Metal compilation
   (root cause of `advanced-sso-atomicCounters`' "stale GL_INVALID_OPERATION": the
   failed link made glUseProgramStages error and the CTS never consumed it).
   `program.c` collects the resources (name + SpvDecorationOffset) per stage;
   `geometry.c`'s `mgl_msl_inject_atomic_counters()` appends
   `volatile device atomic_uint* NAME [[buffer(N)]]` to the entry function and
   rewrites `(thread atomic_uint*)&NAME` uses to the device pointer at offset/4.
   N = the ordinal the renderer's buffer map binds the atomic group at (after
   uniform blocks / DUB / SSBOs, below the tess/xfb pinned indices 24..30).
   `GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS` 0 -> 8 (tess/GS stay 0: hand-authored
   kernels).  Verified VS+FS+CS incl. SSO pipelines: `test_atomic_counters.c`
   (15 checks).  Notes: strips run the vertex fn per triangle on Apple GPUs
   (verified vs raw Metal — exact increment counts checked with GL_TRIANGLES);
   CPU readback of a texture buffer is not coherent after pipeline image writes
   (verified through the FS instead).
2. **TCS/TES storage images** (76752ca): `tessBindStageTextures` gained a
   storage-image loop (image units, level/layer view via geomImageUnitView,
   MSL slot from reflection) for the TCS compute kernel; the render-path loop
   bound the TES's images on the *fragment* texture table — the first stage of
   that loop is always a vertex-side function (VS or post-tess TES), now
   `setVertexTexture`.  `GL_MAX_TESS_*_IMAGE_UNIFORMS` 0 -> 8.
   `test_tess_images.c` (6 checks) covers monolithic draws incl. rasterizer
   discard.  gl_PrimitiveID in the TES is the emulated patch draw's instance id
   (0 unless per-patch instanced) — documented in the test.
3. **Vertex-only pipeline rule** (bb97080): Apple GPUs require rasterization
   disabled exactly for *void* vertex functions, enabled otherwise.  The old
   code always disabled rasterization for FS-less pipelines, which failed PSO
   creation for any vertex-only program that assigns gl_Position; my first fix
   (always rasterize with a cached empty fragment function,
   `geomEmptyFragmentFunction`) broke the opposite shape (the CTS *-vs
   programs are genuinely void).  `mgl_msl_vertex_returns_void()` parses the
   MSL return type and picks the correct variant.
4. **SSO tess record lookup** (bb97080, 2-line surgical edit in
   tessellation.c): a bound pipeline installs its COMPOSITE as state.program
   (name 0); `mgl_tess_active_program` then never consulted the pipeline's
   stage programs, so every SSO separate-TCS/TES draw built its TES render pass
   with the plain vertex function and SEGV'd (repro `zz_tcp2.c`,
   image-independent).  Fixed by resolving name-0 composites through the
   pipeline branch.  **Heads-up for the tess worker: this is a real pre-existing
   crash your round-13 code still has; my hunk is at `mgl_tess_active_program`
   (keep it when merging 4b5e0d1).**

### Shared-file edits (all surgical)
- `program.c`: atomic-counter collection + injection call + msl_binding patch.
- `MGLRenderer.m`: vertex_only block (empty-FS / void rule); storage-image loop
  vertex-table fix.
- `MGLRenderer_Private.h`, `MGLRenderer+Geometry.m`: `geomEmptyFragmentFunction`
  (cached, `newLibraryWithSource` once).
- `MGLRenderer+Tessellation.m`: storage-image loop in tessBindStageTextures
  (binding table only, as instructed).
- `tessellation.c`: the 2-line `name != 0` composite guard (see 4).
- `get.c`: GL_MAX_VERTEX_ATOMIC_COUNTER_BUFFERS / GL_MAX_TESS_*_IMAGE_UNIFORMS = 8.

### CTS
- KHR-GL46.shader_image_size: **50 -> 62/74 Pass** (24 NotSupported -> 12 Fail;
  the tes cases now Pass; the 12 tcs fails need the TCS kernel's image slots
  for 8-image programs — scratch `zz_tc7.c` repro: all units read 0-sized, i.e.
  slots not bound for the canonicalized pair kernel).
- shader_image_load_store stale-INVALID_OPERATION root causes addressed
  (atomic_uint compile failure, vertex-only PSO failure); re-tally next round.
- Not yet re-tallied (next session): KHR-GL46.compute_shader, vertex_attrib_binding.

### Next round
1. TCS kernel image slots for 8-image programs (zz_tc7.c; likely the
   canonicalized pair kernel renumbers texture args — dump it, then bind by the
   pair kernel's own reflection).
2. Re-tally shader_image_load_store (atomic counter + vertex-only fixes should
   move advanced-sso-atomicCounters, basic-allTargets-atomic*), then
   KHR-GL46.compute_shader and vertex_attrib_binding per supervisor priorities.
3. Image atomics on textures remain unimplemented (SPIRV-Cross's 2D-only buffer
   emulation has no binding side in MGL; non-2D subscripts don't even compile;
   native `texture.atomic_*` works on this M4 Pro — feature-gate on GPU family).
   atomic_uint in TCS/TES/GS stages stays 0 (kernel/hand-authored paths).
