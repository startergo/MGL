READY d8909b6 (code tip: d8909b6; report commit on top of the 86c4482 merge of gl46-port a4462d4; headless suite 458 PASS / 0 FAIL, test_geom 10x 593/593, KHR-GL46.compute_shader 42/42).

## Round 43 — KHR-GL46.compute_shader: 33 -> 42/42 Pass (9 Fail fixed)

Started from the round-42 tally (Fail=14 Crash=4 at the second full pass; crashes were
gone by this session, 33 Pass / 9 Fail).  All nine fixed and committed:

1. **Atomic-counter injection fixes** (989fea6, b1dc3aa) — `mgl_msl_inject_atomic_counters`:
   - `atomic_uint` arrays: keep the index expression inside the address expression
     (`(thread atomic_uint*)&ac[i]` -> `(volatile device atomic_uint*)&ac[w+(i)]`); the
     bracket group is part of the replaced span (off-by-one left a stray `]`).
   - Counters used in non-entry functions: MSL has no mutable program-scope globals, so
     `msl_msl_counters_to_helper_params` gives every function that uses a counter —
     directly or through a call-graph fixed point — the counter pointers as trailing
     parameters and rewrites every call site (handles overloads by name-union).
   - The injected `[[buffer(N)]]` index is the renderer's buffer-map ordinal: one entry
     per ELEMENT of every uniform/default-uniform/SSBO resource (block instance arrays
     expand); counting MSL arguments drifts when a declared-but-unused block is stripped
     (resources-max declares g_buffer3/4 that p0 never touches).
   - Compute stage: buffer-size constants moved from the shared slot 25 to
     `MGL_CS_SSBO_SIZE_BUFFER_INDEX` 30 so 12 UBO + 8 SSBO + 8 ACB + DUB = 29 arguments
     fit (render stages stay < 24).
   - A declared-but-unused counter no longer fails the injection (argument still appended
     to keep map ordinals aligned).
   Tests: cs_atomic_array / cs_atomic_helper in tests/headless/test_atomic_counters.c.
2. **tex.fence() for read_write images** (e07a7de) — Apple GPUs don't make a write
   through `texture2d<..., access::read_write>` (GL image2D read_write) visible to a
   later read of the same texel in the same invocation; `mgl_msl_insert_image_fences`
   appends `NAME[i].fence();` after every write statement (resource-image).
3. **Conditional compute dispatch** (aac03e5) — glDispatchCompute/Indirect honour
   glBeginConditionalRender via the queries domain's `mgl_queries_render_allowed`
   (conditional-dispatching).
4. **Skip MSL-stripped image/sampler bindings** (aac03e5) — `msl_binding` now stores the
   established `0xffffffffu` "no Metal argument" sentinel when SPIRV-Cross returns no
   automatic binding (declared-but-unused image stripped from the MSL); the compute
   texture binder skips those instead of binding the GL unit's texture at the unit
   index, which clobbered the real argument that reuses the freed index
   (pipeline-post-fs: pass 1's output image moved to texture(0) and got the input
   texture).  Also fixes `getProgramLocation` assert by decrementing the pending count.
5. **spvUnsafeArray copy overloads** (d8909b6) — SPIRV-Cross's spvArrayCopyFrom* helpers
   only take native C arrays; whole-array assignment to a shared variable's member
   (shared-max) fails the RUNTIME Metal compiler (command-line `metal` accepts it — do
   not trust it; verified with a newLibraryWithSource harness).  Added element-wise
   overloads for spvUnsafeArray<T,A> sources/destinations; the template size parameter
   MUST be size_t (uint fails deduction: "unsigned long vs uint").
6. **Subroutine array index is uint** (d8909b6) — `int _mgl_i` rejected a
   `gl_WorkGroupID.x` (uint) index (no implicit uint->int in GLSL) (resource-subroutine).
7. **No glslang SPIRV-Tools validation** (d8909b6) — it applies Vulkan rules to legal
   GL: textureProjOffset on sampler2DRect in a compute shader (undefined LOD) becomes
   an explicit Lod operand, rejected for rectangular textures (resource-texture).
   SPIRV-Cross still rejects modules it cannot translate.
8. **Compute local-size link error** (d8909b6) — a program whose compute shaders declare
   no work group size must fail the link (glsl-link-time-errors); glslang defaults the
   linked intermediate to (1,1,1), so GLSL sources are scanned for a `local_size`
   qualifier; glShaderBinary / glProgramBinary use the module's LocalSize/LocalSizeId
   execution modes (kept test_program's binary-roundtrip tests green).

Shared-file edits this round (all surgical, listed per commit): program.c (atomic
injection call + base, fence/copy MSL passes, spv options, link check), MGLRenderer.m
(compute texture binder skip + pending count), compute.c (conditional-dispatch gate),
introspect_subroutine.c (uint index), MGLRenderer+Geometry.m (size slot 30).

### CTS harness notes (for the next round)
- run.py restarts batch numbering per invocation: batch_00001.qpa is OVERWRITTEN by a
  --resume run.  When mining old failures, sort batch files by MTIME, newest first.
- rdel.py (in /tmp, recreate if gone): pop cases from a results.json then run with
  --resume to re-run exactly those cases.
- Direct single-case runs (no runner): cd $CTS/build/external/openglcts/modules &&
  MGL_LIBRARY=$PWD-of-worktree/build/libmgl.dylib ./glcts --deqp-case=... (use
  --deqp-surface-width/height=64, --deqp-log-images=disable,
  --deqp-log-shader-sources=disable, fbo surface, mgl context).
- Two transient SIGTRAPs (rc=133) were seen on resource-image direct runs after the
  fence fix; four subsequent runs (incl. the full-group run) were clean.  If it
  recurs, ASan-build the runner path (make -f Makefile.asan lib) first.

## Handoff — next clusters

### KHR-GL46.vertex_attrib_binding (15/30 Fail at r43-vab tally) — ROOT CAUSE FOUND
Failing: advanced-bindingUpdate, basic-input-case2/3/4/6/7/9/10/11/12, inputI-1/2/3,
inputL-1/2.  Every one of them PASSES standalone (verified case1, 2, 10, 12 direct
runs); they only fail in a shared batch — case1 poisons case2 in the SAME process.
- Repro (scratch, untracked): tests/headless/zz_vab12.c — run case1 then case2 in one
  process; case2's DISABLED attribute slot 2 reads (1,2,3,1) = enabled attrib 1's
  vertex-0 data instead of the (0,0,0,0) current value (the CTS compares all 64 xfb
  slots, the scratch originally only compared enabled ones — that's why it looked fine).
- Root cause: **slot-30 collision** — `MGL_VA_CONST_BUFFER_INDEX` 30 (disabled-attribute
  constants, MGLRenderer+VertexAttrib.m extVattribPreDraw) vs `MGL_XFB_MSL_BUFFER_INDEX`
  30 (xfb capture scratch, bound at MGLRenderer+TransformFeedback.m:~218 AFTER the
  constants) — the capture draw overwrites the const binding, so disabled attributes
  read xfb scratch bytes.  The vattrib PLAN is correct (MGL_VATTRIB_DEBUG=1 shows
  attrib 2 fetch 3 const), the MSL [[attribute(2)]] mapping is correct.
- Fix direction: re-pin one of them.  Render-path pins are saturated (program resources
  0..23; geom 24..30 kernel-path only; xfb 30; tess shares 24+).  Candidates: move
  MGL_XFB_MSL_BUFFER_INDEX to 29 (geom MSL_CTL is kernel-path only; the render-encoder
  xfb capture path never coexists with the geometry emulation render pass) — the index
  is compiled into the capture-variant MSL text (transform_feedback.c), so changing the
  define recompiles consistently; alternatively move the vattrib const to 29.  Check
  which render-encoder slots the geometry raster pass really binds
  (MGLRenderer+Geometry.m ~3125 geomBindRenderStageTextures / index-list binding) before
  choosing.  Also grep for other users of both defines (mgl_ext_xfb.h, mgl_ext_vattrib.h).
- After the fix: re-run r43-vab (rdel the 15 + resume), then the full group fresh.
  My scratch pair (zz_vab.c / zz_vab2.c / zz_vab12.c) covers case1/case2 — promote a
  slot-2 disabled-attribute check into a committed test once fixed.

### Round 41/42 leftovers (if cheap)
- KHR-GL46.shader_image_size: 12 tcs fails need the TCS kernel's image slots for
  8-image programs (round-42 report has the zz_tc7.c repro notes).
- shader_image_load_store fails + stale GL_INVALID_OPERATION paths from round 42 —
  re-tally first, several may already be fixed by the msl_binding sentinel / fence work
  this round (image-related).

### Rules reminder
make -j4 only; one CTS batch at a time, foreground, --max-seconds <= 480, <= 2 chunks;
no background glcts; scratch (zz*.c, scratch_*, .agent/*, *.qpa, Makefile.asan,
build-asan/, /tmp helpers) stays out of commits — add files by explicit path; never
git stash / git add -A; stay on gl46/geom; merge gl46-port before the next READY.
