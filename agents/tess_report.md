# tess — fp64 round 5 — HANDOFF (fresh session starts here)

Branch `gl46/tess`, HEAD **<see below — this commit>** (on top of the gl46-port
merge 3dbc3f5).  Task: CTS `KHR-GL46.gpu_shader_fp64` group green + suite green.
**Suite: rc=0, 0 failures (418 PASS lines, 62 test files).**

**Task item (2) done:** `test_fp64` was split — it had grown past run.sh's 120 s
per-test alarm (steady state ~125 s of Metal compilation, no disk cache).
Now `test_fp64.c` (scalar_ops, attributes, flat_varying ~90 s),
`test_fp64b.c` (conversions, ubo_layout, ssbo_compute, extensions) and
`test_fp64c.c` (varying_dvec34, out_param_fns, uniform_roundtrip,
xfb_capture) each finish well under the alarm.

## fp64 group status (this round, warm Metal cache)

| case | status |
|---|---|
| errors | PASS |
| conversions | PASS |
| operators | **PASS** (was FAIL in round 4 — fixed by this round's iface work) |
| valid_constructors | PASS |
| varyings | FAIL — multi-variable batch link (see below) |
| state_query | FAIL — link |
| max_uniform_components | FAIL — link, log: `geometry: SPIR-V → MSL failed (double types are not supported in buffers in MSL.)` |
| named_uniform_blocks | FAIL — link |

NOTE: `operators` takes >10 min on a **cold** Metal cache (the case builds ~200
big fp64 kernels; a first run timed out at 240 s and 620 s, then PASSED in
~2 min warm).  Run it twice before judging.

## What this round fixed (all committed, c8a6a12)

1. **fp64.c — interface blocks as stage IO** (the core blocker): access chains
   through `in/out BLOCK { double m; }` and `BLOCK name[]` now work.
   `tess_pair_iface_members` counterpart for geometry/xfb is not needed (geom
   reflects the GL view).
2. **geometry.c**: in_v/out_v reflected from `program->spirv[stage].ir`
   (GL view — real doubles); the pass-through VS declares u32-slot repr for
   double varyings; glslang keeps `in double` attributes (VULKAN_RULES_RELAXED
   privatizes them — flag dropped when the source declares one).
3. **MGLRenderer+Tessellation.m `tessVsInputKinds`**: rewritten to walk the
   GL-view module (resource list type ids are from the *compiled* module =
   wrong id space for fp64).  Fixed wrong SPIR-V opcodes (OpTypeFloat=22,
   width at ir[m+2]), OpVariable operand order (type, id, storage), and
   has[]=true for float slots (MGL_VA_KIND_FLOAT == 0).  Double generic +
   array attributes now fetch UInt2/UInt4 in the VS kernel (verified pixel
   green: VS double attr → TCS → TES → FS = 5.0 exactly, scratch /tmp/t5.c).
4. **tessellation.c `rewrite_tes`**: the struct-body splice recomputed offsets
   from the tracked `len` which drifted → heap overrun → "MSL source is not
   valid UTF-8" + truncated source.  Now recomputed from `strlen(out)` and
   `cap` is kept in sync after the realloc.

## Repro material (untracked, do not commit)

- `tests/headless/zz4_cts_varyings.c` — faithful port of CTS GPUShaderFP64Test7
  ("varyings") for ONE variable per program: `./build/tests/zz4_cts_varyings
  <type 0..12> [arrayed]`.  Type 0 (double scalar) **links and draws**;
  check `xfb words` and `pixels`.  The multi-variable batches the CTS builds
  still fail.
- `/tmp/t5.c` — minimal VS(double attr)+TCS+TES+FS chain, PASSING (green).
- `MGL_TESS_DEBUG=1`, `MGL_DUMP_MSL=1`, `MGL_FP64_DUMP=/tmp/dir` (dump
  `<n>_input.spv`/`<n>_lowered.spv`; disassemble with
  `external/SPIRV-Tools/build/tools/spirv-dis`), `MGL_GEOM_DUMP=/tmp/dir`,
  `MGL_GEOM_TRACE=1`.

## Where to continue (ordered)

1. **varyings multi-variable batches**: extend zz4 to build batches like the
   CTS (`variables` lists capped by GL_MAX_*_COMPONENTS queries — see
   `executeFunctionalTest` in gl4cGPUShaderFP64Tests.cpp:9228).  Single
   variable links; find the first batch size/type that fails.  Suspects:
   location assignment for several block members (my fp64 repr keeps
   element-index-preserving layouts — see `geom_double_iface` and the iface_ty
   K_ARR cases), and the FS reading `gs_variableN[i].x`.
2. **max_uniform_components / named_uniform_blocks / state_query**: all link
   failures.  max_uniform_components logs
   `geometry: SPIR-V → MSL failed (double types are not supported in buffers in
   MSL.)` — a geometry-stage module with doubles in a BUFFER reaches
   SPIRV-Cross un-lowered.  Likely the geom/xfb record kernel path
   (transform_feedback.c `xfb_glsl_type_name` family) or a capture variant
   compiled by `geom_glsl_to_spirv` whose fp64 lowering is skipped — find which
   module it is (MGL_FP64_DUMP prints every lowering input; if this module is
   missing from the dump it never got lowered).
3. **tess interface-block pairing** (gated: `MGL_TESS_PAIR_BLOCKS=1` in
   tessellation.c): `tess_pair_iface_members` pairs flattened block members
   between producer/consumer MSL structs (works — trace prints the pairs), but
   the canonical-record path it enables (cp_record) still emits invalid MSL
   (heap garbage in the rewrite_tes epilogue).  With it OFF, tess block
   programs still link (SPIRV-Cross writes its own struct per side) but the
   VALUES do not cross TCS→TES (two record slots).  Needed for the CTS
   varyings case.  Debug by enabling the env, dumping
   `mgl_tess_plain_msl`/`mgl_tess_capture_msl` outputs and checking
   `tp->cp_record[]` entries for uninitialized members (`num_cp_record` vs
   what the propagate step copies at tessellation.c ~2806).

## Known gaps / notes

- Whole-column access into a wide (dvec3/dvec4-column) double matrix stage
  variable is rejected with a clear error (fp64.c rw_access); the CTS only
  needs [col].row.
- `glDisableVertexAttribArray` on a never-enabled attribute logs
  GL_INVALID_OPERATION (0x502) in MGL — pre-existing, seen in scratch runs
  (core GL allows it as a no-op).  Not addressed.
- The old `agents/conformance.md` tally is unchanged this round; update it when
  the four failing cases land.

## Round history

Round 4 and earlier: see the "Accumulated report" section below.

---

# Accumulated report (preserved from gl46-port squash; round-3 and earlier)

# tess — final domain summary (round 17)

GL 4.6 tessellation shaders (TCS/TES, GL_PATCHES) on Metal, branch `gl46/tess`.
Files: `MGL/src/tessellation.c` (GL side: program introspection, MSL rewrites,
CPU reference), `MGL/src/MGLRenderer+Tessellation.m` (draw execution), `mgl_ext_tess.h`
(GL/Metal interface), `tests/headless/test_tess.c` (499 checks), `tests/headless/test_tessemu.c`
(emulation-only checks). Hardware `drawPatches` path on Apple silicon.

## fp64 round 3 — HANDOFF (fresh session starts here)

**Commits this session:*### details

* `16ed51a` (headless drawable 200x200), `fddcd25` (UBO/SSBO/conversions
+ fp32 guards), `3f5ac66` (advertise both fp64 extensions), `1924bfb` (CTS hangs: texparam
growth cap + >4KB dub Metal buffers), `3e64955` (xfb capture rework, WIP).

**Suite state:** 28 binaries green (test_fp64 now 401 checks incl. fp64_conversions /
fp64_ubo_layout / fp64_ssbo_compute / fp64_extensions; test_tess 499; test_xfb 249).
`test_gpuaudit shadow_attribute_cache`: UNSKIPPED since round 2 and passing (104 checks, 0
failures, no SKIP in the file).

**CTS:** runner `.agent/cts_run.py 'KHR-GL46.gpu_shader_fp64.*' <timeout> [start-at] [limit]`
(foreground, sequential, results appended to .agent/cts_results.txt; old file has stale
UNKNOWN verdicts - delete before rerunning). With the current build the group's dmatN
cases PASS; the ~110 scalar/dvecN builtin cases fail ONLY on the xfb capture path below.
Still to write: `agents/conformance.md` with the final tally.

**Active bug (one root cause):** fp64 soft-float calls take `thread const ulong&`; when an
argument is an *lvalue* uniform member (`_4499(mgl_DefaultUniforms_1.uniform_0, ...)`) Metal
rejects the bind. Normal FS/VS shaders work (SPIRV-Cross materializes temporaries there);
the xfb capture kernel does not. Repro: `tests/headless/zz1_qs.c` pattern = VS
`uniform double uniform_0..2; flat out double result_0;` + `glTransformFeedbackVaryings`,
then draw with RASTERIZER_DISCARD (see git stash? no - the repro is described in the commit
message; recreate from the CTS case `KHR-GL46.gpu_shader_fp64.builtin.clamp_double`).
Likely fix: fp64.c lowering should emit an OpCopyObject/load-temporary for lvalue arguments
of the soft-float calls, or a post-pass on the capture MSL.

**CTS process rules (supervisor):** foreground only, one batch at a time, <20 min per call,
never nohup/&; add files by path (never `git add -A`); never commit symlinks/external; no
`git stash`; MAKEFLAGS=-j4, low CPU.

**Known gaps:** dvec3/dvec4 stage varyings compile wrong in the FS (`in.dv[0]` rendered as
scalar index into uint4; pre-existing round-2 gap, only scalar/dvec2 varyings verified);
texparam records capped at 1M names (queries above return spec defaults).


### details

* `16ed51a` **headless drawable fixed**: `CppCreateMGLRendererHeadless` pins
  contentsScale 2 so the default framebuffer is always 200x200. Over a bare SSH
  session `[[NSScreen mainScreen] backingScaleFactor]` returned 1.0, halving the
  drawable to 100x100 — every pixel read at y >= 100 returned zeros. This was the
  entire "TES output clipped/shifted" mystery from the round-2 checkpoint (126
  failing checks) AND the test_geom/test_geom_spirv "missing vertexDescriptor
  binding" failures. Full suite green again right after (28 binaries, 0 failures).
  NOTE (shared file MGLRenderer.m — surgical, one flag + one conditional).
* `fddcd25` **real double uniforms/UBO/SSBO layouts** (verified at runtime, new
  tests `fp64_conversions`, `fp64_ubo_layout`, `fp64_ssbo_compute`):
  - OpFConv-from-double declared `mgl_fp64_to_fp32` with an i32 result type
    although the function returns float → malformed SPIR-V; SPIRV-Cross threw
    ("nullptr") for `float(d)` of any non-constant double in ANY stage.
  - std140 scalar-double arrays: fp64 now widens the array element itself to
    vec<stride/8>(u64) (def pass; layout stays decoration-exact) and routes
    element chains through a new `arrlane` marker (loads extract lane 0, stores
    re-fill pad lanes). Fixes the `ulong2`-argument compile error and wrong
    offsets for `double a[n]`; dvecN arrays / dmatN keep using SPIRV-Cross's
    own packing. std140 layout matrix verified end-to-end: s/v2/v3/v4/arr,
    dvec2.y, dvec3.z, dvec4.w, dmat2/3x2/4 columns.
  - **fp32 arithmetic was being lowered to soft-float64 calls** in any program
    containing doubles (MSL silently truncated float args to ulong):
    `vec4(dvec4) * 0.25f` rendered 0. All float ops in the body rewrite now
    guard on the result/operand type containing Float64.
  - SSBO std430 loads + widened std140 stores through a compute shader,
    bit-exact readback (`fp64_ssbo_compute`).
  - default-block double uniforms verified at runtime (dvec4/dvec2/double/dmat2
    through `glUniform*dv`) — they were already correct; earlier "failures" were
    my probes reading UNORM-clamped values > 1.
* `3f5ac66` **extensions advertised**: `GL_ARB_gpu_shader_fp64`,
  `GL_ARB_vertex_attrib_64bit` (+ `fp64_extensions` test: glGetStringi listing +
  glVertexAttribL*/glGetVertexAttribLdv round trip).


GL 4.6 tessellation shaders (TCS/TES, GL_PATCHES) on Metal, branch `gl46/tess`.
Files: `MGL/src/tessellation.c` (GL side: program introspection, MSL rewrites,
CPU reference), `MGL/src/MGLRenderer+Tessellation.m` (draw execution),
`MGL/src/MGLTessKernels.h` (GPU kernel source), `MGL/src/mgl_ext_tess.h`
(state/design), tests `test_tess.c`, `test_tessemu.c`, `test_tesscap.c` +
`tests/headless/mgl_tess_diag.h` (test-only diagnostics).

## What runs on the GPU now

**Hardware path** (triangles/quads without point_mode — Apple GPUs *do*
tessellate; verified by probe in round 10/11):

1. VS as a compute kernel (SPIRV-Cross `VERTEX_FOR_TESSELLATION`), outputs to a
   buffer in a per-draw ring block.
2. Indexed draws are fully GPU-driven (round 14): a scan kernel reduces
   min/max/restart-count with atomics in the ring, a `mgl_tess_scan_setup`
   kernel writes the VS parameter block (vfirst=0, vcount=max+1 clamped to the
   attribute capacity), and the VS is dispatched at the CPU-known capacity with
   threads past vcount returning early. `dispatchThreadgroupsWithIndirectBuffer`
   was rejected: Metal's stage-in fetch does not follow an indirect grid.
3. Gather kernel: expands the element buffer into the linear control-point
   stream (type conversion, base vertex, restart skip; a `modulo` mode indexes a
   cached gapless list for restart ranges).
4. TCS kernel (multi-patch workgroup): control points, patch outputs, and
   `MTLTriangle/QuadTessellationFactorsHalf` written directly into the ring —
   GPU to GPU.
5. `mgl_tess_blob` kernel (round 12): assembles each patch's TES patch-in
   record (levels from the half factors, members copied from the TCS patch
   record by a link-time map).
6. Render pass: post-tessellation TES (`[[patch(...)]]`) with
   `drawPatches`/`drawIndexedPatches` + `setTessellationFactorBuffer`;
   partitioning from the TES spacing, winding mirrored for the y-flip.

**Emulated path** (isolines and point_mode — Metal has no isoline domain and
cannot emit points): the same VS/TCS/blob compute chain, then the D3D11-style
GPU tessellator kernels (count → emit → indirect draw args) writing a
coordinate stream, and a plain-vertex TES variant (reading tess coord + patch
id from buffers) drawn through an indirect draw. No CPU readback on any path.

**Default levels** (no TCS): CPU-written half factor records (constants) +
control points passing through from the VS kernel.

## What CPU work remains, and why

* **No-TCS half-factor fill** — 12 bytes/patch of constants written to the ring
  before dispatch; no GPU dependency, and a kernel dispatch would cost more.
* **`mgl_tess_note_counts`** — arithmetic-only query bookkeeping
  (PRIMITIVES_GENERATED / TES invocations) for TCS-less draws; O(patches), no
  memory traffic. GPU query counters would be the only alternative.
* **Restart handling** — presence check + gapless list build, at most **once per
  element-buffer content** (cached, keyed on the gpuaudit content serial; rounds
  13/15). A restart re-forms patches, which fixed dispatch grids cannot express
  without fully indirect draws; the build reads CPU-visible memory only — no GPU
  wait.
* Link-time work (MSL rewrites, introspection, PSO compilation) — cached per
  program/descriptor.
* `MGL_TESS_CPU=1` reference path — opt-in debug only.

Zero `waitUntilCompleted` on the default path (rounds 13–15; proven by
`mgl_tess_scan_waits()` staying 0 through first draws, redraws, buffer-content
changes and restart ranges), zero CPU blob builds / CPU tessellations
(`mgl_tess_cpu_blob_builds` / `_cpu_generate_runs`), and `glFinish` was verified
to wait for all committed tess command buffers (round 16).

## Trace/diagnostic hygiene (round 17)

All `getenv` uses are once-only cached-flag initializers (`tess_trace()`,
`tTrace()`, `tCpuReference()`, `tNoFixup()`); no per-draw `getenv`. Every
`fprintf`/`NSLog` is gated by those flags or is an `MGL ERROR`/warning on a
failure path. Test-only counters and the opt-in GPU-time instrumentation live in
`tests/headless/mgl_tess_diag.h` (`mgl_tess_diag_enable()` attaches the
completion handlers; off = no per-draw cost).

## Known limits

* **Capacity clamp** (round 14): a draw whose absolute max index+1 exceeds the
  attribute capacity (massively sparse range out of a much larger vertex
  buffer) is clamped by the GPU — vertices past the capacity stay untransformed
  and those patches degrade. Measured cost of the capacity-spanning dispatch:
  GPU time 16–22 µs/frame for a 1M-thread dispatch vs 28–44 µs/frame for the
  6-thread one (the small case pays the vattrib domain's shadow-copy path
  instead) — no capping needed.
* **Half-precision tess factors**: the TCS's levels are consumed (and seen by
  the TES) in Metal's half factor format — same precision as MoltenVK.
* **PRIMITIVES_GENERATED with a TCS** is not noted (levels live on the GPU).
* **Isolines/point_mode are emulated** (no Metal support) — pixel-verified
  against the analytic tessellation and, for points, against plain GL_POINTS.
* **Pixel-edge tie-break** for points/lines goes to the lower/left pixel —
  identical to Apple's native GL (verified in round 11).
* **MGL_TESS_CPU reference path** renders incorrect pixels (crash fixed in
  round 12; geometry and counters correct) — debug-only, unfixed.

## Tests

* `test_tess.c` (468 checks) — hardware path (triangle/quad domains, four
  patches, indexed + instanced + base vertex, restart, consecutive draws, PSO
  cache stress, orientation/culling, program queries, errors, default levels),
  `TessIndexedNoRescan` (zero scan waits, cache invalidation, minv>0 pixels),
  `TessRestartCheckCached` (one presence check per content).
* `test_tessemu.c` (53 checks) — isolines/point_mode pixel-exact (incl. vs.
  plain GL_POINTS), emulated-path patch-blob tests with zero CPU blob builds /
  tessellations, default levels without a TCS.
* `test_tesscap.c` (4 checks) — glFinish-waits verification, GPU-time
  measurement of the capacity dispatch with a pixel assertion.

Full suite: 22/22 files, 0 failures.

---

# Round history (rounds 11–16; records kept for reference)


---

# Round 18 (SPIR-V binary format; assert(0) cleanup; TES-output capture producer)

## A) SPIR-V as a shader binary format — DONE
`test_spirvbin.c` + `spv_vsfs.h` (word arrays generated with MGL's own glslang via
`.agent/gen_spv.c`): the enums, `glGetStringi(GL_SPIR_V_EXTENSIONS)`,
`glShaderBinary(SPIR-V)+glSpecializeShader("main")` linking and drawing (green
pixel) all pass; `get.c` now advertises `GL_ARB_gl_spirv` + `GL_ARB_spirv_extensions`
(commits b91aab6/c67b818).

## B) The five assert(0)s — DONE (7b46f20)
`bitcountForInternalFormat`/`internalFormatForGLFormatType`/
`mtlPixelFormatForGLFormatType` (pixel_utils.c + the MGLTextures.m copy) now return
0 / GL_NONE / MTLPixelFormatInvalid on unknown values; `MGLGetContextInfo` writes 0.
`test_pixelerr`: junk enums through the exported mappers no longer abort (before:
Abort trap 6 at pixel_utils.c:1833).

## C) TES-output capture producer — interface complete, enable incomplete (e18bb07)

**Contract for the geometry / transform-feedback worker** (all in `mgl_ext_tess.h`):
`mgl_tess_set_capture(ctx, program, true)` makes every GL_PATCHES draw of that
program publish a `MGLTessCapture` in `ctx->state.ext_tess->capture`:

* `buffer` + `records_offset` — one packed record per tessellated vertex:
  words 0..3 = `gl_Position` (raw, no y/depth fixup), then user varyings in
  declaration order, `words = elems*cols*rows*base_words` (2 for double), no
  padding — the same rule `geometry.c` uses for its captured VS-output records.
* `member[]` — name/words/offset table; match varyings by NAME (like the blob map).
* `index_offset` — the uint32 vertex indices of the tessellated output;
  `count_offset` — the GPU-written vertex count word (for indirect consumers);
  `stride` — record stride in words; `prim` — GL_TRIANGLES/LINES/POINTS.
* Valid until the next GL_PATCHES draw; the producing draw takes the emulated
  tessellation path (forced) so the stream is GPU-visible; no CPU readback.

The record-writing kernel is the plain TES rewritten (`mgl_tess_capture_msl`),
dispatched after the tessellator kernels in the same encoder. `program.c`'s
"geometry after tessellation" rejection is untouched for geom to remove.

**Known gap**: ~~the draw never sees `tp->capture == true`~~ — fixed in round 19 C
(see below). The producer is complete; the geom/xfb consumer side still has to
call `mgl_tess_set_capture` from its own hook with the live context.

---

# round 19

## C) TES-output capture producer finished (TessOutputCapture PASS)

The producer never worked end to end; five real bugs, each reproduced and fixed
on the M4 Pro (commit "finish the TES-output capture producer"):

1. `mgl_tess_capture_msl` mis-parsed the plain-TES signature: the return type
   carried a trailing space (struct lookup failed), the kernel name was cut at
   the first space inside the parameter list, and the original
   `uint mgl_vid [[vertex_id]])` tail was emitted into the kernel signature.
   Rewritten with explicit tokens + a body cursor; bare `return;` kept.
2. `str_append` truncated appends > 2047 bytes (vsnprintf into a stack buffer),
   silently corrupting every generated MSL with a longer body; now formats
   into the heap when needed.
3. The capture kernel was dispatched with an `MTLFunction` where Metal requires
   a compute PSO (driver crash at setComputePipelineState). The PSO is built
   once per program and cached (`tp->capture_pso`), released with the program.
4. `buffer(31)` is outside Metal's compute range (0..30): the records buffer
   moved to `MGL_TESS_TES_CAP_RECS` (24; a TCS-only reservation, free in the
   TES kernel).
5. `MGLTessCapture.buffer` held the `id<MTLBuffer>` OBJECT — consumers read the
   ObjC object's ivars instead of the storage (that is why the count read 0
   while a trace dump of `ring.contents` saw the GPU-written 3). It now
   publishes `ring.contents` and retains the object in `buffer_object`.

Also: the capture kernel is compiled with the rstate clip-space fixup
neutralised (function constants ul/z01/pps/psz set to their no-op values), so
records hold gl_Position exactly as the TES wrote it (the contract's
"unmodified"); the synthetic `mgl_rs_PointSize` member is not captured; the
published count is the VERTEX count (`totals[0]`), not the draw's index count;
both emulated and CPU-reference paths now release their ring block when the
render command buffer retires (the slots leaked before — 8 emulated draws used
to exhaust the ring).

test_tessemu: TessOutputCapture now covers point-mode triangles (3 corner
records, count=3) AND a filled quad domain (prim=GL_TRIANGLES, count=4 — the
vertex count, not the index count 6). 65 checks, 0 failures.

## D) format mappers

D1 — deleted: `MGLTextures.m` (entire file was `#if 0`), and in pixel_utils.c
`bicountForFormatType` / `bitcountForInternalFormat` /
`internalFormatForGLFormatType` (839 lines, no library callers, junk tables).
Prototypes dropped from pixel_utils.h / glm_context.h / MGLContext.h;
test_pixelerr.c re-targeted at the remaining live mapper.

D2 — `mtlPixelFormatForGLFormatType` (pixel_utils.c) rewritten: every GL 4.6
table 8.2/8.3/8.3a-legal (format,type) pair resolves; illegal pairs return
MTLPixelFormatInvalid. Fixed GL_INT→Sint, UNSIGNED_BYTE+RED/RG→Unorm,
*_INTEGER→Uint/Sint; added RGB/BGR, HALF_FLOAT, 10F_11F_11F_REV→RG11B10Float,
5_9_9_9_REV→RGB9E5Float, 24_8, FLOAT_32_UNSIGNED_INT_24_8_REV, depth with
UNSIGNED_SHORT/UNSIGNED_INT. The packed 16/32-bit layouts were verified on the
GPU (`.agent/probe_packed.m`): Metal's packed-format names list components from
the LOW bit, so A1BGR5 == GL 5_5_5_1, BGR5A1 == GL 1_5_5_5_REV, ABGR4 ==
GL 4_4_4_4, RGB10A2 == GL 2_10_10_10_REV; the old code had the two 5551 forms
swapped and 2_10_10_10_REV mapped to the BGR twin. GL 3_3_2/4_4_4_4_REV/
5_6_5_REV/10_10_10_2 have no bit-exact Metal format and widen (conversion on
upload, verified below).

D3 — primary path audit (texdata_formats.c): `tdValidateFormatType` accepted
only the ES3 legality matrix; now takes the desktop table 8.3 combinations
(packed 4-component types with RGB/BGR, integer twins for 8_8_8_8(_REV)/
2_10_10_10_REV, BGR for the float packed types). `tdPack` now maps
signed-normalized storages read back through unsigned normalized clients from
[-1,1] into [0,1] (GL 4.6 8.5.4); float/signed clients and channels the storage
lacks are exempt. Nothing impossible found: every required internal format
resolves and converts (see D4).

D4 — `tests/headless/test_pixelformats.c` (new): 65 colour + 6 depth/stencil
required sized internal formats × every legal (format,type) upload = **1596
combinations**, reference values encoded into the client type, uploaded, read
back with glGetTexImage and compared through the double quantisation
(client grid → storage grid → client grid), with per-storage/per-client
tolerances. **0 failures.** The 18 advertised compressed formats (BC1–7,
ETC2/EAC, ASTC) all report INTERNALFORMAT_SUPPORTED.

Shared-file edits this round (all listed): glm_context.h, MGLContext.h,
pixel_utils.h (prototype removals); texdata_formats.c (legality matrix,
snorm readback map — texdata agent's file, surgical); test_texdata.c (one
expectation that asserted the ES3 rule for RGB+8_8_8_8, replaced with a
genuinely illegal pair); test_pixelerr.c (deleted-function references);
mgl_ext_tess.h (capture struct gains buffer_object; new CAP_RECS/CAP_COUNT
indices).

Suite: 25 test binaries, 0 failures (250 PASS); test_tess 468 checks,
test_tessemu 65 checks, test_pixelformats 2 tests / 1596-combination sweep.

---

# fp64 progress (round 2, checkpoint 2026-10-20 09:3x)

## Done this session (commit 7aa4145, 0ecf799, f1fc18b)
1. **dvecN vertex attributes bit-exact on the draw path** (item 1 of the gap
   list, supervisor priority).  fp64.c lowers double/dvecN vertex inputs to
   raw u32 word slots (double->uint2, dvec2->uint4, dvec3/dvec4 -> uint4 @loc k
   + uint2/uint4 @loc k+1, the second location is a new Input variable with a
   Location decoration and entry-point interface entry).  Loads reassemble the
   u64 register values with pure bitcasts; constant component access (pos.x via
   access chains) expands to whole-slot load + extract.  The renderer feeds the
   slots: glVertexAttribLPointer data is fetched DIRECTLY from the app buffer
   when aligned (zero CPU work, bit-exact — Metal has no 64-bit vertex format,
   so the words ARE the value), misaligned/converted streams go through the
   gpuaudit shadow cache as one 16/32-byte-per-element stream shared by both
   word slots (glVertexAttribPointer GL_DOUBLE converts fp32 exactly per spec,
   floats widen).  Disabled arrays feed the current generic double value's
   bits.  The tess VS compute kernel gets the same word-slot treatment.
   test_gpuaudit shadow_attribute_cache is UNSKIPPED and passes.
2. **Link restructure**: all stages' SPIR-V is generated before any stage is
   compiled to MSL.  Interleaving compile_stage work with
   glslang_program_SPIRV_generate corrupted glslang's TPoolAllocator (garbage
   pool sizes -> std::bad_alloc on the next generate); hit by any program with
   a dvecN attribute (hidden behind the round-1 skip).
3. **Location rules**: dvec3/dvec4 consume two locations; vertex-input location
   overlap now fails the link (GL 4.6 11.1.1) instead of emitting two Metal
   attributes at one index.  mgl_introspect_slot_owner() resolves word slots.
4. **xfb of double varyings** (transform_feedback.c): GL_DOUBLE* + dmat leaf
   types, words = components*2, capture GLSL writes unpackDouble2x32 per
   component (bit-exact), capture variant runs the fp64 lowering and gets its
   input Locations copied from the program's GL view (lone-shader compiles
   have none — spvc needs them for [[attribute(n)]]).
5. **flat double varyings + uniform round trip**: whole-variable OpLoad/OpStore
   of double stage IO never hit the u32-pair iface path before (only access
   chains did) — the IO pointer pointee is now rewritten up front.  Scalar
   K_FLOAT types had n=0, which sent every scalar pair conversion down the
   vector path (uint2(uint4,uint4) garbage).  glGetUniformdv round-trips
   denormals and 1e308 exactly.

## In progress at checkpoint (NOT green)
Today's Metal runtime changed under us mid-session (the MAIN CHECKOUT's
prebuilt test binaries fail identically; nothing was installed — CLT 26.5):
- compute PSOs with a stage-input descriptor now REQUIRE setStageInRegion
  (the tess VS kernel dispatch was dropped silently -> black) — FIXED.
- drawPatches' N must equal the TES function's [[patch(domain, N)]] (Metal
  dropped the draw) — FIXED with a per-N cached rebuild (textual
  re-substitution on the stored MSL + compile_msl + rstate standin
  specialization); gl_PatchVerticesIn reads the same N, so no-TCS programs
  now report the real GL_PATCH_VERTICES.
- REMAINING: test_tess TES output is still clipped/shifted (10 tests fail;
  the tessellated shape renders but not where expected), test_geom /
  test_geom_spirv fail with 'missing vertexDescriptor Buffer binding at
  index 8' (debug layer).  Next steps: dump one failing TES draw's exact
  gl_Position values (xfb capture is a good probe), fix the geom binding,
  then re-run the whole suite, then items 4/5 (rcp/rsq ulp, extensions +
  limits + the KHR-GL46.gpu_shader_fp64 CTS group + conformance.md).
- Queued by supervisor (AFTER fp64 + CTS green): optional MGL_FP64_FAST=1
  double-single (float32x2) fast path, reference metal-float64 (MIT).



- Queued by supervisor (AFTER fp64 + CTS green): optional MGL_FP64_FAST=1
  double-single (float32x2) fast path, reference metal-float64 (MIT).

---

# fp64 progress (software IEEE doubles, round 1 — earlier session)

Design (from the crashed prior session, kept): `MGL/src/fp64.c` lowers
SPIR-V *after* glslang — every OpTypeFloat 64 becomes u64, every double op is
rewritten into calls into a merged soft-float64 library
(`MGL/src/fp64_softfloat.glsl`, generated from Mesa's
`src/compiler/glsl/float64.glsl` (Berkeley SoftFloat port, MIT) by
`.agent/fp64/gen_lib.py`, which also applies MGL fixes and appends composed
entry points: div/rcp/sqrt/mod/rem/ldexp/frexp + a fused FMA).  The library
SPIR-V is merged per-module by reachability (`merge_lib`), and `program.c`
translates the result with SPIRV-Cross MSL options + a textual fixup
(`thread const ulong&` scalars → by-value — see shared-file edits).

## Library work this session (all verified bit-exact on the GPU)

GPU stress harness `.agent/fp64/runops.m` (+ `mklib.sh`, `x2msl`, `xdis`):
one Metal kernel per library op, 16k structured-random pairs (subnormals,
min/max double, ±0, exact-tie patterns) bit-compared against libc:

  * add/sub/mul/**fused fma**/floor/ceil/trunc/round/fract/sqrt/glmod(OpFMod
    semantics)/frem/ldexp: **0 bad / 16382**.
  * div (rcp + exact-fma Markstein, subnormal operands rescaled by powers of
    two): **1 bad / 15649** (2^-1074/2^-1075-scale corner, 1 ulp).
  * rcp: 1 bad / 16382 (Newton-Raphson fixed point at exact powers of two,
    1 ulp); rsq = rcp∘sqrt inherits it (~12% of *duplicated* structured
    inputs, unique failures few).  Both are **internal only** — GLSL exposes
    no fp64 rcp/rsq/inversesqrt.
  * CPU reference models: `.agent/fp64/cfma.c` (fused FMA, 0 bad / 20M),
    `cf64.c` (add/mul, used to find the port bugs).

Bugs found & fixed in the Mesa port (all in `gen_lib.py`'s fix list, with
comments): round-to-even mask at exact ties clears the wrong bit; the
subnormal-result `increment` recomputed with an unsigned compare (always
false → truncation); ftrunc/fround lose the sign of -0; subnormal+subnormal
add carries out of the subnormal field without rounding; fround's tie test
fires after the half-ulp carry (round(218000.5689) → 218000); rcp seed
overflowed for |src| > fp32 range (now seeded from the normalized
significand; subnormal normalization fixed; sign applied to the seed so NR
converges); fdiv's Markstein residual now uses the fused fma and early-outs
zero/inf quotients.  Mesa's ffma was mul-then-add (two roundings) — replaced
by a genuinely fused mulAdd (exact 192-bit product+addend, one RNE round,
subnormal results rounded once onto the 2^-1074 grid).  Shifts by 0/64 at
alignment corners are split out (UB in GLSL, silently masked by Metal).

## GLSL op coverage (tests/headless/test_fp64.c — 311 checks, 0 failures)

add/sub/mul/div/fma/neg/abs/sign/min/max/mod/rem/clamp/mix/step/smoothstep/
floor/ceil/trunc/round/fract/sqrt/ldexp/exp/log/pow/sin/cos/atan comparisons,
conversions int/uint/float/int64/uint64↔double, unpackDouble2x32 — all
bit-exact vs per-op CPU references (contraction-proofed with `volatile`;
FMA-contracted refs would not match per-op GPU rounding).

## Known gaps (next session)

1. **dvecN vertex attributes** (ARB_vertex_attrib_64bit): the lowering maps
   Input doubles to raw u64-pair attributes, but MGL's renderer feeds
   GL_DOUBLE streams as converted fp32 (attribute cache) → MSL cast error /
   link failure.  `test_gpuaudit.c: shadow_attribute_cache` is SKIPped with
   a note.  Plan: load the fp32 attribute and widen per component with
   `mgl_fp32_to_fp64` (attempted; crashed in SPIRV-Cross (bad_alloc) after
   the rewrite — the OpTypePointer for Input must be rewritten *before* the
   body pass so the load pass sees the iface mark; reverted to keep the
   suite green, see git history of this session for the attempt).
2. rcp/rsq 1-ulp Newton fixed points (internal only; fix = last-ulp
   correction via the exact residual pair).
3. Not yet done (API side): GL_ARB_gpu_shader_fp64 / GL_ARB_vertex_attrib_64bit
   advertisement, glGet* limits, double uniforms/attributes plumbing checks.
4. test_fp64dbg.c / zz_dbg.c debug harnesses deleted (per checkpoint rule);
   the standalone GPU harness lives in `.agent/fp64/` (untracked).

Suite after this session: all test binaries 0 failures
(test_fp64 311; test_tess 499; test_tessemu 65; test_gpuaudit 94 + 1 SKIP).
