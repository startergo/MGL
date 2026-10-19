# tess — final domain summary (round 17)

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
