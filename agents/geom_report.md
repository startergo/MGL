READY 04e4fba (branch tip; fix commit 627bc49 + regression test 95f6fbc, both after merging gl46-port 8949387 and 4087ad8). Round-40 heap crash FOUND+FIXED: stale vattrib plan dbl[]/shadow_stride[] overflowed converted attribute streams; KHR-GL46.direct_state_access 370 -> 371/371 Pass — vertex_arrays_attribute_format fixed by the same change; headless suite 442 PASS lines / 59 files 0 failures, test_geom 593/593, new test_vafmt 84/84; 4 full CTS cluster runs with zero crashes and zero ASan reports

# geom domain — round 40 handoff (the flaky heap crash, vertex_arrays_attribute_format)

Round 40. Merged gl46-port twice mid-round (8949387 = geom round 39, then
4087ad8 = tess round 11). Commits: 8c8a7e2, 627bc49 (fix), test_vafmt.c, this report.

## 1. The flaky heap crash — FOUND and FIXED

Method (reproducible for future hunts):
* `~/Library/Logs/DiagnosticReports/glcts-*.ips` (137 files) bucketed by first
  libmgl frame. Two families: (a) `_xzm_xzone_malloc_freelist` aborts — classic
  heap-corruption-with-random-victim (SPIRV-Cross parse, strdup, os_log, xpc,
  glslang SpvBuilder) — these are the "1 in 3 batch" flake; (b) Metal
  `MTLReleaseAssertionFailure` aborts (see §3).
* ASan build of libmgl: copy Makefile -> `Makefile.asan` with
  `build_dir := build-asan`, `CFLAGS/LIBS += -fsanitize=address`; build with
  `make -j4 -f Makefile.asan lib`; run the CTS case with
  `MGL_LIBRARY=.../build-asan/libmgl.dylib DYLD_INSERT_LIBRARIES=$(clang
  -print-file-name=libclang_rt.asan_osx_dynamic.dylib)`. Note: glcts dlopens the
  library; passing the env through `perl -e 'alarm N; exec @ARGV' env ... glcts`
  works, plain prefix-env + perl exec sometimes hits "interceptors not
  installed"/dlopen deadlock — retry or use the `env` form. ASan heap addresses
  are >16TB, so the `MGL_PLAUSIBLE_PTR_MAX` bound in MGLRenderer.m had to become
  `__has_feature(address_sanitizer)`-aware (commit 8c8a7e2) or nothing renders.
* First ASan run of `KHR-GL46.direct_state_access.vertex_arrays_attribute_format`
  produced the report in seconds: heap-buffer-overflow WRITE of 8 bytes, victim a
  **1-byte** `calloc` in `vattribBindShadow:` (MGLRenderer+VertexAttrib.m),
  writer `mgl_va_fetch_dbl_words` (vertex_attrib.c:185), path
  `xfbCaptureAsync -> extVattribPreDraw -> vattribBindShadow`.
* A standalone local repro (`tests/headless/zz_vaf.c`, scratch) reproduced the
  full sequence with the normal build: doubles pass, then "int" variants return
  garbage and the process aborts with "BUG IN CLIENT OF LIBMALLOC: memory
  corruption of free block" inside glslang — the same crash signature as the CTS
  batches.

Root cause: the vattrib draw plan is rebuilt per draw, but the reset loop only
cleared `fetch/shadow_index/owner_gl` — **not `dbl[]`, `dbl_words[]` or
`shadow_stride[]`**. Any program with double vertex inputs marks its slots
`dbl=1`; the NEXT program (same context, same attribute indices) that needs a
CPU-converted ("shadow") attribute stream inherits `dbl=1`, so
`vattribBindShadow` converted int/float data with `mgl_va_fetch_dbl_words`
(writes 8 bytes per component: 24–32 bytes) into `count*16`-byte buffers, and
computed `stride_out` from the stale `shadow_stride[]` (calloc(count, 0)). One
small calloc overflow per draw ⇒ malloc metadata trashed ⇒ the process aborts
much later at an arbitrary allocation. This is why victims looked random
(SPIRV-Cross/glslang/os_log/xpc) and why ASan pointed somewhere else than the
crash stacks.

Fix (commit 627bc49):
* plan reset zeroes `dbl/dbl_words/shadow_stride`;
* `vattribBindShadow` derives the double element stride from the owner format
  (same `(size*8+15)&~15` formula the plan uses) instead of trusting the plan
  table, and reads the format of `plan->owner_gl[attrib]` (the GL attribute
  that actually feeds the slot);
* validation: `test_vafmt.c` (committed) fails 28/84 without the fix, 84/84
  with it. Two full texture_gather clusters (71/75 = the 4 pre-existing
  texture_gather fails), dsa vertex_arrays 19/19, geometry_shader 134/136 (the
  2 long-standing fails) — zero crashes, zero "bad" results; the attribute-format
  case also Passes under the ASan build with no reports.

## 2. direct_state_access.vertex_arrays_attribute_format — Pass

Same bug (it was both the corruptor and returned wrong XFB values: the int
streams were converted as double words). Now `KHR-GL46.direct_state_access`
371/371 (the group 19/19 in `~/MGL-cts/geom40vaf`).

## 3. Remaining crash families (NOT heap corruption; not fixed this round)

Bucketed from the same 137 reports, for whoever owns them:
* `MTLReleaseAssertionFailure` when `[buf computeCommandEncoder]` creates a
  second encoder on a buffer that still has an active one — 13×, ALL in
  tessellation CTS cases (TessellationShaderTCTEgl_MaxPatchVertices*,
  Invariance, PatchVerticesIn, VertexOrdering), deepest libmgl frame
  `geomXfbTessEncode` (my file; geometry.c:6460 route). geomXfbTessEncode ends
  the render encoder and rejects committed buffers, but cannot see/end an
  encoder left open by another path (`MGLRenderer+Tessellation.m` creates
  compute encoders on `_currentCommandBuffer` in several places; there is no
  public API to query a buffer's active encoder). Suggested owner fix: a
  renderer-wide `_currentComputeEncoder` ivar that every creator sets and
  `endRenderEncoding` also ends. 26 more of the same abort sit in
  `texdataEncodeUpload` (TextureData domain), 11 in `generatePipelineDescriptor`
  (Oct 20 only), 3 in `geomSubAlloc` doesNotRecognizeSelector (Oct 20 only,
  likely fixed by round-36+ type-confusion work).
* `mapBufferRange [buffers.c:~1377]` SIGSEGV in
  `glcts::TessellationShaderUtils::captureTessellationData` (3×, Oct 22 21:49+,
  tess verification): a buffer whose backing is gone/GPU-only being mapped.
* `texextRefreshStage:743` -> `setVertexSamplerState` (2×, Oct 22 20:03/20:08):
  a garbage `MTLSamplerState` passed to Metal — either the texture domain's
  `Sampler->mtl_data` holds a non-sampler object or it is another downstream
  victim of the vattrib overflow (same batched process). Re-run the cluster that
  hit it; if it still reproduces, it is real.

## 4. Test/build state

* `make -j4 lib` clean; headless `tests/headless/run.sh`: 442 PASS lines, 59
  executables, 0 failures (new: test_vafmt 84 checks, 17s). test_geom 593/593.
* `python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT` clean for
  MGL/src/geometry.c and MGLRenderer+Geometry.m.
* Shared-file edits this round (both surgical): `MGLRenderer.m` — the
  plausible-pointer bound macro (ASan-aware; 12 call sites mechanically renamed
  to it); `MGLRenderer+VertexAttrib.m` is the vattrib domain's file (not mine)
  but the fix is entirely in it — listed here because it is the round's main
  change: plan reset + vattribBindShadow stride/owner-format fix.
* Scratch kept out of commits: `Makefile.asan`, `build-asan/`,
  `tests/headless/zz_vaf.c` (local repro; ask if you want it), `TestResults.qpa`,
  `/tmp/mgl_asan.*` reports, `/tmp/*.qpa`.

## 5. Suggested next clusters

texture_gather 71/75: the 4 left are plain-gather-float-2d-rgb(-alpha) style
value mismatches (round-38 notes have per-case detail). Then whatever the
supervisor picks; limits and direct_state_access are now fully green.
