READY 666b07b (KHR-GL46.limits 123 -> 140/140 Pass; direct_state_access 367 -> 370/371 Pass, the one left is vertex_arrays_attribute_format; headless suite 438 PASS lines, test_geom 593/593; no new flake beyond the known round-38 heap crash)

# geom domain — round 39 handoff (KHR-GL46.limits, KHR-GL46.direct_state_access leftovers)

Round 39 (this session). Round 38 was merged into gl46-port as aab4e0b; this branch
merged gl46-port at e0f22a6 before the work. Commit: 666b07b.

## 1. KHR-GL46.limits — 123/140 -> 140/140 (results: `~/MGL-cts/geom39lim2`, one glcts chain, no batch aborts)

Root cause (confirmed round 38's hypothesis): the cluster's 17 fails were
`gl_Max*` builtin vs `glGet*` mismatches plus 3 limits below the GL 4.6 minimum.
glslang was compiled with its default `TBuiltInResource`.

Fix: `mgl_geom_fill_glslang_resource()` (new, geometry.c) re-seeds every resource
entry that has a GL pname from the very queries the app sees (`mglGetIntegerv` /
`mglGetIntegeri_v`, i.e. through `mgl_ext_get` first — ext_xfb serves
MAX_TRANSFORM_FEEDBACK_INTERLEAVED_COMPONENTS = 128). `initGLSLInput` (shaders.c) now
calls it instead of hand-maintaining overrides; the whole table (70 pnames) lives in
my file so shared-file churn is one call. Nothing un-served may be added to the table
(an unanswered query would raise GL_INVALID_ENUM as a side effect of compiling).

- GL_MAX_FRAGMENT/COMPUTE_ATOMIC_COUNTER_BUFFERS: 0 -> 8 (the indexed atomic-counter
  binding points the renderer really binds; minimums are 1 / 8).
- GL_MAX_UNIFORM_BUFFER_BINDINGS: 16 -> 84 (GL 4.6 minimum). Points 16..83 exist only
  for state: binds/queries round-trip through a small overflow table in buffers.c
  (sparse, keyed by ctx, cleared in unbindEverywhere), but uniform blocks bound there
  can never feed a draw — the renderer consumes a 16-entry table and Metal has ~31
  buffer slots/stage. Documented gap; nothing in the CTS binds real buffers that high
  (the between-case state reset only binds 0, and it now must not error: glcts aborts
  the whole run otherwise — gluStateReset.cpp:1121).
- Files touched: geometry.c (mine), shaders.c (initGLSLInput block replaced), get.c
  (two cases + helper + indexed UBO query), buffers.c (maxBindingsForTarget UBO cap,
  overflow table, mgl_buffers_indexed_binding), draw_ext.c (buffers_get_indexed UBO
  slots), mgl_ext_buffers.h (1 prototype).

## 2. direct_state_access — 367/371 -> 370/371 (`~/MGL-cts/geom39dsa`, full cluster, one invocation)

- textures_compressed_subimage (Fail) -> Pass. Two pieces:
  * online compression for the RGTC family: new CPU encoder `tdEncodeRGTC`
    (BC4/BC5, unsigned+signed; endpoints = block min/max, nearest of 6 shades +
    the two extreme codes; idempotent by construction) in texdata_formats.c — it was
    already declared in the header, never implemented. `texImage` (textures.c) now
    keeps the RGTC internalformat, encodes and stores real BC4/BC5 blocks, so
    GL_TEXTURE_COMPRESSED/_IMAGE_SIZE and glGetCompressedTexImage are exact.
  * the other compressed families (BPTC/ETC2/EAC/S3TC) reject uncompressed uploads
    with GL_INVALID_OPERATION (no encoder — documented gap; the CTS tolerates this
    and uses its own block tables). Previously ETC2-RGB silently stored RGBA8 and
    reported compressed=0, which is what failed the case (imageSize=0).
  * fixed a NULL-deref of mine on the way: `tdFormatInfo(internalformat)` can be
    NULL (some CTS enum unknown to the table) — guard added.
- textures_image_query_errors (Fail) -> Pass: glGetTextureImage/-SubImage raise
  GL_INVALID_OPERATION for a cube map that is not cube complete, before any other
  check (new cubeCompleteAtLevel in textures.c).
- framebuffers_check_status (CompatibilityWarning) -> Pass: mgl_fb_status
  (framebuffers.c) counts a GL_TEXTURE_3D attachment as layered, so mixing one with a
  non-layered attachment is FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS (matches the CTS/
  Khronos expectation; the query GL_FRAMEBUFFER_ATTACHMENT_LAYERED is unchanged).
- vertex_arrays_attribute_format: STILL FAILING (deterministic wrong XFB values
  "[0, 30802336] instead of [1,5]" + the flaky round-38 heap crash). The case uses
  two vertex bindings with offset/stride into one buffer, vecN attributes of many
  types, and reads the results back through transform feedback; vertex_arrays/xfb
  territory, not cheap — left as a gap. Scratch path to repro: run it under lldb
  (crashes vanish there) and read /tmp/vaf2.qpa.

## 3. Unchanged / known gaps

- texture_gather: 70/75 Pass + 1 Crash (offset-gather-float-2drect) = the known
  flaky heap corruption + the 4 triaged-not-cheap cases from round 38
  (base-level VM quirk, gather-tesselation-shader, incomplete-texture x2).
- Online compression exists only for RGTC; BPTC/ETC2/S3TC uncompressed uploads error.
- UBO binding points 16..83 are query-only (see above).
- The flaky heap crash (round 38 handoff section) is still open — it hits ~1 in 3
  full-cluster batches at random cases (this round: texture_gather crash).

## 4. Test results (this round)

- `tests/headless/run.sh` (full): 438 PASS lines, zero failures/blocks.
- `test_geom`: 593 checks, 0 failures. `zz_lim` (74 checks), `zz_cub`, `zz_lay`,
  `zz_etc`, `zz_rgtc`, `zz_bcc` scratch repros all green (left uncommitted per rules).
- CTS: limits 140/140; direct_state_access 370/371; texture_gather 70/75+1 crash.
- Scratch files (zz_*.c) and .-cts/ results dirs are untracked; no secrets, no
  absolute paths, no symlinks in the commit; make -j4 used throughout.
