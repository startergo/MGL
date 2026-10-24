# compat domain — progress log

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
