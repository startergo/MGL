# GPU audit — CPU work on the draw / dispatch / upload / query path

Branch `gl46/gpuaudit`. Baseline (full `tests/headless/run.sh`, before any change): **3859 checks,
0 failures** (buffers 267, debug 290, fb 224, introspect 331, program 226, queries 114, regress 21,
rstate 323, smoke 8, texdata 240, texext 121, texparam 706, vattrib 439, xfb 249).

Measurements below were taken with micro-benchmarks on this machine
(`tests/headless/bench_scratch.c`, `-O0` build, default framebuffer, 2000 iterations unless noted).
"per draw" = once per `glDraw*` call, "per upload" = once per `glTexSubImage*`/`glBufferSubData`,
"per pass" = once per Metal render pass.

---

## 1. Findings, ranked by measured impact

### F1 (fixed) — Objective-C runtime reflection in every draw call
`MGLRenderer.m:58 mglCallExtHooks()` — `class_copyMethodList()` + `sel_getName()` +
`strncmp()` over every method of `MGLRenderer` is executed **per draw**
(`processGLState:` line 3282 calls it with `"PreDraw:"`), and also per render pass
(`"RenderPassDescriptor:"`, `"RenderEncoderCreated:"`) and per commit (`"WillCommit:"`).
Sampling profile of 200k identical `glDrawArrays`: ~27% of the samples inside
`mglCallExtHooks` (mostly `class_copyMethodList`, `method_getName`, `strlen`).
GPU alternative: none needed — the method list of a class is immutable after the first
call; memoise the resolved `IMP` list per suffix. Owner: **gpuaudit** (shared file,
surgical). Effort: small. Risk: low.

### F2 (fixed) — `MTLRenderPipelineState` rebuilt from scratch on state changes, never cached
`MGLRenderer.m:3109-3210 processGLState:` — whenever any of
`DIRTY_PROGRAM|DIRTY_VAO|DIRTY_FBO|DIRTY_ALPHA_STATE|DIRTY_RENDER_STATE` is set, a full
`MTLRenderPipelineDescriptor` is built and `newRenderPipelineStateWithDescriptor:` is
called (2-3 unconditional `NSLog`s included). Measured: 1000 `glDrawArrays` with
`glEnable/glDisable(GL_BLEND)` between draws = **1004 PSO creations, 31.5 µs/draw**
(warm identical draws are 2.9 µs/draw). Anything that sets `dirty_bits = DIRTY_ALL`
(`glFlush`, barriers, compute dispatch, xfb capture, readback) forces a rebuild on the
next draw as well. GPU alternative: Metal itself caches nothing for the caller; the fix
is a descriptor-keyed PSO cache (this is what every real Metal backend does).
Owner: **gpuaudit** (shared file, surgical). Effort: medium. Risk: medium (cache key
must cover every descriptor field that can change).

### F3 (fixed) — per-draw CPU scan of the whole index buffer (primitive restart)
`state_ext.c:748 mgl_rstate_index_rewrite_needed()` — when primitive restart is enabled
and the mode is a list/points/strip, the **entire index range is scanned on the CPU on
every draw** to decide whether a rewrite is needed (`draw_buffers.c:593`).
Measured: 100 × `glDrawElements(GL_TRIANGLES, 1M, GL_UNSIGNED_INT)` with restart enabled
= **26.5 ms per draw** (26 ns per index). GPU alternative: none (the decision is needed
synchronously); the work is *avoidable* — cache the answer keyed on the element buffer's
content generation. Owner: **gpuaudit** (+1 field in `mgl_ext_buffers.h`, 2 bump sites in
buffers.c / draw_ext.c). Effort: medium. Risk: medium (stale cache ⇒ wrong draw).

### F4 (fixed) — per-draw CPU index rewriting / primitive emulation / 8-bit widening
* `state_ext.c:897 rs_rewrite()` (restart splitting, provoking-vertex rotation),
* `draw_buffers.c:362 build_emulated()` (line loop / triangle fan / adjacency modes),
* `draw_buffers.c:505` widening of `GL_UNSIGNED_BYTE` indices to 16 bit,
* `MGLRenderer+Draw.m:448 extBufDrawIndexed` then uploads the rewritten list with
  `newBufferWithBytes:` — a fresh `MTLBuffer` allocation **per draw**.
Measured: warm `glDrawArrays(GL_TRIANGLE_FAN, 64 verts)` = **13.3 µs/draw** vs 2.9 µs for
a native mode; 8-bit index draws 6.4 µs/draw. GPU alternative: a compute kernel can do the
rewrite, but the *output count* is data-dependent (restart positions), which would force
degenerate-tail padding and triple the vertex shader invocations for fans; caching the
rewritten list **as an `MTLBuffer`** keyed on the same content generation as F3 removes
all of the per-draw work with no GPU cost. Owner: **gpuaudit** (shared files, surgical).
Effort: medium-large. Risk: medium (same invalidation as F3).

### F5 (fixed) — `glClearBufferData` clears byte-by-byte on the CPU
`buffers.c:1186 clearBufferData()` — the range is filled with `memcpy(dst+i, texel, texel)`
in a loop, one `memcpy` per texel. Measured: 16 MiB `GL_R8` clear = **22.1 ms**
(16.7 M calls of `memcpy(...,1)`); `GL_RGBA32F` = 3.1 ms. GPU alternative:
`MTLBlitCommandEncoder fillBuffer:range:value:` (4-byte repeating pattern) for 1/2/4-byte
texels, staging-buffer + GPU `copyFromBuffer` loop for 8/12/16-byte texels — fully async,
no CPU pass over the data. Owner: **gpuaudit** (+ call site in buffers.c). Effort: small.
Risk: low.

### F6 (fixed) — `glReadPixels` into a `GL_PIXEL_PACK_BUFFER` still round-trips through the CPU
`framebuffer_ext.c:581 mgl_fb_read_pixels()` → `MGLRenderer+Framebuffer.m:1094 fbRead:`
— even with a PBO bound the path is: GPU blit to a temp shared buffer, `waitUntilCompleted`,
per-pixel CPU decode to float4, then a second per-pixel CPU pack loop, then `memcpy` into
the PBO. Measured: 512×512 `GL_RGBA/GL_UNSIGNED_BYTE` = **13.3 ms** and a full pipeline
stall. GPU alternative: when the read surface layout matches the requested
format/type (`GL_RGBA/GL_UNSIGNED_BYTE` from `RGBA8*`, `GL_BGRA` from `BGRA8*`, depth
`GL_FLOAT` from `Depth32Float`), blit the texture straight into the PBO's `MTLBuffer`
(`copyFromTexture:toBuffer:`) with the pack alignment applied to `destinationBytesPerRow`
— zero CPU pixels, zero stall, and the GL command stream keeps ordering because the blit
is encoded on the context's command buffer. Owner: **gpuaudit** (fb files, surgical).
Effort: medium. Risk: medium (format coverage must be exact; everything else falls back
to the existing CPU path).

### F7 (fixed) — indexed draws sync the element buffer to the CPU even when nothing reads it
`draw_buffers.c:581 element_pointer()` is called for **every** indexed draw, and
`mgl_buffers_prepare_cpu_read()` inside it waits for queued GPU writes whenever the
element buffer was written on the GPU (e.g. compute-produced indices) — even for a fully
native mode/type that never looks at the bytes. Fix: only take the CPU path when the draw
can actually need it (8-bit indices, non-native mode, adjacency, restart enabled).
Owner: **gpuaudit** (draw_buffers.c, surgical). Effort: small. Risk: low.

### F8 (deferred → vattrib) — CPU vertex-attribute conversion at every draw
`MGLRenderer+VertexAttrib.m:404 vattribBindShadow()` — attributes whose GL format Metal
cannot fetch (`GL_FIXED`, `GL_DOUBLE`, some packed int variants) are converted on the CPU
into a `float4/int4` shadow stream **per draw**, `calloc` + per-element conversion +
`newBufferWithBytes:` per draw (>4 KB case). GPU alternative: a compute kernel that
expands the format into a shadow buffer, or caching the shadow keyed on the vertex
buffer's content generation + layout (same mechanism as F3/F4). Owner: **vattrib**
(another agent's file). Effort: medium. Risk: medium.

### F9 (deferred → xfb) — transform feedback capture is a synchronous CPU round-trip per draw
`MGLRenderer+TransformFeedback.m:123 xfbCapture` runs the capture VS into a scratch SSBO,
`waitUntilCompleted`, and `transform_feedback.c` assembles primitives on the CPU for every
draw while capture is active; `glDrawTransformFeedback` reads the recorded count back.
GPU alternative: real XFB (`MTLBuffer` + `drawPrimitives` with a GPU-written vertex count
via `MTLIndirectCommandBuffer`/`setVertexBytes` of a count buffer) — a large project.
Owner: **xfb**. Effort: large. Risk: high.

### F10 (deferred → texdata) — a staging `MTLBuffer` is allocated per texture upload
`MGLRenderer+TextureData.m:206 texdataEncodeUpload` — `newBufferWithLength:` +
row-by-row `memcpy` per `glTexSubImage*`; downloads likewise (`texdataDownload:253`).
GPU alternative: a pooled/ring staging allocator, and `didModifyRange`-free shared
buffers. Owner: **texdata**. Effort: medium. Risk: low-medium.

### F11 (deferred) — multi-draw indirect issues one Metal draw per sub-draw and reads each command
`draw_buffers.c:917 mgl_buffers_multi_draw_indirect` — loops `drawcount` times; for each
sub-draw `read_indirect()` copies the 5-word command to the CPU (no stall, but a CPU
touch per command) and one Metal draw is encoded. GPU alternative:
`MTLIndirectCommandBuffer` + a compute kernel that expands the GL command array into ICB
render commands, then one `executeCommandsInBuffer` per pass. Owner: **gpuaudit**, but
deferred: ICB creation/limits and the tess/geom interaction (they also touch the draw
path) make this a poor risk/benefit right now. Effort: large. Risk: high.

### F12 (notes, no action) — stalls that are inherent to the GL API
`glFinish` (`MGLRenderer.m:3621`), `glGetQueryObject(..., GL_QUERY_RESULT)` 
(`MGLRenderer+Queries.m:255` — only when the caller asks for the blocking result),
`glMapBufferRange(GL_MAP_READ_BIT)` and `glGetBufferSubData`
(`draw_ext.c:165 mgl_buffers_prepare_cpu_read` → `extBufWaitSerial`), synchronous
`glReadPixels` without a PBO, `glGetTexImage`, `texdataSubmitAndWait`. These are
spec-mandated synchronisation points; the implementations already use
serial-tracking + `synchronizeResource` rather than polling, and queries use a pooled
visibility buffer so only the final read touches CPU memory. No change.

### F13 (notes, no action) — already on the GPU
Clears (render-pass load action or a clear shader, `fbClear:`), framebuffer blits
(shader-based `fbBlit`, handles format conversion + MS resolve), mipmap generation
(`generateMipmapsForTexture`), `glCopyBufferSubData` (blit encoder), `glCopyTexSubImage`
(blit), compressed-texture decode (compute), compute dispatches (PSO cached on the
`MTLFunction`), fence/`glClientWaitSync` (shared events, no CPU spin when signalled).

### For tess / geom (do-not-touch files)
* `draw_buffers.c:537 warn_patches_once` / `MGL_PRIM_POINT` handling: GL_PATCHES draws are
  skipped entirely today — the tess agent owns the patch path.
* `build_emulated()` adjacency cases (`GL_LINE*_ADJACENCY`, `GL_TRIANGLE*_ADJACENCY`) are
  only reachable with a geometry shader bound (core profile); the geom agent owns the
  adjacency-mode semantics. The cache in F4 is keyed on the mode, so geom's changes
  cannot be served stale results, but they should extend the key if they add new
  geometry-dependent rewrites (e.g. layer/viewports-per-invocation).
* `MGLRenderer.m generatePipelineDescriptor` calls `rstateConfigurePipelineDescriptor`
  and the tess pipeline properties; the PSO cache key (F2) already covers the
  tessellation fields of `MTLRenderPipelineDescriptor`.

---

## 2. Fixes (Phase 2)

| # | Fix | Mechanism | Files |
|---|-----|-----------|-------|
| F1 | memoise `mglCallExtHooks` IMP lists | static cache keyed by suffix string | `MGLRenderer.m` |
| F2 | PSO cache | descriptor-keyed `NSDictionary` → `id<MTLRenderPipelineState>` | `MGLRenderer.m`, new `MGLRenderer+Gpuaudit.m` |
| F3 | restart-scan cache | answer cached per (buffer, content generation, mode, type, count, restart index) | new `mgl_ext_gpuaudit.h`, `gpuaudit.c`, `draw_buffers.c`, `mgl_ext_buffers.h`, `buffers.c`, `draw_ext.c` |
| F4 | index rewrite / emulation / widening cache | rewritten list cached as an `MTLBuffer`, drawn directly | same as F3 + `MGLRenderer+Draw.m` |
| F5 | GPU buffer clear | `fillBuffer:range:value:` / staging + GPU copies | new `MGLRenderer+Gpuaudit.m`, `buffers.c` |
| F6 | PBO readback | `copyFromTexture:toBuffer:` into the PBO's `MTLBuffer` | new `MGLRenderer+Gpuaudit.m`, `framebuffer_ext.c` |
| F7 | skip element-buffer CPU sync when the draw cannot need it | control-flow change only | `draw_buffers.c` |

Validation: every fix has a value/pixel test in `tests/headless/test_gpuaudit.c` plus a
before/after timing, and the full suite must stay at 0 failures.
