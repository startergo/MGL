# GPU audit — CPU work on the draw / dispatch / upload / readback path

Branch `gl46/gpuaudit` (merged `gl46-port` incl. tess at a3f37f7).

**Baseline** (before any change, `tests/headless/run.sh`): **3859 checks, 0 failures**
(buffers 267, debug 290, fb 224, introspect 331, program 226, queries 114, regress 21,
rstate 323, smoke 8, texdata 240, texext 121, texparam 706, vattrib 439, xfb 249).
**Final**: **3897 checks, 0 failures** (same suites plus test_gpuaudit 104 and the three
tess suites 44/2/4 that came in with the tess merge).

Timings are medians of 5 interleaved runs of `tests/headless/ab_bench.c` against the
baseline build (`tests/headless/ab_run.sh`), `-O0`, on the development Mac while a VM
was loading it — absolute numbers move with the load, the ratio is what matters.
"submit" = the GL call only (no `glFinish`), i.e. the CPU cost of the command.

| case (before → after) | old | new | speedup |
|---|---|---|---|
| 2000 identical `glDrawArrays` | 4.6 µs/draw | 2.4 µs/draw | **1.9x** |
| 1000 draws with blend state flips | 46.6 µs/draw | 15.8 µs/draw | **3.0x** |
| 2000 `GL_UNSIGNED_BYTE` indexed draws | 6.7 µs/draw | 1.9 µs/draw | **3.5x** |
| 2000 warm `GL_TRIANGLE_FAN` draws (64 v) | 5.5 µs/draw | 3.2 µs/draw | **1.7x** |
| 20 draws of 512k indices, restart on (submit) | 283 µs/draw | 7 µs/draw | **38x** |
| 16 MiB `glClearBufferData` of a GPU-resident buffer (submit) | 2.15 ms | 25 µs | **87x** |
| `glReadPixels` 512² RGBA into a PBO | 5.7 ms/read | 1.1 ms/read | **5.4x** |
| 2000 draws with a `GL_DOUBLE` attribute (20k verts) | 165.7 µs/draw | 0.82 µs/draw | **202x** |
| transform-feedback capture, 30k vertices | 333 µs/draw | 105 µs/draw | **3.2x** |
| 50 × 1000-command `glMultiDrawElementsIndirect` | 482 µs/call | 460 µs/call | 1.04x (see F11) |

---

## Findings and what was done

### F1 — Objective-C runtime reflection in every draw call *(fixed)*
`MGLRenderer.m:58 mglCallExtHooks()` ran `class_copyMethodList()` + `sel_getName()` +
`strncmp()` over every method of MGLRenderer **per draw** (plus per render pass and per
commit). A sampling profile of 200k identical draws showed ~27% of the samples in that
loop. *Fix*: the resolved IMP list is memoised per (class, suffix) — a method list is
immutable once the class is realised. Draws got 1.9x cheaper on the CPU.

### F2 — `MTLRenderPipelineState` never cached *(fixed)*
`MGLRenderer.m processGLState:` built a full `MTLRenderPipelineDescriptor` and called
`newRenderPipelineStateWithDescriptor:` (plus 2–3 unconditional `NSLog`s) whenever any
of `DIRTY_PROGRAM|DIRTY_VAO|DIRTY_FBO|DIRTY_ALPHA_STATE|DIRTY_RENDER_STATE` was set —
and `glFlush`, compute dispatches, barriers and readbacks all set `DIRTY_ALL`, so this
happened constantly. Measured: 1000 blend-state flips = **1004 PSO builds**.
*Fix*: descriptor-keyed cache (`MGLRenderer+Gpuaudit.m`): every field that can change
the compiled pipeline (functions, vertex descriptor, 8 colour attachments with blend
state, depth/stencil formats, sample count, tessellation fields, amplification count)
is hashed into the key. State churn is now a dictionary lookup: 3.0x on blend flips.

### F3 — per-draw CPU scan of the whole index buffer (primitive restart) *(fixed)*
`state_ext.c mgl_rstate_index_rewrite_needed()` scans every index of the range on the
CPU to decide whether a rewrite is needed (28 ns/index; 26 ms for a 512k-index draw).
A GPU decision is impossible (the answer is needed synchronously), so the *work is
avoided*: the answer is cached per (element buffer **content generation**, offset,
mode, type, count, restart index, provoking convention, program). Every path that
changes a buffer's CPU-visible contents bumps the generation
(`buffers.c bufferDidModify/initBufferData`, `draw_ext.c mgl_buffers_prepare_cpu_read`,
`queries.c` GPU writes); persistently mapped buffers are never cached. Submit cost of a
512k-index restart draw: 283 µs → 7 µs.

### F4 — per-draw CPU index rewriting / primitive emulation / 8-bit widening *(fixed)*
`rs_rewrite()` (restart splitting, provoking rotation), `build_emulated()` (line loop,
triangle fan, adjacency), `GL_UNSIGNED_BYTE` widening — each produced a fresh malloc'd
list per draw and `extBufDrawIndexed` uploaded it with `newBufferWithBytes:` (a new
MTLBuffer **per draw**). *Fix*: the produced list is uploaded once into a private
MTLBuffer and cached under the same key as F3; repeated draws encode a Metal indexed
draw straight from it (new `draw_indexed_mtl` hook). Fan draws 1.7x, ubyte draws 3.5x.
Cached draws are verified byte-identical to the freshly built ones by
`test_gpuaudit.c` (fan, restart split, ubyte+restart, provoking flip, re-upload
invalidation).

### F5 — `glClearBuffer{Sub}Data` cleared texel-by-texel on the CPU *(fixed)*
`buffers.c clearBufferData()` looped `memcpy(dst+i, texel, texel)` — 16.7 M calls for a
16 MiB `GL_R8` clear (22 ms). *Fix* (`MGLRenderer+Gpuaudit.m`): 1-byte texels use
`[blit fillBuffer:range:value:]`; 2–16-byte texels seed a 64 KiB pattern block and
double it up inside the destination (8 blit copies for 16 MiB). Fully asynchronous —
the CPU never touches the range, and the pre-clear `prepare_cpu_write` (which renamed
the buffer, a full copy) is skipped on this path. Submit: 2.15 ms → 25 µs (87x).
CPU-only buffers keep the CPU path (promoting them would cost a full-copy allocation).

### F6 — `glReadPixels` into a `GL_PIXEL_PACK_BUFFER` still went through the CPU *(fixed)*
Even with a PBO bound the path was GPU blit → temp buffer → `waitUntilCompleted` →
per-pixel CPU decode → per-pixel CPU pack → `memcpy` into the PBO (13 ms for 512² plus
a pipeline stall). *Fix*: when the read surface layout matches the request exactly
(RGBA8/RGBA8_sRGB/BGRA8 surfaces with `GL_RGBA`/`GL_BGRA` + `GL_UNSIGNED_BYTE`, integer
RGBA8, `Depth32Float` with `GL_DEPTH_COMPONENT/GL_FLOAT`) and the rectangle is fully
inside, one `copyFromTexture:toBuffer:` blits straight into the PBO's MTLBuffer with
`destinationBytesPerRow` = the GL pack stride; skip rows/pixels and pack alignment are
honoured via the destination offset. Everything else (sRGB, conversion, partial rects,
stencil) keeps the CPU path. 5.4x and no stall.

### F7 — indexed draws synced the element buffer even when nothing read it *(fixed)*
`exec_draw_raw` called `element_pointer()` (→ `mgl_buffers_prepare_cpu_read`, which
waits for GPU writes) for **every** indexed draw. A fully native draw (native mode,
16/32-bit indices) now only validates the range against the buffer size —
compute-generated indices can no longer stall a native draw.

### F8 — CPU vertex-attribute conversion per draw *(fixed; was deferred to vattrib)*
`vattribBindShadow()` converted `GL_DOUBLE`/`GL_FIXED`/unfetchable formats into
float4/int4 streams per draw: `calloc` + per-element conversion + `newBufferWithBytes:`
(165 µs for 20k double4 vertices). *Fix*: the converted stream is cached as an MTLBuffer
keyed on (vertex buffer content generation, binding offset, stride, format, kind) and
bound directly. 202x. Re-uploading the attribute data invalidates and rebuilds (tested).

### F9 — transform feedback: synchronous CPU round trip per capture draw *(fixed)*
`MGLRenderer+TransformFeedback.m xfbCapture` ran the capture VS into a scratch SSBO,
`waitUntilCompleted`, then a CPU loop memcpy'd every captured varying of every output
vertex into the XBOs. *Fix*: the capture pass is now encoded but **not** waited on
(`xfbCaptureAsync`), and a compute kernel (`mgl_xfb_scatter`, compiled once from MSL in
`MGLRenderer+Gpuaudit.m`) appends to the same command buffer and copies the records
into the XBOs on the GPU. Everything the kernel needs (assembled primitive table,
strides, varying layout, write offsets) is known on the CPU — the counts that
`glGetTransformFeedbackPrimitiveCount`/queries report were always CPU-computable. The
scatter command buffer becomes the renderer's current command buffer so the buffer
serial tracking (and therefore a later CPU read of an XBO) waits on exactly it.
Falls back to the original synchronous CPU path when the kernel or a layout is
unsupported, or a buffer has no MTLBuffer. 30k-vertex captures: 333 µs → 105 µs.
Byte-exactness is verified against the CPU path's values by `test_xfb.c` (249 checks)
and a dedicated test in `test_gpuaudit.c`.

### F10 — a staging `MTLBuffer` allocated per texture upload *(measured, rejected)*
A pooled staging ring was implemented (`gpauauditStageBytes`, reset at command-buffer
boundaries, growing up to 32 MiB) and measured: 64×64 uploads 0.97x, 512×512 uploads
1.02–1.12x — Metal already pools small shared allocations and the per-upload cost is
the copy itself (bandwidth bound). The extra ring-reuse complexity bought nothing, so
it was reverted; the finding is documented at the call site
(`MGLRenderer+TextureData.m texdataEncodeUpload`).

### F11 — multi-draw indirect read each command on the CPU *(fixed; gain is small)*
`mgl_buffers_multi_draw_indirect` memcpy'd the 5-word command per sub-draw (only to
feed the pre-draw hooks) and encoded one Metal indirect draw per command. *Fix*: the
counts are only needed while a CPU-counted pipeline-statistics query is active (new
`mgl_queries_counting_active()`), so otherwise every command is handed to Metal
**unread** (`drawIndexedPrimitives:...indirectBuffer:` — the same mechanism the single
indirect draws already use). Measured only 1.04x: the remaining cost is Metal command
encoding itself (~0.46 µs per command), which only an indirect command buffer could
remove. That remains the documented next step: a compute kernel expanding the GL
command array into an `MTLIndirectCommandBuffer` (the `.agent/ref/poly/cl/restart.h`
pattern: upper-bound allocation + GPU-written draw records), then one
`executeCommandsInBuffer` per pass.

### F12 — stalls that are inherent to the GL API *(no action)*
`glFinish`, `glGetQueryObject(GL_QUERY_RESULT)`, `glMapBufferRange(GL_MAP_READ_BIT)`,
`glGetBufferSubData`/`glGetTexImage`, synchronous `glReadPixels` without a PBO,
`texdataSubmitAndWait`. These are spec-mandated synchronisation points; the
implementations already use serial tracking + `synchronizeResource`/shared-event waits
rather than polling.

### F13 — already on the GPU *(no action)*
Clears (render-pass load action), framebuffer blits (`fbBlit` shader handles format
conversion + MS resolve), mipmap generation (`generateMipmapsForTexture`),
`glCopyBufferSubData` (blit), `glCopyTexSubImage`, compressed-texture decode (compute),
compute dispatches (PSO cached on the function), fences/shared events.

### For tess / geom
* The PSO cache key (F2) already covers every tessellation field of
  `MTLRenderPipelineDescriptor`, and the index caches are keyed on the GL draw mode, so
  the tess patch path (which draws `GL_PATCHES`) can never be served a cached rewrite.
* If the tess/geom agents add geometry-dependent rewrites of their own, they should
  extend `MGLIndexKeyState` (`mgl_ext_gpuaudit.h`) with the state they depend on — it
  is part of every cache key.
* `MGLRenderer+Gpuaudit.m` now owns the `_currentCommandBuffer` handoff for the xfb
  scatter; anything that encodes work between an xfb capture and the next encoder
  creation gets a fresh command buffer because the capture buffer is committed.

---

## Shared-file edits (all surgical, one purpose each)

| file | edit |
|---|---|
| `MGL/include/glm_context.h` | 2 lines: `ext_gpuaudit` slots in `GLMState`/`GLMMetalFuncs` |
| `MGL/include/mgl_ext.h` | forward declarations + `mgl_ext_init_gpuaudit/_get_gpuaudit` prototypes |
| `MGL/src/mgl_ext.c` | 2 fan-out lines |
| `MGL/src/MGLRenderer.m` | F1 memoised hook fan-out; F2 cache lookup/store around the PSO build; import |
| `MGL/src/buffers.c` | F5 GPU-clear dispatch + `mgl_gpuaudit_invalidate` in `bufferDidModify`/`initBufferData` |
| `MGL/src/draw_ext.c` | `mgl_gpuaudit_invalidate` when a GPU write becomes CPU-visible |
| `MGL/src/queries.c`, `mgl_ext_queries.h` | `mgl_queries_counting_active()` (F11) |
| `MGL/src/framebuffer_ext.c` | F6 PBO dispatch before the CPU read path |
| `MGL/src/draw_buffers.c` | F3/F4/F7 cache hooks in `exec_draw_raw`/`draw_index_list`; F11 `need_counts` |
| `MGL/src/transform_feedback.c`, `MGLRenderer+TransformFeedback.m`, `mgl_ext_xfb.h` | F9 async capture + GPU scatter with the CPU path kept as fallback |
| `MGL/src/MGLRenderer+VertexAttrib.m` | F8 cached shadow stream (lookup before, store after the existing conversion) |

New files owned by this domain: `MGL/src/mgl_ext_gpuaudit.h`, `MGL/src/gpuaudit.c`,
`MGL/src/MGLRenderer+Gpuaudit.{h,m}`, `tests/headless/test_gpuaudit.c`,
`tests/headless/ab_bench.c`, `tests/headless/ab_run.sh`.

## Known gaps / pre-existing issues found while auditing (not fixed, other domains)

* `glPrimitiveRestartIndex(0xFFFFFFFF)` with `GL_UNSIGNED_BYTE` indices: the restart
  value is not masked to the index width, so 0xFF bytes are not treated as restart
  (rstate domain; reproduced against the baseline).
* MGL renders `GL_TRIANGLE_FAN` with ~8% more pixels than the equivalent triangle list
  (pre-existing emulation/rasterisation difference; identical in the baseline).
* `test_texparam` intermittently aborts under heavy machine load — reproduces on
  pristine `gl46-port` as well.
* XFB capture still allocates its own command buffer per capture draw (the kernel
  removed the wait and the CPU scatter, not the extra command buffer).
