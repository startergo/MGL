# geom round 22 — supervisor item: depth/stencil blits into multisampled framebuffers

Branch `gl46/geom`, commit `2582e26` (on top of merged round-21 work at `55ad0dd`).

## What was implemented

`glBlitFramebuffer` with a **multisampled draw framebuffer** previously hit
`MGLRenderer+Framebuffer.m:~1361` → `NSLog(@"scaled/partial depth-stencil blit into a
multisampled image is not supported")` + skip. That branch is now a real implementation.

### Validation (`MGL/src/framebuffer_ext.c`, one rule changed)

* The "source and destination rectangles must be identical" rule now only fires when
  the **read** framebuffer is multisampled (`rs > 0`), per GL 4.6 §18.3.1 — a
  single-sampled read framebuffer may be blitted into a multisampled draw framebuffer
  with any (scaled/partial) rectangle. Sample-count mismatch (both MS) and
  `GL_LINEAR` + `DEPTH/STENCIL` → `GL_INVALID_OPERATION` were already correct and are
  unchanged. The existing MS→MS same-size exact-copy path stays legal.

### Metal path (`MGL/src/MGLRenderer+Framebuffer.m`, new section)

1. **Source staging** — `fbStageAspect:aspect:` copies the needed aspect of the
   single-sampled source into a shader-readable color texture through one blit
   encoder (buffer round-trip inside the same command buffer, no CPU wait):
   * packed D24S8/D32FS8 depth → `MTLBlitOptionDepthFromDepthStencil` → one 32-bit
     float per pixel → **R32Float**;
   * plain Depth32Float → 4 raw bytes → R32Float; Depth16Unorm → 2 bytes → **R16Unorm**
     (shader reads it as `texture2d<float>`, i.e. exactly GL's unorm depth decode);
   * stencil (packed or Stencil8) → 1 byte → **R8Uint** (read as `texture2d<uint>`).
   A multisampled *read* framebuffer (only reachable for scissored/split same-size
   MS→MS blits) is resolved first by the existing `fbResolveDepthStencil:` helper,
   which gained a `slice:` parameter so layered sources work.
2. **Render pass** — `fbDSBlitEncode:` draws a full-screen triangle into the MS
   destination with the pipeline's `rasterSampleCount` = destination samples; the
   fragment function maps the destination pixel center to the source pixel
   (`src0 + (pos.xy - dst0) * scale`, the same nearest mapping as the CPU resample
   path and the color shader-blit path; handles mirrored rects via negative scale),
   discards out-of-source-bounds pixels, and writes `[[depth(any)]]` and/or
   `[[stencil]]`. One fragment invocation per pixel whose outputs the rasterizer
   replicates over **every sample** — exactly the "each destination sample takes the
   mapped source pixel" semantics of a blit into MS. Destination rect is clipped to
   the image and to the GL scissor (`fbScissorFor`); both aspects of a packed
   depth/stencil texture are attached with load/store so the untouched aspect is
   preserved.
3. **Depth** — depth compare Always + write enabled, the shader outputs the sampled
   depth. Works on every GPU.
4. **Stencil** — prefers fragment stencil export (`uint st [[stencil]]` with
   Always/REPLACE, write mask 0xff). Export support is probed by pipeline creation;
   on failure a `dsblit:no-stencil-export` flag is cached and stencil is written with
   one stencil-REPLACE pass per distinct source stencil value (the ≤256 distinct
   bytes are read back once from the staged R8Uint texture; each pass keeps only
   pixels whose source stencil equals the reference). **Both paths were verified**
   (the flag was force-enabled in a scratch build and the whole test passed).
   Depth+stencil requests share one pass when export is available.
5. Pipelines / depth-stencil states are cached in the existing fb cache keyed by
   mode+format+samples; MSL is generated from a small template.

Bug found on the way: the params struct must match MSL's `float2` alignment (40
bytes, not 36) or `setFragmentBytes` fails Metal draw validation — with the API
validation layer off this silently dropped draws (that is what the "phantom" failures
during test development were).

## Shared-file edits

* `MGL/src/framebuffer_ext.c` — the one validation rule described above (3 lines).
* `MGL/src/MGLRenderer+Framebuffer.m` — replaced the NSLog-and-skip branch with a call
  to the new `fbBlitDepthStencilToMS:`; added the new section (staging, pipeline
  builder, depth-stencil states, value readback, encoder, driver method) in the same
  file next to the other blit helpers so it can use the file-local statics;
  `fbResolveDepthStencil:` gained a `slice:` parameter (single caller updated).
* `tests/headless/test_fb.c` — new `TEST(blit_depth_stencil_to_multisample)` + `RUN`.

## Tests

`tests/headless/test_fb.c::blit_depth_stencil_to_multisample` (293 checks total in the
file, all passing) covers: scaled 2× depth+stencil blit with four quadrant values
(depth **and** stencil verified through a resolve), nearest edge pixels, minified
(8×8→4×4) depth-only blit with untouched stencil, partial destination rectangle,
stencil-only blit, scissored blit (clipped half unchanged), a float-depth
(`GL_DEPTH_COMPONENT32F`) MS target, rendering a depth-tested quad against the blitted
depth (`GL_LESS`), and errors: `GL_LINEAR`+depth, scaled read from a multisampled
framebuffer, and legal resolve afterwards.

```
== 293 checks, 0 failures        (test_fb)
total checks: 4100 failing tests: 0     (full tests/headless/run.sh)
```

`make lib` clean; `agents/stub_scanner.py` clean for the touched files; scratch
diagnostics removed; work committed (`2582e26`).

## Gaps

* A color `GL_LINEAR` blit into a multisampled draw framebuffer is still accepted
  (the pre-existing shader-blit path handles it). Desktop GL arguably requires
  NEAREST there too; left as-is because it is outside this item's scope, works
  correctly, and no test asserts the error.
* The per-reference stencil fallback is at most 256 passes (one per distinct value);
  correct but slow — only used on GPUs without stencil export (Apple GPUs export).
* `glBlitFramebuffer` remains CPU-resampled (as before) when the draw framebuffer is
  single-sampled; only the multisampled destination case moved to the GPU.
