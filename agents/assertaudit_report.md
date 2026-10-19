# assertaudit — removed `assert(0)` audit report

Branch `gl46/assertaudit`. Base: `792cbab` ("adding golden images directory"), audit tree:
`gl46-port @ 61b7ae5`..`65f5860`.

## 1. Method

1. `git diff 792cbab HEAD -- MGL/src` with a per-hunk scanner
   (`agents/assertaudit_extract.py`) → **67 hunks / 592 removed `assert(0)`-lines**
   (626 counting duplicated hunks across file moves).
2. `agents/assertaudit_map.py` maps each removed line to its enclosing function in
   the old tree (`/tmp`-side output; script kept in the repo).
3. `agents/assertaudit_cur.py` summarises the *current* implementation of every
   affected function (length, error calls) → candidates for review.
4. Manual review of every candidate against the GL 4.6 core spec, plus runtime
   probes for the "legal input must not error / must not be dropped" class
   (probes were kept as `test_assertaudit` cases; scratch probes deleted).
5. Current-tree smell scan (no assert history): `default:` branches of GL enum
   mappers, "unsupported"/"not supported" strings, error-only functions.

Class counts over the 592 sites:

| class | sites | notes |
|---|---|---|
| IMPL | 543 | real implementation now (domain agents + this audit) |
| ILLEGAL | 370 | `non_core_unimplemented.c`: compat-profile entry points; `GL_INVALID_OPERATION` is mandated for them in a core context (GL 4.6 appendix "Deleted Features", and per-function error clauses) |
| INTERNAL | 44 | internal invariants / defensive returns after rewrites (Metal resource mapping, hash tables, program introspection internals) — callers pre-validate |
| DEAD | 12 | deleted in fd0ddde |
| MISSING | 3 | fixed in 61b7ae5 and 65f5860 |

## 2. MISSING items (fixed, one commit each)

| file:line | function | legal input that used to fail | fix |
|---|---|---|---|
| `MGL/src/MGLRenderer.m` ~395 (`mapGLBuffersToMTLBufferMap`) | buffer → `[[buffer(n)]]` mapping for vertex/fragment/compute stages | draw / `glDispatchCompute(Indirect)` while a shader-declared UBO, SSBO or atomic-counter block has **no buffer bound**. Raised `GL_INVALID_OPERATION` and dropped the command. GL 4.6 (7.12 "Shader Memory Access", 9.2): unbound block ⇒ undefined accesses, the command still executes. | `nullShaderBuffer`: shared 64 KiB zero-filled `MTLBuffer` (associated-object cache, no header ivar) bound to the slot |
| `MGL/src/vertex_arrays.c` `mglBindVertexArray` / `MGL/src/vertex_attrib.c` `VAO() == NULL` guards / `draw_buffers.c` `validate_vao` | default VAO handling | the **default vertex array object (name 0)** did not exist: `glDrawArrays`, `glVertexAttribPointer`, `glEnableVertexAttribArray`, `glGetVertexAttrib*` with VAO 0 raised `GL_INVALID_OPERATION`. GL 4.6 §10.4: name 0 is a real (default) VAO; only *client-side* pointers are forbidden in core. | new ext slot `MGL/src/mgl_ext_assertaudit.h` + `MGL/src/assertaudit.c` creates it at context init and binds it; `glBindVertexArray(0)` returns to it. The `ARB_vertex_attrib_binding` entry points (`glVertexAttrib*Format`, `glVertexAttribBinding`, `glBindVertexBuffer(s)`, `glVertexBindingDivisor`, `glVertexAttribDivisor`) still generate `GL_INVALID_OPERATION` for the default VAO — spec text "no vertex array object bound" + Mesa core behaviour |
| `MGL/src/MGLRenderer.m` 3526 (`bindTexturesToComputeEncoder`) | compute-stage texture/image binding | `glDispatchCompute(Indirect)` while the kernel declares a **sampled texture or storage image whose unit is unbound** → `return false`, the whole dispatch silently did nothing. GL 4.6: fetches are undefined, the dispatch executes. | `computeFallbackTexture` / `computeFallbackSampler`: shared 1×1 black RGBA8 (read+write) texture + nearest/clamp sampler bound to the missing `[[texture(n)]]` / `[[sampler(n)]]` slots |

### MISSING items owned by other active workers (not touched)

| file | item | owner |
|---|---|---|
| `MGL/src/pixel_utils.c` (`sizeForFormatType`, `sizeForType`, `numComponentsForFormat`) | legacy assert sites in the format-size mappers; tess is rewriting them together with the pixel-format mapper | tess |
| `MGL/src/MGLTextures.m` `mtlPixelFormatForGLFormatType` | same rework | tess |

## 3. DEAD code (deleted, commit fd0ddde)

| item | reason |
|---|---|
| `MGL/src/textures.c.bak`, `textures.c.with_debug`, `textures.c.with_lots_of_debug`, `mgl_funcs_to_be_implemented.c.bak` | stale copies, never compiled (Makefile globs `*.c`/`*.m`), ~7.9k lines |
| `MGLRenderer.m mtlStencilOpForGLOp:` | no callers (rstate `rs_stencil_op` owns stencil ops). The dead copy also switched on `var.stencil_fail` instead of its argument and mapped `GL_DECR`→`MTLStencilOperationInvert`, `GL_INCR_WRAP`→`DecrementClamp` |
| `MGLRenderer.m blendFactorFromGL:` / `blendOperationFromGL:` / `updateBlendStateCache` + 7 `_src_blend_*` / `_rgb_blend_operation` / `_color_mask` ivar arrays (`MGLRenderer_Private.h`) | computed a blend cache nothing ever read (`rstateConfigurePipelineDescriptor` owns pipeline blending); the dead mapper also mapped `GL_CONSTANT_COLOR`→`Source1Color` (should be `BlendColor`) and had no `GL_SRC1_*` cases |
| `MGLRenderer.m mtlReadDrawable:` + C wrapper + `bindObjFuncs` assignment | pixel reads go through the fb/texdata categories; `MGLRenderer+Framebuffer.m` rebinds the `mtlReadDrawable` pointer with `fbReadDrawableC` |

## 4. Full classification table (per file; groups with identical class collapsed)

Legend: IMPL = implemented, ILLEGAL = spec-mandated error, INTERNAL = internal
invariant, DEAD = deleted, MISSING = fixed.

| site (old) | function(s) | class |
|---|---|---|
| `non_core_unimplemented.c` (370 sites) | all compat-only entry points (`glColor*`, `glVertex*`, `glRect*`, `glEval*`, `glIndex*`, `glMaterial*`, `glLight*`, display lists, `glGetnMap*`, selection/feedback, ...) | ILLEGAL — core profile removes them; `GL_INVALID_OPERATION` is correct |
| `legacy_compat.c` (via `gl_core.c` removed asserts) | compat matrix stack, `glShadeModel`, fog, `glLineStipple`, `glPointParameter` min/max/attenuation | ILLEGAL — removed in core (`GL_POINT_SIZE_MIN/MAX/DISTANCE_ATTENUATION` are compat-only; core accepts only `GL_POINT_FADE_THRESHOLD_SIZE` and `GL_POINT_SPRITE_COORD_ORIGIN`, which are implemented) |
| `textures.c` (51) | DSA texture family, `mglTexImage*Multisample`, `mglTextureStorage*`, `mglClearTex(Sub)Image`, `mglTextureView`, `mglTextureBuffer(Range)`, `mglGet(Texture|Compressed)Texture*Image`, `mglBindImageTextures`, `mglInvalidateTex*`, `mglGetTexParameterI*`, `textureIndexFromTarget` | IMPL (texdata/texext/texparam domains; verified by probes: MS storage re-spec on a mutable MS texture, samples>max ⇒ `GL_INVALID_OPERATION` (spec), DSA on a never-bound texture ⇒ `GL_INVALID_OPERATION` (spec/Mesa)) |
| `framebuffers.c` (47) | full framebuffer/renderbuffer DSA + classic family, `framebufferTexture`, `mglBlitNamedFramebuffer`, `mglClearNamedFramebuffer*`, `mglCheckNamedFramebufferStatus` | IMPL (fb domain) |
| `program.c` (22) | `mglAttach/DetachShader`, `mglLinkProgram`, `mglUseProgram`, pipeline objects, introspection queries | IMPL (program/introspect domains) |
| `MGLRenderer.m` (27) | `createMTLTextureFromGLTexture`, `bindTexturesToComputeEncoder`, `newRenderEncoder`, `getProgramBinding/Count/Location`, `getVertexBufferIndexWithAttributeSet`, `swizzleTexDesc`, `createMTLSamplerForTexParam`, `mtlReadDrawable`, blend/stencil helpers, `updateDirtyBuffer` | IMPL (delegating wrappers) / INTERNAL (AGX-recovery defensive blocks, unreachable with validated input) / DEAD (§3) |
| `buffers.c` (12) | map/unmap/flush DSA, `mglBindBuffersRange/Base`, `mglInvalidateBuffer*`, `clearBufferData`, `bufferIndexFromTarget` | IMPL (buffers domain) |
| `uniforms.c` (10) | `mglGetUniform*`, `mglUniformBlockBinding`, `mglGetUniformIndices` | IMPL (delegates to introspect) |
| `tex_param.c` (10) | `setTexParmi/f/iv` family | IMPL (texparam domain) |
| `state.c` (9) | `mglPointParameter*`, `mglBlendFuncSeparatei`, `mglPolygonOffset`, `mglGetPointerv` | IMPL (rstate/texparam) |
| `fence.c` (9) | sync objects, `mglGetSynciv` (all four pnames), `mglClientWaitSync`, `mglWaitSync`, `mglDeleteSync(NULL)` | IMPL |
| `get.c` (6) | `mglGetString(i)`, `mglGetInteger64v/i_v`, `mglGetInternalformativ/i64v` | IMPL |
| `shaders.c` (4) | `glShaderTypeToGLMType`, `getGLSLStage`, `mglGetShaderiv`, `mglCompileShader` | IMPL (only the 6 core shader types exist; the defensive default is unreachable after `GL_INVALID_ENUM` validation) |
| `vertex_arrays.c` (3), `vertex_attrib.c` | `mglGetVertexArray*`, VAO creation/binding | IMPL + MISSING #2 (default VAO) |
| `rendering.c` (3) | `mglClear` | IMPL |
| `samplers.c` (2) | `mglSamplerParameterI*`, `mglGetSamplerParameterI*` | IMPL (verified by probe: last-write-wins border colour visible through both int and uint views) |
| `compute.c` (1), `draw_buffers.c` (1), `error.c` (1), `hash_table.c` (1) | `mglDispatchComputeIndirect`, `getTypeSize` (draw copy), `error_func`, `deleteHashElement` | IMPL / INTERNAL |
| `pixel_utils.c` (3) | format/type size mappers | owner: tess |

### Current-tree smells found (no assert history)

| item | class |
|---|---|
| `default:` branches of `getMTLPrimitiveType` returning `0xFFFFFFFF` for `GL_LINE_LOOP`, `GL_TRIANGLE_FAN`, adjacency and `GL_PATCHES` | INTERNAL-ish/legacy: this legacy path is rebound by the buffers domain (`exec_draw_raw` emulates every primitive); all modes verified rendering + error-free by probe. Kept as the default `mtl_funcs` fallback. |
| fragment/vertex/geometry texture binding skipping unbound units (`if (!ptr) continue;`) | accepted: no GL error, draw executes (verified by probe: unbound sampler in a fragment shader still renders); Apple silicon tolerates the unset `[[texture(n)]]` slot |
| `createMTLTextureFromGLTexture`'s "DISABLED: replaceRegion" upload code | unreachable for data-carrying textures (`texdataBindMTLTexture` realises/uploads all 8 standard targets, `texext` owns MS/buffer/view targets); it is only reached with no levels allocated, i.e. nothing to upload |
| `glLineWidth(4)` accepted while `GL_LINE_WIDTH_RANGE` is 1..1 | gap: Metal has no wide lines; the query stores the requested width unclamped. Hardware limit, reported honestly via the range |
| `GL_COLOR_LOGIC_OP` / `glLogicOp` | gap: state + queries implemented, but Metal has no colour logical operation and the rstate/program pipeline has no post-fragment hook to emulate it; enabling it currently has no raster effect (see §6) |
| `MGL/src/MGLRenderer+Framebuffer.m:417` rebinds `mtl_funcs.mtlReadDrawable` (field itself now has no callers) | vestigial func-pointer field, left in place (shared struct + another domain's file) |

## 5. Verification

* `tests/headless/run.sh` — all 23 suites: **3966 checks, 0 failures**
  (incl. the new `test_assertaudit`: 144 checks).
* `python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT` — clean for every file.
* `make -j8 lib` — no new warnings in touched files.
* Spec-behaviour notes updated in `tests/headless/test_buffers.c` (`draw_errors`)
  and `tests/headless/test_vattrib.c` (`setter_errors`, `pointer_errors`): both
  encoded the old, spec-violating "VAO 0 always errors" behaviour.

## 6. Remaining gaps / emulation limits (user decisions, not silent errors)

1. **Colour logical operation** (`glLogicOp` + `GL_COLOR_LOGIC_OP`): state,
   queries and error checks are implemented; Metal cannot express logic ops, and
   emulation would need a fragment-output rewrite or a post-resolve compute pass
   (a hook the program/rstate domains do not expose). Currently a silent no-op at
   raster time — flagged here so it is a recorded decision rather than a hidden one.
2. **Wide lines / wide points**: `GL_LINE_WIDTH_RANGE` / `GL_POINT_SIZE_RANGE`
   report the hardware capability (1..1 on Apple silicon); `glLineWidth` does not
   clamp the stored value to the range. Geometry-shader or fragment-emulated wide
   lines would be the emulation path.
3. **`pixel_utils.c` / `MGLTextures.m mtlPixelFormatForGLFormatType`**: owned by
   the tess worker; potential MISSING items there are listed in §2 and left to them.
4. **`mtlReadDrawable` func-pointer field** (`glm_context.h`): kept although no
   caller remains — removing it touches another domain's binding code.
