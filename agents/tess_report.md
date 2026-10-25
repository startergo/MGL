# tess domain — round 21 report (tess / shader-linking / GLSL compat)

READY e4c890d

Merge `gl46/tess`; the branch tip after this report-only commit contains the
same code as `e4c890d` (merge of compat s12 `159d900`; code ends at `2a054e2`)

Session goal (supervisor, round 21): finish explicit_uniform_location, then other
KHR-GL46 Fail cases in the tess/GS/shader-linking domain.

## Commits this round (on gl46/tess, unmerged)

| commit | what |
|---|---|
| `9c018e3` | (from round 20 tail, committed this session) textureGatherOffset with runtime offsets rewritten to snapped-coordinate textureGather |
| `8d0c6c5` | glGetProgramResourceLocation flattens arrays-of-arrays subroutine uniforms |
| `ddbc249` | merge gl46-port (compat s11 `5e02260`) |
| `aedf12d` | TES textures bound at vertex slots + commit pending uploads before the private tess compute buffer |
| `f98eaf9` | gl_PerVertex member parse fix (`[]` strip) + link-time gl_PerVertex interface validation |
| `2a054e2` | PROGRAM_SEPARABLE latched at link time; pipeline info-log INVALID_VALUE |
| (merge) | gl46-port compat s12 `159d900` |

## CTS results (before -> after, verified on hardware this round)

* `explicit_uniform_location` **16/16** r18-failures now Pass (48/48 total).
  Last two: `subroutine-loc-arrays-mix` (glGetProgramResourceLocation now
  flattens `u[i][j]` against the declared dims exactly like
  glGetUniformLocation), `uniform-loc-mix-with-implicit-max[-array]` (slow, pass
  with a >150 s per-case timeout).
* `texture_gather`: `base-level` and **`gather-tesselation-shader`** Fail -> Pass.
  - `gather-tesselation-shader` (gather inside TCS kernel / TES post-tess vertex
    function) needed two fixes:
    1. `bindTexturesToCurrentRenderEncoder` (MGLRenderer.m, shared file): with
       tessellation active the stage list already reads the TES's resources, but
       the sampled-image bind keyed on `stage == _VERTEX_SHADER`, so TES textures
       went to **fragment** slots and the post-tessellation vertex function read
       an unbound `texture(0)`.  Now any `stage != _FRAGMENT_SHADER` binds at
       vertex slots (same rule the storage-image loop already used).
    2. `MGLRenderer+Tessellation.m`: the emulated-tessellator path (every
       point_mode/isolines draw with a TCS — `need_readback`) builds a private
       command buffer, but deferred texture uploads / generateMipmap blits can
       still sit uncommitted on the shared `_currentCommandBuffer`; the queue
       could run the compute kernels first, so the TCS read zeros.  The shared
       buffer is now committed first (ordering only, no wait).
    Reproduced pixel-exact in `tests/headless/test_zz_ctstg.c` (verbatim CTS
    shaders/texture); `test_zz_tg.c` isolates the TCS gather.
  - `incomplete-texture[-last-comp]` still Fail: **texdata domain** — MGL
    silently truncates an inconsistent mipmap chain (level 1 redefined 1x1) and
    treats the texture as complete; GL requires the incomplete default
    texture (0,0,0,1).  Needs GL mipmap-completeness (§8.17.3) at bind time.
* `CommonBugs.CommonBug_PerVertexValidation` Fail -> Pass:
  - the gl_PerVertex expander stripped only `]` from a member name, leaving
    `gl_ClipDistance[` — an existing `gl_ClipDistance[]` member was never
    recognized and expanded in twice ("duplicate member name: gl_ClipDistance").
  - link-time gl_PerVertex interface validation (GL 4.6 §11.1.2.5): a consumer
    stage that redeclares the block with a subset demands those members from the
    producer's output block; mismatch fails the link with a message naming
    stage+member.  Only explicit consumer redeclarations are checked (an
    un-redeclared consumer accepts any producer subset —
    `test_compat_glsl glsl_150_gl_per_vertex_redeclaration` stays green).
    Implemented in `mgl_program_check_tess_link` (runs for every program), the
    ORIGINAL shader sources are parsed since the expander normalizes the blocks.
* `sepshaderobjs`: CreateShadProgApi, PipelineApi Fail -> Pass (3/4 now):
  - `GL_PROGRAM_SEPARABLE` queries report the value **latched at the last
    successful link** ("the state of the binary after it was linked"); a failed
    `glCreateShaderProgramv` link therefore queries FALSE, a never-linked program
    FALSE.  glUseProgramStages/pipeline validation use the latched value; the
    live parameter still round-trips through program binaries.
  - `glGetProgramPipelineInfoLog` with a nonexistent pipeline name returns
    GL_INVALID_VALUE (CTS-mandated, "error still missing from the spec").
  - `StateInteraction` still fails: a failed **relink** of a pipeline-referenced
    program must keep drawing with the last successful binary; MGL discards the
    stage executables on the failed link (needs link snapshot/restore).
* `tessellation_shader.gl_InvocationID_PatchVerticesIn_PrimitiveID` (r18
  "Timeout"): Passes now — it was the 1325 s batch deadline, the case alone
  takes ~4 s.

## Verification

* `make -j4 lib`, `make -j4 glx nsgl` clean (the two `incomplete-implementation`
  warnings about `extProgramLibraryForMSL` predate this branch — present with and
  without these changes, introduced by compat s11's category split).
* `tests/headless/run.sh`: **86 test binaries, 0 failures** (test_zz_gcts 4/4
  stable; earlier single flake re-ran green).  `test_tess` 499/0, `test_geom`
  603/0, `test_compat_glsl` 42/0, `test_cts` 71/0.
* Scratch tests added under tests/headless: test_zz_ctstg.c, test_zz_tg.c,
  test_zz_pv.c, test_zz_csp.c, test_zz_pil.c (all green, committed).

## Shared-file edits (this round)

* `MGL/src/MGLRenderer.m` — `bindTexturesToCurrentRenderEncoder`: TES-stage
  sampled textures bind at VERTEX slots (`stage != _FRAGMENT_SHADER`), see above.
* `MGL/src/program.c` — latch `separable_latched` on successful link;
  `GL_PROGRAM_SEPARABLE` query returns the latch.
* (round-20 tail committed here: `MGL/src/cts_glsl_compat.c` gather-offset
  rewrite, `MGL/src/shaders.c` MGL_SRC_DUMP debug hook, now in append mode.)
* All other changes are in my own files: `MGL/src/tessellation.c`,
  `MGL/src/MGLRenderer+Tessellation.m`, `MGL/src/program_introspection.c`,
  `MGL/src/program_ext.c`, `MGL/src/mgl_ext_program.h`.

## Parked / leftovers

* `.agent/test_zz_vssparse.c.bak` — round-20 scratch: `BindBufferRange` (offset
  256, size 8) UBO binds return no data in the VS (buffers domain).
* `.agent/test_zz_gather2.c.bak` — round-20 WIP gather-vs-CPU-reference scratch;
  its own harness was broken (constant output for every offset/uv, also for the
  un-rewritten no-offset program).  The committed rewrite is validated by
  test_zz_gcts.c (5 exact cases) and the CTS texture_gather group.
* `.agent/test_zz_si.c.bak` — StateInteraction repro sketch (pipeline draw black
  even before the relink; superseded by the CTS finding above).
* Raw-Metal probes kept for reference: `.agent/probe_tcsread.m`,
  `.agent/probe_tcsread2.m` (compute-kernel texture reads with a 6-level
  RGBA32F texture work fine on AGX — isolated the MGL-side ordering bug).

## Known gaps (not fixed this round)

1. `texture_gather.incomplete-texture[-last-comp]`: mipmap-completeness
   (inconsistent chain sizes) not evaluated at bind time — texdata domain.
2. `sepshaderobjs.StateInteraction`: failed relink must preserve the last good
   binary for pipeline draws.
3. `shaders.struct.uniform.sampler*` (6) / `uniform_block.random.*` (7) /
   `program_interface_query` (13) / `packed_pixels` (24): not started
   (some are other domains).
