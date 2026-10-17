# MGL GL 4.6 port — brief for domain agents

Goal: every OpenGL 4.6 **core profile** entry point in MGL does real, spec-correct
work on Metal. No stubs: no empty bodies, no `(void)arg;`-only bodies, no
`assert(0)`, no "no-op"/"TODO" placeholders. Where Metal has no hardware
equivalent, emulate it (CPU-side, compute pass, shader rewrite via SPIRV-Cross
options); only where emulation is genuinely impossible, implement full GL state
tracking + queries + spec error checking and record the gap in your report.

## Where you work

* Repo on the Mac: `$MGL_PROJECT_DIR` (the main checkout; your task message gives the concrete path). **Your worktree:** `$MGL_PROJECT_DIR/wt/<domain>`
  on branch `gl46/<domain>`. Work ONLY inside your worktree.
* Run commands on the Mac with `mcp__remote-devices__serena__execute_shell_command`
  (pass `cwd` = your worktree). NEVER call serena `activate_project` (shared by all agents).
* Read/edit files with shell (`sed -n`, `python3` read-modify-write scripts) or serena
  file tools using paths **relative to the project root** (e.g. `wt/queries/MGL/src/queries.c`).
  Your own Read/Write/Edit tools act on a different machine — don't use them for repo files.
  When rewriting a file with python, always read it fresh in the same script.
* Build: `make -j8 lib 2>&1 | grep -E 'error|warning: implicit' ; echo rc=${PIPESTATUS[0]}`
  (in the worktree; `external/` is a symlink to the main checkout's prebuilt deps).
* Headless tests: `tests/headless/run.sh test_<domain>` — write `tests/headless/test_<domain>.c`
  using `tests/headless/mgl_test.h` (TEST/RUN/EXPECT_*; context is created automatically).
  Test error paths (`EXPECT_GL_ERROR`) and real results (query values, readback, buffer contents).
* Inventory: `python3 agents/stub_scanner.py --list EMPTY|NOOP|ASSERT --file <file.c>` — the scanner
  is a heuristic; also review "IMPLEMENTED" functions in your files for fake implementations.
* Commit to your branch with clear messages as you finish logical pieces
  (`git add -A MGL tests && git commit -m "..."`). Do not push. Do not merge.

## Architecture in 60 seconds

* `gl_core.c` → `ctx->dispatch.*` (`glm_dispatch.c`) → `mgl*(GLMContext ctx, ...)` in `MGL/src/*.c`.
  Dispatch is complete; you implement bodies.
* Errors: `ERROR_RETURN(GL_INVALID_ENUM)` / `ERROR_CHECK_RETURN(cond, err)` (see existing code); `STATE(x)` = `ctx->state.x`.
* Objects live in hash tables in `ctx->state` (`*_table`), see `glm_context.h`.
* Metal: `MGL/src/MGLRenderer.m` (class MGLRenderer). C calls it through `ctx->mtl_funcs.*` function
  pointers. Draw time: `processGLState:` translates dirty GL state; `newRenderEncoder` builds the
  render pass; `updateCurrentRenderEncoder` applies encoder state.
* Shaders: GLSL → glslang → SPIR-V → SPIRV-Cross → MSL (`program.c`, `shaders.c`, `bindMTLProgram:`).

## Ownership rules (parallel agents share one codebase)

1. **Your files** (listed in your task) — edit freely.
2. **Your extension slots** — no shared-header edits needed for new state or hooks:
   * `MGL/src/mgl_ext_<domain>.h`: define `struct MGLExtState_<domain>` and `struct MGLExtMetal_<domain>`.
   * `ctx->state.ext_<domain>` (allocate in `mgl_ext_init_<domain>()` in your primary file; ctx starts zeroed).
   * `ctx->mtl_funcs.ext_<domain>`: Metal hook table, filled in your category's
     `- (void) bindExt<Domain>FuncsToGLMContext: (GLMContext) glm_ctx` (called automatically).
   * `mgl_ext_get_<domain>()`: handle extra glGet pnames (`mgl_get_store()` helper in `mgl_ext.h`), return true if handled.
   * Metal code goes in a NEW category file `MGL/src/MGLRenderer+<Domain>.m`
     (`#import "MGLRenderer_Private.h"`, `@implementation MGLRenderer (<Domain>)` — all ivars visible).
     Optional auto-called hooks (see header): `ext<Domain>RenderPassDescriptor:`, `ext<Domain>RenderEncoderCreated:`,
     `ext<Domain>PreDraw:`, `ext<Domain>WillCommit:`.
3. **Shared files** (`MGLRenderer.m`, `MGLRenderer_Private.h`, `glm_context.h`, `glm_params.h`, `mgl.h`,
   `glm_context.c`, `get.c`, `draw_buffers.c`, `program.c`, other domains' files): only small, surgical
   edits when truly required, each one called out in your final report (file + reason). Prefer adding a
   hook call over moving code. Never reformat or reorder shared files. Another agent's branch will
   be merged alongside yours.
4. Don't delete or rename existing functions other domains might call.

## Definition of done

* Every function in your list has a real implementation; `stub_scanner.py` shows no EMPTY/NOOP/ASSERT
  for your files (or each remaining one is justified in the report).
* `make lib` builds with no new warnings in your files; `tests/headless/run.sh test_smoke test_<domain>` passes.
* Final report (your last message): functions implemented, Metal mechanism used for each group,
  shared-file edits, known gaps / emulation limits, test results (paste the PASS/FAIL lines).
