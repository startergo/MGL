# Domain: gpuaudit — find CPU work on the draw/dispatch path that belongs on the GPU, and move it

Worktree `wt/gpuaudit`, branch `gl46/gpuaudit`. First run `git merge gl46-port`, build, run the full
`tests/headless/run.sh` and record the baseline (suite → checks/failures).

## Phase 1 — audit (write `GPU_AUDIT.md` at the worktree root, commit it)
Scan `MGL/src/*.c`, `*.m`, `*.cpp` for CPU work that runs per draw, per dispatch, per upload or per
query, and could be done on the GPU or avoided. Look in particular for:
- **CPU↔GPU stalls:** `waitUntilCompleted`, `waitUntilScheduled`, `synchronizeResource`, busy waits on
  serials, `getBytes`/`contents` reads of GPU-written data right after a commit (e.g. query results,
  xfb counts, indirect args, conditional render, glGetBufferSubData, glReadPixels into a PBO).
- **CPU data loops:** vertex-attribute format conversion (non-Metal-fetchable formats), index
  widening/restart rewriting, line-loop/triangle-fan/quad emulation, base-vertex baking, texture format
  conversion/swizzle/depth-stencil packing at upload, mipmap generation, clears or blits done with
  memcpy/loops, buffer copies/fills (`glClearBufferData`, `glCopyBufferSubData`), indirect draws whose
  counts are read on the CPU, multi-draw loops issuing one Metal draw per sub-draw.
- **Per-draw CPU rebuilds** that could be cached (pipeline states, argument buffers, vertex descriptors,
  shader recompiles keyed on state).
For each finding: file:function:line, what runs on the CPU, how often (per draw / per frame / once),
the GPU alternative (compute kernel, blit encoder, Metal indirect/ICB, MTLFence/event instead of wait,
native Metal format), the estimated effort and risk, and **who owns it**. Rank by impact.

## Phase 2 — fix (only what you own)
`tess` and `geom` GLM agents are working right now. **Do not edit** `tessellation.c`,
`MGLRenderer+Tessellation.m`, `mgl_ext_tess.h`, `geometry.c`, `MGLRenderer+Geometry.m`,
`mgl_ext_geom.h`, `test_tess.c` or `test_geom.c` — list those findings under "For tess/geom" instead.
For the rest, fix the top findings in rank order, one commit each: GPU path first, CPU path kept only as
a fallback where Metal truly lacks the feature (say why). Edits to shared files (`MGLRenderer.m`,
`MGLRenderer+Draw.m`, `draw_buffers.c`, `program.c`) must be small and surgical, since tess/geom
touch them too. Every fix needs a pixel- or value-verified test in `tests/headless/test_gpuaudit.c`
(add it to `run.sh`/the Makefile like the other suites), and the full suite must stay at 0 failures.
Do not make changes that only look faster: measure (e.g. time 1000 draws, before/after) and report it.

## Report
Baseline vs final suite table, findings fixed (with timings), findings deferred and why, findings
handed to tess/geom.

Errors: `ERROR_RETURN(GL_X); return;` (the macro does not return by itself).
