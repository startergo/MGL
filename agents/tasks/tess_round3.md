# tess — round 3: root causes, one step at a time

Branch `gl46/tess` (worktree `wt/tess`). First `git merge gl46-port` (now ea5a3b8, includes geom).

## Rules for this round (read twice)
- Do the steps **in order**. One step = one commit. Do not start a step until the previous one is
  committed with the full suite at 0 failures.
- **Each step starts with a test that fails** (commit-ready, pixel- or value-checked), then the fix.
- **No workarounds.** These are explicitly forbidden as a "fix": one drawPatches per patch, bypassing
  a cache, gating a test behind an env var, waitUntilCompleted, CPU readback, "draw-only" tests.
- If a step is not solved after ~60 turns, STOP: write what you measured (the exact smallest
  difference you found) into agent_report.md and finish. A precise unsolved finding is a good result;
  a workaround is not.

## Step 1 — pipeline cache key (MGLRenderer+Gpuaudit.m `gaPSOKey`), then re-enable it for tess
The key is missing fields and uses raw object pointers:
- add `tessellationFactorScaleEnabled` and `tessellationControlPointIndexType` (the latter MUST match
  the index type of drawIndexedPatches, so a missing field can return an incompatible PSO);
- `vf`/`ff` are `MTLFunction` pointers: when a function is freed and a new one is allocated at the same
  address (tess recompiles plain/variant TES functions), the cache returns a stale PSO. Make the cache
  entry retain the functions it was built from, and compare the retained objects on lookup (or key on
  a stable identity), so a reused address can never match.
Test: compile/link/draw/delete a tess program 50× with different TES bodies (different colours) and
check pixels every iteration with the cache ON. Then remove the tess bypass.

## Step 2 — one drawPatches for all patches (find the real cause)
Isolated probes render every patch; the MGL pipeline renders only patch 0. Find the single difference:
1. Add `MGL_TESS_TRACE=1`: for a 4-patch draw print every encoder call (pipeline descriptor tess fields
   and vertex descriptor, every setVertexBuffer index/offset/length, setTessellationFactorBuffer
   offset/instanceStride, drawPatches arguments) and dump each bound buffer's bytes to
   `.agent/scratch/trace/`.
2. Write `.agent/scratch/replay.m` that loads those dumps and re-issues exactly those calls. Confirm it
   reproduces "only patch 0".
3. Change ONE thing at a time toward `.agent/probe_drawpatches.m` until all 4 patches render. Usual
   suspects: factor-buffer offset not 0 / instanceStride, PerPatch layout stride vs patch record size,
   PerPatchControlPoint stride vs the cp record, the `[[patch(…, N)]]` count vs patch_verts, buffers
   bound at a per-patch offset, `patchStart`.
4. Fix MGL, delete the per-patch loop, ungate the 4-patch test, add a gl_PrimitiveID test (colour by
   patch id, 4 patches).

## Step 3 — no CPU wait with a TCS (~MGLRenderer+Tessellation.m "TCS patch records are read back")
Build the TES-layout patch blob with a compute kernel on the same command buffer (or make the TES
vertex descriptor read the TCS patch-output layout directly). Delete that commit+wait.

## Step 4 — indexed patches without CPU index reads
Replace `mgl_buffers_prepare_cpu_read(ctx, ebuf)` with a gather kernel that binds the element buffer
as a device buffer (ubyte/ushort/uint, base vertex, primitive restart if enabled).

## Step 5 — isolines / point_mode render the fallback image
Same method as step 2 (trace + replay + one-change bisection). Turn test_tessemu into pixel checks.

## Report
Per step: the failing test, the root cause in one sentence, the fix, and the suite result.
