# tess — round 15 handoff: CTS unblocking (ring wedge, point size, inner[1], isoline point_mode, XFB block members)

Branch `gl46/tess`, HEAD cd01f98 (= merge of gl46-port eb540d2 (geom r44) + round-15 commits).
`external` is a symlink to the main checkout's prebuilt deps — `cp -P external .bck && unlink external
&& mkdir external && git merge gl46-port && rm -rf external && mv .bck external` then clean rebuild
(`rm -rf build/MGL && make -j4 lib`).

## What round 15 fixed (all verified against KHR-GL46 on the M4 Pro)

1. **Scratch-ring wedge** (`MGLRenderer+Tessellation.m`): a draw that bailed after
   `tessRingAcquire` — the hot one being a CULLED patch (all relevant outer levels <= 0 ->
   `!S->tess_index_count -> return true`) — retired its ring slot on NO path. Eight such
   draws wedged the 8-slot ring; every later draw then burned the 2 s acquire timeout.
   KHR-GL46 …single.isolines_tessellation looked like a 480 s hang; it was ~24 wedged draws.
   Fix: `tessRingRetireOn(cb, slot)` — release immediately when the block's command buffer
   already completed (the CPU reference path waits), else on `addCompletedHandler`; wired
   into the degenerate exit and every `return false` after a successful acquire.
   **Result: single.isolines_tessellation Pass (10.4 s; was Timeout > 480 s).**
   `MGL_TESS_RING_TRACE=1` logs acquire/release/retire/SPIN diagnostics.

2. **Point size from the last pre-raster stage** (`MGLRenderer+RenderState.m`): the plain-TES
   vertex function was bound into the render pipeline UNSPECIALIZED, so the rstate
   PROGRAM_POINT_SIZE / point-size function constants took their defaults -> size-1 points
   (point_mode.point_rendering failed at every corner). Fix: specialize it through
   `rstateSpecialize:` with a standin Shader (like the hardware TES variant). Also
   `rstateSpecialize` now specializes STANDIN functions (geom vertex override, plain/hw TES)
   even when the program's own stage MSL is unpatched (a GS consuming tess points with a
   trivial VS hit that). **Result: tessellation_shader_point_mode.{point_rendering,
   points_verification} both Pass.**

3. **Triangles ignore gl_TessLevelInner[1]** (`tessellation.c`, `tcs_redirect_tri_inner1`):
   SPIRV-Cross lowers the gl_TessLevelInner array onto the triangle factor record's SCALAR
   inside factor, so a later `[1]` store clobbered the `[0]` value the tessellator needs
   (the CTS always writes both, e.g. (33,-1) -> 36 vertices instead of 803). Every `[1]`
   access is now rewritten to `mgl_levels[gl_PrimitiveID].in[1]` (the float-levels record's
   unused second slot). Verified against the round-14 Mesa oracle (.agent/d3d11/oracle.c):
   triangles (1,33,1)/inner 33 -> 803 verts/1569 prims, (4,4,4)/inner 4 -> 19, quads
   (4,4,4,4)/inner (8,8) -> 65 — GPU query counts now bit-identical. **points_verification
   count half now matches.**

4. **glBeginTransformFeedback(GL_PATCHES)** (`transform_feedback.c`): was rejected with
   GL_INVALID_ENUM, so the whole tessellated XFB capture path never ran
   (tcte.data_pass_through got 0 records). Now accepts POINTS/LINES/TRIANGLES/
   LINES_ADJACENCY/TRIANGLES_ADJACENCY/PATCHES (GL 4.6 13.4.1) and `xfb_mode_compatible`
   knows the new modes.

5. **Interface-block XFB outputs** (`geometry.c`, geom-agent file, surgical): (a) bare member
   names in glTransformFeedbackVaryings ("tc_position" of the TES's OUT_TE block — the CTS
   form) are matched in a second pass of `geom_xfb_resolve`; (b) the tess route's
   capture-member reconcile now also matches SPIRV-Cross's flattened "<instance>_<member>"
   capture field ("te_data_tc_position"). Without (b) the varying was dropped (words=0) and
   the XFB buffer read zeros.

6. **Isolines + point_mode emit** (`MGLTessKernels.h`): `isolineProcess` always emitted the
   line-segment index list; in point mode that overwrote definePoint's identity list and
   overflowed the identity-sized alloc — the geometry import / XFB then read a segment list
   and patch 1+ got patch 0's records. The segment loop is now `if (!ctx.points)`.
   **zz_tcte_pt (both TES-only and GS variants, 3 patches) is fully correct.**

## CTS state (KHR-GL46.tessellation_shader)

| case | r14 | r15 |
|---|---|---|
| single.isolines_tessellation | Timeout(480s) | **Pass (10.4 s)** |
| tessellation_shader_point_mode.point_rendering | Fail | **Pass** |
| tessellation_shader_point_mode.points_verification | Fail | **Pass** |
| tess…control_to_tess…evaluation.data_pass_through | Fail@idx40 | **Fail@idx110** (progress: the
isolines+GS runs pass; see open item 1) |

Repro: `python3 agents/conformance/run.py --pattern '<name>' --results-dir .agent/cts-X
--refresh-caselist --max-seconds 1500 --per-case-timeout 1450 --batch 1`.

## Open items (next round)

1. **tcte.data_pass_through @ index 110** (`expected [2,2,2,2] found [0,0,0,0]`): the first
   failing subtest moved 40 -> 110, i.e. the isolines+point_mode runs now pass; a later run
   (per_patch=55: likely QUADS point_mode or a TRIANGLES ring layout with more patches) still
   reads patch-0 control points for patch 2+. Start from `tests/headless/zz_tcte_pt.c`
   (RUN_GS=1 GS_RUN=1 runs both variants; extend with a QUADS GS variant and >3 patches) and
   the capture diagnostics `MGL_TESS_DUMP_CAP=1` (pack-side) / `MGL_DUMP_MSL=1`.
2. **test_geom pool_wrap_stress_tiny times out** — PRE-EXISTING at gl46-port eb540d2 (geom
   round 44), verified by rebuilding the merge point; not a tess regression. Hand to geom.
3. `.agent/cts-iso*/cts-pm*/cts-tcte*` hold the round's CTS logs; zz_* tests are untracked
   by design (keep zz10_var.c, *.spv, TestResults.qpa untracked, never commit).

## Shared-file edits this round (each surgical, for the merge)

- `MGL/src/MGLRenderer+RenderState.m`: specialize the plain-TES vertex function (2 hunks:
  the standin branch + `standinFn` early-out in `rstateSpecialize`).
- `MGL/src/transform_feedback.c`: accept PATCHES/adjacency TF modes + `xfb_mode_compatible`
  cases.
- `MGL/src/geometry.c`: XFB var resolve pass-2 (bare block members) + the route's
  flattened-name reconcile fallback.
- `MGL/src/MGLRenderer+Geometry.m`: none left (debug prints removed).
- `MGL/src/MGLTessKernels.h`: isoline point_mode emit guard.

Domain files: `MGLRenderer+Tessellation.m` (ring retire + traces), `tessellation.c`
(tri inner[1] redirect), `mgl_ext_tess.h` (unchanged this round).
