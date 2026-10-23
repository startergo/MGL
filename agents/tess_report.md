# tess — round 11 handoff: KHR-GL46.tessellation_shader (in progress)

Branch `gl46/tess`, HEAD = merge of gl46-port (geom r39, 8949387) after 0141b1c.
**Headless suite 438 PASS / 0 FAIL; test_geom 593 checks / 0 failures; test_tess 499/0.**
CTS `tessellation_control_to_tessellation_evaluation.gl_MaxPatchVertices_Position_PointSize`
advanced: position + te_value1 + te_value2 now EXACT (was "Invalid gl_Position data");
now fails only the LAST check (pointsize uniqueness, esextcTessellationShaderTCTE.cpp:3072).

## Done this round (all committed)

1. **Pipeline pair records** (round-10 item 1, `MGLTessProgram.pair` in mgl_ext_tess.h +
   `tess_build_pair/tess_free_pair` in tessellation.c): when a pipeline pairs the TES with a
   tc-only program, `mgl_tess_active_program()` returns a DERIVED record: a copy of the TES's
   tp with has_tcs=1, the tc program's out-vertex count, and the interface layouts
   canonicalized across BOTH modules (union of tcs_out/tes_cp, tcs_po/tes_pi; TCS kernel
   recompiled with canonical structs + tri-domain factor fixups + GL float-level copy; VS
   kernel gets a pair variant when its raw record differs; blob map rebuilt). Pair-aware
   `tessVsPSO(..., draw_tp)`; `mgl_tess_set_capture` keeps base+pair in sync; pairs die with
   either side (delete/relink). Pipeline GL_PATCHES draws now run the TCS (they previously
   silently used default levels).
2. **TCS out-record parse fix** (`msl_tcs_record_names`): a TCS that reads NONE of its cp
   inputs has no spvIn param — the out record is still found (out-only signature accepted).
   This is what made gl_MaxPatchVertices positions AND the interface-block varyings
   (value1/value2) cross TCS→TES: without it the TCS kernel kept its raw SPIRV-Cross struct
   while the TES read canonical offsets (gl_Position aligned by luck; block members garbled).
3. **`tess_annotate_keys` takes an explicit names-Program** (was tp->program): pair members
   resolve through the module that DECLARES them (block member names live in the TCS module)
   — keys now alias exactly like a single-program link. te-only programs no longer
   name-hash flattened block members.
4. **Separable vs-kernel cache hardening** (ext_tess->vs_kernel, 32 slots): compaction on
   program delete (dead slots leaked capacity — after ~16 links the cache filled SILENTLY and
   pipeline draws bailed "VS stage function is not a kernel" → the in-batch
   GL_INVALID_OPERATION leak from round 10!), eviction when >32 live, and the kernel MSL is
   kept so pair rewrites recompile from it (`mgl_tess_vs_kernel_msl`).
5. Repros (scratch, not for commit): tests/headless/zz50_pair.c (mono vs pipeline XFB
   byte-identical, CTS gl_MaxPatchVertices shape), zz51_pt.c (uniform-level point-mode sweep),
   zz52_cts.c (verbatim CTS shaders + mono/pipe COUNT comparison + GetTransformFeedbackVarying).
   All pass. Diagnostics kept env-gated: MGL_TESS_DUMP_CAP (tessxfb route + capdump capture +
   tfmap map), MGL_DUMP_MSL (pair_tcs.metal, cap_tes.metal dumps).

## Where the next session should pick up

**The remaining gl_MaxPatchVertices failure is fully diagnosed:** CTS point-mode runs read
`n_data_vertices` records from the xfb buffer, but the buffer only contains ~2 written
records + zero/stale tail (dumps: "tfmap: 47 ps outliers of 49 records"; records 0-1 correct,
2..48 zero). Meanwhile the tess capture published HEALTHY counts and factors
(capdump: count/stride/pointsize correct, factors half/float exact, pair=1 dom=2). So the
count the ENCODE kernel sees is wrong, not the tessellation:

  a) `geomXfbTessEncode` (MGLRenderer+Geometry.m:1288) dispatches its OWN command buffer that
     reads the tess ring (records + counts) — check whether it waits on the tess capture's
     serial (the route passes `c->serial`... see MGLTessCapture). A missing wait = the encode
     races the tess cb and reads a stale/zero count.
  b) ring-slot reuse: verify the capture slot is held until the ENCODE consumed it (contract
     "valid until the next capture draw"): mgl_tess_publish / tessRingRelease / retire.
  c) The failing runs are PIPELINE draws (pair=1) — mono runs of the same config pass, so
     compare the encode's inputs (in[0] capacity, counts[0]) between mono and pair draws
     with MGL_TESS_DUMP_CAP=1 (tessxfb lines give tfsize=N*44 = n_data; capdump gives the
     published count — mismatch = the encode's count read).

Then: single.isolines count, single.primitive_coverage pixels,
single.xfb_captures_data_from_correct_stage ("Separable program should have succeeded",
esextcTessellationShaderXFB.cpp:228 — a separable TCS/TES link refusal to chase),
data_pass_through (tc_pointSize; gl_in "Invalid rendered value") — unchanged from round 10.

## Incident (supervisor note)

I mistakenly ran `git stash pop` + `git stash drop` while recovering from a checkout
experiment and dropped the GEOM agent's stash@{0} ("WIP on gl46/geom: 9a42d48 geom: plain
(default-block) uniforms in the geometry emulation + fresh vertex-attrib plan for the capture
pass"). Recovered from the dangling commit (83f4cbc, newest WIP-on-9a42d48) via
`git stash store`; the stash list is back to its original two entries. geom should sanity-
check their stash still applies. Rule violation (never stash) acknowledged.
