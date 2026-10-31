READY 3a98010 (r85 = the r84 READY tree a62b8c4 + 3bd6003 + 3a98010; the harness/report commits after this line change no code). CLOSES THE PIQ ROUND: every dEQP-GL45-ES31.program_interface_query crash is gone (71 -> 0), 0 regressions across the whole 4599-case group, KHR-GL46.program_interface_query stays 43/43 Pass, headless 142 files / 11301 checks / 0 failures. Perf: the changed path is link-time-only (introspect_post_link -> mgli_process_link; every consumer of ip->inputs/outputs/vtx_outputs is a glGet*/glGetProgramResource* query — verified by grep, no per-draw call site), so the pre-READY benches target exactly it: per-draw CPU pair-ratio 0.999, link+introspect equal-or-faster, locked glmark2 gl41-geometry +0.9% (ideas/jellyfish unusable this pass — machine was not quiet, details in 5c).

1. THE TWO CODE COMMITS (MGL/src/introspect_spirv.c, tests/headless/test_piq.c — 141 checks pass):
   3bd6003 reflect_io enumerates program in/out per GL 4.6 §7.3.1, mirroring the CTS's own
      reference walk (es31fProgramInterfaceDefinitionUtil): struct members dotted down to every
      leaf (Block.member.field); a bottom-level array of basic types is ONE resource name[0]
      with ARRAY_SIZE = element count; arrays of aggregates enumerate per element; anonymous
      blocks use plain member names. Also fixed while there: GL_LOCATION is -1 for inputs of
      stages other than VS and outputs of stages other than FS unless layout(location=N)
      (11.1.1); IS_PER_PATCH reads the member Patch decoration (glslang decorates members, not
      the block); arrayed stage io (TCS/TES/GS per-vertex) keeps name[0] with ARRAY_SIZE =
      declared length.
   3a98010 per-member layout(location) on an interface block propagates: glslang decorates the
      members that carry it (and ARB_enhanced_layouts requires ALL members carry one), so those
      members report the real location instead of -1 in non-VS-input / non-FS-output interfaces.

2. dEQP-GL45-ES31.functional.program_interface_query.* on 3a98010, NEW glcts
   (opengl-cts-4.6.8.1-0-g067e8832, ~/MGL-cts-new/build/external/openglcts/modules/glcts,
   --deqp-archive-dir .../modules/data). Harness: ab/cts_piq85.sh — one subgroup per glcts
   invocation so a crash can no longer eat the group; per-case fail lists in /tmp/piq85/*.fails.

     TOTAL (4599 cases):  Pass 4287   Fail 309   NotSupported 3   Crash 0
     baseline 965d88f (same glcts, official logs ~/MGL-cts-new/official/r965d88f/
     config-gl45-es31-main-*):  Pass 4003   Fail 522   NotSupported 3   Crash 71
     case-by-case over the 4528 recorded baseline results:
        REGRESSED (base Pass -> now non-Pass): 0
        FIXED    (base non-Pass/absent -> now Pass): 327

     subgroup                    n     r85 P/F/NS      965d88f P/F/NS  crash
     program_input             500    484/16/0         307/151/0        42
     program_output            552    509/43/0         402/121/0        29
     buffer_variable          1313   1242/69/2        1242/69/2          0
     uniform                  1600   1512/88/0        1512/88/0          0
     transform_feedback_varying 420    341/79/0         341/79/0          0
     shader_storage_block       92     83/8/1           83/8/1          0
     uniform_block              92     86/6/0           86/6/0          0
     atomic_counter_buffer      28     28/0/0           28/0/0          0
     buffer_limited_query        2      2/0/0            2/0/0          0

   All 71 crashes were program_input(42)/program_output(29) interface-block-with-struct cases:
   we reflected a struct member of a user block as one resource of GL_TYPE GL_NONE and emitted
   no leaf resources; dEQP's TypeValidator then indexed past its tables (SIGSEGV). The exact
   crash filter re-run on 3a98010 (program_input.type.interface_blocks.*: 126 cases) =
   120 Pass / 6 Fail / 0 Crash, and the 6 fails are all array.mat3x2 (see below).
   The 309 remaining fails, triaged:
   - program_input 16 + program_output 43: every one is an arrays-of-matrices io case — MGL's
     introspection patcher needs an MSL re-emission SPIRV-Cross refuses ("MSL cannot emit
     arrays-of-matrices in input and output variables") or rejects ("nullptr"); reproduced with
     pristine glslang output too. Genuine SPIRV-Cross limitation, documented, not introspection.
   - buffer_variable 69+2NS, uniform 88, transform_feedback_varying 79, shader_storage_block
     8+1NS, uniform_block 6: P/F/NS byte-identical to 965d88f — pre-existing gaps this round
     neither touched nor regressed (left for the introspection backlog).

3. KHR-GL46.program_interface_query.* on 3a98010 (same new glcts): 43/43 Pass. Unchanged.

4. Headless suite tests/headless/run.sh: 142 files, 11301 checks, 0 failures, 0 build fails
   (r84: 137 files / 11236 checks; the growth is the merged tess/compat tests plus the two new
   test_piq cases from 3a98010).

5. PERFORMANCE (pre-READY, as usual; old = a62b8c4 rebuilt from `git archive a62b8c4` into
   /tmp/base85 and installed as ~/MGL-cts/baseline-a62b8c4/{libmgl,libmgl_nsgl}.dylib):
   a. ab/abbench_geom83.sh, CPU clock (MGL_AB_CPU=1), 7 reps, modes 0:2000 1:1000 31:1000;
      build/ab_old|ab_new with @rpath/@loader_path (verified per side via DYLD_PRINT_LIBRARIES):
        drawArrays    x2000   old min 1.012  new min 1.005  (-0.7%)   pair-ratio median 0.999
        blend-flip    x1000   old min 5.289  new min 5.091  (-3.7%)   pair-ratio median 1.039
        blendFunc-flip x1000  old min 4.674  new min 8.034  (min is a single-sample outlier)
                            pair-ratio median 0.912 — this case is bimodal under load on both
                            sides; the median (the script's robust metric) says new is faster.
      => per-draw CPU unchanged, as expected for a link-time-only delta.
   b. ab/link_bench.c (new, committed): times glLinkProgram + the first interface query — the
      exact code the two commits touched. plain (glmark2-style) program 0.574 -> 0.582 ms/link
      (+8 us; run-to-run spread on this loaded box is 0.31-0.76, so this is inside noise);
      worst case for the new leaf walk (VS out + FS in interface block, nested structs + arrays)
      0.728 -> 0.608 ms/link — FASTER: the leaf walk replaced the old per-variable struct path.
   c. glmark2 locked 8+8 (ab/abg85.sh; P = baseline-a62b8c4, C = this branch's nsgl build,
      /usr/bin/lockf -k /tmp/mgl_glmark2.lock, P/C order alternates per round):
        ideas           P median 714.0   C median  594.0   -16.81%
        jellyfish       P median 5520.0  C median 5149.0    -6.72%
        gl41-geometry   P median 2072.0  C median 2090.0    +0.87%
      The ideas/jellyfish medians are UNUSABLE this pass and I do not claim them: the wt/verify
      agent ran 18 UNLOCKED glmark2 shadow probes plus make -j8 and headless GPU tests
      concurrently (load 3.3-5.5 throughout), with its own locked ab_g55/ab_regr/flash_check
      queue right behind mine. Per-round numbers: ideas C is bimodal [528,528,579,589] vs
      [690,710,719] while P is tight [704..748]; jellyfish swings 3772..5645 on BOTH sides.
      Steady-state FPS cannot be affected by a change that runs once per glLinkProgram (glmark2
      links each scene's program before its timed loop), and gl41-geometry — the one scene with
      a geometry shader, i.e. the only scene whose program even reaches the vtx_outputs table —
      is +0.87%. Verdict: equal within noise; to be re-confirmed on a quiet machine.

6. HYGIENE: make lib clean (introspect_spirv.o rebuilt from scratch, 0 new warnings);
   agents/stub_scanner.py --list EMPTY|NOOP|ASSERT = 0 for my files; tree has no uncommitted
   code (git status shows only the pre-existing untracked scratch dirs). No shared-file edits
   this round. Harness added: ab/cts_piq85.sh, ab/abg85.sh, ab/link_bench.c.

Subgroups: GL_KHR_shader_subgroup left UNEXPOSED per the 15:44 user decision — not started.

=====================================================================================
(r84 report, superseded by the READY line above)

READY a62b8c4 (r84 = 51e3e93 + 93cd57d + merge edf067e of gl46-port f8be669; harness a62b8c4). TWO BUGS FIXED, both in the geometry-shader draw path; plain-path and sidecar behavior unchanged (proofs below).

1. EVERY INDEXED GEOMETRY-SHADER DRAW STALLED (robustness + per-draw CPU). The prep-kernel
   branch stamped the element buffer gpu-WRITTEN (mgl_buffers_mark_gpu_use(eb, true), from
   the original prep-kernel commit b2280eb) although the kernel only READS the indices
   (buffer(2) is `const device uchar*`). The next draw's geom_index_list ->
   mgl_buffers_prepare_cpu_read then saw write_serial == the CURRENT serial and waited for
   the current, still-uncommitted command buffer; extBufWaitSerial -> extBufFlush committed
   it mid-frame and set DIRTY_ALL. Consequences per draw: a full GPU round-trip wait (no
   CPU/GPU overlap at all), DIRTY_SHADER set for compatPipelineFastPath (declined), and the
   complete pipeline-descriptor rebuild (~150 us) + full state walk instead of the cache
   hit. Now stamped read (false); a buffer the GPU really writes (xfb into an index buffer,
   compute) is still protected because the WRITER stamps its own write_serial.
   Wall time per indexed GS draw (scratch driver, 3072 indices, pass-through strip GS):
   985 us -> 425 us (-57%); CPU no longer commits+waits inside the draw — the pool ring
   (steps 1-3, ~0.1-0.2 us when a slot has room) now throttles run-ahead instead.
   Audited the other 8 mark_gpu_use sites in my file: all are genuine writers (xfb pack
   targets, atomic counters the kernels maintain, SSBOs the GS writes) — left as-is.

2. r83 ENV-LATCH REGRESSION: gEnv.debug = gEnv.debug != NULL read the still-zero struct
   instead of getenv("MGL_GEOM_DEBUG"), so every gEnv.debug site was dead. Now latched
   correctly (also present as the one conflict when merging gl46-port; my side kept).

3. PER-DRAW CPU (both in the capture/kernel bind paths): (a) the "does this variant declare
   spvBufferSizeConstants" probe ran strstr over the whole MSL text (tens of KB) twice per
   draw — memoized on the text pointer (every rebuild path memsets the gp via
   geom_free_prog and installs a fresh string, so a stale hit is impossible; probe != msl
   re-probes); (b) the capture pass ran mapGLBuffersToMTLBufferMap for the VERTEX stage a
   second time right before its bind loop although nothing between the descriptor build
   and the loop can change the answer — the list is now reused (the loop re-reads
   b->data.mtl_data per entry, so buffers materialised in between are still picked up).

Proofs. Locked glmark2 8+8 (ab/abg84.sh, new baseline f8be669, P/C order alternates per
round, shared lock): ideas +0.21% (MS-sidecar zero-cost when unused holds), jellyfish
+137% but that is load noise — the machine never went quiet (P distribution bimodal
[1553..4551], C median 4143 matches quiet-machine levels; jellyfish has no GS so no code
of mine runs in it), gl41-geometry +19.6% (the fix's target). ab_bench plain path under
load ~6, pair-ratio medians 0.914/0.951/0.988 (new equal-or-faster; min-of-N unusable at
that load). Per-group CTS (ab/cts_groups_geom83.sh, 8 groups): shader_image_size 74P,
shader_image_load_store 51P, shader_texture_image_samples 4P, texture_view 7P,
geometry_shader 136P/0F, sample_variables 152P/38NS, sample_shading 13P,
shader_multisample_interpolation 109P — no Fail anywhere. Full headless gate on the final
tree: 137 test files / 11236 checks / 0 failures. stub_scanner clean for my files.

Glmark2 note for the supervisor's whole-package sweep: geometry_scene FPS scales with the
machine's load (gl41-geometry medians ranged 179-250 across rounds here); the +19.6% is
from the 8-round median under the same load on both sides.

Shared-file edits: none outside my files (the gl46-port merge touched MGLRenderer+Geometry.m
only in the already-conflicting gEnv.debug line, resolved to my fix). draw_ext.c/geometry.c
temporary diagnostics removed before commit (verified by git diff).

Next-round candidates (not started, per priority change): geometry-stage bindmap cache for
geomBindStageBuffers following the compat prefix-cache pattern (pays off only for
SSBO/ACB-heavy geometry shaders — a resource-free GS spends ~1-2 us there); capture-pass
vertex-descriptor + PSO-key cache keyed on a VAO fingerprint (r83's noted candidate;
cap-desc is now ~10 us of a ~170 us CPU budget, the rest being unavoidable Metal encoder/
bind traffic: raster ~64 us is a new render encoder + full state application on it).
