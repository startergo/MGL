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
