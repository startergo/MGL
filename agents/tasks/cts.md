# Domain: cts - fix Khronos GL CTS (KHR-GL46) failure clusters

First pass (agents/conformance.md, harness in agents/conformance/): 9035 cases, Pass 4558 (50.5%), Fail 3228,
NotSupported 1167 (mostly fp64, expected), Crash 60. Harness: `bash agents/conformance/setup.sh` (CTS in
~/MGL-cts, already built), `python3 agents/conformance/run.py ...` (see conformance.md section 2 for single-case runs).
The geom worker owns the transform-feedback capture-variant clusters (#2, #5), geometry_shader, draw_indirect+xfb (#12).
YOU own these clusters, in this order (biggest win first). For each: reproduce one representative case singly, find
the root cause in MGL (not the CTS), add an MGL headless regression test, fix, re-run the WHOLE cluster's group in the
CTS and record before/after counts in agents/conformance.md (section "Fix log"), commit.
1. #1 (736) GLSL identifiers that are MSL keywords/types (`sampler`, `texture`, `depth2d`, ...) collide in SPIRV-Cross
   MSL output -> "Shader creation failed". Fix generally (rename/escape every MSL reserved word and type name SPIRV-Cross
   doesn't already mangle; check spvc options first), not per-name.
2. #3 (581) glGetActiveUniform(siv) reports bool uniforms as GL_UNSIGNED_INT (SPIR-V/storage bool->uint mapping leaks
   into introspection). Introspection must report the GLSL type (GL_BOOL / GL_BOOL_VEC*), storage stays uint.
3. #4 (398) glMapBufferRange returns GL_INVALID_OPERATION on a legal configuration (READ after the CTS wrote the buffer).
   Check every map flag combination against GL 4.6 6.3 and fix.
4. #8 (144) gl_NumSamples / sample-variable built-ins "undeclared" in #version 330/400 shaders with
   GL_ARB_sample_variables (and the core 4.0+ spellings).
5. #7 (148+83+18) packed_pixels: missing/incorrect GL_INVALID_* for invalid format/type combos, GL_UNPACK_ROW_LENGTH
   handling, valid formats failing.
6. #13 (31) layout_location program creation failures; #14 (30) textureGather base level; #11 (48) column_major mat2 in
   UBO instance arrays; #9 (115) DSA renderbuffer storage; #6 (169) compute; #15 shader compile failures; then the
   Crash list (#10: compute_shader.copy-image, shader_storage_buffer_object.*, gl_spirv builtin functions).
Suite (tests/headless/run.sh) green before every commit. agent_report.md stays untracked; keep a running log in
agents/conformance.md. CTS logs are huge: always grep/summarize; keep shell output short.
