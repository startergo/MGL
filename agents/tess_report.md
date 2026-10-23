# tess — round 12 handoff: KHR-GL46.tessellation_shader (in progress)

Branch `gl46/tess`, HEAD **19d6032** (= merge of gl46-port 4087ad8/770eef5 + 3 commits).
**Headless suite 442 PASS / 0 FAIL; test_geom 593 checks / 0 failures; test_tess 499/0.**
CTS `KHR-GL46.tessellation_shader.*`: **44 / 58 PASS** (was 35/58 at round-11 merge).

## Round-12 work (all verified on hardware, CTS-driven)

1. **TF base-bind live size** (`geometry.c` mgl_geom_tess_xfb_route): the context's
   `buffer_base` table snapshots `size` at BIND time, but glBufferData re-defines the store.
   The CTS (gl_TessCoord, max_patch_vertices, XFB) binds once then bufferData's per draw, so
   the pack's fill test used a stale size and dropped records beyond it (zeros tail).
   Fix: build a local copy of the TF binding table where a *Base bind (`o->is_base[]`) uses
   the buffer's CURRENT size (clamped by its Metal backing). Fixed
   `tessellation_shader_tessellation.TES`, `.TCS_TES`, `.input_patch_discard`,
   `tcte.gl_MaxPatchVertices_Position_PointSize` (the round-11 tail).

2. **gl_PerVertex subset-redeclaration compat** (`tessellation.c`
   `mgl_tess_glsl_compat`, hooked in `shaders.c` compile path): GLSL 4.60 §7.1.2 lets any
   stage redeclare gl_in/gl_out as a SUBSET; glslang's cross-stage link rejects it unless
   both sides match member-for-member. The rewrite expands every `gl_PerVertex {...}`
   redeclaration (vs/tcs/te/gs/fs sources) to the full member set
   (gl_CullDistance only for #version >= 450). strtok_r (not strtok) inside!

3. **Block-name TF varying resolution**: glTransformFeedbackVaryings names interface-block
   members as `<BLOCK>.<member>`; MGL resolved instance names. Fixed in three places:
   `mgl_ext_geom.h` MGLGeomVar gained `block[64]` (filled in geom_reflect_stage from the
   SPIR-V struct type name); `geom_xfb_resolve` matches block or instance prefix; the
   tess-route remap (`geometry.c` ~6750) matches the capture member by full name, dotted
   suffix, or `_`-flattened suffix, and consumes split matrix/array members
   (`<name>_<i>`, summed words, lowest offset).

4. **TCS-only separable xfb link** (`mgl_geom_tcs_xfb_link` in geometry.c, dispatched from
   transform_feedback.c): a separable TCS-only program with TF varyings must LINK (it can
   never feed a draw); varyings resolve against the TCS outputs, no capture is built.

5. **Geometry-kernel interface-struct dedupe** (`geometry.c`
   `geom_dedupe_iface_structs`): a GS may declare `in B {...} user_in[]; out B {...}
   user_out;` — the per-statement emitter wrote `struct B` twice → glslang "redefinition".
   Identical duplicates are dropped; different-member duplicates get `_2` renaming.

6. **Canonical record matrix + struct members** (`tessellation.c`): msl_parse_struct now
   splits `floatCxR name;` into columns `name_0..name_(C-1)` (msl_matrix_type had an
   off-by-one: `strlen != 9`, c/r index swap — it never matched!) and inlines
   struct-typed members (`_61 inst;` → `inst_field`, alignment = max field alignment),
   matching SPIRV-Cross's attribute-flattened structs on the tcs/tes sides. Also
   `tess_canon_member_lines` matches flattened stage members by (key, array_idx).
   Fixed `tcte.gl_in` (mat2 + struct outputs through TCS).

7. Small shared-file edits: `compileShader:` NULL guard (MGLRenderer.m — a pipeline
   composite stage with no MSL crashed NSString), env-gated diagnosis prints
   (`program link failed` in program.c, `geom link failed` + `geom: build failed` in
   geometry.c, query/cpu_count + note_draw traces in queries.c/tessellation.c).

Scratch repros (NOT committed, on disk): tests/headless/zz53..zz60.

## CTS scoreboard after round 12 (58 cases)

PASS 44: compilation_and_linking_errors 17/18, single 5/7 (isolines, primitive_coverage,
xfb_captures_data_from_correct_stage NEW, default_values, ext_program_interface_query),
tcte 4/5 (gl_in NEW, gl_MaxPatchVertices_Position_PointSize NEW, gl_PatchVerticesIn,
gl_tessLevel), invariance_rule1/5, quads degenerate_case, triangles 3/3, winding 4/4,
tessellation_shader_tessellation 3/5 (TES/TCS_TES/input_patch_discard NEW).

FAIL 14 (all diagnosed, see below).

## Remaining tails — diagnosis + where to dig

* `tcte.data_pass_through` — tc_pointSize/patch-attr passthrough values wrong (old note).
* `single.max_patch_vertices` case 2: TES-only (no TCS), GL_PATCH_VERTICES=32, TES sums
  `inVertex[i]` over gl_PatchVerticesIn → floats come out ≈45×(1,2,3,4) and ints garbage:
  the plain-TES capture kernel reads control-point records beyond the 32 written
  (implicit-array `gl_in[]` size / ring-slot stale data). Repro: zz53 shape, case 2.
* `invariance_rule2..4,6,7` — outer-edge tessellation coordinates must be invariant to
  the INNER levels. First failure: quads inner(2,3)/outer(2,3,4,5), coordinate (0,1/3,0)
  missing when the reference config changes only inner. The GPU tessellator derives
  boundary vertices from the inner grid instead of independently subdividing each outer
  edge (D3D rule). Look at the emulated tessellator (MGLRenderer+Tessellation.m kernels;
  compare with .agent/ref/poly/tessellator.h ProcessQuad).
* `point_mode.point_rendering`, `points_verification` — vertex COUNT for fractional
  spacing: quads fractional_odd inner(32,64)/outer(1,32,1,32): MGL 2176 vs expected 2052
  (fractional grid vertex-count formula; same tessellator file).
* `quads.inner_tessellation_level_rounding` — "could not find a marker triangle" with
  non-integer inner levels (rounding rule for inner levels, same file).
* `tc_barriers.*` (3) — barrier-guarded cross-invocation reads in the TCS return 0:
  the TCS-as-compute kernel needs real threadgroup shared memory + barrier support
  (one invocation per thread; gl_barrier currently a no-op). MGLRenderer+Tessellation.m
  TCS kernel + `mgl_tess_tcs_msl`.
* `tessellation.gl_InvocationID_PatchVerticesIn_PrimitiveID` — gl_PrimitiveID in the TC
  stage is 3 (invocation?) instead of the patch index 0; the TCS kernel's primitive-id
  parameter is wired to the wrong value.
* `vertex.vertex_ordering` (now runs, fails) — triangles outer (1,33,1,-1) ccw: vertex
  orientation incompatible; `vertex.vertex_spacing` still needs a long run (>150 s).
* `max_in_out_attributes` — NotSupported-style skip: GL_MAX_VERTEX_OUTPUT_COMPONENTS=64
  < GL_MAX_TESS_CONTROL_INPUT_COMPONENTS=128; raising the VS output limit to 128 (Metal
  vertex-to-kernel output is buffer writes, effectively unbounded) would let it run —
  glm_params.c, coordinate with the geom agent.
* `compilation_and_linking_errors.tc_invalid_output_patch_vertex_count` — still CRASH
  (returns FAIL): the invalid-vertex-count link error case; check link driver validation.

## Rules that bit (keep in mind)

* `ERROR_RETURN(GL_X)` does not return; `return` explicitly.
* msl/tessellation.c struct parsing: MSL float3/int3 align 16 (padded) in buffer structs.
* The tfsize the pack uses must be the LIVE buffer size for *Base binds (see item 1) —
  the geom (non-tess) pack at MGLRenderer+Geometry.m:1321/1475 still uses the stale
  `tfb[b].size`; if a geom-domain CTS test re-sizes a bound TF buffer, apply the same fix
  there.
* One CTS batch at a time, `make -j4` only, never `git add -A` (zz*.c stay untracked).
