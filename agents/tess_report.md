# tess — fp64 round 6 — HANDOFF (fresh session starts here)

Branch `gl46/tess`, HEAD **<see below — this commit>** (on top of the gl46-port
merge 2d1d20c + merge commit e79b744).  Task: CTS `KHR-GL46.gpu_shader_fp64`
group green + suite green.
**Suite: rc=0, 0 failures (425 PASS lines; tests/headless/run.sh).**
**fp64.fp64 group: 8/9 Pass in 331 s** (`--deqp-case='KHR-GL46.gpu_shader_fp64.fp64.*'`).
`test_fp64` 14.5 s / `test_fp64b` 6.1 s / `test_fp64c` 11.8 s cold incl. build.

## fp64 group status (round 6)

| case | status |
|---|---|
| errors / conversions / operators / valid_constructors / illegal_conversions | PASS |
| **state_query** | **PASS** (was FAIL) — glGetUniformuiv of a negative double must convert to 0, not the wrapped `(GLuint)llround(d)` (uniforms.c, one-line shared-file edit).  143 sub-cases were hitting this. |
| **max_uniform_components** | **PASS** (was FAIL) — two bugs: (a) the TES-capture kernel's fallback type `mgl_cp_rec` was never declared when the TES reads no control points ("unknown type name 'mgl_cp_rec'" from newLibraryWithSource) — now a default `struct mgl_cp_rec { float4 gl_Position; };` is emitted; (b) the CPU-reference tessellator (taken when the GPU tessellator's worst-case scratch exceeds the 64 MiB budget — 1024-patch draws need ~221 MiB worst-case) never ran the capture kernel, so xfb behind tessellation silently captured nothing.  `tessEnsureCapturePso`/`tessPublishCapture` (MGLRenderer+Tessellation.m) are shared by both paths now; the CPU path dispatches the capture kernel over its CPU-written coords/indices in the draw's ring block before processGLState.  Repro: `tests/headless/zz20_muc.c` (ZZ_STAGE/ZZ_N/ZZ_TY envs) and `zz21_isoxfb.c`. |
| named_uniform_blocks | PASS (round 5) |
| **varyings** | **FAIL** — 14/26 variable batches "A valid program object failed to link".  See the diagnosis below; scalar/dvec2-only batches LINK (my zz22 scratch proves it), every batch containing a dmat*/dvec3/dvec4 variable fails at the TES MSL compile. |

## varyings — exactly where it stops (verified this round)

The chain needed by the CTS (GPUShaderFP64Test7): array/matrix double VERTEX
ATTRIBUTES (current values, `glVertexAttribL4dv(loc+index)`, one location per
(element, column)) -> VS/TCS/TES/GS interface blocks with the SAME variables ->
xfb of every gs_variableN[M] (+ FS expectation).

Done and working (all in fp64.c unless noted):
* xfb names `name[N]` (array ELEMENT) resolve in the geometry capture path —
  geometry.c's lookup now strips the trailing "[N]" and captures ONE element
  (offset into the record, words of one element).
* the geometry kernel's generated GLSL used `uint64BitsToDouble(uvec2(lo,hi))`
  — invalid (takes uint64_t) and glslang "compilation terminated"; now
  `packDouble2x32(uvec2(lo,hi))` (GLSL 4.00 core).
* array/matrix double vertex attributes: `ptr Input (array|mat double)` gets a
  slot repr `array uint2/uint4 K` (attr_slotarr_ty; column-aligned, one Metal
  location per (element, column) half).  Constant (element, column, row) chains
  fold to (slot j, pair) — attrslotidx/attrsgeom/attrorig side tables; loads
  read the slot + extract the pair.  Geometry/attrsgeom verified: batches with
  doubles/dvec2/dvec3/dvec4/matrix inputs LINK the vertex stage.
* stage-IO dvec3/dvec4/matrix members: named-member slot structs
  (iface_slot_struct) instead of arrays-of-uint4 — OpCompositeConstruct(struct)
  avoids SPIRV-Cross's spvArrayCopyFrom*ThreadGroup helpers whose address
  spaces break when MGL moves tess/geom output to a device buffer (this is
  what "no matching function for call to spvArrayCopyFromStackToThreadGroup1"
  was).  Indexed element stores/loads of dvec3-4/matrix elements splice a
  member SLICE (elemslice/elemidx/elemstride/elemorig; literal indexes only —
  SPIRV-Cross can't dynamically index struct-valued stage IO either).

The REMAINING failure: the tessellation canonical-record MSL splice
(msl_rewrite_struct_body + the member renames in tessellation.c) does not
flatten NESTED structs inside interface blocks.  The fp64 slot struct for
`dmat2 tc_variable4` appears as `anon_te_variable4 te_variable4;` in the block;
the splice rewrites the block body and member accesses by NAME from its own
flat member list, producing e.g.
`anon_te_variable4 _139 = anon_te_variable4{ in_data_tc_variable0, ... }`
(member subobject type mismatch: uint4 vs uint2) or `anon_vs_variable4{ }`
(empty composite).  Plan for round 7 (fp64.c only): expand multi-slot block
members into N FLAT uint4 members (`tc_variable4_0`, `tc_variable4_1`) at the
iface_struct level (like the vertex-attribute slots), so no nested struct ever
reaches the splice; whole-member matrix loads/stores then emit per-slot member
chains (record expanded[chain] = {base, nslots} the same way elemslice does).

Scratch repros (untracked, kept): `tests/headless/zz22_varbatch.c`
(ZZ_N/ZZ_FIRST: CTS-faithful multi-variable generator, batch link only),
`zz20_muc.c` (max_uniform_components port, TCS/TES stage, draws + xfb),
`zz21_isoxfb.c` (minimal TES-output xfb probe).  NOTE zz4_cts_varyings.c has a
wrong TYPES table (dmat4/dmat4x3 swapped vs the CTS order) — use zz22.

## Rules of engagement reminders

* Max -j4, CTS foreground, chunks < 15 min, one batch at a time, scratch tests
  under timeout, no background processes, never `git stash`, never `git add -A`
  (it would commit `external/`, a symlink), never commit symlinks.
* The fp64 group wildcard run takes ~5.5 min warm; `operators` compiles ~200
  big kernels, >10 min COLD — run the group twice before judging.
* Full suite: `tests/headless/run.sh` (~6 min, rc must be 0).
