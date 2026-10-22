# tess — fp64 round 8 — HANDOFF

Branch `gl46/tess`, HEAD **4b02364** (merge 0f8e970 = gl46-port 01ec14a + d3db0af +
4b02364).  **DONE this round: `KHR-GL46.gpu_shader_fp64` 9/9 (varyings was the last
one), full headless suite 433 PASS / 0 FAIL.**  Next: the optional `MGL_FP64_FAST=1`
double-single mode.

## What was fixed (all in d3db0af unless noted)

1. **Conformant double vertex-attribute locations** (ARB_vertex_attrib_64bit):
   dvec3/dvec4 VERTEX INPUTS — and every array element / matrix column of them —
   consume **ONE GL location** (`layout(location=4) in dvec4 a[4]` → attributes 4-7);
   the SPIR-V `Location` decoration is the *internal Metal slot* (two per dvec3/dvec4
   element) that the fp64 word-pair interface and the vattrib plans use.
   * `introspect_spirv.c assign_interface_locations`: dual allocation (GL map
     `ip->attrib_gl` + internal slot map), conflict check in GL space, `attrib_overflow`
     (vectors > MAX_VERTEX_ATTRIBS, dvec3/dvec4 counting two — spec-allowed) → link
     error in program.c ("too many vertex attribute components").
   * `MGLIVar.slot` (mgl_ext_introspect.h) carries the internal slot; reflect_io fills
     it for vertex inputs; `mgl_introspect_slot_owner` (program_introspection.c) maps
     any internal slot → owning GL location + component base (unit = 2 slots for
     dvec3/dvec4-based inputs).
   * `mglGetAttribLocation` (program.c) consults the introspection table first.
   * Plans: `MGLVattribDrawPlan.owner_gl[]` (mgl_ext_vattrib.h) +
     `TessAttribSlot.glloc` (MGLRenderer+Tessellation.m) — the const/current-value
     path reads `current[owner_gl]` (one `glVertexAttribL*d` call holds all four
     doubles and feeds both word slots).
2. **fp64.c chain walk**: (a) the generic AccessChain rewrite now sets
   `f->vty[rid] = newrt` (stale ptr-to-double type made rw_load_store emit
   `OpLoad v2uint` from a uint4 slot pointer + an illegal `OpBitcast v2uint→v2ulong`
   which SPIRV-Cross renders as a zero-extending *conversion* — struct-member double
   reads returned 0/garbage); (b) the dvec3/dvec4 element-component fold and the
   matrix column fold add the recorded `elemslice` offset (`elemidx*elemstride`,
   an `OpIAdd` for dynamic column indices) — array element [1]/matrix column c read
   their own slots; (c) scalar-double array attribute elements bitcast directly
   (`as_type<ulong>(uint2)`), not through ulong2.
3. **geometry.c pass-through reader** (`geom_gen_draw_glsl`): emits whole uint4 slots
   as `(lo,hi,lo',hi')` word quadruples from the densely-packed record, with the
   correct per-element stride (the old code emitted `uvec4(uvec2(lo,lo'),0,0)` and a
   zero stride for vectors) and `uvec2` slots for `double[N]` members.
4. **4b02364 — canonical-record keying** (pre-existing failures, present at 9eacc94
   and through the gl46-port merges — the round-7 "422/0" note was wrong):
   * builtins (`gl_Position` & co) key through the fixed table FIRST; the vertex
     stage's top-level `gl_Position` variable was otherwise name-hashed while the TES
     side used the builtin key → the vs→tcs record had TWO gl_Position members and
     the no-TCS (default-levels) hardware path rasterized nothing
     (TessDefaultLevels, EmuDefaultLevelsGpu).
   * flattened consumer members (`tc_arr_0/1` from arrays-of-arrays) re-check the
     stage varying list for the base name (exact or `name[0]`) and key by that
     LOCATION (0 is valid — matched flag, not `key != 0`) so they alias the
     producer's whole-array member (TessArrayedVarying).
   * test_fp64c's conflict subtest now uses `location = 0` twice (a dvec4 at 0 plus a
     vec2 at 1 is legal under the one-location rule).

## Verification
* `KHR-GL46.gpu_shader_fp64.fp64.*`: **9/9 Pass** (varyings 31 s; operators ~13 min —
  needs a >240 s timeout).
* Full `tests/headless/run.sh`: **433 PASS / 0 FAIL** (incl. the gl46-port merge's
  new tests).
* `python3 agents/stub_scanner.py --list "EMPTY|NOOP|ASSERT"` clean for
  tessellation.c / MGLRenderer+Tessellation.m / fp64.c / geometry.c / test_tess.c.

## Scratch repros (untracked; do not commit)
`tests/headless/zz26_varval2.c` (CTS-faithful dvec3[2]+dvec4 chain, xfb + FS pixel,
modes: default/nogs/notess/pix), `zz27_varyarr.c` (flat varying arrays/matrices,
26 checks), `zz28_smallvar.c` (the CTS small-types multi-variable batch),
`zz29_dattr.c` (double[2] attribute current values), `zz30_deflvl.c` (no-TCS default
levels isolation), zz31/zz32 (bisect leftovers — deletable).
Debug knobs: `MGL_TESS_DEBUG`, `MGL_GEOM_DUMP=<dir>` (generated GLSL/MSL),
`MGL_FP64_DUMP` (writes /tmp/fp64msl<stage>.metal + /tmp/fp64lower<stage>.spv,
disassemble with `external/SPIRV-Tools/build/tools/spirv-dis`), `MGL_LOC_DEBUG`.

## Known gaps / notes for MGL_FP64_FAST
* Whole-column access into a wide double matrix stage variable (`m[c]` where the
  column is a dvec3/dvec4) is a link error in fp64.c ("whole-column access ... not
  supported"); the CTS only reads components (`m[c].r`) — fine for conformance, but a
  real app could hit it.
* Internal attribute slots are capped at 16 (= MAX_ATTRIBS arrays everywhere); the
  spec-allowed counting means >8 dvec4s = link error.  Metal allows 31 if ever
  needed (would require widening the plan arrays).
* zz26's `notess` mode (VS+GS, dvec attributes) errors 0x502 — not exercised by the
  CTS (fp64+GS without tess); left unfixed.
* fp64 groups that remain NotSupported in the full pass are unrelated (uniform block
  introspection shapes etc., see conformance.md).
* For MGL_FP64_FAST: the soft-float call set lives in fp64.c (`mgl_dadd` & friends
  injected as GLSL std.450-ish helpers into the lowered module); a double-single
  mode would swap the lowering's arithmetic emission, keeping the SAME
  slot/word-pair interface (which is now CTS-proven end-to-end through
  VS→TCS→TES→GS→FS and xfb).
