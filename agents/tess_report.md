# tess — round 9: MGL_FP64_FAST (double-single fp64) + tessellation conformance

Branch `gl46/tess`, HEAD **807903c** (= gl46-port 0b49dfd + 2d34339 fp64-fast).
Suite: **438 PASS / 0 FAIL** default; **433 PASS** with `MGL_FP64_FAST=1`.

## 1. MGL_FP64_FAST=1 — opt-in double-single (float-float) fp64

`MGL_FP64_FAST=1` (any value except `0`) selects a second arithmetic backend
for MGL's software fp64: **double-single** arithmetic — a double is carried
as a pair of floats (hi, lo) with x = hi + lo, computed with Dekker/Knuth
error-free transformations plus fused multiply-add (the approach of A. Thall,
*"Extended-Precision Floating-Point Numbers for GPU Computation"*, also used
by github.com/philipturner/metal-float64).  Default (unset / `0`) is the
historic exact SoftFloat path, **byte-for-byte unchanged** (verified: full
suite green, exact battery in test_fp64fast checks bit-exactness vs the CPU).

Files: `MGL/src/fp64_df64.glsl` (+ generated `fp64_df64.h`), selection in
`fp64.c`, precise-math compile in `MGLRenderer.m`/`MGLRenderer+Program.m`
via `mslCompileOptionsForSource:` (MGLRenderer+Tessellation.m).

### How it works
* Same 39 entry points/signatures as `fp64_softfloat.glsl`; every entry
  point takes/returns **IEEE-754 double bit patterns**, unpacks to (hi, lo)
  with a few integer ops, computes in floats, packs back.  Memory contents,
  constants, uniforms, xfb capture, stage IO and vertex attributes therefore
  stay exact IEEE doubles; only arithmetic results are approximate.
* Metal compiles MSL with **fast math** by default, which reassociates the
  two-sum EFTs (e.g. `e - (hi - s)` -> `(e - hi) + s`) and silently drops the
  low float (verified live).  The fast library's internal helpers are named
  `df64_*`; `fp64.c` re-adds `OpName` markers for the merged functions
  (merge drops them), the identifier survives glslang -> SPIR-V ->
  SPIRV-Cross into the MSL, and the renderer compiles such sources with
  `mathMode = MTLMathModeSafe` + precise FP functions.
* Also fixed (generic, mode-independent): a type-dedup collision in
  `merge_lib` — when the module lacks a type the library needs (e.g. `float`
  in a double-only compute shader), the new type is emitted with a shifted
  id but derived keys (vectors/pointers/function types) still hashed the
  unshifted library id, which could collide with an unrelated module id
  (observed: the library's `vec2` keyed onto the module's `v2ulong`,
  producing `sqrt(ulong)` compile errors in compute).

### Accuracy / range limits (MGL_FP64_FAST=1)
* ~48 significant bits: relative error < 2^-46 for add/sub/mul/fma,
  slightly worse for div/sqrt/rcp (one Markstein/Newton refinement).
* Cancellation-heavy results (fract, mod) are accurate to ~2^-44 * |inputs|
  absolute, not relative to the small result (inherent to float-float).
* |x| < 2^127 (~1.7e38): doubles beyond the float range clamp to ±inf;
  values below 2^-126 flush to ±0 (no double subnormals); results below
  ~2^-126 keep only single-float precision.
* Exact in fast mode: compares (ordered/unordered), isnan/isinf, abs, neg,
  sign, min/max, ldexp, frexp (integer/bit operations), float<->double
  conversions, double<->int64 conversions of integral values < 2^48,
  floor/ceil/trunc/round/roundeven of values that fit 48 bits.
* The precise-math compile applies to the whole program's MSL, so the user
  shader's own float code is also compiled with IEEE semantics (no fast
  approximations) while the flag is set.
* CTS `KHR-GL46.gpu_shader_fp64` targets exact IEEE behavior and will NOT
  pass in fast mode by design; the three bit-exactness headless tests
  (test_fp64/b/c) fail on last-bit assertions, all other 433 tests pass.

### Tests
`tests/headless/test_fp64fast.c` — 298 checks, both modes in one binary
(mode is read per lowering): 24-operator battery vs CPU double (exact mode
bit-exact; fast mode within 2^-44, input-scaled for fract/mod), NaN/inf
classification + compares via select chains (bit-exact both modes), and a
`distinct` check proving the two modes lower differently.
