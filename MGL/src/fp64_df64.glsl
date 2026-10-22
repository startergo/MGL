/*
 * MGL double-single ("float-float") fp64 library (GLSL).
 *
 * Opt-in FAST arithmetic backend for MGL's software fp64 (MGL_FP64_FAST=1).
 * The default backend (fp64_softfloat.glsl) is an exact Berkeley SoftFloat
 * port; this one trades precision for speed by computing in the classic
 * double-single representation: a double x is carried as a pair of floats
 * (hi, lo) with x = hi + lo, |lo| <= ulp(hi)/2 — Dekker/Knuth error-free
 * transformations plus fused multiply-add, the approach described in
 * Andrew Thall, "Extended-Precision Floating-Point Numbers for GPU
 * Computation" (2006), and used by e.g. github.com/philipturner/metal-float64.
 *
 * The entry points, names and signatures are IDENTICAL to
 * fp64_softfloat.glsl, so the fp64 SPIR-V lowering pass
 * (MGL/src/fp64.c) is unchanged apart from selecting this source string.
 * Every entry point takes and returns uint64_t values holding IEEE-754
 * double BIT PATTERNS: values are unpacked to (hi, lo) with a few integer
 * ops, computed on in float arithmetic and packed back.  Memory contents,
 * constants, uniforms, xfb and stage interfaces therefore stay bit-exact
 * IEEE-754 doubles; only the results of arithmetic are approximate.
 *
 * Accuracy / range limits (documented in agents/tess_report.md):
 *   - ~48 significant bits (relative error < 2^-46) for add/sub/mul/fma;
 *     sqrt/div/rcp round-trips are a little worse (~2^-45 relative worst).
 *   - |x| < 2^127: doubles outside the float range clamp to +-infinity.
 *   - Double subnormals (< 2^-1022) flush to zero; results below ~2^-126
 *     keep only single-float precision.
 *   - Compares, sign/abs/neg, min/max, classification, ldexp, frexp,
 *     float<->double conversion and integral double<->int conversions are
 *     exact (integer/bit operations), including round-tripping through
 *     pack: every float is exactly representable as a double.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

/* NOTE: every internal helper is prefixed df64_.  The identifier survives the
 * glslang -> SPIR-V -> SPIRV-Cross trip into the generated MSL, where
 * MGLRenderer detects it and compiles the library with precise (IEEE) math:
 * Metal's default fast-math reassociates the error-free transformations
 * (Knuth two-sum), which silently drops the low float of every pair. */

/* ================================================================ helpers */

/* renormalize (s, e) with |e| <= ~ulp(s) into a canonical df pair */
vec2 df64_norm(float s, float e)
{
   float hi = s + e;
   float lo = e - (hi - s);
   return vec2(hi, lo);
}

/* exact two-term sum (Knuth): needs no fused op */
vec2 df64_add(vec2 a, vec2 b)
{
   float s = a.x + b.x;
   if (isinf(s))
      return vec2(s, 0.0);                /* inf: the error terms are +-inf/NaN */
   float v = s - a.x;
   float e = (a.x - (s - v)) + (b.x - v);
   e += a.y + b.y;
   return df64_norm(s, e);
}

/* product with the FMA-recovered rounding error */
vec2 df64_mul(vec2 a, vec2 b)
{
   float p = a.x * b.x;
   if (isinf(p))
      return vec2(p, 0.0);
   float e = fma(a.x, b.x, -p);
   e += a.x * b.y + a.y * b.x;
   return df64_norm(p, e);
}

/* fused multiply-add: single rounding of the mathematical a*b+c to df */
vec2 df64_fma(vec2 a, vec2 b, vec2 c)
{
   float p = a.x * b.x;
   float e = fma(a.x, b.x, -p);          /* exact product error */
   e += a.x * b.y + a.y * b.x;
   float s = p + c.x;
   float v = s - p;
   float se = (p - (s - v)) + (c.x - v); /* exact sum error */
   e += se + c.y;
   return df64_norm(s, e);
}

/* quotient by float division plus one Markstein-style correction */
vec2 df64_div(vec2 a, vec2 b)
{
   float q1 = a.x / b.x;
   float r = fma(-q1, b.x, a.x);         /* exact: a.x - q1*b.x */
   r += a.y - q1 * b.y;
   float q2 = r / b.x;
   return df64_norm(q1, q2);
}

/* one Newton refinement on top of the hardware sqrt */
vec2 df64_sqrt(vec2 a)
{
   float s1 = sqrt(a.x);
   if (a.x == 0.0)
      return vec2(0.0);
   vec2 t = df64_div(a, vec2(s1, 0.0));    /* a / s1 */
   /* s = (s1 + a/s1) / 2, carried exactly through the two-sum split */
   float s = s1 + t.x;
   float v = s - s1;
   float e = (s1 - (s - v)) + (t.x - v);
   e += t.y;
   return df64_norm(0.5 * s, 0.5 * e);
}

/* |df| (kept out of the mgl_ namespace) */
vec2 df64_abs2(vec2 x)
{
   return x.x < 0.0 ? -x : x;
}

/* exact widening of a float bit pattern to double bits */
uint64_t df64_f32bits_to_f64(uint fb)
{
   uint s = fb & 0x80000000u;
   uint e = (fb >> 23) & 0xffu;
   uint m = fb & 0x7fffffu;
   if (e == 0xffu) {
      if (m == 0u)
         return packUint2x32(uvec2(0u, s | 0x7ff00000u));
      return packUint2x32(uvec2(0u, s | 0x7ff80000u));
   }
   if (e == 0u) {
      if (m == 0u)
         return packUint2x32(uvec2(0u, s));
      /* subnormal float: normalize the mantissa into the double format */
      int b = findMSB(m);                       /* 0..22 */
      uint64_t mant = uint64_t(m << uint(23 - b));   /* [2^23, 2^24) */
      int ed = b - 172 + 1023;                  /* biased exponent */
      return (uint64_t(s) << 63) | (uint64_t(ed) << 52) |
             ((mant << 29) & 0x000fffffffffffffUL);
   }
   return (uint64_t(s) << 63) | (uint64_t(e + 896u) << 52) | (uint64_t(m) << 29);
}

/* IEEE-754 double bits -> canonical df pair */
vec2 df64_unpack(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);          /* b.y: sign|exponent|mant20 */
   uint s = b.y & 0x80000000u;
   uint e = (b.y >> 20) & 0x7ffu;
   if (e == 0u)
      return vec2(uintBitsToFloat(s), 0.0);          /* +-0; subnormal -> 0 */
   if (e >= 0x7ffu) {
      bool nan = (b.y & 0x000fffffu) != 0u || b.x != 0u;
      return vec2(uintBitsToFloat(s | 0x7f800000u | (nan ? 0x7fffffu : 0u)), 0.0);
   }
   /* below 2^-126 the pair could only live in float subnormals: flush (the
      documented fast-mode lower precision limit) */
   if (e < 897u || e > 1150u)
      return vec2(uintBitsToFloat(s | (e > 1150u ? 0x7f800000u : 0u)), 0.0);
   /* hi: sign, exponent-896, top 23 mantissa bits (truncating split) */
   uint hm = ((b.y & 0x000fffffu) << 3) | (b.x >> 29);
   float hi = uintBitsToFloat(s | ((e - 896u) << 23) | hm);
   /* lo: the remaining 29 bits, rounded to float, scaled by 2^(e-1075) */
   uint r = b.x & 0x1fffffffu;
   float lo = ldexp(float(r), int(e) - 1075);
   lo *= uintBitsToFloat(s | 0x3f800000u);           /* sign of x */
   return vec2(hi, lo);
}

/* df pair -> IEEE-754 double bits (round to nearest at the 53rd bit) */
uint64_t df64_pack(vec2 f)
{
   float hi = f.x, lo = f.y;
   /* keep the pair canonical: |hi| >= |lo| (callers maintain this, but the
      pack itself must never produce garbage from a swapped pair) */
   if (abs(lo) > abs(hi)) {
      float t = hi;
      hi = lo;
      lo = t;
   }
   uint hb = floatBitsToUint(hi);
   uint lb = floatBitsToUint(lo);
   uint s = hb & 0x80000000u;
   if ((hb << 1) == 0u) {                 /* hi == +-0: value is lo */
      if ((lb << 1) == 0u)
         return packUint2x32(uvec2(0u, s));
      return df64_f32bits_to_f64(lb);
   }
   uint e = (hb >> 23) & 0xffu;
   if (e == 0xffu) {                      /* inf / NaN out of the float ops */
      if ((hb & 0x7fffffu) != 0u || isnan(lo))
         return 0x7ff8000000000000UL;
      return packUint2x32(uvec2(0u, s | 0x7ff00000u));
   }
   if (e == 0u)
      return df64_f32bits_to_f64(hb);       /* tiny: exact float value */

   /* normal: 53-bit significand accumulator (implicit bit at 2^52) */
   uint64_t acc = uint64_t((hb & 0x7fffffu) | 0x800000u) << 29;
   uint el = (lb >> 23) & 0xffu;
   if (el != 0u && (lb << 1) != 0u) {
      int sh = int(el) - int(e);          /* <= 0: lo is below hi's ulp */
      if (sh >= -52) {
         uint64_t c;
         if (sh >= -29)
            c = uint64_t((lb & 0x7fffffu) | 0x800000u) << uint(sh + 29);
         else
            /* well-separated pair (heavy cancellation): shift the lo into
               the accumulator's low bits, dropping its own low bits */
            c = uint64_t((lb & 0x7fffffu) | 0x800000u) >> uint(-29 - sh);
         acc = ((lb >> 31) == (hb >> 31)) ? acc + c : acc - c;
      }
   }
   int eo = int(e) - 127 + 1023;
   if (acc >= 0x0020000000000000UL) {     /* carried past bit 52 */
      acc = (acc + 1UL) >> 1;             /* round half up */
      eo += 1;
   }
   if (eo >= 0x7ff)
      return packUint2x32(uvec2(0u, s | 0x7ff00000u));
   /* mantissa's top 20 bits: mask off the implicit bit at acc's bit 52 (it
      would land on the exponent's LSB and double every even-biased result) */
   return packUint2x32(uvec2(uint(acc), s | (uint(eo) << 20) | (uint(acc >> 32) & 0x000fffffu)));
}

/* ====================================================== bit-exact queries */

bool mgl_is_nan(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   return (0xFFE00000u <= (b.y << 1)) && ((b.x != 0u) || ((b.y & 0x000FFFFFu) != 0u));
}

bool mgl_isinf64(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   return (b.y << 1) == 0xFFE00000u;
}

uint64_t mgl_fabs64(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   b.y &= 0x7FFFFFFFu;
   return packUint2x32(b);
}

uint64_t mgl_fneg64(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   b.y ^= 0x80000000u;
   return packUint2x32(b);
}

uint64_t mgl_fsign64(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   uvec2 r;
   r.x = 0u;
   r.y = mix((b.y & 0x80000000u) | 0x3FF00000u, 0u, (b.y << 1 | b.x) == 0u);
   return packUint2x32(r);
}

bool mgl_feq64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return false;
   return a == b || ((a | b) & 0x7FFFFFFFFFFFFFFFUL) == 0UL;
}

bool mgl_fneu64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return true;
   return !(a == b || ((a | b) & 0x7FFFFFFFFFFFFFFFUL) == 0UL);
}

bool mgl_flt64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return false;
   if (((a | b) & 0x7FFFFFFFFFFFFFFFUL) == 0UL)
      return false;                      /* +-0 == -+0 */
   bool sa = (a >> 63) != 0UL;
   bool sb = (b >> 63) != 0UL;
   if (sa != sb)
      return sa;                         /* different signs */
   uint64_t ma = a & 0x7FFFFFFFFFFFFFFFUL;
   uint64_t mb = b & 0x7FFFFFFFFFFFFFFFUL;
   /* same sign: magnitudes order like the values (reversed for negatives) */
   return sa ? (ma > mb) : (ma < mb);
}

bool mgl_fge64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return false;
   return !mgl_flt64(a, b);
}

uint64_t mgl_fmin64(uint64_t a, uint64_t b)
{
   bool b_nan = mgl_is_nan(b);
   bool a_nan = mgl_is_nan(a);
   if (b_nan && a_nan)
      return a;
   if (a_nan)
      return b;
   if (b_nan)
      return a;
   /* equal values with different signs keep the negative one (IEEE minNum) */
   if (mgl_feq64(a, b))
      return ((a >> 63) != 0UL) ? a : b;
   return mgl_flt64(a, b) ? a : b;
}

uint64_t mgl_fmax64(uint64_t a, uint64_t b)
{
   bool b_nan = mgl_is_nan(b);
   bool a_nan = mgl_is_nan(a);
   if (b_nan && a_nan)
      return a;
   if (a_nan)
      return b;
   if (b_nan)
      return a;
   if (mgl_feq64(a, b))
      return ((a >> 63) != 0UL) ? b : a;
   return mgl_flt64(a, b) ? b : a;
}

/* =========================================================== arithmetic  */

uint64_t mgl_fadd64(uint64_t a, uint64_t b)
{
   return df64_pack(df64_add(df64_unpack(a), df64_unpack(b)));
}

uint64_t mgl_fsub64(uint64_t a, uint64_t b)
{
   return df64_pack(df64_add(df64_unpack(a), -df64_unpack(b)));
}

uint64_t mgl_fmul64(uint64_t a, uint64_t b)
{
   return df64_pack(df64_mul(df64_unpack(a), df64_unpack(b)));
}

uint64_t mgl_fdiv64(uint64_t a, uint64_t b)
{
   vec2 x = df64_unpack(a), y = df64_unpack(b);
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return 0x7ff8000000000000UL;
   if (y.x == 0.0) {
      if (x.x == 0.0)
         return 0x7ff8000000000000UL;            /* 0/0 -> NaN */
      return packUint2x32(uvec2(0u, uint(((a ^ b) & 0x8000000000000000UL) >> 32) |
                                0x7ff00000u));   /* x/0 -> signed inf */
   }
   return df64_pack(df64_div(x, y));
}

uint64_t mgl_ffma64(uint64_t a, uint64_t b, uint64_t c)
{
   return df64_pack(df64_fma(df64_unpack(a), df64_unpack(b), df64_unpack(c)));
}

uint64_t mgl_frcp64(uint64_t a)
{
   vec2 x = df64_unpack(a);
   if (x.x == 0.0)
      return 0x7ff8000000000000UL;
   return df64_pack(df64_div(vec2(1.0, 0.0), x));
}

uint64_t mgl_fsqrt64(uint64_t a)
{
   vec2 x = df64_unpack(a);
   if (x.x < 0.0 || mgl_is_nan(a))
      return 0x7ff8000000000000UL;
   if (x.x == 0.0)
      return a & 0x8000000000000000UL;   /* +-0 */
   return df64_pack(df64_sqrt(x));
}

uint64_t mgl_frsq64(uint64_t a)
{
   return df64_pack(df64_div(vec2(1.0, 0.0), df64_unpack(mgl_fsqrt64(a))));
}

/* --------------------------------------------------------- rounding family
 * For |x| >= 2^24 the high float is integral but SPARSE (steps of 2^(e-23)),
 * so the low float can be up to ulp/2 large and the result generally does
 * NOT fit in one float: it is returned as the canonical df pair
 * (h, round(hi)), whose low component is always <= ulp(h)/2.  Small results
 * collapse to (integral float, 0), which pack widens exactly. */

vec2 df64_floor(vec2 x)
{
   float h = floor(x.x);
   if (h == x.x)
      return df64_norm(h, floor(x.y));   /* both terms integral */
   float rem = (x.x - h) + x.y;          /* (-1, 2) */
   if (rem >= 1.0)
      h += 1.0;
   else if (rem < 0.0)
      h -= 1.0;
   return vec2(h, 0.0);
}

vec2 df64_ceil(vec2 x)
{
   float h = ceil(x.x);
   if (h == x.x)
      return df64_norm(h, ceil(x.y));
   float rem = (x.x - h) + x.y;          /* (-2, 0) */
   if (rem > 0.0)
      h += 1.0;
   else if (rem <= -1.0)
      h -= 1.0;
   return vec2(h, 0.0);
}

vec2 df64_trunc(vec2 x)
{
   float h = trunc(x.x);
   if (h == x.x)
      return df64_norm(h, trunc(x.y));
   float rem = (x.x - h) + x.y;          /* (-1.5, 1.5) */
   if (rem >= 1.0)
      h += 1.0;
   else if (rem <= -1.0)
      h -= 1.0;
   return vec2(h, 0.0);
}

uint64_t mgl_ffloor64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   return df64_pack(df64_floor(df64_unpack(a)));
}

uint64_t mgl_fceil64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   return df64_pack(df64_ceil(df64_unpack(a)));
}

uint64_t mgl_ftrunc64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   return df64_pack(df64_trunc(df64_unpack(a)));
}

/* GLSL round(): ties away from zero */
uint64_t mgl_fround64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   vec2 x = df64_unpack(a);
   if (x.x >= 0.0)
      return df64_pack(df64_floor(df64_add(x, vec2(0.5, 0.0))));
   return df64_pack(-df64_floor(df64_add(-x, vec2(0.5, 0.0))));
}

/* GLSL roundEven(): ties to even */
uint64_t mgl_froundeven64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   vec2 x = df64_unpack(a);
   /* nearest, ties away first; repair exact ties to the even neighbour */
   vec2 t = (x.x >= 0.0) ? df64_floor(df64_add(x, vec2(0.5, 0.0)))
                         : -df64_floor(df64_add(-x, vec2(0.5, 0.0)));
   vec2 d = df64_add(x, -t);               /* distance to t: 0, +-0.5, ... */
   uint64_t dd = df64_pack(df64_abs2(d));
   if (dd == 0x3FE0000000000000UL) {     /* |x - t| == 0.5 exactly */
      /* pick the even one of {t, t toward zero} */
      float other = t.x - sign(t.x);
      float pick = (fract(abs(other) * 0.5) == 0.0) ? other : t.x;
      return df64_pack(vec2(pick, 0.0));
   }
   return df64_pack(t);
}

uint64_t mgl_ffract64(uint64_t a)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return 0x7ff8000000000000UL;
   vec2 x = df64_unpack(a);
   return df64_pack(df64_add(x, -df64_floor(x)));
}

/* GLSL mod(): x - y*floor(x/y);  SPIR-V OpFRem: x - y*trunc(x/y) */
uint64_t mgl_fmod64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b) || mgl_isinf64(a))
      return 0x7ff8000000000000UL;
   if (mgl_isinf64(b))
      return a;
   vec2 x = df64_unpack(a), y = df64_unpack(b);
   if (y.x == 0.0)
      return 0x7ff8000000000000UL;
   return df64_pack(df64_add(x, -df64_mul(y, df64_floor(df64_div(x, y)))));
}

uint64_t mgl_frem64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b) || mgl_isinf64(a))
      return 0x7ff8000000000000UL;
   if (mgl_isinf64(b))
      return a;
   vec2 x = df64_unpack(a), y = df64_unpack(b);
   if (y.x == 0.0)
      return 0x7ff8000000000000UL;
   return df64_pack(df64_add(x, -df64_mul(y, df64_trunc(df64_div(x, y)))));
}

/* -------------------------------------------------------------- ldexp/frexp */

uint64_t mgl_fldexp64(uint64_t a, int e)
{
   if (mgl_is_nan(a) || mgl_isinf64(a))
      return a;
   uvec2 b = unpackUint2x32(a);
   if (((b.y << 1) | uint(b.x == 0u)) == 0u || ((b.y & 0x7ff00000u) == 0u && b.x == 0u))
      return a;                          /* +-0 */
   uint be = (b.y >> 20) & 0x7ffu;
   if (be == 0u)
      return a;                          /* subnormal: unsupported (limit) */
   int ne = int(be) + e;
   if (ne >= 0x7ff)
      return packUint2x32(uvec2(0u, (b.y & 0x80000000u) | 0x7ff00000u));
   if (ne <= 0)
      return packUint2x32(uvec2(0u, b.y & 0x80000000u));  /* underflow -> 0 */
   b.y = (b.y & 0x800fffffu) | (uint(ne) << 20);
   return packUint2x32(b);
}

int mgl_frexp64_exp(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   uint be = (b.y >> 20) & 0x7ffu;
   if (be == 0u || be == 0x7ffu)
      return 0;                          /* 0, subnormal, inf, NaN */
   return int(be) - 1022;
}

uint64_t mgl_frexp64_sig(uint64_t a)
{
   uvec2 b = unpackUint2x32(a);
   uint be = (b.y >> 20) & 0x7ffu;
   if (be == 0u || be == 0x7ffu)
      return a;
   b.y = (b.y & 0x800fffffu) | (1022u << 20);
   return packUint2x32(b);               /* [0.5, 1) * sign */
}

/* ------------------------------------------------------------ conversions */

uint64_t mgl_fp32_to_fp64(float f)
{
   return df64_pack(vec2(f, 0.0));
}

float mgl_fp64_to_fp32(uint64_t a)
{
   vec2 x = df64_unpack(a);
   return x.x + x.y;                     /* round to nearest float */
}

/* exact for every uint64 < 2^48, ~48 significant bits beyond: the value is
   decomposed into four exact 16-bit chunks and accumulated with two-sums */
vec2 df64_from_u64(uint64_t x)
{
   vec2 s = vec2(float(uint(x >> 48)) * 281474976710656.0, 0.0);      /* 2^48 */
   s = df64_add(s, vec2(float(uint(x >> 32) & 0xffffu) * 4294967296.0, 0.0));
   s = df64_add(s, vec2(float(uint(x) >> 16) * 65536.0, 0.0));
   s = df64_add(s, vec2(float(uint(x) & 0xffffu), 0.0));
   return s;
}

uint64_t mgl_uint64_to_fp64(uint64_t x)
{
   return df64_pack(df64_from_u64(x));
}

uint64_t mgl_int64_to_fp64(int64_t x)
{
   if (x >= 0)
      return df64_pack(df64_from_u64(uint64_t(x)));
   /* negate in u64 space (INT64_MIN maps to 2^63 exactly) */
   uint64_t mag = (~uint64_t(x)) + 1UL;
   return mgl_fneg64(df64_pack(df64_from_u64(mag)));
}

uint64_t mgl_uint_to_fp64(uint u)
{
   /* every uint < 2^24 is exact; beyond that the df path adds the residual */
   return df64_pack(df64_from_u64(uint64_t(u)));
}

uint64_t mgl_int_to_fp64(int i)
{
   return mgl_int64_to_fp64(int64_t(i));
}

/* truncating conversions: the df value is truncated exactly (both floats are
   integral afterwards), and each integral float converts exactly into
   int64/uint64; the sum is done in 64-bit integer arithmetic */
uint64_t mgl_fp64_to_uint64(uint64_t a)
{
   vec2 x = df64_unpack(a);
   if (mgl_is_nan(a))
      return 0UL;
   if (x.x >= 1.8446744073709552e19)    /* 2^64 */
      return 0xFFFFFFFFFFFFFFFFUL;
   if (x.x <= -1.0)
      return 0UL;
   vec2 t = df64_trunc(x);
   if (t.x <= -1.0 || t.x < 0.0)
      return 0UL;
   uint64_t r = uint64_t(t.x);
   if (t.y > 0.0)
      r += uint64_t(t.y);
   else if (t.y < 0.0)
      r -= uint64_t(-t.y);
   return r;
}

int64_t mgl_fp64_to_int64(uint64_t a)
{
   vec2 x = df64_unpack(a);
   if (mgl_is_nan(a))
      return 0L;
   if (x.x >= 9.2233720368547758e18)    /* 2^63 */
      return 0x7FFFFFFFFFFFFFFFL;
   if (x.x < -9.2233720368547758e18)
      return 0x8000000000000000L;
   vec2 t = df64_trunc(x);
   int64_t r = int64_t(t.x);
   if (t.y > 0.0)
      r += int64_t(t.y);
   else if (t.y < 0.0)
      r -= int64_t(-t.y);
   return r;
}

int mgl_fp64_to_int(uint64_t a)
{
   int64_t v = mgl_fp64_to_int64(a);
   if (v > 0x7FFFFFFFL)
      return 0x7FFFFFFF;
   if (v < -0x80000000L)
      return -0x7FFFFFFF - 1;
   return int(v);
}

uint mgl_fp64_to_uint(uint64_t a)
{
   uint64_t v = mgl_fp64_to_uint64(a);
   return uint(v > 0xFFFFFFFFUL ? 0xFFFFFFFFUL : v);
}

/* keep-alive: glslang removes functions nothing calls.  main() here is
 * dropped when the library is linked into a shader module. */
layout(location = 31) out vec4 mgl_lib_sink;
void main()
{
   uint64_t a = 0x0123456789abcdefUL;
   uint64_t b = 0x0fedcba987654321UL;
   float acc = 0.0;
   acc += float(mgl_fadd64(a, b));
   acc += float(mgl_fsub64(a, b));
   acc += float(mgl_fmul64(a, b));
   acc += float(mgl_fdiv64(a, b));
   acc += float(mgl_frem64(a, b));
   acc += float(mgl_fmod64(a, b));
   acc += float(mgl_fneg64(a));
   acc += float(mgl_fabs64(a));
   acc += float(mgl_fsign64(a));
   acc += mix(0.0f, 1.0f, mgl_feq64(a, b));
   acc += mix(0.0f, 1.0f, mgl_fneu64(a, b));
   acc += mix(0.0f, 1.0f, mgl_flt64(a, b));
   acc += mix(0.0f, 1.0f, mgl_fge64(a, b));
   acc += mix(0.0f, 1.0f, mgl_is_nan(a));
   acc += mix(0.0f, 1.0f, mgl_isinf64(a));
   acc += float(mgl_fceil64(a));
   acc += float(mgl_ffloor64(a));
   acc += float(mgl_ftrunc64(a));
   acc += float(mgl_fround64(a));
   acc += float(mgl_froundeven64(a));
   acc += float(mgl_ffract64(a));
   acc += float(mgl_fsqrt64(a));
   acc += float(mgl_frsq64(a));
   acc += float(mgl_frcp64(a));
   acc += float(mgl_fmin64(a, b));
   acc += float(mgl_fmax64(a, b));
   acc += float(mgl_ffma64(a, b, a));
   acc += float(mgl_fp32_to_fp64(1.5f));
   acc += mgl_fp64_to_fp32(a);
   acc += float(mgl_int_to_fp64(3));
   acc += float(mgl_uint_to_fp64(3u));
   acc += float(mgl_int64_to_fp64(int64_t(a)));
   acc += float(mgl_uint64_to_fp64(a));
   acc += float(mgl_fp64_to_int(a));
   acc += float(mgl_fp64_to_uint(a));
   acc += float(mgl_fp64_to_int64(a));
   acc += float(mgl_fp64_to_uint64(a));
   acc += float(mgl_fldexp64(a, 3));
   acc += float(mgl_frexp64_exp(a));
   acc += float(mgl_frexp64_sig(a));
   mgl_lib_sink = vec4(acc, float(a), float(b), 1.0);
}
