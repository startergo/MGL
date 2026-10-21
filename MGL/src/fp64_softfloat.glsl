/*
 * MGL soft-float64 library (GLSL).
 *
 * The bulk of this file is Mesa's src/compiler/glsl/float64.glsl, a GLSL port
 * of Berkeley SoftFloat release 3e, Copyright (c) 1989-2008 Ian Lance Taylor
 * and distributed under the MIT/X11 license; see the SoftFloat notice in the
 * body below and Mesa's own Copyright for the GLSL port.  Used and relicensed
 * here under Apache-2.0 with the permission of those licenses; the SoftFloat
 * notice is retained as required.
 *
 * The section marked "MGL additions" near the end contains composed entry
 * points (ceil, div, rcp, mod, rem, ldexp, frexp, isinf, rsqrt) ported from
 * Mesa's src/compiler/nir/nir_lower_double_ops.c (MIT license).
 *
 * All functions here are pure uint/uint64_t math: the module contains no
 * 'double' at all, so it compiles to plain Metal integer code.
 */

/*
 * The implementations contained in this file are heavily based on the
 * implementations found in the Berkeley SoftFloat library. As such, they are
 * licensed under the same 3-clause BSD license:
 *
 * License for Berkeley SoftFloat Release 3e
 *
 * John R. Hauser
 * 2018 January 20
 *
 * The following applies to the whole of SoftFloat Release 3e as well as to
 * each source file individually.
 *
 * Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the
 * University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions, and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions, and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the University nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_ARB_shader_bit_encoding : enable
#extension GL_EXT_shader_integer_mix : enable
/* GL_MESA_shader_integer_functions dropped: unused here, unknown to glslang */

#pragma warning(off)

/* Software IEEE floating-point rounding mode.
 * GLSL spec section "4.7.1 Range and Precision":
 * The rounding mode cannot be set and is undefined.
 * But here, we are able to define the rounding mode at the compilation time.
 */
#define FLOAT_ROUND_NEAREST_EVEN    0
#define FLOAT_ROUND_TO_ZERO         1
#define FLOAT_ROUND_DOWN            2
#define FLOAT_ROUND_UP              3
#define FLOAT_ROUNDING_MODE         FLOAT_ROUND_NEAREST_EVEN

/* Relax propagation of NaN.  Binary operations with a NaN source will still
 * produce a NaN result, but it won't follow strict IEEE rules.
 */
#define RELAXED_NAN_PROPAGATION

/* Absolute value of a Float64 :
 * Clear the sign bit
 */
uint64_t
mgl_fabs64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   a.y &= 0x7FFFFFFFu;
   return packUint2x32(a);
}

/* Returns 1 if the double-precision floating-point value `a' is a NaN;
 * otherwise returns 0.
 */
bool
mgl_is_nan(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   return (0xFFE00000u <= (a.y<<1)) &&
      ((a.x != 0u) || ((a.y & 0x000FFFFFu) != 0u));
}

/* Negate value of a Float64 :
 * Toggle the sign bit
 */
uint64_t
mgl_fneg64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   a.y ^= (1u << 31);
   return packUint2x32(a);
}

uint64_t
mgl_fsign64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 retval;
   retval.x = 0u;
   retval.y = mix((a.y & 0x80000000u) | 0x3FF00000u, 0u, (a.y << 1 | a.x) == 0u);
   return packUint2x32(retval);
}

/* Returns the fraction bits of the double-precision floating-point value `a'.*/
uint
mgl_extractFloat64FracLo(uint64_t a)
{
   return unpackUint2x32(a).x;
}

uint
mgl_extractFloat64FracHi(uint64_t a)
{
   return unpackUint2x32(a).y & 0x000FFFFFu;
}

/* Returns the exponent bits of the double-precision floating-point value `a'.*/
int
mgl_extractFloat64Exp(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   return int((a.y>>20) & 0x7FFu);
}

bool
mgl_feq64_nonnan(uint64_t mgl_a, uint64_t mgl_b)
{
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 b = unpackUint2x32(mgl_b);
   return (a.x == b.x) &&
          ((a.y == b.y) || ((a.x == 0u) && (((a.y | b.y)<<1) == 0u)));
}

/* Returns true if the double-precision floating-point value `a' is equal to the
 * corresponding value `b', and false otherwise.  The comparison is performed
 * according to the IEEE Standard for Floating-Point Arithmetic.
 */
bool
mgl_feq64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return false;

   return mgl_feq64_nonnan(a, b);
}

/* Returns true if the double-precision floating-point value `a' is not equal
 * to the corresponding value `b', and false otherwise.  The comparison is
 * performed according to the IEEE Standard for Floating-Point Arithmetic.
 */
bool
mgl_fneu64(uint64_t a, uint64_t b)
{
   if (mgl_is_nan(a) || mgl_is_nan(b))
      return true;

   return !mgl_feq64_nonnan(a, b);
}

/* Returns the sign bit of the double-precision floating-point value `a'.*/
uint
mgl_extractFloat64Sign(uint64_t a)
{
   return unpackUint2x32(a).y & 0x80000000u;
}

/* Returns true if the signed 64-bit value formed by concatenating `a0' and
 * `a1' is less than the signed 64-bit value formed by concatenating `b0' and
 * `b1'.  Otherwise, returns false.
 */
bool
ilt64(uint a0, uint a1, uint b0, uint b1)
{
   return (int(a0) < int(b0)) || ((a0 == b0) && (a1 < b1));
}

bool
mgl_flt64_nonnan(uint64_t mgl_a, uint64_t mgl_b)
{
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 b = unpackUint2x32(mgl_b);

   /* IEEE 754 floating point numbers are specifically designed so that, with
    * two exceptions, values can be compared by bit-casting to signed integers
    * with the same number of bits.
    *
    * From https://en.wikipedia.org/wiki/IEEE_754-1985#Comparing_floating-point_numbers:
    *
    *    When comparing as 2's-complement integers: If the sign bits differ,
    *    the negative number precedes the positive number, so 2's complement
    *    gives the correct result (except that negative zero and positive zero
    *    should be considered equal). If both values are positive, the 2's
    *    complement comparison again gives the correct result. Otherwise (two
    *    negative numbers), the correct FP ordering is the opposite of the 2's
    *    complement ordering.
    *
    * The logic implied by the above quotation is:
    *
    *    !both_are_zero(a, b) && (both_negative(a, b) ? a > b : a < b)
    *
    * This is equivalent to
    *
    *    fneu(a, b) && (both_negative(a, b) ? a >= b : a < b)
    *
    *    fneu(a, b) && (both_negative(a, b) ? !(a < b) : a < b)
    *
    *    fneu(a, b) && ((both_negative(a, b) && !(a < b)) ||
    *                  (!both_negative(a, b) && (a < b)))
    *
    * (A!|B)&(A|!B) is (A xor B) which is implemented here using !=.
    *
    *    fneu(a, b) && (both_negative(a, b) != (a < b))
    */
   bool lt = ilt64(a.y, a.x, b.y, b.x);
   bool both_negative = (a.y & b.y & 0x80000000u) != 0;

   return !mgl_feq64_nonnan(mgl_a, mgl_b) && (lt != both_negative);
}

bool
mgl_flt64_nonnan_minmax(uint64_t mgl_a, uint64_t mgl_b)
{
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 b = unpackUint2x32(mgl_b);

   /* See mgl_flt64_nonnan. For implementing fmin/fmax, we compare -0 < 0, so the
    * implied logic is a bit simpler:
    *
    *    both_negative(a, b) ? a > b : a < b
    *
    * If a == b, it doesn't matter what we return, so that's equivalent to:
    *
    *    both_negative(a, b) ? a >= b : a < b
    *    both_negative(a, b) ? !(a < b) : a < b
    *    both_negative(a, b) ^ (a < b)
    *
    * XOR is again implemented using !=.
    */
   bool lt = ilt64(a.y, a.x, b.y, b.x);
   bool both_negative = (a.y & b.y & 0x80000000u) != 0;

   return (lt != both_negative);
}

/* Returns true if the double-precision floating-point value `a' is less than
 * the corresponding value `b', and false otherwise.  The comparison is performed
 * according to the IEEE Standard for Floating-Point Arithmetic.
 */
bool
mgl_flt64(uint64_t a, uint64_t b)
{
   /* This weird layout matters.  Doing the "obvious" thing results in extra
    * flow control being inserted to implement the short-circuit evaluation
    * rules.  Flow control is bad!
    */
   bool x = !mgl_is_nan(a);
   bool y = !mgl_is_nan(b);
   bool z = mgl_flt64_nonnan(a, b);

   return (x && y && z);
}

/* Returns true if the double-precision floating-point value `a' is greater
 * than or equal to * the corresponding value `b', and false otherwise.  The
 * comparison is performed * according to the IEEE Standard for Floating-Point
 * Arithmetic.
 */
bool
mgl_fge64(uint64_t a, uint64_t b)
{
   /* This weird layout matters.  Doing the "obvious" thing results in extra
    * flow control being inserted to implement the short-circuit evaluation
    * rules.  Flow control is bad!
    */
   bool x = !mgl_is_nan(a);
   bool y = !mgl_is_nan(b);
   bool z = !mgl_flt64_nonnan(a, b);

   return (x && y && z);
}

uint64_t
mgl_fsat64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);

   /* fsat(NaN) should be zero. */
   if (mgl_is_nan(mgl_a) || int(a.y) < 0)
      return 0ul;

   /* IEEE 754 floating point numbers are specifically designed so that, with
    * two exceptions, values can be compared by bit-casting to signed integers
    * with the same number of bits.
    *
    * From https://en.wikipedia.org/wiki/IEEE_754-1985#Comparing_floating-point_numbers:
    *
    *    When comparing as 2's-complement integers: If the sign bits differ,
    *    the negative number precedes the positive number, so 2's complement
    *    gives the correct result (except that negative zero and positive zero
    *    should be considered equal). If both values are positive, the 2's
    *    complement comparison again gives the correct result. Otherwise (two
    *    negative numbers), the correct FP ordering is the opposite of the 2's
    *    complement ordering.
    *
    * We know that both values are not negative, and we know that at least one
    * value is not zero.  Therefore, we can just use the 2's complement
    * comparison ordering.
    */
   if (ilt64(0x3FF00000, 0x00000000, a.y, a.x))
      return 0x3FF0000000000000ul;

   return mgl_a;
}

/* Adds the 64-bit value formed by concatenating `a0' and `a1' to the 64-bit
 * value formed by concatenating `b0' and `b1'.  Addition is modulo 2^64, so
 * any carry out is lost.  The result is broken into two 32-bit pieces which
 * are stored at the locations pointed to by `z0Ptr' and `z1Ptr'.
 */
void
mgl_add64(uint a0, uint a1, uint b0, uint b1,
        out uint z0Ptr,
        out uint z1Ptr)
{
   uint z1 = a1 + b1;
   z1Ptr = z1;
   z0Ptr = a0 + b0 + uint(z1 < a1);
}


/* Subtracts the 64-bit value formed by concatenating `b0' and `b1' from the
 * 64-bit value formed by concatenating `a0' and `a1'.  Subtraction is modulo
 * 2^64, so any borrow out (carry out) is lost.  The result is broken into two
 * 32-bit pieces which are stored at the locations pointed to by `z0Ptr' and
 * `z1Ptr'.
 */
void
mgl_sub64(uint a0, uint a1, uint b0, uint b1,
        out uint z0Ptr,
        out uint z1Ptr)
{
   z1Ptr = a1 - b1;
   z0Ptr = a0 - b0 - uint(a1 < b1);
}

/* Shifts the 64-bit value formed by concatenating `a0' and `a1' right by the
 * number of bits given in `count'.  If any nonzero bits are shifted off, they
 * are "jammed" into the least significant bit of the result by setting the
 * least significant bit to 1.  The value of `count' can be arbitrarily large;
 * in particular, if `count' is greater than 64, the result will be either 0
 * or 1, depending on whether the concatenation of `a0' and `a1' is zero or
 * nonzero.  The result is broken into two 32-bit pieces which are stored at
 * the locations pointed to by `z0Ptr' and `z1Ptr'.
 */
void
mgl_shift64RightJamming(uint a0,
                      uint a1,
                      int count,
                      out uint z0Ptr,
                      out uint z1Ptr)
{
   uint z0;
   uint z1;
   int negCount = (-count) & 31;

   z0 = mix(0u, a0, count == 0);
   z0 = mix(z0, (a0 >> count), count < 32);

   z1 = uint((a0 | a1) != 0u); /* count >= 64 */
   uint z1_lt64 = (a0>>(count & 31)) | uint(((a0<<negCount) | a1) != 0u);
   z1 = mix(z1, z1_lt64, count < 64);
   z1 = mix(z1, (a0 | uint(a1 != 0u)), count == 32);
   uint z1_lt32 = (a0<<negCount) | (a1>>count) | uint ((a1<<negCount) != 0u);
   z1 = mix(z1, z1_lt32, count < 32);
   z1 = mix(z1, a1, count == 0);
   z1Ptr = z1;
   z0Ptr = z0;
}

/* Shifts the 96-bit value formed by concatenating `a0', `a1', and `a2' right
 * by 32 _plus_ the number of bits given in `count'.  The shifted result is
 * at most 64 nonzero bits; these are broken into two 32-bit pieces which are
 * stored at the locations pointed to by `z0Ptr' and `z1Ptr'.  The bits shifted
 * off form a third 32-bit result as follows:  The _last_ bit shifted off is
 * the most-significant bit of the extra result, and the other 31 bits of the
 * extra result are all zero if and only if _all_but_the_last_ bits shifted off
 * were all zero.  This extra result is stored in the location pointed to by
 * `z2Ptr'.  The value of `count' can be arbitrarily large.
 *     (This routine makes more sense if `a0', `a1', and `a2' are considered
 * to form a fixed-point value with binary point between `a1' and `a2'.  This
 * fixed-point value is shifted right by the number of bits given in `count',
 * and the integer part of the result is returned at the locations pointed to
 * by `z0Ptr' and `z1Ptr'.  The fractional part of the result may be slightly
 * corrupted as described above, and is returned at the location pointed to by
 * `z2Ptr'.)
 */
void
mgl_shift64ExtraRightJamming(uint a0, uint a1, uint a2,
                           int count,
                           out uint z0Ptr,
                           out uint z1Ptr,
                           out uint z2Ptr)
{
   uint z0 = 0u;
   uint z1;
   uint z2;
   int negCount = (-count) & 31;

   z2 = mix(uint(a0 != 0u), a0, count == 64);
   z2 = mix(z2, a0 << negCount, count < 64);
   z2 = mix(z2, a1 << negCount, count < 32);

   z1 = mix(0u, (a0 >> (count & 31)), count < 64);
   z1 = mix(z1, (a0<<negCount) | (a1>>count), count < 32);

   a2 = mix(a2 | a1, a2, count < 32);
   z0 = mix(z0, a0 >> count, count < 32);
   z2 |= uint(a2 != 0u);

   z0 = mix(z0, 0u, (count == 32));
   z1 = mix(z1, a0, (count == 32));
   z2 = mix(z2, a1, (count == 32));
   z0 = mix(z0, a0, (count == 0));
   z1 = mix(z1, a1, (count == 0));
   z2 = mix(z2, a2, (count == 0));
   z2Ptr = z2;
   z1Ptr = z1;
   z0Ptr = z0;
}

/* Shifts the 64-bit value formed by concatenating `a0' and `a1' left by the
 * number of bits given in `count'.  Any bits shifted off are lost.  The value
 * of `count' must be less than 32.  The result is broken into two 32-bit
 * pieces which are stored at the locations pointed to by `z0Ptr' and `z1Ptr'.
 */
void
mgl_shortShift64Left(uint a0, uint a1,
                   int count,
                   out uint z0Ptr,
                   out uint z1Ptr)
{
   z1Ptr = a1<<count;
   z0Ptr = mix((a0 << count | (a1 >> ((-count) & 31))), a0, count == 0);
}

/* Packs the sign `zSign', the exponent `zExp', and the significand formed by
 * the concatenation of `zFrac0' and `zFrac1' into a double-precision floating-
 * point value, returning the result.  After being shifted into the proper
 * positions, the three fields `zSign', `zExp', and `zFrac0' are simply added
 * together to form the most significant 32 bits of the result.  This means
 * that any integer portion of `zFrac0' will be added into the exponent.  Since
 * a properly normalized significand will have an integer portion equal to 1,
 * the `zExp' input should be 1 less than the desired result exponent whenever
 * `zFrac0' and `zFrac1' concatenated form a complete, normalized significand.
 */
uint64_t
mgl_packFloat64(uint zSign, int zExp, uint zFrac0, uint zFrac1)
{
   uvec2 z;

   z.y = zSign + (uint(zExp) << 20) + zFrac0;
   z.x = zFrac1;
   return packUint2x32(z);
}

/* Takes an abstract floating-point value having sign `zSign', exponent `zExp',
 * and extended significand formed by the concatenation of `zFrac0', `zFrac1',
 * and `zFrac2', and returns the proper double-precision floating-point value
 * corresponding to the abstract input.  Ordinarily, the abstract value is
 * simply rounded and packed into the double-precision format, with the inexact
 * exception raised if the abstract input cannot be represented exactly.
 * However, if the abstract value is too large, the overflow and inexact
 * exceptions are raised and an infinity or maximal finite value is returned.
 * If the abstract value is too small, the input value is rounded to a
 * subnormal number, and the underflow and inexact exceptions are raised if the
 * abstract input cannot be represented exactly as a subnormal double-precision
 * floating-point number.
 *     The input significand must be normalized or smaller.  If the input
 * significand is not normalized, `zExp' must be 0; in that case, the result
 * returned is a subnormal number, and it must not require rounding.  In the
 * usual case that the input significand is normalized, `zExp' must be 1 less
 * than the "true" floating-point exponent.  The handling of underflow and
 * overflow follows the IEEE Standard for Floating-Point Arithmetic.
 */
uint64_t
mgl_roundAndPackFloat64(uint zSign,
                      int zExp,
                      uint zFrac0,
                      uint zFrac1,
                      uint zFrac2)
{
   bool roundNearestEven;
   bool increment;

   roundNearestEven = FLOAT_ROUNDING_MODE == FLOAT_ROUND_NEAREST_EVEN;
   increment = int(zFrac2) < 0;
   if (!roundNearestEven) {
      if (FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO) {
         increment = false;
      } else {
         if (zSign != 0u) {
            increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) &&
               (zFrac2 != 0u);
         } else {
            increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
               (zFrac2 != 0u);
         }
      }
   }
   if (0x7FD <= zExp) {
      if ((0x7FD < zExp) ||
         ((zExp == 0x7FD) &&
            (0x001FFFFFu == zFrac0 && 0xFFFFFFFFu == zFrac1) &&
               increment)) {
         if ((FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO) ||
            ((zSign != 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP)) ||
               ((zSign == 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN))) {
            return mgl_packFloat64(zSign, 0x7FE, 0x000FFFFFu, 0xFFFFFFFFu);
         }
         return mgl_packFloat64(zSign, 0x7FF, 0u, 0u);
      }
   }

   if (zExp < 0) {
      mgl_shift64ExtraRightJamming(
         zFrac0, zFrac1, zFrac2, -zExp, zFrac0, zFrac1, zFrac2);
      zExp = 0;
      if (roundNearestEven) {
         increment = int(zFrac2) < 0; /* MGL fix: signed (round bit) test */
      } else {
         if (zSign != 0u) {
            increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) &&
               (zFrac2 != 0u);
         } else {
            increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
               (zFrac2 != 0u);
         }
      }
   }

   if (increment) {
      mgl_add64(zFrac0, zFrac1, 0u, 1u, zFrac0, zFrac1);
      /* MGL fix: at an exact tie (round bit set, no sticky bits) clear the
         LSB so the result stays even (SoftFloat's "!= 0" was dropped here) */
      zFrac1 &= ~uint(zFrac2 == 0x80000000u);
   } else {
      zExp = mix(zExp, 0, (zFrac0 | zFrac1) == 0u);
   }
   return mgl_packFloat64(zSign, zExp, zFrac0, zFrac1);
}

uint64_t
mgl_roundAndPackUInt64(uint zSign, uint zFrac0, uint zFrac1, uint zFrac2)
{
   bool roundNearestEven;
   bool increment;
   uint64_t default_nan = 0xFFFFFFFFFFFFFFFFUL;

   roundNearestEven = FLOAT_ROUNDING_MODE == FLOAT_ROUND_NEAREST_EVEN;

   if (zFrac2 >= 0x80000000u)
      increment = false;

   if (!roundNearestEven) {
      if (zSign != 0u) {
         if ((FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) && (zFrac2 != 0u)) {
            increment = false;
         }
      } else {
         increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
            (zFrac2 != 0u);
      }
   }

   if (increment) {
      mgl_add64(zFrac0, zFrac1, 0u, 1u, zFrac0, zFrac1);
      if ((zFrac0 | zFrac1) != 0u)
         zFrac1 &= ~(1u) + uint(zFrac2 == 0u) & uint(roundNearestEven);
   }
   return mix(packUint2x32(uvec2(zFrac1, zFrac0)), default_nan,
              (zSign != 0u && (zFrac0 | zFrac1) != 0u));
}

int64_t
mgl_roundAndPackInt64(uint zSign, uint zFrac0, uint zFrac1, uint zFrac2)
{
   bool roundNearestEven;
   bool increment;
   int64_t default_NegNaN = -0x7FFFFFFFFFFFFFFEL;
   int64_t default_PosNaN = 0xFFFFFFFFFFFFFFFFL;

   roundNearestEven = FLOAT_ROUNDING_MODE == FLOAT_ROUND_NEAREST_EVEN;

   if (zFrac2 >= 0x80000000u)
      increment = false;

   if (!roundNearestEven) {
      if (zSign != 0u) {
         increment = ((FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) &&
            (zFrac2 != 0u));
      } else {
         increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
            (zFrac2 != 0u);
      }
   }

   if (increment) {
      mgl_add64(zFrac0, zFrac1, 0u, 1u, zFrac0, zFrac1);
      if ((zFrac0 | zFrac1) != 0u)
         zFrac1 &= ~(1u) + uint(zFrac2 == 0u) & uint(roundNearestEven);
   }

   int64_t absZ = mix(int64_t(packUint2x32(uvec2(zFrac1, zFrac0))),
                      -int64_t(packUint2x32(uvec2(zFrac1, zFrac0))),
                      zSign != 0u);
   int64_t nan = mix(default_PosNaN, default_NegNaN, zSign != 0u);
   return mix(absZ, nan, ((zSign != 0u) != (absZ < 0)) && bool(absZ));
}

/* Returns the number of leading 0 bits before the most-significant 1 bit of
 * `a'.  If `a' is zero, 32 is returned.
 */
int
mgl_countLeadingZeros32(uint a)
{
   return 31 - findMSB(a);
}

/* Takes an abstract floating-point value having sign `zSign', exponent `zExp',
 * and significand formed by the concatenation of `zSig0' and `zSig1', and
 * returns the proper double-precision floating-point value corresponding
 * to the abstract input.  This routine is just like `mgl_roundAndPackFloat64'
 * except that the input significand has fewer bits and does not have to be
 * normalized.  In all cases, `zExp' must be 1 less than the "true" floating-
 * point exponent.
 */
uint64_t
mgl_normalizeRoundAndPackFloat64(uint zSign,
                               int zExp,
                               uint zFrac0,
                               uint zFrac1)
{
   int shiftCount;
   uint zFrac2;

   if (zFrac0 == 0u) {
      zExp -= 32;
      zFrac0 = zFrac1;
      zFrac1 = 0u;
   }

   shiftCount = mgl_countLeadingZeros32(zFrac0) - 11;
   if (0 <= shiftCount) {
      zFrac2 = 0u;
      mgl_shortShift64Left(zFrac0, zFrac1, shiftCount, zFrac0, zFrac1);
   } else {
      mgl_shift64ExtraRightJamming(
         zFrac0, zFrac1, 0u, -shiftCount, zFrac0, zFrac1, zFrac2);
   }
   zExp -= shiftCount;
   return mgl_roundAndPackFloat64(zSign, zExp, zFrac0, zFrac1, zFrac2);
}

/* Takes two double-precision floating-point values `a' and `b', one of which
 * is a NaN, and returns the appropriate NaN result.
 */
uint64_t
mgl_propagateFloat64NaN(uint64_t mgl_a, uint64_t mgl_b)
{
#if defined RELAXED_NAN_PROPAGATION
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 b = unpackUint2x32(mgl_b);

   return packUint2x32(uvec2(a.x | b.x, a.y | b.y));
#else
   bool aIsNaN = mgl_is_nan(mgl_a);
   bool bIsNaN = mgl_is_nan(mgl_b);
   uvec2 a = unpackUint2x32(mgl_a);
   uvec2 b = unpackUint2x32(mgl_b);
   a.y |= 0x00080000u;
   b.y |= 0x00080000u;

   return packUint2x32(mix(b, mix(a, b, bvec2(bIsNaN, bIsNaN)), bvec2(aIsNaN, aIsNaN)));
#endif
}

/* If a shader is in the soft-fp64 path, it almost certainly has register
 * pressure problems.  Choose a method to exchange two values that does not
 * require a temporary.
 */
#define EXCHANGE(a, b) \
   do {                \
       a ^= b;         \
       b ^= a;         \
       a ^= b;         \
   } while (false)

/* Returns the result of adding the double-precision floating-point values
 * `a' and `b'.  The operation is performed according to the IEEE Standard for
 * Floating-Point Arithmetic.
 */
uint64_t
mgl_fadd64(uint64_t a, uint64_t b)
{
   uint aSign = mgl_extractFloat64Sign(a);
   uint bSign = mgl_extractFloat64Sign(b);
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   uint bFracLo = mgl_extractFloat64FracLo(b);
   uint bFracHi = mgl_extractFloat64FracHi(b);
   int aExp = mgl_extractFloat64Exp(a);
   int bExp = mgl_extractFloat64Exp(b);
   int expDiff = aExp - bExp;
   if (aSign == bSign) {
      uint zFrac0;
      uint zFrac1;
      uint zFrac2;
      int zExp;

      if (expDiff == 0) {
         if (aExp == 0x7FF) {
            bool propagate = ((aFracHi | bFracHi) | (aFracLo| bFracLo)) != 0u;
            return mix(a, mgl_propagateFloat64NaN(a, b), propagate);
         }
         mgl_add64(aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1);
         if (aExp == 0) {
            /* MGL fix: the sum may overflow the subnormal field; round it */
            if (0x00200000u <= zFrac0) {
               mgl_shift64ExtraRightJamming(
                  zFrac0, zFrac1, 0u, 1, zFrac0, zFrac1, zFrac2);
               return mgl_roundAndPackFloat64(aSign, 1, zFrac0, zFrac1, zFrac2);
            }
            return mgl_packFloat64(aSign, 0, zFrac0, zFrac1);
         }
         zFrac2 = 0u;
         zFrac0 |= 0x00200000u;
         zExp = aExp;
         mgl_shift64ExtraRightJamming(
            zFrac0, zFrac1, zFrac2, 1, zFrac0, zFrac1, zFrac2);
      } else {
         if (expDiff < 0) {
            EXCHANGE(aFracHi, bFracHi);
            EXCHANGE(aFracLo, bFracLo);
            EXCHANGE(aExp, bExp);
         }

         if (aExp == 0x7FF) {
            bool propagate = (aFracHi | aFracLo) != 0u;
            return mix(mgl_packFloat64(aSign, 0x7ff, 0u, 0u), mgl_propagateFloat64NaN(a, b), propagate);
         }

         expDiff = mix(abs(expDiff), abs(expDiff) - 1, bExp == 0);
         bFracHi = mix(bFracHi | 0x00100000u, bFracHi, bExp == 0);
         mgl_shift64ExtraRightJamming(
            bFracHi, bFracLo, 0u, expDiff, bFracHi, bFracLo, zFrac2);
         zExp = aExp;

         aFracHi |= 0x00100000u;
         mgl_add64(aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1);
         --zExp;
         if (!(zFrac0 < 0x00200000u)) {
            mgl_shift64ExtraRightJamming(zFrac0, zFrac1, zFrac2, 1, zFrac0, zFrac1, zFrac2);
            ++zExp;
         }
      }
      return mgl_roundAndPackFloat64(aSign, zExp, zFrac0, zFrac1, zFrac2);

   } else {
      int zExp;

      mgl_shortShift64Left(aFracHi, aFracLo, 10, aFracHi, aFracLo);
      mgl_shortShift64Left(bFracHi, bFracLo, 10, bFracHi, bFracLo);
      if (expDiff != 0) {
         uint zFrac0;
         uint zFrac1;

         if (expDiff < 0) {
            EXCHANGE(aFracHi, bFracHi);
            EXCHANGE(aFracLo, bFracLo);
            EXCHANGE(aExp, bExp);
            aSign ^= 0x80000000u;
         }

         if (aExp == 0x7FF) {
            bool propagate = (aFracHi | aFracLo) != 0u;
            return mix(mgl_packFloat64(aSign, 0x7ff, 0u, 0u), mgl_propagateFloat64NaN(a, b), propagate);
         }

         expDiff = mix(abs(expDiff), abs(expDiff) - 1, bExp == 0);
         bFracHi = mix(bFracHi | 0x40000000u, bFracHi, bExp == 0);
         mgl_shift64RightJamming(bFracHi, bFracLo, expDiff, bFracHi, bFracLo);
         aFracHi |= 0x40000000u;
         mgl_sub64(aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1);
         zExp = aExp;
         --zExp;
         return mgl_normalizeRoundAndPackFloat64(aSign, zExp - 10, zFrac0, zFrac1);
      }
      if (aExp == 0x7FF) {
         bool propagate = ((aFracHi | bFracHi) | (aFracLo | bFracLo)) != 0u;
         return mix(0xFFFFFFFFFFFFFFFFUL, mgl_propagateFloat64NaN(a, b), propagate);
      }
      bExp = mix(bExp, 1, aExp == 0);
      aExp = mix(aExp, 1, aExp == 0);

      uint zFrac0;
      uint zFrac1;
      uint sign_of_difference = 0;
      if (bFracHi < aFracHi) {
         mgl_sub64(aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1);
      }
      else if (aFracHi < bFracHi) {
         mgl_sub64(bFracHi, bFracLo, aFracHi, aFracLo, zFrac0, zFrac1);
         sign_of_difference = 0x80000000;
      }
      else if (bFracLo <= aFracLo) {
         /* It is possible that zFrac0 and zFrac1 may be zero after this. */
         mgl_sub64(aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1);
      }
      else {
         mgl_sub64(bFracHi, bFracLo, aFracHi, aFracLo, zFrac0, zFrac1);
         sign_of_difference = 0x80000000;
      }
      zExp = mix(bExp, aExp, sign_of_difference == 0u);
      aSign ^= sign_of_difference;
      uint64_t retval_0 = mgl_packFloat64(uint(FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) << 31, 0, 0u, 0u);
      uint64_t retval_1 = mgl_normalizeRoundAndPackFloat64(aSign, zExp - 11, zFrac0, zFrac1);
      return mix(retval_0, retval_1, zFrac0 != 0u || zFrac1 != 0u);
   }
}

/* Multiplies the 64-bit value formed by concatenating `a0' and `a1' to the
 * 64-bit value formed by concatenating `b0' and `b1' to obtain a 128-bit
 * product.  The product is broken into four 32-bit pieces which are stored at
 * the locations pointed to by `z0Ptr', `z1Ptr', `z2Ptr', and `z3Ptr'.
 */
void
mgl_mul64To128(uint a0, uint a1, uint b0, uint b1,
             out uint z0Ptr,
             out uint z1Ptr,
             out uint z2Ptr,
             out uint z3Ptr)
{
   uint z0 = 0u;
   uint z1 = 0u;
   uint z2 = 0u;
   uint z3 = 0u;
   uint more1 = 0u;
   uint more2 = 0u;

   umulExtended(a1, b1, z2, z3);
   umulExtended(a1, b0, z1, more2);
   mgl_add64(z1, more2, 0u, z2, z1, z2);
   umulExtended(a0, b0, z0, more1);
   mgl_add64(z0, more1, 0u, z1, z0, z1);
   umulExtended(a0, b1, more1, more2);
   mgl_add64(more1, more2, 0u, z2, more1, z2);
   mgl_add64(z0, z1, 0u, more1, z0, z1);
   z3Ptr = z3;
   z2Ptr = z2;
   z1Ptr = z1;
   z0Ptr = z0;
}

/* Normalizes the subnormal double-precision floating-point value represented
 * by the denormalized significand formed by the concatenation of `aFrac0' and
 * `aFrac1'.  The normalized exponent is stored at the location pointed to by
 * `zExpPtr'.  The most significant 21 bits of the normalized significand are
 * stored at the location pointed to by `zFrac0Ptr', and the least significant
 * 32 bits of the normalized significand are stored at the location pointed to
 * by `zFrac1Ptr'.
 */
void
mgl_normalizeFloat64Subnormal(uint aFrac0, uint aFrac1,
                            out int zExpPtr,
                            out uint zFrac0Ptr,
                            out uint zFrac1Ptr)
{
   int shiftCount;
   uint temp_zfrac0, temp_zfrac1;
   shiftCount = mgl_countLeadingZeros32(mix(aFrac0, aFrac1, aFrac0 == 0u)) - 11;
   zExpPtr = mix(1 - shiftCount, -shiftCount - 31, aFrac0 == 0u);

   temp_zfrac0 = mix(aFrac1<<shiftCount, aFrac1>>(-shiftCount), shiftCount < 0);
   temp_zfrac1 = mix(0u, aFrac1<<(shiftCount & 31), shiftCount < 0);

   mgl_shortShift64Left(aFrac0, aFrac1, shiftCount, zFrac0Ptr, zFrac1Ptr);

   zFrac0Ptr = mix(zFrac0Ptr, temp_zfrac0, aFrac0 == 0);
   zFrac1Ptr = mix(zFrac1Ptr, temp_zfrac1, aFrac0 == 0);
}

/* Returns the result of multiplying the double-precision floating-point values
 * `a' and `b'.  The operation is performed according to the IEEE Standard for
 * Floating-Point Arithmetic.
 */
uint64_t
mgl_fmul64(uint64_t a, uint64_t b)
{
   uint zFrac0 = 0u;
   uint zFrac1 = 0u;
   uint zFrac2 = 0u;
   uint zFrac3 = 0u;
   int zExp;

   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   uint bFracLo = mgl_extractFloat64FracLo(b);
   uint bFracHi = mgl_extractFloat64FracHi(b);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);
   int bExp = mgl_extractFloat64Exp(b);
   uint bSign = mgl_extractFloat64Sign(b);
   uint zSign = aSign ^ bSign;
   if (aExp == 0x7FF) {
      if (((aFracHi | aFracLo) != 0u) ||
         ((bExp == 0x7FF) && ((bFracHi | bFracLo) != 0u))) {
         return mgl_propagateFloat64NaN(a, b);
      }
      if ((uint(bExp) | bFracHi | bFracLo) == 0u)
            return 0xFFFFFFFFFFFFFFFFUL;
      return mgl_packFloat64(zSign, 0x7FF, 0u, 0u);
   }
   if (bExp == 0x7FF) {
      /* a cannot be NaN, but is b NaN? */
      if ((bFracHi | bFracLo) != 0u)
#if defined RELAXED_NAN_PROPAGATION
         return b;
#else
         return mgl_propagateFloat64NaN(a, b);
#endif
      if ((uint(aExp) | aFracHi | aFracLo) == 0u)
         return 0xFFFFFFFFFFFFFFFFUL;
      return mgl_packFloat64(zSign, 0x7FF, 0u, 0u);
   }
   if (aExp == 0) {
      if ((aFracHi | aFracLo) == 0u)
         return mgl_packFloat64(zSign, 0, 0u, 0u);
      mgl_normalizeFloat64Subnormal(aFracHi, aFracLo, aExp, aFracHi, aFracLo);
   }
   if (bExp == 0) {
      if ((bFracHi | bFracLo) == 0u)
         return mgl_packFloat64(zSign, 0, 0u, 0u);
      mgl_normalizeFloat64Subnormal(bFracHi, bFracLo, bExp, bFracHi, bFracLo);
   }
   zExp = aExp + bExp - 0x400;
   aFracHi |= 0x00100000u;
   mgl_shortShift64Left(bFracHi, bFracLo, 12, bFracHi, bFracLo);
   mgl_mul64To128(
      aFracHi, aFracLo, bFracHi, bFracLo, zFrac0, zFrac1, zFrac2, zFrac3);
   mgl_add64(zFrac0, zFrac1, aFracHi, aFracLo, zFrac0, zFrac1);
   zFrac2 |= uint(zFrac3 != 0u);
   if (0x00200000u <= zFrac0) {
      mgl_shift64ExtraRightJamming(
         zFrac0, zFrac1, zFrac2, 1, zFrac0, zFrac1, zFrac2);
      ++zExp;
   }
   return mgl_roundAndPackFloat64(zSign, zExp, zFrac0, zFrac1, zFrac2);
}

uint64_t mgl_fusedFMA64(uint64_t a, uint64_t b, uint64_t c);
uint64_t
mgl_ffma64(uint64_t a, uint64_t b, uint64_t c)
{
   return mgl_fusedFMA64(a, b, c);
}

/* Shifts the 64-bit value formed by concatenating `a0' and `a1' right by the
 * number of bits given in `count'.  Any bits shifted off are lost.  The value
 * of `count' can be arbitrarily large; in particular, if `count' is greater
 * than 64, the result will be 0.  The result is broken into two 32-bit pieces
 * which are stored at the locations pointed to by `z0Ptr' and `z1Ptr'.
 */
void
mgl_shift64Right(uint a0, uint a1,
               int count,
               out uint z0Ptr,
               out uint z1Ptr)
{
   uint z0;
   uint z1;
   int negCount = (-count) & 31;

   z0 = 0u;
   z0 = mix(z0, (a0 >> count), count < 32);
   z0 = mix(z0, a0, count == 0);

   z1 = mix(0u, (a0 >> (count & 31)), count < 64);
   z1 = mix(z1, (a0<<negCount) | (a1>>count), count < 32);
   z1 = mix(z1, a0, count == 0);

   z1Ptr = z1;
   z0Ptr = z0;
}

/* Returns the result of converting the double-precision floating-point value
 * `a' to the unsigned integer format.  The conversion is performed according
 * to the IEEE Standard for Floating-Point Arithmetic.
 */
uint
mgl_fp64_to_uint(uint64_t a)
{
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);

   if ((aExp == 0x7FF) && ((aFracHi | aFracLo) != 0u))
      return 0xFFFFFFFFu;

   aFracHi |= mix(0u, 0x00100000u, aExp != 0);

   int shiftDist = 0x427 - aExp;
   if (0 < shiftDist)
      mgl_shift64RightJamming(aFracHi, aFracLo, shiftDist, aFracHi, aFracLo);

   if ((aFracHi & 0xFFFFF000u) != 0u)
      return mix(~0u, 0u, aSign != 0u);

   uint z = 0u;
   uint zero = 0u;
   mgl_shift64Right(aFracHi, aFracLo, 12, zero, z);

   uint expt = mix(~0u, 0u, aSign != 0u);

   return mix(z, expt, (aSign != 0u) && (z != 0u));
}

uint64_t
mgl_uint_to_fp64(uint a)
{
   if (a == 0u)
      return 0ul;

   int shiftDist = mgl_countLeadingZeros32(a) + 21;

   uint aHigh = 0u;
   uint aLow = 0u;
   int negCount = (- shiftDist) & 31;

   aHigh = mix(0u, a<< shiftDist - 32, shiftDist < 64);
   aLow = 0u;
   aHigh = mix(aHigh, 0u, shiftDist == 0);
   aLow = mix(aLow, a, shiftDist ==0);
   aHigh = mix(aHigh, a >> negCount, shiftDist < 32);
   aLow = mix(aLow, a << shiftDist, shiftDist < 32);

   return mgl_packFloat64(0u, 0x432 - shiftDist, aHigh, aLow);
}

uint64_t
mgl_uint64_to_fp64(uint64_t a)
{
   if (a == 0u)
      return 0ul;

   uvec2 aFrac = unpackUint2x32(a);
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);

   if ((aFracHi & 0x80000000u) != 0u) {
      mgl_shift64RightJamming(aFracHi, aFracLo, 1, aFracHi, aFracLo);
      return mgl_roundAndPackFloat64(0, 0x433, aFracHi, aFracLo, 0u);
   } else {
      return mgl_normalizeRoundAndPackFloat64(0, 0x432, aFrac.y, aFrac.x);
   }
}

uint64_t
mgl_fp64_to_uint64(uint64_t a)
{
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);
   uint zFrac2 = 0u;
   uint64_t default_nan = 0xFFFFFFFFFFFFFFFFUL;

   aFracHi = mix(aFracHi, aFracHi | 0x00100000u, aExp != 0);
   int shiftCount = 0x433 - aExp;

   if ( shiftCount <= 0 ) {
      if (shiftCount < -11 && aExp == 0x7FF) {
         if ((aFracHi | aFracLo) != 0u)
            return mgl_propagateFloat64NaN(a, a);
         return mix(default_nan, a, aSign == 0u);
      }
      mgl_shortShift64Left(aFracHi, aFracLo, -shiftCount, aFracHi, aFracLo);
   } else {
      mgl_shift64ExtraRightJamming(aFracHi, aFracLo, zFrac2, shiftCount,
                                 aFracHi, aFracLo, zFrac2);
   }
   return mgl_roundAndPackUInt64(aSign, aFracHi, aFracLo, zFrac2);
}

int64_t
mgl_fp64_to_int64(uint64_t a)
{
   uint zFrac2 = 0u;
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);
   int64_t default_NegNaN = -0x7FFFFFFFFFFFFFFEL;
   int64_t default_PosNaN = 0xFFFFFFFFFFFFFFFFL;

   aFracHi = mix(aFracHi, aFracHi | 0x00100000u, aExp != 0);
   int shiftCount = 0x433 - aExp;

   if (shiftCount <= 0) {
      if (shiftCount < -11 && aExp == 0x7FF) {
         if ((aFracHi | aFracLo) != 0u)
            return default_NegNaN;
         return mix(default_NegNaN, default_PosNaN, aSign == 0u);
      }
      mgl_shortShift64Left(aFracHi, aFracLo, -shiftCount, aFracHi, aFracLo);
   } else {
      mgl_shift64ExtraRightJamming(aFracHi, aFracLo, zFrac2, shiftCount,
                                 aFracHi, aFracLo, zFrac2);
   }

   return mgl_roundAndPackInt64(aSign, aFracHi, aFracLo, zFrac2);
}

uint64_t
mgl_int64_to_fp64(int64_t a)
{
   if (a==0)
      return 0ul;

   uint64_t absA = mix(uint64_t(a), uint64_t(-a), a < 0);
   uint aFracHi = mgl_extractFloat64FracHi(absA);
   uvec2 aFrac = unpackUint2x32(absA);
   uint zSign = uint(unpackInt2x32(a).y) & 0x80000000u;

   if ((aFracHi & 0x80000000u) != 0u) {
      return mix(0ul, mgl_packFloat64(0x80000000u, 0x434, 0u, 0u), a < 0);
   }

   return mgl_normalizeRoundAndPackFloat64(zSign, 0x432, aFrac.y, aFrac.x);
}

/* Returns the result of converting the double-precision floating-point value
 * `a' to the 32-bit two's complement integer format.  The conversion is
 * performed according to the IEEE Standard for Floating-Point Arithmetic---
 * which means in particular that the conversion is rounded according to the
 * current rounding mode.  If `a' is a NaN, the largest positive integer is
 * returned.  Otherwise, if the conversion overflows, the largest integer with
 * the same sign as `a' is returned.
 */
int
mgl_fp64_to_int(uint64_t a)
{
   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);

   uint absZ = 0u;
   uint aFracExtra = 0u;
   int shiftCount = aExp - 0x413;

   if (0 <= shiftCount) {
      if (0x41E < aExp) {
         if ((aExp == 0x7FF) && bool(aFracHi | aFracLo))
            aSign = 0u;
         return mix(0x7FFFFFFF, 0x80000000, aSign != 0u);
      }
      mgl_shortShift64Left(aFracHi | 0x00100000u, aFracLo, shiftCount, absZ, aFracExtra);
   } else {
      if (aExp < 0x3FF)
         return 0;

      aFracHi |= 0x00100000u;
      aFracExtra = ( aFracHi << (shiftCount & 31)) | aFracLo;
      absZ = aFracHi >> (- shiftCount);
   }

   int z = mix(int(absZ), -int(absZ), aSign != 0u);
   int nan = mix(0x7FFFFFFF, 0x80000000, aSign != 0u);
   return mix(z, nan, ((aSign != 0u) != (z < 0)) && bool(z));
}

/* Returns the result of converting the 32-bit two's complement integer `a'
 * to the double-precision floating-point format.  The conversion is performed
 * according to the IEEE Standard for Floating-Point Arithmetic.
 */
uint64_t
mgl_int_to_fp64(int a)
{
   uint zFrac0 = 0u;
   uint zFrac1 = 0u;
   if (a==0)
      return mgl_packFloat64(0u, 0, 0u, 0u);
   uint zSign = uint(a) & 0x80000000u;
   uint absA = mix(uint(a), uint(-a), a < 0);
   int shiftCount = mgl_countLeadingZeros32(absA) - 11;
   if (0 <= shiftCount) {
      zFrac0 = absA << shiftCount;
      zFrac1 = 0u;
   } else {
      mgl_shift64Right(absA, 0u, -shiftCount, zFrac0, zFrac1);
   }
   return mgl_packFloat64(zSign, 0x412 - shiftCount, zFrac0, zFrac1);
}

uint64_t
mgl_bool_to_fp64(bool a)
{
   return packUint2x32(uvec2(0x00000000u, uint(-int(a) & 0x3ff00000)));
}

/* Packs the sign `zSign', exponent `zExp', and significand `zFrac' into a
 * single-precision floating-point value, returning the result.  After being
 * shifted into the proper positions, the three fields are simply added
 * together to form the result.  This means that any integer portion of `zSig'
 * will be added into the exponent.  Since a properly normalized significand
 * will have an integer portion equal to 1, the `zExp' input should be 1 less
 * than the desired result exponent whenever `zFrac' is a complete, normalized
 * significand.
 */
float
mgl_packFloat32(uint zSign, int zExp, uint zFrac)
{
   return uintBitsToFloat(zSign + (uint(zExp)<<23) + zFrac);
}

/* Takes an abstract floating-point value having sign `zSign', exponent `zExp',
 * and significand `zFrac', and returns the proper single-precision floating-
 * point value corresponding to the abstract input.  Ordinarily, the abstract
 * value is simply rounded and packed into the single-precision format, with
 * the inexact exception raised if the abstract input cannot be represented
 * exactly.  However, if the abstract value is too large, the overflow and
 * inexact exceptions are raised and an infinity or maximal finite value is
 * returned.  If the abstract value is too small, the input value is rounded to
 * a subnormal number, and the underflow and inexact exceptions are raised if
 * the abstract input cannot be represented exactly as a subnormal single-
 * precision floating-point number.
 *     The input significand `zFrac' has its binary point between bits 30
 * and 29, which is 7 bits to the left of the usual location.  This shifted
 * significand must be normalized or smaller.  If `zFrac' is not normalized,
 * `zExp' must be 0; in that case, the result returned is a subnormal number,
 * and it must not require rounding.  In the usual case that `zFrac' is
 * normalized, `zExp' must be 1 less than the "true" floating-point exponent.
 * The handling of underflow and overflow follows the IEEE Standard for
 * Floating-Point Arithmetic.
 */
float
mgl_roundAndPackFloat32(uint zSign, int zExp, uint zFrac)
{
   bool roundNearestEven;
   int roundIncrement;
   int roundBits;

   roundNearestEven = FLOAT_ROUNDING_MODE == FLOAT_ROUND_NEAREST_EVEN;
   roundIncrement = 0x40;
   if (!roundNearestEven) {
      if (FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO) {
         roundIncrement = 0;
      } else {
         roundIncrement = 0x7F;
         if (zSign != 0u) {
            if (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP)
               roundIncrement = 0;
         } else {
            if (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN)
               roundIncrement = 0;
         }
      }
   }
   roundBits = int(zFrac & 0x7Fu);
   if (0xFDu <= uint(zExp)) {
      if ((0xFD < zExp) || ((zExp == 0xFD) && (int(zFrac) + roundIncrement) < 0))
         return mgl_packFloat32(zSign, 0xFF, 0u) - float(roundIncrement == 0);
      int count = -zExp;
      bool zexp_lt0 = zExp < 0;
      uint zFrac_lt0 = mix(uint(zFrac != 0u), (zFrac>>count) | uint((zFrac<<((-count) & 31)) != 0u), (-zExp) < 32);
      zFrac = mix(zFrac, zFrac_lt0, zexp_lt0);
      roundBits = mix(roundBits, int(zFrac) & 0x7f, zexp_lt0);
      zExp = mix(zExp, 0, zexp_lt0);
   }
   zFrac = (zFrac + uint(roundIncrement))>>7;
   zFrac &= ~uint(((roundBits ^ 0x40) == 0) && roundNearestEven);

   return mgl_packFloat32(zSign, mix(zExp, 0, zFrac == 0u), zFrac);
}

/* Returns the result of converting the double-precision floating-point value
 * `a' to the single-precision floating-point format.  The conversion is
 * performed according to the IEEE Standard for Floating-Point Arithmetic.
 */
float
mgl_fp64_to_fp32(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   uint zFrac = 0u;
   uint allZero = 0u;

   uint aFracLo = mgl_extractFloat64FracLo(mgl_a);
   uint aFracHi = mgl_extractFloat64FracHi(mgl_a);
   int aExp = mgl_extractFloat64Exp(mgl_a);
   uint aSign = mgl_extractFloat64Sign(mgl_a);
   if (aExp == 0x7FF) {
      mgl_shortShift64Left(a.y, a.x, 12, a.y, a.x);
      float rval = uintBitsToFloat(aSign | 0x7FC00000u | (a.y>>9));
      rval = mix(mgl_packFloat32(aSign, 0xFF, 0u), rval, (aFracHi | aFracLo) != 0u);
      return rval;
   }
   mgl_shift64RightJamming(aFracHi, aFracLo, 22, allZero, zFrac);
   zFrac = mix(zFrac, zFrac | 0x40000000u, aExp != 0);
   return mgl_roundAndPackFloat32(aSign, aExp - 0x381, zFrac);
}

/* Returns the result of converting the single-precision floating-point value
 * `a' to the double-precision floating-point format.
 */
uint64_t
mgl_fp32_to_fp64(float f)
{
   uint a = floatBitsToUint(f);
   uint aFrac = a & 0x007FFFFFu;
   int aExp = int((a>>23) & 0xFFu);
   uint aSign = a & 0x80000000u;
   uint zFrac0 = 0u;
   uint zFrac1 = 0u;

   if (aExp == 0xFF) {
      if (aFrac != 0u) {
         uint nanLo = 0u;
         uint nanHi = a<<9;
         mgl_shift64Right(nanHi, nanLo, 12, nanHi, nanLo);
         nanHi |= aSign | 0x7FF80000u;
         return packUint2x32(uvec2(nanLo, nanHi));
      }
      return mgl_packFloat64(aSign, 0x7FF, 0u, 0u);
    }

   if (aExp == 0) {
      if (aFrac == 0u)
         return mgl_packFloat64(aSign, 0, 0u, 0u);
      /* Normalize subnormal */
      int shiftCount = mgl_countLeadingZeros32(aFrac) - 8;
      aFrac <<= shiftCount;
      aExp = 1 - shiftCount;
      --aExp;
   }

   mgl_shift64Right(aFrac, 0u, 3, zFrac0, zFrac1);
   return mgl_packFloat64(aSign, aExp + 0x380, zFrac0, zFrac1);
}

/* Adds the 96-bit value formed by concatenating `a0', `a1', and `a2' to the
 * 96-bit value formed by concatenating `b0', `b1', and `b2'.  Addition is
 * modulo 2^96, so any carry out is lost.  The result is broken into three
 * 32-bit pieces which are stored at the locations pointed to by `z0Ptr',
 * `z1Ptr', and `z2Ptr'.
 */
void
mgl_add96(uint a0, uint a1, uint a2,
        uint b0, uint b1, uint b2,
        out uint z0Ptr,
        out uint z1Ptr,
        out uint z2Ptr)
{
   uint z2 = a2 + b2;
   uint carry1 = uint(z2 < a2);
   uint z1 = a1 + b1;
   uint carry0 = uint(z1 < a1);
   uint z0 = a0 + b0;
   z1 += carry1;
   z0 += uint(z1 < carry1);
   z0 += carry0;
   z2Ptr = z2;
   z1Ptr = z1;
   z0Ptr = z0;
}

/* Subtracts the 96-bit value formed by concatenating `b0', `b1', and `b2' from
 * the 96-bit value formed by concatenating `a0', `a1', and `a2'.  Subtraction
 * is modulo 2^96, so any borrow out (carry out) is lost.  The result is broken
 * into three 32-bit pieces which are stored at the locations pointed to by
 * `z0Ptr', `z1Ptr', and `z2Ptr'.
 */
void
mgl_sub96(uint a0, uint a1, uint a2,
        uint b0, uint b1, uint b2,
        out uint z0Ptr,
        out uint z1Ptr,
        out uint z2Ptr)
{
   uint z2 = a2 - b2;
   uint borrow1 = uint(a2 < b2);
   uint z1 = a1 - b1;
   uint borrow0 = uint(a1 < b1);
   uint z0 = a0 - b0;
   z0 -= uint(z1 < borrow1);
   z1 -= borrow1;
   z0 -= borrow0;
   z2Ptr = z2;
   z1Ptr = z1;
   z0Ptr = z0;
}

/* Returns an approximation to the 32-bit integer quotient obtained by dividing
 * `b' into the 64-bit value formed by concatenating `a0' and `a1'.  The
 * divisor `b' must be at least 2^31.  If q is the exact quotient truncated
 * toward zero, the approximation returned lies between q and q + 2 inclusive.
 * If the exact quotient q is larger than 32 bits, the maximum positive 32-bit
 * unsigned integer is returned.
 */
uint
mgl_estimateDiv64To32(uint a0, uint a1, uint b)
{
   uint b0;
   uint b1;
   uint rem0 = 0u;
   uint rem1 = 0u;
   uint term0 = 0u;
   uint term1 = 0u;
   uint z;

   if (b <= a0)
      return 0xFFFFFFFFu;
   b0 = b>>16;
   z = (b0<<16 <= a0) ? 0xFFFF0000u : (a0 / b0)<<16;
   umulExtended(b, z, term0, term1);
   mgl_sub64(a0, a1, term0, term1, rem0, rem1);
   while (int(rem0) < 0) {
      z -= 0x10000u;
      b1 = b<<16;
      mgl_add64(rem0, rem1, b0, b1, rem0, rem1);
   }
   rem0 = (rem0<<16) | (rem1>>16);
   z |= (b0<<16 <= rem0) ? 0xFFFFu : rem0 / b0;
   return z;
}

uint
mgl_sqrtOddAdjustments(int index)
{
   uint res = 0u;
   if (index == 0)
      res = 0x0004u;
   if (index == 1)
      res = 0x0022u;
   if (index == 2)
      res = 0x005Du;
   if (index == 3)
      res = 0x00B1u;
   if (index == 4)
      res = 0x011Du;
   if (index == 5)
      res = 0x019Fu;
   if (index == 6)
      res = 0x0236u;
   if (index == 7)
      res = 0x02E0u;
   if (index == 8)
      res = 0x039Cu;
   if (index == 9)
      res = 0x0468u;
   if (index == 10)
      res = 0x0545u;
   if (index == 11)
      res = 0x631u;
   if (index == 12)
      res = 0x072Bu;
   if (index == 13)
      res = 0x0832u;
   if (index == 14)
      res = 0x0946u;
   if (index == 15)
      res = 0x0A67u;

   return res;
}

uint
mgl_sqrtEvenAdjustments(int index)
{
   uint res = 0u;
   if (index == 0)
      res = 0x0A2Du;
   if (index == 1)
      res = 0x08AFu;
   if (index == 2)
      res = 0x075Au;
   if (index == 3)
      res = 0x0629u;
   if (index == 4)
      res = 0x051Au;
   if (index == 5)
      res = 0x0429u;
   if (index == 6)
      res = 0x0356u;
   if (index == 7)
      res = 0x029Eu;
   if (index == 8)
      res = 0x0200u;
   if (index == 9)
      res = 0x0179u;
   if (index == 10)
      res = 0x0109u;
   if (index == 11)
      res = 0x00AFu;
   if (index == 12)
      res = 0x0068u;
   if (index == 13)
      res = 0x0034u;
   if (index == 14)
      res = 0x0012u;
   if (index == 15)
      res = 0x0002u;

   return res;
}

/* Returns an approximation to the square root of the 32-bit significand given
 * by `a'.  Considered as an integer, `a' must be at least 2^31.  If bit 0 of
 * `aExp' (the least significant bit) is 1, the integer returned approximates
 * 2^31*sqrt(`a'/2^31), where `a' is considered an integer.  If bit 0 of `aExp'
 * is 0, the integer returned approximates 2^31*sqrt(`a'/2^30).  In either
 * case, the approximation returned lies strictly within +/-2 of the exact
 * value.
 */
uint
mgl_estimateSqrt32(int aExp, uint a)
{
   uint z;

   int index = int(a>>27 & 15u);
   if ((aExp & 1) != 0) {
      z = 0x4000u + (a>>17) - mgl_sqrtOddAdjustments(index);
      z = ((a / z)<<14) + (z<<15);
      a >>= 1;
   } else {
      z = 0x8000u + (a>>17) - mgl_sqrtEvenAdjustments(index);
      z = a / z + z;
      z = (0x20000u <= z) ? 0xFFFF8000u : (z<<15);
      if (z <= a)
         return uint(int(a)>>1);
   }
   return ((mgl_estimateDiv64To32(a, 0u, z))>>1) + (z>>1);
}

/* Returns the square root of the double-precision floating-point value `a'.
 * The operation is performed according to the IEEE Standard for Floating-Point
 * Arithmetic.
 */
uint64_t
mgl_fsqrt64(uint64_t a)
{
   uint zFrac0 = 0u;
   uint zFrac1 = 0u;
   uint zFrac2 = 0u;
   uint doubleZFrac0 = 0u;
   uint rem0 = 0u;
   uint rem1 = 0u;
   uint rem2 = 0u;
   uint rem3 = 0u;
   uint term0 = 0u;
   uint term1 = 0u;
   uint term2 = 0u;
   uint term3 = 0u;
   uint64_t default_nan = 0xFFFFFFFFFFFFFFFFUL;

   uint aFracLo = mgl_extractFloat64FracLo(a);
   uint aFracHi = mgl_extractFloat64FracHi(a);
   int aExp = mgl_extractFloat64Exp(a);
   uint aSign = mgl_extractFloat64Sign(a);
   if (aExp == 0x7FF) {
      if ((aFracHi | aFracLo) != 0u)
         return mgl_propagateFloat64NaN(a, a);
      if (aSign == 0u)
         return a;
      return default_nan;
   }
   if (aSign != 0u) {
      if ((uint(aExp) | aFracHi | aFracLo) == 0u)
         return a;
      return default_nan;
   }
   if (aExp == 0) {
      if ((aFracHi | aFracLo) == 0u)
         return mgl_packFloat64(0u, 0, 0u, 0u);
      mgl_normalizeFloat64Subnormal(aFracHi, aFracLo, aExp, aFracHi, aFracLo);
   }
   int zExp = ((aExp - 0x3FF)>>1) + 0x3FE;
   aFracHi |= 0x00100000u;
   mgl_shortShift64Left(aFracHi, aFracLo, 11, term0, term1);
   zFrac0 = (mgl_estimateSqrt32(aExp, term0)>>1) + 1u;
   if (zFrac0 == 0u)
      zFrac0 = 0x7FFFFFFFu;
   doubleZFrac0 = zFrac0 + zFrac0;
   mgl_shortShift64Left(aFracHi, aFracLo, 9 - (aExp & 1), aFracHi, aFracLo);
   umulExtended(zFrac0, zFrac0, term0, term1);
   mgl_sub64(aFracHi, aFracLo, term0, term1, rem0, rem1);
   while (int(rem0) < 0) {
      --zFrac0;
      doubleZFrac0 -= 2u;
      mgl_add64(rem0, rem1, 0u, doubleZFrac0 | 1u, rem0, rem1);
   }
   zFrac1 = mgl_estimateDiv64To32(rem1, 0u, doubleZFrac0);
   if ((zFrac1 & 0x1FFu) <= 5u) {
      if (zFrac1 == 0u)
         zFrac1 = 1u;
      umulExtended(doubleZFrac0, zFrac1, term1, term2);
      mgl_sub64(rem1, 0u, term1, term2, rem1, rem2);
      umulExtended(zFrac1, zFrac1, term2, term3);
      mgl_sub96(rem1, rem2, 0u, 0u, term2, term3, rem1, rem2, rem3);
      while (int(rem1) < 0) {
         --zFrac1;
         mgl_shortShift64Left(0u, zFrac1, 1, term2, term3);
         term3 |= 1u;
         term2 |= doubleZFrac0;
         mgl_add96(rem1, rem2, rem3, 0u, term2, term3, rem1, rem2, rem3);
      }
      zFrac1 |= uint((rem1 | rem2 | rem3) != 0u);
   }
   mgl_shift64ExtraRightJamming(zFrac0, zFrac1, 0u, 10, zFrac0, zFrac1, zFrac2);
   return mgl_roundAndPackFloat64(0u, zExp, zFrac0, zFrac1, zFrac2);
}

uint64_t
mgl_ftrunc64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   int aExp = mgl_extractFloat64Exp(mgl_a);
   uint zLo;
   uint zHi;

   int unbiasedExp = aExp - 1023;
   int fracBits = 52 - unbiasedExp;
   uint maskLo = mix(~0u << fracBits, 0u, fracBits >= 32);
   uint maskHi = mix(~0u << (fracBits - 32), ~0u, fracBits < 33);
   zLo = maskLo & a.x;
   zHi = maskHi & a.y;

   /* MGL fix: |x| < 1 truncates to a signed zero, not +0 */
   zLo = mix(zLo, 0u, unbiasedExp < 0);
   zHi = mix(zHi, a.y & 0x80000000u, unbiasedExp < 0);
   zLo = mix(zLo, a.x, unbiasedExp > 52);
   zHi = mix(zHi, a.y, unbiasedExp > 52);
   return packUint2x32(uvec2(zLo, zHi));
}

uint64_t
mgl_ffloor64(uint64_t a)
{
   /* The big assumption is that when 'a' is NaN, mgl_ftrunc(a) returns a.  Based
    * on that assumption, NaN values that don't have the sign bit will safely
    * return NaN (identity).  This is guarded by RELAXED_NAN_PROPAGATION
    * because otherwise the NaN should have the "signal" bit set.  The
    * mgl_fadd64 will ensure that occurs.
    */
   bool is_positive =
#if defined RELAXED_NAN_PROPAGATION
      int(unpackUint2x32(a).y) >= 0
#else
      mgl_fge64(a, 0ul)
#endif
      ;
   uint64_t tr = mgl_ftrunc64(a);

   if (is_positive || mgl_feq64(tr, a)) {
      return tr;
   } else {
      return mgl_fadd64(tr, 0xbff0000000000000ul /* -1.0 */);
   }
}

uint64_t
mgl_fround64(uint64_t mgl_a)
{
   uvec2 a = unpackUint2x32(mgl_a);
   int unbiasedExp = mgl_extractFloat64Exp(mgl_a) - 1023;
   uint aHi = a.y;
   uint aLo = a.x;

   if (unbiasedExp < 20) {
      if (unbiasedExp < 0) {
         /* MGL fix: GLSL round() is half-away-from-zero.  |x| >= 0.5 (exp
            field >= 1022) rounds to +-1.0, everything below (incl. -0) to
            the signed zero of x. */
         aHi = (aHi & 0x80000000u) |
               (uint((aHi & 0x7FFFFFFFu) >= 0x3FE00000u) * 0x3FF00000u);
         aLo = 0u;
      } else {
         uint maskExp = 0x000FFFFFu >> unbiasedExp;
         /* MGL fix: GLSL round() is half-away-from-zero; Mesa's tie test
            runs after the half-ulp add, so carries into `lastBit` are
            mistaken for exact ties and rounded back down */
         aHi += 0x00080000u >> unbiasedExp;
         aHi &= ~maskExp;
         aLo = 0u;
      }
   } else if (unbiasedExp > 51 || unbiasedExp == 1024) {
      return mgl_a;
   } else {
      uint maskExp = 0xFFFFFFFFu >> (unbiasedExp - 20);
      if ((aLo & maskExp) == 0u)
         return mgl_a;
      uint tmp = aLo + (1u << (51 - unbiasedExp));
      if(tmp < aLo)
         aHi += 1u;
      aLo = tmp;
      aLo &= ~maskExp;
   }

   return packUint2x32(uvec2(aLo, aHi));
}

uint64_t
mgl_fmin64(uint64_t a, uint64_t b)
{
   /* This weird layout matters.  Doing the "obvious" thing results in extra
    * flow control being inserted to implement the short-circuit evaluation
    * rules.  Flow control is bad!
    */
   bool b_nan = mgl_is_nan(b);
   bool a_lt_b = mgl_flt64_nonnan_minmax(a, b);
   bool a_nan = mgl_is_nan(a);

   return (b_nan || a_lt_b) && !a_nan ? a : b;
}

uint64_t
mgl_fmax64(uint64_t a, uint64_t b)
{
   /* This weird layout matters.  Doing the "obvious" thing results in extra
    * flow control being inserted to implement the short-circuit evaluation
    * rules.  Flow control is bad!
    */
   bool b_nan = mgl_is_nan(b);
   bool a_lt_b = mgl_flt64_nonnan_minmax(a, b);
   bool a_nan = mgl_is_nan(a);

   return (!b_nan && a_lt_b) || a_nan ? b : a;
}

uint64_t
mgl_ffract64(uint64_t a)
{
   return mgl_fadd64(a, mgl_fneg64(mgl_ffloor64(a)));
}

bool
mgl_fisfinite64(uint64_t mgl_a)
{
   return mgl_extractFloat64Exp(mgl_a) != 0x7ff;
}

/* ==========================================================================
 * MGL additions: composed entry points.
 *
 * Ported from Mesa src/compiler/nir/nir_lower_double_ops.c:
 *   Copyright (C) 2015 Jason Ekstrand <jason.ekstrand@intel.com>
 *   Distributed under the MIT license.
 * Everything else is a direct composition of the SoftFloat primitives above.
 * ==========================================================================
 */
/* forward declarations (GLSL needs them for cross-calls between the composed
 * entry points below) */
uint64_t mgl_fldexp64(uint64_t a, int e);
uint64_t mgl_frcp64(uint64_t src);
int mgl_get_exp64(uint64_t a);

/* ------------------------------------------------------------------ fused
 * multiply-add with a single rounding (GLSL fma() for fp64).  Mesa's soft
 * fp64 lowers ffma to fmul+fadd (two roundings); we keep the exact product
 * in 192-bit fixed point, align it against the addend and round once, RNE.
 */
uint64_t mgl_fusedFMA64(uint64_t a, uint64_t b, uint64_t c)
{
   if (mgl_is_nan(a) || mgl_is_nan(b) || mgl_is_nan(c))
      return mgl_propagateFloat64NaN(mgl_propagateFloat64NaN(a, b), c);

   /* zero products, infinities and an infinite addend stay exact through
      the plain ops (fadd/fmul handle their signs and NaN creation) */
   if (mgl_get_exp64(a) == 0x7ff || mgl_get_exp64(b) == 0x7ff ||
       mgl_get_exp64(c) == 0x7ff ||
       (a & 0x7fffffffffffffffUL) == 0UL || (b & 0x7fffffffffffffffUL) == 0UL)
      return mgl_fadd64(mgl_fmul64(a, b), c);

   /* significands with the implicit bit in ([2^52,2^53)), unbiased exponents;
      value = sig * 2^(E - 52); subnormals are normalized first */
   uint64_t sigA = a & 0x000fffffffffffffUL;
   int Ea = mgl_get_exp64(a) - 1023;
   if (mgl_get_exp64(a) == 0) {
      int sc = ((sigA >> 32) == 0UL)
         ? 32 + int(mgl_countLeadingZeros32(uint(sigA))) - 11
         : int(mgl_countLeadingZeros32(uint(sigA >> 32))) - 11;
      sigA <<= sc;
      Ea = -1022 - sc;
   } else {
      sigA |= 0x0010000000000000UL;
   }
   uint64_t sigB = b & 0x000fffffffffffffUL;
   int Eb = mgl_get_exp64(b) - 1023;
   if (mgl_get_exp64(b) == 0) {
      int sc = ((sigB >> 32) == 0UL)
         ? 32 + int(mgl_countLeadingZeros32(uint(sigB))) - 11
         : int(mgl_countLeadingZeros32(uint(sigB >> 32))) - 11;
      sigB <<= sc;
      Eb = -1022 - sc;
   } else {
      sigB |= 0x0010000000000000UL;
   }
   uint64_t magC = c & 0x7fffffffffffffffUL;
   uint64_t sigC = magC & 0x000fffffffffffffUL;
   int Ec = mgl_get_exp64(c) - 1023;
   if (magC != 0UL && mgl_get_exp64(c) == 0) {
      int sc = ((sigC >> 32) == 0UL)
         ? 32 + int(mgl_countLeadingZeros32(uint(sigC))) - 11
         : int(mgl_countLeadingZeros32(uint(sigC >> 32))) - 11;
      sigC <<= sc;
      Ec = -1022 - sc;
   } else if (magC != 0UL) {
      sigC |= 0x0010000000000000UL;
   }

   /* exact product: 192-bit little-endian (X0, X1, X2) */
   uint p0, p1, p2, p3;
   mgl_mul64To128(uint(sigA >> 32), uint(sigA), uint(sigB >> 32), uint(sigB), p0, p1, p2, p3);
   uint64_t X0 = (uint64_t(p2) << 32) | p3;      /* low 64 of the product */
   uint64_t X1 = (uint64_t(p0) << 32) | p1;      /* high 64 */
   uint64_t X2 = 0UL;
   int Pu = Ea + Eb - 104;              /* product = P * 2^Pu */
   uint jam = 0u;                       /* nonzero bits below the sum's LSB */

   uint sp = ((a ^ b) & 0x8000000000000000UL) != 0UL ? 1u : 0u;
   uint sc_ = (c & 0x8000000000000000UL) != 0UL ? 1u : 0u;
   uint signZ = sp;

   if (magC != 0UL) {
      int Cu = Ec - 52;
      int d = Cu - Pu;
      uint64_t Y0 = 0UL, Y1 = 0UL, Y2 = 0UL;
      if (d >= 138) {
         /* product/c < 2^-84: it can only ever act as a sticky bit */
         jam = uint((X0 | X1) != 0UL);
         X0 = 0UL; X1 = 0UL;
         Pu = Cu;                       /* sum represented at the addend's unit */
         Y0 = sigC;
      } else if (d >= 0) {
         /* the addend's unit is coarser: shift the addend UP into the
            product's units, exactly (53 + d <= 137 bits fit the 192-bit
            sum).  The n == 0 / n == 64 corners are split out: a shift by
            64 (or by 0 on the complementing side) is undefined in GLSL
            and silently masked by the Metal backend. */
         uint n = uint(d);
         if (n == 0u) {
            Y0 = sigC;
         } else if (n < 64u) {
            Y0 = sigC << n;
            Y1 = sigC >> (64u - n);
         } else if (n == 64u) {
            Y1 = sigC;
         } else if (n < 128u) {
            Y1 = sigC << (n - 64u);
            Y2 = sigC >> (128u - n);
         } else {
            Y2 = sigC << (n - 128u);
         }
      } else {
         /* the addend's unit is finer: shift it DOWN; the shifted-out bits
            are strictly below any rounding position (no cancellation can
            make them significant: the product outmasses it by 2^51+) */
         uint n = uint(-d);
         if (n >= 64u) {
            jam = uint(sigC != 0UL);
            Y0 = 0UL;
         } else if (n > 0u) {
            jam = uint((sigC & ((1UL << n) - 1UL)) != 0UL);
            Y0 = sigC >> n;
         } else {
            Y0 = sigC;
         }
      }

      /* X +- Y on the 192-bit magnitudes (exact bits only) */
      if (sp == sc_) {
         uint64_t c0 = uint64_t(X0 + Y0 < X0);
         uint64_t s0 = X0 + Y0;
         uint64_t c1 = uint64_t(X1 + Y1 + c0 < X1);
         X1 = X1 + Y1 + c0;
         X0 = s0;
         X2 = X2 + Y2 + c1;
      } else {
         bool xge = (X2 > Y2) || (X2 == Y2 && X1 > Y1) || (X2 == Y2 && X1 == Y1 && X0 >= Y0);
         if (xge) {
            uint64_t borrow = uint64_t(X0 < Y0);
            X0 = X0 - Y0;
            uint64_t borrow2 = uint64_t(X1 < Y1 + borrow);
            X1 = X1 - Y1 - borrow;
            X2 = X2 - Y2 - borrow2;
         } else {
            signZ = sc_;
            uint64_t borrow = uint64_t(Y0 < X0);
            X0 = Y0 - X0;
            uint64_t borrow2 = uint64_t(Y1 < X1 + borrow);
            X1 = Y1 - X1 - borrow;
            X2 = Y2 - X2 - borrow2;
         }
      }
   }
   if ((X0 | X1 | X2) == 0UL)
      return uint64_t(signZ) << 63;     /* exact (or sticky-only) zero */

   /* top bit of the 192-bit sum */
   int t = 0;
   if (X2 != 0UL) {
      uint h = uint(X2 >> 32);
      t = (h != 0u) ? 128 + 32 + int(findMSB(h)) : 128 + int(findMSB(uint(X2)));
   } else if (X1 != 0UL) {
      uint h = uint(X1 >> 32);
      t = (h != 0u) ? 64 + 32 + int(findMSB(h)) : 64 + int(findMSB(uint(X1)));
   } else {
      uint h = uint(X0 >> 32);
      t = (h != 0u) ? 32 + int(findMSB(h)) : int(findMSB(uint(X0)));
   }

   /* 53-bit significand + round bit + sticky */
   uint64_t sig;
   int roundBit = 0;
   int sticky = int(jam);
   int k = t - 52;
   if (k >= 128) {
      uint kk = uint(k - 128);          /* <= 9 */
      sig = X2 >> kk;
      if (kk > 0u) {
         roundBit = int((X2 >> (kk - 1u)) & 1UL);
         sticky = sticky | int((X2 & ((1UL << (kk - 1u)) - 1UL)) != 0UL);
      } else {
         roundBit = int((X1 >> 63) & 1UL);
      }
      sticky = sticky | int((X1 | X0) != 0UL);
   } else if (k == 64) {
      sig = X1;
      roundBit = int((X0 >> 63) & 1UL);
      sticky = sticky | int((X0 & 0x7fffffffffffffffUL) != 0UL);
   } else if (k >= 64) {
      uint kk = uint(k - 64);           /* 1..63 */
      sig = (X1 >> kk) | (X2 << (64u - kk));
      roundBit = int((X1 >> (kk - 1u)) & 1UL);
      sticky = sticky | int((X1 & ((1UL << (kk - 1u)) - 1UL)) != 0UL);
      sticky = sticky | int(X0 != 0UL);
   } else if (k > 0) {
      uint kk = uint(k);                /* 1..63 */
      sig = (X1 << (64u - kk)) | (X0 >> kk);
      roundBit = int((X0 >> (kk - 1u)) & 1UL);
      if (kk > 1u) sticky = sticky | int((X0 & ((1UL << (kk - 1u)) - 1UL)) != 0UL);
   } else {
      sig = X0 << uint(-k);             /* X1 == X2 == 0 when t <= 52 */
   }
   int E = Pu + t;                      /* value = (X >> (t-52)) * 2^E */

   if (roundBit != 0 && (sticky != 0 || int(sig & 1UL) != 0)) {
      sig += 1UL;
      if (sig == 0x0020000000000000UL) { /* carry out of the 53rd bit */
         sig >>= 1;
         E += 1;
      }
   }

   if (E > 1023)
      return (uint64_t(signZ) << 63) | 0x7ff0000000000000UL;
   if (E >= -1022)
      return (uint64_t(signZ) << 63) | (uint64_t(E + 1023) << 52) |
             (sig & 0x000fffffffffffffUL);

   /* subnormal: round the exact 192-bit sum ONCE onto the 2^-1074 grid
      (rounding to 53 bits first and re-rounding would double-round) */
   int s = -(Pu + 1074);                /* value = X * 2^Pu */
   uint64_t N = 0UL;
   if (s <= 0) {                        /* exact shift left (<= 51 bits) */
      uint u = uint(-s);
      N = (u == 0u) ? X0 : ((X0 << u) | (X1 >> (64u - u)));   /* 64-u in [1,63] */
   } else if (s < 64) {
      uint ss = uint(s);                /* 1..63 */
      N = (X1 << (64u - ss)) | (X0 >> ss);
      int rb = int((X0 >> (ss - 1u)) & 1UL);
      int st = int((X0 & ((1UL << (ss - 1u)) - 1UL)) != 0UL);
      if (rb != 0 && (st != 0 || int(N & 1UL) != 0))
         N += 1UL;
   } else if (s == 64) {
      N = X1;
      int rb = int((X0 >> 63) & 1UL);
      int st = int((X0 & 0x7fffffffffffffffUL) != 0UL);
      if (rb != 0 && (st != 0 || int(N & 1UL) != 0))
         N += 1UL;
   } else if (s < 128) {
      uint ss = uint(s - 64);           /* 1..63 */
      N = (X2 << (64u - ss)) | (X1 >> ss);
      int rb = int((X1 >> (ss - 1u)) & 1UL);
      int st = int((X1 & ((1UL << (ss - 1u)) - 1UL)) != 0UL) | int(X0 != 0UL);
      if (rb != 0 && (st != 0 || int(N & 1UL) != 0))
         N += 1UL;
   } else {
      uint ss = uint(s - 128);
      N = X2 >> ss;
      int rb = ss > 0u ? int((X2 >> (ss - 1u)) & 1UL) : int((X1 >> 63) & 1UL);
      int st = (ss > 0u ? int((X2 & ((1UL << (ss - 1u)) - 1UL)) != 0UL) : 0) | int((X1 | X0) != 0UL);
      if (rb != 0 && (st != 0 || int(N & 1UL) != 0))
         N += 1UL;
   }
   if (N == 0UL)
      return uint64_t(signZ) << 63;
   return (uint64_t(signZ) << 63) | N;
}

/* exponent field of the IEEE double bit pattern (bits 52..62) */
int mgl_get_exp64(uint64_t a)
{
   return int((a >> 52) & 0x7ffUL);
}

/* a with its exponent field replaced by e */
uint64_t mgl_set_exp64(uint64_t a, int e)
{
   return (a & ~(0x7ffUL << 52)) | (uint64_t(e & 0x7ff) << 52);
}

bool mgl_isinf64(uint64_t a)
{
   return !mgl_fisfinite64(a) && !mgl_is_nan(a);
}

/* port of nir_lower_double_ops.c:fix_inv_result */
uint64_t mgl_fix_inv_result(uint64_t res, uint64_t src, int exp)
{
   /* exponent too small, or the input was infinity: flush to a signed zero */
   bool flush = (exp <= 0) || (mgl_fabs64(src) == 0x7ff0000000000000UL);
   uint64_t zero = uint64_t(0UL);
   uint64_t signed_zero = (src & 0x8000000000000000UL) | zero;
   res = mix(res, signed_zero, flush);

   /* NaN in, NaN out (the SoftFloat ops already do this, keep it for safety) */
   res = mix(res, src, mgl_is_nan(src));

   /* zero in: correctly-signed infinity out */
   bool is_zero = (src & 0x7fffffffffffffffUL) == 0UL;
   uint64_t signed_inf = (src & 0x8000000000000000UL) | 0x7ff0000000000000UL;
   res = mix(signed_inf, res, !is_zero);

   return res;
}

/* port of nir_lower_double_ops.c:lower_rcp — fp32 reciprocal + two
 * Newton-Raphson steps with exact fp64 fma */
uint64_t mgl_fsub64(uint64_t a, uint64_t b)
{
   return mgl_fadd64(a, mgl_fneg64(b));
}

uint64_t mgl_frcp64(uint64_t src)
{
   uint64_t mag = src & 0x7fffffffffffffffUL;
   uint64_t sign = src & 0x8000000000000000UL;
   if (mag == 0UL)                                 /* 1/+-0 = signed inf */
      return sign | 0x7ff0000000000000UL;
   if (mgl_get_exp64(src) == 0x7ff)                /* inf/NaN */
      return mgl_is_nan(src) ? src : sign;

   /* significand in [1,2) and its exponent; subnormals are normalized first
      so the fp32 seed is exact for every exponent (Mesa's version converts
      the raw value to fp32, which overflows/underflows for |src| > 2^127) */
   uint fracHi = mgl_extractFloat64FracHi(src);
   uint fracLo = mgl_extractFloat64FracLo(src);
   int E = mgl_get_exp64(src) - 1023;
   if (mgl_get_exp64(src) == 0) {
      int sc = (fracHi == 0u)
         ? 32 + int(mgl_countLeadingZeros32(fracLo)) - 11
         : int(mgl_countLeadingZeros32(fracHi)) - 11;
      mgl_shortShift64Left(fracHi, fracLo, sc, fracHi, fracLo);
      E = -1022 - sc;                   /* src_norm * 2^E = src */
   }
   /* the shifted-in leading 1 must not carry into the packed exponent */
   uint64_t src_norm = mgl_packFloat64(0u, 1023, fracHi & 0x000fffffu, fracLo);

   float f32 = mgl_fp64_to_fp32(src_norm);
   uint64_t ra = mgl_fp32_to_fp64(1.0f / f32);

   /* ra * 2^-E; ldexp handles overflow to inf and subnormal results
      (set_exp64 + a flush here loses |src| beyond ~2^511) */
   ra = mgl_fldexp64(ra, -E);
   if (mgl_get_exp64(ra) == 0x7ff)                 /* reciprocal overflows */
      return ra;
   ra ^= sign;                                     /* seed of 1/x keeps x's sign */

   /* x_new = x + x * (1 - x * src), twice */
   ra = mgl_ffma64(mgl_fneg64(ra), mgl_ffma64(ra, src, mgl_fneg64(mgl_fp32_to_fp64(1.0f))), ra);
   ra = mgl_ffma64(mgl_fneg64(ra), mgl_ffma64(ra, src, mgl_fneg64(mgl_fp32_to_fp64(1.0f))), ra);
   ra = mgl_ffma64(mgl_fneg64(ra), mgl_ffma64(ra, src, mgl_fneg64(mgl_fp32_to_fp64(1.0f))), ra);
   return ra;
}

uint64_t mgl_fdiv64(uint64_t a0, uint64_t b0)
{
   /* quotient + one exact-fma correction step: q = q + (a - b*q) * rcp(b).
      Operands are rescaled through locals: assigning to the parameters
      makes SPIRV-Cross emit `thread ulong&` MSL parameters, which a
      constant-space uniform member cannot bind to. */
   uint64_t a = a0;
   uint64_t b = b0;

   /* +-0 divided by anything: the residual correction would turn the zero
      quotient into NaN when 1/b overflows */
   if ((a & 0x7fffffffffffffffUL) == 0UL) {
      if ((b & 0x7fffffffffffffffUL) == 0UL)
         return 0x7ff8000000000000UL;             /* 0/0 */
      return (a ^ b) & 0x8000000000000000UL;      /* signed zero */
   }
   /* Keep both operands in a healthy range before refining:
      - a subnormal dividend underflows the Markstein residual (a - b*q);
      - a subnormal (or just tiny) divisor makes 1/b overflow.
      Scaling by powers of two commutes with the rounding, so the quotient
      is scaled back exactly at the end. */
   int k = 0, m = 0;
   if (mgl_get_exp64(a) == 0) {
      k = 200;                          /* residual ~2^-53 of a stays normal */
      a = mgl_fldexp64(a, k);
   }
   if (mgl_get_exp64(b) == 0) {
      m = 54;
      b = mgl_fldexp64(b, m);
   }
   uint64_t r = mgl_frcp64(b);
   if (mgl_get_exp64(r) == 0x7ff) {               /* 1/b still overflows */
      m += 64;
      b = mgl_fldexp64(b, 64);
      r = mgl_frcp64(b);
   }
   uint64_t q = mgl_fmul64(a, r);
   if (mgl_get_exp64(q) == 0x7ff)
      return q;                                    /* inf / NaN */
   if (q == 0ul || q == 0x8000000000000000UL)
      return (a ^ b) & 0x8000000000000000UL;      /* signed zero */
   /* Markstein: rem = fl(a - b*q) (exact, fused); q + rem*r is the
      correctly rounded quotient when r approximates 1/b to ~1 ulp */
   q = mgl_ffma64(mgl_ffma64(mgl_fneg64(b), q, a), r, q);
   if (mgl_get_exp64(q) == 0x7ff && mgl_get_exp64(r) != 0x7ff)
      return q;
   if (m != 0)
      q = mgl_fldexp64(q, m);
   if (k != 0)
      q = mgl_fldexp64(q, -k);
   return q;
}

uint64_t mgl_fceil64(uint64_t a)
{
   return mgl_fneg64(mgl_ffloor64(mgl_fneg64(a)));
}

/* GLSL roundEven(): nearest integer, ties to even (fround64 is the GLSL
 * round(), half away from zero).  Exact for every finite input: for
 * |a| < 2^52 the remainder a - trunc(a) and the neighbours t, t +- 1 are all
 * representable, and the parity of an integral double t lives in bit
 * (52 - E) of its u64 pattern. */
uint64_t mgl_froundeven64(uint64_t a)
{
   if (mgl_get_exp64(a) == 0x7ff) return a;              /* NaN / inf */
   uint64_t mag = a & 0x7fffffffffffffffUL;
   if (mag == 0UL) return a;                             /* +-0 */
   if (mag >= 0x4330000000000000UL) return a;            /* |a| >= 2^52: integral */
   uint64_t t = mgl_ftrunc64(a);
   uint64_t f = mgl_fsub64(a, t);                        /* exact, |f| < 1 */
   uint64_t af = f & 0x7fffffffffffffffUL;
   if (af < 0x3fe0000000000000UL) return t;              /* |f| < 0.5 */
   uint64_t up = mgl_fadd64(t, ((a & 0x8000000000000000UL) != 0UL)
                                ? 0xbff0000000000000UL  /* -1.0 */
                                : 0x3ff0000000000000UL);/* +1.0, exact */
   if (af > 0x3fe0000000000000UL) return up;             /* |f| > 0.5: away */
   /* exact tie: the even one of {t, up}.  Bit patterns of non-negative
      doubles order like their values, so 0.5 compares are integer compares. */
   if ((t & 0x7fffffffffffffffUL) == 0UL) return t;      /* |a| == 0.5: t is +-0 */
   uint k = uint(52 - (int((t >> 52) & 0x7ffu) - 1023)); /* t's parity bit */
   return ((t >> k) & 1UL) == 0UL ? t : up;
}

/* a + (-b): float64.glsl has no subtract entry point */
uint64_t mgl_frsq64(uint64_t a)
{
   return mgl_frcp64(mgl_fsqrt64(a));
}

/* GLSL frexp(x, out e): m in [0.5, 1.0) with x = m * 2^e.  +-0 -> m = +-0,
 * e = 0; inf/NaN -> m = x, e = 0 (the SPIR-V FrexpStruct contract). */
uint64_t mgl_frexp64_sig(uint64_t a)
{
   uint64_t mag = a & 0x7fffffffffffffffUL;
   uint sgn = uint(a >> 63);
   if (mag == 0UL || mgl_get_exp64(a) == 0x7ff) return a;
   uint64_t frac;
   if (mgl_get_exp64(a) == 0) {
      frac = mag;                     /* subnormal: normalize into [2^52, 2^53) */
      while (frac < 0x0010000000000000UL) frac <<= 1UL;
   } else {
      frac = (mag & 0x000fffffffffffffUL) | 0x0010000000000000UL;
   }
   /* m = frac * 2^-53 in [0.5, 1): exponent field 1022, mantissa = frac's
      low 52 bits (its implicit bit becomes the 0.5 place) */
   return (uint64_t(sgn) << 63) | 0x3fe0000000000000UL | (frac & 0x000fffffffffffffUL);
}

int mgl_frexp64_exp(uint64_t a)
{
   uint64_t mag = a & 0x7fffffffffffffffUL;
   if (mag == 0UL || mgl_get_exp64(a) == 0x7ff) return 0;
   int E;
   if (mgl_get_exp64(a) == 0) {
      uint64_t frac = mag;
      E = -1022;
      while (frac < 0x0010000000000000UL) { frac <<= 1UL; E -= 1; }
   } else {
      E = mgl_get_exp64(a) - 1023;
   }
   return E + 1;                      /* x = (frac * 2^-53) * 2^(E+1) */
}

/* GLSL mod(): x - y * floor(x / y) */
uint64_t mgl_fmod64(uint64_t a, uint64_t b)
{
   return mgl_fsub64(a, mgl_fmul64(b, mgl_ffloor64(mgl_fdiv64(a, b))));
}

/* SPIR-V OpFRem: x - y * trunc(x / y) */
uint64_t mgl_frem64(uint64_t a, uint64_t b)
{
   return mgl_fsub64(a, mgl_fmul64(b, mgl_ftrunc64(mgl_fdiv64(a, b))));
}

/* ldexp(x, e): x * 2^e (GLSL.std.450 Ldexp), including subnormal results.
 * Derived from first principles: every finite double is frac * 2^(E - 52)
 * with frac in [2^52, 2^53); ldexp just adds e to E. */
uint64_t mgl_fldexp64(uint64_t a, int e)
{
   if (mgl_is_nan(a)) return a;
   uint64_t mag = a & 0x7fffffffffffffffUL;
   if (mag == 0UL) return a;                          /* +-0 */
   if (mgl_get_exp64(a) == 0x7ff) return a;           /* +-inf */

   uint64_t frac;
   int E;
   if (mgl_get_exp64(a) == 0) {
      /* subnormal: value = m * 2^-1074; normalise m into [2^52, 2^53) */
      frac = mag;
      E = -1022;                                    /* value = frac(m) * 2^(E-52) */
      while (frac < 0x0010000000000000UL) {
         frac <<= 1UL;
         E -= 1;                                     /* value unchanged */
      }
   } else {
      frac = 0x0010000000000000UL | (mag & 0x000fffffffffffffUL);
      E = mgl_get_exp64(a) - 1023;                   /* value = 1.m * 2^E */
   }
   E += e;                                           /* result = 1.m * 2^E */

   int B = E + 1023;                                 /* biased exponent */
   if (B >= 0x7ff)
      return (a & 0x8000000000000000UL) | 0x7ff0000000000000UL;
   if (B >= 1)
      return (a & 0x8000000000000000UL) | ((uint64_t(B) & 0x7ffUL) << 52) |
             (frac & 0x000fffffffffffffUL);
   /* subnormal result: value = mant * 2^-1074 with mant = frac >> (-1022 - E)
    * (round to nearest even, like C ldexp) */
   int sh = -1022 - E;                               /* >= 1 here */
   if (sh >= 64)
      return a & 0x8000000000000000UL;               /* underflow to zero */
   uint64_t mant = frac >> sh;
   uint64_t lost = frac & ((1UL << sh) - 1UL);
   uint64_t halfway = 1UL << (sh - 1);
   if (lost > halfway || (lost == halfway && (mant & 1UL) != 0UL))
      mant += 1UL;                                   /* may carry into the normals */
   return (a & 0x8000000000000000UL) | mant;
}
/* keep-alive: glslang removes functions nothing calls.  main() here is
 * dropped when the library is linked into a shader module. */
layout(location = 31) out vec4 mgl_lib_sink;
void main()
{
   uint64_t a = 0x0123456789abcdefUL;
   uint64_t b = 0x0fedcba987654321UL;
   float acc = 0.0;
   acc += float(mgl_fabs64(a));
   acc += mix(0.0f, 1.0f, mgl_is_nan(a));
   acc += float(mgl_fneg64(a));
   acc += float(mgl_fsign64(a));
   acc += float(mgl_extractFloat64FracLo(a));
   acc += float(mgl_extractFloat64FracHi(a));
   acc += float(mgl_extractFloat64Exp(a));
   acc += mix(0.0f, 1.0f, mgl_feq64_nonnan(a, b));
   acc += mix(0.0f, 1.0f, mgl_feq64(a, b));
   acc += mix(0.0f, 1.0f, mgl_fneu64(a, b));
   acc += float(mgl_extractFloat64Sign(a));
   acc += mix(0.0f, 1.0f, mgl_flt64_nonnan(a, b));
   acc += mix(0.0f, 1.0f, mgl_flt64_nonnan_minmax(a, b));
   acc += mix(0.0f, 1.0f, mgl_flt64(a, b));
   acc += mix(0.0f, 1.0f, mgl_fge64(a, b));
   acc += float(mgl_fsat64(a));
   acc += float(mgl_packFloat64(3u, 4, 4u, 4u));
   acc += float(mgl_roundAndPackFloat64(3u, 4, 4u, 4u, 4u));
   acc += float(mgl_roundAndPackUInt64(3u, 4u, 4u, 4u));
   acc += float(mgl_countLeadingZeros32(3u));
   acc += float(mgl_normalizeRoundAndPackFloat64(3u, 4, 4u, 4u));
   acc += float(mgl_propagateFloat64NaN(a, b));
   acc += float(mgl_fadd64(a, b));
   acc += float(mgl_fmul64(a, b));
   acc += float(mgl_fusedFMA64(a, b, b));
   acc += float(mgl_ffma64(a, b, b));
   acc += float(mgl_fp64_to_uint(a));
   acc += float(mgl_uint_to_fp64(3u));
   acc += float(mgl_uint64_to_fp64(a));
   acc += float(mgl_fp64_to_uint64(a));
   acc += float(mgl_int64_to_fp64(int64_t(a)));
   acc += float(mgl_fp64_to_int(a));
   acc += float(mgl_int_to_fp64(3));
   acc += float(mgl_bool_to_fp64(true));
   acc += mgl_packFloat32(3u, 4, 4u);
   acc += mgl_roundAndPackFloat32(3u, 4, 4u);
   acc += mgl_fp64_to_fp32(a);
   acc += float(mgl_fp32_to_fp64(1.5f));
   acc += float(mgl_estimateDiv64To32(3u, 4u, 4u));
   acc += float(mgl_sqrtOddAdjustments(3));
   acc += float(mgl_sqrtEvenAdjustments(3));
   acc += float(mgl_estimateSqrt32(3, 4u));
   acc += float(mgl_fsqrt64(a));
   acc += float(mgl_ftrunc64(a));
   acc += float(mgl_ffloor64(a));
   acc += float(mgl_fround64(a));
   acc += float(mgl_fmin64(a, b));
   acc += float(mgl_fmax64(a, b));
   acc += float(mgl_ffract64(a));
   acc += mix(0.0f, 1.0f, mgl_fisfinite64(a));
   acc += float(mgl_fldexp64(a, 4));
   acc += float(mgl_frcp64(a));
   acc += float(mgl_get_exp64(a));
   acc += float(mgl_set_exp64(a, 4));
   acc += mix(0.0f, 1.0f, mgl_isinf64(a));
   acc += float(mgl_fix_inv_result(a, b, 4));
   acc += float(mgl_fsub64(a, b));
   acc += float(mgl_fdiv64(a, b));
   acc += float(mgl_fceil64(a));
   acc += float(mgl_froundeven64(a));
   acc += float(mgl_frsq64(a));
   acc += float(mgl_frexp64_sig(a));
   acc += float(mgl_frexp64_exp(a));
   acc += float(mgl_fmod64(a, b));
   acc += float(mgl_frem64(a, b));
   mgl_lib_sink = vec4(acc, float(a), float(b), 1.0);
}
