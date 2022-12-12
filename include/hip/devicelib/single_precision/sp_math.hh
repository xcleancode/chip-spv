/*
 * Copyright (c) 2021-22 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef HIP_INCLUDE_DEVICELIB_SP_MATH_H
#define HIP_INCLUDE_DEVICELIB_SP_MATH_H

#include <hip/devicelib/macros.hh>

extern "C++" {

extern __device__ float rint(float x);
extern __device__ float round(float x);
extern __device__ long int convert_long(float x);

extern __device__ float rnorm3df(float a, float b, float c);
extern __device__ float rnorm4df(float a, float b, float c, float d);

extern __device__ float lgamma(float x);

extern __device__ float erf(float x);
extern __device__ float erfc(float x);

extern __device__ float expm1(float x);

extern __device__ float log(float x);
extern __device__ float log2(float x);
extern __device__ float log10(float x);
extern __device__ float log1p(float x);

extern __device__ float ceil(float x);
extern __device__ float floor(float x);
extern __device__ float trunc(float x);

extern __device__ float sin(float x);
extern __device__ float cos(float x);
extern __device__ float tan(float x);
extern __device__ float asin(float x);
extern __device__ float acos(float x);
extern __device__ float atan(float x);
extern __device__ float sinh(float x);
extern __device__ float cosh(float x);
extern __device__ float tanh(float x);

extern __device__ float sqrt(float x);
}

static inline __device__ float rintf(float x) { return rint(x); }
static inline __device__ float roundf(float x) { return round(x); }

static inline __device__ long int lrintf(float x) {
  return convert_long(rint(x));
}
static inline __device__ long int lroundf(float x) {
  return convert_long(round(x));
}

static inline __device__ long long int llrintf(float x) { return lrintf(x); }
static inline __device__ long long int llroundf(float x) { return lroundf(x); }

static inline __device__ float lgammaf(float x) { return (lgamma(x)); };

static inline __device__ float erff(float x) { return ::erf(x); }
static inline __device__ float erfcf(float x) { return ::erfc(x); }

static inline __device__ float expm1f(float x) { return ::expm1(x); }

static inline __device__ float logf(float x) { return ::log(x); }
static inline __device__ float log2f(float x) { return ::log2(x); }
static inline __device__ float log10f(float x) { return ::log10(x); }
static inline __device__ float log1pf(float x) { return ::log1p(x); }

static inline __device__ float ceilf(float x) { return ::ceil(x); }
static inline __device__ float floorf(float x) { return ::floor(x); }
static inline __device__ float truncf(float x) { return ::trunc(x); }

static inline __device__ float sinf(float x) { return ::sin(x); }
static inline __device__ float cosf(float x) { return ::cos(x); }
static inline __device__ float tanf(float x) { return ::tan(x); }
static inline __device__ float asinf(float x) { return ::asin(x); }
static inline __device__ float acosf(float x) { return ::acos(x); }
static inline __device__ float atanf(float x) { return ::atan(x); }
static inline __device__ float sinhf(float x) { return ::sinh(x); }
static inline __device__ float coshf(float x) { return ::cosh(x); }
static inline __device__ float tanhf(float x) { return ::tanh(x); }

static inline __device__ float sqrtf(float x) { return ::sqrt(x); }

namespace std {
using ::erf;
using ::erff;
using ::erfc;
using ::erfcf;
using ::expm1;
using ::log;
using ::logf;
using ::log2;
using ::log2f;
using ::log10;
using ::log10f;
using ::log1p;
using ::ceil;
using ::ceilf;
using ::floor;
using ::floorf;
using ::trunc;
using ::sin;
using ::cos;
using ::tan;
using ::tanf;
using ::asin;
using ::acos;
using ::atan;
using ::sinh;
using ::sinhf;
using ::cosh;
using ::coshf;
using ::tanh;
using ::tanhf;
using ::sqrt;
using ::sqrtf;
using ::lgamma;
using ::lgammaf;
} // namespace std

  // __device__ float acosf(float x)
  // __device__  float acoshf ( float  x )
  // __device__  float asinf ( float  x )
  // __device__  float asinhf ( float  x )
  // __device__  float atan2f ( float  y, float  x )
  // __device__  float atanf ( float  x )
  // __device__  float atanhf ( float  x )
  // __device__  float cbrtf ( float  x )
  // __device__  float ceilf ( float  x )
  // __device__  float copysignf ( float  x, float  y )
  // __device__  float cosf ( float  x )
  // __device__  float coshf ( float  x )
  // __device__  float cospif ( float  x )
  // __device__  float cyl_bessel_i0f ( float  x )
  // __device__  float cyl_bessel_i1f ( float  x )
  // __device__  float erfcf ( float  x )
  // __device__  float erfcinvf ( float  x )
  // __device__  float erfcxf ( float  x )
  // __device__  float erff ( float  x )
  // __device__  float erfinvf ( float  x )
  // __device__  float exp10f ( float  x )
  // __device__  float exp2f ( float  x )
  // __device__  float expf ( float  x )
  // __device__  float expm1f ( float  x )
  // __device__  float fabsf ( float  x )
  // __device__  float fdimf ( float  x, float  y )
  // __device__  float fdividef ( float  x, float  y ) // not available in
  // double precision
  // __device__  float floorf ( float  x )
  // __device__  float fmaf ( float  x, float  y, float  z )
  // __device__  float fmaxf ( float  x, float  y )
  // __device__  float fminf ( float  x, float  y )
  // __device__  float fmodf ( float  x, float  y )
  // __device__  float frexpf ( float  x, int* nptr )
  // __device__  float hypotf ( float  x, float  y )
  // __device__  int ilogbf ( float  x )
  // __device__  __RETURN_TYPE 	isfinite ( float  a )
  // __device__  __RETURN_TYPE 	isinf ( float  a )
  // __device__  __RETURN_TYPE 	isnan ( float  a )
  // __device__  float j0f ( float  x )
  // __device__  float j1f ( float  x )
  // __device__  float jnf ( int  n, float  x )
  // __device__  float ldexpf ( float  x, int  exp )
  // __device__  float log10f ( float  x )
  // __device__  float log1pf ( float  x )
  // __device__  float log2f ( float  x )
  // __device__  float logbf ( float  x )
  // __device__  float logf ( float  x )
  // __device__  float max ( const float  a, const float  b )
  // __device__  float min ( const float  a, const float  b )
  // __device__  float modff ( float  x, float* iptr )
  // __device__  float nanf ( const char* tagp )
  // __device__  float nearbyintf ( float  x )
  // __device__  float nextafterf ( float  x, float  y )
  // __device__  float norm3df ( float  a, float  b, float  c )
  // __device__  float norm4df ( float  a, float  b, float  c, float  d )
  // __device__  float normcdff ( float  x )
  // __device__  float normcdfinvf ( float  x )
  // __device__  float normf ( int  dim, const float* p )
  // __device__  float powf ( float  x, float  y )
  // __device__  float rcbrtf ( float  x )
  // __device__  float remainderf ( float  x, float  y )
  // __device__  float remquof ( float  x, float  y, int* quo )
  // __device__  float rhypotf ( float  x, float  y )
  // __device__  float rnormf ( int  dim, const float* p )
  // __device__  float rsqrtf ( float  x )
  // __device__  float scalblnf ( float  x, long int  n )
  // __device__  float scalbnf ( float  x, int  n )
  // __device__  __RETURN_TYPE 	signbit ( float  a )
  // __device__  void sincosf ( float  x, float* sptr, float* cptr )
  // __device__  void sincospif ( float  x, float* sptr, float* cptr )
  // __device__  float sinf ( float  x )
  // __device__  float sinhf ( float  x )
  // __device__  float sinpif ( float  x )
  // __device__  float sqrtf ( float  x )
  // __device__  float tanf ( float  x )
  // __device__  float tanhf ( float  x )
  // __device__  float tgammaf ( float  x )
  // __device__  float truncf ( float  x )
  // __device__  float y0f ( float  x )
  // __device__  float y1f ( float  x )
  // __device__  float ynf ( int  n, float  x )

#endif // include guard
