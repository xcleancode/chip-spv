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

// from OpenCL device math library
extern __device__ float rint(float x);
extern __device__ float round(float x);
extern __device__ u_int32_t as_uint(float x);
extern __device__ long int convert_long(float x);

extern __device__  float lgamma(float x);

extern __device__ float acos(float x);
extern __device__  float acosh(float x);
extern __device__  float asin(float x);
extern __device__  float asinh(float x);
extern __device__  float atan2(float x, float y);
extern __device__  float atan(float x);
extern __device__  float atanh(float x);

extern __device__  float cbrt(float x);
extern __device__  float ceil(float x);

extern __device__  float copysign(float x, float y) ;
extern __device__  float cos(float x);
extern __device__  float cosh(float x);
extern __device__  float cospi(float x);

extern __device__  float erf(float x);
extern __device__  float erfc(float x);

extern __device__  float exp10(float x);
extern __device__  float exp2(float x);
extern __device__  float exp(float x);
extern __device__  float expm1(float x);

extern __device__  float fabs(float x);
extern __device__  float fdim(float x, float y);
extern __device__  float floor(float x);
extern __device__  float fma(float x, float y, float z);

extern __device__  float fmax(float x, float y);
extern __device__  float fmin(float x, float y);
extern __device__  float fmod(float x, float y);

extern __device__  float frexp(float x, int* nptr);
extern __device__  float hypot(float x, float y);
extern __device__  int ilogb(float x);
extern __device__  float ldexp(float x, int exp);

extern __device__  int isfinite(float  a);
extern __device__  int isinf(float  a);
extern __device__  int isnan(float  a);

extern __device__  float log10(float x);
extern __device__  float log1p(float x);
extern __device__  float log2(float x);
extern __device__  float logb(float x);
extern __device__  float log(float x);

extern __device__  float max(float x, float y);
extern __device__  float min(float x, float y);
extern __device__  float modf(float x, float* iptr);
//extern __device__ float nan ( u_int32_t nancode );
extern __device__  float nextafter(float x, float y);

extern __device__  float pow(float x, float y);
extern __device__ float remainder ( float  x, float  y );
extern __device__ float rsqrt ( float  x );
extern __device__  float remquo(float x, float y, int* quo);

extern __device__  float sincos(float x, float* cosptr);
extern __device__  float sin(float x);
extern __device__  float sinh(float x);
extern __device__  float sinpi(float x);
extern __device__  float sqrt(float x);
extern __device__  float tan(float x);
extern __device__  float tanh(float x);
extern __device__  float tgamma(float x);
extern __device__  float trunc(float x);

// from OCML (bitcode/*)
extern __device__ float rnorm3df(float a, float b, float c);
extern __device__ float rnorm4df(float a, float b, float c, float d);

}

static inline __device__ float rintf(float x) { return rint(x); }
static inline __device__ float roundf(float x) { return round(x); }

static inline __device__ long int lrintf(float x) { return convert_long(rint(x)); }
static inline __device__ long int lroundf(float x) { return convert_long(round(x)); }

static inline __device__ long long int llrintf(float x) { return lrintf(x); }
static inline __device__ long long int llroundf(float x) { return lroundf(x); }

static inline __device__  float lgammaf ( float  x ) { return (lgamma(x)); }

static inline __device__ float acosf(float x) { return acos(x); }
static inline __device__  float acoshf(float x) { return acosh(x); }
static inline __device__  float asinf(float x) { return asin(x); }
static inline __device__  float asinhf(float x) { return asinh(x); }
static inline __device__  float atan2f(float x, float y) { return atan2(x, y); }
static inline __device__  float atanf(float x) { return atan(x); }
static inline __device__  float atanhf(float x) { return atanh(x); }

static inline __device__  float cbrtf(float x) { return cbrt(x); }
static inline __device__  float ceilf(float x) { return ceil(x); }

static inline __device__  float copysignf(float x, float y) { return copysign(x, y); }
static inline __device__  float cosf(float x) { return cos(x); }
static inline __device__  float coshf(float x) { return cosh(x); }
static inline __device__  float cospif(float x) { return cospi(x); }

// OCML
//static inline __device__  float cyl_bessel_i0f(float x);
//static inline __device__  float cyl_bessel_i1f(float x);
//static inline __device__  float erfcinvf(float x);
//static inline __device__  float erfcxf(float x);
//static inline __device__  float erfinvf(float x);

static inline __device__  float erff(float x) { return erf(x); }
static inline __device__  float erfcf(float x) { return erfc(x); }

static inline __device__  float exp10f(float x) { return exp10(x); }
static inline __device__  float exp2f(float x) { return exp2(x); }
static inline __device__  float expf(float x) { return exp(x); }
static inline __device__  float expm1f(float x) { return expm1(x); }

static inline __device__  float fabsf(float x) { return fabs(x); }
static inline __device__  float fdimf(float x, float y) { return fdim(x, y); }
static inline __device__  float floorf(float x) { return floor(x); }
static inline __device__  float fmaf(float x, float y, float z) { return fma(x, y, z); }

static inline __device__  float fmaxf(float x, float y) { return fmax(x, y); }
static inline __device__  float fminf(float x, float y) { return fmin(x, y); }
static inline __device__  float fmodf(float x, float y) { return fmod(x, y); }

static inline __device__  float frexpf(float x, int* nptr) { return frexp(x, nptr); }
static inline __device__  float hypotf(float x, float y) { return hypot(x, y); }
static inline __device__  int ilogbf(float x) { return ilogb(x); }
static inline __device__  float ldexpf(float x, int exp) { return ldexp(x, exp); }

// OCML
//static inline __device__  float j0f(float x);
//static inline __device__  float j1f(float x);
//static inline __device__  float jnf(int n, float x);

static inline __device__  float log10f(float x) { return log10(x); }
static inline __device__  float log1pf(float x) { return log1p(x); }
static inline __device__  float log2f(float x) { return log2(x); }
static inline __device__  float logbf(float x) { return logb(x); }
static inline __device__  float logf(float x) { return log(x); }

static inline __device__  float modff(float x, float* iptr) { return modf(x, iptr); }
//static inline __device__  float nanf(const char* tagp) { return nan(0U); }
static inline __device__  float nearbyintf(float x) { return rint(x); }
static inline __device__  float nextafterf(float x, float y) { return nextafter(x, y); }

// OCML
//static inline __device__  float norm3df(float a, float b, float c);
//static inline __device__  float norm4df(float a, float b, float c, float d);
//static inline __device__  float normcdff(float x);
//static inline __device__  float normcdfinvf(float x);
//static inline __device__  float normf(int dim, const float* p);

static inline __device__  float powf(float x, float y) { return pow(x, y); }
static inline __device__  float remainderf(float x, float y) { return remainder(x, y); }
static inline __device__  float rsqrtf(float x) { return rsqrt(x); }
static inline __device__  float remquof(float x, float y, int* quo) { return remquo(x, y, quo); }

// OCML
//static inline __device__  float rcbrtf(float x);
//static inline __device__  float rhypotf(float x, float y);
//static inline __device__  float rnormf(int dim, const float* p);
//static inline __device__  float scalblnf(float x, long int n);
//static inline __device__  float scalbnf(float x, int n);

static inline __device__  int	signbit(float a) { return (int)(as_uint(a) >> 31); }

static inline __device__  void sincosf(float x, float* sptr, float* cptr) { *sptr = sincos(x, cptr); }

static inline __device__  void sincospif(float x, float* sptr, float* cptr) {
  *sptr = sinpi(x);
  *cptr = cospi(x);
}

static inline __device__  float sinf(float x) { return sin(x); }
static inline __device__  float sinhf(float x) { return sinh(x); }
static inline __device__  float sinpif(float x) { return sinpi(x); }
static inline __device__  float sqrtf(float x) { return sqrt(x); }
static inline __device__  float tanf(float x) { return tan(x); }
static inline __device__  float tanhf(float x) { return tanh(x); }
static inline __device__  float tgammaf(float x) { return tgamma(x); }
static inline __device__  float truncf(float x) { return trunc(x); }

// OCML
//static inline __device__  float y0f(float x);
//static inline __device__  float y1f(float x);
//static inline __device__  float ynf(int n, float x);

#endif // include guard
