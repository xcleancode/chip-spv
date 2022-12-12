/*
 * Copyright (c) 2022 CHIP-SPV developers
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
#include <hip/hip_runtime.h>
#include <cmath>

template <class T> __global__ void callStdAbsKernel(T *Data) {
  *Data = std::abs(*Data);
}

template <class T> __global__ void callStdExpm1Kernel(T *Data) {
  *Data = std::expm1(*Data);
}

template <class T> __global__ void callStdLog1pKernel(T *Data) {
  *Data = std::log1p(*Data);
}

template <class T> __global__ void callStdTruncKernel(T *Data) {
  *Data = std::trunc(*Data);
}

template <class T> __global__ void callStdSinKernel(T *Data) {
  *Data = std::sin(*Data);
}

template <class T> __global__ void callStdCosKernel(T *Data) {
  *Data = std::cos(*Data);
}

#define DEF_UNARY_OP_KERNEL(NAME, OP)                                          \
  template <class T> __global__ void NAME(T *Data) { *Data = OP(*Data); }

DEF_UNARY_OP_KERNEL(callStdAcosKernel, std::acos);
DEF_UNARY_OP_KERNEL(callStdAsinKernel, std::asin);
DEF_UNARY_OP_KERNEL(callStdAtanKernel, std::atan);
DEF_UNARY_OP_KERNEL(callStdCeilKernel, std::ceil);
DEF_UNARY_OP_KERNEL(callStdCeilfKernel, std::ceilf);
DEF_UNARY_OP_KERNEL(callStdFloorKernel, std::floor);
DEF_UNARY_OP_KERNEL(callStdFloorfKernel, std::floorf);
DEF_UNARY_OP_KERNEL(callStdLogKernel, std::log);
DEF_UNARY_OP_KERNEL(callStdLogfKernel, std::logf);
DEF_UNARY_OP_KERNEL(callStdLog2Kernel, std::log2);
DEF_UNARY_OP_KERNEL(callStdLog2fKernel, std::log2f);
DEF_UNARY_OP_KERNEL(callStdLog10Kernel, std::log10);
DEF_UNARY_OP_KERNEL(callStdLog10fKernel, std::log10f);
DEF_UNARY_OP_KERNEL(callStdErfKernel, std::erf);
DEF_UNARY_OP_KERNEL(callStdErffKernel, std::erff);
DEF_UNARY_OP_KERNEL(callStdErfcKernel, std::erfc);
DEF_UNARY_OP_KERNEL(callStdErfcfKernel, std::erfcf);
DEF_UNARY_OP_KERNEL(callStdCoshKernel, std::cosh);
DEF_UNARY_OP_KERNEL(callStdCoshfKernel, std::coshf);
DEF_UNARY_OP_KERNEL(callStdSinhKernel, std::sinh);
DEF_UNARY_OP_KERNEL(callStdSinhfKernel, std::sinhf);
DEF_UNARY_OP_KERNEL(callStdTanhKernel, std::tanh);
DEF_UNARY_OP_KERNEL(callStdTanhfKernel, std::tanhf);
DEF_UNARY_OP_KERNEL(callStdTanKernel, std::tan);
DEF_UNARY_OP_KERNEL(callStdTanfKernel, std::tanf);
DEF_UNARY_OP_KERNEL(callStdSqrtKernel, std::sqrt);
DEF_UNARY_OP_KERNEL(callStdSqrtfKernel, std::sqrtf);
DEF_UNARY_OP_KERNEL(callStdLgammaKernel, std::lgamma);
DEF_UNARY_OP_KERNEL(callStdLgammafKernel, std::lgammaf);
DEF_UNARY_OP_KERNEL(callStdNearbyintKernel, std::nearbyint);
DEF_UNARY_OP_KERNEL(callStdNearbyintfKernel, std::nearbyintf);

bool *b;
signed char *sc;
unsigned char *uc;
signed short *ss;
unsigned short *us;
signed int *si, *si2;
signed long *sl;
float *f, *f2;
double *d;

void testAbs() {
  callStdAbsKernel<<<1, 1>>>(sc);
  callStdAbsKernel<<<1, 1>>>(ss);
  callStdAbsKernel<<<1, 1>>>(si);
  callStdAbsKernel<<<1, 1>>>(sl);

  // std::abs may be used on unsigned types that are convertible to
  // int type (https://en.cppreference.com/w/cpp/numeric/math/abs).
  callStdAbsKernel<<<1, 1>>>(b);
  callStdAbsKernel<<<1, 1>>>(uc);
  callStdAbsKernel<<<1, 1>>>(us);

}

void testExpm1() {
  callStdExpm1Kernel<float><<<1, 1>>>(f);
  callStdExpm1Kernel<double><<<1, 1>>>(d);
}

void testLog1p() {
  callStdLog1pKernel<float><<<1, 1>>>(f);
  callStdLog1pKernel<double><<<1, 1>>>(d);
}

void testTrunc() {
  callStdTruncKernel<float><<<1, 1>>>(f);
  callStdTruncKernel<double><<<1, 1>>>(d);
}

void testSin() {
  callStdSinKernel<float><<<1, 1>>>(f);
  callStdSinKernel<double><<<1, 1>>>(d);
}

void testCos() {
  callStdCosKernel<float><<<1, 1>>>(f);
  callStdCosKernel<double><<<1, 1>>>(d);
}

void testAcos() {
  callStdAcosKernel<int><<<1, 1>>>(si);
  callStdAcosKernel<float><<<1, 1>>>(f);
  callStdAcosKernel<double><<<1, 1>>>(d);
}

void testAsin() {
  callStdAsinKernel<int><<<1, 1>>>(si);
  callStdAsinKernel<float><<<1, 1>>>(f);
  callStdAsinKernel<double><<<1, 1>>>(d);
}

void testAtan() {
  callStdAtanKernel<int><<<1, 1>>>(si);
  callStdAtanKernel<float><<<1, 1>>>(f);
  callStdAtanKernel<double><<<1, 1>>>(d);
}

void testCeil() {
  callStdCeilKernel<int><<<1, 1>>>(si);
  callStdCeilKernel<float><<<1, 1>>>(f);
  callStdCeilfKernel<float><<<1, 1>>>(f);
  callStdCeilKernel<double><<<1, 1>>>(d);
}

void testFloor() {
  callStdFloorKernel<int><<<1, 1>>>(si);
  callStdFloorKernel<float><<<1, 1>>>(f);
  callStdFloorfKernel<float><<<1, 1>>>(f);
  callStdFloorKernel<double><<<1, 1>>>(d);
}

void testLog() {
  callStdLogKernel<int><<<1, 1>>>(si);
  callStdLogKernel<float><<<1, 1>>>(f);
  callStdLogfKernel<float><<<1, 1>>>(f);
  callStdLogKernel<double><<<1, 1>>>(d);
}

void testLog2() {
  callStdLog2Kernel<int><<<1, 1>>>(si);
  callStdLog2Kernel<float><<<1, 1>>>(f);
  callStdLog2fKernel<float><<<1, 1>>>(f);
  callStdLog2Kernel<double><<<1, 1>>>(d);
}

void testLog10() {
  callStdLog10Kernel<int><<<1, 1>>>(si);
  callStdLog10Kernel<float><<<1, 1>>>(f);
  callStdLog10fKernel<float><<<1, 1>>>(f);
  callStdLog10Kernel<double><<<1, 1>>>(d);
}

void testErf() {
  callStdErfKernel<int><<<1, 1>>>(si);
  callStdErfKernel<float><<<1, 1>>>(f);
  callStdErffKernel<float><<<1, 1>>>(f);
  callStdErfKernel<double><<<1, 1>>>(d);
}

void testErfc() {
  callStdErfcKernel<int><<<1, 1>>>(si);
  callStdErfcKernel<float><<<1, 1>>>(f);
  callStdErfcfKernel<float><<<1, 1>>>(f);
  callStdErfcKernel<double><<<1, 1>>>(d);
}

void testCosh() {
  callStdCoshKernel<int><<<1, 1>>>(si);
  callStdCoshKernel<float><<<1, 1>>>(f);
  callStdCoshfKernel<float><<<1, 1>>>(f);
  callStdCoshKernel<double><<<1, 1>>>(d);
}

void testSinh() {
  callStdSinhKernel<int><<<1, 1>>>(si);
  callStdSinhKernel<float><<<1, 1>>>(f);
  callStdSinhfKernel<float><<<1, 1>>>(f);
  callStdSinhKernel<double><<<1, 1>>>(d);
}

void testTanh() {
  callStdTanhKernel<int><<<1, 1>>>(si);
  callStdTanhKernel<float><<<1, 1>>>(f);
  callStdTanhfKernel<float><<<1, 1>>>(f);
  callStdTanhKernel<double><<<1, 1>>>(d);
}

void testTan() {
  callStdTanKernel<int><<<1, 1>>>(si);
  callStdTanKernel<float><<<1, 1>>>(f);
  callStdTanfKernel<float><<<1, 1>>>(f);
  callStdTanKernel<double><<<1, 1>>>(d);
}

void testSqrt() {
  callStdSqrtKernel<int><<<1, 1>>>(si);
  callStdSqrtKernel<float><<<1, 1>>>(f);
  callStdSqrtfKernel<float><<<1, 1>>>(f);
  callStdSqrtKernel<double><<<1, 1>>>(d);
}

void testLgamma() {
  callStdLgammaKernel<int><<<1, 1>>>(si);
  callStdLgammaKernel<float><<<1, 1>>>(f);
  callStdLgammafKernel<float><<<1, 1>>>(f);
  callStdLgammaKernel<double><<<1, 1>>>(d);
}

void testNearbyint() {
  callStdNearbyintKernel<int><<<1, 1>>>(si);
  callStdNearbyintKernel<float><<<1, 1>>>(f);
  callStdNearbyintfKernel<float><<<1, 1>>>(f);
  callStdNearbyintKernel<double><<<1, 1>>>(d);
}
