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

bool *b;
signed char *sc;
unsigned char *uc;
signed short *ss;
unsigned short *us;
signed int *si;
signed long *sl;
float *f;
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
