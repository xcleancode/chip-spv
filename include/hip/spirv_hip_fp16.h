/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#error Include this file after including hip/hip_runtime.h
#endif

#ifndef HIP_INCLUDE_HIP_FP16_H
#define HIP_INCLUDE_HIP_FP16_H

#include <cassert>

#include <algorithm>
#include <type_traits>
#include <utility>

#if defined(__clang__)

#if defined(__HIP_DEVICE_COMPILE__)
typedef _Float16 _hip_f16;
typedef _Float16 _hip_f16_2 __attribute__((ext_vector_type(2)));
#else
typedef api_half _hip_f16;
typedef api_half2 _hip_f16_2;
#endif

struct __half_raw {
  union {
    static_assert(sizeof(_hip_f16) == sizeof(unsigned short), "");

    _hip_f16 data;
    unsigned short x;
  };
};

struct __half2_raw {
  union {
    static_assert(sizeof(_hip_f16_2) == sizeof(unsigned short[2]), "");

    _hip_f16_2 data;
    struct {
      unsigned short x;
      unsigned short y;
    };
  };
};

#if defined(__cplusplus)
#include "spirv_hip_vector_types.h"

namespace std {
template <>
struct is_floating_point<_hip_f16> : std::true_type {};
}  // namespace std

template <bool cond, typename T = void>
using Enable_if_t = typename std::enable_if<cond, T>::type;

// BEGIN STRUCT __HALF
struct __half {
 protected:
  union {
    static_assert(sizeof(_hip_f16) == sizeof(unsigned short), "");

    _hip_f16 data;
    unsigned short __x;
  };

 public:
  // CREATORS
  __host__ __device__ __half() = default;
  __host__ __device__ __half(const __half_raw& x) : data{x.data} {}
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  __host__ __device__ __half(decltype(data) x) : data{x} {}
  template <typename T, Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
  __host__ __device__ __half(T x) : data{static_cast<_hip_f16>(x)} {}
#endif
  __host__ __device__ __half(const __half&) = default;
  __host__ __device__ __half(__half&&) = default;
  __host__ __device__ ~__half() = default;

// CREATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
  __device__ __half(T x) : data{static_cast<_hip_f16>(x)} {}
#endif

  // MANIPULATORS
  __host__ __device__ __half& operator=(const __half&) = default;
  __host__ __device__ __half& operator=(__half&&) = default;
  __host__ __device__ __half& operator=(const __half_raw& x) {
    data = x.data;
    return *this;
  }
  __host__ __device__ volatile __half& operator=(const __half_raw& x) volatile {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(const volatile __half_raw& x) volatile {
    data = x.data;
    return *this;
  }
  __half& operator=(__half_raw&& x) {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(__half_raw&& x) volatile {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(volatile __half_raw&& x) volatile {
    data = x.data;
    return *this;
  }
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
  __host__ __device__ __half& operator=(T x) {
    data = static_cast<_hip_f16>(x);
    return *this;
  }
#endif

// MANIPULATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
  __device__ __half& operator=(T x) {
    data = static_cast<_hip_f16>(x);
    return *this;
  }
#endif

#if !defined(__HIP_NO_HALF_OPERATORS__)
  __device__ __half& operator+=(const __half& x) {
    data += x.data;
    return *this;
  }
  __device__ __half& operator-=(const __half& x) {
    data -= x.data;
    return *this;
  }
  __device__ __half& operator*=(const __half& x) {
    data *= x.data;
    return *this;
  }
  __device__ __half& operator/=(const __half& x) {
    data /= x.data;
    return *this;
  }
  __device__ __half& operator++() {
    ++data;
    return *this;
  }
  __device__ __half operator++(int) {
    __half tmp{*this};
    ++*this;
    return tmp;
  }
  __device__ __half& operator--() {
    --data;
    return *this;
  }
  __device__ __half operator--(int) {
    __half tmp{*this};
    --*this;
    return tmp;
  }
#endif

// ACCESSORS
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<std::is_floating_point<T>{} &&
                                    !std::is_same<T, double>{}>* = nullptr>
  operator T() const {
    return data;
  }
#endif
  __host__ __device__ operator __half_raw() const { return __half_raw{data}; }
  __host__ __device__ operator volatile __half_raw() const volatile {
    return __half_raw{data};
  }

// ACCESSORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
  __device__ operator T() const {
    return data;
  }
#endif

#if !defined(__HIP_NO_HALF_OPERATORS__)
  __device__ __half operator+() const { return *this; }
  __device__ __half operator-() const {
    __half tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
#endif

// FRIENDS
#if !defined(__HIP_NO_HALF_OPERATORS__)
  friend inline __device__ __half operator+(const __half& x, const __half& y) {
    return __half{x} += y;
  }
  friend inline __device__ __half operator-(const __half& x, const __half& y) {
    return __half{x} -= y;
  }
  friend inline __device__ __half operator*(const __half& x, const __half& y) {
    return __half{x} *= y;
  }
  friend inline __device__ __half operator/(const __half& x, const __half& y) {
    return __half{x} /= y;
  }
  friend inline __device__ bool operator==(const __half& x, const __half& y) {
    return x.data == y.data;
  }
  friend inline __device__ bool operator!=(const __half& x, const __half& y) {
    return !(x == y);
  }
  friend inline __device__ bool operator<(const __half& x, const __half& y) {
    return x.data < y.data;
  }
  friend inline __device__ bool operator>(const __half& x, const __half& y) {
    return y.data < x.data;
  }
  friend inline __device__ bool operator<=(const __half& x, const __half& y) {
    return !(y < x);
  }
  friend inline __device__ bool operator>=(const __half& x, const __half& y) {
    return !(x < y);
  }
#endif  // !defined(__HIP_NO_HALF_OPERATORS__)
};
// END STRUCT __HALF

// BEGIN STRUCT __HALF2
struct __half2 {
 protected:
  union {
    static_assert(sizeof(_hip_f16_2) == sizeof(unsigned short[2]), "");

    _hip_f16_2 data;
    struct {
      unsigned short x;
      unsigned short y;
    };
  };

 public:
  // CREATORS
  __host__ __device__ __half2() = default;
  __host__ __device__ __half2(const __half2_raw& x) : data{x.data} {}
  __host__ __device__ __half2(decltype(data) x) : data{x} {}
  __host__ __device__ __half2(const __half& x, const __half& y)
      : data{static_cast<__half_raw>(x).data, static_cast<__half_raw>(y).data} {
  }
  __host__ __device__ __half2(const __half2&) = default;
  __host__ __device__ __half2(__half2&&) = default;
  __host__ __device__ ~__half2() = default;

  // MANIPULATORS
  __host__ __device__ __half2& operator=(const __half2&) = default;
  __host__ __device__ __half2& operator=(__half2&&) = default;
  __host__ __device__ __half2& operator=(const __half2_raw& x) {
    data = x.data;
    return *this;
  }

// MANIPULATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_OPERATORS__)
  __device__ __half2& operator+=(const __half2& x) {
    data += x.data;
    return *this;
  }
  __device__ __half2& operator-=(const __half2& x) {
    data -= x.data;
    return *this;
  }
  __device__ __half2& operator*=(const __half2& x) {
    data *= x.data;
    return *this;
  }
  __device__ __half2& operator/=(const __half2& x) {
    data /= x.data;
    return *this;
  }
  __device__ __half2& operator++() { return *this += _hip_f16_2{1, 1}; }
  __device__ __half2 operator++(int) {
    __half2 tmp{*this};
    ++*this;
    return tmp;
  }
  __device__ __half2& operator--() { return *this -= _hip_f16_2{1, 1}; }
  __device__ __half2 operator--(int) {
    __half2 tmp{*this};
    --*this;
    return tmp;
  }
#endif

  // ACCESSORS
  __host__ __device__ operator decltype(data)() const { return data; }
  __host__ __device__ operator __half2_raw() const { return __half2_raw{data}; }

// ACCESSORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_OPERATORS__)
  __device__ __half2 operator+() const { return *this; }
  __device__ __half2 operator-() const {
    __half2 tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
#endif

// FRIENDS
#if !defined(__HIP_NO_HALF_OPERATORS__)
  friend inline __device__ __half2 operator+(const __half2& x,
                                             const __half2& y) {
    return __half2{x} += y;
  }
  friend inline __device__ __half2 operator-(const __half2& x,
                                             const __half2& y) {
    return __half2{x} -= y;
  }
  friend inline __device__ __half2 operator*(const __half2& x,
                                             const __half2& y) {
    return __half2{x} *= y;
  }
  friend inline __device__ __half2 operator/(const __half2& x,
                                             const __half2& y) {
    return __half2{x} /= y;
  }
  friend inline __device__ bool operator==(const __half2& x, const __half2& y) {
    auto r = x.data == y.data;
    return r.x != 0 && r.y != 0;
  }
  friend inline __device__ bool operator!=(const __half2& x, const __half2& y) {
    return !(x == y);
  }
  friend inline __device__ bool operator<(const __half2& x, const __half2& y) {
    auto r = x.data < y.data;
    return r.x != 0 && r.y != 0;
  }
  friend inline __device__ bool operator>(const __half2& x, const __half2& y) {
    return y < x;
  }
  friend inline __device__ bool operator<=(const __half2& x, const __half2& y) {
    return !(y < x);
  }
  friend inline __device__ bool operator>=(const __half2& x, const __half2& y) {
    return !(x < y);
  }
#endif  // !defined(__HIP_NO_HALF_OPERATORS__)
};
// END STRUCT __HALF2

namespace {
inline __host__ __device__ __half2 make_half2(__half x, __half y) {
  return __half2{x, y};
}

inline __device__ __half __low2half(__half2 x) {
  return __half{__half_raw{static_cast<__half2_raw>(x).data.x}};
}

inline __device__ __half __high2half(__half2 x) {
  return __half{__half_raw{static_cast<__half2_raw>(x).data.y}};
}

inline __device__ __half2 __half2half2(__half x) { return __half2{x, x}; }

inline __device__ __half2 __halves2half2(__half x, __half y) {
  return __half2{x, y};
}

inline __device__ __half2 __low2half2(__half2 x) {
  return __half2{_hip_f16_2{static_cast<__half2_raw>(x).data.x,
                            static_cast<__half2_raw>(x).data.x}};
}

inline __device__ __half2 __high2half2(__half2 x) {
  return __half2_raw{_hip_f16_2{static_cast<__half2_raw>(x).data.y,
                                static_cast<__half2_raw>(x).data.y}};
}

inline __device__ __half2 __lows2half2(__half2 x, __half2 y) {
  return __half2_raw{_hip_f16_2{static_cast<__half2_raw>(x).data.x,
                                static_cast<__half2_raw>(y).data.x}};
}

inline __device__ __half2 __highs2half2(__half2 x, __half2 y) {
  return __half2_raw{_hip_f16_2{static_cast<__half2_raw>(x).data.y,
                                static_cast<__half2_raw>(y).data.y}};
}

inline __device__ __half2 __lowhigh2highlow(__half2 x) {
  return __half2_raw{_hip_f16_2{static_cast<__half2_raw>(x).data.y,
                                static_cast<__half2_raw>(x).data.x}};
}

// Bitcasts
inline __device__ short __half_as_short(__half x) {
  return static_cast<__half_raw>(x).x;
}

inline __device__ unsigned short __half_as_ushort(__half x) {
  return static_cast<__half_raw>(x).x;
}

inline __device__ __half __short_as_half(short x) {
  __half_raw r;
  r.x = x;
  return r;
}

inline __device__ __half __ushort_as_half(unsigned short x) {
  __half_raw r;
  r.x = x;
  return r;
}

// TODO: rounding behaviour is not correct.
// float -> half | half2
inline __device__ __host__ __half __float2half(float x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __host__ __half __float2half_rn(float x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __host__ __half __float2half_rz(float x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __host__ __half __float2half_rd(float x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __host__ __half __float2half_ru(float x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __host__ __half2 __float2half2_rn(float x) {
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(x), static_cast<_hip_f16>(x)}};
}
inline __device__ __host__ __half2 __floats2half2_rn(float x, float y) {
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(x), static_cast<_hip_f16>(y)}};
}
inline __device__ __host__ __half2 __float22half2_rn(float2 x) {
  return __floats2half2_rn(x.x, x.y);
}

// half | half2 -> float
inline __device__ __host__ float __half2float(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ __host__ float __low2float(__half2 x) {
  return static_cast<__half2_raw>(x).data.x;
}
inline __device__ __host__ float __high2float(__half2 x) {
  return static_cast<__half2_raw>(x).data.y;
}
inline __device__ __host__ float2 __half22float2(__half2 x) {
  return make_float2(static_cast<__half2_raw>(x).data.x,
                     static_cast<__half2_raw>(x).data.y);
}

// half -> int
inline __device__ int __half2int_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ int __half2int_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ int __half2int_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ int __half2int_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// int -> half
inline __device__ __half __int2half_rn(int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __int2half_rz(int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __int2half_rd(int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __int2half_ru(int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// half -> short
inline __device__ short __half2short_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ short __half2short_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ short __half2short_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ short __half2short_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// short -> half
inline __device__ __half __short2half_rn(short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __short2half_rz(short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __short2half_rd(short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __short2half_ru(short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// half -> long long
inline __device__ long long __half2ll_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ long long __half2ll_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ long long __half2ll_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ long long __half2ll_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// long long -> half
inline __device__ __half __ll2half_rn(long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ll2half_rz(long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ll2half_rd(long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ll2half_ru(long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// half -> unsigned int
inline __device__ unsigned int __half2uint_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned int __half2uint_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned int __half2uint_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned int __half2uint_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// unsigned int -> half
inline __device__ __half __uint2half_rn(unsigned int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __uint2half_rz(unsigned int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __uint2half_rd(unsigned int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __uint2half_ru(unsigned int x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// half -> unsigned short
inline __device__ unsigned short __half2ushort_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// unsigned short -> half
inline __device__ __half __ushort2half_rn(unsigned short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ushort2half_rz(unsigned short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ushort2half_rd(unsigned short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ushort2half_ru(unsigned short x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// half -> unsigned long long
inline __device__ unsigned long long __half2ull_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// unsigned long long -> half
inline __device__ __half __ull2half_rn(unsigned long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ull2half_rz(unsigned long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ull2half_rd(unsigned long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}
inline __device__ __half __ull2half_ru(unsigned long long x) {
  return __half_raw{static_cast<_hip_f16>(x)};
}

// Load primitives
inline __device__ __half __ldg(const __half* ptr) { return *ptr; }
inline __device__ __half __ldcg(const __half* ptr) { return *ptr; }
inline __device__ __half __ldca(const __half* ptr) { return *ptr; }
inline __device__ __half __ldcs(const __half* ptr) { return *ptr; }

inline __device__ __half2 __ldg(const __half2* ptr) { return *ptr; }
inline __device__ __half2 __ldcg(const __half2* ptr) { return *ptr; }
inline __device__ __half2 __ldca(const __half2* ptr) { return *ptr; }
inline __device__ __half2 __ldcs(const __half2* ptr) { return *ptr; }

// Relations
inline __device__ bool __heq(__half x, __half y) {
  return static_cast<__half_raw>(x).data == static_cast<__half_raw>(y).data;
}
inline __device__ bool __hne(__half x, __half y) {
  return static_cast<__half_raw>(x).data != static_cast<__half_raw>(y).data;
}
inline __device__ bool __hle(__half x, __half y) {
  return static_cast<__half_raw>(x).data <= static_cast<__half_raw>(y).data;
}
inline __device__ bool __hge(__half x, __half y) {
  return static_cast<__half_raw>(x).data >= static_cast<__half_raw>(y).data;
}
inline __device__ bool __hlt(__half x, __half y) {
  return static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data;
}
inline __device__ bool __hgt(__half x, __half y) {
  return static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data;
}
inline __device__ bool __hequ(__half x, __half y) { return __heq(x, y); }
inline __device__ bool __hneu(__half x, __half y) { return __hne(x, y); }
inline __device__ bool __hleu(__half x, __half y) { return __hle(x, y); }
inline __device__ bool __hgeu(__half x, __half y) { return __hge(x, y); }
inline __device__ bool __hltu(__half x, __half y) { return __hlt(x, y); }
inline __device__ bool __hgtu(__half x, __half y) { return __hgt(x, y); }

inline __device__ __half2 __heq2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data == static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hne2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data != static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hle2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data <= static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hge2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data >= static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hlt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hgt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data;
  return __half2_raw{
      _hip_f16_2{static_cast<_hip_f16>(r.x), static_cast<_hip_f16>(r.y)}};
}
inline __device__ __half2 __hequ2(__half2 x, __half2 y) { return __heq2(x, y); }
inline __device__ __half2 __hneu2(__half2 x, __half2 y) { return __hne2(x, y); }
inline __device__ __half2 __hleu2(__half2 x, __half2 y) { return __hle2(x, y); }
inline __device__ __half2 __hgeu2(__half2 x, __half2 y) { return __hge2(x, y); }
inline __device__ __half2 __hltu2(__half2 x, __half2 y) { return __hlt2(x, y); }
inline __device__ __half2 __hgtu2(__half2 x, __half2 y) { return __hgt2(x, y); }

inline __device__ bool __hbeq2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__heq2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hbne2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hne2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hble2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hle2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hbge2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hge2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hblt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hlt2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hbgt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hgt2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __device__ bool __hbequ2(__half2 x, __half2 y) { return __hbeq2(x, y); }
inline __device__ bool __hbneu2(__half2 x, __half2 y) { return __hbne2(x, y); }
inline __device__ bool __hbleu2(__half2 x, __half2 y) { return __hble2(x, y); }
inline __device__ bool __hbgeu2(__half2 x, __half2 y) { return __hbge2(x, y); }
inline __device__ bool __hbltu2(__half2 x, __half2 y) { return __hblt2(x, y); }
inline __device__ bool __hbgtu2(__half2 x, __half2 y) { return __hbgt2(x, y); }

// Arithmetic
}  // Anonymous namespace.

#if !defined(HIP_NO_HALF)
using half = __half;
using half2 = __half2;
#endif
#endif  // defined(__cplusplus)

#endif  // defined(__clang__)

#endif
