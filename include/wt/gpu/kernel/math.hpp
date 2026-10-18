/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "config.hpp"

/*
 * Scalar and vector math for the shared GPU kernels.
 *
 * The operations reproduce the CPU code's arithmetic (glm and wt::m), including
 * operand order, so kernels compute the same values as the CPU up to rounding:
 *   - min(a,b) is (b<a ? b : a) and max(a,b) is (a<b ? b : a), as glm's, which
 *     decides what NaN operands produce;
 *   - dot() is an FMA chain and cross() uses eft::diff_prod, as wt::m::dot and
 *     wt::m::cross (wt/math/vecmath.hpp) and their SIMD variants;
 *   - `eft::` are the error-free transformations of wt/math/eft/eft.hpp. They
 *     need IEEE-conforming arithmetic: never compile kernels with fast math.
 */

namespace wt::gpu::kernel {

WT_GPU_FN float  k_sqrt(float x)  { return sqrtf(x); }
WT_GPU_FN double k_sqrt(double x) { return sqrt(x); }
WT_GPU_FN float  k_fma(float a, float b, float c)    { return fmaf(a,b,c); }
WT_GPU_FN double k_fma(double a, double b, double c) { return fma(a,b,c); }
WT_GPU_FN float  k_abs(float x)  { return fabsf(x); }
WT_GPU_FN double k_abs(double x) { return fabs(x); }
WT_GPU_FN float  k_sin(float x)  { return sinf(x); }
WT_GPU_FN double k_sin(double x) { return sin(x); }
WT_GPU_FN float  k_cos(float x)  { return cosf(x); }
WT_GPU_FN double k_cos(double x) { return cos(x); }
WT_GPU_FN float  k_erf(float x)  { return erff(x); }
WT_GPU_FN double k_erf(double x) { return erf(x); }
WT_GPU_FN float  k_tan(float x)  { return tanf(x); }
WT_GPU_FN double k_tan(double x) { return tan(x); }
WT_GPU_FN float  k_atan2(float y, float x)  { return atan2f(y,x); }
WT_GPU_FN double k_atan2(double y, double x) { return atan2(y,x); }
WT_GPU_FN float  k_exp(float x)  { return expf(x); }
WT_GPU_FN double k_exp(double x) { return exp(x); }
WT_GPU_FN float  k_log(float x)  { return logf(x); }
WT_GPU_FN double k_log(double x) { return log(x); }
WT_GPU_FN float  k_floor(float x)  { return floorf(x); }
WT_GPU_FN double k_floor(double x) { return floor(x); }
WT_GPU_FN float  k_round(float x)  { return roundf(x); }
WT_GPU_FN double k_round(double x) { return round(x); }

WT_GPU_FN real inf() {
#if defined(__METAL_VERSION__)
    return INFINITY;
#elif defined(WT_GPU_DBL_PRECISION)
    return __builtin_huge_val();
#else
    return __builtin_huge_valf();
#endif
}

WT_GPU_FN real sqrt(real x) { return k_sqrt(x); }
WT_GPU_FN real fma(real a, real b, real c) { return k_fma(a,b,c); }
WT_GPU_FN real abs(real x) { return k_abs(x); }
WT_GPU_FN real sqr(real x) { return x*x; }
WT_GPU_FN real sin(real x) { return k_sin(x); }
WT_GPU_FN real cos(real x) { return k_cos(x); }
WT_GPU_FN real erf(real x) { return k_erf(x); }
WT_GPU_FN real tan(real x) { return k_tan(x); }
WT_GPU_FN real atan2(real y, real x) { return k_atan2(y,x); }
WT_GPU_FN real exp(real x) { return k_exp(x); }
WT_GPU_FN real log(real x) { return k_log(x); }
WT_GPU_FN real floor(real x) { return k_floor(x); }
/** @brief Rounds half away from zero (glm::round, std::round). */
WT_GPU_FN real round(real x) { return k_round(x); }
/** @brief glm::mod: x - y*floor(x/y). */
WT_GPU_FN real mod(real x, real y) { return x - y*floor(x/y); }

constexpr real pi = real(3.14159265358979323846);
constexpr real two_pi = real(6.28318530717958647692);
constexpr real pi_2 = real(1.57079632679489661923);
constexpr real pi_4 = real(0.78539816339744830962);
constexpr real inv_pi = real(0.31830988618379067154);
constexpr real inv_two_pi = real(0.15915494309189533577);
constexpr real inv_sqrt_two = real(0.70710678118654752440);
constexpr real inv_sqrt_two_pi = real(0.39894228040143267794);
constexpr real sqrt_pi_2 = real(1.25331413731550025121);

WT_GPU_FN bool isnan(real x) { return x!=x; }
WT_GPU_FN bool isfinite(real x) { return abs(x)<inf(); }
// Sign bit of a non-NaN value, including for -0 and -inf.
WT_GPU_FN bool signbit(real x) { return x<real(0) || (x==real(0) && real(1)/x<real(0)); }

// glm's operand order (see above).
template <typename T> WT_GPU_FN T min(T a, T b) { return b<a ? b : a; }
template <typename T> WT_GPU_FN T max(T a, T b) { return a<b ? b : a; }
template <typename T> WT_GPU_FN T min(T a, T b, T c) { return min(min(a,b),c); }
template <typename T> WT_GPU_FN T max(T a, T b, T c) { return max(max(a,b),c); }
template <typename T> WT_GPU_FN T clamp(T x, T lo, T hi) { return min(max(x,lo),hi); }
template <typename T> WT_GPU_FN void swap(WT_GPU_THREAD T& a, WT_GPU_THREAD T& b) {
    const T t = a;
    a = b;
    b = t;
}


struct vec2 {
    real x, y;
};

struct vec3 {
    real x, y, z;

    WT_GPU_FN real operator[](int i) const { return i==0 ? x : i==1 ? y : z; }
};

WT_GPU_FN vec3 operator+(vec3 a, vec3 b) { return { a.x+b.x, a.y+b.y, a.z+b.z }; }
WT_GPU_FN vec3 operator-(vec3 a, vec3 b) { return { a.x-b.x, a.y-b.y, a.z-b.z }; }
WT_GPU_FN vec3 operator-(vec3 a) { return { -a.x, -a.y, -a.z }; }
WT_GPU_FN vec3 operator*(vec3 a, vec3 b) { return { a.x*b.x, a.y*b.y, a.z*b.z }; }
WT_GPU_FN vec3 operator*(vec3 a, real s) { return { a.x*s, a.y*s, a.z*s }; }
WT_GPU_FN vec3 operator*(real s, vec3 a) { return { s*a.x, s*a.y, s*a.z }; }
WT_GPU_FN vec3 operator/(vec3 a, real s) { return { a.x/s, a.y/s, a.z/s }; }
WT_GPU_FN vec3 abs(vec3 a) { return { abs(a.x), abs(a.y), abs(a.z) }; }
WT_GPU_FN real max_element(vec3 a) { return max(a.x, a.y, a.z); }

WT_GPU_FN vec2 operator+(vec2 a, vec2 b) { return { a.x+b.x, a.y+b.y }; }
WT_GPU_FN vec2 operator-(vec2 a, vec2 b) { return { a.x-b.x, a.y-b.y }; }
WT_GPU_FN vec2 operator*(vec2 a, vec2 b) { return { a.x*b.x, a.y*b.y }; }
WT_GPU_FN vec2 operator*(vec2 a, real s) { return { a.x*s, a.y*s }; }
WT_GPU_FN vec2 operator*(real s, vec2 a) { return { s*a.x, s*a.y }; }
WT_GPU_FN vec2 operator/(vec2 a, real s) { return { a.x/s, a.y/s }; }
WT_GPU_FN vec2 operator-(vec2 a) { return { -a.x, -a.y }; }

WT_GPU_FN bool operator==(vec3 a, vec3 b) { return a.x==b.x && a.y==b.y && a.z==b.z; }


/*
 * Error-free transformations (wt/math/eft/eft.hpp)
 */
namespace eft {

WT_GPU_FN real two_prod(WT_GPU_THREAD real& err, real a, real b) {
    const real prod = a*b;
    err = fma(a, b, -prod);
    return prod;
}
WT_GPU_FN real two_sum(WT_GPU_THREAD real& err, real a, real b) {
    const real sum = a+b;
    const real e1 = sum-a;
    const real e2 = sum-e1;
    err = (b-e1) + (a-e2);
    return sum;
}
// a*b - c*d
WT_GPU_FN real diff_prod(real a, real b, real c, real d) {
    const real cd = c*d;
    const real ret = fma(a,b,-cd);
    return ret + fma(-c,d,cd);
}
// a*b + c*d
WT_GPU_FN real sum_prod(real a, real b, real c, real d) { return diff_prod(a,b, -c,d); }
// Compensated dot product (wt::m::eft::dot).
WT_GPU_FN real dot(vec3 a, vec3 b) {
    real d = 0, err = 0, err1, err2;
    for (int i=0;i<3;++i) {
        const real t = two_prod(err1, a[i], b[i]);
        d = two_sum(err2, d, t);
        err = err + err1 + err2;
    }
    return d + err;
}

}


// wt::m::dot: x*x', then FMA-accumulate the remaining products.
WT_GPU_FN real dot(vec3 a, vec3 b) { return fma(a.z, b.z, fma(a.y, b.y, a.x*b.x)); }
WT_GPU_FN real dot(vec2 a, vec2 b) { return fma(a.y, b.y, a.x*b.x); }
// wt::m::cross
WT_GPU_FN vec3 cross(vec3 a, vec3 b) {
    return {
        eft::diff_prod(a.y,b.z, a.z,b.y),
        eft::diff_prod(a.z,b.x, a.x,b.z),
        eft::diff_prod(a.x,b.y, a.y,b.x),
    };
}
WT_GPU_FN real length2(vec3 a) { return dot(a,a); }
WT_GPU_FN real length2(vec2 a) { return dot(a,a); }
WT_GPU_FN real length(vec3 a) { return sqrt(length2(a)); }
// wt::m::normalize: v/length(v)
WT_GPU_FN vec3 normalize(vec3 a) { return a / length(a); }
WT_GPU_FN vec2 normalize(vec2 a) { return a / sqrt(length2(a)); }

// glm::sign
WT_GPU_FN real sign(real x) { return real(real(0)<x) - real(x<real(0)); }

}
