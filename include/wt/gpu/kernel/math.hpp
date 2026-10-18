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

// glm::sign
WT_GPU_FN real sign(real x) { return real(real(0)<x) - real(x<real(0)); }

}
