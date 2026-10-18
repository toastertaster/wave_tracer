/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "math.hpp"

/*
 * Complex numbers (the operations of std::complex the kernels need).
 */

namespace wt::gpu::kernel {

struct complex {
    real re, im;
};

WT_GPU_FN complex operator+(complex a, complex b) { return { a.re+b.re, a.im+b.im }; }
WT_GPU_FN complex operator-(complex a, complex b) { return { a.re-b.re, a.im-b.im }; }
WT_GPU_FN complex operator-(complex a) { return { -a.re, -a.im }; }
WT_GPU_FN complex operator*(complex a, complex b) {
    return { a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re };
}
WT_GPU_FN complex operator*(complex a, real s) { return { a.re*s, a.im*s }; }
WT_GPU_FN complex operator*(real s, complex a) { return { s*a.re, s*a.im }; }
WT_GPU_FN complex operator/(complex a, complex b) {
    const real d = b.re*b.re + b.im*b.im;
    return { (a.re*b.re + a.im*b.im)/d, (a.im*b.re - a.re*b.im)/d };
}
WT_GPU_FN bool operator==(complex a, complex b) { return a.re==b.re && a.im==b.im; }

WT_GPU_FN complex conj(complex a) { return { a.re, -a.im }; }
/** @brief Squared magnitude (std::norm). */
WT_GPU_FN real norm(complex a) { return a.re*a.re + a.im*a.im; }

/** @brief Principal square root (std::sqrt). */
WT_GPU_FN complex sqrt(complex z) {
    const real r = sqrt(norm(z));
    if (r==real(0))
        return { 0,0 };
    const real t = sqrt((r + abs(z.re)) / 2);
    if (z.re>=real(0))
        return { t, z.im/(2*t) };
    return { abs(z.im)/(2*t), z.im<real(0) ? -t : t };
}

/** @brief exp(i·phase) */
WT_GPU_FN complex expi(real phase) { return { cos(phase), sin(phase) }; }

}
