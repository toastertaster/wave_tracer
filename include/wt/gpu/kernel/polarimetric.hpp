/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "complex.hpp"
#include "shapes.hpp"

/*
 * Polarimetry and Fresnel equations, ported from
 * wt/interaction/polarimetric/{stokes,mueller}.hpp, wt/interaction/fresnel.hpp
 * and wt/math/rotation.hpp.
 *
 * Forward transport only applies Mueller operators of the Fresnel form (see
 * `mueller_operator_t::fresnel`), so `fresnel_mueller` stores just its four
 * distinct elements.
 */

namespace wt::gpu::kernel {

/** @brief Stokes parameters vector (`stokes_parameters_t`). */
struct stokes {
    real s0, s1, s2, s3;
};

/**
 * @brief `stokes_parameters_t::reorient`: re-expresses `S`, given in frame
 *        `cur`, in frame `nw` (both frames share their normal, up to sign).
 */
WT_GPU_FN stokes reorient(stokes S, frame cur, frame nw) {
    // align tangent direction
    const vec2 tou = { dot(nw.t, cur.t), dot(nw.t, cur.b) };
    const vec2 tov = { dot(nw.b, cur.t), dot(nw.b, cur.b) };

    // util::rotation_matrix(dir2_t{ 1,0 }, tou): [[c,-s],[s,c]]
    const real c = eft::sum_prod(1, tou.x, 0, tou.y);
    const real s = eft::diff_prod(1, tou.y, tou.x, 0);
    const real ns = eft::diff_prod(tou.x, 0, 1, tou.y);
    const auto rotate = [&](vec2 v) { return vec2{ c*v.x + ns*v.y, s*v.x + c*v.y }; };

    const vec2 S12 = rotate(rotate(vec2{ S.s1, S.s2 }));
    const vec2 v = rotate(vec2{ 0,1 });
    // handness change?
    if (dot(v, tov)<real(0))
        return { S.s0, S12.x, -S12.y, -S.s3 };
    return { S.s0, S12.x, S12.y, S.s3 };
}

/**
 * @brief A Mueller operator of the Fresnel form (`mueller_operator_t::fresnel`):
 *        [[m00,m01,0,0], [m01,m00,0,0], [0,0,m22,m23], [0,0,-m23,m22]].
 */
struct fresnel_mueller {
    real m00, m01, m22, m23;

    WT_GPU_FN fresnel_mueller operator*(real s) const { return { m00*s, m01*s, m22*s, m23*s }; }
    WT_GPU_FN fresnel_mueller operator/(real s) const { return { m00/s, m01/s, m22/s, m23/s }; }

    /** @brief `mueller_operator_t::operator*(stokes)`, frames assumed to match. */
    WT_GPU_FN stokes operator*(stokes S) const {
        return {
            fma(m01, S.s1, m00*S.s0),
            fma(m00, S.s1, m01*S.s0),
            fma(m23, S.s3, m22*S.s2),
            fma(m22, S.s3, -m23*S.s2),
        };
    }
};

/** @brief `mueller_operator_t::fresnel(fs, fp)` */
WT_GPU_FN fresnel_mueller mueller_fresnel(complex fs, complex fp) {
    const real Rs = norm(fs);
    const real Rp = norm(fp);
    const complex x = fp*conj(fs);
    return { (Rs+Rp)/2, (Rs-Rp)/2, x.re, x.im };
}

/** @brief `fresnel_ret_t`. The CPU's complex `eta_12` and `rs,rp,ts,tp` are real here (see `fresnel`). */
struct fresnel_t {
    vec3 t;
    real eta_12;
    real Z;
    real rs, rp;
    real ts, tp;
    real Ts, Tp;
};

/**
 * @brief `wt::fresnel(eta_12, w)` (normal (0,0,1), `w` pointing away from the
 *        surface). The CPU discards the imaginary part of `eta_12` after
 *        refraction, so all results are real.
 */
WT_GPU_FN fresnel_t fresnel(complex eta_12, vec3 w) {
    if (eta_12==complex{ 1,0 })
        return { -w, 1, 1, 0,0, 1,1, 1,1 };

    const real abs_cosi = abs(w.z);

    // refract(real(eta_12), w)
    const real wn = w.z;
    const real eta = wn>real(0) ? eta_12.re : real(1)/eta_12.re;
    const real cost2 = 1 - sqr(eta) * (1-sqr(wn));

    if (abs_cosi==real(0) || !(cost2>=real(0)))
        return { { 0,0,1 }, eta, 1, 1,1, 0,0, 0,0 };

    const real cost = sqrt(cost2);
    const vec3 t = normalize(vec3{ -eta*w.x, -eta*w.y, eta*(wn-w.z) } - cost*vec3{ 0,0,wn>=real(0) ? real(1) : real(-1) });

    const real rs = (eta * abs_cosi - cost) / (eta * abs_cosi + cost);
    const real rp = (abs_cosi - eta * cost) / (abs_cosi + eta * cost);
    const real ts = rs + 1;
    const real tp = (rp + 1) * eta;
    const real Z = abs(cost/(eta*abs_cosi));

    return { t, eta, Z, rs, rp, ts, tp, min(real(1), Z*sqr(ts)), min(real(1), Z*sqr(tp)) };
}

/** @brief `wt::fresnel_reflection(eta_12, w)` (normal (0,0,1)): conductor reflection coefficients rs, rp. */
WT_GPU_FN void fresnel_reflection(complex eta_12, vec3 w,
                                  WT_GPU_THREAD complex& rs, WT_GPU_THREAD complex& rp) {
    const real wn = w.z;
    if (eta_12==complex{ 1,0 } || wn<real(0)) {
        rs = rp = complex{ 0,0 };
        return;
    }

    const complex t2 = complex{ 1,0 } - (1-sqr(wn)) * (eta_12*eta_12);
    const complex t = sqrt(t2);
    const complex i = { wn,0 };

    rs = (eta_12*i - t) / (eta_12*i + t);
    rp = (i - eta_12*t) / (i + eta_12*t);
}

/** @brief `mueller_operator_t::fresnel(eta_12, reflection, w)` */
WT_GPU_FN fresnel_mueller mueller_fresnel(complex eta_12, bool reflection, vec3 w) {
    if (reflection) {
        complex rs, rp;
        fresnel_reflection(eta_12, w, rs, rp);
        return mueller_fresnel(rs, rp);
    }
    const fresnel_t f = fresnel(eta_12, w);
    return mueller_fresnel(complex{ f.ts,0 }, complex{ f.tp,0 }) * f.Z;
}

}
