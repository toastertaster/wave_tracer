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
 * BSDFs here have Mueller operators of a block form (`mueller_block`):
 * Fresnel interactions (`mueller_operator_t::fresnel`) and depolarizers.
 *
 * Forward transport carries a Stokes parameters vector. Backward transport
 * carries a Mueller operator (`beam_radiometric_data_t<backward>`), composed
 * with each interaction's to the right; a scalar film observes only its first
 * row (times its scale), which therefore composes on its own: the GPU keeps
 * just that row (`row_*` functions).
 */

namespace wt::gpu::kernel {

/** @brief Stokes parameters vector (`stokes_parameters_t`). */
struct stokes {
    real s0, s1, s2, s3;

    WT_GPU_FN stokes operator*(real s) const { return { s0*s, s1*s, s2*s, s3*s }; }
    WT_GPU_FN stokes operator+(stokes o) const { return { s0+o.s0, s1+o.s1, s2+o.s2, s3+o.s3 }; }
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
 * @brief A Mueller operator of the block form [[m00,m01,0,0], [m01,m11,0,0],
 *        [0,0,m22,m23], [0,0,-m23,m22]]: Fresnel (`mueller_operator_t::fresnel`,
 *        m11=m00) or a scaled depolarizer (`perfect_depolarizer`, m00 only).
 */
struct mueller_block {
    real m00, m01, m11, m22, m23;

    WT_GPU_FN mueller_block operator*(real s) const { return { m00*s, m01*s, m11*s, m22*s, m23*s }; }
    WT_GPU_FN mueller_block operator/(real s) const { return { m00/s, m01/s, m11/s, m22/s, m23/s }; }

    /** @brief `mueller_operator_t::operator*(stokes)`, frames assumed to match. */
    WT_GPU_FN stokes operator*(stokes S) const {
        return {
            fma(m01, S.s1, m00*S.s0),
            fma(m11, S.s1, m01*S.s0),
            fma(m23, S.s3, m22*S.s2),
            fma(m22, S.s3, -m23*S.s2),
        };
    }
    /** @brief The row vector `r` times this. */
    WT_GPU_FN stokes row_mul(stokes r) const {
        return {
            fma(r.s1, m01, r.s0*m00),
            fma(r.s1, m11, r.s0*m01),
            fma(-r.s3, m23, r.s2*m22),
            fma(r.s3, m22, r.s2*m23),
        };
    }

    /** @brief `mueller_operator_t::perfect_depolarizer()` times `d` */
    WT_GPU_FN static mueller_block depolarizer(real d) { return { d, 0, 0, 0, 0 }; }
};

/** @brief `frame_t::handness` */
WT_GPU_FN real handness(frame f) {
    return dot(cross(f.n, f.t), f.b)>real(0) ? real(1) : real(-1);
}

/**
 * @brief The row vector `r` times `mueller_operator_t::rotation(t, (1,0))`,
 *        whose middle block rotates by twice the angle of `t`.
 */
WT_GPU_FN stokes row_rotate(stokes r, vec2 t) {
    // util::rotation_matrix(t, (1,0)) = [[tx,ty],[-ty,tx]], squared, transposed
    const real c2 = t.x*t.x - t.y*t.y;
    const real s2 = 2*t.x*t.y;
    return { r.s0, fma(r.s2, s2, r.s1*c2), fma(r.s2, c2, -r.s1*s2), r.s3 };
}

/** @brief `mueller_operator_t::change_incident_frame(old_f, new_f)` of the row vector `r`. */
WT_GPU_FN stokes row_change_incident_frame(stokes r, frame old_f, frame new_f) {
    stokes ret = row_rotate(r, vec2{ dot(new_f.t, old_f.t), dot(new_f.t, old_f.b) });
    // R = rotation * handness_flip
    if (handness(old_f)!=handness(new_f)) {
        ret.s2 = -ret.s2;
        ret.s3 = -ret.s3;
    }
    return ret;
}

/** @brief `compose(M1, M2, M1in, M2out)` of the row vector `r` (M1's first row). */
WT_GPU_FN stokes row_compose(stokes r, mueller_block M2, frame M1in, frame M2out) {
    // R = handness_flip * rotation
    if (handness(M1in)!=handness(M2out)) {
        r.s2 = -r.s2;
        r.s3 = -r.s3;
    }
    r = row_rotate(r, vec2{ dot(M2out.t, M1in.t), dot(M2out.t, M1in.b) });
    return M2.row_mul(r);
}

/** @brief `mueller_operator_t::fresnel(fs, fp)` */
WT_GPU_FN mueller_block mueller_fresnel(complex fs, complex fp) {
    const real Rs = norm(fs);
    const real Rp = norm(fp);
    const complex x = fp*conj(fs);
    const real m00 = (Rs+Rp)/2;
    return { m00, (Rs-Rp)/2, m00, x.re, x.im };
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
WT_GPU_FN mueller_block mueller_fresnel(complex eta_12, bool reflection, vec3 w) {
    if (reflection) {
        complex rs, rp;
        fresnel_reflection(eta_12, w, rs, rp);
        return mueller_fresnel(rs, rp);
    }
    const fresnel_t f = fresnel(eta_12, w);
    return mueller_fresnel(complex{ f.ts,0 }, complex{ f.tp,0 }) * f.Z;
}

}
