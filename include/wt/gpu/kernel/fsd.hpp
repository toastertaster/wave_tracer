/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "beam.hpp"
#include "rng.hpp"
#include "scene.hpp"

/*
 * Free-space diffraction (FSD) by the uniform theory of diffraction (UTD),
 * ported from wt/interaction/fsd/{common,utd,free_space_diffraction}.hpp,
 * src/interaction/fsd/free_space_diffraction.cpp and
 * wt/math/intersect/misc.hpp.
 *
 * An aperture (`free_space_diffraction_t`) is the set of lit edges around an
 * interaction region. Its entries are kept compactly (`aperture_entry`) in
 * per-path scratch memory and expanded to wedges (`utd::wedge_edge_t`) on use.
 */

namespace wt::gpu::kernel {

/** @brief `utd::utd_min_sin_beta` */
constexpr real utd_min_sin_beta = real(1e-3);
/** @brief `utd_IS_sigma_scale` of src/interaction/fsd/free_space_diffraction.cpp */
constexpr real utd_IS_sigma_scale = 45;

/** @brief An aperture edge: `ads_edge | front_is_face1<<31`, and the part `[t1,t2]` of it in the region. */
struct aperture_entry {
    u32 edge_and_face;
    real t1, t2;
};

/** @brief `utd::wedge_edge_t` */
struct wedge {
    vec3 v;             // mid point
    real l;             // length
    vec3 nff, tff;      // front-face normal, and tangent pointing into the wedge
    vec3 nbf;           // back-face normal
    real alpha;         // opening angle
    u32 ads_edge;

    WT_GPU_FN vec3 e() const { return cross(nff, tff); }
};

/** @brief Per-component `m::mix` of quantity vectors: a*(1-x) + b*x, exactly a or b at x=0 or 1. */
WT_GPU_FN vec3 mix(vec3 a, vec3 b, real x) {
    if (x==real(0)) return a;
    if (x==real(1)) return b;
    return a*(1-x) + b*x;
}

WT_GPU_FN wedge make_wedge(const WT_GPU_GLOBAL edge* edges, aperture_entry entry) {
    const u32 id = entry.edge_and_face & 0x7fffffffu;
    const bool f1_is_front = (entry.edge_and_face>>31)!=0;
    const edge ed = edges[id];

    const vec3 v1 = mix(ed.a, ed.b, entry.t1);
    const vec3 v2 = mix(ed.a, ed.b, entry.t2);
    wedge w;
    w.v = (v1+v2)/2;
    w.l = length(v2-v1);
    w.nff = f1_is_front ? ed.n1 : ed.n2;
    w.tff = f1_is_front ? ed.t1 : ed.t2;
    w.nbf = f1_is_front ? ed.n2 : ed.n1;
    w.alpha = ed.alpha;
    w.ads_edge = id;
    return w;
}

struct edge_param_range_t {
    real t1, t2;
};

/** @brief `intersect::intersect_edge_ellipsoid`: where the segment p0p1 crosses the ellipsoid, as segment parameters. */
WT_GPU_FN edge_param_range_t intersect_edge_ellipsoid(vec3 p0, vec3 p1, vec3 centre,
                                                      vec3 x, vec3 y, vec3 axes) {
    const vec3 z = cross(x,y);
    p0 = p0 - centre;
    p1 = p1 - centre;
    const vec3 q0 = { dot(p0,x)/axes.x, dot(p0,y)/axes.y, dot(p0,z)/axes.z };
    const vec3 q1 = { dot(p1,x)/axes.x, dot(p1,y)/axes.y, dot(p1,z)/axes.z };

    const vec3 d = q1-q0;
    const real a = dot(d,d);
    const real b = dot(q0,d)*2;
    const real c = dot(q0,q0)-1;

    const real det2 = b*b - 4*a*c;
    if (det2<=real(0) || a==real(0))
        return { 0,0 };

    const real recp_a = 1/a;
    const real det = sqrt(det2);
    real t1 = real(.5)*(-b - sign(b)*det)*recp_a;
    real t2 = t1==real(0) ? -b*recp_a : c*recp_a/t1;
    if (t1>t2)
        swap(t1,t2);
    return { t1, t2 };
}

WT_GPU_FN real clamp01(real x) { return clamp(x, real(0), real(1)); }

/**
 * @brief The aperture constructor (`free_space_diffraction_t::free_space_diffraction_t`):
 *        the lit parts of `edges` (sorted, unique) within the ellipsoid
 *        (`region_frame`, `region_size`) around `wp`. Returns the entry count.
 */
WT_GPU_FN u32 build_aperture(const WT_GPU_GLOBAL edge* edges,
                             const WT_GPU_GLOBAL u32* edge_ids, u32 edge_count,
                             vec3 wp, frame region_frame, vec3 region_size, vec3 wi,
                             WT_GPU_GLOBAL aperture_entry* out) {
    u32 count = 0;
    for (u32 i=0; i<edge_count; ++i) {
        const edge ed = edges[edge_ids[i]];

        const bool f1_is_front = dot(wi, ed.n1)>real(0);
        const vec3 nff = f1_is_front ? ed.n1 : ed.n2;

        // light incident from inside the wedge?
        if (dot(wi, nff)<=real(0))
            continue;

        // clamp edges to interaction region
        real t1 = 0, t2 = 1;
        if (isfinite(region_size.x) && isfinite(region_size.y) && isfinite(region_size.z)) {
            const auto intr = intersect_edge_ellipsoid(ed.a, ed.b, wp, region_frame.t, region_frame.b, region_size);
            t1 = clamp01(intr.t1);
            t2 = clamp01(intr.t2);
        }
        if (mix(ed.a, ed.b, t1)==mix(ed.a, ed.b, t2))
            continue;

        out[count++] = { edge_ids[i] | (f1_is_front ? 0x80000000u : 0u), t1, t2 };
    }
    return count;
}

/** @brief `wedge_edge_t::diffraction_point(src, dst)`: the point on the edge satisfying Fermat's principle, if any. */
WT_GPU_FN bool diffraction_point(const WT_GPU_THREAD wedge& w, vec3 src, vec3 dst, WT_GPU_THREAD vec3& p) {
    const vec3 e = w.e();
    const vec3 sv = src - w.v, dv = dst - w.v;
    const real sl = sqrt(length2(vec2{ dot(sv, w.tff), dot(sv, w.nff) }));
    const real dl = sqrt(length2(vec2{ dot(dv, w.tff), dot(dv, w.nff) }));
    const real dist = dot(e, sv) + dot(dst - src, e) * sl/(sl + dl);

    if (abs(dist)>w.l/2)
        return false;
    p = w.v + e*dist;
    return !(p==src || p==dst);
}

/** @brief `wedge_edge_t::diffraction_point(src, wo)`: the point diffracting light from `src` into `wo`, if any. */
WT_GPU_FN bool diffraction_point_dir(const WT_GPU_THREAD wedge& w, vec3 src, vec3 wo, WT_GPU_THREAD vec3& p) {
    const vec3 e = w.e();
    const real cos_beta = dot(wo,e);
    const real sin_beta = sqrt(max(real(0), 1-sqr(cos_beta)));
    if (sin_beta<utd_min_sin_beta)
        return false;

    const vec3 sv = src - w.v;
    const real sl = sqrt(length2(vec2{ dot(sv, w.tff), dot(sv, w.nff) }));
    const vec3 prj_src = w.v + dot(sv,e)*e;
    p = prj_src + (sl * (cos_beta / sin_beta)) * e;

    if (length2(p-w.v)>sqr(w.l/2))
        return false;
    return !(p==src);
}


/*
 * UTD
 */

/**
 * @brief The Faddeeva function w(z) = exp(-z²) erfc(-iz), for Im z>=0, by
 *        Weideman's rational approximation ("Computation of the complex error
 *        function", SIAM J. Numer. Anal. 31, 1994) with N=16 terms (N=32 with
 *        64-bit `real`): relative error below 2e-7 (4e-14) where `utd_F` uses
 *        it. Unlike erf's Taylor series, its terms do not cancel, so it is
 *        accurate in single precision. (The CPU uses libcerf.)
 */
WT_GPU_FN complex faddeeva_w(complex z) {
#if defined(WT_GPU_DBL_PRECISION)
    const real L = real(4.7568284600108841);
    const real a[] = {
        real(-1.3025521217935973e-12), real(3.7394878860119718e-12), real(8.0338825869663566e-12),
        real(-2.1542198580526417e-11), real(-5.5438313661948513e-11), real(1.1658270182568486e-10),
        real(4.1537828038284985e-10), real(-5.2310059756033311e-10), real(-3.2080143402835049e-09),
        real(8.1249405814443065e-10), real(2.3797564793759385e-08), real(2.2930441309632066e-08),
        real(-1.4813078776099164e-07), real(-4.1840763650990986e-07), real(4.2558331246611569e-07),
        real(4.4015317307161128e-06), real(6.8210319430633826e-06), real(-2.1409619203438535e-05),
        real(-0.00013075449254629223), real(-0.00024532980270344784), real(0.00039259136069980105),
        real(0.0045195411053481349), real(0.019006155784844603), real(0.057304403529835582),
        real(0.14060716226893638), real(0.29544451071508593), real(0.54601397206393254),
        real(0.90192548936479922), real(1.3455441692345438), real(1.825669629632481),
        real(2.2635372999002663), real(2.5722534081245687),
    };
#else
    const real L = real(3.363585661014858);
    const real a[] = {
        real(9.939322541158483e-07), real(3.9812875769242029e-06), real(-5.5842334096278812e-06),
        real(-2.7346404625247511e-05), real(2.1709867932803717e-05), real(0.00021071056396561316),
        real(8.7031584284871788e-05), real(-0.0015276597401226166), real(-0.0038810151890231043),
        real(0.003682567317091906), real(0.05182240243160996), real(0.19124172674669423),
        real(0.46929090090360304), real(0.88644783020505402), real(1.3622408222719586),
        real(1.7483958860819619),
    };
#endif
    constexpr real inv_sqrt_pi = real(0.56418958354775628695);

    // Z = (L+iz)/(L-iz) lies in the unit disc; w = 2p(Z)/(L-iz)² + 1/(√π (L-iz))
    const complex iz = { -z.im, z.re };
    const complex rden = complex{ 1,0 } / (complex{ L,0 } - iz);
    const complex Z = (complex{ L,0 } + iz) * rden;
    complex p = { 0,0 };
    for (const real c : a)
        p = p*Z + complex{ c,0 };
    return (2*p*rden + complex{ inv_sqrt_pi,0 }) * rden;
}

/** @brief `utd::UTDa<sgn>` */
WT_GPU_FN real utd_a(int sgn, real phi, real n) {
    const real N = round((real(sgn)*pi + phi) * inv_two_pi / n);
    return 2 * sqr(cos(pi*n*N - phi/2));
}

/** @brief `utd::UTDF`: the UTD transition function. */
WT_GPU_FN complex utd_F(real x) {
    const real absx = abs(x);

    complex result;
    if (absx<6) {
        // exp(i|x|) erfc(exp(iπ/4)√|x|) = w(exp(i3π/4)√|x|)
        const real sqrt_x = sqrt(absx);
        const complex w = faddeeva_w(complex{ -inv_sqrt_two*sqrt_x, inv_sqrt_two*sqrt_x });
        result = complex{ 1,1 } * (sqrt_pi_2 * sqrt_x) * w;
    } else {
        // fast approximation for large values
        const real r = 1/(2*absx);
        const real r2 = r*r;
        const real r3 = r2*r;
        const real r4 = r2*r2;
        result = complex{ 1 - 3*r2 + 75*r4, r - 15*r3 };
    }

    return x<real(0) ? conj(result) : result;
}

WT_GPU_FN real cot(real x) { return 1/tan(x); }

struct utd_t {
    complex Ds, Dh;
};

/** @brief `wedge_edge_t::UTD`: the soft and hard diffraction coefficients (without the phase exp(-i k ro)). */
WT_GPU_FN utd_t utd(const WT_GPU_THREAD wedge& w, real k, vec3 wi, vec3 wo, real ro) {
    const vec3 e = w.e();
    const real n = 2 - w.alpha*inv_pi;

    // angles
    const real sin_beta2 = max(real(0), 1-sqr(dot(wi,e)));
    const real sin_beta = sqrt(sin_beta2);
    const real phii = atan2(dot(w.nff,wi), dot(w.tff,wi));
    const real phio = atan2(dot(w.nff,wo), dot(w.tff,wo));

    // distance parameters
    const real Li = ro * sin_beta2;

    // diffraction coefficients
    const real a1 = utd_a(+1, phii-phio, n);
    const real a2 = utd_a(-1, phii-phio, n);
    const real a3 = utd_a(+1, phii+phio, n);
    const real a4 = utd_a(-1, phii+phio, n);
    const complex F1 = utd_F(k*Li * a1);
    const complex F2 = utd_F(k*Li * a2);
    const complex F3 = utd_F(k*Li * a3);
    const complex F4 = utd_F(k*Li * a4);
    const complex D1 = -cot((pi + (phii-phio)) / (2*n)) * F1;
    const complex D2 = -cot((pi - (phii-phio)) / (2*n)) * F2;
    const complex D3 = -cot((pi + (phii+phio)) / (2*n)) * F3;
    const complex D4 = -cot((pi - (phii+phio)) / (2*n)) * F4;

    const real kro = k*ro;
    const complex D = (1/(2*n*sqrt(kro)*sin_beta) * inv_sqrt_two_pi) * expi(-pi_4);

    const real t1 = mod(phii+phio, pi_2);
    const real t2 = mod(phii-phio, pi_2);
    if (abs(t1)<real(1e-5) || abs(t2)<real(1e-5))
        return { { 0,0 }, { 0,0 } };

    const complex Ds = D1+D2-(D3+D4);
    const complex Dh = D1+D2+(D3+D4);
    return { -(D*Ds), -(D*Dh) };
}


/*
 * free_space_diffraction_t::sample and ::pdf
 */

/** @brief `sampler_t::normal2d(u).x` */
WT_GPU_FN real normal2d_x(vec2 u) {
    const real r = sqrt(-2*log(1-u.x));
    return r*cos(two_pi*u.y);
}

/** @brief The Keller-cone angle density of diffracting from `src` into `wo` (`free_space_diffraction_t::pdf`, in 1/rad). */
WT_GPU_FN real fsd_pdf(const WT_GPU_GLOBAL edge* edges,
                       const WT_GPU_GLOBAL aperture_entry* aperture, u32 count,
                       real k, vec3 src, vec3 wo) {
    if (count==0)
        return 0;

    real ret = 0;
    for (u32 i=0; i<count; ++i) {
        const wedge w = make_wedge(edges, aperture[i]);
        vec3 p;
        if (!diffraction_point_dir(w, src, wo, p))
            continue;

        const vec3 ui = src-p;
        if ((dot(wo,w.nff)<=real(0) && dot(wo,w.nbf)<=real(0)) ||
            (dot(ui,w.nff)<=real(0) && dot(ui,w.nbf)<=real(0)))
            continue;

        const real ri = length(src-p);
        const vec3 wi = (src-p)/ri;

        const real phii = atan2(dot(w.nff,wi), dot(w.tff,wi));
        const real phio = atan2(dot(w.nff,wo), dot(w.tff,wo));
        const real sigma = sqrt(utd_IS_sigma_scale / (k*ri));

        const real mean_phi1 = pi + phii;
        const real mean_phi2 = pi - phii;
        real x1 = abs(mod(phio - mean_phi1, two_pi));
        real x2 = abs(mod(phio - mean_phi2, two_pi));
        if (x1>pi) x1 -= two_pi;
        if (x2>pi) x2 -= two_pi;

        // angle density along the Keller cone
        ret += inv_sqrt_two_pi/sigma * (
            exp(real(-.5) * sqr(x1/sigma)) +
            exp(real(-.5) * sqr(x2/sigma))
        )/2;
    }

    return ret / real(count+1);
}

struct fsd_sample_t {
    vec3 wo;
    real weight;    // 0 if the sample failed
};

/** @brief `free_space_diffraction_t::sample`: diffracts light from `src` through the aperture around `wp`. */
WT_GPU_FN fsd_sample_t fsd_sample(const WT_GPU_GLOBAL edge* edges,
                                  const WT_GPU_GLOBAL aperture_entry* aperture, u32 count,
                                  real k, vec3 wp, vec3 src, WT_GPU_THREAD rng& rnd) {
    const fsd_sample_t failed = { { 0,0,1 }, 0 };

    // sampler_t::uniform_int_interval(0, count+1)
    const u32 eidx = min(count, u32(rnd.r()*real(count+1)));
    if (eidx==count) {
        // sampled direct term
        const vec3 wi = normalize(src-wp);
        return { -wi, real(count+1) };
    }

    // sampled edge
    const wedge w = make_wedge(edges, aperture[eidx]);
    const vec3 e = w.e();
    const vec3 p = w.v + ((rnd.r()-real(.5)) * w.l) * e;
    const vec3 ui = src-p;
    if (dot(ui,w.nff)<=real(0) && dot(ui,w.nbf)<=real(0))
        return failed;

    const real ri = length(src-p);
    const vec3 wi = (src-p)/ri;
    const real phii = atan2(dot(w.nff,wi), dot(w.tff,wi));

    const real sigma = sqrt(utd_IS_sigma_scale / (k*ri));
    const real sample = sigma * normal2d_x(rnd.r2());

    const real mean_phi1 = pi + phii;
    const real mean_phi2 = pi - phii;
    const real phio = (rnd.r()<real(.5) ? mean_phi1 : mean_phi2) + sample;

    const real cos_beta = dot(wi,e);
    const real sin_beta = sqrt(max(real(0), 1-sqr(cos_beta)));
    const vec3 wo = sin_beta * (cos(phio)*w.tff + sin(phio)*w.nff) - cos_beta * e;

    if (dot(wo,w.nff)<=real(0) && dot(wo,w.nbf)<=real(0))
        return failed;
    // avoid grazing angles
    if (sin_beta<utd_min_sin_beta)
        return failed;

    const real dpd = fsd_pdf(edges, aperture, count, k, src, wo);
    if (dpd==real(0))
        return failed;
    return { wo, 1/dpd };
}

}
