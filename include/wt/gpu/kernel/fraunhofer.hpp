/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "plt_path.hpp"

/*
 * plt_bdpt's free-space diffraction ("A Free-Space Diffraction BSDF",
 * Steinberg et al., SIGGRAPH 2024): edge-based diffraction of beams under the
 * Fraunhofer approximation, ported from wt/interaction/fsd/fraunhofer/
 * {fsd,fsd_lut,fsd_sampler,free_space_diffraction}.hpp and their sources.
 * Where a beam's interaction region holds no surface but edges, the edges'
 * silhouette (clipped to the beam's cross section, and subdivided) is an
 * aperture, whose angular scattering function (ASF) scatters the beam. Its
 * edges' diffraction lobes are sampled with lookup tables of the sampler
 * (`plt_scene::fsd_lut`), and the ASF by rejection, as the CPU's.
 *
 * Also the flux of a beam's Gaussian wavefront over triangles
 * (`gaussian2d_t::integrate_triangle`), which the aperture's power excludes.
 *
 * Lengths of the ASF's canonical space are millimetres, scaled by the
 * wavenumber (`fsd_unit`).
 *
 * In namespace `fraunhofer`, as the CPU's, apart from fsd.hpp's (UTD) free-space
 * diffraction of plt_path.
 */

namespace wt::gpu::kernel::fraunhofer {

/** @brief The ASF's unit of length (`free_space_diffraction_t::fsd_unit`), in metres. */
constexpr real fsd_unit = real(1e-3);
/** @brief Scattered directions whose sines (squared) exceed this are not sampled (`wo2_cutoff`). */
constexpr real fsd_wo2_cutoff = real(.85);

/** @brief Power contained in χₑ×|𝛂1|² and in χₑ×|𝛂2|² (`fsd::PA1`, `PA2`). */
constexpr real fsd_PA1 = real(0.0049361075794549872500);
constexpr real fsd_PA2 = real(0.21899789398059305541);
/** @brief `fsd::P0_sigma`: the standard deviation of the 0-th order lobe. */
constexpr real fsd_P0_sigma = real(0.288675134594813)/4;

/**
 * @brief The sampler's lookup tables (`fsd_lut_t`), in `plt_scene::fsd_lut`:
 *        the inverse CDFs of the angles of |𝛂1|² and of |𝛂2|² (lobes 0 and
 *        1), of `fsd_theta_samples` each, then of their radii, a row of
 *        `fsd_radius_samples` for each of as many angles.
 */
constexpr u32 fsd_theta_samples = 2048;
constexpr u32 fsd_radius_samples = 3072;
constexpr u64 fsd_lut_size = 2*u64(fsd_theta_samples) + 2*u64(fsd_radius_samples)*fsd_radius_samples;

WT_GPU_FN u64 fsd_lut_theta(u32 lobe) { return u64(lobe)*fsd_theta_samples; }
WT_GPU_FN u64 fsd_lut_radius(u32 lobe) {
    return 2*u64(fsd_theta_samples) + u64(lobe)*fsd_radius_samples*fsd_radius_samples;
}

/**
 * @brief An aperture edge (`fsd::edge_t`): its vector and midpoint (in
 *        `fsd_unit`s), its beam amplitudes, and its probability of being
 *        sampled. The amplitudes of the CPU's are real: `a_b` is a-b, and
 *        `iab_2` is i times `h`, (a+b)/2.
 */
struct fsd_edge {
    vec2 e, v;
    real a_b, h;
    real pdf;
};

/**
 * @brief An aperture (`fsd::fsd_aperture_t`): its `count` edges, from
 *        `offset` on in a path's aperture memory, and its 0-th order lobe.
 *        Empty (no edges) if the beam does not diffract.
 */
struct fsd_aperture {
    u32 offset, count;
    real P0;
    real P0_pdf;                // probability of sampling the 0-th order lobe
    real psi02;                 // magnitude squared of the field amplitude integrated over the opening
    real recp_I;                // reciprocal of the beam intensity integrated over the opening
};


/*
 * The angular scattering function
 */

/** @brief `m::sinc` (boost's) */
WT_GPU_FN real sinc(real x) {
    const real taylor_2_bound = real(0.00034526698300124390839884978618400831996329879769945);
    const real taylor_n_bound = real(0.018581361171917516667460937040007436176452688944747);
    if (abs(x)>=taylor_n_bound)
        return sin(x)/x;
    real result = 1;
    if (abs(x)>=epsilon) {
        const real x2 = x*x;
        result -= x2/real(6);
        if (abs(x)>=taylor_2_bound)
            result += (x2*x2)/real(120);
    }
    return result;
}

/** @brief `fsd::alpha1` and `fsd::alpha2` of `z`, which share their terms. */
WT_GPU_FN vec2 fsd_alphas(vec2 z) {
    if (z.x==real(0))
        return { 0,0 };
    const real s = sinc(z.x/2);
    const real c = inv_two_pi * z.y/(z.x*z.x+z.y*z.y);
    return { c/z.x * (cos(z.x/2)-s), c*s };
}

/** @brief `fsd::chi_e`: the masking function of the diffracted lobes. */
WT_GPU_FN real fsd_chi_e(vec2 xi) {
    const real chi = real(0.830092714835359);
    const real xi2 = dot(xi,xi);
    const real t  = 1+chi*xi2;
    const real t2 = t*t;
    const real t3 = t2*t;
    return max(real(0), 1 - (3/t2 - 2/t3));
}
/** @brief `fsd::chi_0`: the masking function of the 0-th order lobe. */
WT_GPU_FN real fsd_chi_0(vec2 xi) {
    xi = xi / fsd_P0_sigma;
    return exp(-real(.5) * dot(xi,xi));
}

/** @brief `xi * edge_t::Xi()`: `xi` in the edge's (e, m) frame, m=(e.y,-e.x). */
WT_GPU_FN vec2 fsd_zeta(const WT_GPU_THREAD fsd_edge& e, vec2 xi) {
    return { dot(xi, e.e), dot(xi, vec2{ e.e.y,-e.e.x }) };
}

/** @brief `fsd::Psi`: an edge's diffracted amplitude at `xi` (but the 0-th order lobe). */
WT_GPU_FN complex fsd_Psi(const WT_GPU_THREAD fsd_edge& e, vec2 xi) {
    const vec2 alphas = fsd_alphas(fsd_zeta(e, xi));
    const complex a = { e.a_b * alphas.x, e.h * alphas.y };
    const real ee2 = length2(e.e);
    const real vxi = dot(e.v, xi);
    return complex{ ee2*cos(-vxi), ee2*sin(-vxi) } * a;
}

/** @brief `fsd::ASF_unclamped`: the edges' diffraction, unstable about xi=0. */
WT_GPU_FN real fsd_ASF_unclamped(const WT_GPU_GLOBAL fsd_edge* edges, u32 count, vec2 xi) {
    complex amplitude = { 0,0 };
    for (u32 i=0; i<count; ++i) {
        const fsd_edge e = edges[i];
        amplitude = amplitude + fsd_Psi(e, xi);
    }
    return norm(amplitude);
}

/** @brief `fsd::ASF`: the angular scattering function. */
WT_GPU_FN real fsd_ASF(const WT_GPU_THREAD fsd_aperture& ap, const WT_GPU_GLOBAL fsd_edge* edges, vec2 xi) {
    return fsd_ASF_unclamped(edges + ap.offset, ap.count, xi) * fsd_chi_e(xi) + ap.psi02 * fsd_chi_0(xi);
}

/** @brief The ASF and the sampling density at a point. */
struct fsd_densities_t {
    real f;     // `fsd::ASF`
    real g;     // `fsd::sampling_density`: the incoherent sum of the edges' |Ѱ|², and the 0-th order lobe's
};

/**
 * @brief `fsd::ASF` and `fsd::sampling_density` at `xi`, of one pass over the
 *        edges (an edge's |Ѱ|², `fsd::Psi2`, is the norm of its amplitude).
 *        Sampling evaluates both at each of its (many) tries.
 */
WT_GPU_FN fsd_densities_t fsd_densities(const WT_GPU_THREAD fsd_aperture& ap, const WT_GPU_GLOBAL fsd_edge* edges,
                                        vec2 xi) {
    complex amplitude = { 0,0 };
    real diffracted = 0;
    for (u32 i=0; i<ap.count; ++i) {
        const fsd_edge e = edges[ap.offset+i];
        const complex psi = fsd_Psi(e, xi);
        amplitude = amplitude + psi;
        diffracted += norm(psi);
    }
    const real chi_e = fsd_chi_e(xi);
    const real chi_0 = fsd_chi_0(xi);
    return {
        norm(amplitude) * chi_e + ap.psi02 * chi_0,
        diffracted * chi_e + ap.P0 * inv_two_pi / sqr(fsd_P0_sigma) * chi_0,
    };
}


/*
 * Sampling (`fsd_sampler_t`)
 */

/** @brief `fsd_lut_t::lerp`: linear interpolation of a table of `S` values at x in [0,1]. */
WT_GPU_FN real fsd_lerp(const WT_GPU_GLOBAL real* tbl, u32 S, real x) {
    x *= real(S-1);
    const u32 l = u32(x)<S-1 ? u32(x) : S-1;
    const u32 h = l+1<S-1 ? l+1 : S-1;
    const real f = x - floor(x);
    return f*tbl[h] + (1-f)*tbl[l];
}

/** @brief `fsd_lut_t::sample`: a point of lobe `lobe`'s |𝛂|² for random numbers `r` (a random quadrant). */
WT_GPU_FN vec2 fsd_lut_sample(const WT_GPU_GLOBAL real* lut, u32 lobe, real r0, real r1, real r2) {
    const u32 N = fsd_theta_samples, M = fsd_radius_samples;
    const real theta = fsd_lerp(lut + fsd_lut_theta(lobe), N, r0);
    const real theta_fract = theta * 2/pi;

    // bilinear: between the radius tables of the angles about theta_fract
    const WT_GPU_GLOBAL real* radius = lut + fsd_lut_radius(lobe);
    const real x = theta_fract * real(M-1);
    const u32 l = u32(x)<M-1 ? u32(x) : M-1;
    const u32 h = l+1<M-1 ? l+1 : M-1;
    const real f = x - floor(x);
    const real r = max(real(0), f*fsd_lerp(radius + u64(h)*M, M, r1) + (1-f)*fsd_lerp(radius + u64(l)*M, M, r1));

    vec2 zeta = vec2{ cos(theta), sin(theta) } * r;
    const i32 q = min(3, i32(r2*4));
    zeta.x *= ((q+1)/2)%2==0 ? real(1) : real(-1);
    zeta.y *= (q/2)%2==0 ? real(1) : real(-1);
    return zeta;
}

/** @brief `fsd_sample_impl_t::sample1`: a point of an edge's diffraction lobes. */
WT_GPU_FN vec2 fsd_sample_edge(const WT_GPU_GLOBAL real* lut, const WT_GPU_THREAD fsd_edge& e,
                               WT_GPU_THREAD rng& rnd) {
    // one of the 𝛂1,𝛂2 lobes, by power (sampler_t::discrete)
    const real A = sqr(e.a_b), B = sqr(e.h);
    const u32 lobe = rnd.r()*(A+B)<A ? 0u : 1u;
    const real r0 = rnd.r(), r1 = rnd.r(), r2 = rnd.r();
    const vec2 zeta = fsd_lut_sample(lut, lobe, r0, r1, r2);

    // times the inverse of the edge's Ξ matrix (glm::inverse of the columns e, m)
    const vec2 m = { e.e.y,-e.e.x };
    const real recp_det = 1 / (e.e.x*m.y - m.x*e.e.y);
    return { dot(zeta, vec2{ m.y,-e.e.y }*recp_det), dot(zeta, vec2{ -m.x,e.e.x }*recp_det) };
}

/** @brief `fsd_sample_impl_t::sampleN`: the 0-th order lobe, or an edge's, by their power. */
WT_GPU_FN vec2 fsd_sample_lobe(const WT_GPU_GLOBAL real* lut, const WT_GPU_THREAD fsd_aperture& ap,
                               const WT_GPU_GLOBAL fsd_edge* edges, WT_GPU_THREAD rng& rnd) {
    // sampler_t::discrete<true> over the 0-th order lobe and the edges
    const real p = rnd.r();
    real cdf = ap.P0_pdf;
    if (p<cdf || ap.count==0) {
        // sampler_t::normal2d
        const vec2 u = rnd.r2();
        const real r = sqrt(-2*log(1-u.x));
        const real theta = two_pi * u.y;
        return vec2{ r*cos(theta), r*sin(theta) } * fsd_P0_sigma;
    }
    u32 i = 0;
    for (; i+1<ap.count; ++i) {
        cdf += edges[ap.offset+i].pdf;
        if (p<cdf)
            break;
    }
    const fsd_edge e = edges[ap.offset+i];
    return fsd_sample_edge(lut, e, rnd);
}

/** @brief A sampled scattering of an aperture (`free_space_diffraction_t::sample_ret_t`). */
struct fsd_sample_t {
    vec3 wo;            // local, in the frame of the beam that diffracts
    real dpd;           // its density (0: none)
    real weight;
};

/**
 * @brief `free_space_diffraction_t::sample`: samples the ASF by rejection
 *        (`sample_rejection`; an aperture of one edge, which the lookup
 *        tables sample virtually exactly, needs none), at wavenumber `k`.
 */
WT_GPU_FN fsd_sample_t fsd_sample(const WT_GPU_GLOBAL real* lut, const WT_GPU_THREAD fsd_aperture& ap,
                                  const WT_GPU_GLOBAL fsd_edge* edges, real k, WT_GPU_THREAD rng& rnd) {
    const fsd_sample_t none = { { 0,0,1 }, 0, 0 };

    const u32 M = ap.count;
    const bool rejection = M>1;
    const u64 max_tries = u64(M)*1024;
    const real recp_M = real(1)/real(M);
    vec2 xi = { 0,0 };
    real pdf = 0;
    for (u64 tr=0; tr<max_tries; ++tr) {
        const vec2 x = fsd_sample_lobe(lut, ap, edges, rnd);
        const fsd_densities_t d = fsd_densities(ap, edges, x);
        if (!rejection || rnd.r()*d.g < d.f*recp_M) {
            xi = x;
            pdf = d.f * ap.recp_I;
            break;
        }
    }
    if (!(pdf>real(0)))
        return none;

    // tan to sin, and scale
    const vec2 zeta = xi / (k*fsd_unit);
    const vec2 wo = { zeta.x/sqrt(1+sqr(zeta.x)), zeta.y/sqrt(1+sqr(zeta.y)) };
    const real wo2 = length2(wo);
    if (!(wo2<fsd_wo2_cutoff))
        return none;
    return { { wo.x, wo.y, sqrt(1-wo2) }, pdf, 1 };
}

/**
 * @brief `free_space_diffraction_t::pdf` (an approximation), which is also
 *        its ASF (`f`): of the local direction `wo`, at wavenumber `k`.
 */
WT_GPU_FN real fsd_pdf(const WT_GPU_THREAD fsd_aperture& ap, const WT_GPU_GLOBAL fsd_edge* edges, real k, vec3 wo) {
    const real wo2 = sqr(wo.x) + sqr(wo.y);
    if (wo.z<=real(0) || wo2>=fsd_wo2_cutoff)
        return 0;
    // sin to tan, and scale
    const real scale = k*fsd_unit;
    const vec2 xi = vec2{ wo.x/sqrt(1-sqr(wo.x)), wo.y/sqrt(1-sqr(wo.y)) } * scale;
    const real pdf = fsd_ASF(ap, edges, xi) * ap.recp_I;
    return real(0)<=pdf && pdf<real(1e+2) ? pdf : real(0);
}


/*
 * Apertures (`free_space_diffraction_t`'s constructor)
 */

/** @brief `intersect_edge_ellipse` (`intersect_edge_circle_ret_t`'s points and t1, t2). */
struct edge_ellipse_hit_t {
    i32 points;
    real t1, t2;
};

/** @brief `intersect::intersect_edge_ellipse`: of edge p0p1, and the axis-aligned ellipse of axes `r`. */
WT_GPU_FN edge_ellipse_hit_t intersect_edge_ellipse(vec2 p0, vec2 p1, vec2 r) {
    const vec2 recp_scale = { 1/r.x, 1/r.y };
    p0 = p0 * recp_scale;
    p1 = p1 * recp_scale;
    const vec2 d = p1-p0;

    const real a = dot(d,d);
    const real b = 2*dot(p0,d);
    const real c = dot(p0,p0) - 1;
    const real det2 = b*b - 4*a*c;
    if (det2<=real(0) || a==real(0))
        return { 0, 0, 0 };

    const real recp_a = 1/a;
    const real det = sqrt(det2);
    real t1 = real(.5)*(-b - sign(b)*det)*recp_a;
    real t2 = t1==real(0) ? -b*recp_a : c*recp_a/t1;
    if (t1>t2)
        swap(t1,t2);

    const bool u1valid = t1>=real(0) && real(1)>=t1;
    const bool u2valid = t2>=real(0) && real(1)>=t2;
    if (!u1valid && !u2valid)
        return { 0, t1, t2 };
    if (u1valid && u2valid)
        return { 2, t1, t2 };
    return { 1, u1valid ? t1 : t2, u1valid ? t2 : t1 };
}

/** @brief `gaussian_wavefront_t::beam_cross_section_envelope`: a beam's envelope is 3 standard deviations of its intensity. */
constexpr real gaussian_cross_section_envelope = 3;

/** @brief A beam's Gaussian wavefront (`gaussian_wavefront_t`, centred): its intensity's standard deviations. */
struct gaussian_wavefront {
    vec2 sigma;

    WT_GPU_FN bool is_dirac() const { return sigma.x==real(0) || sigma.y==real(0); }
    /** @brief `amplitude_magnitude`: the square root of the intensity's density at `p`. */
    WT_GPU_FN real amplitude(vec2 p) const {
        if (is_dirac())
            return p.x==real(0) && p.y==real(0) ? inf() : real(0);
        const vec2 recp_sigma = { 1/sigma.x, 1/sigma.y };
        const vec2 u = p * recp_sigma;
        return sqrt(inv_two_pi * recp_sigma.x*recp_sigma.y * exp(-dot(u,u)/2));
    }
};

/** @brief `beam_generic_t::wavefront`: of the beam `envelope`, at distance `dist`. */
WT_GPU_FN gaussian_wavefront wavefront_at(const WT_GPU_THREAD cone& envelope, real dist) {
    const vec2 a = envelope.axes(dist);
    return { a / gaussian_cross_section_envelope };
}

/**
 * @brief Builds the aperture of edges `edge_ids[0..count-1]` (BVH edges)
 *        that the beam `envelope` meets where its wavefront is `wf` (in its
 *        local frame), of wavenumber `k`, `total_power` of whose intensity
 *        passes the opening: its edges go to `edges[offset ..]`, up to
 *        `capacity` (more: status_scratch_overflow, and an empty aperture).
 */
WT_GPU_FN fsd_aperture build_fsd_aperture(const WT_GPU_THREAD plt_scene& scene,
                                          const WT_GPU_GLOBAL u32* edge_ids, u32 count,
                                          const WT_GPU_THREAD cone& envelope, gaussian_wavefront wf,
                                          real k, real total_power,
                                          WT_GPU_GLOBAL fsd_edge* edges, u32 offset, u32 capacity,
                                          WT_GPU_THREAD u32& status) {
    fsd_aperture ap = { offset, 0, 0, 1, 0, total_power>real(0) ? 1/total_power : real(0) };

    const frame f = envelope.frame();
    const vec3 o = envelope.r.o;
    const vec3 d = envelope.r.d;
    const vec2 ellipse = wf.sigma * gaussian_cross_section_envelope;
    const real max_edge_length = real(.33) * max(ellipse.x, ellipse.y);
    const auto in_ellipse = [&](vec2 p) {
        const vec2 q = { p.x/ellipse.x, p.y/ellipse.y };
        return dot(q,q)<=real(1);
    };
    const auto mix2 = [](vec2 a, vec2 b, real t) { return a*(1-t) + b*t; };

    real P_total = 0;
    for (u32 j=0; j<count; ++j) {
        const edge ed = scene.edges[edge_ids[j]];
        // only the projected silhouette
        if (dot(d, ed.n1) * dot(d, ed.n2)>=real(0))
            continue;

        const vec3 l1 = f.to_local(ed.a - o), l2 = f.to_local(ed.b - o);
        const vec2 u1 = { l1.x, l1.y }, u2 = { l2.x, l2.y };
        // clamp to the cross-section ellipse (the CPU's interval, as it takes it)
        real t1 = 0, t2 = 1;
        if (!in_ellipse(u1) || !in_ellipse(u2)) {
            const edge_ellipse_hit_t hit = intersect_edge_ellipse(u1, u2, ellipse);
            if (hit.points==0)
                continue;
            t1 = max(real(0), hit.t1);
            t2 = min(real(1), hit.t2);
        }

        // subdivide into segments
        const real len = length(mix2(u1,u2,t1) - mix2(u1,u2,t2));
        const i32 segments = max(1, i32(round(len/max_edge_length) + real(.5)));
        const real seg = real(1)/real(segments);
        vec2 v1 = mix2(u1,u2,t1);
        real a = wf.amplitude(v1);
        for (i32 i=0; i<segments; ++i) {
            const real tt = t1*(1-real(i+1)*seg) + t2*(real(i+1)*seg);
            const vec2 v2 = mix2(u1,u2,tt);
            const real b = wf.amplitude(v2);
            if (a>real(0) || b>real(0)) {
                fsd_edge e;
                e.e = (v2-v1) * (1/fsd_unit);
                e.v = (v1+v2)/2 * (1/fsd_unit);
                e.a_b = a-b;
                e.h = (a+b)/2;
                // fsd::Pj
                const real ee2 = sqr(length2(e.e));
                e.pdf = ee2*fsd_PA1*sqr(e.a_b) + ee2*fsd_PA2*sqr(e.h);
                if (e.pdf>real(0)) {
                    if (ap.count==capacity) {
                        status |= status_scratch_overflow;
                        ap.count = 0;
                        return ap;
                    }
                    edges[offset + ap.count++] = e;
                    P_total += e.pdf;
                }
            }
            v1 = v2;
            a = b;
        }
    }

    // power in the 0-th order lobe (the CPU's approximation)
    const real r0 = 3*fsd_P0_sigma;
    const real s = inv_sqrt_two;
    const WT_GPU_GLOBAL fsd_edge* aes = edges + offset;
    ap.psi02 = (fsd_ASF_unclamped(aes, ap.count, vec2{ -s,-s }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{ -1, 0 }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{ -s,+s }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{  0,+1 }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{ +s,+s }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{ +1, 0 }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{ +s,-s }*r0) +
                fsd_ASF_unclamped(aes, ap.count, vec2{  0,-1 }*r0)) / 8;
    ap.P0 = two_pi * sqr(fsd_P0_sigma) * ap.psi02 / sqr(k*fsd_unit);
    P_total += ap.P0;

    if (P_total>real(0)) {
        const real recp_P_total = 1/P_total;
        ap.P0_pdf = ap.P0 * recp_P_total;
        for (u32 i=0; i<ap.count; ++i)
            edges[offset+i].pdf *= recp_P_total;
    } else {
        ap.P0_pdf = 1;
        ap.count = 0;
    }
    return ap;
}


/*
 * The flux of a Gaussian wavefront over triangles
 */

WT_GPU_FN real erf_gauss(real x) { return erf_lut(x); }

/** @brief `detail::I_gauss_gauss0` of src/math/gaussian2d.cpp */
WT_GPU_FN real I_gauss_gauss0(real a, real b, real c, real d) {
    const real n2 = 1 / (a+2*c*c);
    const real n = sqrt(n2);
    return -sqrt_pi/2 * n * exp(-2*a*sqr(d-b*c)*n2) * (
        erf_gauss((a*b+2*c*d)*n) -
        erf_gauss((a*(1+b)+2*c*(c+d))*n)
    );
}
/** @brief `detail::I_gauss_gauss1` */
WT_GPU_FN real I_gauss_gauss1(real a, real b, real c, real d) {
    const real n2 = 1 / (a+2*c*c);
    const real n = sqrt(n2);
    return -sqrt_pi/2 * n * exp(-2*a*sqr(d-b*c)*n2) * (
        2*erf_gauss(a*(d/c-b)*n) +
        erf_gauss((a*b+2*c*d)*n) +
        erf_gauss((a*(1+b)+2*c*(c+d))*n)
    );
}
/** @brief `detail::I_gauss0` */
WT_GPU_FN real I_gauss0(real a, real b) {
    const real n2 = 1 / a;
    const real n = sqrt(n2);
    return -sqrt_pi/2 * n * (
        erf_gauss(a*b*n) -
        erf_gauss(a*(1+b)*n)
    );
}
/** @brief `detail::I_gauss1` */
WT_GPU_FN real I_gauss1(real a, real b, real c, real d) {
    const real sa  = sqrt(a);
    const real n   = 1/sa;
    const real d_c = d/c;
    return -sqrt_pi/2 * n * (
        sign(b) * erf_gauss(sa*abs(b)) +
        sign(1+b) * erf_gauss(sa*abs(1+b)) -
        2*sign(b-d_c) * erf_gauss(sa*abs(b-d_c))
    );
}
/** @brief `detail::I_gauss_erf` */
WT_GPU_FN real I_gauss_erf(real a, real b, real c, real d) {
    const real d_c = d/c;
    const bool one = c!=real(0) && -d_c>real(0) && -d_c<real(1);
    const auto gg = [&](real s) {
        const real q = sqrt(s);
        return one ? I_gauss_gauss1(a,b,c*q,d*q) : I_gauss_gauss0(a,b,c*q,d*q);
    };
    return
        ((d!=real(0) && c!=real(0)) ? sign(d) : d==real(0) && c!=real(0) ? sign(c) : real(1)) *
        (
            (one ? I_gauss1(a,b,c,d) : I_gauss0(a,b)) -
            2*(
                real(0.2936683276537767)*gg(real(0.6517755981618476)) +
                real(0.135758042187825)*gg(real(3.250040490513459)) +
                real(0.05245255757691102)*gg(real(31.86882707224491)) +
                real(0.01673209873360605)*gg(real(778.6613983601425))
            )
        );
}

/** @brief `util::is_point_in_triangle` (2D) */
WT_GPU_FN bool is_point_in_triangle(vec2 p, vec2 a, vec2 b, vec2 c) {
    const auto sgn = [](vec2 p1, vec2 p2, vec2 p3) {
        return eft::diff_prod(p1.x-p3.x, p2.y-p3.y, p2.x-p3.x, p1.y-p3.y);
    };
    const real s1 = sgn(p, a, b);
    const real s2 = sgn(p, b, c);
    const real s3 = sgn(p, c, a);
    const bool neg = s1<real(0) || s2<real(0) || s3<real(0);
    const bool pos = s1>real(0) || s2>real(0) || s3>real(0);
    return !(neg && pos);
}

/**
 * @brief `gaussian2d_t::integrate_triangle` of the (centred, axis-aligned)
 *        wavefront `wf`: its flux over triangle abc (an approximation, as
 *        the CPU's).
 */
WT_GPU_FN real integrate_triangle(gaussian_wavefront wf, vec2 a, vec2 b, vec2 c) {
    if (wf.is_dirac())
        return is_point_in_triangle(vec2{ 0,0 }, a, b, c) ? real(1) : real(0);

    // radius of the circle, in canonical space, where all the mass is assumed to be
    const real L = 3;

    // fast paths, in canonical space
    const vec2 recp_sigma = { 1/wf.sigma.x, 1/wf.sigma.y };
    a = a * recp_sigma;
    b = b * recp_sigma;
    c = c * recp_sigma;
    if (min(a.x,b.x,c.x)>=L || max(a.x,b.x,c.x)<=-L ||
        min(a.y,b.y,c.y)>=L || max(a.y,b.y,c.y)<=-L)
        return 0;
    const auto in_circle = [&](vec2 p) { return length2(p)<=sqr(L); };
    if (!in_circle(a) && !in_circle(b) && !in_circle(c)) {
        // the vertices are outside the circle: do the edges cross it?
        const vec2 Lv = { L,L };
        if (intersect_edge_ellipse(a, b, Lv).points==0 && intersect_edge_ellipse(a, c, Lv).points==0 &&
            intersect_edge_ellipse(b, c, Lv).points==0)
            return is_point_in_triangle(vec2{ 0,0 }, a, b, c) ? real(1) : real(0);
    }

    const real min_len = min(length2(a-b), length2(a-c), length2(b-c));
    if (min_len<real(1e-3)) {
        // quadrature
        const real delta = real(.002);
        if (b.y<a.y) swap(a,b);
        if (c.y<a.y) swap(a,c);
        const real ab_dxdy = b.y==a.y ? inf() : (b.x-a.x)/(b.y-a.y);
        const real ac_dxdy = c.y==a.y ? inf() : (c.x-a.x)/(c.y-a.y);
        const real bc_dxdy = c.y==b.y ? inf() : (c.x-b.x)/(c.y-b.y);

        real ret = 0;
        for (real y = max(-L, a.y+delta/2); y<min(L, max(b.y,c.y)); y+=delta) {
            real x0 = y<b.y ? ab_dxdy*(y-a.y) + a.x : bc_dxdy*(y-b.y) + b.x;
            real x1 = y<c.y ? ac_dxdy*(y-a.y) + a.x : bc_dxdy*(y-b.y) + b.x;
            if (x0>x1)
                swap(x0,x1);
            for (real x = max(-L,x0)+delta/2; x<min(L,x1); x+=delta)
                ret += exp(-(sqr(x)+sqr(y))/2);
        }
        return ret * inv_two_pi * sqr(delta);
    }

    // analytic approximation; T's columns are b-a and c-a (glm)
    const vec2 c0 = b-a, c1 = c-a;
    const real detT = c0.x*c1.y - c1.x*c0.y;
    const real recp_detT = 1/detT;
    // inverse(T)*a
    const vec2 mu0 = { (c1.y*a.x - c1.x*a.y)*recp_detT, (-c0.y*a.x + c0.x*a.y)*recp_detT };
    // A = transpose(T)*T
    const real Sxx = dot(c0,c0), Sxy = dot(c0,c1), Syy = dot(c1,c1);
    const real detA = Sxx*Syy - Sxy*Sxy;
    if (Syy<=real(0) || detA<=real(0))
        return 0;

    const real denom = 1/sqrt(2*Syy);
    const real pa = detA * sqr(denom);
    const real pb = mu0.x;
    const real cc = Sxy * denom;
    const real dd = (Sxy*mu0.x + Syy*mu0.y)*denom;
    const real q  = real(.5) / denom;

    const real I0 = I_gauss_erf(pa, pb, cc-q, dd+q);
    const real I1 = I_gauss_erf(pa, pb, cc, dd);
    return inv_sqrt_pi/2 * abs(detT * denom) * max(real(0), I0-I1);
}

/** @brief `clip_triangle_ret_t`: a triangle clipped to a z range, a polygon of up to 5 vertices, as up to 3 triangles. */
struct clipped_triangle_t {
    vec3 vs[5];
    i32 tris;

    WT_GPU_FN void triangle(i32 i, WT_GPU_THREAD vec3& a, WT_GPU_THREAD vec3& b, WT_GPU_THREAD vec3& c) const {
        if (i==0)      { a = vs[0]; b = vs[1]; c = vs[2]; }
        else if (i==1) { a = vs[2]; b = vs[0]; c = vs[tris==2 ? 3 : 4]; }
        else           { a = vs[4]; b = vs[2]; c = vs[3]; }
    }
};

/** @brief `intersect::intersect_edge_plane` of edge p0p1 and the plane z=`z`. False if none. */
WT_GPU_FN bool intersect_edge_plane_z(vec3 p0, vec3 p1, real z, WT_GPU_THREAD vec3& p) {
    const real d0 = z - p0.z;
    const real d1 = z - p1.z;
    const vec3 E = p1-p0;
    if (sign(d0)==sign(d1) || E.z==real(0))
        return false;
    const real d = d0 / E.z;
    if (!(d>=real(0) && real(1)>=d))
        return false;
    p = p0 + d*E;
    return true;
}

/** @brief `intersect::clip_triangle_z`: triangle abc (local coordinates) clipped to z in `zr`. */
WT_GPU_FN clipped_triangle_t clip_triangle_z(vec3 a, vec3 b, vec3 c, range zr) {
    const vec3 tri[3] = { a,b,c };
    // classify: 0 in range, +1 far, -1 near
    i32 cls[3];
    for (i32 i=0; i<3; ++i)
        cls[i] = tri[i].z>zr.max ? 1 : tri[i].z<zr.min ? -1 : 0;

    clipped_triangle_t ret;
    i32 idx = 0;
    const auto add_vertex = [&](vec3 v) {
        if (idx<5)
            ret.vs[idx++] = v;
    };
    for (i32 i=0; i<3; ++i) {
        const i32 next = i==2 ? 0 : i+1;
        // the current vertex, if inside the range
        if (cls[i]==0)
            add_vertex(tri[i]);
        // the crossing to the next, if any
        if (cls[next]!=cls[i]) {
            vec3 p;
            const real z = cls[i]==-1 ? zr.min : cls[i]==1 || cls[next]==1 ? zr.max : zr.min;
            add_vertex(intersect_edge_plane_z(tri[i], tri[next], z, p) ? p : cls[i]!=0 ? tri[i] : tri[next]);
            // crossing both planes
            if (cls[next]!=0 && cls[i]!=0)
                add_vertex(intersect_edge_plane_z(tri[i], tri[next], cls[next]==1 ? zr.max : zr.min, p) ? p :
                                                                                                          tri[next]);
        }
    }
    ret.tris = idx<3 ? 0 : idx==3 ? 1 : idx==4 ? 2 : 3;
    return ret;
}

/** @brief `elliptic_cone_t::project_local`: the local point `p` projected on the cross section at distance `z`. */
WT_GPU_FN vec2 project_local(const WT_GPU_THREAD cone& c, vec3 p, real z) {
    const vec2 xy = { p.x, p.y };
    if (c.x0==real(0) && c.tan_alpha==real(0))
        return xy;
    return xy * ((c.tan_alpha*z + c.x0) / abs(c.tan_alpha*p.z + c.x0));
}

/**
 * @brief `find_closest_triangle`'s integration (`integrate_tris`): the flux
 *        of the beam `envelope`'s wavefront `wf` over triangles `tris` (BVH
 *        triangles) in the interaction region `zr` (distances from the
 *        envelope's origin), but those facing otherwise than `front_face`.
 */
WT_GPU_FN real wavefront_flux_over_triangles(const WT_GPU_THREAD plt_scene& scene,
                                             const WT_GPU_GLOBAL u32* tris, u32 count,
                                             const WT_GPU_THREAD cone& envelope, gaussian_wavefront wf,
                                             range zr, bool front_face) {
    const frame f = envelope.frame();
    const vec3 o = envelope.r.o;
    const real csz = (zr.min + zr.max)/2;
    real flux = 0;
    for (u32 i=0; i<count; ++i) {
        const bvh_tri t = scene.bvh.tris[tris[i]];
        if ((dot(t.n, -envelope.r.d)>real(0)) != front_face)
            continue;

        // clip, and integrate over each clipped triangle projected on the cross section
        const clipped_triangle_t ct = clip_triangle_z(f.to_local(t.a-o), f.to_local(t.b-o), f.to_local(t.c-o), zr);
        for (i32 j=0; j<ct.tris; ++j) {
            vec3 a, b, c;
            ct.triangle(j, a, b, c);
            flux += integrate_triangle(wf, project_local(envelope, a, csz), project_local(envelope, b, csz),
                                       project_local(envelope, c, csz));
        }
    }
    return flux;
}

}
