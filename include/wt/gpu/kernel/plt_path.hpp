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
#include "fsd.hpp"
#include "rng.hpp"
#include "scene.hpp"
#include "telescope.hpp"
#include "texture.hpp"

/*
 * plt_path on the GPU: a port of `integrate_forward`, `integrate_backward` and
 * `random_walk` of wt/integrator/plt_path/plt_path_detail.hpp (with
 * `traverse` of wt/integrator/traversal.hpp), the recursion unrolled into
 * steps. Beams propagate ballistically and diffusively, diffract at edges
 * (free-space diffraction), and reflect and refract at specular and diffuse
 * surfaces. Forward transport sources beams at point emitters and splats
 * where they cross a virtual plane sensor; backward transport sources beams
 * at a virtual plane, perspective or telescope sensor and connects diffuse surfaces to
 * point emitters (next-event estimation). The host refuses other scenes
 * (src/gpu/renderer.cpp).
 *
 * A path's state between steps is a `path_state`: `start_path` starts one,
 * `step_path` advances it a step, in two parts (`trace_step`, then
 * `interact`) that backends may run in separate kernels, and `end_path` ends
 * it. The functions are parameterized by the backend's
 *   Geometry: `ray_hit intersect(const ray&, range) const`,
 *             `bool occluded(const ray&, range, u32& status) const`,
 *             `cone_hit intersect_cone(const cone&, range, real z_scale, tri_sink&) const`;
 *   Film:     `void add(u32 index, real value)`, accumulating film elements,
 *             and `void add_weight(u32 index, real weight)`, accumulating
 *             their reconstruction weights (backward transport only).
 *
 * Cone queries return triangle lists, which paths keep in fixed-capacity
 * scratch memory. A path that outgrows it stops with `status_scratch_overflow`
 * and reports how many splats it made, and how much more memory it needs
 * (`scratch_overflow`); as its random numbers depend only on its index, the
 * backend re-traces it with that much more room, skipping those splats.
 */

namespace wt { namespace gpu { namespace compute_kernel {

/** @brief A path outgrew its scratch memory (see above). */
WT_GPU_CONSTANT constexpr u32 status_scratch_overflow = 2;
/** @brief A path made `max_path_iterations` steps (counting null interactions) and was stopped. */
WT_GPU_CONSTANT constexpr u32 status_iteration_limit = 4;

/**
 * @brief With `status_scratch_overflow`, status bits 8 to 23: bit 8+j says
 *        that memory the path outgrew must grow 4^j-fold (j>=1) to hold what
 *        a query found. A path's statuses are or-ed together; the largest
 *        set bit says how much its re-trace needs (`scratch_growth`). The
 *        backends grow all of a path's memory (its triangles, edges, aperture
 *        and plt_bdpt vertices) in proportion.
 */
WT_GPU_CONSTANT constexpr u32 status_scratch_growth_shift = 8;
WT_GPU_CONSTANT constexpr u32 status_scratch_growth_bits = 16;
WT_GPU_CONSTANT constexpr u32 status_scratch_growth_mask =
    ((1u<<status_scratch_growth_bits)-1) << status_scratch_growth_shift;

/**
 * @brief The status of a query that needed `needed` entries of memory of
 *        `capacity` (more than it has): `status_scratch_overflow`, and the
 *        fourfold growths that hold them.
 */
WT_GPU_FN u32 scratch_overflow(u64 needed, u32 capacity) {
    u32 j = 1;
    for (u64 c=u64(capacity)*4; c<needed && j+1<status_scratch_growth_bits; c*=4)
        ++j;
    return status_scratch_overflow | (1u << (status_scratch_growth_shift+j));
}

/**
 * @brief The fourfold growths of its memory that a path with status `status`
 *        needs (`scratch_overflow`): at least 1 if it overflowed, 0 if not.
 */
WT_GPU_FN u32 scratch_growth(u32 status) {
    if (!(status & status_scratch_overflow))
        return 0;
    u32 j = 0;
    for (u32 b=(status & status_scratch_growth_mask) >> status_scratch_growth_shift; b>1; b>>=1)
        ++j;
    return j>0 ? j : 1;
}

/** @brief The scratch capacity that a path that overflowed memory of `capacity` with status `status` needs. */
WT_GPU_FN u32 scratch_capacity_needed(u32 capacity, u32 status) {
    const u64 c = u64(capacity) << (2*scratch_growth(status));
    return c<0xffffffffull ? u32(c) : 0xffffffffu;
}

/**
 * @brief Steps a path may take, counting null interactions, which do not
 *        count towards `max_depth`. (The CPU recurses without a limit.)
 */
WT_GPU_CONSTANT constexpr u32 max_path_iterations = 1u<<14;

/**
 * @brief A path's scratch memory: `tri_capacity` triangles, `edge_capacity`
 *        aperture entries, and `edges` of 3*`tri_capacity` entries (the
 *        edges of as many triangles, before duplicates are dropped).
 */
struct path_scratch {
    WT_GPU_GLOBAL u32* tris;
    WT_GPU_GLOBAL u32* edges;
    WT_GPU_GLOBAL aperture_entry* aperture;
    u32 tri_capacity;
    u32 edge_capacity;
};

/** @brief `sampler_t::concentric_disk` */
WT_GPU_FN vec2 concentric_disk(vec2 u) {
    const vec2 offset = vec2{ 2*u.x-1, 2*u.y-1 };
    if (offset.x==real(0) && offset.y==real(0))
        return { 0,0 };
    real rr, theta;
    if (abs(offset.x)>abs(offset.y)) {
        rr = offset.x;
        theta = pi_4 * (offset.y / offset.x);
    } else {
        rr = offset.y;
        theta = pi_2 - pi_4 * (offset.x / offset.y);
    }
    return { rr*cos(theta), rr*sin(theta) };
}

/** @brief `sampler_t::cosine_hemisphere` (its density is `z/pi`) */
WT_GPU_FN vec3 cosine_hemisphere(vec2 u) {
    const vec2 d = concentric_disk(u);
    return { d.x, d.y, sqrt(max(real(0), 1-sqr(d.x)-sqr(d.y))) };
}

/** @brief `sampler_t::uniform_cone`: a direction in the cone about +z of `solid_angle` (uniform density). */
WT_GPU_FN vec3 uniform_cone(real solid_angle, vec2 u) {
    const real cos_theta_max = 1 - inv_two_pi*solid_angle;
    const real cos_theta = 1 + u.x*(cos_theta_max-1);
    const real sin_theta = sqrt(max(real(0), 1-sqr(cos_theta)));
    const real phi = two_pi*u.y;
    return { cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta };
}

/** @brief `sampler_t::uniform_sphere` */
WT_GPU_FN vec3 uniform_sphere(vec2 u) {
    const real z = 1 - 2*u.x;
    const real rr = sqrt(max(real(0), 1-sqr(z)));
    const real phi = two_pi*u.y;
    return { rr*cos(phi), rr*sin(phi), z };
}

/**
 * @brief `m::erf_lut`: erf, but exactly ±1 from |x|=3.5 on, as the CPU's table.
 *        (Inside, the table interpolates erf linearly, to ~1e-6.) The cut-off
 *        decides which film elements a splat reaches, which shows in
 *        logarithmically tonemapped films.
 */
WT_GPU_FN real erf_lut(real x) {
    return abs(x)>=real(3.5) ? sign(x) : erf(x);
}

/** @brief `gaussian1d_t::integrate` over [lo,hi] of a zero-mean Gaussian. */
WT_GPU_FN real gaussian_integrate(real sigma, real lo, real hi) {
    if (sigma==real(0))
        return lo<=real(0) && real(0)<=hi ? real(1) : real(0);
    const real n = inv_sqrt_two * (real(1)/sigma);
    return (erf_lut(hi*n) - erf_lut(lo*n)) / 2;
}

WT_GPU_FN stokes as_stokes(stokes S) { return S; }
WT_GPU_FN stokes as_stokes(real I) { return { I,0,0,0 }; }

/**
 * @brief `film_t::splat_direct`: splats `observed` (Stokes parameters: of a
 *        polarimetric film, all four; of a scalar one, the intensity) around
 *        element (ex,ey) with fractional offset `offset`, weighted by the
 *        reconstruction filter and the sensor's response.
 */
template <typename Film, typename Observed>
WT_GPU_FN void splat(const WT_GPU_THREAD film_desc& film, film_response response, WT_GPU_THREAD Film& sink,
                     u32 ex, u32 ey, vec2 offset, Observed observed) {
    const stokes watts = as_stokes(observed);
    const i32 r = film.rf_radius;

    // integrate the reconstruction filter, centred on the sample (at `offset`
    // from its element's centre), over the elements' footprints
    real wx[2*max_rf_radius+1], wy[2*max_rf_radius+1];
    for (i32 x=-r; x<=r; ++x) {
        wx[x+r] = gaussian_integrate(film.rf_sigma, real(x)-offset.x-real(.5), real(x)-offset.x+real(.5));
        wy[x+r] = gaussian_integrate(film.rf_sigma, real(x)-offset.y-real(.5), real(x)-offset.y+real(.5));
    }
    real tw = 0;
    for (i32 y=-r; y<=r; ++y)
    for (i32 x=-r; x<=r; ++x)
        tw += max(real(0), wx[x+r]*wy[y+r]);
    const real recp_tw = tw>real(0) ? real(1)/tw : real(0);

    const u32 n = film.polarimetric!=0 ? 4 : 1;
    for (u32 c=0; c<film.components; ++c) {
        const stokes val = watts * response.c[c];
        // avoid NaNs, infs and negatives
        const bool finite = isfinite(val.s0) && (n==1 || (isfinite(val.s1) && isfinite(val.s2) && isfinite(val.s3)));
        if (!(val.s0>real(0)) || !finite)
            continue;
        const real v[4] = { val.s0, val.s1, val.s2, val.s3 };

        for (i32 y=-r; y<=r; ++y)
        for (i32 x=-r; x<=r; ++x) {
            const i32 px = i32(ex)+x, py = i32(ey)+y;
            if (px<0 || py<0 || px>=i32(film.width) || py>=i32(film.height))
                continue;
            const real w = max(real(0), wx[x+r]*wy[y+r]) * recp_tw;
            const u32 idx = (u32(py)*film.width + u32(px))*film.components + c;
            for (u32 i=0; i<n; ++i)
                sink.add(idx*n + i, w*v[i]);
        }
    }
}

/**
 * @brief `film_t::splat` (block splats, backward transport and plt_bdpt):
 *        splats `observed` (as `splat`) and the reconstruction filter's
 *        weights around element (ex,ey); the film develops their ratio.
 *        Every sample splats, zero or not.
 */
template <typename Film, typename Observed>
WT_GPU_FN void splat_weighted(const WT_GPU_THREAD film_desc& film, film_response response, WT_GPU_THREAD Film& sink,
                              u32 ex, u32 ey, vec2 offset, Observed observed) {
    const stokes watts = as_stokes(observed);
    const i32 r = film.rf_radius;

    real wx[2*max_rf_radius+1], wy[2*max_rf_radius+1];
    for (i32 x=-r; x<=r; ++x) {
        wx[x+r] = gaussian_integrate(film.rf_sigma, real(x)-offset.x-real(.5), real(x)-offset.x+real(.5));
        wy[x+r] = gaussian_integrate(film.rf_sigma, real(x)-offset.y-real(.5), real(x)-offset.y+real(.5));
    }
    real tw = 0;
    for (i32 y=-r; y<=r; ++y)
    for (i32 x=-r; x<=r; ++x)
        tw += max(real(0), wx[x+r]*wy[y+r]);
    const real recp_tw = tw>real(0) ? real(1)/tw : real(0);

    const u32 n = film.polarimetric!=0 ? 4 : 1;
    for (u32 c=0; c<film.components; ++c) {
        stokes val = watts * response.c[c];
        // avoid NaNs, infs and negatives (but sum up the weights)
        const bool finite = isfinite(val.s0) && (n==1 || (isfinite(val.s1) && isfinite(val.s2) && isfinite(val.s3)));
        if (!(val.s0>=real(0)) || !finite)
            val = { 0,0,0,0 };
        const real v[4] = { val.s0, val.s1, val.s2, val.s3 };

        for (i32 y=-r; y<=r; ++y)
        for (i32 x=-r; x<=r; ++x) {
            const i32 px = i32(ex)+x, py = i32(ey)+y;
            if (px<0 || py<0 || px>=i32(film.width) || py>=i32(film.height))
                continue;
            const real w = max(real(0), wx[x+r]*wy[y+r]) * recp_tw;
            const u32 idx = (u32(py)*film.width + u32(px))*film.components + c;
            for (u32 i=0; i<n; ++i)
                sink.add(idx*n + i, w*v[i]);
            sink.add_weight(idx, w);
        }
    }
}

/** @brief Truncation to an element index, saturating as the CPU (AArch64, CUDA) conversion does. */
WT_GPU_FN u32 to_element(real x) {
    return x>real(0) ? (x<real(4294967040.0) ? u32(x) : 4294967040u) : 0u;
}

struct sensor_hit_t {
    vec3 p;
    u32 ex, ey;
    vec2 offset;
    bool hit;
};

/**
 * @brief `virtual_plane_sensor_t::Si`: where the beam's central ray crosses
 *        the sensor's front within `rng` (distance from the beam origin).
 */
template <bool P>
WT_GPU_FN sensor_hit_t virtual_plane_hit(const WT_GPU_THREAD sensor_desc& s,
                                         const WT_GPU_THREAD beam_t<P>& b, range rng) {
    const sensor_hit_t none = { { 0,0,0 }, 0, 0, { 0,0 }, false };

    // check side
    const real dn = dot(-b.dir(), s.f.n);
    if (dn<=real(0))
        return none;

    const ray r = b.envelope.r;
    const auto i1 = intersect_ray_tri_scalar(r, s.quad[0], s.quad[1], s.quad[2], rng);
    const auto i2 = intersect_ray_tri_scalar(r, s.quad[2], s.quad[1], s.quad[3], rng);
    if (!i1.hit && !i2.hit)
        return none;

    // element_for_position
    const vec3 p = r.propagate(i1.hit ? i1.dist : i2.dist);
    const vec3 sp = p - s.origin;
    const vec2 efp = vec2{ dot(sp, s.f.t), dot(sp, s.f.b) } * s.recp_element_extent;
    const u32 ex = to_element(efp.x), ey = to_element(efp.y);
    const vec2 offset = efp - vec2{ real(ex), real(ey) } - vec2{ real(.5), real(.5) };

    return { p, ex, ey, offset, true };
}

struct bsdf_sample_t {
    vec3 wo;
    mueller_block M;        // weighted BSDF: bsdf/pdf
    real pdf;               // discrete probability (`discrete`), or solid-angle density
    bool discrete;          // a delta lobe was sampled
    real eta;
};

/*
 * Rough surface profiles (wt/interaction/surface_profile/{fractal,gaussian}.hpp):
 * the PSD of the scattered lobe, its sampling and its density, in the
 * dimensionless parameters of `material` (directions in the shading frame).
 */

/** @brief The profile's specular fraction (`surface_profile_t::alpha`), 1 for a smooth surface. */
WT_GPU_FN real profile_alpha(const WT_GPU_THREAD material& m, vec3 wi, vec3 wo) {
    if (m.profile==profile_dirac)
        return 1;
    return exp(-sqr(abs(wi.z)+abs(wo.z)) * m.alpha_k);
}

/** @brief The profile's PSD (`psd(params, z, k)`) at z = k*`v`, `v` the sum of the directions' xy. */
WT_GPU_FN real profile_psd(const WT_GPU_THREAD material& m, vec2 v) {
    const real v2 = dot(v,v);
    if (m.profile==profile_fractal) {
        const real x = 1 + m.K*v2;
        const real p = m.gamma==real(3) ? x*x : pow(x, (m.gamma+1)/2);
        return m.sigma2_norm * (inv_two_pi * (m.gamma-1) * m.K / p);
    }
    const real e = exp(-(v2/2 / m.s2));
    return e<=epsilon ? real(0) : m.sigma2_norm * (inv_two_pi/m.s2 * e);
}

struct profile_sample_t {
    vec3 wo;
    real pdf, psd;
};

/** @brief `sample_boxmueller_truncated` of gaussian.hpp: a 2D point about `mean`, and its density. */
WT_GPU_FN vec2 sample_boxmueller_truncated(vec2 u, vec2 mean, real sigma2, WT_GPU_THREAD real& pdf) {
    const real l = sqrt(min(real(1), dot(mean,mean)));
    const real coso = sqrt(max(real(0), 1-dot(mean,mean)));
    const real phi_i = mean.x!=real(0) || mean.y!=real(0) ? atan2(mean.y,mean.x) : real(0);

    const real s = exp(-real(.5)*sqr(1+l)/sigma2);
    const real x = (1-s)*max(epsilon,u.x)+s;
    const real r = sqrt(-2 * sigma2 * log(x));

    const real max_phi = r<epsilon || l<epsilon ? pi :
        max(real(1e-2), acos(clamp((sqr(r)+sqr(l)-1)/(2*r*l), real(-1), real(1))));
    const real phi = phi_i + pi + max_phi*(2*u.y-1);

    pdf = real(.5) * x/(max_phi*sigma2) * coso;
    return vec2{ r*cos(phi), r*sin(phi) } + mean;
}

/** @brief `boxmueller_truncated_pdf` of gaussian.hpp */
WT_GPU_FN real boxmueller_truncated_pdf(vec2 wo, vec2 mean, real sigma2) {
    const real l = sqrt(min(real(1), dot(mean,mean)));
    const real coso = sqrt(max(real(0), 1-dot(mean,mean)));

    wo = wo - mean;
    const real r2 = dot(wo,wo);
    const real x = exp(real(-.5) * r2/sigma2);
    const real r = sqrt(r2);
    const real max_phi = r<epsilon || l<epsilon ? pi :
        max(real(1e-2), acos(clamp((sqr(r)+sqr(l)-1)/(2*r*l), real(-1), real(1))));
    return real(.5) * x/(max_phi*sigma2) * coso;
}

/** @brief `fractal_t::sample` or `gaussian_t::sample`: a scattered direction, in `wi`'s hemisphere. */
WT_GPU_FN profile_sample_t sample_profile(const WT_GPU_THREAD material& m, vec3 wi, vec2 u) {
    vec2 wo2;
    real pdf, psd;
    if (m.profile==profile_fractal) {
        // "A Two-Scale Microfacet Reflectance Model Combining Reflection and Diffraction", Holzschuch & Pacanowski 2017
        const real s = sqrt(max(real(0), 1-sqr(wi.z)));
        const real phi_i = s>real(0) ? atan2(wi.y,wi.x) : real(0);
        const real M = 1 - pow(1 + m.K*sqr(1+s), -(m.gamma-1)/2);
        // the sampled spatial frequency over k
        const real f_k = sqrt(pow(1-M*u.x, -2/(m.gamma-1))-1) / sqrt(m.K);
        const real phi_max = f_k==real(0) || s==real(0) ? pi :
            acos(clamp((sqr(f_k) + sqr(s) - 1)/(2*f_k*s), real(-1), real(1)));
        const real phi_f = phi_i + (2*u.y-1)*phi_max;
        const vec2 zeta_k = vec2{ cos(phi_f), sin(phi_f) } * f_k;
        wo2 = zeta_k - vec2{ wi.x,wi.y };
        const real z = sqrt(max(real(0), 1-dot(wo2,wo2)));
        psd = profile_psd(m, zeta_k);
        const real w = inv_pi * phi_max;
        pdf = w>real(1e-2) ? z * psd / w : real(0);
    } else {
        const vec2 mean = -vec2{ wi.x,wi.y };
        wo2 = sample_boxmueller_truncated(u, mean, m.s2, pdf);
        psd = profile_psd(m, wo2-mean);
    }
    const real z = sqrt(max(real(0), 1-dot(wo2,wo2)));
    return { vec3{ wo2.x, wo2.y, wi.z>=real(0) ? z : -z }, pdf, psd };
}

/** @brief `fractal_t::pdf` or `gaussian_t::pdf` */
WT_GPU_FN real pdf_profile(const WT_GPU_THREAD material& m, vec3 wi, vec3 wo) {
    if (m.profile==profile_fractal) {
        const vec2 zeta_k = vec2{ wi.x,wi.y } + vec2{ wo.x,wo.y };
        const real f_k = length(zeta_k);
        const real s = sqrt(max(real(0), 1-sqr(wi.z)));
        const real phi_max = f_k==real(0) || s==real(0) ? pi :
            acos(clamp((sqr(f_k) + sqr(s) - 1)/(2*f_k*s), real(-1), real(1)));
        const real w = inv_pi * phi_max;
        return w>real(1e-2) ? 1/w * abs(wo.z) * profile_psd(m, zeta_k) : real(0);
    }
    return boxmueller_truncated_pdf(vec2{ wo.x,wo.y }, -vec2{ wi.x,wi.y }, m.s2);
}

/** @brief `flip_wo` of src/bsdf/surface_spm.cpp: a scattered direction refracted to the other side. */
WT_GPU_FN vec3 flip_wo(vec3 wo, real eta) {
    const real scale = wo.z>real(0) ? eta : 1/eta;
    const vec2 xy = vec2{ wo.x,wo.y } * scale;
    const real l2 = dot(xy,xy);
    return l2>real(1) ? vec3{ 1,0,0 } :
                        vec3{ xy.x, xy.y, (wo.z>real(0) ? real(-1) : real(1))*sqrt(max(real(0), 1-l2)) };
}

/** @brief The microfacet normal of a scattered pair (surface_spm_t): `wi`+`wo`, in `wi`'s hemisphere. */
WT_GPU_FN vec3 half_vector(vec3 wi, vec3 wo) {
    const vec3 h = wi + wo;
    return normalize(wi.z<real(0) ? -h : h);
}

/**
 * @brief Samples a material's BSDF (`bsdf_t::sample`) for incident direction
 *        `wi` (shading frame, pointing away from the surface).
 */
WT_GPU_FN bool sample_bsdf(const WT_GPU_THREAD material& m, vec3 wi, bool backward, WT_GPU_THREAD rng& rnd,
                           WT_GPU_THREAD bsdf_sample_t& out) {
    // two_sided_t: the back side mirrors the front side
    const bool flip = m.two_sided!=0 && wi.z<real(0);
    if (flip)
        wi.z = -wi.z;

    vec3 wo;
    if (m.type==material_diffuse) {
        // diffuse_t::sample
        if (wi.z<=real(0))
            return false;
        wo = cosine_hemisphere(rnd.r2());
        out.M = mueller_block::depolarizer(m.reflectance) * m.scale;
        out.pdf = inv_pi * wo.z;
        out.discrete = false;
        out.eta = 1;
    } else if (m.type==material_specular) {
        // surface_spm_t::sample: the specular fraction `alpha` of the surface profile (1 if smooth) is
        // specular, the rest scattered
        const real alpha = profile_alpha(m, wi, wi);
        const bool has_specular = alpha>real(0);
        const bool has_scatter = alpha<real(1);
        if (wi.z==real(0) || (!has_specular && !has_scatter))
            return false;

        real pdf = 1;
        bool is_specular = has_specular;
        if (has_specular && has_scatter) {
            is_specular = alpha==real(1) || rnd.r()<alpha;
            pdf = is_specular ? alpha : 1-alpha;
        }

        const fresnel_t f = fresnel(m.eta, wi);
        const real pdf_transmission = (f.Ts+f.Tp)/2;
        bool is_reflection = true;
        if (m.has_transmission!=0) {
            is_reflection = rnd.r()>=pdf_transmission;
            pdf *= is_reflection ? 1-pdf_transmission : pdf_transmission;
        }

        // J = eta² for refraction in backward transport
        const real J = !is_reflection && backward ? sqr(f.eta_12) : real(1);
        const real scale = is_reflection ? m.reflection_scale : m.transmission_scale;
        if (scale==real(0) || (!is_reflection && m.has_transmission==0))
            return false;

        if (is_specular) {
            wo = is_reflection ? vec3{ -wi.x, -wi.y, wi.z } : f.t;
            out.M = (mueller_fresnel(m.eta, is_reflection, wi) * (alpha * J * scale * m.scale)) / pdf;
            out.pdf = pdf;
            out.discrete = true;
        } else {
            // scattered by the surface profile, Fresnel about the microfacet normal
            const profile_sample_t s = sample_profile(m, wi, rnd.r2());
            const mueller_block F = mueller_fresnel(m.eta, is_reflection, wi, half_vector(wi, s.wo));
            wo = is_reflection ? s.wo : flip_wo(s.wo, m.eta.re);
            pdf *= s.pdf;
            out.M = (F * ((1-alpha) * J * abs(wo.z) * s.psd * scale * m.scale)) / pdf;
            out.pdf = pdf;
            out.discrete = false;
        }
        out.eta = is_reflection ? real(1) : f.eta_12;
    } else if (m.type==material_dielectric) {
        // dielectric_t::sample: Fresnel reflection or refraction, by the transmissivity
        const fresnel_t f = fresnel(m.eta, wi);
        const real T = (f.Ts+f.Tp)/2;
        const bool is_reflection = rnd.r()>=T;
        const real pdf = is_reflection ? 1-T : T;
        const real scale = is_reflection ? m.reflection_scale : m.transmission_scale;
        if (scale==real(0))
            return false;

        mueller_block M;
        if (is_reflection) {
            wo = { -wi.x, -wi.y, wi.z };
            M = mueller_fresnel(complex{ f.rs,0 }, complex{ f.rp,0 }) * scale;
        } else {
            // J = eta² for refraction in backward transport
            wo = f.t;
            M = mueller_fresnel(complex{ f.ts,0 }, complex{ f.tp,0 }) *
                (f.Z * scale * (backward ? sqr(f.eta_12) : real(1)));
        }
        out.M = (M * m.scale) / pdf;
        out.pdf = pdf;
        out.discrete = true;
        // (the CPU's, also of reflections)
        out.eta = f.eta_12;
    } else {
        return false;
    }

    out.wo = flip ? vec3{ wo.x, wo.y, -wo.z } : wo;
    return true;
}

/**
 * @brief Evaluates a material's BSDF (`bsdf_t::f`, with the cosine term) for
 *        `wi` and `wo` (shading frame, pointing away from the surface). Zero
 *        for delta BSDFs, which `is_delta` tells.
 */
WT_GPU_FN mueller_block eval_bsdf(const WT_GPU_THREAD material& m, vec3 wi, vec3 wo, bool backward) {
    // two_sided_t
    if (m.two_sided!=0 && wi.z<real(0)) {
        wi.z = -wi.z;
        wo.z = -wo.z;
    }
    if (m.type==material_diffuse && wi.z>real(0) && wo.z>real(0))
        return mueller_block::depolarizer(wo.z * inv_pi * m.reflectance * m.scale);

    if (m.type==material_specular && m.delta_only==0) {
        // surface_spm_t::f: the scattered lobe
        const bool is_reflection = wi.z*wo.z>=real(0);
        if (wi.z==real(0) || wo.z==real(0) || (!is_reflection && m.has_transmission==0))
            return mueller_block::depolarizer(0);

        const vec3 abs_wo = is_reflection ? wo : flip_wo(wo, m.eta.re);
        const real alpha = profile_alpha(m, wi, abs_wo);
        const real J = !is_reflection && backward ? sqr(wi.z<real(0) ? 1/m.eta.re : m.eta.re) : real(1);
        const real scale = is_reflection ? m.reflection_scale : m.transmission_scale;
        // (the CPU evaluates Fresnel with the real part of the refractive index here)
        const mueller_block F = mueller_fresnel(complex{ m.eta.re,0 }, is_reflection, wi, half_vector(wi, abs_wo));
        const real psd = profile_psd(m, vec2{ wi.x,wi.y } + vec2{ abs_wo.x,abs_wo.y });
        return F * ((1-alpha) * J * abs(wo.z) * psd * scale * m.scale);
    }
    return mueller_block::depolarizer(0);
}

/** @brief `bsdf_t::pdf`: the solid-angle density of `sample_bsdf` sampling `wo` (0 for delta BSDFs). */
WT_GPU_FN real pdf_bsdf(const WT_GPU_THREAD material& m, vec3 wi, vec3 wo) {
    // two_sided_t
    if (m.two_sided!=0 && wi.z<real(0)) {
        wi.z = -wi.z;
        wo.z = -wo.z;
    }
    if (m.type==material_diffuse && wi.z>real(0) && wo.z>real(0))
        return inv_pi * wo.z;

    if (m.type==material_specular && m.profile!=profile_dirac) {
        // surface_spm_t::pdf
        const bool is_reflection = wi.z*wo.z>=real(0);
        if (wi.z==real(0) || wo.z==real(0) || (!is_reflection && m.has_transmission==0))
            return 0;
        const vec3 abs_wo = is_reflection ? wo : flip_wo(wo, m.eta.re);
        const real alpha = profile_alpha(m, wi, wi);
        const fresnel_t f = fresnel(complex{ m.eta.re,0 }, wi);
        const real pdf_transmission = (f.Ts+f.Tp)/2;
        return (1-alpha) * pdf_profile(m, wi, abs_wo) * (is_reflection ? 1-pdf_transmission : pdf_transmission);
    }
    return 0;
}

/** @brief `bsdf_t::is_delta_only`: no BSDF evaluation or next-event estimation. */
WT_GPU_FN bool is_delta(const WT_GPU_THREAD material& m) {
    return m.delta_only!=0;
}

/*
 * Path vertices, visibility
 */

WT_GPU_CONSTANT constexpr u32 geo_point = 0;
WT_GPU_CONSTANT constexpr u32 geo_surface = 1;
WT_GPU_CONSTANT constexpr u32 geo_edge = 2;

/** @brief A path vertex's geometry (`vertex_geo_variant_t`, or an `intersection_edge_t`). */
struct vertex_geo {
    vec3 wp;
    u32 kind;       // geo_point, geo_surface or geo_edge
    u32 index;      // BVH triangle of a surface, or edge
};

/** @brief `intersection_edge_t::offseted_ray_origin`: offsets away from the wedge. */
WT_GPU_FN vec3 edge_offseted_ray_origin(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD edge& ed, ray r) {
    const bool has_t2 = ed.tri2!=no_triangle;
    vec3 dir;
    if (!has_t2) {
        dir = -ed.t1;
    } else {
        const vec3 v = ed.t1 + ed.t2;
        dir = length2(v)>real(1e-14) ? -normalize(v) : -ed.t2;
    }

    const bvh_tri t1 = scene.bvh.tris[ed.tri1];
    real d = dot(intersection_triangle_fp_errors(t1.a, t1.b, t1.c, r.o), abs(ed.t1));
    if (has_t2) {
        const bvh_tri t2 = scene.bvh.tris[ed.tri2];
        d = max(d, dot(intersection_triangle_fp_errors(t2.a, t2.b, t2.c, r.o), abs(ed.t2)));
    }
    return r.o + d*dir;
}

/** @brief `offseted_ray_origin(vertex_geo, ray)` of wt/integrator/traversal.hpp */
WT_GPU_FN vec3 offseted_ray_origin(const WT_GPU_THREAD plt_scene& scene, vertex_geo g, ray r) {
    if (g.kind==geo_surface) {
        const bvh_tri t = scene.bvh.tris[g.index];
        return offseted_ray_origin(t.a, t.b, t.c, t.n, r);
    }
    if (g.kind==geo_edge) {
        const edge ed = scene.edges[g.index];
        return edge_offseted_ray_origin(scene, ed, r);
    }
    return r.o;
}

/** @brief A shadow ray: a segment between two vertices, which is blocked if the ray hits anything in its range. */
struct shadow_ray_t {
    ray r;
    range rng;
};

/** @brief The ray of `shadow(ads, a, b)` of wt/integrator/traversal.hpp, between two vertices, offset off both. */
WT_GPU_FN shadow_ray_t shadow_ray(const WT_GPU_THREAD plt_scene& scene, vertex_geo a, vertex_geo b) {
    const ray r = make_ray(a.wp, normalize(b.wp-a.wp));
    const vec3 o = offseted_ray_origin(scene, a, r);
    const vec3 t = offseted_ray_origin(scene, b, make_ray(b.wp, -r.d));
    const real dist = length(t-o);
    const vec3 d = (t-o)/dist;
    return { make_ray(o, d), range{ 0, dist } };
}

/** @brief `shadow(ads, a, b)` of wt/integrator/traversal.hpp: is the segment between two vertices blocked? */
template <typename Geometry>
WT_GPU_FN bool shadowed(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                        vertex_geo a, vertex_geo b, WT_GPU_THREAD u32& status) {
    const shadow_ray_t s = shadow_ray(scene, a, b);
    return geometry.occluded(s.r, s.rng, status);
}


/*
 * Traversal
 */

/** @brief Result of `traverse()` (`traversal_result_t`). Aligned for 16-byte copies to and from memory. */
struct alignas(16) traversal_t {
    vec3 origin;            // traversal origin (offset for self-intersection avoidance)
    real dist;              // distance to the intersection
    real region_depth;      // depth past `dist` over which triangles are considered
    bool hit;
    bool ballistic;
    bool front_face;        // diffusive, if requested (see `traverse`): the closest triangle faces the beam
    // ballistic: the triangle hit
    u32 tri;
    real baryx, baryy;
    // the triangles in the scratch memory: diffusive, those intersected;
    // ballistic, those around the intersection (see `trace_step`)
    u32 tri_count;
};

/**
 * @brief `traverse()` of wt/integrator/traversal.hpp: short ballistic (ray)
 *        segments, after each of which diffusive (cone) propagation is tried.
 *        `envelope` starts at the (offset) traversal origin. Sets
 *        `status_scratch_overflow` if the triangles do not fit. Finds whether
 *        the closest triangle faces the beam (`traversal_t::front_face`) only
 *        if asked to (`FrontFace`): tracking it slows the cone traversals.
 */
template <bool FrontFace = false, typename Geometry>
WT_GPU_FN traversal_t traverse(const WT_GPU_THREAD Geometry& geometry, const WT_GPU_THREAD cone& envelope,
                               real lambda, bool ray_trace, const WT_GPU_THREAD path_scratch& scratch,
                               WT_GPU_THREAD u32& status) {
    const ray r = envelope.r;
    traversal_t ret = { r.o, inf(), 0, false, true, false, no_triangle, 0, 0, 0 };

    const auto ballistic_hit = [&](const WT_GPU_THREAD ray_hit& h) {
        ret.hit = true;
        ret.dist = h.dist;
        ret.tri = h.tri;
        ret.baryx = h.baryx;
        ret.baryy = h.baryy;
    };

    if (ray_trace || envelope.is_ray()) {
        // pure ray tracing
        const ray_hit h = geometry.intersect(r, range::positive());
        status |= h.status;
        if (h.hit) ballistic_hit(h);
        return ret;
    }

    // The ballistic origin is the envelope's, so the minimal ballistic distance
    // (calculate_min_ballistic_distance) is 0.
    real dist = 0;
    for (u32 seg=0;; ++seg) {
        // max_ballistic_distance: 16, 64, 256, ... wavelengths, up to 65536, and infinity from the 16th segment
        const real lambdas = seg<7 ? real(8u<<(2*seg+1)) : real(65536);
        const real ballistic_dist = seg>=16 ? inf() : lambda*lambdas;
        const ray_hit h = geometry.intersect(r, range{ dist, dist + ballistic_dist*real(1.001) });
        status |= h.status;
        if (h.hit) {
            ballistic_hit(h);
            return ret;
        }

        // no intersection found: propagate the beam
        dist += ballistic_dist;
        if (!isfinite(ballistic_dist) || !isfinite(dist))
            return ret;

        // attempt diffusive propagation
        const real min_df_prog = envelope.axes(dist).x / 2;
        tri_sink sink = { scratch.tris, scratch.tri_capacity, 0 };
        const cone_hit ch = geometry.intersect_cone(envelope, range{ dist, inf() }, 2, sink);
        status |= ch.status;
        if (ch.tri_count>scratch.tri_capacity) {
            status |= scratch_overflow(ch.tri_count, scratch.tri_capacity);
            return ret;
        }

        // successful diffusive propagation?
        const bool empty = ch.tri_count==0;
        if (empty || ch.dist-dist>=min_df_prog) {
            ret.origin = envelope.r.o;
            ret.ballistic = false;
            ret.hit = !empty;
            ret.dist = ch.dist;
            ret.tri_count = ch.tri_count;
            if constexpr (FrontFace)
                ret.front_face = ch.front_face!=0;
            ret.region_depth = empty ? real(0) : 2*envelope.axes(ch.dist).x;
            return ret;
        }
        // ... or, too short, continue ballistic path
    }
}

/** @brief `intersect::cone_intersection_tolerance`: a rough bound of cone-test floating-point error. */
WT_GPU_FN real cone_intersection_tolerance(vec3 origin, vec3 a, vec3 b, vec3 c) {
    const real c0 = real(4e-7);
    const real c1 = real(1e-6);
    const real c2 = real(1e-6);

    // max world extent of the primitive's AABB
    const vec3 mn = { min(a.x,b.x,c.x), min(a.y,b.y,c.y), min(a.z,b.z,c.z) };
    const vec3 mx = { max(a.x,b.x,c.x), max(a.y,b.y,c.y), max(a.z,b.z,c.z) };
    const real obj_extent = 2 * max(max_element(abs(mn)), max_element(abs(mx)));

    const vec3 obj_err = (c0+c2)*abs(origin) + vec3{ c1*obj_extent, c1*obj_extent, c1*obj_extent };
    const vec3 wrd_err = (c1+c2)*abs(origin);
    return max_element(obj_err + wrd_err);
}

struct primary_t {
    u32 tri;            // no_triangle if none
    real dist;
    real baryx, baryy;
};

/** @brief `find_closest_triangle`: the closest of `tris` under the interaction region, along the beam's axis. */
WT_GPU_FN primary_t find_closest_triangle(const WT_GPU_THREAD plt_scene& scene,
                                          const WT_GPU_GLOBAL u32* tris, u32 count,
                                          range z_range, vec3 origin, vec3 dir) {
    primary_t ret = { no_triangle, inf(), 0, 0 };
    const ray r = make_ray(origin, dir);
    for (u32 i=0; i<count; ++i) {
        const bvh_tri t = scene.bvh.tris[tris[i]];
        const real fptol = cone_intersection_tolerance(origin, t.a, t.b, t.c);
        const auto h = intersect_ray_tri_scalar(r, t.a, t.b, t.c, z_range.grow(fptol));
        if (h.hit && h.dist<ret.dist)
            ret = { tris[i], h.dist, h.baryx, h.baryy };
    }
    return ret;
}

/** @brief Sorts `v[0..n-1]` ascending (Shell sort, with Ciura's gaps extended by 2.25×). */
WT_GPU_FN void shell_sort(WT_GPU_GLOBAL u32* v, u32 n) {
    const u32 gaps[] = { 79749, 35444, 15753, 7001, 3112, 1383, 701, 301, 132, 57, 23, 10, 4, 1 };
    for (const u32 gap : gaps) {
        for (u32 i=gap; i<n; ++i) {
            const u32 x = v[i];
            u32 j = i;
            for (; j>=gap && v[j-gap]>x; j-=gap)
                v[j] = v[j-gap];
            v[j] = x;
        }
    }
}

/**
 * @brief The edges of `tris` (`cone_work_to_intersection_record`), sorted and
 *        unique as the CPU's set, into `edges` (which holds 3*`tri_count`
 *        entries). Returns false if more than `capacity` remain.
 *        (Collecting all and sorting once is much faster than inserting each
 *        into a sorted list.)
 */
WT_GPU_FN bool collect_edges(const WT_GPU_THREAD plt_scene& scene,
                             const WT_GPU_GLOBAL u32* tris, u32 tri_count,
                             WT_GPU_GLOBAL u32* edges, u32 capacity, WT_GPU_THREAD u32& count) {
    u32 n = 0;
    for (u32 i=0; i<tri_count; ++i) {
        const tri_shading sh = scene.tris[tris[i]];
        for (int j=0; j<3; ++j)
            if (sh.edges[j]!=no_edge)
                edges[n++] = sh.edges[j];
    }
    shell_sort(edges, n);

    // drop duplicates
    count = 0;
    for (u32 i=0; i<n; ++i)
        if (count==0 || edges[i]!=edges[count-1])
            edges[count++] = edges[i];
    return count<=capacity;
}


/*
 * Free-space diffraction
 */

struct fsd_amplitudes_t {
    complex ts, th;

    /** @brief The diffracted intensity (unpolarized: the mean of the soft and hard intensities). */
    WT_GPU_FN real intensity() const { return (norm(ts) + norm(th)) / 2; }
};

/**
 * @brief `do_fsd` of plt_path_detail.hpp: the (soft and hard) amplitudes
 *        reaching `dst` from the source of `cone_from_src` through the
 *        aperture's edges, plus the direct path if `dst` is in the cone.
 */
template <typename Geometry>
WT_GPU_FN fsd_amplitudes_t do_fsd(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                                  real k, const WT_GPU_THREAD cone& cone_from_src, vertex_geo src_geo, vec3 dst,
                                  const WT_GPU_GLOBAL aperture_entry* aperture, u32 aperture_size,
                                  WT_GPU_THREAD u32& status) {
    const vec3 src = cone_from_src.r.o;
    const vertex_geo dst_geo = { dst, geo_point, 0 };

    fsd_amplitudes_t ret = { { 0,0 }, { 0,0 } };
    for (u32 i=0; i<aperture_size; ++i) {
        // free_space_diffraction_t::f
        const wedge w = make_wedge(scene.edges, aperture[i]);
        vec3 p;
        if (!diffraction_point(w, src, dst, p))
            continue;

        const vec3 ui = src-p;
        const vec3 uo = dst-p;
        // ignore into wedge rays
        if ((dot(uo,w.nff)<=real(0) && dot(uo,w.nbf)<=real(0)) ||
            (dot(ui,w.nff)<=real(0) && dot(ui,w.nbf)<=real(0)))
            continue;

        const real ri = length(ui);
        const real ro = length(uo);
        const utd_t f = utd(w, k, ui/ri, uo/ro, ro);
        if (f.Dh==complex{ 0,0 } && f.Ds==complex{ 0,0 })
            continue;

        const vertex_geo eintr = { p, geo_edge, w.ads_edge };
        if (shadowed(scene, geometry, eintr, src_geo, status) ||
            shadowed(scene, geometry, eintr, dst_geo, status))
            continue;

        const complex phase = expi(-((ro + ri)*k));
        ret.ts = ret.ts + phase*f.Ds;
        ret.th = ret.th + phase*f.Dh;
    }

    if (cone_contains(cone_from_src, dst)) {
        if (!shadowed(scene, geometry, src_geo, dst_geo, status)) {
            // direct path
            const complex phase = expi(-(length(dst - src)*k));
            ret.ts = ret.ts + phase;
            ret.th = ret.th + phase;
        }
    }

    return ret;
}


/*
 * The path
 */

/**
 * @brief A direction transformed by row-major `M` (a direction transform of
 *        the host; a pointer to its 9 elements), normalized.
 */
template <typename Matrix>
WT_GPU_FN vec3 transform_direction(Matrix M, vec3 v) {
    return normalize(vec3{
        M[0]*v.x + M[1]*v.y + M[2]*v.z,
        M[3]*v.x + M[4]*v.y + M[5]*v.z,
        M[6]*v.x + M[7]*v.y + M[8]*v.z,
    });
}

/** @brief The point (v.x, v.y, 1, 1) transformed by row-major `M` (a pointer to its 16 elements), projected. */
template <typename Matrix>
WT_GPU_FN vec3 transform_projective(Matrix M, vec2 v) {
    real q[4];
    for (int r=0; r<4; ++r)
        q[r] = M[4*r]*v.x + M[4*r+1]*v.y + M[4*r+2] + M[4*r+3];
    return vec3{ q[0], q[1], q[2] } / q[3];
}

/** @brief `perspective_t::point_on_sensor`: the camera-space point (at z=1) of a film position. */
WT_GPU_FN vec3 point_on_sensor(const WT_GPU_THREAD sensor_desc& s, vec2 film_pos) {
    return transform_projective(s.to_camera, film_pos);
}

/** @brief `perspective_t::point_on_film`: the film position of a camera-space direction. */
WT_GPU_FN vec2 point_on_film(const WT_GPU_THREAD sensor_desc& s, vec3 dir) {
    const vec3 p = dir / abs(dir.z);
    const vec3 f = transform_projective(s.to_film, vec2{ p.x, p.y });
    return { f.x, f.y };
}

/** @brief `perspective_t::recp_solid_angle_density_for_sample`, of a camera-space direction. */
WT_GPU_FN real perspective_recp_dpd(const WT_GPU_THREAD sensor_desc& s, vec3 dir_local) {
    return s.recp_sa_scale * (dir_local.z*dir_local.z*dir_local.z);
}

/**
 * @brief `sensor_t::sample_direct`: a sensor point connecting to `wp`, its
 *        element, and the importance of the connection.
 */
struct sensor_direct_t {
    vec3 p;
    vec3 wd;            // from the sensor point towards `wp`
    u32 ex, ey;
    vec2 offset;
    real importance;    // the sampled beam's (`Se` over the sampling density); 0 if not connectible
};

WT_GPU_FN sensor_direct_t sensor_sample_direct(const WT_GPU_THREAD sensor_desc& s, const WT_GPU_THREAD film_desc& film,
                                               vec3 wp, WT_GPU_THREAD rng& rnd) {
    sensor_direct_t ret;

    if (s.type==sensor_perspective) {
        // perspective_t::sample_direct: the sensor's position, the film position the direction to `wp` images to
        ret.p = s.position;
        const vec3 wdl = wp - s.position;
        const real recp_dist2 = 1 / length2(wdl);
        ret.wd = wdl * sqrt(recp_dist2);
        const vec3 dir_local = transform_direction(s.dir_to_local, ret.wd);
        const vec2 film_pos = point_on_film(s, dir_local);
        ret.ex = to_element(film_pos.x);
        ret.ey = to_element(film_pos.y);
        ret.offset = film_pos - vec2{ floor(film_pos.x), floor(film_pos.y) } - vec2{ real(.5), real(.5) };

        // Se(P, recp_sa) * recp_dist2, within the sensor extent
        const bool inside = dir_local.z>epsilon && ret.ex<film.width && ret.ey<film.height;
        ret.importance = inside ? 1/perspective_recp_dpd(s, dir_local) * recp_dist2 : real(0);
        return ret;
    }

    // virtual_plane_sensor_t::sample_direct: sample point on sensor
    const vec2 splocal = rnd.r2() * s.extent;
    ret.p = s.origin + splocal.x*s.f.t + splocal.y*s.f.b;

    // point's element and element offset
    const vec2 efp = { splocal.x/s.element_extent.x, splocal.y/s.element_extent.y };
    ret.ex = to_element(efp.x);
    ret.ey = to_element(efp.y);
    ret.offset = efp - vec2{ real(ret.ex), real(ret.ey) } - vec2{ real(.5), real(.5) };

    // direction
    const vec3 wdl = wp - ret.p;
    const real dist2 = length2(wdl);
    ret.wd = wdl / sqrt(dist2);
    const real wdz = dot(ret.wd, s.f.n);
    const real recp_dn = wdz>real(0) ? 1/wdz : real(0);

    const real dpd = s.recp_area * dist2 * recp_dn;
    const real recp_dpd = dpd>real(0) ? 1/dpd : real(0);
    // Se(P,k) * recp_dpd * recp_dn
    ret.importance = dpd!=real(0) ? s.importance * max(real(0), wdz) * recp_dpd * recp_dn : real(0);
    return ret;
}

/**
 * @brief A path between steps: its beam and walk data (`path_walk_data_t`).
 *        Its aperture, when it has one, is in the path's scratch memory.
 *        Aligned for 16-byte copies to and from memory. `Polarimetric`: of a
 *        polarimetric sensor (backward transport), whose beams keep their
 *        whole Mueller operators, and which observes Stokes parameters.
 */
template <bool Polarimetric>
struct alignas(16) path_state_t {
    beam_t<Polarimetric> b;
    beam_t<Polarimetric> prev_beam;
    vertex_geo prev_geo;
    rng rnd;
    real throughput;
    u32 depth;
    u32 iteration;
    u32 aperture_size;
    u32 fsd;                // nonzero: diffract through the aperture at the next interaction
    u32 sampled_fsd;        // nonzero: the last interaction sampled a diffraction
    u32 skip;               // splats to skip (made by an earlier, overflowed, attempt)
    u32 splats;             // splats made
    u32 spectral;           // the path's spectral sample (`plt_scene::spectra`)
    real prev_dpd;          // solid-angle density of the last sampled direction, or 0 if discrete (`from_previous_dpd`)

    // forward transport
    real flux;              // the emitter's: Stokes parameters are relative to it

    // backward transport
    u32 ex, ey;             // the sensor element, and offset, sampled
    vec2 offset;
    observed_t<Polarimetric> L;     // radiant flux gathered so far (Stokes parameters, if polarimetric)

    /** @brief Splats a sensor connection (forward transport), unless an earlier attempt made it. */
    template <typename Film>
    WT_GPU_FN void splat(const WT_GPU_THREAD film_desc& desc, film_response response, WT_GPU_THREAD Film& film,
                         u32 ex, u32 ey, vec2 offset, real watts) {
        if (splats++>=skip)
            compute_kernel::splat(desc, response, film, ex, ey, offset, watts);
    }
};

/** @brief A sampled sensor beam's sensor element offset, and sampling densities (`sensor_sample_t`). */
struct sensor_sample_t {
    vec2 offset;
    real ppd;           // area density of the beam's origin, or 0 if discrete (a point sensor)
    real dpd;           // solid-angle density of its direction
};

/**
 * @brief `sensor_t::sample` (backward transport): a beam sourced from
 *        sensor element (ex,ey), at an offset within it, at the wavenumber
 *        of `spec`; its radiometric data is the sampled importance (`Se`)
 *        over the sampling densities.
 */
template <bool P>
WT_GPU_FN sensor_sample_t sample_sensor(const WT_GPU_THREAD sensor_desc& s, const WT_GPU_THREAD spectral_sample& spec,
                                        u32 ex, u32 ey, WT_GPU_THREAD rng& rnd, WT_GPU_THREAD beam_t<P>& b) {
    sensor_sample_t ret;
    ray r;
    real scale;
    if (s.type==sensor_perspective) {
        // perspective_t::sample: Se(P, recp_dpd) * recp_dpd, a unit importance flux
        const vec3 centre = point_on_sensor(s, vec2{ real(ex)+real(.5), real(ey)+real(.5) });
        ret.offset = rnd.r2() - vec2{ real(.5), real(.5) };
        const vec3 dir_local = normalize(centre + ret.offset.x*s.ddir_dx + ret.offset.y*s.ddir_dy);
        const vec3 dir = transform_direction(s.dir_to_world, dir_local);
        // perspective_t::ray_origin (in the kernels' precision)
        r = make_ray(s.position + s.near_clip*dir, dir);
        scale = 1;
        ret.ppd = 0;
        ret.dpd = 1 / perspective_recp_dpd(s, dir_local);
    } else {
        // virtual_plane_sensor_t::sample: Se(P) * recp_ppd / dpd
        ret.offset = rnd.r2() - vec2{ real(.5), real(.5) };
        const vec2 local = vec2{ real(ex)+ret.offset.x+real(.5), real(ey)+ret.offset.y+real(.5) } * s.element_extent;
        const vec3 p = s.origin + local.x*s.f.t + local.y*s.f.b;
        const vec3 wo = cosine_hemisphere(rnd.r2());
        const real dpd = inv_pi * wo.z;
        r = make_ray(p, s.f.to_world(wo));
        scale = dpd>real(0) ? s.importance * max(real(0), dot(r.d, s.f.n)) * s.area / dpd : real(0);
        ret.ppd = s.recp_area;
        ret.dpd = dpd;
    }

    // a perspective sensor's beams start `near_clip` along: as propagated there (beam_t::transform_restart)
    const real near_clip = s.type==sensor_perspective ? s.near_clip : real(0);
    b.envelope = make_isotropic_cone(r, spec.sensor_tan_alpha, spec.sensor_x0 + near_clip*spec.sensor_tan_alpha);
    // the identity operator, scaled (its first row; and the others, for a polarimetric sensor)
    b.S = { scale,0,0,0 };
    b.rows.identity(scale);
    b.Sframe = b.envelope.frame();
    b.self_intersection_distance = 0;
    return ret;
}

/** @brief `spot_t::compute_falloff`: the intensity scale of a spot's (local) direction. */
WT_GPU_FN real spot_falloff(const WT_GPU_GLOBAL emitter& em, vec3 local) {
    const real cos_theta = local.z;
    if (cos_theta<=em.cos_cutoff) return 0;
    if (cos_theta>=em.cos_falloff) return 1;
    return (em.cutoff - acos(cos_theta)) * em.recp_cutoff_range;
}

/** @brief A point on an area emitter (`shape_t::sample_position`: uniform by area), and its triangle. */
struct area_point_t {
    vec3 p;
    u32 tri;
};

WT_GPU_FN area_point_t sample_area_point(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_GLOBAL emitter& em,
                                         WT_GPU_THREAD rng& rnd) {
    // a triangle by area (the first whose inclusive CDF exceeds u)
    const real u = rnd.r();
    u32 lo = 0, hi = em.tri_count-1;
    while (lo<hi) {
        const u32 mid = (lo+hi)/2;
        if (u<scene.area_cdf[em.first_tri+mid]) hi = mid;
        else lo = mid+1;
    }
    const u32 tri = scene.area_tris[em.first_tri+lo];

    // a point on it (`sampler_t::uniform_triangle`)
    vec2 b = rnd.r2();
    if (b.x+b.y>real(1))
        b = vec2{ 1-b.x, 1-b.y };
    const bvh_tri t = scene.bvh.tris[tri];
    return { t.a + b.x*(t.b-t.a) + b.y*(t.c-t.a), tri };
}

/** @brief The power heuristic (`MIS` of plt_path_detail.hpp), of densities `pd1` (non-zero) and `pd2`. */
WT_GPU_FN real mis_power(real pd1, real pd2) {
    if (pd2==real(0))
        return 1;
    return pd1*pd1 / (pd1*pd1 + pd2*pd2);
}

/** @brief `scene_t::sample_emitter` for a uniform random number `u`: an index into the emitter CDF. */
WT_GPU_FN u32 sample_emitter(const WT_GPU_THREAD plt_scene& scene, real u) {
    u32 ei = 0;
    while (ei+1<scene.sampled_emitter_count && !(u<scene.emitter_cdf[ei]))
        ++ei;
    return ei;
}

/** @brief `start_path`'s `spectral` argument: choose a spectral sample of the table (rather than this one). */
WT_GPU_CONSTANT constexpr u32 choose_spectral = 0xffffffffu;

/**
 * @brief Which of an emitter's `count` spectral samples path `path` takes
 *        (of a film of `elements` elements: path `path` is sample
 *        path/elements of element path%elements). The samples, stratified
 *        over the spectrum (see src/gpu/renderer.cpp), are taken along a
 *        Weyl sequence of the element's sample index, from a hash of the
 *        element: an element's successive samples spread evenly over the
 *        spectrum (each band of a multispectral sensor gets its share), as
 *        the CPU's do (plt_path_t::integrate), and different elements start
 *        at different samples. (Not the path's random numbers.)
 */
WT_GPU_FN u32 spectral_sample_for_path(u64 path, u32 count, u64 elements) {
    // the splitmix64 finalizer, of the element
    u64 z = path % elements + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z>>30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z>>27)) * 0x94d049bb133111ebull;
    z = z ^ (z>>31);
    // the golden ratio's fraction, in 32-bit fixed point: x/2^32 = start + sample*0.618... (mod 1)
    const u32 x = u32(z>>32) + u32(path / elements) * 0x9e3779b9u;
    return u32((u64(x) * count) >> 32);
}

/**
 * @brief Starts path `path` (`plt_path::integrate_forward` or
 *        `integrate_backward`): samples an emitter, and one of the emitter's
 *        spectral samples in the table starting at `table` (or takes
 *        `spectral`, unless it is `choose_spectral`), as the CPU samples an
 *        emitter and a wavenumber; then, forward, sources a beam from the
 *        emitter; backward, from sensor element `path` modulo the film's
 *        elements. `rnd` is the path's generator; `skip` the splats an
 *        earlier, overflowed, attempt at this path made.
 */
template <bool P>
WT_GPU_FN path_state_t<P> start_path(const WT_GPU_THREAD plt_scene& scene, rng rnd, u64 path, u32 skip,
                                     u32 table, u32 spectral) {
    path_state_t<P> p;
    const u32 ei = sample_emitter(scene, rnd.r());
    p.spectral = spectral!=choose_spectral ? spectral :
        table + ei*scene.samples_per_emitter +
        spectral_sample_for_path(path, scene.samples_per_emitter, u64(scene.film.width)*scene.film.height);
    p.prev_dpd = 0;

    if (scene.params.backward!=0) {
        const spectral_sample spec = scene.spectra[p.spectral];
        const u32 element = u32(path % (u64(scene.film.width)*scene.film.height));
        p.ex = element % scene.film.width;
        p.ey = element / scene.film.width;
#if defined(REALISTIC_REMOTE_SENSING)
        if (scene.sensor.type==sensor_telescope)
            p.offset = sample_telescope(*scene.telescope, scene.spectra[p.spectral], p.ex, p.ey,
                                        vec2{ real(scene.film.width), real(scene.film.height) }, rnd, p.b).offset;
        else
#endif
        p.offset = sample_sensor(scene.sensor, spec, p.ex, p.ey, rnd, p.b).offset;
        p.prev_geo = { p.b.origin(), geo_point, 0 };
        p.L = {};
        p.flux = 0;
    } else {
        // source a beam (`emitter_t::sample`)
        const WT_GPU_GLOBAL emitter& em = scene.emitters[ei];
        const emission es = scene.emissions[p.spectral*scene.emitter_count + ei];
        ray r;
        real scale = 1;
        u32 surface_tri = no_triangle;
        if (em.type==emitter_point) {
            r = make_ray(em.position, uniform_sphere(rnd.r2()));
        } else if (em.type==emitter_spot) {
            // uniform over the cutoff cone, scaled by the falloff
            const vec3 local = uniform_cone(em.cutoff_solid_angle, rnd.r2());
            r = make_ray(em.position, transform_direction(em.dir_to_world, local));
            scale = spot_falloff(em, local);
        } else if (em.type==emitter_directional) {
            // from outside the world, through a point on the target disc
            const vec2 d = concentric_disk(rnd.r2()) * em.target_radius;
            const vec3 wp = em.target_centre + d.x*em.target.t + d.y*em.target.b;
            r = make_ray(wp + em.far*em.target.n, -em.target.n);
        } else {
            // area: from a point on the shape, cosine-distributed about its normal
            const area_point_t ap = sample_area_point(scene, em, rnd);
            const frame geo = build_shading_frame(scene.bvh.tris[ap.tri].n, scene.tris[ap.tri].dpdu);
            r = make_ray(ap.p, geo.to_world(cosine_hemisphere(rnd.r2())));
            surface_tri = ap.tri;
        }

        p.b.envelope = make_isotropic_cone(r, es.tan_alpha, es.x0);
        p.b.S = { scale,0,0,0 };
        p.b.rows.identity(scale);
        p.b.Sframe = p.b.envelope.frame();
        p.b.self_intersection_distance = 0;
        // The CPU starts from the beam's origin, a point, even on an area emitter; but there, a GPU ray
        // query finds the emitter's triangle itself, at a distance of about 0: start from its surface,
        // whose ray origins are offset off the triangle.
        p.prev_geo = surface_tri!=no_triangle ? vertex_geo{ r.o, geo_surface, surface_tri } :
                                                vertex_geo{ r.o, geo_point, 0 };
        p.flux = es.flux;
        p.ex = p.ey = 0;
        p.offset = { 0,0 };
        p.L = {};
    }

    p.prev_beam = p.b;
    p.rnd = rnd;
    p.throughput = 1;
    p.depth = 1;
    p.iteration = 0;
    p.aperture_size = 0;
    p.fsd = 0;
    p.sampled_fsd = 0;
    p.skip = skip;
    p.splats = 0;
    return p;
}

/**
 * @brief Ends a path (that did not overflow): backward transport splats the
 *        radiant flux it gathered into its sensor element
 *        (`plt_path::splat_backward`), unless an earlier attempt did.
 */
template <bool P, typename Film>
WT_GPU_FN void end_path(const WT_GPU_THREAD plt_scene& scene, WT_GPU_THREAD Film& film,
                        WT_GPU_THREAD path_state_t<P>& p) {
    if (scene.params.backward!=0 && p.splats++>=p.skip)
        splat_weighted(scene.film, scene.spectra[p.spectral].response, film, p.ex, p.ey, p.offset, p.L);
}

/**
 * @brief `scene_t::sample_emitter_direct`: a sampled emitter's connection to
 *        `wp` (`emitter_t::sample_direct`), at spectral sample `spectral`.
 */
struct emitter_direct_t {
    u32 emitter;
    /**
     * @brief The emitted beam's intensity, over the emitter's sampling
     *        probability; 0 if the emitter does not reach `wp`.
     */
    real I;
    vec3 wo;            // the direction from `wp` to the emitter
    vertex_geo geo;     // the emitter point
    real dpd;           // solid-angle density of the connection (area emitters), or 0 if discrete
};

WT_GPU_FN emitter_direct_t sample_emitter_direct(const WT_GPU_THREAD plt_scene& scene, u32 spectral, vec3 wp,
                                                 WT_GPU_THREAD rng& rnd) {
    emitter_direct_t ret;
    ret.emitter = sample_emitter(scene, rnd.r());
    ret.dpd = 0;
    const WT_GPU_GLOBAL emitter& em = scene.emitters[ret.emitter];
    const real intensity = scene.emissions[spectral*scene.emitter_count + ret.emitter].intensity;
    if (em.type==emitter_point || em.type==emitter_spot) {
        const vec3 dl = wp - em.position;
        const real recp_dist2 = 1 / length2(dl);
        const vec3 d = dl * sqrt(recp_dist2);
        ret.I = intensity * recp_dist2 / em.pmf;
        if (em.type==emitter_spot)
            ret.I *= spot_falloff(em, transform_direction(em.dir_to_local, d));
        ret.wo = -d;
        ret.geo = { em.position, geo_point, 0 };
    } else if (em.type==emitter_directional) {
        // from outside the world, if `wp` is in the target disc
        const vec3 v = wp - em.target_centre;
        const vec2 plocal = { dot(v, em.target.t), dot(v, em.target.b) };
        const real scale = length2(plocal)<=sqr(em.target_radius) ? real(1) : real(0);
        const vec3 targetwp = em.target_centre + plocal.x*em.target.t + plocal.y*em.target.b;
        ret.I = intensity * scale / em.pmf;
        ret.wo = em.target.n;
        ret.geo = { targetwp + em.far*em.target.n, geo_point, 0 };
    } else {
        // area: a point on the shape (area_t::sample_direct, pdf_direct)
        const area_point_t ap = sample_area_point(scene, em, rnd);
        const vec3 dl = wp - ap.p;
        const real l2 = length2(dl);
        const vec3 d = dl / sqrt(l2);
        const real dn = max(real(0), dot(d, scene.bvh.tris[ap.tri].n));
        const real recp_dn = dn>real(0) ? 1/dn : real(0);
        ret.dpd = (1/em.area) * l2 * recp_dn;
        const real recp_dpd = ret.dpd>real(0) ? 1/ret.dpd : real(0);
        ret.I = intensity * dn * recp_dpd / em.pmf;
        ret.wo = -d;
        ret.geo = { ap.p, geo_surface, ap.tri };
    }
    return ret;
}

/**
 * @brief `nee_backward`: next-event estimation from the surface `s` of a
 *        non-delta material: connects to a sampled emitter (at spectral
 *        sample `spectral`), and returns the radiant flux (Stokes
 *        parameters) the beam `b` observes through the connection,
 *        MIS-weighted against sampling the BSDF for area emitters (the others
 *        are sampled discretely).
 */
template <bool P, typename Geometry>
WT_GPU_FN observed_t<P> nee_backward(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                            u32 spectral, const WT_GPU_THREAD beam_t<P>& b, const WT_GPU_THREAD surface& s,
                            const WT_GPU_THREAD material& m, WT_GPU_THREAD rng& rnd,
                            WT_GPU_THREAD u32& status) {
    const emitter_direct_t ed = sample_emitter_direct(scene, spectral, s.wp, rnd);
    if (ed.I==real(0))
        return {};
    const real I = ed.I;
    const vec3 woworld = ed.wo;
    const vertex_geo emitter_geo = ed.geo;
    const real pd_direct = ed.dpd * scene.emitters[ed.emitter].pmf;

    const vec3 wiworld = -b.dir();
    const vec3 wi = s.shading.to_local(wiworld);
    const vec3 wo = s.shading.to_local(woworld);
    if (wi.z*dot(wiworld, s.geo.n)<=real(0) || wo.z*dot(woworld, s.geo.n)<=real(0))
        return {};

    const mueller_block f = eval_bsdf(m, wi, wo, true);
    if (f.m00==real(0))
        return {};

    if (shadowed(scene, geometry, vertex_geo{ s.wp, geo_surface, s.tri }, emitter_geo, status))
        return {};

    // integrate_beams over the beam after the interaction, and the unpolarized emitted beam
    beam_t<P> nee = b;
    nee.apply_bsdf(s, woworld, f, true);
    const real mis = pd_direct>real(0) ? mis_power(pd_direct, pdf_bsdf(m, wi, wo)) : real(1);
    return nee.observe_unpolarized(I * mis);
}

/**
 * @brief `emission` of plt_path_detail.hpp (backward transport): the radiant
 *        flux the beam `b` observes from the area emitter `ei` it hit at `s`,
 *        MIS-weighted against next-event estimation if the beam's direction
 *        was sampled from a density `prev_dpd` (`area_t::Li`, `pdf_position`).
 */
template <bool P>
WT_GPU_FN observed_t<P> emission_hit(const WT_GPU_THREAD plt_scene& scene, u32 spectral, u32 ei, real prev_dpd,
                                     const WT_GPU_THREAD beam_t<P>& b, const WT_GPU_THREAD surface& s) {
    const WT_GPU_GLOBAL emitter& em = scene.emitters[ei];
    const real dn = dot(-b.dir(), s.geo.n);
    if (dn<=real(0))
        return {};
    const real L = scene.emissions[spectral*scene.emitter_count + ei].intensity;

    real mis = 1;
    if (prev_dpd>real(0)) {
        // the solid-angle density of next-event estimation sampling this point
        const real l2 = length2(b.origin() - s.wp);
        const real pd_nee = (1/em.area) * l2 / abs(dn);
        mis = mis_power(prev_dpd, pd_nee * em.pmf);
    }
    return b.observe_unpolarized(L * mis);
}

/**
 * @brief The surface the beam `b` hits at `primary`, at the interaction
 *        distance `dist`: the intersection `s`, and its material at spectral
 *        sample `spectral`, textured. Returns the surface's emitter, or
 *        no_emitter.
 */
template <bool P>
WT_GPU_FN u32 hit_surface(const WT_GPU_THREAD plt_scene& scene, u32 spectral, const WT_GPU_THREAD primary_t& primary,
                          real dist, const WT_GPU_THREAD beam_t<P>& b,
                          WT_GPU_THREAD surface& s, WT_GPU_THREAD material& mat) {
    const bvh_tri tri = scene.bvh.tris[primary.tri];
    const tri_shading sh = scene.tris[primary.tri];
    mat = scene.materials[spectral*scene.material_count + sh.material];
    s.tri = primary.tri;
    // interpolated from the triangle, not origin + dist*dir: see the CPU's intersection_surface_t
    const real bz = 1 - primary.baryx - primary.baryy;
    s.wp = primary.baryx*tri.a + primary.baryy*tri.b + bz*tri.c;
    s.geo = build_shading_frame(tri.n, sh.dpdu);
    s.shading = build_shading_frame(normalize(primary.baryx*sh.n[0] + primary.baryy*sh.n[1] + bz*sh.n[2]),
                                    sh.dpdu);
    s.fp = b.surface_footprint_static(s.geo, dist);

    // textures (scenes without have no UVs)
    if (scene.tri_uvs) {
        const material_shading ms = scene.material_shadings[sh.material];
        if (ms.normalmap!=no_texture || is_textured(mat))
            apply_textures(scene, spectral, ms, primary.tri, primary.baryx, primary.baryy, sh.dpdu, s.shading, mat);
    }
    return sh.emitter;
}

/**
 * @brief A path step's first part, its BVH queries: traces the beam to its
 *        next interaction (into `tr`), and finds the triangles around a
 *        ballistic intersection, whose edges may diffract the beam (into the
 *        scratch memory). Returns whether the path goes on (it found an
 *        interaction); adds status flags to `status`. `interact` continues the
 *        step.
 */
template <bool P, typename Geometry>
WT_GPU_FN bool trace_step(const WT_GPU_THREAD plt_scene& scene,
                          const WT_GPU_THREAD Geometry& geometry,
                          const WT_GPU_THREAD path_scratch& scratch,
                          const WT_GPU_THREAD path_state_t<P>& p,
                          WT_GPU_THREAD traversal_t& tr,
                          WT_GPU_THREAD u32& status) {
    const plt_params opts = scene.params;
    // nothing to trace: no depth, or a beam that carries nothing (the end `interact` gives such beams; and a
    // telescope sensor's vignetted samples, which only weigh the film's reconstruction)
    if (opts.max_depth==0 || p.b.S.s0==real(0))
        return false;
    if (p.iteration==max_path_iterations) {
        status |= status_iteration_limit;
        return false;
    }

    const bool ray_trace = opts.ray_tracing_only!=0;
    const WT_GPU_THREAD beam_t<P>& b = p.b;
    cone envelope = b.envelope;
    envelope.r.o = offseted_ray_origin(scene, p.prev_geo, envelope.r);
    tr = traverse(geometry, envelope, two_pi/scene.spectra[p.spectral].k, ray_trace, scratch, status);
    if ((status & status_scratch_overflow) || !tr.hit)
        return false;

    // the triangles around a ballistic intersection (diffusive traversals found theirs)
    if (tr.ballistic && !b.is_ray() && !ray_trace) {
        const real dist = tr.dist;
        const real zdist = b.envelope.axes(dist).x * 2;
        tri_sink sink = { scratch.tris, scratch.tri_capacity, 0 };
        const cone_hit ch = geometry.intersect_cone(b.envelope, range{ dist - zdist/2, dist + zdist/2 }, 1, sink);
        status |= ch.status;
        if (ch.tri_count>scratch.tri_capacity) {
            status |= scratch_overflow(ch.tri_count, scratch.tri_capacity);
            return false;
        }
        tr.tri_count = ch.tri_count;
    }
    return true;
}

/**
 * @brief A path step's second part, after `trace_step` found the interaction
 *        `tr`: makes the path's connections (forward, splatting sensor
 *        connections into `film`; backward, gathering emitted flux into `p.L`)
 *        and samples the interaction. Returns whether the path goes on; adds
 *        status flags to `status`.
 */
template <bool P, typename Geometry, typename Film>
WT_GPU_FN bool interact(const WT_GPU_THREAD plt_scene& scene,
                        const WT_GPU_THREAD Geometry& geometry,
                        WT_GPU_THREAD Film& film,
                        const WT_GPU_THREAD path_scratch& scratch,
                        const WT_GPU_THREAD traversal_t& tr,
                        WT_GPU_THREAD path_state_t<P>& p,
                        WT_GPU_THREAD u32& status) {
    const plt_params opts = scene.params;
    const bool ray_trace = opts.ray_tracing_only!=0;
    const bool backward = opts.backward!=0;
    const spectral_sample spec = scene.spectra[p.spectral];
    const real k = spec.k;

    ++p.iteration;

    WT_GPU_THREAD beam_t<P>& b = p.b;
    WT_GPU_THREAD rng& rnd = p.rnd;

    const real dist_to_interaction = tr.dist;
    const bool is_ballistic = tr.ballistic || b.is_ray();
    const vec3 origin_wp = tr.origin;
    const vec3 interaction_wp = origin_wp + dist_to_interaction*b.dir();

    //
    // evaluate fsd from previous interaction

    if (p.fsd!=0) {
        const cone prev_cone = p.prev_beam.envelope;
        const real f = do_fsd(scene, geometry, k, prev_cone, p.prev_geo, interaction_wp,
                              scratch.aperture, p.aperture_size, status).intensity();
        p.fsd = 0;

        if (p.sampled_fsd!=0)
            b.scale(f);
        else {
            p.prev_beam.transform_region_interaction(origin_wp, length(origin_wp-prev_cone.r.o), b.dir(), f);
            b.add(p.prev_beam, backward);
        }
    }

    //
    // check which triangle falls under the intersection point, if any

    primary_t primary;
    if (is_ballistic)
        primary = { tr.tri, tr.dist, tr.baryx, tr.baryy };
    else
        primary = find_closest_triangle(scene, scratch.tris, tr.tri_count,
                                        range{ dist_to_interaction, dist_to_interaction+tr.region_depth },
                                        origin_wp, b.dir());
    const bool has_surface = primary.tri!=no_triangle;

    // distance to interaction region end, or triangle intersection
    const real interaction_region_end = has_surface ? primary.dist : dist_to_interaction;

    // the surface intersection, its material, and its emitter
    surface s;
    material mat;
    u32 surface_emitter = no_emitter;
    if (has_surface)
        surface_emitter = hit_surface(scene, p.spectral, primary, dist_to_interaction, b, s, mat);

    // edges: of the triangles around a ballistic intersection, or of the diffusive intersection's
    u32 edge_count = 0;
    if ((is_ballistic && !b.is_ray() && !ray_trace) || (!is_ballistic && opts.fsd!=0)) {
        if (!collect_edges(scene, scratch.tris, tr.tri_count, scratch.edges, scratch.edge_capacity, edge_count)) {
            status |= scratch_overflow(edge_count, scratch.edge_capacity);
            return false;
        }
    }

    // if we have edges, construct the fsd aperture
    if (edge_count>0) {
        p.aperture_size = build_aperture(scene.edges, scratch.edges, edge_count,
                                         interaction_wp, b.envelope.frame(), b.footprint3(dist_to_interaction),
                                         -b.dir(), scratch.aperture);
        p.fsd = p.aperture_size>0 ? 1u : 0u;
    }

    //
    // NEE: backward, connect the surface to an emitter; forward, connect the
    // diffracted beam to the sensor

    if (backward) {
        if (p.depth<opts.max_depth && has_surface && !is_delta(mat))
            p.L += nee_backward(scene, geometry, p.spectral, b, s, mat, rnd, status);
    } else if (p.depth<opts.max_depth && p.fsd!=0) {
        const sensor_direct_t sd = sensor_sample_direct(scene.sensor, scene.film, interaction_wp, rnd);
        if (sd.importance>real(0)) {
            const real f = do_fsd(scene, geometry, k, b.envelope, p.prev_geo, sd.p,
                                  scratch.aperture, p.aperture_size, status).intensity();
            if (f!=real(0)) {
                beam_t<P> fsd_beam = b;
                fsd_beam.transform_region_interaction(interaction_wp, dist_to_interaction, -sd.wd, f);
                if (fsd_beam.S.s0>real(0))
                    p.splat(scene.film, spec.response, film, sd.ex, sd.ey, sd.offset,
                            sd.importance * fsd_beam.S.s0 * p.flux);
            }
        }
    }

    //
    // emission (backward: the beam hit an emitter)

    if (backward && surface_emitter!=no_emitter)
        p.L += emission_hit(scene, p.spectral, surface_emitter, p.prev_dpd, b, s);

    //
    // sensing (forward: connection to the virtual sensor), up to the interaction

    if (!backward) {
        const real max_distance = interaction_region_end -
                                  max(real(0), dot(b.dir(), origin_wp - b.origin()));
        const sensor_hit_t sensed = virtual_plane_hit(scene.sensor, b, range{ 0, max_distance });
        if (sensed.hit && b.S.s0>real(0))
            p.splat(scene.film, spec.response, film, sensed.ex, sensed.ey, sensed.offset,
                    b.S.s0 * p.flux * scene.sensor.importance);
    }

    // the path ends after max_depth vertices: a further interaction contributes nothing
    if (p.depth>=opts.max_depth)
        return false;

    //
    // do interactions

    bool sampled_null = false;
    if (has_surface) {
        // sample_surface_interaction
        const vec3 wiworld = -b.dir();
        const vec3 wi = s.shading.to_local(wiworld);
        if (dot(wiworld, s.geo.n)*wi.z<=real(0))
            return false;

        bsdf_sample_t bs;
        if (!sample_bsdf(mat, wi, backward, rnd, bs) || !(bs.pdf>real(0)))
            return false;

        const vec3 woworld = normalize(s.shading.to_world(bs.wo));
        if (dot(woworld, s.geo.n)*bs.wo.z<=real(0))
            return false;

        p.prev_geo = { s.wp, geo_surface, s.tri };
        p.prev_beam = b;
        p.sampled_fsd = 0;
        p.prev_dpd = bs.discrete ? real(0) : bs.pdf;
        b.transform_surface_interaction(s, woworld, bs.M, 1, backward);
        p.throughput *= bs.M.m00;
        if (bs.eta!=real(1))
            p.throughput /= sqr(bs.eta);
    } else if (p.fsd!=0) {
        // sample_fsd_interaction
        const fsd_sample_t fs = fsd_sample(scene.edges, scratch.aperture, p.aperture_size, k,
                                           interaction_wp, p.prev_geo.wp, rnd);
        p.prev_geo = { interaction_wp, geo_point, 0 };
        p.prev_beam = b;
        p.sampled_fsd = 1;
        p.prev_dpd = 0;
        b.transform_region_interaction(interaction_wp, dist_to_interaction, fs.wo, fs.weight);
        p.throughput *= fs.weight;
    } else {
        // no edges: a null interaction, continue propagation
        sampled_null = true;
        b.transform_restart(interaction_wp, dist_to_interaction);
    }

    //
    // continue_walk

    if (b.S.s0==real(0))
        return false;
    if (!sampled_null && opts.russian_roulette!=0) {
        const real q = p.throughput<real(1) ? max(p.throughput, real(.5)) : real(1);
        if (!(rnd.r()<=q))
            return false;
        const real scale = real(1)/q;
        b.scale(scale);
        p.throughput *= scale;
    }
    if (!sampled_null)
        ++p.depth;
    return true;
}

/**
 * @brief One step of a path (an iteration of `random_walk`): traces the beam
 *        to its next interaction, makes its connections and samples the
 *        interaction. Returns whether the path goes on; adds status flags to
 *        `status`. A path that ends without overflowing goes to `end_path`.
 */
template <bool P, typename Geometry, typename Film>
WT_GPU_FN bool step_path(const WT_GPU_THREAD plt_scene& scene,
                         const WT_GPU_THREAD Geometry& geometry,
                         WT_GPU_THREAD Film& film,
                         const WT_GPU_THREAD path_scratch& scratch,
                         WT_GPU_THREAD path_state_t<P>& p,
                         WT_GPU_THREAD u32& status) {
    traversal_t tr;
    return trace_step(scene, geometry, scratch, p, tr, status) &&
           interact(scene, geometry, film, scratch, tr, p, status);
}

} } }
