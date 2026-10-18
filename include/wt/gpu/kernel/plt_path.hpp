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
 * Forward plt_path on the GPU: a port of `integrate_forward` and
 * `random_walk` of wt/integrator/plt_path/plt_path_detail.hpp, one path per
 * GPU thread, the recursion unrolled into a loop.
 *
 * Implemented so far: ray tracing (`sensor_t::ray_trace_only()`), with
 * specular surfaces and sensing by a virtual plane sensor. The host refuses
 * other scenes (src/gpu/renderer.cpp).
 *
 * `trace_forward_path` is parameterized by the backend's
 *   Geometry: `ray_hit intersect(const ray&, range) const`, and
 *   Film:     `void add(u32 index, real value)`, accumulating film elements.
 */

namespace wt::gpu::kernel {

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

/**
 * @brief `film_t::splat_direct`: splats `watts` around element (ex,ey) with
 *        fractional offset `offset`, weighted by the reconstruction filter.
 */
template <typename Film>
WT_GPU_FN void splat(const WT_GPU_THREAD film_desc& film, WT_GPU_THREAD Film& sink,
                     u32 ex, u32 ey, vec2 offset, real watts) {
    const i32 r = film.rf_radius;

    // integrate the reconstruction filter over the elements' footprints
    real wx[2*max_rf_radius+1], wy[2*max_rf_radius+1];
    for (i32 x=-r; x<=r; ++x) {
        wx[x+r] = gaussian_integrate(film.rf_sigma, real(x)+offset.x-real(.5), real(x)+offset.x+real(.5));
        wy[x+r] = gaussian_integrate(film.rf_sigma, real(x)+offset.y-real(.5), real(x)+offset.y+real(.5));
    }
    real tw = 0;
    for (i32 y=-r; y<=r; ++y)
    for (i32 x=-r; x<=r; ++x)
        tw += max(real(0), wx[x+r]*wy[y+r]);
    const real recp_tw = tw>real(0) ? real(1)/tw : real(0);

    for (u32 c=0; c<film.components; ++c) {
        const real val = watts * film.response[c];
        // avoid NaNs, infs and negatives
        if (!(val>real(0)) || !isfinite(val))
            continue;

        for (i32 y=-r; y<=r; ++y)
        for (i32 x=-r; x<=r; ++x) {
            const i32 px = i32(ex)+x, py = i32(ey)+y;
            if (px<0 || py<0 || px>=i32(film.width) || py>=i32(film.height))
                continue;
            const real w = max(real(0), wx[x+r]*wy[y+r]) * recp_tw;
            sink.add((u32(py)*film.width + u32(px))*film.components + c, w*val);
        }
    }
}

/** @brief Truncation to an element index, saturating as the CPU (AArch64, CUDA) conversion does. */
WT_GPU_FN u32 to_element(real x) {
    return x>real(0) ? (x<real(4294967040.0) ? u32(x) : 4294967040u) : 0u;
}

struct sensor_hit_t {
    u32 ex, ey;
    vec2 offset;
    bool hit;
};

/**
 * @brief `virtual_plane_sensor_t::Si`: where the beam's central ray crosses
 *        the sensor's front within `rng` (distance from the beam origin).
 */
WT_GPU_FN sensor_hit_t virtual_plane_hit(const WT_GPU_THREAD virtual_plane_sensor& s,
                                         const WT_GPU_THREAD beam& b, range rng) {
    const sensor_hit_t none = { 0, 0, { 0,0 }, false };

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

    return { ex, ey, offset, true };
}

struct bsdf_sample_t {
    vec3 wo;
    fresnel_mueller M;      // weighted BSDF: bsdf/pdf
    real pdf;
    real eta;
};

/**
 * @brief Samples a material's BSDF (`bsdf_t::sample`) for incident direction
 *        `wi` (shading frame, pointing away from the surface).
 */
WT_GPU_FN bool sample_bsdf(const WT_GPU_THREAD material& m, vec3 wi, WT_GPU_THREAD rng& rnd,
                           WT_GPU_THREAD bsdf_sample_t& out) {
    if (m.type!=material_specular)
        return false;

    // two_sided_t: the back side mirrors the front side
    const bool flip = m.two_sided!=0 && wi.z<real(0);
    if (flip)
        wi.z = -wi.z;

    // surface_spm_t::sample with a Dirac surface profile: all scattering is specular (alpha=1)
    if (wi.z==real(0))
        return false;

    real pdf = 1;
    const fresnel_t f = fresnel(m.eta, wi);
    const real pdf_transmission = (f.Ts+f.Tp)/2;
    bool is_reflection = true;
    if (m.has_transmission!=0) {
        is_reflection = rnd.r()>=pdf_transmission;
        pdf *= is_reflection ? 1-pdf_transmission : pdf_transmission;
    }

    const real scale = is_reflection ? m.reflection_scale : m.transmission_scale;
    if (scale==real(0) || (!is_reflection && m.has_transmission==0))
        return false;

    const vec3 wo = is_reflection ? vec3{ -wi.x, -wi.y, wi.z } : f.t;
    // alpha * J * scale * F, with alpha = J = 1 (forward transport)
    out.M = (mueller_fresnel(m.eta, is_reflection, wi) * scale) / pdf;
    out.wo = flip ? vec3{ wo.x, wo.y, -wo.z } : wo;
    out.pdf = pdf;
    out.eta = is_reflection ? real(1) : f.eta_12;
    return true;
}

constexpr u32 no_triangle = 0xffffffffu;

/**
 * @brief Traces one forward path (`plt_path::integrate_forward`), splatting
 *        its sensor connections into `film`. Returns query status flags.
 */
template <typename Geometry, typename Film>
WT_GPU_FN u32 trace_forward_path(const WT_GPU_THREAD plt_scene& scene,
                                 const WT_GPU_THREAD Geometry& geometry,
                                 WT_GPU_THREAD Film& film,
                                 WT_GPU_THREAD rng& rnd) {
    const plt_path_params opts = scene.params;
    u32 status = status_ok;
    if (opts.max_depth==0)
        return status;

    // sample an emitter (the wavenumber is fixed)
    const real ue = rnd.r();
    u32 ei = 0;
    while (ei+1<scene.emitter_count && !(ue<scene.emitter_cdf[ei]))
        ++ei;
    const point_emitter em = scene.emitters[ei];

    // source a beam (point_t::sample); its Stokes parameters are relative to `em.flux`
    beam b;
    b.envelope = make_isotropic_cone(make_ray(em.position, uniform_sphere(rnd.r2())), em.tan_alpha, em.x0);
    b.S = { 1,0,0,0 };
    b.Sframe = b.envelope.frame();
    b.self_intersection_distance = 0;
    const real watts_per_S0 = em.flux * scene.sensor.importance;

    real throughput = 1;
    // the previous path vertex: the emitter, or the BVH triangle of a surface interaction
    u32 prev_tri = no_triangle;

    for (u32 depth=1;; ++depth) {
        // trace (traverse(), ray tracing), avoiding self intersection with the previous surface
        ray r = b.envelope.r;
        if (prev_tri!=no_triangle) {
            const bvh_tri t = scene.bvh.tris[prev_tri];
            r.o = offseted_ray_origin(t.a, t.b, t.c, t.n, r);
        }
        const ray_hit hit = geometry.intersect(r, range::positive());
        status |= hit.status;
        // no intersection: the beam escapes (and is not sensed)
        if (!hit.hit)
            break;

        const real dist = hit.dist;
        const vec3 origin_wp = r.o;

        // sensing (connection to the virtual sensor), up to the interaction
        const real max_distance = dist - max(real(0), dot(b.dir(), origin_wp - b.origin()));
        const sensor_hit_t sensed = virtual_plane_hit(scene.sensor, b, range{ 0, max_distance });
        if (sensed.hit)
            splat(scene.film, film, sensed.ex, sensed.ey, sensed.offset, b.S.s0 * watts_per_S0);

        // the path ends after max_depth segments: a further interaction contributes nothing
        if (depth>=opts.max_depth)
            break;

        // the surface interaction (intersection_surface_t)
        const bvh_tri tri = scene.bvh.tris[hit.tri];
        const tri_shading sh = scene.tris[hit.tri];
        surface s;
        s.tri = hit.tri;
        s.wp = origin_wp + dist*b.dir();
        s.geo = build_shading_frame(tri.n, sh.dpdu);
        const real bz = 1 - hit.baryx - hit.baryy;
        s.shading = build_shading_frame(normalize(hit.baryx*sh.n[0] + hit.baryy*sh.n[1] + bz*sh.n[2]), sh.dpdu);
        s.fp = b.surface_footprint_static(s.geo, dist);

        // sample_surface_interaction
        const vec3 wiworld = -b.dir();
        const vec3 wi = s.shading.to_local(wiworld);
        if (dot(wiworld, s.geo.n)*wi.z<=real(0))
            break;

        bsdf_sample_t bs;
        if (!sample_bsdf(scene.materials[sh.material], wi, rnd, bs) || !(bs.pdf>real(0)))
            break;

        const vec3 woworld = normalize(s.shading.to_world(bs.wo));
        if (dot(woworld, s.geo.n)*bs.wo.z<=real(0))
            break;

        b.transform_surface_interaction(s, woworld, bs.M, 1);
        throughput *= bs.M.m00;
        if (bs.eta!=real(1))
            throughput /= sqr(bs.eta);
        prev_tri = hit.tri;

        // continue_walk
        if (b.S.s0==real(0))
            break;
        if (opts.russian_roulette!=0) {
            const real p = throughput<real(1) ? max(throughput, real(.5)) : real(1);
            if (!(rnd.r()<=p))
                break;
            const real scale = real(1)/p;
            b.S = { b.S.s0*scale, b.S.s1*scale, b.S.s2*scale, b.S.s3*scale };
            throughput *= scale;
        }
    }

    return status;
}

}
