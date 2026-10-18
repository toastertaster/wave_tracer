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

/*
 * Forward plt_path on the GPU: a port of `integrate_forward` and
 * `random_walk` of wt/integrator/plt_path/plt_path_detail.hpp (with
 * `traverse` of wt/integrator/traversal.hpp), one path per GPU thread, the
 * recursion unrolled into a loop. Beams propagate ballistically and
 * diffusively, diffract at edges (free-space diffraction), reflect and
 * refract at specular surfaces, and are sensed by a virtual plane sensor.
 * The host refuses other scenes (src/gpu/renderer.cpp).
 *
 * `trace_forward_path` is parameterized by the backend's
 *   Geometry: `ray_hit intersect(const ray&, range) const`,
 *             `bool occluded(const ray&, range, u32& status) const`,
 *             `cone_hit intersect_cone(const cone&, range, real z_scale, tri_sink&) const`;
 *   Film:     `void add(u32 index, real value)`, accumulating film elements.
 *
 * Cone queries return triangle lists, which paths keep in fixed-capacity
 * scratch memory. A path that outgrows it stops with `status_scratch_overflow`
 * and reports how many splats it made; as its random numbers depend only on
 * its index, the backend re-traces it with more room, skipping those splats.
 */

namespace wt::gpu::kernel {

/** @brief A path outgrew its scratch memory (see above). */
constexpr u32 status_scratch_overflow = 2;
/** @brief A path made `max_path_iterations` steps (counting null interactions) and was stopped. */
constexpr u32 status_iteration_limit = 4;

/**
 * @brief Steps a path may take, counting null interactions, which do not
 *        count towards `max_depth`. (The CPU recurses without a limit.)
 */
constexpr u32 max_path_iterations = 1u<<14;

/** @brief A path's scratch memory. */
struct path_scratch {
    WT_GPU_GLOBAL u32* tris;
    WT_GPU_GLOBAL u32* edges;
    WT_GPU_GLOBAL aperture_entry* aperture;
    u32 tri_capacity;
    u32 edge_capacity;
};

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

/*
 * Path vertices, visibility
 */

constexpr u32 geo_point = 0;
constexpr u32 geo_surface = 1;
constexpr u32 geo_edge = 2;

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

/** @brief `shadow(ads, a, b)` of wt/integrator/traversal.hpp: is the segment between two vertices blocked? */
template <typename Geometry>
WT_GPU_FN bool shadowed(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                        vertex_geo a, vertex_geo b, WT_GPU_THREAD u32& status) {
    const ray r = make_ray(a.wp, normalize(b.wp-a.wp));
    const vec3 o = offseted_ray_origin(scene, a, r);
    const vec3 t = offseted_ray_origin(scene, b, make_ray(b.wp, -r.d));
    const real dist = length(t-o);
    const vec3 d = (t-o)/dist;
    return geometry.occluded(make_ray(o, d), range{ 0, dist }, status);
}


/*
 * Traversal
 */

/** @brief Result of `traverse()` (`traversal_result_t`). */
struct traversal_t {
    vec3 origin;            // traversal origin (offset for self-intersection avoidance)
    real dist;              // distance to the intersection
    real region_depth;      // depth past `dist` over which triangles are considered
    bool hit;
    bool ballistic;
    // ballistic: the triangle hit
    u32 tri;
    real baryx, baryy;
    // diffusive: the triangles, in the scratch memory
    u32 tri_count;
};

/**
 * @brief `traverse()` of wt/integrator/traversal.hpp: short ballistic (ray)
 *        segments, after each of which diffusive (cone) propagation is tried.
 *        `envelope` starts at the (offset) traversal origin. Sets
 *        `status_scratch_overflow` if the triangles do not fit.
 */
template <typename Geometry>
WT_GPU_FN traversal_t traverse(const WT_GPU_THREAD Geometry& geometry, const WT_GPU_THREAD cone& envelope,
                               real lambda, bool ray_trace, const WT_GPU_THREAD path_scratch& scratch,
                               WT_GPU_THREAD u32& status) {
    const ray r = envelope.r;
    traversal_t ret = { r.o, inf(), 0, false, true, no_triangle, 0, 0, 0 };

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
            status |= status_scratch_overflow;
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

/**
 * @brief The edges of `tris` (`cone_work_to_intersection_record`), sorted and
 *        unique as the CPU's set. Returns false if they do not fit.
 */
WT_GPU_FN bool collect_edges(const WT_GPU_THREAD plt_scene& scene,
                             const WT_GPU_GLOBAL u32* tris, u32 tri_count,
                             WT_GPU_GLOBAL u32* edges, u32 capacity, WT_GPU_THREAD u32& count) {
    count = 0;
    for (u32 i=0; i<tri_count; ++i) {
        const tri_shading sh = scene.tris[tris[i]];
        for (int j=0; j<3; ++j) {
            const u32 e = sh.edges[j];
            if (e==no_edge) continue;

            // insert into the sorted list
            u32 lo = 0, hi = count;
            while (lo<hi) {
                const u32 mid = (lo+hi)/2;
                if (edges[mid]<e) lo = mid+1;
                else hi = mid;
            }
            if (lo<count && edges[lo]==e) continue;
            if (count==capacity) return false;
            for (u32 k=count; k>lo; --k)
                edges[k] = edges[k-1];
            edges[lo] = e;
            ++count;
        }
    }
    return true;
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
                                  const WT_GPU_THREAD cone& cone_from_src, vertex_geo src_geo, vec3 dst,
                                  const WT_GPU_GLOBAL aperture_entry* aperture, u32 aperture_size,
                                  WT_GPU_THREAD u32& status) {
    const real k = scene.params.k;
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

/** @brief Splats a path's sensor connections, skipping the first `skip` (made by an earlier, overflowed, attempt). */
template <typename Film>
struct path_film {
    Film film;
    u32 skip;
    u32 splats;

    WT_GPU_FN void splat(const WT_GPU_THREAD film_desc& desc, u32 ex, u32 ey, vec2 offset, real watts) {
        if (splats++>=skip)
            kernel::splat(desc, film, ex, ey, offset, watts);
    }
};

/** @brief `virtual_plane_sensor_t::sample_direct`: a sensor point connecting to `wp`, its element, and the importance of the connection. */
struct sensor_direct_t {
    vec3 p;
    vec3 wd;            // from the sensor point towards `wp`
    u32 ex, ey;
    vec2 offset;
    real importance;    // 0 if not connectible
};

WT_GPU_FN sensor_direct_t sensor_sample_direct(const WT_GPU_THREAD virtual_plane_sensor& s, vec3 wp,
                                               WT_GPU_THREAD rng& rnd) {
    sensor_direct_t ret;

    // sample point on sensor
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
 * @brief Traces one forward path (`plt_path::integrate_forward`), splatting
 *        its sensor connections into `film`. Returns status flags.
 */
template <typename Geometry, typename Film>
WT_GPU_FN u32 trace_forward_path(const WT_GPU_THREAD plt_scene& scene,
                                 const WT_GPU_THREAD Geometry& geometry,
                                 WT_GPU_THREAD path_film<Film>& film,
                                 const WT_GPU_THREAD path_scratch& scratch,
                                 WT_GPU_THREAD rng& rnd) {
    const plt_path_params opts = scene.params;
    const bool ray_trace = opts.ray_tracing_only!=0;
    const real k = opts.k;
    const real lambda = two_pi/k;

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

    // path_walk_data_t
    vertex_geo prev_geo = { em.position, geo_point, 0 };
    beam prev_beam = b;
    bool sampled_fsd = false;
    bool fsd = false;           // an aperture is in the scratch memory
    u32 aperture_size = 0;
    real throughput = 1;

    u32 depth = 1;
    for (u32 iteration=0;; ++iteration) {
        if (iteration==max_path_iterations) {
            status |= status_iteration_limit;
            break;
        }

        //
        // trace and intersect beam

        cone envelope = b.envelope;
        envelope.r.o = offseted_ray_origin(scene, prev_geo, envelope.r);
        const traversal_t tr = traverse(geometry, envelope, lambda, ray_trace, scratch, status);
        if (status & status_scratch_overflow)
            return status;
        // no intersection found
        if (!tr.hit)
            break;

        const real dist_to_interaction = tr.dist;
        const bool is_ballistic = tr.ballistic || b.is_ray();
        const vec3 origin_wp = tr.origin;
        const vec3 interaction_wp = origin_wp + dist_to_interaction*b.dir();

        //
        // evaluate fsd from previous interaction

        if (fsd) {
            const cone prev_cone = prev_beam.envelope;
            const real f = do_fsd(scene, geometry, prev_cone, prev_geo, interaction_wp,
                                  scratch.aperture, aperture_size, status).intensity();
            fsd = false;

            if (sampled_fsd)
                b.S = b.S*f;
            else {
                prev_beam.transform_region_interaction(origin_wp, length(origin_wp-prev_cone.r.o), b.dir(), f);
                b.add(prev_beam);
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

        // the surface intersection
        surface s;
        if (has_surface) {
            const bvh_tri tri = scene.bvh.tris[primary.tri];
            const tri_shading sh = scene.tris[primary.tri];
            s.tri = primary.tri;
            s.wp = origin_wp + primary.dist*b.dir();
            s.geo = build_shading_frame(tri.n, sh.dpdu);
            const real bz = 1 - primary.baryx - primary.baryy;
            s.shading = build_shading_frame(normalize(primary.baryx*sh.n[0] + primary.baryy*sh.n[1] + bz*sh.n[2]),
                                            sh.dpdu);
            s.fp = b.surface_footprint_static(s.geo, dist_to_interaction);
        }

        // edges: around a ballistic intersection, or those of the diffusive intersection's triangles
        u32 edge_count = 0;
        if (is_ballistic && !b.is_ray() && !ray_trace) {
            const real zdist = b.envelope.axes(dist_to_interaction).x * 2;
            tri_sink sink = { scratch.tris, scratch.tri_capacity, 0 };
            const cone_hit ch = geometry.intersect_cone(
                    b.envelope, range{ dist_to_interaction - zdist/2, dist_to_interaction + zdist/2 }, 1, sink);
            status |= ch.status;
            if (ch.tri_count>scratch.tri_capacity ||
                !collect_edges(scene, scratch.tris, ch.tri_count, scratch.edges, scratch.edge_capacity, edge_count))
                return status | status_scratch_overflow;
        } else if (!is_ballistic && opts.fsd!=0) {
            if (!collect_edges(scene, scratch.tris, tr.tri_count, scratch.edges, scratch.edge_capacity, edge_count))
                return status | status_scratch_overflow;
        }

        // if we have edges, construct the fsd aperture
        if (edge_count>0) {
            aperture_size = build_aperture(scene.edges, scratch.edges, edge_count,
                                           interaction_wp, b.envelope.frame(), b.footprint3(dist_to_interaction),
                                           -b.dir(), scratch.aperture);
            fsd = aperture_size>0;
        }

        //
        // NEE: connect the diffracted beam to the sensor

        if (depth<opts.max_depth && fsd) {
            const sensor_direct_t sd = sensor_sample_direct(scene.sensor, interaction_wp, rnd);
            if (sd.importance>real(0)) {
                const real f = do_fsd(scene, geometry, b.envelope, prev_geo, sd.p,
                                      scratch.aperture, aperture_size, status).intensity();
                if (f!=real(0)) {
                    beam fsd_beam = b;
                    fsd_beam.transform_region_interaction(interaction_wp, dist_to_interaction, -sd.wd, f);
                    if (fsd_beam.S.s0>real(0))
                        film.splat(scene.film, sd.ex, sd.ey, sd.offset, sd.importance * fsd_beam.S.s0 * em.flux);
                }
            }
        }

        //
        // sensing (connection to the virtual sensor), up to the interaction

        {
            const real max_distance = interaction_region_end -
                                      max(real(0), dot(b.dir(), origin_wp - b.origin()));
            const sensor_hit_t sensed = virtual_plane_hit(scene.sensor, b, range{ 0, max_distance });
            if (sensed.hit && b.S.s0>real(0))
                film.splat(scene.film, sensed.ex, sensed.ey, sensed.offset,
                           b.S.s0 * em.flux * scene.sensor.importance);
        }

        // the path ends after max_depth vertices: a further interaction contributes nothing
        if (depth>=opts.max_depth)
            break;

        //
        // do interactions

        bool sampled_null = false;
        if (has_surface) {
            // sample_surface_interaction
            const vec3 wiworld = -b.dir();
            const vec3 wi = s.shading.to_local(wiworld);
            if (dot(wiworld, s.geo.n)*wi.z<=real(0))
                break;

            bsdf_sample_t bs;
            if (!sample_bsdf(scene.materials[scene.tris[s.tri].material], wi, rnd, bs) || !(bs.pdf>real(0)))
                break;

            const vec3 woworld = normalize(s.shading.to_world(bs.wo));
            if (dot(woworld, s.geo.n)*bs.wo.z<=real(0))
                break;

            prev_geo = { s.wp, geo_surface, s.tri };
            prev_beam = b;
            sampled_fsd = false;
            b.transform_surface_interaction(s, woworld, bs.M, 1);
            throughput *= bs.M.m00;
            if (bs.eta!=real(1))
                throughput /= sqr(bs.eta);
        } else if (fsd) {
            // sample_fsd_interaction
            const fsd_sample_t fs = fsd_sample(scene.edges, scratch.aperture, aperture_size, k,
                                               interaction_wp, prev_geo.wp, rnd);
            prev_geo = { interaction_wp, geo_point, 0 };
            prev_beam = b;
            sampled_fsd = true;
            b.transform_region_interaction(interaction_wp, dist_to_interaction, fs.wo, fs.weight);
            throughput *= fs.weight;
        } else {
            // no edges: a null interaction, continue propagation
            sampled_null = true;
            b.transform_restart(interaction_wp, dist_to_interaction);
        }

        //
        // continue_walk

        if (b.S.s0==real(0))
            break;
        if (!sampled_null && opts.russian_roulette!=0) {
            const real p = throughput<real(1) ? max(throughput, real(.5)) : real(1);
            if (!(rnd.r()<=p))
                break;
            const real scale = real(1)/p;
            b.S = b.S*scale;
            throughput *= scale;
        }
        if (!sampled_null)
            ++depth;
    }

    return status;
}

}
