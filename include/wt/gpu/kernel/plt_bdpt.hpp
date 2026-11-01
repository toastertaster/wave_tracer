/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "fraunhofer.hpp"

/*
 * plt_bdpt on the GPU: a port of `plt_bdpt_t::integrate` (src/integrator/plt_bdpt.cpp)
 * and of wt/integrator/plt_bdpt/{plt_bdpt_detail,vertex}.hpp. A path is a sensor
 * subpath and an emitter subpath, random walks of beams whose vertices are kept in
 * memory (`bdpt_vertex`), and the connections of their prefixes (`t` sensor and
 * `s` emitter vertices), weighted by multiple importance sampling. Connections to
 * the sensor (`t` of 0 or 1) splat into the film's light image, the others, summed,
 * into the sensor element the sensor subpath started from.
 *
 * Beams propagate ballistically and diffusively (`traverse`), interact with
 * surfaces, and, with the `FSD` option, diffract where an interaction region
 * holds edges but no surface (Fraunhofer free-space diffraction, see
 * fraunhofer.hpp). (Ray tracing never diffracts.)
 *
 * `trace_bdpt_path` traces a path, parameterized by the backend's Geometry and Film
 * as plt_path's functions are. A path keeps its vertices, and the apertures of
 * its diffractions, in memory the backend provides, up to capacities; a path
 * that outgrows them, or its scratch memory, stops with `status_scratch_overflow`
 * before it splats anything, to be re-traced with more (`scratch_overflow`). A backend may also take
 * over the shadow rays of connections, and their splats (`trace_bdpt_path`'s
 * `deferred`), to trace them in batches.
 */

namespace wt { namespace gpu { namespace compute_kernel {

WT_GPU_CONSTANT constexpr u32 vertex_sensor = 0;
WT_GPU_CONSTANT constexpr u32 vertex_emitter = 1;
WT_GPU_CONSTANT constexpr u32 vertex_surface = 2;
WT_GPU_CONSTANT constexpr u32 vertex_fsd = 3;             // a free-space diffraction

/**
 * @brief A subpath vertex (`plt_bdpt::vertex_t`), and the beam that arrived at
 *        it: of a sensor subpath (`P`: of a polarimetric sensor), or of an
 *        emitter subpath, whose (forward) beams carry Stokes parameters only
 *        (`P` false). Aligned for 16-byte copies to and from memory.
 */
template <bool P>
struct alignas(16) bdpt_vertex {
    beam_t<P> beam;             // the beam that arrived at the vertex (of a subpath's first: the beam it sourced)
    vec3 wp;
    /**
     * @brief `vertex_t::ng()`: the geometric normal of surfaces and area
     *        emitters, and (0,0,1) for others, even for the points of virtual
     *        plane sensors, which are on a surface (`on_surface`) too.
     */
    vec3 ng;
    frame shading;              // surfaces: the shading frame
    material mat;               // surfaces: the material, textured
    fraunhofer::fsd_aperture fsd;   // free-space diffractions: the aperture (in the frame of `beam`)
    u32 type;                   // vertex_sensor, vertex_emitter, vertex_surface or vertex_fsd
    u32 on_surface;             // `vertex_t::is_on_surface()`
    u32 delta;                  // surfaces: a delta BSDF lobe sampled the next vertex
    u32 tri;                    // surfaces and area emitters: the BVH triangle, or no_triangle
    u32 emitter;                // emitters, and surfaces of area emitters: the emitter, or no_emitter
    real pdf;                   // area density of sampling the vertex from the previous one of its subpath
    real pdf_rev;               // ... from the next one (`pdf_reversed`); 0 until there is one
    real rr_weight;             // the subpath's Russian roulette weight, before the vertex's
};

/** @brief Where a vertex is, for densities of sampling it (see `area_density`). */
struct bdpt_point {
    vec3 wp;
    vec3 ng;
    bool on_surface;
};

template <bool P>
WT_GPU_FN bdpt_point point_of(const WT_GPU_GLOBAL bdpt_vertex<P>& v) {
    return { v.wp, v.ng, v.on_surface!=0 };
}

/** @brief `vertex_t::geo`: where shadow rays to and from a vertex start. */
template <bool P>
WT_GPU_FN vertex_geo geo_of(const WT_GPU_GLOBAL bdpt_vertex<P>& v) {
    return v.tri!=no_triangle ? vertex_geo{ v.wp, geo_surface, v.tri } : vertex_geo{ v.wp, geo_point, 0 };
}

#if defined(__METAL_VERSION__)
// Metal: of a vertex being built, in thread memory (elsewhere, the overloads above take it)
template <bool P>
WT_GPU_FN bdpt_point point_of(const thread bdpt_vertex<P>& v) {
    return { v.wp, v.ng, v.on_surface!=0 };
}
template <bool P>
WT_GPU_FN vertex_geo geo_of(const thread bdpt_vertex<P>& v) {
    return v.tri!=no_triangle ? vertex_geo{ v.wp, geo_surface, v.tri } : vertex_geo{ v.wp, geo_point, 0 };
}
#endif

/** @brief `vertex_t::is_connectible` */
template <bool P>
WT_GPU_FN bool is_connectible(const WT_GPU_GLOBAL bdpt_vertex<P>& v) {
    return v.type==vertex_fsd || (v.type==vertex_surface && v.mat.delta_only==0);
}

/** @brief `vertex_t::ns()`: the shading normal of surfaces (and area emitters, which connections never use). */
template <bool P>
WT_GPU_FN vec3 ns_of(const WT_GPU_GLOBAL bdpt_vertex<P>& v) {
    return v.type==vertex_surface ? v.shading.n : vec3{ 0,0,1 };
}

/** @brief The frame of a vertex's arriving beam (`beam_generic_t::frame`): of a diffraction, its aperture's. */
template <bool P>
WT_GPU_FN frame beam_frame_of(const WT_GPU_GLOBAL bdpt_vertex<P>& v) {
    return { v.beam.envelope.x, v.beam.envelope.y, v.beam.envelope.r.d };
}

/**
 * @brief A path's memory of its diffractions' apertures, of `capacity` edges,
 *        `used` of which by its vertices so far; and its wavenumber.
 */
struct fsd_memory {
    WT_GPU_GLOBAL fraunhofer::fsd_edge* edges;
    u32 capacity;
    u32 used;
    real k;
};

/** @brief `free_space_diffraction_t::pdf` (and `f`) of a diffraction vertex, of the world direction `wo`. */
template <bool P>
WT_GPU_FN real fsd_pdf_of(const WT_GPU_GLOBAL bdpt_vertex<P>& v, const WT_GPU_THREAD fsd_memory& apertures,
                          vec3 wo) {
    const fraunhofer::fsd_aperture ap = v.fsd;
    return fraunhofer::fsd_pdf(ap, apertures.edges, apertures.k, beam_frame_of(v).to_local(wo));
}

/** @brief `shading_normals_correction_scale` of wt/integrator/common.hpp (forward transport). */
WT_GPU_FN real shading_normals_correction(bool backward, real wig, real wog, real wis, real wos) {
    if (backward)
        return 1;
    return min(abs(wis*wog/(wos*wig)), real(100));
}


/*
 * Sampling densities of vertices (`vertex_t::pdf*`), as area densities.
 */

/**
 * @brief `vertex_t::convert_directional_density_to_area`: a solid-angle
 *        density `dpdf` (0: discrete) at `p` of sampling `next`, as an area
 *        density at `next`.
 */
WT_GPU_FN real area_density(real dpdf, vec3 p, bdpt_point next) {
    if (dpdf==real(0))
        return 0;
    const vec3 d = next.wp - p;
    const real d2 = length2(d);
    if (d2==real(0))
        return inf();
    real ppdf = dpdf * (1/d2);
    if (next.on_surface)
        ppdf *= abs(dot(next.ng, normalize(d)));
    return ppdf;
}

/** @brief `sensor_t::pdf_direction`: the solid-angle density of the sensor sampling direction `d`. */
WT_GPU_FN real sensor_pdf_direction(const WT_GPU_THREAD sensor_desc& s, vec3 d) {
    if (s.type==sensor_perspective) {
        const vec3 local = transform_direction(s.dir_to_local, d);
        return local.z>epsilon ? 1/perspective_recp_dpd(s, local) : real(0);
    }
    return inv_pi * max(real(0), dot(d, s.f.n));
}

/** @brief `vertex_t::pdf_next_from_sensor`: of the sensor point `p` sampling `next`. */
WT_GPU_FN real pdf_next_from_sensor(const WT_GPU_THREAD sensor_desc& s, vec3 p, bdpt_point next) {
    const vec3 dl = next.wp - p;
    const real recp_dist2 = 1 / length2(dl);
    const vec3 d = dl * sqrt(recp_dist2);
    real ppdf = sensor_pdf_direction(s, d) * recp_dist2;
    if (next.on_surface)
        ppdf *= abs(dot(next.ng, d));
    return ppdf;
}

/** @brief `vertex_t::pdf_sensor`: of the sensor sampling a sensor point (0: discrete). */
WT_GPU_FN real pdf_sensor(const WT_GPU_THREAD sensor_desc& s) {
    return s.type==sensor_perspective ? real(0) : s.recp_area;
}

/** @brief `vertex_t::is_delta_sensor` */
WT_GPU_FN bool is_delta_sensor(const WT_GPU_THREAD sensor_desc& s) {
    return s.type==sensor_perspective;
}

/**
 * @brief `vertex_t::pdf_next_from_emitter`: of the point `p` of emitter `ei`
 *        (of normal `ng` on an area emitter) sampling `next`.
 */
WT_GPU_FN real pdf_next_from_emitter(const WT_GPU_THREAD plt_scene& scene, u32 ei, vec3 p, vec3 ng, bdpt_point next) {
    const WT_GPU_GLOBAL emitter& em = scene.emitters[ei];
    if (em.type==emitter_directional) {
        // infinite_emitter_t::pdf_target_position
        const vec3 v = next.wp - em.target_centre;
        const vec2 plocal = { dot(v, em.target.t), dot(v, em.target.b) };
        return length2(plocal)<=sqr(em.target_radius) ? 1/(pi*sqr(em.target_radius)) : real(0);
    }

    const vec3 dl = next.wp - p;
    const real recp_dist2 = 1 / length2(dl);
    const vec3 d = dl * sqrt(recp_dist2);
    // emitter_t::pdf_direction
    const real dpdf = em.type==emitter_point ? inv_four_pi :
                      em.type==emitter_spot  ? 1/em.cutoff_solid_angle :
                                               inv_pi * max(real(0), dot(d, ng));
    real ppdf = dpdf * recp_dist2;
    if (next.on_surface)
        ppdf *= abs(dot(next.ng, d));
    return ppdf;
}

/** @brief `vertex_t::pdf_emitter`: of sampling emitter `ei`, and a point on it (0: discrete). */
WT_GPU_FN real pdf_emitter(const WT_GPU_THREAD plt_scene& scene, u32 ei) {
    const WT_GPU_GLOBAL emitter& em = scene.emitters[ei];
    // point and spot emitters' positions are discrete, the directional emitter infinite
    return em.type==emitter_area ? em.pmf/em.area : real(0);
}

/** @brief `vertex_t::is_delta_emitter` */
WT_GPU_FN bool is_delta_emitter(const WT_GPU_THREAD plt_scene& scene, u32 ei) {
    return scene.emitters[ei].type!=emitter_area;
}

/**
 * @brief `vertex_t::pdf` of a surface or diffraction vertex `v`: of sampling
 *        `next` from it, having come from `prev`.
 */
template <bool P>
WT_GPU_FN real vertex_pdf(const WT_GPU_GLOBAL bdpt_vertex<P>& v, const WT_GPU_THREAD fsd_memory& apertures,
                          vec3 prev, bdpt_point next) {
    const vec3 woworld = normalize(next.wp - v.wp);
    if (v.type==vertex_fsd)
        return area_density(fsd_pdf_of(v, apertures, woworld), v.wp, next);

    const frame sh = v.shading;
    const material m = v.mat;
    const vec3 wi = sh.to_local(normalize(prev - v.wp));
    const vec3 wo = sh.to_local(woworld);
    return area_density(pdf_bsdf(m, wi, wo), v.wp, next);
}


/*
 * Subpaths
 */

/** @brief A subpath's random walk (`bdpt_walk_data_t`): its beam, and sampling data. */
template <bool P>
struct bdpt_walk_t {
    beam_t<P> b;
    real pdf_from_prev;         // solid-angle density of the beam's direction, or 0 if discrete
    real throughput;
    real rr_weight;
};

/**
 * @brief Starts a sensor subpath (`generate_sensor_subpath`): samples a beam
 *        from sensor element (ex,ey) at the wavenumber of `spec`, and writes
 *        the sensor vertex, `verts[0]`. `offset` receives the beam's offset in
 *        the element.
 */
template <bool P>
WT_GPU_FN bdpt_walk_t<P> start_sensor_subpath(const WT_GPU_THREAD plt_scene& scene,
                                              const WT_GPU_THREAD spectral_sample& spec, u32 ex, u32 ey,
                                              WT_GPU_THREAD rng& rnd, WT_GPU_THREAD vec2& offset,
                                              WT_GPU_GLOBAL bdpt_vertex<P>* verts) {
    bdpt_walk_t<P> w;
    const sensor_sample_t ss = sample_sensor(scene.sensor, spec, ex, ey, rnd, w.b);
    offset = ss.offset;
    w.pdf_from_prev = ss.dpd;
    w.throughput = 1;
    w.rr_weight = 1;

    // vertex_t::create_sensor: virtual plane sensors sample points on a surface
    bdpt_vertex<P> v;
    v.beam = w.b;
    v.wp = w.b.origin();
    v.ng = { 0,0,1 };
    v.type = vertex_sensor;
    v.on_surface = scene.sensor.type==sensor_virtual_plane ? 1u : 0u;
    v.delta = 0;
    v.tri = no_triangle;
    v.emitter = no_emitter;
    v.pdf = ss.ppd;
    v.pdf_rev = 0;
    v.rr_weight = 1;
    verts[0] = v;
    return w;
}

/**
 * @brief Starts an emitter subpath (`generate_emitter_subpath`): sources a beam
 *        from emitter `ei` at spectral sample `spectral` (`emitter_t::sample`,
 *        not divided by the emitter's sampling probability, as the CPU's), and
 *        writes the emitter vertex, `verts[0]`.
 */
WT_GPU_FN bdpt_walk_t<false> start_emitter_subpath(const WT_GPU_THREAD plt_scene& scene, u32 ei, u32 spectral,
                                                   WT_GPU_THREAD rng& rnd, WT_GPU_GLOBAL bdpt_vertex<false>* verts) {
    const WT_GPU_GLOBAL emitter& em = scene.emitters[ei];
    const emission es = scene.emissions[spectral*scene.emitter_count + ei];

    bdpt_vertex<false> v;
    v.ng = { 0,0,1 };
    v.on_surface = 0;
    v.tri = no_triangle;

    // the beam's ray, intensity scale, and the densities of its position (0: discrete) and direction (0: discrete)
    ray r;
    real scale = 1;
    real ppd, dpd;
    if (em.type==emitter_point) {
        r = make_ray(em.position, uniform_sphere(rnd.r2()));
        ppd = 0;
        dpd = inv_four_pi;
    } else if (em.type==emitter_spot) {
        // uniform over the cutoff cone, scaled by the falloff
        const vec3 local = uniform_cone(em.cutoff_solid_angle, rnd.r2());
        r = make_ray(em.position, transform_direction(em.dir_to_world, local));
        scale = spot_falloff(em, local);
        ppd = 0;
        dpd = 1/em.cutoff_solid_angle;
    } else if (em.type==emitter_directional) {
        // from outside the world, through a point on the target disc
        const vec2 d = concentric_disk(rnd.r2()) * em.target_radius;
        const vec3 wp = em.target_centre + d.x*em.target.t + d.y*em.target.b;
        r = make_ray(wp + em.far*em.target.n, -em.target.n);
        ppd = 1/(pi*sqr(em.target_radius));
        dpd = 0;
    } else {
        // area: from a point on the shape, cosine-distributed about its normal
        const area_point_t ap = sample_area_point(scene, em, rnd);
        const frame geo = build_shading_frame(scene.bvh.tris[ap.tri].n, scene.tris[ap.tri].dpdu);
        const vec3 local = cosine_hemisphere(rnd.r2());
        r = make_ray(ap.p, geo.to_world(local));
        ppd = 1/em.area;
        dpd = inv_pi * local.z;
        v.on_surface = 1;
        v.ng = geo.n;
        v.tri = ap.tri;
    }

    bdpt_walk_t<false> w;
    w.b.envelope = make_isotropic_cone(r, es.tan_alpha, es.x0);
    w.b.S = { scale*es.flux,0,0,0 };
    w.b.Sframe = w.b.envelope.frame();
    w.b.self_intersection_distance = 0;
    w.pdf_from_prev = dpd;
    w.throughput = 1;
    w.rr_weight = 1;

    // vertex_t::create_emitter
    v.beam = w.b;
    v.wp = r.o;
    v.type = vertex_emitter;
    v.delta = 0;
    v.emitter = ei;
    v.pdf = ppd * em.pmf;
    v.pdf_rev = 0;
    v.rr_weight = 1;
    verts[0] = v;
    return w;
}

/**
 * @brief The random walk of a subpath (`random_walk`) after its first vertex,
 *        `verts[0]`: appends a vertex for each surface interaction and
 *        free-space diffraction, up to `capacity` vertices, whose apertures
 *        go to `apertures` (a subpath that needs more of either, or of its
 *        scratch memory, stops with `status_scratch_overflow`). `backward`:
 *        of a sensor subpath. Returns the subpath's vertex count.
 */
template <bool P, typename Geometry>
WT_GPU_FN u32 bdpt_random_walk(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                               const WT_GPU_THREAD path_scratch& scratch, u32 spectral, bool backward,
                               WT_GPU_THREAD bdpt_walk_t<P>& w, WT_GPU_GLOBAL bdpt_vertex<P>* verts, u32 capacity,
                               WT_GPU_THREAD fsd_memory& apertures, WT_GPU_THREAD rng& rnd,
                               WT_GPU_THREAD u32& status) {
    const plt_params opts = scene.params;
    const bool ray_trace = opts.ray_tracing_only!=0;
    const real lambda = two_pi/scene.spectra[spectral].k;

    u32 count = 1;
    bdpt_point last = point_of(verts[0]);
    vertex_geo last_geo = geo_of(verts[0]);
    // appends vertex `v`, sampled with solid-angle density `pdf_fwd` (`append_vertex`), from which the previous
    // vertex is sampled with density `pdf_rev`; false if it may not be
    const auto append_vertex = [&](WT_GPU_THREAD bdpt_vertex<P>& v, real pdf_fwd, real pdf_rev) {
        if (count==capacity) {
            // (a subpath has at most max_depth+2 vertices)
            status |= scratch_overflow(u64(scene.params.max_depth)+2, capacity);
            return false;
        }
        if (last.wp==v.wp)
            return false;
        const bdpt_point point = point_of(v);
        v.beam = w.b;
        v.pdf = area_density(w.pdf_from_prev, last.wp, point);
        v.pdf_rev = 0;
        // continue_walk sets it so, before its Russian roulette
        v.rr_weight = w.rr_weight;
        verts[count-1].pdf_rev = area_density(pdf_rev, v.wp, last);
        verts[count++] = v;
        w.pdf_from_prev = pdf_fwd;
        last = point;
        last_geo = geo_of(v);
        return true;
    };

    for (u32 iteration=0;; ++iteration) {
        if (iteration==max_path_iterations) {
            status |= status_iteration_limit;
            break;
        }

        // traverse from the last vertex (null interactions restart the beam, but traversals start off the
        // last vertex's surface, as the CPU's)
        cone envelope = w.b.envelope;
        envelope.r.o = offseted_ray_origin(scene, last_geo, envelope.r);
        // (free-space diffraction needs to know whether the closest triangle faces the beam)
        const traversal_t tr = traverse<true>(geometry, envelope, lambda, ray_trace, scratch, status);
        if ((status & status_scratch_overflow) || !tr.hit)
            break;

        const vec3 dir = w.b.dir();
        const vec3 interaction_wp = tr.origin + tr.dist*dir;
        const primary_t primary = tr.ballistic || w.b.is_ray() ?
            primary_t{ tr.tri, tr.dist, tr.baryx, tr.baryy } :
            find_closest_triangle(scene, scratch.tris, tr.tri_count, range{ tr.dist, tr.dist+tr.region_depth },
                                  tr.origin, dir);

        if (primary.tri==no_triangle) {
            // a (diffusive) interaction region without a surface: free-space diffraction by its edges, if any
            // (sample_fraunhofer_fsd_interaction); else a null interaction, no vertex, no Russian roulette
            fraunhofer::fsd_aperture ap = {};
            if (opts.fsd!=0) {
                // (the edges' ids fit: a path keeps its apertures elsewhere, in `apertures`)
                u32 edge_count = 0;
                collect_edges(scene, scratch.tris, tr.tri_count, scratch.edges, 3*scratch.tri_capacity, edge_count);
                if (edge_count>0) {
                    // the aperture passes the beam's power but that which the region's triangles block
                    const fraunhofer::gaussian_wavefront wf = fraunhofer::wavefront_at(w.b.envelope, tr.dist);
                    const real power = 1 - fraunhofer::wavefront_flux_over_triangles(
                            scene, scratch.tris, tr.tri_count, w.b.envelope, wf,
                            range{ tr.dist, tr.dist+tr.region_depth }, tr.front_face);
                    ap = fraunhofer::build_fsd_aperture(scene, scratch.edges, edge_count, w.b.envelope, wf,
                                                        apertures.k, power, apertures.edges, apertures.used,
                                                        apertures.capacity-apertures.used, status);
                    if (status & status_scratch_overflow)
                        break;
                }
            }
            if (ap.count==0) {
                w.b.transform_restart(interaction_wp, tr.dist);
                continue;
            }

            const fraunhofer::fsd_sample_t fs = fraunhofer::fsd_sample(scene.fsd_lut, ap, apertures.edges,
                                                                       apertures.k, rnd);
            if (!(fs.dpd>real(0)) || fs.weight==real(0))
                break;

            // the reverse density is taken to be the forward one (as the CPU's)
            bdpt_vertex<P> v;
            v.wp = interaction_wp;
            v.ng = { 0,0,1 };
            v.fsd = ap;
            v.type = vertex_fsd;
            v.on_surface = 0;
            v.delta = 0;
            v.tri = no_triangle;
            v.emitter = no_emitter;
            if (!append_vertex(v, fs.dpd, fs.dpd))
                break;
            apertures.used += ap.count;

            // the diffracted beam
            const vec3 woworld = w.b.envelope.frame().to_world(fs.wo);
            w.b.transform_region_interaction(interaction_wp, tr.dist, woworld, fs.weight);
            w.throughput *= fs.weight;
        } else {
            // sample_surface_interaction
            surface s;
            material mat;
            const u32 surface_emitter = hit_surface(scene, spectral, primary, tr.dist, w.b, s, mat);

            const vec3 wiworld = -dir;
            const vec3 wi = s.shading.to_local(wiworld);
            const real wig = dot(wiworld, s.geo.n);
            if (wig*wi.z<=real(0))
                break;

            bsdf_sample_t bs;
            if (!sample_bsdf(mat, wi, backward, rnd, bs) || !(bs.pdf>real(0)))
                break;
            const vec3 woworld = normalize(s.shading.to_world(bs.wo));
            const real wog = dot(woworld, s.geo.n);
            if (wog*bs.wo.z<=real(0))
                break;

            bdpt_vertex<P> v;
            v.wp = s.wp;
            v.ng = s.geo.n;
            v.shading = s.shading;
            v.mat = mat;
            v.type = vertex_surface;
            v.on_surface = 1;
            v.delta = bs.discrete ? 1u : 0u;
            v.tri = s.tri;
            v.emitter = surface_emitter;
            if (!append_vertex(v, bs.discrete ? real(0) : bs.pdf, pdf_bsdf(mat, bs.wo, wi)))
                break;

            // the beam after the interaction (the adjoint BSDF for shading normals)
            const real scale = s.shading.n==s.geo.n ? real(1) :
                               shading_normals_correction(backward, wig, wog, wi.z, bs.wo.z);
            w.b.transform_surface_interaction(s, woworld, bs.M, scale, backward);
            w.throughput *= scale * bs.M.m00;
            if (backward && bs.eta!=real(1))
                w.throughput /= sqr(bs.eta);
        }

        // continue_walk
        if (count>opts.max_depth+1)
            break;
        if (opts.russian_roulette!=0) {
            const real q = w.throughput<real(1) ? max(w.throughput, real(.5)) : real(1);
            if (!(rnd.r()<=q))
                break;
            const real rr = 1/q;
            w.rr_weight *= rr;
            w.throughput *= rr;
        }
    }
    return count;
}


/*
 * Connections
 */

/**
 * @brief `vertex_t::interact` of a surface or diffraction vertex `v`: the
 *        beam that arrived at it, scattered towards `next` (`backward`: of a
 *        sensor subpath), into `b`; of a surface, but its envelope (see
 *        `beam_t::scatter`). False if the vertex does not scatter so.
 */
template <bool P>
WT_GPU_FN bool scatter_towards(const WT_GPU_GLOBAL bdpt_vertex<P>& v, const WT_GPU_THREAD fsd_memory& apertures,
                               vec3 next, bool backward, WT_GPU_THREAD beam_t<P>& b) {
    const vec3 woworld = normalize(next - v.wp);
    if (v.type==vertex_fsd) {
        // the diffracted beam, weighted by the aperture's ASF (`f`, which is its density)
        const real f = fsd_pdf_of(v, apertures, woworld);
        if (f==real(0))
            return false;
        b = v.beam;
        b.transform_region_interaction(v.wp, dot(v.wp - b.origin(), b.dir()), woworld, f);
        return true;
    }

    const frame sh = v.shading;
    const vec3 wiworld = -v.beam.envelope.r.d;
    const vec3 wi = sh.to_local(wiworld);
    const vec3 wo = sh.to_local(woworld);
    const real wig = dot(wiworld, v.ng);
    const real wog = dot(woworld, v.ng);
    if (wig*wi.z<=real(0) || wog*wo.z<=real(0))
        return false;

    real scale = 1/abs(wo.z);
    if (!(sh.n==v.ng))
        scale *= shading_normals_correction(backward, wig, wog, wi.z, wo.z);
    const material m = v.mat;
    const mueller_block M = eval_bsdf(m, wi, wo, backward) * scale;
    if (M.m00==real(0))
        return false;

    b.S = v.beam.S;
    b.Sframe = v.beam.Sframe;
    b.rows = v.beam.rows;
    // the scattering uses the surface's shading frame only
    surface s;
    s.wp = v.wp;
    s.geo = sh;
    s.shading = sh;
    s.fp = { { 1,0 }, 0, 0 };
    s.tri = v.tri;
    b.scatter(s, woworld, M, backward);
    return true;
}

/**
 * @brief What a sensor's direct beam (of `importance`, an identity operator in
 *        frame `f`, the frame of the beam the sensor sources) observes of a
 *        forward beam `b` (`integrate_beams`).
 */
template <bool P>
WT_GPU_FN observed_t<P> observe_directly(real importance, frame f, const WT_GPU_THREAD beam_t<false>& b) {
    beam_t<P> db;
    db.S = { importance,0,0,0 };
    db.rows.identity(importance);
    db.Sframe = f;
    return db.observe(b.S, b.Sframe);
}

WT_GPU_FN real intensity_of(real I) { return I; }
WT_GPU_FN real intensity_of(stokes S) { return S.s0; }

/** @brief What `bdpt_compute_mis_weight` sets for a connection: the densities at its ends, and whether the subpaths start at delta points. */
struct bdpt_mis_t {
    real sensor_rev[2];         // pdf_rev of sensor vertices t-1 and t-2: the first `sensor_revs`
    real emitter_rev[2];        // ... of emitter vertices s-1 and s-2: the first `emitter_revs`
    u32 sensor_revs, emitter_revs;
    real sensor_pdf0;           // t==1: the pdf of the sensor point sampled
    real emitter_pdf0;          // s==1: the pdf of the emitter point sampled
    bool delta_sensor, delta_emitter;
};

/**
 * @brief A connection of `t` sensor and `s` emitter vertices: what the sensor
 *        observes, and where (t<=1). Connections test whether their segments
 *        are blocked (`shadowed`), observing nothing if so; or else leave it
 *        to test later (see `trace_bdpt_path`): then `shadow`, of the
 *        segment `from`-`to`.
 */
template <bool P>
struct bdpt_connection_t {
    observed_t<P> L;            // radiant flux (Stokes parameters, if polarimetric); zero if none
    u32 ex, ey;                 // t<=1: the sensor element connected to, and offset
    vec2 offset;
    bdpt_mis_t mis;             // if L is not zero
    bool shadow = false;
    vertex_geo from, to;
};

/**
 * @brief The segment `a`-`b` of connection `c`: tested now (false if it is
 *        blocked), or, if `Late`, left to test later.
 */
template <bool Late, bool P, typename Geometry>
WT_GPU_FN bool connection_segment(WT_GPU_THREAD bdpt_connection_t<P>& c, const WT_GPU_THREAD plt_scene& scene,
                                  const WT_GPU_THREAD Geometry& geometry, vertex_geo a, vertex_geo b,
                                  WT_GPU_THREAD u32& status) {
    if (!Late)
        return !shadowed(scene, geometry, a, b, status);
    c.shadow = true;
    c.from = a;
    c.to = b;
    return true;
}

/** @brief `area_density_or_one` of `bdpt_compute_mis_weight` */
WT_GPU_FN real density_or_one(real p) {
    return isfinite(p) && p>epsilon ? p : real(1);
}

/** @brief `bdpt_compute_mis_weight`: the balance heuristic over the strategies that sample the connected path. */
template <bool P>
WT_GPU_FN real bdpt_mis_weight(i32 s, i32 t,
                               const WT_GPU_GLOBAL bdpt_vertex<P>* sv, const WT_GPU_GLOBAL bdpt_vertex<false>* ev,
                               const WT_GPU_THREAD bdpt_mis_t& m) {
    if (s+t<=2)
        return 1;

    real sum = 0;
    real ri = 1;
    for (i32 i=t-1; i>=0; --i) {
        const real pdf = t==1 ? m.sensor_pdf0 : sv[i].pdf;
        const i32 j = t-1-i;
        const real rev = j<i32(m.sensor_revs) ? (j==0 ? m.sensor_rev[0] : m.sensor_rev[1]) : sv[i].pdf_rev;
        ri *= density_or_one(rev) / density_or_one(pdf);
        // the connected vertex is not delta
        const bool delta = i<t-1 && sv[i].delta!=0;
        const bool delta_prev = i>0 ? sv[i-1].delta!=0 : m.delta_sensor;
        if (!delta && !delta_prev)
            sum += ri;
    }

    ri = 1;
    for (i32 i=s-1; i>=0; --i) {
        const real pdf = s==1 ? m.emitter_pdf0 : ev[i].pdf;
        const i32 j = s-1-i;
        const real rev = j<i32(m.emitter_revs) ? (j==0 ? m.emitter_rev[0] : m.emitter_rev[1]) : ev[i].pdf_rev;
        ri *= density_or_one(rev) / density_or_one(pdf);
        const bool delta = i<s-1 && ev[i].delta!=0;
        const bool delta_prev = i>0 ? ev[i-1].delta!=0 : m.delta_emitter;
        if (!delta && !delta_prev)
            sum += ri;
    }

    return 1/(1+sum);
}

/**
 * @brief `connect_subpaths` with s==0: the sensor subpath's last vertex is
 *        on an area emitter, which the beam that arrived there observes
 *        (`area_t::Li`).
 */
template <bool P>
WT_GPU_FN bdpt_connection_t<P> connect_to_emitter_surface(const WT_GPU_THREAD plt_scene& scene, u32 spectral, i32 t,
                                                          const WT_GPU_GLOBAL bdpt_vertex<P>* sv) {
    bdpt_connection_t<P> c;
    c.L = {};
    const WT_GPU_GLOBAL bdpt_vertex<P>& last = sv[t-1];
    if (last.emitter==no_emitter || !(dot(-last.beam.envelope.r.d, last.ng)>real(0)))
        return c;

    const real Le = scene.emissions[spectral*scene.emitter_count + last.emitter].intensity;
    beam_t<P> b;
    b.S = last.beam.S;
    b.rows = last.beam.rows;
    c.L = b.observe_unpolarized(Le * last.rr_weight);

    c.mis.sensor_rev[0] = pdf_emitter(scene, last.emitter);
    c.mis.sensor_rev[1] = pdf_next_from_emitter(scene, last.emitter, last.wp, last.ng, point_of(sv[t-2]));
    c.mis.sensor_revs = 2;
    c.mis.emitter_revs = 0;
    c.mis.delta_sensor = is_delta_sensor(scene.sensor);
    c.mis.delta_emitter = true;
    return c;
}

/**
 * @brief `connect_subpaths` with s==1: connects the sensor subpath's last
 *        vertex to a sampled emitter point (`scene_t::sample_emitter_direct`).
 */
template <bool Late, bool P, typename Geometry>
WT_GPU_FN bdpt_connection_t<P> connect_to_emitter(const WT_GPU_THREAD plt_scene& scene,
                                                  const WT_GPU_THREAD Geometry& geometry, u32 spectral, i32 t,
                                                  const WT_GPU_GLOBAL bdpt_vertex<P>* sv,
                                                  const WT_GPU_THREAD fsd_memory& apertures,
                                                  WT_GPU_THREAD rng& rnd, WT_GPU_THREAD u32& status) {
    bdpt_connection_t<P> c;
    c.L = {};
    const WT_GPU_GLOBAL bdpt_vertex<P>& last = sv[t-1];
    if (!is_connectible(last))
        return c;

    const emitter_direct_t ed = sample_emitter_direct(scene, spectral, last.wp, rnd);
    if (!(ed.I>real(0)))
        return c;
    real I = ed.I * last.rr_weight;
    if (last.on_surface!=0)
        I *= abs(dot(ed.wo, ns_of(last)));

    // the beam that arrived at the last vertex, towards the emitter point
    beam_t<P> db;
    if (!scatter_towards(last, apertures, ed.geo.wp, true, db) || db.S.s0==real(0) || I==real(0))
        return c;
    if (!connection_segment<Late>(c, scene, geometry, geo_of(last), ed.geo, status))
        return c;
    c.L = db.observe_unpolarized(I);

    // the emitter point (`create_emitter(emitter_direct_sample_t)`)
    const bool area = ed.geo.kind==geo_surface;
    const bdpt_point e = { ed.geo.wp, area ? scene.bvh.tris[ed.geo.index].n : vec3{ 0,0,1 }, area };
    c.mis.sensor_rev[0] = pdf_next_from_emitter(scene, ed.emitter, e.wp, e.ng, point_of(last));
    c.mis.emitter_rev[0] = vertex_pdf(last, apertures, sv[t-2].wp, e);
    c.mis.emitter_pdf0 = pdf_emitter(scene, ed.emitter);
    c.mis.sensor_revs = 1;
    c.mis.emitter_revs = 1;
    c.mis.delta_sensor = is_delta_sensor(scene.sensor);
    c.mis.delta_emitter = is_delta_emitter(scene, ed.emitter);
    return c;
}

/**
 * @brief `connect_subpaths` with t==1: connects the emitter subpath's last
 *        vertex to a sampled sensor point (`sensor_t::sample_direct`).
 *        Diffractions connect only to virtual plane sensors, as the CPU's.
 */
template <bool Late, bool P, typename Geometry>
WT_GPU_FN bdpt_connection_t<P> connect_to_sensor(const WT_GPU_THREAD plt_scene& scene,
                                                 const WT_GPU_THREAD Geometry& geometry, i32 s,
                                                 const WT_GPU_GLOBAL bdpt_vertex<false>* ev,
                                                 const WT_GPU_THREAD fsd_memory& apertures,
                                                 WT_GPU_THREAD rng& rnd, WT_GPU_THREAD u32& status) {
    bdpt_connection_t<P> c;
    c.L = {};
    const WT_GPU_GLOBAL bdpt_vertex<false>& last = ev[s-1];
    if (!is_connectible(last) || (last.type==vertex_fsd && scene.sensor.type!=sensor_virtual_plane))
        return c;

    const sensor_direct_t sd = sensor_sample_direct(scene.sensor, scene.film, last.wp, rnd);
    if (!(sd.importance>real(0)))
        return c;
    real importance = sd.importance * last.rr_weight;
    if (last.on_surface!=0)
        importance *= abs(dot(sd.wd, ns_of(last)));

    // the beam that arrived at the last vertex, towards the sensor point
    beam_t<false> eb;
    if (!scatter_towards(last, apertures, sd.p, false, eb) || eb.S.s0==real(0) || importance==real(0))
        return c;
    if (!connection_segment<Late>(c, scene, geometry, vertex_geo{ sd.p, geo_point, 0 }, geo_of(last), status))
        return c;
    c.L = observe_directly<P>(importance, build_orthogonal_frame(sd.wd), eb);
    c.ex = sd.ex;
    c.ey = sd.ey;
    c.offset = sd.offset;

    // the sensor point (`create_sensor(sensor_direct_sample_t)`)
    const bdpt_point e = { sd.p, { 0,0,1 }, scene.sensor.type==sensor_virtual_plane };
    c.mis.emitter_rev[0] = pdf_next_from_sensor(scene.sensor, sd.p, point_of(last));
    c.mis.sensor_rev[0] = vertex_pdf(last, apertures, ev[s-2].wp, e);
    c.mis.sensor_pdf0 = pdf_sensor(scene.sensor);
    c.mis.sensor_revs = 1;
    c.mis.emitter_revs = 1;
    c.mis.delta_sensor = is_delta_sensor(scene.sensor);
    c.mis.delta_emitter = is_delta_emitter(scene, ev[0].emitter);
    return c;
}

/**
 * @brief `connect_subpaths` with t==0: the emitter subpath's last segment
 *        crosses a virtual plane sensor (`virtual_plane_sensor_t::Si`).
 */
template <bool P>
WT_GPU_FN bdpt_connection_t<P> connect_through_sensor(const WT_GPU_THREAD plt_scene& scene, i32 s,
                                                      const WT_GPU_GLOBAL bdpt_vertex<false>* ev) {
    bdpt_connection_t<P> c;
    c.L = {};
    if (scene.sensor.type!=sensor_virtual_plane)
        return c;

    // the beam that arrived at the last vertex, from the one before (or a null interaction)
    const WT_GPU_GLOBAL bdpt_vertex<false>& last = ev[s-1];
    const WT_GPU_GLOBAL bdpt_vertex<false>& current = ev[s-2];
    const beam_t<false> b = last.beam;
    const sensor_hit_t hit = virtual_plane_hit(scene.sensor, b, range{ 0, length(last.wp - b.origin()) });
    if (!hit.hit)
        return c;

    // the sensor's beam, of the opposite direction: Se/dn, weighted
    const vec3 d = -b.dir();
    real w = current.rr_weight;
    if (current.type==vertex_surface && current.delta==0)
        w /= abs(dot(d, current.shading.n));
    w /= abs(dot(d, scene.sensor.f.n));
    c.L = observe_directly<P>(scene.sensor.importance * w, build_orthogonal_frame(d), b);
    c.ex = hit.ex;
    c.ey = hit.ey;
    c.offset = hit.offset;

    c.mis.emitter_rev[0] = pdf_sensor(scene.sensor);
    c.mis.emitter_rev[1] = pdf_next_from_sensor(scene.sensor, hit.p, point_of(current));
    c.mis.sensor_revs = 0;
    c.mis.emitter_revs = 2;
    c.mis.delta_sensor = true;
    c.mis.delta_emitter = is_delta_emitter(scene, ev[0].emitter);
    return c;
}

/**
 * @brief `connect_subpaths` with s>=2 and t>=2: connects the subpaths' last
 *        vertices. Diffractions are connectible, but contribute nothing so:
 *        the CPU's connections ignore their scattering (`ignore_fsd`),
 *        weighting their beams by 0.
 */
template <bool Late, bool P, typename Geometry>
WT_GPU_FN bdpt_connection_t<P> connect_vertices(const WT_GPU_THREAD plt_scene& scene,
                                                const WT_GPU_THREAD Geometry& geometry, i32 s, i32 t,
                                                const WT_GPU_GLOBAL bdpt_vertex<P>* sv,
                                                const WT_GPU_GLOBAL bdpt_vertex<false>* ev,
                                                const WT_GPU_THREAD fsd_memory& apertures,
                                                WT_GPU_THREAD u32& status) {
    bdpt_connection_t<P> c;
    c.L = {};
    const WT_GPU_GLOBAL bdpt_vertex<false>& e = ev[s-1];
    const WT_GPU_GLOBAL bdpt_vertex<P>& v = sv[t-1];
    const vec3 dl = e.wp - v.wp;
    if (!is_connectible(e) || !is_connectible(v) || e.type==vertex_fsd || v.type==vertex_fsd ||
        dl==vec3{ 0,0,0 })
        return c;

    // the beams that arrived at each, towards the other
    beam_t<false> eb;
    beam_t<P> db;
    if (!scatter_towards(e, apertures, v.wp, false, eb) || !scatter_towards(v, apertures, e.wp, true, db))
        return c;

    const real recp_d2 = 1 / length2(dl);
    const vec3 d = dl * sqrt(recp_d2);
    eb.scale(e.rr_weight * abs(dot(v.shading.n, d)));
    db.scale(v.rr_weight * recp_d2 * abs(dot(e.shading.n, d)));
    if (db.S.s0==real(0) || eb.S.s0==real(0))
        return c;
    if (!connection_segment<Late>(c, scene, geometry, geo_of(v), geo_of(e), status))
        return c;
    c.L = db.observe(eb.S, eb.Sframe);

    c.mis.emitter_rev[0] = vertex_pdf(v, apertures, sv[t-2].wp, point_of(e));
    c.mis.emitter_rev[1] = vertex_pdf(e, apertures, v.wp, point_of(ev[s-2]));
    c.mis.sensor_rev[0] = vertex_pdf(e, apertures, ev[s-2].wp, point_of(v));
    c.mis.sensor_rev[1] = vertex_pdf(v, apertures, e.wp, point_of(sv[t-2]));
    c.mis.sensor_revs = 2;
    c.mis.emitter_revs = 2;
    c.mis.delta_sensor = is_delta_sensor(scene.sensor);
    c.mis.delta_emitter = is_delta_emitter(scene, ev[0].emitter);
    return c;
}

/** @brief A deferral of `trace_bdpt_path` that takes no connections. */
struct bdpt_no_deferral {
    static WT_GPU_CONSTANT constexpr bool may_defer = false;
    template <typename Observed>
    WT_GPU_FN bool defer(const WT_GPU_THREAD shadow_ray_t&, Observed, u32, u32, u32, vec2, bool) { return false; }
};

/**
 * @brief Traces path `path` (`plt_bdpt_t::integrate`, a sample of a sensor
 *        element): samples an emitter and one of its spectral samples in the
 *        table at `table` (or takes `spectral`, unless it is `choose_spectral`),
 *        as `start_path`; traces a sensor subpath from sensor element `path`
 *        modulo the film's elements, and an emitter subpath; connects them,
 *        and splats the connections: those to the sensor into `light`, the
 *        others into the sensor element's block in `film`. The subpaths' vertices
 *        go to `sensor_verts` and `emitter_verts`, of `capacity` vertices each,
 *        and the edges of their diffractions' apertures to `aperture_edges`, of
 *        `aperture_capacity` edges. Returns the path's spectral sample; adds
 *        status flags to `status`.
 *
 *        `deferred` may take over a connection's shadow ray and splat, if its
 *        `may_defer` (`bdpt_no_deferral` takes none): its `defer(shadow ray,
 *        observed, spectral sample, ex, ey, offset, into light)` returns
 *        whether it will splat it (into `light`, or else unweighted into
 *        `film`) unless the ray finds the segment blocked. The path splats
 *        its film block's weights in any case.
 */
template <bool P, typename Geometry, typename Film, typename Deferral>
WT_GPU_FN u32 trace_bdpt_path(const WT_GPU_THREAD plt_scene& scene, const WT_GPU_THREAD Geometry& geometry,
                              WT_GPU_THREAD Film& film, WT_GPU_THREAD Film& light,
                              WT_GPU_THREAD Deferral& deferred,
                              const WT_GPU_THREAD path_scratch& scratch,
                              WT_GPU_GLOBAL bdpt_vertex<P>* sensor_verts,
                              WT_GPU_GLOBAL bdpt_vertex<false>* emitter_verts, u32 capacity,
                              WT_GPU_GLOBAL fraunhofer::fsd_edge* aperture_edges, u32 aperture_capacity,
                              rng rnd, u64 path, u32 table, u32 spectral, WT_GPU_THREAD u32& status) {
    const plt_params opts = scene.params;

    // the emitter and the wavenumber (scene_t::sample_emitter_and_spectrum), and the sensor element
    const u32 ei = sample_emitter(scene, rnd.r());
    if (spectral==choose_spectral)
        spectral = table + ei*scene.samples_per_emitter + spectral_sample_for_path(path, scene.samples_per_emitter);
    const spectral_sample spec = scene.spectra[spectral];
    const u32 element = u32(path % (u64(scene.film.width)*scene.film.height));
    const u32 ex = element % scene.film.width;
    const u32 ey = element / scene.film.width;

    // the subpaths
    fsd_memory apertures = { aperture_edges, aperture_capacity, 0, spec.k };
    vec2 offset;
    bdpt_walk_t<P> sw = start_sensor_subpath(scene, spec, ex, ey, rnd, offset, sensor_verts);
    const u32 nt = bdpt_random_walk(scene, geometry, scratch, spectral, true, sw, sensor_verts, capacity,
                                    apertures, rnd, status);
    if (status & status_scratch_overflow)
        return spectral;
    bdpt_walk_t<false> ew = start_emitter_subpath(scene, ei, spectral, rnd, emitter_verts);
    const u32 ns = bdpt_random_walk(scene, geometry, scratch, spectral, false, ew, emitter_verts, capacity,
                                    apertures, rnd, status);
    if (status & status_scratch_overflow)
        return spectral;

    // the connections of t sensor and s emitter vertices (blocked ones end before their weights, unless they
    // may be deferred)
    constexpr bool late = Deferral::may_defer;
    observed_t<P> L = {};
    for (i32 t=0; t<=i32(nt); ++t)
    for (i32 s=0; s<=i32(ns); ++s) {
        const i32 depth = t+s-2;
        if ((t==1 && s==1) || depth<0) continue;
        if (opts.emitter_direct==0 && s==1) continue;
        if (opts.sensor_direct==0 && t==1) continue;
        if (depth>i32(opts.max_depth)) break;

        const bdpt_connection_t<P> c =
            s==0 ? connect_to_emitter_surface<P>(scene, spectral, t, sensor_verts) :
            t==0 ? connect_through_sensor<P>(scene, s, emitter_verts) :
            s==1 ? connect_to_emitter<late>(scene, geometry, spectral, t, sensor_verts, apertures, rnd, status) :
            t==1 ? connect_to_sensor<late, P>(scene, geometry, s, emitter_verts, apertures, rnd, status) :
                   connect_vertices<late>(scene, geometry, s, t, sensor_verts, emitter_verts, apertures, status);
        if (intensity_of(c.L)<=real(0))
            continue;

        // MIS, or the strategies' mean, with the spectral weight (the emissions include its MIS form)
        const real w = opts.mis!=0 ? bdpt_mis_weight(s, t, sensor_verts, emitter_verts, c.mis) :
                                     spec.no_mis_scale / real(s+t+1);
        const observed_t<P> Lw = c.L * w;
        if (late && c.shadow) {
            // (a deferral out of room leaves the ray to the path)
            const shadow_ray_t sr = shadow_ray(scene, c.from, c.to);
            if (t>1 ? deferred.defer(sr, Lw, spectral, ex, ey, offset, false) :
                      deferred.defer(sr, Lw, spectral, c.ex, c.ey, c.offset, true))
                continue;
            if (geometry.occluded(sr.r, sr.rng, status))
                continue;
        }
        if (t>1)
            L += Lw;
        else
            splat(scene.film, spec.response, light, c.ex, c.ey, c.offset, Lw);
    }

    splat_weighted(scene.film, spec.response, film, ex, ey, offset, L);
    return spectral;
}

} } }
