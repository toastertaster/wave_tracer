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
 * The telescope sensor on the GPU: a port of `remote_sensing::telescope_t::sample` (src/remote_sensing/telescope.cpp),
 * of the ray tracing of its optics (`optics::telescope_optics_t::trace_from_focal_plane`, and, for stray light,
 * `trace_nonsequential`; `surface_shape_t`, src/remote_sensing/telescope_optics.cpp), of its mirrors' scatter
 * (`optics::abg_scatter_t`) and of its mirrors' coatings (`optics::coating_t::reflection`,
 * src/remote_sensing/telescope_coating.cpp). The host computes what the sensor computes when loading (its focal
 * lengths, the cones of passing directions; see `telescope_desc`).
 *
 * The CPU traces the optics in double precision; the kernels in theirs (`real`). In single precision, directions
 * leaving the telescope are accurate to some 1e-7 radians: a small fraction of a pixel's field of view for focal
 * lengths of a few metres, but not for long ones (the host warns). Hence the tolerances below scale with the
 * precision; and a direction sampled from one of the two cones of the sampling mixture counts as in that cone
 * (rounding cannot exclude it, which would weigh the sample by the other cone's density alone).
 */

namespace wt { namespace gpu { namespace compute_kernel {

#if defined(REALISTIC_REMOTE_SENSING)

/** @brief A rigid row-major 3x4 matrix times a point, and times a direction. */
template <typename Matrix>
WT_GPU_FN vec3 affine_point(Matrix M, vec3 p) {
    return {
        fma(M[2], p.z, fma(M[1], p.y, M[0]*p.x)) + M[3],
        fma(M[6], p.z, fma(M[5], p.y, M[4]*p.x)) + M[7],
        fma(M[10], p.z, fma(M[9], p.y, M[8]*p.x)) + M[11],
    };
}
template <typename Matrix>
WT_GPU_FN vec3 affine_direction(Matrix M, vec3 d) {
    return {
        fma(M[2], d.z, fma(M[1], d.y, M[0]*d.x)),
        fma(M[6], d.z, fma(M[5], d.y, M[4]*d.x)),
        fma(M[10], d.z, fma(M[9], d.y, M[8]*d.x)),
    };
}
/** @brief A row-major 3x3 matrix times a vector. */
template <typename Matrix>
WT_GPU_FN vec3 telescope_rotate(Matrix M, vec3 v) {
    return {
        fma(M[2], v.z, fma(M[1], v.y, M[0]*v.x)),
        fma(M[5], v.z, fma(M[4], v.y, M[3]*v.x)),
        fma(M[8], v.z, fma(M[7], v.y, M[6]*v.x)),
    };
}

/** @brief Newton iterations refining an intersection with a surface (`surface_shape_t::intersect`). */
WT_GPU_CONSTANT constexpr int telescope_newton_iterations = 16;

/**
 * @brief How near a ray's origin, on a surface, intersections are the surface itself (as the CPU's
 *        `self_intersection_epsilon`, 1e-9 m, but no smaller than the precision allows about `o`).
 */
WT_GPU_FN real telescope_epsilon(vec3 o) {
    return max(real(1e-9), 64*epsilon*(1 + abs(o.x) + abs(o.y) + abs(o.z)));
}

/** @brief `surface_shape_t::sag`: false beyond the conic. */
WT_GPU_FN bool telescope_sag(const WT_GPU_GLOBAL telescope_element& e, real r2, WT_GPU_THREAD real& z) {
    const real arg = 1 - (1+e.conic)*e.curvature*e.curvature*r2;
    if (arg<=real(0))
        return false;
    z = e.curvature*r2 / (1 + sqrt(arg));
    real r2n = r2*r2;
    for (u32 i=0; i<e.aspheric_count; ++i) {
        z += e.aspheric[i]*r2n;
        r2n *= r2;
    }
    return true;
}

/** @brief `surface_shape_t::normal`: the unit normal at (x,y), facing local +z; false beyond the conic. */
WT_GPU_FN bool telescope_normal(const WT_GPU_GLOBAL telescope_element& e, real x, real y, WT_GPU_THREAD vec3& n) {
    const real r2 = x*x + y*y;
    const real arg = 1 - (1+e.conic)*e.curvature*e.curvature*r2;
    if (arg<=real(0))
        return false;
    // dz/dx = x g(r²), dz/dy = y g(r²)
    real g = e.curvature / sqrt(arg);
    real r2n = r2;
    real power = 4;
    for (u32 i=0; i<e.aspheric_count; ++i) {
        g += power*e.aspheric[i]*r2n;
        r2n *= r2;
        power += 2;
    }
    n = normalize(vec3{ -x*g, -y*g, 1 });
    return true;
}

/** @brief `telescope_intersect_conic`'s outcomes. */
WT_GPU_CONSTANT constexpr int telescope_conic_hit = 1;
WT_GPU_CONSTANT constexpr int telescope_conic_missed = 0;
/** @brief The ray does not meet the conic's quadric at all. */
WT_GPU_CONSTANT constexpr int telescope_conic_no_roots = -1;

/**
 * @brief The distance `t` along the local ray (o,d) to the conic of vertex curvature `c` and conic constant `k1`-1
 *        (`surface_shape_t`, without aspheric terms), beyond `tmin`, on the sheet through the vertex.
 */
WT_GPU_FN int telescope_intersect_conic(real c, real k1, vec3 o, vec3 d, real tmin, WT_GPU_THREAD real& t) {
    // c (x² + y² + (1+κ) z²) - 2z = 0, as A t² + 2B t + C = 0: of its roots beyond tmin, the nearest on the branch
    // through the vertex, where (1+κ) c z ≤ 1
    const real A = c*(d.x*d.x + d.y*d.y + k1*d.z*d.z);
    const real B = c*(o.x*d.x + o.y*d.y + k1*o.z*d.z) - d.z;
    const real C = c*(o.x*o.x + o.y*o.y + k1*o.z*o.z) - 2*o.z;
    const real D = B*B - A*C;
    if (!(D>=real(0)) || (B==real(0) && A==real(0)))
        return telescope_conic_no_roots;

    real r0, r1;
    if (A==real(0)) {
        r0 = r1 = -C/(2*B);
    } else {
        const real q = -(B + (B<real(0) ? -sqrt(D) : sqrt(D)));
        r0 = q/A;
        r1 = q!=real(0) ? C/q : r0;
        if (r1<r0)
            swap(r0, r1);
    }
    const real branch = 1 + 16*epsilon;
    if (r0>tmin && k1*c*(o.z + r0*d.z)<=branch)
        t = r0;
    else if (r1>tmin && k1*c*(o.z + r1*d.z)<=branch)
        t = r1;
    else
        return telescope_conic_missed;
    return isfinite(t) ? telescope_conic_hit : telescope_conic_missed;
}

/**
 * @brief `surface_shape_t::intersect`: the distance `t` along the local ray (o,d) to the surface, beyond `tmin`, on
 *        the sheet through the vertex. False if the ray misses it.
 */
WT_GPU_FN bool telescope_intersect_surface(const WT_GPU_GLOBAL telescope_element& e, vec3 o, vec3 d, real tmin,
                                           WT_GPU_THREAD real& t) {
    // the base conic's
    const int conic = telescope_intersect_conic(e.curvature, 1+e.conic, o, d, tmin, t);
    if (conic==telescope_conic_missed || (conic==telescope_conic_no_roots && e.aspheric_count==0)) {
        return false;
    } else if (conic==telescope_conic_no_roots) {
        // the aspheric terms may still bring the surface to the ray: start at the vertex plane
        if (d.z==real(0))
            return false;
        t = -o.z/d.z;
    }
    if (!isfinite(t))
        return false;

    // refine on the full surface, to the precision about the ray's points
    const real tolerance = 64*epsilon*(abs(o.x) + abs(o.y) + abs(o.z) + abs(t));
    for (int i=0; i<telescope_newton_iterations; ++i) {
        const vec3 p = o + t*d;
        real z;
        vec3 n;
        if (!telescope_sag(e, p.x*p.x + p.y*p.y, z) || !telescope_normal(e, p.x, p.y, n))
            return false;
        // F(t) = z(t) - sag(x(t),y(t)), F' = d · n / n.z
        const real dF = dot(d, n) / n.z;
        if (dF==real(0))
            return false;
        const real dt = (p.z - z) / dF;
        t -= dt;
        if (abs(dt)<=tolerance)
            return t>tmin;
    }
    return false;
}

/** @brief `aperture_t::contains` */
WT_GPU_FN bool telescope_aperture_contains(const WT_GPU_GLOBAL telescope_element& e, vec3 local) {
    const real dx = local.x - e.aperture_centre.x, dy = local.y - e.aperture_centre.y;
    const real r2 = dx*dx + dy*dy;
    return r2<=e.aperture_radius2 && r2>=e.hole_radius2;
}

/** @brief A tube's cylinder (`element_t::intersect`): the nearer root beyond `tmin`, within its length. */
WT_GPU_FN bool telescope_intersect_tube(const WT_GPU_GLOBAL telescope_element& e, vec3 o, vec3 d, real tmin,
                                        WT_GPU_THREAD real& t) {
    const real A = d.x*d.x + d.y*d.y;
    if (A==real(0))
        return false;
    const real B = o.x*d.x + o.y*d.y;
    const real C = o.x*o.x + o.y*o.y - e.aperture_radius2;
    const real D = B*B - A*C;
    if (D<real(0))
        return false;
    const real q = -(B + (B<real(0) ? -sqrt(D) : sqrt(D)));
    real t0 = q/A, t1 = q!=real(0) ? C/q : t0;
    if (t0>t1)
        swap(t0, t1);
    for (int i=0; i<2; ++i) {
        t = i==0 ? t0 : t1;
        const real z = o.z + t*d.z;
        if (t>tmin && z>=real(0) && z<=e.length)
            return true;
    }
    return false;
}

/** @brief `element_t::intersect`: where the (telescope frame) ray (o,d) meets the element, beyond `tmin`. */
WT_GPU_FN bool telescope_intersect(const WT_GPU_GLOBAL telescope_element& e, vec3 o, vec3 d, real tmin,
                                   WT_GPU_THREAD real& t, WT_GPU_THREAD vec3& local) {
    const vec3 lo = affine_point(e.to_local, o);
    const vec3 ld = affine_direction(e.to_local, d);
    if (e.kind==telescope_tube ? !telescope_intersect_tube(e, lo, ld, tmin, t) :
                                 !telescope_intersect_surface(e, lo, ld, tmin, t))
        return false;
    local = lo + t*ld;
    return true;
}

/** @brief `abg_scatter_t::tis`: a rough mirror's TIS at |β₀| = `beta0` (interpolated in its table). */
WT_GPU_FN real telescope_tis(const WT_GPU_GLOBAL telescope_scatter& sc, real beta0) {
    const real x = clamp(beta0, real(0), real(1)) * real(telescope_tis_samples-1);
    const u32 i = min(u32(x), telescope_tis_samples-2);
    const real f = x - real(i);
    return (1-f)*sc.tis[i] + f*sc.tis[i+1];
}

/** @brief `abg_scatter_t::bsdf` */
WT_GPU_FN real telescope_bsdf(const WT_GPU_GLOBAL telescope_scatter& sc, real offset) {
    return sc.A / (sc.B + pow(offset, sc.g));
}

/**
 * @brief `abg_scatter_t::sample`: an offset β - β₀ from three uniform numbers, and one over its density (in d²β; 0
 *        if none).
 */
WT_GPU_FN vec2 telescope_sample_scatter(const WT_GPU_GLOBAL telescope_scatter& sc, real u1, real u2, real u3,
                                        WT_GPU_THREAD real& recp_pdf) {
    // the bin whose CDF interval holds u1
    u32 lo = 0, hi = telescope_scatter_bins-1;
    while (lo<hi) {
        const u32 mid = (lo+hi)/2;
        if (u1<sc.cdf[mid+1]) hi = mid;
        else lo = mid+1;
    }
    const real a = sc.edges[lo], b = sc.edges[lo+1];
    const real r = sqrt(a*a + u2*(b*b - a*a));
    const real phi = two_pi*u3;
    const real mass = sc.cdf[lo+1] - sc.cdf[lo];
    recp_pdf = mass>real(0) ? pi*(b*b - a*a) / mass : real(0);
    return vec2{ cos(phi), sin(phi) } * r;
}

/** @brief No element. */
WT_GPU_CONSTANT constexpr u32 telescope_none = 0xffffffffu;
/** @brief The focal-plane array, as a ray's nearest obstacle (`telescope_optics_t::stopped_by_detector`). */
WT_GPU_CONSTANT constexpr u32 telescope_detector = 0xfffffffeu;

/** @brief A reflection's kind (`telescope_optics_t::event_kind_e`). */
WT_GPU_CONSTANT constexpr u32 telescope_specular = 0;
WT_GPU_CONSTANT constexpr u32 telescope_scattered = 1;
WT_GPU_CONSTANT constexpr u32 telescope_diffuse = 2;

/** @brief `element_t::physical_hit`: a mirror or obscuration within its aperture, a tube, or a stop's plate. */
WT_GPU_FN bool telescope_physical_hit(const WT_GPU_GLOBAL telescope_element& e, vec3 o, vec3 d, real tmin,
                                      WT_GPU_THREAD real& t, WT_GPU_THREAD vec3& local) {
    if (!telescope_intersect(e, o, d, tmin, t, local))
        return false;
    if (e.kind==telescope_tube)
        return true;
    const bool within = telescope_aperture_contains(e, local);
    return e.kind==telescope_stop ? !within : within;
}

/** @brief `element_t::normal_at`: the unit normal (telescope frame) at a hit's local point. */
WT_GPU_FN vec3 telescope_normal_at(const WT_GPU_GLOBAL telescope_element& e, vec3 local) {
    vec3 n = { 0,0,1 };
    if (e.kind==telescope_tube) {
        n = vec3{ local.x, local.y, 0 };
    } else if (e.kind==telescope_mirror) {
        vec3 nl;
        if (telescope_normal(e, local.x, local.y, nl))
            n = nl;
    }
    return normalize(affine_direction(e.to_telescope, n));
}

/**
 * @brief `telescope_optics_t::nearest`: the nearest element, or the detector (telescope_detector), that the ray
 *        (o,d) physically meets beyond `tmin`; telescope_none if none.
 */
WT_GPU_FN u32 telescope_nearest(const WT_GPU_GLOBAL telescope_desc& T, vec3 o, vec3 d, real tmin,
                                WT_GPU_THREAD real& t, WT_GPU_THREAD vec3& local) {
    u32 found = telescope_none;
    t = inf();
    for (u32 i=0; i<T.element_count; ++i) {
        real ti;
        vec3 li;
        if (telescope_physical_hit(T.elements[i], o, d, tmin, ti, li) && ti<t) {
            t = ti;
            local = li;
            found = i;
        }
    }
    // the detector's surface: its plane, or sphere
    const vec3 lo = affine_point(T.fp_to_local, o);
    const vec3 ld = affine_direction(T.fp_to_local, d);
    real td;
    if (telescope_intersect_conic(T.fp_curvature, 1, lo, ld, tmin, td)==telescope_conic_hit) {
        const vec3 q = lo + td*ld;
        if (td<t && abs(q.x)<=T.fp_half_extent.x && abs(q.y)<=T.fp_half_extent.y) {
            t = td;
            found = telescope_detector;
        }
    }
    return found;
}

/** @brief An orthonormal tangent frame (t, b) about unit `n` (as telescope_optics.cpp's `tangent_frame`). */
WT_GPU_FN void telescope_tangent_frame(vec3 n, WT_GPU_THREAD vec3& t, WT_GPU_THREAD vec3& b) {
    const vec3 helper = abs(n.x)<real(.9) ? vec3{ 1,0,0 } : vec3{ 0,1,0 };
    t = normalize(cross(helper, n));
    b = cross(n, t);
}

/** @brief A direction of the hemisphere about unit `axis`, cosine-distributed, from two uniform numbers. */
WT_GPU_FN vec3 telescope_cosine_hemisphere(vec3 axis, vec2 u) {
    const real r = sqrt(u.x);
    const real phi = two_pi*u.y;
    vec3 t, b;
    telescope_tangent_frame(axis, t, b);
    return normalize(r*cos(phi)*t + r*sin(phi)*b + sqrt(max(real(0), 1-u.x))*axis);
}

/**
 * @brief `telescope_optics_t::obstacle`: whether a mirror, obscuration or tube, other than elements `from` and
 *        `to`, is in the way of the ray (o,d) within distance `tmax`.
 */
WT_GPU_FN bool telescope_obstructed(const WT_GPU_GLOBAL telescope_desc& T, vec3 o, vec3 d, real tmax,
                                    u32 from, u32 to) {
    const real eps = telescope_epsilon(o);
    for (u32 i=0; i<T.element_count; ++i) {
        const WT_GPU_GLOBAL telescope_element& e = T.elements[i];
        if (i==from || i==to || e.kind==telescope_stop)
            continue;
        real t;
        vec3 local;
        if (telescope_intersect(e, o, d, eps, t, local) && t<tmax-eps &&
            (e.kind==telescope_tube || telescope_aperture_contains(e, local)))
            return true;
    }
    return false;
}

/**
 * @brief `telescope_optics_t::trace_from_focal_plane`: traces the ray (o,d), leaving the focal plane, backwards
 *        through the optics. Calls `reflected(element, n, d_in, d_out)` at each mirror (`n` its normal, facing the
 *        ray). Returns whether the ray leaves the telescope, towards the scene, from (o,d); and multiplies `weight`
 *        by rough mirrors' specular fractions (1 - TIS).
 */
template <typename Reflected>
WT_GPU_FN bool telescope_trace(const WT_GPU_GLOBAL telescope_desc& T, WT_GPU_THREAD vec3& o, WT_GPU_THREAD vec3& d,
                               WT_GPU_THREAD real& weight, Reflected reflected) {
    u32 from = telescope_none;
    for (u32 s=T.sequence_count; s-->0;) {
        const u32 idx = T.sequence[s];
        const WT_GPU_GLOBAL telescope_element& e = T.elements[idx];

        real t;
        vec3 local;
        if (!telescope_intersect(e, o, d, from==telescope_none ? real(0) : telescope_epsilon(o), t, local) ||
            !telescope_aperture_contains(e, local))
            return false;
        if (telescope_obstructed(T, o, d, t, from, idx))
            return false;

        const vec3 p = o + t*d;
        if (e.kind==telescope_mirror) {
            vec3 nl;
            if (!telescope_normal(e, local.x, local.y, nl))
                return false;
            vec3 n = normalize(affine_direction(e.to_telescope, nl));
            if (dot(d,n)>real(0))
                n = -n;
            const vec3 dout = normalize(d - 2*dot(d,n)*n);
            reflected(idx, n, d, dout, telescope_specular);
            if (e.scatter!=telescope_smooth) {
                const real c = dot(dout, n);
                weight *= 1 - telescope_tis(T.scatters[e.scatter], sqrt(max(real(0), 1 - c*c)));
            }
            d = dout;
        }
        o = p;
        from = idx;
    }

    // out towards the scene
    return !telescope_obstructed(T, o, d, inf(), from, telescope_none);
}

/**
 * @brief `telescope_optics_t::trace_nonsequential`: traces the ray (o,d), leaving the focal plane, backwards through
 *        the optics non-sequentially (stray light): calls `reflected(element, n, d_in, d_out, kind)` at each
 *        reflection, multiplies `weight` by the probabilities' weights. Returns whether the ray leaves the telescope,
 *        towards the scene, from (o,d).
 */
template <typename Reflected>
WT_GPU_FN bool telescope_trace_nonsequential(const WT_GPU_GLOBAL telescope_desc& T,
                                             WT_GPU_THREAD vec3& o, WT_GPU_THREAD vec3& d,
                                             WT_GPU_THREAD real& weight, WT_GPU_THREAD rng& rnd,
                                             Reflected reflected) {
    for (u32 events=0; ; ++events) {
        real t;
        vec3 local;
        const u32 idx = telescope_nearest(T, o, d, telescope_epsilon(o), t, local);
        if (idx==telescope_none)
            return true;
        if (idx==telescope_detector || events==telescope_max_events)
            return false;

        const WT_GPU_GLOBAL telescope_element& e = T.elements[idx];
        const vec3 p = o + t*d;
        vec3 n = telescope_normal_at(e, local);
        const bool against = dot(d, n)<real(0);    // the ray meets the side n faces
        if (!against)
            n = -n;

        vec3 dout;
        u32 kind = telescope_specular;
        if (e.kind==telescope_mirror) {
            // the back of a mirror absorbs
            if (e.reflective_side!=0 && (e.reflective_side>0)!=against)
                return false;
            const vec3 spec = normalize(d - 2*dot(d,n)*n);
            dout = spec;
            if (e.scatter!=telescope_smooth) {
                const WT_GPU_GLOBAL telescope_scatter& sc = T.scatters[e.scatter];
                const real c = dot(spec, n);
                const real tis = telescope_tis(sc, sqrt(max(real(0), 1 - c*c)));
                const real p_scatter = clamp(max(tis, T.scatter_probability), real(0), real(1));
                if (!(rnd.r()<p_scatter)) {
                    weight *= (1-tis) / (1-p_scatter);
                } else {
                    // about the specular direction, in its projection on the tangent plane
                    vec3 tt, bb;
                    telescope_tangent_frame(n, tt, bb);
                    const vec2 beta0 = { dot(spec,tt), dot(spec,bb) };
                    const real u1 = rnd.r();
                    const vec2 u = rnd.r2();
                    real recp_pdf;
                    const vec2 offset = telescope_sample_scatter(sc, u1, u.x, u.y, recp_pdf);
                    const vec2 beta = beta0 + offset;
                    const real b2 = dot(beta, beta);
                    if (!(b2<real(1)) || recp_pdf==real(0))
                        return false;       // evanescent: nothing scattered there
                    dout = normalize(beta.x*tt + beta.y*bb + sqrt(1-b2)*n);
                    kind = telescope_scattered;
                    weight *= telescope_bsdf(sc, length(offset)) * recp_pdf / p_scatter;
                }
            }
        } else {
            // a Lambertian reflection, of its reflectance (sampled)
            if (!(e.reflectance>real(0)) || !(rnd.r()<e.reflectance))
                return false;
            dout = telescope_cosine_hemisphere(n, rnd.r2());
            kind = telescope_diffuse;
        }

        reflected(idx, n, d, dout, kind);
        o = p;
        d = dout;
    }
}


/*
 * Coatings: the characteristic matrices of thin films (`optics::coating_t::reflection`, in Macleod's convention:
 * refractive indices n - ik).
 */

struct telescope_matrix2 {
    complex a, b, c, d;

    WT_GPU_FN telescope_matrix2 operator*(const WT_GPU_THREAD telescope_matrix2& o) const {
        return { a*o.a + b*o.c, a*o.b + b*o.d,
                 c*o.a + d*o.c, c*o.b + d*o.d };
    }
};

/**
 * @brief cos θ in a medium of (Macleod) refractive index N, for incidence from vacuum at sin θ₀: the root whose
 *        N cos θ has a non-positive imaginary part.
 */
WT_GPU_FN complex telescope_cos_in(complex N, real sin0) {
    const complex s = complex{ sin0,0 } / N;
    complex c = sqrt(complex{ 1,0 } - s*s);
    if ((N*c).im>real(0))
        c = -c;
    return c;
}

/** @brief The reflection coefficient of a stack of characteristic matrix M on a substrate of admittance `eta_sub`. */
WT_GPU_FN complex telescope_reflect(const WT_GPU_THREAD telescope_matrix2& M, complex eta0, complex eta_sub) {
    const complex B = M.a + M.b*eta_sub;
    const complex C = M.c + M.d*eta_sub;
    return (eta0*B - C) / (eta0*B + C);
}

/**
 * @brief `coating_t::reflection`: a mirror's reflection coefficients, in wave_tracer's convention, at wavenumber k
 *        (of the refractive indices `iors`), for light incident at cos θ = `cos_theta`.
 */
WT_GPU_FN void telescope_coating(const WT_GPU_GLOBAL telescope_desc& T, const WT_GPU_GLOBAL telescope_element& e,
                                 real k, const WT_GPU_GLOBAL complex* iors, real cos_theta,
                                 WT_GPU_THREAD complex& rs, WT_GPU_THREAD complex& rp) {
    if (e.substrate==telescope_ideal) {
        rs = complex{ -1,0 };
        rp = complex{ 1,0 };
        return;
    }

    cos_theta = clamp(cos_theta, real(0), real(1));
    const real sin0 = sqrt(1 - cos_theta*cos_theta);

    if (e.layer_count==0) {
        // a bare substrate: wave_tracer's own Fresnel reflection (its refractive-index ratio is outside over inside)
        fresnel_reflection(complex{ 1,0 } / iors[e.substrate], vec3{ sin0, 0, cos_theta }, local_normal, rs, rp);
        return;
    }

    // the layers' characteristic matrices, for s and p, from the outermost
    telescope_matrix2 Ms = { { 1,0 }, { 0,0 }, { 0,0 }, { 1,0 } };
    telescope_matrix2 Mp = Ms;
    for (u32 i=0; i<e.layer_count; ++i) {
        const telescope_layer l = T.layers[e.first_layer+i];
        const complex N = conj(iors[l.ior]);
        const complex c = telescope_cos_in(N, sin0);
        const complex delta = (k*l.thickness) * (N*c);
        // cos δ and i sin δ, times e^-|Im δ| (the reflection coefficient is homogeneous in each layer's matrix): of
        // e^iδ and e^-iδ, the one that grows in an absorbing layer is kept finite
        const real decay = exp(-2*abs(delta.im));
        const complex e1 = expi(delta.re) * (delta.im>real(0) ? decay : real(1));
        const complex e2 = conj(expi(delta.re)) * (delta.im>real(0) ? real(1) : decay);
        const complex cosd = (e1+e2) * real(.5), isind = (e1-e2) * real(.5);
        const complex eta_s = N*c, eta_p = N/c;
        Ms = Ms * telescope_matrix2{ cosd, isind/eta_s, isind*eta_s, cosd };
        Mp = Mp * telescope_matrix2{ cosd, isind/eta_p, isind*eta_p, cosd };
    }
    const complex Ns = conj(iors[e.substrate]);
    const complex cs = telescope_cos_in(Ns, sin0);
    const complex s = telescope_reflect(Ms, complex{ cos_theta,0 }, Ns*cs);
    const complex p = telescope_reflect(Mp, complex{ 1/cos_theta,0 }, Ns/cs);

    // back to wave_tracer's convention: conjugated (its n + ik), and its rp's sign (rp = -rs at normal incidence)
    rs = conj(s);
    rp = -conj(p);
}


/*
 * Sampling
 */

/** @brief A cone of directions: its axis, the cosine of its half-angle, and one minus it (`optics::cone_t`). */
struct telescope_sampling_cone {
    vec3 axis;
    real cos_half_angle;
    real one_minus_cos;

    WT_GPU_FN static telescope_sampling_cone of_half_angle(vec3 axis, real half_angle) {
        const real s = sin(half_angle/2);
        const real omc = 2*s*s;
        return { axis, 1-omc, omc };
    }

    WT_GPU_FN real solid_angle() const { return two_pi*one_minus_cos; }
    WT_GPU_FN bool contains(vec3 d) const { return dot(d, axis)>=cos_half_angle; }
    /** @brief `cone_t::sample` */
    WT_GPU_FN vec3 sample(vec2 u) const {
        const real v = u.x*one_minus_cos;
        const real cos_theta = 1 - v;
        const real sin_theta = sqrt(max(real(0), v*(2-v)));
        const real phi = two_pi*u.y;
        const vec3 helper = abs(axis.x)<real(.9) ? vec3{ 1,0,0 } : vec3{ 0,1,0 };
        const vec3 t = normalize(cross(helper, axis));
        const vec3 b = cross(axis, t);
        return normalize(cos_theta*axis + sin_theta*(cos(phi)*t + sin(phi)*b));
    }
};

/** @brief `telescope_t::passing_cone`: of film position `film_pos` of a film of `dims` elements. */
WT_GPU_FN telescope_sampling_cone telescope_passing_cone(const WT_GPU_GLOBAL telescope_desc& T, vec2 film_pos,
                                                         vec2 dims) {
    const u32 n = telescope_aim_grid_size;
    const real gx = clamp(film_pos.x / dims.x * real(n-1), real(0), real(n-1));
    const real gy = clamp(film_pos.y / dims.y * real(n-1), real(0), real(n-1));
    const u32 ix = min(u32(gx), n-2), iy = min(u32(gy), n-2);
    const real fx = gx - real(ix), fy = gy - real(iy);
    const WT_GPU_GLOBAL telescope_cone& c00 = T.aim_grid[iy*n + ix];
    const WT_GPU_GLOBAL telescope_cone& c10 = T.aim_grid[iy*n + ix+1];
    const WT_GPU_GLOBAL telescope_cone& c01 = T.aim_grid[(iy+1)*n + ix];
    const WT_GPU_GLOBAL telescope_cone& c11 = T.aim_grid[(iy+1)*n + ix+1];
    const vec3 axis = normalize((1-fx)*(1-fy)*c00.axis + fx*(1-fy)*c10.axis +
                                (1-fx)*fy*c01.axis + fx*fy*c11.axis);
    const real widest = max(max(c00.half_angle, c10.half_angle), max(c01.half_angle, c11.half_angle));
    return telescope_sampling_cone::of_half_angle(axis, min(pi, widest*(1+T.aim_margin)));
}

/** @brief `telescope_optics_t::aim`: the cone, from focal-plane point `p`, of the rays that can pass. */
WT_GPU_FN telescope_sampling_cone telescope_aim(const WT_GPU_GLOBAL telescope_desc& T, vec3 p) {
    const vec3 v = T.aim_centre - p;
    const real dist = length(v);
    if (dist<=T.aim_radius)
        return { { 0,0,-1 }, -1, 2 };       // all directions
    const real s = T.aim_radius/dist;
    const real c = sqrt(1 - s*s);
    return { v/dist, c, s*s/(1+c) };
}

/** @brief The s-p frame of direction w at a surface of unit normal n (as telescope.cpp's `sp_frame`). */
WT_GPU_FN frame telescope_sp_frame(vec3 w, vec3 n) {
    const vec3 crs = cross(w, n);
    const real l2 = length2(crs);
    vec3 s = l2<real(1e-14) ? build_orthogonal_frame(n).t : crs/sqrt(l2);
    const bool inwards = dot(w,n)<real(0);
    if (inwards)
        s = -s;
    const vec3 p = cross(s, w);
    return { s, inwards ? -p : p, w };
}

/** @brief A sampled telescope beam's sensor element offset and densities (as `sensor_sample_t`). */
struct telescope_sample_t {
    vec2 offset;
    real ppd;
    real dpd;
};

/**
 * @brief `telescope_t::sample`: a beam sourced from sensor element (ex,ey) of a film of `dims` elements, at an
 *        offset within it, at spectral sample `spec`, traced backwards through the optics: its radiometric data is
 *        the sampled importance over the sampling densities, relative to the film centre's throughput, through the
 *        mirrors' coatings; zero if the optics vignette it.
 */
template <bool P>
WT_GPU_FN telescope_sample_t sample_telescope(const WT_GPU_GLOBAL telescope_desc& T,
                                              const WT_GPU_GLOBAL spectral_sample& spec,
                                              u32 ex, u32 ey, vec2 dims,
                                              WT_GPU_THREAD rng& rnd, WT_GPU_THREAD beam_t<P>& b) {
    const vec2 offset = rnd.r2();
    const vec2 film_pos = vec2{ real(ex)+offset.x, real(ey)+offset.y };
    const vec2 local = (film_pos - dims*real(.5)) * T.pixel_pitch;
    // on the focal surface: its sag, and its normal (`telescope_t::focal_plane_t::point` and `normal`)
    const real c = T.fp_curvature, r2 = dot(local, local);
    const real cos_slope2 = max(real(0), 1 - c*c*r2);
    const real sag = c*r2 / (1 + sqrt(cos_slope2));
    const real g = c / sqrt(max(cos_slope2, real(1e-30)));
    const vec3 p = affine_point(T.fp_to_telescope, vec3{ local.x, local.y, sag });
    const vec3 fp_normal = normalize(affine_direction(T.fp_to_telescope, vec3{ -local.x*g, -local.y*g, 1 }));

    // a direction towards the optics, from the mixture of the cone of passing directions and the cone bounding all
    // that can pass
    // (with stray light, some from the hemisphere in front of the focal plane, cosine-distributed)
    const telescope_sampling_cone tight = telescope_passing_cone(T, film_pos, dims);
    const telescope_sampling_cone bound = telescope_aim(T, p);
    // (about the focal surface's normal there, on the side of `front`)
    const vec3 front = dot(fp_normal, T.front)>=real(0) ? fp_normal : -fp_normal;
    const real h = T.stray_light!=0 ? T.hemisphere_fraction : real(0);
    const vec2 u = rnd.r2();
    const real choice = rnd.r();
    const u32 component = choice<h ? 0 : choice<h + (1-h)*T.aim_tight_fraction ? 1 : 2;
    const vec3 d = component==0 ? telescope_cosine_hemisphere(front, u) :
                   component==1 ? tight.sample(u) : bound.sample(u);
    const real pdf =
        h * max(real(0), dot(d, front)) * inv_pi +
        (1-h) * ((component==1 || tight.contains(d) ? T.aim_tight_fraction / tight.solid_angle() : real(0)) +
                 (component==2 || bound.contains(d) ? (1-T.aim_tight_fraction) / bound.solid_angle() : real(0)));

    // the polarization: the mirrors' reflections, composed in s-p frames, from the detector's frame (Q along the
    // film's x) to the frame of the light arriving from the scene (the beam's operator's incident frame)
    const vec3 fx = { T.fp_to_telescope[0], T.fp_to_telescope[4], T.fp_to_telescope[8] };
    const vec3 fy = { T.fp_to_telescope[1], T.fp_to_telescope[5], T.fp_to_telescope[9] };
    const vec3 dt = normalize(fx - dot(fx,d)*d);
    frame F = { dt, normalize(fy - dot(fy,d)*d - dot(fy,dt)*dt), d };
    stokes S = { 1,0,0,0 };
    b.rows.identity(1);
    const WT_GPU_GLOBAL complex* iors = spec.telescope_iors;
    const real k = spec.k;

    // (scatter depolarizes: a rough mirror's, of its coating's reflectance; a baffle's, whose reflectance is sampled)
    const auto reflected = [&](u32 element, vec3 n, vec3 din, vec3 dout, u32 kind) {
        mueller_block M = mueller_block::depolarizer(1);
        if (kind!=telescope_diffuse) {
            complex rs, rp;
            telescope_coating(T, T.elements[element], k, iors, dot(dout, n), rs, rp);
            M = kind==telescope_specular ? mueller_fresnel(rs, rp) :
                                           mueller_block::depolarizer((norm(rs) + norm(rp)) / 2);
        }
        const frame out = telescope_sp_frame(din, n);
        S = row_compose(S, M, F, out);
        b.rows.compose(M, F, out);
        F = telescope_sp_frame(dout, n);
    };
    vec3 o = p, dir = d;
    real trace_weight = 1;
    const bool passed = T.stray_light!=0 ?
        telescope_trace_nonsequential(T, o, dir, trace_weight, rnd, reflected) :
        telescope_trace(T, o, dir, trace_weight, reflected);

    // projected solid angle over its sampling density, relative to the film centre's
    const real weight = passed && pdf>real(0) ?
        abs(dot(d, fp_normal)) / pdf * T.recp_projected_solid_angle * trace_weight :
        real(0);

    // towards the scene, in world coordinates; starting `near_clip` along, about the film centre's line of sight
    if (!passed) {
        o = p;
        dir = T.line_of_sight;
    }
    const vec3 wd = normalize(telescope_rotate(T.rotation, dir));
    const vec3 wo = T.near_origin + telescope_rotate(T.rotation, o) + T.near_clip*(wd - T.world_line_of_sight);

    const real tan_alpha = spec.sensor_tan_alpha;
    b.envelope = make_isotropic_cone(make_ray(wo, wd), tan_alpha, spec.sensor_x0 + T.near_clip*tan_alpha);
    b.S = S*weight;
    b.rows.scale(weight);
    b.Sframe = {
        normalize(telescope_rotate(T.rotation, F.t)),
        normalize(telescope_rotate(T.rotation, F.b)),
        wd,
    };
    b.self_intersection_distance = 0;

    return { offset - vec2{ real(.5), real(.5) }, 1/(T.pixel_pitch*T.pixel_pitch), pdf };
}

#endif

} } }
