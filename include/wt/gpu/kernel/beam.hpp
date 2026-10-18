/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "intersect.hpp"
#include "polarimetric.hpp"

/*
 * Beams: their geometry (elliptic cones) and polarimetric radiometry, ported
 * from wt/beam/{beam,beam_generic}.hpp, wt/math/shapes/elliptic_cone.hpp,
 * src/math/elliptic_cone.cpp, wt/math/frame.hpp, wt/math/linalg.hpp and
 * wt/interaction/intersection.hpp.
 */

namespace wt::gpu::kernel {

/*
 * Frames
 */

/** @brief `frame_t::build_orthogonal_frame` */
WT_GPU_FN frame build_orthogonal_frame(vec3 n) {
    vec3 b;
    if (abs(n.x)>abs(n.y)) {
        const real x = real(1)/sqrt(sqr(n.x)+sqr(n.z));
        b = { x*n.z, 0, -x*n.x };
    } else {
        const real x = real(1)/sqrt(sqr(n.y)+sqr(n.z));
        b = { 0, x*n.z, -x*n.y };
    }
    return { cross(b,n), b, n };
}

/** @brief `frame_t::build_shading_frame` */
WT_GPU_FN frame build_shading_frame(vec3 n, vec3 dpdu) {
    if (dpdu.x==real(0) && dpdu.y==real(0) && dpdu.z==real(0))
        return build_orthogonal_frame(n);

    const vec3 t = normalize(dpdu - n*dot(n,dpdu));
    const vec3 b = normalize(cross(n,t));
    return { cross(b,n), b, n };
}

/** @brief `ray_t{ o,d }` */
WT_GPU_FN ray make_ray(vec3 o, vec3 d) {
    return { o, d, { real(1)/d.x, real(1)/d.y, real(1)/d.z } };
}


/*
 * 2x2 SVD (wt/math/linalg.hpp). Matrices are given by their elements
 * a=A[0][0], b=A[1][0], c=A[0][1], d=A[1][1] of the column-major glm mat2.
 */

struct svd2_t {
    real Ucos, Usin;
    real Vcos, Vsin;
    real sigma1, sigma2;
};

/** @brief `wt::SVD(mat2_t)` */
WT_GPU_FN svd2_t svd2(real a, real b, real c, real d) {
    const real A00 = a, A11 = d;

    // QR(A)
    real x,y,z,Qc,Qs;
    if (c==real(0)) {
        x = a; y = b; z = d;
        Qc = 1; Qs = 0;
    } else {
        const real m = max(abs(c), abs(d));
        const real recp_m = real(1)/m;
        c *= recp_m;
        d *= recp_m;

        const real r = sqrt(c*c+d*d);
        const real l = real(1)/r;
        x = eft::diff_prod(a,d, b,c)*l;
        y = eft::sum_prod(a,c, b,d)*l;
        z = m*r;
        Qs = -c*l;
        Qc =  d*l;
    }

    real c2 = Qc;
    real s2 = Qs;

    const real n = max(abs(x), abs(y));
    if (n==real(0))
        return { 1, 0, c2, s2, A00, A11 };

    const real numer = (z-x)*(z+x) + sqr(y);
    const real tt = numer!=real(0) ? numer/(n*x*y) : real(0);
    real t = 2*real(tt>=real(0) ? 1 : -1)/(abs(tt) + sqrt(sqr(tt)+4));
    const real c1 = real(1)/sqrt(1 + sqr(t));
    const real s1 = c1*t;

    const real usa = eft::diff_prod(c1,x, s1,y);
    const real usb = eft::sum_prod(s1,x, c1,y);
    const real usc = -s1*z;
    const real usd = c1*z;

    t = eft::sum_prod(c1,c2, s1,s2);
    s2 = eft::diff_prod(c2,s1, c1,s2);
    c2 = t;

    const real sigma1 = sqrt(sqr(usa) + sqr(usc));
    real sigma2 = sqrt(sqr(usb) + sqr(usd));
    real dmax = max(sigma1, sigma2);
    const real usmax1 = sigma2>sigma1 ? usd :  usa;
    const real usmax2 = sigma2>sigma1 ? usb : -usc;

    const real signsigma1 = x*z>real(0) ? real(1) : real(-1);
    dmax *= sigma2>sigma1 ? signsigma1 : real(1);
    sigma2 *= signsigma1;
    const real rd = real(1)/dmax;

    return {
        dmax!=real(0) ? usmax1*rd : real(1),
        dmax!=real(0) ? usmax2*rd : real(0),
        c2, s2,
        sigma1, sigma2,
    };
}


/*
 * Elliptic cones
 */

/** @brief `elliptic_cone_t`'s private constructor (ray, x, x0, tan_alpha, one_over_e, e). */
WT_GPU_FN cone make_cone(ray r, vec3 x, real x0, real tan_alpha, real one_over_e, real e) {
    cone c;
    c.r = r;
    c.x = x;
    c.y = cross(r.d, x);
    c.x0 = x0;
    c.tan_alpha = tan_alpha;
    c.e = e;
    c.one_over_e = one_over_e;
    c.z_apex = x0!=real(0) || tan_alpha!=real(0) ? -x0/tan_alpha : -inf();
    return c;
}

/** @brief `elliptic_cone_t(ray, tan_alpha, initial_x)`: an isotropic cone. */
WT_GPU_FN cone make_isotropic_cone(ray r, real tan_alpha, real x0) {
    return make_cone(r, build_orthogonal_frame(r.d).t, x0, tan_alpha, 1, 1);
}

/** @brief `elliptic_cone_t::set_x0` */
WT_GPU_FN void set_x0(WT_GPU_THREAD cone& c, real x0) {
    c.x0 = x0;
    c.z_apex = x0!=real(0) || c.tan_alpha!=real(0) ? -x0/c.tan_alpha : -inf();
}

/** @brief `intersect::intersect_cone_plane<in_local=false>`: plane `dot(n,p)=d`, world space. */
WT_GPU_FN cone_plane_hit_t intersect_cone_plane(const WT_GPU_THREAD cone& cn, vec3 n, real d, range rng) {
    d -= dot(cn.r.o, n);
    n = cn.frame().to_local(n);
    return intersect_cone_plane_local(cn, n, d, rng);
}

/**
 * @brief `elliptic_cone_t::cone_through_ellipse`: the cone around `r` with
 *        opening `tan_alpha` through the ellipse with axes `x,y` (world
 *        lengths and directions) centred at `r.o` in the plane of normal `n`.
 *        `self_intersection_distance` receives the distance past the origin
 *        over which the cone still intersects that plane.
 */
WT_GPU_FN cone cone_through_ellipse(vec3 x, vec3 y, vec3 n, ray r, real tan_alpha,
                                    WT_GPU_THREAD real& self_intersection_distance) {
    if (x==vec3{ 0,0,0 } && y==vec3{ 0,0,0 }) {
        // degenerate ray case
        self_intersection_distance = 0;
        return make_cone(r, build_orthogonal_frame(r.d).t, 0, tan_alpha, 1, 1);
    }

    // build orthogonal frame around cone direction
    const frame of = build_orthogonal_frame(r.d);
    const vec2 xhat = { dot(x, of.t), dot(x, of.b) };
    const vec2 yhat = { dot(y, of.t), dot(y, of.b) };

    // projected ellipse upon orthogonal frame: mat2_t{ xhat, yhat }
    const svd2_t svd = svd2(xhat.x, yhat.x, xhat.y, yhat.y);

    // extract major axes and their lengths
    vec2 X = { svd.Ucos, -svd.Usin };
    real lX = abs(svd.sigma1);
    real lY = abs(svd.sigma2);
    if (lX<lY) {
        swap(lX,lY);
        X = { svd.Usin, svd.Ucos };
    }

    // eccentricity and cone origin distance from centre (may be a degenerate cone)
    const real e = lY>real(0) ? sqrt(lX/lY) : real(1);
    const vec3 wx = of.t*X.x + of.b*X.y;

    const cone c = make_cone(r, wx, lX, tan_alpha, 1/e, e);

    // self intersection distance (from the origin) of the cone with the ellipse surface
    const cone_plane_hit_t cp = intersect_cone_plane(c, n, dot(n, r.o), range::positive());
    self_intersection_distance = cp.rng.empty() ? real(0) : cp.rng.max;

    return c;
}


/*
 * Surfaces
 */

/** @brief `intersection_footprint_t`: an ellipse in the tangent plane of a surface's geometric frame. */
struct footprint {
    vec2 x;         // direction of major axis
    real la, lb;    // lengths of major and minor axes

    WT_GPU_FN vec2 y() const { return { -x.y, x.x }; }
    WT_GPU_FN vec2 a() const { return x * la; }
    WT_GPU_FN vec2 b() const { return y() * lb; }
};

/** @brief The parts of `intersection_surface_t` beam transport needs. */
struct surface {
    vec3 wp;            // centre of the beam-surface intersection
    frame geo;          // geometric frame
    frame shading;      // shading frame
    footprint fp;       // in the geometric frame
    u32 tri;            // BVH triangle
};

/** @brief `intersection_surface_t::s_direction` */
WT_GPU_FN vec3 s_direction(const WT_GPU_THREAD surface& s, vec3 w) {
    const vec3 crs = cross(w, s.shading.n);
    const real l2 = length2(crs);
    const vec3 ret = l2<real(1e-14) ? s.shading.t : crs/sqrt(l2);

    // identical sp frame for w pointing into or out of the surface
    return dot(w, s.shading.n)<real(0) ? -ret : ret;
}

/** @brief `intersection_surface_t::sp_frame`: s- and p-polarization directions for `w`. */
WT_GPU_FN frame sp_frame(const WT_GPU_THREAD surface& s, vec3 w) {
    const vec3 sd = s_direction(s, w);
    const vec3 p = cross(sd, w);
    return { sd, dot(w, s.shading.n)<real(0) ? -p : p, w };
}

/** @brief `compute_intersection_triangle_fp_errors` of src/interaction/intersection.cpp */
WT_GPU_FN vec3 intersection_triangle_fp_errors(vec3 a, vec3 b, vec3 c, vec3 ro) {
    const real c0 = real(3e-6);
    const real c1 = real(5e-6);
    const real c2 = real(3e-6);

    // twice the maximum extent of the triangle
    const vec3 v0 = abs(a);
    const vec3 e1 = abs(b-a);
    const vec3 e2 = abs(c-a);
    const vec3 extents = e1 + e2 + abs(e1 - e2);
    const real extent = max_element(extents);

    const vec3 obj_err = (c0+c2)*v0 + vec3{ c1*extent, c1*extent, c1*extent };
    const vec3 wrld_err = (c1+c2)*abs(ro);

    return obj_err + wrld_err;
}

/** @brief `intersection_surface_t::offseted_ray_origin`: avoids self intersection with triangle `a,b,c` of normal `ng`. */
WT_GPU_FN vec3 offseted_ray_origin(vec3 a, vec3 b, vec3 c, vec3 ng, ray r) {
    const vec3 err = intersection_triangle_fp_errors(a, b, c, r.o);
    const real offset_dist = dot(err, abs(ng));
    const vec3 offset = offset_dist * ng;

    // offset along face normal in direction of ray travel
    return r.o + (dot(r.d, offset)>=real(0) ? offset : -offset);
}


/*
 * Beams
 */

/**
 * @brief A forward-transport beam (`beam_t<transport_e::forward, ...>`): an
 *        elliptic-cone envelope and a Stokes parameters vector. The Stokes
 *        vector is relative to the flux of the emitter sample that sourced
 *        the beam, which the path tracer applies when it splats.
 */
struct beam {
    cone envelope;
    stokes S;
    frame Sframe;               // frame of `S`
    real self_intersection_distance;

    WT_GPU_FN vec3 dir() const { return envelope.r.d; }
    WT_GPU_FN vec3 origin() const { return envelope.r.o; }

    /** @brief `beam_generic_t::footprint` (beam local frame) */
    WT_GPU_FN vec3 footprint3(real dist) const {
        const vec2 a = envelope.axes(dist);
        return { a.x, a.y, 2*a.x };
    }

    /** @brief `beam_generic_t::surface_footprint_static` */
    WT_GPU_FN footprint surface_footprint_static(frame geo, real dist) const {
        const vec3 ls = footprint3(dist);
        const vec3 x = geo.to_local(envelope.x);
        if (x.x!=real(0) || x.y!=real(0))
            return { normalize(vec2{ x.x,x.y }), ls.x, ls.y };
        const real avg = (ls.x+ls.y)/2;
        return { { 1,0 }, avg, avg };
    }

    /** @brief `beam_t::transform_surface_interaction`: interact with a surface (Mueller operator `M`) and scatter into `wo`. */
    WT_GPU_FN void transform_surface_interaction(const WT_GPU_THREAD surface& s, vec3 wo,
                                                 fresnel_mueller M, real weight) {
        // a new elliptical cone that passes through the interaction footprint
        const vec2 a = s.fp.a();
        const vec2 b = s.fp.b();
        const vec3 wa = s.geo.t*a.x + s.geo.b*a.y;
        const vec3 wb = s.geo.t*b.x + s.geo.b*b.y;
        real sid;
        envelope = cone_through_ellipse(wa, wb, s.geo.n, make_ray(s.wp, wo), envelope.tan_alpha, sid);

        // apply the BSDF, from the beam frame through the sp frames to the new beam frame
        const frame new_frame = envelope.frame();
        const frame SPin  = sp_frame(s, Sframe.n);
        const frame SPout = sp_frame(s, wo);
        S = reorient((M*weight) * reorient(S, Sframe, SPin), SPout, new_frame);
        Sframe = new_frame;
        self_intersection_distance = sid;
    }
};

}
