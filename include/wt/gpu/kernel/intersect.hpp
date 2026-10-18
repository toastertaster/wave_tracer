/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "shapes.hpp"

/*
 * Intersection tests, ported from wt/math/intersect/{ray,cone,ball,misc}.hpp,
 * wt/math/util.hpp and the per-node tests of src/ads/bvh8w.cpp. Each function
 * names the CPU function it mirrors; where the CPU uses a SIMD (wide) variant
 * during BVH traversal, the port follows that variant.
 */

namespace wt::gpu::kernel {

/*
 * Rays
 */

struct ray_tri_hit_t {
    real dist;
    real baryx, baryy;    // weights of the triangle's first two vertices
    bool hit;
};

/** @brief Wide `intersect::intersect_ray_tri` (Möller–Trumbore), as used by BVH ray traversal. */
WT_GPU_FN ray_tri_hit_t intersect_ray_tri(const WT_GPU_THREAD ray& r,
                                          vec3 a, vec3 b, vec3 c,
                                          range rng) {
    const vec3 rr  = r.o - a;
    const vec3 e1  = b - a;
    const vec3 e2  = c - a;
    const vec3 crs = cross(r.d, e2);
    const real det = dot(e1, crs);
    const real recp_det = real(1) / det;

    const vec3 q   = cross(rr, e1);
    const real qe2 = dot(q, e2);
    const real betax = dot(rr, crs);
    const real betay = dot(r.d, q);
    const real z = qe2 * recp_det;

    const real baryy = betax * recp_det;
    const real baryz = betay * recp_det;
    const real baryx = real(1) - (baryy + baryz);

    const bool valid = det!=real(0) &&
                       baryx>=real(0) && baryy>=real(0) && baryz>=real(0) &&
                       rng.contains(z);
    return { z, baryx, baryy, valid };
}

/** @brief Wide `intersect::test_ray_tri`, as used by BVH shadow-ray traversal. */
WT_GPU_FN bool test_ray_tri_wide(const WT_GPU_THREAD ray& r,
                                 vec3 a, vec3 b, vec3 c,
                                 range rng) {
    const vec3 rr  = r.o - a;
    const vec3 e1  = b - a;
    const vec3 e2  = c - a;
    const vec3 crs = cross(r.d, e2);
    const real det = dot(e1, crs);
    const real recp_det = real(1) / det;

    const vec3 q   = cross(rr, e1);
    const real qe2 = dot(q, e2);
    const real betax = dot(rr, crs);
    const real betay = dot(r.d, q);
    const real bxy = betax + betay;
    const real z = qe2 * recp_det;

    return det!=real(0) &&
           betax*recp_det>=real(0) && betay*recp_det>=real(0) && bxy*recp_det<=real(1) &&
           z>=rng.min && z<=rng.max;
}

/** @brief Scalar `intersect::intersect_ray_tri`. */
WT_GPU_FN ray_tri_hit_t intersect_ray_tri_scalar(const WT_GPU_THREAD ray& r,
                                                 vec3 a, vec3 b, vec3 c,
                                                 range rng) {
    const vec3 rr  = r.o - a;
    const vec3 e1  = b - a;
    const vec3 e2  = c - a;
    const vec3 crs = cross(r.d, e2);
    real det = dot(e1, crs);
    if (det==real(0))
        return { inf(), 0, 0, false };

    const real sdet = det>=real(0) ? real(1) : real(-1);
    det *= sdet;

    const vec3 q   = cross(rr, e1);
    const real qe2 = sdet*dot(q, e2);
    const real bx = sdet*dot(rr, crs);
    const real by = sdet*dot(r.d, q);
    const range det_range = { det*rng.min, det*rng.max };
    if (bx>=real(0) && by>=real(0) && bx+by<=det && det_range.contains(qe2)) {
        const real recp_det = real(1)/det;
        const real bux = bx * recp_det;
        const real buy = by * recp_det;
        return { qe2 * recp_det, real(1)-(bux+buy), bux, true };
    }
    return { inf(), 0, 0, false };
}

struct node_hit_t {
    real tmin;
    bool hit;
};

/** @brief Wide `intersect::intersect_ray_aabb_fast` for one box (BVH ray traversal). */
WT_GPU_FN node_hit_t intersect_ray_aabb_fast(const WT_GPU_THREAD ray& r,
                                             vec3 bmin, vec3 bmax,
                                             range rng) {
    const bool sx = signbit(r.invd.x), sy = signbit(r.invd.y), sz = signbit(r.invd.z);
    const vec3 mn = { sx ? bmax.x : bmin.x, sy ? bmax.y : bmin.y, sz ? bmax.z : bmin.z };
    const vec3 mx = { sx ? bmin.x : bmax.x, sy ? bmin.y : bmax.y, sz ? bmin.z : bmax.z };
    const vec3 t1 = (mn - r.o) * r.invd;
    const vec3 t2 = (mx - r.o) * r.invd;
    const real rmin = max(max(t1.x, t1.y, t1.z), rng.min);
    const real rmax = min(min(t2.x, t2.y, t2.z), rng.max);
    return { rmin, rmin<=rmax };
}


/*
 * Cones
 */

/**
 * @brief Box test of BVH cone traversal (`cone_cluster_intersect` in src/ads/bvh8w.cpp):
 *        a ray test against the box grown by the cone's radius at the box's far depth.
 */
WT_GPU_FN node_hit_t intersect_cone_aabb_conservative(const WT_GPU_THREAD cone& cn,
                                                      vec3 bmin, vec3 bmax,
                                                      range rng) {
    const ray r = cn.r;
    const bool sx = signbit(r.invd.x), sy = signbit(r.invd.y), sz = signbit(r.invd.z);
    vec3 omin = bmin - r.o;
    vec3 omax = bmax - r.o;

    // farthest z distance from the cone origin
    const vec3 far = { sx ? omin.x : omax.x, sy ? omin.y : omax.y, sz ? omin.z : omax.z };
    const real maxz = clamp(dot(r.d, far), real(0), rng.max);
    // cone cross section at maxz
    const real enlr = fma(maxz, cn.tan_alpha, cn.x0);

    omin = omin - vec3{ enlr, enlr, enlr };
    omax = omax + vec3{ enlr, enlr, enlr };
    const vec3 ae = { sx ? omax.x : omin.x, sy ? omax.y : omin.y, sz ? omax.z : omin.z };
    const vec3 be = { sx ? omin.x : omax.x, sy ? omin.y : omax.y, sz ? omin.z : omax.z };
    const vec3 dmin = ae * r.invd;
    const vec3 dmax = be * r.invd;

    real tmin = 0;
    real tmax = dmax.x;
    tmin = max(tmin, dmin.x);
    tmax = min(tmax, dmax.y);
    tmin = max(tmin, dmin.y);
    tmax = min(tmax, dmax.z);
    tmin = max(tmin, dmin.z);

    return { tmin, tmin<=tmax && tmax>=rng.min && tmin<=rng.max };
}

/** @brief `intersect::intersect_ray_plane<true>` (line-plane), returning the line parameter. */
WT_GPU_FN bool intersect_line_plane(vec3 p0, vec3 p1, vec3 pp, vec3 n,
                                    WT_GPU_THREAD real& t) {
    const real dn = dot(p1-p0, n);
    if (dn==real(0))
        return false;
    t = dot(pp-p0, n) / dn;
    return true;
}

struct cone_edge_hit_t {
    vec3 p0;        // closest intersection point (cone-local)
    bool hit;
};

/**
 * @brief `intersect::intersect_cone_edge<in_local=true>`: edge (segment) against
 *        cone, with near/far clip planes. Points are in the cone's local frame.
 */
WT_GPU_FN cone_edge_hit_t intersect_cone_edge_local(const WT_GPU_THREAD cone& cn,
                                                    vec3 lp0, vec3 lp1,
                                                    range rng) {
    const cone_edge_hit_t none = { { 0,0,0 }, false };

    const bool p0closer = lp1.z>lp0.z;
    vec3 a = lp0, b = lp1;
    if (!p0closer) swap(a,b);

    const vec3 p = a, l = b-a;
    const real x0 = cn.x0;
    const real ta = cn.tan_alpha;
    const real e  = cn.e;

    const real cs = p.z*ta + x0;
    const real epy = e*p.y;
    const real ely = e*l.y;
    const real lzta = l.z*ta;

    const real cc = sqr(p.x) + eft::diff_prod(epy,epy, cs,cs);
    const real bb = 2 * eft::dot(vec3{ p.x,epy,-lzta }, vec3{ l.x,ely,cs });
    const real aa = sqr(l.x) + eft::diff_prod(ely,ely, lzta,lzta);

    const real D = bb*bb - 4*aa*cc;
    if (D<real(0))
        return none;

    const real sqrtD = sqrt(D);
    real t1 = bb>=real(0) ? (-bb-sqrtD)/(2*aa) : (-bb+sqrtD)/(2*aa);
    real t2 = (-bb/aa)-t1;

    const real zapex = cn.z_apex;
    if (p.z + t1*l.z<=zapex)
        t1 = inf();
    if (p.z + t2*l.z<zapex)
        t2 = inf();

    if (t2<t1) swap(t1,t2);
    real z1 = t1<inf() ? p.z + t1*l.z : -inf();
    real z2 = t2<inf() ? p.z + t2*l.z :  inf();

    if (z1>rng.max || z2<rng.min || (!isfinite(z1) && !isfinite(z2)))
        return none;

    const vec3 pl = p+l;
    if (rng.min>zapex && z1<rng.min) {
        real tmin;
        if (intersect_line_plane(p, pl, vec3{ 0,0,rng.min }, vec3{ 0,0,1 }, tmin)) {
            t1 = tmin;
            z1 = rng.min;
        }
    }
    if (z2>rng.max) {
        real tmax;
        if (intersect_line_plane(p, pl, vec3{ 0,0,rng.max }, vec3{ 0,0,1 }, tmax)) {
            t2 = tmax;
            z2 = rng.max;
        }
    }

    const bool has_v1 = t1>=real(0) && real(1)>=t1;
    const bool has_v2 = t2>=real(0) && real(1)>=t2;
    if (!has_v1 && !has_v2)
        return none;
    return { has_v1 ? a + t1*l : a + t2*l, true };
}

struct cone_plane_hit_t {
    range rng;          // empty if no intersection
    vec3 near, far;     // cone-local
};

/** @brief `intersect::intersect_cone_plane<in_local=true>`: plane `dot(n,p)=d`, cone-local. */
WT_GPU_FN cone_plane_hit_t intersect_cone_plane_local(const WT_GPU_THREAD cone& cn,
                                                      vec3 n, real d,
                                                      range rng) {
    const real x0 = cn.x0;
    const real e = cn.one_over_e;
    const real ta = cn.tan_alpha;

    // cross sectional cone position where intersection occurs
    const real v_denom2 = sqr(n.x)+sqr(e*n.y);
    const vec2 v = v_denom2>real(0) ?
        vec2{ n.x, e*n.y } / sqrt(v_denom2) :
        vec2{ 0,0 };
    const vec2 u = v * vec2{ 1,e };
    const real nu = dot(n, vec3{ u.x,u.y,0 });

    const real zapex = cn.z_apex;
    real z01 = (d - x0*nu) / (n.z + ta*nu);
    real z02 = (d + x0*nu) / (n.z - ta*nu);

    // classify and order
    const bool has_z01 = z01>=zapex && !isnan(z01);
    const bool has_z02 = z02>=zapex && !isnan(z02);
    if (!has_z01) z01 = inf();
    if (!has_z02) z02 = inf();
    const vec3 vinf = { inf(),inf(),inf() };
    vec3 p1 = vinf, p2 = vinf;
    if (has_z01) { const vec2 xy = (z01*ta + x0) *   u;  p1 = { xy.x,xy.y,z01 }; }
    if (has_z02) { const vec2 xy = (z02*ta + x0) * (-u); p2 = { xy.x,xy.y,z02 }; }
    if (z01>z02) {
        swap(z01,z02);
        swap(p1,p2);
    }

    range r0 = { z01,z02 };
    if ((!has_z01 && !has_z02) || r0.intersect(rng).empty())
        return { range::null(), vinf, vinf };

    // the point closest to the cone's axis on the plane's intersection with z=const
    // (the CPU passes `v` as this helper's `u`)
    const auto closest = [&](real z) {
        real px, py;
        if (abs(n.y)>abs(n.x)) {
            py = (d-n.z*z)/n.y;
            px = n.x!=real(0) ? (d-n.z*z-n.y*py)/n.x : real(0);
        } else {
            px = (d-n.z*z)/n.x;
            py = n.y!=real(0) ? (d-n.z*z-n.x*px)/n.y : real(0);
        }
        const vec2 xy = (px*v.x+py*v.y) * v;
        return vec3{ xy.x,xy.y,z };
    };

    if (isfinite(r0.min)) {
        if (r0.min<rng.min) {
            p1 = closest(rng.min);
            r0.min = rng.min;
        }
    }
    // when one intersection point is behind and one in front of the apex, infinite plane point is contained in the cone.
    const bool has_infinite = has_z01 != has_z02;
    if (isfinite(r0.max) || has_infinite) {
        if (r0.max>rng.max) {
            p2 = closest(rng.max);
            r0.max = rng.max;
        }
    }

    return { r0, p1, p2 };
}

/** @brief Scalar `util::is_point_in_triangle` (3D; `p` assumed in the triangle's plane). */
WT_GPU_FN bool is_point_in_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 v0 = b-a, v1 = c-a;
    const vec3 u = p-a;
    const real d00 = dot(v0, v0);
    const real d01 = dot(v0, v1);
    const real d11 = dot(v1, v1);
    const real d20 = dot(u, v0);
    const real d21 = dot(u, v1);

    const real d = eft::diff_prod(d00,d11, d01,d01);
    const real sgn = d>real(0) ? real(1) : real(-1);
    const real alpha = eft::diff_prod(d11,d20, d01,d21);
    const real beta  = eft::diff_prod(d00,d21, d01,d20);

    return sgn*alpha>=real(0) && sgn*beta>=real(0) && sgn*(alpha+beta)<=sgn*d;
}

/** @brief Wide `util::is_point_in_triangle`, as used by BVH ball traversal. */
WT_GPU_FN bool is_point_in_triangle_wide(vec3 p, vec3 a, vec3 b, vec3 c) {
    const vec3 v0 = b-a, v1 = c-a;
    const vec3 u = p-a;
    const real d00 = dot(v0, v0);
    const real d01 = dot(v0, v1);
    const real d11 = dot(v1, v1);
    const real d20 = dot(u, v0);
    const real d21 = dot(u, v1);

    const real d = eft::diff_prod(d00,d11, d01,d01);
    const real alpha = eft::diff_prod(d11,d20, d01,d21);
    const real beta  = eft::diff_prod(d00,d21, d01,d20);

    const real sgn = d>real(0) ? real(1) : real(-1);
    const real sa = alpha * sgn;
    const real sb = beta  * sgn;
    return sa>=real(0) && sb>=real(0) && (sa+sb)<=sgn*d;
}

struct cone_tri_hit_t {
    real dist;
    bool hit;
};

/** @brief `intersect::intersect_cone_tri`: distance along the cone axis to the triangle, if it is hit. */
WT_GPU_FN cone_tri_hit_t intersect_cone_tri(const WT_GPU_THREAD cone& cn,
                                            vec3 a, vec3 b, vec3 c, vec3 n,
                                            range rng) {
    const cone_tri_hit_t none = { inf(), false };

    if (cn.is_ray()) {
        // degenerate case: ray-triangle intersection
        const auto cr = intersect_ray_tri_scalar(cn.r, a,b,c, rng);
        return cr.hit ? cone_tri_hit_t{ cr.dist, true } : none;
    }

    const kernel::frame f = cn.frame();
    const vec3 o = cn.r.o;
    // triangle in the cone's local frame
    const vec3 vs[3] = { f.to_local(a-o), f.to_local(b-o), f.to_local(c-o) };
    const vec3 ln = f.to_local(n);

    bool contains[3];
    for (int i=0;i<3;++i)
        contains[i] = cn.contains_local(vs[i], rng);

    // fast reject: all points before near clip or beyond far clip?
    const real closest_z  = min(vs[0].z, vs[1].z, vs[2].z);
    const real farthest_z = max(vs[0].z, vs[1].z, vs[2].z);
    if (farthest_z<rng.min || closest_z>rng.max)
        return none;

    // fast accept: closest vertex inside?
    for (int i=0;i<3;++i)
        if (contains[i] && vs[i].z==closest_z)
            return { closest_z, true };

    // closest point on the cone-plane intersection conic section
    const auto icp = intersect_cone_plane_local(cn, ln, dot(vs[0],ln), rng);
    if (!icp.rng.empty() && is_point_in_triangle(icp.near, vs[0], vs[1], vs[2]))
        return { icp.rng.min, true };

    // test all edges
    bool found = false;
    real pz = 0;
    for (int i=0;i<3;++i) {
        const int j = (i+1)%3;
        const vec3 ea = vs[i];
        const vec3 eb = vs[j];

        if (contains[i] && contains[j]) continue;
        if (ea.z>rng.max && eb.z>rng.max) continue;
        if (ea.z<rng.min && eb.z<rng.min) continue;

        const auto cp = intersect_cone_edge_local(cn, ea,eb, rng);
        if (cp.hit && (!found || pz>cp.p0.z)) {
            found = true;
            pz = cp.p0.z;
        }
    }

    return found ? cone_tri_hit_t{ pz, true } : none;
}


/*
 * Balls
 */

/** @brief Wide `intersect::test_ball_aabb` (the `intersects` result). */
WT_GPU_FN bool test_ball_aabb(const WT_GPU_THREAD ball& bl, vec3 bmin, vec3 bmax) {
    const vec3 c = bl.centre;
    const vec3 closest = { clamp(c.x, bmin.x, bmax.x), clamp(c.y, bmin.y, bmax.y), clamp(c.z, bmin.z, bmax.z) };
    const vec3 d = c - closest;
    return dot(d,d) <= sqr(bl.radius);
}

/** @brief Wide `intersect::test_ball_tri`. */
WT_GPU_FN bool test_ball_tri(const WT_GPU_THREAD ball& bl, vec3 a, vec3 b, vec3 c, vec3 n) {
    const vec3 bc = bl.centre;
    const real dt = dot(n, a-bc);
    // point on triangle plane closest to ball centre
    const vec3 p = { fma(dt,n.x,bc.x), fma(dt,n.y,bc.y), fma(dt,n.z,bc.z) };

    return bl.contains(a) || bl.contains(b) || bl.contains(c) ||
           (is_point_in_triangle_wide(p, a,b,c) && bl.contains(p));
}

}
