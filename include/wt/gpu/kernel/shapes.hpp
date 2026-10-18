/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include "math.hpp"

/*
 * Query shapes, mirroring the CPU types (all lengths in metres):
 *   range  - wt::pqrange_t<> (inclusive)
 *   ray    - wt::ray_t
 *   cone   - wt::elliptic_cone_t
 *   ball   - wt::ball_t
 * Host code fills them from the CPU types (wt/gpu/gpu.hpp), so derived values
 * (ray::invd, cone frame and apex) are computed exactly as the CPU computes
 * them.
 */

namespace wt::gpu::kernel {

/** @brief Inclusive range, as `wt::pqrange_t<>`. */
struct range {
    real min, max;

    WT_GPU_FN bool contains(real pt) const {
        return (pt<max && min<pt) || pt==min || pt==max;
    }
    WT_GPU_FN bool empty() const {
        // empty when min==max==±∞
        if (min==max && !isfinite(min))
            return true;
        return min>max;
    }
    WT_GPU_FN range intersect(range o) const {
        return { kernel::max(min,o.min), kernel::min(max,o.max) };
    }
    WT_GPU_FN range grow(real extent) const { return { min-extent, max+extent }; }

    WT_GPU_FN static range positive() { return { real(0), inf() }; }
    WT_GPU_FN static range all() { return { -inf(), inf() }; }
    WT_GPU_FN static range null() { return { inf(), -inf() }; }
};

/** @brief A ray, as `wt::ray_t`. `invd` is `1/d`, computed by the host. */
struct ray {
    vec3 o;
    vec3 d;
    vec3 invd;

    WT_GPU_FN vec3 propagate(real dist) const { return o + dist*d; }
};

/** @brief A local orthonormal frame, as `wt::frame_t`. */
struct frame {
    vec3 t, b, n;

    WT_GPU_FN vec3 to_local(vec3 v) const { return { dot(v,t), dot(v,b), dot(v,n) }; }
    WT_GPU_FN vec3 to_world(vec3 v) const { return t*v.x + b*v.y + n*v.z; }
};

/**
 * @brief An elliptic cone, as `wt::elliptic_cone_t`: apex region at `r.o`,
 *        opening around `r.d`, major axis along `x` (`y = cross(d,x)`).
 */
struct cone {
    ray r;
    vec3 x, y;
    real x0;            // initial major axis length
    real tan_alpha;     // tan of half opening angle
    real e;             // major/minor
    real one_over_e;    // minor/major
    real z_apex;        // -inf for a ray

    WT_GPU_FN bool is_ray() const { return tan_alpha==real(0) && x0==real(0); }
    WT_GPU_FN kernel::frame frame() const { return { x, y, r.d }; }

    /** @brief Major and minor axes of the cross section at distance `z`. */
    WT_GPU_FN vec2 axes(real z) const {
        const real rr = tan_alpha*z + x0;
        return vec2{ real(1), one_over_e } * rr;
    }

    /** @brief `elliptic_cone_t::contains_local` (wide variant). */
    WT_GPU_FN bool contains_local(vec3 p, range rng) const {
        const real x2  = sqr(p.x);
        const real ey  = p.y * e;
        const real ztx = fma(p.z, tan_alpha, x0);
        return z_apex<=p.z && rng.contains(p.z) && (x2+sqr(ey))<=sqr(ztx);
    }
};

/** @brief A ball, as `wt::ball_t`. */
struct ball {
    vec3 centre;
    real radius;

    WT_GPU_FN bool contains(vec3 p) const {
        const vec3 cp = centre - p;
        return dot(cp,cp) < sqr(radius);
    }
};

}
