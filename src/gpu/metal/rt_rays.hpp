/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

/*
 * Metal backend: ray queries through Metal's ray tracing, on the GPU's
 * ray-tracing hardware (Apple M3 and later), into an acceleration structure
 * of the BVH's triangles, in the BVH's order (metal_backend.mm). They return
 * what compute_kernel::intersect_ray and compute_kernel::ray_occluded return,
 * as the OptiX programs do (src/gpu/optix/optix_programs.cu): hit points agree
 * with the BVH traversal up to floating-point rounding, but rays grazing an
 * edge or vertex may hit a different neighbouring triangle. Ranges must not
 * start below 0. Metal Shading Language only.
 */

#include <metal_raytracing>

#include <wt/gpu/kernel/bvh.hpp>

namespace wt { namespace gpu { namespace metal_backend {

using acceleration_structure = metal::raytracing::primitive_acceleration_structure;

inline metal::raytracing::ray rt_ray(compute_kernel::ray r, compute_kernel::range rng) {
    return metal::raytracing::ray(float3(r.o.x, r.o.y, r.o.z), float3(r.d.x, r.d.y, r.d.z), rng.min, rng.max);
}

/** @brief compute_kernel::intersect_ray, with Metal's ray tracing; `tris` are the BVH's (for their normals). */
inline compute_kernel::ray_hit rt_intersect(acceleration_structure accel,
                                            const device compute_kernel::bvh_tri* tris,
                                            compute_kernel::ray r, compute_kernel::range rng) {
    using namespace metal::raytracing;
    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    const auto res = isect.intersect(rt_ray(r, rng), accel);

    compute_kernel::ray_hit h = { compute_kernel::inf(), 0, 0, 0, 0, 0, compute_kernel::status_ok };
    if (res.type==intersection_type::triangle) {
        const float2 bary = res.triangle_barycentric_coord;
        h.dist = res.distance;
        // as compute_kernel::intersect_ray_tri: weights of the first two vertices
        h.baryx = 1.f - (bary.x + bary.y);
        h.baryy = bary.x;
        h.tri = res.primitive_id;
        h.hit = 1;
        // front faces are decided by the stored normal, as on the CPU, not by winding
        h.front_face = compute_kernel::dot(tris[res.primitive_id].n, r.d)<=0.f ? 1u : 0u;
    }
    return h;
}

/** @brief compute_kernel::ray_occluded, with Metal's ray tracing. */
inline bool rt_occluded(acceleration_structure accel, compute_kernel::ray r, compute_kernel::range rng) {
    using namespace metal::raytracing;
    intersector<triangle_data> isect;
    isect.assume_geometry_type(geometry_type::triangle);
    isect.force_opacity(forced_opacity::opaque);
    isect.accept_any_intersection(true);
    return isect.intersect(rt_ray(r, rng), accel).type!=intersection_type::none;
}

} } }
