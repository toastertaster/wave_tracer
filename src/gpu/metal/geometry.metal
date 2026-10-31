/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * Metal backend: acceleration-structure query kernels (the Metal counterparts
 * of cuda_backend.cu's), and the struct layout check (see layout.hpp).
 */

#include <wt/gpu/kernel/queries.hpp>

#include "layout.hpp"

using namespace metal;
namespace k = wt::gpu::compute_kernel;

namespace {

// Kernel entry points must name their buffers' address spaces; the shared
// traversals take the BVH as a k::bvh_view of device pointers.
k::bvh_view make_view(const device k::bvh_node* nodes, const device k::bvh_tri* tris) {
    return { nodes, tris };
}

}

kernel void intersect_rays(const device k::bvh_node* nodes   [[buffer(0)]],
                           const device k::bvh_tri* tris     [[buffer(1)]],
                           const device k::ray_query* queries [[buffer(2)]],
                           device k::ray_hit* hits           [[buffer(3)]],
                           constant uint& count              [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    const k::ray_query q = queries[i];
    hits[i] = k::intersect_ray(make_view(nodes, tris), q.ray, q.range);
}

kernel void rays_occluded(const device k::bvh_node* nodes   [[buffer(0)]],
                          const device k::bvh_tri* tris     [[buffer(1)]],
                          const device k::ray_query* queries [[buffer(2)]],
                          device uchar* occluded            [[buffer(3)]],
                          constant uint& count              [[buffer(4)]],
                          device atomic_uint* status        [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    const k::ray_query q = queries[i];
    k::u32 s = k::status_ok;
    occluded[i] = k::ray_occluded(make_view(nodes, tris), q.ray, q.range, s) ? 1 : 0;
    if (s!=k::status_ok)
        atomic_fetch_or_explicit(status, s, memory_order_relaxed);
}

kernel void intersect_cones(const device k::bvh_node* nodes    [[buffer(0)]],
                            const device k::bvh_tri* tris      [[buffer(1)]],
                            const device k::cone_query* queries [[buffer(2)]],
                            device k::cone_hit* hits           [[buffer(3)]],
                            constant uint& count               [[buffer(4)]],
                            constant uint& capacity            [[buffer(5)]],
                            device k::u32* tri_lists           [[buffer(6)]],
                            uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    const k::cone_query q = queries[i];
    k::tri_sink sink = { tri_lists + ulong(i)*capacity, capacity, 0 };
    hits[i] = k::intersect_cone(make_view(nodes, tris), q.cone, q.range, q.z_scale, sink);
}

kernel void intersect_balls(const device k::bvh_node* nodes [[buffer(0)]],
                            const device k::bvh_tri* tris   [[buffer(1)]],
                            const device k::ball* queries   [[buffer(2)]],
                            device k::ball_hit* hits        [[buffer(3)]],
                            constant uint& count            [[buffer(4)]],
                            constant uint& capacity         [[buffer(5)]],
                            device k::u32* tri_lists        [[buffer(6)]],
                            uint i [[thread_position_in_grid]]) {
    if (i>=count) return;
    k::tri_sink sink = { tri_lists + ulong(i)*capacity, capacity, 0 };
    const auto status = k::intersect_ball(make_view(nodes, tris), queries[i], sink);
    hits[i] = { sink.count, status };
}

kernel void struct_layouts(device k::u32* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i==0)
        wt::gpu::metal_backend::write_layouts(out);
}
