/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * OptiX device programs for ray queries, compiled to OptiX-IR and embedded
 * (cmake/wt_gpu.cmake). They return what kernel::intersect_ray and
 * kernel::ray_occluded return, using OptiX's hardware triangle intersection:
 * hit points agree with the CUDA traversal up to floating-point rounding, but
 * rays grazing an edge or vertex may hit a different neighbouring triangle.
 */

#include <optix.h>

#include "optix_params.hpp"

using namespace wt::gpu;
namespace k = wt::gpu::kernel;

// Defined (not declared extern) in a C linkage block, as in the OptiX SDK samples.
extern "C" {
__constant__ optix::launch_params_t params;
}

namespace {

__forceinline__ __device__ float3 to_float3(k::vec3 v) { return make_float3(v.x, v.y, v.z); }

}

extern "C" __global__ void __raygen__closest() {
    const unsigned i = optixGetLaunchIndex().x;
    const ray_query_t q = params.queries[i];

    unsigned hit = 0, t = 0, u = 0, v = 0, prim = 0;
    optixTrace(params.handle, to_float3(q.ray.o), to_float3(q.ray.d),
               q.range.min, q.range.max, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, 1, 0,
               hit, t, u, v, prim);

    k::ray_hit h = { k::inf(), 0, 0, 0, 0, 0, k::status_ok };
    if (hit) {
        const float bu = __uint_as_float(u), bv = __uint_as_float(v);
        h.dist = __uint_as_float(t);
        // as kernel::intersect_ray_tri: weights of the first two vertices
        h.baryx = 1.f - (bu + bv);
        h.baryy = bu;
        h.tri = prim;
        h.hit = 1;
        // front faces are decided by the stored normal, as on the CPU, not by winding
        h.front_face = k::dot(params.tris[prim].n, q.ray.d)<=0.f ? 1u : 0u;
    }
    params.hits[i] = h;
}

extern "C" __global__ void __raygen__shadow() {
    const unsigned i = optixGetLaunchIndex().x;
    const ray_query_t q = params.queries[i];

    unsigned occluded = 1;      // cleared by the miss program
    optixTrace(params.handle, to_float3(q.ray.o), to_float3(q.ray.d),
               q.range.min, q.range.max, 0.f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               0, 1, 0,
               occluded);
    params.occluded[i] = std::uint8_t(occluded);
}

extern "C" __global__ void __closesthit__ray() {
    const float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(1);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(optixGetPrimitiveIndex());
}

extern "C" __global__ void __miss__ray() {
    optixSetPayload_0(0);
}
