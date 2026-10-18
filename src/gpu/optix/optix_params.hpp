/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <optix_types.h>

#include <wt/gpu/backend.hpp>

namespace wt::gpu::optix {

/** @brief Launch parameters of the OptiX ray programs (optix_programs.cu). */
struct launch_params_t {
    OptixTraversableHandle handle;
    const ray_query_t* queries;
    /** @brief Triangles in the BVH's order, for the normals that decide front faces. */
    const kernel::bvh_tri* tris;
    /** @brief Output of closest-hit launches. */
    kernel::ray_hit* hits;
    /** @brief Output of shadow launches. */
    std::uint8_t* occluded;
};

}
