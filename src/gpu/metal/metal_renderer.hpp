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
 * Metal backend: plt_path and plt_bdpt rendering. Objective-C++; included only
 * by the backend's .mm sources.
 */

#include <memory>

#include <wt/gpu/backend.hpp>

#include "metal_util.hpp"

namespace wt::gpu::metal_backend {

/**
 * @brief A scene's geometry on the GPU (`metal_geometry_t`'s), which renders
 *        trace their beams and rays through: its BVH, and, if rays are traced
 *        by hardware, the acceleration structure of its triangles (else nil).
 */
struct bvh_buffers_t {
    id<MTLBuffer> nodes;
    id<MTLBuffer> tris;
    id<MTLAccelerationStructure> accel;
};

/** @brief Uploads `scene`, compiled for the geometry of `bvh`, for rendering. */
[[nodiscard]] std::unique_ptr<backend_renderer_t> create_plt_renderer(std::shared_ptr<context_t> context,
                                                                      bvh_buffers_t bvh,
                                                                      const plt_scene_t& scene);

}
