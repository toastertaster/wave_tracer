/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <wt/gpu/backend.hpp>

/*
 * OptiX ray queries for the CUDA backend (32-bit builds only). The OptiX types
 * stay in optix_rays.cpp.
 */

namespace wt::gpu::optix {

/** @brief OptiX on one CUDA device: device context and the ray-query pipeline. */
class context_t {
public:
    explicit context_t(int ordinal);
    ~context_t() noexcept;

    /** @brief e.g. "9.0" */
    [[nodiscard]] std::string version() const;

    struct impl_t;
    [[nodiscard]] const impl_t& impl() const noexcept { return *pimpl; }

private:
    std::unique_ptr<impl_t> pimpl;
};

/** @brief A scene's triangles in an OptiX acceleration structure. */
class ray_geometry_t {
public:
    /**
     * @param bvh the scene; must have triangles
     * @param device_tris `bvh.tris` in device memory, which must outlive this object
     */
    ray_geometry_t(const context_t& context, const bvh_arrays_t& bvh, const kernel::bvh_tri* device_tris);
    ~ray_geometry_t() noexcept;

    /** @brief As kernel::intersect_ray for each query (ranges must not start below 0); all pointers are device memory. */
    void intersect(const ray_query_t* queries, std::size_t count, kernel::ray_hit* hits) const;
    /** @brief As kernel::ray_occluded for each query (ranges must not start below 0); all pointers are device memory. */
    void occluded(const ray_query_t* queries, std::size_t count, std::uint8_t* occluded) const;

private:
    struct impl_t;
    std::unique_ptr<impl_t> pimpl;
};

}
