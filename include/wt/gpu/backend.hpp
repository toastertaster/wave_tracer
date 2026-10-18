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
 * Interface between wave_tracer and its GPU backends (CUDA/OptiX, later Metal).
 *
 * This header is C++17 and depends only on the standard library and the shared
 * kernel types (wt/gpu/kernel/), so backend sources compiled by GPU compilers
 * (nvcc cannot parse mp-units) can include it. wave_tracer code uses the C++20
 * API in wt/gpu/gpu.hpp, which translates its types into these.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <wt/gpu/kernel/bvh.hpp>

namespace wt::gpu {

/** @brief The BVH, as uploaded to a device. The root is node 0. */
struct bvh_arrays_t {
    std::vector<kernel::bvh_node> nodes;
    std::vector<kernel::bvh_leaf> leaves;
    std::vector<kernel::bvh_tri>  tris;
};

struct ray_query_t {
    kernel::ray ray;
    kernel::range range;
};

struct cone_query_t {
    kernel::cone cone;
    kernel::range range;
    /** @brief Triangles are collected this many cross-section major axes past the closest hit. */
    kernel::real z_scale;
};

/** @brief Result of a ball query. */
struct ball_hit_t {
    std::uint32_t tri_count;    // may exceed the capacity the query was run with
    std::uint32_t status;
};

/**
 * @brief Scene geometry on a device. Queries are batched and synchronous;
 *        results are in query order. Not thread safe.
 */
class backend_geometry_t {
public:
    virtual ~backend_geometry_t() noexcept = default;

    virtual void intersect_rays(const ray_query_t* queries, std::size_t count,
                                kernel::ray_hit* hits) = 0;
    /** @brief `occluded[i]` is 1 if something blocks ray `i`. */
    virtual void rays_occluded(const ray_query_t* queries, std::size_t count,
                               std::uint8_t* occluded, std::uint32_t* status) = 0;

    /**
     * @brief Query `i` writes up to `capacity` triangle indices to
     *        `tris[i*capacity ...]`; `hits[i].tri_count` counts all it found.
     */
    virtual void intersect_cones(const cone_query_t* queries, std::size_t count,
                                 std::uint32_t capacity,
                                 kernel::cone_hit* hits, std::uint32_t* tris) = 0;
    /** @brief As `intersect_cones`. */
    virtual void intersect_balls(const kernel::ball* queries, std::size_t count,
                                 std::uint32_t capacity,
                                 ball_hit_t* hits, std::uint32_t* tris) = 0;
};

/** @brief A GPU backend on one device. */
class backend_t {
public:
    virtual ~backend_t() noexcept = default;

    /** @brief e.g. "CUDA (OptiX 9.0) on NVIDIA GB10". */
    [[nodiscard]] virtual std::string description() const = 0;

    [[nodiscard]] virtual std::unique_ptr<backend_geometry_t> upload(const bvh_arrays_t& bvh) = 0;
};

#if defined(WT_GPU_CUDA)
/**
 * @brief Creates the CUDA backend on CUDA device `ordinal`. With `use_optix`
 *        (requires WT_GPU_OPTIX), ray queries use OptiX's hardware traversal.
 *        Throws std::runtime_error if the device or OptiX is unavailable.
 */
[[nodiscard]] std::unique_ptr<backend_t> create_cuda_backend(int ordinal, bool use_optix);
#endif

}
