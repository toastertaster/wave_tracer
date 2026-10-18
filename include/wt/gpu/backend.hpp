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
#include <wt/gpu/kernel/scene.hpp>

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
 * @brief A scene compiled for forward plt_path rendering of one sensor at one
 *        wavenumber (see wt/gpu/kernel/scene.hpp). The device pointers of
 *        `kernel::plt_scene` are filled in by the backend.
 */
struct plt_path_scene_t {
    std::vector<kernel::tri_shading> tris;     // per BVH triangle
    std::vector<kernel::edge> edges;
    std::vector<kernel::material> materials;
    std::vector<kernel::point_emitter> emitters;
    std::vector<kernel::real> emitter_cdf;
    kernel::virtual_plane_sensor sensor;
    kernel::film_desc film;
    kernel::plt_path_params params;
};

/** @brief Traces paths of a compiled scene on a device, accumulating a film there. Not thread safe. */
class backend_renderer_t {
public:
    virtual ~backend_renderer_t() noexcept = default;

    /**
     * @brief Traces paths `first .. first+count-1` and accumulates them into
     *        the film. A path's index and `seed` determine its random numbers.
     *        Returns the status flags of all paths (kernel::status_*).
     */
    virtual std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed) = 0;

    /** @brief Copies the film to `out`: element (x,y) component c at `out[(y*width + x)*components + c]`. */
    virtual void read_film(double* out) = 0;
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

    /** @brief Uploads a scene compiled for this geometry, for rendering with forward plt_path. */
    [[nodiscard]] virtual std::unique_ptr<backend_renderer_t> create_plt_path_renderer(
            const plt_path_scene_t& scene) = 0;
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
