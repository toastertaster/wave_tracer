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
 * GPU acceleration (experimental).
 *
 * A `gpu::device_t` is one GPU, through the backend this build includes
 * (CMake GPU_CUDA, GPU_OPTIX). Applications create one and set it as
 * `wt_context_t::gpu_device`; the CPU implementation is used when it is null,
 * and a build without a GPU backend never creates one. A `gpu::geometry_t` is
 * a scene's acceleration structure on that device, answering batches of the
 * same queries as `ads::ads_t`.
 */

#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <wt/ads/ads.hpp>
#include <wt/math/intersect/intersect_defs.hpp>
#include <wt/math/shapes/ball.hpp>
#include <wt/math/shapes/elliptic_cone.hpp>
#include <wt/math/shapes/ray.hpp>

namespace wt::gpu {

class backend_t;
class backend_geometry_t;
class geometry_t;

/** @brief A GPU was requested but is unavailable, or cannot do what was asked. */
class gpu_error_t : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/** @brief The GPU backends this build includes, e.g. "CUDA, OptiX"; empty if none. */
[[nodiscard]] std::string compiled_backends();

/** @brief A GPU. */
class device_t {
public:
    struct options_t {
        /** @brief Device index within the backend (CUDA device ordinal). */
        int ordinal = 0;
        /** @brief Use OptiX for ray queries, when built with GPU_OPTIX and GPU_DBL_PRECISION=OFF. */
        bool use_optix = true;
    };

    /**
     * @brief Opens a GPU with the backend this build includes.
     *        Throws gpu_error_t, explaining why, if the build has no GPU backend
     *        or the device cannot be used.
     */
    [[nodiscard]] static std::unique_ptr<device_t> create(const options_t& options);
    [[nodiscard]] static std::unique_ptr<device_t> create() { return create(options_t{}); }

    ~device_t() noexcept;

    /** @brief e.g. "CUDA (OptiX 9.0) on NVIDIA GB10". */
    [[nodiscard]] std::string description() const;

    /**
     * @brief Uploads a scene's acceleration structure. Throws gpu_error_t if the
     *        ADS type is not supported on the GPU (only bvh8w is).
     */
    [[nodiscard]] std::unique_ptr<geometry_t> upload(const ads::ads_t& ads) const;

private:
    explicit device_t(std::unique_ptr<backend_t> backend) noexcept;

    std::unique_ptr<backend_t> backend;
};

/**
 * @brief A scene's acceleration structure on a GPU: batched versions of the
 *        `ads::ads_t` queries, returning the same results (up to floating-point
 *        rounding). Thread safe; concurrent calls are serialized.
 */
class geometry_t {
public:
    struct ray_hit_t {
        intersect::intersect_ray_tri_ret_t intersection;
        ads::tuid_t tuid;
        bool front_face;
    };

    /**
     * @brief Triangle lists of a batch of cone or ball queries: query `i` found
     *        `triangles[offsets[i] .. offsets[i+1])`, in the CPU's traversal order.
     */
    struct triangle_lists_t {
        std::vector<std::size_t> offsets;
        std::vector<ads::tuid_t> triangles;

        [[nodiscard]] std::span<const ads::tuid_t> of(std::size_t i) const noexcept {
            return { triangles.data()+offsets[i], triangles.data()+offsets[i+1] };
        }
    };

    struct cone_hits_t {
        /** @brief Distance to each query's closest hit; infinite if it hit nothing. */
        std::vector<length_t> distances;
        /** @brief Whether each query's closest hit is front facing. */
        std::vector<bool> front_faces;
        triangle_lists_t triangles;
    };

    ~geometry_t() noexcept;

    /** @brief `ads_t::intersect(ray, range)` for each ray. */
    [[nodiscard]] std::vector<std::optional<ray_hit_t>> intersect(
            std::span<const ray_t> rays, const pqrange_t<>& range) const;
    /** @brief `ads_t::shadow(ray, range)` for each ray. */
    [[nodiscard]] std::vector<bool> shadow(
            std::span<const ray_t> rays, const pqrange_t<>& range) const;
    /** @brief `ads_t::intersect(cone, range, opts)` for each cone, with `opts.z_search_range_scale = z_search_range_scale`. */
    [[nodiscard]] cone_hits_t intersect(
            std::span<const elliptic_cone_t> cones, const pqrange_t<>& range,
            f_t z_search_range_scale = 1) const;
    /** @brief `ads_t::intersect(ball)` for each ball. */
    [[nodiscard]] triangle_lists_t intersect(std::span<const ball_t> balls) const;

private:
    friend class device_t;
    explicit geometry_t(std::unique_ptr<backend_geometry_t> backend) noexcept;

    std::unique_ptr<backend_geometry_t> backend;
    mutable std::mutex mutex;
};

}
