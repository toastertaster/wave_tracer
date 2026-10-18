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
 * Rendering on a GPU (experimental).
 *
 * `scene_renderer_t` renders on `wt_context_t::gpu_device` when it is set.
 * GPU rendering implements a subset of wave_tracer; scenes outside it fail
 * with a `gpu_error_t` that names the unsupported feature, rather than
 * silently rendering on the CPU. Supported so far:
 *   - the plt_path integrator, forward transport (beams, free-space
 *     diffraction, or ray tracing);
 *   - virtual_plane (coverage) sensors sensitive to a single wavelength;
 *   - point emitters;
 *   - surface_spm BSDFs with a Dirac surface profile (specular), optionally
 *     wrapped in twosided and composite BSDFs;
 *   - the bvh8w acceleration structure.
 */

#include <cstdint>
#include <memory>

#include <wt/gpu/gpu.hpp>

namespace wt {

class scene_t;
struct wt_context_t;

namespace sensor {
class sensor_t;
class film_storage_handle_t;
}

}

namespace wt::gpu {

class backend_renderer_t;

/** @brief Renders one sensor of a scene on a GPU. Not thread safe. */
class renderer_t {
public:
    /**
     * @brief Compiles the scene for `sensor` and uploads it to `device`.
     *        Throws gpu_error_t, naming the first unsupported scene feature,
     *        if the GPU cannot render it.
     */
    renderer_t(const device_t& device,
               const scene_t& scene,
               const sensor::sensor_t& sensor,
               const ads::ads_t& ads);
    ~renderer_t() noexcept;

    renderer_t(const renderer_t&) = delete;
    renderer_t& operator=(const renderer_t&) = delete;

    /**
     * @brief Paths per sample per element: forward transport traces as many
     *        paths per sample as the sensor has elements, as the CPU does.
     */
    [[nodiscard]] std::uint64_t paths_per_sample() const noexcept { return paths_per_spe; }

    /**
     * @brief Traces samples `first .. first+count-1` of every sensor element
     *        and accumulates them. Some paths may still be in flight on return
     *        (the GPU keeps working on them with the next samples): `finish`
     *        completes them.
     */
    void render(std::uint64_t first, std::uint64_t count);

    /** @brief Completes the paths in flight, so that the film holds all samples rendered. */
    void finish();

    /**
     * @brief Writes the accumulated samples into `film`, a film storage the
     *        sensor created for direct splats (as forward integrators write).
     *        Paths in flight are missing (see `render`).
     */
    void write_film(sensor::film_storage_handle_t& film) const;

private:
    std::unique_ptr<geometry_t> geometry;
    std::unique_ptr<backend_renderer_t> backend;

    std::uint64_t paths_per_spe;
    std::size_t film_values;
    bool warned_iteration_limit = false;

    void check_status(std::uint32_t status);
};

}
