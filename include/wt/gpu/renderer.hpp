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
 *   - the plt_path integrator, forward and backward transport (beams,
 *     free-space diffraction, or ray tracing);
 *   - the plt_bdpt integrator (beams, with its Fraunhofer free-space
 *     diffraction, or ray tracing);
 *   - virtual_plane (coverage) sensors, and perspective sensors (but not
 *     with forward plt_path), polarimetric or not, of any response: paths draw
 *     wavenumbers as the CPU's do (see `spectral_tables_t`);
 *   - point, spot and directional emitters, and area emitters of a constant
 *     radiance;
 *   - surface_spm BSDFs, smooth (Dirac surface profile) or rough (fractal
 *     or gaussian surface profiles), dielectric and diffuse BSDFs,
 *     optionally wrapped in twosided, composite, scale and normalmap BSDFs;
 *   - constant and bitmap textures, transformed, or scaled by constant
 *     textures, for diffuse reflectances, BSDF scales, surface roughness
 *     (perceptual) and normal maps (see `compute_kernel::texture`);
 *   - the bvh8w acceleration structure.
 */

#include <cstdint>
#include <memory>

#include <wt/gpu/gpu.hpp>
#include <wt/gpu/kernel/scene.hpp>

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
class spectral_tables_t;

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
    /**
     * @brief As above, but with `ads` already uploaded (`device_t::upload`),
     *        which the renderer shares rather than uploading it again. Throws
     *        gpu_error_t if `geometry` is not an upload of `ads`.
     */
    renderer_t(std::shared_ptr<const geometry_t> geometry,
               const scene_t& scene,
               const sensor::sensor_t& sensor,
               const ads::ads_t& ads);
    ~renderer_t() noexcept;

    renderer_t(const renderer_t&) = delete;
    renderer_t& operator=(const renderer_t&) = delete;

    /**
     * @brief Paths per sample per element: as many paths per sample as the
     *        sensor has elements, as the CPU traces (backward, a path from
     *        each element; forward, as many paths from the emitters).
     */
    [[nodiscard]] std::uint64_t paths_per_sample() const noexcept { return paths_per_spe; }

    /** @brief The scene's ADS on the GPU, which other renderers of the scene may share. */
    [[nodiscard]] const std::shared_ptr<const geometry_t>& uploaded_geometry() const noexcept { return geometry; }

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
     *        sensor created for the integrator's splats: block splats
     *        (backward plt_path), direct splats (forward plt_path), or both
     *        (plt_bdpt). Paths in flight are missing (see `render`).
     */
    void write_film(sensor::film_storage_handle_t& film) const;

private:
    /** @brief Compiles the scene, then creates the backend renderer on the geometry `upload` returns. */
    template <typename Upload>
    void create(const scene_t& scene, const sensor::sensor_t& sensor, const ads::ads_t& ads,
                const Upload& upload);

    std::shared_ptr<const geometry_t> geometry;
    std::unique_ptr<backend_renderer_t> backend;
    std::unique_ptr<spectral_tables_t> spectra;

    compute_kernel::plt_params params;
    std::uint64_t paths_per_spe;
    std::size_t film_elements;          // film elements' components
    bool polarimetric = false;          // ... each a Stokes parameters vector
    bool warned_iteration_limit = false;

    void check_status(std::uint32_t status);
};

}
