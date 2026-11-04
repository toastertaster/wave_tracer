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
 * Interface between wave_tracer and its GPU backends (CUDA/OptiX, Metal).
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
#include <wt/gpu/kernel/queries.hpp>
#include <wt/gpu/kernel/scene.hpp>

namespace wt::gpu {

/** @brief The BVH, as uploaded to a device. The root is node 0. */
struct bvh_arrays_t {
    std::vector<compute_kernel::bvh_node> nodes;
    std::vector<compute_kernel::bvh_tri>  tris;
};

using ray_query_t  = compute_kernel::ray_query;
using cone_query_t = compute_kernel::cone_query;
using ball_hit_t   = compute_kernel::ball_hit;

/**
 * @brief A scene compiled for plt_path or plt_bdpt rendering of one sensor
 *        (see wt/gpu/kernel/scene.hpp), but for what depends on the
 *        wavenumber: see `spectral_table_t`. The device pointers of
 *        `compute_kernel::plt_scene` are filled in by the backend.
 */
struct plt_scene_t {
    std::vector<compute_kernel::tri_shading> tris;     // per BVH triangle
    std::vector<compute_kernel::edge> edges;
    std::vector<compute_kernel::emitter> emitters;
    std::vector<compute_kernel::real> emitter_cdf;
    std::vector<std::uint32_t> area_tris;      // area emitters' triangles (see `compute_kernel::plt_scene`)
    std::vector<compute_kernel::real> area_cdf;
    std::vector<compute_kernel::tri_uv> tri_uvs;       // per BVH triangle, if any material has textures
    std::vector<compute_kernel::material_shading> material_shadings;   // per material
    std::vector<compute_kernel::texture> textures;
    std::vector<compute_kernel::bitmap> bitmaps;
    std::vector<std::uint32_t> texels;         // `compute_kernel::bitmap`s' texels
    std::vector<compute_kernel::real> texel_luts;
    std::vector<compute_kernel::real> fsd_lut;         // plt_bdpt's free-space diffraction, or empty (see `compute_kernel::plt_scene`)
#if defined(REALISTIC_REMOTE_SENSING)
    std::vector<compute_kernel::telescope_desc> telescope;     // a telescope sensor's, or empty
#endif
    std::uint32_t emitter_count;
    std::uint32_t material_count;
    std::uint32_t samples_per_emitter;         // in each spectral table
    compute_kernel::sensor_desc sensor;
    compute_kernel::film_desc film;
    compute_kernel::plt_params params;
};

/**
 * @brief Spectral samples (see `compute_kernel::plt_scene`): `samples[s]`, with
 *        emitter e at `emissions[s*emitter_count + e]` and material m at
 *        `materials[s*material_count + m]`. A table for `trace` holds
 *        `samples_per_emitter` samples of each emitter, emitter by emitter.
 */
struct spectral_table_t {
    std::vector<compute_kernel::spectral_sample> samples;
    std::vector<compute_kernel::emission> emissions;
    std::vector<compute_kernel::material> materials;
};

/** @brief Traces paths of a compiled scene on a device, accumulating a film there. Not thread safe. */
class backend_renderer_t {
public:
    virtual ~backend_renderer_t() noexcept = default;

    /**
     * @brief Starts paths `first .. first+count-1`, whose spectral samples
     *        are in `spectra`, and traces them, and paths in flight from
     *        earlier calls; may return with some still in flight, for the next
     *        call or `finish`. A path's index and `seed` determine its random
     *        numbers and spectral sample. Returns the status flags of all
     *        paths so far (compute_kernel::status_*).
     */
    virtual std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed,
                                const spectral_table_t& spectra) = 0;

    /** @brief Completes all paths in flight. Returns the status flags of all paths so far. */
    virtual std::uint32_t finish() = 0;

    /**
     * @brief Copies the film to `values`: element (x,y) component c at
     *        `values[(y*width + x)*components + c]` (of a polarimetric film,
     *        its Stokes parameters at `values[((y*width + x)*components + c)*4 + i]`);
     *        and, of block splats (backward plt_path, plt_bdpt), the sums of
     *        the reconstruction weights splatted with them to `weights` (one
     *        per component; nullptr skips them). Paths in flight have not
     *        contributed all of their splats yet.
     */
    virtual void read_film(double* values, double* weights) = 0;

    /** @brief plt_bdpt: copies the film's light image, of direct splats, to `values` (as `read_film`). */
    virtual void read_light_film(double* values) = 0;
};

/**
 * @brief Scene geometry on a device. Queries are batched and synchronous;
 *        results are in query order. Not thread safe.
 */
class backend_geometry_t {
public:
    virtual ~backend_geometry_t() noexcept = default;

    virtual void intersect_rays(const ray_query_t* queries, std::size_t count,
                                compute_kernel::ray_hit* hits) = 0;
    /** @brief `occluded[i]` is 1 if something blocks ray `i`. */
    virtual void rays_occluded(const ray_query_t* queries, std::size_t count,
                               std::uint8_t* occluded, std::uint32_t* status) = 0;

    /**
     * @brief Query `i` writes up to `capacity` triangle indices to
     *        `tris[i*capacity ...]`; `hits[i].tri_count` counts all it found.
     */
    virtual void intersect_cones(const cone_query_t* queries, std::size_t count,
                                 std::uint32_t capacity,
                                 compute_kernel::cone_hit* hits, std::uint32_t* tris) = 0;
    /** @brief As `intersect_cones`. */
    virtual void intersect_balls(const compute_kernel::ball* queries, std::size_t count,
                                 std::uint32_t capacity,
                                 ball_hit_t* hits, std::uint32_t* tris) = 0;

    /**
     * @brief Uploads a scene compiled for this geometry, for rendering with
     *        plt_path or plt_bdpt. (By value: a backend may keep it, for the
     *        paths it traces on the CPU.)
     */
    [[nodiscard]] virtual std::unique_ptr<backend_renderer_t> create_plt_renderer(plt_scene_t scene) = 0;
};

/** @brief A GPU backend on one device. */
class backend_t {
public:
    virtual ~backend_t() noexcept = default;

    /** @brief e.g. "CUDA (OptiX 9.0) on NVIDIA GB10". */
    [[nodiscard]] virtual std::string description() const = 0;

    /** @brief Uploads a BVH. (By value: a backend may keep it, for the paths it traces on the CPU.) */
    [[nodiscard]] virtual std::unique_ptr<backend_geometry_t> upload(bvh_arrays_t bvh) = 0;
};

#if defined(WT_GPU_CUDA)
/**
 * @brief Creates the CUDA backend on CUDA device `ordinal`. With `use_optix`
 *        (requires WT_GPU_OPTIX), ray queries use OptiX's hardware traversal.
 *        Throws std::runtime_error if the device or OptiX is unavailable.
 */
[[nodiscard]] std::unique_ptr<backend_t> create_cuda_backend(int ordinal, bool use_optix);
#endif

#if defined(WT_GPU_METAL)
/**
 * @brief Creates the Metal backend on Metal device `ordinal` (Apple M1 or later).
 *        With `use_hardware_ray_tracing`, ray queries use Metal's ray tracing on
 *        GPUs that support it. Throws std::runtime_error if the device is
 *        unavailable or unsupported.
 */
[[nodiscard]] std::unique_ptr<backend_t> create_metal_backend(int ordinal, bool use_hardware_ray_tracing);
#endif

}
