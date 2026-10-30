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
 * @brief A scene compiled for plt_path or plt_bdpt rendering of one sensor
 *        (see wt/gpu/kernel/scene.hpp), but for what depends on the
 *        wavenumber: see `spectral_table_t`. The device pointers of
 *        `kernel::plt_scene` are filled in by the backend.
 */
struct plt_scene_t {
    std::vector<kernel::tri_shading> tris;     // per BVH triangle
    std::vector<kernel::edge> edges;
    std::vector<kernel::emitter> emitters;
    std::vector<kernel::real> emitter_cdf;
    std::vector<std::uint32_t> area_tris;      // area emitters' triangles (see `kernel::plt_scene`)
    std::vector<kernel::real> area_cdf;
    std::vector<kernel::tri_uv> tri_uvs;       // per BVH triangle, if any material has textures
    std::vector<kernel::material_shading> material_shadings;   // per material
    std::vector<kernel::texture> textures;
    std::vector<kernel::bitmap> bitmaps;
    std::vector<std::uint32_t> texels;         // `kernel::bitmap`s' texels
    std::vector<kernel::real> texel_luts;
    std::vector<kernel::real> fsd_lut;         // plt_bdpt's free-space diffraction, or empty (see `kernel::plt_scene`)
    std::uint32_t emitter_count;
    std::uint32_t material_count;
    std::uint32_t samples_per_emitter;         // in each spectral table
    kernel::sensor_desc sensor;
    kernel::film_desc film;
    kernel::plt_params params;
};

/**
 * @brief Spectral samples (see `kernel::plt_scene`): `samples[s]`, with
 *        emitter e at `emissions[s*emitter_count + e]` and material m at
 *        `materials[s*material_count + m]`. A table for `trace` holds
 *        `samples_per_emitter` samples of each emitter, emitter by emitter.
 */
struct spectral_table_t {
    std::vector<kernel::spectral_sample> samples;
    std::vector<kernel::emission> emissions;
    std::vector<kernel::material> materials;
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
     *        paths so far (kernel::status_*).
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

    /** @brief Uploads a scene compiled for this geometry, for rendering with plt_path or plt_bdpt. */
    [[nodiscard]] virtual std::unique_ptr<backend_renderer_t> create_plt_renderer(const plt_scene_t& scene) = 0;
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
