/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * CUDA backend: runs the shared kernels (wt/gpu/kernel/) on an NVIDIA GPU.
 *
 * Compiled by nvcc, so it includes only wt/gpu/backend.hpp, the kernel headers
 * and CUDA; never other wave_tracer headers (nvcc cannot parse mp-units).
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include <wt/gpu/backend.hpp>
#include <wt/gpu/kernel/plt_path.hpp>
#include "cuda_util.hpp"

#if defined(WT_GPU_OPTIX) && !defined(WT_GPU_DBL_PRECISION)
#include "../optix/optix_rays.hpp"
#define WT_GPU_USE_OPTIX 1
#endif

using namespace wt::gpu;
using namespace wt::gpu::cuda;
namespace k = wt::gpu::kernel;

namespace {

constexpr unsigned block_size = 128;

unsigned grid_size(std::size_t count) { return unsigned((count+block_size-1)/block_size); }

#if defined(WT_GPU_USE_OPTIX)
/** @brief OptiX traces rays forward only: ranges starting below 0 need the CUDA traversal. */
bool optix_can_trace(const ray_query_t* queries, std::size_t count) {
    for (std::size_t i=0; i<count; ++i)
        if (!(queries[i].range.min>=0))
            return false;
    return true;
}
#endif

__global__ void intersect_rays_kernel(k::bvh_view bvh, const ray_query_t* queries, std::size_t count,
                                      k::ray_hit* hits) {
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;
    hits[i] = k::intersect_ray(bvh, queries[i].ray, queries[i].range);
}

__global__ void rays_occluded_kernel(k::bvh_view bvh, const ray_query_t* queries, std::size_t count,
                                     std::uint8_t* occluded, std::uint32_t* status) {
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;
    k::u32 s = k::status_ok;
    occluded[i] = k::ray_occluded(bvh, queries[i].ray, queries[i].range, s) ? 1 : 0;
    if (s!=k::status_ok)
        atomicOr(status, s);
}

__global__ void intersect_cones_kernel(k::bvh_view bvh, const cone_query_t* queries, std::size_t count,
                                       std::uint32_t capacity, k::cone_hit* hits, std::uint32_t* tris) {
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;
    k::tri_sink sink = { tris + i*capacity, capacity, 0 };
    hits[i] = k::intersect_cone(bvh, queries[i].cone, queries[i].range, queries[i].z_scale, sink);
}

__global__ void intersect_balls_kernel(k::bvh_view bvh, const k::ball* queries, std::size_t count,
                                       std::uint32_t capacity, ball_hit_t* hits, std::uint32_t* tris) {
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;
    k::tri_sink sink = { tris + i*capacity, capacity, 0 };
    const auto status = k::intersect_ball(bvh, queries[i], sink);
    hits[i] = { sink.count, status };
}


/*
 * Forward plt_path
 */

/** @brief Geometry queries for the path tracer: the CUDA BVH traversals. */
struct cuda_bvh_geometry_t {
    k::bvh_view bvh;

    __device__ k::ray_hit intersect(const k::ray& r, k::range rng) const {
        return k::intersect_ray(bvh, r, rng);
    }
    __device__ bool occluded(const k::ray& r, k::range rng, k::u32& status) const {
        return k::ray_occluded(bvh, r, rng, status);
    }
    __device__ k::cone_hit intersect_cone(const k::cone& c, k::range rng, k::real z_scale,
                                          k::tri_sink& sink) const {
        return k::intersect_cone(bvh, c, rng, z_scale, sink);
    }
};

/** @brief The film, accumulated in double precision. */
struct cuda_film_t {
    double* data;

    __device__ void add(k::u32 i, k::real v) {
        atomicAdd(data+i, double(v));
    }
};

/** @brief A path to (re-)trace, skipping the splats an earlier attempt made. */
struct path_entry_t {
    std::uint64_t path;
    std::uint32_t skip;
};

struct scratch_view_t {
    k::u32* tris;
    k::u32* edges;
    k::aperture_entry* aperture;
    k::u32 tri_capacity, edge_capacity;
};

/**
 * @brief Traces paths `first+i`, or `list[i]` if `list` is given. Paths that
 *        outgrow their scratch memory are appended to `overflow`.
 *        Three resident blocks per SM cap registers at 168 per thread (from
 *        about 250): more warps hide more memory latency (18% faster on a GB10).
 */
__global__ void __launch_bounds__(block_size, 3) plt_path_forward_kernel(k::plt_scene scene,
                                        std::uint64_t first, std::uint64_t count, const path_entry_t* list,
                                        std::uint32_t seed, double* film, scratch_view_t scratch,
                                        path_entry_t* overflow, std::uint32_t* overflow_count,
                                        std::uint32_t* status) {
    const std::uint64_t i = std::uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const std::uint64_t path = list ? list[i].path : first+i;
    k::rng rnd = k::rng::for_path(seed, k::u32(path), k::u32(path>>32));
    const cuda_bvh_geometry_t geometry = { scene.bvh };
    k::path_film<cuda_film_t> pf = { { film }, list ? list[i].skip : 0, 0 };
    const k::path_scratch ps = {
        scratch.tris + i*scratch.tri_capacity,
        scratch.edges + i*scratch.edge_capacity,
        scratch.aperture + i*scratch.edge_capacity,
        scratch.tri_capacity, scratch.edge_capacity,
    };

    k::u32 s = k::trace_forward_path(scene, geometry, pf, ps, rnd);
    if (s & k::status_scratch_overflow) {
        overflow[atomicAdd(overflow_count, 1u)] = { path, pf.splats };
        s &= ~k::status_scratch_overflow;
    }
    if (s!=k::status_ok)
        atomicOr(status, s);
}

class cuda_plt_path_renderer_t final : public backend_renderer_t {
public:
    cuda_plt_path_renderer_t(int ordinal, k::bvh_view bvh, const plt_path_scene_t& scene)
        : ordinal(ordinal)
    {
        set_device(ordinal);
        tris.upload(scene.tris.data(), scene.tris.size());
        edges.upload(scene.edges.data(), scene.edges.size());
        materials.upload(scene.materials.data(), scene.materials.size());
        emitters.upload(scene.emitters.data(), scene.emitters.size());
        emitter_cdf.upload(scene.emitter_cdf.data(), scene.emitter_cdf.size());

        film_size = std::size_t(scene.film.width) * scene.film.height * scene.film.components;
        film.resize(film_size);
        check(cudaMemset(film.get(), 0, film_size*sizeof(double)), "clearing the film");
        status_word.resize(1);
        overflow_count.resize(1);

        params.bvh = bvh;
        params.tris = tris.get();
        params.edges = edges.get();
        params.materials = materials.get();
        params.emitters = emitters.get();
        params.emitter_cdf = emitter_cdf.get();
        params.emitter_count = k::u32(scene.emitters.size());
        params.sensor = scene.sensor;
        params.film = scene.film;
        params.params = scene.params;
    }

    std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed) override {
        set_device(ordinal);
        std::uint32_t status = k::status_ok;
        status_word.upload(&status, 1);

        // paths that outgrew their scratch memory are re-traced with more room
        std::vector<path_entry_t> pending, overflowed;
        k::u32 tri_capacity = initial_capacity, edge_capacity = initial_capacity;
        for (std::uint64_t start=0; start<count; start+=launch_size) {
            const auto n = std::min<std::uint64_t>(launch_size, count-start);
            launch(first+start, n, nullptr, seed, tri_capacity, edge_capacity, pending);
        }
        while (!pending.empty()) {
            if (tri_capacity>=max_capacity)
                throw std::runtime_error("a path found more than " + std::to_string(max_capacity) +
                                         " triangles or edges in one query");
            tri_capacity *= 4;
            edge_capacity *= 4;

            // as many paths per launch as fit the scratch memory the first launches used
            const std::size_t batch = std::max<std::size_t>(1, launch_size*initial_capacity/tri_capacity);
            overflowed.clear();
            for (std::size_t start=0; start<pending.size(); start+=batch) {
                const auto n = std::min(batch, pending.size()-start);
                path_list.upload(pending.data()+start, n);
                launch(0, n, path_list.get(), seed, tri_capacity, edge_capacity, overflowed);
            }
            std::swap(pending, overflowed);
        }

        status_word.download(&status, 1);
        return status;
    }

    void read_film(double* out) override {
        set_device(ordinal);
        film.download(out, film_size);
    }

private:
    /** @brief Paths per kernel launch. */
    static constexpr std::uint64_t launch_size = std::uint64_t(1)<<18;
    /** @brief Triangles and edges a path holds at first; paths needing more are re-traced. */
    static constexpr k::u32 initial_capacity = 64;
    static constexpr k::u32 max_capacity = 1u<<20;

    /** @brief Launches the kernel for `count` paths, appending those that overflowed to `overflowed`. */
    void launch(std::uint64_t first, std::uint64_t count, const path_entry_t* list, std::uint32_t seed,
                k::u32 tri_capacity, k::u32 edge_capacity, std::vector<path_entry_t>& overflowed) {
        scratch_tris.resize(count*tri_capacity);
        scratch_edges.resize(count*edge_capacity);
        scratch_aperture.resize(count*edge_capacity);
        overflow.resize(count);
        const std::uint32_t zero = 0;
        overflow_count.upload(&zero, 1);

        const scratch_view_t scratch = {
            scratch_tris.get(), scratch_edges.get(), scratch_aperture.get(),
            tri_capacity, edge_capacity,
        };
        plt_path_forward_kernel<<<grid_size(count), block_size>>>(
                params, first, count, list, seed, film.get(), scratch,
                overflow.get(), overflow_count.get(), status_word.get());
        check(cudaGetLastError(), "launching the path tracing kernel");

        std::uint32_t n;
        overflow_count.download(&n, 1);
        const auto offset = overflowed.size();
        overflowed.resize(offset+n);
        overflow.download(overflowed.data()+offset, n);
    }

    int ordinal;
    k::plt_scene params;

    device_buffer_t<k::tri_shading>   tris;
    device_buffer_t<k::edge>          edges;
    device_buffer_t<k::material>      materials;
    device_buffer_t<k::point_emitter> emitters;
    device_buffer_t<k::real>          emitter_cdf;
    device_buffer_t<double>           film;
    std::size_t film_size = 0;
    device_buffer_t<std::uint32_t>    status_word;

    device_buffer_t<k::u32>            scratch_tris;
    device_buffer_t<k::u32>            scratch_edges;
    device_buffer_t<k::aperture_entry> scratch_aperture;
    device_buffer_t<path_entry_t>      overflow;
    device_buffer_t<std::uint32_t>     overflow_count;
    device_buffer_t<path_entry_t>      path_list;
};


class cuda_geometry_t final : public backend_geometry_t {
public:
    cuda_geometry_t(int ordinal, const bvh_arrays_t& bvh
#if defined(WT_GPU_USE_OPTIX)
                    , const optix::context_t* optix
#endif
                    )
        : ordinal(ordinal)
    {
        set_device(ordinal);
        nodes.upload(bvh.nodes.data(), bvh.nodes.size());
        leaves.upload(bvh.leaves.data(), bvh.leaves.size());
        tris.upload(bvh.tris.data(), bvh.tris.size());
        tri_count = bvh.tris.size();
#if defined(WT_GPU_USE_OPTIX)
        if (optix && !bvh.tris.empty())
            optix_rays = std::make_unique<optix::ray_geometry_t>(*optix, bvh, tris.get());
#endif
    }

    void intersect_rays(const ray_query_t* queries, std::size_t count, k::ray_hit* hits) override {
        if (count==0) return;
        set_device(ordinal);
        ray_queries.upload(queries, count);
        ray_hits.resize(count);
#if defined(WT_GPU_USE_OPTIX)
        if (optix_rays && optix_can_trace(queries, count))
            optix_rays->intersect(ray_queries.get(), count, ray_hits.get());
        else
#endif
        {
            intersect_rays_kernel<<<grid_size(count), block_size>>>(view(), ray_queries.get(), count,
                                                                   ray_hits.get());
            check(cudaGetLastError(), "launching the ray kernel");
        }
        ray_hits.download(hits, count);
    }

    void rays_occluded(const ray_query_t* queries, std::size_t count,
                       std::uint8_t* occluded, std::uint32_t* status) override {
        if (count==0) return;
        set_device(ordinal);
        ray_queries.upload(queries, count);
        flags.resize(count);
        status_word.upload(status, 1);
#if defined(WT_GPU_USE_OPTIX)
        if (optix_rays && optix_can_trace(queries, count))
            optix_rays->occluded(ray_queries.get(), count, flags.get());
        else
#endif
        {
            rays_occluded_kernel<<<grid_size(count), block_size>>>(view(), ray_queries.get(), count,
                                                                  flags.get(), status_word.get());
            check(cudaGetLastError(), "launching the shadow-ray kernel");
        }
        flags.download(occluded, count);
        status_word.download(status, 1);
    }

    void intersect_cones(const cone_query_t* queries, std::size_t count, std::uint32_t capacity,
                         k::cone_hit* hits, std::uint32_t* out_tris) override {
        if (count==0) return;
        set_device(ordinal);
        cone_queries.upload(queries, count);
        cone_hits.resize(count);
        tri_lists.resize(count*capacity);
        intersect_cones_kernel<<<grid_size(count), block_size>>>(view(), cone_queries.get(), count, capacity,
                                                                cone_hits.get(), tri_lists.get());
        check(cudaGetLastError(), "launching the cone kernel");
        cone_hits.download(hits, count);
        tri_lists.download(out_tris, count*capacity);
    }

    void intersect_balls(const k::ball* queries, std::size_t count, std::uint32_t capacity,
                         ball_hit_t* hits, std::uint32_t* out_tris) override {
        if (count==0) return;
        set_device(ordinal);
        ball_queries.upload(queries, count);
        ball_hits.resize(count);
        tri_lists.resize(count*capacity);
        intersect_balls_kernel<<<grid_size(count), block_size>>>(view(), ball_queries.get(), count, capacity,
                                                                ball_hits.get(), tri_lists.get());
        check(cudaGetLastError(), "launching the ball kernel");
        ball_hits.download(hits, count);
        tri_lists.download(out_tris, count*capacity);
    }

    [[nodiscard]] std::unique_ptr<backend_renderer_t> create_plt_path_renderer(
            const plt_path_scene_t& scene) override {
        if (scene.tris.size()!=tri_count)
            throw std::runtime_error("the compiled scene does not match the uploaded geometry");
        return std::make_unique<cuda_plt_path_renderer_t>(ordinal, view(), scene);
    }

private:
    [[nodiscard]] k::bvh_view view() const noexcept {
        return { nodes.get(), leaves.get(), tris.get() };
    }

    int ordinal;

    device_buffer_t<k::bvh_node> nodes;
    device_buffer_t<k::bvh_leaf> leaves;
    device_buffer_t<k::bvh_tri>  tris;
    std::size_t tri_count = 0;

    // per-call buffers, reused across calls
    device_buffer_t<ray_query_t>   ray_queries;
    device_buffer_t<k::ray_hit>    ray_hits;
    device_buffer_t<std::uint8_t>  flags;
    device_buffer_t<std::uint32_t> status_word;
    device_buffer_t<cone_query_t>  cone_queries;
    device_buffer_t<k::cone_hit>   cone_hits;
    device_buffer_t<k::ball>       ball_queries;
    device_buffer_t<ball_hit_t>    ball_hits;
    device_buffer_t<std::uint32_t> tri_lists;

#if defined(WT_GPU_USE_OPTIX)
    std::unique_ptr<optix::ray_geometry_t> optix_rays;
#endif
};


class cuda_backend_t final : public backend_t {
public:
    cuda_backend_t(int ordinal, bool use_optix) : ordinal(ordinal) {
        int count = 0;
        const auto err = cudaGetDeviceCount(&count);
        if (err!=cudaSuccess)
            throw std::runtime_error(std::string("no usable CUDA device (") + cudaGetErrorString(err) + ")");
        if (ordinal<0 || ordinal>=count)
            throw std::runtime_error("CUDA device " + std::to_string(ordinal) + " does not exist (" +
                                     std::to_string(count) + " found)");
        set_device(ordinal);
        check(cudaGetDeviceProperties(&properties, ordinal), "querying the CUDA device");

#if defined(WT_GPU_USE_OPTIX)
        if (use_optix)
            optix_context = std::make_unique<optix::context_t>(ordinal);
#else
        (void)use_optix;
#endif
    }

    [[nodiscard]] std::string description() const override {
        std::string api = "CUDA";
#if defined(WT_GPU_USE_OPTIX)
        if (optix_context)
            api += " + OptiX " + optix_context->version();
#endif
#if defined(WT_GPU_DBL_PRECISION)
        api += ", 64-bit";
#endif
        return api + " on " + properties.name + " (compute capability " +
               std::to_string(properties.major) + "." + std::to_string(properties.minor) + ")";
    }

    [[nodiscard]] std::unique_ptr<backend_geometry_t> upload(const bvh_arrays_t& bvh) override {
#if defined(WT_GPU_USE_OPTIX)
        return std::make_unique<cuda_geometry_t>(ordinal, bvh, optix_context.get());
#else
        return std::make_unique<cuda_geometry_t>(ordinal, bvh);
#endif
    }

private:
    int ordinal;
    cudaDeviceProp properties{};
#if defined(WT_GPU_USE_OPTIX)
    std::unique_ptr<optix::context_t> optix_context;
#endif
};

}

std::unique_ptr<backend_t> wt::gpu::create_cuda_backend(int ordinal, bool use_optix) {
    return std::make_unique<cuda_backend_t>(ordinal, use_optix);
}
