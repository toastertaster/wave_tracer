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

#include <stdexcept>
#include <string>

#include <wt/gpu/backend.hpp>
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

private:
    [[nodiscard]] k::bvh_view view() const noexcept {
        return { nodes.get(), leaves.get(), tris.get() };
    }

    int ordinal;

    device_buffer_t<k::bvh_node> nodes;
    device_buffer_t<k::bvh_leaf> leaves;
    device_buffer_t<k::bvh_tri>  tris;

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
