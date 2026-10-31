/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * The Metal backend (Apple silicon): the shared kernels compiled as Metal
 * Shading Language into an embedded metallib, run through Metal compute
 * pipelines. Buffers live in unified memory shared with the CPU. On GPUs with
 * ray-tracing hardware (Apple M3 and later), ray queries, of the geometry and
 * of the path kernels, use Metal's ray tracing (rt_rays.hpp), as the CUDA
 * backend's use OptiX.
 */

#include <wt/gpu/backend.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "layout.hpp"
#include "metal_renderer.hpp"
#include "metal_util.hpp"

// The metallib of src/gpu/metal/*.metal, embedded by cmake/wt_gpu.cmake.
extern const unsigned char wt_metal_kernels[];
extern const std::size_t wt_metal_kernels_size;

using namespace wt::gpu;
using namespace wt::gpu::metal_backend;
namespace k = wt::gpu::compute_kernel;

namespace {

/** @brief Fails unless the host and the Metal compiler lay out the shared structs alike. */
void check_layouts(context_t& context) {
    buffer_t<k::u32> gpu(context.device);
    gpu.fill_zero(layout_values);
    context.dispatch("struct_layouts", 1, [&](id<MTLComputeCommandEncoder> e) { gpu.bind(e, 0); });

    std::vector<k::u32> host(layout_values);
    write_layouts(host.data());
    std::vector<k::u32> device(layout_values);
    gpu.download(device.data(), layout_values);

    static const char* const names[] = {
#define WT_METAL_STRUCT_NAME(T) #T,
        WT_METAL_SHARED_STRUCTS(WT_METAL_STRUCT_NAME)
#undef WT_METAL_STRUCT_NAME
    };
    for (std::size_t i=0; i<host.size(); ++i) {
        if (host[i]!=device[i])
            throw std::runtime_error(std::string("the Metal kernels lay out struct ") + names[i/2] +
                                     " differently than the host (" + (i%2 ? "alignment " : "size ") +
                                     std::to_string(device[i]) + " vs " + std::to_string(host[i]) + ")");
    }
}


/**
 * @brief Builds the acceleration structure of Metal's ray tracing of `bvh`'s
 *        triangles, in the BVH's order: a hit's primitive is its BVH triangle.
 */
id<MTLAccelerationStructure> build_acceleration_structure(context_t& context, const bvh_arrays_t& bvh) {
    // one vertex triple per triangle
    std::vector<float> vertices;
    vertices.reserve(bvh.tris.size()*9);
    for (const auto& t : bvh.tris)
        for (const auto& v : { t.a, t.b, t.c })
            vertices.insert(vertices.end(), { v.x, v.y, v.z });
    buffer_t<float> vertex_buffer(context.device);
    vertex_buffer.upload(vertices.data(), vertices.size());

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = vertex_buffer.get();
    geometry.vertexFormat = MTLAttributeFormatFloat3;
    geometry.vertexStride = 3*sizeof(float);
    geometry.triangleCount = bvh.tris.size();
    geometry.opaque = YES;
    MTLPrimitiveAccelerationStructureDescriptor* descriptor = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[ geometry ];

    const MTLAccelerationStructureSizes sizes = [context.device accelerationStructureSizesWithDescriptor:descriptor];
    id<MTLAccelerationStructure> accel = [context.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [context.device newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 1)
                                                        options:MTLResourceStorageModePrivate];
    if (!accel || !scratch)
        throw std::runtime_error("cannot allocate the ray-tracing acceleration structure");
    @autoreleasepool {
        id<MTLCommandBuffer> commands = [context.queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> encoder = [commands accelerationStructureCommandEncoder];
        [encoder buildAccelerationStructure:accel descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status==MTLCommandBufferStatusError)
            throw std::runtime_error("cannot build the ray-tracing acceleration structure: " +
                                     to_string(commands.error.localizedDescription));
    }
    return accel;
}

/** @brief Metal's ray tracing traces forward only: ranges starting below 0 need the BVH traversal. */
bool hardware_can_trace(const ray_query_t* queries, std::size_t count) {
    for (std::size_t i=0; i<count; ++i)
        if (!(queries[i].range.min>=0))
            return false;
    return true;
}


class metal_geometry_t final : public backend_geometry_t {
public:
    metal_geometry_t(std::shared_ptr<context_t> context, const bvh_arrays_t& bvh, bool hardware_rays)
        : context(std::move(context)),
          nodes(this->context->device), tris(this->context->device),
          ray_queries(this->context->device), ray_hits(this->context->device),
          flags(this->context->device), status_word(this->context->device),
          cone_queries(this->context->device), cone_hits(this->context->device),
          ball_queries(this->context->device), ball_hits(this->context->device),
          tri_lists(this->context->device)
    {
        nodes.upload(bvh.nodes.data(), bvh.nodes.size());
        tris.upload(bvh.tris.data(), bvh.tris.size());
        tri_count = bvh.tris.size();
        if (hardware_rays && !bvh.tris.empty())
            accel = build_acceleration_structure(*this->context, bvh);
    }

    void intersect_rays(const ray_query_t* queries, std::size_t count, k::ray_hit* hits) override {
        if (count==0) return;
        ray_queries.upload(queries, count);
        ray_hits.resize(count);
        const auto n = k::u32(count);
        const bool hardware = accel && hardware_can_trace(queries, count);
        context->dispatch(hardware ? "rt_intersect_rays" : "intersect_rays", count,
                          [&](id<MTLComputeCommandEncoder> e) {
            bind_bvh(e, hardware);
            ray_queries.bind(e, 2);
            ray_hits.bind(e, 3);
            bind_value(e, n, 4);
        });
        ray_hits.download(hits, count);
    }

    void rays_occluded(const ray_query_t* queries, std::size_t count,
                       std::uint8_t* occluded, std::uint32_t* status) override {
        if (count==0) return;
        ray_queries.upload(queries, count);
        flags.resize(count);
        status_word.upload(status, 1);
        const auto n = k::u32(count);
        const bool hardware = accel && hardware_can_trace(queries, count);
        context->dispatch(hardware ? "rt_rays_occluded" : "rays_occluded", count,
                          [&](id<MTLComputeCommandEncoder> e) {
            bind_bvh(e, hardware);
            ray_queries.bind(e, 2);
            flags.bind(e, 3);
            bind_value(e, n, 4);
            status_word.bind(e, 5);
        });
        flags.download(occluded, count);
        status_word.download(status, 1);
    }

    void intersect_cones(const cone_query_t* queries, std::size_t count, std::uint32_t capacity,
                         k::cone_hit* hits, std::uint32_t* out_tris) override {
        if (count==0) return;
        cone_queries.upload(queries, count);
        cone_hits.resize(count);
        tri_lists.resize(count*capacity);
        const auto n = k::u32(count);
        context->dispatch("intersect_cones", count, [&](id<MTLComputeCommandEncoder> e) {
            bind_bvh(e);
            cone_queries.bind(e, 2);
            cone_hits.bind(e, 3);
            bind_value(e, n, 4);
            bind_value(e, capacity, 5);
            tri_lists.bind(e, 6);
        });
        cone_hits.download(hits, count);
        tri_lists.download(out_tris, count*capacity);
    }

    void intersect_balls(const k::ball* queries, std::size_t count, std::uint32_t capacity,
                         ball_hit_t* hits, std::uint32_t* out_tris) override {
        if (count==0) return;
        ball_queries.upload(queries, count);
        ball_hits.resize(count);
        tri_lists.resize(count*capacity);
        const auto n = k::u32(count);
        context->dispatch("intersect_balls", count, [&](id<MTLComputeCommandEncoder> e) {
            bind_bvh(e);
            ball_queries.bind(e, 2);
            ball_hits.bind(e, 3);
            bind_value(e, n, 4);
            bind_value(e, capacity, 5);
            tri_lists.bind(e, 6);
        });
        ball_hits.download(hits, count);
        tri_lists.download(out_tris, count*capacity);
    }

    [[nodiscard]] std::unique_ptr<backend_renderer_t> create_plt_renderer(plt_scene_t scene) override {
        if (scene.tris.size()!=tri_count)
            throw std::runtime_error("the compiled scene does not match the uploaded geometry");
        return metal_backend::create_plt_renderer(context, { nodes.get(), tris.get(), accel }, scene);
    }

private:
    /** @brief Binds the BVH, or, for the hardware's ray queries, the acceleration structure and the triangles. */
    void bind_bvh(id<MTLComputeCommandEncoder> e, bool hardware = false) const {
        if (hardware)
            [e setAccelerationStructure:accel atBufferIndex:0];
        else
            nodes.bind(e, 0);
        tris.bind(e, 1);
    }

    std::shared_ptr<context_t> context;

    buffer_t<k::bvh_node> nodes;
    buffer_t<k::bvh_tri>  tris;
    std::size_t tri_count = 0;
    id<MTLAccelerationStructure> accel = nil;   // of the hardware's ray tracing, if used

    // per-call buffers, reused across calls
    buffer_t<ray_query_t>   ray_queries;
    buffer_t<k::ray_hit>    ray_hits;
    buffer_t<std::uint8_t>  flags;
    buffer_t<std::uint32_t> status_word;
    buffer_t<cone_query_t>  cone_queries;
    buffer_t<k::cone_hit>   cone_hits;
    buffer_t<k::ball>       ball_queries;
    buffer_t<ball_hit_t>    ball_hits;
    buffer_t<std::uint32_t> tri_lists;
};


class metal_backend_t final : public backend_t {
public:
    metal_backend_t(int ordinal, bool use_hardware_ray_tracing) {
        id<MTLDevice> device = nil;
        @autoreleasepool {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if (devices.count==0)
                throw std::runtime_error("no Metal device found");
            if (ordinal<0 || NSUInteger(ordinal)>=devices.count)
                throw std::runtime_error("Metal device " + std::to_string(ordinal) + " does not exist (" +
                                         std::to_string(devices.count) + " found)");
            device = devices[NSUInteger(ordinal)];
        }
        // atomic_float (films) and SIMD-group functions: Apple7 (M1) and later
        if (![device supportsFamily:MTLGPUFamilyApple7])
            throw std::runtime_error(to_string(device.name) +
                                     " is not supported: the Metal backend needs an Apple M1 or later GPU");

        NSError* error = nil;
        dispatch_data_t data = dispatch_data_create(wt_metal_kernels, wt_metal_kernels_size, nullptr,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
        if (!library)
            throw std::runtime_error("cannot load the Metal kernels: " + to_string(error.localizedDescription));

        context = std::make_shared<context_t>(device, library);
        check_layouts(*context);

        // ray-tracing hardware: Apple9 (M3) and later; earlier GPUs emulate Metal's ray tracing in software
        hardware_rays = use_hardware_ray_tracing && [device supportsFamily:MTLGPUFamilyApple9];
    }

    [[nodiscard]] std::string description() const override {
        return std::string("Metal") + (hardware_rays ? " + hardware ray tracing" : "") + " on " +
               to_string(context->device.name);
    }

    [[nodiscard]] std::unique_ptr<backend_geometry_t> upload(bvh_arrays_t bvh) override {
        return std::make_unique<metal_geometry_t>(context, bvh, hardware_rays);
    }

private:
    std::shared_ptr<context_t> context;
    bool hardware_rays = false;
};

}

std::unique_ptr<backend_t> wt::gpu::create_metal_backend(int ordinal, bool use_hardware_ray_tracing) {
    return std::make_unique<metal_backend_t>(ordinal, use_hardware_ray_tracing);
}
