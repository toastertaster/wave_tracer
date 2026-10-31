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
 * Metal backend: device, kernel pipelines, buffers and dispatch. Objective-C++
 * (compiled with ARC); included only by the backend's .mm sources.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace wt::gpu::metal_backend {

inline std::string to_string(NSString* s) { return s ? std::string(s.UTF8String) : std::string{}; }

/** @brief The GPU, its command queue, and the backend's kernels (the embedded metallib). */
class context_t {
public:
    context_t(id<MTLDevice> device, id<MTLLibrary> library)
        : device(device), library(library), queue([device newCommandQueue]) {
        if (!queue)
            throw std::runtime_error("cannot create a Metal command queue");
    }

    /** @brief The compute pipeline of kernel `name`, created on first use. Not thread safe. */
    id<MTLComputePipelineState> pipeline(const char* name) {
        const auto it = pipelines.find(name);
        if (it!=pipelines.end())
            return it->second;
        id<MTLFunction> function = [library newFunctionWithName:@(name)];
        if (!function)
            throw std::runtime_error(std::string("Metal kernel ") + name + " not found");
        NSError* error = nil;
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state)
            throw std::runtime_error(std::string("cannot create the Metal pipeline of ") + name + ": " +
                                     to_string(error.localizedDescription));
        pipelines.emplace(name, state);
        return state;
    }

    /**
     * @brief Runs kernel `name` on `count` threads (one dimension) and waits for
     *        it; `bind` sets its arguments on the encoder. Throws if the GPU
     *        reports an error.
     */
    void dispatch(const char* name, std::size_t count,
                  const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
        if (count==0) return;
        id<MTLComputePipelineState> state = pipeline(name);
        @autoreleasepool {
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:state];
            bind(encoder);
            const NSUInteger width = std::min<NSUInteger>(state.maxTotalThreadsPerThreadgroup, 128);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status==MTLCommandBufferStatusError)
                throw std::runtime_error(std::string("Metal kernel ") + name + " failed: " +
                                         to_string(commands.error.localizedDescription));
        }
    }

    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;

private:
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
};

/**
 * @brief A buffer of `T`s in memory shared by the CPU and the GPU (Apple
 *        silicon's unified memory), so uploads and downloads are copies in
 *        place. Grows on demand and never shrinks; contents are not kept when
 *        it grows. Synchronize with the GPU (dispatch waits) before reading.
 */
template <typename T>
class buffer_t {
public:
    explicit buffer_t(id<MTLDevice> device) : device(device) {}

    void resize(std::size_t count) {
        if (count<=capacity && buffer) return;
        // Metal cannot create empty buffers
        const std::size_t bytes = std::max<std::size_t>(count, 1)*sizeof(T);
        buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer)
            throw std::runtime_error("cannot allocate " + std::to_string(bytes) + " bytes of GPU memory");
        capacity = std::max<std::size_t>(count, 1);
    }
    void upload(const T* data, std::size_t count) {
        resize(count);
        if (count)
            std::memcpy(buffer.contents, data, count*sizeof(T));
    }
    void download(T* data, std::size_t count) const {
        if (count)
            std::memcpy(data, buffer.contents, count*sizeof(T));
    }
    void fill_zero(std::size_t count) {
        resize(count);
        std::memset(buffer.contents, 0, count*sizeof(T));
    }

    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(buffer.contents); }
    [[nodiscard]] id<MTLBuffer> get() const noexcept { return buffer; }
    [[nodiscard]] std::size_t size() const noexcept { return capacity; }
    /** @brief The buffer's GPU address, for pointers stored in structs. */
    [[nodiscard]] std::uint64_t gpu_address() const noexcept { return buffer ? buffer.gpuAddress : 0; }

    void bind(id<MTLComputeCommandEncoder> encoder, NSUInteger index) const {
        [encoder setBuffer:buffer offset:0 atIndex:index];
    }

private:
    id<MTLDevice> device;
    id<MTLBuffer> buffer = nil;
    std::size_t capacity = 0;
};

/** @brief Sets a small value as a kernel argument (`constant T&`). */
template <typename T>
void bind_value(id<MTLComputeCommandEncoder> encoder, const T& value, NSUInteger index) {
    [encoder setBytes:&value length:sizeof(T) atIndex:index];
}

}
