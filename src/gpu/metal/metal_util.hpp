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
#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    id<MTLComputePipelineState> pipeline(const std::string& name) {
        const auto it = pipelines.find(name);
        if (it!=pipelines.end())
            return it->second;
        id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
        if (!function)
            throw std::runtime_error("Metal kernel " + name + " not found");
        NSError* error = nil;
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!state)
            throw std::runtime_error("cannot create the Metal pipeline of " + name + ": " +
                                     to_string(error.localizedDescription));
        pipelines.emplace(name, state);
        return state;
    }

    /**
     * @brief Runs a command buffer of one compute pass, whose dispatches
     *        `encode` encodes (they run one after the other), and waits for
     *        it. Returns the GPU's time running it, in seconds. Throws if the
     *        GPU reports an error (as when macOS stops a command buffer that
     *        runs too long).
     */
    double run(const char* what, const std::function<void(id<MTLComputeCommandEncoder>)>& encode) {
        @autoreleasepool {
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            encode(encoder);
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.status==MTLCommandBufferStatusError)
                throw std::runtime_error(std::string("Metal ") + what + " failed: " +
                                         to_string(commands.error.localizedDescription));
            return commands.GPUEndTime - commands.GPUStartTime;
        }
    }

    /** @brief Encodes kernel `name` on `count` threads (one dimension), in threadgroups of up to 128. */
    void dispatch(id<MTLComputeCommandEncoder> encoder, const std::string& name, std::size_t count) {
        if (count==0) return;
        id<MTLComputePipelineState> state = pipeline(name);
        [encoder setComputePipelineState:state];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadgroup_width(state), 1, 1)];
    }

    /**
     * @brief Encodes kernel `name` on at least `count` threads, in whole
     *        threadgroups (of whole SIMD-groups): for kernels of persistent
     *        SIMD-groups. Returns the threads launched.
     */
    std::size_t dispatch_groups(id<MTLComputeCommandEncoder> encoder, const std::string& name, std::size_t count) {
        if (count==0) return 0;
        id<MTLComputePipelineState> state = pipeline(name);
        const NSUInteger width = threadgroup_width(state);
        const NSUInteger groups = (count+width-1)/width;
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        return groups*width;
    }

    /** @brief Runs kernel `name` on `count` threads and waits for it; `bind` sets its arguments. */
    void dispatch(const char* name, std::size_t count,
                  const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
        if (count==0) return;
        run(name, [&](id<MTLComputeCommandEncoder> encoder) {
            bind(encoder);
            dispatch(encoder, name, count);
        });
    }

    /**
     * @brief The GPU's cores (`gpu-core-count` of its IORegistry entry), or 0
     *        if unknown.
     */
    [[nodiscard]] unsigned gpu_cores() const {
        unsigned cores = 0;
        const io_service_t service =
            IOServiceGetMatchingService(kIOMainPortDefault, IORegistryEntryIDMatching(device.registryID));
        if (service==IO_OBJECT_NULL)
            return 0;
        CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int n = 0;
            if (CFGetTypeID(value)==CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &n) && n>0)
                cores = unsigned(n);
            CFRelease(value);
        }
        IOObjectRelease(service);
        return cores;
    }

    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;

private:
    /** @brief Threadgroups of up to 128 threads, of whole SIMD-groups (CUDA's blocks). */
    static NSUInteger threadgroup_width(id<MTLComputePipelineState> state) {
        const NSUInteger simd = state.threadExecutionWidth;
        const NSUInteger max = state.maxTotalThreadsPerThreadgroup;
        return std::max<NSUInteger>(simd, std::min<NSUInteger>(128, max - max%simd));
    }

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
    /** @brief Copies to elements `offset .. offset+count-1`, which must be allocated. */
    void upload_at(std::size_t offset, const T* data, std::size_t count) {
        if (offset+count>capacity)
            throw std::logic_error("copying past the end of a GPU buffer");
        if (count)
            std::memcpy(this->data()+offset, data, count*sizeof(T));
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
    /** @brief The buffer's GPU address, for pointers stored in structs (null if not allocated). */
    [[nodiscard]] std::uint64_t gpu_address() const noexcept { return buffer ? buffer.gpuAddress : 0; }
    /** @brief The buffer's GPU address as a pointer to `U` (`T` by default), for structs the kernels read. */
    template <typename U = T>
    [[nodiscard]] U* gpu_pointer() const noexcept {
        return reinterpret_cast<U*>(static_cast<std::uintptr_t>(gpu_address()));
    }

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
