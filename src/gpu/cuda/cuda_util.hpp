/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace wt::gpu::cuda {

inline void check(cudaError_t err, const char* what) {
    if (err!=cudaSuccess)
        throw std::runtime_error(std::string("CUDA error ") + what + ": " + cudaGetErrorString(err));
}

inline void set_device(int ordinal) {
    check(cudaSetDevice(ordinal), "selecting the device");
}

/**
 * @brief Device memory for `T`s. Grows on demand and never shrinks. Copies are
 *        synchronous (default stream), so a download waits for prior kernels.
 */
template <typename T>
class device_buffer_t {
public:
    device_buffer_t() noexcept = default;
    device_buffer_t(const device_buffer_t&) = delete;
    device_buffer_t& operator=(const device_buffer_t&) = delete;
    ~device_buffer_t() noexcept { if (ptr) cudaFree(ptr); }

    void resize(std::size_t count) {
        if (count<=capacity) return;
        if (ptr) {
            check(cudaFree(ptr), "freeing device memory");
            ptr = nullptr;
            capacity = 0;
        }
        check(cudaMalloc(&ptr, count*sizeof(T)), "allocating device memory");
        capacity = count;
    }
    void upload(const T* data, std::size_t count) {
        resize(count);
        if (count)
            check(cudaMemcpy(ptr, data, count*sizeof(T), cudaMemcpyHostToDevice), "copying to the device");
    }
    void download(T* data, std::size_t count) const {
        if (count)
            check(cudaMemcpy(data, ptr, count*sizeof(T), cudaMemcpyDeviceToHost), "running a GPU query");
    }

    [[nodiscard]] T* get() const noexcept { return ptr; }
    [[nodiscard]] std::size_t size() const noexcept { return capacity; }

private:
    T* ptr = nullptr;
    std::size_t capacity = 0;
};

}
