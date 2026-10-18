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
 * Shared GPU kernel code.
 *
 * Everything under wt/gpu/kernel/ is written in a subset of C++17 that compiles
 * as plain host C++, as CUDA device code, and as Metal Shading Language:
 *   - no C++ standard library, no exceptions, RTTI, virtual functions, derived
 *     classes, new/delete, thread_local or recursion;
 *   - plain structs and templates only;
 *   - pointers to GPU buffers are declared with WT_GPU_GLOBAL (Metal's `device`
 *     address space), and reference parameters to local variables with
 *     WT_GPU_THREAD (Metal's `thread`); Metal requires an explicit address
 *     space on pointer and reference parameters.
 * The rest of wave_tracer (C++20, mp-units, glm) is never included from here;
 * host code translates its types into these structs (see wt/gpu/gpu.hpp).
 *
 * Precision: kernel scalars are `real`, 32-bit unless WT_GPU_DBL_PRECISION is
 * defined (the GPU_DBL_PRECISION CMake option). This is independent of the
 * CPU's `f_t` (DBL_PRECISION).
 */

#if defined(__CUDACC__)
    #define WT_GPU_FN __host__ __device__ inline
    #define WT_GPU_GLOBAL
    #define WT_GPU_THREAD
#elif defined(__METAL_VERSION__)
    #include <metal_stdlib>
    #define WT_GPU_FN inline
    #define WT_GPU_GLOBAL device
    #define WT_GPU_THREAD thread
#else
    #define WT_GPU_FN inline
    #define WT_GPU_GLOBAL
    #define WT_GPU_THREAD
#endif

#if !defined(__METAL_VERSION__)
    #include <math.h>
#endif

namespace wt::gpu::kernel {

#if defined(WT_GPU_DBL_PRECISION)
    #if defined(__METAL_VERSION__)
        #error "Metal has no 64-bit floating point: build without GPU_DBL_PRECISION"
    #endif
using real = double;
#else
using real = float;
#endif

// 32-bit integers on every target (Metal's int/uint are 32-bit).
using i32 = int;
using u32 = unsigned int;

static_assert(sizeof(i32)==4 && sizeof(u32)==4);

}
