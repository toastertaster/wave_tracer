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
 * as plain host C++, as CUDA device code, and as Metal Shading Language 3.2:
 *   - no C++ standard library, no exceptions, RTTI, virtual functions, derived
 *     classes, new/delete, thread_local or recursion;
 *   - plain structs and templates only;
 *   - functions are WT_GPU_FN, or WT_GPU_NOINLINE_FN to keep a rarely taken
 *     path's code and registers out of its callers (CUDA's __noinline__);
 *   - pointers to GPU buffers are declared with WT_GPU_GLOBAL (Metal's `device`
 *     address space), and reference parameters to local variables with
 *     WT_GPU_THREAD (Metal's `thread`); Metal requires an explicit address
 *     space on pointer and reference parameters;
 *   - constants at namespace scope are `WT_GPU_CONSTANT constexpr` (Metal's
 *     `constant` address space);
 *   - namespaces are declared one level at a time,
 *     `namespace wt { namespace gpu { namespace compute_kernel {`: Metal does not
 *     accept C++17's `namespace wt::gpu::compute_kernel {`. (The namespace is not
 *     `kernel`, which is a Metal keyword.)
 * The rest of wave_tracer (C++20, mp-units, glm) is never included from here;
 * host code translates its types into these structs (see wt/gpu/gpu.hpp).
 *
 * Precision: kernel scalars are `real`, 32-bit unless WT_GPU_DBL_PRECISION is
 * defined (the GPU_DBL_PRECISION CMake option). This is independent of the
 * CPU's `f_t` (DBL_PRECISION).
 */

#if defined(__CUDACC__)
    #define WT_GPU_FN __host__ __device__ inline
    #define WT_GPU_NOINLINE_FN __host__ __device__ __noinline__
    #define WT_GPU_GLOBAL
    #define WT_GPU_THREAD
    #define WT_GPU_CONSTANT
#elif defined(__METAL_VERSION__)
    #include <metal_stdlib>
    #define WT_GPU_FN inline
    #define WT_GPU_NOINLINE_FN inline
    #define WT_GPU_GLOBAL device
    #define WT_GPU_THREAD thread
    #define WT_GPU_CONSTANT constant
#else
    #define WT_GPU_FN inline
    #define WT_GPU_NOINLINE_FN inline
    #define WT_GPU_GLOBAL
    #define WT_GPU_THREAD
    #define WT_GPU_CONSTANT
#endif

#if !defined(__METAL_VERSION__)
    #include <math.h>
#endif

namespace wt { namespace gpu { namespace compute_kernel {

#if defined(WT_GPU_DBL_PRECISION)
    #if defined(__METAL_VERSION__)
        #error "Metal has no 64-bit floating point: build without GPU_DBL_PRECISION"
    #endif
using real = double;
#else
using real = float;
#endif

// 32-bit integers on every target (Metal's int/uint are 32-bit), and 64-bit ones.
using i32 = int;
using u32 = unsigned int;
#if defined(__METAL_VERSION__)
using u64 = ulong;
#else
using u64 = unsigned long long;
#endif

static_assert(sizeof(i32)==4 && sizeof(u32)==4 && sizeof(u64)==8);

} } }
