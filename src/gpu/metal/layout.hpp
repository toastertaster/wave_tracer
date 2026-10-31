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
 * Metal backend: the structs the host and the Metal kernels exchange through
 * buffers. Both compile them from the same headers (wt/gpu/kernel/), but with
 * different compilers; the backend checks at start up that every struct has
 * the same size and alignment on both sides (`write_layouts` runs in a kernel
 * and on the host), so that a layout difference is an error rather than
 * corrupt data.
 */

#include <wt/gpu/kernel/queries.hpp>

// X(struct) for each shared struct, in wt::gpu::compute_kernel
#define WT_METAL_SHARED_STRUCTS(X) \
    X(bvh_node) X(bvh_tri) \
    X(ray_query) X(ray_hit) X(cone_query) X(cone_hit) X(ball) X(ball_hit)

namespace wt { namespace gpu { namespace metal_backend {

#define WT_METAL_COUNT_STRUCT(T) +2
/** @brief Number of values `write_layouts` writes: a size and an alignment per struct. */
WT_GPU_CONSTANT constexpr compute_kernel::u32 layout_values = 0 WT_METAL_SHARED_STRUCTS(WT_METAL_COUNT_STRUCT);
#undef WT_METAL_COUNT_STRUCT

/** @brief Writes the size and alignment of each shared struct to `out[0 .. layout_values-1]`. */
template <typename Out>
WT_GPU_FN void write_layouts(Out out) {
    compute_kernel::u32 n = 0;
#define WT_METAL_WRITE_LAYOUT(T) \
    out[n++] = compute_kernel::u32(sizeof(compute_kernel::T)); \
    out[n++] = compute_kernel::u32(alignof(compute_kernel::T));
    WT_METAL_SHARED_STRUCTS(WT_METAL_WRITE_LAYOUT)
#undef WT_METAL_WRITE_LAYOUT
}

} } }
