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
#include "plt_shared.hpp"

#if defined(REALISTIC_REMOTE_SENSING)
#define WT_METAL_REMOTE_SENSING_STRUCTS(X) X(compute_kernel::telescope_desc)
#else
#define WT_METAL_REMOTE_SENSING_STRUCTS(X)
#endif

// X(struct) for each shared struct, in wt::gpu
#define WT_METAL_SHARED_STRUCTS(X) \
    WT_METAL_REMOTE_SENSING_STRUCTS(X) \
    X(compute_kernel::bvh_node) X(compute_kernel::bvh_tri) \
    X(compute_kernel::ray_query) X(compute_kernel::ray_hit) X(compute_kernel::cone_query) \
    X(compute_kernel::cone_hit) X(compute_kernel::ball) X(compute_kernel::ball_hit) \
    X(compute_kernel::tri_shading) X(compute_kernel::edge) X(compute_kernel::tri_uv) \
    X(compute_kernel::bitmap) X(compute_kernel::texture) X(compute_kernel::material_shading) \
    X(compute_kernel::material) X(compute_kernel::emitter) X(compute_kernel::emission) \
    X(compute_kernel::sensor_desc) X(compute_kernel::film_desc) X(compute_kernel::spectral_sample) \
    X(compute_kernel::plt_scene) X(compute_kernel::aperture_entry) X(compute_kernel::fraunhofer::fsd_edge) \
    X(compute_kernel::traversal_t) X(compute_kernel::path_state_t<false>) X(compute_kernel::path_state_t<true>) \
    X(compute_kernel::bdpt_vertex<false>) X(compute_kernel::bdpt_vertex<true>) \
    X(metal_backend::film_view) X(metal_backend::path_entry) X(metal_backend::step_counters) \
    X(metal_backend::path_pool) X(metal_backend::plt_launch)

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
    out[n++] = compute_kernel::u32(sizeof(T)); \
    out[n++] = compute_kernel::u32(alignof(T));
    WT_METAL_SHARED_STRUCTS(WT_METAL_WRITE_LAYOUT)
#undef WT_METAL_WRITE_LAYOUT
}

} } }
