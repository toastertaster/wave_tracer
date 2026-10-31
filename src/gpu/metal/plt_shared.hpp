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
 * Metal backend: what the plt_path and plt_bdpt kernels (plt.metal) and the
 * host (metal_renderer.mm) exchange. The structs mirror cuda_backend.cu's;
 * pointers are GPU addresses of the renderer's buffers (`MTLBuffer.gpuAddress`),
 * which every launch makes resident (`useResources`). Their layouts are
 * checked at start up (layout.hpp).
 */

#include <wt/gpu/kernel/plt_path.hpp>
#include <wt/gpu/kernel/plt_bdpt.hpp>

namespace wt { namespace gpu { namespace metal_backend {

using compute_kernel::u32;
using compute_kernel::u64;
using compute_kernel::real;

#if defined(__METAL_VERSION__)
WT_GPU_FN u32 atomic_add(WT_GPU_GLOBAL u32* p, u32 v) {
    return metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic_uint*>(p), v,
                                            metal::memory_order_relaxed);
}
WT_GPU_FN void atomic_sub(WT_GPU_GLOBAL u32* p, u32 v) {
    metal::atomic_fetch_sub_explicit(reinterpret_cast<device metal::atomic_uint*>(p), v,
                                     metal::memory_order_relaxed);
}
WT_GPU_FN void atomic_or(WT_GPU_GLOBAL u32* p, u32 v) {
    metal::atomic_fetch_or_explicit(reinterpret_cast<device metal::atomic_uint*>(p), v,
                                    metal::memory_order_relaxed);
}
WT_GPU_FN void atomic_add(WT_GPU_GLOBAL float* p, float v) {
    metal::atomic_fetch_add_explicit(reinterpret_cast<device metal::atomic_float*>(p), v,
                                     metal::memory_order_relaxed);
}
#endif

/**
 * @brief A film image, and its reconstruction weights (of block splats),
 *        accumulated with 32-bit float atomics: Metal has no 64-bit floating
 *        point. The host adds them to its double-precision film and clears
 *        them often enough (`metal_plt_renderer_t::flush_film`) that a float
 *        holds only a bounded number of samples' sums.
 */
struct film_view {
    WT_GPU_GLOBAL float* values;
    WT_GPU_GLOBAL float* weights;

#if defined(__METAL_VERSION__)
    WT_GPU_FN void add(u32 i, real v) { atomic_add(values+i, v); }
    WT_GPU_FN void add_weight(u32 i, real w) { atomic_add(weights+i, w); }
#endif
};

/**
 * @brief A path to (re-)trace, skipping the splats an earlier attempt made,
 *        with its spectral sample (or `compute_kernel::choose_spectral`, for a new path).
 */
struct path_entry {
    u64 path;
    u32 skip;
    u32 seed;
    u32 spectral;
};

/**
 * @brief Counters of a launch: paths that go on, slots freed, paths that
 *        overflowed; of a launch of whole paths, the paths taken, and the
 *        plt_bdpt connections deferred.
 */
struct step_counters {
    u32 queued, freed, overflowed;
    u32 taken;
    u32 deferred;
};

/**
 * @brief Paths in flight, each in a slot: its state and its scratch memory
 *        (plt_path), or its subpaths' vertices (plt_bdpt). Unused slots are
 *        on a stack, `free_slots`. `table_paths` (if not null) counts the
 *        paths in flight of each spectral table, of `table_size` samples.
 *        (cuda_backend.cu's `path_pool_t`.)
 */
struct path_pool {
    WT_GPU_GLOBAL void* paths;                      // `path_state_t<P>` of the paths' sensor (see `states`)
    WT_GPU_GLOBAL compute_kernel::traversal_t* traversals;     // of the step in progress
    WT_GPU_GLOBAL u32* trace_results;               // of the step in progress: status flags, and `trace_found`
    WT_GPU_GLOBAL u32* free_slots;
    WT_GPU_GLOBAL u32* tris;
    WT_GPU_GLOBAL u32* edges;
    WT_GPU_GLOBAL compute_kernel::aperture_entry* aperture;
    u32 tri_capacity, edge_capacity;
    WT_GPU_GLOBAL u32* table_paths;
    u32 table_size;
    // plt_bdpt: `vertex_capacity` vertices of each subpath, and `aperture_capacity` aperture edges of each path
    WT_GPU_GLOBAL void* sensor_vertices;            // `bdpt_vertex<P>` of the paths' sensor (see `sensor_verts`)
    WT_GPU_GLOBAL compute_kernel::bdpt_vertex<false>* emitter_vertices;
    u32 vertex_capacity;
    WT_GPU_GLOBAL compute_kernel::fraunhofer::fsd_edge* aperture_edges;
    u32 aperture_capacity;

#if defined(__METAL_VERSION__)
    /** @brief A path with spectral sample `spectral` ended. */
    WT_GPU_FN void path_ended(u32 spectral) const {
        if (table_paths)
            atomic_sub(table_paths + spectral/table_size, 1u);
    }

    /** @brief The path states, of a polarimetric sensor (`P`) or not. */
    template <bool P>
    WT_GPU_FN WT_GPU_GLOBAL compute_kernel::path_state_t<P>* states() const {
        return static_cast<WT_GPU_GLOBAL compute_kernel::path_state_t<P>*>(paths);
    }
    /** @brief The sensor subpath vertices of a plt_bdpt path, of a polarimetric sensor (`P`) or not. */
    template <bool P>
    WT_GPU_FN WT_GPU_GLOBAL compute_kernel::bdpt_vertex<P>* sensor_verts(u32 slot) const {
        return static_cast<WT_GPU_GLOBAL compute_kernel::bdpt_vertex<P>*>(sensor_vertices) + u64(slot)*vertex_capacity;
    }
    /** @brief The emitter subpath vertices of a plt_bdpt path. */
    WT_GPU_FN WT_GPU_GLOBAL compute_kernel::bdpt_vertex<false>* emitter_verts(u32 slot) const {
        return emitter_vertices + u64(slot)*vertex_capacity;
    }
    /** @brief The edges of the apertures of a plt_bdpt path's free-space diffractions. */
    WT_GPU_FN WT_GPU_GLOBAL compute_kernel::fraunhofer::fsd_edge* apertures(u32 slot) const {
        return aperture_edges + u64(slot)*aperture_capacity;
    }

    WT_GPU_FN compute_kernel::path_scratch scratch(u32 slot) const {
        return {
            tris + u64(slot)*tri_capacity,
            edges + u64(slot)*3*tri_capacity,
            aperture + u64(slot)*edge_capacity,
            tri_capacity, edge_capacity,
        };
    }
#endif
};

/** @brief `path_pool::trace_results` flag: the traversal found an interaction. */
WT_GPU_CONSTANT constexpr u32 trace_found = 1u<<31;

/**
 * @brief A launch of a path kernel: its paths and where its results go.
 *        Paths are `base+i` with `seed` and spectral table `table`, or
 *        `list[base+i]` if `list` is not null (see `path_entry`).
 */
struct plt_launch {
    path_pool pool;
    film_view film;
    film_view light;                    // plt_bdpt's light image, of direct splats
    WT_GPU_GLOBAL step_counters* counters;
    WT_GPU_GLOBAL path_entry* overflow; // paths that outgrew their scratch memory, to re-trace
    WT_GPU_GLOBAL u32* status;          // status flags of all paths (compute_kernel::status_*)
    const WT_GPU_GLOBAL path_entry* list;
    const WT_GPU_GLOBAL u32* queue;     // the wavefront's paths in flight, and those of its next step
    WT_GPU_GLOBAL u32* next_queue;
    u64 base;
    u32 count;
    u32 seed;
    u32 table;
    u32 slots;                          // whole paths: threads with slots
    u32 free_top;                       // wavefront: free slots
    u32 queued;                         // wavefront: paths queued (start_paths appends to them)
    u32 max_steps;                      // step_paths: steps each path takes at most
};

} } }
