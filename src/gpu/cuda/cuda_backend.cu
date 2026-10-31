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

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <wt/gpu/backend.hpp>
#include <wt/gpu/kernel/plt_path.hpp>
#include <wt/gpu/kernel/plt_bdpt.hpp>
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


/*
 * plt_path and plt_bdpt
 */

/** @brief Geometry queries for the path tracers: the CUDA BVH traversals. */
struct cuda_bvh_geometry_t {
    k::bvh_view bvh;

    __device__ k::ray_hit intersect(const k::ray& r, k::range rng) const {
        return k::intersect_ray(bvh, r, rng);
    }
    __device__ bool occluded(const k::ray& r, k::range rng, k::u32& status) const {
        return k::ray_occluded(bvh, r, rng, status);
    }
    __device__ k::cone_hit intersect_cone(const k::cone& c, k::range rng, k::real z_scale,
                                          k::tri_sink& sink) const {
        return k::intersect_cone(bvh, c, rng, z_scale, sink);
    }
};

/**
 * @brief A film image, and its reconstruction weights (of block splats),
 *        accumulated in double precision.
 */
struct cuda_film_t {
    double* values;
    double* weights;

    __device__ void add(k::u32 i, k::real v) {
        atomicAdd(values+i, double(v));
    }
    __device__ void add_weight(k::u32 i, k::real w) {
        atomicAdd(weights+i, double(w));
    }
};

/**
 * @brief A path to (re-)trace, skipping the splats an earlier attempt made,
 *        with its spectral sample (or `k::choose_spectral`, for a new path).
 */
struct path_entry_t {
    std::uint64_t path;
    std::uint32_t skip;
    std::uint32_t seed;
    std::uint32_t spectral;
};

/**
 * @brief Paths in flight, each in a slot: its state and its scratch memory
 *        (plt_path), or its subpaths' vertices (plt_bdpt). Unused slots are
 *        on a stack, `free_slots`. `table_paths` (if not null) counts the
 *        paths in flight of each spectral table, of `table_size` samples.
 */
struct path_pool_t {
    void* paths;                    // `k::path_state_t<P>` of the paths' sensor (see `states`)
    k::traversal_t* traversals;     // of the step in progress
    k::u32* trace_results;          // of the step in progress: status flags, and `trace_found`
    k::u32* free_slots;
    k::u32* tris;
    k::u32* edges;
    k::aperture_entry* aperture;
    k::u32 tri_capacity, edge_capacity;
    k::u32* table_paths;
    k::u32 table_size;
    // plt_bdpt: `vertex_capacity` vertices of each subpath, and `aperture_capacity` aperture edges of each path
    void* sensor_vertices;          // `k::bdpt_vertex<P>` of the paths' sensor (see `sensor_verts`)
    k::bdpt_vertex<false>* emitter_vertices;
    k::u32 vertex_capacity;
    k::fraunhofer::fsd_edge* aperture_edges;
    k::u32 aperture_capacity;

    /** @brief `count` paths start with spectral table `table` (the first thread of a launch counts them). */
    __device__ void paths_started(k::u32 table, k::u32 count) const {
        if (table_paths && blockIdx.x==0 && threadIdx.x==0)
            atomicAdd(table_paths + table/table_size, count);
    }
    /** @brief A path with spectral sample `spectral` ended. */
    __device__ void path_ended(k::u32 spectral) const {
        if (table_paths)
            atomicSub(table_paths + spectral/table_size, 1u);
    }

    /** @brief The path states, of a polarimetric sensor (`P`) or not. */
    template <bool P>
    __device__ k::path_state_t<P>* states() const { return static_cast<k::path_state_t<P>*>(paths); }

    /** @brief The sensor subpath vertices of a plt_bdpt path, of a polarimetric sensor (`P`) or not. */
    template <bool P>
    __device__ k::bdpt_vertex<P>* sensor_verts(k::u32 slot) const {
        return static_cast<k::bdpt_vertex<P>*>(sensor_vertices) + std::size_t(slot)*vertex_capacity;
    }
    /** @brief The emitter subpath vertices of a plt_bdpt path. */
    __device__ k::bdpt_vertex<false>* emitter_verts(k::u32 slot) const {
        return emitter_vertices + std::size_t(slot)*vertex_capacity;
    }
    /** @brief The edges of the apertures of a plt_bdpt path's free-space diffractions. */
    __device__ k::fraunhofer::fsd_edge* apertures(k::u32 slot) const {
        return aperture_edges + std::size_t(slot)*aperture_capacity;
    }

    __device__ k::path_scratch scratch(k::u32 slot) const {
        return {
            tris + std::size_t(slot)*tri_capacity,
            edges + std::size_t(slot)*3*tri_capacity,
            aperture + std::size_t(slot)*edge_capacity,
            tri_capacity, edge_capacity,
        };
    }
};

/**
 * @brief Loads and stores (of path states) in 16-byte words: a plain struct
 *        copy makes 4-byte accesses, many times slower.
 */
template <typename T>
__device__ T load_vectors(const T* src) {
    constexpr int words = sizeof(T)/sizeof(uint4);
    static_assert(words*sizeof(uint4)==sizeof(T));
    uint4 w[words];
    for (int i=0; i<words; ++i)
        w[i] = reinterpret_cast<const uint4*>(src)[i];
    T v;
    memcpy(&v, w, sizeof(v));
    return v;
}
template <typename T>
__device__ void store_vectors(T* dst, const T& v) {
    constexpr int words = sizeof(T)/sizeof(uint4);
    static_assert(words*sizeof(uint4)==sizeof(T));
    uint4 w[words];
    memcpy(w, &v, sizeof(v));
    for (int i=0; i<words; ++i)
        reinterpret_cast<uint4*>(dst)[i] = w[i];
}

/** @brief `trace_results` flag: the traversal found an interaction. */
constexpr k::u32 trace_found = 1u<<31;

/**
 * @brief Counters of a step: paths that go on, slots freed, paths that
 *        overflowed; and of a launch of plt_bdpt paths, the paths taken.
 */
struct step_counters_t {
    std::uint32_t queued, freed, overflowed;
    std::uint32_t taken;
};

__global__ void fill_free_slots_kernel(k::u32* free_slots, k::u32 count) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<count) free_slots[i] = i;
}

/*
 * The path kernels are templates of `P`: whether the sensor is polarimetric
 * (`k::path_state_t`).
 */

/** @brief Path `base+i` with `seed` (and a spectral sample of its table), or `list[base+i]` if `list` is given. */
__device__ path_entry_t path_entry(std::uint64_t base, k::u32 i, const path_entry_t* list, std::uint32_t seed) {
    return list ? list[base+i] : path_entry_t{ base+i, 0, seed, k::choose_spectral };
}

/** @brief A path's random number generator: its index and seed determine it. */
__device__ k::rng path_rng(const path_entry_t& e) {
    return k::rng::for_path(e.seed, k::u32(e.path), k::u32(e.path>>32));
}

/** @brief Path `base+i` with `seed` and spectral table `table`, or `list[base+i]` if `list` is given. */
template <bool P>
__device__ k::path_state_t<P> start_path(const k::plt_scene& scene, std::uint64_t base, k::u32 i,
                                         const path_entry_t* list, std::uint32_t seed, k::u32 table) {
    const path_entry_t e = path_entry(base, i, list, seed);
    return k::start_path<P>(scene, path_rng(e), e.path, e.skip, table, e.spectral);
}

/**
 * @brief Starts `count` paths (see `start_path`). They take the slots on top
 *        of the free stack (`free_top` slots) and are queued after the
 *        `queued` paths of `queue`.
 */
template <bool P>
__global__ void start_paths_kernel(k::plt_scene scene, std::uint64_t base, k::u32 count,
                                   const path_entry_t* list, std::uint32_t seed, k::u32 table, path_pool_t pool,
                                   k::u32 free_top, k::u32* queue, k::u32 queued) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    pool.paths_started(table, count);
    const k::u32 slot = pool.free_slots[free_top-1-i];
    store_vectors(pool.states<P>()+slot, start_path<P>(scene, base, i, list, seed, table));
    queue[queued+i] = slot;
}

/**
 * @brief Ends a path, or lists it in `overflow` if it outgrew its scratch
 *        memory (to be re-traced; `end_path` would splat what it has not finished).
 */
template <bool P>
__device__ void end_path(const k::plt_scene& scene, cuda_film_t& film, const path_pool_t& pool, k::path_state_t<P>& p,
                         k::u32& s, step_counters_t* counters, path_entry_t* overflow) {
    if (s & k::status_scratch_overflow) {
        // the path's generator says which path it is
        const std::uint64_t path = p.rnd.path_lo | std::uint64_t(p.rnd.path_hi)<<32;
        overflow[atomicAdd(&counters->overflowed, 1u)] = { path, p.splats, p.rnd.seed, p.spectral };
        s &= ~k::status_scratch_overflow;
    } else {
        k::end_path(scene, film, p);
    }
    pool.path_ended(p.spectral);
}

/**
 * @brief Ends or requeues a path after a step: paths that go on are queued in
 *        `next_queue`; those that end (`end_path`) free their slots, pushed
 *        above the `free_top` free ones.
 */
template <bool P>
__device__ void finish_step(const k::plt_scene& scene, cuda_film_t& film, const path_pool_t& pool, k::u32 slot,
                            k::path_state_t<P>& p, bool goes_on, k::u32 s, k::u32* next_queue, k::u32 free_top,
                            step_counters_t* counters, path_entry_t* overflow, std::uint32_t* status) {
    if (s & k::status_scratch_overflow)
        goes_on = false;
    if (goes_on) {
        store_vectors(pool.states<P>()+slot, p);
        next_queue[atomicAdd(&counters->queued, 1u)] = slot;
    } else {
        end_path(scene, film, pool, p, s, counters, overflow);
        pool.free_slots[free_top + atomicAdd(&counters->freed, 1u)] = slot;
    }
    if (s!=k::status_ok)
        atomicOr(status, s);
}

/**
 * @brief First part of a step of the `count` queued paths: traces their beams
 *        (`trace_step`). A kernel of its own, so that the BVH traversals run
 *        with fewer registers, without warps diverging into interaction code.
 */
template <bool P>
__global__ void trace_paths_kernel(k::plt_scene scene, path_pool_t pool, const k::u32* queue, k::u32 count) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    const k::path_state_t<P> p = load_vectors(pool.states<P>()+slot);
    const cuda_bvh_geometry_t geometry = { scene.bvh };

    k::traversal_t tr;
    k::u32 s = k::status_ok;
    const bool found = k::trace_step(scene, geometry, pool.scratch(slot), p, tr, s);
    if (found)
        store_vectors(pool.traversals+slot, tr);
    pool.trace_results[slot] = s | (found ? trace_found : 0u);
}

/** @brief Second part of a step of the `count` queued paths (`interact`), after `trace_paths_kernel`. */
template <bool P>
__global__ void __launch_bounds__(block_size, 4) interact_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        const k::u32* queue, k::u32 count,
                                        cuda_film_t film, k::u32* next_queue, k::u32 free_top,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    k::path_state_t<P> p = load_vectors(pool.states<P>()+slot);
    const k::u32 traced = pool.trace_results[slot];
    k::u32 s = traced & ~trace_found;
    bool goes_on = false;
    if (traced & trace_found) {
        const k::traversal_t tr = load_vectors(pool.traversals+slot);
        const cuda_bvh_geometry_t geometry = { scene.bvh };
        goes_on = k::interact(scene, geometry, film, pool.scratch(slot), tr, p, s);
    }
    finish_step(scene, film, pool, slot, p, goes_on, s, next_queue, free_top, counters, overflow, status);
}

/**
 * @brief Advances the `count` queued paths by up to `max_steps` whole steps,
 *        in one kernel: for the last few paths, which would otherwise take a
 *        pair of launches per step.
 */
template <bool P>
__global__ void __launch_bounds__(block_size, 4) step_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        const k::u32* queue, k::u32 count, k::u32 max_steps,
                                        cuda_film_t film, k::u32* next_queue, k::u32 free_top,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    k::path_state_t<P> p = load_vectors(pool.states<P>()+slot);
    const cuda_bvh_geometry_t geometry = { scene.bvh };

    k::u32 s = k::status_ok;
    bool goes_on = true;
    for (k::u32 n=0; n<max_steps && goes_on; ++n)
        goes_on = k::step_path(scene, geometry, film, pool.scratch(slot), p, s);
    finish_step(scene, film, pool, slot, p, goes_on, s, next_queue, free_top, counters, overflow, status);
}

/**
 * @brief Persistent warps, for kernels of whole paths: a warp takes the next
 *        of `count` paths, as many as it has lanes with slots (of `slots`;
 *        the counter `counters->taken` starts at 0), once all of them finished
 *        their last. A lane traces path `i` with the memory of the slot of its
 *        thread index, `trace(slot, i)`. The launch has whole warps.
 *
 *        A launch of a thread per path, or a wavefront of steps (as plt_path's
 *        beams), would idle at each launch's end, until its longest path, or
 *        step, finished: paths' work varies greatly (their lengths, diffusive
 *        traversals, free-space diffractions' rejection sampling). A warp
 *        takes its paths together, so that its lanes trace neighbouring
 *        paths in step, as a launch's would.
 */
template <typename F>
__device__ void take_paths(k::u32 slots, k::u32 count, step_counters_t* counters, F&& trace) {
    const k::u32 slot = blockIdx.x*blockDim.x + threadIdx.x;
    const k::u32 lane = threadIdx.x % 32;
    // the warp's lanes with slots are its first `lanes`
    const k::u32 lanes = __popc(__ballot_sync(~0u, slot<slots));
    if (lanes==0) return;

    for (;;) {
        k::u32 first = 0;
        if (lane==0)
            first = atomicAdd(&counters->taken, lanes);
        first = __shfl_sync(~0u, first, 0);
        if (first>=count)
            return;

        const k::u32 i = first + lane;
        if (lane<lanes && i<count)
            trace(slot, i);
        __syncwarp();
    }
}

/**
 * @brief Traces `count` whole paths (see `start_path`), by persistent warps
 *        (`take_paths`), with the scratch memory of `slots` slots. For ray
 *        tracing, whose steps are cheap: moving path states between launches
 *        would take longer than the steps.
 */
template <bool P>
__global__ void __launch_bounds__(block_size, 3) run_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        k::u32 slots, std::uint64_t base, k::u32 count, const path_entry_t* list,
                                        std::uint32_t seed, k::u32 table, cuda_film_t film,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    pool.paths_started(table, count);
    const cuda_bvh_geometry_t geometry = { scene.bvh };
    take_paths(slots, count, counters, [&](k::u32 slot, k::u32 i) {
        k::path_state_t<P> p = start_path<P>(scene, base, i, list, seed, table);
        k::u32 s = k::status_ok;
        while (k::step_path(scene, geometry, film, pool.scratch(slot), p, s)) {}

        end_path(scene, film, pool, p, s, counters, overflow);
        if (s!=k::status_ok)
            atomicOr(status, s);
    });
}

/**
 * @brief Traces `count` whole plt_bdpt paths (see `k::trace_bdpt_path`), by
 *        persistent warps (`take_paths`), with the vertices and scratch memory
 *        of `slots` slots, splatting into `film` and its light image `light`.
 *        Paths that outgrow them, which splat nothing, are listed in `overflow`.
 */
template <bool P>
__global__ void __launch_bounds__(block_size, 2) run_bdpt_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        k::u32 slots, std::uint64_t base, k::u32 count, const path_entry_t* list,
                                        std::uint32_t seed, k::u32 table, cuda_film_t film, cuda_film_t light,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    pool.paths_started(table, count);
    const cuda_bvh_geometry_t geometry = { scene.bvh };
    take_paths(slots, count, counters, [&](k::u32 slot, k::u32 i) {
        const path_entry_t e = path_entry(base, i, list, seed);
        k::u32 s = k::status_ok;
        const k::u32 spectral = k::trace_bdpt_path(scene, geometry, film, light, pool.scratch(slot),
                                                   pool.sensor_verts<P>(slot), pool.emitter_verts(slot),
                                                   pool.vertex_capacity, pool.apertures(slot),
                                                   pool.aperture_capacity, path_rng(e), e.path, table,
                                                   e.spectral, s);
        if (s & k::status_scratch_overflow) {
            overflow[atomicAdd(&counters->overflowed, 1u)] = { e.path, 0, e.seed, spectral };
            s &= ~k::status_scratch_overflow;
        }
        pool.path_ended(spectral);
        if (s!=k::status_ok)
            atomicOr(status, s);
    });
}

/**
 * @brief plt_path on the GPU, as a wavefront: a pool of paths in flight,
 *        advanced a step at a time, so that warps run paths still going (most
 *        paths end after a step or two), with the traversals in a kernel of
 *        their own. Ended paths' slots are refilled with new paths. Paths in
 *        flight carry over from one `trace` to the next, so that the GPU does
 *        not idle while a pass's last, longest, paths finish.
 *
 *        Ray-traced paths, whose steps are cheap, and plt_bdpt paths run
 *        whole, in launches of persistent warps (see `take_paths`), with a
 *        slot of the pool for each thread the GPU runs at once (plt_bdpt: or
 *        as many as their vertices' memory allows).
 */
class cuda_plt_renderer_t final : public backend_renderer_t {
public:
    cuda_plt_renderer_t(int ordinal, k::bvh_view bvh, const plt_scene_t& scene)
        : ordinal(ordinal),
          polarimetric(scene.film.polarimetric!=0),
          table_size(scene.emitter_count*scene.samples_per_emitter)
    {
        set_device(ordinal);
        if (scene.params.bdpt!=0 || scene.params.ray_tracing_only!=0) {
            // whole paths (`whole_paths`): the threads of persistent warps the GPU runs at once
            int multiprocessors = 0, blocks = 0;
            check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, ordinal),
                  "querying the CUDA device");
            const auto occupancy = [&](auto kernel) {
                check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, block_size, 0),
                      "querying the path kernel's occupancy");
            };
            dispatch([&](auto P) {
                if (scene.params.bdpt!=0) occupancy(run_bdpt_paths_kernel<P.value>);
                else                      occupancy(run_paths_kernel<P.value>);
            });
            whole_path_threads = k::u32(std::max(1, multiprocessors*blocks)) * block_size;
        }
        tris.upload(scene.tris.data(), scene.tris.size());
        edges.upload(scene.edges.data(), scene.edges.size());
        emitters.upload(scene.emitters.data(), scene.emitters.size());
        emitter_cdf.upload(scene.emitter_cdf.data(), scene.emitter_cdf.size());
        area_tris.upload(scene.area_tris.data(), scene.area_tris.size());
        area_cdf.upload(scene.area_cdf.data(), scene.area_cdf.size());
        tri_uvs.upload(scene.tri_uvs.data(), scene.tri_uvs.size());
        material_shadings.upload(scene.material_shadings.data(), scene.material_shadings.size());
        textures.upload(scene.textures.data(), scene.textures.size());
        bitmaps.upload(scene.bitmaps.data(), scene.bitmaps.size());
        texels.upload(scene.texels.data(), scene.texels.size());
        texel_luts.upload(scene.texel_luts.data(), scene.texel_luts.size());
        fsd_lut.upload(scene.fsd_lut.data(), scene.fsd_lut.size());

        // a polarimetric film's components have four values (Stokes parameters), and one weight
        weights_size = std::size_t(scene.film.width) * scene.film.height * scene.film.components;
        film_size = weights_size * (scene.film.polarimetric!=0 ? 4 : 1);
        film.resize(film_size);
        check(cudaMemset(film.get(), 0, film_size*sizeof(double)), "clearing the film");
        if (scene.params.block_splats()) {
            film_weights.resize(weights_size);
            check(cudaMemset(film_weights.get(), 0, weights_size*sizeof(double)), "clearing the film");
        }
        if (scene.params.bdpt!=0) {
            light_film.resize(film_size);
            check(cudaMemset(light_film.get(), 0, film_size*sizeof(double)), "clearing the film");
        }
        const std::uint32_t zero = 0;
        status_word.upload(&zero, 1);
        counters.resize(1);
        const std::uint32_t no_paths[tables] = {};
        table_paths.upload(no_paths, tables);

        params.bvh = bvh;
        params.tris = tris.get();
        params.edges = edges.get();
        params.emitters = emitters.get();
        params.emitter_cdf = emitter_cdf.get();
        params.area_tris = area_tris.get();
        params.area_cdf = area_cdf.get();
        params.tri_uvs = tri_uvs.get();
        params.material_shadings = material_shadings.get();
        params.textures = textures.get();
        params.bitmaps = bitmaps.get();
        params.texels = texels.get();
        params.texel_luts = texel_luts.get();
        params.fsd_lut = fsd_lut.get();
        params.emitter_count = scene.emitter_count;
        params.material_count = scene.material_count;
        params.samples_per_emitter = scene.samples_per_emitter;
        params.sensor = scene.sensor;
        params.film = scene.film;
        params.params = scene.params;
        reserve_spectra(std::size_t(tables)*table_size);
    }

    std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed,
                        const spectral_table_t& spectra) override {
        set_device(ordinal);
        if (spectra.samples.size()!=table_size ||
            spectra.emissions.size()!=std::size_t(table_size)*params.emitter_count ||
            spectra.materials.size()!=std::size_t(table_size)*params.material_count)
            throw std::logic_error("the spectral table does not match the compiled scene");
        if (main.slots==0)
            begin(main, initial_capacity);

        // the next table slot, once the paths of the table it holds have ended
        const k::u32 slot = next_table;
        next_table = (next_table+1) % tables;
        drain_table(slot);
        const k::u32 base = slot*table_size;
        upload_spectra(base, spectra);
        loaded[slot] = { base, table_size, ++generation, spectra };

        path_source_t source = { first, count, nullptr };
        advance(main, source, seed, base, pool_size/2);
        return read_status();
    }

    std::uint32_t finish() override {
        set_device(ordinal);
        if (main.slots>0) {
            path_source_t none = { 0, 0, nullptr };
            advance(main, none, 0, 0, 0);
        }

        // paths that outgrew their scratch memory are re-traced with more room, with the spectral samples
        // they had (kept in `retained`: the tables they came from may have been replaced since)
        k::u32 capacity = initial_capacity;
        std::vector<path_entry_t> pending;
        while (!overflowed.empty()) {
            if (capacity>=max_capacity)
                throw std::runtime_error("a path found more than " + std::to_string(max_capacity) +
                                         " triangles or edges in one query");
            capacity *= 4;
            std::swap(pending, overflowed);
            overflowed.clear();

            // the retained samples are the retries' table (no paths of the main pool are in flight)
            const auto count = k::u32(retained.samples.size());
            reserve_spectra(count);
            upload_spectra(0, retained);
            loaded.fill({});
            loaded[0] = { 0, count, ++generation, std::move(retained) };
            retained = {};
            retained_index.clear();

            wavefront_t retry;
            begin(retry, capacity);
            path_list.upload(pending.data(), pending.size());
            path_source_t source = { 0, pending.size(), path_list.get() };
            advance(retry, source, 0, 0, 0, /* count table paths */ false);
        }
        // the retries reuse the main pool's memory and spectral tables: start them anew
        main = {};
        loaded.fill({});
        next_table = 0;

        return read_status();
    }

    void read_film(double* values, double* weights) override {
        set_device(ordinal);
        film.download(values, film_size);
        if (weights) {
            if (!params.params.block_splats())
                throw std::logic_error("forward transport accumulates no film weights");
            film_weights.download(weights, weights_size);
        }
    }

    void read_light_film(double* values) override {
        set_device(ordinal);
        if (params.params.bdpt==0)
            throw std::logic_error("only plt_bdpt accumulates a light image");
        light_film.download(values, film_size);
    }

private:
    /** @brief Paths in flight. */
    static constexpr k::u32 pool_size = 1u<<18;
    /** @brief Triangles and edges a path holds at first; paths needing more are re-traced. */
    static constexpr k::u32 initial_capacity = 64;
    static constexpr k::u32 max_capacity = 1u<<20;
    /** @brief plt_bdpt: vertices a subpath holds at first (at `initial_capacity`; growing with it). */
    static constexpr k::u32 initial_vertex_capacity = 16;
    /** @brief plt_bdpt: aperture edges of free-space diffractions a path holds at first (as vertices). */
    static constexpr k::u32 initial_aperture_capacity = 256;
    /** @brief plt_bdpt: memory for paths in flight, which bounds their count. */
    static constexpr std::size_t bdpt_pool_bytes = std::size_t(512)<<20;
    /** @brief Whole paths a launch traces at most (its list of overflowed paths holds as many). */
    static constexpr k::u32 launch_paths = 1u<<20;
    /** @brief Paths in flight few enough to finish in one launch, rather than a launch per step. */
    static constexpr k::u32 tail_size = 1024;
    /**
     * @brief Spectral tables on the device: paths of the last few `trace`s
     *        may still be in flight, each with a sample of its pass's table.
     */
    static constexpr k::u32 tables = 4;

    /** @brief A pool of paths in flight. */
    struct wavefront_t {
        k::u32 slots = 0;           // 0: not started
        k::u32 capacity = 0;        // scratch triangles and edges per path
        k::u32 queued = 0;          // paths in flight, in `queues[q]`
        k::u32 free_top = 0;        // free slots
        int q = 0;
    };

    /** @brief Paths to start: `first+i`, or `list[i]` if `list` is given. */
    struct path_source_t {
        std::uint64_t first, count;
        const path_entry_t* list;
        std::uint64_t started = 0;
    };

    /** @brief A spectral table on the device: samples `base .. base+count-1`, and a copy of it. */
    struct loaded_table_t {
        k::u32 base = 0, count = 0;
        std::uint64_t generation = 0;
        spectral_table_t table;
    };

    /** @brief Whether paths run whole, by persistent warps (ray tracing, plt_bdpt), or as a wavefront. */
    [[nodiscard]] bool whole_paths() const noexcept {
        return params.params.bdpt!=0 || params.params.ray_tracing_only!=0;
    }

    /** @brief plt_bdpt: the vertices a subpath holds, at `capacity`; at most as many as it makes. */
    [[nodiscard]] k::u32 vertex_capacity(k::u32 capacity) const noexcept {
        const auto max_vertices = std::uint64_t(params.params.max_depth) + 2;
        return k::u32(std::min<std::uint64_t>(max_vertices, std::uint64_t(initial_vertex_capacity)*capacity/initial_capacity));
    }
    /** @brief plt_bdpt: the aperture edges a path holds, at `capacity` (none without free-space diffraction). */
    [[nodiscard]] k::u32 aperture_capacity(k::u32 capacity) const noexcept {
        return params.params.fsd!=0 ? initial_aperture_capacity*(capacity/initial_capacity) : 0;
    }

    /**
     * @brief Starts an empty pool of paths with `capacity` scratch triangles
     *        and edges each: as many as the main pool has scratch memory for
     *        (plt_bdpt: as many as `whole_path_threads`, or as
     *        `bdpt_pool_bytes` holds).
     */
    void begin(wavefront_t& wf, k::u32 capacity) {
        if (params.params.bdpt!=0) {
            const auto vertices = vertex_capacity(capacity);
            const auto apertures = aperture_capacity(capacity);
            // free-space diffraction collects the edges of a query's triangles
            const std::size_t edges = params.params.fsd!=0 ? 3*std::size_t(capacity) : 0;
            const std::size_t sensor_vertex_size =
                polarimetric ? sizeof(k::bdpt_vertex<true>) : sizeof(k::bdpt_vertex<false>);
            const std::size_t slot_size =
                vertices*(sensor_vertex_size + sizeof(k::bdpt_vertex<false>)) + (capacity+edges)*sizeof(k::u32) +
                apertures*sizeof(k::fraunhofer::fsd_edge);
            const auto slots = k::u32(std::clamp<std::size_t>(bdpt_pool_bytes/slot_size, 1, whole_path_threads));
            sensor_vertices.resize(std::size_t(slots)*vertices*sensor_vertex_size/sizeof(uint4));
            emitter_vertices.resize(std::size_t(slots)*vertices);
            scratch_tris.resize(std::size_t(slots)*capacity);
            scratch_edges.resize(std::size_t(slots)*edges);
            aperture_edges.resize(std::size_t(slots)*apertures);
            overflow.resize(launch_paths);
            wf = { slots, capacity, 0, slots, 0 };
            return;
        }
        if (params.params.ray_tracing_only!=0) {
            // whole paths need only their scratch memory
            const auto slots = k::u32(std::clamp<std::size_t>(std::size_t(pool_size)*initial_capacity/capacity,
                                                              1, whole_path_threads));
            scratch_tris.resize(std::size_t(slots)*capacity);
            scratch_edges.resize(std::size_t(slots)*3*capacity);
            scratch_aperture.resize(std::size_t(slots)*capacity);
            overflow.resize(launch_paths);
            wf = { slots, capacity, 0, slots, 0 };
            return;
        }

        const auto slots = k::u32(std::max<std::size_t>(1, std::size_t(pool_size)*initial_capacity/capacity));
        const std::size_t state_size = polarimetric ? sizeof(k::path_state_t<true>) : sizeof(k::path_state_t<false>);
        paths.resize(std::size_t(slots)*state_size/sizeof(uint4));
        traversals.resize(slots);
        trace_results.resize(slots);
        free_slots.resize(slots);
        queues[0].resize(slots);
        queues[1].resize(slots);
        scratch_tris.resize(std::size_t(slots)*capacity);
        scratch_edges.resize(std::size_t(slots)*3*capacity);
        scratch_aperture.resize(std::size_t(slots)*capacity);
        overflow.resize(slots);

        fill_free_slots_kernel<<<grid_size(slots), block_size>>>(free_slots.get(), slots);
        check(cudaGetLastError(), "launching the path pool kernel");
        wf = { slots, capacity, 0, slots, 0 };
    }

    [[nodiscard]] path_pool_t pool(const wavefront_t& wf, bool count_table_paths) const noexcept {
        return {
            paths.get(), traversals.get(), trace_results.get(), free_slots.get(),
            scratch_tris.get(), scratch_edges.get(), scratch_aperture.get(),
            wf.capacity, wf.capacity,
            count_table_paths ? table_paths.get() : nullptr, table_size,
            sensor_vertices.get(), emitter_vertices.get(), vertex_capacity(wf.capacity),
            aperture_edges.get(), aperture_capacity(wf.capacity),
        };
    }

    [[nodiscard]] cuda_film_t film_view() const noexcept {
        return { film.get(), film_weights.get() };
    }
    /** @brief plt_bdpt's light image, of direct splats. */
    [[nodiscard]] cuda_film_t light_view() const noexcept {
        return { light_film.get(), nullptr };
    }

    /** @brief Room for `count` spectral samples on the device. Moves them: only when no paths are in flight. */
    void reserve_spectra(std::size_t count) {
        spectra.resize(count);
        sample_emissions.resize(count*params.emitter_count);
        sample_materials.resize(count*params.material_count);
        params.spectra = spectra.get();
        params.emissions = sample_emissions.get();
        params.materials = sample_materials.get();
    }

    void upload_spectra(k::u32 base, const spectral_table_t& table) {
        spectra.upload_at(base, table.samples.data(), table.samples.size());
        sample_emissions.upload_at(std::size_t(base)*params.emitter_count, table.emissions.data(), table.emissions.size());
        sample_materials.upload_at(std::size_t(base)*params.material_count, table.materials.data(), table.materials.size());
    }

    /** @brief Steps the main pool until no paths of table `slot` remain in flight (rarely needed). */
    void drain_table(k::u32 slot) {
        for (;;) {
            std::uint32_t counts[tables];
            table_paths.download(counts, tables);
            if (counts[slot]==0)
                return;
            if (main.queued==0)
                throw std::logic_error("a spectral table counts paths in flight, but none are");
            step(main, this->pool(main, true), /* all started */ true);
        }
    }

    /**
     * @brief Starts `source`'s paths (with `seed` and the table at `table`)
     *        and steps the paths in flight, until all started and at most
     *        `keep` remain in flight. Paths that overflow their scratch memory
     *        go to `overflowed`.
     */
    void advance(wavefront_t& wf, path_source_t& source, std::uint32_t seed, k::u32 table, k::u32 keep,
                 bool count_table_paths = true) {
        const path_pool_t pool = this->pool(wf, count_table_paths);
        if (whole_paths()) {
            // by persistent warps, a thread per slot (nothing remains in flight)
            while (source.started<source.count) {
                const auto n = k::u32(std::min<std::uint64_t>(launch_paths, source.count-source.started));
                const auto base = source.list ? source.started : source.first+source.started;
                check(cudaMemset(counters.get(), 0, sizeof(step_counters_t)), "clearing the step counters");
                dispatch([&](auto P) {
                    if (params.params.bdpt!=0)
                        run_bdpt_paths_kernel<P.value><<<grid_size(wf.slots), block_size>>>(
                                params, pool, wf.slots, base, n, source.list, seed, table, film_view(),
                                light_view(), counters.get(), overflow.get(), status_word.get());
                    else
                        run_paths_kernel<P.value><<<grid_size(wf.slots), block_size>>>(
                                params, pool, wf.slots, base, n, source.list, seed, table, film_view(),
                                counters.get(), overflow.get(), status_word.get());
                });
                check(cudaGetLastError(), "launching the path kernel");
                source.started += n;
                collect_overflowed();
            }
            return;
        }

        for (;;) {
            // fill the free slots with new paths
            const auto n = k::u32(std::min<std::uint64_t>(wf.free_top, source.count-source.started));
            if (n>0) {
                dispatch([&](auto P) {
                    start_paths_kernel<P.value><<<grid_size(n), block_size>>>(
                            params, source.list ? source.started : source.first+source.started, n, source.list,
                            seed, table, pool, wf.free_top, queues[wf.q].get(), wf.queued);
                });
                check(cudaGetLastError(), "launching the path start kernel");
                source.started += n;
                wf.queued += n;
                wf.free_top -= n;
            }
            const bool all_started = source.started==source.count;
            if (wf.queued==0 || (all_started && wf.queued<=keep))
                return;
            step(wf, pool, all_started);
        }
    }

    /** @brief Advances the paths in flight a step (or, the last few, to their ends). */
    void step(wavefront_t& wf, const path_pool_t& pool, bool all_started) {
        check(cudaMemset(counters.get(), 0, sizeof(step_counters_t)), "clearing the step counters");
        const auto* queue = queues[wf.q].get();
        auto* next_queue = queues[1-wf.q].get();
        if (all_started && wf.queued<=tail_size) {
            dispatch([&](auto P) {
                step_paths_kernel<P.value><<<grid_size(wf.queued), block_size>>>(
                        params, pool, queue, wf.queued, k::max_path_iterations+1, film_view(),
                        next_queue, wf.free_top, counters.get(), overflow.get(), status_word.get());
            });
            check(cudaGetLastError(), "launching the path step kernel");
        } else {
            dispatch([&](auto P) {
                trace_paths_kernel<P.value><<<grid_size(wf.queued), block_size>>>(params, pool, queue, wf.queued);
            });
            check(cudaGetLastError(), "launching the path tracing kernel");
            dispatch([&](auto P) {
                interact_paths_kernel<P.value><<<grid_size(wf.queued), block_size>>>(
                        params, pool, queue, wf.queued, film_view(),
                        next_queue, wf.free_top, counters.get(), overflow.get(), status_word.get());
            });
            check(cudaGetLastError(), "launching the path interaction kernel");
        }

        const step_counters_t c = collect_overflowed();
        wf.queued = c.queued;
        wf.free_top += c.freed;
        wf.q = 1-wf.q;
    }

    /**
     * @brief The last launch's counters; appends the paths that overflowed to
     *        `overflowed`, with their spectral samples copied to `retained`.
     */
    step_counters_t collect_overflowed() {
        step_counters_t c;
        counters.download(&c, 1);
        if (c.overflowed>0) {
            const auto offset = overflowed.size();
            overflowed.resize(offset+c.overflowed);
            overflow.download(overflowed.data()+offset, c.overflowed);
            for (auto i=offset; i<overflowed.size(); ++i)
                overflowed[i].spectral = retain(overflowed[i].spectral);
        }
        return c;
    }

    /** @brief Copies device spectral sample `spectral` to `retained` (once); returns its index there. */
    k::u32 retain(k::u32 spectral) {
        for (const auto& t : loaded) {
            if (t.count==0 || spectral<t.base || spectral>=t.base+t.count)
                continue;
            const auto i = spectral - t.base;
            const auto [it, inserted] = retained_index.try_emplace(t.generation<<32 | i,
                                                                   k::u32(retained.samples.size()));
            if (inserted) {
                const auto E = params.emitter_count, M = params.material_count;
                retained.samples.push_back(t.table.samples[i]);
                retained.emissions.insert(retained.emissions.end(),
                                          t.table.emissions.begin() + std::size_t(i)*E,
                                          t.table.emissions.begin() + std::size_t(i+1)*E);
                retained.materials.insert(retained.materials.end(),
                                          t.table.materials.begin() + std::size_t(i)*M,
                                          t.table.materials.begin() + std::size_t(i+1)*M);
            }
            return it->second;
        }
        throw std::logic_error("a path's spectral sample is in no loaded table");
    }

    /** @brief Calls `f` with the paths' `P` (a std::bool_constant). */
    template <typename F>
    void dispatch(F&& f) const {
        if (polarimetric) f(std::true_type{});
        else              f(std::false_type{});
    }

    [[nodiscard]] std::uint32_t read_status() const {
        std::uint32_t status;
        status_word.download(&status, 1);
        return status;
    }

    int ordinal;
    k::plt_scene params;
    bool polarimetric;                  // the sensor's: paths are `k::path_state_t<polarimetric>`
    k::u32 whole_path_threads = 0;      // `whole_paths`: the threads of their kernel the GPU runs at once

    device_buffer_t<k::tri_shading>   tris;
    device_buffer_t<k::edge>          edges;
    device_buffer_t<k::emitter>       emitters;
    device_buffer_t<k::real>          emitter_cdf;
    device_buffer_t<k::u32>           area_tris;
    device_buffer_t<k::real>          area_cdf;
    device_buffer_t<k::tri_uv>        tri_uvs;
    device_buffer_t<k::material_shading> material_shadings;
    device_buffer_t<k::texture>       textures;
    device_buffer_t<k::bitmap>        bitmaps;
    device_buffer_t<k::u32>           texels;
    device_buffer_t<k::real>          texel_luts;
    device_buffer_t<k::real>          fsd_lut;
    device_buffer_t<double>           film;
    device_buffer_t<double>           film_weights;     // block splats
    device_buffer_t<double>           light_film;       // plt_bdpt's direct splats
    std::size_t film_size = 0;
    std::size_t weights_size = 0;
    device_buffer_t<std::uint32_t>    status_word;

    // spectral tables: `tables` of `table_size` samples, for the main pool (or one for re-traced paths)
    k::u32 table_size;
    device_buffer_t<k::spectral_sample> spectra;
    device_buffer_t<k::emission>        sample_emissions;
    device_buffer_t<k::material>        sample_materials;
    device_buffer_t<k::u32>             table_paths;      // paths in flight of each table
    std::array<loaded_table_t, tables>  loaded;
    k::u32 next_table = 0;
    std::uint64_t generation = 0;

    // the path pool (`main`, or a pool re-tracing overflowed paths)
    wavefront_t main;
    device_buffer_t<uint4>             paths;            // path states (`path_pool_t::states`)
    device_buffer_t<k::traversal_t>    traversals;
    device_buffer_t<k::u32>            trace_results;
    device_buffer_t<k::u32>            free_slots;
    device_buffer_t<k::u32>            queues[2];
    device_buffer_t<k::u32>            scratch_tris;
    device_buffer_t<k::u32>            scratch_edges;
    device_buffer_t<k::aperture_entry> scratch_aperture;
    device_buffer_t<uint4>             sensor_vertices;  // plt_bdpt (`path_pool_t::sensor_verts`)
    device_buffer_t<k::bdpt_vertex<false>> emitter_vertices;
    device_buffer_t<k::fraunhofer::fsd_edge> aperture_edges;
    device_buffer_t<step_counters_t>   counters;
    device_buffer_t<path_entry_t>      overflow;
    device_buffer_t<path_entry_t>      path_list;
    /** @brief Paths that outgrew their scratch memory, to re-trace with more. */
    std::vector<path_entry_t>          overflowed;
    /** @brief Their spectral samples, indexed by `path_entry_t::spectral`; by (table generation, sample). */
    spectral_table_t                   retained;
    std::unordered_map<std::uint64_t, k::u32> retained_index;
};


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
        tris.upload(bvh.tris.data(), bvh.tris.size());
        tri_count = bvh.tris.size();
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

    [[nodiscard]] std::unique_ptr<backend_renderer_t> create_plt_renderer(const plt_scene_t& scene) override {
        if (scene.tris.size()!=tri_count)
            throw std::runtime_error("the compiled scene does not match the uploaded geometry");
        return std::make_unique<cuda_plt_renderer_t>(ordinal, view(), scene);
    }

private:
    [[nodiscard]] k::bvh_view view() const noexcept {
        return { nodes.get(), tris.get() };
    }

    int ordinal;

    device_buffer_t<k::bvh_node> nodes;
    device_buffer_t<k::bvh_tri>  tris;
    std::size_t tri_count = 0;

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
