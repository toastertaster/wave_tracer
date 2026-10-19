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
#include <stdexcept>
#include <string>

#include <wt/gpu/backend.hpp>
#include <wt/gpu/kernel/plt_path.hpp>
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
 * plt_path
 */

/** @brief Geometry queries for the path tracer: the CUDA BVH traversals. */
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

/** @brief The film, and its reconstruction weights (backward transport), accumulated in double precision. */
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

/** @brief A path to (re-)trace, skipping the splats an earlier attempt made. */
struct path_entry_t {
    std::uint64_t path;
    std::uint32_t skip;
    std::uint32_t seed;
};

/**
 * @brief Paths in flight, each in a slot: its state and its scratch memory.
 *        Unused slots are on a stack, `free_slots`.
 */
struct path_pool_t {
    k::path_state* paths;
    k::traversal_t* traversals;     // of the step in progress
    k::u32* trace_results;          // of the step in progress: status flags, and `trace_found`
    k::u32* free_slots;
    k::u32* tris;
    k::u32* edges;
    k::aperture_entry* aperture;
    k::u32 tri_capacity, edge_capacity;

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

/** @brief Counters of a step: paths that go on, slots freed, paths that overflowed. */
struct step_counters_t {
    std::uint32_t queued, freed, overflowed;
};

__global__ void fill_free_slots_kernel(k::u32* free_slots, k::u32 count) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<count) free_slots[i] = i;
}

/**
 * @brief Starts `count` paths: `base+i` with `seed`, or `list[base+i]` if
 *        `list` is given. They take the slots on top of the free stack
 *        (`free_top` slots) and are queued after the `queued` paths of `queue`.
 */
__global__ void start_paths_kernel(k::plt_scene scene, std::uint64_t base, k::u32 count,
                                   const path_entry_t* list, std::uint32_t seed, path_pool_t pool,
                                   k::u32 free_top, k::u32* queue, k::u32 queued) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const path_entry_t e = list ? list[base+i] : path_entry_t{ base+i, 0, seed };
    const k::u32 slot = pool.free_slots[free_top-1-i];
    const auto rnd = k::rng::for_path(e.seed, k::u32(e.path), k::u32(e.path>>32));
    store_vectors(pool.paths+slot, k::start_path(scene, rnd, e.path, e.skip));
    queue[queued+i] = slot;
}

/**
 * @brief Ends a path, or lists it in `overflow` if it outgrew its scratch
 *        memory (to be re-traced; `end_path` would splat what it has not finished).
 */
__device__ void end_path(const k::plt_scene& scene, cuda_film_t& film, k::path_state& p, k::u32& s,
                         step_counters_t* counters, path_entry_t* overflow) {
    if (s & k::status_scratch_overflow) {
        // the path's generator says which path it is
        const std::uint64_t path = p.rnd.path_lo | std::uint64_t(p.rnd.path_hi)<<32;
        overflow[atomicAdd(&counters->overflowed, 1u)] = { path, p.splats, p.rnd.seed };
        s &= ~k::status_scratch_overflow;
    } else {
        k::end_path(scene, film, p);
    }
}

/**
 * @brief Ends or requeues a path after a step: paths that go on are queued in
 *        `next_queue`; those that end (`end_path`) free their slots, pushed
 *        above the `free_top` free ones.
 */
__device__ void finish_step(const k::plt_scene& scene, cuda_film_t& film, const path_pool_t& pool, k::u32 slot,
                            k::path_state& p, bool goes_on, k::u32 s, k::u32* next_queue, k::u32 free_top,
                            step_counters_t* counters, path_entry_t* overflow, std::uint32_t* status) {
    if (s & k::status_scratch_overflow)
        goes_on = false;
    if (goes_on) {
        store_vectors(pool.paths+slot, p);
        next_queue[atomicAdd(&counters->queued, 1u)] = slot;
    } else {
        end_path(scene, film, p, s, counters, overflow);
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
__global__ void trace_paths_kernel(k::plt_scene scene, path_pool_t pool, const k::u32* queue, k::u32 count) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    const k::path_state p = load_vectors(pool.paths+slot);
    const cuda_bvh_geometry_t geometry = { scene.bvh };

    k::traversal_t tr;
    k::u32 s = k::status_ok;
    const bool found = k::trace_step(scene, geometry, pool.scratch(slot), p, tr, s);
    if (found)
        store_vectors(pool.traversals+slot, tr);
    pool.trace_results[slot] = s | (found ? trace_found : 0u);
}

/** @brief Second part of a step of the `count` queued paths (`interact`), after `trace_paths_kernel`. */
__global__ void __launch_bounds__(block_size, 4) interact_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        const k::u32* queue, k::u32 count,
                                        cuda_film_t film, k::u32* next_queue, k::u32 free_top,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    k::path_state p = load_vectors(pool.paths+slot);
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
__global__ void __launch_bounds__(block_size, 4) step_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        const k::u32* queue, k::u32 count, k::u32 max_steps,
                                        cuda_film_t film, k::u32* next_queue, k::u32 free_top,
                                        step_counters_t* counters, path_entry_t* overflow,
                                        std::uint32_t* status) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const k::u32 slot = queue[i];
    k::path_state p = load_vectors(pool.paths+slot);
    const cuda_bvh_geometry_t geometry = { scene.bvh };

    k::u32 s = k::status_ok;
    bool goes_on = true;
    for (k::u32 n=0; n<max_steps && goes_on; ++n)
        goes_on = k::step_path(scene, geometry, film, pool.scratch(slot), p, s);
    finish_step(scene, film, pool, slot, p, goes_on, s, next_queue, free_top, counters, overflow, status);
}

/**
 * @brief Traces `count` whole paths, `base+i` with `seed` or `list[base+i]`,
 *        one per thread, path `i` with the scratch memory of slot `i`. For
 *        ray tracing, whose steps are cheap: moving path states between
 *        launches would take longer than the steps.
 */
__global__ void __launch_bounds__(block_size, 3) run_paths_kernel(k::plt_scene scene, path_pool_t pool,
                                        std::uint64_t base, k::u32 count, const path_entry_t* list,
                                        std::uint32_t seed, cuda_film_t film, step_counters_t* counters,
                                        path_entry_t* overflow, std::uint32_t* status) {
    const k::u32 i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=count) return;

    const path_entry_t e = list ? list[base+i] : path_entry_t{ base+i, 0, seed };
    k::path_state p = k::start_path(scene, k::rng::for_path(e.seed, k::u32(e.path), k::u32(e.path>>32)),
                                    e.path, e.skip);
    const cuda_bvh_geometry_t geometry = { scene.bvh };

    k::u32 s = k::status_ok;
    while (k::step_path(scene, geometry, film, pool.scratch(i), p, s)) {}

    end_path(scene, film, p, s, counters, overflow);
    if (s!=k::status_ok)
        atomicOr(status, s);
}

/**
 * @brief plt_path on the GPU, as a wavefront: a pool of paths in flight,
 *        advanced a step at a time, so that warps run paths still going (most
 *        paths end after a step or two), with the traversals in a kernel of
 *        their own. Ended paths' slots are refilled with new paths. Paths in
 *        flight carry over from one `trace` to the next, so that the GPU does
 *        not idle while a pass's last, longest, paths finish. Ray-traced
 *        paths, whose steps are cheap, run whole, one per thread.
 */
class cuda_plt_path_renderer_t final : public backend_renderer_t {
public:
    cuda_plt_path_renderer_t(int ordinal, k::bvh_view bvh, const plt_path_scene_t& scene)
        : ordinal(ordinal)
    {
        set_device(ordinal);
        tris.upload(scene.tris.data(), scene.tris.size());
        edges.upload(scene.edges.data(), scene.edges.size());
        materials.upload(scene.materials.data(), scene.materials.size());
        emitters.upload(scene.emitters.data(), scene.emitters.size());
        emitter_cdf.upload(scene.emitter_cdf.data(), scene.emitter_cdf.size());

        film_size = std::size_t(scene.film.width) * scene.film.height * scene.film.components;
        film.resize(film_size);
        check(cudaMemset(film.get(), 0, film_size*sizeof(double)), "clearing the film");
        if (scene.params.backward!=0) {
            film_weights.resize(film_size);
            check(cudaMemset(film_weights.get(), 0, film_size*sizeof(double)), "clearing the film");
        }
        const std::uint32_t zero = 0;
        status_word.upload(&zero, 1);
        counters.resize(1);

        params.bvh = bvh;
        params.tris = tris.get();
        params.edges = edges.get();
        params.materials = materials.get();
        params.emitters = emitters.get();
        params.emitter_cdf = emitter_cdf.get();
        params.emitter_count = k::u32(scene.emitters.size());
        params.sensor = scene.sensor;
        params.film = scene.film;
        params.params = scene.params;
    }

    std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed) override {
        set_device(ordinal);
        if (main.slots==0)
            begin(main, pool_size, initial_capacity);
        path_source_t source = { first, count, nullptr };
        advance(main, source, seed, pool_size/2);
        return read_status();
    }

    std::uint32_t finish() override {
        set_device(ordinal);
        if (main.slots>0) {
            path_source_t none = { 0, 0, nullptr };
            advance(main, none, 0, 0);
        }

        // paths that outgrew their scratch memory are re-traced with more room
        k::u32 capacity = initial_capacity;
        std::vector<path_entry_t> pending;
        while (!overflowed.empty()) {
            if (capacity>=max_capacity)
                throw std::runtime_error("a path found more than " + std::to_string(max_capacity) +
                                         " triangles or edges in one query");
            capacity *= 4;
            std::swap(pending, overflowed);
            overflowed.clear();

            // as many paths in flight as fit the scratch memory of the main pool
            wavefront_t retry;
            begin(retry, k::u32(std::max<std::size_t>(1, std::size_t(pool_size)*initial_capacity/capacity)), capacity);
            path_list.upload(pending.data(), pending.size());
            path_source_t source = { 0, pending.size(), path_list.get() };
            advance(retry, source, 0, 0);
        }
        // the retries reuse the main pool's memory: start it anew
        main = {};

        return read_status();
    }

    void read_film(double* values, double* weights) override {
        set_device(ordinal);
        film.download(values, film_size);
        if (weights) {
            if (params.params.backward==0)
                throw std::logic_error("forward transport accumulates no film weights");
            film_weights.download(weights, film_size);
        }
    }

private:
    /** @brief Paths in flight. */
    static constexpr k::u32 pool_size = 1u<<18;
    /** @brief Triangles and edges a path holds at first; paths needing more are re-traced. */
    static constexpr k::u32 initial_capacity = 64;
    static constexpr k::u32 max_capacity = 1u<<20;
    /** @brief Paths in flight few enough to finish in one launch, rather than a launch per step. */
    static constexpr k::u32 tail_size = 1024;

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

    /** @brief Starts an empty pool of `slots` paths with `capacity` scratch triangles and edges each. */
    void begin(wavefront_t& wf, k::u32 slots, k::u32 capacity) {
        paths.resize(slots);
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

    [[nodiscard]] path_pool_t pool(const wavefront_t& wf) const noexcept {
        return {
            paths.get(), traversals.get(), trace_results.get(), free_slots.get(),
            scratch_tris.get(), scratch_edges.get(), scratch_aperture.get(),
            wf.capacity, wf.capacity,
        };
    }

    [[nodiscard]] cuda_film_t film_view() const noexcept {
        return { film.get(), film_weights.get() };
    }

    /**
     * @brief Starts `source`'s paths (with `seed`) and steps the paths in
     *        flight, until all started and at most `keep` remain in flight.
     *        Paths that overflow their scratch memory go to `overflowed`.
     */
    void advance(wavefront_t& wf, path_source_t& source, std::uint32_t seed, k::u32 keep) {
        const path_pool_t pool = this->pool(wf);
        if (params.params.ray_tracing_only!=0) {
            // whole paths, a pool's worth per launch (nothing remains in flight)
            while (source.started<source.count) {
                const auto n = k::u32(std::min<std::uint64_t>(wf.slots, source.count-source.started));
                check(cudaMemset(counters.get(), 0, sizeof(step_counters_t)), "clearing the step counters");
                run_paths_kernel<<<grid_size(n), block_size>>>(
                        params, pool, source.list ? source.started : source.first+source.started, n, source.list,
                        seed, film_view(), counters.get(), overflow.get(), status_word.get());
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
                start_paths_kernel<<<grid_size(n), block_size>>>(
                        params, source.list ? source.started : source.first+source.started, n, source.list,
                        seed, pool, wf.free_top, queues[wf.q].get(), wf.queued);
                check(cudaGetLastError(), "launching the path start kernel");
                source.started += n;
                wf.queued += n;
                wf.free_top -= n;
            }
            const bool all_started = source.started==source.count;
            if (wf.queued==0 || (all_started && wf.queued<=keep))
                return;

            check(cudaMemset(counters.get(), 0, sizeof(step_counters_t)), "clearing the step counters");
            const auto* queue = queues[wf.q].get();
            auto* next_queue = queues[1-wf.q].get();
            if (all_started && wf.queued<=tail_size) {
                step_paths_kernel<<<grid_size(wf.queued), block_size>>>(
                        params, pool, queue, wf.queued, k::max_path_iterations+1, film_view(),
                        next_queue, wf.free_top, counters.get(), overflow.get(), status_word.get());
                check(cudaGetLastError(), "launching the path step kernel");
            } else {
                trace_paths_kernel<<<grid_size(wf.queued), block_size>>>(params, pool, queue, wf.queued);
                check(cudaGetLastError(), "launching the path tracing kernel");
                interact_paths_kernel<<<grid_size(wf.queued), block_size>>>(
                        params, pool, queue, wf.queued, film_view(),
                        next_queue, wf.free_top, counters.get(), overflow.get(), status_word.get());
                check(cudaGetLastError(), "launching the path interaction kernel");
            }

            const step_counters_t c = collect_overflowed();
            wf.queued = c.queued;
            wf.free_top += c.freed;
            wf.q = 1-wf.q;
        }
    }

    /** @brief The last launch's counters; appends the paths that overflowed to `overflowed`. */
    step_counters_t collect_overflowed() {
        step_counters_t c;
        counters.download(&c, 1);
        if (c.overflowed>0) {
            const auto offset = overflowed.size();
            overflowed.resize(offset+c.overflowed);
            overflow.download(overflowed.data()+offset, c.overflowed);
        }
        return c;
    }

    [[nodiscard]] std::uint32_t read_status() const {
        std::uint32_t status;
        status_word.download(&status, 1);
        return status;
    }

    int ordinal;
    k::plt_scene params;

    device_buffer_t<k::tri_shading>   tris;
    device_buffer_t<k::edge>          edges;
    device_buffer_t<k::material>      materials;
    device_buffer_t<k::point_emitter> emitters;
    device_buffer_t<k::real>          emitter_cdf;
    device_buffer_t<double>           film;
    device_buffer_t<double>           film_weights;     // backward transport
    std::size_t film_size = 0;
    device_buffer_t<std::uint32_t>    status_word;

    // the path pool (`main`, or a pool re-tracing overflowed paths)
    wavefront_t main;
    device_buffer_t<k::path_state>     paths;
    device_buffer_t<k::traversal_t>    traversals;
    device_buffer_t<k::u32>            trace_results;
    device_buffer_t<k::u32>            free_slots;
    device_buffer_t<k::u32>            queues[2];
    device_buffer_t<k::u32>            scratch_tris;
    device_buffer_t<k::u32>            scratch_edges;
    device_buffer_t<k::aperture_entry> scratch_aperture;
    device_buffer_t<step_counters_t>   counters;
    device_buffer_t<path_entry_t>      overflow;
    device_buffer_t<path_entry_t>      path_list;
    /** @brief Paths that outgrew their scratch memory, to re-trace with more. */
    std::vector<path_entry_t>          overflowed;
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

    [[nodiscard]] std::unique_ptr<backend_renderer_t> create_plt_path_renderer(
            const plt_path_scene_t& scene) override {
        if (scene.tris.size()!=tri_count)
            throw std::runtime_error("the compiled scene does not match the uploaded geometry");
        return std::make_unique<cuda_plt_path_renderer_t>(ordinal, view(), scene);
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
