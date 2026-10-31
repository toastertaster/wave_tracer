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
 * GPU backends: tracing paths on the CPU, with the kernels' own code compiled
 * as host C++ (wt/gpu/kernel/), for the paths that overflow their scratch
 * memory again and again: their beams overlap so much geometry that a GPU
 * thread takes seconds to trace one, and their memory leaves room for few at
 * once, so that a GPU's last re-traces wait on a handful of them. A CPU
 * core traces them many times faster. The path traced is the path the GPU
 * would have traced: its random numbers depend only on its index and seed.
 *
 * Plain C++17 and the kernel headers only: included by the backends' host
 * code (cuda_backend.cu, metal_renderer.mm).
 */

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <wt/gpu/kernel/plt_path.hpp>
#include <wt/gpu/kernel/plt_bdpt.hpp>

namespace wt::gpu::cpu_paths {

namespace k = wt::gpu::compute_kernel;

/** @brief Values summed in double precision by several threads at once. */
class film_buffer_t {
public:
    explicit film_buffer_t(std::size_t size = 0) : values(size ? new std::atomic<double>[size] : nullptr), count(size) {
        for (std::size_t i=0; i<size; ++i)
            values[i].store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] std::atomic<double>* data() const noexcept { return values.get(); }
    [[nodiscard]] std::size_t size() const noexcept { return count; }

    /** @brief Adds the values to `out[0 .. size-1]`. Only once no thread adds to them. */
    void add_to(double* out) const {
        for (std::size_t i=0; i<count; ++i)
            out[i] += values[i].load(std::memory_order_relaxed);
    }

private:
    std::unique_ptr<std::atomic<double>[]> values;
    std::size_t count;
};

/** @brief A film's image and reconstruction weights, as the kernels' `Film`. */
struct film_view_t {
    std::atomic<double>* values;
    std::atomic<double>* weights;

    void add(k::u32 i, k::real v) const { add(values[i], double(v)); }
    void add_weight(k::u32 i, k::real w) const { add(weights[i], double(w)); }

private:
    static void add(std::atomic<double>& a, double v) {
        double old = a.load(std::memory_order_relaxed);
        while (!a.compare_exchange_weak(old, old+v, std::memory_order_relaxed)) {}
    }
};

/** @brief The kernels' geometry queries, of the BVH (in host memory). */
struct bvh_geometry_t {
    k::bvh_view bvh;

    [[nodiscard]] k::ray_hit intersect(const k::ray& r, k::range rng) const {
        return k::intersect_ray(bvh, r, rng);
    }
    [[nodiscard]] bool occluded(const k::ray& r, k::range rng, k::u32& status) const {
        return k::ray_occluded(bvh, r, rng, status);
    }
    [[nodiscard]] k::cone_hit intersect_cone(const k::cone& c, k::range rng, k::real z_scale,
                                             k::tri_sink& sink) const {
        return k::intersect_cone(bvh, c, rng, z_scale, sink);
    }
};

/** @brief The memory a path has at a scratch capacity: as on the GPU (see the backends' `begin`). */
struct memory_t {
    k::u32 tris;                // triangles, and aperture entries; 3 times as many edges
    k::u32 vertices;            // plt_bdpt: of each subpath
    k::u32 aperture_edges;      // plt_bdpt: of its free-space diffractions' apertures
};

/**
 * @brief Traces paths on the CPU's threads, as they are queued, until
 *        `wait`: plt_path or plt_bdpt paths of a scene in host memory, of a
 *        polarimetric sensor or not (`P`), into films of its own. `Entry` is
 *        the backend's path entry (`path`, `skip`, `seed`, `spectral`, and
 *        the scratch `capacity` the path needs); `spectral` indexes the
 *        scene's spectral samples. A path that overflows its memory is traced
 *        again with as much more as it asks for, up to `max_capacity`.
 */
template <typename Entry>
class tracer_t {
public:
    /**
     * @param scene  the compiled scene, with host pointers (all its arrays,
     *               and the spectral samples the entries index, must stay
     *               unchanged until `wait`)
     * @param memory a path's memory at a scratch capacity (`memory_t(k::u32)`)
     */
    template <typename Memory>
    tracer_t(const k::plt_scene& scene, Memory memory, k::u32 max_capacity, std::size_t film_size,
             std::size_t weights_size)
        : scene(scene), memory_of(memory), max_capacity(max_capacity),
          film(film_size), weights(scene.params.block_splats() ? weights_size : 0),
          light(scene.params.bdpt!=0 ? film_size : 0)
    {
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned t=0; t<threads; ++t)
            workers.emplace_back([this]() { work(); });
    }
    ~tracer_t() noexcept {
        try { wait(); } catch (...) {}
    }
    tracer_t(const tracer_t&) = delete;
    tracer_t& operator=(const tracer_t&) = delete;

    /** @brief Queues paths to trace. */
    void add(const Entry* entries, std::size_t count) {
        {
            const std::scoped_lock lock(mutex);
            queue.insert(queue.end(), entries, entries+count);
            queued += count;
        }
        wake.notify_all();
    }

    [[nodiscard]] std::size_t paths() const noexcept { return queued; }

    /**
     * @brief Waits for the queued paths. Returns their status flags (but
     *        `status_scratch_overflow`). Throws if a path needed more than
     *        `max_capacity`.
     */
    std::uint32_t wait() {
        {
            const std::scoped_lock lock(mutex);
            closed = true;
        }
        wake.notify_all();
        for (auto& w : workers)
            if (w.joinable()) w.join();
        if (!error.empty())
            throw std::runtime_error(error);
        return status.load();
    }

    /** @brief Adds the films (`film_buffer_t::add_to`); after `wait`. */
    void add_films(double* film_values, double* film_weights, double* light_values) const {
        film.add_to(film_values);
        if (film_weights && weights.size())
            weights.add_to(film_weights);
        if (light_values && light.size())
            light.add_to(light_values);
    }

private:
    /** @brief A thread's memory, grown as its paths need. */
    template <bool P>
    struct scratch_t {
        std::vector<k::u32> tris, edges;
        std::vector<k::aperture_entry> aperture;
        std::vector<k::bdpt_vertex<P>> sensor_vertices;
        std::vector<k::bdpt_vertex<false>> emitter_vertices;
        std::vector<k::fraunhofer::fsd_edge> aperture_edges;

        k::path_scratch view(const memory_t& m) {
            tris.resize(std::max<std::size_t>(tris.size(), m.tris));
            edges.resize(std::max<std::size_t>(edges.size(), 3*std::size_t(m.tris)));
            aperture.resize(std::max<std::size_t>(aperture.size(), m.tris));
            sensor_vertices.resize(std::max<std::size_t>(sensor_vertices.size(), m.vertices));
            emitter_vertices.resize(std::max<std::size_t>(emitter_vertices.size(), m.vertices));
            aperture_edges.resize(std::max<std::size_t>(aperture_edges.size(), m.aperture_edges));
            return { tris.data(), edges.data(), aperture.data(), m.tris, m.tris };
        }
    };

    void work() {
        if (scene.film.polarimetric!=0) trace_queued<true>();
        else                            trace_queued<false>();
    }

    /** @brief A thread's loop: traces queued paths until the queue is closed and empty. */
    template <bool P>
    void trace_queued() {
        scratch_t<P> scratch;
        for (;;) {
            Entry e;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&]() { return closed || !queue.empty(); });
                if (queue.empty())
                    return;
                e = queue.back();
                queue.pop_back();
            }
            try {
                trace<P>(e, scratch);
            } catch (const std::exception& ex) {
                const std::scoped_lock lock(mutex);
                error = ex.what();
            }
        }
    }

    /** @brief Traces a path until it does not overflow its memory. */
    template <bool P>
    void trace(Entry e, scratch_t<P>& scratch) {
        const bvh_geometry_t geometry = { scene.bvh };
        film_view_t film_view = { film.data(), weights.data() };
        film_view_t light_view = { light.data(), nullptr };
        k::u32 capacity = e.capacity;
        for (;;) {
            if (capacity>max_capacity)
                throw std::runtime_error("a path found more than " + std::to_string(max_capacity) +
                                         " triangles or edges in one query");
            const memory_t m = memory_of(capacity);
            const k::path_scratch view = scratch.view(m);
            const k::rng rnd = k::rng::for_path(e.seed, k::u32(e.path), k::u32(e.path>>32));
            k::u32 s = k::status_ok;
            if (scene.params.bdpt!=0) {
                k::bdpt_no_deferral deferred;
                k::trace_bdpt_path(scene, geometry, film_view, light_view, deferred, view,
                                   scratch.sensor_vertices.data(), scratch.emitter_vertices.data(), m.vertices,
                                   scratch.aperture_edges.data(), m.aperture_edges, rnd, e.path, 0, e.spectral, s);
            } else {
                k::path_state_t<P> p = k::start_path<P>(scene, rnd, e.path, e.skip, 0, e.spectral);
                while (k::step_path(scene, geometry, film_view, view, p, s)) {}
                if (s & k::status_scratch_overflow)
                    e.skip = p.splats;      // made, by this attempt: the next skips them
                else
                    k::end_path(scene, film_view, p);
            }
            if (s & k::status_scratch_overflow) {
                capacity = k::scratch_capacity_needed(capacity, s);
                continue;
            }
            status.fetch_or(s);
            return;
        }
    }

    const k::plt_scene scene;
    const std::function<memory_t(k::u32)> memory_of;
    const k::u32 max_capacity;
    film_buffer_t film, weights, light;

    std::mutex mutex;
    std::condition_variable wake;
    std::vector<Entry> queue;
    std::size_t queued = 0;
    bool closed = false;
    std::string error;
    std::atomic<std::uint32_t> status = 0;
    std::vector<std::thread> workers;
};

}
