/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * Metal backend: plt_path and plt_bdpt rendering, running the kernels of
 * plt.metal. The CUDA backend's renderer (`cuda_plt_renderer_t` of
 * cuda_backend.cu) on Metal, with the same path pool, spectral tables and
 * re-tracing of overflowed paths; but:
 *   - films accumulate in 32-bit floats on the GPU (Metal has no 64-bit
 *     floating point), which the host adds to its double-precision films
 *     after every `film_samples_per_flush` samples per element
 *     (`flush_film`), and before they are read;
 *   - launches of whole paths are sized to run for about
 *     `target_launch_seconds`: macOS stops command buffers that run too long;
 *   - the GPU runs as many persistent SIMD-groups as its cores hold
 *     (`resident_threads_per_core`): Metal reports no occupancy.
 */

#include "metal_renderer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "plt_shared.hpp"

using namespace wt::gpu;
using namespace wt::gpu::metal_backend;
namespace k = wt::gpu::compute_kernel;

namespace {

/**
 * @brief plt_path on the GPU, as a wavefront: a pool of paths in flight,
 *        advanced a step at a time, so that SIMD-groups run paths still going
 *        (most paths end after a step or two), with the traversals in a
 *        kernel of their own. Ended paths' slots are refilled with new paths.
 *        Paths in flight carry over from one `trace` to the next, so that the
 *        GPU does not idle while a pass's last, longest, paths finish.
 *
 *        Ray-traced paths, whose steps are cheap, and plt_bdpt paths run
 *        whole, in launches of persistent SIMD-groups (see plt.metal's
 *        `take_paths`), with a slot of the pool for each thread the GPU runs
 *        at once (plt_bdpt: or as many as their vertices' memory allows).
 */
class metal_plt_renderer_t final : public backend_renderer_t {
public:
    metal_plt_renderer_t(std::shared_ptr<context_t> context_, bvh_buffers_t bvh, const plt_scene_t& scene)
        : context(std::move(context_)), device(context->device), bvh(bvh),
          polarimetric(scene.film.polarimetric!=0),
          table_size(scene.emitter_count*scene.samples_per_emitter)
    {
        if (whole_paths(scene.params)) {
            const unsigned cores = context->gpu_cores();
            whole_path_threads = k::u32(std::max(cores, 1u)) * resident_threads_per_core;
            launch_size = whole_path_threads;
        }

        params.bvh = { reinterpret_cast<const k::bvh_node*>(bvh.nodes.gpuAddress),
                       reinterpret_cast<const k::bvh_tri*>(bvh.tris.gpuAddress) };
        params.tris = upload(tris, scene.tris);
        params.edges = upload(edges, scene.edges);
        params.emitters = upload(emitters, scene.emitters);
        params.emitter_cdf = upload(emitter_cdf, scene.emitter_cdf);
        params.area_tris = upload(area_tris, scene.area_tris);
        params.area_cdf = upload(area_cdf, scene.area_cdf);
        params.tri_uvs = upload(tri_uvs, scene.tri_uvs);
        params.material_shadings = upload(material_shadings, scene.material_shadings);
        params.textures = upload(textures, scene.textures);
        params.bitmaps = upload(bitmaps, scene.bitmaps);
        params.texels = upload(texels, scene.texels);
        params.texel_luts = upload(texel_luts, scene.texel_luts);
        params.fsd_lut = upload(fsd_lut, scene.fsd_lut);
        params.emitter_count = scene.emitter_count;
        params.material_count = scene.material_count;
        params.samples_per_emitter = scene.samples_per_emitter;
        params.sensor = scene.sensor;
        params.film = scene.film;
        params.params = scene.params;

        // a polarimetric film's components have four values (Stokes parameters), and one weight
        weights_size = std::size_t(scene.film.width) * scene.film.height * scene.film.components;
        film_size = weights_size * (scene.film.polarimetric!=0 ? 4 : 1);
        film.fill_zero(film_size);
        film_sum.assign(film_size, 0);
        if (params.params.block_splats()) {
            film_weights.fill_zero(weights_size);
            weights_sum.assign(weights_size, 0);
        }
        if (params.params.bdpt!=0) {
            light_film.fill_zero(film_size);
            light_sum.assign(film_size, 0);
        }
        flush_paths = std::uint64_t(scene.film.width) * scene.film.height * film_samples_per_flush;

        status_word.fill_zero(1);
        counters.fill_zero(1);
        table_paths.fill_zero(tables);
        reserve_spectra(std::size_t(tables)*table_size);
    }

    std::uint32_t trace(std::uint64_t first, std::uint64_t count, std::uint32_t seed,
                        const spectral_table_t& spectra) override {
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
        if (main.slots>0) {
            path_source_t none = { 0, 0, nullptr };
            advance(main, none, 0, 0, 0);
        }

        // paths that outgrew their scratch memory are re-traced with more room, with the spectral samples
        // they had (kept in `retained`: the tables they came from may have been replaced since)
        k::u32 capacity = initial_capacity;
        std::vector<path_entry> pending;
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
            path_source_t source = { 0, pending.size(), path_list.gpu_pointer() };
            advance(retry, source, 0, 0, 0, /* count table paths */ false);
        }
        // the retries reuse the main pool's memory and spectral tables: start them anew
        main = {};
        loaded.fill({});
        next_table = 0;

        flush_film();
        return read_status();
    }

    void read_film(double* values, double* weights) override {
        flush_film();
        std::copy(film_sum.begin(), film_sum.end(), values);
        if (weights) {
            if (!params.params.block_splats())
                throw std::logic_error("forward transport accumulates no film weights");
            std::copy(weights_sum.begin(), weights_sum.end(), weights);
        }
    }

    void read_light_film(double* values) override {
        if (params.params.bdpt==0)
            throw std::logic_error("only plt_bdpt accumulates a light image");
        flush_film();
        std::copy(light_sum.begin(), light_sum.end(), values);
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
    /**
     * @brief plt_bdpt: memory for paths in flight, which bounds their count:
     *        a quarter of the GPU's recommended working set, up to 2 GiB.
     *        Paths re-traced with more scratch memory take much more each,
     *        so that a pool of CUDA's 512 MiB would hold too few of them to
     *        fill the GPU; unified memory affords more.
     */
    static constexpr std::size_t max_bdpt_pool_bytes = std::size_t(2)<<30;
    /** @brief Whole paths a launch traces at most (its list of overflowed paths holds as many). */
    static constexpr k::u32 launch_paths = 1u<<20;
    /** @brief Paths in flight few enough to finish in one launch, rather than a launch per step. */
    static constexpr k::u32 tail_size = 1024;
    /**
     * @brief Spectral tables on the device: paths of the last few `trace`s
     *        may still be in flight, each with a sample of its pass's table.
     */
    static constexpr k::u32 tables = 4;
    /**
     * @brief Threads of persistent SIMD-groups per GPU core: more than a core
     *        runs at once (SIMD-groups past those take no paths, once they
     *        run), so that the GPU is full whatever the kernels' registers.
     */
    static constexpr k::u32 resident_threads_per_core = 2048;
    /**
     * @brief The GPU time a launch of whole paths aims at: long enough that
     *        its last paths' tail is short in comparison, well short of the
     *        time after which macOS stops a command buffer.
     */
    static constexpr double target_launch_seconds = .25;
    /**
     * @brief Samples per film element (paths, on average) the GPU's float
     *        films accumulate before the host adds them to its double ones,
     *        bounding their sums' rounding.
     */
    static constexpr std::uint64_t film_samples_per_flush = 64;

    /** @brief A pool of paths in flight. */
    struct wavefront_t {
        k::u32 slots = 0;           // 0: not started
        k::u32 capacity = 0;        // scratch triangles and edges per path
        k::u32 queued = 0;          // paths in flight, in `queues[q]`
        k::u32 free_top = 0;        // free slots
        int q = 0;
    };

    /** @brief Paths to start: `first+i`, or `list[i]` (a GPU address) if `list` is given. */
    struct path_source_t {
        std::uint64_t first, count;
        const path_entry* list;
        std::uint64_t started = 0;
    };

    /** @brief A spectral table on the device: samples `base .. base+count-1`, and a copy of it. */
    struct loaded_table_t {
        k::u32 base = 0, count = 0;
        std::uint64_t generation = 0;
        spectral_table_t table;
    };

    /** @brief Whether paths run whole, by persistent SIMD-groups (ray tracing, plt_bdpt), or as a wavefront. */
    [[nodiscard]] static bool whole_paths(const k::plt_params& p) noexcept {
        return p.bdpt!=0 || p.ray_tracing_only!=0;
    }
    [[nodiscard]] bool whole_paths() const noexcept { return whole_paths(params.params); }

    /** @brief Uploads a scene array; its GPU address, or null if it is empty (as the kernels expect). */
    template <typename T>
    const T* upload(buffer_t<T>& buffer, const std::vector<T>& values) {
        if (values.empty())
            return nullptr;
        buffer.upload(values.data(), values.size());
        return buffer.gpu_pointer();
    }

    /** @brief A path kernel of the sensor's paths: `<name>_scalar` or `<name>_polarimetric`. */
    [[nodiscard]] std::string kernel(const char* name) const {
        return std::string(name) + (polarimetric ? "_polarimetric" : "_scalar");
    }

    /** @brief plt_bdpt: the vertices a subpath holds, at `capacity`; at most as many as it makes. */
    [[nodiscard]] k::u32 vertex_capacity(k::u32 capacity) const noexcept {
        const auto max_vertices = std::uint64_t(params.params.max_depth) + 2;
        return k::u32(std::min<std::uint64_t>(max_vertices,
                                              std::uint64_t(initial_vertex_capacity)*capacity/initial_capacity));
    }
    /** @brief plt_bdpt: the aperture edges a path holds, at `capacity` (none without free-space diffraction). */
    [[nodiscard]] k::u32 aperture_capacity(k::u32 capacity) const noexcept {
        return params.params.fsd!=0 ? initial_aperture_capacity*(capacity/initial_capacity) : 0;
    }

    /**
     * @brief Starts an empty pool of paths with `capacity` scratch triangles
     *        and edges each: as many as the main pool has scratch memory for
     *        (plt_bdpt: as many as `whole_path_threads`, or as its pool's
     *        memory holds, see `max_bdpt_pool_bytes`).
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
            const std::size_t pool_bytes = std::min<std::size_t>(max_bdpt_pool_bytes,
                                                                 device.recommendedMaxWorkingSetSize/4);
            const auto slots = k::u32(std::clamp<std::size_t>(pool_bytes/slot_size, 1, whole_path_threads));
            sensor_vertices.resize(std::size_t(slots)*vertices*sensor_vertex_size);
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
        paths.resize(std::size_t(slots)*state_size);
        traversals.resize(slots);
        trace_results.resize(slots);
        free_slots.resize(slots);
        queues[0].resize(slots);
        queues[1].resize(slots);
        scratch_tris.resize(std::size_t(slots)*capacity);
        scratch_edges.resize(std::size_t(slots)*3*capacity);
        scratch_aperture.resize(std::size_t(slots)*capacity);
        overflow.resize(slots);

        context->run("path pool kernel", [&](id<MTLComputeCommandEncoder> e) {
            free_slots.bind(e, 0);
            bind_value(e, slots, 1);
            context->dispatch(e, "fill_free_slots", slots);
        });
        wf = { slots, capacity, 0, slots, 0 };
    }

    /** @brief A launch of paths of pool `wf`, but for its paths and queues. */
    [[nodiscard]] plt_launch launch(const wavefront_t& wf, bool count_table_paths) const noexcept {
        plt_launch l{};
        l.pool = {
            paths.gpu_pointer<void>(), traversals.gpu_pointer(), trace_results.gpu_pointer(),
            free_slots.gpu_pointer(),
            scratch_tris.gpu_pointer(), scratch_edges.gpu_pointer(), scratch_aperture.gpu_pointer(),
            wf.capacity, wf.capacity,
            count_table_paths ? table_paths.gpu_pointer() : nullptr, table_size,
            sensor_vertices.gpu_pointer<void>(), emitter_vertices.gpu_pointer(), vertex_capacity(wf.capacity),
            aperture_edges.gpu_pointer(), aperture_capacity(wf.capacity),
        };
        l.film = { film.gpu_pointer(), film_weights.gpu_pointer() };
        l.light = { light_film.gpu_pointer(), nullptr };
        l.counters = counters.gpu_pointer();
        l.overflow = overflow.gpu_pointer();
        l.status = status_word.gpu_pointer();
        l.slots = wf.slots;
        l.free_top = wf.free_top;
        l.queued = wf.queued;
        return l;
    }

    /**
     * @brief Runs a command buffer of path kernels, which `encode` encodes:
     *        makes the scene's and the renderer's buffers resident, for the
     *        kernels reach them through pointers. Returns its GPU time.
     */
    double run(const char* what, const std::function<void(id<MTLComputeCommandEncoder>)>& encode) {
        return context->run(what, [&](id<MTLComputeCommandEncoder> e) {
            const auto resources = resident();
            [e useResources:resources.data() count:resources.size()
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            encode(e);
        });
    }

    /** @brief Encodes path kernel `name` (`kernel`) of launch `l`, on `threads` threads. */
    void dispatch(id<MTLComputeCommandEncoder> e, const char* name, const plt_launch& l, std::size_t threads,
                  bool whole_groups = false) {
        bind_value(e, params, 0);
        bind_value(e, l, 1);
        if (whole_groups)
            context->dispatch_groups(e, kernel(name), threads);
        else
            context->dispatch(e, kernel(name), threads);
    }

    /** @brief The buffers the kernels read and write. */
    [[nodiscard]] std::vector<id<MTLResource>> resident() const {
        std::vector<id<MTLResource>> r;
        const auto add = [&](id<MTLBuffer> b) { if (b) r.push_back(b); };
        add(bvh.nodes); add(bvh.tris);
        add(tris.get()); add(edges.get()); add(emitters.get()); add(emitter_cdf.get());
        add(area_tris.get()); add(area_cdf.get()); add(tri_uvs.get()); add(material_shadings.get());
        add(textures.get()); add(bitmaps.get()); add(texels.get()); add(texel_luts.get()); add(fsd_lut.get());
        add(film.get()); add(film_weights.get()); add(light_film.get());
        add(status_word.get()); add(counters.get());
        add(spectra.get()); add(sample_emissions.get()); add(sample_materials.get()); add(table_paths.get());
        add(paths.get()); add(traversals.get()); add(trace_results.get()); add(free_slots.get());
        add(queues[0].get()); add(queues[1].get());
        add(scratch_tris.get()); add(scratch_edges.get()); add(scratch_aperture.get());
        add(sensor_vertices.get()); add(emitter_vertices.get()); add(aperture_edges.get());
        add(overflow.get()); add(path_list.get());
        return r;
    }

    /** @brief Room for `count` spectral samples on the device. Moves them: only when no paths are in flight. */
    void reserve_spectra(std::size_t count) {
        spectra.resize(count);
        sample_emissions.resize(count*params.emitter_count);
        sample_materials.resize(count*params.material_count);
        params.spectra = spectra.gpu_pointer();
        params.emissions = sample_emissions.gpu_pointer();
        params.materials = sample_materials.gpu_pointer();
    }

    void upload_spectra(k::u32 base, const spectral_table_t& table) {
        spectra.upload_at(base, table.samples.data(), table.samples.size());
        sample_emissions.upload_at(std::size_t(base)*params.emitter_count,
                                   table.emissions.data(), table.emissions.size());
        sample_materials.upload_at(std::size_t(base)*params.material_count,
                                   table.materials.data(), table.materials.size());
    }

    /** @brief Steps the main pool until no paths of table `slot` remain in flight (rarely needed). */
    void drain_table(k::u32 slot) {
        while (table_paths.data()[slot]!=0) {
            if (main.queued==0)
                throw std::logic_error("a spectral table counts paths in flight, but none are");
            step(main, /* all started */ true, /* count table paths */ true, nullptr);
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
        if (whole_paths()) {
            // by persistent SIMD-groups, a thread per slot (nothing remains in flight)
            while (source.started<source.count) {
                // (at most a flush's worth of paths, `paths_traced`)
                const auto n = k::u32(std::min<std::uint64_t>({ launch_size, source.count-source.started,
                                                                std::max<std::uint64_t>(flush_paths,
                                                                                        whole_path_threads) }));
                plt_launch l = launch(wf, count_table_paths);
                l.base = source.list ? source.started : source.first+source.started;
                l.count = n;
                l.list = source.list;
                l.seed = seed;
                l.table = table;
                *counters.data() = {};
                const double seconds = run("path kernel", [&](id<MTLComputeCommandEncoder> e) {
                    dispatch(e, params.params.bdpt!=0 ? "run_bdpt_paths" : "run_paths", l, wf.slots, true);
                });
                source.started += n;
                collect_overflowed();
                size_launches(n, seconds);
                paths_traced(n);
            }
            return;
        }

        for (;;) {
            // fill the free slots with new paths, and step them with the paths in flight
            const auto n = k::u32(std::min<std::uint64_t>(wf.free_top, source.count-source.started));
            plt_launch start = launch(wf, count_table_paths);
            start.base = source.list ? source.started : source.first+source.started;
            start.count = n;
            start.list = source.list;
            start.seed = seed;
            start.table = table;
            start.next_queue = queues[wf.q].gpu_pointer();
            source.started += n;
            wf.queued += n;
            wf.free_top -= n;
            paths_traced(n);

            const bool all_started = source.started==source.count;
            if (wf.queued==0 || (all_started && wf.queued<=keep)) {
                if (n>0)
                    run("path start kernel", [&](id<MTLComputeCommandEncoder> e) {
                        dispatch(e, "start_paths", start, n);
                    });
                return;
            }
            step(wf, all_started, count_table_paths, n>0 ? &start : nullptr);
        }
    }

    /**
     * @brief Advances the paths in flight a step (or, the last few, to their
     *        ends), after starting the paths of launch `start`, if given.
     */
    void step(wavefront_t& wf, bool all_started, bool count_table_paths, const plt_launch* start) {
        plt_launch l = launch(wf, count_table_paths);
        l.queue = queues[wf.q].gpu_pointer();
        l.next_queue = queues[1-wf.q].gpu_pointer();
        l.count = wf.queued;
        *counters.data() = {};
        const bool tail = all_started && wf.queued<=tail_size;
        run("path step", [&](id<MTLComputeCommandEncoder> e) {
            if (start)
                dispatch(e, "start_paths", *start, start->count);
            if (tail) {
                l.max_steps = k::max_path_iterations+1;
                dispatch(e, "step_paths", l, wf.queued);
            } else {
                dispatch(e, "trace_paths", l, wf.queued);
                dispatch(e, "interact_paths", l, wf.queued);
            }
        });

        const step_counters c = collect_overflowed();
        wf.queued = c.queued;
        wf.free_top += c.freed;
        wf.q = 1-wf.q;
    }

    /**
     * @brief Sizes the next launches of whole paths for `target_launch_seconds`,
     *        at the rate of the last, of `paths` paths in `seconds` (growing at
     *        most fourfold a launch: paths of the next pass may differ).
     */
    void size_launches(k::u32 paths, double seconds) {
        const double rate = double(paths) / std::max(seconds, 1e-6);
        const double size = std::min(rate*target_launch_seconds, 4.*paths);
        launch_size = k::u32(std::clamp(size, double(whole_path_threads), double(launch_paths)));
    }

    /**
     * @brief `paths` paths were started: flushes the film, once due. Launches
     *        of whole paths trace no more paths than a flush's; a wavefront's
     *        steps splat less, but films too small to fill the wavefront
     *        flush after each step.
     */
    void paths_traced(std::uint64_t paths) {
        paths_since_flush += paths;
        if (paths_since_flush>=flush_paths)
            flush_film();
    }

    /**
     * @brief Adds the GPU's float films to the host's double-precision films,
     *        and clears them. The GPU must be idle (it is between launches).
     */
    void flush_film() {
        const auto add = [](buffer_t<float>& gpu, std::vector<double>& sum) {
            if (sum.empty()) return;
            float* values = gpu.data();
            for (std::size_t i=0; i<sum.size(); ++i)
                sum[i] += double(values[i]);
            std::memset(values, 0, sum.size()*sizeof(float));
        };
        add(film, film_sum);
        add(film_weights, weights_sum);
        add(light_film, light_sum);
        paths_since_flush = 0;
    }

    /**
     * @brief The last launch's counters; appends the paths that overflowed to
     *        `overflowed`, with their spectral samples copied to `retained`.
     */
    step_counters collect_overflowed() {
        const step_counters c = *counters.data();
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

    [[nodiscard]] std::uint32_t read_status() const { return status_word.data()[0]; }

    std::shared_ptr<context_t> context;
    id<MTLDevice> device;               // the context's (initialized before the buffers, which take it)
    bvh_buffers_t bvh;
    k::plt_scene params{};
    bool polarimetric;                  // the sensor's: paths are `k::path_state_t<polarimetric>`
    k::u32 whole_path_threads = 0;      // `whole_paths`: threads of persistent SIMD-groups a launch runs
    k::u32 launch_size = launch_paths;  // `whole_paths`: paths a launch traces (see `size_launches`)

    buffer_t<k::tri_shading>   tris{device};
    buffer_t<k::edge>          edges{device};
    buffer_t<k::emitter>       emitters{device};
    buffer_t<k::real>          emitter_cdf{device};
    buffer_t<k::u32>           area_tris{device};
    buffer_t<k::real>          area_cdf{device};
    buffer_t<k::tri_uv>        tri_uvs{device};
    buffer_t<k::material_shading> material_shadings{device};
    buffer_t<k::texture>       textures{device};
    buffer_t<k::bitmap>        bitmaps{device};
    buffer_t<k::u32>           texels{device};
    buffer_t<k::real>          texel_luts{device};
    buffer_t<k::real>          fsd_lut{device};

    // the GPU's float films (see `flush_film`), and the host's double-precision sums of them
    buffer_t<float>            film{device};
    buffer_t<float>            film_weights{device};    // block splats
    buffer_t<float>            light_film{device};      // plt_bdpt's direct splats
    std::vector<double>        film_sum, weights_sum, light_sum;
    std::size_t film_size = 0;
    std::size_t weights_size = 0;
    std::uint64_t flush_paths = 0;      // paths started between flushes
    std::uint64_t paths_since_flush = 0;
    buffer_t<std::uint32_t>    status_word{device};

    // spectral tables: `tables` of `table_size` samples, for the main pool (or one for re-traced paths)
    k::u32 table_size;
    buffer_t<k::spectral_sample> spectra{device};
    buffer_t<k::emission>        sample_emissions{device};
    buffer_t<k::material>        sample_materials{device};
    buffer_t<k::u32>             table_paths{device};    // paths in flight of each table
    std::array<loaded_table_t, tables>  loaded;
    k::u32 next_table = 0;
    std::uint64_t generation = 0;

    // the path pool (`main`, or a pool re-tracing overflowed paths)
    wavefront_t main;
    buffer_t<std::uint8_t>             paths{device};            // path states (`path_pool::states`)
    buffer_t<k::traversal_t>           traversals{device};
    buffer_t<k::u32>                   trace_results{device};
    buffer_t<k::u32>                   free_slots{device};
    buffer_t<k::u32>                   queues[2] = { buffer_t<k::u32>{device}, buffer_t<k::u32>{device} };
    buffer_t<k::u32>                   scratch_tris{device};
    buffer_t<k::u32>                   scratch_edges{device};
    buffer_t<k::aperture_entry>        scratch_aperture{device};
    buffer_t<std::uint8_t>             sensor_vertices{device};  // plt_bdpt (`path_pool::sensor_verts`)
    buffer_t<k::bdpt_vertex<false>>    emitter_vertices{device};
    buffer_t<k::fraunhofer::fsd_edge>  aperture_edges{device};
    buffer_t<step_counters>            counters{device};
    buffer_t<path_entry>               overflow{device};
    buffer_t<path_entry>               path_list{device};
    /** @brief Paths that outgrew their scratch memory, to re-trace with more. */
    std::vector<path_entry>            overflowed;
    /** @brief Their spectral samples, indexed by `path_entry::spectral`; by (table generation, sample). */
    spectral_table_t                   retained;
    std::unordered_map<std::uint64_t, k::u32> retained_index;
};

}

std::unique_ptr<backend_renderer_t> wt::gpu::metal_backend::create_plt_renderer(std::shared_ptr<context_t> context,
                                                                                bvh_buffers_t bvh,
                                                                                const plt_scene_t& scene) {
    return std::make_unique<metal_plt_renderer_t>(std::move(context), bvh, scene);
}
