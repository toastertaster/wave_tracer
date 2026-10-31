/*
*
* wave tracer
* Copyright  Shlomi Steinberg
*
* LICENSE: Creative Commons Attribution-NonCommercial 4.0 International
*
*/

/*
 * Metal backend: the plt_path and plt_bdpt kernels, the Metal counterparts of
 * cuda_backend.cu's (see there, and metal_renderer.mm, for how they are run).
 * Kernels are templates of `P`, whether the sensor is polarimetric
 * (`path_state_t`), instantiated as `<name>_scalar` and `<name>_polarimetric`.
 *
 * Their pipelines are specialized by the function constant `hardware_rays`:
 * whether the paths' ray queries (ballistic segments, shadow rays) use
 * Metal's ray tracing, in the acceleration structure at buffer 2 (see
 * rt_rays.hpp), or the BVH traversal. Cone queries always traverse the BVH.
 * Unlike OptiX's, Metal's ray queries run within the path kernels.
 */

#include "plt_shared.hpp"
#include "rt_rays.hpp"

using namespace metal;
namespace k = wt::gpu::compute_kernel;
using namespace wt::gpu::metal_backend;

constant bool hardware_rays [[function_constant(0)]];

namespace {

/** @brief Geometry queries for the path tracers: the BVH traversals, and rays traced by hardware if enabled. */
struct bvh_geometry {
    k::bvh_view bvh;
    acceleration_structure accel;       // with `hardware_rays`
    bool hardware_shadows;              // shadow rays too (see `geometry_of`)

    k::ray_hit intersect(k::ray r, k::range rng) const {
        if (hardware_rays)
            return rt_intersect(accel, bvh.tris, r, rng);
        return k::intersect_ray(bvh, r, rng);
    }
    bool occluded(k::ray r, k::range rng, thread k::u32& status) const {
        if (hardware_rays && hardware_shadows)
            return rt_occluded(accel, r, rng);
        return k::ray_occluded(bvh, r, rng, status);
    }
    k::cone_hit intersect_cone(const thread k::cone& c, k::range rng, k::real z_scale,
                               thread k::tri_sink& sink) const {
        return k::intersect_cone(bvh, c, rng, z_scale, sink);
    }
};

/**
 * @brief The path kernels' geometry: `accel` is a kernel argument only with
 *        `hardware_rays`.
 *
 *        Shadow rays are traced by hardware but in plt_path's free-space
 *        diffraction of beams (`do_fsd`), which tests the visibility of the
 *        point a beam hit, on its surface: whether a shadow ray ending on a
 *        surface is blocked by it turns on the rounding of the ray-triangle
 *        test, which the BVH traversal shares with the CPU (a closed range),
 *        and the hardware does not (backward plt_path renders were 12%
 *        brighter). Other shadow rays end off surfaces, offset as the CPU's,
 *        and the two agree.
 */
bvh_geometry geometry_of(const thread k::plt_scene& scene, acceleration_structure accel) {
    bvh_geometry g;
    g.bvh = scene.bvh;
    g.hardware_shadows = false;
    if (hardware_rays) {
        g.accel = accel;
        const k::plt_params o = scene.params;
        g.hardware_shadows = o.bdpt!=0 || o.ray_tracing_only!=0 || o.fsd==0;
    }
    return g;
}

/** @brief Path `base+i` with `seed` (and a spectral sample of its table), or `list[base+i]` if there is a list. */
path_entry entry_of(const thread plt_launch& l, u32 i) {
    return l.list ? l.list[l.base+i] : path_entry{ l.base+i, 0, l.seed, k::choose_spectral, 0 };
}

/** @brief A path's random number generator: its index and seed determine it. */
k::rng path_rng(path_entry e) {
    return k::rng::for_path(e.seed, u32(e.path), u32(e.path>>32));
}

template <bool P>
k::path_state_t<P> start_path(const thread k::plt_scene& scene, const thread plt_launch& l, u32 i) {
    const path_entry e = entry_of(l, i);
    return k::start_path<P>(scene, path_rng(e), e.path, e.skip, l.table, e.spectral);
}

/** @brief `count` paths start with spectral table `table` (the first thread of a launch counts them). */
void paths_started(const thread plt_launch& l, u32 thread_index) {
    if (l.pool.table_paths && thread_index==0)
        atomic_add(l.pool.table_paths + l.table/l.pool.table_size, l.count);
}

/**
 * @brief Ends a path, or lists it in `overflow` if it outgrew its scratch
 *        memory (to be re-traced; `end_path` would splat what it has not finished).
 */
template <bool P>
void end_path(const thread k::plt_scene& scene, const thread plt_launch& l, thread k::path_state_t<P>& p,
              thread u32& s) {
    if (s & k::status_scratch_overflow) {
        // the path's generator says which path it is
        const u64 path = p.rnd.path_lo | u64(p.rnd.path_hi)<<32;
        l.overflow[atomic_add(&l.counters->overflowed, 1u)] = {
            path, p.splats, p.rnd.seed, p.spectral, k::scratch_capacity_needed(l.pool.tri_capacity, s) };
        s &= ~(k::status_scratch_overflow | k::status_scratch_growth_mask);
    } else {
        film_view film = l.film;
        k::end_path(scene, film, p);
    }
    l.pool.path_ended(p.spectral);
}

/**
 * @brief Ends or requeues a path after a step: paths that go on are queued in
 *        `next_queue`; those that end (`end_path`) free their slots, pushed
 *        above the `free_top` free ones.
 */
template <bool P>
void finish_step(const thread k::plt_scene& scene, const thread plt_launch& l, u32 slot,
                 thread k::path_state_t<P>& p, bool goes_on, u32 s) {
    if (s & k::status_scratch_overflow)
        goes_on = false;
    if (goes_on) {
        l.pool.states<P>()[slot] = p;
        l.next_queue[atomic_add(&l.counters->queued, 1u)] = slot;
    } else {
        end_path(scene, l, p, s);
        l.pool.free_slots[l.free_top + atomic_add(&l.counters->freed, 1u)] = slot;
    }
    if (s!=k::status_ok)
        atomic_or(l.status, s);
}

/**
 * @brief Persistent SIMD-groups, for kernels of whole paths (cuda_backend.cu's
 *        `take_paths`): a SIMD-group takes the next of `count` paths, as many
 *        as it has lanes with slots (of `slots`; the counter `counters->taken`
 *        starts at 0), once all of them finished their last. A lane traces
 *        path `i` with the memory of the slot of its thread index,
 *        `trace(slot, i)`. Launches have whole SIMD-groups.
 */
template <typename F>
void take_paths(const thread plt_launch& l, u32 slot, u32 lane, thread F& trace) {
    // the SIMD-group's lanes with slots are its first `lanes`
    const u32 lanes = popcount(uint64_t(simd_vote::vote_t(simd_ballot(slot<l.slots))));
    if (lanes==0) return;

    for (;;) {
        u32 first = 0;
        if (lane==0)
            first = atomic_add(&l.counters->taken, lanes);
        first = simd_broadcast_first(first);
        if (first>=l.count)
            return;

        const u32 i = first + lane;
        if (lane<lanes && i<l.count)
            trace(slot, i);
    }
}

}

kernel void fill_free_slots(device u32* free_slots [[buffer(0)]], constant u32& count [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i<count) free_slots[i] = i;
}

/**
 * @brief Starts `count` paths. They take the slots on top of the free stack
 *        (`free_top` slots) and are queued after the `queued` paths of
 *        `next_queue` (the queue of the next step).
 */
template <bool P>
kernel void start_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i>=launch.count) return;
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;

    paths_started(l, i);
    const u32 slot = l.pool.free_slots[l.free_top-1-i];
    l.pool.states<P>()[slot] = start_path<P>(scene, l, i);
    l.next_queue[l.queued+i] = slot;
}

/**
 * @brief First part of a step of the `count` queued paths: traces their beams
 *        (`trace_step`). A kernel of its own, so that the BVH traversals run
 *        with fewer registers, without SIMD-groups diverging into interaction code.
 */
template <bool P>
kernel void trace_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                        acceleration_structure accel [[buffer(2), function_constant(hardware_rays)]],
                        uint i [[thread_position_in_grid]]) {
    if (i>=launch.count) return;
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;

    const u32 slot = l.queue[i];
    const k::path_state_t<P> p = l.pool.states<P>()[slot];
    const bvh_geometry geometry = geometry_of(scene, accel);

    k::traversal_t tr;
    u32 s = k::status_ok;
    const bool found = k::trace_step(scene, geometry, l.pool.scratch(slot), p, tr, s);
    if (found)
        l.pool.traversals[slot] = tr;
    l.pool.trace_results[slot] = s | (found ? trace_found : 0u);
}

/** @brief Second part of a step of the `count` queued paths (`interact`), after `trace_paths`. */
template <bool P>
kernel void interact_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                           acceleration_structure accel [[buffer(2), function_constant(hardware_rays)]],
                           uint i [[thread_position_in_grid]]) {
    if (i>=launch.count) return;
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;

    const u32 slot = l.queue[i];
    k::path_state_t<P> p = l.pool.states<P>()[slot];
    const u32 traced = l.pool.trace_results[slot];
    u32 s = traced & ~trace_found;
    bool goes_on = false;
    if (traced & trace_found) {
        const k::traversal_t tr = l.pool.traversals[slot];
        const bvh_geometry geometry = geometry_of(scene, accel);
        film_view film = l.film;
        goes_on = k::interact(scene, geometry, film, l.pool.scratch(slot), tr, p, s);
    }
    finish_step(scene, l, slot, p, goes_on, s);
}

/**
 * @brief Advances the `count` queued paths by up to `max_steps` whole steps,
 *        in one kernel: for the last few paths, which would otherwise take a
 *        pair of launches per step.
 */
template <bool P>
kernel void step_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                       acceleration_structure accel [[buffer(2), function_constant(hardware_rays)]],
                       uint i [[thread_position_in_grid]]) {
    if (i>=launch.count) return;
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;

    const u32 slot = l.queue[i];
    k::path_state_t<P> p = l.pool.states<P>()[slot];
    const bvh_geometry geometry = geometry_of(scene, accel);
    film_view film = l.film;

    u32 s = k::status_ok;
    bool goes_on = true;
    for (u32 n=0; n<l.max_steps && goes_on; ++n)
        goes_on = k::step_path(scene, geometry, film, l.pool.scratch(slot), p, s);
    finish_step(scene, l, slot, p, goes_on, s);
}

/**
 * @brief Traces `count` whole paths, by persistent SIMD-groups (`take_paths`),
 *        with the scratch memory of `slots` slots. For ray tracing, whose
 *        steps are cheap: moving path states between launches would take
 *        longer than the steps.
 */
template <bool P>
kernel void run_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                      acceleration_structure accel [[buffer(2), function_constant(hardware_rays)]],
                      uint thread_index [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;
    paths_started(l, thread_index);
    const bvh_geometry geometry = geometry_of(scene, accel);
    auto trace = [&](u32 slot, u32 i) {
        k::path_state_t<P> p = start_path<P>(scene, l, i);
        film_view film = l.film;
        u32 s = k::status_ok;
        while (k::step_path(scene, geometry, film, l.pool.scratch(slot), p, s)) {}

        end_path(scene, l, p, s);
        if (s!=k::status_ok)
            atomic_or(l.status, s);
    };
    take_paths(l, thread_index, lane, trace);
}

/**
 * @brief Traces `count` whole plt_bdpt paths (see `k::trace_bdpt_path`), by
 *        persistent SIMD-groups (`take_paths`), with the vertices and scratch
 *        memory of `slots` slots, splatting into the film and its light
 *        image. Paths that outgrow them, which splat nothing, are listed in
 *        `overflow`.
 */
template <bool P>
kernel void run_bdpt_paths(constant k::plt_scene& scene_ [[buffer(0)]], constant plt_launch& launch [[buffer(1)]],
                           acceleration_structure accel [[buffer(2), function_constant(hardware_rays)]],
                           uint thread_index [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const k::plt_scene scene = scene_;
    const plt_launch l = launch;
    paths_started(l, thread_index);
    const bvh_geometry geometry = geometry_of(scene, accel);
    auto trace = [&](u32 slot, u32 i) {
        const path_entry e = entry_of(l, i);
        film_view film = l.film, light = l.light;
        k::bdpt_no_deferral deferred;
        u32 s = k::status_ok;
        const u32 spectral = k::trace_bdpt_path(scene, geometry, film, light, deferred, l.pool.scratch(slot),
                                                l.pool.sensor_verts<P>(slot), l.pool.emitter_verts(slot),
                                                l.pool.vertex_capacity, l.pool.apertures(slot),
                                                l.pool.aperture_capacity, path_rng(e), e.path, l.table,
                                                e.spectral, s);
        if (s & k::status_scratch_overflow) {
            l.overflow[atomic_add(&l.counters->overflowed, 1u)] = {
                e.path, 0, e.seed, spectral, k::scratch_capacity_needed(l.pool.tri_capacity, s) };
            s &= ~(k::status_scratch_overflow | k::status_scratch_growth_mask);
        }
        l.pool.path_ended(spectral);
        if (s!=k::status_ok)
            atomic_or(l.status, s);
    };
    take_paths(l, thread_index, lane, trace);
}

#define WT_PATH_KERNEL(name, P, suffix, ...) \
    template [[host_name(#name "_" #suffix)]] kernel void name<P>(constant k::plt_scene&, constant plt_launch&, \
                                                                  __VA_ARGS__);
#define WT_PATH_KERNELS(P, suffix) \
    WT_PATH_KERNEL(start_paths, P, suffix, uint) \
    WT_PATH_KERNEL(trace_paths, P, suffix, acceleration_structure, uint) \
    WT_PATH_KERNEL(interact_paths, P, suffix, acceleration_structure, uint) \
    WT_PATH_KERNEL(step_paths, P, suffix, acceleration_structure, uint) \
    WT_PATH_KERNEL(run_paths, P, suffix, acceleration_structure, uint, uint) \
    WT_PATH_KERNEL(run_bdpt_paths, P, suffix, acceleration_structure, uint, uint)
WT_PATH_KERNELS(false, scalar)
WT_PATH_KERNELS(true, polarimetric)
